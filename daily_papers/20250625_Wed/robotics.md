# 机器人 cs.RO

- **最新发布 38 篇**

- **更新 25 篇**

## 最新发布

#### [new 001] Ground-Effect-Aware Modeling and Control for Multicopters
- **分类: cs.RO**

- **简介: 该论文属于多旋翼飞行器控制任务，解决地面效应带来的控制误差问题。通过建模与实验分析，提出结合动态逆和扰动模型的控制方法，有效降低控制误差。**

- **链接: [http://arxiv.org/pdf/2506.19424v1](http://arxiv.org/pdf/2506.19424v1)**

> **作者:** Tiankai Yang; Kaixin Chai; Jialin Ji; Yuze Wu; Chao Xu; Fei Gao
>
> **摘要:** The ground effect on multicopters introduces several challenges, such as control errors caused by additional lift, oscillations that may occur during near-ground flight due to external torques, and the influence of ground airflow on models such as the rotor drag and the mixing matrix. This article collects and analyzes the dynamics data of near-ground multicopter flight through various methods, including force measurement platforms and real-world flights. For the first time, we summarize the mathematical model of the external torque of multicopters under ground effect. The influence of ground airflow on rotor drag and the mixing matrix is also verified through adequate experimentation and analysis. Through simplification and derivation, the differential flatness of the multicopter's dynamic model under ground effect is confirmed. To mitigate the influence of these disturbance models on control, we propose a control method that combines dynamic inverse and disturbance models, ensuring consistent control effectiveness at both high and low altitudes. In this method, the additional thrust and variations in rotor drag under ground effect are both considered and compensated through feedforward models. The leveling torque of ground effect can be equivalently represented as variations in the center of gravity and the moment of inertia. In this way, the leveling torque does not explicitly appear in the dynamic model. The final experimental results show that the method proposed in this paper reduces the control error (RMSE) by \textbf{45.3\%}. Please check the supplementary material at: https://github.com/ZJU-FAST-Lab/Ground-effect-controller.
>
---
#### [new 002] The MOTIF Hand: A Robotic Hand for Multimodal Observations with Thermal, Inertial, and Force Sensors
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决多模态传感不足的问题。工作包括设计集成多种传感器的MOTIF机械手，提升操作安全性与物体识别能力。**

- **链接: [http://arxiv.org/pdf/2506.19201v1](http://arxiv.org/pdf/2506.19201v1)**

> **作者:** Hanyang Zhou; Haozhe Lou; Wenhao Liu; Enyu Zhao; Yue Wang; Daniel Seita
>
> **摘要:** Advancing dexterous manipulation with multi-fingered robotic hands requires rich sensory capabilities, while existing designs lack onboard thermal and torque sensing. In this work, we propose the MOTIF hand, a novel multimodal and versatile robotic hand that extends the LEAP hand by integrating: (i) dense tactile information across the fingers, (ii) a depth sensor, (iii) a thermal camera, (iv), IMU sensors, and (v) a visual sensor. The MOTIF hand is designed to be relatively low-cost (under 4000 USD) and easily reproducible. We validate our hand design through experiments that leverage its multimodal sensing for two representative tasks. First, we integrate thermal sensing into 3D reconstruction to guide temperature-aware, safe grasping. Second, we show how our hand can distinguish objects with identical appearance but different masses - a capability beyond methods that use vision only.
>
---
#### [new 003] Zero-Shot Parameter Learning of Robot Dynamics Using Bayesian Statistics and Prior Knowledge
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人动力学参数识别任务，旨在解决传统方法依赖大量测量和缺乏先验知识的问题，通过贝叶斯统计实现零样本学习，提升泛化能力和物理可行性。**

- **链接: [http://arxiv.org/pdf/2506.19350v1](http://arxiv.org/pdf/2506.19350v1)**

> **作者:** Carsten Reiners; Minh Trinh; Lukas Gründel; Sven Tauchmann; David Bitterolf; Oliver Petrovic; Christian Brecher
>
> **备注:** Carsten Reiners and Minh Trinh contributed equally to this work
>
> **摘要:** Inertial parameter identification of industrial robots is an established process, but standard methods using Least Squares or Machine Learning do not consider prior information about the robot and require extensive measurements. Inspired by Bayesian statistics, this paper presents an identification method with improved generalization that incorporates prior knowledge and is able to learn with only a few or without additional measurements (Zero-Shot Learning). Furthermore, our method is able to correctly learn not only the inertial but also the mechanical and base parameters of the MABI Max 100 robot while ensuring physical feasibility and specifying the confidence intervals of the results. We also provide different types of priors for serial robots with 6 degrees of freedom, where datasheets or CAD models are not available.
>
---
#### [new 004] A Survey on Soft Robot Adaptability: Implementations, Applications, and Prospects
- **分类: cs.RO**

- **简介: 本文综述软体机器人适应性，探讨其设计、感知与控制方法，分析在医疗、穿戴等领域的应用及挑战，旨在提升软体机器人的环境与内部适应能力。**

- **链接: [http://arxiv.org/pdf/2506.19397v1](http://arxiv.org/pdf/2506.19397v1)**

> **作者:** Zixi Chen; Di Wu; Qinghua Guan; David Hardman; Federico Renda; Josie Hughes; Thomas George Thuruthel; Cosimo Della Santina; Barbara Mazzolai; Huichan Zhao; Cesare Stefanini
>
> **备注:** 12 pages, 4 figures, accepted by IEEE Robotics & Automation Magazine
>
> **摘要:** Soft robots, compared to rigid robots, possess inherent advantages, including higher degrees of freedom, compliance, and enhanced safety, which have contributed to their increasing application across various fields. Among these benefits, adaptability is particularly noteworthy. In this paper, adaptability in soft robots is categorized into external and internal adaptability. External adaptability refers to the robot's ability to adjust, either passively or actively, to variations in environments, object properties, geometries, and task dynamics. Internal adaptability refers to the robot's ability to cope with internal variations, such as manufacturing tolerances or material aging, and to generalize control strategies across different robots. As the field of soft robotics continues to evolve, the significance of adaptability has become increasingly pronounced. In this review, we summarize various approaches to enhancing the adaptability of soft robots, including design, sensing, and control strategies. Additionally, we assess the impact of adaptability on applications such as surgery, wearable devices, locomotion, and manipulation. We also discuss the limitations of soft robotics adaptability and prospective directions for future research. By analyzing adaptability through the lenses of implementation, application, and challenges, this paper aims to provide a comprehensive understanding of this essential characteristic in soft robotics and its implications for diverse applications.
>
---
#### [new 005] CronusVLA: Transferring Latent Motion Across Time for Multi-Frame Prediction in Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机械操作中的多帧预测任务，解决单帧观察限制问题，通过引入多帧运动信息提升模型性能。**

- **链接: [http://arxiv.org/pdf/2506.19816v1](http://arxiv.org/pdf/2506.19816v1)**

> **作者:** Hao Li; Shuai Yang; Yilun Chen; Yang Tian; Xiaoda Yang; Xinyi Chen; Hanqing Wang; Tai Wang; Feng Zhao; Dahua Lin; Jiangmiao Pang
>
> **备注:** 36 pages, 21 figures
>
> **摘要:** Recent vision-language-action (VLA) models built on pretrained vision-language models (VLMs) have demonstrated strong generalization across manipulation tasks. However, they remain constrained by a single-frame observation paradigm and cannot fully benefit from the motion information offered by aggregated multi-frame historical observations, as the large vision-language backbone introduces substantial computational cost and inference latency. We propose CronusVLA, a unified framework that extends single-frame VLA models to the multi-frame paradigm through an efficient post-training stage. CronusVLA comprises three key components: (1) single-frame pretraining on large-scale embodied datasets with autoregressive action tokens prediction, which establishes an embodied vision-language foundation; (2) multi-frame encoding, adapting the prediction of vision-language backbones from discrete action tokens to motion features during post-training, and aggregating motion features from historical frames into a feature chunking; (3) cross-frame decoding, which maps the feature chunking to accurate actions via a shared decoder with cross-attention. By reducing redundant token computation and caching past motion features, CronusVLA achieves efficient inference. As an application of motion features, we further propose an action adaptation mechanism based on feature-action retrieval to improve model performance during finetuning. CronusVLA achieves state-of-the-art performance on SimplerEnv with 70.9% success rate, and 12.7% improvement over OpenVLA on LIBERO. Real-world Franka experiments also show the strong performance and robustness.
>
---
#### [new 006] Faster Motion Planning via Restarts
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决随机算法运行时间不稳定的问题，通过重启技术提升速度与效率。**

- **链接: [http://arxiv.org/pdf/2506.19016v1](http://arxiv.org/pdf/2506.19016v1)**

> **作者:** Nancy Amato; Stav Ashur; Sariel Har-Peled%
>
> **备注:** arXiv admin note: text overlap with arXiv:2503.04633
>
> **摘要:** Randomized methods such as PRM and RRT are widely used in motion planning. However, in some cases, their running-time suffers from inherent instability, leading to ``catastrophic'' performance even for relatively simple instances. We apply stochastic restart techniques, some of them new, for speeding up Las Vegas algorithms, that provide dramatic speedups in practice (a factor of $3$ [or larger] in many cases). Our experiments demonstrate that the new algorithms have faster runtimes, shorter paths, and greater gains from multi-threading (when compared with straightforward parallel implementation). We prove the optimality of the new variants. Our implementation is open source, available on github, and is easy to deploy and use.
>
---
#### [new 007] UniTac-NV: A Unified Tactile Representation For Non-Vision-Based Tactile Sensors
- **分类: cs.RO**

- **简介: 该论文属于 tactile sensing 任务，旨在解决非视觉触觉传感器数据统一问题。提出一种编码器-解码器架构，实现跨传感器数据对齐与迁移。**

- **链接: [http://arxiv.org/pdf/2506.19699v1](http://arxiv.org/pdf/2506.19699v1)**

> **作者:** Jian Hou; Xin Zhou; Qihan Yang; Adam J. Spiers
>
> **备注:** 7 pages, 8 figures. Accepted version to appear in: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Generalizable algorithms for tactile sensing remain underexplored, primarily due to the diversity of sensor modalities. Recently, many methods for cross-sensor transfer between optical (vision-based) tactile sensors have been investigated, yet little work focus on non-optical tactile sensors. To address this gap, we propose an encoder-decoder architecture to unify tactile data across non-vision-based sensors. By leveraging sensor-specific encoders, the framework creates a latent space that is sensor-agnostic, enabling cross-sensor data transfer with low errors and direct use in downstream applications. We leverage this network to unify tactile data from two commercial tactile sensors: the Xela uSkin uSPa 46 and the Contactile PapillArray. Both were mounted on a UR5e robotic arm, performing force-controlled pressing sequences against distinct object shapes (circular, square, and hexagonal prisms) and two materials (rigid PLA and flexible TPU). Another more complex unseen object was also included to investigate the model's generalization capabilities. We show that alignment in latent space can be implicitly learned from joint autoencoder training with matching contacts collected via different sensors. We further demonstrate the practical utility of our approach through contact geometry estimation, where downstream models trained on one sensor's latent representation can be directly applied to another without retraining.
>
---
#### [new 008] The Starlink Robot: A Platform and Dataset for Mobile Satellite Communication
- **分类: cs.RO**

- **简介: 该论文提出Starlink机器人平台及数据集，用于研究移动中的卫星通信性能，解决运动与环境遮挡对连接的影响问题。**

- **链接: [http://arxiv.org/pdf/2506.19781v1](http://arxiv.org/pdf/2506.19781v1)**

> **作者:** Boyi Liu; Qianyi Zhang; Qiang Yang; Jianhao Jiao; Jagmohan Chauhan; Dimitrios Kanoulas
>
> **摘要:** The integration of satellite communication into mobile devices represents a paradigm shift in connectivity, yet the performance characteristics under motion and environmental occlusion remain poorly understood. We present the Starlink Robot, the first mobile robotic platform equipped with Starlink satellite internet, comprehensive sensor suite including upward-facing camera, LiDAR, and IMU, designed to systematically study satellite communication performance during movement. Our multi-modal dataset captures synchronized communication metrics, motion dynamics, sky visibility, and 3D environmental context across diverse scenarios including steady-state motion, variable speeds, and different occlusion conditions. This platform and dataset enable researchers to develop motion-aware communication protocols, predict connectivity disruptions, and optimize satellite communication for emerging mobile applications from smartphones to autonomous vehicles. The project is available at https://github.com/StarlinkRobot.
>
---
#### [new 009] Look to Locate: Vision-Based Multisensory Navigation with 3-D Digital Maps for GNSS-Challenged Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于导航任务，解决GNSS受限环境下的车辆定位问题。通过融合视觉与3D地图，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.19827v1](http://arxiv.org/pdf/2506.19827v1)**

> **作者:** Ola Elmaghraby; Eslam Mounier; Paulo Ricardo Marques de Araujo; Aboelmagd Noureldin
>
> **摘要:** In Global Navigation Satellite System (GNSS)-denied environments such as indoor parking structures or dense urban canyons, achieving accurate and robust vehicle positioning remains a significant challenge. This paper proposes a cost-effective, vision-based multi-sensor navigation system that integrates monocular depth estimation, semantic filtering, and visual map registration (VMR) with 3-D digital maps. Extensive testing in real-world indoor and outdoor driving scenarios demonstrates the effectiveness of the proposed system, achieving sub-meter accuracy of 92% indoors and more than 80% outdoors, with consistent horizontal positioning and heading average root mean-square errors of approximately 0.98 m and 1.25 {\deg}, respectively. Compared to the baselines examined, the proposed solution significantly reduced drift and improved robustness under various conditions, achieving positioning accuracy improvements of approximately 88% on average. This work highlights the potential of cost-effective monocular vision systems combined with 3D maps for scalable, GNSS-independent navigation in land vehicles.
>
---
#### [new 010] FORTE: Tactile Force and Slip Sensing on Compliant Fingers for Delicate Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决柔性抓取中的力与滑动感知问题。提出FORTE系统，实现低延迟、高精度的触觉反馈，提升对易损物体的抓取能力。**

- **链接: [http://arxiv.org/pdf/2506.18960v1](http://arxiv.org/pdf/2506.18960v1)**

> **作者:** Siqi Shang; Mingyo Seo; Yuke Zhu; Lilly Chin
>
> **摘要:** Handling delicate and fragile objects remains a major challenge for robotic manipulation, especially for rigid parallel grippers. While the simplicity and versatility of parallel grippers have led to widespread adoption, these grippers are limited by their heavy reliance on visual feedback. Tactile sensing and soft robotics can add responsiveness and compliance. However, existing methods typically involve high integration complexity or suffer from slow response times. In this work, we introduce FORTE, a tactile sensing system embedded in compliant gripper fingers. FORTE uses 3D-printed fin-ray grippers with internal air channels to provide low-latency force and slip feedback. FORTE applies just enough force to grasp objects without damaging them, while remaining easy to fabricate and integrate. We find that FORTE can accurately estimate grasping forces from 0-8 N with an average error of 0.2 N, and detect slip events within 100 ms of occurring. We demonstrate FORTE's ability to grasp a wide range of slippery, fragile, and deformable objects. In particular, FORTE grasps fragile objects like raspberries and potato chips with a 98.6% success rate, and achieves 93% accuracy in detecting slip events. These results highlight FORTE's potential as a robust and practical solution for enabling delicate robotic manipulation. Project page: https://merge-lab.github.io/FORTE
>
---
#### [new 011] Estimating Spatially-Dependent GPS Errors Using a Swarm of Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于定位误差估计任务，旨在解决GPS在不同位置的误差问题。通过机器人团队协作，利用SBE算法和GPR模型进行误差建模与路径规划。**

- **链接: [http://arxiv.org/pdf/2506.19712v1](http://arxiv.org/pdf/2506.19712v1)**

> **作者:** Praneeth Somisetty; Robert Griffin; Victor M. Baez; Miguel F. Arevalo-Castiblanco; Aaron T. Becker; Jason M. O'Kane
>
> **备注:** 6 pages, 7 figures, 2025 IEEE 21st International Conference on Automation Science and Engineering
>
> **摘要:** External factors, including urban canyons and adversarial interference, can lead to Global Positioning System (GPS) inaccuracies that vary as a function of the position in the environment. This study addresses the challenge of estimating a static, spatially-varying error function using a team of robots. We introduce a State Bias Estimation Algorithm (SBE) whose purpose is to estimate the GPS biases. The central idea is to use sensed estimates of the range and bearing to the other robots in the team to estimate changes in bias across the environment. A set of drones moves in a 2D environment, each sampling data from GPS, range, and bearing sensors. The biases calculated by the SBE at estimated positions are used to train a Gaussian Process Regression (GPR) model. We use a Sparse Gaussian process-based Informative Path Planning (IPP) algorithm that identifies high-value regions of the environment for data collection. The swarm plans paths that maximize information gain in each iteration, further refining their understanding of the environment's positional bias landscape. We evaluated SBE and IPP in simulation and compared the IPP methodology to an open-loop strategy.
>
---
#### [new 012] ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG
- **分类: cs.RO**

- **简介: 该论文属于意图检测任务，旨在解决sEMG信号在不同用户间快速、可靠识别的问题。提出一种零样本、低延迟的框架，实现手势实时识别与跟踪。**

- **链接: [http://arxiv.org/pdf/2506.19815v1](http://arxiv.org/pdf/2506.19815v1)**

> **作者:** Runsheng Wang; Xinyue Zhu; Ava Chen; Jingxi Xu; Lauren Winterbottom; Dawn M. Nilsen; Joel Stein; Matei Ciocarlie
>
> **摘要:** Surface electromyography (sEMG) signals show promise for effective human-computer interfaces, particularly in rehabilitation and prosthetics. However, challenges remain in developing systems that respond quickly and reliably to user intent, across different subjects and without requiring time-consuming calibration. In this work, we propose a framework for EMG-based intent detection that addresses these challenges. Unlike traditional gesture recognition models that wait until a gesture is completed before classifying it, our approach uses a segmentation strategy to assign intent labels at every timestep as the gesture unfolds. We introduce a novel masked modeling strategy that aligns muscle activations with their corresponding user intents, enabling rapid onset detection and stable tracking of ongoing gestures. In evaluations against baseline methods, considering both accuracy and stability for device control, our approach surpasses state-of-the-art performance in zero-shot transfer conditions, demonstrating its potential for wearable robotics and next-generation prosthetic systems. Our project page is available at: https://reactemg.github.io
>
---
#### [new 013] Robotics Under Construction: Challenges on Job Sites
- **分类: cs.RO; cs.AI; cs.AR; cs.ET; cs.SY; eess.SY**

- **简介: 该论文属于建筑自动化任务，旨在解决施工场地中物料运输问题。研究开发了自主导航系统，并探索环境感知与传感器优化技术。**

- **链接: [http://arxiv.org/pdf/2506.19597v1](http://arxiv.org/pdf/2506.19597v1)**

> **作者:** Haruki Uchiito; Akhilesh Bhat; Koji Kusaka; Xiaoya Zhang; Hiraku Kinjo; Honoka Uehara; Motoki Koyama; Shinji Natsume
>
> **备注:** Workshop on Field Robotics, ICRA
>
> **摘要:** As labor shortages and productivity stagnation increasingly challenge the construction industry, automation has become essential for sustainable infrastructure development. This paper presents an autonomous payload transportation system as an initial step toward fully unmanned construction sites. Our system, based on the CD110R-3 crawler carrier, integrates autonomous navigation, fleet management, and GNSS-based localization to facilitate material transport in construction site environments. While the current system does not yet incorporate dynamic environment adaptation algorithms, we have begun fundamental investigations into external-sensor based perception and mapping system. Preliminary results highlight the potential challenges, including navigation in evolving terrain, environmental perception under construction-specific conditions, and sensor placement optimization for improving autonomy and efficiency. Looking forward, we envision a construction ecosystem where collaborative autonomous agents dynamically adapt to site conditions, optimizing workflow and reducing human intervention. This paper provides foundational insights into the future of robotics-driven construction automation and identifies critical areas for further technological development.
>
---
#### [new 014] Probabilistic modelling and safety assurance of an agriculture robot providing light-treatment
- **分类: cs.RO; cs.FL; cs.SE; cs.SY; eess.SY**

- **简介: 该论文属于农业机器人安全保证任务，解决机器人在农田中避障与防撞问题，通过概率建模与风险分析提升系统安全性。**

- **链接: [http://arxiv.org/pdf/2506.19620v1](http://arxiv.org/pdf/2506.19620v1)**

> **作者:** Mustafa Adam; Kangfeng Ye; David A. Anisi; Ana Cavalcanti; Jim Woodcock; Robert Morris
>
> **摘要:** Continued adoption of agricultural robots postulates the farmer's trust in the reliability, robustness and safety of the new technology. This motivates our work on safety assurance of agricultural robots, particularly their ability to detect, track and avoid obstacles and humans. This paper considers a probabilistic modelling and risk analysis framework for use in the early development phases. Starting off with hazard identification and a risk assessment matrix, the behaviour of the mobile robot platform, sensor and perception system, and any humans present are captured using three state machines. An auto-generated probabilistic model is then solved and analysed using the probabilistic model checker PRISM. The result provides unique insight into fundamental development and engineering aspects by quantifying the effect of the risk mitigation actions and risk reduction associated with distinct design concepts. These include implications of adopting a higher performance and more expensive Object Detection System or opting for a more elaborate warning system to increase human awareness. Although this paper mainly focuses on the initial concept-development phase, the proposed safety assurance framework can also be used during implementation, and subsequent deployment and operation phases.
>
---
#### [new 015] Multimodal Anomaly Detection with a Mixture-of-Experts
- **分类: cs.RO**

- **简介: 该论文属于机器人异常检测任务，旨在解决机器人与环境交互中的多模态异常识别问题。通过融合视觉语言模型和高斯混合回归，提升检测准确性与效率。**

- **链接: [http://arxiv.org/pdf/2506.19077v1](http://arxiv.org/pdf/2506.19077v1)**

> **作者:** Christoph Willibald; Daniel Sliwowski; Dongheui Lee
>
> **备注:** 8 pages, 5 figures, 1 table, the paper has been accepted for publication in the Proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** With a growing number of robots being deployed across diverse applications, robust multimodal anomaly detection becomes increasingly important. In robotic manipulation, failures typically arise from (1) robot-driven anomalies due to an insufficient task model or hardware limitations, and (2) environment-driven anomalies caused by dynamic environmental changes or external interferences. Conventional anomaly detection methods focus either on the first by low-level statistical modeling of proprioceptive signals or the second by deep learning-based visual environment observation, each with different computational and training data requirements. To effectively capture anomalies from both sources, we propose a mixture-of-experts framework that integrates the complementary detection mechanisms with a visual-language model for environment monitoring and a Gaussian-mixture regression-based detector for tracking deviations in interaction forces and robot motions. We introduce a confidence-based fusion mechanism that dynamically selects the most reliable detector for each situation. We evaluate our approach on both household and industrial tasks using two robotic systems, demonstrating a 60% reduction in detection delay while improving frame-wise anomaly detection performance compared to individual detectors.
>
---
#### [new 016] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with Vision-Language Models
- **分类: cs.RO; I.2.9; I.2.10; I.4.8; H.5.2**

- **简介: 该论文属于机器人操作任务，解决VLM在空间表示提取上的适应性不足问题。提出T-Rex框架，动态选择适合任务的空间表示方案，提升效率与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.19498v1](http://arxiv.org/pdf/2506.19498v1)**

> **作者:** Yiteng Chen; Wenbo Li; Shiyi Wang; Huiping Zhuang; Qingyao Wu
>
> **备注:** submitted to NeurIPS 2025
>
> **摘要:** Building a general robotic manipulation system capable of performing a wide variety of tasks in real-world settings is a challenging task. Vision-Language Models (VLMs) have demonstrated remarkable potential in robotic manipulation tasks, primarily due to the extensive world knowledge they gain from large-scale datasets. In this process, Spatial Representations (such as points representing object positions or vectors representing object orientations) act as a bridge between VLMs and real-world scene, effectively grounding the reasoning abilities of VLMs and applying them to specific task scenarios. However, existing VLM-based robotic approaches often adopt a fixed spatial representation extraction scheme for various tasks, resulting in insufficient representational capability or excessive extraction time. In this work, we introduce T-Rex, a Task-Adaptive Framework for Spatial Representation Extraction, which dynamically selects the most appropriate spatial representation extraction scheme for each entity based on specific task requirements. Our key insight is that task complexity determines the types and granularity of spatial representations, and Stronger representational capabilities are typically associated with Higher overall system operation costs. Through comprehensive experiments in real-world robotic environments, we show that our approach delivers significant advantages in spatial understanding, efficiency, and stability without additional training.
>
---
#### [new 017] Robotic Perception with a Large Tactile-Vision-Language Model for Physical Property Inference
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在通过融合视觉与触觉信息，提升对物体物理属性的推理能力，解决单一模态数据不足的问题。**

- **链接: [http://arxiv.org/pdf/2506.19303v1](http://arxiv.org/pdf/2506.19303v1)**

> **作者:** Zexiang Guo; Hengxiang Chen; Xinheng Mai; Qiusang Qiu; Gan Ma; Zhanat Kappassov; Qiang Li; Nutan Chen
>
> **备注:** This paper has been accepted by the 2025 International Conference on Climbing and Walking Robots (CLAWAR). These authors contributed equally to this work: Zexiang Guo, Hengxiang Chen, Xinheng Mai
>
> **摘要:** Inferring physical properties can significantly enhance robotic manipulation by enabling robots to handle objects safely and efficiently through adaptive grasping strategies. Previous approaches have typically relied on either tactile or visual data, limiting their ability to fully capture properties. We introduce a novel cross-modal perception framework that integrates visual observations with tactile representations within a multimodal vision-language model. Our physical reasoning framework, which employs a hierarchical feature alignment mechanism and a refined prompting strategy, enables our model to make property-specific predictions that strongly correlate with ground-truth measurements. Evaluated on 35 diverse objects, our approach outperforms existing baselines and demonstrates strong zero-shot generalization. Keywords: tactile perception, visual-tactile fusion, physical property inference, multimodal integration, robot perception
>
---
#### [new 018] Scaffolding Dexterous Manipulation with Vision-Language Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决复杂抓取训练难题。通过视觉-语言模型生成轨迹引导强化学习，提升策略泛化与实际应用能力。**

- **链接: [http://arxiv.org/pdf/2506.19212v1](http://arxiv.org/pdf/2506.19212v1)**

> **作者:** Vincent de Bakker; Joey Hejna; Tyler Ga Wei Lum; Onur Celik; Aleksandar Taranovic; Denis Blessing; Gerhard Neumann; Jeannette Bohg; Dorsa Sadigh
>
> **摘要:** Dexterous robotic hands are essential for performing complex manipulation tasks, yet remain difficult to train due to the challenges of demonstration collection and high-dimensional control. While reinforcement learning (RL) can alleviate the data bottleneck by generating experience in simulation, it typically relies on carefully designed, task-specific reward functions, which hinder scalability and generalization. Thus, contemporary works in dexterous manipulation have often bootstrapped from reference trajectories. These trajectories specify target hand poses that guide the exploration of RL policies and object poses that enable dense, task-agnostic rewards. However, sourcing suitable trajectories - particularly for dexterous hands - remains a significant challenge. Yet, the precise details in explicit reference trajectories are often unnecessary, as RL ultimately refines the motion. Our key insight is that modern vision-language models (VLMs) already encode the commonsense spatial and semantic knowledge needed to specify tasks and guide exploration effectively. Given a task description (e.g., "open the cabinet") and a visual scene, our method uses an off-the-shelf VLM to first identify task-relevant keypoints (e.g., handles, buttons) and then synthesize 3D trajectories for hand motion and object motion. Subsequently, we train a low-level residual RL policy in simulation to track these coarse trajectories or "scaffolds" with high fidelity. Across a number of simulated tasks involving articulated objects and semantic understanding, we demonstrate that our method is able to learn robust dexterous manipulation policies. Moreover, we showcase that our method transfers to real-world robotic hands without any human demonstrations or handcrafted rewards.
>
---
#### [new 019] Ontology Neural Network and ORTSF: A Framework for Topological Reasoning and Delay-Robust Control
- **分类: cs.RO; cs.SY; eess.SY; 68T40, 93C41; I.2.9; I.2.8; F.2.2**

- **简介: 该论文属于认知机器人任务，旨在解决动态环境中语义表示与控制延迟问题。提出ONN和ORTSF框架，实现语义推理与鲁棒控制的统一。**

- **链接: [http://arxiv.org/pdf/2506.19277v1](http://arxiv.org/pdf/2506.19277v1)**

> **作者:** Jaehong Oh
>
> **备注:** 12 pages, 5 figures, includes theoretical proofs and simulation results
>
> **摘要:** The advancement of autonomous robotic systems has led to impressive capabilities in perception, localization, mapping, and control. Yet, a fundamental gap remains: existing frameworks excel at geometric reasoning and dynamic stability but fall short in representing and preserving relational semantics, contextual reasoning, and cognitive transparency essential for collaboration in dynamic, human-centric environments. This paper introduces a unified architecture comprising the Ontology Neural Network (ONN) and the Ontological Real-Time Semantic Fabric (ORTSF) to address this gap. The ONN formalizes relational semantic reasoning as a dynamic topological process. By embedding Forman-Ricci curvature, persistent homology, and semantic tensor structures within a unified loss formulation, ONN ensures that relational integrity and topological coherence are preserved as scenes evolve over time. The ORTSF transforms reasoning traces into actionable control commands while compensating for system delays. It integrates predictive and delay-aware operators that ensure phase margin preservation and continuity of control signals, even under significant latency conditions. Empirical studies demonstrate the ONN + ORTSF framework's ability to unify semantic cognition and robust control, providing a mathematically principled and practically viable solution for cognitive robotics.
>
---
#### [new 020] CUPID: Curating Data your Robot Loves with Influence Functions
- **分类: cs.RO; cs.AI; cs.LG; I.2.6; I.2.9**

- **简介: 该论文属于机器人模仿学习任务，旨在解决数据质量影响策略性能的问题。提出CUPID方法，通过影响函数理论筛选有效数据，提升策略表现。**

- **链接: [http://arxiv.org/pdf/2506.19121v1](http://arxiv.org/pdf/2506.19121v1)**

> **作者:** Christopher Agia; Rohan Sinha; Jingyun Yang; Rika Antonova; Marco Pavone; Haruki Nishimura; Masha Itkina; Jeannette Bohg
>
> **备注:** Project page: https://cupid-curation.github.io. 28 pages, 15 figures
>
> **摘要:** In robot imitation learning, policy performance is tightly coupled with the quality and composition of the demonstration data. Yet, developing a precise understanding of how individual demonstrations contribute to downstream outcomes - such as closed-loop task success or failure - remains a persistent challenge. We propose CUPID, a robot data curation method based on a novel influence function-theoretic formulation for imitation learning policies. Given a set of evaluation rollouts, CUPID estimates the influence of each training demonstration on the policy's expected return. This enables ranking and selection of demonstrations according to their impact on the policy's closed-loop performance. We use CUPID to curate data by 1) filtering out training demonstrations that harm policy performance and 2) subselecting newly collected trajectories that will most improve the policy. Extensive simulated and hardware experiments show that our approach consistently identifies which data drives test-time performance. For example, training with less than 33% of curated data can yield state-of-the-art diffusion policies on the simulated RoboMimic benchmark, with similar gains observed in hardware. Furthermore, hardware experiments show that our method can identify robust strategies under distribution shift, isolate spurious correlations, and even enhance the post-training of generalist robot policies. Additional materials are made available at: https://cupid-curation.github.io.
>
---
#### [new 021] Soft Robotic Delivery of Coiled Anchors for Cardiac Interventions
- **分类: cs.RO**

- **简介: 该论文属于心脏介入任务，旨在解决传统导管平台操作不足的问题。研究开发了柔性机器人平台，实现高精度锚定线圈输送。**

- **链接: [http://arxiv.org/pdf/2506.19602v1](http://arxiv.org/pdf/2506.19602v1)**

> **作者:** Leonardo Zamora Yanez; Jacob Rogatinsky; Dominic Recco; Sang-Yoep Lee; Grace Matthews; Andrew P. Sabelhaus; Tommaso Ranzani
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Trans-catheter cardiac intervention has become an increasingly available option for high-risk patients without the complications of open heart surgery. However, current catheterbased platforms suffer from a lack of dexterity, force application, and compliance required to perform complex intracardiac procedures. An exemplary task that would significantly ease minimally invasive intracardiac procedures is the implantation of anchor coils, which can be used to fix and implant various devices in the beating heart. We introduce a robotic platform capable of delivering anchor coils. We develop a kineto-statics model of the robotic platform and demonstrate low positional error. We leverage the passive compliance and high force output of the actuator in a multi-anchor delivery procedure against a motile in-vitro simulator with millimeter level accuracy.
>
---
#### [new 022] AnchorDP3: 3D Affordance Guided Sparse Diffusion Policy for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AnchorDP3，用于双臂机器人操作任务，解决高随机环境下的操控问题。通过语义分割、任务编码和关键位姿预测，提升成功率。**

- **链接: [http://arxiv.org/pdf/2506.19269v1](http://arxiv.org/pdf/2506.19269v1)**

> **作者:** Ziyan Zhao; Ke Fan; He-Yang Xu; Ning Qiao; Bo Peng; Wenlong Gao; Dongjiang Li; Hui Shen
>
> **摘要:** We present AnchorDP3, a diffusion policy framework for dual-arm robotic manipulation that achieves state-of-the-art performance in highly randomized environments. AnchorDP3 integrates three key innovations: (1) Simulator-Supervised Semantic Segmentation, using rendered ground truth to explicitly segment task-critical objects within the point cloud, which provides strong affordance priors; (2) Task-Conditioned Feature Encoders, lightweight modules processing augmented point clouds per task, enabling efficient multi-task learning through a shared diffusion-based action expert; (3) Affordance-Anchored Keypose Diffusion with Full State Supervision, replacing dense trajectory prediction with sparse, geometrically meaningful action anchors, i.e., keyposes such as pre-grasp pose, grasp pose directly anchored to affordances, drastically simplifying the prediction space; the action expert is forced to predict both robot joint angles and end-effector poses simultaneously, which exploits geometric consistency to accelerate convergence and boost accuracy. Trained on large-scale, procedurally generated simulation data, AnchorDP3 achieves a 98.7% average success rate in the RoboTwin benchmark across diverse tasks under extreme randomization of objects, clutter, table height, lighting, and backgrounds. This framework, when integrated with the RoboTwin real-to-sim pipeline, has the potential to enable fully autonomous generation of deployable visuomotor policies from only scene and instruction, totally eliminating human demonstrations from learning manipulation skills.
>
---
#### [new 023] Analysis and experiments of the dissipative Twistcar: direction reversal and asymptotic approximations
- **分类: cs.RO**

- **简介: 该论文研究Twistcar的能耗与方向反转问题，通过理论分析和实验验证，探讨其在非完整约束下的动态特性及方向控制方法。**

- **链接: [http://arxiv.org/pdf/2506.19112v1](http://arxiv.org/pdf/2506.19112v1)**

> **作者:** Rom Levy; Ari Dantus; Zitao Yu; Yizhar Or
>
> **摘要:** Underactuated wheeled vehicles are commonly studied as nonholonomic systems with periodic actuation. Twistcar is a classical example inspired by a riding toy, which has been analyzed using a planar model of a dynamical system with nonholonomic constraints. Most of the previous analyses did not account for energy dissipation due to friction. In this work, we study a theoretical two-link model of the Twistcar while incorporating dissipation due to rolling resistance. We obtain asymptotic expressions for the system's small-amplitude steady-state periodic dynamics, which reveals the possibility of reversing the direction of motion upon varying the geometric and mass properties of the vehicle. Next, we design and construct a robotic prototype of the Twistcar whose center-of-mass position can be shifted by adding and removing a massive block, enabling demonstration of the Twistcar's direction reversal phenomenon. We also conduct parameter fitting for the frictional resistance in order to improve agreement with experiments.
>
---
#### [new 024] Situated Haptic Interaction: Exploring the Role of Context in Affective Perception of Robotic Touch
- **分类: cs.RO**

- **简介: 该论文属于人机情感交互任务，研究情境如何影响机器人触觉反馈的情感感知。通过实验揭示视觉与触觉信息的相互作用，为更自然的情感交流提供依据。**

- **链接: [http://arxiv.org/pdf/2506.19179v1](http://arxiv.org/pdf/2506.19179v1)**

> **作者:** Qiaoqiao Ren; Tony Belpaeme
>
> **摘要:** Affective interaction is not merely about recognizing emotions; it is an embodied, situated process shaped by context and co-created through interaction. In affective computing, the role of haptic feedback within dynamic emotional exchanges remains underexplored. This study investigates how situational emotional cues influence the perception and interpretation of haptic signals given by a robot. In a controlled experiment, 32 participants watched video scenarios in which a robot experienced either positive actions (such as being kissed), negative actions (such as being slapped) or neutral actions. After each video, the robot conveyed its emotional response through haptic communication, delivered via a wearable vibration sleeve worn by the participant. Participants rated the robot's emotional state-its valence (positive or negative) and arousal (intensity)-based on the video, the haptic feedback, and the combination of the two. The study reveals a dynamic interplay between visual context and touch. Participants' interpretation of haptic feedback was strongly shaped by the emotional context of the video, with visual context often overriding the perceived valence of the haptic signal. Negative haptic cues amplified the perceived valence of the interaction, while positive cues softened it. Furthermore, haptics override the participants' perception of arousal of the video. Together, these results offer insights into how situated haptic feedback can enrich affective human-robot interaction, pointing toward more nuanced and embodied approaches to emotional communication with machines.
>
---
#### [new 025] A Verification Methodology for Safety Assurance of Robotic Autonomous Systems
- **分类: cs.RO; cs.FL; cs.SE; cs.SY; eess.SY**

- **简介: 该论文属于安全验证任务，旨在解决自主农业机器人在共享环境中的安全问题。通过构建安全控制器模型并进行验证，确保系统符合安全要求。**

- **链接: [http://arxiv.org/pdf/2506.19622v1](http://arxiv.org/pdf/2506.19622v1)**

> **作者:** Mustafa Adam; David A. Anisi; Pedro Ribeiro
>
> **备注:** In Proc. of the 26th TAROS (Towards Autonomous Robotic Systems) Conference, York, UK, August, 2025
>
> **摘要:** Autonomous robots deployed in shared human environments, such as agricultural settings, require rigorous safety assurance to meet both functional reliability and regulatory compliance. These systems must operate in dynamic, unstructured environments, interact safely with humans, and respond effectively to a wide range of potential hazards. This paper presents a verification workflow for the safety assurance of an autonomous agricultural robot, covering the entire development life-cycle, from concept study and design to runtime verification. The outlined methodology begins with a systematic hazard analysis and risk assessment to identify potential risks and derive corresponding safety requirements. A formal model of the safety controller is then developed to capture its behaviour and verify that the controller satisfies the specified safety properties with respect to these requirements. The proposed approach is demonstrated on a field robot operating in an agricultural setting. The results show that the methodology can be effectively used to verify safety-critical properties and facilitate the early identification of design issues, contributing to the development of safer robots and autonomous systems.
>
---
#### [new 026] ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Model
- **分类: cs.RO**

- **简介: 该论文属于多任务双臂操作任务，旨在解决双臂协作中的动态建模问题。通过构建分层高斯世界模型，提升双臂协同操作性能。**

- **链接: [http://arxiv.org/pdf/2506.19842v1](http://arxiv.org/pdf/2506.19842v1)**

> **作者:** Tengbo Yu; Guanxing Lu; Zaijia Yang; Haoyuan Deng; Season Si Chen; Jiwen Lu; Wenbo Ding; Guoqiang Hu; Yansong Tang; Ziwei Wang
>
> **摘要:** Multi-task robotic bimanual manipulation is becoming increasingly popular as it enables sophisticated tasks that require diverse dual-arm collaboration patterns. Compared to unimanual manipulation, bimanual tasks pose challenges to understanding the multi-body spatiotemporal dynamics. An existing method ManiGaussian pioneers encoding the spatiotemporal dynamics into the visual representation via Gaussian world model for single-arm settings, which ignores the interaction of multiple embodiments for dual-arm systems with significant performance drop. In this paper, we propose ManiGaussian++, an extension of ManiGaussian framework that improves multi-task bimanual manipulation by digesting multi-body scene dynamics through a hierarchical Gaussian world model. To be specific, we first generate task-oriented Gaussian Splatting from intermediate visual features, which aims to differentiate acting and stabilizing arms for multi-body spatiotemporal dynamics modeling. We then build a hierarchical Gaussian world model with the leader-follower architecture, where the multi-body spatiotemporal dynamics is mined for intermediate visual representation via future scene prediction. The leader predicts Gaussian Splatting deformation caused by motions of the stabilizing arm, through which the follower generates the physical consequences resulted from the movement of the acting arm. As a result, our method significantly outperforms the current state-of-the-art bimanual manipulation techniques by an improvement of 20.2% in 10 simulated tasks, and achieves 60% success rate on average in 9 challenging real-world tasks. Our code is available at https://github.com/April-Yz/ManiGaussian_Bimanual.
>
---
#### [new 027] Fake or Real, Can Robots Tell? Evaluating Embodied Vision-Language Models on Real and 3D-Printed Objects
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于机器人场景理解任务，研究如何利用视觉-语言模型识别真实与3D打印物体，比较不同描述策略的效果。**

- **链接: [http://arxiv.org/pdf/2506.19579v1](http://arxiv.org/pdf/2506.19579v1)**

> **作者:** Federico Tavella; Kathryn Mearns; Angelo Cangelosi
>
> **摘要:** Robotic scene understanding increasingly relies on vision-language models (VLMs) to generate natural language descriptions of the environment. In this work, we present a comparative study of captioning strategies for tabletop scenes captured by a robotic arm equipped with an RGB camera. The robot collects images of objects from multiple viewpoints, and we evaluate several models that generate scene descriptions. We compare the performance of various captioning models, like BLIP and VLMs. Our experiments examine the trade-offs between single-view and multi-view captioning, and difference between recognising real-world and 3D printed objects. We quantitatively evaluate object identification accuracy, completeness, and naturalness of the generated captions. Results show that VLMs can be used in robotic settings where common objects need to be recognised, but fail to generalise to novel representations. Our findings provide practical insights into deploying foundation models for embodied agents in real-world settings.
>
---
#### [new 028] Preserving Sense of Agency: User Preferences for Robot Autonomy and User Control across Household Tasks
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，探讨用户对家用机器人自主性和控制权的偏好，旨在解决如何平衡机器人自主与用户控制的问题。研究通过实验分析不同自主级别对用户控制感的影响。**

- **链接: [http://arxiv.org/pdf/2506.19202v1](http://arxiv.org/pdf/2506.19202v1)**

> **作者:** Claire Yang; Heer Patel; Max Kleiman-Weiner; Maya Cakmak
>
> **备注:** Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)
>
> **摘要:** Roboticists often design with the assumption that assistive robots should be fully autonomous. However, it remains unclear whether users prefer highly autonomous robots, as prior work in assistive robotics suggests otherwise. High robot autonomy can reduce the user's sense of agency, which represents feeling in control of one's environment. How much control do users, in fact, want over the actions of robots used for in-home assistance? We investigate how robot autonomy levels affect users' sense of agency and the autonomy level they prefer in contexts with varying risks. Our study asked participants to rate their sense of agency as robot users across four distinct autonomy levels and ranked their robot preferences with respect to various household tasks. Our findings revealed that participants' sense of agency was primarily influenced by two factors: (1) whether the robot acts autonomously, and (2) whether a third party is involved in the robot's programming or operation. Notably, an end-user programmed robot highly preserved users' sense of agency, even though it acts autonomously. However, in high-risk settings, e.g., preparing a snack for a child with allergies, they preferred robots that prioritized their control significantly more. Additional contextual factors, such as trust in a third party operator, also shaped their preferences.
>
---
#### [new 029] AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决传统V2X系统在农村和郊区的覆盖不足问题。通过无人机辅助，构建了AirV2X-Perception数据集，支持V2D算法开发与评估。**

- **链接: [http://arxiv.org/pdf/2506.19283v1](http://arxiv.org/pdf/2506.19283v1)**

> **作者:** Xiangbo Gao; Yuheng Wu; Xuewen Luo; Keshu Wu; Xinghao Chen; Yuping Wang; Chenxi Liu; Yang Zhou; Zhengzhong Tu
>
> **摘要:** While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at https://github.com/taco-group/AirV2X-Perception.
>
---
#### [new 030] Low-Cost Infrastructure-Free 3D Relative Localization with Sub-Meter Accuracy in Near Field
- **分类: eess.SY; cs.MA; cs.RO; cs.SY**

- **简介: 该论文属于3D相对定位任务，解决UAV在近场环境下的高精度定位问题。通过UWB传感器和算法设计，实现低成本、无基础设施的定位方案。**

- **链接: [http://arxiv.org/pdf/2506.19199v1](http://arxiv.org/pdf/2506.19199v1)**

> **作者:** Qiangsheng Gao; Ka Ho Cheng; Li Qiu; Zijun Gong
>
> **摘要:** Relative localization in the near-field scenario is critically important for unmanned vehicle (UxV) applications. Although related works addressing 2D relative localization problem have been widely studied for unmanned ground vehicles (UGVs), the problem in 3D scenarios for unmanned aerial vehicles (UAVs) involves more uncertainties and remains to be investigated. Inspired by the phenomenon that animals can achieve swarm behaviors solely based on individual perception of relative information, this study proposes an infrastructure-free 3D relative localization framework that relies exclusively on onboard ultra-wideband (UWB) sensors. Leveraging 2D relative positioning research, we conducted feasibility analysis, system modeling, simulations, performance evaluation, and field tests using UWB sensors. The key contributions of this work include: derivation of the Cram\'er-Rao lower bound (CRLB) and geometric dilution of precision (GDOP) for near-field scenarios; development of two localization algorithms -- one based on Euclidean distance matrix (EDM) and another employing maximum likelihood estimation (MLE); comprehensive performance comparison and computational complexity analysis against state-of-the-art methods; simulation studies and field experiments; a novel sensor deployment strategy inspired by animal behavior, enabling single-sensor implementation within the proposed framework for UxV applications. The theoretical, simulation, and experimental results demonstrate strong generalizability to other 3D near-field localization tasks, with significant potential for a cost-effective cross-platform UxV collaborative system.
>
---
#### [new 031] EvDetMAV: Generalized MAV Detection from Moving Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决MAV在复杂场景下的检测难题。通过分析事件相机中的螺旋桨特征，提出新方法并构建首个事件数据集，提升检测精度。**

- **链接: [http://arxiv.org/pdf/2506.19416v1](http://arxiv.org/pdf/2506.19416v1)**

> **作者:** Yin Zhang; Zian Ning; Xiaoyu Zhang; Shiliang Guo; Peidong Liu; Shiyu Zhao
>
> **备注:** 8 pages, 7 figures. This paper is accepted by IEEE Robotics and Automation Letters
>
> **摘要:** Existing micro aerial vehicle (MAV) detection methods mainly rely on the target's appearance features in RGB images, whose diversity makes it difficult to achieve generalized MAV detection. We notice that different types of MAVs share the same distinctive features in event streams due to their high-speed rotating propellers, which are hard to see in RGB images. This paper studies how to detect different types of MAVs from an event camera by fully exploiting the features of propellers in the original event stream. The proposed method consists of three modules to extract the salient and spatio-temporal features of the propellers while filtering out noise from background objects and camera motion. Since there are no existing event-based MAV datasets, we introduce a novel MAV dataset for the community. This is the first event-based MAV dataset comprising multiple scenarios and different types of MAVs. Without training, our method significantly outperforms state-of-the-art methods and can deal with challenging scenarios, achieving a precision rate of 83.0\% (+30.3\%) and a recall rate of 81.5\% (+36.4\%) on the proposed testing dataset. The dataset and code are available at: https://github.com/WindyLab/EvDetMAV.
>
---
#### [new 032] Is an object-centric representation beneficial for robotic manipulation ?
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究机器人操作任务中对象中心表示的有效性，旨在提升数据效率和泛化能力。通过模拟环境测试，对比了对象中心与整体表示方法的效果。**

- **链接: [http://arxiv.org/pdf/2506.19408v1](http://arxiv.org/pdf/2506.19408v1)**

> **作者:** Alexandre Chapin; Emmanuel Dellandrea; Liming Chen
>
> **摘要:** Object-centric representation (OCR) has recently become a subject of interest in the computer vision community for learning a structured representation of images and videos. It has been several times presented as a potential way to improve data-efficiency and generalization capabilities to learn an agent on downstream tasks. However, most existing work only evaluates such models on scene decomposition, without any notion of reasoning over the learned representation. Robotic manipulation tasks generally involve multi-object environments with potential inter-object interaction. We thus argue that they are a very interesting playground to really evaluate the potential of existing object-centric work. To do so, we create several robotic manipulation tasks in simulated environments involving multiple objects (several distractors, the robot, etc.) and a high-level of randomization (object positions, colors, shapes, background, initial positions, etc.). We then evaluate one classical object-centric method across several generalization scenarios and compare its results against several state-of-the-art hollistic representations. Our results exhibit that existing methods are prone to failure in difficult scenarios involving complex scene structures, whereas object-centric methods help overcome these challenges.
>
---
#### [new 033] ReLink: Computational Circular Design of Planar Linkage Mechanisms Using Available Standard Parts
- **分类: cs.CE; cs.RO**

- **简介: 该论文属于机械设计任务，旨在用标准零件设计平面连杆机构，解决传统设计依赖定制部件的问题。通过生成与逆向设计方法，实现可持续机械系统。**

- **链接: [http://arxiv.org/pdf/2506.19657v1](http://arxiv.org/pdf/2506.19657v1)**

> **作者:** Maxime Escande; Kristina Shea
>
> **备注:** 29 pages, 18 figures, submitted to the Journal of Cleaner Production
>
> **摘要:** The Circular Economy framework emphasizes sustainability by reducing resource consumption and waste through the reuse of components and materials. This paper presents ReLink, a computational framework for the circular design of planar linkage mechanisms using available standard parts. Unlike most mechanism design methods, which assume the ability to create custom parts and infinite part availability, ReLink prioritizes the reuse of discrete, standardized components, thus minimizing the need for new parts. The framework consists of two main components: design generation, where a generative design algorithm generates mechanisms from an inventory of available parts, and inverse design, which uses optimization methods to identify designs that match a user-defined trajectory curve. The paper also examines the trade-offs between kinematic performance and CO2 footprint when incorporating new parts. Challenges such as the combinatorial nature of the design problem and the enforcement of valid solutions are addressed. By combining sustainability principles with kinematic synthesis, ReLink lays the groundwork for further research into computational circular design to support the development of systems that integrate reused components into mechanical products.
>
---
#### [new 034] Systematic Comparison of Projection Methods for Monocular 3D Human Pose Estimation on Fisheye Images
- **分类: cs.CV; cs.RO; I.2.10; I.2.9; I.4.8; I.4.9**

- **简介: 该论文属于单目3D人体姿态估计任务，旨在解决鱼眼图像中人体姿态准确检测的问题。通过系统比较不同投影方法，提出基于边界框的模型选择策略，并构建了新数据集FISHnCHIPS。**

- **链接: [http://arxiv.org/pdf/2506.19747v1](http://arxiv.org/pdf/2506.19747v1)**

> **作者:** Stephanie Käs; Sven Peter; Henrik Thillmann; Anton Burenko; David Benjamin Adrian; Dennis Mack; Timm Linder; Bastian Leibe
>
> **备注:** Presented at IEEE International Conference on Robotics and Automation 2025
>
> **摘要:** Fisheye cameras offer robots the ability to capture human movements across a wider field of view (FOV) than standard pinhole cameras, making them particularly useful for applications in human-robot interaction and automotive contexts. However, accurately detecting human poses in fisheye images is challenging due to the curved distortions inherent to fisheye optics. While various methods for undistorting fisheye images have been proposed, their effectiveness and limitations for poses that cover a wide FOV has not been systematically evaluated in the context of absolute human pose estimation from monocular fisheye images. To address this gap, we evaluate the impact of pinhole, equidistant and double sphere camera models, as well as cylindrical projection methods, on 3D human pose estimation accuracy. We find that in close-up scenarios, pinhole projection is inadequate, and the optimal projection method varies with the FOV covered by the human pose. The usage of advanced fisheye models like the double sphere model significantly enhances 3D human pose estimation accuracy. We propose a heuristic for selecting the appropriate projection model based on the detection bounding box to enhance prediction quality. Additionally, we introduce and evaluate on our novel dataset FISHnCHIPS, which features 3D human skeleton annotations in fisheye images, including images from unconventional angles, such as extreme close-ups, ground-mounted cameras, and wide-FOV poses, available at: https://www.vision.rwth-aachen.de/fishnchips
>
---
#### [new 035] Da Yu: Towards USV-Based Image Captioning for Waterway Surveillance and Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于图像描述生成任务，旨在解决水道环境的语义理解问题。通过构建WaterCaption数据集和提出Da Yu模型，提升USV的场景认知能力。**

- **链接: [http://arxiv.org/pdf/2506.19288v1](http://arxiv.org/pdf/2506.19288v1)**

> **作者:** Runwei Guan; Ningwei Ouyang; Tianhao Xu; Shaofeng Liang; Wei Dai; Yafeng Sun; Shang Gao; Songning Lai; Shanliang Yao; Xuming Hu; Ryan Wen Liu; Yutao Yue; Hui Xiong
>
> **备注:** 14 pages, 13 figures
>
> **摘要:** Automated waterway environment perception is crucial for enabling unmanned surface vessels (USVs) to understand their surroundings and make informed decisions. Most existing waterway perception models primarily focus on instance-level object perception paradigms (e.g., detection, segmentation). However, due to the complexity of waterway environments, current perception datasets and models fail to achieve global semantic understanding of waterways, limiting large-scale monitoring and structured log generation. With the advancement of vision-language models (VLMs), we leverage image captioning to introduce WaterCaption, the first captioning dataset specifically designed for waterway environments. WaterCaption focuses on fine-grained, multi-region long-text descriptions, providing a new research direction for visual geo-understanding and spatial scene cognition. Exactly, it includes 20.2k image-text pair data with 1.8 million vocabulary size. Additionally, we propose Da Yu, an edge-deployable multi-modal large language model for USVs, where we propose a novel vision-to-language projector called Nano Transformer Adaptor (NTA). NTA effectively balances computational efficiency with the capacity for both global and fine-grained local modeling of visual features, thereby significantly enhancing the model's ability to generate long-form textual outputs. Da Yu achieves an optimal balance between performance and efficiency, surpassing state-of-the-art models on WaterCaption and several other captioning benchmarks.
>
---
#### [new 036] Correspondence-Free Multiview Point Cloud Registration via Depth-Guided Joint Optimisation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多视角点云配准任务，旨在解决复杂环境中特征提取与数据关联困难的问题。提出一种无需对应关系的方法，通过深度图联合优化估计点云位姿和全局地图。**

- **链接: [http://arxiv.org/pdf/2506.18922v1](http://arxiv.org/pdf/2506.18922v1)**

> **作者:** Yiran Zhou; Yingyu Wang; Shoudong Huang; Liang Zhao
>
> **备注:** 8 pages, accepted for publication in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Multiview point cloud registration is a fundamental task for constructing globally consistent 3D models. Existing approaches typically rely on feature extraction and data association across multiple point clouds; however, these processes are challenging to obtain global optimal solution in complex environments. In this paper, we introduce a novel correspondence-free multiview point cloud registration method. Specifically, we represent the global map as a depth map and leverage raw depth information to formulate a non-linear least squares optimisation that jointly estimates poses of point clouds and the global map. Unlike traditional feature-based bundle adjustment methods, which rely on explicit feature extraction and data association, our method bypasses these challenges by associating multi-frame point clouds with a global depth map through their corresponding poses. This data association is implicitly incorporated and dynamically refined during the optimisation process. Extensive evaluations on real-world datasets demonstrate that our method outperforms state-of-the-art approaches in accuracy, particularly in challenging environments where feature extraction and data association are difficult.
>
---
#### [new 037] Adaptive Domain Modeling with Language Models: A Multi-Agent Approach to Task Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出TAPAS框架，结合大语言模型与符号规划，解决复杂任务规划问题，无需手动定义环境模型。**

- **链接: [http://arxiv.org/pdf/2506.19592v1](http://arxiv.org/pdf/2506.19592v1)**

> **作者:** Harisankar Babu; Philipp Schillinger; Tamim Asfour
>
> **摘要:** We introduce TAPAS (Task-based Adaptation and Planning using AgentS), a multi-agent framework that integrates Large Language Models (LLMs) with symbolic planning to solve complex tasks without the need for manually defined environment models. TAPAS employs specialized LLM-based agents that collaboratively generate and adapt domain models, initial states, and goal specifications as needed using structured tool-calling mechanisms. Through this tool-based interaction, downstream agents can request modifications from upstream agents, enabling adaptation to novel attributes and constraints without manual domain redefinition. A ReAct (Reason+Act)-style execution agent, coupled with natural language plan translation, bridges the gap between dynamically generated plans and real-world robot capabilities. TAPAS demonstrates strong performance in benchmark planning domains and in the VirtualHome simulated real-world environment.
>
---
#### [new 038] Unified Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出UniVLA，一种统一的视觉-语言-动作模型，解决机器人操作中动作生成问题，通过联合建模视觉、语言和动作信号，提升长时任务性能。**

- **链接: [http://arxiv.org/pdf/2506.19850v1](http://arxiv.org/pdf/2506.19850v1)**

> **作者:** Yuqi Wang; Xinghang Li; Wenxuan Wang; Junbo Zhang; Yingyan Li; Yuntao Chen; Xinlong Wang; Zhaoxiang Zhang
>
> **备注:** technical report
>
> **摘要:** Vision-language-action models (VLAs) have garnered significant attention for their potential in advancing robotic manipulation. However, previous approaches predominantly rely on the general comprehension capabilities of vision-language models (VLMs) to generate action signals, often overlooking the rich temporal and causal structure embedded in visual observations. In this paper, we present UniVLA, a unified and native multimodal VLA model that autoregressively models vision, language, and action signals as discrete token sequences. This formulation enables flexible multimodal tasks learning, particularly from large-scale video data. By incorporating world modeling during post-training, UniVLA captures causal dynamics from videos, facilitating effective transfer to downstream policy learning--especially for long-horizon tasks. Our approach sets new state-of-the-art results across several widely used simulation benchmarks, including CALVIN, LIBERO, and Simplenv-Bridge, significantly surpassing previous methods. For example, UniVLA achieves 95.5% average success rate on LIBERO benchmark, surpassing pi0-FAST's 85.5%. We further demonstrate its broad applicability on real-world ALOHA manipulation and autonomous driving.
>
---
## 更新

#### [replaced 001] ContactDexNet: Multi-fingered Robotic Hand Grasping in Cluttered Environments through Hand-object Contact Semantic Mapping
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2404.08844v3](http://arxiv.org/pdf/2404.08844v3)**

> **作者:** Lei Zhang; Kaixin Bai; Guowen Huang; Zhenshan Bing; Zhaopeng Chen; Alois Knoll; Jianwei Zhang
>
> **备注:** 8 pages
>
> **摘要:** The deep learning models has significantly advanced dexterous manipulation techniques for multi-fingered hand grasping. However, the contact information-guided grasping in cluttered environments remains largely underexplored. To address this gap, we have developed a method for generating multi-fingered hand grasp samples in cluttered settings through contact semantic map. We introduce a contact semantic conditional variational autoencoder network (CoSe-CVAE) for creating comprehensive contact semantic map from object point cloud. We utilize grasp detection method to estimate hand grasp poses from the contact semantic map. Finally, an unified grasp evaluation model PointNetGPD++ is designed to assess grasp quality and collision probability, substantially improving the reliability of identifying optimal grasps in cluttered scenarios. Our grasp generation method has demonstrated remarkable success, outperforming state-of-the-art methods by at least 4.65% with 81.0% average grasping success rate in real-world single-object environment and 75.3% grasping success rate in cluttered scenes. We also proposed the multi-modal multi-fingered grasping dataset generation method. Our multi-fingered hand grasping dataset outperforms previous datasets in scene diversity, modality diversity. The dataset, code and supplementary materials can be found at https://sites.google.com/view/contact-dexnet.
>
---
#### [replaced 002] Terrain-aware Low Altitude Path Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.07141v2](http://arxiv.org/pdf/2505.07141v2)**

> **作者:** Yixuan Jia; Andrea Tagliabue; Annika Thomas; Navid Dadkhah Tehrani; Jonathan P. How
>
> **摘要:** In this paper, we study the problem of generating low-altitude path plans for nap-of-the-earth (NOE) flight in real time with only RGB images from onboard cameras and the vehicle pose. We propose a novel training method that combines behavior cloning and self-supervised learning, where the self-supervision component allows the learned policy to refine the paths generated by the expert planner. Simulation studies show 24.7% reduction in average path elevation compared to the standard behavior cloning approach.
>
---
#### [replaced 003] Learning Realistic Joint Space Boundaries for Range of Motion Analysis of Healthy and Impaired Human Arms
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.10653v3](http://arxiv.org/pdf/2311.10653v3)**

> **作者:** Shafagh Keyvanian; Michelle J. Johnson; Nadia Figueroa
>
> **摘要:** A realistic human kinematic model that satisfies anatomical constraints is essential for human-robot interaction, biomechanics and robot-assisted rehabilitation. Modeling realistic joint constraints, however, is challenging as human arm motion is constrained by joint limits, inter- and intra-joint dependencies, self-collisions, individual capabilities and muscular or neurological constraints which are difficult to represent. Hence, physicians and researchers have relied on simple box-constraints, ignoring important anatomical factors. In this paper, we propose a data-driven method to learn realistic anatomically constrained upper-limb range of motion (RoM) boundaries from motion capture data. This is achieved by fitting a one-class support vector machine to a dataset of upper-limb joint space exploration motions with an efficient hyper-parameter tuning scheme. Our approach outperforms similar works focused on valid RoM learning. Further, we propose an impairment index (II) metric that offers a quantitative assessment of capability/impairment when comparing healthy and impaired arms. We validate the metric on healthy subjects physically constrained to emulate hemiplegia and different disability levels as stroke patients. [https://sites.google.com/seas.upenn.edu/learning-rom]
>
---
#### [replaced 004] Human-Robot Teaming Field Deployments: A Comparison Between Verbal and Non-verbal Communication
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2506.08890v2](http://arxiv.org/pdf/2506.08890v2)**

> **作者:** Tauhid Tanjim; Promise Ekpo; Huajie Cao; Jonathan St. George; Kevin Ching; Hee Rin Lee; Angelique Taylor
>
> **备注:** This is the author's original submitted version of the paper accepted to the 2025 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). \c{opyright} 2025 IEEE. Personal use of this material is permitted. For any other use, please contact IEEE
>
> **摘要:** Healthcare workers (HCWs) encounter challenges in hospitals, such as retrieving medical supplies quickly from crash carts, which could potentially result in medical errors and delays in patient care. Robotic crash carts (RCCs) have shown promise in assisting healthcare teams during medical tasks through guided object searches and task reminders. Limited exploration has been done to determine what communication modalities are most effective and least disruptive to patient care in real-world settings. To address this gap, we conducted a between-subjects experiment comparing the RCC's verbal and non-verbal communication of object search with a standard crash cart in resuscitation scenarios to understand the impact of robot communication on workload and attitudes toward using robots in the workplace. Our findings indicate that verbal communication significantly reduced mental demand and effort compared to visual cues and with a traditional crash cart. Although frustration levels were slightly higher during collaborations with the robot compared to a traditional cart, these research insights provide valuable implications for human-robot teamwork in high-stakes environments.
>
---
#### [replaced 005] DroneDiffusion: Robust Quadrotor Dynamics Learning with Diffusion Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11292v2](http://arxiv.org/pdf/2409.11292v2)**

> **作者:** Avirup Das; Rishabh Dev Yadav; Sihao Sun; Mingfei Sun; Samuel Kaski; Wei Pan
>
> **备注:** Accepted to the International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** An inherent fragility of quadrotor systems stems from model inaccuracies and external disturbances. These factors hinder performance and compromise the stability of the system, making precise control challenging. Existing model-based approaches either make deterministic assumptions, utilize Gaussian-based representations of uncertainty, or rely on nominal models, all of which often fall short in capturing the complex, multimodal nature of real-world dynamics. This work introduces DroneDiffusion, a novel framework that leverages conditional diffusion models to learn quadrotor dynamics, formulated as a sequence generation task. DroneDiffusion achieves superior generalization to unseen, complex scenarios by capturing the temporal nature of uncertainties and mitigating error propagation. We integrate the learned dynamics with an adaptive controller for trajectory tracking with stability guarantees. Extensive experiments in both simulation and real-world flights demonstrate the robustness of the framework across a range of scenarios, including unfamiliar flight paths and varying payloads, velocities, and wind disturbances.
>
---
#### [replaced 006] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making
- **分类: cs.RO; cs.AI; I.2.9; I.2.10; I.4.8; H.5.2**

- **链接: [http://arxiv.org/pdf/2506.12374v2](http://arxiv.org/pdf/2506.12374v2)**

> **作者:** Wenbo Li; Shiyi Wang; Yiteng Chen; Huiping Zhuang; Qingyao Wu
>
> **备注:** submitted to NeurIPS 2025
>
> **摘要:** Vision-Language Models (VLMs) encode knowledge and reasoning capabilities for robotic manipulation within high-dimensional representation spaces. However, current approaches often project them into compressed intermediate representations, discarding important task-specific information such as fine-grained spatial or semantic details. To address this, we propose AntiGrounding, a new framework that reverses the instruction grounding process. It lifts candidate actions directly into the VLM representation space, renders trajectories from multiple views, and uses structured visual question answering for instruction-based decision making. This enables zero-shot synthesis of optimal closed-loop robot trajectories for new tasks. We also propose an offline policy refinement module that leverages past experience to enhance long-term performance. Experiments in both simulation and real-world environments show that our method outperforms baselines across diverse robotic manipulation tasks.
>
---
#### [replaced 007] Experimental Setup and Software Pipeline to Evaluate Optimization based Autonomous Multi-Robot Search Algorithms
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.16710v2](http://arxiv.org/pdf/2506.16710v2)**

> **作者:** Aditya Bhatt; Mary Katherine Corra; Franklin Merlo; Prajit KrisshnaKumar; Souma Chowdhury
>
> **备注:** IDETC 2025
>
> **摘要:** Signal source localization has been a problem of interest in the multi-robot systems domain given its applications in search & rescue and hazard localization in various industrial and outdoor settings. A variety of multi-robot search algorithms exist that usually formulate and solve the associated autonomous motion planning problem as a heuristic model-free or belief model-based optimization process. Most of these algorithms however remains tested only in simulation, thereby losing the opportunity to generate knowledge about how such algorithms would compare/contrast in a real physical setting in terms of search performance and real-time computing performance. To address this gap, this paper presents a new lab-scale physical setup and associated open-source software pipeline to evaluate and benchmark multi-robot search algorithms. The presented physical setup innovatively uses an acoustic source (that is safe and inexpensive) and small ground robots (e-pucks) operating in a standard motion-capture environment. This setup can be easily recreated and used by most robotics researchers. The acoustic source also presents interesting uncertainty in terms of its noise-to-signal ratio, which is useful to assess sim-to-real gaps. The overall software pipeline is designed to readily interface with any multi-robot search algorithm with minimal effort and is executable in parallel asynchronous form. This pipeline includes a framework for distributed implementation of multi-robot or swarm search algorithms, integrated with a ROS (Robotics Operating System)-based software stack for motion capture supported localization. The utility of this novel setup is demonstrated by using it to evaluate two state-of-the-art multi-robot search algorithms, based on swarm optimization and batch-Bayesian Optimization (called Bayes-Swarm), as well as a random walk baseline.
>
---
#### [replaced 008] Robustness Assessment of Assemblies in Frictional Contact
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.09810v2](http://arxiv.org/pdf/2411.09810v2)**

> **作者:** Philippe Nadeau; Jonathan Kelly
>
> **备注:** Submitted to IEEE Transactions on Automation Science and Engineering. Contains 14 pages, 16 figures, and 3 tables
>
> **摘要:** This work establishes a solution to the problem of assessing the capacity of multi-object assemblies to withstand external forces without becoming unstable. Our physically-grounded approach handles arbitrary structures made from rigid objects of any shape and mass distribution without relying on heuristics or approximations. The result is a method that provides a foundation for autonomous robot decision-making when interacting with objects in frictional contact. Our strategy relies on a contact interface graph representation to reason about instabilities and makes use of object shape information to decouple sub-problems and improve efficiency. Our algorithm can be used by motion planners to produce safe assembly transportation plans, and by object placement planners to select better poses. Compared to prior work, our approach is more generally applicable than commonly used heuristics and more efficient than dynamics simulations.
>
---
#### [replaced 009] DynNPC: Finding More Violations Induced by ADS in Simulation Testing through Dynamic NPC Behavior Generation
- **分类: cs.SE; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.19567v2](http://arxiv.org/pdf/2411.19567v2)**

> **作者:** You Lu; Yifan Tian; Dingji Wang; Bihuan Chen; Xin Peng
>
> **摘要:** Recently, a number of simulation testing approaches have been proposed to generate diverse driving scenarios for autonomous driving systems (ADSs) testing. However, the behaviors of NPC vehicles in these scenarios generated by previous approaches are predefined and mutated before simulation execution, ignoring traffic signals and the behaviors of the Ego vehicle. Thus, a large number of the violations they found are induced by unrealistic behaviors of NPC vehicles, revealing no bugs of ADSs. Besides, the vast scenario search space of NPC behaviors during the iterative mutations limits the efficiency of previous approaches. To address these limitations, we propose a novel scenario-based testing framework, DynNPC, to generate more violation scenarios induced by the ADS. Specifically, DynNPC allows NPC vehicles to dynamically generate behaviors using different driving strategies during simulation execution based on traffic signals and the real-time behavior of the Ego vehicle. We compare DynNPC with five state-of-the-art scenario-based testing approaches. Our evaluation has demonstrated the effectiveness and efficiency of DynNPC in finding more violation scenarios induced by the ADS.
>
---
#### [replaced 010] Fully distributed and resilient source seeking for robot swarms
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2410.15921v2](http://arxiv.org/pdf/2410.15921v2)**

> **作者:** Jesús Bautista; Antonio Acuaviva; José Hinojosa; Weijia Yao; Juan Jiménez; Héctor García de Marina
>
> **备注:** 16 pages, submitted version to T-TAC. Jesus Bautista and Antonio Acuaviva contributed equally to this work. arXiv admin note: text overlap with arXiv:2309.02937
>
> **摘要:** We propose a self-contained, resilient and fully distributed solution for locating the maximum of an unknown scalar field using a swarm of robots that travel at a constant speed. Unlike conventional reactive methods relying on gradient information, our methodology enables the swarm to determine an ascending direction so that it approaches the source with an arbitrary precision. Our source-seeking solution consists of three distributed algorithms running simultaneously in a slow-fast closed-loop system. The fastest algorithm provides the centroid-relative coordinates of the robots and the next slower one provides the ascending direction to be tracked. The tracking of the ascending direction by single integrators is instantaneous; howeverin this paper we will also focus on 2D unicycle-like robots with a constant speed. The third algorithm, the slowest one since the speed of the robots can be chosen arbitrarily slow, is the individual control law for the unicycle to track the estimated ascending direction.We will show that the three distributed algorithms converge exponentially fast to their objectives, allowing for a feasible slow-fast closed-loop system. The robots are not constrained to any particular geometric formation, and we study both discrete and continuous distributions of robots.The swarm shape analysis reveals the resiliency of our approach as expected in robot swarms, i.e., by amassing robots we ensure the source-seeking functionality in the event of missing or misplaced individuals or even if the robot network splits in two or more disconnected subnetworks.We exploit such an analysis so that the swarm can adapt to unknown environments by morphing its shape and maneuvering while still following an ascending direction. We analyze our solution with robots as kinematic points in n-dimensional Euclidean spaces and extend the analysis to 2D unicycle-like robots with constant speeds.
>
---
#### [replaced 011] Toward Teach and Repeat Across Seasonal Deep Snow Accumulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.01339v2](http://arxiv.org/pdf/2505.01339v2)**

> **作者:** Matěj Boxan; Alexander Krawciw; Timothy D. Barfoot; François Pomerleau
>
> **摘要:** Teach and repeat is a rapid way to achieve autonomy in challenging terrain and off-road environments. A human operator pilots the vehicles to create a network of paths that are mapped and associated with odometry. Immediately after teaching, the system can drive autonomously within its tracks. This precision lets operators remain confident that the robot will follow a traversable route. However, this operational paradigm has rarely been explored in off-road environments that change significantly through seasonal variation. This paper presents preliminary field trials using lidar and radar implementations of teach and repeat. Using a subset of the data from the upcoming FoMo dataset, we attempted to repeat routes that were 4 days, 44 days, and 113 days old. Lidar teach and repeat demonstrated a stronger ability to localize when the ground points were removed. FMCW radar was often able to localize on older maps, but only with small deviations from the taught path. Additionally, we highlight specific cases where radar localization failed with recent maps due to the high pitch or roll of the vehicle. We highlight lessons learned during the field deployment and highlight areas to improve to achieve reliable teach and repeat with seasonal changes in the environment. Please follow the dataset at https://norlab-ulaval.github.io/FoMo-website for updates and information on the data release.
>
---
#### [replaced 012] Pseudo-Kinematic Trajectory Control and Planning of Tracked Vehicles
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.18641v2](http://arxiv.org/pdf/2409.18641v2)**

> **作者:** Michele Focchi; Daniele Fontanelli; Davide Stocco; Riccardo Bussola; Luigi Palopoli
>
> **摘要:** Tracked vehicles distribute their weight continuously over a large surface area (the tracks). This distinctive feature makes them the preferred choice for vehicles required to traverse soft and uneven terrain. From a robotics perspective, however, this flexibility comes at a cost: the complexity of modelling the system and the resulting difficulty in designing theoretically sound navigation solutions. In this paper, we aim to bridge this gap by proposing a framework for the navigation of tracked vehicles, built upon three key pillars. The first pillar comprises two models: a simulation model and a control-oriented model. The simulation model captures the intricate terramechanics dynamics arising from soil-track interaction and is employed to develop faithful digital twins of the system across a wide range of operating conditions. The control-oriented model is pseudo-kinematic and mathematically tractable, enabling the design of efficient and theoretically robust control schemes. The second pillar is a Lyapunov-based feedback trajectory controller that provides certifiable tracking guarantees. The third pillar is a portfolio of motion planning solutions, each offering different complexity-accuracy trade-offs. The various components of the proposed approach are validated through an extensive set of simulation and experimental data.
>
---
#### [replaced 013] Learning Accurate Whole-body Throwing with High-frequency Residual Policy and Pullback Tube Acceleration
- **分类: cs.RO; 68T40, 93C85, 70E60; I.2.9; I.2.10; I.2.8**

- **链接: [http://arxiv.org/pdf/2506.16986v3](http://arxiv.org/pdf/2506.16986v3)**

> **作者:** Yuntao Ma; Yang Liu; Kaixian Qu; Marco Hutter
>
> **备注:** 8 pages, IROS 2025
>
> **摘要:** Throwing is a fundamental skill that enables robots to manipulate objects in ways that extend beyond the reach of their arms. We present a control framework that combines learning and model-based control for prehensile whole-body throwing with legged mobile manipulators. Our framework consists of three components: a nominal tracking policy for the end-effector, a high-frequency residual policy to enhance tracking accuracy, and an optimization-based module to improve end-effector acceleration control. The proposed controller achieved the average of 0.28 m landing error when throwing at targets located 6 m away. Furthermore, in a comparative study with university students, the system achieved a velocity tracking error of 0.398 m/s and a success rate of 56.8%, hitting small targets randomly placed at distances of 3-5 m while throwing at a specified speed of 6 m/s. In contrast, humans have a success rate of only 15.2%. This work provides an early demonstration of prehensile throwing with quantified accuracy on hardware, contributing to progress in dynamic whole-body manipulation.
>
---
#### [replaced 014] FusionForce: End-to-end Differentiable Neural-Symbolic Layer for Trajectory Prediction
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.10156v4](http://arxiv.org/pdf/2502.10156v4)**

> **作者:** Ruslan Agishev; Karel Zimmermann
>
> **备注:** Code: https://github.com/ctu-vras/fusionforce
>
> **摘要:** We propose end-to-end differentiable model that predicts robot trajectories on rough offroad terrain from camera images and/or lidar point clouds. The model integrates a learnable component that predicts robot-terrain interaction forces with a neural-symbolic layer that enforces the laws of classical mechanics and consequently improves generalization on out-of-distribution data. The neural-symbolic layer includes a differentiable physics engine that computes the robot's trajectory by querying these forces at the points of contact with the terrain. As the proposed architecture comprises substantial geometrical and physics priors, the resulting model can also be seen as a learnable physics engine conditioned on real sensor data that delivers $10^4$ trajectories per second. We argue and empirically demonstrate that this architecture reduces the sim-to-real gap and mitigates out-of-distribution sensitivity. The differentiability, in conjunction with the rapid simulation speed, makes the model well-suited for various applications including model predictive control, trajectory shooting, supervised and reinforcement learning, or SLAM.
>
---
#### [replaced 015] TeViR: Text-to-Video Reward with Diffusion Models for Efficient Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.19769v2](http://arxiv.org/pdf/2505.19769v2)**

> **作者:** Yuhui Chen; Haoran Li; Zhennan Jiang; Haowei Wen; Dongbin Zhao
>
> **摘要:** Developing scalable and generalizable reward engineering for reinforcement learning (RL) is crucial for creating general-purpose agents, especially in the challenging domain of robotic manipulation. While recent advances in reward engineering with Vision-Language Models (VLMs) have shown promise, their sparse reward nature significantly limits sample efficiency. This paper introduces TeViR, a novel method that leverages a pre-trained text-to-video diffusion model to generate dense rewards by comparing the predicted image sequence with current observations. Experimental results across 11 complex robotic tasks demonstrate that TeViR outperforms traditional methods leveraging sparse rewards and other state-of-the-art (SOTA) methods, achieving better sample efficiency and performance without ground truth environmental rewards. TeViR's ability to efficiently guide agents in complex environments highlights its potential to advance reinforcement learning applications in robotic manipulation.
>
---
#### [replaced 016] Why Sample Space Matters: Keyframe Sampling Optimization for LiDAR-based Place Recognition
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.02643v3](http://arxiv.org/pdf/2410.02643v3)**

> **作者:** Nikolaos Stathoulopoulos; Vidya Sumathy; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** The work is no longer intended for consideration in its current form. Readers are instead encouraged to refer to our related and more complete study, arXiv:2501.01791, which should be considered as a stand-alone contribution
>
> **摘要:** Recent advances in robotics are driving real-world autonomy for long-term and large-scale missions, where loop closures via place recognition are vital for mitigating pose estimation drift. However, achieving real-time performance remains challenging for resource-constrained mobile robots and multi-robot systems due to the computational burden of high-density sampling, which increases the complexity of comparing and verifying query samples against a growing map database. Conventional methods often retain redundant information or miss critical data by relying on fixed sampling intervals or operating in 3-D space instead of the descriptor feature space. To address these challenges, we introduce the concept of sample space and propose a novel keyframe sampling approach for LiDAR-based place recognition. Our method minimizes redundancy while preserving essential information in the hyper-dimensional descriptor space, supporting both learning-based and handcrafted descriptors. The proposed approach incorporates a sliding window optimization strategy to ensure efficient keyframe selection and real-time performance, enabling seamless integration into robotic pipelines. In sum, our approach demonstrates robust performance across diverse datasets, with the ability to adapt seamlessly from indoor to outdoor scenarios without parameter tuning, reducing loop closure detection times and memory requirements.
>
---
#### [replaced 017] Employing Laban Shape for Generating Emotionally and Functionally Expressive Trajectories in Robotic Manipulators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11716v2](http://arxiv.org/pdf/2505.11716v2)**

> **作者:** Srikrishna Bangalore Raghu; Clare Lohrmann; Akshay Bakshi; Jennifer Kim; Jose Caraveo Herrera; Bradley Hayes; Alessandro Roncone
>
> **备注:** Accepted for presentation at the 2025 IEEE RO-MAN Conference
>
> **摘要:** Successful human-robot collaboration depends on cohesive communication and a precise understanding of the robot's abilities, goals, and constraints. While robotic manipulators offer high precision, versatility, and productivity, they exhibit expressionless and monotonous motions that conceal the robot's intention, resulting in a lack of efficiency and transparency with humans. In this work, we use Laban notation, a dance annotation language, to enable robotic manipulators to generate trajectories with functional expressivity, where the robot uses nonverbal cues to communicate its abilities and the likelihood of succeeding at its task. We achieve this by introducing two novel variants of Hesitant expressive motion (Spoke-Like and Arc-Like). We also enhance the emotional expressivity of four existing emotive trajectories (Happy, Sad, Shy, and Angry) by augmenting Laban Effort usage with Laban Shape. The functionally expressive motions are validated via a human-subjects study, where participants equate both variants of Hesitant motion with reduced robot competency. The enhanced emotive trajectories are shown to be viewed as distinct emotions using the Valence-Arousal-Dominance (VAD) spectrum, corroborating the usage of Laban Shape.
>
---
#### [replaced 018] Energy-Efficient Motion Planner for Legged Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.06050v2](http://arxiv.org/pdf/2503.06050v2)**

> **作者:** Alexander Schperberg; Marcel Menner; Stefano Di Cairano
>
> **备注:** This paper has been accepted for publication at the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025). 8 pages, 8 figures
>
> **摘要:** We propose an online motion planner for legged robot locomotion with the primary objective of achieving energy efficiency. The conceptual idea is to leverage a placement set of footstep positions based on the robot's body position to determine when and how to execute steps. In particular, the proposed planner uses virtual placement sets beneath the hip joints of the legs and executes a step when the foot is outside of such placement set. Furthermore, we propose a parameter design framework that considers both energy-efficiency and robustness measures to optimize the gait by changing the shape of the placement set along with other parameters, such as step height and swing time, as a function of walking speed. We show that the planner produces trajectories that have a low Cost of Transport (CoT) and high robustness measure, and evaluate our approach against model-free Reinforcement Learning (RL) and motion imitation using biological dog motion priors as the reference. Overall, within low to medium velocity range, we show a 50.4% improvement in CoT and improved robustness over model-free RL, our best performing baseline. Finally, we show ability to handle slippery surfaces, gait transitions, and disturbances in simulation and hardware with the Unitree A1 robot.
>
---
#### [replaced 019] SemGauss-SLAM: Dense Semantic Gaussian Splatting SLAM
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.07494v4](http://arxiv.org/pdf/2403.07494v4)**

> **作者:** Siting Zhu; Renjie Qin; Guangming Wang; Jiuming Liu; Hesheng Wang
>
> **备注:** IROS 2025
>
> **摘要:** We propose SemGauss-SLAM, a dense semantic SLAM system utilizing 3D Gaussian representation, that enables accurate 3D semantic mapping, robust camera tracking, and high-quality rendering simultaneously. In this system, we incorporate semantic feature embedding into 3D Gaussian representation, which effectively encodes semantic information within the spatial layout of the environment for precise semantic scene representation. Furthermore, we propose feature-level loss for updating 3D Gaussian representation, enabling higher-level guidance for 3D Gaussian optimization. In addition, to reduce cumulative drift in tracking and improve semantic reconstruction accuracy, we introduce semantic-informed bundle adjustment. By leveraging multi-frame semantic associations, this strategy enables joint optimization of 3D Gaussian representation and camera poses, resulting in low-drift tracking and accurate semantic mapping. Our SemGauss-SLAM demonstrates superior performance over existing radiance field-based SLAM methods in terms of mapping and tracking accuracy on Replica and ScanNet datasets, while also showing excellent capabilities in high-precision semantic segmentation and dense semantic mapping.
>
---
#### [replaced 020] cuVSLAM: CUDA accelerated visual odometry and mapping
- **分类: cs.RO; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.04359v2](http://arxiv.org/pdf/2506.04359v2)**

> **作者:** Alexander Korovko; Dmitry Slepichev; Alexander Efitorov; Aigul Dzhumamuratova; Viktor Kuznetsov; Hesam Rabeti; Joydeep Biswas; Soha Pouya
>
> **摘要:** Accurate and robust pose estimation is a key requirement for any autonomous robot. We present cuVSLAM, a state-of-the-art solution for visual simultaneous localization and mapping, which can operate with a variety of visual-inertial sensor suites, including multiple RGB and depth cameras, and inertial measurement units. cuVSLAM supports operation with as few as one RGB camera to as many as 32 cameras, in arbitrary geometric configurations, thus supporting a wide range of robotic setups. cuVSLAM is specifically optimized using CUDA to deploy in real-time applications with minimal computational overhead on edge-computing devices such as the NVIDIA Jetson. We present the design and implementation of cuVSLAM, example use cases, and empirical results on several state-of-the-art benchmarks demonstrating the best-in-class performance of cuVSLAM.
>
---
#### [replaced 021] Stochastic Motion Planning as Gaussian Variational Inference: Theory and Algorithms
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2308.14985v3](http://arxiv.org/pdf/2308.14985v3)**

> **作者:** Hongzhe Yu; Yongxin Chen
>
> **备注:** 20 pages
>
> **摘要:** We present a novel formulation for motion planning under uncertainties based on variational inference where the optimal motion plan is modeled as a posterior distribution. We propose a Gaussian variational inference-based framework, termed Gaussian Variational Inference Motion Planning (GVI-MP), to approximate this posterior by a Gaussian distribution over the trajectories. We show that the GVI-MP framework is dual to a special class of stochastic control problems and brings robustness into the decision-making in motion planning. We develop two algorithms to numerically solve this variational inference and the equivalent control formulations for motion planning. The first algorithm uses a natural gradient paradigm to iteratively update a Gaussian proposal distribution on the sparse motion planning factor graph. We propose a second algorithm, the Proximal Covariance Steering Motion Planner (PCS-MP), to solve the same inference problem in its stochastic control form with an additional terminal constraint. We leverage a proximal gradient paradigm where, at each iteration, we quadratically approximate nonlinear state costs and solve a linear covariance steering problem in closed form. The efficacy of the proposed algorithms is demonstrated through extensive experiments on various robot models. An implementation is provided in https://github.com/hzyu17/VIMP.
>
---
#### [replaced 022] ros2 fanuc interface: Design and Evaluation of a Fanuc CRX Hardware Interface in ROS2
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.14487v2](http://arxiv.org/pdf/2506.14487v2)**

> **作者:** Paolo Franceschi; Marco Faroni; Stefano Baraldo; Anna Valente
>
> **摘要:** This paper introduces the ROS2 control and the Hardware Interface (HW) integration for the Fanuc CRX- robot family. It explains basic implementation details and communication protocols, and its integration with the Moveit2 motion planning library. We conducted a series of experiments to evaluate relevant performances in the robotics field. We tested the developed ros2_fanuc_interface for four relevant robotics cases: step response, trajectory tracking, collision avoidance integrated with Moveit2, and dynamic velocity scaling, respectively. Results show that, despite a non-negligible delay between command and feedback, the robot can track the defined path with negligible errors (if it complies with joint velocity limits), ensuring collision avoidance. Full code is open source and available at https://github.com/paolofrance/ros2_fanuc_interface.
>
---
#### [replaced 023] Overlap-Aware Feature Learning for Robust Unsupervised Domain Adaptation for 3D Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.01668v3](http://arxiv.org/pdf/2504.01668v3)**

> **作者:** Junjie Chen; Yuecong Xu; Haosheng Li; Kemi Ding
>
> **备注:** This paper has been accepted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** 3D point cloud semantic segmentation (PCSS) is a cornerstone for environmental perception in robotic systems and autonomous driving, enabling precise scene understanding through point-wise classification. While unsupervised domain adaptation (UDA) mitigates label scarcity in PCSS, existing methods critically overlook the inherent vulnerability to real-world perturbations (e.g., snow, fog, rain) and adversarial distortions. This work first identifies two intrinsic limitations that undermine current PCSS-UDA robustness: (a) unsupervised features overlap from unaligned boundaries in shared-class regions and (b) feature structure erosion caused by domain-invariant learning that suppresses target-specific patterns. To address the proposed problems, we propose a tripartite framework consisting of: 1) a robustness evaluation model quantifying resilience against adversarial attack/corruption types through robustness metrics; 2) an invertible attention alignment module (IAAM) enabling bidirectional domain mapping while preserving discriminative structure via attention-guided overlap suppression; and 3) a contrastive memory bank with quality-aware contrastive learning that progressively refines pseudo-labels with feature quality for more discriminative representations. Extensive experiments on SynLiDAR-to-SemanticPOSS adaptation demonstrate a maximum mIoU improvement of 14.3\% under adversarial attack.
>
---
#### [replaced 024] Perspective-Shifted Neuro-Symbolic World Models: A Framework for Socially-Aware Robot Navigation
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.20425v2](http://arxiv.org/pdf/2503.20425v2)**

> **作者:** Kevin Alcedo; Pedro U. Lima; Rachid Alami
>
> **备注:** Accepted as a regular paper at the 2025 IEEE International Conference on Robot & Human Interactive Communication (RO-MAN). \c{opyright} 2025 IEEE. The final version will appear in IEEE Xplore (DOI TBD)
>
> **摘要:** Navigating in environments alongside humans requires agents to reason under uncertainty and account for the beliefs and intentions of those around them. Under a sequential decision-making framework, egocentric navigation can naturally be represented as a Markov Decision Process (MDP). However, social navigation additionally requires reasoning about the hidden beliefs of others, inherently leading to a Partially Observable Markov Decision Process (POMDP), where agents lack direct access to others' mental states. Inspired by Theory of Mind and Epistemic Planning, we propose (1) a neuro-symbolic model-based reinforcement learning architecture for social navigation, addressing the challenge of belief tracking in partially observable environments; and (2) a perspective-shift operator for belief estimation, leveraging recent work on Influence-based Abstractions (IBA) in structured multi-agent settings.
>
---
#### [replaced 025] Help or Hindrance: Understanding the Impact of Robot Communication in Action Teams
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08892v3](http://arxiv.org/pdf/2506.08892v3)**

> **作者:** Tauhid Tanjim; Jonathan St. George; Kevin Ching; Angelique Taylor
>
> **备注:** This is the author's original submitted version of the paper accepted to the 2025 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). \c{opyright} 2025 IEEE. Personal use of this material is permitted. For any other use, please contact IEEE
>
> **摘要:** The human-robot interaction (HRI) field has recognized the importance of enabling robots to interact with teams. Human teams rely on effective communication for successful collaboration in time-sensitive environments. Robots can play a role in enhancing team coordination through real-time assistance. Despite significant progress in human-robot teaming research, there remains an essential gap in how robots can effectively communicate with action teams using multimodal interaction cues in time-sensitive environments. This study addresses this knowledge gap in an experimental in-lab study to investigate how multimodal robot communication in action teams affects workload and human perception of robots. We explore team collaboration in a medical training scenario where a robotic crash cart (RCC) provides verbal and non-verbal cues to help users remember to perform iterative tasks and search for supplies. Our findings show that verbal cues for object search tasks and visual cues for task reminders reduce team workload and increase perceived ease of use and perceived usefulness more effectively than a robot with no feedback. Our work contributes to multimodal interaction research in the HRI field, highlighting the need for more human-robot teaming research to understand best practices for integrating collaborative robots in time-sensitive environments such as in hospitals, search and rescue, and manufacturing applications.
>
---
