# 机器人 cs.RO

- **最新发布 21 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] Autonomous UAV Navigation for Search and Rescue Missions Using Computer Vision and Convolutional Neural Networks
- **分类: cs.RO**

- **简介: 该论文属于无人机自主导航任务，旨在解决搜救任务中人员探测、识别与跟踪问题。论文实现了基于计算机视觉与CNN的无人机系统，集成ROS2框架和PD控制器，实现对个体的实时检测、人脸识别与跟踪，提升了搜救效率与自动化水平。**

- **链接: [http://arxiv.org/pdf/2507.18160v1](http://arxiv.org/pdf/2507.18160v1)**

> **作者:** Luka Šiktar; Branimir Ćaran; Bojan Šekoranja; Marko Švaco
>
> **备注:** The paper is accepted and presented on the 34th International Conference on Robotics in Alpe-Adria-Danube Region, RAAD 2025, Belgrade Serbia
>
> **摘要:** In this paper, we present a subsystem, using Unmanned Aerial Vehicles (UAV), for search and rescue missions, focusing on people detection, face recognition and tracking of identified individuals. The proposed solution integrates a UAV with ROS2 framework, that utilizes multiple convolutional neural networks (CNN) for search missions. System identification and PD controller deployment are performed for autonomous UAV navigation. The ROS2 environment utilizes the YOLOv11 and YOLOv11-pose CNNs for tracking purposes, and the dlib library CNN for face recognition. The system detects a specific individual, performs face recognition and starts tracking. If the individual is not yet known, the UAV operator can manually locate the person, save their facial image and immediately initiate the tracking process. The tracking process relies on specific keypoints identified on the human body using the YOLOv11-pose CNN model. These keypoints are used to track a specific individual and maintain a safe distance. To enhance accurate tracking, system identification is performed, based on measurement data from the UAVs IMU. The identified system parameters are used to design PD controllers that utilize YOLOv11-pose to estimate the distance between the UAVs camera and the identified individual. The initial experiments, conducted on 14 known individuals, demonstrated that the proposed subsystem can be successfully used in real time. The next step involves implementing the system on a large experimental UAV for field use and integrating autonomous navigation with GPS-guided control for rescue operations planning.
>
---
#### [new 002] ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决语义多样环境下操作任务理解与执行的统一问题。现有方法存在语义粒度粗、缺乏实时闭环规划、鲁棒性差等问题。论文提出ReSem3D框架，结合视觉基础模型和多模态大语言模型，通过分层递归推理实现细粒度语义定位，动态构建3D空间约束，提升操作的适应性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2507.18262v1](http://arxiv.org/pdf/2507.18262v1)**

> **作者:** Chenyu Su; Weiwei Shang; Chen Qian; Fei Zhang; Shuang Cong
>
> **备注:** 12 pages,9 figures
>
> **摘要:** Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos at https://resem3d.github.io.
>
---
#### [new 003] G2S-ICP SLAM: Geometry-aware Gaussian Splatting ICP SLAM
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在提升RGB-D相机的实时三维重建与定位精度。论文提出G2S-ICP SLAM，采用几何感知的高斯投影与各向异性协方差先验，增强多视角一致性与几何建模，实现高保真实时SLAM。**

- **链接: [http://arxiv.org/pdf/2507.18344v1](http://arxiv.org/pdf/2507.18344v1)**

> **作者:** Gyuhyeon Pak; Hae Min Cho; Euntai Kim
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** In this paper, we present a novel geometry-aware RGB-D Gaussian Splatting SLAM system, named G2S-ICP SLAM. The proposed method performs high-fidelity 3D reconstruction and robust camera pose tracking in real-time by representing each scene element using a Gaussian distribution constrained to the local tangent plane. This effectively models the local surface as a 2D Gaussian disk aligned with the underlying geometry, leading to more consistent depth interpretation across multiple viewpoints compared to conventional 3D ellipsoid-based representations with isotropic uncertainty. To integrate this representation into the SLAM pipeline, we embed the surface-aligned Gaussian disks into a Generalized ICP framework by introducing anisotropic covariance prior without altering the underlying registration formulation. Furthermore we propose a geometry-aware loss that supervises photometric, depth, and normal consistency. Our system achieves real-time operation while preserving both visual and geometric fidelity. Extensive experiments on the Replica and TUM-RGBD datasets demonstrate that G2S-ICP SLAM outperforms prior SLAM systems in terms of localization accuracy, reconstruction completeness, while maintaining the rendering quality.
>
---
#### [new 004] Evaluation of facial landmark localization performance in a surgical setting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于计算机视觉与医疗技术交叉任务，旨在解决手术环境中面部关键点定位准确性问题。论文通过控制实验，评估MediaPipe算法在不同姿态和光照条件下的检测效果，验证其在固定手术光照下的性能提升，探讨其在医疗场景中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2507.18248v1](http://arxiv.org/pdf/2507.18248v1)**

> **作者:** Ines Frajtag; Marko Švaco; Filip Šuligoj
>
> **摘要:** The use of robotics, computer vision, and their applications is becoming increasingly widespread in various fields, including medicine. Many face detection algorithms have found applications in neurosurgery, ophthalmology, and plastic surgery. A common challenge in using these algorithms is variable lighting conditions and the flexibility of detection positions to identify and precisely localize patients. The proposed experiment tests the MediaPipe algorithm for detecting facial landmarks in a controlled setting, using a robotic arm that automatically adjusts positions while the surgical light and the phantom remain in a fixed position. The results of this study demonstrate that the improved accuracy of facial landmark detection under surgical lighting significantly enhances the detection performance at larger yaw and pitch angles. The increase in standard deviation/dispersion occurs due to imprecise detection of selected facial landmarks. This analysis allows for a discussion on the potential integration of the MediaPipe algorithm into medical procedures.
>
---
#### [new 005] A Step-by-step Guide on Nonlinear Model Predictive Control for Safe Mobile Robot Navigation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于移动机器人导航任务，旨在解决在障碍物环境中安全导航的问题。通过非线性模型预测控制（NMPC）方法，逐步实现状态和输入约束下的避障与鲁棒性控制，强调安全性和性能保证，为研究者提供从理论到实现的实用指南。**

- **链接: [http://arxiv.org/pdf/2507.17856v1](http://arxiv.org/pdf/2507.17856v1)**

> **作者:** Dennis Benders; Laura Ferranti; Johannes Köhler
>
> **备注:** 51 pages, 3 figures
>
> **摘要:** Designing a Model Predictive Control (MPC) scheme that enables a mobile robot to safely navigate through an obstacle-filled environment is a complicated yet essential task in robotics. In this technical report, safety refers to ensuring that the robot respects state and input constraints while avoiding collisions with obstacles despite the presence of disturbances and measurement noise. This report offers a step-by-step approach to implementing Nonlinear Model Predictive Control (NMPC) schemes addressing these safety requirements. Numerous books and survey papers provide comprehensive overviews of linear MPC (LMPC) \cite{bemporad2007robust,kouvaritakis2016model}, NMPC \cite{rawlings2017model,allgower2004nonlinear,mayne2014model,grune2017nonlinear,saltik2018outlook}, and their applications in various domains, including robotics \cite{nascimento2018nonholonomic,nguyen2021model,shi2021advanced,wei2022mpc}. This report does not aim to replicate those exhaustive reviews. Instead, it focuses specifically on NMPC as a foundation for safe mobile robot navigation. The goal is to provide a practical and accessible path from theoretical concepts to mathematical proofs and implementation, emphasizing safety and performance guarantees. It is intended for researchers, robotics engineers, and practitioners seeking to bridge the gap between theoretical NMPC formulations and real-world robotic applications. This report is not necessarily meant to remain fixed over time. If someone finds an error in the presented theory, please reach out via the given email addresses. We are happy to update the document if necessary.
>
---
#### [new 006] Residual Koopman Model Predictive Control for Enhanced Vehicle Dynamics with Small On-Track Data Input
- **分类: cs.RO**

- **简介: 论文属于车辆轨迹跟踪控制任务，旨在提升自动驾驶车辆的跟踪精度与稳定性。现有方法如纯追踪控制忽略车辆模型约束，影响安全性；传统模型预测控制依赖准确建模，但难以兼顾非线性动态与计算效率。论文提出残差Koopman模型预测控制（RKMPC），结合线性MPC与神经网络残差模型，实现更优控制。实验表明其在少量数据下优于传统方法。**

- **链接: [http://arxiv.org/pdf/2507.18396v1](http://arxiv.org/pdf/2507.18396v1)**

> **作者:** Yonghao Fu; Cheng Hu; Haokun Xiong; Zhangpeng Bao; Wenyuan Du; Edoardo Ghignone; Michele Magno; Lei Xie; Hongye Su
>
> **摘要:** In vehicle trajectory tracking tasks, the simplest approach is the Pure Pursuit (PP) Control. However, this single-point preview tracking strategy fails to consider vehicle model constraints, compromising driving safety. Model Predictive Control (MPC) as a widely adopted control method, optimizes control actions by incorporating mechanistic models and physical constraints. While its control performance critically depends on the accuracy of vehicle modeling. Traditional vehicle modeling approaches face inherent trade-offs between capturing nonlinear dynamics and maintaining computational efficiency, often resulting in reduced control performance. To address these challenges, this paper proposes Residual Koopman Model Predictive Control (RKMPC) framework. This method uses two linear MPC architecture to calculate control inputs: a Linear Model Predictive Control (LMPC) computes the baseline control input based on the vehicle kinematic model, and a neural network-based RKMPC calculates the compensation input. The final control command is obtained by adding these two components. This design preserves the reliability and interpretability of traditional mechanistic model while achieving performance optimization through residual modeling. This method has been validated on the Carsim-Matlab joint simulation platform and a physical 1:10 scale F1TENTH racing car. Experimental results show that RKMPC requires only 20% of the training data needed by traditional Koopman Model Predictive Control (KMPC) while delivering superior tracking performance. Compared to traditional LMPC, RKMPC reduces lateral error by 11.7%-22.1%, decreases heading error by 8.9%-15.8%, and improves front-wheel steering stability by up to 27.6%. The implementation code is available at: https://github.com/ZJU-DDRX/Residual Koopman.
>
---
#### [new 007] Adaptive Articulated Object Manipulation On The Fly with Foundation Model Reasoning and Part Grounding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决真实场景中多类联结物体操作的泛化问题。现有方法因物体几何多样性和功能差异难以适应。论文提出AdaRPG框架，利用基础模型提取具有局部几何相似性的物体部件，提升功能技能的视觉可操作性泛化，并通过部件功能推理生成高层控制指令，实现跨新联结物体类别的强泛化操作。**

- **链接: [http://arxiv.org/pdf/2507.18276v1](http://arxiv.org/pdf/2507.18276v1)**

> **作者:** Xiaojie Zhang; Yuanfei Wang; Ruihai Wu; Kunqi Xu; Yu Li; Liuyu Xiang; Hao Dong; Zhaofeng He
>
> **备注:** ICCV 2025
>
> **摘要:** Articulated objects pose diverse manipulation challenges for robots. Since their internal structures are not directly observable, robots must adaptively explore and refine actions to generate successful manipulation trajectories. While existing works have attempted cross-category generalization in adaptive articulated object manipulation, two major challenges persist: (1) the geometric diversity of real-world articulated objects complicates visual perception and understanding, and (2) variations in object functions and mechanisms hinder the development of a unified adaptive manipulation strategy. To address these challenges, we propose AdaRPG, a novel framework that leverages foundation models to extract object parts, which exhibit greater local geometric similarity than entire objects, thereby enhancing visual affordance generalization for functional primitive skills. To support this, we construct a part-level affordance annotation dataset to train the affordance model. Additionally, AdaRPG utilizes the common knowledge embedded in foundation models to reason about complex mechanisms and generate high-level control codes that invoke primitive skill functions based on part affordance inference. Simulation and real-world experiments demonstrate AdaRPG's strong generalization ability across novel articulated object categories.
>
---
#### [new 008] Evaluating the Pre-Dressing Step: Unfolding Medical Garments Via Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决医疗场景中折叠状态的衣物需展开后才能进行机器人辅助穿戴的问题。作者通过模仿学习训练三种操作动作，结合视觉分类器判断衣物展开状态，实验验证了动作组合对展开衣物的有效性。**

- **链接: [http://arxiv.org/pdf/2507.18436v1](http://arxiv.org/pdf/2507.18436v1)**

> **作者:** David Blanco-Mulero; Júlia Borràs; Carme Torras
>
> **备注:** 6 pages, 4 figures, 2 tables. Accepted to IEEE/RSJ IROS 2025. Project website: https://sites.google.com/view/pre-dressing
>
> **摘要:** Robotic-assisted dressing has the potential to significantly aid both patients as well as healthcare personnel, reducing the workload and improving the efficiency in clinical settings. While substantial progress has been made in robotic dressing assistance, prior works typically assume that garments are already unfolded and ready for use. However, in medical applications gowns and aprons are often stored in a folded configuration, requiring an additional unfolding step. In this paper, we introduce the pre-dressing step, the process of unfolding garments prior to assisted dressing. We leverage imitation learning for learning three manipulation primitives, including both high and low acceleration motions. In addition, we employ a visual classifier to categorise the garment state as closed, partly opened, and fully opened. We conduct an empirical evaluation of the learned manipulation primitives as well as their combinations. Our results show that highly dynamic motions are not effective for unfolding freshly unpacked garments, where the combination of motions can efficiently enhance the opening configuration.
>
---
#### [new 009] MoRPI-PINN: A Physics-Informed Framework for Mobile Robot Pure Inertial Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于移动机器人导航任务，旨在解决纯惯性导航中误差漂移的问题。通过设计蛇形运动结合物理信息神经网络（MoRPI-PINN），提升惯性信号质量，实现高精度定位。论文提出了一种轻量级框架，适用于边缘设备，提高了导航的准确性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.18206v1](http://arxiv.org/pdf/2507.18206v1)**

> **作者:** Arup Kumar Sahoo; Itzik Klein
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** A fundamental requirement for full autonomy in mobile robots is accurate navigation even in situations where satellite navigation or cameras are unavailable. In such practical situations, relying only on inertial sensors will result in navigation solution drift due to the sensors' inherent noise and error terms. One of the emerging solutions to mitigate drift is to maneuver the robot in a snake-like slithering motion to increase the inertial signal-to-noise ratio, allowing the regression of the mobile robot position. In this work, we propose MoRPI-PINN as a physics-informed neural network framework for accurate inertial-based mobile robot navigation. By embedding physical laws and constraints into the training process, MoRPI-PINN is capable of providing an accurate and robust navigation solution. Using real-world experiments, we show accuracy improvements of over 85% compared to other approaches. MoRPI-PINN is a lightweight approach that can be implemented even on edge devices and used in any typical mobile robot application.
>
---
#### [new 010] Modular Robot and Landmark Localisation Using Relative Bearing Measurements
- **分类: cs.RO; cs.SY; eess.SP; eess.SY**

- **简介: 论文研究模块化非线性最小二乘滤波方法，用于由多个独立子系统组成的系统状态估计，特别应用于机器人与路标定位问题。通过结合协方差交集（CI）算法避免信息重复计算，提升估计精度。论文通过仿真对比验证了该方法在通信受限下的性能表现。**

- **链接: [http://arxiv.org/pdf/2507.18070v1](http://arxiv.org/pdf/2507.18070v1)**

> **作者:** Behzad Zamani; Jochen Trumpf; Chris Manzie
>
> **备注:** Submitted to RA-L
>
> **摘要:** In this paper we propose a modular nonlinear least squares filtering approach for systems composed of independent subsystems. The state and error covariance estimate of each subsystem is updated independently, even when a relative measurement simultaneously depends on the states of multiple subsystems. We integrate the Covariance Intersection (CI) algorithm as part of our solution in order to prevent double counting of information when subsystems share estimates with each other. An alternative derivation of the CI algorithm based on least squares estimation makes this integration possible. We particularise the proposed approach to the robot-landmark localization problem. In this problem, noisy measurements of the bearing angle to a stationary landmark position measured relative to the SE(2) pose of a moving robot couple the estimation problems for the robot pose and the landmark position. In a randomized simulation study, we benchmark the proposed modular method against a monolithic joint state filter to elucidate their respective trade-offs. In this study we also include variants of the proposed method that achieve a graceful degradation of performance with reduced communication and bandwidth requirements.
>
---
#### [new 011] A Modular Residual Learning Framework to Enhance Model-Based Approach for Robust Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决模型不确定性导致的控制性能下降问题。作者提出一种模块化残差学习框架，结合模型驱动与学习方法，提升足式机器人在复杂环境中的运动鲁棒性与控制效率。**

- **链接: [http://arxiv.org/pdf/2507.18138v1](http://arxiv.org/pdf/2507.18138v1)**

> **作者:** Min-Gyu Kim; Dongyun Kang; Hajun Kim; Hae-Won Park
>
> **备注:** 8 pages, IEEE RA-L accepted (July 2025)
>
> **摘要:** This paper presents a novel approach that combines the advantages of both model-based and learning-based frameworks to achieve robust locomotion. The residual modules are integrated with each corresponding part of the model-based framework, a footstep planner and dynamic model designed using heuristics, to complement performance degradation caused by a model mismatch. By utilizing a modular structure and selecting the appropriate learning-based method for each residual module, our framework demonstrates improved control performance in environments with high uncertainty, while also achieving higher learning efficiency compared to baseline methods. Moreover, we observed that our proposed methodology not only enhances control performance but also provides additional benefits, such as making nominal controllers more robust to parameter tuning. To investigate the feasibility of our framework, we demonstrated residual modules combined with model predictive control in a real quadrupedal robot. Despite uncertainties beyond the simulation, the robot successfully maintains balance and tracks the commanded velocity.
>
---
#### [new 012] Experimental Comparison of Whole-Body Control Formulations for Humanoid Robots in Task Acceleration and Task Force Spaces
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在比较两种全身控制方法在动态任务中的表现。论文通过实验对比了基于逆动力学和基于无源性的全身控制器在足部运动、负重下蹲和跳跃任务中的性能，分析其在非理想条件下的鲁棒性差异。**

- **链接: [http://arxiv.org/pdf/2507.18502v1](http://arxiv.org/pdf/2507.18502v1)**

> **作者:** Sait Sovukluk; Grazia Zambella; Tobias Egle; Christian Ott
>
> **备注:** This paper has been accepted for publication in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025). - Link to video: https://youtu.be/Nfm50ycz-FU
>
> **摘要:** This paper studies the experimental comparison of two different whole-body control formulations for humanoid robots: inverse dynamics whole-body control (ID-WBC) and passivity-based whole-body control (PB-WBC). The two controllers fundamentally differ from each other as the first is formulated in task acceleration space and the latter is in task force space with passivity considerations. Even though both control methods predict stability under ideal conditions in closed-loop dynamics, their robustness against joint friction, sensor noise, unmodeled external disturbances, and non-perfect contact conditions is not evident. Therefore, we analyze and experimentally compare the two controllers on a humanoid robot platform through swing foot position and orientation control, squatting with and without unmodeled additional weights, and jumping. We also relate the observed performance and characteristic differences with the controller formulations and highlight each controller's advantages and disadvantages.
>
---
#### [new 013] PinchBot: Long-Horizon Deformable Manipulation with Guided Diffusion Policy
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决基于捏压动作的柔性物体长期多步态形变控制问题。作者提出PinchBot系统，结合扩散策略模型、3D点云嵌入、任务进度预测和碰撞约束动作投影，实现简单陶器制作目标。**

- **链接: [http://arxiv.org/pdf/2507.17846v1](http://arxiv.org/pdf/2507.17846v1)**

> **作者:** Alison Bartsch; Arvind Car; Amir Barati Farimani
>
> **摘要:** Pottery creation is a complicated art form that requires dexterous, precise and delicate actions to slowly morph a block of clay to a meaningful, and often useful 3D goal shape. In this work, we aim to create a robotic system that can create simple pottery goals with only pinch-based actions. This pinch pottery task allows us to explore the challenges of a highly multi-modal and long-horizon deformable manipulation task. To this end, we present PinchBot, a goal-conditioned diffusion policy model that when combined with pre-trained 3D point cloud embeddings, task progress prediction and collision-constrained action projection, is able to successfully create a variety of simple pottery goals. For experimental videos and access to the demonstration dataset, please visit our project website: https://sites.google.com/andrew.cmu.edu/pinchbot/home.
>
---
#### [new 014] AF-RLIO: Adaptive Fusion of Radar-LiDAR-Inertial Information for Robust Odometry in Challenging Environments
- **分类: cs.RO**

- **简介: 论文提出AF-RLIO，属于机器人导航中的鲁棒里程计任务，旨在解决复杂动态环境中单一传感器失效导致的定位不准问题。工作包括多传感器自适应融合、动态点去除、紧耦合IMU与优化定位。**

- **链接: [http://arxiv.org/pdf/2507.18317v1](http://arxiv.org/pdf/2507.18317v1)**

> **作者:** Chenglong Qian; Yang Xu; Xiufang Shi; Jiming Chen; Liang Li
>
> **摘要:** In robotic navigation, maintaining precise pose estimation and navigation in complex and dynamic environments is crucial. However, environmental challenges such as smoke, tunnels, and adverse weather can significantly degrade the performance of single-sensor systems like LiDAR or GPS, compromising the overall stability and safety of autonomous robots. To address these challenges, we propose AF-RLIO: an adaptive fusion approach that integrates 4D millimeter-wave radar, LiDAR, inertial measurement unit (IMU), and GPS to leverage the complementary strengths of these sensors for robust odometry estimation in complex environments. Our method consists of three key modules. Firstly, the pre-processing module utilizes radar data to assist LiDAR in removing dynamic points and determining when environmental conditions are degraded for LiDAR. Secondly, the dynamic-aware multimodal odometry selects appropriate point cloud data for scan-to-map matching and tightly couples it with the IMU using the Iterative Error State Kalman Filter. Lastly, the factor graph optimization module balances weights between odometry and GPS data, constructing a pose graph for optimization. The proposed approach has been evaluated on datasets and tested in real-world robotic environments, demonstrating its effectiveness and advantages over existing methods in challenging conditions such as smoke and tunnels.
>
---
#### [new 015] OpenNav: Open-World Navigation with Multimodal Large Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人导航任务，旨在解决开放世界中复杂语言指令到机器人动作的映射问题。利用多模态大语言模型（MLLMs）理解自由形式语言，并生成导航轨迹。通过与视觉语言模型交互生成价值图，结合语义与空间信息，提升导航鲁棒性与适用性。**

- **链接: [http://arxiv.org/pdf/2507.18033v1](http://arxiv.org/pdf/2507.18033v1)**

> **作者:** Mingfeng Yuan; Letian Wang; Steven L. Waslander
>
> **摘要:** Pre-trained large language models (LLMs) have demonstrated strong common-sense reasoning abilities, making them promising for robotic navigation and planning tasks. However, despite recent progress, bridging the gap between language descriptions and actual robot actions in the open-world, beyond merely invoking limited predefined motion primitives, remains an open challenge. In this work, we aim to enable robots to interpret and decompose complex language instructions, ultimately synthesizing a sequence of trajectory points to complete diverse navigation tasks given open-set instructions and open-set objects. We observe that multi-modal large language models (MLLMs) exhibit strong cross-modal understanding when processing free-form language instructions, demonstrating robust scene comprehension. More importantly, leveraging their code-generation capability, MLLMs can interact with vision-language perception models to generate compositional 2D bird-eye-view value maps, effectively integrating semantic knowledge from MLLMs with spatial information from maps to reinforce the robot's spatial understanding. To further validate our approach, we effectively leverage large-scale autonomous vehicle datasets (AVDs) to validate our proposed zero-shot vision-language navigation framework in outdoor navigation tasks, demonstrating its capability to execute a diverse range of free-form natural language navigation instructions while maintaining robustness against object detection errors and linguistic ambiguities. Furthermore, we validate our system on a Husky robot in both indoor and outdoor scenes, demonstrating its real-world robustness and applicability. Supplementary videos are available at https://trailab.github.io/OpenNav-website/
>
---
#### [new 016] A Novel Monte-Carlo Compressed Sensing and Dictionary Learning Method for the Efficient Path Planning of Remote Sensing Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划与环境监测任务，旨在解决如何高效采集环境数据并精确重建污染地图的问题。论文提出了一种结合蒙特卡洛压缩感知与字典学习的新方法，优化机器人采样路径，减少行走距离和信号重构误差，实验验证了其在重建NO₂污染地图中的高效性。**

- **链接: [http://arxiv.org/pdf/2507.18462v1](http://arxiv.org/pdf/2507.18462v1)**

> **作者:** Alghalya Al-Hajri; Ejmen Al-Ubejdij; Aiman Erbad; Ali Safa
>
> **摘要:** In recent years, Compressed Sensing (CS) has gained significant interest as a technique for acquiring high-resolution sensory data using fewer measurements than traditional Nyquist sampling requires. At the same time, autonomous robotic platforms such as drones and rovers have become increasingly popular tools for remote sensing and environmental monitoring tasks, including measurements of temperature, humidity, and air quality. Within this context, this paper presents, to the best of our knowledge, the first investigation into how the structure of CS measurement matrices can be exploited to design optimized sampling trajectories for robotic environmental data collection. We propose a novel Monte Carlo optimization framework that generates measurement matrices designed to minimize both the robot's traversal path length and the signal reconstruction error within the CS framework. Central to our approach is the application of Dictionary Learning (DL) to obtain a data-driven sparsifying transform, which enhances reconstruction accuracy while further reducing the number of samples that the robot needs to collect. We demonstrate the effectiveness of our method through experiments reconstructing $NO_2$ pollution maps over the Gulf region. The results indicate that our approach can reduce robot travel distance to less than $10\%$ of a full-coverage path, while improving reconstruction accuracy by over a factor of five compared to traditional CS methods based on DCT and polynomial dictionaries, as well as by a factor of two compared to previously-proposed Informative Path Planning (IPP) methods.
>
---
#### [new 017] Automated Brake Onset Detection in Naturalistic Driving Data
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于自动驾驶系统评估任务，旨在解决在缺乏车辆控制信号时，自动检测刹车开始时间的问题。作者提出了一种基于分段线性加速度模型的算法，可自动估计刹车 onset，并通过与人工标注结果对比验证其准确性，从而实现对自动驾驶系统和人类驾驶响应时间的客观评估。**

- **链接: [http://arxiv.org/pdf/2507.17943v1](http://arxiv.org/pdf/2507.17943v1)**

> **作者:** Shu-Yuan Liu; Johan Engström; Gustav Markkula
>
> **摘要:** Response timing measures play a crucial role in the assessment of automated driving systems (ADS) in collision avoidance scenarios, including but not limited to establishing human benchmarks and comparing ADS to human driver response performance. For example, measuring the response time (of a human driver or ADS) to a conflict requires the determination of a stimulus onset and a response onset. In existing studies, response onset relies on manual annotation or vehicle control signals such as accelerator and brake pedal movements. These methods are not applicable when analyzing large scale data where vehicle control signals are not available. This holds in particular for the rapidly expanding sets of ADS log data where the behavior of surrounding road users is observed via onboard sensors. To advance evaluation techniques for ADS and enable measuring response timing when vehicle control signals are not available, we developed a simple and efficient algorithm, based on a piecewise linear acceleration model, to automatically estimate brake onset that can be applied to any type of driving data that includes vehicle longitudinal time series data. We also proposed a manual annotation method to identify brake onset and used it as ground truth for validation. R2 was used as a confidence metric to measure the accuracy of the algorithm, and its classification performance was analyzed using naturalistic collision avoidance data of both ADS and humans, where our method was validated against human manual annotation. Although our algorithm is subject to certain limitations, it is efficient, generalizable, applicable to any road user and scenario types, and is highly configurable.
>
---
#### [new 018] Rapid Modeling Architecture for Lightweight Simulator to Accelerate and Improve Decision Making for Industrial Systems
- **分类: eess.SY; cs.MA; cs.RO; cs.SY**

- **简介: 该论文属于工业系统设计任务，旨在解决传统仿真建模耗时长、影响早期决策的问题。作者提出了一种轻量级快速建模架构（RMA），并开发了相应仿真工具，显著缩短建模时间，提升决策效率。实验表明，建模时间减少了78.3%。**

- **链接: [http://arxiv.org/pdf/2507.17990v1](http://arxiv.org/pdf/2507.17990v1)**

> **作者:** Takumi Kato; Zhi Li Hu
>
> **备注:** 8 pages, 13 figures. Manuscript accepted at the 2025 IEEE 21st International Conference on Automation Science and Engineering (CASE 2025)
>
> **摘要:** Designing industrial systems, such as building, improving, and automating distribution centers and manufacturing plants, involves critical decision-making with limited information in the early phases. The lack of information leads to less accurate designs of the systems, which are often difficult to resolve later. It is effective to use simulators to model the designed system and find out the issues early. However, the modeling time required by conventional simulators is too long to allow for rapid model creation to meet decision-making demands. In this paper, we propose a Rapid Modeling Architecture (RMA) for a lightweight industrial simulator that mitigates the modeling burden while maintaining the essential details in order to accelerate and improve decision-making. We have prototyped a simulator based on the RMA and applied it to the actual factory layout design problem. We also compared the modeling time of our simulator to that of an existing simulator, and as a result, our simulator achieved a 78.3% reduction in modeling time compared to conventional simulators.
>
---
#### [new 019] DSFormer: A Dual-Scale Cross-Learning Transformer for Visual Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉地点识别（VPR）任务，旨在解决环境和视角变化下识别不准确的问题。作者提出DSFormer模型，通过双尺度特征交互和跨尺度学习增强特征表示，并采用新的数据聚类策略优化训练集，提升识别鲁棒性和计算效率。**

- **链接: [http://arxiv.org/pdf/2507.18444v1](http://arxiv.org/pdf/2507.18444v1)**

> **作者:** Haiyang Jiang; Songhao Piao; Chao Gao; Lei Yu; Liguo Chen
>
> **摘要:** Visual Place Recognition (VPR) is crucial for robust mobile robot localization, yet it faces significant challenges in maintaining reliable performance under varying environmental conditions and viewpoints. To address this, we propose a novel framework that integrates Dual-Scale-Former (DSFormer), a Transformer-based cross-learning module, with an innovative block clustering strategy. DSFormer enhances feature representation by enabling bidirectional information transfer between dual-scale features extracted from the final two CNN layers, capturing both semantic richness and spatial details through self-attention for long-range dependencies within each scale and shared cross-attention for cross-scale learning. Complementing this, our block clustering strategy repartitions the widely used San Francisco eXtra Large (SF-XL) training dataset from multiple distinct perspectives, optimizing data organization to further bolster robustness against viewpoint variations. Together, these innovations not only yield a robust global embedding adaptable to environmental changes but also reduce the required training data volume by approximately 30\% compared to previous partitioning methods. Comprehensive experiments demonstrate that our approach achieves state-of-the-art performance across most benchmark datasets, surpassing advanced reranking methods like DELG, Patch-NetVLAD, TransVPR, and R2Former as a global retrieval solution using 512-dim global descriptors, while significantly improving computational efficiency.
>
---
#### [new 020] ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense
- **分类: cs.LG; cs.CR; cs.RO**

- **简介: 该论文属于多智能体协同防御任务，旨在解决无人水面艇（USV）拦截高机动性攻击者以保护目标区域的问题。作者提出ARBoids框架，结合残差强化学习与仿生Boids模型，实现高效协同防御，并通过仿真验证其优于传统方法，具备强适应性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2502.18549v2](http://arxiv.org/pdf/2502.18549v2)**

> **作者:** Jiyue Tao; Tongsheng Shen; Dexin Zhao; Feitian Zhang
>
> **摘要:** The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter.
>
---
#### [new 021] FishDet-M: A Unified Large-Scale Benchmark for Robust Fish Detection and CLIP-Guided Model Selection in Diverse Aquatic Visual Domains
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决水下鱼类检测中数据碎片化、成像条件多样和评估标准不统一的问题。论文构建了大规模统一基准 FishDet-M，整合13个公开数据集，并系统评估28种检测模型。同时提出基于CLIP的零样本模型选择框架，提升跨域检测的适应性和效率。**

- **链接: [http://arxiv.org/pdf/2507.17859v1](http://arxiv.org/pdf/2507.17859v1)**

> **作者:** Muayad Abujabal; Lyes Saad Saoud; Irfan Hussain
>
> **摘要:** Accurate fish detection in underwater imagery is essential for ecological monitoring, aquaculture automation, and robotic perception. However, practical deployment remains limited by fragmented datasets, heterogeneous imaging conditions, and inconsistent evaluation protocols. To address these gaps, we present \textit{FishDet-M}, the largest unified benchmark for fish detection, comprising 13 publicly available datasets spanning diverse aquatic environments including marine, brackish, occluded, and aquarium scenes. All data are harmonized using COCO-style annotations with both bounding boxes and segmentation masks, enabling consistent and scalable cross-domain evaluation. We systematically benchmark 28 contemporary object detection models, covering the YOLOv8 to YOLOv12 series, R-CNN based detectors, and DETR based models. Evaluations are conducted using standard metrics including mAP, mAP@50, and mAP@75, along with scale-specific analyses (AP$_S$, AP$_M$, AP$_L$) and inference profiling in terms of latency and parameter count. The results highlight the varying detection performance across models trained on FishDet-M, as well as the trade-off between accuracy and efficiency across models of different architectures. To support adaptive deployment, we introduce a CLIP-based model selection framework that leverages vision-language alignment to dynamically identify the most semantically appropriate detector for each input image. This zero-shot selection strategy achieves high performance without requiring ensemble computation, offering a scalable solution for real-time applications. FishDet-M establishes a standardized and reproducible platform for evaluating object detection in complex aquatic scenes. All datasets, pretrained models, and evaluation tools are publicly available to facilitate future research in underwater computer vision and intelligent marine systems.
>
---
## 更新

#### [replaced 001] Adaptive Relative Pose Estimation Framework with Dual Noise Tuning for Safe Approaching Maneuvers
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.16214v2](http://arxiv.org/pdf/2507.16214v2)**

> **作者:** Batu Candan; Simone Servadio
>
> **摘要:** Accurate and robust relative pose estimation is crucial for enabling challenging Active Debris Removal (ADR) missions targeting tumbling derelict satellites such as ESA's ENVISAT. This work presents a complete pipeline integrating advanced computer vision techniques with adaptive nonlinear filtering to address this challenge. A Convolutional Neural Network (CNN), enhanced with image preprocessing, detects structural markers (corners) from chaser imagery, whose 2D coordinates are converted to 3D measurements using camera modeling. These measurements are fused within an Unscented Kalman Filter (UKF) framework, selected for its ability to handle nonlinear relative dynamics, to estimate the full relative pose. Key contributions include the integrated system architecture and a dual adaptive strategy within the UKF: dynamic tuning of the measurement noise covariance compensates for varying CNN measurement uncertainty, while adaptive tuning of the process noise covariance, utilizing measurement residual analysis, accounts for unmodeled dynamics or maneuvers online. This dual adaptation enhances robustness against both measurement imperfections and dynamic model uncertainties. The performance of the proposed adaptive integrated system is evaluated through high-fidelity simulations using a realistic ENVISAT model, comparing estimates against ground truth under various conditions, including measurement outages. This comprehensive approach offers an enhanced solution for robust onboard relative navigation, significantly advancing the capabilities required for safe proximity operations during ADR missions.
>
---
#### [replaced 002] Hand Gesture Recognition for Collaborative Robots Using Lightweight Deep Learning in Real-Time Robotic Systems
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.10055v2](http://arxiv.org/pdf/2507.10055v2)**

> **作者:** Muhtadin; I Wayan Agus Darmawan; Muhammad Hilmi Rusydiansyah; I Ketut Eddy Purnama; Chastine Fatichah; Mauridhi Hery Purnomo
>
> **摘要:** Direct and natural interaction is essential for intuitive human-robot collaboration, eliminating the need for additional devices such as joysticks, tablets, or wearable sensors. In this paper, we present a lightweight deep learning-based hand gesture recognition system that enables humans to control collaborative robots naturally and efficiently. This model recognizes eight distinct hand gestures with only 1,103 parameters and a compact size of 22 KB, achieving an accuracy of 93.5%. To further optimize the model for real-world deployment on edge devices, we applied quantization and pruning using TensorFlow Lite, reducing the final model size to just 7 KB. The system was successfully implemented and tested on a Universal Robot UR5 collaborative robot within a real-time robotic framework based on ROS2. The results demonstrate that even extremely lightweight models can deliver accurate and responsive hand gesture-based control for collaborative robots, opening new possibilities for natural human-robot interaction in constrained environments.
>
---
#### [replaced 003] Fast Bilateral Teleoperation and Imitation Learning Using Sensorless Force Control via Accurate Dynamics Model
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.06174v5](http://arxiv.org/pdf/2507.06174v5)**

> **作者:** Koki Yamane; Yunhan Li; Masashi Konosu; Koki Inami; Junji Oaki; Sho Sakaino; Toshiaki Tsuji
>
> **备注:** 20 pages, 9 figures, Submitted to CoRL 2025
>
> **摘要:** In recent years, the advancement of imitation learning has led to increased interest in teleoperating low-cost manipulators to collect demonstration data. However, most existing systems rely on unilateral control, which only transmits target position values. While this approach is easy to implement and suitable for slow, non-contact tasks, it struggles with fast or contact-rich operations due to the absence of force feedback. This work demonstrates that fast teleoperation with force feedback is feasible even with force-sensorless, low-cost manipulators by leveraging 4-channel bilateral control. Based on accurately identified manipulator dynamics, our method integrates nonlinear terms compensation, velocity and external force estimation, and variable gain corresponding to inertial variation. Furthermore, using data collected by 4-channel bilateral control, we show that incorporating force information into both the input and output of learned policies improves performance in imitation learning. These results highlight the practical effectiveness of our system for high-fidelity teleoperation and data collection on affordable hardware.
>
---
#### [replaced 004] Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.16685v2](http://arxiv.org/pdf/2506.16685v2)**

> **作者:** Xiaomeng Xu; Yifan Hou; Zeyi Liu; Shuran Song
>
> **摘要:** We address key challenges in Dataset Aggregation (DAgger) for real-world contact-rich manipulation: how to collect informative human correction data and how to effectively update policies with this new data. We introduce Compliant Residual DAgger (CR-DAgger), which contains two novel components: 1) a Compliant Intervention Interface that leverages compliance control, allowing humans to provide gentle, accurate delta action corrections without interrupting the ongoing robot policy execution; and 2) a Compliant Residual Policy formulation that learns from human corrections while incorporating force feedback and force control. Our system significantly enhances performance on precise contact-rich manipulation tasks using minimal correction data, improving base policy success rates by over 50\% on two challenging tasks (book flipping and belt assembly) while outperforming both retraining-from-scratch and finetuning approaches. Through extensive real-world experiments, we provide practical guidance for implementing effective DAgger in real-world robot learning tasks. Result videos are available at: https://compliant-residual-dagger.github.io/
>
---
#### [replaced 005] RUMI: Rummaging Using Mutual Information
- **分类: cs.RO; cs.AI; I.2.9**

- **链接: [http://arxiv.org/pdf/2408.10450v2](http://arxiv.org/pdf/2408.10450v2)**

> **作者:** Sheng Zhong; Nima Fazeli; Dmitry Berenson
>
> **备注:** 20 pages, 20 figures, accepted by IEEE Transactions on Robotics (T-RO), preprint
>
> **摘要:** This paper presents Rummaging Using Mutual Information (RUMI), a method for online generation of robot action sequences to gather information about the pose of a known movable object in visually-occluded environments. Focusing on contact-rich rummaging, our approach leverages mutual information between the object pose distribution and robot trajectory for action planning. From an observed partial point cloud, RUMI deduces the compatible object pose distribution and approximates the mutual information of it with workspace occupancy in real time. Based on this, we develop an information gain cost function and a reachability cost function to keep the object within the robot's reach. These are integrated into a model predictive control (MPC) framework with a stochastic dynamics model, updating the pose distribution in a closed loop. Key contributions include a new belief framework for object pose estimation, an efficient information gain computation strategy, and a robust MPC-based control scheme. RUMI demonstrates superior performance in both simulated and real tasks compared to baseline methods.
>
---
#### [replaced 006] Differentiable Motion Manifold Primitives for Reactive Motion Generation under Kinodynamic Constraints
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.12193v2](http://arxiv.org/pdf/2410.12193v2)**

> **作者:** Yonghyeon Lee
>
> **备注:** 6 pages and 9 figures
>
> **摘要:** Real-time motion generation -- which is essential for achieving reactive and adaptive behavior -- under kinodynamic constraints for high-dimensional systems is a crucial yet challenging problem. We address this with a two-step approach: offline learning of a lower-dimensional trajectory manifold of task-relevant, constraint-satisfying trajectories, followed by rapid online search within this manifold. Extending the discrete-time Motion Manifold Primitives (MMP) framework, we propose Differentiable Motion Manifold Primitives (DMMP), a novel neural network architecture that encodes and generates continuous-time, differentiable trajectories, trained using data collected offline through trajectory optimizations, with a strategy that ensures constraint satisfaction -- absent in existing methods. Experiments on dynamic throwing with a 7-DoF robot arm demonstrate that DMMP outperforms prior methods in planning speed, task success, and constraint satisfaction.
>
---
#### [replaced 007] Diffusion Beats Autoregressive in Data-Constrained Settings
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15857v2](http://arxiv.org/pdf/2507.15857v2)**

> **作者:** Mihir Prabhudesai; Menging Wu; Amir Zadeh; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage: https://diffusion-scaling.github.io
>
> **摘要:** Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: https://diffusion-scaling.github.io.
>
---
#### [replaced 008] Leveraging multi-source and heterogeneous signals for fatigue detection
- **分类: cs.RO; cs.AI; 62H30; I.2**

- **链接: [http://arxiv.org/pdf/2507.16859v2](http://arxiv.org/pdf/2507.16859v2)**

> **作者:** Luobin Cui; Yanlai Wu; Tang Ying; Weikai Li
>
> **备注:** 1figures,32pages
>
> **摘要:** Fatigue detection plays a critical role in safety-critical applications such as aviation, mining, and long-haul transport. However, most existing methods rely on high-end sensors and controlled environments, limiting their applicability in real world settings. This paper formally defines a practical yet underexplored problem setting for real world fatigue detection, where systems operating with context-appropriate sensors aim to leverage knowledge from differently instrumented sources including those using impractical sensors deployed in controlled environments. To tackle this challenge, we propose a heterogeneous and multi-source fatigue detection framework that adaptively utilizes the available modalities in the target domain while benefiting from the diverse configurations present in source domains. Our experiments, conducted using a realistic field-deployed sensor setup and two publicly available datasets, demonstrate the practicality, robustness, and improved generalization of our approach, paving the practical way for effective fatigue monitoring in sensor-constrained scenarios.
>
---
#### [replaced 009] Designing Effective Human-Swarm Interaction Interfaces: Insights from a User Study on Task Performance
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.02250v2](http://arxiv.org/pdf/2504.02250v2)**

> **作者:** Wasura D. Wattearachchi; Erandi Lakshika; Kathryn Kasmarik; Michael Barlow
>
> **备注:** 8 pages, 4 figures, 5 tables
>
> **摘要:** In this paper, we present a systematic method of design for human-swarm interaction interfaces, combining theoretical insights with empirical evaluation. We first derived ten design principles from existing literature, applying them to key information dimensions identified through goal-directed task analysis and developed a tablet-based interface for a target search task. We then conducted a user study with 31 participants where humans were required to guide a robotic swarm to a target in the presence of three types of hazards that pose a risk to the robots: Distributed, Moving, and Spreading. Performance was measured based on the proximity of the robots to the target and the number of deactivated robots at the end of the task. Results indicate that at least one robot was brought closer to the target in 98% of tasks, demonstrating the interface's success in fulfilling the primary objective of the task. Additionally, in nearly 67% of tasks, more than 50% of the robots reached the target. Moreover, particularly better performance was noted in moving hazards. Additionally, the interface appeared to help minimise robot deactivation, as evidenced by nearly 94% of tasks where participants managed to keep more than 50% of the robots active, ensuring that most of the swarm remained operational. However, its effectiveness varied across hazards, with robot deactivation being lowest in distributed hazard scenarios, suggesting that the interface provided the most support in these conditions.
>
---
#### [replaced 010] Learning Gentle Grasping Using Vision, Sound, and Touch
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.07926v2](http://arxiv.org/pdf/2503.07926v2)**

> **作者:** Ken Nakahara; Roberto Calandra
>
> **备注:** 8 pages. Accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** In our daily life, we often encounter objects that are fragile and can be damaged by excessive grasping force, such as fruits. For these objects, it is paramount to grasp gently -- not using the maximum amount of force possible, but rather the minimum amount of force necessary. This paper proposes using visual, tactile, and auditory signals to learn to grasp and regrasp objects stably and gently. Specifically, we use audio signals as an indicator of gentleness during the grasping, and then train an end-to-end action-conditional model from raw visuo-tactile inputs that predicts both the stability and the gentleness of future grasping candidates, thus allowing the selection and execution of the most promising action. Experimental results on a multi-fingered hand over 1,500 grasping trials demonstrated that our model is useful for gentle grasping by validating the predictive performance (3.27% higher accuracy than the vision-only variant) and providing interpretations of their behavior. Finally, real-world experiments confirmed that the grasping performance with the trained multi-modal model outperformed other baselines (17% higher rate for stable and gentle grasps than vision-only). Our approach requires neither tactile sensor calibration nor analytical force modeling, drastically reducing the engineering effort to grasp fragile objects. Dataset and videos are available at https://lasr.org/research/gentle-grasping.
>
---
#### [replaced 011] Compositional Coordination for Multi-Robot Teams with Large Language Models
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2507.16068v2](http://arxiv.org/pdf/2507.16068v2)**

> **作者:** Zhehui Huang; Guangyao Shi; Yuwei Wu; Vijay Kumar; Gaurav S. Sukhatme
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Multi-robot coordination has traditionally relied on a mission-specific and expert-driven pipeline, where natural language mission descriptions are manually translated by domain experts into mathematical formulation, algorithm design, and executable code. This conventional process is labor-intensive, inaccessible to non-experts, and inflexible to changes in mission requirements. Here, we propose LAN2CB (Language to Collective Behavior), a novel framework that leverages large language models (LLMs) to streamline and generalize the multi-robot coordination pipeline. LAN2CB transforms natural language (NL) mission descriptions into executable Python code for multi-robot systems through two core modules: (1) Mission Analysis, which parses mission descriptions into behavior trees, and (2) Code Generation, which leverages the behavior tree and a structured knowledge base to generate robot control code. We further introduce a dataset of natural language mission descriptions to support development and benchmarking. Experiments in both simulation and real-world environments demonstrate that LAN2CB enables robust and flexible multi-robot coordination from natural language, significantly reducing manual engineering effort and supporting broad generalization across diverse mission types. Website: https://sites.google.com/view/lan-cb
>
---
#### [replaced 012] Terrain-Aware Adaptation for Two-Dimensional UAV Path Planners
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17519v2](http://arxiv.org/pdf/2507.17519v2)**

> **作者:** Kostas Karakontis; Thanos Petsanis; Athanasios Ch. Kapoutsis; Pavlos Ch. Kapoutsis; Elias B. Kosmatopoulos
>
> **摘要:** Multi-UAV Coverage Path Planning (mCPP) algorithms in popular commercial software typically treat a Region of Interest (RoI) only as a 2D plane, ignoring important3D structure characteristics. This leads to incomplete 3Dreconstructions, especially around occluded or vertical surfaces. In this paper, we propose a modular algorithm that can extend commercial two-dimensional path planners to facilitate terrain-aware planning by adjusting altitude and camera orientations. To demonstrate it, we extend the well-known DARP (Divide Areas for Optimal Multi-Robot Coverage Path Planning) algorithm and produce DARP-3D. We present simulation results in multiple 3D environments and a real-world flight test using DJI hardware. Compared to baseline, our approach consistently captures improved 3D reconstructions, particularly in areas with significant vertical features. An open-source implementation of the algorithm is available here:https://github.com/konskara/TerraPlan
>
---
#### [replaced 013] Target Tracking via LiDAR-RADAR Sensor Fusion for Autonomous Racing
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20043v2](http://arxiv.org/pdf/2505.20043v2)**

> **作者:** Marcello Cellina; Matteo Corno; Sergio Matteo Savaresi
>
> **备注:** IEEE Conference, 6 pages
>
> **摘要:** High Speed multi-vehicle Autonomous Racing will increase the safety and performance of road-going Autonomous Vehicles. Precise vehicle detection and dynamics estimation from a moving platform is a key requirement for planning and executing complex autonomous overtaking maneuvers. To address this requirement, we have developed a Latency-Aware EKF-based Multi Target Tracking algorithm fusing LiDAR and RADAR measurements. The algorithm explots the different sensor characteristics by explicitly integrating the Range Rate in the EKF Measurement Function, as well as a-priori knowledge of the racetrack during state prediction. It can handle Out-Of-Sequence Measurements via Reprocessing using a double State and Measurement Buffer, ensuring sensor delay compensation with no information loss. This algorithm has been implemented on Team PoliMOVE's autonomous racecar, and was proved experimentally by completing a number of fully autonomous overtaking maneuvers at speeds up to 275 km/h.
>
---
#### [replaced 014] An Efficient Numerical Function Optimization Framework for Constrained Nonlinear Robotic Problems
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2501.17349v3](http://arxiv.org/pdf/2501.17349v3)**

> **作者:** Sait Sovukluk; Christian Ott
>
> **备注:** \c{opyright} 2025 the authors. This work has been accepted to IFAC for publication under a Creative Commons Licence CC-BY-NC-ND. - Implementation: https://github.com/ssovukluk/ENFORCpp
>
> **摘要:** This paper presents a numerical function optimization framework designed for constrained optimization problems in robotics. The tool is designed with real-time considerations and is suitable for online trajectory and control input optimization problems. The proposed framework does not require any analytical representation of the problem and works with constrained block-box optimization functions. The method combines first-order gradient-based line search algorithms with constraint prioritization through nullspace projections onto constraint Jacobian space. The tool is implemented in C++ and provided online for community use, along with some numerical and robotic example implementations presented in this paper.
>
---
#### [replaced 015] B4P: Simultaneous Grasp and Motion Planning for Object Placement via Parallelized Bidirectional Forests and Path Repair
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04598v2](http://arxiv.org/pdf/2504.04598v2)**

> **作者:** Benjamin H. Leebron; Kejia Ren; Yiting Chen; Kaiyu Hang
>
> **摘要:** Robot pick and place systems have traditionally decoupled grasp, placement, and motion planning to build sequential optimization pipelines with the assumption that the individual components will be able to work together. However, this separation introduces sub-optimality, as grasp choices may limit or even prohibit feasible motions for a robot to reach the target placement pose, particularly in cluttered environments with narrow passages. To this end, we propose a forest-based planning framework to simultaneously find grasp configurations and feasible robot motions that explicitly satisfy downstream placement configurations paired with the selected grasps. Our proposed framework leverages a bidirectional sampling-based approach to build a start forest, rooted at the feasible grasp regions, and a goal forest, rooted at the feasible placement regions, to facilitate the search through randomly explored motions that connect valid pairs of grasp and placement trees. We demonstrate that the framework's inherent parallelism enables superlinear speedup, making it scalable for applications for redundant robot arms (e.g., 7 Degrees of Freedom) to work efficiently in highly cluttered environments. Extensive experiments in simulation demonstrate the robustness and efficiency of the proposed framework in comparison with multiple baselines under diverse scenarios.
>
---
#### [replaced 016] Realtime Limb Trajectory Optimization for Humanoid Running Through Centroidal Angular Momentum Dynamics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.17351v3](http://arxiv.org/pdf/2501.17351v3)**

> **作者:** Sait Sovukluk; Robert Schuller; Johannes Englsberger; Christian Ott
>
> **备注:** This paper has been accepted for publication at the IEEE International Conference on Robotics and Automation (ICRA), Atlanta 2025. Link to video: https://www.youtube.com/watch?v=czfHjwh_A0Y
>
> **摘要:** One of the essential aspects of humanoid robot running is determining the limb-swinging trajectories. During the flight phases, where the ground reaction forces are not available for regulation, the limb swinging trajectories are significant for the stability of the next stance phase. Due to the conservation of angular momentum, improper leg and arm swinging results in highly tilted and unsustainable body configurations at the next stance phase landing. In such cases, the robotic system fails to maintain locomotion independent of the stability of the center of mass trajectories. This problem is more apparent for fast and high flight time trajectories. This paper proposes a real-time nonlinear limb trajectory optimization problem for humanoid running. The optimization problem is tested on two different humanoid robot models, and the generated trajectories are verified using a running algorithm for both robots in a simulation environment.
>
---
#### [replaced 017] A Differentiated Reward Method for Reinforcement Learning based Multi-Vehicle Cooperative Decision-Making Algorithms
- **分类: cs.AI; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.00352v2](http://arxiv.org/pdf/2502.00352v2)**

> **作者:** Ye Han; Lijun Zhang; Dejian Meng; Zhuang Zhang
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Reinforcement learning (RL) shows great potential for optimizing multi-vehicle cooperative driving strategies through the state-action-reward feedback loop, but it still faces challenges such as low sample efficiency. This paper proposes a differentiated reward method based on steady-state transition systems, which incorporates state transition gradient information into the reward design by analyzing traffic flow characteristics, aiming to optimize action selection and policy learning in multi-vehicle cooperative decision-making. The performance of the proposed method is validated in RL algorithms such as MAPPO, MADQN, and QMIX under varying autonomous vehicle penetration. The results show that the differentiated reward method significantly accelerates training convergence and outperforms centering reward and others in terms of traffic efficiency, safety, and action rationality. Additionally, the method demonstrates strong scalability and environmental adaptability, providing a novel approach for multi-agent cooperative decision-making in complex traffic scenarios.
>
---
#### [replaced 018] CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.17727v2](http://arxiv.org/pdf/2507.17727v2)**

> **作者:** Robel Mamo; Taeyeong Choi
>
> **备注:** Accepted for publication at the 12th European Conference on Mobile Robots (ECMR 2025)
>
> **摘要:** State-of-the-art visual under-canopy navigation methods are designed with deep learning-based perception models to distinguish traversable space from crop rows. While these models have demonstrated successful performance, they require large amounts of training data to ensure reliability in real-world field deployment. However, data collection is costly, demanding significant human resources for in-field sampling and annotation. To address this challenge, various data augmentation techniques are commonly employed during model training, such as color jittering, Gaussian blur, and horizontal flip, to diversify training data and enhance model robustness. In this paper, we hypothesize that utilizing only these augmentation techniques may lead to suboptimal performance, particularly in complex under-canopy environments with frequent occlusions, debris, and non-uniform spacing of crops. Instead, we propose a novel augmentation method, so-called Crop-Aligned Cutout (CA-Cut) which masks random regions out in input images that are spatially distributed around crop rows on the sides to encourage trained models to capture high-level contextual features even when fine-grained information is obstructed. Our extensive experiments with a public cornfield dataset demonstrate that masking-based augmentations are effective for simulating occlusions and significantly improving robustness in semantic keypoint predictions for visual navigation. In particular, we show that biasing the mask distribution toward crop rows in CA-Cut is critical for enhancing both prediction accuracy and generalizability across diverse environments achieving up to a 36.9% reduction in prediction error. In addition, we conduct ablation studies to determine the number of masks, the size of each mask, and the spatial distribution of masks to maximize overall performance.
>
---
#### [replaced 019] Spatio-Temporal Motion Retargeting for Quadruped Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.11557v3](http://arxiv.org/pdf/2404.11557v3)**

> **作者:** Taerim Yoon; Dongho Kang; Seungmin Kim; Jin Cheng; Minsung Ahn; Stelian Coros; Sungjoon Choi
>
> **备注:** 20 pages, 12 figures, videos available at https://taerimyoon.me/Spatio-Temporal-Motion-Retargeting-for-Quadruped-Robots/
>
> **摘要:** This work presents a motion retargeting approach for legged robots, aimed at transferring the dynamic and agile movements to robots from source motions. In particular, we guide the imitation learning procedures by transferring motions from source to target, effectively bridging the morphological disparities while ensuring the physical feasibility of the target system. In the first stage, we focus on motion retargeting at the kinematic level by generating kinematically feasible whole-body motions from keypoint trajectories. Following this, we refine the motion at the dynamic level by adjusting it in the temporal domain while adhering to physical constraints. This process facilitates policy training via reinforcement learning, enabling precise and robust motion tracking. We demonstrate that our approach successfully transforms noisy motion sources, such as hand-held camera videos, into robot-specific motions that align with the morphology and physical properties of the target robots. Moreover, we demonstrate terrain-aware motion retargeting to perform BackFlip on top of a box. We successfully deployed these skills to four robots with different dimensions and physical properties in the real world through hardware experiments.
>
---
#### [replaced 020] RoboCar: A Rapidly Deployable Open-Source Platform for Autonomous Driving Research
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.03572v2](http://arxiv.org/pdf/2405.03572v2)**

> **作者:** Mehdi Testouri; Gamal Elghazaly; Raphael Frank
>
> **摘要:** This paper introduces RoboCar, an open-source research platform for autonomous driving developed at the University of Luxembourg. RoboCar provides a modular, cost-effective framework for the development of experimental Autonomous Driving Systems (ADS), utilizing the 2018 KIA Soul EV. The platform integrates a robust hardware and software architecture that aligns with the vehicle's existing systems, minimizing the need for extensive modifications. It supports various autonomous driving functions and has undergone real-world testing on public roads in Luxembourg City. This paper outlines the platform's architecture, integration challenges, and initial test results, offering insights into its application in advancing autonomous driving research. RoboCar is available to anyone at https://github.com/sntubix/robocar and is released under an open-source MIT license.
>
---
#### [replaced 021] Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.06736v2](http://arxiv.org/pdf/2503.06736v2)**

> **作者:** Daniel Morton; Marco Pavone
>
> **摘要:** Safe real-time control of robotic manipulators in unstructured environments requires handling numerous safety constraints without compromising task performance. Traditional approaches, such as artificial potential fields (APFs), suffer from local minima, oscillations, and limited scalability, while model predictive control (MPC) can be computationally expensive. Control barrier functions (CBFs) offer a promising alternative due to their high level of robustness and low computational cost, but these safety filters must be carefully designed to avoid significant reductions in the overall performance of the manipulator. In this work, we introduce an Operational Space Control Barrier Function (OSCBF) framework that integrates safety constraints while preserving task-consistent behavior. Our approach scales to hundreds of simultaneous constraints while retaining real-time control rates, ensuring collision avoidance, singularity prevention, and workspace containment even in highly cluttered settings or during dynamic motions. By explicitly accounting for the task hierarchy in the CBF objective, we prevent degraded performance across both joint-space and operational-space tasks, when at the limit of safety. We validate performance in both simulation and hardware, and release our open-source high-performance code and media on our project webpage, https://stanfordasl.github.io/oscbf/
>
---
#### [replaced 022] PRIX: Learning to Plan from Raw Pixels for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17596v2](http://arxiv.org/pdf/2507.17596v2)**

> **作者:** Maciej K. Wozniak; Lianhang Liu; Yixi Cai; Patric Jensfelt
>
> **备注:** under review
>
> **摘要:** While end-to-end autonomous driving models show promising results, their practical deployment is often hindered by large model sizes, a reliance on expensive LiDAR sensors and computationally intensive BEV feature representations. This limits their scalability, especially for mass-market vehicles equipped only with cameras. To address these challenges, we propose PRIX (Plan from Raw Pixels). Our novel and efficient end-to-end driving architecture operates using only camera data, without explicit BEV representation and forgoing the need for LiDAR. PRIX leverages a visual feature extractor coupled with a generative planning head to predict safe trajectories from raw pixel inputs directly. A core component of our architecture is the Context-aware Recalibration Transformer (CaRT), a novel module designed to effectively enhance multi-level visual features for more robust planning. We demonstrate through comprehensive experiments that PRIX achieves state-of-the-art performance on the NavSim and nuScenes benchmarks, matching the capabilities of larger, multimodal diffusion planners while being significantly more efficient in terms of inference speed and model size, making it a practical solution for real-world deployment. Our work is open-source and the code will be at https://maxiuw.github.io/prix.
>
---
