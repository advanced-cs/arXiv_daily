# 机器人 cs.RO

- **最新发布 106 篇**

- **更新 58 篇**

## 最新发布

#### [new 001] Constrained Decoding for Robotics Foundation Models
- **分类: cs.RO; cs.LG; cs.LO**

- **简介: 该论文提出约束解码框架，解决机器人基础模型缺乏安全约束的问题。通过在解码时强制信号时序逻辑（STL）规范，确保生成动作满足安全条件，无需重新训练模型。**

- **链接: [http://arxiv.org/pdf/2509.01728v1](http://arxiv.org/pdf/2509.01728v1)**

> **作者:** Parv Kapoor; Akila Ganlath; Changliu Liu; Sebastian Scherer; Eunsuk Kang
>
> **摘要:** Recent advances in the development of robotic foundation models have led to promising end-to-end and general-purpose capabilities in robotic systems. These models are pretrained on vast datasets of robot trajectories to process multi- modal inputs and directly output a sequence of action that the system then executes in the real world. Although this approach is attractive from the perspective of im- proved generalization across diverse tasks, these models are still data-driven and, therefore, lack explicit notions of behavioral correctness and safety constraints. We address these limitations by introducing a constrained decoding framework for robotics foundation models that enforces logical constraints on action trajec- tories in dynamical systems. Our method ensures that generated actions provably satisfy signal temporal logic (STL) specifications at runtime without retraining, while remaining agnostic of the underlying foundation model. We perform com- prehensive evaluation of our approach across state-of-the-art navigation founda- tion models and we show that our decoding-time interventions are useful not only for filtering unsafe actions but also for conditional action-generation. Videos available on our website: https://constrained-robot-fms.github.io
>
---
#### [new 002] A Reactive Grasping Framework for Multi-DoF Grippers via Task Space Velocity Fields and Joint Space QP
- **分类: cs.RO**

- **简介: 论文提出基于任务空间速度场与关节空间QP的分层框架，解决多自由度夹爪实时反应式抓取问题，实现动态环境下的碰撞规避与适应。**

- **链接: [http://arxiv.org/pdf/2509.01044v1](http://arxiv.org/pdf/2509.01044v1)**

> **作者:** Yonghyeon Lee; Tzu-Yuan Lin; Alexander Alexiev; Sangbae Kim
>
> **备注:** 8 pages, 12 figures, under review
>
> **摘要:** We present a fast and reactive grasping framework for multi-DoF grippers that combines task-space velocity fields with a joint-space Quadratic Program (QP) in a hierarchical structure. Reactive, collision-free global motion planning is particularly challenging for high-DoF systems, since simultaneous increases in state dimensionality and planning horizon trigger a combinatorial explosion of the search space, making real-time planning intractable. To address this, we plan globally in a lower-dimensional task space, such as fingertip positions, and track locally in the full joint space while enforcing all constraints. This approach is realized by constructing velocity fields in multiple task-space coordinates (or in some cases a subset of joint coordinates) and solving a weighted joint-space QP to compute joint velocities that track these fields with appropriately assigned priorities. Through simulation experiments with privileged knowledge and real-world tests using the recent pose-tracking algorithm FoundationPose, we verify that our method enables high-DoF arm-hand systems to perform real-time, collision-free reaching motions while adapting to dynamic environments and external disturbances.
>
---
#### [new 003] NeuralSVCD for Efficient Swept Volume Collision Detection
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出NeuralSVCD，结合几何与时间局部性，提升扫过体积碰撞检测的效率与准确性，适用于机器人运动规划。**

- **链接: [http://arxiv.org/pdf/2509.00499v1](http://arxiv.org/pdf/2509.00499v1)**

> **作者:** Dongwon Son; Hojin Jung; Beomjoon Kim
>
> **备注:** CoRL 2025
>
> **摘要:** Robot manipulation in unstructured environments requires efficient and reliable Swept Volume Collision Detection (SVCD) for safe motion planning. Traditional discrete methods potentially miss collisions between these points, whereas SVCD continuously checks for collisions along the entire trajectory. Existing SVCD methods typically face a trade-off between efficiency and accuracy, limiting practical use. In this paper, we introduce NeuralSVCD, a novel neural encoder-decoder architecture tailored to overcome this trade-off. Our approach leverages shape locality and temporal locality through distributed geometric representations and temporal optimization. This enhances computational efficiency without sacrificing accuracy. Comprehensive experiments show that NeuralSVCD consistently outperforms existing state-of-the-art SVCD methods in terms of both collision detection accuracy and computational efficiency, demonstrating its robust applicability across diverse robotic manipulation scenarios. Code and videos are available at https://neuralsvcd.github.io/.
>
---
#### [new 004] First Order Model-Based RL through Decoupled Backpropagation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出一种模型基强化学习方法，通过解耦轨迹生成与梯度计算，利用可微模型替代模拟器梯度，提升策略优化效率与准确性，适用于复杂控制任务。**

- **链接: [http://arxiv.org/pdf/2509.00215v1](http://arxiv.org/pdf/2509.00215v1)**

> **作者:** Joseph Amigo; Rooholla Khorrambakht; Elliot Chane-Sane; Nicolas Mansard; Ludovic Righetti
>
> **备注:** CoRL 2025. Project website: https://machines-in-motion.github.io/DMO/
>
> **摘要:** There is growing interest in reinforcement learning (RL) methods that leverage the simulator's derivatives to improve learning efficiency. While early gradient-based approaches have demonstrated superior performance compared to derivative-free methods, accessing simulator gradients is often impractical due to their implementation cost or unavailability. Model-based RL (MBRL) can approximate these gradients via learned dynamics models, but the solver efficiency suffers from compounding prediction errors during training rollouts, which can degrade policy performance. We propose an approach that decouples trajectory generation from gradient computation: trajectories are unrolled using a simulator, while gradients are computed via backpropagation through a learned differentiable model of the simulator. This hybrid design enables efficient and consistent first-order policy optimization, even when simulator gradients are unavailable, as well as learning a critic from simulation rollouts, which is more accurate. Our method achieves the sample efficiency and speed of specialized optimizers such as SHAC, while maintaining the generality of standard approaches like PPO and avoiding ill behaviors observed in other first-order MBRL methods. We empirically validate our algorithm on benchmark control tasks and demonstrate its effectiveness on a real Go2 quadruped robot, across both quadrupedal and bipedal locomotion tasks.
>
---
#### [new 005] DyPho-SLAM : Real-time Photorealistic SLAM in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文提出DyPho-SLAM，解决动态环境中的实时视觉SLAM问题。通过整合先验图像生成精炼掩码、自适应特征提取策略，提升动态场景下的定位精度与高保真地图重建能力。**

- **链接: [http://arxiv.org/pdf/2509.00741v1](http://arxiv.org/pdf/2509.00741v1)**

> **作者:** Yi Liu; Keyu Fan; Bin Lan; Houde Liu
>
> **备注:** Accepted by ICME 2025(Oral)
>
> **摘要:** Visual SLAM algorithms have been enhanced through the exploration of Gaussian Splatting representations, particularly in generating high-fidelity dense maps. While existing methods perform reliably in static environments, they often encounter camera tracking drift and fuzzy mapping when dealing with the disturbances caused by moving objects. This paper presents DyPho-SLAM, a real-time, resource-efficient visual SLAM system designed to address the challenges of localization and photorealistic mapping in environments with dynamic objects. Specifically, the proposed system integrates prior image information to generate refined masks, effectively minimizing noise from mask misjudgment. Additionally, to enhance constraints for optimization after removing dynamic obstacles, we devise adaptive feature extraction strategies significantly improving the system's resilience. Experiments conducted on publicly dynamic RGB-D datasets demonstrate that the proposed system achieves state-of-the-art performance in camera pose estimation and dense map reconstruction, while operating in real-time in dynamic scenes.
>
---
#### [new 006] Autonomous Aggregate Sorting in Construction and Mining via Computer Vision-Aided Robotic Arm Systems
- **分类: cs.RO**

- **简介: 该论文提出基于计算机视觉的机械臂系统，解决传统集料分拣精度低、适应性差的问题。通过YOLOv8检测、立体视觉定位、运动控制等技术实现自主分拣，实验验证成功率97.5%，为建筑与采矿领域提供智能化解决方案。**

- **链接: [http://arxiv.org/pdf/2509.00339v1](http://arxiv.org/pdf/2509.00339v1)**

> **作者:** Md. Taherul Islam Shawon; Yuan Li; Yincai Cai; Junjie Niu; Ting Peng
>
> **摘要:** Traditional aggregate sorting methods, whether manual or mechanical, often suffer from low precision, limited flexibility, and poor adaptability to diverse material properties such as size, shape, and lithology. To address these limitations, this study presents a computer vision-aided robotic arm system designed for autonomous aggregate sorting in construction and mining applications. The system integrates a six-degree-of-freedom robotic arm, a binocular stereo camera for 3D perception, and a ROS-based control framework. Core techniques include an attention-augmented YOLOv8 model for aggregate detection, stereo matching for 3D localization, Denavit-Hartenberg kinematic modeling for arm motion control, minimum enclosing rectangle analysis for size estimation, and hand-eye calibration for precise coordinate alignment. Experimental validation with four aggregate types achieved an average grasping and sorting success rate of 97.5%, with comparable classification accuracy. Remaining challenges include the reliable handling of small aggregates and texture-based misclassification. Overall, the proposed system demonstrates significant potential to enhance productivity, reduce operational costs, and improve safety in aggregate handling, while providing a scalable framework for advancing smart automation in construction, mining, and recycling industries.
>
---
#### [new 007] Generalizing Unsupervised Lidar Odometry Model from Normal to Snowy Weather Conditions
- **分类: cs.RO**

- **简介: 该论文针对LiDAR里程计在雪天噪声下的性能问题，提出无监督模型。通过PSM模块和PPWP方法有效去噪，结合强度阈值与多模态融合提升鲁棒性，实现晴天训练、雪天测试的跨天气适应，增强自主系统可靠性。**

- **链接: [http://arxiv.org/pdf/2509.02011v1](http://arxiv.org/pdf/2509.02011v1)**

> **作者:** Beibei Zhou; Zhiyuan Zhang; Zhenbo Song; Jianhui Guo; Hui Kong
>
> **摘要:** Deep learning-based LiDAR odometry is crucial for autonomous driving and robotic navigation, yet its performance under adverse weather, especially snowfall, remains challenging. Existing models struggle to generalize across conditions due to sensitivity to snow-induced noise, limiting real-world use. In this work, we present an unsupervised LiDAR odometry model to close the gap between clear and snowy weather conditions. Our approach focuses on effective denoising to mitigate the impact of snowflake noise and outlier points on pose estimation, while also maintaining computational efficiency for real-time applications. To achieve this, we introduce a Patch Spatial Measure (PSM) module that evaluates the dispersion of points within each patch, enabling effective detection of sparse and discrete noise. We further propose a Patch Point Weight Predictor (PPWP) to assign adaptive point-wise weights, enhancing their discriminative capacity within local regions. To support real-time performance, we first apply an intensity threshold mask to quickly suppress dense snowflake clusters near the LiDAR, and then perform multi-modal feature fusion to refine the point-wise weight prediction, improving overall robustness under adverse weather. Our model is trained in clear weather conditions and rigorously tested across various scenarios, including snowy and dynamic. Extensive experimental results confirm the effectiveness of our method, demonstrating robust performance in both clear and snowy weather. This advancement enhances the model's generalizability and paves the way for more reliable autonomous systems capable of operating across a wider range of environmental conditions.
>
---
#### [new 008] Towards Data-Driven Metrics for Social Robot Navigation Benchmarking
- **分类: cs.RO**

- **简介: 该论文提出数据驱动的社会机器人导航基准度量方法，解决现有评估标准不足问题。通过收集真实与模拟轨迹数据，建立评分数据集并训练RNN基线模型，为策略优化提供量化基准。**

- **链接: [http://arxiv.org/pdf/2509.01251v1](http://arxiv.org/pdf/2509.01251v1)**

> **作者:** Pilar Bachiller-Burgos; Ulysses Bernardet; Luis V. Calderita; Pranup Chhetri; Anthony Francis; Noriaki Hirose; Noé Pérez; Dhruv Shah; Phani T. Singamaneni; Xuesu Xiao; Luis J. Manso
>
> **摘要:** This paper presents a joint effort towards the development of a data-driven Social Robot Navigation metric to facilitate benchmarking and policy optimization. We provide our motivations for our approach and describe our proposal for storing rated social navigation trajectory datasets. Following these guidelines, we compiled a dataset with 4427 trajectories -- 182 real and 4245 simulated -- and presented it to human raters, yielding a total of 4402 rated trajectories after data quality assurance. We also trained an RNN-based baseline metric on the dataset and present quantitative and qualitative results. All data, software, and model weights are publicly available.
>
---
#### [new 009] One-Step Model Predictive Path Integral for Manipulator Motion Planning Using Configuration Space Distance Fields
- **分类: cs.RO**

- **简介: 论文提出结合配置空间距离场（CDFs）与模型预测路径积分（MPPI）的框架，解决机械臂运动规划中传统方法易陷入局部极小值及MPPI计算成本高的问题。通过统一成本函数并缩短时间范围至一步，实现高效避障，实验显示在复杂环境中成功率高且控制频率达750Hz以上。**

- **链接: [http://arxiv.org/pdf/2509.00836v1](http://arxiv.org/pdf/2509.00836v1)**

> **作者:** Yulin Li; Tetsuro Miyazaki; Kenji Kawashima
>
> **摘要:** Motion planning for robotic manipulators is a fundamental problem in robotics. Classical optimization-based methods typically rely on the gradients of signed distance fields (SDFs) to impose collision-avoidance constraints. However, these methods are susceptible to local minima and may fail when the SDF gradients vanish. Recently, Configuration Space Distance Fields (CDFs) have been introduced, which directly model distances in the robot's configuration space. Unlike workspace SDFs, CDFs are differentiable almost everywhere and thus provide reliable gradient information. On the other hand, gradient-free approaches such as Model Predictive Path Integral (MPPI) control leverage long-horizon rollouts to achieve collision avoidance. While effective, these methods are computationally expensive due to the large number of trajectory samples, repeated collision checks, and the difficulty of designing cost functions with heterogeneous physical units. In this paper, we propose a framework that integrates CDFs with MPPI to enable direct navigation in the robot's configuration space. Leveraging CDF gradients, we unify the MPPI cost in joint-space and reduce the horizon to one step, substantially cutting computation while preserving collision avoidance in practice. We demonstrate that our approach achieves nearly 100% success rates in 2D environments and consistently high success rates in challenging 7-DOF Franka manipulator simulations with complex obstacles. Furthermore, our method attains control frequencies exceeding 750 Hz, substantially outperforming both optimization-based and standard MPPI baselines. These results highlight the effectiveness and efficiency of the proposed CDF-MPPI framework for high-dimensional motion planning.
>
---
#### [new 010] A novel parameter estimation method for pneumatic soft hand control applying logarithmic decrement for pseudo rigid body modeling
- **分类: cs.RO**

- **简介: 论文提出PRBM plus LDM方法，解决软体机械手控制中模型计算效率低和参数识别复杂的问题，通过测试验证其在位置和力控制中的高精度与实时性。**

- **链接: [http://arxiv.org/pdf/2509.01113v1](http://arxiv.org/pdf/2509.01113v1)**

> **作者:** Haiyun Zhang; Kelvin HoLam Heung; Gabrielle J. Naquila; Ashwin Hingwe; Ashish D. Deshpande
>
> **摘要:** The rapid advancement in physical human-robot interaction (HRI) has accelerated the development of soft robot designs and controllers. Controlling soft robots, especially soft hand grasping, is challenging due to their continuous deformation, motivating the use of reduced model-based controllers for real-time dynamic performance. Most existing models, however, suffer from computational inefficiency and complex parameter identification, limiting their real-time applicability. To address this, we propose a paradigm coupling Pseudo-Rigid Body Modeling with the Logarithmic Decrement Method for parameter estimation (PRBM plus LDM). Using a soft robotic hand test bed, we validate PRBM plus LDM for predicting position and force output from pressure input and benchmark its performance. We then implement PRBM plus LDM as the basis for closed-loop position and force controllers. Compared to a simple PID controller, the PRBM plus LDM position controller achieves lower error (average maximum error across all fingers: 4.37 degrees versus 20.38 degrees). For force control, PRBM plus LDM outperforms constant pressure grasping in pinching tasks on delicate objects: potato chip 86 versus 82.5, screwdriver 74.42 versus 70, brass coin 64.75 versus 35. These results demonstrate PRBM plus LDM as a computationally efficient and accurate modeling technique for soft actuators, enabling stable and flexible grasping with precise force regulation.
>
---
#### [new 011] A Robust Numerical Method for Solving Trigonometric Equations in Robotic Kinematics
- **分类: cs.RO; math.AG**

- **简介: 该论文提出一种稳健数值方法，解决机器人运动学中的三角方程求解问题。通过多项式替换与SVD分析，分别处理非奇异和奇异矩阵，实现高精度（<1e-15）与100%成功率，适用于逆运动学等应用。**

- **链接: [http://arxiv.org/pdf/2509.01010v1](http://arxiv.org/pdf/2509.01010v1)**

> **作者:** Hai-Jun Su
>
> **摘要:** This paper presents a robust numerical method for solving systems of trigonometric equations commonly encountered in robotic kinematics. Our approach employs polynomial substitution techniques combined with eigenvalue decomposition to handle singular matrices and edge cases effectively. The method demonstrates superior numerical stability compared to traditional approaches and has been implemented as an open-source Python package. For non-singular matrices, we employ Weierstrass substitution to transform the system into a quartic polynomial, ensuring all analytical solutions are found. For singular matrices, we develop specialized geometric constraint methods using SVD analysis. The solver demonstrates machine precision accuracy ($< 10^{-15}$ error) with 100\% success rate on extensive test cases, making it particularly valuable for robotics applications such as inverse kinematics problems.
>
---
#### [new 012] Jacobian Exploratory Dual-Phase Reinforcement Learning for Dynamic Endoluminal Navigation of Deformable Continuum Robots
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出JEDP-RL框架，解决柔性连续体机器人动态内窥镜导航中非线性变形与部分可观测问题，通过分阶段雅可比估计与策略执行提升强化学习性能。**

- **链接: [http://arxiv.org/pdf/2509.00329v1](http://arxiv.org/pdf/2509.00329v1)**

> **作者:** Yu Tian; Chi Kit Ng; Hongliang Ren
>
> **摘要:** Deformable continuum robots (DCRs) present unique planning challenges due to nonlinear deformation mechanics and partial state observability, violating the Markov assumptions of conventional reinforcement learning (RL) methods. While Jacobian-based approaches offer theoretical foundations for rigid manipulators, their direct application to DCRs remains limited by time-varying kinematics and underactuated deformation dynamics. This paper proposes Jacobian Exploratory Dual-Phase RL (JEDP-RL), a framework that decomposes planning into phased Jacobian estimation and policy execution. During each training step, we first perform small-scale local exploratory actions to estimate the deformation Jacobian matrix, then augment the state representation with Jacobian features to restore approximate Markovianity. Extensive SOFA surgical dynamic simulations demonstrate JEDP-RL's three key advantages over proximal policy optimization (PPO) baselines: 1) Convergence speed: 3.2x faster policy convergence, 2) Navigation efficiency: requires 25% fewer steps to reach the target, and 3) Generalization ability: achieve 92% success rate under material property variations and achieve 83% (33% higher than PPO) success rate in the unseen tissue environment.
>
---
#### [new 013] OpenTie: Open-vocabulary Sequential Rebar Tying System
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出OpenTie系统，解决传统钢筋绑扎需模型训练且仅适配平面场景的问题。通过RGB到点云生成与开放词汇检测，结合机械臂实现无需3D训练的高精度水平/垂直钢筋绑扎，实验验证其实际有效性。**

- **链接: [http://arxiv.org/pdf/2509.00064v1](http://arxiv.org/pdf/2509.00064v1)**

> **作者:** Mingze Liu; Sai Fan; Haozhen Li; Haobo Liang; Yixing Yuan; Yanke Wang
>
> **备注:** This article is under its initial revision
>
> **摘要:** Robotic practices on the construction site emerge as an attention-attracting manner owing to their capability of tackle complex challenges, especially in the rebar-involved scenarios. Most of existing products and research are mainly focused on flat rebar setting with model training demands. To fulfill this gap, we propose OpenTie, a 3D training-free rebar tying framework utilizing a RGB-to-point-cloud generation and an open-vocabulary detection. We implements the OpenTie via a robotic arm with a binocular camera and guarantees a high accuracy by applying the prompt-based object detection method on the image filtered by our propose post-processing procedure based a image to point cloud generation framework. The system is flexible for horizontal and vertical rebar tying tasks and the experiments on the real-world rebar setting verifies that the effectiveness of the system in practice.
>
---
#### [new 014] Align-Then-stEer: Adapting the Vision-Language Action Models through Unified Latent Guidance
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ATE框架，解决VLA模型在跨本体/任务适应中的动作分布不匹配问题。通过统一潜在空间对齐和引导机制，提升模拟与现实中的多任务成功率，实现高效轻量的模型适配。**

- **链接: [http://arxiv.org/pdf/2509.02055v1](http://arxiv.org/pdf/2509.02055v1)**

> **作者:** Yang Zhang; Chenwei Wang; Ouyang Lu; Yuan Zhao; Yunfei Ge; Zhenglong Sun; Xiu Li; Chi Zhang; Chenjia Bai; Xuelong Li
>
> **备注:** The first three authors contributed equally
>
> **摘要:** Vision-Language-Action (VLA) models pre-trained on large, diverse datasets show remarkable potential for general-purpose robotic manipulation. However, a primary bottleneck remains in adapting these models to downstream tasks, especially when the robot's embodiment or the task itself differs from the pre-training data. This discrepancy leads to a significant mismatch in action distributions, demanding extensive data and compute for effective fine-tuning. To address this challenge, we introduce \textbf{Align-Then-stEer (\texttt{ATE})}, a novel, data-efficient, and plug-and-play adaptation framework. \texttt{ATE} first aligns disparate action spaces by constructing a unified latent space, where a variational autoencoder constrained by reverse KL divergence embeds adaptation actions into modes of the pre-training action latent distribution. Subsequently, it steers the diffusion- or flow-based VLA's generation process during fine-tuning via a guidance mechanism that pushes the model's output distribution towards the target domain. We conduct extensive experiments on cross-embodiment and cross-task manipulation in both simulation and real world. Compared to direct fine-tuning of representative VLAs, our method improves the average multi-task success rate by up to \textbf{9.8\%} in simulation and achieves a striking \textbf{32\% success rate gain} in a real-world cross-embodiment setting. Our work presents a general and lightweight solution that greatly enhances the practicality of deploying VLA models to new robotic platforms and tasks.
>
---
#### [new 015] Extended Diffeomorphism for Real-Time Motion Replication in Workspaces with Different Spatial Arrangements
- **分类: cs.RO**

- **简介: 该论文针对多机器人协同任务中因工作区空间差异导致的运动复制误差问题，提出基于扩展微分同胚的映射方法，通过预定义关键点实现主从机器人姿态的精确且平滑转换。**

- **链接: [http://arxiv.org/pdf/2509.00491v1](http://arxiv.org/pdf/2509.00491v1)**

> **作者:** Masaki Saito; Shunki Itadera; Toshiyuki Murakami
>
> **摘要:** This paper presents two types of extended diffeomorphism designs to compensate for spatial placement differences between robot workspaces. Teleoperation of multiple robots is attracting attention to expand the utilization of the robot embodiment. Real-time reproduction of robot motion would facilitate the efficient execution of similar tasks by multiple robots. A challenge in the motion reproduction is compensating for the spatial arrangement errors of target keypoints in robot workspaces. This paper proposes a methodology for smooth mappings that transform primary robot poses into follower robot poses based on the predefined key points in each workspace. Through a picking task experiment using a dual-arm UR5 robot, this study demonstrates that the proposed mapping generation method can balance lower mapping errors for precise operation and lower mapping gradients for smooth replicated movement.
>
---
#### [new 016] A Framework for Task and Motion Planning based on Expanding AND/OR Graphs
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出TMP-EAOG框架，解决空间环境下机器人任务与运动规划的不确定性与约束问题。通过扩展AND/OR图结合实时运动评估，实现鲁棒、可控的自主规划，并在模拟中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.00317v1](http://arxiv.org/pdf/2509.00317v1)**

> **作者:** Fulvio Mastrogiovanni; Antony Thomas
>
> **备注:** Accepted for an oral presentation at ASTRA Conference, 2025
>
> **摘要:** Robot autonomy in space environments presents unique challenges, including high perception and motion uncertainty, strict kinematic constraints, and limited opportunities for human intervention. Therefore, Task and Motion Planning (TMP) may be critical for autonomous servicing, surface operations, or even in-orbit missions, just to name a few, as it models tasks as discrete action sequencing integrated with continuous motion feasibility assessments. In this paper, we introduce a TMP framework based on expanding AND/OR graphs, referred to as TMP-EAOG, and demonstrate its adaptability to different scenarios. TMP-EAOG encodes task-level abstractions within an AND/OR graph, which expands iteratively as the plan is executed, and performs in-the-loop motion planning assessments to ascertain their feasibility. As a consequence, TMP-EAOG is characterised by the desirable properties of (i) robustness to a certain degree of uncertainty, because AND/OR graph expansion can accommodate for unpredictable information about the robot environment, (ii) controlled autonomy, since an AND/OR graph can be validated by human experts, and (iii) bounded flexibility, in that unexpected events, including the assessment of unfeasible motions, can lead to different courses of action as alternative paths in the AND/OR graph. We evaluate TMP-EAOG on two benchmark domains. We use a simulated mobile manipulator as a proxy for space-grade autonomous robots. Our evaluation shows that TMP-EAOG can deal with a wide range of challenges in the benchmarks.
>
---
#### [new 017] Adaptive Navigation Strategy for Low-Thrust Proximity Operations in Circular Relative Orbit
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出自适应观测器导航策略，解决圆相对轨道中编队飞行与非合作目标检查的导航问题，通过动态增益调整及Lyapunov分析提升轨迹精度，减少控制输入切换。**

- **链接: [http://arxiv.org/pdf/2509.02204v1](http://arxiv.org/pdf/2509.02204v1)**

> **作者:** Dario Ruggiero; Mauro Mancini; Elisa Capello
>
> **备注:** This work has been accepted and presented at the 35th AAS/AIAA Space Flight Mechanics Meeting, 2025, Kaua'i, Hawai
>
> **摘要:** This paper presents an adaptive observer-based navigation strategy for spacecraft in Circular Relative Orbit (CRO) scenarios, addressing challenges in proximity operations like formation flight and uncooperative target inspection. The proposed method adjusts observer gains based on the estimated state to achieve fast convergence and low noise sensitivity in state estimation. A Lyapunov-based analysis ensures stability and accuracy, while simulations using vision-based sensor data validate the approach under realistic conditions. Compared to classical observers with time-invariant gains, the proposed method enhances trajectory tracking precision and reduces control input switching, making it a promising solution for autonomous spacecraft localization and control.
>
---
#### [new 018] Mechanistic interpretability for steering vision-language-action models
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出通过机制可解释性框架控制VLA模型，解决现有方法缺乏解释性和控制能力的问题，利用内部表示识别语义方向实现实时行为调控。**

- **链接: [http://arxiv.org/pdf/2509.00328v1](http://arxiv.org/pdf/2509.00328v1)**

> **作者:** Bear Häon; Kaylene Stocking; Ian Chuang; Claire Tomlin
>
> **备注:** CoRL 2025. Project website: https://vla-mech-interp.github.io/
>
> **摘要:** Vision-Language-Action (VLA) models are a promising path to realizing generalist embodied agents that can quickly adapt to new tasks, modalities, and environments. However, methods for interpreting and steering VLAs fall far short of classical robotics pipelines, which are grounded in explicit models of kinematics, dynamics, and control. This lack of mechanistic insight is a central challenge for deploying learned policies in real-world robotics, where robustness and explainability are critical. Motivated by advances in mechanistic interpretability for large language models, we introduce the first framework for interpreting and steering VLAs via their internal representations, enabling direct intervention in model behavior at inference time. We project feedforward activations within transformer layers onto the token embedding basis, identifying sparse semantic directions - such as speed and direction - that are causally linked to action selection. Leveraging these findings, we introduce a general-purpose activation steering method that modulates behavior in real time, without fine-tuning, reward signals, or environment interaction. We evaluate this method on two recent open-source VLAs, Pi0 and OpenVLA, and demonstrate zero-shot behavioral control in simulation (LIBERO) and on a physical robot (UR5). This work demonstrates that interpretable components of embodied VLAs can be systematically harnessed for control - establishing a new paradigm for transparent and steerable foundation models in robotics.
>
---
#### [new 019] Geometric Control of Mechanical Systems with Symmetries Based on Sliding Modes
- **分类: cs.RO**

- **简介: 该论文提出基于滑模的几何控制框架，用于具有对称性的机械系统。通过利用对称性将控制分为底空间与结构群两阶段，降低设计复杂度并避免坐标选择困难，结合李雅普诺夫分析证明稳定性，应用于航天器与无人车控制。**

- **链接: [http://arxiv.org/pdf/2509.01985v1](http://arxiv.org/pdf/2509.01985v1)**

> **作者:** Eduardo Espindola; Yu Tang
>
> **备注:** 32 pages, 3 figures, journal submission
>
> **摘要:** In this paper, we propose a framework for designing sliding mode controllers for a class of mechanical systems with symmetry, both unconstrained and constrained, that evolve on principal fiber bundles. Control laws are developed based on the reduced motion equations by exploring symmetries, leading to a sliding mode control strategy where the reaching stage is executed on the base space, and the sliding stage is performed on the structure group. Thus, design complexity is reduced, and difficult choices for coordinate representations when working with a particular Lie group are avoided. For this purpose, a sliding subgroup is constructed on the structure group based on a kinematic controller, and the sliding variable will converge to the identity of the state manifold upon reaching the sliding subgroup. A reaching law based on a general sliding vector field is then designed on the base space using the local form of the mechanical connection to drive the sliding variable to the sliding subgroup, and its time evolution is given according to the appropriate covariant derivative. Almost global asymptotic stability and local exponential stability are demonstrated using a Lyapunov analysis. We apply the results to a fully actuated system (a rigid spacecraft actuated by reaction wheels) and a subactuated nonholonomic system (unicycle mobile robot actuated by wheels), which is also simulated for illustration.
>
---
#### [new 020] Multi-vessel Interaction-Aware Trajectory Prediction and Collision Risk Assessment
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出基于Transformer的多船轨迹预测框架，整合碰撞风险分析，解决传统单船预测忽略交互与风险评估的问题，通过因果卷积、空间变换及混合位置嵌入提升预测精度与安全决策支持。**

- **链接: [http://arxiv.org/pdf/2509.01836v1](http://arxiv.org/pdf/2509.01836v1)**

> **作者:** Md Mahbub Alam; Jose F. Rodrigues-Jr; Gabriel Spadon
>
> **摘要:** Accurate vessel trajectory prediction is essential for enhancing situational awareness and preventing collisions. Still, existing data-driven models are constrained mainly to single-vessel forecasting, overlooking vessel interactions, navigation rules, and explicit collision risk assessment. We present a transformer-based framework for multi-vessel trajectory prediction with integrated collision risk analysis. For a given target vessel, the framework identifies nearby vessels. It jointly predicts their future trajectories through parallel streams encoding kinematic and derived physical features, causal convolutions for temporal locality, spatial transformations for positional encoding, and hybrid positional embeddings that capture both local motion patterns and long-range dependencies. Evaluated on large-scale real-world AIS data using joint multi-vessel metrics, the model demonstrates superior forecasting capabilities beyond traditional single-vessel displacement errors. By simulating interactions among predicted trajectories, the framework further quantifies potential collision risks, offering actionable insights to strengthen maritime safety and decision support.
>
---
#### [new 021] ConceptBot: Enhancing Robot's Autonomy through Task Decomposition with Large Language Models and Knowledge Graph
- **分类: cs.RO**

- **简介: 该论文提出ConceptBot框架，通过整合大语言模型与知识图谱解决机器人任务分解中的自然语言歧义和环境理解问题。其核心工作包括三个模块：对象属性提取、用户指令解析与上下文感知规划，显著提升机器人在复杂场景下的任务成功率与风险控制能力。**

- **链接: [http://arxiv.org/pdf/2509.00570v1](http://arxiv.org/pdf/2509.00570v1)**

> **作者:** Alessandro Leanza; Angelo Moroncelli; Giuseppe Vizzari; Francesco Braghin; Loris Roveda; Blerina Spahiu
>
> **摘要:** ConceptBot is a modular robotic planning framework that combines Large Language Models and Knowledge Graphs to generate feasible and risk-aware plans despite ambiguities in natural language instructions and correctly analyzing the objects present in the environment - challenges that typically arise from a lack of commonsense reasoning. To do that, ConceptBot integrates (i) an Object Property Extraction (OPE) module that enriches scene understanding with semantic concepts from ConceptNet, (ii) a User Request Processing (URP) module that disambiguates and structures instructions, and (iii) a Planner that generates context-aware, feasible pick-and-place policies. In comparative evaluations against Google SayCan, ConceptBot achieved 100% success on explicit tasks, maintained 87% accuracy on implicit tasks (versus 31% for SayCan), reached 76% on risk-aware tasks (versus 15%), and outperformed SayCan in application-specific scenarios, including material classification (70% vs. 20%) and toxicity detection (86% vs. 36%). On SafeAgentBench, ConceptBot achieved an overall score of 80% (versus 46% for the next-best baseline). These results, validated in both simulation and laboratory experiments, demonstrate ConceptBot's ability to generalize without domain-specific training and to significantly improve the reliability of robotic policies in unstructured environments. Website: https://sites.google.com/view/conceptbot
>
---
#### [new 022] Poke and Strike: Learning Task-Informed Exploration Policies
- **分类: cs.RO**

- **简介: 论文针对动态机器人任务中物体物理属性估计难题，提出基于强化学习的任务引导探索策略，通过敏感性奖励和不确定性机制优化探索效率，实现高成功率与低探索时间，在物理机器人上验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.00178v1](http://arxiv.org/pdf/2509.00178v1)**

> **作者:** Marina Y. Aoyama; Joao Moura; Juan Del Aguila Ferrandis; Sethu Vijayakumar
>
> **备注:** 8 pages (main paper), 27 pages (including references and appendices), 6 figures (main paper), 21 figures (including appendices), Conference of Robot Learning 2025, For videos and the project website, see https://marina-aoyama.github.io/poke-and-strike/
>
> **摘要:** In many dynamic robotic tasks, such as striking pucks into a goal outside the reachable workspace, the robot must first identify the relevant physical properties of the object for successful task execution, as it is unable to recover from failure or retry without human intervention. To address this challenge, we propose a task-informed exploration approach, based on reinforcement learning, that trains an exploration policy using rewards automatically generated from the sensitivity of a privileged task policy to errors in estimated properties. We also introduce an uncertainty-based mechanism to determine when to transition from exploration to task execution, ensuring sufficient property estimation accuracy with minimal exploration time. Our method achieves a 90% success rate on the striking task with an average exploration time under 1.2 seconds, significantly outperforming baselines that achieve at most 40% success or require inefficient querying and retraining in a simulator at test time. Additionally, we demonstrate that our task-informed rewards capture the relative importance of physical properties in both the striking task and the classical CartPole example. Finally, we validate our approach by demonstrating its ability to identify object properties and adjust task execution in a physical setup using the KUKA iiwa robot arm.
>
---
#### [new 023] Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety
- **分类: cs.RO; cs.AI**

- **简介: 论文提出统一框架，结合提示组装、状态管理及安全验证，提升LLM集成机器人系统的安全性和可靠性，应对对抗攻击与复杂环境挑战。**

- **链接: [http://arxiv.org/pdf/2509.02163v1](http://arxiv.org/pdf/2509.02163v1)**

> **作者:** Wenxiao Zhang; Xiangrui Kong; Conan Dewitt; Thomas Bräunl; Jin B. Hong
>
> **摘要:** Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/
>
---
#### [new 024] Inverse Kinematics for a 6-Degree-of-Freedom Robot Manipulator Using Comprehensive Gröbner Systems
- **分类: cs.RO; cs.SC; math.AC; 68W30, 13P10, 13P25, 68U07, 68R10**

- **简介: 论文提出基于综合Gröbner系统的逆运动学方法，解决更广泛6-DOF机械臂的逆运动学问题，通过参数化关节参数避免重复计算，提升求解效率。**

- **链接: [http://arxiv.org/pdf/2509.00823v1](http://arxiv.org/pdf/2509.00823v1)**

> **作者:** Takumu Okazaki; Akira Terui; Masahiko Mikawa
>
> **备注:** 24 pages
>
> **摘要:** We propose an effective method for solving the inverse kinematic problem of a specific model of 6-degree-of-freedom (6-DOF) robot manipulator using computer algebra. It is known that when the rotation axes of three consecutive rotational joints of a manipulator intersect at a single point, the inverse kinematics problem can be divided into determining position and orientation. We extend this method to more general manipulators in which the rotational axes of two consecutive joints intersect. This extension broadens the class of 6-DOF manipulators for which the inverse kinematics problem can be solved, and is expected to enable more efficient solutions. The inverse kinematic problem is solved using the Comprehensive Gr\"obner System (CGS) with joint parameters of the robot appearing as parameters in the coefficients to prevent repetitive calculations of the Gr\"obner bases. The effectiveness of the proposed method is shown by experiments.
>
---
#### [new 025] ManiFlow: A General Robot Manipulation Policy via Consistency Flow Training
- **分类: cs.RO**

- **简介: 本文提出ManiFlow，通过流匹配与一致性训练，实现多模态输入下的高效机器人操作策略，提升动作生成精度与成功率，适用于单臂、双臂及人形机器人，展现强泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.01819v1](http://arxiv.org/pdf/2509.01819v1)**

> **作者:** Ge Yan; Jiyue Zhu; Yuquan Deng; Shiqi Yang; Ri-Zhao Qiu; Xuxin Cheng; Marius Memmel; Ranjay Krishna; Ankit Goyal; Xiaolong Wang; Dieter Fox
>
> **摘要:** This paper introduces ManiFlow, a visuomotor imitation learning policy for general robot manipulation that generates precise, high-dimensional actions conditioned on diverse visual, language and proprioceptive inputs. We leverage flow matching with consistency training to enable high-quality dexterous action generation in just 1-2 inference steps. To handle diverse input modalities efficiently, we propose DiT-X, a diffusion transformer architecture with adaptive cross-attention and AdaLN-Zero conditioning that enables fine-grained feature interactions between action tokens and multi-modal observations. ManiFlow demonstrates consistent improvements across diverse simulation benchmarks and nearly doubles success rates on real-world tasks across single-arm, bimanual, and humanoid robot setups with increasing dexterity. The extensive evaluation further demonstrates the strong robustness and generalizability of ManiFlow to novel objects and background changes, and highlights its strong scaling capability with larger-scale datasets. Our website: maniflow-policy.github.io.
>
---
#### [new 026] Galaxea Open-World Dataset and G0 Dual-System VLA Model
- **分类: cs.RO; cs.CV**

- **简介: 论文提出Galaxea数据集与G0双系统VLA模型，解决真实环境中机器人多模态规划与细粒度执行问题。通过三阶段训练框架，结合数据集与单实体预训练，提升表征能力，实现复杂任务如移动操作与少样本学习。**

- **链接: [http://arxiv.org/pdf/2509.00576v1](http://arxiv.org/pdf/2509.00576v1)**

> **作者:** Tao Jiang; Tianyuan Yuan; Yicheng Liu; Chenhao Lu; Jianning Cui; Xiao Liu; Shuiqi Cheng; Jiyang Gao; Huazhe Xu; Hang Zhao
>
> **备注:** https://opengalaxea.github.io/G0/
>
> **摘要:** We present Galaxea Open-World Dataset, a large-scale, diverse collection of robot behaviors recorded in authentic human living and working environments. All demonstrations are gathered using a consistent robotic embodiment, paired with precise subtask-level language annotations to facilitate both training and evaluation. Building on this dataset, we introduce G0, a dual-system framework that couples a Vision-Language Model (VLM) for multimodal planning with a Vision-Language-Action (VLA) model for fine-grained execution. G0 is trained using a three-stage curriculum: cross-embodiment pre-training, single-embodiment pre-training, and task-specific post-training. A comprehensive benchmark spanning tabletop manipulation, few-shot learning, and long-horizon mobile manipulation, demonstrates the effectiveness of our approach. In particular, we find that the single-embodiment pre-training stage, together with the Galaxea Open-World Dataset, plays a critical role in achieving strong performance.
>
---
#### [new 027] Contact-Aided Navigation of Flexible Robotic Endoscope Using Deep Reinforcement Learning in Dynamic Stomach
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出基于深度强化学习的接触辅助导航策略，解决柔性内窥镜在动态胃部环境中的精准导航问题。通过物理仿真训练，利用接触力反馈提升运动稳定性，实现高成功率与低误差。**

- **链接: [http://arxiv.org/pdf/2509.00319v1](http://arxiv.org/pdf/2509.00319v1)**

> **作者:** Chi Kit Ng; Huxin Gao; Tian-Ao Ren; Jiewen Lai; Hongliang Ren
>
> **摘要:** Navigating a flexible robotic endoscope (FRE) through the gastrointestinal tract is critical for surgical diagnosis and treatment. However, navigation in the dynamic stomach is particularly challenging because the FRE must learn to effectively use contact with the deformable stomach walls to reach target locations. To address this, we introduce a deep reinforcement learning (DRL) based Contact-Aided Navigation (CAN) strategy for FREs, leveraging contact force feedback to enhance motion stability and navigation precision. The training environment is established using a physics-based finite element method (FEM) simulation of a deformable stomach. Trained with the Proximal Policy Optimization (PPO) algorithm, our approach achieves high navigation success rates (within 3 mm error between the FRE's end-effector and target) and significantly outperforms baseline policies. In both static and dynamic stomach environments, the CAN agent achieved a 100% success rate with 1.6 mm average error, and it maintained an 85% success rate in challenging unseen scenarios with stronger external disturbances. These results validate that the DRL-based CAN strategy substantially enhances FRE navigation performance over prior methods.
>
---
#### [new 028] Aleatoric Uncertainty from AI-based 6D Object Pose Predictors for Object-relative State Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对物体相对状态估计任务，解决AI预测器输出的aleatoric不确定性量化问题。通过在现有6D姿态预测器中添加分离的MLP模块，实现不确定性推断，结合EKF提升估计性能，适用于边缘设备部署。**

- **链接: [http://arxiv.org/pdf/2509.01583v1](http://arxiv.org/pdf/2509.01583v1)**

> **作者:** Thomas Jantos; Stephan Weiss; Jan Steinbrener
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Deep Learning (DL) has become essential in various robotics applications due to excelling at processing raw sensory data to extract task specific information from semantic objects. For example, vision-based object-relative navigation relies on a DL-based 6D object pose predictor to provide the relative pose between the object and the robot as measurements to the robot's state estimator. Accurately knowing the uncertainty inherent in such Deep Neural Network (DNN) based measurements is essential for probabilistic state estimators subsequently guiding the robot's tasks. Thus, in this letter, we show that we can extend any existing DL-based object-relative pose predictor for aleatoric uncertainty inference simply by including two multi-layer perceptrons detached from the translational and rotational part of the DL predictor. This allows for efficient training while freezing the existing pre-trained predictor. We then use the inferred 6D pose and its uncertainty as a measurement and corresponding noise covariance matrix in an extended Kalman filter (EKF). Our approach induces minimal computational overhead such that the state estimator can be deployed on edge devices while benefiting from the dynamically inferred measurement uncertainty. This increases the performance of the object-relative state estimation task compared to a fix-covariance approach. We conduct evaluations on synthetic data and real-world data to underline the benefits of aleatoric uncertainty inference for the object-relative state estimation task.
>
---
#### [new 029] Model Predictive Control for a Soft Robotic Finger with Stochastic Behavior based on Fokker-Planck Equation
- **分类: cs.RO**

- **简介: 该论文提出基于Fokker-Planck方程的模型预测控制方法，用于解决软体机器人手指因灵活性带来的高度不确定性问题，通过概率分布控制提升系统鲁棒性，并通过仿真验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.01065v1](http://arxiv.org/pdf/2509.01065v1)**

> **作者:** Sumitaka Honji; Takahiro Wada
>
> **备注:** 6 pages, 7 figures, presented/published at 2025 IEEE 8th International Conference on Soft Robotics (RoboSoft)
>
> **摘要:** The inherent flexibility of soft robots offers numerous advantages, such as enhanced adaptability and improved safety. However, this flexibility can also introduce challenges regarding highly uncertain and nonlinear motion. These challenges become particularly problematic when using open-loop control methods, which lack a feedback mechanism and are commonly employed in soft robot control. Though one potential solution is model-based control, typical deterministic models struggle with uncertainty as mentioned above. The idea is to use the Fokker-Planck Equation (FPE), a master equation of a stochastic process, to control not the state of soft robots but the probabilistic distribution. In this study, we propose and implement a stochastic-based control strategy, termed FPE-based Model Predictive Control (FPE-MPC), for a soft robotic finger. Two numerical simulation case studies examine the performance and characteristics of this control method, revealing its efficacy in managing the uncertainty inherent in soft robotic systems.
>
---
#### [new 030] Physics-Informed Machine Learning with Adaptive Grids for Optical Microrobot Depth Estimation
- **分类: cs.RO**

- **简介: 论文提出一种结合物理信息与自适应网格的深度估计框架，解决光学微机器人在透明、低对比度环境下的深度感知难题。通过物理指标与网格动态分配，提升精度并降低数据依赖，实现在有限数据下超越传统模型。**

- **链接: [http://arxiv.org/pdf/2509.02343v1](http://arxiv.org/pdf/2509.02343v1)**

> **作者:** Lan Wei; Lou Genoud; Dandan Zhang
>
> **备注:** 2025 IEEE International Conference on Cyborg and Bionic Systems (CBS 2025)
>
> **摘要:** Optical microrobots actuated by optical tweezers (OT) offer great potential for biomedical applications such as cell manipulation and microscale assembly. These tasks demand accurate three-dimensional perception to ensure precise control in complex and dynamic biological environments. However, the transparent nature of microrobots and low-contrast microscopic imaging challenge conventional deep learning methods, which also require large annotated datasets that are costly to obtain. To address these challenges, we propose a physics-informed, data-efficient framework for depth estimation of optical microrobots. Our method augments convolutional feature extraction with physics-based focus metrics, such as entropy, Laplacian of Gaussian, and gradient sharpness, calculated using an adaptive grid strategy. This approach allocates finer grids over microrobot regions and coarser grids over background areas, enhancing depth sensitivity while reducing computational complexity. We evaluate our framework on multiple microrobot types and demonstrate significant improvements over baseline models. Specifically, our approach reduces mean squared error (MSE) by over 60% and improves the coefficient of determination (R^2) across all test cases. Notably, even when trained on only 20% of the available data, our model outperforms ResNet50 trained on the full dataset, highlighting its robustness under limited data conditions. Our code is available at: https://github.com/LannWei/CBS2025.
>
---
#### [new 031] Needle Biopsy And Fiber-Optic Compatible Robotic Insertion Platform
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种机器人插入平台，解决组织活检中手动采样不准确和病理测试耗时的问题。设计兼容针和光纤的机械系统，实现精准导航与多模态诊断，并通过测试验证其性能。**

- **链接: [http://arxiv.org/pdf/2509.00530v1](http://arxiv.org/pdf/2509.00530v1)**

> **作者:** Fanxin Wang; Yikun Cheng; Chuyuan Tao; Rohit Bhargava; Thenkurussi Kesavadas
>
> **备注:** Presented in EMBC 2025
>
> **摘要:** Tissue biopsy is the gold standard for diagnosing many diseases, involving the extraction of diseased tissue for histopathology analysis by expert pathologists. However, this procedure has two main limitations: 1) Manual sampling through tissue biopsy is prone to inaccuracies; 2) The extraction process is followed by a time-consuming pathology test. To address these limitations, we present a compact, accurate, and maneuverable robotic insertion platform to overcome the limitations in traditional histopathology. Our platform is capable of steering a variety of tools with different sizes, including needle for tissue extraction and optical fibers for vibrational spectroscopy applications. This system facilitates the guidance of end-effector to the tissue and assists surgeons in navigating to the biopsy target area for multi-modal diagnosis. In this paper, we outline the general concept of our device, followed by a detailed description of its mechanical design and control scheme. We conclude with the validation of the system through a series of tests, including positioning accuracy, admittance performance, and tool insertion efficacy.
>
---
#### [new 032] Learn from What We HAVE: History-Aware VErifier that Reasons about Past Interactions Online
- **分类: cs.RO**

- **简介: 该论文提出History-Aware VErifier（HAVE）方法，解决机器人在线处理视觉模糊对象时的不确定性问题。通过分离动作生成与历史感知验证，利用过去交互信息提升决策质量，实验证明其优于基线方法。**

- **链接: [http://arxiv.org/pdf/2509.00271v1](http://arxiv.org/pdf/2509.00271v1)**

> **作者:** Yishu Li; Xinyi Mao; Ying Yuan; Kyutae Sim; Ben Eisner; David Held
>
> **备注:** CoRL 2025
>
> **摘要:** We introduce a novel History-Aware VErifier (HAVE) to disambiguate uncertain scenarios online by leveraging past interactions. Robots frequently encounter visually ambiguous objects whose manipulation outcomes remain uncertain until physically interacted with. While generative models alone could theoretically adapt to such ambiguity, in practice they obtain suboptimal performance in ambiguous cases, even when conditioned on action history. To address this, we propose explicitly decoupling action generation from verification: we use an unconditional diffusion-based generator to propose multiple candidate actions and employ our history-aware verifier to select the most promising action by reasoning about past interactions. Through theoretical analysis, we demonstrate that employing a verifier significantly improves expected action quality. Empirical evaluations and analysis across multiple simulated and real-world environments including articulated objects, multi-modal doors, and uneven object pick-up confirm the effectiveness of our method and improvements over baselines. Our project website is available at: https://liy1shu.github.io/HAVE_CoRL25/
>
---
#### [new 033] Safe and Efficient Lane-Changing for Autonomous Vehicles: An Improved Double Quintic Polynomial Approach with Time-to-Collision Evaluation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对自动驾驶车辆在混合交通中的安全高效车道变更问题，提出改进的双五次多项式方法，整合时间到碰撞评估机制，实现实时安全轨迹生成，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.00582v1](http://arxiv.org/pdf/2509.00582v1)**

> **作者:** Rui Bai; Rui Xu; Teng Rui; Jiale Liu; Qi Wei Oung; Hoi Leong Lee; Zhen Tian; Fujiang Yuan
>
> **摘要:** Autonomous driving technology has made significant advancements in recent years, yet challenges remain in ensuring safe and comfortable interactions with human-driven vehicles (HDVs), particularly during lane-changing maneuvers. This paper proposes an improved double quintic polynomial approach for safe and efficient lane-changing in mixed traffic environments. The proposed method integrates a time-to-collision (TTC) based evaluation mechanism directly into the trajectory optimization process, ensuring that the ego vehicle proactively maintains a safe gap from surrounding HDVs throughout the maneuver. The framework comprises state estimation for both the autonomous vehicle (AV) and HDVs, trajectory generation using double quintic polynomials, real-time TTC computation, and adaptive trajectory evaluation. To the best of our knowledge, this is the first work to embed an analytic TTC penalty directly into the closed-form double-quintic polynomial solver, enabling real-time safety-aware trajectory generation without post-hoc validation. Extensive simulations conducted under diverse traffic scenarios demonstrate the safety, efficiency, and comfort of the proposed approach compared to conventional methods such as quintic polynomials, Bezier curves, and B-splines. The results highlight that the improved method not only avoids collisions but also ensures smooth transitions and adaptive decision-making in dynamic environments. This work bridges the gap between model-based and adaptive trajectory planning approaches, offering a stable solution for real-world autonomous driving applications.
>
---
#### [new 034] FLUID: A Fine-Grained Lightweight Urban Signalized-Intersection Dataset of Dense Conflict Trajectories
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出FLUID数据集，解决现有交通轨迹数据在场景代表性和信息丰富度不足的问题。通过无人机采集城市信号交叉口密集冲突轨迹，构建包含20,000+交通参与者、高时空精度的细粒度数据集，用于交通行为分析与自动驾驶研究。**

- **链接: [http://arxiv.org/pdf/2509.00497v1](http://arxiv.org/pdf/2509.00497v1)**

> **作者:** Yiyang Chen; Zhigang Wu; Guohong Zheng; Xuesong Wu; Liwen Xu; Haoyuan Tang; Zhaocheng He; Haipeng Zeng
>
> **备注:** 26 pages, 14 figures
>
> **摘要:** The trajectory data of traffic participants (TPs) is a fundamental resource for evaluating traffic conditions and optimizing policies, especially at urban intersections. Although data acquisition using drones is efficient, existing datasets still have limitations in scene representativeness, information richness, and data fidelity. This study introduces FLUID, comprising a fine-grained trajectory dataset that captures dense conflicts at typical urban signalized intersections, and a lightweight, full-pipeline framework for drone-based trajectory processing. FLUID covers three distinct intersection types, with approximately 5 hours of recording time and featuring over 20,000 TPs across 8 categories. Notably, the dataset averages two vehicle conflicts per minute, involving roughly 25% of all motor vehicles. FLUID provides comprehensive data, including trajectories, traffic signals, maps, and raw videos. Comparison with the DataFromSky platform and ground-truth measurements validates its high spatio-temporal accuracy. Through a detailed classification of motor vehicle conflicts and violations, FLUID reveals a diversity of interactive behaviors, demonstrating its value for human preference mining, traffic behavior modeling, and autonomous driving research.
>
---
#### [new 035] Novel bio-inspired soft actuators for upper-limb exoskeletons: design, fabrication and feasibility study
- **分类: cs.RO**

- **简介: 该论文设计两种生物启发软执行器（LISPER/SCASPER）用于上肢外骨骼，解决现有软机器人响应慢、输出力低及控制精度不足问题，通过建模与实验验证其性能。**

- **链接: [http://arxiv.org/pdf/2509.01145v1](http://arxiv.org/pdf/2509.01145v1)**

> **作者:** Haiyun Zhang; Gabrielle Naquila; Jung Hyun Bae; Zonghuan Wu; Ashwin Hingwe; Ashish Deshpande
>
> **摘要:** Soft robots have been increasingly utilized as sophisticated tools in physical rehabilitation, particularly for assisting patients with neuromotor impairments. However, many soft robotics for rehabilitation applications are characterized by limitations such as slow response times, restricted range of motion, and low output force. There are also limited studies on the precise position and force control of wearable soft actuators. Furthermore, not many studies articulate how bellow-structured actuator designs quantitatively contribute to the robots' capability. This study introduces a paradigm of upper limb soft actuator design. This paradigm comprises two actuators: the Lobster-Inspired Silicone Pneumatic Robot (LISPER) for the elbow and the Scallop-Shaped Pneumatic Robot (SCASPER) for the shoulder. LISPER is characterized by higher bandwidth, increased output force/torque, and high linearity. SCASPER is characterized by high output force/torque and simplified fabrication processes. Comprehensive analytical models that describe the relationship between pressure, bending angles, and output force for both actuators were presented so the geometric configuration of the actuators can be set to modify the range of motion and output forces. The preliminary test on a dummy arm is conducted to test the capability of the actuators.
>
---
#### [new 036] Speculative Design of Equitable Robotics: Queer Fictions and Futures
- **分类: cs.RO; cs.CY; cs.HC; I.2.9; J.5; K.4.2**

- **简介: 该论文通过设想设计，探讨为LGBTQ+群体创建公平机器人，提出三个设计提案，讨论伦理问题，并展望未来。**

- **链接: [http://arxiv.org/pdf/2509.01643v1](http://arxiv.org/pdf/2509.01643v1)**

> **作者:** Minja Axelsson
>
> **备注:** Accepted at the British Computer Society's Special Interest Group in Human Computer Interaction Conference (BCS HCI 2025), Futures track. 5 pages, no figures
>
> **摘要:** This paper examines the speculative topic of equitable robots through an exploratory essay format. It focuses specifically on robots by and for LGBTQ+ populations. It aims to provoke thought and conversations in the field about what aspirational queer robotics futures may look like, both in the arts and sciences. First, it briefly reviews the state-of-the-art of queer robotics in fiction and science, drawing together threads from each. Then, it discusses queering robots through three speculative design proposals for queer robot roles: 1) reflecting the queerness of their ''in-group'' queer users, building and celebrating ''in-group'' identity, 2) a new kind of queer activism by implementing queer robot identity performance to interact with ''out-group'' users, with a goal of reducing bigotry through familiarisation, and 3) a network of queer-owned robots, through which the community could reach each other, and distribute and access important resources. The paper then questions whether robots should be queered, and what ethical implications this raises. Finally, the paper makes suggestions for what aspirational queer robotics futures may look like, and what would be required to get there.
>
---
#### [new 037] Classification of Vision-Based Tactile Sensors: A Review
- **分类: cs.RO**

- **简介: 该论文旨在综述基于视觉的触觉传感器（VBTS）分类方法，解决其多样化设计带来的分类混乱问题。通过提出基于标记与强度的双原理分类框架，系统比较四种传感器类型，分析硬件特性与数据解读方法，揭示技术挑战与未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.02478v1](http://arxiv.org/pdf/2509.02478v1)**

> **作者:** Haoran Li; Yijiong Lin; Chenghua Lu; Max Yang; Efi Psomopoulou; Nathan F Lepora
>
> **备注:** 15 pages
>
> **摘要:** Vision-based tactile sensors (VBTS) have gained widespread application in robotic hands, grippers and prosthetics due to their high spatial resolution, low manufacturing costs, and ease of customization. While VBTSs have common design features, such as a camera module, they can differ in a rich diversity of sensing principles, material compositions, multimodal approaches, and data interpretation methods. Here, we propose a novel classification of VBTS that categorizes the technology into two primary sensing principles based on the underlying transduction of contact into a tactile image: the Marker-Based Transduction Principle and the Intensity-Based Transduction Principle. Marker-Based Transduction interprets tactile information by detecting marker displacement and changes in marker density. In contrast, Intensity-Based Transduction maps external disturbances with variations in pixel values. Depending on the design of the contact module, Marker-Based Transduction can be further divided into two subtypes: Simple Marker-Based (SMB) and Morphological Marker-Based (MMB) mechanisms. Similarly, the Intensity-Based Transduction Principle encompasses the Reflective Layer-based (RLB) and Transparent Layer-Based (TLB) mechanisms. This paper provides a comparative study of the hardware characteristics of these four types of sensors including various combination types, and discusses the commonly used methods for interpreting tactile information. This~comparison reveals some current challenges faced by VBTS technology and directions for future research.
>
---
#### [new 038] MoTo: A Zero-shot Plug-in Interaction-aware Navigation for General Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出MoTo模块，解决移动操作中任务泛化与零样本挑战，通过交互感知导航和视觉语言模型框架，结合基础模型实现无需额外数据的移动操作。**

- **链接: [http://arxiv.org/pdf/2509.01658v1](http://arxiv.org/pdf/2509.01658v1)**

> **作者:** Zhenyu Wu; Angyuan Ma; Xiuwei Xu; Hang Yin; Yinan Liang; Ziwei Wang; Jiwen Lu; Haibin Yan
>
> **备注:** Accepted to CoRL 2025. Project Page: https://gary3410.github.io/MoTo/
>
> **摘要:** Mobile manipulation stands as a core challenge in robotics, enabling robots to assist humans across varied tasks and dynamic daily environments. Conventional mobile manipulation approaches often struggle to generalize across different tasks and environments due to the lack of large-scale training. However, recent advances in manipulation foundation models demonstrate impressive generalization capability on a wide range of fixed-base manipulation tasks, which are still limited to a fixed setting. Therefore, we devise a plug-in module named MoTo, which can be combined with any off-the-shelf manipulation foundation model to empower them with mobile manipulation ability. Specifically, we propose an interaction-aware navigation policy to generate robot docking points for generalized mobile manipulation. To enable zero-shot ability, we propose an interaction keypoints framework via vision-language models (VLM) under multi-view consistency for both target object and robotic arm following instructions, where fixed-base manipulation foundation models can be employed. We further propose motion planning objectives for the mobile base and robot arm, which minimize the distance between the two keypoints and maintain the physical feasibility of trajectories. In this way, MoTo guides the robot to move to the docking points where fixed-base manipulation can be successfully performed, and leverages VLM generation and trajectory optimization to achieve mobile manipulation in a zero-shot manner, without any requirement on mobile manipulation expert data. Extensive experimental results on OVMM and real-world demonstrate that MoTo achieves success rates of 2.68% and 16.67% higher than the state-of-the-art mobile manipulation methods, respectively, without requiring additional training data.
>
---
#### [new 039] Hybrid Perception and Equivariant Diffusion for Robust Multi-Node Rebar Tying
- **分类: cs.RO; cs.CV**

- **简介: 论文提出混合感知与等变扩散模型，用于自动化多节点钢筋绑扎。解决密集环境中姿态估计与碰撞避障问题，通过几何特征提取和少量示范训练，实现高效、鲁棒的绑扎，减少数据需求，提升施工安全与效率。**

- **链接: [http://arxiv.org/pdf/2509.00065v1](http://arxiv.org/pdf/2509.00065v1)**

> **作者:** Zhitao Wang; Yirong Xiong; Roberto Horowitz; Yanke Wang; Yuxing Han
>
> **备注:** Accepted by The IEEE International Conference on Automation Science and Engineering (CASE) 2025
>
> **摘要:** Rebar tying is a repetitive but critical task in reinforced concrete construction, typically performed manually at considerable ergonomic risk. Recent advances in robotic manipulation hold the potential to automate the tying process, yet face challenges in accurately estimating tying poses in congested rebar nodes. In this paper, we introduce a hybrid perception and motion planning approach that integrates geometry-based perception with Equivariant Denoising Diffusion on SE(3) (Diffusion-EDFs) to enable robust multi-node rebar tying with minimal training data. Our perception module utilizes density-based clustering (DBSCAN), geometry-based node feature extraction, and principal component analysis (PCA) to segment rebar bars, identify rebar nodes, and estimate orientation vectors for sequential ranking, even in complex, unstructured environments. The motion planner, based on Diffusion-EDFs, is trained on as few as 5-10 demonstrations to generate sequential end-effector poses that optimize collision avoidance and tying efficiency. The proposed system is validated on various rebar meshes, including single-layer, multi-layer, and cluttered configurations, demonstrating high success rates in node detection and accurate sequential tying. Compared with conventional approaches that rely on large datasets or extensive manual parameter tuning, our method achieves robust, efficient, and adaptable multi-node tying while significantly reducing data requirements. This result underscores the potential of hybrid perception and diffusion-driven planning to enhance automation in on-site construction tasks, improving both safety and labor efficiency.
>
---
#### [new 040] Systematic Evaluation of Trade-Offs in Motion Planning Algorithms for Optimal Industrial Robotic Work Cell Design
- **分类: cs.RO**

- **简介: 该论文针对工业机器人工作单元设计中的双层优化问题，研究运动规划权衡对整体性能的影响，提出评估指标并进行仿真分析，以平衡计算复杂度与解的质量。**

- **链接: [http://arxiv.org/pdf/2509.02146v1](http://arxiv.org/pdf/2509.02146v1)**

> **作者:** G. de Mathelin; C. Hartl-Nesic; A. Kugi
>
> **备注:** This work has been accepted to IFAC for publication under a Creative Commons Licence CC-BY-NC-ND
>
> **摘要:** The performance of industrial robotic work cells depends on optimizing various hyperparameters referring to the cell layout, such as robot base placement, tool placement, and kinematic design. Achieving this requires a bilevel optimization approach, where the high-level optimization adjusts these hyperparameters, and the low-level optimization computes robot motions. However, computing the optimal robot motion is computationally infeasible, introducing trade-offs in motion planning to make the problem tractable. These trade-offs significantly impact the overall performance of the bilevel optimization, but their effects still need to be systematically evaluated. In this paper, we introduce metrics to assess these trade-offs regarding optimality, time gain, robustness, and consistency. Through extensive simulation studies, we investigate how simplifications in motion-level optimization affect the high-level optimization outcomes, balancing computational complexity with solution quality. The proposed algorithms are applied to find the time-optimal kinematic design for a modular robot in two palletization scenarios.
>
---
#### [new 041] Embodied Spatial Intelligence: from Implicit Scene Modeling to Spatial Reasoning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出具身空间智能，解决机器人基于自然语言指令的感知与行动问题。通过隐式场景建模与空间推理方法，包括自监督校准、深度生成、导航基准及状态反馈机制，提升语言-行动协同能力。**

- **链接: [http://arxiv.org/pdf/2509.00465v1](http://arxiv.org/pdf/2509.00465v1)**

> **作者:** Jiading Fang
>
> **摘要:** This thesis introduces "Embodied Spatial Intelligence" to address the challenge of creating robots that can perceive and act in the real world based on natural language instructions. To bridge the gap between Large Language Models (LLMs) and physical embodiment, we present contributions on two fronts: scene representation and spatial reasoning. For perception, we develop robust, scalable, and accurate scene representations using implicit neural models, with contributions in self-supervised camera calibration, high-fidelity depth field generation, and large-scale reconstruction. For spatial reasoning, we enhance the spatial capabilities of LLMs by introducing a novel navigation benchmark, a method for grounding language in 3D, and a state-feedback mechanism to improve long-horizon decision-making. This work lays a foundation for robots that can robustly perceive their surroundings and intelligently act upon complex, language-based commands.
>
---
#### [new 042] Sem-RaDiff: Diffusion-Based 3D Radar Semantic Perception in Cluttered Agricultural Environments
- **分类: cs.RO**

- **简介: 该论文提出Sem-RaDiff框架，解决农业环境中雷达传感器污染导致的3D语义感知退化问题。通过并行帧积累、扩散模型分层学习和稀疏网络设计，实现细结构重建与分类，提升性能并降低计算成本。**

- **链接: [http://arxiv.org/pdf/2509.02283v1](http://arxiv.org/pdf/2509.02283v1)**

> **作者:** Ruibin Zhang; Fei Gao
>
> **摘要:** Accurate and robust environmental perception is crucial for robot autonomous navigation. While current methods typically adopt optical sensors (e.g., camera, LiDAR) as primary sensing modalities, their susceptibility to visual occlusion often leads to degraded performance or complete system failure. In this paper, we focus on agricultural scenarios where robots are exposed to the risk of onboard sensor contamination. Leveraging radar's strong penetration capability, we introduce a radar-based 3D environmental perception framework as a viable alternative. It comprises three core modules designed for dense and accurate semantic perception: 1) Parallel frame accumulation to enhance signal-to-noise ratio of radar raw data. 2) A diffusion model-based hierarchical learning framework that first filters radar sidelobe artifacts then generates fine-grained 3D semantic point clouds. 3) A specifically designed sparse 3D network optimized for processing large-scale radar raw data. We conducted extensive benchmark comparisons and experimental evaluations on a self-built dataset collected in real-world agricultural field scenes. Results demonstrate that our method achieves superior structural and semantic prediction performance compared to existing methods, while simultaneously reducing computational and memory costs by 51.3% and 27.5%, respectively. Furthermore, our approach achieves complete reconstruction and accurate classification of thin structures such as poles and wires-which existing methods struggle to perceive-highlighting its potential for dense and accurate 3D radar perception.
>
---
#### [new 043] TopoNav: Topological Graphs as a Key Enabler for Advanced Object Navigation
- **分类: cs.RO**

- **简介: 论文提出TopoNav框架，通过拓扑图解决ObjectNav中的长任务和动态场景内存管理问题，提升导航性能。**

- **链接: [http://arxiv.org/pdf/2509.01364v1](http://arxiv.org/pdf/2509.01364v1)**

> **作者:** Peiran Liu; Qiang Zhang; Daojie Peng; Lingfeng Zhang; Yihao Qin; Hang Zhou; Jun Ma; Renjing Xu; Yiding Ji
>
> **摘要:** Object Navigation (ObjectNav) has made great progress with large language models (LLMs), but still faces challenges in memory management, especially in long-horizon tasks and dynamic scenes. To address this, we propose TopoNav, a new framework that leverages topological structures as spatial memory. By building and updating a topological graph that captures scene connections, adjacency, and semantic meaning, TopoNav helps agents accumulate spatial knowledge over time, retrieve key information, and reason effectively toward distant goals. Our experiments show that TopoNav achieves state-of-the-art performance on benchmark ObjectNav datasets, with higher success rates and more efficient paths. It particularly excels in diverse and complex environments, as it connects temporary visual inputs with lasting spatial understanding.
>
---
#### [new 044] Analyzing Reluctance to Ask for Help When Cooperating With Robots: Insights to Integrate Artificial Agents in HRC
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究人类在人机协作中寻求帮助的犹豫现象，分析设计决策（如人类/AI辅助、按需/主动帮助）对情绪、生产力和偏好的影响，旨在优化人工代理在协作任务中的集成设计。**

- **链接: [http://arxiv.org/pdf/2509.01450v1](http://arxiv.org/pdf/2509.01450v1)**

> **作者:** Ane San Martin; Michael Hagenow; Julie Shah; Johan Kildal; Elena Lazkano
>
> **备注:** 8 pages, 5 figures. Accepted for IEEE RO-MAN 2025
>
> **摘要:** As robot technology advances, collaboration between humans and robots will become more prevalent in industrial tasks. When humans run into issues in such scenarios, a likely future involves relying on artificial agents or robots for aid. This study identifies key aspects for the design of future user-assisting agents. We analyze quantitative and qualitative data from a user study examining the impact of on-demand assistance received from a remote human in a human-robot collaboration (HRC) assembly task. We study scenarios in which users require help and we assess their experiences in requesting and receiving assistance. Additionally, we investigate participants' perceptions of future non-human assisting agents and whether assistance should be on-demand or unsolicited. Through a user study, we analyze the impact that such design decisions (human or artificial assistant, on-demand or unsolicited help) can have on elicited emotional responses, productivity, and preferences of humans engaged in HRC tasks.
>
---
#### [new 045] TARA: A Low-Cost 3D-Printed Robotic Arm for Accessible Robotics Education
- **分类: cs.RO**

- **简介: 论文设计低成本3D打印机械臂TARA，解决机器人教育高成本问题。提供开源资源支持构建与定制，实验验证基础操作能力，强调教育可重复性与扩展性。**

- **链接: [http://arxiv.org/pdf/2509.01043v1](http://arxiv.org/pdf/2509.01043v1)**

> **作者:** Thays Leach Mitre
>
> **备注:** 6 pages, 5 figures. Preprint submission
>
> **摘要:** The high cost of robotic platforms limits students' ability to gain practical skills directly applicable in real-world scenarios. To address this challenge, this paper presents TARA, a low-cost, 3D-printed robotic arm designed for accessible robotics education. TARA includes an open-source repository with design files, assembly instructions, and baseline code, enabling users to build and customize the platform. The system balances affordability and functionality, offering a highly capable robotic arm for approximately 200 USD, significantly lower than industrial systems that often cost thousands of dollars. Experimental validation confirmed accurate performance in basic manipulation tasks. Rather than focusing on performance benchmarking, this work prioritizes educational reproducibility, providing a platform that students and educators can reliably replicate and extend.
>
---
#### [new 046] Fail2Progress: Learning from Real-World Robot Failures with Stein Variational Inference
- **分类: cs.RO**

- **简介: 该论文提出Fail2Progress方法，利用Stein变分推断生成失败场景数据，提升机器人长期操作任务的鲁棒性，通过模拟和实验验证效果优于基线。**

- **链接: [http://arxiv.org/pdf/2509.01746v1](http://arxiv.org/pdf/2509.01746v1)**

> **作者:** Yixuan Huang; Novella Alvina; Mohanraj Devendran Shanthi; Tucker Hermans
>
> **备注:** Project page: sites.google.com/view/fail2progress. 25 pages, 8 figures. Accepted to the Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Skill effect models for long-horizon manipulation tasks are prone to failures in conditions not covered by training data distributions. Therefore, enabling robots to reason about and learn from failures is necessary. We investigate the problem of efficiently generating a dataset targeted to observed failures. After fine-tuning a skill effect model on this dataset, we evaluate the extent to which the model can recover from failures and minimize future failures. We propose Fail2Progress, an approach that leverages Stein variational inference to generate multiple simulation environments in parallel, enabling efficient data sample generation similar to observed failures. Our method is capable of handling several challenging mobile manipulation tasks, including transporting multiple objects, organizing a constrained shelf, and tabletop organization. Through large-scale simulation and real-world experiments, we demonstrate that our approach excels at learning from failures across different numbers of objects. Furthermore, we show that Fail2Progress outperforms several baselines.
>
---
#### [new 047] Reinforcement Learning of Dolly-In Filming Using a Ground-Based Robot
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出用强化学习控制地面机器人实现自动推轨拍摄，解决传统控制精度与稳定性问题，通过联合控制策略与实验证明方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.00564v1](http://arxiv.org/pdf/2509.00564v1)**

> **作者:** Philip Lorimer; Jack Saunders; Alan Hunter; Wenbin Li
>
> **备注:** Authors' accepted manuscript (IROS 2024, Abu Dhabi, Oct 14-18, 2024). Please cite the version of record: DOI 10.1109/IROS58592.2024.10802717. 8 pages
>
> **摘要:** Free-roaming dollies enhance filmmaking with dynamic movement, but challenges in automated camera control remain unresolved. Our study advances this field by applying Reinforcement Learning (RL) to automate dolly-in shots using free-roaming ground-based filming robots, overcoming traditional control hurdles. We demonstrate the effectiveness of combined control for precise film tasks by comparing it to independent control strategies. Our robust RL pipeline surpasses traditional Proportional-Derivative controller performance in simulation and proves its efficacy in real-world tests on a modified ROSBot 2.0 platform equipped with a camera turret. This validates our approach's practicality and sets the stage for further research in complex filming scenarios, contributing significantly to the fusion of technology with cinematic creativity. This work presents a leap forward in the field and opens new avenues for research and development, effectively bridging the gap between technological advancement and creative filmmaking.
>
---
#### [new 048] TReF-6: Inferring Task-Relevant Frames from a Single Demonstration for One-Shot Skill Generalization
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对机器人单次示范技能泛化难题，提出TReF-6方法。通过几何分析确定轨迹影响点构建局部坐标系，结合视觉模型实现任务相关帧推理，解决空间表示不足问题，支持跨场景的一次性技能迁移。**

- **链接: [http://arxiv.org/pdf/2509.00310v1](http://arxiv.org/pdf/2509.00310v1)**

> **作者:** Yuxuan Ding; Shuangge Wang; Tesca Fitzgerald
>
> **摘要:** Robots often struggle to generalize from a single demonstration due to the lack of a transferable and interpretable spatial representation. In this work, we introduce TReF-6, a method that infers a simplified, abstracted 6DoF Task-Relevant Frame from a single trajectory. Our approach identifies an influence point purely from the trajectory geometry to define the origin for a local frame, which serves as a reference for parameterizing a Dynamic Movement Primitive (DMP). This influence point captures the task's spatial structure, extending the standard DMP formulation beyond start-goal imitation. The inferred frame is semantically grounded via a vision-language model and localized in novel scenes by Grounded-SAM, enabling functionally consistent skill generalization. We validate TReF-6 in simulation and demonstrate robustness to trajectory noise. We further deploy an end-to-end pipeline on real-world manipulation tasks, showing that TReF-6 supports one-shot imitation learning that preserves task intent across diverse object configurations.
>
---
#### [new 049] Generative Visual Foresight Meets Task-Agnostic Pose Estimation in Robotic Table-Top Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GVF-TAPE框架，解决机器人桌面操作中跨任务泛化与实时控制问题。通过结合生成视觉预知与任务无关位姿估计，实现无需任务特定数据的可扩展操作，提升复杂环境下的适应性与可靠性。**

- **链接: [http://arxiv.org/pdf/2509.00361v1](http://arxiv.org/pdf/2509.00361v1)**

> **作者:** Chuye Zhang; Xiaoxiong Zhang; Wei Pan; Linfang Zheng; Wei Zhang
>
> **备注:** 9th Conference on Robot Learning (CoRL 2025), Seoul, Korea
>
> **摘要:** Robotic manipulation in unstructured environments requires systems that can generalize across diverse tasks while maintaining robust and reliable performance. We introduce {GVF-TAPE}, a closed-loop framework that combines generative visual foresight with task-agnostic pose estimation to enable scalable robotic manipulation. GVF-TAPE employs a generative video model to predict future RGB-D frames from a single side-view RGB image and a task description, offering visual plans that guide robot actions. A decoupled pose estimation model then extracts end-effector poses from the predicted frames, translating them into executable commands via low-level controllers. By iteratively integrating video foresight and pose estimation in a closed loop, GVF-TAPE achieves real-time, adaptive manipulation across a broad range of tasks. Extensive experiments in both simulation and real-world settings demonstrate that our approach reduces reliance on task-specific action data and generalizes effectively, providing a practical and scalable solution for intelligent robotic systems.
>
---
#### [new 050] Language-Guided Long Horizon Manipulation with LLM-based Planning and Visual Perception
- **分类: cs.RO**

- **简介: 论文提出基于LLM和VLM的统一框架，解决语言引导下布料折叠等可变形物体操作中的高自由度、复杂动态及视觉-语言对齐问题，实现多步骤任务的高效规划与执行，实验验证其在模拟和真实环境中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.02324v1](http://arxiv.org/pdf/2509.02324v1)**

> **作者:** Changshi Zhou; Haichuan Xu; Ningquan Gu; Zhipeng Wang; Bin Cheng; Pengpeng Zhang; Yanchao Dong; Mitsuhiro Hayashibe; Yanmin Zhou; Bin He
>
> **摘要:** Language-guided long-horizon manipulation of deformable objects presents significant challenges due to high degrees of freedom, complex dynamics, and the need for accurate vision-language grounding. In this work, we focus on multi-step cloth folding, a representative deformable-object manipulation task that requires both structured long-horizon planning and fine-grained visual perception. To this end, we propose a unified framework that integrates a Large Language Model (LLM)-based planner, a Vision-Language Model (VLM)-based perception system, and a task execution module. Specifically, the LLM-based planner decomposes high-level language instructions into low-level action primitives, bridging the semantic-execution gap, aligning perception with action, and enhancing generalization. The VLM-based perception module employs a SigLIP2-driven architecture with a bidirectional cross-attention fusion mechanism and weight-decomposed low-rank adaptation (DoRA) fine-tuning to achieve language-conditioned fine-grained visual grounding. Experiments in both simulation and real-world settings demonstrate the method's effectiveness. In simulation, it outperforms state-of-the-art baselines by 2.23, 1.87, and 33.3 on seen instructions, unseen instructions, and unseen tasks, respectively. On a real robot, it robustly executes multi-step folding sequences from language instructions across diverse cloth materials and configurations, demonstrating strong generalization in practical scenarios. Project page: https://language-guided.netlify.app/
>
---
#### [new 051] U2UData-2: A Scalable Swarm UAVs Autonomous Flight Dataset for Long-horizon Tasks
- **分类: cs.RO; cs.AI; cs.MA; cs.MM**

- **简介: 论文提出U2UData-2数据集及平台，解决长周期（LH）任务中无人机群自主飞行数据不足问题。通过15架无人机采集多场景数据，支持动态目标适应与算法闭环验证，推动低空经济应用。**

- **链接: [http://arxiv.org/pdf/2509.00055v1](http://arxiv.org/pdf/2509.00055v1)**

> **作者:** Tongtong Feng; Xin Wang; Feilin Han; Leping Zhang; Wenwu Zhu
>
> **摘要:** Swarm UAV autonomous flight for Long-Horizon (LH) tasks is crucial for advancing the low-altitude economy. However, existing methods focus only on specific basic tasks due to dataset limitations, failing in real-world deployment for LH tasks. LH tasks are not mere concatenations of basic tasks, requiring handling long-term dependencies, maintaining persistent states, and adapting to dynamic goal shifts. This paper presents U2UData-2, the first large-scale swarm UAV autonomous flight dataset for LH tasks and the first scalable swarm UAV data online collection and algorithm closed-loop verification platform. The dataset is captured by 15 UAVs in autonomous collaborative flights for LH tasks, comprising 12 scenes, 720 traces, 120 hours, 600 seconds per trajectory, 4.32M LiDAR frames, and 12.96M RGB frames. This dataset also includes brightness, temperature, humidity, smoke, and airflow values covering all flight routes. The platform supports the customization of simulators, UAVs, sensors, flight algorithms, formation modes, and LH tasks. Through a visual control window, this platform allows users to collect customized datasets through one-click deployment online and to verify algorithms by closed-loop simulation. U2UData-2 also introduces an LH task for wildlife conservation and provides comprehensive benchmarks with 9 SOTA models. U2UData-2 can be found at https://fengtt42.github.io/U2UData-2/.
>
---
#### [new 052] Hybrid Autonomy Framework for a Future Mars Science Helicopter
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出混合自主框架，用于火星科学直升机长距离探索，解决通信延迟与任务复杂性问题。整合FSM与BT实现动态行为调整，通过仿真与测试验证框架的鲁棒性与适应性，支持多系统集成。**

- **链接: [http://arxiv.org/pdf/2509.01980v1](http://arxiv.org/pdf/2509.01980v1)**

> **作者:** Luca Di Pierno; Robert Hewitt; Stephan Weiss; Roland Brockers
>
> **备注:** 8 pages, IEEE CASE 2025 Conference
>
> **摘要:** Autonomous aerial vehicles, such as NASA's Ingenuity, enable rapid planetary surface exploration beyond the reach of ground-based robots. Thus, NASA is studying a Mars Science Helicopter (MSH), an advanced concept capable of performing long-range science missions and autonomously navigating challenging Martian terrain. Given significant Earth-Mars communication delays and mission complexity, an advanced autonomy framework is required to ensure safe and efficient operation by continuously adapting behavior based on mission objectives and real-time conditions, without human intervention. This study presents a deterministic high-level control framework for aerial exploration, integrating a Finite State Machine (FSM) with Behavior Trees (BTs) to achieve a scalable, robust, and computationally efficient autonomy solution for critical scenarios like deep space exploration. In this paper we outline key capabilities of a possible MSH and detail the FSM-BT hybrid autonomy framework which orchestrates them to achieve the desired objectives. Monte Carlo simulations and real field tests validate the framework, demonstrating its robustness and adaptability to both discrete events and real-time system feedback. These inputs trigger state transitions or dynamically adjust behavior execution, enabling reactive and context-aware responses. The framework is middleware-agnostic, supporting integration with systems like F-Prime and extending beyond aerial robotics.
>
---
#### [new 053] Disentangled Multi-Context Meta-Learning: Unlocking robust and Generalized Task Learning
- **分类: cs.RO**

- **简介: 论文提出解耦多上下文元学习框架，解决传统方法中任务因素混合导致的泛化与鲁棒性不足问题，通过分离上下文向量提升跨任务共享与适应能力。**

- **链接: [http://arxiv.org/pdf/2509.01297v1](http://arxiv.org/pdf/2509.01297v1)**

> **作者:** Seonsoo Kim; Jun-Gill Kang; Taehong Kim; Seongil Hong
>
> **备注:** Accepted to The Conference on Robot Learning (CoRL) 2025 Project Page: seonsoo-p1.github.io/DMCM
>
> **摘要:** In meta-learning and its downstream tasks, many methods rely on implicit adaptation to task variations, where multiple factors are mixed together in a single entangled representation. This makes it difficult to interpret which factors drive performance and can hinder generalization. In this work, we introduce a disentangled multi-context meta-learning framework that explicitly assigns each task factor to a distinct context vector. By decoupling these variations, our approach improves robustness through deeper task understanding and enhances generalization by enabling context vector sharing across tasks with shared factors. We evaluate our approach in two domains. First, on a sinusoidal regression task, our model outperforms baselines on out-of-distribution tasks and generalizes to unseen sine functions by sharing context vectors associated with shared amplitudes or phase shifts. Second, in a quadruped robot locomotion task, we disentangle the robot-specific properties and the characteristics of the terrain in the robot dynamics model. By transferring disentangled context vectors acquired from the dynamics model into reinforcement learning, the resulting policy achieves improved robustness under out-of-distribution conditions, surpassing the baselines that rely on a single unified context. Furthermore, by effectively sharing context, our model enables successful sim-to-real policy transfer to challenging terrains with out-of-distribution robot-specific properties, using just 20 seconds of real data from flat terrain, a result not achievable with single-task adaptation.
>
---
#### [new 054] Learning Dolly-In Filming From Demonstration Using a Ground-Based Robot
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出基于演示的生成对抗模仿学习方法，解决传统强化学习需手动设计奖励函数的问题，实现地面机器人自动化跟拍镜头控制，无需奖励函数即可在模拟与真实场景中提升拍摄性能。**

- **链接: [http://arxiv.org/pdf/2509.00574v1](http://arxiv.org/pdf/2509.00574v1)**

> **作者:** Philip Lorimer; Alan Hunter; Wenbin Li
>
> **备注:** Preprint; under double-anonymous review. 6 pages
>
> **摘要:** Cinematic camera control demands a balance of precision and artistry - qualities that are difficult to encode through handcrafted reward functions. While reinforcement learning (RL) has been applied to robotic filmmaking, its reliance on bespoke rewards and extensive tuning limits creative usability. We propose a Learning from Demonstration (LfD) approach using Generative Adversarial Imitation Learning (GAIL) to automate dolly-in shots with a free-roaming, ground-based filming robot. Expert trajectories are collected via joystick teleoperation in simulation, capturing smooth, expressive motion without explicit objective design. Trained exclusively on these demonstrations, our GAIL policy outperforms a PPO baseline in simulation, achieving higher rewards, faster convergence, and lower variance. Crucially, it transfers directly to a real-world robot without fine-tuning, achieving more consistent framing and subject alignment than a prior TD3-based method. These results show that LfD offers a robust, reward-free alternative to RL in cinematic domains, enabling real-time deployment with minimal technical effort. Our pipeline brings intuitive, stylized camera control within reach of creative professionals, bridging the gap between artistic intent and robotic autonomy.
>
---
#### [new 055] AI-Driven Marine Robotics: Emerging Trends in Underwater Perception and Ecosystem Monitoring
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文分析水下AI在海洋生态监测中的应用，解决气候变化导致的生态压力问题。通过识别环境需求、数据民主化及研究迁移三大驱动因素，探讨水下感知挑战推动的AI技术创新，提出弱监督学习等方法，并拓展至通用计算机视觉与环境监测领域。**

- **链接: [http://arxiv.org/pdf/2509.01878v1](http://arxiv.org/pdf/2509.01878v1)**

> **作者:** Scarlett Raine; Tobias Fischer
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Marine ecosystems face increasing pressure due to climate change, driving the need for scalable, AI-powered monitoring solutions. This paper examines the rapid emergence of underwater AI as a major research frontier and analyzes the factors that have transformed marine perception from a niche application into a catalyst for AI innovation. We identify three convergent drivers: environmental necessity for ecosystem-scale monitoring, democratization of underwater datasets through citizen science platforms, and researcher migration from saturated terrestrial computer vision domains. Our analysis reveals how unique underwater challenges - turbidity, cryptic species detection, expert annotation bottlenecks, and cross-ecosystem generalization - are driving fundamental advances in weakly supervised learning, open-set recognition, and robust perception under degraded conditions. We survey emerging trends in datasets, scene understanding and 3D reconstruction, highlighting the paradigm shift from passive observation toward AI-driven, targeted intervention capabilities. The paper demonstrates how underwater constraints are pushing the boundaries of foundation models, self-supervised learning, and perception, with methodological innovations that extend far beyond marine applications to benefit general computer vision, robotics, and environmental monitoring.
>
---
#### [new 056] Gray-Box Computed Torque Control for Differential-Drive Mobile Robot Tracking
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对差速驱动移动机器人跟踪控制问题，提出融合灰盒CTC与DRL的算法，通过TD3优化控制器参数，约束物理合理范围并确保闭环稳定性，提升样本效率与控制性能。**

- **链接: [http://arxiv.org/pdf/2509.00571v1](http://arxiv.org/pdf/2509.00571v1)**

> **作者:** Arman Javan Sekhavat Pishkhani
>
> **摘要:** This study presents a learning-based nonlinear algorithm for tracking control of differential-drive mobile robots. The Computed Torque Method (CTM) suffers from inaccurate knowledge of system parameters, while Deep Reinforcement Learning (DRL) algorithms are known for sample inefficiency and weak stability guarantees. The proposed method replaces the black-box policy network of a DRL agent with a gray-box Computed Torque Controller (CTC) to improve sample efficiency and ensure closed-loop stability. This approach enables finding an optimal set of controller parameters for an arbitrary reward function using only a few short learning episodes. The Twin-Delayed Deep Deterministic Policy Gradient (TD3) algorithm is used for this purpose. Additionally, some controller parameters are constrained to lie within known value ranges, ensuring the RL agent learns physically plausible values. A technique is also applied to enforce a critically damped closed-loop time response. The controller's performance is evaluated on a differential-drive mobile robot simulated in the MuJoCo physics engine and compared against the raw CTC and a conventional kinematic controller.
>
---
#### [new 057] Embodied AI in Social Spaces: Responsible and Adaptive Robots in Complex Setting - UKAIRS 2025 (Copy)
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出开发多人类多机器人系统，解决复杂动态环境中的责任与适应性问题。通过整合伦理框架、多模态感知与共设计方法，创建情感响应、情境感知的AI机器人，推动可持续、人本导向的智能系统应用。**

- **链接: [http://arxiv.org/pdf/2509.00218v1](http://arxiv.org/pdf/2509.00218v1)**

> **作者:** Aleksandra Landowska; Aislinn D Gomez Bergin; Ayodeji O. Abioye; Jayati Deshmukh; Andriana Bouadouki; Maria Wheadon; Athina Georgara; Dominic Price; Tuyen Nguyen; Shuang Ao; Lokesh Singh; Yi Long; Raffaele Miele; Joel E. Fischer; Sarvapali D. Ramchurn
>
> **摘要:** This paper introduces and overviews a multidisciplinary project aimed at developing responsible and adaptive multi-human multi-robot (MHMR) systems for complex, dynamic settings. The project integrates co-design, ethical frameworks, and multimodal sensing to create AI-driven robots that are emotionally responsive, context-aware, and aligned with the needs of diverse users. We outline the project's vision, methodology, and early outcomes, demonstrating how embodied AI can support sustainable, ethical, and human-centred futures.
>
---
#### [new 058] Enhanced Mean Field Game for Interactive Decision-Making with Varied Stylish Multi-Vehicles
- **分类: cs.RO**

- **简介: 该论文提出基于改进均值场博弈的多车交互决策框架，解决自动驾驶异构交通中的安全协同问题。通过量化驾驶风格参数、空间影响模型与安全变道算法，实现零碰撞的高效决策，优于传统博弈论方法。**

- **链接: [http://arxiv.org/pdf/2509.00981v1](http://arxiv.org/pdf/2509.00981v1)**

> **作者:** Liancheng Zheng; Zhen Tian; Yangfan He; Shuo Liu; Ke Gong; Huilin Chen; Zhihao Lin
>
> **摘要:** This paper presents an MFG-based decision-making framework for autonomous driving in heterogeneous traffic. To capture diverse human behaviors, we propose a quantitative driving style representation that maps abstract traits to parameters such as speed, safety factors, and reaction time. These parameters are embedded into the MFG through a spatial influence field model. To ensure safe operation in dense traffic, we introduce a safety-critical lane-changing algorithm that leverages dynamic safety margins, time-to-collision analysis, and multi-layered constraints. Real-world NGSIM data is employed for style calibration and empirical validation. Experimental results demonstrate zero collisions across six style combinations, two 15-vehicle scenarios, and NGSIM-based trials, consistently outperforming conventional game-theoretic baselines. Overall, our approach provides a scalable, interpretable, and behavior-aware planning framework for real-world autonomous driving applications.
>
---
#### [new 059] AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 论文提出AutoDrive-R²框架，通过CoT处理与强化学习提升VLA模型在自动驾驶中的推理与自反思能力，解决决策可解释性及行动合理性问题。采用nuScenesR²-6K数据集及GRPO算法优化奖励框架，实现高效轨迹规划。**

- **链接: [http://arxiv.org/pdf/2509.01944v1](http://arxiv.org/pdf/2509.01944v1)**

> **作者:** Zhenlong Yuan; Jing Tang; Jinguo Luo; Rui Chen; Chengxuan Qian; Lei Sun; Xiangxiang Chu; Yujun Cai; Dapeng Zhang; Shuo Li
>
> **摘要:** Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.
>
---
#### [new 060] Toward a Holistic Multi-Criteria Trajectory Evaluation Framework for Autonomous Driving in Mixed Traffic Environment
- **分类: cs.RO**

- **简介: 本文提出一种综合轨迹评估框架，整合安全、舒适和效率指标，用于自动驾驶在混合交通环境中的路径优化。通过自适应椭圆量化碰撞风险，利用PSO算法优化，结合真实实验与模拟验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.01291v1](http://arxiv.org/pdf/2509.01291v1)**

> **作者:** Nouhed Naidja; Stéphane Font; Marc Revilloud; Guillaume Sandou
>
> **摘要:** This paper presents a unified framework for the evaluation and optimization of autonomous vehicle trajectories, integrating formal safety, comfort, and efficiency criteria. An innovative geometric indicator, based on the analysis of safety zones using adaptive ellipses, is used to accurately quantify collision risks. Our method applies the Shoelace formula to compute the intersection area in the case of misaligned and time-varying configurations. Comfort is modeled using indicators centered on longitudinal and lateral jerk, while efficiency is assessed by overall travel time. These criteria are aggregated into a comprehensive objective function solved using a PSO based algorithm. The approach was successfully validated under real traffic conditions via experiments conducted in an urban intersection involving an autonomous vehicle interacting with a human-operated vehicle, and in simulation using data recorded from human driving in real traffic.
>
---
#### [new 061] MIRAGE: Multimodal Intention Recognition and Admittance-Guided Enhancement in VR-based Multi-object Teleoperation
- **分类: cs.RO; cs.HC**

- **简介: 该论文针对VR多对象遥操作中的人机交互问题，提出MIRAGE框架，结合多模态意图识别（MMIPN）与虚拟阻抗模型（VA），通过融合注视、运动及环境信息提升抓取成功率，优化运动轨迹，解决感知模糊与单模态局限，增强操作效率与自然性。**

- **链接: [http://arxiv.org/pdf/2509.01996v1](http://arxiv.org/pdf/2509.01996v1)**

> **作者:** Chi Sun; Xian Wang; Abhishek Kumar; Chengbin Cui; Lik-Hang Lee
>
> **备注:** Accepted by ISMAR 2025
>
> **摘要:** Effective human-robot interaction (HRI) in multi-object teleoperation tasks faces significant challenges due to perceptual ambiguities in virtual reality (VR) environments and the limitations of single-modality intention recognition. This paper proposes a shared control framework that combines a virtual admittance (VA) model with a Multimodal-CNN-based Human Intention Perception Network (MMIPN) to enhance teleoperation performance and user experience. The VA model employs artificial potential fields to guide operators toward target objects by adjusting admittance force and optimizing motion trajectories. MMIPN processes multimodal inputs, including gaze movement, robot motions, and environmental context, to estimate human grasping intentions, helping to overcome depth perception challenges in VR. Our user study evaluated four conditions across two factors, and the results showed that MMIPN significantly improved grasp success rates, while the VA model enhanced movement efficiency by reducing path lengths. Gaze data emerged as the most crucial input modality. These findings demonstrate the effectiveness of combining multimodal cues with implicit guidance in VR-based teleoperation, providing a robust solution for multi-object grasping tasks and enabling more natural interactions across various applications in the future.
>
---
#### [new 062] Correspondence-Free, Function-Based Sim-to-Real Learning for Deformable Surface Control
- **分类: cs.RO**

- **简介: 该论文提出一种无需点对应关系的sim-to-real学习方法，用于可变形表面控制。传统方法依赖标记点，而该方法通过神经网络联合学习变形函数空间与置信度图，支持点云或标记点输入，提升软体机器人控制的灵活性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.00060v1](http://arxiv.org/pdf/2509.00060v1)**

> **作者:** Yingjun Tian; Guoxin Fang; Renbo Su; Aoran Lyu; Neelotpal Dutta; Simeon Gill; Andrew Weightman; Charlie C. L. Wang
>
> **摘要:** This paper presents a correspondence-free, function-based sim-to-real learning method for controlling deformable freeform surfaces. Unlike traditional sim-to-real transfer methods that strongly rely on marker points with full correspondences, our approach simultaneously learns a deformation function space and a confidence map -- both parameterized by a neural network -- to map simulated shapes to their real-world counterparts. As a result, the sim-to-real learning can be conducted by input from either a 3D scanner as point clouds (without correspondences) or a motion capture system as marker points (tolerating missed markers). The resultant sim-to-real transfer can be seamlessly integrated into a neural network-based computational pipeline for inverse kinematics and shape control. We demonstrate the versatility and adaptability of our method on both vision devices and across four pneumatically actuated soft robots: a deformable membrane, a robotic mannequin, and two soft manipulators.
>
---
#### [new 063] Fault-tolerant Model Predictive Control for Spacecraft
- **分类: cs.RO**

- **简介: 该论文提出一种容错模型预测控制方法，解决航天器多执行器故障下的轨迹与设定点稳定化问题，确保安全导航，通过理论分析与实验验证其稳定性与可行性。**

- **链接: [http://arxiv.org/pdf/2509.02527v1](http://arxiv.org/pdf/2509.02527v1)**

> **作者:** Raphael Stöckner; Pedro Roque; Maria Charitidou; Dimos V. Dimarogonas
>
> **备注:** The paper has been submitted to CDC2025
>
> **摘要:** Given the cost and critical functions of satellite constellations, ensuring mission longevity and safe decommissioning is essential for space sustainability. This article presents a Model Predictive Control for spacecraft trajectory and setpoint stabilization under multiple actuation failures. The proposed solution allows us to efficiently control the faulty spacecraft enabling safe navigation towards servicing or collision-free trajectories. The proposed scheme ensures closed-loop asymptotic stability and is shown to be recursively feasible. We demonstrate its efficacy through open-source numerical results and realistic experiments using the ATMOS platform.
>
---
#### [new 064] Articulated Object Estimation in the Wild
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ArtiPoint框架，解决动态摄像机和部分观测下刚体对象3D运动估计问题，结合深度点跟踪与因子图优化，提出Arti4D数据集推动研究。**

- **链接: [http://arxiv.org/pdf/2509.01708v1](http://arxiv.org/pdf/2509.01708v1)**

> **作者:** Abdelrhman Werby; Martin Büchner; Adrian Röfer; Chenguang Huang; Wolfram Burgard; Abhinav Valada
>
> **备注:** 9th Conference on Robot Learning (CoRL), 2025
>
> **摘要:** Understanding the 3D motion of articulated objects is essential in robotic scene understanding, mobile manipulation, and motion planning. Prior methods for articulation estimation have primarily focused on controlled settings, assuming either fixed camera viewpoints or direct observations of various object states, which tend to fail in more realistic unconstrained environments. In contrast, humans effortlessly infer articulation by watching others manipulate objects. Inspired by this, we introduce ArtiPoint, a novel estimation framework that can infer articulated object models under dynamic camera motion and partial observability. By combining deep point tracking with a factor graph optimization framework, ArtiPoint robustly estimates articulated part trajectories and articulation axes directly from raw RGB-D videos. To foster future research in this domain, we introduce Arti4D, the first ego-centric in-the-wild dataset that captures articulated object interactions at a scene level, accompanied by articulation labels and ground-truth camera poses. We benchmark ArtiPoint against a range of classical and learning-based baselines, demonstrating its superior performance on Arti4D. We make code and Arti4D publicly available at https://artipoint.cs.uni-freiburg.de.
>
---
#### [new 065] Manipulation as in Simulation: Enabling Accurate Geometry Perception in Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对机器人几何感知任务，解决传统2D视觉泛化差及深度相机噪声问题。提出Camera Depth Models（CDMs），通过模拟生成高质量数据，实现高精度深度预测，弥合模拟到现实的差距，使策略无缝迁移至真实机器人操作。**

- **链接: [http://arxiv.org/pdf/2509.02530v1](http://arxiv.org/pdf/2509.02530v1)**

> **作者:** Minghuan Liu; Zhengbang Zhu; Xiaoshen Han; Peng Hu; Haotong Lin; Xinyao Li; Jingxiao Chen; Jiafeng Xu; Yichu Yang; Yunfeng Lin; Xinghang Li; Yong Yu; Weinan Zhang; Tao Kong; Bingyi Kang
>
> **备注:** 32 pages, 18 figures, project page: https://manipulation-as-in-simulation.github.io/
>
> **摘要:** Modern robotic manipulation primarily relies on visual observations in a 2D color space for skill learning but suffers from poor generalization. In contrast, humans, living in a 3D world, depend more on physical properties-such as distance, size, and shape-than on texture when interacting with objects. Since such 3D geometric information can be acquired from widely available depth cameras, it appears feasible to endow robots with similar perceptual capabilities. Our pilot study found that using depth cameras for manipulation is challenging, primarily due to their limited accuracy and susceptibility to various types of noise. In this work, we propose Camera Depth Models (CDMs) as a simple plugin on daily-use depth cameras, which take RGB images and raw depth signals as input and output denoised, accurate metric depth. To achieve this, we develop a neural data engine that generates high-quality paired data from simulation by modeling a depth camera's noise pattern. Our results show that CDMs achieve nearly simulation-level accuracy in depth prediction, effectively bridging the sim-to-real gap for manipulation tasks. Notably, our experiments demonstrate, for the first time, that a policy trained on raw simulated depth, without the need for adding noise or real-world fine-tuning, generalizes seamlessly to real-world robots on two challenging long-horizon tasks involving articulated, reflective, and slender objects, with little to no performance degradation. We hope our findings will inspire future research in utilizing simulation data and 3D information in general robot policies.
>
---
#### [new 066] OpenGuide: Assistive Object Retrieval in Indoor Spaces for Individuals with Visual Impairments
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出OpenGuide系统，解决视障人士在复杂室内环境中多物体检索问题。通过融合自然语言理解、视觉-语言模型与POMDP规划，实现高效目标定位与搜索，提升任务成功率与效率。**

- **链接: [http://arxiv.org/pdf/2509.02425v1](http://arxiv.org/pdf/2509.02425v1)**

> **作者:** Yifan Xu; Qianwei Wang; Vineet Kamat; Carol Menassa
>
> **备注:** 32 pages, 6 figures
>
> **摘要:** Indoor built environments like homes and offices often present complex and cluttered layouts that pose significant challenges for individuals who are blind or visually impaired, especially when performing tasks that involve locating and gathering multiple objects. While many existing assistive technologies focus on basic navigation or obstacle avoidance, few systems provide scalable and efficient multi-object search capabilities in real-world, partially observable settings. To address this gap, we introduce OpenGuide, an assistive mobile robot system that combines natural language understanding with vision-language foundation models (VLM), frontier-based exploration, and a Partially Observable Markov Decision Process (POMDP) planner. OpenGuide interprets open-vocabulary requests, reasons about object-scene relationships, and adaptively navigates and localizes multiple target items in novel environments. Our approach enables robust recovery from missed detections through value decay and belief-space reasoning, resulting in more effective exploration and object localization. We validate OpenGuide in simulated and real-world experiments, demonstrating substantial improvements in task success rate and search efficiency over prior methods. This work establishes a foundation for scalable, human-centered robotic assistance in assisted living environments.
>
---
#### [new 067] A Comparative Study of Spline-Based Trajectory Reconstruction Methods Across Varying Automatic Vehicle Location Data Densities
- **分类: cs.RO**

- **简介: 该论文比较13种轨迹重建方法，解决AVL数据密度不均导致的轨迹重建问题，评估速度、位置等因素影响，提出VCHIP-ME方法，平衡准确性和效率，适用于实时与历史分析。**

- **链接: [http://arxiv.org/pdf/2509.00119v1](http://arxiv.org/pdf/2509.00119v1)**

> **作者:** Jake Robbennolt; Sirajum Munira; Stephen D. Boyles
>
> **摘要:** Automatic vehicle location (AVL) data offers insights into transit dynamics, but its effectiveness is often hampered by inconsistent update frequencies, necessitating trajectory reconstruction. This research evaluates 13 trajectory reconstruction methods, including several novel approaches, using high-resolution AVL data from Austin, Texas. We examine the interplay of four critical factors -- velocity, position, smoothing, and data density -- on reconstruction performance. A key contribution of this study is evaluation of these methods across sparse and dense datasets, providing insights into the trade-off between accuracy and resource allocation. Our evaluation framework combines traditional mathematical error metrics for positional and velocity with practical considerations, such as physical realism (e.g., aligning velocity and acceleration with stopped states, deceleration rates, and speed variability). In addition, we provide insight into the relative value of each method in calculating realistic metrics for infrastructure evaluations. Our findings indicate that velocity-aware methods consistently outperform position-only approaches. Interestingly, we discovered that smoothing-based methods can degrade overall performance in complex, congested urban environments, although enforcing monotonicity remains critical. The velocity constrained Hermite interpolation with monotonicity enforcement (VCHIP-ME) yields optimal results, offering a balance between high accuracy and computational efficiency. Its minimal overhead makes it suitable for both historical analysis and real-time applications, providing significant predictive power when combined with dense datasets. These findings offer practical guidance for researchers and practitioners implementing trajectory reconstruction systems and emphasize the importance of investing in higher-frequency AVL data collection for improved analysis.
>
---
#### [new 068] U-ARM : Ultra low-cost general teleoperation interface for robot manipulation
- **分类: cs.RO**

- **简介: 该论文提出U-Arm，一种低成本、通用的远程操作接口，解决现有系统高成本和兼容性问题。通过优化机械设计与控制逻辑，实现高效机器人操作，并开源硬件与数据促进应用。**

- **链接: [http://arxiv.org/pdf/2509.02437v1](http://arxiv.org/pdf/2509.02437v1)**

> **作者:** Yanwen Zou; Zhaoye Zhou; Chenyang Shi; Zewei Ye; Junda Huang; Yan Ding; Bo Zhao
>
> **摘要:** We propose U-Arm, a low-cost and rapidly adaptable leader-follower teleoperation framework designed to interface with most of commercially available robotic arms. Our system supports teleoperation through three structurally distinct 3D-printed leader arms that share consistent control logic, enabling seamless compatibility with diverse commercial robot configurations. Compared with previous open-source leader-follower interfaces, we further optimized both the mechanical design and servo selection, achieving a bill of materials (BOM) cost of only \$50.5 for the 6-DoF leader arm and \$56.8 for the 7-DoF version. To enhance usability, we mitigate the common challenge in controlling redundant degrees of freedom by %engineering methods mechanical and control optimizations. Experimental results demonstrate that U-Arm achieves 39\% higher data collection efficiency and comparable task success rates across multiple manipulation scenarios compared with Joycon, another low-cost teleoperation interface. We have open-sourced all CAD models of three configs and also provided simulation support for validating teleoperation workflows. We also open-sourced real-world manipulation data collected with U-Arm. The project website is https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm.
>
---
#### [new 069] Human-Inspired Soft Anthropomorphic Hand System for Neuromorphic Object and Pose Recognition Using Multimodal Signals
- **分类: cs.RO**

- **简介: 论文提出仿生软手系统，整合触觉、本体觉和热信号，通过生物启发编码与SNN处理实现高效物体姿态识别，解决软手感知精度不足问题，创新点包括多模态融合与新型神经元模型提升分类性能。**

- **链接: [http://arxiv.org/pdf/2509.02275v1](http://arxiv.org/pdf/2509.02275v1)**

> **作者:** Fengyi Wang; Xiangyu Fu; Nitish Thakor; Gordon Cheng
>
> **摘要:** The human somatosensory system integrates multimodal sensory feedback, including tactile, proprioceptive, and thermal signals, to enable comprehensive perception and effective interaction with the environment. Inspired by the biological mechanism, we present a sensorized soft anthropomorphic hand equipped with diverse sensors designed to emulate the sensory modalities of the human hand. This system incorporates biologically inspired encoding schemes that convert multimodal sensory data into spike trains, enabling highly-efficient processing through Spiking Neural Networks (SNNs). By utilizing these neuromorphic signals, the proposed framework achieves 97.14% accuracy in object recognition across varying poses, significantly outperforming previous studies on soft hands. Additionally, we introduce a novel differentiator neuron model to enhance material classification by capturing dynamic thermal responses. Our results demonstrate the benefits of multimodal sensory fusion and highlight the potential of neuromorphic approaches for achieving efficient, robust, and human-like perception in robotic systems.
>
---
#### [new 070] Vehicle-in-Virtual-Environment (VVE) Method for Developing and Evaluating VRU Safety of Connected and Autonomous Driving with Focus on Bicyclist Safety
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 本论文应用VVE方法，针对自动驾驶中VRU（尤其是骑自行车者）安全问题，开发并评估自动转向与刹车系统，填补现有研究在统一规划、延迟控制及测试方法上的不足。**

- **链接: [http://arxiv.org/pdf/2509.00624v1](http://arxiv.org/pdf/2509.00624v1)**

> **作者:** Haochong Chen; Xincheng Cao; Bilin Aksun-Guvenc; Levent Guvenc
>
> **摘要:** Extensive research has already been conducted in the autonomous driving field to help vehicles navigate safely and efficiently. At the same time, plenty of current research on vulnerable road user (VRU) safety is performed which largely concentrates on perception, localization, or trajectory prediction of VRUs. However, existing research still exhibits several gaps, including the lack of a unified planning and collision avoidance system for autonomous vehicles, limited investigation into delay tolerant control strategies, and the absence of an efficient and standardized testing methodology. Ensuring VRU safety remains one of the most pressing challenges in autonomous driving, particularly in dynamic and unpredictable environments. In this two year project, we focused on applying the Vehicle in Virtual Environment (VVE) method to develop, evaluate, and demonstrate safety functions for Vulnerable Road Users (VRUs) using automated steering and braking of ADS. In this current second year project report, our primary focus was on enhancing the previous year results while also considering bicyclist safety.
>
---
#### [new 071] FGO-SLAM: Enhancing Gaussian SLAM with Globally Consistent Opacity Radiance Field
- **分类: cs.RO**

- **简介: 论文提出FGO-SLAM，解决视觉SLAM中高精度场景重建与姿态优化问题，通过引入全局一致的opacity radiance field，结合全局调整与深度/法线优化，提升跟踪与映射性能。**

- **链接: [http://arxiv.org/pdf/2509.01547v1](http://arxiv.org/pdf/2509.01547v1)**

> **作者:** Fan Zhu; Yifan Zhao; Ziyu Chen; Biao Yu; Hui Zhu
>
> **备注:** ICRA 2025
>
> **摘要:** Visual SLAM has regained attention due to its ability to provide perceptual capabilities and simulation test data for Embodied AI. However, traditional SLAM methods struggle to meet the demands of high-quality scene reconstruction, and Gaussian SLAM systems, despite their rapid rendering and high-quality mapping capabilities, lack effective pose optimization methods and face challenges in geometric reconstruction. To address these issues, we introduce FGO-SLAM, a Gaussian SLAM system that employs an opacity radiance field as the scene representation to enhance geometric mapping performance. After initial pose estimation, we apply global adjustment to optimize camera poses and sparse point cloud, ensuring robust tracking of our approach. Additionally, we maintain a globally consistent opacity radiance field based on 3D Gaussians and introduce depth distortion and normal consistency terms to refine the scene representation. Furthermore, after constructing tetrahedral grids, we identify level sets to directly extract surfaces from 3D Gaussians. Results across various real-world and large-scale synthetic datasets demonstrate that our method achieves state-of-the-art tracking accuracy and mapping performance.
>
---
#### [new 072] An Effective Trajectory Planning and an Optimized Path Planning for a 6-Degree-of-Freedom Robot Manipulator
- **分类: cs.RO; cs.SC; math.AC; 68W30, 13P10, 13P25, 68U07, 68R10**

- **简介: 该论文针对六自由度机械臂的运动规划问题，提出通过计算可行区域、生成关节配置序列及Dijkstra算法优化逆运动学解，实现复杂路径下的高效轨迹与路径优化。**

- **链接: [http://arxiv.org/pdf/2509.00828v1](http://arxiv.org/pdf/2509.00828v1)**

> **作者:** Takumu Okazaki; Akira Terui; Masahiko Mikawa
>
> **备注:** 26 pages
>
> **摘要:** An effective method for optimizing path planning for a specific model of a 6-degree-of-freedom (6-DOF) robot manipulator is presented as part of the motion planning of the manipulator using computer algebra. We assume that we are given a path in the form of a set of line segments that the end-effector should follow. We also assume that we have a method to solve the inverse kinematic problem of the manipulator at each via-point of the trajectory. The proposed method consists of three steps. First, we calculate the feasible region of the manipulator under a specific configuration of the end-effector. Next, we aim to find a trajectory on the line segments and a sequence of joint configurations the manipulator should follow to move the end-effector along the specified trajectory. Finally, we find the optimal combination of solutions to the inverse kinematic problem at each via-point along the trajectory by reducing the problem to a shortest-path problem of the graph and applying Dijkstra's algorithm. We show the effectiveness of the proposed method by experiments.
>
---
#### [new 073] Coral: A Unifying Abstraction Layer for Composable Robotics Software
- **分类: cs.RO**

- **简介: 该论文提出Coral框架，解决机器人软件集成复杂、修改成本高的问题。通过高层抽象层实现组件可组合性，减少配置负担，提升复用性与系统灵活性，适用于多场景任务。**

- **链接: [http://arxiv.org/pdf/2509.02453v1](http://arxiv.org/pdf/2509.02453v1)**

> **作者:** Steven Swanbeck; Mitch Pryor
>
> **摘要:** Despite the multitude of excellent software components and tools available in the robotics and broader software engineering communities, successful integration of software for robotic systems remains a time-consuming and challenging task for users of all knowledge and skill levels. And with robotics software often being built into tightly coupled, monolithic systems, even minor alterations to improve performance, adjust to changing task requirements, or deploy to new hardware can require significant engineering investment. To help solve this problem, this paper presents Coral, an abstraction layer for building, deploying, and coordinating independent software components that maximizes composability to allow for rapid system integration without modifying low-level code. Rather than replacing existing tools, Coral complements them by introducing a higher-level abstraction that constrains the integration process to semantically meaningful choices, reducing the configuration burden without limiting adaptability to diverse domains, systems, and tasks. We describe Coral in detail and demonstrate its utility in integrating software for scenarios of increasing complexity, including LiDAR-based SLAM and multi-robot corrosion mitigation tasks. By enabling practical composability in robotics software, Coral offers a scalable solution to a broad range of robotics system integration challenges, improving component reusability, system reconfigurability, and accessibility to both expert and non-expert users. We release Coral open source.
>
---
#### [new 074] Non-conflicting Energy Minimization in Reinforcement Learning based Robot Control
- **分类: cs.RO**

- **简介: 该论文提出无需超参数的强化学习方法，通过策略梯度投影平衡任务与能耗，实现机器人控制中能量最小化与任务性能的非冲突优化，实验验证在标准基准和真实机器人上均有效。**

- **链接: [http://arxiv.org/pdf/2509.01765v1](http://arxiv.org/pdf/2509.01765v1)**

> **作者:** Skand Peri; Akhil Perincherry; Bikram Pandit; Stefan Lee
>
> **备注:** 17 pages, 6 figures. Accepted as Oral presentation at Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Efficient robot control often requires balancing task performance with energy expenditure. A common approach in reinforcement learning (RL) is to penalize energy use directly as part of the reward function. This requires carefully tuning weight terms to avoid undesirable trade-offs where energy minimization harms task success. In this work, we propose a hyperparameter-free gradient optimization method to minimize energy expenditure without conflicting with task performance. Inspired by recent works in multitask learning, our method applies policy gradient projection between task and energy objectives to derive policy updates that minimize energy expenditure in ways that do not impact task performance. We evaluate this technique on standard locomotion benchmarks of DM-Control and HumanoidBench and demonstrate a reduction of 64% energy usage while maintaining comparable task performance. Further, we conduct experiments on a Unitree GO2 quadruped showcasing Sim2Real transfer of energy efficient policies. Our method is easy to implement in standard RL pipelines with minimal code changes, is applicable to any policy gradient method, and offers a principled alternative to reward shaping for energy efficient control policies.
>
---
#### [new 075] Data Retrieval with Importance Weights for Few-Shot Imitation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对少样本模仿学习中的数据检索问题，指出现有方法依赖高方差近邻估计且忽略先验分布。提出IWR方法，通过高斯核密度估计计算重要性权重，提升检索效果。**

- **链接: [http://arxiv.org/pdf/2509.01657v1](http://arxiv.org/pdf/2509.01657v1)**

> **作者:** Amber Xie; Rahul Chand; Dorsa Sadigh; Joey Hejna
>
> **备注:** Conference on Robot Learning 2025
>
> **摘要:** While large-scale robot datasets have propelled recent progress in imitation learning, learning from smaller task specific datasets remains critical for deployment in new environments and unseen tasks. One such approach to few-shot imitation learning is retrieval-based imitation learning, which extracts relevant samples from large, widely available prior datasets to augment a limited demonstration dataset. To determine the relevant data from prior datasets, retrieval-based approaches most commonly calculate a prior data point's minimum distance to a point in the target dataset in latent space. While retrieval-based methods have shown success using this metric for data selection, we demonstrate its equivalence to the limit of a Gaussian kernel density (KDE) estimate of the target data distribution. This reveals two shortcomings of the retrieval rule used in prior work. First, it relies on high-variance nearest neighbor estimates that are susceptible to noise. Second, it does not account for the distribution of prior data when retrieving data. To address these issues, we introduce Importance Weighted Retrieval (IWR), which estimates importance weights, or the ratio between the target and prior data distributions for retrieval, using Gaussian KDEs. By considering the probability ratio, IWR seeks to mitigate the bias of previous selection rules, and by using reasonable modeling parameters, IWR effectively smooths estimates using all data points. Across both simulation environments and real-world evaluations on the Bridge dataset we find that our method, IWR, consistently improves performance of existing retrieval-based methods, despite only requiring minor modifications.
>
---
#### [new 076] SR-SLAM: Scene-reliability Based RGB-D SLAM in Diverse Environments
- **分类: cs.RO**

- **简介: 论文提出SRR-SLAM框架，解决动态环境RGB-D SLAM的适应性与可靠性问题，通过场景可靠性评估及多策略优化提升性能。**

- **链接: [http://arxiv.org/pdf/2509.01111v1](http://arxiv.org/pdf/2509.01111v1)**

> **作者:** Haolan Zhang; Chenghao Li; Thanh Nguyen Canh; Lijun Wang; Nak Young Chong
>
> **备注:** submitted
>
> **摘要:** Visual simultaneous localization and mapping (SLAM) plays a critical role in autonomous robotic systems, especially where accurate and reliable measurements are essential for navigation and sensing. In feature-based SLAM, the quantityand quality of extracted features significantly influence system performance. Due to the variations in feature quantity and quality across diverse environments, current approaches face two major challenges: (1) limited adaptability in dynamic feature culling and pose estimation, and (2) insufficient environmental awareness in assessment and optimization strategies. To address these issues, we propose SRR-SLAM, a scene-reliability based framework that enhances feature-based SLAM through environment-aware processing. Our method introduces a unified scene reliability assessment mechanism that incorporates multiple metrics and historical observations to guide system behavior. Based on this assessment, we develop: (i) adaptive dynamic region selection with flexible geometric constraints, (ii) depth-assisted self-adjusting clustering for efficient dynamic feature removal in high-dimensional settings, and (iii) reliability-aware pose refinement that dynamically integrates direct methods when features are insufficient. Furthermore, we propose (iv) reliability-based keyframe selection and a weighted optimization scheme to reduce computational overhead while improving estimation accuracy. Extensive experiments on public datasets and real world scenarios show that SRR-SLAM outperforms state-of-the-art dynamic SLAM methods, achieving up to 90% improvement in accuracy and robustness across diverse environments. These improvements directly contribute to enhanced measurement precision and reliability in autonomous robotic sensing systems.
>
---
#### [new 077] Robotic Fire Risk Detection based on Dynamic Knowledge Graph Reasoning: An LLM-Driven Approach with Graph Chain-of-Thought
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出IOG框架，通过构建动态知识图谱与大模型结合，解决火灾场景中感知不足、响应延迟等问题，实现风险检测与应急决策。**

- **链接: [http://arxiv.org/pdf/2509.00054v1](http://arxiv.org/pdf/2509.00054v1)**

> **作者:** Haimei Pan; Jiyun Zhang; Qinxi Wei; Xiongnan Jin; Chen Xinkai; Jie Cheng
>
> **摘要:** Fire is a highly destructive disaster, but effective prevention can significantly reduce its likelihood of occurrence. When it happens, deploying emergency robots in fire-risk scenarios can help minimize the danger to human responders. However, current research on pre-disaster warnings and disaster-time rescue still faces significant challenges due to incomplete perception, inadequate fire situational awareness, and delayed response. To enhance intelligent perception and response planning for robots in fire scenarios, we first construct a knowledge graph (KG) by leveraging large language models (LLMs) to integrate fire domain knowledge derived from fire prevention guidelines and fire rescue task information from robotic emergency response documents. We then propose a new framework called Insights-on-Graph (IOG), which integrates the structured fire information of KG and Large Multimodal Models (LMMs). The framework generates perception-driven risk graphs from real-time scene imagery to enable early fire risk detection and provide interpretable emergency responses for task module and robot component configuration based on the evolving risk situation. Extensive simulations and real-world experiments show that IOG has good applicability and practical application value in fire risk detection and rescue decision-making.
>
---
#### [new 078] OpenMulti: Open-Vocabulary Instance-Level Multi-Agent Distributed Implicit Mapping
- **分类: cs.RO**

- **简介: 该论文提出OpenMulti框架，解决多智能体分布式映射中实例级与语义理解不足问题。通过跨智能体实例对齐与交叉渲染监督，提升几何与语义精度，并支持实例级检索，增强下游应用效果。**

- **链接: [http://arxiv.org/pdf/2509.01228v1](http://arxiv.org/pdf/2509.01228v1)**

> **作者:** Jianyu Dou; Yinan Deng; Jiahui Wang; Xingsi Tang; Yi Yang; Yufeng Yue
>
> **备注:** Accepted to IEEE Robotics and Automation Letters. Project website: https://openmulti666.github.io/
>
> **摘要:** Multi-agent distributed collaborative mapping provides comprehensive and efficient representations for robots. However, existing approaches lack instance-level awareness and semantic understanding of environments, limiting their effectiveness for downstream applications. To address this issue, we propose OpenMulti, an open-vocabulary instance-level multi-agent distributed implicit mapping framework. Specifically, we introduce a Cross-Agent Instance Alignment module, which constructs an Instance Collaborative Graph to ensure consistent instance understanding across agents. To alleviate the degradation of mapping accuracy due to the blind-zone optimization trap, we leverage Cross Rendering Supervision to enhance distributed learning of the scene. Experimental results show that OpenMulti outperforms related algorithms in both fine-grained geometric accuracy and zero-shot semantic accuracy. In addition, OpenMulti supports instance-level retrieval tasks, delivering semantic annotations for downstream applications. The project website of OpenMulti is publicly available at https://openmulti666.github.io/.
>
---
#### [new 079] A Geometric Method for Base Parameter Analysis in Robot Inertia Identification Based on Projective Geometric Algebra
- **分类: cs.RO**

- **简介: 该论文提出基于射影几何代数的新型几何方法，用于机器人惯性参数识别。通过TP模型和DRNG算法，实现基础参数的高效准确识别，适用于多种机器人结构。**

- **链接: [http://arxiv.org/pdf/2509.02071v1](http://arxiv.org/pdf/2509.02071v1)**

> **作者:** Guangzhen Sun; Ye Ding; Xiangyang Zhu
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** This paper proposes a novel geometric method for analytically determining the base inertial parameters of robotic systems. The rigid body dynamics is reformulated using projective geometric algebra, leading to a new identification model named ``tetrahedral-point (TP)" model. Based on the rigid body TP model, coefficients in the regresoor matrix of the identification model are derived in closed-form, exhibiting clear geometric interpretations. Building directly from the dynamic model, three foundational principles for base parameter analysis are proposed: the shared points principle, fixed points principle, and planar rotations principle. With these principles, algorithms are developed to automatically determine all the base parameters. The core algorithm, referred to as Dynamics Regressor Nullspace Generator (DRNG), achieves $O(1)$-complexity theoretically following an $O(N)$-complexity preprocessing stage, where $N$ is the number of rigid bodies. The proposed method and algorithms are validated across four robots: Puma560, Unitree Go2, a 2RRU-1RRS parallel kinematics mechanism (PKM), and a 2PRS-1PSR PKM. In all cases, the algorithms successfully identify the complete set of base parameters. Notably, the approach demonstrates high robustness and computational efficiency, particularly in the cases of PKMs. Through the comprehensive demonstrations, the method is shown to be general, robust, and efficient.
>
---
#### [new 080] A Hybrid Input based Deep Reinforcement Learning for Lane Change Decision-Making of Autonomous Vehicle
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶车辆变道决策问题，提出融合周围车辆轨迹预测与多模态信息的深度强化学习方法，提升变道安全性与合理性。通过混合状态空间设计及端到端控制集成，实现安全高效的车道变换决策。**

- **链接: [http://arxiv.org/pdf/2509.01611v1](http://arxiv.org/pdf/2509.01611v1)**

> **作者:** Ziteng Gao; Jiaqi Qu; Chaoyu Chen
>
> **摘要:** Lane change decision-making for autonomous vehicles is a complex but high-reward behavior. In this paper, we propose a hybrid input based deep reinforcement learning (DRL) algorithm, which realizes abstract lane change decisions and lane change actions for autonomous vehicles within traffic flow. Firstly, a surrounding vehicles trajectory prediction method is proposed to reduce the risk of future behavior of surrounding vehicles to ego vehicle, and the prediction results are input into the reinforcement learning model as additional information. Secondly, to comprehensively leverage environmental information, the model extracts feature from high-dimensional images and low-dimensional sensor data simultaneously. The fusion of surrounding vehicle trajectory prediction and multi-modal information are used as state space of reinforcement learning to improve the rationality of lane change decision. Finally, we integrate reinforcement learning macro decisions with end-to-end vehicle control to achieve a holistic lane change process. Experiments were conducted within the CARLA simulator, and the results demonstrated that the utilization of a hybrid state space significantly enhances the safety of vehicle lane change decisions.
>
---
#### [new 081] A Risk-aware Spatial-temporal Trajectory Planning Framework for Autonomous Vehicles Using QP-MPC and Dynamic Hazard Fields
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出基于QP-MPC和动态危险场的风险感知时空轨迹规划框架，解决现有方法计算成本高、动态稳定性差及场景验证不足的问题。通过整合动态危险场成本函数、时空安全规划及舒适效率子奖励，实现多目标优化，提升自动驾驶车辆的轨迹规划效率、稳定性和舒适度。**

- **链接: [http://arxiv.org/pdf/2509.00643v1](http://arxiv.org/pdf/2509.00643v1)**

> **作者:** Zhen Tian; Zhihao Lin; Dezong Zhao; Christos Anagnostopoulos; Qiyuan Wang; Wenjing Zhao; Xiaodan Wang; Chongfeng Wei
>
> **摘要:** Trajectory planning is a critical component in ensuring the safety, stability, and efficiency of autonomous vehicles. While existing trajectory planning methods have achieved progress, they often suffer from high computational costs, unstable performance in dynamic environments, and limited validation across diverse scenarios. To overcome these challenges, we propose an enhanced QP-MPC-based framework that incorporates three key innovations: (i) a novel cost function designed with a dynamic hazard field, which explicitly balances safety, efficiency, and comfort; (ii) seamless integration of this cost function into the QP-MPC formulation, enabling direct optimization of desired driving behaviors; and (iii) extensive validation of the proposed framework across complex tasks. The spatial safe planning is guided by a dynamic hazard field (DHF) for risk assessment, while temporal safe planning is based on a space-time graph. Besides, the quintic polynomial sampling and sub-reward of comforts are used to ensure comforts during lane-changing. The sub-reward of efficiency is used to maintain driving efficiency. Finally, the proposed DHF-enhanced objective function integrates multiple objectives, providing a proper optimization tasks for QP-MPC. Extensive simulations demonstrate that the proposed framework outperforms benchmark optimization methods in terms of efficiency, stability, and comfort across a variety of scenarios likes lane-changing, overtaking, and crossing intersections.
>
---
#### [new 082] CARIS: A Context-Adaptable Robot Interface System for Personalized and Scalable Human-Robot Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出CARIS系统，解决传统WoZ工具上下文适应性差的问题，通过整合多模态功能，在心理健康陪伴和导游场景中验证其有效性，并提出改进建议，提供公开工具促进HRI研究。**

- **链接: [http://arxiv.org/pdf/2509.00660v1](http://arxiv.org/pdf/2509.00660v1)**

> **作者:** Felipe Arias-Russi; Yuanchen Bai; Angelique Taylor
>
> **摘要:** The human-robot interaction (HRI) field has traditionally used Wizard-of-Oz (WoZ) controlled robots to explore navigation, conversational dynamics, human-in-the-loop interactions, and more to explore appropriate robot behaviors in everyday settings. However, existing WoZ tools are often limited to one context, making them less adaptable across different settings, users, and robotic platforms. To mitigate these issues, we introduce a Context-Adaptable Robot Interface System (CARIS) that combines advanced robotic capabilities such teleoperation, human perception, human-robot dialogue, and multimodal data recording. Through pilot studies, we demonstrate the potential of CARIS to WoZ control a robot in two contexts: 1) mental health companion and as a 2) tour guide. Furthermore, we identified areas of improvement for CARIS, including smoother integration between movement and communication, clearer functionality separation, recommended prompts, and one-click communication options to enhance the usability wizard control of CARIS. This project offers a publicly available, context-adaptable tool for the HRI community, enabling researchers to streamline data-driven approaches to intelligent robot behavior.
>
---
#### [new 083] Learning Social Heuristics for Human-Aware Path Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出HPLSV方法，通过学习社会价值函数作为启发式，解决机器人在社会场景中遵循社交规范的路径规划问题，应用于排队场景以实现通用化。**

- **链接: [http://arxiv.org/pdf/2509.02134v1](http://arxiv.org/pdf/2509.02134v1)**

> **作者:** Andrea Eirale; Matteo Leonetti; Marcello Chiaberge
>
> **摘要:** Social robotic navigation has been at the center of numerous studies in recent years. Most of the research has focused on driving the robotic agent along obstacle-free trajectories, respecting social distances from humans, and predicting their movements to optimize navigation. However, in order to really be socially accepted, the robots must be able to attain certain social norms that cannot arise from conventional navigation, but require a dedicated learning process. We propose Heuristic Planning with Learned Social Value (HPLSV), a method to learn a value function encapsulating the cost of social navigation, and use it as an additional heuristic in heuristic-search path planning. In this preliminary work, we apply the methodology to the common social scenario of joining a queue of people, with the intention of generalizing to further human activities.
>
---
#### [new 084] ER-LoRA: Effective-Rank Guided Adaptation for Weather-Generalized Depth Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对恶劣天气下的单目深度估计任务，解决真实数据缺乏和合成数据域差距问题，提出ER-LoRA方法，通过STM策略结合有效秩分解与参数高效微调，提升模型泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.00665v1](http://arxiv.org/pdf/2509.00665v1)**

> **作者:** Weilong Yan; Xin Zhang; Robby T. Tan
>
> **摘要:** Monocular depth estimation under adverse weather conditions (e.g.\ rain, fog, snow, and nighttime) remains highly challenging due to the lack of reliable ground truth and the difficulty of learning from unlabeled real-world data. Existing methods often rely on synthetic adverse data with pseudo-labels, which suffer from domain gaps, or employ self-supervised learning, which violates photometric assumptions in adverse scenarios. In this work, we propose to achieve weather--generalized depth estimation by Parameter--Efficient Fine--Tuning (PEFT) of Vision Foundation Models (VFMs), using only a small amount of high--visibility (normal) data. While PEFT has shown strong performance in semantic tasks such as segmentation, it remains underexplored for geometry--centric tasks like depth estimation -- especially in terms of balancing effective adaptation with the preservation of pretrained knowledge. To this end, we introduce the Selecting--Tuning--Maintaining (STM) strategy, which structurally decomposes the pretrained weights of VFMs based on two kinds of effective ranks (entropy--rank and stable--rank). In the tuning phase, we adaptively select the proper rank number as well as the task--aware singular directions for initialization, based on the entropy--rank and full--tuned weight; while in the maintaining stage, we enforce a principal direction regularization based on the stable--rank. This design guarantees flexible task adaptation while preserving the strong generalization capability of the pretrained VFM. Extensive experiments on four real--world benchmarks across diverse weather conditions demonstrate that STM not only outperforms existing PEFT methods and full fine--tuning but also surpasses methods trained with adverse synthetic data, and even the depth foundation model
>
---
#### [new 085] AI-driven Dispensing of Coral Reseeding Devices for Broad-scale Restoration of the Great Barrier Reef
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 本论文提出AI驱动的珊瑚修复设备，通过自动化底质分类与实时检测，提升大堡礁修复效率，减少人工依赖，并公开数据集促进研究。**

- **链接: [http://arxiv.org/pdf/2509.01019v1](http://arxiv.org/pdf/2509.01019v1)**

> **作者:** Scarlett Raine; Benjamin Moshirian; Tobias Fischer
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Coral reefs are on the brink of collapse, with climate change, ocean acidification, and pollution leading to a projected 70-90% loss of coral species within the next decade. Restoration efforts are crucial, but their success hinges on introducing automation to upscale efforts. We present automated deployment of coral re-seeding devices powered by artificial intelligence, computer vision, and robotics. Specifically, we perform automated substrate classification, enabling detection of areas of the seafloor suitable for coral growth, thus significantly reducing reliance on human experts and increasing the range and efficiency of restoration. Real-world testing of the algorithms on the Great Barrier Reef leads to deployment accuracy of 77.8%, sub-image patch classification of 89.1%, and real-time model inference at 5.5 frames per second. Further, we present and publicly contribute a large collection of annotated substrate image data to foster future research in this area.
>
---
#### [new 086] Quantum game models for interaction-aware decision-making in automated driving
- **分类: cs.GT; cs.RO**

- **简介: 该论文提出量子博弈模型QG-U1和QG-G4，用于自动驾驶中交互感知决策，解决传统方法忽略交互导致保守行为的问题，通过模拟验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.01582v1](http://arxiv.org/pdf/2509.01582v1)**

> **作者:** Karim Essalmi; Fernando Garrido; Fawzi Nashashibi
>
> **备注:** 8 pages, 8 figures, submitted to ICAR 2025
>
> **摘要:** Decision-making in automated driving must consider interactions with surrounding agents to be effective. However, traditional methods often neglect or oversimplify these interactions because they are difficult to model and solve, which can lead to overly conservative behavior of the ego vehicle. To address this gap, we propose two quantum game models, QG-U1 (Quantum Game - Unitary 1) and QG-G4 (Quantum Game - Gates 4), for interaction-aware decision-making. These models extend classical game theory by incorporating principles of quantum mechanics, such as superposition, interference, and entanglement. Specifically, QG-U1 and QG-G4 are designed for two-player games with two strategies per player and can be executed in real time on a standard computer without requiring quantum hardware. We evaluate both models in merging and roundabout scenarios and compare them with classical game-theoretic methods and baseline approaches (IDM, MOBIL, and a utility-based technique). Results show that QG-G4 achieves lower collision rates and higher success rates compared to baseline methods, while both quantum models yield higher expected payoffs than classical game approaches under certain parameter settings.
>
---
#### [new 087] An AI-Based Shopping Assistant System to Support the Visually Impaired
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出AI购物助手系统，解决视障人士在超市导航与产品识别的难题。通过集成计算机视觉、语音识别及室内导航技术，设计原型系统并实验验证，提升其自主购物体验与社会融入能力。**

- **链接: [http://arxiv.org/pdf/2509.01246v1](http://arxiv.org/pdf/2509.01246v1)**

> **作者:** Larissa R. de S. Shibata; Ankit A. Ravankar; Jose Victorio Salazar Luces; Yasuhisa Hirata
>
> **备注:** 7 pages, Accepted for 2025 SICE-FES conference (IEEE)
>
> **摘要:** Shopping plays a significant role in shaping consumer identity and social integration. However, for individuals with visual impairments, navigating in supermarkets and identifying products can be an overwhelming and challenging experience. This paper presents an AI-based shopping assistant prototype designed to enhance the autonomy and inclusivity of visually impaired individuals in supermarket environments. The system integrates multiple technologies, including computer vision, speech recognition, text-to-speech synthesis, and indoor navigation, into a single, user-friendly platform. Using cameras for ArUco marker detection and real-time environmental scanning, the system helps users navigate the store, identify product locations, provide real-time auditory guidance, and gain context about their surroundings. The assistant interacts with the user through voice commands and multimodal feedback, promoting a more dynamic and engaging shopping experience. The system was evaluated through experiments, which demonstrated its ability to guide users effectively and improve their shopping experience. This paper contributes to the development of inclusive AI-driven assistive technologies aimed at enhancing accessibility and user independence for the shopping experience.
>
---
#### [new 088] Design and Testing of a Low-Cost 3D-Printed Servo Gimbal for Thrust Vector Control in Model Rockets
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文设计并测试了一种低成本3D打印伺服陀螺仪，用于解决传统推力矢量控制系统的高成本与技术门槛问题。通过迭代设计、动态模拟与性能评估，验证了其在模型火箭中的可行性，为STEM教育和实验研究提供了可复制平台。**

- **链接: [http://arxiv.org/pdf/2509.00061v1](http://arxiv.org/pdf/2509.00061v1)**

> **作者:** Ekansh Singh
>
> **备注:** 8 pages, 6 figures, 1 table
>
> **摘要:** Thrust vector control (TVC) is a key mechanism for stabilizing rockets during flight, yet conventional implementations remain costly and technically inaccessible to students and hobbyists. This paper presents the design, fabrication, and testing of a low-cost, 3D-printed, servo-driven two-dimensional gimbal developed for model rocket applications. The gimbal underwent more than 60 CAD iterations, with servo selection guided by torque, response time, and stability requirements. A high-speed camera and Fusion 360 parameter simulations were used to emulate dynamic instability, enabling evaluation of angular deflection, servo responsiveness, and structural durability. The results demonstrated stable actuation within plus or minus 5 degrees, with response times on the average order of 44.5 ms, while limitations included servo fatigue and pin-joint stress under extended loading. The project highlights the feasibility of student-accessible thrust vector control systems and their potential as a reproducible platform for STEM education and experimental aerospace research.
>
---
#### [new 089] Embodied AI: Emerging Risks and Opportunities for Policy Action
- **分类: cs.CY; cs.AI; cs.RO**

- **简介: 论文针对EAI政策漏洞，通过风险分类、政策评估及建议，提出应对措施，以规范其发展并减轻潜在风险。**

- **链接: [http://arxiv.org/pdf/2509.00117v1](http://arxiv.org/pdf/2509.00117v1)**

> **作者:** Jared Perlo; Alexander Robey; Fazl Barez; Luciano Floridi; Jakob Mökander
>
> **摘要:** The field of embodied AI (EAI) is rapidly advancing. Unlike virtual AI, EAI can exist in, learn from, reason about, and act in the physical world. Given recent innovations in large language and multimodal models, along with increasingly advanced and responsive hardware, EAI systems are rapidly growing in capabilities and operational domains. These advances present significant risks, including physical harm from malicious use, mass surveillance, and economic and societal disruption. However, these risks have been severely overlooked by policymakers. Existing policies, such as international standards for industrial robots or statutes governing autonomous vehicles, are insufficient to address the full range of concerns. While lawmakers are increasingly focused on AI, there is now an urgent need to extend and adapt existing frameworks to account for the unique risks of EAI. To help bridge this gap, this paper makes three contributions: first, we provide a foundational taxonomy of key physical, informational, economic, and social EAI risks. Secondly, we analyze policies in the US, EU, and UK to identify how existing frameworks address these risks and where these policies leave critical gaps. We conclude by offering concrete policy recommendations to address the coming wave of EAI innovation, including mandatory testing and certification for EAI systems, clarified liability frameworks, and forward-looking strategies to manage and prepare for transformative economic and societal impacts.
>
---
#### [new 090] Online Identification using Adaptive Laws and Neural Networks for Multi-Quadrotor Centralized Transportation System
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文针对多四旋翼集中运输系统的鲁棒控制问题，提出基于在线自适应神经网络的误差子空间分解方法，通过Lyapunov稳定理论实现对时变扰动和模型不确定性的实时补偿，无需持续激励和离线训练，提升系统稳定性与适应性。**

- **链接: [http://arxiv.org/pdf/2509.01951v1](http://arxiv.org/pdf/2509.01951v1)**

> **作者:** Tianhua Gao; Kohji Tomita; Akiya Kamimura
>
> **摘要:** This paper introduces an adaptive-neuro identification method that enhances the robustness of a centralized multi-quadrotor transportation system. This method leverages online tuning and learning on decomposed error subspaces, enabling efficient real-time compensation to time-varying disturbances and model uncertainties acting on the payload. The strategy is to decompose the high-dimensional error space into a set of low-dimensional subspaces. In this way, the identification problem for unseen features is naturally transformed into submappings (``slices'') addressed by multiple adaptive laws and shallow neural networks, which are updated online via Lyapunov-based adaptation without requiring persistent excitation (PE) and offline training. Due to the model-free nature of neural networks, this approach can be well adapted to highly coupled and nonlinear centralized transportation systems. It serves as a feedforward compensator for the payload controller without explicitly relying on the dynamics coupled with the payload, such as cables and quadrotors. The proposed control system has been proven to be stable in the sense of Lyapunov, and its enhanced robustness under time-varying disturbances and model uncertainties was demonstrated by numerical simulations.
>
---
#### [new 091] Robustness Enhancement for Multi-Quadrotor Centralized Transportation System via Online Tuning and Learning
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文针对多四旋翼协同运输系统的鲁棒性问题，提出结合在线调参和学习的自适应神经几何控制方法，通过实时调整模型参数与扰动估计，利用Lyapunov方法保证稳定性，并通过仿真验证增强系统在扰动环境下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.01952v1](http://arxiv.org/pdf/2509.01952v1)**

> **作者:** Tianhua Gao; Kohji Tomita; Akiya Kamimura
>
> **摘要:** This paper introduces an adaptive-neuro geometric control for a centralized multi-quadrotor cooperative transportation system, which enhances both adaptivity and disturbance rejection. Our strategy is to coactively tune the model parameters and learn the external disturbances in real-time. To realize this, we augmented the existing geometric control with multiple neural networks and adaptive laws, where the estimated model parameters and the weights of the neural networks are simultaneously tuned and adjusted online. The Lyapunov-based adaptation guarantees bounded estimation errors without requiring either pre-training or the persistent excitation (PE) condition. The proposed control system has been proven to be stable in the sense of Lyapunov under certain preconditions, and its enhanced robustness under scenarios of disturbed environment and model-unmatched plant was demonstrated by numerical simulations.
>
---
#### [new 092] EgoTouch: On-Body Touch Input Using AR/VR Headset Cameras
- **分类: cs.HC; cs.CV; cs.RO**

- **简介: 该论文提出EgoTouch系统，利用AR/VR头显摄像头实现无需额外设备的裸手触控输入，解决空中界面精度低的问题。通过RGB摄像头捕捉皮肤接触信息，在多场景下保持高精度，并提供触控力、手指识别等元数据，推动身体界面实用化。**

- **链接: [http://arxiv.org/pdf/2509.01786v1](http://arxiv.org/pdf/2509.01786v1)**

> **作者:** Vimal Mollyn; Chris Harrison
>
> **备注:** Published at UIST 2024. More info at https://www.figlab.com/research/2024/egotouch
>
> **摘要:** In augmented and virtual reality (AR/VR) experiences, a user's arms and hands can provide a convenient and tactile surface for touch input. Prior work has shown on-body input to have significant speed, accuracy, and ergonomic benefits over in-air interfaces, which are common today. In this work, we demonstrate high accuracy, bare hands (i.e., no special instrumentation of the user) skin input using just an RGB camera, like those already integrated into all modern XR headsets. Our results show this approach can be accurate, and robust across diverse lighting conditions, skin tones, and body motion (e.g., input while walking). Finally, our pipeline also provides rich input metadata including touch force, finger identification, angle of attack, and rotation. We believe these are the requisite technical ingredients to more fully unlock on-skin interfaces that have been well motivated in the HCI literature but have lacked robust and practical methods.
>
---
#### [new 093] TransForSeg: A Multitask Stereo ViT for Joint Stereo Segmentation and 3D Force Estimation in Catheterization
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出TransForSeg，一种多任务立体Vision Transformer模型，用于导管插入中的立体分割与3D力估计。通过处理双视角X光图像，模型同时实现高精度分割与力估计，超越现有方法。**

- **链接: [http://arxiv.org/pdf/2509.01605v1](http://arxiv.org/pdf/2509.01605v1)**

> **作者:** Pedram Fekri; Mehrdad Zadeh; Javad Dargahi
>
> **备注:** Preprint version. This work is intended for future journal submission
>
> **摘要:** Recently, the emergence of multitask deep learning models has enhanced catheterization procedures by providing tactile and visual perception data through an end-to-end architec- ture. This information is derived from a segmentation and force estimation head, which localizes the catheter in X-ray images and estimates the applied pressure based on its deflection within the image. These stereo vision architectures incorporate a CNN- based encoder-decoder that captures the dependencies between X-ray images from two viewpoints, enabling simultaneous 3D force estimation and stereo segmentation of the catheter. With these tasks in mind, this work approaches the problem from a new perspective. We propose a novel encoder-decoder Vision Transformer model that processes two input X-ray images as separate sequences. Given sequences of X-ray patches from two perspectives, the transformer captures long-range dependencies without the need to gradually expand the receptive field for either image. The embeddings generated by both the encoder and decoder are fed into two shared segmentation heads, while a regression head employs the fused information from the decoder for 3D force estimation. The proposed model is a stereo Vision Transformer capable of simultaneously segmenting the catheter from two angles while estimating the generated forces at its tip in 3D. This model has undergone extensive experiments on synthetic X-ray images with various noise levels and has been compared against state-of-the-art pure segmentation models, vision-based catheter force estimation methods, and a multitask catheter segmentation and force estimation approach. It outperforms existing models, setting a new state-of-the-art in both catheter segmentation and force estimation.
>
---
#### [new 094] Ensemble-Based Event Camera Place Recognition Under Varying Illumination
- **分类: cs.CV; cs.RO**

- **简介: 论文提出基于集成的方法，解决事件相机在变化光照下的视觉地点识别问题，通过融合多模块提升鲁棒性，评估数据集并分析关键设计，Recall@1提升57%。**

- **链接: [http://arxiv.org/pdf/2509.01968v1](http://arxiv.org/pdf/2509.01968v1)**

> **作者:** Therese Joseph; Tobias Fischer; Michael Milford
>
> **摘要:** Compared to conventional cameras, event cameras provide a high dynamic range and low latency, offering greater robustness to rapid motion and challenging lighting conditions. Although the potential of event cameras for visual place recognition (VPR) has been established, developing robust VPR frameworks under severe illumination changes remains an open research problem. In this paper, we introduce an ensemble-based approach to event camera place recognition that combines sequence-matched results from multiple event-to-frame reconstructions, VPR feature extractors, and temporal resolutions. Unlike previous event-based ensemble methods, which only utilise temporal resolution, our broader fusion strategy delivers significantly improved robustness under varied lighting conditions (e.g., afternoon, sunset, night), achieving a 57% relative improvement in Recall@1 across day-night transitions. We evaluate our approach on two long-term driving datasets (with 8 km per traverse) without metric subsampling, thereby preserving natural variations in speed and stop duration that influence event density. We also conduct a comprehensive analysis of key design choices, including binning strategies, polarity handling, reconstruction methods, and feature extractors, to identify the most critical components for robust performance. Additionally, we propose a modification to the standard sequence matching framework that enhances performance at longer sequence lengths. To facilitate future research, we will release our codebase and benchmarking framework.
>
---
#### [new 095] A Layered Control Perspective on Legged Locomotion: Embedding Reduced Order Models via Hybrid Zero Dynamics
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文提出分层控制框架，将减少阶模型嵌入混合零动态，确保腿足机器人稳定行走，通过合成零动态流形并证明稳定性，统一ROM与全阶模型方法。**

- **链接: [http://arxiv.org/pdf/2509.00294v1](http://arxiv.org/pdf/2509.00294v1)**

> **作者:** Sergio A. Esteban; Max H. Cohen; Adrian B. Ghansah; Aaron D. Ames
>
> **摘要:** Reduced-order models (ROMs) provide a powerful means of synthesizing dynamic walking gaits on legged robots. Yet this approach lacks the formal guarantees enjoyed by methods that utilize the full-order model (FOM) for gait synthesis, e.g., hybrid zero dynamics. This paper aims to unify these approaches through a layered control perspective. In particular, we establish conditions on when a ROM of locomotion yields stable walking on the full-order hybrid dynamics. To achieve this result, given an ROM we synthesize a zero dynamics manifold encoding the behavior of the ROM -- controllers can be synthesized that drive the FOM to this surface, yielding hybrid zero dynamics. We prove that a stable periodic orbit in the ROM implies an input-to-state stable periodic orbit of the FOM's hybrid zero dynamics, and hence the FOM dynamics. This result is demonstrated in simulation on a linear inverted pendulum ROM and a 5-link planar walking FOM.
>
---
#### [new 096] AGS: Accelerating 3D Gaussian Splatting SLAM via CODEC-Assisted Frame Covisibility Detection
- **分类: cs.AR; cs.RO**

- **简介: 该论文提出AGS框架，解决3DGS-SLAM系统吞吐量低的问题。通过软硬件协同优化，采用粗细粒度姿态跟踪、高斯贡献共享及CODEC辅助的帧共视检测引擎，实现高效SLAM。**

- **链接: [http://arxiv.org/pdf/2509.00433v1](http://arxiv.org/pdf/2509.00433v1)**

> **作者:** Houshu He; Naifeng Jing; Li Jiang; Xiaoyao Liang; Zhuoran Song
>
> **备注:** 15 pages
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) is a critical task that enables autonomous vehicles to construct maps and localize themselves in unknown environments. Recent breakthroughs combine SLAM with 3D Gaussian Splatting (3DGS) to achieve exceptional reconstruction fidelity. However, existing 3DGS-SLAM systems provide insufficient throughput due to the need for multiple training iterations per frame and the vast number of Gaussians. In this paper, we propose AGS, an algorithm-hardware co-design framework to boost the efficiency of 3DGS-SLAM based on the intuition that SLAM systems process frames in a streaming manner, where adjacent frames exhibit high similarity that can be utilized for acceleration. On the software level: 1) We propose a coarse-then-fine-grained pose tracking method with respect to the robot's movement. 2) We avoid redundant computations of Gaussians by sharing their contribution information across frames. On the hardware level, we propose a frame covisibility detection engine to extract intermediate data from the video CODEC. We also implement a pose tracking engine and a mapping engine with workload schedulers to efficiently deploy the AGS algorithm. Our evaluation shows that AGS achieves up to $17.12\times$, $6.71\times$, and $5.41\times$ speedups against the mobile and high-end GPUs, and a state-of-the-art 3DGS accelerator, GSCore.
>
---
#### [new 097] Domain Adaptation-Based Crossmodal Knowledge Distillation for 3D Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D语义分割中3D标注成本高的问题，提出跨模态知识蒸馏方法，利用2D图像数据训练3D模型，通过领域自适应和自校准卷积减少对3D标注的依赖。**

- **链接: [http://arxiv.org/pdf/2509.00379v1](http://arxiv.org/pdf/2509.00379v1)**

> **作者:** Jialiang Kang; Jiawen Wang; Dingsheng Luo
>
> **备注:** ICRA 2025
>
> **摘要:** Semantic segmentation of 3D LiDAR data plays a pivotal role in autonomous driving. Traditional approaches rely on extensive annotated data for point cloud analysis, incurring high costs and time investments. In contrast, realworld image datasets offer abundant availability and substantial scale. To mitigate the burden of annotating 3D LiDAR point clouds, we propose two crossmodal knowledge distillation methods: Unsupervised Domain Adaptation Knowledge Distillation (UDAKD) and Feature and Semantic-based Knowledge Distillation (FSKD). Leveraging readily available spatio-temporally synchronized data from cameras and LiDARs in autonomous driving scenarios, we directly apply a pretrained 2D image model to unlabeled 2D data. Through crossmodal knowledge distillation with known 2D-3D correspondence, we actively align the output of the 3D network with the corresponding points of the 2D network, thereby obviating the necessity for 3D annotations. Our focus is on preserving modality-general information while filtering out modality-specific details during crossmodal distillation. To achieve this, we deploy self-calibrated convolution on 3D point clouds as the foundation of our domain adaptation module. Rigorous experimentation validates the effectiveness of our proposed methods, consistently surpassing the performance of state-of-the-art approaches in the field.
>
---
#### [new 098] Harnessing ADAS for Pedestrian Safety: A Data-Driven Exploration of Fatality Reduction
- **分类: cs.CY; cs.RO**

- **简介: 该论文通过分析FARS数据，评估ADAS功能（如PAEB、FCW、LDW）对降低行人致命事故的效果，探讨其在不同环境条件下的有效性，旨在为提升行人安全提供数据支持与政策建议。**

- **链接: [http://arxiv.org/pdf/2509.00048v1](http://arxiv.org/pdf/2509.00048v1)**

> **作者:** Methusela Sulle; Judith Mwakalonge; Gurcan Comert; Saidi Siuhi; Nana Kankam Gyimah
>
> **摘要:** Pedestrian fatalities continue to rise in the United States, driven by factors such as human distraction, increased vehicle size, and complex traffic environments. Advanced Driver Assistance Systems (ADAS) offer a promising avenue for improving pedestrian safety by enhancing driver awareness and vehicle responsiveness. This study conducts a comprehensive data-driven analysis utilizing the Fatality Analysis Reporting System (FARS) to quantify the effectiveness of specific ADAS features like Pedestrian Automatic Emergency Braking (PAEB), Forward Collision Warning (FCW), and Lane Departure Warning (LDW), in lowering pedestrian fatalities. By linking vehicle specifications with crash data, we assess how ADAS performance varies under different environmental and behavioral conditions, such as lighting, weather, and driver/pedestrian distraction. Results indicate that while ADAS can reduce crash severity and prevent some fatalities, its effectiveness is diminished in low-light and adverse weather. The findings highlight the need for enhanced sensor technologies and improved driver education. This research informs policymakers, transportation planners, and automotive manufacturers on optimizing ADAS deployment to improve pedestrian safety and reduce traffic-related deaths.
>
---
#### [new 099] Curve-based slicer for multi-axis DLP 3D printing
- **分类: cs.GR; cs.RO**

- **简介: 该论文提出基于曲线的切片方法，解决多轴DLP 3D打印中悬垂与阶梯效应问题。通过优化参数曲线定义层和运动轨迹，实现无碰撞、高质量打印，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.00040v1](http://arxiv.org/pdf/2509.00040v1)**

> **作者:** Chengkai Dai; Tao Liu; Dezhao Guo; Binzhi Sun; Guoxin Fang; Yeung Yam; Charlie C. L. Wang
>
> **摘要:** This paper introduces a novel curve-based slicing method for generating planar layers with dynamically varying orientations in digital light processing (DLP) 3D printing. Our approach effectively addresses key challenges in DLP printing, such as regions with large overhangs and staircase artifacts, while preserving its intrinsic advantages of high resolution and fast printing speeds. We formulate the slicing problem as an optimization task, in which parametric curves are computed to define both the slicing layers and the model partitioning through their tangent planes. These curves inherently define motion trajectories for the build platform and can be optimized to meet critical manufacturing objectives, including collision-free motion and floating-free deposition. We validate our method through physical experiments on a robotic multi-axis DLP printing setup, demonstrating that the optimized curves can robustly guide smooth, high-quality fabrication of complex geometries.
>
---
#### [new 100] Learning to Coordinate: Distributed Meta-Trajectory Optimization Via Differentiable ADMM-DDP
- **分类: cs.LG; cs.MA; cs.RO; cs.SY; eess.SY**

- **简介: 论文提出L2C框架，通过元学习自动调整多智能体系统分布式轨迹优化的超参数，结合ADMM-DDP与神经网络，实现高效协同控制，适应多样任务与配置。**

- **链接: [http://arxiv.org/pdf/2509.01630v1](http://arxiv.org/pdf/2509.01630v1)**

> **作者:** Bingheng Wang; Yichao Gao; Tianchen Sun; Lin Zhao
>
> **摘要:** Distributed trajectory optimization via ADMM-DDP is a powerful approach for coordinating multi-agent systems, but it requires extensive tuning of tightly coupled hyperparameters that jointly govern local task performance and global coordination. In this paper, we propose Learning to Coordinate (L2C), a general framework that meta-learns these hyperparameters, modeled by lightweight agent-wise neural networks, to adapt across diverse tasks and agent configurations. L2C differentiates end-to-end through the ADMM-DDP pipeline in a distributed manner. It also enables efficient meta-gradient computation by reusing DDP components such as Riccati recursions and feedback gains. These gradients correspond to the optimal solutions of distributed matrix-valued LQR problems, coordinated across agents via an auxiliary ADMM framework that becomes convex under mild assumptions. Training is further accelerated by truncating iterations and meta-learning ADMM penalty parameters optimized for rapid residual reduction, with provable Lipschitz-bounded gradient errors. On a challenging cooperative aerial transport task, L2C generates dynamically feasible trajectories in high-fidelity simulation using IsaacSIM, reconfigures quadrotor formations for safe 6-DoF load manipulation in tight spaces, and adapts robustly to varying team sizes and task conditions, while achieving up to $88\%$ faster gradient computation than state-of-the-art methods.
>
---
#### [new 101] Robix: A Unified Model for Robot Interaction, Reasoning and Planning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出Robix，构建机器人交互、推理与规划的统一框架，解决复杂指令执行与多场景任务规划问题。通过三阶段训练策略（预训练、监督微调、强化学习），实现动态命令生成、主动对话及长时任务一致性，超越现有模型在多样化任务中的表现。**

- **链接: [http://arxiv.org/pdf/2509.01106v1](http://arxiv.org/pdf/2509.01106v1)**

> **作者:** Huang Fang; Mengxi Zhang; Heng Dong; Wei Li; Zixuan Wang; Qifeng Zhang; Xueyun Tian; Yucheng Hu; Hang Li
>
> **备注:** Tech report. Project page: https://robix-seed.github.io/robix/
>
> **摘要:** We introduce Robix, a unified model that integrates robot reasoning, task planning, and natural language interaction within a single vision-language architecture. Acting as the high-level cognitive layer in a hierarchical robot system, Robix dynamically generates atomic commands for the low-level controller and verbal responses for human interaction, enabling robots to follow complex instructions, plan long-horizon tasks, and interact naturally with human within an end-to-end framework. Robix further introduces novel capabilities such as proactive dialogue, real-time interruption handling, and context-aware commonsense reasoning during task execution. At its core, Robix leverages chain-of-thought reasoning and adopts a three-stage training strategy: (1) continued pretraining to enhance foundational embodied reasoning abilities including 3D spatial understanding, visual grounding, and task-centric reasoning; (2) supervised finetuning to model human-robot interaction and task planning as a unified reasoning-action sequence; and (3) reinforcement learning to improve reasoning-action consistency and long-horizon task coherence. Extensive experiments demonstrate that Robix outperforms both open-source and commercial baselines (e.g., GPT-4o and Gemini 2.5 Pro) in interactive task execution, demonstrating strong generalization across diverse instruction types (e.g., open-ended, multi-stage, constrained, invalid, and interrupted) and various user-involved tasks such as table bussing, grocery shopping, and dietary filtering.
>
---
#### [new 102] Metamorphic Testing of Multimodal Human Trajectory Prediction
- **分类: cs.SE; cs.RO**

- **简介: 该论文提出基于蜕变测试的多模态人类轨迹预测模型测试方法，通过设计五类蜕变关系及概率距离指标，解决缺乏明确测试 oracle 的问题，评估模型对输入变换的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.01294v1](http://arxiv.org/pdf/2509.01294v1)**

> **作者:** Helge Spieker; Nadjib Lazaar; Arnaud Gotlieb; Nassim Belmecheri
>
> **备注:** Information and Software Technology
>
> **摘要:** Context: Predicting human trajectories is crucial for the safety and reliability of autonomous systems, such as automated vehicles and mobile robots. However, rigorously testing the underlying multimodal Human Trajectory Prediction (HTP) models, which typically use multiple input sources (e.g., trajectory history and environment maps) and produce stochastic outputs (multiple possible future paths), presents significant challenges. The primary difficulty lies in the absence of a definitive test oracle, as numerous future trajectories might be plausible for any given scenario. Objectives: This research presents the application of Metamorphic Testing (MT) as a systematic methodology for testing multimodal HTP systems. We address the oracle problem through metamorphic relations (MRs) adapted for the complexities and stochastic nature of HTP. Methods: We present five MRs, targeting transformations of both historical trajectory data and semantic segmentation maps used as an environmental context. These MRs encompass: 1) label-preserving geometric transformations (mirroring, rotation, rescaling) applied to both trajectory and map inputs, where outputs are expected to transform correspondingly. 2) Map-altering transformations (changing semantic class labels, introducing obstacles) with predictable changes in trajectory distributions. We propose probabilistic violation criteria based on distance metrics between probability distributions, such as the Wasserstein or Hellinger distance. Conclusion: This study introduces tool, a MT framework for the oracle-less testing of multimodal, stochastic HTP systems. It allows for assessment of model robustness against input transformations and contextual changes without reliance on ground-truth trajectories.
>
---
#### [new 103] Nonlinear Model Predictive Control-Based Reverse Path-Planning and Path-Tracking Control of a Vehicle with Trailer System
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对车辆-挂车系统倒车停车控制难题，提出基于NMPC的路径规划与跟踪方法，通过将挂车视为独立车辆进行规划并利用逆运动学传递控制输入，提升停车精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.01820v1](http://arxiv.org/pdf/2509.01820v1)**

> **作者:** Xincheng Cao; Haochong Chen; Bilin Aksun-Guvenc; Levent Guvenc; Brian Link; Peter J Richmond; Dokyung Yim; Shihong Fan; John Harber
>
> **摘要:** Reverse parking maneuvers of a vehicle with trailer system is a challenging task to complete for human drivers due to the unstable nature of the system and unintuitive controls required to orientate the trailer properly. This paper hence proposes an optimization-based automation routine to handle the path-planning and path-tracking control process of such type of maneuvers. The proposed approach utilizes nonlinear model predictive control (NMPC) to robustly guide the vehicle-trailer system into the desired parking space, and an optional forward repositioning maneuver can be added as an additional stage of the parking process to obtain better system configurations, before backward motion can be attempted again to get a good final pose. The novelty of the proposed approach is the simplicity of its formulation, as the path-planning and path-tracking operations are only conducted on the trailer being viewed as a standalone vehicle, before the control inputs are propagated to the tractor vehicle via inverse kinematic relationships also derived in this paper. Simulation case studies and hardware-in-the-loop tests are performed, and the results demonstrate the efficacy of the proposed approach.
>
---
#### [new 104] End-to-End Low-Level Neural Control of an Industrial-Grade 6D Magnetic Levitation System
- **分类: eess.SY; cs.AI; cs.RO; cs.SY; I.2.9; I.2.8; I.2.6; D.4.7; C.3; J.7**

- **简介: 论文提出首个端到端神经控制器，用于六维磁悬浮系统，解决传统控制方法保守性问题，通过直接映射传感器数据与姿态至电流指令，实现泛化与鲁棒控制。**

- **链接: [http://arxiv.org/pdf/2509.01388v1](http://arxiv.org/pdf/2509.01388v1)**

> **作者:** Philipp Hartmann; Jannick Stranghöner; Klaus Neumann
>
> **备注:** 8 pages, 7 figures, 2 tables
>
> **摘要:** Magnetic levitation is poised to revolutionize industrial automation by integrating flexible in-machine product transport and seamless manipulation. It is expected to become the standard drive for automated manufacturing. However, controlling such systems is inherently challenging due to their complex, unstable dynamics. Traditional control approaches, which rely on hand-crafted control engineering, typically yield robust but conservative solutions, with their performance closely tied to the expertise of the engineering team. In contrast, neural control learning presents a promising alternative. This paper presents the first neural controller for 6D magnetic levitation. Trained end-to-end on interaction data from a proprietary controller, it directly maps raw sensor data and 6D reference poses to coil current commands. The neural controller can effectively generalize to previously unseen situations while maintaining accurate and robust control. These results underscore the practical feasibility of learning-based neural control in complex physical systems and suggest a future where such a paradigm could enhance or even substitute traditional engineering approaches in demanding real-world applications. The trained neural controller, source code, and demonstration videos are publicly available at https://sites.google.com/view/neural-maglev.
>
---
#### [new 105] MV-SSM: Multi-View State Space Modeling for 3D Human Pose Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MV-SSM框架，针对多视角3D人体姿态估计中相机配置变化导致的泛化问题，通过多视角特征与关键点建模结合，采用PSS块和GTBS扫描，提升跨数据集性能。**

- **链接: [http://arxiv.org/pdf/2509.00649v1](http://arxiv.org/pdf/2509.00649v1)**

> **作者:** Aviral Chharia; Wenbo Gou; Haoye Dong
>
> **备注:** CVPR 2025; Project Website: https://aviralchharia.github.io/MV-SSM
>
> **摘要:** While significant progress has been made in single-view 3D human pose estimation, multi-view 3D human pose estimation remains challenging, particularly in terms of generalizing to new camera configurations. Existing attention-based transformers often struggle to accurately model the spatial arrangement of keypoints, especially in occluded scenarios. Additionally, they tend to overfit specific camera arrangements and visual scenes from training data, resulting in substantial performance drops in new settings. In this study, we introduce a novel Multi-View State Space Modeling framework, named MV-SSM, for robustly estimating 3D human keypoints. We explicitly model the joint spatial sequence at two distinct levels: the feature level from multi-view images and the person keypoint level. We propose a Projective State Space (PSS) block to learn a generalized representation of joint spatial arrangements using state space modeling. Moreover, we modify Mamba's traditional scanning into an effective Grid Token-guided Bidirectional Scanning (GTBS), which is integral to the PSS block. Multiple experiments demonstrate that MV-SSM achieves strong generalization, outperforming state-of-the-art methods: +10.8 on AP25 (+24%) on the challenging three-camera setting in CMU Panoptic, +7.0 on AP25 (+13%) on varying camera arrangements, and +15.3 PCP (+38%) on Campus A1 in cross-dataset evaluations. Project Website: https://aviralchharia.github.io/MV-SSM
>
---
#### [new 106] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Movable Obstacles
- **分类: cs.AI; cs.MA; cs.RO; 93A16 93A16**

- **简介: 该论文提出Block Rearrangement Problem（BRaP），解决仓库密集环境中存储块重排问题。通过形式化为图搜索问题，设计五种基于搜索的算法，结合多智能体路径规划与启发式方法，高效处理深埋块的重排任务。**

- **链接: [http://arxiv.org/pdf/2509.01022v1](http://arxiv.org/pdf/2509.01022v1)**

> **作者:** Bo Fu; Zhe Chen; Rahul Chandan; Alex Barbosa; Michael Caldara; Joey Durham; Federico Pecora
>
> **摘要:** We introduce the Block Rearrangement Problem (BRaP), a challenging component of large warehouse management which involves rearranging storage blocks within dense grids to achieve a target state. We formally define the BRaP as a graph search problem. Building on intuitions from sliding puzzle problems, we propose five search-based solution algorithms, leveraging joint configuration space search, classical planning, multi-agent pathfinding, and expert heuristics. We evaluate the five approaches empirically for plan quality and scalability. Despite the exponential relation between search space size and block number, our methods demonstrate efficiency in creating rearrangement plans for deeply buried blocks in up to 80x80 grids.
>
---
## 更新

#### [replaced 001] ManipBench: Benchmarking Vision-Language Models for Low-Level Robot Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.09698v2](http://arxiv.org/pdf/2505.09698v2)**

> **作者:** Enyu Zhao; Vedant Raval; Hejia Zhang; Jiageng Mao; Zeyu Shangguan; Stefanos Nikolaidis; Yue Wang; Daniel Seita
>
> **备注:** Conference on Robot Learning (CoRL) 2025. 50 pages and 30 figures. v2 is the camera-ready and includes a few more new experiments compared to v1
>
> **摘要:** Vision-Language Models (VLMs) have revolutionized artificial intelligence and robotics due to their commonsense reasoning capabilities. In robotic manipulation, VLMs are used primarily as high-level planners, but recent work has also studied their lower-level reasoning ability, which refers to making decisions about precise robot movements. However, the community currently lacks a clear and common benchmark that can evaluate how well VLMs can aid low-level reasoning in robotics. Consequently, we propose a novel benchmark, ManipBench, to evaluate the low-level robot manipulation reasoning capabilities of VLMs across various dimensions, including how well they understand object-object interactions and deformable object manipulation. We extensively test 33 representative VLMs across 10 model families on our benchmark, including variants to test different model sizes. Our evaluation shows that the performance of VLMs significantly varies across tasks, and there is a strong correlation between this performance and trends in our real-world manipulation tasks. It also shows that there remains a significant gap between these models and human-level understanding. See our website at: https://manipbench.github.io.
>
---
#### [replaced 002] HERMES: Human-to-Robot Embodied Learning from Multi-Source Motion Data for Mobile Dexterous Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20085v3](http://arxiv.org/pdf/2508.20085v3)**

> **作者:** Zhecheng Yuan; Tianming Wei; Langzhe Gu; Pu Hua; Tianhai Liang; Yuanpei Chen; Huazhe Xu
>
> **摘要:** Leveraging human motion data to impart robots with versatile manipulation skills has emerged as a promising paradigm in robotic manipulation. Nevertheless, translating multi-source human hand motions into feasible robot behaviors remains challenging, particularly for robots equipped with multi-fingered dexterous hands characterized by complex, high-dimensional action spaces. Moreover, existing approaches often struggle to produce policies capable of adapting to diverse environmental conditions. In this paper, we introduce HERMES, a human-to-robot learning framework for mobile bimanual dexterous manipulation. First, HERMES formulates a unified reinforcement learning approach capable of seamlessly transforming heterogeneous human hand motions from multiple sources into physically plausible robotic behaviors. Subsequently, to mitigate the sim2real gap, we devise an end-to-end, depth image-based sim2real transfer method for improved generalization to real-world scenarios. Furthermore, to enable autonomous operation in varied and unstructured environments, we augment the navigation foundation model with a closed-loop Perspective-n-Point (PnP) localization mechanism, ensuring precise alignment of visual goals and effectively bridging autonomous navigation and dexterous manipulation. Extensive experimental results demonstrate that HERMES consistently exhibits generalizable behaviors across diverse, in-the-wild scenarios, successfully performing numerous complex mobile bimanual dexterous manipulation tasks. Project Page:https://gemcollector.github.io/HERMES/.
>
---
#### [replaced 003] SPGrasp: Spatiotemporal Prompt-driven Grasp Synthesis in Dynamic Scenes
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.20547v2](http://arxiv.org/pdf/2508.20547v2)**

> **作者:** Yunpeng Mei; Hongjie Cao; Yinqiu Xia; Wei Xiao; Zhaohan Feng; Gang Wang; Jie Chen
>
> **摘要:** Real-time interactive grasp synthesis for dynamic objects remains challenging as existing methods fail to achieve low-latency inference while maintaining promptability. To bridge this gap, we propose SPGrasp (spatiotemporal prompt-driven dynamic grasp synthesis), a novel framework extending segment anything model v2 (SAMv2) for video stream grasp estimation. Our core innovation integrates user prompts with spatiotemporal context, enabling real-time interaction with end-to-end latency as low as 59 ms while ensuring temporal consistency for dynamic objects. In benchmark evaluations, SPGrasp achieves instance-level grasp accuracies of 90.6% on OCID and 93.8% on Jacquard. On the challenging GraspNet-1Billion dataset under continuous tracking, SPGrasp achieves 92.0% accuracy with 73.1 ms per-frame latency, representing a 58.5% reduction compared to the prior state-of-the-art promptable method RoG-SAM while maintaining competitive accuracy. Real-world experiments involving 13 moving objects demonstrate a 94.8% success rate in interactive grasping scenarios. These results confirm SPGrasp effectively resolves the latency-interactivity trade-off in dynamic grasp synthesis.
>
---
#### [replaced 004] NetRoller: Interfacing General and Specialized Models for End-to-End Autonomous Driving
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.14589v2](http://arxiv.org/pdf/2506.14589v2)**

> **作者:** Ren Xin; Hongji Liu; Xiaodong Mei; Wenru Liu; Maosheng Ye; Zhili Chen; Jun Ma
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Integrating General Models (GMs) such as Large Language Models (LLMs), with Specialized Models (SMs) in autonomous driving tasks presents a promising approach to mitigating challenges in data diversity and model capacity of existing specialized driving models. However, this integration leads to problems of asynchronous systems, which arise from the distinct characteristics inherent in GMs and SMs. To tackle this challenge, we propose NetRoller, an adapter that incorporates a set of novel mechanisms to facilitate the seamless integration of GMs and specialized driving models. Specifically, our mechanisms for interfacing the asynchronous GMs and SMs are organized into three key stages. NetRoller first harvests semantically rich and computationally efficient representations from the reasoning processes of LLMs using an early stopping mechanism, which preserves critical insights on driving context while maintaining low overhead. It then applies learnable query embeddings, nonsensical embeddings, and positional layer embeddings to facilitate robust and efficient cross-modality translation. At last, it employs computationally efficient Query Shift and Feature Shift mechanisms to enhance the performance of SMs through few-epoch fine-tuning. Based on the mechanisms formalized in these three stages, NetRoller enables specialized driving models to operate at their native frequencies while maintaining situational awareness of the GM. Experiments conducted on the nuScenes dataset demonstrate that integrating GM through NetRoller significantly improves human similarity and safety in planning tasks, and it also achieves noticeable precision improvements in detection and mapping tasks for end-to-end autonomous driving. The code and models are available at https://github.com/Rex-sys-hk/NetRoller .
>
---
#### [replaced 005] Tactile SoftHand-A: 3D-Printed, Tactile, Highly-underactuated, Anthropomorphic Robot Hand with an Antagonistic Tendon Mechanism
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2406.12731v2](http://arxiv.org/pdf/2406.12731v2)**

> **作者:** Haoran Li; Christopher J. Ford; Chenghua Lu; Yijiong Lin; Matteo Bianchi; Manuel G. Catalano; Efi Psomopoulou; Nathan F. Lepora
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** A challenging and important problem for tendon-driven multi-fingered robotic hands is to ensure grasping adaptivity while minimizing the number of actuators needed to provide human-like functionality. Inspired by the Pisa/IIT SoftHand, this paper introduces a 3D-printed, highly-underactuated, tactile-sensorized, five-finger robotic hand named the Tactile SoftHand-A, which features an antagonistic mechanism to actively open and close the hand. Our proposed dual-tendon design gives options that allow active control of specific (distal or proximal interphalangeal) joints; for example, to adjust from an enclosing to fingertip grasp or to manipulate an object with a fingertip. We also develop and integrate a new design of fully 3D-printed vision-based tactile sensor within the fingers that requires minimal hand assembly. A control scheme based on analytically extracting contact location and slip from the tactile images is used to coordinate the antagonistic tendon mechanism (using a marker displacement density map, suitable for TacTip-based sensors). We perform extensive testing of a single finger, the entire hand, and the tactile capabilities to show the improvements in reactivity, load-bearing, and manipulability in comparison to a SoftHand that lacks the antagonistic mechanism. We also demonstrate the hand's reactivity to contact disturbances including slip, and how this enables teleoperated control from human hand gestures. Overall, this study points the way towards a class of low-cost, accessible, 3D-printable, tactile, underactuated human-like robotic hands, and we openly release the designs to facilitate others to build upon this work. The designs are open-sourced at https://github.com/HaoranLi-Data/Tactile_SoftHand_A
>
---
#### [replaced 006] Frontier Shepherding: A Bio-inspired Multi-robot Framework for Large-Scale Exploration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.10931v2](http://arxiv.org/pdf/2409.10931v2)**

> **作者:** John Lewis; Meysam Basiri; Pedro U. Lima
>
> **备注:** 8 page article accepted at IEEE/RSJ International Conferenceo on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Efficient exploration of large-scale environments remains a critical challenge in robotics, with applications ranging from environmental monitoring to search and rescue operations. This article proposes Frontier Shepherding (FroShe), a bio-inspired multi-robot framework for large-scale exploration. The framework heuristically models frontier exploration based on the shepherding behavior of herding dogs, where frontiers are treated as a swarm of sheep reacting to robots modeled as shepherding dogs. FroShe is robust across varying environment sizes and obstacle densities, requiring minimal parameter tuning for deployment across multiple agents. Simulation results demonstrate that the proposed method performs consistently, regardless of environment complexity, and outperforms state-of-the-art exploration strategies by an average of 20% with three UAVs. The approach was further validated in real-world experiments using single- and dual-drone deployments in a forest-like environment.
>
---
#### [replaced 007] DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07118v2](http://arxiv.org/pdf/2508.07118v2)**

> **作者:** Aiden Swann; Alex Qiu; Matthew Strong; Angelina Zhang; Samuel Morstein; Kai Rayle; Monroe Kennedy III
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** DexFruit is a robotic manipulation framework that enables gentle, autonomous handling of fragile fruit and precise evaluation of damage. Many fruits are fragile and prone to bruising, thus requiring humans to manually harvest them with care. In this work, we demonstrate by using optical tactile sensing, autonomous manipulation of fruit with minimal damage can be achieved. We show that our tactile informed diffusion policies outperform baselines in both reduced bruising and pick-and-place success rate across three fruits: strawberries, tomatoes, and blackberries. In addition, we introduce FruitSplat, a novel technique to represent and quantify visual damage in high-resolution 3D representation via 3D Gaussian Splatting (3DGS). Existing metrics for measuring damage lack quantitative rigor or require expensive equipment. With FruitSplat, we distill a 2D strawberry mask as well as a 2D bruise segmentation mask into the 3DGS representation. Furthermore, this representation is modular and general, compatible with any relevant 2D model. Overall, we demonstrate a 92% grasping policy success rate, up to a 20% reduction in visual bruising, and up to an 31% improvement in grasp success rate on challenging fruit compared to our baselines across our three tested fruits. We rigorously evaluate this result with over 630 trials. Please checkout our website at https://dex-fruit.github.io .
>
---
#### [replaced 008] Dyna-LfLH: Learning Agile Navigation in Dynamic Environments from Learned Hallucination
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.17231v2](http://arxiv.org/pdf/2403.17231v2)**

> **作者:** Saad Abdul Ghani; Zizhao Wang; Peter Stone; Xuesu Xiao
>
> **备注:** Accepted at International Conference on Intelligent Robots and Systems (IROS) 2025 Hangzhou, China
>
> **摘要:** This paper introduces Dynamic Learning from Learned Hallucination (Dyna-LfLH), a self-supervised method for training motion planners to navigate environments with dense and dynamic obstacles. Classical planners struggle with dense, unpredictable obstacles due to limited computation, while learning-based planners face challenges in acquiring high- quality demonstrations for imitation learning or dealing with exploration inefficiencies in reinforcement learning. Building on Learning from Hallucination (LfH), which synthesizes training data from past successful navigation experiences in simpler environments, Dyna-LfLH incorporates dynamic obstacles by generating them through a learned latent distribution. This enables efficient and safe motion planner training. We evaluate Dyna-LfLH on a ground robot in both simulated and real environments, achieving up to a 25% improvement in success rate compared to baselines.
>
---
#### [replaced 009] Benchmarking LLM Privacy Recognition for Social Robot Decision Making
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.16124v2](http://arxiv.org/pdf/2507.16124v2)**

> **作者:** Dakota Sullivan; Shirley Zhang; Jennica Li; Heather Kirkorian; Bilge Mutlu; Kassem Fawaz
>
> **备注:** 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
>
> **摘要:** While robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-powered robots for enhanced human-robot interaction (HRI). To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within private environments, such as homes. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household robots. In this work, we present a set of privacy-relevant scenarios developed using the Contextual Integrity (CI) framework. We first surveyed users' privacy preferences regarding in-home robot behaviors and then examined how their privacy orientations affected their choices of these behaviors (N = 450). We then provided the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and found that the agreement between humans and LLMs was generally low. To further investigate the capabilities of LLMs as potential privacy controllers, we implemented four additional prompting strategies and compared their results. We discuss the performance of the evaluated models as well as the implications and potential of AI privacy awareness in human-robot interaction.
>
---
#### [replaced 010] A Physics-Based Continuum Model for Versatile, Scalable, and Fast Terramechanics Simulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05643v2](http://arxiv.org/pdf/2507.05643v2)**

> **作者:** Huzaifa Unjhawala; Luning Bakke; Harry Zhang; Michael Taylor; Ganesh Arivoli; Radu Serban; Dan Negrut
>
> **备注:** 32 pages, 21 figures, Submitted to Journal of Terramechanics
>
> **摘要:** This paper discusses Chrono's Continuous Representation Model (called herein Chrono::CRM), a general-purpose, scalable, and efficient simulation solution for terramechanics problems. Built on Chrono's Smoothed Particle Hydrodynamics (SPH) framework, Chrono::CRM moves beyond semi-empirical terramechanics approaches, e.g., Bekker-Wong/Janosi-Hanamoto, to provide a physics-based model able to address complex tasks such as digging, grading, as well as interaction with deformable wheels and complex grouser/lug patterns. The terramechanics model is versatile in that it allows the terrain to interact with both rigid and flexible implements simulated via the Chrono dynamics engine. We validate Chrono::CRM against experimental data from three physical tests, including one involving NASA's MGRU3 rover. In addition, the simulator is benchmarked against a high-fidelity Discrete Element Method (DEM) simulation of a digging scenario involving the Regolith Advanced Surface Systems Operations Robot (RASSOR). Being GPU-accelerated, Chrono::CRM achieves computational efficiency comparable to that of semi-empirical simulation approaches for terramechanics problems. Through an ``active domains'' implementation, Chrono::CRM can handle terrain stretches up to 10 km long with 100 million SPH particles at near interactive rates, making high-fidelity off-road simulations at large scales feasible. As a component of the Chrono package, the CRM model is open source and released under a BSD-3 license. All models and simulations used in this contribution are available in a public GitHub repository for reproducibility studies and further research.
>
---
#### [replaced 011] Making Physical Objects with Generative AI and Robotic Assembly: Considering Fabrication Constraints, Sustainability, Time, Functionality, and Accessibility
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2504.19131v3](http://arxiv.org/pdf/2504.19131v3)**

> **作者:** Alexander Htet Kyaw; Se Hwan Jeon; Miana Smith; Neil Gershenfeld
>
> **备注:** ACM CHI Conference on Human Factors in Computing Systems (CHI 2025), Workshop on Generative AI and Human-Computer Interaction, Yokohama, Japan, April 26 to May 1, 2025
>
> **摘要:** 3D generative AI enables rapid and accessible creation of 3D models from text or image inputs. However, translating these outputs into physical objects remains a challenge due to the constraints in the physical world. Recent studies have focused on improving the capabilities of 3D generative AI to produce fabricable outputs, with 3D printing as the main fabrication method. However, this workshop paper calls for a broader perspective by considering how fabrication methods align with the capabilities of 3D generative AI. As a case study, we present a novel system using discrete robotic assembly and 3D generative AI to make physical objects. Through this work, we identified five key aspects to consider in a physical making process based on the capabilities of 3D generative AI. 1) Fabrication Constraints: Current text-to-3D models can generate a wide range of 3D designs, requiring fabrication methods that can adapt to the variability of generative AI outputs. 2) Time: While generative AI can generate 3D models in seconds, fabricating physical objects can take hours or even days. Faster production could enable a closer iterative design loop between humans and AI in the making process. 3) Sustainability: Although text-to-3D models can generate thousands of models in the digital world, extending this capability to the real world would be resource-intensive, unsustainable and irresponsible. 4) Functionality: Unlike digital outputs from 3D generative AI models, the fabrication method plays a crucial role in the usability of physical objects. 5) Accessibility: While generative AI simplifies 3D model creation, the need for fabrication equipment can limit participation, making AI-assisted creation less inclusive. These five key aspects provide a framework for assessing how well a physical making process aligns with the capabilities of 3D generative AI and values in the world.
>
---
#### [replaced 012] An Exploratory Study on Human-Robot Interaction using Semantics-based Situational Awareness
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17376v2](http://arxiv.org/pdf/2507.17376v2)**

> **作者:** Tianshu Ruan; Aniketh Ramesh; Rustam Stolkin; Manolis Chiou
>
> **摘要:** In this paper, we investigate the impact of high-level semantics (evaluation of the environment) on Human-Robot Teams (HRT) and Human-Robot Interaction (HRI) in the context of mobile robot deployments. Although semantics has been widely researched in AI, how high-level semantics can benefit the HRT paradigm is underexplored, often fuzzy, and intractable. We applied a semantics-based framework that could reveal different indicators of the environment (i.e. how much semantic information exists) in a mock-up disaster response mission. In such missions, semantics are crucial as the HRT should handle complex situations and respond quickly with correct decisions, where humans might have a high workload and stress. Especially when human operators need to shift their attention between robots and other tasks, they will struggle to build Situational Awareness (SA) quickly. The experiment suggests that the presented semantics: 1) alleviate the perceived workload of human operators; 2) increase the operator's trust in the SA; and 3) help to reduce the reaction time in switching the level of autonomy when needed. Additionally, we find that participants with higher trust in the system are encouraged by high-level semantics to use teleoperation mode more.
>
---
#### [replaced 013] Safety-Critical Human-Machine Shared Driving for Vehicle Collision Avoidance based on Hamilton-Jacobi reachability
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.10610v3](http://arxiv.org/pdf/2502.10610v3)**

> **作者:** Shiyue Zhao; Junzhi Zhang; Rui Zhou; Neda Masoud; Jianxiong Li; Helai Huang; Shijie Zhao
>
> **备注:** 36 pages, 15 figures
>
> **摘要:** Road safety continues to be a pressing global issue, with vehicle collisions imposing significant human, societal, and economic burdens. Human-machine shared collision avoidance in critical collision scenarios aims to aid drivers' accident avoidance through intervening only when necessary. Existing methods count on replanning collision-free trajectories and imposing human-machine tracking, which usually interrupts the driver's intent and increases the risk of conflict. This paper introduces a Reachability-Aware Reinforcement Learning (RL) framework for shared control, guided by Hamilton-Jacobi (HJ) reachability analysis. Machine intervention is activated only when the vehicle approaches the Collision Avoidance Reachable Set (CARS), which represents states where collision is unavoidable. First, we precompute the reachability distributions and the CARS by solving the Bellman equation using offline data. To reduce human-machine conflicts, we develop a driver model for sudden obstacles and propose an authority allocation strategy considering key collision avoidance features. Finally, we train a RL agent to reduce human-machine conflicts while enforcing the hard constraint of avoiding entry into the CARS. The proposed method was tested on a real vehicle platform. Results show that the controller intervenes effectively near CARS to prevent collisions while maintaining improved original driving task performance. Robustness analysis further supports its flexibility across different driver attributes.
>
---
#### [replaced 014] Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.20396v2](http://arxiv.org/pdf/2502.20396v2)**

> **作者:** Toru Lin; Kartik Sachdev; Linxi Fan; Jitendra Malik; Yuke Zhu
>
> **备注:** Published at CoRL 2025. Project page can be found at https://toruowo.github.io/recipe/
>
> **摘要:** Learning generalizable robot manipulation policies, especially for complex multi-fingered humanoids, remains a significant challenge. Existing approaches primarily rely on extensive data collection and imitation learning, which are expensive, labor-intensive, and difficult to scale. Sim-to-real reinforcement learning (RL) offers a promising alternative, but has mostly succeeded in simpler state-based or single-hand setups. How to effectively extend this to vision-based, contact-rich bimanual manipulation tasks remains an open question. In this paper, we introduce a practical sim-to-real RL recipe that trains a humanoid robot to perform three challenging dexterous manipulation tasks: grasp-and-reach, box lift and bimanual handover. Our method features an automated real-to-sim tuning module, a generalized reward formulation based on contact and object goals, a divide-and-conquer policy distillation framework, and a hybrid object representation strategy with modality-specific augmentation. We demonstrate high success rates on unseen objects and robust, adaptive policy behaviors -- highlighting that vision-based dexterous manipulation via sim-to-real RL is not only viable, but also scalable and broadly applicable to real-world humanoid manipulation tasks.
>
---
#### [replaced 015] Perspective-Shifted Neuro-Symbolic World Models: A Framework for Socially-Aware Robot Navigation
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.20425v3](http://arxiv.org/pdf/2503.20425v3)**

> **作者:** Kevin Alcedo; Pedro U. Lima; Rachid Alami
>
> **备注:** Accepted as a regular paper at the 2025 IEEE International Conference on Robot & Human Interactive Communication (RO-MAN). \c{opyright} 2025 IEEE. The final version will appear in IEEE Xplore
>
> **摘要:** Navigating in environments alongside humans requires agents to reason under uncertainty and account for the beliefs and intentions of those around them. Under a sequential decision-making framework, egocentric navigation can naturally be represented as a Markov Decision Process (MDP). However, social navigation additionally requires reasoning about the hidden beliefs of others, inherently leading to a Partially Observable Markov Decision Process (POMDP), where agents lack direct access to others' mental states. Inspired by Theory of Mind and Epistemic Planning, we propose (1) a neuro-symbolic model-based reinforcement learning architecture for social navigation, addressing the challenge of belief tracking in partially observable environments; and (2) a perspective-shift operator for belief estimation, leveraging recent work on Influence-based Abstractions (IBA) in structured multi-agent settings.
>
---
#### [replaced 016] From Experts to a Generalist: Toward General Whole-Body Control for Humanoid Robots
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.12779v3](http://arxiv.org/pdf/2506.12779v3)**

> **作者:** Yuxuan Wang; Ming Yang; Ziluo Ding; Yu Zhang; Weishuai Zeng; Xinrun Xu; Haobin Jiang; Zongqing Lu
>
> **摘要:** Achieving general agile whole-body control on humanoid robots remains a major challenge due to diverse motion demands and data conflicts. While existing frameworks excel in training single motion-specific policies, they struggle to generalize across highly varied behaviors due to conflicting control requirements and mismatched data distributions. In this work, we propose BumbleBee (BB), an expert-generalist learning framework that combines motion clustering and sim-to-real adaptation to overcome these challenges. BB first leverages an autoencoder-based clustering method to group behaviorally similar motions using motion features and motion descriptions. Expert policies are then trained within each cluster and refined with real-world data through iterative delta action modeling to bridge the sim-to-real gap. Finally, these experts are distilled into a unified generalist controller that preserves agility and robustness across all motion types. Experiments on two simulations and a real humanoid robot demonstrate that BB achieves state-of-the-art general whole-body control, setting a new benchmark for agile, robust, and generalizable humanoid performance in the real world. The project webpage is available at https://beingbeyond.github.io/BumbleBee/.
>
---
#### [replaced 017] Co-Design of Soft Gripper with Neural Physics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20404v3](http://arxiv.org/pdf/2505.20404v3)**

> **作者:** Sha Yi; Xueqian Bai; Adabhav Singh; Jianglong Ye; Michael T Tolley; Xiaolong Wang
>
> **摘要:** For robot manipulation, both the controller and end-effector design are crucial. Soft grippers are generalizable by deforming to different geometries, but designing such a gripper and finding its grasp pose remains challenging. In this paper, we propose a co-design framework that generates an optimized soft gripper's block-wise stiffness distribution and its grasping pose, using a neural physics model trained in simulation. We derived a uniform-pressure tendon model for a flexure-based soft finger, then generated a diverse dataset by randomizing both gripper pose and design parameters. A neural network is trained to approximate this forward simulation, yielding a fast, differentiable surrogate. We embed that surrogate in an end-to-end optimization loop to optimize the ideal stiffness configuration and best grasp pose. Finally, we 3D-print the optimized grippers of various stiffness by changing the structural parameters. We demonstrate that our co-designed grippers significantly outperform baseline designs in both simulation and hardware experiments. More info: http://yswhynot.github.io/codesign-soft/
>
---
#### [replaced 018] Multi Object Tracking for Predictive Collision Avoidance
- **分类: cs.RO; cs.SY; eess.SY; 68T40**

- **链接: [http://arxiv.org/pdf/2307.02161v2](http://arxiv.org/pdf/2307.02161v2)**

> **作者:** Bruk Gebregziabher; Hadush Hailu
>
> **摘要:** The safe and efficient operation of Autonomous Mobile Robots (AMRs) in complex environments, such as manufacturing, logistics, and agriculture, necessitates accurate multi-object tracking and predictive collision avoidance. This paper presents algorithms and techniques for addressing these challenges using Lidar sensor data, emphasizing ensemble Kalman filter. The developed predictive collision avoidance algorithm employs the data provided by lidar sensors to track multiple objects and predict their velocities and future positions, enabling the AMR to navigate safely and effectively. A modification to the dynamic windowing approach is introduced to enhance the performance of the collision avoidance system. The overall system architecture encompasses object detection, multi-object tracking, and predictive collision avoidance control. The experimental results, obtained from both simulation and real-world data, demonstrate the effectiveness of the proposed methods in various scenarios, which lays the foundation for future research on global planners, other controllers, and the integration of additional sensors. This thesis contributes to the ongoing development of safe and efficient autonomous systems in complex and dynamic environments.
>
---
#### [replaced 019] A Survey on Vision-Language-Action Models for Embodied AI
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2405.14093v5](http://arxiv.org/pdf/2405.14093v5)**

> **作者:** Yueen Ma; Zixing Song; Yuzheng Zhuang; Jianye Hao; Irwin King
>
> **备注:** Project page: https://github.com/yueen-ma/Awesome-VLA
>
> **摘要:** Embodied AI is widely recognized as a key element of artificial general intelligence because it involves controlling embodied agents to perform tasks in the physical world. Building on the success of large language models and vision-language models, a new category of multimodal models -- referred to as vision-language-action models (VLAs) -- has emerged to address language-conditioned robotic tasks in embodied AI by leveraging their distinct ability to generate actions. In recent years, a myriad of VLAs have been developed, making it imperative to capture the rapidly evolving landscape through a comprehensive survey. To this end, we present the first survey on VLAs for embodied AI. This work provides a detailed taxonomy of VLAs, organized into three major lines of research. The first line focuses on individual components of VLAs. The second line is dedicated to developing control policies adept at predicting low-level actions. The third line comprises high-level task planners capable of decomposing long-horizon tasks into a sequence of subtasks, thereby guiding VLAs to follow more general user instructions. Furthermore, we provide an extensive summary of relevant resources, including datasets, simulators, and benchmarks. Finally, we discuss the challenges faced by VLAs and outline promising future directions in embodied AI. We have created a project associated with this survey, which is available at https://github.com/yueen-ma/Awesome-VLA.
>
---
#### [replaced 020] Domain-Conditioned Scene Graphs for State-Grounded Task Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06661v2](http://arxiv.org/pdf/2504.06661v2)**

> **作者:** Jonas Herzog; Jiangpin Liu; Yue Wang
>
> **备注:** Accepted for IROS 2025
>
> **摘要:** Recent robotic task planning frameworks have integrated large multimodal models (LMMs) such as GPT-4o. To address grounding issues of such models, it has been suggested to split the pipeline into perceptional state grounding and subsequent state-based planning. As we show in this work, the state grounding ability of LMM-based approaches is still limited by weaknesses in granular, structured, domain-specific scene understanding. To address this shortcoming, we develop a more structured state grounding framework that features a domain-conditioned scene graph as its scene representation. We show that such representation is actionable in nature as it is directly mappable to a symbolic state in planning languages such as the Planning Domain Definition Language (PDDL). We provide an instantiation of our state grounding framework where the domain-conditioned scene graph generation is implemented with a lightweight vision-language approach that classifies domain-specific predicates on top of domain-relevant object detections. Evaluated across three domains, our approach achieves significantly higher state rounding accuracy and task planning success rates compared to LMM-based approaches.
>
---
#### [replaced 021] SafeLink: Safety-Critical Control Under Dynamic and Irregular Unsafe Regions
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.16551v2](http://arxiv.org/pdf/2503.16551v2)**

> **作者:** Songqiao Hu; Zidong Wang; Zeyi Liu; Zhen Shen; Xiao He
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Control barrier functions (CBFs) provide a theoretical foundation for safety-critical control in robotic systems. However, most existing methods rely on the analytical expressions of unsafe state regions, which are often impractical for irregular and dynamic unsafe regions. This paper introduces SafeLink, a novel CBF construction method based on cost-sensitive incremental random vector functional-link (RVFL) neural networks. By designing a valid cost function, SafeLink assigns different sensitivities to safe and unsafe state points, thereby eliminating false negatives in classification of unsafe state points. Furthermore, an incremental update theorem is established, enabling precise real-time adaptation to changes in unsafe regions. An analytical expression for the gradient of SafeLink is also derived to facilitate control input computation. The proposed method is validated on the endpoint position control task of a nonlinear two-link manipulator. Experimental results demonstrate that the method effectively learns the unsafe regions and rapidly adapts as these regions change, achieving an update speed significantly faster than comparison methods, while safely reaching the target position. The source code is available at https://github.com/songqiaohu/SafeLink.
>
---
#### [replaced 022] Temporal Preference Optimization for Long-Form Video Understanding
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.13919v3](http://arxiv.org/pdf/2501.13919v3)**

> **作者:** Rui Li; Xiaohan Wang; Yuhui Zhang; Orr Zohar; Zeyu Wang; Serena Yeung-Levy
>
> **摘要:** Despite significant advancements in video large multimodal models (video-LMMs), achieving effective temporal grounding in long-form videos remains a challenge for existing models. To address this limitation, we propose Temporal Preference Optimization (TPO), a novel post-training framework designed to enhance the temporal grounding capabilities of video-LMMs through preference learning. TPO adopts a self-training approach that enables models to differentiate between well-grounded and less accurate temporal responses by leveraging curated preference datasets at two granularities: localized temporal grounding, which focuses on specific video segments, and comprehensive temporal grounding, which captures extended temporal dependencies across entire video sequences. By optimizing on these preference datasets, TPO significantly enhances temporal understanding while reducing reliance on manually annotated data. Extensive experiments on three long-form video understanding benchmarks--LongVideoBench, MLVU, and Video-MME--demonstrate the effectiveness of TPO across two state-of-the-art video-LMMs. Notably, LLaVA-Video-TPO establishes itself as the leading 7B model on the Video-MME benchmark, underscoring the potential of TPO as a scalable and efficient solution for advancing temporal reasoning in long-form video understanding. Project page: https://ruili33.github.io/tpo_website.
>
---
#### [replaced 023] General agents contain world models
- **分类: cs.AI; cs.LG; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2506.01622v4](http://arxiv.org/pdf/2506.01622v4)**

> **作者:** Jonathan Richens; David Abel; Alexis Bellot; Tom Everitt
>
> **备注:** Accepted ICML 2025. Typos corrected
>
> **摘要:** Are world models a necessary ingredient for flexible, goal-directed behaviour, or is model-free learning sufficient? We provide a formal answer to this question, showing that any agent capable of generalizing to multi-step goal-directed tasks must have learned a predictive model of its environment. We show that this model can be extracted from the agent's policy, and that increasing the agents performance or the complexity of the goals it can achieve requires learning increasingly accurate world models. This has a number of consequences: from developing safe and general agents, to bounding agent capabilities in complex environments, and providing new algorithms for eliciting world models from agents.
>
---
#### [replaced 024] Toward Real-World Cooperative and Competitive Soccer with Quadrupedal Robot Teams
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13834v2](http://arxiv.org/pdf/2505.13834v2)**

> **作者:** Zhi Su; Yuman Gao; Emily Lukas; Yunfei Li; Jiaze Cai; Faris Tulbah; Fei Gao; Chao Yu; Zhongyu Li; Yi Wu; Koushil Sreenath
>
> **备注:** 11 pages, 12 figures, CoRL 2025
>
> **摘要:** Achieving coordinated teamwork among legged robots requires both fine-grained locomotion control and long-horizon strategic decision-making. Robot soccer offers a compelling testbed for this challenge, combining dynamic, competitive, and multi-agent interactions. In this work, we present a hierarchical multi-agent reinforcement learning (MARL) framework that enables fully autonomous and decentralized quadruped robot soccer. First, a set of highly dynamic low-level skills is trained for legged locomotion and ball manipulation, such as walking, dribbling, and kicking. On top of these, a high-level strategic planning policy is trained with Multi-Agent Proximal Policy Optimization (MAPPO) via Fictitious Self-Play (FSP). This learning framework allows agents to adapt to diverse opponent strategies and gives rise to sophisticated team behaviors, including coordinated passing, interception, and dynamic role allocation. With an extensive ablation study, the proposed learning method shows significant advantages in the cooperative and competitive multi-agent soccer game. We deploy the learned policies to real quadruped robots relying solely on onboard proprioception and decentralized localization, with the resulting system supporting autonomous robot-robot and robot-human soccer matches on indoor and outdoor soccer courts.
>
---
#### [replaced 025] Stretchable Electrohydraulic Artificial Muscle for Full Motion Ranges in Musculoskeletal Antagonistic Joints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11017v2](http://arxiv.org/pdf/2409.11017v2)**

> **作者:** Amirhossein Kazemipour; Ronan Hinchet; Robert K. Katzschmann
>
> **备注:** This paper has been accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** Artificial muscles play a crucial role in musculoskeletal robotics and prosthetics to approximate the force-generating functionality of biological muscle. However, current artificial muscle systems are typically limited to either contraction or extension, not both. This limitation hinders the development of fully functional artificial musculoskeletal systems. We address this challenge by introducing an artificial antagonistic muscle system capable of both contraction and extension. Our design integrates non-stretchable electrohydraulic soft actuators (HASELs) with electrostatic clutches within an antagonistic musculoskeletal framework. This configuration enables an antagonistic joint to achieve a full range of motion without displacement loss due to tendon slack. We implement a synchronization method to coordinate muscle and clutch units, ensuring smooth motion profiles and speeds. This approach facilitates seamless transitions between antagonistic muscles at operational frequencies of up to 3.2 Hz. While our prototype utilizes electrohydraulic actuators, this muscle-clutch concept is adaptable to other non-stretchable artificial muscles, such as McKibben actuators, expanding their capability for extension and full range of motion in antagonistic setups. Our design represents a significant advancement in the development of fundamental components for more functional and efficient artificial musculoskeletal systems, bringing their capabilities closer to those of their biological counterparts.
>
---
#### [replaced 026] ViTaMIn: Learning Contact-Rich Tasks Through Robot-Free Visuo-Tactile Manipulation Interface
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06156v2](http://arxiv.org/pdf/2504.06156v2)**

> **作者:** Fangchen Liu; Chuanyu Li; Yihua Qin; Jing Xu; Pieter Abbeel; Rui Chen
>
> **摘要:** Tactile information plays a crucial role for humans and robots to interact effectively with their environment, particularly for tasks requiring the understanding of contact properties. Solving such dexterous manipulation tasks often relies on imitation learning from demonstration datasets, which are typically collected via teleoperation systems and often demand substantial time and effort. To address these challenges, we present ViTaMIn, an embodiment-free manipulation interface that seamlessly integrates visual and tactile sensing into a hand-held gripper, enabling data collection without the need for teleoperation. Our design employs a compliant Fin Ray gripper with tactile sensing, allowing operators to perceive force feedback during manipulation for more intuitive operation. Additionally, we propose a multimodal representation learning strategy to obtain pre-trained tactile representations, improving data efficiency and policy robustness. Experiments on seven contact-rich manipulation tasks demonstrate that ViTaMIn significantly outperforms baseline methods, demonstrating its effectiveness for complex manipulation tasks.
>
---
#### [replaced 027] Goal-Conditioned Data Augmentation for Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.20519v2](http://arxiv.org/pdf/2412.20519v2)**

> **作者:** Xingshuai Huang; Di Wu; Benoit Boulet
>
> **摘要:** Offline reinforcement learning (RL) enables policy learning from pre-collected offline datasets, relaxing the need to interact directly with the environment. However, limited by the quality of offline datasets, it generally fails to learn well-qualified policies in suboptimal datasets. To address datasets with insufficient optimal demonstrations, we introduce Goal-cOnditioned Data Augmentation (GODA), a novel goal-conditioned diffusion-based method for augmenting samples with higher quality. Leveraging recent advancements in generative modelling, GODA incorporates a novel return-oriented goal condition with various selection mechanisms. Specifically, we introduce a controllable scaling technique to provide enhanced return-based guidance during data sampling. GODA learns a comprehensive distribution representation of the original offline datasets while generating new data with selectively higher-return goals, thereby maximizing the utility of limited optimal demonstrations. Furthermore, we propose a novel adaptive gated conditioning method for processing noisy inputs and conditions, enhancing the capture of goal-oriented guidance. We conduct experiments on the D4RL benchmark and real-world challenges, specifically traffic signal control (TSC) tasks, to demonstrate GODA's effectiveness in enhancing data quality and superior performance compared to state-of-the-art data augmentation methods across various offline RL algorithms.
>
---
#### [replaced 028] Wavelet Policy: Imitation Policy Learning in the Scale Domain with Wavelet Transforms
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04991v3](http://arxiv.org/pdf/2504.04991v3)**

> **作者:** Changchuan Yang; Yuhang Dong; Guanzhong Tian; Haizhou Ge; Hongrui Zhu
>
> **摘要:** Recent imitation learning policies, often framed as time series prediction tasks, directly map robotic observations into the action space, such as high-dimensional visual data and proprioception. When deploying at the edge, we found the underutilization of frequency domain analysis in robotic manipulation trajectory prediction leads to neglecting the inherent rhythm information embedded within action sequences, resulting in errors at critical moments. To address this, we reframe imitation learning policies through the lens of time-scale domain and introduce the Wavelet Policy. This novel approach employs wavelet transforms (WT) and new Features Extractor (FE) for feature preprocessing and extracts multi-scale features using the Single Encoder to Multiple Decoder (SE2MD) architecture. Furthermore, to enhance feature mapping in the scale domain and appropriately increase model capacity, we introduce a Learnable Scale Domain Filter (LSDF) after each decoder, improving adaptability under different visual conditions. Our results show that the Wavelet Policy maintaining a comparable parameter count outperforms SOTA end-to-end methods on four challenging simulation robotic arm tasks and real tasks, especially at critical moments and remote settings simultaneously. We release the source code and model checkpoint of simulation task at https://github.com/lurenjia384/Wavelet_Policy.
>
---
#### [replaced 029] YORI: Autonomous Cooking System Utilizing a Modular Robotic Kitchen and a Dual-Arm Proprioceptive Manipulator
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.11094v2](http://arxiv.org/pdf/2405.11094v2)**

> **作者:** Donghun Noh; Hyunwoo Nam; Kyle Gillespie; Yeting Liu; Dennis Hong
>
> **备注:** This work has been submitted to IEEE Robotics & Automation Magazine for possible publication
>
> **摘要:** This paper presents Yummy Operations Robot Initiative (YORI), a proprioceptive dual-arm robotic system that demonstrates autonomous multi-dish cooking for scalable food service applications. YORI integrates a dual-arm manipulator equipped with proprioceptive actuators, custom-designed tools, appliances, and a structured kitchen environment to address the complexities of cooking tasks. The proprioceptive actuators enable fast, precise, force-controlled movements while mitigating the risks associated with cooking-related impacts. The system's modular kitchen design and flexible tool-changing mechanism support simultaneous multi-dish preparation through torque control and optimization-based motion planning and scheduling. A comprehensive scheduling framework with dynamic rescheduling ensures reliable adaptation to new orders and delays. The system was publicly validated through live demonstrations, reliably preparing steak-frites across multiple convention sessions. This paper details YORI's design and explores future directions in kitchen optimization, task planning, and food quality control, demonstrating its potential as a scalable robotic cooking solution. A system introduction and cooking videos are available online
>
---
#### [replaced 030] Autonomous Task Planning for Heterogeneous Multi-Agent Systems
- **分类: cs.RO; cs.FL**

- **链接: [http://arxiv.org/pdf/2209.08611v2](http://arxiv.org/pdf/2209.08611v2)**

> **作者:** Anatoli A. Tziola; Savvas G. Loizou
>
> **备注:** Long version of paper submitted to the IEEE ICRA 2023 Conference
>
> **摘要:** This paper presents a solution to the automatic task planning problem for multi-agent systems. A formal framework is developed based on the Nondeterministic Finite Automata with $\epsilon$-transitions, where given the capabilities, constraints and failure modes of the agents involved, an initial state of the system and a task specification, an optimal solution is generated that satisfies the system constraints and the task specification. The resulting solution is guaranteed to be complete and optimal; moreover a heuristic solution that offers significant reduction of the computational requirements while relaxing the completeness and optimality requirements is proposed. The constructed system model is independent from the initial condition and the task specification, alleviating the need to repeat the costly pre-processing cycle for solving other scenarios, while allowing the incorporation of failure modes on-the-fly. Two case studies are provided: a simple one to showcase the concepts of the proposed methodology and a more elaborate one to demonstrate the effectiveness and validity of the methodology.
>
---
#### [replaced 031] Towards a cognitive architecture to enable natural language interaction in co-constructive task learning
- **分类: cs.RO; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.23760v3](http://arxiv.org/pdf/2503.23760v3)**

> **作者:** Manuel Scheibl; Birte Richter; Alissa Müller; Michael Beetz; Britta Wrede
>
> **备注:** 8 pages, 5 figures, The paper has been accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (ROMAN), IEEE Copyright Policy: https://www.ieee.org/publications/rights/copyright-policy
>
> **摘要:** This research addresses the question, which characteristics a cognitive architecture must have to leverage the benefits of natural language in Co-Constructive Task Learning (CCTL). To provide context, we first discuss Interactive Task Learning (ITL), the mechanisms of the human memory system, and the significance of natural language and multi-modality. Next, we examine the current state of cognitive architectures, analyzing their capabilities to inform a concept of CCTL grounded in multiple sources. We then integrate insights from various research domains to develop a unified framework. Finally, we conclude by identifying the remaining challenges and requirements necessary to achieve CCTL in Human-Robot Interaction (HRI).
>
---
#### [replaced 032] Wheeled Lab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.07380v2](http://arxiv.org/pdf/2502.07380v2)**

> **作者:** Tyler Han; Preet Shah; Sidharth Rajagopal; Yanda Bao; Sanghun Jung; Sidharth Talia; Gabriel Guo; Bryan Xu; Bhaumik Mehta; Emma Romig; Rosario Scalise; Byron Boots
>
> **备注:** To appear at Conference on Robot Learning, 2025
>
> **摘要:** Reinforcement Learning (RL) has been pivotal in recent robotics milestones and is poised to play a prominent role in the future. However, these advances can rely on proprietary simulators, expensive hardware, and a daunting range of tools and skills. As a result, broader communities are disconnecting from the state-of-the-art; education curricula are poorly equipped to teach indispensable modern robotics skills involving hardware, deployment, and iterative development. To address this gap between the broader and scientific communities, we contribute Wheeled Lab, an ecosystem which integrates accessible, open-source wheeled robots with Isaac Lab, an open-source robot learning and simulation framework, that is widely adopted in the state-of-the-art. To kickstart research and education, this work demonstrates three state-of-the-art zero-shot policies for small-scale RC cars developed through Wheeled Lab: controlled drifting, elevation traversal, and visual navigation. The full stack, from hardware to software, is low-cost and open-source. Videos and additional materials can be found at: https://uwrobotlearning.github.io/WheeledLab/
>
---
#### [replaced 033] Nav-SCOPE: Swarm Robot Cooperative Perception and Coordinated Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.10049v3](http://arxiv.org/pdf/2409.10049v3)**

> **作者:** Chenxi Li; Weining Lu; Qingquan Lin; Litong Meng; Haolu Li; Bin Liang
>
> **备注:** 11 pages, 9 figures, accepted in IEEE Transactions on Automation Science and Engineering
>
> **摘要:** This paper proposes a lightweight systematic solution for multi-robot coordinated navigation with decentralized cooperative perception. An information flow is first created to facilitate real-time observation sharing over unreliable ad-hoc networks. Then, the environmental uncertainties of each robot are reduced by interaction fields that deliver complementary information. Finally, path optimization is achieved, enabling self-organized coordination with effective convergence, divergence, and collision avoidance. Our method is fully interpretable and ready for deployment without gaps. Comprehensive simulations and real-world experiments demonstrate reduced path redundancy, robust performance across various tasks, and minimal demands on computation and communication.
>
---
#### [replaced 034] Computational Design and Fabrication of Modular Robots with Untethered Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.05410v2](http://arxiv.org/pdf/2508.05410v2)**

> **作者:** Manas Bhargava; Takefumi Hiraki; Malina Strugaru; Yuhan Zhang; Michal Piovarci; Chiara Daraio; Daisuke Iwai; Bernd Bickel
>
> **摘要:** Natural organisms utilize distributed actuation through their musculoskeletal systems to adapt their gait for traversing diverse terrains or to morph their bodies for varied tasks. A longstanding challenge in robotics is to emulate this capability of natural organisms, which has motivated the development of numerous soft robotic systems. However, such systems are generally optimized for a single functionality, lack the ability to change form or function on demand, or remain tethered to bulky control systems. To address these limitations, we present a framework for designing and controlling robots that utilize distributed actuation. We propose a novel building block that integrates 3D-printed bones with liquid crystal elastomer (LCE) muscles as lightweight actuators, enabling the modular assembly of musculoskeletal robots. We developed LCE rods that contract in response to infrared radiation, thereby providing localized, untethered control over the distributed skeletal network and producing global deformations of the robot. To fully capitalize on the extensive design space, we introduce two computational tools: one for optimizing the robot's skeletal graph to achieve multiple target deformations, and another for co-optimizing skeletal designs and control gaits to realize desired locomotion. We validate our framework by constructing several robots that demonstrate complex shape morphing, diverse control schemes, and environmental adaptability. Our system integrates advances in modular material building, untethered and distributed control, and computational design to introduce a new generation of robots that brings us closer to the capabilities of living organisms.
>
---
#### [replaced 035] Unscented Kalman Filter with a Nonlinear Propagation Model for Navigation Applications
- **分类: cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2507.10082v4](http://arxiv.org/pdf/2507.10082v4)**

> **作者:** Amit Levy; Itzik Klein
>
> **备注:** 6 pages, 4 figures
>
> **摘要:** The unscented Kalman filter is a nonlinear estimation algorithm commonly used in navigation applications. The prediction of the mean and covariance matrix is crucial to the stable behavior of the filter. This prediction is done by propagating the sigma points according to the dynamic model at hand. In this paper, we introduce an innovative method to propagate the sigma points according to the nonlinear dynamic model of the navigation error state vector. This improves the filter accuracy and navigation performance. We demonstrate the benefits of our proposed approach using real sensor data recorded by an autonomous underwater vehicle during several scenarios.
>
---
#### [replaced 036] RALLY: Role-Adaptive LLM-Driven Yoked Navigation for Agentic UAV Swarms
- **分类: cs.MA; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.01378v2](http://arxiv.org/pdf/2507.01378v2)**

> **作者:** Ziyao Wang; Rongpeng Li; Sizhao Li; Yuming Xiang; Haiping Wang; Zhifeng Zhao; Honggang Zhang
>
> **摘要:** Intelligent control of Unmanned Aerial Vehicles (UAVs) swarms has emerged as a critical research focus, and it typically requires the swarm to navigate effectively while avoiding obstacles and achieving continuous coverage over multiple mission targets. Although traditional Multi-Agent Reinforcement Learning (MARL) approaches offer dynamic adaptability, they are hindered by the semantic gap in numerical communication and the rigidity of homogeneous role structures, resulting in poor generalization and limited task scalability. Recent advances in Large Language Model (LLM)-based control frameworks demonstrate strong semantic reasoning capabilities by leveraging extensive prior knowledge. However, due to the lack of online learning and over-reliance on static priors, these works often struggle with effective exploration, leading to reduced individual potential and overall system performance. To address these limitations, we propose a Role-Adaptive LLM-Driven Yoked navigation algorithm RALLY. Specifically, we first develop an LLM-driven semantic decision framework that uses structured natural language for efficient semantic communication and collaborative reasoning. Afterward, we introduce a dynamic role-heterogeneity mechanism for adaptive role switching and personalized decision-making. Furthermore, we propose a Role-value Mixing Network (RMIX)-based assignment strategy that integrates LLM offline priors with MARL online policies to enable semi-offline training of role selection strategies. Experiments in the Multi-Agent Particle Environment (MPE) environment and a Software-In-The-Loop (SITL) platform demonstrate that RALLY outperforms conventional approaches in terms of task coverage, convergence speed, and generalization, highlighting its strong potential for collaborative navigation in agentic multi-UAV systems.
>
---
#### [replaced 037] NarraGuide: an LLM-based Narrative Mobile Robot for Remote Place Exploration
- **分类: cs.HC; cs.RO; 68**

- **链接: [http://arxiv.org/pdf/2508.01235v3](http://arxiv.org/pdf/2508.01235v3)**

> **作者:** Yaxin Hu; Arissa J. Sato; Jingxin Du; Chenming Ye; Anjun Zhu; Pragathi Praveena; Bilge Mutlu
>
> **摘要:** Robotic telepresence enables users to navigate and experience remote environments. However, effective navigation and situational awareness depend on users' prior knowledge of the environment, limiting the usefulness of these systems for exploring unfamiliar places. We explore how integrating location-aware LLM-based narrative capabilities into a mobile robot can support remote exploration. We developed a prototype system, called NarraGuide, that provides narrative guidance for users to explore and learn about a remote place through a dialogue-based interface. We deployed our prototype in a geology museum, where remote participants (n=20) used the robot to tour the museum. Our findings reveal how users perceived the robot's role, engaged in dialogue in the tour, and expressed preferences for bystander encountering. Our work demonstrates the potential of LLM-enabled robotic capabilities to deliver location-aware narrative guidance and enrich the experience of exploring remote environments.
>
---
#### [replaced 038] Morphologically Symmetric Reinforcement Learning for Ambidextrous Bimanual Manipulation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05287v2](http://arxiv.org/pdf/2505.05287v2)**

> **作者:** Zechu Li; Yufeng Jin; Daniel Ordonez Apraez; Claudio Semini; Puze Liu; Georgia Chalvatzaki
>
> **摘要:** Humans naturally exhibit bilateral symmetry in their gross manipulation skills, effortlessly mirroring simple actions between left and right hands. Bimanual robots-which also feature bilateral symmetry-should similarly exploit this property to perform tasks with either hand. Unlike humans, who often favor a dominant hand for fine dexterous skills, robots should ideally execute ambidextrous manipulation with equal proficiency. To this end, we introduce SYMDEX (SYMmetric DEXterity), a reinforcement learning framework for ambidextrous bi-manipulation that leverages the robot's inherent bilateral symmetry as an inductive bias. SYMDEX decomposes complex bimanual manipulation tasks into per-hand subtasks and trains dedicated policies for each. By exploiting bilateral symmetry via equivariant neural networks, experience from one arm is inherently leveraged by the opposite arm. We then distill the subtask policies into a global ambidextrous policy that is independent of the hand-task assignment. We evaluate SYMDEX on six challenging simulated manipulation tasks and demonstrate successful real-world deployment on two of them. Our approach strongly outperforms baselines on complex task in which the left and right hands perform different roles. We further demonstrate SYMDEX's scalability by extending it to a four-arm manipulation setup, where our symmetry-aware policies enable effective multi-arm collaboration and coordination. Our results highlight how structural symmetry as inductive bias in policy learning enhances sample efficiency, robustness, and generalization across diverse dexterous manipulation tasks.
>
---
#### [replaced 039] Efficient Online Learning and Adaptive Planning for Robotic Information Gathering Based on Streaming Data
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13053v2](http://arxiv.org/pdf/2507.13053v2)**

> **作者:** Sanjeev Ramkumar Sudha; Joel Jose; Erlend M. Coates
>
> **备注:** Accepted for presentation at 2025 European Conference on Mobile Robots
>
> **摘要:** Robotic information gathering (RIG) techniques refer to methods where mobile robots are used to acquire data about the physical environment with a suite of sensors. Informative planning is an important part of RIG where the goal is to find sequences of actions or paths that maximize efficiency or the quality of information collected. Many existing solutions solve this problem by assuming that the environment is known in advance. However, real environments could be unknown or time-varying, and adaptive informative planning remains an active area of research. Adaptive planning and incremental online mapping are required for mapping initially unknown or varying spatial fields. Gaussian process (GP) regression is a widely used technique in RIG for mapping continuous spatial fields. However, it falls short in many applications as its real-time performance does not scale well to large datasets. To address these challenges, this paper proposes an efficient adaptive informative planning approach for mapping continuous scalar fields with GPs with streaming sparse GPs. Simulation experiments are performed with a synthetic dataset and compared against existing benchmarks. Finally, it is also verified with a real-world dataset to further validate the efficacy of the proposed method. Results show that our method achieves similar mapping accuracy to the baselines while reducing computational complexity for longer missions.
>
---
#### [replaced 040] SoK: Cybersecurity Assessment of Humanoid Ecosystem
- **分类: cs.CR; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.17481v2](http://arxiv.org/pdf/2508.17481v2)**

> **作者:** Priyanka Prakash Surve; Asaf Shabtai; Yuval Elovici
>
> **摘要:** Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.
>
---
#### [replaced 041] ExoStart: Efficient learning for dexterous manipulation with sensorized exoskeleton demonstrations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.11775v2](http://arxiv.org/pdf/2506.11775v2)**

> **作者:** Zilin Si; Jose Enrique Chen; M. Emre Karagozler; Antonia Bronars; Jonathan Hutchinson; Thomas Lampe; Nimrod Gileadi; Taylor Howell; Stefano Saliceti; Lukasz Barczyk; Ilan Olivarez Correa; Tom Erez; Mohit Shridhar; Murilo Fernandes Martins; Konstantinos Bousmalis; Nicolas Heess; Francesco Nori; Maria Bauza Villalonga
>
> **摘要:** Recent advancements in teleoperation systems have enabled high-quality data collection for robotic manipulators, showing impressive results in learning manipulation at scale. This progress suggests that extending these capabilities to robotic hands could unlock an even broader range of manipulation skills, especially if we could achieve the same level of dexterity that human hands exhibit. However, teleoperating robotic hands is far from a solved problem, as it presents a significant challenge due to the high degrees of freedom of robotic hands and the complex dynamics occurring during contact-rich settings. In this work, we present ExoStart, a general and scalable learning framework that leverages human dexterity to improve robotic hand control. In particular, we obtain high-quality data by collecting direct demonstrations without a robot in the loop using a sensorized low-cost wearable exoskeleton, capturing the rich behaviors that humans can demonstrate with their own hands. We also propose a simulation-based dynamics filter that generates dynamically feasible trajectories from the collected demonstrations and use the generated trajectories to bootstrap an auto-curriculum reinforcement learning method that relies only on simple sparse rewards. The ExoStart pipeline is generalizable and yields robust policies that transfer zero-shot to the real robot. Our results demonstrate that ExoStart can generate dexterous real-world hand skills, achieving a success rate above 50% on a wide range of complex tasks such as opening an AirPods case or inserting and turning a key in a lock. More details and videos can be found in https://sites.google.com/view/exostart.
>
---
#### [replaced 042] Towards Efficient LLM Grounding for Embodied Multi-Agent Collaboration
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.14314v3](http://arxiv.org/pdf/2405.14314v3)**

> **作者:** Yang Zhang; Shixin Yang; Chenjia Bai; Fei Wu; Xiu Li; Zhen Wang; Xuelong Li
>
> **备注:** accepted by ACL'2025
>
> **摘要:** Grounding the reasoning ability of large language models (LLMs) for embodied tasks is challenging due to the complexity of the physical world. Especially, LLM planning for multi-agent collaboration requires communication of agents or credit assignment as the feedback to re-adjust the proposed plans and achieve effective coordination. However, existing methods that overly rely on physical verification or self-reflection suffer from excessive and inefficient querying of LLMs. In this paper, we propose a novel framework for multi-agent collaboration that introduces Reinforced Advantage feedback (ReAd) for efficient self-refinement of plans. Specifically, we perform critic regression to learn a sequential advantage function from LLM-planned data, and then treat the LLM planner as an optimizer to generate actions that maximize the advantage function. It endows the LLM with the foresight to discern whether the action contributes to accomplishing the final task. We provide theoretical analysis by extending advantage-weighted regression in reinforcement learning to multi-agent systems. Experiments on Overcooked-AI and a difficult variant of RoCoBench show that ReAd surpasses baselines in success rate, and also significantly decreases the interaction steps of agents and query rounds of LLMs, demonstrating its high efficiency for grounding LLMs. More results are given at https://read-llm.github.io.
>
---
#### [replaced 043] EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21112v2](http://arxiv.org/pdf/2508.21112v2)**

> **作者:** Delin Qu; Haoming Song; Qizhi Chen; Zhaoqing Chen; Xianqiang Gao; Xinyi Ye; Qi Lv; Modi Shi; Guanghui Ren; Cheng Ruan; Maoqing Yao; Haoran Yang; Jiacheng Bao; Bin Zhao; Dong Wang
>
> **摘要:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models.
>
---
#### [replaced 044] Diffusion Dynamics Models with Generative State Estimation for Cloth Manipulation
- **分类: cs.RO; cs.CV; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.11999v2](http://arxiv.org/pdf/2503.11999v2)**

> **作者:** Tongxuan Tian; Haoyang Li; Bo Ai; Xiaodi Yuan; Zhiao Huang; Hao Su
>
> **备注:** CoRL 2025. Project website: https://uniclothdiff.github.io/
>
> **摘要:** Cloth manipulation is challenging due to its highly complex dynamics, near-infinite degrees of freedom, and frequent self-occlusions, which complicate both state estimation and dynamics modeling. Inspired by recent advances in generative models, we hypothesize that these expressive models can effectively capture intricate cloth configurations and deformation patterns from data. Therefore, we propose a diffusion-based generative approach for both perception and dynamics modeling. Specifically, we formulate state estimation as reconstructing full cloth states from partial observations and dynamics modeling as predicting future states given the current state and robot actions. Leveraging a transformer-based diffusion model, our method achieves accurate state reconstruction and reduces long-horizon dynamics prediction errors by an order of magnitude compared to prior approaches. We integrate our dynamics models with model predictive control and show that our framework enables effective cloth folding on real robotic systems, demonstrating the potential of generative models for deformable object manipulation under partial observability and complex dynamics.
>
---
#### [replaced 045] Energy-Aware Lane Planning for Connected Electric Vehicles in Urban Traffic: Design and Vehicle-in-the-Loop Validation
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2503.23228v2](http://arxiv.org/pdf/2503.23228v2)**

> **作者:** Hansung Kim; Eric Yongkeun Choi; Eunhyek Joa; Hotae Lee; Linda Lim; Scott Moura; Francesco Borrelli
>
> **备注:** Accepted at 2025 IEEE Conference on Decision and Control (CDC25')
>
> **摘要:** Urban driving with connected and automated vehicles (CAVs) offers potential for energy savings, yet most eco-driving strategies focus solely on longitudinal speed control within a single lane. This neglects the significant impact of lateral decisions, such as lane changes, on overall energy efficiency, especially in environments with traffic signals and heterogeneous traffic flow. To address this gap, we propose a novel energy-aware motion planning framework that jointly optimizes longitudinal speed and lateral lane-change decisions using vehicle-to-infrastructure (V2I) communication. Our approach estimates long-term energy costs using a graph-based approximation and solves short-horizon optimal control problems under traffic constraints. Using a data-driven energy model calibrated to an actual battery electric vehicle, we demonstrate with vehicle-in-the-loop experiments that our method reduces motion energy consumption by up to 24 percent compared to a human driver, highlighting the potential of connectivity-enabled planning for sustainable urban autonomy.
>
---
#### [replaced 046] SAVOR: Skill Affordance Learning from Visuo-Haptic Perception for Robot-Assisted Bite Acquisition
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02353v2](http://arxiv.org/pdf/2506.02353v2)**

> **作者:** Zhanxin Wu; Bo Ai; Tom Silver; Tapomayukh Bhattacharjee
>
> **备注:** Conference on Robot Learning, Oral
>
> **摘要:** Robot-assisted feeding requires reliable bite acquisition, a challenging task due to the complex interactions between utensils and food with diverse physical properties. These interactions are further complicated by the temporal variability of food properties-for example, steak becomes firm as it cools even during a meal. To address this, we propose SAVOR, a novel approach for learning skill affordances for bite acquisition-how suitable a manipulation skill (e.g., skewering, scooping) is for a given utensil-food interaction. In our formulation, skill affordances arise from the combination of tool affordances (what a utensil can do) and food affordances (what the food allows). Tool affordances are learned offline through calibration, where different utensils interact with a variety of foods to model their functional capabilities. Food affordances are characterized by physical properties such as softness, moisture, and viscosity, initially inferred through commonsense reasoning using a visually-conditioned language model and then dynamically refined through online multi-modal visuo-haptic perception using SAVOR-Net during interaction. Our method integrates these offline and online estimates to predict skill affordances in real time, enabling the robot to select the most appropriate skill for each food item. Evaluated on 20 single-item foods and 10 in-the-wild meals, our approach improves bite acquisition success rate by 13% over state-of-the-art (SOTA) category-based methods (e.g. use skewer for fruits). These results highlight the importance of modeling interaction-driven skill affordances for generalizable and effective robot-assisted bite acquisition. Website: https://emprise.cs.cornell.edu/savor/
>
---
#### [replaced 047] Large VLM-based Vision-Language-Action Models for Robotic Manipulation: A Survey
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13073v2](http://arxiv.org/pdf/2508.13073v2)**

> **作者:** Rui Shao; Wei Li; Lingsen Zhang; Renshan Zhang; Zhiyang Liu; Ran Chen; Liqiang Nie
>
> **备注:** Project Page: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
>
> **摘要:** Robotic manipulation, a key frontier in robotics and embodied AI, requires precise motor control and multimodal understanding, yet traditional rule-based methods fail to scale or generalize in unstructured, novel environments. In recent years, Vision-Language-Action (VLA) models, built upon Large Vision-Language Models (VLMs) pretrained on vast image-text datasets, have emerged as a transformative paradigm. This survey provides the first systematic, taxonomy-oriented review of large VLM-based VLA models for robotic manipulation. We begin by clearly defining large VLM-based VLA models and delineating two principal architectural paradigms: (1) monolithic models, encompassing single-system and dual-system designs with differing levels of integration; and (2) hierarchical models, which explicitly decouple planning from execution via interpretable intermediate representations. Building on this foundation, we present an in-depth examination of large VLM-based VLA models: (1) integration with advanced domains, including reinforcement learning, training-free optimization, learning from human videos, and world model integration; (2) synthesis of distinctive characteristics, consolidating architectural traits, operational strengths, and the datasets and benchmarks that support their development; (3) identification of promising directions, including memory mechanisms, 4D perception, efficient adaptation, multi-agent cooperation, and other emerging capabilities. This survey consolidates recent advances to resolve inconsistencies in existing taxonomies, mitigate research fragmentation, and fill a critical gap through the systematic integration of studies at the intersection of large VLMs and robotic manipulation. We provide a regularly updated project page to document ongoing progress: https://github.com/JiuTian-VL/Large-VLM-based-VLA-for-Robotic-Manipulation
>
---
#### [replaced 048] Enhancing Security in Multi-Robot Systems through Co-Observation Planning, Reachability Analysis, and Network Flow
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2403.13266v2](http://arxiv.org/pdf/2403.13266v2)**

> **作者:** Ziqi Yang; Roberto Tron
>
> **备注:** 12 pages, 6 figures, submitted to IEEE Transactions on Control of Network Systems
>
> **摘要:** This paper addresses security challenges in multi-robot systems (MRS) where adversaries may compromise robot control, risking unauthorized access to forbidden areas. We propose a novel multi-robot optimal planning algorithm that integrates mutual observations and introduces reachability constraints for enhanced security. This ensures that, even with adversarial movements, compromised robots cannot breach forbidden regions without missing scheduled co-observations. The reachability constraint uses ellipsoidal over-approximation for efficient intersection checking and gradient computation. To enhance system resilience and tackle feasibility challenges, we also introduce sub-teams. These cohesive units replace individual robot assignments along each route, enabling redundant robots to deviate for co-observations across different trajectories, securing multiple sub-teams without requiring modifications. We formulate the cross-trajectory co-observation plan by solving a network flow coverage problem on the checkpoint graph generated from the original unsecured MRS trajectories, providing the same security guarantees against plan-deviation attacks. We demonstrate the effectiveness and robustness of our proposed algorithm, which significantly strengthens the security of multi-robot systems in the face of adversarial threats.
>
---
#### [replaced 049] RobotxR1: Enabling Embodied Robotic Intelligence on Large Language Models through Closed-Loop Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.03238v2](http://arxiv.org/pdf/2505.03238v2)**

> **作者:** Liam Boyle; Nicolas Baumann; Paviththiren Sivasothilingam; Michele Magno; Luca Benini
>
> **摘要:** Future robotic systems operating in real-world environments will require on-board embodied intelligence without continuous cloud connection, balancing capabilities with constraints on computational power and memory. This work presents an extension of the R1-zero approach, which enables the usage of low parameter-count Large Language Models (LLMs) in the robotic domain. The R1-Zero approach was originally developed to enable mathematical reasoning in LLMs using static datasets. We extend it to the robotics domain through integration in a closed-loop Reinforcement Learning (RL) framework. This extension enhances reasoning in Embodied Artificial Intelligence (Embodied AI) settings without relying solely on distillation of large models through Supervised Fine-Tuning (SFT). We show that small-scale LLMs can achieve effective reasoning performance by learning through closed-loop interaction with their environment, which enables tasks that previously required significantly larger models. In an autonomous driving setting, a performance gain of 20.2%-points over the SFT-based baseline is observed with a Qwen2.5-1.5B model. Using the proposed training procedure, Qwen2.5-3B achieves a 63.3% control adaptability score, surpassing the 58.5% obtained by the much larger, cloud-bound GPT-4o. These results highlight that practical, on-board deployment of small LLMs is not only feasible but can outperform larger models if trained through environmental feedback, underscoring the importance of an interactive learning framework for robotic Embodied AI, one grounded in practical experience rather than static supervision.
>
---
#### [replaced 050] NMPCB: A Lightweight and Safety-Critical Motion Control Framework for Ackermann Mobile Robot
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.01752v2](http://arxiv.org/pdf/2505.01752v2)**

> **作者:** Longze Zheng; Qinghe Liu
>
> **摘要:** In multi-obstacle environments, real-time performance and safety in robot motion control have long been challenging issues, as conventional methods often struggle to balance the two. In this paper, we propose a novel motion control framework composed of a Neural network-based path planner and a Model Predictive Control (MPC) controller based on control Barrier function (NMPCB) . The planner predicts the next target point through a lightweight neural network and generates a reference trajectory for the controller. In the design of the controller, we introduce the dual problem of control barrier function (CBF) as the obstacle avoidance constraint, enabling it to ensure robot motion safety while significantly reducing computation time. The controller directly outputs control commands to the robot by tracking the reference trajectory. This framework achieves a balance between real-time performance and safety. We validate the feasibility of the framework through numerical simulations and real-world experiments.
>
---
#### [replaced 051] Robust Deterministic Policy Gradient for Disturbance Attenuation and Its Application to Quadrotor Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.21057v4](http://arxiv.org/pdf/2502.21057v4)**

> **作者:** Taeho Lee; Donghwan Lee
>
> **备注:** 24 pages
>
> **摘要:** Practical control systems pose significant challenges in identifying optimal control policies due to uncertainties in the system model and external disturbances. While $H_\infty$ control techniques are commonly used to design robust controllers that mitigate the effects of disturbances, these methods often require complex and computationally intensive calculations. To address this issue, this paper proposes a reinforcement learning algorithm called Robust Deterministic Policy Gradient (RDPG), which formulates the $H_\infty$ control problem as a two-player zero-sum dynamic game. In this formulation, one player (the user) aims to minimize the cost, while the other player (the adversary) seeks to maximize it. We then employ deterministic policy gradient (DPG) and its deep reinforcement learning counterpart to train a robust control policy with effective disturbance attenuation. In particular, for practical implementation, we introduce an algorithm called robust deep deterministic policy gradient (RDDPG), which employs a deep neural network architecture and integrates techniques from the twin-delayed deep deterministic policy gradient (TD3) to enhance stability and learning efficiency. To evaluate the proposed algorithm, we implement it on an unmanned aerial vehicle (UAV) tasked with following a predefined path in a disturbance-prone environment. The experimental results demonstrate that the proposed method outperforms other control approaches in terms of robustness against disturbances, enabling precise real-time tracking of moving targets even under severe disturbance conditions.
>
---
#### [replaced 052] CLONE: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08931v2](http://arxiv.org/pdf/2506.08931v2)**

> **作者:** Yixuan Li; Yutang Lin; Jieming Cui; Tengyu Liu; Wei Liang; Yixin Zhu; Siyuan Huang
>
> **备注:** 18 pages, 13 figures
>
> **摘要:** Humanoid teleoperation plays a vital role in demonstrating and collecting data for complex humanoid-scene interactions. However, current teleoperation systems face critical limitations: they decouple upper- and lower-body control to maintain stability, restricting natural coordination, and operate open-loop without real-time position feedback, leading to accumulated drift. The fundamental challenge is achieving precise, coordinated whole-body teleoperation over extended durations while maintaining accurate global positioning. Here we show that an MoE-based teleoperation system, CLONE, with closed-loop error correction enables unprecedented whole-body teleoperation fidelity, maintaining minimal positional drift over long-range trajectories using only head and hand tracking from an MR headset. Unlike previous methods that either sacrifice coordination for stability or suffer from unbounded drift, CLONE learns diverse motion skills while preventing tracking error accumulation through real-time feedback, enabling complex coordinated movements such as ``picking up objects from the ground.'' These results establish a new milestone for whole-body humanoid teleoperation for long-horizon humanoid-scene interaction tasks.
>
---
#### [replaced 053] DivScene: Towards Open-Vocabulary Object Navigation with Large Vision Language Models in Diverse Scenes
- **分类: cs.CV; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.02730v3](http://arxiv.org/pdf/2410.02730v3)**

> **作者:** Zhaowei Wang; Hongming Zhang; Tianqing Fang; Ye Tian; Yue Yang; Kaixin Ma; Xiaoman Pan; Yangqiu Song; Dong Yu
>
> **备注:** EMNLP 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved significant progress in tasks like visual question answering and document understanding. However, their potential to comprehend embodied environments and navigate within them remains underexplored. In this work, we first study the challenge of open-vocabulary object navigation by introducing DivScene, a large-scale dataset with 4,614 houses across 81 scene types and 5,707 kinds of target objects. Our dataset provides a much greater diversity of target objects and scene types than existing datasets, enabling a comprehensive task evaluation. We evaluated various methods with LVLMs and LLMs on our dataset and found that current models still fall short of open-vocab object navigation ability. Then, we fine-tuned LVLMs to predict the next action with CoT explanations. We observe that LVLM's navigation ability can be improved substantially with only BFS-generated shortest paths without any human supervision, surpassing GPT-4o by over 20% in success rates.
>
---
#### [replaced 054] Efficient Manipulation-Enhanced Semantic Mapping With Uncertainty-Informed Action Selection
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02286v2](http://arxiv.org/pdf/2506.02286v2)**

> **作者:** Nils Dengler; Jesper Mücke; Rohit Menon; Maren Bennewitz
>
> **摘要:** Service robots operating in cluttered human environments such as homes, offices, and schools cannot rely on predefined object arrangements and must continuously update their semantic and spatial estimates while dealing with possible frequent rearrangements. Efficient and accurate mapping under such conditions demands selecting informative viewpoints and targeted manipulations to reduce occlusions and uncertainty. In this work, we present a manipulation-enhanced semantic mapping framework for occlusion-heavy shelf scenes that integrates evidential metric-semantic mapping with reinforcement-learning-based next-best view planning and targeted action selection. Our method thereby exploits uncertainty estimates from Dirichlet and Beta distributions in the map prediction networks to guide both active sensor placement and object manipulation, focusing on areas with high uncertainty and selecting actions with high expected information gain. Furthermore, we introduce an uncertainty-informed push strategy that targets occlusion-critical objects and generates minimally invasive actions to reveal hidden regions by reducing overall uncertainty in the scene. The experimental evaluation shows that our framework enables to accurately map cluttered scenes, while substantially reducing object displacement and achieving a 95% reduction in planning time compared to the state-of-the-art, thereby realizing real-world applicability.
>
---
#### [replaced 055] Self-Supervised Learning-Based Path Planning and Obstacle Avoidance Using PPO and B-Splines in Unknown Environments
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.02176v2](http://arxiv.org/pdf/2412.02176v2)**

> **作者:** Shahab Shokouhi; Oguzhan Oruc; May-Win Thein
>
> **摘要:** This paper introduces SmartBSP, an advanced self-supervised learning framework for real-time path planning and obstacle avoidance in autonomous robotics navigating through complex environments. The proposed system integrates Proximal Policy Optimization (PPO) with Convolutional Neural Networks (CNN) and Actor-Critic architecture to process limited LIDAR inputs and compute spatial decision-making probabilities. The robot's perceptual field is discretized into a grid format, which the CNN analyzes to produce a spatial probability distribution. During the training process a nuanced cost function is minimized that accounts for path curvature, endpoint proximity, and obstacle avoidance. Simulations results in different scenarios validate the algorithm's resilience and adaptability across diverse operational scenarios. Subsequently, Real-time experiments, employing the Robot Operating System (ROS), were carried out to assess the efficacy of the proposed algorithm.
>
---
#### [replaced 056] LocoTouch: Learning Dynamic Quadrupedal Transport with Tactile Sensing
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.23175v2](http://arxiv.org/pdf/2505.23175v2)**

> **作者:** Changyi Lin; Yuxin Ray Song; Boda Huo; Mingyang Yu; Yikai Wang; Shiqi Liu; Yuxiang Yang; Wenhao Yu; Tingnan Zhang; Jie Tan; Yiyue Luo; Ding Zhao
>
> **备注:** Project page: https://linchangyi1.github.io/LocoTouch
>
> **摘要:** Quadrupedal robots have demonstrated remarkable agility and robustness in traversing complex terrains. However, they struggle with dynamic object interactions, where contact must be precisely sensed and controlled. To bridge this gap, we present LocoTouch, a system that equips quadrupedal robots with tactile sensing to address a particularly challenging task in this category: long-distance transport of unsecured cylindrical objects, which typically requires custom mounting or fastening mechanisms to maintain stability. For efficient large-area tactile sensing, we design a high-density distributed tactile sensor that covers the entire back of the robot. To effectively leverage tactile feedback for robot control, we develop a simulation environment with high-fidelity tactile signals, and train tactile-aware transport policies using a two-stage learning pipeline. Furthermore, we design a novel reward function to promote robust, symmetric, and frequency-adaptive locomotion gaits. After training in simulation, LocoTouch transfers zero-shot to the real world, reliably transporting a wide range of unsecured cylindrical objects with diverse sizes, weights, and surface properties. Moreover, it remains robust over long distances, on uneven terrain, and under severe perturbations.
>
---
#### [replaced 057] Multi-Touch and Bending Perception Using Electrical Impedance Tomography for Robotics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.13048v3](http://arxiv.org/pdf/2503.13048v3)**

> **作者:** Haofeng Chen; Bedrich Himmel; Bin Li; Xiaojie Wang; Matej Hoffmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Electrical Impedance Tomography (EIT) offers a promising solution for distributed tactile sensing with minimal wiring and full-surface coverage in robotic applications. However, EIT-based tactile sensors face significant challenges during surface bending. Deformation alters the baseline impedance distribution and couples with touch-induced conductivity variations, complicating signal interpretation. To address this challenge, we present a novel sensing framework that integrates a deep neural network for interaction state classification with a dynamic adaptive reference strategy to decouple touch and deformation signals, while a data-driven regression model translates EIT voltage changes into continuous bending angles. The framework is validated using a magnetic hydrogel composite sensor that conforms to bendable surfaces. Experimental evaluations demonstrate that the proposed framework achieves precise and robust bending angle estimation, high accuracy in distinguishing touch, bending, and idle states, and significantly improves touch localization quality under bending deformation compared to conventional fixed-reference methods. Real-time experiments confirm the system's capability to reliably detect multi-touch interactions and track bending angles across varying deformation conditions. This work paves the way for flexible EIT-based robotic skins capable of rich multimodal sensing in robotics and human-robot interaction.
>
---
#### [replaced 058] Force Myography based Torque Estimation in Human Knee and Ankle Joints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11061v2](http://arxiv.org/pdf/2409.11061v2)**

> **作者:** Charlotte Marquardt; Arne Schulz; Miha Dezman; Gunther Kurz; Thorsten Stein; Tamim Asfour
>
> **备注:** This file corresponds to the manuscript presented at the IEEE International Conference on Robotics and Automation (ICRA), May 2025
>
> **摘要:** The online adaptation of exoskeleton control based on muscle activity sensing offers a promising approach to personalizing exoskeleton behavior based on the user's biosignals. While electromyography (EMG)-based methods have demonstrated improvements in joint torque estimation, EMG sensors require direct skin contact and extensive post-processing. In contrast, force myography (FMG) measures normal forces resulting from changes in muscle volume due to muscle activity. We propose an FMG-based method to estimate knee and ankle joint torques by integrating joint angles and velocities with muscle activity data. We learn a model for joint torque estimation using Gaussian process regression (GPR). The effectiveness of the proposed FMG-based method is validated on isokinetic motions performed by ten participants. The model is compared to a baseline model that uses only joint angle and velocity, as well as a model augmented by EMG data. The results indicate that incorporating FMG into exoskeleton control can improve the estimation of joint torque for the ankle and knee joints in novel task characteristics within a single participant. Although the findings suggest that this approach may not improve the generalizability of estimates between multiple participants, they highlight the need for further research into its potential applications in exoskeleton control.
>
---
