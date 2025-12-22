# 机器人 cs.RO

- **最新发布 31 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] Semantic Co-Speech Gesture Synthesis and Real-Time Control for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属人机交互任务，旨在解决机器人缺乏自然、语义一致的实时手势表达问题。提出端到端框架：用LLM+Motion-GPT生成语义匹配的共语手势，经GMR动作重定向后，由MotionTracker控制Unitree G1机器人实时执行并保持平衡。**

- **链接: [https://arxiv.org/pdf/2512.17183v1](https://arxiv.org/pdf/2512.17183v1)**

> **作者:** Gang Zhang
>
> **摘要:** We present an innovative end-to-end framework for synthesizing semantically meaningful co-speech gestures and deploying them in real-time on a humanoid robot. This system addresses the challenge of creating natural, expressive non-verbal communication for robots by integrating advanced gesture generation techniques with robust physical control. Our core innovation lies in the meticulous integration of a semantics-aware gesture synthesis module, which derives expressive reference motions from speech input by leveraging a generative retrieval mechanism based on large language models (LLMs) and an autoregressive Motion-GPT model. This is coupled with a high-fidelity imitation learning control policy, the MotionTracker, which enables the Unitree G1 humanoid robot to execute these complex motions dynamically and maintain balance. To ensure feasibility, we employ a robust General Motion Retargeting (GMR) method to bridge the embodiment gap between human motion data and the robot platform. Through comprehensive evaluation, we demonstrate that our combined system produces semantically appropriate and rhythmically coherent gestures that are accurately tracked and executed by the physical robot. To our knowledge, this work represents a significant step toward general real-world use by providing a complete pipeline for automatic, semantic-aware, co-speech gesture generation and synchronized real-time physical deployment on a humanoid robot.
>
---
#### [new 002] Learning-Based Safety-Aware Task Scheduling for Efficient Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文属人机协作任务调度研究，旨在解决安全约束导致机器人效率下降的问题。提出基于深度学习的安全感知调度方法：不预测人体运动，而是学习系统状态与安全降速的关系，实时优化任务选择以最小化周期时间并满足安全要求。**

- **链接: [https://arxiv.org/pdf/2512.17560v1](https://arxiv.org/pdf/2512.17560v1)**

> **作者:** M. Faroni; A. Spano; A. M. Zanchettin; P. Rocco
>
> **备注:** 8 pages
>
> **摘要:** Ensuring human safety in collaborative robotics can compromise efficiency because traditional safety measures increase robot cycle time when human interaction is frequent. This paper proposes a safety-aware approach to mitigate efficiency losses without assuming prior knowledge of safety logic. Using a deep-learning model, the robot learns the relationship between system state and safety-induced speed reductions based on execution data. Our framework does not explicitly predict human motions but directly models the interaction effects on robot speed, simplifying implementation and enhancing generalizability to different safety logics. At runtime, the learned model optimizes task selection to minimize cycle time while adhering to safety requirements. Experiments on a pick-and-packaging scenario demonstrated significant reductions in cycle times.
>
---
#### [new 003] Lang2Manip: A Tool for LLM-Based Symbolic-to-Geometric Planning for Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Lang2Manip，解决LLM生成的符号计划难以在仿真中通用执行的问题。它构建统一管道，将自然语言指令转为符号动作，并通过Kautham框架实现机器人无关、多规划器兼容的几何轨迹生成与执行，支持语言驱动的任务与运动规划（TAMP）。**

- **链接: [https://arxiv.org/pdf/2512.17062v1](https://arxiv.org/pdf/2512.17062v1)**

> **作者:** Muhayy Ud Din; Jan Rosell; Waseem Akram; Irfan Hussain
>
> **备注:** Submitted to ICARA
>
> **摘要:** Simulation is essential for developing robotic manipulation systems, particularly for task and motion planning (TAMP), where symbolic reasoning interfaces with geometric, kinematic, and physics-based execution. Recent advances in Large Language Models (LLMs) enable robots to generate symbolic plans from natural language, yet executing these plans in simulation often requires robot-specific engineering or planner-dependent integration. In this work, we present a unified pipeline that connects an LLM-based symbolic planner with the Kautham motion planning framework to achieve generalizable, robot-agnostic symbolic-to-geometric manipulation. Kautham provides ROS-compatible support for a wide range of industrial manipulators and offers geometric, kinodynamic, physics-driven, and constraint-based motion planning under a single interface. Our system converts language instructions into symbolic actions and computes and executes collision-free trajectories using any of Kautham's planners without additional coding. The result is a flexible and scalable tool for language-driven TAMP that is generalized across robots, planning modalities, and manipulation tasks.
>
---
#### [new 004] Research on Dead Reckoning Algorithm for Self-Propelled Pipeline Robots in Three-Dimensional Complex Pipelines
- **分类: cs.RO; cs.AI**

- **简介: 该论文属机器人定位任务，旨在解决复杂三维管道中传统定位方法易受环境干扰、易失效的问题。提出基于EKF融合IMU与轮式里程计的自主推算（Dead Reckoning）算法，兼顾运动灵活性与定位精度，并在矩形环形管道中验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.17215v1](https://arxiv.org/pdf/2512.17215v1)**

> **作者:** Yan Gao; Jiliang Wang; Minghan Wang; Xiaohua Chen; Demin Chen; Zhiyong Ren; Tian-Yun Huang
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** In the field of gas pipeline location, existing pipeline location methods mostly rely on pipeline location instruments. However, when faced with complex and curved pipeline scenarios, these methods often fail due to problems such as cable entanglement and insufficient equipment flexibility. To address this pain point, we designed a self-propelled pipeline robot. This robot can autonomously complete the location work of complex and curved pipelines in complex pipe networks without external dragging. In terms of pipeline mapping technology, traditional visual mapping and laser mapping methods are easily affected by lighting conditions and insufficient features in the confined space of pipelines, resulting in mapping drift and divergence problems. In contrast, the pipeline location method that integrates inertial navigation and wheel odometers is less affected by pipeline environmental factors. Based on this, this paper proposes a pipeline robot location method based on extended Kalman filtering (EKF). Firstly, the body attitude angle is initially obtained through an inertial measurement unit (IMU). Then, the extended Kalman filtering algorithm is used to improve the accuracy of attitude angle estimation. Finally, high-precision pipeline location is achieved by combining wheel odometers. During the testing phase, the roll wheels of the pipeline robot needed to fit tightly against the pipe wall to reduce slippage. However, excessive tightness would reduce the flexibility of motion control due to excessive friction. Therefore, a balance needed to be struck between the robot's motion capability and positioning accuracy. Experiments were conducted using the self-propelled pipeline robot in a rectangular loop pipeline, and the results verified the effectiveness of the proposed dead reckoning algorithm.
>
---
#### [new 005] Flying in Clutter on Monocular RGB by Learning in 3D Radiance Fields with Domain Adaptation
- **分类: cs.RO**

- **简介: 该论文属自主导航任务，旨在解决单目RGB图像下飞行机器人在杂乱环境中的真实世界导航难题。提出融合3D高斯溅射仿真与对抗域自适应的框架，在仿真中训练策略并缩小感知域差距，实现零样本迁移至真实场景。**

- **链接: [https://arxiv.org/pdf/2512.17349v1](https://arxiv.org/pdf/2512.17349v1)**

> **作者:** Xijie Huang; Jinhan Li; Tianyue Wu; Xin Zhou; Zhichao Han; Fei Gao
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Modern autonomous navigation systems predominantly rely on lidar and depth cameras. However, a fundamental question remains: Can flying robots navigate in clutter using solely monocular RGB images? Given the prohibitive costs of real-world data collection, learning policies in simulation offers a promising path. Yet, deploying such policies directly in the physical world is hindered by the significant sim-to-real perception gap. Thus, we propose a framework that couples the photorealism of 3D Gaussian Splatting (3DGS) environments with Adversarial Domain Adaptation. By training in high-fidelity simulation while explicitly minimizing feature discrepancy, our method ensures the policy relies on domain-invariant cues. Experimental results demonstrate that our policy achieves robust zero-shot transfer to the physical world, enabling safe and agile flight in unstructured environments with varying illumination.
>
---
#### [new 006] Deep Learning-based Robust Autonomous Navigation of Aerial Robots in Dense Forests
- **分类: cs.RO**

- **简介: 该论文属自主导航任务，旨在解决无人机在密林中GNSS拒止、感知退化、薄障碍物多等挑战。提出改进的深度学习框架，融合语义增强深度编码与运动原语评估，并加入横向控制、时序一致性、视觉惯性里程计及安全监督层，提升鲁棒性与实时性。**

- **链接: [https://arxiv.org/pdf/2512.17553v1](https://arxiv.org/pdf/2512.17553v1)**

> **作者:** Guglielmo Del Col; Väinö Karjalainen; Teemu Hakala; Yibo Zhang; Eija Honkavaara
>
> **摘要:** Autonomous aerial navigation in dense natural environments remains challenging due to limited visibility, thin and irregular obstacles, GNSS-denied operation, and frequent perceptual degradation. This work presents an improved deep learning-based navigation framework that integrates semantically enhanced depth encoding with neural motion-primitive evaluation for robust flight in cluttered forests. Several modules are incorporated on top of the original sevae-ORACLE algorithm to address limitations observed during real-world deployment, including lateral control for sharper maneuvering, a temporal consistency mechanism to suppress oscillatory planning decisions, a stereo-based visual-inertial odometry solution for drift-resilient state estimation, and a supervisory safety layer that filters unsafe actions in real time. A depth refinement stage is included to improve the representation of thin branches and reduce stereo noise, while GPU optimization increases onboard inference throughput from 4 Hz to 10 Hz. The proposed approach is evaluated against several existing learning-based navigation methods under identical environmental conditions and hardware constraints. It demonstrates higher success rates, more stable trajectories, and improved collision avoidance, particularly in highly cluttered forest settings. The system is deployed on a custom quadrotor in three boreal forest environments, achieving fully autonomous completion in all flights in moderate and dense clutter, and 12 out of 15 flights in highly dense underbrush. These results demonstrate improved reliability and safety over existing navigation methods in complex natural environments.
>
---
#### [new 007] Adaptive Covariance and Quaternion-Focused Hybrid Error-State EKF/UKF for Visual-Inertial Odometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向无人机视觉-惯性里程计（VIO）任务，解决复杂环境下传感器可靠性动态变化导致的位姿估计精度与效率难以兼顾的问题。提出自适应协方差与四元数聚焦的混合误差状态EKF/UKF算法，融合图像熵、运动模糊等指标动态调整观测噪声，并分步优化姿态与整体状态。**

- **链接: [https://arxiv.org/pdf/2512.17505v1](https://arxiv.org/pdf/2512.17505v1)**

> **作者:** Ufuk Asil; Efendi Nasibov
>
> **摘要:** This study presents an innovative hybrid Visual-Inertial Odometry (VIO) method for Unmanned Aerial Vehicles (UAVs) that is resilient to environmental challenges and capable of dynamically assessing sensor reliability. Built upon a loosely coupled sensor fusion architecture, the system utilizes a novel hybrid Quaternion-focused Error-State EKF/UKF (Qf-ES-EKF/UKF) architecture to process inertial measurement unit (IMU) data. This architecture first propagates the entire state using an Error-State Extended Kalman Filter (ESKF) and then applies a targeted Scaled Unscented Kalman Filter (SUKF) step to refine only the orientation. This sequential process blends the accuracy of SUKF in quaternion estimation with the overall computational efficiency of ESKF. The reliability of visual measurements is assessed via a dynamic sensor confidence score based on metrics, such as image entropy, intensity variation, motion blur, and inference quality, adapting the measurement noise covariance to ensure stable pose estimation even under challenging conditions. Comprehensive experimental analyses on the EuRoC MAV dataset demonstrate key advantages: an average improvement of 49% in position accuracy in challenging scenarios, an average of 57% in rotation accuracy over ESKF-based methods, and SUKF-comparable accuracy achieved with approximately 48% lower computational cost than a full SUKF implementation. These findings demonstrate that the presented approach strikes an effective balance between computational efficiency and estimation accuracy, and significantly enhances UAV pose estimation performance in complex environments with varying sensor reliability.
>
---
#### [new 008] A Dual Quaternion based RRT* Path Planning Approach for Satellite Rendezvous and Docking
- **分类: cs.RO**

- **简介: 该论文属卫星交会对接路径规划任务，旨在解决六自由度姿态轨迹生成中姿态不连续与避障难问题。提出基于对偶四元数的RRT*算法，将SE(3)空间中的螺旋运动自然融入采样规划，提升姿态连续性与障碍物规避能力，为后续动力学优化提供初始轨迹。**

- **链接: [https://arxiv.org/pdf/2512.17680v1](https://arxiv.org/pdf/2512.17680v1)**

> **作者:** Ana Stankovic; Mohamed Khalil Ben-Larbi; Wolfgang H. Müller
>
> **备注:** 6 pages, CAMSAT 2025, This work has been accepted to IFAC
>
> **摘要:** This paper proposes a sampling-based motion planner that employs a dual quaternion representation to generate smooth, collision-free six-degree-of-freedom pose trajectories for satellite rendezvous and docking under keep-out zone constraints. The proposed planner integrates the dual quaternion algebra directly into an RRT* framework, thereby enabling natural screw motion interpolation in SE(3). The dual quaternion-based RRT* has been implemented in Python and demonstrated on a representative multi-obstacle scenario. A comparison with a standard RRT* using separate translation and quaternion steering highlights the enhanced pose continuity and obstacle avoidance of the proposed method. The present approach is purely kinematic in nature and does not take into account relative orbital dynamics. Consequently, the resulting path provides a preliminary estimate for a subsequent optimisation-based trajectory planner, which will refine the motion with dynamic constraints for the purpose of practical satellite rendezvous and docking missions.
>
---
#### [new 009] Planning as Descent: Goal-Conditioned Latent Trajectory Synthesis in Learned Energy Landscapes
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出“规划即下降”（PaD）框架，面向离线目标条件强化学习任务，解决无奖励、小样本下规划鲁棒性差的问题。它学习目标条件的能量函数，将规划建模为潜在轨迹上的梯度下降优化，并通过自监督后见目标重标定训练，在噪声数据上实现95%高成功率。**

- **链接: [https://arxiv.org/pdf/2512.17846v1](https://arxiv.org/pdf/2512.17846v1)**

> **作者:** Carlos Vélez García; Miguel Cazorla; Jorge Pomares
>
> **摘要:** We present Planning as Descent (PaD), a framework for offline goal-conditioned reinforcement learning that grounds trajectory synthesis in verification. Instead of learning a policy or explicit planner, PaD learns a goal-conditioned energy function over entire latent trajectories, assigning low energy to feasible, goal-consistent futures. Planning is realized as gradient-based refinement in this energy landscape, using identical computation during training and inference to reduce train-test mismatch common in decoupled modeling pipelines. PaD is trained via self-supervised hindsight goal relabeling, shaping the energy landscape around the planning dynamics. At inference, multiple trajectory candidates are refined under different temporal hypotheses, and low-energy plans balancing feasibility and efficiency are selected. We evaluate PaD on OGBench cube manipulation tasks. When trained on narrow expert demonstrations, PaD achieves state-of-the-art 95\% success, strongly outperforming prior methods that peak at 68\%. Remarkably, training on noisy, suboptimal data further improves success and plan efficiency, highlighting the benefits of verification-driven planning. Our results suggest learning to evaluate and refine trajectories provides a robust alternative to direct policy learning for offline, reward-free planning.
>
---
#### [new 010] Kinematics-Aware Diffusion Policy with Consistent 3D Observation and Action Space for Whole-Arm Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文面向全臂机器人操作任务，解决传统方法因关节空间与3D任务空间不一致导致的泛化难、样本效率低问题。提出基于扩散策略的运动学感知模仿学习框架，统一用3D点表征观测、状态与动作，并嵌入运动学先验，保障动作可行性，再通过优化求解逆运动学执行。**

- **链接: [https://arxiv.org/pdf/2512.17568v1](https://arxiv.org/pdf/2512.17568v1)**

> **作者:** Kangchen Lv; Mingrui Yu; Yongyi Jia; Chenyu Zhang; Xiang Li
>
> **备注:** The first two authors contributed equally. Project Website: https://kinematics-aware-diffusion-policy.github.io
>
> **摘要:** Whole-body control of robotic manipulators with awareness of full-arm kinematics is crucial for many manipulation scenarios involving body collision avoidance or body-object interactions, which makes it insufficient to consider only the end-effector poses in policy learning. The typical approach for whole-arm manipulation is to learn actions in the robot's joint space. However, the unalignment between the joint space and actual task space (i.e., 3D space) increases the complexity of policy learning, as generalization in task space requires the policy to intrinsically understand the non-linear arm kinematics, which is difficult to learn from limited demonstrations. To address this issue, this letter proposes a kinematics-aware imitation learning framework with consistent task, observation, and action spaces, all represented in the same 3D space. Specifically, we represent both robot states and actions using a set of 3D points on the arm body, naturally aligned with the 3D point cloud observations. This spatially consistent representation improves the policy's sample efficiency and spatial generalizability while enabling full-body control. Built upon the diffusion policy, we further incorporate kinematics priors into the diffusion processes to guarantee the kinematic feasibility of output actions. The joint angle commands are finally calculated through an optimization-based whole-body inverse kinematics solver for execution. Simulation and real-world experimental results demonstrate higher success rates and stronger spatial generalizability of our approach compared to existing methods in body-aware manipulation policy learning.
>
---
#### [new 011] On Using Neural Networks to Learn Safety Speed Reduction in Human-Robot Collaboration: A Comparative Analysis
- **分类: cs.RO**

- **简介: 该论文属预测任务，旨在解决人机协作中因安全机制导致的机器人减速难以准确建模、影响周期时间估计与调度效率的问题。作者提出用深度学习（尤以简单前馈网络为主）直接从执行数据中学习并预测安全缩放因子。**

- **链接: [https://arxiv.org/pdf/2512.17579v1](https://arxiv.org/pdf/2512.17579v1)**

> **作者:** Marco Faroni; Alessio Spanò; Andrea M. Zanchettin; Paolo Rocco
>
> **备注:** Accepted at IEEE Internation Conference on Emerging Technologies and Factory Automation (ETFA) 2025
>
> **摘要:** In Human-Robot Collaboration, safety mechanisms such as Speed and Separation Monitoring and Power and Force Limitation dynamically adjust the robot's speed based on human proximity. While essential for risk reduction, these mechanisms introduce slowdowns that makes cycle time estimation a hard task and impact job scheduling efficiency. Existing methods for estimating cycle times or designing schedulers often rely on predefined safety models, which may not accurately reflect real-world safety implementations, as these depend on case-specific risk assessments. In this paper, we propose a deep learning approach to predict the robot's safety scaling factor directly from process execution data. We analyze multiple neural network architectures and demonstrate that a simple feed-forward network effectively estimates the robot's slowdown. This capability is crucial for improving cycle time predictions and designing more effective scheduling algorithms in collaborative robotic environments.
>
---
#### [new 012] Conservative Bias in Multi-Teacher Learning: Why Agents Prefer Low-Reward Advisors
- **分类: cs.RO; cs.AI**

- **简介: 该论文属交互式强化学习任务，探究多教师环境下智能体的教师选择偏好问题。发现智能体存在保守偏差，倾向选择低奖励但高一致性教师（93.16%选择率），并识别出教师可用性与准确率的关键阈值（≥0.6），在概念漂移下性能提升159%。**

- **链接: [https://arxiv.org/pdf/2512.17180v1](https://arxiv.org/pdf/2512.17180v1)**

> **作者:** Maher Mesto; Francisco Cruz
>
> **备注:** 10 pages, 5 figures. Accepted at ACRA 2025 (Australasian Conference on Robotics and Automation)
>
> **摘要:** Interactive reinforcement learning (IRL) has shown promise in enabling autonomous agents and robots to learn complex behaviours from human teachers, yet the dynamics of teacher selection remain poorly understood. This paper reveals an unexpected phenomenon in IRL: when given a choice between teachers with different reward structures, learning agents overwhelmingly prefer conservative, low-reward teachers (93.16% selection rate) over those offering 20x higher rewards. Through 1,250 experimental runs in navigation tasks with multiple expert teachers, we discovered: (1) Conservative bias dominates teacher selection: agents systematically choose the lowest-reward teacher, prioritising consistency over optimality; (2) Critical performance thresholds exist at teacher availability rho >= 0.6 and accuracy omega >= 0.6, below which the framework fails catastrophically; (3) The framework achieves 159% improvement over baseline Q-learning under concept drift. These findings challenge fundamental assumptions about optimal teaching in RL and suggest potential implications for human-robot collaboration, where human preferences for safety and consistency may align with the observed agent selection behaviour, potentially informing training paradigms for safety-critical robotic applications.
>
---
#### [new 013] Design and Research of a Self-Propelled Pipeline Robot Based on Force Analysis and Dynamic Simulation
- **分类: cs.RO**

- **简介: 该论文属于管道机器人设计任务，旨在解决传统缆控机器人行程受限、垂直爬升失败及T型支管通过性差等问题。工作包括：基于受力分析与ADAMS动态仿真优化轮式模块化结构，完成SolidWorks建模、仿真优化与 acrylic 管道实验验证。**

- **链接: [https://arxiv.org/pdf/2512.17212v1](https://arxiv.org/pdf/2512.17212v1)**

> **作者:** Yan Gao; Jiliang Wang; Ming Cheng; Tianyun Huang
>
> **备注:** 7 pages, 14 figures
>
> **摘要:** In pipeline inspection, traditional tethered inspection robots are severely constrained by cable length and weight, which greatly limit their travel range and accessibility. To address these issues, this paper proposes a self-propelled pipeline robot design based on force analysis and dynamic simulation, with a specific focus on solving core challenges including vertical climbing failure and poor passability in T-branch pipes. Adopting a wheeled configuration and modular design, the robot prioritizes the core demand of body motion control. Specifically, 3D modeling of the robot was first completed using SolidWorks. Subsequently, the model was imported into ADAMS for dynamic simulation, which provided a basis for optimizing the drive module and motion control strategy.To verify the robot's dynamic performance, an experimental platform with acrylic pipes was constructed. Through adjusting its body posture to surmount obstacles and select directions, the robot has demonstrated its ability to stably traverse various complex pipeline scenarios. Notably, this work offers a technical feasibility reference for the application of pipeline robots in the inspection of medium and low-pressure urban gas pipelines.
>
---
#### [new 014] Vidarc: Embodied Video Diffusion Model for Closed-loop Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出Vidarc，一种面向闭环控制的具身视频扩散模型，解决数据稀缺下机械臂操作中动态建模难、延迟高、接地弱的问题。通过掩码逆动力学增强与缓存式自回归生成，实现快速精准闭环控制，并在跨平台任务中展现强泛化与纠错能力。**

- **链接: [https://arxiv.org/pdf/2512.17661v1](https://arxiv.org/pdf/2512.17661v1)**

> **作者:** Yao Feng; Chendong Xiang; Xinyi Mao; Hengkai Tan; Zuyue Zhang; Shuhe Huang; Kaiwen Zheng; Haitian Liu; Hang Su; Jun Zhu
>
> **摘要:** Robotic arm manipulation in data-scarce settings is a highly challenging task due to the complex embodiment dynamics and diverse contexts. Recent video-based approaches have shown great promise in capturing and transferring the temporal and physical interactions by pre-training on Internet-scale video data. However, such methods are often not optimized for the embodiment-specific closed-loop control, typically suffering from high latency and insufficient grounding. In this paper, we present Vidarc (Video Diffusion for Action Reasoning and Closed-loop Control), a novel autoregressive embodied video diffusion approach augmented by a masked inverse dynamics model. By grounding video predictions with action-relevant masks and incorporating real-time feedback through cached autoregressive generation, Vidarc achieves fast, accurate closed-loop control. Pre-trained on one million cross-embodiment episodes, Vidarc surpasses state-of-the-art baselines, achieving at least a 15% higher success rate in real-world deployment and a 91% reduction in latency. We also highlight its robust generalization and error correction capabilities across previously unseen robotic platforms.
>
---
#### [new 015] Mr.MSTE: Multi-robot Multi-Source Term Estimation with Wind-Aware Coverage Control
- **分类: cs.RO**

- **简介: 该论文面向多机器人气体源定位任务，解决未知数量、动态变化的空中气体释放源参数估计问题。提出MRMSTE框架：融合物理模型的混合贝叶斯推理用于多源联合估计，并设计风感知覆盖控制（WCC）策略，利用风向信息优化机器人部署，提升检测效率与源分离精度。**

- **链接: [https://arxiv.org/pdf/2512.17001v1](https://arxiv.org/pdf/2512.17001v1)**

> **作者:** Rohit V. Nanavati; Tim J. Glover; Matthew J. Coombes; Cunjia Liu
>
> **摘要:** This paper presents a Multi-Robot Multi-Source Term Estimation (MRMSTE) framework that enables teams of mobile robots to collaboratively sample gas concentrations and infer the parameters of an unknown number of airborne releases. The framework is built on a hybrid Bayesian inference scheme that represents the joint multi-source probability density and incorporates physics-informed state transitions, including source birth, removal, and merging induced by atmospheric dispersion. A superposition-based measurement model is naturally accommodated, allowing sparse concentration measurements to be exploited efficiently. To guide robot deployment, we introduce a wind-aware coverage control (WCC) strategy that integrates the evolving multi-source belief with local wind information to prioritize regions of high detection likelihood. Unlike conventional coverage control or information-theoretic planners, WCC explicitly accounts for anisotropic plume transport when modelling sensor performance, leading to more effective sensor placement for multi-source estimation. Monte Carlo studies demonstrate faster convergence and improved separation of individual source beliefs compared to traditional coverage-based strategies and small-scale static sensor networks. Real-world experiments with CO2 releases using TurtleBot platforms further validate the proposed approach, demonstrating its practicality for scalable multi-robot gas-sensing applications.
>
---
#### [new 016] Personalized Gait Patterns During Exoskeleton-Aided Training May Have Minimal Effect on User Experience. Insights from a Pilot Study
- **分类: cs.RO**

- **简介: 该论文属康复机器人领域，旨在解决外骨骼步态训练中预设非个性化轨迹导致自然性与舒适性差的问题。作者提出数据驱动的多平面步态个性化框架，基于人体测量等数据生成个体化轨迹，并在10人实验中对比主观体验，发现个性化未显著提升感受，而用户适应性影响更大。**

- **链接: [https://arxiv.org/pdf/2512.17425v1](https://arxiv.org/pdf/2512.17425v1)**

> **作者:** Beatrice Luciani; Katherine Lin Poggensee; Heike Vallery; Alex van den Berg; Severin David Woernle; Mostafa Mogharabi; Stefano Dalla Gasperina; Laura Marchal-Crespo
>
> **摘要:** Robot-aided gait rehabilitation facilitates high-intensity and repeatable therapy. However, most exoskeletons rely on pre-recorded, non-personalized gait trajectories constrained to the sagittal plane, potentially limiting movement naturalness and user comfort. We present a data-driven gait personalization framework for an exoskeleton that supports multi-planar motion, including hip abduction/adduction and pelvic translation and rotation. Personalized trajectories to individual participants were generated using regression models trained on anthropometric, demographic, and walking speed data from a normative database. In a within-subject experiment involving ten unimpaired participants, these personalized trajectories were evaluated in regard to comfort, naturalness, and overall experience and compared against two standard patterns from the same database: one averaging all the trajectories, and one randomly selected. We did not find relevant differences across pattern conditions, despite all trajectories being executed with high accuracy thanks to a stiff position-derivative controller. We found, however, that pattern conditions in later trials were rated as more comfortable and natural than those in the first trial, suggesting that participants might have adapted to walking within the exoskeleton, regardless of the enforced gait pattern. Our findings highlight the importance of integrating subjective feedback when designing personalized gait controllers and accounting for user adaptation during experimentation.
>
---
#### [new 017] Neuro-Symbolic Control with Large Language Models for Language-Guided Spatial Tasks
- **分类: cs.RO**

- **简介: 该论文面向语言引导的空间操作任务，解决LLM直接用于连续控制时的不稳、低效与幻觉问题。提出轻量神经-符号框架：本地LLM做符号级语义推理，神经delta控制器执行连续动作。实验表明其显著提升成功率、效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.17321v1](https://arxiv.org/pdf/2512.17321v1)**

> **作者:** Momina Liaqat Ali; Muhammad Abid
>
> **摘要:** Although large language models (LLMs) have recently become effective tools for language-conditioned control in embodied systems, instability, slow convergence, and hallucinated actions continue to limit their direct application to continuous control. A modular neuro-symbolic control framework that clearly distinguishes between low-level motion execution and high-level semantic reasoning is proposed in this work. While a lightweight neural delta controller performs bounded, incremental actions in continuous space, a locally deployed LLM interprets symbolic tasks. We assess the suggested method in a planar manipulation setting with spatial relations between objects specified by language. Numerous tasks and local language models, such as Mistral, Phi, and LLaMA-3.2, are used in extensive experiments to compare LLM-only control, neural-only control, and the suggested LLM+DL framework. In comparison to LLM-only baselines, the results show that the neuro-symbolic integration consistently increases both success rate and efficiency, achieving average step reductions exceeding 70% and speedups of up to 8.83x while remaining robust to language model quality. The suggested framework enhances interpretability, stability, and generalization without any need of reinforcement learning or costly rollouts by controlling the LLM to symbolic outputs and allocating uninterpreted execution to a neural controller trained on artificial geometric data. These outputs show empirically that neuro-symbolic decomposition offers a scalable and principled way to integrate language understanding with ongoing control, this approach promotes the creation of dependable and effective language-guided embodied systems.
>
---
#### [new 018] A Service Robot's Guide to Interacting with Busy Customers
- **分类: cs.RO; cs.HC**

- **简介: 该论文属人机交互任务，旨在解决服务机器人在用户忙碌时如何有效沟通的问题。通过模拟餐厅场景实验，对比声、视、微动三种模态及多模态组合在注意力捕获与意图传达上的效果，发现语音擅抓注意，视觉最利意图理解。**

- **链接: [https://arxiv.org/pdf/2512.17241v1](https://arxiv.org/pdf/2512.17241v1)**

> **作者:** Suraj Nukala; Meera Sushma; Leimin Tian; Akansel Cosgun; Dana Kulic
>
> **备注:** Presented at ACRA 2025. 10 pages, 4 figures. Includes a user study (N=24) using the Temi robot evaluating speech, visual, and micromotion modalities
>
> **摘要:** The growing use of service robots in hospitality highlights the need to understand how to effectively communicate with pre-occupied customers. This study investigates the efficacy of commonly used communication modalities by service robots, namely, acoustic/speech, visual display, and micromotion gestures in capturing attention and communicating intention with a user in a simulated restaurant scenario. We conducted a two-part user study (N=24) using a Temi robot to simulate delivery tasks, with participants engaged in a typing game (MonkeyType) to emulate a state of busyness. The participants' engagement in the typing game is measured by words per minute (WPM) and typing accuracy. In Part 1, we compared non-verbal acoustic cue versus baseline conditions to assess attention capture during a single-cup delivery task. In Part 2, we evaluated the effectiveness of speech, visual display, micromotion and their multimodal combination in conveying specific intentions (correct cup selection) during a two-cup delivery task. The results indicate that, while speech is highly effective in capturing attention, it is less successful in clearly communicating intention. Participants rated visual as the most effective modality for intention clarity, followed by speech, with micromotion being the lowest ranked.These findings provide insights into optimizing communication strategies for service robots, highlighting the distinct roles of attention capture and intention communication in enhancing user experience in dynamic hospitality settings.
>
---
#### [new 019] AnyTask: an Automated Task and Data Generation Framework for Advancing Sim-to-Real Policy Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AnyTask框架，解决仿真到现实（sim-to-real）策略学习中任务设计与数据生成依赖人工、成本高的问题。它利用GPU并行仿真与多模态基础模型，自动化生成多样化操作任务及专家演示，并通过行为克隆训练策略，在真实机器人上实现44%平均成功率。**

- **链接: [https://arxiv.org/pdf/2512.17853v1](https://arxiv.org/pdf/2512.17853v1)**

> **作者:** Ran Gong; Xiaohan Zhang; Jinghuan Shang; Maria Vittoria Minniti; Jigarkumar Patel; Valerio Pepe; Riedana Yan; Ahmet Gundogdu; Ivan Kapelyukh; Ali Abbas; Xiaoqiang Yan; Harsh Patel; Laura Herlant; Karl Schmeckpeper
>
> **备注:** 28 pages, 25 figures. The first four authors contributed equally
>
> **摘要:** Generalist robot learning remains constrained by data: large-scale, diverse, and high-quality interaction data are expensive to collect in the real world. While simulation has become a promising way for scaling up data collection, the related tasks, including simulation task design, task-aware scene generation, expert demonstration synthesis, and sim-to-real transfer, still demand substantial human effort. We present AnyTask, an automated framework that pairs massively parallel GPU simulation with foundation models to design diverse manipulation tasks and synthesize robot data. We introduce three AnyTask agents for generating expert demonstrations aiming to solve as many tasks as possible: 1) ViPR, a novel task and motion planning agent with VLM-in-the-loop Parallel Refinement; 2) ViPR-Eureka, a reinforcement learning agent with generated dense rewards and LLM-guided contact sampling; 3) ViPR-RL, a hybrid planning and learning approach that jointly produces high-quality demonstrations with only sparse rewards. We train behavior cloning policies on generated data, validate them in simulation, and deploy them directly on real robot hardware. The policies generalize to novel object poses, achieving 44% average success across a suite of real-world pick-and-place, drawer opening, contact-rich pushing, and long-horizon manipulation tasks. Our project website is at https://anytask.rai-inst.com .
>
---
#### [new 020] ImagineNav++: Prompting Vision-Language Models as Embodied Navigator through Scene Imagination
- **分类: cs.RO**

- **简介: 该论文面向**无地图视觉导航任务**，解决VLMs因缺乏空间感知而难以直接用于机器人导航的问题。提出ImagineNav++框架：通过场景想象生成候选视图，以图像为提示让VLM选择最优视角，并引入选择性中央凹记忆机制实现长程空间推理，在开放词汇目标导航中达到SOTA。**

- **链接: [https://arxiv.org/pdf/2512.17435v1](https://arxiv.org/pdf/2512.17435v1)**

> **作者:** Teng Wang; Xinxin Zhao; Wenzhe Cai; Changyin Sun
>
> **备注:** 17 pages, 10 figures. arXiv admin note: text overlap with arXiv:2410.09874
>
> **摘要:** Visual navigation is a fundamental capability for autonomous home-assistance robots, enabling long-horizon tasks such as object search. While recent methods have leveraged Large Language Models (LLMs) to incorporate commonsense reasoning and improve exploration efficiency, their planning remains constrained by textual representations, which cannot adequately capture spatial occupancy or scene geometry--critical factors for navigation decisions. We explore whether Vision-Language Models (VLMs) can achieve mapless visual navigation using only onboard RGB/RGB-D streams, unlocking their potential for spatial perception and planning. We achieve this through an imagination-powered navigation framework, ImagineNav++, which imagines future observation images from candidate robot views and translates navigation planning into a simple best-view image selection problem for VLMs. First, a future-view imagination module distills human navigation preferences to generate semantically meaningful viewpoints with high exploration potential. These imagined views then serve as visual prompts for the VLM to identify the most informative viewpoint. To maintain spatial consistency, we develop a selective foveation memory mechanism, which hierarchically integrates keyframe observations via a sparse-to-dense framework, constructing a compact yet comprehensive memory for long-term spatial reasoning. This approach transforms goal-oriented navigation into a series of tractable point-goal navigation tasks. Extensive experiments on open-vocabulary object and instance navigation benchmarks show that ImagineNav++ achieves SOTA performance in mapless settings, even surpassing most map-based methods, highlighting the importance of scene imagination and memory in VLM-based spatial reasoning.
>
---
#### [new 021] Towards Senior-Robot Interaction: Reactive Robot Dog Gestures
- **分类: cs.RO**

- **简介: 该论文面向老年群体，解决陪伴机器人社交表达不足与交互不直观的问题。提出基于手势/头部识别的无遥控输入，及用强化学习训练的拟人化机械狗姿态输出（如抬爪），并在仿真与Unitree真机上验证可行性，同时分析了仿真到现实的迁移挑战。**

- **链接: [https://arxiv.org/pdf/2512.17136v1](https://arxiv.org/pdf/2512.17136v1)**

> **作者:** Chunyang Meng; Eduardo B. Sandoval; Ricardo Sosa; Francisco Cruz
>
> **备注:** Accepted at the Australasian Conference on Robotics and Automation (ACRA) 2025
>
> **摘要:** As the global population ages, many seniors face the problem of loneliness. Companion robots offer a potential solution. However, current companion robots often lack advanced functionality, while task-oriented robots are not designed for social interaction, limiting their suitability and acceptance by seniors. Our work introduces a senior-oriented system for quadruped robots that allows for more intuitive user input and provides more socially expressive output. For user input, we implemented a MediaPipe-based module for hand gesture and head movement recognition, enabling control without a remote. For output, we designed and trained robotic dog gestures using curriculum-based reinforcement learning in Isaac Gym, progressing from simple standing to three-legged balancing and leg extensions, and more. The final tests achieved over 95\% success on average in simulation, and we validated a key social gesture (the paw-lift) on a Unitree robot. Real-world tests demonstrated the feasibility and social expressiveness of this framework, while also revealing sim-to-real challenges in joint compliance, load distribution, and balance control. These contributions advance the development of practical quadruped robots as social companions for the senior and outline pathways for sim-to-real adaptation and inform future user studies.
>
---
#### [new 022] TakeAD: Preference-based Post-optimization for End-to-end Autonomous Driving with Expert Takeover Data
- **分类: cs.RO; cs.AI**

- **简介: 该论文属端到端自动驾驶任务，旨在解决IL模型开环训练与闭环部署间的性能鸿沟问题。作者提出TakeAD框架，利用专家接管数据，结合DAgger（模仿接管行为）和DPO（偏好对齐）进行后优化，提升闭环驾驶鲁棒性与恢复能力。**

- **链接: [https://arxiv.org/pdf/2512.17370v1](https://arxiv.org/pdf/2512.17370v1)**

> **作者:** Deqing Liu; Yinfeng Gao; Deheng Qian; Qichao Zhang; Xiaoqing Ye; Junyu Han; Yupeng Zheng; Xueyi Liu; Zhongpu Xia; Dawei Ding; Yifeng Pan; Dongbin Zhao
>
> **摘要:** Existing end-to-end autonomous driving methods typically rely on imitation learning (IL) but face a key challenge: the misalignment between open-loop training and closed-loop deployment. This misalignment often triggers driver-initiated takeovers and system disengagements during closed-loop execution. How to leverage those expert takeover data from disengagement scenarios and effectively expand the IL policy's capability presents a valuable yet unexplored challenge. In this paper, we propose TakeAD, a novel preference-based post-optimization framework that fine-tunes the pre-trained IL policy with this disengagement data to enhance the closed-loop driving performance. First, we design an efficient expert takeover data collection pipeline inspired by human takeover mechanisms in real-world autonomous driving systems. Then, this post optimization framework integrates iterative Dataset Aggregation (DAgger) for imitation learning with Direct Preference Optimization (DPO) for preference alignment. The DAgger stage equips the policy with fundamental capabilities to handle disengagement states through direct imitation of expert interventions. Subsequently, the DPO stage refines the policy's behavior to better align with expert preferences in disengagement scenarios. Through multiple iterations, the policy progressively learns recovery strategies for disengagement states, thereby mitigating the open-loop gap. Experiments on the closed-loop Bench2Drive benchmark demonstrate our method's effectiveness compared with pure IL methods, with comprehensive ablations confirming the contribution of each component.
>
---
#### [new 023] UniStateDLO: Unified Generative State Estimation and Tracking of Deformable Linear Objects Under Occlusion for Constrained Manipulation
- **分类: cs.RO**

- **简介: 该论文提出UniStateDLO，面向受限操作中被严重遮挡的可变形线状物体（如电缆），解决其单帧状态估计与跨帧跟踪难题。采用基于扩散模型的统一生成式方法，仅用合成数据训练，实现强鲁棒性、高精度、实时的遮挡下感知与零样本迁移。**

- **链接: [https://arxiv.org/pdf/2512.17764v1](https://arxiv.org/pdf/2512.17764v1)**

> **作者:** Kangchen Lv; Mingrui Yu; Shihefeng Wang; Xiangyang Ji; Xiang Li
>
> **备注:** The first two authors contributed equally. Project page: https://unistatedlo.github.io
>
> **摘要:** Perception of deformable linear objects (DLOs), such as cables, ropes, and wires, is the cornerstone for successful downstream manipulation. Although vision-based methods have been extensively explored, they remain highly vulnerable to occlusions that commonly arise in constrained manipulation environments due to surrounding obstacles, large and varying deformations, and limited viewpoints. Moreover, the high dimensionality of the state space, the lack of distinctive visual features, and the presence of sensor noises further compound the challenges of reliable DLO perception. To address these open issues, this paper presents UniStateDLO, the first complete DLO perception pipeline with deep-learning methods that achieves robust performance under severe occlusion, covering both single-frame state estimation and cross-frame state tracking from partial point clouds. Both tasks are formulated as conditional generative problems, leveraging the strong capability of diffusion models to capture the complex mapping between highly partial observations and high-dimensional DLO states. UniStateDLO effectively handles a wide range of occlusion patterns, including initial occlusion, self-occlusion, and occlusion caused by multiple objects. In addition, it exhibits strong data efficiency as the entire network is trained solely on a large-scale synthetic dataset, enabling zero-shot sim-to-real generalization without any real-world training data. Comprehensive simulation and real-world experiments demonstrate that UniStateDLO outperforms all state-of-the-art baselines in both estimation and tracking, producing globally smooth yet locally precise DLO state predictions in real time, even under substantial occlusions. Its integration as the front-end module in a closed-loop DLO manipulation system further demonstrates its ability to support stable feedback control in complex, constrained 3-D environments.
>
---
#### [new 024] Optimized Scheduling and Positioning of Mobile Manipulators in Collaborative Applications
- **分类: cs.RO**

- **简介: 该论文属机器人协同调度任务，旨在解决人机共融场景下移动机械臂的路径规划与任务调度问题。提出基于数字模型的黑箱优化框架，采用粒子群算法（PSO）协同优化基座位姿序列与任务时序，提升循环时间、任务顺序合理性及对人类存在的适应性。**

- **链接: [https://arxiv.org/pdf/2512.17584v1](https://arxiv.org/pdf/2512.17584v1)**

> **作者:** Christian Cella; Sole Ester Sonnino; Marco Faroni; Andrea Zanchettin; Paolo Rocco
>
> **备注:** Accepted at The IFAC Joint Conference on Computers, Cognition and Communication (J3C) 2025
>
> **摘要:** The growing integration of mobile robots in shared workspaces requires efficient path planning and coordination between the agents, accounting for safety and productivity. In this work, we propose a digital model-based optimization framework for mobile manipulators in human-robot collaborative environments, in order to determine the sequence of robot base poses and the task scheduling for the robot. The complete problem is treated as black-box, and Particle Swarm Optimization (PSO) is employed to balance conflicting Key-Performance Indicators (KPIs). We demonstrate improvements in cycle time, task sequencing, and adaptation to human presence in a collaborative box-packing scenario.
>
---
#### [new 025] RecipeMasterLLM: Revisiting RoboEarth in the Era of Large Language Models
- **分类: cs.RO**

- **简介: 该论文属知识图谱构建与机器人规划任务，旨在解决RoboEarth中手工构建OWL动作本体效率低的问题。作者提出RecipeMasterLLM，利用微调LLM+RAG，自动将用户指令转化为符合RoboEarth标准的知识图谱动作描述。**

- **链接: [https://arxiv.org/pdf/2512.17309v1](https://arxiv.org/pdf/2512.17309v1)**

> **作者:** Asil Kaan Bozcuoglu; Ziyuan Liu
>
> **摘要:** RoboEarth was a pioneering initiative in cloud robotics, establishing a foundational framework for robots to share and exchange knowledge about actions, objects, and environments through a standardized knowledge graph. Initially, this knowledge was predominantly hand-crafted by engineers using RDF triples within OWL Ontologies, with updates, such as changes in an object's pose, being asserted by the robot's control and perception routines. However, with the advent and rapid development of Large Language Models (LLMs), we believe that the process of knowledge acquisition can be significantly automated. To this end, we propose RecipeMasterLLM, a high-level planner, that generates OWL action ontologies based on a standardized knowledge graph in response to user prompts. This architecture leverages a fine-tuned LLM specifically trained to understand and produce action descriptions consistent with the RoboEarth standardized knowledge graph. Moreover, during the Retrieval-Augmented Generation (RAG) phase, environmental knowledge is supplied to the LLM to enhance its contextual understanding and improve the accuracy of the generated action descriptions.
>
---
#### [new 026] Diffusion Forcing for Multi-Agent Interaction Sequence Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向多智能体交互序列建模任务，旨在解决长时序、强耦合、变规模下多人运动生成难泛化的问题。提出MAGNet——基于扩散强迫的统一自回归扩散框架，支持双人预测、伙伴补全及多人协同生成，并显式建模智能体间耦合，实现高保真、可扩展的多角色运动合成。**

- **链接: [https://arxiv.org/pdf/2512.17900v1](https://arxiv.org/pdf/2512.17900v1)**

> **作者:** Vongani H. Maluleke; Kie Horiuchi; Lea Wilken; Evonne Ng; Jitendra Malik; Angjoo Kanazawa
>
> **摘要:** Understanding and generating multi-person interactions is a fundamental challenge with broad implications for robotics and social computing. While humans naturally coordinate in groups, modeling such interactions remains difficult due to long temporal horizons, strong inter-agent dependencies, and variable group sizes. Existing motion generation methods are largely task-specific and do not generalize to flexible multi-agent generation. We introduce MAGNet (Multi-Agent Diffusion Forcing Transformer), a unified autoregressive diffusion framework for multi-agent motion generation that supports a wide range of interaction tasks through flexible conditioning and sampling. MAGNet performs dyadic prediction, partner inpainting, and full multi-agent motion generation within a single model, and can autoregressively generate ultra-long sequences spanning hundreds of v. Building on Diffusion Forcing, we introduce key modifications that explicitly model inter-agent coupling during autoregressive denoising, enabling coherent coordination across agents. As a result, MAGNet captures both tightly synchronized activities (e.g, dancing, boxing) and loosely structured social interactions. Our approach performs on par with specialized methods on dyadic benchmarks while naturally extending to polyadic scenarios involving three or more interacting people, enabled by a scalable architecture that is agnostic to the number of agents. We refer readers to the supplemental video, where the temporal dynamics and spatial coordination of generated interactions are best appreciated. Project page: https://von31.github.io/MAGNet/
>
---
#### [new 027] RadarGen: Automotive Radar Point Cloud Generation from Cameras
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出RadarGen，一种基于扩散模型的跨模态生成方法，旨在从多视角相机图像合成逼真的车载雷达点云。它将雷达数据表示为鸟瞰图（含RCS与多普勒信息），融合深度、语义和运动线索引导生成，并通过轻量重建得到点云，解决雷达数据稀缺问题，提升多模态仿真可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.17897v1](https://arxiv.org/pdf/2512.17897v1)**

> **作者:** Tomer Borreda; Fangqiang Ding; Sanja Fidler; Shengyu Huang; Or Litany
>
> **备注:** Project page: https://radargen.github.io/
>
> **摘要:** We present RadarGen, a diffusion model for synthesizing realistic automotive radar point clouds from multi-view camera imagery. RadarGen adapts efficient image-latent diffusion to the radar domain by representing radar measurements in bird's-eye-view form that encodes spatial structure together with radar cross section (RCS) and Doppler attributes. A lightweight recovery step reconstructs point clouds from the generated maps. To better align generation with the visual scene, RadarGen incorporates BEV-aligned depth, semantic, and motion cues extracted from pretrained foundation models, which guide the stochastic generation process toward physically plausible radar patterns. Conditioning on images makes the approach broadly compatible, in principle, with existing visual datasets and simulation frameworks, offering a scalable direction for multimodal generative simulation. Evaluations on large-scale driving data show that RadarGen captures characteristic radar measurement distributions and reduces the gap to perception models trained on real data, marking a step toward unified generative simulation across sensing modalities.
>
---
#### [new 028] Fixed-Priority and EDF Schedules for ROS2 Graphs on Uniprocessor
- **分类: cs.DC; cs.OS; cs.RO; cs.SE**

- **简介: 该论文属实时调度任务，旨在解决ROS2对任意DAG图缺乏严格实时调度支持的问题。作者利用事件执行器实现固定优先级与EDF调度，将ROS2应用抽象为树森林，映射至传统DAG任务模型，并设计LIFO消息队列，在无显式优先级信息下复现经典调度行为。**

- **链接: [https://arxiv.org/pdf/2512.16926v1](https://arxiv.org/pdf/2512.16926v1)**

> **作者:** Oren Bell; Harun Teper; Mario Günzel; Chris Gill; Jian-Jia Chen
>
> **备注:** 18 pages, 5 figure
>
> **摘要:** This paper addresses limitations of current scheduling methods in the Robot Operating System (ROS)2, focusing on scheduling tasks beyond simple chains and analyzing arbitrary Directed Acyclic Graphs (DAGs). While previous research has focused mostly on chain-based scheduling with ad-hoc response time analyses, we propose a novel approach using the events executor to implement fixed-job-level-priority schedulers for arbitrary ROS2 graphs on uniprocessor systems. We demonstrate that ROS 2 applications can be abstracted as forests of trees, enabling the mapping of ROS 2 applications to traditional real-time DAG task models. Our usage of the events executor requires a special implementation of the events queue and a communication middleware that supports LIFO-ordered message delivery, features not yet standard in ROS2. We show that our implementation generates the same schedules as a conventional fixed-priority DAG task scheduler, in spite of lacking access to the precedence information that usually is required. This further closes the gap between established real-time systems theory and ROS2 scheduling analyses.
>
---
#### [new 029] Learning to Plan, Planning to Learn: Adaptive Hierarchical RL-MPC for Sample-Efficient Decision Making
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属智能决策任务，旨在解决复杂分层规划中样本效率低、鲁棒性差的问题。提出自适应分层RL-MPC框架：用RL动作引导MPPI采样，动态聚合样本优化价值估计，并在不确定性高处增强探索，显著提升数据效率与策略性能。**

- **链接: [https://arxiv.org/pdf/2512.17091v1](https://arxiv.org/pdf/2512.17091v1)**

> **作者:** Toshiaki Hori; Jonathan DeCastro; Deepak Gopinath; Avinash Balachandran; Guy Rosman
>
> **备注:** 23 pages, 8 figures. Under review
>
> **摘要:** We propose a new approach for solving planning problems with a hierarchical structure, fusing reinforcement learning and MPC planning. Our formulation tightly and elegantly couples the two planning paradigms. It leverages reinforcement learning actions to inform the MPPI sampler, and adaptively aggregates MPPI samples to inform the value estimation. The resulting adaptive process leverages further MPPI exploration where value estimates are uncertain, and improves training robustness and the overall resulting policies. This results in a robust planning approach that can handle complex planning problems and easily adapts to different applications, as demonstrated over several domains, including race driving, modified Acrobot, and Lunar Lander with added obstacles. Our results in these domains show better data efficiency and overall performance in terms of both rewards and task success, with up to a 72% increase in success rate compared to existing approaches, as well as accelerated convergence (x2.1) compared to non-adaptive sampling.
>
---
#### [new 030] Learning Safe Autonomous Driving Policies Using Predictive Safety Representations
- **分类: cs.LG; cs.RO**

- **简介: 该论文属安全强化学习任务，旨在解决自动驾驶中性能优化与安全约束的冲突问题。作者将SRPL框架应用于真实驾驶数据（WOMD/NuPlan），验证其提升奖励-安全权衡、鲁棒性及跨数据集泛化能力的有效性。**

- **链接: [https://arxiv.org/pdf/2512.17586v1](https://arxiv.org/pdf/2512.17586v1)**

> **作者:** Mahesh Keswani; Raunak Bhattacharyya
>
> **备注:** 8 pages, 4 figures. Submitted to ICRA 2026
>
> **摘要:** Safe reinforcement learning (SafeRL) is a prominent paradigm for autonomous driving, where agents are required to optimize performance under strict safety requirements. This dual objective creates a fundamental tension, as overly conservative policies limit driving efficiency while aggressive exploration risks safety violations. The Safety Representations for Safer Policy Learning (SRPL) framework addresses this challenge by equipping agents with a predictive model of future constraint violations and has shown promise in controlled environments. This paper investigates whether SRPL extends to real-world autonomous driving scenarios. Systematic experiments on the Waymo Open Motion Dataset (WOMD) and NuPlan demonstrate that SRPL can improve the reward-safety tradeoff, achieving statistically significant improvements in success rate (effect sizes r = 0.65-0.86) and cost reduction (effect sizes r = 0.70-0.83), with p < 0.05 for observed improvements. However, its effectiveness depends on the underlying policy optimizer and the dataset distribution. The results further show that predictive safety representations play a critical role in improving robustness to observation noise. Additionally, in zero-shot cross-dataset evaluation, SRPL-augmented agents demonstrate improved generalization compared to non-SRPL methods. These findings collectively demonstrate the potential of predictive safety representations to strengthen SafeRL for autonomous driving.
>
---
#### [new 031] DiffeoMorph: Learning to Morph 3D Shapes Using Differentiable Agent-Based Simulations
- **分类: cs.LG; cs.MA; cs.RO; q-bio.QM**

- **简介: 该论文提出DiffeoMorph，一种端到端可微框架，用于学习分布式智能体（如细胞）的形态发生协议，使其自组织成目标3D形状。它解决无中心控制下多智能体协同构形问题，创新性地引入基于Zernike多项式的SO(3)-不变形状匹配损失，并通过隐式微分实现对齐优化。**

- **链接: [https://arxiv.org/pdf/2512.17129v1](https://arxiv.org/pdf/2512.17129v1)**

> **作者:** Seong Ho Pahng; Guoye Guan; Benjamin Fefferman; Sahand Hormoz
>
> **摘要:** Biological systems can form complex three-dimensional structures through the collective behavior of identical agents -- cells that follow the same internal rules and communicate without central control. How such distributed control gives rise to precise global patterns remains a central question not only in developmental biology but also in distributed robotics, programmable matter, and multi-agent learning. Here, we introduce DiffeoMorph, an end-to-end differentiable framework for learning a morphogenesis protocol that guides a population of agents to morph into a target 3D shape. Each agent updates its position and internal state using an attention-based SE(3)-equivariant graph neural network, based on its own internal state and signals received from other agents. To train this system, we introduce a new shape-matching loss based on the 3D Zernike polynomials, which compares the predicted and target shapes as continuous spatial distributions, not as discrete point clouds, and is invariant to agent ordering, number of agents, and rigid-body transformations. To enforce full SO(3) invariance -- invariant to rotations yet sensitive to reflections, we include an alignment step that optimally rotates the predicted Zernike spectrum to match the target before computing the loss. This results in a bilevel problem, with the inner loop optimizing a unit quaternion for the best alignment and the outer loop updating the agent model. We compute gradients through the alignment step using implicit differentiation. We perform systematic benchmarking to establish the advantages of our shape-matching loss over other standard distance metrics for shape comparison tasks. We then demonstrate that DiffeoMorph can form a range of shapes -- from simple ellipsoids to complex morphologies -- using only minimal spatial cues.
>
---
## 更新

#### [replaced 001] Symphony: A Heuristic Normalized Calibrated Advantage Actor and Critic Algorithm in application for Humanoid Robots
- **分类: cs.RO; cs.NE**

- **简介: 该论文提出Symphony算法，面向人形机器人从零开始的强化学习任务，旨在解决样本效率低、动作不安全及训练不稳定问题。工作包括：引入“襁褓”正则化约束动作强度、设计过渡策略确定性AC架构、采用衰减回放缓冲区与时间优势机制，提升安全性、样本邻近性与训练效率。**

- **链接: [https://arxiv.org/pdf/2512.10477v3](https://arxiv.org/pdf/2512.10477v3)**

> **作者:** Timur Ishuov; Michele Folgheraiter; Madi Nurmanov; Goncalo Gordo; Richárd Farkas; József Dombi
>
> **备注:** https://github.com/SuspensionRailway/symphony
>
> **摘要:** In our work we not explicitly hint that it is a misconception to think that humans learn fast. Learning process takes time. Babies start learning to move in the restricted liquid area called placenta. Children often are limited by underdeveloped body. Even adults are not allowed to participate in complex competitions right away. However, with robots, when learning from scratch, we often don't have the privilege of waiting for dozen millions of steps. "Swaddling" regularization is responsible for restraining an agent in rapid but unstable development penalizing action strength in a specific way not affecting actions directly. The Symphony, Transitional-policy Deterministic Actor and Critic algorithm, is a concise combination of different ideas for possibility of training humanoid robots from scratch with Sample Efficiency, Sample Proximity and Safety of Actions in mind. It is no secret that continuous increase in Gaussian noise without appropriate smoothing is harmful for motors and gearboxes. Compared to Stochastic algorithms, we set a limited parametric noise and promote a reduced strength of actions, safely increasing entropy, since the actions are kind of immersed in weaker noise. When actions require more extreme values, actions rise above the weak noise. Training becomes empirically much safer for both the environment around and the robot's mechanisms. We use Fading Replay Buffer: using a fixed formula containing the hyperbolic tangent, we adjust the batch sampling probability: the memory contains a recent memory and a long-term memory trail. Fading Replay Buffer allows us to use Temporal Advantage when we improve the current Critic Network prediction compared to the exponential moving average. Temporal Advantage allows us to update Actor and Critic in one pass, as well as combine Actor and Critic in one Object and implement their Losses in one line.
>
---
#### [replaced 002] Efficient Image-Goal Navigation with Representative Latent World Model
- **分类: cs.RO**

- **简介: 该论文面向图像目标导航任务，旨在解决传统世界模型因像素级重建导致的计算开销大、规划效率低的问题。提出ReL-NWM方法，在DINOv3预训练语义表征空间中融合动作与历史信息，实现无需显式重建的高效潜空间预测与导航规划，并在仿真与真实机器人上验证了其性能。**

- **链接: [https://arxiv.org/pdf/2511.11011v2](https://arxiv.org/pdf/2511.11011v2)**

> **作者:** Zhiwei Zhang; Hui Zhang; Kaihong Huang; Chenghao Shi; Huimin Lu
>
> **摘要:** World models enable robots to conduct counterfactual reasoning in physical environments by predicting future world states. While conventional approaches often prioritize pixel-level reconstruction of future scenes, such detailed rendering is computationally intensive and unnecessary for planning tasks like navigation. We therefore propose that prediction and planning can be efficiently performed directly within a latent space of high-level semantic representations. To realize this, we introduce the Representative Latent space Navigation World Model (ReL-NWM). Rather than relying on reconstructionoriented latent embeddings, our method leverages a pre-trained representation encoder, DINOv3, and incorporates specialized mechanisms to effectively integrate action signals and historical context within this representation space. By operating entirely in the latent domain, our model bypasses expensive explicit reconstruction and achieves highly efficient navigation planning. Experiments show state-of-the-art trajectory prediction and image-goal navigation performance on multiple benchmarks. Additionally, we demonstrate real-world applicability by deploying the system on a Unitree G1 humanoid robot, confirming its efficiency and robustness in practical navigation scenarios.
>
---
#### [replaced 003] Mitigating Undesired Conditions in Flexible Production with Product-Process-Resource Asset Knowledge Graphs
- **分类: cs.RO**

- **简介: 该论文属工业智能任务，旨在解决柔性生产中难以分析与缓解异常工况的问题。提出产品-工艺-资源资产知识图谱（PPR-AKG）模型，融合语义技术与大语言模型，支持自然语言交互，提升异常识别、资源调度与质量保障能力。**

- **链接: [https://arxiv.org/pdf/2508.06278v2](https://arxiv.org/pdf/2508.06278v2)**

> **作者:** Petr Novak; Stefan Biffl; Marek Obitko; Petr Kadera
>
> **备注:** Originally published online by CEUR Workshop Proceedings (CEUR-WS.org, ISSN 1613-0073) within ISWC 2025 Companion Volume. Available online: https://ceur-ws.org/Vol-4085/ and https://ceur-ws.org/Vol-4085/paper9.pdf
>
> **摘要:** Contemporary industrial cyber-physical production systems (CPPS) composed of robotic workcells face significant challenges in the analysis of undesired conditions due to the flexibility of Industry 4.0 that disrupts traditional quality assurance mechanisms. This paper presents a novel industry-oriented semantic model called Product-Process-Resource Asset Knowledge Graph (PPR-AKG), which is designed to analyze and mitigate undesired conditions in flexible CPPS. Built on top of the well-proven Product-Process-Resource (PPR) model originating from ISA-95 and VDI-3682, a comprehensive OWL ontology addresses shortcomings of conventional model-driven engineering for CPPS, particularly inadequate undesired condition and error handling representation. The integration of semantic technologies with large language models (LLMs) provides intuitive interfaces for factory operators, production planners, and engineers to interact with the entire model using natural language. Evaluation with the use case addressing electric vehicle battery remanufacturing demonstrates that the PPR-AKG approach efficiently supports resource allocation based on explicitly represented capabilities as well as identification and mitigation of undesired conditions in production. The key contributions include (1) a holistic PPR-AKG model capturing multi-dimensional production knowledge, and (2) the useful combination of the PPR-AKG with LLM-based chatbots for human interaction.
>
---
#### [replaced 004] A Practical Guide for Incorporating Symmetry in Diffusion Policy
- **分类: cs.RO**

- **简介: 该论文面向机器人策略学习任务，旨在解决扩散策略中引入对称性时实现复杂、难集成的问题。提出三种轻量方法：相对动作+眼在手感知实现SE(3)不变性、集成等变视觉编码器、用帧平均进行对称特征提取，并实验证明其性能媲美全等变模型且更易实现。**

- **链接: [https://arxiv.org/pdf/2505.13431v4](https://arxiv.org/pdf/2505.13431v4)**

> **作者:** Dian Wang; Boce Hu; Shuran Song; Robin Walters; Robert Platt
>
> **备注:** NeurIPS 2025
>
> **摘要:** Recently, equivariant neural networks for policy learning have shown promising improvements in sample efficiency and generalization, however, their wide adoption faces substantial barriers due to implementation complexity. Equivariant architectures typically require specialized mathematical formulations and custom network design, posing significant challenges when integrating with modern policy frameworks like diffusion-based models. In this paper, we explore a number of straightforward and practical approaches to incorporate symmetry benefits into diffusion policies without the overhead of full equivariant designs. Specifically, we investigate (i) invariant representations via relative trajectory actions and eye-in-hand perception, (ii) integrating equivariant vision encoders, and (iii) symmetric feature extraction with pretrained encoders using Frame Averaging. We first prove that combining eye-in-hand perception with relative or delta action parameterization yields inherent SE(3)-invariance, thus improving policy generalization. We then perform a systematic experimental study on those design choices for integrating symmetry in diffusion policies, and conclude that an invariant representation with equivariant feature extraction significantly improves the policy performance. Our method achieves performance on par with or exceeding fully equivariant architectures while greatly simplifying implementation.
>
---
#### [replaced 005] DHP: Discrete Hierarchical Planning for Hierarchical Reinforcement Learning Agents
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文面向分层强化学习（HRL）的长程视觉规划任务，解决因连续距离度量误差导致的子目标不可行问题。提出离散分层规划（DHP）：用离散可达性检查替代距离估计，递归构建树状计划，引入新优势估计与无专家数据的探索策略，显著提升成功率与泛化性。**

- **链接: [https://arxiv.org/pdf/2502.01956v3](https://arxiv.org/pdf/2502.01956v3)**

> **作者:** Shashank Sharma; Janina Hoffmann; Vinay Namboodiri
>
> **摘要:** Hierarchical Reinforcement Learning (HRL) agents often struggle with long-horizon visual planning due to their reliance on error-prone distance metrics. We propose Discrete Hierarchical Planning (DHP), a method that replaces continuous distance estimates with discrete reachability checks to evaluate subgoal feasibility. DHP recursively constructs tree-structured plans by decomposing long-term goals into sequences of simpler subtasks, using a novel advantage estimation strategy that inherently rewards shorter plans and generalizes beyond training depths. In addition, to address the data efficiency challenge, we introduce an exploration strategy that generates targeted training examples for the planning modules without needing expert data. Experiments in 25-room navigation environments demonstrate a 100% success rate (vs. 90% baseline). We also present an offline variant that achieves state-of-the-art results on OGBench benchmarks, with up to 71% absolute gains on giant HumanoidMaze tasks, demonstrating our core contributions are architecture-agnostic. The method also generalizes to momentum-based control tasks and requires only log N steps for replanning. Theoretical analysis and ablations validate our design choices.
>
---
#### [replaced 006] An Anatomy of Vision-Language-Action Models: From Modules to Milestones and Challenges
- **分类: cs.RO**

- **简介: 该论文是一篇综述，旨在系统梳理视觉-语言-动作（VLA）模型的研究现状。它聚焦机器人具身智能，按模块、里程碑、挑战三阶段展开，重点剖析表征、执行、泛化、安全及数据评估五大核心挑战，为研究者提供入门指南与前沿 roadmap。**

- **链接: [https://arxiv.org/pdf/2512.11362v3](https://arxiv.org/pdf/2512.11362v3)**

> **作者:** Chao Xu; Suyu Zhang; Yang Liu; Baigui Sun; Weihong Chen; Bo Xu; Qi Liu; Juncheng Wang; Shujun Wang; Shan Luo; Jan Peters; Athanasios V. Vasilakos; Stefanos Zafeiriou; Jiankang Deng
>
> **备注:** project page: https://suyuz1.github.io/VLA-Survey-Anatomy/
>
> **摘要:** Vision-Language-Action (VLA) models are driving a revolution in robotics, enabling machines to understand instructions and interact with the physical world. This field is exploding with new models and datasets, making it both exciting and challenging to keep pace with. This survey offers a clear and structured guide to the VLA landscape. We design it to follow the natural learning path of a researcher: we start with the basic Modules of any VLA model, trace the history through key Milestones, and then dive deep into the core Challenges that define recent research frontier. Our main contribution is a detailed breakdown of the five biggest challenges in: (1) Representation, (2) Execution, (3) Generalization, (4) Safety, and (5) Dataset and Evaluation. This structure mirrors the developmental roadmap of a generalist agent: establishing the fundamental perception-action loop, scaling capabilities across diverse embodiments and environments, and finally ensuring trustworthy deployment-all supported by the essential data infrastructure. For each of them, we review existing approaches and highlight future opportunities. We position this paper as both a foundational guide for newcomers and a strategic roadmap for experienced researchers, with the dual aim of accelerating learning and inspiring new ideas in embodied intelligence. A live version of this survey, with continuous updates, is maintained on our \href{https://suyuz1.github.io/VLA-Survey-Anatomy/}{project page}.
>
---
#### [replaced 007] Collaborative Object Handover in a Robot Crafting Assistant
- **分类: cs.RO**

- **简介: 该论文研究人机协作中的物体交接任务，旨在提升机器人在自然手工场景中安全高效地与人交接物体的能力。作者基于人类遥操作数据训练协作交接模型，并通过交叉验证和用户研究评估性能，结果表明模型有效但仍有改进空间。**

- **链接: [https://arxiv.org/pdf/2502.19991v2](https://arxiv.org/pdf/2502.19991v2)**

> **作者:** Leimin Tian; Shiyu Xu; Kerry He; Rachel Love; Akansel Cosgun; Dana Kulic
>
> **备注:** Published at the 2025 Australasian Conference on Robotics and Automation (ACRA 2025): https://www.araa.asn.au/conference/acra-2025/
>
> **摘要:** Robots are increasingly working alongside people, delivering food to patrons in restaurants or helping workers on assembly lines. These scenarios often involve object handovers between the person and the robot. To achieve safe and efficient human-robot collaboration (HRC), it is important to incorporate human context in a robot's handover strategies. We develop a collaborative handover model trained on human teleoperation data collected in a naturalistic crafting task. To evaluate its performance, we conduct cross-validation experiments on the training dataset as well as a user study in the same HRC crafting task. The handover episodes and user perceptions of the autonomous handover policy were compared with those of the human teleoperated handovers. While the cross-validation experiment and user study indicate that the autonomous policy successfully achieved collaborative handovers, the comparison with human teleoperation revealed avenues for further improvements.
>
---
#### [replaced 008] VLA-AN: An Efficient and Onboard Vision-Language-Action Framework for Aerial Navigation in Complex Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VLA-AN框架，面向轻量无人机在复杂环境中的自主导航任务。旨在解决域偏移、时序推理弱、生成式动作不安全及机载部署难四大问题。工作包括：构建3D-GS仿真数据集、设计三阶段训练策略、开发轻量安全动作模块、优化机载推理，实现实时高效闭环导航。**

- **链接: [https://arxiv.org/pdf/2512.15258v2](https://arxiv.org/pdf/2512.15258v2)**

> **作者:** Yuze Wu; Mo Zhu; Xingxing Li; Yuheng Du; Yuxin Fan; Wenjun Li; Zhichao Han; Xin Zhou; Fei Gao
>
> **摘要:** This paper proposes VLA-AN, an efficient and onboard Vision-Language-Action (VLA) framework dedicated to autonomous drone navigation in complex environments. VLA-AN addresses four major limitations of existing large aerial navigation models: the data domain gap, insufficient temporal navigation with reasoning, safety issues with generative action policies, and onboard deployment constraints. First, we construct a high-fidelity dataset utilizing 3D Gaussian Splatting (3D-GS) to effectively bridge the domain gap. Second, we introduce a progressive three-stage training framework that sequentially reinforces scene comprehension, core flight skills, and complex navigation capabilities. Third, we design a lightweight, real-time action module coupled with geometric safety correction. This module ensures fast, collision-free, and stable command generation, mitigating the safety risks inherent in stochastic generative policies. Finally, through deep optimization of the onboard deployment pipeline, VLA-AN achieves a robust real-time 8.3x improvement in inference throughput on resource-constrained UAVs. Extensive experiments demonstrate that VLA-AN significantly improves spatial grounding, scene reasoning, and long-horizon navigation, achieving a maximum single-task success rate of 98.1%, and providing an efficient, practical solution for realizing full-chain closed-loop autonomy in lightweight aerial robots.
>
---
#### [replaced 009] Maintaining the Level of a Payload carried by Multi-Robot System on Irregular Surface
- **分类: cs.RO**

- **简介: 该论文研究多机器人协同运载任务，旨在未知不平地形下保持负载姿态水平。提出开环与PID闭环结合的控制器，通过各机器人顶部线性执行器实时调节高度，实现负载姿态稳定。工作包括系统设计、控制策略提出及复杂仿真地形验证。**

- **链接: [https://arxiv.org/pdf/2512.16024v2](https://arxiv.org/pdf/2512.16024v2)**

> **作者:** Rishabh Dev Yadav; Shrey Agrawal; Kamalakar Karlapalem
>
> **摘要:** In this paper, we introduce a multi robot payload transport system to carry payloads through an environment of unknown and uneven inclinations while maintaining the desired orientation of the payload. For this task, we used custom built robots with a linear actuator (pistons) mounted on top of each robot. The system continuously monitors the payload's orientation and computes the required piston height of each robot to maintain the desired orientation of the payload. In this work, we propose an open loop controller coupled with a closed loop PID controller to achieve the goal. As our modelling makes no assumptions on the type of terrain, the system can work on any unknown and uneven terrains and inclinations. We showcase the efficacy of our proposed controller by testing it on various simulated environments with varied and complex terrains.
>
---
#### [replaced 010] Cooperative Task Spaces for Multi-Arm Manipulation Control based on Similarity Transformations
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出基于相似变换与共形几何代数的多臂协作任务空间理论框架，解决高自由度多臂系统协同控制建模难问题；推导了对应雅可比矩阵，支持嵌入操作空间控制，并自然保留零空间用于次级目标优化。**

- **链接: [https://arxiv.org/pdf/2510.26362v2](https://arxiv.org/pdf/2510.26362v2)**

> **作者:** Tobias Löw; Cem Bilaloglu; Sylvain Calinon
>
> **摘要:** Many tasks in human environments require collaborative behavior between multiple kinematic chains, either to provide additional support for carrying big and bulky objects or to enable the dexterity that is required for in-hand manipulation. Since these complex systems often have a very high number of degrees of freedom coordinating their movements is notoriously difficult to model. In this article, we present the derivation of the theoretical foundations for cooperative task spaces of multi-arm robotic systems based on geometric primitives defined using conformal geometric algebra. Based on the similarity transformations of these cooperative geometric primitives, we derive an abstraction of complex robotic systems that enables representing these systems in a way that directly corresponds to single-arm systems. By deriving the associated analytic and geometric Jacobian matrices, we then show the straightforward integration of our approach into classical control techniques rooted in operational space control. We demonstrate this using bimanual manipulators, humanoids and multi-fingered hands in optimal control experiments for reaching desired geometric primitives and in teleoperation experiments using differential kinematics control. We then discuss how the geometric primitives naturally embed nullspace structures into the controllers that can be exploited for introducing secondary control objectives. This work, represents the theoretical foundations of this cooperative manipulation control framework, and thus the experiments are presented in an abstract way, while giving pointers towards potential future applications.
>
---
#### [replaced 011] Deadlock-Free Hybrid RL-MAPF Framework for Zero-Shot Multi-Robot Navigation
- **分类: cs.RO**

- **简介: 该论文面向多机器人导航任务，解决RL策略在未知环境易致死锁与泛化差的问题。提出融合RL反应式导航与按需MAPF的混合框架，通过安全层检测死锁并触发全局路径重规划，实现零样本下高成功率、低冲突的协同导航。**

- **链接: [https://arxiv.org/pdf/2511.22685v2](https://arxiv.org/pdf/2511.22685v2)**

> **作者:** Haoyi Wang; Licheng Luo; Yiannis Kantaros; Bruno Sinopoli; Mingyu Cai
>
> **备注:** 18 pages (including appendix), 3 figures. Project page (videos, animations, additional resources): https://wanghaoyi518.github.io/rl-mapf-project-page/
>
> **摘要:** Multi-robot navigation in cluttered environments presents fundamental challenges in balancing reactive collision avoidance with long-range goal achievement. When navigating through narrow passages or confined spaces, deadlocks frequently emerge that prevent agents from reaching their destinations, particularly when Reinforcement Learning (RL) control policies encounter novel configurations out of learning distribution. Existing RL-based approaches suffer from limited generalization capability in unseen environments. We propose a hybrid framework that seamlessly integrates RL-based reactive navigation with on-demand Multi-Agent Path Finding (MAPF) to explicitly resolve topological deadlocks. Our approach integrates a safety layer that monitors agent progress to detect deadlocks and, when detected, triggers a coordination controller for affected agents. The framework constructs globally feasible trajectories via MAPF and regulates waypoint progression to reduce inter-agent conflicts during navigation. Extensive evaluation on dense multi-agent benchmarks shows that our method boosts task completion from marginal to near-universal success, markedly reducing deadlocks and collisions. When integrated with hierarchical task planning, it enables coordinated navigation for heterogeneous robots, demonstrating that coupling reactive RL navigation with selective MAPF intervention yields a robust, zero-shot performance.
>
---
#### [replaced 012] MiVLA: Towards Generalizable Vision-Language-Action Model with Human-Robot Mutual Imitation Pre-training
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MiVLA模型，属视觉-语言-动作（VLA）任务，旨在解决现有VLAs因视角、外观和形态差异导致的跨人类-机器人泛化能力弱问题。通过人类与机器人双向行为模仿预训练，利用手/臂运动学对齐，融合真实人类视频与仿真机器人数据，提升下游任务泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.15411v2](https://arxiv.org/pdf/2512.15411v2)**

> **作者:** Zhenhan Yin; Xuanhan Wang; Jiahao Jiang; Kaiyuan Deng; Pengqi Chen; Shuangle Li; Chong Liu; Xing Xu; Jingkuan Song; Lianli Gao; Heng Tao Shen
>
> **摘要:** While leveraging abundant human videos and simulated robot data poses a scalable solution to the scarcity of real-world robot data, the generalization capability of existing vision-language-action models (VLAs) remains limited by mismatches in camera views, visual appearance, and embodiment morphologies. To overcome this limitation, we propose MiVLA, a generalizable VLA empowered by human-robot mutual imitation pre-training, which leverages inherent behavioral similarity between human hands and robotic arms to build a foundation of strong behavioral priors for both human actions and robotic control. Specifically, our method utilizes kinematic rules with left/right hand coordinate systems for bidirectional alignment between human and robot action spaces. Given human or simulated robot demonstrations, MiVLA is trained to forecast behavior trajectories for one embodiment, and imitate behaviors for another one unseen in the demonstration. Based on this mutual imitation, it integrates the behavioral fidelity of real-world human data with the manipulative diversity of simulated robot data into a unified model, thereby enhancing the generalization capability for downstream tasks. Extensive experiments conducted on both simulation and real-world platforms with three robots (ARX, PiPer and LocoMan), demonstrate that MiVLA achieves strong improved generalization capability, outperforming state-of-the-art VLAs (e.g., $\boldsymbolπ_{0}$, $\boldsymbolπ_{0.5}$ and H-RDT) by 25% in simulation, and 14% in real-world robot control tasks.
>
---
#### [replaced 013] mimic-video: Video-Action Models for Generalizable Robot Control Beyond VLAs
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出mimic-video，一种视频-动作模型（VAM），旨在解决视觉语言动作模型（VLAs）缺乏物理因果理解、依赖大量专家数据的问题。它用预训练视频模型联合建模语义与动态，并通过流匹配解码器实现逆动力学映射，提升机器人操控的泛化性、样本效率与收敛速度。**

- **链接: [https://arxiv.org/pdf/2512.15692v2](https://arxiv.org/pdf/2512.15692v2)**

> **作者:** Jonas Pai; Liam Achenbach; Victoriano Montesinos; Benedek Forrai; Oier Mees; Elvis Nava
>
> **备注:** Revised Introduction, Related Work, and Appendix. Additional minor notational and grammatical fixes
>
> **摘要:** Prevailing Vision-Language-Action Models (VLAs) for robotic manipulation are built upon vision-language backbones pretrained on large-scale, but disconnected static web data. As a result, despite improved semantic generalization, the policy must implicitly infer complex physical dynamics and temporal dependencies solely from robot trajectories. This reliance creates an unsustainable data burden, necessitating continuous, large-scale expert data collection to compensate for the lack of innate physical understanding. We contend that while vision-language pretraining effectively captures semantic priors, it remains blind to physical causality. A more effective paradigm leverages video to jointly capture semantics and visual dynamics during pretraining, thereby isolating the remaining task of low-level control. To this end, we introduce mimic-video, a novel Video-Action Model (VAM) that pairs a pretrained Internet-scale video model with a flow matching-based action decoder conditioned on its latent representations. The decoder serves as an Inverse Dynamics Model (IDM), generating low-level robot actions from the latent representation of video-space action plans. Our extensive evaluation shows that our approach achieves state-of-the-art performance on simulated and real-world robotic manipulation tasks, improving sample efficiency by 10x and convergence speed by 2x compared to traditional VLA architectures.
>
---
#### [replaced 014] Accelerating Hybrid Model Predictive Control using Warm-Started Generalized Benders Decomposition
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对混合模型预测控制（MPC）实时性不足问题，提出基于广义Benders分解的加速算法，通过在线缓存并迁移割平面实现跨步长热启动，理论分析其子最优界，并在多机器人仿真中验证速度提升2–3倍、达1000 Hz。**

- **链接: [https://arxiv.org/pdf/2406.00780v2](https://arxiv.org/pdf/2406.00780v2)**

> **作者:** Xuan Lin
>
> **备注:** Significantly revised to emphasize theoretical bounds. The heuristic Master Problem algorithm and GCS tightening experiments are preserved in v1
>
> **摘要:** Hybrid model predictive control with both continuous and discrete variables is widely applicable to robotic control tasks, especially those involving contacts with the environment. Due to combinatorial complexity, the solving speed of hybrid MPC can be insufficient for real-time applications. In this paper, we propose a hybrid MPC algorithm based on Generalized Benders Decomposition. The algorithm enumerates and stores cutting planes online inside a finite buffer and transfers them across MPC iterations to provide warm-starts for new problem instances, significantly enhancing solving speed. We theoretically analyze this warm-starting performance by modeling the deviation of mode sequences through temporal shifting and stretching, deriving bounds on the dual gap between transferred optimality cuts and the true optimal costs, and utilizing these bounds to quantify the level of suboptimality guaranteed in the first solve of the Benders Master Problem. Our algorithm is validated in simulation through controlling a cart-pole system with soft contact walls, a free-flying robot navigating around obstacles, and a humanoid robot standing on one leg while pushing against walls with its hands for balance. For our benchmark problems, the algorithm enumerates cuts on the order of only tens to hundreds while reaching speeds 2-3 times faster than the off-the-shelf solver Gurobi, oftentimes exceeding 1000 Hz. The code is available at https://github.com/XuanLin/Benders-MPC.
>
---
#### [replaced 015] Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究VLA模型在物理传感器攻击下的鲁棒性问题，属安全与鲁棒性任务。首次系统评估相机/麦克风物理攻击对VLA的影响，提出“Real-Sim-Real”框架模拟并验证8类攻击，揭示脆弱性规律，并设计对抗训练防御方法提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.10008v2](https://arxiv.org/pdf/2511.10008v2)**

> **作者:** Xuancun Lu; Jiaxiang Chen; Shilin Xiao; Zizhi Jin; Zhangrui Chen; Hanwen Yu; Bohan Qian; Ruochen Zhou; Xiaoyu Ji; Wenyuan Xu
>
> **备注:** Accepted by AAAI 2026 main track
>
> **摘要:** Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored. To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel "Real-Sim-Real" framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.
>
---
#### [replaced 016] CBMC-V3: A CNS-inspired Control Framework Towards Manipulation Agility with SNN
- **分类: cs.RO**

- **简介: 该论文提出CBMC-V3框架，属机器人控制任务，旨在解决服务场景中机械臂敏捷操纵难题。基于CNS启发，构建五模块、三层级、双通路的脉冲神经网络（SNN）控制架构，实现动态轨迹跟踪与实时自适应调节，并在仿真与实物平台验证其优于工业级位置控制。**

- **链接: [https://arxiv.org/pdf/2511.04109v2](https://arxiv.org/pdf/2511.04109v2)**

> **作者:** Yanbo Pang; Qingkai Li; Mingguo Zhao
>
> **摘要:** As robotic arm applications extend beyond industrial settings into service-oriented sectors such as catering, household and retail, existing control algorithms struggle to achieve the agile manipulation required for complex environments with dynamic trajectories, unpredictable interactions, and diverse objects. This paper presents a biomimetic control framework based on Spiking Neural Networks (SNNs), inspired by the human Central Nervous System (CNS), to achieve agile control in such environments. The proposed framework features five control modules (cerebral cortex, cerebellum, thalamus, brainstem, and spinal cord), three hierarchical control levels (first-order, second-order, and third-order), and two information pathways (ascending and descending). Each module is fully implemented using SNN. The spinal cord module uses spike encoding and Leaky Integrate-and-Fire (LIF) neurons for feedback control. The brainstem module employs a network of LIF and non-spiking LIF neurons to dynamically adjust spinal cord parameters via reinforcement learning. The thalamus module similarly employs a network of LIF and non-spiking LIF neurons to adjust the cerebellum's torque outputs via reinforcement learning. The cerebellum module, which provides feedfoward gravity compensation torques, uses a recurrent SNN to learn the robotic arm's dynamics through regression. The framework is validated both in simulation and on real-world robotic arm platform under various loads and trajectories. Results demonstrate that our method outperforms the industrial-grade position control in manipulation agility.
>
---
#### [replaced 017] MILE: A Mechanically Isomorphic Exoskeleton Data Collection System with Fingertip Visuotactile Sensing for Dexterous Manipulation
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文提出MILE系统，属机器人模仿学习任务，旨在解决手部灵巧操作数据匮乏、运动重定向不准及缺失高分辨率指尖触觉的问题。工作包括设计机械同构的外骨骼与机械手，实现零重定向映射，并集成指尖视触觉传感，高效采集多模态高保真数据。**

- **链接: [https://arxiv.org/pdf/2512.00324v2](https://arxiv.org/pdf/2512.00324v2)**

> **作者:** Jinda Du; Jieji Ren; Qiaojun Yu; Ningbin Zhang; Yu Deng; Xingyu Wei; Yufei Liu; Guoying Gu; Xiangyang Zhu
>
> **摘要:** Imitation learning provides a promising approach to dexterous hand manipulation, but its effectiveness is limited by the lack of large-scale, high-fidelity data. Existing data-collection pipelines suffer from inaccurate motion retargeting, low data-collection efficiency, and missing high-resolution fingertip tactile sensing. We address this gap with MILE, a mechanically isomorphic teleoperation and data-collection system co-designed from human hand to exoskeleton to robotic hand. The exoskeleton is anthropometrically derived from the human hand, and the robotic hand preserves one-to-one joint-position isomorphism, eliminating nonlinear retargeting and enabling precise, natural control. The exoskeleton achieves a multi-joint mean absolute angular error below one degree, while the robotic hand integrates compact fingertip visuotactile modules that provide high-resolution tactile observations. Built on this retargeting-free interface, we teleoperate complex, contact-rich in-hand manipulation and efficiently collect a multimodal dataset comprising high-resolution fingertip visuotactile signals, RGB-D images, and joint positions. The teleoperation pipeline achieves a mean success rate improvement of 64%. Incorporating fingertip tactile observations further increases the success rate by an average of 25% over the vision-only baseline, validating the fidelity and utility of the dataset. Further details are available at: https://sites.google.com/view/mile-system.
>
---
