# 机器人 cs.RO

- **最新发布 30 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] An Informative Planning Framework for Target Tracking and Active Mapping in Dynamic Environments with ASVs
- **分类: cs.RO**

- **简介: 论文研究动态环境中基于ASV的目标跟踪与主动建图任务，解决漂移目标的时空预测与地图更新问题，提出融合时空预测网络与自适应规划目标的框架，并通过仿真与实地测试验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.14636v1](http://arxiv.org/pdf/2508.14636v1)**

> **作者:** Sanjeev Ramkumar Sudha; Marija Popović; Erlend M. Coates
>
> **备注:** Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Mobile robot platforms are increasingly being used to automate information gathering tasks such as environmental monitoring. Efficient target tracking in dynamic environments is critical for applications such as search and rescue and pollutant cleanups. In this letter, we study active mapping of floating targets that drift due to environmental disturbances such as wind and currents. This is a challenging problem as it involves predicting both spatial and temporal variations in the map due to changing conditions. We propose an informative path planning framework to map an arbitrary number of moving targets with initially unknown positions in dynamic environments. A key component of our approach is a spatiotemporal prediction network that predicts target position distributions over time. We propose an adaptive planning objective for target tracking that leverages these predictions. Simulation experiments show that our proposed planning objective improves target tracking performance compared to existing methods that consider only entropy reduction as the planning objective. Finally, we validate our approach in field tests using an autonomous surface vehicle, showcasing its ability to track targets in real-world monitoring scenarios.
>
---
#### [new 002] TRUST-Planner: Topology-guided Robust Trajectory Planner for AAVs with Uncertain Obstacle Spatial-temporal Avoidance
- **分类: cs.RO**

- **简介: 论文针对AAVs在动态环境中的避障问题，提出TRUST-Planner，采用拓扑引导的分层框架，结合DEV-PRM、UTF-MINCO和DDF，实现高效鲁棒的时空避障，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.14610v1](http://arxiv.org/pdf/2508.14610v1)**

> **作者:** Junzhi Li; Teng Long; Jingliang Sun; Jianxin Zhong
>
> **摘要:** Despite extensive developments in motion planning of autonomous aerial vehicles (AAVs), existing frameworks faces the challenges of local minima and deadlock in complex dynamic environments, leading to increased collision risks. To address these challenges, we present TRUST-Planner, a topology-guided hierarchical planning framework for robust spatial-temporal obstacle avoidance. In the frontend, a dynamic enhanced visible probabilistic roadmap (DEV-PRM) is proposed to rapidly explore topological paths for global guidance. The backend utilizes a uniform terminal-free minimum control polynomial (UTF-MINCO) and dynamic distance field (DDF) to enable efficient predictive obstacle avoidance and fast parallel computation. Furthermore, an incremental multi-branch trajectory management framework is introduced to enable spatio-temporal topological decision-making, while efficiently leveraging historical information to reduce replanning time. Simulation results show that TRUST-Planner outperforms baseline competitors, achieving a 96\% success rate and millisecond-level computation efficiency in tested complex environments. Real-world experiments further validate the feasibility and practicality of the proposed method.
>
---
#### [new 003] No More Marching: Learning Humanoid Locomotion for Short-Range SE(2) Targets
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对双足人形机器人短程SE(2)目标姿态移动任务，解决传统方法因优化速度而非直接姿态到达导致的低效"行军式"问题，提出基于强化学习的新型奖励函数与评估框架，实现高效自然的目标导向运动。**

- **链接: [http://arxiv.org/pdf/2508.14098v1](http://arxiv.org/pdf/2508.14098v1)**

> **作者:** Pranay Dugar; Mohitvishnu S. Gadde; Jonah Siekmann; Yesh Godse; Aayam Shrestha; Alan Fern
>
> **摘要:** Humanoids operating in real-world workspaces must frequently execute task-driven, short-range movements to SE(2) target poses. To be practical, these transitions must be fast, robust, and energy efficient. While learning-based locomotion has made significant progress, most existing methods optimize for velocity-tracking rather than direct pose reaching, resulting in inefficient, marching-style behavior when applied to short-range tasks. In this work, we develop a reinforcement learning approach that directly optimizes humanoid locomotion for SE(2) targets. Central to this approach is a new constellation-based reward function that encourages natural and efficient target-oriented movement. To evaluate performance, we introduce a benchmarking framework that measures energy consumption, time-to-target, and footstep count on a distribution of SE(2) goals. Our results show that the proposed approach consistently outperforms standard methods and enables successful transfer from simulation to hardware, highlighting the importance of targeted reward design for practical short-range humanoid locomotion.
>
---
#### [new 004] EAROL: Environmental Augmented Perception-Aware Planning and Robust Odometry via Downward-Mounted Tilted LiDAR
- **分类: cs.RO**

- **简介: 论文针对无人机在开放顶场景中的定位漂移与感知-规划耦合问题，提出EAROL框架，结合倾斜LiDAR、LIO系统及优化算法，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.14554v1](http://arxiv.org/pdf/2508.14554v1)**

> **作者:** Xinkai Liang; Yigu Ge; Yangxi Shi; Haoyu Yang; Xu Cao; Hao Fang
>
> **备注:** Accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025). This work has been submitted to the IEEE for possible publication
>
> **摘要:** To address the challenges of localization drift and perception-planning coupling in unmanned aerial vehicles (UAVs) operating in open-top scenarios (e.g., collapsed buildings, roofless mazes), this paper proposes EAROL, a novel framework with a downward-mounted tilted LiDAR configuration (20{\deg} inclination), integrating a LiDAR-Inertial Odometry (LIO) system and a hierarchical trajectory-yaw optimization algorithm. The hardware innovation enables constraint enhancement via dense ground point cloud acquisition and forward environmental awareness for dynamic obstacle detection. A tightly-coupled LIO system, empowered by an Iterative Error-State Kalman Filter (IESKF) with dynamic motion compensation, achieves high level 6-DoF localization accuracy in feature-sparse environments. The planner, augmented by environment, balancing environmental exploration, target tracking precision, and energy efficiency. Physical experiments demonstrate 81% tracking error reduction, 22% improvement in perceptual coverage, and near-zero vertical drift across indoor maze and 60-meter-scale outdoor scenarios. This work proposes a hardware-algorithm co-design paradigm, offering a robust solution for UAV autonomy in post-disaster search and rescue missions. We will release our software and hardware as an open-source package for the community. Video: https://youtu.be/7av2ueLSiYw.
>
---
#### [new 005] Taming VR Teleoperation and Learning from Demonstration for Multi-Task Bimanual Table Service Manipulation
- **分类: cs.RO**

- **简介: 该论文解决多任务双臂桌面服务操作问题，通过结合VR远程操作与学习示范（LfD），利用高保真远程控制和ACT策略训练，高效完成桌布展开、披萨放置及容器开关等任务，获ICRA 2025竞赛冠军。**

- **链接: [http://arxiv.org/pdf/2508.14542v1](http://arxiv.org/pdf/2508.14542v1)**

> **作者:** Weize Li; Zhengxiao Han; Lixin Xu; Xiangyu Chen; Harrison Bounds; Chenrui Zhang; Yifan Xu
>
> **备注:** Technical report of First-place/Champion solution at IEEE ICRA 2025 What Bimanuals Can Do (WBCD) Challenge - Table Services Track
>
> **摘要:** This technical report presents the champion solution of the Table Service Track in the ICRA 2025 What Bimanuals Can Do (WBCD) competition. We tackled a series of demanding tasks under strict requirements for speed, precision, and reliability: unfolding a tablecloth (deformable-object manipulation), placing a pizza onto the table (pick-and-place), and opening and closing a food container with the lid. Our solution combines VR-based teleoperation and Learning from Demonstrations (LfD) to balance robustness and autonomy. Most subtasks were executed through high-fidelity remote teleoperation, while the pizza placement was handled by an ACT-based policy trained from 100 in-person teleoperated demonstrations with randomized initial configurations. By carefully integrating scoring rules, task characteristics, and current technical capabilities, our approach achieved both high efficiency and reliability, ultimately securing the first place in the competition.
>
---
#### [new 006] Consistent Pose Estimation of Unmanned Ground Vehicles through Terrain-Aided Multi-Sensor Fusion on Geometric Manifolds
- **分类: cs.RO**

- **简介: 该论文旨在提升无人车姿态估计的长期准确性，通过地形辅助的多传感器融合与几何流形滤波，提出M-ESEKF算法，结合地形几何优化传感器数据校正，增强滤波一致性与稳定性。**

- **链接: [http://arxiv.org/pdf/2508.14661v1](http://arxiv.org/pdf/2508.14661v1)**

> **作者:** Alexander Raab; Stephan Weiss; Alessandro Fornasier; Christian Brommer; Abdalrahman Ibrahim
>
> **摘要:** Aiming to enhance the consistency and thus long-term accuracy of Extended Kalman Filters for terrestrial vehicle localization, this paper introduces the Manifold Error State Extended Kalman Filter (M-ESEKF). By representing the robot's pose in a space with reduced dimensionality, the approach ensures feasible estimates on generic smooth surfaces, without introducing artificial constraints or simplifications that may degrade a filter's performance. The accompanying measurement models are compatible with common loosely- and tightly-coupled sensor modalities and also implicitly account for the ground geometry. We extend the formulation by introducing a novel correction scheme that embeds additional domain knowledge into the sensor data, giving more accurate uncertainty approximations and further enhancing filter consistency. The proposed estimator is seamlessly integrated into a validated modular state estimation framework, demonstrating compatibility with existing implementations. Extensive Monte Carlo simulations across diverse scenarios and dynamic sensor configurations show that the M-ESEKF outperforms classical filter formulations in terms of consistency and stability. Moreover, it eliminates the need for scenario-specific parameter tuning, enabling its application in a variety of real-world settings.
>
---
#### [new 007] Fair-CoPlan: Negotiated Flight Planning with Fair Deconfliction for Urban Air Mobility
- **分类: cs.RO**

- **简介: 该论文提出Fair-CoPlan，解决城市空中交通中无人机路径规划的公平性与冲突协调问题。通过半分布式方法，结合运营商与服务提供商协商，优化飞行路径公平性，确保空域安全与参与度。**

- **链接: [http://arxiv.org/pdf/2508.14380v1](http://arxiv.org/pdf/2508.14380v1)**

> **作者:** Nicole Fronda; Phil Smith; Bardh Hoxha; Yash Pant; Houssam Abbas
>
> **备注:** Accepted to IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025
>
> **摘要:** Urban Air Mobility (UAM) is an emerging transportation paradigm in which Uncrewed Aerial Systems (UAS) autonomously transport passengers and goods in cities. The UAS have different operators with different, sometimes competing goals, yet must share the airspace. We propose a negotiated, semi-distributed flight planner that optimizes UAS' flight lengths {\em in a fair manner}. Current flight planners might result in some UAS being given disproportionately shorter flight paths at the expense of others. We introduce Fair-CoPlan, a planner in which operators and a Provider of Service to the UAM (PSU) together compute \emph{fair} flight paths. Fair-CoPlan has three steps: First, the PSU constrains take-off and landing choices for flights based on capacity at and around vertiports. Then, operators plan independently under these constraints. Finally, the PSU resolves any conflicting paths, optimizing for path length fairness. By fairly spreading the cost of deconfliction Fair-CoPlan encourages wider participation in UAM, ensures safety of the airspace and the areas below it, and promotes greater operator flexibility. We demonstrate Fair-CoPlan through simulation experiments and find fairer outcomes than a non-fair planner with minor delays as a trade-off.
>
---
#### [new 008] Efficient Environment Design for Multi-Robot Navigation via Continuous Control
- **分类: cs.RO**

- **简介: 论文研究多机器人连续状态下的导航任务，解决强化学习在现实应用中的样本效率低和训练周期长问题，提出高效可定制的MDP环境，通过多种RL方法测试并验证其在农业场景中的实际应用效果。**

- **链接: [http://arxiv.org/pdf/2508.14105v1](http://arxiv.org/pdf/2508.14105v1)**

> **作者:** Jahid Chowdhury Choton; John Woods; William Hsu
>
> **备注:** 12 pages, 3 figures, conference
>
> **摘要:** Multi-robot navigation and path planning in continuous state and action spaces with uncertain environments remains an open challenge. Deep Reinforcement Learning (RL) is one of the most popular paradigms for solving this task, but its real-world application has been limited due to sample inefficiency and long training periods. Moreover, the existing works using RL for multi-robot navigation lack formal guarantees while designing the environment. In this paper, we introduce an efficient and highly customizable environment for continuous-control multi-robot navigation, where the robots must visit a set of regions of interest (ROIs) by following the shortest paths. The task is formally modeled as a Markov Decision Process (MDP). We describe the multi-robot navigation task as an optimization problem and relate it to finding an optimal policy for the MDP. We crafted several variations of the environment and measured the performance using both gradient and non-gradient based RL methods: A2C, PPO, TRPO, TQC, CrossQ and ARS. To show real-world applicability, we deployed our environment to a 3-D agricultural field with uncertainties using the CoppeliaSim robot simulator and measured the robustness by running inference on the learned models. We believe our work will guide the researchers on how to develop MDP-based environments that are applicable to real-world systems and solve them using the existing state-of-the-art RL methods with limited resources and within reasonable time periods.
>
---
#### [new 009] Task and Motion Planning for Humanoid Loco-manipulation
- **分类: cs.RO**

- **简介: 该论文针对人形机器人运动与操作规划问题，提出基于优化的TAMP框架，通过共享接触模式实现运动与操作统一规划，解决全身动力学及约束下的复杂行为生成。**

- **链接: [http://arxiv.org/pdf/2508.14099v1](http://arxiv.org/pdf/2508.14099v1)**

> **作者:** Michal Ciebielski; Victor Dhédin; Majid Khadiv
>
> **摘要:** This work presents an optimization-based task and motion planning (TAMP) framework that unifies planning for locomotion and manipulation through a shared representation of contact modes. We define symbolic actions as contact mode changes, grounding high-level planning in low-level motion. This enables a unified search that spans task, contact, and motion planning while incorporating whole-body dynamics, as well as all constraints between the robot, the manipulated object, and the environment. Results on a humanoid platform show that our method can generate a broad range of physically consistent loco-manipulation behaviors over long action sequences requiring complex reasoning. To the best of our knowledge, this is the first work that enables the resolution of an integrated TAMP formulation with fully acyclic planning and whole body dynamics with actuation constraints for the humanoid loco-manipulation problem.
>
---
#### [new 010] Research on UAV Applications in Public Administration: Based on an Improved RRT Algorithm
- **分类: cs.RO**

- **简介: 该论文旨在优化UAV在公共管理中的路径规划，解决能耗、避障与空域约束问题，提出改进的dRRT算法，结合目标偏差、动态步长等策略，通过仿真验证其优越性。**

- **链接: [http://arxiv.org/pdf/2508.14096v1](http://arxiv.org/pdf/2508.14096v1)**

> **作者:** Zhanxi Xie; Baili Lu; Yanzhao Gu; Zikun Li; Junhao Wei; Ngai Cheong
>
> **摘要:** This study investigates the application of unmanned aerial vehicles (UAVs) in public management, focusing on optimizing path planning to address challenges such as energy consumption, obstacle avoidance, and airspace constraints. As UAVs transition from 'technical tools' to 'governance infrastructure', driven by advancements in low-altitude economy policies and smart city demands, efficient path planning becomes critical. The research proposes an enhanced Rapidly-exploring Random Tree algorithm (dRRT), incorporating four strategies: Target Bias (to accelerate convergence), Dynamic Step Size (to balance exploration and obstacle navigation), Detour Priority (to prioritize horizontal detours over vertical ascents), and B-spline smoothing (to enhance path smoothness). Simulations in a 500 m3 urban environment with randomized buildings demonstrate dRRT's superiority over traditional RRT, A*, and Ant Colony Optimization (ACO). Results show dRRT achieves a 100\% success rate with an average runtime of 0.01468s, shorter path lengths, fewer waypoints, and smoother trajectories (maximum yaw angles <45{\deg}). Despite improvements, limitations include increased computational overhead from added mechanisms and potential local optima due to goal biasing. The study highlights dRRT's potential for efficient UAV deployment in public management scenarios like emergency response and traffic monitoring, while underscoring the need for integration with real-time obstacle avoidance frameworks. This work contributes to interdisciplinary advancements in urban governance, robotics, and computational optimization.
>
---
#### [new 011] FBI: Learning Dexterous In-hand Manipulation with Dynamic Visuotactile Shortcut Policy
- **分类: cs.RO**

- **简介: 论文针对灵巧抓取操作中的多模态融合与动态适应问题，提出FBI框架，通过动态融合触觉与视觉信息，构建因果关系模型，采用Transformer模块和扩散策略实现高效执行，提升机器人抓取性能。**

- **链接: [http://arxiv.org/pdf/2508.14441v1](http://arxiv.org/pdf/2508.14441v1)**

> **作者:** Yijin Chen; Wenqiang Xu; Zhenjun Yu; Tutian Tang; Yutong Li; Siqiong Yao; Cewu Lu
>
> **摘要:** Dexterous in-hand manipulation is a long-standing challenge in robotics due to complex contact dynamics and partial observability. While humans synergize vision and touch for such tasks, robotic approaches often prioritize one modality, therefore limiting adaptability. This paper introduces Flow Before Imitation (FBI), a visuotactile imitation learning framework that dynamically fuses tactile interactions with visual observations through motion dynamics. Unlike prior static fusion methods, FBI establishes a causal link between tactile signals and object motion via a dynamics-aware latent model. FBI employs a transformer-based interaction module to fuse flow-derived tactile features with visual inputs, training a one-step diffusion policy for real-time execution. Extensive experiments demonstrate that the proposed method outperforms the baseline methods in both simulation and the real world on two customized in-hand manipulation tasks and three standard dexterous manipulation tasks. Code, models, and more results are available in the website https://sites.google.com/view/dex-fbi.
>
---
#### [new 012] SLAM-based Safe Indoor Exploration Strategy
- **分类: cs.RO**

- **简介: 该论文提出基于SLAM的室内安全探索策略，解决移动机器人在障碍环境中安全避障与高效探索的问题，通过融合IMU、LiDAR和视觉数据实现SLAM，并采用安全骨架算法优先保障安全。**

- **链接: [http://arxiv.org/pdf/2508.14235v1](http://arxiv.org/pdf/2508.14235v1)**

> **作者:** Omar Mostafa; Nikolaos Evangeliou; Anthony Tzes
>
> **备注:** 5 pages, 8 figures. Published in the 2025 11th International Conference on Automation, Robotics, and Applications (ICARA)
>
> **摘要:** This paper suggests a 2D exploration strategy for a planar space cluttered with obstacles. Rather than using point robots capable of adjusting their position and altitude instantly, this research is tailored to classical agents with circular footprints that cannot control instantly their pose. Inhere, a self-balanced dual-wheeled differential drive system is used to explore the place. The system is equipped with linear accelerometers and angular gyroscopes, a 3D-LiDAR, and a forward-facing RGB-D camera. The system performs RTAB-SLAM using the IMU and the LiDAR, while the camera is used for loop closures. The mobile agent explores the planar space using a safe skeleton approach that places the agent as far as possible from the static obstacles. During the exploration strategy, the heading is towards any offered openings of the space. This space exploration strategy has as its highest priority the agent's safety in avoiding the obstacles followed by the exploration of undetected space. Experimental studies with a ROS-enabled mobile agent are presented indicating the path planning strategy while exploring the space.
>
---
#### [new 013] FiReFly: Fair Distributed Receding Horizon Planning for Multiple UAVs
- **分类: cs.RO**

- **简介: 论文研究多无人机的公平分布式路径规划，解决资源分配与任务成功率的平衡问题，提出FiReFly算法通过公平能耗分配提升任务成功率，实验证明可支持15架无人机，扩展至50架需权衡效率与公平性。**

- **链接: [http://arxiv.org/pdf/2508.14381v1](http://arxiv.org/pdf/2508.14381v1)**

> **作者:** Nicole Fronda; Bardh Hoxha; Houssam Abbas
>
> **备注:** Accepted to IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025
>
> **摘要:** We propose injecting notions of fairness into multi-robot motion planning. When robots have competing interests, it is important to optimize for some kind of fairness in their usage of resources. In this work, we explore how the robots' energy expenditures might be fairly distributed among them, while maintaining mission success. We formulate a distributed fair motion planner and integrate it with safe controllers in a algorithm called FiReFly. For simulated reach-avoid missions, FiReFly produces fairer trajectories and improves mission success rates over a non-fair planner. We find that real-time performance is achievable up to 15 UAVs, and that scaling up to 50 UAVs is possible with trade-offs between runtime and fairness improvements.
>
---
#### [new 014] Safe and Transparent Robots for Human-in-the-Loop Meat Processing
- **分类: cs.RO**

- **简介: 该论文旨在开发安全透明的通用协作机器人，解决肉加工中自动化系统专用性高、成本贵的问题。通过手部检测、力传感器刀具确保安全，结合不确定性检测与可视化界面提升透明度，实现人机协同加工。**

- **链接: [http://arxiv.org/pdf/2508.14763v1](http://arxiv.org/pdf/2508.14763v1)**

> **作者:** Sagar Parekh; Casey Grothoff; Ryan Wright; Robin White; Dylan P. Losey
>
> **摘要:** Labor shortages have severely affected the meat processing sector. Automated technology has the potential to support the meat industry, assist workers, and enhance job quality. However, existing automation in meat processing is highly specialized, inflexible, and cost intensive. Instead of forcing manufacturers to buy a separate device for each step of the process, our objective is to develop general-purpose robotic systems that work alongside humans to perform multiple meat processing tasks. Through a recently conducted survey of industry experts, we identified two main challenges associated with integrating these collaborative robots alongside human workers. First, there must be measures to ensure the safety of human coworkers; second, the coworkers need to understand what the robot is doing. This paper addresses both challenges by introducing a safety and transparency framework for general-purpose meat processing robots. For safety, we implement a hand-detection system that continuously monitors nearby humans. This system can halt the robot in situations where the human comes into close proximity of the operating robot. We also develop an instrumented knife equipped with a force sensor that can differentiate contact between objects such as meat, bone, or fixtures. For transparency, we introduce a method that detects the robot's uncertainty about its performance and uses an LED interface to communicate that uncertainty to the human. Additionally, we design a graphical interface that displays the robot's plans and allows the human to provide feedback on the planned cut. Overall, our framework can ensure safe operation while keeping human workers in-the-loop about the robot's actions which we validate through a user study.
>
---
#### [new 015] Domain Translation of a Soft Robotic Arm using Conditional Cycle Generative Adversarial Network
- **分类: cs.RO; cs.AI**

- **简介: 该论文通过条件循环GAN实现软机器人跨域控制，解决材料退化导致的物理特性差异问题，验证了模型在不同粘度环境下的轨迹跟踪能力。**

- **链接: [http://arxiv.org/pdf/2508.14100v1](http://arxiv.org/pdf/2508.14100v1)**

> **作者:** Nilay Kushawaha; Carlo Alessi; Lorenzo Fruzzetti; Egidio Falotico
>
> **备注:** Accepted at IEEE International Conference on Robotic Systems and Applications
>
> **摘要:** Deep learning provides a powerful method for modeling the dynamics of soft robots, offering advantages over traditional analytical approaches that require precise knowledge of the robot's structure, material properties, and other physical characteristics. Given the inherent complexity and non-linearity of these systems, extracting such details can be challenging. The mappings learned in one domain cannot be directly transferred to another domain with different physical properties. This challenge is particularly relevant for soft robots, as their materials gradually degrade over time. In this paper, we introduce a domain translation framework based on a conditional cycle generative adversarial network (CCGAN) to enable knowledge transfer from a source domain to a target domain. Specifically, we employ a dynamic learning approach to adapt a pose controller trained in a standard simulation environment to a domain with tenfold increased viscosity. Our model learns from input pressure signals conditioned on corresponding end-effector positions and orientations in both domains. We evaluate our approach through trajectory-tracking experiments across five distinct shapes and further assess its robustness under noise perturbations and periodicity tests. The results demonstrate that CCGAN-GP effectively facilitates cross-domain skill transfer, paving the way for more adaptable and generalizable soft robotic controllers.
>
---
#### [new 016] D$^2$-LIO: Enhanced Optimization for LiDAR-IMU Odometry Considering Directional Degeneracy
- **分类: cs.RO**

- **简介: 该论文提出D²-LIO框架，解决LiDAR-IMU里程计中特征退化导致的状态估计问题。通过自适应异常值去除、扫描到子地图注册及融合IMU协方差的加权矩阵，提升在复杂环境下的鲁棒性与定位精度。**

- **链接: [http://arxiv.org/pdf/2508.14355v1](http://arxiv.org/pdf/2508.14355v1)**

> **作者:** Guodong Yao; Hao Wang; Qing Chang
>
> **备注:** 7 page, 2 figures
>
> **摘要:** LiDAR-inertial odometry (LIO) plays a vital role in achieving accurate localization and mapping, especially in complex environments. However, the presence of LiDAR feature degeneracy poses a major challenge to reliable state estimation. To overcome this issue, we propose an enhanced LIO framework that integrates adaptive outlier-tolerant correspondence with a scan-to-submap registration strategy. The core contribution lies in an adaptive outlier removal threshold, which dynamically adjusts based on point-to-sensor distance and the motion amplitude of platform. This mechanism improves the robustness of feature matching in varying conditions. Moreover, we introduce a flexible scan-to-submap registration method that leverages IMU data to refine pose estimation, particularly in degenerate geometric configurations. To further enhance localization accuracy, we design a novel weighting matrix that fuses IMU preintegration covariance with a degeneration metric derived from the scan-to-submap process. Extensive experiments conducted in both indoor and outdoor environments-characterized by sparse or degenerate features-demonstrate that our method consistently outperforms state-of-the-art approaches in terms of both robustness and accuracy.
>
---
#### [new 017] DEXTER-LLM: Dynamic and Explainable Coordination of Multi-Robot Systems in Unknown Environments via Large Language Models
- **分类: cs.RO**

- **简介: 论文提出DEXTER-LLM框架，解决多机器人系统在未知环境中的动态协调与可解释性问题，通过四模块整合LLM与优化方法，实现任务分解、动态分配及在线适应，实验显示高效完成任务且提升计划质量。**

- **链接: [http://arxiv.org/pdf/2508.14387v1](http://arxiv.org/pdf/2508.14387v1)**

> **作者:** Yuxiao Zhu; Junfeng Chen; Xintong Zhang; Meng Guo; Zhongkui Li
>
> **备注:** submitted to IROS 2025
>
> **摘要:** Online coordination of multi-robot systems in open and unknown environments faces significant challenges, particularly when semantic features detected during operation dynamically trigger new tasks. Recent large language model (LLMs)-based approaches for scene reasoning and planning primarily focus on one-shot, end-to-end solutions in known environments, lacking both dynamic adaptation capabilities for online operation and explainability in the processes of planning. To address these issues, a novel framework (DEXTER-LLM) for dynamic task planning in unknown environments, integrates four modules: (i) a mission comprehension module that resolves partial ordering of tasks specified by natural languages or linear temporal logic formulas (LTL); (ii) an online subtask generator based on LLMs that improves the accuracy and explainability of task decomposition via multi-stage reasoning; (iii) an optimal subtask assigner and scheduler that allocates subtasks to robots via search-based optimization; and (iv) a dynamic adaptation and human-in-the-loop verification module that implements multi-rate, event-based updates for both subtasks and their assignments, to cope with new features and tasks detected online. The framework effectively combines LLMs' open-world reasoning capabilities with the optimality of model-based assignment methods, simultaneously addressing the critical issue of online adaptability and explainability. Experimental evaluations demonstrate exceptional performances, with 100% success rates across all scenarios, 160 tasks and 480 subtasks completed on average (3 times the baselines), 62% less queries to LLMs during adaptation, and superior plan quality (2 times higher) for compound tasks. Project page at https://tcxm.github.io/DEXTER-LLM/
>
---
#### [new 018] Lightweight Tracking Control for Computationally Constrained Aerial Systems with the Newton-Raphson Method
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 该论文针对计算受限的空中系统设计轻量级跟踪控制器，采用牛顿-拉夫森方法，在微型球状飞行器和四旋翼无人机上验证其性能，对比传统控制方法，证明其在跟踪精度、计算效率和能耗方面表现优异。**

- **链接: [http://arxiv.org/pdf/2508.14185v1](http://arxiv.org/pdf/2508.14185v1)**

> **作者:** Evanns Morales-Cuadrado; Luke Baird; Yorai Wardi; Samuel Coogan
>
> **摘要:** We investigate the performance of a lightweight tracking controller, based on a flow version of the Newton-Raphson method, applied to a miniature blimp and a mid-size quadrotor. This tracking technique has been shown to enjoy theoretical guarantees of performance and has been applied with success in simulation studies and on mobile robots with simple motion models. This paper investigates the technique through real-world flight experiments on aerial hardware platforms subject to realistic deployment and onboard computational constraints. The technique's performance is assessed in comparison with the established control frameworks of feedback linearization for the blimp, and nonlinear model predictive control for both quadrotor and blimp. The performance metrics under consideration are (i) root mean square error of flight trajectories with respect to target trajectories, (ii) algorithms' computation times, and (iii) CPU energy consumption associated with the control algorithms. The experimental findings show that the Newton-Raphson flow-based tracking controller achieves comparable or superior tracking performance to the baseline methods with substantially reduced computation time and energy expenditure.
>
---
#### [new 019] Offline Imitation Learning upon Arbitrary Demonstrations by Pre-Training Dynamics Representations
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对离线模仿学习中的数据不足问题，提出通过预训练动力学表示（基于转移动力学分解）提升性能，利用非专家数据缓解数据瓶颈，并在模拟与真实场景中验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.14383v1](http://arxiv.org/pdf/2508.14383v1)**

> **作者:** Haitong Ma; Bo Dai; Zhaolin Ren; Yebin Wang; Na Li
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Limited data has become a major bottleneck in scaling up offline imitation learning (IL). In this paper, we propose enhancing IL performance under limited expert data by introducing a pre-training stage that learns dynamics representations, derived from factorizations of the transition dynamics. We first theoretically justify that the optimal decision variable of offline IL lies in the representation space, significantly reducing the parameters to learn in the downstream IL. Moreover, the dynamics representations can be learned from arbitrary data collected with the same dynamics, allowing the reuse of massive non-expert data and mitigating the limited data issues. We present a tractable loss function inspired by noise contrastive estimation to learn the dynamics representations at the pre-training stage. Experiments on MuJoCo demonstrate that our proposed algorithm can mimic expert policies with as few as a single trajectory. Experiments on real quadrupeds show that we can leverage pre-trained dynamics representations from simulator data to learn to walk from a few real-world demonstrations.
>
---
#### [new 020] Action-Constrained Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 论文研究动作约束下的模仿学习（ACIL），解决专家与模仿者动作空间差异导致的occupancy measure不匹配问题，提出DTWIL方法通过轨迹对齐和MPC生成符合约束的surrogate数据集，提升机器人控制性能。**

- **链接: [http://arxiv.org/pdf/2508.14379v1](http://arxiv.org/pdf/2508.14379v1)**

> **作者:** Chia-Han Yeh; Tse-Sheng Nan; Risto Vuorio; Wei Hung; Hung-Yen Wu; Shao-Hua Sun; Ping-Chun Hsieh
>
> **备注:** Published in ICML 2025
>
> **摘要:** Policy learning under action constraints plays a central role in ensuring safe behaviors in various robot control and resource allocation applications. In this paper, we study a new problem setting termed Action-Constrained Imitation Learning (ACIL), where an action-constrained imitator aims to learn from a demonstrative expert with larger action space. The fundamental challenge of ACIL lies in the unavoidable mismatch of occupancy measure between the expert and the imitator caused by the action constraints. We tackle this mismatch through \textit{trajectory alignment} and propose DTWIL, which replaces the original expert demonstrations with a surrogate dataset that follows similar state trajectories while adhering to the action constraints. Specifically, we recast trajectory alignment as a planning problem and solve it via Model Predictive Control, which aligns the surrogate trajectories with the expert trajectories based on the Dynamic Time Warping (DTW) distance. Through extensive experiments, we demonstrate that learning from the dataset generated by DTWIL significantly enhances performance across multiple robot control tasks and outperforms various benchmark imitation learning algorithms in terms of sample efficiency. Our code is publicly available at https://github.com/NYCU-RL-Bandits-Lab/ACRL-Baselines.
>
---
#### [new 021] SimGenHOI: Physically Realistic Whole-Body Humanoid-Object Interaction via Generative Modeling and Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 论文提出SimGenHOI，旨在生成物理真实的全身体态交互（HOI）。针对现有方法生成的接触异常等问题，结合生成模型与强化学习，设计接触感知控制策略及互细调机制，提升动作真实性和跟踪鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.14120v1](http://arxiv.org/pdf/2508.14120v1)**

> **作者:** Yuhang Lin; Yijia Xie; Jiahong Xie; Yuehao Huang; Ruoyu Wang; Jiajun Lv; Yukai Ma; Xingxing Zuo
>
> **摘要:** Generating physically realistic humanoid-object interactions (HOI) is a fundamental challenge in robotics. Existing HOI generation approaches, such as diffusion-based models, often suffer from artifacts such as implausible contacts, penetrations, and unrealistic whole-body actions, which hinder successful execution in physical environments. To address these challenges, we introduce SimGenHOI, a unified framework that combines the strengths of generative modeling and reinforcement learning to produce controllable and physically plausible HOI. Our HOI generative model, based on Diffusion Transformers (DiT), predicts a set of key actions conditioned on text prompts, object geometry, sparse object waypoints, and the initial humanoid pose. These key actions capture essential interaction dynamics and are interpolated into smooth motion trajectories, naturally supporting long-horizon generation. To ensure physical realism, we design a contact-aware whole-body control policy trained with reinforcement learning, which tracks the generated motions while correcting artifacts such as penetration and foot sliding. Furthermore, we introduce a mutual fine-tuning strategy, where the generative model and the control policy iteratively refine each other, improving both motion realism and tracking robustness. Extensive experiments demonstrate that SimGenHOI generates realistic, diverse, and physically plausible humanoid-object interactions, achieving significantly higher tracking success rates in simulation and enabling long-horizon manipulation tasks. Code will be released upon acceptance on our project page: https://xingxingzuo.github.io/simgen_hoi.
>
---
#### [new 022] Adapting Biological Reflexes for Dynamic Reorientation in Space Manipulator Systems
- **分类: cs.RO; physics.bio-ph**

- **简介: 该论文旨在通过生物反射机制（如蜥蜴空中翻转）优化空间机械臂控制，解决微重力环境下机械臂与航天器基座动态耦合导致的控制难题，提升其机动性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.14258v1](http://arxiv.org/pdf/2508.14258v1)**

> **作者:** Daegyun Choi; Alhim Vera; Donghoon Kim
>
> **备注:** 18 pages, 11 figures, 2025 AAS/AIAA Astrodynamics Specialist Conference
>
> **摘要:** Robotic arms mounted on spacecraft, known as space manipulator systems (SMSs), are critical for enabling on-orbit assembly, satellite servicing, and debris removal. However, controlling these systems in microgravity remains a significant challenge due to the dynamic coupling between the manipulator and the spacecraft base. This study explores the potential of using biological inspiration to address this issue, focusing on animals, particularly lizards, that exhibit mid-air righting reflexes. Based on similarities between SMSs and these animals in terms of behavior, morphology, and environment, their air-righting motion trajectories are extracted from high-speed video recordings using computer vision techniques. These trajectories are analyzed within a multi-objective optimization framework to identify the key behavioral goals and assess their relative importance. The resulting motion profiles are then applied as reference trajectories for SMS control, with baseline controllers used to track them. The findings provide a step toward translating evolved animal behaviors into interpretable, adaptive control strategies for space robotics, with implications for improving maneuverability and robustness in future missions.
>
---
#### [new 023] Can LLM Agents Solve Collaborative Tasks? A Study on Urgency-Aware Planning and Coordination
- **分类: cs.RO; cs.AI**

- **简介: 论文研究LLM代理在结构化救援任务中的协作能力，通过紧急意识规划和协调解决分工与优先级问题，评估其在已知环境中的表现，探索其优势与局限性，为多代理系统优化提供参考。**

- **链接: [http://arxiv.org/pdf/2508.14635v1](http://arxiv.org/pdf/2508.14635v1)**

> **作者:** João Vitor de Carvalho Silva; Douglas G. Macharet
>
> **摘要:** The ability to coordinate actions across multiple agents is critical for solving complex, real-world problems. Large Language Models (LLMs) have shown strong capabilities in communication, planning, and reasoning, raising the question of whether they can also support effective collaboration in multi-agent settings. In this work, we investigate the use of LLM agents to solve a structured victim rescue task that requires division of labor, prioritization, and cooperative planning. Agents operate in a fully known graph-based environment and must allocate resources to victims with varying needs and urgency levels. We systematically evaluate their performance using a suite of coordination-sensitive metrics, including task success rate, redundant actions, room conflicts, and urgency-weighted efficiency. This study offers new insights into the strengths and failure modes of LLMs in physically grounded multi-agent collaboration tasks, contributing to future benchmarks and architectural improvements.
>
---
#### [new 024] Making Pose Representations More Expressive and Disentangled via Residual Vector Quantization
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对文本到动作生成中姿态代码表达不足的问题，提出残差向量量化方法，增强姿态表示的表达性与解耦性，提升运动细节捕捉能力。**

- **链接: [http://arxiv.org/pdf/2508.14561v1](http://arxiv.org/pdf/2508.14561v1)**

> **作者:** Sukhyun Jeong; Hong-Gi Shin; Yong-Hoon Choi
>
> **摘要:** Recent progress in text-to-motion has advanced both 3D human motion generation and text-based motion control. Controllable motion generation (CoMo), which enables intuitive control, typically relies on pose code representations, but discrete pose codes alone cannot capture fine-grained motion details, limiting expressiveness. To overcome this, we propose a method that augments pose code-based latent representations with continuous motion features using residual vector quantization (RVQ). This design preserves the interpretability and manipulability of pose codes while effectively capturing subtle motion characteristics such as high-frequency details. Experiments on the HumanML3D dataset show that our model reduces Frechet inception distance (FID) from 0.041 to 0.015 and improves Top-1 R-Precision from 0.508 to 0.510. Qualitative analysis of pairwise direction similarity between pose codes further confirms the model's controllability for motion editing.
>
---
#### [new 025] Virtual Community: An Open World for Humans, Robots, and Society
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 论文构建开放世界平台Virtual Community，研究人机协作与社会智能。通过模拟人类与机器人共存场景，设计社区规划与机器人协作挑战，评估多智能体在开放环境中的合作与规划能力。**

- **链接: [http://arxiv.org/pdf/2508.14893v1](http://arxiv.org/pdf/2508.14893v1)**

> **作者:** Qinhong Zhou; Hongxin Zhang; Xiangye Lin; Zheyuan Zhang; Yutian Chen; Wenjun Liu; Zunzhe Zhang; Sunli Chen; Lixing Fang; Qiushi Lyu; Xinyu Sun; Jincheng Yang; Zeyuan Wang; Bao Chi Dang; Zhehuan Chen; Daksha Ladia; Jiageng Liu; Chuang Gan
>
> **备注:** website https://virtual-community-ai.github.io/
>
> **摘要:** The rapid progress in AI and Robotics may lead to a profound societal transformation, as humans and robots begin to coexist within shared communities, introducing both opportunities and challenges. To explore this future, we present Virtual Community-an open-world platform for humans, robots, and society-built on a universal physics engine and grounded in real-world 3D scenes. With Virtual Community, we aim to study embodied social intelligence at scale: 1) How robots can intelligently cooperate or compete; 2) How humans develop social relations and build community; 3) More importantly, how intelligent robots and humans can co-exist in an open world. To support these, Virtual Community features: 1) An open-source multi-agent physics simulator that supports robots, humans, and their interactions within a society; 2) A large-scale, real-world aligned community generation pipeline, including vast outdoor space, diverse indoor scenes, and a community of grounded agents with rich characters and appearances. Leveraging Virtual Community, we propose two novel challenges. The Community Planning Challenge evaluates multi-agent reasoning and planning ability in open-world settings, such as cooperating to help agents with daily activities and efficiently connecting other agents. The Community Robot Challenge requires multiple heterogeneous robots to collaborate in solving complex open-world tasks. We evaluate various baselines on these tasks and demonstrate the challenges in both high-level open-world task planning and low-level cooperation controls. We hope that Virtual Community will unlock further study of human-robot coexistence within open-world environments.
>
---
#### [new 026] RynnEC: Bringing MLLMs into Embodied World
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出RynnEC，一种用于具身认知的视频多模态大模型，解决区域级视频交互与数据稀缺问题，通过区域编码器、掩码解码器及自研数据管道提升物体理解与空间推理能力，并构建RynnEC-Bench基准。**

- **链接: [http://arxiv.org/pdf/2508.14160v1](http://arxiv.org/pdf/2508.14160v1)**

> **作者:** Ronghao Dang; Yuqian Yuan; Yunxuan Mao; Kehan Li; Jiangpin Liu; Zhikai Wang; Xin Li; Fan Wang; Deli Zhao
>
> **备注:** The technical report of RynnEC, an embodied cognition MLLM
>
> **摘要:** We introduce RynnEC, a video multimodal large language model designed for embodied cognition. Built upon a general-purpose vision-language foundation model, RynnEC incorporates a region encoder and a mask decoder, enabling flexible region-level video interaction. Despite its compact architecture, RynnEC achieves state-of-the-art performance in object property understanding, object segmentation, and spatial reasoning. Conceptually, it offers a region-centric video paradigm for the brain of embodied agents, providing fine-grained perception of the physical world and enabling more precise interactions. To mitigate the scarcity of annotated 3D datasets, we propose an egocentric video based pipeline for generating embodied cognition data. Furthermore, we introduce RynnEC-Bench, a region-centered benchmark for evaluating embodied cognitive capabilities. We anticipate that RynnEC will advance the development of general-purpose cognitive cores for embodied agents and facilitate generalization across diverse embodied tasks. The code, model checkpoints, and benchmark are available at: https://github.com/alibaba-damo-academy/RynnEC
>
---
#### [new 027] Beyond Fixed Morphologies: Learning Graph Policies with Trust Region Compensation in Variable Action Spaces
- **分类: cs.LG; cs.RO; cs.SY; eess.SY**

- **简介: 论文研究强化学习中信任区域方法在变量动作空间下的策略优化，分析TRPO和PPO的理论表现，并通过Swimmer环境验证形态泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.14102v1](http://arxiv.org/pdf/2508.14102v1)**

> **作者:** Thomas Gallien
>
> **摘要:** Trust region-based optimization methods have become foundational reinforcement learning algorithms that offer stability and strong empirical performance in continuous control tasks. Growing interest in scalable and reusable control policies translate also in a demand for morphological generalization, the ability of control policies to cope with different kinematic structures. Graph-based policy architectures provide a natural and effective mechanism to encode such structural differences. However, while these architectures accommodate variable morphologies, the behavior of trust region methods under varying action space dimensionality remains poorly understood. To this end, we conduct a theoretical analysis of trust region-based policy optimization methods, focusing on both Trust Region Policy Optimization (TRPO) and its widely used first-order approximation, Proximal Policy Optimization (PPO). The goal is to demonstrate how varying action space dimensionality influence the optimization landscape, particularly under the constraints imposed by KL-divergence or policy clipping penalties. Complementing the theoretical insights, an empirical evaluation under morphological variation is carried out using the Gymnasium Swimmer environment. This benchmark offers a systematically controlled setting for varying the kinematic structure without altering the underlying task, making it particularly well-suited to study morphological generalization.
>
---
#### [new 028] Fusing Monocular RGB Images with AIS Data to Create a 6D Pose Estimation Dataset for Marine Vessels
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在通过融合单目RGB图像与AIS数据，解决纯AIS定位的可靠性问题，提出基于YOLOX-X和PnP方法生成无需人工标注的海洋船舶6D姿态估计数据集BONK-pose。**

- **链接: [http://arxiv.org/pdf/2508.14767v1](http://arxiv.org/pdf/2508.14767v1)**

> **作者:** Fabian Holst; Emre Gülsoylu; Simone Frintrop
>
> **备注:** Author version of the submission to the IEEE Journal of Oceanic Engineering
>
> **摘要:** The paper presents a novel technique for creating a 6D pose estimation dataset for marine vessels by fusing monocular RGB images with Automatic Identification System (AIS) data. The proposed technique addresses the limitations of relying purely on AIS for location information, caused by issues like equipment reliability, data manipulation, and transmission delays. By combining vessel detections from monocular RGB images, obtained using an object detection network (YOLOX-X), with AIS messages, the technique generates 3D bounding boxes that represent the vessels' 6D poses, i.e. spatial and rotational dimensions. The paper evaluates different object detection models to locate vessels in image space. We also compare two transformation methods (homography and Perspective-n-Point) for aligning AIS data with image coordinates. The results of our work demonstrate that the Perspective-n-Point (PnP) method achieves a significantly lower projection error compared to homography-based approaches used before, and the YOLOX-X model achieves a mean Average Precision (mAP) of 0.80 at an Intersection over Union (IoU) threshold of 0.5 for relevant vessel classes. We show indication that our approach allows the creation of a 6D pose estimation dataset without needing manual annotation. Additionally, we introduce the Boats on Nordelbe Kehrwieder (BONK-pose), a publicly available dataset comprising 3753 images with 3D bounding box annotations for pose estimation, created by our data fusion approach. This dataset can be used for training and evaluating 6D pose estimation networks. In addition we introduce a set of 1000 images with 2D bounding box annotations for ship detection from the same scene.
>
---
#### [new 029] Towards Unified Probabilistic Verification and Validation of Vision-Based Autonomy
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 论文提出统一视觉自主系统概率验证与验证方法，通过区间MDPs解决传统严格假设限制，适应不确定环境，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.14181v1](http://arxiv.org/pdf/2508.14181v1)**

> **作者:** Jordan Peper; Yan Miao; Sayan Mitra; Ivan Ruchkin
>
> **备注:** Accepted by the 23rd International Symposium on Automated Technology for Verification and Analysis (ATVA'25)
>
> **摘要:** Precise and comprehensive situational awareness is a critical capability of modern autonomous systems. Deep neural networks that perceive task-critical details from rich sensory signals have become ubiquitous; however, their black-box behavior and sensitivity to environmental uncertainty and distribution shifts make them challenging to verify formally. Abstraction-based verification techniques for vision-based autonomy produce safety guarantees contingent on rigid assumptions, such as bounded errors or known unique distributions. Such overly restrictive and inflexible assumptions limit the validity of the guarantees, especially in diverse and uncertain test-time environments. We propose a methodology that unifies the verification models of perception with their offline validation. Our methodology leverages interval MDPs and provides a flexible end-to-end guarantee that adapts directly to the out-of-distribution test-time conditions. We evaluate our methodology on a synthetic perception Markov chain with well-defined state estimation distributions and a mountain car benchmark. Our findings reveal that we can guarantee tight yet rigorous bounds on overall system safety.
>
---
#### [new 030] Learning Point Cloud Representations with Pose Continuity for Depth-Based Category-Level 6D Object Pose Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文针对类别级6D姿态估计中姿态不连续问题，提出HRC-Pose框架，通过对比学习学习点云表示，分解姿态为旋转和翻译，提升泛化能力和实时性能。**

- **链接: [http://arxiv.org/pdf/2508.14358v1](http://arxiv.org/pdf/2508.14358v1)**

> **作者:** Zhujun Li; Shuo Zhang; Ioannis Stamos
>
> **备注:** Accepted by ICCV 2025 Workshop on Recovering 6D Object Pose (R6D)
>
> **摘要:** Category-level object pose estimation aims to predict the 6D pose and 3D size of objects within given categories. Existing approaches for this task rely solely on 6D poses as supervisory signals without explicitly capturing the intrinsic continuity of poses, leading to inconsistencies in predictions and reduced generalization to unseen poses. To address this limitation, we propose HRC-Pose, a novel depth-only framework for category-level object pose estimation, which leverages contrastive learning to learn point cloud representations that preserve the continuity of 6D poses. HRC-Pose decouples object pose into rotation and translation components, which are separately encoded and leveraged throughout the network. Specifically, we introduce a contrastive learning strategy for multi-task, multi-category scenarios based on our 6D pose-aware hierarchical ranking scheme, which contrasts point clouds from multiple categories by considering rotational and translational differences as well as categorical information. We further design pose estimation modules that separately process the learned rotation-aware and translation-aware embeddings. Our experiments demonstrate that HRC-Pose successfully learns continuous feature spaces. Results on REAL275 and CAMERA25 benchmarks show that our method consistently outperforms existing depth-only state-of-the-art methods and runs in real-time, demonstrating its effectiveness and potential for real-world applications. Our code is at https://github.com/zhujunli1993/HRC-Pose.
>
---
## 更新

#### [replaced 001] Active Disturbance Rejection Control for Trajectory Tracking of a Seagoing USV: Design, Simulation, and Field Experiments
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.21265v2](http://arxiv.org/pdf/2506.21265v2)**

> **作者:** Jelmer van der Saag; Elia Trevisan; Wouter Falkena; Javier Alonso-Mora
>
> **备注:** Accepted for presentation at IROS 2025. Accepted version
>
> **摘要:** Unmanned Surface Vessels (USVs) face significant control challenges due to uncertain environmental disturbances like waves and currents. This paper proposes a trajectory tracking controller based on Active Disturbance Rejection Control (ADRC) implemented on the DUS V2500. A custom simulation incorporating realistic waves and current disturbances is developed to validate the controller's performance, supported by further validation through field tests in the harbour of Scheveningen, the Netherlands, and at sea. Simulation results demonstrate that ADRC significantly reduces cross-track error across all tested conditions compared to a baseline PID controller but increases control effort and energy consumption. Field trials confirm these findings while revealing a further increase in energy consumption during sea trials compared to the baseline.
>
---
#### [replaced 002] SDS -- See it, Do it, Sorted: Quadruped Skill Synthesis from Single Video Demonstration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.11571v2](http://arxiv.org/pdf/2410.11571v2)**

> **作者:** Maria Stamatopoulou; Jeffrey Li; Dimitrios Kanoulas
>
> **摘要:** Imagine a robot learning locomotion skills from any single video, without labels or reward engineering. We introduce SDS ("See it. Do it. Sorted."), an automated pipeline for skill acquisition from unstructured demonstrations. Using GPT-4o, SDS applies novel prompting techniques, in the form of spatio-temporal grid-based visual encoding ($G_{v}$) and structured input decomposition (SUS). These produce executable reward functions (RF) from the raw input videos. The RFs are used to train PPO policies and are optimized through closed-loop evolution, using training footage and performance metrics as self-supervised signals. SDS allows quadrupeds (e.g. Unitree Go1) to learn four gaits -- trot, bound, pace, and hop -- achieving 100% gait matching fidelity, Dynamic Time Warping (DTW) distance in the order of $10^{-6}$, and stable locomotion with zero failures, both in simulation and the real world. SDS generalizes to morphologically different quadrupeds (e.g. ANYmal) and outperforms prior work in data efficiency, training time and engineering effort. Further materials and the code are open-source under: https://rpl-cs-ucl.github.io/SDSweb/.
>
---
#### [replaced 003] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00288v2](http://arxiv.org/pdf/2508.00288v2)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Gua; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [replaced 004] 3D FlowMatch Actor: Unified 3D Policy for Single- and Dual-Arm Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.11002v2](http://arxiv.org/pdf/2508.11002v2)**

> **作者:** Nikolaos Gkanatsios; Jiahe Xu; Matthew Bronars; Arsalan Mousavian; Tsung-Wei Ke; Katerina Fragkiadaki
>
> **备注:** Project page: https://3d-flowmatch-actor.github.io/
>
> **摘要:** We present 3D FlowMatch Actor (3DFA), a 3D policy architecture for robot manipulation that combines flow matching for trajectory prediction with 3D pretrained visual scene representations for learning from demonstration. 3DFA leverages 3D relative attention between action and visual tokens during action denoising, building on prior work in 3D diffusion-based single-arm policy learning. Through a combination of flow matching and targeted system-level and architectural optimizations, 3DFA achieves over 30x faster training and inference than previous 3D diffusion-based policies, without sacrificing performance. On the bimanual PerAct2 benchmark, it establishes a new state of the art, outperforming the next-best method by an absolute margin of 41.4%. In extensive real-world evaluations, it surpasses strong baselines with up to 1000x more parameters and significantly more pretraining. In unimanual settings, it sets a new state of the art on 74 RLBench tasks by directly predicting dense end-effector trajectories, eliminating the need for motion planning. Comprehensive ablation studies underscore the importance of our design choices for both policy effectiveness and efficiency.
>
---
#### [replaced 005] Accelerating Signal-Temporal-Logic-Based Task and Motion Planning of Bipedal Navigation using Benders Decomposition
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13407v2](http://arxiv.org/pdf/2508.13407v2)**

> **作者:** Jiming Ren; Xuan Lin; Roman Mineyev; Karen M. Feigh; Samuel Coogan; Ye Zhao
>
> **备注:** 16 pages, 7 figures, 6 tables
>
> **摘要:** Task and motion planning under Signal Temporal Logic constraints is known to be NP-hard. A common class of approaches formulates these hybrid problems, which involve discrete task scheduling and continuous motion planning, as mixed-integer programs (MIP). However, in applications for bipedal locomotion, introduction of non-convex constraints such as kinematic reachability and footstep rotation exacerbates the computational complexity of MIPs. In this work, we present a method based on Benders Decomposition to address scenarios where solving the entire monolithic optimization problem is prohibitively intractable. Benders Decomposition proposes an iterative cutting-plane technique that partitions the problem into a master problem to prototype a plan that meets the task specification, and a series of subproblems for kinematics and dynamics feasibility checks. Our experiments demonstrate that this method achieves faster planning compared to alternative algorithms for solving the resulting optimization program with nonlinear constraints.
>
---
#### [replaced 006] Multi-Robot Navigation in Social Mini-Games: Definitions, Taxonomy, and Algorithms
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.13459v2](http://arxiv.org/pdf/2508.13459v2)**

> **作者:** Rohan Chandra; Shubham Singh; Abhishek Jha; Dannon Andrade; Hriday Sainathuni; Katia Sycara
>
> **摘要:** The ``Last Mile Challenge'' has long been considered an important, yet unsolved, challenge for autonomous vehicles, public service robots, and delivery robots. A central issue in this challenge is the ability of robots to navigate constrained and cluttered environments that have high agency (e.g., doorways, hallways, corridor intersections), often while competing for space with other robots and humans. We refer to these environments as ``Social Mini-Games'' (SMGs). Traditional navigation approaches designed for MRN do not perform well in SMGs, which has led to focused research on dedicated SMG solvers. However, publications on SMG navigation research make different assumptions (on centralized versus decentralized, observability, communication, cooperation, etc.), and have different objective functions (safety versus liveness). These assumptions and objectives are sometimes implicitly assumed or described informally. This makes it difficult to establish appropriate baselines for comparison in research papers, as well as making it difficult for practitioners to find the papers relevant to their concrete application. Such ad-hoc representation of the field also presents a barrier to new researchers wanting to start research in this area. SMG navigation research requires its own taxonomy, definitions, and evaluation protocols to guide effective research moving forward. This survey is the first to catalog SMG solvers using a well-defined and unified taxonomy and to classify existing methods accordingly. It also discusses the essential properties of SMG solvers, defines what SMGs are and how they appear in practice, outlines how to evaluate SMG solvers, and highlights the differences between SMG solvers and general navigation systems. The survey concludes with an overview of future directions and open challenges in the field.
>
---
#### [replaced 007] Into the Wild: When Robots Are Not Welcome
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.12075v2](http://arxiv.org/pdf/2508.12075v2)**

> **作者:** Shaul Ashkenazi; Gabriel Skantze; Jane Stuart-Smith; Mary Ellen Foster
>
> **备注:** Accepted at the workshop on Real-World HRI in Public and Private Spaces: Successes, Failures, and Lessons Learned (PubRob-Fails), held at the IEEE RO-MAN Conference, 2025. 3 pages
>
> **摘要:** Social robots are increasingly being deployed in public spaces, where they face not only technological difficulties and unexpected user utterances, but also objections from stakeholders who may not be comfortable with introducing a robot into those spaces. We describe our difficulties with deploying a social robot in two different public settings: 1) Student services center; 2) Refugees and asylum seekers drop-in service. Although this is a failure report, in each use case we eventually managed to earn the trust of the staff and form a relationship with them, allowing us to deploy our robot and conduct our studies.
>
---
#### [replaced 008] LGR2: Language Guided Reward Relabeling for Accelerating Hierarchical Reinforcement Learning
- **分类: cs.LG; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.05881v5](http://arxiv.org/pdf/2406.05881v5)**

> **作者:** Utsav Singh; Pramit Bhattacharyya; Vinay P. Namboodiri
>
> **摘要:** Large language models (LLMs) have shown remarkable abilities in logical reasoning, in-context learning, and code generation. However, translating natural language instructions into effective robotic control policies remains a significant challenge, especially for tasks requiring long-horizon planning and operating under sparse reward conditions. Hierarchical Reinforcement Learning (HRL) provides a natural framework to address this challenge in robotics; however, it typically suffers from non-stationarity caused by the changing behavior of the lower-level policy during training, destabilizing higher-level policy learning. We introduce LGR2, a novel HRL framework that leverages LLMs to generate language-guided reward functions for the higher-level policy. By decoupling high-level reward generation from low-level policy changes, LGR2 fundamentally mitigates the non-stationarity problem in off-policy HRL, enabling stable and efficient learning. To further enhance sample efficiency in sparse environments, we integrate goal-conditioned hindsight experience relabeling. Extensive experiments across simulated and real-world robotic navigation and manipulation tasks demonstrate LGR2 outperforms both hierarchical and non-hierarchical baselines, achieving over 55% success rates on challenging tasks and robust transfer to real robots, without additional fine-tuning.
>
---
#### [replaced 009] MinD: Learning A Dual-System World Model for Real-Time Planning and Implicit Risk Analysis
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.18897v2](http://arxiv.org/pdf/2506.18897v2)**

> **作者:** Xiaowei Chi; Kuangzhi Ge; Jiaming Liu; Siyuan Zhou; Peidong Jia; Zichen He; Yuzhen Liu; Tingguang Li; Lei Han; Sirui Han; Shanghang Zhang; Yike Guo
>
> **摘要:** Video Generation Models (VGMs) have become powerful backbones for Vision-Language-Action (VLA) models, leveraging large-scale pretraining for robust dynamics modeling. However, current methods underutilize their distribution modeling capabilities for predicting future states. Two challenges hinder progress: integrating generative processes into feature learning is both technically and conceptually underdeveloped, and naive frame-by-frame video diffusion is computationally inefficient for real-time robotics. To address these, we propose Manipulate in Dream (MinD), a dual-system world model for real-time, risk-aware planning. MinD uses two asynchronous diffusion processes: a low-frequency visual generator (LoDiff) that predicts future scenes and a high-frequency diffusion policy (HiDiff) that outputs actions. Our key insight is that robotic policies do not require fully denoised frames but can rely on low-resolution latents generated in a single denoising step. To connect early predictions to actions, we introduce DiffMatcher, a video-action alignment module with a novel co-training strategy that synchronizes the two diffusion models. MinD achieves a 63% success rate on RL-Bench, 60% on real-world Franka tasks, and operates at 11.3 FPS, demonstrating the efficiency of single-step latent features for control signals. Furthermore, MinD identifies 74% of potential task failures in advance, providing real-time safety signals for monitoring and intervention. This work establishes a new paradigm for efficient and reliable robotic manipulation using generative world models.
>
---
#### [replaced 010] Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.21205v2](http://arxiv.org/pdf/2506.21205v2)**

> **作者:** Elia Trevisan; Khaled A. Mustafa; Godert Notten; Xinwei Wang; Javier Alonso-Mora
>
> **备注:** Accepted for presentation at IROS 2025. Accepted Version
>
> **摘要:** Deploying mobile robots safely among humans requires the motion planner to account for the uncertainty in the other agents' predicted trajectories. This remains challenging in traditional approaches, especially with arbitrarily shaped predictions and real-time constraints. To address these challenges, we propose a Dynamic Risk-Aware Model Predictive Path Integral control (DRA-MPPI), a motion planner that incorporates uncertain future motions modelled with potentially non-Gaussian stochastic predictions. By leveraging MPPI's gradient-free nature, we propose a method that efficiently approximates the joint Collision Probability (CP) among multiple dynamic obstacles for several hundred sampled trajectories in real-time via a Monte Carlo (MC) approach. This enables the rejection of samples exceeding a predefined CP threshold or the integration of CP as a weighted objective within the navigation cost function. Consequently, DRA-MPPI mitigates the freezing robot problem while enhancing safety. Real-world and simulated experiments with multiple dynamic obstacles demonstrate DRA-MPPI's superior performance compared to state-of-the-art approaches, including Scenario-based Model Predictive Control (S-MPC), Frenet planner, and vanilla MPPI.
>
---
#### [replaced 011] Robust simultaneous UWB-anchor calibration and robot localization for emergency situations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.22272v2](http://arxiv.org/pdf/2503.22272v2)**

> **作者:** Xinghua Liu; Ming Cao
>
> **备注:** Submit to IEEE SMC 2025. This work has been submitted to the IEEE for possible publication
>
> **摘要:** In this work, we propose a factor graph optimization (FGO) framework to simultaneously solve the calibration problem for Ultra-WideBand (UWB) anchors and the robot localization problem. Calibrating UWB anchors manually can be time-consuming and even impossible in emergencies or those situations without special calibration tools. Therefore, automatic estimation of the anchor positions becomes a necessity. The proposed method enables the creation of a soft sensor providing the position information of the anchors in a UWB network. This soft sensor requires only UWB and LiDAR measurements measured from a moving robot. The proposed FGO framework is suitable for the calibration of an extendable large UWB network. Moreover, the anchor calibration problem and robot localization problem can be solved simultaneously, which saves time for UWB network deployment. The proposed framework also helps to avoid artificial errors in the UWB-anchor position estimation and improves the accuracy and robustness of the robot-pose. The experimental results of the robot localization using LiDAR and a UWB network in a 3D environment are discussed, demonstrating the performance of the proposed method. More specifically, the anchor calibration problem with four anchors and the robot localization problem can be solved simultaneously and automatically within 30 seconds by the proposed framework. The supplementary video and codes can be accessed via https://github.com/LiuxhRobotAI/Simultaneous_calibration_localization.
>
---
#### [replaced 012] LaViPlan : Language-Guided Visual Path Planning with RLVR
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12911v4](http://arxiv.org/pdf/2507.12911v4)**

> **作者:** Hayeon Oh
>
> **备注:** Accepted to the 2nd ICCV 2025 Workshop on the Challenge of Out-of-Label Hazards in Autonomous Driving (13 pages, 6 figures)
>
> **摘要:** Out-of-distribution (OOD) scenarios in autonomous driving pose critical challenges, as planners often fail to generalize beyond their training experience, leading to unsafe or unexpected behavior. Vision-Language Models (VLMs) have shown promise in handling such scenarios by providing high-level scene understanding and user-aligned decisions. However, existing VLMs often exhibit a misalignment between their language-based reasoning and the low-level trajectories required for action-level planning. In this paper, we propose LaViPlan, a framework that leverages Reinforcement Learning with Verifiable Rewards (RLVR) to fine-tune VLMs using planning-oriented metrics. Experimental results show that LaViPlan improves planning performance across both in-domain and out-of-domain datasets. While linguistic fidelity slightly decreases after RLVR-based fine-tuning, qualitative evaluation indicates that the outputs remain coherent. We also conduct ablation studies to analyze the effects of sampling ratio and reasoning guidance, highlighting how these design choices influence performance. These findings demonstrate the potential of RLVR as a post-training paradigm for aligning language-guided reasoning with action-level planning in autonomous driving.
>
---
#### [replaced 013] Extremum Flow Matching for Offline Goal Conditioned Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19717v2](http://arxiv.org/pdf/2505.19717v2)**

> **作者:** Quentin Rouxel; Clemente Donoso; Fei Chen; Serena Ivaldi; Jean-Baptiste Mouret
>
> **备注:** 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids), Sep 2025, Seoul, South Korea
>
> **摘要:** Imitation learning is a promising approach for enabling generalist capabilities in humanoid robots, but its scaling is fundamentally constrained by the scarcity of high-quality expert demonstrations. This limitation can be mitigated by leveraging suboptimal, open-ended play data, often easier to collect and offering greater diversity. This work builds upon recent advances in generative modeling, specifically Flow Matching, an alternative to Diffusion models. We introduce a method for estimating the minimum or maximum of the learned distribution by leveraging the unique properties of Flow Matching, namely, deterministic transport and support for arbitrary source distributions. We apply this method to develop several goal-conditioned imitation and reinforcement learning algorithms based on Flow Matching, where policies are conditioned on both current and goal observations. We explore and compare different architectural configurations by combining core components, such as critic, planner, actor, or world model, in various ways. We evaluated our agents on the OGBench benchmark and analyzed how different demonstration behaviors during data collection affect performance in a 2D non-prehensile pushing task. Furthermore, we validated our approach on real hardware by deploying it on the Talos humanoid robot to perform complex manipulation tasks based on high-dimensional image observations, featuring a sequence of pick-and-place and articulated object manipulation in a realistic kitchen environment. Experimental videos and code are available at: https://hucebot.github.io/extremum_flow_matching_website/
>
---
#### [replaced 014] MetAdv: A Unified and Interactive Adversarial Testing Platform for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06534v2](http://arxiv.org/pdf/2508.06534v2)**

> **作者:** Aishan Liu; Jiakai Wang; Tianyuan Zhang; Hainan Li; Jiangfan Liu; Siyuan Liang; Yilong Ren; Xianglong Liu; Dacheng Tao
>
> **备注:** Accepted by ACM MM 2025 Demo/Videos track
>
> **摘要:** Evaluating and ensuring the adversarial robustness of autonomous driving (AD) systems is a critical and unresolved challenge. This paper introduces MetAdv, a novel adversarial testing platform that enables realistic, dynamic, and interactive evaluation by tightly integrating virtual simulation with physical vehicle feedback. At its core, MetAdv establishes a hybrid virtual-physical sandbox, within which we design a three-layer closed-loop testing environment with dynamic adversarial test evolution. This architecture facilitates end-to-end adversarial evaluation, ranging from high-level unified adversarial generation, through mid-level simulation-based interaction, to low-level execution on physical vehicles. Additionally, MetAdv supports a broad spectrum of AD tasks, algorithmic paradigms (e.g., modular deep learning pipelines, end-to-end learning, vision-language models). It supports flexible 3D vehicle modeling and seamless transitions between simulated and physical environments, with built-in compatibility for commercial platforms such as Apollo and Tesla. A key feature of MetAdv is its human-in-the-loop capability: besides flexible environmental configuration for more customized evaluation, it enables real-time capture of physiological signals and behavioral feedback from drivers, offering new insights into human-machine trust under adversarial conditions. We believe MetAdv can offer a scalable and unified framework for adversarial assessment, paving the way for safer AD.
>
---
#### [replaced 015] Gaussian-LIC: Real-Time Photo-Realistic SLAM with Gaussian Splatting and LiDAR-Inertial-Camera Fusion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.06926v3](http://arxiv.org/pdf/2404.06926v3)**

> **作者:** Xiaolei Lang; Laijian Li; Chenming Wu; Chen Zhao; Lina Liu; Yong Liu; Jiajun Lv; Xingxing Zuo
>
> **备注:** ICRA 2025
>
> **摘要:** In this paper, we present a real-time photo-realistic SLAM method based on marrying Gaussian Splatting with LiDAR-Inertial-Camera SLAM. Most existing radiance-field-based SLAM systems mainly focus on bounded indoor environments, equipped with RGB-D or RGB sensors. However, they are prone to decline when expanding to unbounded scenes or encountering adverse conditions, such as violent motions and changing illumination. In contrast, oriented to general scenarios, our approach additionally tightly fuses LiDAR, IMU, and camera for robust pose estimation and photo-realistic online mapping. To compensate for regions unobserved by the LiDAR, we propose to integrate both the triangulated visual points from images and LiDAR points for initializing 3D Gaussians. In addition, the modeling of the sky and varying camera exposure have been realized for high-quality rendering. Notably, we implement our system purely with C++ and CUDA, and meticulously design a series of strategies to accelerate the online optimization of the Gaussian-based scene representation. Extensive experiments demonstrate that our method outperforms its counterparts while maintaining real-time capability. Impressively, regarding photo-realistic mapping, our method with our estimated poses even surpasses all the compared approaches that utilize privileged ground-truth poses for mapping. Our code has been released on https://github.com/APRIL-ZJU/Gaussian-LIC.
>
---
#### [replaced 016] Hybrid Action Based Reinforcement Learning for Multi-Objective Compatible Autonomous Driving
- **分类: cs.RO; cs.AI; cs.ET; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.08096v3](http://arxiv.org/pdf/2501.08096v3)**

> **作者:** Guizhe Jin; Zhuoren Li; Bo Leng; Wei Han; Lu Xiong; Chen Sun
>
> **备注:** 13 pages, 10 figures, 5 tables, Submitted to IEEE T-NNLS (under review, 2nd round)
>
> **摘要:** Reinforcement Learning (RL) has shown excellent performance in solving decision-making and control problems of autonomous driving, which is increasingly applied in diverse driving scenarios. However, driving is a multi-attribute problem, leading to challenges in achieving multi-objective compatibility for current RL methods, especially in both policy updating and policy execution. On the one hand, a single value evaluation network limits the policy updating in complex scenarios with coupled driving objectives. On the other hand, the common single-type action space structure limits driving flexibility or results in large behavior fluctuations during policy execution. To this end, we propose a Multi-objective Ensemble-Critic reinforcement learning method with Hybrid Parametrized Action for multi-objective compatible autonomous driving. Specifically, an advanced MORL architecture is constructed, in which the ensemble-critic focuses on different objectives through independent reward functions. The architecture integrates a hybrid parameterized action space structure, and the generated driving actions contain both abstract guidance that matches the hybrid road modality and concrete control commands. Additionally, an uncertainty-based exploration mechanism that supports hybrid actions is developed to learn multi-objective compatible policies more quickly. Experimental results demonstrate that, in both simulator-based and HighD dataset-based multi-lane highway scenarios, our method efficiently learns multi-objective compatible autonomous driving with respect to efficiency, action consistency, and safety.
>
---
#### [replaced 017] From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems
- **分类: cs.CY; cs.CE; cs.CL; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04996v4](http://arxiv.org/pdf/2507.04996v4)**

> **作者:** Jiangbo Yu
>
> **摘要:** Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are viewed as vehicular systems capable of perceiving their environment and executing pre-programmed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 0 to 5); Examples of this outpace include the interaction with humans with natural language, goal adaptation, contextual reasoning, external tool use, and unseen ethical dilemma handling, largely empowered by multi-modal large language models (LLMs). These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this gap, this paper introduces the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI systems to reason, adapt, and interact within complex environments. This paper proposes the term AgVs and their distinguishing characteristics from conventional AuVs. It synthesizes relevant advances in integrating LLMs and AuVs and highlights how AgVs might transform future mobility systems and ensure the systems are human-centered. The paper concludes by identifying key challenges in the development and governance of AgVs, and how they can play a significant role in future agentic transportation systems.
>
---
