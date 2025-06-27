# 机器人 cs.RO

- **最新发布 34 篇**

- **更新 16 篇**

## 最新发布

#### [new 001] Dynamic Risk-Aware MPPI for Mobile Robots in Crowds via Efficient Monte Carlo Approximations
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决在人群中的安全导航问题。通过引入动态风险感知的MPPI方法，提升对不确定轨迹的处理能力，增强机器人安全性与实时性。**

- **链接: [http://arxiv.org/pdf/2506.21205v1](http://arxiv.org/pdf/2506.21205v1)**

> **作者:** Elia Trevisan; Khaled A. Mustafa; Godert Notten; Xinwei Wang; Javier Alonso-Mora
>
> **备注:** Accepted for presentation at IROS 2025. Submitted Version
>
> **摘要:** Deploying mobile robots safely among humans requires the motion planner to account for the uncertainty in the other agents' predicted trajectories. This remains challenging in traditional approaches, especially with arbitrarily shaped predictions and real-time constraints. To address these challenges, we propose a Dynamic Risk-Aware Model Predictive Path Integral control (DRA-MPPI), a motion planner that incorporates uncertain future motions modelled with potentially non-Gaussian stochastic predictions. By leveraging MPPI's gradient-free nature, we propose a method that efficiently approximates the joint Collision Probability (CP) among multiple dynamic obstacles for several hundred sampled trajectories in real-time via a Monte Carlo (MC) approach. This enables the rejection of samples exceeding a predefined CP threshold or the integration of CP as a weighted objective within the navigation cost function. Consequently, DRA-MPPI mitigates the freezing robot problem while enhancing safety. Real-world and simulated experiments with multiple dynamic obstacles demonstrate DRA-MPPI's superior performance compared to state-of-the-art approaches, including Scenario-based Model Predictive Control (S-MPC), Frenet planner, and vanilla MPPI.
>
---
#### [new 002] Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作模型的后训练研究，旨在提升模型在具体任务中的表现。通过类比人类运动学习，提出后训练策略以增强环境感知、身体意识和任务理解。**

- **链接: [http://arxiv.org/pdf/2506.20966v1](http://arxiv.org/pdf/2506.20966v1)**

> **作者:** Tian-Yu Xiang; Ao-Qun Jin; Xiao-Hu Zhou; Mei-Jiang Gui; Xiao-Liang Xie; Shi-Qi Liu; Shuang-Yi Wang; Sheng-Bin Duan; Fu-Chao Xie; Wen-Kai Wang; Si-Cheng Wang; Ling-Yun Li; Tian Tu; Zeng-Guang Hou
>
> **摘要:** Vision-language-action (VLA) models extend vision-language models (VLM) by integrating action generation modules for robotic manipulation. Leveraging strengths of VLM in vision perception and instruction understanding, VLA models exhibit promising generalization across diverse manipulation tasks. However, applications demanding high precision and accuracy reveal performance gaps without further adaptation. Evidence from multiple domains highlights the critical role of post-training to align foundational models with downstream applications, spurring extensive research on post-training VLA models. VLA model post-training aims to address the challenge of improving an embodiment's ability to interact with the environment for the given tasks, analogous to the process of humans motor skills acquisition. Accordingly, this paper reviews post-training strategies for VLA models through the lens of human motor learning, focusing on three dimensions: environments, embodiments, and tasks. A structured taxonomy is introduced aligned with human learning mechanisms: (1) enhancing environmental perception, (2) improving embodiment awareness, (3) deepening task comprehension, and (4) multi-component integration. Finally, key challenges and trends in post-training VLA models are identified, establishing a conceptual framework to guide future research. This work delivers both a comprehensive overview of current VLA model post-training methods from a human motor learning perspective and practical insights for VLA model development. (Project website: https://github.com/AoqunJin/Awesome-VLA-Post-Training)
>
---
#### [new 003] ACTLLM: Action Consistency Tuned Large Language Model
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决动态环境中视觉表示与空间推理不足的问题。通过语言构建场景描述符并引入动作一致性约束，提升机器人任务执行能力。**

- **链接: [http://arxiv.org/pdf/2506.21250v1](http://arxiv.org/pdf/2506.21250v1)**

> **作者:** Jing Bi; Lianggong Bruce Wen; Zhang Liu; Chenliang Xu
>
> **摘要:** This paper introduces ACTLLM (Action Consistency Tuned Large Language Model), a novel approach for robot manipulation in dynamic environments. Traditional vision-based systems often struggle to learn visual representations that excel in both task execution and spatial reasoning, thereby limiting their adaptability in dynamic environments. ACTLLM addresses these challenges by harnessing language to craft structured scene descriptors, providing a uniform interface for both spatial understanding and task performance through flexible language instructions. Moreover, we introduce a novel action consistency constraint that aligns visual perception with corresponding actions, thereby enhancing the learning of actionable visual representations. Additionally, we have reformulated the Markov decision process for manipulation tasks into a multi-turn visual dialogue framework. This approach enables the modeling of long-term task execution with enhanced contextual relevance derived from the history of task execution. During our evaluation, ACTLLM excels in diverse scenarios, proving its effectiveness on challenging vision-based robot manipulation tasks.
>
---
#### [new 004] Model-Based Real-Time Pose and Sag Estimation of Overhead Power Lines Using LiDAR for Drone Inspection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于电力线路巡检任务，解决无人机定位与导线姿态估计问题，通过LiDAR和几何模型实现快速准确的导线状态估计。**

- **链接: [http://arxiv.org/pdf/2506.20812v1](http://arxiv.org/pdf/2506.20812v1)**

> **作者:** Alexandre Girard; Steven A. Parkison; Philippe Hamelin
>
> **备注:** Submitted to IEEE case 2025
>
> **摘要:** Drones can inspect overhead power lines while they remain energized, significantly simplifying the inspection process. However, localizing a drone relative to all conductors using an onboard LiDAR sensor presents several challenges: (1) conductors provide minimal surface for LiDAR beams limiting the number of conductor points in a scan, (2) not all conductors are consistently detected, and (3) distinguishing LiDAR points corresponding to conductors from other objects, such as trees and pylons, is difficult. This paper proposes an estimation approach that minimizes the error between LiDAR measurements and a single geometric model representing the entire conductor array, rather than tracking individual conductors separately. Experimental results, using data from a power line drone inspection, demonstrate that this method achieves accurate tracking, with a solver converging under 50 ms per frame, even in the presence of partial observations, noise, and outliers. A sensitivity analysis shows that the estimation approach can tolerate up to twice as many outlier points as valid conductors measurements.
>
---
#### [new 005] IMA-Catcher: An IMpact-Aware Nonprehensile Catching Framework based on Combined Optimization and Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决高冲击力导致的抓取失败和硬件损坏问题。通过优化与学习结合的方法，在抓取前后阶段降低冲击力并稳定物体。**

- **链接: [http://arxiv.org/pdf/2506.20801v1](http://arxiv.org/pdf/2506.20801v1)**

> **作者:** Francesco Tassi; Jianzhuang Zhao; Gustavo J. G. Lahr; Luna Gava; Marco Monforte; Arren Glover; Chiara Bartolozzi; Arash Ajoudani
>
> **备注:** 25 pages, 17 figures, accepted by International Journal of Robotics Research (IJRR)
>
> **摘要:** Robotic catching of flying objects typically generates high impact forces that might lead to task failure and potential hardware damages. This is accentuated when the object mass to robot payload ratio increases, given the strong inertial components characterizing this task. This paper aims to address this problem by proposing an implicitly impact-aware framework that accomplishes the catching task in both pre- and post-catching phases. In the first phase, a motion planner generates optimal trajectories that minimize catching forces, while in the second, the object's energy is dissipated smoothly, minimizing bouncing. In particular, in the pre-catching phase, a real-time optimal planner is responsible for generating trajectories of the end-effector that minimize the velocity difference between the robot and the object to reduce impact forces during catching. In the post-catching phase, the robot's position, velocity, and stiffness trajectories are generated based on human demonstrations when catching a series of free-falling objects with unknown masses. A hierarchical quadratic programming-based controller is used to enforce the robot's constraints (i.e., joint and torque limits) and create a stack of tasks that minimizes the reflected mass at the end-effector as a secondary objective. The initial experiments isolate the problem along one dimension to accurately study the effects of each contribution on the metrics proposed. We show how the same task, without velocity matching, would be infeasible due to excessive joint torques resulting from the impact. The addition of reflected mass minimization is then investigated, and the catching height is increased to evaluate the method's robustness. Finally, the setup is extended to catching along multiple Cartesian axes, to prove its generalization in space.
>
---
#### [new 006] Active Disturbance Rejection Control for Trajectory Tracking of a Seagoing USV: Design, Simulation, and Field Experiments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于USV轨迹跟踪任务，解决环境干扰下的控制问题。提出ADRC控制器，并通过仿真和实船测试验证其性能。**

- **链接: [http://arxiv.org/pdf/2506.21265v1](http://arxiv.org/pdf/2506.21265v1)**

> **作者:** Jelmer van der Saag; Elia Trevisan; Wouter Falkena; Javier Alonso-Mora
>
> **备注:** Accepted for presentation at IROS 2025. Submitted version
>
> **摘要:** Unmanned Surface Vessels (USVs) face significant control challenges due to uncertain environmental disturbances like waves and currents. This paper proposes a trajectory tracking controller based on Active Disturbance Rejection Control (ADRC) implemented on the DUS V2500. A custom simulation incorporating realistic waves and current disturbances is developed to validate the controller's performance, supported by further validation through field tests in the harbour of Scheveningen, the Netherlands, and at sea. Simulation results demonstrate that ADRC significantly reduces cross-track error across all tested conditions compared to a baseline PID controller but increases control effort and energy consumption. Field trials confirm these findings while revealing a further increase in energy consumption during sea trials compared to the baseline.
>
---
#### [new 007] Knowledge-Driven Imitation Learning: Enabling Generalization Across Diverse Conditions
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决模仿学习在不同条件下的泛化问题。通过引入结构语义知识和关键点图，提升模型在少量专家示范下的表现。**

- **链接: [http://arxiv.org/pdf/2506.21057v1](http://arxiv.org/pdf/2506.21057v1)**

> **作者:** Zhuochen Miao; Jun Lv; Hongjie Fang; Yang Jin; Cewu Lu
>
> **备注:** IROS 2025
>
> **摘要:** Imitation learning has emerged as a powerful paradigm in robot manipulation, yet its generalization capability remains constrained by object-specific dependencies in limited expert demonstrations. To address this challenge, we propose knowledge-driven imitation learning, a framework that leverages external structural semantic knowledge to abstract object representations within the same category. We introduce a novel semantic keypoint graph as a knowledge template and develop a coarse-to-fine template-matching algorithm that optimizes both structural consistency and semantic similarity. Evaluated on three real-world robotic manipulation tasks, our method achieves superior performance, surpassing image-based diffusion policies with only one-quarter of the expert demonstrations. Extensive experiments further demonstrate its robustness across novel objects, backgrounds, and lighting conditions. This work pioneers a knowledge-driven approach to data-efficient robotic learning in real-world settings. Code and more materials are available on https://knowledge-driven.github.io/.
>
---
#### [new 008] Cooperative Circumnavigation for Multi-Quadrotor Systems via Onboard Sensing
- **分类: cs.RO**

- **简介: 该论文属于多旋翼协同绕行任务，解决无外部定位下的目标跟踪问题，通过视觉-惯性融合与分布式滤波实现精准相对定位和自主编队控制。**

- **链接: [http://arxiv.org/pdf/2506.20954v1](http://arxiv.org/pdf/2506.20954v1)**

> **作者:** Xueming Liu; Lin Li; Xiang Zhou; Qingrui Zhang; Tianjiang Hu
>
> **备注:** 8 Pages, 7 figures. Accepted by RA-L
>
> **摘要:** A cooperative circumnavigation framework is proposed for multi-quadrotor systems to enclose and track a moving target without reliance on external localization systems. The distinct relationships between quadrotor-quadrotor and quadrotor-target interactions are evaluated using a heterogeneous perception strategy and corresponding state estimation algorithms. A modified Kalman filter is developed to fuse visual-inertial odometry with range measurements to enhance the accuracy of inter-quadrotor relative localization. An event-triggered distributed Kalman filter is designed to achieve robust target state estimation under visual occlusion by incorporating neighbor measurements and estimated inter-quadrotor relative positions. Using the estimation results, a cooperative circumnavigation controller is constructed, leveraging an oscillator-based autonomous formation flight strategy. We conduct extensive indoor and outdoor experiments to validate the efficiency of the proposed circumnavigation framework in occluded environments. Furthermore, a quadrotor failure experiment highlights the inherent fault tolerance property of the proposed framework, underscoring its potential for deployment in search-and-rescue operations.
>
---
#### [new 009] Fault-Tolerant Spacecraft Attitude Determination using State Estimation Techniques
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于航天器姿态确定任务，旨在解决传感器故障下的姿态估计问题，通过分析多种滤波算法及其故障检测与恢复技术。**

- **链接: [http://arxiv.org/pdf/2506.21016v1](http://arxiv.org/pdf/2506.21016v1)**

> **作者:** B. Chidambaram; A. Hilbert; M. Silva
>
> **备注:** 8 pages, 19 figures
>
> **摘要:** The extended and unscented Kalman filter, and the particle filter provide a robust framework for fault-tolerant attitude estimation on spacecraft. This paper explores how each filter performs for a large satellite in a low earth orbit. Additionally, various techniques, built on these filters, for fault detection, isolation and recovery from erroneous sensor measurements, are analyzed. Key results from this analysis include filter performance for various fault modes.
>
---
#### [new 010] CURL-SLAM: Continuous and Compact LiDAR Mapping
- **分类: cs.RO**

- **简介: 该论文属于3D LiDAR SLAM任务，旨在解决大场景下地图存储与一致性问题，提出CURL-SLAM方法实现紧凑、连续的地图构建与优化。**

- **链接: [http://arxiv.org/pdf/2506.21077v1](http://arxiv.org/pdf/2506.21077v1)**

> **作者:** Kaicheng Zhang; Shida Xu; Yining Ding; Xianwen Kong; Sen Wang
>
> **摘要:** This paper studies 3D LiDAR mapping with a focus on developing an updatable and localizable map representation that enables continuity, compactness and consistency in 3D maps. Traditional LiDAR Simultaneous Localization and Mapping (SLAM) systems often rely on 3D point cloud maps, which typically require extensive storage to preserve structural details in large-scale environments. In this paper, we propose a novel paradigm for LiDAR SLAM by leveraging the Continuous and Ultra-compact Representation of LiDAR (CURL) introduced in [1]. Our proposed LiDAR mapping approach, CURL-SLAM, produces compact 3D maps capable of continuous reconstruction at variable densities using CURL's spherical harmonics implicit encoding, and achieves global map consistency after loop closure. Unlike popular Iterative Closest Point (ICP)-based LiDAR odometry techniques, CURL-SLAM formulates LiDAR pose estimation as a unique optimization problem tailored for CURL and extends it to local Bundle Adjustment (BA), enabling simultaneous pose refinement and map correction. Experimental results demonstrate that CURL-SLAM achieves state-of-the-art 3D mapping quality and competitive LiDAR trajectory accuracy, delivering sensor-rate real-time performance (10 Hz) on a CPU. We will release the CURL-SLAM implementation to the community.
>
---
#### [new 011] ThermalDiffusion: Visual-to-Thermal Image-to-Image Translation for Autonomous Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于图像到图像的翻译任务，旨在解决热成像数据不足的问题，通过条件扩散模型将RGB图像转换为热图像。**

- **链接: [http://arxiv.org/pdf/2506.20969v1](http://arxiv.org/pdf/2506.20969v1)**

> **作者:** Shruti Bansal; Wenshan Wang; Yifei Liu; Parv Maheshwari
>
> **备注:** Accepted at Thermal Infrared in Robotics (TIRO) Workshop, ICRA 2025
>
> **摘要:** Autonomous systems rely on sensors to estimate the environment around them. However, cameras, LiDARs, and RADARs have their own limitations. In nighttime or degraded environments such as fog, mist, or dust, thermal cameras can provide valuable information regarding the presence of objects of interest due to their heat signature. They make it easy to identify humans and vehicles that are usually at higher temperatures compared to their surroundings. In this paper, we focus on the adaptation of thermal cameras for robotics and automation, where the biggest hurdle is the lack of data. Several multi-modal datasets are available for driving robotics research in tasks such as scene segmentation, object detection, and depth estimation, which are the cornerstone of autonomous systems. However, they are found to be lacking in thermal imagery. Our paper proposes a solution to augment these datasets with synthetic thermal data to enable widespread and rapid adaptation of thermal cameras. We explore the use of conditional diffusion models to convert existing RGB images to thermal images using self-attention to learn the thermal properties of real-world objects.
>
---
#### [new 012] V2X-REALM: Vision-Language Model-Based Robust End-to-End Cooperative Autonomous Driving with Adaptive Long-Tail Modeling
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决城市环境中长尾场景下的协同决策问题。提出V2X-REALM框架，通过视觉-语言模型提升鲁棒性和安全性。**

- **链接: [http://arxiv.org/pdf/2506.21041v1](http://arxiv.org/pdf/2506.21041v1)**

> **作者:** Junwei You; Pei Li; Zhuoyu Jiang; Zilin Huang; Rui Gan; Haotian Shi; Bin Ran
>
> **摘要:** Ensuring robust planning and decision-making under rare, diverse, and visually degraded long-tail scenarios remains a fundamental challenge for autonomous driving in urban environments. This issue becomes more critical in cooperative settings, where vehicles and infrastructure jointly perceive and reason across complex environments. To address this challenge, we propose V2X-REALM, a vision-language model (VLM)-based framework with adaptive multimodal learning for robust cooperative autonomous driving under long-tail scenarios. V2X-REALM introduces three core innovations: (i) a prompt-driven long-tail scenario generation and evaluation pipeline that leverages foundation models to synthesize realistic long-tail conditions such as snow and fog across vehicle- and infrastructure-side views, enriching training diversity efficiently; (ii) a gated multi-scenario adaptive attention module that modulates the visual stream using scenario priors to recalibrate ambiguous or corrupted features; and (iii) a multi-task scenario-aware contrastive learning objective that improves multimodal alignment and promotes cross-scenario feature separability. Extensive experiments demonstrate that V2X-REALM significantly outperforms existing baselines in robustness, semantic reasoning, safety, and planning accuracy under complex, challenging driving conditions, advancing the scalability of end-to-end cooperative autonomous driving.
>
---
#### [new 013] STEP Planner: Constructing cross-hierarchical subgoal tree as an embodied long-horizon task planner
- **分类: cs.RO**

- **简介: 该论文属于机器人长期任务规划任务，解决LLM在长时序任务中成功率低的问题。提出STEP框架，构建分层子目标树，提升任务完成率。**

- **链接: [http://arxiv.org/pdf/2506.21030v1](http://arxiv.org/pdf/2506.21030v1)**

> **作者:** Zhou Tianxing; Wang Zhirui; Ao Haojia; Chen Guangyan; Xing Boyang; Cheng Jingwen; Yang Yi; Yue Yufeng
>
> **摘要:** The ability to perform reliable long-horizon task planning is crucial for deploying robots in real-world environments. However, directly employing Large Language Models (LLMs) as action sequence generators often results in low success rates due to their limited reasoning ability for long-horizon embodied tasks. In the STEP framework, we construct a subgoal tree through a pair of closed-loop models: a subgoal decomposition model and a leaf node termination model. Within this framework, we develop a hierarchical tree structure that spans from coarse to fine resolutions. The subgoal decomposition model leverages a foundation LLM to break down complex goals into manageable subgoals, thereby spanning the subgoal tree. The leaf node termination model provides real-time feedback based on environmental states, determining when to terminate the tree spanning and ensuring each leaf node can be directly converted into a primitive action. Experiments conducted in both the VirtualHome WAH-NL benchmark and on real robots demonstrate that STEP achieves long-horizon embodied task completion with success rates up to 34% (WAH-NL) and 25% (real robot) outperforming SOTA methods.
>
---
#### [new 014] WorldVLA: Towards Autoregressive Action World Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出WorldVLA，融合视觉-语言-动作模型与世界模型，解决动作生成与环境预测问题，通过互惠机制提升性能。**

- **链接: [http://arxiv.org/pdf/2506.21539v1](http://arxiv.org/pdf/2506.21539v1)**

> **作者:** Jun Cen; Chaohui Yu; Hangjie Yuan; Yuming Jiang; Siteng Huang; Jiayan Guo; Xin Li; Yibing Song; Hao Luo; Fan Wang; Deli Zhao; Hao Chen
>
> **备注:** Code: https://github.com/alibaba-damo-academy/WorldVLA
>
> **摘要:** We present WorldVLA, an autoregressive action world model that unifies action and image understanding and generation. Our WorldVLA intergrates Vision-Language-Action (VLA) model and world model in one single framework. The world model predicts future images by leveraging both action and image understanding, with the purpose of learning the underlying physics of the environment to improve action generation. Meanwhile, the action model generates the subsequent actions based on image observations, aiding in visual understanding and in turn helps visual generation of the world model. We demonstrate that WorldVLA outperforms standalone action and world models, highlighting the mutual enhancement between the world model and the action model. In addition, we find that the performance of the action model deteriorates when generating sequences of actions in an autoregressive manner. This phenomenon can be attributed to the model's limited generalization capability for action prediction, leading to the propagation of errors from earlier actions to subsequent ones. To address this issue, we propose an attention mask strategy that selectively masks prior actions during the generation of the current action, which shows significant performance improvement in the action chunk generation task.
>
---
#### [new 015] Real-time Terrain Analysis for Off-road Autonomous Vehicles
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶车辆控制任务，旨在解决路面不平度导致的行驶偏差问题。通过贝叶斯校准方法实时估计路面粗糙度，并结合控制器提升安全性。**

- **链接: [http://arxiv.org/pdf/2506.21347v1](http://arxiv.org/pdf/2506.21347v1)**

> **作者:** Edwina Lewis; Aditya Parameshwaran; Laura Redmond; Yue Wang
>
> **摘要:** This research addresses critical autonomous vehicle control challenges arising from road roughness variation, which induces course deviations and potential loss of road contact during steering operations. We present a novel real-time road roughness estimation system employing Bayesian calibration methodology that processes axle accelerations to predict terrain roughness with quantifiable confidence measures. The technical framework integrates a Gaussian process surrogate model with a simulated half-vehicle model, systematically processing vehicle velocity and road surface roughness parameters to generate corresponding axle acceleration responses. The Bayesian calibration routine performs inverse estimation of road roughness from observed accelerations and velocities, yielding posterior distributions that quantify prediction uncertainty for adaptive risk management. Training data generation utilizes Latin Hypercube sampling across comprehensive velocity and roughness parameter spaces, while the calibrated model integrates seamlessly with a Simplex controller architecture to dynamically adjust velocity limits based on real-time roughness predictions. Experimental validation on stochastically generated surfaces featuring varying roughness regions demonstrates robust real-time characterization capabilities, with the integrated Simplex control strategy effectively enhancing autonomous vehicle operational safety through proactive surface condition response. This innovative Bayesian framework establishes a comprehensive foundation for mitigating roughness-related operational risks while simultaneously improving efficiency and safety margins in autonomous vehicle systems.
>
---
#### [new 016] Control of Marine Robots in the Era of Data-Driven Intelligence
- **分类: cs.RO**

- **简介: 本文属于海洋机器人控制任务，旨在解决传统方法在非线性与不确定性环境中的局限性，通过数据驱动智能提升控制性能。**

- **链接: [http://arxiv.org/pdf/2506.21063v1](http://arxiv.org/pdf/2506.21063v1)**

> **作者:** Lin Hong; Lu Liu; Zhouhua Peng; Fumin Zhang
>
> **摘要:** The control of marine robots has long relied on model-based methods grounded in classical and modern control theory. However, the nonlinearity and uncertainties inherent in robot dynamics, coupled with the complexity of marine environments, have revealed the limitations of conventional control methods. The rapid evolution of machine learning has opened new avenues for incorporating data-driven intelligence into control strategies, prompting a paradigm shift in the control of marine robots. This paper provides a review of recent progress in marine robot control through the lens of this emerging paradigm. The review covers both individual and cooperative marine robotic systems, highlighting notable achievements in data-driven control of marine robots and summarizing open-source resources that support the development and validation of advanced control methods. Finally, several future perspectives are outlined to guide research toward achieving high-level autonomy for marine robots in real-world applications. This paper aims to serve as a roadmap toward the next-generation control framework of marine robots in the era of data-driven intelligence.
>
---
#### [new 017] Online Planning for Cooperative Air-Ground Robot Systems with Unknown Fuel Requirements
- **分类: cs.RO**

- **简介: 该论文研究在线协同空地机器人系统路径规划问题，解决未知燃油消耗下的任务调度与动态补给难题，提出分阶段规划方法确保任务完成。**

- **链接: [http://arxiv.org/pdf/2506.20804v1](http://arxiv.org/pdf/2506.20804v1)**

> **作者:** Ritvik Agarwal; Behnoushsadat Hatami; Alvika Gautam; Parikshit Maini
>
> **备注:** Submitted to RSS (MRS Workshop)
>
> **摘要:** We consider an online variant of the fuel-constrained UAV routing problem with a ground-based mobile refueling station (FCURP-MRS), where targets incur unknown fuel costs. We develop a two-phase solution: an offline heuristic-based planner computes initial UAV and UGV paths, and a novel online planning algorithm that dynamically adjusts rendezvous points based on real-time fuel consumption during target processing. Preliminary Gazebo simulations demonstrate the feasibility of our approach in maintaining UAV-UGV path validity, ensuring mission completion. Link to video: https://youtu.be/EmpVj-fjqNY
>
---
#### [new 018] UAIbot: Beginner-friendly web-based simulator for interactive robotics learning and research
- **分类: cs.RO; 68T40; I.2.9; I.6.3**

- **简介: 该论文介绍UAIbot，一个面向教育和研究的开源网页机器人模拟器，解决传统平台安装复杂、学习门槛高的问题，提供Python和JavaScript接口以支持交互式学习与实验。**

- **链接: [http://arxiv.org/pdf/2506.21178v1](http://arxiv.org/pdf/2506.21178v1)**

> **作者:** Johnata Brayan; Armando Alves Neto; Pavel Petrovič; Gustavo M Freitas; Vinicius Mariano Gonçalves
>
> **备注:** 12 pages, 8 figures, submitted to Springer proceedings
>
> **摘要:** This paper presents UAIbot, a free and open-source web-based robotics simulator designed to address the educational and research challenges conventional simulation platforms generally face. The Python and JavaScript interfaces of UAIbot enable accessible hands-on learning experiences without cumbersome installations. By allowing users to explore fundamental mathematical and physical principles interactively, ranging from manipulator kinematics to pedestrian flow dynamics, UAIbot provides an effective tool for deepening student understanding, facilitating rapid experimentation, and enhancing research dissemination.
>
---
#### [new 019] How do Foundation Models Compare to Skeleton-Based Approaches for Gesture Recognition in Human-Robot Interaction?
- **分类: cs.CV; cs.HC; cs.RO; I.2.10; I.2.9; I.5.4; I.4.8; I.4.9; H.1.2**

- **简介: 该论文属于手势识别任务，旨在比较基础模型与骨架方法在人机交互中的表现，通过实验验证不同模型的适用性。**

- **链接: [http://arxiv.org/pdf/2506.20795v1](http://arxiv.org/pdf/2506.20795v1)**

> **作者:** Stephanie Käs; Anton Burenko; Louis Markert; Onur Alp Culha; Dennis Mack; Timm Linder; Bastian Leibe
>
> **摘要:** Gestures enable non-verbal human-robot communication, especially in noisy environments like agile production. Traditional deep learning-based gesture recognition relies on task-specific architectures using images, videos, or skeletal pose estimates as input. Meanwhile, Vision Foundation Models (VFMs) and Vision Language Models (VLMs) with their strong generalization abilities offer potential to reduce system complexity by replacing dedicated task-specific modules. This study investigates adapting such models for dynamic, full-body gesture recognition, comparing V-JEPA (a state-of-the-art VFM), Gemini Flash 2.0 (a multimodal VLM), and HD-GCN (a top-performing skeleton-based approach). We introduce NUGGET, a dataset tailored for human-robot communication in intralogistics environments, to evaluate the different gesture recognition approaches. In our experiments, HD-GCN achieves best performance, but V-JEPA comes close with a simple, task-specific classification head - thus paving a possible way towards reducing system complexity, by using it as a shared multi-task model. In contrast, Gemini struggles to differentiate gestures based solely on textual descriptions in the zero-shot setting, highlighting the need of further research on suitable input representations for gestures.
>
---
#### [new 020] Our Coding Adventure: Using LLMs to Personalise the Narrative of a Tangible Programming Robot for Preschoolers
- **分类: cs.CY; cs.RO; K.3.1**

- **简介: 该论文属于教育技术任务，旨在利用LLMs为幼儿编程机器人Cubetto生成个性化故事，解决屏幕时间过长和教学内容单一问题。通过行动研究开发了可复现的叙事生成流程。**

- **链接: [http://arxiv.org/pdf/2506.20982v1](http://arxiv.org/pdf/2506.20982v1)**

> **作者:** Martin Ruskov
>
> **备注:** accepted at D-SAIL Workshop - Transformative Curriculum Design: Digitalization, Sustainability, and AI Literacy for 21st Century Learning
>
> **摘要:** Finding balanced ways to employ Large Language Models (LLMs) in education is a challenge due to inherent risks of poor understanding of the technology and of a susceptible audience. This is particularly so with younger children, who are known to have difficulties with pervasive screen time. Working with a tangible programming robot called Cubetto, we propose an approach to benefit from the capabilities of LLMs by employing such models in the preparation of personalised storytelling, necessary for preschool children to get accustomed to the practice of commanding the robot. We engage in action research to develop an early version of a formalised process to rapidly prototype game stories for Cubetto. Our approach has both reproducible results, because it employs open weight models, and is model-agnostic, because we test it with 5 different LLMs. We document on one hand the process, the used materials and prompts, and on the other the learning experience and outcomes. We deem the generation successful for the intended purposes of using the results as a teacher aid. Testing the models on 4 different task scenarios, we encounter issues of consistency and hallucinations and document the corresponding evaluation process and attempts (some successful and some not) to overcome these issues. Importantly, the process does not expose children to LLMs directly. Rather, the technology is used to help teachers easily develop personalised narratives on children's preferred topics. We believe our method is adequate for preschool classes and we are planning to further experiment in real-world educational settings.
>
---
#### [new 021] Real-Time ESFP: Estimating, Smoothing, Filtering, and Pose-Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ESFP系统，将单目视频转换为桌面机械臂的关节轨迹，解决动作捕捉与控制问题，通过估计、平滑、过滤和姿态映射实现精准控制。**

- **链接: [http://arxiv.org/pdf/2506.21234v1](http://arxiv.org/pdf/2506.21234v1)**

> **作者:** Qifei Cui; Yuang Zhou; Ruichen Deng
>
> **摘要:** This paper presents ESFP, an end-to-end pipeline that converts monocular RGB video into executable joint trajectories for a low-cost 4-DoF desktop arm. ESFP comprises four sequential modules. (1) Estimating: ROMP lifts each frame to a 24-joint 3-D skeleton. (2) Smoothing: the proposed HPSTM-a sequence-to-sequence Transformer with self-attention-combines long-range temporal context with a differentiable forward-kinematics decoder, enforcing constant bone lengths and anatomical plausibility while jointly predicting joint means and full covariances. (3) Filtering: root-normalized trajectories are variance-weighted according to HPSTM's uncertainty estimates, suppressing residual noise. (4) Pose-Mapping: a geometric retargeting layer transforms shoulder-elbow-wrist triples into the uArm's polar workspace, preserving wrist orientation.
>
---
#### [new 022] SAM4D: Segment Anything in Camera and LiDAR Streams
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SAM4D，用于相机与LiDAR的跨模态分割任务，解决自动驾驶中数据标注效率低和时序一致性差的问题。**

- **链接: [http://arxiv.org/pdf/2506.21547v1](http://arxiv.org/pdf/2506.21547v1)**

> **作者:** Jianyun Xu; Song Wang; Ziqian Ni; Chunyong Hu; Sheng Yang; Jianke Zhu; Qiang Li
>
> **备注:** Accepted by ICCV2025, Project Page: https://SAM4D-Project.github.io
>
> **摘要:** We present SAM4D, a multi-modal and temporal foundation model designed for promptable segmentation across camera and LiDAR streams. Unified Multi-modal Positional Encoding (UMPE) is introduced to align camera and LiDAR features in a shared 3D space, enabling seamless cross-modal prompting and interaction. Additionally, we propose Motion-aware Cross-modal Memory Attention (MCMA), which leverages ego-motion compensation to enhance temporal consistency and long-horizon feature retrieval, ensuring robust segmentation across dynamically changing autonomous driving scenes. To avoid annotation bottlenecks, we develop a multi-modal automated data engine that synergizes VFM-driven video masklets, spatiotemporal 4D reconstruction, and cross-modal masklet fusion. This framework generates camera-LiDAR aligned pseudo-labels at a speed orders of magnitude faster than human annotation while preserving VFM-derived semantic fidelity in point cloud representations. We conduct extensive experiments on the constructed Waymo-4DSeg, which demonstrate the powerful cross-modal segmentation ability and great potential in data annotation of proposed SAM4D.
>
---
#### [new 023] SEPT: Standard-Definition Map Enhanced Scene Perception and Topology Reasoning for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的场景感知与拓扑推理任务，旨在解决在线环境理解受限问题。通过融合标准地图提升感知与推理性能。**

- **链接: [http://arxiv.org/pdf/2505.12246v1](http://arxiv.org/pdf/2505.12246v1)**

> **作者:** Muleilan Pei; Jiayao Shan; Peiliang Li; Jieqi Shi; Jing Huo; Yang Gao; Shaojie Shen
>
> **备注:** Accepted by IEEE Robotics and Automation Letters
>
> **摘要:** Online scene perception and topology reasoning are critical for autonomous vehicles to understand their driving environments, particularly for mapless driving systems that endeavor to reduce reliance on costly High-Definition (HD) maps. However, recent advances in online scene understanding still face limitations, especially in long-range or occluded scenarios, due to the inherent constraints of onboard sensors. To address this challenge, we propose a Standard-Definition (SD) Map Enhanced scene Perception and Topology reasoning (SEPT) framework, which explores how to effectively incorporate the SD map as prior knowledge into existing perception and reasoning pipelines. Specifically, we introduce a novel hybrid feature fusion strategy that combines SD maps with Bird's-Eye-View (BEV) features, considering both rasterized and vectorized representations, while mitigating potential misalignment between SD maps and BEV feature spaces. Additionally, we leverage the SD map characteristics to design an auxiliary intersection-aware keypoint detection task, which further enhances the overall scene understanding performance. Experimental results on the large-scale OpenLane-V2 dataset demonstrate that by effectively integrating SD map priors, our framework significantly improves both scene perception and topology reasoning, outperforming existing methods by a substantial margin.
>
---
#### [new 024] Out-of-Distribution Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于3D语义占用预测任务，旨在解决OoD物体检测问题。通过引入新数据集和OccOoD框架，提升OoD检测性能。**

- **链接: [http://arxiv.org/pdf/2506.21185v1](http://arxiv.org/pdf/2506.21185v1)**

> **作者:** Yuheng Zhang; Mengfei Duan; Kunyu Peng; Yuhang Wang; Ruiping Liu; Fei Teng; Kai Luo; Zhiyong Li; Kailun Yang
>
> **备注:** The established datasets and source code will be made publicly available at https://github.com/7uHeng/OccOoD
>
> **摘要:** 3D Semantic Occupancy Prediction is crucial for autonomous driving, providing a dense, semantically rich environmental representation. However, existing methods focus on in-distribution scenes, making them susceptible to Out-of-Distribution (OoD) objects and long-tail distributions, which increases the risk of undetected anomalies and misinterpretations, posing safety hazards. To address these challenges, we introduce Out-of-Distribution Semantic Occupancy Prediction, targeting OoD detection in 3D voxel space. To fill the gaps in the dataset, we propose a Synthetic Anomaly Integration Pipeline that injects synthetic anomalies while preserving realistic spatial and occlusion patterns, enabling the creation of two datasets: VAA-KITTI and VAA-KITTI-360. We introduce OccOoD, a novel framework integrating OoD detection into 3D semantic occupancy prediction, with Voxel-BEV Progressive Fusion (VBPF) leveraging an RWKV-based branch to enhance OoD detection via geometry-semantic fusion. Experimental results demonstrate that OccOoD achieves state-of-the-art OoD detection with an AuROC of 67.34% and an AuPRCr of 29.21% within a 1.2m region, while maintaining competitive occupancy prediction performance. The established datasets and source code will be made publicly available at https://github.com/7uHeng/OccOoD.
>
---
#### [new 025] ConViTac: Aligning Visual-Tactile Fusion with Contrastive Representations
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多模态感知任务，旨在解决视觉与触觉特征融合不佳的问题。提出ConViTac网络，通过对比嵌入条件机制提升特征对齐效果。**

- **链接: [http://arxiv.org/pdf/2506.20757v1](http://arxiv.org/pdf/2506.20757v1)**

> **作者:** Zhiyuan Wu; Yongqiang Zhao; Shan Luo
>
> **摘要:** Vision and touch are two fundamental sensory modalities for robots, offering complementary information that enhances perception and manipulation tasks. Previous research has attempted to jointly learn visual-tactile representations to extract more meaningful information. However, these approaches often rely on direct combination, such as feature addition and concatenation, for modality fusion, which tend to result in poor feature integration. In this paper, we propose ConViTac, a visual-tactile representation learning network designed to enhance the alignment of features during fusion using contrastive representations. Our key contribution is a Contrastive Embedding Conditioning (CEC) mechanism that leverages a contrastive encoder pretrained through self-supervised contrastive learning to project visual and tactile inputs into unified latent embeddings. These embeddings are used to couple visual-tactile feature fusion through cross-modal attention, aiming at aligning the unified representations and enhancing performance on downstream tasks. We conduct extensive experiments to demonstrate the superiority of ConViTac in real world over current state-of-the-art methods and the effectiveness of our proposed CEC mechanism, which improves accuracy by up to 12.0% in material classification and grasping prediction tasks.
>
---
#### [new 026] GoIRL: Graph-Oriented Inverse Reinforcement Learning for Multimodal Trajectory Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的轨迹预测任务，旨在解决多模态轨迹预测问题。提出GoIRL框架，结合图结构与逆强化学习，提升预测准确性和多样性。**

- **链接: [http://arxiv.org/pdf/2506.21121v1](http://arxiv.org/pdf/2506.21121v1)**

> **作者:** Muleilan Pei; Shaoshuai Shi; Lu Zhang; Peiliang Li; Shaojie Shen
>
> **备注:** Accepted by ICML 2025
>
> **摘要:** Trajectory prediction for surrounding agents is a challenging task in autonomous driving due to its inherent uncertainty and underlying multimodality. Unlike prevailing data-driven methods that primarily rely on supervised learning, in this paper, we introduce a novel Graph-oriented Inverse Reinforcement Learning (GoIRL) framework, which is an IRL-based predictor equipped with vectorized context representations. We develop a feature adaptor to effectively aggregate lane-graph features into grid space, enabling seamless integration with the maximum entropy IRL paradigm to infer the reward distribution and obtain the policy that can be sampled to induce multiple plausible plans. Furthermore, conditioned on the sampled plans, we implement a hierarchical parameterized trajectory generator with a refinement module to enhance prediction accuracy and a probability fusion strategy to boost prediction confidence. Extensive experimental results showcase our approach not only achieves state-of-the-art performance on the large-scale Argoverse & nuScenes motion forecasting benchmarks but also exhibits superior generalization abilities compared to existing supervised models.
>
---
#### [new 027] ToosiCubix: Monocular 3D Cuboid Labeling via Vehicle Part Annotations
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，解决车辆3D立方体标注问题。提出ToosiCubix方法，仅用单目图像和相机参数，通过用户点击标注关键点，实现高效准确的3D立方体标注。**

- **链接: [http://arxiv.org/pdf/2506.21358v1](http://arxiv.org/pdf/2506.21358v1)**

> **作者:** Behrooz Nasihatkon; Hossein Resani; Amirreza Mehrzadian
>
> **摘要:** Many existing methods for 3D cuboid annotation of vehicles rely on expensive and carefully calibrated camera-LiDAR or stereo setups, limiting their accessibility for large-scale data collection. We introduce ToosiCubix, a simple yet powerful approach for annotating ground-truth cuboids using only monocular images and intrinsic camera parameters. Our method requires only about 10 user clicks per vehicle, making it highly practical for adding 3D annotations to existing datasets originally collected without specialized equipment. By annotating specific features (e.g., wheels, car badge, symmetries) across different vehicle parts, we accurately estimate each vehicle's position, orientation, and dimensions up to a scale ambiguity (8 DoF). The geometric constraints are formulated as an optimization problem, which we solve using a coordinate descent strategy, alternating between Perspective-n-Points (PnP) and least-squares subproblems. To handle common ambiguities such as scale and unobserved dimensions, we incorporate probabilistic size priors, enabling 9 DoF cuboid placements. We validate our annotations against the KITTI and Cityscapes3D datasets, demonstrating that our method offers a cost-effective and scalable solution for high-quality 3D cuboid annotation.
>
---
#### [new 028] "Who Should I Believe?": User Interpretation and Decision-Making When a Family Healthcare Robot Contradicts Human Memory
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，研究用户在医疗机器人与记忆冲突时的信任与决策问题。通过实验分析透明度和社交性对用户判断的影响。**

- **链接: [http://arxiv.org/pdf/2506.21322v1](http://arxiv.org/pdf/2506.21322v1)**

> **作者:** Hong Wang; Natalia Calvo-Barajas; Katie Winkle; Ginevra Castellano
>
> **备注:** 8 pages
>
> **摘要:** Advancements in robotic capabilities for providing physical assistance, psychological support, and daily health management are making the deployment of intelligent healthcare robots in home environments increasingly feasible in the near future. However, challenges arise when the information provided by these robots contradicts users' memory, raising concerns about user trust and decision-making. This paper presents a study that examines how varying a robot's level of transparency and sociability influences user interpretation, decision-making and perceived trust when faced with conflicting information from a robot. In a 2 x 2 between-subjects online study, 176 participants watched videos of a Furhat robot acting as a family healthcare assistant and suggesting a fictional user to take medication at a different time from that remembered by the user. Results indicate that robot transparency influenced users' interpretation of information discrepancies: with a low transparency robot, the most frequent assumption was that the user had not correctly remembered the time, while with the high transparency robot, participants were more likely to attribute the discrepancy to external factors, such as a partner or another household member modifying the robot's information. Additionally, participants exhibited a tendency toward overtrust, often prioritizing the robot's recommendations over the user's memory, even when suspecting system malfunctions or third-party interference. These findings highlight the impact of transparency mechanisms in robotic systems, the complexity and importance associated with system access control for multi-user robots deployed in home environments, and the potential risks of users' over reliance on robots in sensitive domains such as healthcare.
>
---
#### [new 029] Unlocking Constraints: Source-Free Occlusion-Aware Seamless Segmentation
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出SFOASS任务，解决无源数据下的全景图像分割问题。通过UNLOCK框架，实现无监督域适应与遮挡感知分割。**

- **链接: [http://arxiv.org/pdf/2506.21198v1](http://arxiv.org/pdf/2506.21198v1)**

> **作者:** Yihong Cao; Jiaming Zhang; Xu Zheng; Hao Shi; Kunyu Peng; Hang Liu; Kailun Yang; Hui Zhang
>
> **备注:** Accepted to ICCV 2025. All data and code will be made publicly available at https://github.com/yihong-97/UNLOCK
>
> **摘要:** Panoramic image processing is essential for omni-context perception, yet faces constraints like distortions, perspective occlusions, and limited annotations. Previous unsupervised domain adaptation methods transfer knowledge from labeled pinhole data to unlabeled panoramic images, but they require access to source pinhole data. To address these, we introduce a more practical task, i.e., Source-Free Occlusion-Aware Seamless Segmentation (SFOASS), and propose its first solution, called UNconstrained Learning Omni-Context Knowledge (UNLOCK). Specifically, UNLOCK includes two key modules: Omni Pseudo-Labeling Learning and Amodal-Driven Context Learning. While adapting without relying on source data or target labels, this framework enhances models to achieve segmentation with 360{\deg} viewpoint coverage and occlusion-aware reasoning. Furthermore, we benchmark the proposed SFOASS task through both real-to-real and synthetic-to-real adaptation settings. Experimental results show that our source-free method achieves performance comparable to source-dependent methods, yielding state-of-the-art scores of 10.9 in mAAP and 11.6 in mAP, along with an absolute improvement of +4.3 in mAPQ over the source-only method. All data and code will be made publicly available at https://github.com/yihong-97/UNLOCK.
>
---
#### [new 030] EndoFlow-SLAM: Real-Time Endoscopic SLAM with Flow-Constrained Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于端到端SLAM任务，解决内窥镜场景中的3D重建与定位问题，通过引入光流约束和深度正则化提升性能。**

- **链接: [http://arxiv.org/pdf/2506.21420v1](http://arxiv.org/pdf/2506.21420v1)**

> **作者:** Taoyu Wu; Yiyi Miao; Zhuoxiao Li; Haocheng Zhao; Kang Dang; Jionglong Su; Limin Yu; Haoang Li
>
> **摘要:** Efficient three-dimensional reconstruction and real-time visualization are critical in surgical scenarios such as endoscopy. In recent years, 3D Gaussian Splatting (3DGS) has demonstrated remarkable performance in efficient 3D reconstruction and rendering. Most 3DGS-based Simultaneous Localization and Mapping (SLAM) methods only rely on the appearance constraints for optimizing both 3DGS and camera poses. However, in endoscopic scenarios, the challenges include photometric inconsistencies caused by non-Lambertian surfaces and dynamic motion from breathing affects the performance of SLAM systems. To address these issues, we additionally introduce optical flow loss as a geometric constraint, which effectively constrains both the 3D structure of the scene and the camera motion. Furthermore, we propose a depth regularisation strategy to mitigate the problem of photometric inconsistencies and ensure the validity of 3DGS depth rendering in endoscopic scenes. In addition, to improve scene representation in the SLAM system, we improve the 3DGS refinement strategy by focusing on viewpoints corresponding to Keyframes with suboptimal rendering quality frames, achieving better rendering results. Extensive experiments on the C3VD static dataset and the StereoMIS dynamic dataset demonstrate that our method outperforms existing state-of-the-art methods in novel view synthesis and pose estimation, exhibiting high performance in both static and dynamic surgical scenes. The source code will be publicly available upon paper acceptance.
>
---
#### [new 031] Effect of Haptic Feedback on Avoidance Behavior and Visual Exploration in Dynamic VR Pedestrian Environment
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于VR人机交互任务，研究haptic反馈对动态虚拟环境中行人避让行为和视觉探索的影响，通过实验验证其提升用户碰撞敏感性和行为反应的效果。**

- **链接: [http://arxiv.org/pdf/2506.20952v1](http://arxiv.org/pdf/2506.20952v1)**

> **作者:** Kyosuke Ishibashi; Atsushi Saito; Zin Y. Tun; Lucas Ray; Megan C. Coram; Akihiro Sakurai; Allison M. Okamura; Ko Yamamoto
>
> **摘要:** Human crowd simulation in virtual reality (VR) is a powerful tool with potential applications including emergency evacuation training and assessment of building layout. While haptic feedback in VR enhances immersive experience, its effect on walking behavior in dense and dynamic pedestrian flows is unknown. Through a user study, we investigated how haptic feedback changes user walking motion in crowded pedestrian flows in VR. The results indicate that haptic feedback changed users' collision avoidance movements, as measured by increased walking trajectory length and change in pelvis angle. The displacements of users' lateral position and pelvis angle were also increased in the instantaneous response to a collision with a non-player character (NPC), even when the NPC was inside the field of view. Haptic feedback also enhanced users' awareness and visual exploration when an NPC approached from the side and back. Furthermore, variation in walking speed was increased by the haptic feedback. These results suggested that the haptic feedback enhanced users' sensitivity to a collision in VR environment.
>
---
#### [new 032] World-aware Planning Narratives Enhance Large Vision-Language Model Planner
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于视觉语言模型的规划任务，旨在解决复杂场景下的多步骤目标规划问题。通过引入环境感知能力，提升模型的视觉推理与长期规划效果。**

- **链接: [http://arxiv.org/pdf/2506.21230v1](http://arxiv.org/pdf/2506.21230v1)**

> **作者:** Junhao Shi; Zhaoye Fei; Siyin Wang; Qipeng Guo; Jingjing Gong; Xipeng QIu
>
> **摘要:** Large Vision-Language Models (LVLMs) show promise for embodied planning tasks but struggle with complex scenarios involving unfamiliar environments and multi-step goals. Current approaches rely on environment-agnostic imitation learning that disconnects instructions from environmental contexts, causing models to struggle with context-sensitive instructions and rely on supplementary cues rather than visual reasoning during long-horizon interactions. In this work, we propose World-Aware Planning Narrative Enhancement (WAP), a framework that infuses LVLMs with comprehensive environmental understanding through four cognitive capabilities (visual appearance modeling, spatial reasoning, functional abstraction, and syntactic grounding) while developing and evaluating models using only raw visual observations through curriculum learning. Evaluations on the EB-ALFRED benchmark demonstrate substantial improvements, with Qwen2.5-VL achieving a 60.7 absolute improvement in task success rates, particularly in commonsense reasoning (+60.0) and long-horizon planning (+70.0). Notably, our enhanced open-source models outperform proprietary systems like GPT-4o and Claude-3.5-Sonnet by a large margin.
>
---
#### [new 033] Whole-Body Conditioned Egocentric Video Prediction
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文属于视频预测任务，旨在通过人体动作预测第一视角视频。工作包括构建模型、使用真实数据训练，并设计评估协议分析模型能力。**

- **链接: [http://arxiv.org/pdf/2506.21552v1](http://arxiv.org/pdf/2506.21552v1)**

> **作者:** Yutong Bai; Danny Tran; Amir Bar; Yann LeCun; Trevor Darrell; Jitendra Malik
>
> **备注:** Project Page: https://dannytran123.github.io/PEVA
>
> **摘要:** We train models to Predict Ego-centric Video from human Actions (PEVA), given the past video and an action represented by the relative 3D body pose. By conditioning on kinematic pose trajectories, structured by the joint hierarchy of the body, our model learns to simulate how physical human actions shape the environment from a first-person point of view. We train an auto-regressive conditional diffusion transformer on Nymeria, a large-scale dataset of real-world egocentric video and body pose capture. We further design a hierarchical evaluation protocol with increasingly challenging tasks, enabling a comprehensive analysis of the model's embodied prediction and control abilities. Our work represents an initial attempt to tackle the challenges of modeling complex real-world environments and embodied agent behaviors with video prediction from the perspective of a human.
>
---
#### [new 034] Flow-Based Single-Step Completion for Efficient and Expressive Policy Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决生成模型在离线RL中推理成本高、训练不稳定的问题。提出SSCP方法，实现高效且表达能力强的策略学习。**

- **链接: [http://arxiv.org/pdf/2506.21427v1](http://arxiv.org/pdf/2506.21427v1)**

> **作者:** Prajwal Koirala; Cody Fleming
>
> **摘要:** Generative models such as diffusion and flow-matching offer expressive policies for offline reinforcement learning (RL) by capturing rich, multimodal action distributions, but their iterative sampling introduces high inference costs and training instability due to gradient propagation across sampling steps. We propose the \textit{Single-Step Completion Policy} (SSCP), a generative policy trained with an augmented flow-matching objective to predict direct completion vectors from intermediate flow samples, enabling accurate, one-shot action generation. In an off-policy actor-critic framework, SSCP combines the expressiveness of generative models with the training and inference efficiency of unimodal policies, without requiring long backpropagation chains. Our method scales effectively to offline, offline-to-online, and online RL settings, offering substantial gains in speed and adaptability over diffusion-based baselines. We further extend SSCP to goal-conditioned RL, enabling flat policies to exploit subgoal structures without explicit hierarchical inference. SSCP achieves strong results across standard offline RL and behavior cloning benchmarks, positioning it as a versatile, expressive, and efficient framework for deep RL and sequential decision-making.
>
---
## 更新

#### [replaced 001] Rapid Gyroscope Calibration: A Deep Learning Approach
- **分类: cs.LG; cs.AI; cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2409.00488v3](http://arxiv.org/pdf/2409.00488v3)**

> **作者:** Yair Stolero; Itzik Klein
>
> **备注:** 10 Pages, 14 Figures
>
> **摘要:** Low-cost gyroscope calibration is essential for ensuring the accuracy and reliability of gyroscope measurements. Stationary calibration estimates the deterministic parts of measurement errors. To this end, a common practice is to average the gyroscope readings during a predefined period and estimate the gyroscope bias. Calibration duration plays a crucial role in performance, therefore, longer periods are preferred. However, some applications require quick startup times and calibration is therefore allowed only for a short time. In this work, we focus on reducing low-cost gyroscope calibration time using deep learning methods. We propose an end-to-end convolutional neural network for the application of gyroscope calibration. We explore the possibilities of using multiple real and virtual gyroscopes to improve the calibration performance of single gyroscopes. To train and validate our approach, we recorded a dataset consisting of 186.6 hours of gyroscope readings, using 36 gyroscopes of four different brands. We also created a virtual dataset consisting of simulated gyroscope readings. The six datasets were used to evaluate our proposed approach. One of our key achievements in this work is reducing gyroscope calibration time by up to 89% using three low-cost gyroscopes. Our dataset is publicly available to allow reproducibility of our work and to increase research in the field.
>
---
#### [replaced 002] Using Explainable AI and Hierarchical Planning for Outreach with Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.00808v3](http://arxiv.org/pdf/2404.00808v3)**

> **作者:** Rushang Karia; Jayesh Nagpal; Daksh Dobhal; Pulkit Verma; Rashmeet Kaur Nayyar; Naman Shah; Siddharth Srivastava
>
> **摘要:** Understanding how robots plan and execute tasks is crucial in today's world, where they are becoming more prevalent in our daily lives. However, teaching non-experts, such as K-12 students, the complexities of robot planning can be challenging. This work presents an open-source platform, JEDAI.Ed, that simplifies the process using a visual interface that abstracts the details of various planning processes that robots use for performing complex mobile manipulation tasks. Using principles developed in the field of explainable AI, this intuitive platform enables students to use a high-level intuitive instruction set to perform complex tasks, visualize them on an in-built simulator, and to obtain helpful hints and natural language explanations for errors. Finally, JEDAI.Ed, includes an adaptive curriculum generation method that provides students with customized learning ramps. This platform's efficacy was tested through a user study with university students who had little to no computer science background. Our results show that JEDAI.Ed is highly effective in increasing student engagement, teaching robotics programming, and decreasing the time need to solve tasks as compared to baselines.
>
---
#### [replaced 003] Consensus-Driven Uncertainty for Robotic Grasping based on RGB Perception
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.20045v2](http://arxiv.org/pdf/2506.20045v2)**

> **作者:** Eric C. Joyce; Qianwen Zhao; Nathaniel Burgdorfer; Long Wang; Philippos Mordohai
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Deep object pose estimators are notoriously overconfident. A grasping agent that both estimates the 6-DoF pose of a target object and predicts the uncertainty of its own estimate could avoid task failure by choosing not to act under high uncertainty. Even though object pose estimation improves and uncertainty quantification research continues to make strides, few studies have connected them to the downstream task of robotic grasping. We propose a method for training lightweight, deep networks to predict whether a grasp guided by an image-based pose estimate will succeed before that grasp is attempted. We generate training data for our networks via object pose estimation on real images and simulated grasping. We also find that, despite high object variability in grasping trials, networks benefit from training on all objects jointly, suggesting that a diverse variety of objects can nevertheless contribute to the same goal.
>
---
#### [replaced 004] Steering Your Diffusion Policy with Latent Space Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15799v2](http://arxiv.org/pdf/2506.15799v2)**

> **作者:** Andrew Wagenmaker; Mitsuhiko Nakamoto; Yunchu Zhang; Seohong Park; Waleed Yagoub; Anusha Nagabandi; Abhishek Gupta; Sergey Levine
>
> **摘要:** Robotic control policies learned from human demonstrations have achieved impressive results in many real-world applications. However, in scenarios where initial performance is not satisfactory, as is often the case in novel open-world settings, such behavioral cloning (BC)-learned policies typically require collecting additional human demonstrations to further improve their behavior -- an expensive and time-consuming process. In contrast, reinforcement learning (RL) holds the promise of enabling autonomous online policy improvement, but often falls short of achieving this due to the large number of samples it typically requires. In this work we take steps towards enabling fast autonomous adaptation of BC-trained policies via efficient real-world RL. Focusing in particular on diffusion policies -- a state-of-the-art BC methodology -- we propose diffusion steering via reinforcement learning (DSRL): adapting the BC policy by running RL over its latent-noise space. We show that DSRL is highly sample efficient, requires only black-box access to the BC policy, and enables effective real-world autonomous policy improvement. Furthermore, DSRL avoids many of the challenges associated with finetuning diffusion policies, obviating the need to modify the weights of the base policy at all. We demonstrate DSRL on simulated benchmarks, real-world robotic tasks, and for adapting pretrained generalist policies, illustrating its sample efficiency and effective performance at real-world policy improvement.
>
---
#### [replaced 005] ReLink: Computational Circular Design of Planar Linkage Mechanisms Using Available Standard Parts
- **分类: cs.CE; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19657v2](http://arxiv.org/pdf/2506.19657v2)**

> **作者:** Maxime Escande; Kristina Shea
>
> **备注:** 29 pages, 18 figures, Submitted
>
> **摘要:** The Circular Economy framework emphasizes sustainability by reducing resource consumption and waste through the reuse of components and materials. This paper presents ReLink, a computational framework for the circular design of planar linkage mechanisms using available standard parts. Unlike most mechanism design methods, which assume the ability to create custom parts and infinite part availability, ReLink prioritizes the reuse of discrete, standardized components, thus minimizing the need for new parts. The framework consists of two main components: design generation, where a generative design algorithm generates mechanisms from an inventory of available parts, and inverse design, which uses optimization methods to identify designs that match a user-defined trajectory curve. The paper also examines the trade-offs between kinematic performance and CO2 footprint when incorporating new parts. Challenges such as the combinatorial nature of the design problem and the enforcement of valid solutions are addressed. By combining sustainability principles with kinematic synthesis, ReLink lays the groundwork for further research into computational circular design to support the development of systems that integrate reused components into mechanical products.
>
---
#### [replaced 006] 3D Hierarchical Panoptic Segmentation in Real Orchard Environments Across Different Sensors
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.13188v2](http://arxiv.org/pdf/2503.13188v2)**

> **作者:** Matteo Sodano; Federico Magistri; Elias Marks; Fares Hosn; Aibek Zurbayev; Rodrigo Marcuzzi; Meher V. R. Malladi; Jens Behley; Cyrill Stachniss
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Crop yield estimation is a relevant problem in agriculture, because an accurate yield estimate can support farmers' decisions on harvesting or precision intervention. Robots can help to automate this process. To do so, they need to be able to perceive the surrounding environment to identify target objects such as trees and plants. In this paper, we introduce a novel approach to address the problem of hierarchical panoptic segmentation of apple orchards on 3D data from different sensors. Our approach is able to simultaneously provide semantic segmentation, instance segmentation of trunks and fruits, and instance segmentation of trees (a trunk with its fruits). This allows us to identify relevant information such as individual plants, fruits, and trunks, and capture the relationship among them, such as precisely estimate the number of fruits associated to each tree in an orchard. To efficiently evaluate our approach for hierarchical panoptic segmentation, we provide a dataset designed specifically for this task. Our dataset is recorded in Bonn, Germany, in a real apple orchard with a variety of sensors, spanning from a terrestrial laser scanner to a RGB-D camera mounted on different robots platforms. The experiments show that our approach surpasses state-of-the-art approaches in 3D panoptic segmentation in the agricultural domain, while also providing full hierarchical panoptic segmentation. Our dataset is publicly available at https://www.ipb.uni-bonn.de/data/hops/. The open-source implementation of our approach is available at https://github.com/PRBonn/hapt3D.
>
---
#### [replaced 007] Learning Efficient and Robust Language-conditioned Manipulation using Textual-Visual Relevancy and Equivariant Language Mapping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2406.15677v2](http://arxiv.org/pdf/2406.15677v2)**

> **作者:** Mingxi Jia; Haojie Huang; Zhewen Zhang; Chenghao Wang; Linfeng Zhao; Dian Wang; Jason Xinyu Liu; Robin Walters; Robert Platt; Stefanie Tellex
>
> **摘要:** Controlling robots through natural language is pivotal for enhancing human-robot collaboration and synthesizing complex robot behaviors. Recent works that are trained on large robot datasets show impressive generalization abilities. However, such pretrained methods are (1) often fragile to unseen scenarios, and (2) expensive to adapt to new tasks. This paper introduces Grounded Equivariant Manipulation (GEM), a robust yet efficient approach that leverages pretrained vision-language models with equivariant language mapping for language-conditioned manipulation tasks. Our experiments demonstrate GEM's high sample efficiency and generalization ability across diverse tasks in both simulation and the real world. GEM achieves similar or higher performance with orders of magnitude fewer robot data compared with major data-efficient baselines such as CLIPort and VIMA. Finally, our approach demonstrates greater robustness compared to large VLA model, e.g, OpenVLA, at correctly interpreting natural language commands on unseen objects and poses. Code, data, and training details are available https://saulbatman.github.io/gem_page/
>
---
#### [replaced 008] What Foundation Models can Bring for Robot Learning in Manipulation : A Survey
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.18201v4](http://arxiv.org/pdf/2404.18201v4)**

> **作者:** Dingzhe Li; Yixiang Jin; Yuhao Sun; Yong A; Hongze Yu; Jun Shi; Xiaoshuai Hao; Peng Hao; Huaping Liu; Fuchun Sun; Jianwei Zhang; Bin Fang
>
> **摘要:** The realization of universal robots is an ultimate goal of researchers. However, a key hurdle in achieving this goal lies in the robots' ability to manipulate objects in their unstructured surrounding environments according to different tasks. The learning-based approach is considered an effective way to address generalization. The impressive performance of foundation models in the fields of computer vision and natural language suggests the potential of embedding foundation models into manipulation tasks as a viable path toward achieving general manipulation capability. However, we believe achieving general manipulation capability requires an overarching framework akin to auto driving. This framework should encompass multiple functional modules, with different foundation models assuming distinct roles in facilitating general manipulation capability. This survey focuses on the contributions of foundation models to robot learning for manipulation. We propose a comprehensive framework and detail how foundation models can address challenges in each module of the framework. What's more, we examine current approaches, outline challenges, suggest future research directions, and identify potential risks associated with integrating foundation models into this domain.
>
---
#### [replaced 009] PCF-Grasp: Converting Point Completion to Geometry Feature to Enhance 6-DoF Grasp
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.16320v2](http://arxiv.org/pdf/2504.16320v2)**

> **作者:** Yaofeng Cheng; Fusheng Zha; Wei Guo; Pengfei Wang; Chao Zeng; Lining Sun; Chenguang Yang
>
> **摘要:** The 6-Degree of Freedom (DoF) grasp method based on point clouds has shown significant potential in enabling robots to grasp target objects. However, most existing methods are based on the point clouds (2.5D points) generated from single-view depth images. These point clouds only have one surface side of the object providing incomplete geometry information, which mislead the grasping algorithm to judge the shape of the target object, resulting in low grasping accuracy. Humans can accurately grasp objects from a single view by leveraging their geometry experience to estimate object shapes. Inspired by humans, we propose a novel 6-DoF grasping framework that converts the point completion results as object shape features to train the 6-DoF grasp network. Here, point completion can generate approximate complete points from the 2.5D points similar to the human geometry experience, and converting it as shape features is the way to utilize it to improve grasp efficiency. Furthermore, due to the gap between the network generation and actual execution, we integrate a score filter into our framework to select more executable grasp proposals for the real robot. This enables our method to maintain a high grasp quality in any camera viewpoint. Extensive experiments demonstrate that utilizing complete point features enables the generation of significantly more accurate grasp proposals and the inclusion of a score filter greatly enhances the credibility of real-world robot grasping. Our method achieves a 17.8\% success rate higher than the state-of-the-art method in real-world experiments.
>
---
#### [replaced 010] RAMBO: RL-augmented Model-based Whole-body Control for Loco-manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.06662v2](http://arxiv.org/pdf/2504.06662v2)**

> **作者:** Jin Cheng; Dongho Kang; Gabriele Fadini; Guanya Shi; Stelian Coros
>
> **摘要:** Loco-manipulation, physical interaction of various objects that is concurrently coordinated with locomotion, remains a major challenge for legged robots due to the need for both precise end-effector control and robustness to unmodeled dynamics. While model-based controllers provide precise planning via online optimization, they are limited by model inaccuracies. In contrast, learning-based methods offer robustness, but they struggle with precise modulation of interaction forces. We introduce RAMBO, a hybrid framework that integrates model-based whole-body control within a feedback policy trained with reinforcement learning. The model-based module generates feedforward torques by solving a quadratic program, while the policy provides feedback corrective terms to enhance robustness. We validate our framework on a quadruped robot across a diverse set of real-world loco-manipulation tasks, such as pushing a shopping cart, balancing a plate, and holding soft objects, in both quadrupedal and bipedal walking. Our experiments demonstrate that RAMBO enables precise manipulation capabilities while achieving robust and dynamic locomotion.
>
---
#### [replaced 011] EFEAR-4D: Ego-Velocity Filtering for Efficient and Accurate 4D radar Odometry
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.09780v2](http://arxiv.org/pdf/2405.09780v2)**

> **作者:** Xiaoyi Wu; Yushuai Chen; Zhan Li; Ziyang Hong; Liang Hu
>
> **摘要:** Odometry is a crucial component for successfully implementing autonomous navigation, relying on sensors such as cameras, LiDARs and IMUs. However, these sensors may encounter challenges in extreme weather conditions, such as snowfall and fog. The emergence of FMCW radar technology offers the potential for robust perception in adverse conditions. As the latest generation of FWCW radars, the 4D mmWave radar provides point cloud with range, azimuth, elevation, and Doppler velocity information, despite inherent sparsity and noises in the point cloud. In this paper, we propose EFEAR-4D, an accurate, highly efficient, and learning-free method for large-scale 4D radar odometry estimation. EFEAR-4D exploits Doppler velocity information delicately for robust ego-velocity estimation, resulting in a highly accurate prior guess. EFEAR-4D maintains robustness against point-cloud sparsity and noises across diverse environments through dynamic object removal and effective region-wise feature extraction. Extensive experiments on two publicly available 4D radar datasets demonstrate state-of-the-art reliability and localization accuracy of EFEAR-4D under various conditions. Furthermore, we have collected a dataset following the same route but varying installation heights of the 4D radar, emphasizing the significant impact of radar height on point cloud quality - a crucial consideration for real-world deployments. Our algorithm and dataset will be available soon at https://github.com/CLASS-Lab/EFEAR-4D.
>
---
#### [replaced 012] Finding the Easy Way Through -- the Probabilistic Gap Planner for Social Robot Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.20320v2](http://arxiv.org/pdf/2506.20320v2)**

> **作者:** Malte Probst; Raphael Wenzel; Tim Puphal; Monica Dasi; Nico A. Steinhardt; Sango Matsuzaki; Misa Komuro
>
> **摘要:** In Social Robot Navigation, autonomous agents need to resolve many sequential interactions with other agents. State-of-the art planners can efficiently resolve the next, imminent interaction cooperatively and do not focus on longer planning horizons. This makes it hard to maneuver scenarios where the agent needs to select a good strategy to find gaps or channels in the crowd. We propose to decompose trajectory planning into two separate steps: Conflict avoidance for finding good, macroscopic trajectories, and cooperative collision avoidance (CCA) for resolving the next interaction optimally. We propose the Probabilistic Gap Planner (PGP) as a conflict avoidance planner. PGP modifies an established probabilistic collision risk model to include a general assumption of cooperativity. PGP biases the short-term CCA planner to head towards gaps in the crowd. In extensive simulations with crowds of varying density, we show that using PGP in addition to state-of-the-art CCA planners improves the agents' performance: On average, agents keep more space to others, create less tension, and cause fewer collisions. This typically comes at the expense of slightly longer paths. PGP runs in real-time on WaPOCHI mobile robot by Honda R&D.
>
---
#### [replaced 013] ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19815v2](http://arxiv.org/pdf/2506.19815v2)**

> **作者:** Runsheng Wang; Xinyue Zhu; Ava Chen; Jingxi Xu; Lauren Winterbottom; Dawn M. Nilsen; Joel Stein; Matei Ciocarlie
>
> **摘要:** Surface electromyography (sEMG) signals show promise for effective human-computer interfaces, particularly in rehabilitation and prosthetics. However, challenges remain in developing systems that respond quickly and reliably to user intent, across different subjects and without requiring time-consuming calibration. In this work, we propose a framework for EMG-based intent detection that addresses these challenges. Unlike traditional gesture recognition models that wait until a gesture is completed before classifying it, our approach uses a segmentation strategy to assign intent labels at every timestep as the gesture unfolds. We introduce a novel masked modeling strategy that aligns muscle activations with their corresponding user intents, enabling rapid onset detection and stable tracking of ongoing gestures. In evaluations against baseline methods, considering both accuracy and stability for device control, our approach surpasses state-of-the-art performance in zero-shot transfer conditions, demonstrating its potential for wearable robotics and next-generation prosthetic systems. Our project page is available at: https://reactemg.github.io
>
---
#### [replaced 014] CREStE: Scalable Mapless Navigation with Internet Scale Priors and Counterfactual Guidance
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.03921v2](http://arxiv.org/pdf/2503.03921v2)**

> **作者:** Arthur Zhang; Harshit Sikchi; Amy Zhang; Joydeep Biswas
>
> **备注:** 18 pages, 10 figures, 5 tables
>
> **摘要:** We introduce CREStE, a scalable learning-based mapless navigation framework to address the open-world generalization and robustness challenges of outdoor urban navigation. Key to achieving this is learning perceptual representations that generalize to open-set factors (e.g. novel semantic classes, terrains, dynamic entities) and inferring expert-aligned navigation costs from limited demonstrations. CREStE addresses both these issues, introducing 1) a visual foundation model (VFM) distillation objective for learning open-set structured bird's-eye-view perceptual representations, and 2) counterfactual inverse reinforcement learning (IRL), a novel active learning formulation that uses counterfactual trajectory demonstrations to reason about the most important cues when inferring navigation costs. We evaluate CREStE on the task of kilometer-scale mapless navigation in a variety of city, offroad, and residential environments and find that it outperforms all state-of-the-art approaches with 70% fewer human interventions, including a 2-kilometer mission in an unseen environment with just 1 intervention; showcasing its robustness and effectiveness for long-horizon mapless navigation. Videos and additional materials can be found on the project page: https://amrl.cs.utexas.edu/creste
>
---
#### [replaced 015] The Starlink Robot: A Platform and Dataset for Mobile Satellite Communication
- **分类: cs.RO; cs.NI**

- **链接: [http://arxiv.org/pdf/2506.19781v2](http://arxiv.org/pdf/2506.19781v2)**

> **作者:** Boyi Liu; Qianyi Zhang; Qiang Yang; Jianhao Jiao; Jagmohan Chauhan; Dimitrios Kanoulas
>
> **摘要:** The integration of satellite communication into mobile devices represents a paradigm shift in connectivity, yet the performance characteristics under motion and environmental occlusion remain poorly understood. We present the Starlink Robot, the first mobile robotic platform equipped with Starlink satellite internet, comprehensive sensor suite including upward-facing camera, LiDAR, and IMU, designed to systematically study satellite communication performance during movement. Our multi-modal dataset captures synchronized communication metrics, motion dynamics, sky visibility, and 3D environmental context across diverse scenarios including steady-state motion, variable speeds, and different occlusion conditions. This platform and dataset enable researchers to develop motion-aware communication protocols, predict connectivity disruptions, and optimize satellite communication for emerging mobile applications from smartphones to autonomous vehicles. The project is available at https://github.com/StarlinkRobot.
>
---
#### [replaced 016] IMPACT: Behavioral Intention-aware Multimodal Trajectory Prediction with Adaptive Context Trimming
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.09103v2](http://arxiv.org/pdf/2504.09103v2)**

> **作者:** Jiawei Sun; Xibin Yue; Jiahui Li; Tianle Shen; Chengran Yuan; Shuo Sun; Sheng Guo; Quanyun Zhou; Marcelo H Ang Jr
>
> **备注:** under review
>
> **摘要:** While most prior research has focused on improving the precision of multimodal trajectory predictions, the explicit modeling of multimodal behavioral intentions (e.g., yielding, overtaking) remains relatively underexplored. This paper proposes a unified framework that jointly predicts both behavioral intentions and trajectories to enhance prediction accuracy, interpretability, and efficiency. Specifically, we employ a shared context encoder for both intention and trajectory predictions, thereby reducing structural redundancy and information loss. Moreover, we address the lack of ground-truth behavioral intention labels in mainstream datasets (Waymo, Argoverse) by auto-labeling these datasets, thus advancing the community's efforts in this direction. We further introduce a vectorized occupancy prediction module that infers the probability of each map polyline being occupied by the target vehicle's future trajectory. By leveraging these intention and occupancy prediction priors, our method conducts dynamic, modality-dependent pruning of irrelevant agents and map polylines in the decoding stage, effectively reducing computational overhead and mitigating noise from non-critical elements. Our approach ranks first among LiDAR-free methods on the Waymo Motion Dataset and achieves first place on the Waymo Interactive Prediction Dataset. Remarkably, even without model ensembling, our single-model framework improves the soft mean average precision (softmAP) by 10 percent compared to the second-best method in the Waymo Interactive Prediction Leaderboard. Furthermore, the proposed framework has been successfully deployed on real vehicles, demonstrating its practical effectiveness in real-world applications.
>
---
