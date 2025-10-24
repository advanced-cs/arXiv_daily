# 机器人 cs.RO

- **最新发布 32 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] MR-UBi: Mixed Reality-Based Underwater Robot Arm Teleoperation System with Reaction Torque Indicator via Bilateral Control
- **分类: cs.RO**

- **简介: 该论文提出MR-UBi系统，解决水下机械臂遥操作中触觉反馈不足问题。通过混合现实头显叠加力矩指示器，融合视觉与力反馈，提升操作精度与用户体验，显著改善抓取力控制性能和任务效率。**

- **链接: [http://arxiv.org/pdf/2510.20407v1](http://arxiv.org/pdf/2510.20407v1)**

> **作者:** Kohei Nishi; Masato Kobayashi; Yuki Uranishi
>
> **摘要:** We present a mixed reality-based underwater robot arm teleoperation system with a reaction torque indicator via bilateral control (MR-UBi). The reaction torque indicator (RTI) overlays a color and length-coded torque bar in the MR-HMD, enabling seamless integration of visual and haptic feedback during underwater robot arm teleoperation. User studies with sixteen participants compared MR-UBi against a bilateral-control baseline. MR-UBi significantly improved grasping-torque control accuracy, increasing the time within the optimal torque range and reducing both low and high grasping torque range during lift and pick-and-place tasks with objects of different stiffness. Subjective evaluations further showed higher usability (SUS) and lower workload (NASA--TLX). Overall, the results confirm that \textit{MR-UBi} enables more stable, accurate, and user-friendly underwater robot-arm teleoperation through the integration of visual and haptic feedback. For additional material, please check: https://mertcookimg.github.io/mr-ubi
>
---
#### [new 002] Simultaneous learning of state-to-state minimum-time planning and control
- **分类: cs.RO**

- **简介: 该论文针对无人机任意状态间最短时间飞行任务，提出一种基于强化学习的联合规划与控制方法。通过点质量模型轨迹作为代理奖励，并结合课程学习，实现高效训练与泛化能力。仿真与实验证明该方法在复杂环境中具备高速、稳定飞行性能，且可部署于小型嵌入式设备。**

- **链接: [http://arxiv.org/pdf/2510.20008v1](http://arxiv.org/pdf/2510.20008v1)**

> **作者:** Swati Dantu; Robert Pěnička; Martin Saska
>
> **摘要:** This paper tackles the challenge of learning a generalizable minimum-time flight policy for UAVs, capable of navigating between arbitrary start and goal states while balancing agile flight and stable hovering. Traditional approaches, particularly in autonomous drone racing, achieve impressive speeds and agility but are constrained to predefined track layouts, limiting real-world applicability. To address this, we propose a reinforcement learning-based framework that simultaneously learns state-to-state minimum-time planning and control and generalizes to arbitrary state-to-state flights. Our approach leverages Point Mass Model (PMM) trajectories as proxy rewards to approximate the true optimal flight objective and employs curriculum learning to scale the training process efficiently and to achieve generalization. We validate our method through simulation experiments, comparing it against Nonlinear Model Predictive Control (NMPC) tracking PMM-generated trajectories and conducting ablation studies to assess the impact of curriculum learning. Finally, real-world experiments confirm the robustness of our learned policy in outdoor environments, demonstrating its ability to generalize and operate on a small ARM-based single-board computer.
>
---
#### [new 003] Multi-Modal Decentralized Reinforcement Learning for Modular Reconfigurable Lunar Robots
- **分类: cs.RO; cs.MA**

- **简介: 该论文针对模块化可重构月球机器人在复杂形态下的统一控制难题，提出一种去中心化强化学习框架。各模块独立学习策略：轮式模块用SAC优化运动，7自由度机械臂用PPO执行操控。实现在仿真与实地测试中对未见构型的零样本泛化，实现自主移动、转向与初步对齐，系统运行平稳高效。**

- **链接: [http://arxiv.org/pdf/2510.20347v1](http://arxiv.org/pdf/2510.20347v1)**

> **作者:** Ashutosh Mishra; Shreya Santra; Elian Neppel; Edoardo M. Rossi Lombardi; Shamistan Karimov; Kentaro Uno; Kazuya Yoshida
>
> **备注:** Accepted in IEEE iSpaRo 2025. Awaiting Publication
>
> **摘要:** Modular reconfigurable robots suit task-specific space operations, but the combinatorial growth of morphologies hinders unified control. We propose a decentralized reinforcement learning (Dec-RL) scheme where each module learns its own policy: wheel modules use Soft Actor-Critic (SAC) for locomotion and 7-DoF limbs use Proximal Policy Optimization (PPO) for steering and manipulation, enabling zero-shot generalization to unseen configurations. In simulation, the steering policy achieved a mean absolute error of 3.63{\deg} between desired and induced angles; the manipulation policy plateaued at 84.6 % success on a target-offset criterion; and the wheel policy cut average motor torque by 95.4 % relative to baseline while maintaining 99.6 % success. Lunar-analogue field tests validated zero-shot integration for autonomous locomotion, steering, and preliminary alignment for reconfiguration. The system transitioned smoothly among synchronous, parallel, and sequential modes for Policy Execution, without idle states or control conflicts, indicating a scalable, reusable, and robust approach for modular lunar robots.
>
---
#### [new 004] Calibration of Parallel Kinematic Machine Based on Stewart Platform-A Literature Review
- **分类: cs.RO**

- **简介: 该论文属于文献综述任务，聚焦基于Stewart平台的并联机器人（PKM）标定技术。针对高精度3D运动控制需求，研究提出利用逆运动学方法提升定位精度，系统分析了外置仪器、约束及自标定等标定方法，总结了结构误差与环境因素影响下的标定成果，旨在推动该领域进一步发展。**

- **链接: [http://arxiv.org/pdf/2510.20070v1](http://arxiv.org/pdf/2510.20070v1)**

> **作者:** Sourabh Karmakar; Apurva Patel; Cameron J. Turner
>
> **摘要:** Stewart platform-based Parallel Kinematic (PKM) Machines have been extensively studied by researchers due to their inherent finer control characteristics. This has opened its potential deployment opportunities in versatile critical applications like the medical field, engineering machines, space research, electronic chip manufacturing, automobile manufacturing, etc. All these precise, complicated, and repeatable motion applications require micro and nano-scale movement control in 3D space; a 6-DOF PKM can take this challenge smartly. For this, the PKM must be more accurate than the desired application accuracy level and thus proper calibration for a PKM robot is essential. Forward kinematics-based calibration for such hexapod machines becomes unnecessarily complex and inverse kinematics complete this task with much ease. To analyze different techniques, an external instrument-based, constraint-based, and auto or self-calibration-based approaches have been used for calibration. This survey has been done by reviewing these key methodologies, their outcome, and important points related to inverse kinematic-based PKM calibrations in general. It is observed in this study that the researchers focused on improving the accuracy of the platform position and orientation considering the errors contributed by a single source or multiple sources. The error sources considered are mainly structural, in some cases, environmental factors are also considered, however, these calibrations are done under no-load conditions. This study aims to understand the current state of the art in this field and to expand the scope for other researchers in further exploration in a specific area.
>
---
#### [new 005] A Contact-Driven Framework for Manipulating in the Blind
- **分类: cs.RO**

- **简介: 该论文针对视觉受限环境下的机器人抓取任务，提出一种融合接触反馈与结构先验的完整框架。通过接触检测、占用估计与鲁棒规划三模块协同，实现无视觉条件下的高效避障与操作，在仿真与真实机器人上验证了其在复杂家居场景中的优越性能。**

- **链接: [http://arxiv.org/pdf/2510.20177v1](http://arxiv.org/pdf/2510.20177v1)**

> **作者:** Muhammad Suhail Saleem; Lai Yuan; Maxim Likhachev
>
> **摘要:** Robots often face manipulation tasks in environments where vision is inadequate due to clutter, occlusions, or poor lighting--for example, reaching a shutoff valve at the back of a sink cabinet or locating a light switch above a crowded shelf. In such settings, robots, much like humans, must rely on contact feedback to distinguish free from occupied space and navigate around obstacles. Many of these environments often exhibit strong structural priors--for instance, pipes often span across sink cabinets--that can be exploited to anticipate unseen structure and avoid unnecessary collisions. We present a theoretically complete and empirically efficient framework for manipulation in the blind that integrates contact feedback with structural priors to enable robust operation in unknown environments. The framework comprises three tightly coupled components: (i) a contact detection and localization module that utilizes joint torque sensing with a contact particle filter to detect and localize contacts, (ii) an occupancy estimation module that uses the history of contact observations to build a partial occupancy map of the workspace and extrapolate it into unexplored regions with learned predictors, and (iii) a planning module that accounts for the fact that contact localization estimates and occupancy predictions can be noisy, computing paths that avoid collisions and complete tasks efficiently without eliminating feasible solutions. We evaluate the system in simulation and in the real world on a UR10e manipulator across two domestic tasks--(i) manipulating a valve under a kitchen sink surrounded by pipes and (ii) retrieving a target object from a cluttered shelf. Results show that the framework reliably solves these tasks, achieving up to a 2x reduction in task completion time compared to baselines, with ablations confirming the contribution of each module.
>
---
#### [new 006] VAMOS: A Hierarchical Vision-Language-Action Model for Capability-Modulated and Steerable Navigation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出VAMOS，一种分层视觉-语言-动作模型，用于解决机器人在不同物理形态下跨环境导航的泛化与可行性问题。通过解耦语义规划与身体约束建模，实现高成功率、可语言控制的导航，并支持轮式与足式机器人跨体态部署。**

- **链接: [http://arxiv.org/pdf/2510.20818v1](http://arxiv.org/pdf/2510.20818v1)**

> **作者:** Mateo Guaman Castro; Sidharth Rajagopal; Daniel Gorbatov; Matt Schmittle; Rohan Baijal; Octi Zhang; Rosario Scalise; Sidharth Talia; Emma Romig; Celso de Melo; Byron Boots; Abhishek Gupta
>
> **摘要:** A fundamental challenge in robot navigation lies in learning policies that generalize across diverse environments while conforming to the unique physical constraints and capabilities of a specific embodiment (e.g., quadrupeds can walk up stairs, but rovers cannot). We propose VAMOS, a hierarchical VLA that decouples semantic planning from embodiment grounding: a generalist planner learns from diverse, open-world data, while a specialist affordance model learns the robot's physical constraints and capabilities in safe, low-cost simulation. We enabled this separation by carefully designing an interface that lets a high-level planner propose candidate paths directly in image space that the affordance model then evaluates and re-ranks. Our real-world experiments show that VAMOS achieves higher success rates in both indoor and complex outdoor navigation than state-of-the-art model-based and end-to-end learning methods. We also show that our hierarchical design enables cross-embodied navigation across legged and wheeled robots and is easily steerable using natural language. Real-world ablations confirm that the specialist model is key to embodiment grounding, enabling a single high-level planner to be deployed across physically distinct wheeled and legged robots. Finally, this model significantly enhances single-robot reliability, achieving 3X higher success rates by rejecting physically infeasible plans. Website: https://vamos-vla.github.io/
>
---
#### [new 007] Real-Time Gait Adaptation for Quadrupeds using Model Predictive Control and Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对四足机器人在复杂环境中实时适应不同步态的问题，结合MPPI与Dreamer模块，实现基于强化学习的连续步态空间优化。通过联合优化动作与步态变量，提升速度跟踪精度、能效与稳定性，显著降低能耗并实现自适应步态切换。**

- **链接: [http://arxiv.org/pdf/2510.20706v1](http://arxiv.org/pdf/2510.20706v1)**

> **作者:** Ganga Nair B; Prakrut Kotecha; Shishir Kolathaya
>
> **摘要:** Model-free reinforcement learning (RL) has enabled adaptable and agile quadruped locomotion; however, policies often converge to a single gait, leading to suboptimal performance. Traditionally, Model Predictive Control (MPC) has been extensively used to obtain task-specific optimal policies but lacks the ability to adapt to varying environments. To address these limitations, we propose an optimization framework for real-time gait adaptation in a continuous gait space, combining the Model Predictive Path Integral (MPPI) algorithm with a Dreamer module to produce adaptive and optimal policies for quadruped locomotion. At each time step, MPPI jointly optimizes the actions and gait variables using a learned Dreamer reward that promotes velocity tracking, energy efficiency, stability, and smooth transitions, while penalizing abrupt gait changes. A learned value function is incorporated as terminal reward, extending the formulation to an infinite-horizon planner. We evaluate our framework in simulation on the Unitree Go1, demonstrating an average reduction of up to 36.48\% in energy consumption across varying target speeds, while maintaining accurate tracking and adaptive, task-appropriate gaits.
>
---
#### [new 008] Simultaneous Stiffness and Trajectory Optimization for Energy Minimization of Pick-and-Place Tasks of SEA-Actuated Parallel Kinematic Manipulators
- **分类: cs.RO; math.DS**

- **简介: 该论文研究串联弹性执行器（SEA）驱动的并联机器人在拾放任务中的能耗优化问题。针对长期重复运行的拾放任务，提出同时优化运动轨迹与SEA刚度的协同优化方法，利用弹簧振荡特性降低能耗。通过建立动力学模型并求解最优控制问题，在两个机器人案例中验证了该方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.20490v1](http://arxiv.org/pdf/2510.20490v1)**

> **作者:** Thomas Kordik; Hubert Gattringer; Andreas Mueller
>
> **摘要:** A major field of industrial robot applications deals with repetitive tasks that alternate between operating points. For these so-called pick-and-place operations, parallel kinematic manipulators (PKM) are frequently employed. These tasks tend to automatically run for a long period of time and therefore minimizing energy consumption is always of interest. Recent research addresses this topic by the use of elastic elements and particularly series elastic actuators (SEA). This paper explores the possibilities of minimizing energy consumption of SEA actuated PKM performing pick-and-place tasks. The basic idea is to excite eigenmotions that result from the actuator springs and exploit their oscillating characteristics. To this end, a prescribed cyclic pick-and-place operation is analyzed and a dynamic model of SEA driven PKM is derived. Subsequently, an energy minimizing optimal control problem is formulated where operating trajectories as well as SEA stiffnesses are optimized simultaneously. Here, optimizing the actuator stiffness does not account for variable stiffness actuators. It serves as a tool for the design and dimensioning process. The hypothesis on energy reduction is tested on two (parallel) robot applications where redundant actuation is also addressed. The results confirm the validity of this approach.
>
---
#### [new 009] PointMapPolicy: Structured Point Cloud Processing for Multi-Modal Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人操作中多模态感知的精度与泛化问题，提出PointMapPolicy方法。通过结构化点云网格融合RGB信息，利用xLSTM实现高效多模态特征融合，提升几何与语义理解能力，在复杂操作任务中实现最优性能。**

- **链接: [http://arxiv.org/pdf/2510.20406v1](http://arxiv.org/pdf/2510.20406v1)**

> **作者:** Xiaogang Jia; Qian Wang; Anrui Wang; Han A. Wang; Balázs Gyenes; Emiliyan Gospodinov; Xinkai Jiang; Ge Li; Hongyi Zhou; Weiran Liao; Xi Huang; Maximilian Beck; Moritz Reuss; Rudolf Lioutikov; Gerhard Neumann
>
> **摘要:** Robotic manipulation systems benefit from complementary sensing modalities, where each provides unique environmental information. Point clouds capture detailed geometric structure, while RGB images provide rich semantic context. Current point cloud methods struggle to capture fine-grained detail, especially for complex tasks, which RGB methods lack geometric awareness, which hinders their precision and generalization. We introduce PointMapPolicy, a novel approach that conditions diffusion policies on structured grids of points without downsampling. The resulting data type makes it easier to extract shape and spatial relationships from observations, and can be transformed between reference frames. Yet due to their structure in a regular grid, we enable the use of established computer vision techniques directly to 3D data. Using xLSTM as a backbone, our model efficiently fuses the point maps with RGB data for enhanced multi-modal perception. Through extensive experiments on the RoboCasa and CALVIN benchmarks and real robot evaluations, we demonstrate that our method achieves state-of-the-art performance across diverse manipulation tasks. The overview and demos are available on our project page: https://point-map.github.io/Point-Map/
>
---
#### [new 010] NODA-MMH: Certified Learning-Aided Nonlinear Control for Magnetically-Actuated Swarm Experiment Toward On-Orbit Proof
- **分类: cs.RO**

- **简介: 该论文研究磁力驱动卫星集群的非线性控制问题，针对多星编队长期维持中的非完整约束、欠驱动、可扩展性与计算成本难题，提出基于学习的时序电流控制与神经优化分配方法（NODA-MMH），在地面空气轴承平台上验证了其可控性与功率最优性。**

- **链接: [http://arxiv.org/pdf/2510.20231v1](http://arxiv.org/pdf/2510.20231v1)**

> **作者:** Yuta Takahashi; Atsuki Ochi; Yoichi Tomioka; Shin-Ichiro Sakai
>
> **备注:** Accepted for presentation at the 2025 International Conference on Space Robotics (iSpaRo 2025)
>
> **摘要:** This study experimentally validates the principle of large-scale satellite swarm control through learning-aided magnetic field interactions generated by satellite-mounted magnetorquers. This actuation presents a promising solution for the long-term formation maintenance of multiple satellites and has primarily been demonstrated in ground-based testbeds for two-satellite position control. However, as the number of satellites increases beyond three, fundamental challenges coupled with the high nonlinearity arise: 1) nonholonomic constraints, 2) underactuation, 3) scalability, and 4) computational cost. Previous studies have shown that time-integrated current control theoretically solves these problems, where the average actuator outputs align with the desired command, and a learning-based technique further enhances their performance. Through multiple experiments, we validate critical aspects of learning-aided time-integrated current control: (1) enhanced controllability of the averaged system dynamics, with a theoretically guaranteed error bound, and (2) decentralized current management. We design two-axis coils and a ground-based experimental setup utilizing an air-bearing platform, enabling a mathematical replication of orbital dynamics. Based on the effectiveness of the learned interaction model, we introduce NODA-MMH (Neural power-Optimal Dipole Allocation for certified learned Model-based Magnetically swarm control Harness) for model-based power-optimal swarm control. This study complements our tutorial paper on magnetically actuated swarms for the long-term formation maintenance problem.
>
---
#### [new 011] Degradation-Aware Cooperative Multi-Modal GNSS-Denied Localization Leveraging LiDAR-Based Robot Detections
- **分类: cs.RO**

- **简介: 该论文针对GNSS拒止环境下机器人长期高精度定位问题，提出一种自适应多模态多机器人协同定位方法。通过融合异步视觉-惯性、激光雷达-惯性里程计及机器人间3D检测数据，利用因子图与插值因子实现松耦合融合，动态权重调整提升抗传感器退化能力，显著改善复杂环境下的定位精度。**

- **链接: [http://arxiv.org/pdf/2510.20480v1](http://arxiv.org/pdf/2510.20480v1)**

> **作者:** Václav Pritzl; Xianjia Yu; Tomi Westerlund; Petr Štěpán; Martin Saska
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Accurate long-term localization using onboard sensors is crucial for robots operating in Global Navigation Satellite System (GNSS)-denied environments. While complementary sensors mitigate individual degradations, carrying all the available sensor types on a single robot significantly increases the size, weight, and power demands. Distributing sensors across multiple robots enhances the deployability but introduces challenges in fusing asynchronous, multi-modal data from independently moving platforms. We propose a novel adaptive multi-modal multi-robot cooperative localization approach using a factor-graph formulation to fuse asynchronous Visual-Inertial Odometry (VIO), LiDAR-Inertial Odometry (LIO), and 3D inter-robot detections from distinct robots in a loosely-coupled fashion. The approach adapts to changing conditions, leveraging reliable data to assist robots affected by sensory degradations. A novel interpolation-based factor enables fusion of the unsynchronized measurements. LIO degradations are evaluated based on the approximate scan-matching Hessian. A novel approach of weighting odometry data proportionally to the Wasserstein distance between the consecutive VIO outputs is proposed. A theoretical analysis is provided, investigating the cooperative localization problem under various conditions, mainly in the presence of sensory degradations. The proposed method has been extensively evaluated on real-world data gathered with heterogeneous teams of an Unmanned Ground Vehicle (UGV) and Unmanned Aerial Vehicles (UAVs), showing that the approach provides significant improvements in localization accuracy in the presence of various sensory degradations.
>
---
#### [new 012] Dino-Diffusion Modular Designs Bridge the Cross-Domain Gap in Autonomous Parking
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦自动驾驶泊车任务，针对域偏移下模型泛化能力差的问题，提出Dino-Diffusion Parking（DDP）框架。通过融合视觉基础模型与扩散规划，实现跨域感知与鲁棒决策，在零样本迁移下保持90%以上成功率，并验证了良好仿真到现实的迁移能力。**

- **链接: [http://arxiv.org/pdf/2510.20335v1](http://arxiv.org/pdf/2510.20335v1)**

> **作者:** Zixuan Wu; Hengyuan Zhang; Ting-Hsuan Chen; Yuliang Guo; David Paz; Xinyu Huang; Liu Ren
>
> **备注:** Code is at https://github.com/ChampagneAndfragrance/Dino_Diffusion_Parking_Official
>
> **摘要:** Parking is a critical pillar of driving safety. While recent end-to-end (E2E) approaches have achieved promising in-domain results, robustness under domain shifts (e.g., weather and lighting changes) remains a key challenge. Rather than relying on additional data, in this paper, we propose Dino-Diffusion Parking (DDP), a domain-agnostic autonomous parking pipeline that integrates visual foundation models with diffusion-based planning to enable generalized perception and robust motion planning under distribution shifts. We train our pipeline in CARLA at regular setting and transfer it to more adversarial settings in a zero-shot fashion. Our model consistently achieves a parking success rate above 90% across all tested out-of-distribution (OOD) scenarios, with ablation studies confirming that both the network architecture and algorithmic design significantly enhance cross-domain performance over existing baselines. Furthermore, testing in a 3D Gaussian splatting (3DGS) environment reconstructed from a real-world parking lot demonstrates promising sim-to-real transfer.
>
---
#### [new 013] Robot Path and Trajectory Planning Considering a Spatially Fixed TCP
- **分类: cs.RO**

- **简介: 该论文针对工业机器人在固定工具中心点（TCP）下的路径规划问题，提出基于B-splines的轨迹规划方法。通过考虑工件加工路径与空间固定TCP，实现平滑连续的机器人运动轨迹，兼顾指定姿态与TCP速度。实验验证了方法在实际系统中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.20473v1](http://arxiv.org/pdf/2510.20473v1)**

> **作者:** Bernhard Rameder; Hubert Gattringer; Andreas Mueller; Ronald Naderer
>
> **摘要:** This paper presents a method for planning a trajectory in workspace coordinates using a spatially fixed tool center point (TCP), while taking into account the processing path on a part. This approach is beneficial if it is easier to move the part rather than moving the tool. Whether a mathematical description that defines the shape to be processed or single points from a design program are used, the robot path is finally represented using B-splines. The use of splines enables the path to be continuous with a desired degree, which finally leads to a smooth robot trajectory. While calculating the robot trajectory through prescribed orientation, additionally a given velocity at the TCP has to be considered. The procedure was validated on a real system using an industrial robot moving an arbitrary defined part.
>
---
#### [new 014] Configuration-Dependent Robot Kinematics Model and Calibration
- **分类: cs.RO**

- **简介: 该论文针对关节型机器人在复杂工作空间中因非几何因素导致的配置依赖性误差问题，提出一种基于局部POE模型与傅里叶基函数插值的全局运动学标定框架。通过多配置参数识别与高效插值，显著提升整体定位精度，验证表明最大误差降低超50%，适用于高精度冷喷制造等场景。**

- **链接: [http://arxiv.org/pdf/2510.19962v1](http://arxiv.org/pdf/2510.19962v1)**

> **作者:** Chen-Lung Lu; Honglu He; Agung Julius; John T. Wen
>
> **摘要:** Accurate robot kinematics is essential for precise tool placement in articulated robots, but non-geometric factors can introduce configuration-dependent model discrepancies. This paper presents a configuration-dependent kinematic calibration framework for improving accuracy across the entire workspace. Local Product-of-Exponential (POE) models, selected for their parameterization continuity, are identified at multiple configurations and interpolated into a global model. Inspired by joint gravity load expressions, we employ Fourier basis function interpolation parameterized by the shoulder and elbow joint angles, achieving accuracy comparable to neural network and autoencoder methods but with substantially higher training efficiency. Validation on two 6-DoF industrial robots shows that the proposed approach reduces the maximum positioning error by over 50%, meeting the sub-millimeter accuracy required for cold spray manufacturing. Robots with larger configuration-dependent discrepancies benefit even more. A dual-robot collaborative task demonstrates the framework's practical applicability and repeatability.
>
---
#### [new 015] GSWorld: Closed-Loop Photo-Realistic Simulation Suite for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出GSWorld，一个用于机器人操作的闭环真实感仿真系统。针对真实场景数据获取难、仿真实现与现实差距大的问题，融合3D高斯点云与物理引擎，构建GSDF资产格式，实现高保真渲染与可复现评估，支持零样本模拟到现实的视觉策略学习与数据采集。**

- **链接: [http://arxiv.org/pdf/2510.20813v1](http://arxiv.org/pdf/2510.20813v1)**

> **作者:** Guangqi Jiang; Haoran Chang; Ri-Zhao Qiu; Yutong Liang; Mazeyu Ji; Jiyue Zhu; Zhao Dong; Xueyan Zou; Xiaolong Wang
>
> **摘要:** This paper presents GSWorld, a robust, photo-realistic simulator for robotics manipulation that combines 3D Gaussian Splatting with physics engines. Our framework advocates "closing the loop" of developing manipulation policies with reproducible evaluation of policies learned from real-robot data and sim2real policy training without using real robots. To enable photo-realistic rendering of diverse scenes, we propose a new asset format, which we term GSDF (Gaussian Scene Description File), that infuses Gaussian-on-Mesh representation with robot URDF and other objects. With a streamlined reconstruction pipeline, we curate a database of GSDF that contains 3 robot embodiments for single-arm and bimanual manipulation, as well as more than 40 objects. Combining GSDF with physics engines, we demonstrate several immediate interesting applications: (1) learning zero-shot sim2real pixel-to-action manipulation policy with photo-realistic rendering, (2) automated high-quality DAgger data collection for adapting policies to deployment environments, (3) reproducible benchmarking of real-robot manipulation policies in simulation, (4) simulation data collection by virtual teleoperation, and (5) zero-shot sim2real visual reinforcement learning. Website: https://3dgsworld.github.io/.
>
---
#### [new 016] A Parameter-Linear Formulation of the Optimal Path Following Problem for Robotic Manipulator
- **分类: cs.RO; math.DS**

- **简介: 该论文研究机器人机械臂的时间最优路径跟踪问题。针对传统方法在路径参数化时出现零速度奇异性的计算难题，提出一种基于最大化路径速度的线性参数化方法，实现平滑轨迹的高效数值规划，显著降低计算复杂度。**

- **链接: [http://arxiv.org/pdf/2510.20496v1](http://arxiv.org/pdf/2510.20496v1)**

> **作者:** Tobias Marauli; Hubert Gattringer; Andreas Mueller
>
> **摘要:** In this paper the computational challenges of time-optimal path following are addressed. The standard approach is to minimize the travel time, which inevitably leads to singularities at zero path speed, when reformulating the optimization problem in terms of a path parameter. Thus, smooth trajectory generation while maintaining a low computational effort is quite challenging, since the singularities have to be taken into account. To this end, a different approach is presented in this paper. This approach is based on maximizing the path speed along a prescribed path. Furthermore, the approach is capable of planning smooth trajectories numerically efficient. Moreover, the discrete reformulation of the underlying problem is linear in optimization variables.
>
---
#### [new 017] FieldGen: From Teleoperated Pre-Manipulation Trajectories to Field-Guided Data Generation
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文提出FieldGen框架，用于高效生成高质量机器人操作数据。针对现有方法在数据规模、多样性和质量间的权衡难题，通过分解操作为预操作与精细操作阶段，结合人类示范与吸引力场生成多样化轨迹，实现低人力成本下的高质数据采集，并引入奖励标注提升策略学习效果。**

- **链接: [http://arxiv.org/pdf/2510.20774v1](http://arxiv.org/pdf/2510.20774v1)**

> **作者:** Wenhao Wang; Kehe Ye; Xinyu Zhou; Tianxing Chen; Cao Min; Qiaoming Zhu; Xiaokang Yang; Yongjian Shen; Yang Yang; Maoqing Yao; Yao Mu
>
> **备注:** Webpage: https://fieldgen.github.io/
>
> **摘要:** Large-scale and diverse datasets are vital for training robust robotic manipulation policies, yet existing data collection methods struggle to balance scale, diversity, and quality. Simulation offers scalability but suffers from sim-to-real gaps, while teleoperation yields high-quality demonstrations with limited diversity and high labor cost. We introduce FieldGen, a field-guided data generation framework that enables scalable, diverse, and high-quality real-world data collection with minimal human supervision. FieldGen decomposes manipulation into two stages: a pre-manipulation phase, allowing trajectory diversity, and a fine manipulation phase requiring expert precision. Human demonstrations capture key contact and pose information, after which an attraction field automatically generates diverse trajectories converging to successful configurations. This decoupled design combines scalable trajectory diversity with precise supervision. Moreover, FieldGen-Reward augments generated data with reward annotations to further enhance policy learning. Experiments demonstrate that policies trained with FieldGen achieve higher success rates and improved stability compared to teleoperation-based baselines, while significantly reducing human effort in long-term real-world data collection. Webpage is available at https://fieldgen.github.io/.
>
---
#### [new 018] The Reality Gap in Robotics: Challenges, Solutions, and Best Practices
- **分类: cs.RO; cs.AI; cs.LG; stat.ML; I.2.6; I.2.8; I.2.9**

- **简介: 该论文聚焦于机器人领域的“现实差距”问题，即仿真与现实环境间的差异导致模型迁移失败。针对此问题，系统综述了根源、解决方案（如域随机化、协同训练）及评估指标，旨在推动仿真到现实的高效迁移。**

- **链接: [http://arxiv.org/pdf/2510.20808v1](http://arxiv.org/pdf/2510.20808v1)**

> **作者:** Elie Aljalbout; Jiaxu Xing; Angel Romero; Iretiayo Akinola; Caelan Reed Garrett; Eric Heiden; Abhishek Gupta; Tucker Hermans; Yashraj Narang; Dieter Fox; Davide Scaramuzza; Fabio Ramos
>
> **备注:** Accepted for Publication as part of the Annual Review of Control, Robotics, and Autonomous Systems 2026
>
> **摘要:** Machine learning has facilitated significant advancements across various robotics domains, including navigation, locomotion, and manipulation. Many such achievements have been driven by the extensive use of simulation as a critical tool for training and testing robotic systems prior to their deployment in real-world environments. However, simulations consist of abstractions and approximations that inevitably introduce discrepancies between simulated and real environments, known as the reality gap. These discrepancies significantly hinder the successful transfer of systems from simulation to the real world. Closing this gap remains one of the most pressing challenges in robotics. Recent advances in sim-to-real transfer have demonstrated promising results across various platforms, including locomotion, navigation, and manipulation. By leveraging techniques such as domain randomization, real-to-sim transfer, state and action abstractions, and sim-real co-training, many works have overcome the reality gap. However, challenges persist, and a deeper understanding of the reality gap's root causes and solutions is necessary. In this survey, we present a comprehensive overview of the sim-to-real landscape, highlighting the causes, solutions, and evaluation metrics for the reality gap and sim-to-real transfer.
>
---
#### [new 019] Push Anything: Single- and Multi-Object Pushing From First Sight with Contact-Implicit MPC
- **分类: cs.RO**

- **简介: 该论文研究非抓取式物体推移任务，针对未知物体物理属性与复杂接触交互的挑战，提出C3+算法，实现单/多物体精准推移。通过集成扫描、重建与控制全流程，在硬件上达成98%成功率，显著提升求解速度与实时性。**

- **链接: [http://arxiv.org/pdf/2510.19974v1](http://arxiv.org/pdf/2510.19974v1)**

> **作者:** Hien Bui; Yufeiyang Gao; Haoran Yang; Eric Cui; Siddhant Mody; Brian Acosta; Thomas Stephen Felix; Bibit Bianchini; Michael Posa
>
> **备注:** Hien Bui, Yufeiyang Gao, and Haoran Yang contributed equally to this work
>
> **摘要:** Non-prehensile manipulation of diverse objects remains a core challenge in robotics, driven by unknown physical properties and the complexity of contact-rich interactions. Recent advances in contact-implicit model predictive control (CI-MPC), with contact reasoning embedded directly in the trajectory optimization, have shown promise in tackling the task efficiently and robustly, yet demonstrations have been limited to narrowly curated examples. In this work, we showcase the broader capabilities of CI-MPC through precise planar pushing tasks over a wide range of object geometries, including multi-object domains. These scenarios demand reasoning over numerous inter-object and object-environment contacts to strategically manipulate and de-clutter the environment, challenges that were intractable for prior CI-MPC methods. To achieve this, we introduce Consensus Complementarity Control Plus (C3+), an enhanced CI-MPC algorithm integrated into a complete pipeline spanning object scanning, mesh reconstruction, and hardware execution. Compared to its predecessor C3, C3+ achieves substantially faster solve times, enabling real-time performance even in multi-object pushing tasks. On hardware, our system achieves overall 98% success rate across 33 objects, reaching pose goals within tight tolerances. The average time-to-goal is approximately 0.5, 1.6, 3.2, and 5.3 minutes for 1-, 2-, 3-, and 4-object tasks, respectively. Project page: https://dairlab.github.io/push-anything.
>
---
#### [new 020] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World
- **分类: cs.RO**

- **简介: 该论文聚焦于开放世界中的持续物体导航任务，旨在解决智能体在动态环境中学习新物体导航技能时易遗忘旧知识的问题。提出C-Nav框架，通过双路径抗遗忘机制与自适应采样策略，实现持续学习下的高效导航，显著提升性能并降低内存开销。**

- **链接: [http://arxiv.org/pdf/2510.20685v1](http://arxiv.org/pdf/2510.20685v1)**

> **作者:** Ming-Ming Yu; Fei Zhu; Wenzhuo Liu; Yirong Yang; Qunbo Wang; Wenjun Wu; Jing Liu
>
> **摘要:** Embodied agents are expected to perform object navigation in dynamic, open-world environments. However, existing approaches typically rely on static trajectories and a fixed set of object categories during training, overlooking the real-world requirement for continual adaptation to evolving scenarios. To facilitate related studies, we introduce the continual object navigation benchmark, which requires agents to acquire navigation skills for new object categories while avoiding catastrophic forgetting of previously learned knowledge. To tackle this challenge, we propose C-Nav, a continual visual navigation framework that integrates two key innovations: (1) A dual-path anti-forgetting mechanism, which comprises feature distillation that aligns multi-modal inputs into a consistent representation space to ensure representation consistency, and feature replay that retains temporal features within the action decoder to ensure policy consistency. (2) An adaptive sampling strategy that selects diverse and informative experiences, thereby reducing redundancy and minimizing memory overhead. Extensive experiments across multiple model architectures demonstrate that C-Nav consistently outperforms existing approaches, achieving superior performance even compared to baselines with full trajectory retention, while significantly lowering memory requirements. The code will be publicly available at https://bigtree765.github.io/C-Nav-project.
>
---
#### [new 021] Dual Control Reference Generation for Optimal Pick-and-Place Execution under Payload Uncertainty
- **分类: cs.RO; cs.IT; math.IT**

- **简介: 论文针对负载不确定下的抓取与放置任务，解决未知动态下的精准控制问题。通过预设带自适应机制的反馈策略，提出两种参考轨迹生成方法：基于期望代价最小化的鲁棒优化与基于最优性损失最小化的信息敏感度优化，均利用费舍尔信息实现任务执行与系统辨识的协同优化，提升控制精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.20483v1](http://arxiv.org/pdf/2510.20483v1)**

> **作者:** Victor Vantilborgh; Hrishikesh Sathyanarayan; Guillaume Crevecoeur; Ian Abraham; Tom Lefebvre
>
> **摘要:** This work addresses the problem of robot manipulation tasks under unknown dynamics, such as pick-and-place tasks under payload uncertainty, where active exploration and(/for) online parameter adaptation during task execution are essential to enable accurate model-based control. The problem is framed as dual control seeking a closed-loop optimal control problem that accounts for parameter uncertainty. We simplify the dual control problem by pre-defining the structure of the feedback policy to include an explicit adaptation mechanism. Then we propose two methods for reference trajectory generation. The first directly embeds parameter uncertainty in robust optimal control methods that minimize the expected task cost. The second method considers minimizing the so-called optimality loss, which measures the sensitivity of parameter-relevant information with respect to task performance. We observe that both approaches reason over the Fisher information as a natural side effect of their formulations, simultaneously pursuing optimal task execution. We demonstrate the effectiveness of our approaches for a pick-and-place manipulation task. We show that designing the reference trajectories whilst taking into account the control enables faster and more accurate task performance and system identification while ensuring stable and efficient control.
>
---
#### [new 022] Design of a Bed Rotation Mechanism to Facilitate In-Situ Photogrammetric Reconstruction of Printed Parts
- **分类: cs.RO**

- **简介: 该论文属于3D打印工艺研究任务，旨在解决消费级打印机缺乏可重复性与可监控性的问题。作者设计并实现了一种可旋转加热床的机构，配合少相机数量的原位摄影测量系统，实时记录打印过程中的几何变化，以关联工艺参数与成型缺陷。**

- **链接: [http://arxiv.org/pdf/2510.20079v1](http://arxiv.org/pdf/2510.20079v1)**

> **作者:** Travis A. Roberts; Sourabh Karmakar; Cameron J. Turner
>
> **摘要:** Additive manufacturing, or 3D printing, is a complex process that creates free-form geometric objects by sequentially placing material to construct an object, usually in a layer-by-layer process. One of the most widely used methods is Fused Deposition Modeling (FDM). FDM is used in many of the consumer-grade polymer 3D printers available today. While consumer grade machines are cheap and plentiful, they lack many of the features desired in a machine used for research purposes and are often closed-source platforms. Commercial-grade models are more expensive and are also usually closed-source platforms that do not offer flexibility for modifications often needed for research. The authors designed and fabricated a machine to be used as a test bed for research in the field of polymer FDM processes. The goal was to create a platform that tightly controls and/or monitors the FDM build parameters so that experiments can be repeated with a known accuracy. The platform offers closed loop position feedback, control of the hot end and bed temperature, and monitoring of environment temperature and humidity. Additionally, the platform is equipped with cameras and a mechanism for in-situ photogrammetry, creating a geometric record of the printing throughout the printing process. Through photogrammetry, backtracking and linking process parameters to observable geometric defects can be achieved. This paper focuses on the design of a novel mechanism for spinning the heated bed to allow for photogrammetric reconstruction of the printed part using a minimal number of cameras, as implemented on this platform.
>
---
#### [new 023] RubbleSim: A Photorealistic Structural Collapse Simulator for Confined Space Mapping
- **分类: cs.RO**

- **简介: 该论文提出RubbleSim，一个用于模拟受限空间坍塌结构的开源、可重构的光真实感仿真器。针对灾后搜救中内部空隙数据难以获取的问题，通过物理引擎生成随机碎石堆，提供精确的地面真值。利用该平台评估视觉算法在复杂环境下的性能退化，支持多系统运行，代码公开。**

- **链接: [http://arxiv.org/pdf/2510.20529v1](http://arxiv.org/pdf/2510.20529v1)**

> **作者:** Constantine Frost; Chad Council; Margaret McGuinness; Nathaniel Hanson
>
> **备注:** Accepted to 2025 IEEE International Symposium on Safety, Security, and Rescue Robotics
>
> **摘要:** Despite well-reported instances of robots being used in disaster response, there is scant published data on the internal composition of the void spaces within structural collapse incidents. Data collected during these incidents is mired in legal constraints, as ownership is often tied to the responding agencies, with little hope of public release for research. While engineered rubble piles are used for training, these sites are also reluctant to release information about their proprietary training grounds. To overcome this access challenge, we present RubbleSim -- an open-source, reconfigurable simulator for photorealistic void space exploration. The design of the simulation assets is directly informed by visits to numerous training rubble sites at differing levels of complexity. The simulator is implemented in Unity with multi-operating system support. The simulation uses a physics-based approach to build stochastic rubble piles, allowing for rapid iteration between simulation worlds while retaining absolute knowledge of the ground truth. Using RubbleSim, we apply a state-of-the-art structure-from-motion algorithm to illustrate how perception performance degrades under challenging visual conditions inside the emulated void spaces. Pre-built binaries and source code to implement are available online: https://github.com/mit-ll/rubble_pile_simulator.
>
---
#### [new 024] Kinaema: a recurrent sequence model for memory and pose in motion
- **分类: cs.RO; cs.CV; I.2.10**

- **简介: 该论文提出Kinaema模型，用于机器人在连续运动中实现空间定位。针对机器人需基于先前观测高效导航的问题，设计了一种无显式记忆的循环序列模型，通过递归更新隐式记忆，压缩视觉流并响应查询图像预测相对位置。实验表明其在“Mem-Nav”任务中具高效性与强表征能力。**

- **链接: [http://arxiv.org/pdf/2510.20261v1](http://arxiv.org/pdf/2510.20261v1)**

> **作者:** Mert Bulent Sariyildiz; Philippe Weinzaepfel; Guillaume Bono; Gianluca Monaci; Christian Wolf
>
> **备注:** 10 pages + references + checklist + appendix, 29 pages total
>
> **摘要:** One key aspect of spatially aware robots is the ability to "find their bearings", ie. to correctly situate themselves in previously seen spaces. In this work, we focus on this particular scenario of continuous robotics operations, where information observed before an actual episode start is exploited to optimize efficiency. We introduce a new model, Kinaema, and agent, capable of integrating a stream of visual observations while moving in a potentially large scene, and upon request, processing a query image and predicting the relative position of the shown space with respect to its current position. Our model does not explicitly store an observation history, therefore does not have hard constraints on context length. It maintains an implicit latent memory, which is updated by a transformer in a recurrent way, compressing the history of sensor readings into a compact representation. We evaluate the impact of this model in a new downstream task we call "Mem-Nav". We show that our large-capacity recurrent model maintains a useful representation of the scene, navigates to goals observed before the actual episode start, and is computationally efficient, in particular compared to classical transformers with attention over an observation history.
>
---
#### [new 025] PathFormer: A Transformer with 3D Grid Constraints for Digital Twin Robot-Arm Trajectory Generation
- **分类: cs.RO; 68T07, 68T40; I.2.9; I.2.10; I.2.11**

- **简介: 该论文提出PathFormer，一种基于3D网格约束的Transformer模型，用于机器人机械臂轨迹生成。针对传统序列模型忽略运动结构导致轨迹无效或低效的问题，引入“位置-动作-时间”三元表示与约束掩码解码，确保路径合法性与任务合理性。在数字孪生环境中验证，实现高精度轨迹生成与鲁棒的现实世界执行。**

- **链接: [http://arxiv.org/pdf/2510.20161v1](http://arxiv.org/pdf/2510.20161v1)**

> **作者:** Ahmed Alanazi; Duy Ho; Yugyung Lee
>
> **备注:** 8 pages, 7 figures, 7 tables
>
> **摘要:** Robotic arms require precise, task-aware trajectory planning, yet sequence models that ignore motion structure often yield invalid or inefficient executions. We present a Path-based Transformer that encodes robot motion with a 3-grid (where/what/when) representation and constraint-masked decoding, enforcing lattice-adjacent moves and workspace bounds while reasoning over task graphs and action order. Trained on 53,755 trajectories (80% train / 20% validation), the model aligns closely with ground truth -- 89.44% stepwise accuracy, 93.32% precision, 89.44% recall, and 90.40% F1 -- with 99.99% of paths legal by construction. Compiled to motor primitives on an xArm Lite 6 with a depth-camera digital twin, it attains up to 97.5% reach and 92.5% pick success in controlled tests, and 86.7% end-to-end success across 60 language-specified tasks in cluttered scenes, absorbing slips and occlusions via local re-grounding without global re-planning. These results show that path-structured representations enable Transformers to generate accurate, reliable, and interpretable robot trajectories, bridging graph-based planning and sequence-based learning and providing a practical foundation for general-purpose manipulation and sim-to-real transfer.
>
---
#### [new 026] Reinforcement Learning-based Robust Wall Climbing Locomotion Controller in Ferromagnetic Environment
- **分类: cs.RO**

- **简介: 该论文针对磁吸附四足机器人在复杂环境中的壁面爬行任务，解决磁力吸附不确定性带来的稳定性问题。通过构建物理吸附模型并设计三阶段课程学习框架，实现鲁棒的强化学习控制，在仿真与硬件实验中均展现出强抗脱落能力与稳定爬行性能。**

- **链接: [http://arxiv.org/pdf/2510.20174v1](http://arxiv.org/pdf/2510.20174v1)**

> **作者:** Yong Um; Young-Ha Shin; Joon-Ha Kim; Soonpyo Kwon; Hae-Won Park
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** We present a reinforcement learning framework for quadrupedal wall-climbing locomotion that explicitly addresses uncertainty in magnetic foot adhesion. A physics-based adhesion model of a quadrupedal magnetic climbing robot is incorporated into simulation to capture partial contact, air-gap sensitivity, and probabilistic attachment failures. To stabilize learning and enable reliable transfer, we design a three-phase curriculum: (1) acquire a crawl gait on flat ground without adhesion, (2) gradually rotate the gravity vector to vertical while activating the adhesion model, and (3) inject stochastic adhesion failures to encourage slip recovery. The learned policy achieves a high success rate, strong adhesion retention, and rapid recovery from detachment in simulation under degraded adhesion. Compared with a model predictive control (MPC) baseline that assumes perfect adhesion, our controller maintains locomotion when attachment is intermittently lost. Hardware experiments with the untethered robot further confirm robust vertical crawling on steel surfaces, maintaining stability despite transient misalignment and incomplete attachment. These results show that combining curriculum learning with realistic adhesion modeling provides a resilient sim-to-real framework for magnetic climbing robots in complex environments.
>
---
#### [new 027] MemER: Scaling Up Memory for Robot Control via Experience Retrieval
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对机器人长期任务中记忆能力缺失的问题，提出MemER框架。通过高层策略选择关键经验帧，结合近期视觉输入生成指令，使低层策略高效处理长时序依赖。基于VLA模型，在真实场景下显著提升长周期操作性能。**

- **链接: [http://arxiv.org/pdf/2510.20328v1](http://arxiv.org/pdf/2510.20328v1)**

> **作者:** Ajay Sridhar; Jennifer Pan; Satvik Sharma; Chelsea Finn
>
> **备注:** Project page: https://jen-pan.github.io/memer/
>
> **摘要:** Humans routinely rely on memory to perform tasks, yet most robot policies lack this capability; our goal is to endow robot policies with the same ability. Naively conditioning on long observation histories is computationally expensive and brittle under covariate shift, while indiscriminate subsampling of history leads to irrelevant or redundant information. We propose a hierarchical policy framework, where the high-level policy is trained to select and track previous relevant keyframes from its experience. The high-level policy uses selected keyframes and the most recent frames when generating text instructions for a low-level policy to execute. This design is compatible with existing vision-language-action (VLA) models and enables the system to efficiently reason over long-horizon dependencies. In our experiments, we finetune Qwen2.5-VL-7B-Instruct and $\pi_{0.5}$ as the high-level and low-level policies respectively, using demonstrations supplemented with minimal language annotations. Our approach, MemER, outperforms prior methods on three real-world long-horizon robotic manipulation tasks that require minutes of memory. Videos and code can be found at https://jen-pan.github.io/memer/.
>
---
#### [new 028] NeuralTouch: Neural Descriptors for Precise Sim-to-Real Tactile Robot Control
- **分类: cs.RO**

- **简介: 该论文提出NeuralTouch，解决机器人抓取中因视觉误差导致的定位不准问题。通过融合神经描述符场（NDF）与触觉反馈，利用深度强化学习策略实现无需预设接触几何的精准抓取，显著提升模拟到现实的迁移性能与抓取鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.20390v1](http://arxiv.org/pdf/2510.20390v1)**

> **作者:** Yijiong Lin; Bowen Deng; Chenghua Lu; Max Yang; Efi Psomopoulou; Nathan F. Lepora
>
> **摘要:** Grasping accuracy is a critical prerequisite for precise object manipulation, often requiring careful alignment between the robot hand and object. Neural Descriptor Fields (NDF) offer a promising vision-based method to generate grasping poses that generalize across object categories. However, NDF alone can produce inaccurate poses due to imperfect camera calibration, incomplete point clouds, and object variability. Meanwhile, tactile sensing enables more precise contact, but existing approaches typically learn policies limited to simple, predefined contact geometries. In this work, we introduce NeuralTouch, a multimodal framework that integrates NDF and tactile sensing to enable accurate, generalizable grasping through gentle physical interaction. Our approach leverages NDF to implicitly represent the target contact geometry, from which a deep reinforcement learning (RL) policy is trained to refine the grasp using tactile feedback. This policy is conditioned on the neural descriptors and does not require explicit specification of contact types. We validate NeuralTouch through ablation studies in simulation and zero-shot transfer to real-world manipulation tasks--such as peg-out-in-hole and bottle lid opening--without additional fine-tuning. Results show that NeuralTouch significantly improves grasping accuracy and robustness over baseline methods, offering a general framework for precise, contact-rich robotic manipulation.
>
---
#### [new 029] Behavior-Aware Online Prediction of Obstacle Occupancy using Zonotopes
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对自动驾驶中未知环境下的车辆运动预测任务，解决无先验信息时周围车辆占用区域的准确预测问题。提出基于扩展卡尔曼滤波与线性规划的在线方法，估计控制动作的紧凑区间集，并通过可达性分析预测未来占用区域，实现无需训练数据的高精度、紧凑预测。**

- **链接: [http://arxiv.org/pdf/2510.20437v1](http://arxiv.org/pdf/2510.20437v1)**

> **作者:** Alvaro Carrizosa-Rendon; Jian Zhou; Erik Frisk; Vicenc Puig; Fatiha Nejjari
>
> **备注:** 64th IEEE Conference on Decision and Control
>
> **摘要:** Predicting the motion of surrounding vehicles is key to safe autonomous driving, especially in unstructured environments without prior information. This paper proposes a novel online method to accurately predict the occupancy sets of surrounding vehicles based solely on motion observations. The approach is divided into two stages: first, an Extended Kalman Filter and a Linear Programming (LP) problem are used to estimate a compact zonotopic set of control actions; then, a reachability analysis propagates this set to predict future occupancy. The effectiveness of the method has been validated through simulations in an urban environment, showing accurate and compact predictions without relying on prior assumptions or prior training data.
>
---
#### [new 030] ALICE-LRI: A General Method for Lossless Range Image Generation for Spinning LiDAR Sensors without Calibration Metadata
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ALICE-LRI，解决无校准信息下旋转激光雷达点云损失性投影问题。通过自动反演传感器内在参数，实现无需校准数据的无损范围图像生成，保证点云完全重建与几何精度，支持实时处理与压缩优化，推动高精度遥感应用发展。**

- **链接: [http://arxiv.org/pdf/2510.20708v1](http://arxiv.org/pdf/2510.20708v1)**

> **作者:** Samuel Soutullo; Miguel Yermo; David L. Vilariño; Óscar G. Lorenzo; José C. Cabaleiro; Francisco F. Rivera
>
> **摘要:** 3D LiDAR sensors are essential for autonomous navigation, environmental monitoring, and precision mapping in remote sensing applications. To efficiently process the massive point clouds generated by these sensors, LiDAR data is often projected into 2D range images that organize points by their angular positions and distances. While these range image representations enable efficient processing, conventional projection methods suffer from fundamental geometric inconsistencies that cause irreversible information loss, compromising high-fidelity applications. We present ALICE-LRI (Automatic LiDAR Intrinsic Calibration Estimation for Lossless Range Images), the first general, sensor-agnostic method that achieves lossless range image generation from spinning LiDAR point clouds without requiring manufacturer metadata or calibration files. Our algorithm automatically reverse-engineers the intrinsic geometry of any spinning LiDAR sensor by inferring critical parameters including laser beam configuration, angular distributions, and per-beam calibration corrections, enabling lossless projection and complete point cloud reconstruction with zero point loss. Comprehensive evaluation across the complete KITTI and DurLAR datasets demonstrates that ALICE-LRI achieves perfect point preservation, with zero points lost across all point clouds. Geometric accuracy is maintained well within sensor precision limits, establishing geometric losslessness with real-time performance. We also present a compression case study that validates substantial downstream benefits, demonstrating significant quality improvements in practical applications. This paradigm shift from approximate to lossless LiDAR projections opens new possibilities for high-precision remote sensing applications requiring complete geometric preservation.
>
---
#### [new 031] Deep Learning-Powered Visual SLAM Aimed at Assisting Visually Impaired Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉导航中低纹理、运动模糊等挑战性场景下的定位不稳问题，提出基于深度学习的视觉SLAM框架SELM-SLAM3，融合SuperPoint与LightGlue实现鲁棒特征提取与匹配，显著提升在复杂环境中的定位精度与跟踪稳定性，为视障者辅助导航提供可靠技术支撑。**

- **链接: [http://arxiv.org/pdf/2510.20549v1](http://arxiv.org/pdf/2510.20549v1)**

> **作者:** Marziyeh Bamdad; Hans-Peter Hutter; Alireza Darvishy
>
> **备注:** 8 pages, 7 figures, 4 tables
>
> **摘要:** Despite advancements in SLAM technologies, robust operation under challenging conditions such as low-texture, motion-blur, or challenging lighting remains an open challenge. Such conditions are common in applications such as assistive navigation for the visually impaired. These challenges undermine localization accuracy and tracking stability, reducing navigation reliability and safety. To overcome these limitations, we present SELM-SLAM3, a deep learning-enhanced visual SLAM framework that integrates SuperPoint and LightGlue for robust feature extraction and matching. We evaluated our framework using TUM RGB-D, ICL-NUIM, and TartanAir datasets, which feature diverse and challenging scenarios. SELM-SLAM3 outperforms conventional ORB-SLAM3 by an average of 87.84% and exceeds state-of-the-art RGB-D SLAM systems by 36.77%. Our framework demonstrates enhanced performance under challenging conditions, such as low-texture scenes and fast motion, providing a reliable platform for developing navigation aids for the visually impaired.
>
---
#### [new 032] EmbodiedBrain: Expanding Performance Boundaries of Task Planning for Embodied Intelligence
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EmbodiedBrain，一种面向具身智能的任务规划模型，旨在解决当前大模型在空间感知、实时性与评估真实性方面的不足。通过设计对齐代理的数据结构、创新训练方法与综合奖励机制，显著提升长周期任务成功率，并构建了新型仿真评估体系，实现性能新突破。**

- **链接: [http://arxiv.org/pdf/2510.20578v1](http://arxiv.org/pdf/2510.20578v1)**

> **作者:** Ding Zou; Feifan Wang; Mengyu Ge; Siyuan Fan; Zongbing Zhang; Wei Chen; Lingfeng Wang; Zhongyou Hu; Wenrui Yan; Zhengwei Gao; Hao Wang; Weizhao Jin; Yu Zhang; Hainan Zhao; Mingliang Zhang; Xianxian Xi; Yaru Zhang; Wenyuan Li; Zhengguang Gao; Yurui Zhu
>
> **摘要:** The realization of Artificial General Intelligence (AGI) necessitates Embodied AI agents capable of robust spatial perception, effective task planning, and adaptive execution in physical environments. However, current large language models (LLMs) and multimodal LLMs (MLLMs) for embodied tasks suffer from key limitations, including a significant gap between model design and agent requirements, an unavoidable trade-off between real-time latency and performance, and the use of unauthentic, offline evaluation metrics. To address these challenges, we propose EmbodiedBrain, a novel vision-language foundation model available in both 7B and 32B parameter sizes. Our framework features an agent-aligned data structure and employs a powerful training methodology that integrates large-scale Supervised Fine-Tuning (SFT) with Step-Augumented Group Relative Policy Optimization (Step-GRPO), which boosts long-horizon task success by integrating preceding steps as Guided Precursors. Furthermore, we incorporate a comprehensive reward system, including a Generative Reward Model (GRM) accelerated at the infrastructure level, to improve training efficiency. For enable thorough validation, we establish a three-part evaluation system encompassing General, Planning, and End-to-End Simulation Benchmarks, highlighted by the proposal and open-sourcing of a novel, challenging simulation environment. Experimental results demonstrate that EmbodiedBrain achieves superior performance across all metrics, establishing a new state-of-the-art for embodied foundation models. Towards paving the way for the next generation of generalist embodied agents, we open-source all of our data, model weight, and evaluating methods, which are available at https://zterobot.github.io/EmbodiedBrain.github.io.
>
---
## 更新

#### [replaced 001] Leveraging Analytic Gradients in Provably Safe Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01665v3](http://arxiv.org/pdf/2506.01665v3)**

> **作者:** Tim Walter; Hannah Markgraf; Jonathan Külz; Matthias Althoff
>
> **备注:** 21 pages, 10 figures
>
> **摘要:** The deployment of autonomous robots in safety-critical applications requires safety guarantees. Provably safe reinforcement learning is an active field of research that aims to provide such guarantees using safeguards. These safeguards should be integrated during training to reduce the sim-to-real gap. While there are several approaches for safeguarding sampling-based reinforcement learning, analytic gradient-based reinforcement learning often achieves superior performance from fewer environment interactions. However, there is no safeguarding approach for this learning paradigm yet. Our work addresses this gap by developing the first effective safeguard for analytic gradient-based reinforcement learning. We analyse existing, differentiable safeguards, adapt them through modified mappings and gradient formulations, and integrate them into a state-of-the-art learning algorithm and a differentiable simulation. Using numerical experiments on three control tasks, we evaluate how different safeguards affect learning. The results demonstrate safeguarded training without compromising performance. Additional visuals are provided at \href{https://timwalter.github.io/safe-agb-rl.github.io}{timwalter.github.io/safe-agb-rl.github.io}.
>
---
#### [replaced 002] Efficient Vision-Language-Action Models for Embodied Manipulation: A Systematic Survey
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.17111v3](http://arxiv.org/pdf/2510.17111v3)**

> **作者:** Weifan Guan; Qinghao Hu; Aosheng Li; Jian Cheng
>
> **摘要:** Vision-Language-Action (VLA) models extend vision-language models to embodied control by mapping natural-language instructions and visual observations to robot actions. Despite their capabilities, VLA systems face significant challenges due to their massive computational and memory demands, which conflict with the constraints of edge platforms such as on-board mobile manipulators that require real-time performance. Addressing this tension has become a central focus of recent research. In light of the growing efforts toward more efficient and scalable VLA systems, this survey provides a systematic review of approaches for improving VLA efficiency, with an emphasis on reducing latency, memory footprint, and training and inference costs. We categorize existing solutions into four dimensions: model architecture, perception feature, action generation, and training/inference strategies, summarizing representative techniques within each category. Finally, we discuss future trends and open challenges, highlighting directions for advancing efficient embodied intelligence.
>
---
#### [replaced 003] S$^2$-Diffusion: Generalizing from Instance-level to Category-level Skills in Robot Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.09389v3](http://arxiv.org/pdf/2502.09389v3)**

> **作者:** Quantao Yang; Michael C. Welle; Danica Kragic; Olov Andersson
>
> **摘要:** Recent advances in skill learning has propelled robot manipulation to new heights by enabling it to learn complex manipulation tasks from a practical number of demonstrations. However, these skills are often limited to the particular action, object, and environment \textit{instances} that are shown in the training data, and have trouble transferring to other instances of the same category. In this work we present an open-vocabulary Spatial-Semantic Diffusion policy (S$^2$-Diffusion) which enables generalization from instance-level training data to category-level, enabling skills to be transferable between instances of the same category. We show that functional aspects of skills can be captured via a promptable semantic module combined with a spatial representation. We further propose leveraging depth estimation networks to allow the use of only a single RGB camera. Our approach is evaluated and compared on a diverse number of robot manipulation tasks, both in simulation and in the real world. Our results show that S$^2$-Diffusion is invariant to changes in category-irrelevant factors as well as enables satisfying performance on other instances within the same category, even if it was not trained on that specific instance. Project website: https://s2-diffusion.github.io.
>
---
#### [replaced 004] Compositional Coordination for Multi-Robot Teams with Large Language Models
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2507.16068v3](http://arxiv.org/pdf/2507.16068v3)**

> **作者:** Zhehui Huang; Guangyao Shi; Yuwei Wu; Vijay Kumar; Gaurav S. Sukhatme
>
> **备注:** IEEE International Symposium on Multi-Robot & Multi-Agent Systems (MRS 2025) Oral
>
> **摘要:** Multi-robot coordination has traditionally relied on a mission-specific and expert-driven pipeline, where natural language mission descriptions are manually translated by domain experts into mathematical formulation, algorithm design, and executable code. This conventional process is labor-intensive, inaccessible to non-experts, and inflexible to changes in mission requirements. Here, we propose LAN2CB (Language to Collective Behavior), a novel framework that leverages large language models (LLMs) to streamline and generalize the multi-robot coordination pipeline. LAN2CB transforms natural language (NL) mission descriptions into executable Python code for multi-robot systems through two core modules: (1) Mission Analysis, which parses mission descriptions into behavior trees, and (2) Code Generation, which leverages the behavior tree and a structured knowledge base to generate robot control code. We further introduce a dataset of natural language mission descriptions to support development and benchmarking. Experiments in both simulation and real-world environments demonstrate that LAN2CB enables robust and flexible multi-robot coordination from natural language, significantly reducing manual engineering effort and supporting broad generalization across diverse mission types. Website: https://sites.google.com/view/lan-cb
>
---
#### [replaced 005] Local Guidance for Configuration-Based Multi-Agent Pathfinding
- **分类: cs.MA; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.19072v2](http://arxiv.org/pdf/2510.19072v2)**

> **作者:** Tomoki Arita; Keisuke Okumura
>
> **备注:** 10 pages
>
> **摘要:** Guidance is an emerging concept that improves the empirical performance of real-time, sub-optimal multi-agent pathfinding (MAPF) methods. It offers additional information to MAPF algorithms to mitigate congestion on a global scale by considering the collective behavior of all agents across the entire workspace. This global perspective helps reduce agents' waiting times, thereby improving overall coordination efficiency. In contrast, this study explores an alternative approach: providing local guidance in the vicinity of each agent. While such localized methods involve recomputation as agents move and may appear computationally demanding, we empirically demonstrate that supplying informative spatiotemporal cues to the planner can significantly improve solution quality without exceeding a moderate time budget. When applied to LaCAM, a leading configuration-based solver, this form of guidance establishes a new performance frontier for MAPF.
>
---
#### [replaced 006] FIMD: Fast Isolated Marker Detection for UV-Based Visual Relative Localisation in Agile UAV Swarms
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.19663v2](http://arxiv.org/pdf/2510.19663v2)**

> **作者:** Vojtěch Vrba; Viktor Walter; Petr Štěpán; Martin Saska
>
> **摘要:** A novel approach for the fast onboard detection of isolated markers for visual relative localisation of multiple teammates in agile UAV swarms is introduced in this paper. As the detection forms a key component of real-time localisation systems, a three-fold innovation is presented, consisting of an optimised procedure for CPUs, a GPU shader program, and a functionally equivalent FPGA streaming architecture. For the proposed CPU and GPU solutions, the mean processing time per pixel of input camera frames was accelerated by two to three orders of magnitude compared to the \rev{unoptimised state-of-the-art approach}. For the localisation task, the proposed FPGA architecture offered the most significant overall acceleration by minimising the total delay from camera exposure to detection results. Additionally, the proposed solutions were evaluated on various 32-bit and 64-bit embedded platforms to demonstrate their efficiency, as well as their feasibility for applications using low-end UAVs and MAVs. Thus, it has become a crucial enabling technology for agile UAV swarming.
>
---
#### [replaced 007] LiDAR, GNSS and IMU Sensor Alignment through Dynamic Time Warping to Construct 3D City Maps
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.08420v2](http://arxiv.org/pdf/2507.08420v2)**

> **作者:** Haitian Wang; Hezam Albaqami; Xinyu Wang; Muhammad Ibrahim; Zainy M. Malakan; Abdullah M. Algamdi; Mohammed H. Alghamdi; Ajmal Mian
>
> **摘要:** LiDAR-based 3D mapping suffers from cumulative drift causing global misalignment, particularly in GNSS-constrained environments. To address this, we propose a unified framework that fuses LiDAR, GNSS, and IMU data for high-resolution city-scale mapping. The method performs velocity-based temporal alignment using Dynamic Time Warping and refines GNSS and IMU signals via extended Kalman filtering. Local maps are built using Normal Distributions Transform-based registration and pose graph optimization with loop closure detection, while global consistency is enforced using GNSS-constrained anchors followed by fine registration of overlapping segments. We also introduce a large-scale multimodal dataset captured in Perth, Western Australia to facilitate future research in this direction. Our dataset comprises 144,000 frames acquired with a 128-channel Ouster LiDAR, synchronized RTK-GNSS trajectories, and MEMS-IMU measurements across 21 urban loops. To assess geometric consistency, we evaluated our method using alignment metrics based on road centerlines and intersections to capture both global and local accuracy. The proposed framework reduces the average global alignment error from 3.32m to 1.24m, achieving a 61.4% improvement, and significantly decreases the intersection centroid offset from 13.22m to 2.01m, corresponding to an 84.8% enhancement. The constructed high-fidelity map is publicly available through https://ieee-dataport.org/documents/perth-cbd-high-resolution-lidar-map-gnss-and-imu-calibration and its visualization can be viewed in the provided in https://www.youtube.com/watch?v=-ZUgs1KyMks. This dataset and method together establish a new benchmark for evaluating 3D city mapping in GNSS-constrained environments.
>
---
#### [replaced 008] Prognostic Framework for Robotic Manipulators Operating Under Dynamic Task Severities
- **分类: cs.RO; cs.LG; cs.SY; eess.SY; stat.AP**

- **链接: [http://arxiv.org/pdf/2412.00538v2](http://arxiv.org/pdf/2412.00538v2)**

> **作者:** Ayush Mohanty; Jason Dekarske; Stephen K. Robinson; Sanjay Joshi; Nagi Gebraeel
>
> **备注:** Accepted for Publication in IEEE Transactions on Systems, Man, and Cybernetics: Systems
>
> **摘要:** Robotic manipulators are critical in many applications but are known to degrade over time. This degradation is influenced by the nature of the tasks performed by the robot. Tasks with higher severity, such as handling heavy payloads, can accelerate the degradation process. One way this degradation is reflected is in the position accuracy of the robot's end-effector. In this paper, we present a prognostic modeling framework that predicts a robotic manipulator's Remaining Useful Life (RUL) while accounting for the effects of task severity. Our framework represents the robot's position accuracy as a Brownian motion process with a random drift parameter that is influenced by task severity. The dynamic nature of task severity is modeled using a continuous-time Markov chain (CTMC). To evaluate RUL, we discuss two approaches -- (1) a novel closed-form expression for Remaining Lifetime Distribution (RLD), and (2) Monte Carlo simulations, commonly used in prognostics literature. Theoretical results establish the equivalence between these RUL computation approaches. We validate our framework through experiments using two distinct physics-based simulators for planar and spatial robot fleets. Our findings show that robots in both fleets experience shorter RUL when handling a higher proportion of high-severity tasks.
>
---
#### [replaced 009] ViTacGen: Robotic Pushing with Vision-to-Touch Generation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.14117v2](http://arxiv.org/pdf/2510.14117v2)**

> **作者:** Zhiyuan Wu; Yijiong Lin; Yongqiang Zhao; Xuyang Zhang; Zhuo Chen; Nathan Lepora; Shan Luo
>
> **摘要:** Robotic pushing is a fundamental manipulation task that requires tactile feedback to capture subtle contact forces and dynamics between the end-effector and the object. However, real tactile sensors often face hardware limitations such as high costs and fragility, and deployment challenges involving calibration and variations between different sensors, while vision-only policies struggle with satisfactory performance. Inspired by humans' ability to infer tactile states from vision, we propose ViTacGen, a novel robot manipulation framework designed for visual robotic pushing with vision-to-touch generation in reinforcement learning to eliminate the reliance on high-resolution real tactile sensors, enabling effective zero-shot deployment on visual-only robotic systems. Specifically, ViTacGen consists of an encoder-decoder vision-to-touch generation network that generates contact depth images, a standardized tactile representation, directly from visual image sequence, followed by a reinforcement learning policy that fuses visual-tactile data with contrastive learning based on visual and generated tactile observations. We validate the effectiveness of our approach in both simulation and real world experiments, demonstrating its superior performance and achieving a success rate of up to 86\%.
>
---
#### [replaced 010] MoTVLA: A Vision-Language-Action Model with Unified Fast-Slow Reasoning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.18337v3](http://arxiv.org/pdf/2510.18337v3)**

> **作者:** Wenhui Huang; Changhe Chen; Han Qi; Chen Lv; Yilun Du; Heng Yang
>
> **摘要:** Integrating visual-language instructions into visuomotor policies is gaining momentum in robot learning for enhancing open-world generalization. Despite promising advances, existing approaches face two challenges: limited language steerability when no generated reasoning is used as a condition, or significant inference latency when reasoning is incorporated. In this work, we introduce MoTVLA, a mixture-of-transformers (MoT)-based vision-language-action (VLA) model that integrates fast-slow unified reasoning with behavior policy learning. MoTVLA preserves the general intelligence of pre-trained VLMs (serving as the generalist) for tasks such as perception, scene understanding, and semantic planning, while incorporating a domain expert, a second transformer that shares knowledge with the pretrained VLM, to generate domain-specific fast reasoning (e.g., robot motion decomposition), thereby improving policy execution efficiency. By conditioning the action expert on decomposed motion instructions, MoTVLA can learn diverse behaviors and substantially improve language steerability. Extensive evaluations across natural language processing benchmarks, robotic simulation environments, and real-world experiments confirm the superiority of MoTVLA in both fast-slow reasoning and manipulation task performance.
>
---
#### [replaced 011] CE-Nav: Flow-Guided Reinforcement Refinement for Cross-Embodiment Local Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.23203v2](http://arxiv.org/pdf/2509.23203v2)**

> **作者:** Kai Yang; Tianlin Zhang; Zhengbo Wang; Zedong Chu; Xiaolong Wu; Yang Cai; Mu Xu
>
> **备注:** Project Page: https://ce-nav.github.io/. Code is available at https://github.com/amap-cvlab/CE-Nav
>
> **摘要:** Generalizing local navigation policies across diverse robot morphologies is a critical challenge. Progress is often hindered by the need for costly and embodiment-specific data, the tight coupling of planning and control, and the "disastrous averaging" problem where deterministic models fail to capture multi-modal decisions (e.g., turning left or right). We introduce CE-Nav, a novel two-stage (IL-then-RL) framework that systematically decouples universal geometric reasoning from embodiment-specific dynamic adaptation. First, we train an embodiment-agnostic General Expert offline using imitation learning. This expert, a conditional normalizing flow model named VelFlow, learns the full distribution of kinematically-sound actions from a large-scale dataset generated by a classical planner, completely avoiding real robot data and resolving the multi-modality issue. Second, for a new robot, we freeze the expert and use it as a guiding prior to train a lightweight, Dynamics-Aware Refiner via online reinforcement learning. This refiner rapidly learns to compensate for the target robot's specific dynamics and controller imperfections with minimal environmental interaction. Extensive experiments on quadrupeds, bipeds, and quadrotors show that CE-Nav achieves state-of-the-art performance while drastically reducing adaptation cost. Successful real-world deployments further validate our approach as an efficient and scalable solution for building generalizable navigation systems. Code is available at https://github.com/amap-cvlab/CE-Nav.
>
---
#### [replaced 012] Leveraging Sidewalk Robots for Walkability-Related Analyses
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12148v2](http://arxiv.org/pdf/2507.12148v2)**

> **作者:** Xing Tong; Michele D. Simoni; Kaj Munhoz Arfvidsson; Jonas Mårtensson
>
> **摘要:** Walkability is a key component of sustainable urban development, while collecting detailed data on sidewalks (or pedestrian infrastructures) remains challenging due to the high costs and limited scalability of traditional methods. Sidewalk delivery robots, increasingly deployed in urban environments, offer a promising solution to these limitations. This paper explores how these robots can serve as mobile data collection platforms, capturing sidewalk-level features related to walkability in a scalable, automated, and real-time manner. A sensor-equipped robot was deployed on a sidewalk network at KTH in Stockholm, completing 101 trips covering 900 segment records. From the collected data, different typologies of features are derived, including robot trip characteristics (e.g., speed, duration), sidewalk conditions (e.g., width, surface unevenness), and sidewalk utilization (e.g., pedestrian density). Their walkability-related implications were investigated with a series of analyses. The results demonstrate that pedestrian movement patterns are strongly influenced by sidewalk characteristics, with higher density, reduced width, and surface irregularity associated with slower and more variable trajectories. Notably, robot speed closely mirrors pedestrian behavior, highlighting its potential as a proxy for assessing pedestrian dynamics. The proposed framework enables continuous monitoring of sidewalk conditions and pedestrian behavior, contributing to the development of more walkable, inclusive, and responsive urban environments.
>
---
#### [replaced 013] SafeDiver: Cooperative AUV-USV Assisted Diver Communication via Multi-agent Reinforcement Learning Approach
- **分类: cs.MA; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.11508v2](http://arxiv.org/pdf/2509.11508v2)**

> **作者:** Tinglong Deng; Hang Tao; Xinxiang Wang; Yinyan Wang; Hanjiang Luo
>
> **备注:** Withdrawn to reorganize and extend the current findings in a future version
>
> **摘要:** As underwater human activities are increasing, the demand for underwater communication service presents a significant challenge. Existing underwater diver communication methods face hurdles due to inherent disadvantages and complex underwater environments. To address this issue, we propose a scheme that utilizes maritime unmanned systems to assist divers with reliable and high-speed communication. Multiple AUVs are equipped with optical and acoustic multimodal communication devices as relay nodes, providing adaptive communication services based on changes in the diver's activity area. By using a multi-agent reinforcement learning (MARL) approach to control the cooperative movement of AUVs, high-speed and reliable data transmission between divers can be achieved. At the same time, utilizing the advantages of on-demand deployment and wide coverage of unmanned surface vehicles (USVs) as surface relay nodes to coordinate and forward information from AUVs, and controlling AUVs to adaptively select relay USV nodes for data transmission, high-quality communication between divers and surface platform can be achieved. Through simulation verification, the proposed scheme can effectively achieve reliable and high-speed communication for divers.
>
---
#### [replaced 014] DexCanvas: Bridging Human Demonstrations and Robot Learning for Dexterous Manipulation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15786v2](http://arxiv.org/pdf/2510.15786v2)**

> **作者:** Xinyue Xu; Jieqiang Sun; Jing; Dai; Siyuan Chen; Lanjie Ma; Ke Sun; Bin Zhao; Jianbo Yuan; Sheng Yi; Haohua Zhu; Yiwen Lu
>
> **摘要:** We present DexCanvas, a large-scale hybrid real-synthetic human manipulation dataset containing 7,000 hours of dexterous hand-object interactions seeded from 70 hours of real human demonstrations, organized across 21 fundamental manipulation types based on the Cutkosky taxonomy. Each entry combines synchronized multi-view RGB-D, high-precision mocap with MANO hand parameters, and per-frame contact points with physically consistent force profiles. Our real-to-sim pipeline uses reinforcement learning to train policies that control an actuated MANO hand in physics simulation, reproducing human demonstrations while discovering the underlying contact forces that generate the observed object motion. DexCanvas is the first manipulation dataset to combine large-scale real demonstrations, systematic skill coverage based on established taxonomies, and physics-validated contact annotations. The dataset can facilitate research in robotic manipulation learning, contact-rich control, and skill transfer across different hand morphologies.
>
---
#### [replaced 015] Towards Robust Zero-Shot Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.15382v2](http://arxiv.org/pdf/2510.15382v2)**

> **作者:** Kexin Zheng; Lauriane Teyssier; Yinan Zheng; Yu Luo; Xianyuan Zhan
>
> **备注:** Neurips 2025, 29 pages, 19 figures
>
> **摘要:** The recent development of zero-shot reinforcement learning (RL) has opened a new avenue for learning pre-trained generalist policies that can adapt to arbitrary new tasks in a zero-shot manner. While the popular Forward-Backward representations (FB) and related methods have shown promise in zero-shot RL, we empirically found that their modeling lacks expressivity and that extrapolation errors caused by out-of-distribution (OOD) actions during offline learning sometimes lead to biased representations, ultimately resulting in suboptimal performance. To address these issues, we propose Behavior-REgularizEd Zero-shot RL with Expressivity enhancement (BREEZE), an upgraded FB-based framework that simultaneously enhances learning stability, policy extraction capability, and representation learning quality. BREEZE introduces behavioral regularization in zero-shot RL policy learning, transforming policy optimization into a stable in-sample learning paradigm. Additionally, BREEZE extracts the policy using a task-conditioned diffusion model, enabling the generation of high-quality and multimodal action distributions in zero-shot RL settings. Moreover, BREEZE employs expressive attention-based architectures for representation modeling to capture the complex relationships between environmental dynamics. Extensive experiments on ExORL and D4RL Kitchen demonstrate that BREEZE achieves the best or near-the-best performance while exhibiting superior robustness compared to prior offline zero-shot RL methods. The official implementation is available at: https://github.com/Whiterrrrr/BREEZE.
>
---
#### [replaced 016] IRIS: An Immersive Robot Interaction System
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.03297v3](http://arxiv.org/pdf/2502.03297v3)**

> **作者:** Xinkai Jiang; Qihao Yuan; Enes Ulas Dincer; Hongyi Zhou; Ge Li; Xueyin Li; Xiaogang Jia; Timo Schnizer; Nicolas Schreiber; Weiran Liao; Julius Haag; Kailai Li; Gerhard Neumann; Rudolf Lioutikov
>
> **摘要:** This paper introduces IRIS, an Immersive Robot Interaction System leveraging Extended Reality (XR). Existing XR-based systems enable efficient data collection but are often challenging to reproduce and reuse due to their specificity to particular robots, objects, simulators, and environments. IRIS addresses these issues by supporting immersive interaction and data collection across diverse simulators and real-world scenarios. It visualizes arbitrary rigid and deformable objects, robots from simulation, and integrates real-time sensor-generated point clouds for real-world applications. Additionally, IRIS enhances collaborative capabilities by enabling multiple users to simultaneously interact within the same virtual scene. Extensive experiments demonstrate that IRIS offers efficient and intuitive data collection in both simulated and real-world settings.
>
---
#### [replaced 017] VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15530v3](http://arxiv.org/pdf/2510.15530v3)**

> **作者:** Zehao Ni; Yonghao He; Lingfeng Qian; Jilei Mao; Fa Fu; Wei Sui; Hu Su; Junran Peng; Zhipeng Wang; Bin He
>
> **摘要:** In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.
>
---
