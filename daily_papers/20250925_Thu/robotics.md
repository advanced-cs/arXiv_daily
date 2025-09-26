# 机器人 cs.RO

- **最新发布 66 篇**

- **更新 24 篇**

## 最新发布

#### [new 001] Lidar-based Tracking of Traffic Participants with Sensor Nodes in Existing Urban Infrastructure
- **分类: cs.RO**

- **简介: 该论文提出一种基于激光雷达的交通参与者跟踪框架，结合城市基础设施部署低成本传感器节点。任务是实现高效、实时的交通状态估计与跟踪。工作包括设计计算高效的CPU端观测器，集成状态估计、尺寸和类别识别，验证了在复杂环境下的鲁棒性和实时性。**

- **链接: [http://arxiv.org/pdf/2509.20009v1](http://arxiv.org/pdf/2509.20009v1)**

> **作者:** Simon Schäfer; Bassam Alrifaee; Ehsan Hashemi
>
> **备注:** 21 pages, 9 figures, this work was submitted to Wileys'Advanced Intelligent Systems for review
>
> **摘要:** This paper presents a lidar-only state estimation and tracking framework, along with a roadside sensing unit for integration with existing urban infrastructure. Urban deployments demand scalable, real-time tracking solutions, yet traditional remote sensing remains costly and computationally intensive, especially under perceptually degraded conditions. Our sensor node couples a single lidar with an edge computing unit and runs a computationally efficient, GPU-free observer that simultaneously estimates object state, class, dimensions, and existence probability. The pipeline performs: (i) state updates via an extended Kalman filter, (ii) dimension estimation using a 1D grid-map/Bayesian update, (iii) class updates via a lookup table driven by the most probable footprint, and (iv) existence estimation from track age and bounding-box consistency. Experiments in dynamic urban-like scenes with diverse traffic participants demonstrate real-time performance and high precision: The complete end-to-end pipeline finishes within \SI{100}{\milli\second} for \SI{99.88}{\%} of messages, with an excellent detection rate. Robustness is further confirmed under simulated wind and sensor vibration. These results indicate that reliable, real-time roadside tracking is feasible on CPU-only edge hardware, enabling scalable, privacy-friendly deployments within existing city infrastructure. The framework integrates with existing poles, traffic lights, and buildings, reducing deployment costs and simplifying large-scale urban rollouts and maintenance efforts.
>
---
#### [new 002] Agentic Scene Policies: Unifying Space, Semantics, and Affordances for Robot Action
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Agentic Scene Policies（ASP），旨在解决机器人执行开放语言指令的问题。通过结合语义、空间和功能表示，ASP构建可查询的场景模型，提升复杂任务下的零样本执行能力，适用于桌面操作和房间级导航任务。**

- **链接: [http://arxiv.org/pdf/2509.19571v1](http://arxiv.org/pdf/2509.19571v1)**

> **作者:** Sacha Morin; Kumaraditya Gupta; Mahtab Sandhu; Charlie Gauthier; Francesco Argenziano; Kirsty Ellis; Liam Paull
>
> **备注:** Project page: https://montrealrobotics.ca/agentic-scene-policies.github.io/
>
> **摘要:** Executing open-ended natural language queries is a core problem in robotics. While recent advances in imitation learning and vision-language-actions models (VLAs) have enabled promising end-to-end policies, these models struggle when faced with complex instructions and new scenes. An alternative is to design an explicit scene representation as a queryable interface between the robot and the world, using query results to guide downstream motion planning. In this work, we present Agentic Scene Policies (ASP), an agentic framework that leverages the advanced semantic, spatial, and affordance-based querying capabilities of modern scene representations to implement a capable language-conditioned robot policy. ASP can execute open-vocabulary queries in a zero-shot manner by explicitly reasoning about object affordances in the case of more complex skills. Through extensive experiments, we compare ASP with VLAs on tabletop manipulation problems and showcase how ASP can tackle room-level queries through affordance-guided navigation, and a scaled-up scene representation. (Project page: https://montrealrobotics.ca/agentic-scene-policies.github.io/)
>
---
#### [new 003] Real-Time Reinforcement Learning for Dynamic Tasks with a Parallel Soft Robot
- **分类: cs.RO**

- **简介: 该论文研究动态任务中的实时强化学习控制软体机器人。针对软体机器人非线性响应和传统方法效率低的问题，提出基于课程学习的RL方法，在单次硬件部署中实现可靠平衡控制，并验证了系统在部分执行器失效时的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.19525v1](http://arxiv.org/pdf/2509.19525v1)**

> **作者:** James Avtges; Jake Ketchum; Millicent Schlafly; Helena Young; Taekyoung Kim; Allison Pinosky; Ryan L. Truby; Todd D. Murphey
>
> **备注:** Published at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Closed-loop control remains an open challenge in soft robotics. The nonlinear responses of soft actuators under dynamic loading conditions limit the use of analytic models for soft robot control. Traditional methods of controlling soft robots underutilize their configuration spaces to avoid nonlinearity, hysteresis, large deformations, and the risk of actuator damage. Furthermore, episodic data-driven control approaches such as reinforcement learning (RL) are traditionally limited by sample efficiency and inconsistency across initializations. In this work, we demonstrate RL for reliably learning control policies for dynamic balancing tasks in real-time single-shot hardware deployments. We use a deformable Stewart platform constructed using parallel, 3D-printed soft actuators based on motorized handed shearing auxetic (HSA) structures. By introducing a curriculum learning approach based on expanding neighborhoods of a known equilibrium, we achieve reliable single-deployment balancing at arbitrary coordinates. In addition to benchmarking the performance of model-based and model-free methods, we demonstrate that in a single deployment, Maximum Diffusion RL is capable of learning dynamic balancing after half of the actuators are effectively disabled, by inducing buckling and by breaking actuators with bolt cutters. Training occurs with no prior data, in as fast as 15 minutes, with performance nearly identical to the fully-intact platform. Single-shot learning on hardware facilitates soft robotic systems reliably learning in the real world and will enable more diverse and capable soft robots.
>
---
#### [new 004] ROPA: Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出ROPA方法，用于双臂操作的RGB-D数据增强。针对真实演示数据收集成本高、覆盖范围有限的问题，通过优化生成合成机器人姿态及对应动作标签，提升模仿学习的数据多样性与物理一致性。**

- **链接: [http://arxiv.org/pdf/2509.19454v1](http://arxiv.org/pdf/2509.19454v1)**

> **作者:** Jason Chen; I-Chun Arthur Liu; Gaurav Sukhatme; Daniel Seita
>
> **摘要:** Training robust bimanual manipulation policies via imitation learning requires demonstration data with broad coverage over robot poses, contacts, and scene contexts. However, collecting diverse and precise real-world demonstrations is costly and time-consuming, which hinders scalability. Prior works have addressed this with data augmentation, typically for either eye-in-hand (wrist camera) setups with RGB inputs or for generating novel images without paired actions, leaving augmentation for eye-to-hand (third-person) RGB-D training with new action labels less explored. In this paper, we propose Synthetic Robot Pose Generation for RGB-D Bimanual Data Augmentation (ROPA), an offline imitation learning data augmentation method that fine-tunes Stable Diffusion to synthesize third-person RGB and RGB-D observations of novel robot poses. Our approach simultaneously generates corresponding joint-space action labels while employing constrained optimization to enforce physical consistency through appropriate gripper-to-object contact constraints in bimanual scenarios. We evaluate our method on 5 simulated and 3 real-world tasks. Our results across 2625 simulation trials and 300 real-world trials demonstrate that ROPA outperforms baselines and ablations, showing its potential for scalable RGB and RGB-D data augmentation in eye-to-hand bimanual manipulation. Our project website is available at: https://ropaaug.github.io/.
>
---
#### [new 005] Towards Autonomous Robotic Electrosurgery via Thermal Imaging
- **分类: cs.RO**

- **简介: 该论文提出ThERMO方法，利用热成像反馈自主控制电外科手术工具速度，以减少热损伤并平衡切割力。任务是实现自主电外科手术，解决传统固定速度方法不适应环境变化的问题。实验表明ThERMO显著提升切割成功率和降低切割力。**

- **链接: [http://arxiv.org/pdf/2509.19725v1](http://arxiv.org/pdf/2509.19725v1)**

> **作者:** Naveed D. Riaziat; Joseph Chen; Axel Krieger; Jeremy D. Brown
>
> **备注:** Accepted for publication in the proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Electrosurgery is a surgical technique that can improve tissue cutting by reducing cutting force and bleeding. However, electrosurgery adds a risk of thermal injury to surrounding tissue. Expert surgeons estimate desirable cutting velocities based on experience but have no quantifiable reference to indicate if a particular velocity is optimal. Furthermore, prior demonstrations of autonomous electrosurgery have primarily used constant tool velocity, which is not robust to changes in electrosurgical tissue characteristics, power settings, or tool type. Thermal imaging feedback provides information that can be used to reduce thermal injury while balancing cutting force by controlling tool velocity. We introduce Thermography for Electrosurgical Rate Modulation via Optimization (ThERMO) to autonomously reduce thermal injury while balancing cutting force by intelligently controlling tool velocity. We demonstrate ThERMO in tissue phantoms and compare its performance to the constant velocity approach. Overall, ThERMO improves cut success rate by a factor of three and can reduce peak cutting force by a factor of two. ThERMO responds to varying environmental disturbances, reduces damage to tissue, and completes cutting tasks that would otherwise result in catastrophic failure for the constant velocity approach.
>
---
#### [new 006] Chasing Stability: Humanoid Running via Control Lyapunov Function Guided Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于人形机器人动态运动控制任务，旨在解决跑步等复杂动作的稳定性与鲁棒性问题。作者将控制李雅普诺夫函数（CLF）与强化学习结合，提出CLF-RL方法，在训练中嵌入稳定性和动态轨迹信息，提升控制器性能并实现可靠跑步控制。**

- **链接: [http://arxiv.org/pdf/2509.19573v1](http://arxiv.org/pdf/2509.19573v1)**

> **作者:** Zachary Olkin; Kejun Li; William D. Compton; Aaron D. Ames
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** Achieving highly dynamic behaviors on humanoid robots, such as running, requires controllers that are both robust and precise, and hence difficult to design. Classical control methods offer valuable insight into how such systems can stabilize themselves, but synthesizing real-time controllers for nonlinear and hybrid dynamics remains challenging. Recently, reinforcement learning (RL) has gained popularity for locomotion control due to its ability to handle these complex dynamics. In this work, we embed ideas from nonlinear control theory, specifically control Lyapunov functions (CLFs), along with optimized dynamic reference trajectories into the reinforcement learning training process to shape the reward. This approach, CLF-RL, eliminates the need to handcraft and tune heuristic reward terms, while simultaneously encouraging certifiable stability and providing meaningful intermediate rewards to guide learning. By grounding policy learning in dynamically feasible trajectories, we expand the robot's dynamic capabilities and enable running that includes both flight and single support phases. The resulting policy operates reliably on a treadmill and in outdoor environments, demonstrating robustness to disturbances applied to the torso and feet. Moreover, it achieves accurate global reference tracking utilizing only on-board sensors, making a critical step toward integrating these dynamic motions into a full autonomy stack.
>
---
#### [new 007] Parse-Augment-Distill: Learning Generalizable Bimanual Visuomotor Policies from Single Human Video
- **分类: cs.RO**

- **简介: 该论文提出PAD框架，旨在从单个人类视频中学习通用双臂视觉运动策略。针对现有方法依赖大量数据和仿真导致泛化性差的问题，PAD通过解析视频轨迹、无仿真大规模增强演示和蒸馏生成策略，实现了高效且具泛化的双臂控制，在多个实际任务中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.20286v1](http://arxiv.org/pdf/2509.20286v1)**

> **作者:** Georgios Tziafas; Jiayun Zhang; Hamidreza Kasaei
>
> **摘要:** Learning visuomotor policies from expert demonstrations is an important frontier in modern robotics research, however, most popular methods require copious efforts for collecting teleoperation data and struggle to generalize out-ofdistribution. Scaling data collection has been explored through leveraging human videos, as well as demonstration augmentation techniques. The latter approach typically requires expensive simulation rollouts and trains policies with synthetic image data, therefore introducing a sim-to-real gap. In parallel, alternative state representations such as keypoints have shown great promise for category-level generalization. In this work, we bring these avenues together in a unified framework: PAD (Parse-AugmentDistill), for learning generalizable bimanual policies from a single human video. Our method relies on three steps: (a) parsing a human video demo into a robot-executable keypoint-action trajectory, (b) employing bimanual task-and-motion-planning to augment the demonstration at scale without simulators, and (c) distilling the augmented trajectories into a keypoint-conditioned policy. Empirically, we showcase that PAD outperforms state-ofthe-art bimanual demonstration augmentation works relying on image policies with simulation rollouts, both in terms of success rate and sample/cost efficiency. We deploy our framework in six diverse real-world bimanual tasks such as pouring drinks, cleaning trash and opening containers, producing one-shot policies that generalize in unseen spatial arrangements, object instances and background distractors. Supplementary material can be found in the project webpage https://gtziafas.github.io/PAD_project/.
>
---
#### [new 008] Generalist Robot Manipulation beyond Action Labeled Data
- **分类: cs.RO**

- **简介: 该论文属于通用机器人操作任务，旨在解决高质量带动作标签数据稀缺的问题。提出一种方法，利用无动作标签的视频数据，通过3D点云和动态预测器实现自监督学习，并结合少量标注数据进行对齐，提升机器人在新任务上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.19958v1](http://arxiv.org/pdf/2509.19958v1)**

> **作者:** Alexander Spiridonov; Jan-Nico Zaech; Nikolay Nikolov; Luc Van Gool; Danda Pani Paudel
>
> **备注:** Accepted at Conference on Robot Learning 2025
>
> **摘要:** Recent advances in generalist robot manipulation leverage pre-trained Vision-Language Models (VLMs) and large-scale robot demonstrations to tackle diverse tasks in a zero-shot manner. A key challenge remains: scaling high-quality, action-labeled robot demonstration data, which existing methods rely on for robustness and generalization. To address this, we propose a method that benefits from videos without action labels - featuring humans and/or robots in action - enhancing open-vocabulary performance and enabling data-efficient learning of new tasks. Our method extracts dense, dynamic 3D point clouds at the hand or gripper location and uses a proposed 3D dynamics predictor for self-supervision. This predictor is then tuned to an action predictor using a smaller labeled dataset for action alignment. We show that our method not only learns from unlabeled human and robot demonstrations - improving downstream generalist robot policies - but also enables robots to learn new tasks without action labels (i.e., out-of-action generalization) in both real-world and simulated settings.
>
---
#### [new 009] TopoCut: Learning Multi-Step Cutting with Spectral Rewards and Discrete Diffusion Policies
- **分类: cs.RO**

- **简介: 该论文提出TopoCut，针对机器人多步骤切割变形物体任务，解决拓扑变化跟踪、状态感知和评估困难的问题。工作包括：高保真仿真环境、基于谱奖励的评估方法、结合动力学感知与扩散策略的学习框架。**

- **链接: [http://arxiv.org/pdf/2509.19712v1](http://arxiv.org/pdf/2509.19712v1)**

> **作者:** Liquan Wang; Jiangjie Bian; Eric Heiden; Animesh Garg
>
> **摘要:** Robotic manipulation tasks involving cutting deformable objects remain challenging due to complex topological behaviors, difficulties in perceiving dense object states, and the lack of efficient evaluation methods for cutting outcomes. In this paper, we introduce TopoCut, a comprehensive benchmark for multi-step robotic cutting tasks that integrates a cutting environment and generalized policy learning. TopoCut is built upon three core components: (1) We introduce a high-fidelity simulation environment based on a particle-based elastoplastic solver with compliant von Mises constitutive models, augmented by a novel damage-driven topology discovery mechanism that enables accurate tracking of multiple cutting pieces. (2) We develop a comprehensive reward design that integrates the topology discovery with a pose-invariant spectral reward model based on Laplace-Beltrami eigenanalysis, facilitating consistent and robust assessment of cutting quality. (3) We propose an integrated policy learning pipeline, where a dynamics-informed perception module predicts topological evolution and produces particle-wise, topology-aware embeddings to support PDDP (Particle-based Score-Entropy Discrete Diffusion Policy) for goal-conditioned policy learning. Extensive experiments demonstrate that TopoCut supports trajectory generation, scalable learning, precise evaluation, and strong generalization across diverse object geometries, scales, poses, and cutting goals.
>
---
#### [new 010] Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains
- **分类: cs.RO; math.DS**

- **简介: 该论文提出一种记忆增强势场理论框架，用于非凸环境下的自适应控制。针对传统方法易陷入局部最优的问题，通过整合历史轨迹信息优化控制器策略，并在MPPI控制器中验证了其有效性。**

- **链接: [http://arxiv.org/pdf/2509.19672v1](http://arxiv.org/pdf/2509.19672v1)**

> **作者:** Dongzhe Zheng; Wenjie Mei
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Stochastic optimal control methods often struggle in complex non-convex landscapes, frequently becoming trapped in local optima due to their inability to learn from historical trajectory data. This paper introduces Memory-Augmented Potential Field Theory, a unified mathematical framework that integrates historical experience into stochastic optimal control. Our approach dynamically constructs memory-based potential fields that identify and encode key topological features of the state space, enabling controllers to automatically learn from past experiences and adapt their optimization strategy. We provide a theoretical analysis showing that memory-augmented potential fields possess non-convex escape properties, asymptotic convergence characteristics, and computational efficiency. We implement this theoretical framework in a Memory-Augmented Model Predictive Path Integral (MPPI) controller that demonstrates significantly improved performance in challenging non-convex environments. The framework represents a generalizable approach to experience-based learning within control systems (especially robotic dynamics), enhancing their ability to navigate complex state spaces without requiring specialized domain knowledge or extensive offline training.
>
---
#### [new 011] A Bimanual Gesture Interface for ROS-Based Mobile Manipulators Using TinyML and Sensor Fusion
- **分类: cs.RO**

- **简介: 该论文提出一种基于ROS的双手机势接口，结合TinyML与传感器融合，解决移动机械臂控制的可靠性与效率问题。通过左右手分别控制导航与操作，实现低功耗实时手势识别，提升人机交互的直观性与协调性。**

- **链接: [http://arxiv.org/pdf/2509.19521v1](http://arxiv.org/pdf/2509.19521v1)**

> **作者:** Najeeb Ahmed Bhuiyan; M. Nasimul Huq; Sakib H. Chowdhury; Rahul Mangharam
>
> **备注:** 12 pages, 11 figures
>
> **摘要:** Gesture-based control for mobile manipulators faces persistent challenges in reliability, efficiency, and intuitiveness. This paper presents a dual-hand gesture interface that integrates TinyML, spectral analysis, and sensor fusion within a ROS framework to address these limitations. The system uses left-hand tilt and finger flexion, captured using accelerometer and flex sensors, for mobile base navigation, while right-hand IMU signals are processed through spectral analysis and classified by a lightweight neural network. This pipeline enables TinyML-based gesture recognition to control a 7-DOF Kinova Gen3 manipulator. By supporting simultaneous navigation and manipulation, the framework improves efficiency and coordination compared to sequential methods. Key contributions include a bimanual control architecture, real-time low-power gesture recognition, robust multimodal sensor fusion, and a scalable ROS-based implementation. The proposed approach advances Human-Robot Interaction (HRI) for industrial automation, assistive robotics, and hazardous environments, offering a cost-effective, open-source solution with strong potential for real-world deployment and further optimization.
>
---
#### [new 012] Supercomputing for High-speed Avoidance and Reactive Planning in Robots
- **分类: cs.RO; cs.DC**

- **简介: 该论文研究了如何利用高性能计算（HPC）提升机器人实时避障能力。针对机器人在动态环境中反应速度不足的问题，提出SHARP系统，通过将轨迹规划任务卸载到HPC集群，实现了毫秒级响应，验证了HPC在实时机器人控制中的可行性。**

- **链接: [http://arxiv.org/pdf/2509.19486v1](http://arxiv.org/pdf/2509.19486v1)**

> **作者:** Kieran S. Lachmansingh; José R. González-Estrada; Ryan E. Grant; Matthew K. X. J. Pan
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** This paper presents SHARP (Supercomputing for High-speed Avoidance and Reactive Planning), a proof-of-concept study demonstrating how high-performance computing (HPC) can enable millisecond-scale responsiveness in robotic control. While modern robots face increasing demands for reactivity in human--robot shared workspaces, onboard processors are constrained by size, power, and cost. Offloading to HPC offers massive parallelism for trajectory planning, but its feasibility for real-time robotics remains uncertain due to network latency and jitter. We evaluate SHARP in a stress-test scenario where a 7-DOF manipulator must dodge high-speed foam projectiles. Using a parallelized multi-goal A* search implemented with MPI on both local and remote HPC clusters, the system achieves mean planning latencies of 22.9 ms (local) and 30.0 ms (remote, ~300 km away), with avoidance success rates of 84% and 88%, respectively. These results show that when round-trip latency remains within the tens-of-milliseconds regime, HPC-side computation is no longer the bottleneck, enabling avoidance well below human reaction times. The SHARP results motivate hybrid control architectures: low-level reflexes remain onboard for safety, while bursty, high-throughput planning tasks are offloaded to HPC for scalability. By reporting per-stage timing and success rates, this study provides a reproducible template for assessing real-time feasibility of HPC-driven robotics. Collectively, SHARP reframes HPC offloading as a viable pathway toward dependable, reactive robots in dynamic environments.
>
---
#### [new 013] SAGE:State-Aware Guided End-to-End Policy for Multi-Stage Sequential Tasks via Hidden Markov Decision Process
- **分类: cs.RO**

- **简介: 该论文提出SAGE，一种基于隐马尔可夫决策过程的端到端策略框架，用于解决多阶段机器人操作任务中的状态模糊问题。通过建模任务阶段和状态感知策略，有效提升任务成功率，并减少人工标注需求。**

- **链接: [http://arxiv.org/pdf/2509.19853v1](http://arxiv.org/pdf/2509.19853v1)**

> **作者:** BinXu Wu; TengFei Zhang; Chen Yang; JiaHao Wen; HaoCheng Li; JingTian Ma; Zhen Chen; JingYuan Wang
>
> **摘要:** Multi-stage sequential (MSS) robotic manipulation tasks are prevalent and crucial in robotics. They often involve state ambiguity, where visually similar observations correspond to different actions. We present SAGE, a state-aware guided imitation learning framework that models tasks as a Hidden Markov Decision Process (HMDP) to explicitly capture latent task stages and resolve ambiguity. We instantiate the HMDP with a state transition network that infers hidden states, and a state-aware action policy that conditions on both observations and hidden states to produce actions, thereby enabling disambiguation across task stages. To reduce manual annotation effort, we propose a semi-automatic labeling pipeline combining active learning and soft label interpolation. In real-world experiments across multiple complex MSS tasks with state ambiguity, SAGE achieved 100% task success under the standard evaluation protocol, markedly surpassing the baselines. Ablation studies further show that such performance can be maintained with manual labeling for only about 13% of the states, indicating its strong effectiveness.
>
---
#### [new 014] C-3TO: Continuous 3D Trajectory Optimization on Neural Euclidean Signed Distance Fields
- **分类: cs.RO**

- **简介: 该论文提出C-3TO，一种基于连续神经ESDF的三维轨迹优化框架，用于复杂环境中无人机的安全高效局部重规划。解决了传统方法依赖离散网格导致精度不足的问题，通过连续参数化和非线性优化实现平滑、避障的动态轨迹生成。**

- **链接: [http://arxiv.org/pdf/2509.20084v1](http://arxiv.org/pdf/2509.20084v1)**

> **作者:** Guillermo Gil; Jose Antonio Cobano; Luis Merino; Fernando Caballero
>
> **备注:** 9 pages, 5 figures, submitted to ICRA 2026
>
> **摘要:** This paper introduces a novel framework for continuous 3D trajectory optimization in cluttered environments, leveraging online neural Euclidean Signed Distance Fields (ESDFs). Unlike prior approaches that rely on discretized ESDF grids with interpolation, our method directly optimizes smooth trajectories represented by fifth-order polynomials over a continuous neural ESDF, ensuring precise gradient information throughout the entire trajectory. The framework integrates a two-stage nonlinear optimization pipeline that balances efficiency, safety and smoothness. Experimental results demonstrate that C-3TO produces collision-aware and dynamically feasible trajectories. Moreover, its flexibility in defining local window sizes and optimization parameters enables straightforward adaptation to diverse user's needs without compromising performance. By combining continuous trajectory parameterization with a continuously updated neural ESDF, C-3TO establishes a robust and generalizable foundation for safe and efficient local replanning in aerial robotics.
>
---
#### [new 015] VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出VisualMimic，一个用于人形机器人视觉运动控制的框架，旨在解决无结构环境中动作与感知整合的问题。通过结合低级关键点跟踪器和高级策略生成指令，实现了从仿真到现实的零样本迁移，完成多种操作任务。**

- **链接: [http://arxiv.org/pdf/2509.20322v1](http://arxiv.org/pdf/2509.20322v1)**

> **作者:** Shaofeng Yin; Yanjie Ze; Hong-Xing Yu; C. Karen Liu; Jiajun Wu
>
> **备注:** Website: https://visualmimic.github.io
>
> **摘要:** Humanoid loco-manipulation in unstructured environments demands tight integration of egocentric perception and whole-body control. However, existing approaches either depend on external motion capture systems or fail to generalize across diverse tasks. We introduce VisualMimic, a visual sim-to-real framework that unifies egocentric vision with hierarchical whole-body control for humanoid robots. VisualMimic combines a task-agnostic low-level keypoint tracker -- trained from human motion data via a teacher-student scheme -- with a task-specific high-level policy that generates keypoint commands from visual and proprioceptive input. To ensure stable training, we inject noise into the low-level policy and clip high-level actions using human motion statistics. VisualMimic enables zero-shot transfer of visuomotor policies trained in simulation to real humanoid robots, accomplishing a wide range of loco-manipulation tasks such as box lifting, pushing, football dribbling, and kicking. Beyond controlled laboratory settings, our policies also generalize robustly to outdoor environments. Videos are available at: https://visualmimic.github.io .
>
---
#### [new 016] HL-IK: A Lightweight Implementation of Human-Like Inverse Kinematics in Humanoid Arms
- **分类: cs.RO**

- **简介: 该论文提出HL-IK，一种轻量级仿人逆运动学方法。针对传统冗余机械臂逆运动学缺乏人类自然性的问题，通过学习肘部姿态先验，结合运动数据与优化算法，提升机械臂动作的类人性，适用于人形机器人。**

- **链接: [http://arxiv.org/pdf/2509.20263v1](http://arxiv.org/pdf/2509.20263v1)**

> **作者:** Bingjie Chen; Zihan Wang; Zhe Han; Guoping Pan; Yi Cheng; Houde Liu
>
> **摘要:** Traditional IK methods for redundant humanoid manipulators emphasize end-effector (EE) tracking, frequently producing configurations that are valid mechanically but not human-like. We present Human-Like Inverse Kinematics (HL-IK), a lightweight IK framework that preserves EE tracking while shaping whole-arm configurations to appear human-like, without full-body sensing at runtime. The key idea is a learned elbow prior: using large-scale human motion data retargeted to the robot, we train a FiLM-modulated spatio-temporal attention network (FiSTA) to predict the next-step elbow pose from the EE target and a short history of EE-elbow states.This prediction is incorporated as a small residual alongside EE and smoothness terms in a standard Levenberg-Marquardt optimizer, making HL-IK a drop-in addition to numerical IK stacks. Over 183k simulation steps, HL-IK reduces arm-similarity position and direction error by 30.6% and 35.4% on average, and by 42.2% and 47.4% on the most challenging trajectories. Hardware teleoperation on a robot distinct from simulation further confirms the gains in anthropomorphism. HL-IK is simple to integrate, adaptable across platforms via our pipeline, and adds minimal computation, enabling human-like motions for humanoid robots. Project page: https://hl-ik.github.io/
>
---
#### [new 017] DB-TSDF: Directional Bitmask-based Truncated Signed Distance Fields for Efficient Volumetric Mapping
- **分类: cs.RO**

- **简介: 该论文提出DB-TSDF，一种基于定向位掩码的截断符号距离场方法，用于高效体素地图构建。针对高分辨率实时3D重建问题，实现了CPU端点云数据融合，处理时间恒定且不依赖GPU，提升了映射效率与精度。**

- **链接: [http://arxiv.org/pdf/2509.20081v1](http://arxiv.org/pdf/2509.20081v1)**

> **作者:** Jose E. Maese; Luis Merino; Fernando Caballero
>
> **摘要:** This paper presents a high-efficiency, CPU-only volumetric mapping framework based on a Truncated Signed Distance Field (TSDF). The system incrementally fuses raw LiDAR point-cloud data into a voxel grid using a directional bitmask-based integration scheme, producing dense and consistent TSDF representations suitable for real-time 3D reconstruction. A key feature of the approach is that the processing time per point-cloud remains constant, regardless of the voxel grid resolution, enabling high resolution mapping without sacrificing runtime performance. In contrast to most recent TSDF/ESDF methods that rely on GPU acceleration, our method operates entirely on CPU, achieving competitive results in speed. Experiments on real-world open datasets demonstrate that the generated maps attain accuracy on par with contemporary mapping techniques.
>
---
#### [new 018] Where Did I Leave My Glasses? Open-Vocabulary Semantic Exploration in Real-World Semi-Static Environments
- **分类: cs.RO**

- **简介: 该论文提出一种面向半静态环境的开放词汇语义探索系统，解决机器人在动态家居环境中构建和维护一致语义地图的问题。通过概率模型跟踪物体变化并结合LLM推理导航，提升了地图更新效率与目标导航速度。**

- **链接: [http://arxiv.org/pdf/2509.19851v1](http://arxiv.org/pdf/2509.19851v1)**

> **作者:** Benjamin Bogenberger; Oliver Harrison; Orrin Dahanaggamaarachchi; Lukas Brunke; Jingxing Qian; Siqi Zhou; Angela P. Schoellig
>
> **摘要:** Robots deployed in real-world environments, such as homes, must not only navigate safely but also understand their surroundings and adapt to environment changes. To perform tasks efficiently, they must build and maintain a semantic map that accurately reflects the current state of the environment. Existing research on semantic exploration largely focuses on static scenes without persistent object-level instance tracking. A consistent map is, however, crucial for real-world robotic applications where objects in the environment can be removed, reintroduced, or shifted over time. In this work, to close this gap, we propose an open-vocabulary, semantic exploration system for semi-static environments. Our system maintains a consistent map by building a probabilistic model of object instance stationarity, systematically tracking semi-static changes, and actively exploring areas that have not been visited for a prolonged period of time. In addition to active map maintenance, our approach leverages the map's semantic richness with LLM-based reasoning for open-vocabulary object-goal navigation. This enables the robot to search more efficiently by prioritizing contextually relevant areas. We evaluate our approach across multiple real-world semi-static environments. Our system detects 95% of map changes on average, improving efficiency by more than 29% as compared to random and patrol baselines. Overall, our approach achieves a mapping precision within 2% of a fully rebuilt map while requiring substantially less exploration and further completes object goal navigation tasks about 14% faster than the next-best tested strategy (coverage patrolling). A video of our work can be found at http://tiny.cc/sem-explor-semi-static .
>
---
#### [new 019] BBoE: Leveraging Bundle of Edges for Kinodynamic Bidirectional Motion Planning
- **分类: cs.RO**

- **简介: 该论文提出BBoE，一种双向、满足动力学约束的运动规划算法。针对复杂障碍环境中的快速低代价路径规划问题，通过预计算状态遍历和优化探索策略，提升了规划效率与成功率。**

- **链接: [http://arxiv.org/pdf/2509.20333v1](http://arxiv.org/pdf/2509.20333v1)**

> **作者:** Srikrishna Bangalore Raghu; Alessandro Roncone
>
> **备注:** 8 Pages, 7 Figures
>
> **摘要:** In this work, we introduce BBoE, a bidirectional, kinodynamic, sampling-based motion planner that consistently and quickly finds low-cost solutions in environments with varying obstacle clutter. The algorithm combines exploration and exploitation while relying on precomputed robot state traversals, resulting in efficient convergence towards the goal. Our key contributions include: i) a strategy to navigate through obstacle-rich spaces by sorting and sequencing preprocessed forward propagations; and ii) BBoE, a robust bidirectional kinodynamic planner that utilizes this strategy to produce fast and feasible solutions. The proposed framework reduces planning time, diminishes solution cost and increases success rate in comparison to previous approaches.
>
---
#### [new 020] An effective control of large systems of active particles: An application to evacuation problem
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究大规模主动粒子系统的控制问题，旨在解决疏散场景中个体控制方法的可扩展性和鲁棒性不足的问题。提出结合强化学习与人工力的方法，通过领导者引导群体，应用于机器人救援疏散任务，实现高效、稳健的控制策略。**

- **链接: [http://arxiv.org/pdf/2509.19972v1](http://arxiv.org/pdf/2509.19972v1)**

> **作者:** Albina Klepach; Egor E. Nuzhin; Alexey A. Tsukanov; Nikolay V. Brilliantov
>
> **摘要:** Manipulation of large systems of active particles is a serious challenge across diverse domains, including crowd management, control of robotic swarms, and coordinated material transport. The development of advanced control strategies for complex scenarios is hindered, however, by the lack of scalability and robustness of the existing methods, in particular, due to the need of an individual control for each agent. One possible solution involves controlling a system through a leader or a group of leaders, which other agents tend to follow. Using such an approach we develop an effective control strategy for a leader, combining reinforcement learning (RL) with artificial forces acting on the system. To describe the guidance of active particles by a leader we introduce the generalized Vicsek model. This novel method is then applied to the problem of the effective evacuation by a robot-rescuer (leader) of large groups of people from hazardous places. We demonstrate, that while a straightforward application of RL yields suboptimal results, even for advanced architectures, our approach provides a robust and efficient evacuation strategy. The source code supporting this study is publicly available at: https://github.com/cinemere/evacuation.
>
---
#### [new 021] Minimalistic Autonomous Stack for High-Speed Time-Trial Racing
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究高速时间赛自动驾驶系统，针对实车测试受限的问题，提出一种极简式自动驾驶框架，实现快速部署与高效集成。通过325公里、11小时的赛道验证，最高时速达206 km/h，并分析了系统性能与安全性。**

- **链接: [http://arxiv.org/pdf/2509.19636v1](http://arxiv.org/pdf/2509.19636v1)**

> **作者:** Mahmoud Ali; Hassan Jardali; Youwei Yu; Durgakant Pushp; Lantao Liu
>
> **备注:** The data associated with this paper is available at https://doi.org/10.5281/zenodo.17187680
>
> **摘要:** Autonomous racing has seen significant advancements, driven by competitions such as the Indy Autonomous Challenge (IAC) and the Abu Dhabi Autonomous Racing League (A2RL). However, developing an autonomous racing stack for a full-scale car is often constrained by limited access to dedicated test tracks, restricting opportunities for real-world validation. While previous work typically requires extended development cycles and significant track time, this paper introduces a minimalistic autonomous racing stack for high-speed time-trial racing that emphasizes rapid deployment and efficient system integration with minimal on-track testing. The proposed stack was validated on real speedways, achieving a top speed of 206 km/h within just 11 hours' practice run on the track with 325 km in total. Additionally, we present the system performance analysis, including tracking accuracy, vehicle dynamics, and safety considerations, offering insights for teams seeking to rapidly develop and deploy an autonomous racing stack with limited track access.
>
---
#### [new 022] Techno-Economic analysis for Smart Hangar inspection operations through Sensing and Localisation at scale
- **分类: cs.RO**

- **简介: 该论文属于技术经济分析任务，旨在解决智能机库中定位系统的准确性、鲁棒性和成本问题。论文对比了多种定位技术，并提出了一种优化相机布局的方法，以实现高效、低成本的智能检测系统部署。**

- **链接: [http://arxiv.org/pdf/2509.20229v1](http://arxiv.org/pdf/2509.20229v1)**

> **作者:** Angelos Plastropoulos; Nicolas P. Avdelidis; Argyrios Zolotas
>
> **摘要:** The accuracy, resilience, and affordability of localisation are fundamental to autonomous robotic inspection within aircraft maintenance and overhaul (MRO) hangars. Hangars typically feature tall ceilings and are often made of materials such as metal. Due to its nature, it is considered a GPS-denied environment, with extensive multipath effects and stringent operational constraints that collectively create a uniquely challenging environment. This persistent gap highlights the need for domain-specific comparative studies, including rigorous cost, accuracy, and integration assessments, to inform a reliable and scalable deployment of a localisation system in the Smart Hangar. This paper presents the first techno-economic roadmap that benchmarks motion capture (MoCap), ultra-wideband (UWB), and a ceiling-mounted camera network across three operational scenarios: robot localisation, asset tracking, and surface defect detection within a 40x50 m hangar bay. A dual-layer optimisation for camera selection and positioning framework is introduced, which couples market-based camera-lens selection with an optimisation solver, producing camera layouts that minimise hardware while meeting accuracy targets. The roadmap equips MRO planners with an actionable method to balance accuracy, coverage, and budget, demonstrating that an optimised vision architecture has the potential to unlock robust and cost-effective sensing for next-generation Smart Hangars.
>
---
#### [new 023] D3Grasp: Diverse and Deformable Dexterous Grasping for General Objects
- **分类: cs.RO**

- **简介: 该论文提出D3Grasp，针对通用和变形物体的多样化稳定抓取问题。通过多模态感知与强化学习框架，融合视觉与触觉信息，设计非对称训练架构与策略，提升抓取鲁棒性与适应性，在真实环境中实现95.1%的成功率。**

- **链接: [http://arxiv.org/pdf/2509.19892v1](http://arxiv.org/pdf/2509.19892v1)**

> **作者:** Keyu Wang; Bingcong Lu; Zhengxue Cheng; Hengdi Zhang; Li Song
>
> **摘要:** Achieving diverse and stable dexterous grasping for general and deformable objects remains a fundamental challenge in robotics, due to high-dimensional action spaces and uncertainty in perception. In this paper, we present D3Grasp, a multimodal perception-guided reinforcement learning framework designed to enable Diverse and Deformable Dexterous Grasping. We firstly introduce a unified multimodal representation that integrates visual and tactile perception to robustly grasp common objects with diverse properties. Second, we propose an asymmetric reinforcement learning architecture that exploits privileged information during training while preserving deployment realism, enhancing both generalization and sample efficiency. Third, we meticulously design a training strategy to synthesize contact-rich, penetration-free, and kinematically feasible grasps with enhanced adaptability to deformable and contact-sensitive objects. Extensive evaluations confirm that D3Grasp delivers highly robust performance across large-scale and diverse object categories, and substantially advances the state of the art in dexterous grasping for deformable and compliant objects, even under perceptual uncertainty and real-world disturbances. D3Grasp achieves an average success rate of 95.1% in real-world trials,outperforming prior methods on both rigid and deformable objects benchmarks.
>
---
#### [new 024] Diffusion-Based Impedance Learning for Contact-Rich Manipulation Tasks
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出基于扩散模型的阻抗学习框架，用于接触丰富的操作任务。旨在融合信息域轨迹生成与能量域物理交互控制，通过Transformer模型和SLERP四元数调度器实现高精度位姿控制与自适应阻抗调节，在机器人自主操作中取得良好效果。**

- **链接: [http://arxiv.org/pdf/2509.19696v1](http://arxiv.org/pdf/2509.19696v1)**

> **作者:** Noah Geiger; Tamim Asfour; Neville Hogan; Johannes Lachner
>
> **备注:** 15 pages, 12 figures
>
> **摘要:** Learning methods excel at motion generation in the information domain but are not primarily designed for physical interaction in the energy domain. Impedance Control shapes physical interaction but requires task-aware tuning by selecting feasible impedance parameters. We present Diffusion-Based Impedance Learning, a framework that combines both domains. A Transformer-based Diffusion Model with cross-attention to external wrenches reconstructs a simulated Zero-Force Trajectory (sZFT). This captures both translational and rotational task-space behavior. For rotations, we introduce a novel SLERP-based quaternion noise scheduler that ensures geometric consistency. The reconstructed sZFT is then passed to an energy-based estimator that updates stiffness and damping parameters. A directional rule is applied that reduces impedance along non task axes while preserving rigidity along task directions. Training data were collected for a parkour scenario and robotic-assisted therapy tasks using teleoperation with Apple Vision Pro. With only tens of thousands of samples, the model achieved sub-millimeter positional accuracy and sub-degree rotational accuracy. Its compact model size enabled real-time torque control and autonomous stiffness adaptation on a KUKA LBR iiwa robot. The controller achieved smooth parkour traversal within force and velocity limits and 30/30 success rates for cylindrical, square, and star peg insertions without any peg-specific demonstrations in the training data set. All code for the Transformer-based Diffusion Model, the robot controller, and the Apple Vision Pro telemanipulation framework is publicly available. These results mark an important step towards Physical AI, fusing model-based control for physical interaction with learning-based methods for trajectory generation.
>
---
#### [new 025] mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies
- **分类: cs.RO**

- **简介: 该论文提出mindmap，一种用于3D动作策略的深度特征图空间记忆方法。针对机器人控制中需记忆场景空间结构的问题，构建基于语义3D重建的扩散策略，在模拟实验中表现优于无记忆机制方法。**

- **链接: [http://arxiv.org/pdf/2509.20297v1](http://arxiv.org/pdf/2509.20297v1)**

> **作者:** Remo Steiner; Alexander Millane; David Tingdahl; Clemens Volk; Vikram Ramasamy; Xinjie Yao; Peter Du; Soha Pouya; Shiwei Sheng
>
> **备注:** Accepted to CoRL 2025 Workshop RemembeRL
>
> **摘要:** End-to-end learning of robot control policies, structured as neural networks, has emerged as a promising approach to robotic manipulation. To complete many common tasks, relevant objects are required to pass in and out of a robot's field of view. In these settings, spatial memory - the ability to remember the spatial composition of the scene - is an important competency. However, building such mechanisms into robot learning systems remains an open research problem. We introduce mindmap (Spatial Memory in Deep Feature Maps for 3D Action Policies), a 3D diffusion policy that generates robot trajectories based on a semantic 3D reconstruction of the environment. We show in simulation experiments that our approach is effective at solving tasks where state-of-the-art approaches without memory mechanisms struggle. We release our reconstruction system, training code, and evaluation tasks to spur research in this direction.
>
---
#### [new 026] Self-evolved Imitation Learning in Simulated World
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出SEIL框架，用于少样本模仿学习任务。针对专家示范成本高的问题，通过模拟器交互自动生成高质量轨迹，并采用双层增强和轻量筛选机制提升性能。在LIBERO基准上取得新SOTA。**

- **链接: [http://arxiv.org/pdf/2509.19460v1](http://arxiv.org/pdf/2509.19460v1)**

> **作者:** Yifan Ye; Jun Cen; Jing Chen; Zhihe Lu
>
> **摘要:** Imitation learning has been a trend recently, yet training a generalist agent across multiple tasks still requires large-scale expert demonstrations, which are costly and labor-intensive to collect. To address the challenge of limited supervision, we propose Self-Evolved Imitation Learning (SEIL), a framework that progressively improves a few-shot model through simulator interactions. The model first attempts tasksin the simulator, from which successful trajectories are collected as new demonstrations for iterative refinement. To enhance the diversity of these demonstrations, SEIL employs dual-level augmentation: (i) Model-level, using an Exponential Moving Average (EMA) model to collaborate with the primary model, and (ii) Environment-level, introducing slight variations in initial object positions. We further introduce a lightweight selector that filters complementary and informative trajectories from the generated pool to ensure demonstration quality. These curated samples enable the model to achieve competitive performance with far fewer training examples. Extensive experiments on the LIBERO benchmark show that SEIL achieves a new state-of-the-art performance in few-shot imitation learning scenarios. Code is available at https://github.com/Jasper-aaa/SEIL.git.
>
---
#### [new 027] MARG: MAstering Risky Gap Terrains for Legged Robots with Elevation Mapping
- **分类: cs.RO**

- **简介: 该论文提出MARG，一种用于四足机器人穿越高风险间隙地形的DRL控制器。针对现有方法在安全性和效率上的不足，融合地形图与本体感知，设计奖励机制并提出TMG模型以提升稳定性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.20036v1](http://arxiv.org/pdf/2509.20036v1)**

> **作者:** Yinzhao Dong; Ji Ma; Liu Zhao; Wanyue Li; Peng Lu
>
> **摘要:** Deep Reinforcement Learning (DRL) controllers for quadrupedal locomotion have demonstrated impressive performance on challenging terrains, allowing robots to execute complex skills such as climbing, running, and jumping. However, existing blind locomotion controllers often struggle to ensure safety and efficient traversal through risky gap terrains, which are typically highly complex, requiring robots to perceive terrain information and select appropriate footholds during locomotion accurately. Meanwhile, existing perception-based controllers still present several practical limitations, including a complex multi-sensor deployment system and expensive computing resource requirements. This paper proposes a DRL controller named MAstering Risky Gap Terrains (MARG), which integrates terrain maps and proprioception to dynamically adjust the action and enhance the robot's stability in these tasks. During the training phase, our controller accelerates policy optimization by selectively incorporating privileged information (e.g., center of mass, friction coefficients) that are available in simulation but unmeasurable directly in real-world deployments due to sensor limitations. We also designed three foot-related rewards to encourage the robot to explore safe footholds. More importantly, a terrain map generation (TMG) model is proposed to reduce the drift existing in mapping and provide accurate terrain maps using only one LiDAR, providing a foundation for zero-shot transfer of the learned policy. The experimental results indicate that MARG maintains stability in various risky terrain tasks.
>
---
#### [new 028] LLM Trainer: Automated Robotic Data Generating via Demonstration Augmentation using LLMs
- **分类: cs.RO**

- **简介: 该论文提出LLM Trainer，一个自动化管道，利用大语言模型（LLM）将少量人类演示转化为大量机器人数据，用于模仿学习。通过离线注解与在线姿态调整生成新轨迹，并结合前馈计划与反馈控制器提升性能，验证了方法在真实机器人上的可行性。**

- **链接: [http://arxiv.org/pdf/2509.20070v1](http://arxiv.org/pdf/2509.20070v1)**

> **作者:** Abraham George; Amir Barati Farimani
>
> **备注:** 9 pages, 5 figures, 4 tables. Submitted to ICRA 2026
>
> **摘要:** We present LLM Trainer, a fully automated pipeline that leverages the world knowledge of Large Language Models (LLMs) to transform a small number of human demonstrations (as few as one) into a large robot dataset for imitation learning. Our approach decomposes demonstration generation into two steps: (1) offline demonstration annotation that extracts keyframes, salient objects, and pose-object relations; and (2) online keypose retargeting that adapts those keyframes to a new scene, given an initial observation. Using these modified keypoints, our system warps the original demonstration to generate a new trajectory, which is then executed, and the resulting demo, if successful, is saved. Because the annotation is reusable across scenes, we use Thompson sampling to optimize the annotation, significantly improving generation success rate. We evaluate our method on a range of tasks, and find that our data annotation method consistently outperforms expert-engineered baselines. We further show an ensemble policy that combines the optimized LLM feed-forward plan with a learned feedback imitation learning controller. Finally, we demonstrate hardware feasibility on a Franka Emika Panda robot. For additional materials and demonstration videos, please see the project website: https://sites.google.com/andrew.cmu.edu/llm-trainer
>
---
#### [new 029] DynaFlow: Dynamics-embedded Flow Matching for Physically Consistent Motion Generation from State-only Demonstrations
- **分类: cs.RO**

- **简介: 该论文提出DynaFlow，一种将动力学模型嵌入流匹配框架的方法，用于从仅有状态的演示中生成物理一致的运动轨迹。通过结合动作空间生成与仿真映射，实现了端到端训练，并在真实机器人上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2509.19804v1](http://arxiv.org/pdf/2509.19804v1)**

> **作者:** Sowoo Lee; Dongyun Kang; Jaehyun Park; Hae-Won Park
>
> **备注:** 8 pages
>
> **摘要:** This paper introduces DynaFlow, a novel framework that embeds a differentiable simulator directly into a flow matching model. By generating trajectories in the action space and mapping them to dynamically feasible state trajectories via the simulator, DynaFlow ensures all outputs are physically consistent by construction. This end-to-end differentiable architecture enables training on state-only demonstrations, allowing the model to simultaneously generate physically consistent state trajectories while inferring the underlying action sequences required to produce them. We demonstrate the effectiveness of our approach through quantitative evaluations and showcase its real-world applicability by deploying the generated actions onto a physical Go1 quadruped robot. The robot successfully reproduces diverse gait present in the dataset, executes long-horizon motions in open-loop control and translates infeasible kinematic demonstrations into dynamically executable, stylistic behaviors. These hardware experiments validate that DynaFlow produces deployable, highly effective motions on real-world hardware from state-only demonstrations, effectively bridging the gap between kinematic data and real-world execution.
>
---
#### [new 030] Discrete Diffusion for Reflective Vision-Language-Action Models in Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文提出ReflectDrive，一种用于自动驾驶的反射式视觉-语言-动作模型。针对端到端方法在轨迹生成中安全性和物理规则编码不足的问题，通过离散扩散与安全自修正机制实现安全轨迹规划，在NAVSIM基准上表现优异。**

- **链接: [http://arxiv.org/pdf/2509.20109v1](http://arxiv.org/pdf/2509.20109v1)**

> **作者:** Pengxiang Li; Yinan Zheng; Yue Wang; Huimin Wang; Hang Zhao; Jingjing Liu; Xianyuan Zhan; Kun Zhan; Xianpeng Lang
>
> **摘要:** End-to-End (E2E) solutions have emerged as a mainstream approach for autonomous driving systems, with Vision-Language-Action (VLA) models representing a new paradigm that leverages pre-trained multimodal knowledge from Vision-Language Models (VLMs) to interpret and interact with complex real-world environments. However, these methods remain constrained by the limitations of imitation learning, which struggles to inherently encode physical rules during training. Existing approaches often rely on complex rule-based post-refinement, employ reinforcement learning that remains largely limited to simulation, or utilize diffusion guidance that requires computationally expensive gradient calculations. To address these challenges, we introduce ReflectDrive, a novel learning-based framework that integrates a reflection mechanism for safe trajectory generation via discrete diffusion. We first discretize the two-dimensional driving space to construct an action codebook, enabling the use of pre-trained Diffusion Language Models for planning tasks through fine-tuning. Central to our approach is a safety-aware reflection mechanism that performs iterative self-correction without gradient computation. Our method begins with goal-conditioned trajectory generation to model multi-modal driving behaviors. Based on this, we apply local search methods to identify unsafe tokens and determine feasible solutions, which then serve as safe anchors for inpainting-based regeneration. Evaluated on the NAVSIM benchmark, ReflectDrive demonstrates significant advantages in safety-critical trajectory generation, offering a scalable and reliable solution for autonomous driving systems.
>
---
#### [new 031] Simultaneous estimation of contact position and tool shape with high-dimensional parameters using force measurements and particle filtering
- **分类: cs.RO**

- **简介: 该论文研究接触状态估计任务，旨在通过力信号同时估计工具形状和接触位置。针对传统方法依赖已知工具形状或低维参数的问题，提出基于粒子滤波的高维参数估计方法，提升接触位置估计精度。**

- **链接: [http://arxiv.org/pdf/2509.19732v1](http://arxiv.org/pdf/2509.19732v1)**

> **作者:** Kyo Kutsuzawa; Mitsuhiro Hayashibe
>
> **备注:** Accepted to The International Journal of Robotics Research (IJRR)
>
> **摘要:** Estimating the contact state between a grasped tool and the environment is essential for performing contact tasks such as assembly and object manipulation. Force signals are valuable for estimating the contact state, as they can be utilized even when the contact location is obscured by the tool. Previous studies proposed methods for estimating contact positions using force/torque signals; however, most methods require the geometry of the tool surface to be known. Although several studies have proposed methods that do not require the tool shape, these methods require considerable time for estimation or are limited to tools with low-dimensional shape parameters. Here, we propose a method for simultaneously estimating the contact position and tool shape, where the tool shape is represented by a grid, which is high-dimensional (more than 1000 dimensional). The proposed method uses a particle filter in which each particle has individual tool shape parameters, thereby to avoid directly handling a high-dimensional parameter space. The proposed method is evaluated through simulations and experiments using tools with curved shapes on a plane. Consequently, the proposed method can estimate the shape of the tool simultaneously with the contact positions, making the contact-position estimation more accurate.
>
---
#### [new 032] EgoBridge: Domain Adaptation for Generalizable Imitation from Egocentric Human Data
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出EgoBridge，用于解决从第一视角人类数据到机器人操作的领域适配问题。通过联合对齐策略潜在空间和动作信息，实现跨领域知识迁移，在真实任务中显著提升策略成功率并具备良好泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.19626v1](http://arxiv.org/pdf/2509.19626v1)**

> **作者:** Ryan Punamiya; Dhruv Patel; Patcharapong Aphiwetsa; Pranav Kuppili; Lawrence Y. Zhu; Simar Kareer; Judy Hoffman; Danfei Xu
>
> **备注:** Accepted at 39th Conference on Neural Information Processing Systems (NeurIPS 2025) and Oral at Conference on Robot Learning (CoRL 2025)
>
> **摘要:** Egocentric human experience data presents a vast resource for scaling up end-to-end imitation learning for robotic manipulation. However, significant domain gaps in visual appearance, sensor modalities, and kinematics between human and robot impede knowledge transfer. This paper presents EgoBridge, a unified co-training framework that explicitly aligns the policy latent spaces between human and robot data using domain adaptation. Through a measure of discrepancy on the joint policy latent features and actions based on Optimal Transport (OT), we learn observation representations that not only align between the human and robot domain but also preserve the action-relevant information critical for policy learning. EgoBridge achieves a significant absolute policy success rate improvement by 44% over human-augmented cross-embodiment baselines in three real-world single-arm and bimanual manipulation tasks. EgoBridge also generalizes to new objects, scenes, and tasks seen only in human data, where baselines fail entirely. Videos and additional information can be found at https://ego-bridge.github.io
>
---
#### [new 033] Beyond Human Demonstrations: Diffusion-Based Reinforcement Learning to Generate Data for VLA Training
- **分类: cs.RO**

- **简介: 该论文提出基于扩散策略优化的强化学习方法，用于生成视觉-语言-动作（VLA）模型训练数据。旨在解决依赖人力示范导致的扩展性问题，通过生成高质量、低方差轨迹，在LIBERO长时序任务中取得优于人类和传统方法的效果。**

- **链接: [http://arxiv.org/pdf/2509.19752v1](http://arxiv.org/pdf/2509.19752v1)**

> **作者:** Rushuai Yang; Hangxing Wei; Ran Zhang; Zhiyuan Feng; Xiaoyu Chen; Tong Li; Chuheng Zhang; Li Zhao; Jiang Bian; Xiu Su; Yi Chen
>
> **摘要:** Vision-language-action (VLA) models have shown strong generalization across tasks and embodiments; however, their reliance on large-scale human demonstrations limits their scalability owing to the cost and effort of manual data collection. Reinforcement learning (RL) offers a potential alternative to generate demonstrations autonomously, yet conventional RL algorithms often struggle on long-horizon manipulation tasks with sparse rewards. In this paper, we propose a modified diffusion policy optimization algorithm to generate high-quality and low-variance trajectories, which contributes to a diffusion RL-powered VLA training pipeline. Our algorithm benefits from not only the high expressiveness of diffusion models to explore complex and diverse behaviors but also the implicit regularization of the iterative denoising process to yield smooth and consistent demonstrations. We evaluate our approach on the LIBERO benchmark, which includes 130 long-horizon manipulation tasks, and show that the generated trajectories are smoother and more consistent than both human demonstrations and those from standard Gaussian RL policies. Further, training a VLA model exclusively on the diffusion RL-generated data achieves an average success rate of 81.9%, which outperforms the model trained on human data by +5.3% and that on Gaussian RL-generated data by +12.6%. The results highlight our diffusion RL as an effective alternative for generating abundant, high-quality, and low-variance demonstrations for VLA models.
>
---
#### [new 034] Formal Safety Verification and Refinement for Generative Motion Planners via Certified Local Stabilization
- **分类: cs.RO; cs.LG; cs.SY; eess.SY; math.OC**

- **简介: 该论文针对生成式运动规划器（GMP）的安全性验证问题，提出通过小规模控制器稳定GMP输出并结合神经网络验证的方法，实现闭环安全认证与动态可行性保障。**

- **链接: [http://arxiv.org/pdf/2509.19688v1](http://arxiv.org/pdf/2509.19688v1)**

> **作者:** Devesh Nath; Haoran Yin; Glen Chou
>
> **备注:** 10 pages, 12 figures
>
> **摘要:** We present a method for formal safety verification of learning-based generative motion planners. Generative motion planners (GMPs) offer advantages over traditional planners, but verifying the safety and dynamic feasibility of their outputs is difficult since neural network verification (NNV) tools scale only to a few hundred neurons, while GMPs often contain millions. To preserve GMP expressiveness while enabling verification, our key insight is to imitate the GMP by stabilizing references sampled from the GMP with a small neural tracking controller and then applying NNV to the closed-loop dynamics. This yields reachable sets that rigorously certify closed-loop safety, while the controller enforces dynamic feasibility. Building on this, we construct a library of verified GMP references and deploy them online in a way that imitates the original GMP distribution whenever it is safe to do so, improving safety without retraining. We evaluate across diverse planners, including diffusion, flow matching, and vision-language models, improving safety in simulation (on ground robots and quadcopters) and on hardware (differential-drive robot).
>
---
#### [new 035] RoMoCo: Robotic Motion Control Toolbox for Reduced-Order Model-Based Locomotion on Bipedal and Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出RoMoCo，一个开源C++工具箱，用于双足和人形机器人的运动规划与控制。旨在解决不同机器人平台上运动控制器设计复杂、难以复现的问题。通过统一接口和降阶模型，实现快速开发与跨平台验证。**

- **链接: [http://arxiv.org/pdf/2509.19545v1](http://arxiv.org/pdf/2509.19545v1)**

> **作者:** Min Dai; Aaron D. Ames
>
> **摘要:** We present RoMoCo, an open-source C++ toolbox for the synthesis and evaluation of reduced-order model-based planners and whole-body controllers for bipedal and humanoid robots. RoMoCo's modular architecture unifies state-of-the-art planners and whole-body locomotion controllers under a consistent API, enabling rapid prototyping and reproducible benchmarking. By leveraging reduced-order models for platform-agnostic gait generation, RoMoCo enables flexible controller design across diverse robots. We demonstrate its versatility and performance through extensive simulations on the Cassie, Unitree H1, and G1 robots, and validate its real-world efficacy with hardware experiments on the Cassie and G1 humanoids.
>
---
#### [new 036] HUNT: High-Speed UAV Navigation and Tracking in Unstructured Environments via Instantaneous Relative Frames
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出HUNT框架，用于无人机在无结构环境中高速导航与目标跟踪。针对无全局定位和感知退化问题，通过瞬时相对观测实现避障、搜索与跟踪一体化控制，提升了复杂场景下的自主性。**

- **链接: [http://arxiv.org/pdf/2509.19452v1](http://arxiv.org/pdf/2509.19452v1)**

> **作者:** Alessandro Saviolo; Jeffrey Mao; Giuseppe Loianno
>
> **摘要:** Search and rescue operations require unmanned aerial vehicles to both traverse unknown unstructured environments at high speed and track targets once detected. Achieving both capabilities under degraded sensing and without global localization remains an open challenge. Recent works on relative navigation have shown robust tracking by anchoring planning and control to a visible detected object, but cannot address navigation when no target is in the field of view. We present HUNT (High-speed UAV Navigation and Tracking), a real-time framework that unifies traversal, acquisition, and tracking within a single relative formulation. HUNT defines navigation objectives directly from onboard instantaneous observables such as attitude, altitude, and velocity, enabling reactive high-speed flight during search. Once a target is detected, the same perception-control pipeline transitions seamlessly to tracking. Outdoor experiments in dense forests, container compounds, and search-and-rescue operations with vehicles and mannequins demonstrate robust autonomy where global methods fail.
>
---
#### [new 037] A Biomimetic Vertebraic Soft Robotic Tail for High-Speed, High-Force Dynamic Maneuvering
- **分类: cs.RO**

- **简介: 该论文提出一种仿生椎骨软体机械尾（BVSR），旨在解决刚性尾部力量强但安全性差、软体尾部速度和力度不足的问题。通过结合仿生椎骨结构与气动执行，实现高速高力动态操作，验证了其在多种场景下的应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.20219v1](http://arxiv.org/pdf/2509.20219v1)**

> **作者:** Sicong Liu; Jianhui Liu; Fang Chen; Wenjian Yang; Juan Yi; Yu Zheng; Zheng Wang; Wanchao Chi; Chaoyang Song
>
> **备注:** 20 pages, 11 figures, 4 tables. Submitted Under Review
>
> **摘要:** Robotic tails can enhance the stability and maneuverability of mobile robots, but current designs face a trade-off between the power of rigid systems and the safety of soft ones. Rigid tails generate large inertial effects but pose risks in unstructured environments, while soft tails lack sufficient speed and force. We present a Biomimetic Vertebraic Soft Robotic (BVSR) tail that resolves this challenge through a compliant pneumatic body reinforced by a passively jointed vertebral column inspired by musculoskeletal structures. This hybrid design decouples load-bearing and actuation, enabling high-pressure actuation (up to 6 bar) for superior dynamics while preserving compliance. A dedicated kinematic and dynamic model incorporating vertebral constraints is developed and validated experimentally. The BVSR tail achieves angular velocities above 670{\deg}/s and generates inertial forces and torques up to 5.58 N and 1.21 Nm, indicating over 200% improvement compared to non-vertebraic designs. Demonstrations on rapid cart stabilization, obstacle negotiation, high-speed steering, and quadruped integration confirm its versatility and practical utility for agile robotic platforms.
>
---
#### [new 038] Trajectory Planning Using Safe Ellipsoidal Corridors as Projections of Orthogonal Trust Regions
- **分类: cs.RO**

- **简介: 该论文针对机器人轨迹规划任务，旨在解决复杂环境中轨迹平滑性和计算效率问题。提出基于椭球走廊的参数化方法，并设计凸优化求解器Orth-TRP，实现高效、光滑的轨迹规划。**

- **链接: [http://arxiv.org/pdf/2509.19734v1](http://arxiv.org/pdf/2509.19734v1)**

> **作者:** Akshay Jaitly; Jon Arrizabalaga; Guanrui Li
>
> **摘要:** Planning collision free trajectories in complex environments remains a core challenge in robotics. Existing corridor based planners which rely on decomposition of the free space into collision free subsets scale poorly with environmental complexity and require explicit allocations of time windows to trajectory segments. We introduce a new trajectory parameterization that represents trajectories in a nonconvex collision free corridor as being in a convex cartesian product of balls. This parameterization allows us to decouple problem size from geometric complexity of the solution and naturally avoids explicit time allocation by allowing trajectories to evolve continuously inside ellipsoidal corridors. Building on this representation, we formulate the Orthogonal Trust Region Problem (Orth-TRP), a specialized convex program with separable block constraints, and develop a solver that exploits this parallel structure and the unique structure of each parallel subproblem for efficient optimization. Experiments on a quadrotor trajectory planning benchmark show that our approach produces smoother trajectories and lower runtimes than state-of-the-art corridor based planners, especially in highly complicated environments.
>
---
#### [new 039] AnchDrive: Bootstrapping Diffusion Policies with Hybrid Trajectory Anchors for End-to-End Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AnchDrive，一种用于端到端驾驶的扩散策略引导框架。针对传统生成模型计算成本高、轨迹多样性不足的问题，通过静态先验与动态感知特征结合生成混合轨迹锚点，并利用扩散模型进行精细化优化，提升轨迹质量与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.20253v1](http://arxiv.org/pdf/2509.20253v1)**

> **作者:** Jinhao Chai; Anqing Jiang; Hao Jiang; Shiyi Mu; Zichong Gu; Shugong Xu
>
> **备注:** IWACIII 2025
>
> **摘要:** End-to-end multi-modal planning has become a transformative paradigm in autonomous driving, effectively addressing behavioral multi-modality and the generalization challenge in long-tail scenarios. We propose AnchDrive, a framework for end-to-end driving that effectively bootstraps a diffusion policy to mitigate the high computational cost of traditional generative models. Rather than denoising from pure noise, AnchDrive initializes its planner with a rich set of hybrid trajectory anchors. These anchors are derived from two complementary sources: a static vocabulary of general driving priors and a set of dynamic, context-aware trajectories. The dynamic trajectories are decoded in real-time by a Transformer that processes dense and sparse perceptual features. The diffusion model then learns to refine these anchors by predicting a distribution of trajectory offsets, enabling fine-grained refinement. This anchor-based bootstrapping design allows for efficient generation of diverse, high-quality trajectories. Experiments on the NAVSIM benchmark confirm that AnchDrive sets a new state-of-the-art and shows strong gen?eralizability
>
---
#### [new 040] Queryable 3D Scene Representation: A Multi-Modal Framework for Semantic Reasoning and Robotic Task Planning
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文提出3D Queryable Scene Representation（3D QSR）框架，融合多模态数据实现三维场景的语义理解和任务规划，旨在解决机器人在复杂环境中理解人类指令并执行任务的问题。**

- **链接: [http://arxiv.org/pdf/2509.20077v1](http://arxiv.org/pdf/2509.20077v1)**

> **作者:** Xun Li; Rodrigo Santa Cruz; Mingze Xi; Hu Zhang; Madhawa Perera; Ziwei Wang; Ahalya Ravendran; Brandon J. Matthews; Feng Xu; Matt Adcock; Dadong Wang; Jiajun Liu
>
> **摘要:** To enable robots to comprehend high-level human instructions and perform complex tasks, a key challenge lies in achieving comprehensive scene understanding: interpreting and interacting with the 3D environment in a meaningful way. This requires a smart map that fuses accurate geometric structure with rich, human-understandable semantics. To address this, we introduce the 3D Queryable Scene Representation (3D QSR), a novel framework built on multimedia data that unifies three complementary 3D representations: (1) 3D-consistent novel view rendering and segmentation from panoptic reconstruction, (2) precise geometry from 3D point clouds, and (3) structured, scalable organization via 3D scene graphs. Built on an object-centric design, the framework integrates with large vision-language models to enable semantic queryability by linking multimodal object embeddings, and supporting object-level retrieval of geometric, visual, and semantic information. The retrieved data are then loaded into a robotic task planner for downstream execution. We evaluate our approach through simulated robotic task planning scenarios in Unity, guided by abstract language instructions and using the indoor public dataset Replica. Furthermore, we apply it in a digital duplicate of a real wet lab environment to test QSR-supported robotic task planning for emergency response. The results demonstrate the framework's ability to facilitate scene understanding and integrate spatial and semantic reasoning, effectively translating high-level human instructions into precise robotic task planning in complex 3D environments.
>
---
#### [new 041] Bioinspired SLAM Approach for Unmanned Surface Vehicle
- **分类: cs.RO**

- **简介: 该论文提出OpenRatSLAM2，一种受啮齿动物海马体启发的SLAM方法，用于无人水面艇（USV）。针对GPS信号缺失环境下的定位与建图问题，改进了系统架构并提供了实验验证，实现了低计算成本的视觉惯性SLAM。**

- **链接: [http://arxiv.org/pdf/2509.19522v1](http://arxiv.org/pdf/2509.19522v1)**

> **作者:** Fabio Coelho; Joao Victor T. Borges; Paulo Padrao; Jose Fuentes; Ramon R. Costa; Liu Hsu; Leonardo Bobadilla
>
> **摘要:** This paper presents OpenRatSLAM2, a new version of OpenRatSLAM - a bioinspired SLAM framework based on computational models of the rodent hippocampus. OpenRatSLAM2 delivers low-computation-cost visual-inertial based SLAM, suitable for GPS-denied environments. Our contributions include a ROS2-based architecture, experimental results on new waterway datasets, and insights into system parameter tuning. This work represents the first known application of RatSLAM on USVs. The estimated trajectory was compared with ground truth data using the Hausdorff distance. The results show that the algorithm can generate a semimetric map with an error margin acceptable for most robotic applications.
>
---
#### [new 042] CU-Multi: A Dataset for Multi-Robot Collaborative Perception
- **分类: cs.RO**

- **简介: 该论文提出CU-Multi数据集，用于多机器人协同感知任务。针对现有数据集不足的问题，采集了包含RGB-D、LiDAR和高精度定位的多机器人户外数据，提供同步轨迹与语义标注，支持可复现的协同感知研究。**

- **链接: [http://arxiv.org/pdf/2509.19463v1](http://arxiv.org/pdf/2509.19463v1)**

> **作者:** Doncey Albin; Daniel McGann; Miles Mena; Annika Thomas; Harel Biggie; Xuefei Sun; Steve McGuire; Jonathan P. How; Christoffer Heckman
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** A central challenge for multi-robot systems is fusing independently gathered perception data into a unified representation. Despite progress in Collaborative SLAM (C-SLAM), benchmarking remains hindered by the scarcity of dedicated multi-robot datasets. Many evaluations instead partition single-robot trajectories, a practice that may only partially reflect true multi-robot operations and, more critically, lacks standardization, leading to results that are difficult to interpret or compare across studies. While several multi-robot datasets have recently been introduced, they mostly contain short trajectories with limited inter-robot overlap and sparse intra-robot loop closures. To overcome these limitations, we introduce CU-Multi, a dataset collected over multiple days at two large outdoor sites on the University of Colorado Boulder campus. CU-Multi comprises four synchronized runs with aligned start times and controlled trajectory overlap, replicating the distinct perspectives of a robot team. It includes RGB-D sensing, RTK GPS, semantic LiDAR, and refined ground-truth odometry. By combining overlap variation with dense semantic annotations, CU-Multi provides a strong foundation for reproducible evaluation in multi-robot collaborative perception tasks.
>
---
#### [new 043] From Space to Time: Enabling Adaptive Safety with Learned Value Functions via Disturbance Recasting
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出SPACE2TIME方法，用于在未知空间扰动下安全部署离线学习的安全过滤器。任务是增强自主系统的适应性安全性，通过将空间扰动重构为时间扰动，使预计算值函数可在线应用，解决了现实环境中模型不匹配的问题。**

- **链接: [http://arxiv.org/pdf/2509.19597v1](http://arxiv.org/pdf/2509.19597v1)**

> **作者:** Sander Tonkens; Nikhil Uday Shinde; Azra Begzadić; Michael C. Yip; Jorge Cortés; Sylvia L. Herbert
>
> **备注:** The first three authors contributed equally. This work has been accepted for publication at the Conference on Robot Learning
>
> **摘要:** The widespread deployment of autonomous systems in safety-critical environments such as urban air mobility hinges on ensuring reliable, performant, and safe operation under varying environmental conditions. One such approach, value function-based safety filters, minimally modifies a nominal controller to ensure safety. Recent advances leverage offline learned value functions to scale these safety filters to high-dimensional systems. However, these methods assume detailed priors on all possible sources of model mismatch, in the form of disturbances in the environment -- information that is rarely available in real world settings. Even in well-mapped environments like urban canyons or industrial sites, drones encounter complex, spatially-varying disturbances arising from payload-drone interaction, turbulent airflow, and other environmental factors. We introduce SPACE2TIME, which enables safe and adaptive deployment of offline-learned safety filters under unknown, spatially-varying disturbances. The key idea is to reparameterize spatial variations in disturbance as temporal variations, enabling the use of precomputed value functions during online operation. We validate SPACE2TIME on a quadcopter through extensive simulations and hardware experiments, demonstrating significant improvement over baselines.
>
---
#### [new 044] Robot Trajectron V2: A Probabilistic Shared Control Framework for Navigation
- **分类: cs.RO**

- **简介: 该论文提出Robot Trajectron V2（RT-V2），一种用于导航的**概率共享控制框架**，旨在解决**人机交互中的意图预测与安全辅助问题**。通过结合先验意图模型和实时用户输入的后验更新，RT-V2实现了更准确的意图估计与自主性平衡的导航支持。**

- **链接: [http://arxiv.org/pdf/2509.19954v1](http://arxiv.org/pdf/2509.19954v1)**

> **作者:** Pinhao Song; Yurui Du; Ophelie Saussus; Sofie De Schrijver; Irene Caprara; Peter Janssen; Renaud Detry
>
> **备注:** 26 pages, 20 figures
>
> **摘要:** We propose a probabilistic shared-control solution for navigation, called Robot Trajectron V2 (RT-V2), that enables accurate intent prediction and safe, effective assistance in human-robot interaction. RT-V2 jointly models a user's long-term behavioral patterns and their noisy, low-dimensional control signals by combining a prior intent model with a posterior update that accounts for real-time user input and environmental context. The prior captures the multimodal and history-dependent nature of user intent using recurrent neural networks and conditional variational autoencoders, while the posterior integrates this with uncertain user commands to infer desired actions. We conduct extensive experiments to validate RT-V2 across synthetic benchmarks, human-computer interaction studies with keyboard input, and brain-machine interface experiments with non-human primates. Results show that RT-V2 outperforms the state of the art in intent estimation, provides safe and efficient navigation support, and adequately balances user autonomy with assistive intervention. By unifying probabilistic modeling, reinforcement learning, and safe optimization, RT-V2 offers a principled and generalizable approach to shared control for diverse assistive technologies.
>
---
#### [new 045] AnySafe: Adapting Latent Safety Filters at Runtime via Safety Constraint Parameterization in the Latent Space
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出AnySafe，一种可在运行时适应用户指定安全约束的潜在空间安全过滤器。针对传统方法在部署时无法动态调整安全约束的问题，通过参数化约束并利用潜在空间相似性实现自适应控制，在视觉任务中保持性能的同时提升安全性。**

- **链接: [http://arxiv.org/pdf/2509.19555v1](http://arxiv.org/pdf/2509.19555v1)**

> **作者:** Sankalp Agrawal; Junwon Seo; Kensuke Nakamura; Ran Tian; Andrea Bajcsy
>
> **摘要:** Recent works have shown that foundational safe control methods, such as Hamilton-Jacobi (HJ) reachability analysis, can be applied in the latent space of world models. While this enables the synthesis of latent safety filters for hard-to-model vision-based tasks, they assume that the safety constraint is known a priori and remains fixed during deployment, limiting the safety filter's adaptability across scenarios. To address this, we propose constraint-parameterized latent safety filters that can adapt to user-specified safety constraints at runtime. Our key idea is to define safety constraints by conditioning on an encoding of an image that represents a constraint, using a latent-space similarity measure. The notion of similarity to failure is aligned in a principled way through conformal calibration, which controls how closely the system may approach the constraint representation. The parameterized safety filter is trained entirely within the world model's imagination, treating any image seen by the model as a potential test-time constraint, thereby enabling runtime adaptation to arbitrary safety constraints. In simulation and hardware experiments on vision-based control tasks with a Franka manipulator, we show that our method adapts at runtime by conditioning on the encoding of user-specified constraint images, without sacrificing performance. Video results can be found on https://any-safe.github.io
>
---
#### [new 046] RoboSSM: Scalable In-context Imitation Learning via State-Space Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboSSM，一种基于状态空间模型的上下文模仿学习方法，用于机器人任务学习。针对Transformer在长提示和扩展性上的不足，采用Longhorn SSM实现高效、可扩展的少样本适应，实验表明其在长序列和未见任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.19658v1](http://arxiv.org/pdf/2509.19658v1)**

> **作者:** Youngju Yoo; Jiaheng Hu; Yifeng Zhu; Bo Liu; Qiang Liu; Roberto Martín-Martín; Peter Stone
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** In-context imitation learning (ICIL) enables robots to learn tasks from prompts consisting of just a handful of demonstrations. By eliminating the need for parameter updates at deployment time, this paradigm supports few-shot adaptation to novel tasks. However, recent ICIL methods rely on Transformers, which have computational limitations and tend to underperform when handling longer prompts than those seen during training. In this work, we introduce RoboSSM, a scalable recipe for in-context imitation learning based on state-space models (SSM). Specifically, RoboSSM replaces Transformers with Longhorn -- a state-of-the-art SSM that provides linear-time inference and strong extrapolation capabilities, making it well-suited for long-context prompts. We evaluate our approach on the LIBERO benchmark and compare it against strong Transformer-based ICIL baselines. Experiments show that RoboSSM extrapolates effectively to varying numbers of in-context demonstrations, yields high performance on unseen tasks, and remains robust in long-horizon scenarios. These results highlight the potential of SSMs as an efficient and scalable backbone for ICIL. Our code is available at https://github.com/youngjuY/RoboSSM.
>
---
#### [new 047] Hybrid Safety Verification of Multi-Agent Systems using $ψ$-Weighted CBFs and PAC Guarantees
- **分类: cs.RO**

- **简介: 该论文提出一种混合安全验证框架，用于多智能体系统在有界随机扰动下的闭环控制。通过引入ψ-加权控制屏障函数和PAC保证，结合确定性与蒙特卡洛验证方法，提供概率安全证书，解决多智能体系统的安全验证问题。**

- **链接: [http://arxiv.org/pdf/2509.20093v1](http://arxiv.org/pdf/2509.20093v1)**

> **作者:** Venkat Margapuri; Garik Kazanjian; Naren Kosaraju
>
> **摘要:** This study proposes a hybrid safety verification framework for closed-loop multi-agent systems under bounded stochastic disturbances. The proposed approach augments control barrier functions with a novel $\psi$-weighted formulation that encodes directional control alignment between agents into the safety constraints. Deterministic admissibility is combined with empirical validation via Monte Carlo rollouts, and a PAC-style guarantee is derived based on margin-aware safety violations to provide a probabilistic safety certificate. The results from the experiments conducted under different bounded stochastic disturbances validate the feasibility of the proposed approach.
>
---
#### [new 048] GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference
- **分类: cs.RO**

- **简介: 该论文提出GUIDE框架，用于解决结构化复杂室内环境中的自主探索任务。针对现有方法在建模未观测空间和全局路径规划上的不足，结合全局图推理与扩散决策机制，提升了探索效率与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.19916v1](http://arxiv.org/pdf/2509.19916v1)**

> **作者:** Zijun Che; Yinghong Zhang; Shengyi Liang; Boyu Zhou; Jun Ma; Jinni Zhou
>
> **摘要:** Autonomous exploration in structured and complex indoor environments remains a challenging task, as existing methods often struggle to appropriately model unobserved space and plan globally efficient paths. To address these limitations, we propose GUIDE, a novel exploration framework that synergistically combines global graph inference with diffusion-based decision-making. We introduce a region-evaluation global graph representation that integrates both observed environmental data and predictions of unexplored areas, enhanced by a region-level evaluation mechanism to prioritize reliable structural inferences while discounting uncertain predictions. Building upon this enriched representation, a diffusion policy network generates stable, foresighted action sequences with significantly reduced denoising steps. Extensive simulations and real-world deployments demonstrate that GUIDE consistently outperforms state-of-the-art methods, achieving up to 18.3% faster coverage completion and a 34.9% reduction in redundant movements.
>
---
#### [new 049] Look as You Leap: Planning Simultaneous Motion and Perception for High-DOF Robots
- **分类: cs.RO**

- **简介: 该论文研究高自由度机器人的运动与感知协同规划问题，旨在动态环境中高效完成感知任务。针对现有方法的不足，提出基于GPU并行和神经代理模型的PS-PRM规划器，实现高质量在线重规划，在仿真和实机实验中表现优异。**

- **链接: [http://arxiv.org/pdf/2509.19610v1](http://arxiv.org/pdf/2509.19610v1)**

> **作者:** Qingxi Meng; Emiliano Flores; Carlos Quintero-Peña; Peizhu Qian; Zachary Kingston; Shannan K. Hamlin; Vaibhav Unhelkar; Lydia E. Kavraki
>
> **备注:** 16 pages, 10 figures, under review
>
> **摘要:** In this work, we address the problem of planning robot motions for a high-degree-of-freedom (DoF) robot that effectively achieves a given perception task while the robot and the perception target move in a dynamic environment. Achieving navigation and perception tasks simultaneously is challenging, as these objectives often impose conflicting requirements. Existing methods that compute motion under perception constraints fail to account for obstacles, are designed for low-DoF robots, or rely on simplified models of perception. Furthermore, in dynamic real-world environments, robots must replan and react quickly to changes and directly evaluating the quality of perception (e.g., object detection confidence) is often expensive or infeasible at runtime. This problem is especially important in human-centered environments such as homes and hospitals, where effective perception is essential for safe and reliable operation. To address these challenges, we propose a GPU-parallelized perception-score-guided probabilistic roadmap planner with a neural surrogate model (PS-PRM). The planner explicitly incorporates the estimated quality of a perception task into motion planning for high-DoF robots. Our method uses a learned model to approximate perception scores and leverages GPU parallelism to enable efficient online replanning in dynamic settings. We demonstrate that our planner, evaluated on high-DoF robots, outperforms baseline methods in both static and dynamic environments in both simulation and real-robot experiments.
>
---
#### [new 050] Autonomous Elemental Characterization Enabled by a Low Cost Robotic Platform Built Upon a Generalized Software Architecture
- **分类: cs.RO**

- **简介: 该论文提出一种低成本、通用的机器人平台及软件架构，用于自动化实验室中的矿物与材料表征任务。通过集成3D打印适配器和手持LIBS设备，实现高密度化学映射，解决实验室自动化程度低和成本高的问题。**

- **链接: [http://arxiv.org/pdf/2509.19541v1](http://arxiv.org/pdf/2509.19541v1)**

> **作者:** Xuan Cao; Yuxin Wu; Michael L. Whittaker
>
> **摘要:** Despite the rapidly growing applications of robots in industry, the use of robots to automate tasks in scientific laboratories is less prolific due to lack of generalized methodologies and high cost of hardware. This paper focuses on the automation of characterization tasks necessary for reducing cost while maintaining generalization, and proposes a software architecture for building robotic systems in scientific laboratory environment. A dual-layer (Socket.IO and ROS) action server design is the basic building block, which facilitates the implementation of a web-based front end for user-friendly operations and the use of ROS Behavior Tree for convenient task planning and execution. A robotic platform for automating mineral and material sample characterization is built upon the architecture, with an open source, low-cost three-axis computer numerical control gantry system serving as the main robot. A handheld laser induced breakdown spectroscopy (LIBS) analyzer is integrated with a 3D printed adapter, enabling automated 2D chemical mapping. We demonstrate the utility of automated chemical mapping by scanning of the surface of a spodumene-bearing pegmatite core sample with a 1071-point dense hyperspectral map acquired at a rate of 1520 bits per second. Automated LIBS scanning enables controlled chemical quantification in the laboratory that complements field-based measurements acquired with the same handheld device, linking resource exploration and processing steps in the supply chain for lithium-based battery materials.
>
---
#### [new 051] Terra: Hierarchical Terrain-Aware 3D Scene Graph for Task-Agnostic Outdoor Mapping
- **分类: cs.RO**

- **简介: 该论文提出Terra，一种面向户外环境的层次化地形感知3D场景图方法。旨在解决传统地图缺乏语义与结构表达的问题，结合几何与语义信息，构建轻量、任务无关的地图，支持机器人规划与操作。**

- **链接: [http://arxiv.org/pdf/2509.19579v1](http://arxiv.org/pdf/2509.19579v1)**

> **作者:** Chad R. Samuelson; Abigail Austin; Seth Knoop; Blake Romrell; Gabriel R. Slade; Timothy W. McLain; Joshua G. Mangelson
>
> **摘要:** Outdoor intelligent autonomous robotic operation relies on a sufficiently expressive map of the environment. Classical geometric mapping methods retain essential structural environment information, but lack a semantic understanding and organization to allow high-level robotic reasoning. 3D scene graphs (3DSGs) address this limitation by integrating geometric, topological, and semantic relationships into a multi-level graph-based map. Outdoor autonomous operations commonly rely on terrain information either due to task-dependence or the traversability of the robotic platform. We propose a novel approach that combines indoor 3DSG techniques with standard outdoor geometric mapping and terrain-aware reasoning, producing terrain-aware place nodes and hierarchically organized regions for outdoor environments. Our method generates a task-agnostic metric-semantic sparse map and constructs a 3DSG from this map for downstream planning tasks, all while remaining lightweight for autonomous robotic operation. Our thorough evaluation demonstrates our 3DSG method performs on par with state-of-the-art camera-based 3DSG methods in object retrieval and surpasses them in region classification while remaining memory efficient. We demonstrate its effectiveness in diverse robotic tasks of object retrieval and region monitoring in both simulation and real-world environments.
>
---
#### [new 052] Orbital Stabilization and Time Synchronization of Unstable Periodic Motions in Underactuated Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究欠驱动机器人的轨道稳定与时间同步问题，提出结合时变LQR和滑模控制的方法，扩展横向线性化框架以实现周期轨迹的同步稳定，并通过实验验证了集中与分散控制策略的有效性。**

- **链接: [http://arxiv.org/pdf/2509.20082v1](http://arxiv.org/pdf/2509.20082v1)**

> **作者:** Surov Maksim
>
> **摘要:** This paper presents a control methodology for achieving orbital stabilization with simultaneous time synchronization of periodic trajectories in underactuated robotic systems. The proposed approach extends the classical transverse linearization framework to explicitly incorporate time-desynchronization dynamics. To stabilize the resulting extended transverse dynamics, we employ a combination of time-varying LQR and sliding-mode control. The theoretical results are validated experimentally through the implementation of both centralized and decentralized control strategies on a group of six Butterfly robots.
>
---
#### [new 053] Crater Observing Bio-inspired Rolling Articulator (COBRA)
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出COBRA，一种仿生蛇形机器人，用于解决月球陨石坑中水资源探测的移动难题。通过结合蠕动与翻滚模式，提升在极端地形中的适应性与效率，支持自主探测任务。**

- **链接: [http://arxiv.org/pdf/2509.19473v1](http://arxiv.org/pdf/2509.19473v1)**

> **作者:** Adarsh Salagame; Henry Noyes; Alireza Ramezani; Eric Sihite; Arash Kalantari
>
> **摘要:** NASA aims to establish a sustainable human basecamp on the Moon as a stepping stone for future missions to Mars and beyond. The discovery of water ice on the Moon's craters located in permanently shadowed regions, which can provide drinking water, oxygen, and rocket fuel, is therefore of critical importance. However, current methods to access lunar ice deposits are limited. While rovers have been used to explore the lunar surface for decades, they face significant challenges in navigating harsh terrains, such as permanently shadowed craters, due to the high risk of immobilization. This report introduces COBRA (Crater Observing Bio-inspired Rolling Articulator), a multi-modal snake-style robot designed to overcome mobility challenges in Shackleton Crater's rugged environment. COBRA combines slithering and tumbling locomotion to adapt to various crater terrains. In snake mode, it uses sidewinding to traverse flat or low inclined surfaces, while in tumbling mode, it forms a circular barrel by linking its head and tail, enabling rapid movement with minimal energy on steep slopes. Equipped with an onboard computer, stereo camera, inertial measurement unit, and joint encoders, COBRA facilitates real-time data collection and autonomous operation. This paper highlights COBRAs robustness and efficiency in navigating extreme terrains through both simulations and experimental validation.
>
---
#### [new 054] OmniVLA: An Omni-Modal Vision-Language-Action Model for Robot Navigation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出OmniVLA，一个用于机器人导航的多模态视觉-语言-动作模型。针对现有导航策略依赖单一目标模态的问题，研究通过融合2D姿态、图像和自然语言等多模态信息，提升模型在复杂环境中的适应性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.19480v1](http://arxiv.org/pdf/2509.19480v1)**

> **作者:** Noriaki Hirose; Catherine Glossop; Dhruv Shah; Sergey Levine
>
> **备注:** 9 pages, 7 figures, 6 tables
>
> **摘要:** Humans can flexibly interpret and compose different goal specifications, such as language instructions, spatial coordinates, or visual references, when navigating to a destination. In contrast, most existing robotic navigation policies are trained on a single modality, limiting their adaptability to real-world scenarios where different forms of goal specification are natural and complementary. In this work, we present a training framework for robotic foundation models that enables omni-modal goal conditioning for vision-based navigation. Our approach leverages a high-capacity vision-language-action (VLA) backbone and trains with three primary goal modalities: 2D poses, egocentric images, and natural language, as well as their combinations, through a randomized modality fusion strategy. This design not only expands the pool of usable datasets but also encourages the policy to develop richer geometric, semantic, and visual representations. The resulting model, OmniVLA, achieves strong generalization to unseen environments, robustness to scarce modalities, and the ability to follow novel natural language instructions. We demonstrate that OmniVLA outperforms specialist baselines across modalities and offers a flexible foundation for fine-tuning to new modalities and tasks. We believe OmniVLA provides a step toward broadly generalizable and flexible navigation policies, and a scalable path for building omni-modal robotic foundation models. We present videos showcasing OmniVLA performance and will release its checkpoints and training code on our project page.
>
---
#### [new 055] PersONAL: Towards a Comprehensive Benchmark for Personalized Embodied Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出了PersONAL，一个用于个性化具身智能体的综合基准。针对家庭场景中用户个性化需求建模困难的问题，构建了包含2000多个高真实感场景的数据集，并设计了自然语言驱动的导航与定位任务，推动个性化机器人研究。**

- **链接: [http://arxiv.org/pdf/2509.19843v1](http://arxiv.org/pdf/2509.19843v1)**

> **作者:** Filippo Ziliotto; Jelin Raphael Akkara; Alessandro Daniele; Lamberto Ballan; Luciano Serafini; Tommaso Campari
>
> **摘要:** Recent advances in Embodied AI have enabled agents to perform increasingly complex tasks and adapt to diverse environments. However, deploying such agents in realistic human-centered scenarios, such as domestic households, remains challenging, particularly due to the difficulty of modeling individual human preferences and behaviors. In this work, we introduce PersONAL (PERSonalized Object Navigation And Localization, a comprehensive benchmark designed to study personalization in Embodied AI. Agents must identify, retrieve, and navigate to objects associated with specific users, responding to natural-language queries such as "find Lily's backpack". PersONAL comprises over 2,000 high-quality episodes across 30+ photorealistic homes from the HM3D dataset. Each episode includes a natural-language scene description with explicit associations between objects and their owners, requiring agents to reason over user-specific semantics. The benchmark supports two evaluation modes: (1) active navigation in unseen environments, and (2) object grounding in previously mapped scenes. Experiments with state-of-the-art baselines reveal a substantial gap to human performance, highlighting the need for embodied agents capable of perceiving, reasoning, and memorizing over personalized information; paving the way towards real-world assistive robot.
>
---
#### [new 056] Learning from Observation: A Survey of Recent Advances
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文属于模仿学习任务，旨在解决专家动作难以获取的问题。论文综述了仅依赖专家状态信息的“从观察中学习”方法，提出统一框架分类现有方法，并探讨其与离线强化学习等领域的联系，指出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.19379v1](http://arxiv.org/pdf/2509.19379v1)**

> **作者:** Returaj Burnwal; Hriday Mehta; Nirav Pravinbhai Bhatt; Balaraman Ravindran
>
> **摘要:** Imitation Learning (IL) algorithms offer an efficient way to train an agent by mimicking an expert's behavior without requiring a reward function. IL algorithms often necessitate access to state and action information from expert demonstrations. Although expert actions can provide detailed guidance, requiring such action information may prove impractical for real-world applications where expert actions are difficult to obtain. To address this limitation, the concept of learning from observation (LfO) or state-only imitation learning (SOIL) has recently gained attention, wherein the imitator only has access to expert state visitation information. In this paper, we present a framework for LfO and use it to survey and classify existing LfO methods in terms of their trajectory construction, assumptions and algorithm's design choices. This survey also draws connections between several related fields like offline RL, model-based RL and hierarchical RL. Finally, we use our framework to identify open problems and suggest future research directions.
>
---
#### [new 057] The Impact of 2D Segmentation Backbones on Point Cloud Predictions Using 4D Radar
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究了在4D雷达点云预测任务中，不同分割主干网络对生成效果的影响。旨在降低自动驾驶对昂贵LiDAR的依赖。通过实验发现，合适的高容量分割主干可使性能提升23.7%。**

- **链接: [http://arxiv.org/pdf/2509.19644v1](http://arxiv.org/pdf/2509.19644v1)**

> **作者:** William L. Muckelroy III; Mohammed Alsakabi; John M. Dolan; Ozan K. Tonguz
>
> **摘要:** LiDAR's dense, sharp point cloud (PC) representations of the surrounding environment enable accurate perception and significantly improve road safety by offering greater scene awareness and understanding. However, LiDAR's high cost continues to restrict the broad adoption of high-level Autonomous Driving (AD) systems in commercially available vehicles. Prior research has shown progress towards circumventing the need for LiDAR by training a neural network, using LiDAR point clouds as ground truth (GT), to produce LiDAR-like 3D point clouds using only 4D Radars. One of the best examples is a neural network created to train a more efficient radar target detector with a modular 2D convolutional neural network (CNN) backbone and a temporal coherence network at its core that uses the RaDelft dataset for training (see arXiv:2406.04723). In this work, we investigate the impact of higher-capacity segmentation backbones on the quality of the produced point clouds. Our results show that while very high-capacity models may actually hurt performance, an optimal segmentation backbone can provide a 23.7% improvement over the state-of-the-art (SOTA).
>
---
#### [new 058] Scensory: Automated Real-Time Fungal Identification and Spatial Mapping
- **分类: eess.SP; cs.RO**

- **简介: 该论文提出Scensory，一种基于机器人和VOC传感器的实时真菌识别与定位系统。针对传统方法耗时、昂贵且缺乏空间分辨率的问题，利用深度学习分析VOC动态，实现快速、低成本的环境监测与源追踪。**

- **链接: [http://arxiv.org/pdf/2509.19318v1](http://arxiv.org/pdf/2509.19318v1)**

> **作者:** Yanbaihui Liu; Erica Babusci; Claudia K. Gunsch; Boyuan Chen
>
> **备注:** Our project website is at: http://generalroboticslab.com/Scensory
>
> **摘要:** Indoor fungal contamination poses significant risks to public health, yet existing detection methods are slow, costly, and lack spatial resolution. Conventional approaches rely on laboratory analysis or high-concentration sampling, making them unsuitable for real-time monitoring and scalable deployment. We introduce \textbf{\textit{Scensory}}, a robot-enabled olfactory system that simultaneously identifies fungal species and localizes their spatial origin using affordable volatile organic compound (VOC) sensor arrays and deep learning. Our key idea is that temporal VOC dynamics encode both chemical and spatial signatures, which we decode through neural architectures trained on robot-automated data collection. We demonstrate two operational modes: a passive multi-array configuration for environmental monitoring, and a mobile single-array configuration for active source tracking. Across five fungal species, our system achieves up to 89.85\% accuracy in species detection and 87.31\% accuracy in localization under ambient conditions, where each prediction only takes 3--7\,s sensor inputs. Additionally, by computationally analyzing model behavior, we can uncover key biochemical signatures without additional laboratory experiments. Our approach enables real-time, spatially aware fungal monitoring and establishes a scalable and affordable framework for autonomous environmental sensing.
>
---
#### [new 059] Video models are zero-shot learners and reasoners
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文探讨视频模型是否能像大语言模型一样具备通用视觉理解能力。研究发现，Veo 3在未训练任务上表现出零样本学习与推理能力，如物体分割、边缘检测、物理属性理解等，表明视频模型正向统一的视觉基础模型发展。**

- **链接: [http://arxiv.org/pdf/2509.20328v1](http://arxiv.org/pdf/2509.20328v1)**

> **作者:** Thaddäus Wiedemer; Yuxuan Li; Paul Vicol; Shixiang Shane Gu; Nick Matarese; Kevin Swersky; Been Kim; Priyank Jaini; Robert Geirhos
>
> **备注:** Project page: https://video-zero-shot.github.io/
>
> **摘要:** The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today's generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn't explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo's emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models.
>
---
#### [new 060] VIMD: Monocular Visual-Inertial Motion and Depth Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VIMD，一种单目视觉-惯性运动与深度估计框架，旨在解决高效准确的密集度量深度估计问题。通过结合MSCKF运动跟踪和多视角信息迭代优化，提升了深度估计精度和鲁棒性，适用于资源受限场景。**

- **链接: [http://arxiv.org/pdf/2509.19713v1](http://arxiv.org/pdf/2509.19713v1)**

> **作者:** Saimouli Katragadda; Guoquan Huang
>
> **摘要:** Accurate and efficient dense metric depth estimation is crucial for 3D visual perception in robotics and XR. In this paper, we develop a monocular visual-inertial motion and depth (VIMD) learning framework to estimate dense metric depth by leveraging accurate and efficient MSCKF-based monocular visual-inertial motion tracking. At the core the proposed VIMD is to exploit multi-view information to iteratively refine per-pixel scale, instead of globally fitting an invariant affine model as in the prior work. The VIMD framework is highly modular, making it compatible with a variety of existing depth estimation backbones. We conduct extensive evaluations on the TartanAir and VOID datasets and demonstrate its zero-shot generalization capabilities on the AR Table dataset. Our results show that VIMD achieves exceptional accuracy and robustness, even with extremely sparse points as few as 10-20 metric depth points per image. This makes the proposed VIMD a practical solution for deployment in resource constrained settings, while its robust performance and strong generalization capabilities offer significant potential across a wide range of scenarios.
>
---
#### [new 061] RDAR: Reward-Driven Agent Relevance Estimation for Autonomous Driving
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出RDAR方法，用于自动驾驶中的智能体相关性估计。针对现有注意力机制计算成本高的问题，通过强化学习策略学习每个智能体对车辆行为的影响，从而减少输入智能体数量，在保证驾驶性能的同时提升效率。**

- **链接: [http://arxiv.org/pdf/2509.19789v1](http://arxiv.org/pdf/2509.19789v1)**

> **作者:** Carlo Bosio; Greg Woelki; Noureldin Hendy; Nicholas Roy; Byungsoo Kim
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Human drivers focus only on a handful of agents at any one time. On the other hand, autonomous driving systems process complex scenes with numerous agents, regardless of whether they are pedestrians on a crosswalk or vehicles parked on the side of the road. While attention mechanisms offer an implicit way to reduce the input to the elements that affect decisions, existing attention mechanisms for capturing agent interactions are quadratic, and generally computationally expensive. We propose RDAR, a strategy to learn per-agent relevance -- how much each agent influences the behavior of the controlled vehicle -- by identifying which agents can be excluded from the input to a pre-trained behavior model. We formulate the masking procedure as a Markov Decision Process where the action consists of a binary mask indicating agent selection. We evaluate RDAR on a large-scale driving dataset, and demonstrate its ability to learn an accurate numerical measure of relevance by achieving comparable driving performance, in terms of overall progress, safety and performance, while processing significantly fewer agents compared to a state of the art behavior model.
>
---
#### [new 062] Hyperspectral Adapter for Semantic Segmentation with Vision Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对语义分割任务，旨在解决现有方法在高光谱成像（HSI）输入下性能不足的问题。提出了一种结合预训练视觉基础模型的高光谱适配器，通过引入光谱变换器和模态感知交互模块，有效融合空间-光谱信息，实现了优于现有方法的分割效果。**

- **链接: [http://arxiv.org/pdf/2509.20107v1](http://arxiv.org/pdf/2509.20107v1)**

> **作者:** JuanaJuana Valeria Hurtado; Rohit Mohan; Abhinav Valada
>
> **摘要:** Hyperspectral imaging (HSI) captures spatial information along with dense spectral measurements across numerous narrow wavelength bands. This rich spectral content has the potential to facilitate robust robotic perception, particularly in environments with complex material compositions, varying illumination, or other visually challenging conditions. However, current HSI semantic segmentation methods underperform due to their reliance on architectures and learning frameworks optimized for RGB inputs. In this work, we propose a novel hyperspectral adapter that leverages pretrained vision foundation models to effectively learn from hyperspectral data. Our architecture incorporates a spectral transformer and a spectrum-aware spatial prior module to extract rich spatial-spectral features. Additionally, we introduce a modality-aware interaction block that facilitates effective integration of hyperspectral representations and frozen vision Transformer features through dedicated extraction and injection mechanisms. Extensive evaluations on three benchmark autonomous driving datasets demonstrate that our architecture achieves state-of-the-art semantic segmentation performance while directly using HSI inputs, outperforming both vision-based and hyperspectral segmentation methods. We make the code available at https://hyperspectraladapter.cs.uni-freiburg.de.
>
---
#### [new 063] Robust Near-Optimal Nonlinear Target Enclosing Guidance
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文提出一种非线性最优引导律，用于实现无人机在任意几何形状下对目标的鲁棒包围。通过状态依赖Riccati方程和积分滑模方法，解决了传统圆形包围的局限性，提高了抗干扰能力和灵活性。**

- **链接: [http://arxiv.org/pdf/2509.19477v1](http://arxiv.org/pdf/2509.19477v1)**

> **作者:** Abhinav Sinha; Rohit V. Nanavati
>
> **摘要:** This paper proposes a nonlinear optimal guidance law that enables a pursuer to enclose a target within arbitrary geometric patterns, which extends beyond conventional circular encirclement. The design operates using only relative state measurements and formulates a target enclosing guidance law in which the vehicle's lateral acceleration serves as the steering control, making it well-suited for aerial vehicles with turning constraints. Our approach generalizes and extends existing guidance strategies that are limited to target encirclement and provides a degree of optimality. At the same time, the exact information of the target's maneuver is unnecessary during the design. The guidance law is developed within the framework of a state-dependent Riccati equation (SDRE), thereby providing a systematic way to handle nonlinear dynamics through a pseudo-linear representation to design locally optimal feedback guidance commands through state-dependent weighting matrices. While SDRE ensures near-optimal performance in the absence of strong disturbances, we further augment the design to incorporate an integral sliding mode manifold to compensate when disturbances push the system away from the nominal trajectory, and demonstrate that the design provides flexibility in the sense that the (possibly time-varying) stand-off curvature could also be treated as unknown. Simulations demonstrate the efficacy of the proposed approach.
>
---
#### [new 064] Score the Steps, Not Just the Goal: VLM-Based Subgoal Evaluation for Robotic Manipulation
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出StepEval，一个基于视觉-语言模型的子目标评估框架，用于机器人操作任务。旨在解决传统二元成功率指标无法反映多步骤任务中部分能力的问题。通过子目标级评估向量，提升评估的细粒度与可解释性，并推动社区共建开源标准。**

- **链接: [http://arxiv.org/pdf/2509.19524v1](http://arxiv.org/pdf/2509.19524v1)**

> **作者:** Ramy ElMallah; Krish Chhajer; Chi-Guhn Lee
>
> **备注:** Accepted to the CoRL 2025 Eval&Deploy Workshop
>
> **摘要:** Robot learning papers typically report a single binary success rate (SR), which obscures where a policy succeeds or fails along a multi-step manipulation task. We argue that subgoal-level reporting should become routine: for each trajectory, a vector of per-subgoal SRs that makes partial competence visible (e.g., grasp vs. pour). We propose a blueprint for StepEval, a cost-aware plug-in evaluation framework that utilizes vision-language models (VLMs) as automated judges of subgoal outcomes from recorded images or videos. Rather than proposing new benchmarks or APIs, our contribution is to outline design principles for a scalable, community-driven open-source project. In StepEval, the primary artifact for policy evaluation is the per-subgoal SR vector; however, other quantities (e.g., latency or cost estimates) are also considered for framework-optimization diagnostics to help the community tune evaluation efficiency and accuracy when ground-truth subgoal success labels are available. We discuss how such a framework can remain model-agnostic, support single- or multi-view inputs, and be lightweight enough to adopt across labs. The intended contribution is a shared direction: a minimal, extensible seed that invites open-source contributions, so that scoring the steps, not just the final goal, becomes a standard and reproducible practice.
>
---
#### [new 065] On Robustness of Consensus over Pseudo-Undirected Path Graphs
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; math.DS; math.OC**

- **简介: 该论文研究多智能体系统在伪无向路径图上的共识问题，提出一种非对称但双向连通的网络框架，允许边权为负且有界。通过分析拉普拉斯矩阵，扩展了可达共识值范围，并应用于移动目标协同拦截任务。**

- **链接: [http://arxiv.org/pdf/2509.20314v1](http://arxiv.org/pdf/2509.20314v1)**

> **作者:** Abhinav Sinha; Dwaipayan Mukherjee; Shashi Ranjan Kumar
>
> **摘要:** Consensus over networked agents is typically studied using undirected or directed communication graphs. Undirected graphs enforce symmetry in information exchange, leading to convergence to the average of initial states, while directed graphs permit asymmetry but make consensus dependent on root nodes and their influence. Both paradigms impose inherent restrictions on achievable consensus values and network robustness. This paper introduces a theoretical framework for achieving consensus over a class of network topologies, termed pseudo-undirected graphs, which retains bidirectional connectivity between node pairs but allows the corresponding edge weights to differ, including the possibility of negative values under bounded conditions. The resulting Laplacian is generally non-symmetric, yet it guarantees consensus under connectivity assumptions, to expand the solution space, which enables the system to achieve a stable consensus value that can lie outside the convex hull of the initial state set. We derive admissibility bounds for negative weights for a pseudo-undirected path graph, and show an application in the simultaneous interception of a moving target.
>
---
#### [new 066] Embodied AI: From LLMs to World Models
- **分类: cs.AI; cs.CL; cs.RO**

- **简介: 该论文综述了具身AI的发展，重点探讨大语言模型（LLMs）和世界模型（WMs）在实现AGI中的作用。论文梳理关键技术、架构及应用，并提出融合MLLM与WM的联合架构对未来复杂任务的重要性。**

- **链接: [http://arxiv.org/pdf/2509.20021v1](http://arxiv.org/pdf/2509.20021v1)**

> **作者:** Tongtong Feng; Xin Wang; Yu-Gang Jiang; Wenwu Zhu
>
> **备注:** Accepted by IEEE CASM
>
> **摘要:** Embodied Artificial Intelligence (AI) is an intelligent system paradigm for achieving Artificial General Intelligence (AGI), serving as the cornerstone for various applications and driving the evolution from cyberspace to physical systems. Recent breakthroughs in Large Language Models (LLMs) and World Models (WMs) have drawn significant attention for embodied AI. On the one hand, LLMs empower embodied AI via semantic reasoning and task decomposition, bringing high-level natural language instructions and low-level natural language actions into embodied cognition. On the other hand, WMs empower embodied AI by building internal representations and future predictions of the external world, facilitating physical law-compliant embodied interactions. As such, this paper comprehensively explores the literature in embodied AI from basics to advances, covering both LLM driven and WM driven works. In particular, we first present the history, key technologies, key components, and hardware systems of embodied AI, as well as discuss its development via looking from unimodal to multimodal angle. We then scrutinize the two burgeoning fields of embodied AI, i.e., embodied AI with LLMs/multimodal LLMs (MLLMs) and embodied AI with WMs, meticulously delineating their indispensable roles in end-to-end embodied cognition and physical laws-driven embodied interactions. Building upon the above advances, we further share our insights on the necessity of the joint MLLM-WM driven embodied AI architecture, shedding light on its profound significance in enabling complex tasks within physical worlds. In addition, we examine representative applications of embodied AI, demonstrating its wide applicability in real-world scenarios. Last but not least, we point out future research directions of embodied AI that deserve further investigation.
>
---
## 更新

#### [replaced 001] Optimal Multi-agent Path Finding in Continuous Time
- **分类: cs.MA; cs.DM; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.16410v2](http://arxiv.org/pdf/2508.16410v2)**

> **作者:** Alvin Combrink; Sabino Francesco Roselli; Martin Fabian
>
> **备注:** 35 pages
>
> **摘要:** Continuous-time Conflict Based-Search (CCBS) has long been viewed as the standard optimal baseline for multi-agent path finding in continuous time (MAPFR), yet recent critiques show that the theoretically described CCBS can fail to terminate on solvable MAPFR problems while the publicly available reference implementation can return sub-optimal solutions. This work presents an analytical framework that yields simple and sufficient conditions under which any CCBS-style algorithm is both sound and solution complete. Investigating the reference CCBS implementation reveals that it violates our sufficient conditions for soundness, with counterexamples demonstrating sub-optimality. Leveraging the framework, we introduce a branching rule ($\delta$-BR) and prove it restores soundness and termination guarantees. Consequently, the resulting CCBS variant is both sound and solution complete. To our knowledge, this is the first MAPFR solver matching the guarantees of the discrete-time CBS. On a constructed example, CCBS with $\delta$-BR improves sum-of-costs from 10.707 to 9.000 ($\approx$ 16% lower) compared to the reference CCBS implementation. Across benchmarks, the reference CCBS implementation is generally able to find solutions faster than CCBS with $\delta$-BR due to its more aggressive pruning. However, this comes at the cost of occasional sub-optimality and potential non-termination when all solutions are pruned, whereas $\delta$-BR preserves optimality and guarantees termination by design. Because $\delta$-BR largely only affects the branching step, it can be adopted as a drop-in replacement in existing codebases. Beyond CCBS, the analytical framework and termination criterion provide a systematic way to evaluate other CCBS-like MAPFR solvers and future extensions, thereby offering tools for rigorous analysis of next-generation MAPFR algorithms.
>
---
#### [replaced 002] Online Language Splatting
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.09447v2](http://arxiv.org/pdf/2503.09447v2)**

> **作者:** Saimouli Katragadda; Cho-Ying Wu; Yuliang Guo; Xinyu Huang; Guoquan Huang; Liu Ren
>
> **摘要:** To enable AI agents to interact seamlessly with both humans and 3D environments, they must not only perceive the 3D world accurately but also align human language with 3D spatial representations. While prior work has made significant progress by integrating language features into geometrically detailed 3D scene representations using 3D Gaussian Splatting (GS), these approaches rely on computationally intensive offline preprocessing of language features for each input image, limiting adaptability to new environments. In this work, we introduce Online Language Splatting, the first framework to achieve online, near real-time, open-vocabulary language mapping within a 3DGS-SLAM system without requiring pre-generated language features. The key challenge lies in efficiently fusing high-dimensional language features into 3D representations while balancing the computation speed, memory usage, rendering quality and open-vocabulary capability. To this end, we innovatively design: (1) a high-resolution CLIP embedding module capable of generating detailed language feature maps in 18ms per frame, (2) a two-stage online auto-encoder that compresses 768-dimensional CLIP features to 15 dimensions while preserving open-vocabulary capabilities, and (3) a color-language disentangled optimization approach to improve rendering quality. Experimental results show that our online method not only surpasses the state-of-the-art offline methods in accuracy but also achieves more than 40x efficiency boost, demonstrating the potential for dynamic and interactive AI applications.
>
---
#### [replaced 003] Human-Interpretable Uncertainty Explanations for Point Cloud Registration
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.18786v2](http://arxiv.org/pdf/2509.18786v2)**

> **作者:** Johannes A. Gaus; Loris Schneider; Yitian Shi; Jongseok Lee; Rania Rayyes; Rudolph Triebel
>
> **摘要:** In this paper, we address the point cloud registration problem, where well-known methods like ICP fail under uncertainty arising from sensor noise, pose-estimation errors, and partial overlap due to occlusion. We develop a novel approach, Gaussian Process Concept Attribution (GP-CA), which not only quantifies registration uncertainty but also explains it by attributing uncertainty to well-known sources of errors in registration problems. Our approach leverages active learning to discover new uncertainty sources in the wild by querying informative instances. We validate GP-CA on three publicly available datasets and in our real-world robot experiment. Extensive ablations substantiate our design choices. Our approach outperforms other state-of-the-art methods in terms of runtime, high sample-efficiency with active learning, and high accuracy. Our real-world experiment clearly demonstrates its applicability. Our video also demonstrates that GP-CA enables effective failure-recovery behaviors, yielding more robust robotic perception.
>
---
#### [replaced 004] Uncertainty-aware Latent Safety Filters for Avoiding Out-of-Distribution Failures
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.00779v2](http://arxiv.org/pdf/2505.00779v2)**

> **作者:** Junwon Seo; Kensuke Nakamura; Andrea Bajcsy
>
> **备注:** Conference on Robot Learning (CoRL 2025)
>
> **摘要:** Recent advances in generative world models have enabled classical safe control methods, such as Hamilton-Jacobi (HJ) reachability, to generalize to complex robotic systems operating directly from high-dimensional sensor observations. However, obtaining comprehensive coverage of all safety-critical scenarios during world model training is extremely challenging. As a result, latent safety filters built on top of these models may miss novel hazards and even fail to prevent known ones, overconfidently misclassifying risky out-of-distribution (OOD) situations as safe. To address this, we introduce an uncertainty-aware latent safety filter that proactively steers robots away from both known and unseen failures. Our key idea is to use the world model's epistemic uncertainty as a proxy for identifying unseen potential hazards. We propose a principled method to detect OOD world model predictions by calibrating an uncertainty threshold via conformal prediction. By performing reachability analysis in an augmented state space-spanning both the latent representation and the epistemic uncertainty-we synthesize a latent safety filter that can reliably safeguard arbitrary policies from both known and unseen safety hazards. In simulation and hardware experiments on vision-based control tasks with a Franka manipulator, we show that our uncertainty-aware safety filter preemptively detects potential unsafe scenarios and reliably proposes safe, in-distribution actions. Video results can be found on the project website at https://cmu-intentlab.github.io/UNISafe
>
---
#### [replaced 005] AORRTC: Almost-Surely Asymptotically Optimal Planning with RRT-Connect
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.10542v4](http://arxiv.org/pdf/2505.10542v4)**

> **作者:** Tyler Wilson; Wil Thomason; Zachary Kingston; Jonathan Gammell
>
> **备注:** IEEE Robotics and Automation Letters (RA-L). 8 pages, 4 figures, 1 table. A video of AORRTC can be found at https://www.youtube.com/watch?v=j1itxP3KuiM . Information on the implementation of AORRTC is available at https://robotic-esp.com/code/aorrtc/
>
> **摘要:** Finding high-quality solutions quickly is an important objective in motion planning. This is especially true for high-degree-of-freedom robots. Satisficing planners have traditionally found feasible solutions quickly but provide no guarantees on their optimality, while almost-surely asymptotically optimal (a.s.a.o.) planners have probabilistic guarantees on their convergence towards an optimal solution but are more computationally expensive. This paper uses the AO-x meta-algorithm to extend the satisficing RRT-Connect planner to optimal planning. The resulting Asymptotically Optimal RRT-Connect (AORRTC) finds initial solutions in similar times as RRT-Connect and uses any additional planning time to converge towards the optimal solution in an anytime manner. It is proven to be probabilistically complete and a.s.a.o. AORRTC was tested with the Panda (7 DoF) and Fetch (8 DoF) robotic arms on the MotionBenchMaker dataset. These experiments show that AORRTC finds initial solutions as fast as RRT-Connect and faster than the tested state-of-the-art a.s.a.o. algorithms while converging to better solutions faster. AORRTC finds solutions to difficult high-DoF planning problems in milliseconds where the other a.s.a.o. planners could not consistently find solutions in seconds. This performance was demonstrated both with and without single instruction/multiple data (SIMD) acceleration.
>
---
#### [replaced 006] Design optimization and robustness analysis of rigid-link flapping mechanisms
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.21204v2](http://arxiv.org/pdf/2503.21204v2)**

> **作者:** Shyam Sunder Nishad; Anupam Saxena
>
> **摘要:** Rigid link flapping mechanisms remain the most practical choice for flapping wing micro-aerial vehicles (MAVs) to carry useful payloads and onboard batteries for free flight due to their long-term durability and reliability. However, MAVs with these mechanisms require significant weight reduction to achieve high agility and maneuverability. One approach involves using single-DOF planar rigid linkages, which are rarely optimized dimensionally for high lift and low power, considering their sweeping kinematics and the unsteady aerodynamic effects. We integrated a mechanism simulator based on a quasistatic nonlinear finite element method with an unsteady vortex lattice method-based aerodynamic analysis tool within an optimization routine. We optimized three different mechanism topologies from the literature. Significant power savings were observed up to 34% in some cases, due to increased amplitude and higher lift coefficients resulting from optimized asymmetric sweeping velocity profiles. We also conducted a robustness analysis to quantify performance sensitivity to manufacturing tolerances. It provided a trade-off between performance and reliability and revealed the need for tight manufacturing tolerances and careful material selection. Finally, the analysis helped select the best mechanism topology, as we observed significant variation in sensitivity to manufacturing tolerances and peak input torque values across different topologies for a given design lift value. The presented unified computational tool can find application in flapping mechanism topology optimization, as it can simulate any generic single-DOF planar rigid linkage without supplying kinematics manually.
>
---
#### [replaced 007] CUPID: Curating Data your Robot Loves with Influence Functions
- **分类: cs.RO; cs.AI; cs.LG; I.2.6; I.2.9**

- **链接: [http://arxiv.org/pdf/2506.19121v2](http://arxiv.org/pdf/2506.19121v2)**

> **作者:** Christopher Agia; Rohan Sinha; Jingyun Yang; Rika Antonova; Marco Pavone; Haruki Nishimura; Masha Itkina; Jeannette Bohg
>
> **备注:** Project page: https://cupid-curation.github.io. 27 pages, 15 figures. Accepted to the Conference on Robot Learning (CoRL) 2025
>
> **摘要:** In robot imitation learning, policy performance is tightly coupled with the quality and composition of the demonstration data. Yet, developing a precise understanding of how individual demonstrations contribute to downstream outcomes - such as closed-loop task success or failure - remains a persistent challenge. We propose CUPID, a robot data curation method based on a novel influence function-theoretic formulation for imitation learning policies. Given a set of evaluation rollouts, CUPID estimates the influence of each training demonstration on the policy's expected return. This enables ranking and selection of demonstrations according to their impact on the policy's closed-loop performance. We use CUPID to curate data by 1) filtering out training demonstrations that harm policy performance and 2) subselecting newly collected trajectories that will most improve the policy. Extensive simulated and hardware experiments show that our approach consistently identifies which data drives test-time performance. For example, training with less than 33% of curated data can yield state-of-the-art diffusion policies on the simulated RoboMimic benchmark, with similar gains observed in hardware. Furthermore, hardware experiments show that our method can identify robust strategies under distribution shift, isolate spurious correlations, and even enhance the post-training of generalist robot policies. Videos and code are made available at: https://cupid-curation.github.io.
>
---
#### [replaced 008] CHILD (Controller for Humanoid Imitation and Live Demonstration): a Whole-Body Humanoid Teleoperation System
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.00162v2](http://arxiv.org/pdf/2508.00162v2)**

> **作者:** Noboru Myers; Obin Kwon; Sankalp Yamsani; Joohyung Kim
>
> **备注:** 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids)
>
> **摘要:** Recent advances in teleoperation have demonstrated robots performing complex manipulation tasks. However, existing works rarely support whole-body joint-level teleoperation for humanoid robots, limiting the diversity of tasks that can be accomplished. This work presents Controller for Humanoid Imitation and Live Demonstration (CHILD), a compact reconfigurable teleoperation system that enables joint level control over humanoid robots. CHILD fits within a standard baby carrier, allowing the operator control over all four limbs, and supports both direct joint mapping for full-body control and loco-manipulation. Adaptive force feedback is incorporated to enhance operator experience and prevent unsafe joint movements. We validate the capabilities of this system by conducting loco-manipulation and full-body control demonstrations on a humanoid robot and multiple dual-arm systems. Lastly, we open-source the design of the hardware promoting accessibility and reproducibility. Additional details and open-source information are available at our project website: https://uiuckimlab.github.io/CHILD-pages.
>
---
#### [replaced 009] RG-Attn: Radian Glue Attention for Multi-modality Multi-agent Cooperative Perception
- **分类: cs.RO; cs.CV; cs.NI; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.16803v3](http://arxiv.org/pdf/2501.16803v3)**

> **作者:** Lantao Li; Kang Yang; Wenqi Zhang; Xiaoxue Wang; Chen Sun
>
> **备注:** Accepted by ICCV 2025 DriveX workshop (Final Version)
>
> **摘要:** Cooperative perception enhances autonomous driving by leveraging Vehicle-to-Everything (V2X) communication for multi-agent sensor fusion. However, most existing methods rely on single-modal data sharing, limiting fusion performance, particularly in heterogeneous sensor settings involving both LiDAR and cameras across vehicles and roadside units (RSUs). To address this, we propose Radian Glue Attention (RG-Attn), a lightweight and generalizable cross-modal fusion module that unifies intra-agent and inter-agent fusion via transformation-based coordinate alignment and a unified sampling/inversion strategy. RG-Attn efficiently aligns features through a radian-based attention constraint, operating column-wise on geometrically consistent regions to reduce overhead and preserve spatial coherence, thereby enabling accurate and robust fusion. Building upon RG-Attn, we propose three cooperative architectures. The first, Paint-To-Puzzle (PTP), prioritizes communication efficiency but assumes all agents have LiDAR, optionally paired with cameras. The second, Co-Sketching-Co-Coloring (CoS-CoCo), offers maximal flexibility, supporting any sensor setup (e.g., LiDAR-only, camera-only, or both) and enabling strong cross-modal generalization for real-world deployment. The third, Pyramid-RG-Attn Fusion (PRGAF), aims for peak detection accuracy with the highest computational overhead. Extensive evaluations on simulated and real-world datasets show our framework delivers state-of-the-art detection accuracy with high flexibility and efficiency. GitHub Link: https://github.com/LantaoLi/RG-Attn
>
---
#### [replaced 010] Data-fused Model Predictive Control with Guarantees: Application to Flying Humanoid Robots
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2509.10353v3](http://arxiv.org/pdf/2509.10353v3)**

> **作者:** Davide Gorbani; Mohamed Elobaid; Giuseppe L'Erario; Hosameldin Awadalla Omer Mohamed; Daniele Pucci
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** This paper introduces a Data-Fused Model Predictive Control (DFMPC) framework that combines physics-based models with data-driven representations of unknown dynamics. Leveraging Willems' Fundamental Lemma and an artificial equilibrium formulation, the method enables tracking of changing, potentially unreachable setpoints while explicitly handling measurement noise through slack variables and regularization. We provide guarantees of recursive feasibility and practical stability under input-output constraints for a specific class of reference signals. The approach is validated on the iRonCub flying humanoid robot, integrating analytical momentum models with data-driven turbine dynamics. Simulations show improved tracking and robustness compared to a purely model-based MPC, while maintaining real-time feasibility.
>
---
#### [replaced 011] Learning to Drive by Imitating Surrounding Vehicles
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.05997v2](http://arxiv.org/pdf/2503.05997v2)**

> **作者:** Yasin Sonmez; Hanna Krasowski; Murat Arcak
>
> **摘要:** Imitation learning is a promising approach for training autonomous vehicles (AV) to navigate complex traffic environments by mimicking expert driver behaviors. While existing imitation learning frameworks focus on leveraging expert demonstrations, they often overlook the potential of additional complex driving data from surrounding traffic participants. In this paper, we study a data augmentation strategy that leverages the observed trajectories of nearby vehicles, captured by the AV's sensors, as additional demonstrations. We introduce a simple vehicle-selection sampling and filtering strategy that prioritizes informative and diverse driving behaviors, contributing to a richer dataset for training. We evaluate this idea with a representative learning-based planner on a large real-world dataset and demonstrate improved performance in complex driving scenarios. Specifically, the approach reduces collision rates and improves safety metrics compared to the baseline. Notably, even when using only 10 percent of the original dataset, the method matches or exceeds the performance of the full dataset. Through ablations, we analyze selection criteria and show that naive random selection can degrade performance. Our findings highlight the value of leveraging diverse real-world trajectory data in imitation learning and provide insights into data augmentation strategies for autonomous driving.
>
---
#### [replaced 012] SoFar: Language-Grounded Orientation Bridges Spatial Reasoning and Object Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.13143v2](http://arxiv.org/pdf/2502.13143v2)**

> **作者:** Zekun Qi; Wenyao Zhang; Yufei Ding; Runpei Dong; Xinqiang Yu; Jingwen Li; Lingyun Xu; Baoyu Li; Xialin He; Guofan Fan; Jiazhao Zhang; Jiawei He; Jiayuan Gu; Xin Jin; Kaisheng Ma; Zhizheng Zhang; He Wang; Li Yi
>
> **备注:** Accepted at NeurIPS 2025 Spotlight
>
> **摘要:** While spatial reasoning has made progress in object localization relationships, it often overlooks object orientation-a key factor in 6-DoF fine-grained manipulation. Traditional pose representations rely on pre-defined frames or templates, limiting generalization and semantic grounding. In this paper, we introduce the concept of semantic orientation, which defines object orientations using natural language in a reference-frame-free manner (e.g., the "plug-in" direction of a USB or the "handle" direction of a cup). To support this, we construct OrienText300K, a large-scale dataset of 3D objects annotated with semantic orientations, and develop PointSO, a general model for zero-shot semantic orientation prediction. By integrating semantic orientation into VLM agents, our SoFar framework enables 6-DoF spatial reasoning and generates robotic actions. Extensive experiments demonstrated the effectiveness and generalization of our SoFar, e.g., zero-shot 48.7% successful rate on Open6DOR and zero-shot 74.9% successful rate on SIMPLER-Env.
>
---
#### [replaced 013] Active Shadowing (ASD): Manipulating Perception of Robotic Behaviors via Implicit Virtual Communication
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.01468v3](http://arxiv.org/pdf/2407.01468v3)**

> **作者:** Andrew Boateng; Prakhar Bhartiya; Taha Shaheen; Yu Zhang
>
> **摘要:** Explicit communication is often valued for its directness in presenting information but requires attention during exchange, resulting in cognitive interruptions. On the other hand, implicit communication contributes to tacit and smooth interaction, making it more suitable for teaming, but requires inference for interpretation. This paper studies a novel type of implicit visual communication (IVC) using shadows via visual projection with augmented reality, referred to as active shadowing (ASD). Prior IVC methods, such as legible motion, are often used to influence the perception of robot behavior to make it more understandable. They often require changing the physical robot behavior, resulting in suboptimality. In our work, we investigate how ASD can be used to achieve similar effects without losing optimality. Our evaluations with user studies demonstrates that ASD can effectively creates ''illusions'' that maintain optimal physical behavior without compromising its understandability. We also show that ASD can be more informative than other explicit communication methods, and examine the conditions under which ASD becomes less effective.
>
---
#### [replaced 014] Do You Need Proprioceptive States in Visuomotor Policies?
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18644v2](http://arxiv.org/pdf/2509.18644v2)**

> **作者:** Juntu Zhao; Wenbo Lu; Di Zhang; Yufeng Liu; Yushen Liang; Tianluo Zhang; Yifeng Cao; Junyuan Xie; Yingdong Hu; Shengjie Wang; Junliang Guo; Dequan Wang; Yang Gao
>
> **备注:** Project page: https://statefreepolicy.github.io
>
> **摘要:** Imitation-learning-based visuomotor policies have been widely used in robot manipulation, where both visual observations and proprioceptive states are typically adopted together for precise control. However, in this study, we find that this common practice makes the policy overly reliant on the proprioceptive state input, which causes overfitting to the training trajectories and results in poor spatial generalization. On the contrary, we propose the State-free Policy, removing the proprioceptive state input and predicting actions only conditioned on visual observations. The State-free Policy is built in the relative end-effector action space, and should ensure the full task-relevant visual observations, here provided by dual wide-angle wrist cameras. Empirical results demonstrate that the State-free policy achieves significantly stronger spatial generalization than the state-based policy: in real-world tasks such as pick-and-place, challenging shirt-folding, and complex whole-body manipulation, spanning multiple robot embodiments, the average success rate improves from 0% to 85% in height generalization and from 6% to 64% in horizontal generalization. Furthermore, they also show advantages in data efficiency and cross-embodiment adaptation, enhancing their practicality for real-world deployment. Discover more by visiting: https://statefreepolicy.github.io.
>
---
#### [replaced 015] Towards Data-Driven Adaptive Exoskeleton Assistance for Post-stroke Gait
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.00691v2](http://arxiv.org/pdf/2508.00691v2)**

> **作者:** Fabian C. Weigend; Dabin K. Choe; Santiago Canete; Conor J. Walsh
>
> **备注:** 8 pages, 6 figures, 2 tables
>
> **摘要:** Recent work has shown that exoskeletons controlled through data-driven methods can dynamically adapt assistance to various tasks for healthy young adults. However, applying these methods to populations with neuromotor gait deficits, such as post-stroke hemiparesis, is challenging. This is due not only to high population heterogeneity and gait variability but also to a lack of post-stroke gait datasets to train accurate models. Despite these challenges, data-driven methods offer a promising avenue for control, potentially allowing exoskeletons to function safely and effectively in unstructured community settings. This work presents a first step towards enabling adaptive plantarflexion and dorsiflexion assistance from data-driven torque estimation during post-stroke walking. We trained a multi-task Temporal Convolutional Network (TCN) using collected data from four post-stroke participants walking on a treadmill ($R^2$ of $0.74 \pm 0.13$). The model uses data from three inertial measurement units (IMU) and was pretrained on healthy walking data from 6 participants. We implemented a wearable prototype for our ankle torque estimation approach for exoskeleton control and demonstrated the viability of real-time sensing, estimation, and actuation with one post-stroke participant.
>
---
#### [replaced 016] U-ARM : Ultra low-cost general teleoperation interface for robot manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.02437v2](http://arxiv.org/pdf/2509.02437v2)**

> **作者:** Yanwen Zou; Zhaoye Zhou; Chenyang Shi; Zewei Ye; Junda Huang; Yan Ding; Bo Zhao
>
> **摘要:** We propose U-Arm, a low-cost and rapidly adaptable leader-follower teleoperation framework designed to interface with most of commercially available robotic arms. Our system supports teleoperation through three structurally distinct 3D-printed leader arms that share consistent control logic, enabling seamless compatibility with diverse commercial robot configurations. Compared with previous open-source leader-follower interfaces, we further optimized both the mechanical design and servo selection, achieving a bill of materials (BOM) cost of only \$50.5 for the 6-DoF leader arm and \$56.8 for the 7-DoF version. To enhance usability, we mitigate the common challenge in controlling redundant degrees of freedom by %engineering methods mechanical and control optimizations. Experimental results demonstrate that U-Arm achieves 39\% higher data collection efficiency and comparable task success rates across multiple manipulation scenarios compared with Joycon, another low-cost teleoperation interface. We have open-sourced all CAD models of three configs and also provided simulation support for validating teleoperation workflows. We also open-sourced real-world manipulation data collected with U-Arm. The project website is https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm.
>
---
#### [replaced 017] Coverage Path Planning for Holonomic UAVs via Uniaxial-Feasible, Gap-Severity Guided Decomposition
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2505.08060v2](http://arxiv.org/pdf/2505.08060v2)**

> **作者:** Pedro Antonio Alarcon Granadeno; Jane Cleland-Huang
>
> **备注:** 8 pages, 4 figures,
>
> **摘要:** Modern coverage path planning (CPP) for holonomic UAVs in emergency response must contend with diverse environments where regions of interest (ROIs) often take the form of highly irregular polygons, characterized by asymmetric shapes, dense clusters of concavities, and multiple internal holes. Modern CPP pipelines typically rely on decomposition strategies that overfragment such polygons into numerous subregions. This increases the number of sweep segments and connectors, which in turn adds inter-region travel and forces more frequent reorientation. These effects ultimately result in longer completion times and degraded trajectory quality. We address this with a decomposition strategy that applies a recursive dual-axis monotonicity criterion with cuts guided by a cumulative gap severity metric. This approach distributes clusters of concavities more evenly across subregions and produces a minimal set of partitions that remain sweepable under a parallel-track maneuver. We pair this with a global optimizer that jointly selects sweep paths and inter-partition transitions to minimize total path length, transition overhead, and turn count. We demonstrate that our proposed approach achieves the lowest mean overhead in path length and completion time across 13 notable CPP pipelines.
>
---
#### [replaced 018] VLM See, Robot Do: Human Demo Video to Robot Action Plan via Vision Language Model
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.08792v2](http://arxiv.org/pdf/2410.08792v2)**

> **作者:** Beichen Wang; Juexiao Zhang; Shuwen Dong; Irving Fang; Chen Feng
>
> **摘要:** Vision Language Models (VLMs) have recently been adopted in robotics for their capability in common sense reasoning and generalizability. Existing work has applied VLMs to generate task and motion planning from natural language instructions and simulate training data for robot learning. In this work, we explore using VLM to interpret human demonstration videos and generate robot task planning. Our method integrates keyframe selection, visual perception, and VLM reasoning into a pipeline. We named it SeeDo because it enables the VLM to ''see'' human demonstrations and explain the corresponding plans to the robot for it to ''do''. To validate our approach, we collected a set of long-horizon human videos demonstrating pick-and-place tasks in three diverse categories and designed a set of metrics to comprehensively benchmark SeeDo against several baselines, including state-of-the-art video-input VLMs. The experiments demonstrate SeeDo's superior performance. We further deployed the generated task plans in both a simulation environment and on a real robot arm.
>
---
#### [replaced 019] ExoStart: Efficient learning for dexterous manipulation with sensorized exoskeleton demonstrations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.11775v3](http://arxiv.org/pdf/2506.11775v3)**

> **作者:** Zilin Si; Jose Enrique Chen; M. Emre Karagozler; Antonia Bronars; Jonathan Hutchinson; Thomas Lampe; Nimrod Gileadi; Taylor Howell; Stefano Saliceti; Lukasz Barczyk; Ilan Olivarez Correa; Tom Erez; Mohit Shridhar; Murilo Fernandes Martins; Konstantinos Bousmalis; Nicolas Heess; Francesco Nori; Maria Bauza
>
> **摘要:** Recent advancements in teleoperation systems have enabled high-quality data collection for robotic manipulators, showing impressive results in learning manipulation at scale. This progress suggests that extending these capabilities to robotic hands could unlock an even broader range of manipulation skills, especially if we could achieve the same level of dexterity that human hands exhibit. However, teleoperating robotic hands is far from a solved problem, as it presents a significant challenge due to the high degrees of freedom of robotic hands and the complex dynamics occurring during contact-rich settings. In this work, we present ExoStart, a general and scalable learning framework that leverages human dexterity to improve robotic hand control. In particular, we obtain high-quality data by collecting direct demonstrations without a robot in the loop using a sensorized low-cost wearable exoskeleton, capturing the rich behaviors that humans can demonstrate with their own hands. We also propose a simulation-based dynamics filter that generates dynamically feasible trajectories from the collected demonstrations and use the generated trajectories to bootstrap an auto-curriculum reinforcement learning method that relies only on simple sparse rewards. The ExoStart pipeline is generalizable and yields robust policies that transfer zero-shot to the real robot. Our results demonstrate that ExoStart can generate dexterous real-world hand skills, achieving a success rate above 50% on a wide range of complex tasks such as opening an AirPods case or inserting and turning a key in a lock. More details and videos can be found in https://sites.google.com/view/exostart.
>
---
#### [replaced 020] GraphEQA: Using 3D Semantic Scene Graphs for Real-time Embodied Question Answering
- **分类: cs.RO; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.14480v2](http://arxiv.org/pdf/2412.14480v2)**

> **作者:** Saumya Saxena; Blake Buchanan; Chris Paxton; Peiqi Liu; Bingqing Chen; Narunas Vaskevicius; Luigi Palmieri; Jonathan Francis; Oliver Kroemer
>
> **备注:** Project website: https://saumyasaxena.github.io/grapheqa
>
> **摘要:** In Embodied Question Answering (EQA), agents must explore and develop a semantic understanding of an unseen environment to answer a situated question with confidence. This problem remains challenging in robotics, due to the difficulties in obtaining useful semantic representations, updating these representations online, and leveraging prior world knowledge for efficient planning and exploration. To address these limitations, we propose GraphEQA, a novel approach that utilizes real-time 3D metric-semantic scene graphs (3DSGs) and task relevant images as multi-modal memory for grounding Vision-Language Models (VLMs) to perform EQA tasks in unseen environments. We employ a hierarchical planning approach that exploits the hierarchical nature of 3DSGs for structured planning and semantics-guided exploration. We evaluate GraphEQA in simulation on two benchmark datasets, HM-EQA and OpenEQA, and demonstrate that it outperforms key baselines by completing EQA tasks with higher success rates and fewer planning steps. We further demonstrate GraphEQA in multiple real-world home and office environments.
>
---
#### [replaced 021] SEM: Enhancing Spatial Understanding for Robust Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16196v3](http://arxiv.org/pdf/2505.16196v3)**

> **作者:** Xuewu Lin; Tianwei Lin; Lichao Huang; Hongyu Xie; Yiwei Jin; Keyu Li; Zhizhong Su
>
> **摘要:** A key challenge in robot manipulation lies in developing policy models with strong spatial understanding, the ability to reason about 3D geometry, object relations, and robot embodiment. Existing methods often fall short: 3D point cloud models lack semantic abstraction, while 2D image encoders struggle with spatial reasoning. To address this, we propose SEM (Spatial Enhanced Manipulation model), a novel diffusion-based policy framework that explicitly enhances spatial understanding from two complementary perspectives. A spatial enhancer augments visual representations with 3D geometric context, while a robot state encoder captures embodiment-aware structure through graphbased modeling of joint dependencies. By integrating these modules, SEM significantly improves spatial understanding, leading to robust and generalizable manipulation across diverse tasks that outperform existing baselines.
>
---
#### [replaced 022] ByteWrist: A Parallel Robotic Wrist Enabling Flexible and Anthropomorphic Motion for Confined Spaces
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.18084v2](http://arxiv.org/pdf/2509.18084v2)**

> **作者:** Jiawen Tian; Liqun Huang; Zhongren Cui; Jingchao Qiao; Jiafeng Xu; Xiao Ma; Zeyu Ren
>
> **备注:** Tech Report.13 pages, 9 figures. Project page: https://bytewrist.github.io/
>
> **摘要:** This paper introduces ByteWrist, a novel highly-flexible and anthropomorphic parallel wrist for robotic manipulation. ByteWrist addresses the critical limitations of existing serial and parallel wrists in narrow-space operations through a compact three-stage parallel drive mechanism integrated with arc-shaped end linkages. The design achieves precise RPY (Roll-Pitch-Yaw) motion while maintaining exceptional compactness, making it particularly suitable for complex unstructured environments such as home services, medical assistance, and precision assembly. The key innovations include: (1) a nested three-stage motor-driven linkages that minimize volume while enabling independent multi-DOF control, (2) arc-shaped end linkages that optimize force transmission and expand motion range, and (3) a central supporting ball functioning as a spherical joint that enhances structural stiffness without compromising flexibility. Meanwhile, we present comprehensive kinematic modeling including forward / inverse kinematics and a numerical Jacobian solution for precise control. Empirically, we observe ByteWrist demonstrates strong performance in narrow-space maneuverability and dual-arm cooperative manipulation tasks, outperforming Kinova-based systems. Results indicate significant improvements in compactness, efficiency, and stiffness compared to traditional designs, establishing ByteWrist as a promising solution for next-generation robotic manipulation in constrained environments.
>
---
#### [replaced 023] ASC-SW: Atrous strip convolution network with sliding windows
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12744v2](http://arxiv.org/pdf/2507.12744v2)**

> **作者:** Cheng Liu; Fan Zhu; Yifeng Xu; Baoru Huang; Mohd Rizal Arshad
>
> **摘要:** With the rapid development of lightweight visual neural network architectures, traditional high-performance vision models have undergone significant compression, enhancing their computational and energy efficiency and enabling deployment on resource-constrained edge devices. In order to enable the mobile robot to avoid the ground wires, we propose a visual-assisted navigation framework called Atrous Strip Convolution Sliding Window (ASC-SW). This framework compensates for the limitations of traditional light detection and range (LiDAR) sensors to detect ground-level obstacles such as wires. A lightweight and efficient segmentation model, Atrous Strip Convolution Network (ASCnet) was proposed, for detecting deformable linear objects (DLOs). Atrous Strip Convolution Spatial Pyramid Pooling (ASCSPP) is designed to extract DLOs features effectively. Atrous Strip Convolution is integrated into ASCSPP to accurately identify the linear structure of DLOs with low computational cost. Additionally, a Sliding Window (SW) post processing module is proposed to denoise the output in complex environments, improving recognition accuracy. ASC-SW achieves 75.3% MIoU at 217 FPS on a self-built real world dataset and real-robot experiment was demonstrated that our proposed framework. It can be successfully verified on the real-robot on the edge device(Jetson platform) at that were originally inoperable.
>
---
#### [replaced 024] AURA: Autonomous Upskilling with Retrieval-Augmented Agents
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02507v2](http://arxiv.org/pdf/2506.02507v2)**

> **作者:** Alvin Zhu; Yusuke Tanaka; Andrew Goldberg; Dennis Hong
>
> **摘要:** Designing reinforcement learning curricula for agile robots traditionally requires extensive manual tuning of reward functions, environment randomizations, and training configurations. We introduce AURA (Autonomous Upskilling with Retrieval-Augmented Agents), a schema-validated curriculum reinforcement learning (RL) framework that leverages Large Language Models (LLMs) as autonomous designers of multi-stage curricula. AURA transforms user prompts into YAML workflows that encode full reward functions, domain randomization strategies, and training configurations. All files are statically validated before any GPU time is used, ensuring efficient and reliable execution. A retrieval-augmented feedback loop allows specialized LLM agents to design, execute, and refine curriculum stages based on prior training results stored in a vector database, enabling continual improvement over time. Quantitative experiments show that AURA consistently outperforms LLM-guided baselines in generation success rate, humanoid locomotion, and manipulation tasks. Ablation studies highlight the importance of schema validation and retrieval for curriculum quality. AURA successfully trains end-to-end policies directly from user prompts and deploys them zero-shot on a custom humanoid robot in multiple environments - capabilities that did not exist previously with manually designed controllers. By abstracting the complexity of curriculum design, AURA enables scalable and adaptive policy learning pipelines that would be complex to construct by hand.
>
---
