# 机器人 cs.RO

- **最新发布 32 篇**

- **更新 35 篇**

## 最新发布

#### [new 001] BEAVR: Bimanual, multi-Embodiment, Accessible, Virtual Reality Teleoperation System for Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种开源的双人多形态VR远程操作系统BEAVR，解决跨机器人平台实时控制与数据记录问题，通过零拷贝架构和异步控制实现低延迟，支持多模态演示记录，并优化网络API适配多机器人协作。**

- **链接: [http://arxiv.org/pdf/2508.09606v1](http://arxiv.org/pdf/2508.09606v1)**

> **作者:** Alejandro Posadas-Nava; Alejandro Carrasco; Richard Linares
>
> **备注:** Accepted for presentation on ICCR Kyoto 2025
>
> **摘要:** \textbf{BEAVR} is an open-source, bimanual, multi-embodiment Virtual Reality (VR) teleoperation system for robots, designed to unify real-time control, data recording, and policy learning across heterogeneous robotic platforms. BEAVR enables real-time, dexterous teleoperation using commodity VR hardware, supports modular integration with robots ranging from 7-DoF manipulators to full-body humanoids, and records synchronized multi-modal demonstrations directly in the LeRobot dataset schema. Our system features a zero-copy streaming architecture achieving $\leq$35\,ms latency, an asynchronous ``think--act'' control loop for scalable inference, and a flexible network API optimized for real-time, multi-robot operation. We benchmark BEAVR across diverse manipulation tasks and demonstrate its compatibility with leading visuomotor policies such as ACT, DiffusionPolicy, and SmolVLA. All code is publicly available, and datasets are released on Hugging Face\footnote{Code, datasets, and VR app available at https://github.com/ARCLab-MIT/BEAVR-Bot.
>
---
#### [new 002] DAgger Diffusion Navigation: DAgger Boosted Diffusion Policy for Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出一种基于扩散模型的视觉-语言导航方法DifNav，解决传统两阶段框架的全局子优化和依赖起点的问题，通过单阶段扩散政策直接建模多模态动作分布，结合DAgger实现在线训练与轨迹增强，显著提升导航性能。**

- **链接: [http://arxiv.org/pdf/2508.09444v1](http://arxiv.org/pdf/2508.09444v1)**

> **作者:** Haoxiang Shi; Xiang Deng; Zaijing Li; Gongwei Chen; Yaowei Wang; Liqiang Nie
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLN-CE) requires agents to follow natural language instructions through free-form 3D spaces. Existing VLN-CE approaches typically use a two-stage waypoint planning framework, where a high-level waypoint predictor generates the navigable waypoints, and then a navigation planner suggests the intermediate goals in the high-level action space. However, this two-stage decomposition framework suffers from: (1) global sub-optimization due to the proxy objective in each stage, and (2) a performance bottleneck caused by the strong reliance on the quality of the first-stage predicted waypoints. To address these limitations, we propose DAgger Diffusion Navigation (DifNav), an end-to-end optimized VLN-CE policy that unifies the traditional two stages, i.e. waypoint generation and planning, into a single diffusion policy. Notably, DifNav employs a conditional diffusion policy to directly model multi-modal action distributions over future actions in continuous navigation space, eliminating the need for a waypoint predictor while enabling the agent to capture multiple possible instruction-following behaviors. To address the issues of compounding error in imitation learning and enhance spatial reasoning in long-horizon navigation tasks, we employ DAgger for online policy training and expert trajectory augmentation, and use the aggregated data to further fine-tune the policy. This approach significantly improves the policy's robustness and its ability to recover from error states. Extensive experiments on benchmark datasets demonstrate that, even without a waypoint predictor, the proposed method substantially outperforms previous state-of-the-art two-stage waypoint-based models in terms of navigation performance. Our code is available at: https://github.com/Tokishx/DifNav.
>
---
#### [new 003] Reactive Model Predictive Contouring Control for Robot Manipulators
- **分类: cs.RO**

- **简介: 论文提出基于Reactive Model Predictive Contouring Control（RMPCC）的机器人路径跟随框架，解决动态环境中避障、奇异性和自碰撞问题，通过CBFs与雅可比线性化实现高速控制，实验验证其在低误差和低加速度下有效处理动态障碍物。**

- **链接: [http://arxiv.org/pdf/2508.09502v1](http://arxiv.org/pdf/2508.09502v1)**

> **作者:** Junheon Yoon; Woo-Jeong Baek; Jaeheung Park
>
> **备注:** 8 pages, 7 figures, 3 tables, conference paper, Accepted for publication at IEEE/RSJ International Conference on Intelligent Robots and Systems(IROS) 2025
>
> **摘要:** This contribution presents a robot path-following framework via Reactive Model Predictive Contouring Control (RMPCC) that successfully avoids obstacles, singularities and self-collisions in dynamic environments at 100 Hz. Many path-following methods rely on the time parametrization, but struggle to handle collision and singularity avoidance while adhering kinematic limits or other constraints. Specifically, the error between the desired path and the actual position can become large when executing evasive maneuvers. Thus, this paper derives a method that parametrizes the reference path by a path parameter and performs the optimization via RMPCC. In particular, Control Barrier Functions (CBFs) are introduced to avoid collisions and singularities in dynamic environments. A Jacobian-based linearization and Gauss-Newton Hessian approximation enable solving the nonlinear RMPCC problem at 100 Hz, outperforming state-of-the-art methods by a factor of 10. Experiments confirm that the framework handles dynamic obstacles in real-world settings with low contouring error and low robot acceleration.
>
---
#### [new 004] Toward Human-Robot Teaming: Learning Handover Behaviors from 3D Scenes
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文提出一种基于RGB图像和3D场景重建的机器人手部交接学习方法，解决模拟与真实环境视觉域差距及数据采集成本高的问题，通过相机位姿映射生成机器人演示，提升HRT的稳定性与安全性。**

- **链接: [http://arxiv.org/pdf/2508.09855v1](http://arxiv.org/pdf/2508.09855v1)**

> **作者:** Yuekun Wu; Yik Lung Pang; Andrea Cavallaro; Changjae Oh
>
> **备注:** 3 pages, 3 figures
>
> **摘要:** Human-robot teaming (HRT) systems often rely on large-scale datasets of human and robot interactions, especially for close-proximity collaboration tasks such as human-robot handovers. Learning robot manipulation policies from raw, real-world image data requires a large number of robot-action trials in the physical environment. Although simulation training offers a cost-effective alternative, the visual domain gap between simulation and robot workspace remains a major limitation. We introduce a method for training HRT policies, focusing on human-to-robot handovers, solely from RGB images without the need for real-robot training or real-robot data collection. The goal is to enable the robot to reliably receive objects from a human with stable grasping while avoiding collisions with the human hand. The proposed policy learner leverages sparse-view Gaussian Splatting reconstruction of human-to-robot handover scenes to generate robot demonstrations containing image-action pairs captured with a camera mounted on the robot gripper. As a result, the simulated camera pose changes in the reconstructed scene can be directly translated into gripper pose changes. Experiments in both Gaussian Splatting reconstructed scene and real-world human-to-robot handover experiments demonstrate that our method serves as a new and effective representation for the human-to-robot handover task, contributing to more seamless and robust HRT.
>
---
#### [new 005] FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出基于强化学习的FLARE框架，解决四旋翼悬吊载荷系统敏捷飞行问题，通过高保真仿真直接学习导航策略，实现高速度和零样本迁移。**

- **链接: [http://arxiv.org/pdf/2508.09797v1](http://arxiv.org/pdf/2508.09797v1)**

> **作者:** Dongcheng Cao; Jin Zhou; Xian Wang; Shuo Li
>
> **摘要:** Agile flight for the quadrotor cable-suspended payload system is a formidable challenge due to its underactuated, highly nonlinear, and hybrid dynamics. Traditional optimization-based methods often struggle with high computational costs and the complexities of cable mode transitions, limiting their real-time applicability and maneuverability exploitation. In this letter, we present FLARE, a reinforcement learning (RL) framework that directly learns agile navigation policy from high-fidelity simulation. Our method is validated across three designed challenging scenarios, notably outperforming a state-of-the-art optimization-based approach by a 3x speedup during gate traversal maneuvers. Furthermore, the learned policies achieve successful zero-shot sim-to-real transfer, demonstrating remarkable agility and safety in real-world experiments, running in real time on an onboard computer.
>
---
#### [new 006] SMART-OC: A Real-time Time-risk Optimal Replanning Algorithm for Dynamic Obstacles and Spatio-temporally Varying Currents
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种实时时间-风险优化路径规划算法SMART-OC，解决动态障碍物和时空变化洋流下的无人船导航难题，通过融合路径风险与时间成本实现最优路径选择。**

- **链接: [http://arxiv.org/pdf/2508.09508v1](http://arxiv.org/pdf/2508.09508v1)**

> **作者:** Reema Raval; Shalabh Gupta
>
> **摘要:** Typical marine environments are highly complex with spatio-temporally varying currents and dynamic obstacles, presenting significant challenges to Unmanned Surface Vehicles (USVs) for safe and efficient navigation. Thus, the USVs need to continuously adapt their paths with real-time information to avoid collisions and follow the path of least resistance to the goal via exploiting ocean currents. In this regard, we introduce a novel algorithm, called Self-Morphing Adaptive Replanning Tree for dynamic Obstacles and Currents (SMART-OC), that facilitates real-time time-risk optimal replanning in dynamic environments. SMART-OC integrates the obstacle risks along a path with the time cost to reach the goal to find the time-risk optimal path. The effectiveness of SMART-OC is validated by simulation experiments, which demonstrate that the USV performs fast replannings to avoid dynamic obstacles and exploit ocean currents to successfully reach the goal.
>
---
#### [new 007] Embodied Tactile Perception of Soft Objects Properties
- **分类: cs.RO**

- **简介: 该论文研究机器人触觉感知，通过多模态传感和交互策略优化，设计模块化e-Skin并提出潜滤模型分析因果机械属性，揭示环境与机械特性交互影响。**

- **链接: [http://arxiv.org/pdf/2508.09836v1](http://arxiv.org/pdf/2508.09836v1)**

> **作者:** Anirvan Dutta; Alexis WM Devillard; Zhihuan Zhang; Xiaoxiao Cheng; Etienne Burdet
>
> **摘要:** To enable robots to develop human-like fine manipulation, it is essential to understand how mechanical compliance, multi-modal sensing, and purposeful interaction jointly shape tactile perception. In this study, we use a dedicated modular e-Skin with tunable mechanical compliance and multi-modal sensing (normal, shear forces and vibrations) to systematically investigate how sensing embodiment and interaction strategies influence robotic perception of objects. Leveraging a curated set of soft wave objects with controlled viscoelastic and surface properties, we explore a rich set of palpation primitives-pressing, precession, sliding that vary indentation depth, frequency, and directionality. In addition, we propose the latent filter, an unsupervised, action-conditioned deep state-space model of the sophisticated interaction dynamics and infer causal mechanical properties into a structured latent space. This provides generalizable and in-depth interpretable representation of how embodiment and interaction determine and influence perception. Our investigation demonstrates that multi-modal sensing outperforms uni-modal sensing. It highlights a nuanced interaction between the environment and mechanical properties of e-Skin, which should be examined alongside the interaction by incorporating temporal dynamics.
>
---
#### [new 008] Decision-Making-Based Path Planning for Autonomous UAVs: A Survey
- **分类: cs.RO**

- **简介: 该论文综述自主无人机路径规划中的决策方法，探讨探索与信息路径规划策略，分析数据建模特点，并指出挑战。**

- **链接: [http://arxiv.org/pdf/2508.09304v1](http://arxiv.org/pdf/2508.09304v1)**

> **作者:** Kelen C. Teixeira Vivaldini; Robert Pěnička; Martin Saska
>
> **摘要:** One of the most critical features for the successful operation of autonomous UAVs is the ability to make decisions based on the information acquired from their surroundings. Each UAV must be able to make decisions during the flight in order to deal with uncertainties in its system and the environment, and to further act upon the information being received. Such decisions influence the future behavior of the UAV, which is expressed as the path plan. Thus, decision-making in path planning is an enabling technique for deploying autonomous UAVs in real-world applications. This survey provides an overview of existing studies that use aspects of decision-making in path planning, presenting the research strands for Exploration Path Planning and Informative Path Planning, and focusing on characteristics of how data have been modeled and understood. Finally, we highlight the existing challenges for relevant topics in this field.
>
---
#### [new 009] Masquerade: Learning from In-the-wild Human Videos using Data-Editing
- **分类: cs.RO**

- **简介: 论文提出Masquerade方法，通过编辑真实世界真人视频生成机器人演示，解决数据稀缺问题，提升机器人控制性能。**

- **链接: [http://arxiv.org/pdf/2508.09976v1](http://arxiv.org/pdf/2508.09976v1)**

> **作者:** Marion Lepert; Jiaying Fang; Jeannette Bohg
>
> **备注:** Project website at https://masquerade-robot.github.io/
>
> **摘要:** Robot manipulation research still suffers from significant data scarcity: even the largest robot datasets are orders of magnitude smaller and less diverse than those that fueled recent breakthroughs in language and vision. We introduce Masquerade, a method that edits in-the-wild egocentric human videos to bridge the visual embodiment gap between humans and robots and then learns a robot policy with these edited videos. Our pipeline turns each human video into robotized demonstrations by (i) estimating 3-D hand poses, (ii) inpainting the human arms, and (iii) overlaying a rendered bimanual robot that tracks the recovered end-effector trajectories. Pre-training a visual encoder to predict future 2-D robot keypoints on 675K frames of these edited clips, and continuing that auxiliary loss while fine-tuning a diffusion policy head on only 50 robot demonstrations per task, yields policies that generalize significantly better than prior work. On three long-horizon, bimanual kitchen tasks evaluated in three unseen scenes each, Masquerade outperforms baselines by 5-6x. Ablations show that both the robot overlay and co-training are indispensable, and performance scales logarithmically with the amount of edited human video. These results demonstrate that explicitly closing the visual embodiment gap unlocks a vast, readily available source of data from human videos that can be used to improve robot policies.
>
---
#### [new 010] How Safe Will I Be Given What I Saw? Calibrated Prediction of Safety Chances for Image-Controlled Autonomy
- **分类: cs.RO**

- **简介: 论文提出面向图像控制自主系统的安全预测框架，通过变分自编码器和循环预测器校准风险估计，解决传统方法在分布偏移下的局限性，理论保证校准有效性，实验验证其在长时任务中的稳健性。**

- **链接: [http://arxiv.org/pdf/2508.09346v1](http://arxiv.org/pdf/2508.09346v1)**

> **作者:** Zhenjiang Mao; Mrinall Eashaan Umasudhan; Ivan Ruchkin
>
> **摘要:** Autonomous robots that rely on deep neural network controllers pose critical challenges for safety prediction, especially under partial observability and distribution shift. Traditional model-based verification techniques are limited in scalability and require access to low-dimensional state models, while model-free methods often lack reliability guarantees. This paper addresses these limitations by introducing a framework for calibrated safety prediction in end-to-end vision-controlled systems, where neither the state-transition model nor the observation model is accessible. Building on the foundation of world models, we leverage variational autoencoders and recurrent predictors to forecast future latent trajectories from raw image sequences and estimate the probability of satisfying safety properties. We distinguish between monolithic and composite prediction pipelines and introduce a calibration mechanism to quantify prediction confidence. In long-horizon predictions from high-dimensional observations, the forecasted inputs to the safety evaluator can deviate significantly from the training distribution due to compounding prediction errors and changing environmental conditions, leading to miscalibrated risk estimates. To address this, we incorporate unsupervised domain adaptation to ensure robustness of safety evaluation under distribution shift in predictions without requiring manual labels. Our formulation provides theoretical calibration guarantees and supports practical evaluation across long prediction horizons. Experimental results on three benchmarks show that our UDA-equipped evaluators maintain high accuracy and substantially lower false positive rates under distribution shift. Similarly, world model-based composite predictors outperform their monolithic counterparts on long-horizon tasks, and our conformal calibration provides reliable statistical bounds.
>
---
#### [new 011] Vision-driven River Following of UAV via Safe Reinforcement Learning using Semantic Dynamics Model
- **分类: cs.RO; cs.AI**

- **简介: 论文提出基于视觉的无人机河流跟随任务，解决GPS不可靠问题，通过子模ularity MDP和安全强化学习框架（MGAE、SDM、CADE）提升性能与安全性。**

- **链接: [http://arxiv.org/pdf/2508.09971v1](http://arxiv.org/pdf/2508.09971v1)**

> **作者:** Zihan Wang; Nina Mahmoudian
>
> **备注:** Submitted to Robotics and Autonomous Systems (RAS) journal
>
> **摘要:** Vision-driven autonomous river following by Unmanned Aerial Vehicles is critical for applications such as rescue, surveillance, and environmental monitoring, particularly in dense riverine environments where GPS signals are unreliable. We formalize river following as a coverage control problem in which the reward function is submodular, yielding diminishing returns as more unique river segments are visited, thereby framing the task as a Submodular Markov Decision Process. First, we introduce Marginal Gain Advantage Estimation, which refines the reward advantage function by using a sliding window baseline computed from historical episodic returns, thus aligning the advantage estimation with the agent's evolving recognition of action value in non-Markovian settings. Second, we develop a Semantic Dynamics Model based on patchified water semantic masks that provides more interpretable and data-efficient short-term prediction of future observations compared to latent vision dynamics models. Third, we present the Constrained Actor Dynamics Estimator architecture, which integrates the actor, the cost estimator, and SDM for cost advantage estimation to form a model-based SafeRL framework capable of solving partially observable Constrained Submodular Markov Decision Processes. Simulation results demonstrate that MGAE achieves faster convergence and superior performance over traditional critic-based methods like Generalized Advantage Estimation. SDM provides more accurate short-term state predictions that enable the cost estimator to better predict potential violations. Overall, CADE effectively integrates safety regulation into model-based RL, with the Lagrangian approach achieving the soft balance of reward and safety during training, while the safety layer enhances performance during inference by hard action overlay.
>
---
#### [new 012] CaRoBio: 3D Cable Routing with a Bio-inspired Gripper Fingernail
- **分类: cs.RO; cs.AI**

- **简介: 论文提出基于仿生指甲的单次抓取3D电缆路由方法，解决传统夹钳过紧过拉问题，通过视觉估计和运动原语规划提升效率。**

- **链接: [http://arxiv.org/pdf/2508.09558v1](http://arxiv.org/pdf/2508.09558v1)**

> **作者:** Jiahui Zuo; Boyang Zhang; Fumin Zhang
>
> **摘要:** The manipulation of deformable linear flexures has a wide range of applications in industry, such as cable routing in automotive manufacturing and textile production. Cable routing, as a complex multi-stage robot manipulation scenario, is a challenging task for robot automation. Common parallel two-finger grippers have the risk of over-squeezing and over-tension when grasping and guiding cables. In this paper, a novel eagle-inspired fingernail is designed and mounted on the gripper fingers, which helps with cable grasping on planar surfaces and in-hand cable guiding operations. Then we present a single-grasp end-to-end 3D cable routing framework utilizing the proposed fingernails, instead of the common pick-and-place strategy. Continuous control is achieved to efficiently manipulate cables through vision-based state estimation of task configurations and offline trajectory planning based on motion primitives. We evaluate the effectiveness of the proposed framework with a variety of cables and channel slots, significantly outperforming the pick-and-place manipulation process under equivalent perceptual conditions. Our reconfigurable task setting and the proposed framework provide a reference for future cable routing manipulations in 3D space.
>
---
#### [new 013] Interpretable Robot Control via Structured Behavior Trees and Large Language Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出结合行为树与大语言模型的可解释机器人控制框架，解决传统HRI接口在动态环境中的局限性，通过自然语言指令解析与感知模块集成，实现高精度执行。**

- **链接: [http://arxiv.org/pdf/2508.09621v1](http://arxiv.org/pdf/2508.09621v1)**

> **作者:** Ingrid Maéva Chekam; Ines Pastor-Martinez; Ali Tourani; Jose Andres Millan-Romera; Laura Ribeiro; Pedro Miguel Bastos Soares; Holger Voos; Jose Luis Sanchez-Lopez
>
> **备注:** 15 pages, 5 figures, 3 tables
>
> **摘要:** As intelligent robots become more integrated into human environments, there is a growing need for intuitive and reliable Human-Robot Interaction (HRI) interfaces that are adaptable and more natural to interact with. Traditional robot control methods often require users to adapt to interfaces or memorize predefined commands, limiting usability in dynamic, unstructured environments. This paper presents a novel framework that bridges natural language understanding and robotic execution by combining Large Language Models (LLMs) with Behavior Trees. This integration enables robots to interpret natural language instructions given by users and translate them into executable actions by activating domain-specific plugins. The system supports scalable and modular integration, with a primary focus on perception-based functionalities, such as person tracking and hand gesture recognition. To evaluate the system, a series of real-world experiments was conducted across diverse environments. Experimental results demonstrate that the proposed approach is practical in real-world scenarios, with an average cognition-to-execution accuracy of approximately 94%, making a significant contribution to HRI systems and robots. The complete source code of the framework is publicly available at https://github.com/snt-arg/robot_suite.
>
---
#### [new 014] CLF-RL: Control Lyapunov Function Guided Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出基于CLF的强化学习方法，解决奖励设计繁琐与目标敏感问题，通过模型规划与CLF奖励提升机器人运动鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.09354v1](http://arxiv.org/pdf/2508.09354v1)**

> **作者:** Kejun Li; Zachary Olkin; Yisong Yue; Aaron D. Ames
>
> **备注:** 8 pages; 8 figures
>
> **摘要:** Reinforcement learning (RL) has shown promise in generating robust locomotion policies for bipedal robots, but often suffers from tedious reward design and sensitivity to poorly shaped objectives. In this work, we propose a structured reward shaping framework that leverages model-based trajectory generation and control Lyapunov functions (CLFs) to guide policy learning. We explore two model-based planners for generating reference trajectories: a reduced-order linear inverted pendulum (LIP) model for velocity-conditioned motion planning, and a precomputed gait library based on hybrid zero dynamics (HZD) using full-order dynamics. These planners define desired end-effector and joint trajectories, which are used to construct CLF-based rewards that penalize tracking error and encourage rapid convergence. This formulation provides meaningful intermediate rewards, and is straightforward to implement once a reference is available. Both the reference trajectories and CLF shaping are used only during training, resulting in a lightweight policy at deployment. We validate our method both in simulation and through extensive real-world experiments on a Unitree G1 robot. CLF-RL demonstrates significantly improved robustness relative to the baseline RL policy and better performance than a classic tracking reward RL formulation.
>
---
#### [new 015] GBC: Generalized Behavior-Cloning Framework for Whole-Body Humanoid Imitation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出GBC框架，解决机器人模仿学习中数据泛化问题，通过自适应数据管道和DAgger-MMPPO算法实现端到端学习，并开源平台支持多形态人形机器人控制。**

- **链接: [http://arxiv.org/pdf/2508.09960v1](http://arxiv.org/pdf/2508.09960v1)**

> **作者:** Yifei Yao; Chengyuan Luo; Jiaheng Du; Wentao He; Jun-Guo Lu
>
> **摘要:** The creation of human-like humanoid robots is hindered by a fundamental fragmentation: data processing and learning algorithms are rarely universal across different robot morphologies. This paper introduces the Generalized Behavior Cloning (GBC) framework, a comprehensive and unified solution designed to solve this end-to-end challenge. GBC establishes a complete pathway from human motion to robot action through three synergistic innovations. First, an adaptive data pipeline leverages a differentiable IK network to automatically retarget any human MoCap data to any humanoid. Building on this foundation, our novel DAgger-MMPPO algorithm with its MMTransformer architecture learns robust, high-fidelity imitation policies. To complete the ecosystem, the entire framework is delivered as an efficient, open-source platform based on Isaac Lab, empowering the community to deploy the full workflow via simple configuration scripts. We validate the power and generality of GBC by training policies on multiple heterogeneous humanoids, demonstrating excellent performance and transfer to novel motions. This work establishes the first practical and unified pathway for creating truly generalized humanoid controllers.
>
---
#### [new 016] A Shank Angle-Based Control System Enables Soft Exoskeleton to Assist Human Non-Steady Locomotion
- **分类: cs.RO; cs.SY; eess.SY; I.2.9**

- **简介: 论文提出基于小腿角度的控制系统，解决外骨骼在非稳态运动中的辅助问题，通过在线生成辅助模式和模型前馈控制，适应个体差异，验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.09876v1](http://arxiv.org/pdf/2508.09876v1)**

> **作者:** Xiaowei Tan; Weizhong Jiang; Bi Zhang; Wanxin Chen; Yiwen Zhao; Ning Li; Lianqing Liu; Xingang Zhao
>
> **备注:** 49 pages, 20 figures, 4 tables
>
> **摘要:** Exoskeletons have been shown to effectively assist humans during steady locomotion. However, their effects on non-steady locomotion, characterized by nonlinear phase progression within a gait cycle, remain insufficiently explored, particularly across diverse activities. This work presents a shank angle-based control system that enables the exoskeleton to maintain real-time coordination with human gait, even under phase perturbations, while dynamically shaping assistance profiles to match the biological ankle moment patterns across walking, running, stair negotiation tasks. The control system consists of an assistance profile online generation method and a model-based feedforward control method. The assistance profile is formulated as a dual-Gaussian model with the shank angle as the independent variable. Leveraging only IMU measurements, the model parameters are updated online each stride to adapt to inter- and intra-individual biomechanical variability. The profile tracking control employs a human-exoskeleton kinematics and stiffness model as a feedforward component, reducing reliance on historical control data due to the lack of clear and consistent periodicity in non-steady locomotion. Three experiments were conducted using a lightweight soft exoskeleton with multiple subjects. The results validated the effectiveness of each individual method, demonstrated the robustness of the control system against gait perturbations across various activities, and revealed positive biomechanical and physiological responses of human users to the exoskeleton's mechanical assistance.
>
---
#### [new 017] ESCoT: An Enhanced Step-based Coordinate Trajectory Planning Method for Multiple Car-like Robots
- **分类: cs.RO**

- **简介: 该论文提出一种多车机器人协同轨迹规划方法ESCoT，通过协作规划与重复配置重规划策略解决多机器人路径冲突问题，提升稀疏与密集场景下的规划效率与成功率。**

- **链接: [http://arxiv.org/pdf/2508.09581v1](http://arxiv.org/pdf/2508.09581v1)**

> **作者:** Junkai Jiang; Yihe Chen; Yibin Yang; Ruochen Li; Shaobing Xu; Jianqiang Wang
>
> **摘要:** Multi-vehicle trajectory planning (MVTP) is one of the key challenges in multi-robot systems (MRSs) and has broad applications across various fields. This paper presents ESCoT, an enhanced step-based coordinate trajectory planning method for multiple car-like robots. ESCoT incorporates two key strategies: collaborative planning for local robot groups and replanning for duplicate configurations. These strategies effectively enhance the performance of step-based MVTP methods. Through extensive experiments, we show that ESCoT 1) in sparse scenarios, significantly improves solution quality compared to baseline step-based method, achieving up to 70% improvement in typical conflict scenarios and 34% in randomly generated scenarios, while maintaining high solving efficiency; and 2) in dense scenarios, outperforms all baseline methods, maintains a success rate of over 50% even in the most challenging configurations. The results demonstrate that ESCoT effectively solves MVTP, further extending the capabilities of step-based methods. Finally, practical robot tests validate the algorithm's applicability in real-world scenarios.
>
---
#### [new 018] Immersive Teleoperation of Beyond-Human-Scale Robotic Manipulators: Challenges and Future Directions
- **分类: cs.RO**

- **简介: 论文研究沉浸式远程操作超越人类规模的机器人机械臂，聚焦其控制、认知与界面挑战，提出提升操作员安全、减少感知运动不匹配及增强存在感的解决方案，探讨新型评估工具与安全模型的发展方向。**

- **链接: [http://arxiv.org/pdf/2508.09700v1](http://arxiv.org/pdf/2508.09700v1)**

> **作者:** Mahdi Hejrati; Jouni Mattila
>
> **备注:** This work has been accepted for presentation at the 2025 IEEE Conference on Telepresence, to be held in Leiden, Netherlands
>
> **摘要:** Teleoperation of beyond-human-scale robotic manipulators (BHSRMs) presents unique challenges that differ fundamentally from conventional human-scale systems. As these platforms gain relevance in industrial domains such as construction, mining, and disaster response, immersive interfaces must be rethought to support scalable, safe, and effective human-robot collaboration. This paper investigates the control, cognitive, and interface-level challenges of immersive teleoperation in BHSRMs, with a focus on ensuring operator safety, minimizing sensorimotor mismatch, and enhancing the sense of embodiment. We analyze design trade-offs in haptic and visual feedback systems, supported by early experimental comparisons of exoskeleton- and joystick-based control setups. Finally, we outline key research directions for developing new evaluation tools, scaling strategies, and human-centered safety models tailored to large-scale robotic telepresence.
>
---
#### [new 019] Whole-Body Bilateral Teleoperation with Multi-Stage Object Parameter Estimation for Wheeled Humanoid Locomanipulation
- **分类: cs.RO**

- **简介: 论文提出轮式人形机器人全身体半双工远程操作框架，结合多阶段物体参数估计解决动态同步与力反馈问题，提升操控精度。**

- **链接: [http://arxiv.org/pdf/2508.09846v1](http://arxiv.org/pdf/2508.09846v1)**

> **作者:** Donghoon Baek; Amartya Purushottam; Jason J. Choi; Joao Ramos
>
> **摘要:** This paper presents an object-aware whole-body bilateral teleoperation framework for wheeled humanoid loco-manipulation. This framework combines whole-body bilateral teleoperation with an online multi-stage object inertial parameter estimation module, which is the core technical contribution of this work. The multi-stage process sequentially integrates a vision-based object size estimator, an initial parameter guess generated by a large vision-language model (VLM), and a decoupled hierarchical sampling strategy. The visual size estimate and VLM prior offer a strong initial guess of the object's inertial parameters, significantly reducing the search space for sampling-based refinement and improving the overall estimation speed. A hierarchical strategy first estimates mass and center of mass, then infers inertia from object size to ensure physically feasible parameters, while a decoupled multi-hypothesis scheme enhances robustness to VLM prior errors. Our estimator operates in parallel with high-fidelity simulation and hardware, enabling real-time online updates. The estimated parameters are then used to update the wheeled humanoid's equilibrium point, allowing the operator to focus more on locomotion and manipulation. This integration improves the haptic force feedback for dynamic synchronization, enabling more dynamic whole-body teleoperation. By compensating for object dynamics using the estimated parameters, the framework also improves manipulation tracking while preserving compliant behavior. We validate the system on a customized wheeled humanoid with a robotic gripper and human-machine interface, demonstrating real-time execution of lifting, delivering, and releasing tasks with a payload weighing approximately one-third of the robot's body weight.
>
---
#### [new 020] HapticGiant: A Novel Very Large Kinesthetic Haptic Interface with Hierarchical Force Control
- **分类: cs.RO; cs.HC; cs.SY; eess.SY**

- **简介: 论文提出HapticGiant，一种大尺度触觉接口，解决传统设备受限于工作空间、自由度及运动学匹配的问题，通过分层优化的适应型力控方案实现自然运动与全反馈，推动高沉浸VR应用。**

- **链接: [http://arxiv.org/pdf/2508.09595v1](http://arxiv.org/pdf/2508.09595v1)**

> **作者:** Michael Fennel; Markus Walker; Dominik Pikos; Uwe D. Hanebeck
>
> **备注:** Final Version - Accepted on IEEE Transactions on Haptics
>
> **摘要:** Research in virtual reality and haptic technologies has consistently aimed to enhance immersion. While advanced head-mounted displays are now commercially available, kinesthetic haptic interfaces still face challenges such as limited workspaces, insufficient degrees of freedom, and kinematics not matching the human arm. In this paper, we present HapticGiant, a novel large-scale kinesthetic haptic interface designed to match the properties of the human arm as closely as possible and to facilitate natural user locomotion while providing full haptic feedback. The interface incorporates a novel admittance-type force control scheme, leveraging hierarchical optimization to render both arbitrary serial kinematic chains and Cartesian admittances. Notably, the proposed control scheme natively accounts for system limitations, including joint and Cartesian constraints, as well as singularities. Experimental results demonstrate the effectiveness of HapticGiant and its control scheme, paving the way for highly immersive virtual reality applications.
>
---
#### [new 021] PPL: Point Cloud Supervised Proprioceptive Locomotion Reinforcement Learning for Legged Robots in Crawl Spaces
- **分类: cs.RO**

- **简介: 论文提出一种基于点云的监督 proprioceptive 强化学习方法，解决腿足机器人在受限空间中导航的挑战，通过状态估计网络和奖励函数提升敏捷性，无需外部传感器。**

- **链接: [http://arxiv.org/pdf/2508.09950v1](http://arxiv.org/pdf/2508.09950v1)**

> **作者:** Bida Ma; Nuo Xu; Chenkun Qi; Xin Liu; Yule Mo; Jinkai Wang; Chunpeng Lu
>
> **摘要:** The legged locomotion in spatially constrained structures (called crawl spaces) is challenging. In crawl spaces, current exteroceptive locomotion learning methods are limited by large noises and errors of the sensors in possible low visibility conditions, and current proprioceptive locomotion learning methods are difficult in traversing crawl spaces because only ground features are inferred. In this study, a point cloud supervised proprioceptive locomotion reinforcement learning method for legged robots in crawl spaces is proposed. A state estimation network is designed to estimate the robot's surrounding ground and spatial features as well as the robot's collision states using historical proprioceptive sensor data. The point cloud is represented in polar coordinate frame and a point cloud processing method is proposed to efficiently extract the ground and spatial features that are used to supervise the state estimation network learning. Comprehensive reward functions that guide the robot to traverse through crawl spaces after collisions are designed. Experiments demonstrate that, compared to existing methods, our method exhibits more agile locomotion in crawl spaces. This study enhances the ability of legged robots to traverse spatially constrained environments without requiring exteroceptive sensors.
>
---
#### [new 022] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos
- **分类: cs.CV; cs.AI; cs.CE; cs.LG; cs.RO**

- **简介: 论文提出TRACE框架，通过将3D点建模为刚体粒子，直接学习其物理动力学，解决无标注视频中复杂场景运动建模问题，实现未来帧外推。**

- **链接: [http://arxiv.org/pdf/2508.09811v1](http://arxiv.org/pdf/2508.09811v1)**

> **作者:** Jinxi Li; Ziyang Song; Bo Yang
>
> **备注:** ICCV 2025. Code and data are available at: https://github.com/vLAR-group/TRACE
>
> **摘要:** In this paper, we aim to model 3D scene geometry, appearance, and physical information just from dynamic multi-view videos in the absence of any human labels. By leveraging physics-informed losses as soft constraints or integrating simple physics models into neural nets, existing works often fail to learn complex motion physics, or doing so requires additional labels such as object types or masks. We propose a new framework named TRACE to model the motion physics of complex dynamic 3D scenes. The key novelty of our method is that, by formulating each 3D point as a rigid particle with size and orientation in space, we directly learn a translation rotation dynamics system for each particle, explicitly estimating a complete set of physical parameters to govern the particle's motion over time. Extensive experiments on three existing dynamic datasets and one newly created challenging synthetic datasets demonstrate the extraordinary performance of our method over baselines in the task of future frame extrapolation. A nice property of our framework is that multiple objects or parts can be easily segmented just by clustering the learned physical parameters.
>
---
#### [new 023] SegDAC: Segmentation-Driven Actor-Critic for Visual Reinforcement Learning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出SegDAC，针对视觉强化学习中高维输入与噪声奖励的挑战，通过SAM分解与YOLO-World语义接地，结合动态Transformer架构实现在线RL学习，提升视觉泛化与样本效率。**

- **链接: [http://arxiv.org/pdf/2508.09325v1](http://arxiv.org/pdf/2508.09325v1)**

> **作者:** Alexandre Brown; Glen Berseth
>
> **摘要:** Visual reinforcement learning (RL) is challenging due to the need to learn both perception and actions from high-dimensional inputs and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains unclear. We propose SegDAC, a Segmentation-Driven Actor-Critic method. SegDAC uses Segment Anything (SAM) for object-centric decomposition and YOLO-World to ground segments semantically via text prompts. It includes a novel transformer-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks.
>
---
#### [new 024] Safety Perspective on Assisted Lane Changes: Insights from Open-Road, Live-Traffic Experiments
- **分类: physics.soc-ph; cs.RO**

- **简介: 论文研究五类ADAS辅助车道变更系统的安全性，通过实地实验分析其运动学特性与安全边际，发现部分系统存在违规设计及交通冲击风险。**

- **链接: [http://arxiv.org/pdf/2508.09233v1](http://arxiv.org/pdf/2508.09233v1)**

> **作者:** Konstantinos Mattas; Sandor Vass; Gergely Zachar; Junyi Ji; Derek Gloudemans; Davide Maggi; Akos Kriston; Mohamed Brahmi; Maria Christina Galassi; Daniel B Work; Biagio Ciuffo
>
> **备注:** 21 pages, 8 Figures
>
> **摘要:** This study investigates the assisted lane change functionality of five different vehicles equipped with advanced driver assistance systems (ADAS). The goal is to examine novel, under-researched features of commercially available ADAS technologies. The experimental campaign, conducted in the I-24 highway near Nashville, TN, US, collected data on the kinematics and safety margins of assisted lane changes in real-world conditions. The results show that the kinematics of assisted lane changes are consistent for each system, with four out of five vehicles using slower speeds and decelerations than human drivers. However, one system consistently performed more assertive lane changes, completing the maneuver in around 5 seconds. Regarding safety margins, only three vehicles are investigated. Those operated in the US are not restricted by relevant UN regulations, and their designs were found not to adhere to these regulatory requirements. A simulation method used to classify the challenge level for the vehicle receiving the lane change, showing that these systems can force trailing vehicles to decelerate to keep a safe gap. One assisted system was found to have performed a maneuver that posed a hard challenge level for the other vehicle, raising concerns about the safety of these systems in real-world operation. All three vehicles were found to carry out lane changes that induced decelerations to the vehicle in the target lane. Those decelerations could affect traffic flow, inducing traffic shockwaves.
>
---
#### [new 025] Plane Detection and Ranking via Model Information Optimization
- **分类: cs.CV; cs.RO**

- **简介: 本文提出基于模型信息优化的平面检测与排名框架，解决RANSAC假阳性问题，通过生成多模型并计算信息，选择最优，同时利用神经网络加速，提升平面参数准确性。**

- **链接: [http://arxiv.org/pdf/2508.09625v1](http://arxiv.org/pdf/2508.09625v1)**

> **作者:** Daoxin Zhong; Jun Li; Meng Yee Michael Chuah
>
> **备注:** Accepted as contributed paper in the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Plane detection from depth images is a crucial subtask with broad robotic applications, often accomplished by iterative methods such as Random Sample Consensus (RANSAC). While RANSAC is a robust strategy with strong probabilistic guarantees, the ambiguity of its inlier threshold criterion makes it susceptible to false positive plane detections. This issue is particularly prevalent in complex real-world scenes, where the true number of planes is unknown and multiple planes coexist. In this paper, we aim to address this limitation by proposing a generalised framework for plane detection based on model information optimization. Building on previous works, we treat the observed depth readings as discrete random variables, with their probability distributions constrained by the ground truth planes. Various models containing different candidate plane constraints are then generated through repeated random sub-sampling to explain our observations. By incorporating the physics and noise model of the depth sensor, we can calculate the information for each model, and the model with the least information is accepted as the most likely ground truth. This information optimization process serves as an objective mechanism for determining the true number of planes and preventing false positive detections. Additionally, the quality of each detected plane can be ranked by summing the information reduction of inlier points for each plane. We validate these properties through experiments with synthetic data and find that our algorithm estimates plane parameters more accurately compared to the default Open3D RANSAC plane segmentation. Furthermore, we accelerate our algorithm by partitioning the depth map using neural network segmentation, which enhances its ability to generate more realistic plane parameters in real-world data.
>
---
#### [new 026] WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization
- **分类: cs.CV; cs.RO; I.4.10**

- **简介: 论文提出WeatherPrompt框架，通过多模态融合解决无人机视觉地理定位在恶劣天气下的泛化问题，采用动态门控机制与对比学习优化场景与天气特征分离，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2508.09560v1](http://arxiv.org/pdf/2508.09560v1)**

> **作者:** Jiahao Wen; Hang Yu; Zhedong Zheng
>
> **备注:** 13 pages, 4figures
>
> **摘要:** Visual geo-localization for drones faces critical degradation under weather perturbations, \eg, rain and fog, where existing methods struggle with two inherent limitations: 1) Heavy reliance on limited weather categories that constrain generalization, and 2) Suboptimal disentanglement of entangled scene-weather features through pseudo weather categories. We present WeatherPrompt, a multi-modality learning paradigm that establishes weather-invariant representations through fusing the image embedding with the text context. Our framework introduces two key contributions: First, a Training-free Weather Reasoning mechanism that employs off-the-shelf large multi-modality models to synthesize multi-weather textual descriptions through human-like reasoning. It improves the scalability to unseen or complex weather, and could reflect different weather strength. Second, to better disentangle the scene and weather feature, we propose a multi-modality framework with the dynamic gating mechanism driven by the text embedding to adaptively reweight and fuse visual features across modalities. The framework is further optimized by the cross-modal objectives, including image-text contrastive learning and image-text matching, which maps the same scene with different weather conditions closer in the respresentation space. Extensive experiments validate that, under diverse weather conditions, our method achieves competitive recall rates compared to state-of-the-art drone geo-localization methods. Notably, it improves Recall@1 by +13.37\% under night conditions and by 18.69\% under fog and snow conditions.
>
---
#### [new 027] Online Safety under Multiple Constraints and Input Bounds using gatekeeper: Theory and Applications
- **分类: eess.SY; cs.MA; cs.RO; cs.SY**

- **简介: 论文提出gatekeeper框架，针对多约束与输入边界下的在线安全问题，通过递归保证无限时序轨迹满足系统动态与约束，结合备份控制器实现高效计算，应用于多智能体编队飞行。**

- **链接: [http://arxiv.org/pdf/2508.09963v1](http://arxiv.org/pdf/2508.09963v1)**

> **作者:** Devansh R. Agrawal; Dimitra Panagou
>
> **备注:** 6 pages, 2 figures. Accepted for publication in IEEE L-CSS 2025
>
> **摘要:** This letter presents an approach to guarantee online safety of a cyber-physical system under multiple state and input constraints. Our proposed framework, called gatekeeper, recursively guarantees the existence of an infinite-horizon trajectory that satisfies all constraints and system dynamics. Such trajectory is constructed using a backup controller, which we define formally in this paper. gatekeeper relies on a small number of verifiable assumptions, and is computationally efficient since it requires optimization over a single scalar variable. We make two primary contributions in this letter. (A) First, we develop the theory of gatekeeper: we derive a sub-optimality bound relative to a full nonlinear trajectory optimization problem, and show how this can be used in runtime to validate performance. This also informs the design of the backup controllers and sets. (B) Second, we demonstrate in detail an application of gatekeeper for multi-agent formation flight, where each Dubins agent must avoid multiple obstacles and weapons engagement zones, both of which are nonlinear, nonconvex constraints.
>
---
#### [new 028] Predictive Uncertainty for Runtime Assurance of a Real-Time Computer Vision-Based Landing System
- **分类: cs.CV; cs.RO**

- **简介: 论文提出一种实时计算机视觉降落系统中的预测不确定性方法，通过高效神经网络、校准损失函数和Residual-based RAIM实现安全关键应用的运行时保障，提升姿态估计精度与故障检测能力。**

- **链接: [http://arxiv.org/pdf/2508.09732v1](http://arxiv.org/pdf/2508.09732v1)**

> **作者:** Romeo Valentin; Sydney M. Katz; Artur B. Carneiro; Don Walker; Mykel J. Kochenderfer
>
> **备注:** 8 pages, 5 figures, accepted at DASC 2025
>
> **摘要:** Recent advances in data-driven computer vision have enabled robust autonomous navigation capabilities for civil aviation, including automated landing and runway detection. However, ensuring that these systems meet the robustness and safety requirements for aviation applications remains a major challenge. In this work, we present a practical vision-based pipeline for aircraft pose estimation from runway images that represents a step toward the ability to certify these systems for use in safety-critical aviation applications. Our approach features three key innovations: (i) an efficient, flexible neural architecture based on a spatial Soft Argmax operator for probabilistic keypoint regression, supporting diverse vision backbones with real-time inference; (ii) a principled loss function producing calibrated predictive uncertainties, which are evaluated via sharpness and calibration metrics; and (iii) an adaptation of Residual-based Receiver Autonomous Integrity Monitoring (RAIM), enabling runtime detection and rejection of faulty model outputs. We implement and evaluate our pose estimation pipeline on a dataset of runway images. We show that our model outperforms baseline architectures in terms of accuracy while also producing well-calibrated uncertainty estimates with sub-pixel precision that can be used downstream for fault detection.
>
---
#### [new 029] RayletDF: Raylet Distance Fields for Generalizable 3D Surface Reconstruction from Point Clouds or Gaussians
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 论文提出RayletDF方法，通过射线距离场直接预测表面点，解决传统坐标方法计算成本高、泛化性差的问题，实现3D表面重建。**

- **链接: [http://arxiv.org/pdf/2508.09830v1](http://arxiv.org/pdf/2508.09830v1)**

> **作者:** Shenxing Wei; Jinxi Li; Yafei Yang; Siyuan Zhou; Bo Yang
>
> **备注:** ICCV 2025 Highlight. Shenxing and Jinxi are co-first authors. Code and data are available at: https://github.com/vLAR-group/RayletDF
>
> **摘要:** In this paper, we present a generalizable method for 3D surface reconstruction from raw point clouds or pre-estimated 3D Gaussians by 3DGS from RGB images. Unlike existing coordinate-based methods which are often computationally intensive when rendering explicit surfaces, our proposed method, named RayletDF, introduces a new technique called raylet distance field, which aims to directly predict surface points from query rays. Our pipeline consists of three key modules: a raylet feature extractor, a raylet distance field predictor, and a multi-raylet blender. These components work together to extract fine-grained local geometric features, predict raylet distances, and aggregate multiple predictions to reconstruct precise surface points. We extensively evaluate our method on multiple public real-world datasets, demonstrating superior performance in surface reconstruction from point clouds or 3D Gaussians. Most notably, our method achieves exceptional generalization ability, successfully recovering 3D surfaces in a single-forward pass across unseen datasets in testing.
>
---
#### [new 030] Collision-Free Bearing-Driven Formation Tracking for Euler-Lagrange Systems
- **分类: eess.SY; cs.RO; cs.SY; math-ph; math.MP; math.OC; nlin.PS**

- **简介: 本文研究异构欧拉-拉格朗日系统基于轴承驱动的无碰撞形成跟踪问题，通过分布式观测器与自适应机制实现控制，确保轨迹避撞。**

- **链接: [http://arxiv.org/pdf/2508.09908v1](http://arxiv.org/pdf/2508.09908v1)**

> **作者:** Haoshu Cheng; Martin Guay; Shimin Wang; Yunhong Che
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** In this paper, we investigate the problem of tracking formations driven by bearings for heterogeneous Euler-Lagrange systems with parametric uncertainty in the presence of multiple moving leaders. To estimate the leaders' velocities and accelerations, we first design a distributed observer for the leader system, utilizing a bearing-based localization condition in place of the conventional connectivity assumption. This observer, coupled with an adaptive mechanism, enables the synthesis of a novel distributed control law that guides the formation towards the target formation, without requiring prior knowledge of the system parameters. Furthermore, we establish a sufficient condition, dependent on the initial formation configuration, that ensures collision avoidance throughout the formation evolution. The effectiveness of the proposed approach is demonstrated through a numerical example.
>
---
#### [new 031] Surg-InvNeRF: Invertible NeRF for 3D tracking and reconstruction in surgical vision
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出基于可逆NeRF的Surg-InvNeRF方法，用于手术视觉中的长期3D点跟踪与重建，解决传统方法受限于2D运动和缺乏一致性的问题，通过逆向渲染和多尺度结构提升精度与效率。**

- **链接: [http://arxiv.org/pdf/2508.09681v1](http://arxiv.org/pdf/2508.09681v1)**

> **作者:** Gerardo Loza; Junlei Hu; Dominic Jones; Sharib Ali; Pietro Valdastri
>
> **备注:** 10 pages
>
> **摘要:** We proposed a novel test-time optimisation (TTO) approach framed by a NeRF-based architecture for long-term 3D point tracking. Most current methods in point tracking struggle to obtain consistent motion or are limited to 2D motion. TTO approaches frame the solution for long-term tracking as optimising a function that aggregates correspondences from other specialised state-of-the-art methods. Unlike the state-of-the-art on TTO, we propose parametrising such a function with our new invertible Neural Radiance Field (InvNeRF) architecture to perform both 2D and 3D tracking in surgical scenarios. Our approach allows us to exploit the advantages of a rendering-based approach by supervising the reprojection of pixel correspondences. It adapts strategies from recent rendering-based methods to obtain a bidirectional deformable-canonical mapping, to efficiently handle a defined workspace, and to guide the rays' density. It also presents our multi-scale HexPlanes for fast inference and a new algorithm for efficient pixel sampling and convergence criteria. We present results in the STIR and SCARE datasets, for evaluating point tracking and testing the integration of kinematic data in our pipeline, respectively. In 2D point tracking, our approach surpasses the precision and accuracy of the TTO state-of-the-art methods by nearly 50% on average precision, while competing with other approaches. In 3D point tracking, this is the first TTO approach, surpassing feed-forward methods while incorporating the benefits of a deformable NeRF-based reconstruction.
>
---
#### [new 032] Distilling LLM Prior to Flow Model for Generalizable Agent's Imagination in Object Goal Navigation
- **分类: cs.CV; cs.RO**

- **简介: 论文针对物体目标导航任务，提出GOAL框架，通过蒸馏LLM先验知识至生成流模型，解决传统方法受限于室内布局不确定性的泛化问题，实现更强的环境适应性。**

- **链接: [http://arxiv.org/pdf/2508.09423v1](http://arxiv.org/pdf/2508.09423v1)**

> **作者:** Badi Li; Ren-jie Lu; Yu Zhou; Jingke Meng; Wei-shi Zheng
>
> **摘要:** The Object Goal Navigation (ObjectNav) task challenges agents to locate a specified object in an unseen environment by imagining unobserved regions of the scene. Prior approaches rely on deterministic and discriminative models to complete semantic maps, overlooking the inherent uncertainty in indoor layouts and limiting their ability to generalize to unseen environments. In this work, we propose GOAL, a generative flow-based framework that models the semantic distribution of indoor environments by bridging observed regions with LLM-enriched full-scene semantic maps. During training, spatial priors inferred from large language models (LLMs) are encoded as two-dimensional Gaussian fields and injected into target maps, distilling rich contextual knowledge into the flow model and enabling more generalizable completions. Extensive experiments demonstrate that GOAL achieves state-of-the-art performance on MP3D and Gibson, and shows strong generalization in transfer settings to HM3D. Codes and pretrained models are available at https://github.com/Badi-Li/GOAL.
>
---
## 更新

#### [replaced 001] Learning Whole-Body Loco-Manipulation for Omni-Directional Task Space Pose Tracking with a Wheeled-Quadrupedal-Manipulator
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.03012v2](http://arxiv.org/pdf/2412.03012v2)**

> **作者:** Kaiwen Jiang; Zhen Fu; Junde Guo; Wei Zhang; Hua Chen
>
> **摘要:** In this paper, we study the whole-body loco-manipulation problem using reinforcement learning (RL). Specifically, we focus on the problem of how to coordinate the floating base and the robotic arm of a wheeled-quadrupedal manipulator robot to achieve direct six-dimensional (6D) end-effector (EE) pose tracking in task space. Different from conventional whole-body loco-manipulation problems that track both floating-base and end-effector commands, the direct EE pose tracking problem requires inherent balance among redundant degrees of freedom in the whole-body motion. We leverage RL to solve this challenging problem. To address the associated difficulties, we develop a novel reward fusion module (RFM) that systematically integrates reward terms corresponding to different tasks in a nonlinear manner. In such a way, the inherent multi-stage and hierarchical feature of the loco-manipulation problem can be carefully accommodated. By combining the proposed RFM with the a teacher-student RL training paradigm, we present a complete RL scheme to achieve 6D EE pose tracking for the wheeled-quadruped manipulator robot. Extensive simulation and hardware experiments demonstrate the significance of the RFM. In particular, we enable smooth and precise tracking performance, achieving state-of-the-art tracking position error of less than 5 cm, and rotation error of less than 0.1 rad. Please refer to https://clearlab-sustech.github.io/RFM_loco_mani/ for more experimental videos.
>
---
#### [replaced 002] OC-SOP: Enhancing Vision-Based 3D Semantic Occupancy Prediction by Object-Centric Awareness
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18798v2](http://arxiv.org/pdf/2506.18798v2)**

> **作者:** Helin Cao; Sven Behnke
>
> **备注:** 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Vienna, Austria, Oct 2025
>
> **摘要:** Autonomous driving perception faces significant challenges due to occlusions and incomplete scene data in the environment. To overcome these issues, the task of semantic occupancy prediction (SOP) is proposed, which aims to jointly infer both the geometry and semantic labels of a scene from images. However, conventional camera-based methods typically treat all categories equally and primarily rely on local features, leading to suboptimal predictions, especially for dynamic foreground objects. To address this, we propose Object-Centric SOP (OC-SOP), a framework that integrates high-level object-centric cues extracted via a detection branch into the semantic occupancy prediction pipeline. This object-centric integration significantly enhances the prediction accuracy for foreground objects and achieves state-of-the-art performance among all categories on SemanticKITTI.
>
---
#### [replaced 003] BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.08241v3](http://arxiv.org/pdf/2508.08241v3)**

> **作者:** Qiayuan Liao; Takara E. Truong; Xiaoyu Huang; Guy Tevet; Koushil Sreenath; C. Karen Liu
>
> **备注:** coin toss authorship, minor changes
>
> **摘要:** Learning skills from human motions offers a promising path toward generalizable policies for versatile humanoid whole-body control, yet two key cornerstones are missing: (1) a high-quality motion tracking framework that faithfully transforms large-scale kinematic references into robust and extremely dynamic motions on real hardware, and (2) a distillation approach that can effectively learn these motion primitives and compose them to solve downstream tasks. We address these gaps with BeyondMimic, a real-world framework to learn from human motions for versatile and naturalistic humanoid control via guided diffusion. Our framework provides a motion tracking pipeline capable of challenging skills such as jumping spins, sprinting, and cartwheels with state-of-the-art motion quality. Moving beyond simply mimicking existing motions, we further introduce a unified diffusion policy that enables zero-shot task-specific control at test time using simple cost functions. Deployed on hardware, BeyondMimic performs diverse tasks at test time, including waypoint navigation, joystick teleoperation, and obstacle avoidance, bridging sim-to-real motion tracking and flexible synthesis of human motion primitives for whole-body control. https://beyondmimic.github.io/.
>
---
#### [replaced 004] LM-MCVT: A Lightweight Multi-modal Multi-view Convolutional-Vision Transformer Approach for 3D Object Recognition
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19256v3](http://arxiv.org/pdf/2504.19256v3)**

> **作者:** Songsong Xiong; Hamidreza Kasaei
>
> **摘要:** In human-centered environments such as restaurants, homes, and warehouses, robots often face challenges in accurately recognizing 3D objects. These challenges stem from the complexity and variability of these environments, including diverse object shapes. In this paper, we propose a novel Lightweight Multi-modal Multi-view Convolutional-Vision Transformer network (LM-MCVT) to enhance 3D object recognition in robotic applications. Our approach leverages the Globally Entropy-based Embeddings Fusion (GEEF) method to integrate multi-views efficiently. The LM-MCVT architecture incorporates pre- and mid-level convolutional encoders and local and global transformers to enhance feature extraction and recognition accuracy. We evaluate our method on the synthetic ModelNet40 dataset and achieve a recognition accuracy of 95.6% using a four-view setup, surpassing existing state-of-the-art methods. To further validate its effectiveness, we conduct 5-fold cross-validation on the real-world OmniObject3D dataset using the same configuration. Results consistently show superior performance, demonstrating the method's robustness in 3D object recognition across synthetic and real-world 3D data.
>
---
#### [replaced 005] MetaFold: Language-Guided Multi-Category Garment Folding Framework via Trajectory Generation and Foundation Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08372v2](http://arxiv.org/pdf/2503.08372v2)**

> **作者:** Haonan Chen; Junxiao Li; Ruihai Wu; Yiwei Liu; Yiwen Hou; Zhixuan Xu; Jingxiang Guo; Chongkai Gao; Zhenyu Wei; Shensi Xu; Jiaqi Huang; Lin Shao
>
> **摘要:** Garment folding is a common yet challenging task in robotic manipulation. The deformability of garments leads to a vast state space and complex dynamics, which complicates precise and fine-grained manipulation. Previous approaches often rely on predefined key points or demonstrations, limiting their generalization across diverse garment categories. This paper presents a framework, MetaFold, that disentangles task planning from action prediction, learning each independently to enhance model generalization. It employs language-guided point cloud trajectory generation for task planning and a low-level foundation model for action prediction. This structure facilitates multi-category learning, enabling the model to adapt flexibly to various user instructions and folding tasks. Experimental results demonstrate the superiority of our proposed framework. Supplementary materials are available on our website: https://meta-fold.github.io/.
>
---
#### [replaced 006] Chemist Eye: A Visual Language Model-Powered System for Safety Monitoring and Robot Decision-Making in Self-Driving Laboratories
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.05148v2](http://arxiv.org/pdf/2508.05148v2)**

> **作者:** Francisco Munguia-Galeano; Zhengxue Zhou; Satheeshkumar Veeramani; Hatem Fakhruldeen; Louis Longley; Rob Clowes; Andrew I. Cooper
>
> **摘要:** The integration of robotics and automation into self-driving laboratories (SDLs) can introduce additional safety complexities, in addition to those that already apply to conventional research laboratories. Personal protective equipment (PPE) is an essential requirement for ensuring the safety and well-being of workers in laboratories, self-driving or otherwise. Fires are another important risk factor in chemical laboratories. In SDLs, fires that occur close to mobile robots, which use flammable lithium batteries, could have increased severity. Here, we present Chemist Eye, a distributed safety monitoring system designed to enhance situational awareness in SDLs. The system integrates multiple stations equipped with RGB, depth, and infrared cameras, designed to monitor incidents in SDLs. Chemist Eye is also designed to spot workers who have suffered a potential accident or medical emergency, PPE compliance and fire hazards. To do this, Chemist Eye uses decision-making driven by a vision-language model (VLM). Chemist Eye is designed for seamless integration, enabling real-time communication with robots. Based on the VLM recommendations, the system attempts to drive mobile robots away from potential fire locations, exits, or individuals not wearing PPE, and issues audible warnings where necessary. It also integrates with third-party messaging platforms to provide instant notifications to lab personnel. We tested Chemist Eye with real-world data from an SDL equipped with three mobile robots and found that the spotting of possible safety hazards and decision-making performances reached 97 % and 95 %, respectively.
>
---
#### [replaced 007] TinyMPC: Model-Predictive Control on Resource-Constrained Microcontrollers
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2310.16985v4](http://arxiv.org/pdf/2310.16985v4)**

> **作者:** Anoushka Alavilli; Khai Nguyen; Sam Schoedel; Brian Plancher; Zachary Manchester
>
> **备注:** Accepted at ICRA 2024. First three authors contributed equally and are listed in alphabetical order. Publicly available at https://tinympc.org
>
> **摘要:** Model-predictive control (MPC) is a powerful tool for controlling highly dynamic robotic systems subject to complex constraints. However, MPC is computationally demanding, and is often impractical to implement on small, resource-constrained robotic platforms. We present TinyMPC, a high-speed MPC solver with a low memory footprint targeting the microcontrollers common on small robots. Our approach is based on the alternating direction method of multipliers (ADMM) and leverages the structure of the MPC problem for efficiency. We demonstrate TinyMPC's effectiveness by benchmarking against the state-of-the-art solver OSQP, achieving nearly an order of magnitude speed increase, as well as through hardware experiments on a 27 gram quadrotor, demonstrating high-speed trajectory tracking and dynamic obstacle avoidance. TinyMPC is publicly available at https://tinympc.org.
>
---
#### [replaced 008] A multi-strategy improved snake optimizer for three-dimensional UAV path planning and engineering problems
- **分类: cs.RO; cs.AI; cs.CE**

- **链接: [http://arxiv.org/pdf/2507.14043v2](http://arxiv.org/pdf/2507.14043v2)**

> **作者:** Genliang Li; Yaxin Cui; Jinyu Su
>
> **备注:** 59 pages, 22 figures
>
> **摘要:** Metaheuristic algorithms have gained widespread application across various fields owing to their ability to generate diverse solutions. One such algorithm is the Snake Optimizer (SO), a progressive optimization approach. However, SO suffers from the issues of slow convergence speed and susceptibility to local optima. In light of these shortcomings, we propose a novel Multi-strategy Improved Snake Optimizer (MISO). Firstly, we propose a new adaptive random disturbance strategy based on sine function to alleviate the risk of getting trapped in a local optimum. Secondly, we introduce adaptive Levy flight strategy based on scale factor and leader and endow the male snake leader with flight capability, which makes it easier for the algorithm to leap out of the local optimum and find the global optimum. More importantly, we put forward a position update strategy combining elite leadership and Brownian motion, effectively accelerating the convergence speed while ensuring precision. Finally, to demonstrate the performance of MISO, we utilize 30 CEC2017 test functions and the CEC2022 test suite, comparing it with 11 popular algorithms across different dimensions to validate its effectiveness. Moreover, Unmanned Aerial Vehicle (UAV) has been widely used in various fields due to its advantages of low cost, high mobility and easy operation. However, the UAV path planning problem is crucial for flight safety and efficiency, and there are still challenges in establishing and optimizing the path model. Therefore, we apply MISO to the UAV 3D path planning problem as well as 6 engineering design problems to assess its feasibility in practical applications. The experimental results demonstrate that MISO exceeds other competitive algorithms in terms of solution quality and stability, establishing its strong potential for application.
>
---
#### [replaced 009] Deep Learning Warm Starts for Trajectory Optimization on the International Space Station
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05588v2](http://arxiv.org/pdf/2505.05588v2)**

> **作者:** Somrita Banerjee; Abhishek Cauligi; Marco Pavone
>
> **备注:** Submitted to 2025 International Conference on Space Robotics (iSpaRo). Presented at RSS 2025 Workshop on Space Robotics
>
> **摘要:** Trajectory optimization is a cornerstone of modern robot autonomy, enabling systems to compute trajectories and controls in real-time while respecting safety and physical constraints. However, it has seen limited usage in spaceflight applications due to its heavy computational demands that exceed the capability of most flight computers. In this work, we provide results on the first flight demonstration of using machine learning-based warm starts for accelerating trajectory optimization for the Astrobee free-flying robot on-board the International Space Station (ISS). We formulate a data-driven optimal control approach that trains a neural network to learn the structure of the trajectory generation problem being solved for by sequential convex programming (SCP). On-board, this trained neural network predicts solutions for the trajectory generation problem and relies on using the SCP solver to enforce safety constraints for the system. Our trained network reduces the number of solver iterations required for convergence in cases including rotational dynamics by 60% and in cases with obstacles drawn from the training distribution of the warm start model by 50%. This work represents a significant milestone in the use of learning-based control for spaceflight applications and a stepping stone for future advances in the use of machine learning for autonomous guidance, navigation, & control.
>
---
#### [replaced 010] On learning racing policies with reinforcement learning
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.02420v2](http://arxiv.org/pdf/2504.02420v2)**

> **作者:** Grzegorz Czechmanowski; Jan Węgrzynowski; Piotr Kicki; Krzysztof Walas
>
> **备注:** This paper has been accepted for publication in the Proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Fully autonomous vehicles promise enhanced safety and efficiency. However, ensuring reliable operation in challenging corner cases requires control algorithms capable of performing at the vehicle limits. We address this requirement by considering the task of autonomous racing and propose solving it by learning a racing policy using Reinforcement Learning (RL). Our approach leverages domain randomization, actuator dynamics modeling, and policy architecture design to enable reliable and safe zero-shot deployment on a real platform. Evaluated on the F1TENTH race car, our RL policy not only surpasses a state-of-the-art Model Predictive Control (MPC), but, to the best of our knowledge, also represents the first instance of an RL policy outperforming expert human drivers in RC racing. This work identifies the key factors driving this performance improvement, providing critical insights for the design of robust RL-based control strategies for autonomous vehicles.
>
---
#### [replaced 011] MolmoAct: Action Reasoning Models that can Reason in Space
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07917v2](http://arxiv.org/pdf/2508.07917v2)**

> **作者:** Jason Lee; Jiafei Duan; Haoquan Fang; Yuquan Deng; Shuo Liu; Boyang Li; Bohan Fang; Jieyu Zhang; Yi Ru Wang; Sangho Lee; Winson Han; Wilbert Pumacay; Angelica Wu; Rose Hendrix; Karen Farley; Eli VanderBilt; Ali Farhadi; Dieter Fox; Ranjay Krishna
>
> **备注:** Appendix include. Code, Data and Weights: https://allenai.org/blog/molmoact
>
> **摘要:** Reasoning is central to purposeful action, yet most robotic foundation models map perception and instructions directly to control, which limits adaptability, generalization, and semantic grounding. We introduce Action Reasoning Models (ARMs), a class of robotic foundation models that integrate perception, planning, and control through a structured three-stage pipeline. Our model, MolmoAct, encodes observations and instructions into depth-aware perception tokens, generates mid-level spatial plans as editable trajectory traces, and predicts precise low-level actions, enabling explainable and steerable behavior. MolmoAct-7B-D achieves strong performance across simulation and real-world settings: 70.5% zero-shot accuracy on SimplerEnv Visual Matching tasks, surpassing closed-source Pi-0 and GR00T N1; 86.6% average success on LIBERO, including an additional 6.3% gain over ThinkAct on long-horizon tasks; and in real-world fine-tuning, an additional 10% (single-arm) and an additional 22.7% (bimanual) task progression over Pi-0-FAST. It also outperforms baselines by an additional 23.3% on out-of-distribution generalization and achieves top human-preference scores for open-ended instruction following and trajectory steering. Furthermore, we release, for the first time, the MolmoAct Dataset -- a mid-training robot dataset comprising over 10,000 high quality robot trajectories across diverse scenarios and tasks. Training with this dataset yields an average 5.5% improvement in general performance over the base model. We release all model weights, training code, our collected dataset, and our action reasoning dataset, establishing MolmoAct as both a state-of-the-art robotics foundation model and an open blueprint for building ARMs that transform perception into purposeful action through structured reasoning. Blogpost: https://allenai.org/blog/molmoact
>
---
#### [replaced 012] RIZE: Regularized Imitation Learning via Distributional Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.20089v2](http://arxiv.org/pdf/2502.20089v2)**

> **作者:** Adib Karimi; Mohammad Mehdi Ebadzadeh
>
> **备注:** Major revision - completely rewritten mathematical formulation and proofs, with substantial updates to methodology and expanded appendix for supporting derivations
>
> **摘要:** We propose a novel Inverse Reinforcement Learning (IRL) method that mitigates the rigidity of fixed reward structures and the limited flexibility of implicit reward regularization. Building on the Maximum Entropy IRL framework, our approach incorporates a squared temporal-difference (TD) regularizer with adaptive targets that evolve dynamically during training, thereby imposing adaptive bounds on recovered rewards and promoting robust decision-making. To capture richer return information, we integrate distributional RL into the learning process. Empirically, our method achieves expert-level performance on complex MuJoCo tasks, surpassing baseline methods on the Humanoid task with 3 demonstrations. Extensive experiments and ablation studies further validate the effectiveness of the approach and provide insights into reward dynamics in imitation learning.
>
---
#### [replaced 013] Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07701v2](http://arxiv.org/pdf/2508.07701v2)**

> **作者:** Bo Jia; Yanan Guo; Ying Chang; Benkui Zhang; Ying Xie; Kangning Du; Lin Cao
>
> **备注:** This paper has been accepted by IROS 2025. Code: https://github.com/Bistu3DV/MND-GS/
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves remarkable results in the field of surface reconstruction. However, when Gaussian normal vectors are aligned within the single-view projection plane, while the geometry appears reasonable in the current view, biases may emerge upon switching to nearby views. To address the distance and global matching challenges in multi-view scenes, we design multi-view normal and distance-guided Gaussian splatting. This method achieves geometric depth unification and high-accuracy reconstruction by constraining nearby depth maps and aligning 3D normals. Specifically, for the reconstruction of small indoor and outdoor scenes, we propose a multi-view distance reprojection regularization module that achieves multi-view Gaussian alignment by computing the distance loss between two nearby views and the same Gaussian surface. Additionally, we develop a multi-view normal enhancement module, which ensures consistency across views by matching the normals of pixel points in nearby views and calculating the loss. Extensive experimental results demonstrate that our method outperforms the baseline in both quantitative and qualitative evaluations, significantly enhancing the surface reconstruction capability of 3DGS. Our code will be made publicly available at (https://github.com/Bistu3DV/MND-GS/).
>
---
#### [replaced 014] Accelerated Reeds-Shepp and Under-Specified Reeds-Shepp Algorithms for Mobile Robot Path Planning
- **分类: cs.RO; cs.CG**

- **链接: [http://arxiv.org/pdf/2504.05921v2](http://arxiv.org/pdf/2504.05921v2)**

> **作者:** Ibrahim Ibrahim; Wilm Decré; Jan Swevers
>
> **备注:** 19 pages, 27 figures
>
> **摘要:** In this study, we present a simple and intuitive method for accelerating optimal Reeds-Shepp path computation. Our approach uses geometrical reasoning to analyze the behavior of optimal paths, resulting in a new partitioning of the state space and a further reduction in the minimal set of viable paths. We revisit and reimplement classic methodologies from the literature, which lack contemporary open-source implementations, to serve as benchmarks for evaluating our method. Additionally, we address the under-specified Reeds-Shepp planning problem where the final orientation is unspecified. We perform exhaustive experiments to validate our solutions. Compared to the modern C++ implementation of the original Reeds-Shepp solution in the Open Motion Planning Library, our method demonstrates a 15x speedup, while classic methods achieve a 5.79x speedup. Both approaches exhibit machine-precision differences in path lengths compared to the original solution. We release our proposed C++ implementations for both the accelerated and under-specified Reeds-Shepp problems as open-source code.
>
---
#### [replaced 015] DualMap: Online Open-Vocabulary Semantic Mapping for Natural Language Navigation in Dynamic Changing Scenes
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.01950v3](http://arxiv.org/pdf/2506.01950v3)**

> **作者:** Jiajun Jiang; Yiming Zhu; Zirui Wu; Jie Song
>
> **备注:** 14 pages, 14 figures. Code: https://github.com/Eku127/DualMap Project page: https://eku127.github.io/DualMap/
>
> **摘要:** We introduce DualMap, an online open-vocabulary mapping system that enables robots to understand and navigate dynamically changing environments through natural language queries. Designed for efficient semantic mapping and adaptability to changing environments, DualMap meets the essential requirements for real-world robot navigation applications. Our proposed hybrid segmentation frontend and object-level status check eliminate the costly 3D object merging required by prior methods, enabling efficient online scene mapping. The dual-map representation combines a global abstract map for high-level candidate selection with a local concrete map for precise goal-reaching, effectively managing and updating dynamic changes in the environment. Through extensive experiments in both simulation and real-world scenarios, we demonstrate state-of-the-art performance in 3D open-vocabulary segmentation, efficient scene mapping, and online language-guided navigation.Project page: https://eku127.github.io/DualMap/
>
---
#### [replaced 016] Decoupling Geometry from Optimization in 2D Irregular Cutting and Packing Problems: an Open-Source Collision Detection Engine
- **分类: cs.CG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.08341v2](http://arxiv.org/pdf/2508.08341v2)**

> **作者:** Jeroen Gardeyn; Greet Vanden Berghe; Tony Wauters
>
> **备注:** 25 pages, 16 figures
>
> **摘要:** Addressing irregular cutting and packing (C&P) optimization problems poses two distinct challenges: the geometric challenge of determining whether or not an item can be placed feasibly at a certain position, and the optimization challenge of finding a good solution according to some objective function. Until now, those tackling such problems have had to address both challenges simultaneously, requiring two distinct sets of expertise and a lot of research & development effort. One way to lower this barrier is to decouple the two challenges. In this paper we introduce a powerful collision detection engine (CDE) for 2D irregular C&P problems which assumes full responsibility for the geometric challenge. The CDE (i) allows users to focus with full confidence on their optimization challenge by abstracting geometry away and (ii) enables independent advances to propagate to all optimization algorithms built atop it. We present a set of core principles and design philosophies to model a general and adaptable CDE focused on maximizing performance, accuracy and robustness. These principles are accompanied by a concrete open-source implementation called $\texttt{jagua-rs}$. This paper together with its implementation serves as a catalyst for future advances in irregular C&P problems by providing a solid foundation which can either be used as it currently exists or be further improved upon.
>
---
#### [replaced 017] RoHOI: Robustness Benchmark for Human-Object Interaction Detection
- **分类: cs.CV; cs.HC; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2507.09111v2](http://arxiv.org/pdf/2507.09111v2)**

> **作者:** Di Wen; Kunyu Peng; Kailun Yang; Yufan Chen; Ruiping Liu; Junwei Zheng; Alina Roitberg; Danda Pani Paudel; Luc Van Gool; Rainer Stiefelhagen
>
> **备注:** Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI
>
> **摘要:** Human-Object Interaction (HOI) detection is crucial for robot-human assistance, enabling context-aware support. However, models trained on clean datasets degrade in real-world conditions due to unforeseen corruptions, leading to inaccurate prediction. To address this, we introduce the first robustness benchmark for HOI detection, evaluating model resilience under diverse challenges. Despite advances, current models struggle with environmental variability, occlusions, and noise. Our benchmark, RoHOI, includes 20 corruption types based on the HICO-DET and V-COCO datasets and a new robustness-focused metric. We systematically analyze existing models in the HOI field, revealing significant performance drops under corruptions. To improve robustness, we propose a Semantic-Aware Masking-based Progressive Learning (SAMPL) strategy to guide the model to be optimized based on holistic and partial cues, thus dynamically adjusting the model's optimization to enhance robust feature learning. Extensive experiments show that our approach outperforms state-of-the-art methods, setting a new standard for robust HOI detection. Benchmarks, datasets, and code will be made publicly available at https://github.com/Kratos-Wen/RoHOI.
>
---
#### [replaced 018] BridgeDepth: Bridging Monocular and Stereo Reasoning with Latent Alignment
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.04611v2](http://arxiv.org/pdf/2508.04611v2)**

> **作者:** Tongfan Guan; Jiaxin Guo; Chen Wang; Yun-Hui Liu
>
> **备注:** ICCV 2025 Highlight
>
> **摘要:** Monocular and stereo depth estimation offer complementary strengths: monocular methods capture rich contextual priors but lack geometric precision, while stereo approaches leverage epipolar geometry yet struggle with ambiguities such as reflective or textureless surfaces. Despite post-hoc synergies, these paradigms remain largely disjoint in practice. We introduce a unified framework that bridges both through iterative bidirectional alignment of their latent representations. At its core, a novel cross-attentive alignment mechanism dynamically synchronizes monocular contextual cues with stereo hypothesis representations during stereo reasoning. This mutual alignment resolves stereo ambiguities (e.g., specular surfaces) by injecting monocular structure priors while refining monocular depth with stereo geometry within a single network. Extensive experiments demonstrate state-of-the-art results: \textbf{it reduces zero-shot generalization error by $\!>\!40\%$ on Middlebury and ETH3D}, while addressing longstanding failures on transparent and reflective surfaces. By harmonizing multi-view geometry with monocular context, our approach enables robust 3D perception that transcends modality-specific limitations. Codes available at https://github.com/aeolusguan/BridgeDepth.
>
---
#### [replaced 019] Human2Robot: Learning Robot Actions from Paired Human-Robot Videos
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.16587v3](http://arxiv.org/pdf/2502.16587v3)**

> **作者:** Sicheng Xie; Haidong Cao; Zejia Weng; Zhen Xing; Haoran Chen; Shiwei Shen; Jiaqi Leng; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Distilling knowledge from human demonstrations is a promising way for robots to learn and act. Existing methods, which often rely on coarsely-aligned video pairs, are typically constrained to learning global or task-level features. As a result, they tend to neglect the fine-grained frame-level dynamics required for complex manipulation and generalization to novel tasks. We posit that this limitation stems from a vicious circle of inadequate datasets and the methods they inspire. To break this cycle, we propose a paradigm shift that treats fine-grained human-robot alignment as a conditional video generation problem. To this end, we first introduce H&R, a novel third-person dataset containing 2,600 episodes of precisely synchronized human and robot motions, collected using a VR teleoperation system. We then present Human2Robot, a framework designed to leverage this data. Human2Robot employs a Video Prediction Model to learn a rich and implicit representation of robot dynamics by generating robot videos from human input, which in turn guides a decoupled action decoder. Our real-world experiments demonstrate that this approach not only achieves high performance on seen tasks but also exhibits significant one-shot generalization to novel positions, objects, instances, and even new task categories.
>
---
#### [replaced 020] Navigating Robot Swarm Through a Virtual Tube with Flow-Adaptive Distribution Control
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2501.11938v2](http://arxiv.org/pdf/2501.11938v2)**

> **作者:** Yongwei Zhang; Shuli Lv; Kairong Liu; Quanyi Liang; Quan Quan; Zhikun She
>
> **备注:** 8 pages(brief paper), 12 figures
>
> **摘要:** With the rapid development of robot swarm technology and its diverse applications, navigating robot swarms through complex environments has emerged as a critical research direction. To ensure safe navigation and avoid potential collisions with obstacles, the concept of virtual tubes has been introduced to define safe and navigable regions. However, current control methods in virtual tubes face the congestion issues, particularly in narrow ones with low throughput. To address these challenges, we first propose a novel control method that combines a modified artificial potential field (APF) for swarm navigation and density feedback control for distribution regulation. Then we generate a global velocity field that not only ensures collision-free navigation but also achieves locally input-to-state stability (LISS) for density tracking. Finally, numerical simulations and realistic applications validate the effectiveness and advantages of the proposed method in navigating robot swarms through narrow virtual tubes.
>
---
#### [replaced 021] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.08896v2](http://arxiv.org/pdf/2508.08896v2)**

> **作者:** Haoyu Zhao; Linghao Zhuang; Xingyue Zhao; Cheng Zeng; Haoran Xu; Yuming Jiang; Jun Cen; Kexiang Wang; Jiayan Guo; Siteng Huang; Xin Li; Deli Zhao; Hua Zou
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
>
---
#### [replaced 022] A Minimal Model for Emergent Collective Behaviors in Autonomous Robotic Multi-Agent Systems
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.08473v2](http://arxiv.org/pdf/2508.08473v2)**

> **作者:** Hossein B. Jond
>
> **备注:** Accepted for IEEE Transactions on Cognitive and Developmental Systems. Simulation video for Fig. 8: https://youtube.com/shorts/StHrtnSJyyg Simulation video for Fig. 10: https://youtu.be/Z26m7M-63D4
>
> **摘要:** Collective behaviors such as swarming and flocking emerge from simple, decentralized interactions in biological systems. Existing models, such as Vicsek and Cucker-Smale, lack collision avoidance, whereas the Olfati-Saber model imposes rigid formations, limiting their applicability in swarm robotics. To address these limitations, this paper proposes a minimal yet expressive model that governs agent dynamics using relative positions, velocities, and local density, modulated by two tunable parameters: the spatial offset and kinetic offset. The model achieves spatially flexible, collision-free behaviors that reflect naturalistic group dynamics. Furthermore, we extend the framework to cognitive autonomous systems, enabling energy-aware phase transitions between swarming and flocking through adaptive control parameter tuning. This cognitively inspired approach offers a robust foundation for real-world applications in multi-robot systems, particularly autonomous aerial swarms.
>
---
#### [replaced 023] GeoVLA: Empowering 3D Representations in Vision-Language-Action Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.09071v2](http://arxiv.org/pdf/2508.09071v2)**

> **作者:** Lin Sun; Bin Xie; Yingfei Liu; Hao Shi; Tiancai Wang; Jiale Cao
>
> **备注:** The project is visible at https://linsun449.github.io/GeoVLA/
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising approach for enabling robots to follow language instructions and predict corresponding actions. However, current VLA models mainly rely on 2D visual inputs, neglecting the rich geometric information in the 3D physical world, which limits their spatial awareness and adaptability. In this paper, we present GeoVLA, a novel VLA framework that effectively integrates 3D information to advance robotic manipulation. It uses a vision-language model (VLM) to process images and language instructions,extracting fused vision-language embeddings. In parallel, it converts depth maps into point clouds and employs a customized point encoder, called Point Embedding Network, to generate 3D geometric embeddings independently. These produced embeddings are then concatenated and processed by our proposed spatial-aware action expert, called 3D-enhanced Action Expert, which combines information from different sensor modalities to produce precise action sequences. Through extensive experiments in both simulation and real-world environments, GeoVLA demonstrates superior performance and robustness. It achieves state-of-the-art results in the LIBERO and ManiSkill2 simulation benchmarks and shows remarkable robustness in real-world tasks requiring height adaptability, scale awareness and viewpoint invariance.
>
---
#### [replaced 024] AgentWorld: An Interactive Simulation Platform for Scene Construction and Mobile Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07770v2](http://arxiv.org/pdf/2508.07770v2)**

> **作者:** Yizheng Zhang; Zhenjun Yu; Jiaxin Lai; Cewu Lu; Lei Han
>
> **备注:** Accepted by Conference on Robot Learning 2025
>
> **摘要:** We introduce AgentWorld, an interactive simulation platform for developing household mobile manipulation capabilities. Our platform combines automated scene construction that encompasses layout generation, semantic asset placement, visual material configuration, and physics simulation, with a dual-mode teleoperation system supporting both wheeled bases and humanoid locomotion policies for data collection. The resulting AgentWorld Dataset captures diverse tasks ranging from primitive actions (pick-and-place, push-pull, etc.) to multistage activities (serve drinks, heat up food, etc.) across living rooms, bedrooms, and kitchens. Through extensive benchmarking of imitation learning methods including behavior cloning, action chunking transformers, diffusion policies, and vision-language-action models, we demonstrate the dataset's effectiveness for sim-to-real transfer. The integrated system provides a comprehensive solution for scalable robotic skill acquisition in complex home environments, bridging the gap between simulation-based training and real-world deployment. The code, datasets will be available at https://yizhengzhang1.github.io/agent_world/
>
---
#### [replaced 025] Barriers on the EDGE: A scalable CBF architecture over EDGE for safe aerial-ground multi-agent coordination
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2411.16608v2](http://arxiv.org/pdf/2411.16608v2)**

> **作者:** Viswa Narayanan Sankaranarayanan; Achilleas Santi Seisa; Akshit Saradagi; Sumeet Satpute; George Nikolakopoulos
>
> **备注:** 6 pages, 2 figures, first draft of a paper currently under review
>
> **摘要:** In this article, we propose a control architecture for the safe, coordinated operation of a multi-agent system with aerial (UAVs) and ground (UGVs) robots in a confined task space. We consider the case where the aerial and ground operations are coupled, enabled by the capability of the aerial robots to land on moving ground robots. The proposed method uses time-varying Control Barrier Functions (CBFs) to impose safety constraints associated with (i) collision avoidance between agents, (ii) landing of UAVs on mobile UGVs, and (iii) task space restriction. Further, this article addresses the challenge induced by the rapid increase in the number of CBF constraints with the increasing number of agents through a hybrid centralized-distributed coordination approach that determines the set of CBF constraints that is relevant for every aerial and ground agent at any given time. A centralized node (Watcher), hosted by an edge computing cluster, activates the relevant constraints, thus reducing the network complexity and the need for high onboard processing on the robots. The CBF constraints are enforced in a distributed manner by individual robots that run a nominal controller and safety filter locally to overcome latency and other network nonidealities.
>
---
#### [replaced 026] Responsive Noise-Relaying Diffusion Policy: Responsive and Efficient Visuomotor Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.12724v2](http://arxiv.org/pdf/2502.12724v2)**

> **作者:** Zhuoqun Chen; Xiu Yuan; Tongzhou Mu; Hao Su
>
> **备注:** Project website: https://rnr-dp.github.io
>
> **摘要:** Imitation learning is an efficient method for teaching robots a variety of tasks. Diffusion Policy, which uses a conditional denoising diffusion process to generate actions, has demonstrated superior performance, particularly in learning from multi-modal demonstrates. However, it relies on executing multiple actions predicted from the same inference step to retain performance and prevent mode bouncing, which limits its responsiveness, as actions are not conditioned on the most recent observations. To address this, we introduce Responsive Noise-Relaying Diffusion Policy (RNR-DP), which maintains a noise-relaying buffer with progressively increasing noise levels and employs a sequential denoising mechanism that generates immediate, noise-free actions at the head of the sequence, while appending noisy actions at the tail. This ensures that actions are responsive and conditioned on the latest observations, while maintaining motion consistency through the noise-relaying buffer. This design enables the handling of tasks requiring responsive control, and accelerates action generation by reusing denoising steps. Experiments on response-sensitive tasks demonstrate that, compared to Diffusion Policy, ours achieves 18% improvement in success rate. Further evaluation on regular tasks demonstrates that RNR-DP also exceeds the best acceleration method (DDIM) by 6.9% in success rate, highlighting its computational efficiency advantage in scenarios where responsiveness is less critical. Our project page is available at https://rnr-dp.github.io
>
---
#### [replaced 027] Audio-3DVG: Unified Audio -- Point Cloud Fusion for 3D Visual Grounding
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00669v2](http://arxiv.org/pdf/2507.00669v2)**

> **作者:** Duc Cao-Dinh; Khai Le-Duc; Anh Dao; Bach Phan Tat; Chris Ngo; Duy M. H. Nguyen; Nguyen X. Khanh; Thanh Nguyen-Tang
>
> **备注:** Preprint, 51 pages
>
> **摘要:** 3D Visual Grounding (3DVG) involves localizing target objects in 3D point clouds based on natural language. While prior work has made strides using textual descriptions, leveraging spoken language-known as Audio-based 3D Visual Grounding-remains underexplored and challenging. Motivated by advances in automatic speech recognition (ASR) and speech representation learning, we propose Audio-3DVG, a simple yet effective framework that integrates audio and spatial information for enhanced grounding. Rather than treating speech as a monolithic input, we decompose the task into two complementary components. First, we introduce (i) Object Mention Detection, a multi-label classification task that explicitly identifies which objects are referred to in the audio, enabling more structured audio-scene reasoning. Second, we propose an (ii) Audio-Guided Attention module that models the interactions between target candidates and mentioned objects, enhancing discrimination in cluttered 3D environments. To support benchmarking, we (iii) synthesize audio descriptions for standard 3DVG datasets, including ScanRefer, Sr3D, and Nr3D. Experimental results demonstrate that Audio-3DVG not only achieves new state-of-the-art performance in audio-based grounding, but also competes with text-based methods, highlight the promise of integrating spoken language into 3D vision tasks.
>
---
#### [replaced 028] Episodic Memory Verbalization using Hierarchical Representations of Life-Long Robot Experience
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.17702v2](http://arxiv.org/pdf/2409.17702v2)**

> **作者:** Leonard Bärmann; Chad DeChant; Joana Plewnia; Fabian Peller-Konrad; Daniel Bauer; Tamim Asfour; Alex Waibel
>
> **备注:** Humanoids 2025. Code, data and demo videos at https://hierarchical-emv.github.io
>
> **摘要:** Verbalization of robot experience, i.e., summarization of and question answering about a robot's past, is a crucial ability for improving human-robot interaction. Previous works applied rule-based systems or fine-tuned deep models to verbalize short (several-minute-long) streams of episodic data, limiting generalization and transferability. In our work, we apply large pretrained models to tackle this task with zero or few examples, and specifically focus on verbalizing life-long experiences. For this, we derive a tree-like data structure from episodic memory (EM), with lower levels representing raw perception and proprioception data, and higher levels abstracting events to natural language concepts. Given such a hierarchical representation built from the experience stream, we apply a large language model as an agent to interactively search the EM given a user's query, dynamically expanding (initially collapsed) tree nodes to find the relevant information. The approach keeps computational costs low even when scaling to months of robot experience data. We evaluate our method on simulated household robot data, human egocentric videos, and real-world robot recordings, demonstrating its flexibility and scalability.
>
---
#### [replaced 029] Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning Code LLMs
- **分类: cs.CL; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.20179v4](http://arxiv.org/pdf/2405.20179v4)**

> **作者:** Zichao Hu; Junyi Jessy Li; Arjun Guha; Joydeep Biswas
>
> **摘要:** Code LLMs have shown promising results with converting tasks in natural language to programs that can be executed by service robots. We are interested in finetuning small, specialized LLMs for this purpose, but collecting datasets of task-program pairs specific to each robot is time-consuming and expensive. While approaches such as SELF-INSTRUCT and EVOL-INSTRUCT are capable of generating novel tasks given a few examples, they are unable to provide the corresponding programs that correctly abide by physical-world and robot-constraints using the provided programming interface. Using a simulator is a natural potential solution to checking for such constraints, but building simulation environments that can handle arbitrary tasks and their necessary objects and locations, is challenging. To address these challenges, we introduce ROBO-INSTRUCT, which synthesizes task-specific simulation environments on the fly during program execution, by opportunistically inferring entity properties and enforcing corresponding constraints based on how the entities are used in the task program. Additionally, ROBO-INSTRUCT integrates an LLM-aided post-processing procedure to refine instructions for better alignment with robot programs. We demonstrate the effectiveness of ROBO-INSTRUCT across multiple LLMs, showing that our fine-tuned models outperform all baseline methods and even match or surpass the performance of several larger and proprietary models.
>
---
#### [replaced 030] GraspClutter6D: A Large-scale Real-world Dataset for Robust Perception and Grasping in Cluttered Scenes
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.06866v2](http://arxiv.org/pdf/2504.06866v2)**

> **作者:** Seunghyeok Back; Joosoon Lee; Kangmin Kim; Heeseon Rho; Geonhyup Lee; Raeyoung Kang; Sangbeom Lee; Sangjun Noh; Youngjin Lee; Taeyeop Lee; Kyoobin Lee
>
> **摘要:** Robust grasping in cluttered environments remains an open challenge in robotics. While benchmark datasets have significantly advanced deep learning methods, they mainly focus on simplistic scenes with light occlusion and insufficient diversity, limiting their applicability to practical scenarios. We present GraspClutter6D, a large-scale real-world grasping dataset featuring: (1) 1,000 highly cluttered scenes with dense arrangements (14.1 objects/scene, 62.6\% occlusion), (2) comprehensive coverage across 200 objects in 75 environment configurations (bins, shelves, and tables) captured using four RGB-D cameras from multiple viewpoints, and (3) rich annotations including 736K 6D object poses and 9.3B feasible robotic grasps for 52K RGB-D images. We benchmark state-of-the-art segmentation, object pose estimation, and grasp detection methods to provide key insights into challenges in cluttered environments. Additionally, we validate the dataset's effectiveness as a training resource, demonstrating that grasping networks trained on GraspClutter6D significantly outperform those trained on existing datasets in both simulation and real-world experiments. The dataset, toolkit, and annotation tools are publicly available on our project website: https://sites.google.com/view/graspclutter6d.
>
---
#### [replaced 031] SWA-SOP: Spatially-aware Window Attention for Semantic Occupancy Prediction in Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18785v2](http://arxiv.org/pdf/2506.18785v2)**

> **作者:** Helin Cao; Rafael Materla; Sven Behnke
>
> **备注:** 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC), Vienna, Austria, Oct 2025
>
> **摘要:** Perception systems in autonomous driving rely on sensors such as LiDAR and cameras to perceive the 3D environment. However, due to occlusions and data sparsity, these sensors often fail to capture complete information. Semantic Occupancy Prediction (SOP) addresses this challenge by inferring both occupancy and semantics of unobserved regions. Existing transformer-based SOP methods lack explicit modeling of spatial structure in attention computation, resulting in limited geometric awareness and poor performance in sparse or occluded areas. To this end, we propose Spatially-aware Window Attention (SWA), a novel mechanism that incorporates local spatial context into attention. SWA significantly improves scene completion and achieves state-of-the-art results on LiDAR-based SOP benchmarks. We further validate its generality by integrating SWA into a camera-based SOP pipeline, where it also yields consistent gains across modalities.
>
---
#### [replaced 032] ParkDiffusion: Heterogeneous Multi-Agent Multi-Modal Trajectory Prediction for Automated Parking using Diffusion Models
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.00586v2](http://arxiv.org/pdf/2505.00586v2)**

> **作者:** Jiarong Wei; Niclas Vödisch; Anna Rehr; Christian Feist; Abhinav Valada
>
> **备注:** IROS 2025 Camera-Ready Version
>
> **摘要:** Automated parking is a critical feature of Advanced Driver Assistance Systems (ADAS), where accurate trajectory prediction is essential to bridge perception and planning modules. Despite its significance, research in this domain remains relatively limited, with most existing studies concentrating on single-modal trajectory prediction of vehicles. In this work, we propose ParkDiffusion, a novel approach that predicts the trajectories of both vehicles and pedestrians in automated parking scenarios. ParkDiffusion employs diffusion models to capture the inherent uncertainty and multi-modality of future trajectories, incorporating several key innovations. First, we propose a dual map encoder that processes soft semantic cues and hard geometric constraints using a two-step cross-attention mechanism. Second, we introduce an adaptive agent type embedding module, which dynamically conditions the prediction process on the distinct characteristics of vehicles and pedestrians. Third, to ensure kinematic feasibility, our model outputs control signals that are subsequently used within a kinematic framework to generate physically feasible trajectories. We evaluate ParkDiffusion on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Our work establishes a new baseline for heterogeneous trajectory prediction in parking scenarios, outperforming existing methods by a considerable margin.
>
---
#### [replaced 033] Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.13265v2](http://arxiv.org/pdf/2506.13265v2)**

> **作者:** Rohit Mohan; Julia Hindel; Florian Drews; Claudius Gläser; Daniele Cattaneo; Abhinav Valada
>
> **摘要:** Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods.
>
---
#### [replaced 034] REACT: Real-time Efficient Attribute Clustering and Transfer for Updatable 3D Scene Graph
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03412v2](http://arxiv.org/pdf/2503.03412v2)**

> **作者:** Phuoc Nguyen; Francesco Verdoja; Ville Kyrki
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Modern-day autonomous robots need high-level map representations to perform sophisticated tasks. Recently, 3D scene graphs (3DSGs) have emerged as a promising alternative to traditional grid maps, blending efficient memory use and rich feature representation. However, most efforts to apply them have been limited to static worlds. This work introduces REACT, a framework that efficiently performs real-time attribute clustering and transfer to relocalize object nodes in a 3DSG. REACT employs a novel method for comparing object instances using an embedding model trained on triplet loss, facilitating instance clustering and matching. Experimental results demonstrate that REACT is able to relocalize objects while maintaining computational efficiency. The REACT framework's source code will be available as an open-source project, promoting further advancements in reusable and updatable 3DSGs.
>
---
#### [replaced 035] Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.05968v2](http://arxiv.org/pdf/2506.05968v2)**

> **作者:** Motoki Omura; Kazuki Ota; Takayuki Osa; Yusuke Mukuta; Tatsuya Harada
>
> **备注:** Accepted at ICML 2025. Source code: https://github.com/motokiomura/annealed-q-learning
>
> **摘要:** For continuous action spaces, actor-critic methods are widely used in online reinforcement learning (RL). However, unlike RL algorithms for discrete actions, which generally model the optimal value function using the Bellman optimality operator, RL algorithms for continuous actions typically model Q-values for the current policy using the Bellman operator. These algorithms for continuous actions rely exclusively on policy updates for improvement, which often results in low sample efficiency. This study examines the effectiveness of incorporating the Bellman optimality operator into actor-critic frameworks. Experiments in a simple environment show that modeling optimal values accelerates learning but leads to overestimation bias. To address this, we propose an annealing approach that gradually transitions from the Bellman optimality operator to the Bellman operator, thereby accelerating learning while mitigating bias. Our method, combined with TD3 and SAC, significantly outperforms existing approaches across various locomotion and manipulation tasks, demonstrating improved performance and robustness to hyperparameters related to optimality. The code for this study is available at https://github.com/motokiomura/annealed-q-learning.
>
---
