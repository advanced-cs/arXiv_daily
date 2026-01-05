# 机器人 cs.RO

- **最新发布 24 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] LLM-Based Agentic Exploration for Robot Navigation & Manipulation with Skill Orchestration
- **分类: cs.RO**

- **简介: 该论文研究机器人导航与操作任务，解决从自然语言指令到多店导航和物体抓取的端到端问题。通过LLM生成动作，结合模块化控制实现有效路径规划与执行。**

- **链接: [https://arxiv.org/pdf/2601.00555v1](https://arxiv.org/pdf/2601.00555v1)**

> **作者:** Abu Hanif Muhammad Syarubany; Farhan Zaki Rahmani; Trio Widianto
>
> **摘要:** This paper presents an end-to-end LLM-based agentic exploration system for an indoor shopping task, evaluated in both Gazebo simulation and a corresponding real-world corridor layout. The robot incrementally builds a lightweight semantic map by detecting signboards at junctions and storing direction-to-POI relations together with estimated junction poses, while AprilTags provide repeatable anchors for approach and alignment. Given a natural-language shopping request, an LLM produces a constrained discrete action at each junction (direction and whether to enter a store), and a ROS finite-state main controller executes the decision by gating modular motion primitives, including local-costmap-based obstacle avoidance, AprilTag approaching, store entry, and grasping. Qualitative results show that the integrated stack can perform end-to-end task execution from user instruction to multi-store navigation and object retrieval, while remaining modular and debuggable through its text-based map and logged decision history.
>
---
#### [new 002] Replaceable Bit-based Gripper for Picking Cluttered Food Items
- **分类: cs.RO**

- **简介: 该论文属于食品抓取任务，解决杂乱食品重量处理问题。设计可更换夹具系统，提升抓取与投放精度，适用于不同形状食品。**

- **链接: [https://arxiv.org/pdf/2601.00305v1](https://arxiv.org/pdf/2601.00305v1)**

> **作者:** Prashant Kumar; Yukiyasu Domae; Weiwei Wan; Kensuke Harada
>
> **摘要:** The food packaging industry goes through changes in food items and their weights quite rapidly. These items range from easy-to-pick, single-piece food items to flexible, long and cluttered ones. We propose a replaceable bit-based gripper system to tackle the challenge of weight-based handling of cluttered food items. The gripper features specialized food attachments(bits) that enhance its grasping capabilities, and a belt replacement system allows switching between different food items during packaging operations. It offers a wide range of control options, enabling it to grasp and drop specific weights of granular, cluttered, and entangled foods. We specifically designed bits for two flexible food items that differ in shape: ikura(salmon roe) and spaghetti. They represent the challenging categories of sticky, granular food and long, sticky, cluttered food, respectively. The gripper successfully picked up both spaghetti and ikura and demonstrated weight-specific dropping of these items with an accuracy over 80% and 95% respectively. The gripper system also exhibited quick switching between different bits, leading to the handling of a large range of food items.
>
---
#### [new 003] Space Debris Removal using Nano-Satellites controlled by Low-Power Autonomous Agents
- **分类: cs.RO**

- **简介: 论文研究利用低功耗自主代理控制纳米卫星清除太空碎片的任务，旨在解决太空碎片威胁卫星安全的问题，通过实验验证了自主纳米卫星群的可行性与能效。**

- **链接: [https://arxiv.org/pdf/2601.00465v1](https://arxiv.org/pdf/2601.00465v1)**

> **作者:** Dennis Christmann; Juan F. Gutierrez; Sthiti Padhi; Patrick Plörer; Aditya Takur; Simona Silvestri; Andres Gomez
>
> **备注:** This is an open-access, author-archived version of a manuscript published in European Conference on Multi-Agent Systems 2024
>
> **摘要:** Space debris is an ever-increasing problem in space travel. There are already many old, no longer functional spacecraft and debris orbiting the earth, which endanger both the safe operation of satellites and space travel. Small nano-satellite swarms can address this problem by autonomously de-orbiting debris safely into the Earth's atmosphere. This work builds on the recent advances of autonomous agents deployed in resource-constrained platforms and shows a first simplified approach how such intelligent and autonomous nano-satellite swarms can be realized. We implement our autonomous agent software on wireless microcontrollers and perform experiments on a specialized test-bed to show the feasibility and overall energy efficiency of our approach.
>
---
#### [new 004] Calling for Backup: How Children Navigate Successive Robot Communication Failures
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究儿童在机器人连续故障时的反应，属于人机交互任务。旨在解决儿童如何应对机器人错误的问题，通过实验分析其行为与情绪变化。**

- **链接: [https://arxiv.org/pdf/2601.00754v1](https://arxiv.org/pdf/2601.00754v1)**

> **作者:** Maria Teresa Parreira; Isabel Neto; Filipa Rocha; Wendy Ju
>
> **摘要:** How do children respond to repeated robot errors? While prior research has examined adult reactions to successive robot errors, children's responses remain largely unexplored. In this study, we explore children's reactions to robot social errors and performance errors. For the latter, this study reproduces the successive robot failure paradigm of Liu et al. with child participants (N=59, ages 8-10) to examine how young users respond to repeated robot conversational errors. Participants interacted with a robot that failed to understand their prompts three times in succession, with their behavioral responses video-recorded and analyzed. We found both similarities and differences compared to adult responses from the original study. Like adults, children adjusted their prompts, modified their verbal tone, and exhibited increasingly emotional non-verbal responses throughout successive errors. However, children demonstrated more disengagement behaviors, including temporarily ignoring the robot or actively seeking an adult. Errors did not affect participants' perception of the robot, suggesting more flexible conversational expectations in children. These findings inform the design of more effective and developmentally appropriate human-robot interaction systems for young users.
>
---
#### [new 005] SLAP: Slapband-based Autonomous Perching Drone with Failure Recovery for Vertical Tree Trunks
- **分类: cs.RO**

- **简介: 该论文属于无人机自主停靠任务，旨在解决垂直表面停靠及故障恢复问题。研究提出SLAP系统，包含视觉检测、IMU故障检测和弹性抓取等模块，实现安全停靠与失败恢复。**

- **链接: [https://arxiv.org/pdf/2601.00238v1](https://arxiv.org/pdf/2601.00238v1)**

> **作者:** Julia Di; Kenneth A. W. Hoffmann; Tony G. Chen; Tian-Ao Ren; Mark R. Cutkosky
>
> **备注:** Paper accepted to IEEE Aerospace Conference 2026. This is a pre-print
>
> **摘要:** Perching allows unmanned aerial vehicles (UAVs) to reduce energy consumption, remain anchored for surface sampling operations, or stably survey their surroundings. Previous efforts for perching on vertical surfaces have predominantly focused on lightweight mechanical design solutions with relatively scant system-level integration. Furthermore, perching strategies for vertical surfaces commonly require high-speed, aggressive landing operations that are dangerous for a surveyor drone with sensitive electronics onboard. This work presents the preliminary investigation of a perching approach suitable for larger drones that both gently perches on vertical tree trunks and reacts and recovers from perch failures. The system in this work, called SLAP, consists of vision-based perch site detector, an IMU (inertial-measurement-unit)-based perch failure detector, an attitude controller for soft perching, an optical close-range detection system, and a fast active elastic gripper with microspines made from commercially-available slapbands. We validated this approach on a modified 1.2 kg commercial quadrotor with component and system analysis. Initial human-in-the-loop autonomous indoor flight experiments achieved a 75% perch success rate on a real oak tree segment across 20 flights, and 100% perch failure recovery across 2 flights with induced failures.
>
---
#### [new 006] RoboReward: General-Purpose Vision-Language Reward Models for Robotics
- **分类: cs.RO**

- **简介: 该论文提出RoboReward，解决机器人强化学习中奖励设计困难的问题。构建了大规模机器人数据集，训练视觉-语言奖励模型，提升机器人任务的政策学习效果。**

- **链接: [https://arxiv.org/pdf/2601.00675v1](https://arxiv.org/pdf/2601.00675v1)**

> **作者:** Tony Lee; Andrew Wagenmaker; Karl Pertsch; Percy Liang; Sergey Levine; Chelsea Finn
>
> **摘要:** A well-designed reward is critical for effective reinforcement learning-based policy improvement. In real-world robotic domains, obtaining such rewards typically requires either labor-intensive human labeling or brittle, handcrafted objectives. Vision-language models (VLMs) have shown promise as automatic reward models, yet their effectiveness on real robot tasks is poorly understood. In this work, we aim to close this gap by introducing (1) \textbf{RoboReward}, a robotics reward dataset and benchmark built on large-scale real-robot corpora from Open X-Embodiment (OXE) and RoboArena, and (2) vision-language reward models trained on this dataset (RoboReward 4B/8B). Because OXE is success-heavy and lacks failure examples, we propose a \emph{negative examples data augmentation} pipeline that generates calibrated \emph{negatives} and \emph{near-misses} via counterfactual relabeling of successful episodes and temporal clipping to create partial-progress outcomes from the same videos. Using this framework, we produce an extensive training and evaluation dataset that spans diverse tasks and embodiments and enables systematic evaluation of whether state-of-the-art VLMs can reliably provide rewards for robotics. Our evaluation of leading open-weight and proprietary VLMs reveals that no model excels across all tasks, underscoring substantial room for improvement. We then train general-purpose 4B- and 8B-parameter models that outperform much larger VLMs in assigning rewards for short-horizon robotic tasks. Finally, we deploy the 8B-parameter reward VLM in real-robot reinforcement learning and find that it improves policy learning over Gemini Robotics-ER 1.5, a frontier physical reasoning VLM trained on robotics data, by a large margin, while substantially narrowing the gap to RL training with human-provided rewards.
>
---
#### [new 007] SLEI3D: Simultaneous Exploration and Inspection via Heterogeneous Fleets under Limited Communication
- **分类: cs.RO**

- **简介: 该论文研究未知环境下的协同探索与检测任务，解决有限通信条件下多机器人协作问题，提出SLEI3D框架实现3D探索、自适应检测和及时通信。**

- **链接: [https://arxiv.org/pdf/2601.00163v1](https://arxiv.org/pdf/2601.00163v1)**

> **作者:** Junfeng Chen; Yuxiao Zhu; Xintong Zhang; Bing Luo; Meng Guo
>
> **摘要:** Robotic fleets such as unmanned aerial and ground vehicles have been widely used for routine inspections of static environments, where the areas of interest are known and planned in advance. However, in many applications, such areas of interest are unknown and should be identified online during exploration. Thus, this paper considers the problem of simultaneous exploration, inspection of unknown environments and then real-time communication to a mobile ground control station to report the findings. The heterogeneous robots are equipped with different sensors, e.g., long-range lidars for fast exploration and close-range cameras for detailed inspection. Furthermore, global communication is often unavailable in such environments, where the robots can only communicate with each other via ad-hoc wireless networks when they are in close proximity and free of obstruction. This work proposes a novel planning and coordination framework (SLEI3D) that integrates the online strategies for collaborative 3D exploration, adaptive inspection and timely communication (via the intermit-tent or proactive protocols). To account for uncertainties w.r.t. the number and location of features, a multi-layer and multi-rate planning mechanism is developed for inter-and-intra robot subgroups, to actively meet and coordinate their local plans. The proposed framework is validated extensively via high-fidelity simulations of numerous large-scale missions with up to 48 robots and 384 thousand cubic meters. Hardware experiments of 7 robots are also conducted. Project website is available at https://junfengchen-robotics.github.io/SLEI3D/.
>
---
#### [new 008] Reinforcement learning with timed constraints for robotics motion planning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动规划任务，解决在动态环境中满足时间约束的强化学习问题。通过将MITL公式转换为Timed-LDGBA，构建适合Q-learning的模型，实现可靠的时间约束规划。**

- **链接: [https://arxiv.org/pdf/2601.00087v1](https://arxiv.org/pdf/2601.00087v1)**

> **作者:** Zhaoan Wang; Junchao Li; Mahdi Mohammad; Shaoping Xiao
>
> **摘要:** Robotic systems operating in dynamic and uncertain environments increasingly require planners that satisfy complex task sequences while adhering to strict temporal constraints. Metric Interval Temporal Logic (MITL) offers a formal and expressive framework for specifying such time-bounded requirements; however, integrating MITL with reinforcement learning (RL) remains challenging due to stochastic dynamics and partial observability. This paper presents a unified automata-based RL framework for synthesizing policies in both Markov Decision Processes (MDPs) and Partially Observable Markov Decision Processes (POMDPs) under MITL specifications. MITL formulas are translated into Timed Limit-Deterministic Generalized Büchi Automata (Timed-LDGBA) and synchronized with the underlying decision process to construct product timed models suitable for Q-learning. A simple yet expressive reward structure enforces temporal correctness while allowing additional performance objectives. The approach is validated in three simulation studies: a $5 \times 5$ grid-world formulated as an MDP, a $10 \times 10$ grid-world formulated as a POMDP, and an office-like service-robot scenario. Results demonstrate that the proposed framework consistently learns policies that satisfy strict time-bounded requirements under stochastic transitions, scales to larger state spaces, and remains effective in partially observable environments, highlighting its potential for reliable robotic planning in time-critical and uncertain settings.
>
---
#### [new 009] Vision-based Goal-Reaching Control for Mobile Robots Using a Hierarchical Learning Framework
- **分类: cs.RO**

- **简介: 该论文属于移动机器人控制任务，解决大型机器人在复杂地形中安全到达目标的问题。通过分层学习框架，结合视觉定位、强化学习与模型预测控制，确保系统稳定与安全。**

- **链接: [https://arxiv.org/pdf/2601.00610v1](https://arxiv.org/pdf/2601.00610v1)**

> **作者:** Mehdi Heydari Shahna; Pauli Mustalahti; Jouni Mattila
>
> **摘要:** Reinforcement learning (RL) is effective in many robotic applications, but it requires extensive exploration of the state-action space, during which behaviors can be unsafe. This significantly limits its applicability to large robots with complex actuators operating on unstable terrain. Hence, to design a safe goal-reaching control framework for large-scale robots, this paper decomposes the whole system into a set of tightly coupled functional modules. 1) A real-time visual pose estimation approach is employed to provide accurate robot states to 2) an RL motion planner for goal-reaching tasks that explicitly respects robot specifications. The RL module generates real-time smooth motion commands for the actuator system, independent of its underlying dynamic complexity. 3) In the actuation mechanism, a supervised deep learning model is trained to capture the complex dynamics of the robot and provide this model to 4) a model-based robust adaptive controller that guarantees the wheels track the RL motion commands even on slip-prone terrain. 5) Finally, to reduce human intervention, a mathematical safety supervisor monitors the robot, stops it on unsafe faults, and autonomously guides it back to a safe inspection area. The proposed framework guarantees uniform exponential stability of the actuation system and safety of the whole operation. Experiments on a 6,000 kg robot in different scenarios confirm the effectiveness of the proposed framework.
>
---
#### [new 010] Pure Inertial Navigation in Challenging Environments with Wheeled and Chassis Mounted Inertial Sensors
- **分类: cs.RO**

- **简介: 该论文属于导航任务，解决在无GNSS环境下纯惯性导航误差大的问题，通过融合轮式和车身惯性传感器提升定位精度。**

- **链接: [https://arxiv.org/pdf/2601.00275v1](https://arxiv.org/pdf/2601.00275v1)**

> **作者:** Dusan Nemec; Gal Versano; Itai Savin; Vojtech Simak; Juraj Kekelak; Itzik Klein
>
> **摘要:** Autonomous vehicles and wheeled robots are widely used in many applications in both indoor and outdoor settings. In practical situations with limited GNSS signals or degraded lighting conditions, the navigation solution may rely only on inertial sensors and as result drift in time due to errors in the inertial measurement. In this work, we propose WiCHINS, a wheeled and chassis inertial navigation system by combining wheel-mounted-inertial sensors with a chassis-mounted inertial sensor for accurate pure inertial navigation. To that end, we derive a three-stage framework, each with a dedicated extended Kalman filter. This framework utilizes the benefits of each location (wheel/body) during the estimation process. To evaluate our proposed approach, we employed a dataset with five inertial measurement units with a total recording time of 228.6 minutes. We compare our approach with four other inertial baselines and demonstrate an average position error of 11.4m, which is $2.4\%$ of the average traveled distance, using two wheels and one body inertial measurement units. As a consequence, our proposed method enables robust navigation in challenging environments and helps bridge the pure-inertial performance gap.
>
---
#### [new 011] Variable Elimination in Hybrid Factor Graphs for Discrete-Continuous Inference & Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人领域的估计与推理任务，解决混合连续与离散变量的建模问题。提出一种高效的混合因子图框架及变量消去算法，实现精确的最大后验估计与边缘化。**

- **链接: [https://arxiv.org/pdf/2601.00545v1](https://arxiv.org/pdf/2601.00545v1)**

> **作者:** Varun Agrawal; Frank Dellaert
>
> **摘要:** Many hybrid problems in robotics involve both continuous and discrete components, and modeling them together for estimation tasks has been a long standing and difficult problem. Hybrid Factor Graphs give us a mathematical framework to model these types of problems, however existing approaches for solving them are based on approximations. In this work, we propose an efficient Hybrid Factor Graph framework alongwith a variable elimination algorithm to produce a hybrid Bayes network, which can then be used for exact Maximum A Posteriori estimation and marginalization over both sets of variables. Our approach first develops a novel hybrid Gaussian factor which can connect to both discrete and continuous variables, and a hybrid conditional which can represent multiple continuous hypotheses conditioned on the discrete variables. Using these representations, we derive the process of hybrid variable elimination under the Conditional Linear Gaussian scheme, giving us exact posteriors as hybrid Bayes network. To bound the number of discrete hypotheses, we use a tree-structured representation of the factors coupled with a simple pruning and probabilistic assignment scheme, which allows for tractable inference. We demonstrate the applicability of our framework on a SLAM dataset with ambiguous measurements, where discrete choices for the most likely measurement have to be made. Our demonstrated results showcase the accuracy, generality, and simplicity of our hybrid factor graph framework.
>
---
#### [new 012] DefVINS: Visual-Inertial Odometry for Deformable Scenes
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-惯性里程计任务，解决变形场景下的定位问题。提出DefVINS框架，分离刚性与非刚性运动，提升非刚性环境下的定位鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.00702v1](https://arxiv.org/pdf/2601.00702v1)**

> **作者:** Samuel Cerezo; Javier Civera
>
> **备注:** 4 figures, 3 tables. Submitted to RA-L
>
> **摘要:** Deformable scenes violate the rigidity assumptions underpinning classical visual-inertial odometry (VIO), often leading to over-fitting to local non-rigid motion or severe drift when deformation dominates visual parallax. We introduce DefVINS, a visual-inertial odometry framework that explicitly separates a rigid, IMU-anchored state from a non--rigid warp represented by an embedded deformation graph. The system is initialized using a standard VIO procedure that fixes gravity, velocity, and IMU biases, after which non-rigid degrees of freedom are activated progressively as the estimation becomes well conditioned. An observability analysis is included to characterize how inertial measurements constrain the rigid motion and render otherwise unobservable modes identifiable in the presence of deformation. This analysis motivates the use of IMU anchoring and informs a conditioning-based activation strategy that prevents ill-posed updates under poor excitation. Ablation studies demonstrate the benefits of combining inertial constraints with observability-aware deformation activation, resulting in improved robustness under non-rigid environments.
>
---
#### [new 013] Vehicle Painting Robot Path Planning Using Hierarchical Optimization
- **分类: cs.RO; cs.NE**

- **简介: 该论文属于路径规划任务，解决车辆喷涂机器人路径设计问题。通过分层优化方法自动生成满足约束的喷涂路径，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2601.00271v1](https://arxiv.org/pdf/2601.00271v1)**

> **作者:** Yuya Nagai; Hiromitsu Nakamura; Narito Shinmachi; Yuta Higashizono; Satoshi Ono
>
> **摘要:** In vehicle production factories, the vehicle painting process employs multiple robotic arms to simultaneously apply paint to car bodies advancing along a conveyor line. Designing paint paths for these robotic arms, which involves assigning car body areas to arms and determining paint sequences for each arm, remains a time-consuming manual task for engineers, indicating the demand for automation and design time reduction. The unique constraints of the painting process hinder the direct application of conventional robotic path planning techniques, such as those used in welding. Therefore, this paper formulates the design of paint paths as a hierarchical optimization problem, where the upper-layer subproblem resembles a vehicle routing problem (VRP), and the lower-layer subproblem involves detailed path planning. This approach allows the use of different optimization algorithms at each layer, and permits flexible handling of constraints specific to the vehicle painting process through the design of variable representation, constraints, repair operators, and an initialization process at the upper and lower layers. Experiments with three commercially available vehicle models demonstrated that the proposed method can automatically design paths that satisfy all constraints for vehicle painting with quality comparable to those created manually by engineers.
>
---
#### [new 014] Priority-Aware Multi-Robot Coverage Path Planning
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多机器人覆盖路径规划任务，解决优先级区域覆盖效率问题。提出PA-MCPP方法，通过优化路径减少高优先级区域的延迟，同时控制整体完成时间。**

- **链接: [https://arxiv.org/pdf/2601.00580v1](https://arxiv.org/pdf/2601.00580v1)**

> **作者:** Kanghoon Lee; Hyeonjun Kim; Jiachen Li; Jinkyoo Park
>
> **备注:** IEEE Robotics and Automation Letters, 8 pages, 10 figures
>
> **摘要:** Multi-robot systems are widely used for coverage tasks that require efficient coordination across large environments. In Multi-Robot Coverage Path Planning (MCPP), the objective is typically to minimize the makespan by generating non-overlapping paths for full-area coverage. However, most existing methods assume uniform importance across regions, limiting their effectiveness in scenarios where some zones require faster attention. We introduce the Priority-Aware MCPP (PA-MCPP) problem, where a subset of the environment is designated as prioritized zones with associated weights. The goal is to minimize, in lexicographic order, the total priority-weighted latency of zone coverage and the overall makespan. To address this, we propose a scalable two-phase framework combining (1) greedy zone assignment with local search, spanning-tree-based path planning, and (2) Steiner-tree-guided residual coverage. Experiments across diverse scenarios demonstrate that our method significantly reduces priority-weighted latency compared to standard MCPP baselines, while maintaining competitive makespan. Sensitivity analyses further show that the method scales well with the number of robots and that zone coverage behavior can be effectively controlled by adjusting priority weights.
>
---
#### [new 015] Compositional Diffusion with Guided search for Long-Horizon Planning
- **分类: cs.RO**

- **简介: 该论文属于长期规划任务，解决组合生成模型中的模式平均问题。提出CDGS方法，在扩散过程中嵌入搜索，提升全局一致性与局部可行性。**

- **链接: [https://arxiv.org/pdf/2601.00126v1](https://arxiv.org/pdf/2601.00126v1)**

> **作者:** Utkarsh A Mishra; David He; Yongxin Chen; Danfei Xu
>
> **备注:** 38 pages, 18 figures
>
> **摘要:** Generative models have emerged as powerful tools for planning, with compositional approaches offering particular promise for modeling long-horizon task distributions by composing together local, modular generative models. This compositional paradigm spans diverse domains, from multi-step manipulation planning to panoramic image synthesis to long video generation. However, compositional generative models face a critical challenge: when local distributions are multimodal, existing composition methods average incompatible modes, producing plans that are neither locally feasible nor globally coherent. We propose Compositional Diffusion with Guided Search (CDGS), which addresses this \emph{mode averaging} problem by embedding search directly within the diffusion denoising process. Our method explores diverse combinations of local modes through population-based sampling, prunes infeasible candidates using likelihood-based filtering, and enforces global consistency through iterative resampling between overlapping segments. CDGS matches oracle performance on seven robot manipulation tasks, outperforming baselines that lack compositionality or require long-horizon training data. The approach generalizes across domains, enabling coherent text-guided panoramic images and long videos through effective local-to-global message passing. More details: https://cdgsearch.github.io/
>
---
#### [new 016] NMPC-Augmented Visual Navigation and Safe Learning Control for Large-Scale Mobile Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于移动机器人导航与控制任务，解决大尺度机器人在松软地形上的稳定与安全问题，提出融合视觉、模型预测控制和神经网络的控制框架。**

- **链接: [https://arxiv.org/pdf/2601.00609v1](https://arxiv.org/pdf/2601.00609v1)**

> **作者:** Mehdi Heydari Shahna; Pauli Mustalahti; Jouni Mattila
>
> **摘要:** A large-scale mobile robot (LSMR) is a high-order multibody system that often operates on loose, unconsolidated terrain, which reduces traction. This paper presents a comprehensive navigation and control framework for an LSMR that ensures stability and safety-defined performance, delivering robust operation on slip-prone terrain by jointly leveraging high-performance techniques. The proposed architecture comprises four main modules: (1) a visual pose-estimation module that fuses onboard sensors and stereo cameras to provide an accurate, low-latency robot pose, (2) a high-level nonlinear model predictive control that updates the wheel motion commands to correct robot drift from the robot reference pose on slip-prone terrain, (3) a low-level deep neural network control policy that approximates the complex behavior of the wheel-driven actuation mechanism in LSMRs, augmented with robust adaptive control to handle out-of-distribution disturbances, ensuring that the wheels accurately track the updated commands issued by high-level control module, and (4) a logarithmic safety module to monitor the entire robot stack and guarantees safe operation. The proposed low-level control framework guarantees uniform exponential stability of the actuation subsystem, while the safety module ensures the whole system-level safety during operation. Comparative experiments on a 6,000 kg LSMR actuated by two complex electro-hydrostatic drives, while synchronizing modules operating at different frequencies.
>
---
#### [new 017] From 2D to 3D terrain-following area coverage path planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于3D地形跟随区域覆盖路径规划任务，解决农机在复杂地形中高效覆盖的问题，通过生成等距且保持特定高度的路径实现。**

- **链接: [https://arxiv.org/pdf/2601.00614v1](https://arxiv.org/pdf/2601.00614v1)**

> **作者:** Mogens Plessen
>
> **备注:** 6 pages, 10 figures, 1 table
>
> **摘要:** An algorithm for 3D terrain-following area coverage path planning is presented. Multiple adjacent paths are generated that are (i) locally apart from each other by a distance equal to the working width of a machinery, while (ii) simultaneously floating at a projection distance equal to a specific working height above the terrain. The complexities of the algorithm in comparison to its 2D equivalent are highlighted. These include uniformly spaced elevation data generation using an Inverse Distance Weighting-approach and a local search. Area coverage path planning results for real-world 3D data within an agricultural context are presented to validate the algorithm.
>
---
#### [new 018] Optimal Transport-Based Decentralized Multi-Agent Distribution Matching
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体系统任务，解决分布式匹配问题。通过最优传输理论设计去中心化控制框架，实现智能体群体的终端分布匹配。**

- **链接: [https://arxiv.org/pdf/2601.00548v1](https://arxiv.org/pdf/2601.00548v1)**

> **作者:** Kooktae Lee
>
> **摘要:** This paper presents a decentralized control framework for distribution matching in multi-agent systems (MAS), where agents collectively achieve a prescribed terminal spatial distribution. The problem is formulated using optimal transport (Wasserstein distance), which provides a principled measure of distributional discrepancy and serves as the basis for the control design. To avoid solving the global optimal transport problem directly, the distribution-matching objective is reformulated into a tractable per-agent decision process, enabling each agent to identify its desired terminal locations using only locally available information. A sequential weight-update rule is introduced to construct feasible local transport plans, and a memory-based correction mechanism is incorporated to maintain reliable operation under intermittent and range-limited communication. Convergence guarantees are established, showing cycle-wise improvement of a surrogate transport cost under both linear and nonlinear agent dynamics. Simulation results demonstrate that the proposed framework achieves effective and scalable distribution matching while operating fully in a decentralized manner.
>
---
#### [new 019] RGS-SLAM: Robust Gaussian Splatting SLAM with One-Shot Dense Initialization
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出RGS-SLAM，解决SLAM中初始化不稳定问题，通过一阶段密集对应点三角化生成高斯种子，提升定位与重建精度，实现高效实时映射。**

- **链接: [https://arxiv.org/pdf/2601.00705v1](https://arxiv.org/pdf/2601.00705v1)**

> **作者:** Wei-Tse Cheng; Yen-Jen Chiou; Yuan-Fu Yang
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** We introduce RGS-SLAM, a robust Gaussian-splatting SLAM framework that replaces the residual-driven densification stage of GS-SLAM with a training-free correspondence-to-Gaussian initialization. Instead of progressively adding Gaussians as residuals reveal missing geometry, RGS-SLAM performs a one-shot triangulation of dense multi-view correspondences derived from DINOv3 descriptors refined through a confidence-aware inlier classifier, generating a well-distributed and structure-aware Gaussian seed prior to optimization. This initialization stabilizes early mapping and accelerates convergence by roughly 20\%, yielding higher rendering fidelity in texture-rich and cluttered scenes while remaining fully compatible with existing GS-SLAM pipelines. Evaluated on the TUM RGB-D and Replica datasets, RGS-SLAM achieves competitive or superior localization and reconstruction accuracy compared with state-of-the-art Gaussian and point-based SLAM systems, sustaining real-time mapping performance at up to 925 FPS.
>
---
#### [new 020] Bayesian Inverse Games with High-Dimensional Multi-Modal Observations
- **分类: cs.LG; cs.GT; cs.RO**

- **简介: 该论文属于逆博弈任务，解决多智能体目标未知的问题。通过贝叶斯方法，结合多模态观测数据，提升目标推断的准确性和安全性。**

- **链接: [https://arxiv.org/pdf/2601.00696v1](https://arxiv.org/pdf/2601.00696v1)**

> **作者:** Yash Jain; Xinjie Liu; Lasse Peters; David Fridovich-Keil; Ufuk Topcu
>
> **摘要:** Many multi-agent interaction scenarios can be naturally modeled as noncooperative games, where each agent's decisions depend on others' future actions. However, deploying game-theoretic planners for autonomous decision-making requires a specification of all agents' objectives. To circumvent this practical difficulty, recent work develops maximum likelihood techniques for solving inverse games that can identify unknown agent objectives from interaction data. Unfortunately, these methods only infer point estimates and do not quantify estimator uncertainty; correspondingly, downstream planning decisions can overconfidently commit to unsafe actions. We present an approximate Bayesian inference approach for solving the inverse game problem, which can incorporate observation data from multiple modalities and be used to generate samples from the Bayesian posterior over the hidden agent objectives given limited sensor observations in real time. Concretely, the proposed Bayesian inverse game framework trains a structured variational autoencoder with an embedded differentiable Nash game solver on interaction datasets and does not require labels of agents' true objectives. Extensive experiments show that our framework successfully learns prior and posterior distributions, improves inference quality over maximum likelihood estimation-based inverse game approaches, and enables safer downstream decision-making without sacrificing efficiency. When trajectory information is uninformative or unavailable, multimodal inference further reduces uncertainty by exploiting additional observation modalities.
>
---
#### [new 021] Application Research of a Deep Learning Model Integrating CycleGAN and YOLO in PCB Infrared Defect Detection
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于PCB缺陷检测任务，解决红外数据稀缺问题，通过融合CycleGAN与YOLOv8生成伪红外数据，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2601.00237v1](https://arxiv.org/pdf/2601.00237v1)**

> **作者:** Chao Yang; Haoyuan Zheng; Yue Ma
>
> **备注:** 8 pages,8 figures
>
> **摘要:** This paper addresses the critical bottleneck of infrared (IR) data scarcity in Printed Circuit Board (PCB) defect detection by proposing a cross-modal data augmentation framework integrating CycleGAN and YOLOv8. Unlike conventional methods relying on paired supervision, we leverage CycleGAN to perform unpaired image-to-image translation, mapping abundant visible-light PCB images into the infrared domain. This generative process synthesizes high-fidelity pseudo-IR samples that preserve the structural semantics of defects while accurately simulating thermal distribution patterns. Subsequently, we construct a heterogeneous training strategy that fuses generated pseudo-IR data with limited real IR samples to train a lightweight YOLOv8 detector. Experimental results demonstrate that this method effectively enhances feature learning under low-data conditions. The augmented detector significantly outperforms models trained on limited real data alone and approaches the performance benchmarks of fully supervised training, proving the efficacy of pseudo-IR synthesis as a robust augmentation strategy for industrial inspection.
>
---
#### [new 022] CropNeRF: A Neural Radiance Field-Based Framework for Crop Counting
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CropNeRF，用于作物计数任务。解决户外环境下因遮挡和形态相似导致的计数难题，通过3D实例分割和多视角图像实现精准计数。**

- **链接: [https://arxiv.org/pdf/2601.00207v1](https://arxiv.org/pdf/2601.00207v1)**

> **作者:** Md Ahmed Al Muzaddid; William J. Beksi
>
> **备注:** 8 pages, 10 figures, and 2 tables
>
> **摘要:** Rigorous crop counting is crucial for effective agricultural management and informed intervention strategies. However, in outdoor field environments, partial occlusions combined with inherent ambiguity in distinguishing clustered crops from individual viewpoints poses an immense challenge for image-based segmentation methods. To address these problems, we introduce a novel crop counting framework designed for exact enumeration via 3D instance segmentation. Our approach utilizes 2D images captured from multiple viewpoints and associates independent instance masks for neural radiance field (NeRF) view synthesis. We introduce crop visibility and mask consistency scores, which are incorporated alongside 3D information from a NeRF model. This results in an effective segmentation of crop instances in 3D and highly-accurate crop counts. Furthermore, our method eliminates the dependence on crop-specific parameter tuning. We validate our framework on three agricultural datasets consisting of cotton bolls, apples, and pears, and demonstrate consistent counting performance despite major variations in crop color, shape, and size. A comparative analysis against the state of the art highlights superior performance on crop counting tasks. Lastly, we contribute a cotton plant dataset to advance further research on this topic.
>
---
#### [new 023] Efficient Prediction of Dense Visual Embeddings via Distillation and RGB-D Transformers
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉嵌入任务，旨在提升机器人对环境的理解能力。通过知识蒸馏和RGB-D Transformer，提出DVEFormer模型，实现高效、细粒度的视觉嵌入，支持文本查询和3D建图。**

- **链接: [https://arxiv.org/pdf/2601.00359v1](https://arxiv.org/pdf/2601.00359v1)**

> **作者:** Söhnke Benedikt Fischedick; Daniel Seichter; Benedict Stephan; Robin Schmidt; Horst-Michael Gross
>
> **备注:** Published in Proc. IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** In domestic environments, robots require a comprehensive understanding of their surroundings to interact effectively and intuitively with untrained humans. In this paper, we propose DVEFormer - an efficient RGB-D Transformer-based approach that predicts dense text-aligned visual embeddings (DVE) via knowledge distillation. Instead of directly performing classical semantic segmentation with fixed predefined classes, our method uses teacher embeddings from Alpha-CLIP to guide our efficient student model DVEFormer in learning fine-grained pixel-wise embeddings. While this approach still enables classical semantic segmentation, e.g., via linear probing, it further enables flexible text-based querying and other applications, such as creating comprehensive 3D maps. Evaluations on common indoor datasets demonstrate that our approach achieves competitive performance while meeting real-time requirements, operating at 26.3 FPS for the full model and 77.0 FPS for a smaller variant on an NVIDIA Jetson AGX Orin. Additionally, we show qualitative results that highlight the effectiveness and possible use cases in real-world applications. Overall, our method serves as a drop-in replacement for traditional segmentation approaches while enabling flexible natural-language querying and seamless integration into 3D mapping pipelines for mobile robotics.
>
---
#### [new 024] GRL-SNAM: Geometric Reinforcement Learning with Path Differential Hamiltonians for Simultaneous Navigation and Mapping in Unknown Environments
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出GRL-SNAM框架，解决未知环境中同时导航与建图问题。通过局部感知和哈密顿优化实现高效路径搜索，无需全局地图。**

- **链接: [https://arxiv.org/pdf/2601.00116v1](https://arxiv.org/pdf/2601.00116v1)**

> **作者:** Aditya Sai Ellendula; Yi Wang; Minh Nguyen; Chandrajit Bajaj
>
> **摘要:** We present GRL-SNAM, a geometric reinforcement learning framework for Simultaneous Navigation and Mapping(SNAM) in unknown environments. A SNAM problem is challenging as it needs to design hierarchical or joint policies of multiple agents that control the movement of a real-life robot towards the goal in mapless environment, i.e. an environment where the map of the environment is not available apriori, and needs to be acquired through sensors. The sensors are invoked from the path learner, i.e. navigator, through active query responses to sensory agents, and along the motion path. GRL-SNAM differs from preemptive navigation algorithms and other reinforcement learning methods by relying exclusively on local sensory observations without constructing a global map. Our approach formulates path navigation and mapping as a dynamic shortest path search and discovery process using controlled Hamiltonian optimization: sensory inputs are translated into local energy landscapes that encode reachability, obstacle barriers, and deformation constraints, while policies for sensing, planning, and reconfiguration evolve stagewise via updating Hamiltonians. A reduced Hamiltonian serves as an adaptive score function, updating kinetic/potential terms, embedding barrier constraints, and continuously refining trajectories as new local information arrives. We evaluate GRL-SNAM on two different 2D navigation tasks. Comparing against local reactive baselines and global policy learning references under identical stagewise sensing constraints, it preserves clearance, generalizes to unseen layouts, and demonstrates that Geometric RL learning via updating Hamiltonians enables high-quality navigation through minimal exploration via local energy refinement rather than extensive global mapping. The code is publicly available on \href{https://github.com/CVC-Lab/GRL-SNAM}{Github}.
>
---
## 更新

#### [replaced 001] Video-Based Detection and Analysis of Errors in Robotic Surgical Training
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人训练分析任务，旨在解决干实验中手术错误自动检测难题。通过图像处理算法检测错误，分析学习过程中的表现变化。**

- **链接: [https://arxiv.org/pdf/2504.19571v3](https://arxiv.org/pdf/2504.19571v3)**

> **作者:** Hanna Kossowsky Lev; Yarden Sharon; Alex Geftler; Ilana Nisky
>
> **备注:** Title change; 9 pages, 4 figures, 1 table. Alex Geftler and Ilana Nisky contributed equally to this work
>
> **摘要:** Robot-assisted minimally invasive surgeries offer many advantages but require complex motor tasks that take surgeons years to master. There is currently a lack of knowledge on how surgeons acquire these robotic surgical skills. Toward bridging this gap, a previous study followed surgical residents learning complex surgical dry lab tasks on a surgical robot over six months. Errors are an important measure for training and skill evaluation, but unlike in virtual simulations, in dry lab training, errors are difficult to monitor automatically. Here, we analyzed errors in the ring tower transfer task, in which surgical residents moved a ring along a curved wire as quickly and accurately as possible. We developed an image-processing algorithm using color and size thresholds, optical flow and short time Fourier transforms to detect collision errors and achieved a detection accuracy of approximately 95%. Using the detected errors and task completion time, we found that the residents reduced their completion time and number of errors over the six months, while the percentage of task time spent making errors remained relatively constant on average. This analysis sheds light on the learning process of the residents and can serve as a step towards providing error-related feedback to robotic surgeons.
>
---
#### [replaced 002] World In Your Hands: A Large-Scale and Open-source Ecosystem for Learning Human-centric Manipulation in the Wild
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决手部操作数据不足的问题。提出WiYH生态系统，包含数据集、标注和基准，提升策略泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.24310v2](https://arxiv.org/pdf/2512.24310v2)**

> **作者:** TARS Robotics; Yupeng Zheng; Jichao Peng; Weize Li; Yuhang Zheng; Xiang Li; Yujie Jin; Julong Wei; Guanhua Zhang; Ruiling Zheng; Ming Cao; Songen Gu; Zhenhong Zou; Kaige Li; Ke Wu; Mingmin Yang; Jiahao Liu; Pengfei Li; Hengjie Si; Feiyu Zhu; Wang Fu; Likun Wang; Ruiwen Yao; Jieru Zhao; Yilun Chen; Wenchao Ding
>
> **备注:** This dataset represents the first large-scale collection of real-world, human-centric multimodal data integrating vision, language, tactile sensing, and action (VLTA)
>
> **摘要:** Large-scale pre-training is fundamental for generalization in language and vision models, but data for dexterous hand manipulation remains limited in scale and diversity, hindering policy generalization. Limited scenario diversity, misaligned modalities, and insufficient benchmarking constrain current human manipulation datasets. To address these gaps, we introduce World In Your Hands (WiYH), a large-scale open-source ecosystem for human-centric manipulation learning. WiYH includes (1) the Oracle Suite, a wearable data collection kit with an auto-labeling pipeline for accurate motion capture; (2) the WiYH Dataset, featuring over 1,000 hours of multi-modal manipulation data across hundreds of skills in diverse real-world scenarios; and (3) extensive annotations and benchmarks supporting tasks from perception to action. Furthermore, experiments based on the WiYH ecosystem show that integrating WiYH's human-centric data significantly enhances the generalization and robustness of dexterous hand policies in tabletop manipulation tasks. We believe that World In Your Hands will bring new insights into human-centric data collection and policy learning to the community.
>
---
#### [replaced 003] Flattening Hierarchies with Policy Bootstrapping
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习领域，解决长时序目标导向任务中的策略训练问题。通过引入非层级策略，利用子目标策略进行自举，提升算法在高维状态空间中的扩展性与性能。**

- **链接: [https://arxiv.org/pdf/2505.14975v3](https://arxiv.org/pdf/2505.14975v3)**

> **作者:** John L. Zhou; Jonathan C. Kao
>
> **备注:** NeurIPS 2025 (Spotlight, top 3.2%)
>
> **摘要:** Offline goal-conditioned reinforcement learning (GCRL) is a promising approach for pretraining generalist policies on large datasets of reward-free trajectories, akin to the self-supervised objectives used to train foundation models for computer vision and natural language processing. However, scaling GCRL to longer horizons remains challenging due to the combination of sparse rewards and discounting, which obscures the comparative advantages of primitive actions with respect to distant goals. Hierarchical RL methods achieve strong empirical results on long-horizon goal-reaching tasks, but their reliance on modular, timescale-specific policies and subgoal generation introduces significant additional complexity and hinders scaling to high-dimensional goal spaces. In this work, we introduce an algorithm to train a flat (non-hierarchical) goal-conditioned policy by bootstrapping on subgoal-conditioned policies with advantage-weighted importance sampling. Our approach eliminates the need for a generative model over the (sub)goal space, which we find is key for scaling to high-dimensional control in large state spaces. We further show that existing hierarchical and bootstrapping-based approaches correspond to specific design choices within our derivation. Across a comprehensive suite of state- and pixel-based locomotion and manipulation benchmarks, our method matches or surpasses state-of-the-art offline GCRL algorithms and scales to complex, long-horizon tasks where prior approaches fail. Project page: https://johnlyzhou.github.io/saw/
>
---
#### [replaced 004] PLK-Calib: Single-shot and Target-less LiDAR-Camera Extrinsic Calibration using Plücker Lines
- **分类: cs.RO**

- **简介: 该论文属于LiDAR-相机外参标定任务，解决无目标、单次拍摄的标定问题。通过线特征构建约束，提出两种算法提升标定精度。**

- **链接: [https://arxiv.org/pdf/2503.07955v2](https://arxiv.org/pdf/2503.07955v2)**

> **作者:** Yanyu Zhang; Jie Xu; Wei Ren
>
> **摘要:** Accurate LiDAR-Camera (LC) calibration is challenging but crucial for autonomous systems and robotics. In this paper, we propose two single-shot and target-less algorithms to estimate the calibration parameters between LiDAR and camera using line features. The first algorithm constructs line-to-line constraints by defining points-to-line projection errors and minimizes the projection error. The second algorithm (PLK-Calib) utilizes the co-perpendicular and co-parallel geometric properties of lines in Plücker (PLK) coordinate, and decouples the rotation and translation into two constraints, enabling more accurate estimates. Our degenerate analysis and Monte Carlo simulation indicate that three nonparallel line pairs are the minimal requirements to estimate the extrinsic parameters. Furthermore, we collect an LC calibration dataset with varying extrinsic under three different scenarios and use it to evaluate the performance of our proposed algorithms.
>
---
#### [replaced 005] Tackling the Kidnapped Robot Problem via Sparse Feasible Hypothesis Sampling and Reliable Batched Multi-Stage Inference
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决 kidnapped robot problem（KRP），通过稀疏采样和多阶段推理实现高效全局重定位。**

- **链接: [https://arxiv.org/pdf/2511.01219v3](https://arxiv.org/pdf/2511.01219v3)**

> **作者:** Muhua Zhang; Lei Ma; Ying Wu; Kai Shen; Deqing Huang; Henry Leung
>
> **备注:** 10 pages, 8 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper addresses the Kidnapped Robot Problem (KRP), a core localization challenge of relocalizing a robot in a known map without prior pose estimate when localization loss or at SLAM initialization. For this purpose, a passive 2-D global relocalization framework is proposed. It estimates the global pose efficiently and reliably from a single LiDAR scan and an occupancy grid map while the robot remains stationary, thereby enhancing the long-term autonomy of mobile robots. The proposed framework casts global relocalization as a non-convex problem and solves it via the multi-hypothesis scheme with batched multi-stage inference and early termination, balancing completeness and efficiency. The Rapidly-exploring Random Tree (RRT), under traversability constraints, asymptotically covers the reachable space to generate sparse, uniformly distributed feasible positional hypotheses, fundamentally reducing the sampling space. The hypotheses are preliminarily ordered by the proposed Scan Mean Absolute Difference (SMAD), a coarse beam-error level metric that facilitates the early termination by prioritizing high-likelihood candidates. The SMAD computation is optimized for non-panoramic scans. The Translation-Affinity Scan-to-Map Alignment Metric (TAM) is proposed for reliable orientation selection at hypothesized positions and accurate final pose evaluation to mitigate degradation in conventional likelihood-field metrics under translational uncertainty induced by sparse hypotheses, as well as non-panoramic LiDAR scan and environmental changes. Real-world experiments on a resource-constrained mobile robot with non-panoramic LiDAR scans show that the proposed framework achieves competitive performance in both global relocalization success rate and computational efficiency.
>
---
#### [replaced 006] AutoTrust: Benchmarking Trustworthiness in Large Vision Language Models for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶领域，旨在评估大视觉语言模型的可信度。通过构建数据集并测试多个模型，揭示其在安全、隐私和公平性方面的漏洞，提出改进方向。**

- **链接: [https://arxiv.org/pdf/2412.15206v2](https://arxiv.org/pdf/2412.15206v2)**

> **作者:** Shuo Xing; Hongyuan Hua; Xiangbo Gao; Shenzhe Zhu; Renjie Li; Kexin Tian; Xiaopeng Li; Heng Huang; Tianbao Yang; Zhangyang Wang; Yang Zhou; Huaxiu Yao; Zhengzhong Tu
>
> **备注:** Published at TMLR 2025
>
> **摘要:** Recent advancements in large vision language models (VLMs) tailored for autonomous driving (AD) have shown strong scene understanding and reasoning capabilities, making them undeniable candidates for end-to-end driving systems. However, limited work exists on studying the trustworthiness of DriveVLMs -- a critical factor that directly impacts public transportation safety. In this paper, we introduce AutoTrust, a comprehensive trustworthiness benchmark for large vision-language models in autonomous driving (DriveVLMs), considering diverse perspectives -- including trustfulness, safety, robustness, privacy, and fairness. We constructed the largest visual question-answering dataset for investigating trustworthiness issues in driving scenarios, comprising over 10k unique scenes and 18k queries. We evaluated six publicly available VLMs, spanning from generalist to specialist, from open-source to commercial models. Our exhaustive evaluations have unveiled previously undiscovered vulnerabilities of DriveVLMs to trustworthiness threats. Specifically, we found that the general VLMs like LLaVA-v1.6 and GPT-4o-mini surprisingly outperform specialized models fine-tuned for driving in terms of overall trustworthiness. DriveVLMs like DriveLM-Agent are particularly vulnerable to disclosing sensitive information. Additionally, both generalist and specialist VLMs remain susceptible to adversarial attacks and struggle to ensure unbiased decision-making across diverse environments and populations. Our findings call for immediate and decisive action to address the trustworthiness of DriveVLMs -- an issue of critical importance to public safety and the welfare of all citizens relying on autonomous transportation systems. We release all the codes and datasets in https://github.com/taco-group/AutoTrust.
>
---
#### [replaced 007] Hybrid A* Path Planning with Multi-Modal Motion Extension for Four-Wheel Steering Mobile Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于路径规划任务，解决4WIS机器人在复杂环境中的多模式运动规划问题。提出扩展的Hybrid A*框架，融合多种运动模式，提升规划灵活性与适应性。**

- **链接: [https://arxiv.org/pdf/2509.06115v3](https://arxiv.org/pdf/2509.06115v3)**

> **作者:** Runjiao Bao; Lin Zhang; Tianwei Niu; Haoyu Yuan; Shoukun Wang
>
> **备注:** Updated method details, parameters, and experimental scenarios
>
> **摘要:** Four-wheel independent steering (4WIS) systems provide mobile robots with a rich set of motion modes, such as Ackermann steering, lateral steering, and parallel movement, offering superior maneuverability in constrained environments. However, existing path planning methods generally assume a single kinematic model and thus fail to fully exploit the multi-modal capabilities of 4WIS platforms. To address this limitation, we propose an extended Hybrid A* framework that operates in a four-dimensional state space incorporating both spatial states and motion modes. Within this framework, we design multi-modal Reeds-Shepp curves tailored to the distinct kinematic constraints of each motion mode, develop an enhanced heuristic function that accounts for mode-switching costs, and introduce a terminal connection strategy with intelligent mode selection to ensure smooth transitions between different steering patterns. The proposed planner enables seamless integration of multiple motion modalities within a single path, significantly improving flexibility and adaptability in complex environments. Results demonstrate significantly improved planning performance for 4WIS robots in complex environments.
>
---
#### [replaced 008] Iterative Tuning of Nonlinear Model Predictive Control for Robotic Manufacturing Tasks
- **分类: cs.RO; cs.LG; eess.SY; math.OC**

- **简介: 该论文属于机器人制造任务，解决重复操作中控制参数需频繁调整的问题。通过迭代学习优化NMPC权重矩阵，提升跟踪精度与控制效率。**

- **链接: [https://arxiv.org/pdf/2512.13170v2](https://arxiv.org/pdf/2512.13170v2)**

> **作者:** Deepak Ingole; Valentin Bhend; Shiva Ganesh Murali; Oliver Dobrich; Alisa Rupenyan
>
> **摘要:** Manufacturing processes are often perturbed by drifts in the environment and wear in the system, requiring control re-tuning even in the presence of repetitive operations. This paper presents an iterative learning framework for automatic tuning of Nonlinear Model Predictive Control (NMPC) weighting matrices based on task-level performance feedback. Inspired by norm-optimal Iterative Learning Control (ILC), the proposed method adaptively adjusts NMPC weights Q and R across task repetitions to minimize key performance indicators (KPIs) related to tracking accuracy, control effort, and saturation. Unlike gradient-based approaches that require differentiating through the NMPC solver, we construct an empirical sensitivity matrix, enabling structured weight updates without analytic derivatives. The framework is validated through simulation on a UR10e robot performing carbon fiber winding on a tetrahedral core. Results demonstrate that the proposed approach converges to near-optimal tracking performance (RMSE within 0.3% of offline Bayesian Optimization (BO)) in just 4 online repetitions, compared to 100 offline evaluations required by BO algorithm. The method offers a practical solution for adaptive NMPC tuning in repetitive robotic tasks, combining the precision of carefully optimized controllers with the flexibility of online adaptation.
>
---
#### [replaced 009] NeRF-VIO: Map-Based Visual-Inertial Odometry with Initialization Leveraging Neural Radiance Fields
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出NeRF-VIO，用于增强现实中的视觉惯性定位。通过结合NeRF和MSCKF，解决初始化精度与漂移问题，提升定位性能。**

- **链接: [https://arxiv.org/pdf/2503.07952v2](https://arxiv.org/pdf/2503.07952v2)**

> **作者:** Yanyu Zhang; Dongming Wang; Jie Xu; Mengyuan Liu; Pengxiang Zhu; Wei Ren
>
> **摘要:** A prior map serves as a foundational reference for localization in context-aware applications such as augmented reality (AR). Providing valuable contextual information about the environment, the prior map is a vital tool for mitigating drift. In this paper, we propose a map-based visual-inertial localization algorithm (NeRF-VIO) with initialization using neural radiance fields (NeRF). Our algorithm utilizes a multilayer perceptron model and redefines the loss function as the geodesic distance on \(SE(3)\), ensuring the invariance of the initialization model under a frame change within \(\mathfrak{se}(3)\). The evaluation demonstrates that our model outperforms existing NeRF-based initialization solution in both accuracy and efficiency. By integrating a two-stage update mechanism within a multi-state constraint Kalman filter (MSCKF) framework, the state of NeRF-VIO is constrained by both captured images from an onboard camera and rendered images from a pre-trained NeRF model. The proposed algorithm is validated using a real-world AR dataset, the results indicate that our two-stage update pipeline outperforms MSCKF across all data sequences.
>
---
#### [replaced 010] Dynamic Gap: Safe Gap-based Navigation in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于导航任务，解决动态环境中安全路径规划问题。提出动态间隙方法，通过分析自由空间变化实现可靠避障。**

- **链接: [https://arxiv.org/pdf/2210.05022v3](https://arxiv.org/pdf/2210.05022v3)**

> **作者:** Max Asselmeier; Dhruv Ahuja; Abdel Zaro; Ahmad Abuaish; Ye Zhao; Patricio A. Vela
>
> **备注:** Accepted to 2025 International Conference on Robotics and Automation (ICRA)
>
> **摘要:** This paper extends the family of gap-based local planners to unknown dynamic environments through generating provable collision-free properties for hierarchical navigation systems. Existing perception-informed local planners that operate in dynamic environments rely on emergent or empirical robustness for collision avoidance as opposed to performing formal analysis of dynamic obstacles. In addition to this, the obstacle tracking that is performed in these existent planners is often achieved with respect to a global inertial frame, subjecting such tracking estimates to transformation errors from odometry drift. The proposed local planner, dynamic gap, shifts the tracking paradigm to modeling how the free space, represented as gaps, evolves over time. Gap crossing and closing conditions are developed to aid in determining the feasibility of passage through gaps, and a breadth of simulation benchmarking is performed against other navigation planners in the literature where the proposed dynamic gap planner achieves the highest success rate out of all planners tested in all environments.
>
---
#### [replaced 011] MDE-AgriVLN: Agricultural Vision-and-Language Navigation with Monocular Depth Estimation
- **分类: cs.RO**

- **简介: 该论文属于农业视觉语言导航任务，旨在解决单目视觉导致的空间感知不足问题。通过引入深度估计模块，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.03958v3](https://arxiv.org/pdf/2512.03958v3)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xin Chen; Xiang Li
>
> **摘要:** Agricultural robots are serving as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily relying on manual operations or railway systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extended Vision-and-Language Navigation (VLN) to the agricultural domain, enabling a robot to navigate to a target position following a natural language instruction. Unlike human binocular vision, most agricultural robots are only given a single camera for monocular vision, which results in limited spatial perception. To bridge this gap, we present the method of Agricultural Vision-and-Language Navigation with Monocular Depth Estimation (MDE-AgriVLN), in which we propose the MDE module generating depth features from RGB images, to assist the decision-maker on multimodal reasoning. When evaluated on the A2A benchmark, our MDE-AgriVLN method successfully increases Success Rate from 0.23 to 0.32 and decreases Navigation Error from 4.43m to 4.08m, demonstrating the state-of-the-art performance in the agricultural VLN domain. Code: https://github.com/AlexTraveling/MDE-AgriVLN.
>
---
#### [replaced 012] Digital Twin based Automatic Reconfiguration of Robotic Systems in Smart Environments
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于机器人自主控制任务，旨在解决动态环境中机器人适应性不足的问题。通过数字孪生技术实现机器人控制器的自动重构，提升其在变化环境中的可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2511.00094v2](https://arxiv.org/pdf/2511.00094v2)**

> **作者:** Angelos Alexopoulos; Agorakis Bompotas; Nikitas Rigas Kalogeropoulos; Panagiotis Kechagias; Athanasios P. Kalogeras; Christos Alexakos
>
> **备注:** Accepted for presentation to 11th IEEE International Smart Cities Conference (ISC2 2025)
>
> **摘要:** Robotic systems have become integral to smart environments, enabling applications ranging from urban surveillance and automated agriculture to industrial automation. However, their effective operation in dynamic settings - such as smart cities and precision farming - is challenged by continuously evolving topographies and environmental conditions. Traditional control systems often struggle to adapt quickly, leading to inefficiencies or operational failures. To address this limitation, we propose a novel framework for autonomous and dynamic reconfiguration of robotic controllers using Digital Twin technology. Our approach leverages a virtual replica of the robot's operational environment to simulate and optimize movement trajectories in response to real-world changes. By recalculating paths and control parameters in the Digital Twin and deploying the updated code to the physical robot, our method ensures rapid and reliable adaptation without manual intervention. This work advances the integration of Digital Twins in robotics, offering a scalable solution for enhancing autonomy in smart, dynamic environments.
>
---
#### [replaced 013] Symphony: A Heuristic Normalized Calibrated Advantage Actor and Critic Algorithm in application for Humanoid Robots
- **分类: cs.RO; cs.NE**

- **简介: 该论文属于强化学习任务，旨在解决人形机器人训练中的样本效率、安全性和稳定性问题。提出Symphony算法，结合多种技术提升训练安全性与效果。**

- **链接: [https://arxiv.org/pdf/2512.10477v5](https://arxiv.org/pdf/2512.10477v5)**

> **作者:** Timur Ishuov; Michele Folgheraiter; Madi Nurmanov; Goncalo Gordo; Richárd Farkas; József Dombi
>
> **备注:** https://github.com/SuspensionRailway/symphony
>
> **摘要:** In our work we not explicitly hint that it is a misconception to think that humans learn fast. Learning process takes time. Babies start learning to move in the restricted liquid area called placenta. Children often are limited by underdeveloped body. Even adults are not allowed to participate in complex competitions right away. However, with robots, when learning from scratch, we often don't have the privilege of waiting for dozen millions of steps. "Swaddling" regularization is responsible for restraining an agent in rapid but unstable development penalizing action strength in a specific way not affecting actions directly. The Symphony, Transitional-policy Deterministic Actor and Critic algorithm, is a concise combination of different ideas for possibility of training humanoid robots from scratch with Sample Efficiency, Sample Proximity and Safety of Actions in mind. It is no secret that continuous increase in Gaussian noise without appropriate smoothing is harmful for motors and gearboxes. Compared to Stochastic algorithms, we set a limited parametric noise and promote a reduced strength of actions, safely increasing entropy, since the actions are kind of immersed in weaker noise. When actions require more extreme values, actions rise above the weak noise. Training becomes empirically much safer for both the environment around and the robot's mechanisms. We use Fading Replay Buffer: using a fixed formula containing the hyperbolic tangent, we adjust the batch sampling probability: the memory contains a recent memory and a long-term memory trail. Fading Replay Buffer allows us to use Temporal Advantage when we improve the current Critic Network prediction compared to the exponential moving average. Temporal Advantage allows us to update Actor and Critic in one pass, as well as combine Actor and Critic in one Object and implement their Losses in one line.
>
---
#### [replaced 014] Unified Embodied VLM Reasoning with Robotic Action via Autoregressive Discretized Pre-training
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉语言导航任务，旨在解决VLA模型在开放环境中泛化与精确执行的矛盾。提出ERIQ基准和FACT方法，提升机器人推理与动作的协同能力。**

- **链接: [https://arxiv.org/pdf/2512.24125v2](https://arxiv.org/pdf/2512.24125v2)**

> **作者:** Yi Liu; Sukai Wang; Dafeng Wei; Xiaowei Cai; Linqing Zhong; Jiange Yang; Guanghui Ren; Jinyu Zhang; Maoqing Yao; Chuankang Li; Xindong He; Liliang Chen; Jianlan Luo
>
> **摘要:** General-purpose robotic systems operating in open-world environments must achieve both broad generalization and high-precision action execution, a combination that remains challenging for existing Vision-Language-Action (VLA) models. While large Vision-Language Models (VLMs) improve semantic generalization, insufficient embodied reasoning leads to brittle behavior, and conversely, strong reasoning alone is inadequate without precise control. To provide a decoupled and quantitative assessment of this bottleneck, we introduce Embodied Reasoning Intelligence Quotient (ERIQ), a large-scale embodied reasoning benchmark in robotic manipulation, comprising 6K+ question-answer pairs across four reasoning dimensions. By decoupling reasoning from execution, ERIQ enables systematic evaluation and reveals a strong positive correlation between embodied reasoning capability and end-to-end VLA generalization. To bridge the gap from reasoning to precise execution, we propose FACT, a flow-matching-based action tokenizer that converts continuous control into discrete sequences while preserving high-fidelity trajectory reconstruction. The resulting GenieReasoner jointly optimizes reasoning and action in a unified space, outperforming both continuous-action and prior discrete-action baselines in real-world tasks. Together, ERIQ and FACT provide a principled framework for diagnosing and overcoming the reasoning-precision trade-off, advancing robust, general-purpose robotic manipulation. Project page: https://geniereasoner.github.io/GenieReasoner/
>
---
#### [replaced 015] Efficient Multi-Task Scene Analysis with RGB-D Transformers
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EMSARFormer，用于高效多任务场景分析，解决移动平台计算受限下的场景理解问题。通过RGB-D Transformer编码器实现多任务同时处理，并优化推理速度。**

- **链接: [https://arxiv.org/pdf/2306.05242v2](https://arxiv.org/pdf/2306.05242v2)**

> **作者:** Söhnke Benedikt Fischedick; Daniel Seichter; Robin Schmidt; Leonard Rabes; Horst-Michael Gross
>
> **备注:** Published in Proc. International Joint Conference on Neural Networks (IJCNN 2023)
>
> **摘要:** Scene analysis is essential for enabling autonomous systems, such as mobile robots, to operate in real-world environments. However, obtaining a comprehensive understanding of the scene requires solving multiple tasks, such as panoptic segmentation, instance orientation estimation, and scene classification. Solving these tasks given limited computing and battery capabilities on mobile platforms is challenging. To address this challenge, we introduce an efficient multi-task scene analysis approach, called EMSAFormer, that uses an RGB-D Transformer-based encoder to simultaneously perform the aforementioned tasks. Our approach builds upon the previously published EMSANet. However, we show that the dual CNN-based encoder of EMSANet can be replaced with a single Transformer-based encoder. To achieve this, we investigate how information from both RGB and depth data can be effectively incorporated in a single encoder. To accelerate inference on robotic hardware, we provide a custom NVIDIA TensorRT extension enabling highly optimization for our EMSAFormer approach. Through extensive experiments on the commonly used indoor datasets NYUv2, SUNRGB-D, and ScanNet, we show that our approach achieves state-of-the-art performance while still enabling inference with up to 39.1 FPS on an NVIDIA Jetson AGX Orin 32 GB.
>
---
