# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Uncertainty-Aware Ankle Exoskeleton Control
- **分类: cs.RO**

- **简介: 该论文设计不确定性感知控制框架，解决外骨骼在多样化场景下的控制问题，通过不确定性估计器自动断开非预期动作，提升安全性和自主性。**

- **链接: [http://arxiv.org/pdf/2508.21221v1](http://arxiv.org/pdf/2508.21221v1)**

> **作者:** Fatima Mumtaza Tourk; Bishoy Galoaa; Sanat Shajan; Aaron J. Young; Michael Everett; Max K. Shepherd
>
> **摘要:** Lower limb exoskeletons show promise to assist human movement, but their utility is limited by controllers designed for discrete, predefined actions in controlled environments, restricting their real-world applicability. We present an uncertainty-aware control framework that enables ankle exoskeletons to operate safely across diverse scenarios by automatically disengaging when encountering unfamiliar movements. Our approach uses an uncertainty estimator to classify movements as similar (in-distribution) or different (out-of-distribution) relative to actions in the training set. We evaluated three architectures (model ensembles, autoencoders, and generative adversarial networks) on an offline dataset and tested the strongest performing architecture (ensemble of gait phase estimators) online. The online test demonstrated the ability of our uncertainty estimator to turn assistance on and off as the user transitioned between in-distribution and out-of-distribution tasks (F1: 89.2). This new framework provides a path for exoskeletons to safely and autonomously support human movement in unstructured, everyday environments.
>
---
#### [new 002] QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 论文提出QuadKAN框架，用于四足视觉运动控制，结合本体感觉与视觉输入，通过KAN和样条参数化策略提升样本效率与稳定性，采用MMDR+PPO训练，在复杂地形中实现更优运动表现。**

- **链接: [http://arxiv.org/pdf/2508.19153v1](http://arxiv.org/pdf/2508.19153v1)**

> **作者:** Allen Wang; Gavin Tao
>
> **备注:** 14pages, 9 figures, Journal paper
>
> **摘要:** We address vision-guided quadruped motion control with reinforcement learning (RL) and highlight the necessity of combining proprioception with vision for robust control. We propose QuadKAN, a spline-parameterized cross-modal policy instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates a spline encoder for proprioception and a spline fusion head for proprioception-vision inputs. This structured function class aligns the state-to-action mapping with the piecewise-smooth nature of gait, improving sample efficiency, reducing action jitter and energy consumption, and providing interpretable posture-action sensitivities. We adopt Multi-Modal Delay Randomization (MMDR) and perform end-to-end training with Proximal Policy Optimization (PPO). Evaluations across diverse terrains, including both even and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate that QuadKAN achieves consistently higher returns, greater distances, and fewer collisions than state-of-the-art (SOTA) baselines. These results show that spline-parameterized policies offer a simple, effective, and interpretable alternative for robust vision-guided locomotion. A repository will be made available upon acceptance.
>
---
#### [new 003] Multi-Modal Model Predictive Path Integral Control for Collision Avoidance
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出多模态MPPIC算法，解决自动驾驶车辆避障与稳定控制问题。通过Sobol采样与解析解结合，探索多样化轨迹模式，优化转向与加速参数，确保碰撞规避并提升复杂路况下的行驶稳定性。**

- **链接: [http://arxiv.org/pdf/2508.21364v1](http://arxiv.org/pdf/2508.21364v1)**

> **作者:** Alberto Bertipaglia; Dariu M. Gavrila; Barys Shyrokau
>
> **备注:** Accepted as an oral presentation at the 29th IAVSD. August 18-22, 2025. Shanghai, China
>
> **摘要:** This paper proposes a novel approach to motion planning and decision-making for automated vehicles, using a multi-modal Model Predictive Path Integral control algorithm. The method samples with Sobol sequences around the prior input and incorporates analytical solutions for collision avoidance. By leveraging multiple modes, the multi-modal control algorithm explores diverse trajectories, such as manoeuvring around obstacles or stopping safely before them, mitigating the risk of sub-optimal solutions. A non-linear single-track vehicle model with a Fiala tyre serves as the prediction model, and tyre force constraints within the friction circle are enforced to ensure vehicle stability during evasive manoeuvres. The optimised steering angle and longitudinal acceleration are computed to generate a collision-free trajectory and to control the vehicle. In a high-fidelity simulation environment, we demonstrate that the proposed algorithm can successfully avoid obstacles, keeping the vehicle stable while driving a double lane change manoeuvre on high and low-friction road surfaces and occlusion scenarios with moving obstacles, outperforming a standard Model Predictive Path Integral approach.
>
---
#### [new 004] Dynamics-Compliant Trajectory Diffusion for Super-Nominal Payload Manipulation
- **分类: cs.RO**

- **简介: 该论文提出一种基于扩散模型的轨迹生成方法，解决传统方法对机器人负载限制的保守估计问题，通过整合动态约束提升超额载荷操作能力，实验验证在高负载下仍可保持广泛工作空间可用性。**

- **链接: [http://arxiv.org/pdf/2508.21375v1](http://arxiv.org/pdf/2508.21375v1)**

> **作者:** Anuj Pasricha; Joewie Koh; Jay Vakil; Alessandro Roncone
>
> **备注:** Accepted to 2025 Conference on Robot Learning [CoRL]
>
> **摘要:** Nominal payload ratings for articulated robots are typically derived from worst-case configurations, resulting in uniform payload constraints across the entire workspace. This conservative approach severely underutilizes the robot's inherent capabilities -- our analysis demonstrates that manipulators can safely handle payloads well above nominal capacity across broad regions of their workspace while staying within joint angle, velocity, acceleration, and torque limits. To address this gap between assumed and actual capability, we propose a novel trajectory generation approach using denoising diffusion models that explicitly incorporates payload constraints into the planning process. Unlike traditional sampling-based methods that rely on inefficient trial-and-error, optimization-based methods that are prohibitively slow, or kinodynamic planners that struggle with problem dimensionality, our approach generates dynamically feasible joint-space trajectories in constant time that can be directly executed on physical hardware without post-processing. Experimental validation on a 7 DoF Franka Emika Panda robot demonstrates that up to 67.6% of the workspace remains accessible even with payloads exceeding 3 times the nominal capacity. This expanded operational envelope highlights the importance of a more nuanced consideration of payload dynamics in motion planning algorithms.
>
---
#### [new 005] Few-Shot Neuro-Symbolic Imitation Learning for Long-Horizon Planning and Acting
- **分类: cs.RO**

- **简介: 该论文提出神经符号框架，通过符号抽象与神经控制结合，解决长周期任务中少样本模仿学习的挑战，实现高效数据利用和跨任务泛化。**

- **链接: [http://arxiv.org/pdf/2508.21501v1](http://arxiv.org/pdf/2508.21501v1)**

> **作者:** Pierrick Lorang; Hong Lu; Johannes Huemer; Patrik Zips; Matthias Scheutz
>
> **备注:** Accepted at CoRL 2025; to appear in PMLR
>
> **摘要:** Imitation learning enables intelligent systems to acquire complex behaviors with minimal supervision. However, existing methods often focus on short-horizon skills, require large datasets, and struggle to solve long-horizon tasks or generalize across task variations and distribution shifts. We propose a novel neuro-symbolic framework that jointly learns continuous control policies and symbolic domain abstractions from a few skill demonstrations. Our method abstracts high-level task structures into a graph, discovers symbolic rules via an Answer Set Programming solver, and trains low-level controllers using diffusion policy imitation learning. A high-level oracle filters task-relevant information to focus each controller on a minimal observation and action space. Our graph-based neuro-symbolic framework enables capturing complex state transitions, including non-spatial and temporal relations, that data-driven learning or clustering techniques often fail to discover in limited demonstration datasets. We validate our approach in six domains that involve four robotic arms, Stacking, Kitchen, Assembly, and Towers of Hanoi environments, and a distinct Automated Forklift domain with two environments. The results demonstrate high data efficiency with as few as five skill demonstrations, strong zero- and few-shot generalizations, and interpretable decision making.
>
---
#### [new 006] Estimated Informed Anytime Search for Sampling-Based Planning via Adaptive Sampler
- **分类: cs.RO**

- **简介: 该论文提出MIT*算法，解决高维机器人路径规划中传统方法效率低的问题。通过预估启发式集合和自适应采样器优化搜索，提升收敛速度与计算效率，适用于R^4至R^16空间及实际机器人任务。**

- **链接: [http://arxiv.org/pdf/2508.21549v1](http://arxiv.org/pdf/2508.21549v1)**

> **作者:** Liding Zhang; Kuanqi Cai; Yu Zhang; Zhenshan Bing; Chaoqun Wang; Fan Wu; Sami Haddadin; Alois Knoll
>
> **摘要:** Path planning in robotics often involves solving continuously valued, high-dimensional problems. Popular informed approaches include graph-based searches, such as A*, and sampling-based methods, such as Informed RRT*, which utilize informed set and anytime strategies to expedite path optimization incrementally. Informed sampling-based planners define informed sets as subsets of the problem domain based on the current best solution cost. However, when no solution is found, these planners re-sample and explore the entire configuration space, which is time-consuming and computationally expensive. This article introduces Multi-Informed Trees (MIT*), a novel planner that constructs estimated informed sets based on prior admissible solution costs before finding the initial solution, thereby accelerating the initial convergence rate. Moreover, MIT* employs an adaptive sampler that dynamically adjusts the sampling strategy based on the exploration process. Furthermore, MIT* utilizes length-related adaptive sparse collision checks to guide lazy reverse search. These features enhance path cost efficiency and computation times while ensuring high success rates in confined scenarios. Through a series of simulations and real-world experiments, it is confirmed that MIT* outperforms existing single-query, sampling-based planners for problems in R^4 to R^16 and has been successfully applied to real-world robot manipulation tasks. A video showcasing our experimental results is available at: https://youtu.be/30RsBIdexTU
>
---
#### [new 007] RoboInspector: Unveiling the Unreliability of Policy Code for LLM-enabled Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对LLM生成机器人策略代码的可靠性问题，设计RoboInspector框架，通过分析任务复杂度与指令粒度识别四类不可靠行为，并提出反馈优化方法，提升代码可靠性达35%。**

- **链接: [http://arxiv.org/pdf/2508.21378v1](http://arxiv.org/pdf/2508.21378v1)**

> **作者:** Chenduo Ying; Linkang Du; Peng Cheng; Yuanchao Shu
>
> **摘要:** Large language models (LLMs) demonstrate remarkable capabilities in reasoning and code generation, enabling robotic manipulation to be initiated with just a single instruction. The LLM carries out various tasks by generating policy code required to control the robot. Despite advances in LLMs, achieving reliable policy code generation remains a significant challenge due to the diverse requirements of real-world tasks and the inherent complexity of user instructions. In practice, different users may provide distinct instructions to drive the robot for the same task, which may cause the unreliability of policy code generation. To bridge this gap, we design RoboInspector, a pipeline to unveil and characterize the unreliability of the policy code for LLM-enabled robotic manipulation from two perspectives: the complexity of the manipulation task and the granularity of the instruction. We perform comprehensive experiments with 168 distinct combinations of tasks, instructions, and LLMs in two prominent frameworks. The RoboInspector identifies four main unreliable behaviors that lead to manipulation failure. We provide a detailed characterization of these behaviors and their underlying causes, giving insight for practical development to reduce unreliability. Furthermore, we introduce a refinement approach guided by failure policy code feedback that improves the reliability of policy code generation by up to 35% in LLM-enabled robotic manipulation, evaluated in both simulation and real-world environments.
>
---
#### [new 008] Multi-robot Path Planning and Scheduling via Model Predictive Optimal Transport (MPC-OT)
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出基于最优传输与模型预测控制的多机器人路径规划方法，解决机器人导航中因路径重叠导致的死锁问题，通过空间离散化与成本结构设计实现非重叠轨迹规划，并整合时间结构处理动态情况。**

- **链接: [http://arxiv.org/pdf/2508.21205v1](http://arxiv.org/pdf/2508.21205v1)**

> **作者:** Usman A. Khan; Mouhacine Benosman; Wenliang Liu; Federico Pecora; Joseph W. Durham
>
> **备注:** 2025 IEEE Conference on Decision and Control
>
> **摘要:** In this paper, we propose a novel methodology for path planning and scheduling for multi-robot navigation that is based on optimal transport theory and model predictive control. We consider a setup where $N$ robots are tasked to navigate to $M$ targets in a common space with obstacles. Mapping robots to targets first and then planning paths can result in overlapping paths that lead to deadlocks. We derive a strategy based on optimal transport that not only provides minimum cost paths from robots to targets but also guarantees non-overlapping trajectories. We achieve this by discretizing the space of interest into $K$ cells and by imposing a ${K\times K}$ cost structure that describes the cost of transitioning from one cell to another. Optimal transport then provides \textit{optimal and non-overlapping} cell transitions for the robots to reach the targets that can be readily deployed without any scheduling considerations. The proposed solution requires $\unicode{x1D4AA}(K^3\log K)$ computations in the worst-case and $\unicode{x1D4AA}(K^2\log K)$ for well-behaved problems. To further accommodate potentially overlapping trajectories (unavoidable in certain situations) as well as robot dynamics, we show that a temporal structure can be integrated into optimal transport with the help of \textit{replans} and \textit{model predictive control}.
>
---
#### [new 009] Learning to Assemble the Soma Cube with Legal-Action Masked DQN and Safe ZYZ Regrasp on a Doosan M0609
- **分类: cs.RO; stat.CO**

- **简介: 该论文提出基于合法动作掩码DQN和安全ZYZ重抓策略的Soma立方体组装方法，解决动作空间爆炸、不安全规划及策略学习问题，通过分层架构和课程学习实现高效组装，三难度级别成功率分别为100%、92.9%、39.9%。**

- **链接: [http://arxiv.org/pdf/2508.21272v1](http://arxiv.org/pdf/2508.21272v1)**

> **作者:** Jaehong Oh; Seungjun Jung; Sawoong Kim
>
> **备注:** 13 figures, 17 pages
>
> **摘要:** This paper presents the first comprehensive application of legal-action masked Deep Q-Networks with safe ZYZ regrasp strategies to an underactuated gripper-equipped 6-DOF collaborative robot for autonomous Soma cube assembly learning. Our approach represents the first systematic integration of constraint-aware reinforcement learning with singularity-safe motion planning on a Doosan M0609 collaborative robot. We address critical challenges in robotic manipulation: combinatorial action space explosion, unsafe motion planning, and systematic assembly strategy learning. Our system integrates a legal-action masked DQN with hierarchical architecture that decomposes Q-function estimation into orientation and position components, reducing computational complexity from $O(3,132)$ to $O(116) + O(27)$ while maintaining solution completeness. The robot-friendly reward function encourages ground-first, vertically accessible assembly sequences aligned with manipulation constraints. Curriculum learning across three progressive difficulty levels (2-piece, 3-piece, 7-piece) achieves remarkable training efficiency: 100\% success rate for Level 1 within 500 episodes, 92.9\% for Level 2, and 39.9\% for Level 3 over 105,300 total training episodes.
>
---
#### [new 010] Learning Agile Gate Traversal via Analytical Optimal Policy Gradient
- **分类: cs.RO**

- **简介: 该论文提出混合框架，结合模型预测控制（MPC）与神经网络，通过分析策略梯度优化，解决四旋翼穿越狭窄门的高效控制问题，显著提升样本效率。**

- **链接: [http://arxiv.org/pdf/2508.21592v1](http://arxiv.org/pdf/2508.21592v1)**

> **作者:** Tianchen Sun; Bingheng Wang; Longbin Tang; Yichao Gao; Lin Zhao
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Traversing narrow gates presents a significant challenge and has become a standard benchmark for evaluating agile and precise quadrotor flight. Traditional modularized autonomous flight stacks require extensive design and parameter tuning, while end-to-end reinforcement learning (RL) methods often suffer from low sample efficiency and limited interpretability. In this work, we present a novel hybrid framework that adaptively fine-tunes model predictive control (MPC) parameters online using outputs from a neural network (NN) trained offline. The NN jointly predicts a reference pose and cost-function weights, conditioned on the coordinates of the gate corners and the current drone state. To achieve efficient training, we derive analytical policy gradients not only for the MPC module but also for an optimization-based gate traversal detection module. Furthermore, we introduce a new formulation of the attitude tracking error that admits a simplified representation, facilitating effective learning with bounded gradients. Hardware experiments demonstrate that our method enables fast and accurate quadrotor traversal through narrow gates in confined environments. It achieves several orders of magnitude improvement in sample efficiency compared to naive end-to-end RL approaches.
>
---
#### [new 011] Robust Convex Model Predictive Control with collision avoidance guarantees for robot manipulators
- **分类: cs.RO**

- **简介: 该论文提出一种鲁棒凸模型预测控制方法，用于解决工业机械臂在复杂环境中的安全快速运动规划问题，通过结合管状MPC与走廊规划算法，在模型不确定性下实现碰撞避免与高效控制。**

- **链接: [http://arxiv.org/pdf/2508.21677v1](http://arxiv.org/pdf/2508.21677v1)**

> **作者:** Bernhard Wullt; Johannes Köhler; Per Mattsson; Mikeal Norrlöf; Thomas B. Schön
>
> **摘要:** Industrial manipulators are normally operated in cluttered environments, making safe motion planning important. Furthermore, the presence of model-uncertainties make safe motion planning more difficult. Therefore, in practice the speed is limited in order to reduce the effect of disturbances. There is a need for control methods that can guarantee safe motions that can be executed fast. We address this need by suggesting a novel model predictive control (MPC) solution for manipulators, where our two main components are a robust tube MPC and a corridor planning algorithm to obtain collision-free motion. Our solution results in a convex MPC, which we can solve fast, making our method practically useful. We demonstrate the efficacy of our method in a simulated environment with a 6 DOF industrial robot operating in cluttered environments with uncertainties in model parameters. We outperform benchmark methods, both in terms of being able to work under higher levels of model uncertainties, while also yielding faster motion.
>
---
#### [new 012] The Rosario Dataset v2: Multimodal Dataset for Agricultural Robotics
- **分类: cs.RO; cs.CV; cs.SY; eess.SY; I.2.9**

- **简介: 该论文提出农业机器人多模态数据集Rosario v2，解决自然光照、地形复杂等挑战，用于评估SLAM算法，发布数据集及工具。**

- **链接: [http://arxiv.org/pdf/2508.21635v1](http://arxiv.org/pdf/2508.21635v1)**

> **作者:** Nicolas Soncini; Javier Cremona; Erica Vidal; Maximiliano García; Gastón Castro; Taihú Pire
>
> **备注:** First published on The International Journal of Robotics Research: https://journals.sagepub.com/doi/10.1177/02783649251368909
>
> **摘要:** We present a multi-modal dataset collected in a soybean crop field, comprising over two hours of recorded data from sensors such as stereo infrared camera, color camera, accelerometer, gyroscope, magnetometer, GNSS (Single Point Positioning, Real-Time Kinematic and Post-Processed Kinematic), and wheel odometry. This dataset captures key challenges inherent to robotics in agricultural environments, including variations in natural lighting, motion blur, rough terrain, and long, perceptually aliased sequences. By addressing these complexities, the dataset aims to support the development and benchmarking of advanced algorithms for localization, mapping, perception, and navigation in agricultural robotics. The platform and data collection system is designed to meet the key requirements for evaluating multi-modal SLAM systems, including hardware synchronization of sensors, 6-DOF ground truth and loops on long trajectories. We run multimodal state-of-the art SLAM methods on the dataset, showcasing the existing limitations in their application on agricultural settings. The dataset and utilities to work with it are released on https://cifasis.github.io/rosariov2/.
>
---
#### [new 013] Observability-driven Assignment of Heterogeneous Sensors for Multi-Target Tracking
- **分类: cs.RO**

- **简介: 该论文解决多目标跟踪中异构传感器分配问题，提出基于matroid理论的贪心算法，通过优化机器人-目标匹配降低状态估计不确定性，保证近似比并验证算法有效性。**

- **链接: [http://arxiv.org/pdf/2508.21309v1](http://arxiv.org/pdf/2508.21309v1)**

> **作者:** Seyed Ali Rakhshan; Mehdi Golestani; He Kong
>
> **备注:** This paper has been accepted to the 2025 IEEE/RSJ IROS
>
> **摘要:** This paper addresses the challenge of assigning heterogeneous sensors (i.e., robots with varying sensing capabilities) for multi-target tracking. We classify robots into two categories: (1) sufficient sensing robots, equipped with range and bearing sensors, capable of independently tracking targets, and (2) limited sensing robots, which are equipped with only range or bearing sensors and need to at least form a pair to collaboratively track a target. Our objective is to optimize tracking quality by minimizing uncertainty in target state estimation through efficient robot-to-target assignment. By leveraging matroid theory, we propose a greedy assignment algorithm that dynamically allocates robots to targets to maximize tracking quality. The algorithm guarantees constant-factor approximation bounds of 1/3 for arbitrary tracking quality functions and 1/2 for submodular functions, while maintaining polynomial-time complexity. Extensive simulations demonstrate the algorithm's effectiveness in accurately estimating and tracking targets over extended periods. Furthermore, numerical results confirm that the algorithm's performance is close to that of the optimal assignment, highlighting its robustness and practical applicability.
>
---
#### [new 014] Robust Real-Time Coordination of CAVs: A Distributed Optimization Framework under Uncertainty
- **分类: cs.RO**

- **简介: 该论文提出一种分布式优化框架，解决CAVs在动态不确定环境下的安全实时协同问题。通过鲁棒轨迹规划、ADMM-DTN算法和交互注意力机制，提升协调安全性与效率，实验验证其在碰撞率降低和计算需求减少方面的优势。**

- **链接: [http://arxiv.org/pdf/2508.21322v1](http://arxiv.org/pdf/2508.21322v1)**

> **作者:** Haojie Bai; Yang Wang; Cong Guo; Xiongwei Zhao; Hai Zhu
>
> **摘要:** Achieving both safety guarantees and real-time performance in cooperative vehicle coordination remains a fundamental challenge, particularly in dynamic and uncertain environments. This paper presents a novel coordination framework that resolves this challenge through three key innovations: 1) direct control of vehicles' trajectory distributions during coordination, formulated as a robust cooperative planning problem with adaptive enhanced safety constraints, ensuring a specified level of safety regarding the uncertainty of the interactive trajectory, 2) a fully parallel ADMM-based distributed trajectory negotiation (ADMM-DTN) algorithm that efficiently solves the optimization problem while allowing configurable negotiation rounds to balance solution quality and computational resources, and 3) an interactive attention mechanism that selectively focuses on critical interactive participants to further enhance computational efficiency. Both simulation results and practical experiments demonstrate that our framework achieves significant advantages in safety (reducing collision rates by up to 40.79\% in various scenarios) and real-time performance compared to state-of-the-art methods, while maintaining strong scalability with increasing vehicle numbers. The proposed interactive attention mechanism further reduces the computational demand by 14.1\%. The framework's effectiveness is further validated through real-world experiments with unexpected dynamic obstacles, demonstrating robust coordination in complex environments. The experiment demo could be found at https://youtu.be/4PZwBnCsb6Q.
>
---
#### [new 015] EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出EO-1模型与EO-Data1.5M数据集，解决通用机器人控制中多模态交错推理与交互不足的问题。通过统一架构与交错视觉-文本-动作预训练，实现灵活的机器人控制与多模态理解。**

- **链接: [http://arxiv.org/pdf/2508.21112v1](http://arxiv.org/pdf/2508.21112v1)**

> **作者:** Delin Qu; Haoming Song; Qizhi Chen; Zhaoqing Chen; Xianqiang Gao; Xinyi Ye; Qi Lv; Modi Shi; Guanghui Ren; Cheng Ruan; Maoqing Yao; Haoran Yang; Jiacheng Bao; Bin Zhao; Dong Wang
>
> **摘要:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models.
>
---
#### [new 016] Mini Autonomous Car Driving based on 3D Convolutional Neural Networks
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对微型自动驾驶汽车控制任务，提出基于RGB-D数据与3D卷积神经网络的方法，解决传统模型训练复杂、效率低的问题，通过模拟环境对比实验验证其在任务完成率与驾驶一致性上的优越性。**

- **链接: [http://arxiv.org/pdf/2508.21271v1](http://arxiv.org/pdf/2508.21271v1)**

> **作者:** Pablo Moraes; Monica Rodriguez; Kristofer S. Kappel; Hiago Sodre; Santiago Fernandez; Igor Nunes; Bruna Guterres; Ricardo Grando
>
> **摘要:** Autonomous driving applications have become increasingly relevant in the automotive industry due to their potential to enhance vehicle safety, efficiency, and user experience, thereby meeting the growing demand for sophisticated driving assistance features. However, the development of reliable and trustworthy autonomous systems poses challenges such as high complexity, prolonged training periods, and intrinsic levels of uncertainty. Mini Autonomous Cars (MACs) are used as a practical testbed, enabling validation of autonomous control methodologies on small-scale setups. This simplified and cost-effective environment facilitates rapid evaluation and comparison of machine learning models, which is particularly useful for algorithms requiring online training. To address these challenges, this work presents a methodology based on RGB-D information and three-dimensional convolutional neural networks (3D CNNs) for MAC autonomous driving in simulated environments. We evaluate the proposed approach against recurrent neural networks (RNNs), with architectures trained and tested on two simulated tracks with distinct environmental features. Performance was assessed using task completion success, lap-time metrics, and driving consistency. Results highlight how architectural modifications and track complexity influence the models' generalization capability and vehicle control performance. The proposed 3D CNN demonstrated promising results when compared with RNNs.
>
---
#### [new 017] Assessing Human Cooperation for Enhancing Social Robot Navigation
- **分类: cs.RO**

- **简介: 该论文研究社会机器人导航中的合作性评估，解决人类行为不可预测导致的导航困难。提出评估方法与指标，利用几何分析生成沟通策略，提升机器人与人类协作能力。**

- **链接: [http://arxiv.org/pdf/2508.21455v1](http://arxiv.org/pdf/2508.21455v1)**

> **作者:** Hariharan Arunachalam; Phani Teja Singamaneni; Rachid Alami
>
> **摘要:** Socially aware robot navigation is a planning paradigm where the robot navigates in human environments and tries to adhere to social constraints while interacting with the humans in the scene. These navigation strategies were further improved using human prediction models, where the robot takes the potential future trajectory of humans while computing its own. Though these strategies significantly improve the robot's behavior, it faces difficulties from time to time when the human behaves in an unexpected manner. This happens as the robot fails to understand human intentions and cooperativeness, and the human does not have a clear idea of what the robot is planning to do. In this paper, we aim to address this gap through effective communication at an appropriate time based on a geometric analysis of the context and human cooperativeness in head-on crossing scenarios. We provide an assessment methodology and propose some evaluation metrics that could distinguish a cooperative human from a non-cooperative one. Further, we also show how geometric reasoning can be used to generate appropriate verbal responses or robot actions.
>
---
#### [new 018] Can a mobile robot learn from a pedestrian model to prevent the sidewalk salsa?
- **分类: cs.RO**

- **简介: 该论文研究机器人通过强化学习学习行人行为模型以避免人行道上的“sidewalk salsa”现象。利用Communication-Enabled Interaction框架建模行人交互，设计RL代理学习有效沟通策略，降低碰撞风险，提升机器人与行人的安全互动。**

- **链接: [http://arxiv.org/pdf/2508.21690v1](http://arxiv.org/pdf/2508.21690v1)**

> **作者:** Olger Siebinga; David Abbink
>
> **摘要:** Pedestrians approaching each other on a sidewalk sometimes end up in an awkward interaction known as the "sidewalk salsa": they both (repeatedly) deviate to the same side to avoid a collision. This provides an interesting use case to study interactions between pedestrians and mobile robots because, in the vast majority of cases, this phenomenon is avoided through a negotiation based on implicit communication. Understanding how it goes wrong and how pedestrians end up in the sidewalk salsa will therefore provide insight into the implicit communication. This understanding can be used to design safe and acceptable robotic behaviour. In a previous attempt to gain this understanding, a model of pedestrian behaviour based on the Communication-Enabled Interaction (CEI) framework was developed that can replicate the sidewalk salsa. However, it is unclear how to leverage this model in robotic planning and decision-making since it violates the assumptions of game theory, a much-used framework in planning and decision-making. Here, we present a proof-of-concept for an approach where a Reinforcement Learning (RL) agent leverages the model to learn how to interact with pedestrians. The results show that a basic RL agent successfully learned to interact with the CEI model. Furthermore, a risk-averse RL agent that had access to the perceived risk of the CEI model learned how to effectively communicate its intention through its motion and thereby substantially lowered the perceived risk, and displayed effort by the modelled pedestrian. These results show this is a promising approach and encourage further exploration.
>
---
#### [new 019] Observer Design for Optical Flow-Based Visual-Inertial Odometry with Almost-Global Convergence
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出级联观测器架构，融合光流与IMU数据，通过Riccati观测器和梯度下降算法实现视觉惯性里程计的几乎全局渐近稳定估计。**

- **链接: [http://arxiv.org/pdf/2508.21163v1](http://arxiv.org/pdf/2508.21163v1)**

> **作者:** Tarek Bouazza; Soulaimane Berkane; Minh-Duc Hua; Tarek Hamel
>
> **备注:** 8 pages, 6 figures. To appear in IEEE CDC 2025
>
> **摘要:** This paper presents a novel cascaded observer architecture that combines optical flow and IMU measurements to perform continuous monocular visual-inertial odometry (VIO). The proposed solution estimates body-frame velocity and gravity direction simultaneously by fusing velocity direction information from optical flow measurements with gyro and accelerometer data. This fusion is achieved using a globally exponentially stable Riccati observer, which operates under persistently exciting translational motion conditions. The estimated gravity direction in the body frame is then employed, along with an optional magnetometer measurement, to design a complementary observer on $\mathbf{SO}(3)$ for attitude estimation. The resulting interconnected observer architecture is shown to be almost globally asymptotically stable. To extract the velocity direction from sparse optical flow data, a gradient descent algorithm is developed to solve a constrained minimization problem on the unit sphere. The effectiveness of the proposed algorithms is validated through simulation results.
>
---
#### [new 020] Remarks on stochastic cloning and delayed-state filtering
- **分类: cs.RO; eess.SP; math.ST; stat.TH**

- **简介: 该论文针对机器人导航中的延迟状态测量问题，比较随机克隆与延迟卡尔曼滤波器，证明后者无需状态增强即可实现相同估计效果，提升计算效率和内存使用。**

- **链接: [http://arxiv.org/pdf/2508.21260v1](http://arxiv.org/pdf/2508.21260v1)**

> **作者:** Tara Mina; Lindsey Marinello; John Christian
>
> **摘要:** Many estimation problems in robotics and navigation involve measurements that depend on prior states. A prominent example is odometry, which measures the relative change between states over time. Accurately handling these delayed-state measurements requires capturing their correlations with prior state estimates, and a widely used approach is stochastic cloning (SC), which augments the state vector to account for these correlations. This work revisits a long-established but often overlooked alternative--the delayed-state Kalman filter--and demonstrates that a properly derived filter yields exactly the same state and covariance update as SC, without requiring state augmentation. Moreover, the generalized Kalman filter formulation provides computational advantages, while also reducing memory requirements for higher-dimensional states. Our findings clarify a common misconception that Kalman filter variants are inherently unable to handle correlated delayed-state measurements, demonstrating that an alternative formulation achieves the same results more efficiently.
>
---
#### [new 021] Tree-Guided Diffusion Planner
- **分类: cs.AI; cs.RO**

- **简介: 论文提出Tree-Guided Diffusion Planner（TDP），解决测试时引导控制中非凸奖励、多目标等挑战。通过树搜索与扩散模型结合，利用预训练模型和测试奖励信号，实现零样本规划，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.21800v1](http://arxiv.org/pdf/2508.21800v1)**

> **作者:** Hyeonseong Jeon; Cheolhong Min; Jaesik Park
>
> **备注:** 20 pages, 11 figures, 14 tables (main paper + appendix) / under review / project page will be available after the paper becomes public in arxiv
>
> **摘要:** Planning with pretrained diffusion models has emerged as a promising approach for solving test-time guided control problems. However, standard gradient guidance typically performs optimally under convex and differentiable reward landscapes, showing substantially reduced effectiveness in real-world scenarios involving non-convex objectives, non-differentiable constraints, and multi-reward structures. Furthermore, recent supervised planning approaches require task-specific training or value estimators, which limits test-time flexibility and zero-shot generalization. We propose a Tree-guided Diffusion Planner (TDP), a zero-shot test-time planning framework that balances exploration and exploitation through structured trajectory generation. We frame test-time planning as a tree search problem using a bi-level sampling process: (1) diverse parent trajectories are produced via training-free particle guidance to encourage broad exploration, and (2) sub-trajectories are refined through fast conditional denoising guided by task objectives. TDP addresses the limitations of gradient guidance by exploring diverse trajectory regions and harnessing gradient information across this expanded solution space using only pretrained models and test-time reward signals. We evaluate TDP on three diverse tasks: maze gold-picking, robot arm block manipulation, and AntMaze multi-goal exploration. TDP consistently outperforms state-of-the-art approaches on all tasks. The project page can be found at: tree-diffusion-planner.github.io.
>
---
#### [new 022] 2COOOL: 2nd Workshop on the Challenge Of Out-Of-Label Hazards in Autonomous Driving
- **分类: cs.CV; cs.RO; 68T45 (Machine vision and scene understanding); I.2.10; I.4.8**

- **简介: 该研讨会聚焦自动驾驶中标签外危险问题，旨在解决新场景导致的安全隐患。通过推动分布外检测、视觉-语言模型等技术，促进安全算法与基准测试研究，提升自动驾驶可靠性。**

- **链接: [http://arxiv.org/pdf/2508.21080v1](http://arxiv.org/pdf/2508.21080v1)**

> **作者:** Ali K. AlShami; Ryan Rabinowitz; Maged Shoman; Jianwu Fang; Lukas Picek; Shao-Yuan Lo; Steve Cruz; Khang Nhut Lam; Nachiket Kamod; Lei-Lei Li; Jugal Kalita; Terrance E. Boult
>
> **备注:** 11 pages, 2 figures, Accepted to ICCV 2025 Workshop on Out-of-Label Hazards in Autonomous Driving (2COOOL)
>
> **摘要:** As the computer vision community advances autonomous driving algorithms, integrating vision-based insights with sensor data remains essential for improving perception, decision making, planning, prediction, simulation, and control. Yet we must ask: Why don't we have entirely safe self-driving cars yet? A key part of the answer lies in addressing novel scenarios, one of the most critical barriers to real-world deployment. Our 2COOOL workshop provides a dedicated forum for researchers and industry experts to push the state of the art in novelty handling, including out-of-distribution hazard detection, vision-language models for hazard understanding, new benchmarking and methodologies, and safe autonomous driving practices. The 2nd Workshop on the Challenge of Out-of-Label Hazards in Autonomous Driving (2COOOL) will be held at the International Conference on Computer Vision (ICCV) 2025 in Honolulu, Hawaii, on October 19, 2025. We aim to inspire the development of new algorithms and systems for hazard avoidance, drawing on ideas from anomaly detection, open-set recognition, open-vocabulary modeling, domain adaptation, and related fields. Building on the success of its inaugural edition at the Winter Conference on Applications of Computer Vision (WACV) 2025, the workshop will feature a mix of academic and industry participation.
>
---
#### [new 023] Complete Gaussian Splats from a Single Image with Denoising Diffusion Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出基于扩散模型的单图像3D场景重建方法，解决遮挡区域重建问题，通过生成式模型和自监督学习实现完整高斯点云生成。**

- **链接: [http://arxiv.org/pdf/2508.21542v1](http://arxiv.org/pdf/2508.21542v1)**

> **作者:** Ziwei Liao; Mohamed Sayed; Steven L. Waslander; Sara Vicente; Daniyar Turmukhambetov; Michael Firman
>
> **备注:** Main paper: 11 pages; Supplementary materials: 7 pages
>
> **摘要:** Gaussian splatting typically requires dense observations of the scene and can fail to reconstruct occluded and unobserved areas. We propose a latent diffusion model to reconstruct a complete 3D scene with Gaussian splats, including the occluded parts, from only a single image during inference. Completing the unobserved surfaces of a scene is challenging due to the ambiguity of the plausible surfaces. Conventional methods use a regression-based formulation to predict a single "mode" for occluded and out-of-frustum surfaces, leading to blurriness, implausibility, and failure to capture multiple possible explanations. Thus, they often address this problem partially, focusing either on objects isolated from the background, reconstructing only visible surfaces, or failing to extrapolate far from the input views. In contrast, we propose a generative formulation to learn a distribution of 3D representations of Gaussian splats conditioned on a single input image. To address the lack of ground-truth training data, we propose a Variational AutoReconstructor to learn a latent space only from 2D images in a self-supervised manner, over which a diffusion model is trained. Our method generates faithful reconstructions and diverse samples with the ability to complete the occluded surfaces for high-quality 360-degree renderings.
>
---
#### [new 024] A-MHA*: Anytime Multi-Heuristic A*
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出A-MHA*算法，解决传统MHA*无法随时间优化解的问题。通过融合ARA*的任何时间特性，使算法在路径规划和拼图任务中持续改进子优解，保持原有理论保证并提升效率。**

- **链接: [http://arxiv.org/pdf/2508.21637v1](http://arxiv.org/pdf/2508.21637v1)**

> **作者:** Ramkumar Natarajan; Muhammad Suhail Saleem; William Xiao; Sandip Aine; Howie Choset; Maxim Likhachev
>
> **摘要:** Designing good heuristic functions for graph search requires adequate domain knowledge. It is often easy to design heuristics that perform well and correlate with the underlying true cost-to-go values in certain parts of the search space but these may not be admissible throughout the domain thereby affecting the optimality guarantees of the search. Bounded suboptimal search using several such partially good but inadmissible heuristics was developed in Multi-Heuristic A* (MHA*). Although MHA* leverages multiple inadmissible heuristics to potentially generate a faster suboptimal solution, the original version does not improve the solution over time. It is a one shot algorithm that requires careful setting of inflation factors to obtain a desired one time solution. In this work, we tackle this issue by extending MHA* to an anytime version that finds a feasible suboptimal solution quickly and continually improves it until time runs out. Our work is inspired from the Anytime Repairing A* (ARA*) algorithm. We prove that our precise adaptation of ARA* concepts in the MHA* framework preserves the original suboptimal and completeness guarantees and enhances MHA* to perform in an anytime fashion. Furthermore, we report the performance of A-MHA* in 3-D path planning domain and sliding tiles puzzle and compare against MHA* and other anytime algorithms.
>
---
#### [new 025] GENNAV: Polygon Mask Generation for Generalized Referring Navigable Regions
- **分类: cs.CV; cs.RO**

- **简介: 论文提出GENNAV方法，解决基于自然语言和图像识别模糊边界目标区域的问题，通过预测存在性并生成分割掩码，构建GRiN-Drive基准并验证其在真实环境中的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.21102v1](http://arxiv.org/pdf/2508.21102v1)**

> **作者:** Kei Katsumata; Yui Iioka; Naoki Hosomi; Teruhisa Misu; Kentaro Yamada; Komei Sugiura
>
> **备注:** Accepted for presentation at CoRL2025
>
> **摘要:** We focus on the task of identifying the location of target regions from a natural language instruction and a front camera image captured by a mobility. This task is challenging because it requires both existence prediction and segmentation, particularly for stuff-type target regions with ambiguous boundaries. Existing methods often underperform in handling stuff-type target regions, in addition to absent or multiple targets. To overcome these limitations, we propose GENNAV, which predicts target existence and generates segmentation masks for multiple stuff-type target regions. To evaluate GENNAV, we constructed a novel benchmark called GRiN-Drive, which includes three distinct types of samples: no-target, single-target, and multi-target. GENNAV achieved superior performance over baseline methods on standard evaluation metrics. Furthermore, we conducted real-world experiments with four automobiles operated in five geographically distinct urban areas to validate its zero-shot transfer performance. In these experiments, GENNAV outperformed baseline methods and demonstrated its robustness across diverse real-world environments. The project page is available at https://gennav.vercel.app/.
>
---
#### [new 026] Detecting Domain Shifts in Myoelectric Activations: Challenges and Opportunities in Stream Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对肌电信号领域转移检测任务，解决非平稳信号下的实时检测难题。通过流学习框架，利用KPCA预处理DB6数据集，评估CUSUM等方法，揭示现有技术局限，并探索流式方法在维持EMG解码稳定性的潜力。**

- **链接: [http://arxiv.org/pdf/2508.21278v1](http://arxiv.org/pdf/2508.21278v1)**

> **作者:** Yibin Sun; Nick Lim; Guilherme Weigert Cassales; Heitor Murilo Gomes; Bernhard Pfahringer; Albert Bifet; Anany Dwivedi
>
> **备注:** 16 pages, 5 figures, 1 table, PRICAI25
>
> **摘要:** Detecting domain shifts in myoelectric activations poses a significant challenge due to the inherent non-stationarity of electromyography (EMG) signals. This paper explores the detection of domain shifts using data stream (DS) learning techniques, focusing on the DB6 dataset from the Ninapro database. We define domains as distinct time-series segments based on different subjects and recording sessions, applying Kernel Principal Component Analysis (KPCA) with a cosine kernel to pre-process and highlight these shifts. By evaluating multiple drift detection methods such as CUSUM, Page-Hinckley, and ADWIN, we reveal the limitations of current techniques in achieving high performance for real-time domain shift detection in EMG signals. Our results underscore the potential of streaming-based approaches for maintaining stable EMG decoding models, while highlighting areas for further research to enhance robustness and accuracy in real-world scenarios.
>
---
#### [new 027] Cooperative Sensing Enhanced UAV Path-Following and Obstacle Avoidance with Variable Formation
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对无人机路径跟踪与避障任务，解决多子任务高效协同问题。提出基于DRL的路径模型、ISAC感知与VFEO算法，设计在线避障方案及分层融合策略，实现高精度路径跟随与动态障碍物避让。**

- **链接: [http://arxiv.org/pdf/2508.21316v1](http://arxiv.org/pdf/2508.21316v1)**

> **作者:** Changheng Wang; Zhiqing Wei; Wangjun Jiang; Haoyue Jiang; Zhiyong Feng
>
> **摘要:** The high mobility of unmanned aerial vehicles (UAVs) enables them to be used in various civilian fields, such as rescue and cargo transport. Path-following is a crucial way to perform these tasks while sensing and collision avoidance are essential for safe flight. In this paper, we investigate how to efficiently and accurately achieve path-following, obstacle sensing and avoidance subtasks, as well as their conflict-free fusion scheduling. Firstly, a high precision deep reinforcement learning (DRL)-based UAV formation path-following model is developed, and the reward function with adaptive weights is designed from the perspective of distance and velocity errors. Then, we use integrated sensing and communication (ISAC) signals to detect the obstacle and derive the Cramer-Rao lower bound (CRLB) for obstacle sensing by information-level fusion, based on which we propose the variable formation enhanced obstacle position estimation (VFEO) algorithm. In addition, an online obstacle avoidance scheme without pretraining is designed to solve the sparse reward. Finally, with the aid of null space based (NSB) behavioral method, we present a hierarchical subtasks fusion strategy. Simulation results demonstrate the effectiveness and superiority of the subtask algorithms and the hierarchical fusion strategy.
>
---
## 更新

#### [replaced 001] Knowledge in multi-robot systems: an interplay of dynamics, computation and communication
- **分类: cs.LO; cs.DC; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.18309v2](http://arxiv.org/pdf/2501.18309v2)**

> **作者:** Giorgio Cignarale; Stephan Felber; Eric Goubault; Bernardo Hummes Flores; Hugo Rincon Galeana
>
> **摘要:** In this paper, we provide a framework integrating distributed multi-robot systems and temporal epistemic logic. We show that continuous-discrete hybrid systems are compatible with logical models of knowledge already used in distributed computing, and demonstrate its usefulness by deriving sufficient epistemic conditions for exploration and gathering robot tasks to be solvable. We provide a separation of the physical and computational aspects of a robotic system, allowing us to decouple the problems related to each and directly use methods from control theory and distributed computing, fields that are traditionally distant in the literature. Finally, we demonstrate a novel approach for reasoning about the knowledge in multi-robot systems through a principled method of converting a switched hybrid dynamical system into a temporal-epistemic logic model, passing through an abstract state machine representation. This creates space for methods and results to be exchanged across the fields of control theory, distributed computing and temporal-epistemic logic, while reasoning about multi-robot systems.
>
---
#### [replaced 002] COBRA-PPM: A Causal Bayesian Reasoning Architecture Using Probabilistic Programming for Robot Manipulation Under Uncertainty
- **分类: cs.RO; cs.AI; cs.LG; stat.AP; I.2.9; I.2.8; I.2.3; G.3; I.2.6; I.6.8; I.2.4; I.2.10**

- **链接: [http://arxiv.org/pdf/2403.14488v4](http://arxiv.org/pdf/2403.14488v4)**

> **作者:** Ricardo Cannizzaro; Michael Groom; Jonathan Routley; Robert Osazuwa Ness; Lars Kunze
>
> **备注:** 8 pages, 7 figures, accepted to the 2025 IEEE European Conference on Mobile Robots (ECMR 2025)
>
> **摘要:** Manipulation tasks require robots to reason about cause and effect when interacting with objects. Yet, many data-driven approaches lack causal semantics and thus only consider correlations. We introduce COBRA-PPM, a novel causal Bayesian reasoning architecture that combines causal Bayesian networks and probabilistic programming to perform interventional inference for robot manipulation under uncertainty. We demonstrate its capabilities through high-fidelity Gazebo-based experiments on an exemplar block stacking task, where it predicts manipulation outcomes with high accuracy (Pred Acc: 88.6%) and performs greedy next-best action selection with a 94.2% task success rate. We further demonstrate sim2real transfer on a domestic robot, showing effectiveness in handling real-world uncertainty from sensor noise and stochastic actions. Our generalised and extensible framework supports a wide range of manipulation scenarios and lays a foundation for future work at the intersection of robotics and causality.
>
---
#### [replaced 003] Towards Embodiment Scaling Laws in Robot Locomotion
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05753v2](http://arxiv.org/pdf/2505.05753v2)**

> **作者:** Bo Ai; Liu Dai; Nico Bohlinger; Dichen Li; Tongzhou Mu; Zhanxin Wu; K. Fay; Henrik I. Christensen; Jan Peters; Hao Su
>
> **备注:** Conference on Robot Learning (CoRL), 2025. Project website: https://embodiment-scaling-laws.github.io/
>
> **摘要:** Cross-embodiment generalization underpins the vision of building generalist embodied agents for any robot, yet its enabling factors remain poorly understood. We investigate embodiment scaling laws, the hypothesis that increasing the number of training embodiments improves generalization to unseen ones, using robot locomotion as a test bed. We procedurally generate ~1,000 embodiments with topological, geometric, and joint-level kinematic variations, and train policies on random subsets. We observe positive scaling trends supporting the hypothesis, and find that embodiment scaling enables substantially broader generalization than data scaling on fixed embodiments. Our best policy, trained on the full dataset, transfers zero-shot to novel embodiments in simulation and the real world, including the Unitree Go2 and H1. These results represent a step toward general embodied intelligence, with relevance to adaptive control for configurable robots, morphology co-design, and beyond.
>
---
#### [replaced 004] Ontology Neural Network and ORTSF: A Framework for Topological Reasoning and Delay-Robust Control
- **分类: cs.RO; cs.SY; eess.SY; 68T40, 93C41; I.2.9; I.2.8; F.2.2**

- **链接: [http://arxiv.org/pdf/2506.19277v2](http://arxiv.org/pdf/2506.19277v2)**

> **作者:** Jaehong Oh
>
> **备注:** 12 pages, 5 figures, includes theoretical proofs and simulation results
>
> **摘要:** The advancement of autonomous robotic systems has led to impressive capabilities in perception, localization, mapping, and control. Yet, a fundamental gap remains: existing frameworks excel at geometric reasoning and dynamic stability but fall short in representing and preserving relational semantics, contextual reasoning, and cognitive transparency essential for collaboration in dynamic, human-centric environments. This paper introduces a unified architecture comprising the Ontology Neural Network (ONN) and the Ontological Real-Time Semantic Fabric (ORTSF) to address this gap. The ONN formalizes relational semantic reasoning as a dynamic topological process. By embedding Forman-Ricci curvature, persistent homology, and semantic tensor structures within a unified loss formulation, ONN ensures that relational integrity and topological coherence are preserved as scenes evolve over time. The ORTSF transforms reasoning traces into actionable control commands while compensating for system delays. It integrates predictive and delay-aware operators that ensure phase margin preservation and continuity of control signals, even under significant latency conditions. Empirical studies demonstrate the ONN + ORTSF framework's ability to unify semantic cognition and robust control, providing a mathematically principled and practically viable solution for cognitive robotics.
>
---
#### [replaced 005] CoRI: Communication of Robot Intent for Physical Human-Robot Interaction
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20537v2](http://arxiv.org/pdf/2505.20537v2)**

> **作者:** Junxiang Wang; Emek Barış Küçüktabak; Rana Soltani Zarrin; Zackory Erickson
>
> **备注:** To be published in Proceedings of the 9th Conference on Robot Learning (CoRL). 34 pages, 10 figures
>
> **摘要:** Clear communication of robot intent fosters transparency and interpretability in physical human-robot interaction (pHRI), particularly during assistive tasks involving direct human-robot contact. We introduce CoRI, a pipeline that automatically generates natural language communication of a robot's upcoming actions directly from its motion plan and visual perception. Our pipeline first processes the robot's image view to identify human poses and key environmental features. It then encodes the planned 3D spatial trajectory (including velocity and force) onto this view, visually grounding the path and its dynamics. CoRI queries a vision-language model with this visual representation to interpret the planned action within the visual context before generating concise, user-directed statements, without relying on task-specific information. Results from a user study involving robot-assisted feeding, bathing, and shaving tasks across two different robots indicate that CoRI leads to statistically significant difference in communication clarity compared to a baseline communication strategy. Specifically, CoRI effectively conveys not only the robot's high-level intentions but also crucial details about its motion and any collaborative user action needed. Video and code of our project can be found on our project website: https://cori-phri.github.io/
>
---
#### [replaced 006] LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.11849v2](http://arxiv.org/pdf/2508.11849v2)**

> **作者:** Yinuo Wang; Gavin Tao
>
> **备注:** 13 pages
>
> **摘要:** We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget.
>
---
#### [replaced 007] Multi-critic Learning for Whole-body End-effector Twist Tracking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.08656v2](http://arxiv.org/pdf/2507.08656v2)**

> **作者:** Aravind Elanjimattathil Vijayan; Andrei Cramariuc; Mattia Risiglione; Christian Gehring; Marco Hutter
>
> **摘要:** Learning whole-body control for locomotion and arm motions in a single policy has challenges, as the two tasks have conflicting goals. For instance, efficient locomotion typically favors a horizontal base orientation, while end-effector tracking may benefit from base tilting to extend reachability. Additionally, current Reinforcement Learning (RL) approaches using a pose-based task specification lack the ability to directly control the end-effector velocity, making smoothly executing trajectories very challenging. To address these limitations, we propose an RL-based framework that allows for dynamic, velocity-aware whole-body end-effector control. Our method introduces a multi-critic actor architecture that decouples the reward signals for locomotion and manipulation, simplifying reward tuning and allowing the policy to resolve task conflicts more effectively. Furthermore, we design a twist-based end-effector task formulation that can track both discrete poses and motion trajectories. We validate our approach through a set of simulation and hardware experiments using a quadruped robot equipped with a robotic arm. The resulting controller can simultaneously walk and move its end-effector and shows emergent whole-body behaviors, where the base assists the arm in extending the workspace, despite a lack of explicit formulations. Videos and supplementary material can be found at multi-critic-locomanipulation.github.io.
>
---
#### [replaced 008] Traversing the Narrow Path: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20661v2](http://arxiv.org/pdf/2508.20661v2)**

> **作者:** TianChen Huang; Wei Gao; Runchen Xu; Shiwu Zhang
>
> **备注:** Project website: https://huangtc233.github.io/Traversing-the-Narrow-Path/
>
> **摘要:** Traversing narrow beams is challenging for humanoids due to sparse, safety-critical contacts and the fragility of purely learned policies. We propose a physically grounded, two-stage framework that couples an XCoM/LIPM footstep template with a lightweight residual planner and a simple low-level tracker. Stage-1 is trained on flat ground: the tracker learns to robustly follow footstep targets by adding small random perturbations to heuristic footsteps, without any hand-crafted centerline locking, so it acquires stable contact scheduling and strong target-tracking robustness. Stage-2 is trained in simulation on a beam: a high-level planner predicts a body-frame residual (Delta x, Delta y, Delta psi) for the swing foot only, refining the template step to prioritize safe, precise placement under narrow support while preserving interpretability. To ease deployment, sensing is kept minimal and consistent between simulation and hardware: the planner consumes compact, forward-facing elevation cues together with onboard IMU and joint signals. On a Unitree G1, our system reliably traverses a 0.2 m-wide, 3 m-long beam. Across simulation and real-world studies, residual refinement consistently outperforms template-only and monolithic baselines in success rate, centerline adherence, and safety margins, while the structured footstep interface enables transparent analysis and low-friction sim-to-real transfer.
>
---
#### [replaced 009] Pellet-based 3D Printing of Soft Thermoplastic Elastomeric Membranes for Soft Robotic Applications
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.20957v2](http://arxiv.org/pdf/2503.20957v2)**

> **作者:** Nick Willemstein; Mohammad Ebrahim Imanian; Herman van der Kooij; Ali Sadeghi
>
> **摘要:** Additive Manufacturing (AM) is a promising solution for handling the complexity of fabricating soft robots. However, the AM of hyperelastic materials is still challenging with a limited material range. Within this work, pellet-based 3D printing of very soft thermoplastic elastomers (TPEs) was explored (down to Shore Hardness 00-30). Our results show that TPEs can have similar engineering stress and maximum elongation as Ecoflex OO-10. In addition, we 3D-printed airtight thin membranes (0.2-1.2 mm), which could inflate up to a stretch of 1320%. Combining the membrane's large expansion and softness with the 3D printing of hollow structures simplified the design of a bending actuator that can bend 180 degrees and reach a blocked force of 238 times its weight. In addition, by 3D printing TPE pellets and rigid filaments, the soft membrane could grasp objects by enveloping an object or as a sensorized sucker, which relied on the TPE's softness to conform to the object or act as a seal. In addition, the membrane of the sucker acted as a tactile sensor to detect an object before adhesion. These results suggest the feasibility of AM of soft robots using soft TPEs and membranes as a promising class of materials and sensorized actuators, respectively.
>
---
#### [replaced 010] Merging and Disentangling Views in Visual Reinforcement Learning for Robotic Manipulation
- **分类: cs.LG; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.04619v2](http://arxiv.org/pdf/2505.04619v2)**

> **作者:** Abdulaziz Almuzairee; Rohan Patil; Dwait Bhatt; Henrik I. Christensen
>
> **备注:** Accepted at CoRL 2025. For project website and code, see https://aalmuzairee.github.io/mad
>
> **摘要:** Vision is well-known for its use in manipulation, especially using visual servoing. Due to the 3D nature of the world, using multiple camera views and merging them creates better representations for Q-learning and in turn, trains more sample efficient policies. Nevertheless, these multi-view policies are sensitive to failing cameras and can be burdensome to deploy. To mitigate these issues, we introduce a Merge And Disentanglement (MAD) algorithm that efficiently merges views to increase sample efficiency while simultaneously disentangling views by augmenting multi-view feature inputs with single-view features. This produces robust policies and allows lightweight deployment. We demonstrate the efficiency and robustness of our approach using Meta-World and ManiSkill3. For project website and code, see https://aalmuzairee.github.io/mad
>
---
#### [replaced 011] Centralization vs. decentralization in multi-robot sweep coverage with ground robots and UAVs
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.06553v4](http://arxiv.org/pdf/2408.06553v4)**

> **作者:** Aryo Jamshidpey; Mostafa Wahby; Michael Allwright; Weixu Zhu; Marco Dorigo; Mary Katherine Heinrich
>
> **备注:** IRIDIA, Universite Libre de Bruxelles, Brussels, Belgium, 2021
>
> **摘要:** In swarm robotics, decentralized control is often proposed as a more scalable and fault-tolerant alternative to centralized control. However, centralized behaviors are often faster and more efficient than their decentralized counterparts. In any given application, the goals and constraints of the task being solved should guide the choice to use centralized control, decentralized control, or a combination of the two. Currently, the exact trade-offs that exist between centralization and decentralization are not well defined. In this paper, we compare the performance of centralization and decentralization in the example task of sweep coverage, across five different types of multi-robot control structures: random walk, decentralized with beacons, hybrid formation control using self-organizing hierarchy, centralized formation control, and predetermined. In all five approaches, the coverage task is completed by a group of ground robots. In each approach, except for the random walk, the ground robots are assisted by UAVs, acting as supervisors or beacons. We compare the approaches in terms of three performance metrics for which centralized approaches are expected to have an advantage -- coverage completeness, coverage uniformity, and sweep completion time -- and two metrics for which decentralized approaches are expected to have an advantage -- scalability (4, 8, or 16 ground robots) and fault tolerance (0%, 25%, 50%, or 75% ground robot failure).
>
---
#### [replaced 012] Visual Imitation Enables Contextual Humanoid Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.03729v5](http://arxiv.org/pdf/2505.03729v5)**

> **作者:** Arthur Allshire; Hongsuk Choi; Junyi Zhang; David McAllister; Anthony Zhang; Chung Min Kim; Trevor Darrell; Pieter Abbeel; Jitendra Malik; Angjoo Kanazawa
>
> **备注:** Project website: https://www.videomimic.net/
>
> **摘要:** How can we teach humanoids to climb staircases and sit on chairs using the surrounding environment context? Arguably, the simplest way is to just show them-casually capture a human motion video and feed it to humanoids. We introduce VIDEOMIMIC, a real-to-sim-to-real pipeline that mines everyday videos, jointly reconstructs the humans and the environment, and produces whole-body control policies for humanoid robots that perform the corresponding skills. We demonstrate the results of our pipeline on real humanoid robots, showing robust, repeatable contextual control such as staircase ascents and descents, sitting and standing from chairs and benches, as well as other dynamic whole-body skills-all from a single policy, conditioned on the environment and global root commands. VIDEOMIMIC offers a scalable path towards teaching humanoids to operate in diverse real-world environments.
>
---
#### [replaced 013] Soft Manipulation Surface With Reduced Actuator Density For Heterogeneous Object Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.14290v2](http://arxiv.org/pdf/2411.14290v2)**

> **作者:** Pratik Ingle; Kasper Støy; Andres Faiña
>
> **摘要:** Object manipulation in robotics faces challenges due to diverse object shapes, sizes, and fragility. Gripper-based methods offer precision and low degrees of freedom (DOF) but the gripper limits the kind of objects to grasp. On the other hand, surface-based approaches provide flexibility for handling fragile and heterogeneous objects but require numerous actuators, increasing complexity. We propose new manipulation hardware that utilizes equally spaced linear actuators placed vertically and connected by a soft surface. In this setup, object manipulation occurs on the soft surface through coordinated movements of the surrounding actuators. This approach requires fewer actuators to cover a large manipulation area, offering a cost-effective solution with a lower DOF compared to dense actuator arrays. It also effectively handles heterogeneous objects of varying shapes and weights, even when they are significantly smaller than the distance between actuators. This method is particularly suitable for managing highly fragile objects in the food industry.
>
---
#### [replaced 014] Unified Path Planner with Adaptive Safety and Optimality
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.23197v2](http://arxiv.org/pdf/2505.23197v2)**

> **作者:** Jatin Kumar Arora; Soutrik Bandyopadhyay; Shubhendu Bhasin
>
> **备注:** 6 pages,4 figures
>
> **摘要:** Path planning for autonomous robots presents a fundamental trade-off between optimality and safety. While conventional algorithms typically prioritize one of these objectives, we introduce the Unified Path Planner (UPP), a unified framework that simultaneously addresses both. UPP is a graph-search-based algorithm that employs a modified heuristic function incorporating a dynamic safety cost, enabling an adaptive balance between path length and obstacle clearance. We establish theoretical sub-optimality bounds for the planner and demonstrate that its safety-to-optimality ratio can be tuned via adjustable parameters, with a trade-off in computational complexity. Extensive simulations show that UPP achieves a high success rate, generating near-optimal paths with only a negligible increase in cost over traditional A*, while ensuring safety margins that closely approach those of the classical Voronoi planner. Finally, the practical efficacy of UPP is validated through a hardware implementation on a TurtleBot, confirming its ability to navigate cluttered environments by generating safe, sub-optimal paths.
>
---
#### [replaced 015] Divide, Discover, Deploy: Factorized Skill Learning with Symmetry and Style Priors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.19953v2](http://arxiv.org/pdf/2508.19953v2)**

> **作者:** Rafael Cathomen; Mayank Mittal; Marin Vlastelica; Marco Hutter
>
> **备注:** Accepted to CoRL 2025. For code and videos, please check: https://leggedrobotics.github.io/d3-skill-discovery/
>
> **摘要:** Unsupervised Skill Discovery (USD) allows agents to autonomously learn diverse behaviors without task-specific rewards. While recent USD methods have shown promise, their application to real-world robotics remains underexplored. In this paper, we propose a modular USD framework to address the challenges in the safety, interpretability, and deployability of the learned skills. Our approach employs user-defined factorization of the state space to learn disentangled skill representations. It assigns different skill discovery algorithms to each factor based on the desired intrinsic reward function. To encourage structured morphology-aware skills, we introduce symmetry-based inductive biases tailored to individual factors. We also incorporate a style factor and regularization penalties to promote safe and robust behaviors. We evaluate our framework in simulation using a quadrupedal robot and demonstrate zero-shot transfer of the learned skills to real hardware. Our results show that factorization and symmetry lead to the discovery of structured human-interpretable behaviors, while the style factor and penalties enhance safety and diversity. Additionally, we show that the learned skills can be used for downstream tasks and perform on par with oracle policies trained with hand-crafted rewards.
>
---
#### [replaced 016] Latent Adaptive Planner for Dynamic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03077v2](http://arxiv.org/pdf/2505.03077v2)**

> **作者:** Donghun Noh; Deqian Kong; Minglu Zhao; Andrew Lizarraga; Jianwen Xie; Ying Nian Wu; Dennis Hong
>
> **摘要:** We present the Latent Adaptive Planner (LAP), a trajectory-level latent-variable policy for dynamic nonprehensile manipulation (e.g., box catching) that formulates planning as inference in a low-dimensional latent space and is learned effectively from human demonstration videos. During execution, LAP achieves real-time adaptation by maintaining a posterior over the latent plan and performing variational replanning as new observations arrive. To bridge the embodiment gap between humans and robots, we introduce a model-based proportional mapping that regenerates accurate kinematic-dynamic joint states and object positions from human demonstrations. Through challenging box catching experiments with varying object properties, LAP demonstrates superior success rates, trajectory smoothness, and energy efficiency by learning human-like compliant motions and adaptive behaviors. Overall, LAP enables dynamic manipulation with real-time adaptation and successfully transfer across heterogeneous robot platforms using the same human demonstration videos.
>
---
#### [replaced 017] SignLoc: Robust Localization using Navigation Signs and Public Maps
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.18606v2](http://arxiv.org/pdf/2508.18606v2)**

> **作者:** Nicky Zimmerman; Joel Loo; Ayush Agrawal; David Hsu
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Navigation signs and maps, such as floor plans and street maps, are widely available and serve as ubiquitous aids for way-finding in human environments. Yet, they are rarely used by robot systems. This paper presents SignLoc, a global localization method that leverages navigation signs to localize the robot on publicly available maps -- specifically floor plans and OpenStreetMap (OSM) graphs -- without prior sensor-based mapping. SignLoc first extracts a navigation graph from the input map. It then employs a probabilistic observation model to match directional and locational cues from the detected signs to the graph, enabling robust topo-semantic localization within a Monte Carlo framework. We evaluated SignLoc in diverse large-scale environments: part of a university campus, a shopping mall, and a hospital complex. Experimental results show that SignLoc reliably localizes the robot after observing only one to two signs.
>
---
#### [replaced 018] Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16084v2](http://arxiv.org/pdf/2505.16084v2)**

> **作者:** Zewei Zhang; Chenhao Li; Takahiro Miki; Marco Hutter
>
> **备注:** Conference on Robot Learning (CoRL)
>
> **摘要:** Reinforcement learning (RL)-based motion imitation methods trained on demonstration data can effectively learn natural and expressive motions with minimal reward engineering but often struggle to generalize to novel environments. We address this by proposing a hierarchical RL framework in which a low-level policy is first pre-trained to imitate animal motions on flat ground, thereby establishing motion priors. A subsequent high-level, goal-conditioned policy then builds on these priors, learning residual corrections that enable perceptive locomotion, local obstacle avoidance, and goal-directed navigation across diverse and rugged terrains. Simulation experiments illustrate the effectiveness of learned residuals in adapting to progressively challenging uneven terrains while still preserving the locomotion characteristics provided by the motion priors. Furthermore, our results demonstrate improvements in motion regularization over baseline models trained without motion priors under similar reward setups. Real-world experiments with an ANYmal-D quadruped robot confirm our policy's capability to generalize animal-like locomotion skills to complex terrains, demonstrating smooth and efficient locomotion and local navigation performance amidst challenging terrains with obstacles.
>
---
#### [replaced 019] QuaDreamer: Controllable Panoramic Video Generation for Quadruped Robots
- **分类: cs.RO; cs.CV; eess.IV**

- **链接: [http://arxiv.org/pdf/2508.02512v2](http://arxiv.org/pdf/2508.02512v2)**

> **作者:** Sheng Wu; Fei Teng; Hao Shi; Qi Jiang; Kai Luo; Kaiwei Wang; Kailun Yang
>
> **备注:** Accepted to CoRL 2025. The source code and model weights will be publicly available at \url{https://github.com/losehu/QuaDreamer
>
> **摘要:** Panoramic cameras, capturing comprehensive 360-degree environmental data, are suitable for quadruped robots in surrounding perception and interaction with complex environments. However, the scarcity of high-quality panoramic training data-caused by inherent kinematic constraints and complex sensor calibration challenges-fundamentally limits the development of robust perception systems tailored to these embodied platforms. To address this issue, we propose QuaDreamer-the first panoramic data generation engine specifically designed for quadruped robots. QuaDreamer focuses on mimicking the motion paradigm of quadruped robots to generate highly controllable, realistic panoramic videos, providing a data source for downstream tasks. Specifically, to effectively capture the unique vertical vibration characteristics exhibited during quadruped locomotion, we introduce Vertical Jitter Encoding (VJE). VJE extracts controllable vertical signals through frequency-domain feature filtering and provides high-quality prompts. To facilitate high-quality panoramic video generation under jitter signal control, we propose a Scene-Object Controller (SOC) that effectively manages object motion and boosts background jitter control through the attention mechanism. To address panoramic distortions in wide-FoV video generation, we propose the Panoramic Enhancer (PE)-a dual-stream architecture that synergizes frequency-texture refinement for local detail enhancement with spatial-structure correction for global geometric consistency. We further demonstrate that the generated video sequences can serve as training data for the quadruped robot's panoramic visual perception model, enhancing the performance of multi-object tracking in 360-degree scenes. The source code and model weights will be publicly available at https://github.com/losehu/QuaDreamer.
>
---
#### [replaced 020] UltraTac: Integrated Ultrasound-Augmented Visuotactile Sensor for Enhanced Robotic Perception
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20982v2](http://arxiv.org/pdf/2508.20982v2)**

> **作者:** Junhao Gong; Kit-Wa Sou; Shoujie Li; Changqing Guo; Yan Huang; Chuqiao Lyu; Ziwu Song; Wenbo Ding
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Visuotactile sensors provide high-resolution tactile information but are incapable of perceiving the material features of objects. We present UltraTac, an integrated sensor that combines visuotactile imaging with ultrasound sensing through a coaxial optoacoustic architecture. The design shares structural components and achieves consistent sensing regions for both modalities. Additionally, we incorporate acoustic matching into the traditional visuotactile sensor structure, enabling integration of the ultrasound sensing modality without compromising visuotactile performance. Through tactile feedback, we dynamically adjust the operating state of the ultrasound module to achieve flexible functional coordination. Systematic experiments demonstrate three key capabilities: proximity sensing in the 3-8 cm range ($R^2=0.90$), material classification (average accuracy: 99.20%), and texture-material dual-mode object recognition achieving 92.11% accuracy on a 15-class task. Finally, we integrate the sensor into a robotic manipulation system to concurrently detect container surface patterns and internal content, which verifies its potential for advanced human-machine interaction and precise robotic manipulation.
>
---
#### [replaced 021] PUB: A Plasma-Propelled Ultra-Quiet Blimp with Two-DOF Vector Thrusting
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2508.12395v2](http://arxiv.org/pdf/2508.12395v2)**

> **作者:** Zihan Wang
>
> **摘要:** This study presents the design and control of a Plasma-propelled Ultra-silence Blimp (PUB), a novel aerial robot employing plasma vector propulsion for ultra-quiet flight without mechanical propellers. The system utilizes a helium-lift platform for extended endurance and a four-layer ring asymmetric capacitor to generate ionic wind thrust. The modular propulsion units allow flexible configuration to meet mission-specific requirements, while a two-degree-of-freedom (DOF) head enables thrust vector control. A closed-loop slip control scheme is implemented for stable maneuvering. Flight experiments demonstrate full-envelope capability, including take-off, climb, hover, descent, and smooth landing, confirming the feasibility of plasma vector propulsion, the effectiveness of DOF vector control, and the stability of the control system. Owing to its low acoustic signature, structural simplicity, and high maneuverability, PUB is well suited for noise-sensitive, enclosed, and near-space applications.
>
---
