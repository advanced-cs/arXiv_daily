# 机器人 cs.RO

- **最新发布 40 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Effective Game-Theoretic Motion Planning via Nested Search
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出Game-Theoretic Nested Search（GTNS），用于在多智能体交互中高效计算纳什均衡，解决传统方法因动态简化或轨迹枚举导致的可扩展性差与局部最优问题，实现复杂动态系统下的实时安全规划。**

- **链接: []()**

> **作者:** Avishav Engle; Andrey Zhitnikov; Oren Salzman; Omer Ben-Porat; Kiril Solovey
>
> **摘要:** To facilitate effective, safe deployment in the real world, individual robots must reason about interactions with other agents, which often occur without explicit communication. Recent work has identified game theory, particularly the concept of Nash Equilibrium (NE), as a key enabler for behavior-aware decision-making. Yet, existing work falls short of fully unleashing the power of game-theoretic reasoning. Specifically, popular optimization-based methods require simplified robot dynamics and tend to get trapped in local minima due to convexification. Other works that rely on payoff matrices suffer from poor scalability due to the explicit enumeration of all possible trajectories. To bridge this gap, we introduce Game-Theoretic Nested Search (GTNS), a novel, scalable, and provably correct approach for computing NEs in general dynamical systems. GTNS efficiently searches the action space of all agents involved, while discarding trajectories that violate the NE constraint (no unilateral deviation) through an inner search over a lower-dimensional space. Our algorithm enables explicit selection among equilibria by utilizing a user-specified global objective, thereby capturing a rich set of realistic interactions. We demonstrate the approach on a variety of autonomous driving and racing scenarios where we achieve solutions in mere seconds on commodity hardware.
>
---
#### [new 002] A Two-Layer Electrostatic Film Actuator with High Actuation Stress and Integrated Brake
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出一种双层静电薄膜执行器，解决传统执行器应力低、需外置制动的问题。通过上下层交错电极设计，实现241 N/m²的高驱动力（提升90.5%），并集成静电吸附制动，验证了其在机器人中的高效驱动与锁紧能力。**

- **链接: []()**

> **作者:** Huacen Wang; Hongqiang Wang
>
> **摘要:** Robotic systems driven by conventional motors often suffer from challenges such as large mass, complex control algorithms, and the need for additional braking mechanisms, which limit their applications in lightweight and compact robotic platforms. Electrostatic film actuators offer several advantages, including thinness, flexibility, lightweight construction, and high open-loop positioning accuracy. However, the actuation stress exhibited by conventional actuators in air still needs improvement, particularly for the widely used three-phase electrode design. To enhance the output performance of actuators, this paper presents a two-layer electrostatic film actuator with an integrated brake. By alternately distributing electrodes on both the top and bottom layers, a smaller effective electrode pitch is achieved under the same fabrication constraints, resulting in an actuation stress of approximately 241~N/m$^2$, representing a 90.5\% improvement over previous three-phase actuators operating in air. Furthermore, its integrated electrostatic adhesion mechanism enables load retention under braking mode. Several demonstrations, including a tug-of-war between a conventional single-layer actuator and the proposed two-layer actuator, a payload operation, a one-degree-of-freedom robotic arm, and a dual-mode gripper, were conducted to validate the actuator's advantageous capabilities in both actuation and braking modes.
>
---
#### [new 003] Intuitive control of supernumerary robotic limbs through a tactile-encoded neural interface
- **分类: cs.RO**

- **简介: 该论文提出一种触觉编码的脑机接口，通过激活感觉传入通路实现对超数机械肢的直观控制，解决多自由度辅助运动与自然动作冲突问题，实验证明其可无干扰地实时控制双臂辅助双手法任务。**

- **链接: []()**

> **作者:** Tianyu Jia; Xingchen Yang; Ciaran McGeady; Yifeng Li; Jinzhi Lin; Kit San Ho; Feiyu Pan; Linhong Ji; Chong Li; Dario Farina
>
> **摘要:** Brain-computer interfaces (BCIs) promise to extend human movement capabilities by enabling direct neural control of supernumerary effectors, yet integrating augmented commands with multiple degrees of freedom without disrupting natural movement remains a key challenge. Here, we propose a tactile-encoded BCI that leverages sensory afferents through a novel tactile-evoked P300 paradigm, allowing intuitive and reliable decoding of supernumerary motor intentions even when superimposed with voluntary actions. The interface was evaluated in a multi-day experiment comprising of a single motor recognition task to validate baseline BCI performance and a dual task paradigm to assess the potential influence between the BCI and natural human movement. The brain interface achieved real-time and reliable decoding of four supernumerary degrees of freedom, with significant performance improvements after only three days of training. Importantly, after training, performance did not differ significantly between the single- and dual-BCI task conditions, and natural movement remained unimpaired during concurrent supernumerary control. Lastly, the interface was deployed in a movement augmentation task, demonstrating its ability to command two supernumerary robotic arms for functional assistance during bimanual tasks. These results establish a new neural interface paradigm for movement augmentation through stimulation of sensory afferents, expanding motor degrees of freedom without impairing natural movement.
>
---
#### [new 004] PerspAct: Enhancing LLM Situated Collaboration Skills through Perspective Taking and Active Vision
- **分类: cs.RO; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于多智能体协作任务，旨在解决LLM缺乏视角采择与主动视觉理解的问题。通过扩展ReAct框架并引入七种复杂场景，结合主动视觉探索与视角提示，显著提升了LLM对他人观点的理解与协作能力。**

- **链接: []()**

> **作者:** Sabrina Patania; Luca Annese; Anita Pellegrini; Silvia Serino; Anna Lambiase; Luca Pallonetto; Silvia Rossi; Simone Colombani; Tom Foulsham; Azzurra Ruggeri; Dimitri Ognibene
>
> **备注:** Accepted at IAS19
>
> **摘要:** Recent advances in Large Language Models (LLMs) and multimodal foundation models have significantly broadened their application in robotics and collaborative systems. However, effective multi-agent interaction necessitates robust perspective-taking capabilities, enabling models to interpret both physical and epistemic viewpoints. Current training paradigms often neglect these interactive contexts, resulting in challenges when models must reason about the subjectivity of individual perspectives or navigate environments with multiple observers. This study evaluates whether explicitly incorporating diverse points of view using the ReAct framework, an approach that integrates reasoning and acting, can enhance an LLM's ability to understand and ground the demands of other agents. We extend the classic Director task by introducing active visual exploration across a suite of seven scenarios of increasing perspective-taking complexity. These scenarios are designed to challenge the agent's capacity to resolve referential ambiguity based on visual access and interaction, under varying state representations and prompting strategies, including ReAct-style reasoning. Our results demonstrate that explicit perspective cues, combined with active exploration strategies, significantly improve the model's interpretative accuracy and collaborative effectiveness. These findings highlight the potential of integrating active perception with perspective-taking mechanisms in advancing LLMs' application in robotics and multi-agent systems, setting a foundation for future research into adaptive and context-aware AI systems.
>
---
#### [new 005] Real-Time Performance Analysis of Multi-Fidelity Residual Physics-Informed Neural Process-Based State Estimation for Robotic Systems
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出一种多保真残差物理信息神经过程（MFR-PINP）方法，用于机器人实时状态估计，解决传统模型失配与不确定性量化问题，结合分裂共形预测框架提供可靠误差界，实验表明其性能优于经典卡尔曼滤波器。**

- **链接: []()**

> **作者:** Devin Hunter; Chinwendu Enyioha
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Various neural network architectures are used in many of the state-of-the-art approaches for real-time nonlinear state estimation. With the ever-increasing incorporation of these data-driven models into the estimation domain, model predictions with reliable margins of error are a requirement -- especially for safety-critical applications. This paper discusses the application of a novel real-time, data-driven estimation approach based on the multi-fidelity residual physics-informed neural process (MFR-PINP) toward the real-time state estimation of a robotic system. Specifically, we address the model-mismatch issue of selecting an accurate kinematic model by tasking the MFR-PINP to also learn the residuals between simple, low-fidelity predictions and complex, high-fidelity ground-truth dynamics. To account for model uncertainty present in a physical implementation, robust uncertainty guarantees from the split conformal (SC) prediction framework are modeled in the training and inference paradigms. We provide implementation details of our MFR-PINP-based estimator for a hybrid online learning setting to validate our model's usage in real-time applications. Experimental results of our approach's performance in comparison to the state-of-the-art variants of the Kalman filter (i.e. unscented Kalman filter and deep Kalman filter) in estimation scenarios showed promising results for the MFR-PINP model as a viable option in real-time estimation tasks.
>
---
#### [new 006] Local Path Planning with Dynamic Obstacle Avoidance in Unstructured Environments
- **分类: cs.RO; math.DS; math.OC**

- **简介: 该论文针对非结构化环境中无人车的局部路径规划问题，提出融合切线规划与轨迹外推的算法，实现在动态障碍物（多项式轨迹）中的安全避障，仿真验证了其有效性和实时性。**

- **链接: []()**

> **作者:** Okan Arif Guvenkaya; Selim Ahmet Iz; Mustafa Unel
>
> **摘要:** Obstacle avoidance and path planning are essential for guiding unmanned ground vehicles (UGVs) through environments that are densely populated with dynamic obstacles. This paper develops a novel approach that combines tangentbased path planning and extrapolation methods to create a new decision-making algorithm for local path planning. In the assumed scenario, a UGV has a prior knowledge of its initial and target points within the dynamic environment. A global path has already been computed, and the robot is provided with waypoints along this path. As the UGV travels between these waypoints, the algorithm aims to avoid collisions with dynamic obstacles. These obstacles follow polynomial trajectories, with their initial positions randomized in the local map and velocities randomized between O and the allowable physical velocity limit of the robot, along with some random accelerations. The developed algorithm is tested in several scenarios where many dynamic obstacles move randomly in the environment. Simulation results show the effectiveness of the proposed local path planning strategy by gradually generating a collision free path which allows the robot to navigate safely between initial and the target locations.
>
---
#### [new 007] Testing and Evaluation of Underwater Vehicle Using Hardware-In-The-Loop Simulation with HoloOcean
- **分类: cs.RO**

- **简介: 该论文提出基于HoloOcean 2.0的软硬件在环（HIL/SIL）仿真系统，解决水下机器人在有限空间内难以测试传感器与控制算法的问题，通过ROS 2桥接实现实物AUV与高保真仿真交互，并验证仿真结果与实测数据的一致性。**

- **链接: []()**

> **作者:** Braden Meyers; Joshua G. Mangelson
>
> **备注:** Published in IEEE/MTS OCEANS Conference proceedings 2025 Great Lakes
>
> **摘要:** Testing marine robotics systems in controlled environments before field tests is challenging, especially when acoustic-based sensors and control surfaces only function properly underwater. Deploying robots in indoor tanks and pools often faces space constraints that complicate testing of control, navigation, and perception algorithms at scale. Recent developments of high-fidelity underwater simulation tools have the potential to address these problems. We demonstrate the utility of the recently released HoloOcean 2.0 simulator with improved dynamics for torpedo AUV vehicles and a new ROS 2 interface. We have successfully demonstrated a Hardware-in-the-Loop (HIL) and Software-in-the-Loop (SIL) setup for testing and evaluating a CougUV torpedo autonomous underwater vehicle (AUV) that was built and developed in our lab. With this HIL and SIL setup, simulations are run in HoloOcean using a ROS 2 bridge such that simulated sensor data is sent to the CougUV (mimicking sensor drivers) and control surface commands are sent back to the simulation, where vehicle dynamics and sensor data are calculated. We compare our simulated results to real-world field trial results.
>
---
#### [new 008] AVOID-JACK: Avoidance of Jackknifing for Swarms of Long Heavy Articulated Vehicles
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出一种去中心化 swarm 智能策略，解决长铰接车辆集群的侧翻（jackknifing）与碰撞问题，首次实现纯反应式避障控制，仿真验证其在单车与多车场景下高效规避侧翻（>98%）并避免碰撞（>99%）。**

- **链接: []()**

> **作者:** Adrian Schönnagel; Michael Dubé; Christoph Steup; Felix Keppler; Sanaz Mostaghim
>
> **备注:** 6+1 pages, 9 figures, accepted for publication in IEEE MRS 2025
>
> **摘要:** This paper presents a novel approach to avoiding jackknifing and mutual collisions in Heavy Articulated Vehicles (HAVs) by leveraging decentralized swarm intelligence. In contrast to typical swarm robotics research, our robots are elongated and exhibit complex kinematics, introducing unique challenges. Despite its relevance to real-world applications such as logistics automation, remote mining, airport baggage transport, and agricultural operations, this problem has not been addressed in the existing literature. To tackle this new class of swarm robotics problems, we propose a purely reaction-based, decentralized swarm intelligence strategy tailored to automate elongated, articulated vehicles. The method presented in this paper prioritizes jackknifing avoidance and establishes a foundation for mutual collision avoidance. We validate our approach through extensive simulation experiments and provide a comprehensive analysis of its performance. For the experiments with a single HAV, we observe that for 99.8% jackknifing was successfully avoided and that 86.7% and 83.4% reach their first and second goals, respectively. With two HAVs interacting, we observe 98.9%, 79.4%, and 65.1%, respectively, while 99.7% of the HAVs do not experience mutual collisions.
>
---
#### [new 009] Time-Aware Policy Learning for Adaptive and Punctual Robot Control
- **分类: cs.RO**

- **简介: 该论文提出时间感知策略学习，将时间作为可控变量引入强化学习，使机器人能自适应调整动作节奏，提升效率、鲁棒性与人机协同能力，解决传统方法忽视时间维度的问题。**

- **链接: []()**

> **作者:** Yinsen Jia; Boyuan Chen
>
> **摘要:** Temporal awareness underlies intelligent behavior in both animals and humans, guiding how actions are sequenced, paced, and adapted to changing goals and environments. Yet most robot learning algorithms remain blind to time. We introduce time-aware policy learning, a reinforcement learning framework that enables robots to explicitly perceive and reason with time as a first-class variable. The framework augments conventional reinforcement policies with two complementary temporal signals, the remaining time and a time ratio, which allow a single policy to modulate its behavior continuously from rapid and dynamic to cautious and precise execution. By jointly optimizing punctuality and stability, the robot learns to balance efficiency, robustness, resiliency, and punctuality without re-training or reward adjustment. Across diverse manipulation domains from long-horizon pick and place, to granular-media pouring, articulated-object handling, and multi-agent object delivery, the time-aware policy produces adaptive behaviors that outperform standard reinforcement learning baselines by up to 48% in efficiency, 8 times more robust in sim-to-real transfer, and 90% in acoustic quietness while maintaining near-perfect success rates. Explicit temporal reasoning further enables real-time human-in-the-loop control and multi-agent coordination, allowing robots to recover from disturbances, re-synchronize after delays, and align motion tempo with human intent. By treating time not as a constraint but as a controllable dimension of behavior, time-aware policy learning provides a unified foundation for efficient, robust, resilient, and human-aligned robot autonomy.
>
---
#### [new 010] High-Altitude Balloon Station-Keeping with First Order Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文研究高空浮空器定点控制问题，提出基于可微分模型预测控制（FOMPC），通过在线优化实现比强化学习更优的定点精度，无需离线训练，验证了模型基方法在复杂风场下的有效性。**

- **链接: []()**

> **作者:** Myles Pasetsky; Jiawei Lin; Bradley Guo; Sarah Dean
>
> **摘要:** High-altitude balloons (HABs) are common in scientific research due to their wide range of applications and low cost. Because of their nonlinear, underactuated dynamics and the partial observability of wind fields, prior work has largely relied on model-free reinforcement learning (RL) methods to design near-optimal control schemes for station-keeping. These methods often compare only against hand-crafted heuristics, dismissing model-based approaches as impractical given the system complexity and uncertain wind forecasts. We revisit this assumption about the efficacy of model-based control for station-keeping by developing First-Order Model Predictive Control (FOMPC). By implementing the wind and balloon dynamics as differentiable functions in JAX, we enable gradient-based trajectory optimization for online planning. FOMPC outperforms a state-of-the-art RL policy, achieving a 24% improvement in time-within-radius (TWR) without requiring offline training, though at the cost of greater online computation per control step. Through systematic ablations of modeling assumptions and control factors, we show that online planning is effective across many configurations, including under simplified wind and dynamics models.
>
---
#### [new 011] Model Predictive Control via Probabilistic Inference: A Tutorial
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制理论与机器学习交叉任务，旨在解决传统MPC在非线性、非可微系统中优化困难的问题。通过概率推断框架，统一阐述并推导了MPPI等采样驱动方法，提供了一套可扩展的非梯度优化指南。**

- **链接: []()**

> **作者:** Kohei Honda
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** Model Predictive Control (MPC) is a fundamental framework for optimizing robot behavior over a finite future horizon. While conventional numerical optimization methods can efficiently handle simple dynamics and cost structures, they often become intractable for the nonlinear or non-differentiable systems commonly encountered in robotics. This article provides a tutorial on probabilistic inference-based MPC, presenting a unified theoretical foundation and a comprehensive overview of representative methods. Probabilistic inference-based MPC approaches, such as Model Predictive Path Integral (MPPI) control, have gained significant attention by reinterpreting optimal control as a problem of probabilistic inference. Rather than relying on gradient-based numerical optimization, these methods estimate optimal control distributions through sampling-based techniques, accommodating arbitrary cost functions and dynamics. We first derive the optimal control distribution from the standard optimal control problem, elucidating its probabilistic interpretation and key characteristics. The widely used MPPI algorithm is then derived as a practical example, followed by discussions on prior and variational distribution design, tuning principles, and theoretical aspects. This article aims to serve as a systematic guide for researchers and practitioners seeking to understand, implement, and extend these methods in robotics and beyond.
>
---
#### [new 012] Benchmarking Resilience and Sensitivity of Polyurethane-Based Vision-Based Tactile Sensors
- **分类: cs.RO**

- **简介: 该论文对比硅胶与聚氨酯基视觉触觉传感器的耐久性与灵敏度，提出标准化评估协议，解决硅胶易磨损问题，证实聚氨酯在高负载场景下更具鲁棒性，同时保持实用灵敏度。**

- **链接: []()**

> **作者:** Benjamin Davis; Hannah Stuart
>
> **摘要:** Vision-based tactile sensors (VBTSs) are a promising technology for robots, providing them with dense signals that can be translated into an understanding of normal and shear load, contact region, texture classification, and more. However, existing VBTS tactile surfaces make use of silicone gels, which provide high sensitivity but easily deteriorate from loading and surface wear. We propose that polyurethane rubber, used for high-load applications like shoe soles, rubber wheels, and industrial gaskets, may provide improved physical gel resilience, potentially at the cost of sensitivity. To compare the resilience and sensitivity of silicone and polyurethane VBTS gels, we propose a series of standard evaluation benchmarking protocols. Our resilience tests assess sensor durability across normal loading, shear loading, and abrasion. For sensitivity, we introduce model-free assessments of force and spatial sensitivity to directly measure the physical capabilities of each gel without effects introduced from data and model quality. Finally, we include a bottle cap loosening and tightening demonstration as an example where polyurethane gels provide an advantage over their silicone counterparts.
>
---
#### [new 013] Safe and Optimal Learning from Preferences via Weighted Temporal Logic with Applications in Robotics and Formula 1
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出基于加权时序逻辑（WSTL）的偏好学习方法，解决安全关键系统中从人类反馈学习最优行为但缺乏安全保证的问题，通过结构剪枝与对数变换将其转化为混合整数线性规划，确保安全性与高效性。**

- **链接: []()**

> **作者:** Ruya Karagulle; Cristian-Ioan Vasile; Necmiye Ozay
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Autonomous systems increasingly rely on human feedback to align their behavior, expressed as pairwise comparisons, rankings, or demonstrations. While existing methods can adapt behaviors, they often fail to guarantee safety in safety-critical domains. We propose a safety-guaranteed, optimal, and efficient approach to solve the learning problem from preferences, rankings, or demonstrations using Weighted Signal Temporal Logic (WSTL). WSTL learning problems, when implemented naively, lead to multi-linear constraints in the weights to be learned. By introducing structural pruning and log-transform procedures, we reduce the problem size and recast the problem as a Mixed-Integer Linear Program while preserving safety guarantees. Experiments on robotic navigation and real-world Formula 1 data demonstrate that the method effectively captures nuanced preferences and models complex task objectives.
>
---
#### [new 014] LLM-GROP: Visually Grounded Robot Task and Motion Planning with Large Language Models
- **分类: cs.RO**

- **简介: 论文提出LLM-GROP框架，结合大语言模型与视觉感知，解决多物体移动机器人任务与运动规划（TAMP）问题，利用常识知识与基位选择策略，实现复杂场景下的自适应物体重组。**

- **链接: []()**

> **作者:** Xiaohan Zhang; Yan Ding; Yohei Hayamizu; Zainab Altaweel; Yifeng Zhu; Yuke Zhu; Peter Stone; Chris Paxton; Shiqi Zhang
>
> **摘要:** Task planning and motion planning are two of the most important problems in robotics, where task planning methods help robots achieve high-level goals and motion planning methods maintain low-level feasibility. Task and motion planning (TAMP) methods interleave the two processes of task planning and motion planning to ensure goal achievement and motion feasibility. Within the TAMP context, we are concerned with the mobile manipulation (MoMa) of multiple objects, where it is necessary to interleave actions for navigation and manipulation. In particular, we aim to compute where and how each object should be placed given underspecified goals, such as ``set up dinner table with a fork, knife and plate.'' We leverage the rich common sense knowledge from large language models (LLMs), e.g., about how tableware is organized, to facilitate both task-level and motion-level planning. In addition, we use computer vision methods to learn a strategy for selecting base positions to facilitate MoMa behaviors, where the base position corresponds to the robot's ``footprint'' and orientation in its operating space. Altogether, this article provides a principled TAMP framework for MoMa tasks that accounts for common sense about object rearrangement and is adaptive to novel situations that include many objects that need to be moved. We performed quantitative experiments in both real-world settings and simulated environments. We evaluated the success rate and efficiency in completing long-horizon object rearrangement tasks. While the robot completed 84.4\% real-world object rearrangement trials, subjective human evaluations indicated that the robot's performance is still lower than experienced human waiters.
>
---
#### [new 015] A CODECO Case Study and Initial Validation for Edge Orchestration of Autonomous Mobile Robots
- **分类: cs.RO; cs.ET; cs.NI**

- **简介: 该论文针对边缘环境下自主移动机器人（AMR）的容器编排问题，对比CODECO与Kubernetes的性能，验证CODECO在降低CPU开销、稳定通信方面的优势，虽有轻微内存与延迟代价，更适合资源受限的边缘场景。**

- **链接: []()**

> **作者:** H. Zhu; T. Samizadeh; R. C. Sofia
>
> **摘要:** Autonomous Mobile Robots (AMRs) increasingly adopt containerized micro-services across the Edge-Cloud continuum. While Kubernetes is the de-facto orchestrator for such systems, its assumptions of stable networks, homogeneous resources, and ample compute capacity do not fully hold in mobile, resource-constrained robotic environments. This paper describes a case study on smart-manufacturing AMRs and performs an initial comparison between CODECO orchestration and standard Kubernetes using a controlled KinD environment. Metrics include pod deployment and deletion times, CPU and memory usage, and inter-pod data rates. The observed results indicate that CODECO offers reduced CPU consumption and more stable communication patterns, at the cost of modest memory overhead (10-15%) and slightly increased pod lifecycle latency due to secure overlay initialization.
>
---
#### [new 016] EquiMus: Energy-Equivalent Dynamic Modeling and Simulation of Musculoskeletal Robots Driven by Linear Elastic Actuators
- **分类: cs.RO**

- **简介: 论文提出EquiMus框架，解决软硬混合肌骨机器人动态建模与仿真难题，通过能量等效方法实现高效模拟，并在仿生腿上验证其在控制设计与学习中的有效性。**

- **链接: []()**

> **作者:** Yinglei Zhu; Xuguang Dong; Qiyao Wang; Qi Shao; Fugui Xie; Xinjun Liu; Huichan Zhao
>
> **摘要:** Dynamic modeling and control are critical for unleashing soft robots' potential, yet remain challenging due to their complex constitutive behaviors and real-world operating conditions. Bio-inspired musculoskeletal robots, which integrate rigid skeletons with soft actuators, combine high load-bearing capacity with inherent flexibility. Although actuation dynamics have been studied through experimental methods and surrogate models, accurate and effective modeling and simulation remain a significant challenge, especially for large-scale hybrid rigid--soft robots with continuously distributed mass, kinematic loops, and diverse motion modes. To address these challenges, we propose EquiMus, an energy-equivalent dynamic modeling framework and MuJoCo-based simulation for musculoskeletal rigid--soft hybrid robots with linear elastic actuators. The equivalence and effectiveness of the proposed approach are validated and examined through both simulations and real-world experiments on a bionic robotic leg. EquiMus further demonstrates its utility for downstream tasks, including controller design and learning-based control strategies.
>
---
#### [new 017] A Comprehensive Experimental Characterization of Mechanical Layer Jamming Systems
- **分类: cs.RO**

- **简介: 该论文研究机械层锁定系统，旨在解决软体机器人刚度调控问题。通过双层带齿结构实现弯曲和扭转下的刚度调制，系统测试了齿形参数对性能的影响，发现刚度可提升5倍（弯曲）和3.2倍（扭转），并量化了层间分离力，为设计提供指导。**

- **链接: []()**

> **作者:** Jessica Gumowski; Krishna Manaswi Digumarti; David Howard
>
> **备注:** 6 pages, 9 figures, RoboSoft 2026
>
> **摘要:** Organisms in nature, such as Cephalopods and Pachyderms, exploit stiffness modulation to achieve amazing dexterity in the control of their appendages. In this paper, we explore the phenomenon of layer jamming, which is a popular stiffness modulation mechanism that provides an equivalent capability for soft robots. More specifically, we focus on mechanical layer jamming, which we realise through two-layer multi material structure with tooth-like protrusions. We identify key design parameters for mechanical layer jamming systems, including the ability to modulate stiffness, and perform a variety of comprehensive tests placing the specimens under bending and torsional loads to understand the influence of our selected design parameters (mainly tooth geometry) on the performance of the jammed structures. We note the ability of these structures to produce a peak change in stiffness of 5 times in bending and 3.2 times in torsion. We also measure the force required to separate the two jammed layers, an often ignored parameter in the study of jamming-induced stiffness change. This study aims to shed light on the principled design of mechanical layer jammed systems and guide researchers in the selection of appropriate designs for their specific application domains.
>
---
#### [new 018] Learning Omnidirectional Locomotion for a Salamander-Like Quadruped Robot
- **分类: cs.RO**

- **简介: 该论文针对蝾螈型四足机器人，提出一种无参考轨迹的强化学习框架，通过相位变量与对称性数据增强，自主学习22种全向步态，解决传统方法依赖预设轨迹、灵活性不足的问题。**

- **链接: []()**

> **作者:** Zhiang Liu; Yang Liu; Yongchun Fang; Xian Guo
>
> **摘要:** Salamander-like quadruped robots are designed inspired by the skeletal structure of their biological counterparts. However, existing controllers cannot fully exploit these morphological features and largely rely on predefined gait patterns or joint trajectories, which prevents the generation of diverse and flexible locomotion and limits their applicability in real-world scenarios. In this paper, we propose a learning framework that enables the robot to acquire a diverse repertoire of omnidirectional gaits without reference motions. Each body part is controlled by a phase variable capable of forward and backward evolution, with a phase coverage reward to promote the exploration of the leg phase space. Additionally, morphological symmetry of the robot is incorporated via data augmentation, improving sample efficiency and enforcing both motion-level and task-level symmetry in learned behaviors. Extensive experiments show that the robot successfully acquires 22 omnidirectional gaits exhibiting both dynamic and symmetric movements, demonstrating the effectiveness of the proposed learning framework.
>
---
#### [new 019] Human Motion Intent Inferencing in Teleoperation Through a SINDy Paradigm
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出Psychic框架，用于遥操作中的人体运动意图推断，通过SINDy模型结合跳跃-漂移-扩散方程，检测轨迹突变并融合连续/离散动力学，实现未结构化场景下的意图早期识别与行为建模。**

- **链接: []()**

> **作者:** Michael Bowman; Xiaoli Zhang
>
> **备注:** Open source software and video examples here: https://github.com/namwob44/Psychic
>
> **摘要:** Intent inferencing in teleoperation has been instrumental in aligning operator goals and coordinating actions with robotic partners. However, current intent inference methods often ignore subtle motion that can be strong indicators for a sudden change in intent. Specifically, we aim to tackle 1) if we can detect sudden jumps in operator trajectories, 2) how we appropriately use these sudden jump motions to infer an operator's goal state, and 3) how to incorporate these discontinuous and continuous dynamics to infer operator motion. Our framework, called Psychic, models these small indicative motions through a jump-drift-diffusion stochastic differential equation to cover discontinuous and continuous dynamics. Kramers-Moyal (KM) coefficients allow us to detect jumps with a trajectory which we pair with a statistical outlier detection algorithm to nominate goal transitions. Through identifying jumps, we can perform early detection of existing goals and discover undefined goals in unstructured scenarios. Our framework then applies a Sparse Identification of Nonlinear Dynamics (SINDy) model using KM coefficients with the goal transitions as a control input to infer an operator's motion behavior in unstructured scenarios. We demonstrate Psychic can produce probabilistic reachability sets and compare our strategy to a negative log-likelihood model fit. We perform a retrospective study on 600 operator trajectories in a hands-free teleoperation task to evaluate the efficacy of our opensource package, Psychic, in both offline and online learning.
>
---
#### [new 020] Navigating the Wild: Pareto-Optimal Visual Decision-Making in Image Space
- **分类: cs.RO**

- **简介: 该论文提出一种轻量级图像空间导航框架，解决传统方法依赖地图或大量数据、泛化差的问题，融合语义理解、帕累托最优决策与视觉伺服，实现无需地图的实时自适应视觉导航。**

- **链接: []()**

> **作者:** Durgakant Pushp; Weizhe Chen; Zheng Chen; Chaomin Luo; Jason M. Gregory; Lantao Liu
>
> **摘要:** Navigating complex real-world environments requires semantic understanding and adaptive decision-making. Traditional reactive methods without maps often fail in cluttered settings, map-based approaches demand heavy mapping effort, and learning-based solutions rely on large datasets with limited generalization. To address these challenges, we present Pareto-Optimal Visual Navigation, a lightweight image-space framework that combines data-driven semantics, Pareto-optimal decision-making, and visual servoing for real-time navigation.
>
---
#### [new 021] RoboTAG: End-to-end Robot Configuration Estimation via Topological Alignment Graph
- **分类: cs.RO; cs.CV**

- **简介: 论文提出RoboTAG，用于从单目图像估计机器人位姿，解决标注数据稀缺与3D先验缺失问题。通过双分支图结构融合2D/3D表示，利用拓扑一致性实现无监督训练，提升跨机器人类型泛化能力。**

- **链接: []()**

> **作者:** Yifan Liu; Fangneng Zhan; Wanhua Li; Haowen Sun; Katerina Fragkiadaki; Hanspeter Pfister
>
> **摘要:** Estimating robot pose from a monocular RGB image is a challenge in robotics and computer vision. Existing methods typically build networks on top of 2D visual backbones and depend heavily on labeled data for training, which is often scarce in real-world scenarios, causing a sim-to-real gap. Moreover, these approaches reduce the 3D-based problem to 2D domain, neglecting the 3D priors. To address these, we propose Robot Topological Alignment Graph (RoboTAG), which incorporates a 3D branch to inject 3D priors while enabling co-evolution of the 2D and 3D representations, alleviating the reliance on labels. Specifically, the RoboTAG consists of a 3D branch and a 2D branch, where nodes represent the states of the camera and robot system, and edges capture the dependencies between these variables or denote alignments between them. Closed loops are then defined in the graph, on which a consistency supervision across branches can be applied. This design allows us to utilize in-the-wild images as training data without annotations. Experimental results demonstrate that our method is effective across robot types, highlighting its potential to alleviate the data bottleneck in robotics.
>
---
#### [new 022] CAVER: Curious Audiovisual Exploring Robot
- **分类: cs.RO**

- **简介: CAVER是一种好奇驱动的音频视觉机器人，通过专用末端执行器激发物体声音，构建融合视觉与声学特征的表征，并采用好奇心算法高效探索，提升材料分类与听音模仿性能。**

- **链接: []()**

> **作者:** Luca Macesanu; Boueny Folefack; Samik Singh; Ruchira Ray; Ben Abbatematteo; Roberto Martín-Martín
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Multimodal audiovisual perception can enable new avenues for robotic manipulation, from better material classification to the imitation of demonstrations for which only audio signals are available (e.g., playing a tune by ear). However, to unlock such multimodal potential, robots need to learn the correlations between an object's visual appearance and the sound it generates when they interact with it. Such an active sensorimotor experience requires new interaction capabilities, representations, and exploration methods to guide the robot in efficiently building increasingly rich audiovisual knowledge. In this work, we present CAVER, a novel robot that builds and utilizes rich audiovisual representations of objects. CAVER includes three novel contributions: 1) a novel 3D printed end-effector, attachable to parallel grippers, that excites objects' audio responses, 2) an audiovisual representation that combines local and global appearance information with sound features, and 3) an exploration algorithm that uses and builds the audiovisual representation in a curiosity-driven manner that prioritizes interacting with high uncertainty objects to obtain good coverage of surprising audio with fewer interactions. We demonstrate that CAVER builds rich representations in different scenarios more efficiently than several exploration baselines, and that the learned audiovisual representation leads to significant improvements in material classification and the imitation of audio-only human demonstrations. https://caver-bot.github.io/
>
---
#### [new 023] A Supervised Autonomous Resection and Retraction Framework for Transurethral Enucleation of the Prostatic Median Lobe
- **分类: cs.RO**

- **简介: 该论文提出一种监督式自主系统，用于经尿道前列腺中叶切除。结合基于模型的切除规划器与学习型牵拉网络（PushCVAE），在仿生前列腺模型上实现97.1%的精准切除，推动微创机器人手术自动化。**

- **链接: []()**

> **作者:** Mariana Smith; Tanner Watts; Susheela Sharma Stern; Brendan Burkhart; Hao Li; Alejandro O. Chara; Nithesh Kumar; James Ferguson; Ayberk Acar; Jesse F. d'Almeida; Lauren Branscombe; Lauren Shepard; Ahmed Ghazi; Ipek Oguz; Jie Ying Wu; Robert J. Webster; Axel Krieger; Alan Kuntz
>
> **备注:** Submitted to International Symposium on Medial Robotics (ISMR) 2026. 7 pages, 8 figures
>
> **摘要:** Concentric tube robots (CTRs) offer dexterous motion at millimeter scales, enabling minimally invasive procedures through natural orifices. This work presents a coordinated model-based resection planner and learning-based retraction network that work together to enable semi-autonomous tissue resection using a dual-arm transurethral concentric tube robot (the Virtuoso). The resection planner operates directly on segmented CT volumes of prostate phantoms, automatically generating tool trajectories for a three-phase median lobe resection workflow: left/median trough resection, right/median trough resection, and median blunt dissection. The retraction network, PushCVAE, trained on surgeon demonstrations, generates retractions according to the procedural phase. The procedure is executed under Level-3 (supervised) autonomy on a prostate phantom composed of hydrogel materials that replicate the mechanical and cutting properties of tissue. As a feasibility study, we demonstrate that our combined autonomous system achieves a 97.1% resection of the targeted volume of the median lobe. Our study establishes a foundation for image-guided autonomy in transurethral robotic surgery and represents a first step toward fully automated minimally-invasive prostate enucleation.
>
---
#### [new 024] Prioritizing Perception-Guided Self-Supervision: A New Paradigm for Causal Modeling in End-to-End Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文针对端到端自动驾驶中的因果混淆问题，提出感知引导自监督（PGS）方法，利用感知输出（如车道线、车辆轨迹）作为监督信号，替代专家轨迹，提升决策模型的因果建模能力，在Bench2Drive上显著超越现有方法。**

- **链接: []()**

> **作者:** Yi Huang; Zhan Qu; Lihui Jiang; Bingbing Liu; Hongbo Zhang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** End-to-end autonomous driving systems, predominantly trained through imitation learning, have demonstrated considerable effectiveness in leveraging large-scale expert driving data. Despite their success in open-loop evaluations, these systems often exhibit significant performance degradation in closed-loop scenarios due to causal confusion. This confusion is fundamentally exacerbated by the overreliance of the imitation learning paradigm on expert trajectories, which often contain unattributable noise and interfere with the modeling of causal relationships between environmental contexts and appropriate driving actions. To address this fundamental limitation, we propose Perception-Guided Self-Supervision (PGS) - a simple yet effective training paradigm that leverages perception outputs as the primary supervisory signals, explicitly modeling causal relationships in decision-making. The proposed framework aligns both the inputs and outputs of the decision-making module with perception results, such as lane centerlines and the predicted motions of surrounding agents, by introducing positive and negative self-supervision for the ego trajectory. This alignment is specifically designed to mitigate causal confusion arising from the inherent noise in expert trajectories. Equipped with perception-driven supervision, our method, built on a standard end-to-end architecture, achieves a Driving Score of 78.08 and a mean success rate of 48.64% on the challenging closed-loop Bench2Drive benchmark, significantly outperforming existing state-of-the-art methods, including those employing more complex network architectures and inference pipelines. These results underscore the effectiveness and robustness of the proposed PGS framework and point to a promising direction for addressing causal confusion and enhancing real-world generalization in autonomous driving.
>
---
#### [new 025] SeFA-Policy: Fast and Accurate Visuomotor Policy Learning with Selective Flow Alignment
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出SeFA-Policy，用于机器人模仿学习中的视觉运动策略学习，解决流模型迭代蒸馏中动作与观测脱节导致的误差累积问题。通过选择性流对齐机制，在保持单步推理效率的同时，确保动作与观测一致，显著提升精度与鲁棒性。**

- **链接: []()**

> **作者:** Rong Xue; Jiageng Mao; Mingtong Zhang; Yue Wang
>
> **摘要:** Developing efficient and accurate visuomotor policies poses a central challenge in robotic imitation learning. While recent rectified flow approaches have advanced visuomotor policy learning, they suffer from a key limitation: After iterative distillation, generated actions may deviate from the ground-truth actions corresponding to the current visual observation, leading to accumulated error as the reflow process repeats and unstable task execution. We present Selective Flow Alignment (SeFA), an efficient and accurate visuomotor policy learning framework. SeFA resolves this challenge by a selective flow alignment strategy, which leverages expert demonstrations to selectively correct generated actions and restore consistency with observations, while preserving multimodality. This design introduces a consistency correction mechanism that ensures generated actions remain observation-aligned without sacrificing the efficiency of one-step flow inference. Extensive experiments across both simulated and real-world manipulation tasks show that SeFA Policy surpasses state-of-the-art diffusion-based and flow-based policies, achieving superior accuracy and robustness while reducing inference latency by over 98%. By unifying rectified flow efficiency with observation-consistent action generation, SeFA provides a scalable and dependable solution for real-time visuomotor policy learning. Code is available on https://github.com/RongXueZoe/SeFA.
>
---
#### [new 026] X-IONet: Cross-Platform Inertial Odometry Network with Dual-Stage Attention
- **分类: cs.RO; cs.LG**

- **简介: X-IONet面向跨平台惯性里程计任务，解决人与四足机器人运动差异导致的模型性能下降问题，提出双阶段注意力网络与专家选择机制，仅用单IMU实现高精度位姿估计，显著提升两类平台的定位精度。**

- **链接: []()**

> **作者:** Dehan Shen; Changhao Chen
>
> **摘要:** Learning-based inertial odometry has achieved remarkable progress in pedestrian navigation. However, extending these methods to quadruped robots remains challenging due to their distinct and highly dynamic motion patterns. Models that perform well on pedestrian data often experience severe degradation when deployed on legged platforms. To tackle this challenge, we introduce X-IONet, a cross-platform inertial odometry framework that operates solely using a single Inertial Measurement Unit (IMU). X-IONet incorporates a rule-based expert selection module to classify motion platforms and route IMU sequences to platform-specific expert networks. The displacement prediction network features a dual-stage attention architecture that jointly models long-range temporal dependencies and inter-axis correlations, enabling accurate motion representation. It outputs both displacement and associated uncertainty, which are further fused through an Extended Kalman Filter (EKF) for robust state estimation. Extensive experiments on public pedestrian datasets and a self-collected quadruped robot dataset demonstrate that X-IONet achieves state-of-the-art performance, reducing Absolute Trajectory Error (ATE) by 14.3% and Relative Trajectory Error (RTE) by 11.4% on pedestrian data, and by 52.8% and 41.3% on quadruped robot data. These results highlight the effectiveness of X-IONet in advancing accurate and robust inertial navigation across both human and legged robot platforms.
>
---
#### [new 027] Virtual Traffic Lights for Multi-Robot Navigation: Decentralized Planning with Centralized Conflict Resolution
- **分类: cs.RO**

- **简介: 该论文提出一种混合多机器人协调框架，解决集中式规划僵化与分布式规划易冲突的问题。通过分布式路径规划+集中式虚拟红绿灯冲突仲裁，仅下发停止指令而非完整路径，提升成功率并减少死锁，已在仿真与真实机器人中验证。**

- **链接: []()**

> **作者:** Sagar Gupta; Thanh Vinh Nguyen; Thieu Long Phan; Vidul Attri; Archit Gupta; Niroshinie Fernando; Kevin Lee; Seng W. Loke; Ronny Kutadinata; Benjamin Champion; Akansel Cosgun
>
> **摘要:** We present a hybrid multi-robot coordination framework that combines decentralized path planning with centralized conflict resolution. In our approach, each robot autonomously plans its path and shares this information with a centralized node. The centralized system detects potential conflicts and allows only one of the conflicting robots to proceed at a time, instructing others to stop outside the conflicting area to avoid deadlocks. Unlike traditional centralized planning methods, our system does not dictate robot paths but instead provides stop commands, functioning as a virtual traffic light. In simulation experiments with multiple robots, our approach increased the success rate of robots reaching their goals while reducing deadlocks. Furthermore, we successfully validated the system in real-world experiments with two quadruped robots and separately with wheeled Duckiebots.
>
---
#### [new 028] ViPRA: Video Prediction for Robot Actions
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: ViPRA提出一种预训练-微调框架，利用无动作标注的视频学习机器人连续控制。通过视频语言模型预测视觉观测与物理一致的潜在动作，再用少量演示映射为机器人动作，实现高效泛化与高频控制。**

- **链接: []()**

> **作者:** Sandeep Routray; Hengkai Pan; Unnat Jain; Shikhar Bahl; Deepak Pathak
>
> **备注:** Website: https://vipra-project.github.io
>
> **摘要:** Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present Video Prediction for Robot Actions (ViPRA), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict both future visual observations and motion-centric latent actions, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We will release models and code at https://vipra-project.github.io
>
---
#### [new 029] USV Obstacles Detection and Tracking in Marine Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文面向无人水面艇（USV）在海洋环境中的障碍物检测与跟踪任务，提出一种融合相机与LiDAR的混合方法，评估并优化了原有系统在真实数据集上的性能，实现高精度实时环境感知。**

- **链接: []()**

> **作者:** Yara AlaaEldin; Enrico Simetti; Francesca Odone
>
> **摘要:** Developing a robust and effective obstacle detection and tracking system for Unmanned Surface Vehicle (USV) at marine environments is a challenging task. Research efforts have been made in this area during the past years by GRAAL lab at the university of Genova that resulted in a methodology for detecting and tracking obstacles on the image plane and, then, locating them in the 3D LiDAR point cloud. In this work, we continue on the developed system by, firstly, evaluating its performance on recently published marine datasets. Then, we integrate the different blocks of the system on ROS platform where we could test it in real-time on synchronized LiDAR and camera data collected in various marine conditions available in the MIT marine datasets. We present a thorough experimental analysis of the results obtained using two approaches; one that uses sensor fusion between the camera and LiDAR to detect and track the obstacles and the other uses only the LiDAR point cloud for the detection and tracking. In the end, we propose a hybrid approach that merges the advantages of both approaches to build an informative obstacles map of the surrounding environment to the USV.
>
---
#### [new 030] Dual-MPC Footstep Planning for Robust Quadruped Locomotion
- **分类: cs.RO**

- **简介: 该论文提出一种双模型预测控制（Dual-MPC）步态规划方法，解决四足机器人因忽略角速度导致的体姿态不稳定问题，通过协同优化足端位置与地面反力，实现更鲁棒、低振荡的复杂地形行走。**

- **链接: []()**

> **作者:** Byeong-Il Ham; Hyun-Bin Kim; Jeonguk Kang; Keun Ha Choi; Kyung-Soo Kim
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** In this paper, we propose a footstep planning strategy based on model predictive control (MPC) that enables robust regulation of body orientation against undesired body rotations by optimizing footstep placement. Model-based locomotion approaches typically adopt heuristic methods or planning based on the linear inverted pendulum model. These methods account for linear velocity in footstep planning, while excluding angular velocity, which leads to angular momentum being handled exclusively via ground reaction force (GRF). Footstep planning based on MPC that takes angular velocity into account recasts the angular momentum control problem as a dual-input approach that coordinates GRFs and footstep placement, instead of optimizing GRFs alone, thereby improving tracking performance. A mutual-feedback loop couples the footstep planner and the GRF MPC, with each using the other's solution to iteratively update footsteps and GRFs. The use of optimal solutions reduces body oscillation and enables extended stance and swing phases. The method is validated on a quadruped robot, demonstrating robust locomotion with reduced oscillations, longer stance and swing phases across various terrains.
>
---
#### [new 031] Occlusion-Aware Ground Target Search by a UAV in an Urban Environment
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究城市环境中无人机搜索移动目标的任务，解决遮挡导致感知失效问题。提出基于概率可见体积与迭代深化A*的路径规划方法，通过最大化观测概率提升搜索效率，优于传统方法，尤其在高误报场景下。**

- **链接: []()**

> **作者:** Collin Hague; Artur Wolek
>
> **备注:** 18 pages, 18 figures, 5 tables
>
> **摘要:** This paper considers the problem of searching for a point of interest (POI) moving along an urban road network with an uncrewed aerial vehicle (UAV). The UAV is modeled as a variable-speed Dubins vehicle with a line-of-sight sensor in an urban environment that may occlude the sensor's view of the POI. A search strategy is proposed that exploits a probabilistic visibility volume (VV) to plan its future motion with iterative deepening $A^\ast$. The probabilistic VV is a time-varying three-dimensional representation of the sensing constraints for a particular distribution of the POI's state. To find the path most likely to view the POI, the planner uses a heuristic to optimistically estimate the probability of viewing the POI over a time horizon. The probabilistic VV is max-pooled to create a variable-timestep planner that reduces the search space and balances long-term and short-term planning. The proposed path planning method is compared to prior work with a Monte-Carlo simulation and is shown to outperform the baseline methods in cluttered environments when the UAV's sensor has a higher false alarm probability.
>
---
#### [new 032] A QP Framework for Improving Data Collection: Quantifying Device-Controller Performance in Robot Teleoperation
- **分类: cs.RO**

- **简介: 该论文面向机器人遥操作数据采集任务，解决不同设备与控制器组合导致的数据质量差异问题，提出一种基于QP的自适应控制器，通过阻抗跟踪与零空间优化提升轨迹精度与平滑性，量化设备-控制器性能对数据质量的影响。**

- **链接: []()**

> **作者:** Yuxuan Zhao; Yuanchen Tang; Jindi Zhang; Hongyu Yu
>
> **摘要:** Robot learning empowers the robot system with human brain-like intelligence to autonomously acquire and adapt skills through experience, enhancing flexibility and adaptability in various environments. Aimed at achieving a similar level of capability in large language models (LLMs) for embodied intelligence, data quality plays a crucial role in training a foundational model with diverse robot skills. In this study, we investigate the collection of data for manipulation tasks using teleoperation devices. Different devices yield varying effects when paired with corresponding controller strategies, including position-based inverse kinematics (IK) control, torque-based inverse dynamics (ID) control, and optimization-based compliance control. In this paper, we develop a teleoperation pipeline that is compatible with different teleoperation devices and manipulator controllers. Within the pipeline, we construct the optimal QP formulation with the dynamic nullspace and the impedance tracking as the novel optimal controller to achieve compliant pose tracking and singularity avoidance. Regarding the optimal controller, it adaptively adjusts the weights assignment depending on the robot joint manipulability that reflects the state of joint configuration for the pose tracking in the form of impedance control and singularity avoidance with nullspace tracking. Analysis of quantitative experimental results suggests the quality of the teleoperated trajectory data, including tracking error, occurrence of singularity, and the smoothness of the joints' trajectory, with different combinations of teleoperation interface and the motion controller.
>
---
#### [new 033] SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI; cs.CV; cs.GR; eess.SY**

- **简介: 论文提出SONIC，通过大规模扩展模型、数据与算力，将运动追踪作为通用任务，构建能自然控制人形机器人的基础模型，支持多模态输入与实时任务执行，无需人工奖励设计。**

- **链接: []()**

> **作者:** Zhengyi Luo; Ye Yuan; Tingwu Wang; Chenran Li; Sirui Chen; Fernando Castañeda; Zi-Ang Cao; Jiefeng Li; David Minor; Qingwei Ben; Xingye Da; Runyu Ding; Cyrus Hogg; Lina Song; Edy Lim; Eugene Jeong; Tairan He; Haoru Xue; Wenli Xiao; Zi Wang; Simon Yuen; Jan Kautz; Yan Chang; Umar Iqbal; Linxi "Jim" Fan; Yuke Zhu
>
> **备注:** Project page: https://nvlabs.github.io/SONIC/
>
> **摘要:** Despite the rise of billion-parameter foundation models trained across thousands of GPUs, similar scaling gains have not been shown for humanoid control. Current neural controllers for humanoids remain modest in size, target a limited behavior set, and are trained on a handful of GPUs over several days. We show that scaling up model capacity, data, and compute yields a generalist humanoid controller capable of creating natural and robust whole-body movements. Specifically, we posit motion tracking as a natural and scalable task for humanoid control, leverageing dense supervision from diverse motion-capture data to acquire human motion priors without manual reward engineering. We build a foundation model for motion tracking by scaling along three axes: network size (from 1.2M to 42M parameters), dataset volume (over 100M frames, 700 hours of high-quality motion data), and compute (9k GPU hours). Beyond demonstrating the benefits of scale, we show the practical utility of our model through two mechanisms: (1) a real-time universal kinematic planner that bridges motion tracking to downstream task execution, enabling natural and interactive control, and (2) a unified token space that supports various motion input interfaces, such as VR teleoperation devices, human videos, and vision-language-action (VLA) models, all using the same policy. Scaling motion tracking exhibits favorable properties: performance improves steadily with increased compute and data diversity, and learned representations generalize to unseen motions, establishing motion tracking at scale as a practical foundation for humanoid control.
>
---
#### [new 034] Dynamic Sparsity: Challenging Common Sparsity Assumptions for Learning World Models in Robotic Reinforcement Learning Benchmarks
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文挑战了机器人强化学习中动态模型的通用稀疏性假设，通过分析MuJoCo环境的真实动力学，发现稀疏性是状态依赖且局部时变的，而非全局稳定，呼吁构建更贴合实际的归纳偏差。**

- **链接: []()**

> **作者:** Muthukumar Pandaram; Jakob Hollenstein; David Drexel; Samuele Tosatto; Antonio Rodríguez-Sánchez; Justus Piater
>
> **摘要:** The use of learned dynamics models, also known as world models, can improve the sample efficiency of reinforcement learning. Recent work suggests that the underlying causal graphs of such dynamics models are sparsely connected, with each of the future state variables depending only on a small subset of the current state variables, and that learning may therefore benefit from sparsity priors. Similarly, temporal sparsity, i.e. sparsely and abruptly changing local dynamics, has also been proposed as a useful inductive bias. In this work, we critically examine these assumptions by analyzing ground-truth dynamics from a set of robotic reinforcement learning environments in the MuJoCo Playground benchmark suite, aiming to determine whether the proposed notions of state and temporal sparsity actually tend to hold in typical reinforcement learning tasks. We study (i) whether the causal graphs of environment dynamics are sparse, (ii) whether such sparsity is state-dependent, and (iii) whether local system dynamics change sparsely. Our results indicate that global sparsity is rare, but instead the tasks show local, state-dependent sparsity in their dynamics and this sparsity exhibits distinct structures, appearing in temporally localized clusters (e.g., during contact events) and affecting specific subsets of state dimensions. These findings challenge common sparsity prior assumptions in dynamics learning, emphasizing the need for grounded inductive biases that reflect the state-dependent sparsity structure of real-world dynamics.
>
---
#### [new 035] ARGUS: A Framework for Risk-Aware Path Planning in Tactical UGV Operations
- **分类: eess.SY; cs.RO**

- **简介: ARGUS是一个面向战术无人地面载具（UGV）的风险感知路径规划框架，旨在将战场态势与指挥意图转化为安全高效的动态路径。通过融合地形与威胁数据，实现任务目标与风险的平衡，并支持实时重规划与系统集成。**

- **链接: []()**

> **作者:** Nuno Soares; António Grilo
>
> **摘要:** This thesis presents the development of ARGUS, a framework for mission planning for Unmanned Ground Vehicles (UGVs) in tactical environments. The system is designed to translate battlefield complexity and the commander's intent into executable action plans. To this end, ARGUS employs a processing pipeline that takes as input geospatial terrain data, military intelligence on existing threats and their probable locations, and mission priorities defined by the commander. Through a set of integrated modules, the framework processes this information to generate optimized trajectories that balance mission objectives against the risks posed by threats and terrain characteristics. A fundamental capability of ARGUS is its dynamic nature, which allows it to adapt plans in real-time in response to unforeseen events, reflecting the fluid nature of the modern battlefield. The system's interoperability were validated in a practical exercise with the Portuguese Army, where it was successfully demonstrated that the routes generated by the model can be integrated and utilized by UGV control systems. The result is a decision support tool that not only produces an optimal trajectory but also provides the necessary insights for its execution, thereby contributing to greater effectiveness and safety in the employment of autonomous ground systems.
>
---
#### [new 036] Probabilistic Safety Guarantee for Stochastic Control Systems Using Average Reward MDPs
- **分类: eess.SY; cs.LG; cs.RO; math.OC**

- **简介: 该论文针对随机控制系统的安全保证问题，将安全约束满足转化为平均回报MDP优化任务，提出新算法通过线性规划高效计算高置信度安全策略，并在双积分器与倒立摆上验证其优越性。**

- **链接: []()**

> **作者:** Saber Omidi; Marek Petrik; Se Young Yoon; Momotaz Begum
>
> **备注:** Submitted to the Learning for Dynamics & Control (L4DC) 2026 conference
>
> **摘要:** Safety in stochastic control systems, which are subject to random noise with a known probability distribution, aims to compute policies that satisfy predefined operational constraints with high confidence throughout the uncertain evolution of the state variables. The unpredictable evolution of state variables poses a significant challenge for meeting predefined constraints using various control methods. To address this, we present a new algorithm that computes safe policies to determine the safety level across a finite state set. This algorithm reduces the safety objective to the standard average reward Markov Decision Process (MDP) objective. This reduction enables us to use standard techniques, such as linear programs, to compute and analyze safe policies. We validate the proposed method numerically on the Double Integrator and the Inverted Pendulum systems. Results indicate that the average-reward MDPs solution is more comprehensive, converges faster, and offers higher quality compared to the minimum discounted-reward solution.
>
---
#### [new 037] Statistically Assuring Safety of Control Systems using Ensembles of Safety Filters and Conformal Prediction
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文针对学习型控制系统的安全保证问题，提出基于共形预测与集成HJ值函数的安全滤波框架，为学习得到的策略提供概率性安全保证，解决传统方法计算昂贵且学习结果不可靠的问题。**

- **链接: []()**

> **作者:** Ihab Tabbara; Yuxuan Yang; Hussein Sibai
>
> **摘要:** Safety assurance is a fundamental requirement for deploying learning-enabled autonomous systems. Hamilton-Jacobi (HJ) reachability analysis is a fundamental method for formally verifying safety and generating safe controllers. However, computing the HJ value function that characterizes the backward reachable set (BRS) of a set of user-defined failure states is computationally expensive, especially for high-dimensional systems, motivating the use of reinforcement learning approaches to approximate the value function. Unfortunately, a learned value function and its corresponding safe policy are not guaranteed to be correct. The learned value function evaluated at a given state may not be equal to the actual safety return achieved by following the learned safe policy. To address this challenge, we introduce a conformal prediction-based (CP) framework that bounds such uncertainty. We leverage CP to provide probabilistic safety guarantees when using learned HJ value functions and policies to prevent control systems from reaching failure states. Specifically, we use CP to calibrate the switching between the unsafe nominal controller and the learned HJ-based safe policy and to derive safety guarantees under this switched policy. We also investigate using an ensemble of independently trained HJ value functions as a safety filter and compare this ensemble approach to using individual value functions alone.
>
---
#### [new 038] An Image-Based Path Planning Algorithm Using a UAV Equipped with Stereo Vision
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种基于双目视觉的无人机图像路径规划算法，解决二维图像无法识别地形凹凸（如陨石坑、山丘）导致路径不安全的问题。通过立体匹配生成深度图，结合视觉特征检测与ArUco标记定位，实现自动路径规划，并与A*、PRM算法对比验证有效性。**

- **链接: []()**

> **作者:** Selim Ahmet Iz; Mustafa Unel
>
> **摘要:** This paper presents a novel image-based path planning algorithm that was developed using computer vision techniques, as well as its comparative analysis with well-known deterministic and probabilistic algorithms, namely A* and Probabilistic Road Map algorithm (PRM). The terrain depth has a significant impact on the calculated path safety. The craters and hills on the surface cannot be distinguished in a two-dimensional image. The proposed method uses a disparity map of the terrain that is generated by using a UAV. Several computer vision techniques, including edge, line and corner detection methods, as well as the stereo depth reconstruction technique, are applied to the captured images and the found disparity map is used to define candidate way-points of the trajectory. The initial and desired points are detected automatically using ArUco marker pose estimation and circle detection techniques. After presenting the mathematical model and vision techniques, the developed algorithm is compared with well-known algorithms on different virtual scenes created in the V-REP simulation program and a physical setup created in a laboratory environment. Results are promising and demonstrate effectiveness of the proposed algorithm.
>
---
#### [new 039] Work-in-Progress: Function-as-Subtask API Replacing Publish/Subscribe for OS-Native DAG Scheduling
- **分类: cs.OS; cs.RO**

- **简介: 该论文提出Function-as-Subtask（FasS）API，替代ROS 2的发布/订阅机制，以强制保证DAG任务的优先级语义。通过函数式接口绑定输入输出边，消除人为约定依赖，实现内核级DAG调度，并为Linux sched_ext提供设计指导。**

- **链接: []()**

> **作者:** Takahiro Ishikawa-Aso; Atsushi Yano; Yutaro Kobayashi; Takumi Jin; Yuuki Takano; Shinpei Kato
>
> **备注:** 4 pages, 6 figures. Accepted for IEEE RTSS 2025; this is the author-accepted manuscript
>
> **摘要:** The Directed Acyclic Graph (DAG) task model for real-time scheduling finds its primary practical target in Robot Operating System 2 (ROS 2). However, ROS 2's publish/subscribe API leaves DAG precedence constraints unenforced: a callback may publish mid-execution, and multi-input callbacks let developers choose topic-matching policies. Thus preserving DAG semantics relies on conventions; once violated, the model collapses. We propose the Function-as-Subtask (FasS) API, which expresses each subtask as a function whose arguments/return values are the subtask's incoming/outgoing edges. By minimizing description freedom, DAG semantics is guaranteed at the API rather than by programmer discipline. We implement a DAG-native scheduler using FasS on a Rust-based experimental kernel and evaluate its semantic fidelity, and we outline design guidelines for applying FasS to Linux Linux sched_ext.
>
---
#### [new 040] Multistep Quasimetric Learning for Scalable Goal-conditioned Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文面向长周期目标条件强化学习任务，解决时序距离估计难题，提出多步拟度量学习方法，融合蒙特卡洛返回与局部更新，首次在真实机器人操作中实现端到端的多步拼接，显著提升长 horizon 性能。**

- **链接: []()**

> **作者:** Bill Chunyuan Zheng; Vivek Myers; Benjamin Eysenbach; Sergey Levine
>
> **摘要:** Learning how to reach goals in an environment is a longstanding challenge in AI, yet reasoning over long horizons remains a challenge for modern methods. The key question is how to estimate the temporal distance between pairs of observations. While temporal difference methods leverage local updates to provide optimality guarantees, they often perform worse than Monte Carlo methods that perform global updates (e.g., with multi-step returns), which lack such guarantees. We show how these approaches can be integrated into a practical GCRL method that fits a quasimetric distance using a multistep Monte-Carlo return. We show our method outperforms existing GCRL methods on long-horizon simulated tasks with up to 4000 steps, even with visual observations. We also demonstrate that our method can enable stitching in the real-world robotic manipulation domain (Bridge setup). Our approach is the first end-to-end GCRL method that enables multistep stitching in this real-world manipulation domain from an unlabeled offline dataset of visual observations.
>
---
## 更新

#### [replaced 001] Learning Vision-Based Neural Network Controllers with Semi-Probabilistic Safety Guarantees
- **分类: cs.RO; cs.AI**

- **链接: []()**

> **作者:** Xinhang Ma; Junlin Wu; Hussein Sibai; Yiannis Kantaros; Yevgeniy Vorobeychik
>
> **摘要:** Ensuring safety in autonomous systems with vision-based control remains a critical challenge due to the high dimensionality of image inputs and the fact that the relationship between true system state and its visual manifestation is unknown. Existing methods for learning-based control in such settings typically lack formal safety guarantees. To address this challenge, we introduce a novel semi-probabilistic verification framework that integrates reachability analysis with conditional generative networks and distribution-free tail bounds to enable efficient and scalable verification of vision-based neural network controllers. Next, we develop a gradient-based training approach that employs a novel safety loss function, safety-aware data-sampling strategy to efficiently select and store critical training examples, and curriculum learning, to efficiently synthesize safe controllers in the semi-probabilistic framework. Empirical evaluations in X-Plane 11 airplane landing simulation, CARLA-simulated autonomous lane following, F1Tenth vehicle lane following in a physical visually-rich miniature environment, and Airsim-simulated drone navigation and obstacle avoidance demonstrate the effectiveness of our method in achieving formal safety guarantees while maintaining strong nominal performance.
>
---
#### [replaced 002] MiniBEE: A New Form Factor for Compact Bimanual Dexterity
- **分类: cs.RO**

- **链接: []()**

> **作者:** Sharfin Islam; Zewen Chen; Zhanpeng He; Swapneel Bhatt; Andres Permuy; Brock Taylor; James Vickery; Zhengbin Lu; Cheng Zhang; Pedro Piacenza; Matei Ciocarlie
>
> **摘要:** Bimanual robot manipulators can achieve impressive dexterity, but typically rely on two full six- or seven- degree-of-freedom arms so that paired grippers can coordinate effectively. This traditional framework increases system complexity while only exploiting a fraction of the overall workspace for dexterous interaction. We introduce the MiniBEE (Miniature Bimanual End-effector), a compact system in which two reduced-mobility arms (3+ DOF each) are coupled into a kinematic chain that preserves full relative positioning between grippers. To guide our design, we formulate a kinematic dexterity metric that enlarges the dexterous workspace while keeping the mechanism lightweight and wearable. The resulting system supports two complementary modes: (i) wearable kinesthetic data collection with self-tracked gripper poses, and (ii) deployment on a standard robot arm, extending dexterity across its entire workspace. We present kinematic analysis and design optimization methods for maximizing dexterous range, and demonstrate an end-to-end pipeline in which wearable demonstrations train imitation learning policies that perform robust, real-world bimanual manipulation.
>
---
#### [replaced 003] Towards Adaptive Humanoid Control via Multi-Behavior Distillation and Reinforced Fine-Tuning
- **分类: cs.RO**

- **链接: []()**

> **作者:** Yingnan Zhao; Xinmiao Wang; Dewei Wang; Xinzhe Liu; Dan Lu; Qilong Han; Peng Liu; Chenjia Bai
>
> **摘要:** Humanoid robots are promising to learn a diverse set of human-like locomotion behaviors, including standing up, walking, running, and jumping. However, existing methods predominantly require training independent policies for each skill, yielding behavior-specific controllers that exhibit limited generalization and brittle performance when deployed on irregular terrains and in diverse situations. To address this challenge, we propose Adaptive Humanoid Control (AHC) that adopts a two-stage framework to learn an adaptive humanoid locomotion controller across different skills and terrains. Specifically, we first train several primary locomotion policies and perform a multi-behavior distillation process to obtain a basic multi-behavior controller, facilitating adaptive behavior switching based on the environment. Then, we perform reinforced fine-tuning by collecting online feedback in performing adaptive behaviors on more diverse terrains, enhancing terrain adaptability for the controller. We conduct experiments in both simulation and real-world experiments in Unitree G1 robots. The results show that our method exhibits strong adaptability across various situations and terrains. Project website: https://ahc-humanoid.github.io.
>
---
#### [replaced 004] Token Is All You Need: Cognitive Planning through Belief-Intent Co-Evolution
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: []()**

> **作者:** Shiyao Sang
>
> **备注:** 7 pages, 3 figures. A paradigm shift from reconstructing the world to understanding it: planning through belief-intent co-evolution
>
> **摘要:** We challenge the long-standing assumption that exhaustive scene modeling is required for high-performance end-to-end autonomous driving (E2EAD). Inspired by cognitive science, we propose that effective planning arises not from reconstructing the world, but from the co-evolution of belief and intent within a minimal set of semantically rich tokens. Experiments on the nuPlan benchmark (720 scenarios, 11k+ samples) reveal three principles: (1) sparse intent tokens alone achieve 0.487 m ADE, demonstrating strong performance without future prediction; (2) conditioning trajectory decoding on predicted future tokens reduces ADE to 0.382 m, a 21.6% improvement, showing that performance emerges from cognitive planning; and (3) explicit reconstruction loss degrades performance, confirming that task-driven belief-intent co-evolution suffices under reliable perception inputs. Crucially, we observe the emergence of cognitive consistency: through prolonged training, the model spontaneously develops stable token dynamics that balance current perception (belief) and future goals (intent). This process, accompanied by "temporal fuzziness," enables robustness under uncertainty and continuous self-optimization. Our work establishes a new paradigm: intelligence lies not in pixel fidelity, but in the tokenized duality of belief and intent. By reframing planning as understanding rather than reaction, TIWM bridges the gap between world models and VLA systems, paving the way for foresightful agents that plan through imagination. Note: Numerical comparisons with methods reporting results on nuScenes are indicative only, as nuPlan presents a more challenging planning-focused evaluation.
>
---
#### [replaced 005] When To Seek Help: Trust-Aware Assistance Seeking in Human-Supervised Autonomy
- **分类: cs.RO**

- **链接: []()**

> **作者:** Dong Hae Mangalindan; Ericka Rovira; Vaibhav Srivastava
>
> **摘要:** Our goal is to model and experimentally assess trust evolution to predict future beliefs and behaviors of human-robot teams in dynamic environments. Research suggests that maintaining trust among team members in a human-robot team is vital for successful team performance. Research suggests that trust is a multi-dimensional and latent entity that relates to past experiences and future actions in a complex manner. Employing a human-robot collaborative task, we design an optimal assistance-seeking strategy for the robot using a POMDP framework. In the task, the human supervises an autonomous mobile manipulator collecting objects in an environment. The supervisor's task is to ensure that the robot safely executes its task. The robot can either choose to attempt to collect the object or seek human assistance. The human supervisor actively monitors the robot's activities, offering assistance upon request, and intervening if they perceive the robot may fail. In this setting, human trust is the hidden state, and the primary objective is to optimize team performance. We execute two sets of human-robot interaction experiments. The data from the first experiment are used to estimate POMDP parameters, which are used to compute an optimal assistance-seeking policy evaluated in the second experiment. The estimated POMDP parameters reveal that, for most participants, human intervention is more probable when trust is low, particularly in high-complexity tasks. Our estimates suggest that the robot's action of asking for assistance in high-complexity tasks can positively impact human trust. Our experimental results show that the proposed trust-aware policy is better than an optimal trust-agnostic policy. By comparing model estimates of human trust, obtained using only behavioral data, with the collected self-reported trust values, we show that model estimates are isomorphic to self-reported responses.
>
---
#### [replaced 006] 3D Cal: An Open-Source Software Library for Calibrating Tactile Sensors
- **分类: cs.RO**

- **链接: []()**

> **作者:** Rohan Kota; Kaival Shah; J. Edward Colgate; Gregory Reardon
>
> **摘要:** Tactile sensing plays a key role in enabling dexterous and reliable robotic manipulation, but realizing this capability requires substantial calibration to convert raw sensor readings into physically meaningful quantities. Despite its near-universal necessity, the calibration process remains ad hoc and labor-intensive. Here, we introduce 3D Cal, an open-source library that transforms a low-cost 3D printer into an automated probing device capable of generating large volumes of labeled training data for tactile sensor calibration. We demonstrate the utility of 3D Cal by calibrating two commercially available vision-based tactile sensors, DIGIT and GelSight Mini, to reconstruct high-quality depth maps using the collected data and a custom convolutional neural network. In addition, we perform a data ablation study to determine how much data is needed for accurate calibration, providing practical guidelines for researchers working with these specific sensors, and we benchmark the trained models on previously unseen objects to evaluate calibration accuracy and generalization performance. By automating tactile sensor calibration, 3D Cal can accelerate tactile sensing research, simplify sensor deployment, and promote the practical integration of tactile sensing in robotic platforms.
>
---
#### [replaced 007] Heuristic Adaptation of Potentially Misspecified Domain Support for Likelihood-Free Inference in Stochastic Dynamical Systems
- **分类: cs.RO; cs.LG**

- **链接: []()**

> **作者:** Georgios Kamaras; Craig Innes; Subramanian Ramamoorthy
>
> **备注:** 20 pages, 18 figures, algorithm lines cleveref fixed for pdflatex 2025
>
> **摘要:** In robotics, likelihood-free inference (LFI) can provide the domain distribution that adapts a learnt agent in a parametric set of deployment conditions. LFI assumes an arbitrary support for sampling, which remains constant as the initial generic prior is iteratively refined to more descriptive posteriors. However, a potentially misspecified support can lead to suboptimal, yet falsely certain, posteriors. To address this issue, we propose three heuristic LFI variants: EDGE, MODE, and CENTRE. Each interprets the posterior mode shift over inference steps in its own way and, when integrated into an LFI step, adapts the support alongside posterior inference. We first expose the support misspecification issue and evaluate our heuristics using stochastic dynamical benchmarks. We then evaluate the impact of heuristic support adaptation on parameter inference and policy learning for a dynamic deformable linear object (DLO) manipulation task. Inference results in a finer length and stiffness classification for a parametric set of DLOs. When the resulting posteriors are used as domain distributions for sim-based policy learning, they lead to more robust object-centric agent performance.
>
---
#### [replaced 008] On the Surprising Effectiveness of Spectral Clipping in Learning Stable Linear and Latent-Linear Dynamical Systems
- **分类: cs.RO; eess.SY**

- **链接: []()**

> **作者:** Hanyao Guo; Yunhai Han; Harish Ravichandar
>
> **摘要:** When learning stable linear dynamical systems from data, three important properties are desirable: i) predictive accuracy, ii) verifiable stability, and iii) computational efficiency. Unconstrained minimization of prediction errors leads to high accuracy and efficiency but cannot guarantee stability. Existing methods to enforce stability often preserve accuracy, but do so only at the cost of increased computation. In this work, we investigate if a seemingly-naive procedure can simultaneously offer all three desiderata. Specifically, we consider a post-hoc procedure in which we surgically manipulate the spectrum of the linear system after it was learned using unconstrained least squares. We call this approach spectral clipping (SC) as it involves eigen decomposition and subsequent reconstruction of the system matrix after any eigenvalues whose magnitude exceeds one have been clipped to one (without altering the eigenvectors). We also show that SC can be readily combined with Koopman operators to learn nonlinear dynamical systems that can generate stable predictions of nonlinear phenomena, such as those underlying complex dexterous manipulation skills involving multi-fingered robotic hands. Through comprehensive experiments involving two different applications and publicly available benchmark datasets, we show that this simple technique can efficiently learn highly-accurate predictive dynamics that are provably-stable. Notably, we find that SC can match or outperform strong baselines while being orders-of-magnitude faster. Finally, we find that SC can learn stable robot policies even when the training data includes unsuccessful or truncated demonstrations. Our code and datasets can be found at https://github.com/GT-STAR-Lab/spec_clip.
>
---
#### [replaced 009] JaxRobotarium: Training and Deploying Multi-Robot Policies in 10 Minutes
- **分类: cs.RO; cs.LG; cs.MA**

- **链接: []()**

> **作者:** Shalin Anand Jain; Jiazhen Liu; Siva Kailas; Harish Ravichandar
>
> **备注:** 22 pages, 14 figures, 10 tables. https://github.com/GT-STAR-Lab/JaxRobotarium. Manuscript accepted for publication at the 9th Conference on Robot Learning (CoRL 2025), Seoul, Korea
>
> **摘要:** Multi-agent reinforcement learning (MARL) has emerged as a promising solution for learning complex and scalable coordination behaviors in multi-robot systems. However, established MARL platforms (e.g., SMAC and MPE) lack robotics relevance and hardware deployment, leaving multi-robot learning researchers to develop bespoke environments and hardware testbeds dedicated to the development and evaluation of their individual contributions. The Multi-Agent RL Benchmark and Learning Environment for the Robotarium (MARBLER) is an exciting recent step in providing a standardized robotics-relevant platform for MARL, by bridging the Robotarium testbed with existing MARL software infrastructure. However, MARBLER lacks support for parallelization and GPU/TPU execution, making the platform prohibitively slow compared to modern MARL environments and hindering adoption. We contribute JaxRobotarium, a Jax-powered end-to-end simulation, learning, deployment, and benchmarking platform for the Robotarium. JaxRobotarium enables rapid training and deployment of multi-robot RL (MRRL) policies with realistic robot dynamics and safety constraints, supporting parallelization and hardware acceleration. Our generalizable learning interface integrates easily with SOTA MARL libraries (e.g., JaxMARL). In addition, JaxRobotarium includes eight standardized coordination scenarios, including four novel scenarios that bring established MARL benchmark tasks (e.g., RWARE and Level-Based Foraging) to a robotics setting. We demonstrate that JaxRobotarium retains high simulation fidelity while achieving dramatic speedups over baseline (20x in training and 150x in simulation), and provides an open-access sim-to-real evaluation pipeline through the Robotarium testbed, accelerating and democratizing access to multi-robot learning research and evaluation. Our code is available at https://github.com/GT-STAR-Lab/JaxRobotarium.
>
---
#### [replaced 010] QP Chaser: Polynomial Trajectory Generation for Autonomous Aerial Tracking
- **分类: cs.RO; eess.SY**

- **链接: []()**

> **作者:** Yunwoo Lee; Jungwon Park; Seungwoo Jung; Boseong Jeon; Dahyun Oh; H. Jin Kim
>
> **备注:** 18 pages, 16 figures
>
> **摘要:** Maintaining the visibility of the target is one of the major objectives of aerial tracking missions. This paper proposes a target-visible trajectory planning pipeline using quadratic programming. Our approach can handle various tracking settings, including single and dual target following and both static and dynamic environments, unlike other works that focus on a single specific setup. In contrast to other studies that fully trust the predicted trajectory of the target and consider only the visibility of the center of the target, our pipeline considers error in target path prediction and the entire body of the target to maintain the target visibility robustly. First, a prediction module uses a sample-check strategy to quickly calculate the reachable areas of moving objects, which represent the areas their bodies can reach, considering obstacles. Subsequently, the planning module formulates a single QP problem, considering path homotopy, to generate a tracking trajectory that maximizes the visibility of the target's reachable area among obstacles. The performance of the planner is validated in multiple scenarios, through high-fidelity simulations and real-world experiments.
>
---
#### [replaced 011] SCoTT: Strategic Chain-of-Thought Tasking for Wireless-Aware Robot Navigation in Digital Twins
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **链接: []()**

> **作者:** Aladin Djuhera; Amin Seffo; Vlad C. Andrei; Holger Boche; Walid Saad
>
> **摘要:** Path planning under wireless performance constraints is a complex challenge in robot navigation. However, naively incorporating such constraints into classical planning algorithms often incurs prohibitive search costs. In this paper, we propose SCoTT, a wireless-aware path planning framework that leverages vision-language models (VLMs) to co-optimize average path gains and trajectory length using wireless heatmap images and ray-tracing data from a digital twin (DT). At the core of our framework is Strategic Chain-of-Thought Tasking (SCoTT), a novel prompting paradigm that decomposes the exhaustive search problem into structured subtasks, each solved via chain-of-thought prompting. To establish strong baselines, we compare classical A* and wireless-aware extensions of it, and derive DP-WA*, an optimal, iterative dynamic programming algorithm that incorporates all path gains and distance metrics from the DT, but at significant computational cost. In extensive experiments, we show that SCoTT achieves path gains within 2% of DP-WA* while consistently generating shorter trajectories. Moreover, SCoTT's intermediate outputs can be used to accelerate DP-WA* by reducing its search space, saving up to 62% in execution time. We validate our framework using four VLMs, demonstrating effectiveness across both large and small models, thus making it applicable to a wide range of compact models at low inference cost. We also show the practical viability of our approach by deploying SCoTT as a ROS node within Gazebo simulations. Finally, we discuss data acquisition pipelines, compute requirements, and deployment considerations for VLMs in 6G-enabled DTs, underscoring the potential of natural language interfaces for wireless-aware navigation in real-world applications.
>
---
#### [replaced 012] Residual Rotation Correction using Tactile Equivariance
- **分类: cs.RO**

- **链接: []()**

> **作者:** Yizhe Zhu; Zhang Ye; Boce Hu; Haibo Zhao; Yu Qi; Dian Wang; Robert Platt
>
> **备注:** 8 pages
>
> **摘要:** Visuotactile policy learning augments vision-only policies with tactile input, facilitating contact-rich manipulation. However, the high cost of tactile data collection makes sample efficiency the key requirement for developing visuotactile policies. We present EquiTac, a framework that exploits the inherent SO(2) symmetry of in-hand object rotation to improve sample efficiency and generalization for visuotactile policy learning. EquiTac first reconstructs surface normals from raw RGB inputs of vision-based tactile sensors, so rotations of the normal vector field correspond to in-hand object rotations. An SO(2)-equivariant network then predicts a residual rotation action that augments a base visuomotor policy at test time, enabling real-time rotation correction without additional reorientation demonstrations. On a real robot, EquiTac accurately achieves robust zero-shot generalization to unseen in-hand orientations with very few training samples, where baselines fail even with more training data. To our knowledge, this is the first tactile learning method to explicitly encode tactile equivariance for policy learning, yielding a lightweight, symmetry-aware module that improves reliability in contact-rich tasks.
>
---
#### [replaced 013] Keep on Going: Learning Robust Humanoid Motion Skills via Selective Adversarial Training
- **分类: cs.RO**

- **链接: []()**

> **作者:** Yang Zhang; Zhanxiang Cao; Buqing Nie; Haoyang Li; Zhong Jiangwei; Qiao Sun; Xiaoyi Hu; Xiaokang Yang; Yue Gao
>
> **备注:** 13 pages, 10 figures, AAAI2026
>
> **摘要:** Humanoid robots are expected to operate reliably over long horizons while executing versatile whole-body skills. Yet Reinforcement Learning (RL) motion policies typically lose stability under prolonged operation, sensor/actuator noise, and real world disturbances. In this work, we propose a Selective Adversarial Attack for Robust Training (SA2RT) to enhance the robustness of motion skills. The adversary is learned to identify and sparsely perturb the most vulnerable states and actions under an attack-budget constraint, thereby exposing true weakness without inducing conservative overfitting. The resulting non-zero sum, alternating optimization continually strengthens the motion policy against the strongest discovered attacks. We validate our approach on the Unitree G1 humanoid robot across perceptive locomotion and whole-body control tasks. Experimental results show that adversarially trained policies improve the terrain traversal success rate by 40%, reduce the trajectory tracking error by 32%, and maintain long horizon mobility and tracking performance. Together, these results demonstrate that selective adversarial attacks are an effective driver for learning robust, long horizon humanoid motion skills.
>
---
#### [replaced 014] Tackling the Kidnapped Robot Problem via Sparse Feasible Hypothesis Sampling and Reliable Batched Multi-Stage Inference
- **分类: cs.RO**

- **链接: []()**

> **作者:** Muhua Zhang; Lei Ma; Ying Wu; Kai Shen; Deqing Huang; Henry Leung
>
> **备注:** 10 pages, 8 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper addresses the Kidnapped Robot Problem (KRP), a core localization challenge of relocalizing a robot in a known map without prior pose estimate when localization loss or at SLAM initialization. For this purpose, a passive 2-D global relocalization framework is proposed. It estimates the global pose efficiently and reliably from a single LiDAR scan and an occupancy grid map while the robot remains stationary, thereby enhancing the long-term autonomy of mobile robots. The proposed framework casts global relocalization as a non-convex problem and solves it via the multi-hypothesis scheme with batched multi-stage inference and early termination, balancing completeness and efficiency. The Rapidly-exploring Random Tree (RRT), under traversability constraints, asymptotically covers the reachable space to generate sparse, uniformly distributed feasible positional hypotheses, fundamentally reducing the sampling space. The hypotheses are preliminarily ordered by the proposed Scan Mean Absolute Difference (SMAD), a coarse beam-error level metric that facilitates the early termination by prioritizing high-likelihood candidates. The SMAD computation is optimized for non-panoramic scans. The Translation-Affinity Scan-to-Map Alignment Metric (TAM) is proposed for reliable orientation selection at hypothesized positions and accurate final pose evaluation to mitigate degradation in conventional likelihood-field metrics under translational uncertainty induced by sparse hypotheses, as well as non-panoramic LiDAR scan and environmental changes. Real-world experiments on a resource-constrained mobile robot with non-panoramic LiDAR scans show that the proposed framework achieves competitive performance in both global relocalization success rate and computational efficiency.
>
---
#### [replaced 015] EndoIR: Degradation-Agnostic All-in-One Endoscopic Image Restoration via Noise-Aware Routing Diffusion
- **分类: eess.IV; cs.AI; cs.CV; cs.RO**

- **链接: []()**

> **作者:** Tong Chen; Xinyu Ma; Long Bai; Wenyang Wang; Yue Sun; Luping Zhou
>
> **摘要:** Endoscopic images often suffer from diverse and co-occurring degradations such as low lighting, smoke, and bleeding, which obscure critical clinical details. Existing restoration methods are typically task-specific and often require prior knowledge of the degradation type, limiting their robustness in real-world clinical use. We propose EndoIR, an all-in-one, degradation-agnostic diffusion-based framework that restores multiple degradation types using a single model. EndoIR introduces a Dual-Domain Prompter that extracts joint spatial-frequency features, coupled with an adaptive embedding that encodes both shared and task-specific cues as conditioning for denoising. To mitigate feature confusion in conventional concatenation-based conditioning, we design a Dual-Stream Diffusion architecture that processes clean and degraded inputs separately, with a Rectified Fusion Block integrating them in a structured, degradation-aware manner. Furthermore, Noise-Aware Routing Block improves efficiency by dynamically selecting only noise-relevant features during denoising. Experiments on SegSTRONG-C and CEC datasets demonstrate that EndoIR achieves state-of-the-art performance across multiple degradation scenarios while using fewer parameters than strong baselines, and downstream segmentation experiments confirm its clinical utility.
>
---
#### [replaced 016] Uncertainty-Aware Active Source Tracking of Marine Pollution using Unmanned Surface Vehicles
- **分类: cs.RO**

- **链接: []()**

> **作者:** Song Ma; Yanchao Wang; Richard Bucknall; Yuanchang Liu
>
> **摘要:** This paper proposes an uncertainty-aware marine pollution source tracking framework for unmanned surface vehicles (USVs). By integrating high-fidelity marine pollution dispersion simulation with informative path planning techniques, we demonstrate effective identification of pollution sources in marine environments. The proposed approach is implemented based on Robot Operating System (ROS), processing real-time sensor data to update probabilistic source location estimates. The system progressively refines the estimation of source location while quantifying uncertainty levels in its predictions. Experiments conducted in simulated environments with varying source locations, wave conditions, and starting positions demonstrate the framework's ability to localise pollution sources with high accuracy. Results show that the proposed approach achieves reliable source localisation efficiently and outperforms the existing baseline. This work contributes to the development of full autonomous environmental monitoring capabilities essential for rapid response to marine pollution incidents.
>
---
#### [replaced 017] Towards Embodied Agentic AI: Review and Classification of LLM- and VLM-Driven Robot Autonomy and Interaction
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: []()**

> **作者:** Sahar Salimpour; Lei Fu; Farhad Keramat; Leonardo Militano; Giovanni Toffetti; Harry Edelman; Jorge Peña Queralta
>
> **摘要:** Foundation models, including large language models (LLMs) and vision-language models (VLMs), have recently enabled novel approaches to robot autonomy and human-robot interfaces. In parallel, vision-language-action models (VLAs) or large behavior models (LBMs) are increasing the dexterity and capabilities of robotic systems. This survey paper reviews works that advance agentic applications and architectures, including initial efforts with GPT-style interfaces and more complex systems where AI agents function as coordinators, planners, perception actors, or generalist interfaces. Such agentic architectures allow robots to reason over natural language instructions, invoke APIs, plan task sequences, or assist in operations and diagnostics. In addition to peer-reviewed research, due to the fast-evolving nature of the field, we highlight and include community-driven projects, ROS packages, and industrial frameworks that show emerging trends. We propose a taxonomy for classifying model integration approaches and present a comparative analysis of the role that agents play in different solutions in today's literature.
>
---
#### [replaced 018] Hestia: Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction
- **分类: cs.RO; cs.CV**

- **链接: []()**

> **作者:** Cheng-You Lu; Zhuoli Zhuang; Nguyen Thanh Trung Le; Da Xiao; Yu-Cheng Chang; Thomas Do; Srinath Sridhar; Chin-teng Lin
>
> **摘要:** Advances in 3D reconstruction and novel view synthesis have enabled efficient and photorealistic rendering. However, images for reconstruction are still either largely manual or constrained by simple preplanned trajectories. To address this issue, recent works propose generalizable next-best-view planners that do not require online learning. Nevertheless, robustness and performance remain limited across various shapes. Hence, this study introduces Voxel-Face-Aware Hierarchical Next-Best-View Acquisition for Efficient 3D Reconstruction (Hestia), which addresses the shortcomings of the reinforcement learning-based generalizable approaches for five-degree-of-freedom viewpoint prediction. Hestia systematically improves the planners through four components: a more diverse dataset to promote robustness, a hierarchical structure to manage the high-dimensional continuous action search space, a close-greedy strategy to mitigate spurious correlations, and a face-aware design to avoid overlooking geometry. Experimental results show that Hestia achieves non-marginal improvements, with at least a 4% gain in coverage ratio, while reducing Chamfer Distance by 50% and maintaining real-time inference. In addition, Hestia outperforms prior methods by at least 12% in coverage ratio with a 5-image budget and remains robust to object placement variations. Finally, we demonstrate that Hestia, as a next-best-view planner, is feasible for the real-world application. Our project page is https://johnnylu305.github.io/hestia web.
>
---
#### [replaced 019] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors
- **分类: cs.RO**

- **链接: []()**

> **作者:** Haoyu Zhao; Linghao Zhuang; Xingyue Zhao; Cheng Zeng; Haoran Xu; Yuming Jiang; Jun Cen; Kexiang Wang; Jiayan Guo; Siteng Huang; Xin Li; Deli Zhao; Hua Zou
>
> **备注:** AAAI 2026
>
> **摘要:** A dexterous hand capable of generalizable grasping objects is fundamental for the development of general-purpose embodied AI. However, previous methods focus narrowly on low-level grasp stability metrics, neglecting affordance-aware positioning and human-like poses which are crucial for downstream manipulation. To address these limitations, we propose AffordDex, a novel framework with two-stage training that learns a universal grasping policy with an inherent understanding of both motion priors and object affordances. In the first stage, a trajectory imitator is pre-trained on a large corpus of human hand motions to instill a strong prior for natural movement. In the second stage, a residual module is trained to adapt these general human-like motions to specific object instances. This refinement is critically guided by two components: our Negative Affordance-aware Segmentation (NAA) module, which identifies functionally inappropriate contact regions, and a privileged teacher-student distillation process that ensures the final vision-based policy is highly successful. Extensive experiments demonstrate that AffordDex not only achieves universal dexterous grasping but also remains remarkably human-like in posture and functionally appropriate in contact location. As a result, AffordDex significantly outperforms state-of-the-art baselines across seen objects, unseen instances, and even entirely novel categories.
>
---
#### [replaced 020] Accelerating Visual-Policy Learning through Parallel Differentiable Simulation
- **分类: cs.LG; cs.RO**

- **链接: []()**

> **作者:** Haoxiang You; Yilang Liu; Ian Abraham
>
> **摘要:** In this work, we propose a computationally efficient algorithm for visual policy learning that leverages differentiable simulation and first-order analytical policy gradients. Our approach decouple the rendering process from the computation graph, enabling seamless integration with existing differentiable simulation ecosystems without the need for specialized differentiable rendering software. This decoupling not only reduces computational and memory overhead but also effectively attenuates the policy gradient norm, leading to more stable and smoother optimization. We evaluate our method on standard visual control benchmarks using modern GPU-accelerated simulation. Experiments show that our approach significantly reduces wall-clock training time and consistently outperforms all baseline methods in terms of final returns. Notably, on complex tasks such as humanoid locomotion, our method achieves a $4\times$ improvement in final return, and successfully learns a humanoid running policy within 4 hours on a single GPU.
>
---
