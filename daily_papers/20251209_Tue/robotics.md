# 机器人 cs.RO

- **最新发布 77 篇**

- **更新 42 篇**

## 最新发布

#### [new 001] From Zero to High-Speed Racing: An Autonomous Racing Stack
- **分类: cs.RO**

- **简介: 该论文针对高速全尺寸自动驾驶赛车的挑战，提出并迭代优化了自主赛车系统ARS，涵盖定位、感知、规划与控制。工作包括系统架构设计、多场景性能评估及高速多传感器数据集发布，支撑自动驾驶技术在极端条件下的验证与进步。**

- **链接: [https://arxiv.org/pdf/2512.06892v1](https://arxiv.org/pdf/2512.06892v1)**

> **作者:** Hassan Jardali; Durgakant Pushp; Youwei Yu; Mahmoud Ali; Ihab S. Mohamed; Alejandro Murillo-Gonzalez; Paul D. Coen; Md. Al-Masrur Khan; Reddy Charan Pulivendula; Saeoul Park; Lingchuan Zhou; Lantao Liu
>
> **摘要:** High-speed, head-to-head autonomous racing presents substantial technical and logistical challenges, including precise localization, rapid perception, dynamic planning, and real-time control-compounded by limited track access and costly hardware. This paper introduces the Autonomous Race Stack (ARS), developed by the IU Luddy Autonomous Racing team for the Indy Autonomous Challenge (IAC). We present three iterations of our ARS, each validated on different tracks and achieving speeds up to 260 km/h. Our contributions include: (i) the modular architecture and evolution of the ARS across ARS1, ARS2, and ARS3; (ii) a detailed performance evaluation that contrasts control, perception, and estimation across oval and road-course environments; and (iii) the release of a high-speed, multi-sensor dataset collected from oval and road-course tracks. Our findings highlight the unique challenges and insights from real-world high-speed full-scale autonomous racing.
>
---
#### [new 002] See Once, Then Act: Vision-Language-Action Model with Task Learning from One-Shot Video Demonstrations
- **分类: cs.RO**

- **简介: 该论文研究机器人操作任务，旨在解决现有视觉-语言-动作模型泛化能力不足的问题。提出ViVLA模型，通过单次专家演示视频学习新技能，并构建大规模配对数据集训练，实现跨任务、跨形态的有效迁移，在仿真和真实场景中均显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.07582v1](https://arxiv.org/pdf/2512.07582v1)**

> **作者:** Guangyan Chen; Meiling Wang; Qi Shao; Zichen Zhou; Weixin Mao; Te Cui; Minzhao Zhu; Yinan Deng; Luojie Yang; Zhanqi Zhang; Yi Yang; Hua Chen; Yufeng Yue
>
> **摘要:** Developing robust and general-purpose manipulation policies represents a fundamental objective in robotics research. While Vision-Language-Action (VLA) models have demonstrated promising capabilities for end-to-end robot control, existing approaches still exhibit limited generalization to tasks beyond their training distributions. In contrast, humans possess remarkable proficiency in acquiring novel skills by simply observing others performing them once. Inspired by this capability, we propose ViVLA, a generalist robotic manipulation policy that achieves efficient task learning from a single expert demonstration video at test time. Our approach jointly processes an expert demonstration video alongside the robot's visual observations to predict both the demonstrated action sequences and subsequent robot actions, effectively distilling fine-grained manipulation knowledge from expert behavior and transferring it seamlessly to the agent. To enhance the performance of ViVLA, we develop a scalable expert-agent pair data generation pipeline capable of synthesizing paired trajectories from easily accessible human videos, further augmented by curated pairs from publicly available datasets. This pipeline produces a total of 892,911 expert-agent samples for training ViVLA. Experimental results demonstrate that our ViVLA is able to acquire novel manipulation skills from only a single expert demonstration video at test time. Our approach achieves over 30% improvement on unseen LIBERO tasks and maintains above 35% gains with cross-embodiment videos. Real-world experiments demonstrate effective learning from human videos, yielding more than 38% improvement on unseen tasks.
>
---
#### [new 003] Efficient Computation of a Continuous Topological Model of the Configuration Space of Tethered Mobile Robots
- **分类: cs.RO**

- **简介: 该论文研究系绳移动机器人的路径规划问题，旨在解决现有方法依赖离散构型空间、忽略连续性与拓扑信息的问题。作者提出一种基于工作空间多边形表示的连续拓扑模型构建方法，通过关联万有覆盖空间，设计算法生成单纯复形模型，显著提升计算效率与规划灵活性。**

- **链接: [https://arxiv.org/pdf/2512.07303v1](https://arxiv.org/pdf/2512.07303v1)**

> **作者:** Gianpietro Battocletti; Dimitris Boskos; Bart De Schutter
>
> **备注:** 7 pages, 3 figures, submitted to IFAC World Congress 2026
>
> **摘要:** Despite the attention that the problem of path planning for tethered robots has garnered in the past few decades, the approaches proposed to solve it typically rely on a discrete representation of the configuration space and do not exploit a model that can simultaneously capture the topological information of the tether and the continuous location of the robot. In this work, we explicitly build a topological model of the configuration space of a tethered robot starting from a polygonal representation of the workspace where the robot moves. To do so, we first establish a link between the configuration space of the tethered robot and the universal covering space of the workspace, and then we exploit this link to develop an algorithm to compute a simplicial complex model of the configuration space. We show how this approach improves the performances of existing algorithms that build other types of representations of the configuration space. The proposed model can be computed in a fraction of the time required to build traditional homotopy-augmented graphs, and is continuous, allowing to solve the path planning task for tethered robots using a broad set of path planning algorithms.
>
---
#### [new 004] Ground Compliance Improves Retention of Visual Feedback-Based Propulsion Training for Gait Rehabilitation
- **分类: cs.RO**

- **简介: 该论文研究步态康复中视觉反馈结合地面顺应性对推进力训练效果的影响。通过split-belt treadmill实验，发现加入地面顺应性可增强推动力建立及保持，促进更稳定的运动学习，有助于中风等患者长期康复。**

- **链接: [https://arxiv.org/pdf/2512.06897v1](https://arxiv.org/pdf/2512.06897v1)**

> **作者:** Bradley Hobbs; Panagiotis Artemiadis
>
> **摘要:** This study investigates whether adding ground compliance to visual feedback (VF) gait training is more effective at increasing push-off force (POF) compared to using VF alone, with implications for gait rehabilitation. Ten healthy participants walked on a custom split-belt treadmill. All participants received real-time visual feedback of their ground reaction forces. One group also experienced changes in ground compliance, while a control group received only visual feedback. Intentional increases in propulsive ground reaction forces (POF) were successfully achieved and sustained post-intervention, especially in the group that experienced ground compliance. This group also demonstrated lasting after-effects in muscle activity and joint kinematics, indicating a more robust learning of natural strategies to increase propulsion. This work demonstrates how visual and proprioceptive systems coordinate during gait adaptation. It uniquely shows that combining ground compliance with visual feedback enhances the learning of propulsive forces, supporting the potential use of compliant terrain in long-term rehabilitation targeting propulsion deficits, such as those following a stroke.
>
---
#### [new 005] Where to Fly, What to Send: Communication-Aware Aerial Support for Ground Robots
- **分类: cs.RO**

- **简介: 该论文研究通信受限下空中机器人如何为地面机器人提供信息支持。通过价值感知的信息选择、混合整数规划传输量与效用导向探索，优化通信内容与飞行路径，平衡地图传输量与地面导航成本。**

- **链接: [https://arxiv.org/pdf/2512.06207v1](https://arxiv.org/pdf/2512.06207v1)**

> **作者:** Harshil Suthar; Dipankar Maity
>
> **备注:** Submitted to conference
>
> **摘要:** In this work we consider a multi-robot team operating in an unknown environment where one aerial agent is tasked to map the environment and transmit (a portion of) the mapped environment to a group of ground agents that are trying to reach their goals. The entire operation takes place over a bandwidth-limited communication channel, which motivates the problem of determining what and how much information the assisting agent should transmit and when while simultaneously performing exploration/mapping. The proposed framework enables the assisting aerial agent to decide what information to transmit based on the Value-of-Information (VoI), how much to transmit using a Mixed-Integer Linear Programming (MILP), and how to acquire additional information through an utility score-based environment exploration strategy. We perform a communication-motion trade-off analysis between the total amount of map data communicated by the aerial agent and the navigation cost incurred by the ground agents.
>
---
#### [new 006] Fault Tolerant Control of Mecanum Wheeled Mobile Robots
- **分类: cs.RO**

- **简介: 该论文研究Mecanum轮移动机器人的容错控制，旨在解决执行器完全与部分故障（如扭矩退化）导致的性能下降问题。提出基于后验概率估计的容错控制策略，通过概率加权融合多故障控制律，提升系统鲁棒性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.06444v1](https://arxiv.org/pdf/2512.06444v1)**

> **作者:** Xuehui Ma; Shiliang Zhang; Zhiyong Sun
>
> **摘要:** Mecanum wheeled mobile robots (MWMRs) are highly susceptible to actuator faults that degrade performance and risk mission failure. Current fault tolerant control (FTC) schemes for MWMRs target complete actuator failures like motor stall, ignoring partial faults e.g., in torque degradation. We propose an FTC strategy handling both fault types, where we adopt posterior probability to learn real-time fault parameters. We derive the FTC law by aggregating probability-weighed control laws corresponding to predefined faults. This ensures the robustness and safety of MWMR control despite varying levels of fault occurrence. Simulation results demonstrate the effectiveness of our FTC under diverse scenarios.
>
---
#### [new 007] A Hetero-Associative Sequential Memory Model Utilizing Neuromorphic Signals: Validated on a Mobile Manipulator
- **分类: cs.RO**

- **简介: 该论文提出一种基于脉冲神经信号的异联想记忆模型，用于移动机械臂的低功耗动作学习。通过关节状态与触觉的二值化绑定及旋转位置嵌入，实现力控伪柔顺与多关节抓取序列的联想记忆与泛化。**

- **链接: [https://arxiv.org/pdf/2512.07032v1](https://arxiv.org/pdf/2512.07032v1)**

> **作者:** Runcong Wang; Fengyi Wang; Gordon Cheng
>
> **摘要:** This paper presents a hetero-associative sequential memory system for mobile manipulators that learns compact, neuromorphic bindings between robot joint states and tactile observations to produce step-wise action decisions with low compute and memory cost. The method encodes joint angles via population place coding and converts skin-measured forces into spike-rate features using an Izhikevich neuron model; both signals are transformed into bipolar binary vectors and bound element-wise to create associations stored in a large-capacity sequential memory. To improve separability in binary space and inject geometry from touch, we introduce 3D rotary positional embeddings that rotate subspaces as a function of sensed force direction, enabling fuzzy retrieval through a softmax weighted recall over temporally shifted action patterns. On a Toyota Human Support Robot covered by robot skin, the hetero-associative sequential memory system realizes a pseudocompliance controller that moves the link under touch in the direction and with speed correlating to the amplitude of applied force, and it retrieves multi-joint grasp sequences by continuing tactile input. The system sets up quickly, trains from synchronized streams of states and observations, and exhibits a degree of generalization while remaining economical. Results demonstrate single-joint and full-arm behaviors executed via associative recall, and suggest extensions to imitation learning, motion planning, and multi-modal integration.
>
---
#### [new 008] Affordance Field Intervention: Enabling VLAs to Escape Memory Traps in Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在分布外场景中陷入“记忆陷阱”、缺乏空间推理的问题，提出一种轻量级混合框架AFI。通过引入3D空间可供性场（SAF）提供几何线索，检测并跳出记忆轨迹，引导VLA生成更鲁棒的操作动作，提升在新环境中的适应能力。**

- **链接: [https://arxiv.org/pdf/2512.07472v1](https://arxiv.org/pdf/2512.07472v1)**

> **作者:** Siyu Xu; Zijian Wang; Yunke Wang; Chenghao Xia; Tao Huang; Chang Xu
>
> **摘要:** Vision-Language-Action (VLA) models have shown great performance in robotic manipulation by mapping visual observations and language instructions directly to actions. However, they remain brittle under distribution shifts: when test scenarios change, VLAs often reproduce memorized trajectories instead of adapting to the updated scene, which is a failure mode we refer to as the "Memory Trap". This limitation stems from the end-to-end design, which lacks explicit 3D spatial reasoning and prevents reliable identification of actionable regions in unfamiliar environments. To compensate for this missing spatial understanding, 3D Spatial Affordance Fields (SAFs) can provide a geometric representation that highlights where interactions are physically feasible, offering explicit cues about regions the robot should approach or avoid. We therefore introduce Affordance Field Intervention (AFI), a lightweight hybrid framework that uses SAFs as an on-demand plug-in to guide VLA behavior. Our system detects memory traps through proprioception, repositions the robot to recent high-affordance regions, and proposes affordance-driven waypoints that anchor VLA-generated actions. A SAF-based scorer then selects trajectories with the highest cumulative affordance. Extensive experiments demonstrate that our method achieves an average improvement of 23.5% across different VLA backbones ($π_{0}$ and $π_{0.5}$) under out-of-distribution scenarios on real-world robotic platforms, and 20.2% on the LIBERO-Pro benchmark, validating its effectiveness in enhancing VLA robustness to distribution shifts.
>
---
#### [new 009] Time-Varying Formation Tracking Control of Wheeled Mobile Robots With Region Constraint: A Generalized Udwadia-Kalaba Framework
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究带区域约束的轮式移动机器人时变编队跟踪控制。针对有向通信拓扑，将控制目标与区域约束转化为受限动力学问题，基于广义Udwadia-Kalaba框架设计控制器，确保机器人安全与编队性能，并通过仿真验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.07137v1](https://arxiv.org/pdf/2512.07137v1)**

> **作者:** Kang Yijie; Hao Yuqing; Wang Qingyun; Chen Guanrong
>
> **备注:** 10 pages,9 figures
>
> **摘要:** In this paper, the time-varying formation tracking control of wheeled mobile robots with region constraint is investigated from a generalized Udwadia-Kalaba framework. The communication topology is directed, weighted and has a spanning tree with the leader being the root. By reformulating the time-varying formation tracking control objective as a constrained equation and transforming the region constraint by a diffeomorphism, the time-varying formation tracking controller with the region constraint is designed under the generalized Udwadia-Kalaba framework. Compared with the existing works on time-varying formation tracking control, the region constraint is takeninto account in this paper, which ensures the safety of the robots.Finally, some numerical simulations are presented to illustrate the effectiveness of the proposed control strategy.
>
---
#### [new 010] CERNet: Class-Embedding Predictive-Coding RNN for Unified Robot Motion, Recognition, and Confidence Estimation
- **分类: cs.RO**

- **简介: 该论文提出CERNet模型，解决机器人运动生成、行为识别与置信度估计的统一建模问题。基于预测编码RNN与类嵌入向量，实现生成与推理双模式运行，支持实时动作生成与意图识别，并通过内部预测误差评估置信度。**

- **链接: [https://arxiv.org/pdf/2512.07041v1](https://arxiv.org/pdf/2512.07041v1)**

> **作者:** Hiroki Sawada; Alexandre Pitti; Mathias Quoy
>
> **摘要:** Robots interacting with humans must not only generate learned movements in real-time, but also infer the intent behind observed behaviors and estimate the confidence of their own inferences. This paper proposes a unified model that achieves all three capabilities within a single hierarchical predictive-coding recurrent neural network (PC-RNN) equipped with a class embedding vector, CERNet, which leverages a dynamically updated class embedding vector to unify motor generation and recognition. The model operates in two modes: generation and inference. In the generation mode, the class embedding constrains the hidden state dynamics to a class-specific subspace; in the inference mode, it is optimized online to minimize prediction error, enabling real-time recognition. Validated on a humanoid robot across 26 kinesthetically taught alphabets, our hierarchical model achieves 76% lower trajectory reproduction error than a parameter-matched single-layer baseline, maintains motion fidelity under external perturbations, and infers the demonstrated trajectory class online with 68% Top-1 and 81% Top-2 accuracy. Furthermore, internal prediction errors naturally reflect the model's confidence in its recognition. This integration of robust generation, real-time recognition, and intrinsic uncertainty estimation within a compact PC-RNN framework offers a compact and extensible approach to motor memory in physical robots, with potential applications in intent-sensitive human-robot collaboration.
>
---
#### [new 011] db-LaCAM: Fast and Scalable Multi-Robot Kinodynamic Motion Planning with Discontinuity-Bounded Search and Lightweight MAPF
- **分类: cs.RO**

- **简介: 该论文研究多机器人运动规划，旨在解决现有动力学规划器扩展性差、速度慢的问题。作者提出db-LaCAM方法，结合轻量级MAPF与动力学感知的运动原语搜索，在保证解质量的同时显著提升规划速度与规模，支持多种机器人动力学，并在真实机器人上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2512.06796v1](https://arxiv.org/pdf/2512.06796v1)**

> **作者:** Akmaral Moldagalieva; Keisuke Okumura; Amanda Prorok; Wolfgang Hönig
>
> **摘要:** State-of-the-art multi-robot kinodynamic motion planners struggle to handle more than a few robots due to high computational burden, which limits their scalability and results in slow planning time. In this work, we combine the scalability and speed of modern multi-agent path finding (MAPF) algorithms with the dynamic-awareness of kinodynamic planners to address these limitations. To this end, we propose discontinuity-Bounded LaCAM (db-LaCAM), a planner that utilizes a precomputed set of motion primitives that respect robot dynamics to generate horizon-length motion sequences, while allowing a user-defined discontinuity between successive motions. The planner db-LaCAM is resolution-complete with respect to motion primitives and supports arbitrary robot dynamics. Extensive experiments demonstrate that db-LaCAM scales efficiently to scenarios with up to 50 robots, achieving up to ten times faster runtime compared to state-of-the-art planners, while maintaining comparable solution quality. The approach is validated in both 2D and 3D environments with dynamics such as the unicycle and 3D double integrator. We demonstrate the safe execution of trajectories planned with db-LaCAM in two distinct physical experiments involving teams of flying robots and car-with-trailer robots.
>
---
#### [new 012] Multi-Domain Motion Embedding: Expressive Real-Time Mimicry for Legged Robots
- **分类: cs.RO**

- **简介: 该论文研究机器人实时模仿人类/动物运动的任务，旨在解决现有方法忽略运动中周期性与随机性特征的问题。提出多域运动嵌入（MDME），结合小波编码与概率表示，实现跨形态、高保真、免重定向的实时运动复现。**

- **链接: [https://arxiv.org/pdf/2512.07673v1](https://arxiv.org/pdf/2512.07673v1)**

> **作者:** Matthias Heyrman; Chenhao Li; Victor Klemm; Dongho Kang; Stelian Coros; Marco Hutter
>
> **备注:** 15 pages
>
> **摘要:** Effective motion representation is crucial for enabling robots to imitate expressive behaviors in real time, yet existing motion controllers often ignore inherent patterns in motion. Previous efforts in representation learning do not attempt to jointly capture structured periodic patterns and irregular variations in human and animal movement. To address this, we present Multi-Domain Motion Embedding (MDME), a motion representation that unifies the embedding of structured and unstructured features using a wavelet-based encoder and a probabilistic embedding in parallel. This produces a rich representation of reference motions from a minimal input set, enabling improved generalization across diverse motion styles and morphologies. We evaluate MDME on retargeting-free real-time motion imitation by conditioning robot control policies on the learned embeddings, demonstrating accurate reproduction of complex trajectories on both humanoid and quadruped platforms. Our comparative studies confirm that MDME outperforms prior approaches in reconstruction fidelity and generalizability to unseen motions. Furthermore, we demonstrate that MDME can reproduce novel motion styles in real-time through zero-shot deployment, eliminating the need for task-specific tuning or online retargeting. These results position MDME as a generalizable and structure-aware foundation for scalable real-time robot imitation.
>
---
#### [new 013] Model-Less Feedback Control of Space-based Continuum Manipulators using Backbone Tension Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究空间连续体机械臂的无模型反馈控制，旨在解决传统建模方法因形变复杂、摩擦未建模等导致的控制不准确与不稳定问题。提出一种基于背脊张力优化的实时控制框架，通过在线更新雅可比矩阵和优化肌腱张力，实现高精度轨迹跟踪。**

- **链接: [https://arxiv.org/pdf/2512.06754v1](https://arxiv.org/pdf/2512.06754v1)**

> **作者:** Shrreya Rajneesh; Nikita Pavle; Rakesh Kumar Sahoo; Manoranjan Sinha
>
> **摘要:** Continuum manipulators offer intrinsic dexterity and safe geometric compliance for navigation within confined and obstacle-rich environments. However, their infinite-dimensional backbone deformation, unmodeled internal friction, and configuration-dependent stiffness fundamentally limit the reliability of model-based kinematic formulations, resulting in inaccurate Jacobian predictions, artificial singularities, and unstable actuation behavior. Motivated by these limitations, this work presents a complete model-less control framework that bypasses kinematic modeling by using an empirically initialized Jacobian refined online through differential convex updates. Tip motion is generated via a real-time quadratic program that computes actuator increments while enforcing tendon slack avoidance and geometric limits. A backbone tension optimization term is introduced in this paper to regulate axial loading and suppress co-activation compression. The framework is validated across circular, pentagonal, and square trajectories, demonstrating smooth convergence, stable tension evolution, and sub-millimeter steady-state accuracy without any model calibration or parameter identification. These results establish the proposed controller as a scalable alternative to model-dependent continuum manipulation in a constrained environment.
>
---
#### [new 014] SINRL: Socially Integrated Navigation with Reinforcement Learning using Spiking Neural Networks
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文研究机器人社交导航任务，旨在解决现有类脑方法在深度强化学习中训练不稳定的问题。提出一种混合架构SINRL，结合脉冲神经网络与人工神经网络，提升社交导航性能并显著降低能耗。**

- **链接: [https://arxiv.org/pdf/2512.07266v1](https://arxiv.org/pdf/2512.07266v1)**

> **作者:** Florian Tretter; Daniel Flögel; Alexandru Vasilache; Max Grobbel; Jürgen Becker; Sören Hohmann
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Integrating autonomous mobile robots into human environments requires human-like decision-making and energy-efficient, event-based computation. Despite progress, neuromorphic methods are rarely applied to Deep Reinforcement Learning (DRL) navigation approaches due to unstable training. We address this gap with a hybrid socially integrated DRL actor-critic approach that combines Spiking Neural Networks (SNNs) in the actor with Artificial Neural Networks (ANNs) in the critic and a neuromorphic feature extractor to capture temporal crowd dynamics and human-robot interactions. Our approach enhances social navigation performance and reduces estimated energy consumption by approximately 1.69 orders of magnitude.
>
---
#### [new 015] FedDSR: Federated Deep Supervision and Regularization Towards Autonomous Driving
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究联邦学习在自动驾驶中的应用，旨在解决非独立同分布数据导致的模型泛化差与收敛慢问题。提出FedDSR方法，通过中间层监督与正则化增强特征学习，提升性能与收敛速度，适用于多种模型与联邦算法。**

- **链接: [https://arxiv.org/pdf/2512.06676v1](https://arxiv.org/pdf/2512.06676v1)**

> **作者:** Wei-Bin Kou; Guangxu Zhu; Bingyang Cheng; Chen Zhang; Yik-Chung Wu; Jianping Wang
>
> **备注:** 9 pages
>
> **摘要:** Federated Learning (FL) enables collaborative training of autonomous driving (AD) models across distributed vehicles while preserving data privacy. However, FL encounters critical challenges such as poor generalization and slow convergence due to non-independent and identically distributed (non-IID) data from diverse driving environments. To overcome these obstacles, we introduce Federated Deep Supervision and Regularization (FedDSR), a paradigm that incorporates multi-access intermediate layer supervision and regularization within federated AD system. Specifically, FedDSR comprises following integral strategies: (I) to select multiple intermediate layers based on predefined architecture-agnostic standards. (II) to compute mutual information (MI) and negative entropy (NE) on those selected layers to serve as intermediate loss and regularizer. These terms are integrated into the output-layer loss to form a unified optimization objective, enabling comprehensive optimization across the network hierarchy. (III) to aggregate models from vehicles trained based on aforementioned rules of (I) and (II) to generate the global model on central server. By guiding and penalizing the learning of feature representations at intermediate stages, FedDSR enhances the model generalization and accelerates model convergence for federated AD. We then take the semantic segmentation task as an example to assess FedDSR and apply FedDSR to multiple model architectures and FL algorithms. Extensive experiments demonstrate that FedDSR achieves up to 8.93% improvement in mIoU and 28.57% reduction in training rounds, compared to other FL baselines, making it highly suitable for practical deployment in federated AD ecosystems.
>
---
#### [new 016] AMBER: Aerial deployable gripping crawler with compliant microspine for canopy manipulation
- **分类: cs.RO**

- **简介: 该论文提出AMBER，一种可空中部署的树冠爬行机器人，旨在解决复杂树冠环境下的移动与采样难题。通过微刺履带、双轨旋转夹持器和弹性尾部设计，实现稳定攀爬与低能耗运行，结合无人机部署系统，支持生态监测任务。**

- **链接: [https://arxiv.org/pdf/2512.07680v1](https://arxiv.org/pdf/2512.07680v1)**

> **作者:** P. A. Wigner; L. Romanello; A. Hammad; P. H. Nguyen; T. Lan; S. F. Armanini; B. B. Kocer; M. Kovac
>
> **摘要:** This paper presents an aerially deployable crawler designed for adaptive locomotion and manipulation within tree canopies. The system combines compliant microspine-based tracks, a dual-track rotary gripper, and an elastic tail, enabling secure attachment and stable traversal across branches of varying curvature and inclination. Experiments demonstrate reliable gripping up to 90 degrees of body roll and inclination, while effective climbing on branches inclined up to 67.5 degrees, achieving a maximum speed of 0.55 body lengths per second on horizontal branches. The compliant tracks allow yaw steering of up to 10 degrees, enhancing maneuverability on irregular surfaces. Power measurements show efficient operation with a dimensionless cost of transport over an order of magnitude lower than typical hovering power consumption in aerial robots. Integrated within a drone-tether deployment system, the crawler provides a robust, low-power platform for environmental sampling and in-canopy sensing, bridging the gap between aerial and surface-based ecological robotics.
>
---
#### [new 017] Model Predictive Control for Cooperative Docking Between Autonomous Surface Vehicles with Disturbance Rejection
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究无人水面艇（USV）间的协同靠泊任务，解决传统方法中单艇机动、效率低的问题。提出基于集中式模型预测控制（MPC）的协同方法，通过预测模型抑制干扰（如水流），实现快速、高效的双艇协同靠泊。**

- **链接: [https://arxiv.org/pdf/2512.07316v1](https://arxiv.org/pdf/2512.07316v1)**

> **作者:** Gianpietro Battocletti; Dimitris Boskos; Bart De Schutter
>
> **备注:** 7 pages, 4 figures, submitted to IFAC World Congress 2026
>
> **摘要:** Uncrewed Surface Vehicles (USVs) are a popular and efficient type of marine craft that find application in a large number of water-based tasks. When multiple USVs operate in the same area, they may be required to dock to each other to perform a shared task. Existing approaches for the docking between autonomous USVs generally consider one USV as a stationary target, while the second one is tasked to reach the required docking pose. In this work, we propose a cooperative approach for USV-USV docking, where two USVs work together to dock at an agreed location. We use a centralized Model Predictive Control (MPC) approach to solve the control problem, obtaining feasible trajectories that also guarantee constraint satisfaction. Owing to its model-based nature, this approach allows the rejection of disturbances, inclusive of exogenous inputs, by anticipating their effect on the USVs through the MPC prediction model. This is particularly effective in case of almost-stationary disturbances such as water currents. In simulations, we demonstrate how the proposed approach allows for a faster and more efficient docking with respect to existing approaches.
>
---
#### [new 018] Closed-Loop Robotic Manipulation of Transparent Substrates for Self-Driving Laboratories using Deep Learning Micro-Error Correction
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对自驱动实验室中透明基底操作易出错的问题，提出一种基于深度学习视觉纠错的闭环机器人操控方法ASHE，实现高精度自动装卸透明基底，显著提升实验自动化可靠性。**

- **链接: [https://arxiv.org/pdf/2512.06038v1](https://arxiv.org/pdf/2512.06038v1)**

> **作者:** Kelsey Fontenot; Anjali Gorti; Iva Goel; Tonio Buonassisi; Alexander E. Siemenn
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Self-driving laboratories (SDLs) have accelerated the throughput and automation capabilities for discovering and improving chemistries and materials. Although these SDLs have automated many of the steps required to conduct chemical and materials experiments, a commonly overlooked step in the automation pipeline is the handling and reloading of substrates used to transfer or deposit materials onto for downstream characterization. Here, we develop a closed-loop method of Automated Substrate Handling and Exchange (ASHE) using robotics, dual-actuated dispensers, and deep learning-driven computer vision to detect and correct errors in the manipulation of fragile and transparent substrates for SDLs. Using ASHE, we demonstrate a 98.5% first-time placement accuracy across 130 independent trials of reloading transparent glass substrates into an SDL, where only two substrate misplacements occurred and were successfully detected as errors and automatically corrected. Through the development of more accurate and reliable methods for handling various types of substrates, we move toward an improvement in the automation capabilities of self-driving laboratories, furthering the acceleration of novel chemical and materials discoveries.
>
---
#### [new 019] Real-Time Spatiotemporal Tubes for Dynamic Unsafe Sets
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对动态环境中非线性系统的实时避障与任务完成问题，提出一种实时时空管（STT）框架。通过在线调整状态空间中的时变球体，实现无需模型逼近的闭环控制，确保系统在规定时间内安全完成任务。**

- **链接: [https://arxiv.org/pdf/2512.06151v1](https://arxiv.org/pdf/2512.06151v1)**

> **作者:** Ratnangshu Das; Siddhartha Upadhyay; Pushpak Jagtap
>
> **摘要:** This paper presents a real-time control framework for nonlinear pure-feedback systems with unknown dynamics to satisfy reach-avoid-stay tasks within a prescribed time in dynamic environments. To achieve this, we introduce a real-time spatiotemporal tube (STT) framework. An STT is defined as a time-varying ball in the state space whose center and radius adapt online using only real-time sensory input. A closed-form, approximation-free control law is then derived to constrain the system output within the STT, ensuring safety and task satisfaction. We provide formal guarantees for obstacle avoidance and on-time task completion. The effectiveness and scalability of the framework are demonstrated through simulations and hardware experiments on a mobile robot and an aerial vehicle, navigating in cluttered dynamic environments.
>
---
#### [new 020] Cascaded Tightly-Coupled Observer Design for Single-Range-Aided Inertial Navigation
- **分类: cs.RO**

- **简介: 该论文研究单距离辅助的惯性导航问题，旨在仅用IMU、体框架矢量和单一距离测量实现全状态估计。提出级联紧耦合观测器，分步估计位置、速度、重力方向和姿态，具备几乎全局渐近稳定性，适用于轻量化自主导航。**

- **链接: [https://arxiv.org/pdf/2512.06198v1](https://arxiv.org/pdf/2512.06198v1)**

> **作者:** Oussama Sifour; Soulaimane Berkane; Abdelhamid Tayebi
>
> **备注:** 8 pages
>
> **摘要:** This work introduces a single-range-aided navigation observer that reconstructs the full state of a rigid body using only an Inertial Measurement Unit (IMU), a body-frame vector measurement (e.g., magnetometer), and a distance measurement from a fixed anchor point. The design first formulates an extended linear time-varying (LTV) system to estimate body-frame position, body-frame velocity, and the gravity direction. The recovered gravity direction, combined with the body-frame vector measurement, is then used to reconstruct the full orientation on $\mathrm{SO}(3)$, resulting in a cascaded observer architecture. Almost Global Asymptotic Stability (AGAS) of the cascaded design is established under a uniform observability condition, ensuring robustness to sensor noise and trajectory variations. Simulation studies on three-dimensional trajectories demonstrate accurate estimation of position, velocity, and orientation, highlighting single-range aiding as a lightweight and effective modality for autonomous navigation.
>
---
#### [new 021] Inchworm-Inspired Soft Robot with Groove-Guided Locomotion
- **分类: cs.RO**

- **简介: 该论文提出一种受尺蠖启发的软体机器人，通过3D打印基底上的沟槽图案被动控制其运动方向。采用单一轮式介电弹性体驱动器，简化设计并降低能耗，解决了复杂地形中定向控制需多驱动器导致系统复杂的问题，适用于搜救、管道检测等场景。**

- **链接: [https://arxiv.org/pdf/2512.07813v1](https://arxiv.org/pdf/2512.07813v1)**

> **作者:** Hari Prakash Thanabalan; Lars Bengtsson; Ugo Lafont; Giovanni Volpe
>
> **摘要:** Soft robots require directional control to navigate complex terrains. However, achieving such control often requires multiple actuators, which increases mechanical complexity, complicates control systems, and raises energy consumption. Here, we introduce an inchworm-inspired soft robot whose locomotion direction is controlled passively by patterned substrates. The robot employs a single rolled dielectric elastomer actuator, while groove patterns on a 3D-printed substrate guide its alignment and trajectory. Through systematic experiments, we demonstrate that varying groove angles enables precise control of locomotion direction without the need for complex actuation strategies. This groove-guided approach reduces energy consumption, simplifies robot design, and expands the applicability of bio-inspired soft robots in fields such as search and rescue, pipe inspection, and planetary exploration.
>
---
#### [new 022] Leveraging Port-Hamiltonian Theory for Impedance Control Benchmarking
- **分类: cs.RO; eess.SY**

- **简介: 该论文属机器人控制任务，旨在解决 impedance 控制缺乏统一评估标准的问题。提出基于端口-哈密顿理论的可微度量，无需力矩传感即可验证多自由度系统无源性，并定义保真度指标。通过仿真验证了方法在机械臂和四足腿上的有效性。**

- **链接: [https://arxiv.org/pdf/2512.06423v1](https://arxiv.org/pdf/2512.06423v1)**

> **作者:** Leonardo F. Dos Santos; Elisa G. Vergamini; Cícero Zanette; Lucca Maitan; Thiago Boaventura
>
> **备注:** This is the author's version of the paper accepted for publication in the 2025 International Conference on Advanced Robotics (ICAR). The final version will be available at IEEE Xplore
>
> **摘要:** This work proposes PH-based metrics for benchmarking impedance control. A causality-consistent PH model is introduced for mass-spring-damper impedance in Cartesian space. Based on this model, a differentiable, force-torque sensing-independent, n-DoF passivity condition is derived, valid for time-varying references. An impedance fidelity metric is also defined from step-response power in free motion, capturing dynamic decoupling. The proposed metrics are validated in Gazebo simulations with a six-DoF manipulator and a quadruped leg. Results demonstrate the suitability of the PH framework for standardized impedance control benchmarking.
>
---
#### [new 023] A New Trajectory-Oriented Approach to Enhancing Comprehensive Crowd Navigation Performance
- **分类: cs.RO**

- **简介: 该论文研究人群导航任务，针对现有方法忽略轨迹平滑性及评估不公的问题，提出统一评估框架和强调轨迹曲率优化的新奖励机制，提升轨迹质量与适应性，在多场景下实现优于现有方法的综合性能。**

- **链接: [https://arxiv.org/pdf/2512.06608v1](https://arxiv.org/pdf/2512.06608v1)**

> **作者:** Xinyu Zhou; Songhao Piao; Chao Gao; Liguo Chen
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Crowd navigation has garnered considerable research interest in recent years, especially with the proliferating application of deep reinforcement learning (DRL) techniques. Many studies, however, do not sufficiently analyze the relative priorities among evaluation metrics, which compromises the fair assessment of methods with divergent objectives. Furthermore, trajectory-continuity metrics, specifically those requiring $C^2$ smoothness, are rarely incorporated. Current DRL approaches generally prioritize efficiency and proximal comfort, often neglecting trajectory optimization or addressing it only through simplistic, unvalidated smoothness reward. Nevertheless, effective trajectory optimization is essential to ensure naturalness, enhance comfort, and maximize the energy efficiency of any navigation system. To address these gaps, this paper proposes a unified framework that enables the fair and transparent assessment of navigation methods by examining the prioritization and joint evaluation of multiple optimization objectives. We further propose a novel reward-shaping strategy that explicitly emphasizes trajectory-curvature optimization. The resulting trajectory quality and adaptability are significantly enhanced across multi-scale scenarios. Through extensive 2D and 3D experiments, we demonstrate that the proposed method achieves superior performance compared to state-of-the-art approaches.
>
---
#### [new 024] A Flexible Funnel-Shaped Robotic Hand with an Integrated Single-Sheet Valve for Milligram-Scale Powder Handling
- **分类: cs.RO**

- **简介: 该论文针对毫克制粉体自动化操作难题，提出一种带单片集成阀的柔性漏斗状机械手，结合流动模型与在线参数识别实现精准控制，提升称量精度与适应性，适用于实验室自动化中的粉末处理任务。**

- **链接: [https://arxiv.org/pdf/2512.07091v1](https://arxiv.org/pdf/2512.07091v1)**

> **作者:** Tomoya Takahashi; Yusaku Nakajima; Cristian Camilo Beltran-Hernandez; Yuki Kuroda; Kazutoshi Tanaka; Masashi Hamaya; Kanta Ono; Yoshitaka Ushiku
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Laboratory Automation (LA) has the potential to accelerate solid-state materials discovery by enabling continuous robotic operation without human intervention. While robotic systems have been developed for tasks such as powder grinding and X-ray diffraction (XRD) analysis, fully automating powder handling at the milligram scale remains a significant challenge due to the complex flow dynamics of powders and the diversity of laboratory tasks. To address this challenge, this study proposes a novel, funnel-shaped, flexible robotic hand that preserves the softness and conical sheet designs in prior work while incorporating a controllable valve at the cone apex to enable precise, incremental dispensing of milligram-scale powder quantities. The hand is integrated with an external balance through a feedback control system based on a model of powder flow and online parameter identification. Experimental evaluations with glass beads, monosodium glutamate, and titanium dioxide demonstrated that 80% of the trials achieved an error within 2 mg, and the maximum error observed was approximately 20 mg across a target range of 20 mg to 3 g. In addition, by incorporating flow prediction models commonly used for hoppers and performing online parameter identification, the system is able to adapt to variations in powder dynamics. Compared to direct PID control, the proposed model-based control significantly improved both accuracy and convergence speed. These results highlight the potential of the proposed system to enable efficient and flexible powder weighing, with scalability toward larger quantities and applicability to a broad range of laboratory automation tasks.
>
---
#### [new 025] Multi-Rigid-Body Approximation of Human Hands with Application to Digital Twin
- **分类: cs.RO; cs.GR**

- **简介: 该论文研究数字孪生中的人手建模任务，旨在解决高精度与实时仿真难以兼顾的问题。提出从动捕数据构建个性化多刚体手模型的 pipeline，通过闭式解和 BCH 修正迭代法将 MANO 模型映射到带解剖约束的 URDF 模型，实现外观真实且可实时物理仿真的数字手。**

- **链接: [https://arxiv.org/pdf/2512.07359v1](https://arxiv.org/pdf/2512.07359v1)**

> **作者:** Bin Zhao; Yiwen Lu; Haohua Zhu; Xiao Li; Sheng Yi
>
> **备注:** 10 pages, 4 figures. Accepted at ICBSR'25 (International Conference on Biomechanical Systems and Robotics)
>
> **摘要:** Human hand simulation plays a critical role in digital twin applications, requiring models that balance anatomical fidelity with computational efficiency. We present a complete pipeline for constructing multi-rigid-body approximations of human hands that preserve realistic appearance while enabling real-time physics simulation. Starting from optical motion capture of a specific human hand, we construct a personalized MANO (Multi-Abstracted hand model with Neural Operations) model and convert it to a URDF (Unified Robot Description Format) representation with anatomically consistent joint axes. The key technical challenge is projecting MANO's unconstrained SO(3) joint rotations onto the kinematically constrained joints of the rigid-body model. We derive closed-form solutions for single degree-of-freedom joints and introduce a Baker-Campbell-Hausdorff (BCH)-corrected iterative method for two degree-of-freedom joints that properly handles the non-commutativity of rotations. We validate our approach through digital twin experiments where reinforcement learning policies control the multi-rigid-body hand to replay captured human demonstrations. Quantitative evaluation shows sub-centimeter reconstruction error and successful grasp execution across diverse manipulation tasks.
>
---
#### [new 026] Parametric Design of a Cable-Driven Coaxial Spherical Parallel Mechanism for Ultrasound Scans
- **分类: cs.RO; physics.class-ph**

- **简介: 该论文设计了一种用于超声扫描的绳驱同轴球面并联机构，旨在解决医疗遥操作中力反馈装置在工作空间、刚度与惯量间的权衡问题。通过轻量化末端与解耦驱动，实现高精度、高响应的纯旋转运动，提升触觉反馈性能。**

- **链接: [https://arxiv.org/pdf/2512.06995v1](https://arxiv.org/pdf/2512.06995v1)**

> **作者:** Maryam Seraj; Mohammad Hossein Kamrava; Carlo Tiseo
>
> **摘要:** Haptic interfaces play a critical role in medical teleoperation by enabling surgeons to interact with remote environments through realistic force and motion feedback. Achieving high fidelity in such systems requires balancing performance trade-off among workspace, dexterity, stiffness, inertia, and bandwidth, particularly in applications demanding pure rotational motion. This paper presents the design methodology and kinematic analysis of a Cable-Driven Coaxial Spherical Parallel Mechanism (CDC-SPM) developed to address these challenges. The proposed cable-driven interface design allows for reducing the mass placed at the robot arm end-effector, thereby minimizing inertial loads, enhancing stiffness, and improving dynamic responsiveness. Through parallel and coaxial actuation, the mechanism achieves decoupled rotational degrees of freedom with isotropic force and torque transmission. Simulation and analysis demonstrate that the CDC-SPM provides accurate, responsive, and safe motion characteristics suitable for high-precision haptic applications. These results highlight the mechanism's potential for medical teleoperation tasks such as ultrasound imaging, where precise and intuitive manipulation is essential.
>
---
#### [new 027] MagicSkin: Balancing Marker and Markerless Modes in Vision-Based Tactile Sensors with a Translucent Skin
- **分类: cs.RO**

- **简介: 该论文提出MagicSkin，一种用于视觉触觉传感器的半透明标记皮肤，旨在解决标记与无标记模式间的性能权衡问题。它兼顾力与切向位移测量及表面细节保留，无需额外硬件，提升多任务感知性能。**

- **链接: [https://arxiv.org/pdf/2512.06829v1](https://arxiv.org/pdf/2512.06829v1)**

> **作者:** Oluwatimilehin Tijani; Zhuo Chen; Jiankang Deng; Shan Luo
>
> **备注:** Submitted to ICRA2026
>
> **摘要:** Vision-based tactile sensors (VBTS) face a fundamental trade-off in marker and markerless design on the tactile skin: opaque ink markers enable measurement of force and tangential displacement but completely occlude geometric features necessary for object and texture classification, while markerless skin preserves surface details but struggles in measuring tangential displacements effectively. Current practice to solve the above problem via UV lighting or virtual transfer using learning-based models introduces hardware complexity or computing burdens. This paper introduces MagicSkin, a novel tactile skin with translucent, tinted markers balancing the modes of marker and markerless for VBTS. It enables simultaneous tangential displacement tracking, force prediction, and surface detail preservation. This skin is easy to plug into GelSight-family sensors without requiring additional hardware or software tools. We comprehensively evaluate MagicSkin in downstream tasks. The translucent markers impressively enhance rather than degrade sensing performance compared with traditional markerless and inked marker design: it achieves best performance in object classification (99.17\%), texture classification (93.51\%), tangential displacement tracking (97\% point retention) and force prediction (66\% improvement in total force error). These experimental results demonstrate that translucent skin eliminates the traditional performance trade-off in marker or markerless modes, paving the way for multimodal tactile sensing essential in tactile robotics. See videos at this \href{https://zhuochenn.github.io/MagicSkin_project/}{link}.
>
---
#### [new 028] Task adaptation of Vision-Language-Action model: 1st Place Solution for the 2025 BEHAVIOR Challenge
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对50项长视野家庭任务的视觉-语言-动作模型适配问题，基于Pi0.5架构提出改进。引入相关噪声流匹配、混合注意力与系统2跟踪，提升训练效率与动作连贯性，在BEHAVIOR挑战赛中获第一，平均q-score达26%。**

- **链接: [https://arxiv.org/pdf/2512.06951v1](https://arxiv.org/pdf/2512.06951v1)**

> **作者:** Ilia Larchenko; Gleb Zarin; Akash Karnatak
>
> **备注:** 2025 NeurIPS Behavior Challenge 1st place solution
>
> **摘要:** We present a vision-action policy that won 1st place in the 2025 BEHAVIOR Challenge - a large-scale benchmark featuring 50 diverse long-horizon household tasks in photo-realistic simulation, requiring bimanual manipulation, navigation, and context-aware decision making. Building on the Pi0.5 architecture, we introduce several innovations. Our primary contribution is correlated noise for flow matching, which improves training efficiency and enables correlation-aware inpainting for smooth action sequences. We also apply learnable mixed-layer attention and System 2 stage tracking for ambiguity resolution. Training employs multi-sample flow matching to reduce variance, while inference uses action compression and challenge-specific correction rules. Our approach achieves 26% q-score across all 50 tasks on both public and private leaderboards.
>
---
#### [new 029] Interconnection and Damping Assignment Passivity-Based Control using Sparse Neural ODEs
- **分类: cs.RO**

- **简介: 该论文属控制理论任务，旨在解决IDA-PBC方法因需解复杂PDE而难应用的问题。提出基于稀疏神经ODE的学习框架，通过数据驱动方式近似满足匹配条件，实现复杂任务下端口哈密尔顿结构的控制器设计。**

- **链接: [https://arxiv.org/pdf/2512.06935v1](https://arxiv.org/pdf/2512.06935v1)**

> **作者:** Nicolò Botteghi; Owen Brook; Urban Fasel; Federico Califano
>
> **摘要:** Interconnection and Damping Assignment Passivity-Based Control (IDA-PBC) is a nonlinear control technique that assigns a port-Hamiltonian (pH) structure to a controlled system using a state-feedback law. While IDA-PBC has been extensively studied and applied to many systems, its practical implementation often remains confined to academic examples and, almost exclusively, to stabilization tasks. The main limitation of IDA-PBC stems from the complexity of analytically solving a set of partial differential equations (PDEs), referred to as the matching conditions, which enforce the pH structure of the closed-loop system. However, this is extremely challenging, especially for complex physical systems and tasks. In this work, we propose a novel numerical approach for designing IDA-PBC controllers without solving the matching PDEs exactly. We cast the IDA-PBC problem as the learning of a neural ordinary differential equation. In particular, we rely on sparse dictionary learning to parametrize the desired closed-loop system as a sparse linear combination of nonlinear state-dependent functions. Optimization of the controller parameters is achieved by solving a multi-objective optimization problem whose cost function is composed of a generic task-dependent cost and a matching condition-dependent cost. Our numerical results show that the proposed method enables (i) IDA-PBC to be applicable to complex tasks beyond stabilization, such as the discovery of periodic oscillatory behaviors, (ii) the derivation of closed-form expressions of the controlled system, including residual terms
>
---
#### [new 030] ESPADA: Execution Speedup via Semantics Aware Demonstration Data Downsampling for Imitation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人模仿学习任务，旨在解决行为克隆策略继承人类演示慢速问题。作者提出ESPADA框架，利用VLM-LLM提取任务语义与3D关系，识别关键阶段并选择性下采样非关键段，结合DTW传播标签，实现2倍加速且保持成功率。**

- **链接: [https://arxiv.org/pdf/2512.07371v1](https://arxiv.org/pdf/2512.07371v1)**

> **作者:** Byungju Kim; Jinu Pahk; Chungwoo Lee; Jaejoon Kim; Jangha Lee; Theo Taeyeong Kim; Kyuhwan Shim; Jun Ki Lee; Byoung-Tak Zhang
>
> **备注:** project page: https://project-espada.github.io/espada/
>
> **摘要:** Behavior-cloning based visuomotor policies enable precise manipulation but often inherit the slow, cautious tempo of human demonstrations, limiting practical deployment. However, prior studies on acceleration methods mainly rely on statistical or heuristic cues that ignore task semantics and can fail across diverse manipulation settings. We present ESPADA, a semantic and spatially aware framework that segments demonstrations using a VLM-LLM pipeline with 3D gripper-object relations, enabling aggressive downsampling only in non-critical segments while preserving precision-critical phases, without requiring extra data or architectural modifications, or any form of retraining. To scale from a single annotated episode to the full dataset, ESPADA propagates segment labels via Dynamic Time Warping (DTW) on dynamics-only features. Across both simulation and real-world experiments with ACT and DP baselines, ESPADA achieves approximately a 2x speed-up while maintaining success rates, narrowing the gap between human demonstrations and efficient robot control.
>
---
#### [new 031] POrTAL: Plan-Orchestrated Tree Assembly for Lookahead
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人在部分可观环境中基于不确定性的任务规划问题，提出轻量级算法POrTAL，融合FF-Replan与POMCP优势，提升规划效率与步数表现，并验证其在不同时间约束下的性能。**

- **链接: [https://arxiv.org/pdf/2512.06002v1](https://arxiv.org/pdf/2512.06002v1)**

> **作者:** Evan Conway; David Porfirio; David Chan; Mark Roberts; Laura M. Hiatt
>
> **备注:** Submitted to ICRA 26
>
> **摘要:** Assigning tasks to robots often involves supplying the robot with an overarching goal, such as through natural language, and then relying on the robot to uncover and execute a plan to achieve that goal. In many settings common to human-robot interaction, however, the world is only partially observable to the robot, requiring that it create plans under uncertainty. Although many probabilistic planning algorithms exist for this purpose, these algorithms can be inefficient if executed with the robot's limited computational resources, or may require more steps than expected to achieve the goal. We thereby created a new, lightweight, probabilistic planning algorithm, Plan-Orchestrated Tree Assembly for Lookahead (POrTAL), that combines the strengths of two baseline planning algorithms, FF-Replan and POMCP. In a series of case studies, we demonstrate POrTAL's ability to quickly arrive at solutions that outperform these baselines in terms of number of steps. We additionally demonstrate how POrTAL performs under varying temporal constraints.
>
---
#### [new 032] Delay-Aware Diffusion Policy: Bridging the Observation-Execution Gap in Dynamic Tasks
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人在动态任务中的感知-执行延迟问题，提出延迟感知扩散策略（DA-DP），将实际延迟显式引入训练与推理过程，通过轨迹校正和延迟条件增强策略鲁棒性，提升高延迟下的任务成功率，适用于多种任务、机器人与架构。**

- **链接: [https://arxiv.org/pdf/2512.07697v1](https://arxiv.org/pdf/2512.07697v1)**

> **作者:** Aileen Liao; Dong-Ki Kim; Max Olan Smith; Ali-akbar Agha-mohammadi; Shayegan Omidshafiei
>
> **摘要:** As a robot senses and selects actions, the world keeps changing. This inference delay creates a gap of tens to hundreds of milliseconds between the observed state and the state at execution. In this work, we take the natural generalization from zero delay to measured delay during training and inference. We introduce Delay-Aware Diffusion Policy (DA-DP), a framework for explicitly incorporating inference delays into policy learning. DA-DP corrects zero-delay trajectories to their delay-compensated counterparts, and augments the policy with delay conditioning. We empirically validate DA-DP on a variety of tasks, robots, and delays and find its success rate more robust to delay than delay-unaware methods. DA-DP is architecture agnostic and transfers beyond diffusion policies, offering a general pattern for delay-aware imitation learning. More broadly, DA-DP encourages evaluation protocols that report performance as a function of measured latency, not just task difficulty.
>
---
#### [new 033] Control of Powered Ankle-Foot Prostheses on Compliant Terrain: A Quantitative Approach to Stability Enhancement
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究如何提升假肢在松软地面上行走的稳定性。针对下肢截肢者在 compliant 地面易摔倒的问题，提出一种基于导纳控制的方法，通过调节假肢踝关节准刚度，实验证明其比传统控制器更稳定。**

- **链接: [https://arxiv.org/pdf/2512.06896v1](https://arxiv.org/pdf/2512.06896v1)**

> **作者:** Chrysostomos Karakasis; Camryn Scully; Robert Salati; Panagiotis Artemiadis
>
> **摘要:** Walking on compliant terrain presents a substantial challenge for individuals with lower-limb amputation, further elevating their already high risk of falling. While powered ankle-foot prostheses have demonstrated adaptability across speeds and rigid terrains, control strategies optimized for soft or compliant surfaces remain underexplored. This work experimentally validates an admittance-based control strategy that dynamically adjusts the quasi-stiffness of powered prostheses to enhance gait stability on compliant ground. Human subject experiments were conducted with three healthy individuals walking on two bilaterally compliant surfaces with ground stiffness values of 63 and 25 kN/m, representative of real-world soft environments. Controller performance was quantified using phase portraits and two walking stability metrics, offering a direct assessment of fall risk. Compared to a standard phase-variable controller developed for rigid terrain, the proposed admittance controller consistently improved gait stability across all compliant conditions. These results demonstrate the potential of adaptive, stability-aware prosthesis control to reduce fall risk in real-world environments and advance the robustness of human-prosthesis interaction in rehabilitation robotics.
>
---
#### [new 034] GuideNav: User-Informed Development of a Vision-Only Robotic Navigation Assistant For Blind Travelers
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文聚焦盲人导航辅助机器人，旨在解决现有系统缺乏用户参与设计的问题。通过用户研究获取需求，提出并实现了仅依赖视觉的“教学-复现”导航系统GuideNav，可在多变户外环境中稳定复现千米级路径，验证了其可行性和实用性。**

- **链接: [https://arxiv.org/pdf/2512.06147v1](https://arxiv.org/pdf/2512.06147v1)**

> **作者:** Hochul Hwang; Soowan Yang; Jahir Sadik Monon; Nicholas A Giudice; Sunghoon Ivan Lee; Joydeep Biswas; Donghyun Kim
>
> **摘要:** While commendable progress has been made in user-centric research on mobile assistive systems for blind and low-vision (BLV) individuals, references that directly inform robot navigation design remain rare. To bridge this gap, we conducted a comprehensive human study involving interviews with 26 guide dog handlers, four white cane users, nine guide dog trainers, and one O\&M trainer, along with 15+ hours of observing guide dog-assisted walking. After de-identification, we open-sourced the dataset to promote human-centered development and informed decision-making for assistive systems for BLV people. Building on insights from this formative study, we developed GuideNav, a vision-only, teach-and-repeat navigation system. Inspired by how guide dogs are trained and assist their handlers, GuideNav autonomously repeats a path demonstrated by a sighted person using a robot. Specifically, the system constructs a topological representation of the taught route, integrates visual place recognition with temporal filtering, and employs a relative pose estimator to compute navigation actions - all without relying on costly, heavy, power-hungry sensors such as LiDAR. In field tests, GuideNav consistently achieved kilometer-scale route following across five outdoor environments, maintaining reliability despite noticeable scene variations between teach and repeat runs. A user study with 3 guide dog handlers and 1 guide dog trainer further confirmed the system's feasibility, marking (to our knowledge) the first demonstration of a quadruped mobile system retrieving a path in a manner comparable to guide dogs.
>
---
#### [new 035] Situation-Aware Interactive MPC Switching for Autonomous Driving
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究自动驾驶中交互场景的控制问题，旨在平衡控制性能与计算开销。提出基于情境感知的MPC切换策略，通过神经网络分类器在不同交互能力的MPC间切换，提升关键场景性能并降低整体计算负担。**

- **链接: [https://arxiv.org/pdf/2512.06182v1](https://arxiv.org/pdf/2512.06182v1)**

> **作者:** Shuhao Qi; Qiling Aori; Luyao Zhang; Mircea Lazar; Sofie Haesaert
>
> **摘要:** To enable autonomous driving in interactive traffic scenarios, various model predictive control (MPC) formulations have been proposed, each employing different interaction models. While higher-fidelity models enable more intelligent behavior, they incur increased computational cost. Since strong interactions are relatively infrequent in traffic, a practical strategy for balancing performance and computational overhead is to invoke an appropriate controller based on situational demands. To achieve this approach, we first conduct a comparative study to assess and hierarchize the interactive capabilities of different MPC formulations. Furthermore, we develop a neural network-based classifier to enable situation-aware switching among controllers with different levels of interactive capability. We demonstrate that this situation-aware switching can both substantially improve overall performance by activating the most advanced interactive MPC in rare but critical situations, and significantly reduce computational load by using a basic MPC in the majority of scenarios.
>
---
#### [new 036] From Real-World Traffic Data to Relevant Critical Scenarios
- **分类: cs.RO; cs.AI**

- **简介: 该论文旨在识别高速公路上与自动驾驶相关的关键驾驶场景。针对“未知不安全”场景难以穷举的问题，提出基于真实交通数据提取关键场景，并通过合成方法生成新场景，提升验证效率。**

- **链接: [https://arxiv.org/pdf/2512.07482v1](https://arxiv.org/pdf/2512.07482v1)**

> **作者:** Florian Lüttner; Nicole Neis; Daniel Stadler; Robin Moss; Mirjam Fehling-Kaschek; Matthias Pfriem; Alexander Stolz; Jens Ziehn
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** The reliable operation of autonomous vehicles, automated driving functions, and advanced driver assistance systems across a wide range of relevant scenarios is critical for their development and deployment. Identifying a near-complete set of relevant driving scenarios for such functionalities is challenging due to numerous degrees of freedom involved, each affecting the outcomes of the driving scenario differently. Moreover, with increasing technical complexity of new functionalities, the number of potentially relevant, particularly "unknown unsafe" scenarios is increasing. To enhance validation efficiency, it is essential to identify relevant scenarios in advance, starting with simpler domains like highways before moving to more complex environments such as urban traffic. To address this, this paper focuses on analyzing lane change scenarios in highway traffic, which involve multiple degrees of freedom and present numerous safetyrelevant scenarios. We describe the process of data acquisition and processing of real-world data from public highway traffic, followed by the application of criticality measures on trajectory data to evaluate scenarios, as conducted within the AVEAS project (www.aveas.org). By linking the calculated measures to specific lane change driving scenarios and the conditions under which the data was collected, we facilitate the identification of safetyrelevant driving scenarios for various applications. Further, to tackle the extensive range of "unknown unsafe" scenarios, we propose a way to generate relevant scenarios by creating synthetic scenarios based on recorded ones. Consequently, we demonstrate and evaluate a processing chain that enables the identification of safety-relevant scenarios, the development of data-driven methods for extracting these scenarios, and the generation of synthetic critical scenarios via sampling on highways.
>
---
#### [new 037] Dynamic Visual SLAM using a General 3D Prior
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决动态环境中相机位姿估计易受运动物体干扰的问题。作者提出一种融合几何优化与前馈重建模型的方法，利用3D先验过滤动态区域并辅助深度估计，提升动态场景下SLAM的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.06868v1](https://arxiv.org/pdf/2512.06868v1)**

> **作者:** Xingguang Zhong; Liren Jin; Marija Popović; Jens Behley; Cyrill Stachniss
>
> **备注:** 8 pages
>
> **摘要:** Reliable incremental estimation of camera poses and 3D reconstruction is key to enable various applications including robotics, interactive visualization, and augmented reality. However, this task is particularly challenging in dynamic natural environments, where scene dynamics can severely deteriorate camera pose estimation accuracy. In this work, we propose a novel monocular visual SLAM system that can robustly estimate camera poses in dynamic scenes. To this end, we leverage the complementary strengths of geometric patch-based online bundle adjustment and recent feed-forward reconstruction models. Specifically, we propose a feed-forward reconstruction model to precisely filter out dynamic regions, while also utilizing its depth prediction to enhance the robustness of the patch-based visual SLAM. By aligning depth prediction with estimated patches from bundle adjustment, we robustly handle the inherent scale ambiguities of the batch-wise application of the feed-forward reconstruction model.
>
---
#### [new 038] Mimir: Hierarchical Goal-Driven Diffusion with Uncertainty Propagation for End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文面向端到端自动驾驶任务，旨在解决高阶导航信号不准与计算开销大的问题。提出Mimir框架，通过不确定性估计和多速率引导机制，提升轨迹生成鲁棒性与推理速度，在导航基准上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.07130v1](https://arxiv.org/pdf/2512.07130v1)**

> **作者:** Zebin Xing; Yupeng Zheng; Qichao Zhang; Zhixing Ding; Pengxuan Yang; Songen Gu; Zhongpu Xia; Dongbin Zhao
>
> **摘要:** End-to-end autonomous driving has emerged as a pivotal direction in the field of autonomous systems. Recent works have demonstrated impressive performance by incorporating high-level guidance signals to steer low-level trajectory planners. However, their potential is often constrained by inaccurate high-level guidance and the computational overhead of complex guidance modules. To address these limitations, we propose Mimir, a novel hierarchical dual-system framework capable of generating robust trajectories relying on goal points with uncertainty estimation: (1) Unlike previous approaches that deterministically model, we estimate goal point uncertainty with a Laplace distribution to enhance robustness; (2) To overcome the slow inference speed of the guidance system, we introduce a multi-rate guidance mechanism that predicts extended goal points in advance. Validated on challenging Navhard and Navtest benchmarks, Mimir surpasses previous state-of-the-art methods with a 20% improvement in the driving score EPDMS, while achieving 1.6 times improvement in high-level module inference speed without compromising accuracy. The code and models will be released soon to promote reproducibility and further development. The code is available at https://github.com/ZebinX/Mimir-Uncertainty-Driving
>
---
#### [new 039] VideoVLA: Video Generators Can Be Generalizable Robot Manipulators
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出VideoVLA，将视频生成模型转化为机器人视觉-语言-动作控制器。通过联合预测动作与未来视觉结果，实现跨任务、对象和场景的泛化操作，提升机器人在开放环境中的适应能力。**

- **链接: [https://arxiv.org/pdf/2512.06963v1](https://arxiv.org/pdf/2512.06963v1)**

> **作者:** Yichao Shen; Fangyun Wei; Zhiying Du; Yaobo Liang; Yan Lu; Jiaolong Yang; Nanning Zheng; Baining Guo
>
> **备注:** Project page: https://videovla-nips2025.github.io
>
> **摘要:** Generalization in robot manipulation is essential for deploying robots in open-world environments and advancing toward artificial general intelligence. While recent Vision-Language-Action (VLA) models leverage large pre-trained understanding models for perception and instruction following, their ability to generalize to novel tasks, objects, and settings remains limited. In this work, we present VideoVLA, a simple approach that explores the potential of transforming large video generation models into robotic VLA manipulators. Given a language instruction and an image, VideoVLA predicts an action sequence as well as the future visual outcomes. Built on a multi-modal Diffusion Transformer, VideoVLA jointly models video, language, and action modalities, using pre-trained video generative models for joint visual and action forecasting. Our experiments show that high-quality imagined futures correlate with reliable action predictions and task success, highlighting the importance of visual imagination in manipulation. VideoVLA demonstrates strong generalization, including imitating other embodiments' skills and handling novel objects. This dual-prediction strategy - forecasting both actions and their visual consequences - explores a paradigm shift in robot learning and unlocks generalization capabilities in manipulation systems.
>
---
#### [new 040] Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input
- **分类: cs.RO**

- **简介: 该论文研究 humanoid 足球机器人在感知噪声下的敏捷射门技能学习。针对感知不准确和环境干扰问题，提出四阶段强化学习框架，结合噪声建模与在线约束强化学习，实现从仿真到现实的稳健连续射门控制。**

- **链接: [https://arxiv.org/pdf/2512.06571v1](https://arxiv.org/pdf/2512.06571v1)**

> **作者:** Zifan Xu; Myoungkyu Seo; Dongmyeong Lee; Hao Fu; Jiaheng Hu; Jiaxun Cui; Yuqian Jiang; Zhihan Wang; Anastasiia Brund; Joydeep Biswas; Peter Stone
>
> **摘要:** Learning fast and robust ball-kicking skills is a critical capability for humanoid soccer robots, yet it remains a challenging problem due to the need for rapid leg swings, postural stability on a single support foot, and robustness under noisy sensory input and external perturbations (e.g., opponents). This paper presents a reinforcement learning (RL)-based system that enables humanoid robots to execute robust continual ball-kicking with adaptability to different ball-goal configurations. The system extends a typical teacher-student training framework -- in which a "teacher" policy is trained with ground truth state information and the "student" learns to mimic it with noisy, imperfect sensing -- by including four training stages: (1) long-distance ball chasing (teacher); (2) directional kicking (teacher); (3) teacher policy distillation (student); and (4) student adaptation and refinement (student). Key design elements -- including tailored reward functions, realistic noise modeling, and online constrained RL for adaptation and refinement -- are critical for closing the sim-to-real gap and sustaining performance under perceptual uncertainty. Extensive evaluations in both simulation and on a real robot demonstrate strong kicking accuracy and goal-scoring success across diverse ball-goal configurations. Ablation studies further highlight the necessity of the constrained RL, noise modeling, and the adaptation stage. This work presents a system for learning robust continual humanoid ball-kicking under imperfect perception, establishing a benchmark task for visuomotor skill learning in humanoid whole-body control.
>
---
#### [new 041] Probabilistic Weapon Engagement Zones for a Turn Constrained Pursuer
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究带转向约束的追击者在参数不确定下的威胁区域建模。提出曲线-直线概率交战区（CSPEZ），通过四种不确定性传播方法生成低风险逃逸轨迹，并集成到路径优化中以规避捕获风险。**

- **链接: [https://arxiv.org/pdf/2512.06130v1](https://arxiv.org/pdf/2512.06130v1)**

> **作者:** Grant Stagg; Isaac E. Weintraub; Cameron K. Peterson
>
> **备注:** Accepted for presentation at AIAA SciTech 2026. 17 pages, 7 figures
>
> **摘要:** Curve-straight probabilistic engagement zones (CSPEZ) quantify the spatial regions an evader should avoid to reduce capture risk from a turn-rate-limited pursuer following a curve-straight path with uncertain parameters including position, heading, velocity, range, and maximum turn rate. This paper presents methods for generating evader trajectories that minimize capture risk under such uncertainty. We first derive an analytic solution for the deterministic curve-straight basic engagement zone (CSBEZ), then extend this formulation to a probabilistic framework using four uncertainty-propagation approaches: Monte Carlo sampling, linearization, quadratic approximation, and neural-network regression. We evaluate the accuracy and computational cost of each approximation method and demonstrate how CSPEZ constraints can be integrated into a trajectory-optimization algorithm to produce safe paths that explicitly account for pursuer uncertainty.
>
---
#### [new 042] TacFinRay: Soft Tactile Fin-Ray Finger with Indirect Tactile Sensing for Robust Grasping
- **分类: cs.RO**

- **简介: 该论文提出TacFinRay软体触觉手指，通过间接感知实现接触位置与压入深度检测。旨在解决软体机器人中触觉传感难以集成的问题，采用仿生结构与视觉触觉传感器结合，优化设计实现高精度感知，并成功应用于不确定抓取的放置任务。**

- **链接: [https://arxiv.org/pdf/2512.06524v1](https://arxiv.org/pdf/2512.06524v1)**

> **作者:** Saekwang Nam; Bowen Deng; Loong Yi Lee; Jonathan M. Rossiter; Nathan F. Lepora
>
> **备注:** Accepted in IEEE Robotics Automation Letters. S. Nam, B. Deng co-first authors
>
> **摘要:** We present a tactile-sensorized Fin-Ray finger that enables simultaneous detection of contact location and indentation depth through an indirect sensing approach. A hinge mechanism is integrated between the soft Fin-Ray structure and a rigid sensing module, allowing deformation and translation information to be transferred to a bottom crossbeam upon which are an array of marker-tipped pins based on the biomimetic structure of the TacTip vision-based tactile sensor. Deformation patterns captured by an internal camera are processed using a convolutional neural network to infer contact conditions without directly sensing the finger surface. The finger design was optimized by varying pin configurations and hinge orientations, achieving 0.1\,mm depth and 2mm location-sensing accuracies. The perception demonstrated robust generalization to various indenter shapes and sizes, which was applied to a pick-and-place task under uncertain picking positions, where the tactile feedback significantly improved placement accuracy. Overall, this work provides a lightweight, flexible, and scalable tactile sensing solution suitable for soft robotic structures where the sensing needs situating away from the contact interface.
>
---
#### [new 043] Surrogate compliance modeling enables reinforcement learned locomotion gaits for soft robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究软体机器人运动控制问题，提出代理顺应性建模方法，在刚体仿真中引入间接变量表征软体变形，结合强化学习实现高效策略训练，并成功迁移至实体机器人，实现在多种复杂地形下的高效、稳定运动。**

- **链接: [https://arxiv.org/pdf/2512.07114v1](https://arxiv.org/pdf/2512.07114v1)**

> **作者:** Jue Wang; Mingsong Jiang; Luis A. Ramirez; Bilige Yang; Mujun Zhang; Esteban Figueroa; Wenzhong Yan; Rebecca Kramer-Bottiglio
>
> **摘要:** Adaptive morphogenetic robots adapt their morphology and control policies to meet changing tasks and environmental conditions. Many such systems leverage soft components, which enable shape morphing but also introduce simulation and control challenges. Soft-body simulators remain limited in accuracy and computational tractability, while rigid-body simulators cannot capture soft-material dynamics. Here, we present a surrogate compliance modeling approach: rather than explicitly modeling soft-body physics, we introduce indirect variables representing soft-material deformation within a rigid-body simulator. We validate this approach using our amphibious robotic turtle, a quadruped with soft morphing limbs designed for multi-environment locomotion. By capturing deformation effects as changes in effective limb length and limb center of mass, and by applying reinforcement learning with extensive randomization of these indirect variables, we achieve reliable policy learning entirely in a rigid-body simulation. The resulting gaits transfer directly to hardware, demonstrating high-fidelity sim-to-real performance on hard, flat substrates and robust, though lower-fidelity, transfer on rheologically complex terrains. The learned closed-loop gaits exhibit unprecedented terrestrial maneuverability and achieve an order-of-magnitude reduction in cost of transport compared to open-loop baselines. Field experiments with the robot further demonstrate stable, multi-gait locomotion across diverse natural terrains, including gravel, grass, and mud.
>
---
#### [new 044] REWW-ARM -- Remote Wire-Driven Mobile Robot: Design, Control, and Experimental Validation
- **分类: cs.RO**

- **简介: 该论文提出“远程绳驱”系统REWW-ARM，旨在解决电子设备限制机器人环境适应性的问题。通过将电机与控制单元远离主体，实现无电子器件的远端机器人移动、姿态控制和操作，验证了其在陆地和水下运行的能力。**

- **链接: [https://arxiv.org/pdf/2512.06192v1](https://arxiv.org/pdf/2512.06192v1)**

> **作者:** Takahiro Hattori; Kento Kawaharazuka; Temma Suzuki; Keita Yoneda; Kei Okada
>
> **备注:** Accepted on Advanced Intelligent Systems
>
> **摘要:** Electronic devices are essential for robots but limit their usable environments. To overcome this, methods excluding electronics from the operating environment while retaining advanced electronic control and actuation have been explored. These include the remote hydraulic drive of electronics-free mobile robots, which offer high reachability, and long wire-driven robot arms with motors consolidated at the base, which offer high environmental resistance. To combine the advantages of both, this study proposes a new system, "Remote Wire Drive." As a proof-of-concept, we designed and developed the Remote Wire-Driven robot "REWW-ARM", which consists of the following components: 1) a novel power transmission mechanism, the "Remote Wire Transmission Mechanism" (RWTM), the key technology of the Remote Wire Drive; 2) an electronics-free distal mobile robot driven by it; and 3) a motor-unit that generates power and provides electronic closed-loop control based on state estimation via the RWTM. In this study, we evaluated the mechanical and control performance of REWW-ARM through several experiments, demonstrating its capability for locomotion, posture control, and object manipulation both on land and underwater. This suggests the potential for applying the Remote Wire-Driven system to various types of robots, thereby expanding their operational range.
>
---
#### [new 045] Using Vision-Language Models as Proxies for Social Intelligence in Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文研究人机交互中机器人如何识别人类非语言信号以决定互动时机。通过真实场景部署分析，提出两阶段管道，用轻量感知触发视觉-语言模型查询，实现对社会线索的响应，使机器人能自然、恰当地参与交互。**

- **链接: [https://arxiv.org/pdf/2512.07177v1](https://arxiv.org/pdf/2512.07177v1)**

> **作者:** Fanjun Bu; Melina Tsai; Audrey Tjokro; Tapomayukh Bhattacharjee; Jorge Ortiz; Wendy Ju
>
> **摘要:** Robots operating in everyday environments must often decide when and whether to engage with people, yet such decisions often hinge on subtle nonverbal cues that unfold over time and are difficult to model explicitly. Drawing on a five-day Wizard-of-Oz deployment of a mobile service robot in a university cafe, we analyze how people signal interaction readiness through nonverbal behaviors and how expert wizards use these cues to guide engagement. Motivated by these observations, we propose a two-stage pipeline in which lightweight perceptual detectors (gaze shifts and proxemics) are used to selectively trigger heavier video-based vision-language model (VLM) queries at socially meaningful moments. We evaluate this pipeline on replayed field interactions and compare two prompting strategies. Our findings suggest that selectively using VLMs as proxies for social reasoning enables socially responsive robot behavior, allowing robots to act appropriately by attending to the cues people naturally provide in real-world interactions.
>
---
#### [new 046] Spatiotemporal Calibration and Ground Truth Estimation for High-Precision SLAM Benchmarking in Extended Reality
- **分类: cs.RO**

- **简介: 该论文属于SLAM基准评测任务，旨在解决MoCap系统在时空校准和测量抖动方面的局限性。作者提出一种融合IMU数据的连续时间最大似然估计方法，实现高精度时空标定与真值估计，提升XR中SLAM算法的评测准确性。**

- **链接: [https://arxiv.org/pdf/2512.07221v1](https://arxiv.org/pdf/2512.07221v1)**

> **作者:** Zichao Shu; Shitao Bei; Lijun Li; Zetao Chen
>
> **摘要:** Simultaneous localization and mapping (SLAM) plays a fundamental role in extended reality (XR) applications. As the standards for immersion in XR continue to increase, the demands for SLAM benchmarking have become more stringent. Trajectory accuracy is the key metric, and marker-based optical motion capture (MoCap) systems are widely used to generate ground truth (GT) because of their drift-free and relatively accurate measurements. However, the precision of MoCap-based GT is limited by two factors: the spatiotemporal calibration with the device under test (DUT) and the inherent jitter in the MoCap measurements. These limitations hinder accurate SLAM benchmarking, particularly for key metrics like rotation error and inter-frame jitter, which are critical for immersive XR experiences. This paper presents a novel continuous-time maximum likelihood estimator to address these challenges. The proposed method integrates auxiliary inertial measurement unit (IMU) data to compensate for MoCap jitter. Additionally, a variable time synchronization method and a pose residual based on screw congruence constraints are proposed, enabling precise spatiotemporal calibration across multiple sensors and the DUT. Experimental results demonstrate that our approach outperforms existing methods, achieving the precision necessary for comprehensive benchmarking of state-of-the-art SLAM algorithms in XR applications. Furthermore, we thoroughly validate the practicality of our method by benchmarking several leading XR devices and open-source SLAM algorithms. The code is publicly available at https://github.com/ylab-xrpg/xr-hpgt.
>
---
#### [new 047] Energy-Efficient Navigation for Surface Vehicles in Vortical Flow Fields
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究自主水面航行器在涡旋流场中的节能导航问题。针对传统方法因部分可观测性失效的挑战，提出基于软演员评论家算法的强化学习框架，仅利用局部流速信息实现高效路径规划，显著降低能耗30%-50%，提升长时海洋作业能力。**

- **链接: [https://arxiv.org/pdf/2512.06912v1](https://arxiv.org/pdf/2512.06912v1)**

> **作者:** Rushiraj Gadhvi; Sandeep Manjanna
>
> **备注:** Under Review for International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** For centuries, khalasi have skillfully harnessed ocean currents to navigate vast waters with minimal effort. Emulating this intuition in autonomous systems remains a significant challenge, particularly for Autonomous Surface Vehicles tasked with long duration missions under strict energy budgets. In this work, we present a learning-based approach for energy-efficient surface vehicle navigation in vortical flow fields, where partial observability often undermines traditional path-planning methods. We present an end to end reinforcement learning framework based on Soft Actor Critic that learns flow-aware navigation policies using only local velocity measurements. Through extensive evaluation across diverse and dynamically rich scenarios, our method demonstrates substantial energy savings and robust generalization to previously unseen flow conditions, offering a promising path toward long term autonomy in ocean environments. The navigation paths generated by our proposed approach show an improvement in energy conservation 30 to 50 percent compared to the existing state of the art techniques.
>
---
#### [new 048] Entropy-Controlled Intrinsic Motivation Reinforcement Learning for Quadruped Robot Locomotion in Complex Terrains
- **分类: cs.RO**

- **简介: 该论文研究四足机器人在复杂地形中的运动控制任务，旨在解决强化学习中策略早敛导致性能下降的问题。提出熵控内在动机（ECIM）算法，结合熵控制与内在动机，提升探索能力，在多种地形下显著提高稳定性、降低能耗。**

- **链接: [https://arxiv.org/pdf/2512.06486v1](https://arxiv.org/pdf/2512.06486v1)**

> **作者:** Wanru Gong; Xinyi Zheng; Xiaopeng Yang; Xiaoqing Zhu
>
> **摘要:** Learning is the basis of both biological and artificial systems when it comes to mimicking intelligent behaviors. From the classical PPO (Proximal Policy Optimization), there is a series of deep reinforcement learning algorithms which are widely used in training locomotion policies for quadrupedal robots because of their stability and sample efficiency. However, among all these variants, experiments and simulations often converge prematurely, leading to suboptimal locomotion and reduced task performance. Therefore, in this paper, we introduce Entropy-Controlled Intrinsic Motivation (ECIM), an entropy-based reinforcement learning algorithm in contrast with the PPO series, that can reduce premature convergence by combining intrinsic motivation with adaptive exploration. For experiments, in order to parallel with other baselines, we chose to apply it in Isaac Gym across six terrain categories: upward slopes, downward slopes, uneven rough terrain, ascending stairs, descending stairs, and flat ground as widely used. For comparison, our experiments consistently achieve better performance: task rewards increase by 4--12%, peak body pitch oscillation is reduced by 23--29%, joint acceleration decreases by 20--32%, and joint torque consumption declines by 11--20%. Overall, our model ECIM, by combining entropy control and intrinsic motivation control, achieves better results in stability across different terrains for quadrupedal locomotion, and at the same time reduces energetic cost and makes it a practical choice for complex robotic control tasks.
>
---
#### [new 049] Training-Free Robot Pose Estimation using Off-the-Shelf Foundational Models
- **分类: cs.RO; eess.IV**

- **简介: 该论文研究无需训练的机器人姿态估计任务，利用现成的视觉-语言模型从单张图像中估计机械臂关节角度，解决可靠姿态估计问题。作者在合成与真实数据上评估前沿模型性能，并发现测试时扩展或参数扩展难以提升预测精度。**

- **链接: [https://arxiv.org/pdf/2512.06017v1](https://arxiv.org/pdf/2512.06017v1)**

> **作者:** Laurence Liang
>
> **备注:** Accepted at CVIS 2025
>
> **摘要:** Pose estimation of a robot arm from visual inputs is a challenging task. However, with the increasing adoption of robot arms for both industrial and residential use cases, reliable joint angle estimation can offer improved safety and performance guarantees, and also be used as a verifier to further train robot policies. This paper introduces using frontier vision-language models (VLMs) as an ``off-the-shelf" tool to estimate a robot arm's joint angles from a single target image. By evaluating frontier VLMs on both synthetic and real-world image-data pairs, this paper establishes a performance baseline attained by current FLMs. In addition, this paper presents empirical results suggesting that test time scaling or parameter scaling alone does not lead to improved joint angle predictions.
>
---
#### [new 050] Efficient and Compliant Control Framework for Versatile Human-Humanoid Collaborative Transportation
- **分类: cs.RO**

- **简介: 该论文研究人形机器人与人类协同搬运任务，解决协作中平移与旋转运动的控制问题。提出包含规划、控制与刚度调节的框架，基于I-LIP与MPC生成步态，QP控制器执行，调节交互刚度以提升协作效率，并在Digit机器人上验证。**

- **链接: [https://arxiv.org/pdf/2512.07819v1](https://arxiv.org/pdf/2512.07819v1)**

> **作者:** Shubham S. Kumbhar; Abhijeet M. Kulkarni; Panagiotis Artemiadis
>
> **摘要:** We present a control framework that enables humanoid robots to perform collaborative transportation tasks with a human partner. The framework supports both translational and rotational motions, which are fundamental to co-transport scenarios. It comprises three components: a high-level planner, a low-level controller, and a stiffness modulation mechanism. At the planning level, we introduce the Interaction Linear Inverted Pendulum (I-LIP), which, combined with an admittance model and an MPC formulation, generates dynamically feasible footstep plans. These are executed by a QP-based whole-body controller that accounts for the coupled humanoid-object dynamics. Stiffness modulation regulates robot-object interaction, ensuring convergence to the desired relative configuration defined by the distance between the object and the robot's center of mass. We validate the effectiveness of the framework through real-world experiments conducted on the Digit humanoid platform. To quantify collaboration quality, we propose an efficiency metric that captures both task performance and inter-agent coordination. We show that this metric highlights the role of compliance in collaborative tasks and offers insights into desirable trajectory characteristics across both high- and low-level control layers. Finally, we showcase experimental results on collaborative behaviors, including translation, turning, and combined motions such as semi circular trajectories, representative of naturally occurring co-transportation tasks.
>
---
#### [new 051] Error-Centric PID Untrained Neural-Net (EC-PIDUNN) For Nonlinear Robotics Control
- **分类: cs.RO**

- **简介: 该论文提出一种新型EC-PIDUNN架构，属非线性机器人控制任务。针对传统PID在复杂非线性系统中控制性能下降的问题，结合未训练神经网络与改进PID，利用误差输入和动态参数调整，提升控制收敛性与稳定性，无需系统动力学先验知识。**

- **链接: [https://arxiv.org/pdf/2512.06578v1](https://arxiv.org/pdf/2512.06578v1)**

> **作者:** Waleed Razzaq
>
> **备注:** Under review at SoftComputing
>
> **摘要:** Classical Proportional-Integral-Derivative (PID) control has been widely successful across various industrial systems such as chemical processes, robotics, and power systems. However, as these systems evolved, the increase in the nonlinear dynamics and the complexity of interconnected variables have posed challenges that classical PID cannot effectively handle, often leading to instability, overshooting, or prolonged settling times. Researchers have proposed PIDNN models that combine the function approximation capabilities of neural networks with PID control to tackle these nonlinear challenges. However, these models require extensive, highly refined training data and have significant computational costs, making them less favorable for real-world applications. In this paper, We propose a novel EC-PIDUNN architecture, which integrates an untrained neural network with an improved PID controller, incorporating a stabilizing factor (\(τ\)) to generate the control signal. Like classical PID, our architecture uses the steady-state error \(e_t\) as input bypassing the need for explicit knowledge of the systems dynamics. By forming an input vector from \(e_t\) within the neural network, we increase the dimensionality of input allowing for richer data representation. Additionally, we introduce a vector of parameters \( ρ_t \) to shape the output trajectory and a \textit{dynamic compute} function to adjust the PID coefficients from predefined values. We validate the effectiveness of EC-PIDUNN on multiple nonlinear robotics applications: (1) nonlinear unmanned ground vehicle systems that represent the Ackermann steering mechanism and kinematics control, (2) Pan-Tilt movement system. In both tests, it outperforms classical PID in convergence and stability achieving a nearly critically damped response.
>
---
#### [new 052] Statistic-Augmented, Decoupled MoE Routing and Aggregating in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属自动驾驶语义分割任务，针对MoE模型中专家路由不准与聚合低效问题，提出统计增强的解耦MoE路由与聚合机制（MoE-RAM），通过统计特征匹配优化专家选择与加权融合，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2512.06664v1](https://arxiv.org/pdf/2512.06664v1)**

> **作者:** Wei-Bin Kou; Guangxu Zhu; Jingreng Lei; Chen Zhang; Yik-Chung Wu; Jianping Wang
>
> **备注:** 9 pages
>
> **摘要:** Autonomous driving (AD) scenarios are inherently complex and diverse, posing significant challenges for a single deep learning model to effectively cover all possible conditions, such as varying weather, traffic densities, and road types. Large Model (LM)-Driven Mixture of Experts (MoE) paradigm offers a promising solution, where LM serves as the backbone to extract latent features while MoE serves as the downstream head to dynamically select and aggregate specialized experts to adapt to different scenarios. However, routing and aggregating in MoE face intrinsic challenges, including imprecise expert selection due to flawed routing strategy and inefficient expert aggregation leading to suboptimal prediction. To address these issues, we propose a statistic-augmented, decoupled MoE }outing and Aggregating Mechanism (MoE-RAM) driven by LM. Specifically, on the one hand, MoE-RAM enhances expert routing by incorporating statistical retrieval mechanism to match LM-extracted latent features with cached prototypical features of the most relevant experts; on the other hand, MoE-RAM adaptively reweights experts' outputs in fusion by measuring statistical distances of experts' instant features against LM-extracted latent features. Benefiting from the synergy of the statistic-augmented MoE's routing and aggregating, MoE-RAM ultimately improves the prediction performance. We take the AD semantic segmentation task as an example to assess the proposed MoE-RAM. Extensive experiments on AD datasets demonstrate the superiority of MoE-RAM compared to other MoE baselines and conventional single-model approaches.
>
---
#### [new 053] VP-AutoTest: A Virtual-Physical Fusion Autonomous Driving Testing Platform
- **分类: cs.RO; cs.SE**

- **简介: 该论文提出VP-AutoTest平台，属自动驾驶测试任务，旨在解决传统测试方法真实性低、成本高及虚拟-物理融合测试元素少、评估单一等问题。工作包括构建多元素虚实融合系统、支持协同测试与AI评估，并实现可信度自评价。**

- **链接: [https://arxiv.org/pdf/2512.07507v1](https://arxiv.org/pdf/2512.07507v1)**

> **作者:** Yiming Cui; Shiyu Fang; Jiarui Zhang; Yan Huang; Chengkai Xu; Bing Zhu; Hao Zhang; Peng Hang; Jian Sun
>
> **摘要:** The rapid development of autonomous vehicles has led to a surge in testing demand. Traditional testing methods, such as virtual simulation, closed-course, and public road testing, face several challenges, including unrealistic vehicle states, limited testing capabilities, and high costs. These issues have prompted increasing interest in virtual-physical fusion testing. However, despite its potential, virtual-physical fusion testing still faces challenges, such as limited element types, narrow testing scope, and fixed evaluation metrics. To address these challenges, we propose the Virtual-Physical Testing Platform for Autonomous Vehicles (VP-AutoTest), which integrates over ten types of virtual and physical elements, including vehicles, pedestrians, and roadside infrastructure, to replicate the diversity of real-world traffic participants. The platform also supports both single-vehicle interaction and multi-vehicle cooperation testing, employing adversarial testing and parallel deduction to accelerate fault detection and explore algorithmic limits, while OBU and Redis communication enable seamless vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I) cooperation across all levels of cooperative automation. Furthermore, VP-AutoTest incorporates a multidimensional evaluation framework and AI-driven expert systems to conduct comprehensive performance assessment and defect diagnosis. Finally, by comparing virtual-physical fusion test results with real-world experiments, the platform performs credibility self-evaluation to ensure both the fidelity and efficiency of autonomous driving testing. Please refer to the website for the full testing functionalities on the autonomous driving public service platform OnSite:https://www.onsite.com.cn.
>
---
#### [new 054] Robust Optimization-based Autonomous Dynamic Soaring with a Fixed-Wing UAV
- **分类: cs.RO**

- **简介: 该论文研究固定翼无人机自主动态滑翔任务，旨在利用风剪获取能量实现持久飞行。提出一种鲁棒优化框架，通过显式风场建模与鲁棒路径跟踪控制，提升对风场误差和扰动的适应性，并经仿真与实飞验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.06610v1](https://arxiv.org/pdf/2512.06610v1)**

> **作者:** Marvin Harms; Jaeyoung Lim; David Rohr; Friedrich Rockenbauer; Nicholas Lawrance; Roland Siegwart
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Dynamic soaring is a flying technique to exploit the energy available in wind shear layers, enabling potentially unlimited flight without the need for internal energy sources. We propose a framework for autonomous dynamic soaring with a fixed-wing unmanned aerial vehicle (UAV). The framework makes use of an explicit representation of the wind field and a classical approach for guidance and control of the UAV. Robustness to wind field estimation error is achieved by constructing point-wise robust reference paths for dynamic soaring and the development of a robust path following controller for the fixed-wing UAV. The framework is evaluated in dynamic soaring scenarios in simulation and real flight tests. In simulation, we demonstrate robust dynamic soaring flight subject to varied wind conditions, estimation errors and disturbances. Critical components of the framework, including energy predictions and path-following robustness, are further validated in real flights to assure small sim-to-real gap. Together, our results strongly indicate the ability of the proposed framework to achieve autonomous dynamic soaring flight in wind shear.
>
---
#### [new 055] Vision-Guided Grasp Planning for Prosthetic Hands in Unstructured Environments
- **分类: cs.RO**

- **简介: 该论文研究假肢手在非结构化环境中的视觉引导抓取任务，旨在提升其自主抓取能力。提出一种融合视觉感知、抓取规划与控制的模块化方法，通过BVH分割物体、RRT*生成轨迹、DLS求解逆运动学，实现自适应实时抓取，并在仿真和Linker Hand O7平台验证。**

- **链接: [https://arxiv.org/pdf/2512.06517v1](https://arxiv.org/pdf/2512.06517v1)**

> **作者:** Shifa Sulaiman; Akash Bachhar; Ming Shen; Simon Bøgh
>
> **摘要:** Recent advancements in prosthetic technology have increasingly focused on enhancing dexterity and autonomy through intelligent control systems. Vision-based approaches offer promising results for enabling prosthetic hands to interact more naturally with diverse objects in dynamic environments. Building on this foundation, the paper presents a vision-guided grasping algorithm for a prosthetic hand, integrating perception, planning, and control for dexterous manipulation. A camera mounted on the set up captures the scene, and a Bounding Volume Hierarchy (BVH)-based vision algorithm is employed to segment an object for grasping and define its bounding box. Grasp contact points are then computed by generating candidate trajectories using Rapidly-exploring Random Tree Star algorithm, and selecting fingertip end poses based on the minimum Euclidean distance between these trajectories and the objects point cloud. Each finger grasp pose is determined independently, enabling adaptive, object-specific configurations. Damped Least Square (DLS) based Inverse kinematics solver is used to compute the corresponding joint angles, which are subsequently transmitted to the finger actuators for execution. This modular pipeline enables per-finger grasp planning and supports real-time adaptability in unstructured environments. The proposed method is validated in simulation, and experimental integration on a Linker Hand O7 platform.
>
---
#### [new 056] Safe Model Predictive Diffusion with Shielding
- **分类: cs.RO**

- **简介: 该论文针对复杂机器人系统的安全轨迹生成问题，提出无需训练的Safe MPD方法，结合模型预测扩散与安全屏蔽机制，在去噪过程中保证轨迹的运动学可行性与安全性，避免后处理缺陷，显著提升成功率与安全性。**

- **链接: [https://arxiv.org/pdf/2512.06261v1](https://arxiv.org/pdf/2512.06261v1)**

> **作者:** Taekyung Kim; Keyvan Majd; Hideki Okamoto; Bardh Hoxha; Dimitra Panagou; Georgios Fainekos
>
> **备注:** Project page: https://www.taekyung.me/safe-mpd
>
> **摘要:** Generating safe, kinodynamically feasible, and optimal trajectories for complex robotic systems is a central challenge in robotics. This paper presents Safe Model Predictive Diffusion (Safe MPD), a training-free diffusion planner that unifies a model-based diffusion framework with a safety shield to generate trajectories that are both kinodynamically feasible and safe by construction. By enforcing feasibility and safety on all samples during the denoising process, our method avoids the common pitfalls of post-processing corrections, such as computational intractability and loss of feasibility. We validate our approach on challenging non-convex planning problems, including kinematic and acceleration-controlled tractor-trailer systems. The results show that it substantially outperforms existing safety strategies in success rate and safety, while achieving sub-second computation times.
>
---
#### [new 057] OptMap: Geometric Map Distillation via Submodular Maximization
- **分类: cs.RO**

- **简介: 该论文属于机器人自主导航任务，旨在解决LiDAR地图数据冗余与应用需求多样化的矛盾。作者提出OptMap，通过子模最大化算法实现高效、实时、任务定制的几何地图蒸馏，提升信息密度并降低计算开销。**

- **链接: [https://arxiv.org/pdf/2512.07775v1](https://arxiv.org/pdf/2512.07775v1)**

> **作者:** David Thorne; Nathan Chan; Christa S. Robison; Philip R. Osteen; Brett T. Lopez
>
> **摘要:** Autonomous robots rely on geometric maps to inform a diverse set of perception and decision-making algorithms. As autonomy requires reasoning and planning on multiple scales of the environment, each algorithm may require a different map for optimal performance. Light Detection And Ranging (LiDAR) sensors generate an abundance of geometric data to satisfy these diverse requirements, but selecting informative, size-constrained maps is computationally challenging as it requires solving an NP-hard combinatorial optimization. In this work we present OptMap: a geometric map distillation algorithm which achieves real-time, application-specific map generation via multiple theoretical and algorithmic innovations. A central feature is the maximization of set functions that exhibit diminishing returns, i.e., submodularity, using polynomial-time algorithms with provably near-optimal solutions. We formulate a novel submodular reward function which quantifies informativeness, reduces input set sizes, and minimizes bias in sequentially collected datasets. Further, we propose a dynamically reordered streaming submodular algorithm which improves empirical solution quality and addresses input order bias via an online approximation of the value of all scans. Testing was conducted on open-source and custom datasets with an emphasis on long-duration mapping sessions, highlighting OptMap's minimal computation requirements. Open-source ROS1 and ROS2 packages are available and can be used alongside any LiDAR SLAM algorithm.
>
---
#### [new 058] MIND-V: Hierarchical Video Generation for Long-Horizon Robotic Manipulation with RL-based Physical Alignment
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作视频生成任务，旨在解决长视野、多样化操作数据稀缺的问题。提出MIND-V框架，通过分层结构与基于强化学习的物理对齐，实现语义连贯且符合物理规律的长时域视频生成。**

- **链接: [https://arxiv.org/pdf/2512.06628v1](https://arxiv.org/pdf/2512.06628v1)**

> **作者:** Ruicheng Zhang; Mingyang Zhang; Jun Zhou; Zhangrui Guo; Xiaofan Liu; Zunnan Xu; Zhizhou Zhong; Puxin Yan; Haocheng Luo; Xiu Li
>
> **摘要:** Embodied imitation learning is constrained by the scarcity of diverse, long-horizon robotic manipulation data. Existing video generation models for this domain are limited to synthesizing short clips of simple actions and often rely on manually defined trajectories. To this end, we introduce MIND-V, a hierarchical framework designed to synthesize physically plausible and logically coherent videos of long-horizon robotic manipulation. Inspired by cognitive science, MIND-V bridges high-level reasoning with pixel-level synthesis through three core components: a Semantic Reasoning Hub (SRH) that leverages a pre-trained vision-language model for task planning; a Behavioral Semantic Bridge (BSB) that translates abstract instructions into domain-invariant representations; and a Motor Video Generator (MVG) for conditional video rendering. MIND-V employs Staged Visual Future Rollouts, a test-time optimization strategy to enhance long-horizon robustness. To align the generated videos with physical laws, we introduce a GRPO reinforcement learning post-training phase guided by a novel Physical Foresight Coherence (PFC) reward. PFC leverages the V-JEPA world model to enforce physical plausibility by aligning the predicted and actual dynamic evolutions in the feature space. MIND-V demonstrates state-of-the-art performance in long-horizon robotic manipulation video generation, establishing a scalable and controllable paradigm for embodied data synthesis.
>
---
#### [new 059] Toward Seamless Physical Human-Humanoid Interaction: Insights from Control, Intent, and Modeling with a Vision for What Comes Next
- **分类: cs.RO**

- **简介: 该论文综述物理人-人形机器人交互（pHHI）领域，旨在解决交互中建模、控制与意图识别割裂的问题。通过梳理三大支柱：人形机器人控制、人类意图估计与计算人体模型，提出跨模块融合路径，并构建统一交互分类体系，推动安全、直观的协同交互发展。**

- **链接: [https://arxiv.org/pdf/2512.07765v1](https://arxiv.org/pdf/2512.07765v1)**

> **作者:** Gustavo A. Cardona; Shubham S. Kumbhar; Panagiotis Artemiadis
>
> **备注:** 60 pages, 5 figures, 3 tables
>
> **摘要:** Physical Human-Humanoid Interaction (pHHI) is a rapidly advancing field with significant implications for deploying robots in unstructured, human-centric environments. In this review, we examine the current state of the art in pHHI through three core pillars: (i) humanoid modeling and control, (ii) human intent estimation, and (iii) computational human models. For each pillar, we survey representative approaches, identify open challenges, and analyze current limitations that hinder robust, scalable, and adaptive interaction. These include the need for whole-body control strategies capable of handling uncertain human dynamics, real-time intent inference under limited sensing, and modeling techniques that account for variability in human physical states. Although significant progress has been made within each domain, integration across pillars remains limited. We propose pathways for unifying methods across these areas to enable cohesive interaction frameworks. This structure enables us not only to map the current landscape but also to propose concrete directions for future research that aim to bridge these domains. Additionally, we introduce a unified taxonomy of interaction types based on modality, distinguishing between direct interactions (e.g., physical contact) and indirect interactions (e.g., object-mediated), and on the level of robot engagement, ranging from assistance to cooperation and collaboration. For each category in this taxonomy, we provide the three core pillars that highlight opportunities for cross-pillar unification. Our goal is to suggest avenues to advance robust, safe, and intuitive physical interaction, providing a roadmap for future research that will allow humanoid systems to effectively understand, anticipate, and collaborate with human partners in diverse real-world settings.
>
---
#### [new 060] Embodied Referring Expression Comprehension in Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文研究具身指代表达理解任务，旨在解决现有数据集视角单一、缺乏非语言手势及室内外场景不足的问题。作者提出Refer360数据集和MuRes模型，提升机器人对多模态人类指令的理解能力。**

- **链接: [https://arxiv.org/pdf/2512.06558v1](https://arxiv.org/pdf/2512.06558v1)**

> **作者:** Md Mofijul Islam; Alexi Gladstone; Sujan Sarker; Ganesh Nanduru; Md Fahim; Keyan Du; Aman Chadha; Tariq Iqbal
>
> **备注:** 14 pages, 7 figures, accepted at the ACM/IEEE International Conference on Human-Robot Interaction (HRI) 2026
>
> **摘要:** As robots enter human workspaces, there is a crucial need for them to comprehend embodied human instructions, enabling intuitive and fluent human-robot interaction (HRI). However, accurate comprehension is challenging due to a lack of large-scale datasets that capture natural embodied interactions in diverse HRI settings. Existing datasets suffer from perspective bias, single-view collection, inadequate coverage of nonverbal gestures, and a predominant focus on indoor environments. To address these issues, we present the Refer360 dataset, a large-scale dataset of embodied verbal and nonverbal interactions collected across diverse viewpoints in both indoor and outdoor settings. Additionally, we introduce MuRes, a multimodal guided residual module designed to improve embodied referring expression comprehension. MuRes acts as an information bottleneck, extracting salient modality-specific signals and reinforcing them into pre-trained representations to form complementary features for downstream tasks. We conduct extensive experiments on four HRI datasets, including the Refer360 dataset, and demonstrate that current multimodal models fail to capture embodied interactions comprehensively; however, augmenting them with MuRes consistently improves performance. These findings establish Refer360 as a valuable benchmark and exhibit the potential of guided residual learning to advance embodied referring expression comprehension in robots operating within human environments.
>
---
#### [new 061] Gait-Adaptive Perceptive Humanoid Locomotion with Real-Time Under-Base Terrain Reconstruction
- **分类: cs.RO**

- **简介: 该论文研究全尺寸人形机器人在复杂地形下的稳定行走问题，提出一种融合实时地形感知、步态调节与全身控制的强化学习框架，通过下视深度相机和U-Net构建实时高度图，实现自适应步态与平衡控制。**

- **链接: [https://arxiv.org/pdf/2512.07464v1](https://arxiv.org/pdf/2512.07464v1)**

> **作者:** Haolin Song; Hongbo Zhu; Tao Yu; Yan Liu; Mingqi Yuan; Wengang Zhou; Hua Chen; Houqiang Li
>
> **摘要:** For full-size humanoid robots, even with recent advances in reinforcement learning-based control, achieving reliable locomotion on complex terrains, such as long staircases, remains challenging. In such settings, limited perception, ambiguous terrain cues, and insufficient adaptation of gait timing can cause even a single misplaced or mistimed step to result in rapid loss of balance. We introduce a perceptive locomotion framework that merges terrain sensing, gait regulation, and whole-body control into a single reinforcement learning policy. A downward-facing depth camera mounted under the base observes the support region around the feet, and a compact U-Net reconstructs a dense egocentric height map from each frame in real time, operating at the same frequency as the control loop. The perceptual height map, together with proprioceptive observations, is processed by a unified policy that produces joint commands and a global stepping-phase signal, allowing gait timing and whole-body posture to be adapted jointly to the commanded motion and local terrain geometry. We further adopt a single-stage successive teacher-student training scheme for efficient policy learning and knowledge transfer. Experiments conducted on a 31-DoF, 1.65 m humanoid robot demonstrate robust locomotion in both simulation and real-world settings, including forward and backward stair ascent and descent, as well as crossing a 46 cm gap. Project Page:https://ga-phl.github.io/
>
---
#### [new 062] WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究自动驾驶中端到端轨迹规划任务，提出WAM-Flow模型，将规划建模为离散流匹配问题。通过并行双向去噪实现粗到精的生成，结合几何感知损失与强化学习对齐，提升规划效率与性能，在NAVSIM上取得优于自回归和扩散模型的效果。**

- **链接: [https://arxiv.org/pdf/2512.06112v1](https://arxiv.org/pdf/2512.06112v1)**

> **作者:** Yifang Xu; Jiahao Cui; Feipeng Cai; Zhihao Zhu; Hanlin Shang; Shan Luan; Mingwang Xu; Neng Zhang; Yaoyi Li; Jia Cai; Siyu Zhu
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** We introduce WAM-Flow, a vision-language-action (VLA) model that casts ego-trajectory planning as discrete flow matching over a structured token space. In contrast to autoregressive decoders, WAM-Flow performs fully parallel, bidirectional denoising, enabling coarse-to-fine refinement with a tunable compute-accuracy trade-off. Specifically, the approach combines a metric-aligned numerical tokenizer that preserves scalar geometry via triplet-margin learning, a geometry-aware flow objective and a simulator-guided GRPO alignment that integrates safety, ego progress, and comfort rewards while retaining parallel generation. A multi-stage adaptation converts a pre-trained auto-regressive backbone (Janus-1.5B) from causal decoding to non-causal flow model and strengthens road-scene competence through continued multimodal pretraining. Thanks to the inherent nature of consistency model training and parallel decoding inference, WAM-Flow achieves superior closed-loop performance against autoregressive and diffusion-based VLA baselines, with 1-step inference attaining 89.1 PDMS and 5-step inference reaching 90.3 PDMS on NAVSIM v1 benchmark. These results establish discrete flow matching as a new promising paradigm for end-to-end autonomous driving. The code will be publicly available soon.
>
---
#### [new 063] Proportional integral derivative booster for neural networks-based time-series prediction: Case of water demand prediction
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对多步时间序列预测中神经网络精度与复杂度的平衡问题，提出一种受PID控制启发的增强方法。通过调整预测值以逼近真实值，提升水需求和能耗预测精度，且不增加模型复杂度。**

- **链接: [https://arxiv.org/pdf/2512.06357v1](https://arxiv.org/pdf/2512.06357v1)**

> **作者:** Tony Sallooma; Okyay Kaynak; Xinbo Yub; Wei He
>
> **备注:** Engineering Applications of Artificial Intelligence 2022
>
> **摘要:** Multi-step time-series prediction is an essential supportive step for decision-makers in several industrial areas. Artificial intelligence techniques, which use a neural network component in various forms, have recently frequently been used to accomplish this step. However, the complexity of the neural network structure still stands up as a critical problem against prediction accuracy. In this paper, a method inspired by the proportional-integral-derivative (PID) control approach is investigated to enhance the performance of neural network models used for multi-step ahead prediction of periodic time-series information while maintaining a negligible impact on the complexity of the system. The PID-based method is applied to the predicted value at each time step to bring that value closer to the real value. The water demand forecasting problem is considered as a case study, where two deep neural network models from the literature are used to prove the effectiveness of the proposed boosting method. Furthermore, to prove the applicability of this PID-based booster to other types of periodic time-series prediction problems, it is applied to enhance the accuracy of a neural network model used for multi-step forecasting of hourly energy consumption. The comparison between the results of the original prediction models and the results after using the proposed technique demonstrates the superiority of the proposed method in terms of prediction accuracy and system complexity.
>
---
#### [new 064] Method of UAV Inspection of Photovoltaic Modules Using Thermal and RGB Data Fusion
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究无人机对光伏组件的智能巡检，解决传统方法中热成像偏差、数据冗余和带宽过高等问题。提出融合热成像与RGB图像的多模态框架，结合自适应重采集和地理去重模块，实现全自动、高精度、低通信开销的缺陷检测与报警。**

- **链接: [https://arxiv.org/pdf/2512.06504v1](https://arxiv.org/pdf/2512.06504v1)**

> **作者:** Andrii Lysyi; Anatoliy Sachenko; Pavlo Radiuk; Mykola Lysyi; Oleksandr Melnychenko; Diana Zahorodnia
>
> **摘要:** The subject of this research is the development of an intelligent, integrated framework for the automated inspection of photovoltaic (PV) infrastructure that addresses the critical shortcomings of conventional methods, including thermal palette bias, data redundancy, and high communication bandwidth requirements. The goal of this study is to design, develop, and validate a comprehensive, multi-modal system that fully automates the monitoring workflow, from data acquisition to the generation of actionable, geo-located maintenance alerts, thereby enhancing plant safety and operational efficiency. The methods employed involve a synergistic architecture that begins with a palette-invariant thermal embedding, learned by enforcing representational consistency, which is fused with a contrast-normalized RGB stream via a gated mechanism. This is supplemented by a closed-loop, adaptive re-acquisition controller that uses Rodrigues-based updates for targeted confirmation of ambiguous anomalies and a geospatial deduplication module that clusters redundant alerts using DBSCAN over the haversine distance. In conclusion, this study establishes a powerful new paradigm for proactive PV inspection, with the proposed system achieving a mean Average Precision (mAP@0.5) of 0.903 on the public PVF-10 benchmark, a significant 12-15% improvement over single-modality baselines. Field validation confirmed the system's readiness, achieving 96% recall, while the de-duplication process reduced duplicate-induced false positives by 15-20%, and relevance-only telemetry cut airborne data transmission by 60-70%.
>
---
#### [new 065] Unifying Entropy Regularization in Optimal Control: From and Back to Classical Objectives via Iterated Soft Policies and Path Integral Solutions
- **分类: math.OC; cs.LG; cs.RO; eess.SY**

- **简介: 该论文研究熵正则化在最优控制中的统一框架，旨在通过分离策略与状态转移的KL正则项，推广经典随机与风险敏感控制。提出软策略形式，证明其可迭代收敛至原问题，并发现特定参数下具备路径积分解与线性Bellman方程等优良性质。**

- **链接: [https://arxiv.org/pdf/2512.06109v1](https://arxiv.org/pdf/2512.06109v1)**

> **作者:** Ajinkya Bhole; Mohammad Mahmoudi Filabadi; Guillaume Crevecoeur; Tom Lefebvre
>
> **摘要:** This paper develops a unified perspective on several stochastic optimal control formulations through the lens of Kullback-Leibler regularization. We propose a central problem that separates the KL penalties on policies and transitions, assigning them independent weights, thereby generalizing the standard trajectory-level KL-regularization commonly used in probabilistic and KL-regularized control. This generalized formulation acts as a generative structure allowing to recover various control problems. These include the classical Stochastic Optimal Control (SOC), Risk-Sensitive Optimal Control (RSOC), and their policy-based KL-regularized counterparts. The latter we refer to as soft-policy SOC and RSOC, facilitating alternative problems with tractable solutions. Beyond serving as regularized variants, we show that these soft-policy formulations majorize the original SOC and RSOC problem. This means that the regularized solution can be iterated to retrieve the original solution. Furthermore, we identify a structurally synchronized case of the risk-seeking soft-policy RSOC formulation, wherein the policy and transition KL-regularization weights coincide. Remarkably, this specific setting gives rise to several powerful properties such as a linear Bellman equation, path integral solution, and, compositionality, thereby extending these computationally favourable properties to a broad class of control problems.
>
---
#### [new 066] FishDetector-R1: Unified MLLM-Based Framework with Reinforcement Fine-Tuning for Weakly Supervised Fish Detection, Segmentation, and Counting
- **分类: cs.CV; cs.CY; cs.RO; eess.IV**

- **简介: 该论文针对水下鱼类图像因标注成本高导致的弱监督检测难题，提出FishDetector-R1框架，结合新型提示与强化学习方法，在检测、分割和计数任务中显著提升性能，并具备良好跨域鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.05996v1](https://arxiv.org/pdf/2512.05996v1)**

> **作者:** Yi Liu; Jingyu Song; Vedanth Kallakuri; Katherine A. Skinner
>
> **备注:** 18 pages, under review
>
> **摘要:** Analyzing underwater fish imagery is critical for ecological monitoring but remains difficult due to visual degradation and costly annotations. We introduce FishDetector-R1, a unified MLLM-based framework for fish detection, segmentation, and counting under weak supervision. On the DeepFish dataset, our framework achieves substantial gains over baselines, improving AP by 20% and mIoU by 10%, while reducing MAE by 30% and GAME by 35%. These improvements stem from two key components: a novel detect-to-count prompt that enforces spatially consistent detections and counts, and Reinforcement Learning from Verifiable Reward (RLVR) with a complementary scalable paradigm leveraging sparse point labels. Ablation studies further validate the effectiveness of this reward design. Moreover, the improvement generalizes well to other underwater datasets, confirming strong cross-domain robustness. Overall, FishDetector-R1 provides a reliable and scalable solution for accurate marine visual understanding via weak supervision. The project page for FishDetector-R1 is https://umfieldrobotics.github.io/FishDetector-R1.
>
---
#### [new 067] sim2art: Accurate Articulated Object Modeling from a Single Video using Synthetic Training Data Only
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究单目视频中可动部件物体的建模任务，旨在从自由移动相机拍摄的视频中联合预测物体部分分割与关节参数。仅用合成数据训练，实现了在真实场景中的良好泛化，支持动态环境下的实时应用。**

- **链接: [https://arxiv.org/pdf/2512.07698v1](https://arxiv.org/pdf/2512.07698v1)**

> **作者:** Arslan Artykov; Corentin Sautier; Vincent Lepetit
>
> **摘要:** Understanding articulated objects is a fundamental challenge in robotics and digital twin creation. To effectively model such objects, it is essential to recover both part segmentation and the underlying joint parameters. Despite the importance of this task, previous work has largely focused on setups like multi-view systems, object scanning, or static cameras. In this paper, we present the first data-driven approach that jointly predicts part segmentation and joint parameters from monocular video captured with a freely moving camera. Trained solely on synthetic data, our method demonstrates strong generalization to real-world objects, offering a scalable and practical solution for articulated object understanding. Our approach operates directly on casually recorded video, making it suitable for real-time applications in dynamic environments. Project webpage: https://aartykov.github.io/sim2art/
>
---
#### [new 068] UltrasODM: A Dual Stream Optical Flow Mamba Network for 3D Freehand Ultrasound Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自由手超声重建中因操作导致的误差问题，提出UltrasODM双流网络，通过光学流、Mamba模块与不确定性感知的闭环反馈，提升6-DoF位姿估计精度，并引入医生交互机制增强重建可靠性与临床可用性。**

- **链接: [https://arxiv.org/pdf/2512.07756v1](https://arxiv.org/pdf/2512.07756v1)**

> **作者:** Mayank Anand; Ujair Alam; Surya Prakash; Priya Shukla; Gora Chand Nandi; Domenec Puig
>
> **摘要:** Clinical ultrasound acquisition is highly operator-dependent, where rapid probe motion and brightness fluctuations often lead to reconstruction errors that reduce trust and clinical utility. We present UltrasODM, a dual-stream framework that assists sonographers during acquisition through calibrated per-frame uncertainty, saliency-based diagnostics, and actionable prompts. UltrasODM integrates (i) a contrastive ranking module that groups frames by motion similarity, (ii) an optical-flow stream fused with Dual-Mamba temporal modules for robust 6-DoF pose estimation, and (iii) a Human-in-the-Loop (HITL) layer combining Bayesian uncertainty, clinician-calibrated thresholds, and saliency maps highlighting regions of low confidence. When uncertainty exceeds the threshold, the system issues unobtrusive alerts suggesting corrective actions such as re-scanning highlighted regions or slowing the sweep. Evaluated on a clinical freehand ultrasound dataset, UltrasODM reduces drift by 15.2%, distance error by 12.1%, and Hausdorff distance by 10.1% relative to UltrasOM, while producing per-frame uncertainty and saliency outputs. By emphasizing transparency and clinician feedback, UltrasODM improves reconstruction reliability and supports safer, more trustworthy clinical workflows. Our code is publicly available at https://github.com/AnandMayank/UltrasODM.
>
---
#### [new 069] AQUILA: A QUIC-Based Link Architecture for Resilient Long-Range UAV Communication
- **分类: cs.NI; cs.RO**

- **简介: 该论文针对远距离无人机通信中延迟高、可靠性低的问题，提出基于QUIC的跨层架构AQUILA，通过统一传输、优先级调度和自适应拥塞控制，实现控制指令低延迟与视频高吞吐的协同传输，提升链路鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.06889v1](https://arxiv.org/pdf/2512.06889v1)**

> **作者:** Ximing Huang; Yirui Rao
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** The proliferation of autonomous Unmanned Aerial Vehicles (UAVs) in Beyond Visual Line of Sight (BVLOS) applications is critically dependent on resilient, high-bandwidth, and low-latency communication links. Existing solutions face critical limitations: TCP's head-of-line blocking stalls time-sensitive data, UDP lacks reliability and congestion control, and cellular networks designed for terrestrial users degrade severely for aerial platforms. This paper introduces AQUILA, a cross-layer communication architecture built on QUIC to address these challenges. AQUILA contributes three key innovations: (1) a unified transport layer using QUIC's reliable streams for MAVLink Command and Control (C2) and unreliable datagrams for video, eliminating head-of-line blocking under unified congestion control; (2) a priority scheduling mechanism that structurally ensures C2 latency remains bounded and independent of video traffic intensity; (3) a UAV-adapted congestion control algorithm extending SCReAM with altitude-adaptive delay targeting and telemetry headroom reservation. AQUILA further implements 0-RTT connection resumption to minimize handover blackouts with application-layer replay protection, deployed over an IP-native architecture enabling global operation. Experimental validation demonstrates that AQUILA significantly outperforms TCP- and UDP-based approaches in C2 latency, video quality, and link resilience under realistic conditions, providing a robust foundation for autonomous BVLOS missions.
>
---
#### [new 070] A Novel Deep Neural Network Architecture for Real-Time Water Demand Forecasting
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于短时水需求预测任务，旨在解决深度学习模型复杂度高及极端点预测误差大的问题。提出一种基于GRU和K-means的新型低复杂度神经网络，并通过数据扩展降低非线性，显著减少误差与模型参数。**

- **链接: [https://arxiv.org/pdf/2512.06714v1](https://arxiv.org/pdf/2512.06714v1)**

> **作者:** Tony Salloom; Okyay Kaynak; Wei He
>
> **摘要:** Short-term water demand forecasting (StWDF) is the foundation stone in the derivation of an optimal plan for controlling water supply systems. Deep learning (DL) approaches provide the most accurate solutions for this purpose. However, they suffer from complexity problem due to the massive number of parameters, in addition to the high forecasting error at the extreme points. In this work, an effective method to alleviate the error at these points is proposed. It is based on extending the data by inserting virtual data within the actual data to relieve the nonlinearity around them. To our knowledge, this is the first work that considers the problem related to the extreme points. Moreover, the water demand forecasting model proposed in this work is a novel DL model with relatively low complexity. The basic model uses the gated recurrent unit (GRU) to handle the sequential relationship in the historical demand data, while an unsupervised classification method, K-means, is introduced for the creation of new features to enhance the prediction accuracy with less number of parameters. Real data obtained from two different water plants in China are used to train and verify the model proposed. The prediction results and the comparison with the state-of-the-art illustrate that the method proposed reduces the complexity of the model six times of what achieved in the literature while conserving the same accuracy. Furthermore, it is found that extending the data set significantly reduces the error by about 30%. However, it increases the training time.
>
---
#### [new 071] More than Segmentation: Benchmarking SAM 3 for Segmentation, 3D Perception, and Reconstruction in Robotic Surgery
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于医疗图像分析与手术机器人领域，旨在评估SAM 3在手术场景中的分割、3D感知与重建能力。研究评测了其零样本分割、语言提示分割、视频跟踪及3D结构重建性能，揭示其在动态手术场景中的优势与局限。**

- **链接: [https://arxiv.org/pdf/2512.07596v1](https://arxiv.org/pdf/2512.07596v1)**

> **作者:** Wenzhen Dong; Jieming Yu; Yiming Huang; Hongqiu Wang; Lei Zhu; Albert C. S. Chung; Hongliang Ren; Long Bai
>
> **备注:** Technical Report
>
> **摘要:** The recent Segment Anything Model (SAM) 3 has introduced significant advancements over its predecessor, SAM 2, particularly with the integration of language-based segmentation and enhanced 3D perception capabilities. SAM 3 supports zero-shot segmentation across a wide range of prompts, including point, bounding box, and language-based prompts, allowing for more flexible and intuitive interactions with the model. In this empirical evaluation, we assess the performance of SAM 3 in robot-assisted surgery, benchmarking its zero-shot segmentation with point and bounding box prompts and exploring its effectiveness in dynamic video tracking, alongside its newly introduced language prompt segmentation. While language prompts show potential, their performance in the surgical domain is currently suboptimal, highlighting the need for further domain-specific training. Additionally, we investigate SAM 3's 3D reconstruction abilities, demonstrating its capacity to process surgical scene data and reconstruct 3D anatomical structures from 2D images. Through comprehensive testing on the MICCAI EndoVis 2017 and EndoVis 2018 benchmarks, SAM 3 shows clear improvements over SAM and SAM 2 in both image and video segmentation under spatial prompts, while zero-shot evaluations on SCARED, StereoMIS, and EndoNeRF indicate strong monocular depth estimation and realistic 3D instrument reconstruction, yet also reveal remaining limitations in complex, highly dynamic surgical scenes.
>
---
#### [new 072] Characterizing Lane-Changing Behavior in Mixed Traffic
- **分类: cs.MA; cs.GT; cs.RO; eess.SY**

- **简介: 该论文研究混合交通中自动驾驶车辆（AV）与人类驾驶车辆（HDV）的换道行为交互，旨在揭示合作行为特征及演化规律。基于Waymo数据，采用聚类与博弈论方法分析7,636次换道事件，量化合作程度并构建效用模型，发现AV更倾向合作，且重复交互促进合作演化。**

- **链接: [https://arxiv.org/pdf/2512.07219v1](https://arxiv.org/pdf/2512.07219v1)**

> **作者:** Sungyong Chung; Alireza Talebpour; Samer H. Hamdar
>
> **摘要:** Characterizing and understanding lane-changing behavior in the presence of automated vehicles (AVs) is crucial to ensuring safety and efficiency in mixed traffic. Accordingly, this study aims to characterize the interactions between the lane-changing vehicle (active vehicle) and the vehicle directly impacted by the maneuver in the target lane (passive vehicle). Utilizing real-world trajectory data from the Waymo Open Motion Dataset (WOMD), this study explores patterns in lane-changing behavior and provides insight into how these behaviors evolve under different AV market penetration rates (MPRs). In particular, we propose a game-theoretic framework to analyze cooperative and defective behaviors in mixed traffic, applied to the 7,636 observed lane-changing events in the WOMD. First, we utilize k-means clustering to classify vehicles as cooperative or defective, revealing that the proportions of cooperative AVs are higher than those of HDVs in both active and passive roles. Next, we jointly estimate the utilities of active and passive vehicles to model their behaviors using the quantal response equilibrium framework. Empirical payoff tables are then constructed based on these utilities. Using these payoffs, we analyze the presence of social dilemmas and examine the evolution of cooperative behaviors using evolutionary game theory. Our results reveal the presence of social dilemmas in approximately 4% and 11% of lane-changing events for active and passive vehicles, respectively, with most classified as Stag Hunt or Prisoner's Dilemma (Chicken Game rarely observed). Moreover, the Monte Carlo simulation results show that repeated lane-changing interactions consistently lead to increased cooperative behavior over time, regardless of the AV penetration rate.
>
---
#### [new 073] VAT: Vision Action Transformer by Unlocking Full Representation of ViT
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对机器人学习中视觉Transformer仅用最后一层特征导致表征不足的问题，提出Vision Action Transformer（VAT），通过全层特征融合实现感知与动作生成的深度结合，在模仿学习任务中显著提升性能，达到SOTA。**

- **链接: [https://arxiv.org/pdf/2512.06013v1](https://arxiv.org/pdf/2512.06013v1)**

> **作者:** Wenhao Li; Chengwei Ma; Weixin Mao
>
> **摘要:** In robot learning, Vision Transformers (ViTs) are standard for visual perception, yet most methods discard valuable information by using only the final layer's features. We argue this provides an insufficient representation and propose the Vision Action Transformer (VAT), a novel architecture that is extended from ViT and unlocks the full feature hierarchy of ViT. VAT processes specialized action tokens with visual features across all transformer layers, enabling a deep and progressive fusion of perception and action generation. On a suite of simulated manipulation tasks, VAT achieves a 98.15\% average success rate across four LIBERO benchmarks, establishing a new state-of-the-art by outperforming prior methods like OpenVLA-OFT. Our work presents not only a powerful model for imitation learning but also demonstrates the critical importance of leveraging the complete ''representation trajectory'' of vision models to advance robotic policy. The GitHub URL for the project code is https://github.com/sellerbubble/VAT.
>
---
#### [new 074] Deep Neural Network-Based Aerial Transport in the Presence of Cooperative and Uncooperative UAS
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究无人机群在合作与非合作环境下的自主运输任务，旨在实现鲁棒、稳定的分布式覆盖控制。提出基于深度神经网络的分层通信架构，通过动态调整通信边与前向调度机制，保障系统在非合作干扰下的收敛性与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.06577v1](https://arxiv.org/pdf/2512.06577v1)**

> **作者:** Muhammad Junayed Hasan Zahed; Hossein Rastgoftar
>
> **摘要:** We present a resilient deep neural network (DNN) framework for decentralized transport and coverage using uncrewed aerial systems (UAS) operating in $\mathbb{R}^n$. The proposed DNN-based mass-transport architecture constructs a layered inter-UAS communication graph from an initial formation, assigns time-varying communication weights through a forward scheduling mechanism that guides the team from the initial to the final configuration, and ensures stability and convergence of the resulting multi-agent transport dynamics. The framework is explicitly designed to remain robust in the presence of uncooperative agents that deviate from or refuse to follow the prescribed protocol. Our method preserves a fixed feed-forward topology but dynamically prunes edges to uncooperative agents, maintains convex, feedforward mentoring among cooperative agents, and computes global desired set points through a sparse linear relation consistent with leader references. The target set is abstracted by $N$ points that become final desired positions, enabling coverage-optimal transport while keeping computation low and guarantees intact. Extensive simulations demonstrate that, under full cooperation, all agents converge rapidly to the target zone with a 10\% boundary margin and under partial cooperation with uncooperative agents, the system maintains high convergence among cooperative agents with performance degradation localized near the disruptions, evidencing graceful resilience and scalability. These results confirm that forward-weight scheduling, hierarchical mentor--mentee coordination, and on-the-fly DNN restructuring yield robust, provably stable UAS transport in realistic fault scenarios.
>
---
#### [new 075] Obstacle Avoidance of UAV in Dynamic Environments Using Direction and Velocity-Adaptive Artificial Potential Field
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究无人机在动态环境中的避障任务，解决传统人工势场法易陷入局部极小及忽略障碍物运动状态的问题。提出方向与相对速度自适应的人工序场方法，并结合模型预测控制实现安全、平滑的避障。**

- **链接: [https://arxiv.org/pdf/2512.07609v1](https://arxiv.org/pdf/2512.07609v1)**

> **作者:** Nikita Vaibhav Pavle; Shrreya Rajneesh; Rakesh Kumar Sahoo; Manoranjan Sinha
>
> **摘要:** The conventional Artificial Potential Field (APF) is fundamentally limited by the local minima issue and its inability to account for the kinematics of moving obstacles. This paper addresses the critical challenge of autonomous collision avoidance for Unmanned Aerial Vehicles (UAVs) operating in dynamic and cluttered airspace by proposing a novel Direction and Relative Velocity Weighted Artificial Potential Field (APF). In this approach, a bounded weighting function, $ω(θ,v_{e})$, is introduced to dynamically scale the repulsive potential based on the direction and velocity of the obstacle relative to the UAV. This robust APF formulation is integrated within a Model Predictive Control (MPC) framework to generate collision-free trajectories while adhering to kinematic constraints. Simulation results demonstrate that the proposed method effectively resolves local minima and significantly enhances safety by enabling smooth, predictive avoidance maneuvers. The system ensures superior path integrity and reliable performance, confirming its viability for autonomous navigation in complex environments.
>
---
#### [new 076] KAN-Dreamer: Benchmarking Kolmogorov-Arnold Networks as Function Approximators in World Models
- **分类: cs.LG; cs.AI; cs.CV; cs.NE; cs.RO**

- **简介: 该论文研究Kolmogorov-Arnold网络（KAN）在DreamerV3世界模型中的应用，旨在提升参数效率与可解释性。作者提出KAN-Dreamer，将KAN/FastKAN替代MLP组件，并优化实现以保持效率。实验表明FastKAN在性能和训练速度上与原模型相当。**

- **链接: [https://arxiv.org/pdf/2512.07437v1](https://arxiv.org/pdf/2512.07437v1)**

> **作者:** Chenwei Shi; Xueyu Luan
>
> **备注:** 23 pages, 8 figures, 3 tables
>
> **摘要:** DreamerV3 is a state-of-the-art online model-based reinforcement learning (MBRL) algorithm known for remarkable sample efficiency. Concurrently, Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to Multi-Layer Perceptrons (MLPs), offering superior parameter efficiency and interpretability. To mitigate KANs' computational overhead, variants like FastKAN leverage Radial Basis Functions (RBFs) to accelerate inference. In this work, we investigate integrating KAN architectures into the DreamerV3 framework. We introduce KAN-Dreamer, replacing specific MLP and convolutional components of DreamerV3 with KAN and FastKAN layers. To ensure efficiency within the JAX-based World Model, we implement a tailored, fully vectorized version with simplified grid management. We structure our investigation into three subsystems: Visual Perception, Latent Prediction, and Behavior Learning. Empirical evaluations on the DeepMind Control Suite (walker_walk) analyze sample efficiency, training time, and asymptotic performance. Experimental results demonstrate that utilizing our adapted FastKAN as a drop-in replacement for the Reward and Continue predictors yields performance on par with the original MLP-based architecture, maintaining parity in both sample efficiency and training speed. This report serves as a preliminary study for future developments in KAN-based world models.
>
---
#### [new 077] Beyond Model Jailbreak: Systematic Dissection of the "Ten DeadlySins" in Embodied Intelligence
- **分类: cs.CR; cs.RO**

- **简介: 该论文聚焦具身智能系统安全，揭示Unitree Go2平台中存在的“十大漏洞”。通过多层分析技术，发现跨无线、核心模块与外部接口的系统性风险，表明仅保护模型不足以确保安全，需全栈防护。**

- **链接: [https://arxiv.org/pdf/2512.06387v1](https://arxiv.org/pdf/2512.06387v1)**

> **作者:** Yuhang Huang; Junchao Li; Boyang Ma; Xuelong Dai; Minghui Xu; Kaidi Xu; Yue Zhang; Jianping Wang; Xiuzhen Cheng
>
> **摘要:** Embodied AI systems integrate language models with real world sensing, mobility, and cloud connected mobile apps. Yet while model jailbreaks have drawn significant attention, the broader system stack of embodied intelligence remains largely unexplored. In this work, we conduct the first holistic security analysis of the Unitree Go2 platform and uncover ten cross layer vulnerabilities the "Ten Sins of Embodied AI Security." Using BLE sniffing, traffic interception, APK reverse engineering, cloud API testing, and hardware probing, we identify systemic weaknesses across three architectural layers: wireless provisioning, core modules, and external interfaces. These include hard coded keys, predictable handshake tokens, WiFi credential leakage, missing TLS validation, static SSH password, multilingual safety bypass behavior, insecure local relay channels, weak binding logic, and unrestricted firmware access. Together, they allow adversaries to hijack devices, inject arbitrary commands, extract sensitive information, or gain full physical control.Our findings show that securing embodied AI requires far more than aligning the model itself. We conclude with system level lessons learned and recommendations for building embodied platforms that remain robust across their entire software hardware ecosystem.
>
---
## 更新

#### [replaced 001] Pretraining in Actor-Critic Reinforcement Learning for Robot Locomotion
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人强化学习中的预训练方法，旨在解决运动技能从零学习效率低的问题。通过无任务探索收集数据，训练逆动力学模型，并将权重用于PPO算法的策略网络初始化，提升样本效率与性能。**

- **链接: [https://arxiv.org/pdf/2510.12363v2](https://arxiv.org/pdf/2510.12363v2)**

> **作者:** Jiale Fan; Andrei Cramariuc; Tifanny Portela; Marco Hutter
>
> **摘要:** The pretraining-finetuning paradigm has facilitated numerous transformative advancements in artificial intelligence research in recent years. However, in the domain of reinforcement learning (RL) for robot locomotion, individual skills are often learned from scratch despite the high likelihood that some generalizable knowledge is shared across all task-specific policies belonging to the same robot embodiment. This work aims to define a paradigm for pretraining neural network models that encapsulate such knowledge and can subsequently serve as a basis for warm-starting the RL process in classic actor-critic algorithms, such as Proximal Policy Optimization (PPO). We begin with a task-agnostic exploration-based data collection algorithm to gather diverse, dynamic transition data, which is then used to train a Proprioceptive Inverse Dynamics Model (PIDM) through supervised learning. The pretrained weights are then loaded into both the actor and critic networks to warm-start the policy optimization of actual tasks. We systematically validated our proposed method with 9 distinct robot locomotion RL environments comprising 3 different robot embodiments, showing significant benefits of this initialization strategy. Our proposed approach on average improves sample efficiency by 36.9% and task performance by 7.3% compared to random initialization. We further present key ablation studies and empirical analyses that shed light on the mechanisms behind the effectiveness of this method.
>
---
#### [replaced 002] MAPLE: Encoding Dexterous Robotic Manipulation Priors Learned From Egocentric Videos
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MAPLE方法，旨在利用第一视角人类操作视频中的接触点与手部姿态先验知识，提升灵巧机器人操作策略学习效果。属于机器人灵巧操作任务，解决了传统数据驱动方法对精细操控建模不足的问题。**

- **链接: [https://arxiv.org/pdf/2504.06084v2](https://arxiv.org/pdf/2504.06084v2)**

> **作者:** Alexey Gavryushin; Xi Wang; Robert J. S. Malate; Chenyu Yang; Davide Liconti; René Zurbrügg; Robert K. Katzschmann; Marc Pollefeys
>
> **摘要:** Large-scale egocentric video datasets capture diverse human activities across a wide range of scenarios, offering rich and detailed insights into how humans interact with objects, especially those that require fine-grained dexterous control. Such complex, dexterous skills with precise controls are crucial for many robotic manipulation tasks, yet are often insufficiently addressed by traditional data-driven approaches to robotic manipulation. To address this gap, we leverage manipulation priors learned from large-scale egocentric video datasets to improve policy learning for dexterous robotic manipulation tasks. We present MAPLE, a novel method for dexterous robotic manipulation that learns features to predict object contact points and detailed hand poses at the moment of contact from egocentric images. We then use the learned features to train policies for downstream manipulation tasks. Experimental results demonstrate the effectiveness of MAPLE across 4 existing simulation benchmarks, as well as a newly designed set of 4 challenging simulation tasks requiring fine-grained object control and complex dexterous skills. The benefits of MAPLE are further highlighted in real-world experiments using a 17 DoF dexterous robotic hand, whereas the simultaneous evaluation across both simulation and real-world experiments has remained underexplored in prior work. We additionally showcase the efficacy of our model on an egocentric contact point prediction task, validating its usefulness beyond dexterous manipulation policy learning.
>
---
#### [replaced 003] LOG-Nav: Efficient Layout-Aware Object-Goal Navigation with Hierarchical Planning
- **分类: cs.RO**

- **简介: 该论文研究物体目标导航任务，旨在解决复杂室内环境中导航效率与准确性问题。提出LOG-Nav方法，结合布局感知的分层规划与LLM代理，实现高效、无需人工干预的导航，在MP3D上显著提升成功率与路径效率。**

- **链接: [https://arxiv.org/pdf/2505.06131v3](https://arxiv.org/pdf/2505.06131v3)**

> **作者:** Jiawei Hou; Yuting Xiao; Xiangyang Xue; Taiping Zeng
>
> **摘要:** We introduce LOG-Nav, an efficient layout-aware object-goal navigation approach designed for complex multi-room indoor environments. By planning hierarchically leveraging a global topologigal map with layout information and local imperative approach with detailed scene representation memory, LOG-Nav achieves both efficient and effective navigation. The process is managed by an LLM-powered agent, ensuring seamless effective planning and navigation, without the need for human interaction, complex rewards, or costly training. Our experimental results on the MP3D benchmark achieves 85\% object navigation success rate (SR) and 79\% success rate weighted by path length (SPL) (over 40\% point improvement in SR and 60\% improvement in SPL compared to exsisting methods). Furthermore, we validate the robustness of our approach through virtual agent and real-world robotic deployment, showcasing its capability in practical scenarios.
>
---
#### [replaced 004] LabUtopia: High-Fidelity Simulation and Hierarchical Benchmark for Scientific Embodied Agents
- **分类: cs.RO; cs.SE**

- **简介: 该论文聚焦科学具身智能体的模拟与评估，旨在解决实验室环境中缺乏高保真模拟器和层次化基准的问题。作者提出LabUtopia，集成高保真多物理化学模拟、可扩展场景生成和五级复杂度基准，支持30项任务，推动感知、规划与控制在科学自动化中的发展。**

- **链接: [https://arxiv.org/pdf/2505.22634v2](https://arxiv.org/pdf/2505.22634v2)**

> **作者:** Rui Li; Zixuan Hu; Wenxi Qu; Jinouwen Zhang; Zhenfei Yin; Sha Zhang; Xuantuo Huang; Hanqing Wang; Tai Wang; Jiangmiao Pang; Wanli Ouyang; Lei Bai; Wangmeng Zuo; Ling-Yu Duan; Dongzhan Zhou; Shixiang Tang
>
> **备注:** Accepted by NeurIPS 2025 Dataset and Benchmark Track
>
> **摘要:** Scientific embodied agents play a crucial role in modern laboratories by automating complex experimental workflows. Compared to typical household environments, laboratory settings impose significantly higher demands on perception of physical-chemical transformations and long-horizon planning, making them an ideal testbed for advancing embodied intelligence. However, its development has been long hampered by the lack of suitable simulator and benchmarks. In this paper, we address this gap by introducing LabUtopia, a comprehensive simulation and benchmarking suite designed to facilitate the development of generalizable, reasoning-capable embodied agents in laboratory settings. Specifically, it integrates i) LabSim, a high-fidelity simulator supporting multi-physics and chemically meaningful interactions; ii) LabScene, a scalable procedural generator for diverse scientific scenes; and iii) LabBench, a hierarchical benchmark spanning five levels of complexity from atomic actions to long-horizon mobile manipulation. LabUtopia supports 30 distinct tasks and includes more than 200 scene and instrument assets, enabling large-scale training and principled evaluation in high-complexity environments. We demonstrate that LabUtopia offers a powerful platform for advancing the integration of perception, planning, and control in scientific-purpose agents and provides a rigorous testbed for exploring the practical capabilities and generalization limits of embodied intelligence in future research.
>
---
#### [replaced 005] Unveiling the Impact of Data and Model Scaling on High-Level Control for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文聚焦 humanoid 机器人高层控制任务，解决从人类视频中高效挖掘可学习动作表示的问题。作者构建大规模数据集 Humanoid-Union，并提出可扩展框架 SCHUR，验证了数据与模型缩放对运动生成和文本对齐的显著提升。**

- **链接: [https://arxiv.org/pdf/2511.09241v2](https://arxiv.org/pdf/2511.09241v2)**

> **作者:** Yuxi Wei; Zirui Wang; Kangning Yin; Yue Hu; Jingbo Wang; Siheng Chen
>
> **摘要:** Data scaling has long remained a critical bottleneck in robot learning. For humanoid robots, human videos and motion data are abundant and widely available, offering a free and large-scale data source. Besides, the semantics related to the motions enable modality alignment and high-level robot control learning. However, how to effectively mine raw video, extract robot-learnable representations, and leverage them for scalable learning remains an open problem. To address this, we introduce Humanoid-Union, a large-scale dataset generated through an autonomous pipeline, comprising over 260 hours of diverse, high-quality humanoid robot motion data with semantic annotations derived from human motion videos. The dataset can be further expanded via the same pipeline. Building on this data resource, we propose SCHUR, a scalable learning framework designed to explore the impact of large-scale data on high-level control in humanoid robots. Experimental results demonstrate that SCHUR achieves high robot motion generation quality and strong text-motion alignment under data and model scaling, with 37\% reconstruction improvement under MPJPE and 25\% alignment improvement under FID comparing with previous methods. Its effectiveness is further validated through deployment in real-world humanoid robot.
>
---
#### [replaced 006] High Torque Density PCB Axial Flux Permanent Magnet Motor for Micro Robots
- **分类: cs.RO**

- **简介: 该论文致力于提升微型机器人用电机的扭矩密度。针对传统绕组铜填充率低导致连续扭矩受限的问题，提出采用PCB式定子与HDI技术，实现45%高铜填充率，研制出直径19mm、厚5mm的高性能微型轴向磁通永磁电机，并通过电磁热分析与实验验证性能。**

- **链接: [https://arxiv.org/pdf/2509.23561v2](https://arxiv.org/pdf/2509.23561v2)**

> **作者:** Jianren Wang; Quanting Xie; Jie Han; Yang Zhang; Christopher G. Atkeson; Abhinav Gupta; Deepak Pathak; Yonatan Bisk
>
> **摘要:** Quasi-direct-drive (QDD) actuation is transforming legged and manipulator robots by eliminating high-ratio gearboxes, yet it demands motors that deliver very high torque at low speed within a thin, disc-shaped joint envelope. Axial-flux permanent-magnet (AFPM) machines meet these geometric and torque requirements, but scaling them below a 20mm outer diameter is hampered by poor copper fill in conventional wound stators, inflating resistance and throttling continuous torque. This paper introduces a micro-scale AFPM motor that overcomes these limitations through printed-circuit-board (PCB) windings fabricated with advanced IC-substrate high-density interconnect (HDI) technology. The resulting 48-layer stator-formed by stacking four 12-layer HDI modules-achieves a record 45\% copper fill in a package only 5mm thick and 19mm in diameter. We perform comprehensive electromagnetic and thermal analyses to inform the motor design, then fabricate a prototype whose performance characteristics are experimentally verified.
>
---
#### [replaced 007] Kinodynamic Motion Planning for Collaborative Object Transportation by Multiple Mobile Manipulators
- **分类: cs.RO; cs.MA; math.OC**

- **简介: 该论文研究多移动机械臂协同运载物体的运动规划问题，旨在动态环境中实现安全高效运动。提出全局路径规划与局部动力学约束下的在线轨迹优化方法，结合移动底盘与机械臂协同，利用凸锥法避免自碰撞，提升规划可行性与控制效率。**

- **链接: [https://arxiv.org/pdf/2409.14910v2](https://arxiv.org/pdf/2409.14910v2)**

> **作者:** Keshab Patra; Arpita Sinha; Anirban Guha
>
> **备注:** Video: https://youtu.be/LhE_HcK4g-s
>
> **摘要:** This work proposes a kinodynamic motion planning technique for collaborative object transportation by multiple mobile manipulators in dynamic environments. A global path planner computes a linear piecewise path from start to goal. A novel algorithm detects the narrow regions between the static obstacles and aids in defining the obstacle-free region to enhance the feasibility of the global path. We then formulate a local online motion planning technique for trajectory generation that minimizes the control efforts in a receding horizon manner. It plans the trajectory for finite time horizons, considering the kinodynamic constraints and the static and dynamic obstacles. The planning technique jointly plans for the mobile bases and the arms to utilize the locomotion capability of the mobile base and the manipulation capability of the arm efficiently. We use a convex cone approach to avoid self-collision of the formation by modifying the mobile manipulators admissible state without imposing additional constraints. Numerical simulations and hardware experiments showcase the efficiency of the proposed approach.
>
---
#### [replaced 008] Safety-Aware Reinforcement Learning for Control via Risk-Sensitive Action-Value Iteration and Quantile Regression
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究安全强化学习控制任务，旨在解决传统方法在高方差环境中因过估计导致的安全性差问题。作者提出一种基于CVaR风险正则化的分位数行动值迭代算法，在不增加网络复杂度下显式融入安全约束，并提供收敛性理论保证，实验验证了其在机器人避障中更优的安全与性能平衡。**

- **链接: [https://arxiv.org/pdf/2506.06954v2](https://arxiv.org/pdf/2506.06954v2)**

> **作者:** Clinton Enwerem; Aniruddh G. Puranic; John S. Baras; Calin Belta
>
> **备注:** 13 pages, 4 figures. Expanded version of a paper to appear in the Proceedings of the 2025 IEEE Conference on Decision and Control (CDC)
>
> **摘要:** Mainstream approximate action-value iteration reinforcement learning (RL) algorithms suffer from overestimation bias, leading to suboptimal policies in high-variance stochastic environments. Quantile-based action-value iteration methods reduce this bias by learning a distribution of the expected cost-to-go using quantile regression. However, ensuring that the learned policy satisfies safety constraints remains a challenge when these constraints are not explicitly integrated into the RL framework. Existing methods often require complex neural architectures or manual tradeoffs due to combined cost functions. To address this, we propose a risk-regularized quantile-based algorithm integrating Conditional Value-at-Risk (CVaR) to enforce safety without complex architectures. We also provide theoretical guarantees on the contraction properties of the risk-sensitive distributional Bellman operator in Wasserstein space, ensuring convergence to a unique cost distribution. Simulations of a mobile robot in a dynamic reach-avoid task show that our approach leads to more goal successes, fewer collisions, and better safety-performance trade-offs than risk-neutral methods.
>
---
#### [replaced 009] Collaborative Drill Alignment in Surgical Robotics
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究手术机器人中的协作钻孔对准任务，旨在提升钻孔精度。提出一种基于虚拟棱柱关节和非线性弹簧的虚拟夹具控制器，结合视觉反馈与校准方法，实现轴向约束与 surgeon 主动控制的结合，应用于犬类骨骼螺钉置入，验证了更高角度精度和更小误差。**

- **链接: [https://arxiv.org/pdf/2503.05791v2](https://arxiv.org/pdf/2503.05791v2)**

> **作者:** Daniel Larby; Joshua Kershaw; Matthew Allen; Fulvio Forni
>
> **备注:** 14 pages, 12 figures, accepted to IEEE Transactions on Control Systems Technology
>
> **摘要:** Robotic assistance allows surgeries to be reliably and accurately executed while still under direct supervision of the surgeon, combining the strengths of robotic technology with the surgeon's expertise. This paper describes a robotic system designed to assist in surgical procedures by implementing a virtual drill guide. The system integrates virtual-fixture functionality using a novel virtual-mechanism controller with additional visual feedback. The controller constrains the tool to the desired axis, while allowing axial motion to remain under the surgeon's control. Compared to prior virtual-fixture approaches -- which primarily perform pure energy-shaping and damping injection with linear springs and dampers -- our controller uses a virtual prismatic joint to which the robot is constrained by nonlinear springs, allowing us to easily shape the dynamics of the system. We detail the calibration procedures required to achieve sufficient precision, and describe the implementation of the controller. We apply this system to a veterinary procedure: drilling for transcondylar screw placement in dogs. The results of the trials on 3D-printed bone models demonstrate sufficient precision to perform the procedure and suggest improved angular accuracy and reduced exit translation errors compared to patient specific guides (PSG). Discussion and future improvements follow.
>
---
#### [replaced 010] BEDI: A Comprehensive Benchmark for Evaluating Embodied Agents on UAVs
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出BEDI，一个面向无人机具身智能体的标准化评测基准。针对现有评估缺乏统一标准的问题，构建了基于感知-决策-行动循环的动态任务链范式，设计六项核心子技能评估框架，并开发虚实结合的开放测试平台，推动具身智能模型客观比较与优化。**

- **链接: [https://arxiv.org/pdf/2505.18229v2](https://arxiv.org/pdf/2505.18229v2)**

> **作者:** Mingning Guo; Mengwei Wu; Jiarun He; Shaoxian Li; Haifeng Li; Chao Tao
>
> **摘要:** With the rapid advancement of low-altitude remote sensing and Vision-Language Models (VLMs), Embodied Agents based on Unmanned Aerial Vehicles (UAVs) have shown significant potential in autonomous tasks. However, current evaluation methods for UAV-Embodied Agents (UAV-EAs) remain constrained by the lack of standardized benchmarks, diverse testing scenarios and open system interfaces. To address these challenges, we propose BEDI (Benchmark for Embodied Drone Intelligence), a systematic and standardized benchmark designed for evaluating UAV-EAs. Specifically, we introduce a novel Dynamic Chain-of-Embodied-Task paradigm based on the perception-decision-action loop, which decomposes complex UAV tasks into standardized, measurable subtasks. Building on this paradigm, we design a unified evaluation framework encompassing six core sub-skills: semantic perception, spatial perception, motion control, tool utilization, task planning and action generation. Furthermore, we develop a hybrid testing platform that incorporates a wide range of both virtual and real-world scenarios, enabling a comprehensive evaluation of UAV-EAs across diverse contexts. The platform also offers open and standardized interfaces, allowing researchers to customize tasks and extend scenarios, thereby enhancing flexibility and scalability in the evaluation process. Finally, through empirical evaluations of several state-of-the-art (SOTA) VLMs, we reveal their limitations in embodied UAV tasks, underscoring the critical role of the BEDI benchmark in advancing embodied intelligence research and model optimization. By filling the gap in systematic and standardized evaluation within this field, BEDI facilitates objective model comparison and lays a robust foundation for future development in this field. Our benchmark is now publicly available at https://github.com/lostwolves/BEDI.
>
---
#### [replaced 011] An Open-Source Soft Robotic Platform for Autonomous Aerial Manipulation in the Wild
- **分类: cs.RO; eess.SY**

- **简介: 该论文聚焦自主空中操作任务，旨在解决依赖外部感知系统的局限性。作者提出一种全自主开源软体机器人平台，仅使用机载感知，在室内外复杂环境中实现稳定抓取操作，提升系统实用性与可扩展性，并公开软硬件设计以促进科研发展。**

- **链接: [https://arxiv.org/pdf/2409.07662v2](https://arxiv.org/pdf/2409.07662v2)**

> **作者:** Erik Bauer; Marc Blöchlinger; Pascal Strauch; Arman Raayatsanati; Curdin Cavelti; Robert K. Katzschmann
>
> **备注:** Project website: https://sites.google.com/view/open-source-soft-platform/open-source-soft-robotic-platform GitHub: https://github.com/raptor-ethz
>
> **摘要:** Aerial manipulation combines the versatility and speed of flying platforms with the functional capabilities of mobile manipulation, which presents significant challenges due to the need for precise localization and control. Traditionally, researchers have relied on offboard perception systems, which are limited to expensive and impractical specially equipped indoor environments. In this work, we introduce a novel platform for autonomous aerial manipulation that exclusively utilizes onboard perception systems. Our platform can perform aerial manipulation in various indoor and outdoor environments without depending on external perception systems. Our experimental results demonstrate the platform's ability to autonomously grasp various objects in diverse settings. This advancement significantly improves the scalability and practicality of aerial manipulation applications by eliminating the need for costly tracking solutions. To accelerate future research, we open source our ROS 2 software stack and custom hardware design, making our contributions accessible to the broader research community.
>
---
#### [replaced 012] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions
- **分类: cs.LG; cs.RO; eess.SY; math.OC**

- **简介: 该论文致力于为强化学习策略提供稳定性证明，属于控制理论与机器学习交叉任务。针对传统李雅普诺夫函数难以构造的问题，提出通过增强值函数并引入多步平均下降条件，学习广义李雅普诺夫函数，实现对非线性系统下RL策略的稳定性认证。**

- **链接: [https://arxiv.org/pdf/2505.10947v3](https://arxiv.org/pdf/2505.10947v3)**

> **作者:** Kehan Long; Jorge Cortés; Nikolay Atanasov
>
> **备注:** NeurIPS 2025
>
> **摘要:** Establishing stability certificates for closed-loop systems under reinforcement learning (RL) policies is essential to move beyond empirical performance and offer guarantees of system behavior. Classical Lyapunov methods require a strict stepwise decrease in the Lyapunov function but such certificates are difficult to construct for learned policies. The RL value function is a natural candidate but it is not well understood how it can be adapted for this purpose. To gain intuition, we first study the linear quadratic regulator (LQR) problem and make two key observations. First, a Lyapunov function can be obtained from the value function of an LQR policy by augmenting it with a residual term related to the system dynamics and stage cost. Second, the classical Lyapunov decrease requirement can be relaxed to a generalized Lyapunov condition requiring only decrease on average over multiple time steps. Using this intuition, we consider the nonlinear setting and formulate an approach to learn generalized Lyapunov functions by augmenting RL value functions with neural network residual terms. Our approach successfully certifies the stability of RL policies trained on Gymnasium and DeepMind Control benchmarks. We also extend our method to jointly train neural controllers and stability certificates using a multi-step Lyapunov loss, resulting in larger certified inner approximations of the region of attraction compared to the classical Lyapunov approach. Overall, our formulation enables stability certification for a broad class of systems with learned policies by making certificates easier to construct, thereby bridging classical control theory and modern learning-based methods.
>
---
#### [replaced 013] Enhanced Spatiotemporal Consistency for Image-to-LiDAR Data Pretraining
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文聚焦LiDAR表征学习，旨在减少对人工标注的依赖。针对现有方法忽略时间动态性的问题，提出SuperFlow++框架，通过融合时空一致性机制，在多数据集上实现更优的跨模态预训练与下游任务性能。**

- **链接: [https://arxiv.org/pdf/2503.19912v2](https://arxiv.org/pdf/2503.19912v2)**

> **作者:** Xiang Xu; Lingdong Kong; Hui Shuai; Wenwei Zhang; Liang Pan; Kai Chen; Ziwei Liu; Qingshan Liu
>
> **备注:** IEEE Transactions on Pattern Analysis and Machine Intelligence
>
> **摘要:** LiDAR representation learning has emerged as a promising approach to reducing reliance on costly and labor-intensive human annotations. While existing methods primarily focus on spatial alignment between LiDAR and camera sensors, they often overlook the temporal dynamics critical for capturing motion and scene continuity in driving scenarios. To address this limitation, we propose SuperFlow++, a novel framework that integrates spatiotemporal cues in both pretraining and downstream tasks using consecutive LiDAR-camera pairs. SuperFlow++ introduces four key components: (1) a view consistency alignment module to unify semantic information across camera views, (2) a dense-to-sparse consistency regularization mechanism to enhance feature robustness across varying point cloud densities, (3) a flow-based contrastive learning approach that models temporal relationships for improved scene understanding, and (4) a temporal voting strategy that propagates semantic information across LiDAR scans to improve prediction consistency. Extensive evaluations on 11 heterogeneous LiDAR datasets demonstrate that SuperFlow++ outperforms state-of-the-art methods across diverse tasks and driving conditions. Furthermore, by scaling both 2D and 3D backbones during pretraining, we uncover emergent properties that provide deeper insights into developing scalable 3D foundation models. With strong generalizability and computational efficiency, SuperFlow++ establishes a new benchmark for data-efficient LiDAR-based perception in autonomous driving. The code is publicly available at https://github.com/Xiangxu-0103/SuperFlow
>
---
#### [replaced 014] Dribble Master: Learning Agile Humanoid Dribbling Through Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文研究人形机器人足球运球任务，解决传统方法在动态平衡与实时控球上的不足。提出两阶段课程学习框架，结合虚拟相机模型与启发式奖励，在仿真中训练并成功迁移到实体机器人，实现灵活、稳定的视觉引导运球。**

- **链接: [https://arxiv.org/pdf/2505.12679v2](https://arxiv.org/pdf/2505.12679v2)**

> **作者:** Zhuoheng Wang; Jinyin Zhou; Qi Wu
>
> **摘要:** Humanoid soccer dribbling is a highly challenging task that demands dexterous ball manipulation while maintaining dynamic balance. Traditional rule-based methods often struggle to achieve accurate ball control due to their reliance on fixed walking patterns and limited adaptability to real-time ball dynamics. To address these challenges, we propose a two-stage curriculum learning framework that enables a humanoid robot to acquire dribbling skills without explicit dynamics or predefined trajectories. In the first stage, the robot learns basic locomotion skills; in the second stage, we fine-tune the policy for agile dribbling maneuvers. We further introduce a virtual camera model in simulation that simulates the field of view and perception constraints of the real robot, enabling realistic ball perception during training. We also design heuristic rewards to encourage active sensing, promoting a broader visual range for continuous ball perception. The policy is trained in simulation and successfully transferred to a physical humanoid robot. Experiment results demonstrate that our method enables effective ball manipulation, achieving flexible and visually appealing dribbling behaviors across multiple environments. This work highlights the potential of reinforcement learning in developing agile humanoid soccer robots. Additional details and videos are available at https://zhuoheng0910.github.io/dribble-master/.
>
---
#### [replaced 015] NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文针对图像生成中结构对齐难题，提出相位保持扩散（φ-PD），保留输入相位、随机化幅度，实现结构对齐生成。无需模型修改，兼容现有扩散模型，提升sim-to-real等任务性能。**

- **链接: [https://arxiv.org/pdf/2512.05106v2](https://arxiv.org/pdf/2512.05106v2)**

> **作者:** Yu Zeng; Charles Ochoa; Mingyuan Zhou; Vishal M. Patel; Vitor Guizilini; Rowan McAllister
>
> **摘要:** Standard diffusion corrupts data using Gaussian noise whose Fourier coefficients have random magnitudes and random phases. While effective for unconditional or text-to-image generation, corrupting phase components destroys spatial structure, making it ill-suited for tasks requiring geometric consistency, such as re-rendering, simulation enhancement, and image-to-image translation. We introduce Phase-Preserving Diffusion φ-PD, a model-agnostic reformulation of the diffusion process that preserves input phase while randomizing magnitude, enabling structure-aligned generation without architectural changes or additional parameters. We further propose Frequency-Selective Structured (FSS) noise, which provides continuous control over structural rigidity via a single frequency-cutoff parameter. φ-PD adds no inference-time cost and is compatible with any diffusion model for images or videos. Across photorealistic and stylized re-rendering, as well as sim-to-real enhancement for driving planners, φ-PD produces controllable, spatially aligned results. When applied to the CARLA simulator, φ-PD improves CARLA-to-Waymo planner performance by 50\%. The method is complementary to existing conditioning approaches and broadly applicable to image-to-image and video-to-video generation. Videos, additional examples, and code are available on our \href{https://yuzeng-at-tri.github.io/ppd-page/}{project page}.
>
---
#### [replaced 016] Much Ado About Noising: Dispelling the Myths of Generative Robotic Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究生成式机器人控制策略（GCPs）在行为克隆中的有效性。通过实验证明，GCP的成功主要源于迭代计算与适当随机性，而非多模态建模能力。提出轻量化的两步最小迭代策略（MIP），性能媲美现有模型。**

- **链接: [https://arxiv.org/pdf/2512.01809v2](https://arxiv.org/pdf/2512.01809v2)**

> **作者:** Chaoyi Pan; Giri Anantharaman; Nai-Chieh Huang; Claire Jin; Daniel Pfrommer; Chenyang Yuan; Frank Permenter; Guannan Qu; Nicholas Boffi; Guanya Shi; Max Simchowitz
>
> **摘要:** Generative models, like flows and diffusions, have recently emerged as popular and efficacious policy parameterizations in robotics. There has been much speculation as to the factors underlying their successes, ranging from capturing multi-modal action distribution to expressing more complex behaviors. In this work, we perform a comprehensive evaluation of popular generative control policies (GCPs) on common behavior cloning (BC) benchmarks. We find that GCPs do not owe their success to their ability to capture multi-modality or to express more complex observation-to-action mappings. Instead, we find that their advantage stems from iterative computation, as long as intermediate steps are supervised during training and this supervision is paired with a suitable level of stochasticity. As a validation of our findings, we show that a minimum iterative policy (MIP), a lightweight two-step regression-based policy, essentially matches the performance of flow GCPs, and often outperforms distilled shortcut models. Our results suggest that the distribution-fitting component of GCPs is less salient than commonly believed, and point toward new design spaces focusing solely on control performance. Project page: https://simchowitzlabpublic.github.io/much-ado-about-noising-project/
>
---
#### [replaced 017] Energy-Aware Lane Planning for Connected Electric Vehicles in Urban Traffic: Design and Vehicle-in-the-Loop Validation
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究城市交通中联网电动车的节能驾驶问题，提出一种联合优化纵向速度与横向换道决策的能效规划框架，利用车路通信和图模型预测能耗，通过车辆在环实验验证可降低能耗最高达24%。**

- **链接: [https://arxiv.org/pdf/2503.23228v4](https://arxiv.org/pdf/2503.23228v4)**

> **作者:** Hansung Kim; Eric Yongkeun Choi; Eunhyek Joa; Hotae Lee; Linda Lim; Scott Moura; Francesco Borrelli
>
> **备注:** Accepted at 2025 IEEE Conference on Decision and Control (CDC25')
>
> **摘要:** Urban driving with connected and automated vehicles (CAVs) offers potential for energy savings, yet most eco-driving strategies focus solely on longitudinal speed control within a single lane. This neglects the significant impact of lateral decisions, such as lane changes, on overall energy efficiency, especially in environments with traffic signals and heterogeneous traffic flow. To address this gap, we propose a novel energy-aware motion planning framework that jointly optimizes longitudinal speed and lateral lane-change decisions using vehicle-to-infrastructure (V2I) communication. Our approach estimates long-term energy costs using a graph-based approximation and solves short-horizon optimal control problems under traffic constraints. Using a data-driven energy model calibrated to an actual battery electric vehicle, we demonstrate with vehicle-in-the-loop experiments that our method reduces motion energy consumption by up to 24 percent compared to a human driver, highlighting the potential of connectivity-enabled planning for sustainable urban autonomy.
>
---
#### [replaced 018] First Responders' Perceptions of Semantic Information for Situational Awareness in Robot-Assisted Emergency Response
- **分类: cs.RO**

- **简介: 该论文探讨急救人员对语义信息在机器人应急响应中提升态势感知的态度。通过跨国问卷调查22名急救人员，分析其对机器人及语义信息的看法，揭示其对语义信息的需求与信任阈值，并指出实验室技术与实际应用间的差距。**

- **链接: [https://arxiv.org/pdf/2510.16692v2](https://arxiv.org/pdf/2510.16692v2)**

> **作者:** Tianshu Ruan; Zoe Betta; Georgios Tzoumas; Rustam Stolkin; Manolis Chiou
>
> **摘要:** This study investigates First Responders' (FRs) attitudes toward the use of semantic information and Situational Awareness (SA) in robotic systems during emergency operations. A structured questionnaire was administered to 22 FRs across eight countries, capturing their demographic profiles, general attitudes toward robots, and experiences with semantics-enhanced SA. Results show that most FRs expressed positive attitudes toward robots, and rated the usefulness of semantic information for building SA at an average of 3.6 out of 5. Semantic information was also valued for its role in predicting unforeseen emergencies (mean 3.9). Participants reported requiring an average of 74.6\% accuracy to trust semantic outputs and 67.8\% for them to be considered useful, revealing a willingness to use imperfect but informative AI support tools. To the best of our knowledge, this study offers novel insights by being one of the first to directly survey FRs on semantic-based SA in a cross-national context. It reveals the types of semantic information most valued in the field, such as object identity, spatial relationships, and risk context-and connects these preferences to the respondents' roles, experience, and education levels. The findings also expose a critical gap between lab-based robotics capabilities and the realities of field deployment, highlighting the need for more meaningful collaboration between FRs and robotics researchers. These insights contribute to the development of more user-aligned and situationally aware robotic systems for emergency response.
>
---
#### [replaced 019] RealD$^2$iff: Bridging Real-World Gap in Robot Manipulation via Depth Diffusion
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中的视觉sim2real差距问题，提出RealD²iff方法，通过深度扩散模型将干净的仿真深度转化为含噪的真实感深度。其核心是构建分层扩散框架，结合频域引导监督和差异引导优化，实现无需真实数据采集的零样本sim2real迁移。**

- **链接: [https://arxiv.org/pdf/2511.22505v2](https://arxiv.org/pdf/2511.22505v2)**

> **作者:** Xiujian Liang; Jiacheng Liu; Mingyang Sun; Qichen He; Cewu Lu; Jianhua Sun
>
> **备注:** We are the author team of the paper "RealD$^2$iff: Bridging Real-World Gap in Robot Manipulation via Depth Diffusion". After self-examination, our team discovered inappropriate wording in the citation of related work, the introduction, and the contribution statement, which may affect the contribution of other related works. Therefore, we have decided to revise the paper and request its withdrawal
>
> **摘要:** Robot manipulation in the real world is fundamentally constrained by the visual sim2real gap, where depth observations collected in simulation fail to reflect the complex noise patterns inherent to real sensors. In this work, inspired by the denoising capability of diffusion models, we invert the conventional perspective and propose a clean-to-noisy paradigm that learns to synthesize noisy depth, thereby bridging the visual sim2real gap through purely simulation-driven robotic learning. Building on this idea, we introduce RealD$^2$iff, a hierarchical coarse-to-fine diffusion framework that decomposes depth noise into global structural distortions and fine-grained local perturbations. To enable progressive learning of these components, we further develop two complementary strategies: Frequency-Guided Supervision (FGS) for global structure modeling and Discrepancy-Guided Optimization (DGO) for localized refinement. To integrate RealD$^2$iff seamlessly into imitation learning, we construct a pipeline that spans six stages. We provide comprehensive empirical and experimental validation demonstrating the effectiveness of this paradigm. RealD$^2$iff enables two key applications: (1) generating real-world-like depth to construct clean-noisy paired datasets without manual sensor data collection. (2) Achieving zero-shot sim2real robot manipulation, substantially improving real-world performance without additional fine-tuning.
>
---
#### [replaced 020] PosA-VLA: Enhancing Action Generation via Pose-Conditioned Anchor Attention
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究视觉-语言-动作（VLA）模型在具身任务中的动作生成问题，旨在解决因注意力分散导致的动作冗余与不精确。作者提出PosA-VLA框架，通过姿态条件锚定注意力机制，提升动作的精准性与效率，无需额外感知模块，具有轻量与强泛化优势。**

- **链接: [https://arxiv.org/pdf/2512.03724v2](https://arxiv.org/pdf/2512.03724v2)**

> **作者:** Ziwen Li; Xin Wang; Hanlue Zhang; Runnan Chen; Runqi Lin; Xiao He; Han Huang; Yandong Guo; Fakhri Karray; Tongliang Liu; Mingming Gong
>
> **摘要:** The Vision-Language-Action (VLA) models have demonstrated remarkable performance on embodied tasks and shown promising potential for real-world applications. However, current VLAs still struggle to produce consistent and precise target-oriented actions, as they often generate redundant or unstable motions along trajectories, limiting their applicability in time-sensitive scenarios.In this work, we attribute these redundant actions to the spatially uniform perception field of existing VLAs, which causes them to be distracted by target-irrelevant objects, especially in complex environments.To address this issue, we propose an efficient PosA-VLA framework that anchors visual attention via pose-conditioned supervision, consistently guiding the model's perception toward task-relevant regions. The pose-conditioned anchor attention mechanism enables the model to better align instruction semantics with actionable visual cues, thereby improving action generation precision and efficiency. Moreover, our framework adopts a lightweight architecture and requires no auxiliary perception modules (e.g., segmentation or grounding networks), ensuring efficient inference. Extensive experiments verify that our method executes embodied tasks with precise and time-efficient behavior across diverse robotic manipulation benchmarks and shows robust generalization in a variety of challenging environments.
>
---
#### [replaced 021] Grasping a Handful: Sequential Multi-Object Dexterous Grasp Generation
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究多物体灵巧抓取任务，解决传统方法抓取成功率低、生成速度慢的问题。提出SeqGrasp算法生成序列化抓取数据，并构建SeqDataset数据集，训练出快速且高成功率的扩散模型SeqDiffuser。**

- **链接: [https://arxiv.org/pdf/2503.22370v5](https://arxiv.org/pdf/2503.22370v5)**

> **作者:** Haofei Lu; Yifei Dong; Zehang Weng; Florian T. Pokorny; Jens Lundell; Danica Kragic
>
> **备注:** We replace the sets in Section II with an odered sequences
>
> **摘要:** We introduce the sequential multi-object robotic grasp sampling algorithm SeqGrasp that can robustly synthesize stable grasps on diverse objects using the robotic hand's partial Degrees of Freedom (DoF). We use SeqGrasp to construct the large-scale Allegro Hand sequential grasping dataset SeqDataset and use it for training the diffusion-based sequential grasp generator SeqDiffuser. We experimentally evaluate SeqGrasp and SeqDiffuser against the state-of-the-art non-sequential multi-object grasp generation method MultiGrasp in simulation and on a real robot. The experimental results demonstrate that SeqGrasp and SeqDiffuser reach an 8.71%-43.33% higher grasp success rate than MultiGrasp. Furthermore, SeqDiffuser is approximately 1000 times faster at generating grasps than SeqGrasp and MultiGrasp. Project page: https://yulihn.github.io/SeqGrasp/.
>
---
#### [replaced 022] Optimal Virtual Model Control for Robotics: Design and Tuning of Passivity-Based Controllers
- **分类: cs.RO**

- **简介: 该论文研究机器人中基于无源性的控制器设计与优化。针对其调参难题，提出一种基于“虚拟机构”的直观设计方法，并利用算法微分优化控制参数，在保证稳定性的同时提升性能。**

- **链接: [https://arxiv.org/pdf/2411.06627v3](https://arxiv.org/pdf/2411.06627v3)**

> **作者:** Daniel Larby; Fulvio Forni
>
> **备注:** 16 pages, 19 figures
>
> **摘要:** Passivity-based control is a cornerstone of control theory and an established design approach in robotics. Its strength is based on the passivity theorem, which provides a powerful interconnection framework for robotics. However, the design of passivity-based controllers and their optimal tuning remain challenging. We propose here an intuitive design approach for fully actuated robots, where the control action is determined by a `virtual-mechanism' as in classical virtual model control. The result is a robot whose controlled behavior can be understood in terms of physics. We achieve optimal tuning by applying algorithmic differentiation to ODE simulations of the rigid body dynamics. Overall, this leads to a flexible design and optimization approach: stability is proven by passivity of the virtual mechanism, while performance is obtained by optimization using algorithmic differentiation.
>
---
#### [replaced 023] MM-ACT: Learn from Multimodal Parallel Generation to Act
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出MM-ACT，一种统一的视觉-语言-动作模型，旨在提升机器人策略的语义理解与环境交互能力。通过多模态并行生成与共享上下文学习，实现跨模态协同，增强动作生成效率与泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.00975v2](https://arxiv.org/pdf/2512.00975v2)**

> **作者:** Haotian Liang; Xinyi Chen; Bin Wang; Mingkang Chen; Yitian Liu; Yuhao Zhang; Zanxin Chen; Tianshuo Yang; Yilun Chen; Jiangmiao Pang; Dong Liu; Xiaokang Yang; Yao Mu; Wenqi Shao; Ping Luo
>
> **备注:** 17 pages
>
> **摘要:** A generalist robotic policy needs both semantic understanding for task planning and the ability to interact with the environment through predictive capabilities. To tackle this, we present MM-ACT, a unified Vision-Language-Action (VLA) model that integrates text, image, and action in shared token space and performs generation across all three modalities. MM-ACT adopts a re-mask parallel decoding strategy for text and image generation, and employs a one-step parallel decoding strategy for action generation to improve efficiency. We introduce Context-Shared Multimodal Learning, a unified training paradigm that supervises generation in all three modalities from a shared context, enhancing action generation through cross-modal learning. Experiments were conducted on the LIBERO simulation and Franka real-robot setups as well as RoboTwin2.0 to assess in-domain and out-of-domain performances respectively. Our approach achieves a success rate of 96.3% on LIBERO, 72.0% across three tasks of real Franka, and 52.38% across eight bimanual tasks of RoboTwin2.0 with an additional gain of 9.25% from cross-modal learning. We release our codes, models and data at https://github.com/HHYHRHY/MM-ACT.
>
---
#### [replaced 024] Variational Shape Inference for Grasp Diffusion on SE(3)
- **分类: cs.RO**

- **简介: 该论文研究机器人抓取中的多模态抓取合成任务，旨在解决物体形状噪声和点云稀疏下的抓取生成问题。作者提出变分形状推断框架，结合隐式神经表示与SE(3)扩散模型，并引入测试时优化，提升鲁棒性与实际抓取成功率。**

- **链接: [https://arxiv.org/pdf/2508.17482v2](https://arxiv.org/pdf/2508.17482v2)**

> **作者:** S. Talha Bukhari; Kaivalya Agrawal; Zachary Kingston; Aniket Bera
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Grasp synthesis is a fundamental task in robotic manipulation which usually has multiple feasible solutions. Multimodal grasp synthesis seeks to generate diverse sets of stable grasps conditioned on object geometry, making the robust learning of geometric features crucial for success. To address this challenge, we propose a framework for learning multimodal grasp distributions that leverages variational shape inference to enhance robustness against shape noise and measurement sparsity. Our approach first trains a variational autoencoder for shape inference using implicit neural representations, and then uses these learned geometric features to guide a diffusion model for grasp synthesis on the SE(3) manifold. Additionally, we introduce a test-time grasp optimization technique that can be integrated as a plugin to further enhance grasping performance. Experimental results demonstrate that our shape inference for grasp synthesis formulation outperforms state-of-the-art multimodal grasp synthesis methods on the ACRONYM dataset by 6.3%, while demonstrating robustness to deterioration in point cloud density compared to other approaches. Furthermore, our trained model achieves zero-shot transfer to real-world manipulation of household objects, generating 34% more successful grasps than baselines despite measurement noise and point cloud calibration errors.
>
---
#### [replaced 025] Building Gradient by Gradient: Decentralised Energy Functions for Bimanual Robot Assembly
- **分类: cs.RO**

- **简介: 该论文研究双臂机器人装配任务，旨在解决高精度装配中因摩擦、形变等难以建模的动态因素导致的传统规划方法重规划慢、灵活性差的问题。提出一种去中心化的基于梯度的能量函数框架，通过自适应势函数组合实现快速重规划与自主重试、协调运动和手间传递。**

- **链接: [https://arxiv.org/pdf/2510.04696v2](https://arxiv.org/pdf/2510.04696v2)**

> **作者:** Alexander L. Mitchell; Joe Watson; Ingmar Posner
>
> **备注:** 8 pages, 7 figures, 1 table
>
> **摘要:** There are many challenges in bimanual assembly, including high-level sequencing, multi-robot coordination, and low-level, contact-rich operations such as component mating. Task and motion planning (TAMP) methods, while effective in this domain, may be prohibitively slow to converge when adapting to disturbances that require new task sequencing and optimisation. These events are common during tight-tolerance assembly, where difficult-to-model dynamics such as friction or deformation require rapid replanning and reattempts. Moreover, defining explicit task sequences for assembly can be cumbersome, limiting flexibility when task replanning is required. To simplify this planning, we introduce a decentralised gradient-based framework that uses a piecewise continuous energy function through the automatic composition of adaptive potential functions. This approach generates sub-goals using only myopic optimisation, rather than long-horizon planning. It demonstrates effectiveness at solving long-horizon tasks due to the structure and adaptivity of the energy function. We show that our approach scales to physical bimanual assembly tasks for constructing tight-tolerance assemblies. In these experiments, we discover that our gradient-based rapid replanning framework generates automatic retries, coordinated motions and autonomous handovers in an emergent fashion.
>
---
#### [replaced 026] Exploring Adversarial Obstacle Attacks in Search-based Path Planning for Autonomous Mobile Robots
- **分类: cs.RO**

- **简介: 该论文研究搜索式路径规划中对抗性障碍攻击的安全问题，旨在评估A*算法在恶意障碍干扰下的鲁棒性。作者通过仿真与实物实验，分析攻击对机器人路径延迟的影响，揭示环境约束显著影响算法抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2504.06154v2](https://arxiv.org/pdf/2504.06154v2)**

> **作者:** Adrian Szvoren; Jianwei Liu; Dimitrios Kanoulas; Nilufer Tuptuk
>
> **摘要:** Path planning algorithms, such as the search-based A*, are a critical component of autonomous mobile robotics, enabling robots to navigate from a starting point to a destination efficiently and safely. We investigated the resilience of the A* algorithm in the face of potential adversarial interventions known as obstacle attacks. The adversary's goal is to delay the robot's timely arrival at its destination by introducing obstacles along its original path. We developed malicious software to execute the attacks and conducted experiments to assess their impact, both in simulation using TurtleBot in Gazebo and in real-world deployment with the Unitree Go1 robot. In simulation, the attacks resulted in an average delay of 36\%, with the most significant delays occurring in scenarios where the robot was forced to take substantially longer alternative paths. In real-world experiments, the delays were even more pronounced, with all attacks successfully rerouting the robot and causing measurable disruptions. These results highlight that the algorithm's robustness is not solely an attribute of its design but is significantly influenced by the operational environment. For example, in constrained environments like tunnels, the delays were maximized due to the limited availability of alternative routes.
>
---
#### [replaced 027] Safe MPC Alignment with Human Directional Feedback
- **分类: cs.RO**

- **简介: 该论文研究安全关键场景下的机器人控制，旨在从人类方向性反馈中学习安全约束。提出一种可验证的MPC对齐方法，仅需反馈方向即可高效更新假设空间，并保证学习上限或检测假设不匹配，实验证明其能用少量反馈有效学习安全策略。**

- **链接: [https://arxiv.org/pdf/2407.04216v3](https://arxiv.org/pdf/2407.04216v3)**

> **作者:** Zhixian Xie; Wenlong Zhang; Yi Ren; Zhaoran Wang; George J. Pappas; Wanxin Jin
>
> **备注:** 20 pages, pre-print, submitted to TRO
>
> **摘要:** In safety-critical robot planning or control, manually specifying safety constraints or learning them from demonstrations can be challenging. In this article, we propose a certifiable alignment method for a robot to learn a safety constraint in its model predictive control (MPC) policy from human online directional feedback. To our knowledge, it is the first method to learn safety constraints from human feedback. The proposed method is based on an empirical observation: human directional feedback, when available, tends to guide the robot toward safer regions. The method only requires the direction of human feedback to update the learning hypothesis space. It is certifiable, providing an upper bound on the total number of human feedback in the case of successful learning, or declaring the hypothesis misspecification, i.e., the true safety constraint cannot be found within the specified hypothesis space. We evaluated the proposed method in numerical examples and user studies with two simulation games. Additionally, we tested the proposed method on a real-world Franka robot arm performing mobile water-pouring tasks. The results demonstrate the efficacy and efficiency of our method, showing that it enables a robot to successfully learn safety constraints with a small handful (tens) of human directional corrections.
>
---
#### [replaced 028] UMI-on-Air: Embodiment-Aware Guidance for Embodiment-Agnostic Visuomotor Policies
- **分类: cs.RO**

- **简介: 该论文研究机器人操纵策略的跨形态迁移，解决通用视觉运动策略在特定机器人（如空中机械臂）上部署时因动力学不匹配导致的性能下降问题。提出UMI-on-Air框架，通过结合高阶策略与低阶控制器的梯度反馈，在推理时实现动态可行的轨迹生成，提升执行成功率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.02614v2](https://arxiv.org/pdf/2510.02614v2)**

> **作者:** Harsh Gupta; Xiaofeng Guo; Huy Ha; Chuer Pan; Muqing Cao; Dongjae Lee; Sebastian Scherer; Shuran Song; Guanya Shi
>
> **备注:** Result videos can be found at umi-on-air.github.io
>
> **摘要:** We introduce UMI-on-Air, a framework for embodiment-aware deployment of embodiment-agnostic manipulation policies. Our approach leverages diverse, unconstrained human demonstrations collected with a handheld gripper (UMI) to train generalizable visuomotor policies. A central challenge in transferring these policies to constrained robotic embodiments-such as aerial manipulators-is the mismatch in control and robot dynamics, which often leads to out-of-distribution behaviors and poor execution. To address this, we propose Embodiment-Aware Diffusion Policy (EADP), which couples a high-level UMI policy with a low-level embodiment-specific controller at inference time. By integrating gradient feedback from the controller's tracking cost into the diffusion sampling process, our method steers trajectory generation towards dynamically feasible modes tailored to the deployment embodiment. This enables plug-and-play, embodiment-aware trajectory adaptation at test time. We validate our approach on multiple long-horizon and high-precision aerial manipulation tasks, showing improved success rates, efficiency, and robustness under disturbances compared to unguided diffusion baselines. Finally, we demonstrate deployment in previously unseen environments, using UMI demonstrations collected in the wild, highlighting a practical pathway for scaling generalizable manipulation skills across diverse-and even highly constrained-embodiments. All code, data, and checkpoints will be publicly released after acceptance. Result videos can be found at umi-on-air.github.io.
>
---
#### [replaced 029] Griffin: Aerial-Ground Cooperative Detection and Tracking Dataset and Benchmark
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦空中-地面协同感知任务，旨在解决缺乏高质量公开数据集的问题。作者构建了Griffin数据集与基准，包含多场景3D标注及通信、定位等真实干扰评估协议，推动无人机-车辆协同检测与跟踪研究。**

- **链接: [https://arxiv.org/pdf/2503.06983v2](https://arxiv.org/pdf/2503.06983v2)**

> **作者:** Jiahao Wang; Xiangyu Cao; Jiaru Zhong; Yuner Zhang; Zeyu Han; Haibao Yu; Chuang Zhang; Lei He; Shaobing Xu; Jianqiang Wang
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** While cooperative perception can overcome the limitations of single-vehicle systems, the practical implementation of vehicle-to-vehicle and vehicle-to-infrastructure systems is often impeded by significant economic barriers. Aerial-ground cooperation (AGC), which pairs ground vehicles with drones, presents a more economically viable and rapidly deployable alternative. However, this emerging field has been held back by a critical lack of high-quality public datasets and benchmarks. To bridge this gap, we present \textit{Griffin}, a comprehensive AGC 3D perception dataset, featuring over 250 dynamic scenes (37k+ frames). It incorporates varied drone altitudes (20-60m), diverse weather conditions, realistic drone dynamics via CARLA-AirSim co-simulation, and critical occlusion-aware 3D annotations. Accompanying the dataset is a unified benchmarking framework for cooperative detection and tracking, with protocols to evaluate communication efficiency, altitude adaptability, and robustness to communication latency, data loss and localization noise. By experiments through different cooperative paradigms, we demonstrate the effectiveness and limitations of current methods and provide crucial insights for future research. The dataset and codes are available at https://github.com/wang-jh18-SVM/Griffin.
>
---
#### [replaced 030] Primal-Dual iLQR for GPU-Accelerated Learning and Control in Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决腿式机器人模型预测控制（MPC）计算效率低的问题。作者提出一种基于GPU加速的原对偶iLQR算法，实现时间和状态空间并行化，显著提升计算速度，并支持多机器人与强化学习集成。**

- **链接: [https://arxiv.org/pdf/2506.07823v3](https://arxiv.org/pdf/2506.07823v3)**

> **作者:** Lorenzo Amatucci; João Sousa-Pinto; Giulio Turrisi; Dominique Orban; Victor Barasuol; Claudio Semini
>
> **摘要:** This paper introduces a novel Model Predictive Control (MPC) implementation for legged robot locomotion that leverages GPU parallelization. Our approach enables both temporal and state-space parallelization by incorporating a parallel associative scan to solve the primal-dual Karush-Kuhn-Tucker (KKT) system. In this way, the optimal control problem is solved in $\mathcal{O}(\log^2(n)\log{N} + \log^2(m))$ complexity, instead of $\mathcal{O}(N(n + m)^3)$, where $n$, $m$, and $N$ are the dimension of the system state, control vector, and the length of the prediction horizon. We demonstrate the advantages of this implementation over two state-of-the-art solvers (acados and crocoddyl), achieving up to a 60\% improvement in runtime for Whole Body Dynamics (WB)-MPC and a 700\% improvement for Single Rigid Body Dynamics (SRBD)-MPC when varying the prediction horizon length. The presented formulation scales efficiently with the problem state dimensions as well, enabling the definition of a centralized controller for up to 16 legged robots that can be computed in less than 25 ms. Furthermore, thanks to the JAX implementation, the solver supports large-scale parallelization across multiple environments, allowing the possibility of performing learning with the MPC in the loop directly in GPU. The code associated with this work can be found at https://github.com/iit-DLSLab/mpx.
>
---
#### [replaced 031] SwarmDiffusion: End-To-End Traversability-Guided Diffusion for Embodiment-Agnostic Navigation of Heterogeneous Robots
- **分类: cs.RO**

- **简介: 该论文研究视觉导航任务，旨在解决现有方法依赖手工提示、泛化性差及需外部规划轨迹的问题。提出SwarmDiffusion模型，端到端联合预测可通行性并生成轨迹，无需标注路径或提示工程，支持多机器人平台，实现高效、跨实体的导航。**

- **链接: [https://arxiv.org/pdf/2512.02851v3](https://arxiv.org/pdf/2512.02851v3)**

> **作者:** Iana Zhura; Sausar Karaf; Faryal Batool; Nipun Dhananjaya Weerakkodi Mudalige; Valerii Serpiva; Ali Alridha Abdulkarim; Aleksey Fedoseev; Didar Seyidov; Hajira Amjad; Dzmitry Tsetserukou
>
> **备注:** This work has been submitted for publication and is currently under review
>
> **摘要:** Visual traversability estimation is critical for autonomous navigation, but existing VLM-based methods rely on hand-crafted prompts, generalize poorly across embodiments, and output only traversability maps, leaving trajectory generation to slow external planners. We propose SwarmDiffusion, a lightweight end-to-end diffusion model that jointly predicts traversability and generates a feasible trajectory from a single RGB image. To remove the need for annotated or planner-produced paths, we introduce a planner-free trajectory construction pipeline based on randomized waypoint sampling, Bezier smoothing, and regularization enforcing connectivity, safety, directionality, and path thinness. This enables learning stable motion priors without demonstrations. SwarmDiffusion leverages VLM-derived supervision without prompt engineering and conditions the diffusion process on a compact embodiment state, producing physically consistent, traversable paths that transfer across different robot platforms. Across indoor environments and two embodiments (quadruped and aerial), the method achieves 80-100% navigation success and 0.09s inference, and adapts to a new robot using only-500 additional visual samples. It generalizes reliably to unseen environments in simulation and real-world trials, offering a scalable, prompt-free approach to unified traversability reasoning and trajectory generation.
>
---
#### [replaced 032] Inversely Learning Transferable Rewards via Abstracted States
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究逆强化学习，旨在从不同任务的行为轨迹中抽象出可迁移的奖励函数。其目标是让机器人在新任务中利用已学的内在偏好，无需重新编程即可适应新环境。方法通过抽象状态学习通用奖励，并在未见任务中验证其有效性。**

- **链接: [https://arxiv.org/pdf/2501.01669v3](https://arxiv.org/pdf/2501.01669v3)**

> **作者:** Yikang Gui; Prashant Doshi
>
> **摘要:** Inverse reinforcement learning (IRL) has progressed significantly toward accurately learning the underlying rewards in both discrete and continuous domains from behavior data. The next advance is to learn {\em intrinsic} preferences in ways that produce useful behavior in settings or tasks which are different but aligned with the observed ones. In the context of robotic applications, this helps integrate robots into processing lines involving new tasks (with shared intrinsic preferences) without programming from scratch. We introduce a method to inversely learn an abstract reward function from behavior trajectories in two or more differing instances of a domain. The abstract reward function is then used to learn task behavior in another separate instance of the domain. This step offers evidence of its transferability and validates its correctness. We evaluate the method on trajectories in tasks from multiple domains in OpenAI's Gym testbed and AssistiveGym and show that the learned abstract reward functions can successfully learn task behaviors in instances of the respective domains, which have not been seen previously.
>
---
#### [replaced 033] FASTer: Toward Efficient Autoregressive Vision Language Action Modeling via Neural Action Tokenization
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究机器人操作中的视觉-语言-动作（VLA）模型，旨在解决动作标记化中重建精度与推理效率的权衡问题。作者提出FASTer框架，包括高效动作编码的FASTerVQ和快速解码的FASTerVLA，兼顾高压缩率、高质量重建与快速推理，在多任务和多形态机器人上表现优越。**

- **链接: [https://arxiv.org/pdf/2512.04952v2](https://arxiv.org/pdf/2512.04952v2)**

> **作者:** Yicheng Liu; Shiduo Zhang; Zibin Dong; Baijun Ye; Tianyuan Yuan; Xiaopeng Yu; Linqi Yin; Chenhao Lu; Junhao Shi; Luca Jiang-Tao Yu; Liangtao Zheng; Tao Jiang; Jingjing Gong; Xipeng Qiu; Hang Zhao
>
> **摘要:** Autoregressive vision-language-action (VLA) models have recently demonstrated strong capabilities in robotic manipulation. However, their core process of action tokenization often involves a trade-off between reconstruction fidelity and inference efficiency. We introduce FASTer, a unified framework for efficient and generalizable robot learning that integrates a learnable tokenizer with an autoregressive policy built upon it. FASTerVQ encodes action chunks as single-channel images, capturing global spatio-temporal dependencies while maintaining a high compression ratio. FASTerVLA builds on this tokenizer with block-wise autoregressive decoding and a lightweight action expert, achieving both faster inference and higher task performance. Extensive experiments across simulated and real-world benchmarks show that FASTerVQ delivers superior reconstruction quality, high token utilization, and strong cross-task and cross-embodiment generalization, while FASTerVLA further improves overall capability, surpassing previous state-of-the-art VLA models in both inference speed and task performance.
>
---
#### [replaced 034] Dejavu: Towards Experience Feedback Learning for Embodied Intelligence
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对具身智能体部署后难以持续学习的问题，提出Dejavu框架。通过构建经验反馈网络（EFN），结合视觉-语言-动作策略与历史执行记忆，实现任务中持续学习，提升适应性与成功率。**

- **链接: [https://arxiv.org/pdf/2510.10181v2](https://arxiv.org/pdf/2510.10181v2)**

> **作者:** Shaokai Wu; Yanbiao Ji; Qiuchang Li; Zhiyi Zhang; Qichen He; Wenyuan Xie; Guodong Zhang; Bayram Bayramli; Yue Ding; Hongtao Lu
>
> **摘要:** Embodied agents face a fundamental limitation: once deployed in real-world environments to perform specific tasks, they are unable to acquire additional knowledge to enhance task performance. In this paper, we propose a general post-deployment learning framework Dejavu, which employs an Experience Feedback Network (EFN) and augments the frozen Vision-Language-Action (VLA) policy with retrieved execution memories. EFN identifies contextually prior action experiences and conditions action prediction on this retrieved guidance. We adopt reinforcement learning with semantic similarity rewards to train EFN, ensuring that the predicted actions align with past behaviors under current observations. During deployment, EFN continually enriches its memory with new trajectories, enabling the agent to exhibit "learning from experience". Experiments across diverse embodied tasks show that EFN improves adaptability, robustness, and success rates over frozen baselines. We provide code and demo in our supplementary material.
>
---
#### [replaced 035] Quantization-Free Autoregressive Action Transformer
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决离散化动作表示破坏连续动作空间结构的问题。作者提出一种无量化、基于连续动作的自回归Transformer方法，简化流程并提升性能。**

- **链接: [https://arxiv.org/pdf/2503.14259v3](https://arxiv.org/pdf/2503.14259v3)**

> **作者:** Ziyad Sheebaelhamd; Michael Tschannen; Michael Muehlebach; Claire Vernade
>
> **摘要:** Current transformer-based imitation learning approaches introduce discrete action representations and train an autoregressive transformer decoder on the resulting latent code. However, the initial quantization breaks the continuous structure of the action space thereby limiting the capabilities of the generative model. We propose a quantization-free method instead that leverages Generative Infinite-Vocabulary Transformers (GIVT) as a direct, continuous policy parametrization for autoregressive transformers. This simplifies the imitation learning pipeline while achieving state-of-the-art performance on a variety of popular simulated robotics tasks. We enhance our policy roll-outs by carefully studying sampling algorithms, further improving the results.
>
---
#### [replaced 036] Switch-JustDance: Benchmarking Whole Body Motion Tracking Policies Using a Commercial Console Game
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Switch-JustDance，利用《舞力全开》游戏构建低成本、可复现的机器人全身控制评测基准。旨在解决现有评估缺乏标准化、硬件因素和人机对比的问题，通过游戏得分量化控制器性能，并验证其可靠性与有效性。**

- **链接: [https://arxiv.org/pdf/2511.17925v2](https://arxiv.org/pdf/2511.17925v2)**

> **作者:** Jeonghwan Kim; Wontaek Kim; Yidan Lu; Jin Cheng; Fatemeh Zargarbashi; Zicheng Zeng; Zekun Qi; Zhiyang Dou; Nitish Sontakke; Donghoon Baek; Sehoon Ha; Tianyu Li
>
> **摘要:** Recent advances in whole-body robot control have enabled humanoid and legged robots to perform increasingly agile and coordinated motions. However, standardized benchmarks for evaluating these capabilities in real-world settings, and in direct comparison to humans, remain scarce. Existing evaluations often rely on pre-collected human motion datasets or simulation-based experiments, which limit reproducibility, overlook hardware factors, and hinder fair human-robot comparisons. We present Switch-JustDance, a low-cost and reproducible benchmarking pipeline that leverages motion-sensing console games, Just Dance on the Nintendo Switch, to evaluate robot whole-body control. Using Just Dance on the Nintendo Switch as a representative platform, Switch-JustDance converts in-game choreography into robot-executable motions through streaming, motion reconstruction, and motion retargeting modules and enables users to evaluate controller performance through the game's built-in scoring system. We first validate the evaluation properties of Just Dance, analyzing its reliability, validity, sensitivity, and potential sources of bias. Our results show that the platform provides consistent and interpretable performance measures, making it a suitable tool for benchmarking embodied AI. Building on this foundation, we benchmark three state-of-the-art humanoid whole-body controllers on hardware and provide insights into their relative strengths and limitations.
>
---
#### [replaced 037] Concerns and Values in Human-Robot Interactions: A Focus on Social Robotics
- **分类: cs.RO; cs.HC**

- **简介: 该论文聚焦社会机器人中的人机交互问题，旨在识别医疗、教育和家庭场景中的伦理关切与核心价值。通过文献综述与专家研讨构建HRI价值罗盘工具，并开展初步评估，辅助研究人员在设计中融入人类价值观。**

- **链接: [https://arxiv.org/pdf/2501.05628v2](https://arxiv.org/pdf/2501.05628v2)**

> **作者:** Giulio Antonio Abbo; Tony Belpaeme; Micol Spitale
>
> **备注:** 31 pages, 7 figures, 6 tables; 4 appendices
>
> **摘要:** Robots, as AI with physical instantiation, inhabit our social and physical world, where their actions have both social and physical consequences, posing challenges for researchers when designing social robots. This study starts with a scoping review to identify discussions and potential concerns arising from interactions with robotic systems in the context of healthcare, education, and private homes. Two focus groups of technology ethics experts then validated a comprehensive list of key topics and values in human-robot interaction (HRI) literature in these contexts. These insights were integrated into the HRI Value Compass web tool, to help HRI researchers identify these values in robot design. The tool was evaluated in a pilot study. This work benefits the HRI community by highlighting key concerns in human-robot interactions and providing an instrument to help researchers design robots that align with human values, ensuring future robotic systems adhere to these values in social applications.
>
---
#### [replaced 038] Incremental Generalized Hybrid A*
- **分类: cs.RO**

- **简介: 该论文研究实时运动规划任务，解决传统Hybrid A*因固定网格导致的效率与安全性矛盾问题。提出Incremental Generalized Hybrid A*（IGHA*），通过动态组织节点扩展，实现更高效的树搜索，在保证性能的同时显著减少计算开销。**

- **链接: [https://arxiv.org/pdf/2508.13392v2](https://arxiv.org/pdf/2508.13392v2)**

> **作者:** Sidharth Talia; Oren Salzman; Siddhartha Srinivasa
>
> **备注:** 8 pages, 7 figures, Accepted to IEEE RA-L, Nov 2025
>
> **摘要:** We address the problem of efficiently organizing search over very large trees, which arises in many applications ranging from autonomous driving to aerial vehicles. Here, we are motivated by off-road autonomy, where real-time planning is essential. Classical approaches use graphs of motion primitives and exploit dominance to mitigate the curse of dimensionality and prune expansions efficiently. However, for complex dynamics, repeatedly solving two-point boundary-value problems makes graph construction too slow for fast kinodynamic planning. Hybrid A* (HA*) addressed this challenge by searching over a tree of motion primitives and introducing approximate pruning using a grid-based dominance check. However, choosing the grid resolution is difficult: too coarse risks failure, while too fine leads to excessive expansions and slow planning. We propose Incremental Generalized Hybrid A* (IGHA*), an anytime tree-search framework that dynamically organizes vertex expansions without rigid pruning. IGHA* provably matches or outperforms HA*. For both on-road kinematic and off-road kinodynamic planning queries for a car-like robot, variants of IGHA* use 6x fewer expansions to the best solution compared to an optimized version of HA* (HA*M, an internal baseline). In simulated off- road experiments in a high-fidelity simulator, IGHA* outper- forms HA*M when both are used in the loop with a model predictive controller. We demonstrate real-time performance both in simulation and on a small-scale off-road vehicle, enabling fast, robust planning under complex dynamics. Website: https: //personalrobotics.github.io/IGHAStar/
>
---
#### [replaced 039] DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit
- **分类: cs.RO**

- **简介: 该论文提出DexFruit框架，解决脆弱水果自主抓取易损伤的问题。结合光学触觉感知与扩散策略实现轻柔操作，并提出FruitSplat方法，利用3D高斯溅射量化水果损伤，提升抓取成功率并减少 bruising。**

- **链接: [https://arxiv.org/pdf/2508.07118v3](https://arxiv.org/pdf/2508.07118v3)**

> **作者:** Aiden Swann; Alex Qiu; Matthew Strong; Angelina Zhang; Samuel Morstein; Kai Rayle; Monroe Kennedy
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** DexFruit is a robotic manipulation framework that enables gentle, autonomous handling of fragile fruit and precise evaluation of damage. Many fruits are fragile and prone to bruising, thus requiring humans to manually harvest them with care. In this work, we demonstrate by using optical tactile sensing, autonomous manipulation of fruit with minimal damage can be achieved. We show that our tactile informed diffusion policies outperform baselines in both reduced bruising and pick-and-place success rate across three fruits: strawberries, tomatoes, and blackberries. In addition, we introduce FruitSplat, a novel technique to represent and quantify visual damage in high-resolution 3D representation via 3D Gaussian Splatting (3DGS). Existing metrics for measuring damage lack quantitative rigor or require expensive equipment. With FruitSplat, we distill a 2D strawberry mask as well as a 2D bruise segmentation mask into the 3DGS representation. Furthermore, this representation is modular and general, compatible with any relevant 2D model. Overall, we demonstrate a 92% grasping policy success rate, up to a 20% reduction in visual bruising, and up to an 31% improvement in grasp success rate on challenging fruit compared to our baselines across our three tested fruits. We rigorously evaluate this result with over 630 trials. Please checkout our website at https://dex-fruit.github.io .
>
---
#### [replaced 040] MeshA*: Efficient Path Planning With Motion Primitives
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究基于运动基元的路径规划任务，旨在解决传统格点搜索因分支因子大导致效率低的问题。提出MeshA*算法，在保持最优性与完备性的同时，通过在网格单元上同步拟合运动基元序列，显著提升搜索效率。**

- **链接: [https://arxiv.org/pdf/2412.10320v2](https://arxiv.org/pdf/2412.10320v2)**

> **作者:** Marat Agranovskiy; Konstantin Yakovlev
>
> **备注:** Accepted to AAAI-2026
>
> **摘要:** We study a path planning problem where the possible move actions are represented as a finite set of motion primitives aligned with the grid representation of the environment. That is, each primitive corresponds to a short kinodynamically-feasible motion of an agent and is represented as a sequence of the swept cells of a grid. Typically, heuristic search, i.e. A*, is conducted over the lattice induced by these primitives (lattice-based planning) to find a path. However, due to the large branching factor, such search may be inefficient in practice. To this end, we suggest a novel technique rooted in the idea of searching over the grid cells (as in vanilla A*) simultaneously fitting the possible sequences of the motion primitives into these cells. The resultant algorithm, MeshA*, provably preserves the guarantees on completeness and optimality, on the one hand, and is shown to notably outperform conventional lattice-based planning (x1.5-x2 decrease in the runtime), on the other hand.
>
---
#### [replaced 041] Disturbance Compensation for Safe Kinematic Control of Robotic Systems with Closed Architecture
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对闭源机器人系统内环不可修改且存在扰动与模型不确定性的问题，提出一种可集成于外环的补偿控制方法，结合扰动抑制与安全控制，实现高精度、鲁棒且安全的运动控制。**

- **链接: [https://arxiv.org/pdf/2512.05292v2](https://arxiv.org/pdf/2512.05292v2)**

> **作者:** Fan Zhang; Jinfeng Chen; Joseph J. B. Mvogo Ahanda; Hanz Richter; Ge Lv; Bin Hu; Qin Lin
>
> **备注:** Extended version of the paper submitted for publication. This document contains detailed mathematical derivations and additional experimental results omitted from the submission due to page constraints
>
> **摘要:** In commercial robotic systems, it is common to encounter a closed inner-loop torque controller that is not user-modifiable. However, the outer-loop controller, which sends kinematic commands such as position or velocity for the inner-loop controller to track, is typically exposed to users. In this work, we focus on the development of an easily integrated add-on at the outer-loop layer by combining disturbance rejection control and robust control barrier function for high-performance tracking and safe control of the whole dynamic system of an industrial manipulator. This is particularly beneficial when 1) the inner-loop controller is imperfect, unmodifiable, and uncertain; and 2) the dynamic model exhibits significant uncertainty. Stability analysis, formal safety guarantee proof, and hardware experiments with a PUMA robotic manipulator are presented. Our solution demonstrates superior performance in terms of simplicity of implementation, robustness, tracking precision, and safety compared to the state of the art. Video: https://youtu.be/zw1tanvrV8Q
>
---
#### [replaced 042] SIGN: Safety-Aware Image-Goal Navigation for Autonomous Drones via Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文研究自主无人机的安全视觉目标导航任务，解决未知环境中无定位信息下的导航与避障问题。提出基于强化学习的端到端框架SIGN，结合视觉表征增强训练与深度安全模块，实现自主探索、避障与目标到达。**

- **链接: [https://arxiv.org/pdf/2508.12394v2](https://arxiv.org/pdf/2508.12394v2)**

> **作者:** Zichen Yan; Rui Huang; Lei He; Shao Guo; Lin Zhao
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Image-goal navigation (ImageNav) tasks a robot with autonomously exploring an unknown environment and reaching a location that visually matches a given target image. While prior works primarily study ImageNav for ground robots, enabling this capability for autonomous drones is substantially more challenging due to their need for high-frequency feedback control and global localization for stable flight. In this paper, we propose a novel sim-to-real framework that leverages reinforcement learning (RL) to achieve ImageNav for drones. To enhance visual representation ability, our approach trains the vision backbone with auxiliary tasks, including image perturbations and future transition prediction, which results in more effective policy training. The proposed algorithm enables end-to-end ImageNav with direct velocity control, eliminating the need for external localization. Furthermore, we integrate a depth-based safety module for real-time obstacle avoidance, allowing the drone to safely navigate in cluttered environments. Unlike most existing drone navigation methods that focus solely on reference tracking or obstacle avoidance, our framework supports comprehensive navigation behaviors, including autonomous exploration, obstacle avoidance, and image-goal seeking, without requiring explicit global mapping. Code and model checkpoints are available at https://github.com/Zichen-Yan/SIGN.
>
---
