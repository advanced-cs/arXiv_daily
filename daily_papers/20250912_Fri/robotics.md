# 机器人 cs.RO

- **最新发布 25 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] A Hybrid Hinge-Beam Continuum Robot with Passive Safety Capping for Real-Time Fatigue Awareness
- **分类: cs.RO**

- **简介: 论文提出一种具有被动安全限位的混合铰链梁连续体机器人，用于实时疲劳感知。针对连续体机器人长期使用中因疲劳导致性能下降和结构失效的问题，设计了混合结构、被动限位装置和实时疲劳估计方法，提升其耐久性和安全性。**

- **链接: [http://arxiv.org/pdf/2509.09404v1](http://arxiv.org/pdf/2509.09404v1)**

> **作者:** Tongshun Chen; Zezhou Sun; Yanhan Sun; Yuhao Wang; Dezhen Song; Ke Wu
>
> **摘要:** Cable-driven continuum robots offer high flexibility and lightweight design, making them well-suited for tasks in constrained and unstructured environments. However, prolonged use can induce mechanical fatigue from plastic deformation and material degradation, compromising performance and risking structural failure. In the state of the art, fatigue estimation of continuum robots remains underexplored, limiting long-term operation. To address this, we propose a fatigue-aware continuum robot with three key innovations: (1) a Hybrid Hinge-Beam structure where TwistBeam and BendBeam decouple torsion and bending: passive revolute joints in the BendBeam mitigate stress concentration, while TwistBeam's limited torsional deformation reduces BendBeam stress magnitude, enhancing durability; (2) a Passive Stopper that safely constrains motion via mechanical constraints and employs motor torque sensing to detect corresponding limit torque, ensuring safety and enabling data collection; and (3) a real-time fatigue-awareness method that estimates stiffness from motor torque at the limit pose, enabling online fatigue estimation without additional sensors. Experiments show that the proposed design reduces fatigue accumulation by about 49% compared with a conventional design, while passive mechanical limiting combined with motor-side sensing allows accurate estimation of structural fatigue and damage. These results confirm the effectiveness of the proposed architecture for safe and reliable long-term operation.
>
---
#### [new 002] ObjectReact: Learning Object-Relative Control for Visual Navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出ObjectReact，一种基于对象相对控制的视觉导航方法，解决传统图像相对控制在跨形态部署中的局限性。通过构建相对3D场景图和“WayObject Costmap”表示，实现无需RGB输入的高泛化导航，适用于多种传感器高度和复杂任务。**

- **链接: [http://arxiv.org/pdf/2509.09594v1](http://arxiv.org/pdf/2509.09594v1)**

> **作者:** Sourav Garg; Dustin Craggs; Vineeth Bhat; Lachlan Mares; Stefan Podgorski; Madhava Krishna; Feras Dayoub; Ian Reid
>
> **备注:** CoRL 2025; 23 pages including appendix
>
> **摘要:** Visual navigation using only a single camera and a topological map has recently become an appealing alternative to methods that require additional sensors and 3D maps. This is typically achieved through an "image-relative" approach to estimating control from a given pair of current observation and subgoal image. However, image-level representations of the world have limitations because images are strictly tied to the agent's pose and embodiment. In contrast, objects, being a property of the map, offer an embodiment- and trajectory-invariant world representation. In this work, we present a new paradigm of learning "object-relative" control that exhibits several desirable characteristics: a) new routes can be traversed without strictly requiring to imitate prior experience, b) the control prediction problem can be decoupled from solving the image matching problem, and c) high invariance can be achieved in cross-embodiment deployment for variations across both training-testing and mapping-execution settings. We propose a topometric map representation in the form of a "relative" 3D scene graph, which is used to obtain more informative object-level global path planning costs. We train a local controller, dubbed "ObjectReact", conditioned directly on a high-level "WayObject Costmap" representation that eliminates the need for an explicit RGB input. We demonstrate the advantages of learning object-relative control over its image-relative counterpart across sensor height variations and multiple navigation tasks that challenge the underlying spatial understanding capability, e.g., navigating a map trajectory in the reverse direction. We further show that our sim-only policy is able to generalize well to real-world indoor environments. Code and supplementary material are accessible via project page: https://object-react.github.io/
>
---
#### [new 003] MOFU: Development of a MOrphing Fluffy Unit with Expansion and Contraction Capabilities and Evaluation of the Animacy of Its Movements
- **分类: cs.RO**

- **简介: 论文开发了可膨胀收缩的MOFU机器人，研究其运动对人类感知“生命感”的影响。通过实验发现，体积变化运动显著提升机器人被感知为有生命的程度，为社交机器人设计提供新思路。**

- **链接: [http://arxiv.org/pdf/2509.09613v1](http://arxiv.org/pdf/2509.09613v1)**

> **作者:** Taisei Mogi; Mari Saito; Yoshihiro Nakata
>
> **摘要:** Robots for therapy and social interaction are often intended to evoke "animacy" in humans. While many robots imitate appearance and joint movements, little attention has been given to whole-body expansion-contraction, volume-changing movements observed in living organisms, and their effect on animacy perception. We developed a mobile robot called "MOFU (Morphing Fluffy Unit)," capable of whole-body expansion-contraction with a single motor and covered with a fluffy exterior. MOFU employs a "Jitterbug" structure, a geometric transformation mechanism that enables smooth volume change in diameter from 210 to 280 mm using one actuator. It is also equipped with a differential two-wheel drive mechanism for locomotion. To evaluate the effect of expansion-contraction movements, we conducted an online survey using videos of MOFU's behavior. Participants rated impressions with the Godspeed Questionnaire Series. First, we compared videos of MOFU in a stationary state with and without expansion-contraction and turning, finding that expansion-contraction significantly increased perceived animacy. Second, we hypothesized that presenting two MOFUs would increase animacy compared with a single robot; however, this was not supported, as no significant difference emerged. Exploratory analyses further compared four dual-robot motion conditions. Third, when expansion-contraction was combined with locomotion, animacy ratings were higher than locomotion alone. These results suggest that volume-changing movements such as expansion and contraction enhance perceived animacy in robots and should be considered an important design element in future robot development aimed at shaping human impressions.
>
---
#### [new 004] Rapid Manufacturing of Lightweight Drone Frames Using Single-Tow Architected Composites
- **分类: cs.RO; physics.app-ph**

- **简介: 该论文提出一种基于单丝纤维的3DFiT技术，制造轻量化无人机框架，解决传统复合材料制造复杂结构难、接合处薄弱的问题。通过构建面心立方晶格结构，实现高强度与减重，提升飞行性能。**

- **链接: [http://arxiv.org/pdf/2509.09024v1](http://arxiv.org/pdf/2509.09024v1)**

> **作者:** Md Habib Ullah Khan; Kaiyue Deng; Ismail Mujtaba Khan; Kelvin Fu
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** The demand for lightweight and high-strength composite structures is rapidly growing in aerospace and robotics, particularly for optimized drone frames. However, conventional composite manufacturing methods struggle to achieve complex 3D architectures for weight savings and rely on assembling separate components, which introduce weak points at the joints. Additionally, maintaining continuous fiber reinforcement remains challenging, limiting structural efficiency. In this study, we demonstrate the lightweight Face Centered Cubic (FFC) lattice structured conceptualization of drone frames for weight reduction and complex topology fabrication through 3D Fiber Tethering (3DFiT) using continuous single tow fiber ensuring precise fiber alignment, eliminating weak points associated with traditional composite assembly. Mechanical testing demonstrates that the fabricated drone frame exhibits a high specific strength of around four to eight times the metal and thermoplastic, outperforming other conventional 3D printing methods. The drone frame weighs only 260 g, making it 10% lighter than the commercial DJI F450 frame, enhancing structural integrity and contributing to an extended flight time of three minutes, while flight testing confirms its stability and durability under operational conditions. The findings demonstrate the potential of single tow lattice truss-based drone frames, with 3DFiT serving as a scalable and efficient manufacturing method.
>
---
#### [new 005] BagIt! An Adaptive Dual-Arm Manipulation of Fabric Bags for Object Bagging
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种双臂机器人系统，用于自动将物体装入可变形袋中。通过自适应结构兴趣点策略、视觉反馈与优化算法，解决传统袋装任务中因袋子变形带来的挑战，实现精准、鲁棒的自动化装袋。属于机器人柔性物体操作任务。**

- **链接: [http://arxiv.org/pdf/2509.09484v1](http://arxiv.org/pdf/2509.09484v1)**

> **作者:** Peng Zhou; Jiaming Qi; Hongmin Wu; Chen Wang; Yizhou Chen; Zeqing Zhang
>
> **摘要:** Bagging tasks, commonly found in industrial scenarios, are challenging considering deformable bags' complicated and unpredictable nature. This paper presents an automated bagging system from the proposed adaptive Structure-of-Interest (SOI) manipulation strategy for dual robot arms. The system dynamically adjusts its actions based on real-time visual feedback, removing the need for pre-existing knowledge of bag properties. Our framework incorporates Gaussian Mixture Models (GMM) for estimating SOI states, optimization techniques for SOI generation, motion planning via Constrained Bidirectional Rapidly-exploring Random Tree (CBiRRT), and dual-arm coordination using Model Predictive Control (MPC). Extensive experiments validate the capability of our system to perform precise and robust bagging across various objects, showcasing its adaptability. This work offers a new solution for robotic deformable object manipulation (DOM), particularly in automated bagging tasks. Video of this work is available at https://youtu.be/6JWjCOeTGiQ.
>
---
#### [new 006] RENet: Fault-Tolerant Motion Control for Quadruped Robots via Redundant Estimator Networks under Visual Collapse
- **分类: cs.RO**

- **简介: 该论文提出RENet框架，用于解决四足机器人在户外视觉失效时的运动控制问题。通过冗余估计网络实现双估计器架构，在视觉退化情况下保持稳定运动，提升室外环境下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.09283v1](http://arxiv.org/pdf/2509.09283v1)**

> **作者:** Yueqi Zhang; Quancheng Qian; Taixian Hou; Peng Zhai; Xiaoyi Wei; Kangmai Hu; Jiafu Yi; Lihua Zhang
>
> **备注:** Accepted for IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Vision-based locomotion in outdoor environments presents significant challenges for quadruped robots. Accurate environmental prediction and effective handling of depth sensor noise during real-world deployment remain difficult, severely restricting the outdoor applications of such algorithms. To address these deployment challenges in vision-based motion control, this letter proposes the Redundant Estimator Network (RENet) framework. The framework employs a dual-estimator architecture that ensures robust motion performance while maintaining deployment stability during onboard vision failures. Through an online estimator adaptation, our method enables seamless transitions between estimation modules when handling visual perception uncertainties. Experimental validation on a real-world robot demonstrates the framework's effectiveness in complex outdoor environments, showing particular advantages in scenarios with degraded visual perception. This framework demonstrates its potential as a practical solution for reliable robotic deployment in challenging field conditions. Project website: https://RENet-Loco.github.io/
>
---
#### [new 007] Occupancy-aware Trajectory Planning for Autonomous Valet Parking in Uncertain Dynamic Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种基于占用感知的轨迹规划方法，用于解决自动驾驶代客泊车在动态不确定环境中的停车效率与安全性问题。通过预测车位占用状态和自适应规划策略，提升泊车效率与轨迹平滑性。**

- **链接: [http://arxiv.org/pdf/2509.09206v1](http://arxiv.org/pdf/2509.09206v1)**

> **作者:** Farhad Nawaz; Faizan M. Tariq; Sangjae Bae; David Isele; Avinash Singh; Nadia Figueroa; Nikolai Matni; Jovin D'sa
>
> **摘要:** Accurately reasoning about future parking spot availability and integrated planning is critical for enabling safe and efficient autonomous valet parking in dynamic, uncertain environments. Unlike existing methods that rely solely on instantaneous observations or static assumptions, we present an approach that predicts future parking spot occupancy by explicitly distinguishing between initially vacant and occupied spots, and by leveraging the predicted motion of dynamic agents. We introduce a probabilistic spot occupancy estimator that incorporates partial and noisy observations within a limited Field-of-View (FoV) model and accounts for the evolving uncertainty of unobserved regions. Coupled with this, we design a strategy planner that adaptively balances goal-directed parking maneuvers with exploratory navigation based on information gain, and intelligently incorporates wait-and-go behaviors at promising spots. Through randomized simulations emulating large parking lots, we demonstrate that our framework significantly improves parking efficiency, safety margins, and trajectory smoothness compared to existing approaches.
>
---
#### [new 008] AGILOped: Agile Open-Source Humanoid Robot for Research
- **分类: cs.RO**

- **简介: 论文提出AGILOped，一款开源人形机器人，旨在解决高性能与可及性之间的矛盾。采用市售高功率密度执行器和标准电子元件，实现低成本、易操作的科研平台，支持行走、跳跃等实验，推动人形机器人研究。**

- **链接: [http://arxiv.org/pdf/2509.09364v1](http://arxiv.org/pdf/2509.09364v1)**

> **作者:** Grzegorz Ficht; Luis Denninger; Sven Behnke
>
> **备注:** 10th IEEE International Conference on Advanced Robotics and Mechatronics (ARM), Portsmouth, UK, August 2025
>
> **摘要:** With academic and commercial interest for humanoid robots peaking, multiple platforms are being developed. Through a high level of customization, they showcase impressive performance. Most of these systems remain closed-source or have high acquisition and maintenance costs, however. In this work, we present AGILOped - an open-source humanoid robot that closes the gap between high performance and accessibility. Our robot is driven by off-the-shelf backdrivable actuators with high power density and uses standard electronic components. With a height of 110 cm and weighing only 14.5 kg, AGILOped can be operated without a gantry by a single person. Experiments in walking, jumping, impact mitigation and getting-up demonstrate its viability for use in research.
>
---
#### [new 009] A Neuromorphic Incipient Slip Detection System using Papillae Morphology
- **分类: cs.RO**

- **简介: 该论文提出一种基于神经形态触觉传感的初滑检测系统，用于提升机器人抓取安全性。通过仿生乳突结构与脉冲卷积网络实现高精度分类，在动态条件下提前360ms检测初滑，解决边缘设备能耗与实时性问题。**

- **链接: [http://arxiv.org/pdf/2509.09546v1](http://arxiv.org/pdf/2509.09546v1)**

> **作者:** Yanhui Lu; Zeyu Deng; Stephen J. Redmond; Efi Psomopoulou; Benjamin Ward-Cherrier
>
> **备注:** 7 pages, 12 figures. Submitted to IEEE Robotics and Automation Letters (RAL), under review
>
> **摘要:** Detecting incipient slip enables early intervention to prevent object slippage and enhance robotic manipulation safety. However, deploying such systems on edge platforms remains challenging, particularly due to energy constraints. This work presents a neuromorphic tactile sensing system based on the NeuroTac sensor with an extruding papillae-based skin and a spiking convolutional neural network (SCNN) for slip-state classification. The SCNN model achieves 94.33% classification accuracy across three classes (no slip, incipient slip, and gross slip) in slip conditions induced by sensor motion. Under the dynamic gravity-induced slip validation conditions, after temporal smoothing of the SCNN's final-layer spike counts, the system detects incipient slip at least 360 ms prior to gross slip across all trials, consistently identifying incipient slip before gross slip occurs. These results demonstrate that this neuromorphic system has stable and responsive incipient slip detection capability.
>
---
#### [new 010] Kinetostatics and Particle-Swarm Optimization of Vehicle-Mounted Underactuated Metamorphic Loading Manipulators
- **分类: cs.RO**

- **简介: 论文提出一种欠驱动变胞装载机械臂，解决传统固定自由度装载机构的冗余执行器、复杂控制与适应性差问题。通过几何约束实现拓扑重构与柔性运动，结合粒子群优化提升抓取性能，验证其在动态环境中的有效性与通用性。**

- **链接: [http://arxiv.org/pdf/2509.09093v1](http://arxiv.org/pdf/2509.09093v1)**

> **作者:** Nan Mao; Guanglu Jia; Junpeng Chen; Emmanouil Spyrakos-Papastavridis; Jian S. Dai
>
> **备注:** 50 pages, 19 figures
>
> **摘要:** Fixed degree-of-freedom (DoF) loading mechanisms often suffer from excessive actuators, complex control, and limited adaptability to dynamic tasks. This study proposes an innovative mechanism of underactuated metamorphic loading manipulators (UMLM), integrating a metamorphic arm with a passively adaptive gripper. The metamorphic arm exploits geometric constraints, enabling the topology reconfiguration and flexible motion trajectories without additional actuators. The adaptive gripper, driven entirely by the arm, conforms to diverse objects through passive compliance. A structural model is developed, and a kinetostatics analysis is conducted to investigate isomorphic grasping configurations. To optimize performance, Particle-Swarm Optimization (PSO) is utilized to refine the gripper's dimensional parameters, ensuring robust adaptability across various applications. Simulation results validate the UMLM's easily implemented control strategy, operational versatility, and effectiveness in grasping diverse objects in dynamic environments. This work underscores the practical potential of underactuated metamorphic mechanisms in applications requiring efficient and adaptable loading solutions. Beyond the specific design, this generalized modeling and optimization framework extends to a broader class of manipulators, offering a scalable approach to the development of robotic systems that require efficiency, flexibility, and robust performance.
>
---
#### [new 011] LIPM-Guided Reinforcement Learning for Stable and Perceptive Locomotion in Bipedal Robots
- **分类: cs.RO**

- **简介: 论文提出基于LIPM的强化学习方法，解决双足机器人在复杂户外环境中的稳定感知运动问题。设计新型奖励函数与双评价器架构，提升地形适应性与抗干扰能力。**

- **链接: [http://arxiv.org/pdf/2509.09106v1](http://arxiv.org/pdf/2509.09106v1)**

> **作者:** Haokai Su; Haoxiang Luo; Shunpeng Yang; Kaiwen Jiang; Wei Zhang; Hua Chen
>
> **摘要:** Achieving stable and robust perceptive locomotion for bipedal robots in unstructured outdoor environments remains a critical challenge due to complex terrain geometry and susceptibility to external disturbances. In this work, we propose a novel reward design inspired by the Linear Inverted Pendulum Model (LIPM) to enable perceptive and stable locomotion in the wild. The LIPM provides theoretical guidance for dynamic balance by regulating the center of mass (CoM) height and the torso orientation. These are key factors for terrain-aware locomotion, as they help ensure a stable viewpoint for the robot's camera. Building on this insight, we design a reward function that promotes balance and dynamic stability while encouraging accurate CoM trajectory tracking. To adaptively trade off between velocity tracking and stability, we leverage the Reward Fusion Module (RFM) approach that prioritizes stability when needed. A double-critic architecture is adopted to separately evaluate stability and locomotion objectives, improving training efficiency and robustness. We validate our approach through extensive experiments on a bipedal robot in both simulation and real-world outdoor environments. The results demonstrate superior terrain adaptability, disturbance rejection, and consistent performance across a wide range of speeds and perceptual conditions.
>
---
#### [new 012] Dexplore: Scalable Neural Control for Dexterous Manipulation from Reference-Scoped Exploration
- **分类: cs.RO; cs.CV**

- **简介: 论文提出Dexplore方法，解决机器人灵巧操作中MoCap数据利用不足与误差累积问题。通过单循环优化联合重定向与跟踪，直接从大规模MoCap数据学习控制策略，并蒸馏为视觉技能控制器，提升泛化与鲁棒性。属于机器人控制与模仿学习任务。**

- **链接: [http://arxiv.org/pdf/2509.09671v1](http://arxiv.org/pdf/2509.09671v1)**

> **作者:** Sirui Xu; Yu-Wei Chao; Liuyu Bian; Arsalan Mousavian; Yu-Xiong Wang; Liang-Yan Gui; Wei Yang
>
> **备注:** CoRL 2025
>
> **摘要:** Hand-object motion-capture (MoCap) repositories offer large-scale, contact-rich demonstrations and hold promise for scaling dexterous robotic manipulation. Yet demonstration inaccuracies and embodiment gaps between human and robot hands limit the straightforward use of these data. Existing methods adopt a three-stage workflow, including retargeting, tracking, and residual correction, which often leaves demonstrations underused and compound errors across stages. We introduce Dexplore, a unified single-loop optimization that jointly performs retargeting and tracking to learn robot control policies directly from MoCap at scale. Rather than treating demonstrations as ground truth, we use them as soft guidance. From raw trajectories, we derive adaptive spatial scopes, and train with reinforcement learning to keep the policy in-scope while minimizing control effort and accomplishing the task. This unified formulation preserves demonstration intent, enables robot-specific strategies to emerge, improves robustness to noise, and scales to large demonstration corpora. We distill the scaled tracking policy into a vision-based, skill-conditioned generative controller that encodes diverse manipulation skills in a rich latent representation, supporting generalization across objects and real-world deployment. Taken together, these contributions position Dexplore as a principled bridge that transforms imperfect demonstrations into effective training signals for dexterous manipulation.
>
---
#### [new 013] Multi Robot Coordination in Highly Dynamic Environments: Tackling Asymmetric Obstacles and Limited Communication
- **分类: cs.RO; cs.AI**

- **简介: 论文研究多机器人在通信受限、环境部分可观测且存在非对称动态障碍物下的协同任务分配问题。提出一种基于市场机制的分布式协调方法，有效减少任务重叠，已在仿真和RoboCup比赛中验证。**

- **链接: [http://arxiv.org/pdf/2509.08859v1](http://arxiv.org/pdf/2509.08859v1)**

> **作者:** Vincenzo Suriani; Daniele Affinita; Domenico D. Bloisi; Daniele Nardi
>
> **备注:** The 19th International Conference on Intelligent Autonomous Systems (IAS 19), 2025, Genoa
>
> **摘要:** Coordinating a fully distributed multi-agent system (MAS) can be challenging when the communication channel has very limited capabilities in terms of sending rate and packet payload. When the MAS has to deal with active obstacles in a highly partially observable environment, the communication channel acquires considerable relevance. In this paper, we present an approach to deal with task assignments in extremely active scenarios, where tasks need to be frequently reallocated among the agents participating in the coordination process. Inspired by market-based task assignments, we introduce a novel distributed coordination method to orchestrate autonomous agents' actions efficiently in low communication scenarios. In particular, our algorithm takes into account asymmetric obstacles. While in the real world, the majority of obstacles are asymmetric, they are usually treated as symmetric ones, thus limiting the applicability of existing methods. To summarize, the presented architecture is designed to tackle scenarios where the obstacles are active and asymmetric, the communication channel is poor and the environment is partially observable. Our approach has been validated in simulation and in the real world, using a team of NAO robots during official RoboCup competitions. Experimental results show a notable reduction in task overlaps in limited communication settings, with a decrease of 52% in the most frequent reallocated task.
>
---
#### [new 014] VLA-Adapter: An Effective Paradigm for Tiny-Scale Vision-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文提出VLA-Adapter，解决小规模视觉-语言-动作模型训练成本高的问题。通过轻量策略模块和桥接注意力机制，在无需大规模预训练和机器人数据下，实现高性能与快速推理，降低部署门槛。**

- **链接: [http://arxiv.org/pdf/2509.09372v1](http://arxiv.org/pdf/2509.09372v1)**

> **作者:** Yihao Wang; Pengxiang Ding; Lingxiao Li; Can Cui; Zirui Ge; Xinyang Tong; Wenxuan Song; Han Zhao; Wei Zhao; Pengxu Hou; Siteng Huang; Yifan Tang; Wenhui Wang; Ru Zhang; Jianyi Liu; Donglin Wang
>
> **摘要:** Vision-Language-Action (VLA) models typically bridge the gap between perceptual and action spaces by pre-training a large-scale Vision-Language Model (VLM) on robotic data. While this approach greatly enhances performance, it also incurs significant training costs. In this paper, we investigate how to effectively bridge vision-language (VL) representations to action (A). We introduce VLA-Adapter, a novel paradigm designed to reduce the reliance of VLA models on large-scale VLMs and extensive pre-training. To this end, we first systematically analyze the effectiveness of various VL conditions and present key findings on which conditions are essential for bridging perception and action spaces. Based on these insights, we propose a lightweight Policy module with Bridge Attention, which autonomously injects the optimal condition into the action space. In this way, our method achieves high performance using only a 0.5B-parameter backbone, without any robotic data pre-training. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that VLA-Adapter not only achieves state-of-the-art level performance, but also offers the fast inference speed reported to date. Furthermore, thanks to the proposed advanced bridging paradigm, VLA-Adapter enables the training of a powerful VLA model in just 8 hours on a single consumer-grade GPU, greatly lowering the barrier to deploying the VLA model. Project page: https://vla-adapter.github.io/.
>
---
#### [new 015] KoopMotion: Learning Almost Divergence Free Koopman Flow Fields for Motion Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出KoopMotion方法，利用Koopman算子学习几乎无散度的流场，解决运动规划中从初始状态收敛到目标轨迹的问题。通过参数化动力系统，实现高效、平滑的轨迹跟踪，并在多个数据集和物理机器人上验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.09074v1](http://arxiv.org/pdf/2509.09074v1)**

> **作者:** Alice Kate Li; Thales C Silva; Victoria Edwards; Vijay Kumar; M. Ani Hsieh
>
> **备注:** Accepted to CoRL 2025 (Conference on Robot Learning). 15 pages 11 figures
>
> **摘要:** In this work, we propose a novel flow field-based motion planning method that drives a robot from any initial state to a desired reference trajectory such that it converges to the trajectory's end point. Despite demonstrated efficacy in using Koopman operator theory for modeling dynamical systems, Koopman does not inherently enforce convergence to desired trajectories nor to specified goals -- a requirement when learning from demonstrations (LfD). We present KoopMotion which represents motion flow fields as dynamical systems, parameterized by Koopman Operators to mimic desired trajectories, and leverages the divergence properties of the learnt flow fields to obtain smooth motion fields that converge to a desired reference trajectory when a robot is placed away from the desired trajectory, and tracks the trajectory until the end point. To demonstrate the effectiveness of our approach, we show evaluations of KoopMotion on the LASA human handwriting dataset and a 3D manipulator end-effector trajectory dataset, including spectral analysis. We also perform experiments on a physical robot, verifying KoopMotion on a miniature autonomous surface vehicle operating in a non-static fluid flow environment. Our approach is highly sample efficient in both space and time, requiring only 3\% of the LASA dataset to generate dense motion plans. Additionally, KoopMotion provides a significant improvement over baselines when comparing metrics that measure spatial and temporal dynamics modeling efficacy.
>
---
#### [new 016] AEOS: Active Environment-aware Optimal Scanning Control for UAV LiDAR-Inertial Odometry in Complex Scenes
- **分类: cs.RO**

- **简介: 该论文提出AEOS框架，解决无人机LiDAR-Inertial Odometry在复杂场景中的定位精度问题。通过结合MPC与RL，实现自适应扫描控制，提升环境感知与地图构建性能。**

- **链接: [http://arxiv.org/pdf/2509.09141v1](http://arxiv.org/pdf/2509.09141v1)**

> **作者:** Jianping Li; Xinhang Xu; Zhongyuan Liu; Shenghai Yuan; Muqing Cao; Lihua Xie
>
> **摘要:** LiDAR-based 3D perception and localization on unmanned aerial vehicles (UAVs) are fundamentally limited by the narrow field of view (FoV) of compact LiDAR sensors and the payload constraints that preclude multi-sensor configurations. Traditional motorized scanning systems with fixed-speed rotations lack scene awareness and task-level adaptability, leading to degraded odometry and mapping performance in complex, occluded environments. Inspired by the active sensing behavior of owls, we propose AEOS (Active Environment-aware Optimal Scanning), a biologically inspired and computationally efficient framework for adaptive LiDAR control in UAV-based LiDAR-Inertial Odometry (LIO). AEOS combines model predictive control (MPC) and reinforcement learning (RL) in a hybrid architecture: an analytical uncertainty model predicts future pose observability for exploitation, while a lightweight neural network learns an implicit cost map from panoramic depth representations to guide exploration. To support scalable training and generalization, we develop a point cloud-based simulation environment with real-world LiDAR maps across diverse scenes, enabling sim-to-real transfer. Extensive experiments in both simulation and real-world environments demonstrate that AEOS significantly improves odometry accuracy compared to fixed-rate, optimization-only, and fully learned baselines, while maintaining real-time performance under onboard computational constraints. The project page can be found at https://kafeiyin00.github.io/AEOS/.
>
---
#### [new 017] SMapper: A Multi-Modal Data Acquisition Platform for SLAM Benchmarking
- **分类: cs.RO**

- **简介: 该论文提出SMapper，一种多模态数据采集平台，用于SLAM研究。旨在解决现有数据集在传感器多样性、环境覆盖及硬件可复现性方面的不足，提供同步多传感器数据与高精度地面真实轨迹，推动SLAM算法的开发与评估。**

- **链接: [http://arxiv.org/pdf/2509.09509v1](http://arxiv.org/pdf/2509.09509v1)**

> **作者:** Pedro Miguel Bastos Soares; Ali Tourani; Miguel Fernandez-Cortizas; Asier Bikandi Noya; Jose Luis Sanchez-Lopez; Holger Voos
>
> **备注:** 12 pages, 6 figures, 5 tables
>
> **摘要:** Advancing research in fields like Simultaneous Localization and Mapping (SLAM) and autonomous navigation critically depends on reliable and reproducible multimodal datasets. While several influential datasets have driven progress in these domains, they often suffer from limitations in sensing modalities, environmental diversity, and the reproducibility of the underlying hardware setups. To address these challenges, this paper introduces SMapper, a novel open-hardware, multi-sensor platform designed explicitly for, though not limited to, SLAM research. The device integrates synchronized LiDAR, multi-camera, and inertial sensing, supported by a robust calibration and synchronization pipeline that ensures precise spatio-temporal alignment across modalities. Its open and replicable design allows researchers to extend its capabilities and reproduce experiments across both handheld and robot-mounted scenarios. To demonstrate its practicality, we additionally release SMapper-light, a publicly available SLAM dataset containing representative indoor and outdoor sequences. The dataset includes tightly synchronized multimodal data and ground-truth trajectories derived from offline LiDAR-based SLAM with sub-centimeter accuracy, alongside dense 3D reconstructions. Furthermore, the paper contains benchmarking results on state-of-the-art LiDAR and visual SLAM frameworks using the SMapper-light dataset. By combining open-hardware design, reproducible data collection, and comprehensive benchmarking, SMapper establishes a robust foundation for advancing SLAM algorithm development, evaluation, and reproducibility.
>
---
#### [new 018] OmniEVA: Embodied Versatile Planner via Task-Adaptive 3D-Grounded and Embodiment-aware Reasoning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出OmniEVA，解决具身智能系统中的几何适应性与具身约束性问题。通过任务自适应3D grounding和具身感知推理，提升多模态模型在复杂空间任务中的规划能力与可行性。**

- **链接: [http://arxiv.org/pdf/2509.09332v1](http://arxiv.org/pdf/2509.09332v1)**

> **作者:** Yuecheng Liu; Dafeng Chi; Shiguang Wu; Zhanguang Zhang; Yuzheng Zhuang; Bowen Yang; He Zhu; Lingfeng Zhang; Pengwei Xie; David Gamaliel Arcos Bravo; Yingxue Zhang; Jianye Hao; Xingyue Quan
>
> **摘要:** Recent advances in multimodal large language models (MLLMs) have opened new opportunities for embodied intelligence, enabling multimodal understanding, reasoning, and interaction, as well as continuous spatial decision-making. Nevertheless, current MLLM-based embodied systems face two critical limitations. First, Geometric Adaptability Gap: models trained solely on 2D inputs or with hard-coded 3D geometry injection suffer from either insufficient spatial information or restricted 2D generalization, leading to poor adaptability across tasks with diverse spatial demands. Second, Embodiment Constraint Gap: prior work often neglects the physical constraints and capacities of real robots, resulting in task plans that are theoretically valid but practically infeasible.To address these gaps, we introduce OmniEVA -- an embodied versatile planner that enables advanced embodied reasoning and task planning through two pivotal innovations: (1) a Task-Adaptive 3D Grounding mechanism, which introduces a gated router to perform explicit selective regulation of 3D fusion based on contextual requirements, enabling context-aware 3D grounding for diverse embodied tasks. (2) an Embodiment-Aware Reasoning framework that jointly incorporates task goals and embodiment constraints into the reasoning loop, resulting in planning decisions that are both goal-directed and executable. Extensive experimental results demonstrate that OmniEVA not only achieves state-of-the-art general embodied reasoning performance, but also exhibits a strong ability across a wide range of downstream scenarios. Evaluations of a suite of proposed embodied benchmarks, including both primitive and composite tasks, confirm its robust and versatile planning capabilities. Project page: https://omnieva.github.io
>
---
#### [new 019] SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出SimpleVLA-RL，通过强化学习优化视觉语言动作模型，解决数据稀缺与泛化能力不足问题。方法包括轨迹采样、并行化与损失优化，实现实验室与真实任务的SoTA性能。**

- **链接: [http://arxiv.org/pdf/2509.09674v1](http://arxiv.org/pdf/2509.09674v1)**

> **作者:** Haozhan Li; Yuxin Zuo; Jiale Yu; Yuhao Zhang; Zhaohui Yang; Kaiyan Zhang; Xuekai Zhu; Yuchen Zhang; Tianxing Chen; Ganqu Cui; Dehui Wang; Dingxiang Luo; Yuchen Fan; Youbang Sun; Jia Zeng; Jiangmiao Pang; Shanghang Zhang; Yu Wang; Yao Mu; Bowen Zhou; Ning Ding
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as a powerful paradigm for robotic manipulation. Despite substantial progress enabled by large-scale pretraining and supervised fine-tuning (SFT), these models face two fundamental challenges: (i) the scarcity and high cost of large-scale human-operated robotic trajectories required for SFT scaling, and (ii) limited generalization to tasks involving distribution shift. Recent breakthroughs in Large Reasoning Models (LRMs) demonstrate that reinforcement learning (RL) can dramatically enhance step-by-step reasoning capabilities, raising a natural question: Can RL similarly improve the long-horizon step-by-step action planning of VLA? In this work, we introduce SimpleVLA-RL, an efficient RL framework tailored for VLA models. Building upon veRL, we introduce VLA-specific trajectory sampling, scalable parallelization, multi-environment rendering, and optimized loss computation. When applied to OpenVLA-OFT, SimpleVLA-RL achieves SoTA performance on LIBERO and even outperforms $\pi_0$ on RoboTwin 1.0\&2.0 with the exploration-enhancing strategies we introduce. SimpleVLA-RL not only reduces dependence on large-scale data and enables robust generalization, but also remarkably surpasses SFT in real-world tasks. Moreover, we identify a novel phenomenon ``pushcut'' during RL training, wherein the policy discovers previously unseen patterns beyond those seen in the previous training process. Github: https://github.com/PRIME-RL/SimpleVLA-RL
>
---
#### [new 020] Model-Agnostic Open-Set Air-to-Air Visual Object Detection for Reliable UAV Perception
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 论文提出一种模型无关的开放集检测框架，用于提升无人机在复杂空对空场景下的视觉感知可靠性。解决传统封闭集检测器在领域偏移和数据损坏下的性能下降问题，通过嵌入空间熵建模与谱归一化等方法增强未知目标拒绝能力，并在真实飞行测试中验证有效性。**

- **链接: [http://arxiv.org/pdf/2509.09297v1](http://arxiv.org/pdf/2509.09297v1)**

> **作者:** Spyridon Loukovitis; Anastasios Arsenos; Vasileios Karampinis; Athanasios Voulodimos
>
> **摘要:** Open-set detection is crucial for robust UAV autonomy in air-to-air object detection under real-world conditions. Traditional closed-set detectors degrade significantly under domain shifts and flight data corruption, posing risks to safety-critical applications. We propose a novel, model-agnostic open-set detection framework designed specifically for embedding-based detectors. The method explicitly handles unknown object rejection while maintaining robustness against corrupted flight data. It estimates semantic uncertainty via entropy modeling in the embedding space and incorporates spectral normalization and temperature scaling to enhance open-set discrimination. We validate our approach on the challenging AOT aerial benchmark and through extensive real-world flight tests. Comprehensive ablation studies demonstrate consistent improvements over baseline methods, achieving up to a 10\% relative AUROC gain compared to standard YOLO-based detectors. Additionally, we show that background rejection further strengthens robustness without compromising detection accuracy, making our solution particularly well-suited for reliable UAV perception in dynamic air-to-air environments.
>
---
#### [new 021] Curriculum-Based Multi-Tier Semantic Exploration via Deep Reinforcement Learning
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出一种基于课程学习的多层级语义探索DRL框架，解决自主智能体在复杂环境中高效探索与语义理解的问题。通过集成VLM和分层奖励机制，提升对象发现率与语义导航能力。**

- **链接: [http://arxiv.org/pdf/2509.09356v1](http://arxiv.org/pdf/2509.09356v1)**

> **作者:** Abdel Hakim Drid; Vincenzo Suriani; Daniele Nardi; Abderrezzak Debilou
>
> **备注:** The 19th International Conference on Intelligent Autonomous Systems (IAS 19), 2025, Genoa
>
> **摘要:** Navigating and understanding complex and unknown environments autonomously demands more than just basic perception and movement from embodied agents. Truly effective exploration requires agents to possess higher-level cognitive abilities, the ability to reason about their surroundings, and make more informed decisions regarding exploration strategies. However, traditional RL approaches struggle to balance efficient exploration and semantic understanding due to limited cognitive capabilities embedded in the small policies for the agents, leading often to human drivers when dealing with semantic exploration. In this paper, we address this challenge by presenting a novel Deep Reinforcement Learning (DRL) architecture that is specifically designed for resource efficient semantic exploration. A key methodological contribution is the integration of a Vision-Language Model (VLM) common-sense through a layered reward function. The VLM query is modeled as a dedicated action, allowing the agent to strategically query the VLM only when deemed necessary for gaining external guidance, thereby conserving resources. This mechanism is combined with a curriculum learning strategy designed to guide learning at different levels of complexity to ensure robust and stable learning. Our experimental evaluation results convincingly demonstrate that our agent achieves significantly enhanced object discovery rates and develops a learned capability to effectively navigate towards semantically rich regions. Furthermore, it also shows a strategic mastery of when to prompt for external environmental information. By demonstrating a practical and scalable method for embedding common-sense semantic reasoning with autonomous agents, this research provides a novel approach to pursuing a fully intelligent and self-guided exploration in robotics.
>
---
#### [new 022] Classification of Driver Behaviour Using External Observation Techniques for Autonomous Vehicles
- **分类: cs.CV; cs.AI; cs.ET; cs.RO; eess.IV**

- **简介: 该论文提出一种基于外部观测的驾驶员行为分类系统，用于自动驾驶车辆。任务是识别分心和受干扰驾驶行为，采用YOLO模型和自定义车道估计算法，解决非联网车辆行为分析问题，提升道路安全。**

- **链接: [http://arxiv.org/pdf/2509.09349v1](http://arxiv.org/pdf/2509.09349v1)**

> **作者:** Ian Nell; Shane Gilroy
>
> **摘要:** Road traffic accidents remain a significant global concern, with human error, particularly distracted and impaired driving, among the leading causes. This study introduces a novel driver behavior classification system that uses external observation techniques to detect indicators of distraction and impairment. The proposed framework employs advanced computer vision methodologies, including real-time object tracking, lateral displacement analysis, and lane position monitoring. The system identifies unsafe driving behaviors such as excessive lateral movement and erratic trajectory patterns by implementing the YOLO object detection model and custom lane estimation algorithms. Unlike systems reliant on inter-vehicular communication, this vision-based approach enables behavioral analysis of non-connected vehicles. Experimental evaluations on diverse video datasets demonstrate the framework's reliability and adaptability across varying road and environmental conditions.
>
---
#### [new 023] ProgD: Progressive Multi-scale Decoding with Dynamic Graphs for Joint Multi-agent Motion Forecasting
- **分类: cs.AI; cs.RO**

- **简介: 论文提出ProgD方法，用于多智能体联合运动预测任务，解决交互关系动态变化带来的不确定性问题。通过动态异构图建模和多尺度解码策略，实现更准确的未来运动预测，在多个基准测试中取得最优性能。**

- **链接: [http://arxiv.org/pdf/2509.09210v1](http://arxiv.org/pdf/2509.09210v1)**

> **作者:** Xing Gao; Zherui Huang; Weiyao Lin; Xiao Sun
>
> **摘要:** Accurate motion prediction of surrounding agents is crucial for the safe planning of autonomous vehicles. Recent advancements have extended prediction techniques from individual agents to joint predictions of multiple interacting agents, with various strategies to address complex interactions within future motions of agents. However, these methods overlook the evolving nature of these interactions. To address this limitation, we propose a novel progressive multi-scale decoding strategy, termed ProgD, with the help of dynamic heterogeneous graph-based scenario modeling. In particular, to explicitly and comprehensively capture the evolving social interactions in future scenarios, given their inherent uncertainty, we design a progressive modeling of scenarios with dynamic heterogeneous graphs. With the unfolding of such dynamic heterogeneous graphs, a factorized architecture is designed to process the spatio-temporal dependencies within future scenarios and progressively eliminate uncertainty in future motions of multiple agents. Furthermore, a multi-scale decoding procedure is incorporated to improve on the future scenario modeling and consistent prediction of agents' future motion. The proposed ProgD achieves state-of-the-art performance on the INTERACTION multi-agent prediction benchmark, ranking $1^{st}$, and the Argoverse 2 multi-world forecasting benchmark.
>
---
#### [new 024] Visual Grounding from Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Talk2Event基准，解决事件相机与自然语言理解结合的问题。通过构建包含场景、标注对象和参考表达的数据集，支持可解释的多模态感知任务，推动动态环境下的语义理解与机器人等应用发展。**

- **链接: [http://arxiv.org/pdf/2509.09584v1](http://arxiv.org/pdf/2509.09584v1)**

> **作者:** Lingdong Kong; Dongyue Lu; Ao Liang; Rong Li; Yuhao Dong; Tianshuai Hu; Lai Xing Ng; Wei Tsang Ooi; Benoit R. Cottereau
>
> **备注:** Abstract Paper (Non-Archival) @ ICCV 2025 NeVi Workshop
>
> **摘要:** Event cameras capture changes in brightness with microsecond precision and remain reliable under motion blur and challenging illumination, offering clear advantages for modeling highly dynamic scenes. Yet, their integration with natural language understanding has received little attention, leaving a gap in multimodal perception. To address this, we introduce Talk2Event, the first large-scale benchmark for language-driven object grounding using event data. Built on real-world driving scenarios, Talk2Event comprises 5,567 scenes, 13,458 annotated objects, and more than 30,000 carefully validated referring expressions. Each expression is enriched with four structured attributes -- appearance, status, relation to the viewer, and relation to surrounding objects -- that explicitly capture spatial, temporal, and relational cues. This attribute-centric design supports interpretable and compositional grounding, enabling analysis that moves beyond simple object recognition to contextual reasoning in dynamic environments. We envision Talk2Event as a foundation for advancing multimodal and temporally-aware perception, with applications spanning robotics, human-AI interaction, and so on.
>
---
#### [new 025] Global Optimization of Stochastic Black-Box Functions with Arbitrary Noise Distributions using Wilson Score Kernel Density Estimation
- **分类: stat.ML; cs.LG; cs.RO**

- **简介: 论文提出使用WS-KDE方法优化具有任意噪声分布的随机黑盒函数，解决贝叶斯优化中置信区间估计依赖模型或大量评估的问题。该方法适用于输出在[0,1]区间内的函数，应用于振动送料器自动陷阱设计。**

- **链接: [http://arxiv.org/pdf/2509.09238v1](http://arxiv.org/pdf/2509.09238v1)**

> **作者:** Thorbjørn Mosekjær Iversen; Lars Carøe Sørensen; Simon Faarvang Mathiesen; Henrik Gordon Petersen
>
> **摘要:** Many optimization problems in robotics involve the optimization of time-expensive black-box functions, such as those involving complex simulations or evaluation of real-world experiments. Furthermore, these functions are often stochastic as repeated experiments are subject to unmeasurable disturbances. Bayesian optimization can be used to optimize such methods in an efficient manner by deploying a probabilistic function estimator to estimate with a given confidence so that regions of the search space can be pruned away. Consequently, the success of the Bayesian optimization depends on the function estimator's ability to provide informative confidence bounds. Existing function estimators require many function evaluations to infer the underlying confidence or depend on modeling of the disturbances. In this paper, it is shown that the confidence bounds provided by the Wilson Score Kernel Density Estimator (WS-KDE) are applicable as excellent bounds to any stochastic function with an output confined to the closed interval [0;1] regardless of the distribution of the output. This finding opens up the use of WS-KDE for stable global optimization on a wider range of cost functions. The properties of WS-KDE in the context of Bayesian optimization are demonstrated in simulation and applied to the problem of automated trap design for vibrational part feeders.
>
---
## 更新

#### [replaced 001] Multi-Robot Navigation in Social Mini-Games: Definitions, Taxonomy, and Algorithms
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2508.13459v3](http://arxiv.org/pdf/2508.13459v3)**

> **作者:** Rohan Chandra; Shubham Singh; Wenhao Luo; Katia Sycara
>
> **摘要:** The ``Last Mile Challenge'' has long been considered an important, yet unsolved, challenge for autonomous vehicles, public service robots, and delivery robots. A central issue in this challenge is the ability of robots to navigate constrained and cluttered environments that have high agency (e.g., doorways, hallways, corridor intersections), often while competing for space with other robots and humans. We refer to these environments as ``Social Mini-Games'' (SMGs). Traditional navigation approaches designed for MRN do not perform well in SMGs, which has led to focused research on dedicated SMG solvers. However, publications on SMG navigation research make different assumptions (on centralized versus decentralized, observability, communication, cooperation, etc.), and have different objective functions (safety versus liveness). These assumptions and objectives are sometimes implicitly assumed or described informally. This makes it difficult to establish appropriate baselines for comparison in research papers, as well as making it difficult for practitioners to find the papers relevant to their concrete application. Such ad-hoc representation of the field also presents a barrier to new researchers wanting to start research in this area. SMG navigation research requires its own taxonomy, definitions, and evaluation protocols to guide effective research moving forward. This survey is the first to catalog SMG solvers using a well-defined and unified taxonomy and to classify existing methods accordingly. It also discusses the essential properties of SMG solvers, defines what SMGs are and how they appear in practice, outlines how to evaluate SMG solvers, and highlights the differences between SMG solvers and general navigation systems. The survey concludes with an overview of future directions and open challenges in the field. Our project is open-sourced at https://socialminigames.github.io/.
>
---
#### [replaced 002] Beyond Pairwise Comparisons: Unveiling Structural Landscape of Mobile Robot Models
- **分类: cs.DC; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.19805v3](http://arxiv.org/pdf/2508.19805v3)**

> **作者:** Shota Naito; Tsukasa Ninomiya; Koichi Wada
>
> **摘要:** Understanding the computational power of mobile robot systems is a fundamental challenge in distributed computing. While prior work has focused on pairwise separations between models, we explore how robot capabilities, light observability, and scheduler synchrony interact in more complex ways. We first show that the Exponential Times Expansion (ETE) problem is solvable only in the strongest model -- fully-synchronous robots with full mutual lights ($\mathcal{LUMT}^F$). We then introduce the Hexagonal Edge Traversal (HET) and TAR(d)* problems to demonstrate how internal memory and lights interact with synchrony: under weak synchrony, internal memory alone is insufficient, while full synchrony can substitute for both lights and memory. In the asynchronous setting, we classify problems such as LP-MLCv, VEC, and ZCC to show fine-grained separations between $\mathcal{FSTA}$ and $\mathcal{FCOM}$ robots. We also analyze Vertex Traversal Rendezvous (VTR) and Leave Place Convergence (LP-Cv), illustrating the limitations of internal memory in symmetric settings. These results extend the known separation map of 14 canonical robot models, revealing structural phenomena only visible through higher-order comparisons. Our work provides new impossibility criteria and deepens the understanding of how observability, memory, and synchrony collectively shape the computational power of mobile robots.
>
---
#### [replaced 003] villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.23682v2](http://arxiv.org/pdf/2507.23682v2)**

> **作者:** Xiaoyu Chen; Hangxing Wei; Pushi Zhang; Chuheng Zhang; Kaixin Wang; Yanjiang Guo; Rushuai Yang; Yucen Wang; Xinquan Xiao; Li Zhao; Jianyu Chen; Jiang Bian
>
> **备注:** Project page: https://aka.ms/villa-x
>
> **摘要:** Visual-Language-Action (VLA) models have emerged as a popular paradigm for learning robot manipulation policies that can follow language instructions and generalize to novel scenarios. Recent work has begun to explore the incorporation of latent actions, an abstract representation of visual change between two frames, into VLA pre-training. In this paper, we introduce villa-X, a novel Visual-Language-Latent-Action (ViLLA) framework that advances latent action modeling for learning generalizable robot manipulation policies. Our approach improves both how latent actions are learned and how they are incorporated into VLA pre-training. Together, these contributions enable villa-X to achieve superior performance across simulated environments including SIMPLER and LIBERO, as well as on two real-world robot setups including gripper and dexterous hand manipulation. We believe the ViLLA paradigm holds significant promise, and that our villa-X provides a strong foundation for future research.
>
---
#### [replaced 004] Sampling-Based Multi-Modal Multi-Robot Multi-Goal Path Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03509v2](http://arxiv.org/pdf/2503.03509v2)**

> **作者:** Valentin N. Hartmann; Tirza Heinle; Yijiang Huang; Stelian Coros
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** In many robotics applications, multiple robots are working in a shared workspace to complete a set of tasks as fast as possible. Such settings can be treated as multi-modal multi-robot multi-goal path planning problems, where each robot has to reach a set of goals. Existing approaches to this type of problem solve this using prioritization or assume synchronous task completion, and are thus neither optimal nor complete. We formalize this problem as a single centralized path planning problem and present planners that are probabilistically complete and asymptotically optimal. The planners plan in the composite space of all robots and are modifications of standard sampling-based planners with the required changes to work in our multi-modal, multi-robot, multi-goal setting. We validate the planners on a diverse range of problems including scenarios with various robots, planning horizons, and collaborative tasks such as handovers, and compare the planners against a suboptimal prioritized planner. Videos and code for the planners and the benchmark is available at https://vhartmann.com/mrmg-planning/.
>
---
#### [replaced 005] Learning-Based Modeling of a Magnetically Steerable Soft Suction Device for Endoscopic Endonasal Interventions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15155v2](http://arxiv.org/pdf/2507.15155v2)**

> **作者:** Majid Roshanfar; Alex Zhang; Changyan He; Amir Hooshiar; Dale J. Podolsky; Thomas Looi; Eric Diller
>
> **摘要:** This letter introduces a novel learning-based modeling framework for a magnetically steerable soft suction device designed for endoscopic endonasal brain tumor resection. The device is miniaturized (4 mm outer diameter, 2 mm inner diameter, 40 mm length), 3D printed using biocompatible SIL 30 material, and integrates embedded Fiber Bragg Grating (FBG) sensors for real-time shape feedback. Shape reconstruction is represented using four Bezier control points, enabling a compact and smooth model of the device's deformation. A data-driven model was trained on 5,097 experimental samples covering a range of magnetic field magnitudes (0-14 mT), actuation frequencies (0.2-1.0 Hz), and vertical tip distances (90-100 mm), using both Neural Network (NN) and Random Forest (RF) architectures. The RF model outperformed the NN across all metrics, achieving a mean root mean square error of 0.087 mm in control point prediction and a mean shape reconstruction error of 0.064 mm. Feature importance analysis further revealed that magnetic field components predominantly influence distal control points, while frequency and distance affect the base configuration. This learning-based approach effectively models the complex nonlinear behavior of hyperelastic soft robots under magnetic actuation without relying on simplified physical assumptions. By enabling sub-millimeter shape prediction accuracy and real-time inference, this work represents an advancement toward the intelligent control of magnetically actuated soft robotic tools in minimally invasive neurosurgery.
>
---
#### [replaced 006] SiLVR: Scalable Lidar-Visual Radiance Field Reconstruction with Uncertainty Quantification
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.02657v2](http://arxiv.org/pdf/2502.02657v2)**

> **作者:** Yifu Tao; Maurice Fallon
>
> **备注:** Accepted by T-RO. Webpage: https://dynamic.robots.ox.ac.uk/projects/silvr/
>
> **摘要:** We present a neural radiance field (NeRF) based large-scale reconstruction system that fuses lidar and vision data to generate high-quality reconstructions that are geometrically accurate and capture photorealistic texture. Our system adopts the state-of-the-art NeRF representation to incorporate lidar. Adding lidar data adds strong geometric constraints on the depth and surface normals, which is particularly useful when modelling uniform texture surfaces which contain ambiguous visual reconstruction cues. A key contribution of this work is a novel method to quantify the epistemic uncertainty of the lidar-visual NeRF reconstruction by estimating the spatial variance of each point location in the radiance field given the sensor observations from the cameras and lidar. This provides a principled approach to evaluate the contribution of each sensor modality to the final reconstruction. In this way, reconstructions that are uncertain (due to e.g. uniform visual texture, limited observation viewpoints, or little lidar coverage) can be identified and removed. Our system is integrated with a real-time lidar SLAM system which is used to bootstrap a Structure-from-Motion (SfM) reconstruction procedure. It also helps to properly constrain the overall metric scale which is essential for the lidar depth loss. The refined SLAM trajectory can then be divided into submaps using Spectral Clustering to group sets of co-visible images together. This submapping approach is more suitable for visual reconstruction than distance-based partitioning. Our uncertainty estimation is particularly effective when merging submaps as their boundaries often contain artefacts due to limited observations. We demonstrate the reconstruction system using a multi-camera, lidar sensor suite in experiments involving both robot-mounted and handheld scanning. Our test datasets cover a total area of more than 20,000 square metres.
>
---
#### [replaced 007] Joint Model-based Model-free Diffusion for Planning with Constraints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.08775v2](http://arxiv.org/pdf/2509.08775v2)**

> **作者:** Wonsuhk Jung; Utkarsh A. Mishra; Nadun Ranawaka Arachchige; Yongxin Chen; Danfei Xu; Shreyas Kousik
>
> **备注:** The first two authors contributed equally. Last three authors advised equally. Accepted to CoRL 2025
>
> **摘要:** Model-free diffusion planners have shown great promise for robot motion planning, but practical robotic systems often require combining them with model-based optimization modules to enforce constraints, such as safety. Naively integrating these modules presents compatibility challenges when diffusion's multi-modal outputs behave adversarially to optimization-based modules. To address this, we introduce Joint Model-based Model-free Diffusion (JM2D), a novel generative modeling framework. JM2D formulates module integration as a joint sampling problem to maximize compatibility via an interaction potential, without additional training. Using importance sampling, JM2D guides modules outputs based only on evaluations of the interaction potential, thus handling non-differentiable objectives commonly arising from non-convex optimization modules. We evaluate JM2D via application to aligning diffusion planners with safety modules on offline RL and robot manipulation. JM2D significantly improves task performance compared to conventional safety filters without sacrificing safety. Further, we show that conditional generation is a special case of JM2D and elucidate key design choices by comparing with SOTA gradient-based and projection-based diffusion planners. More details at: https://jm2d-corl25.github.io/.
>
---
#### [replaced 008] LiDAR-BIND-T: Improved and Temporally Consistent Sensor Modality Translation and Fusion for Robotic Applications
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.05728v2](http://arxiv.org/pdf/2509.05728v2)**

> **作者:** Niels Balemans; Ali Anwar; Jan Steckel; Siegfried Mercelis
>
> **摘要:** This paper extends LiDAR-BIND, a modular multi-modal fusion framework that binds heterogeneous sensors (radar, sonar) to a LiDAR-defined latent space, with mechanisms that explicitly enforce temporal consistency. We introduce three contributions: (i) temporal embedding similarity that aligns consecutive latent representations, (ii) a motion-aligned transformation loss that matches displacement between predictions and ground truth LiDAR, and (iii) windowed temporal fusion using a specialised temporal module. We further update the model architecture to better preserve spatial structure. Evaluations on radar/sonar-to-LiDAR translation demonstrate improved temporal and spatial coherence, yielding lower absolute trajectory error and better occupancy map accuracy in Cartographer-based SLAM (Simultaneous Localisation and Mapping). We propose different metrics based on the Fr\'echet Video Motion Distance (FVMD) and a correlation-peak distance metric providing practical temporal quality indicators to evaluate SLAM performance. The proposed temporal LiDAR-BIND, or LiDAR-BIND-T, maintains plug-and-play modality fusion while substantially enhancing temporal stability, resulting in improved robustness and performance for downstream SLAM.
>
---
#### [replaced 009] Robix: A Unified Model for Robot Interaction, Reasoning and Planning
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.01106v2](http://arxiv.org/pdf/2509.01106v2)**

> **作者:** Huang Fang; Mengxi Zhang; Heng Dong; Wei Li; Zixuan Wang; Qifeng Zhang; Xueyun Tian; Yucheng Hu; Hang Li
>
> **备注:** Tech report. Project page: https://robix-seed.github.io/robix/
>
> **摘要:** We introduce Robix, a unified model that integrates robot reasoning, task planning, and natural language interaction within a single vision-language architecture. Acting as the high-level cognitive layer in a hierarchical robot system, Robix dynamically generates atomic commands for the low-level controller and verbal responses for human interaction, enabling robots to follow complex instructions, plan long-horizon tasks, and interact naturally with human within an end-to-end framework. Robix further introduces novel capabilities such as proactive dialogue, real-time interruption handling, and context-aware commonsense reasoning during task execution. At its core, Robix leverages chain-of-thought reasoning and adopts a three-stage training strategy: (1) continued pretraining to enhance foundational embodied reasoning abilities including 3D spatial understanding, visual grounding, and task-centric reasoning; (2) supervised finetuning to model human-robot interaction and task planning as a unified reasoning-action sequence; and (3) reinforcement learning to improve reasoning-action consistency and long-horizon task coherence. Extensive experiments demonstrate that Robix outperforms both open-source and commercial baselines (e.g., GPT-4o and Gemini 2.5 Pro) in interactive task execution, demonstrating strong generalization across diverse instruction types (e.g., open-ended, multi-stage, constrained, invalid, and interrupted) and various user-involved tasks such as table bussing, grocery shopping, and dietary filtering.
>
---
#### [replaced 010] Imagine, Verify, Execute: Memory-guided Agentic Exploration with Vision-Language Models
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.07815v3](http://arxiv.org/pdf/2505.07815v3)**

> **作者:** Seungjae Lee; Daniel Ekpo; Haowen Liu; Furong Huang; Abhinav Shrivastava; Jia-Bin Huang
>
> **备注:** Project webpage: https://ive-robot.github.io/
>
> **摘要:** Exploration is essential for general-purpose robotic learning, especially in open-ended environments where dense rewards, explicit goals, or task-specific supervision are scarce. Vision-language models (VLMs), with their semantic reasoning over objects, spatial relations, and potential outcomes, present a compelling foundation for generating high-level exploratory behaviors. However, their outputs are often ungrounded, making it difficult to determine whether imagined transitions are physically feasible or informative. To bridge the gap between imagination and execution, we present IVE (Imagine, Verify, Execute), an agentic exploration framework inspired by human curiosity. Human exploration is often driven by the desire to discover novel scene configurations and to deepen understanding of the environment. Similarly, IVE leverages VLMs to abstract RGB-D observations into semantic scene graphs, imagine novel scenes, predict their physical plausibility, and generate executable skill sequences through action tools. We evaluate IVE in both simulated and real-world tabletop environments. The results show that IVE enables more diverse and meaningful exploration than RL baselines, as evidenced by a 4.1 to 7.8x increase in the entropy of visited states. Moreover, the collected experience supports downstream learning, producing policies that closely match or exceed the performance of those trained on human-collected demonstrations.
>
---
#### [replaced 011] Single-Stage Optimization of Open-loop Stable Limit Cycles with Smooth, Symbolic Derivatives
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2312.10647v3](http://arxiv.org/pdf/2312.10647v3)**

> **作者:** Muhammad Saud Ul Hassan; Christian Hubicki
>
> **备注:** Accepted at IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** Open-loop stable limit cycles are foundational to legged robotics, providing inherent self-stabilization that minimizes the need for computationally intensive feedback-based gait correction. While previous methods have primarily targeted specific robotic models, this paper introduces a general framework for rapidly generating limit cycles across various dynamical systems, with the flexibility to impose arbitrarily tight stability bounds. We formulate the problem as a single-stage constrained optimization problem and use Direct Collocation to transcribe it into a nonlinear program with closed-form expressions for constraints, objectives, and their gradients. Our method supports multiple stability formulations. In particular, we tested two popular formulations for limit cycle stability in robotics: (1) based on the spectral radius of a discrete return map, and (2) based on the spectral radius of the monodromy matrix, and tested five different constraint-satisfaction formulations of the eigenvalue problem to bound the spectral radius. We compare the performance and solution quality of the various formulations on a robotic swing-leg model, highlighting the Schur decomposition of the monodromy matrix as a method with broader applicability due to weaker assumptions and stronger numerical convergence properties. As a case study, we apply our method on a hopping robot model, generating open-loop stable gaits in under 2 seconds on an Intel Core i7-6700K, while simultaneously minimizing energy consumption even under tight stability constraints.
>
---
#### [replaced 012] RESPLE: Recursive Spline Estimation for LiDAR-Based Odometry
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.11580v3](http://arxiv.org/pdf/2504.11580v3)**

> **作者:** Ziyu Cao; William Talbot; Kailai Li
>
> **摘要:** We present a novel recursive Bayesian estimation framework using B-splines for continuous-time 6-DoF dynamic motion estimation. The state vector consists of a recurrent set of position control points and orientation control point increments, enabling efficient estimation via a modified iterated extended Kalman filter without involving error-state formulations. The resulting recursive spline estimator (RESPLE) is further leveraged to develop a versatile suite of direct LiDAR-based odometry solutions, supporting the integration of one or multiple LiDARs and an IMU. We conduct extensive real-world evaluations using public datasets and our own experiments, covering diverse sensor setups, platforms, and environments. Compared to existing systems, RESPLE achieves comparable or superior estimation accuracy and robustness, while attaining real-time efficiency. Our results and analysis demonstrate RESPLE's strength in handling highly dynamic motions and complex scenes within a lightweight and flexible design, showing strong potential as a universal framework for multi-sensor motion estimation. We release the source code and experimental datasets at https://github.com/ASIG-X/RESPLE .
>
---
#### [replaced 013] No Need to Look! Locating and Grasping Objects by a Robot Arm Covered with Sensitive Skin
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.17986v2](http://arxiv.org/pdf/2508.17986v2)**

> **作者:** Karel Bartunek; Lukas Rustler; Matej Hoffmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Locating and grasping of objects by robots is typically performed using visual sensors. Haptic feedback from contacts with the environment is only secondary if present at all. In this work, we explored an extreme case of searching for and grasping objects in complete absence of visual input, relying on haptic feedback only. The main novelty lies in the use of contacts over the complete surface of a robot manipulator covered with sensitive skin. The search is divided into two phases: (1) coarse workspace exploration with the complete robot surface, followed by (2) precise localization using the end-effector equipped with a force/torque sensor. We systematically evaluated this method in simulation and on the real robot, demonstrating that diverse objects can be located, grasped, and put in a basket. The overall success rate on the real robot for one object was 85.7% with failures mainly while grasping specific objects. The method using whole-body contacts is six times faster compared to a baseline that uses haptic feedback only on the end-effector. We also show locating and grasping multiple objects on the table. This method is not restricted to our specific setup and can be deployed on any platform with the ability of sensing contacts over the entire body surface. This work holds promise for diverse applications in areas with challenging visual perception (due to lighting, dust, smoke, occlusion) such as in agriculture when fruits or vegetables need to be located inside foliage and picked.
>
---
#### [replaced 014] Shaken, Not Stirred: A Novel Dataset for Visual Understanding of Glasses in Human-Robot Bartending Tasks
- **分类: cs.RO; cs.CV; 68T40; I.2.9; I.4.8**

- **链接: [http://arxiv.org/pdf/2503.04308v3](http://arxiv.org/pdf/2503.04308v3)**

> **作者:** Lukáš Gajdošech; Hassan Ali; Jan-Gerrit Habekost; Martin Madaras; Matthias Kerzel; Stefan Wermter
>
> **备注:** Submitted and Accepted for Presentation at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Datasets for object detection often do not account for enough variety of glasses, due to their transparent and reflective properties. Specifically, open-vocabulary object detectors, widely used in embodied robotic agents, fail to distinguish subclasses of glasses. This scientific gap poses an issue for robotic applications that suffer from accumulating errors between detection, planning, and action execution. This paper introduces a novel method for acquiring real-world data from RGB-D sensors that minimizes human effort. We propose an auto-labeling pipeline that generates labels for all the acquired frames based on the depth measurements. We provide a novel real-world glass object dataset GlassNICOLDataset that was collected on the Neuro-Inspired COLlaborator (NICOL), a humanoid robot platform. The dataset consists of 7850 images recorded from five different cameras. We show that our trained baseline model outperforms state-of-the-art open-vocabulary approaches. In addition, we deploy our baseline model in an embodied agent approach to the NICOL platform, on which it achieves a success rate of 81% in a human-robot bartending scenario.
>
---
#### [replaced 015] Extended Neural Contractive Dynamical Systems: On Multiple Tasks and Riemannian Safety Regions
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.11405v3](http://arxiv.org/pdf/2411.11405v3)**

> **作者:** Hadi Beik Mohammadi; Søren Hauberg; Georgios Arvanitidis; Gerhard Neumann; Leonel Rozo
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2401.09352
>
> **摘要:** Stability guarantees are crucial when ensuring that a fully autonomous robot does not take undesirable or potentially harmful actions. We recently proposed the Neural Contractive Dynamical Systems (NCDS), which is a neural network architecture that guarantees contractive stability. With this, learning-from-demonstrations approaches can trivially provide stability guarantees. However, our early work left several unanswered questions, which we here address. Beyond providing an in-depth explanation of NCDS, this paper extends the framework with more careful regularization, a conditional variant of the framework for handling multiple tasks, and an uncertainty-driven approach to latent obstacle avoidance. Experiments verify that the developed system has the flexibility of ordinary neural networks while providing the stability guarantees needed for autonomous robotics.
>
---
#### [replaced 016] Symmetry-Guided Multi-Agent Inverse Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.08257v2](http://arxiv.org/pdf/2509.08257v2)**

> **作者:** Yongkai Tian; Yirong Qi; Xin Yu; Wenjun Wu; Jie Luo
>
> **备注:** 8pages, 6 figures. Accepted for publication in the Proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025) as oral presentation
>
> **摘要:** In robotic systems, the performance of reinforcement learning depends on the rationality of predefined reward functions. However, manually designed reward functions often lead to policy failures due to inaccuracies. Inverse Reinforcement Learning (IRL) addresses this problem by inferring implicit reward functions from expert demonstrations. Nevertheless, existing methods rely heavily on large amounts of expert demonstrations to accurately recover the reward function. The high cost of collecting expert demonstrations in robotic applications, particularly in multi-robot systems, severely hinders the practical deployment of IRL. Consequently, improving sample efficiency has emerged as a critical challenge in multi-agent inverse reinforcement learning (MIRL). Inspired by the symmetry inherent in multi-agent systems, this work theoretically demonstrates that leveraging symmetry enables the recovery of more accurate reward functions. Building upon this insight, we propose a universal framework that integrates symmetry into existing multi-agent adversarial IRL algorithms, thereby significantly enhancing sample efficiency. Experimental results from multiple challenging tasks have demonstrated the effectiveness of this framework. Further validation in physical multi-robot systems has shown the practicality of our method.
>
---
#### [replaced 017] The Oxford Spires Dataset: Benchmarking Large-Scale LiDAR-Visual Localisation, Reconstruction and Radiance Field Methods
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.10546v2](http://arxiv.org/pdf/2411.10546v2)**

> **作者:** Yifu Tao; Miguel Ángel Muñoz-Bañón; Lintong Zhang; Jiahao Wang; Lanke Frank Tarimo Fu; Maurice Fallon
>
> **备注:** Accepted by IJRR. Website: https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/
>
> **摘要:** This paper introduces a large-scale multi-modal dataset captured in and around well-known landmarks in Oxford using a custom-built multi-sensor perception unit as well as a millimetre-accurate map from a Terrestrial LiDAR Scanner (TLS). The perception unit includes three synchronised global shutter colour cameras, an automotive 3D LiDAR scanner, and an inertial sensor - all precisely calibrated. We also establish benchmarks for tasks involving localisation, reconstruction, and novel-view synthesis, which enable the evaluation of Simultaneous Localisation and Mapping (SLAM) methods, Structure-from-Motion (SfM) and Multi-view Stereo (MVS) methods as well as radiance field methods such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting. To evaluate 3D reconstruction the TLS 3D models are used as ground truth. Localisation ground truth is computed by registering the mobile LiDAR scans to the TLS 3D models. Radiance field methods are evaluated not only with poses sampled from the input trajectory, but also from viewpoints that are from trajectories which are distant from the training poses. Our evaluation demonstrates a key limitation of state-of-the-art radiance field methods: we show that they tend to overfit to the training poses/images and do not generalise well to out-of-sequence poses. They also underperform in 3D reconstruction compared to MVS systems using the same visual inputs. Our dataset and benchmarks are intended to facilitate better integration of radiance field methods and SLAM systems. The raw and processed data, along with software for parsing and evaluation, can be accessed at https://dynamic.robots.ox.ac.uk/datasets/oxford-spires/.
>
---
#### [replaced 018] GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14456v4](http://arxiv.org/pdf/2507.14456v4)**

> **作者:** Chi Wan; Yixin Cui; Jiatong Du; Shuo Yang; Yulong Bai; Peng Yi; Nan Li; Yanjun Huang
>
> **摘要:** End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert and a Scene-Adaptive Experts Group, equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves both adaptability and robustness across diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. The code is available at https://github.com/newbrains1/GEMINUS.
>
---
#### [replaced 019] LLMs for sensory-motor control: Combining in-context and iterative learning
- **分类: cs.AI; cs.HC; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.04867v2](http://arxiv.org/pdf/2506.04867v2)**

> **作者:** Jônata Tyska Carvalho; Stefano Nolfi
>
> **备注:** Article updated with results from gpt-oss:120b. 24 pages (13 pages are from appendix), 6 figures, code for experiments replication and supplementary material provided at https://github.com/jtyska/llm-robotics-article/
>
> **摘要:** We propose a method that enables large language models (LLMs) to control embodied agents by directly mapping continuous observation vectors to continuous action vectors. At the outset, the LLMs generate a control strategy based on a textual description of the agent, its environment, and the intended goal. This strategy is then iteratively refined through a learning process in which the LLMs are repeatedly prompted to improve the current strategy, using performance feedback and sensory-motor data collected during its evaluation. The method is validated on classic control tasks from the Gymnasium library and the inverted pendulum task from the MuJoCo library. The approach proves effective with relatively compact models such as Gpt-oss:120b and Qwen2.5:72b. In most cases, it successfully identifies optimal or near-optimal solutions by integrating symbolic knowledge derived through reasoning with sub-symbolic sensory-motor data gathered as the agent interacts with its environment.
>
---
#### [replaced 020] V-HOP: Visuo-Haptic 6D Object Pose Tracking
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.17434v2](http://arxiv.org/pdf/2502.17434v2)**

> **作者:** Hongyu Li; Mingxi Jia; Tuluhan Akbulut; Yu Xiang; George Konidaris; Srinath Sridhar
>
> **备注:** Accepted by RSS 2025
>
> **摘要:** Humans naturally integrate vision and haptics for robust object perception during manipulation. The loss of either modality significantly degrades performance. Inspired by this multisensory integration, prior object pose estimation research has attempted to combine visual and haptic/tactile feedback. Although these works demonstrate improvements in controlled environments or synthetic datasets, they often underperform vision-only approaches in real-world settings due to poor generalization across diverse grippers, sensor layouts, or sim-to-real environments. Furthermore, they typically estimate the object pose for each frame independently, resulting in less coherent tracking over sequences in real-world deployments. To address these limitations, we introduce a novel unified haptic representation that effectively handles multiple gripper embodiments. Building on this representation, we introduce a new visuo-haptic transformer-based object pose tracker that seamlessly integrates visual and haptic input. We validate our framework in our dataset and the Feelsight dataset, demonstrating significant performance improvement on challenging sequences. Notably, our method achieves superior generalization and robustness across novel embodiments, objects, and sensor types (both taxel-based and vision-based tactile sensors). In real-world experiments, we demonstrate that our approach outperforms state-of-the-art visual trackers by a large margin. We further show that we can achieve precise manipulation tasks by incorporating our real-time object tracking result into motion plans, underscoring the advantages of visuo-haptic perception. Project website: https://ivl.cs.brown.edu/research/v-hop
>
---
#### [replaced 021] Diffusion Graph Neural Networks for Robustness in Olfaction Sensors and Datasets
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00455v3](http://arxiv.org/pdf/2506.00455v3)**

> **作者:** Kordel K. France; Ovidiu Daescu
>
> **摘要:** Robotic odour source localization (OSL) is a critical capability for autonomous systems operating in complex environments. However, current OSL methods often suffer from ambiguities, particularly when robots misattribute odours to incorrect objects due to limitations in olfactory datasets and sensor resolutions. To address this challenge, we introduce a novel machine learning method using diffusion-based molecular generation to enhance odour localization accuracy that can be used by itself or with automated olfactory dataset construction pipelines. This generative process of our diffusion model expands the chemical space beyond the limitations of both current olfactory datasets and training methods, enabling the identification of potential odourant molecules not previously documented. The generated molecules can then be more accurately validated using advanced olfactory sensors, enabling them to detect more compounds and inform better hardware design. By integrating visual analysis, language processing, and molecular generation, our framework enhances the ability of olfaction-vision models on robots to accurately associate odours with their correct sources, thereby improving navigation and decision-making through better sensor selection for a target compound in critical applications such as explosives detection, narcotics screening, and search and rescue. Our methodology represents a foundational advancement in the field of artificial olfaction, offering a scalable solution to challenges posed by limited olfactory data and sensor ambiguities. Code and data are made available to the community at the following URL: https://github.com/KordelFranceTech/OlfactionVisionLanguage-Dataset.
>
---
#### [replaced 022] 3D and 4D World Modeling: A Survey
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.07996v2](http://arxiv.org/pdf/2509.07996v2)**

> **作者:** Lingdong Kong; Wesley Yang; Jianbiao Mei; Youquan Liu; Ao Liang; Dekai Zhu; Dongyue Lu; Wei Yin; Xiaotao Hu; Mingkai Jia; Junyuan Deng; Kaiwen Zhang; Yang Wu; Tianyi Yan; Shenyuan Gao; Song Wang; Linfeng Li; Liang Pan; Yong Liu; Jianke Zhu; Wei Tsang Ooi; Steven C. H. Hoi; Ziwei Liu
>
> **备注:** Survey; 34 pages, 10 figures, 14 tables; GitHub Repo at https://github.com/worldbench/survey
>
> **摘要:** World modeling has become a cornerstone in AI research, enabling agents to understand, represent, and predict the dynamic environments they inhabit. While prior work largely emphasizes generative methods for 2D image and video data, they overlook the rapidly growing body of work that leverages native 3D and 4D representations such as RGB-D imagery, occupancy grids, and LiDAR point clouds for large-scale scene modeling. At the same time, the absence of a standardized definition and taxonomy for ``world models'' has led to fragmented and sometimes inconsistent claims in the literature. This survey addresses these gaps by presenting the first comprehensive review explicitly dedicated to 3D and 4D world modeling and generation. We establish precise definitions, introduce a structured taxonomy spanning video-based (VideoGen), occupancy-based (OccGen), and LiDAR-based (LiDARGen) approaches, and systematically summarize datasets and evaluation metrics tailored to 3D/4D settings. We further discuss practical applications, identify open challenges, and highlight promising research directions, aiming to provide a coherent and foundational reference for advancing the field. A systematic summary of existing literature is available at https://github.com/worldbench/survey
>
---
