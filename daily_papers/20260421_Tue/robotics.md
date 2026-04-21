# 机器人 cs.RO

- **最新发布 82 篇**

- **更新 45 篇**

## 最新发布

#### [new 001] Interdisciplinary Workshop on Mechanical Intelligence: Summary Report
- **分类: cs.RO**

- **简介: 该论文属于综述任务，旨在总结机械智能研讨会的成果，探讨机械系统自身如何通过结构实现智能，解决传统计算智能以外的新型智能形式问题。**

- **链接: [https://arxiv.org/pdf/2604.16381](https://arxiv.org/pdf/2604.16381)**

> **作者:** Victoria A. Webster-Wood; Nicholas Gravish; Amir Alavi; Andres F Arrieta; Sarah Bergbreiter; Anthony Bloch; Laura Blumenschein; C. Chase Cao; Aja Mia Carter; Paolo Celli; Tony Chen; Margaret Coad; Mark Cutkosky; Michael Dickey; Brian Do; Robert Full; Mahdi Haghshenas-Jaryani; Kaushik Jayaram; Aaron Johnson; Eva Kanso; Emma Lejeune; Chen Li; Suyi Li; Jeffrey Lipton; Rob MacCurdy; Matt McHenry; Jean-Michel Mongeau; Todd Murphey; Mark Plecnik; Jordan Raney; Ryan D. Sochol; Hannah Stuart; Zeynep Temel; Michael Tolley; Barry Trimmer; T.J. Wallin; Kon-Well Wang; Wenzhong Yan; Mark Yim; Wenlong Zhang
>
> **摘要:** This report provides a summary of the outcomes of the Interdisciplinary Workshop on Mechanical Intelligence held in 2024. Mechanical Intelligence (MI) represents the phenomenon that novel structural features of material/biological/robotic systems can encode intelligence through responsiveness, adaptivity, memory, and learning in the mechanical structure itself. This is in contrast to computational intelligence, wherein the intelligence functions occur through electrical signaling and computer code. The two-day workshop was held at NSF headquarters on May 30-31 and included 38 invited academic researcher participants, and 8 program officers from the NSF. The workshop was structured around active small and large group discussions in groups of 4-5 and 9-10 with the goal of addressing topical questions on MI. Working groups entered notes into shared presentation slides for each discussion session and presented their outcomes in a final presentation on the last day. Here we summarize the overall outcomes of the workshop.
>
---
#### [new 002] COFFAIL: A Dataset of Successful and Anomalous Robot Skill Executions in the Context of Coffee Preparation
- **分类: cs.RO**

- **简介: 该论文介绍COFFAIL数据集，用于机器人技能执行研究，包含成功与异常案例。属于机器人学习任务，旨在提升技能泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.18236](https://arxiv.org/pdf/2604.18236)**

> **作者:** Alex Mitrevski; Ayush Salunke
>
> **备注:** Presented as an extended abstract at the 2nd German Robotics Conference (GRC)
>
> **摘要:** In the context of robot learning for manipulation, curated datasets are an important resource for advancing the state of the art; however, available datasets typically only include successful executions or are focused on one particular type of skill. In this short paper, we briefly describe a dataset of various skills performed in the context of coffee preparation. The dataset, which we call COFFAIL, includes both successful and anomalous skill execution episodes collected with a physical robot in a kitchen environment, a couple of which are performed with bimanual manipulation. In addition to describing the data collection setup and the collected data, the paper illustrates the use of the data in COFFAIL to learn a robot policy using imitation learning.
>
---
#### [new 003] Unmasking the Illusion of Embodied Reasoning in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型在真实环境中的推理能力问题。研究提出BeTTER基准，揭示现有模型在动态场景中的失败原因，指出其架构缺陷导致语义表征崩溃。**

- **链接: [https://arxiv.org/pdf/2604.18000](https://arxiv.org/pdf/2604.18000)**

> **作者:** Haiweng Xu; Sipeng Zheng; Hao Luo; Wanpeng Zhang; Ziheng Xi; Zongqing Lu
>
> **摘要:** Recent Vision-Language-Action (VLA) models report impressive success rates on standard robotic benchmarks, fueling optimism about general-purpose physical intelligence. However, recent evidence suggests a systematic misalignment between standard benchmark success and true embodied reasoning, raising the question of whether these high scores reflect genuine cognitive capability. To address this gap, we introduce BeTTER, a diagnostic Benchmark for Testing True Embodied Reasoning in robotic policies. BeTTER applies targeted causal interventions (e.g., spatial layout shifts, temporal extrapolation) while enforcing kinematic isolation to explicitly decouple high-level reasoning failures from low-level execution limits. Through systematic evaluation, we reveal that state-of-the-art VLAs catastrophically fail in dynamic scenarios, exhibiting severe lexical-kinematic shortcuts, behavioral inertia, and semantic feature collapse. Crucially, our mechanistic analysis traces these symptoms to fundamental architectural bottlenecks - such as capacity compression and myopic downsampling - which systematically degrade the model's foundational semantic representation. We demonstrate that highly static evaluation protocols effectively mask this degradation by allowing optimization to overfit to sensorimotor priors. Supported by real-world robotic validation, our findings confirm that this representational breakdown is not a simulation artifact, highlighting the critical need for future VLA paradigms to resolve the structural tension between high-frequency control and high-level reasoning.
>
---
#### [new 004] Rewind-IL: Online Failure Detection and State Respawning for Imitation Learning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人模仿学习任务，解决长期任务中执行失败无法恢复的问题。提出Rewind-IL框架，通过在线检测和状态重置提升可靠性。**

- **链接: [https://arxiv.org/pdf/2604.16683](https://arxiv.org/pdf/2604.16683)**

> **作者:** Gehan Zheng; Sanjay Seenivasan; Matthew Johnson-Roberson; Weiming Zhi
>
> **备注:** 9 pages, 8 figures, 6 tables. Project page at this https URL
>
> **摘要:** Imitation learning has enabled robots to acquire complex visuomotor manipulation skills from demonstrations, but deployment failures remain a major obstacle, especially for long-horizon action-chunked policies. Once execution drifts off the demonstration manifold, these policies often continue producing locally plausible actions without recovering from the failure. Existing runtime monitors either require failure data, over-trigger under benign feature drift, or stop at failure detection without providing a recovery mechanism. We present Rewind-IL, a training-free online safeguard framework for generative action-chunked imitation policies. Rewind-IL combines a zero-shot failure detector based on Temporal Inter-chunk Discrepancy Estimate (TIDE), calibrated with split conformal prediction, with a state-respawning mechanism that returns the robot to a semantically verified safe intermediate state. Offline, a vision-language model identifies recovery checkpoints in demonstrations, and the frozen policy encoder is used to construct a compact checkpoint feature database. Online, Rewind-IL monitors self-consistency in overlapping action chunks, tracks similarity to the checkpoint library, and, upon failure, rewinds execution to the latest verified safe state before restarting inference from a clean policy state. Experiments on real-world and simulated long-horizon manipulation tasks, including transfer to flow-matching action-chunked policies, demonstrate that policy-internal consistency coupled with semantically grounded respawning offers a practical route to improved reliability in imitation learning. Supplemental materials are available at this https URL
>
---
#### [new 005] AnchorRefine: Synergy-Manipulation Based on Trajectory Anchor and Residual Refinement for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作模型任务，解决动作生成中全局与局部精度不协调的问题。提出AnchorRefine框架，分离轨迹锚点与残差修正，提升操作精度。**

- **链接: [https://arxiv.org/pdf/2604.17787](https://arxiv.org/pdf/2604.17787)**

> **作者:** Tingzheng Jia; Kan Guo; Lanping Qian; Yongli Hu; Daxin Tian; Guixian Qu; Chunmian Lin; Baocai Yin; Jiapu Wang
>
> **摘要:** Precision-critical manipulation requires both global trajectory organization and local execution correction, yet most vision-language-action (VLA) policies generate actions within a single unified space. This monolithic formulation forces macro-level transport and micro-level refinement to be optimized under the same objective, causing large motions to dominate learning while suppressing small but failure-critical corrective signals. In contrast, human manipulation is structured by global movement planning together with continuous local adjustment during execution. Motivated by this principle, we propose AnchorRefine, a hierarchical framework that factorizes VLA action modeling into trajectory anchor and residual refinement. The anchor planner predicts a coarse motion scaffold, while the refinement module corrects execution-level deviations to improve geometric and contact precision. We further introduce a decision-aware gripper refinement mechanism to better capture the discrete and boundary-sensitive nature of gripper control. Experiments on LIBERO, CALVIN, and real-robot tasks demonstrate that AnchorRefine consistently improves both regression-based and diffusion-based VLA backbones, yielding gains of up to 7.8% in simulation success rate and 18% in real-world success rate.
>
---
#### [new 006] DAG-STL: A Hierarchical Framework for Zero-Shot Trajectory Planning under Signal Temporal Logic Specifications
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出DAG-STL框架，解决未知动态下基于STL的轨迹规划问题。通过分解、分配、生成三阶段方法，提升长时序任务的零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.18343](https://arxiv.org/pdf/2604.18343)**

> **作者:** Ruijia Liu; Ancheng Hou; Xiao Yu; Xiang Yin
>
> **摘要:** Signal Temporal Logic (STL) is a powerful language for specifying temporally structured robotic tasks. Planning executable trajectories under STL constraints remains difficult when system dynamics and environment structure are not analytically available. Existing methods typically either assume explicit models or learn task-specific behaviors, limiting zero-shot generalization to unseen STL tasks. In this work, we study offline STL planning under unknown dynamics using only task-agnostic trajectory data. Our central design philosophy is to separate logical reasoning from trajectory realization. We instantiate this idea in DAG-STL, a hierarchical framework that converts long-horizon STL planning into three stages. It first decomposes an STL formula into reachability and invariance progress conditions linked by shared timing constraints. It then allocates timed waypoints using learned reachability-time estimates. Finally, it synthesizes trajectories between these waypoints with a diffusion-based generator. This decomposition--allocation--generation pipeline reduces global planning to shorter, better-supported subproblems. To bridge the gap between planning-level correctness and execution-level feasibility, we further introduce a rollout-free dynamic consistency metric, an anytime refinement search procedure for improving multiple allocation hypotheses under finite budgets, and a hierarchical online replanning mechanism for execution-time recovery. Experiments in Maze2D, OGBench AntMaze, and the Cube domain show that DAG-STL substantially outperforms direct robustness-guided diffusion on complex long-horizon STL tasks and generalizes across navigation and manipulation settings. In a custom environment with an optimization-based reference, DAG-STL recovers most model-solvable tasks while retaining a clear computational advantage over direct optimization based on the explicit system model.
>
---
#### [new 007] A Rapid Deployment Pipeline for Autonomous Humanoid Grasping Based on Foundation Models
- **分类: cs.RO**

- **简介: 该论文属于人形机器人抓取任务，旨在缩短新物体部署时间。通过集成基础模型，实现快速标注、3D重建和六自由度姿态跟踪，将部署时间从一天缩短至30分钟。**

- **链接: [https://arxiv.org/pdf/2604.17258](https://arxiv.org/pdf/2604.17258)**

> **作者:** Yifei Yan; Yankai Liao; Linqi Ye
>
> **摘要:** Deploying a humanoid robot to manipulate a new object has traditionally required one to two days of effort: data collection, manual annotation, 3D model acquisition, and model training. This paper presents an end-to-end rapid deployment pipeline that integrates three foundation-model components to shorten the onboarding cycle for a new object to approximately 30 minutes: (i) Roboflow-based automatic annotation to assist in training a YOLOv8 object detector; (ii) 3D reconstruction based on Meta SAM 3D, which eliminates the need for a dedicated laser scanner; and (iii) zero-shot 6-DoF pose tracking based on FoundationPose, using the SAM~3D-generated mesh directly as the template. The estimated pose drives a Unity-based inverse kinematics planner, whose joint commands are streamed via UDP to a Unitree~G1 humanoid and executed through the Unitree SDK. We demonstrate detection accuracy of mAP@0.5 = 0.995, pose tracking precision of $\sigma < 1.05$ mm, and successful grasping on a real robot at five positions within the workspace. We further verify the generality of the pipeline on an automobile-window glue-application task. The results show that combining foundation models for perception with everyday imaging devices (e.g., smartphones) can substantially lower the deployment barrier for humanoid manipulation tasks.
>
---
#### [new 008] Compiling OpenSCENARIO 2.1 for Scenario-Based Testing in CARLA
- **分类: cs.RO; cs.PL; eess.SY**

- **简介: 该论文属于场景测试任务，旨在解决OpenSCENARIO 2.1在CARLA中的集成问题。通过设计编译器架构，将DSL转换为CARLA可执行行为，实现高效、可重复的场景测试。**

- **链接: [https://arxiv.org/pdf/2604.16452](https://arxiv.org/pdf/2604.16452)**

> **作者:** Thoshitha Gamage; Lasanthi Gamage
>
> **摘要:** While the ASAM OpenSCENARIO 2.1 Domain-Specific Language (DSL) enables declarative, intent-driven authoring for Scenario-Based Testing (SBT), its integration into open-source simulators like CARLA remains limited by legacy parsers. We propose a multi-pass modern compiler architecture that translates the OpenSCENARIO 2.1 DSL directly into executable CARLA behaviors. The pipeline features an ANTLR4 frontend for Abstract Syntax Tree (AST) generation, a semantic middle-end, and a runtime backend that synthesizes deterministic py_trees behavior trees. Mapping the standardized domain ontology directly to CARLA's procedural API via a custom method registry eliminates the need for external logic solvers. A demonstrative multi-actor cut-in and evasive maneuver, selected from a wider suite of validated scenarios, confirms the compiler's ability to process concurrent actions, dynamic mathematical expressions, and asynchronous signaling. This framework establishes a functional baseline for reproducible, large-scale SBT, paving the way for future C++ optimizations to mitigate current Python-based computational overhead.
>
---
#### [new 009] LiDAR-based Crowd Navigation with Visible Edge Group Representation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决拥挤环境中安全社交导航问题。通过提出可见边缘群体表示方法，提升导航效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.16741](https://arxiv.org/pdf/2604.16741)**

> **作者:** Allan Wang; Aaron Steinfeld
>
> **备注:** Under review
>
> **摘要:** Robot navigation in crowded pedestrian environments is a well-known challenge and we explore the practical deployment of group-based representations in this setting. Pedestrian groups have been empirically shown to enable a mobile robot's navigation behavior to be safer and more social. However, existing approaches either explored groups only in limited scenarios with no high-density crowds or depended on external detection modules to track individuals, which are prone to noise and errors due to occlusions in crowds. We show that group prediction accuracy affects navigation performance only marginally in crowded environments. Based on this observation, we propose the visible edge-based group representation. We additionally demonstrate via simulation experiments that our navigation framework, integrated with the simplified group representation, performs comparatively in terms of safety and socialness in dense crowds, while achieving faster computation speed. Finally, we deploy our navigation framework on a real robot to explore the benefits of practically deploying group-based representations in the real world.
>
---
#### [new 010] Leveraging VR Robot Games to Facilitate Data Collection for Embodied Intelligence Tasks
- **分类: cs.RO**

- **简介: 该论文属于 embodied intelligence 领域，旨在解决数据收集成本高、难度大的问题。通过VR游戏框架，实现高效的数据采集与任务评估。**

- **链接: [https://arxiv.org/pdf/2604.16903](https://arxiv.org/pdf/2604.16903)**

> **作者:** Yihan Zhang; Ziyun Huang; Linqi Ye
>
> **摘要:** Collecting embodied interaction data at scale remains costly and difficult due to the limited accessibility of conventional interfaces. We present a gamified data collection framework based on Unity that combines procedural scene generation, VR-based humanoid robot control, automatic task evaluation, and trajectory logging. A trash pick-and-place task prototype is developed to validate the full this http URL results indicate that the collected demonstrations exhibit broad coverage of the state-action space, and that increasing task difficulty leads to higher motion intensity as well as more extensive exploration of the arm's workspace. The proposed framework demonstrates that game-oriented virtual environments can serve as an effective and extensible solution for embodied data collection.
>
---
#### [new 011] Chain Of Interaction Benchmark (COIN): When Reasoning meets Embodied Interaction
- **分类: cs.RO**

- **简介: 该论文提出COIN基准，用于评估机器人在复杂任务中的交互推理能力。针对现有基准不足，构建了多个任务集并开发评估指标，揭示当前方法在视觉与执行间的差距。**

- **链接: [https://arxiv.org/pdf/2604.16886](https://arxiv.org/pdf/2604.16886)**

> **作者:** Xianhao Wang; Xiaojian Ma; Haozhe Hu; Rongpeng Su; Yutian Cheng; Zhou Ziheng; Hangxin Liu; Lei Liu; Bin Li; Qing Li
>
> **摘要:** Generalist embodied agents must perform interactive, causally-dependent reasoning, continually interacting with the environment, acquiring information, and updating plans to solve long-horizon tasks before they could be adopted in real-life scenarios. For instance, retrieving an apple from a cabinet may require opening multiple doors and drawers before the apple becomes visible and reachable, demanding sequential interaction under partial observability. However, existing benchmarks fail to systematically evaluate this essential capability. We introduce COIN, a benchmark designed to assess interactive reasoning in realistic robotic manipulation through three key contributions. First, we construct COIN-50: 50 interactive tasks in daily scenarios, and create COIN-Primitive required by causally-dependent tasks, and COIN-Composition with mid-term complexity for skill learning and generalization evaluation. Second, we develop a low-cost mobile AR teleoperation system and collect the COIN-Primitive Dataset with 50 demonstrations per primitive task (1,000 in total). Third, we develop systematic evaluation metrics about execution stability and generalization robustness to evaluate CodeAsPolicy, VLA, and language-conditioned H-VLA approaches. Our comprehensive evaluation reveals critical limitations in current methods: models struggle with interactive reasoning tasks due to significant gaps between visual understanding and motor execution. We provide fine-grained analysis of these limitations.
>
---
#### [new 012] Chatting about Conditional Trajectory Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于轨迹预测任务，解决机器人交互中轨迹预测不准确的问题。提出CiT方法，结合自身运动与社会交互信息，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2604.18126](https://arxiv.org/pdf/2604.18126)**

> **作者:** Yuxiang Zhao; Wei Huang; Haipeng Zeng; Huan Zhao; Yujie Song
>
> **摘要:** Human behavior has the nature of mutual dependencies, which requires human-robot interactive systems to predict surrounding agents trajectories by modeling complex social interactions, avoiding collisions and executing safe path planning. While there exist many trajectory prediction methods, most of them do not incorporate the own motion of the ego agent and only model interactions based on static information. We are inspired by the humans theory of mind during trajectory selection and propose a Cross time domain intention-interactive method for conditional Trajectory prediction(CiT). Our proposed CiT conducts joint analysis of behavior intentions over time, and achieves information complementarity and integration across different time domains. The intention in its own time domain can be corrected by the social interaction information from the other time domain to obtain a more precise intention representation. In addition, CiT is designed to closely integrate with robotic motion planning and control modules, capable of generating a set of optional trajectory prediction results for all surrounding agents based on potential motions of the ego agent. Extensive experiments demonstrate that the proposed CiT significantly outperforms the existing methods, achieving state-of-the-art performance in the benchmarks.
>
---
#### [new 013] LatentMimic: Terrain-Adaptive Locomotion via Latent Space Imitation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，旨在解决四足机器人在复杂地形中保持运动风格的同时实现自适应行走。工作包括提出LatentMimic框架，分离风格与几何约束，并引入地形适应模块提升性能。**

- **链接: [https://arxiv.org/pdf/2604.16440](https://arxiv.org/pdf/2604.16440)**

> **作者:** Zhiquan Wang; Yunyu Liu; Dipam Patel; Ayush Kumar; Aniket Bera; Bedrich Benes
>
> **摘要:** Developing natural and diverse locomotion controllers for quadruped robots that can adapt to complex terrains while preserving motion style remains a significant challenge. Existing imitation-based methods face a fundamental optimization trade-off: strict adherence to motion capture (mocap) references penalizes the geometric deviations required for terrain adaptability, whereas terrain-centric policies often compromise stylistic fidelity. We introduce LatentMimic, a novel locomotion learning framework that decouples stylistic fidelity from geometric constraints. By minimizing the marginal latent divergence between the policy's state-action distribution and a learned mocap prior, our approach provides a conditional relaxation of rigid pose-tracking objectives. This formulation preserves gait topology while permitting independent end-effector adaptations for irregular terrains. We further introduce a terrain adaptation module with a dynamic replay buffer to resolve the policy's distribution shifts across different terrains. We validate our method across four locomotion styles and four terrains, demonstrating that LatentMimic enables effective terrain-adaptive locomotion, achieving higher terrain traversal success rates than state-of-the-art motion-tracking methods while maintaining high stylistic fidelity.
>
---
#### [new 014] Planning Smooth and Safe Control Laws for a Unicycle Robot Among Obstacles
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决在障碍物环境中安全导航问题。提出一种QP框架生成平滑向量场，并设计非线性控制器确保安全收敛。**

- **链接: [https://arxiv.org/pdf/2604.17212](https://arxiv.org/pdf/2604.17212)**

> **作者:** Aref Amiri; Basak Sakcak; Steven M. LaValle
>
> **备注:** This work has been accepted for publication in the 2026 European Control Conference (ECC)
>
> **摘要:** This paper presents a framework for safe navigation of a unicycle point robot to a goal position in an environment populated with obstacles from almost any admissible state, considering input limits. We introduce a novel QP formulation to create a Cinfinity-smooth vector field with reduced total bending and total turning. Then we design an analytic, non-linear feedback controller that inherently satisfies the conditions of Nagumo's theorem, ensuring forward invariance of the safe set without requiring any online optimization. We have demonstrated that our controller, even under hard input limits, safely converges to the goal position. Simulations confirm the effectiveness of the proposed framework, resulting in a twice faster arrival time with over 50\% lower angular control effort compared to the baseline.
>
---
#### [new 015] ReconVLA: An Uncertainty-Guided and Failure-Aware Vision-Language-Action Framework for Robotic Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ReconVLA，解决机器人控制中不确定性与故障预测问题。通过校准置信度和检测异常状态，提升控制可靠性。**

- **链接: [https://arxiv.org/pdf/2604.16677](https://arxiv.org/pdf/2604.16677)**

> **作者:** Lingling Chen; Zongyao Lyu; William J. Beksi
>
> **备注:** 17 pages, 9 figures, and 7 tables
>
> **摘要:** Vision-language-action (VLA) models have emerged as generalist robotic controllers capable of mapping visual observations and natural language instructions to continuous action sequences. However, VLAs provide no calibrated measure of confidence in their action predictions, thus limiting their reliability in real-world settings where uncertainty and failures must be anticipated. To address this problem we introduce ReconVLA, a reliable conformal model that produces uncertainty-guided and failure-aware control signals. Concretely, our approach applies conformal prediction directly to the action token outputs of pretrained VLA policies, yielding calibrated uncertainty estimates that correlate with execution quality and task success. Furthermore, we extend conformal prediction to the robot state space to detect outliers or unsafe states before failures occur, providing a simple yet effective failure detection mechanism that complements the action-level uncertainty. We evaluate ReconVLA in both simulation and real robot experiments across diverse manipulation tasks. Our results show that conformalized action predictions consistently improve failure anticipation, reduce catastrophic errors, and provide a calibrated measure of confidence without retraining or modifying the underlying VLA.
>
---
#### [new 016] Enhancing Glass Surface Reconstruction via Depth Prior for Robot Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人导航任务，解决玻璃表面导致深度传感器失效的问题。通过融合深度先验与原始数据，提升玻璃区域的重建精度。**

- **链接: [https://arxiv.org/pdf/2604.18336](https://arxiv.org/pdf/2604.18336)**

> **作者:** Jiamin Zheng; Jingwen Yu; Guangcheng Chen; Hong Zhang
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Indoor robot navigation is often compromised by glass surfaces, which severely corrupt depth sensor measurements. While foundation models like Depth Anything 3 provide excellent geometric priors, they lack an absolute metric scale. We propose a training-free framework that leverages depth foundation models as a structural prior, employing a robust local RANSAC-based alignment to fuse it with raw sensor depth. This naturally avoids contamination from erroneous glass measurements and recovers an accurate metric scale. Furthermore, we introduce \ti{GlassRecon}, a novel RGB-D dataset with geometrically derived ground truth for glass regions. Extensive experiments demonstrate that our approach consistently outperforms state-of-the-art baselines, especially under severe sensor depth corruption. The dataset and related code will be released at this https URL.
>
---
#### [new 017] Time-Division Multiplexing Actuation in Tendon-Driven Arms: Lightweight Design and Fault Tolerance
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决航天应用中轻量化与故障容错问题。提出TDMA方法，减少执行器数量，提升可靠性与精度。**

- **链接: [https://arxiv.org/pdf/2604.16887](https://arxiv.org/pdf/2604.16887)**

> **作者:** Shoujie Li; Changqing Guo; Jianle Xu; Hong Luo; Xueqian Wang; Wenbo Ding; Bin Liang
>
> **备注:** 11 pages
>
> **摘要:** Robotic manipulators for aerospace applications require a delicate balance between lightweight construction and fault-tolerant operation to satisfy strict weight limitations and ensure reliability in remote, hazardous environments. This paper presents Time-Division Multiplexing Actuation (TDMA), a practical approach for tendon-driven robots that significantly reduces actuator count while preserving high torque output and intrinsic fault tolerance. The key hardware employs a vertically-stacked rotational selection structure that integrates self-rotating TDM motors for rapid configuration, electromagnetic clutches enabling sub-0.1 second engagement, a worm gear reducer for enhanced load capacity and self-locking capability, and a dual-encoder system for precise, long-term positioning. Leveraging TDMA, the proposed MuxArm achieves a self-weight of 2.17 kg, supports an actuator driving capacity of 10 kg, and maintains end-effector accuracy up to 1% of its length, even under partial servo failure. Additionally, an actuation space trajectory planning algorithm is developed, enabling fault-tolerant control and reducing tendon load by up to 50% compared to conventional methods. Comprehensive experiments demonstrate MuxArm's robust performance in diverse settings, including free-space, cluttered, and confined environments.
>
---
#### [new 018] Memory Centric Power Allocation for Multi-Agent Embodied Question Answering
- **分类: cs.RO; cs.IT**

- **简介: 该论文属于多智能体具身问答任务，解决长期记忆质量优化问题。提出QoM模型和MCPA算法，在通信约束下最大化记忆质量。**

- **链接: [https://arxiv.org/pdf/2604.17810](https://arxiv.org/pdf/2604.17810)**

> **作者:** Chengyang Li; Shuai Wang; Kejiang Ye; Weijie Yuan; Boyu Zhou; Yik-Chung Wu; Chengzhong Xu; Huseyin Arslan
>
> **备注:** 6 pages, submitted to GLOBECOM 2026
>
> **摘要:** This paper considers multi-agent embodied question answering (MA-EQA), which aims to query robot teams on what they have seen over a long horizon. In contrast to existing edge resource management methods that emphasize sensing, communication, or computation performance metrics, MA-EQA emphasizes the memory qualities. To cope with this paradigm shift, we propose a quality of memory (QoM) model based on generative adversarial exam (GAE), which leverages forward simulation to assess memory retrieval and uses the resulting exam scores to compute QoM values. Then we propose memory centric power allocation (MCPA), which maximizes the QoM function under communication resource constraints. Through asymptotic analysis, it is found that the transmit powers are proportional to the GAE error probability, thus prioritizing towards high-QoM robots. Extensive experiments demonstrate that MCPA achieves significant improvements over extensive benchmarks in terms of diverse metrics in various scenarios.
>
---
#### [new 019] BrainMem: Brain-Inspired Evolving Memory for Embodied Agent Task Planning
- **分类: cs.RO; cs.AI; cs.CV; cs.MA**

- **简介: 该论文属于具身智能任务规划领域，旨在解决代理在复杂3D环境中长期任务执行中的记忆与经验积累问题。提出BrainMem系统，通过类脑记忆机制提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.16331](https://arxiv.org/pdf/2604.16331)**

> **作者:** Xiaoyu Ma; Lianyu Hu; Wenbing Tang; Zixuan Hu; Zeqin Liao; Zhizhen Wu; Yang Liu
>
> **摘要:** Embodied task planning requires agents to execute long-horizon, goal-directed actions in complex 3D environments, where success depends on both immediate perception and accumulated experience across tasks. However, most existing LLM-based planners are stateless and reactive, operating without persistent memory and therefore repeating errors and struggling with spatial or temporal dependencies. We propose BrainMem(Brain-Inspired Evolving Memory), a training-free hierarchical memory system that equips embodied agents with working, episodic, and semantic memory inspired by human cognition. BrainMem continuously transforms interaction histories into structured knowledge graphs and distilled symbolic guidelines, enabling planners to retrieve, reason over, and adapt behaviors from past experience without any model fine-tuning or additional training. This plug-and-play design integrates seamlessly with arbitrary multi-modal LLMs and greatly reduces reliance on task-specific prompt engineering. Extensive experiments on four representative benchmarks, including EB-ALFRED, EB-Navigation, EB-Manipulation, and EB-Habitat, demonstrate that BrainMem significantly enhances task success rates across diverse models and difficulty subsets, with the largest gains observed on long-horizon and spatially complex tasks. These results highlight evolving memory as a promising and scalable mechanism for generalizable embodied intelligence.
>
---
#### [new 020] Periodic Steady-State Control of a Handkerchief-Spinning Task Using a Parallel Anti-Parallelogram Tendon-driven Wrist
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于柔性物体操控任务，解决手帕旋转中的周期稳态控制问题。设计了并联反平行四边形肌腱驱动手腕，结合粒子-弹簧模型，实现高精度操控与稳定运动。**

- **链接: [https://arxiv.org/pdf/2604.17863](https://arxiv.org/pdf/2604.17863)**

> **作者:** Lei Liu; Haonan Zhang; Huahang Xu; Zefan Zhang; Lulu Chang; Lei Lv; Andrew Ross McIntosh; Kai Sun; Zhenshan Bing; Jiahong Dong; Fuchun Sun
>
> **备注:** ICRA2026
>
> **摘要:** Spinning flexible objects, exemplified by traditional Chinese handkerchief performances, demands periodic steady-state motions under nonlinear dynamics with frictional contacts and boundary constraints. To address these challenges, we first design an intuitive dexterous wrist based on a parallel anti-parallelogram tendon-driven structure, which achieves 90 degrees omnidirectional rotation with low inertia and decoupled roll-pitch sensing, and implement a high-low level hierarchical control scheme. We then develop a particle-spring model of the handkerchief for control-oriented abstraction and strategy evaluation. Hardware experiments validate this framework, achieving an unfolding ratio of approximately 99% and fingertip tracking error of RMSE = 2.88 mm in high-dynamic spinning. These results demonstrate that integrating control-oriented modeling with a task-tailored dexterous wrist enables robust rest-to-steady-state transitions and precise periodic manipulation of highly flexible objects. More visualizations: this https URL
>
---
#### [new 021] Disentangled Robot Learning via Separate Forward and Inverse Dynamics Pretraining
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，解决视觉-动作耦合导致的训练效率低问题。提出DeFI框架，分离前向与逆向动力学预训练，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.16391](https://arxiv.org/pdf/2604.16391)**

> **作者:** Wenyao Zhang; Bozhou Zhang; Zekun Qi; Wenjun Zeng; Xin Jin; Li Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** Vision-language-action (VLA) models have shown great potential in building generalist robots, but still face a dilemma-misalignment of 2D image forecasting and 3D action prediction. Besides, such a vision-action entangled training manner limits model learning from large-scale, action-free web video data. To address these issues, we propose DeFI, a novel framework that Decouples visual Forward and Inverse dynamics pretraining to exploit respective data sources, wherein video generation and action prediction are disentangled. We introduce the General Forward Dynamics Model (GFDM), pretrained on diverse human and robot videos for future prediction, and the General Inverse Dynamics Model (GIDM), trained via self-supervised learning to infer latent actions from unlabeled video transitions. These models are then integrated into a unified architecture for end-to-end finetuning on downstream tasks. In this manner, GFDM and GIDM first shine separately and then cooperate for mutual benefit. Extensive experiments on CALVIN ABC-D and SimplerEnv demonstrate state-of-the-art performance, with DeFI achieving an average task length of 4.51 for CALVIN, 51.2% success rate on SimplerEnv-Fractal benchmark and 81.3% success rate in real-world deployment, significantly outperforming prior methods.
>
---
#### [new 022] NaviFormer: A Deep Reinforcement Learning Transformer-like Model to Holistically Solve the Navigation Problem
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于路径规划任务，旨在同时解决高阶路线规划和低阶轨迹规划问题。提出NaviFormer模型，结合深度强化学习与Transformer架构，实现高效全局导航。**

- **链接: [https://arxiv.org/pdf/2604.16967](https://arxiv.org/pdf/2604.16967)**

> **作者:** Daniel Fuertes; Andrea Cavallaro; Carlos R. del-Blanco; Fernando Jaureguizar; Narciso García
>
> **备注:** Published in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Path planning is usually solved by addressing either the (high-level) route planning problem (waypoint sequencing to achieve the final goal) or the (low-level) path planning problem (trajectory prediction between two waypoints avoiding collisions). However, real-world problems usually require simultaneous solutions to the route and path planning subproblems with a holistic and efficient approach. In this paper, we introduce NaviFormer, a deep reinforcement learning model based on a Transformer architecture that solves the global navigation problem by predicting both high-level routes and low-level trajectories. To evaluate NaviFormer, several experiments have been conducted, including comparisons with other algorithms. Results show competitive accuracy from NaviFormer since it can understand the constraints and difficulties of each subproblem and act consequently to improve performance. Moreover, its superior computation speed proves its suitability for real-time missions.
>
---
#### [new 023] SYMBOLIZER: Symbolic Model-free Task Planning with VLMs
- **分类: cs.RO**

- **简介: 该论文属于任务与运动规划（TAMP）领域，旨在解决传统方法依赖手工符号模型的问题。通过结合视觉语言模型和通用搜索策略，实现无需动作模型的符号化规划。**

- **链接: [https://arxiv.org/pdf/2604.17830](https://arxiv.org/pdf/2604.17830)**

> **作者:** Sami Azirar; Zlatan Ajanovic; Hermann Blum
>
> **备注:** under review
>
> **摘要:** Traditional Task and Motion Planning (TAMP) systems depend on physics models for motion planning and discrete symbolic models for task planning. Although physics model are often available, symbolic models (consisting of symbolic state interpretation and action models) must be meticulously handcrafted or learned from labeled data. This process is both resource-intensive and constrains the solution to the specific domain, limiting scalability and adaptability. On the other hand, Visual Language Models (VLMs) show desirable zero-shot visual understanding (due to their extensive training on heterogeneous data), but still achieve limited planning capabilities. Therefore, integrating VLMs with classical planning for long-horizon reasoning in TAMP problems offers high potential. Recent works in this direction still lack generality and depend on handcrafted, task-specific solutions, e.g. describing all possible objects in advance, or using symbolic action models. We propose a framework that generalizes well to unseen problem instances. The method requires only lifted predicates describing relations among objects and uses VLMs to ground them from images to obtain the symbolic state. Planning is performed with domain-independent heuristic search using goal-count and width-based heuristics, without need for action models. Symbolic search over VLM-grounded state-space outperforms direct VLM-based planning and performs on par with approaches that use a VLM-derived heuristic. This shows that domain-independent search can effectively solve problems across domains with large combinatorial state spaces. We extensively evaluate on extensively evaluate our method and achieve state-of-the-art results on the ProDG and ViPlan benchmarks.
>
---
#### [new 024] Think before Go: Hierarchical Reasoning for Image-goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于图像目标导航任务，旨在解决代理在复杂环境中难以定位远距离目标的问题。通过分层推理框架HRNav，结合高阶规划与低阶执行，提升导航效率。**

- **链接: [https://arxiv.org/pdf/2604.17407](https://arxiv.org/pdf/2604.17407)**

> **作者:** Pengna Li; Kangyi Wu; Shaoqing Xu; Fang Li; Lin Zhao; Long Chen; Zhi-Xin Yang; Nanning Zheng
>
> **备注:** Accepted by ACL2026 (main conference)
>
> **摘要:** Image-goal navigation steers an agent to a target location specified by an image in unseen environments. Existing methods primarily handle this task by learning an end-to-end navigation policy, which compares the similarities of target and observation images and directly predicts the actions. However, when the target is distant or lies in another room, such methods fail to extract informative visual cues, leading the agent to wander around. Motivated by the human cognitive principle that deliberate, high-level reasoning guides fast, reactive execution in complex tasks, we propose Hierarchical Reasoning Navigation (HRNav), a framework that decomposes image-goal navigation into high-level planning and low-level execution. In high-level planning, a vision-language model is trained on a self-collected dataset to generate a short-horizon plan, such as whether the agent should walk through the door or down the hallway. This downgrades the difficulty of the long-horizon task, making it more amenable to the execution part. In low-level execution, an online reinforcement learning policy is utilized to decide actions conditioned on the short-horizon plan. We also devise a novel Wandering Suppression Penalty (WSP) to further reduce the wandering problem. Together, these components form a hierarchical framework for Image-Goal Navigation. Extensive experiments in both simulation and real-world environments demonstrate the superiority of our method.
>
---
#### [new 025] An Edge-Host-Cloud Architecture for Robot-Agnostic, Caregiver-in-the-Loop Personalized Cognitive Exercise: Multi-Site Deployment in Dementia Care
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出一种面向痴呆护理的机器人协作架构，解决个性化认知训练问题。融合照护者知识、边缘计算与机器人，实现低延迟、隐私保护的互动系统。**

- **链接: [https://arxiv.org/pdf/2604.16408](https://arxiv.org/pdf/2604.16408)**

> **作者:** Wenzheng Zhao; Ruth Palan Lopez; Shu Fen Wung; Fengpei Yuan
>
> **备注:** 21 pages, 6 figures, 10 tables, submitted to IEEE Transactions on Robotics (T-RO)
>
> **摘要:** We present Speaking Memories, a distributed, stakeholder-in-the-loop robotic interaction platform for personalized cognitive exercise support. Rather than a single robot-centric system, Speaking Memories is designed as a generalizable robotics architecture that integrates caregiver-authored knowledge, local edge intelligence, and embodied robotic agents into a unified socio-technical loop. The platform fuses auditory, visual, and textual signals to enable emotion-aware, personalized dialogue, while decoupling multimodal perception and reasoning from robot-specific hardware through a local edge interaction server. This design achieves low-latency, privacy-preserving operation and supports scalable deployment across heterogeneous robotic embodiments. Caregivers and family members contribute structured biographical knowledge via a secure cloud portal, which conditions downstream dialogue policies and enables longitudinal personalization across interaction sessions. Beyond real-time interaction, the system incorporates an automated multimodal evaluation layer that continuously analyzes user responses, affective cues, and engagement patterns, producing structured interaction metrics at scale. These metrics support systematic assessment of interaction quality, enable data-driven model fine-tuning, and lay the foundation for future clinician- and caregiver-informed personalization and intervention planning. We evaluate the platform through real-world deployments, measuring end-to-end latency, dialogue coherence, interaction stability, and stakeholder-reported usability and engagement. Results demonstrate sub-6-second response latency, robust multimodal synchronization, and consistently positive feedback from both participants and caregivers. Furthermore, subsets of the dataset can be shared upon request, subject to participant consent and IRB constraints.
>
---
#### [new 026] RHINO-AR: An Augmented Reality Exhibit for Teaching Mobile Robotics Concepts in Museums
- **分类: cs.RO; cs.CY**

- **简介: 该论文属于增强现实（AR）应用任务，旨在解决虚拟现实（VR）展览中与真实环境脱节的问题。通过AR技术将历史机器人RHINO融入真实博物馆环境，提升参观体验和理解效果。**

- **链接: [https://arxiv.org/pdf/2604.16384](https://arxiv.org/pdf/2604.16384)**

> **作者:** Nils Dengler; Tim Graf; Leif Van Holland; Patrick Stotko; Reinhard Klein; Maren Bennewitz
>
> **摘要:** We present RHINO-AR, an interactive Augmented Reality (AR) museum exhibit that reintroduces the historical mobile robot RHINO into its original exhibition environment at the Deutsches Museum Bonn. The system builds on our previous work RHINO-VR, which reconstructed the robot and the environment in virtual reality. Although this created an engaging experience, it also revealed an important limitation, because visitors were separated from the real exhibition space and from the physical robot on display. RHINO-AR addresses this reality gap by placing a virtual reconstruction of the robot directly into the real museum space. Implemented on a Magic Leap~2 headset using Unity, our system combines real-time environment meshing with interactive visualizations of LiDAR sensing, traversability, and path planning to make otherwise invisible robotics processes understandable to non-expert visitors. We evaluated RHINO-AR in a two-day museum study with 22 participants, assessing usability, technical performance, satisfaction, conceptual understanding, and preference comparison to RHINO-VR. The results show that RHINO-AR was well received, effectively conveyed key navigation concepts, and generally preferred over the VR exhibit due to its stronger physical grounding and increased realism.
>
---
#### [new 027] On-Orbit Space AI: Federated, Multi-Agent, and Collaborative Algorithms for Satellite Constellations
- **分类: cs.RO; astro-ph.IM; cs.AI**

- **简介: 该论文属于空间AI任务，解决卫星星座协同与自主问题，提出联邦学习、多智能体算法和协同感知方法，提升星座整体效能与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.16518](https://arxiv.org/pdf/2604.16518)**

> **作者:** Ziyang Wang
>
> **备注:** Accepted by Algorithms, MDPI
>
> **摘要:** Satellite constellations are transforming space systems from isolated spacecraft into networked, software-defined platforms capable of on-orbit perception, decision making, and adaptation. Yet much of the existing AI studies remains centered on single-satellite inference, while constellation-scale autonomy introduces fundamentally new algorithmic requirements: learning and coordination under dynamic inter-satellite connectivity, strict SWaP-C limits, radiation-induced faults, non-IID data, concept drift, and safety-critical operational constraints. This survey consolidates the emerging field of on-orbit space AI through three complementary paradigms: (i) {federated learning} for cross-satellite training, personalization, and secure aggregation; (ii) {multi-agent algorithms} for cooperative planning, resource allocation, scheduling, formation control, and collision avoidance; and (iii) {collaborative sensing and distributed inference} for multi-satellite fusion, tracking, split/early-exit inference, and cross-layer co-design with constellation networking. We provide a system-level view and a taxonomy that unifies collaboration architectures, temporal mechanisms, and trust models. To support community development and keep this review actionable over time, we continuously curate relevant papers and resources at this https URL.
>
---
#### [new 028] OFlow: Injecting Object-Aware Temporal Flow Matching for Robust Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决现有模型在时间预测和对象感知上的不足。提出OFlow框架，统一时空预测与对象感知，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.17876](https://arxiv.org/pdf/2604.17876)**

> **作者:** Kuanning Wang; Ke Fan; Chenhao Qiu; Zeyu Shangguan; Yuqian Fu; Yanwei Fu; Daniel Seita; Xiangyang Xue
>
> **摘要:** Robust robotic manipulation requires not only predicting how the scene evolves over time, but also recognizing task-relevant objects in complex scenes. However, existing VLA models face two limitations. They typically act only on the current frame, while future prediction and object-aware reasoning are often learned in separate latent spaces. We propose OFlow (injecting Object-Aware Temporal Flow Matching into VLAs), a framework that addresses both limitations by unifying temporal foresight and object-aware reasoning in a shared semantic latent space. Our method forecasts future latents with temporal flow matching, factorizes them into object-aware representations that emphasize physically relevant cues while filtering task-irrelevant variation, and conditions continuous action generation on these predictions. By integrating OFlow into VLA pipelines, our method enables more reliable control under distribution shifts. Extensive experiments across LIBERO, LIBERO-Plus, MetaWorld, and SimplerEnv benchmarks and real-world tasks demonstrate that object-aware foresight consistently enhances robustness and success.
>
---
#### [new 029] Neural Network-Based Adaptive Event-Triggered Control for Dual-Arm Unmanned Aerial Manipulator Systems
- **分类: cs.RO**

- **简介: 该论文属于控制任务，解决双臂无人机操作系统的稳定与精确控制问题。通过神经网络和事件触发机制，提高系统性能并减少通信负担。**

- **链接: [https://arxiv.org/pdf/2604.17048](https://arxiv.org/pdf/2604.17048)**

> **作者:** Yang Wang; Hai Yu; Wei He; Jianda Han; Yongchun Fang; Xiao Liang
>
> **摘要:** This paper investigates the control problem of dual-arm unmanned aerial manipulator systems (DAUAMs). Strong coupling between the dual-arm and the multirotor platform, together with unmodeled dynamics and external disturbances, poses significant challenges to stable and accurate operation. An adaptive event-triggered control scheme with neural network-based approximation is proposed to address these issues while explicitly considering communication constraints. First, a dynamic model of the DAUAM system is derived, and a command-filter-based backstepping framework with error compensation is constructed. Then, a neural network is employed to approximate external frictions, and an event-triggered mechanism is designed to reduce the transmission frequency of control updates, thereby alleviating communication and energy burdens. Lyapunov-based analysis shows that all closed-loop signals remain bounded and that the tracking error converges to a neighborhood of the desired trajectory within a fixed time. Finally, experiments on a self-built DAUAM platform demonstrate that the proposed approach achieves accurate trajectory tracking.
>
---
#### [new 030] Safer Trajectory Planning with CBF-guided Diffusion Model for Unmanned Aerial Vehicles
- **分类: cs.RO**

- **简介: 该论文属于无人机轨迹规划任务，旨在解决扩散模型在复杂机动中安全性不足的问题。通过引入CBF引导采样，提升轨迹安全性并减少对安全数据的依赖。**

- **链接: [https://arxiv.org/pdf/2604.17527](https://arxiv.org/pdf/2604.17527)**

> **作者:** Peiwen Yang; Shiyu Bai; Weisong Wen; Yixin Gao; Jiahao Hu
>
> **摘要:** Safe and agile trajectory planning is essential for autonomous systems, especially during complex aerobatic maneuvers. Motivated by the recent success of diffusion models in generative tasks, this paper introduces AeroTrajGen, a novel framework for diffusion-based trajectory generation that incorporates control barrier function (CBF)-guided sampling during inference, specifically designed for unmanned aerial vehicles (UAVs). The proposed CBF-guided sampling addresses two critical challenges: (1) mitigating the inherent unpredictability and potential safety violations of diffusion models, and (2) reducing reliance on extensively safety-verified training data. During the reverse diffusion process, CBF-based guidance ensures collision-free trajectories by seamlessly integrating safety constraint gradients with the diffusion model's score function. The model features an obstacle-aware diffusion transformer architecture with multi-modal conditioning, including trajectory history, obstacles, maneuver styles, and goal, enabling the generation of smooth, highly agile trajectories across 14 distinct aerobatic maneuvers. Trained on a dataset of 2,000 expert demonstrations, AeroTrajGen is rigorously evaluated in simulation under multi-obstacle environments. Simulation results demonstrate that CBF-guided sampling reduces collision rates by 94.7% compared to unguided diffusion baselines, while preserving trajectory agility and diversity. Our code is open-sourced at this https URL.
>
---
#### [new 031] Learning-Based Sparsification of Dynamic Graphs in Robotic Exploration Algorithms
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人探索任务，解决动态图冗余问题，通过强化学习框架实现图的智能剪枝，提升探索一致性。**

- **链接: [https://arxiv.org/pdf/2604.16509](https://arxiv.org/pdf/2604.16509)**

> **作者:** Adithya V. Sastry; Bibek Poudel; Weizi Li
>
> **摘要:** Many robotic exploration algorithms rely on graph structures for frontier-based exploration and dynamic path planning. However, these graphs grow rapidly, accumulating redundant information and impacting performance. We present a transformer-based framework trained with Proximal Policy Optimization (PPO) to prune these graphs during exploration, limiting their growth and reducing the accumulation of excess information. The framework was evaluated on simulations of a robotic agent using Rapidly Exploring Random Trees (RRT) to carry out frontier-based exploration, where the learned policy reduces graph size by up to 96%. We find preliminary evidence that our framework learns to associate pruning decisions with exploration outcomes despite sparse, delayed reward signals. We also observe that while intelligent pruning achieves a lower rate of exploration compared to baselines, it yields the lowest standard deviation, producing the most consistent exploration across varied environments. To the best of our knowledge, these results are the first suggesting the viability of RL in sparsification of dynamic graphs used in robotic exploration algorithms.
>
---
#### [new 032] Diffusion-Based Optimization for Accelerated Convergence of Redundant Dual-Arm Minimum Time Problems
- **分类: cs.RO**

- **简介: 该论文属于机器人路径优化任务，解决冗余双臂系统快速跟踪指定轨迹的问题。通过扩散算法提升求解效率，减少计算时间并降低轨迹误差。**

- **链接: [https://arxiv.org/pdf/2604.16670](https://arxiv.org/pdf/2604.16670)**

> **作者:** Jushan Chen; Jonathan Fried; Santiago Paternain
>
> **备注:** Under review for conference publication
>
> **摘要:** We present a framework leveraging a novel variant of the model-based diffusion algorithm to minimize the time required for a redundant dual-arm robot configuration to follow a desired relative Cartesian path. Our prior work proposed a bi-level optimization approach for the dual-arm problem, where we derived the analytical solution to the lower-level convex sub-problem and solved the high-level nonconvex problem using a primal-dual approach. However, the gradient-based nature leads to a large computation overhead, and it prohibits directly imposing an $L_{\infty}$ Cartesian error constraint along the joint trajectory due to the sparsity of the gradient. In this work, we propose a diffusion-based framework that relies on probabilistic sampling to tackle the aforementioned challenges in the nonconvex high-level problem, leading to a 35x reduction in the runtime and 34\% less Cartesian error compared to our prior work.
>
---
#### [new 033] Greedy Kalman-Swarm: Improving State Estimation in Robot Swarms in Harsh Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人协同任务，解决恶劣环境下机器人群的状态估计问题。提出一种分布式“贪心”方法，提升群体精度，无需全局通信。**

- **链接: [https://arxiv.org/pdf/2604.16868](https://arxiv.org/pdf/2604.16868)**

> **作者:** Phunyapa Suksomboon; Paulo Garcia
>
> **备注:** accepted at ECTI-CON 2026
>
> **摘要:** State estimation is a fundamental requirement in robotics, where the accurate determination of a robot's state is essential for stable operation despite inherent process disturbances and sensor noise. Traditionally, this is achieved through Kalman filtering, providing a statistically optimal estimate by balancing predictive models with noisy measurements. In the context of robotic swarms, the challenge shifts from individual accuracy to collective coordination, where the integration of global dynamics can significantly enhance the precision of the entire group. Existing estimation techniques rely on centralized processing or heavy communication protocols to reach a global consensus, which are frequently impractical in real-world deployments. Here we show that a localized, "greedy" approach to distributed state estimation (termed "Greedy Kalman-Swarm") allows individual robots to leverage relative inter-robot sensing for improved accuracy without requiring full data availability or global communication. Simulations in communication-constrained environments show robots can effectively integrate all currently available neighbor data at each iteration to refine their internal states, yet remain robust and functional even when data is missing. This results in a performance profile that strikes a balance between the low overhead of independent estimation and the high accuracy of centralized systems, specifically under harsh or dynamic environmental conditions. Our results demonstrate that global state awareness can be emergent rather than enforced, providing a scalable framework for maintaining swarm cohesion in unpredictable terrains. We anticipate that this decentralized methodology will serve as a foundation for more resilient autonomous systems, particularly in search-and-rescue or space exploration missions where reliable, high-bandwidth communication cannot be guaranteed.
>
---
#### [new 034] Human Cognition in Machines: A Unified Perspective of World Models
- **分类: cs.RO; cs.AI; cs.CV; cs.ET**

- **简介: 该论文属于人工智能领域，旨在构建统一的世界模型框架，整合认知功能。解决现有模型在动机和元认知方面的不足，提出新方向。**

- **链接: [https://arxiv.org/pdf/2604.16592](https://arxiv.org/pdf/2604.16592)**

> **作者:** Timothy Rupprecht; Pu Zhao; Amir Taherin; Arash Akbari; Arman Akbari; Yumei He; Sean Duffy; Juyi Lin; Yixiao Chen; Rahul Chowdhury; Enfu Nan; Yixin Shen; Yifan Cao; Haochen Zeng; Weiwei Chen; Geng Yuan; Jennifer Dy; Sarah Ostadabbas; Silvia Zhang; David Kaeli; Edmund Yeh; Yanzhi Wang
>
> **摘要:** This comprehensive report distinguishes prior works by the cognitive functions they innovate. Many works claim an almost "human-like" cognitive capability in their world models. To evaluate these claims requires a proper grounding in first principles in Cognitive Architecture Theory (CAT). We present a conceptual unified framework for world models that fully incorporates all the cognitive functions associated with CAT (i.e. memory, perception, language, reasoning, imagining, motivation, and meta-cognition) and identify gaps in the research as a guide for future states of the art. In particular, we find that motivation (especially intrinsic motivation) and meta-cognition remain drastically under-researched, and we propose concrete directions informed by active inference and global workspace theory to address them. We further introduce Epistemic World Models, a new category encompassing agent frameworks for scientific discovery that operate over structured knowledge. Our taxonomy, applied across video, embodied, and epistemic world models, suggests research directions where prior taxonomies have not.
>
---
#### [new 035] Learning Whole-Body Humanoid Locomotion via Motion Generation and Motion Tracking
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，解决传统方法在复杂地形下适应性不足的问题。通过结合运动生成与跟踪，实现地形感知的全身运动控制。**

- **链接: [https://arxiv.org/pdf/2604.17335](https://arxiv.org/pdf/2604.17335)**

> **作者:** Zewei Zhang; Kehan Wen; Michael Xu; Junzhe He; Chenhao Li; Takahiro Miki; Clemens Schwarke; Chong Zhang; Xue Bin Peng; Marco Hutter
>
> **摘要:** Whole-body humanoid locomotion is challenging due to high-dimensional control, morphological instability, and the need for real-time adaptation to various terrains using onboard perception. Directly applying reinforcement learning (RL) with reward shaping to humanoid locomotion often leads to lower-body-dominated behaviors, whereas imitation-based RL can learn more coordinated whole-body skills but is typically limited to replaying reference motions without a mechanism to adapt them online from perception for terrain-aware locomotion. To address this gap, we propose a whole-body humanoid locomotion framework that combines skills learned from reference motions with terrain-aware adaptation. We first train a diffusion model on retargeted human motions for real-time prediction of terrain-aware reference motions. Concurrently, we train a whole-body reference tracker with RL using this motion data. To improve robustness under imperfectly generated references, we further fine-tune the tracker with a frozen motion generator in a closed-loop setting. The resulting system supports directional goal-reaching control with terrain-aware whole-body adaptation, and can be deployed on a Unitree G1 humanoid robot with onboard perception and computation. The hardware experiments demonstrate successful traversal over boxes, hurdles, stairs, and mixed terrain combinations. Quantitative results further show the benefits of incorporating online motion generation and fine-tuning the motion tracker for improved generalization and robustness.
>
---
#### [new 036] FLASH: Fast Learning via GPU-Accelerated Simulation for High-Fidelity Deformable Manipulation in Minutes
- **分类: cs.RO**

- **简介: 该论文提出FLASH框架，解决软体物体操作中的高保真模拟难题，通过GPU加速实现快速学习与真实世界迁移。**

- **链接: [https://arxiv.org/pdf/2604.17513](https://arxiv.org/pdf/2604.17513)**

> **作者:** Siyuan Luo; Bingyang Zhou; Chong Zhang; Xin Liu; Zhenhao Huang; Gang Yang; Zhengtao Han; Xiaotian Hu; Eric Yang; Rymon Yu; Ziqiu Zeng; Fan Shi
>
> **摘要:** Simulation frameworks such as Isaac Sim have enabled scalable robot learning for locomotion and rigid-body manipulation; however, contact-rich simulation remains a major bottleneck for deformable object manipulation. The continuously changing geometry of soft materials, together with large numbers of vertices and contact constraints, makes it difficult to achieve high accuracy, speed, and stability required for large-scale interactive learning. We present FLASH, a GPU-native simulation framework for contact-rich deformable manipulation, built on an accurate NCP-based solver that enforces strict contact and deformation constraints while being explicitly designed for fine-grained GPU parallelism. Rather than porting conventional single-instruction-multiple-data (SIMD) solvers to GPUs, FLASH redesigns the physics engine from the ground up to leverage modern GPU architectures, including optimized collision handling and memory layouts. As a result, FLASH scales to over 3 million degrees of freedom at 30 FPS on a single RTX 5090, while accurately simulating physical interactions. Policies trained solely on FLASH-generated synthetic data in minutes achieve robust zero-shot sim-to-real transfer, which we validate on physical robots performing challenging deformable manipulation tasks such as towel folding and garment folding, without any real-world demonstration, providing a practical alternative to labor-intensive real-world data collection.
>
---
#### [new 037] GaLa: Hypergraph-Guided Visual Language Models for Procedural Planning
- **分类: cs.RO**

- **简介: 该论文提出GaLa框架，解决 embodied AI 中程序规划的问题，通过超图建模物体间语义关系，提升视觉语言模型的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.17241](https://arxiv.org/pdf/2604.17241)**

> **作者:** Kun Wang; Yiming Li; Mingcheng Qu; Aqiang Zhang; Guang Yang; Tonghua Su
>
> **备注:** 14pages, 7figures
>
> **摘要:** Implicit spatial relations and deep semantic structures encoded in object attributes are crucial for procedural planning in embodied AI systems. However, existing approaches often over rely on the reasoning capabilities of vision language models (VLMs) themselves, while overlooking the rich structured semantic information that can be mined from multimodal inputs. As a result, models struggle to effectively understand functional spatial relationships in complex scenes. To fully exploit implicit spatial relations and deep semantic structures in multimodal data, we propose GaLa, a vision language framework for multimodal procedural planning. GaLa introduces a hypergraph-based representation, where object instances in the image are modeled as nodes, and region-level hyperedges are constructed by aggregating objects according to their attributes and functional semantics. This design explicitly captures implicit semantic relations among objects as well as the hierarchical organization of functional regions. Furthermore, we design a TriView HyperGraph Encoder that enforces semantic consistency across the node view, area view, and node area association view via contrastive learning, enabling hypergraph semantics to be more effectively injected into downstream VLM reasoning. Extensive experiments on the ActPlan1K and ALFRED benchmarks demonstrate that GaLa significantly outperforms existing methods in terms of execution success rate, LCS, and planning correctness.
>
---
#### [new 038] ReFineVLA: Multimodal Reasoning-Aware Generalist Robotic Policies via Teacher-Guided Fine-Tuning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在提升视觉-语言-动作模型的推理能力。通过教师引导的微调，增强模型在复杂任务中的逻辑理解和泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.17800](https://arxiv.org/pdf/2604.17800)**

> **作者:** Tuan Van Vo; Tan Q. Nguyen; Khang Nguyen; Nhat Xuan Tran; Duy H. M. Nguyen; An T. Le; Ngo Anh Vien; Minh Nhat Vu
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2505.19080
>
> **摘要:** Vision-Language-Action (VLA) models have gained much attention from the research community thanks to their strength in translating multimodal observations with linguistic instructions into desired robotic actions. Despite their advancements, VLAs often overlook explicit reasoning and learn the functional input-action mappings, omitting crucial logical steps, which are especially pronounced in interpretability and generalization for complex, long-horizon manipulation tasks. In this work, we propose ReFineVLA, a multimodal reasoning-aware framework that fine-tunes VLAs with teacher-guided reasons. We first augment robotic datasets with reasoning rationales generated by an expert teacher model, guiding VLA models to learn to reason about their actions. Then, we fine-tune pre-trained VLAs with the reasoning-enriched datasets with ReFineVLA, while maintaining the underlying generalization abilities and boosting reasoning capabilities. We also conduct attention map visualization to analyze the alignment among visual observation, linguistic prompts, and to-be-executed actions of ReFineVLA, reflecting the model is ability to focus on relevant tasks and actions. Through this additional step, we explore that ReFineVLA-trained models exhibit a meaningful agreement between vision-language and action domains, highlighting the enhanced multimodal understanding and generalization. Evaluated across a suite of simulated manipulation benchmarks on SimplerEnv with both WidowX and Google Robot tasks, ReFineVLA achieves state-of-the-art performance, in success rate over the second-best method on the both the WidowX benchmark and Google Robot Tasks.
>
---
#### [new 039] Visual-RRT: Finding Paths toward Visual-Goals via Differentiable Rendering
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人运动规划任务，解决视觉目标下路径规划问题。通过结合可微渲染与RRT，提出vRRT方法，实现从视觉输入到路径生成的端到端规划。**

- **链接: [https://arxiv.org/pdf/2604.16388](https://arxiv.org/pdf/2604.16388)**

> **作者:** Sebin Lee; Jumin Lee; Taeyeon Kim; Younju Na; Woobin Im; Sung-Eui Yoon
>
> **摘要:** Rapidly-exploring random trees (RRTs) have been widely adopted for robot motion planning due to their robustness and theoretical guarantees. However, existing RRT-based planners require explicit goal configurations specified as numerical joint angles, while many practical applications provide goal specifications through visual observations such as images or demonstration videos where precise goal configurations are unavailable. In this paper, we propose visual-RRT (vRRT), a motion planner that enables visual-goal planning by unifying gradient-based exploitation from differentiable robot rendering with sampling-based exploration from RRTs. We further introduce (i) a frontier-based exploration-exploitation strategy that adaptively prioritizes visually promising search regions, and (ii) inertial gradient tree expansion that inherits optimization states across tree branches for momentum-consistent gradient exploitation. Extensive experiments across various robot manipulators including Franka, UR5e, and Fetch demonstrate that vRRT achieves effective visual-goal planning in both simulated and real-world settings, bridging the gap between sampling-based planning and vision-centric robot applications. Our code is available at this https URL.
>
---
#### [new 040] A Hamilton-Jacobi Reachability-Guided Search Framework for Efficient and Safe Indoor Planar Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于室内机器人导航任务，解决动态环境中高效安全路径规划问题。结合HJ可达性与图搜索，提升规划效率与安全性。**

- **链接: [https://arxiv.org/pdf/2604.17679](https://arxiv.org/pdf/2604.17679)**

> **作者:** Hanyang Hu; Cameron Siu; Mo Chen
>
> **摘要:** Autonomous navigation requires planning to reach a goal safely and efficiently in complex and potentially dynamic environments. Graph search-based algorithms are widely adopted due to their generality and theoretical guarantees when equipped with admissible heuristics. However, the computational complexity of graph search grows rapidly with the dimensionality of the search space, often making real-time planning in dynamic environments intractable. In this paper, we combine offline Hamilton-Jacobi (HJ) reachability with online graph search to leverage the complementary strengths of both. Precomputed HJ value functions, used as informative heuristics and proactive safety constraints, amortize online computation of the graph search process. At the same time, graph search enables reachability-based reasoning to be incorporated into online planning, overcoming the long-standing challenge of HJ reachability requiring full knowledge of the environment. Extensive simulation studies and real-world experiments demonstrate that the proposed approach consistently outperforms baseline methods in terms of planning efficiency and navigation safety, in environments with and without human presence.
>
---
#### [new 041] Relative State Estimation using Event-Based Propeller Sensing
- **分类: cs.RO; cs.CV; eess.SY**

- **简介: 该论文属于多无人机相对状态估计任务，解决传统视觉方法在复杂环境中的性能问题。通过事件相机和螺旋桨频率分析，实现高精度、实时的相对定位。**

- **链接: [https://arxiv.org/pdf/2604.18289](https://arxiv.org/pdf/2604.18289)**

> **作者:** Ravi Kumar Thakur; Luis Granados Segura; Jan Klivan; Radim Špetlík; Tobiáš Vinklárek; Matouš Vrba; Martin Saska
>
> **摘要:** Autonomous swarms of multi-Unmanned Aerial Vehicle (UAV) system requires an accurate and fast relative state estimation. Although monocular frame-based camera methods perform well in ideal conditions, they are slow, suffer scale ambiguity, and often struggle in visually challenging conditions. The advent of event cameras addresses these challenging tasks by providing low latency, high dynamic range, and microsecond-level temporal resolution. This paper proposes a framework for relative state estimation for quadrotors using event-based propeller sensing. The propellers in the event stream are tracked by detection to extract the region-of-interests. The event streams in these regions are processed in temporal chunks to estimate per-propeller frequencies. These frequency measurements drive a kinematic state estimation module as a thrust input, while camera-derived position measurements provide the update step. Additionally, we use geometric primitives derived from event streams to estimate the orientation of the quadrotor by fitting an ellipse over a propeller and backprojecting it to recover body-frame tilt-axis. The existing event-based approaches for quadrotor state estimation use the propeller frequency in simulated flight sequences. Our approach estimates the propeller frequency under 3% error on a test dataset of five real-world outdoor flight sequences, providing a method for decentralized relative localization for multi-robot systems using event camera.
>
---
#### [new 042] MM-Hand: A 21-DOF Multi-modal Modular Dexterous Robotic Hand with Remote Actuation
- **分类: cs.RO**

- **简介: 该论文属于机器人手设计任务，解决高自由度机械手的紧凑驱动与散热问题。提出MM-Hand，采用远程绳索驱动，实现轻量化、高感知和可维护的灵巧操作平台。**

- **链接: [https://arxiv.org/pdf/2604.17245](https://arxiv.org/pdf/2604.17245)**

> **作者:** Zhuoheng Li; Qingquan Lin; Checheng Yu; Qiangyu Chen; Zhiqian Lan; Lutong Zhang; Hongyang Li; Ping Luo
>
> **摘要:** High-DOF dexterous hands require compact actuation, rich sensing, and reliable thermal behavior, but conventional designs often occupy valuable in-hand space, increase end-effector mass, and suffer from heat accumulation near the hand. Remote tendon-driven actuation offers an alternative by relocating motors to the robot base or an external motor hub, thereby freeing the fingers and palm for additional degrees of freedom, sensing modules, and maintainable mechanical structures. This paper presents MM-Hand, a 21-DOF Multimodal Modular dexterous hand based on remote tendon-driven actuation. The hand integrates spring-return tendon-driven fingers, modular 3D-printed finger and palm structures, quick tendon connectors for maintenance, and a multimodal sensing system including joint angle sensors, tactile sensors, motor-side feedback, and in-palm stereo vision. We further analyze tendon-sheath length variation and friction loss to guide the design of the routing, motor hub, and closed-loop joint control. Experiments validate the transmission, output force, sensing, and control capability of the system. The fingertip force reaches 25N under a 1m remote sheath transmission, demonstrating practical load capacity despite long-distance tendon routing. Closed-loop joint-level experiments further evaluate command tracking with a static arm and during arm motion. These results show that MM-Hand provides a lightweight, sensor-rich, and maintainable hardware platform for dexterous manipulation research. To support the community, all hardware designs and software frameworks are made fully open-source at this https URL.
>
---
#### [new 043] Web-Gewu: A Browser-Based Interactive Playground for Robot Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文提出Web-Gewu，一个基于浏览器的机器人强化学习交互平台，解决教育中计算成本高和环境配置复杂的问题。通过云边端协同架构，实现低延迟、低成本的实时交互教学。**

- **链接: [https://arxiv.org/pdf/2604.17050](https://arxiv.org/pdf/2604.17050)**

> **作者:** Kaixuan Chen; Linqi Ye
>
> **摘要:** With the rapid development of embodied intelligence, robotics education faces a dual challenge: high computational barriers and cumbersome environment configuration. Existing centralized cloud simulation solutions incur substantial GPU and bandwidth costs that preclude large-scale deployment, while pure local computing is severely constrained by learners' hardware limitations. To address these issues, we propose \href{this http URL}{Web-Gewu}, an interactive robotics education platform built on a WebRTC cloud-edge-client collaborative architecture. The system offloads all physics simulation and reinforcement learning (RL) training to the edge node, while the cloud server acts exclusively as a lightweight signaling relay, enabling extremely low-cost browser-based peer-to-peer (P2P) real-time streaming. Learners can interact with multi-form robots at low end-to-end latency directly in a web browser without any local installation, and simultaneously observe real-time visualization of multi-dimensional monitoring data, including reinforcement learning reward curves. Combined with a predefined robust command communication protocol, Web-Gewu provides a highly scalable, out-of-the-box, and barrier-free teaching infrastructure for embodied intelligence, significantly lowering the barrier to entry for cutting-edge robotics technology.
>
---
#### [new 044] Driving risk emerges from the required two-dimensional joint evasive acceleration
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全任务，解决风险评估不准确问题。提出二维避撞加速度（EA）方法，更真实地量化碰撞风险。**

- **链接: [https://arxiv.org/pdf/2604.17841](https://arxiv.org/pdf/2604.17841)**

> **作者:** Hao Cheng; Yanbo Jiang; Wenhao Yu; Rui Zhou; Jiang Bian; Keyu Chen; Zhiyuan Liu; Heye Huang; Hailun Zhang; Fang Zhang; Jianqiang Wang; Sifa Zheng
>
> **备注:** 23 pages, 5 figures; supplementary information provided as an ancillary file
>
> **摘要:** Most autonomous driving safety benchmarks use time-to-collision (TTC) to assess risk and guide safe behaviour. However, TTC-based methods treat risk as a one-dimensional closing problem, despite the inherently two-dimensional nature of collision avoidance, and therefore cannot faithfully capture risk or its evolution over time. Here, we report evasive acceleration (EA), a hyperparameter-free and physically interpretable two-dimensional paradigm for risk quantification. By evaluating all possible directions of collision avoidance, EA defines risk as the minimum magnitude of a constant relative acceleration vector required to alter the relative motion and make the interaction collision-free. Using interaction data from five open datasets and more than 600 real crashes, we derive percentile-based warning thresholds and show that EA provides the earliest statistically significant warning across all thresholds. Moreover, EA provides the best discrimination of eventual collision outcomes and improves information retention by 54.2-241.4% over all compared baselines. Adding EA to existing methods yields 17.5-95.5 times more information gain than adding existing methods to EA, indicating that EA captures much of the outcome-relevant information in existing methods while contributing substantial additional nonredundant information. Overall, EA better captures the structure of collision risk and provides a foundation for next-generation autonomous driving systems.
>
---
#### [new 045] Modeling, Control and Self-sensing of Dielectric Elastomer Soft Actuators: A Review
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于软体机器人领域，旨在解决DEA建模、控制与自感知难题。通过综述多种建模和控制方法，提升DEA的性能与应用潜力。**

- **链接: [https://arxiv.org/pdf/2604.17199](https://arxiv.org/pdf/2604.17199)**

> **作者:** Y. Zhao; G. Meng
>
> **摘要:** Dielectric elastomer actuators (DEAs) have garnered extensive attention especially in soft robotic applications over the past few decades owing to the advantages of lightweight, large strain, fast response and high energy density. However, because the DEAs suffer from nonlinear elasticity, inherent viscoelastic creep, hysteresis and vibrational dynamics, the modeling, control and self-sensing of DEAs are challenging, thereby hindering the practical applications of DEAs. In order to address these challenges, numerous studies have been conducted. In this review, various physics-based modeling methods and phenomenological modeling methods for predicting the electromechanical response of DEAs are presented and discussed. Different control methods for DEAs are reviewed, which are classified into open-loop feedforward control, feedback control, feedforward-feedback control and adaptive feedforward control. Physics-based self-sensing methods and data-driven self-sensing methods for reconstructing the DEA displacement without the need for additional sensors are discussed. Finally, the existing problems and new opportunities for the further studies are summarized.
>
---
#### [new 046] Refinement of Accelerated Demonstrations via Incremental Iterative Reference Learning Control for Fast Contact-Rich Imitation Learning
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于机器人模仿学习任务，旨在解决高速接触操作中演示轨迹失真的问题。通过改进的迭代参考学习控制方法，提升演示精度与执行速度。**

- **链接: [https://arxiv.org/pdf/2604.16850](https://arxiv.org/pdf/2604.16850)**

> **作者:** Koki Yamane; Cristian C. Beltran-Hernandez; Steven Oh; Masashi Hamaya; Sho Sakaino
>
> **备注:** 8 pages, 11 figures, submitted to IROS 2026
>
> **摘要:** Fast execution of contact-rich manipulation is critical for practical deployment, yet providing fast demonstrations for imitation learning (IL) remains challenging: humans cannot demonstrate at high speed, and naively accelerating demonstrations alters contact dynamics and induces large tracking errors. We present a method to autonomously refine time-accelerated demonstrations by repurposing Iterative Reference Learning Control (IRLC) to iteratively update the reference trajectory from observed tracking errors. However, applying IRLC directly at high speed tends to produce larger early-iteration errors and less stable transients. To address this issue, we propose Incremental Iterative Reference Learning Control (I2RLC), which gradually increases the speed while updating the reference, yielding high-fidelity trajectories. We validate on real-robot whiteboard erasing and peg-in-hole tasks using a teleoperation setup with a compliance-controlled follower and a 3D-printed haptic leader. Both IRLC and I2RLC achieve up to 10x faster demonstrations with reduced tracking error; moreover, I2RLC improves spatial similarity to the original trajectories by 22.5% on average over IRLC across three tasks and multiple speeds (3x-10x). We then use the refined trajectories to train IL policies; the resulting policies execute faster than the demonstrations and achieve 100% success rates in the peg-in-hole task at both seen and unseen positions, with I2RLC-trained policies exhibiting lower contact forces than those trained on IRLC-refined demonstrations. These results indicate that gradual speed scheduling coupled with reference adaptation provides a practical path to fast, contact-rich IL.
>
---
#### [new 047] Novel Algorithms for Smoothly Differentiable and Efficiently Vectorizable Contact Manifold Construction
- **分类: cs.RO**

- **简介: 该论文属于机器人接触环境下的行为生成任务，旨在解决接触模拟中梯度获取困难的问题，通过提出可微且可向量化的碰撞检测方法实现。**

- **链接: [https://arxiv.org/pdf/2604.17538](https://arxiv.org/pdf/2604.17538)**

> **作者:** Onur Beker; Andreas René Geist; Anselm Paulus; Georg Martius
>
> **备注:** Accepted for publication at the ICRA 2026 Workshop on Contact-Rich Control and Representation
>
> **摘要:** Generating intelligent robot behavior in contact-rich settings is a research problem where zeroth-order methods currently prevail. Developing methods that make use of first/second order information about the dynamics holds great promise in terms of increasing the solution speed and computational efficiency. The main bottleneck in this research direction is the difficulty in obtaining useful gradients and Hessians, due to pathologies in all three steps of a common simulation pipeline: i) collision detection, ii) contact dynamics, iii) time integration. This abstract proposes a method that can address the collision detection part of the puzzle in a manner that is smoothly differentiable and massively vectorizable. This is achieved via two contributions: i) a highly expressive class of analytical SDF primitives that can efficiently represent complex 3D surfaces, ii) a novel contact manifold generation routine that makes use of this geometry representation.
>
---
#### [new 048] LongBench: Evaluating Robotic Manipulation Policies on Real-World Long-Horizon Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决长时域操作策略失效问题。通过构建真实世界基准LongBench，评估策略在不同情境下的鲁棒性和一致性。**

- **链接: [https://arxiv.org/pdf/2604.16788](https://arxiv.org/pdf/2604.16788)**

> **作者:** Xueyao Chen; Jingkai Jia; Tong Yang; Yibo Fu; Wei Li; Wenqiang Zhang
>
> **摘要:** Robotic manipulation policies often degrade over extended horizons, yet existing benchmarks provide limited insight into why such failures occur. Most prior benchmarks are either simulation-based or report aggregate success, making it difficult to disentangle the distinct sources of temporal difficulty in real-world execution. We introduce LongBench, a real-world benchmark for evaluating long-horizon manipulation. LongBench consists of over 1,000 real-world episodes, covering two complementary regimes: Context-Independent (fully observable) and Context-Dependent (ambiguity-driven). By organizing tasks into capability- and ambiguity-specific subsets, LongBench enables mechanism-aware evaluation of execution robustness, temporal consistency, and context-dependent reasoning. Evaluating six state-of-the-art policies reveals that long-horizon performance is not governed by a single factor. We observe that performance in fully observable settings is more strongly associated with execution robustness, while contextual difficulty varies across tasks and is not consistently improved by memory-based methods. We hope that LongBench serves as a useful benchmark for studying long-horizon manipulation and for developing policies with stronger robustness across both execution and contextual challenges.
>
---
#### [new 049] Muscle-inspired magnetic actuators that push, pull, crawl, and grasp
- **分类: cs.RO; cond-mat.mtrl-sci; cond-mat.soft; physics.app-ph**

- **简介: 该论文属于软体机器人领域，旨在开发具有多种运动功能的磁性执行器。通过激光粉末床熔融技术，实现材料刚度与磁性的精确调控，解决传统执行器功能单一的问题。**

- **链接: [https://arxiv.org/pdf/2604.18090](https://arxiv.org/pdf/2604.18090)**

> **作者:** Muhammad Bilal Khan; Florian Hofmann; Kilian Schäfer; Matthias Lutzi; Oliver Gutfleisch
>
> **摘要:** Functional magnetic composites capable of large deformation, load bearing, and multifunctional motion are essential for next-generation adaptive soft robots. Here, we present muscle-inspired magnetic actuators (MMA), additively manufactured from a thermoplastic/permanent magnet polyurethane/Nd2Fe14B (TPU/MQP-S) composite using laser powder bed fusion (LPBF). By tuning the laser-energy scale between 1.0 and 3.0, both mechanical stiffness and magnetic response are precisely controlled: the tensile strength increases from 0.28 to 0.99 MPa while maintaining 30-45% elongation at break. This process enables the creation of 0.5 mm-thick flexural hinges, which reversibly bend and fold under moderate magnetic fields without damage. Two actuator types are reported showing the system versatility. The elongated actuator with self-weight of 1.57 g, magnetized in its contracted state, achieves linear contraction under a 500 mT field, lifting 50 g (32x its own weight) and sustaining performance over at least 50 cycles. Equipped with anisotropic frictional feet, it supports movement of a magnetic crawling robot that achieves up to 100% locomotion success on textured substrates. The expandable actuator exhibits reversible opening and closing under a 300 mT field, reliably grasping and releasing different objects, including soft berries and rigid 3D printed geometries. It can also anchor in a tube while holding suspended 50 g loads. This work demonstrates a LPBF-based strategy to program both stiffness and magnetization within a single material system, enabling remotely driven, reconfigurable, and fatigue-resistant soft actuators. The approach opens new possibilities for force controlled, multifunctional magnetic soft robots for adaptive gripping, locomotion, and minimally invasive manipulation of biomedical tools.
>
---
#### [new 050] Emergency Stopping for Liquid-manipulating Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人液态物质操作任务，解决突发情况下的紧急停止问题。通过最优控制方法生成无溢出的快速停止轨迹，提升操作安全性。**

- **链接: [https://arxiv.org/pdf/2604.16667](https://arxiv.org/pdf/2604.16667)**

> **作者:** Samuli Hynninen; Ville Kyrki
>
> **摘要:** Manipulating open liquid containers is challenging because liquids are highly sensitive to vessel accelerations and jerks. Although spill-free liquid manipulation has been widely studied, emergency stopping under unexpected hazards has received little attention, despite the fact that abrupt braking may cause hazardous spills. This letter presents an emergency stop system for robots manipulating liquids in open containers. We formulate emergency stopping as an optimal control problem and solve it in a model predictive control framework to generate time-optimal, spill-free stopping trajectories. The method operates as a plug-and-play safety layer on top of existing slosh-free motion planning methods, enabling immediate reaction to detected hazards while accounting for nonlinear liquid dynamics. We demonstrate, through simulation and on a 7-DoF Franka Emika Panda robot, that the proposed approach achieves fast emergency stopping without spilling.
>
---
#### [new 051] DART: Learning-Enhanced Model Predictive Control for Dual-Arm Non-Prehensile Manipulation
- **分类: cs.RO**

- **简介: 该论文研究非抓取双臂操作任务，解决物体在动态托盘上精准移动的问题。提出DART框架，结合MPC与优化阻抗控制，融合三种动力学建模方法提升控制精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.17833](https://arxiv.org/pdf/2604.17833)**

> **作者:** Autrio Das; Shreya Bollimuntha; Madala Venkata Renu Jeevesh; Keshab Patra; Tashmoy Gosh; Nagamanikandan G; Arun Kumar; Madhava Krishna
>
> **摘要:** What appears effortless to a human waiter remains a major challenge for robots. Manipulating objects nonprehensilely on a tray is inherently difficult, and the complexity is amplified in dual-arm settings. Such tasks are highly relevant to service robotics in domains such as hotels and hospitality, where robots must transport and reposition diverse objects with precision. We present DART, a novel dual-arm framework that integrates nonlinear Model Predictive Control (MPC) with an optimization-based impedance controller to achieve accurate object motion relative to a dynamically controlled tray. The framework systematically evaluates three complementary strategies for modeling tray-object dynamics as the state transition function within our MPC formulation: (i) a physics-based analytical model, (ii) an online regression based identification model that adapts in real-time, and (iii) a reinforcement learning-based dynamics model that generalizes across object properties. Our pipeline is validated in simulation with objects of varying mass, geometry, and friction coefficients. Extensive evaluations highlight the trade-offs among the three modeling strategies in terms of settling time, steady-state error, control effort, and generalization across objects. To the best of our knowledge, DART constitutes the first framework for non-prehensile dual-arm manipulation of objects on a tray. Project Link: this https URL
>
---
#### [new 052] ST-$π$: Structured SpatioTemporal VLA for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ST-π模型，解决机器人操作中的细粒度时空操控问题。通过结构化时空视觉语言模型和动作专家，显式规划时空行为并优化局部控制。**

- **链接: [https://arxiv.org/pdf/2604.17880](https://arxiv.org/pdf/2604.17880)**

> **作者:** Chuanhao Ma; Hanyu Zhou; Shihan Peng; Yan Li; Tao Gu; Luxin Yan
>
> **摘要:** Vision-language-action (VLA) models have achieved great success on general robotic tasks, but still face challenges in fine-grained spatiotemporal manipulation. Typically, existing methods mainly embed spatiotemporal knowledge into visual and action representations, and directly perform a cross-modal mapping for step-level action prediction. However, such spatiotemporal reasoning remains largely implicit, making it difficult to handle multiple sequential behaviors with explicit spatiotemporal boundaries. In this work, we propose ST-$\pi$, a structured spatiotemporal VLA model for robotic manipulation. Our model is guided by two key designs: 1) Spatiotemporal VLM. We encode 4D observations and task instructions into latent spaces, and feed them into the LLM to generate a sequence of causally ordered chunk-level action prompts consisting of sub-tasks, spatial grounding and temporal grounding. 2) Spatiotemporal action expert. Conditioned on chunk-level action prompts, we design a structured dual-generator guidance to jointly model spatial dependencies and temporal causality, thus predicting step-level action parameters. Within this structured framework, the VLM explicitly plans global spatiotemporal behavior, and the action expert further refines local spatiotemporal control. In addition, we propose a real-world robotic dataset with structured spatiotemporal annotations for fine-tuning. Extensive experiments have been conducted to demonstrate the effectiveness of our model. Our code link: this https URL.
>
---
#### [new 053] Locomotion of an Elastic Snake Robot via Natural Dynamics
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在提升弹性蛇形机器人的移动效率。通过研究自然动力学，设计高效步态，比较不同方法的性能。**

- **链接: [https://arxiv.org/pdf/2604.17895](https://arxiv.org/pdf/2604.17895)**

> **作者:** Tristan Ehlert; Arne Sachtler; Annika Schmidt; Davide Calzolari; Alin Albu-Schäffer
>
> **摘要:** Nature suggests that exploiting the elasticities and natural dynamics of robotic systems could increase their locomotion efficiency. Prior work on elastic snake robots supports this hypothesis, but has not fully exploited the nonlinear dynamic behavior of the systems. Recent advances in eigenmanifold theory enable a better characterization of the natural dynamics in complex nonlinear systems. This letter investigates if and how the nonlinear natural dynamics of a kinematic elastic snake robot can be used to design efficient gaits. Two types of gaits based on natural dynamics are presented and compared to a state-of-the-art approach using dynamics simulations. The results reveal that a gait generated by switching between two nonlinear normal modes does not improve the locomotion efficiency of the robot. In contrast, gaits based on non-brake periodic trajectories (non-brake orbits) are perfectly efficient in the energy-conservative case. Further simulations with friction reveal that, in a more realistic scenario, non-brake orbit gaits achieve higher efficiency compared to the baseline gait on the rigid system. Overall, the investigation offers promising insights into the design of gaits based on natural dynamics, fostering further research.
>
---
#### [new 054] OmniVLA-RL: A Vision-Language-Action Model with Spatial Understanding and Online RL
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决空间感知不准确、多模态融合不佳和强化学习不稳定问题，提出OmniVLA-RL架构与Flow-GSPO方法提升性能。**

- **链接: [https://arxiv.org/pdf/2604.17706](https://arxiv.org/pdf/2604.17706)**

> **作者:** Haoxiang Jie; Yaoyuan Yan; Xiangyu Wei; Kailin Wang; Hongjie Yan; Zhiyou Heng; Daocheng Chen
>
> **摘要:** Visual-Language-Action (VLA) models represent a paradigm shift in embodied AI, yet existing frameworks often struggle with imprecise spatial perception, suboptimal multimodal fusion, and instability in reinforcement learning. To bridge these gaps, we propose OmniVLA-RL, a novel architecture that leverages a Mix-of-Transformers (MoT) design to synergistically integrate reasoning, spatial, and action experts. Furthermore, we introduce Flow-GSPO, which reformulates flow matching as a Stochastic Differential Equation (SDE) process and integrates it with Group Segmented Policy Optimization (GSPO) to enhance action precision and training robustness. Extensive evaluations on the LIBERO and LIBERO-Plus benchmarks demonstrate that OmniVLA-RL significantly outperforms state-of-the-art methods, effectively overcoming the fundamental limitations of current VLA models.
>
---
#### [new 055] Multi-stage Planning for Multi-target Surveillance using Aircrafts Equipped with Synthetic Aperture Radars Aware of Target Visibility
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于路径规划任务，解决SAR飞机在复杂地形下多目标监视的轨迹生成问题。通过多阶段规划系统，优化飞行路径以提升目标可见性与成像质量。**

- **链接: [https://arxiv.org/pdf/2604.16962](https://arxiv.org/pdf/2604.16962)**

> **作者:** Daniel Fuertes; Carlos R. del-Blanco; Fernando Jaureguizar; Juan José Navarro-Corcuera; Narciso García
>
> **备注:** Published in IEEE/RAS International Conference on Automation Science and Engineering 2025
>
> **摘要:** Generating trajectories for synthetic aperture radar (SAR)-equipped aircraft poses significant challenges due to terrain constraints, and the need for straight-flight segments to ensure high-quality imaging. Related works usually focus on trajectory optimization for predefined straight-flight segments that do not adapt to the target visibility, which depends on the 3D terrain and aircraft orientation. In addition, this assumption does not scale well for the multi-target problem, where multiple straight-flight segments that maximize target visibility must be defined for real-time operations. For this purpose, this paper presents a multi-stage planning system. First, the waypoint sequencing to visit all the targets is estimated. Second, straight-flight segments maximizing target visibility according to the 3D terrain are predicted using a novel neural network trained with deep reinforcement learning. Finally, the segments are connected to create a trajectory via optimization that imposes 3D Dubins curves. Evaluations demonstrate the robustness of the system for SAR missions since it ensures high-quality multi-target SAR image acquisition aware of 3D terrain and target visibility, and real-time performance.
>
---
#### [new 056] EmbodiedLGR: Integrating Lightweight Graph Representation and Retrieval for Semantic-Spatial Memory in Robotic Agents
- **分类: cs.RO**

- **简介: 该论文提出EmbodiedLGR-Agent，用于机器人高效构建和检索语义-空间记忆，解决环境信息存储与快速查询问题。**

- **链接: [https://arxiv.org/pdf/2604.18271](https://arxiv.org/pdf/2604.18271)**

> **作者:** Paolo Riva; Leonardo Gargani; Matteo Frosi; Matteo Matteucci
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** As the world of agentic artificial intelligence applied to robotics evolves, the need for agents capable of building and retrieving memories and observations efficiently is increasing. Robots operating in complex environments must build memory structures to enable useful human-robot interactions by leveraging the mnemonic representation of the current operating context. People interacting with robots may expect the embodied agent to provide information about locations, events, or objects, which requires the agent to provide precise answers within human-like inference times to be perceived as responsive. We propose the Embodied Light Graph Retrieval Agent (EmbodiedLGR-Agent), a visual-language model (VLM)-driven agent architecture that constructs dense and efficient representations of robot operating environments. EmbodiedLGR-Agent directly addresses the need for an efficient memory representation of the environment by providing a hybrid building-retrieval approach built on parameter-efficient VLMs that store low-level information about objects and their positions in a semantic graph, while retaining high-level descriptions of the observed scenes with a traditional retrieval-augmented architecture. EmbodiedLGR-Agent is evaluated on the popular NaVQA dataset, achieving state-of-the-art performance in inference and querying times for embodied agents, while retaining competitive accuracy on the global task relative to the current state-of-the-art approaches. Moreover, EmbodiedLGR-Agent was successfully deployed on a physical robot, showing practical utility in real-world contexts through human-robot interaction, while running the visual-language model and the building-retrieval pipeline locally.
>
---
#### [new 057] StableIDM: Stabilizing Inverse Dynamics Model against Manipulator Truncation via Spatio-Temporal Refinement
- **分类: cs.RO**

- **简介: 该论文提出StableIDM，解决机械臂截断导致的逆动力学模型不稳定问题，通过时空优化提升动作预测准确性与任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.17887](https://arxiv.org/pdf/2604.17887)**

> **作者:** Kerui Li; Zhe Jing; Xiaofeng Wang; Zheng Zhu; Yukun Zhou; Guan Huang; Dongze Li; Qingkai Yang; Huaibo Huang
>
> **摘要:** Inverse Dynamics Models (IDMs) map visual observations to low-level action commands, serving as central components for data labeling and policy execution in embodied AI. However, their performance degrades severely under manipulator truncation, a common failure mode that makes state recovery ill-posed and leads to unstable control. We present StableIDM, a spatio-temporal framework that refines features from visual inputs to stabilize action predictions under such partial observability. StableIDM integrates three complementary components: (1) auxiliary robot-centric masking to suppress background clutter, (2) Directional Feature Aggregation (DFA) for geometry-aware spatial reasoning, which extracts anisotropic features along directions inferred from the visible arm and (3) Temporal Dynamics Refinement (TDR) to smooth and correct predictions via motion continuity. Extensive evaluations validate our approach: StableIDM improves strict action accuracy by 12.1% under severe truncation on the AgiBot benchmark, and increases average task success by 9.7% in real-robot replay. Moreover, it boosts end-to-end grasp success by 11.5% when decoding video-generated plans, and improves downstream VLA real-robot success by 17.6% when functioning as an automatic annotator. These results demonstrate that StableIDM provides a robust and scalable backbone for both policy execution and data generation in embodied artificial intelligence.
>
---
#### [new 058] ICAT: Incident-Case-Grounded Adaptive Testing for Physical-Risk Prediction in Embodied World Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于物理风险预测任务，解决世界模型在预测危险时忽略关键线索的问题。提出ICAT方法，通过真实事故数据构建风险记忆，提升风险预测可靠性。**

- **链接: [https://arxiv.org/pdf/2604.16405](https://arxiv.org/pdf/2604.16405)**

> **作者:** Zhenglin Lai; Sirui Huang; Yuteng Li; Changxin Huang; Jianqiang Li; Bingzhe Wu
>
> **摘要:** Video-generative world models are increasingly used as neural simulators for embodied planning and policy learning, yet their ability to predict physical risk and severe consequences is rarely this http URL find that these models often downplay or omit key danger cues and severe outcomes for hazardous actions, which can induce unsafe preferences during planning and training on imagined rollouts. We propose ICAT, which grounds testing in real incident reports and safety manuals by building structured risk memories and retrieving/composing them to constrain the generation of risk cases with causal chains and severity labels. Experiments on an ICAT-based benchmark show that mainstream world models frequently miss mechanisms and triggering conditions and miscalibrate severity, falling short of the reliability required for safety-critical embodied deployment.
>
---
#### [new 059] Autonomous Vehicle Collision Avoidance With Racing Parameterized Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决碰撞避免问题。通过参数化DRL策略，在仿真中实现高效避障，优于传统方法。**

- **链接: [https://arxiv.org/pdf/2604.16702](https://arxiv.org/pdf/2604.16702)**

> **作者:** Shathushan Sivashangaran; Vihaan Dutta; Apoorva Khairnar; Sepideh Gohari; Azim Eskandarian
>
> **摘要:** Road traffic accidents are a leading cause of fatalities worldwide. In the US, human error causes 94% of crashes, resulting in excess of 7,000 pedestrian fatalities and $500 billion in costs annually. Autonomous Vehicles (AVs) with emergency collision avoidance systems that operate at the limits of vehicle dynamics at a high frequency, a dual constraint of nonlinear kinodynamic accuracy and computational efficiency, further enhance safety benefits during adverse weather and cybersecurity breaches, and to evade dangerous human driving when AVs and human drivers share roads. This paper parameterizes a Deep Reinforcement Learning (DRL) collision avoidance policy Out-Of-Distribution (OOD) utilizing race car overtaking, without explicit geometric mimicry reference trajectory guidance, in simulation, with a physics-informed, simulator exploit-aware reward to encode nonlinear vehicle kinodynamics. Two policies are evaluated, a default uni-direction and a reversed heading variant that navigates in the opposite direction to other cars, which both consistently outperform a Model Predictive Control and Artificial Potential Function (MPC-APF) baseline, with zero-shot transfer to proportionally scaled hardware, across three intersection collision scenarios, at 31x fewer Floating Point Operations (FLOPS) and 64x lower inference latency. The reversed heading policy outperforms the default racing overtaking policy in head-to-head collisions by 30% and the baseline by 50%, and matches the former in side collisions, where both DRL policies evade 10% greater than numerical optimal control.
>
---
#### [new 060] Will People Enjoy a Robot Trainer? A Case Study with Snoopie the Pacerbot
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人作为健身教练的有效性问题。研究提出SNOOPIE机器人，通过实验对比验证其在跑步训练中的优势。**

- **链接: [https://arxiv.org/pdf/2604.18331](https://arxiv.org/pdf/2604.18331)**

> **作者:** Maximilian Du; Jennifer Grannen; Shuran Song; Dorsa Sadigh
>
> **备注:** 8 pages, 4 figures. To appear at ICRA 2026
>
> **摘要:** The physicality of exercise makes the role of athletic trainers unique. Their physical presence allows them to guide a student through a motion, demonstrate an exercise, and give intuitive feedback. Robot quadrupeds are also embodied agents with robust agility and athleticism. In our work, we investigate whether a robot quadruped can serve as an effective and enjoyable personal trainer device. We focus on a case study of interval training for runners: a repetitive, long-horizon task where precision and consistency are important. To meet this challenge, we propose SNOOPIE, an autonomous robot quadruped pacer capable of running interval training exercises tailored to challenge a user's personal abilities. We conduct a set of user experiments that compare the robot trainer to a wearable trainer device--the Apple Watch--to investigate the benefits of a physical embodiment in exercise-based interactions. We demonstrate 60.6% better adherence to a pace schedule and were 45.9% more consistent across their running speeds with the quadruped trainer. Subjective results also showed that participants strongly preferred training with the robot over wearable devices across many qualitative axes, including its ease of use (+56.7%), enjoyability of the interaction (+60.6%), and helpfulness (+39.1%). Additional videos and visualizations can be found on our website: this https URL
>
---
#### [new 061] SpaceDex: Generalizable Dexterous Grasping in Tiered Workspaces
- **分类: cs.RO**

- **简介: 该论文属于机械臂抓取任务，旨在解决层级工作空间中灵活抓取的难题。通过分层框架和臂手特征解耦，提升复杂环境下的抓取成功率。**

- **链接: [https://arxiv.org/pdf/2604.17888](https://arxiv.org/pdf/2604.17888)**

> **作者:** Wensheng Wang; Chuanjun Guo; Wei Wei; Tong Wu; Ning Tan
>
> **摘要:** Generalizable grasping with high-degree-of-freedom (DoF) dexterous hands remains challenging in tiered workspaces, where occlusion, narrow clearances, and height-dependent constraints are substantially stronger than in open tabletop scenes. Most existing methods are evaluated in relatively unoccluded settings and typically do not explicitly model the distinct control requirements of arm navigation and hand articulation under spatial constraints. We present SpaceDex, a hierarchical framework for dexterous manipulation in constrained 3D environments. At the high level, a Vision-Language Model (VLM) planner parses user intent, reasons about occlusion and height relations across multiple camera views, and generates target bounding boxes for zero-shot segmentation and mask tracking. This stage provides structured spatial guidance for downstream control instead of relying on single-view target selection. At the low level, we introduce an arm-hand Feature Separation Network that decouples global trajectory control for the arm from geometry-aware grasp mode selection for the hand, reducing feature interference between reaching and grasping objectives. The controller further integrates multi-view perception, fingertip tactile sensing, and a small set of recovery demonstrations to improve robustness to partial observability and off-nominal contacts. In 100 real-world trials involving over 30 unseen objects across four categories, SpaceDex achieves a 63.0\% success rate, compared with 39.0\% for a strong tabletop baseline. These results indicate that combining hierarchical spatial planning with arm-hand representation decoupling improves dexterous grasping performance in spatially constrained environments.
>
---
#### [new 062] Shepherding UAV Swarm with Action Prediction Based on Movement Constraints
- **分类: cs.RO**

- **简介: 该论文属于群体无人机控制任务，旨在解决实际运动约束下无人机群的高效引导问题。通过预测行为并结合运动约束设计控制策略，提升引导效率与安全性。**

- **链接: [https://arxiv.org/pdf/2604.17189](https://arxiv.org/pdf/2604.17189)**

> **作者:** Yusuke Tsunoda; Yusuke Goto; Takao Sato
>
> **摘要:** In this study, we propose a new sheepdog-inspired control method for a swarm of small unmanned aerial vehicles (UAVs), which predicts the swarm behavior while explicitly accounting for the motion constraints of real robots. Sheepdog-inspired guidance control refers to a framework in which a small number of navigator agents (sheepdog agents) indirectly drive a large number of autonomous agents (a flock of sheep agents) so as to steer the group toward a target position. In conventional studies on sheepdog-inspired guidance, both types of agents have typically been modeled as point masses, and the guidance law for the navigator agents has been designed using simple interaction vectors based on the instantaneous relative positions between the agents. However, when implementing such methods on real robots such as drones, it is necessary to consider each agent's motion constraints, including upper bounds on velocity and acceleration. Moreover, we argue that guidance can be made more efficient by predicting the future behavior of the autonomous swarm that is observable to the navigator agents. To this end, we propose a three-dimensional guidance control law based on behavior prediction of autonomous agents under motion constraints, inspired by the Dynamic Window Approach (DWA). At each control cycle, the navigator agent generates a set of feasible motion candidates that satisfy its motion constraints, and predicts the short-horizon swarm evolution using an internal model of the autonomous agents maintained within the navigator agent. The motion candidates are then evaluated according to criteria such as the progress velocity toward the target, the positioning strategy with respect to the swarm, and safety margins, and the optimal motion is selected to achieve safe and efficient guidance. Numerical simulation results demonstrate the effectiveness of the proposed guidance control law.
>
---
#### [new 063] Fisher Decorator: Refining Flow Policy via A Local Transport Map
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于离线强化学习任务，旨在解决流策略在表达性、最优性和效率间的权衡问题。通过引入局部运输映射和Fisher信息矩阵，实现更精确的策略优化。**

- **链接: [https://arxiv.org/pdf/2604.17919](https://arxiv.org/pdf/2604.17919)**

> **作者:** Xiaoyuan Cheng; Haoyu Wang; Wenxuan Yuan; Ziyan Wang; Zonghao Chen; Li Zeng; Zhuo Sun
>
> **摘要:** Recent advances in flow-based offline reinforcement learning (RL) have achieved strong performance by parameterizing policies via flow matching. However, they still face critical trade-offs among expressiveness, optimality, and efficiency. In particular, existing flow policies interpret the $L_2$ regularization as an upper bound of the 2-Wasserstein distance ($W_2$), which can be problematic in offline settings. This issue stems from a fundamental geometric mismatch: the behavioral policy manifold is inherently anisotropic, whereas the $L_2$ (or upper bound of $W_2$) regularization is isotropic and density-insensitive, leading to systematically misaligned optimization directions. To address this, we revisit offline RL from a geometric perspective and show that policy refinement can be formulated as a local transport map: an initial flow policy augmented by a residual displacement. By analyzing the induced density transformation, we derive a local quadratic approximation of the KL-constrained objective governed by the Fisher information matrix, enabling a tractable anisotropic optimization formulation. By leveraging the score function embedded in the flow velocity, we obtain a corresponding quadratic constraint for efficient optimization. Our results reveal that the optimality gap in prior methods arises from their isotropic approximation. In contrast, our framework achieves a controllable approximation error within a provable neighborhood of the optimal solution. Extensive experiments demonstrate state-of-the-art performance across diverse offline RL benchmarks. See project page: this https URL.
>
---
#### [new 064] ScenarioControl: Vision-Language Controllable Vectorized Latent Scenario Generation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ScenarioControl，用于生成可控的驾驶场景。属于场景生成任务，解决如何根据文本或图像生成多样化、真实的3D场景问题。通过向量空间和跨模态控制实现精细场景生成。**

- **链接: [https://arxiv.org/pdf/2604.17147](https://arxiv.org/pdf/2604.17147)**

> **作者:** Lili Gao; Yanbo Xu; William Koch; Samuele Ruffino; Luke Rowe; Behdad Chalaki; Dmitriy Rivkin; Julian Ost; Roger Girgis; Mario Bijelic; Felix Heide
>
> **摘要:** We introduce ScenarioControl, the first vision-language control mechanism for learned driving scenario generation. Given a text prompt or an input image, Scenario-Control synthesizes diverse, realistic 3D scenario rollouts - including map, 3D boxes of reactive actors over time, pedestrians, driving infrastructure, and ego camera observations. The method generates scenes in a vectorized latent space that represents road structure and dynamic agents jointly. To connect multimodal control with sparse vectorized scene elements, we propose a cross-global control mechanism that integrates crossattention with a lightweight global-context branch, enabling fine-grained control over road layout and traffic conditions while preserving realism. The method produces temporally consistent scenario rollouts from the perspectives different actors in the scene, supporting long-horizon continuation of driving scenarios. To facilitate training and evaluation, we release a dataset with text annotations aligned to vectorized map structures. Extensive experiments validate that the control adherence and fidelity of ScenarioControl compare favorable to all tested methods across all experiments. Project webpage: this https URL
>
---
#### [new 065] LAGS: Low-Altitude Gaussian Splatting with Groupwise Heterogeneous Graph Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景重建任务，解决低空无人机图像资源分配效率问题。提出GW-HGNN模型，平衡数据质量和传输成本。**

- **链接: [https://arxiv.org/pdf/2604.16910](https://arxiv.org/pdf/2604.16910)**

> **作者:** Yikun Wang; Yujie Wan; Wei Zuo; Shuai Wang; Yik-Chung Wu; Chengzhong Xu; Huseyin Arslan
>
> **备注:** 5 pages, 8 figures
>
> **摘要:** Low-altitude Gaussian splatting (LAGS) facilitates 3D scene reconstruction by aggregating aerial images from distributed drones. However, as LAGS prioritizes maximizing reconstruction quality over communication throughput, existing low-altitude resource allocation schemes become inefficient. This inefficiency stems from their failure to account for image diversity introduced by varying viewpoints. To fill this gap, we propose a groupwise heterogeneous graph neural network (GW-HGNN) for LAGS resource allocation. GW-HGNN explicitly models the non-uniform contribution of different image groups to the reconstruction process, thus automatically balancing data fidelity and transmission cost. The key insight of GW-HGNN is to transform LAGS losses and communication constraints into graph learning costs for dual-level message passing. Experiments on real-world LAGS datasets demonstrate that GW-HGNN significantly outperforms state-of-the-art benchmarks across key rendering metrics, including PSNR, SSIM, and LPIPS. Furthermore, GW-HGNN reduces computational latency by approximately 100x compared to the widely-used MOSEK solver, achieving millisecond-level inference suitable for real-time deployment.
>
---
#### [new 066] Does "Do Differentiable Simulators Give Better Policy Gradients?'' Give Better Policy Gradients?
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决政策梯度估计中的偏差问题。通过引入DDCG和IVW-H方法，提升梯度估计的稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2604.18161](https://arxiv.org/pdf/2604.18161)**

> **作者:** Ku Onoda; Paavo Parmas; Manato Yaguchi; Yutaka Matsuo
>
> **备注:** ICLR2026
>
> **摘要:** In policy gradient reinforcement learning, access to a differentiable model enables 1st-order gradient estimation that accelerates learning compared to relying solely on derivative-free 0th-order estimators. However, discontinuous dynamics cause bias and undermine the effectiveness of 1st-order estimators. Prior work addressed this bias by constructing a confidence interval around the REINFORCE 0th-order gradient estimator and using these bounds to detect discontinuities. However, the REINFORCE estimator is notoriously noisy, and we find that this method requires task-specific hyperparameter tuning and has low sample efficiency. This paper asks whether such bias is the primary obstacle and what minimal fixes suffice. First, we re-examine standard discontinuous settings from prior work and introduce DDCG, a lightweight test that switches estimators in nonsmooth regions; with a single hyperparameter, DDCG achieves robust performance and remains reliable with small samples. Second, on differentiable robotics control tasks, we present IVW-H, a per-step inverse-variance implementation that stabilizes variance without explicit discontinuity detection and yields strong results. Together, these findings indicate that while estimator switching improves robustness in controlled studies, careful variance control often dominates in practical deployments.
>
---
#### [new 067] Can Explicit Physical Feasibility Benefit VLA Learning? An Empirical Study
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）学习任务，旨在解决VLA模型在物理可行性上的不足。通过引入显式可行性监督，提升模型的物理可靠性和任务性能。**

- **链接: [https://arxiv.org/pdf/2604.17896](https://arxiv.org/pdf/2604.17896)**

> **作者:** Yubai Wei; Chen Wu; Hashem Haghbayan
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Vision-Language-Action (VLA) models map multimodal inputs directly to robot actions and are typically trained through large-scale imitation learning. While this paradigm has shown strong performance, prevailing VLA training procedures do not explicitly supervise hard physical constraints such as obstacle avoidance or kinematic feasibility. As a result, the geometric structure underlying physically feasible behavior must be inferred only implicitly from demonstrations. In this paper, we study whether introducing explicit feasibility supervision can provide effective structured guidance for VLA policies. We formulate a simple geometry-grounded feasibility objective and integrate it into the training stage of a diffusion-based VLA policy. To evaluate this idea systematically, we use obstacle-aware manipulation as a controlled probe of geometry-dependent physical feasibility. Empirical results show that augmenting VLA training with feasibility supervision improves both physical reliability and overall task performance, while also enhancing learning efficiency in the low-data regime. These findings indicate that explicit feasibility signals can effectively complement imitation-based VLA learning, highlighting their potential for developing more reliable VLA policies.
>
---
#### [new 068] Safe Control using Learned Safety Filters and Adaptive Conformal Inference
- **分类: eess.SY; cs.LG; cs.RO**

- **简介: 该论文属于安全控制任务，旨在解决安全过滤器可靠性不足的问题。通过结合学习方法与自适应共形推理，提出ACoFi方法，动态调整安全切换标准，提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2604.18482](https://arxiv.org/pdf/2604.18482)**

> **作者:** Sacha Huriot; Ihab Tabbara; Hussein Sibai
>
> **备注:** Accepted to L4DC 2026
>
> **摘要:** Safety filters have been shown to be effective tools to ensure the safety of control systems with unsafe nominal policies. To address scalability challenges in traditional synthesis methods, learning-based approaches have been proposed for designing safety filters for systems with high-dimensional state and control spaces. However, the inevitable errors in the decisions of these models raise concerns about their reliability and the safety guarantees they offer. This paper presents Adaptive Conformal Filtering (ACoFi), a method that combines learned Hamilton-Jacobi reachability-based safety filters with adaptive conformal inference. Under ACoFi, the filter dynamically adjusts its switching criteria based on the observed errors in its predictions of the safety of actions. The range of possible safety values of the nominal policy's output is used to quantify uncertainty in safety assessment. The filter switches from the nominal policy to the learned safe one when that range suggests it might be unsafe. We show that ACoFi guarantees that the rate of incorrectly quantifying uncertainty in the predicted safety of the nominal policy is asymptotically upper bounded by a user-defined parameter. This gives a soft safety guarantee rather than a hard safety guarantee. We evaluate ACoFi in a Dubins car simulation and a Safety Gymnasium environment, empirically demonstrating that it significantly outperforms the baseline method that uses a fixed switching threshold by achieving higher learned safety values and fewer safety violations, especially in out-of-distribution scenarios.
>
---
#### [new 069] A Comparative Evaluation of Geometric Accuracy in NeRF and Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，旨在解决神经渲染方法在几何精度上的评估问题。通过构建评估流程和基准数据集，系统评估重建方法的表面和形状保真度。**

- **链接: [https://arxiv.org/pdf/2604.18205](https://arxiv.org/pdf/2604.18205)**

> **作者:** Mikolaj Zielinski; Eryk Vykysaly; Bartlomiej Biesiada; Jan Baturo; Mateusz Capala; Dominik Belter
>
> **摘要:** Recent advances in neural rendering have introduced numerous 3D scene representations. Although standard computer vision metrics evaluate the visual quality of generated images, they often overlook the fidelity of surface geometry. This limitation is particularly critical in robotics, where accurate geometry is essential for tasks such as grasping and object manipulation. In this paper, we present an evaluation pipeline for neural rendering methods that focuses on geometric accuracy, along with a benchmark comprising 19 diverse scenes. Our approach enables a systematic assessment of reconstruction methods in terms of surface and shape fidelity, complementing traditional visual metrics.
>
---
#### [new 070] Continuous Focus Groups: A Longitudinal Method for Clinical HRI in Autism Care
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互研究，旨在解决临床场景中用户参与度低、观点变化难以捕捉的问题。通过连续焦点小组方法，持续收集参与者反馈，促进设计与治疗的协同迭代。**

- **链接: [https://arxiv.org/pdf/2604.18197](https://arxiv.org/pdf/2604.18197)**

> **作者:** Ghiglino Davide; Foglino Caterina; Wykowska Agnieszka
>
> **摘要:** Qualitative methods are important to use alongside quantitative methods to improve Human-Robot Interaction (HRI), yet they are often applied in static or one-off formats that cannot capture how stakeholder perspectives evolve over time. This limitation is especially evident in clinical contexts, where families and patients face heavy burdens and cannot easily participate in repeated research encounters. To address this gap, we introduce continuous focus groups, a longitudinal and co-agential method designed to sustain dialogue with assistive care professionals working with children with autism spectrum disorder (ASD). Three focus groups were organized across successive phases of a robot-assisted therapeutic protocol, enabling participants to revisit and refine earlier views as the intervention progressed. Results show that continuity fostered trust, supported the integration of tacit clinical expertise into design decisions, and functioned as an ethical safeguard by allowing participants to renegotiate involvement and surface new concerns. By bridging the therapeutic iteration of families, children, and clinicians with the research-design iteration of researchers and developers, continuous focus groups provide a methodological contribution that is both feasible in practice and rigorous in design. Beyond autism care, this approach offers a transferable framework for advancing qualitative research in HRI, particularly in sensitive domains where direct user participation is limited and continuity is essential.
>
---
#### [new 071] XEmbodied: A Foundation Model with Enhanced Geometric and Physical Cues for Large-Scale Embodied Environments
- **分类: cs.CV; cs.MM; cs.RO**

- **简介: 该论文提出XEmbodied，解决VLA模型在复杂环境中的几何与物理理解不足问题，通过引入3D几何和物理信号提升其性能。**

- **链接: [https://arxiv.org/pdf/2604.18484](https://arxiv.org/pdf/2604.18484)**

> **作者:** Kangan Qian; ChuChu Xie; Yang Zhong; Jingrui Pang; Siwen Jiao; Sicong Jiang; Zilin Huang; Yunlong Wang; Kun Jiang; Mengmeng Yang; Hao Ye; Guanghao Zhang; Hangjun Ye; Guang Chen; Long Chen; Diange Yang
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Vision-Language-Action (VLA) models drive next-generation autonomous systems, but training them requires scalable, high-quality annotations from complex environments. Current cloud pipelines rely on generic vision-language models (VLMs) that lack geometric reasoning and domain semantics due to their 2D image-text pretraining. To address this mismatch, we propose XEmbodied, a cloud-side foundation model that endows VLMs with intrinsic 3D geometric awareness and interaction with physical cues (e.g., occupancy grids, 3D boxes). Instead of treating geometry as auxiliary input, XEmbodied integrates geometric representations via a structured 3D Adapter and distills physical signals into context tokens using an Efficient Image-Embodied Adapter. Through progressive domain curriculum and reinforcement learning post-training, XEmbodied preserves general capabilities while demonstrating robust performance across 18 public benchmarks. It significantly improves spatial reasoning, traffic semantics, embodied affordance, and out-of-distribution generalization for large-scale scenario mining and embodied VQA.
>
---
#### [new 072] A Survey of Spatial Memory Representations for Efficient Robot Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人导航任务，解决空间记忆效率问题。通过分析88篇文献，提出评估指标α，揭示内存架构对部署可行性的影响，并设计标准化评估方法。**

- **链接: [https://arxiv.org/pdf/2604.16482](https://arxiv.org/pdf/2604.16482)**

> **作者:** Ma. Madecheen S. Pangaliman; Steven S. Sison; Erwin P. Quilloy; Rowel Atienza
>
> **备注:** Accepted at the Women in Computer Vision (WiCV) Workshop at CVPR 2026
>
> **摘要:** As vision-based robots navigate larger environments, their spatial memory grows without bound, eventually exhausting computational resources, particularly on embedded platforms (8-16GB shared memory, $<$30W) where adding hardware is not an option. This survey examines the spatial memory efficiency problem across 88 references spanning 52 systems (1989-2025), from occupancy grids to neural implicit representations. We introduce the $\alpha = M_{\text{peak}} / M_{\text{map}}$, the ratio of peak runtime memory (the total RAM or GPU memory consumed during operation) to saved map size (the persistent checkpoint written to disk), exposing the gap between published map sizes and actual deployment cost. Independent profiling on an NVIDIA A100 GPU reveals that $\alpha$ spans two orders of magnitude within neural methods alone, ranging from 2.3 (Point-SLAM) to 215 (NICE-SLAM, whose 47,MB map requires 10GB at runtime), showing that memory architecture, not paradigm label, determines deployment feasibility. We propose a standardized evaluation protocol comprising memory growth rate, query latency, memory-completeness curves, and throughput degradation, none of which current benchmarks capture. Through a Pareto frontier analysis with explicit benchmark separation, we show that no single paradigm dominates within its evaluation regime: 3DGS methods achieve the best absolute accuracy at 90-254,MB map size on Replica, while scene graphs provide semantic abstraction at predictable cost. We provide the first independently measured $\alpha$ reference values and an $\alpha$-aware budgeting algorithm enabling practitioners to assess deployment feasibility on target hardware prior to implementation.
>
---
#### [new 073] BOIL: Learning Environment Personalized Information
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出BOIL方法，用于多智能体系统在复杂环境中的信息提取与策略生成。解决如何从有限信息中获取有价值洞察的问题，通过Pagerank和信息最大化实现高效策略优化。**

- **链接: [https://arxiv.org/pdf/2604.17137](https://arxiv.org/pdf/2604.17137)**

> **作者:** Rohan Patil; Henrik I. Christensen
>
> **摘要:** Navigating complex environments poses challenges for multi-agent systems, requiring efficient extraction of insights from limited information. In this paper, we introduce the Blackbox Oracle Information Learning (BOIL) process, a scalable solution for extracting valuable insights from the environment structure. Leveraging the Pagerank algorithm and common information maximization, BOIL facilitates the extraction of information to guide long-term agent behavior applicable to problems such as coverage, patrolling, and stochastic reachability. Through experiments, we demonstrate the efficacy of BOIL in generating strategy distributions conducive to improved performance over extended time horizons, surpassing heuristic approaches in complex environments.
>
---
#### [new 074] Heterogeneous Self-Play for Realistic Highway Traffic Simulation
- **分类: cs.AI; cs.LG; cs.MA; cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，解决高速公路场景生成难题。通过PHASE框架，实现可控、真实且覆盖广泛的交通模拟，提升多车交互的可信度。**

- **链接: [https://arxiv.org/pdf/2604.16406](https://arxiv.org/pdf/2604.16406)**

> **作者:** Jinkai Qiu; Alessandro Saviolo; Chaojie Wang; Mingke Wang; Xiaoyu Huang
>
> **备注:** 8 pages, 2026 CVPR SAD Workshop
>
> **摘要:** Realistic highway simulation is critical for scalable safety evaluation of autonomous vehicles, particularly for interactions that are too rare to study from logged data alone. Yet highway traffic generation remains challenging because it requires broad coverage across speeds and maneuvers, controllable generation of rare safety-critical scenarios, and behavioral credibility in multi-agent interactions. We present PHASE, Policy for Heterogeneous Agent Self-play on Expressway, a context-aware self-play framework that addresses these three requirements through explicit per-agent conditioning for controllability, synthetic scenario generation for broad highway coverage, and closed-loop multi-agent training for realistic interaction dynamics. PHASE further supports different vehicle profiles, for example, passenger cars and articulated trailer trucks, within a single policy via vehicle-aware dynamics and context-conditioned actions, and stabilizes self-play with early termination of unrecoverable states, at-fault collision attribution, highway-aware reward shaping, coupled curricula, and robust policy optimization. Despite being trained only on synthetic data, PHASE transfers zero-shot to 512 unseen high-interaction real scenarios in exiD, achieving a 96.3% success rate and reducing ADE/FDE from 6.57/12.07 m to 2.44/5.25 m relative to a prior self-play baseline. In a learned trajectory embedding space, it also improves behavioral realism over IDM, reducing Frechet trajectory distance by 13.1% and energy distance by 20.2%. These results show that synthetic self-play can provide a scalable route to controllable and realistic highway scenario generation without direct imitation of expert logs.
>
---
#### [new 075] Seeing Isn't Believing: Mitigating Belief Inertia via Active Intervention in Embodied Agents
- **分类: cs.CL; cs.AI; cs.RO**

- **简介: 该论文属于 embodied agents 任务，解决 agents 因固有信念导致的决策失误问题。提出 EVU 机制，通过预测、验证和更新信念，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.17252](https://arxiv.org/pdf/2604.17252)**

> **作者:** Hanlin Wang; Chak Tou Leong; Jian Wang; Wenjie Li
>
> **备注:** Accepted by ACL2026 Fingdings
>
> **摘要:** Recent advancements in large language models (LLMs) have enabled agents to tackle complex embodied tasks through environmental interaction. However, these agents still make suboptimal decisions and perform ineffective actions, as they often overlook critical environmental feedback that differs from their internal beliefs. Through a formal probing analysis, we characterize this as belief inertia, a phenomenon where agents stubbornly adhere to prior beliefs despite explicit observations. To address this, we advocate active belief intervention, moving from passive understanding to active management. We introduce the Estimate-Verify-Update (EVU) mechanism, which empowers agents to predict expected outcomes, verify them against observations through explicit reasoning, and actively update prior beliefs based on the verification evidence. EVU is designed as a unified intervention mechanism that generates textual belief states explicitly, and can be integrated into both prompting-based and training-based agent reasoning methods. Extensive experiments across three embodied benchmarks demonstrate that EVU consistently yields substantial gains in task success rates. Further analyses validate that our approach effectively mitigates belief inertia, advancing the development of more robust embodied agents. Our code is available at this https URL.
>
---
#### [new 076] Fringe Projection Based Vision Pipeline for Autonomous Hard Drive Disassembly
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自主硬盘拆解的视觉任务，解决传统方法在3D感知和螺钉定位上的不足。提出一种基于条纹投影的视觉流水线，结合深度补全与实例分割，实现高效精准的拆解感知。**

- **链接: [https://arxiv.org/pdf/2604.17231](https://arxiv.org/pdf/2604.17231)**

> **作者:** Badrinath Balasubramaniam; Vignesh Suresh; Benjamin Metcalf; Beiwen Li
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Unrecovered e-waste represents a significant economic loss. Hard disk drives (HDDs) comprise a valuable e-waste stream necessitating robotic disassembly. Automating the disassembly of HDDs requires holistic 3D sensing, scene understanding, and fastener localization, however current methods are fragmented, lack robust 3D sensing, and lack fastener localization. We propose an autonomous vision pipeline which performs 3D sensing using a Fringe Projection Profilometry (FPP) module, with selective triggering of a depth completion module where FPP fails, and integrates this module with a lightweight, real-time instance segmentation network for scene understanding and critical component localization. By utilizing the same FPP camera-projector system for both our depth sensing and component localization modules, our depth maps and derived 3D geometry are inherently pixel-wise aligned with the segmentation masks without registration, providing an advantage over RGB-D perception systems common in industrial sensing. We optimize both our trained depth completion and instance segmentation networks for deployment-oriented inference. The proposed system achieves a box mAP@50 of 0.960 and mask mAP@50 of 0.957 for instance segmentation, while the selected depth completion configuration with the Depth Anything V2 Base backbone achieves an RMSE of 2.317 mm and MAE of 1.836 mm; the Platter Facing learned inference stack achieved a combined latency of 12.86 ms and a throughput of 77.7 Frames Per Second (FPS) on the evaluation workstation. Finally, we adopt a sim-to-real transfer learning approach to augment our physical dataset. The proposed perception pipeline provides both high-fidelity semantic and spatial data which can be valuable for downstream robotic disassembly. The synthetic dataset developed for HDD instance segmentation will be made publicly available.
>
---
#### [new 077] Using large language models for embodied planning introduces systematic safety risks
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于机器人规划安全研究，旨在评估大语言模型在具身规划中的安全性。工作包括构建基准测试DESPITE，分析模型规划能力与安全意识的关系，发现规模增大提升规划能力但安全意识提升有限。**

- **链接: [https://arxiv.org/pdf/2604.18463](https://arxiv.org/pdf/2604.18463)**

> **作者:** Tao Zhang; Kaixian Qu; Zhibin Li; Jiajun Wu; Marco Hutter; Manling Li; Fan Shi
>
> **备注:** Project page: this https URL
>
> **摘要:** Large language models are increasingly used as planners for robotic systems, yet how safely they plan remains an open question. To evaluate safe planning systematically, we introduce DESPITE, a benchmark of 12,279 tasks spanning physical and normative dangers with fully deterministic validation. Across 23 models, even near-perfect planning ability does not ensure safety: the best-planning model fails to produce a valid plan on only 0.4% of tasks but produces dangerous plans on 28.3%. Among 18 open-source models from 3B to 671B parameters, planning ability improves substantially with scale (0.4-99.3%) while safety awareness remains relatively flat (38-57%). We identify a multiplicative relationship between these two capacities, showing that larger models complete more tasks safely primarily through improved planning, not through better danger avoidance. Three proprietary reasoning models reach notably higher safety awareness (71-81%), while non-reasoning proprietary models and open-source reasoning models remain below 57%. As planning ability approaches saturation for frontier models, improving safety awareness becomes a central challenge for deploying language-model planners in robotic systems.
>
---
#### [new 078] OneVL: One-Step Latent Reasoning and Planning with Vision-Language Explanation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文提出OneVL，解决VLA自动驾驶中轨迹预测的延迟问题。通过融合视觉-语言解释的潜在推理与世界模型，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.18486](https://arxiv.org/pdf/2604.18486)**

> **作者:** Jinghui Lu; Jiayi Guan; Zhijian Huang; Jinlong Li; Guang Li; Lingdong Kong; Yingyan Li; Han Wang; Shaoqing Xu; Yuechen Luo; Fang Li; Chenxu Dang; Junli Wang; Tao Xu; Jing Wu; Jianhua Wu; Xiaoshuai Hao; Wen Zhang; Tianyi Jiang; Lingfeng Zhang; Lei Zhou; Yingbo Tang; Jie Wang; Yinfeng Gao; Xizhou Bu; Haochen Tian; Yihang Qiu; Feiyang Jia; Lin Liu; Yigu Ge; Hanbing Li; Yuannan Shen; Jianwei Cui; Hongwei Xie; Bing Wang; Haiyang Sun; Jingwei Zhao; Jiahui Huang; Pei Liu; Zeyu Zhu; Yuncheng Jiang; Zibin Guo; Chuhong Gong; Hanchao Leng; Kun Ma; Naiyang Wang; Guang Chen; Kuiyuan Yang; Hangjun Ye; Long Chen
>
> **备注:** Technical Report; 49 pages, 22 figures, 10 tables; Project Page at this https URL
>
> **摘要:** Chain-of-Thought (CoT) reasoning has become a powerful driver of trajectory prediction in VLA-based autonomous driving, yet its autoregressive nature imposes a latency cost that is prohibitive for real-time deployment. Latent CoT methods attempt to close this gap by compressing reasoning into continuous hidden states, but consistently fall short of their explicit counterparts. We suggest that this is due to purely linguistic latent representations compressing a symbolic abstraction of the world, rather than the causal dynamics that actually govern driving. Thus, we present OneVL (One-step latent reasoning and planning with Vision-Language explanations), a unified VLA and World Model framework that routes reasoning through compact latent tokens supervised by dual auxiliary decoders. Alongside a language decoder that reconstructs text CoT, we introduce a visual world model decoder that predicts future-frame tokens, forcing the latent space to internalize the causal dynamics of road geometry, agent motion, and environmental change. A three-stage training pipeline progressively aligns these latents with trajectory, language, and visual objectives, ensuring stable joint optimization. At inference, the auxiliary decoders are discarded and all latent tokens are prefilled in a single parallel pass, matching the speed of answer-only prediction. Across four benchmarks, OneVL becomes the first latent CoT method to surpass explicit CoT, delivering state-of-the-art accuracy at answer-only latency, and providing direct evidence that tighter compression, when guided in both language and world-model supervision, produces more generalizable representations than verbose token-by-token reasoning. Project Page: this https URL
>
---
#### [new 079] Rule-VLN: Bridging Perception and Compliance via Semantic Reasoning and Geometric Rectification
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决现有代理过度关注物理路径而忽视规则约束的问题。提出Rule-VLN基准和SNRM模块，提升导航的合规性与安全性。**

- **链接: [https://arxiv.org/pdf/2604.16993](https://arxiv.org/pdf/2604.16993)**

> **作者:** Jiawen Wen; Penglei Sun; Wenjie Zhang; Suixuan Qiu; Weisheng Xu; Xiaofei Yang; Xiaowen Chu
>
> **摘要:** As embodied AI transitions to real-world deployment, the success of the Vision-and-Language Navigation (VLN) task tends to evolve from mere reachability to social compliance. However, current agents suffer from a "goal-driven trap", prioritizing physical geometry ("can I go?") over semantic rules ("may I go?"), frequently overlooking subtle regulatory constraints. To bridge this gap, we establish Rule-VLN, the first large-scale urban benchmark for rule-compliant navigation. Spanning a massive 29k-node environment, it injects 177 diverse regulatory categories into 8k constrained nodes across four curriculum levels, challenging agents with fine-grained visual and behavioral constraints. We further propose the Semantic Navigation Rectification Module (SNRM), a universal, zero-shot module designed to equip pre-trained agents with safety awareness. SNRM integrates a coarse-to-fine visual perception VLM framework with an epistemic mental map for dynamic detour planning. Experiments demonstrate that while Rule-VLN challenges state-of-the-art models, SNRM significantly restores navigation capabilities, reducing CVR by 19.26% and boosting TC by 5.97%.
>
---
#### [new 080] Positive-Only Drifting Policy Optimization
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于在线强化学习任务，旨在解决传统策略在表达能力、梯度裁剪和信任区域上的限制。提出PODPO方法，通过仅使用正优势样本进行策略更新，提升性能并预防错误。**

- **链接: [https://arxiv.org/pdf/2604.16519](https://arxiv.org/pdf/2604.16519)**

> **作者:** Qi Zhang
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** In the field of online reinforcement learning (RL), traditional Gaussian policies and flow-based methods are often constrained by their unimodal expressiveness, complex gradient clipping, or stringent trust-region requirements. Moreover, they all rely on post-hoc penalization of negative samples to correct erroneous actions. This paper introduces Positive-Only Drifting Policy Optimization (PODPO), a likelihood-free and gradient-clipping-free generative approach for online RL. By leveraging the drifting model, PODPO performs policy updates via advantage-weighted local contrastive drifting. Relying solely on positive-advantage samples, it elegantly steers actions toward high-return regions while exploiting the inherent local smoothness of the generative model to enable proactive error prevention. In doing so, PODPO opens a promising new pathway for generative policy learning in online settings.
>
---
#### [new 081] Infrastructure-Centric World Models: Bridging Temporal Depth and Spatial Breadth for Roadside Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出基础设施中心的世界模型（I-WM），解决自动驾驶中环境感知不足问题，通过融合多传感器数据实现时空互补的交通预测与理解。**

- **链接: [https://arxiv.org/pdf/2604.17651](https://arxiv.org/pdf/2604.17651)**

> **作者:** Siyuan Meng; Chengbo Ai
>
> **备注:** 18 pages, 7 tables, 1 figure, vision paper
>
> **摘要:** World models, generative AI systems that simulate how environments evolve, are transforming autonomous driving, yet all existing approaches adopt an ego-vehicle perspective, leaving the infrastructure viewpoint unexplored. We argue that infrastructure-centric world models offer a fundamentally complementary capability: the bird's-eye, multi-sensor, persistent viewpoint that roadside systems uniquely possess. Central to our thesis is a spatio-temporal complementarity: fixed roadside sensors excel at temporal depth, accumulating long-term behavioral distributions including rare safety-critical events, while vehicle-borne sensors excel at spatial breadth, sampling diverse scenes across large road networks. This paper presents a vision for Infrastructure-centric World Models (I-WM) in three phases: (I) generative scene understanding with quality-aware uncertainty propagation, (II) physics-informed predictive dynamics with multi-agent counterfactual reasoning, and (III) collaborative world models for V2X communication via latent space alignment. We propose a dual-layer architecture, annotation-free perception as a multi-modal data engine feeding end-to-end generative world models, with a phased sensor strategy from LiDAR through 4D radar and signal phase data to event cameras. We establish a taxonomy of driving world model paradigms, position I-WM relative to LeCun's JEPA, Li Fei-Fei's spatial intelligence, and VLA architectures, and introduce Infrastructure VLA (I-VLA) as a novel unification of roadside perception, language commands, and traffic control actions. Our vision builds upon existing multi-LiDAR pipelines and identifies open-source foundations for each phase, providing a path toward infrastructure that understands and anticipates traffic.
>
---
#### [new 082] Re$^2$MoGen: Open-Vocabulary Motion Generation via LLM Reasoning and Physics-Aware Refinement
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于文本到动作生成任务，解决开放词汇下运动生成的物理合理性问题。通过LLM推理和物理增强的强化学习优化，生成语义一致且物理可信的运动。**

- **链接: [https://arxiv.org/pdf/2604.17807](https://arxiv.org/pdf/2604.17807)**

> **作者:** Jiakun Zheng; Ting Xiao; Shiqin Cao; Xinran Li; Zhe Wang; Chenjia Bai
>
> **摘要:** Text-to-motion (T2M) generation aims to control the behavior of a target character via textual descriptions. Leveraging text-motion paired datasets, existing T2M models have achieved impressive performance in generating high-quality motions within the distribution of their training data. However, their performance deteriorates notably when the motion descriptions differ significantly from the training texts. To address this issue, we propose Re$^2$MoGen, a Reasoning and Refinement open-vocabulary Motion Generation framework that leverages enhanced Large Language Model (LLM) reasoning to generate an initial motion planning and then refine its physical plausibility via reinforcement learning (RL) post-training. Specifically, Re$^2$MoGen consists of three stages: We first employ Monte Carlo tree search to enhance the LLM's reasoning ability in generating reasonable keyframes of the motion based on text prompts, specifying only the root and several key joints' positions to ease the reasoning process. Then, we apply a human pose model as a prior to optimize the full-body poses based on the planned keyframes and use the resulting incomplete motion to supervise fine-tuning a pre-trained motion generator via a dynamic temporal matching objective, enabling spatiotemporal completion. Finally, we use post-training with physics-aware reward to refine motion quality to eliminate physical implausibility in LLM-planned motions. Extensive experiments demonstrate that our framework can generate semantically consistent and physically plausible motions and achieve state-of-the-art performance in open-vocabulary motion generation.
>
---
## 更新

#### [replaced 001] BOP-ASK: Object-Interaction Reasoning for Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BOP-ASK数据集，用于视觉语言模型的对象交互推理任务，解决现有基准在精细空间理解上的不足。**

- **链接: [https://arxiv.org/pdf/2511.16857](https://arxiv.org/pdf/2511.16857)**

> **作者:** Vineet Bhat; Sungsu Kim; Valts Blukis; Greg Heinrich; Prashanth Krishnamurthy; Ramesh Karri; Stan Birchfield; Farshad Khorrami; Jonathan Tremblay
>
> **备注:** Accepted at CVPR 2026. Code, Datasets & Benchmark available at this https URL
>
> **摘要:** Vision Language Models (VLMs) have achieved impressive performance on spatial reasoning benchmarks, yet these evaluations mask critical weaknesses in understanding object interactions. Current benchmarks test high level relationships ('left of,' 'behind', etc.) but ignore fine-grained spatial understanding needed for real world applications: precise 3D localization, physical compatibility between objects, object affordances and multi step spatial planning. In this work, we present BOP-ASK, a novel large scale dataset for object interaction reasoning for both training and benchmarking. Our data generation pipeline leverages 6D object poses from the Benchmark for Object Pose Estimation (BOP) datasets from which we derive fine grained annotations such as grasp poses, referred object poses, path planning trajectories, relative spatial and depth relationships, and object-to-object relationships. BOP-ASK comprises over 150k images and 33M question answer pairs spanning six tasks (four novel), providing a rich resource for training and evaluating VLMs. We evaluate proprietary and open sourced VLMs, and conduct human evaluations on BOP-ASK-core, a contributed test benchmark. We also release BOP-ASK-lab, an out-of-distribution benchmark with images not sourced from BOP, enabling testing of generalization. Our experiments demonstrate that models trained on BOP-ASK outperform baselines and exhibit emergent capabilities such as precise object and grasp pose estimation, trajectory planning, and fine-grained object-centric spatial reasoning in cluttered environments.
>
---
#### [replaced 002] Real-Time Structural Detection for Indoor Navigation from 3D LiDAR Using Bird's-Eye-View Images
- **分类: cs.RO**

- **简介: 该论文属于室内导航中的结构检测任务，旨在解决资源受限机器人实时高效感知问题。通过将3D LiDAR数据转为2D BEV图像，结合YOLO-OBB实现快速准确的结构检测。**

- **链接: [https://arxiv.org/pdf/2603.19830](https://arxiv.org/pdf/2603.19830)**

> **作者:** Guanliang Li; Pedro Espinosa-Angulo; David Perez-Saura; Santiago Tapia-Fernandez
>
> **摘要:** Efficient structural perception is essential for mapping and autonomous navigation on resource-constrained robots. Existing 3D methods are computationally prohibitive, while traditional 2D geometric approaches lack robustness. This paper presents a lightweight, real-time framework that projects 3D LiDAR data into 2D Bird's-Eye-View (BEV) images to enable efficient detection of structural elements relevant to mapping and navigation. Within this representation, we systematically evaluate several feature extraction strategies, including classical geometric techniques (Hough Transform, RANSAC, and LSD) and a deep learning detector based on YOLO-OBB. The resulting detections are integrated through a spatiotemporal fusion module that improves stability and robustness across consecutive frames. Experiments conducted on a standard mobile robotic platform highlight clear performance trade-offs. Classical methods such as Hough and LSD provide fast responses but exhibit strong sensitivity to noise, with LSD producing excessive segment fragmentation that leads to system congestion. RANSAC offers improved robustness but fails to meet real-time constraints. In contrast, the YOLO-OBB-based approach achieves the best balance between robustness and computational efficiency, maintaining an end-to-end latency (satisfying 10 Hz operation) while effectively filtering cluttered observations in a low-power single-board computer (SBC) without using GPU acceleration. The main contribution of this work is a computationally efficient BEV-based perception pipeline enabling reliable real-time structural detection from 3D LiDAR on resource-constrained robotic platforms that cannot rely on GPU-intensive processing. The source code and pre-trained models are publicly available.
>
---
#### [replaced 003] End-to-end Listen, Look, Speak and Act
- **分类: cs.AI; cs.CL; cs.CV; cs.RO; eess.AS**

- **简介: 该论文提出ELLSA模型，解决多模态交互任务，实现语音、视觉、文本和动作的同步感知与生成，支持自然的人机互动。**

- **链接: [https://arxiv.org/pdf/2510.16756](https://arxiv.org/pdf/2510.16756)**

> **作者:** Siyin Wang; Wenyi Yu; Xianzhao Chen; Xiaohai Tian; Jun Zhang; Lu Lu; Chao Zhang
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Human interaction is inherently multimodal and full-duplex: we listen while watching, speak while acting, and fluidly adapt to turn-taking and interruptions. Realizing these capabilities is essential for building models simulating humans. We present ELLSA (End-to-end Listen, Look, Speak and Act), which, to our knowledge, is the first full-duplex, end-to-end model that simultaneously perceives and generates across vision, text, speech, and action within a single architecture, enabling interaction patterns previously out of reach, yielding more natural, human-like behaviors. At its core is a novel SA-MoE architecture (Self-Attention Mixture-of-Experts) that routes each modality to specialized experts and fuses them through a unified attention backbone. This provides a generalizable solution for joint multimodal perception and concurrent generation, leveraging strong pre-trained components while enabling efficient modality integration and mitigating modality interference. On speech-interaction and robot-manipulation benchmarks, ELLSA matches modality-specific baselines, while uniquely supporting advanced multimodal and full-duplex behaviors such as dialogue and action turn-taking, defective instruction rejection, speaking-while-acting, context-grounded visual question answering, and action barge-ins. We contend that ELLSA represents a step toward more natural and general interactive intelligence, contributing to the broader pursuit of artificial general intelligence. All data, code and model checkpoints will be released at this https URL.
>
---
#### [replaced 004] Sensorimotor Self-Recognition in Multimodal Large Language Model-Driven Robots
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究多模态大语言模型驱动的机器人自我识别问题，通过传感器运动体验探索其自识能力，验证了模型在环境感知与自我认知中的表现。**

- **链接: [https://arxiv.org/pdf/2505.19237](https://arxiv.org/pdf/2505.19237)**

> **作者:** Iñaki Dellibarda Varela; Pablo Romero-Sorozabal; Diego Torricelli; Gabriel Delgado-Oleas; Jose Ignacio Serrano; Maria Dolores del Castillo Sobrino; Eduardo Rocon; Manuel Cebrian
>
> **备注:** 16 pages, 3 figures, 1 table
>
> **摘要:** Self-recognition -- the ability to maintain an internal representation of one's own body within the environment -- underpins intelligent, autonomous behavior. As a foundational component of the minimal self, self-recognition provides the initial substrate from which higher forms of self-awareness may eventually emerge. Recent advances in large language models achieve human-like performance in tasks integrating multimodal information, raising growing interest in the embodiment capabilities of AI agents deployed on nonhuman platforms such as robots. We investigate whether multimodal LLMs can develop self-recognition through sensorimotor experience by integrating an LLM into an autonomous mobile robot. The system exhibits robust environmental awareness, self-identification, and predictive awareness, enabling it to infer its robotic nature and motion characteristics. Structural equation modeling reveals how sensory integration influences distinct dimensions of the minimal self and their coordination with past-present memory, as well as the hierarchical internal associations that drive self-identification. Ablation tests of sensory inputs demonstrate compensatory interactions among sensors and confirm the essential role of structured and episodic memory. Given appropriate sensory information about the world and itself, multimodal LLMs open the door to artificial selfhood in embodied cognitive systems.
>
---
#### [replaced 005] Integrated Wheel Sensor Communication using ESP32 -- A Contribution towards a Digital Twin of the Road System
- **分类: cs.RO**

- **简介: 该论文属于智能交通任务，旨在解决轮胎与路面交互数据传输问题。通过ESP32实现高效通信，减少数据丢失，提升实时数据采集能力。**

- **链接: [https://arxiv.org/pdf/2509.04061](https://arxiv.org/pdf/2509.04061)**

> **作者:** Ventseslav Yordanov; Simon Schäfer; Alexander Mann; Stefan Kowalewski; Bassam Alrifaee; Lutz Eckstein
>
> **备注:** 6 pages, 2 figures, this work was submitted to and accepted by IEEE International Conference on Intelligent Transportation Systems (ITSC) 2025
>
> **摘要:** While current onboard state estimation methods are adequate for most driving and safety-related applications, they do not provide insights into the interaction between tires and road surfaces. This paper explores a novel communication concept for efficiently transmitting integrated wheel sensor data from an ESP32 microcontroller. Our proposed approach utilizes a publish-subscribe system, surpassing comparable solutions in the literature regarding data transmission volume. We tested this approach on a drum tire test rig with our prototype sensors system utilizing a diverse selection of sample frequencies between 1 Hz and 32 000 Hz to demonstrate the efficacy of our communication concept. The implemented prototype sensor showcases minimal data loss, approximately 0.1% of the sampled data, validating the reliability of our developed communication system. This work contributes to advancing real-time data acquisition, providing insights into optimizing integrated wheel sensor communication.
>
---
#### [replaced 006] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文属于视觉与语言导航任务，解决虚假前提指令下的导航问题。通过构建VLN-NF基准和提出ROAM方法，提升代理在目标不存在时的探索与决策能力。**

- **链接: [https://arxiv.org/pdf/2604.10533](https://arxiv.org/pdf/2604.10533)**

> **作者:** Hung-Ting Su; Ting-Jun Wang; Jia-Fong Yeh; Min Sun; Winston H. Hsu
>
> **备注:** ACL 2026 camera ready
>
> **摘要:** Conventional Vision-and-Language Navigation (VLN) benchmarks assume instructions are feasible and the referenced target exists, leaving agents ill-equipped to handle false-premise goals. We introduce VLN-NF, a benchmark with false-premise instructions where the target is absent from the specified room and agents must navigate, gather evidence through in-room exploration, and explicitly output NOT-FOUND. VLN-NF is constructed via a scalable pipeline that rewrites VLN instructions using an LLM and verifies target absence with a VLM, producing plausible yet factually incorrect goals. We further propose REV-SPL to jointly evaluate room reaching, exploration coverage, and decision correctness. To address this challenge, we present ROAM, a two-stage hybrid that combines supervised room-level navigation with LLM/VLM-driven in-room exploration guided by a free-space clearance prior. ROAM achieves the best REV-SPL among compared methods, while baselines often under-explore and terminate prematurely under unreliable instructions. VLN-NF project page can be found at this https URL.
>
---
#### [replaced 007] Chasing Ghosts: A Simulation-to-Real Olfactory Navigation Stack with Optional Vision Augmentation
- **分类: cs.RO**

- **简介: 该论文属于无人机嗅觉导航任务，解决在复杂空气流中定位气味源的问题。提出一个基于最小传感器的系统，结合仿真训练与真实飞行验证，实现无需外部定位的自主导航。**

- **链接: [https://arxiv.org/pdf/2602.19577](https://arxiv.org/pdf/2602.19577)**

> **作者:** Kordel K. France; Ovidiu Daescu; Latifur Khan; Rohith Peddi
>
> **摘要:** Autonomous odor source localization remains a challenging problem for aerial robots due to turbulent airflow, sparse and delayed sensory signals, and strict payload and compute constraints. While prior unmanned aerial vehicle (UAV)-based olfaction systems have demonstrated gas distribution mapping or reactive plume tracing, they rely on predefined coverage patterns, external infrastructure, or extensive sensing and coordination. In this work, we present a complete, open-source UAV system for online odor source localization using a minimal sensor suite. The system integrates custom olfaction hardware, onboard sensing, and a learning-based navigation policy trained in simulation and deployed on a real quadrotor. Through our minimal framework, the UAV is able to navigate directly toward an odor source without constructing an explicit gas distribution map or relying on external positioning systems. Vision is incorporated as an optional complementary modality to accelerate navigation under certain conditions. We validate the proposed system through real-world flight experiments in a large indoor environment using an ethanol source, demonstrating consistent source-finding behavior under realistic airflow conditions. The primary contribution of this work is a reproducible system and methodological framework for UAV-based olfactory navigation and source finding under minimal sensing assumptions. We elaborate on our hardware design and open source our UAV firmware, simulation code, olfaction-vision dataset, and circuit board to the community. Code, data, and designs will be made available at this https URL.
>
---
#### [replaced 008] City-Wide Low-Altitude Urban Air Mobility: A Scalable Global Path Planning Approach via Risk-Aware Multi-Scale Cell Decomposition
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于城市空中交通路径规划任务，解决复杂城市环境下的安全路径生成问题。提出一种多尺度风险感知的单元分解方法，提升路径安全性与计算效率。**

- **链接: [https://arxiv.org/pdf/2408.02786](https://arxiv.org/pdf/2408.02786)**

> **作者:** Josue N. Rivera; Dengfeng Sun; Chen Lv
>
> **备注:** 6 pages
>
> **摘要:** The realization of Urban Air Mobility (UAM) necessitates scalable global path planning algorithms capable of ensuring safe navigation within complex urban environments. This paper proposes a multi-scale risk-aware cell decomposition method that efficiently partitions city-scale airspace into variable-granularity sectors, assigning each cell an analytically estimated risk value based on obstacle proximity and expected risk. Unlike uniform grid approaches or sampling-based methods, our approach dynamically balances resolution with computational speed by bounding cell risk via Mahalanobis distance projections, eliminating exhaustive field sampling. Comparative experiments against classical A*, Artificial Potential Fields (APF), and Informed RRT* across five diverse urban topologies demonstrate that our method generates safer paths with lower cumulative risk while reducing computation time by orders of magnitude. The proposed framework, Larp Path Planner, is open-sourced and supports any map provider via its modified GeoJSON internal representation, with experiments conducted using OpenStreetMap data to facilitate reproducible research in city-wide aerial navigation.
>
---
#### [replaced 009] Diffusion Sequence Models for Generative In-Context Meta-Learning of Robot Dynamics
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于机器人动力学建模任务，解决分布偏移下的准确建模问题。通过扩散序列模型提升元学习的鲁棒性，实验表明其在不同场景下表现优异。**

- **链接: [https://arxiv.org/pdf/2604.13366](https://arxiv.org/pdf/2604.13366)**

> **作者:** Angelo Moroncelli; Matteo Rufolo; Gunes Cagin Aydin; Asad Ali Shahid; Loris Roveda
>
> **备注:** Angelo Moroncelli, Matteo Rufolo and Gunes Cagin Aydin contributed equally to this work
>
> **摘要:** Accurate modeling of robot dynamics is essential for model-based control, yet remains challenging under distributional shifts and real-time constraints. In this work, we formulate system identification as an in-context meta-learning problem and compare deterministic and generative sequence models for forward dynamics prediction. We take a Transformer-based meta-model, as a strong deterministic baseline, and introduce to this setting two complementary diffusion-based approaches: (i) inpainting diffusion (Diffuser), which learns the joint input-observation distribution, and (ii) conditioned diffusion models (CNN and Transformer), which generate future observations conditioned on control inputs. Through large-scale randomized simulations, we analyze performance across in-distribution and out-of-distribution regimes, as well as computational trade-offs relevant for control. We show that diffusion models significantly improve robustness under distribution shift, with inpainting diffusion achieving the best performance in our experiments. Finally, we demonstrate that warm-started sampling enables diffusion models to operate within real-time constraints, making them viable for control applications. These results highlight generative meta-models as a promising direction for robust system identification in robotics.
>
---
#### [replaced 010] Advancing MAPF Toward the Real World: A Scalable Multi-Agent Realistic Testbed (SMART)
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体路径规划任务，解决真实环境中MAPF算法评估难题。提出SMART测试平台，实现千级机器人真实仿真与高效评估。**

- **链接: [https://arxiv.org/pdf/2503.04798](https://arxiv.org/pdf/2503.04798)**

> **作者:** Jingtian Yan; Zhifei Li; William Kang; Kevin Zheng; Yulun Zhang; Zhe Chen; Yue Zhang; Daniel Harabor; Stephen F. Smith; Jiaoyang Li
>
> **摘要:** We present Scalable Multi-Agent Realistic Testbed (SMART), a realistic and efficient software tool for evaluating Multi-Agent Path Finding (MAPF) algorithms. MAPF focuses on planning collision-free paths for a group of robots. While state-of-the-art MAPF planners can plan paths for hundreds of robots in seconds, they often rely on simplified robot models, making their real-world performance unclear. Researchers typically lack access to hundreds of physical robots in laboratory settings to evaluate the algorithms. Meanwhile, industrial professionals who lack expertise in MAPF require an easy-to-use simulator to efficiently test and understand the performance of MAPF planners in their specific settings. SMART fills this gap with several advantages: (1) SMART uses physics-engine-based simulators to create realistic simulation environments, accounting for complex real-world factors such as robot kinodynamics and execution uncertainties, (2) SMART uses an execution monitor framework based on the Action Dependency Graph, facilitating seamless integration with various MAPF planners and robot models, and (3) SMART scales to thousands of robots. The code is publicly available at this https URL with an online service available at this https URL.
>
---
#### [replaced 011] RAYEN: Imposition of Hard Convex Constraints on Neural Networks
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出RAYEN框架，用于在神经网络中强制硬凸约束，解决约束满足与计算效率问题，支持多种约束类型，提升轨迹优化效率。**

- **链接: [https://arxiv.org/pdf/2307.08336](https://arxiv.org/pdf/2307.08336)**

> **作者:** Jesus Tordesillas; Victor Klemm; Jonathan P. How; Marco Hutter
>
> **摘要:** Despite the numerous applications of convex constraints in Robotics, enforcing them within learning-based frameworks remains an open challenge. Existing techniques either fail to guarantee satisfaction at all times, or incur prohibitive computational costs. This paper presents RAYEN, a framework for imposing hard convex constraints on the output or latent variables of a neural network. RAYEN guarantees constraint satisfaction during both training and testing, for any input and any network weights. Unlike prior approaches, RAYEN avoids computationally expensive orthogonal projections, soft constraints, conservative approximations of the feasible set, and slow iterative corrections. RAYEN supports any combination of linear, convex quadratic, second-order cone (SOC), and linear matrix inequality (LMI) constraints, with negligible overhead compared to unconstrained networks. For instance, it imposes 1K quadratic constraints on a 1K-dimensional variable with only 8 ms of overhead compared to a network that does not enforce these constraints. An LMI constraint with 300x300 dense matrices on a 10K-dimensional variable can be guaranteed with only 12 ms additional overhead. When used in neural networks that approximate the solution of constrained trajectory optimization problems, RAYEN runs 20 to 7468 times faster than state-of-the-art algorithms, while guaranteeing constraint satisfaction at all times and achieving a near-optimal cost (<1.5% optimality gap). Finally, we demonstrate RAYEN's ability to enforce actuator constraints on a learned locomotion policy by validating constraint satisfaction in both simulation and real-world experiments on a quadruped robot. The code is available at this https URL
>
---
#### [replaced 012] Linking Exteroception and Proprioception through Improved Contact Modeling for Soft Growing Robots
- **分类: cs.RO**

- **简介: 该论文研究软体机器人在非结构化环境中的导航与建图任务，旨在通过改进接触建模提升其环境感知能力，解决碰撞与变形理解不足的问题。工作包括分析碰撞行为、构建几何模拟器，并利用蒙特卡洛方法优化探索路径。**

- **链接: [https://arxiv.org/pdf/2507.10694](https://arxiv.org/pdf/2507.10694)**

> **作者:** Francesco Fuentes; Serigne Diagne; Zachary Kingston; Laura H. Blumenschein
>
> **备注:** Accepted to International Journal of Robotics Research (IJRR), 23 pages, 22 figures, 1 table
>
> **摘要:** Passive deformation due to compliance is a commonly used benefit of soft robots, providing opportunities to achieve robust actuation with few active degrees of freedom. Soft growing robots in particular have shown promise in navigation of unstructured environments due to their passive deformation. If their collisions and subsequent deformations can be better understood, soft robots could be used to understand the structure of the environment from direct tactile measurements. In this work, we propose the use of soft growing robots as mapping and exploration tools. We do this by first characterizing collision behavior during discrete turns, then leveraging this model to develop a geometry-based simulator that models robot trajectories in 2D environments. Finally, we demonstrate the model and simulator validity by mapping unknown environments using Monte Carlo sampling to estimate the optimal next deployment given current knowledge. Over both uniform and non-uniform environments, this selection method rapidly approaches ideal actions, showing the potential for soft growing robots in unstructured environment exploration and mapping.
>
---
#### [replaced 013] Stable Language Guidance for Vision-Language-Action Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于视觉-语言-动作模型任务，解决语言扰动导致的模型脆弱问题。提出RSS框架，通过分离视觉与语义信息提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.04052](https://arxiv.org/pdf/2601.04052)**

> **作者:** Zhihao Zhan; Yuhao Chen; Jiaying Zhou; Qinhan Lyu; Hao Liu; Keze Wang; Liang Lin; Guangrun Wang
>
> **备注:** Accepted to ACL2026 main conference
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated impressive capabilities in generalized robotic control; however, they remain notoriously brittle to linguistic perturbations. We identify a critical ``modality collapse'' phenomenon where strong visual priors overwhelm sparse linguistic signals, causing agents to overfit to specific instruction phrasings while ignoring the underlying semantic intent. To address this, we propose Residual Semantic Steering (RSS), a probabilistic framework that disentangles physical affordance from semantic execution. RSS introduces two theoretical innovations: (1) Monte Carlo Syntactic Integration, which approximates the true semantic posterior via dense, LLM-driven distributional expansion, and (2) Residual Affordance Steering, a dual-stream decoding mechanism that explicitly isolates the causal influence of language by subtracting the visual affordance prior. Theoretical analysis suggests that RSS effectively maximizes the mutual information between action and intent while suppressing visual distractors. Empirical results across diverse manipulation benchmarks demonstrate that RSS achieves state-of-the-art robustness, maintaining performance even under adversarial linguistic perturbations. We release our code at this https URL.
>
---
#### [replaced 014] Neuromorphic BrailleNet: Accurate and Generalizable Braille Reading Beyond Single Characters through Event-Based Optical Tactile Sensing
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉识别任务，旨在解决传统盲文阅读速度慢、计算量大等问题。提出基于事件驱动的触觉传感系统，实现高效、准确的连续盲文识别。**

- **链接: [https://arxiv.org/pdf/2601.19079](https://arxiv.org/pdf/2601.19079)**

> **作者:** Naqash Afzal; Niklas Funk; Erik Helmut; Jan Peters; Benjamin Ward-Cherrier
>
> **摘要:** Conventional robotic Braille readers typically rely on discrete, character-by-character scanning, limiting reading speed and disrupting natural flow. Vision-based alternatives often require substantial computation, introduce latency, and degrade in real-world conditions. In this work, we present a high accuracy, real-time pipeline for continuous Braille recognition using Evetac, an open-source neuromorphic event-based tactile sensor. Unlike frame-based vision systems, the neuromorphic tactile modality directly encodes dynamic contact events during continuous sliding, closely emulating human finger-scanning strategies. Our approach combines spatiotemporal segmentation with a lightweight ResNet-based classifier to process sparse event streams, enabling robust character recognition across varying indentation depths and scanning speeds. The proposed system achieves near-perfect accuracy (>=98%) at standard depths, generalizes across multiple Braille board layouts, and maintains strong performance under fast scanning. On a physical Braille board containing daily-living vocabulary, the system attains over 90% word-level accuracy, demonstrating robustness to temporal compression effects that challenge conventional methods. These results position neuromorphic tactile sensing as a scalable, low latency solution for robotic Braille reading, with broader implications for tactile perception in assistive and robotic applications.
>
---
#### [replaced 015] TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决VLM在动态空间中推理能力不足的问题。通过引入拓扑结构增强模型的全局动作推理能力，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.02972](https://arxiv.org/pdf/2603.02972)**

> **作者:** Jiaxing Liu; Zexi Zhang; Xiaoyan Li; Boyue Wang; Yongli Hu; Baocai Yin
>
> **摘要:** Vision-Language Navigation (VLN) presents a unique challenge for Large Vision-Language Models (VLMs) due to their inherent architectural mismatch: VLMs are primarily pretrained on static, disembodied vision-language tasks, which fundamentally clash with the dynamic, embodied, and spatially-structured nature of navigation. Existing large-model-based methods often resort to converting rich visual and spatial information into text, forcing models to implicitly infer complex visual-topological relationships or limiting their global action capabilities. To bridge this gap, we propose TagaVLM (Topology-Aware Global Action reasoning), an end-to-end framework that explicitly injects topological structures into the VLM backbone. To introduce topological edge information, Spatial Topology Aware Residual Attention (STAR-Att) directly integrates it into the VLM's self-attention mechanism, enabling intrinsic spatial reasoning while preserving pretrained knowledge. To enhance topological node information, an Interleaved Navigation Prompt strengthens node-level visual-text alignment. Finally, with the embedded topological graph, the model is capable of global action reasoning, allowing for robust path correction. On the R2R benchmark, TagaVLM achieves state-of-the-art performance among large-model-based methods, with a Success Rate (SR) of 51.09% and SPL of 47.18 in unseen environments, outperforming prior work by 3.39% in SR and 9.08 in SPL. This demonstrates that, for embodied spatial reasoning, targeted enhancements on smaller open-source VLMs can be more effective than brute-force model scaling. The code can be found on our project page: this https URL
>
---
#### [replaced 016] PTLD: Sim-to-real Privileged Tactile Latent Distillation for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉操作任务，旨在解决sim-to-real迁移中触觉感知不足的问题。通过引入PTLD方法，利用真实触觉数据提升仿真中本体感觉策略的性能。**

- **链接: [https://arxiv.org/pdf/2603.04531](https://arxiv.org/pdf/2603.04531)**

> **作者:** Rosy Chen; Mustafa Mukadam; Michael Kaess; Tingfan Wu; Francois R Hogan; Jitendra Malik; Akash Sharma
>
> **摘要:** Tactile dexterous manipulation is essential to automating complex household tasks, yet learning effective control policies remains a challenge. While recent work has relied on imitation learning, obtaining high quality demonstrations for multi-fingered hands via robot teleoperation or kinesthetic teaching is prohibitive. Alternatively, with reinforcement we can learn skills in simulation, but fast and realistic simulation of tactile observations is challenging. To bridge this gap, we introduce PTLD: sim-to-real Privileged Tactile Latent Distillation, a novel approach to learning tactile manipulation skills without requiring tactile simulation. Instead of simulating tactile sensors or relying purely on proprioceptive policies to transfer zero-shot sim-to-real, our key idea is to leverage privileged sensors in the real world to collect real-world tactile policy data. This data is then used to distill a robust state estimator that operates on tactile input. We demonstrate from our experiments that PTLD can be used to improve proprioceptive manipulation policies trained in simulation significantly by incorporating tactile sensing. On the benchmark in-hand rotation task, PTLD achieves a 182% improvement over a proprioception only policy. We also show that PTLD enables learning the challenging task of tactile in-hand reorientation where we see a 57% improvement in the number of goals reached over using proprioception alone. Website: this https URL.
>
---
#### [replaced 017] eCP: Equivariant Conformal Prediction with pre-trained models
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于不确定性量化任务，旨在解决长时序预测中置信区域过大的问题。通过引入对称性信息，改进共形预测方法，提升预测精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.03986](https://arxiv.org/pdf/2602.03986)**

> **作者:** Nikolaos Bousias; Lars Lindemann; George Pappas
>
> **摘要:** Conformal prediction, a post-hoc, distribution-free, finite-sample method of uncertainty quantification that offers formal coverage guarantees under the assumption of data exchangeability. Unfortunately, the resulting uncertainty regions can grow significantly in long horizon missions, rendering the statistical guarantees uninformative. To that end, we propose infusing CP with geometric information via group-averaging of the pretrained predictor to distribute the non-conformity mass across the orbits. Each sample now is treated as a representative of an orbit, thus uncertainty can be mitigated by other samples entangled to it via the orbit inducing elements of the symmetry group. Our approach provably yields contracted non-conformity scores in increasing convex order, implying improved exponential-tail bounds and sharper conformal prediction sets in expectation, especially at high confidence levels. We then propose an experimental design to test these theoretical claims in pedestrian trajectory prediction.
>
---
#### [replaced 018] Conformal Prediction-Based MPC for Stochastic Linear Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制理论领域，解决随机线性系统在未知扰动下的约束满足问题。通过结合共形预测与MPC，构建置信区域，实现高效、可靠的控制策略。**

- **链接: [https://arxiv.org/pdf/2512.10738](https://arxiv.org/pdf/2512.10738)**

> **作者:** Lukas Vogel; Andrea Carron; Eleftherios E. Vlahakis; Dimos V. Dimarogonas
>
> **备注:** 7 pages, 1 figure. This is an extended version of the publication to the 24th European Control Conference (ECC 2026)
>
> **摘要:** We propose a stochastic model predictive control (MPC) framework for linear systems subject to joint-in-time chance constraints under unknown disturbance distributions. Unlike existing approaches that rely on parametric or Gaussian assumptions, or require expensive offline computation, the method uses conformal prediction to construct finite-sample confidence regions for the system's error trajectories with minimal computational effort. These probabilistic sets enable relaxation of the joint-in-time chance constraints into a deterministic closed-loop formulation based on indirect feedback, ensuring recursive feasibility and chance constraint satisfaction. Further, we extend to the output feedback setting and establish analogous guarantees from output measurements alone, given access to noise samples. Numerical examples demonstrate the effectiveness and advantages compared to existing approaches.
>
---
#### [replaced 019] Bridging the Ex-Vivo to In-Vivo Gap: Synthetic Priors for Monocular Depth Estimation in Specular Surgical Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目深度估计任务，旨在解决手术环境中真实与模拟数据间的差异问题。通过引入高保真合成先验和动态低秩适配方法，提升模型在临床场景下的性能。**

- **链接: [https://arxiv.org/pdf/2512.23786](https://arxiv.org/pdf/2512.23786)**

> **作者:** Ankan Aich; Emma D. Ryan; Kris Moe; Isaac Schmale; Li-Xing Man; Yangming Lee
>
> **摘要:** Accurate Monocular Depth Estimation (MDE) is critical for autonomous robotic surgery. However, existing self-supervised methods often exhibit a severe "ex-vivo to in-vivo gap": they achieve high accuracy on public datasets but struggle in actual clinical deployments. This disparity arises because the severe specular reflections and fluid-filled deformations inherent to real surgeries. Models trained on noisy real-world pseudo-labels consequently suffer from severe boundary collapse. To address this, we leverage the high-fidelity synthetic priors of the \textit{Depth Anything V2} architecture, which inherently capture precise geometric details, and efficiently adapt them to the medical domain using Dynamic Vector Low-Rank Adaptation (DV-LORA). Our contributions are two-fold. Technically, our approach establishes a new state-of-the-art on the public SCARED dataset; under a novel physically-stratified evaluation protocol, it reduces Squared Relative Error by over 17\% in high-specularity regimes compared to strong baselines. Furthermore, to provide a rigorous reality check for the field, we introduce \textbf{ROCAL-T 90} (Real Operative CT-Aligned Laparoscopic Trajectories 90), the first real-surgery validation dataset featuring 90 clinical endoscopic sequences with sub-millimeter ($< 1$mm) ground-truth trajectories. Evaluations on ROCAL-T 90 demonstrate our model's superior robustness in true clinical settings.
>
---
#### [replaced 020] STL-Based Motion Planning and Uncertainty-Aware Risk Analysis for Human-Robot Collaboration with a Multi-Rotor Aerial Vehicle
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决多旋翼无人机在复杂环境中的安全路径规划与风险分析问题。通过STL逻辑和优化方法实现动态轨迹生成与不确定性下的风险评估。**

- **链接: [https://arxiv.org/pdf/2509.10692](https://arxiv.org/pdf/2509.10692)**

> **作者:** Giuseppe Silano; Amr Afifi; Martin Saska; Antonio Franchi
>
> **备注:** 46 pages, 14 figures
>
> **摘要:** This paper presents a motion planning and risk analysis framework for enhancing human-robot collaboration with a Multi-Rotor Aerial Vehicle. The proposed method employs Signal Temporal Logic to encode key mission objectives, including safety, temporal requirements, and human preferences, with particular emphasis on ergonomics and comfort. An optimization-based planner generates dynamically feasible trajectories while explicitly accounting for the vehicle's nonlinear dynamics and actuation constraints. To address the resulting non-convex and non-smooth optimization problem, smooth robustness approximations and gradient-based techniques are adopted. In addition, an uncertainty-aware risk analysis is introduced to quantify the likelihood of specification violations under human-pose uncertainty. A robustness-aware event-triggered replanning strategy further enables online recovery from disturbances and unforeseen events by preserving safety margins during execution. The framework is validated through MATLAB and Gazebo simulations on an object handover task inspired by power line maintenance scenarios. Results demonstrate the ability of the proposed method to achieve safe, efficient, and resilient human-robot collaboration under realistic operating conditions.
>
---
#### [replaced 021] Driving in Corner Case: A Real-World Adversarial Closed-Loop Evaluation Platform for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，旨在解决真实场景下极端案例难以收集的问题。通过构建闭环评估平台，生成对抗性交互以测试模型性能。**

- **链接: [https://arxiv.org/pdf/2512.16055](https://arxiv.org/pdf/2512.16055)**

> **作者:** Jiaheng Geng; Jiatong Du; Xinyu Zhang; Ye Li; Panqu Wang; Yanjun Huang
>
> **备注:** Update some experimental details
>
> **摘要:** Safety-critical corner cases, difficult to collect in the real world, are crucial for evaluating end-to-end autonomous driving. Adversarial interaction is an effective method to generate such safety-critical corner cases. While existing adversarial evaluation methods are built for models operating in simplified simulation environments, adversarial evaluation for real-world end-to-end autonomous driving has been little explored. To address this challenge, we propose a closed-loop evaluation platform for end-to-end autonomous driving, which can generate adversarial interactions in real-world scenes. In our platform, the real-world image generator cooperates with an adversarial traffic policy to evaluate various end-to-end models trained on real-world data. The generator, based on flow matching, efficiently and stably generates real-world images according to the traffic environment information. The efficient adversarial surrounding vehicle policy is designed to model challenging interactions and create corner cases that current autonomous driving systems struggle to handle. Experimental results demonstrate that the platform can generate realistic driving images efficiently. Through evaluating the end-to-end models such as UniAD and VAD, we demonstrate that based on the adversarial policy, our platform evaluates the performance degradation of the tested model in corner cases. This result indicates that this platform can effectively detect the model's potential issues, which will facilitate the safety and robustness of end-to-end autonomous driving.
>
---
#### [replaced 022] On the Importance of Tactile Sensing for Imitation Learning: A Case Study on Robotic Match Lighting
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决动态接触任务中模仿学习的效率问题。通过融合触觉信息与视觉数据，提出一种新型模仿学习框架，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2504.13618](https://arxiv.org/pdf/2504.13618)**

> **作者:** Niklas Funk; Changqi Chen; Tim Schneider; Georgia Chalvatzaki; Roberto Calandra; Jan Peters
>
> **摘要:** The field of robotic manipulation has advanced significantly in recent years. At the sensing level, several novel tactile sensors have been developed, capable of providing accurate contact information. On a methodological level, learning from demonstrations has proven an efficient paradigm to obtain performant robotic manipulation policies. The combination of both holds the promise to extract crucial contact-related information from the demonstration data and actively exploit it during policy rollouts. However, this integration has so far been underexplored, most notably in dynamic, contact-rich manipulation tasks where precision and reactivity are essential. This work therefore proposes a multimodal, visuotactile imitation learning framework that integrates a modular transformer architecture with a flow-based generative model, enabling efficient learning of fast and dexterous manipulation policies. We evaluate our framework on the dynamic, contact-rich task of robotic match lighting - a task in which tactile feedback influences human manipulation performance. The experimental results highlight the effectiveness of our approach and show that adding tactile information improves policy performance, thereby underlining their combined potential for learning dynamic manipulation from few demonstrations. Project website: this https URL .
>
---
#### [replaced 023] R3D2: Realistic 3D Asset Insertion via Diffusion for Autonomous Driving Simulation
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出R3D2，解决自动驾驶仿真中动态3D资产真实插入的问题。通过扩散模型实现快速、真实的场景渲染效果生成，提升仿真 realism 和可扩展性。**

- **链接: [https://arxiv.org/pdf/2506.07826](https://arxiv.org/pdf/2506.07826)**

> **作者:** William Ljungbergh; Bernardo Taveira; Wenzhao Zheng; Adam Tonderski; Chensheng Peng; Fredrik Kahl; Christoffer Petersson; Michael Felsberg; Kurt Keutzer; Masayoshi Tomizuka; Wei Zhan
>
> **摘要:** Validating autonomous driving (AD) systems requires diverse and safety-critical testing, making photorealistic virtual environments essential. Traditional simulation platforms, while controllable, are resource-intensive to scale and often suffer from a domain gap with real-world data. In contrast, neural reconstruction methods like 3D Gaussian Splatting (3DGS) offer a scalable solution for creating photorealistic digital twins of real-world driving scenes. However, they struggle with dynamic object manipulation and reusability as their per-scene optimization-based methodology tends to result in incomplete object models with integrated illumination effects. This paper introduces R3D2, a lightweight, one-step diffusion model designed to overcome these limitations and enable realistic insertion of complete 3D assets into existing scenes by generating plausible rendering effects-such as shadows and consistent lighting-in real time. This is achieved by training R3D2 on a novel dataset: 3DGS object assets are generated from in-the-wild AD data using an image-conditioned 3D generative model, and then synthetically placed into neural rendering-based virtual environments, allowing R3D2 to learn realistic integration. Quantitative and qualitative evaluations demonstrate that R3D2 significantly enhances the realism of inserted assets, enabling use-cases like text-to-3D asset insertion and cross-scene/dataset object transfer, allowing for true scalability in AD validation. To promote further research in scalable and realistic AD simulation, we release our code, see this https URL.
>
---
#### [replaced 024] Satellite Chasers: Divergent Adversarial Reinforcement Learning to Engage Intelligent Adversaries on Orbit
- **分类: cs.RO**

- **简介: 该论文属于多智能体强化学习任务，旨在解决卫星在对抗环境中的自主避让问题。提出DARL方法，提升卫星应对多个敌对航天器的能力。**

- **链接: [https://arxiv.org/pdf/2409.17443](https://arxiv.org/pdf/2409.17443)**

> **作者:** Cameron Mehlman; Gregory Falco
>
> **摘要:** As space becomes increasingly crowded and contested, robust autonomous capabilities for multi-agent environments are gaining critical importance. Current autonomous systems in space primarily rely on optimization-based path planning or long-range orbital maneuvers, which have not yet proven effective in adversarial scenarios where one satellite is actively pursuing another. We introduce Divergent Adversarial Reinforcement Learning (DARL), a two-stage Multi-Agent Reinforcement Learning (MARL) approach designed to train autonomous evasion strategies for satellites engaged with multiple adversarial spacecraft. Our method enhances exploration during training by promoting diverse adversarial strategies, leading to more robust and adaptable evader models. We validate DARL through a cat-and-mouse satellite scenario, modeled as a partially observable multi-agent capture the flag game where two adversarial ``cat" spacecraft pursue a single ``mouse" evader. DARL's performance is compared against several benchmarks, including an optimization-based satellite path planner, demonstrating its ability to produce highly robust models for adversarial multi-agent space environments.
>
---
#### [replaced 025] Special Unitary Parameterized Estimators of Rotation
- **分类: cs.RO**

- **简介: 该论文属于旋转估计任务，解决如何用特殊酉矩阵表示旋转的问题。通过重构Wahba问题，提出线性约束和两种新的神经网络旋转表示方法。**

- **链接: [https://arxiv.org/pdf/2411.13109](https://arxiv.org/pdf/2411.13109)**

> **作者:** Akshay Chandrasekhar
>
> **备注:** Final version to be published at ICLR 2026; added code link; 33 pages
>
> **摘要:** This paper revisits the topic of rotation estimation through the lens of special unitary matrices. We begin by reformulating Wahba's problem using $SU(2)$ to derive multiple solutions that yield linear constraints on corresponding quaternion parameters. We then explore applications of these constraints by formulating efficient methods for related problems. Finally, from this theoretical foundation, we propose two novel continuous representations for learning rotations in neural networks. Extensive experiments validate the effectiveness of the proposed methods.
>
---
#### [replaced 026] Social Learning Strategies for Evolved Virtual Soft Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究虚拟软体机器人的身体与脑部协同优化问题，通过社会学习策略提升控制参数优化效率。**

- **链接: [https://arxiv.org/pdf/2604.12482](https://arxiv.org/pdf/2604.12482)**

> **作者:** K. Ege de Bruin; Kyrre Glette; Kai Olav Ellefsen; Giorgia Nadizar; Eric Medvet
>
> **摘要:** Optimizing the body and brain of a robot is a coupled challenge: the morphology determines what control strategies are effective, while the control parameters influence how well the morphology performs. This joint optimization can be done through nested loops of evolutionary and learning processes, where the control parameters of each robot are learned independently. However, the control parameters learned by one robot may contain valuable information for others. Thus, we introduce a social learning approach in which robots can exploit optimized parameters from their peers to accelerate their own brain optimization. Within this framework, we systematically investigate how the selection of teachers, deciding which and how many robots to learn from, affects performance, experimenting with virtual soft robots in four tasks and environments. In particular, we study the effect of inheriting experience from morphologically similar robots due to the tightly coupled body and brain in robot optimization. Our results confirm the effectiveness of building on others' experience, as social learning clearly outperforms learning from scratch under equivalent computational budgets. In addition, while the optimal teacher selection strategy remains open, our findings suggest that incorporating knowledge from multiple teachers can yield more consistent and robust improvements.
>
---
#### [replaced 027] EgoWalk: A Multimodal Dataset for Robot Navigation in the Wild
- **分类: cs.RO**

- **简介: 该论文提出EgoWalk数据集，用于机器人导航任务。解决真实环境数据不足的问题，收集50小时多场景人类导航数据，并提供辅助任务数据和处理工具。**

- **链接: [https://arxiv.org/pdf/2505.21282](https://arxiv.org/pdf/2505.21282)**

> **作者:** Timur Akhtyamov; Mohamad Al Mdfaa; Javier Antonio Ramirez Benavides; Arthur Nigmatzyanov; Sergey Bakulin; German Devchich; Denis Fatykhov; Diego Ruiz Salinas; Alexander Mazurov; Kristina Zipa; Malik Mohrat; Pavel Kolesnik; Ivan Sosin; Gonzalo Ferrer
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Data-driven navigation algorithms are critically dependent on large-scale, high-quality real-world data collection for successful training and robust performance in realistic and uncontrolled conditions. To enhance the growing family of navigation-related real-world datasets, we introduce EgoWalk - a dataset of 50 hours of human navigation in a diverse set of indoor/outdoor, varied seasons, and location environments. Along with the raw and Imitation Learning-ready data, we introduce several pipelines to automatically create subsidiary datasets for other navigation-related tasks, namely natural language goal annotations and traversability segmentation masks. Diversity studies, use cases, and benchmarks for the proposed dataset are provided to demonstrate its practical applicability. We openly release all data processing pipelines and the description of the hardware platform used for data collection to support future research and development in robot navigation systems.
>
---
#### [replaced 028] 2D or 3D: Who Governs Salience in VLA Models? -- Tri-Stage Token Pruning Framework with Modality Salience Awareness
- **分类: cs.MM; cs.CV; cs.RO**

- **简介: 该论文属于多模态模型优化任务，解决MVLA模型中2D/3D模态token pruning问题。提出三阶段框架，提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2604.09244](https://arxiv.org/pdf/2604.09244)**

> **作者:** Zihao Zheng; Sicheng Tian; Zhihao Mao; Lingyue Zhang; Chenyue Li; Ziyun Zhang; Hong Gao; Yuchen Huang; Yutong Xu; Guojie Luo; Xiang Chen
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as the mainstream of embodied intelligence. Recent VLA models have expanded their input modalities from 2D-only to 2D+3D paradigms, forming multi-visual-modal VLA (MVLA) models. Despite achieving improved spatial perception, MVLA faces a greater acceleration demand due to the increased number of input tokens caused by modal expansion. Token pruning is an effective optimization methods tailored to MVLA models. However, existing token pruning schemes are designed for 2D-only VLA models, ignoring 2D/3D modality salience differences. In this paper, we follow the application process of multi-modal data in MVLA models and develop a tri-stage analysis to capture the discrepancy and dynamics of 2D/3D modality salience. Based on these, we propose a corresponding tri-stage token pruning framework for MVLA models to achieve optimal 2D/3D token selection and efficient pruning. Experiments show that our framework achieves up to a 2.55x inference speedup with minimal accuracy loss, while only costing 5.8% overhead. Our Code is coming soon.
>
---
#### [replaced 029] HAVEN: Hierarchical Adversary-aware Visibility-Enabled Navigation with Cover Utilization using Deep Transformer Q-Networks
- **分类: cs.RO**

- **简介: 该论文提出HAVEN框架，解决部分可观测环境下的自主导航问题。通过结合深度Transformer Q网络与低级控制器，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2512.00592](https://arxiv.org/pdf/2512.00592)**

> **作者:** Mihir Chauhan; Damon Conover; Aniket Bera
>
> **摘要:** Autonomous navigation in partially observable environments requires agents to reason beyond immediate sensor input, exploit occlusion, and ensure safety while progressing toward a goal. These challenges arise in many robotics domains, from urban driving and warehouse automation to defense and surveillance. Classical path planning approaches and memoryless reinforcement learning often fail under limited fields of view (FoVs) and occlusions, committing to unsafe or inefficient maneuvers. We propose a hierarchical navigation framework that integrates a Deep Transformer Q-Network (DTQN) as a high-level subgoal selector with a modular low-level controller for waypoint execution. The DTQN consumes short histories of task-aware features, encoding odometry, goal direction, obstacle proximity, and visibility cues, and outputs Q-values to rank candidate subgoals. Visibility-aware candidate generation introduces masking and exposure penalties, rewarding the use of cover and anticipatory safety. A low-level potential field controller then tracks the selected subgoal, ensuring smooth short-horizon obstacle avoidance. We validate our approach in 2D simulation and extend it directly to a 3D Unity-ROS environment by projecting point-cloud perception into the same feature schema, enabling transfer without architectural changes. Results show consistent improvements over classical planners and RL baselines in success rate, safety margins, and time to goal, with ablations confirming the value of temporal memory and visibility-aware candidate design. These findings highlight a generalizable framework for safe navigation under uncertainty, with broad relevance across robotic platforms.
>
---
#### [replaced 030] Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching and Differentiable Optimization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于多机器人路径优化任务，解决集中式优化计算不可行的问题。提出Flow-Opt方法，通过生成模型和安全过滤器实现高效、平滑的轨迹规划。**

- **链接: [https://arxiv.org/pdf/2510.09204](https://arxiv.org/pdf/2510.09204)**

> **作者:** Simon Idoko; Arun Kumar Singh
>
> **摘要:** Centralized trajectory optimization in the joint space of multiple robots allows access to a larger feasible space that can result in smoother trajectories, especially while planning in tight spaces. Unfortunately, it is often computationally intractable beyond a very small swarm size. In this paper, we propose Flow-Opt, a learning-based approach towards improving the computational tractability of centralized multi-robot trajectory optimization. Specifically, we reduce the problem to first learning a generative model to sample different candidate trajectories and then using a learned Safety-Filter(SF) to ensure fast inference-time constraint satisfaction. We propose a flow-matching model with a diffusion transformer (DiT) augmented with permutation invariant robot position and map encoders as the generative model. We develop a custom solver for our SF and equip it with a neural network that predicts context-specific initialization. The initialization network is trained in a self-supervised manner, taking advantage of the differentiability of the SF solver. We advance the state-of-the-art in the following respects. First, we show that we can generate trajectories of tens of robots in cluttered environments in a few tens of milliseconds. This is several times faster than existing centralized optimization approaches. Moreover, our approach also generates smoother trajectories orders of magnitude faster than competing baselines based on diffusion models. Second, each component of our approach can be batched, allowing us to solve a few tens of problem instances in a fraction of a second. We believe this is a first such result; no existing approach provides such capabilities. Finally, our approach can generate a diverse set of trajectories between a given set of start and goal locations, which can capture different collision-avoidance behaviors.
>
---
#### [replaced 031] Optimal control of differentially flat underactuated planar robots in the perspective of oscillation mitigation
- **分类: cs.RO**

- **简介: 该论文研究如何通过最优控制与微分平坦控制结合，解决欠驱动平面机器人轨迹跟踪中的振荡问题，提升控制精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.15528](https://arxiv.org/pdf/2603.15528)**

> **作者:** Stefano Lovato; Michele Tonan; Matteo Bottin; Matteo Massaro; Alberto Doria; Giulio Rosati
>
> **备注:** Accepted to European Control Conference (ECC 2026)
>
> **摘要:** Underactuated robots are characterized by a larger number of degrees of freedom than actuators and if they are designed with a specific mass distribution, they can be controlled by means of differential flatness theory. This structural property enables the development of lightweight and cost-effective robotic systems with enhanced dexterity. However, a key challenge lies in managing the passive joints, whose control demands precise and comprehensive dynamic modeling of the system. To simplify dynamic models, particularly for low-speed trajectories, friction is often neglected. While this assumption simplifies analysis and control design, it introduces residual oscillations of the end-effector about the target position. In this paper, the possibility of using optimal control along with differential flatness control is investigated to improve the tracking of the planned trajectories. First, the study was carried out through formal analysis, and then, it was validated by means of numerical simulations. Results highlight that optimal control can be used to plan the flat variables considering different (quadratic) performance indices: control effort, i.e. motor torque, and potential energy of the considered underactuated joint. Moreover, the minimization of potential energy can be used to design motion laws that are robust against variation of the stiffness and damping of the underactuated joint, thus reducing oscillations in the case of stiffness/damping mismatch.
>
---
#### [replaced 032] InternScenes: A Large-scale Simulatable Indoor Scene Dataset with Realistic Layouts
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出InternScenes数据集，解决室内场景数据不足的问题。整合真实扫描、生成和设计场景，包含4万场景和1.96M物体，用于Embodied AI任务，提升场景多样性与真实性。**

- **链接: [https://arxiv.org/pdf/2509.10813](https://arxiv.org/pdf/2509.10813)**

> **作者:** Weipeng Zhong; Peizhou Cao; Yichen Jin; Li Luo; Wenzhe Cai; Jingli Lin; Hanqing Wang; Zhaoyang Lyu; Tai Wang; Bo Dai; Xudong Xu; Jiangmiao Pang
>
> **摘要:** The advancement of Embodied AI heavily relies on large-scale, simulatable 3D scene datasets characterized by scene diversity and realistic layouts. However, existing datasets typically suffer from limitations in data scale or diversity, sanitized layouts lacking small items, and severe object collisions. To address these shortcomings, we introduce \textbf{InternScenes}, a novel large-scale simulatable indoor scene dataset comprising approximately 40,000 diverse scenes by integrating three disparate scene sources, real-world scans, procedurally generated scenes, and designer-created scenes, including 1.96M 3D objects and covering 15 common scene types and 288 object classes. We particularly preserve massive small items in the scenes, resulting in realistic and complex layouts with an average of 41.5 objects per region. Our comprehensive data processing pipeline ensures simulatability by creating real-to-sim replicas for real-world scans, enhances interactivity by incorporating interactive objects into these scenes, and resolves object collisions by physical simulations. We demonstrate the value of InternScenes with two benchmark applications: scene layout generation and point-goal navigation. Both show the new challenges posed by the complex and realistic layouts. More importantly, InternScenes paves the way for scaling up the model training for both tasks, making the generation and navigation in such complex scenes possible. We commit to open-sourcing the data, models, and benchmarks to benefit the whole community.
>
---
#### [replaced 033] DeepThinkVLA: Enhancing Reasoning Capability of Vision-Language-Action Models
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决CoT推理在VLA中效果有限的问题。通过分析两个必要条件，提出DeepThinkVLA模型提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15669](https://arxiv.org/pdf/2511.15669)**

> **作者:** Cheng Yin; Yankai Lin; Wang Xu; Sikyuen Tam; Xiangrui Zeng; Zhiyuan Liu; Zhouping Yin
>
> **备注:** 19 pages, 6 figures, conference
>
> **摘要:** Does Chain-of-Thought (CoT) reasoning genuinely improve Vision-Language-Action (VLA) models, or does it merely add overhead? Existing CoT-VLA systems report limited and inconsistent gains, yet no prior work has rigorously diagnosed when and why CoT helps robots act. Through systematic experiments, we identify two necessary conditions that must be jointly satisfied for CoT to be effective in VLA: (1) Decoding Alignment -- CoT and actions must be generated with modality-appropriate mechanisms; forcing both through a single autoregressive decoder is not merely suboptimal but actively harmful, degrading performance by 4.2 percentage points; (2) Causal Alignment -- CoT must be causally linked to task success via outcome-based optimization; without it, supervised CoT is indistinguishable from no reasoning at all under distribution shift, exhibiting a 32.0\,pp performance drop nearly identical to the 31.6\,pp drop of a reasoning-free baseline. Guided by these findings, we build DeepThinkVLA: a hybrid-attention decoder satisfies Condition~1 by pairing causal attention for language with bidirectional attention for parallel action decoding, while a two-stage SFT-then-RL pipeline satisfies Condition~2 by aligning the full reasoning--action chain with sparse task-success rewards. DeepThinkVLA achieves 97.0\% success on LIBERO, 79.0\% robustness on LIBERO-Plus (vs.\ 61.6\% for $\pi_0$-FAST), and 59.3\% success on RoboTwin~2.0, exceeding the strongest baseline by 21.7 points. Furthermore, we validate the practical effectiveness of our approach through real-world robot experiments. Code available at this https URL
>
---
#### [replaced 034] Contact-Rich Robotic Assembly in Construction via Diffusion Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人装配任务，旨在解决施工中因误差导致的装配难题。通过扩散策略学习，提升机器人在不确定环境下的装配精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.17774](https://arxiv.org/pdf/2511.17774)**

> **作者:** Salma Mozaffari; Daniel Ruan; William van den Bogert; Nima Fazeli; Sigrid Adriaenssens; Arash Adel
>
> **摘要:** Fabrication uncertainty arising from tolerance accumulation, material imperfection, and positioning errors remains a critical barrier to automated robotic assembly in construction, particularly for contact-rich manipulation tasks governed by friction and geometric constraints. This paper investigates the deployment of diffusion policy learning on construction-scale industrial robots to enable robust, high-precision assembly under such uncertainty, using tight-fitting mortise and tenon timber joinery as a representative case study. Sensory-motor diffusion policies are trained using teleoperated demonstrations collected from an industrial robotic workcell equipped with force/torque sensing. A two-phase experimental study evaluates baseline performance and robustness under randomized positional perturbations up to 10 mm, far exceeding the sub-millimeter joint clearance. The best-performing policy achieved 100% success under nominal conditions and 75% average success under uncertainty. These results provide initial evidence that diffusion policies compensate for misalignments through contact-aware control, representing a step toward robust robotic assembly in construction under tight tolerances.
>
---
#### [replaced 035] VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决规划中的不确定性问题。提出VADv2模型，通过概率规划实现端到端驾驶，提升性能。**

- **链接: [https://arxiv.org/pdf/2402.13243](https://arxiv.org/pdf/2402.13243)**

> **作者:** Bo Jiang; Shaoyu Chen; Hao Gao; Bencheng Liao; Qian Zhang; Wenyu Liu; Xinggang Wang
>
> **备注:** Accepted to ICLR 2026. Code is available at this https URL
>
> **摘要:** Learning a human-like driving policy from large-scale driving demonstrations is promising, but the uncertainty and non-deterministic nature of planning make it challenging. Existing learning-based planning methods follow a deterministic paradigm to directly regress the action, failing to cope with the uncertainty problem. In this work, we propose a probabilistic planning model for end-to-end autonomous driving, termed VADv2. We resort to a probabilistic field function to model the mapping from the action space to the probabilistic distribution. Since the planning action space is a high-dimensional continuous spatiotemporal space and hard to tackle, we first discretize the planning action space to a large planning vocabulary and then tokenize the planning vocabulary into planning tokens. Planning tokens interact with scene tokens and output the probabilistic distribution of action. Mass driving demonstrations are leveraged to supervise the distribution. VADv2 achieves state-of-the-art closed-loop performance on the CARLA Town05 benchmark, significantly outperforming existing methods, and also leads the recent Bench2Drive benchmark. We further provide comprehensive evaluations on NAVSIM and a large-scale 3DGS-based benchmark, demonstrating its effectiveness in real-world applications. Code is available at this https URL.
>
---
#### [replaced 036] Topology-Preserving Scalar Field Optimization for Boundary-Conforming Spiral Toolpaths on Multiply Connected Freeform Surfaces
- **分类: cs.RO; cs.GR**

- **简介: 该论文属于数控加工路径规划任务，解决多连通自由曲面边界一致的螺旋刀具路径优化问题，通过拓扑保持方法提升加工效率与路径质量。**

- **链接: [https://arxiv.org/pdf/2512.22502](https://arxiv.org/pdf/2512.22502)**

> **作者:** Shen Changqing; Xu Bingzhou; Qi Bosong; Zhang Xiaojian; Yan Sijie; Ding Han
>
> **备注:** Reorganized the manuscript and added more detailed explanations of the workflow and multiple case studies
>
> **摘要:** Ball-end milling path planning on multiply connected freeform surfaces is pivotal for high-quality and efficient machining of components in automotive and aerospace manufacturing. Although scalar-field-based optimization provides a unified framework for multi-objective toolpath generation, maintaining boundary conformity while eliminating zero-gradient singularities that cause iso-curve branching or termination and disrupt toolpath continuity remains challenging on multiply connected surfaces. We propose an efficient strategy to robustly enforce these constraints throughout optimization. Conformal slit mapping is employed to construct a feasible, singularity-free initial scalar field. The optimization is reformulated as a topology-preserving mesh deformation governed by boundary-synchronous updates, enabling globally optimized spacing, scallop-height uniformity, and smooth trajectory transitions. Consequently, the toolpaths are continuous, boundary-conforming, and free of self-intersections. Milling experiments demonstrate that, compared with a state-of-the-art conformal slit mapping-based method, the proposed approach increases machining efficiency by 14.24%, improves scallop-height uniformity by 5.70%, and reduces milling impact-induced vibrations by over 10%. The strategy offers broad applicability in high-performance machining scenarios.
>
---
#### [replaced 037] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Robot Task Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划领域，解决真实环境中任务规划的泛化问题。通过预训练统一PDDL域，提升机器人任务成功率和计划优化性。**

- **链接: [https://arxiv.org/pdf/2507.21545](https://arxiv.org/pdf/2507.21545)**

> **作者:** Haoming Ye; Yunxiao Xiao; Cewu Lu; Panpan Cai
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Robotic task planning in real-world environments requires reasoning over implicit constraints from language and vision. While LLMs and VLMs offer strong priors, they struggle with long-horizon structure and symbolic grounding. Existing methods that combine LLMs with symbolic planning often rely on handcrafted or narrow domains, limiting generalization. We propose UniDomain, a framework that pre-trains a PDDL domain from robot manipulation demonstrations and applies it for online robotic task planning. It extracts atomic domains from 12,393 manipulation videos to form a unified domain with 3137 operators, 2875 predicates, and 16481 causal edges. Given a target class of tasks, it retrieves relevant atomics from the unified domain and systematically fuses them into high-quality meta-domains to support compositional generalization in planning. Experiments on diverse real-world tasks show that UniDomain solves complex, unseen tasks in a zero-shot manner, achieving up to 58% higher task success and 160% improvement in plan optimality over state-of-the-art LLM and LLM-PDDL baselines.
>
---
#### [replaced 038] ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于智能体规划任务，解决现实环境中对象可操作性未明确的问题。提出DynAfford基准和ADAPT模块，增强代理对动态 affordance 的感知与适应能力。**

- **链接: [https://arxiv.org/pdf/2604.14902](https://arxiv.org/pdf/2604.14902)**

> **作者:** Pei-An Chen; Yong-Ching Liang; Jia-Fong Yeh; Hung-Ting Su; Yi-Ting Chen; Min Sun; Winston Hsu
>
> **摘要:** Intelligent embodied agents should not simply follow instructions, as real-world environments often involve unexpected conditions and exceptions. However, existing methods usually focus on directly executing instructions, without considering whether the target objects can actually be manipulated, meaning they fail to assess available affordances. To address this limitation, we introduce DynAfford, a benchmark that evaluates embodied agents in dynamic environments where object affordances may change over time and are not specified in the instruction. DynAfford requires agents to perceive object states, infer implicit preconditions, and adapt their actions accordingly. To enable this capability, we introduce ADAPT, a plug-and-play module that augments existing planners with explicit affordance reasoning. Experiments demonstrate that incorporating ADAPT significantly improves robustness and task success across both seen and unseen environments. We also show that a domain-adapted, LoRA-finetuned vision-language model used as the affordance inference backend outperforms a commercial LLM (GPT-4o), highlighting the importance of task-aligned affordance grounding.
>
---
#### [replaced 039] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人安全控制任务，旨在解决VLAs在真实部署中的安全问题。通过集成安全方法，约束策略以提升安全性并保持任务成功率。**

- **链接: [https://arxiv.org/pdf/2503.03480](https://arxiv.org/pdf/2503.03480)**

> **作者:** Borong Zhang; Yuhao Zhang; Jiaming Ji; Yingshan Lei; Yishuai Cai; Josef Dai; Yuanpei Chen; Yaodong Yang
>
> **备注:** Accepted by NeurIPS 2025 Spotlight Presentation
>
> **摘要:** Vision-language-action models (VLAs) show potential as generalist robot policies. However, these models pose extreme safety challenges during real-world deployment, including the risk of harm to the environment, the robot itself, and humans. How can safety constraints be explicitly integrated into VLAs? We address this by exploring an integrated safety approach (ISA), systematically modeling safety requirements, then actively eliciting diverse unsafe behaviors, effectively constraining VLA policies via safe reinforcement learning, and rigorously assuring their safety through targeted evaluations. Leveraging the constrained Markov decision process (CMDP) paradigm, ISA optimizes VLAs from a min-max perspective against elicited safety risks. Thus, policies aligned through this comprehensive approach achieve the following key features: (I) effective safety-performance trade-offs, reducing the cumulative cost of safety violations by 83.58% compared to the state-of-the-art method, while also maintaining task success rate (+3.85%). (II) strong safety assurance, with the ability to mitigate long-tail risks and handle extreme failure scenarios. (III) robust generalization of learned safety behaviors to various out-of-distribution perturbations. The effectiveness is evaluated on long-horizon mobile manipulation tasks. Our data, models and newly proposed benchmark environment are available at this https URL.
>
---
#### [replaced 040] J-PARSE: Jacobian-based Projection Algorithm for Resolving Singularities Effectively in Inverse Kinematic Control of Serial Manipulators
- **分类: cs.RO**

- **简介: 该论文提出J-PARSE算法，解决串联机械臂逆运动学控制中的奇异问题，通过改进雅可比矩阵实现安全运动，提升控制精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2505.00306](https://arxiv.org/pdf/2505.00306)**

> **作者:** Shivani Guptasarma; Matthew Strong; Honghao Zhen; Monroe Kennedy III
>
> **备注:** 21 pages, 13 figures. v1: Fig. 1 replaced with faster-loading version. v2: Website at this https URL. v3: Proofs revised and new material added. v4: Proofs further revised and more new material added
>
> **摘要:** J-PARSE is an algorithm for smooth first-order inverse kinematic control of a serial manipulator near kinematic singularities. The commanded end-effector velocity is interpreted component-wise, according to the available mobility in each dimension of the task space. First, a substitute "Safety" Jacobian matrix is created, keeping the aspect ratio of the manipulability ellipsoid above a threshold value. The desired motion is then projected onto non-singular and singular directions, and the latter projection scaled down by a factor informed by the threshold value. A right-inverse of the non-singular Safety Jacobian is applied to the modified command. In the absence of joint limits and collisions, this ensures safe transition into and out of low-rank configurations, guaranteeing asymptotic stability for reaching target poses within the workspace, and stability for those outside. Velocity control with J-PARSE is benchmarked against approaches from the literature, and shows high accuracy in reaching and leaving singular target poses. By expanding the available workspace of manipulators, the algorithm finds applications in teleoperation, servoing, and learning. Videos and code are available at this https URL.
>
---
#### [replaced 041] From Kinematics to Dynamics: Learning to Refine Hybrid Plans for Physically Feasible Execution
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决混合离散-连续动作序列与物理约束不匹配的问题。通过强化学习优化轨迹，提升动态可行性。**

- **链接: [https://arxiv.org/pdf/2604.12474](https://arxiv.org/pdf/2604.12474)**

> **作者:** Lidor Erez; Shahaf S. Shperberg; Ayal Taitler
>
> **摘要:** In many robotic tasks, agents must traverse a sequence of spatial regions to complete a mission. Such problems are inherently mixed discrete-continuous: a high-level action sequence and a physically feasible continuous trajectory. The resulting trajectory and action sequence must also satisfy problem constraints such as deadlines, time windows, and velocity or acceleration limits. While hybrid temporal planners attempt to address this challenge, they typically model motion using linear (first-order) dynamics, which cannot guarantee that the resulting plan respects the robot's true physical constraints. Consequently, even when the high-level action sequence is fixed, producing a dynamically feasible trajectory becomes a bi-level optimization problem. We address this problem via reinforcement learning in continuous space. We define a Markov Decision Process that explicitly incorporates analytical second-order constraints and use it to refine first-order plans generated by a hybrid planner. Our results show that this approach can reliably recover physical feasibility and effectively bridge the gap between a planner's initial first-order trajectory and the dynamics required for real execution.
>
---
#### [replaced 042] AeroScene: Progressive Scene Synthesis for Aerial Robotics
- **分类: cs.RO**

- **简介: 该论文提出AeroScene，一种用于无人机仿真的3D场景生成模型，解决手动构建环境耗时且难以扩展的问题。通过层次化扩散模型生成物理合理、语义一致的场景，提升无人机导航等任务的仿真效果。**

- **链接: [https://arxiv.org/pdf/2603.23224](https://arxiv.org/pdf/2603.23224)**

> **作者:** Nghia Vu; Tuong Do; Dzung Tran; Binh X. Nguyen; Hoan Nguyen; Erman Tjiputra; Quang D. Tran; Hai-Nguyen Nguyen; Anh Nguyen
>
> **备注:** 8 pages. Accepted to ICRA 2026
>
> **摘要:** Generative models have shown substantial impact across multiple domains, their potential for scene synthesis remains underexplored in robotics. This gap is more evident in drone simulators, where simulation environments still rely heavily on manual efforts, which are time-consuming to create and difficult to scale. In this work, we introduce AeroScene, a hierarchical diffusion model for progressive 3D scene synthesis. Our approach leverages hierarchy-aware tokenization and multi-branch feature extraction to reason across both global layouts and local details, ensuring physical plausibility and semantic consistency. This makes AeroScene particularly suited for generating realistic scenes for aerial robotics tasks such as navigation, landing, and perching. We demonstrate its effectiveness through extensive experiments on our newly collected dataset and a public benchmark, showing that AeroScene significantly outperforms prior methods. Furthermore, we use AeroScene to generate a large-scale dataset of over 1,000 physics-ready, high fidelity 3D scenes that can be directly integrated into NVIDIA Isaac Sim. Finally, we illustrate the utility of these generated environments on downstream drone navigation tasks. Our code and dataset are publicly available at this http URL
>
---
#### [replaced 043] ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling
- **分类: cs.RO**

- **简介: 该论文属于机器人安全部署任务，旨在解决工业场景中策略验证难题。通过两阶段采样方法发现失败边界，提升策略安全性。**

- **链接: [https://arxiv.org/pdf/2603.22126](https://arxiv.org/pdf/2603.22126)**

> **作者:** Azuki Kim
>
> **备注:** 15 pages, 5 figures, 8-entry VLA leaderboard, 4-robot cross-robot analysis (Franka Panda + UR3e + UR5e + UR10e), open-source code and 50K+ failure pattern dataset at this https URL. v4: added 8 references (LIBERO-PRO, LIBERO-Plus, vla-eval, FIPER, RoboMIND, RoboArena, RobotArena-Inf, RoboCasa365) + new Section 2.6 distinguishing intra-sim vs cross-sim collapse
>
> **摘要:** Deploying learned robot manipulation policies in industrial settings requires rigorous pre-deployment validation, yet exhaustive testing across high-dimensional parameter spaces is intractable. We present ROBOGATE, a deployment risk management framework that combines physics-based simulation with a two-stage adaptive sampling strategy to efficiently discover failure boundaries in the operational parameter space. Stage 1 employs Latin Hypercube Sampling (LHS) across an 8-dimensional parameter space; Stage 2 applies boundary-focused sampling concentrated in the 30-70% success rate transition zone. Using NVIDIA Isaac Sim with Newton physics, we evaluate a scripted pick-and-place controller across four robot embodiments -- Franka Panda (7-DOF), UR3e (6-DOF), UR5e (6-DOF), and UR10e (6-DOF) -- totaling over 50,000 experiments. Our logistic regression risk model achieves AUC 0.780 and identifies a closed-form failure boundary equation. We further benchmark eight VLA (Vision-Language-Action) policies, including a fine-tuned NVIDIA GR00T N1.6 (3B) trained on LIBERO-Spatial for 20K steps. The same checkpoint achieves 97.65% success rate on LIBERO (MuJoCo) but 0% on RoboGate's 68 industrial scenarios in NVIDIA Isaac Sim -- a 97.65 percentage point cross-simulator gap on a single model that underscores the deployment validation challenge. Inspired by the validation-layer paradigm NVIDIA codified for quantum computing with Ising, ROBOGATE provides this validation layer for Physical AI. Open-source.
>
---
#### [replaced 044] A Real-World Grasping-in-Clutter Performance Evaluation Benchmark for Robotic Food Waste Sorting
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决食品垃圾分拣中杂乱环境下的抓取难题。提出GRAB基准，包含多样数据集和评估方法，分析抓取失败原因，提升抓取性能。**

- **链接: [https://arxiv.org/pdf/2602.18835](https://arxiv.org/pdf/2602.18835)**

> **作者:** Moniesha Thilakarathna; Xing Wang; Min Wang; David Hinwood; Shuangzhe Liu; Damith Herath
>
> **备注:** 26 pages, 13 Figures, 4 Tables. Revised manuscript with a clearer state-of-the-art discussion, reorganized methodology, and updated figures and content
>
> **摘要:** Food waste management is critical for sustainability, yet inorganic contaminants hinder recycling potential. Robotic automation accelerates sorting through automated contaminant removal. Nevertheless, the diverse and unpredictable nature of contaminants introduces major challenges for reliable robotic grasping. Grasp performance benchmarking provides a rigorous methodology for evaluating these challenges in underexplored field contexts like food waste sorting. However, existing approaches suffer from limited simulation datasets, over-reliance on simplistic metrics like success rate, inability to account for object-related pre-grasp conditions, and lack of comprehensive failure analysis. To address these gaps, this work introduces GRAB, a real-world grasping-in-clutter (GIC) performance benchmark incorporating: (1) diverse deformable object datasets, (2) advanced 6D grasp pose estimation, and (3) explicit evaluation of pre-grasp conditions through graspability metrics. The benchmark compares industrial grasping across three gripper modalities through 1,750 grasp attempts across four randomized clutter levels. Results reveal a clear hierarchy among graspability parameters, with object quality emerging as the dominant factor governing grasp performance across modalities. Failure mode analysis shows that physical interaction constraints, rather than perception or control limitations, constitute the primary source of grasp failures in cluttered environments. By enabling identification of dominant factors influencing grasp performance, GRAB provides a principled foundation for designing robust, adaptive grasping systems for complex, cluttered food waste sorting.
>
---
#### [replaced 045] World-Value-Action Model: Implicit Planning for Vision-Language-Action Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-语言-动作系统任务，解决长时序决策和轨迹规划问题。提出WAV模型，通过隐式规划提升复杂任务性能。**

- **链接: [https://arxiv.org/pdf/2604.14732](https://arxiv.org/pdf/2604.14732)**

> **作者:** Runze Li; Hongyin Zhang; Junxi Jin; Qixin Zeng; Zifeng Zhuang; Yiqi Tang; Shangke Lyu; Donglin Wang
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for building embodied agents that ground perception and language into action. However, most existing approaches rely on direct action prediction, lacking the ability to reason over long-horizon trajectories and evaluate their consequences, which limits performance in complex decision-making tasks. In this work, we introduce World-Value-Action (WAV) model, a unified framework that enables implicit planning in VLA systems. Rather than performing explicit trajectory optimization, WAV model learn a structured latent representation of future trajectories conditioned on visual observations and language instructions. A learned world model predicts future states, while a trajectory value function evaluates their long-horizon utility. Action generation is then formulated as inference in this latent space, where the model progressively concentrates probability mass on high-value and dynamically feasible trajectories. We provide a theoretical perspective showing that planning directly in action space suffers from an exponential decay in the probability of feasible trajectories as the horizon increases. In contrast, latent-space inference reshapes the search distribution toward feasible regions, enabling efficient long-horizon decision making. Extensive simulations and real-world experiments demonstrate that the WAV model consistently outperforms state-of-the-art methods, achieving significant improvements in task success rate, generalization ability, and robustness, especially in long-horizon and compositional scenarios. Code is available at this https URL.
>
---
