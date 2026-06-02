# 机器人 cs.RO

- **最新发布 121 篇**

- **更新 76 篇**

## 最新发布

#### [new 001] FATE-VLA:Failue-aware test generation for vision-language-action models
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型评估不足的问题。通过主动发现故障的测试生成方法，提升模型鲁棒性评估效果。**

- **链接: [https://arxiv.org/pdf/2606.02307](https://arxiv.org/pdf/2606.02307)**

> **作者:** Arusa Kanwal; Pablo Valle; Shaukat Ali; Aitor Arrieta
>
> **摘要:** Vision-Language-Action (VLA) models are increasingly used as generalist robot policies, yet their evaluation still relies largely on static benchmarks that randomly sample task scenes. In high-dimensional embodied spaces, failures are sparse and clustered, so static benchmarking can underestimate robustness risks. We reframe VLA evaluation as an active failure-discovery problem and propose a failure-aware test-generation approach that combines diversity-driven exploration with surrogate models learned from observed executions. The method steers testing toward high-risk yet diverse scene regions. Across four state-of-the-art VLA models, it uncovers substantially more failures (up to +29.7 % over selected baselines) while revealing more diverse failure modes. This mean that, for instance, in the case of GR00T-N1.6, success rate dropped from 64.4% to 34.7%. More broadly, our findings call for a shift in VLA evaluation: from passive measurement on fixed task suites to adaptive, failure-seeking test generation that exposes the structure of model weaknesses before deployment.
>
---
#### [new 002] Towards Precise Intent-Aligned VLA Aerial Navigation via Expert-Guided GRPO
- **分类: cs.RO**

- **简介: 该论文属于无人机导航任务，解决指令对齐与强化学习效率问题。提出EG-GRPO框架，结合专家数据提升导航成功率和意图对齐度。**

- **链接: [https://arxiv.org/pdf/2606.02313](https://arxiv.org/pdf/2606.02313)**

> **作者:** Tianyang Chen; Wenjun Li; Xin zhou; Yuze Wu; Fei Gao
>
> **摘要:** Vision-Language-Action (VLA) models offer a promising end-to-end paradigm for unmanned aerial vehicles (UAVs) to accomplish complex tasks specified by fine-grained instructions. However, standard supervised fine-tuning (SFT) suffers from data scarcity, limited generalization, and weak supervision for nuanced and complicated human intents. Reinforcement fine-tuning offers a natural way to mitigate these challenges and align policy behaviors with human intents through designable feedback, but applying it to aerial navigation remains challenging due to inefficient exploration in expansive continuous spaces. To address these challenges, we introduce an efficient reinforcement learning (RL) framework for VLA-based aerial navigation. At its core, we propose EG-GRPO (Expert-Guided Group Relative Policy Optimization) to augment online rollouts with few-shot expert data. Additionally, we design a heterogeneous pipeline enabling parallel simulation and inference, which reduces rollout time by 43.5%. Across multiple tasks specified by complex human intents, EG-GRPO improves the success rate to 2.13x that of the SFT baseline, while improving intent alignment performance by 60.9%. These results demonstrate that our framework can move aerial navigation toward precise intent-aligned flight.
>
---
#### [new 003] BEVIO: Efficient Bird's-Eye-View based Sparse-Update Visual-Inertial Odometry for Lunar Day-Night Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉惯性里程计任务，解决行星探测器在极端资源限制下稀疏视觉更新的导航问题。提出BEVIO方法，实现低频视觉更新下的可靠昼夜导航。**

- **链接: [https://arxiv.org/pdf/2606.00709](https://arxiv.org/pdf/2606.00709)**

> **作者:** Mohit Singh; Shehryar Khattak; Ashish Goel; Michael Paton; Kostas Alexis; Issa A. Nesnas
>
> **备注:** Accepted at the 2026 IEEE International Conference on Robotics and Automation, Vienna
>
> **摘要:** Visual-Inertial Odometry (VIO) provides smooth, high-rate state estimates and has been widely used for robotic navigation in both terrestrial and planetary applications. However, its performance is typically dependent on the frequency of visual updates, which is a challenge for planetary rovers operating under extreme resource constraints and low frame rates. This work investigates enabling reliable VIO with very sparse visual updates for lunar rover applications, addressing both day and night-time operations where feature associations become especially difficult under self-illumination conditions. We propose a Bird's Eye View (BEV)-based image matching scheme that remains robust to larger inter-frame motions and more reliable feature matching despite significant visual appearance changes. We extensively evaluate our proposed approach, BEVIO, through high-fidelity photorealistic lunar and real-time robotic experiments conducted using a half-scale lunar rover, in a long-term day-night deployment at Plaster City, CA, USA. The results demonstrate that our method enables reliable day and nighttime self-illuminated traverses at visual update rates as low as 0.25 Hz, underscoring its suitability for navigation on power- and compute-limited lunar rovers.
>
---
#### [new 004] Implicit Drifting Policy: One-Step Action Generation via Conditional Expert Geometry
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于模仿学习任务，旨在解决高频率机器人控制中生成动作策略的延迟问题。提出IDP框架，在不显式估计漂移场的情况下，通过几何结构约束提升动作生成质量。**

- **链接: [https://arxiv.org/pdf/2606.01098](https://arxiv.org/pdf/2606.01098)**

> **作者:** Zemin Yang; Yaoyu He; Yiming Zhong; Yuhao Zhang; Xinge Zhu; Yao Mu; Qingqiu Huang; Yuexin Ma
>
> **摘要:** Generative action policies based on diffusion or flow matching excel in behavior cloning, yet their iterative sampling is prohibitive for high-frequency robot control. While recent one-step formulations alleviate this latency, they inevitably discard the intermediate trajectory evolution that provides crucial action correction. Directly recovering this mechanism by explicitly estimating a training-time drifting field is mathematically ill-posed due to extreme conditional demonstration sparsity. We introduce Implicit Drifting Policy (IDP), a one-step imitation learning framework that brings the training-time correction of Drifting into policy learning without explicit vector field estimation. IDP extracts a conditional expert geometry from the local variation of observation-similar expert actions, and compares it against a global reference geometry to isolate condition-specific constraints. This local geometric structure adaptively weights a scalar potential objective. Combined with an expert-proximal terminal evaluation, IDP directly enforces manifold constraints on the one-step generator during training. Extensive evaluations across 2D, 3D, and real-world manipulation tasks show IDP effectively maintains adherence to valid action manifolds, improving upon explicit drifting methods and achieving competitive performance with strong one-step baselines.
>
---
#### [new 005] Learning Action-Conditional and Object-Centric Gaussian Splatting World Models for Rigid Objects
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出MRO-GWM模型，用于学习刚体对象的动态世界模型，解决多物体场景下的动作预测与控制问题。通过对象中心高斯表示和时空变换器实现精准预测。**

- **链接: [https://arxiv.org/pdf/2606.01950](https://arxiv.org/pdf/2606.01950)**

> **作者:** Jens U. Kreber; Lukas Mack; Joerg Stueckler
>
> **摘要:** World models enable intelligent agents to predict the consequences of their actions on the environment. In this paper, we propose Multi Rigid Object Gaussian World Model (MRO-GWM), a novel model that learns action-conditional dynamics of rigid objects in 3D. By representing the scene by object-centric Gaussians, we can represent arbitrary object shapes and multi-object scenes. We develop a novel spatio-temporal transformer architecture that predicts future rigid body motion from a history of object Gaussians and future actions. Objects are represented by their Gaussians in a canonical frame, which allows for describing object motion as rigid body transformation. Our model is trained on reconstructions from multiple viewpoints, which requires the model to handle partial observations of objects due to occlusions. We analyze prediction performance of our approach on synthetic datasets composed of typical household objects with multi-object dynamics and interactions by a robot end effector. We also evaluate our model in model-predictive control for non-prehensile manipulation in simulation.
>
---
#### [new 006] Position: Good Embodied Reward Models Need Bad Behavior Data
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决奖励模型偏差问题。指出当前模型依赖成功数据，缺乏负面行为数据，导致奖励不准确。提出需收集和利用负面数据以提升模型对人类偏好的对齐。**

- **链接: [https://arxiv.org/pdf/2606.01036](https://arxiv.org/pdf/2606.01036)**

> **作者:** Ran Tian; Yilin Wu; Andrea Bajcsy
>
> **备注:** This position paper has been accepted by the ICML 2026 position track as a spotlight paper
>
> **摘要:** This position paper argues that to obtain reliable embodied reward models, the community must invest in ``bad'' robot data: failed, suboptimal, error-prone, and even hazardous behaviors. While reward models are central to any foundation model's lifecycle, today's embodied reward models are trained primarily on successful behaviors. We analyze three state-of-the-art embodied reward models and find that they systematically over-reward behaviors that real human evaluators would penalize, including unsafe interactions, poor execution, and shortcut strategies that only superficially satisfy tasks. We attribute these failures to a key data gap: the scarcity of negative embodied data which is costly to collect and often filtered out or withheld in existing robotics datasets. Furthermore, we show that even modest exposure to real bad behavior data can improve alignment with human preferences and reduce costly false positives. We therefore call on the embodied AI community to curate and release their bad robot data, build synthetic bad data generation engines, develop more decentralized physical evaluation systems, and design benchmarks for fine-grained embodied reward model evaluations.
>
---
#### [new 007] Dexterity-BEV: Aligning 3D World and Actions for Generalizable Robot Policies Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在解决2D视觉输入与3D操作不匹配的问题。通过引入3D对齐表示和BEV框架，提升策略的泛化能力与一致性。**

- **链接: [https://arxiv.org/pdf/2606.02274](https://arxiv.org/pdf/2606.02274)**

> **作者:** Huayi Zhou; Wei Gao; Dekun Lu; Ruiji Liu; Zhanqi Zhang; Ziyang Zhang; Jian Chen; Wenlve Zhou; Sheng Xu; Shumin Li; Kangyi Guo; Shichen Xu; Zixin Huang; Yongyi Su; Kui Jia
>
> **备注:** under review
>
> **摘要:** End-to-end manipulation policies, combined with web-scale pretrained Vision-Language Models (VLMs), show the promise for generalizable and dexterous robotic manipulation. However, they inherit two key limitations from 2D foundation models: 1) the reliance on 2D RGB inputs that ignores the intrinsically 3D nature of manipulation; and 2) the lack of spatial 3D alignment between input-output spaces as well as across diverse robot embodiments, camera setups, and trajectory datasets. In this paper, we present a series of contributions to address these issues. First, we introduce aligned vertex map and vertex spectrum -- a pixel-wise 3D representation that elevates 2D visual inputs to 3D, using camera calibration and optional depth. This novel input representation marries 3D awareness with the generalization of 2D large VLMs. Then, we propose to align the inputs and outputs of manipulation policies by expressing per-pixel 3D information of each camera view and robot actions to a shared coordinate. Based on this, we designate a canonical Bird's-Eye-View (BEV) alignment frame and innovatively propose to construct BEV images, producing a view-invariant representation robust to camera pose variations. To enable training and evaluation at scale, we develop a comprehensive data processing pipeline to perform such alignments; we also introduce a novel temporal alignment scheme for trajectories across diverse robots, human operators, and datasets. These contributions collectively mitigate input and output spatial-temporal misalignments, improving the consistency and generalization for real-world manipulation. Pretrained checkpoint, source code and data processing pipeline are available in this https URL.
>
---
#### [new 008] Shape Your Body: Value Gradients for Multi-Embodiment Robot Design
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人设计任务，旨在解决多形态机器人优化问题。通过训练通用价值函数，利用梯度优化新机器人设计，提升设计效率与性能分析能力。**

- **链接: [https://arxiv.org/pdf/2606.00702](https://arxiv.org/pdf/2606.00702)**

> **作者:** Nico Bohlinger; Jan Peters
>
> **摘要:** We propose to turn generalist multi-embodiment value functions into reusable models for robot design. Instead of running a new reinforcement learning co-design loop for each robot, we first train an embodiment-aware policy and value function across many robot designs. After training, the frozen value function is used as a differentiable surrogate to optimize candidate embodiments through value gradients. We evaluate our approach across different robot design settings, from perturbed single robots to held-out robots across morphology classes, with single models trained on up to 50 robots and design spaces of over 1100 continuous embodiment parameters. Beyond optimizing complete embodiments, we show that value gradients can identify performance-limiting design and control parameters, enabling both the optimization and the analysis of new robot designs.
>
---
#### [new 009] AI-IoT-Robotics Integration: Survey of Frameworks, Emerging Trends, and the Path Toward Connected Robotics
- **分类: cs.RO; cs.AI; cs.NI; eess.SY**

- **简介: 该论文属于融合AI、IoT与机器人技术的综述任务，旨在解决三者集成框架不足的问题，提出模块化架构并分析其应用前景。**

- **链接: [https://arxiv.org/pdf/2606.01015](https://arxiv.org/pdf/2606.01015)**

> **作者:** Ranulfo Bezerra; Satoshi Tadokoro; Kazunori Ohno
>
> **备注:** 15 pages, 3 figures, 3 tables. Published in IEEE Internet of Things Journal
>
> **摘要:** The convergence of Artificial Intelligence, the Internet of Things, and Robotics is no longer a futuristic vision; it is rapidly becoming the foundation of real-time, intelligent, and context-aware systems. AI enables perception and reasoning, IoT provides scalable sensing and communication, and robotics delivers embodied actuation. Despite significant progress in pairwise combinations such as AIoT and the Internet of Robotic Things (IoRT), there remains a lack of unified design frameworks that fully integrate all three. This survey synthesizes the state-of-the-art across these domains, emphasizing the emerging role of Small Language Models (SLMs) at the edge and Large Language Models (LLMs) in the cloud for distributed cognition and autonomous decision-making. We propose a modular system architecture that aligns with these trends, analyze persistent gaps in interoperability and feedback control, and classify existing work by integration depth. Our review highlights how hybrid SLM-LLM systems, when coupled with IoT infrastructure and robotic agents, can address challenges in real-time adaptation, scalability, and reliability. This work offers a conceptual and technical roadmap for designing next-generation AI-IoT-Robotic ecosystems that are modular, interpretable, and capable of learning within dynamic environments, paving the way for the emerging paradigm of Connected Robotics and Physical AI.
>
---
#### [new 010] Completion at the Boundary (CaB): Deployable Switching with Completion-Aware Control under Limited Calibration
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作代理任务，解决指令完成检测问题。针对短复合指令中的手柄失误，提出CaB方法，通过边界阶段标记实现稳定切换与控制。**

- **链接: [https://arxiv.org/pdf/2606.00145](https://arxiv.org/pdf/2606.00145)**

> **作者:** Yusuke Sano; Takeshi Itoga
>
> **摘要:** Vision-language-action (VLA) agents can execute natural-language instructions, yet deployed systems still lack an operational interface: deciding when the instruction is complete. This gap is acute in short composites ("do A, then B"), where mistimed handoffs cascade into downstream failures. Completion is inherently closed-loop because switching is an intervention that changes the instruction context and thus future actions and observations. We study completion under a deployable low-calibration regime motivated by open-ended instruction spaces, enforcing no test-time relearning and a single globally calibrated switching rule selected once on development set and reused unchanged on test set. Under this constraint, collapsing asymmetric boundary evidence into a single scalar can be brittle under polarity shifts across tasks. We propose Completion at the Boundary (CaB), which predicts an event-local completion object in the form of Boundary-Phase Tokens (Before/Hit/After), retaining two-sided boundary evidence under this discipline. CaB-When converts this completion object into a minimal, auditable switching decision (when), while CaB-How reuses the same completion object to condition action generation for boundary-stable control through handoffs (how). Using an intervention-aware E1/E2 protocol, we show that CaB improves composite execution and handoff quality on a first-person Minecraft VLA benchmark under matched capacity and deployability constraints.
>
---
#### [new 011] AFUN: Towards an Affordance Foundation Model for Functionality Understanding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决开放环境中物体功能理解的问题。通过预测交互位置和3D运动轨迹，提出一种新的仿生基础模型，实现跨环境、对象和任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.02551](https://arxiv.org/pdf/2606.02551)**

> **作者:** Zhaoning Wang; Yi Zhong; Jiawei Fu; Henrik I. Christensen; Jun Gao
>
> **摘要:** Affordance understanding bridges visual perception and physical action, serving as an explainable interface for robot manipulation in open and unstructured real-world environments. Yet, building an affordance foundation model that not only understands where and how the interaction should happen, but also generalizes across diverse environments, objects, and tasks, remains a long-standing research challenge. Existing methods typically address only part of this challenge, either localizing task-relevant regions without specifying executable motion, or predicting motion but with limited scalability. In this paper, we present ourmodel, a step towards an affordance foundation model for functionality understanding. From a single RGB-D observation and a language task description, ourmodel predicts a task-conditional functional mask (where to interact) and a 3D post-contact motion curve (how to interact). To support open-world generalization, we build a large-scale standardized data pipeline that converts heterogeneous robot, human, simulation, and real-world scan data into a shared affordance schema with language, masks, and object-centric 3D motion labels. We evaluate ourmodel from three aspects: for affordance segmentation, ourmodel outperforms all baselines by a large margin across 8 test sets from 4 benchmarks, improving mean gIoU/cIoU by +23.9/+26.3; for contact-point prediction, it predicts substantially more accurate points, with a 12.7--61.3% hit-rate gain over the best baseline; and for 3D motion, it achieves the best performance on all three test sets. ourmodel can be deployed for real-world robot manipulation without finetuning for robot embodiment or using task-specific heuristics, demonstrating the ability to adapt to open-world affordance tasks. Project page: this https URL
>
---
#### [new 012] Hierarchical Object Representation for Spatial Robot Perception: Points, Meshes, and Superquadrics
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决3DSG中物体几何表示不足的问题。通过构建分层物体表示，融合点云、网格和超二次曲面，提升重建精度与导航安全性。**

- **链接: [https://arxiv.org/pdf/2606.01545](https://arxiv.org/pdf/2606.01545)**

> **作者:** Ceng Zhang; Wan Su; Mohamed Samshad; Gregory S. Chirikjian; Rajat Talak
>
> **备注:** 18 pages, 5 figures, 4 tables
>
> **摘要:** Hierarchical 3D Scene Graphs (3DSG) have emerged as an actionable and scalable representation for long-term autonomy incorporating metric, semantic, and topological information in the scene. However, the question of geometric representation of objects in 3DSG has been overlooked as most methods use simplified geometric models such as partial point clouds or 3D bounding boxes. In this work, we introduce a hierarchical object representation that can be leveraged for high-fidelity object-level reconstruction, object-based robust re-localization or map alignment, and efficient and analytical collision checking for safe robot navigation planning in dense and cluttered environments. The representation is structurally organized into four distinct layers, progressively abstracting the scene from raw sensor data to dense 3D meshes to analytical primitives such as superquadrics, which provide a sparse and analytical representation for object geometry. We develop a pipeline that builds the hierarchical object representation from RGB-D image stream captured by a robot, and demonstrate its working in real-world open-set object scenes in both indoor and outdoor environments. Extensive experiments across diverse datasets including HOPE, ReplicaCAD, Kimera-Multi, and NUS Campus Dataset collected using Unitree B2 Robot validate our pipeline in both indoor and outdoor environments. We show that our superquadric-based map alignment method outperforms the current state-of-the-art object based map alignment method ROMAN. Our code can be found at this https URL.
>
---
#### [new 013] From Human Videos to Robot Manipulation: A Survey on Scalable Vision-Language-Action Learning with Human-Centric Data
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言-动作学习任务，旨在利用人类视频提升机器人操作能力。解决如何将人类视频转化为VLA模型有效知识的问题，通过四类方法提取动作信息，并指出三大挑战。**

- **链接: [https://arxiv.org/pdf/2606.00054](https://arxiv.org/pdf/2606.00054)**

> **作者:** Zhiyuan Feng; Qixiu Li; Huizhi Liang; Rushuai Yang; Yichao Shen; Zhiying Du; Zhaowei Zhang; Yu Deng; Li Zhao; Hao Zhao; Zongqing Lu; Oier Mees; Marc Pollefeys; Jiaolong Yang; Baining Guo
>
> **备注:** Accepted to IJCAI 2026 Survey Track. Project page: this https URL
>
> **摘要:** Recent progress in generalizable embodied control has been driven by large-scale pretraining of Vision-Language-Action (VLA) models. However, most existing approaches rely on large collections of robot demonstrations, which are costly to obtain and tightly coupled to specific embodiments. Human videos, by contrast, are abundant and capture rich interactions, providing diverse semantic and physical cues for real-world manipulation. Yet, embodiment differences and the frequent absence of task-aligned annotations make their direct use in VLA models challenging. This survey provides a unified view of how human videos are transformed into effective knowledge for VLA models. We categorize existing approaches into four classes based on the action-related information they derive: (i) latent action representations that encode inter-frame changes; (ii) predictive world models that forecast future frames; (iii) explicit 2D supervision that extracts image-plane cues; and (iv) explicit 3D reconstruction that recovers geometry or motion. Beyond this taxonomy, we highlight three key open challenges in this area: structuring unstructured videos into training-ready episodes, grounding video-derived supervision into robot-executable actions under embodiment and viewpoint heterogeneity, and designing evaluation protocols that better predict real-world deployment performance and transfer efficiency, thereby informing future research directions. A curated list of papers and resources is available at this https URL.
>
---
#### [new 014] PACE: Phase-Aware Chunk Execution for Robot Policies with Action Chunking
- **分类: cs.RO**

- **简介: 该论文提出PACE方法，解决机器人策略执行中执行窗口选择的问题，通过分析动作块内的速度变化自动调整执行长度，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2606.00537](https://arxiv.org/pdf/2606.00537)**

> **作者:** Junnan Nie; Jiayi Li; Jiachen Zhang; Junyi Lao; Chenghao Liu; Tianle Zhang; Songfang Huang
>
> **备注:** 21 pages, 7 figures, 6 tables. Preprint
>
> **摘要:** Recent vision-language-action and diffusion-based robot policies often use action chunking, where each policy query predicts a sequence of future actions and the robot executes an open-loop prefix before re-querying. While this interface improves local motion continuity, deployment still requires choosing the execution horizon: how much of each predicted chunk should be executed before acquiring a new observation. However, our experiments show that success is strongly task-dependent and non-monotonic with respect to the execution horizon, making a single constant horizon an unreliable deployment rule. We propose PACE (Phase-Aware Chunk Execution), a training-free test-time execution method that selects the execution horizon online from the predicted chunk itself. PACE exploits the phase-dependent kinematic structure of manipulation trajectories by identifying low-speed transition points in the predicted speed profile and using them as candidate replanning boundaries. Because PACE uses only the predicted action chunk, it is plug-and-play and requires no retraining or access to policy internals. We validate PACE through large-scale evaluations in both simulation and real-robot settings. On 50 RoboTwin2.0 tasks, PACE raises the average success rate from 57.8% to 64.2%. In real-robot experiments on bimanual ALOHA and single-arm Franka platforms, PACE improves the average task score from 60.7 to 77.7 and the average success rate from 50.7% to 70.4%. Ablations and rollout-level analyses show that PACE adapts execution horizons across manipulation phases, shortening near transitions while preserving longer execution during coherent motion.
>
---
#### [new 015] Crazyflow: An Accurate, GPU-Accelerated, Differentiable Drone Simulator in JAX
- **分类: cs.RO; cs.AI; cs.MA; eess.SY**

- **简介: 该论文提出Crazyflow，一个高速、可微的无人机模拟器，解决传统模拟器在速度、精度和多智能体支持上的不足，支持快速强化学习与大规模仿真。**

- **链接: [https://arxiv.org/pdf/2606.01478](https://arxiv.org/pdf/2606.01478)**

> **作者:** Martin Schuck; Marcel P. Rath; Yufei Hua; AbhisheK Goudar; SiQi Zhou; Angela P. Schoellig
>
> **摘要:** High-quality, large-scale synthetic data from simulations is becoming a cornerstone for pushing the capabilities of robot algorithms. While aerial robotics simulators have evolved to support specialized needs such as fidelity, differentiability, and swarms independently, a unified platform that can synthesize data across all these domains is missing. In this work, we propose Crazyflow, a simulator designed to push the limits of aerial-robotics algorithm development, from model-based to data-driven methods, gradient-based to sampling-based approaches, and single-agent to multi-agent systems. Compared to existing state-of-the-art drone simulators, it achieves speeds more than an order of magnitude faster for a single drone and can simulate thousands of swarms of 4000 drones each. Real-world experiments show Crazyflow supports both analytical-gradient-based policy learning, achieving sub-centimeter trajectory tracking accuracy without domain randomization, and sampling-based obstacle avoidance at speeds exceeding half a billion steps per second. Breaking the traditional train-then-deploy paradigm, we show that its unprecedented speed even enables in-flight reinforcement learning; we demonstrate this by throwing a physical drone into the air and training a recovery policy from scratch in 0.38 seconds, successfully stabilizing the drone. Crazyflow supports multiple levels of simulation abstraction, is directly compatible with all open-source Crazyflie models, and enables rapid reconfiguration across custom drone platforms and applications by providing a light-weight system identification pipeline. By pushing accuracy, speed, and differentiability simultaneously, Crazyflow serves as an open-source resource for synthetic data generation, with emerging capabilities for large-scale parallelization for online, in-execution learning and optimization, opening the door to novel algorithm development.
>
---
#### [new 016] Modeling Robotics Dataset Construction as an Artifact-Based Build Process
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人数据集构建任务，解决传统脚本效率低的问题。通过建模为基于工件的构建过程，提出Bagzel工具提升数据集生成效率与可重复性。**

- **链接: [https://arxiv.org/pdf/2606.00162](https://arxiv.org/pdf/2606.00162)**

> **作者:** Leon Pohl; Lukas Beer; George Sebastian; Mirko Maehlisch
>
> **备注:** Accepted 2026 IEEE 22nd International Conference on Automation Science and Engineering (CASE 2026), 6 pages, 6 figures, 2 tables
>
> **摘要:** Robotic systems generate large volumes of multimodal sensor data, but converting ROS bag recordings into machine learning datasets is often handled by ad hoc sequential scripts, creating engineering overhead and slow iteration cycles. We model dataset construction as an artifact-based build process over a dependency graph and implement this approach in Bagzel, an open-source Bazel extension for reproducible, incremental dataset generation (including nuScenes-format export). We compare Bagzel and Bagzel-xattr (server-side digest management) against a sequential rosbag2nuscenes baseline. Bagzel reduces runtime in all evaluated execution modes, with the largest gains in iterative workflows (up to 386.26x in warm builds and 7.21x in incremental builds on a 20.4 GB dataset). Across dataset sizes from 5.1 to 20.4 GB, Bagzel variants show markedly better scaling behavior than the baseline, especially in warm and incremental modes. Bagzel-xattr provides additional gains, with a mean runtime reduction of 5.9% compared to Bagzel in the input granularity study. Overall, modeling robotics dataset construction as an artifact-based build process substantially reduces dataset update latency while maintaining a deterministic build design that supports reproducibility. Bagzel is publicly available at this https URL.
>
---
#### [new 017] $τ_0$-WM: A Unified Video-Action World Model for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决动作生成与未来后果预测问题。提出$\tau_0$-WM模型，集成策略学习、视频预测和动作评估，提升操作性能。**

- **链接: [https://arxiv.org/pdf/2606.01027](https://arxiv.org/pdf/2606.01027)**

> **作者:** Pengfei Zhou; Shengcong Chen; Di Chen; Jiaxu Wang; Rongjun Jin; Bingwen Zhu; Yike Pan; Songen Gu; Kuanning Wang; Shufeng Nan; Xingyu Qiu; Chenhao Qiu; Pu Yang; Yunuo Cai; Jianxiong Gao; Yifan Li; Yanwei Fu; Xiangyu Yue; Zhi Chen; Jianlan Luo
>
> **备注:** Our project homepge: this https URL
>
> **摘要:** Robotic manipulation requires models that generate executable actions while anticipating and evaluating their future consequences before physical execution. We present $\tau_0$-World Model ($\tau_0$-WM), a unified video-action world model that integrates policy learning, video prediction, and action evaluation within a single future-predictive framework. Built on a shared video diffusion backbone, $\tau_0$-WM provides two complementary interfaces. First, a video action model jointly predicts future visual latents and continuous action chunks from multi-view observations, language instructions, and robot state. Second, an action-conditioned video simulator rolls out candidate action chunks into multi-view futures and predicts dense task-progress scores. The model is trained on approximately $27{,}300$ hours of real-robot teleoperation, UMI-style interaction, egocentric human videos, and rollout or failure trajectories using modality-specific supervision masks. At inference time, $\tau_0$-WM uses test-time computation to sample action candidates, rank them with re-denoising consistency, and invoke simulator-based rectification for low-quality candidates. On challenging long-horizon and fine-grained robotic manipulation tasks, $\tau_0$-WM shows superior performance over other relevant baselines.
>
---
#### [new 018] Closed-Form Pose Estimation of Endoluminal Medical Devices via Gradiometer-Based Electromagnetic Localization System
- **分类: cs.RO**

- **简介: 该论文属于医疗设备位姿估计任务，解决无需预标定的六自由度定位问题，提出GELS系统通过磁梯度传感器阵列实现快速高精度定位。**

- **链接: [https://arxiv.org/pdf/2606.01946](https://arxiv.org/pdf/2606.01946)**

> **作者:** Zhiwei Wu; Jiahao Luo; Yubo Pu; Siyi Wei; Yuankai Chen; Jinhui Zhang
>
> **摘要:** Embedded magnetic tracking holds highly attractive prospects for remote navigation of endoluminal medical devices. However, existing six-degree-of-freedom pose recovery approaches often require pre-calibrated workspace field maps or iterative nonlinear optimization. This letter presents a Gradiometer-Based Electromagnetic Localization System (GELS), a closed-form tracking framework that uses a compact magnetometer array as an embedded quasi-gradiometer to estimate local magnetic fields and gradient tensors. These quantities are mapped by the Euler homogeneous relation to displacements between source and array, from which multi-source Procrustes registration recovers the array orientation and position using at least three non-collinear sources. The algorithm requires known source positions and array geometry, but no pre-calibrated workspace field maps, initial pose guesses, or calibrated excitation-source moments. The recovered pose also enables a proof-of-concept sub-level dipole localization task by serving as a mobile magnetic reference frame. Benchtop experiments across sensor-array configurations and excitation modes demonstrate sequence-averaged position errors of \SI{10.80}{\milli\meter}--\SI{15.57}{\milli\meter}, a fastest update rate of \SI{14.49}{\hertz}, and a median solver runtime of \SI{172.00}{\micro\second}. A perturbation-based error propagation analysis further identifies inter-sensor inconsistency and dipole-model mismatch as the dominant accuracy limits, thereby informing future sensor array and magnetic source design for further reducing pose-estimation error.
>
---
#### [new 019] Learning Multi-Modal Trajectory Policies for Data-Efficient Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决多模态输入下的轨迹预测问题。针对数据稀缺场景，提出MATE框架，通过多模态MoE架构提升轨迹预测性能。**

- **链接: [https://arxiv.org/pdf/2606.01047](https://arxiv.org/pdf/2606.01047)**

> **作者:** Zijia Chen; Yuenan Hou; Xinhua Jiang; Yu Li; Weijie Li; Li Liu
>
> **摘要:** Robotic manipulation requires the effective integration of heterogeneous inputs, including visual observations, language instructions, and trajectory representations, to generate accurate actions. Existing transformer-based policies typically process these heterogeneous modalities within a shared parameter space, which often leads to modality interference and inefficient representation learning, especially in data-scarce scenarios. While Mixture-of-Experts (MoE) offers a scalable solution through expert specialization, conventional routing mechanisms are often sensitive to such cross-modal representation discrepancies, resulting in unstable expert assignment and expert collapse. In this work, we propose MATE (Multi-ModAl TrajEctory Policies), a novel trajectory prediction framework built upon MoE. Specifically, we introduce a Multi-Modal MoE architecture to achieve fine-grained sub-token feature decoupling, and design a cross-modal cosine router for stable and scale-invariant expert assignment across heterogeneous modalities. We further employ temperature-controlled routing and stochastic noise injection to improve expert balance and prevent premature routing collapse under scarce demonstrations. Experiments on the LIBERO benchmark show that our MATE consistently outperforms prior work under data scarcity. It achieves a 4.75% improvement in average success rate over the trajectory-guided counterpart. Real-world experiments on robotic ping-pong also suggest that the predicted trajectories can provide useful guidance for downstream robotic execution, further indicating the practical feasibility of our algorithm.
>
---
#### [new 020] RoboSemanticBench: Diagnosing Semantic Grounding in Action Prediction for VLA Models
- **分类: cs.RO**

- **简介: 该论文提出RoboSemanticBench，用于评估VLA模型在动作预测中对语义的准确理解与应用。任务是诊断语义基础是否有效指导机器人动作，解决模型在复杂指令下选择正确目标的问题。**

- **链接: [https://arxiv.org/pdf/2606.02277](https://arxiv.org/pdf/2606.02277)**

> **作者:** Bin Yu; Yao Zhang; Haishan Liu; Shijie Lian; Yuliang Wei; Xiaopeng Lin; Zhaolong Shen; Changti Wu; Ruina Hu; Bailing Wang; Cong Huang; Kai Chen
>
> **备注:** GitHub: this https URL
>
> **摘要:** Vision-language-action (VLA) models are built on the premise that semantic understanding from pretrained language or vision-language backbones should guide robot action prediction. Yet robot fine-tuning is optimized as imitation over task-specific action distributions, and many evaluations can be solved through visual or instruction-action shortcuts. We introduce RoboSemanticBench (RSB), an embodied benchmark for diagnosing semantic grounding in action prediction: whether post-trained VLA models can use complex instruction semantics to select and manipulate the correct physical target. In each episode, a robot receives a multiple-choice math or general-knowledge question, observes candidate answer blocks, and must grasp the block corresponding to the correct answer. RSB covers controlled arithmetic, grade-school mathematical understanding, and commonsense or factual understanding under four-choice and ten-choice suites. Across representative VLA models, we find that many policies learn to grasp candidate blocks but select the semantically correct block at near-random or below-random rates after controlling for grasp success, revealing a persistent gap between backbone-level semantic competence and action prediction.
>
---
#### [new 021] Network Distributed Multi-Agent Reinforcement Learning for Consensus Control of Quadcopters
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于多智能体强化学习任务，解决无人机编队控制问题。提出ND-MARL框架，通过分布式策略实现高效通信下的协同控制，具备零样本扩展性。**

- **链接: [https://arxiv.org/pdf/2606.02107](https://arxiv.org/pdf/2606.02107)**

> **作者:** Youssef Mahran; Zeyad Gamal; Aamir Ahmad; Ayman El-Badawy
>
> **备注:** This is the Author Accepted Manuscript version of a paper accepted for publication. The final published version is available via IEEE Xplore
>
> **摘要:** This paper proposes a Network Distributed Multi-Agent Reinforcement Learning (ND-MARL) framework for quadcopter consensus control. Compared to conventional multi-agent MARL formulations that rely on centralized planning or fully decentralized execution, ND-MARL incorporates the swarm communication graph into the decision process. Under a 2-Neighbor communication topology, each agent observes information of only two neighbors and outputs an action through a distributed policy. A high-level distributed consensus planner is trained using Multi-Agent Soft Actor-Critic (MASAC) and embedded in a hierarchical stack to generate reference target positions tracked by a low-level quadcopter controller. Results demonstrate smooth consensus trajectories and planner-tracker integration when compared to a centralized MARL controller. Most notably, the learned controller exhibits zero-shot scalability, as policies trained on a three-agent system are deployed to swarms of up to 250 agents under the same 2-Neighbor communication topology without retraining or fine-tuning, achieving consistent convergence with increasing steady-state spread at large team sizes due to sparse information propagation. These findings highlight ND-MARL as a stable framework for distributed, communication-aware quadcopter consensus control.
>
---
#### [new 022] The Lie We Tell: Correcting the Euclidean Fallacy in Vision Language Action Policies via Score Matching on Tangent Space
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操作任务，解决SE(3)姿态表示中的欧几里得谬误问题。通过在切空间进行分数匹配，提出Lie Diffuser Actor模型，提升轨迹的几何正确性与效率。**

- **链接: [https://arxiv.org/pdf/2606.01847](https://arxiv.org/pdf/2606.01847)**

> **作者:** Bing-Cheng Chuang; I-Hsuan Chu; Bor-Jiun Lin; YuanFu Yang; Min Sun; Chun-Yi Lee
>
> **备注:** ICML 2026 Accepted
>
> **摘要:** Diffusion-based Vision-Language-Action policies achieve remarkable success in robotic manipulation, yet commit a fundamental geometric error we term the $\textbf{Euclidean Fallacy}$: representing SE(3) poses as flat $\mathbb{R}^{12}$ vectors. This approximation induces (1) manifold drift violating SO(3) constraints, (2) broken equivariance under coordinate transformations, and (3) non-geodesic trajectories with excessive kinematic cost. We introduce $\textbf{Lie Diffuser Actor (LDA)}$, a diffusion framework operating intrinsically on SE(3). Our method injects noise through left-invariant SDEs, predicts scores in the tangent space, and retracts samples via the exponential map. This formulation eliminates manifold drift by construction while guaranteeing coordinate-frame equivariance and geodesic optimality. On CALVIN ABC$\rightarrow$D, LDA improves average task length from $3.27$ to $3.51$ ($+7.3\%$). We further validate our method on real robot and the results show that our methodology outperforms the baseline on majority tasks.
>
---
#### [new 023] Cuttlebot: a platform demonstration for complex, autonomous, bio-inspired swimmers
- **分类: cs.RO**

- **简介: 该论文属于水下机器人研究，旨在开发仿生自主游泳机器人。提出CORE平台与Cuttlebot，解决人工肌肉集成难题，实现三维运动与多轴控制。**

- **链接: [https://arxiv.org/pdf/2606.00197](https://arxiv.org/pdf/2606.00197)**

> **作者:** Alexander Nicholas White; Ang Leo Li; Alexander Yin; Derrick Roseman; Valeria Saro-Cortes; Hannah Wiswell; Aimy Wissa; Mihai Duduta
>
> **摘要:** Increasing interest in deep-sea operations and resources motivates the development of ecologically sensitive but environmentally durable robots. Dielectric elastomer actuator artificial muscles are good candidates for powering such systems due to their pressure and temperature tolerance and soft makeup, but they are difficult to integrate with robotic systems. This work presents an autonomous robotic platform: the CORE, capable of driving six artificial muscles while sensing visual and spatial information. To validate the platform, we developed the Cuttlebot - a cuttlefish-inspired robot that swims in three dimensions using undulatory fin locomotion. The Cuttlebot has four primary artificial muscles in its fins in addition to a tentacle-inspired soft gripper. The robot was evaluated in a series of tethered and untethered swimming tests, demonstrating a top speed of 2.5 centimeters per second translation and 10 degrees per second rotation. Furthermore, the CORE system was capable of driving specialized control signals into the artificial muscles to controllably output force and torque in six axes. This work provides a platform for developing complex, bio-inspired swimming robots for ocean exploration and monitoring, laying the foundation with our leading example: the Cuttlebot.
>
---
#### [new 024] Make Your VLA More Robust Without More Data By Interleaving Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，旨在提升VLA模型在长时序任务中的鲁棒性。通过将运动规划与VLA结合，解决执行误差累积和目标定位困难问题。**

- **链接: [https://arxiv.org/pdf/2606.00985](https://arxiv.org/pdf/2606.00985)**

> **作者:** Dan BW Choe; Sundhar Vinodh Sangeetha; Samuel Coogan; Shreyas Kousik
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable progress for mobile manipulation, but their performance on long-horizon tasks remains poor. These tasks are especially challenging because (1) progress toward high-level goals must be maintained across extended sequences of spatially distributed subtasks, and (2) early execution errors compound rapidly over the task horizon. These challenges persist despite finetuning on large human teleoperated mobile manipulation data, indicating that more data alone may not resolve the problem. To address these challenges, we propose MPVI: Motion Planner / VLA Interleaving, a framework that integrates model-based motion planning with VLAs to improve robustness without further training. The proposed integration enables localization and navigation to distant or occluded target objects through cluttered scenes using open-vocabulary object detection, frontier exploration and motion planning. However, such integration is non-trivial, requiring reliable switching between modules; we show one way forward via VLM-based completion checking with proprioceptive triggers. We evaluate our approach on the BEHAVIOR-1K benchmark and demonstrate 113% improvement in task progress over a top end-to-end VLA baseline. Additional details are available at the project page: this https URL.
>
---
#### [new 025] A Sonar-Visual Dataset for Cross-Modal Underwater Robot Perception
- **分类: cs.RO**

- **简介: 该论文提出SOVIS数据集，解决水下机器人跨模态感知问题，通过融合声呐与视觉数据提升感知能力。**

- **链接: [https://arxiv.org/pdf/2606.01398](https://arxiv.org/pdf/2606.01398)**

> **作者:** Weitung Chen; Phil Tinn; Per Gunnar Auran; Martin Ludvigsen; Peter Halland Haro
>
> **备注:** 6 pages, 7 figures, 3 tables. Accepted to IEEE ICRA 2026 S2S Workshop (From Sea to Space: Advancing Perception in Harsh Domains)
>
> **摘要:** Underwater robots typically use both cameras and sonar for perception to leverage the rich semantic details of vision and the robust range measurements of acoustics. However, learning to map between these modalities via cross-modal prediction remains underexplored due to limited sonar-visual paired datasets. We present SOVIS, a sonar-visual dataset for cross-modal underwater perception. SOVIS comprises over 76,000 paired frames collected across 17 dives at six sites in the Trondheimfjord, supported by an end-to-end pipeline that cleans and synchronizes the cross-modal sensor data. We also introduce an interactive annotation tool designed to accelerate the labeling process for this paired data. Finally, we demonstrate a proof-of-concept cross-modal fish detection task using a small subset of labeled data, achieving a 7x improvement in mAP@0.10 over a monocular camera baseline. SOVIS serves as the first step toward advancing cross-modal underwater perception research, enabling research directions such as dense sonar prediction from monocular images.
>
---
#### [new 026] Ontology-Guided Reasoning for Affordance-Based Explanations of Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航解释任务，解决机器人在复杂环境中如何生成可解释的导航决策问题。通过构建本体引导的 affordance 逻辑，提升解释的准确性和实用性。**

- **链接: [https://arxiv.org/pdf/2606.00117](https://arxiv.org/pdf/2606.00117)**

> **作者:** Amar Halilovic; Vahidin Hasic; Senka Krivic
>
> **摘要:** This paper proposes ontology-guided reasoning for affordance-based explanations of robot navigation. In human environments, it is not sufficient for a robot to detect that its route is blocked. It must also reason about what nearby objects afford, which state changes are possible, and which of these changes would allow it to continue safely. We address this problem by representing nearby entities, their affordances, affordance states, and qualitative spatial relations in a local affordance ontology and by evaluating hypothetical object--affordance state changes as candidate explanation factors. This yields explanations that are not only semantically grounded but also actionable. We instantiate the approach in a lightweight benchmark centered on a robot librarian scenario and evaluate it on procedurally generated navigation cases. The results show that ontology-guided reasoning identifies relevant explanation factors more accurately than a semantic-only baseline and remains robust as semantic clutter increases. Overall, the paper argues that affordance ontologies can serve not merely as semantic descriptions of the environment, but as reasoning foundations for explainability and reliable robot autonomy.
>
---
#### [new 027] Permissive Safety Through Trusted Inference: Verifiable Belief-Space Neural Safety Filters for Assured Interactive Robotics
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于交互式机器人安全领域，解决BeliefSF在运行时推理误差导致的安全保障问题。通过结合置信预测，提升安全过滤器的宽松性与可靠性。**

- **链接: [https://arxiv.org/pdf/2606.02562](https://arxiv.org/pdf/2606.02562)**

> **作者:** Haimin Hu
>
> **备注:** Accepted to the 17th World Symposium on the Algorithmic Foundations of Robotics (WAFR 2026)
>
> **摘要:** Autonomous robots that interact with people must make safe and efficient decisions under human-induced uncertainty, such as their preferences, goals, competency, and willingness to cooperate. Safety filters are a popular approach for ensuring safety in interactive robotics, since their modular design separates safety from performance, allowing robots to operate safely around people with minimal impact on task efficiency. While traditional safety filters typically operate only in the physical space, neglecting the robot's ability to learn and adapt online, the recently proposed belief-space safety filter (BeliefSF) reasons about robot safety in closed-loop with runtime inference that actively reduces the robot's uncertainty online, thereby reducing conservativeness in filtering. However, providing formal safety guarantees for robots deploying BeliefSF remains a significant challenge due to errors in runtime inference and neural approximation of safety filters required to handle the high dimensionality of belief spaces. In this paper, we propose an algorithmic approach to certify high-probability safety of BeliefSF using conformal prediction, while explicitly accounting for the reliability of the robot's runtime inference module. Our method leverages the structure of belief-space safety filtering by focusing verification on a region where inference is expected to be reliable. It preserves the simplicity and sample complexity of standard conformal prediction, yet can certify a substantially less conservative safety filter. Through a simulated human-vehicle interaction benchmark, we show that our approach verifies a significantly more permissive belief-space safety filter than a standard conformal prediction baseline.
>
---
#### [new 028] World Models for Robotic Manipulation: A Survey
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决如何通过世界模型预测未来以提升操作能力。工作包括分类模型、分析预测与行动关系，并提出评估标准。**

- **链接: [https://arxiv.org/pdf/2606.00113](https://arxiv.org/pdf/2606.00113)**

> **作者:** Fangyuan Wang; Ziyuan Wang; Guorui Pei; Mengshi Zhang; Canxi Liang; Jun Hu; Zhongxuan Li; Jinsong Wu; Ning Han; Zeqing Zhang; Jiaming Qi; Hongmin Wu; Shiyao Zhang; Pai Zheng; Jia Pan; David Navarro-Alarcon; Sichao Liu; Peng Zhou
>
> **摘要:** Robotic manipulation depends on the ability to anticipate how actions reshape objects, contacts, and scene geometry before execution. Learned world models provide this capability by predicting task-relevant future evolution under robot intervention, yet the term now spans latent dynamics models, action-conditioned video generators, three- and four-dimensional scene predictors, physics-informed simulators, and predictive modules inside vision-language-action systems. This breadth has fragmented the literature and obscured the design choices that matter for manipulation. We survey world models for robotic manipulation through three questions: what future representation is predicted, how prediction is connected to action, and when prediction is used in the robot-learning pipeline. We operationally define a world model as an action-conditioned predictive system and distinguish it from perception modules, inverse models, policies, rewards, and value functions. We then organize existing work into five representation families, develop a functional taxonomy that separates integrated prediction-action models from explicit predictive planners, and characterize infrastructure roles including synthetic experience generation, candidate filtering, search-based evaluation, learned environments, and outcome verification. We further map these roles across pretraining, post-training, and inference adaptation, review 34 manipulation datasets, and synthesize evaluation protocols for predictive fidelity, task performance, and simulator reliability. This survey shows that world models are evolving from task-specific dynamics predictors into predictive infrastructure for robot learning, while exposing open challenges in contact modeling, hallucination control, action alignment, and benchmarking under closed-loop use.
>
---
#### [new 029] Threading Optimization for Vision-Language-Action Model Inference in Low-Cost Smart Agricultural Manipulation
- **分类: cs.RO**

- **简介: 该论文属于智能农业操控任务，解决VLA模型推理速度慢和运动控制不精细的问题。通过优化线程实现，提升系统响应速度与稳定性。**

- **链接: [https://arxiv.org/pdf/2606.00966](https://arxiv.org/pdf/2606.00966)**

> **作者:** Keith Truongcao; Christopher Nhu; Zijian An; Phong Nguyen; Siwei Cai; Lifeng Zhou
>
> **摘要:** Vision-Language Action (VLA) models continue to face challenges such as slow inference speed and difficulty performing fine-grained motion adjustments, limiting their widespread adoption in industry. While the Real-Time Action Chunking (RTAC) algorithm has been proposed to address these bottlenecks, bridging the gap between the algorithm provided in pseudocode to a stable, real-world deployment on a low-cost robotic arm remains a challenge. In this work, we present a complete system-level implementation of RTAC tailored for a low-cost robotic manipulation system. We advance beyond the original high-level pseudocode by optimizing the threading implementation for the policy inference and control pipeline, reducing end-to-end latency and improving responsiveness without modifying the underlying policy. We evaluate this system on tasks involving the manipulation of agricultural produce, specifically garlic bulbs and walnuts. Experimental results demonstrate that our custom threading implementation significantly improves control stability and speed compared to the base implementation of RTAC.
>
---
#### [new 030] Invascal: Inverse-Vacuity Self-Calibration for Uncertainty-Aware LiDAR Range-View Semantic Segmentation
- **分类: cs.RO; eess.IV**

- **简介: 该论文属于LiDAR语义分割任务，解决预测不确定性估计问题。提出Invascal方法，通过分解预测提升不确定性校准，实现准确且实时的分割。**

- **链接: [https://arxiv.org/pdf/2606.00069](https://arxiv.org/pdf/2606.00069)**

> **作者:** Kerim Turacan; Hannes Reichert; Andrei Bolandut; Konrad Doll
>
> **备注:** Accepted for publication at the 2026 IEEE 29th International Conference on Intelligent Transportation Systems (ITSC)
>
> **摘要:** LiDAR semantic segmentation is a core perception capability for autonomous vehicles and mobile robots. However, safe operation also depends on knowing when predictions are unreliable. Existing approaches typically rely on softmax confidence, which is often miscalibrated and overconfident, while stronger uncertainty estimates from Monte Carlo dropout or ensembles are often computationally expensive for real-time use. To this end, we introduce a novel, architecture-agnostic uncertainty-aware Adapter Head. It decomposes the prediction into a Preference Head for class ranking and a Strength Head that refines uncertainty assessment, thereby enabling a principled construction of evidential Dirichlet representations. Building on this design, we propose our inverse-vacuity self-calibration objective (Invascal), which directly supervises the strength signal to produce reliable and well-calibrated uncertainty estimates while preventing runaway evidence growth. We evaluate our framework across multiple LiDAR datasets and backbone architectures. We compare against deterministic training, Monte Carlo dropout and ensembles, and prior evidential methods. Our approach consistently improves uncertainty calibration over traditional deterministic methods with minimal computational overhead. At the same time, it preserves competitive segmentation accuracy, where prior evidential methods often suffer performance degradation.
>
---
#### [new 031] OneVLA: A Unified Framework for Embodied Tasks
- **分类: cs.RO**

- **简介: 该论文提出OneVLA，解决机器人导航与操作任务分离的问题，通过统一框架实现两者协同，提升通用性。**

- **链接: [https://arxiv.org/pdf/2606.01241](https://arxiv.org/pdf/2606.01241)**

> **作者:** Lingfeng Zhang; Xiaoshuai Hao; Yingbo Tang; Lei Zhou; Shuyi Zhang; Jinkun Liu; Hongsheng Li; Chenhao Zhang; Qiang Zhang; Hangjun Ye; Xiaojun Liang; Long Chen; Wenbo Ding
>
> **摘要:** Navigation and manipulation are fundamental capabilities of embodied intelligence, enabling robots to interpret natural language commands and interact physically with their surroundings. However, current Vision-Language-Action (VLA) models remain constrained by task-specific architectures, specializing in either navigation or manipulation, which hinders the development of general-purpose robotic agents. To bridge this gap, we introduce OneVLA, a unified architecture that integrates these distinct tasks into a single, cohesive framework. Specifically, we design a unified action head capable of generating both navigation and manipulation actions without requiring task-specific variants. Furthermore, we propose a multi stage progressive training strategy-incorporating curated data construction and Chain-of-Thought (CoT) fine-tuning that facilitates strong positive transfer and mutual reinforcement between the two domains. Extensive experiments in both simulated and real-world environments demonstrate that OneVLA achieves state-of-the-art performance, significantly outperforming both specialized single-task and existing cross-task models. By unifying these core capabilities, OneVLA paves the way for truly general-purpose robotic systems. The model and source code will be publicly released.
>
---
#### [new 032] GraspGen-X: Cross-Embodiment 6-DOF Diffusion-based Grasping
- **分类: cs.RO**

- **简介: 该论文属于6-DOF抓取任务，解决跨机械臂形态的抓取泛化问题。通过扩展扩散模型，引入机械臂表示，提升模型对新物体和新夹具的零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.00998](https://arxiv.org/pdf/2606.00998)**

> **作者:** Beining Han; Yu-Wei Chao; Erwin Coumans; Clemens Eppner; Balakumar Sundaralingam; Jia Deng; Stan Birchfield; Adithyavairavan Murali
>
> **摘要:** We study cross-embodiment 6-DOF robot grasping. Unlike prior works, we require the model not only to generalize to novel objects / scenes but also to novel gripper morphologies and physical grasping processes. Our method extends diffusion model based generative 6-DOF grasping models to condition on the additional gripper's representation. We propose a swept-volume heuristic for encoding the gripper. We train our cross-embodiment model with procedural grippers and a large-scale dataset of 2 Billion grasps. In simulation experiments, our model has the best zero-shot generalization to novel real-world grippers and objects over baseline methods. Our model also serves as a good initialization for fine-tuning to adapt to novel grippers. In ablations, we demonstrate the efficiency of our sweep-volume gripper representation and our procedural gripper training dataset. Last, we show zero-shot generalization to real-world novel grippers for 6-DOF grasping, surpassing baselines in cross-embodiment generalization.
>
---
#### [new 033] PEACE: A Planner-Executor Agent with Constraint Enforcement for UAVs
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PEACE系统，用于PX4无人机的规划与执行，解决高延迟和约束难以控制的问题。通过解耦任务规划与低层控制，提升可解释性和安全性。**

- **链接: [https://arxiv.org/pdf/2606.00104](https://arxiv.org/pdf/2606.00104)**

> **作者:** Erdem Uysal; Timo Kehrer; Sebastiano Panichella
>
> **备注:** Accepted to ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy: From Environment Understanding and Reasoning to Safe Interaction
>
> **摘要:** Foundation models are increasingly used to drive autonomous systems, yet existing approaches either keep the model in a tight control loop, raising latency and hallucination risk, or compile natural language into opaque end-to-end policies that are hard to explain, constraint and require domain-specific datasets and fine-tuning. We propose a planner-executor agent for PX4-based drones that decouples high-level mission planning from low-level control. A large language model performs single-pass task planning, while execution is handled through a structured ROS 2 tool-calling interface bridged to MAVLink. The system constructs a world model by combining modular 2D detectors (e.g., YOLO or vision-language models) with a pinhole depth projection module for 3D object localization. A constraint enforcement layer enforces altitude limits and horizontal geofencing, and bounded replanning enables recovery from execution-time action failures. We position our approach within three common design patterns for foundation-model-based robotics systems and demonstrate its feasibility in PX4 software-in-the-loop simulations in Gazebo. Results highlight improved explainability, constraint enforcement, and reduced LLM calls compared to tightly coupled LLM control. The code, dataset, videos, and other material can be found at the following link: this https URL
>
---
#### [new 034] WALL-WM: Carving World Action Modeling at the Event Joints
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出WALL-WM，属于视频动作建模任务，解决传统方法在时间粒度上的不匹配问题，通过事件驱动的预训练实现更高效的视觉-语言-动作学习。**

- **链接: [https://arxiv.org/pdf/2606.01955](https://arxiv.org/pdf/2606.01955)**

> **作者:** Shalfun Li; Victor Yao; Charles Yang; Truth Qu; Regis Cheng; Ryan Yu; Howard Lu; Newton Von; Vincent Chen; Yohann Tang; Maeve Zhang; Ellie Ma; Gody Li; Sage Yang; Lorien Shu; J.W. Gao; Ethan Chen; Colin Ye; Yu Sun; Elise Mon; PS Zhang; Neo Li; Lily Li; James Wang; Ping Yang; Chris Pan; Lucy Liang; Hang Su; Roy Gan; Hao Wang; Qian Wang
>
> **摘要:** WALL-WM is a World Action Model that shifts video-action learning from chunk-centric optimization to event-grounded Vision-Language-Action pretraining, using semantically coherent action events as the atomic unit of learning. Existing WAMs commonly initialize from multimodal or video foundation models and then optimize fixed-length action chunks conditioned directly on the current observation and instruction. Although convenient, this chunk-centric formulation creates a fundamental granularity mismatch. Language describes semantic goals and events, vision evolves through continuous scene dynamics, and actions operate at control-level timescales; forcing all three into the same fixed-length prediction window turns VLA training into short-horizon correlation fitting. WALL-WM addresses this mismatch by organizing both supervision and data around semantic events. Specifically, it pairs event-grounded VLA pretraining with a data ecosystem built from event-level captions and cluster-balanced sampling, enabling scalable learning over diverse behaviors, scenes, and task structures. From the same event-pretrained backbone, WALL-WM supports two complementary inference modes. The event mode consumes next-event descriptions and enables variable-length execution chunks, while the unified mode uses a VLM with Staircase Decoding to condition conventional fixed-length chunk inference while preserving a gradient-continuous VLA path. Together with Muon-optimizer-based large-scale pretraining infrastructure, WALL-WM provides a practical scale-up recipe for general-purpose WAMs. Experiments show that WALL-WM generalizes broadly across language, scenes, and tasks, achieving state-of-the-art performance in large-scale real-world generalization evaluation.
>
---
#### [new 035] Beyond Task Success: Behavioral and Representational Diagnostics for WAM and VLA
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人操控中的视觉-语言-动作策略，探讨未来预测对行为和表征的影响。旨在解决WAM与VLA在行为和表征上的差异问题，通过诊断框架分析其性能与结构。**

- **链接: [https://arxiv.org/pdf/2606.01095](https://arxiv.org/pdf/2606.01095)**

> **作者:** Hung Mai; Bin Zhu; Tuan Do
>
> **摘要:** Vision-language-action (VLA) policies and World-Action Models (WAM) represent two increasingly important paradigms for robotic manipulation. However, it remains unclear whether future prediction in WAMs leads to behaviorally meaningful improvements beyond final task success. In this paper, we ask whether WAMs merely add future prediction, or whether they change robot behavior and internal representations in ways that are actionable for control. We introduce a model-agnostic diagnostic framework that compares WAMs and VLAs through two complementary lenses: behavioral rollout analysis and sparse-autoencoder-based feature analysis. The behavioral protocol measures action dynamics consistency, target-object progress, distractor disturbance, and runtime cost. The feature-space protocol characterizes internal representations as memorized, reactive, or predictive, revealing whether models encode future-oriented structure. Across LIBERO and RoboTwin2.0, we evaluate 7 policies spanning direct VLAs and joint, sequential, and auxiliary WAMs. Our results show that success alone hides key differences: WAMs often improve object-level behavior and target selectivity, but their gains depend on architecture and incur higher inference cost. Sequential WAMs show the clearest predictive structure, while auxiliary and joint WAMs respectively compress or entangle future information. These findings suggest future directions for WAMs design to preserve behaviorally actionable future representations for efficient manipulation.
>
---
#### [new 036] RoboDream: Compositional World Models for Scalable Robot Data Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决真实数据收集成本高、效率低的问题。通过构建可扩展的虚拟环境生成模型，实现高质量模拟数据生成，提升下游策略性能。**

- **链接: [https://arxiv.org/pdf/2606.02577](https://arxiv.org/pdf/2606.02577)**

> **作者:** Junjie Ye; Rong Xue; Basile Van Hoorick; Runhao Li; Harshitha Rajaprakash; Pavel Tokmakov; Muhammad Zubair Irshad; Vitor Guizilini; Yue Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** Scaling robot learning requires large-scale, diverse demonstrations, yet real-world data collection via teleoperation remains prohibitively expensive and time-consuming. While video diffusion models offer a promising avenue for data scaling, existing generative approaches are often limited to superficial visual augmentation, or suffer from embodiment hallucinations that yield physically infeasible motions. We present a generalizable embodiment-centric world model that achieves scalable data generation by synthesizing photorealistic demonstrations with novel objects, in novel scenes, and from novel viewpoints. Our approach anchors generation to rendered robot motion while conditioning on explicit scene and object priors, effectively decoupling trajectory execution from environment synthesis. This formulation has the potential to unlock two powerful data scaling capabilities: (1) retrieval and rebirth, which repurposes existing trajectories into entirely new contexts without new motion data; and (2) prop-free teleoperation, where operators manipulate empty air and the model hallucinates the target objects and scene afterwards, eliminating reset time. We demonstrate with real-world experiments that our generated data consistently improves downstream policy performance and significantly reduces real-world data requirements across diverse manipulation tasks.
>
---
#### [new 037] Per-Group Error, Not Total MSE: Fine-Tuning Vision-Language-Action Models for 11-DoF Mobile Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究视觉-语言-动作模型在11自由度移动机械臂上的微调问题，指出总均方误差不是最佳选择，提出按组分析误差更可靠。**

- **链接: [https://arxiv.org/pdf/2606.00253](https://arxiv.org/pdf/2606.00253)**

> **作者:** Pau Montagut Bofi; Mario García Blasco; Tessa Pulli; Markus Vincze
>
> **备注:** 4 pages, 3 figures, 3 tables. Accepted as poster at ICRA 2026 Workshop "From Data to Decisions: VLA Pipelines for Real Robots". Code: [this https URL](this https URL)
>
> **摘要:** Fine-tuning Vision-Language-Action (VLA) models for mobile manipulators with heterogeneous joint spaces can produce a counterintuitive result: the checkpoint with the lowest aggregate MSE is not the one that performs best on the real robot. We argue this is a predictable consequence of collapsing heterogeneous joint groups (arm, gripper, head, wheeled base) into a single metric, where easy-to-predict joints can mask joints that still fail. We fine-tune SmolVLA (450M, action-expert only) on the 11-DoF Toyota HSR and compare it against $\pi_{0.5}$ (3.3B), a stronger pretrained baseline. Per-group analysis exposes two patterns: in SmolVLA, the mobile base converges slowest and limits overall performance. In expert-only fine-tuning of $\pi_{0.5}$ (training only the action head, backbone frozen), total MSE drops below the baseline but arm accuracy degrades. On 60 real-robot trials (20 per model), $\pi_{0.5}$ 80k (4.0/4) significantly outperforms both fine-tuned variants (expert-only 3k: 3.75/4; HSR-SmolVLA: 3.5/4; Mann-Whitney $p \leq 0.010$), despite expert-only 3k having the lowest total MSE. This separation is most consistent with the offline arm-group error, not total MSE or base-group error. We conclude that per-group error is a more reliable signal than total MSE for checkpoint selection on robots with heterogeneous action spaces. Code: this https URL
>
---
#### [new 038] Embedding Semantic Risk into Distance Fields and CBFs for Online Monocular Safe Control
- **分类: cs.RO**

- **简介: 该论文属于安全控制任务，解决单目视觉下障碍物风险感知不足的问题。通过将语义风险嵌入距离场，提升CBF控制的安全性与效率。**

- **链接: [https://arxiv.org/pdf/2606.01605](https://arxiv.org/pdf/2606.01605)**

> **作者:** Dawei Zhang; Nuo Chen; Shuo Liu; Roberto Tron; Zhiwen Fan
>
> **摘要:** We propose an online monocular perception-to-control framework that embeds semantic risk into the distance field used by Control Barrier Function (CBF)-based safe navigation and teleoperation. Many perception-based safety filters assign the same distance-based safety margin to all mapped obstacles or use semantics only as a downstream controller adjustment, rather than encoding semantic risk in the spatial representation. Our framework instead reasons online about obstacle geometry and class-dependent risk by embedding semantic information directly into the Euclidean Signed Distance Field (ESDF). This design encodes semantic risk before control optimization, so high-risk objects exert a larger spatial influence in the safety field while retaining efficient ESDF queries at runtime. Specifically, a foundation-model-based SLAM front end reconstructs dense 3-D geometry from monocular RGB video, while per-frame semantic segmentation provides pixel-level class labels that are fused into the reconstructed geometry. The resulting geometric-semantic representation is then converted into an ESDF, where semantic labels identify safety-relevant regions and impose class-dependent inflation before field computation. The semantic-aware ESDF provides the local distance values and spatial derivatives required by the CBF controller, while class-dependent gains further regulate the controller response. Extensive simulation and hardware experiments demonstrate online operation at 10--20 Hz and semantic-aware safe behavior in both teleoperation and autonomous navigation.
>
---
#### [new 039] RocketSmith: An Agentic System for High-Powered Rocket Design and Manufacturing
- **分类: cs.RO**

- **简介: 论文介绍RocketSmith系统，用于高能火箭设计与制造。该任务旨在实现智能自动化设计与优化，解决传统流程效率低的问题。工作包括开发系统、优化参数并成功测试火箭。**

- **链接: [https://arxiv.org/pdf/2606.00097](https://arxiv.org/pdf/2606.00097)**

> **作者:** Peter Pak; Jesse Barkley; Rumi Loghmani; Derek Baich; Ananya Pamal; Amir Barati Farimani
>
> **摘要:** This work presents RocketSmith, an agentic system capable of the design, manufacturing, and optimization processes in high powered rocket development. The system enables the intelligent automation of software tools as to not only validate factors such as flight stability but also generate the parametric design components for the rocket assembly. A collection of subagents and skills enable optimization workflows of flight parameters via iteration in both zero-shot and human-in-the-loop workflows. With this system, four distinct high power rockets with various motor and assembly configurations were developed utilizing the unique design capabilities of additive manufacturing. These assembly components were fabricated using various FDM printers, manually evaluated for flight readiness, and flight tested at a launch event. From these tests, all rockets achieved a stable launched and two of the four rockets were successfully recovered in reflyable condition. Within the collected flight data, an 84% accuracy was achieved when comparing measured apogee to that calculated in flight simulations.
>
---
#### [new 040] Market-Based Replanning for Safety-Critical UAV Swarms in Search and Rescue Missions
- **分类: cs.RO; cs.MA; eess.SY**

- **简介: 该论文属于无人机编队任务，解决SAR中故障容错问题。提出IRDS框架，通过拍卖机制和几何共识实现自主重规划，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.01970](https://arxiv.org/pdf/2606.01970)**

> **作者:** Luiz Giacomossi; Andrea Haglund; Claire Namatovu; Emily Zainali; Esaias Målqvist; Yonatan M. Beyene; Ivan Tomasic; Baran Çürüklü; Håkan Forsberg
>
> **备注:** 6 pages, 4 figures, accepted at MIPRO 2026
>
> **摘要:** Reliable autonomous UAV swarms in Search and Rescue (SAR) missions require fault-tolerant coordination capable of sustaining operations despite agent degradation. This paper introduces the Intelligent Replanning Drone Swarm (IRDS), a distributed coordination architecture designed for resource-constrained environments. The proposed framework employs a Reverse-Auction market mechanism where agents bid to service search sectors based on a distance-weighted cost function, coupled with a geometric consensus protocol for target verification. We evaluate the approach through physics-based simulations (N=8 agents, 8x8 grid) subjected to stochastic fault injection. Results indicate that the swarm autonomously reallocates tasks from failed agents with low latency relative to the total mission duration, maintaining a mission success rate of 93% under 25% workforce degradation. The proposed framework demonstrates a robust, empirically tested method for self-healing aerial robotic coordination.
>
---
#### [new 041] Autopilot-Preserving Residual Q-Learning with HJB-Inspired Finite-Action Risk Filtering for Fixed-Wing UAV Command Supervision
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于无人机控制任务，解决固定翼无人机在复杂天气下的路径跟踪问题。通过引入HJB启发的残差Q学习方法，提升控制精度并降低误差。**

- **链接: [https://arxiv.org/pdf/2606.01397](https://arxiv.org/pdf/2606.01397)**

> **作者:** Mehmet Iscan; Batuhan Temiz
>
> **备注:** 47 pages, 12 figures, 20 tables. Simulation-based study with a code-traceable benchmark, source code and a demonstration video are linked in the paper
>
> **摘要:** A fixed-wing UAV must hold airspeed, altitude, and heading references under wind, gusts, and turbulence, channels coupled so that correcting one can degrade another. Classical autopilots stabilize the airframe well but adapt poorly when a hard crosswind meets an aggressive turn, while reinforcement-learning (RL) policies acting directly on the surfaces concentrate exploration risk at the actuator interface. We place a learned supervisor above an unchanged autopilot rather than inside it: it selects a residual from a finite, bounded action set on the commanded airspeed, altitude, and heading; the modified reference is projected into an admissible command envelope before reaching the autopilot, which stays the only actuator-facing controller. What is new is how the residual is chosen. HJB residual scores candidates with a semi-discrete value-iteration critic in the spirit of the Hamilton-Jacobi-Bellman (HJB) equation, ranks them by a no-op-relative Hamiltonian advantage, and filters them through a control-Lyapunov- and control-barrier-inspired finite-action shield that always keeps a no-op fallback. On a shared 12-state runtime holding the plant, autopilot, and actuator model fixed, so the comparison is at the package level, HJB residual lowers mean RMS path-tracking error to 44.809 m, against 338.617 m for the baseline autopilot and 88.809 m for a tabular-Q residual, an 86.77% reduction over the baseline and 49.54% over Q-learning. The gain concentrates where the baseline fails worst and comes with a measured rise in airspeed error, so no method dominates every metric. We present this autopilot-preserving residual command-supervision design and benchmark with its trade-offs reported intact.
>
---
#### [new 042] Can Predicted Dynamics Exist in the Physical World?
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究预测性物理AI系统的物理可行性问题，旨在判断预测动态是否可在现实世界中执行。通过构建评估框架，提升预测的物理可执行性。**

- **链接: [https://arxiv.org/pdf/2606.00089](https://arxiv.org/pdf/2606.00089)**

> **作者:** Barak Or
>
> **备注:** 17 pages
>
> **摘要:** Predictive Physical AI systems output state rollouts, action chunks, and latent plans, yet a low root-mean-square error (RMSE) does not imply that a particular proposal is physically executable. We formulate physical admissibility as a prediction-control interface: before execution, a decoded proposal is treated as candidate dynamics and evaluated using kinematic, dynamic, and direct-to-composed horizon conditions. Passing is not a certificate of task success; rejection identifies violation of the specified physical envelope and gives a component-level reason. On Hugging Face LeRobot PushT, controlled falsification shows that one-step prediction-RMSE and standardized dynamics residuals reach area under the receiver operating characteristic curve (AUC) 0.982 and 0.972, kinematic-only conditions reach AUC 0.592, and the full gate reaches AUC 0.957 with condition-level attribution. In replay-based intervention experiments, residual-based filters and the full physical-admissibility gate prevent 87-$89% of invalid proposals while preserving mean progress near 0.998.
>
---
#### [new 043] ScaRF-SLAM: Scale-Consistent Reconstruction with Feed-Forward Models and Classical Visual SLAM
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决几何模型预测不准确影响定位的问题。通过将传统SLAM与几何基础模型解耦，提升重建精度与一致性。**

- **链接: [https://arxiv.org/pdf/2606.00307](https://arxiv.org/pdf/2606.00307)**

> **作者:** Yuhao Zhang; Yifu Tao; Frank Dellaert; Maurice Fallon
>
> **备注:** 8 pages
>
> **摘要:** Recent works have explored unifying SLAM with geometric foundation models (GFMs). However, directly using GFM predictions for tracking is highly sensitive to model capability and uncertainty, as geometric inaccuracies in the predictions can adversely affect pose estimation. To address this limitation, we present a decoupled framework that integrates classical feature-based SLAM with GFMs, which achieves higher quality and more consistent dense reconstruction. In brief, we use classical visual SLAM for robust low-latency tracking and use GFMs exclusively for mapping. By anchoring mapping to poses produced by the SLAM module and optimizing across depth scales, the proposed design avoids propagating inaccuracies from GFM predictions into pose estimation while imposing geometric constraints on the reconstruction. The system builds submaps from multiple posed keyframes and enforces scale consistency via lightweight frame and submap scale optimization. It also performs projection-based point cloud fusion within each submap, and updates submaps online to reflect trajectory updates from the feature-based SLAM. To evaluate tracking and reconstruction of our method, we introduce a loop-rich, building-scale indoor dataset with accurate sensor trajectories and LiDAR ground-truth. Experiments show that our approach achieves superior trajectory accuracy while improving reconstruction precision by 10%-20% over existing methods, with about 2 cm reconstruction error per 10 m chunk on building-scale dataset. On large-scale outdoor datasets, it attains 10 cm error per 30 m chunk (w.r.t LiDAR ground-truth models).
>
---
#### [new 044] OSCAR: Obstacle Survival Curves for Adaptive Robot Navigation
- **分类: cs.RO**

- **简介: 该论文提出OSCAR框架，用于解决机器人在动态障碍物下的自适应导航问题。通过学习障碍物清除时间分布，优化等待与绕行决策，提升导航效率。**

- **链接: [https://arxiv.org/pdf/2606.00990](https://arxiv.org/pdf/2606.00990)**

> **作者:** Hshmat Sahak; Aoran Jiao; Nicholas Rhinehart; Tim Barfoot
>
> **备注:** 8 pages main text, appendices included
>
> **摘要:** A mobile robot following a graph of known routes can make costly navigation errors when a temporary obstacle blocks a critical edge: waiting too long behind a parked cart wastes time, but immediately rerouting around a person who would move in a few seconds is also inefficient. Standard reactive obstacle avoidance addresses local motion around obstacles, while fixed wait-or-reroute rules ignore how long different obstacle types tend to persist. We propose OSCAR: an adaptive survival-modeling framework for graph-based navigation with temporary blockages. Assuming obstacle class labels are available at encounter time, the robot learns class-conditioned residual clearance-time distributions from online experience, including right-censored observations when it reroutes before observing clearance. These survival models are integrated into a time-dependent graph planner that maintains obstacle memory and computes a patience threshold at each blocked edge: how long to wait before taking an alternate route. The method continuously updates its clearance estimates across episodes and uses them to balance waiting against rerouting. We evaluate the approach in simulation and on a real mobile robot in a university atrium with obstacles including people, chairs, bins, and tubes. In simulation, the learned policy's time-to-goal converges to within 1% of an oracle with access to ground-truth clearance distributions after fewer than 20 observations per obstacle class, outperforming all heuristic baselines. Real-world deployment confirms that the policy improves online, adapting its patience thresholds from experience across 50 navigation episodes.
>
---
#### [new 045] Series-Parallel Integrated Nonlinear Elastic Actuator applied to the lean motion of a bicycle simulator
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉交互领域，旨在解决高扭矩与高精度力控制的难题。提出SPINEA结构，融合SEA与PEA优势，实现单弹性元件双功能，用于自行车模拟器的倾斜驱动。**

- **链接: [https://arxiv.org/pdf/2606.00201](https://arxiv.org/pdf/2606.00201)**

> **作者:** Christina Kohler; Michiel Plooij; Nuria Peña-Perez; Arend L. Schwab; Heike Vallery
>
> **摘要:** Designing robots for high-torque, high-fidelity haptic interaction is challenging. Parallel Elastic Actuators (PEAs) use elastic elements in parallel to smaller motors to complement torques, and Series Elastic Actuators (SEAs) use elastic elements in series to decouple motor impedance and improve force control. Recent work combines SEAs and PEAs to obtain both benefits but requires separate elastic elements or clutching. This paper presents the Series Parallel Integrated Nonlinear Elastic Actuator (SPINEA), which merges SEA and PEA such that a single elastic element takes on dual roles simultaneously, parallel and series. This is achieved by a nonlinear transmission in which the motor and load have misaligned rotation axes and are elastically connected. This geometry enables both high peak torque and precise torque tracking. We apply SPINEA to actuate lean of a haptic bicycle simulator, which requires high moments and precise rendering for safe and realistic rider interactions. We realized a prototype and performed experiments, both with an external excitation setup and with riders cycling. Our results confirm SPINEA's low impedance and precise torque tracking, up to 4.25 Hz with the bicycle frame fixed and up to 4 Hz with riders. The benefits may transfer to other applications requiring compact, high-performance actuation.
>
---
#### [new 046] IMAC-AgriVLN: Can Agricultural Vision-and-Language Navigation Agents be Aware of Instruction Mistakes?
- **分类: cs.RO**

- **简介: 该论文属于农业视觉语言导航任务，解决指令错误识别问题。提出A2A-MI基准和IMAC模块，提升导航代理对指令错误的感知与纠正能力。**

- **链接: [https://arxiv.org/pdf/2606.02519](https://arxiv.org/pdf/2606.02519)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xin Chen; Xiang Li
>
> **摘要:** Agricultural robots are serving as powerful assistants across a wide range of agricultural tasks, nevertheless, still heavily relying on manual operations or railway systems for movement. The AgriVLN method and the A2A benchmark pioneeringly extended Vision-and-Language Navigation (VLN) to the agricultural domain, enabling a robot to navigate to a target position following a natural language instruction. However, almost all the prior methods adopt an ideal assumption that the given instructions themselves are correct, which does not align with the realistic scenarios, because anybody may say an instruction with mistakes. To bridge this gap, we propose the A2A-MI benchmark, in which we build a semi-automatic data annotator to insert three mistake classifications into each original instruction in a more diversified and efficient way. We test several state-of-the-art agricultural VLN agents on it and observe a sufficient drop with -57% on SR and -9% on NE, from which we suggest that an agricultural VLN agent tends to assume that the given instruction is correct, so does not have the awareness to doubt it when the scenes it sees do not align with the instruction it receives. To build the awareness on instruction mistake, we propose the IMAC module analyzing the instruction and the current front-facing image, to judge whether the instruction has mistakes and attempt to correct it when needed. We integrate IMAC into the baseline model, and observe a noteworthy improvement, sufficiently narrowing the gap to the performance on instructions without mistakes. Project: this https URL.
>
---
#### [new 047] LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World
- **分类: cs.RO**

- **简介: 该论文提出LEGS系统，解决人形机器人操作中数据收集成本高的问题。通过合成数据提升视觉-语言-动作策略的泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.01458](https://arxiv.org/pdf/2606.01458)**

> **作者:** Hojune Kim; Timothy Chen; Jiankai Sun; Lars W. Osterberg; Qianzhong Chen; Ke Wang; Mac Schwager
>
> **备注:** this https URL
>
> **摘要:** Training vision-language-action (VLA) policies for humanoid loco-manipulation is constrained by the high cost and complexity of collecting human teleoperation demonstrations. VLA policies fine-tuned in simulators have, until now, failed to transfer effectively in humanoid loco-manipulation tasks. We present LEGS (Loco-manipulation via Embodied Gaussian Splatting), a hybrid simulator that composites a mesh foreground (robot, objects, props) over a photorealistic 3D Gaussian Splatting (3DGS) background reconstructed from a handheld scene capture. LEGS uses a procedural motion-primitive generator to synthesize labeled demonstrations at scale without human teleoperation, and a deterministic two-stage color calibration to align the rendered 3DGS image to the robot's deployment camera. On a Unitree G1 humanoid robot, across three pick-and-place tasks of increasing whole-body difficulty and three VLA backbones (psi_0, pi_0.5, GR00T N1.6), a policy trained purely on LEGS data matches or exceeds one trained on human teleoperation demos on every experiment. It also outperforms a mesh-only simulation baseline that ablates the effect of the 3DGS background, showing that photorealistic rendering is a key enabler for synthetic data transfer. Humanoid motion is recorded independently of scene appearance in LEGS, allowing the same auto-generated demonstrations to be re-rendered under new backgrounds and object meshes--covering a new scene at more than 15x lower cost than teleoperation--to augment training data for robustness to scene variations. Under combined object-and-scene appearance shift, the policy trained on re-rendered LEGS-AUG data maintains task success while the baseline trained on teleoperation data fails entirely. Our project page is located at this https URL.
>
---
#### [new 048] FlipItRight: Stable Pose-Targeted Throw-Flip Across Diverse Objects
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文提出FlipItRight，解决高自由度机械臂稳定投掷翻转任务。通过分解为物体级和机器人级规划，实现不同物体的精准投掷，无需先验数据，成功率90%。**

- **链接: [https://arxiv.org/pdf/2606.01713](https://arxiv.org/pdf/2606.01713)**

> **作者:** Axel Dawne; Shinkyu Park
>
> **摘要:** We propose FlipItRight, a framework for stable planar pose-targeted throw-flip with a high-DoF manipulator. The task is decomposed into an object-level planner, which generates candidate release states satisfying the desired landing pose, and a robot-level planner, which evaluates executability and constructs a feasible swing motion. Treating the release state as an explicit intermediate representation enables principled candidate filtering, adaptive selection of release and pre-swing configurations, and structured near-release motion design -- in particular, approximately constant end-effector velocities during the final swing phase to improve robustness to release-timing uncertainty. We validate on a real platform across objects of varying shape, size, and mass, achieving a 90% success rate across 120 trials. Ablation studies confirm that each design choice contributes to throwing performance, and the framework requires no prior data or learned model, enabling direct deployment on new objects and targets without environment-specific calibration or data collection.
>
---
#### [new 049] Belief Consistency Between Foundation-Model Evidence and Geometric Perception in Persistent Robotic Maps
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人持久地图构建任务，解决几何感知与基础模型语义预测不一致的问题。提出一种更新算子，通过校准门控和冲突丢弃窗口提升地图准确性。**

- **链接: [https://arxiv.org/pdf/2606.00318](https://arxiv.org/pdf/2606.00318)**

> **作者:** Christoffer Heckman; Harel Biggie; Brendan Crowe; Nicholas Roy
>
> **摘要:** Persistent maps used by autonomous robots increasingly fuse a geometric perception stack whose assertions are well-characterized with a foundation-model channel that produces semantic claims without calibrated reliability about the same scene. Contemporary mapping systems integrate the two channels by treating the foundation-model channel as an additional voter into a per-element posterior, uncalibrated for its own per-class reliability and without machinery to flag when the two channels contradict each other at a given moment. We propose an update operator with two cooperating mechanisms: a per-class calibrated commit gate, and a per-event conflict-drop window that refuses to commit foundation-model claims contradicted by the geometric channel at the moment of the claim. We evaluate on KITTI-360 and ScanNet, with an oracle geometric channel (panoptic ground truth) and an off-the-shelf online semantic segmenter (Mask2Former) to demonstrate real-world performance. The operator produces substantially more accurate committed maps (KITTI is car commit precision 99.7% vs. 43.9% for the calibration-only operator; mean per-class IoU 0.522 vs. 0.180), retains more compositional true positives at higher precision than a monolithic compositional VLM prompt. The framework operates at deployment quality across both oracle and off-the-shelf-segmenter geometric channels, and is invariant under foundation-model substitution.
>
---
#### [new 050] Literary Emotions in Motion: A Soft Robotics Installation for Tactile Storytelling
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于情感交互任务，旨在通过软体机器人实现文本情感的触觉表达。工作包括情感分析、软体致动器设计及多感官交互评估。**

- **链接: [https://arxiv.org/pdf/2606.00418](https://arxiv.org/pdf/2606.00418)**

> **作者:** Carolina Silva-Plata; Abraham Villavicencio-Carmona; Miguel Silva Plata; Stefan Escaida; Ruben Fernandez
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Soft robotics is increasingly explored in artistic contexts, where tactile interaction provides audiences with embodied engagement beyond visual or auditory signals. This work presents an interactive installation that maps semantic emotion analysis of narrative text into variable stiffness of soft pneumatic modules. A natural language model identifies two dominant emotions from a predefined set of six, driving the inflation of seven hexagonally arranged soft actuators. The central actuator represents the primary emotion, while the surrounding ones express the secondary. We develop and mechanically characterize silicone actuators, called soft modules, featuring a thin membrane layer, demonstrating how this morphological control expands the achievable stiffness range while preserving simplicity and low-cost fabrication. A user study with ten participants further evaluates how multisensory coupling of stiffness and LEDs intensity influences emotional perception. The results suggest that stiffness modulation accompanied by color change can support emotionally meaningful and engaging tactile interaction in soft robotic installations.
>
---
#### [new 051] Global-Local Attention Decomposition for Terrain Encoding in Humanoid Perceptive Locomotion
- **分类: cs.RO**

- **简介: 该论文属于人形机器人感知运动任务，旨在解决稀疏足点地形下的运动控制问题。提出GLAD方法分离全局与局部注意力，提升地形编码效果。**

- **链接: [https://arxiv.org/pdf/2606.00637](https://arxiv.org/pdf/2606.00637)**

> **作者:** Shengcheng Fu; Yang Zhang; Zhanxiang Cao; Liyun Yan; Yizhi Chen; Yunpeng Yin; Yue Gao
>
> **摘要:** Although reinforcement learning has significantly advanced humanoid locomotion, perceptive policies still struggle on sparse-foothold terrain and constrained environments. Success in these scenarios requires both broad terrain awareness and precise foothold selection, two perceptual roles that conventional encoders often entangle. To address this challenge, we propose Global-Local Attention Decomposition (GLAD) for terrain encoding in humanoid locomotion. Realized by a coarse-to-fine encoder over a robot-centric elevation map, GLAD explicitly separates these objectives: a global attention branch utilizes attention pooling to summarize the surrounding terrain context, while a state-conditioned local attention branch sparsifies and encodes precise foothold-relevant geometry. This explicit attention decomposition prevents the dilution of fine-grained spatial cues while reducing training overhead. Experiments demonstrate that GLAD enables reliable locomotion over challenging gaps, stepping stones, and stairs. Furthermore, the learned policy exhibits emergent terrain-responsive behaviors, autonomously following narrow paths and avoiding obstacles under simple velocity commands without explicit navigation planners. In real-world deployment on a Unitree G1 humanoid robot using onboard LiDAR, the proposed method achieves robust zero-shot sim-to-real transfer across diverse sparse-foothold and obstacle-rich domains.
>
---
#### [new 052] V2I Work Zone Geometry Reconstruction with Pose-Conditioned UWB Range Denoising
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于工作区几何重建任务，解决UWB测距中的NLOS误差等问题，提出一种姿态条件的去噪方法，提升测距精度和几何重建效果。**

- **链接: [https://arxiv.org/pdf/2606.00119](https://arxiv.org/pdf/2606.00119)**

> **作者:** Jiaxi Liu; Hangyu Li; Yang Cheng; Rui Gana; Junwei You; Weizhe Tang; Peng Zhang; Steven T. Parker; Xiaopeng Li; Bin Ran
>
> **摘要:** Reliable work zone mapping is important for connected and autonomous vehicles (CAVs) to navigate safely and smoothly through work zone areas. Cone-mounted ultra-wideband (UWB) roadside units (RSU) offer a cost-effective way for work zone layout inference, as roadside anchors and vehicle tags provide direct vehicle-to-infrastructure (V2I) range constraints for work zone geometry reconstruction. However, UWB range estimation is degraded by bursty outliers, non-line-of-sight (NLOS) errors, arbitrary anchor-ordering issues, and vehicle pose uncertainties in practical field deployments. To address these challenges, this study proposes a pose-conditioned, permutation-equivariant predictive denoiser for multi-anchor UWB ranging. The model employs shared anchor-wise temporal prediction to capture range dynamics, symmetric set aggregation to handle unordered and missing anchors, and pose-conditioned residual decoding to incorporate vehicle motion as a geometric prior. A two-stage training strategy first learns prediction from observed ranges, and then fine-tunes the denoiser with NLOS-weighted supervision. The method is evaluated on rare real-world V2I UWB field data collected with a CAV, as well as on controlled large-scale simulation benchmarks for ablative insights. Results show that the proposed method substantially improves range accuracy, cone localization, and work zone geometry reconstruction in challenging NLOS-dominated regimes, remains robust to anchor re-indexing and moderate anchor dropout, and reduces measurement-weighted field MSE by 66.9% relative to the raw input.
>
---
#### [new 053] ImagineUAV: Aerial Vision-Language Navigation via World-Action Modeling and Kinodynamic Planning
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决UAV在部分可观测环境下执行自由指令的问题。通过生成未来观测和优化轨迹，提升导航鲁棒性与实用性。**

- **链接: [https://arxiv.org/pdf/2606.01205](https://arxiv.org/pdf/2606.01205)**

> **作者:** Xuchen Liu; Jiawei Huang; Shihao Xia; Bingxi Liu; Jinqiang Cui; Jiankun Yang
>
> **备注:** Video demo: this https URL
>
> **摘要:** Vision-language navigation (VLN) for UAVs demands grounding free-form instructions into 6-DoF flight under partial observability. While Vision-Language-Action (VLA) models excel at semantic reasoning, they suffer from brittleness due to geometric inconsistency and dynamics mismatch. To address this, we propose ImagineUAV, an imagination-driven framework leveraging cascaded world-action modeling. Instead of direct regression, ImagineUAV employs a latent video diffusion model to generate instruction-conditioned future observations, explicitly imagining environmental evolution, from which 6-DoF motions are inferred via an action extractor. A kinodynamic planner then refines these estimates into collision-free trajectories. Additionally, a step-distilled inference pipeline ensures real-time execution. With only 1.3B parameters, ImagineUAV outperforms prior VLN and VLA baselines on benchmarks and real-world flights, validating the practicality of imagination-driven aerial navigation.
>
---
#### [new 054] DeepIPCv3: Event-Aware Multi-Modal Sensor Fusion for Sudden Pedestrian Crossing Avoidance
- **分类: cs.RO; cs.AI; cs.CV; eess.IV; eess.SY**

- **简介: 论文提出DeepIPCv3，解决自动驾驶中突发行人横穿的安全问题，融合LiDAR与DVS数据，提升动态场景感知与控制精度。**

- **链接: [https://arxiv.org/pdf/2606.01277](https://arxiv.org/pdf/2606.01277)**

> **作者:** Oskar Natan; Andi Dharmawan; Aufaclav Zatu Kusuma Frisky; Jazi Eko Istiyanto; Jun Miura
>
> **摘要:** Current end-to-end autonomous driving systems predominantly rely on frame-based sensors, which suffer from inherent perception latency and motion blur during highly dynamic encounters, specifically sudden pedestrian crossings. To address this critical safety vulnerability, we propose DeepIPCv3, a novel multi-modal autonomous navigation framework that synergizes the dense 3D spatial geometry of LiDAR point clouds with the microsecond-level asynchronous event streams of a Dynamic Vision Sensor (DVS). We introduce a Transformer-inspired cross-modal attention mechanism to dynamically correlate these distinct modalities, allowing the network to instantaneously prioritize high-speed dynamic updates without sacrificing structural scene awareness. The fused latent representations are then mapped to safe local waypoints and executable control commands via a hybrid policy network that blends heuristic trajectory tracking with direct neural predictions. Due to the severe physical risks associated with live testing of these sudden crossing scenarios, the framework is rigorously evaluated offline using a custom multi-modal dataset collected across both well-illuminated noon and challenging evening conditions. Extensive comparative and ablation studies demonstrate that DeepIPCv3 achieves state-of-the-art predictive performance. By effectively eliminating exposure failures and motion blur, the proposed LiDAR and DVS fusion yields the lowest trajectory and control command errors, enabling highly reactive, mathematically bounded evasive maneuvers regardless of ambient illumination. To support future research, we will release the codes to our GitHub repo at this https URL.
>
---
#### [new 055] Continuous Reasoning for Vision-Language-Action
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出一种用于视觉-语言-动作模型的连续推理机制，解决动作控制与语言推理粒度不匹配的问题，通过共享的高斯潜在接口提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2606.00229](https://arxiv.org/pdf/2606.00229)**

> **作者:** Yueh-Hua Wu; Tatsuya Matsushima; Kei Ota
>
> **备注:** Project page: this https URL
>
> **摘要:** Natural language is a powerful reasoning medium for language and vision-language models, but it is mismatched to the granularity of continuous control. Text and explicit subgoals operate at task-level granularity, whereas vision-language-action (VLA) policies must choose actions at a much finer temporal scale; a single reasoning step can therefore span many action chunks while remaining only weakly coupled to the action needed now. This suggests a different question for VLA: what should play the role of language? We argue that a useful VLA reasoning medium must be shareable across model instances, verifiable through downstream action improvement, and aligned with temporally extended control structure. Based on this view, we propose Continuous Reasoning for Vision-Language-Action. Our model first predicts continuous reasoning in the form of a structured set of continuous thoughts, then reuses them as shared context for chunk-structured action generation. Better action prediction alone does not certify good reasoning: if the same internal medium cannot be shared across model instances and independently verified through improved downstream control, the added latent may simply become a model-private shortcut that helps on seen behaviors without supporting generalizable control. We therefore instantiate continuous reasoning as a shared Gaussian latent interface and train it with a self-verification objective in which an exponential-moving-average teacher must successfully consume the student's reasoning when predicting target actions. Empirically, Continuous Reasoning improves LIBERO-PRO robustness and performs strongly on real robots, raising mean subtask success over {\pi}0.5 by 40.4% on TX-G2, an AgiBot G2-compatible variant, and 26.3% on HSR. This suggests that reasoning in VLA is less about extra tokens than about a shareable, verifiable internal language for action.
>
---
#### [new 056] Balancing Accuracy and Efficiency: Adaptive Dynamics Orchestration for Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决MPC中模型精度与实时性之间的平衡问题。提出ADO框架，动态选择合适模型，提升导航可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2606.00085](https://arxiv.org/pdf/2606.00085)**

> **作者:** Francesco Cancelliere; Aniket Datar; Giovanni Muscato; Xuesu Xiao
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Model Predictive Control (MPC) for autonomous navigation faces a fundamental trade-off between model accuracy and real-time efficiency. High-fidelity dynamics models can accurately predict complex vehicle-terrain interactions during trajectory rollouts, but incur significant computational cost, increasing inference latency and reducing control frequency. Conversely, lightweight models enable fast updates and dense sampling, yet may produce erroneous predictions under safety-critical conditions, potentially leading to catastrophic failures such as vehicle rollover. To address this trade-off, we propose Adaptive Dynamics Orchestration (ADO), a framework that dynamically selects the most appropriate dynamics model for the current navigation context. ADO maintains a library of models spanning diverse accuracy-efficiency profiles and continuously refines terrain-conditioned performance estimates using residual errors from online counterfactual rollouts, where executed control actions are replayed across the model library to assess predictive discrepancy. These estimates guide model selection in real time, balancing computational efficiency and predictive accuracy. Real-world experiments on an off-road ground robot demonstrate that ADO significantly reduces modeling error compared to a fixed low-latency baseline, while approaching the accuracy of the highest-fidelity model without incurring its computational cost, resulting in more reliable and effective navigation in challenging terrain.
>
---
#### [new 057] Safe2Drive: Evaluating Safe Driving Behaviors of E2E Autonomous Driving Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶安全评估任务，旨在解决E2E模型在真实危险场景下的安全行为问题。研究提出Safe2Drive测试集和SDS指标，评估模型在工作区、行人横穿等场景的表现。**

- **链接: [https://arxiv.org/pdf/2606.00191](https://arxiv.org/pdf/2606.00191)**

> **作者:** Nishad Sahu; Kalpana Panda; Congyuan Yu; Changzhong Qian; Shounak Sural; Ragunathan Rajkumar
>
> **摘要:** Recent end-to-end (E2E) autonomous driving policies achieve high driving scores in closed-loop simulations. Yet it remains unclear whether these policies handle common safety-critical scenarios. We present Safe2Drive (S2D), a set of Bench2Drive-aligned scenario extensions focused on three frequent families of road hazards: work zones, pedestrian jaywalking, and occluded vulnerable road users (VRUs). Safe2Drive adds 100 common but challenging scenarios and introduces SafeDriving Score (SDS), a safety-centric metric that augments prior evaluators with pre-crash braking, work zone-object contact, lane centering, and smoothness checks. Evaluating two state-of-the-art policies (LEAD and SimLingo) on S2D, we find that their driving scores drop sharply relative to their reported Bench2Drive baselines (LEAD: from 94.70 DS on Bench2Drive to 39.95 DS on S2D; SimLingo: from 85.07 DS on Bench2Drive to 41.00 DS on S2D) and that SDS on S2D is low (11.85 for LEAD and 15.27 for Sim-Lingo). These results are consistent with brittle safe-driving behaviors such as poor work-zone understanding, red-light violations, and late or absent braking for pedestrians. This study highlights a lack of safe behavioral reasoning in E2E models even when tested on CARLA towns that are part of the training set. We plan to release the code and videos for all 100 S2D scenarios.
>
---
#### [new 058] Robust Integrated Planning and Control for Quadrotors in Dynamic Environments via NMPC with CBF Penalties
- **分类: cs.RO**

- **简介: 该论文属于无人机路径规划与控制任务，解决动态环境中 quadrotor 的安全避障问题。通过 NMPC 结合 CBF 惩罚项和 HGDO、KF 提升系统鲁棒性与实时性。**

- **链接: [https://arxiv.org/pdf/2606.01038](https://arxiv.org/pdf/2606.01038)**

> **作者:** Zeinab Shayan; Mohammadreza Izadi; Reza Faieghi
>
> **备注:** Accepted to Conference on Robots and Vision (CRV 2026), Vancouver, Canada
>
> **摘要:** This paper presents a new robust integrated planning and control (IPC) strategy for multirotor uncrewed aerial vehicles. We propose a nonlinear model predictive control (NMPC) formulation that embeds control barrier functions (CBFs) as exponential penalties, improving feasibility while ensuring smooth obstacle avoidance under tight input bounds. The penalty weights provide a practical tuning knob to trade off tracking accuracy against avoidance aggressiveness. We enhance the system robustness by employing a high-gain disturbance observer (HGDO) to estimate and compensate for external disturbances. We also incorporate a Kalman filter (KF) for computationally efficient, real-time prediction of obstacle motion, enabling avoidance of moving obstacles. Comparative studies against both conventional NMPC and NMPC with hard CBF constraints, validated in Gazebo and hardware experiments, demonstrate superior feasibility, safety, and robustness. To the best of our knowledge, this is the first hardware-validated NMPC-CBF IPC framework, offering a practical step toward safe quadrotor deployment in dynamic environments.
>
---
#### [new 059] Constrained Whole-Body Tracking for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决 humanoid 机器人在强化学习中确保安全与约束满足的问题。提出 ConstrainedMimic 框架，实现实时约束 enforcement，保障运动安全与稳定性。**

- **链接: [https://arxiv.org/pdf/2606.00374](https://arxiv.org/pdf/2606.00374)**

> **作者:** Daniel Morton; Pranit Mohnot; Marco Pavone
>
> **摘要:** Recent advances in reinforcement learning (RL) have demonstrated impressive whole-body agility for humanoid robots, yet ensuring safety and satisfying constraints -- particularly those specified after training -- remains a challenge. Towards this goal, we present ConstrainedMimic, a control framework that leverages whole-body kinematics and dynamics for real-time constraint enforcement within RL tracking policies. By integrating principles from operational space control and control barrier functions (CBFs), we enable the satisfaction of arbitrary runtime constraints on both the kinematic reference motion and the underlying dynamics. In whole-body motion-tracking and teleoperation experiments on a (simulated) Unitree G1 with a learned policy, we demonstrate collision avoidance (both with the robot body and external obstacles), joint limits, and center of mass stability constraints. By remaining consistent with the current contact mode and tracking objectives, we minimally restrict the capabilities of the policy when constraints are active. Our method is fully differentiable, runs on CPU, GPU, and TPU, and can be deployed at up to 300-500 Hz. All software will be freely available upon publication.
>
---
#### [new 060] Set-Supervised Diffusion Policy: Learning Action-Chunking Diffusion through Corrections
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决扩散策略在分布偏移下的脆弱性问题。通过利用人类纠正数据，提出SDP框架提升策略鲁棒性与数据效率。**

- **链接: [https://arxiv.org/pdf/2606.01865](https://arxiv.org/pdf/2606.01865)**

> **作者:** Zhaoting Li; Gang Chen; Javier Alonso-Mora; Cosimo Della Santina; Jens Kober
>
> **摘要:** Diffusion policies have recently emerged as a powerful framework for robotic manipulation. However, like other behavior cloning methods, they remain vulnerable to distributional shift, often requiring human-in-the-loop interventions to correct failures during deployment. These interactions naturally provide paired supervision in the form of the robot's undesired actions and the human teacher's corrective actions. Yet existing data aggregation pipelines and standard behavior cloning losses largely ignore this negative signal from undesired actions, leading to overfitting to teacher's actions and an increasing reliance on costly expert data. To address this limitation, we propose Set-Supervised Diffusion Policy (SDP), a novel learning framework that utilizes contrastive action-chunk data to train diffusion policies from human corrections. From paired positive and negative action-chunks, SDP constructs a set of desired action-chunks and designs a training pipeline that encourages the diffusion policy to align with the set. Through extensive experiments across multiple robotic manipulation tasks, we demonstrate that SDP consistently improves policy performance, with particularly strong gains in robustness to noisy data. Moreover, SDP induces high-quality aggregated datasets, enabling more efficient and reliable policy learning from human-in-the-loop corrections. Our code is available at this https URL.
>
---
#### [new 061] PaCo-VLA: Passivity-Shielded Compliance Prior for Contact-Rich Vision-Language-Action Manipulation
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文提出PaCo-VLA，解决接触密集操作中语义与控制的衔接问题。通过引入被动性保护机制，提升VLA模型在力敏感任务中的安全性与精度。**

- **链接: [https://arxiv.org/pdf/2606.00515](https://arxiv.org/pdf/2606.00515)**

> **作者:** Haofan Cao; Zhaoyang Li; Zhichao You; Liang Guo; Tianrui Li
>
> **备注:** Under review, code will be available soon
>
> **摘要:** Contact-rich manipulation demands both high-level semantic reasoning and the safe regulation of high-frequency contact dynamics. While Vision-Language-Action (VLA) models provide unprecedented semantic generalization, their low-rate outputs lack the reliability required for direct plant authority in force-sensitive tasks. To bridge this semantic-to-control gap, we introduce PaCo-VLA, a passivity-shielded compliance prior that recasts the VLA interface. Rather than trusting VLAs with direct motor commands, PaCo-VLA treats network outputs as task-level compliance proposals: semantic bindings, task stages, and admittance schedules. A high-frequency, proposal-independent passivity shield governs these proposals through energy-tank accounting and boundary checks, preventing invalid, stale, or unverified model predictions from bypassing low-level contact physics. This decoupled architecture also enables causal evaluation, isolating semantic contributions from geometric shortcuts. Extensive simulated and real-world connector-insertion experiments demonstrate that PaCo-VLA achieves superior precision over unshielded VLA baselines, sustaining zero passivity violations even under adversarial compliance shifts. This framework establishes a provably sampled-passive runtime contract at the admittance port and provides a runtime interface for deploying foundation models in contact-rich domains.
>
---
#### [new 062] World-Task Factorization for Robot Learning
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于机器人学习任务，旨在解决策略泛化问题。通过分离世界与任务因素，提升模型在新环境中的适应能力。**

- **链接: [https://arxiv.org/pdf/2606.02027](https://arxiv.org/pdf/2606.02027)**

> **作者:** Eduardo Sebastián; Adrian Pfisterer; Vito Mengers; Oliver Brock; Amanda Prorok
>
> **摘要:** Robot learning must produce policies that generalize to new combinations of constraints, teammates, and environments. To achieve this, we must structurally factor the policy, which is a choice that dictates what generalizes, what requires retraining, and what remains entangled. Existing methods span a wide spectrum, from expecting structure to emerge from data scaling, to hand-designing it via hierarchies, skill libraries or learned specializations. In this paper, we study what we argue is the most fundamental factorization in robotics: separating the world from the task. We investigate the conditions under which this factorization is principled. World factors are properties of the embodied system and the environment; they exist independently of intent. Task factors are defined by the task's logic over what the world admits. We formalize this asymmetry through Bayesian model evidence: it aligns with the data-generating process, maintains high likelihood through an analytical world model, and reduces the Occam razor's penalty on task parameters. We instantiate this factorization by pairing AICON, a differentiable graph of recursive estimators and interconnections that is compositional, operates without task-specific data, and propagates cost gradients to actuators, with a compact, learned policy that modulates gradient paths. Gradients serve as the interface between the two factors: they carry world structure through the graph and task structure through costs, enabling low-dimensional learning while preserving structural generalization. We test the world/task factorization across three problems that encompass heterogeneous robots, environments, task logic and sensorimotor modalities. Our framework outperforms end-to-end baselines and analytical heuristics in all settings, generalizes zero-shot to out-of-distribution configurations, and transfers to real hardware without retraining.
>
---
#### [new 063] DriveAnchor: Progressive Anchor-based Flow Learning for Autonomous Driving Planning
- **分类: cs.RO**

- **简介: 该论文提出DriveAnchor，用于自动驾驶规划，解决行为多样性、可控性和安全性问题。通过三阶段框架实现高效轨迹生成与优化。**

- **链接: [https://arxiv.org/pdf/2606.00519](https://arxiv.org/pdf/2606.00519)**

> **作者:** Limin Yan; Haoyun Tang; Yutao Qiu; Hongqing Liu; Haoyu Xu
>
> **摘要:** We present DriveAnchor, a three-stage framework for autonomous driving planning that achieves behavioral diversity, controllability, and safety in a composable pipeline. Demonstration Flow Pretraining replaces the unstructured Gaussian prior with a vocabulary of 2,398 trajectory shapes constructed by farthest-point sampling, structurally grounding behavioral diversity in vocabulary coverage. Guided Flow Post-training jointly post-trains an Energy Field module with flow matching (FM), conditioning the Energy Field on static road geometry alone, to relocate anchors toward user-specified corridor polygons before flow generation, adding controllability without differentiable guidance; after Stage 2, new corridor presets require only Energy Field updates, not FM retraining. Reward-Refined Flow Fine-tuning applies zeroth-order reinforcement learning to align each anchor's output with collision-avoidance objectives: because the flow-matching model is a deterministic feedforward network in single-step mode, each anchor uniquely determines the output trajectory, reducing reward optimization to a direction search in anchor space without log-likelihood computation or ODE-to-SDE conversion. Evaluated on approximately 2 million held-out driving scenarios, DriveAnchor reduces near-range collision rates by 89% and improves mean reward by 32% without degradation in imitation accuracy, with 2.06 ms inference on NVIDIA Drive Orin. DriveAnchor has been validated through real-world vehicle testing, confirming its practicality for production deployment.
>
---
#### [new 064] Generative Multi-Robot Motion Planning via Diffusion Modeling with Multi-Agent Reinforcement Learning Guidance
- **分类: cs.RO**

- **简介: 该论文属于多机器人运动规划任务，解决协同问题。通过结合扩散模型与MARL，实现去中心化轨迹生成与协调，降低冲突率。**

- **链接: [https://arxiv.org/pdf/2606.00933](https://arxiv.org/pdf/2606.00933)**

> **作者:** Suk Ki Lee; Venkata Sai Deepak Mutta; Hyunwoong Ko
>
> **备注:** 11 pages, 6 figures, 1 table. This paper has been accepted for publication in the proceedings of ASME IDETC-CIE 2026
>
> **摘要:** Coordinating multiple robots in shared environments requires generating feasible trajectories for each agent while accounting for interactions among agents. Centralized planning approaches become difficult to scale as the number of robots increases, while decentralized approaches that allow each agent to plan independently do not inherently account for inter-agent interactions. This paper presents a framework for coordinated multi-robot motion planning that combines decentralized generative trajectory planning with multi-agent reinforcement learning (MARL)-based coordination. Each robot independently generates candidate trajectories using a diffusion model trained on single-agent motion data, leveraging the generative model's ability to produce feasible and diverse trajectories. To reduce conflicts between agents, a centralized value function trained via MARL guides the reverse diffusion process through gradient-based steering, enabling interaction-aware trajectory generation without centralized joint planning or retraining of the generative model. This guidance follows an exponential tilting formulation, in which the value function biases the denoising distribution toward trajectories with higher expected multi-agent return. The framework is evaluated in a simulated maze environment with four mobile robots. Experimental results show that the proposed value-guided diffusion planning reduces the inter-agent interference rate from 55.4% to 41.8%, demonstrating that coordination can be effectively achieved while preserving the scalability of decentralized trajectory generation. These results suggest that MARL-based value guidance can effectively introduce coordination into decentralized generative planners without requiring a fully joint multi-robot model.
>
---
#### [new 065] Spatio-Temporal Reconnection for Multi-Robot Networks using Adaptive Prescribed-Time CBFs
- **分类: cs.RO**

- **简介: 该论文属于多机器人系统任务，解决通信连接受限下的高效任务执行问题。提出自适应预定时间控制屏障函数框架，实现机器人临时断连后可靠重连。**

- **链接: [https://arxiv.org/pdf/2606.01526](https://arxiv.org/pdf/2606.01526)**

> **作者:** Hao Liu; Yupeng Yang; Yanze Zhang; Wenhao Luo
>
> **备注:** 6 pages, 6 figures, accepted by IFAC 2026
>
> **摘要:** In multi-robot systems, maintaining persistent communication graph connectivity is often overly restrictive, especially when robots have limited communication ranges but operate in large environments. Instead, allowing robots to temporarily disconnect and later reconnect is often more desirable for efficient task execution while still ensuring timely information sharing across the team. In this paper, we propose an adaptive prescribed-time control barrier function (adaptive PT-CBF) framework that enables robots to temporarily disconnect and re-enter the communication range within an adjustable and feasible prescribed time. Moreover, we introduce a reconnection triggering mechanism that jointly considers task execution and reconnection urgency, thereby providing a principled way to decide when reconnection should occur. Theoretical analysis justifies convergence to the satisfying reconnection within a prescribed finite time. Experimental results validate the performance of our proposed adaptive PT-CBF with improved task efficiency and satisfying reconnections.
>
---
#### [new 066] Physics-Informed Modeling and Control of Emergent Behaviors in Robot Swarms
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于机器人集群控制任务，旨在解决多阶段群体行为建模与控制问题。提出PhySwarm框架，结合物理约束的宏观与微观模型，实现可解释的群体行为生成与调控。**

- **链接: [https://arxiv.org/pdf/2606.01597](https://arxiv.org/pdf/2606.01597)**

> **作者:** Zixuan Jin; Wenzhuo Zhang; Shuxian Quan; Zirui Dong; Fangwen Ye; Yuchen Shi; Cheng Xu
>
> **摘要:** Robot swarms can exhibit coherent collective behaviors through local perception, limited communication and decentralized decision-making, yet modeling and controlling such emergence remains challenging when behaviors unfold over multiple phases. Here we introduce PhySwarm, a physics-informed micro--macro framework that represents multi-stage swarm emergence as physically constrained density-field evolution coupled to executable robot motion. At the macroscopic level, a multi-phase advection--diffusion--reaction model (Macro-ADR) describes phase-dependent swarm-density evolution through directed transport, diffusion-based spatial regulation and behavioral phase transitions. At the microscopic level, an equivalent deterministic motion model (Micro-EDM) realizes these mechanisms through potential-field advection, density-gradient compensation and rate- or event-gated phase switching. A neural-physics controller (NPC) maps local observations and temporal memory to bounded physical parameters, and is trained with a reinforcement learning--PINN objective that combines task rewards with macro-scale density residuals and micro-scale motion-consistency constraints. In several proof-of-concept swarm missions -- including trail-guided foraging, formation-reconfigurable navigation and role-adaptive search and rescue -- we demonstrate that PhySwarm can generate distinct multi-stage emergent behaviors within a unified physics-informed modeling framework. The learned density fields and physical parameters provide interpretable evidence of how advection, diffusion and reaction jointly regulate multi-stage swarm organization. These results establish a physics-informed route for learning, interpreting and controlling emergent behaviors in robot swarms.
>
---
#### [new 067] Whole-Body Inverse Kinematics with Graph Diffusion
- **分类: cs.RO**

- **简介: 该论文属于机器人逆运动学任务，旨在解决多结构机器人配置生成问题。提出GraphDiff-IK框架，通过图扩散模型生成符合末端位姿的关节配置，支持多种机器人结构。**

- **链接: [https://arxiv.org/pdf/2606.00086](https://arxiv.org/pdf/2606.00086)**

> **作者:** Helong Huang; Kai Tan; Feng Wen; Guowei Huang; Xingyue Quan
>
> **摘要:** Inverse kinematics (IK) is a fundamental problem in robotics, requiring the generation of joint configurations that satisfy target end-effector poses. Existing approaches often struggle to generalize across diverse robot morphologies and to effectively model the multi-modal nature of IK, particularly in articulated systems with multiple kinematic branches. In this work, we propose GraphDiff-IK, a structure-aware graph diffusion framework for inverse kinematics. Specifically, we represent the robot as a kinematic graph constructed from the robot URDF, where nodes correspond to actuated joints and edges encode kinematic dependencies. Building upon this representation, we formulate IK as a conditional graph diffusion process that directly generates joint configurations on the robot graph. To better capture structural dependencies in articulated systems, we further introduce a structure-aware graph reasoning framework with hierarchical stage-wise message passing and torso-aware conditioning for multi-branch robots. In addition, we incorporate noisy forward kinematics feedback and task-space supervision to improve geometric consistency during denoising. The proposed framework provides a unified formulation that naturally supports single-arm robots, dual-arm systems, and articulated robots with torso or waist structures. Extensive experiments on diverse robotic platforms demonstrate that the proposed method achieves accurate and stable IK performance while preserving the ability to generate multiple feasible solutions for redundant robotic systems.
>
---
#### [new 068] Tether-Aware Dynamic Collision Avoidance for USV-HROV Systems
- **分类: cs.RO**

- **简介: 该论文属于水下机器人协同控制任务，解决USV跟踪HROV时的动态避障问题。通过引入系缆安全域和速度障碍方法，确保避障同时避免系缆绷紧。**

- **链接: [https://arxiv.org/pdf/2606.01112](https://arxiv.org/pdf/2606.01112)**

> **作者:** Yang Gu; Ziyang Hong; Xuanlin Chen; Hao Wei; Cheng Wang; Shujie Yang; Yulin Si
>
> **摘要:** Heterogeneous marine robotic systems composed of an unmanned surface vehicle (USV) and a hybrid remotely operated vehicle (HROV) have shown great potential for subsea cable inspection. In such missions, the USV tracks the HROV at the surface while supplying power and communication through an umbilical tether. However, dynamic collision avoidance for the USV during HROV tracking is challenging because the submerged tether may scrape against passing vessels, while evasive maneuvers can enlarge the USV--HROV separation, thereby increasing the likelihood of tether tautness and compromising HROV operations. To address these challenges, this work proposes a tether-aware dynamic collision avoidance method for a USV tracking an HROV. First, a tether safety-aware planar domain is introduced to represent the three-dimensional collision risk between the tether and obstacle vessels without an explicit tether shape model. Second, a tether tautness-aware velocity obstacle method is developed to achieve safe avoidance while reducing the likelihood of tether tautness. Finally, the method is integrated with line-of-sight guidance to coordinate HROV tracking and collision avoidance. Gazebo-based simulations show that the proposed method avoids dynamic obstacle vessels while maintaining tether safety and reducing the likelihood of tether tautness during USV evasive maneuvers.
>
---
#### [new 069] Dynamic Resilient Spatio-Semantic Memory with Hybrid Localization for Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出DREAM框架，解决动态室内环境中移动操作的场景表示问题，通过在线构建语义空间记忆和优化定位方法，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2606.00576](https://arxiv.org/pdf/2606.00576)**

> **作者:** Zhijie Yan; Shufei Li; Ze Zhang; Xin Liu; Yuhang Zheng; Zuoxu Wang
>
> **备注:** Code, CAD model, and real-robot demonstrations are available at this https URL
>
> **摘要:** Reliable mobile manipulation in dynamic indoor environments requires a scene representation that remains geometrically consistent, semantically queryable, and computationally bounded as the environment changes. Existing systems often rely on pre-built maps, static-scene assumptions, or highly accurate camera poses, which can lead to stale or misaligned scene information when target objects are relocated or pose estimates are corrected. This paper presents DREAM, a real-robot mobile manipulation framework that integrates perception, memory, localization, navigation, and manipulation in previously unseen indoor environments without a pre-built map. DREAM constructs an online spatio-semantic voxel memory from RGB-D observations registered by a LiDAR-inertial-visual SLAM backend. It further introduces pose-graph-aware Redundancy-Aware Memory Pruning (RMP) to update historical observations after pose corrections while keeping long-horizon observation history bounded. For target localization and reacquisition, DREAM combines language-conditioned 3D retrieval, open-vocabulary image detection, and multimodal large language model based semantic verification. Real-robot experiments in four dynamic indoor laboratory scenes show that DREAM improves long-horizon task success rates from 40%-60% with DynaMem to 55%-70%, while maintaining a memory footprint of 0.37-0.63 GB and an online memory-update time of 0.43-0.53 s across scenes.
>
---
#### [new 070] Beyond Pure Sampling: Hybrid Optimization Mechanisms for Non-Convex Model Predictive Control
- **分类: cs.RO; math.OC**

- **简介: 该论文属于机器人路径规划任务，解决非凸模型预测控制中的优化问题，提出混合优化机制ME-DDP，结合梯度与采样方法提升控制效果。**

- **链接: [https://arxiv.org/pdf/2606.00737](https://arxiv.org/pdf/2606.00737)**

> **作者:** Yuichiro Aoyama; Minchan Jung; Akash Ratheesh; Evangelos A. Theodorou
>
> **备注:** 28 pages, 13 figures
>
> **摘要:** This paper investigates the optimization mechanisms of non-convex Model Predictive Control (MPC) using the Maximum Entropy Differential Dynamic Programming (ME-DDP) framework. Navigating non-convex cost landscapes induced by nonlinear dynamics, multiple obstacles, etc. remains a fundamental challenge in robotics, where gradient-based methods frequently converge to suboptimal local minima. We demonstrate a dual-step optimization mechanism designed to overcome these traps. (1) an initial phase of using DDP to exploit the gradient of the cost landscape, followed by (2) disruption of the optimization via sampling from policies characterized by the inverse Hessian of the action-value function. We provide a rigorous analysis of this sampling mechanism of three ME-DDP variants: Unimodal Gaussian ME-DDP, Multimodal Gaussian ME-DDP, and Stein Variational DDP. Furthermore, with navigation tasks of four robotic systems under cluttered environments, we conduct extensive benchmarking of three variants of the ME-DDP, against deterministic DDP, and one of the most successful sampling-based schemes, Model Predictive Path Integral (MPPI) control with three policy parameterizations and update laws that correspond to those of ME-DDPs. The results show that in low-dimensional systems where the cost landscapes are relatively simple and local information is sufficiently representative, our framework consistently outperforms MPPIs. In high-dimensional systems, MPPI can occasionally discover aggressive maneuvers that enable it to steer the systems faster than DDP-based methods, whereas our method maintains a higher, more stable success rate. Finally, we validate the practical efficacy of the framework through hardware experiments with a quadrotor navigating a dense, non-convex obstacle field, confirming the robustness of the proposed framework for real-world deployment.
>
---
#### [new 071] Trans2Occ: Voxel Occupancy Estimation and Grasp for Transparent Objects from Simulation to Reality
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，解决透明物体检测与抓取问题。通过单视角RGB图像预测体素占用，实现有效抓取，适用于真实环境。**

- **链接: [https://arxiv.org/pdf/2606.01777](https://arxiv.org/pdf/2606.01777)**

> **作者:** Yixuan Yang; Sha Zhang; Rui Li; Zhenfei Yin; Xinzhu Ma; Yiran Qin; Lei Bai; Xudong Xu; Shilin Shan; Wangmeng Zuo; Yanyong Zhang; Wanli Ouyang; Feng Zheng; Shixiang Tang; Dongzhan Zhou
>
> **摘要:** Transparent objects remain challenging for robotic perception due to unreliable depth sensing caused by refraction and reflection. While prior approaches rely on multi-view reconstruction or depth completion, they are often difficult to scale or deploy in real-world robotic systems. In this paper, we present a practical framework for transparent object perception and manipulation based on single-view RGB input. Our approach predicts voxel-space occupancy directly from a single image, providing a geometry-aware representation that supports downstream robotic grasping. To enable large-scale training, we construct a simulation pipeline that generates paired RGB images and voxel occupancy annotations under diverse materials and lighting conditions. We demonstrate that the predicted occupancy representation is robust to domain shifts and transfers effectively from simulation to real-world robotic setups without fine-tuning. A simple rule-based grasping strategy built on top of the occupancy further achieves reliable grasp performance on transparent objects. Extensive experiments in both simulation and real-world environments show that our framework provides accurate 3D understanding and enables practical manipulation of transparent objects. These results suggest that single-view occupancy prediction offers a scalable and effective solution for transparent object perception in robotics.
>
---
#### [new 072] A Kinetic Theory of Encounter-Based Information Propagation in Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文研究多机器人系统中基于接触的信息传播问题，旨在解决信息时效性与跟踪精度的关系。通过理论分析与仿真验证，提出信息传播的三个限制因素。**

- **链接: [https://arxiv.org/pdf/2606.02296](https://arxiv.org/pdf/2606.02296)**

> **作者:** Alkesh K. Srivastava; Philip Dames
>
> **摘要:** Multi-robot systems cannot assume persistent network connectivity. We study this problem through target tracking, where performance depends on how quickly target information is sensed, transported through the team, and used before it becomes stale. When robots exchange information only through physical encounters, tracking becomes a kinetic information-transport problem: robot motion induces encounters, encounters carry target-state estimates, information age determines staleness, and stale information produces tracking error. This paper develops a kinetic theory of encounter-based information propagation and identifies three limits. The first is an access limit -- information cannot support team-level coordination unless it spreads beyond the robots that sensed it. The second is a staleness limit -- even propagated information loses value as the target moves. The third is a geometry limit -- when target motion outpaces information transport, tracking error approaches a saturation regime where communication improvements alone have diminishing returns. We evaluate the theory through large-scale simulations varying team size, operating area, communication range, and target speed. Results support the proposed access-staleness-geometry decomposition: communication coverage governs the access transition; once information is accessible, tracking error is shaped by target displacement; and this response is locally linear in restricted regimes but nonlinear over broader ranges because of sensing refreshes and bounded geometry. Across controlled sweeps and joint variation, the derived access and staleness coordinates reliably describe tracking performance. Together, these results establish a kinetic-theoretic framework for predicting and designing encounter-based multi-robot systems.
>
---
#### [new 073] DRL-Based Pose Control for Double-Ackermann Robots Under Actuation Uncertainties
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决DRL在真实机器人部署中的泛化问题。通过改进DRL框架，提升双阿克曼机器人姿态控制的鲁棒性，并采用sim-to-sim-to-real方法提升迁移效果。**

- **链接: [https://arxiv.org/pdf/2606.00313](https://arxiv.org/pdf/2606.00313)**

> **作者:** Oussama Zaim; Mélodie Daniel; Aly Magassouba; Miguel Aranda; Olivier Ly
>
> **备注:** 6 pages, 4 figures, 2 tables, Accepted for Uncertainty in Open-World Robotics an IEEE International Conference on Robotics & Automation (ICRA 2026) workshop
>
> **摘要:** Robust deployment of deep reinforcement learning (DRL) policies on real robots remains challenging due to discrepancies between simulation and real-world dynamics. We address this issue in the context of maneuvering with double-Ackermann-steering mobile robots, which introduce additional constraints due to their non-holonomic nature. Building upon the DRL framework ManeuverNet, we extend its objective from position control to full pose control, resulting in a more challenging task. We further investigate the impact of actuation-related uncertainties on policy transfer. The use of simplified actuation models during training of the extended policy can lead to poor generalization, shown by a success rate drop from 100% in PyBullet to 25% in Gazebo under stricter evaluation conditions. To address this limitation, we adopt a sim-to-sim-to-real approach, where actuation effects observed in Gazebo are incorporated into the PyBullet training environment. Using multi-environment DRL with SAC and CrossQ, we learn policies that remain robust despite modeling inaccuracies. This approach can significantly reduce the performance gap across simulators, achieving up to 92% success rate in Gazebo and maintaining 69% under stricter thresholds, with successful transfer to a real robot without additional tuning.
>
---
#### [new 074] A Simulation Platform for Flapping-Wing Vehicles
- **分类: cs.RO**

- **简介: 该论文属于飞行器仿真任务，旨在解决FWAV自主性不足的问题。通过构建高保真仿真平台FWAV-Sim，整合气动模型、湍流生成和真实传感器模拟，提升仿真与现实的匹配度。**

- **链接: [https://arxiv.org/pdf/2606.02370](https://arxiv.org/pdf/2606.02370)**

> **作者:** Haichuan Li; Tomi Westerlund
>
> **摘要:** Flapping-wing aerial vehicles (FWAVs) demonstrate remarkable agility but face substantial autonomy challenges due to their high sensitivity to aerodynamic disturbances and limited sensor payload capacity. Current simulation platforms typically rely on oversimplified laminar flow assumptions and idealized sensor models, failing to capture the complex turbulence patterns and perceptual limitations encountered in real-world operation. This simulation-to-reality discrepancy significantly impedes the development of robust autonomy systems for FWAVs. We introduce FWAV-Sim, a high-fidelity Unity-based simulation framework that integrates: (1) a composite aerodynamic model combining quasi-steady blade-element theory with bluff-body drag effects, (2) spatiotemporally correlated turbulence generation through fractal noise synthesis, and (3) realistic sensor simulation including noisy IMU measurements, LiDAR point clouds, and RGB camera feeds. Our platform enables scalable generation of synchronized datasets containing ground-truth vehicle states, aerodynamic forces, turbulent wind fields, and multi-modal sensor streams. Experimental validation demonstrates that autonomy pipelines (including both controllers and perception systems) developed in FWAV-Sim exhibit significantly improved simulation capability, thereby advancing the outstanding performance in simulation-based development for flapping-wing aerial systems.
>
---
#### [new 075] FAIR^2 Drones: An AI-Ready Standard for Cross-Domain Wildlife Drone Datasets
- **分类: cs.RO**

- **简介: 该论文提出FAIR^2 Drones标准，解决跨领域无人机数据共享问题，整合生态、机器人和计算机视觉，提升数据复用与协作效率。**

- **链接: [https://arxiv.org/pdf/2606.00355](https://arxiv.org/pdf/2606.00355)**

> **作者:** Jenna Kline; Kilian Meier; Vandita Shukla; Edouard G. A. Rolland; Elena Iannino; Lucie Laporte-Devylder; Constanza Andrea Molina Catricheo; Blair Costelloe; Elizabeth Campolongo; Henrik S. Midtiby; Devis Tuia; Benjamin Risse; Ulrik P.S. Lundquist; Anders Lyhne Christensen; Fabio Remondino; Thomas Richardson; Tanya Berger-Wolf
>
> **摘要:** Animal ecology data collection using drones represents a substantial investment of time, expertise, and financial resources. Yet most existing datasets serve only a single research community, limiting interdisciplinary reuse. We propose a unified drone dataset standard, FAIR^2 Drones, that bridges ecology, robotics, and computer vision by building on existing FAIR and AI-ready data frameworks while adding essential platform metadata and annotation specifications. Our standard enables datasets to simultaneously support ecological analysis, robotics algorithm development, and computer vision benchmarking. We provide open-source validation tools, reference implementations, and multimodal extensions linking drone imagery with complementary sensors such as camera traps, GPS, and acoustics. By standardizing metadata across disciplines, this framework maximizes the scientific return on investment for costly field deployments and accelerates cross-domain collaboration in environmental monitoring.
>
---
#### [new 076] PSG-Nav: Probabilistic Scene Graph Navigation via Multiverse Decision Making
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于开放词汇导航任务，解决感知不确定性带来的导航决策问题。提出PSG-Nav方法，构建概率场景图并采用多宇宙决策提升导航效果。**

- **链接: [https://arxiv.org/pdf/2606.01313](https://arxiv.org/pdf/2606.01313)**

> **作者:** Rufeng Chen; Yue Chang; Xiaqiang Tang; Hechang Chen; Sihong Xie
>
> **备注:** 21 pages, 7 figures. ICML 2026
>
> **摘要:** Open-vocabulary navigation requires embodied agents to manage significant perception uncertainty stemming from semantic ambiguity and model errors. However, most existing works settle for local optimal deterministic approaches, depriving complex navigation decision-making over multiple composite possibilities that are critical for globally better solutions. In this paper, we propose Probabilistic Scene Graph Navigation (PSG-Nav), which constructs a 3D Probabilistic Scene Graph that uses full semantic categorical distributions to account for perception uncertainty. To efficiently use the local distributions to compose and reason about the optimal navigation landmarks, we propose Multiverse Decision to sample multiple most likely world settings from the joint distribution, and evaluate navigation landmarks based on the compatibility between landmarks and multiverses. To mitigate false positives due to epistemic uncertainty in open-vocabulary navigation, we introduce the Evidential Experience Calibrator, which enables online lifelong adaptation by cross-validating detections against memories of past successes and failures. Extensive experiments on widely-used benchmarks MP3D, HM3D, and HSSD demonstrate that PSG-Nav establishes new state-of-the-art results, achieving Success Rates of 66.1%, 44.8%, and 67.9%, respectively. Code is available at: this https URL
>
---
#### [new 077] Coarse-to-Fine Compositional Diffusion for Long-Horizon Planning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于长程规划任务，解决组合生成中全局结构不明确的问题。提出CoFi方法，先构建全局结构再细化局部细节，提升生成质量与效率。**

- **链接: [https://arxiv.org/pdf/2606.00837](https://arxiv.org/pdf/2606.00837)**

> **作者:** Byoungwoo Park; Utkarsh A. Mishra; Jaemoo Choi; Juho Lee; Yongxin Chen
>
> **备注:** Project page: this https URL
>
> **摘要:** Diffusion models provide strong priors for generating structured data, but many tasks require outputs beyond the scale on which these models are typically trained. Compositional generation addresses this by composing overlapping local plans from a pretrained short-horizon prior into a long-horizon output. However, standard composition primarily enforces agreement between neighboring local plans, yielding local consistency without directly specifying the global structure of the full composition. As a result, locally compatible plans may still form an implausible route, task sequence, or temporal evolution. Existing methods improve global coherence by repeatedly propagating local consistency signals or by adding inference-time optimization, but these procedures become expensive as the number or dimensionality of local plans increases. We propose Coarse-to-Fine Compositional Diffusion (CoFi), an inference-time sampler that separates global structure formation from local detail refinement. CoFi first aligns local denoised estimates around a shared coarse structure, producing a global scaffold that captures the long-range task-level arrangement. It then diffuses this scaffold to an intermediate noise level and denoises it with the same pretrained local prior, restoring local fine structure while preserving the scaffold-induced global coherence. Across long-horizon robotic planning, panoramic image generation, and long video generation, CoFi not only improves both global coherence and local sample quality over prior compositional baselines, but also requires 2-8x fewer denoiser evaluations.
>
---
#### [new 078] Behavior Cloning of MPC for 3-DOF Robotic Manipulators
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于控制任务，旨在解决MPC计算负担过重的问题。通过行为克隆方法，用神经网络替代MPC，提升3-DOF机械臂的实时控制效率。**

- **链接: [https://arxiv.org/pdf/2606.00383](https://arxiv.org/pdf/2606.00383)**

> **作者:** Theo Guegan; Dexter Wen Jie Teo
>
> **备注:** Accepted at the IEEE ICRA 2026 Workshop on Reinforcement Learning in the Era of Imitation Learning (RL4IL), 6 pages excluding references
>
> **摘要:** While Model Predictive Control (MPC) provides strong stability and robustness, it imposes a significant computational burden on real-time systems. This paper investigates the application of Behavior Cloning to approximate MPC policies for the real-time control of a 3-degree-of-freedom robotic manipulator. We present a baseline controller combining Inverse Kinematics with MPC and evaluate neural network architectures, ranging from classical regression algorithms to deep learning models including Deep MLPs and RNNs, to derive computationally efficient surrogate policies. We analyze generalization capabilities, stability considerations, and the trade-offs inherent in different architectural choices. Our empirical study employs both online and offline evaluations to assess performance regarding accuracy, computational efficiency, and fidelity to the original MPC policy. Our results demonstrate that Behavior Cloning can effectively reduce the computational burden of MPC policies for 3-DOF robotic manipulators, achieving a 3x reduction in inference latency with a 84.98% success rate under relaxed tolerances. Notably, we find that static architectures outperform temporal variants, confirming the sufficiency of instantaneous state observations for this task. However, we observe a precision gap under strict tolerances, which suggest that while Behavior Cloning captures the global optimal trajectory, further research is needed to minimize terminal steady-state error.
>
---
#### [new 079] SafeVLA-Bench: A Benchmark for the Success-Safety Gap in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型在完成任务时的安全性问题。提出SafeVLA-Bench框架，评估策略在成功完成任务的同时是否违反安全规范。**

- **链接: [https://arxiv.org/pdf/2606.00773](https://arxiv.org/pdf/2606.00773)**

> **作者:** Jialiang Fan; Weizhe Xu; Oleg Sokolsky; Insup Lee; Fanxin Kong
>
> **备注:** 27 pages, 5 figures
>
> **摘要:** Vision-language-action (VLA) benchmarks measure whether a policy completes a requested manipulation task, but binary success can hide safety-relevant trajectory behavior: reaching the goal while applying excessive contact, disturbing bystander objects, destabilizing the held object, or entering robot self-contact. We present SafeVLA-Bench, a post-hoc safety-evaluation framework for existing simulator-based VLA benchmarks. It formalizes task-aware safety requirements as Signal Temporal Logic (STL) specifications and reports native success with two unsafe-success metrics: Succ-But-Unsafe (SBU), the fraction of rollouts that both succeed and violate safety, and Violation Severity Index (VSI), a bounded worst-violation depth score. We instantiate SafeVLA-Bench on LIBERO and RoboCasa-365, evaluating nine policy-benchmark entries across tabletop and kitchen manipulation tasks. High task success does not imply safe execution: high-SR tabletop baselines still leave 13 to 15 percent unsafe-episode rates,and 36 to 56 percent of successful RoboCasa-365 rollouts violate at least one active safety clause. Project page: this https URL.
>
---
#### [new 080] ROG-Grasp: Root-Oriented Geometry for Robotic Grasping and Placement
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决农产品定向抓取与放置问题。通过几何感知估计作物方向，实现稳定抓取与精准定位。**

- **链接: [https://arxiv.org/pdf/2606.00449](https://arxiv.org/pdf/2606.00449)**

> **作者:** Zijian An; Augustus Sroka; Ran Yang; Bill Cai; Satoru Eto; Brian Poon; Kelvin Cai; Shijie Geng; Feng Liu; Yiming Feng; Lifeng Zhou
>
> **备注:** Comments: 7 pages, 6 figures. Video: this https URL
>
> **摘要:** Orientation-aware manipulation is essential in post-harvest agricultural processing, where produce must be grasped and placed in consistent configurations. This paper presents ROG-Grasp, a geometry-based robotic grasping and placement framework that estimates the produce orientation from root surface geometry using RGB-D perception. A YOLO-based root detector and point cloud plane fitting are used to infer the root normal, enabling stable grasp pose generation and orientation-constrained Cartesian motion planning. Experiments on tomatoes and onions demonstrate high success rates and stable execution time in both isolated and cluttered scenarios. Compared with vision-language-action (VLA) policies, the proposed method achieves more reliable and accurate grasp completion with faster execution. These results highlight the effectiveness of geometry-driven perception for practical orientation-controlled manipulation tasks. A video of our paper is available online this https URL.
>
---
#### [new 081] HOIST: Humanoid Optimization with Imitation and Sample-efficient Tuning for Manipulating Suspended Loads
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于人形机器人操控悬吊负载的任务，解决如何安全高效地控制悬吊负载的问题。通过结合模仿学习与强化学习，提升放置精度和停止行为。**

- **链接: [https://arxiv.org/pdf/2606.00252](https://arxiv.org/pdf/2606.00252)**

> **作者:** Songyang Liu; Shunyu Yao; Dingyuan Huang; Shuai Li
>
> **摘要:** Manipulating suspended payloads with humanoid robots is challenging because the robot can only influence an underactuated, oscillatory load through whole-body motion and intermittent contact. Imitation learning provides safe initial behavior but does not directly optimize final placement, while reinforcement learning from scratch is unsafe and sample-inefficient on real humanoids. We present HOIST-Humanoid Optimized with Imitation and Sample-efficient Tuning for manipulating suspended loads. HOIST first finetunes a high-level vision-language-action (VLA) policy from virtual-reality (VR) teleoperation demonstrations and executes its commands through a whole-body controller. It then uses VLA rollouts and iterative batched RL to improve placement accuracy and stopping behavior. Experiments in simulation and on a real humanoid show that HOIST improves over imitation-only and additional-demonstration baselines; compared with pure VLA rollouts, HOIST reduces translational placement error by 19.9 cm and raw angular error by 3.56 degrees, demonstrating the potential of humanoids for underactuated material-handling tasks.
>
---
#### [new 082] NDPP-Grasp: Non-Differentiable Physical Plausibility Constraint-Guided Task-Oriented Dexterous Grasp Generation
- **分类: cs.RO**

- **简介: 该论文属于机械臂抓取任务，解决物理合理性与任务目标不一致的问题。通过在扩散模型中引入物理约束，提升抓取的合理性与有效性。**

- **链接: [https://arxiv.org/pdf/2606.02432](https://arxiv.org/pdf/2606.02432)**

> **作者:** Qiuchi Xiang; Haoxuan Qu; Hossein Rahmani; Jun Liu
>
> **摘要:** Task-oriented dexterous grasp generation aims to produce dexterous grasp poses that are both physically plausible and functionally suitable for specified manipulation tasks. Existing diffusion-based methods often address these two requirements in a decoupled manner: they first train a grasp diffusion model for task alignment and then rely on post-generation refinement to improve physical plausibility. However, this after-the-fact correction strategy applies physical plausibility guidance only once the grasp has already been generated, leaving the generation trajectory itself unguided by physical constraints and potentially leading to suboptimal grasps. To address this problem, we propose a novel framework that directly injects physical plausibility guidance into the denoising process of a task-aligned grasp diffusion model in a practical and effective manner, even when physical plausibility constraints are non-differentiable. This allows physical plausibility to shape grasp generation throughout denoising while preserving task alignment. Extensive experiments demonstrate the efficacy of our framework.
>
---
#### [new 083] Infeasible optimization problems and the hierarchical augmented Lagrangian method in imitation learning
- **分类: cs.RO**

- **简介: 该论文属于模仿学习任务，解决约束不可行导致的训练不稳定问题。通过改进的增广拉格朗日方法，使策略逼近可行的约束问题。**

- **链接: [https://arxiv.org/pdf/2606.00730](https://arxiv.org/pdf/2606.00730)**

> **作者:** Roland Andrews; Justin Carpentier; Ajay Sathya
>
> **摘要:** Imitation learning (IL) is an effective approach to train complex robotics policies. Recent works have introduced hard constraints into imitation-learning optimization problems to ensure safety, stability, and robustness of the learned policy. However, we argue that these constraints are sometimes infeasible, which can lead to unstable or difficult training dynamics. We study a simple remedy for such situations based on recent theoretical results on the augmented Lagrangian method in infeasible settings. We show that our approach drives the learned policy toward the solution of a closest-feasible constrained IL problem with desirable properties. The method is illustrated on a toy driving example with a total-acceleration constraint and pedestrian-safety constraints, a setting in which infeasibility can naturally arise while still allowing a safe learned policy.
>
---
#### [new 084] Silent Failures in Physical AI: A Literature Review of Runtime Action Authorization for Autonomous Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于AI安全任务，旨在解决物理AI系统中因模型误判导致的无声故障问题。通过综述现有技术，提出运行时授权边界框架，以提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2606.00090](https://arxiv.org/pdf/2606.00090)**

> **作者:** Barak Or
>
> **备注:** 23 pages
>
> **摘要:** Physical AI systems increasingly map multimodal observations, language instructions, and learned world representations into physically consequential actions. Robotics foundation models, vision-language-action models, and world-model-based autonomous systems can condition decisions that move vehicles, robots, drones, and industrial machines. This transition exposes a safety problem that is not fully captured by conventional AI content moderation or by classical robot safety alone: a black-box model may issue a physically consequential action while appearing confident, plausible, and semantically aligned. The resulting failure can be silent, arising from sensor drift, occlusion, state-estimation error, distribution shift, hallucinated affordances, or invalid physical assumptions before downstream hardware controllers detect a violation. Across embodied foundation models, world models, robotics simulation, embodied safety benchmarks, safe control, runtime assurance, uncertainty estimation, verification, and guardrail evaluation, model capability and safety mechanisms have advanced along largely separate technical tracks. A recurring gap synthesized here is that no single stream surveyed in this review supplies a complete runtime authorization boundary between black-box Physical AI models and physical execution. The resulting analysis develops a bounded problem formulation, a definition of silent physical-action failure, a taxonomy of runtime guardrail functions, and evaluation requirements for comparing guardrails as Physical AI assurance mechanisms.
>
---
#### [new 085] PHASOR: Phase-Anchored Universal Action Representations for Humanoid Embodiments
- **分类: cs.RO**

- **简介: 该论文提出PHASOR方法，解决机器人动作嵌入空间不通用的问题。通过分解运动为周期相位和姿态，构建可迁移的动作表示，提升机器人任务性能。**

- **链接: [https://arxiv.org/pdf/2606.01851](https://arxiv.org/pdf/2606.01851)**

> **作者:** Kihyun Kim; Chaeyun Kim; Jongho Shin; Taeyoun Kwon; Junghyun Kim; Mijin Koo; Haon Park
>
> **摘要:** Learning a good action embedding space is fundamental to scalable robot policy learning, yet existing methods treat action latents as task-specific intermediates rather than first-class representations. The resulting latents are unstructured, embodiment-specific, and weakly tied to motion semantics, limiting interpretability, controllability, and transferability across robots. We position the action embedding space itself as a first-class design target, with downstream policy quality emerging from representation quality. Exploiting motion's intrinsic periodicity, we factorize it into a phase manifold that captures cyclic structure via FFT-parametric coefficients, together with a pose branch that conditions the manifold on non-periodic configuration detail. Combined with motion-semantic distillation, this factorized structure yields a cross-embodiment motion manifold that is interpretable and embodiment-agnostic by design. Anchoring multiple humanoid robots to a shared human-pretrained manifold then produces a unified action embedding space across diverse platforms, achieving strong cross-embodiment retrieval and consistent gains on downstream robot tasks.
>
---
#### [new 086] Training-Free Imitation Learning with Closed-Form Diffusion Policies
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于模仿学习任务，旨在解决扩散策略训练耗时的问题。提出无需训练的闭式扩散策略，直接从数据集进行快速模仿，提升推理速度并支持编辑预训练策略。**

- **链接: [https://arxiv.org/pdf/2606.01238](https://arxiv.org/pdf/2606.01238)**

> **作者:** Raghav Mishra; Ian R. Manchester
>
> **摘要:** While diffusion-based policies have impressive performance and expressivity, their long offline training slows down the data collection and policy deployment loop. We introduce Closed-Form Diffusion Policies, a class of training-free diffusion-based policies for imitation learning using the closed-form score derived from the demonstration dataset. We deploy CFDP with real-time inference with a mobile CPU in hardware experiments, showing it can successfully perform imitation directly from the dataset in milliseconds and with faster inference than neural diffusion policies. In experiments on imitation learning benchmarks, we show that CFDP is competitive against neural baselines that require hours of training, providing a favorable tradeoff between training time and performance. Finally, we show how closed-form diffusion policies act as a composable primitive that enables data-driven inference-time editing of pre-trained neural diffusion policies, including policy guidance and novel demonstration augmentation.
>
---
#### [new 087] Hierarchical Semantic-Augmented Navigation: Optimal Transport and Graph-Driven Reasoning for Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决复杂环境下的长期路径规划问题。提出HSAN框架，结合语义图、最优传输和图感知强化学习，提升导航准确性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.01565](https://arxiv.org/pdf/2606.01565)**

> **作者:** Xiang Fang; Wanlong Fang; Changshuo Wang
>
> **备注:** Published in NeurIPS 2025, address some typos
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLN-CE) poses a formidable challenge for autonomous agents, requiring seamless integration of natural language instructions and visual observations to navigate complex 3D indoor spaces. Existing approaches often falter in long-horizon tasks due to limited scene understanding, inefficient planning, and lack of robust decision-making frameworks. We introduce the \textbf{Hierarchical Semantic-Augmented Navigation (HSAN)} framework, a groundbreaking approach that redefines VLN-CE through three synergistic innovations. First, HSAN constructs a dynamic hierarchical semantic scene graph, leveraging vision-language models to capture multi-level environmental representations, from objects to regions to zones, enabling nuanced spatial reasoning. Second, it employs an optimal transport-based topological planner, grounded in Kantorovich's duality, to select long-term goals by balancing semantic relevance and spatial accessibility with theoretical guarantees of optimality. Third, a graph-aware reinforcement learning policy ensures precise low-level control, navigating subgoals while robustly avoiding obstacles. By integrating spectral graph theory, optimal transport, and advanced multi-modal learning, HSAN addresses the shortcomings of static maps and heuristic planners prevalent in prior work. Extensive experiments on multiple challenging VLN-CE datasets demonstrate that HSAN achieves state-of-the-art performance, with significant improvements in navigation success and generalization to unseen environments.
>
---
#### [new 088] ActMVS: Active Scene Reconstruction with Monocular Multi-View Stereo
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉场景重建任务，解决单目主动重建中实时生成高质量深度图的问题。提出ActMVS框架，结合视图图与全局优化，实现在线、一致的深度地图生成。**

- **链接: [https://arxiv.org/pdf/2606.01367](https://arxiv.org/pdf/2606.01367)**

> **作者:** Guo Pu; Yixuan Han; Zhouhui Lian
>
> **备注:** ICRA 2026
>
> **摘要:** Active scene reconstruction enables robots/UAVs to autonomously plan trajectories and reconstruct environments without costly manual data acquisition. Unlike passive methods, active reconstruction requires real-time construction of high-confidence occupancy maps for collision-free navigation. Existing approaches rely on depth sensors for occupancy map updates, increasing platform cost and weight. To advance spatial intelligence, we aim for a vision-only monocular solution. However, current monocular scene reconstruction methods operate offline and fail to deliver globally consistent dense depth at the frame rates required for robots/UAVs navigation. To bridge this gap, we introduce ActMVS, the first framework for monocular active reconstruction. Our framework integrates a view factor graph construction for informed Multi-View Stereo depth prediction, along with a global depth optimization, to enable the online generation of high-quality, globally consistent dense depth maps. This enables monocular robots/UAVs to maintain reliable occupancy maps for safe trajectory planning during reconstruction. Experiments on Replica datasets demonstrate performance competitive with RGB-D methods. Our code and data are available at this https URL.
>
---
#### [new 089] S2M-Trek: From Single to Multi-Sphere Transport via Per-Frame Deep Sets on a Wheel-Legged Robot
- **分类: cs.RO**

- **简介: 该论文研究多球运输任务，解决无固定装置下多球动态搬运问题。提出Per-Frame Deep Sets模型，实现帧级排列不变性，成功完成五球运输。**

- **链接: [https://arxiv.org/pdf/2606.01332](https://arxiv.org/pdf/2606.01332)**

> **作者:** Zong Chen; Xuebin Li; Jinpeng Xiao; Shaoyang Li; Ben Liu; Min Li; Zhouping Yin; Yiqun Li
>
> **摘要:** We study the problem of scaling dynamic loco-manipulation from a single free-rolling sphere to multiple spheres transported simultaneously on the back of a wheel-legged quadruped, without fences, grippers, or mechanical stops. Multiple identical free-rolling spheres form an unordered set with no persistent identity: their ordering may change independently at each history frame, creating a \emph{per-frame permutation symmetry} that standard history-concatenation set encoders do not explicitly enforce -- these encoders impose only a shared, diagonal permutation symmetry over the full history. We show that this symmetry mismatch leads to a concrete failure mode in curriculum-based reinforcement learning. Within the same PPO training budget, flat MLPs and branch-wise encoders plateau at or below the two-sphere stage, while a history-concatenation Deep Sets baseline (\HCDS) fails to progress past the two-sphere stage in our runs unless ball-to-slot assignments are randomised during training, suggesting that it exploits slot indices as a curriculum shortcut rather than learning identity-free multi-sphere dynamics. We propose \textbf{Per-Frame Deep Sets (\PFDS)}, which performs permutation-invariant pooling within each history frame before temporal readout; we prove that \PFDS is $\Gframe$-invariant and universally approximates continuous $\Gframe$-invariant policies. A $2{\times}2$ ablation over encoder architecture and slot randomisation separates the architectural and data-augmentation pathways, and \PFDS reaches the five-sphere stage with 100\% no-drop transport in simulation across all five random seeds. We further distill the \PFDS teacher into \TactSet via DAgger, replacing privileged sphere-state observations with a $16{\times}16$ Boolean union contact map, yielding a compact and naturally $\Gframe$-invariant tactile representation.
>
---
#### [new 090] A passive universal grasping mechanism based on an everting shell
- **分类: cs.RO; cond-mat.soft**

- **简介: 该论文属于机械设计任务，旨在解决通用抓取问题。提出一种基于展开壳的被动抓取机制，通过弹性双稳态壳体实现对任意形状物体的抓取与释放。**

- **链接: [https://arxiv.org/pdf/2606.00470](https://arxiv.org/pdf/2606.00470)**

> **作者:** Mythra V. S. Balakuntala; Safvan Palathingal; G. K. Ananthasuresh
>
> **摘要:** A passive monolithic compliant grasping mechanism that works based on the eversion of an elastically deformable bistable shell is conceptualized. It comprises grasping arms made of beam segments that work in conjunction with the everting shell. The grasper is capable of picking up a stiff object of any shape up to a maximum size and weight. The bistable shell everts upon contact with the object to enable the grasping arms envelop the object forming an enclosure. The mechanism then stays in that configuration until it is actuated again to turn the shell back to its original configuration and thereby opening the enclosure to release the object. The stiffness of the arms decides the payload of the mechanism. The size of the arms decides the largest object that can be grasped and held. The arms have distributed compliance so that they can conform to the shape of the object without applying undue force on it.
>
---
#### [new 091] Expanding Spatial and Temporal Context for Robotic Imitation Learning With Scene Graphs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人模仿学习任务，旨在解决部分观测和长期推理问题。通过引入场景图作为结构化记忆机制，提升策略性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.01072](https://arxiv.org/pdf/2606.01072)**

> **作者:** Jianing Qian; Qinhe Peng; Emmanuel Panov; Leonor Fermoselle; Dinesh Jayaraman; Bernadette Bucher; Tarik Kelestemur
>
> **摘要:** Imitation learning enables robots to learn how to execute tasks via observation. However, real-world environments like homes and offices are often severely partially observed due to their large spatial scales. In addition, many tasks involve executing a series of subtasks requiring autonomous robots to reason over extended time horizons. To address these challenges, we propose using scene graphs as an explicit and structured memory mechanism in imitation learning. By maintaining a dynamic scene graph that captures object-centric relationships and their evolution over time, our method allows the agent to retain relevant historical context during task execution to efficiently reason over incrementally accrued scene information. Our experiments on simulated mobile manipulation and real-world tabletop manipulation demonstrate that our approach substantially improves policy performance, particularly in settings that demand long-term reasoning and robust generalization under partial observability.
>
---
#### [new 092] Dynamics Are Learned, Not Told: Semi-Supervised Discovery of Latent Dynamics Geometries For Zero-Shot Policy Adaptation
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，解决机器人在动态变化环境中的策略适应问题。通过半监督方法学习潜在动力学几何，提升策略在未见动态下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.02280](https://arxiv.org/pdf/2606.02280)**

> **作者:** Zhiming Xu; Weitao Zhou; Xianghui Pan; Nanshan Deng; Chengju Liu; Qijun Chen; Chenpeng Yao
>
> **备注:** Proceedings of the 43rd International Conference on Machine Learning
>
> **摘要:** Real-world dynamics shifts pose a critical challenge for reinforcement learning in robotics, as policies tightly coupled to nominal environments often fail catastrophically when physical conditions change. Most existing methods rely on encoding explicitly identified physical parameters into a latent context, a parameter-centric paradigm that depends on pre-specified axes of variation and becomes brittle under unmodeled or compound dynamics changes. We revisit dynamics adaptation from an outcome-centric perspective: rather than telling policies what the dynamics are, we enable them to learn how dynamics affect interaction outcomes. Theoretically, this is grounded in a monotonic relationship between target-domain regret and the Lipschitz constant of a trajectory dynamics encoder. Practically, this constant can be upper-bounded through contrastive learning, yielding a smooth, task-relevant latent topology without privileged dynamics information. On MuJoCo benchmarks, our method consistently outperforms parameter-centric baselines under severe dynamics shifts, including unmodeled and time-varying parameters, while also improving in-distribution stability and latent interpretability. Overall, these results validate that controlling latent geometry is a principled mechanism for robust adaptation.
>
---
#### [new 093] FW-NKF: Frequency-Weighted Neural Kalman Filters
- **分类: cs.RO; cs.AI; eess.SP**

- **简介: 该论文属于状态估计任务，解决传感器噪声和模型不匹配问题。提出FW-NKF方法，结合频率加权与深度学习，提升定位与姿态精度。**

- **链接: [https://arxiv.org/pdf/2606.02251](https://arxiv.org/pdf/2606.02251)**

> **作者:** Adnan Harun Dogan; Berken Utku Demirel; Christian Holz
>
> **备注:** Published at ICRA 2026
>
> **摘要:** Robust state estimation is central to robotic autonomy, yet classical Kalman filters struggle with frequency-dependent disturbances and model mismatch such as sensor vibrations, electromagnetic interference, and periodic noise. Although Deep Kalman Filter (DKF) variants extend the Extended Kalman Filtering (EKF) framework by learning latent transitions, they lack explicit mechanisms to suppress band-limited noise components that typically corrupt sensor measurements in real-world scenarios. We introduce the Frequency-Weighted Neural Kalman Filter (FW-NKF), a unified hybrid approach that embeds a causal spectral-shaping operator into the Kalman measurement residual and jointly learns observation, and transition networks. By adapting both the filter spectrum and the latent state representation, FW-NKF attenuates the noise-dominated frequency bands while capturing complex residual structures. We conduct extensive experiments on four heterogeneous benchmarks, including chaotic systems such as multi-dimensional Lorenz systems and full-body inertial pose estimation, and find a reduction in localization error of up to 10% as well as marked improvements in orientation accuracy. Our ablation studies confirm that frequency weighting and deep latent-state modeling contribute to overall performance.
>
---
#### [new 094] SoFiE: Soft Finger Exoskeleton for Intelligent Grasping
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于辅助穿戴机器人领域，旨在解决手部功能障碍者的抓握问题。设计了一种软指外骨骼SoFiE，采用3D打印柔性材料和新型传感技术，实现可靠抓握辅助。**

- **链接: [https://arxiv.org/pdf/2606.00397](https://arxiv.org/pdf/2606.00397)**

> **作者:** Magnus Malthe Sigsgaard Nielsen; Nicklas Nikolaj Grønvall; Xiaofeng Xiong; Saravana Prashanth Murali Babu
>
> **摘要:** Soft wearable robotic systems have emerged as a promising solution for assisting individuals with reduced hand function. This paper presents SoFiE, a modular soft finger exoskeleton designed to assist index-finger flexion during grasping tasks. The proposed system is primarily fabricated using 3D-printed flexible materials, enabling a lightweight, low-profile, and modular design. Actuation is achieved through a tendon-driven mechanism powered by a compact DC motor, while passive extension is provided by a compliant conductive spring. This element, termed StretchSense, also functions as a proprioceptive sensor by exhibiting resistance changes under deformation. Furthermore, a novel tactile sensing approach, MagSense, is introduced, using a magnet and magnetometer pair embedded in a soft fingertip structure to estimate contact force and object compliance. The system is fully untethered and controlled by an embedded microcontroller. In addition, actuator-level sensing through motor encoder feedback enables estimation of the system state, providing a foundation for safe and adaptive control strategies. Experimental validation demonstrates the capability of the system to provide reliable pose estimation, distinguish between materials with different stiffness, and generate distinct sensor signatures across different grasping tasks. This paper details the design, fabrication, and sensing concepts of the proposed exoskeleton as a proof of concept toward modular, soft, and assistive wearable robotics.
>
---
#### [new 095] Linear Motility Maps in Nonlinear Viscous Fluids
- **分类: cs.RO; math-ph; physics.flu-dyn**

- **简介: 该论文研究非牛顿流体中的运动机制，探讨线性运动图谱在幂律粘度流体中的适用性，并提出通过非线性阻力实现逆向运动的净位移。任务是分析和设计复杂流体中的运动系统。**

- **链接: [https://arxiv.org/pdf/2606.00063](https://arxiv.org/pdf/2606.00063)**

> **作者:** Yishun Zhou; Shai Revzen
>
> **摘要:** Systems moving in low Reynolds number fluid regimes are known to be governed by a ``motility map'' which linearly relates their shape change rates to they body frame velocity moving through the fluid. A consequence of this is ``Purcell's Scallop Theorem'' -- a locomotion system that undergoes shape changes that follow the same path forward and backward in time (reciprocal body deformations) cannot achieve net displacement, regardless of pacing of those this http URL show that linear-in-velocity motility maps extend to any power law viscosity (a.k.a. Ostwald--de Waele fluid), and therefore to many biological fluids in intermediate shear ranges. We also show that the linear-in-velocity property can be violated in Carreau-Yasuda fluids to produce net motion using an ``inchworm'' model consisting of two unequal masses with unequal drag coefficients performing reciprocal motions. Interestingly, the direction of motion can be switched by changing speeds. Our results show that the linear motility map of geometric mechaincs can be used to analyze and design locomotion in power-law fluids, and that some nonlinear drag relationships such as Carreau-Yasuda can be exploited to generate net locomotion in seeming violation of the ``scallop theorem''.
>
---
#### [new 096] Adaptive PD Gains for Energy-Conscious Control in Physical Human-Robot Interaction
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于物理人机交互控制任务，旨在解决传统力控和扭矩控制造成的传感器依赖问题，提出一种自适应PD控制器以限制机器人能量，确保安全交互。**

- **链接: [https://arxiv.org/pdf/2606.00459](https://arxiv.org/pdf/2606.00459)**

> **作者:** Danyal Saqib; Francisco Andrade Chavez; Marie Charbonneau
>
> **摘要:** Compliant force or torque control are approaches often investigated to achieve safe physical human-robot interaction (pHRI). However, these approaches have limitations. Force control requires a robot to be equipped with external force sensors to track the amplitude and direction of applied forces. Torque control requires torque sensing or estimation in each joint. As this is not available on every robot, energy-based approaches offer a promising alternative. Such approaches aim to achieve safe pHRI by limiting the mechanical energy of the robot. Current schemes leveraging an energy-based approach tend to have a complex implementation, and some may require further stability verification. We hence propose an adaptive proportional-derivative (PD) controller that can limit a robot's energy under any given limit to achieve safe pHRI. The proposed controller can limit both the kinetic and potential energy of a robot, and the behaviour of the controller gains can be shaped using various parameters, defining precisely the cutoff limit and sharpness. We construct a stability proof for the controller and define a condition to ensure the controller's stability. The proposed controller's behaviour and compliance are tested on the TALOS robot from PAL Robotics both in simulation and on hardware, verifying the expected compliant and energy-limiting behaviour of the controller.
>
---
#### [new 097] Reinforcement Learning for Optimal Experiment Design in Parameter Identification of Mechatronic Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于系统辨识任务，旨在解决机械电子系统参数识别中的激励信号设计问题。通过强化学习自主生成安全且有效的激励信号，提升辨识精度。**

- **链接: [https://arxiv.org/pdf/2606.00059](https://arxiv.org/pdf/2606.00059)**

> **作者:** Julian Langschwert; Georg Schaefer; Jakob Rehrl; Stefan Huber; Simon Hirlaender
>
> **备注:** Accepted at DEXA AI4IP 2026
>
> **摘要:** Informative excitation signals are critical for accurate system identification of mechatronic systems, yet classical system identification (SI) approaches require expert knowledge and hand-crafted signal design to respect hardware safety constraints, limiting their generalizability. We propose a reinforcement learning (RL) agent that learns optimal excitation signals for a Quanser Aero 2 testbed while autonomously enforcing safety constraints through reward shaping. Evaluated across 10 independent training seeds, our comprehensive agent achieves competitive estimation accuracy across all three identified parameters, outperforming classical baselines while incurring only 0.75% safety violations.
>
---
#### [new 098] Co-training with Ego-centric Video and Demonstration for Robot Navigation Task
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决真实机器人数据收集成本高的问题。通过将第一视角视频转换为机器人可学习的数据，提升VLA模型性能。**

- **链接: [https://arxiv.org/pdf/2606.01951](https://arxiv.org/pdf/2606.01951)**

> **作者:** Shoya Kuno; Yumo Ouchi; Kanata Suzuki
>
> **摘要:** Vision-language-action (VLA) models are promising for diverse robotic tasks, but their performance heavily depends on large-scale high-quality training data, whose collection on real robots is costly and time-consuming. While prior work has explored augmenting manipulation datasets with egocentric human videos, applying such approaches to mobile robot navigation remains challenging due to viewpoint changes during locomotion. In this paper, we propose a framework that converts egocentric walking videos into datasets for mobile robot imitation learning. The proposed method estimates camera motion from human videos and transforms it into action representations compatible with ground mobile robots. By jointly training a VLA model on human-derived and robot-collected datasets, the model achieves improved language understanding and more robust action generation than training with either data source alone. Experiments on a fruit-search navigation task demonstrate that human egocentric videos provide an effective and scalable data source for mobile robot learning.
>
---
#### [new 099] DisFlow: Scene Flow from Distance Field for Object Pose, Velocity Tracking, and Dynamic Object Reconstruction
- **分类: cs.RO**

- **简介: 该论文提出DisFlow，用于动态物体位姿、运动跟踪和表面重建。解决场景流估计问题，通过距离场实现实时高精度的几何与运动信息获取。**

- **链接: [https://arxiv.org/pdf/2606.01824](https://arxiv.org/pdf/2606.01824)**

> **作者:** Lan Wu; Sheila Sutjipto; Jennifer Wakulicz; Teresa Vidal-Calleja
>
> **摘要:** We present \emph{DisFlow}, a novel framework for online scene flow estimation from distance field that enables \emph{6DoF dynamic object pose estimation}, \emph{motion tracking}, and \emph{surface reconstruction}. The scene is represented by Gaussian Process Implicit Surfaces (GPIS), with surface normals serving as derivative constraints, enabling accurate signed distance computations near the surface and gradient queries with uncertainty. With this representation as a foundation, we compute a scene flow from the distance field that describes how surface points are transported over time in consecutive frames. Through our flow, we can estimate an object's pose and motion by incrementally registering a new observed point cloud via an elegant closed-form optimisation. Unlike prior methods that operate in the camera or world frame, our approach performs probabilistic fusion directly in the \emph{object frame}, where the object remains geometrically consistent over time. The tight coupling of the DisFlow method in space and time yields dense geometry, surface normals, object pose trajectories, velocities, and uncertainty, all at real-time rates. We evaluate DisFlow on dynamic object sequences and demonstrate that it achieves accurate pose and motion tracking while simultaneously reconstructing high-quality object surfaces. Code publicly available at \href{this https URL}{this https URL\_ros2}
>
---
#### [new 100] VLAMotor: Test-Guided Enhancement of Vision-Language-Action Models via Agent-BasedData Synthesis
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型增强任务，旨在解决模型在边缘场景下的失败问题。通过测试引导的数据合成，提升模型的泛化能力和成功率。**

- **链接: [https://arxiv.org/pdf/2606.00053](https://arxiv.org/pdf/2606.00053)**

> **作者:** Zeqin Liao; Peifan Ren; Zixu Gao; Hongyu Gong; Lianyu Hu; Wenbing Tang; Yuhong Nan; Zibin Zheng; Yang Liu
>
> **摘要:** Vision-Language-Action (VLA) models follow a data-driven paradigm and are constrained by the coverage of training data, making them prone to failure on edge-case configurations after deployment. To mitigate such risks, it is essential to expose high-quality failure modes and convert the resulting failures into supervisory data for model enhancement. Existing studies largely stop at failure detection and lack a mechanism for leveraging discovered failures for model repair. We propose VLAMotor, the first analysis framework for VLA enhancement, which integrates distance-aware model testing for failure exposure and agent-based data synthesis for model finetunning. First, VLAMotor estimates input uncertainty based on the distance to training samples, and combines uncertainty ranking with redundancy elimination to build compact test sets that expose diverse failures. Then, VLAMotor abstracts failure trajectories into structured semantic representations, and plans parameterized repair-skill sequences, which are then realized as executable trajectories through inverse kinematics and motion execution. The resulting successful trajectories are automatically labeled and used to fine-tune the original VLA model, yielding an enhanced VLA model. Evaluation on four representative robotic manipulation tasks shows that 92.33% of the in-simulation test cases generated by VLAMotor trigger VLA failures, and VLAMotor improves test coverage over the state-of-the-art tool by 18.93%. By fine-tuning VLA models with synthetic data derived from failed test cases, VLAMotor further enhances the overall success rate of VLA models by 49.25%. When deployed on real hardware, the simulation-enhanced models improve the success rate over the original VLA models by 57.50%, demonstrating an effective and low-cost direction for VLA enhancement.
>
---
#### [new 101] SKIP: Sparse Keyframe Interpolation Paradigm for Efficient Embodied World Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SKIP框架，解决机器人世界模型中长序列生成效率低的问题。通过关键帧插值，提升生成速度并保持任务事件完整性。属于机器人视觉与控制任务。**

- **链接: [https://arxiv.org/pdf/2606.00664](https://arxiv.org/pdf/2606.00664)**

> **作者:** Ziheng He; Yixiang Chen; Ning Yang; Zhanqian Wu; Qisen Ma; Yuan Xu; Jiabing Yang; Peiyan Li; Xiangnan Wu; Xiaofeng Wang; Zheng Zhu; Jing Liu; Nianfeng Liu; Yan Huang
>
> **备注:** 25 pages, 10 figures
>
> **摘要:** Embodied world models have emerged as a promising paradigm in robotics by predicting how robot actions affect the surrounding scene. However, the rollout inference remains computationally expensive in pixel space, as long-horizon manipulation videos typically have to be generated frame by frame. This cost cannot be easily reduced by indiscriminately dropping frames, since downstream policies rely on complete preservation of sparse task-relevant events such as approach, contact, grasp, and release. To address this challenge, we propose Sparse Keyframe Interpolation Paradigm (SKIP), an event-preserving sparse-to-dense framework that avoids dense frame-by-frame generation. SKIP first identifies task-relevant keyframes by leveraging robot-aware multimodal features. It then synthesizes only these keyframes with a sparse video diffusion model. A learned gap predictor and an action-conditioned interpolator subsequently reconstruct the missing intervals according to the robot actions. On LIBERO, SKIP generates dense rollouts $4.16\times$ faster than a dense baseline while improving visual fidelity and reducing aggregate FVD by $89.0\%$. Importantly, SKIP-generated videos are effective policy-training data. Even when they fully replace real demonstrations, $\pi_{0.5}$ success drops only $1.3$ pp in LIBERO simulation and $6.7$ pp on the real robot, whereas fully dense frame-by-frame generation collapses by $48$ to $58$ pp.
>
---
#### [new 102] Intercepting the Future: Latent-Space Predictive World Model for Dynamic VLA Manipulation
- **分类: cs.RO**

- **简介: 该论文属于动态物体操作任务，解决VLA模型在物体运动时的延迟问题。通过引入AHEAD模型，预测未来状态并调整动作，提升动态场景下的操作成功率。**

- **链接: [https://arxiv.org/pdf/2606.02486](https://arxiv.org/pdf/2606.02486)**

> **作者:** Shahram Najam Syed; Arthur Jakobsson; Haoran Hao; Jeffrey Ichnowski
>
> **备注:** 28 pages, 7 figures, 16 tables, Su
>
> **摘要:** Vision-Language-Action (VLA) models generalize across static manipulation but fail when objects move during task execution. They map the current observation to an action and assume the scene is stationary between observation and execution, so at any non-trivial object speed the resulting latency exceeds the time available to grasp. We close this gap with AHEAD (Anticipatory Horizon Extrapolation with Adaptive Dynamics), a predict-then-act wrapper that augments a frozen VLA with a motion-aware latent world model. A small world model trained on manipulation video forecasts future patch tokens in the VLA's feature space, conditioned on per-token velocity and acceleration from optical flow. A language-and-motion saliency mask concentrates prediction on task-relevant patches, and the model rolls forward for an adaptive horizon, halting when prediction uncertainty crosses a threshold. The frozen action decoder then receives the predicted future tokens in place of the current ones. AHEAD adds 4.9M parameters to a frozen 7B OpenVLA and reaches 79 to 97% success across 20 dynamic simulation scenarios where the strongest baseline reaches 31 to 58%. On a physical UFactory xArm 7, AHEAD succeeds on 29/30 to 30/30 on three conveyor and rolling-ball tasks, 23/30 on paddle interception, and 19/30 on projectile catching where every baseline scores 0/30.
>
---
#### [new 103] STEM: Semantic Target Search and Exploration using MAVs in Cluttered Environments
- **分类: cs.RO**

- **简介: 该论文属于自主目标搜索任务，解决复杂3D环境中 MAV 的高效目标定位问题。提出语义引导的视角规划框架，结合语义优先级与LLM相似度评分，提升搜索效率。**

- **链接: [https://arxiv.org/pdf/2606.00762](https://arxiv.org/pdf/2606.00762)**

> **作者:** Nikhil Sethi; Max Lodel; Laura Ferranti; Robert Babuška; Javier Alonso-Mora
>
> **备注:** Accepted to Autonomous Robots Journal. Nikhil Sethi and Max Lodel contributed equally
>
> **摘要:** Autonomous target search is crucial for deploying Micro Aerial Vehicles (MAVs) in emergency response and rescue missions. Existing approaches either focus on 2D semantic navigation in structured environments -- which is less effective in complex 3D settings, or on robotic exploration in cluttered spaces -- which often lacks the semantic reasoning needed for efficient target search. This paper overcomes these limitations by proposing a novel framework that utilizes a semantically-guided viewpoint planner to minimize target search and exploration time in unstructured 3D environments using an MAV. Specifically, we develop a combinatorial planner that generates efficient semantic exploration plans by prioritizing viewpoints that likely lead to the target. To guide the planner towards the target, an active perception pipeline is developed that propagates semantic priorities of observed objects into neighboring frontier voxels for computing semantic information gains of frontier viewpoints. In addition, we demonstrate how LLM-based similarity scores can be leveraged as semantic priority input to our pipeline. Evaluations in two distinct simulation environments show that the proposed method consistently outperforms baselines by quickly finding the target while maintaining reasonable exploration times. Real-world experiments with an MAV further demonstrate the method's ability to handle practical constraints like limited battery life, small sensor range, and semantic uncertainty.
>
---
#### [new 104] From Cues to Horizons: Dynamic Risk Horizon Profiling for Trajectory Prediction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于轨迹预测任务，旨在解决传统方法忽视风险未来变化的问题。通过引入风险时间剖面模块，提升预测准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2606.00857](https://arxiv.org/pdf/2606.00857)**

> **作者:** Xinyi Ning; Zilin Bian; Dachuan Zuo; Semiha Ergan; Kaan Ozbay
>
> **备注:** 11 pages, 7 figures, submitted to IEEE Transactions on Intelligent Transportation Systems (T-ITS)
>
> **摘要:** Accurate and reliable vehicle trajectory prediction is essential for safe autonomous driving. Recent studies have incorporated safety risk into trajectory prediction to quantify dangers posed by surrounding agents. However, most risk-aware approaches use past risk information as a secondary signal to help guide decisions, overlooking its future evolution and uncertainty. In this paper, we propose a risk horizon profiling (RHP) module that incorporates a continuous, learnable potential field model for risk-aware trajectory prediction. The RHP module calculates the spatial-temporal proximity of surrounding objects to profile risk distributions across future horizons, which supports better trajectory prediction by adaptively identifying what human drivers perceive as critical moments. We evaluate our method on two datasets from different driving settings, highD for highway corridors and SHRP2 for urban streets, which cover diverse risk scenarios including safe, near-crash, and crash events. Compared to the baseline methods, our framework achieves a 25.0\% reduction in 5s RMSE on the highD dataset and a 29.1\% reduction in 5s minFDE on SHRP2. These results indicate strong performance for both short and long horizon prediction and robust generalization across highway and urban scenarios. The proposed method enables more realistic AV path planning and strategic selection, thereby supporting safer autonomous driving and more advanced driver-assistance systems. The source code for this work is available at: this https URL
>
---
#### [new 105] Not All Points Are Equal: Uncertainty-Aware 4D LiDAR Scene Synthesis
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于4D场景生成任务，旨在解决LiDAR数据中不同区域不确定性差异的问题。通过引入不确定性感知机制，提升场景生成的准确性和连贯性。**

- **链接: [https://arxiv.org/pdf/2606.02510](https://arxiv.org/pdf/2606.02510)**

> **作者:** Xiang Xu; Alan Liang; Youquan Liu; Xian Sun; Linfeng Li; Lingdong Kong; Ziwei Liu; Qingshan Liu
>
> **备注:** CVPR 2026 E2E3D Workshop; GitHub at this https URL
>
> **摘要:** Constructing faithful 4D worlds from LiDAR-acquired sequences is crucial for embodied AI, yet current generative frameworks apply uniform modeling capacity across all spatial regions. This ignores that perceptual difficulty varies dramatically within a single scan: distant surfaces, occluded boundaries, and small-scale objects carry far higher uncertainty than well-observed structures. We present U4D, a new framework that explicitly leverages spatial uncertainty to guide LiDAR scene generation in a "hard-to-easy" schedule. U4D derives per-point uncertainty maps via Shannon Entropy from a pretrained segmentor, then applies an unconditional diffusion stage to synthesize high-entropy areas with precise geometry, followed by a conditional completion stage that fills in the remaining regions using these structures as priors. A MoST (Mixture of Spatio-Temporal) block further maintains cross-frame coherence by dynamically balancing spatial detail and temporal continuity. Extensive experiments on nuScenes and SemanticKITTI demonstrate state-of-the-art scene fidelity, temporal consistency, and downstream performance.
>
---
#### [new 106] A Machine-to-Machine Knowledge-Guided LLM Agent for Generalizable Radiotherapy Treatment Planning
- **分类: physics.med-ph; cs.RO**

- **简介: 该论文属于放射治疗计划自动化任务，旨在解决传统方法依赖人工且效率低的问题。通过结合深度强化学习与大语言模型，实现自主、高效、通用的治疗计划制定。**

- **链接: [https://arxiv.org/pdf/2606.00922](https://arxiv.org/pdf/2606.00922)**

> **作者:** Md Mainul Abrar; Xun Jia; Yujie Chi
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** In this work, we propose a prototype machine-to-machine (M2M) knowledge-guided Large Language Model (LLM) framework for automated radiotherapy treatment planning. In the proposed paradigm, Treatment Planning Parameter (TPP) distribution knowledge discovered by a Deep Reinforcement Learning (DRL) agent is transferred to an LLM agent through in-context learning, enabling autonomous iterative planning without human intervention. While standard LLM-based planning often lacks physical intuition and struggles with convergence, the integration of DRL-derived guidance constrains the agent to a physically valid parameter space. Experimental evaluations are performed across three diverse planning scenarios: basic prostate cases, complex prostate configurations with increased organ-at-risk (OAR) constraints, and liver cases. The evaluation results demonstrate that the guided LLM agent consistently achieves optimal planning scores while significantly reducing the number of iterations compared to unguided planning. Analysis of the final TPP configurations reveals that the agent successfully learns a hierarchical priority of objectives, effectively restoring a logical "cause-and-effect" relationship between parameter tuning and dosimetric outcomes. Crucially, this prototype framework exhibits robust generalizability, maintaining high planning quality regardless of specific patient anatomy, treatment site, or initial plan quality. By bridging the specialized optimization of DRL with the adaptive reasoning of LLMs, this M2M framework establishes a scalable foundation towards generalizable autonomous treatment planning, ultimately benefiting clinical practice in realistic environments.
>
---
#### [new 107] Coordinating Task Switching in a Robotics Multi-Agent System Using Behavior Trees
- **分类: cs.MA; cs.RO**

- **简介: 论文研究多机器人系统的任务切换协调问题，针对VSSS机器人足球比赛中的团队协作提出基于行为树的解决方案，并与传统有限状态机方法进行比较。**

- **链接: [https://arxiv.org/pdf/2606.01170](https://arxiv.org/pdf/2606.01170)**

> **作者:** Lucas Haug; Anarosa Alves Franco Brandão; Arthur Casals
>
> **备注:** 7 pages, 7 figures. Preprint of a manuscript submitted to the XXVI Congresso Brasileiro de Automática (CBA 2026)
>
> **摘要:** The application of multi-agent systems in robotics is a very challenging field. Several competitions involving such systems are proposed to foster research and development of strategies and mechanisms using games as the underlying domain. Among them are the ones from the \textit{IEEE Very Small Soccer (VSSS)} category, which is the case study described in this paper. In VSSS, two teams of three robots each compete in a very dynamic environment of a soccer game. Thus, coordination of robots' behavior during the game is crucial to win it. In this paper, we present a Behavior-Tree-based approach to support multi-robot coordination within the VSSS team of the ThundeRatz robotics team from the Universidade de S$\tilde{a}$o Paulo. Moreover, a comparison between the proposed approach and the previous one, which was based on a Finite State Machine (FSM), was conducted using the FIRASim simulator. Besides that, the performance of this new strategy was further evaluated in an academic robotics competition.
>
---
#### [new 108] Edge-Based QoS-Aware Adaptive Task Placement: A Closed-Loop Control in Multi-Robot Systems
- **分类: cs.OS; cs.DC; cs.NI; cs.RO; eess.SY**

- **简介: 该论文研究多机器人系统中的任务调度问题，旨在解决边缘计算中的QoS保障。通过设计自适应任务放置控制器，优化任务分配以降低延迟和资源争用。**

- **链接: [https://arxiv.org/pdf/2606.00552](https://arxiv.org/pdf/2606.00552)**

> **作者:** Thien Tran; Jonathan Kua; Thuong Hoang; Minh Tran; Honghao Lyu; Jiong Jin
>
> **备注:** 6 pages, 2 figure, 1 algorithm, accepted as a regular paper on the 24th IEEE International Conference on Industrial Informatics (INDIN), 26-29 July, 2026, Melbourne, Australia
>
> **摘要:** Multi-robot systems (MRS) increasingly offload compute-intensive perception tasks to edge nodes to meet strict time-sensitive Quality-of-Service (QoS) constraints. However, static task orchestration on a shared edge node can severely degrade QoS due to network latency, jitter, and edge-resource contention. We present a pilot edge-centric MRS testbed using Raspberry Pi nodes to evaluate a camera-to-manipulator pipeline under three modes: local execution, static offloading, and a QoS-aware Adaptive Task Placement (ATP) controller. ATP scores candidate placements using a multi-metric cost (normalized latency, CPU utilization, and switching overhead) over two-second control windows. The closed-loop visual servoing testbed is instrumented with sub-millisecond clock synchronization, network emulation, and detailed monitoring of multiple metrics across nodes to capture realistic jitter. Experimental results under compute-stress and network-fault scenarios show that static edge offloading reduces on-board CPU load but amplifies tail latency and deadline misses. In contrast, the QoS-aware ATP controller, by switching task placement based on measured latency and utilization thresholds, consistently lowers deadline violations and tail latency. Overall, the results position ATP as a practical edge-side control primitive for MRS and concrete design guidelines for Cloud-Edge Robotics deployments within the broader cloud-fog automation, while motivating QoS-aware multi-objective workload orchestration for industrial cyber-physical systems.
>
---
#### [new 109] From Demonstrations to Rewards: Test-Time Prompt Optimization for VLM Reward Models
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决奖励函数设计困难的问题。通过少量示范优化视觉-语言模型的奖励函数，提升政策学习效果。**

- **链接: [https://arxiv.org/pdf/2606.00083](https://arxiv.org/pdf/2606.00083)**

> **作者:** Christian Gumbsch; Leonardo Barcellona; Lennard Schünemann; Platon Karageorgis; Andrii Zadaianchuk; Zehao Wang; Sergey Zakharov; Fabien Despinoy; Rahaf Aljundi; Efstratios Gavves
>
> **摘要:** Reinforcement learning relies on accurate reward functions, which are often hand-crafted or even unavailable in real-world applications, such as robotics. Recent work has explored the zero-shot reasoning capabilities of pre-trained Vision-Language Models (VLMs) as reward models. However, without careful prompt engineering, these approaches tend to produce suboptimal rewards, where false positive predictions can severely degrade downstream policy learning. In robotics, limited datasets comprising expert demonstrations are often collected to bootstrap policy learning. This scenario provides an opportunity to optimize a reward model prior policy training. We propose Demo2Reward a test-time adaptation technique to optimize the language instruction of a reward model based on a few demonstrations (3-10 trajectories) to reduce false positives while preserving true positives. Crucially, this requires no additional model training or computation resources during policy learning. We show that Demo2Reward consistently outperforms existing zero- and few-shot VLM reward models across a range of simulated robotic tasks and policy backbones. Finally, we demonstrate that Demo2Reward effectively transfers to a real-world robotic learning scenario, enabling policy learning without manually engineering a reward function.
>
---
#### [new 110] StressDream: Steering Video World Models for Robust Policy Evaluation and Improvement
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出StressDream，用于增强视频世界模型的策略评估与改进。任务是提升策略在高影响但合理结果上的鲁棒性，通过优化噪声引导生成更可靠的未来预测。**

- **链接: [https://arxiv.org/pdf/2606.00267](https://arxiv.org/pdf/2606.00267)**

> **作者:** Junwon Seo; Sushant Veer; Ran Tian; Wenhao Ding; Apoorva Sharma; Karen Leung; Edward Schmerling; Marco Pavone; Andrea Bajcsy
>
> **备注:** Project page: this https URL
>
> **摘要:** Video world models (WMs) have shown promise for policy evaluation and improvement by imagining realistic future observations conditioned on ego-robot actions. While WMs can model distributions over futures, policy evaluation and improvement typically rely on nominal imaginations, which can miss high-impact outcomes of robot actions unless prohibitively many samples are drawn. To enable robust policy evaluation and improvement over WM imaginations, we propose StressDream, which steers imaginations toward high-impact yet plausible outcomes specified at inference time by optimizing the initial noise of diffusion-based WMs. However, optimizing high-dimensional noise is challenging: the optimization must reason about nuanced, scene-dependent target events in generated videos while avoiding out-of-distribution (OOD) noise that yields implausible imaginations. We address this with two complementary objectives: a semantic objective with a Vision-Language Model that provides informative gradients by reasoning about the generated video, and a plausibility objective that prevents the optimized noise from drifting OOD. With state-of-the-art video world models for autonomous driving and robotic manipulation, we show that StressDream effectively steers imaginations toward high-impact yet plausible outcomes specified by text at inference time, such as task failures, enabling robust policy evaluation and improvement by identifying actions whose plausible futures include undesirable outcomes. Video results are available at this https URL.
>
---
#### [new 111] A Four-Tier Communication Architecture and Sim-to-Real Validation of a Graphical Open-Source Platform for Robotic Engineering Education
- **分类: cs.HC; cs.ET; cs.RO**

- **简介: 该论文属于机器人教育任务，旨在解决虚拟与物理机器人教学脱节的问题。提出四层通信架构，融合3D建模与ROS，实现仿真到现实的可靠传输。**

- **链接: [https://arxiv.org/pdf/2606.00550](https://arxiv.org/pdf/2606.00550)**

> **作者:** Thien Tran; Khang Duong; Minh Tran; Jonathan Kua; Thuong Hoang; Jiong Jin
>
> **备注:** 4 pages, 4 figures, accepted as a Work-in-Progress (WiP) paper, on the 24th IEEE International Conference on Industrial Informatics (INDIN), 26-29 July, 2026, Melbourne, Australia
>
> **摘要:** The persistent challenge in scaling authentic manipulator education within university laboratories is a structural dichotomy: commercial digital twins are often cost-prohibitive and rigidly scripted, whereas open-source robotics middleware (ROS) imposes steep technical and syntax barriers for novices. To resolve this logistical and educational friction, this Work-in-Progress (WiP) paper proposes a scalable four-tier communication architecture tailored for sustainable robotic curricula. Rather than focusing on software application design, our study examines the underlying data exchange mechanisms required to bridge visual conceptual environments with physical robotic endpoints, utilizing the Graphical Open-Source Platform (GOSP) as a foundational instantiation. This WiP details the framework's technical integration of 3D visual armature modeling with a robust ROS middleware backend, emphasizing the serialization, routing, and encapsulation of intricate communication routines. Preliminary sim-to-real validation using multi-axis spatial trajectories confirms that encapsulating these communication pipelines provides a sufficient fidelity hardware-agnostic pathway. By bridging virtual design and physical execution, this architectural blueprint offers a viable infrastructure for engineering education.
>
---
#### [new 112] FlatVPR: Plug-and-play Geo-linear Residual Adapter for Geometric Rectification of Foundation Model Feature Manifolds
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出FlatVPR，解决视觉定位中地图轻量化与定位精度的平衡问题，通过几何校正提升特征流形的线性性。**

- **链接: [https://arxiv.org/pdf/2606.01734](https://arxiv.org/pdf/2606.01734)**

> **作者:** Rai Hisada; Kanji Tanaka
>
> **备注:** 5 pages, 1 figure, technical report
>
> **摘要:** This paper proposes ``FlatVPR,'' a novel geometric rectification paradigm that effectively bridges the trade-off between map lightweightness and localization accuracy in visual place recognition (VPR) by enforcing a feature manifold structure where any descriptor between two adjacent anchors $\mathbf{z}_A$ and $\mathbf{z}_B$ can be accurately reconstructed via linear interpolation $\hat{\mathbf{z}}_{pseudo} = (1-t)\mathbf{z}_A + t\mathbf{z}_B$, where $t \in [0,1]$ denotes the relative position. While state-of-the-art foundation models such as DINOv2-ViT-S/14 provide robust semantic features, their latent manifolds exhibit prominent curvature, projecting uniform linear motion in physical space onto highly non-linear trajectories in the feature space, which hinders reliable reconstruction under sparse anchor conditions. To enable the aforementioned interpolation-based reconstruction, we introduce a residual transformation $\hat{\mathbf{z}} = \mathbf{z} + \text{Res}(\mathbf{z})$ to the raw foundation features $\mathbf{z}$, where $\text{Res}(\cdot)$ represents a learnable adapter. Our method explicitly suppresses manifold curvature using a mathematically grounded Pullback Flatness Loss that minimizes the deviation of intermediate features from the linear segment connecting adjacent anchors, thereby minimizing the intrinsic curvature of the manifold. Through this spatial flattening, map construction is formulated within an Expectation-Maximization (EM) framework, decoupled into a continuous M-step for manifold adaptation and a conceptual E-step for optimal anchor selection guidelines. Experiments on the NCLT dataset demonstrate that the application of our adapter leads to significant performance improvements even under extremely sparse anchor conditions with 100m intervals and extreme seasonal changes.
>
---
#### [new 113] Time-Optimal Collision Avoidance Via a Greedy Polynomial Backward Sweep
- **分类: math.OC; cs.RO**

- **简介: 该论文属于航天器碰撞规避任务，解决低推力卫星如何确定最晚机动时间的问题。提出一种贪心时间最优方法，通过逆向传播选择最优推力方向，确保安全并提高计算效率。**

- **链接: [https://arxiv.org/pdf/2606.01169](https://arxiv.org/pdf/2606.01169)**

> **作者:** Zeno Pavanello; Frank De Veld; Roberto Armellin
>
> **摘要:** Spacecraft collision avoidance for low-thrust satellites often requires determining not only how to maneuver, but also how late a maneuver can begin while still ensuring safety. This paper presents a greedy time-optimal (GTO) backward-sweep method to find the latest maneuver initiation time. The method starts from the nominal time of closest approach and iteratively propagates the maneuver backward in time, selecting at each step the thrust direction that locally minimizes the chosen danger metric. Differential algebra is used to efficiently propagate state sensitivities and update the time of closest approach online. The method is tested on a large dataset of conjunctions, using both miss distance and probability of collision as safety metrics. The approach achieves accurate results and only a small loss of optimality relative to an optimal-control benchmark, while retaining runtimes suitable for on-board implementation.
>
---
#### [new 114] Global Convergence of a Line-Search Filter Differential Dynamic Programming Method
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文研究最优控制问题，解决非线性约束下的全局收敛性问题。提出FilterDDP算法，通过后向-前向过程实现全局收敛。**

- **链接: [https://arxiv.org/pdf/2606.01487](https://arxiv.org/pdf/2606.01487)**

> **作者:** Ming Xu; Iman Shames
>
> **摘要:** In this article, we establish the global convergence properties of the FilterDDP algorithm, which extends the discrete-time differential dynamic programming (DDP) algorithm of Mayne and Jacobson [\emph{International Journal of Control}, 3, (1966), pp. 85-95] to handle nonlinear constraints over states and controls, in addition to the dynamics. FilterDDP adopts a line-search filter procedure for step acceptance. However, instead of a damped Newton step applied in the general nonlinear programming setting, the computation of a trial point involves applying a backward recursion and a forward simulation. We establish the global convergence of FilterDDP by showing that for a subset of constrained optimal control problems, the this backward-forward procedure satisfies the same properties as a Newton step for the purpose of establishing global convergence of a line-search filter method, following the analysis of Wächter and Biegler [\emph{SIAM Journal on Optimization}, 16 (2005), pp. 1-31].
>
---
#### [new 115] Predicted-Flow Control Barrier Functions for Real-Time Safe Optimal Control
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于安全最优控制任务，解决传统CBF方法的局限性，提出P-CBF方法以实现更可靠的实时安全控制。**

- **链接: [https://arxiv.org/pdf/2606.00297](https://arxiv.org/pdf/2606.00297)**

> **作者:** Amirsaeid Safari; Jesse B. Hoagg
>
> **摘要:** Control barrier functions (CBFs) provide real-time safety guarantees through pointwise conditions on the state. However, synthesizing a valid CBF is difficult and the resulting controllers are myopic. To address myopia, this article introduces predicted-flow control barrier functions (P-CBFs), which generalize the CBF from a function of the current state to a functional of a predicted flow under a parametrized control plan over a finite prediction horizon. For safety, a P-CBF can certify that the predicted flow is in a safe set over the entire prediction horizon. However, candidate P-CBFs suffer from the same challenge as candidate CBFs, namely, control constraints make it difficult to guarantee that the P-CBF is valid. This article resolves this challenge by introducing a terminal candidate P-CBF requiring that the predicted flow end in a backup safe set at the terminal time, and a planning-time shift that modulates the prediction horizon, providing an additional degree of freedom to ensure feasibility. The real-time control and the evolution of the control-plan parameter and planning-time shift are determined jointly by a single convex optimization that is guaranteed to be feasible and renders the associated safe set forward invariant. The resulting safe optimal flow control provides a safety certificate over the entire prediction horizon and unifies finite-horizon integral-cost optimization with safety certification. This optimization reduces to a quadratic program (QP) if the control constraints are a convex polytope. The QP implementation, termed FlowBarrier, is validated on a nonholonomic ground robot navigating a dense environment. FlowBarrier is compared to nonlinear model predictive control and two CBF-based safety filter methods across 100 trials, where FlowBarrier achieves the highest goal-reaching rate, zero safety violations, and the lowest computation time.
>
---
#### [new 116] Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决VLM在3D空间推理上的不足。提出HSGM结构化地图，融合语义与几何信息，提升导航可靠性。**

- **链接: [https://arxiv.org/pdf/2606.00095](https://arxiv.org/pdf/2606.00095)**

> **作者:** Kailing Li; Tianwen Qian; Lijin Yang; Yuqian Fu; Jingyu Gong; Xiaoling Wang; Liang He
>
> **摘要:** Vision-Language Navigation (VLN) enables embodied agents to reach target locations in unseen environments by following language instructions. Despite recent progress with vision-language models (VLMs), a critical semantic-geometric gap remains: while VLMs excel at language and 2D visual understanding, they struggle with 3D spatial reasoning and fail to capture the causal dynamics between actions and spatial transitions, resulting in unreliable navigation, particularly in zero-shot settings. To bridge this gap, we propose a Hierarchical Semantic-Geometric Map (HSGM) that transforms 3D geometric information into a structured representation compatible with VLMs, effectively linking them to the physical world. Specifically, HSGM is represented as a multi-channel top-down map organized into three levels: (1) geometric level that records navigable regions and obstacles, (2) semantic level that represents objects and their relations, and (3) decision level that supports high-level task reasoning and goal selection. During navigation, the VLM acts as a high-level semantic planner, interpreting the spatial layout encoded in the HSGM to select geometrically valid waypoints, while low-level, collision-free movements between waypoints are executed by a classical path-planning algorithm, fully decoupling semantic reasoning from action execution. Additionally, complex instructions are decomposed into subtasks to alleviate the problem of progress forgetting or hallucinating in long-horizon navigation. Extensive experiments on R2R-CE and RxR-CE benchmarks demonstrate that our zero-shot framework achieves state-of-the-art performance and even outperforms several supervised methods. Code is available at this https URL.
>
---
#### [new 117] RoboTrustBench: Benchmarking the Trustworthiness of Video World Models for Robotic Manipulation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于机器人视觉建模任务，旨在评估视频世界模型的可信度。针对现有基准不足，提出RoboTrustBench，涵盖四种场景，发现模型在约束推理等方面存在缺陷。**

- **链接: [https://arxiv.org/pdf/2606.01600](https://arxiv.org/pdf/2606.01600)**

> **作者:** Huiqiong Li; Jiayu Wang; Zhiting Mei; Anirudha Majumdar; Jingjing Chen; Bin Zhu
>
> **备注:** Project: this https URL
>
> **摘要:** Video world models are increasingly used in robotic manipulation, yet existing benchmarks mostly evaluate them under valid, feasible, and safe instructions. We introduce RoboTrustBench, a benchmark for evaluating the trustworthiness of video world models under four scenarios: Normal, Constraint-Sensitive, Counterfactual, and Adversarial. Built from real-world DROID episodes, RoboTrustBench contains 1,207 expert-validated instruction-image pairs and a six-dimensional evaluation protocol with 13 fine-grained criteria. Evaluating seven representative video world models with human and MLLM assessment, we find that current models often generate visually coherent videos, but struggle with constraint reasoning, counterfactual grounding, physical interaction, and unsafe-instruction suppression. These results show that visual quality and surface-level instruction following are insufficient for trustworthy robotic video world modeling.
>
---
#### [new 118] Goal2Pixel: Grounding Goals to Pixels for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决传统方法依赖低级动作预测效率低的问题。提出Goal2Pixel，通过像素级导航实现更高效、长距离的路径规划。**

- **链接: [https://arxiv.org/pdf/2606.01621](https://arxiv.org/pdf/2606.01621)**

> **作者:** Muyi Bao; Yuxin Cai; Hang Xu; Zongtai Li; Jinxi He; Jingfan Tang; Chen Lv; Ji Zhang; Yaqi Xie; Wenshan Wang
>
> **备注:** 8 pages
>
> **摘要:** Vision-language models (VLMs) have become a common foundation for vision-and-language navigation in continuous environments (VLN-CE). Yet most VLM-based methods cast navigation as low-level action prediction, an interface that is ambiguous, tied to short-horizon motion primitives, and inefficient due to repeated VLM querying. We propose Goal2Pixel, a pure pixel-based paradigm that reformulates VLN-CE as navigable pixel grounding. Rather than predicting actions, Goal2Pixel uses the image plane as a unified spatial interface between VLM reasoning and robot motion: the model predicts a visible navigable pixel to the agent, which is back-projected into a 3D waypoint for forward navigation. For non-forward actions, we append auxiliary directive regions to the image plane, where the left/right/bottom regions are interpreted as turning left, turning right, and stopping, respectively. To enable long-horizon navigation, we propose a visibility-aware keyframe memory for compact and informative history representation. To adapt pretrained VLMs to navigable pixel grounding, we introduce semantic embeddings and coordinate-aware auxiliary losses. Goal2Pixel achieves competitive state-of-the-art performance while requiring fewer VLM inference calls than prior methods. On R2R-CE Val-Unseen it achieves 54.1% SR and 52.5% SPL with just 7.75 VLM calls per episode, 6x fewer than the 46.62 required by direct action prediction at 32.9% SR. The same trend holds on this http URL Page: this https URL.
>
---
#### [new 119] General Covariant Action Modeling: Constructing Generalized Manifolds via Spatio-Temporal Decoupling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于 embodied intelligence 领域，解决有限数据下泛化能力不足的问题。提出 GAM 框架，通过时空解耦实现动作流形构建，提升迁移与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.00110](https://arxiv.org/pdf/2606.00110)**

> **作者:** Huaihai Lyu; Chaofan Chen; Mingyu Cao; Yuheng Ji; Changsheng Xu
>
> **摘要:** Achieving robust generalization from limited data is a central challenge in embodied intelligence. Prevailing methods fail by regressing absolute coordinates, which violates the principle of general covariance. Fundamentally, this conflates the intrinsic task geometry with rigid execution patterns, binding policies to specific motion styles and fixed speeds. To resolve this, we propose the Generalized Action Manifold (GAM) framework that enforces general covariance through structural disentanglement. Specifically, GAM realizes the manifold by enforcing invariance across two orthogonal dimensions: (1) Temporal Invariance, utilizing an Arc-Length Parameterizer to orthogonalize the spatial path geometry from temporal dynamics, ensuring robustness to velocity variations; (2) Geometric Invariance, where a Schema-Affine-Factorization mechanism maps trajectories to canonical ``world lines'' in a pose-normalized coordinate frame. This distinguishes invariant geometric schemas from affine modulations, ensuring spatial generalizability. By integrating GAM within a structured Vision-Language-Action (VLA) architecture, we enable sparse demonstrations to densely populate a continuous, valid action manifold. Empirical results demonstrate that GAM enables superior transfer and robustness capabilities, outperforming geometry-agnostic baselines.
>
---
#### [new 120] TIDES: Time-Derivative Event Simulation via Deformable Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出TIDES，一种基于3D场景的连续时间事件模拟器，解决真实事件数据稀缺问题，通过精确预测阈值交叉提升事件流质量。**

- **链接: [https://arxiv.org/pdf/2606.02058](https://arxiv.org/pdf/2606.02058)**

> **作者:** Christopher Thirgood; Dipon Kumar Ghosh; Simon Hadfield
>
> **摘要:** Event cameras emit asynchronous events in response to environmental appearance changes. The scarcity of real-world event datasets makes simulation essential. However, most simulators infer event timestamps from frame sequences, forcing many threshold crossings to share a small set of discrete times; a failure mode we term timestamp batching that worsens under fast motion and occlusion. We present TIDES, a continuous-time event simulator built on dynamic Gaussian splatting. Because TIDES operates on an explicit 3D scene representation with learnt geometry and motion, it can derive per-pixel intensity dynamics directly from the scene, rather than by differencing rendered frames. This enables accurate threshold-crossing prediction, including multiple crossings per rendering step, without temporal upsampling or frame interpolation. The same 3D scene model reveals where objects partially occlude one another; TIDES uses this to guide adaptive time stepping, concentrating computation only in regions where occlusion dynamics make simple models of brightness change unreliable. Finally, we model finite sensor bandwidth using a tile-level arbiter whose throughput, jitter, and event drops reproduce realistic sensor artifacts. Across paired RGB-event benchmarks, TIDES attains state-of-the-art event-stream fidelity. We also show that events simulated by TIDES transfer more effectively to real downstream tasks than competitors'.
>
---
#### [new 121] GABI: Geometry-Aware Boundary Integration for Spacecraft Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于航天器分割任务，旨在解决空间环境下图像多样性导致的分割泛化问题。提出GABI架构，结合边界感知与距离场监督，提升分割精度与效率。**

- **链接: [https://arxiv.org/pdf/2606.00886](https://arxiv.org/pdf/2606.00886)**

> **作者:** Iason Georgios Velentzas; Dhruv Ahuja; Panagiotis Tsiotras
>
> **备注:** Accepted to AI4Space at CVPR 2026
>
> **摘要:** Accurate segmentation is crucial for autonomous spacecraft, as it directly affects downstream tasks related to 3D situational awareness. The harsh illumination conditions of space, however, produce images with high variability in appearance, hindering the generalization of segmentation approaches across different spacecraft and environments. In this work, we propose GABI, a lightweight boundary-aware multi-task segmentation architecture that augments a convolutional backbone with an auxiliary distance-field prediction head. The distance field provides dense geometric supervision around object boundaries, encouraging the network to learn spatially consistent representations of spacecraft structures while maintaining low model complexity suitable for onboard perception systems. We evaluated GABI against both an established convolutional baseline and a heavier transformer-based architecture. On the SPARK benchmark, distance-field supervision improves the baseline by up to $5\%$ in Average Precision while achieving performance comparable to the transformer models. In generalization experiments, GABI improves Average Precision by more than $50\%$ over the baseline. In cross-domain evaluation, the lightweight GABI variant performs within $5\%$ in IoU and F1-score of the heavier transformer model while being approximately ten times smaller. At the same time, the heavier GABI variant surpasses the transformer architectures while remaining nearly three times lighter.
>
---
## 更新

#### [replaced 001] Update-Free On-Policy Steering via Verifiers
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在解决行为克隆策略脆弱、精度不足的问题。通过训练验证器函数，无需更新策略参数即可提升执行成功率。**

- **链接: [https://arxiv.org/pdf/2603.10282](https://arxiv.org/pdf/2603.10282)**

> **作者:** Maria Attarian; Ian Vyse; Claas Voelcker; Jasper Gerigk; Evgenii Opryshko; Anas Almasri; Sumeet Singh; Yilun Du; Igor Gilitschenski
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** In recent years, Behavior Cloning (BC) has become one of the most prevalent methods for learning manipulation from human demonstrations. Despite their successes, BC policies are often brittle and struggle with precise manipulation. To overcome these issues, we propose UF-OPS, an Update-Free On-Policy Steering method that enables the robot to predict the success likelihood of its actions and adapt its strategy at execution time. We accomplish this by training verifier functions using policy rollout data obtained during an initial evaluation of the policy. These verifiers are subsequently used to steer the base policy toward actions with a higher likelihood of success. Our method improves the performance of black-box diffusion policies, without changing the base parameters, making it lightweight and flexible. We present results from both simulation and real-world data and achieve an average 49% improvement in success rate over the base policy across 5 real tasks.
>
---
#### [replaced 002] Highly Deformable Proprioceptive Membrane for Real-Time 3D Shape Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决复杂环境下3D形状重建问题。提出一种基于光学波导的柔性 proprioceptive 膜，通过自身形变实时恢复三维形状，具有高精度和抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2601.13574](https://arxiv.org/pdf/2601.13574)**

> **作者:** Guanyu Xu; Jiaqi Wang; Dezhong Tong; Xiaonan Huang
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Reconstructing the three-dimensional (3D) geometry of object surfaces is essential for robot perception, yet vision-based approaches degrade under low illumination or occlusion. This limitation motivates the design of a proprioceptive membrane that conforms to the surface of interest and infers 3D geometry by reconstructing its own deformation. Conventional deformation-aware membranes typically rely on resistive, capacitive, or magneto-sensitive mechanisms, but can suffer from structural complexity, limited compliance during large-scale deformation, and susceptibility to electromagnetic interference. This work presents a soft, flexible, and stretchable proprioceptive silicone membrane based on optical waveguide sensing. The membrane integrates edge-mounted LEDs and centrally-distributed photodiodes (PDs) within a multilayer elastomeric composite. Rich deformation-dependent light-intensity signals are decoded by a data-driven model to recover the membrane geometry. Real-time reconstruction is demonstrated on a customized 140 mm square membrane at an end-to-end update rate of 90 Hz, achieving an average reconstruction error of 1.307 mm for out-of-plane deformation of up to 25 mm. The proposed sensor also demonstrates accurate reconstruction under large in-plane deformation, achieving reliable shape recovery up to 75% strain with an average Chamfer distance of 1.214 mm. The proposed framework provides a scalable, robust, and low-profile solution for global shape perception in deformable robotic systems.
>
---
#### [replaced 003] Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文提出Qwen-VLA，一个统一的视觉-语言-动作模型，解决多任务、多环境、多机器人形态的具身智能问题，通过联合预训练和提示条件实现跨任务迁移。**

- **链接: [https://arxiv.org/pdf/2605.30280](https://arxiv.org/pdf/2605.30280)**

> **作者:** Qiuyue Wang; Mingsheng Li; Jian Guan; Jinhui Ye; Sicheng Xie; Yitao Liu; Junhao Chen; Zhixuan Liang; Jie Zhang; Xintong Hu; Xuhong Huang; Pei Lin; Junyang Lin; Dayiheng Liu; Shuai Bai; Jingren Zhou; Jiazhao Zhang; Haoqi Yuan; Gengze Zhou; Hang Yin; Ye Wang; Yiyang Huang; Zixing Lei; Wujian Peng; Delin Chen; Yingming Zheng; Jingyang Fan; Xianwei Zhuang; Xin Zhou; Haoyang Li; Anzhe Chen; Tong Zhang; Xuejing Liu; Yuchong Sun; Ruizhe Chen; Zhaohai Li; Chenxu Lü; Zhibo Yang; Tao Yu; Xionghui Chen
>
> **备注:** 34 pages
>
> **摘要:** Embodied intelligence is often studied through specialized models for individual tasks such as manipulation or navigation, resulting in fragmented capabilities and limited generalization across tasks, environments, and robot embodiments. In this work, we study whether heterogeneous embodied decision-making problems can be unified within a single vision-language-action model. We present Qwen-VLA, a unified embodied foundation model that extends Qwen's vision-language modeling stack from perception, understanding, and reasoning to continuous action and trajectory generation through a DiT-based action decoder. Qwen-VLA is trained with a large-scale joint pretraining recipe over diverse data sources, including robotics manipulation trajectories, human egocentric demonstrations, synthetic simulation data, vision-and-language navigation data, trajectory-centric supervision, and auxiliary vision-language data. To support multiple robot platforms, we introduce embodiment-aware prompt conditioning, where robot-specific textual descriptions specify the current embodiment and control convention. We further cast manipulation, navigation, and trajectory prediction into a unified action-and-trajectory prediction framework, enabling transferable visual grounding, spatial reasoning, and continuous action generation across robot morphologies, task families, and environments. Experiments on manipulation, navigation, and trajectory-centric benchmarks show consistent multi-task performance and out-of-distribution generalization under variations in scene layout, background, lighting, object configuration, and robot embodiment. Qwen-VLA-Instruct achieves 97.9% on LIBERO, 73.7% on Simpler-WidowX, 86.1%/87.2% on RoboTwin-Easy/Hard, 69.0% OSR on R2R, 59.6% SR on RxR, 76.9% average OOD success in real-world ALOHA experiments, and 26.6% zero-shot success on DOMINO dynamic manipulation.
>
---
#### [replaced 004] Interpretable Multimodal Gesture Recognition for Drone and Mobile Robot Teleoperation via Log-Likelihood Ratio Fusion
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于手势识别任务，旨在解决无人机和移动机器人远程操作中的可靠性问题。通过融合惯性数据与电容传感信号，提升识别性能并增强可解释性。**

- **链接: [https://arxiv.org/pdf/2602.23694](https://arxiv.org/pdf/2602.23694)**

> **作者:** Seungyeol Baek; Jaspreet Singh; Lala Shakti Swarup Ray; Hymalai Bello; Paul Lukowicz; Sungho Suh
>
> **摘要:** Human operators are still frequently exposed to hazardous environments such as disaster zones and industrial facilities, where intuitive and reliable teleoperation of mobile robots and Unmanned Aerial Vehicles (UAVs) is essential. In this context, hands-free teleoperation enhances operator mobility and situational awareness, thereby improving safety in hazardous environments. While vision-based gesture recognition has been explored as one method for hands-free teleoperation, its performance often deteriorates under occlusions, lighting variations, and cluttered backgrounds, limiting its applicability in real-world operations. To overcome these limitations, we propose a multimodal gesture recognition framework that integrates inertial data (accelerometer, gyroscope, and orientation) from Apple Watches on both wrists with capacitive sensing signals from custom gloves. We design a late fusion strategy based on the log-likelihood ratio (LLR), which not only enhances recognition performance but also provides interpretability by quantifying modality-specific contributions. To support this research, we introduce a new dataset of 20 distinct gestures inspired by aircraft marshalling signals, comprising synchronized RGB video, IMU, and capacitive sensor data. Experimental results demonstrate that our framework achieves performance comparable to a state-of-the-art vision-based baseline while significantly reducing computational cost, model size, and training time, making it well suited for real-time robot control. We therefore underscore the potential of sensor-based multimodal fusion as a robust and interpretable solution for gesture-driven mobile robot and drone teleoperation.
>
---
#### [replaced 005] Control of a Twin Rotor using Twin Delayed Deep Deterministic Policy Gradient (TD3)
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于控制任务，旨在解决TRAS系统的稳定与轨迹跟踪问题。通过TD3算法实现无需模型的强化学习控制，并验证其在仿真和实际环境中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.13356](https://arxiv.org/pdf/2512.13356)**

> **作者:** Zeyad Gamal; Youssef Mahran; Ayman El-Badawy
>
> **备注:** This is the Author Accepted Manuscript version of a paper accepted for publication. The final published version is available via IEEE Xplore
>
> **摘要:** This paper proposes a reinforcement learning (RL) framework for controlling and stabilizing the Twin Rotor Aerodynamic System (TRAS) at specific pitch and azimuth angles and tracking a given trajectory. The complex dynamics and non-linear characteristics of the TRAS make it challenging to control using traditional control algorithms. However, recent developments in RL have attracted interest due to their potential applications in the control of multirotors. The Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm was used in this paper to train the RL agent. This algorithm is used for environments with continuous state and action spaces, similar to the TRAS, as it does not require a model of the system. The simulation results illustrated the effectiveness of the RL control method. Next, external disturbances in the form of wind disturbances were used to test the controller's effectiveness compared to conventional PID controllers. Lastly, experiments on a laboratory setup were carried out to confirm the controller's effectiveness in real-world applications.
>
---
#### [replaced 006] Semantic-Geometric Task Representations for Bimanual Manipulation from Human Demonstrations to Robot Action Planning
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究双臂操作任务，解决从人类演示中学习结构化任务表示的问题。通过语义-几何图模型，联合编码对象身份与运动历史，提升任务规划效果。**

- **链接: [https://arxiv.org/pdf/2601.11460](https://arxiv.org/pdf/2601.11460)**

> **作者:** Franziska Herbert; Vignesh Prasad; Han Liu; Dorothea Koert; Georgia Chalvatzaki
>
> **备注:** 9 pages, 7 figures, preprint
>
> **摘要:** Learning structured task representations from human demonstrations is essential for bimanual manipulation, where action ordering, object involvement, and interaction geometry vary significantly across executions. A key challenge lies in jointly capturing the discrete semantic task structure and the temporal evolution of object-centric geometric relations in a form that supports reasoning over task progression. We introduce a semantic--geometric graph-based task representation that jointly encodes object identities, inter-object semantic relations, and per-object motion histories, via a Message Passing Neural Network (MPNN) encoder and a Transformer-based decoder. The encoder operates solely on the temporal scene graph, producing structured representations decoupled from action labels. The decoder then conditions on action-context to forecast future actions, associated objects, and object motions. This decoupling learns task-agnostic representations, enabling encoder reuse across embodiments through decoder-only finetuning on a small robot dataset. Across eleven bimanual tasks from two datasets, we find that the benefit of structured semantic--geometric representations over simpler sequence-based models grows with task variability in action ordering and object involvement. At deployment, a planner couples the action and motion predictions with learned Probabilistic Movement Primitives, achieving full task success on two real-robot bimanual tasks and outperforming graph ablations, Transformer, decoder-only, and finetuned vision-language model baselines.
>
---
#### [replaced 007] URDF-Anything+: End-to-End Generation for Simulation-Ready Articulated Assets
- **分类: cs.RO**

- **简介: 该论文提出URDF-Anything+，解决从单张图像生成可执行URDF模型的问题。通过端到端扩散框架，直接生成关节结构与几何，提升重建质量和效率。**

- **链接: [https://arxiv.org/pdf/2603.14010](https://arxiv.org/pdf/2603.14010)**

> **作者:** Zhuangzhe Wu; Yue Xin; Chengkai Hou; Minghao Chen; Yaoxu Lyu; Jieyu Zhang; Shanghang Zhang
>
> **摘要:** Articulated objects are fundamental for robotics, simulation of physics, and interactive virtual environments. However, recovering them from visual observations is inherently challenging, as images provide only partial and ambiguous cues about both part geometry and their underlying kinematic structure. Existing approaches typically rely on multi-stage pipelines, retrieval from asset libraries, or explicit part segmentation. We present URDF-Anything+, an end-to-end autoregressive diffusion framework that generates simulation-ready URDF models directly from a single RGB image. Conditioned on visual observations and object geometry, URDF-Anything+ operates in a structured latent space and jointly models part geometry and articulation in a unified generation process. Specifically, the model sequentially predicts each articulated part together with its associated joint parameters, while a termination token dynamically determines the number of parts. This design enables direct generation of fully executable URDFs without external retrieval or post-processing stages. Experiments on large-scale articulated object benchmarks demonstrate that URDF-Anything+ outperforms prior methods in geometric reconstruction quality, joint parameter estimation, and physical executability, while being substantially more efficient than existing multi-stage approaches. Furthermore, the generated URDFs serve as faithful digital twins, enabling the zero-shot transfer of manipulation policies trained purely in simulation.
>
---
#### [replaced 008] RCM-ACT: Imitation Learning with Dynamic RCM Calibration for Autonomous Intraocular Foreign Body Removal
- **分类: cs.RO**

- **简介: 该论文属于自主眼内异物取出任务，解决机器人操作中的运动学不确定性问题。提出RCM-ACT框架，结合动态校准与模仿学习，实现精准抓取。**

- **链接: [https://arxiv.org/pdf/2508.19191](https://arxiv.org/pdf/2508.19191)**

> **作者:** Yue Wang; Wenjie Deng; Haotian Xue; Di Cui; Yiqi Chen; Mingchuan Zhou; Haochao Ying; Jian Wu
>
> **摘要:** Intraocular foreign body removal demands millimeter-level precision in confined intraocular spaces, yet existing robotic systems predominantly rely on manual teleoperation with steep learning curves. To address the challenges of autonomous manipulation, particularly kinematic uncertainties from variable motion scaling and Remote Center of Motion (RCM) point variation, we propose RCM-ACT, an imitation learning framework for autonomous intraocular foreign body ring manipulation. Our approach integrates RCM dynamic calibration to resolve coordinate system inconsistencies caused by intraocular instrument variation and introduces the RCM-ACT architecture, which combines action chunking transformers with episode-level kinematic realignment. Trained solely on stereo visual data and instrument kinematics from expert demonstrations in an artificial eye model, RCM-ACT successfully completes ring grasping and positioning tasks without explicit depth sensing. Experimental validation demonstrates the successful implementation of end-to-end autonomy under uncalibrated microscopy conditions, achieving a mean 3-D Euclidean grasp deviation of 0.686 mm and 11/20 full-task successes. The results provide a viable framework for developing intelligent eye surgical systems capable of complex intraocular procedures.
>
---
#### [replaced 009] Wall-OSS-0.5 Technical Report
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决VLA预训练是否能直接生成可执行行为的问题。工作包括构建Wall-OSS-0.5模型，验证其无需微调即可实现真实机器人操作。**

- **链接: [https://arxiv.org/pdf/2605.30877](https://arxiv.org/pdf/2605.30877)**

> **作者:** Ryan Yu; Pushi Zhang; Starrick Liu; Brae Liu; Miracle Kang; Shalfun Li; Lights Shi; Ellie Ma; Ping Yang; Chris Pan; Jerry Chen; Dongxiu Liu; Rain Sun; Miles Guo; Byron Zhang; Hugo Zhou; Zach Xu; Vincent Chen; Harrison Huang; James Wang; Dance Kuzi; Andy Zhai; Hang Su; Roy Gan; Lucy Liang; Hao Wang; Qian Wang
>
> **摘要:** Large-scale Vision-Language-Action (VLA) pretraining is increasingly adopted as the foundation for robot policies, yet the evidence for pretrained VLAs is almost invariably reported after task-specific fine-tuning. This leaves a foundational question unanswered: does VLA pretraining itself yield executable robot behavior, or does it merely furnish a better initialization for downstream policy learning? We present Wall-OSS-0.5, an open-source 4B VLA built upon a 3B VLM backbone augmented with action-generation components, designed so that pretrained robotic capability is directly measurable on physical hardware. The model is pretrained across more than 20 embodiments, processing over one million robot trajectories per epoch alongside a grounded multimodal corpus. We adopt a gradient-bridged co-training recipe in which three objectives play distinct and complementary roles: discrete action prediction routes strong VLM-native gradients into the backbone, multimodal prediction preserves grounded vision-language understanding, and continuous flow matching serves as the deployment-time action interface. Before task-specific fine-tuning, the pretrained checkpoint achieves non-trivial zero-shot real-robot behavior, completing several tasks, including a held-out deformable manipulation task, at high task progress on a 17-task suite. After fine-tuning, the same checkpoint serves as a stronger adaptation prior, reaching 60.5% average task progress on 15 real-robot tasks and outperforming \pi_0.5 by 17.5%. Multimodal evaluations further confirm that action training does not erode grounded vision-language competence: the model preserves broad vision-language ability while strengthening embodied grounding. Together, these results reposition VLA pretraining from an initialization strategy to a directly testable, already useful source of robot capability.
>
---
#### [replaced 010] A Unified Framework for Probabilistic Dynamic-, Trajectory- and Vision-based Virtual Fixtures
- **分类: cs.RO**

- **简介: 该论文提出一种统一框架，用于概率性虚拟夹具，解决人机协作中的动态、轨迹和视觉引导问题，实现从手动到全自动化任务的平滑切换。**

- **链接: [https://arxiv.org/pdf/2506.10239](https://arxiv.org/pdf/2506.10239)**

> **作者:** Maximilian Mühlbauer; Bernhard Weber; Sylvain Calinon; Freek Stulp; Alin Albu-Schäffer; João Silvério
>
> **备注:** for the supplementary video, see this https URL
>
> **摘要:** Probabilistic Virtual Fixtures (VFs) enable the adaptive selection of the most suitable haptic feedback for each phase of a task, based on learned or perceived uncertainty. While keeping the human in the loop remains essential, for instance, to ensure high precision, partial automation of certain task phases is critical for productivity. We present a unified framework for probabilistic VFs that seamlessly switches between manual fixtures, semi-automated fixtures (with the human handling precise tasks), and full autonomy. We introduce a novel probabilistic Dynamical System-based VF for coarse guidance, enabling the robot to autonomously complete certain task phases while keeping the human operator in the loop. For tasks requiring precise guidance, we extend probabilistic position-based trajectory fixtures with automation, allowing for seamless human interaction, geometry-awareness and optimal impedance gains. For manual tasks requiring very precise guidance, we also extend visual servoing fixtures with the same geometry-awareness and impedance behavior. We validate our approach on different robots, including an evaluation with expert users, showcasing operation modes, the ease of programming fixtures and lower interaction forces and favorable usability compared to a baseline.
>
---
#### [replaced 011] FDIO: Frequency Decomposed Inertial Odometry
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于行人惯性里程计任务，解决双设备设置下IMU信号耦合问题。通过频率分解和Mamba模块提升定位精度。**

- **链接: [https://arxiv.org/pdf/2511.15645](https://arxiv.org/pdf/2511.15645)**

> **作者:** Shanshan Zhang; Liqin Wu; Wenying Cao; Lingxiang Zheng; Yu Yang
>
> **摘要:** Pedestrian inertial odometry (PIO) estimates autonomous pedestrian motion using only acceleration and angular velocity measurements collected by an inertial measurement unit (IMU), making it highly valuable for consumer level localization applications. However, under a dual device acquisition setting, IMU signals collected by a freely carried mobile device are inherently composite signals in which the global motion of the human torso is coupled with perturbations induced by local limb motion. This coupling makes accurate human motion modeling more challenging. To address this issue, this paper proposes frequency decomposed inertial odometry (FDIO). The proposed method first decomposes input IMU signals into low frequency and high frequency components using a Laplacian pyramid. It then adopts a Mamba module to model long range motion information from the low frequency component and uses a multi scale convolution module to extract fine grained local dynamic features from the high frequency component. Experiments on five public PIO datasets show that FDIO achieves an average absolute trajectory error of 3.221~m and an average relative trajectory error of 2.550~m, reducing the errors by 33.3\% and 16.7\% compared with the RoNIN ResNet baseline, respectively. These results validate the effectiveness of the proposed frequency decomposition strategy. To the best of our knowledge, this work is among the first efforts to introduce Mamba and a frequency decomposition architecture into inertial odometry.
>
---
#### [replaced 012] Provably Safe Motion Planning Under Unknown Disturbances
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动规划任务，解决在未知扰动下保证安全的问题。通过学习Wasserstein模糊带，构建满足约束的采样规划树，提升算法安全性和效率。**

- **链接: [https://arxiv.org/pdf/2605.26625](https://arxiv.org/pdf/2605.26625)**

> **作者:** Ibon Gracia; Qi Heng Ho; Luca Laurenti; Morteza Lahijanian
>
> **摘要:** We present a provably safe sampling-based motion planning algorithm for robotic systems affected by random disturbances of unknown distribution. We consider systems with linear or linearizable dynamics evolving in workspace with arbitrary-shaped obstacles subject to state and control constraints. Safety requirements are formulated as chance-constraints. Our approach leverages data from trajectories of the system to learn a Wasserstein ambiguity tube, i.e., a sequence of ambiguity sets, which contains the trajectory of the system's state distribution with high confidence. This ambiguity tube is then used in a probabilistically complete algorithm to grow a sampling-based motion planning tree that respects the constraints of the problem. We show that learning several lower-dimensional ambiguity tubes instead of a single high-dimensional one effectively reduces the conservatism and boosts scalability. Additionally, we design an efficient bandit-based validity checker that remarkably increases the empirical performance of our approach without sacrificing probabilistic completeness. Case studies show our algorithm finds valid plans in cluttered environments under strict safety thresholds, outperforming state-of-the-art methods.
>
---
#### [replaced 013] 3D RL-DWA: A Hybrid Reinforcement Learning and Dynamic Window Approach for Goal-Directed Local Navigation in Multi-DoF Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人局部导航任务，解决多自由度机器人在复杂环境中的3D导航问题。结合强化学习与动态窗口方法，提升导航与形变能力。**

- **链接: [https://arxiv.org/pdf/2605.12689](https://arxiv.org/pdf/2605.12689)**

> **作者:** Chiara Castellani; Enrico Turco; Domenico Prattichizzo
>
> **备注:** Accepted for publication in the Proceedings of the IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM 2026)
>
> **摘要:** In this paper, we present a novel hybrid approach that combines Reinforcement Learning (RL) with Dynamic Window Approach (DWA) for adaptive 3D local navigation of high-degree-of-freedom robotic systems. Our method leverages sparse point cloud data to dynamically adjust both the motion and the shape of a deformable microrobot, enabling the system to navigate toward a goal in complex, constrained environments while maximizing the occupied volume. We evaluate our framework in a simulated vascular network. Experimental results, based on 1080 trials, indicate that integrating RL with a DWA-based local planner significantly enhances both deformation and navigation capabilities compared to pure RL and model-based methods. In particular, the proposed autonomous controller consistently achieves high deformation and near-perfect path completion during training and maintains robust performance in unseen scenarios. These findings highlight the potential of hybrid planning strategies for efficient and adaptive 3D navigation under sparse sensory conditions.
>
---
#### [replaced 014] SPARC: Spatial-Aware Path Planning via Attentive Agent Communication
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人路径规划任务，解决通信效率问题。提出RMHA机制，通过空间距离增强注意力，提升密集环境下的协作成功率。**

- **链接: [https://arxiv.org/pdf/2603.02845](https://arxiv.org/pdf/2603.02845)**

> **作者:** Sayang Mu; Xiangyu Wu; Bo An
>
> **备注:** The manuscript is being withdrawn at the request of the first author for the purpose of revising content and re-uploading a revised version with updated data/figures/text . The revised manuscript will be resubmitted to arXiv promptly with the same author list and research theme
>
> **摘要:** Efficient communication is critical for decentralized Multi-Robot Path Planning (MRPP), yet existing learned communication methods treat all neighboring robots equally regardless of their spatial proximity, leading to diluted attention in congested regions where coordination matters most. We propose Relation enhanced Multi Head Attention (RMHA), a communication mechanism that explicitly embeds pairwise Manhattan distances into the attention weight computation, enabling each robot to dynamically prioritize messages from spatially relevant neighbors. Combined with a distance-constrained attention mask and GRU gated message fusion, RMHA integrates seamlessly with MAPPO for stable end-to-end training. In zero-shot generalization from 8 training robots to 128 test robots on 40x40 grids, RMHA achieves approximately 75 percent success rate at 30 percent obstacle density outperforming the best baseline by over 25 percentage points. Ablation studies confirm that distance-relation encoding is the key contributor to success rate improvement in high-density environments. Index Terms-Multi-robot path planning, graph attention mechanism, multi-head attention, communication optimization, cooperative decision-making
>
---
#### [replaced 015] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决动作生成的顺序固定和信息碎片问题。提出Discrete Diffusion VLA，通过离散扩散建模实现自适应解码和高效动作生成。**

- **链接: [https://arxiv.org/pdf/2508.20072](https://arxiv.org/pdf/2508.20072)**

> **作者:** Zhixuan Liang; Yizhuo Li; Tianshuo Yang; Chengyue Wu; Sitong Mao; Liuao Pei; Tian Nian; Shunbo Zhou; Xiaokang Yang; Jiangmiao Pang; Yao Mu; Ping Luo
>
> **备注:** Accepted by ICML 2026. 17 pages
>
> **摘要:** Vision-Language-Action (VLA) models adapt large vision-language backbones to map images and instructions into robot actions. However, prevailing VLAs either generate actions autoregressively in a fixed left-to-right order with poor performance or attach separate diffusion heads outside the backbone that fragments information pathways and hinders unified, scalable architectures. Instead, we present Discrete Diffusion VLA that discretizes action chunks and models them with discrete diffusion pattern retaining progressive refinement inside the unified transformer backbone. Our method achieves an adaptive decoding order that resolves high-confidence action elements before harder ones and employs secondary re-masking to revisit uncertain predictions, enabling robust error correction. This design preserves pretrained vision-language priors, supports parallel decoding, and improves the efficiency. Discrete Diffusion VLA achieves 96.4% avg. success on LIBERO, 71.2% visual matching on SimplerEnv-Fractal, and 54.2% overall on SimplerEnv-Bridge. On out-of-distribution tests of LIBERO-Goal, our method exhibits only 0.8% language degradation versus 8.0% of parallel decoding, and 20.4% vision degradation versus 29.0% for continuous diffusion, demonstrating well retention of pretrained vision-language capabilities. We also conduct two real-robot evaluations on AgileX Cobot Magic platform to show the method's effectiveness.
>
---
#### [replaced 016] CloSE: A Geometric Shape-Agnostic Cloth State Representation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于计算机视觉或机器人学任务，旨在解决衣物变形状态表示问题。提出CloSE表示法，用于准确预测衣物褶皱位置并支持语义标注与规划。**

- **链接: [https://arxiv.org/pdf/2504.05033](https://arxiv.org/pdf/2504.05033)**

> **作者:** Jay Kamat; Júlia Borràs; Carme Torras
>
> **备注:** Accepted at ICRA 2026 (8 pages, 11 figures, 1 table). Project page: this https URL
>
> **摘要:** Cloth manipulation is a difficult problem mainly because of the non-rigid nature of cloth, which makes a good representation of deformation essential. We present a new representation for the deformation-state of clothes. First, we propose the dGLI disk representation based on topological indices computed for edge segments of the cloth border that are arranged on a circular grid. The heat-map of the dGLI disk uncovers patterns that correspond to features of the cloth state that are consistent for different shapes, sizes or orientation of the cloth. We then abstract these important features from the dGLI disk into a circle, calling it the Cloth StatE representation (CloSE). This representation is compact, continuous, and general for different shapes. We show that this representation is able to accurately predict the fold locations for several simulation clothing datasets. Finally, we also show the strengths of this representation in two relevant applications: semantic labeling and high- and low-level planning. The code and the dataset can be accessed from: this https URL
>
---
#### [replaced 017] Prior Availability in Industrial Visual Sim-to-Real: A Review of CAD-Guided and CAD-Unavailable Regimes
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于工业视觉领域，解决sim-to-real迁移中的先验知识缺失问题。通过分析CAD可用与不可用场景，提出分类框架并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.30581](https://arxiv.org/pdf/2605.30581)**

> **作者:** Chenxi Tao; Seung-Kyum Choi
>
> **备注:** Review article; 103 references; 9 main figures; empirical anchors on T-LESS/BOP, MVTec AD, and VisA
>
> **摘要:** Industrial visual sim-to-real is often described as transferring from synthetic images to real images, but industrial deployment usually involves a broader mismatch between available evidence and required decisions. A system may be built from CAD renderings, simulated RGB-D observations, normal reference images, synthetic defects, pretrained feature spaces, or language prompts, yet deployed under different sensors, lighting, materials, fixtures, calibration, production variation, and rare defect modes. This review reframes industrial visual sim-to-real as a domain-gap problem organized by prior availability. We distinguish CAD-available settings, where explicit object geometry can support rendering, calibration, pose estimation, segmentation, and test-time geometric verification; CAD-unavailable settings, where geometry is replaced by normal-reference appearance, feature distributions, teacher-student residuals, synthetic anomaly assumptions, foundation features, or vision-language priors; and boundary-prior settings, where approximate models, templates, reference views, or semantic correspondences preserve only part of the CAD role. This framing connects CAD-based detection and 6D pose-estimation literature with industrial anomaly and surface-inspection literature that is usually reviewed separately. To make the taxonomy concrete, we use empirical anchors on T-LESS/BOP, MVTec AD, and VisA. The anchors show that CAD render count alone does not close transfer; source-distribution design, detector capacity, and small real calibration can matter more. They also show that CAD at test time creates a distinct verification channel through mask, pose, and depth consistency, whereas CAD-unavailable inspection relies on calibrated normality and feature deviation. The review therefore argues against a single cross-task leaderboard and instead asks what prior grounds the deployment decision.
>
---
#### [replaced 018] Feedback Matters: Augmenting Autonomous Dissection with Visual and Topological Feedback
- **分类: cs.RO**

- **简介: 该论文属于自主手术任务，旨在解决组织分离中的动态环境适应问题。通过引入视觉与拓扑反馈机制，提升系统的自主性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.04074](https://arxiv.org/pdf/2510.04074)**

> **作者:** Chung-Pang Wang; Changwei Chen; Xiao Liang; Soofiyan Atar; Florian Richter; Michael Yip
>
> **摘要:** Autonomous surgical systems must adapt to highly dynamic environments where tissue properties and visual cues evolve rapidly. Central to such adaptability is feedback: the ability to sense, interpret, and respond to changes during execution. While feedback mechanisms have been explored in surgical robotics, ranging from tool and tissue tracking to error detection, existing methods remain limited in handling the topological and perceptual challenges of tissue dissection. In this work, we propose a feedback-enabled framework for autonomous tissue dissection that explicitly reasons about topological changes from endoscopic images after each dissection action. This structured feedback guides subsequent actions, enabling the system to localize dissection progress and adapt policies online. To improve the reliability of such feedback, we introduce visibility metrics that quantify tissue exposure and formulate optimal controller designs that actively manipulate tissue to maximize visibility. Finally, we integrate these feedback mechanisms with both planning-based and learning-based dissection methods, and demonstrate experimentally that they significantly enhance autonomy, reduce errors, and improve robustness in complex surgical scenarios.
>
---
#### [replaced 019] Genie 4D: Semantic-Prior-Guided 4D Dynamic Scene Reconstruction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于4D动态场景重建任务，解决动态场景中几何与语义结合的问题。提出Genie 4D框架，融合语义先验与实时几何重建，提升跟踪精度与场景完整性。**

- **链接: [https://arxiv.org/pdf/2604.09877](https://arxiv.org/pdf/2604.09877)**

> **作者:** Yiru Yang; Zhuojie Wu; Nishant Kumar Singh; Max Schulthess
>
> **摘要:** At the intersection of computer vision and robotic perception, 4D reconstruction of dynamic scenes connects low-level geometric sensing with high-level semantic understanding. We present Genie 4D, a framework that turns hand-held phone capture into a semantically grounded, action-controllable 4D world model. Genie 4D couples a real-time visual-inertial Gaussian splatting front-end for metric geometry with a feed-forward 4D backbone regularized by frozen DINOv3 features acting as structural priors. The semantic priors suppress identity drift during dynamic tracking, while a short conditional diffusion refiner recovers high-frequency surface detail that regression backbones smooth away. Finally, a lightweight latent-action head exposes the reconstructed 4D state to a Genie-style world model trained with a JEPA-style next-embedding objective, so that the scene can be rolled forward under user actions. On the Point Odyssey and TUM-Dynamics benchmarks, Genie 4D retains the linear time complexity O(T) of feed-forward baselines while improving 3D tracking accuracy (APD) and reconstruction completeness, and it runs interactively on a single consumer GPU (RTX 5090) from iPhone, Mac, Windows, and Linux capture clients. Genie 4D offers a practical, semantic-prior-guided path toward physically grounded world models.
>
---
#### [replaced 020] Hybrid TD3: Overestimation Bias Analysis and Stable Policy Optimization for Hybrid Action Space
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，解决混合动作空间中的策略优化问题。提出Hybrid TD3算法，分析过估计偏差并提升训练稳定性。**

- **链接: [https://arxiv.org/pdf/2603.01302](https://arxiv.org/pdf/2603.01302)**

> **作者:** Thanh-Tuan Tran; Thanh Nguyen Canh; Nak Young Chong; Xiem HoangVan
>
> **摘要:** Reinforcement learning in discrete-continuous hybrid action spaces presents fundamental challenges for robotic manipulation, where high-level task decisions and low-level joint-space execution must be jointly optimized. Existing approaches either discretize continuous components or relax discrete choices into continuous approximations, which suffer from scalability limitations and training instability in high-dimensional action spaces and under domain randomization. In this paper, we propose Hybrid TD3, an extension of Twin Delayed Deep Deterministic Policy Gradient (TD3) that natively handles parameterized hybrid action spaces in a principled manner. We conduct a rigorous theoretical analysis of overestimation bias in hybrid action settings, deriving formal bounds under twin-critic architectures and establishing a complete bias ordering across five algorithmic variants under synchronized Gaussian error assumptions. Building on this analysis, we introduce a weighted clipped Q-learning target that marginalizes over the discrete action distribution, achieving equivalent bias reduction to standard clipped minimization while improving policy smoothness. Experimental results demonstrate that Hybrid TD3 achieves superior training stability and competitive performance against state-of-the-art hybrid action baselines.
>
---
#### [replaced 021] Self-Imitated Diffusion Policy for Efficient and Robust Visual Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决扩散策略因依赖专家示范导致效率低、冗余的问题，提出SIDP框架通过自模仿机制提升导航效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.22965](https://arxiv.org/pdf/2601.22965)**

> **作者:** Runhua Zhang; Junyi Hou; Changxu Cheng; Qiyi Chen; Tao Wang; Wuyue Zhao
>
> **备注:** Preprint
>
> **摘要:** Diffusion policies (DP) have demonstrated significant potential in visual navigation by capturing diverse multi-modal trajectory distributions. However, standard imitation learning (IL), which most DP methods rely on for training, often inherits sub-optimality and redundancy from expert demonstrations, thereby necessitating a computationally intensive "generate-then-filter" pipeline that relies on auxiliary selectors during inference. To address these challenges, we propose Self-Imitated Diffusion Policy (SIDP), a novel framework that learns improved planning by selectively imitating a set of trajectories sampled from itself. Specifically, SIDP introduces a reward-guided self-imitation mechanism that encourages the policy to consistently produce high-quality trajectories efficiently, rather than outputs of inconsistent quality, thereby reducing reliance on extensive sampling and post-filtering. During training, we employ a reward-driven curriculum learning paradigm to mitigate inefficient data utility, and goal-agnostic exploration for trajectory augmentation to improve planning robustness. Extensive evaluations on a comprehensive simulation benchmark show that SIDP significantly outperforms previous methods, with real-world experiments confirming its effectiveness across multiple robotic platforms. On Jetson Orin Nano, SIDP delivers a 2.5$\times$ faster inference than the baseline NavDP, i.e., 110ms VS 273ms, enabling efficient real-time deployment.
>
---
#### [replaced 022] CART: Context-Aware Terrain Adaptation using Temporal Sequence Selection for Legged Robots
- **分类: cs.RO**

- **简介: 该论文提出CART方法，解决腿式机器人在复杂地形上的适应问题，通过融合视觉与本体感知提升行走稳定性。**

- **链接: [https://arxiv.org/pdf/2604.14344](https://arxiv.org/pdf/2604.14344)**

> **作者:** Kartikeya Singh; Youngjin Kim; Yash Turkar; Karthik Dantu
>
> **摘要:** Animals in nature combine multiple modalities, such as sight and feel, to perceive terrain and develop an understanding of how to walk on uneven terrain in an efficient manner. Similarly, legged robots need to develop their ability to stably walk on complex terrains by developing an understanding of the relationship between vision and proprioception. Most current terrain-adaptation methods remain susceptible to failure on complex off-road terrain because they do not explicitly model the context between exteroceptive terrain appearance and proprioceptive physical interaction. This experience-based learning often creates a Visual-Texture Paradox between what has been seen and how it actually feels. In this work, we introduce CART, a high-level controller built on a context-aware terrain adaptation approach that integrates proprioception and exteroception from onboard sensing to achieve a robust understanding of terrain. We evaluate our method on multiple terrains using the Unitree Go2 and ANYmal-C robot on the IsaacSim simulator and a Boston Dynamics SPOT robot for our real-world experiments. To evaluate whether the learned context improves locomotion behavior under the various paradox circumstances, we measure the robot s stability, traversal success, and task completion time in both simulation and real-world experiments. We compare CART against state-of-the-art locomotion and terrain- adaptation baselines across diverse terrain conditions. CART improves the average success rate by 5% over the baselines in simulation, while improving context-conditioned locomotion behavior, including up to 41% lower base oscillation in simulation and 22% in the real world, without increasing the time required to complete the locomotion tasks.
>
---
#### [replaced 023] Simple Recipe Works: Vision-Language-Action Models are Natural Continual Learners with Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究视觉-语言-动作模型的持续强化学习任务，解决模型在连续学习中遗忘问题。通过简单序列微调结合低秩适应，实现稳定且高效的持续学习。**

- **链接: [https://arxiv.org/pdf/2603.11653](https://arxiv.org/pdf/2603.11653)**

> **作者:** Jiaheng Hu; Jay Shim; Chen Tang; Yoonchang Sung; Bo Liu; Peter Stone; Roberto Martin-Martin
>
> **备注:** Accepted at RLC 2026
>
> **摘要:** Continual Reinforcement Learning (CRL) for Vision-Language-Action (VLA) models is a promising direction toward self-improving embodied agents that can adapt in openended, evolving environments. However, conventional wisdom from continual learning suggests that naive Sequential Fine-Tuning (Seq. FT) leads to catastrophic forgetting, necessitating complex CRL strategies. In this work, we take a step back and conduct a systematic study of CRL for large pretrained VLAs across diverse lifelong RL benchmarks. We find that, contrary to established belief, simple Seq. FT with low-rank adaptation (LoRA) is remarkably strong: it achieves high plasticity, exhibits little to no forgetting, and retains strong zero-shot generalization, frequently outperforming more sophisticated CRL methods. Through detailed analysis, we show that this robustness arises from a synergy between the large pretrained model, parameter-efficient adaptation, and on-policy RL. Together, these components reshape the stability-plasticity trade-off, making continual adaptation both stable and scalable. Our results position Sequential Fine-Tuning as a powerful method for continual RL with VLAs and provide new insights into lifelong learning in the large model era. Code is available at this http URL.
>
---
#### [replaced 024] HyperDet: 3D Object Detection with Hyper 4D Radar Point Clouds
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于3D目标检测任务，解决4D雷达点云稀疏、噪声多的问题。通过构建任务感知的超4D雷达点云，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2602.11554](https://arxiv.org/pdf/2602.11554)**

> **作者:** Yichun Xiao; Runwei Guan; Jin Jin; Fangqiang Ding
>
> **备注:** 11 pages, 3 figures, 3 tables
>
> **摘要:** How far can 3D object detection go using 4D radar alone? Despite offering weather-robust and velocity-aware sensing for autonomous perception, modern 4D radar still yields sparse, noisy, and unstable point clouds, limiting radar-only 3D detection. We present HyperDet, a detector-agnostic framework that constructs task-aware hyper 4D radar point clouds before detection. HyperDet first refines short-window surround-view radar observations through spatio-temporal accumulation, cross-sensor validation, and Doppler-guided motion compensation, improving return reliability and temporal coherence. It then performs foreground generative enhancement using LiDAR-guided pseudo-radar supervision available only during training, enriching object geometry while preserving measured radar background and radar-native attributes. During detector training, radar-aware object-level augmentation further preserves Doppler consistency under geometric relocation. At inference time, HyperDet requires radar input alone and can be directly paired with standard 3D detectors. Experiments on two public surround-view 4D radar datasets demonstrate consistent improvements over raw radar inputs across standard 3D detectors, validating input-level radar enhancement as an effective approach to radar-only 3D detection.
>
---
#### [replaced 025] Sim-to-Real Transfer for Muscle-Actuated Robots via Generalized Actuator Networks
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决肌肉驱动机器人从仿真到现实的迁移问题。通过提出GenAN模型，实现无需扭矩传感器的关节轨迹建模，成功部署了多项控制策略。**

- **链接: [https://arxiv.org/pdf/2604.09487](https://arxiv.org/pdf/2604.09487)**

> **作者:** Jan Schneider; Mridul Mahajan; Le Chen; Simon Guist; Bernhard Schölkopf; Ingmar Posner; Dieter Büchler
>
> **摘要:** Tendon drives paired with soft muscle actuation enable faster and safer robots while potentially accelerating skill acquisition. Still, these systems are rarely used in practice due to inherent nonlinearities, friction, and hysteresis, which complicate modeling and control. So far, these challenges have hindered policy transfer from simulation to real systems. To bridge this gap, we propose a sim-to-real pipeline that learns a neural network model of this complex actuation and leverages established rigid body simulation for the arm dynamics and interactions with the environment. Our method, called Generalized Actuator Network (GenAN), enables actuation model identification across a wide range of robots by learning directly from joint position trajectories rather than requiring torque sensors. Using GenAN on PAMY2, a tendon-driven robot powered by pneumatic artificial muscles, we successfully deploy dynamic but precise goal-reaching, ball-in-a-cup, and table tennis policies, trained entirely in simulation. To the best of our knowledge, this result constitutes the first successful sim-to-real transfer for a four-degrees-of-freedom muscle-actuated robot arm.
>
---
#### [replaced 026] AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出AGILE框架，解决单目视频中手物交互重建问题，通过生成式方法和鲁棒跟踪策略，提升几何精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2602.04672](https://arxiv.org/pdf/2602.04672)**

> **作者:** Jin-Chuan Shi; Binhong Ye; Tao Liu; Junzhe He; Yangjinhui Xu; Xiaoyang Liu; Zeju Li; Hao Chen; Chunhua Shen
>
> **备注:** 16 pages, SIGGRAPH 2026
>
> **摘要:** Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, ARCTIC, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior arts frequently collapse. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications. Project page: this https URL.
>
---
#### [replaced 027] LAP: Fast LAtent Diffusion Planner for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶规划任务，解决扩散模型延迟高和低层运动学占用模型容量的问题。提出LAP框架，在潜在空间中分离高层意图与低层运动，提升规划效率与质量。**

- **链接: [https://arxiv.org/pdf/2512.00470](https://arxiv.org/pdf/2512.00470)**

> **作者:** Jinhao Zhang; Wenlong Xia; Zhexuan Zhou; Haoming Song; Youmin Gong; Jie Mei
>
> **摘要:** Diffusion models have demonstrated strong capabilities for modeling human-like driving behaviors in autonomous driving, but their iterative sampling process induces substantial latency, and operating directly on raw trajectory points forces the model to spend capacity on low-level kinematics, rather than high-level multi-modal semantics. To address these limitations, we propose LAtent Planner (LAP), a framework that plans in a VAE-learned latent space that disentangles high-level intents from low-level kinematics, enabling our planner to capture rich, multi-modal driving strategies. To bridge the representational gap between the high-level semantic planning space and the vectorized scene context, we introduce an intermediate feature alignment mechanism that facilitates robust information fusion. Notably, LAP can produce high-quality plans in one single denoising step, substantially reducing computational overhead. Through extensive evaluations on the large-scale nuPlan benchmark, LAP achieves state-of-the-art closed-loop performance among learning-based planning methods, while demonstrating an inference speed-up of at most 10x over previous SOTA approaches.
>
---
#### [replaced 028] Improving Diffusion Planners by Self-Supervised Action Gating with Energies
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决扩散规划中因局部动态不一致导致的执行脆弱问题。提出SAGE方法，在推理时通过能量惩罚不一致计划，提升性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.02650](https://arxiv.org/pdf/2603.02650)**

> **作者:** Yuan Lu; Dongqi Han; Yansen Wang; Dongsheng Li
>
> **摘要:** Diffusion planners are a strong approach for offline reinforcement learning, but they can fail when value-guided selection favours trajectories that score well yet are locally inconsistent with the environment dynamics, resulting in brittle execution. We propose Self-supervised Action Gating with Energies (SAGE), an inference-time re-ranking method that penalises dynamically inconsistent plans using a latent consistency signal. SAGE trains a Joint-Embedding Predictive Architecture (JEPA) encoder on offline state sequences and an action-conditioned latent predictor for short horizon transitions. At test time, SAGE assigns each sampled candidate an energy given by its latent prediction error and combines this feasibility score with value estimates to select actions. SAGE can integrate into existing diffusion planning pipelines that can sample trajectories and select actions via value scoring; it requires no environment rollouts and no policy re-training. Across locomotion, navigation, and manipulation benchmarks, SAGE improves the performance and robustness of diffusion planners.
>
---
#### [replaced 029] Contrastive Representation Regularization for Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在提升模型对机器人信号的敏感性。通过引入RS-CL损失函数，增强控制相关表示学习，提高机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2510.01711](https://arxiv.org/pdf/2510.01711)**

> **作者:** Taeyoung Kim; Jimin Lee; Myungkyu Koo; Dongyoung Kim; Kyungmin Lee; Changyeon Kim; Younggyo Seo; Jinwoo Shin
>
> **备注:** ICML 2026
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong capabilities in robot manipulation by leveraging rich representations from pre-trained Vision-Language Models (VLMs). However, their representations arguably remain suboptimal, lacking sensitivity to robotic signals such as control actions and proprioceptive information. To address the issue, we introduce Robot State-aware Contrastive Loss (RS-CL), a simple and effective representation regularization for VLA models, designed to bridge the gap between VLM representations and robotic signals. In particular, RS-CL aligns the representations more closely with the robot's proprioceptive states by using relative distances between the states as soft supervision. Complementing the original action prediction objective, RS-CL enhances control-relevant representation learning, while being lightweight and fully compatible with standard VLA training pipelines. Our empirical results demonstrate that RS-CL substantially improves the performance of state-of-the-art VLA models; it pushes the prior art to 69.7% achieving the state-of-the-art performance on the RoboCasa-Kitchen benchmark, and boosts success rates from 45.0% to 58.3% on challenging real-robot manipulation tasks.
>
---
#### [replaced 030] RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于4D场景重建任务，旨在解决动态环境中SLAM的不确定性与模糊图像问题。通过引入时间因素和不确定性感知，提升动态场景的重建与跟踪精度。**

- **链接: [https://arxiv.org/pdf/2602.20807](https://arxiv.org/pdf/2602.20807)**

> **作者:** Yangfan Zhao; Hanwei Zhang; Ke Huang; Qiufeng Wang; Zhenzhou Shao; Dengyu Wu
>
> **摘要:** Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: this https URL
>
---
#### [replaced 031] State-Conditional Adversarial Learning: An Off-Policy Visual Domain Transfer Method for End-to-End Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于视觉域迁移任务，解决目标域数据稀缺且非策略的模仿学习问题。提出SCAL方法，通过状态条件对抗学习实现高效迁移。**

- **链接: [https://arxiv.org/pdf/2512.05335](https://arxiv.org/pdf/2512.05335)**

> **作者:** Yuxiang Liu; Shengfan Cao
>
> **摘要:** We study visual domain transfer for end-to-end imitation learning in a realistic and challenging setting where target-domain data are strictly off-policy, expert-free, and scarce. We first provide a theoretical analysis showing that the target-domain imitation loss can be upper bounded by the source-domain loss plus a state-conditional latent KL divergence between source and target observation models. Guided by this result, we propose State- Conditional Adversarial Learning, an off-policy adversarial framework that aligns latent distributions conditioned on system state using a discriminator-based estimator of the conditional KL term. Experiments on visually diverse autonomous driving environments built on the BARC-CARLA simulator demonstrate that SCAL achieves robust transfer and strong sample efficiency.
>
---
#### [replaced 032] Dynamic Entropy Tuning in Reinforcement Learning Low-Level Quadcopter Control: Stochasticity vs Determinism
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文研究强化学习中动态熵调节对四旋翼飞行器低层控制的影响，比较随机与确定性策略。旨在提升控制效果，解决探索效率和灾难性遗忘问题。**

- **链接: [https://arxiv.org/pdf/2512.18336](https://arxiv.org/pdf/2512.18336)**

> **作者:** Youssef Mahran; Zeyad Gamal; Ayman El-Badawy
>
> **备注:** This is the Author Accepted Manuscript version of a paper accepted for publication. The final published version is available via IEEE Xplore
>
> **摘要:** This paper explores the impact of dynamic entropy tuning in Reinforcement Learning (RL) algorithms that train a stochastic policy. Its performance is compared against algorithms that train a deterministic one. Stochastic policies optimize a probability distribution over actions to maximize rewards, while deterministic policies select a single deterministic action per state. The effect of training a stochastic policy with both static entropy and dynamic entropy and then executing deterministic actions to control the quadcopter is explored. It is then compared against training a deterministic policy and executing deterministic actions. For the purpose of this research, the Soft Actor-Critic (SAC) algorithm was chosen for the stochastic algorithm while the Twin Delayed Deep Deterministic Policy Gradient (TD3) was chosen for the deterministic algorithm. The training and simulation results show the positive effect the dynamic entropy tuning has on controlling the quadcopter by preventing catastrophic forgetting and improving exploration efficiency.
>
---
#### [replaced 033] Reinforcement Learning Position Control of a Quadrotor Using Soft Actor-Critic (SAC)
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于无人机控制任务，旨在解决传统RPM控制效率低的问题。通过RL方法直接控制推力矢量，提升路径跟踪性能。**

- **链接: [https://arxiv.org/pdf/2512.18333](https://arxiv.org/pdf/2512.18333)**

> **作者:** Youssef Mahran; Zeyad Gamal; Ayman El-Badawy
>
> **备注:** This is the Author Accepted Manuscript version of a paper accepted for publication. The final published version is available via IEEE Xplore
>
> **摘要:** This paper proposes a new Reinforcement Learning (RL) based control architecture for quadrotors. With the literature focusing on controlling the four rotors' RPMs directly, this paper aims to control the quadrotor's thrust vector. The RL agent computes the percentage of overall thrust along the quadrotor's z-axis along with the desired Roll ($\phi$) and Pitch ($\theta$) angles. The agent then sends the calculated control signals along with the current quadrotor's Yaw angle ($\psi$) to an attitude PID controller. The PID controller then maps the control signals to motor RPMs. The Soft Actor-Critic algorithm, a model-free off-policy stochastic RL algorithm, was used to train the RL agents. Training results show the faster training time of the proposed thrust vector controller in comparison to the conventional RPM controllers. Simulation results show smoother and more accurate path-following for the proposed thrust vector controller.
>
---
#### [replaced 034] BlueME: Robust Underwater Robot-to-Robot Communication Using Compact Magnetoelectric Antennas
- **分类: cs.RO; eess.SP**

- **简介: 论文介绍BlueME系统，一种用于水下机器人通信的紧凑磁电天线阵列。该任务旨在解决水下通信难题，通过VLF信号实现远距离、低功耗可靠传输。工作包括设计、仿真、制造及实地测试。**

- **链接: [https://arxiv.org/pdf/2411.09241](https://arxiv.org/pdf/2411.09241)**

> **作者:** Mehron Talebi; Sultan Mahmud; Adam Khalifa; Md Jahidul Islam
>
> **摘要:** We present the design, development, and experimental validation of BlueME, a compact magnetoelectric (ME) antenna array system for underwater robot-to-robot communication. BlueME employs ME antennas operating at their natural mechanical resonance frequency to efficiently transmit and receive very-low-frequency (VLF) electromagnetic signals underwater. We outline the design, simulation, fabrication, and integration of the proposed system on low-power embedded platforms, focusing on portable and scalable applications. For performance evaluation, we deployed BlueME on an autonomous surface vehicle (ASV) and a remotely operated vehicle (ROV) in open-water field trials. Ocean trials demonstrate that BlueME maintains reliable signal transmission at distances beyond 700 meters while consuming only 10 watts of power. Field trials show that the system operates effectively in challenging underwater conditions such as turbidity, obstacles, and multipath interference -- conditions that generally affect acoustics and optics. Our analysis also examines the impact of complete submersion on system performance and identifies key deployment considerations. This work represents the first practical underwater deployment of ME antennas outside the laboratory and implements the largest VLF ME array system to date. BlueME demonstrates significant potential for marine robotics and automation in multi-robot cooperative systems and remote sensor networks.
>
---
#### [replaced 035] LeARN: Learnable and Adaptive Representations for Nonlinear Dynamics in System Identification
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于系统识别任务，旨在解决非线性动态系统建模问题。提出LeARN框架，通过数据学习基函数，无需领域知识，实现自适应系统建模。**

- **链接: [https://arxiv.org/pdf/2412.12036](https://arxiv.org/pdf/2412.12036)**

> **作者:** Arunabh Singh; Joyjit Mukherjee
>
> **备注:** This work has been accepted at the 34th Mediterranean Conference on Control and Automation (MED 2026)
>
> **摘要:** System identification, the process of deriving mathematical models of dynamical systems from observed input-output data, has undergone a paradigm shift with the advent of learning-based methods. Addressing the intricate challenges of data-driven discovery in nonlinear dynamical systems, these methods have garnered significant attention. Among them, Sparse Identification of Nonlinear Dynamics (SINDy) has emerged as a transformative approach, distilling complex dynamical behaviors into interpretable linear combinations of basis functions. However, SINDy's reliance on domain-specific expertise to construct its foundational 'library' of basis functions limits its adaptability and universality. In this work, we introduce a nonlinear system identification framework LeARN that transcends the need for prior domain knowledge by learning the library of basis functions directly from data. To enhance adaptability to evolving system dynamics under varying noise conditions, we employ a novel meta-learning-based system identification approach that utilizes a light-weight Deep Neural Network (DNN) to dynamically refine these basis functions. This not only captures intricate system behaviors but also adapts effectively to new dynamical regimes. We validate our framework on the Neural Fly dataset, showcasing its robust adaptation and generalization capabilities. Despite its simplicity, our LeARN achieves competitive dynamical error performance to SINDy. This work presents a step towards autonomous discovery of dynamical systems, paving the way for a future where machine learning uncovers the governing principles of complex systems without requiring extensive domain-specific interventions.
>
---
#### [replaced 036] AffordGen: Generating Diverse Demonstrations for Generalizable Object Manipulation with Afford Correspondence
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决因数据多样性不足导致的泛化能力差问题。通过生成多样化的操作轨迹，提升机器人学习的数据效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.10579](https://arxiv.org/pdf/2604.10579)**

> **作者:** Jiawei Zhang; Kaizhe Hu; Yingqian Huang; Yuanchen Ju; Zhengrong Xue; Huazhe Xu
>
> **摘要:** Despite the recent success of modern imitation learning methods in robot manipulation, their performance is often constrained by geometric variations due to limited data diversity. Leveraging powerful 3D generative models and vision foundation models (VFMs), the proposed AffordGen framework overcomes this limitation by utilizing the semantic correspondence of meaningful keypoints across large-scale 3D meshes to generate new robot manipulation trajectories. This large-scale, affordance-aware dataset is then used to train a robust, closed-loop visuomotor policy, combining the semantic generalizability of affordances with the reactive robustness of end-to-end learning. Experiments in simulation and the real world show that policies trained with AffordGen achieve high success rates and enable zero-shot generalization to truly unseen objects, significantly improving data efficiency in robot learning. Project Page: this https URL
>
---
#### [replaced 037] SpaceTools: Tool-Augmented Spatial Reasoning via Double Interactive RL
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于空间推理任务，旨在解决VLM在精确空间推理上的不足。通过引入DIRL框架，使模型协调多工具进行交互学习，提升空间理解与实际操作能力。**

- **链接: [https://arxiv.org/pdf/2512.04069](https://arxiv.org/pdf/2512.04069)**

> **作者:** Siyi Chen; Mikaela Angelina Uy; Chan Hee Song; Faisal Ladhak; Adithyavairavan Murali; Qing Qu; Stan Birchfield; Valts Blukis; Jonathan Tremblay
>
> **备注:** CVPR 2026
>
> **摘要:** Vision Language Models (VLMs) demonstrate strong qualitative visual understanding, but struggle with metrically precise spatial reasoning required for embodied applications. The agentic paradigm promises that VLMs can use a wide variety of tools that could augment these capabilities, such as depth estimators, segmentation models, and pose estimators. Yet it remains an open challenge how to realize this vision without solely relying on handcrafted prompting strategies or enforcing fixed, predefined tool pipelines that limit VLMs' ability to discover optimal tool-use patterns. Reinforcement Learning could overcome this gap, but has so far been limited to reasoning with a single visual tool due to the large search space in multi-tool reasoning. We introduce Double Interactive Reinforcement Learning (DIRL), a two-phase training framework where VLMs learn to coordinate multiple tools through interactive exploration and feedback. In the teaching phase, we combine demonstrations from a single tool specialist trained via interactive RL with traces from a frontier model using all tools. In the exploration phase, the model further refines multi-tool coordination through continued RL. Our model, SpaceTools, with tool-augmented spatial reasoning ability, achieves state-of-the-art performance on spatial understanding benchmarks (RoboSpatial-Home, BLINK, BOP-ASK) and demonstrates reliable real-world manipulation using a 7-DOF robot as a tool. DIRL provides substantial improvements over the vanilla SFT (+12% on RoboSpatial) and RL (+16% on RoboSpatial) baselines. Project page: this https URL.
>
---
#### [replaced 038] Approximate Imitation Learning for Event-based Quadrotor Flight in Cluttered Environments
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，解决事件相机在机器人学习中因高频率数据模拟导致的计算成本过高问题。通过分离表示学习与策略搜索，提升训练效率并实现高效飞行控制。**

- **链接: [https://arxiv.org/pdf/2603.07578](https://arxiv.org/pdf/2603.07578)**

> **作者:** Nico Messikommer; Jiaxu Xing; Leonard Bauersfeld; Marco Cannici; Elie Aljalbout; Davide Scaramuzza
>
> **摘要:** Event cameras offer high temporal resolution and low latency, making them ideal sensors for high-speed robotic applications where conventional cameras suffer from motion blur. However, their widespread adoption in robot learning is severely bottlenecked by the computational cost of simulating high-frequency event data during online training. In this work, we present Approximate Imitation Learning, a novel framework that fundamentally resolves this bottleneck, reducing policy training time for complex, agile drone flight from 52.44 hours to just 1.86 hours - a 28x computational speedup. Our key insight is to separate representation learning from policy search. We first leverage a large-scale offline dataset to learn a task-specific representation space. Subsequently, the policy is fine-tuned through online interactions that rely solely on lightweight state information, completely eliminating the need to render events during the active policy search phase. This training paradigm drastically reduces development overhead and enables event-based control policies to scale to complex environments. Furthermore, our approach eliminates the reliance on standard cameras or intermediate representations during deployment, mapping events directly to control commands. In simulation, our method matches or exceeds the performance of standard imitation learning baselines that require full online event rendering. Finally, we successfully validate the framework in the real world, demonstrating that a policy trained via this ultra-efficient paradigm enables a quadrotor to fly through highly cluttered environments at remarkable speeds of up to 9.8 m/s.
>
---
#### [replaced 039] See, Plan, Rewind: Progress-Aware Vision-Language-Action Models for Robust Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在提升机器人执行复杂任务的鲁棒性。通过引入SPR框架，将语言指令转化为可操作的子目标，实现任务进度感知与错误恢复。**

- **链接: [https://arxiv.org/pdf/2603.09292](https://arxiv.org/pdf/2603.09292)**

> **作者:** Tingjun Dai; Mingfei Han; Tingwen Du; Zhiheng Liu; Zihao Zhang; Zhihui Li; Salman Khan; Jun Yu; Xiaojun Chang
>
> **备注:** Suggested to CVPR Findings. this https URL
>
> **摘要:** Measurement of task progress through explicit, actionable milestones is critical for robust robotic manipulation. This progress awareness enables a model to ground its current task status, anticipate verifiable intermediate states, and detect and recover from failures when progress stalls. To embody this capability, we introduce \textbf{S}ee, \textbf{P}lan, \textbf{R}ewind (SPR), a progress-aware vision-language-action framework that dynamically grounds language instructions into a sequence of spatial subgoals. SPR operates through a continuous core cycle, Seeing the current state and upcoming milestone, Planning a trajectory towards the next 2D waypoint, and Rewinding to a recoverable state upon failure by monitoring progress against the expected sequence. This closed-loop approach enables robust error correction without requiring additional training data or auxiliary models. Extensive experiments demonstrate the framework's effectiveness, generalization and robustness: SPR outperforms the MolmoAct baseline by 5\% on the LIBERO benchmark. On the challenging LIBERO-Plus benchmark with unseen instructions and initial states, SPR achieves state-of-the-art robustness with the smallest performance drop, surpassing OpenVLA-OFT and UniVLA, demonstrating superior out-of-distribution robustness.
>
---
#### [replaced 040] ExpertGen: Scalable Sim-to-Real Expert Policy Learning from Imperfect Behavior Priors
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ExpertGen，解决机器人行为克隆中高质量数据获取困难的问题。通过模拟学习实现可扩展的端到端策略迁移，提升真实环境下的任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.15956](https://arxiv.org/pdf/2603.15956)**

> **作者:** Zifan Xu; Ran Gong; Maria Vittoria Minniti; Kausik Sivakumar; Ahmet Salih Gundogdu; Eric Rosen; Riedana Yan; Tushar Kusnur; Zixing Wang; Di Deng; Peter Stone; Xiaohan Zhang; Karl Schmeckpeper
>
> **摘要:** Learning generalizable and robust behavior cloning policies requires large volumes of high-quality robotics data. While human demonstrations (e.g., through teleoperation) serve as the standard source for expert behaviors, acquiring such data at scale in the real world is prohibitively expensive. This paper introduces ExpertGen, a framework that automates expert policy learning in simulation to enable scalable sim-to-real transfer. ExpertGen first initializes a behavior prior using a diffusion policy trained on imperfect demonstrations, which may be synthesized by large language models or provided by humans. Reinforcement learning is then used to steer this prior toward high task success by optimizing the diffusion model's initial noise while keep original policy frozen. By keeping the pretrained diffusion policy frozen, ExpertGen regularizes exploration to remain within safe, human-like behavior manifolds, while also enabling effective learning with only sparse rewards. Empirical evaluations on challenging manipulation benchmarks demonstrate that ExpertGen reliably produces high-quality expert policies with no reward engineering. On industrial assembly tasks, ExpertGen achieves a 90.5% overall success rate, while on long-horizon manipulation tasks it attains 85% overall success, outperforming all baseline methods. The resulting policies exhibit dexterous control and remain robust across diverse initial configurations and failure states. To validate sim-to-real transfer, the learned state-based expert policies are further distilled into visuomotor policies via DAgger and successfully deployed on real robotic hardware.
>
---
#### [replaced 041] GuidedVLA: Specifying Task-Relevant Factors via Plug-and-Play Action Attention Specialization
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型因依赖隐式监督而过拟合的问题。通过引入GuidedVLA框架，显式引导动作解码器关注任务相关因素，提升模型泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.12369](https://arxiv.org/pdf/2605.12369)**

> **作者:** Xiaosong Jia; Bowen Yang; Zuhao Ge; Xian Nie; Yuchen Zhou; Cunxin Fan; Yufeng Li; Yilin Chai; Chao Jing; Zijian Liang; Qingwen Bu; Haidong Cao; Chao Wu; Qifeng Li; Zhenjie Yang; Chenhe Zhang; Hongyang Li; Zuxuan Wu; Junchi Yan; Yu-Gang Jiang
>
> **备注:** Accepted to RSS 2026. Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models aim for general robot learning by aligning action as a modality within powerful Vision-Language Models (VLMs). Existing VLAs rely on end-to-end supervision to implicitly enable the action decoding process to learn task-relevant features. However, without explicit guidance, these models often overfit to spurious correlations, such as visual shortcuts or environmental noise, limiting their generalization. In this paper, we introduce GuidedVLA, a framework designed to manually guide the action generation to focus on task-relevant factors. Our core insight is to treat the action decoder not as a monolithic learner, but as an assembly of functional components. Individual attention heads are supervised by manually defined auxiliary signals to capture distinct factors. As an initial study, we instantiate this paradigm with three specialized heads: object grounding, spatial geometry, and temporal skill logic. Across simulation and real-robot experiments, GuidedVLA improves success rates in both in-domain and out-of-domain settings compared to strong VLA baselines. Finally, we show that the quality of these specialized factors correlates positively with task performance and that our mechanism yields decoupled, high-quality features. Our results suggest that explicitly guiding action-decoder learning is a promising direction for building more robust and general VLA models.
>
---
#### [replaced 042] DAG-Plan: Generating Directed Acyclic Dependency Graphs for Dual-Arm Cooperative Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DAG-Plan框架，用于双臂机器人任务规划。解决传统方法在处理复杂依赖和并行执行上的不足，通过DAG结构高效管理任务依赖，提升规划成功率和效率。**

- **链接: [https://arxiv.org/pdf/2406.09953](https://arxiv.org/pdf/2406.09953)**

> **作者:** Zeyu Gao; Yao Mu; Jinye Qu; Mengkang Hu; Shijia Peng; Chengkai Hou; Lingyue Guo; Ping Luo; Shanghang Zhang; Yanfeng Lu
>
> **备注:** ICRA 2026
>
> **摘要:** Dual-arm robots promise greater efficiency but require planning for complex tasks with nonlinear sub-task dependencies. Current methods using Large Language Models (LLMs) suffer from a fundamental trade-off: generating linear sequences is efficient but fails to model parallelism and adapt to changes, while iterative querying is adaptive but too slow and costly. To bridge this gap, we introduce DAG-Plan, a novel task planning framework that for the first time employs a Directed Acyclic Graph (DAG) as the central representation for dual-arm coordination. The key insight is that a DAG natively captures complex sub-task dependencies and explicitly reveals opportunities for parallel execution. Within this framework, an LLM is used only once as a powerful semantic parser to translate a natural language instruction into a structured DAG. During execution, our system dynamically assigns candidate nodes to the suitable arm based on real-time environmental observations, enabling truly adaptive and parallel operation. Extensive evaluation on a dual-arm kitchen benchmark shows that DAG-Plan's structured approach fundamentally outperforms existing paradigms. It achieves a 48% higher success rate than single-query linear sequence methods with dual arm by robustly managing dependencies, and an 84.1% higher execution efficiency than iterative querying methods by eliminating the latency of repeated LLM calls. Our work demonstrates that a principled, graph-based representation is the key to unlocking efficient and reliable LLM-based planning for complex robotic systems. More demos and code are available on this https URL.
>
---
#### [replaced 043] A Predictive Control Strategy to Offset-Point Tracking for Agricultural Mobile Robots
- **分类: cs.RO**

- **简介: 该论文属于农业机器人路径跟踪任务，解决传统控制器忽略附属设备位置导致的跟踪误差问题。提出一种预测控制策略，建模附属设备为刚性偏移点，提升跟踪精度与作业安全性。**

- **链接: [https://arxiv.org/pdf/2603.28439](https://arxiv.org/pdf/2603.28439)**

> **作者:** Stephane Ngnepiepaye Wembe; Vincent Rousseau; Johann Laconte; Roland Lenain
>
> **备注:** Accepted in the journal IEEE Transaction on Field Robotics
>
> **摘要:** Robots are increasingly being deployed in agriculture to support sustainable practices and improve productivity. They offer strong potential to enable precise, efficient, and environmentally friendly operations. However, most existing path-following controllers focus solely on the robot's center of motion and neglect the spatial footprint and dynamics of attached implements. In practice, implements such as mechanical weeders or spring-tine cultivators are often large, rigidly mounted, and directly interacting with crops and soil; ignoring their position can degrade tracking performance and increase the risk of crop damage. To address this limitation, we propose a closed-form predictive control strategy extending the approach introduced in [1]. The method is developed specifically for Ackermann-type agricultural vehicles and explicitly models the implement as a rigid offset point, while accounting for lateral slip and lever-arm effects. The approach is benchmarked against state-of-the-art baseline controllers, including a reactive geometric method, a reactive backstepping method, and a model-based predictive scheme. Real-world agricultural experiments with two different implements show that the proposed method reduces the median tracking error by 24% to 56%, and decreases peak errors during curvature transitions by up to 70%. These improvements translate into enhanced operational safety, particularly in scenarios where the implement operates in close proximity to crop rows.
>
---
#### [replaced 044] GIFT: Geometry-Induced Functional Transfer for Category-level Object Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决不熟悉物体在新环境中的技能迁移问题。通过几何引导的功能迁移框架，实现从人类示范中学习并泛化操作技能。**

- **链接: [https://arxiv.org/pdf/2503.15371](https://arxiv.org/pdf/2503.15371)**

> **作者:** Cristiana de Farias; Luis Figueredo; Riddhiman Laha; Maxime Adjigble; Brahim Tamadazte; Rustam Stolkin; Sami Haddadin; Naresh Marturi
>
> **备注:** 8 pages, 6 figures. ICRA 2026
>
> **摘要:** Robotic manipulation of unfamiliar objects in new environments is challenging due to limited generalisation capabilities. We propose a new skill transfer framework, GIFT (Geometry-Induced Functional Transfer), which enables a robot to transfer complex object manipulation skills and constraints from a single human demonstration. Our approach addresses the challenge of skill acquisition and task execution by deriving geometric representations from demonstrations focusing on object-centric interactions. By leveraging the Functional Maps (FMC) framework, we efficiently map interaction functions between objects and their environments, allowing the robot to replicate task operations across objects of similar topologies or categories, even when they have significantly different shapes. Additionally, our method incorporates screw interpolation (ScLERP) for generating smooth, geometrically-aware robot paths to ensure the transferred skills adhere to the demonstrated task constraints. We validate the effectiveness and adaptability of our approach through extensive experiments, demonstrating successful skill transfer and task execution in diverse real-world environments without requiring additional training.
>
---
#### [replaced 045] DeepIPCv2: LiDAR-powered Robust Environmental Perception and Navigational Control for Autonomous Vehicle
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出DeepIPCv2，用于自动驾驶的环境感知与控制，解决光照变化和动作不平衡问题。融合LiDAR数据与控制学习，提升驾驶鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2307.06647](https://arxiv.org/pdf/2307.06647)**

> **作者:** Oskar Natan; Jun Miura
>
> **备注:** This work has been accepted for publication in IEEE Access. this https URL
>
> **摘要:** We propose DeepIPCv2, an end-to-end autonomous driving framework that integrates LiDAR-based environmental perception with command-specific control learning. Unlike prior camera-reliant models, DeepIPCv2 employs point cloud segmentation and multi-view projection to construct robust scene representations. These features are fused and decoded through a combination of gated recurrent units, command-specific multi-layer perceptrons, and PID controllers to estimate both waypoints and navigational control commands. This design enhances maneuverability and addresses action imbalance in driving datasets. To validate the model, we constructed a dataset covering diverse illumination conditions and conducted ablation studies and comparative tests against recent methods, including TransFuser. Results demonstrate that DeepIPCv2 achieves the lowest total metric error and the fewest driving interventions, highlighting both its robustness to illumination changes and its improved control accuracy. By releasing the codes at this https URL later, we aim to support reproducibility and future advancements in end-to-end autonomous driving research.
>
---
#### [replaced 046] SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on Vision-Language-Action Models
- **分类: cs.CR; cs.AI; cs.RO**

- **简介: 该论文属于安全领域，针对VLA模型的后门攻击问题，提出SILENTDRIFT方法，利用动作分块漏洞实现隐蔽攻击，提升攻击成功率并保持任务性能。**

- **链接: [https://arxiv.org/pdf/2601.14323](https://arxiv.org/pdf/2601.14323)**

> **作者:** Bingxin Xu; Yuzhang Shang; Binghui Wang; Emilio Ferrara
>
> **备注:** Accepted to ACL Findings 2026
>
> **摘要:** Vision-Language-Action (VLA) models are increasingly deployed in safety-critical robotic applications, yet their security vulnerabilities remain underexplored. We identify a fundamental security flaw in modern VLA systems: the combination of action chunking and delta pose representations creates an intra-chunk visual open-loop. This mechanism forces the robot to execute K-step action sequences, allowing per-step perturbations to accumulate through integration. We propose SILENTDRIFT, a stealthy black-box backdoor attack exploiting this vulnerability. Our method employs the Smootherstep function to construct perturbations with guaranteed C2 continuity, ensuring zero velocity and acceleration at trajectory boundaries to satisfy strict kinematic consistency constraints. Furthermore, our keyframe attack strategy selectively poisons only the critical approach phase, maximizing impact while minimizing trigger exposure. The resulting poisoned trajectories are visually indistinguishable from successful demonstrations. Evaluated on the LIBERO, SILENTDRIFT achieves a 93.2% Attack Success Rate with a poisoning rate under 2%, while maintaining a 95.3% Clean Task Success Rate.
>
---
#### [replaced 047] Learning Transferable Motor Skills for Geometry-Aware Robotic Surface Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人表面操作任务，解决几何规划与运动执行脱节的问题。通过模块化框架分离几何路径与运动规则，提升任务迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.24881](https://arxiv.org/pdf/2605.24881)**

> **作者:** Miroslav David; Karla Stepanova; Robert Babuska
>
> **备注:** In: Workshop on Geometry in the Age of Data-Driven Robotics, ICRA 2026, Vienna, 2026
>
> **摘要:** Robotic surface-interaction tasks, such as spray painting or welding, require both accurate geometric planning and precise motion execution. While modern motion planners generate valid geometric paths, they often lack the expert motor patterns observed in human operators. Conversely, learning from demonstration often tightly couples task execution to the specific training geometry, limiting transferability. We propose a modular framework that decouples geometric motion planning from execution-level expertise. Expert behavior is represented as a vocabulary of interpretable, atomic motor rules, such as velocity scaling and orientation offsets, that systematically modify a geometrically planned reference path. We train a multimodal neural network to infer rule parameters jointly from kinematic trajectory data and CAD model geometry. We evaluate our approach through dynamic simulation on L-shaped and window-shaped objects, demonstrating on simulated data that the model successfully extracts velocity and orientation rules across both topologies.
>
---
#### [replaced 048] TRANS: Terrain-aware Reinforcement Learning for Agile Navigation of Quadruped Robots under Social Interactions
- **分类: cs.RO**

- **简介: 该论文属于四足机器人社交导航任务，解决动态环境中地形感知与社交交互问题。提出TRANS框架，结合强化学习实现高效导航。**

- **链接: [https://arxiv.org/pdf/2602.12724](https://arxiv.org/pdf/2602.12724)**

> **作者:** Wei Zhu; Irfan Tito Kurniawan; Ye Zhao; Mitsuhiro Hayashibe
>
> **摘要:** This study introduces TRANS: Terrain-aware Reinforcement learning for Agile Navigation under Social interactions, a deep reinforcement learning (DRL) framework for quadrupedal social navigation over unstructured terrains. Conventional quadrupedal navigation typically separates motion planning from locomotion control, neglecting whole-body constraints and terrain awareness. On the other hand, end-to-end methods are more integrated but require high-frequency sensing, which is often noisy and computationally costly. In addition, most existing approaches assume static environments, limiting their use in human-populated settings. To address these limitations, we propose a two-stage training framework with three DRL pipelines. (1) TRANS-Loco employs an asymmetric actor-critic (AC) model for quadrupedal locomotion, enabling traversal of uneven terrains without explicit terrain or contact observations. (2) TRANS-Nav applies a symmetric AC framework for social navigation, directly mapping transformed LiDAR data to ego-agent actions under differential-drive kinematics. (3) A unified pipeline, TRANS, integrates TRANS-Loco and TRANS-Nav, supporting terrain-aware quadrupedal navigation in uneven and socially interactive environments. Comprehensive benchmarks against locomotion and social navigation baselines demonstrate the effectiveness of TRANS. Hardware experiments further confirm its potential for sim-to-real transfer.
>
---
#### [replaced 049] MiNI-Q: A Miniature, Wire-Free Quadruped with Unbounded, Independently Actuated Leg Joints
- **分类: cs.RO**

- **简介: 该论文介绍了一种微型无线四足机器人MiNI-Q，解决腿部关节运动受限问题。通过设计无机械限制的2-DOF腿关节，实现多种运动方式，提升灵活性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.11537](https://arxiv.org/pdf/2603.11537)**

> **作者:** Daniel Koh; Suraj Shah; Yufeng Wu; Dennis Hong
>
> **备注:** 7 pages, 11 figures. Submitted to the IEEE RAS Conference on Ubiquitous Robots (UR 2026)
>
> **摘要:** Physical joint limits are common in legged robots and can restrict workspace, constrain gait design, and increase the risk of hardware damage. This paper introduces MiNI-Q^2, a miniature, wire-free quadruped robot with independently actuated, mechanically unbounded 2-DOF leg joints. We present the mechanical design, kinematic analysis, and experimental validation of the proposed robot. The leg mechanism enables both oscillatory gaits and rotary locomotion while allowing the robot to fold to a minimum height of 2.5 cm. Experimentally, MiNI-Q achieves speeds up to 0.46 m/s and demonstrates low-clearance crawling, stair climbing, inverted locomotion, jumping, and backflipping. The wire-free architecture extends our previous Q8bot design, improving assembly reliability at miniature scale. All mechanical and electrical design files are released open source to support reproducibility and further research.
>
---
#### [replaced 050] SpeedAug: Policy Acceleration via Tempo-Enriched Policy and RL Fine-Tuning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对机器人操作任务中的策略加速问题，提出SpeedAug框架，通过强化学习优化执行节奏，提升任务效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.00062](https://arxiv.org/pdf/2512.00062)**

> **作者:** Taewook Nam; Junmo Cho; Youngsoo Jang; Sung Ju Hwang
>
> **摘要:** Robotic policy learning for complex real-world manipulation tasks has seen rapid recent progress, enabled in large part by the ability to collect demonstrations through human operation. However, policies trained from such demonstrations often execute tasks far more slowly than the robot's physical capabilities, as demonstration data is collected under practical constraints that favor conservative, success-oriented trajectories over execution speed. Existing policy acceleration methods determine execution tempo through data preprocessing or heuristic rules, rather than learning execution speed optimized for the task. In this paper, we propose SpeedAug, a policy acceleration framework that enables policies to learn task-optimal execution tempo via reinforcement learning (RL). SpeedAug first learns a tempo-enriched prior policy from speed-augmented demonstrations that captures diverse execution tempos. Building on this tempo-enriched prior, RL fine-tuning guides exploration to refine action trajectories and optimize execution tempo efficiently. Experiments on robotic manipulation benchmarks demonstrate that SpeedAug substantially improves the sample efficiency of policy acceleration while maintaining high success rates, achieving fast and stable task execution. Applied to a real-world manipulation task, SpeedAug improves task throughput by 1.8x using only 16 minutes of online interactions without compromising the success rate.
>
---
#### [replaced 051] DIPOLE: Fusing Vision and Geometry for Robust Visuomotor Generalization
- **分类: cs.RO**

- **简介: 该论文提出DIPOLE模型，解决视觉-几何融合问题，提升机器人在不同视觉条件下的运动泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22445](https://arxiv.org/pdf/2511.22445)**

> **作者:** Yikai Tang; Haoran Geng; Jindou Jia; Yuxuan Hu; Sheng Zang; Jianfei Yang; Pieter Abbeel; Jitendra Malik
>
> **摘要:** Imitation learning has emerged as a crucial approach for acquiring visuomotor skills from demonstrations, where designing effective observation encoders is essential for policy generalization. However, existing methods tend to struggle once test-time conditions differ from the demonstrations, such as changes in lighting, texture, viewpoint, object placement, or object identity. To address this challenge, we propose DIffusion POlicy with compLementarity Encoders (DIPOLE), a visuomotor policy that learns to fuse complementary modalities through a training-time mechanism rather than a specialized fusion architecture. A modality-wise dropout masks one branch at each training step, encouraging each modality to remain individually informative. A lightweight cross-attention layer then exchanges complementary cues between the two. This design endows DIPOLE with five core strengths: stable high performance across diverse tasks, robustness to visual changes, spatial generalization at sub-centimeter precision, emergent capability beyond either modality, and zero-shot transfer to unseen objects. Across 18 simulated and 4 real-world tasks, DIPOLE outperforms six baselines by 39.1% on average, with gains of 41.5% under unseen visual distractors and 15.2% under randomized object placement.
>
---
#### [replaced 052] CLAW: A Vision-Language-Action Framework for Weight-Aware Robotic Grasping
- **分类: cs.RO**

- **简介: 该论文提出CLAW框架，解决机器人抓取中的重量感知问题。通过分离条件判断与动作生成，结合视觉语言和动作控制，提升任务准确性。**

- **链接: [https://arxiv.org/pdf/2509.14143](https://arxiv.org/pdf/2509.14143)**

> **作者:** Zijian An; Ran Yang; Yiming Feng; Lifeng Zhou
>
> **备注:** 8 pages, 5 figures, Video: this https URL
>
> **摘要:** Vision-language-action (VLA) models have recently emerged as a promising paradigm for robotic control, enabling end-to-end policies that ground natural language instructions into visuomotor actions. However, current VLAs often struggle to satisfy precise task constraints, such as stopping based on numeric thresholds, since their observation-to-action mappings are implicitly shaped by training data and lack explicit mechanisms for condition monitoring. In this work, we propose CLAW (CLIP-Language-Action for Weight), a framework that decouples condition evaluation from action generation. CLAW leverages a fine-tuned CLIP model as a lightweight prompt generator, which continuously monitors the digital readout of a scale and produces discrete directives based on task-specific weight thresholds. These prompts are then consumed by $\pi_0$, a flow-based VLA policy, which integrates the prompts with multi-view camera observations to produce continuous robot actions. This design enables CLAW to combine symbolic weight reasoning with high-frequency visuomotor control. We validate CLAW on three experimental setups: single-object grasping and mixed-object tasks requiring dual-arm manipulation. Across all conditions, CLAW reliably executes weight-aware behaviors and outperforms both raw-$\pi_0$ and fine-tuned $\pi_0$ models. A video of our paper is available online this https URL.
>
---
#### [replaced 053] Magnetic Indoor Localization through CNN Regression and Rotation Invariance
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于室内定位任务，解决磁力场数据对设备方向敏感的问题。通过引入旋转不变特征和轻量CNN模型，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.22896](https://arxiv.org/pdf/2604.22896)**

> **作者:** Helge Rosé; Konstantin Klipp; Tom Koubek; Bernd Schäufele; Ilja Radusch
>
> **备注:** Published and presented at the 2026 4th International Conference on Mechatronics, Control and Robotics (ICMCR)
>
> **摘要:** Indoor positioning is an essential technology for a wide range of applications in GNSS-denied environments, including indoor navigation and IoT systems. Combining convolutional neural networks (CNNs) and magnetic field-based features offers a low-cost, infrastructure-free solution for precise positioning. While magnetic fingerprints are a promising approach for indoor positioning, models trained on raw 3D magnetometer data are highly sensitive to device orientation. We address this by using two rotation invariant features derived from the 3D magnetic field: the norm (Mn) and the projection onto the gravity axis (Mg). We train a lightweight 7-layer dilated CNN (MagNetS/XL) on magnetic sequences to directly regress (x, y) positions. Using the MagPie dataset (three buildings, handheld trajectories), we systematically evaluate fixed and random rotations of test and/or train data. Raw 3D inputs (Mx, My , Mz) exhibit isotropic error increases under fixed 90° rotations and further degrade with growing random rotations. In contrast, 2D (Mn, Mg) inputs maintain rotation invariant accuracy and surpass the 3D inputs once rotation exceeds building-specific thresholds for three reference buildings: 0° for Loomis (large), 5° for Talbot (medium), and 6° for CSL (small). MagNetXL achieves or exceeds state-of-the-art accuracy on the MagPie dataset, and MagNetS delivers similar performance with roughly one third of the parameters, favoring mobile deployment. These results show that the robustness gained from rotation invariant inputs outweighs the loss of input dimensionality in realistic usage, allowing mapping and localization without orientation alignment or added infrastructure.
>
---
#### [replaced 054] PLanAR: Planning-Language-Grounded Agentic Reasoning for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出PLanAR框架，解决机器人长周期操作中的环境状态推理问题。通过规划语言接口，结合视觉语言模型实现任务规划与执行验证。**

- **链接: [https://arxiv.org/pdf/2602.01662](https://arxiv.org/pdf/2602.01662)**

> **作者:** Pengyuan Guo; Zhonghao Mai; Zhengtong Xu; Kaidi Zhang; Quan Khanh Luu; Heng Zhang; Zichen Miao; Arash Ajoudani; Zachary Kingston; Qiang Qiu; Yu She
>
> **备注:** New version with updated framing, contributions, experiments, and figures
>
> **摘要:** Recent advances in vision-language models (VLMs) have enabled increasing progress in real-world robot manipulation. However, long-horizon manipulation in unstructured environments requires VLMs to reason about changing scene states, action constraints, and execution outcomes, which remains difficult with natural language reasoning alone. We present PLanAR, a planning-language-grounded robot agent framework for open-vocabulary, long-horizon manipulation. PLanAR uses a planning-language interface to define the VLM reasoning space: object predicates represent scene states, action schemas specify robot skills with preconditions and effects, and symbolic plans provide executable intermediate representations. This interface enables stepwise verification: after each action, PLanAR uses onboard observations to check whether the expected symbolic effects have been achieved, allowing the VLM-based agent to update task states, detect failures, and replan when execution deviates from expectation. Across robot embodiments, VLM backends, and tasks including stacking, crossword solving, and long-horizon kitchen workflows, PLanAR demonstrates strong real-world capability while revealing key limitations of current VLMs in embodied reasoning.
>
---
#### [replaced 055] MARFT: Multi-Agent Reinforcement Fine-Tuning
- **分类: cs.MA; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，旨在解决LLM-based多智能体系统（LaMAS）的微调问题。提出MARFT框架，优化异步交互与架构差异，提升系统适应性与协同能力。**

- **链接: [https://arxiv.org/pdf/2504.16129](https://arxiv.org/pdf/2504.16129)**

> **作者:** Junwei Liao; Muning Wen; Jun Wang; Weinan Zhang
>
> **备注:** 37 pages
>
> **摘要:** Large Language Model (LLM)-based Multi-Agent Systems (LaMAS) have demonstrated strong capabilities on complex agentic tasks requiring multifaceted reasoning and collaboration, from high-quality presentation generation to scientific research. Meanwhile, Reinforcement Learning (RL) is widely recognized for enhancing agent intelligence, but limited work has studied fine-tuning LaMAS with foundational RL techniques. Directly applying conventional Multi-Agent Reinforcement Learning (MARL) to LaMAS also introduces major challenges due to the unique mechanisms of LaMAS. To address these challenges, this article presents a comprehensive study of LLM-based MARL and proposes Multi-Agent Reinforcement Fine-Tuning (MARFT). We introduce Flex-MG, a new Markov Game formulation aligned with real-world LaMAS optimization, together with a universal algorithmic framework tailored to LaMAS. We review the evolution from traditional RL to Reinforcement Fine-Tuning (RFT), then analyze the multi-agent counterpart. For LaMAS, we identify key differences between classical MARL and MARFT, including asynchronous agent interactions, profile-aware agent design, and heterogeneous architectures. These differences motivate a LaMAS-oriented formulation of RFT. We present a robust and scalable MARFT framework, detail its modular algorithm, and provide an open-source implementation to support adoption and further research. The paper further discusses application perspectives and open challenges, including dynamic environment modeling, sample inefficiency, and the lack of cohesive frameworks. By connecting theoretical foundations with practical methodology, this work aims to serve as a roadmap for advancing MARFT toward resilient, adaptive, and human-aligned agentic systems. Implementation: this https URL.
>
---
#### [replaced 056] Plan-R1: Safe and Feasible Trajectory Planning as Language Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶轨迹规划任务，旨在解决传统方法依赖专家数据且缺乏安全意识的问题。提出Plan-R1框架，通过两阶段训练提升安全性与可行性。**

- **链接: [https://arxiv.org/pdf/2505.17659](https://arxiv.org/pdf/2505.17659)**

> **作者:** Xiaolong Tang; Meina Kan; Shiguang Shan; Xilin Chen
>
> **摘要:** Safe and feasible trajectory planning is critical for real-world autonomous driving systems. However, existing learning-based planners rely heavily on expert demonstrations, which not only lack explicit safety awareness but also risk inheriting undesirable behaviors such as speeding from suboptimal human driving data. Inspired by the success of large language models, we propose Plan-R1, a two-stage trajectory planning framework that decouples principle alignment from behavior learning. In the first stage, a general trajectory predictor is pre-trained on expert data to capture diverse, human-like driving behaviors. In the second stage, the model is fine-tuned with rule-based rewards using Group Relative Policy Optimization (GRPO), explicitly aligning ego planning with principles such as safety, comfort, and traffic rule compliance. This two-stage paradigm retains human-like behaviors while enhancing safety awareness and discarding undesirable patterns from demonstrations. Furthermore, we identify a key limitation of directly applying GRPO to planning: group-wise normalization erases cross-group scale differences, causing rare, high-variance safety-violation groups to have similar advantages as abundant low-variance safe groups, thereby suppressing optimization for safety-critical objectives. To address this, we propose Variance-Decoupled GRPO (VD-GRPO), which replaces normalization with centering and fixed scaling to preserve absolute reward magnitudes, ensuring that safety-critical objectives remain dominant throughout training. Experiments on the nuPlan benchmark demonstrate that Plan-R1 significantly improves planning safety and feasibility, achieving state-of-the-art performance, particularly in realistic reactive settings. Our code is available at this https URL.
>
---
#### [replaced 057] Scalar-Measurement Attitude Estimation on $\mathbf{SO}(3)$ with Bias Compensation
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于姿态估计任务，解决仅使用标量测量实现可靠姿态估计的问题。提出SO(3)上的非线性观测器，具备陀螺仪偏差补偿，保证稳定性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.02478](https://arxiv.org/pdf/2603.02478)**

> **作者:** Alessandro Melis; Tarek Bouazza; Hassan Alnahhal; Sifeddine Benahmed; Soulaimane Berkane; Tarek Hamel
>
> **备注:** 9 pages, 4 figures. Accepted to ICRA 2026
>
> **摘要:** Attitude estimation methods typically rely on full vector measurements from inertial sensors such as accelerometers and magnetometers. This paper shows that reliable estimation can also be achieved using only scalar measurements, which naturally arise either as components of vector readings or as independent constraints from other sensing modalities. We propose nonlinear deterministic observers on $\mathbf{SO}(3)$ that incorporate gyroscope bias compensation and guarantee uniform local exponential stability under suitable observability conditions. A key feature of the framework is its robustness to partial sensing: accurate estimation is maintained even when only a subset of vector components is available. Experimental validation on the BROAD dataset confirms consistent performance across progressively reduced measurement configurations, with estimation errors remaining small even under severe information loss. To the best of our knowledge, this is the first work to establish fundamental observability results showing that two scalar measurements under suitable excitation suffice for attitude estimation, and that three are enough in the static case. These results position scalar-measurement-based observers as a practical and reliable alternative to conventional vector-based approaches.
>
---
#### [replaced 058] Strategizing at Speed: A Learned Model Predictive Game for Multi-Agent Drone Racing
- **分类: cs.RO; cs.GT**

- **简介: 该论文属于多智能体无人机竞速任务，解决高速下策略制定与交互的问题。通过比较MPG与MPC，提出LMPG以降低延迟并提升性能。**

- **链接: [https://arxiv.org/pdf/2602.06925](https://arxiv.org/pdf/2602.06925)**

> **作者:** Andrei-Carlo Papuc; Lasse Peters; Sihao Sun; Laura Ferranti; Javier Alonso-Mora
>
> **摘要:** Autonomous drone racing pushes the boundaries of high-speed motion planning and multi-agent strategic decision-making. Success in this domain requires drones not only to navigate at their limits but also to anticipate and counteract competitors' actions. In this paper, we study a fundamental question that arises in this domain: how deeply should an agent strategize before taking an action? To this end, we compare two planning paradigms: the Model Predictive Game (MPG), which finds interaction-aware strategies at the expense of longer computation times, and contouring Model Predictive Control (MPC), which computes strategies rapidly but does not reason about interactions. We perform extensive experiments to study this trade-off, revealing that MPG outperforms MPC at moderate velocities but loses its advantage at higher speeds due to latency. To address this shortcoming, we propose a Learned Model Predictive Game (LMPG) approach that amortizes model predictive gameplay to reduce latency. In both simulation and hardware experiments, we benchmark our approach against MPG and MPC in head-to-head races, finding that LMPG outperforms both baselines.
>
---
#### [replaced 059] NestRL: A Nested Training Regime for Mutual Adaptation in Human-AI Teaming
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文属于人机协作任务，解决人类与AI在互动中适应性不足的问题。提出NestRL框架，通过分层训练提升AI对动态人类行为的适应能力。**

- **链接: [https://arxiv.org/pdf/2602.17737](https://arxiv.org/pdf/2602.17737)**

> **作者:** Upasana Biswas; Durgesh Kalwar; Subbarao Kambhampati; Sarath Sreedharan
>
> **摘要:** Mutual adaptation is a central challenge in human-AI teaming, as humans naturally adjust their strategies in response to an AI agent's behavior. Existing approaches attempt to approximate human behavior by diversifying training partners; however, these partners are typically static and fail to capture the adaptive nature of human teammates. When agents are trained jointly in standard multi-agent settings, they often converge to opaque coordination strategies that work only with their co-trained partners, leading to poor generalization. To model adaptive human behavior, we formulate human-AI teaming as an Interactive Partially Observable Markov Decision Process (I-POMDP). We propose NestRL, a nested training regime that learns the solution to a finite-level I-POMDP by training agents at each level against adaptive agents from the level below. This exposes agents to adaptive behavior while preventing emergence of opaque coordination strategies. We provide theoretical analysis showing that NestRL agents avoid convergence to partner-specific strategies, and validate this empirically in the Overcooked domain against state-of-the-art baselines. NestRL achieves higher task performance with both unseen adaptive agents and real human teammates, while exhibiting significantly greater adaptability over the course of interaction.
>
---
#### [replaced 060] Degeneration of Sliding-Window Factor Graph Optimization into Iterated Extended Kalman Filtering
- **分类: cs.RO**

- **简介: 该论文研究滑动窗口因子图优化与迭代扩展卡尔曼滤波的理论关系，通过构建递归因子图优化方法，证明两者的等价性，解决两者融合问题。**

- **链接: [https://arxiv.org/pdf/2511.00306](https://arxiv.org/pdf/2511.00306)**

> **作者:** Baoshan Song; Ruijie Xu; Zhi Zhan; Li-Ta Hsu
>
> **备注:** Accepted by Nature Partner Journal Wireless Technology
>
> **摘要:** Sliding window factor graph optimization (SW-FGO) is widely recognized for its robustness, yet its theoretical relationship with the extended Kalman filter (EKF) remains a subject of debate. This paper establishes the sufficient conditions to bridge SW-FGO with the iterated extended Kalman filter (IEKF). We introduce recursive FGO (Re-FGO), a conceptual perspective that employs a two-stage marginalization pipeline to mathematically degenerate the factor graph optimization to the IEKF recursive update. By enforcing the Markov assumption and a single-state window, we prove the theoretical equivalence between the IEKF and Re-FGO. This degeneration is validated through simulations and real-world urban GNSS and INS tightly coupled fusion experiments. The results confirm that Re-FGO exactly reproduces IEKF estimation behavior, demonstrating that the two-stage marginalization pipeline is foundational to enforce structural consistency, thereby successfully uniting graph-based smoothing and filtering paradigms under unified optimization principles.
>
---
#### [replaced 061] AttenA+: Rectifying Action Inequality in Robotic Foundation Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决VLA和WAM模型在长任务中因动作不均衡导致的性能限制，提出AttenA+框架通过速度驱动的注意力机制提升模型对关键动作的识别与学习。**

- **链接: [https://arxiv.org/pdf/2605.13548](https://arxiv.org/pdf/2605.13548)**

> **作者:** Daojie Peng; Fulong Ma; Jiahang Cao; Qiang Zhang; Xupeng Xie; Jian Guo; Ping Luo; Andrew F. Luo; Boyu Zhou; Jun Ma
>
> **摘要:** Existing robotic foundation models, while powerful, are predicated on an implicit assumption of temporal homogeneity: treating all actions as equally informative during optimization. This "flat" training paradigm, inherited from language modeling, remains indifferent to the underlying physical hierarchy of manipulation. In reality, robot trajectories are fundamentally heterogeneous, where low-velocity segments often dictate task success through precision-demanding interactions, while high-velocity motions serve as error-tolerant transitions. Such a misalignment between uniform loss weighting and physical criticality fundamentally limits the performance of current Vision-Language-Action (VLA) models and World-Action Models (WAM) in complex, long-horizon tasks. To rectify this, we introduce AttenA+, an architecture-agnostic framework that prioritizes kinematically critical segments via velocity-driven action attention. By reweighting the training objective based on the inverse velocity field, AttenA+ naturally aligns the model's learning capacity with the physical demands of manipulation. As a plug-and-play enhancement, AttenA+ can be integrated into existing backbones without structural modifications or additional parameters. Extensive experiments demonstrate that AttenA+ significantly elevates the ceilings of current state-of-the-art models. Specifically, it improves OpenVLA-OFT to 98.6% (+1.5%) on the Libero benchmark and pushes FastWAM to 92.4% (+0.6%) on RoboTwin 2.0. Real-world validation on a Franka manipulator further showcases its robustness and cross-task generalization. Our work suggests that mining the intrinsic structural priors of action sequences offers a highly efficient, physics-aware complement to standard scaling laws, paving a new path for general-purpose robotic control.
>
---
#### [replaced 062] HALO: Learning Human-Robot Collaboration via Heterogeneous-Agent Lyapunov Policy Optimization
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机协作任务，解决HRC中因人类与机器人异质性导致的策略不一致问题。提出HALO框架，通过Lyapunov优化稳定多智能体强化学习，提升协作鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03741](https://arxiv.org/pdf/2603.03741)**

> **作者:** Hao Zhang; Yaru Niu; Yikai Wang; Ding Zhao; H. Eric Tseng
>
> **备注:** this https URL
>
> **摘要:** To improve generalization and resilience in human-robot collaboration (HRC), robots must contend with diverse combinations of human behaviors and contexts, motivating multi-agent reinforcement learning (MARL). However, inherent heterogeneity between robots and humans creates a rationality gap (RG), where decentralized policy updates deviate from cooperative joint optimization. The resulting learning problem is a general-sum differentiable game, so independent policy-gradient updates can oscillate or diverge without added structure. We propose heterogeneous-agent Lyapunov policy optimization (HALO), a framework that stabilizes decentralized MARL by enforcing Lyapunov-based contraction in policy-parameter space. Unlike Lyapunov-based safe RL, which targets state/trajectory constraints in constrained Markov decision processes, HALO uses Lyapunov certification to stabilize decentralized policy learning. HALO rectifies decentralized gradients via optimal quadratic projections, ensuring monotonic contraction of RG and enabling effective exploration of open-ended interaction spaces. Extensive simulations and real-world humanoid-robot experiments show that this certified stability improves generalization and robustness in collaborative corner cases. Our project website is available at this https URL.
>
---
#### [replaced 063] Towards Drone-based Mapping of Volcanic Gases using Gas Tomography
- **分类: cs.RO**

- **简介: 该论文属于火山气体监测任务，旨在解决无人机测量气体时因旋翼下洗导致的气溶胶分散问题。通过模型化气体断层扫描技术，实现准确的气体分布映射。**

- **链接: [https://arxiv.org/pdf/2605.27180](https://arxiv.org/pdf/2605.27180)**

> **作者:** Marius Schaab; Niklas Karbach; Antonia Rabe; Thomas Wiedemann; Patrick Hinsen; Dmitriy Shutin; Thorsten Hoffmann; Achim J. Lilienthal
>
> **摘要:** Volcanoes emit large amounts of CO2, directly influencing human lives. Mapping volcanic gas emissions helps to forecast eruptions and understand the impact of volcanoes on climate and the environment. Drone-based gas sensing significantly reduces risks in volcanic monitoring but faces technical limitations when measuring gas, as rotor downwash disperses the gas plume before detection. Gas Tomography using remote gas sensing addresses this challenge. At the Salinelle dei Cappuccini mud volcanoes, we demonstrate that while drone-mounted in-situ sensors failed to detect CO2 emissions due to aerodynamic disturbance, open-path sensing successfully enabled remote gas distribution mapping. We present a novel model-based gas tomographic reconstruction approach that incorporates a Lagrangian model to compensate for wind-induced advection. The resulting gas distribution maps align with manually collected in-situ measurements, confirming that model-based gas tomography effectively overcomes downwash limitations and enables accurate mapping of volcanic emissions.
>
---
#### [replaced 064] Replicable Simulation-Based Robot Validation through Provenance
- **分类: cs.RO**

- **简介: 该论文属于机器人验证任务，旨在解决仿真测试的可复现性问题。通过引入数据溯源和FAIR元数据，增强测试过程的透明度与可追溯性。**

- **链接: [https://arxiv.org/pdf/2605.29973](https://arxiv.org/pdf/2605.29973)**

> **作者:** Argentina Ortega; Samuel Wiest; Frederik Pasch; Nico Hochgeschwender
>
> **备注:** Accepted for publication at 2026 IEEE RAS International Conference on Engineering Reliable Autonomous Systems (ERAS)
>
> **摘要:** Robot behavior is often validated through simulation-based testing, yet the replicability of such campaigns depends critically on transparent documentation of how tests are configured, executed, and post-processed. We argue that data provenance, coupled with the FAIR principles (findability, accessibility, interoperability, and reusability), addresses this gap by explicitly tracking links between artifacts and by attaching machine-readable metadata about file origins and key design decisions. Moreover, provenance and metadata cannot be treated as an afterthought confined to final datasets; they must be integrated into the testing processes that generate those datasets so that evidence can be reconstructed end-to-end. We demonstrate this by augmenting an existing simulation-based testing framework with provenance tracking and metadata collection mechanisms, and by using these extensions to enrich a mobile robot navigation dataset with structured provenance and FAIR-aligned metadata. Finally, we discuss obstacles encountered in this integration -- such as vocabulary alignment, attribute selection, and adoption of domain standards -- and provide actionable recommendations for implementing provenance-centric, FAIR metadata in robotics validation workflows.
>
---
#### [replaced 065] Situation-Aware Interactive MPC Switching for Autonomous Driving
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主驾驶任务，解决交互场景下的控制问题。通过情境感知的MPC切换策略，在保证性能的同时降低计算负担。**

- **链接: [https://arxiv.org/pdf/2512.06182](https://arxiv.org/pdf/2512.06182)**

> **作者:** Shuhao Qi; Qiling Aori; Luyao Zhang; Mircea Lazar; Sofie Haesaert
>
> **摘要:** Autonomous driving in interactive traffic scenarios remains challenging because of the mutual influence among vehicles and the inherent uncertainty of surrounding agents. Several model predictive control (MPC) formulations have been proposed to address this challenge, each adopting a different model of inter-agent interaction. While higher-fidelity interaction models enable more intelligent behavior, they incur substantially greater computational cost. Since strong interactions arise only occasionally in real traffic, a practical strategy for balancing performance and computational overhead is to invoke an appropriate controller based on situational demands. To this end, we first conduct a comparative study to assess and hierarchize the interactive capabilities of different MPC formulations. Building on this hierarchy, we then develop a neural network-based classifier for situation-aware switching among these controllers. We demonstrate that, by invoking the most advanced interactive MPC only in rare but critical situations and relying on a basic MPC in the majority of situations, situation-aware switching substantially improves overall performance while significantly reducing computational load.
>
---
#### [replaced 066] Proactive-reactive detection and mitigation of intermittent faults in robot swarms
- **分类: cs.RO; cs.MA; eess.SY**

- **简介: 该论文属于机器人集群故障容错任务，解决间歇性故障检测与缓解问题。通过自组织备份层和分布式共识方法，实现故障的主动预防与快速响应。**

- **链接: [https://arxiv.org/pdf/2509.19246](https://arxiv.org/pdf/2509.19246)**

> **作者:** Sinan Oğuz; Emanuele Garone; Marco Dorigo; Mary Katherine Heinrich
>
> **摘要:** Intermittent faults are transient errors that sporadically appear and disappear. Although intermittent faults pose substantial challenges to reliability and coordination, existing studies of fault tolerance in robot swarms focus instead on permanent faults. One reason for this is that intermittent faults are prohibitively difficult to detect in the fully self-organized ad-hoc networks typical of robot swarms, as their network topologies are transient and often unpredictable. However, in the recently introduced self-organizing nervous systems (SoNS) approach, robot swarms are able to self-organize persistent network structures for the first time, easing the problem of detecting intermittent faults. To address intermittent faults in robot swarms that have persistent networks, we propose a novel proactive-reactive strategy to detection and mitigation, based on self-organized backup layers and distributed consensus in a multiplex network. Proactively, the robots self-organize dynamic backup paths before faults occur, adapting to changes in the primary network topology and the robots' relative positions. Reactively, robots use one-shot likelihood ratio tests to compare information received along different paths in the multiplex network, enabling early fault detection. Upon detection, communication is temporarily rerouted in a self-organized way, until the detected fault resolves. We validate the approach in representative scenarios of faulty positional data occurring during formation control, demonstrating that intermittent faults are prevented from disrupting convergence to desired formations, with high fault detection accuracy and low rates of false positives.
>
---
#### [replaced 067] CrazyMARL: Decentralized Direct Motor Control Policies for Cooperative Aerial Transport of Cable-Suspended Payloads
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多无人机协同控制任务，旨在解决电缆悬挂负载运输中的协调与鲁棒性问题。提出CrazyMARL框架，实现高效、稳定的多机协作。**

- **链接: [https://arxiv.org/pdf/2509.14126](https://arxiv.org/pdf/2509.14126)**

> **作者:** Viktor Lorentz; Khaled Wahba; Sayantan Auddy; Marc Toussaint; Wolfgang Hönig
>
> **备注:** International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** Collaborative transportation of cable-suspended payloads by teams of UAVs has the potential to enhance payload capacity, adapt to different payload shapes, and provide built-in compliance, making it attractive for applications ranging from disaster relief to precision logistics. However, multi-UAV coordination under disturbances, nonlinear payload dynamics, and slack-taut cable modes remains a challenging control problem. To our knowledge, no prior work has addressed these cable mode transitions in the multi-UAV context, instead relying on simplifying rigid-link assumptions. We propose CrazyMARL, a decentralized RL framework for multi-UAV cable-suspended payload transport. Simulation results demonstrate that the learned policies can outperform classical decentralized controllers in terms of disturbance rejection and tracking precision, achieving an 80% recovery rate from harsh conditions compared to 44% for the baseline method. We also achieve successful zero-shot sim-to-real transfer and demonstrate that our policies are highly robust under harsh conditions, including wind, random external disturbances, and transitions between slack and taut cable dynamics. This work paves the way for autonomous, resilient UAV teams capable of executing complex payload missions in unstructured environments. Code and videos can be found on the website: this https URL.
>
---
#### [replaced 068] Seq-DeepIPC: Sequential Sensing for End-to-End Control in Legged Robot Navigation
- **分类: cs.RO; cs.CV; eess.IV; eess.SY**

- **简介: 该论文提出Seq-DeepIPC，用于腿部机器人导航的端到端感知与控制。解决多模态感知与控制融合问题，通过序列输入提升性能。**

- **链接: [https://arxiv.org/pdf/2510.23057](https://arxiv.org/pdf/2510.23057)**

> **作者:** Oskar Natan; Jun Miura
>
> **备注:** This work has been accepted for publication in the IEEE Sensors Journal. this https URL
>
> **摘要:** We present Seq-DeepIPC, a sequential end-to-end perception-to-control model for legged robot navigation in real-world environments. Seq-DeepIPC advances intelligent sensing for autonomous legged navigation by tightly integrating multi-modal perception (RGB-D + GNSS) with temporal fusion and control. The model jointly predicts semantic segmentation and depth estimation, giving richer spatial features for planning and control. For efficient deployment on edge devices, we use a lightweight model as the encoder, reducing computation while maintaining accuracy. Heading estimation is simplified by removing the noisy IMU and instead deriving global heading via differential analysis of sequential GNSS coordinates. We collected a larger and more diverse dataset that includes both road and grass terrains, and validated Seq-DeepIPC on a robot dog. Comparative and ablation studies show that sequential inputs improve perception and control in our models, while other baselines do not benefit. Seq-DeepIPC achieves competitive or better results with reasonable model size; although GNSS-only heading is less reliable near tall buildings, it is robust in open areas. Overall, Seq-DeepIPC extends end-to-end navigation beyond wheeled robots to more versatile and temporally-aware systems. To support future research, we will release the codes to our GitHub repo at this https URL.
>
---
#### [replaced 069] RynnVLA-002: A Unified Vision-Language-Action and World Model
- **分类: cs.RO**

- **简介: 该论文提出RynnVLA-002，融合视觉、语言、动作与世界模型，解决环境理解与动作规划问题，通过联合学习提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.17502](https://arxiv.org/pdf/2511.17502)**

> **作者:** Jun Cen; Siteng Huang; Yuqian Yuan; Kehan Li; Hangjie Yuan; Chaohui Yu; Bohan Hou; Yuming Jiang; Jiayan Guo; Xin Li; Hao Luo; Fan Wang; Deli Zhao; Hao Chen
>
> **摘要:** We introduce RynnVLA-002, a unified Vision-Language-Action (VLA) and world model. The world model leverages action and visual inputs to predict future image states, learning the underlying physics of the environment to refine action generation. Conversely, the VLA model produces subsequent actions from image observations, enhancing visual understanding and supporting the world model's image generation. The unified framework of RynnVLA-002 enables joint learning of environmental dynamics and action planning. Our experiments show that RynnVLA-002 surpasses individual VLA and world models, demonstrating their mutual enhancement. We evaluate RynnVLA-002 in both simulation and real-world robot tasks. RynnVLA-002 achieves 97.4% success rate on the LIBERO simulation benchmark without pretraining, while in real-world LeRobot experiments, its integrated world model boosts the overall success rate by 50%.
>
---
#### [replaced 070] ShelfAware: Real-Time Semantic Localization in Quasi-Static Environments with Low-Cost Sensors
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ShelfAware，解决动态环境中基于语义的实时定位问题。通过融合语义信息与几何数据，提升定位鲁棒性，适用于低成本硬件。**

- **链接: [https://arxiv.org/pdf/2512.09065](https://arxiv.org/pdf/2512.09065)**

> **作者:** Shivendra Agrawal; Jake Brawer; Ashutosh Naik; Alessandro Roncone; Bradley Hayes
>
> **备注:** 8 pages
>
> **摘要:** Many indoor workspaces are quasi-static: their global geometric layout is stable, but local semantics change continually, producing repetitive geometry, dynamic clutter, and perceptual noise that defeat standard vision-based localization. We present ShelfAware, a semantic particle filter for robust global localization that treats scene semantics as statistical evidence over object categories rather than fixed quantity landmarks. ShelfAware fuses a depth likelihood with a category-centric semantic similarity and uses a precomputed bank of semantic viewpoints to perform inverse semantic proposals inside Monte Carlo Localization (MCL), yielding fast, targeted hypothesis generation on low-cost, vision-only hardware. To demonstrate perception-agnostic scalability, we evaluate ShelfAware across two domains. In a rigorously controlled mock retail environment, ShelfAware achieves a 97% global localization success rate, maintaining the highest tracking success (66%) across cart, wearable, and dynamic occlusion conditions. Furthermore, in a 3,500 sq. ft. operational grocery store leveraging an open-vocabulary vision pipeline, ShelfAware significantly outperforms both geometric and fixed-quantity semantic baselines. By modeling semantics distributionally and leveraging inverse proposals, ShelfAware resolves geometric aliasing, providing an infrastructure-free building block for mobile and assistive robots in dynamic real-world environments.
>
---
#### [replaced 071] Motion-aware Event Suppression for Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种运动感知事件抑制框架，用于过滤事件相机中由IMOs和自运动引起的噪声。任务是提升事件流处理的准确性与效率，通过实时分割与预测运动实现动态事件的提前抑制。**

- **链接: [https://arxiv.org/pdf/2602.23204](https://arxiv.org/pdf/2602.23204)**

> **作者:** Roberto Pellerito; Nico Messikommer; Giovanni Cioffi; Marco Cannici; Davide Scaramuzza
>
> **备注:** Robotics: Science and Systems (RSS) 2026
>
> **摘要:** In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.
>
---
#### [replaced 072] LLM Trainer: Automated Robotic Data Generation via Demonstration Augmentation using LLMs
- **分类: cs.RO**

- **简介: 该论文提出LLM Trainer，用于自动化机器人数据生成。解决少量人类示范生成大量数据的问题，通过LLM进行关键帧标注和姿态迁移，提升模仿学习效果。**

- **链接: [https://arxiv.org/pdf/2509.20070](https://arxiv.org/pdf/2509.20070)**

> **作者:** Abraham George; Amir Barati Farimani
>
> **备注:** 9 pages, 5 figures, 4 tables. Accepted in ICRA 2026
>
> **摘要:** We present LLM Trainer, a fully automated pipeline that leverages the world knowledge of Large Language Models (LLMs) to transform a small number of human demonstrations (as few as one) into a large robot dataset for imitation learning. Our approach decomposes demonstration generation into two steps: (1) offline demonstration annotation that extracts keyframes, salient objects, and pose-object relations; and (2) online keypose retargeting that adapts those keyframes to a new scene, given an initial observation. Using these modified keypoints, our system warps the original demonstration to generate a new trajectory, which is then executed, and the resulting demo, if successful, is saved. Because the annotation is reusable across scenes, we use Thompson sampling to optimize the annotation, significantly improving generation success rate. We evaluate our method on a range of tasks, and find that our data annotation method consistently outperforms expert-engineered baselines. We further show an ensemble policy that combines the optimized LLM feed-forward plan with a learned feedback imitation learning controller. Finally, we demonstrate hardware feasibility on a Franka Emika Panda robot. For additional materials and demonstration videos, please see the project website: this https URL
>
---
#### [replaced 073] An Asynchronous Two-Speed Kalman Filter for Real-Time UUV Cooperative Navigation Under Acoustic Delays
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于水下无人潜航器协同导航任务，解决水声延迟导致的实时状态估计问题。提出异步双速卡尔曼滤波器，提升导航精度与实时性。**

- **链接: [https://arxiv.org/pdf/2604.02878](https://arxiv.org/pdf/2604.02878)**

> **作者:** Shuyue Li; Miguel López-Benítez; Eng Gee Lim; Fei Ma; Qian Dong; Mengze Cao; Limin Yu; Xiaohui Qin
>
> **备注:** 6 pages, 6 figures. Accepted for publication in the 2026 IEEE International Conference on Industrial Informatics (INDIN). \c{opyright} 2026 IEEE. Personal use of this material is permitted. See PDF for the full IEEE copyright notice
>
> **摘要:** In Global Navigation Satellite System (GNSS)-denied underwater environments, individual unmanned underwater vehicles (UUVs) suffer from unbounded dead-reckoning drift, making collaborative navigation (CN) crucial for accurate state estimation. However, the severe communication delay inherent in underwater acoustic channels poses serious challenges to real-time state estimation. Traditional filters, such as Extended Kalman Filters (EKFs) or Unscented Kalman Filters (UKFs), usually block the main control loop while waiting for delayed data, or effectively discard Out-of-Sequence Measurements (OOSMs), resulting in serious drift. To address this, we propose an Asynchronous Two-Speed Kalman Filter (TSKF) enhanced by a novel projection mechanism, which we term Variational History Distillation (VHD). The proposed architecture decouples the estimation process into two parallel threads: a fast-rate thread that utilizes Gaussian Process (GP) compensated dead reckoning to guarantee high-frequency real-time control, and a slow-rate thread dedicated to processing asynchronously delayed collaborative information. By introducing a Finite-Length Circular State Buffer (FLCSB), the algorithm applies delayed measurements to their corresponding historical states, and utilizes a VHD-based projection to fast-forward the correction to the current time without computationally heavy recalculations. Simulation results demonstrate that the proposed TSKF maintains a trajectory error comparable to computationally intensive batch-optimization methods under severe delays (up to 30\,s). Executing in sub-millisecond time, it significantly outperforms standard EKF/UKF. The results demonstrate an effective control, communication, and computing (3C) co-design that significantly enhances the resilience of autonomous marine automation systems.
>
---
#### [replaced 074] RoboBenchMart: Benchmarking Robots in Retail Environment
- **分类: cs.RO; cs.AI**

- **简介: 论文提出RoboBenchMart，用于评估机器人在零售环境中的操作能力。针对现有基准多限于家庭场景的问题，该工作构建了模拟零售环境基准，测试机器人处理复杂物品的能力，以推动机器人在零售领域的应用。**

- **链接: [https://arxiv.org/pdf/2511.10276](https://arxiv.org/pdf/2511.10276)**

> **作者:** Konstantin Soshin; Alexander Krapukhin; Andrei Spiridonov; Gregorii Bukhtuev; Andrey Kuznetsov; Vlad Shakhuro; Denis Shepelev
>
> **摘要:** Most existing robotic manipulation benchmarks focus on tabletop or household scenarios. While these setups have driven impressive progress, it remains unclear whether generalist VLAs that excel there can truly generalize to domains with different geometry, semantics, and workflows. We introduce RoboBenchMart, an open-source simulated benchmark targeting retail dark-store environments, where a mobile manipulator must perform complex manipulation tasks with diverse grocery items. This setting presents significant challenges, including dense object clutter and varied spatial configurations, with items positioned at different heights, depths, and in close proximity. By targeting on the retail domain, our benchmark addresses a setting with strong potential for near-term automation impact. Using generated trajectories, we model a standard, realistic fine-tuning setup for current generalist VLAs and evaluate several state-of-the-art models. We find that they still struggle even on common retail tasks, indicating that these models are not yet truly general across domains. To support further research, we release the RoboBenchMart suite, which includes a procedural store layout generator, a trajectory generation pipeline, evaluation tools, and fine-tuned baseline models.
>
---
#### [replaced 075] Picasso: Holistic Scene Reconstruction with Physics-Constrained Sampling
- **分类: cs.CV; cs.AI; cs.RO; eess.SY**

- **简介: 该论文提出Picasso方法，解决多物体场景重建中的物理合理性问题。通过考虑物体交互和物理约束，提升场景重建的准确性与真实性。**

- **链接: [https://arxiv.org/pdf/2602.08058](https://arxiv.org/pdf/2602.08058)**

> **作者:** Xihang Yu; Rajat Talak; Lorenzo Shaikewitz; Luca Carlone
>
> **备注:** 15 pages, accepted to Robotics: Science and Systems (RSS) 2026
>
> **摘要:** In the presence of occlusions and measurement noise, geometrically accurate scene reconstructions -- which fit the sensor data -- can still be physically incorrect. For instance, when estimating the poses and shapes of objects in the scene and importing the resulting estimates into a simulator, small errors might translate to implausible configurations including object interpenetration or unstable equilibrium. This makes it difficult to predict the dynamic behavior of the scene using a digital twin, an important step in simulation-based planning and control of contact-rich behaviors. In this paper, we posit that object pose and shape estimation requires reasoning holistically over the scene (instead of reasoning about each object in isolation), accounting for object interactions and physical plausibility. Towards this goal, our first contribution is Picasso, a physics-constrained reconstruction pipeline that builds multi-object scene reconstructions by considering geometry, non-penetration, and physics. Picasso relies on a fast rejection sampling method that reasons over multi-object interactions, leveraging an inferred object contact graph to guide samples. Second, we propose the Picasso dataset, a collection of 10 contact-rich real-world scenes with ground truth annotations, as well as a metric to quantify physical plausibility, which we open-source as part of our benchmark. Finally, we provide an extensive evaluation of Picasso on our newly introduced dataset and on the YCB-V dataset, and show it largely outperforms the state of the art while providing reconstructions that are both physically plausible and more aligned with human intuition.
>
---
#### [replaced 076] SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes
- **分类: cs.RO; cs.AI; cs.CV; cs.GR**

- **简介: 该论文提出SceneSmith，用于生成逼真且物理合理的室内场景，解决现有环境缺乏多样性和复杂性的问题。通过自然语言生成仿真环境，提升机器人训练效果。**

- **链接: [https://arxiv.org/pdf/2602.09153](https://arxiv.org/pdf/2602.09153)**

> **作者:** Nicholas Pfaff; Thomas Cohn; Sergey Zakharov; Rick Cory; Russ Tedrake
>
> **备注:** ICML 2026 Spotlight; Project page: this https URL
>
> **摘要:** Simulation has become a key tool for training and evaluating home robots at scale, yet existing environments fail to capture the diversity and physical complexity of real indoor spaces. Current scene synthesis methods produce sparsely furnished rooms that lack the dense clutter, articulated furniture, and physical properties essential for robotic manipulation. We introduce SceneSmith, a hierarchical agentic framework that generates simulation-ready indoor environments from natural language prompts. SceneSmith constructs scenes through successive stages$\unicode{x2013}$from architectural layout to furniture placement to small object population$\unicode{x2013}$each implemented as an interaction among VLM agents: designer, critic, and orchestrator. The framework tightly integrates asset generation through text-to-3D synthesis for static objects, dataset retrieval for articulated objects, and physical property estimation. SceneSmith generates 3-6x more objects than prior methods, with <2% inter-object collisions and 96% of objects remaining stable under physics simulation. In a user study with 205 participants, it achieves 92% average realism and 91% average prompt faithfulness win rates against baselines. We further demonstrate that these environments can be used in an end-to-end pipeline for automatic robot policy evaluation.
>
---
