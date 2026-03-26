# 机器人 cs.RO

- **最新发布 46 篇**

- **更新 27 篇**

## 最新发布

#### [new 001] Human-in-the-Loop Pareto Optimization: Trade-off Characterization for Assist-as-Needed Training and Performance Evaluation
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于人机协同优化任务，旨在解决训练中任务难度与用户表现的权衡问题。通过混合模型进行帕累托优化，实现高效评估与协议设计。**

- **链接: [https://arxiv.org/pdf/2603.23777](https://arxiv.org/pdf/2603.23777)**

> **作者:** Harun Tolasa; Volkan Patoglu
>
> **备注:** Under review for publication in IEEE Transactions on Haptics
>
> **摘要:** During human motor skill training and physical rehabilitation, there is an inherent trade-off between task difficulty and user performance. Characterizing this trade-off is crucial for evaluating user performance, designing assist-as-needed (AAN) protocols, and assessing the efficacy of training protocols. In this study, we propose a novel human-in-the-loop (HiL) Pareto optimization approach to characterize the trade-off between task performance and the perceived challenge level of motor learning or rehabilitation tasks. We adapt Bayesian multi-criteria optimization to systematically and efficiently perform HiL Pareto characterizations. Our HiL optimization employs a hybrid model that measures performance with a quantitative metric, while the perceived challenge level is captured with a qualitative metric. We demonstrate the feasibility of the proposed HiL Pareto characterization through a user study. Furthermore, we present the utility of the framework through three use cases in the context of a manual skill training task with haptic feedback. First, we demonstrate how the characterized trade-off can be used to design a sample AAN training protocol for a motor learning task and to evaluate the group-level efficacy of the proposed AAN protocol relative to a baseline adaptive assistance protocol. Second, we demonstrate that individual-level comparisons of the trade-offs characterized before and after the training session enable fair evaluation of training progress under different assistance levels. This evaluation method is more general than standard performance evaluations, as it can provide insights even when users cannot perform the task without assistance. Third, we show that the characterized trade-offs also enable fair performance comparisons among different users, as they capture the best possible performance of each user under all feasible assistance levels.
>
---
#### [new 002] Chameleon: Episodic Memory for Long-Horizon Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，解决感知混淆下的长期决策问题。提出Chameleon模型，通过记忆机制提升决策可靠性与长时序控制能力。**

- **链接: [https://arxiv.org/pdf/2603.24576](https://arxiv.org/pdf/2603.24576)**

> **作者:** Xinying Guo; Chenxi Jiang; Hyun Bin Kim; Ying Sun; Yang Xiao; Yuhang Han; Jianfei Yang
>
> **备注:** Code is available at this https URL
>
> **摘要:** Robotic manipulation often requires memory: occlusion and state changes can make decision-time observations perceptually aliased, making action selection non-Markovian at the observation level because the same observation may arise from different interaction histories. Most embodied agents implement memory via semantically compressed traces and similarity-based retrieval, which discards disambiguating fine-grained perceptual cues and can return perceptually similar but decision-irrelevant episodes. Inspired by human episodic memory, we propose Chameleon, which writes geometry-grounded multimodal tokens to preserve disambiguating context and produces goal-directed recall through a differentiable memory stack. We also introduce Camo-Dataset, a real-robot UR5e dataset spanning episodic recall, spatial tracking, and sequential manipulation under perceptual aliasing. Across tasks, Chameleon consistently improves decision reliability and long-horizon control over strong baselines in perceptually confusable settings.
>
---
#### [new 003] Evidence of an Emergent "Self" in Continual Robot Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于人工智能领域，旨在探索智能系统中的“自我”概念。通过分析机器人在持续学习中的认知结构，发现其形成稳定的子网络，暗示“自我”的存在。**

- **链接: [https://arxiv.org/pdf/2603.24350](https://arxiv.org/pdf/2603.24350)**

> **作者:** Adidev Jhunjhunwala; Judah Goldfeder; Hod Lipson
>
> **备注:** 39 pages, 17 figures, includes supplementary materials
>
> **摘要:** A key challenge to understanding self-awareness has been a principled way of quantifying whether an intelligent system has a concept of a "self," and if so how to differentiate the "self" from other cognitive structures. We propose that the "self" can be isolated by seeking the invariant portion of cognitive process that changes relatively little compared to more rapidly acquired cognitive knowledge and skills, because our self is the most persistent aspect of our experiences. We used this principle to analyze the cognitive structure of robots under two conditions: One robot learns a constant task, while a second robot is subjected to continual learning under variable tasks. We find that robots subjected to continual learning develop an invariant subnetwork that is significantly more stable (p < 0.001) compared to the control. We suggest that this principle can offer a window into exploring selfhood in other cognitive AI systems.
>
---
#### [new 004] QuadFM: Foundational Text-Driven Quadruped Motion Dataset for Generation and Control
- **分类: cs.RO**

- **简介: 该论文提出QuadFM数据集，解决四足机器人运动生成与控制问题，整合多样动作、情感表达和语言语义，支持语言指令驱动的运动合成。**

- **链接: [https://arxiv.org/pdf/2603.24021](https://arxiv.org/pdf/2603.24021)**

> **作者:** Li Gao; Fuzhi Yang; Jianhui Chen; Liu Liu; Yao Zheng; Yang Cai; Ziqiao Li
>
> **摘要:** Despite significant advances in quadrupedal robotics, a critical gap persists in foundational motion resources that holistically integrate diverse locomotion, emotionally expressive behaviors, and rich language semantics-essential for agile, intuitive human-robot interaction. Current quadruped motion datasets are limited to a few mocap primitives (e.g., walk, trot, sit) and lack diverse behaviors with rich language grounding. To bridge this gap, we introduce Quadruped Foundational Motion (QuadFM) , the first large-scale, ultra-high-fidelity dataset designed for text-to-motion generation and general motion control. QuadFM contains 11,784 curated motion clips spanning locomotion, interactive, and emotion-expressive behaviors (e.g., dancing, stretching, peeing), each with three-layer annotation-fine-grained action labels, interaction scenarios, and natural language commands-totaling 35,352 descriptions to support language-conditioned understanding and command execution. We further propose Gen2Control RL, a unified framework that jointly trains a general motion controller and a text-to-motion generator, enabling efficient end-to-end inference on edge hardware. On a real quadruped robot with an NVIDIA Orin, our system achieves real-time motion synthesis (<500 ms latency). Simulation and real-world results show realistic, diverse motions while maintaining robust physical interaction. The dataset will be released at this https URL.
>
---
#### [new 005] Environment-Grounded Multi-Agent Workflow for Autonomous Penetration Testing
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主渗透测试任务，旨在提升机器人系统的安全性。针对复杂网络环境下的安全评估难题，提出一种基于大语言模型的多智能体架构，实现高效、可追溯的自动化测试。**

- **链接: [https://arxiv.org/pdf/2603.24221](https://arxiv.org/pdf/2603.24221)**

> **作者:** Michael Somma; Markus Großpointner; Paul Zabalegui; Eppu Heilimo; Branka Stojanović
>
> **摘要:** The increasing complexity and interconnectivity of digital infrastructures make scalable and reliable security assessment methods essential. Robotic systems represent a particularly important class of operational technology, as modern robots are highly networked cyber-physical systems deployed in domains such as industrial automation, logistics, and autonomous services. This paper explores the use of large language models for automated penetration testing in robotic environments. We propose an environment-grounded multi-agent architecture tailored to Robotics-based systems. The approach dynamically constructs a shared graph-based memory during execution that captures the observable system state, including network topology, communication channels, vulnerabilities, and attempted exploits. This enables structured automation while maintaining traceability and effective context management throughout the testing process. Evaluated across multiple iterations within a specialized robotics Capture-the-Flag scenario (ROS/ROS2), the system demonstrated high reliability, successfully completing the challenge in 100\% of test runs (n=5). This performance significantly exceeds literature benchmarks while maintaining the traceability and human oversight required by frameworks like the EU AI Act.
>
---
#### [new 006] MIRROR: Visual Motion Imitation via Real-time Retargeting and Teleoperation with Parallel Differential Inverse Kinematics
- **分类: cs.RO**

- **简介: 该论文属于人形机器人实时遥操作任务，解决逆运动学求解在冗余和自碰撞约束下的响应与安全性问题。提出一种基于GPU并行的微分逆运动学方法，提升逃离局部极小值的能力，保障实时性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.23995](https://arxiv.org/pdf/2603.23995)**

> **作者:** Junheng Li; Lizhi Yang; Aaron D. Ames
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Real-time humanoid teleoperation requires inverse kinematics (IK) solvers that are both responsive and constraint-safe under kinematic redundancy and self-collision constraints. While differential IK enables efficient online retargeting, its locally linearized updates are inherently basin-dependent and often become trapped near joint limits, singularities, or active collision boundaries, leading to unsafe or stagnant behavior. We propose a GPU-parallelized, continuation-based differential IK that improves escape from such constraint-induced local minima while preserving real-time performance, promoting safety and stability. Multiple constrained IK quadratic programs are evaluated in parallel, together with a self-collision avoidance control barrier function (CBF), and a Lyapunov-based progression criterion selects updates that reduce the final global task-space error. The method is paired with a visual skeletal pose estimation pipeline that enables robust, real-time upper-body teleoperation on the THEMIS humanoid robot hardware in real-world tasks.
>
---
#### [new 007] MonoSIM: An open source SIL framework for Ackermann Vehicular Systems with Monocular Vision
- **分类: cs.RO; eess.IV**

- **简介: 该论文提出MonoSIM，一个用于阿克曼车辆的开源SIL框架，解决自动驾驶算法验证问题。采用单目视觉和滑动窗口车道检测，支持控制策略测试与比较。**

- **链接: [https://arxiv.org/pdf/2603.23965](https://arxiv.org/pdf/2603.23965)**

> **作者:** Shantanu Rahman; Nayeb Hasin; Mainul Islam; Md. Zubair Alom Rony; Golam Sarowar
>
> **备注:** 6 pages, 16 figures, Published in "IEEE 12th International Conference on Automation, Robotics and Application 2026"
>
> **摘要:** This paper presents an open-source Software-in-the-Loop (SIL) simulation platform designed for autonomous Ackerman vehicle research and education. The proposed framework focuses on simplicity, while making it easy to work with small-scale experimental setups, such as the XTENTH-CAR platform. The system was designed using open source tools, creating an environment with a monocular camera vision system to capture stimuli from it with minimal computational overhead through a sliding window based lane detection method. The platform supports a flexible algorithm testing and validation environment, allowing researchers to implement and compare various control strategies within an easy-to-use virtual environment. To validate the working of the platform, Model Predictive Control (MPC) and Proportional-Integral-Derivative (PID) algorithms were implemented within the SIL framework. The results confirm that the platform provides a reliable environment for algorithm verification, making it an ideal tool for future multi-agent system research, educational purposes, and low-cost AGV development. Our code is available at this https URL.
>
---
#### [new 008] Knowledge-Guided Manipulation Using Multi-Task Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出KG-M3PO框架，解决部分可观测环境下的多任务机器人操作问题。通过结合知识图谱与强化学习，提升任务成功率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.24083](https://arxiv.org/pdf/2603.24083)**

> **作者:** Aditya Narendra; Mukhammadrizo Maribjonov; Dmitry Makarov; Dmitry Yudin; Aleksandr Panov
>
> **备注:** 8 pages, 8 figures. Accepted to IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** This paper introduces Knowledge Graph based Massively Multi-task Model-based Policy Optimization (KG-M3PO), a framework for multi-task robotic manipulation in partially observable settings that unifies Perception, Knowledge, and Policy. The method augments egocentric vision with an online 3D scene graph that grounds open-vocabulary detections into a metric, relational representation. A dynamic-relation mechanism updates spatial, containment, and affordance edges at every step, and a graph neural encoder is trained end-to-end through the RL objective so that relational features are shaped directly by control performance. Multiple observation modalities (visual, proprioceptive, linguistic, and graph-based) are encoded into a shared latent space, upon which the RL agent operates to drive the control loop. The policy conditions on lightweight graph queries alongside visual and proprioceptive inputs, yielding a compact, semantically informed state for decision making. Experiments on a suite of manipulation tasks with occlusions, distractors, and layout shifts demonstrate consistent gains over strong baselines: the knowledge-conditioned agent achieves higher success rates, improved sample efficiency, and stronger generalization to novel objects and unseen scene configurations. These results support the premise that structured, continuously maintained world knowledge is a powerful inductive bias for scalable, generalizable manipulation: when the knowledge module participates in the RL computation graph, relational representations align with control, enabling robust long-horizon behavior under partial observability.
>
---
#### [new 009] Design, Modelling and Characterisation of a Miniature Fibre-Reinforced Soft Bending Actuator for Endoluminal Interventions
- **分类: cs.RO**

- **简介: 该论文属于软体机器人任务，旨在解决微型软性执行器在人体腔道内操作的问题。设计并验证了一种纤维增强的软弯曲执行器，用于内窥镜介入应用。**

- **链接: [https://arxiv.org/pdf/2603.24461](https://arxiv.org/pdf/2603.24461)**

> **作者:** Xiangyi Tan; Aoife McDonald-Bowyer; Danail Stoyanov; Agostino Stilli
>
> **摘要:** Miniaturised soft pneumatic actuators are crucial for robotic intervention within highly constrained anatomical pathways. This work presents the design and validation of a fibre-reinforced soft actuator at the centimetre scale for inte- gration into an endoluminal robotic platform for natural-orifice interventional and diagnostic applications. A single-chamber geometry reinforced with embedded Kevlar fibre was de- signed to maximise curvature while preserving sealing integrity, fabricated using a multi-stage multi-stiffness silicone casting process, and validated against a high-fidelity Abaqus FEM using experimentally parametrised hyperelastic material models and embedded beam reinforcement. The semi-cylindrical actuator has an outer diameter of 18,mm and a length of 37.5,mm. Single and double helix winding configurations, fibre pitch, and fibre density were investigated. The optimal 100 SH configuration achieved a bending angle of 202.9° experimentally and 297.6° in simulation, with structural robustness maintained up to 100,kPa and radial expansion effectively constrained by the fibre reinforcement. Workspace evaluation confirmed suitability for integration into the target device envelope, demonstrating that fibre-reinforcement strategies can be effectively translated to the centimetre regime while retaining actuator performance.
>
---
#### [new 010] A Sensorless, Inherently Compliant Anthropomorphic Musculoskeletal Hand Driven by Electrohydraulic Actuators
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决传统机械手安全性与灵活性不足的问题。通过设计一种无传感器、具有内在柔顺性的仿生手，利用电液压致动器实现安全、灵活的抓取操作。**

- **链接: [https://arxiv.org/pdf/2603.24357](https://arxiv.org/pdf/2603.24357)**

> **作者:** Misato Sonoda; Ronan Hinchet; Amirhossein Kazemipour; Yasunori Toshimitsu; Robert K. Katzschmann
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Robotic manipulation in unstructured environments requires end-effectors that combine high kinematic dexterity with physical compliance. While traditional rigid hands rely on complex external sensors for safe interaction, electrohydraulic actuators offer a promising alternative. This paper presents the design, control, and evaluation of a novel musculoskeletal robotic hand architecture powered entirely by remote Peano-HASEL actuators, specifically optimized for safe manipulation. By relocating the actuators to the forearm, we functionally isolate the grasping interface from electrical hazards while maintaining a slim, human-like profile. To address the inherently limited linear contraction of these soft actuators, we integrate a 1:2 pulley routing mechanism that mechanically amplifies tendon displacement. The resulting system prioritizes compliant interaction over high payload capacity, leveraging the intrinsic force-limiting characteristics of the actuators to provide a high level of inherent safety. Furthermore, this physical safety is augmented by the self-sensing nature of the HASEL actuators. By simply monitoring the operating current, we achieve real-time grasp detection and closed-loop contact-aware control without relying on external force transducers or encoders. Experimental results validate the system's dexterity and inherent safety, demonstrating the successful execution of various grasp taxonomies and the non-destructive grasping of highly fragile objects, such as a paper balloon. These findings highlight a significant step toward simplified, inherently compliant soft robotic manipulation.
>
---
#### [new 011] Bio-Inspired Event-Based Visual Servoing for Ground Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉伺服控制任务，解决地面机器人运动感知与控制问题。通过仿生事件视觉机制，直接获取状态反馈，提升响应速度与计算效率。**

- **链接: [https://arxiv.org/pdf/2603.23672](https://arxiv.org/pdf/2603.23672)**

> **作者:** Maral Mordad; Kian Behzad; Debojyoti Biswas; Noah J. Cowan; Milad Siami
>
> **摘要:** Biological sensory systems are inherently adaptive, filtering out constant stimuli and prioritizing relative changes, likely enhancing computational and metabolic efficiency. Inspired by active sensing behaviors across a wide range of animals, this paper presents a novel event-based visual servoing framework for ground robots. Utilizing a Dynamic Vision Sensor (DVS), we demonstrate that by applying a fixed spatial kernel to the asynchronous event stream generated from structured logarithmic intensity-change patterns, the resulting net event flux analytically isolates specific kinematic states. We establish a generalized theoretical bound for this event rate estimator and show that linear and quadratic spatial profiles isolate the robot's velocity and position-velocity product, respectively. Leveraging these properties, we employ a multi-pattern stimulus to directly synthesize a nonlinear state-feedback term entirely without traditional state estimation. To overcome the inescapable loss of linear observability at equilibrium inherent in event sensing, we propose a bio-inspired active sensing limit-cycle controller. Experimental validation on a 1/10-scale autonomous ground vehicle confirms the efficacy, extreme low-latency, and computational efficiency of the proposed direct-sensing approach.
>
---
#### [new 012] AgentChemist: A Multi-Agent Experimental Robotic Platform Integrating Chemical Perception and Precise Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于实验室自动化任务，旨在解决传统平台对多样化实验任务适应性差的问题。通过多智能体系统实现动态任务分解与自适应控制，提升实验灵活性与通用性。**

- **链接: [https://arxiv.org/pdf/2603.23886](https://arxiv.org/pdf/2603.23886)**

> **作者:** Xiangyi Wei; Fei Wang; Haotian Zhang; Xin An; Haitian Zhu; Lianrui Hu; Yang Li; Changbo Wang; Xiao He
>
> **摘要:** Chemical laboratory automation has long been constrained by rigid workflows and poor adaptability to the long-tail distribution of experimental tasks. While most automated platforms perform well on a narrow set of standardized procedures, real laboratories involve diverse, infrequent, and evolving operations that fall outside predefined protocols. This mismatch prevents existing systems from generalizing to novel reaction conditions, uncommon instrument configurations, and unexpected procedural variations. We present a multi-agent robotic platform designed to address this long-tail challenge through collaborative task decomposition, dynamic scheduling, and adaptive control. The system integrates chemical perception for real-time reaction monitoring with feedback-driven execution, enabling it to adjust actions based on evolving experimental states rather than fixed scripts. Validation via acid-base titration demonstrates autonomous progress tracking, adaptive dispensing control, and reliable end-to-end experiment execution. By improving generalization across diverse laboratory scenarios, this platform provides a practical pathway toward intelligent, flexible, and scalable laboratory automation.
>
---
#### [new 013] LATS: Large Language Model Assisted Teacher-Student Framework for Multi-Agent Reinforcement Learning in Traffic Signal Control
- **分类: cs.RO**

- **简介: 该论文属于交通信号控制任务，旨在解决MARL在复杂交通环境中的表现不足问题。通过结合LLM与MARL，提出LATS框架，提升模型的表示能力和泛化性能。**

- **链接: [https://arxiv.org/pdf/2603.24361](https://arxiv.org/pdf/2603.24361)**

> **作者:** Yifeng Zhang; Peizhuo Li; Tingguang Zhou; Mingfeng Fan; Guillaume Sartoretti
>
> **摘要:** Adaptive Traffic Signal Control (ATSC) aims to optimize traffic flow and minimize delays by adjusting traffic lights in real time. Recent advances in Multi-agent Reinforcement Learning (MARL) have shown promise for ATSC, yet existing approaches still suffer from limited representational capacity, often leading to suboptimal performance and poor generalization in complex and dynamic traffic environments. On the other hand, Large Language Models (LLMs) excel at semantic representation, reasoning, and analysis, yet their propensity for hallucination and slow inference speeds often hinder their direct application to decision-making tasks. To address these challenges, we propose a novel learning paradigm named LATS that integrates LLMs and MARL, leveraging the former's strong prior knowledge and inductive abilities to enhance the latter's decision-making process. Specifically, we introduce a plug-and-play teacher-student learning module, where a trained embedding LLM serves as a teacher to generate rich semantic features that capture each intersection's topology structures and traffic dynamics. A much simpler (student) neural network then learns to emulate these features through knowledge distillation in the latent space, enabling the final model to operate independently from the LLM for downstream use in the RL decision-making process. This integration significantly enhances the overall model's representational capacity across diverse traffic scenarios, thus leading to more efficient and generalizable control strategies. Extensive experiments across diverse traffic datasets empirically demonstrate that our method enhances the representation learning capability of RL models, thereby leading to improved overall performance and generalization over both traditional RL and LLM-only approaches. [...]
>
---
#### [new 014] Form-Fitting, Large-Area Sensor Mounting for Obstacle Detection
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，解决传感器在非平面表面安装与定位的问题。通过CAD生成适配3D表面的皮肤单元，实现传感器的固定安装，用于障碍物检测。**

- **链接: [https://arxiv.org/pdf/2603.23725](https://arxiv.org/pdf/2603.23725)**

> **作者:** Anna Soukhovei; Carson Kohlbrenner; Caleb Escobedo; Alexander Gholmieh; Alexander Dickhans; Alessandro Roncone
>
> **备注:** Accepted at 2025 Humanoids Workshop on Advances in Contact-Rich Robotics: Rich Tactile-Based Physical Interaction [ConRich]
>
> **摘要:** We introduce a low-cost method for mounting sensors onto robot links for large-area sensing coverage that does not require the sensor's positions or orientations to be calibrated before use. Using computer aided design (CAD), a robot skin covering, or skin unit, can be procedurally generated to fit around a nondevelopable surface, a 3D surface that cannot be flattened into a 2D plane without distortion, of a robot. The skin unit embeds mounts for printed circuit boards of any size to keep sensors in fixed and known locations. We demonstrate our method by constructing point cloud images of obstacles within the proximity of a Franka Research 3 robot's operational environment using an array of time of flight (ToF) imagers mounted on a printed skin unit and attached to the robot arm.
>
---
#### [new 015] Task-Space Singularity Avoidance for Control Affine Systems Using Control Barrier Functions
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决机械系统奇异问题。通过控制屏障函数避免奇异配置，确保安全轨迹跟踪，提升控制稳定性。**

- **链接: [https://arxiv.org/pdf/2603.23753](https://arxiv.org/pdf/2603.23753)**

> **作者:** Kimia Forghani; Suraj Raval; Lamar Mair; Axel Krieger; Yancy Diaz-Mercado
>
> **摘要:** Singularities in robotic and dynamical systems arise when the mapping from control inputs to task-space motion loses rank, leading to an inability to determine inputs. This limits the system's ability to generate forces and torques in desired directions and prevents accurate trajectory tracking. This paper presents a control barrier function (CBF) framework for avoiding such singularities in control-affine systems. Singular configurations are identified through the eigenvalues of a state-dependent input-output mapping matrix, and barrier functions are constructed to maintain a safety margin from rank-deficient regions. Conditions for theoretical guarantees on safety are provided as a function of actuator dynamics. Simulations on a planar 2-link manipulator and a magnetically actuated needle demonstrate smooth trajectory tracking while avoiding singular configurations and reducing control input spikes by up to 100x compared to the nominal controller.
>
---
#### [new 016] Toward Generalist Neural Motion Planners for Robotic Manipulators: Challenges and Opportunities
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动规划任务，旨在解决神经运动规划器在复杂环境中的泛化能力不足问题，分析现有方法并提出建立通用神经运动规划器的方向。**

- **链接: [https://arxiv.org/pdf/2603.24318](https://arxiv.org/pdf/2603.24318)**

> **作者:** Davood Soleymanzadeh; Ivan Lopez-Sanchez; Hao Su; Yunzhu Li; Xiao Liang; Minghui Zheng
>
> **摘要:** State-of-the-art generalist manipulation policies have enabled the deployment of robotic manipulators in unstructured human environments. However, these frameworks struggle in cluttered environments primarily because they utilize auxiliary modules for low-level motion planning and control. Motion planning remains challenging due to the high dimensionality of the robot's configuration space and the presence of workspace obstacles. Neural motion planners have enhanced motion planning efficiency by offering fast inference and effectively handling the inherent multi-modality of the motion planning problem. Despite such benefits, current neural motion planners often struggle to generalize to unseen, out-of-distribution planning settings. This paper reviews and analyzes the state-of-the-art neural motion planners, highlighting both their benefits and limitations. It also outlines a path toward establishing generalist neural motion planners capable of handling domain-specific challenges. For a list of the reviewed papers, please refer to this https URL.
>
---
#### [new 017] SOMA: Strategic Orchestration and Memory-Augmented System for Vision-Language-Action Model Robustness via In-Context Adaptation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在提升视觉-语言-动作模型在分布外任务中的鲁棒性。通过引入SOMA系统，增强模型的记忆与动态干预能力，实现无需参数微调的上下文适应。**

- **链接: [https://arxiv.org/pdf/2603.24060](https://arxiv.org/pdf/2603.24060)**

> **作者:** Zhuoran Li; Zhiyang Li; Kaijun Zhou; Jinyu Gu
>
> **备注:** 9 pages, 16 figures, 3 table. Submitted to IROS 2026
>
> **摘要:** Despite the promise of Vision-Language-Action (VLA) models as generalist robotic controllers, their robustness against perceptual noise and environmental variations in out-of-distribution (OOD) tasks remains fundamentally limited by the absence of long-term memory, causal failure attribution, and dynamic intervention capability. To address this, we propose SOMA, a Strategic Orchestration and Memory-Augmented System that upgrades frozen VLA policies for robust in-context adaptation without parameter fine-tuning. Specifically, SOMA operates through an online pipeline of contrastive Dual-Memory Retrieval-Augmented Generation (RAG), an Attribution-Driven Large-Language-Model (LLM) Orchestrator, and extensible Model Context Protocol (MCP) interventions, while an offline Memory Consolidation module continuously distills the execution traces into reliable priors. Experimental evaluations across three backbone models (pi0, pi0.5, and SmolVLA) on LIBERO-PRO and our proposed LIBERO-SOMA benchmarks demonstrate that SOMA achieves an average absolute success rate gain of 56.6%. This includes a significant absolute improvement of 89.1% in long-horizon task chaining. Project page and source code are available at: this https URL.
>
---
#### [new 018] Enhancing Drone Light Shows Performances: Optimal Allocation and Trajectories for Swarm Drone Formations
- **分类: cs.RO**

- **简介: 该论文属于无人机编队控制任务，解决大规模无人机轨迹规划与分配问题，提出UATG框架实现高效、安全的空中表演。**

- **链接: [https://arxiv.org/pdf/2603.24401](https://arxiv.org/pdf/2603.24401)**

> **作者:** Yunes Alqudsi
>
> **摘要:** Drone light shows (DLShows) represent a rapidly growing application of swarm robotics, creating captivating aerial displays through the synchronized flight of hundreds or thousands of unmanned aerial vehicles (UAVs) as environmentally friendly and reusable alternatives to traditional pyrotechnics. This domain presents unique challenges in optimally assigning drones to visual waypoints and generating smooth, collision-free trajectories at a very large scale. This article introduces the Unified Assignment and Trajectory Generation (UATG) framework. The proposed approach concurrently solves two core problems: the optimal assignment of drones to designated goal locations and the generation of dynamically feasible, collision-free, time-parameterized trajectories. The UATG framework is specifically designed for DLShows, ensuring minimal transition times between formations and guaranteeing inter-drone collision avoidance. A key innovation is its exceptional computational efficiency, enabling the coordination of large-scale in real-time; for instance, it computes the optimal assignment and trajectories for 1008 drones in approximately one second on a standard laptop. Extensive simulations in realistic environments validate the framework's performance, demonstrating its capability to orchestrate complex formations, from alphanumeric characters to intricate 3D shapes, with precision and visual smoothness. This work provides a critical advancement for the DLShow industry, offering a practical and scalable solution for generating complex aerial choreography and establishing a valuable benchmark for ground control station software designed for the efficient coordination of multiple UAVs. A supplemental animated simulation of this work is available at this https URL.
>
---
#### [new 019] 3D-Mix for VLA: A Plug-and-Play Module for Integrating VGGT-based 3D Information into Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决MLLM在3D感知上的不足。通过提出3D-Mix模块，融合VGGT的3D信息，提升空间智能。**

- **链接: [https://arxiv.org/pdf/2603.24393](https://arxiv.org/pdf/2603.24393)**

> **作者:** Bin Yu; Shijie Lian; Xiaopeng Lin; Zhaolong Shen; Yuliang Wei; Haishan Liu; Changti Wu; Hang Yuan; Bailing Wang; Cong Huang; Kai Chen
>
> **备注:** 13 pages
>
> **摘要:** Vision-Language-Action (VLA) models leverage Multimodal Large Language Models (MLLMs) for robotic control, but recent studies reveal that MLLMs exhibit limited spatial intelligence due to training predominantly on 2D data, resulting in inadequate 3D perception for manipulation tasks. While recent approaches incorporate specialized 3D vision models such as VGGT to enhance spatial understanding, they employ diverse integration mechanisms without systematic investigation, leaving the optimal fusion strategy unclear. We conduct a comprehensive pilot study comparing nine VGGT integration schemes on standardized benchmarks and find that semantic-conditioned gated fusion, which adaptively balances 2D semantic and 3D geometric features based on task context, achieved the strongest performance among all nine evaluated fusion schemes in our pilot study. We present 3D-Mix, a plug-and-play module that integrates into diverse VLA architectures (GR00T-style and $\pi$-style) without modifying existing MLLM or action expert components. Experiments across six MLLM series (nine model variants, 2B--8B parameters) on SIMPLER and LIBERO show that 3D-Mix delivers consistent performance gains, averaging +7.0% on the out-of-domain (OOD) SIMPLER benchmark across all nine GR00T-style variants, establishing a principled approach for enhancing spatial intelligence in VLA systems.
>
---
#### [new 020] ROSCell: A ROS2-Based Framework for Automated Formation and Orchestration of Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文提出ROSCell框架，解决多机器人系统动态任务配置问题。通过ROS2实现灵活的计算连续体管理，提升效率与适应性。**

- **链接: [https://arxiv.org/pdf/2603.23690](https://arxiv.org/pdf/2603.23690)**

> **作者:** Jiangtao Shuai; Marvin Carl May; Sonja Schimmler; Manfred Hauswirth
>
> **摘要:** Modern manufacturing under High-Mix-Low-Volume requirements increasingly relies on flexible and adaptive matrix production systems, which depend on interconnected heterogeneous devices and rapid task reconfiguration. To address these needs, we present ROSCell, a ROS2-based framework that enables the flexible formation and management of a computing continuum across various devices. ROSCell allows users to package existing robotic software as deployable skills and, with simple requests, assemble isolated cells, automatically deploy skill instances, and coordinate their communication to meet task objectives. It provides a scalable and low-overhead foundation for adaptive multi-robot computing in dynamic production environments. Experimental results show that, in the idle state, ROSCell substantially reduces CPU, memory, and network overhead compared to K3s-based solutions on edge devices, highlighting its energy efficiency and cost-effectiveness for large-scale deployment in production settings. The source code, examples, and documentation will be provided on Github.
>
---
#### [new 021] Accelerated Spline-Based Time-Optimal Motion Planning with Continuous Safety Guarantees for Non-Differentially Flat Systems
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决非微分平坦系统的时间最优轨迹生成问题。通过解耦分离超平面与优化问题，降低计算复杂度并保证安全。**

- **链接: [https://arxiv.org/pdf/2603.24133](https://arxiv.org/pdf/2603.24133)**

> **作者:** Dries Dirckx; Jan Swevers; Wilm Decré
>
> **备注:** Submitted to the 2026 10th IEEE Conference on Control Technology and Applications (CCTA)
>
> **摘要:** Generating time-optimal, collision-free trajectories for autonomous mobile robots involves a fundamental trade-off between guaranteeing safety and managing computational complexity. State-of-the-art approaches formulate spline-based motion planning as a single Optimal Control Problem (OCP) but often suffer from high computational cost because they include separating hyperplane parameters as decision variables to enforce continuous collision avoidance. This paper presents a novel method that alleviates this bottleneck by decoupling the determination of separating hyperplanes from the OCP. By treating the separation theorem as an independent classification problem solvable via a linear system or quadratic program, the proposed method eliminates hyperplane parameters from the optimisation variables, effectively transforming non-convex constraints into linear ones. Experimental validation demonstrates that this decoupled approach reduces trajectory computation times up to almost 60% compared to fully coupled methods in obstacle-rich environments, while maintaining rigorous continuous safety guarantees.
>
---
#### [new 022] Learning Actionable Manipulation Recovery via Counterfactual Failure Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决执行错误后自主恢复的问题。通过合成真实失败场景，生成可执行的纠正轨迹，提升机器人故障恢复能力。**

- **链接: [https://arxiv.org/pdf/2603.13528](https://arxiv.org/pdf/2603.13528)**

> **作者:** Dayou Li; Jiuzhou Lei; Hao Wang; Lulin Liu; Yunhao Yang; Zihan Wang; Bangya Liu; Minghui Zheng; Zhiwen Fan
>
> **摘要:** While recent foundation models have significantly advanced robotic manipulation, these systems still struggle to autonomously recover from execution errors. Current failure-learning paradigms rely on either costly and unsafe real-world data collection or simulator-based perturbations, which introduce a severe sim-to-real gap. Furthermore, existing visual analyzers predominantly output coarse, binary diagnoses rather than the executable, trajectory-level corrections required for actual recovery. To bridge the gap between failure diagnosis and actionable recovery, we introduce Dream2Fix, a framework that synthesizes photorealistic, counterfactual failure rollouts directly from successful real-world demonstrations. By perturbing actions within a generative world model, Dream2Fix creates paired failure-correction data without relying on simulators. To ensure the generated data is physically viable for robot learning, we implement a structured verification mechanism that strictly filters rollouts for task validity, visual coherence, and kinematic safety. This engine produces a high-fidelity dataset of over 120k paired samples. Using this dataset, we fine-tune a vision-language model to jointly predict failure types and precise recovery trajectories, mapping visual anomalies directly to corrective actions. Extensive real-world robotic experiments show our approach achieves state-of-the-art correction accuracy, improving from 19.7% to 81.3% over prior baselines, and successfully enables zero-shot closed-loop failure recovery in physical deployments.
>
---
#### [new 023] Equivariant Filter Transformations for Consistent and Efficient Visual--Inertial Navigation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于视觉-惯性导航任务，解决系统一致性与计算效率问题。通过等变滤波变换，建立状态映射与误差态转换关系，提升导航精度与运行速度。**

- **链接: [https://arxiv.org/pdf/2603.24130](https://arxiv.org/pdf/2603.24130)**

> **作者:** Chungeng Tian; Fenghua He; Ning Hao
>
> **备注:** 28 papes, 11 figures
>
> **摘要:** This paper presents an equivariant filter (EqF) transformation approach for visual--inertial navigation. By establishing analytical links between EqFs with different symmetries, the proposed approach enables systematic consistency design and efficient implementation. First, we formalize the mapping from the global system state to the local error-state and prove that it induces a nonsingular linear transformation between the error-states of any two EqFs. Second, we derive transformation laws for the associated linearized error-state systems and unobservable subspaces. These results yield a general consistency design principle: for any unobservable system, a consistent EqF with a state-independent unobservable subspace can be synthesized by transforming the local coordinate chart, thereby avoiding ad hoc symmetry analysis. Third, to mitigate the computational burden arising from the non-block-diagonal Jacobians required for consistency, we propose two efficient implementation strategies. These strategies exploit the Jacobians of a simpler EqF with block-diagonal structure to accelerate covariance operations while preserving consistency. Extensive Monte Carlo simulations and real-world experiments validate the proposed approach in terms of both accuracy and runtime.
>
---
#### [new 024] Quadrature Oscillation System for Coordinated Motion in Crawling Origami Robot
- **分类: cs.RO; physics.app-ph**

- **简介: 该论文属于机器人控制任务，旨在解决无电子元件origami机器人复杂运动控制问题。研究提出一种四相位振荡系统，实现协调运动控制。**

- **链接: [https://arxiv.org/pdf/2603.23666](https://arxiv.org/pdf/2603.23666)**

> **作者:** Sean Liu; Ankur Mehta; Wenzhong Yan
>
> **备注:** 8 pages, 11 figures, Accepted to ICRA 2026
>
> **摘要:** Origami-inspired robots offer rapid, accessible design and manufacture with diverse functionalities. In particular, origami robots without conventional electronics have the unique advantage of functioning in extreme environments such as ones with high radiation or large magnetic fields. However, the absence of sophisticated control systems limits these robots to simple autonomous behaviors. In our previous studies, we developed a printable, electronics-free, and self-sustained oscillator that generates simple complementary square-wave signals. Our study presents a quadrature oscillation system capable of generating four square-wave signals a quarter-cycle out of phase, enabling four distinct states. Such control signals are important in various engineering and robotics applications, such as orchestrating limb movements in bio-inspired robots. We demonstrate the practicality and value of this oscillation system by designing and constructing an origami crawling robot that utilizes the quadrature oscillator to achieve coordinated locomotion. Together, the oscillator and robot illustrate the potential for more complex control and functions in origami robotics, paving the way for more electronics-free, rapid-design origami robots with advanced autonomous behaviors.
>
---
#### [new 025] Learning What Can Be Picked: Active Reachability Estimation for Efficient Robotic Fruit Harvesting
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于农业机器人任务，解决果实可抓取性判断问题。通过结合RGB-D感知与主动学习，提升感知效率，减少标注需求，提高采摘决策准确性。**

- **链接: [https://arxiv.org/pdf/2603.23679](https://arxiv.org/pdf/2603.23679)**

> **作者:** Nur Afsa Syeda; Mohamed Elmahallawy; Luis Fernando de la Torre; John Miller
>
> **摘要:** Agriculture remains a cornerstone of global health and economic sustainability, yet labor-intensive tasks such as harvesting high-value crops continue to face growing workforce shortages. Robotic harvesting systems offer a promising solution; however, their deployment in unstructured orchard environments is constrained by inefficient perception-to-action pipelines. In particular, existing approaches often rely on exhaustive inverse kinematics or motion planning to determine whether a target fruit is reachable, leading to unnecessary computation and delayed decision-making. Our approach combines RGB-D perception with active learning to directly learn reachability as a binary decision problem. We then leverage active learning to selectively query the most informative samples for reachability labeling, significantly reducing annotation effort while maintaining high predictive accuracy. Extensive experiments demonstrate that the proposed framework achieves accurate reachability prediction with substantially fewer labeled samples, yielding approximately 6--8% higher accuracy than random sampling and enabling label-efficient adaptation to new orchard configurations. Among the evaluated strategies, entropy- and margin-based sampling outperform Query-by-Committee and standard uncertainty sampling in low-label regimes, while all strategies converge to comparable performance as the labeled set grows. These results highlight the effectiveness of active learning for task-level perception in agricultural robotics and position our approach as a scalable alternative to computation-heavy kinematic reachability analysis. Our code is available through this https URL.
>
---
#### [new 026] PCHC: Enabling Preference Conditioned Humanoid Control via Multi-Objective Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于多目标强化学习任务，旨在解决人形机器人在多个冲突目标间动态平衡的问题。通过引入偏好条件机制，实现单一策略的多样化行为控制。**

- **链接: [https://arxiv.org/pdf/2603.24047](https://arxiv.org/pdf/2603.24047)**

> **作者:** Huanyu Li; Dewei Wang; Xinmiao Wang; Xinzhe Liu; Peng Liu; Chenjia Bai; Xuelong Li
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Humanoid robots often need to balance competing objectives, such as maximizing speed while minimizing energy consumption. While current reinforcement learning (RL) methods can master complex skills like fall recovery and perceptive locomotion, they are constrained by fixed weighting strategies that produce a single suboptimal policy, rather than providing a diverse set of solutions for sophisticated multi-objective control. In this paper, we propose a novel framework leveraging Multi-Objective Reinforcement Learning (MORL) to achieve Preference-Conditioned Humanoid Control (PCHC). Unlike conventional methods that require training a series of policies to approximate the Pareto front, our framework enables a single, preference-conditioned policy to exhibit a wide spectrum of diverse behaviors. To effectively integrate these requirements, we introduce a Beta distribution-based alignment mechanism based on preference vectors modulating a Mixture-of-Experts (MoE) module. We validated our approach on two representative humanoid tasks. Extensive simulations and real-world experiments demonstrate that the proposed framework allows the robot to adaptively shift its objective priorities in real-time based on the input preference condition.
>
---
#### [new 027] Goal-Oriented Reactive Simulation for Closed-Loop Trajectory Prediction
- **分类: cs.RO**

- **简介: 该论文属于轨迹预测任务，解决开放环训练导致的现实部署问题。提出闭环训练方法，增强模型反应能力，提升碰撞避免效果。**

- **链接: [https://arxiv.org/pdf/2603.24155](https://arxiv.org/pdf/2603.24155)**

> **作者:** Harsh Yadav; Tobias Meisen
>
> **摘要:** Current trajectory prediction models are primarily trained in an open-loop manner, which often leads to covariate shift and compounding errors when deployed in real-world, closed-loop settings. Furthermore, relying on static datasets or non-reactive log-replay simulators severs the interactive loop, preventing the ego agent from learning to actively negotiate surrounding traffic. In this work, we propose an on-policy closed-loop training paradigm optimized for high-frequency, receding horizon ego prediction. To ground the ego prediction in a realistic representation of traffic interactions and to achieve reactive consistency, we introduce a goal-oriented, transformer-based scene decoder, resulting in an inherently reactive training simulation. By exposing the ego agent to a mixture of open-loop data and simulated, self-induced states, the model learns recovery behaviors to correct its own execution errors. Extensive evaluation demonstrates that closed-loop training significantly enhances collision avoidance capabilities at high replanning frequencies, yielding relative collision rate reductions of up to 27.0% on nuScenes and 79.5% in dense DeepScenario intersections compared to open-loop baselines. Additionally, we show that a hybrid simulation combining reactive with non-reactive surrounding agents achieves optimal balance between immediate interactivity and long-term behavioral stability.
>
---
#### [new 028] Robust Distributed Cooperative Path-Following and Local Replanning for Multi-UAVs Under Differentiated Low-Altitude Paths
- **分类: cs.RO**

- **简介: 该论文属于多无人机协同路径跟踪任务，解决低空复杂环境下路径跟随与避障问题，提出分布式控制策略和高效局部重规划方法。**

- **链接: [https://arxiv.org/pdf/2603.23968](https://arxiv.org/pdf/2603.23968)**

> **作者:** Zimao Sheng; Zirui Yu; Hong'an Yang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Multiple fixed-wing unmanned aerial vehicles (multi-UAVs) encounter significant challenges in cooperative path following over complex Digital Elevation Model (DEM) low-altitude airspace, including wind field disturbances, sudden obstacles, and requirements of distributed temporal synchronization during differentiated path tracking. Existing methods lack efficient distributed coordination mechanisms for time-consistent tracking of 3D differentiated paths, fail to quantify robustness against disturbances, and lack effective online obstacle avoidance replanning capabilities. To address these gaps, a cooperative control strategy is proposed: first, the distributed cooperative path-following problem is quantified via time indices, and consistency is ensured through a distributed communication protocol; second, a longitudinal-lateral look-ahead angle adjustment method coupled with a robust guidance law is developed to achieve finite-time stabilization of path following error to zero under wind disturbances; third, an efficient local path replanning method with minimal time cost is designed for real-time online obstacle this http URL validations demonstrate the effectiveness and superiority of the $\ $proposed strategy.
>
---
#### [new 029] SafeFlow: Real-Time Text-Driven Humanoid Whole-Body Control via Physics-Guided Rectified Flow and Selective Safety Gating
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于人形机器人实时控制任务，解决物理不合规和安全问题。提出SafeFlow框架，结合物理引导生成与三阶段安全门，提升运动轨迹的可行性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.23983](https://arxiv.org/pdf/2603.23983)**

> **作者:** Hanbyel Cho; Sang-Hun Kim; Jeonguk Kang; Donghan Koo
>
> **备注:** Project Page: this https URL
>
> **摘要:** Recent advances in real-time interactive text-driven motion generation have enabled humanoids to perform diverse behaviors. However, kinematics-only generators often exhibit physical hallucinations, producing motion trajectories that are physically infeasible to track with a downstream motion tracking controller or unsafe for real-world deployment. These failures often arise from the lack of explicit physics-aware objectives for real-robot execution and become more severe under out-of-distribution (OOD) user inputs. Hence, we propose SafeFlow, a text-driven humanoid whole-body control framework that combines physics-guided motion generation with a 3-Stage Safety Gate driven by explicit risk indicators. SafeFlow adopts a two-level architecture. At the high level, we generate motion trajectories using Physics-Guided Rectified Flow Matching in a VAE latent space to improve real-robot executability, and further accelerate sampling via Reflow to reduce the number of function evaluations (NFE) for real-time control. The 3-Stage Safety Gate enables selective execution by detecting semantic OOD prompts using a Mahalanobis score in text-embedding space, filtering unstable generations via a directional sensitivity discrepancy metric, and enforcing final hard kinematic constraints such as joint and velocity limits before passing the generated trajectory to a low-level motion tracking controller. Extensive experiments on the Unitree G1 demonstrate that SafeFlow outperforms prior diffusion-based methods in success rate, physical compliance, and inference speed, while maintaining diverse expressiveness.
>
---
#### [new 030] Decentralized End-to-End Multi-AAV Pursuit Using Predictive Spatio-Temporal Observation via Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于多无人机协同追踪任务，解决复杂环境中感知不确定性问题。提出一种基于深度强化学习的端到端框架，利用预测时空观测实现高效协作追踪。**

- **链接: [https://arxiv.org/pdf/2603.24238](https://arxiv.org/pdf/2603.24238)**

> **作者:** Yude Li; Zhexuan Zhou; Huizhe Li; Yanke Sun; Yenan Wu; Yichen Lai; Yiming Wang; Youmin Gong; Jie Mei
>
> **摘要:** Decentralized cooperative pursuit in cluttered environments is challenging for autonomous aerial swarms, especially under partial and noisy perception. Existing methods often rely on abstracted geometric features or privileged ground-truth states, and therefore sidestep perceptual uncertainty in real-world settings. We propose a decentralized end-to-end multi-agent reinforcement learning (MARL) framework that maps raw LiDAR observations directly to continuous control commands. Central to the framework is the Predictive Spatio-Temporal Observation (PSTO), an egocentric grid representation that aligns obstacle geometry with predictive adversarial intent and teammate motion in a unified, fixed-resolution projection. Built on PSTO, a single decentralized policy enables agents to navigate static obstacles, intercept dynamic targets, and maintain cooperative encirclement. Simulations demonstrate that the proposed method achieves superior capture efficiency and competitive success rates compared to state-of-the-art learning-based approaches relying on privileged obstacle information. Furthermore, the unified policy scales seamlessly across different team sizes without retraining. Finally, fully autonomous outdoor experiments validate the framework on a quadrotor swarm relying on only onboard sensing and computing.
>
---
#### [new 031] Event-Driven Proactive Assistive Manipulation with Grounded Vision-Language Planning
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决传统依赖用户指令的辅助操作问题。通过事件驱动机制，机器人根据环境变化主动提供帮助，提升协作效率与智能化水平。**

- **链接: [https://arxiv.org/pdf/2603.23950](https://arxiv.org/pdf/2603.23950)**

> **作者:** Fengkai Liu; Hao Su; Haozhuang Chi; Rui Geng; Congzhi Ren; Xuqing Liu; Yucheng Xu; Yuichi Ohsita; Liyun Zhang
>
> **摘要:** Assistance in collaborative manipulation is often initiated by user instructions, making high-level reasoning request-driven. In fluent human teamwork, however, partners often infer the next helpful step from the observed outcome of an action rather than waiting for instructions. Motivated by this, we introduce a shift from request-driven assistance to event-driven proactive assistance, where robot actions are initiated by workspace state transitions induced by human--object interactions rather than user-provided task instructions. To this end, we propose an event-driven framework that tracks interaction progress with an event monitor and, upon event completion, extracts stabilized pre/post snapshots that characterize the resulting state transition. Given the stabilized snapshots, the planner analyzes the implied state transition to infer a task-level goal and decide whether to intervene; if so, it generates a sequence of assistive actions. To make outputs executable and verifiable, we restrict actions to a set of action primitives and reference objects via integer IDs. We evaluate the framework on a real tabletop number-block collaboration task, demonstrating that explicit pre/post state-change evidence improves proactive completion on solvable scenes and appropriate waiting on unsolvable ones.
>
---
#### [new 032] Object Search in Partially-Known Environments via LLM-informed Model-based Planning and Prompt Selection
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于目标搜索任务，解决部分已知环境中高效搜索的问题。通过LLM引导的建模规划和提示选择方法，提升搜索性能与效率。**

- **链接: [https://arxiv.org/pdf/2603.23800](https://arxiv.org/pdf/2603.23800)**

> **作者:** Abhishek Paudel; Abhish Khanal; Raihan I. Arnob; Shahriar Hossain; Gregory J. Stein
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** We present a novel LLM-informed model-based planning framework, and a novel prompt selection method, for object search in partially-known environments. Our approach uses an LLM to estimate statistics about the likelihood of finding the target object when searching various locations throughout the scene that, combined with travel costs extracted from the environment map, are used to instantiate a model, thus using the LLM to inform planning and achieve effective search performance. Moreover, the abstraction upon which our approach relies is amenable to deployment-time model selection via the recent offline replay approach, an insight we leverage to enable fast prompt and LLM selection during deployment. Simulation experiments demonstrate that our LLM-informed model-based planning approach outperforms the baseline planning strategy that fully relies on LLM and optimistic strategy with as much as 11.8% and 39.2% improvements respectively, and our bandit-like selection approach enables quick selection of best prompts and LLMs resulting in 6.5% lower average cost and 33.8% lower average cumulative regret over baseline UCB bandit selection. Real-robot experiments in an apartment demonstrate similar improvements and so further validate our approach.
>
---
#### [new 033] TAG: Target-Agnostic Guidance for Stable Object-Centric Inference in Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决 clutter 场景下的目标误识别问题。提出 TAG 方法，在推理阶段增强目标证据，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.24584](https://arxiv.org/pdf/2603.24584)**

> **作者:** Jiaying Zhou; Zhihao Zhan; Ruifeng Zhai; Qinhan Lyu; Hao Liu; Keze Wang; Liang Lin; Guangrun Wang
>
> **摘要:** Vision--Language--Action (VLA) policies have shown strong progress in mapping language instructions and visual observations to robotic actions, yet their reliability degrades in cluttered scenes with distractors. By analyzing failure cases, we find that many errors do not arise from infeasible motions, but from instance-level grounding failures: the policy often produces a plausible grasp trajectory that lands slightly off-target or even on the wrong object instance. To address this issue, we propose TAG (Target-Agnostic Guidance), a simple inference-time guidance mechanism that explicitly reduces distractor- and appearance-induced bias in VLA policies. Inspired by classifier-free guidance (CFG), TAG contrasts policy predictions under the original observation and an object-erased observation, and uses their difference as a residual steering signal that strengthens the influence of object evidence in the decision process. TAG does not require modifying the policy architecture and can be integrated with existing VLA policies with minimal training and inference changes. We evaluate TAG on standard manipulation benchmarks, including LIBERO, LIBERO-Plus, and VLABench, where it consistently improves robustness under clutter and reduces near-miss and wrong-object executions.
>
---
#### [new 034] Grounding Vision and Language to 3D Masks for Long-Horizon Box Rearrangement
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究3D环境中的长时序物体重排任务，解决自然语言指令下多步骤操作问题。提出RAMP-3D模型，通过3D掩码预测实现有效规划。**

- **链接: [https://arxiv.org/pdf/2603.23676](https://arxiv.org/pdf/2603.23676)**

> **作者:** Ashish Malik; Caleb Lowe; Aayam Shrestha; Stefan Lee; Fuxin Li; Alan Fern
>
> **摘要:** We study long-horizon planning in 3D environments from under-specified natural-language goals using only visual observations, focusing on multi-step 3D box rearrangement tasks. Existing approaches typically rely on symbolic planners with brittle relational grounding of states and goals, or on direct action-sequence generation from 2D vision-language models (VLMs). Both approaches struggle with reasoning over many objects, rich 3D geometry, and implicit semantic constraints. Recent advances in 3D VLMs demonstrate strong grounding of natural-language referents to 3D segmentation masks, suggesting the potential for more general planning capabilities. We extend existing 3D grounding models and propose Reactive Action Mask Planner (RAMP-3D), which formulates long-horizon planning as sequential reactive prediction of paired 3D masks: a "which-object" mask indicating what to pick and a "which-target-region" mask specifying where to place it. The resulting system processes RGB-D observations and natural-language task specifications to reactively generate multi-step pick-and-place actions for 3D box rearrangement. We conduct experiments across 11 task variants in warehouse-style environments with 1-30 boxes and diverse natural-language constraints. RAMP-3D achieves 79.5% success rate on long-horizon rearrangement tasks and significantly outperforms 2D VLM-based baselines, establishing mask-based reactive policies as a promising alternative to symbolic pipelines for long-horizon planning.
>
---
#### [new 035] SLAT-Phys: Fast Material Property Field Prediction from Structured 3D Latents
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出SLAT-Phys，用于从单张RGB图像快速预测3D资产的材料属性场，解决传统方法计算慢、依赖3D信息的问题。**

- **链接: [https://arxiv.org/pdf/2603.23973](https://arxiv.org/pdf/2603.23973)**

> **作者:** Rocktim Jyoti Das; Dinesh Manocha
>
> **备注:** 8 page, 4 figures
>
> **摘要:** Estimating the material property field of 3D assets is critical for physics-based simulation, robotics, and digital twin generation. Existing vision-based approaches are either too expensive and slow or rely on 3D information. We present SLAT-Phys, an end-to-end method that predicts spatially varying material property fields of 3D assets directly from a single RGB image without explicit 3D reconstruction. Our approach leverages spatially organised latent features from a pretrained 3D asset generation model that encodes rich geometry and semantic prior, and trains a lightweight neural decoder to estimate Young's modulus, density, and Poisson's ratio. The coarse volumetric layout and semantic cues of the latent representation about object geometry and appearance enable accurate material estimation. Our experiments demonstrate that our method provides competitive accuracy in predicting continuous material parameters when compared against prior approaches, while significantly reducing computation time. In particular, SLAT-Phys requires only 9.9 seconds per object on an NVIDIA RTXA5000 GPU and avoids reconstruction and voxelization preprocessing. This results in 120x speedup compared to prior methods and enables faster material property estimation from a single image.
>
---
#### [new 036] CoordLight: Learning Decentralized Coordination for Network-Wide Traffic Signal Control
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于交通信号控制任务，旨在解决多智能体协作中的观测不完全和协调难题。提出CoordLight框架，结合QDSE状态表示和NAPO算法，提升路口协同决策能力，优化网络级交通流。**

- **链接: [https://arxiv.org/pdf/2603.24366](https://arxiv.org/pdf/2603.24366)**

> **作者:** Yifeng Zhang; Harsh Goel; Peizhuo Li; Mehul Damani; Sandeep Chinchali; Guillaume Sartoretti
>
> **备注:** \c{opyright} 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Adaptive traffic signal control (ATSC) is crucial in alleviating congestion, maximizing throughput and promoting sustainable mobility in ever-expanding cities. Multi-Agent Reinforcement Learning (MARL) has recently shown significant potential in addressing complex traffic dynamics, but the intricacies of partial observability and coordination in decentralized environments still remain key challenges in formulating scalable and efficient control strategies. To address these challenges, we present CoordLight, a MARL-based framework designed to improve intra-neighborhood traffic by enhancing decision-making at individual junctions (agents), as well as coordination with neighboring agents, thereby scaling up to network-level traffic optimization. Specifically, we introduce the Queue Dynamic State Encoding (QDSE), a novel state representation based on vehicle queuing models, which strengthens the agents' capability to analyze, predict, and respond to local traffic dynamics. We further propose an advanced MARL algorithm, named Neighbor-aware Policy Optimization (NAPO). It integrates an attention mechanism that discerns the state and action dependencies among adjacent agents, aiming to facilitate more coordinated decision-making, and to improve policy learning updates through robust advantage calculation. This enables agents to identify and prioritize crucial interactions with influential neighbors, thus enhancing the targeted coordination and collaboration among agents. Through comprehensive evaluations against state-of-the-art traffic signal control methods over three real-world traffic datasets composed of up to 196 intersections, we empirically show that CoordLight consistently exhibits superior performance across diverse traffic networks with varying traffic flows. The code is available at this https URL
>
---
#### [new 037] Towards Safe Learning-Based Non-Linear Model Predictive Control through Recurrent Neural Network Modeling
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于控制领域，旨在解决NMPC在线计算成本高的问题。通过引入序列神经策略和安全机制，提升控制安全性与效率。**

- **链接: [https://arxiv.org/pdf/2603.24503](https://arxiv.org/pdf/2603.24503)**

> **作者:** Mihaela-Larisa Clement; Mónika Farsang; Agnes Poks; Johannes Edelmann; Manfred Plöchl; Radu Grosu; Ezio Bartocci
>
> **摘要:** The practical deployment of nonlinear model predictive control (NMPC) is often limited by online computation: solving a nonlinear program at high control rates can be expensive on embedded hardware, especially when models are complex or horizons are long. Learning-based NMPC approximations shift this computation offline but typically demand large expert datasets and costly training. We propose Sequential-AMPC, a sequential neural policy that generates MPC candidate control sequences by sharing parameters across the prediction horizon. For deployment, we wrap the policy in a safety-augmented online evaluation and fallback mechanism, yielding Safe Sequential-AMPC. Compared to a naive feedforward policy baseline across several benchmarks, Sequential-AMPC requires substantially fewer expert MPC rollouts and yields candidate sequences with higher feasibility rates and improved closed-loop safety. On high-dimensional systems, it also exhibits better learning dynamics and performance in fewer epochs while maintaining stable validation improvement where the feedforward baseline can stagnate.
>
---
#### [new 038] LongTail Driving Scenarios with Reasoning Traces: The KITScenes LongTail Dataset
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出KITScenes LongTail数据集，用于解决自动驾驶中罕见场景的泛化问题。通过多视角视频、轨迹和推理轨迹，支持少样本学习与指令遵循评估。**

- **链接: [https://arxiv.org/pdf/2603.23607](https://arxiv.org/pdf/2603.23607)**

> **作者:** Royden Wagner; Omer Sahin Tas; Jaime Villa; Felix Hauser; Yinzhe Shen; Marlon Steiner; Dominik Strutz; Carlos Fernandez; Christian Kinzig; Guillermo S. Guitierrez-Cabello; Hendrik Königshof; Fabian Immel; Richard Schwarzkopf; Nils Alexander Rack; Kevin Rösch; Kaiwen Wang; Jan-Hendrik Pauls; Martin Lauer; Igor Gilitschenski; Holger Caesar; Christoph Stiller
>
> **备注:** 21 pages
>
> **摘要:** In real-world domains such as self-driving, generalization to rare scenarios remains a fundamental challenge. To address this, we introduce a new dataset designed for end-to-end driving that focuses on long-tail driving events. We provide multi-view video data, trajectories, high-level instructions, and detailed reasoning traces, facilitating in-context learning and few-shot generalization. The resulting benchmark for multimodal models, such as VLMs and VLAs, goes beyond safety and comfort metrics by evaluating instruction following and semantic coherence between model outputs. The multilingual reasoning traces in English, Spanish, and Chinese are from domain experts with diverse cultural backgrounds. Thus, our dataset is a unique resource for studying how different forms of reasoning affect driving competence. Our dataset is available at: this https URL
>
---
#### [new 039] High-Density Automated Valet Parking with Relocation-Free Sequential Operations
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自动化停车任务，解决高密度停车与车辆取回中的无移车问题。通过设计无移车序列和布局优化，提升空间利用率。**

- **链接: [https://arxiv.org/pdf/2603.23803](https://arxiv.org/pdf/2603.23803)**

> **作者:** Bon Choe; Minhee Kang; Heejin Ahn
>
> **备注:** 7 pages, 6 figure. The results from all experiments are available at: this https URL
>
> **摘要:** In this paper, we present DROP, high-Density Relocation-free sequential OPerations in automated valet parking. DROP addresses the challenges in high-density parking & vehicle retrieval without relocations. Each challenge is handled by jointly providing area-efficient layouts and relocation-free parking & exit sequences, considering accessibility with relocation-free sequential operations. To generate such sequences, relocation-free constraints are formulated as explicit logical conditions expressed in boolean variables. Recursive search strategies are employed to derive the logical conditions and enumerate relocation-free sequences under sequential constraints. We demonstrate the effectiveness of our framework through extensive simulations, showing its potential to significantly improve area utilization with relocation-free constraints. We also examine its viability on an application problem with prescribed operational order. The results from all experiments are available at: this https URL.
>
---
#### [new 040] DreamerAD: Efficient Reinforcement Learning via Latent World Model for Autonomous Driving
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决自动驾驶中训练成本高、安全性差的问题。通过构建潜空间模型，提升训练效率并保持视觉可解释性。**

- **链接: [https://arxiv.org/pdf/2603.24587](https://arxiv.org/pdf/2603.24587)**

> **作者:** Pengxuan Yang; Yupeng Zheng; Deheng Qian; Zebin Xing; Qichao Zhang; Linbo Wang; Yichen Zhang; Shaoyu Guo; Zhongpu Xia; Qiang Chen; Junyu Han; Lingyun Xu; Yifeng Pan; Dongbin Zhao
>
> **备注:** first version
>
> **摘要:** We introduce DreamerAD, the first latent world model framework that enables efficient reinforcement learning for autonomous driving by compressing diffusion sampling from 100 steps to 1 - achieving 80x speedup while maintaining visual interpretability. Training RL policies on real-world driving data incurs prohibitive costs and safety risks. While existing pixel-level diffusion world models enable safe imagination-based training, they suffer from multi-step diffusion inference latency (2s/frame) that prevents high-frequency RL interaction. Our approach leverages denoised latent features from video generation models through three key mechanisms: (1) shortcut forcing that reduces sampling complexity via recursive multi-resolution step compression, (2) an autoregressive dense reward model operating directly on latent representations for fine-grained credit assignment, and (3) Gaussian vocabulary sampling for GRPO that constrains exploration to physically plausible trajectories. DreamerAD achieves 87.7 EPDMS on NavSim v2, establishing state-of-the-art performance and demonstrating that latent-space RL is effective for autonomous driving.
>
---
#### [new 041] Engagement-Zone-Aware Input-Constrained Guidance for Safe Target Interception in Contested Environments
- **分类: eess.SY; cs.MA; cs.RO**

- **简介: 该论文属于目标拦截任务，解决对抗环境中安全拦截问题。通过构建防御者诱导的拦截区和输入饱和模型，提出一种安全引导策略，确保拦截同时避免威胁。**

- **链接: [https://arxiv.org/pdf/2603.23649](https://arxiv.org/pdf/2603.23649)**

> **作者:** Praveen Kumar Ranjan; Abhinav Sinha; Yongcan Cao
>
> **摘要:** We address target interception in contested environments in the presence of multiple defenders whose interception capability is limited by finite ranges. Conventional methods typically impose conservative stand-off constraints based on maximum engagement distance and neglect the interceptors' actuator limitations. Instead, we formulate safety constraints using defender-induced engagement zones. To account for actuator limits, the vehicle model is augmented with input saturation dynamics. A time-varying safe-set tightening parameter is introduced to compensate for transient constraint violations induced by actuator dynamics. To ensure scalable safety enforcement in multi-defender scenarios, a smooth aggregate safety function is constructed using a log-sum-exp operator combining individual threat measures associated with each defender's capability. A smooth switching guidance strategy is then developed to coordinate interception and safety objectives. The attacker pursues the target when sufficiently distant from threat boundaries and progressively activates evasive motion as the EZ boundaries are approached. The resulting controller relies only on relative measurements and does not require knowledge of defender control inputs, thus facilitating a fully distributed and scalable implementation. Rigorous analysis provides sufficient conditions guaranteeing target interception, practical safety with respect to all defender engagement zones, and satisfaction of actuator bounds. An input-constrained guidance law based on conservative stand-off distance is also developed to quantify the conservatism of maximum-range-based safety formulations. Simulations with stationary and maneuvering defenders demonstrate that the proposed formulation yields shorter interception paths and reduced interception time compared with conventional methods while maintaining safety throughout the engagement.
>
---
#### [new 042] Latent-WAM: Latent World Action Modeling for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Latent-WAM，用于端到端自动驾驶的轨迹规划任务。针对现有方法表示压缩不足、空间理解有限和时间动态利用不充分的问题，设计了两个核心模块提升规划效果。**

- **链接: [https://arxiv.org/pdf/2603.24581](https://arxiv.org/pdf/2603.24581)**

> **作者:** Linbo Wang; Yupeng Zheng; Qiang Chen; Shiwei Li; Yichen Zhang; Zebin Xing; Qichao Zhang; Xiang Li; Deheng Qian; Pengxuan Yang; Yihang Dong; Ce Hao; Xiaoqing Ye; Junyu han; Yifeng Pan; Dongbin Zhao
>
> **摘要:** We introduce Latent-WAM, an efficient end-to-end autonomous driving framework that achieves strong trajectory planning through spatially-aware and dynamics-informed latent world representations. Existing world-model-based planners suffer from inadequately compressed representations, limited spatial understanding, and underutilized temporal dynamics, resulting in sub-optimal planning under constrained data and compute budgets. Latent-WAM addresses these limitations with two core modules: a Spatial-Aware Compressive World Encoder (SCWE) that distills geometric knowledge from a foundation model and compresses multi-view images into compact scene tokens via learnable queries, and a Dynamic Latent World Model (DLWM) that employs a causal Transformer to autoregressively predict future world status conditioned on historical visual and motion representations. Extensive experiments on NAVSIM v2 and HUGSIM demonstrate new state-of-the-art results: 89.3 EPDMS on NAVSIM v2 and 28.9 HD-Score on HUGSIM, surpassing the best prior perception-free method by 3.2 EPDMS with significantly less training data and a compact 104M-parameter model.
>
---
#### [new 043] Do 3D Large Language Models Really Understand 3D Spatial Relationships?
- **分类: cs.CL; cs.RO**

- **简介: 该论文属于3D视觉语言理解任务，旨在解决3D-LLMs是否真正理解空间关系的问题。研究发现现有模型依赖文本线索而非3D信息，提出新基准Real-3DQA和3D重加权训练方法提升性能。**

- **链接: [https://arxiv.org/pdf/2603.23523](https://arxiv.org/pdf/2603.23523)**

> **作者:** Xianzheng Ma; Tao Sun; Shuai Chen; Yash Bhalgat; Jindong Gu; Angel X Chang; Iro Armeni; Iro Laina; Songyou Peng; Victor Adrian Prisacariu
>
> **备注:** ICLR 2026
>
> **摘要:** Recent 3D Large-Language Models (3D-LLMs) claim to understand 3D worlds, especially spatial relationships among objects. Yet, we find that simply fine-tuning a language model on text-only question-answer pairs can perform comparably or even surpass these methods on the SQA3D benchmark without using any 3D input. This indicates that the SQA3D benchmark may not be able to detect if the model exploits textual shortcuts rather than engages in 3D-aware reasoning. To address this issue, we introduce Real-3DQA, a more rigorous evaluation benchmark that filters out easy-to-guess questions and introduces a structured taxonomy to assess various aspects of 3D reasoning. Experiments on Real-3DQA confirm that existing 3D-LLMs struggle with spatial relationships once simple cues are removed. We further propose a 3D-reweighted training objective that guides model to rely more on 3D visual clues, substantially enhancing 3D-LLMs performance in spatial reasoning tasks. Our findings underscore the need for robust benchmarks and tailored training strategies to advance genuine 3D vision-language understanding. Project page: this https URL.
>
---
#### [new 044] Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决仓库自动化中长期运行的冲突路径问题。通过结合强化学习与优先规划，提升系统吞吐量和适应性。**

- **链接: [https://arxiv.org/pdf/2603.23838](https://arxiv.org/pdf/2603.23838)**

> **作者:** Han Zheng; Yining Ma; Brandon Araki; Jingkai Chen; Cathy Wu
>
> **摘要:** Lifelong Multi-Agent Path Finding (MAPF) is critical for modern warehouse automation, which requires multiple robots to continuously navigate conflict-free paths to optimize the overall system throughput. However, the complexity of warehouse environments and the long-term dynamics of lifelong MAPF often demand costly adaptations to classical search-based solvers. While machine learning methods have been explored, their superiority over search-based methods remains inconclusive. In this paper, we introduce Reinforcement Learning (RL) guided Rolling Horizon Prioritized Planning (RL-RH-PP), the first framework integrating RL with search-based planning for lifelong MAPF. Specifically, we leverage classical Prioritized Planning (PP) as a backbone for its simplicity and flexibility in integrating with a learning-based priority assignment policy. By formulating dynamic priority assignment as a Partially Observable Markov Decision Process (POMDP), RL-RH-PP exploits the sequential decision-making nature of lifelong planning while delegating complex spatial-temporal interactions among agents to reinforcement learning. An attention-based neural network autoregressively decodes priority orders on-the-fly, enabling efficient sequential single-agent planning by the PP planner. Evaluations in realistic warehouse simulations show that RL-RH-PP achieves the highest total throughput among baselines and generalizes effectively across agent densities, planning horizons, and warehouse layouts. Our interpretive analyses reveal that RL-RH-PP proactively prioritizes congested agents and strategically redirects agents from congestion, easing traffic flow and boosting throughput. These findings highlight the potential of learning-guided approaches to augment traditional heuristics in modern warehouse automation.
>
---
#### [new 045] Off-Policy Safe Reinforcement Learning with Constrained Optimistic Exploration
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于安全强化学习任务，旨在解决离策略方法中因探索和成本估计偏差导致的约束违反问题。提出COX-Q算法，结合成本约束探索与保守价值学习，提升安全性与样本效率。**

- **链接: [https://arxiv.org/pdf/2603.23889](https://arxiv.org/pdf/2603.23889)**

> **作者:** Guopeng Li; Matthijs T.J. Spaan; Julian F.P. Kooij
>
> **备注:** 21 pages, 9 figures, accepted by ICLR 2026 poster
>
> **摘要:** When safety is formulated as a limit of cumulative cost, safe reinforcement learning (RL) aims to learn policies that maximize return subject to the cost constraint in data collection and deployment. Off-policy safe RL methods, although offering high sample efficiency, suffer from constraint violations due to cost-agnostic exploration and estimation bias in cumulative cost. To address this issue, we propose Constrained Optimistic eXploration Q-learning (COX-Q), an off-policy safe RL algorithm that integrates cost-bounded online exploration and conservative offline distributional value learning. First, we introduce a novel cost-constrained optimistic exploration strategy that resolves gradient conflicts between reward and cost in the action space and adaptively adjusts the trust region to control the training cost. Second, we adopt truncated quantile critics to stabilize the cost value learning. Quantile critics also quantify epistemic uncertainty to guide exploration. Experiments on safe velocity, safe navigation, and autonomous driving tasks demonstrate that COX-Q achieves high sample efficiency, competitive test safety performance, and controlled data collection cost. The results highlight COX-Q as a promising RL method for safety-critical applications.
>
---
#### [new 046] Aesthetics of Robot-Mediated Applied Drama: A Case Study on REMind
- **分类: cs.HC; cs.RO**

- **简介: 论文探讨了机器人辅助应用戏剧（RMAD）的设计，聚焦于如何在机器人表达能力有限的情况下，创造情感和美学吸引力。任务是提升机器人戏剧的体验设计，解决机器人表达不足的问题，通过REMind案例展示艺术经验对设计的贡献。**

- **链接: [https://arxiv.org/pdf/2603.23816](https://arxiv.org/pdf/2603.23816)**

> **作者:** Elaheh Sanoubari; Alicia Pan; Keith Rebello; Neil Fernandes; Andrew Houston; Kerstin Dautenhahn
>
> **备注:** 15 pages, 6 figures. Preprint submitted to the 18th International Conference on Social Robotics (ICSR 2026)
>
> **摘要:** Social robots are increasingly used in education, but most applications cast them as tutors offering explanation-based instruction. We explore an alternative: Robot-Mediated Applied Drama (RMAD), in which robots function as life-like puppets in interactive dramatic experiences designed to support reflection and social-emotional learning. This paper presents REMind, an anti-bullying robot role-play game that helps children rehearse bystander intervention and peer support. We focus on a central design challenge in RMAD: how to make robot drama emotionally and aesthetically engaging despite the limited expressive capacities of current robotic platforms. Through the development of REMind, we show how performing arts expertise informed this process, and argue that the aesthetics of robot drama arise from the coordinated design of the wider experience, not from robot expressivity alone.
>
---
## 更新

#### [replaced 001] Point Bridge: 3D Representations for Cross Domain Policy Learning
- **分类: cs.RO**

- **简介: 该论文提出Point Bridge，解决仿真到现实的策略迁移问题。通过点云表示和VLMs，实现无需视觉对齐的跨域策略学习，提升合成数据有效性。**

- **链接: [https://arxiv.org/pdf/2601.16212](https://arxiv.org/pdf/2601.16212)**

> **作者:** Siddhant Haldar; Lars Johannsmeier; Lerrel Pinto; Abhishek Gupta; Dieter Fox; Yashraj Narang; Ajay Mandlekar
>
> **摘要:** Robot foundation models are beginning to deliver on the promise of generalist robotic agents, yet progress remains constrained by the scarcity of large-scale real-world manipulation datasets. Simulation and synthetic data generation offer a scalable alternative, but their usefulness is limited by the visual domain gap between simulation and reality. In this work, we present Point Bridge, a framework that leverages unified, domain-agnostic point-based representations to unlock synthetic datasets for zero-shot sim-to-real policy transfer, without explicit visual or object-level alignment. Point Bridge combines automated point-based representation extraction via Vision-Language Models (VLMs), transformer-based policy learning, and efficient inference-time pipelines to train capable real-world manipulation agents using only synthetic data. With additional co-training on small sets of real demonstrations, Point Bridge further improves performance, substantially outperforming prior vision-based sim-and-real co-training methods. It achieves up to 44% gains in zero-shot sim-to-real transfer and up to 66% with limited real data across both single-task and multitask settings. Videos of the robot are best viewed at: this https URL
>
---
#### [replaced 002] E0: Enhancing Generalization and Fine-Grained Control in VLA Models via Tweedie Discrete Diffusion
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决VLA模型泛化能力差和动作控制粗略的问题。提出E0框架，通过离散扩散方法提升动作生成的精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.21542](https://arxiv.org/pdf/2511.21542)**

> **作者:** Zhihao Zhan; Jiaying Zhou; Likui Zhang; Qinhan Lv; Hao Liu; Jusheng Zhang; Weizheng Li; Ziliang Chen; Tianshui Chen; Ruifeng Zhai; Keze Wang; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. However, existing VLA systems still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We argue that these limitations are closely tied to the structural properties of actions in VLA settings, including the inherent multi-peaked nature of action distributions, the token-based symbolic reasoning of pretrained VLM/VLA backbones, and the effective finite resolution imposed by real-world robotic control. Motivated by these properties, we introduce E0, a tweedie discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. By operating in a discrete action space with a principled diffusion process, E0 naturally aligns with token-based reasoning, supports fine-grained yet executable action control, and avoids the distributional mismatch of masking-based discrete diffusion. We further introduce a spherical viewpoint perturbation augmentation to enhance robustness to camera shifts without additional data. Experiments on LIBERO, VLABench, ManiSkill, and a real-world Franka arm demonstrate that E0 achieves state-of-the-art performance across 14 diverse environments, outperforming strong baselines by 10.7% on average.
>
---
#### [replaced 003] Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出Xiaomi-Robotics-0，一个用于实时执行的视觉-语言-动作模型，解决机器人在真实环境中高效、精准操作的问题。通过优化训练和部署策略，提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.12684](https://arxiv.org/pdf/2602.12684)**

> **作者:** Rui Cai; Jun Guo; Xinze He; Piaopiao Jin; Jie Li; Bingxuan Lin; Futeng Liu; Wei Liu; Fei Ma; Kun Ma; Feng Qiu; Heng Qu; Yifei Su; Qiao Sun; Dong Wang; Donghao Wang; Yunhong Wang; Rujie Wu; Diyun Xiang; Yu Yang; Hangjun Ye; Yuan Zhang; Quanyun Zhou
>
> **备注:** Project page: this https URL
>
> **摘要:** In this report, we introduce Xiaomi-Robotics-0, an advanced vision-language-action (VLA) model optimized for high performance and fast and smooth real-time execution. The key to our method lies in a carefully designed training recipe and deployment strategy. Xiaomi-Robotics-0 is first pre-trained on large-scale cross-embodiment robot trajectories and vision-language data, endowing it with broad and generalizable action-generation capabilities while avoiding catastrophic forgetting of the visual-semantic knowledge of the underlying pre-trained VLM. During post-training, we propose several techniques for training the VLA model for asynchronous execution to address the inference latency during real-robot rollouts. During deployment, we carefully align the timesteps of consecutive predicted action chunks to ensure continuous and seamless real-time rollouts. We evaluate Xiaomi-Robotics-0 extensively in simulation benchmarks and on two challenging real-robot tasks that require precise and dexterous bimanual manipulation. Results show that our method achieves state-of-the-art performance across all simulation benchmarks. Moreover, Xiaomi-Robotics-0 can roll out fast and smoothly on real robots using a consumer-grade GPU, achieving high success rates and throughput on both real-robot tasks. To facilitate future research, code and model checkpoints are open-sourced at this https URL
>
---
#### [replaced 004] MiniBEE: A New Form Factor for Compact Bimanual Dexterity
- **分类: cs.RO**

- **简介: 该论文提出MiniBEE，一种紧凑的双臂操作系统，解决传统双臂机器人系统复杂、空间利用率低的问题。通过优化设计提升操作灵巧性。**

- **链接: [https://arxiv.org/pdf/2510.01603](https://arxiv.org/pdf/2510.01603)**

> **作者:** Sharfin Islam; Zewen Chen; Zhanpeng He; Swapneel Bhatt; Andres Permuy; Brock Taylor; James Vickery; Zhengbin Lu; Cheng Zhang; Pedro Piacenza; Matei Ciocarlie
>
> **摘要:** Bimanual robot manipulators can achieve impressive dexterity, but typically rely on two full six- or seven- degree-of-freedom arms so that paired grippers can coordinate effectively. This traditional framework increases system complexity while only exploiting a fraction of the overall workspace for dexterous interaction. We introduce the MiniBEE (Miniature Bimanual End-effector), a compact system in which two reduced-mobility arms (3+ DOF each) are coupled into a kinematic chain that preserves full relative positioning between grippers. To guide our design, we formulate a kinematic dexterity metric that enlarges the dexterous workspace while keeping the mechanism lightweight and wearable. The resulting system supports two complementary modes: (i) wearable kinesthetic data collection with self-tracked gripper poses, and (ii) deployment on a standard robot arm, extending dexterity across its entire workspace. We present kinematic analysis and design optimization methods for maximizing dexterous range, and demonstrate an end-to-end pipeline in which wearable demonstrations train imitation learning policies that perform robust, real-world bimanual manipulation.
>
---
#### [replaced 005] Pixel-level Scene Understanding in One Token: Visual States Need What-is-Where Composition
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉状态表示学习任务，旨在解决机器人在动态环境中如何有效编码场景元素及其位置的问题。提出CroBo框架，通过全局到局部重建学习细粒度的视觉状态，支持序列决策。**

- **链接: [https://arxiv.org/pdf/2603.13904](https://arxiv.org/pdf/2603.13904)**

> **作者:** Seokmin Lee; Yunghee Lee; Byeonghyun Pak; Byeongju Woo
>
> **备注:** Accepted to CVPR 2026 Workshop: Pixel-level Video Understanding in the Wild
>
> **摘要:** For robotic agents operating in dynamic environments, learning visual state representations from streaming video observations is essential for sequential decision making. Recent self-supervised learning methods have shown strong transferability across vision tasks, but they do not explicitly address what a good visual state should encode. We argue that effective visual states must capture what-is-where by jointly encoding the semantic identities of scene elements and their spatial locations, enabling reliable detection of subtle dynamics across observations. To this end, we propose CroBo, a visual state representation learning framework based on a global-to-local reconstruction objective. Given a reference observation compressed into a compact bottleneck token, CroBo learns to reconstruct heavily masked patches in a local target crop from sparse visible cues, using the global bottleneck token as context. This learning objective encourages the bottleneck token to encode a fine-grained representation of scene-wide semantic entities, including their identities, spatial locations, and configurations. As a result, the learned visual states reveal how scene elements move and interact over time, supporting sequential decision making. We evaluate CroBo on diverse vision-based robot policy learning benchmarks, where it achieves state-of-the-art performance. Reconstruction analyses and perceptual straightness experiments further show that the learned representations preserve pixel-level scene composition and encode what-moves-where across observations. Project page available at: this https URL.
>
---
#### [replaced 006] A Hybrid Neural-Assisted Unscented Kalman Filter for Unmanned Ground Vehicle Navigation
- **分类: cs.RO**

- **简介: 该论文属于无人地面车辆导航任务，旨在解决传统滤波器在动态环境中的噪声建模不足问题。通过融合深度学习与卡尔曼滤波，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2603.11649](https://arxiv.org/pdf/2603.11649)**

> **作者:** Gal Versano; Itzik Klein
>
> **摘要:** Modern autonomous navigation for unmanned ground vehicles relies on different estimators to fuse inertial sensors and GNSS measurements. However, the constant noise covariance matrices often struggle to account for dynamic real-world conditions. In this work we propose a hybrid estimation framework that bridges classical state estimation foundations with modern deep learning approaches. Instead of altering the fundamental unscented Kalman filter equations, a dedicated deep neural network is developed to predict the process and measurement noise uncertainty directly from raw inertial and GNSS measurements. We present a sim2real approach, with training performed only on simulative data. In this manner, we offer perfect ground truth data and relieves the burden of extensive data recordings. To evaluate our proposed approach and examine its generalization capabilities, we employed a 160-minutes test set from three datasets each with different types of vehicles (off-road vehicle, passenger car, and mobile robot), inertial sensors, road surface, and environmental conditions. We demonstrate across the three datasets a position improvement of $12.7\%$ compared to the adaptive model-based approach. Thus, offering a scalable and a more robust solution for unmanned ground vehicles navigation tasks.
>
---
#### [replaced 007] Memory-Augmented Potential Field Theory: A Framework for Adaptive Control in Non-Convex Domains
- **分类: cs.RO; math.DS**

- **简介: 该论文属于控制理论任务，旨在解决非凸环境中路径规划的问题。通过引入记忆增强的势场理论，提升控制器的自适应能力，改善优化效果。**

- **链接: [https://arxiv.org/pdf/2509.19672](https://arxiv.org/pdf/2509.19672)**

> **作者:** Dongzhe Zheng; Wenjie Mei
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Stochastic optimal control methods often struggle in complex non-convex landscapes, frequently becoming trapped in local optima due to their inability to learn from historical trajectory data. This paper introduces Memory-Augmented Potential Field Theory, a unified mathematical framework that integrates historical experience into stochastic optimal control. Our approach dynamically constructs memory-based potential fields that identify and encode key topological features of the state space, enabling controllers to automatically learn from past experiences and adapt their optimization strategy. We provide a theoretical analysis showing that memory-augmented potential fields possess non-convex escape properties, asymptotic convergence characteristics, and computational efficiency. We implement this theoretical framework in a Memory-Augmented Model Predictive Path Integral (MPPI) controller that demonstrates significantly improved performance in challenging non-convex environments. The framework represents a generalizable approach to experience-based learning within control systems (especially robotic dynamics), enhancing their ability to navigate complex state spaces without requiring specialized domain knowledge or extensive offline training.
>
---
#### [replaced 008] ACG: Action Coherence Guidance for Flow-based Vision-Language-Action models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决动作一致性问题。针对模仿学习中人类示范的噪声导致的轨迹漂移，提出ACG算法提升动作一致性与成功率。**

- **链接: [https://arxiv.org/pdf/2510.22201](https://arxiv.org/pdf/2510.22201)**

> **作者:** Minho Park; Kinam Kim; Junha Hyung; Hyojin Jang; Hoiyeong Jin; Jooyeol Yun; Hojoon Lee; Jaegul Choo
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Diffusion and flow matching models have emerged as powerful robot policies, enabling Vision-Language-Action (VLA) models to generalize across diverse scenes and instructions. Yet, when trained via imitation learning, their high generative capacity makes them sensitive to noise in human demonstrations: jerks, pauses, and jitter which reduce action coherence. Reduced action coherence causes instability and trajectory drift during deployment, failures that are catastrophic in fine-grained manipulation where precision is crucial. In this paper, we present Action Coherence Guidance (ACG) for VLA models, a training-free test-time guidance algorithm that improves action coherence and thereby yields performance gains. Evaluated on RoboCasa, DexMimicGen, and real-world SO-101 tasks, ACG consistently improves action coherence and boosts success rates across diverse manipulation tasks. Code and project page are available at this https URL and this https URL , respectively.
>
---
#### [replaced 009] Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Unified Diffusion VLA模型，解决多模态理解与行动生成问题。通过联合扩散过程实现视觉、语言和动作的同步生成与优化，提升任务协同效率。**

- **链接: [https://arxiv.org/pdf/2511.01718](https://arxiv.org/pdf/2511.01718)**

> **作者:** Jiayi Chen; Wenxuan Song; Pengxiang Ding; Ziyang Zhou; Han Zhao; Feilong Tang; Donglin Wang; Haoang Li
>
> **摘要:** Vision-language-action (VLA) models aim to understand natural language instructions and visual observations and to execute corresponding actions as an embodied agent. Recent work integrates future images into the understanding-acting loop, yielding unified VLAs that jointly understand, generate, and act -- reading text and images and producing future images and actions. However, these models either rely on external experts for modality unification or treat image generation and action prediction as separate processes, limiting the benefits of direct synergy between these tasks. Our core philosophy is to optimize generation and action jointly through a synchronous denoising process, where the iterative refinement enables actions to evolve from initialization, under constant and sufficient visual guidance. We ground this philosophy in our proposed Unified Diffusion VLA and Joint Discrete Denoising Diffusion Process (JD3P), which is a joint diffusion process that integrates multiple modalities into a single denoising trajectory to serve as the key mechanism enabling understanding, generation, and acting to be intrinsically synergistic. Our model and theory are built on a unified tokenized space of all modalities and a hybrid attention mechanism. We further propose a two-stage training pipeline and several inference-time techniques that optimize performance and efficiency. Our approach achieves state-of-the-art performance on benchmarks such as CALVIN, LIBERO, and SimplerEnv with 4$\times$ faster inference than autoregressive methods, and we demonstrate its effectiveness through in-depth analysis and real-world evaluations. Our project page is available at this https URL.
>
---
#### [replaced 010] HortiMulti: A Multi-Sensor Dataset for Localisation and Mapping in Horticultural Polytunnels
- **分类: cs.RO**

- **简介: 该论文提出HortiMulti数据集，用于解决农业机器人在温室环境中的定位与建图问题，涵盖多传感器数据和真实场景挑战。**

- **链接: [https://arxiv.org/pdf/2603.20150](https://arxiv.org/pdf/2603.20150)**

> **作者:** Shuoyuan Xu; Zhipeng Zhong; Tiago Barros; Matthew Coombes; Cristiano Premebida; Hao Wu; Cunjia Liu
>
> **摘要:** Agricultural robotics is gaining increasing relevance in both research and real-world deployment. As these systems are expected to operate autonomously in more complex tasks, the availability of representative real-world datasets becomes essential. While domains such as urban and forestry robotics benefit from large and established benchmarks, horticultural environments remain comparatively under-explored despite the economic significance of this sector. To address this gap, we present HortiMulti, a multimodal, cross-season dataset collected in commercial strawberry and raspberry polytunnels across an entire growing season, capturing substantial appearance variation, dynamic foliage, specular reflections from plastic covers, severe perceptual aliasing, and GNSS-unreliable conditions, all of which directly degrade existing localisation and perception algorithms. The sensor suite includes two 3D LiDARs, four RGB cameras, an IMU, GNSS, and wheel odometry. Ground truth trajectories are derived from a combination of Total Station surveying, AprilTag fiducial markers, and LiDAR-inertial odometry, spanning dense, sparse, and marker-free coverage to support evaluation under both controlled and realistic conditions. We release time-synchronised raw measurements, calibration files, reference trajectories, and baseline benchmarks for visual, LiDAR, and multi-sensor SLAM, with results confirming that current state-of-the-art methods remain inadequate for reliable polytunnel deployment, establishing HortiMulti as a one-stop resource for developing and testing robotic perception systems in horticulture environments.
>
---
#### [replaced 011] Onboard MuJoCo-based Model Predictive Control for Shipboard Crane with Double-Pendulum Sway Suppression
- **分类: cs.RO**

- **简介: 该论文属于船舶起重机控制任务，旨在解决双摆晃动问题。通过基于MuJoCo的模型预测控制，实现实时有效抑制晃动，优于传统方法。**

- **链接: [https://arxiv.org/pdf/2603.16407](https://arxiv.org/pdf/2603.16407)**

> **作者:** Oscar Pang; Lisa Coiffard; Paul Templier; Luke Beddow; Kamil Dreczkowski; Antoine Cully
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Transferring heavy payloads in maritime settings relies on efficient crane operation, limited by hazardous double-pendulum payload sway. This sway motion is further exacerbated in offshore environments by external perturbations from wind and ocean waves. Manual suppression of these oscillations on an underactuated crane system by human operators is challenging. Existing control methods struggle in such settings, often relying on simplified analytical models, while deep reinforcement learning (RL) approaches tend to generalise poorly to unseen conditions. Deploying a predictive controller onto compute-constrained, highly non-linear physical systems without relying on extensive offline training or complex analytical models remains a significant challenge. Here we show a complete real-time control pipeline centered on the MuJoCo MPC framework that leverages a cross-entropy method planner to evaluate candidate action sequences directly within a physics simulator. By using simulated rollouts, this sampling-based approach successfully reconciles the conflicting objectives of dynamic target tracking and sway damping without relying on complex analytical models. We demonstrate that the controller can run effectively on a resource-constrained embedded hardware, while outperforming traditional PID and RL baselines in counteracting external base perturbations. Furthermore, our system demonstrates robustness even when subjected to unmodeled physical discrepancies like the introduction of a second payload.
>
---
#### [replaced 012] NaviMaster: Learning a Unified Policy for GUI and Embodied Navigation Tasks
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出NaviMaster，统一处理GUI导航和具身导航任务。解决两者独立发展、数据与训练方式不一致的问题，通过统一框架提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.02046](https://arxiv.org/pdf/2508.02046)**

> **作者:** Zhihao Luo; Wentao Yan; Jingyu Gong; Min Wang; Zhizhong Zhang; Xuhong Wang; Yuan Xie; Xin Tan
>
> **备注:** 20 pages, 11 figures
>
> **摘要:** Recent advances in Graphical User Interface (GUI) and embodied navigation have driven progress, yet these domains have largely evolved in isolation, with disparate datasets and training paradigms. In this paper, we observe that both tasks can be formulated as Markov Decision Processes (MDP), suggesting a foundational principle for their unification. Hence, we present NaviMaster, the first unified agent capable of unifying GUI navigation and embodied navigation within a single framework. Specifically, NaviMaster (i) proposes a visual-target trajectory collection pipeline that generates trajectories for both GUI and embodied tasks using a single formulation. (ii) employs a unified reinforcement learning framework on the mix data to improve generalization. (iii) designs a novel distance-aware reward to ensure efficient learning from the trajectories. Through extensive experiments on out-of-domain benchmarks, NaviMaster is shown to outperform state-of-the-art agents in GUI navigation, spatial affordance prediction, and embodied navigation. Ablation studies further demonstrate the efficacy of our unified training strategy, data mixing strategy, and reward design. Our codes, data, and checkpoints are available at this https URL .
>
---
#### [replaced 013] Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决仿真到现实的迁移问题。通过在训练中注入状态相关关节扭矩扰动，提升政策对现实环境的适应能力。**

- **链接: [https://arxiv.org/pdf/2504.06585](https://arxiv.org/pdf/2504.06585)**

> **作者:** Junhyeok Rui Cha; Woohyun Cha; Jaeyong Shin; Donghyeon Kim; Jaeheung Park
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper proposes a novel alternative to existing sim-to-real methods for training control policies with simulated experiences. Prior sim-to-real methods for legged robots mostly rely on the domain randomization approach, where a fixed finite set of simulation parameters is randomized during training. Instead, our method adds state-dependent perturbations to the input joint torque used for forward simulation during the training phase. These state-dependent perturbations are designed to simulate a broader range of reality gaps than those captured by randomizing a fixed set of simulation parameters. Experimental results show that our method enables humanoid locomotion policies that achieve greater robustness against complex reality gaps unseen in the training domain.
>
---
#### [replaced 014] Dynamic Neural Potential Field: Online Trajectory Optimization in the Presence of Moving Obstacles
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决动态环境中安全轨迹优化问题。提出NPField-GPT框架，结合Transformer与MPC，实现实时、约束感知的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2410.06819](https://arxiv.org/pdf/2410.06819)**

> **作者:** Aleksei Staroverov; Muhammad Alhaddad; Aditya Narendra; Konstantin Mironov; Aleksandr Panov
>
> **摘要:** Generalist robot policies must operate safely and reliably in everyday human environments such as homes, offices, and warehouses, where people and objects move unpredictably. We present Dynamic Neural Potential Field (NPField-GPT), a learning-enhanced model predictive control (MPC) framework that couples classical optimization with a Transformer-based predictor of footprint-aware repulsive potentials. Given an occupancy sub-map, robot footprint, and optional dynamic-obstacle cues, our NPField-GPT model forecasts a horizon of differentiable potentials that are injected into a sequential quadratic MPC program via L4CasADi, yielding real-time, constraint-aware trajectory optimization. We additionally study two baselines: NPField-StaticMLP, where a dynamic scene is treated as a sequence of static maps; and NPField-DynamicMLP, which predicts the future potential sequence in parallel with an MLP. In dynamic indoor scenarios from BenchMR and on a Husky UGV in office corridors, NPField-GPT produces more efficient and safer trajectories under motion changes, while StaticMLP/DynamicMLP offer lower latency. We also compare with the CIAO* and MPPI baselines. Across methods, the Transformer+MPC synergy preserves the transparency and stability of model-based planning while learning only the part that benefits from data: spatiotemporal collision risk. Code and trained models are available at this https URL
>
---
#### [replaced 015] KINESIS: Motion Imitation for Human Musculoskeletal Locomotion
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; q-bio.NC**

- **简介: 该论文提出KINESIS，一个无需模型的运动模仿框架，解决人体肌肉骨骼运动控制问题，通过学习人类运动数据生成生理合理的肌肉活动模式。**

- **链接: [https://arxiv.org/pdf/2503.14637](https://arxiv.org/pdf/2503.14637)**

> **作者:** Merkourios Simos; Alberto Silvio Chiappa; Alexander Mathis
>
> **备注:** Accepted to ICRA. Here we include an appendix
>
> **摘要:** How do humans move? Advances in reinforcement learning (RL) have produced impressive results in capturing human motion using physics-based humanoid control. However, torque-controlled humanoids fail to model key aspects of human motor control such as biomechanical joint constraints & non-linear and overactuated musculotendon control. We present KINESIS, a model-free motion imitation framework that tackles these challenges. KINESIS is trained on 1.8 hours of locomotion data and achieves strong motion imitation performance on unseen trajectories. Through a negative mining approach, KINESIS learns robust locomotion priors that we leverage to deploy the policy on several downstream tasks such as text-to-control, target point reaching, and football penalty kicks. Importantly, KINESIS learns to generate muscle activity patterns that correlate well with human EMG activity. We show that these results scale seamlessly across biomechanical model complexity, demonstrating control of up to 290 muscles. Overall, the physiological plausibility makes KINESIS a promising model for tackling challenging problems in human motor control. Code, videos and benchmarks are available at this https URL.
>
---
#### [replaced 016] Rotor-Failure-Aware Quadrotors Flight in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文属于无人机自主飞行任务，解决旋翼故障下的飞行控制问题。提出一种融合故障检测与控制的系统，实现复杂环境中故障无人机的自主导航。**

- **链接: [https://arxiv.org/pdf/2510.11306](https://arxiv.org/pdf/2510.11306)**

> **作者:** Xiaobin Zhou; Miao Wang; Chengao Li; Can Cui; Ruibin Zhang; Yongchao Wang; Chao Xu; Fei Gao
>
> **摘要:** Rotor failures in quadrotors may result in high-speed rotation and vibration due to rotor imbalance, which introduces significant challenges for autonomous flight in unknown environments. The mainstream approaches against rotor failures rely on fault-tolerant control (FTC) and predefined trajectory tracking. To the best of our knowledge, online failure detection and diagnosis (FDD), trajectory planning, and FTC of the post-failure quadrotors in unknown and complex environments have not yet been achieved. This paper presents a rotor-failure-aware quadrotor navigation system designed to mitigate the impacts of rotor imbalance. First, a composite FDD-based nonlinear model predictive controller (NMPC), incorporating motor dynamics, is designed to ensure fast failure detection and flight stability. Second, a rotor-failure-aware planner is designed to leverage FDD results and spatial-temporal joint optimization, while a LiDAR-based quadrotor platform with four anti-torque plates is designed to enable reliable perception under high-speed rotation. Lastly, extensive benchmarks against state-of-the-art methods highlight the superior performance of the proposed approach in addressing rotor failures, including propeller unloading and motor stoppage. The experimental results demonstrate, for the first time, that our approach enables autonomous quadrotor flight with rotor failures in challenging environments, including cluttered rooms and unknown forests.
>
---
#### [replaced 017] Symmetry-Guided Memory Augmentation for Efficient Locomotion Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决腿部机器人运动学习中训练效率低的问题。通过结合对称性和记忆增强，生成有效训练经验，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2502.01521](https://arxiv.org/pdf/2502.01521)**

> **作者:** Kaixi Bao; Chenhao Li; Yarden As; Andreas Krause; Marco Hutter
>
> **摘要:** Training reinforcement learning (RL) policies for legged locomotion often requires extensive environment interactions, which are costly and time-consuming. We propose Symmetry-Guided Memory Augmentation (SGMA), a framework that improves training efficiency by combining structured experience augmentation with memory-based context inference. Our method leverages robot and task symmetries to generate additional, physically consistent training experiences without requiring extra interactions. To avoid the pitfalls of naive augmentation, we extend these transformations to the policy's memory states, enabling the agent to retain task-relevant context and adapt its behavior accordingly. We evaluate the approach on quadruped and humanoid robots in simulation, as well as on a real quadruped platform. Across diverse locomotion tasks involving joint failures and payload variations, our method achieves efficient policy training while maintaining robust performance, demonstrating a practical route toward data-efficient RL for legged robots.
>
---
#### [replaced 018] HiSync: Spatio-Temporally Aligning Hand Motion from Wearable IMU and On-Robot Camera for Command Source Identification in Long-Range HRI
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出HiSync，解决长距离人机交互中的命令源识别问题。通过融合惯性传感器与摄像头数据，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.11809](https://arxiv.org/pdf/2603.11809)**

> **作者:** Chengwen Zhang; Chun Yu; Borong Zhuang; Haopeng Jin; Qingyang Wan; Zhuojun Li; Zhe He; Zhoutong Ye; Yu Mei; Chang Liu; Weinan Shi; Yuanchun Shi
>
> **摘要:** Long-range Human-Robot Interaction (HRI) remains underexplored. Within it, Command Source Identification (CSI) - determining who issued a command - is especially challenging due to multi-user and distance-induced sensor ambiguity. We introduce HiSync, an optical-inertial fusion framework that treats hand motion as binding cues by aligning robot-mounted camera optical flow with hand-worn IMU signals. We first elicit a user-defined (N=12) gesture set and collect a multimodal command gesture dataset (N=38) in long-range multi-user HRI scenarios. Next, HiSync extracts frequency-domain hand motion features from both camera and IMU data, and a learned CSINet denoises IMU readings, temporally aligns modalities, and performs distance-aware multi-window fusion to compute cross-modal similarity of subtle, natural gestures, enabling robust CSI. In three-person scenes up to 34m, HiSync achieves 92.32% CSI accuracy, outperforming the prior SOTA by 48.44%. HiSync is also validated on real-robot deployment. By making CSI reliable and natural, HiSync provides a practical primitive and design guidance for public-space HRI. this https URL
>
---
#### [replaced 019] TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉和语言依赖导致的操控不足问题。通过引入触觉信息，提出TacVLA模型，提升精细操作和接触场景下的性能。**

- **链接: [https://arxiv.org/pdf/2603.12665](https://arxiv.org/pdf/2603.12665)**

> **作者:** Kaidi Zhang; Heng Zhang; Zhengtong Xu; Zhiyuan Zhang; Md Rakibul Islam Prince; Xiang Li; Xiaojing Han; Yuhao Zhou; Arash Ajoudani; Yu She
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated significant advantages in robotic manipulation. However, their reliance on vision and language often leads to suboptimal performance in tasks involving visual occlusion, fine-grained manipulation, and physical contact. To address these challenges, we propose TacVLA, a fine-tuned VLA model by incorporating tactile modalities into the transformer-based policy to enhance fine-grained manipulation capabilities. Specifically, we introduce a contact-aware gating mechanism that selectively activates tactile tokens only when contact is detected, enabling adaptive multimodal fusion while avoiding irrelevant tactile interference. The fused visual, language, and tactile tokens are jointly processed within the transformer architecture to strengthen cross-modal grounding during contact-rich interaction. Extensive experiments on constraint-locked disassembly, in-box picking and robustness evaluations demonstrate that our model outperforms baselines, improving the performance by averaging 20% success rate in disassembly, 60% in in-box picking and 2.1x improvement in scenarios with visual occlusion. Videos are available at this https URL and code will be released.
>
---
#### [replaced 020] Sim-to-Real of Humanoid Locomotion Policies via Joint Torque Space Perturbation Injection
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决仿真到现实的迁移问题。通过注入状态依赖的关节扭矩扰动，提升拟人机器人运动策略的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.21853](https://arxiv.org/pdf/2603.21853)**

> **作者:** Junhyeok Rui Cha; Woohyun Cha; Jaeyong Shin; Donghyeon Kim; Jaeheung Park
>
> **备注:** Duplication, resubmission of our previous paper arXiv:2504.06585
>
> **摘要:** This paper proposes a novel alternative to existing sim-to-real methods for training control policies with simulated experiences. Unlike prior methods that typically rely on domain randomization over a fixed finite set of parameters, the proposed approach injects state-dependent perturbations into the input joint torque during forward simulation. These perturbations are designed to simulate a broader spectrum of reality gaps than standard parameter randomization without requiring additional training. By using neural networks as flexible perturbation generators, the proposed method can represent complex, state-dependent uncertainties, such as nonlinear actuator dynamics and contact compliance, that parametric randomization cannot capture. Experimental results demonstrate that the proposed approach enables humanoid locomotion policies to achieve superior robustness against complex, unseen reality gaps in both simulation and real-world deployment.
>
---
#### [replaced 021] Reward Evolution with Graph-of-Thoughts: A Bi-Level Language Model Framework for Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于强化学习领域，旨在解决奖励函数设计困难的问题。通过结合大语言模型与视觉语言模型，利用图思维结构进行自动奖励演化，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2509.16136](https://arxiv.org/pdf/2509.16136)**

> **作者:** Changwei Yao; Xinzi Liu; Chen Li; Marios Savvides
>
> **摘要:** Designing effective reward functions remains a major challenge in reinforcement learning (RL), often requiring considerable human expertise and iterative refinement. Recent advances leverage Large Language Models (LLMs) for automated reward design, but these approaches are limited by hallucinations, reliance on human feedback, and challenges with handling complex, multi-step tasks. In this work, we introduce Reward Evolution with Graph-of-Thoughts (RE-GoT), a novel bi-level framework that enhances LLMs with structured graph-based reasoning and integrates Visual Language Models (VLMs) for automated rollout evaluation. RE-GoT first decomposes tasks into text-attributed graphs, enabling comprehensive analysis and reward function generation, and then iteratively refines rewards using visual feedback from VLMs without human intervention. Extensive experiments on 10 RoboGen and 4 ManiSkill2 tasks demonstrate that RE-GoT consistently outperforms existing LLM-based baselines. On RoboGen, our method improves average task success rates by 32.25%, with notable gains on complex multi-step tasks. On ManiSkill2, RE-GoT achieves an average success rate of 93.73% across four diverse manipulation tasks, significantly surpassing prior LLM-based approaches and even exceeding expert-designed rewards. Our results indicate that combining LLMs and VLMs with graph-of-thoughts reasoning provides a scalable and effective solution for autonomous reward evolution in RL.
>
---
#### [replaced 022] DIDLM: A SLAM Dataset for Difficult Scenarios Featuring Infrared, Depth Cameras, LIDAR, 4D Radar, and Others under Adverse Weather, Low Light Conditions, and Rough Roads
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于SLAM任务，旨在解决恶劣天气、低光和复杂地形下的导航难题。构建了包含多种传感器数据的综合数据集，以支持更鲁棒的SLAM研究。**

- **链接: [https://arxiv.org/pdf/2404.09622](https://arxiv.org/pdf/2404.09622)**

> **作者:** Weisheng Gong; Chen He; Kaijie Su; Qingyong Li; Tong Wu; Z. Jane Wang
>
> **摘要:** Adverse weather conditions, low-light environments, and bumpy road surfaces pose significant challenges to SLAM in robotic navigation and autonomous driving. Existing datasets in this field predominantly rely on single sensors or combinations of LiDAR, cameras, and IMUs. However, 4D millimeter-wave radar demonstrates robustness in adverse weather, infrared cameras excel in capturing details under low-light conditions, and depth images provide richer spatial information. Multi-sensor fusion methods also show potential for better adaptation to bumpy roads. Despite some SLAM studies incorporating these sensors and conditions, there remains a lack of comprehensive datasets addressing low-light environments and bumpy road conditions, or featuring a sufficiently diverse range of sensor data. In this study, we introduce a multi-sensor dataset covering challenging scenarios such as snowy weather, rainy weather, nighttime conditions, speed bumps, and rough terrains. The dataset includes rarely utilized sensors for extreme conditions, such as 4D millimeter-wave radar, infrared cameras, and depth cameras, alongside 3D LiDAR, RGB cameras, GPS, and IMU. It supports both autonomous driving and ground robot applications and provides reliable GPS/INS ground truth data, covering structured and semi-structured terrains. We evaluated various SLAM algorithms using this dataset, including RGB images, infrared images, depth images, LiDAR, and 4D millimeter-wave radar. The dataset spans a total of 18.5 km, 69 minutes, and approximately 660 GB, offering a valuable resource for advancing SLAM research under complex and extreme conditions. Our dataset is available at this https URL.
>
---
#### [replaced 023] Instrument-Splatting++: Towards Controllable Surgical Instrument Digital Twin Using Gaussian Splatting
- **分类: cs.RO**

- **简介: 该论文提出Instrument-Splatting++，用于构建可控制的手术器械数字孪生，解决机器人手术中真实到模拟的重建问题。通过3DGS和语义跟踪实现高精度姿态估计与纹理学习。**

- **链接: [https://arxiv.org/pdf/2603.22792](https://arxiv.org/pdf/2603.22792)**

> **作者:** Shuojue Yang; Zijian Wu; Chengjiaao Liao; Qian Li; Daiyun Shen; Chang Han Low; Septimiu E. Salcudean; Yueming Jin
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** High-quality and controllable digital twins of surgical instruments are critical for Real2Sim in robot-assisted surgery, as they enable realistic simulation, synthetic data generation, and perception learning under novel poses. We present Instrument-Splatting++, a monocular 3D Gaussian Splatting (3DGS) framework that reconstructs surgical instruments as a fully controllable Gaussian asset with high fidelity. Our pipeline starts with part-wise geometry pretraining that injects CAD priors into Gaussian primitives and equips the representation with part-aware semantic rendering. Built on the pretrained model, we propose a semantics-aware pose estimation and tracking (SAPET) method to recover per-frame 6-DoF pose and joint angles from unposed endoscopic videos, where a gripper-tip network trained purely from synthetic semantics provides robust supervision and a loose regularization suppresses singular articulations. Finally, we introduce Robust Texture Learning (RTL), which alternates pose refinement and robust appearance optimization, mitigating pose noise during texture learning. The proposed framework can perform pose estimation and learn realistic texture from unposed videos. We validate our method on sequences extracted from EndoVis17/18, SAR-RARP, and an in-house dataset, showing superior photometric quality and improved geometric accuracy over state-of-the-art baselines. We further demonstrate a downstream keypoint detection task where unseen-pose data augmentation from our controllable instrument Gaussian improves performance.
>
---
#### [replaced 024] ManiDreams: An Open-Source Library for Robust Object Manipulation via Uncertainty-aware Task-specific Intuitive Physics
- **分类: cs.RO**

- **简介: 该论文提出ManiDreams，一个用于不确定环境下物体操作的框架，解决机器人操作中的不确定性问题。通过集成不确定性处理，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.18336](https://arxiv.org/pdf/2603.18336)**

> **作者:** Gaotian Wang; Kejia Ren; Andrew S. Morgan; Kaiyu Hang
>
> **备注:** 9 pages, 10 figures. Project page at this https URL
>
> **摘要:** Dynamics models, whether simulators or learned world models, have long been central to robotic manipulation, but most focus on minimizing prediction error rather than confronting a more fundamental challenge: real-world manipulation is inherently uncertain. We argue that robust manipulation under uncertainty is fundamentally an integration problem: uncertainties must be represented, propagated, and constrained within the planning loop, not merely suppressed during training. We present and open-source ManiDreams, a modular framework for uncertainty-aware manipulation planning over intuitive physics models. It realizes this integration through composable abstractions for distributional state representation, backend-agnostic dynamics prediction, and declarative constraint specification for action optimization. The framework explicitly addresses three sources of uncertainty: perceptual, parametric, and structural. It wraps any base policy with a sample-predict-constrain loop that evaluates candidate actions against distributional outcomes, adding robustness without retraining. Experiments on ManiSkill tasks show that ManiDreams maintains robust performance under various perturbations where the RL baseline degrades significantly. Runnable examples on pushing, picking, catching, and real-world deployment demonstrate flexibility across different policies, optimizers, physics backends, and executors. The framework is publicly available at this https URL
>
---
#### [replaced 025] Co-Designing a Peer Social Robot for Young Newcomers' Language and Cultural Learning
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决社区语言学习中师资不足的问题。通过设计一款协作型社交机器人，辅助教师进行个性化语言与文化教学。**

- **链接: [https://arxiv.org/pdf/2603.18804](https://arxiv.org/pdf/2603.18804)**

> **作者:** Neil Fernandes; Cheng Tang; Tehniyat Shahbaz; Alex Hauschildt; Emily Davies-Robinson; Yue Hu; Kerstin Dautenhahn
>
> **摘要:** Community literacy programs supporting young newcomer children in Canada face limited staffing and scarce one-to-one time, which constrains personalized English and cultural learning support. This paper reports on a co-design study with United for Literacy tutors that informed Maple, a table-top, peer-like Socially Assistive Robot (SAR) designed as a practice partner within tutor-mediated sessions. From shadowing and co-design interviews, we derived newcomer-specific requirements and added them in an integrated prototype that uses short story-based activities, multi-modal scaffolding and embedded quizzes that support attention while producing tutor-actionable formative signals. We contribute system design implications for tutor-in-the-loop SARs supporting language socialization in community settings and outline directions for child-centered evaluation in authentic programs.
>
---
#### [replaced 026] Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人任务，旨在提升视觉-语言-动作模型的鲁棒性。通过生成多样化指令，识别模型漏洞并优化其表现。**

- **链接: [https://arxiv.org/pdf/2603.12510](https://arxiv.org/pdf/2603.12510)**

> **作者:** Siddharth Srikanth; Freddie Liang; Ya-Chuan Hsu; Varun Bhatt; Shihan Zhao; Henry Chen; Bryon Tjanaka; Minjune Hwang; Akanksha Saran; Daniel Seita; Aaquib Tabrez; Stefanos Nikolaidis
>
> **摘要:** Vision-Language-Action (VLA) models have significant potential to enable general-purpose robotic systems for a range of vision-language tasks. However, the performance of VLA-based robots is highly sensitive to the precise wording of language instructions, and it remains difficult to predict when such robots will fail. To improve the robustness of VLAs to different wordings, we present Q-DIG (Quality Diversity for Diverse Instruction Generation), which performs red-teaming by scalably identifying diverse natural language task descriptions that induce failures while remaining task-relevant. Q-DIG integrates Quality Diversity (QD) techniques with Vision-Language Models (VLMs) to generate a broad spectrum of adversarial instructions that expose meaningful vulnerabilities in VLA behavior. Our results across multiple simulation benchmarks show that Q-DIG finds more diverse and meaningful failure modes compared to baseline methods, and that fine-tuning VLAs on the generated instructions improves task success rates. Furthermore, results from a user study highlight that Q-DIG generates prompts judged to be more natural and human-like than those from baselines. Finally, real-world evaluations of Q-DIG prompts show results consistent with simulation, and fine-tuning VLAs on the generated prompts further success rates on unseen instructions. Together, these findings suggest that Q-DIG is a promising approach for identifying vulnerabilities and improving the robustness of VLA-based robots. Our anonymous project website is at this http URL.
>
---
#### [replaced 027] Unicorn: A Universal and Collaborative Reinforcement Learning Approach Towards Generalizable Network-Wide Traffic Signal Control
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于交通信号控制任务，旨在解决复杂交通网络中的通用且高效控制问题。提出Unicorn框架，通过统一表示和协作强化学习实现跨场景的优化控制。**

- **链接: [https://arxiv.org/pdf/2503.11488](https://arxiv.org/pdf/2503.11488)**

> **作者:** Yifeng Zhang; Yilin Liu; Ping Gong; Peizhuo Li; Mingfeng Fan; Guillaume Sartoretti
>
> **备注:** \c{opyright} 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Adaptive traffic signal control (ATSC) is crucial in reducing congestion, maximizing throughput, and improving mobility in rapidly growing urban areas. Recent advancements in parameter-sharing multi-agent reinforcement learning (MARL) have greatly enhanced the scalable and adaptive optimization of complex, dynamic flows in large-scale homogeneous networks. However, the inherent heterogeneity of real-world traffic networks, with their varied intersection topologies and interaction dynamics, poses substantial challenges to achieving scalable and effective ATSC across different traffic scenarios. To address these challenges, we present Unicorn, a universal and collaborative MARL framework designed for efficient and adaptable network-wide ATSC. Specifically, we first propose a unified approach to map the states and actions of intersections with varying topologies into a common structure based on traffic movements. Next, we design a Universal Traffic Representation (UTR) module with a decoder-only network for general feature extraction, enhancing the model's adaptability to diverse traffic scenarios. Additionally, we incorporate an Intersection Specifics Representation (ISR) module, designed to identify key latent vectors that represent the unique intersection's topology and traffic dynamics through variational inference techniques. To further refine these latent representations, we employ a contrastive learning approach in a self-supervised manner, which enables better differentiation of intersection-specific features. Moreover, we integrate the state-action dependencies of neighboring agents into policy optimization, which effectively captures dynamic agent interactions and facilitates efficient regional collaboration. [...]. The code is available at this https URL
>
---
