# 机器人 cs.RO

- **最新发布 25 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] Design for One, Deploy for Many: Navigating Tree Mazes with Multiple Agents
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多智能体在树状迷宫中的协同遍历任务，针对通信受限与拥堵问题，提出一种分布式算法。通过领导切换机制，使多代理协作模拟单代理路径，实现高效遍历。实验验证了其在仿真与真实机器人平台上的有效性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.26900v1](http://arxiv.org/pdf/2510.26900v1)**

> **作者:** Jahir Argote-Gerald; Genki Miyauchi; Julian Rau; Paul Trodden; Roderich Gross
>
> **备注:** 7 pages, 7 figures, to be published in MRS 2025
>
> **摘要:** Maze-like environments, such as cave and pipe networks, pose unique challenges for multiple robots to coordinate, including communication constraints and congestion. To address these challenges, we propose a distributed multi-agent maze traversal algorithm for environments that can be represented by acyclic graphs. It uses a leader-switching mechanism where one agent, assuming a head role, employs any single-agent maze solver while the other agents each choose an agent to follow. The head role gets transferred to neighboring agents where necessary, ensuring it follows the same path as a single agent would. The multi-agent maze traversal algorithm is evaluated in simulations with groups of up to 300 agents, various maze sizes, and multiple single-agent maze solvers. It is compared against strategies that are na\"ive, or assume either global communication or full knowledge of the environment. The algorithm outperforms the na\"ive strategy in terms of makespan and sum-of-fuel. It is superior to the global-communication strategy in terms of makespan but is inferior to it in terms of sum-of-fuel. The findings suggest it is asymptotically equivalent to the full-knowledge strategy with respect to either metric. Moreover, real-world experiments with up to 20 Pi-puck robots confirm the feasibility of the approach.
>
---
#### [new 002] Learning Generalizable Visuomotor Policy through Dynamics-Alignment
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人视觉运动策略泛化能力差的问题，提出DAP方法，通过融合动作相关动力学预测与策略学习，实现策略与动力学模型的相互校正。有效提升真实场景下的泛化性能，尤其在分布外场景中表现稳健。**

- **链接: [http://arxiv.org/pdf/2510.27114v1](http://arxiv.org/pdf/2510.27114v1)**

> **作者:** Dohyeok Lee; Jung Min Lee; Munkyung Kim; Seokhun Ju; Jin Woo Koo; Kyungjae Lee; Dohyeong Kim; TaeHyun Cho; Jungwoo Lee
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Behavior cloning methods for robot learning suffer from poor generalization due to limited data support beyond expert demonstrations. Recent approaches leveraging video prediction models have shown promising results by learning rich spatiotemporal representations from large-scale datasets. However, these models learn action-agnostic dynamics that cannot distinguish between different control inputs, limiting their utility for precise manipulation tasks and requiring large pretraining datasets. We propose a Dynamics-Aligned Flow Matching Policy (DAP) that integrates dynamics prediction into policy learning. Our method introduces a novel architecture where policy and dynamics models provide mutual corrective feedback during action generation, enabling self-correction and improved generalization. Empirical validation demonstrates generalization performance superior to baseline methods on real-world robotic manipulation tasks, showing particular robustness in OOD scenarios including visual distractions and lighting variations.
>
---
#### [new 003] RepV: Safety-Separable Latent Spaces for Scalable Neurosymbolic Plan Verification
- **分类: cs.RO; cs.AI; cs.CL; cs.FL**

- **简介: 该论文提出RepV，一种用于可扩展神经符号计划验证的新方法。针对安全关键领域中规则合规性验证难题，通过学习安全与不安全计划线性可分的潜在空间，结合语言模型生成推理，实现对自然语言规则的高效、可解释验证，并提供概率保障以驱动规划器优化。**

- **链接: [http://arxiv.org/pdf/2510.26935v1](http://arxiv.org/pdf/2510.26935v1)**

> **作者:** Yunhao Yang; Neel P. Bhatt; Pranay Samineni; Rohan Siva; Zhanyang Wang; Ufuk Topcu
>
> **备注:** Code and data are available at: https://repv-project.github.io/
>
> **摘要:** As AI systems migrate to safety-critical domains, verifying that their actions comply with well-defined rules remains a challenge. Formal methods provide provable guarantees but demand hand-crafted temporal-logic specifications, offering limited expressiveness and accessibility. Deep learning approaches enable evaluation of plans against natural-language constraints, yet their opaque decision process invites misclassifications with potentially severe consequences. We introduce RepV, a neurosymbolic verifier that unifies both views by learning a latent space where safe and unsafe plans are linearly separable. Starting from a modest seed set of plans labeled by an off-the-shelf model checker, RepV trains a lightweight projector that embeds each plan, together with a language model-generated rationale, into a low-dimensional space; a frozen linear boundary then verifies compliance for unseen natural-language rules in a single forward pass. Beyond binary classification, RepV provides a probabilistic guarantee on the likelihood of correct verification based on its position in the latent space. This guarantee enables a guarantee-driven refinement of the planner, improving rule compliance without human annotations. Empirical evaluations show that RepV improves compliance prediction accuracy by up to 15% compared to baseline methods while adding fewer than 0.2M parameters. Furthermore, our refinement framework outperforms ordinary fine-tuning baselines across various planning domains. These results show that safety-separable latent spaces offer a scalable, plug-and-play primitive for reliable neurosymbolic plan verification. Code and data are available at: https://repv-project.github.io/.
>
---
#### [new 004] Learning Soft Robotic Dynamics with Active Exploration
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SoftAE框架，解决软体机器人高维非线性动力学建模难题。通过不确定性驱动的主动探索，实现无需任务监督的高效状态空间覆盖，构建通用、可复用的动力学模型，显著提升零样本控制性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.27428v1](http://arxiv.org/pdf/2510.27428v1)**

> **作者:** Hehui Zheng; Bhavya Sukhija; Chenhao Li; Klemens Iten; Andreas Krause; Robert K. Katzschmann
>
> **摘要:** Soft robots offer unmatched adaptability and safety in unstructured environments, yet their compliant, high-dimensional, and nonlinear dynamics make modeling for control notoriously difficult. Existing data-driven approaches often fail to generalize, constrained by narrowly focused task demonstrations or inefficient random exploration. We introduce SoftAE, an uncertainty-aware active exploration framework that autonomously learns task-agnostic and generalizable dynamics models of soft robotic systems. SoftAE employs probabilistic ensemble models to estimate epistemic uncertainty and actively guides exploration toward underrepresented regions of the state-action space, achieving efficient coverage of diverse behaviors without task-specific supervision. We evaluate SoftAE on three simulated soft robotic platforms -- a continuum arm, an articulated fish in fluid, and a musculoskeletal leg with hybrid actuation -- and on a pneumatically actuated continuum soft arm in the real world. Compared with random exploration and task-specific model-based reinforcement learning, SoftAE produces more accurate dynamics models, enables superior zero-shot control on unseen tasks, and maintains robustness under sensing noise, actuation delays, and nonlinear material effects. These results demonstrate that uncertainty-driven active exploration can yield scalable, reusable dynamics models across diverse soft robotic morphologies, representing a step toward more autonomous, adaptable, and data-efficient control in compliant robots.
>
---
#### [new 005] Vectorized Online POMDP Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对自主机器人在部分可观测环境下的规划问题，提出向量化在线POMDP规划器（VOPP）。通过将规划数据结构转为张量并实现全向量化计算，消除了传统方法中的依赖与同步瓶颈，显著提升并行效率，实验表明其性能优于现有最优并行求解器至少20倍。**

- **链接: [http://arxiv.org/pdf/2510.27191v1](http://arxiv.org/pdf/2510.27191v1)**

> **作者:** Marcus Hoerger; Muhammad Sudrajat; Hanna Kurniawati
>
> **备注:** 8 pages, 3 figures. Submitted to ICRA 2026
>
> **摘要:** Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization of today's hardware, but parallelizing POMDP solvers has been challenging. They rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can quickly offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation that analytically solves part of the optimization component, leaving only the estimation of expectations for numerical computation. VOPP represents all data structures related to planning as a collection of tensors and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel solver with no dependencies and synchronization bottlenecks between parallel computations. Experimental results indicate that VOPP is at least 20X more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver.
>
---
#### [new 006] A Multi-Modal Neuro-Symbolic Approach for Spatial Reasoning-Based Visual Grounding in Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对机器人视觉接地中的空间推理任务，解决现有模型依赖图像、缺乏显式逻辑推理的问题。提出多模态神经符号框架，融合全景图与3D点云，通过感知与符号推理模块构建结构化场景图，实现精准可解释的空间关系理解，在复杂环境中表现优异。**

- **链接: [http://arxiv.org/pdf/2510.27033v1](http://arxiv.org/pdf/2510.27033v1)**

> **作者:** Simindokht Jahangard; Mehrzad Mohammadi; Abhinav Dhall; Hamid Rezatofighi
>
> **摘要:** Visual reasoning, particularly spatial reasoning, is a challenging cognitive task that requires understanding object relationships and their interactions within complex environments, especially in robotics domain. Existing vision_language models (VLMs) excel at perception tasks but struggle with fine-grained spatial reasoning due to their implicit, correlation-driven reasoning and reliance solely on images. We propose a novel neuro_symbolic framework that integrates both panoramic-image and 3D point cloud information, combining neural perception with symbolic reasoning to explicitly model spatial and logical relationships. Our framework consists of a perception module for detecting entities and extracting attributes, and a reasoning module that constructs a structured scene graph to support precise, interpretable queries. Evaluated on the JRDB-Reasoning dataset, our approach demonstrates superior performance and reliability in crowded, human_built environments while maintaining a lightweight design suitable for robotics and embodied AI applications.
>
---
#### [new 007] EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出EBT-Policy，一种基于能量模型的机器人策略框架。针对扩散模型计算成本高、暴露偏差和推理不稳的问题，利用能量基变换器实现高效、鲁棒的物理推理。在仿真与真实任务中表现优异，仅需两步推理即收敛，具备零样本失败恢复等涌现能力。**

- **链接: [http://arxiv.org/pdf/2510.27545v1](http://arxiv.org/pdf/2510.27545v1)**

> **作者:** Travis Davies; Yiqi Huang; Alexi Gladstone; Yunxin Liu; Xiang Chen; Heng Ji; Huxian Liu; Luhui Hu
>
> **备注:** 9 pages, 6 figures, 4 tables
>
> **摘要:** Implicit policies parameterized by generative models, such as Diffusion Policy, have become the standard for policy learning and Vision-Language-Action (VLA) models in robotics. However, these approaches often suffer from high computational cost, exposure bias, and unstable inference dynamics, which lead to divergence under distribution shifts. Energy-Based Models (EBMs) address these issues by learning energy landscapes end-to-end and modeling equilibrium dynamics, offering improved robustness and reduced exposure bias. Yet, policies parameterized by EBMs have historically struggled to scale effectively. Recent work on Energy-Based Transformers (EBTs) demonstrates the scalability of EBMs to high-dimensional spaces, but their potential for solving core challenges in physically embodied models remains underexplored. We introduce a new energy-based architecture, EBT-Policy, that solves core issues in robotic and real-world settings. Across simulated and real-world tasks, EBT-Policy consistently outperforms diffusion-based policies, while requiring less training and inference computation. Remarkably, on some tasks it converges within just two inference steps, a 50x reduction compared to Diffusion Policy's 100. Moreover, EBT-Policy exhibits emergent capabilities not seen in prior models, such as zero-shot recovery from failed action sequences using only behavior cloning and without explicit retry training. By leveraging its scalar energy for uncertainty-aware inference and dynamic compute allocation, EBT-Policy offers a promising path toward robust, generalizable robot behavior under distribution shifts.
>
---
#### [new 008] Towards a Multi-Embodied Grasping Agent
- **分类: cs.RO; I.2.9**

- **简介: 该论文聚焦多具身抓取任务，旨在实现跨不同夹爪设计的通用抓取能力。针对现有方法依赖大量数据、隐式学习机械结构的问题，提出一种基于JAX的流模型架构，显式利用夹爪与场景几何信息，实现高效、等变的抓取合成，并支持批量处理，显著提升训练效率与推理速度。**

- **链接: [http://arxiv.org/pdf/2510.27420v1](http://arxiv.org/pdf/2510.27420v1)**

> **作者:** Roman Freiberg; Alexander Qualmann; Ngo Anh Vien; Gerhard Neumann
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Multi-embodiment grasping focuses on developing approaches that exhibit generalist behavior across diverse gripper designs. Existing methods often learn the kinematic structure of the robot implicitly and face challenges due to the difficulty of sourcing the required large-scale data. In this work, we present a data-efficient, flow-based, equivariant grasp synthesis architecture that can handle different gripper types with variable degrees of freedom and successfully exploit the underlying kinematic model, deducing all necessary information solely from the gripper and scene geometry. Unlike previous equivariant grasping methods, we translated all modules from the ground up to JAX and provide a model with batching capabilities over scenes, grippers, and grasps, resulting in smoother learning, improved performance and faster inference time. Our dataset encompasses grippers ranging from humanoid hands to parallel yaw grippers and includes 25,000 scenes and 20 million grasps.
>
---
#### [new 009] A Hermetic, Transparent Soft Growing Vine Robot System for Pipe Inspection
- **分类: cs.RO**

- **简介: 该论文针对老旧管道内部检测难题，提出一种密封透明的软体生长藤蔓机器人系统。通过封闭式设计保护内部元件并实现视觉感知，开发了自适应末端安装结构，并在实际污水管道中验证了其条件评估与建图能力，推动了软体机器人在工业管道检测中的应用。**

- **链接: [http://arxiv.org/pdf/2510.27010v1](http://arxiv.org/pdf/2510.27010v1)**

> **作者:** William E. Heap; Yimeng Qin; Kai Hammond; Anish Bayya; Haonon Kong; Allison M. Okamura
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Rehabilitation of aging pipes requires accurate condition assessment and mapping far into the pipe interiors. Soft growing vine robot systems are particularly promising for navigating confined, sinuous paths such as in pipes, but are currently limited by complex subsystems and a lack of validation in real-world industrial settings. In this paper, we introduce the concept and implementation of a hermetic and transparent vine robot system for visual condition assessment and mapping within non-branching pipes. This design encloses all mechanical and electrical components within the vine robot's soft, airtight, and transparent body, protecting them from environmental interference while enabling visual sensing. Because this approach requires an enclosed mechanism for transporting sensors, we developed, modeled, and tested a passively adapting enclosed tip mount. Finally, we validated the hermetic and transparent vine robot system concept through a real-world condition assessment and mapping task in a wastewater pipe. This work advances the use of soft-growing vine robots in pipe inspection by developing and demonstrating a robust, streamlined, field-validated system suitable for continued development and deployment.
>
---
#### [new 010] Confined Space Underwater Positioning Using Collaborative Robots
- **分类: cs.RO; /**

- **简介: 该论文针对水下受限空间定位难题，提出协作式水下定位系统CAP。通过水面母船与潜航器协同，融合传感器数据，在无GPS、无固定设施环境下实现高精度实时定位，实验验证平均误差70mm，解决了复杂环境中水下机器人可靠定位问题。**

- **链接: [http://arxiv.org/pdf/2510.27151v1](http://arxiv.org/pdf/2510.27151v1)**

> **作者:** Xueliang Cheng; Kanzhong Yao; Andrew West; Ognjen Marjanovic; Barry Lennox; Keir Groves
>
> **备注:** 31 pages including appendix, 24 figures
>
> **摘要:** Positioning of underwater robots in confined and cluttered spaces remains a key challenge for field operations. Existing systems are mostly designed for large, open-water environments and struggle in industrial settings due to poor coverage, reliance on external infrastructure, and the need for feature-rich surroundings. Multipath effects from continuous sound reflections further degrade signal quality, reducing accuracy and reliability. Accurate and easily deployable positioning is essential for repeatable autonomous missions; however, this requirement has created a technological bottleneck limiting underwater robotic deployment. This paper presents the Collaborative Aquatic Positioning (CAP) system, which integrates collaborative robotics and sensor fusion to overcome these limitations. Inspired by the "mother-ship" concept, the surface vehicle acts as a mobile leader to assist in positioning a submerged robot, enabling localization even in GPS-denied and highly constrained environments. The system is validated in a large test tank through repeatable autonomous missions using CAP's position estimates for real-time trajectory control. Experimental results demonstrate a mean Euclidean distance (MED) error of 70 mm, achieved in real time without requiring fixed infrastructure, extensive calibration, or environmental features. CAP leverages advances in mobile robot sensing and leader-follower control to deliver a step change in accurate, practical, and infrastructure-free underwater localization.
>
---
#### [new 011] Preliminary Prototyping of Avoidance Behaviors Triggered by a User's Physical Approach to a Robot
- **分类: cs.RO; cs.HC; I.2.9; I.3.6**

- **简介: 该论文研究人机交互中机器人对用户接近的回避行为。针对物理接近时的不适感，构建基于PAD模型主导性维度的内部状态，实现从耐受到避让的渐进式反应。通过臂式机器人验证了从距离感知到分级动作的完整响应流程。**

- **链接: [http://arxiv.org/pdf/2510.27436v1](http://arxiv.org/pdf/2510.27436v1)**

> **作者:** Tomoko Yonezawa; Hirotake Yamazoe; Atsuo Fujino; Daigo Suhara; Takaya Tamamoto; Yuto Nishiguchi
>
> **备注:** Workshop on Socially Aware and Cooperative Intelligent Systems in HAI 2025
>
> **摘要:** Human-robot interaction frequently involves physical proximity or contact. In human-human settings, people flexibly accept, reject, or tolerate such approaches depending on the relationship and context. We explore the design of a robot's rejective internal state and corresponding avoidance behaviors, such as withdrawing or pushing away, when a person approaches. We model the accumulation and decay of discomfort as a function of interpersonal distance, and implement tolerance (endurance) and limit-exceeding avoidance driven by the Dominance axis of the PAD affect model. The behaviors and their intensities are realized on an arm robot. Results illustrate a coherent pipeline from internal state parameters to graded endurance motions and, once a limit is crossed, to avoidance actions.
>
---
#### [new 012] NaviTrace: Evaluating Embodied Navigation of Vision-Language Models
- **分类: cs.RO**

- **简介: 该论文提出NaviTrace，一个用于评估视觉语言模型在机器人导航中表现的高质量VQA基准。针对现有评估受限于真实实验成本高、仿真简化及基准不足的问题，构建了包含1000场景与3000专家轨迹的数据集，并引入语义感知的轨迹评分指标，系统评测8个SOTA模型，揭示其空间定位与目标识别能力不足。**

- **链接: [http://arxiv.org/pdf/2510.26909v1](http://arxiv.org/pdf/2510.26909v1)**

> **作者:** Tim Windecker; Manthan Patel; Moritz Reuss; Richard Schwarzkopf; Cesar Cadena; Rudolf Lioutikov; Marco Hutter; Jonas Frey
>
> **备注:** 9 pages, 6 figures, under review at IEEE conference
>
> **摘要:** Vision-language models demonstrate unprecedented performance and generalization across a wide range of tasks and scenarios. Integrating these foundation models into robotic navigation systems opens pathways toward building general-purpose robots. Yet, evaluating these models' navigation capabilities remains constrained by costly real-world trials, overly simplified simulations, and limited benchmarks. We introduce NaviTrace, a high-quality Visual Question Answering benchmark where a model receives an instruction and embodiment type (human, legged robot, wheeled robot, bicycle) and must output a 2D navigation trace in image space. Across 1000 scenarios and more than 3000 expert traces, we systematically evaluate eight state-of-the-art VLMs using a newly introduced semantic-aware trace score. This metric combines Dynamic Time Warping distance, goal endpoint error, and embodiment-conditioned penalties derived from per-pixel semantics and correlates with human preferences. Our evaluation reveals consistent gap to human performance caused by poor spatial grounding and goal localization. NaviTrace establishes a scalable and reproducible benchmark for real-world robotic navigation. The benchmark and leaderboard can be found at https://leggedrobotics.github.io/navitrace_webpage/.
>
---
#### [new 013] Leveraging Foundation Models for Enhancing Robot Perception and Action
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究如何利用基础模型提升机器人在非结构化环境中的感知与操作能力，聚焦定位、交互与抓取任务，通过构建语义感知的智能框架，系统性地增强机器人理解与应对复杂场景的能力。**

- **链接: [http://arxiv.org/pdf/2510.26855v1](http://arxiv.org/pdf/2510.26855v1)**

> **作者:** Reihaneh Mirjalili
>
> **备注:** Doctoral thesis
>
> **摘要:** This thesis investigates how foundation models can be systematically leveraged to enhance robotic capabilities, enabling more effective localization, interaction, and manipulation in unstructured environments. The work is structured around four core lines of inquiry, each addressing a fundamental challenge in robotics while collectively contributing to a cohesive framework for semantics-aware robotic intelligence.
>
---
#### [new 014] A Modular and Scalable System Architecture for Heterogeneous UAV Swarms Using ROS 2 and PX4-Autopilot
- **分类: cs.RO**

- **简介: 该论文针对异构无人机集群的协同控制问题，提出基于ROS 2与PX4的模块化可扩展架构。通过独立节点实现硬件集成，抽象通信层支持多技术融合，支持编队飞行、目标跟踪与地面站管控，并在仿真与实测中验证了系统有效性。**

- **链接: [http://arxiv.org/pdf/2510.27327v1](http://arxiv.org/pdf/2510.27327v1)**

> **作者:** Robert Pommeranz; Kevin Tebbe; Ralf Heynicke; Gerd Scholl
>
> **摘要:** In this paper a modular and scalable architecture for heterogeneous swarm-based Counter Unmanned Aerial Systems (C-UASs) built on PX4-Autopilot and Robot Operating System 2 (ROS 2) framework is presented. The proposed architecture emphasizes seamless integration of hardware components by introducing independent ROS 2 nodes for each component of a Unmanned Aerial Vehicle (UAV). Communication between swarm participants is abstracted in software, allowing the use of various technologies without architectural changes. Key functionalities are supported, e.g. leader following and formation flight to maneuver the swarm. The system also allows computer vision algorithms to be integrated for the detection and tracking of UAVs. Additionally, a ground station control is integrated for the coordination of swarm operations. Swarm-based Unmanned Aerial System (UAS) architecture is verified within a Gazebo simulation environment but also in real-world demonstrations.
>
---
#### [new 015] Toward Accurate Long-Horizon Robotic Manipulation: Language-to-Action with Foundation Models via Scene Graphs
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文面向长时程机器人操作任务，提出基于场景图的通用框架，利用预训练基础模型实现无需领域训练的多模态感知与任务推理。通过动态维护场景图增强空间理解，实现精准的任务序列规划与执行。**

- **链接: [http://arxiv.org/pdf/2510.27558v1](http://arxiv.org/pdf/2510.27558v1)**

> **作者:** Sushil Samuel Dinesh; Shinkyu Park
>
> **摘要:** This paper presents a framework that leverages pre-trained foundation models for robotic manipulation without domain-specific training. The framework integrates off-the-shelf models, combining multimodal perception from foundation models with a general-purpose reasoning model capable of robust task sequencing. Scene graphs, dynamically maintained within the framework, provide spatial awareness and enable consistent reasoning about the environment. The framework is evaluated through a series of tabletop robotic manipulation experiments, and the results highlight its potential for building robotic manipulation systems directly on top of off-the-shelf foundation models.
>
---
#### [new 016] Whole-Body Proprioceptive Morphing: A Modular Soft Gripper for Robust Cross-Scale Grasping
- **分类: cs.RO**

- **简介: 该论文针对机器人跨尺度抓取能力不足的问题，提出一种基于全身本体感知变形的模块化软抓手。通过分布式自感应气动单元实现整体形态智能重构，融合多模态感知与控制，可灵活切换多种抓握模式，显著提升对不同尺寸、形状物体的适应性与操作多样性。**

- **链接: [http://arxiv.org/pdf/2510.27666v1](http://arxiv.org/pdf/2510.27666v1)**

> **作者:** Dong Heon Han; Xiaohao Xu; Yuxi Chen; Yusheng Zhou; Xinqi Zhang; Jiaqi Wang; Daniel Bruder; Xiaonan Huang
>
> **摘要:** Biological systems, such as the octopus, exhibit masterful cross-scale manipulation by adaptively reconfiguring their entire form, a capability that remains elusive in robotics. Conventional soft grippers, while compliant, are mostly constrained by a fixed global morphology, and prior shape-morphing efforts have been largely confined to localized deformations, failing to replicate this biological dexterity. Inspired by this natural exemplar, we introduce the paradigm of collaborative, whole-body proprioceptive morphing, realized in a modular soft gripper architecture. Our design is a distributed network of modular self-sensing pneumatic actuators that enables the gripper to intelligently reconfigure its entire topology, achieving multiple morphing states that are controllable to form diverse polygonal shapes. By integrating rich proprioceptive feedback from embedded sensors, our system can seamlessly transition from a precise pinch to a large envelope grasp. We experimentally demonstrate that this approach expands the grasping envelope and enhances generalization across diverse object geometries (standard and irregular) and scales (up to 10$\times$), while also unlocking novel manipulation modalities such as multi-object and internal hook grasping. This work presents a low-cost, easy-to-fabricate, and scalable framework that fuses distributed actuation with integrated sensing, offering a new pathway toward achieving biological levels of dexterity in robotic manipulation.
>
---
#### [new 017] Heterogeneous Robot Collaboration in Unstructured Environments with Grounded Generative Intelligence
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对异构机器人在非结构化环境中协作任务的挑战，提出SPINE-HT框架。通过三阶段流程将语言指令转化为可执行子任务，结合机器人能力与在线反馈进行任务规划与优化，显著提升任务成功率，实现了基于生成式智能的动态协同。**

- **链接: [http://arxiv.org/pdf/2510.26915v1](http://arxiv.org/pdf/2510.26915v1)**

> **作者:** Zachary Ravichandran; Fernando Cladera; Ankit Prabhu; Jason Hughes; Varun Murali; Camillo Taylor; George J. Pappas; Vijay Kumar
>
> **摘要:** Heterogeneous robot teams operating in realistic settings often must accomplish complex missions requiring collaboration and adaptation to information acquired online. Because robot teams frequently operate in unstructured environments -- uncertain, open-world settings without prior maps -- subtasks must be grounded in robot capabilities and the physical world. While heterogeneous teams have typically been designed for fixed specifications, generative intelligence opens the possibility of teams that can accomplish a wide range of missions described in natural language. However, current large language model (LLM)-enabled teaming methods typically assume well-structured and known environments, limiting deployment in unstructured environments. We present SPINE-HT, a framework that addresses these limitations by grounding the reasoning abilities of LLMs in the context of a heterogeneous robot team through a three-stage process. Given language specifications describing mission goals and team capabilities, an LLM generates grounded subtasks which are validated for feasibility. Subtasks are then assigned to robots based on capabilities such as traversability or perception and refined given feedback collected during online operation. In simulation experiments with closed-loop perception and control, our framework achieves nearly twice the success rate compared to prior LLM-enabled heterogeneous teaming approaches. In real-world experiments with a Clearpath Jackal, a Clearpath Husky, a Boston Dynamics Spot, and a high-altitude UAV, our method achieves an 87\% success rate in missions requiring reasoning about robot capabilities and refining subtasks with online feedback. More information is provided at https://zacravichandran.github.io/SPINE-HT.
>
---
#### [new 018] Hybrid Gripper Finger Enabling In-Grasp Friction Modulation Using Inflatable Silicone Pockets
- **分类: cs.RO**

- **简介: 该论文针对机器人抓取不同材质物体时易损伤或打滑的问题，提出一种融合刚性外壳与可充气硅胶腔的混合夹指。通过调节腔内气压实现表面摩擦力的主动调控，无需增大夹持力即可稳定抓取重、滑、脆性物体，提升抓取安全性与适应性。**

- **链接: [http://arxiv.org/pdf/2510.27184v1](http://arxiv.org/pdf/2510.27184v1)**

> **作者:** Hoang Hiep Ly; Cong-Nhat Nguyen; Doan-Quang Tran; Quoc-Khanh Dang; Ngoc Duy Tran; Thi Thoa Mac; Anh Nguyen; Xuan-Thuan Nguyen; Tung D. Ta
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** Grasping objects with diverse mechanical properties, such as heavy, slippery, or fragile items, remains a significant challenge in robotics. Conventional grippers often rely on applying high normal forces, which can cause damage to objects. To address this limitation, we present a hybrid gripper finger that combines a rigid structural shell with a soft, inflatable silicone pocket. The gripper finger can actively modulate its surface friction by controlling the internal air pressure of the silicone pocket. Results from fundamental experiments indicate that increasing the internal pressure results in a proportional increase in the effective coefficient of friction. This enables the gripper to stably lift heavy and slippery objects without increasing the gripping force and to handle fragile or deformable objects, such as eggs, fruits, and paper cups, with minimal damage by increasing friction rather than applying excessive force. The experimental results demonstrate that the hybrid gripper finger with adaptable friction provides a robust and safer alternative to relying solely on high normal forces, thereby enhancing the gripper flexibility in handling delicate, fragile, and diverse objects.
>
---
#### [new 019] Force Characterization of Insect-Scale Aquatic Propulsion Based on Fluid-Structure Interaction
- **分类: cs.RO; physics.flu-dyn**

- **简介: 该论文研究昆虫尺度水下推进器的力特性，针对微机器人在流体中高效推进的问题，基于流固耦合机制设计单尾与双尾推进器。通过自制微牛级力传感器，首次测得其瞬时推力，为微型水下机器人推进系统优化提供关键数据支持。**

- **链接: [http://arxiv.org/pdf/2510.26837v1](http://arxiv.org/pdf/2510.26837v1)**

> **作者:** Conor K. Trygstad; Nestor O. Perez-Arancibia
>
> **备注:** To be presented at ICAR 2025 in San Juan, Argentina
>
> **摘要:** We present force characterizations of two newly developed insect-scale propulsors--one single-tailed and one double-tailed--for microrobotic swimmers that leverage fluid-structure interaction (FSI) to generate thrust. The designs of these two devices were inspired by anguilliform swimming and are driven by soft tails excited by high-work-density (HWD) actuators powered by shape-memory alloy (SMA) wires. While these propulsors have been demonstrated to be suitable for microrobotic aquatic locomotion and controllable with simple architectures for trajectory tracking in the two-dimensional (2D) space, the characteristics and magnitudes of the associated forces have not been studied systematically. In the research presented here, we adopted a theoretical framework based on the notion of reactive forces and obtained experimental data for characterization using a custom-built micro-N-resolution force sensor. We measured maximum and cycle-averaged force values with multi-test means of respectively 0.45 mN and 2.97 micro-N, for the tested single-tail propulsor. For the dual-tail propulsor, we measured maximum and cycle-averaged force values with multi-test means of 0.61 mN and 22.6 micro-N, respectively. These results represent the first measurements of the instantaneous thrust generated by insect-scale propulsors of this type and provide insights into FSI for efficient microrobotic propulsion.
>
---
#### [new 020] Modified-Emergency Index (MEI): A Criticality Metric for Autonomous Driving in Lateral Conflict
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶中横向冲突安全评估难题，提出改进的紧急指数（MEI） metric，用于精确量化横向避让努力。基于Argoverse-2数据集验证，MEI优于传统ACT与PET指标，在风险演化捕捉与临界性评估上表现更优，提升城市道路安全评估能力。**

- **链接: [http://arxiv.org/pdf/2510.27333v1](http://arxiv.org/pdf/2510.27333v1)**

> **作者:** Hao Cheng; Yanbo Jiang; Qingyuan Shi; Qingwen Meng; Keyu Chen; Wenhao Yu; Jianqiang Wang; Sifa Zheng
>
> **摘要:** Effective, reliable, and efficient evaluation of autonomous driving safety is essential to demonstrate its trustworthiness. Criticality metrics provide an objective means of assessing safety. However, as existing metrics primarily target longitudinal conflicts, accurately quantifying the risks of lateral conflicts - prevalent in urban settings - remains challenging. This paper proposes the Modified-Emergency Index (MEI), a metric designed to quantify evasive effort in lateral conflicts. Compared to the original Emergency Index (EI), MEI refines the estimation of the time available for evasive maneuvers, enabling more precise risk quantification. We validate MEI on a public lateral conflict dataset based on Argoverse-2, from which we extract over 1,500 high-quality AV conflict cases, including more than 500 critical events. MEI is then compared with the well-established ACT and the widely used PET metrics. Results show that MEI consistently outperforms them in accurately quantifying criticality and capturing risk evolution. Overall, these findings highlight MEI as a promising metric for evaluating urban conflicts and enhancing the safety assessment framework for autonomous driving. The open-source implementation is available at https://github.com/AutoChengh/MEI.
>
---
#### [new 021] SpikeATac: A Multimodal Tactile Finger with Taxelized Dynamic Sensing for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文提出SpikeATac，一种融合动态PVDF与静态电容传感的多模态触觉手指，用于灵巧操作。针对脆弱易变形物体抓取难的问题，通过高采样率动态感知实现快速响应，并结合人类反馈强化学习优化抓握力控制，实现了精准的在手操控。**

- **链接: [http://arxiv.org/pdf/2510.27048v1](http://arxiv.org/pdf/2510.27048v1)**

> **作者:** Eric T. Chang; Peter Ballentine; Zhanpeng He; Do-Gon Kim; Kai Jiang; Hua-Hsuan Liang; Joaquin Palacios; William Wang; Pedro Piacenza; Ioannis Kymissis; Matei Ciocarlie
>
> **备注:** 9 pages, 8 figures, under review
>
> **摘要:** In this work, we introduce SpikeATac, a multimodal tactile finger combining a taxelized and highly sensitive dynamic response (PVDF) with a static transduction method (capacitive) for multimodal touch sensing. Named for its `spiky' response, SpikeATac's 16-taxel PVDF film sampled at 4 kHz provides fast, sensitive dynamic signals to the very onset and breaking of contact. We characterize the sensitivity of the different modalities, and show that SpikeATac provides the ability to stop quickly and delicately when grasping fragile, deformable objects. Beyond parallel grasping, we show that SpikeATac can be used in a learning-based framework to achieve new capabilities on a dexterous multifingered robot hand. We use a learning recipe that combines reinforcement learning from human feedback with tactile-based rewards to fine-tune the behavior of a policy to modulate force. Our hardware platform and learning pipeline together enable a difficult dexterous and contact-rich task that has not previously been achieved: in-hand manipulation of fragile objects. Videos are available at \href{https://roamlab.github.io/spikeatac/}{roamlab.github.io/spikeatac}.
>
---
#### [new 022] MobiDock: Design and Control of A Modular Self Reconfigurable Bimanual Mobile Manipulator via Robotic Docking
- **分类: cs.RO**

- **简介: 该论文提出MobiDock系统，解决多移动机械臂协同控制与动态稳定性难题。通过视觉引导自主对接和螺纹锁紧机制，实现两机器人物理重组为单一人形双臂平台，显著提升运动稳定性和任务效率。**

- **链接: [http://arxiv.org/pdf/2510.27178v1](http://arxiv.org/pdf/2510.27178v1)**

> **作者:** Xuan-Thuan Nguyen; Khac Nam Nguyen; Ngoc Duy Tran; Thi Thoa Mac; Anh Nguyen; Hoang Hiep Ly; Tung D. Ta
>
> **备注:** ICRA2026 submited
>
> **摘要:** Multi-robot systems, particularly mobile manipulators, face challenges in control coordination and dynamic stability when working together. To address this issue, this study proposes MobiDock, a modular self-reconfigurable mobile manipulator system that allows two independent robots to physically connect and form a unified mobile bimanual platform. This process helps transform a complex multi-robot control problem into the management of a simpler, single system. The system utilizes an autonomous docking strategy based on computer vision with AprilTag markers and a new threaded screw-lock mechanism. Experimental results show that the docked configuration demonstrates better performance in dynamic stability and operational efficiency compared to two independently cooperating robots. Specifically, the unified system has lower Root Mean Square (RMS) Acceleration and Jerk values, higher angular precision, and completes tasks significantly faster. These findings confirm that physical reconfiguration is a powerful design principle that simplifies cooperative control, improving stability and performance for complex tasks in real-world environments.
>
---
#### [new 023] WildfireX-SLAM: A Large-scale Low-altitude RGB-D Dataset for Wildfire SLAM and Beyond
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对大尺度森林场景下3DGS-SLAM缺乏高质量数据的问题，构建了首个大规模低空RGB-D合成数据集WildfireX-SLAM，包含16km²森林区域的5.5k张图像。通过引擎实现多模态数据采集与环境可控性，支持火灾应急响应等应用，并开展基准测试揭示挑战与改进方向。**

- **链接: [http://arxiv.org/pdf/2510.27133v1](http://arxiv.org/pdf/2510.27133v1)**

> **作者:** Zhicong Sun; Jacqueline Lo; Jinxing Hu
>
> **备注:** This paper has been accepted by MMM 2026
>
> **摘要:** 3D Gaussian splatting (3DGS) and its subsequent variants have led to remarkable progress in simultaneous localization and mapping (SLAM). While most recent 3DGS-based SLAM works focus on small-scale indoor scenes, developing 3DGS-based SLAM methods for large-scale forest scenes holds great potential for many real-world applications, especially for wildfire emergency response and forest management. However, this line of research is impeded by the absence of a comprehensive and high-quality dataset, and collecting such a dataset over real-world scenes is costly and technically infeasible. To this end, we have built a large-scale, comprehensive, and high-quality synthetic dataset for SLAM in wildfire and forest environments. Leveraging the Unreal Engine 5 Electric Dreams Environment Sample Project, we developed a pipeline to easily collect aerial and ground views, including ground-truth camera poses and a range of additional data modalities from unmanned aerial vehicle. Our pipeline also provides flexible controls on environmental factors such as light, weather, and types and conditions of wildfire, supporting the need for various tasks covering forest mapping, wildfire emergency response, and beyond. The resulting pilot dataset, WildfireX-SLAM, contains 5.5k low-altitude RGB-D aerial images from a large-scale forest map with a total size of 16 km2. On top of WildfireX-SLAM, a thorough benchmark is also conducted, which not only reveals the unique challenges of 3DGS-based SLAM in the forest but also highlights potential improvements for future works. The dataset and code will be publicly available. Project page: https://zhicongsun.github.io/wildfirexslam.
>
---
#### [new 024] Cooperative Integrated Estimation-Guidance for Simultaneous Interception of Moving Targets
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; math.OC**

- **简介: 该论文针对多无人车协同拦截非机动目标的任务，解决传感器分布不均导致的目标状态估计与制导难题。提出一种融合估计与制导的协同框架，利用有向通信拓扑和预设时间观测器，使无传感器车辆实现状态估计；结合精确时间到相遇制导，确保在预定时间内完成拦截与时间到相遇一致。**

- **链接: [http://arxiv.org/pdf/2510.26948v1](http://arxiv.org/pdf/2510.26948v1)**

> **作者:** Lohitvel Gopikannan; Shashi Ranjan Kumar; Abhinav Sinha
>
> **摘要:** This paper proposes a cooperative integrated estimation-guidance framework for simultaneous interception of a non-maneuvering target using a team of unmanned autonomous vehicles, assuming only a subset of vehicles are equipped with dedicated sensors to measure the target's states. Unlike earlier approaches that focus solely on either estimation or guidance design, the proposed framework unifies both within a cooperative architecture. To circumvent the limitation posed by heterogeneity in target observability, sensorless vehicles estimate the target's state by leveraging information exchanged with neighboring agents over a directed communication topology through a prescribed-time observer. The proposed approach employs true proportional navigation guidance (TPNG), which uses an exact time-to-go formulation and is applicable across a wide spectrum of target motions. Furthermore, prescribed-time observer and controller are employed to achieve convergence to true target's state and consensus in time-to-go within set predefined times, respectively. Simulations demonstrate the effectiveness of the proposed framework under various engagement scenarios.
>
---
#### [new 025] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DUST框架，用于增强视觉-语言-动作模型（VLAs）的机器人策略学习。针对观测与动作模态差异导致的联合预测难题，设计双流扩散架构，分离处理模态并支持跨模态共享，引入独立噪声与解耦损失，实现双向联合建模。实验表明其在仿真与真实世界任务中均显著提升性能。**

- **链接: [http://arxiv.org/pdf/2510.27607v1](http://arxiv.org/pdf/2510.27607v1)**

> **作者:** John Won; Kyungmin Lee; Huiwon Jang; Dongyoung Kim; Jinwoo Shin
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Recently, augmenting Vision-Language-Action models (VLAs) with world modeling has shown promise in improving robotic policy learning. However, it remains challenging to jointly predict next-state observations and action sequences because of the inherent difference between the two modalities. To address this, we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework that handles the modality conflict and enhances the performance of VLAs across diverse tasks. Specifically, we propose a multimodal diffusion transformer architecture that explicitly maintains separate modality streams while still enabling cross-modal knowledge sharing. In addition, we introduce independent noise perturbations for each modality and a decoupled flow-matching loss. This design enables the model to learn the joint distribution in a bidirectional manner while avoiding the need for a unified latent space. Based on the decoupling of modalities during training, we also introduce a joint sampling method that supports test-time scaling, where action and vision tokens evolve asynchronously at different rates. Through experiments on simulated benchmarks such as RoboCasa and GR-1, DUST achieves up to 6% gains over baseline methods, while our test-time scaling approach provides an additional 2-5% boost. On real-world tasks with the Franka Research 3, DUST improves success rates by 13%, confirming its effectiveness beyond simulation. Furthermore, pre-training on action-free videos from BridgeV2 yields significant transfer gains on RoboCasa, underscoring DUST's potential for large-scale VLA pretraining.
>
---
## 更新

#### [replaced 001] Mechanical Intelligence-Aware Curriculum Reinforcement Learning for Humanoids with Parallel Actuation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00273v3](http://arxiv.org/pdf/2507.00273v3)**

> **作者:** Yusuke Tanaka; Alvin Zhu; Quanyou Wang; Yeting Liu; Dennis Hong
>
> **备注:** Proceeding to the IEEE Humanoid Conference 2025
>
> **摘要:** Reinforcement learning (RL) has enabled advances in humanoid robot locomotion, yet most learning frameworks do not account for mechanical intelligence embedded in parallel actuation mechanisms due to limitations in simulator support for closed kinematic chains. This omission can lead to inaccurate motion modeling and suboptimal policies, particularly for robots with high actuation complexity. This paper presents general formulations and simulation methods for three types of parallel mechanisms: a differential pulley, a five-bar linkage, and a four-bar linkage, and trains a parallel-mechanism aware policy through an end-to-end curriculum RL framework for BRUCE, a kid-sized humanoid robot. Unlike prior approaches that rely on simplified serial approximations, we simulate all closed-chain constraints natively using GPU-accelerated MuJoCo (MJX), preserving the hardware's mechanical nonlinear properties during training. We benchmark our RL approach against a model predictive controller (MPC), demonstrating better surface generalization and performance in real-world zero-shot deployment. This work highlights the computational approaches and performance benefits of fully simulating parallel mechanisms in end-to-end learning pipelines for legged humanoids. Project codes with parallel mechanisms: https://github.com/alvister88/og_bruce
>
---
#### [replaced 002] A Practical-Driven Framework for Transitioning Drive-by-Wire to Autonomous Driving Systems: A Case Study with a Chrysler Pacifica Hybrid Vehicle
- **分类: cs.RO; cs.OS; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.06492v2](http://arxiv.org/pdf/2410.06492v2)**

> **作者:** Dada Zhang; Md Ruman Islam; Pei-Chi Huang; Chun-Hsing Ho
>
> **备注:** This updated version includes further implementation details and experimental validation. Accepted for presentation at The 22nd International Conference on Automation Technology (AUTOMATION 2025), Taipei, Taiwan, November 2025
>
> **摘要:** Transitioning from a Drive-by-Wire (DBW) system to a fully autonomous driving system (ADS) involves multiple stages of development and demands robust positioning and sensing capabilities. This paper presents a practice-driven framework for facilitating the DBW-to-ADS transition using a 2022 Chrysler Pacifica Hybrid Minivan equipped with cameras, LiDAR, GNSS, and onboard computing hardware configured with the Robot Operating System (ROS) and Autoware.AI. The implementation showcases offline autonomous operations utilizing pre-recorded LiDAR and camera data, point clouds, and vector maps, enabling effective localization and path planning within a structured test environment. The study addresses key challenges encountered during the transition, particularly those related to wireless-network-assisted sensing and positioning. It offers practical solutions for overcoming software incompatibility constraints, sensor synchronization issues, and limitations in real-time perception. Furthermore, the integration of sensing, data fusion, and automation is emphasized as a critical factor in supporting autonomous driving systems in map generation, simulation, and training. Overall, the transition process outlined in this work aims to provide actionable strategies for researchers pursuing DBW-to-ADS conversion. It offers direction for incorporating real-time perception, GNSS-LiDAR-camera integration, and fully ADS-equipped autonomous vehicle operations, thus contributing to the advancement of robust autonomous vehicle technologies.
>
---
#### [replaced 003] Robust Offline Reinforcement Learning with Linearly Structured f-Divergence Regularization
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.18612v2](http://arxiv.org/pdf/2411.18612v2)**

> **作者:** Cheng Tang; Zhishuai Liu; Pan Xu
>
> **备注:** 41 pages, 3 figures, 2 tables. Published in Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** The Robust Regularized Markov Decision Process (RRMDP) is proposed to learn policies robust to dynamics shifts by adding regularization to the transition dynamics in the value function. Existing methods mostly use unstructured regularization, potentially leading to conservative policies under unrealistic transitions. To address this limitation, we propose a novel framework, the $d$-rectangular linear RRMDP ($d$-RRMDP), which introduces latent structures into both transition kernels and regularization. We focus on offline reinforcement learning, where an agent learns policies from a precollected dataset in the nominal environment. We develop the Robust Regularized Pessimistic Value Iteration (R2PVI) algorithm that employs linear function approximation for robust policy learning in $d$-RRMDPs with $f$-divergence based regularization terms on transition kernels. We provide instance-dependent upper bounds on the suboptimality gap of R2PVI policies, demonstrating that these bounds are influenced by how well the dataset covers state-action spaces visited by the optimal robust policy under robustly admissible transitions. We establish information-theoretic lower bounds to verify that our algorithm is near-optimal. Finally, numerical experiments validate that R2PVI learns robust policies and exhibits superior computational efficiency compared to baseline methods.
>
---
#### [replaced 004] Online Adaptation for Flying Quadrotors in Tight Formations
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.17488v3](http://arxiv.org/pdf/2506.17488v3)**

> **作者:** Pei-An Hsieh; Kong Yao Chee; M. Ani Hsieh
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** The task of flying in tight formations is challenging for teams of quadrotors because the complex aerodynamic wake interactions can destabilize individual team members as well as the team. Furthermore, these aerodynamic effects are highly nonlinear and fast-paced, making them difficult to model and predict. To overcome these challenges, we present L1 KNODE-DW MPC, an adaptive, mixed expert learning based control framework that allows individual quadrotors to accurately track trajectories while adapting to time-varying aerodynamic interactions during formation flights. We evaluate L1 KNODE-DW MPC in two different three-quadrotor formations and show that it outperforms several MPC baselines. Our results show that the proposed framework is capable of enabling the three-quadrotor team to remain vertically aligned in close proximity throughout the flight. These findings show that the L1 adaptive module compensates for unmodeled disturbances most effectively when paired with an accurate dynamics model. A video showcasing our framework and the physical experiments is available here: https://youtu.be/9QX1Q5Ut9Rs
>
---
#### [replaced 005] Faster Model Predictive Control via Self-Supervised Initialization Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.03394v3](http://arxiv.org/pdf/2408.03394v3)**

> **作者:** Zhaoxin Li; Xiaoke Wang; Letian Chen; Rohan Paleja; Subramanya Nageshrao; Matthew Gombolay
>
> **摘要:** Model Predictive Control (MPC) is widely used in robot control by optimizing a sequence of control outputs over a finite-horizon. Computational approaches for MPC include deterministic methods (e.g., iLQR and COBYLA), as well as sampling-based methods (e.g., MPPI and CEM). However, complex system dynamics and non-convex or non-differentiable cost terms often lead to prohibitive optimization times that limit real-world deployment. Prior efforts to accelerate MPC have limitations on: (i) reusing previous solutions fails under sharp state changes and (ii) pure imitation learning does not target compute efficiency directly and suffers from suboptimality in the training data. To address these, We propose a warm-start framework that learns a policy to generate high-quality initial guesses for MPC solver. The policy is first trained via behavior cloning from expert MPC rollouts and then fine-tuned online with reinforcement learning to directly minimize MPC optimization time. We empirically validate that our approach improves both deterministic and sampling-based MPC methods, achieving up to 21.6% faster optimization and 34.1% more tracking accuracy for deterministic MPC in Formula 1 track path-tracking domain, and improving safety by 100%, path efficiency by 12.8%, and steering smoothness by 7.2% for sampling-based MPC in obstacle-rich navigation domain. These results demonstrate that our framework not only accelerates MPC but also improves overall control performance. Furthermore, it can be applied to a broader range of control algorithms that benefit from good initial guesses.
>
---
#### [replaced 006] PoseDiff: A Unified Diffusion Model Bridging Robot Pose Estimation and Video-to-Action Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.24591v2](http://arxiv.org/pdf/2509.24591v2)**

> **作者:** Haozhuo Zhang; Michele Caprio; Jing Shao; Qiang Zhang; Jian Tang; Shanghang Zhang; Wei Pan
>
> **备注:** The experimental setup and metrics lacks rigor, affecting the fairness of the comparisons
>
> **摘要:** We present PoseDiff, a conditional diffusion model that unifies robot state estimation and control within a single framework. At its core, PoseDiff maps raw visual observations into structured robot states-such as 3D keypoints or joint angles-from a single RGB image, eliminating the need for multi-stage pipelines or auxiliary modalities. Building upon this foundation, PoseDiff extends naturally to video-to-action inverse dynamics: by conditioning on sparse video keyframes generated by world models, it produces smooth and continuous long-horizon action sequences through an overlap-averaging strategy. This unified design enables scalable and efficient integration of perception and control. On the DREAM dataset, PoseDiff achieves state-of-the-art accuracy and real-time performance for pose estimation. On Libero-Object manipulation tasks, it substantially improves success rates over existing inverse dynamics modules, even under strict offline settings. Together, these results show that PoseDiff provides a scalable, accurate, and efficient bridge between perception, planning, and control in embodied AI. The video visualization results can be found on the project page: https://haozhuo-zhang.github.io/PoseDiff-project-page/.
>
---
#### [replaced 007] RObotic MAnipulation Network (ROMAN) -- Hybrid Hierarchical Learning for Solving Complex Sequential Tasks
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2307.00125v3](http://arxiv.org/pdf/2307.00125v3)**

> **作者:** Eleftherios Triantafyllidis; Fernando Acero; Zhaocheng Liu; Zhibin Li
>
> **备注:** To appear in Nature Machine Intelligence. Includes the main and supplementary manuscript. Total of 70 pages, with a total of 9 Figures and 17 Tables
>
> **摘要:** Solving long sequential tasks poses a significant challenge in embodied artificial intelligence. Enabling a robotic system to perform diverse sequential tasks with a broad range of manipulation skills is an active area of research. In this work, we present a Hybrid Hierarchical Learning framework, the Robotic Manipulation Network (ROMAN), to address the challenge of solving multiple complex tasks over long time horizons in robotic manipulation. ROMAN achieves task versatility and robust failure recovery by integrating behavioural cloning, imitation learning, and reinforcement learning. It consists of a central manipulation network that coordinates an ensemble of various neural networks, each specialising in distinct re-combinable sub-tasks to generate their correct in-sequence actions for solving complex long-horizon manipulation tasks. Experimental results show that by orchestrating and activating these specialised manipulation experts, ROMAN generates correct sequential activations for accomplishing long sequences of sophisticated manipulation tasks and achieving adaptive behaviours beyond demonstrations, while exhibiting robustness to various sensory noises. These results demonstrate the significance and versatility of ROMAN's dynamic adaptability featuring autonomous failure recovery capabilities, and highlight its potential for various autonomous manipulation tasks that demand adaptive motor skills.
>
---
#### [replaced 008] Understanding the Application of Utility Theory in Robotics and Artificial Intelligence: A Survey
- **分类: cs.RO; cs.AI; cs.MA; cs.NE; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2306.09445v2](http://arxiv.org/pdf/2306.09445v2)**

> **作者:** Qin Yang; Rui Liu
>
> **备注:** I am not sure whether withdrawing this paper is suitable. However, right now this paper has significant changes in its topic and author. So, I do not want to lead to any confusion about this paper. In the future, it will have a new version. I hope people will not have issues and confusion about the older one
>
> **摘要:** As a unifying concept in economics, game theory, and operations research, even in the Robotics and AI field, the utility is used to evaluate the level of individual needs, preferences, and interests. Especially for decision-making and learning in multi-agent/robot systems (MAS/MRS), a suitable utility model can guide agents in choosing reasonable strategies to achieve their current needs and learning to cooperate and organize their behaviors, optimizing the system's utility, building stable and reliable relationships, and guaranteeing each group member's sustainable development, similar to the human society. Although these systems' complex, large-scale, and long-term behaviors are strongly determined by the fundamental characteristics of the underlying relationships, there has been less discussion on the theoretical aspects of mechanisms and the fields of applications in Robotics and AI. This paper introduces a utility-orient needs paradigm to describe and evaluate inter and outer relationships among agents' interactions. Then, we survey existing literature in relevant fields to support it and propose several promising research directions along with some open problems deemed necessary for further investigations.
>
---
#### [replaced 009] Object-Centric Kinodynamic Planning for Nonprehensile Robot Rearrangement Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.00261v4](http://arxiv.org/pdf/2410.00261v4)**

> **作者:** Kejia Ren; Gaotian Wang; Andrew S. Morgan; Lydia E. Kavraki; Kaiyu Hang
>
> **摘要:** Nonprehensile actions such as pushing are crucial for addressing multi-object rearrangement problems. Many traditional methods generate robot-centric actions, which differ from intuitive human strategies and are typically inefficient. To this end, we adopt an object-centric planning paradigm and propose a unified framework for addressing a range of large-scale, physics-intensive nonprehensile rearrangement problems challenged by modeling inaccuracies and real-world uncertainties. By assuming each object can actively move without being driven by robot interactions, our planner first computes desired object motions, which are then realized through robot actions generated online via a closed-loop pushing strategy. Through extensive experiments and in comparison with state-of-the-art baselines in both simulation and on a physical robot, we show that our object-centric planning framework can generate more intuitive and task-effective robot actions with significantly improved efficiency. In addition, we propose a benchmarking protocol to standardize and facilitate future research in nonprehensile rearrangement.
>
---
#### [replaced 010] A Tactile Feedback Approach to Path Recovery after High-Speed Impacts for Collision-Resilient Drones
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.14249v3](http://arxiv.org/pdf/2410.14249v3)**

> **作者:** Anton Bredenbeck; Teaya Yang; Salua Hamaza; Mark W. Mueller
>
> **摘要:** Aerial robots are a well-established solution for exploration, monitoring, and inspection, thanks to their superior maneuverability and agility. However, in many environments, they risk crashing and sustaining damage after collisions. Traditional methods focus on avoiding obstacles entirely, but these approaches can be limiting, particularly in cluttered spaces or on weight-and compute-constrained platforms such as drones. This paper presents a novel approach to enhance drone robustness and autonomy by developing a path recovery and adjustment method for a high-speed collision-resilient aerial robot equipped with lightweight, distributed tactile sensors. The proposed system explicitly models collisions using pre-collision velocities, rates and tactile feedback to predict post-collision dynamics, improving state estimation accuracy. Additionally, we introduce a computationally efficient vector-field-based path representation that guarantees convergence to a user-specified path, while naturally avoiding known obstacles. Post-collision, contact point locations are incorporated into the vector field as a repulsive potential, enabling the drone to avoid obstacles while naturally returning to its path. The effectiveness of this method is validated through Monte Carlo simulations and demonstrated on a physical prototype, showing successful path following, collision recovery, and adjustment at speeds up to 3.7 m/s.
>
---
#### [replaced 011] Sim2Real Diffusion: Leveraging Foundation Vision Language Models for Adaptive Automated Driving
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00236v3](http://arxiv.org/pdf/2507.00236v3)**

> **作者:** Chinmay Vilas Samak; Tanmay Vilas Samak; Bing Li; Venkat Krovi
>
> **备注:** Accepted in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Simulation-based design, optimization, and validation of autonomous vehicles have proven to be crucial for their improvement over the years. Nevertheless, the ultimate measure of effectiveness is their successful transition from simulation to reality (sim2real). However, existing sim2real transfer methods struggle to address the autonomy-oriented requirements of balancing: (i) conditioned domain adaptation, (ii) robust performance with limited examples, (iii) modularity in handling multiple domain representations, and (iv) real-time performance. To alleviate these pain points, we present a unified framework for learning cross-domain adaptive representations through conditional latent diffusion for sim2real transferable automated driving. Our framework offers options to leverage: (i) alternate foundation models, (ii) a few-shot fine-tuning pipeline, and (iii) textual as well as image prompts for mapping across given source and target domains. It is also capable of generating diverse high-quality samples when diffusing across parameter spaces such as times of day, weather conditions, seasons, and operational design domains. We systematically analyze the presented framework and report our findings in terms of performance benchmarks and ablation studies. Additionally, we demonstrate its serviceability for autonomous driving using behavioral cloning case studies. Our experiments indicate that the proposed framework is capable of bridging the perceptual sim2real gap by over 40%.
>
---
#### [replaced 012] GenSwarm: Scalable Multi-Robot Code-Policy Generation and Deployment via Language Models
- **分类: cs.RO; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2503.23875v2](http://arxiv.org/pdf/2503.23875v2)**

> **作者:** Wenkang Ji; Huaben Chen; Mingyang Chen; Guobin Zhu; Lufeng Xu; Roderich Groß; Rui Zhou; Ming Cao; Shiyu Zhao
>
> **备注:** This article has been accepted for publication in npj Robotics
>
> **摘要:** The development of control policies for multi-robot systems traditionally follows a complex and labor-intensive process, often lacking the flexibility to adapt to dynamic tasks. This has motivated research on methods to automatically create control policies. However, these methods require iterative processes of manually crafting and refining objective functions, thereby prolonging the development cycle. This work introduces \textit{GenSwarm}, an end-to-end system that leverages large language models to automatically generate and deploy control policies for multi-robot tasks based on simple user instructions in natural language. As a multi-language-agent system, GenSwarm achieves zero-shot learning, enabling rapid adaptation to altered or unseen tasks. The white-box nature of the code policies ensures strong reproducibility and interpretability. With its scalable software and hardware architectures, GenSwarm supports efficient policy deployment on both simulated and real-world multi-robot systems, realizing an instruction-to-execution end-to-end functionality that could prove valuable for robotics specialists and non-specialists alike.The code of the proposed GenSwarm system is available online: https://github.com/WindyLab/GenSwarm.
>
---
#### [replaced 013] SafeAgentBench: A Benchmark for Safe Task Planning of Embodied LLM Agents
- **分类: cs.CR; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.13178v5](http://arxiv.org/pdf/2412.13178v5)**

> **作者:** Sheng Yin; Xianghe Pang; Yuanzhuo Ding; Menglan Chen; Yutong Bi; Yichen Xiong; Wenhao Huang; Zhen Xiang; Jing Shao; Siheng Chen
>
> **备注:** 28 pages, 19 tables, 15 figures
>
> **摘要:** With the integration of large language models (LLMs), embodied agents have strong capabilities to understand and plan complicated natural language instructions. However, a foreseeable issue is that those embodied agents can also flawlessly execute some hazardous tasks, potentially causing damages in the real world. Existing benchmarks predominantly overlook critical safety risks, focusing solely on planning performance, while a few evaluate LLMs' safety awareness only on non-interactive image-text data. To address this gap, we present SafeAgentBench -- the first comprehensive benchmark for safety-aware task planning of embodied LLM agents in interactive simulation environments, covering both explicit and implicit hazards. SafeAgentBench includes: (1) an executable, diverse, and high-quality dataset of 750 tasks, rigorously curated to cover 10 potential hazards and 3 task types; (2) SafeAgentEnv, a universal embodied environment with a low-level controller, supporting multi-agent execution with 17 high-level actions for 9 state-of-the-art baselines; and (3) reliable evaluation methods from both execution and semantic perspectives. Experimental results show that, although agents based on different design frameworks exhibit substantial differences in task success rates, their overall safety awareness remains weak. The most safety-conscious baseline achieves only a 10% rejection rate for detailed hazardous tasks. Moreover, simply replacing the LLM driving the agent does not lead to notable improvements in safety awareness. Dataset and codes are available in https://github.com/shengyin1224/SafeAgentBench and https://huggingface.co/datasets/safeagentbench/SafeAgentBench.
>
---
#### [replaced 014] From Canada to Japan: How 10,000 km Affect User Perception in Robot Teleoperation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.05143v2](http://arxiv.org/pdf/2508.05143v2)**

> **作者:** Siméon Capy; Thomas M. Kwok; Kevin Joseph; Yuichiro Kawasumi; Koichi Nagashima; Tomoya Sasaki; Yue Hu; Eiichi Yoshida
>
> **备注:** Author preprint - Accepted for Humanoids 2025
>
> **摘要:** Robot teleoperation (RTo) has emerged as a viable alternative to local control, particularly when human intervention is still necessary. This research aims to study the distance effect on user perception in RTo, exploring the potential of teleoperated robots for older adult care. We propose an evaluation of non-expert users' perception of long-distance RTo, examining how their perception changes before and after interaction, as well as comparing it to that of locally operated robots. We have designed a specific protocol consisting of multiple questionnaires, along with a dedicated software architecture using the Robotics Operating System (ROS) and Unity. The results revealed no statistically significant differences between the local and remote robot conditions, suggesting that robots may be a viable alternative to traditional local control.
>
---
#### [replaced 015] A Study on Human-Swarm Interaction: A Framework for Assessing Situation Awareness and Task Performance
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.14810v3](http://arxiv.org/pdf/2503.14810v3)**

> **作者:** Wasura D. Wattearachchi; Erandi Lakshika; Kathryn Kasmarik; Michael Barlow
>
> **备注:** 10 pages, 8 figures, 2 tables, 2 equations
>
> **摘要:** This paper introduces a framework for human swarm interaction studies that measures situation awareness in dynamic environments. A tablet-based interface was developed for a user study by implementing the concepts introduced in the framework, where operators guided a robotic swarm in a single-target search task, marking hazardous cells unknown to the swarm. Both subjective and objective situation awareness measures were used, with task performance evaluated based on how close the robots were to the target. The framework enabled a structured investigation of the role of situation awareness in human swarm interaction, leading to key findings such as improved task performance across attempts, showing the interface was learnable, centroid active robot position proved to be a useful task performance metric for assessing situation awareness, perception and projection played a key role in task performance, highlighting their importance in interface design and objective situation awareness influenced both subjective situation awareness and task performance, emphasizing the need for interfaces that emphasise objective situation awareness. These findings validate our framework as a structured approach for integrating situation awareness concepts into human swarm interaction studies, offering a systematic way to assess situation awareness and task performance. The framework can be applied to other swarming studies to evaluate interface learnability, identify meaningful task performance metrics, and refine interface designs to enhance situation awareness, ultimately improving human swarm interaction in dynamic environments.
>
---
#### [replaced 016] Uncertainty-Based Smooth Policy Regularisation for Reinforcement Learning with Few Demonstrations
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2509.15981v2](http://arxiv.org/pdf/2509.15981v2)**

> **作者:** Yujie Zhu; Charles A. Hepburn; Matthew Thorpe; Giovanni Montana
>
> **摘要:** In reinforcement learning with sparse rewards, demonstrations can accelerate learning, but determining when to imitate them remains challenging. We propose Smooth Policy Regularisation from Demonstrations (SPReD), a framework that addresses the fundamental question: when should an agent imitate a demonstration versus follow its own policy? SPReD uses ensemble methods to explicitly model Q-value distributions for both demonstration and policy actions, quantifying uncertainty for comparisons. We develop two complementary uncertainty-aware methods: a probabilistic approach estimating the likelihood of demonstration superiority, and an advantage-based approach scaling imitation by statistical significance. Unlike prevailing methods (e.g. Q-filter) that make binary imitation decisions, SPReD applies continuous, uncertainty-proportional regularisation weights, reducing gradient variance during training. Despite its computational simplicity, SPReD achieves remarkable gains in experiments across eight robotics tasks, outperforming existing approaches by up to a factor of 14 in complex tasks while maintaining robustness to demonstration quality and quantity. Our code is available at https://github.com/YujieZhu7/SPReD.
>
---
#### [replaced 017] Vision-Based Online Key Point Estimation of Deformable Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2211.05222v3](http://arxiv.org/pdf/2211.05222v3)**

> **作者:** Hehui Zheng; Sebastian Pinzello; Barnabas Gavin Cangan; Thomas Buchner; Robert K. Katzschmann
>
> **摘要:** The precise control of soft and continuum robots requires knowledge of their shape, which has, in contrast to classical rigid robots, infinite degrees of freedom. To partially reconstruct the shape, proprioceptive techniques use built-in sensors resulting in inaccurate results and increased fabrication complexity. Exteroceptive methods so far rely on expensive tracking systems with reflective markers placed on all components, which are infeasible for deformable robots interacting with the environment due to marker occlusion and damage. Here, a regression approach is presented for 3D key point estimation using a convolutional neural network. The proposed approach takes advantage of data-driven supervised learning and is capable of online marker-less estimation during inference. Two images of a robotic system are taken simultaneously at 25 Hz from two different perspectives, and are fed to the network, which returns for each pair the parameterized key point or PCC shape representations. The proposed approach outperforms marker-less state-of-the-art methods by a maximum of 4.5% in estimation accuracy while at the same time being more robust and requiring no prior knowledge of the shape. Online evaluations on two types of soft robotic arms and a soft robotic fish demonstrate our method's accuracy and versatility on highly deformable systems.
>
---
#### [replaced 018] Panoramic Out-of-Distribution Segmentation for Autonomous Driving
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2505.03539v2](http://arxiv.org/pdf/2505.03539v2)**

> **作者:** Mengfei Duan; Yuheng Zhang; Yihong Cao; Fei Teng; Kai Luo; Jiaming Zhang; Kailun Yang; Zhiyong Li
>
> **备注:** Code and datasets will be available at https://github.com/MengfeiD/PanOoS
>
> **摘要:** Panoramic imaging enables capturing 360{\deg} images with an ultra-wide Field-of-View (FoV) for dense omnidirectional perception, which is critical to applications, such as autonomous driving and augmented reality, etc. However, current panoramic semantic segmentation methods fail to identify outliers, and pinhole Out-of-distribution Segmentation (OoS) models perform unsatisfactorily in the panoramic domain due to background clutter and pixel distortions. To address these issues, we introduce a new task, Panoramic Out-of-distribution Segmentation (PanOoS), with the aim of achieving comprehensive and safe scene understanding. Furthermore, we propose the first solution, POS, which adapts to the characteristics of panoramic images through text-guided prompt distribution learning. Specifically, POS integrates a disentanglement strategy designed to materialize the cross-domain generalization capability of CLIP. The proposed Prompt-based Restoration Attention (PRA) optimizes semantic decoding by prompt guidance and self-adaptive correction, while Bilevel Prompt Distribution Learning (BPDL) refines the manifold of per-pixel mask embeddings via semantic prototype supervision. Besides, to compensate for the scarcity of PanOoS datasets, we establish two benchmarks: DenseOoS, which features diverse outliers in complex environments, and QuadOoS, captured by a quadruped robot with a panoramic annular lens system. Extensive experiments demonstrate superior performance of POS, with AuPRC improving by 34.25% and FPR95 decreasing by 21.42% on DenseOoS, outperforming state-of-the-art pinhole-OoS methods. Moreover, POS achieves leading closed-set segmentation capabilities and advances the development of panoramic understanding. Code and datasets will be available at https://github.com/MengfeiD/PanOoS.
>
---
