# 机器人 cs.RO

- **最新发布 61 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains
- **分类: cs.RO**

- **简介: 该论文属于机器人感知运动任务，解决复杂地形下多方向稳健行走问题。提出RPL框架，通过两阶段训练提升机器人在不同地形上的适应能力。**

- **链接: [https://arxiv.org/pdf/2602.03002v1](https://arxiv.org/pdf/2602.03002v1)**

> **作者:** Yuanhang Zhang; Younggyo Seo; Juyue Chen; Yifu Yuan; Koushil Sreenath; Pieter Abbeel; Carmelo Sferrazza; Karen Liu; Rocky Duan; Guanya Shi
>
> **摘要:** Humanoid perceptive locomotion has made significant progress and shows great promise, yet achieving robust multi-directional locomotion on complex terrains remains underexplored. To tackle this challenge, we propose RPL, a two-stage training framework that enables multi-directional locomotion on challenging terrains, and remains robust with payloads. RPL first trains terrain-specific expert policies with privileged height map observations to master decoupled locomotion and manipulation skills across different terrains, and then distills them into a transformer policy that leverages multiple depth cameras to cover a wide range of views. During distillation, we introduce two techniques to robustify multi-directional locomotion, depth feature scaling based on velocity commands and random side masking, which are critical for asymmetric depth observations and unseen widths of terrains. For scalable depth distillation, we develop an efficient multi-depth system that ray-casts against both dynamic robot meshes and static terrain meshes in massively parallel environments, achieving a 5-times speedup over the depth rendering pipelines in existing simulators while modeling realistic sensor latency, noise, and dropout. Extensive real-world experiments demonstrate robust multi-directional locomotion with payloads (2kg) across challenging terrains, including 20° slopes, staircases with different step lengths (22 cm, 25 cm, 30 cm), and 25 cm by 25 cm stepping stones separated by 60 cm gaps.
>
---
#### [new 002] CMR: Contractive Mapping Embeddings for Robust Humanoid Locomotion on Unstructured Terrains
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，解决复杂地形下人体模型的鲁棒性问题。通过构建收缩映射嵌入框架，减少扰动影响，提升运动稳定性。**

- **链接: [https://arxiv.org/pdf/2602.03511v1](https://arxiv.org/pdf/2602.03511v1)**

> **作者:** Qixin Zeng; Hongyin Zhang; Shangke Lyu; Junxi Jin; Donglin Wang; Chao Huang
>
> **摘要:** Robust disturbance rejection remains a longstanding challenge in humanoid locomotion, particularly on unstructured terrains where sensing is unreliable and model mismatch is pronounced. While perception information, such as height map, enhances terrain awareness, sensor noise and sim-to-real gaps can destabilize policies in practice. In this work, we provide theoretical analysis that bounds the return gap under observation noise, when the induced latent dynamics are contractive. Furthermore, we present Contractive Mapping for Robustness (CMR) framework that maps high-dimensional, disturbance-prone observations into a latent space, where local perturbations are attenuated over time. Specifically, this approach couples contrastive representation learning with Lipschitz regularization to preserve task-relevant geometry while explicitly controlling sensitivity. Notably, the formulation can be incorporated into modern deep reinforcement learning pipelines as an auxiliary loss term with minimal additional technical effort required. Further, our extensive humanoid experiments show that CMR potently outperforms other locomotion algorithms under increased noise.
>
---
#### [new 003] AffordanceGrasp-R1:Leveraging Reasoning-Based Affordance Segmentation with Reinforcement Learning for Robotic Grasping
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在提升复杂场景下的抓取性能。通过结合推理与强化学习，提出AffordanceGrasp-R1框架，优化抓取候选生成与筛选过程。**

- **链接: [https://arxiv.org/pdf/2602.03547v1](https://arxiv.org/pdf/2602.03547v1)**

> **作者:** Dingyi Zhou; Mu He; Zhuowei Fang; Xiangtong Yao; Yinlong Liu; Alois Knoll; Hu Cao
>
> **备注:** Preprint version
>
> **摘要:** We introduce AffordanceGrasp-R1, a reasoning-driven affordance segmentation framework for robotic grasping that combines a chain-of-thought (CoT) cold-start strategy with reinforcement learning to enhance deduction and spatial grounding. In addition, we redesign the grasping pipeline to be more context-aware by generating grasp candidates from the global scene point cloud and subsequently filtering them using instruction-conditioned affordance masks. Extensive experiments demonstrate that AffordanceGrasp-R1 consistently outperforms state-of-the-art (SOTA) methods on benchmark datasets, and real-world robotic grasping evaluations further validate its robustness and generalization under complex language-conditioned manipulation scenarios.
>
---
#### [new 004] Hierarchical Proportion Models for Motion Generation via Integration of Motion Primitives
- **分类: cs.RO**

- **简介: 该论文属于机器人运动学习任务，旨在提升模仿学习的数据效率与适应性。通过分层框架整合运动基元，解决复杂任务生成问题。**

- **链接: [https://arxiv.org/pdf/2602.03188v1](https://arxiv.org/pdf/2602.03188v1)**

> **作者:** Yu-Han Shu; Toshiaki Tsuji; Sho Sakaino
>
> **备注:** 6 pages, 9 figures. Accepted for publication in IEEE AMC 2026
>
> **摘要:** Imitation learning (IL) enables robots to acquire human-like motion skills from demonstrations, but it still requires extensive high-quality data and retraining to handle complex or long-horizon tasks. To improve data efficiency and adaptability, this study proposes a hierarchical IL framework that integrates motion primitives with proportion-based motion synthesis. The proposed method employs a two-layer architecture, where the upper layer performs long-term planning, while a set of lower-layer models learn individual motion primitives, which are combined according to specific proportions. Three model variants are introduced to explore different trade-offs between learning flexibility, computational cost, and adaptability: a learning-based proportion model, a sampling-based proportion model, and a playback-based proportion model, which differ in how the proportions are determined and whether the upper layer is trainable. Through real-robot pick-and-place experiments, the proposed models successfully generated complex motions not included in the primitive set. The sampling-based and playback-based proportion models achieved more stable and adaptable motion generation than the standard hierarchical model, demonstrating the effectiveness of proportion-based motion integration for practical robot learning.
>
---
#### [new 005] Investigating the Influence of Spatial Ability in Augmented Reality-assisted Robot Programming
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于教育技术领域，研究AR对机器人编程学习的影响，探讨空间能力在其中的调节作用。通过实验发现AR可补偿低空间能力学习者的劣势。**

- **链接: [https://arxiv.org/pdf/2602.03544v1](https://arxiv.org/pdf/2602.03544v1)**

> **作者:** Nicolas Leins; Jana Gonnermann-Müller; Malte Teichmann; Sebastian Pokutta
>
> **摘要:** Augmented Reality (AR) offers promising opportunities to enhance learning, but its mechanisms and effects are not yet fully understood. As learning becomes increasingly personalized, considering individual learner characteristics becomes more important. This study investigates the moderating effect of spatial ability on learning experience with AR in the context of robot programming. A between-subjects experiment ($N=71$) compared conventional robot programming to an AR-assisted approach using a head-mounted display. Participants' spatial ability was assessed using the Mental Rotation Test. The learning experience was measured through the System Usability Scale (SUS) and cognitive load. The results indicate that AR support does not significantly improve the learning experience compared to the conventional approach. However, AR appears to have a compensatory effect on the influence of spatial ability. In the control group, spatial ability was significantly positively associated with SUS scores and negatively associated with extraneous cognitive load, indicating that higher spatial ability predicts a better learning experience. In the AR condition, these relationships were not observable, suggesting that AR mitigated the disadvantage typically experienced by learners with lower spatial abilities. These findings suggest that AR can serve a compensatory function by reducing the influence of learner characteristics. Future research should further explore this compensatory role of AR to guide the design of personalized learning environments that address diverse learner needs and reduce barriers for learners with varying cognitive profiles.
>
---
#### [new 006] HUSKY: Humanoid Skateboarding System via Physics-Aware Whole-Body Control
- **分类: cs.RO**

- **简介: 该论文研究人形滑板任务，解决动态平衡与复杂交互问题。提出HUSKY框架，融合物理模型与学习方法，实现稳定滑板操控。**

- **链接: [https://arxiv.org/pdf/2602.03205v1](https://arxiv.org/pdf/2602.03205v1)**

> **作者:** Jinrui Han; Dewei Wang; Chenyun Zhang; Xinzhe Liu; Ping Luo; Chenjia Bai; Xuelong Li
>
> **摘要:** While current humanoid whole-body control frameworks predominantly rely on the static environment assumptions, addressing tasks characterized by high dynamism and complex interactions presents a formidable challenge. In this paper, we address humanoid skateboarding, a highly challenging task requiring stable dynamic maneuvering on an underactuated wheeled platform. This integrated system is governed by non-holonomic constraints and tightly coupled human-object interactions. Successfully executing this task requires simultaneous mastery of hybrid contact dynamics and robust balance control on a mechanically coupled, dynamically unstable skateboard. To overcome the aforementioned challenges, we propose HUSKY, a learning-based framework that integrates humanoid-skateboard system modeling and physics-aware whole-body control. We first model the coupling relationship between board tilt and truck steering angles, enabling a principled analysis of system dynamics. Building upon this, HUSKY leverages Adversarial Motion Priors (AMP) to learn human-like pushing motions and employs a physics-guided, heading-oriented strategy for lean-to-steer behaviors. Moreover, a trajectory-guided mechanism ensures smooth and stable transitions between pushing and steering. Experimental results on the Unitree G1 humanoid platform demonstrate that our framework enables stable and agile maneuvering on skateboards in real-world scenarios. The project page is available on https://husky-humanoid.github.io/.
>
---
#### [new 007] Manipulation via Force Distribution at Contact
- **分类: cs.RO**

- **简介: 该论文研究接触丰富的操作任务，解决传统点接触模型无法准确模拟摩擦和扭矩的问题。提出FDLC模型，并构建双层优化框架，提升轨迹效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.03350v1](https://arxiv.org/pdf/2602.03350v1)**

> **作者:** Haegu Lee; Yitaek Kim; Casper Hewson Rask; Christoffer Sloth
>
> **摘要:** Efficient and robust trajectories play a crucial role in contact-rich manipulation, which demands accurate mod- eling of object-robot interactions. Many existing approaches rely on point contact models due to their computational effi- ciency. Simple contact models are computationally efficient but inherently limited for achieving human-like, contact-rich ma- nipulation, as they fail to capture key frictional dynamics and torque generation observed in human manipulation. This study introduces a Force-Distributed Line Contact (FDLC) model in contact-rich manipulation and compares it against conventional point contact models. A bi-level optimization framework is constructed, in which the lower-level solves an optimization problem for contact force computation, and the upper-level optimization applies iLQR for trajectory optimization. Through this framework, the limitations of point contact are demon- strated, and the benefits of the FDLC in generating efficient and robust trajectories are established. The effectiveness of the proposed approach is validated by a box rotating task, demonstrating that FDLC enables trajectories generated via non-uniform force distributions along the contact line, while requiring lower control effort and less motion of the robot.
>
---
#### [new 008] Latent Perspective-Taking via a Schrödinger Bridge in Influence-Augmented Local Models
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，解决机器人在不确定环境中推理他人心理状态的问题。通过学习结构化心理模型和视角转换机制，实现社会感知决策。**

- **链接: [https://arxiv.org/pdf/2602.02857v1](https://arxiv.org/pdf/2602.02857v1)**

> **作者:** Kevin Alcedo; Pedro U. Lima; Rachid Alami
>
> **备注:** Extended Abstract & Poster, Presented at World Modeling Workshop 2026
>
> **摘要:** Operating in environments alongside humans requires robots to make decisions under uncertainty. In addition to exogenous dynamics, they must reason over others' hidden mental-models and mental-states. While Interactive POMDPs and Bayesian Theory of Mind formulations are principled, exact nested-belief inference is intractable, and hand-specified models are brittle in open-world settings. We address both by learning structured mental-models and an estimator of others' mental-states. Building on the Influence-Based Abstraction, we instantiate an Influence-Augmented Local Model to decompose socially-aware robot tasks into local dynamics, social influences, and exogenous factors. We propose (a) a neuro-symbolic world model instantiating a factored, discrete Dynamic Bayesian Network, and (b) a perspective-shift operator modeled as an amortized Schrödinger Bridge over the learned local dynamics that transports factored egocentric beliefs into other-centric beliefs. We show that this architecture enables agents to synthesize socially-aware policies in model-based reinforcement learning, via decision-time mental-state planning (a Schrödinger Bridge in belief space), with preliminary results in a MiniGrid social navigation task.
>
---
#### [new 009] Adaptive Linear Path Model-Based Diffusion
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决扩散模型对参数敏感的问题。提出LP-MBD和ALP-MBD方法，简化调度并提升适应性与效率。**

- **链接: [https://arxiv.org/pdf/2602.02831v1](https://arxiv.org/pdf/2602.02831v1)**

> **作者:** Yutaka Shimizu; Masayoshi Tomizuka
>
> **备注:** ICRA 2026
>
> **摘要:** The interest in combining model-based control approaches with diffusion models has been growing. Although we have seen many impressive robotic control results in difficult tasks, the performance of diffusion models is highly sensitive to the choice of scheduling parameters, making parameter tuning one of the most critical challenges. We introduce Linear Path Model-Based Diffusion (LP-MBD), which replaces the variance-preserving schedule with a flow-matching-inspired linear probability path. This yields a geometrically interpretable and decoupled parameterization that reduces tuning complexity and provides a stable foundation for adaptation. Building on this, we propose Adaptive LP-MBD (ALP-MBD), which leverages reinforcement learning to adjust diffusion steps and noise levels according to task complexity and environmental conditions. Across numerical studies, Brax benchmarks, and mobile-robot trajectory tracking, LP-MBD simplifies scheduling while maintaining strong performance, and ALP-MBD further improves robustness, adaptability, and real-time efficiency.
>
---
#### [new 010] Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决单一策略难以适应不同人形机器人的问题。通过EAGLE框架，实现跨机器人、多行为的统一控制。**

- **链接: [https://arxiv.org/pdf/2602.02960v1](https://arxiv.org/pdf/2602.02960v1)**

> **作者:** Quanquan Peng; Yunfeng Lin; Yufei Xue; Jiangmiao Pang; Weinan Zhang
>
> **摘要:** Humanoid Whole-Body Controllers trained with reinforcement learning (RL) have recently achieved remarkable performance, yet many target a single robot embodiment. Variations in dynamics, degrees of freedom (DoFs), and kinematic topology still hinder a single policy from commanding diverse humanoids. Moreover, obtaining a generalist policy that not only transfers across embodiments but also supports richer behaviors-beyond simple walking to squatting, leaning-remains especially challenging. In this work, we tackle these obstacles by introducing EAGLE, an iterative generalist-specialist distillation framework that produces a single unified policy that controls multiple heterogeneous humanoids without per-robot reward tuning. During each cycle, embodiment-specific specialists are forked from the current generalist, refined on their respective robots, and new skills are distilled back into the generalist by training on the pooled embodiment set. Repeating this loop until performance convergence produces a robust Whole-Body Controller validated on robots such as Unitree H1, G1, and Fourier N1. We conducted experiments on five different robots in simulation and four in real-world settings. Through quantitative evaluations, EAGLE achieves high tracking accuracy and robustness compared to other methods, marking a step toward scalable, fleet-level humanoid control. See more details at https://eagle-wbc.github.io/
>
---
#### [new 011] HetroD: A High-Fidelity Drone Dataset and Benchmark for Autonomous Driving in Heterogeneous Traffic
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出HetroD数据集和基准，用于解决异质交通中自动驾驶的挑战，聚焦于脆弱道路使用者的行为建模与预测。**

- **链接: [https://arxiv.org/pdf/2602.03447v1](https://arxiv.org/pdf/2602.03447v1)**

> **作者:** Yu-Hsiang Chen; Wei-Jer Chang; Christian Kotulla; Thomas Keutgens; Steffen Runde; Tobias Moers; Christoph Klas; Wei Zhan; Masayoshi Tomizuka; Yi-Ting Chen
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** We present HetroD, a dataset and benchmark for developing autonomous driving systems in heterogeneous environments. HetroD targets the critical challenge of navi- gating real-world heterogeneous traffic dominated by vulner- able road users (VRUs), including pedestrians, cyclists, and motorcyclists that interact with vehicles. These mixed agent types exhibit complex behaviors such as hook turns, lane splitting, and informal right-of-way negotiation. Such behaviors pose significant challenges for autonomous vehicles but remain underrepresented in existing datasets focused on structured, lane-disciplined traffic. To bridge the gap, we collect a large- scale drone-based dataset to provide a holistic observation of traffic scenes with centimeter-accurate annotations, HD maps, and traffic signal states. We further develop a modular toolkit for extracting per-agent scenarios to support downstream task development. In total, the dataset comprises over 65.4k high- fidelity agent trajectories, 70% of which are from VRUs. HetroD supports modeling of VRU behaviors in dense, het- erogeneous traffic and provides standardized benchmarks for forecasting, planning, and simulation tasks. Evaluation results reveal that state-of-the-art prediction and planning models struggle with the challenges presented by our dataset: they fail to predict lateral VRU movements, cannot handle unstructured maneuvers, and exhibit limited performance in dense and multi-agent scenarios, highlighting the need for more robust approaches to heterogeneous traffic. See our project page for more examples: https://hetroddata.github.io/HetroD/
>
---
#### [new 012] A Scene Graph Backed Approach to Open Set Semantic Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决开放集语义映射中的一致性与可扩展性问题。提出以3DSSG为底层结构的映射架构，实现实时更新与高效推理。**

- **链接: [https://arxiv.org/pdf/2602.03781v1](https://arxiv.org/pdf/2602.03781v1)**

> **作者:** Martin Günther; Felix Igelbrink; Oscar Lima; Lennart Niecksch; Marian Renz; Martin Atzmueller
>
> **摘要:** While Open Set Semantic Mapping and 3D Semantic Scene Graphs (3DSSGs) are established paradigms in robotic perception, deploying them effectively to support high-level reasoning in large-scale, real-world environments remains a significant challenge. Most existing approaches decouple perception from representation, treating the scene graph as a derivative layer generated post hoc. This limits both consistency and scalability. In contrast, we propose a mapping architecture where the 3DSSG serves as the foundational backend, acting as the primary knowledge representation for the entire mapping process. Our approach leverages prior work on incremental scene graph prediction to infer and update the graph structure in real-time as the environment is explored. This ensures that the map remains topologically consistent and computationally efficient, even during extended operations in large-scale settings. By maintaining an explicit, spatially grounded representation that supports both flat and hierarchical topologies, we bridge the gap between sub-symbolic raw sensor data and high-level symbolic reasoning. Consequently, this provides a stable, verifiable structure that knowledge-driven frameworks, ranging from knowledge graphs and ontologies to Large Language Models (LLMs), can directly exploit, enabling agents to operate with enhanced interpretability, trustworthiness, and alignment to human concepts.
>
---
#### [new 013] Modular Isoperimetric Soft Robotic Truss for Lunar Applications
- **分类: cs.RO**

- **简介: 该论文提出一种模块化软体桁架结构，用于月球任务。解决空间部署与适应性问题，通过可变形三角形实现可重构结构，具备高压缩比和无需额外气源的形状变化能力。**

- **链接: [https://arxiv.org/pdf/2602.02915v1](https://arxiv.org/pdf/2602.02915v1)**

> **作者:** Mihai Stanciu; Isaac Weaver; Adam Rose; James Wade; Kaden Paxton; Chris Paul; Spencer Stowell; Nathan Usevitch
>
> **摘要:** We introduce a large-scale robotic system designed as a lightweight, modular, and reconfigurable structure for lunar applications. The system consists of truss-like robotic triangles formed by continuous inflated fabric tubes routed through two robotic roller units and a connecting unit. A newly developed spherical joint enables up to three triangles to connect at a vertex, allowing construction of truss assemblies beyond a single octahedron. When deflated, the triangles compact to approximately the volume of the roller units, achieving a stowed-to-deployed volume ratio of 1:18.3. Upon inflation, the roller units pinch the tubes, locally reducing bending stiffness to form effective joints. Electric motors then translate the roller units along the tube, shifting the pinch point by lengthening one edge while shortening another at the same rate, thereby preserving a constant perimeter (isoperimetric). This shape-changing process requires no additional compressed air, enabling untethered operation after initial inflation. We demonstrate the system as a 12-degree-of-freedom solar array capable of tilting up to 60 degrees and sweeping 360 degrees, and as a 14-degree-of-freedom locomotion device using a step-and-slide gait. This modular, shape-adaptive system addresses key challenges for sustainable lunar operations and future space missions.
>
---
#### [new 014] Human-in-the-Loop Failure Recovery with Adaptive Task Allocation
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，解决机器人故障时如何高效分配给合适操作员的问题。通过建模操作员能力并动态分配故障，提升系统效率与工作负载平衡。**

- **链接: [https://arxiv.org/pdf/2602.03603v1](https://arxiv.org/pdf/2602.03603v1)**

> **作者:** Lorena Maria Genua; Nikita Boguslavskii; Zhi Li
>
> **摘要:** Since the recent Covid-19 pandemic, mobile manipulators and humanoid assistive robots with higher levels of autonomy have increasingly been adopted for patient care and living assistance. Despite advancements in autonomy, these robots often struggle to perform reliably in dynamic and unstructured environments and require human intervention to recover from failures. Effective human-robot collaboration is essential to enable robots to receive assistance from the most competent operator, in order to reduce their workload and minimize disruptions in task execution. In this paper, we propose an adaptive method for allocating robotic failures to human operators (ARFA). Our proposed approach models the capabilities of human operators, and continuously updates these beliefs based on their actual performance for failure recovery. For every failure to be resolved, a reward function calculates expected outcomes based on operator capabilities and historical data, task urgency, and current workload distribution. The failure is then assigned to the operator with the highest expected reward. Our simulations and user studies show that ARFA outperforms random allocation, significantly reducing robot idle time, improving overall system performance, and leading to a more distributed workload among operators.
>
---
#### [new 015] Multi-Player, Multi-Strategy Quantum Game Model for Interaction-Aware Decision-Making in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶决策任务，旨在解决交互感知不足的问题。通过引入量子博弈模型，提升多车交互下的决策性能。**

- **链接: [https://arxiv.org/pdf/2602.03571v1](https://arxiv.org/pdf/2602.03571v1)**

> **作者:** Karim Essalmi; Fernando Garrido; Fawzi Nashashibi
>
> **摘要:** Although significant progress has been made in decision-making for automated driving, challenges remain for deployment in the real world. One challenge lies in addressing interaction-awareness. Most existing approaches oversimplify interactions between the ego vehicle and surrounding agents, and often neglect interactions among the agents themselves. A common solution is to model these interactions using classical game theory. However, its formulation assumes rational players, whereas human behavior is frequently uncertain or irrational. To address these challenges, we propose the Quantum Game Decision-Making (QGDM) model, a novel framework that combines classical game theory with quantum mechanics principles (such as superposition, entanglement, and interference) to tackle multi-player, multi-strategy decision-making problems. To the best of our knowledge, this is one of the first studies to apply quantum game theory to decision-making for automated driving. QGDM runs in real time on a standard computer, without requiring quantum hardware. We evaluate QGDM in simulation across various scenarios, including roundabouts, merging, and highways, and compare its performance with multiple baseline methods. Results show that QGDM significantly improves success rates and reduces collision rates compared to classical approaches, particularly in scenarios with high interaction.
>
---
#### [new 016] PokeNet: Learning Kinematic Models of Articulated Objects from Human Observations
- **分类: cs.RO**

- **简介: 论文提出PokeNet，用于从人类操作中学习可动物体的运动学模型。解决无先验知识下准确估计关节参数和操作顺序的问题，通过单次人类示范实现。**

- **链接: [https://arxiv.org/pdf/2602.02741v1](https://arxiv.org/pdf/2602.02741v1)**

> **作者:** Anmol Gupta; Weiwei Gu; Omkar Patil; Jun Ki Lee; Nakul Gopalan
>
> **摘要:** Articulation modeling enables robots to learn joint parameters of articulated objects for effective manipulation which can then be used downstream for skill learning or planning. Existing approaches often rely on prior knowledge about the objects, such as the number or type of joints. Some of these approaches also fail to recover occluded joints that are only revealed during interaction. Others require large numbers of multi-view images for every object, which is impractical in real-world settings. Furthermore, prior works neglect the order of manipulations, which is essential for many multi-DoF objects where one joint must be operated before another, such as a dishwasher. We introduce PokeNet, an end-to-end framework that estimates articulation models from a single human demonstration without prior object knowledge. Given a sequence of point cloud observations of a human manipulating an unknown object, PokeNet predicts joint parameters, infers manipulation order, and tracks joint states over time. PokeNet outperforms existing state-of-the-art methods, improving joint axis and state estimation accuracy by an average of over 27% across diverse objects, including novel and unseen categories. We demonstrate these gains in both simulation and real-world environments.
>
---
#### [new 017] RDT2: Exploring the Scaling Limit of UMI Data Towards Zero-Shot Cross-Embodiment Generalization
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型在数据稀缺和跨平台泛化上的问题。通过构建RDT2模型，实现零样本跨硬件平台的通用任务执行。**

- **链接: [https://arxiv.org/pdf/2602.03310v1](https://arxiv.org/pdf/2602.03310v1)**

> **作者:** Songming Liu; Bangguo Li; Kai Ma; Lingxuan Wu; Hengkai Tan; Xiao Ouyang; Hang Su; Jun Zhu
>
> **摘要:** Vision-Language-Action (VLA) models hold promise for generalist robotics but currently struggle with data scarcity, architectural inefficiencies, and the inability to generalize across different hardware platforms. We introduce RDT2, a robotic foundation model built upon a 7B parameter VLM designed to enable zero-shot deployment on novel embodiments for open-vocabulary tasks. To achieve this, we collected one of the largest open-source robotic datasets--over 10,000 hours of demonstrations in diverse families--using an enhanced, embodiment-agnostic Universal Manipulation Interface (UMI). Our approach employs a novel three-stage training recipe that aligns discrete linguistic knowledge with continuous control via Residual Vector Quantization (RVQ), flow-matching, and distillation for real-time inference. Consequently, RDT2 becomes one of the first models that simultaneously zero-shot generalizes to unseen objects, scenes, instructions, and even robotic platforms. Besides, it outperforms state-of-the-art baselines in dexterous, long-horizon, and dynamic downstream tasks like playing table tennis. See https://rdt-robotics.github.io/rdt2/ for more information.
>
---
#### [new 018] Learning-based Adaptive Control of Quadruped Robots for Active Stabilization on Moving Platforms
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决四足机器人在移动平台上的稳定问题。通过引入学习驱动的自平衡策略和状态估计方法，提升机器人在复杂运动环境中的稳定性。**

- **链接: [https://arxiv.org/pdf/2602.03367v1](https://arxiv.org/pdf/2602.03367v1)**

> **作者:** Minsung Yoon; Heechan Shin; Jeil Jeong; Sung-Eui Yoon
>
> **备注:** Accepted to IROS 2024. <a href="https://sgvr.kaist.ac.kr/~msyoon/papers/IROS24/" rel="external noopener nofollow" class="link-external link-https">Project Page</a>
>
> **摘要:** A quadruped robot faces balancing challenges on a six-degrees-of-freedom moving platform, like subways, buses, airplanes, and yachts, due to independent platform motions and resultant diverse inertia forces on the robot. To alleviate these challenges, we present the Learning-based Active Stabilization on Moving Platforms (\textit{LAS-MP}), featuring a self-balancing policy and system state estimators. The policy adaptively adjusts the robot's posture in response to the platform's motion. The estimators infer robot and platform states based on proprioceptive sensor data. For a systematic training scheme across various platform motions, we introduce platform trajectory generation and scheduling methods. Our evaluation demonstrates superior balancing performance across multiple metrics compared to three baselines. Furthermore, we conduct a detailed analysis of the \textit{LAS-MP}, including ablation studies and evaluation of the estimators, to validate the effectiveness of each component.
>
---
#### [new 019] When Attention Betrays: Erasing Backdoor Attacks in Robotic Policies by Reconstructing Visual Tokens
- **分类: cs.RO**

- **简介: 该论文属于机器人安全任务，解决VLA模型中的后门攻击问题。通过分析注意力机制，提出Bera框架，在不重新训练模型的情况下检测并消除后门，提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2602.03153v1](https://arxiv.org/pdf/2602.03153v1)**

> **作者:** Xuetao Li; Pinhan Fu; Wenke Huang; Nengyuan Pan; Songhua Yang; Kaiyan Zhao; Guancheng Wan; Mengde Li; Jifeng Xuan; Miao Li
>
> **备注:** ICRA2026 accepted
>
> **摘要:** Downstream fine-tuning of vision-language-action (VLA) models enhances robotics, yet exposes the pipeline to backdoor risks. Attackers can pretrain VLAs on poisoned data to implant backdoors that remain stealthy but can trigger harmful behavior during inference. However, existing defenses either lack mechanistic insight into multimodal backdoors or impose prohibitive computational costs via full-model retraining. To this end, we uncover a deep-layer attention grabbing mechanism: backdoors redirect late-stage attention and form compact embedding clusters near the clean manifold. Leveraging this insight, we introduce Bera, a test-time backdoor erasure framework that detects tokens with anomalous attention via latent-space localization, masks suspicious regions using deep-layer cues, and reconstructs a trigger-free image to break the trigger-unsafe-action mapping while restoring correct behavior. Unlike prior defenses, Bera requires neither retraining of VLAs nor any changes to the training pipeline. Extensive experiments across multiple embodied platforms and tasks show that Bera effectively maintains nominal performance, significantly reduces attack success rates, and consistently restores benign behavior from backdoored outputs, thereby offering a robust and practical defense mechanism for securing robotic systems.
>
---
#### [new 020] Collision Detection with Analytical Derivatives of Contact Kinematics
- **分类: cs.RO**

- **简介: 该论文属于机器人接触检测任务，解决非光滑接触映射问题，通过几何正则化构建平滑接触模型，提出iDCOL框架实现精确的接触动力学计算。**

- **链接: [https://arxiv.org/pdf/2602.03250v1](https://arxiv.org/pdf/2602.03250v1)**

> **作者:** Anup Teejo Mathew; Anees Peringal; Daniele Caradonna; Frederic Boyer; Federico Renda
>
> **备注:** 12 pages, 9 figures, 2 tables
>
> **摘要:** Differentiable contact kinematics are essential for gradient-based methods in robotics, yet the mapping from robot state to contact distance, location, and normal becomes non-smooth in degenerate configurations of shapes with zero or undefined curvature. We address this inherent limitation by selectively regularizing such geometries into strictly convex implicit representations, restoring uniqueness and smoothness of the contact map. Leveraging this geometric regularization, we develop iDCOL, an implicit differentiable collision detection and contact kinematics framework. iDCOL represents colliding bodies using strictly convex implicit surfaces and computes collision detection and contact kinematics by solving a fixed-size nonlinear system derived from a geometric scaling-based convex optimization formulation. By applying the Implicit Function Theorem to the resulting system residual, we derive analytical derivatives of the contact kinematic quantities. We develop a fast Newton-based solver for iDCOL and provide an open-source C++ implementation of the framework. The robustness of the approach is evaluated through extensive collision simulations and benchmarking, and applicability is demonstrated in gradient-based kinematic path planning and differentiable contact physics, including multi-body rigid collisions and a soft-robot interaction example.
>
---
#### [new 021] Omnidirectional Solid-State mmWave Radar Perception for UAV Power Line Collision Avoidance
- **分类: cs.RO**

- **简介: 该论文属于无人机避障任务，旨在解决电力线检测与避让问题。通过集成毫米波雷达模块，实现全方位感知，提升飞行安全性。**

- **链接: [https://arxiv.org/pdf/2602.03229v1](https://arxiv.org/pdf/2602.03229v1)**

> **作者:** Nicolaj Haarhøj Malle; Emad Ebeid
>
> **备注:** Accepted for publication at the 2026 IEEE International Conference on Robotics and Automation (ICRA). Video at https://www.youtube.com/watch?v=rJW3eEC-5Ao (youtube)
>
> **摘要:** Detecting and estimating distances to power lines is a challenge for both human UAV pilots and autonomous systems, which increases the risk of unintended collisions. We present a mmWave radar-based perception system that provides spherical sensing coverage around a small UAV for robust power line detection and avoidance. The system integrates multiple compact solid-state mmWave radar modules to synthesize an omnidirectional field of view while remaining lightweight. We characterize the sensing behavior of this omnidirectional radar arrangement in power line environments and develop a robust detection-and-avoidance algorithm tailored to that behavior. Field experiments on real power lines demonstrate reliable detection at ranges up to 10 m, successful avoidance maneuvers at flight speeds upwards of 10 m/s, and detection of wires as thin as 1.2 mm in diameter. These results indicate the approach's suitability as an additional safety layer for both autonomous and manual UAV flight.
>
---
#### [new 022] Multi-function Robotized Surgical Dissector for Endoscopic Pulmonary Thromboendarterectomy: Preclinical Study and Evaluation
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决传统手术工具灵活性不足的问题。研究设计了一种新型柔顺手术分离器，提升肺动脉内复杂结构的手术操作能力。**

- **链接: [https://arxiv.org/pdf/2602.03147v1](https://arxiv.org/pdf/2602.03147v1)**

> **作者:** Runfeng Zhu; Xin Zhong; Qingxiang Zhao; Jing Lin; Zhong Wu; Kang Li
>
> **摘要:** Patients suffering chronic severe pulmonary thromboembolism need Pulmonary Thromboendarterectomy (PTE) to remove the thromb and intima located inside pulmonary artery (PA). During the surgery, a surgeon holds tweezers and a dissector to delicately strip the blockage, but available tools for this surgery are rigid and straight, lacking distal dexterity to access into thin branches of PA. Therefore, this work presents a novel robotized dissector based on concentric push/pull robot (CPPR) structure, enabling entering deep thin branch of tortuous PA. Compared with conventional rigid dissectors, our design characterizes slenderness and dual-segment-bending dexterity. Owing to the hollow and thin-walled structure of the CPPR-based dissector as it has a slender body of 3.5mm in diameter, the central lumen accommodates two channels for irrigation and tip tool, and space for endoscopic camera's signal wire. To provide accurate surgical manipulation, optimization-based kinematics model was established, realizing a 2mm accuracy in positioning the tip tool (60mm length) under open-loop control strategy. As such, with the endoscopic camera, traditional PTE is possible to be upgraded as endoscopic PTE. Basic physic performance of the robotized dissector including stiffness, motion accuracy and maneuverability was evaluated through experiments. Surgery simulation on ex vivo porcine lung also demonstrates its dexterity and notable advantages in PTE.
>
---
#### [new 023] Enhancing Navigation Efficiency of Quadruped Robots via Leveraging Personal Transportation Platforms
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在提升四足机器人的长距离导航效率。通过引入RL-ATR方法，使机器人能利用个人交通工具，减少能耗并提高运动效率。**

- **链接: [https://arxiv.org/pdf/2602.03397v1](https://arxiv.org/pdf/2602.03397v1)**

> **作者:** Minsung Yoon; Sung-Eui Yoon
>
> **备注:** Accepted to ICRA 2025. <a href="https://sgvr.kaist.ac.kr/~msyoon/papers/ICRA25/" rel="external noopener nofollow" class="link-external link-https">Project Page</a>
>
> **摘要:** Quadruped robots face limitations in long-range navigation efficiency due to their reliance on legs. To ameliorate the limitations, we introduce a Reinforcement Learning-based Active Transporter Riding method (\textit{RL-ATR}), inspired by humans' utilization of personal transporters, including Segways. The \textit{RL-ATR} features a transporter riding policy and two state estimators. The policy devises adequate maneuvering strategies according to transporter-specific control dynamics, while the estimators resolve sensor ambiguities in non-inertial frames by inferring unobservable robot and transporter states. Comprehensive evaluations in simulation validate proficient command tracking abilities across various transporter-robot models and reduced energy consumption compared to legged locomotion. Moreover, we conduct ablation studies to quantify individual component contributions within the \textit{RL-ATR}. This riding ability could broaden the locomotion modalities of quadruped robots, potentially expanding the operational range and efficiency.
>
---
#### [new 024] StepNav: Structured Trajectory Priors for Efficient and Multimodal Visual Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决复杂环境中生成安全、高效轨迹的问题。提出StepNav框架，利用结构化多模态先验提升导航性能。**

- **链接: [https://arxiv.org/pdf/2602.02590v1](https://arxiv.org/pdf/2602.02590v1)**

> **作者:** Xubo Luo; Aodi Wu; Haodong Han; Xue Wan; Wei Zhang; Leizheng Shu; Ruisuo Wang
>
> **备注:** 8 pages, 7 figures; Accepted by ICRA 2026
>
> **摘要:** Visual navigation is fundamental to autonomous systems, yet generating reliable trajectories in cluttered and uncertain environments remains a core challenge. Recent generative models promise end-to-end synthesis, but their reliance on unstructured noise priors often yields unsafe, inefficient, or unimodal plans that cannot meet real-time requirements. We propose StepNav, a novel framework that bridges this gap by introducing structured, multimodal trajectory priors derived from variational principles. StepNav first learns a geometry-aware success probability field to identify all feasible navigation corridors. These corridors are then used to construct an explicit, multi-modal mixture prior that initializes a conditional flow-matching process. This refinement is formulated as an optimal control problem with explicit smoothness and safety regularization. By replacing unstructured noise with physically-grounded candidates, StepNav generates safer and more efficient plans in significantly fewer steps. Experiments in both simulation and real-world benchmarks demonstrate consistent improvements in robustness, efficiency, and safety over state-of-the-art generative planners, advancing reliable trajectory generation for practical autonomous navigation. The code has been released at https://github.com/LuoXubo/StepNav.
>
---
#### [new 025] Language Movement Primitives: Grounding Language Models in Robot Motion
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决从自然语言指令到机器人运动的映射问题。通过结合视觉语言模型与动态运动基元，提出LMP框架，实现零样本机器人操作。**

- **链接: [https://arxiv.org/pdf/2602.02839v1](https://arxiv.org/pdf/2602.02839v1)**

> **作者:** Yinlong Dai; Benjamin A. Christie; Daniel J. Evans; Dylan P. Losey; Simon Stepputtis
>
> **摘要:** Enabling robots to perform novel manipulation tasks from natural language instructions remains a fundamental challenge in robotics, despite significant progress in generalized problem solving with foundational models. Large vision and language models (VLMs) are capable of processing high-dimensional input data for visual scene and language understanding, as well as decomposing tasks into a sequence of logical steps; however, they struggle to ground those steps in embodied robot motion. On the other hand, robotics foundation models output action commands, but require in-domain fine-tuning or experience before they are able to perform novel tasks successfully. At its core, there still remains the fundamental challenge of connecting abstract task reasoning with low-level motion control. To address this disconnect, we propose Language Movement Primitives (LMPs), a framework that grounds VLM reasoning in Dynamic Movement Primitive (DMP) parameterization. Our key insight is that DMPs provide a small number of interpretable parameters, and VLMs can set these parameters to specify diverse, continuous, and stable trajectories. Put another way: VLMs can reason over free-form natural language task descriptions, and semantically ground their desired motions into DMPs -- bridging the gap between high-level task reasoning and low-level position and velocity control. Building on this combination of VLMs and DMPs, we formulate our LMP pipeline for zero-shot robot manipulation that effectively completes tabletop manipulation problems by generating a sequence of DMP motions. Across 20 real-world manipulation tasks, we show that LMP achieves 80% task success as compared to 31% for the best-performing baseline. See videos at our website: https://collab.me.vt.edu/lmp
>
---
#### [new 026] Bimanual High-Density EMG Control for In-Home Mobile Manipulation by a User with Quadriplegia
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决高位截瘫用户控制家用机器人的问题。通过开发双臂高密度肌电系统，实现手势控制机器人完成日常任务。**

- **链接: [https://arxiv.org/pdf/2602.02773v1](https://arxiv.org/pdf/2602.02773v1)**

> **作者:** Jehan Yang; Eleanor Hodgson; Cindy Sun; Zackory Erickson; Doug Weber
>
> **备注:** 14 pages, 17 figures
>
> **摘要:** Mobile manipulators in the home can enable people with cervical spinal cord injury (cSCI) to perform daily physical household tasks that they could not otherwise do themselves. However, paralysis in these users often limits access to traditional robot control interfaces such as joysticks or keyboards. In this work, we introduce and deploy the first system that enables a user with quadriplegia to control a mobile manipulator in their own home using bimanual high-density electromyography (HDEMG). We develop a pair of custom, fabric-integrated HDEMG forearm sleeves, worn on both arms, that capture residual neuromotor activity from clinically paralyzed degrees of freedom and support real-time gesture-based robot control. Second, by integrating vision, language, and motion planning modules, we introduce a shared autonomy framework that supports robust and user-driven teleoperation, with particular benefits for navigation-intensive tasks in home environments. Finally, to demonstrate the system in the wild, we present a twelve-day in-home user study evaluating real-time use of the wearable EMG interface for daily robot control. Together, these system components enable effective robot control for performing activities of daily living and other household tasks in a real home environment.
>
---
#### [new 027] Model-based Optimal Control for Rigid-Soft Underactuated Systems
- **分类: cs.RO**

- **简介: 该论文研究软-刚混合系统的动态控制问题，针对其欠驱动和变形复杂性，提出三种基于模型的最优控制方法，并在仿真中验证效果。**

- **链接: [https://arxiv.org/pdf/2602.03435v1](https://arxiv.org/pdf/2602.03435v1)**

> **作者:** Daniele Caradonna; Nikhil Nair; Anup Teejo Mathew; Daniel Feliu Talegón; Imran Afgan; Egidio Falotico; Cosimo Della Santina; Federico Renda
>
> **摘要:** Continuum soft robots are inherently underactuated and subject to intrinsic input constraints, making dynamic control particularly challenging, especially in hybrid rigid-soft robots. While most existing methods focus on quasi-static behaviors, dynamic tasks such as swing-up require accurate exploitation of continuum dynamics. This has led to studies on simple low-order template systems that often fail to capture the complexity of real continuum deformations. Model-based optimal control offers a systematic solution; however, its application to rigid-soft robots is often limited by the computational cost and inaccuracy of numerical differentiation for high-dimensional models. Building on recent advances in the Geometric Variable Strain model that enable analytical derivatives, this work investigates three optimal control strategies for underactuated soft systems-Direct Collocation, Differential Dynamic Programming, and Nonlinear Model Predictive Control-to perform dynamic swing-up tasks. To address stiff continuum dynamics and constrained actuation, implicit integration schemes and warm-start strategies are employed to improve numerical robustness and computational efficiency. The methods are evaluated in simulation on three Rigid-Soft and high-order soft benchmark systems-the Soft Cart-Pole, the Soft Pendubot, and the Soft Furuta Pendulum- highlighting their performance and computational trade-offs.
>
---
#### [new 028] A Unified Candidate Set with Scene-Adaptive Refinement via Diffusion for End-to-End Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶轨迹规划任务，解决传统候选集在复杂场景下表现不足的问题。提出CdDrive方法，结合固定候选与场景自适应生成候选，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2602.03112v1](https://arxiv.org/pdf/2602.03112v1)**

> **作者:** Zhengfei Wu; Shuaixi Pan; Shuohan Chen; Shuo Yang; Yanjun Huang
>
> **备注:** Code:https://github.com/WWW-TJ/CdDrive
>
> **摘要:** End-to-end autonomous driving is increasingly adopting a multimodal planning paradigm that generates multiple trajectory candidates and selects the final plan, making candidate-set design critical. A fixed trajectory vocabulary provides stable coverage in routine driving but often misses optimal solutions in complex interactions, while scene-adaptive refinement can cause over-correction in simple scenarios by unnecessarily perturbing already strong vocabulary trajectories.We propose CdDrive, which preserves the original vocabulary candidates and augments them with scene-adaptive candidates generated by vocabulary-conditioned diffusion denoising. Both candidate types are jointly scored by a shared selection module, enabling reliable performance across routine and highly interactive scenarios. We further introduce HATNA (Horizon-Aware Trajectory Noise Adapter) to improve the smoothness and geometric continuity of diffusion candidates via temporal smoothing and horizon-aware noise modulation. Experiments on NAVSIM v1 and NAVSIM v2 demonstrate leading performance, and ablations verify the contribution of each component.
>
---
#### [new 029] Learning-based Initialization of Trajectory Optimization for Path-following Problems of Redundant Manipulators
- **分类: cs.RO**

- **简介: 该论文属于路径跟踪任务，解决冗余机械臂轨迹优化初始轨迹选择困难的问题。通过学习方法生成高质量初始轨迹，提升优化效率与可行性。**

- **链接: [https://arxiv.org/pdf/2602.03418v1](https://arxiv.org/pdf/2602.03418v1)**

> **作者:** Minsung Yoon; Mincheul Kang; Daehyung Park; Sung-Eui Yoon
>
> **备注:** Accepted to ICRA 2023. <a href="https://sgvr.kaist.ac.kr/~msyoon/papers/ICRA23_RLITG/" rel="external noopener nofollow" class="link-external link-https">Project Page</a>
>
> **摘要:** Trajectory optimization (TO) is an efficient tool to generate a redundant manipulator's joint trajectory following a 6-dimensional Cartesian path. The optimization performance largely depends on the quality of initial trajectories. However, the selection of a high-quality initial trajectory is non-trivial and requires a considerable time budget due to the extremely large space of the solution trajectories and the lack of prior knowledge about task constraints in configuration space. To alleviate the issue, we present a learning-based initial trajectory generation method that generates high-quality initial trajectories in a short time budget by adopting example-guided reinforcement learning. In addition, we suggest a null-space projected imitation reward to consider null-space constraints by efficiently learning kinematically feasible motion captured in expert demonstrations. Our statistical evaluation in simulation shows the improved optimality, efficiency, and applicability of TO when we plug in our method's output, compared with three other baselines. We also show the performance improvement and feasibility via real-world experiments with a seven-degree-of-freedom manipulator.
>
---
#### [new 030] ProAct: A Benchmark and Multimodal Framework for Structure-Aware Proactive Response
- **分类: cs.RO**

- **简介: 该论文提出ProAct-75基准和ProAct-Helper框架，解决 proactive agent 开发中资源不足的问题，通过结构感知的多模态方法提升任务执行效率与并行性。**

- **链接: [https://arxiv.org/pdf/2602.03430v1](https://arxiv.org/pdf/2602.03430v1)**

> **作者:** Xiaomeng Zhu; Fengming Zhu; Weijie Zhou; Ye Tian; Zhenlin Hu; Yufei Huang; Yuchun Guo; Xinyu Wu; Zhengyou Zhang; Fangzhen Lin; Xuantang Xiong
>
> **摘要:** While passive agents merely follow instructions, proactive agents align with higher-level objectives, such as assistance and safety by continuously monitoring the environment to determine when and how to act. However, developing proactive agents is hindered by the lack of specialized resources. To address this, we introduce ProAct-75, a benchmark designed to train and evaluate proactive agents across diverse domains, including assistance, maintenance, and safety monitoring. Spanning 75 tasks, our dataset features 91,581 step-level annotations enriched with explicit task graphs. These graphs encode step dependencies and parallel execution possibilities, providing the structural grounding necessary for complex decision-making. Building on this benchmark, we propose ProAct-Helper, a reference baseline powered by a Multimodal Large Language Model (MLLM) that grounds decision-making in state detection, and leveraging task graphs to enable entropy-driven heuristic search for action selection, allowing agents to execute parallel threads independently rather than mirroring the human's next step. Extensive experiments demonstrate that ProAct-Helper outperforms strong closed-source models, improving trigger detection mF1 by 6.21%, saving 0.25 more steps in online one-step decision, and increasing the rate of parallel actions by 15.58%.
>
---
#### [new 031] A thin and soft optical tactile sensor for highly sensitive object perception
- **分类: cs.RO; physics.app-ph; physics.optics**

- **简介: 该论文属于触觉传感任务，旨在解决传统光学触觉传感器体积大、刚性等问题。提出一种薄而柔软的光学触觉传感器，通过光斑变化实现高精度力检测和纹理识别。**

- **链接: [https://arxiv.org/pdf/2602.03248v1](https://arxiv.org/pdf/2602.03248v1)**

> **作者:** Yanchen Shen; Kohei Tsuji; Haruto Koizumi; Jiseon Hong; Tomoaki Niiyama; Hiroyuki Kuwabara; Hayato Ishida; Jun Hiramitsu; Mitsuhito Mase; Satoshi Sunada
>
> **摘要:** Tactile sensing is crucial in robotics and wearable devices for safe perception and interaction with the environment. Optical tactile sensors have emerged as promising solutions, as they are immune to electromagnetic interference and have high spatial resolution. However, existing optical approaches, particularly vision-based tactile sensors, rely on complex optical assemblies that involve lenses and cameras, resulting in bulky, rigid, and alignment-sensitive designs. In this study, we present a thin, compact, and soft optical tactile sensor featuring an alignment-free configuration. The soft optical sensor operates by capturing deformation-induced changes in speckle patterns generated within a soft silicone material, thereby enabling precise force measurements and texture recognition via machine learning. The experimental results show a root-mean-square error of 40 mN in the force measurement and a classification accuracy of 93.33% over nine classes of textured surfaces, including Mahjong tiles. The proposed speckle-based approach provides a compact, easily fabricated, and mechanically compliant platform that bridges optical sensing with flexible shape-adaptive architectures, thereby demonstrating its potential as a novel tactile-sensing paradigm for soft robotics and wearable haptic interfaces.
>
---
#### [new 032] Depth Completion in Unseen Field Robotics Environments Using Extremely Sparse Depth Measurements
- **分类: cs.RO**

- **简介: 该论文属于深度补全任务，解决野外机器人环境中深度感知不足的问题。通过合成数据训练模型，利用稀疏深度测量预测密集深度，实现实时部署。**

- **链接: [https://arxiv.org/pdf/2602.03209v1](https://arxiv.org/pdf/2602.03209v1)**

> **作者:** Marco Job; Thomas Stastny; Eleni Kelasidi; Roland Siegwart; Michael Pantic
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Autonomous field robots operating in unstructured environments require robust perception to ensure safe and reliable operations. Recent advances in monocular depth estimation have demonstrated the potential of low-cost cameras as depth sensors; however, their adoption in field robotics remains limited due to the absence of reliable scale cues, ambiguous or low-texture conditions, and the scarcity of large-scale datasets. To address these challenges, we propose a depth completion model that trains on synthetic data and uses extremely sparse measurements from depth sensors to predict dense metric depth in unseen field robotics environments. A synthetic dataset generation pipeline tailored to field robotics enables the creation of multiple realistic datasets for training purposes. This dataset generation approach utilizes textured 3D meshes from Structure from Motion and photorealistic rendering with novel viewpoint synthesis to simulate diverse field robotics scenarios. Our approach achieves an end-to-end latency of 53 ms per frame on a Nvidia Jetson AGX Orin, enabling real-time deployment on embedded platforms. Extensive evaluation demonstrates competitive performance across diverse real-world field robotics scenarios.
>
---
#### [new 033] Estimation of Ground Reaction Forces from Kinematic Data during Locomotion
- **分类: cs.RO**

- **简介: 该论文属于运动分析任务，旨在解决无力板系统估算地面反作用力的问题。通过运动捕捉数据估计并分解GRF，实现临床可用的步态分析。**

- **链接: [https://arxiv.org/pdf/2602.03177v1](https://arxiv.org/pdf/2602.03177v1)**

> **作者:** Gautami Golani; Dong Anh Khoa To; Ananda Sidarta; Arun-Kumar Kaliya-Perumal; Oliver Roberts; Lek Syn Lim; Jim Patton; Domenico Campolo
>
> **摘要:** Ground reaction forces (GRFs) provide fundamental insight into human gait mechanics and are widely used to assess joint loading, limb symmetry, balance control, and motor function. Despite their clinical relevance, the use of GRF remains underutilised in clinical workflows due to the practical limitations of force plate systems. In this work, we present a force-plate-free approach for estimating GRFs using only marker-based motion capture data. This kinematics only method to estimate and decompose GRF makes it well suited for widespread clinical depolyment. By using kinematics from sixteen body segments, we estimate the centre of mass (CoM) and compute GRFs, which are subsequently decomposed into individual components through a minimization-based approach. Through this framework, we can identify gait stance phases and provide access to clinically meaningful kinetic measures without a dedicated force plate system. Experimental results demonstrate the viability of CoM and GRF estimation based solely on kinematic data, supporting force-plate-free gait analysis.
>
---
#### [new 034] Moving On, Even When You're Broken: Fail-Active Trajectory Generation via Diffusion Policies Conditioned on Embodiment and Task
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人在执行任务时遭遇故障仍能继续操作的fail-active控制问题。提出DEFT方法，基于扩散策略生成轨迹，适应不同故障情况，提升任务完成率。**

- **链接: [https://arxiv.org/pdf/2602.02895v1](https://arxiv.org/pdf/2602.02895v1)**

> **作者:** Gilberto G. Briscoe-Martinez; Yaashia Gautam; Rahul Shetty; Anuj Pasricha; Marco M. Nicotra; Alessandro Roncone
>
> **备注:** To be published in the 2026 IEEE International Conference on Robotics & Automation
>
> **摘要:** Robot failure is detrimental and disruptive, often requiring human intervention to recover. Maintaining safe operation under impairment to achieve task completion, i.e. fail-active operation, is our target. Focusing on actuation failures, we introduce DEFT, a diffusion-based trajectory generator conditioned on the robot's current embodiment and task constraints. DEFT generalizes across failure types, supports constrained and unconstrained motions, and enables task completion under arbitrary failure. We evaluated DEFT in both simulation and real-world scenarios using a 7-DoF robotic arm. In simulation over thousands of joint-failure cases across multiple tasks, DEFT outperformed the baseline by up to 2 times. On failures unseen during training, it continued to outperform the baseline, indicating robust generalization in simulation. Further, we performed real-world evaluations on two multi-step tasks, drawer manipulation and whiteboard erasing. These experiments demonstrated DEFT succeeding on tasks where classical methods failed. Our results show that DEFT achieves fail-active manipulation across arbitrary failure configurations and real-world deployments.
>
---
#### [new 035] Self-supervised Physics-Informed Manipulation of Deformable Linear Objects with Non-negligible Dynamics
- **分类: cs.RO**

- **简介: 该论文属于动态操控任务，解决变形线性物体的控制问题。提出SPiD框架，结合物理模型与自监督学习，实现高效、鲁棒的操控。**

- **链接: [https://arxiv.org/pdf/2602.03623v1](https://arxiv.org/pdf/2602.03623v1)**

> **作者:** Youyuan Long; Gokhan Solak; Sara Zeynalpour; Heng Zhang; Arash Ajoudani
>
> **备注:** Submitted to IEEE Transactions on Robotics. Video: https://youtu.be/lgX2J-00TRM
>
> **摘要:** We address dynamic manipulation of deformable linear objects by presenting SPiD, a physics-informed self-supervised learning framework that couples an accurate deformable object model with an augmented self-supervised training strategy. On the modeling side, we extend a mass-spring model to more accurately capture object dynamics while remaining lightweight enough for high-throughput rollouts during self-supervised learning. On the learning side, we train a neural controller using a task-oriented cost, enabling end-to-end optimization through interaction with the differentiable object model. In addition, we propose a self-supervised DAgger variant that detects distribution shift during deployment and performs offline self-correction to further enhance robustness without expert supervision. We evaluate our method primarily on the rope stabilization task, where a robot must bring a swinging rope to rest as quickly and smoothly as possible. Extensive experiments in both simulation and the real world demonstrate that the proposed controller achieves fast and smooth rope stabilization, generalizing across unseen initial states, rope lengths, masses, non-uniform mass distributions, and external disturbances. Additionally, we develop an affordable markerless rope perception method and demonstrate that our controller maintains performance with noisy and low-frequency state updates. Furthermore, we demonstrate the generality of the framework by extending it to the rope trajectory tracking task. Overall, SPiD offers a data-efficient, robust, and physically grounded framework for dynamic manipulation of deformable linear objects, featuring strong sim-to-real generalization.
>
---
#### [new 036] BridgeV2W: Bridging Video Generation Models to Embodied World Models via Embodiment Masks
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉与生成模型任务，解决动作与视频对齐、视角敏感及架构不统一问题，通过引入Embodiment Masks和运动损失提升视频生成质量。**

- **链接: [https://arxiv.org/pdf/2602.03793v1](https://arxiv.org/pdf/2602.03793v1)**

> **作者:** Yixiang Chen; Peiyan Li; Jiabing Yang; Keji He; Xiangnan Wu; Yuan Xu; Kai Wang; Jing Liu; Nianfeng Liu; Yan Huang; Liang Wang
>
> **摘要:** Embodied world models have emerged as a promising paradigm in robotics, most of which leverage large-scale Internet videos or pretrained video generation models to enrich visual and motion priors. However, they still face key challenges: a misalignment between coordinate-space actions and pixel-space videos, sensitivity to camera viewpoint, and non-unified architectures across embodiments. To this end, we present BridgeV2W, which converts coordinate-space actions into pixel-aligned embodiment masks rendered from the URDF and camera parameters. These masks are then injected into a pretrained video generation model via a ControlNet-style pathway, which aligns the action control signals with predicted videos, adds view-specific conditioning to accommodate camera viewpoints, and yields a unified world model architecture across embodiments. To mitigate overfitting to static backgrounds, BridgeV2W further introduces a flow-based motion loss that focuses on learning dynamic and task-relevant regions. Experiments on single-arm (DROID) and dual-arm (AgiBot-G1) datasets, covering diverse and challenging conditions with unseen viewpoints and scenes, show that BridgeV2W improves video generation quality compared to prior state-of-the-art methods. We further demonstrate the potential of BridgeV2W on downstream real-world tasks, including policy evaluation and goal-conditioned planning. More results can be found on our project website at https://BridgeV2W.github.io .
>
---
#### [new 037] Variance-Reduced Model Predictive Path Integral via Quadratic Model Approximation
- **分类: cs.RO**

- **简介: 该论文提出一种改进的MPPI方法，用于降低采样方差和提高样本效率。通过引入二次模型近似，分解目标函数并生成更有效的采样分布，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2602.03639v1](https://arxiv.org/pdf/2602.03639v1)**

> **作者:** Fabian Schramm; Franki Nguimatsia Tiofack; Nicolas Perrin-Gilbert; Marc Toussaint; Justin Carpentier
>
> **摘要:** Sampling-based controllers, such as Model Predictive Path Integral (MPPI) methods, offer substantial flexibility but often suffer from high variance and low sample efficiency. To address these challenges, we introduce a hybrid variance-reduced MPPI framework that integrates a prior model into the sampling process. Our key insight is to decompose the objective function into a known approximate model and a residual term. Since the residual captures only the discrepancy between the model and the objective, it typically exhibits a smaller magnitude and lower variance than the original objective. Although this principle applies to general modeling choices, we demonstrate that adopting a quadratic approximation enables the derivation of a closed-form, model-guided prior that effectively concentrates samples in informative regions. Crucially, the framework is agnostic to the source of geometric information, allowing the quadratic model to be constructed from exact derivatives, structural approximations (e.g., Gauss- or Quasi-Newton), or gradient-free randomized smoothing. We validate the approach on standard optimization benchmarks, a nonlinear, underactuated cart-pole control task, and a contact-rich manipulation problem with non-smooth dynamics. Across these domains, we achieve faster convergence and superior performance in low-sample regimes compared to standard MPPI. These results suggest that the method can make sample-based control strategies more practical in scenarios where obtaining samples is expensive or limited.
>
---
#### [new 038] IMAGINE: Intelligent Multi-Agent Godot-based Indoor Networked Exploration
- **分类: cs.RO; cs.LG; cs.MA; cs.NI; eess.SY**

- **简介: 该论文属于多智能体协作探索任务，解决GNSS拒止环境下的自主协同问题。通过MARL和Godot仿真，实现高效室内探索与决策。**

- **链接: [https://arxiv.org/pdf/2602.02858v1](https://arxiv.org/pdf/2602.02858v1)**

> **作者:** Tiago Leite; Maria Conceição; António Grilo
>
> **备注:** 12 pages, submitted to a journal
>
> **摘要:** The exploration of unknown, Global Navigation Satellite System (GNSS) denied environments by an autonomous communication-aware and collaborative group of Unmanned Aerial Vehicles (UAVs) presents significant challenges in coordination, perception, and decentralized decision-making. This paper implements Multi-Agent Reinforcement Learning (MARL) to address these challenges in a 2D indoor environment, using high-fidelity game-engine simulations (Godot) and continuous action spaces. Policy training aims to achieve emergent collaborative behaviours and decision-making under uncertainty using Network-Distributed Partially Observable Markov Decision Processes (ND-POMDPs). Each UAV is equipped with a Light Detection and Ranging (LiDAR) sensor and can share data (sensor measurements and a local occupancy map) with neighbouring agents. Inter-agent communication constraints include limited range, bandwidth and latency. Extensive ablation studies evaluated MARL training paradigms, reward function, communication system, neural network (NN) architecture, memory mechanisms, and POMDP formulations. This work jointly addresses several key limitations in prior research, namely reliance on discrete actions, single-agent or centralized formulations, assumptions of a priori knowledge and permanent connectivity, inability to handle dynamic obstacles, short planning horizons and architectural complexity in Recurrent NNs/Transformers. Results show that the scalable training paradigm, combined with a simplified architecture, enables rapid autonomous exploration of an indoor area. The implementation of Curriculum-Learning (five increasingly complex levels) also enabled faster, more robust training. This combination of high-fidelity simulation, MARL formulation, and computational efficiency establishes a strong foundation for deploying learned cooperative strategies in physical robotic systems.
>
---
#### [new 039] Accelerating Structured Chain-of-Thought in Autonomous Vehicles
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决CoT推理延迟问题。提出FastDriveCoT方法，通过并行解码加速结构化CoT生成，减少计算步骤，提升实时性。**

- **链接: [https://arxiv.org/pdf/2602.02864v1](https://arxiv.org/pdf/2602.02864v1)**

> **作者:** Yi Gu; Yan Wang; Yuxiao Chen; Yurong You; Wenjie Luo; Yue Wang; Wenhao Ding; Boyi Li; Heng Yang; Boris Ivanovic; Marco Pavone
>
> **摘要:** Chain-of-Thought (CoT) reasoning enhances the decision-making capabilities of vision-language-action models in autonomous driving, but its autoregressive nature introduces significant inference latency, making it impractical for real-time applications. To address this, we introduce FastDriveCoT, a novel parallel decoding method that accelerates template-structured CoT. Our approach decomposes the reasoning process into a dependency graph of distinct sub-tasks, such as identifying critical objects and summarizing traffic rules, some of which can be generated in parallel. By generating multiple independent reasoning steps concurrently within a single forward pass, we significantly reduce the number of sequential computations. Experiments demonstrate a 3-4$\times$ speedup in CoT generation and a substantial reduction in end-to-end latency across various model architectures, all while preserving the original downstream task improvements brought by incorporating CoT reasoning.
>
---
#### [new 040] PlanTRansformer: Unified Prediction and Planning with Goal-conditioned Transformer
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶中的轨迹预测与规划任务，解决预测与规划脱节问题。提出PlanTRansformer，统一预测与规划，提升准确性与可行性。**

- **链接: [https://arxiv.org/pdf/2602.03376v1](https://arxiv.org/pdf/2602.03376v1)**

> **作者:** Constantin Selzer; Fabina B. Flohr
>
> **备注:** Submitted and accepted at IEEE IV 2026
>
> **摘要:** Trajectory prediction and planning are fundamental yet disconnected components in autonomous driving. Prediction models forecast surrounding agent motion under unknown intentions, producing multimodal distributions, while planning assumes known ego objectives and generates deterministic trajectories. This mismatch creates a critical bottleneck: prediction lacks supervision for agent intentions, while planning requires this information. Existing prediction models, despite strong benchmarking performance, often remain disconnected from planning constraints such as collision avoidance and dynamic feasibility. We introduce Plan TRansformer (PTR), a unified Gaussian Mixture Transformer framework integrating goal-conditioned prediction, dynamic feasibility, interaction awareness, and lane-level topology reasoning. A teacher-student training strategy progressively masks surrounding agent commands during training to align with inference conditions where agent intentions are unavailable. PTR achieves 4.3%/3.5% improvement in marginal/joint mAP compared to the baseline Motion Transformer (MTR) and 15.5% planning error reduction at 5s horizon compared to GameFormer. The architecture-agnostic design enables application to diverse Transformer-based prediction models. Project Website: https://github.com/SelzerConst/PlanTRansformer
>
---
#### [new 041] HMVLA: Hyperbolic Multimodal Fusion for Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决多模态语义对齐问题。提出HMVLA框架，通过超球空间嵌入和稀疏专家机制提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2602.02533v1](https://arxiv.org/pdf/2602.02533v1)**

> **作者:** Kun Wang; Xiao Feng; Mingcheng Qu; Tonghua Su
>
> **备注:** 5 pages,5 figures,ICASSP
>
> **摘要:** Vision Language Action (VLA) models have recently shown great potential in bridging multimodal perception with robotic control. However, existing methods often rely on direct fine-tuning of pre-trained Vision-Language Models (VLMs), feeding semantic and visual features directly into a policy network without fully addressing the unique semantic alignment challenges in the VLA domain. In this paper, we propose HMVLA, a novel VLA framework that exploits the inherent hierarchical structures in vision and language for comprehensive semantic alignment. Unlike traditional methods that perform alignment in Euclidean space, our HMVLA embeds multimodal features in hyperbolic space, enabling more effective modeling of the hierarchical relationships present in image text data. Furthermore, we introduce a sparsely gated Mixture of Experts (MoE) mechanism tailored for semantic alignment, which enhances multimodal comprehension between images and text while improving efficiency. Extensive experiments demonstrate that HMVLA surpasses baseline methods in both accuracy and generalization. In addition, we validate its robustness by reconstructing datasets to further test cross domain adaptability.
>
---
#### [new 042] AROLA: A Modular Layered Architecture for Scaled Autonomous Racing
- **分类: cs.RO; cs.SE**

- **简介: 该论文提出AROLA架构，解决自动驾驶赛车软件模块化与评估问题，通过分层设计和标准化接口提升开发效率与可重复性。**

- **链接: [https://arxiv.org/pdf/2602.02730v1](https://arxiv.org/pdf/2602.02730v1)**

> **作者:** Fam Shihata; Mohammed Abdelazim; Ahmed Hussein
>
> **备注:** 6 pages, 6 figures, IV 2026
>
> **摘要:** Autonomous racing has advanced rapidly, particularly on scaled platforms, and software stacks must evolve accordingly. In this work, AROLA is introduced as a modular, layered software architecture in which fragmented and monolithic designs are reorganized into interchangeable layers and components connected through standardized ROS 2 interfaces. The autonomous-driving pipeline is decomposed into sensing, pre-processing, perception, localization and mapping, planning, behavior, control, and actuation, enabling rapid module replacement and objective benchmarking without reliance on custom message definitions. To support consistent performance evaluation, a Race Monitor framework is introduced as a lightweight system through which lap timing, trajectory quality, and computational load are logged in real time and standardized post-race analyses are generated. AROLA is validated in simulation and on hardware using the RoboRacer platform, including deployment at the 2025 RoboRacer IV25 competition. Together, AROLA and Race Monitor demonstrate that modularity, transparent interfaces, and systematic evaluation can accelerate development and improve reproducibility in scaled autonomous racing.
>
---
#### [new 043] Training and Simulation of Quadrupedal Robot in Adaptive Stair Climbing for Indoor Firefighting: An End-to-End Reinforcement Learning Approach
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于室内消防救援任务，旨在解决四足机器人在复杂楼梯环境中的导航与攀爬问题。通过两阶段强化学习方法，提升机器人适应不同楼梯结构的能力。**

- **链接: [https://arxiv.org/pdf/2602.03087v1](https://arxiv.org/pdf/2602.03087v1)**

> **作者:** Baixiao Huang; Baiyu Huang; Yu Hou
>
> **备注:** 8 pages, 9 figures, 43rd International Symposium on Automation and Robotics in Construction
>
> **摘要:** Quadruped robots are used for primary searches during the early stages of indoor fires. A typical primary search involves quickly and thoroughly looking for victims under hazardous conditions and monitoring flammable materials. However, situational awareness in complex indoor environments and rapid stair climbing across different staircases remain the main challenges for robot-assisted primary searches. In this project, we designed a two-stage end-to-end deep reinforcement learning (RL) approach to optimize both navigation and locomotion. In the first stage, the quadrupeds, Unitree Go2, were trained to climb stairs in Isaac Lab's pyramid-stair terrain. In the second stage, the quadrupeds were trained to climb various realistic indoor staircases in the Isaac Lab engine, with the learned policy transferred from the previous stage. These indoor staircases are straight, L-shaped, and spiral, to support climbing tasks in complex environments. This project explores how to balance navigation and locomotion and how end-to-end RL methods can enable quadrupeds to adapt to different stair shapes. Our main contributions are: (1) A two-stage end-to-end RL framework that transfers stair-climbing skills from abstract pyramid terrain to realistic indoor stair topologies. (2) A centerline-based navigation formulation that enables unified learning of navigation and locomotion without hierarchical planning. (3) Demonstration of policy generalization across diverse staircases using only local height-map perception. (4) An empirical analysis of success, efficiency, and failure modes under increasing stair difficulty.
>
---
#### [new 044] Kino-PAX$^+$: Near-Optimal Massively Parallel Kinodynamic Sampling-based Motion Planner
- **分类: cs.RO; cs.DC**

- **简介: 该论文属于机器人运动规划任务，解决高维空间中实时、近优解的运动规划问题。提出Kino-PAX⁺，通过并行化实现快速且近最优的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2602.02846v1](https://arxiv.org/pdf/2602.02846v1)**

> **作者:** Nicolas Perrault; Qi Heng Ho; Morteza Lahijanian
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** Sampling-based motion planners (SBMPs) are widely used for robot motion planning with complex kinodynamic constraints in high-dimensional spaces, yet they struggle to achieve \emph{real-time} performance due to their serial computation design. Recent efforts to parallelize SBMPs have achieved significant speedups in finding feasible solutions; however, they provide no guarantees of optimizing an objective function. We introduce Kino-PAX$^{+}$, a massively parallel kinodynamic SBMP with asymptotic near-optimal guarantees. Kino-PAX$^{+}$ builds a sparse tree of dynamically feasible trajectories by decomposing traditionally serial operations into three massively parallel subroutines. The algorithm focuses computation on the most promising nodes within local neighborhoods for propagation and refinement, enabling rapid improvement of solution cost. We prove that, while maintaining probabilistic $δ$-robust completeness, this focus on promising nodes ensures asymptotic $δ$-robust near-optimality. Our results show that Kino-PAX$^{+}$ finds solutions up to three orders of magnitude faster than existing serial methods and achieves lower solution costs than a state-of-the-art GPU-based planner.
>
---
#### [new 045] MVP-LAM: Learning Action-Centric Latent Action via Cross-Viewpoint Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MVP-LAM模型，用于从多视角视频中学习具有行动信息的潜在动作，解决无标签情况下动作表示的问题，提升视觉-语言-动作预训练效果。**

- **链接: [https://arxiv.org/pdf/2602.03668v1](https://arxiv.org/pdf/2602.03668v1)**

> **作者:** Jung Min Lee; Dohyeok Lee; Seokhun Ju; Taehyun Cho; Jin Woo Koo; Li Zhao; Sangwoo Hong; Jungwoo Lee
>
> **摘要:** Learning \emph{latent actions} from diverse human videos enables scaling robot learning beyond embodiment-specific robot datasets, and these latent actions have recently been used as pseudo-action labels for vision-language-action (VLA) model pretraining. To make VLA pretraining effective, latent actions should contain information about the underlying agent's actions despite the absence of ground-truth labels. We propose \textbf{M}ulti-\textbf{V}iew\textbf{P}oint \textbf{L}atent \textbf{A}ction \textbf{M}odel (\textbf{MVP-LAM}), which learns discrete latent actions that are highly informative about ground-truth actions from time-synchronized multi-view videos. MVP-LAM trains latent actions with a \emph{cross-viewpoint reconstruction} objective, so that a latent action inferred from one view must explain the future in another view, reducing reliance on viewpoint-specific cues. On Bridge V2, MVP-LAM produces more action-centric latent actions, achieving higher mutual information with ground-truth actions and improved action prediction, including under out-of-distribution evaluation. Finally, pretraining VLAs with MVP-LAM latent actions improves downstream manipulation performance on the SIMPLER and LIBERO-Long benchmarks.
>
---
#### [new 046] Conformal Reachability for Safe Control in Unknown Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于安全控制任务，解决未知环境中系统安全性验证问题。通过结合置信预测与可达性分析，构建概率验证框架，确保控制策略的安全性与性能。**

- **链接: [https://arxiv.org/pdf/2602.03799v1](https://arxiv.org/pdf/2602.03799v1)**

> **作者:** Xinhang Ma; Junlin Wu; Yiannis Kantaros; Yevgeniy Vorobeychik
>
> **摘要:** Designing provably safe control is a core problem in trustworthy autonomy. However, most prior work in this regard assumes either that the system dynamics are known or deterministic, or that the state and action space are finite, significantly limiting application scope. We address this limitation by developing a probabilistic verification framework for unknown dynamical systems which combines conformal prediction with reachability analysis. In particular, we use conformal prediction to obtain valid uncertainty intervals for the unknown dynamics at each time step, with reachability then verifying whether safety is maintained within the conformal uncertainty bounds. Next, we develop an algorithmic approach for training control policies that optimize nominal reward while also maximizing the planning horizon with sound probabilistic safety guarantees. We evaluate the proposed approach in seven safe control settings spanning four domains -- cartpole, lane following, drone control, and safe navigation -- for both affine and nonlinear safety specifications. Our experiments show that the policies we learn achieve the strongest provable safety guarantees while still maintaining high average reward.
>
---
#### [new 047] Deep-Learning-Based Control of a Decoupled Two-Segment Continuum Robot for Endoscopic Submucosal Dissection
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人控制任务，旨在解决内镜下黏膜剥离术（ESD）中器械灵活性不足的问题。研究开发了双段解耦机器人DESectBot，并采用GRU控制器提升操作精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.03406v1](https://arxiv.org/pdf/2602.03406v1)**

> **作者:** Yuancheng Shao; Yao Zhang; Jia Gu; Zixi Chen; Di Wu; Yuqiao Chen; Bo Lu; Wenjie Liu; Cesare Stefanini; Peng Qi
>
> **摘要:** Manual endoscopic submucosal dissection (ESD) is technically demanding, and existing single-segment robotic tools offer limited dexterity. These limitations motivate the development of more advanced solutions. To address this, DESectBot, a novel dual segment continuum robot with a decoupled structure and integrated surgical forceps, enabling 6 degrees of freedom (DoFs) tip dexterity for improved lesion targeting in ESD, was developed in this work. Deep learning controllers based on gated recurrent units (GRUs) for simultaneous tip position and orientation control, effectively handling the nonlinear coupling between continuum segments, were proposed. The GRU controller was benchmarked against Jacobian based inverse kinematics, model predictive control (MPC), a feedforward neural network (FNN), and a long short-term memory (LSTM) network. In nested-rectangle and Lissajous trajectory tracking tasks, the GRU achieved the lowest position/orientation RMSEs: 1.11 mm/ 4.62° and 0.81 mm/ 2.59°, respectively. For orientation control at a fixed position (four target poses), the GRU attained a mean RMSE of 0.14 mm and 0.72°, outperforming all alternatives. In a peg transfer task, the GRU achieved a 100% success rate (120 success/120 attempts) with an average transfer time of 11.8s, the STD significantly outperforms novice-controlled systems. Additionally, an ex vivo ESD demonstration grasping, elevating, and resecting tissue as the scalpel completed the cut confirmed that DESectBot provides sufficient stiffness to divide thick gastric mucosa and an operative workspace adequate for large lesions.These results confirm that GRU-based control significantly enhances precision, reliability, and usability in ESD surgical training scenarios.
>
---
#### [new 048] Formal Evidence Generation for Assurance Cases for Robotic Software Models
- **分类: cs.SE; cs.FL; cs.LO; cs.RO**

- **简介: 该论文属于形式化验证任务，旨在解决机器人软件安全论证证据生成困难的问题。通过模型驱动方法，自动将需求转化为形式化断言，并集成验证结果作为证据。**

- **链接: [https://arxiv.org/pdf/2602.03550v1](https://arxiv.org/pdf/2602.03550v1)**

> **作者:** Fang Yan; Simon Foster; Ana Cavalcanti; Ibrahim Habli; James Baxter
>
> **备注:** This is a preprint. The paper is currently under review at Software and Systems Modeling
>
> **摘要:** Robotics and Autonomous Systems are increasingly deployed in safety-critical domains, so that demonstrating their safety is essential. Assurance Cases (ACs) provide structured arguments supported by evidence, but generating and maintaining this evidence is labour-intensive, error-prone, and difficult to keep consistent as systems evolve. We present a model-based approach to systematically generating AC evidence by embedding formal verification into the assurance workflow. The approach addresses three challenges: systematically deriving formal assertions from natural language requirements using templates, orchestrating multiple formal verification tools to handle diverse property types, and integrating formal evidence production into the workflow. Leveraging RoboChart, a domain-specific modelling language with formal semantics, we combine model checking and theorem proving in our approach. Structured requirements are automatically transformed into formal assertions using predefined templates, and verification results are automatically integrated as evidence. Case studies demonstrate the effectiveness of our approach.
>
---
#### [new 049] When Should Agents Coordinate in Differentiable Sequential Decision Problems?
- **分类: cs.MA; cs.GT; cs.RO; math.OC**

- **简介: 该论文属于多智能体协作任务，研究在不同梯度规划问题中何时需要协调。解决的问题是协调的时机与成本平衡，通过分析目标的二阶性质提出协调算法。**

- **链接: [https://arxiv.org/pdf/2602.03674v1](https://arxiv.org/pdf/2602.03674v1)**

> **作者:** Caleb Probine; Su Ann Low; David Fridovich-Keil; Ufuk Topcu
>
> **备注:** 15 content pages, 2 pages for references, 4 figures
>
> **摘要:** Multi-robot teams must coordinate to operate effectively. When a team operates in an uncoordinated manner, and agents choose actions that are only individually optimal, the team's outcome can suffer. However, in many domains, coordination requires costly communication. We explore the value of coordination in a broad class of differentiable motion-planning problems. In particular, we model coordinated behavior as a spectrum: at one extreme, agents jointly optimize a common team objective, and at the other, agents make unilaterally optimal decisions given their individual decision variables, i.e., they operate at Nash equilibria. We then demonstrate that reasoning about coordination in differentiable motion-planning problems reduces to reasoning about the second-order properties of agents' objectives, and we provide algorithms that use this second-order reasoning to determine at which times a team of agents should coordinate.
>
---
#### [new 050] CRL-VLA: Continual Vision-Language-Action Learning
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于持续学习任务，解决VLA模型在终身机器人场景中平衡稳定性与可塑性的难题。提出CRL-VLA框架，通过双评论器结构实现有效持续训练。**

- **链接: [https://arxiv.org/pdf/2602.03445v1](https://arxiv.org/pdf/2602.03445v1)**

> **作者:** Qixin Zeng; Shuo Zhang; Hongyin Zhang; Renjie Wang; Han Zhao; Libang Zhao; Runze Li; Donglin Wang; Chao Huang
>
> **摘要:** Lifelong learning is critical for embodied agents in open-world environments, where reinforcement learning fine-tuning has emerged as an important paradigm to enable Vision-Language-Action (VLA) models to master dexterous manipulation through environmental interaction. Thus, Continual Reinforcement Learning (CRL) is a promising pathway for deploying VLA models in lifelong robotic scenarios, yet balancing stability (retaining old skills) and plasticity (learning new ones) remains a formidable challenge for existing methods. We introduce CRL-VLA, a framework for continual post-training of VLA models with rigorous theoretical bounds. We derive a unified performance bound linking the stability-plasticity trade-off to goal-conditioned advantage magnitude, scaled by policy divergence. CRL-VLA resolves this dilemma via asymmetric regulation: constraining advantage magnitudes on prior tasks while enabling controlled growth on new tasks. This is realized through a simple but effective dual-critic architecture with novel Goal-Conditioned Value Formulation (GCVF), where a frozen critic anchors semantic consistency and a trainable estimator drives adaptation. Experiments on the LIBERO benchmark demonstrate that CRL-VLA effectively harmonizes these conflicting objectives, outperforming baselines in both anti-forgetting and forward adaptation.
>
---
#### [new 051] Input-to-State Safe Backstepping: Robust Safety-Critical Control with Unmatched Uncertainties
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于非线性控制任务，解决不匹配不确定性下的安全关键控制问题。提出一种基于输入到状态安全的反步方法，构建安全约束函数以保证系统安全。**

- **链接: [https://arxiv.org/pdf/2602.03691v1](https://arxiv.org/pdf/2602.03691v1)**

> **作者:** Max H. Cohen; Pio Ong; Aaron D. Ames
>
> **备注:** To appear at the 2026 American Control Conference
>
> **摘要:** Guaranteeing safety in the presence of unmatched disturbances -- uncertainties that cannot be directly canceled by the control input -- remains a key challenge in nonlinear control. This paper presents a constructive approach to safety-critical control of nonlinear systems with unmatched disturbances. We first present a generalization of the input-to-state safety (ISSf) framework for systems with these uncertainties using the recently developed notion of an Optimal Decay CBF, which provides more flexibility for satisfying the associated Lyapunov-like conditions for safety. From there, we outline a procedure for constructing ISSf-CBFs for two relevant classes of systems with unmatched uncertainties: i) strict-feedback systems; ii) dual-relative-degree systems, which are similar to differentially flat systems. Our theoretical results are illustrated via numerical simulations of an inverted pendulum and planar quadrotor.
>
---
#### [new 052] Causal Flow Q-Learning for Robust Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决离线RL中的混淆偏差问题。通过因果视角构建鲁棒目标，提升策略在存在隐式混淆数据下的性能。**

- **链接: [https://arxiv.org/pdf/2602.02847v1](https://arxiv.org/pdf/2602.02847v1)**

> **作者:** Mingxuan Li; Junzhe Zhang; Elias Bareinboim
>
> **摘要:** Expressive policies based on flow-matching have been successfully applied in reinforcement learning (RL) more recently due to their ability to model complex action distributions from offline data. These algorithms build on standard policy gradients, which assume that there is no unmeasured confounding in the data. However, this condition does not necessarily hold for pixel-based demonstrations when a mismatch exists between the demonstrator's and the learner's sensory capabilities, leading to implicit confounding biases in offline data. We address the challenge by investigating the problem of confounded observations in offline RL from a causal perspective. We develop a novel causal offline RL objective that optimizes policies' worst-case performance that may arise due to confounding biases. Based on this new objective, we introduce a practical implementation that learns expressive flow-matching policies from confounded demonstrations, employing a deep discriminator to assess the discrepancy between the target policy and the nominal behavioral policy. Experiments across 25 pixel-based tasks demonstrate that our proposed confounding-robust augmentation procedure achieves a success rate 120\% that of confounding-unaware, state-of-the-art offline RL methods.
>
---
#### [new 053] Ethical Asymmetry in Human-Robot Interaction - An Empirical Test of Sparrow's Hypothesis
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互伦理研究，旨在检验Sparrow的道德不对称假设。通过实验分析道德许可与美德评分的关系，发现结果呈对称性，未支持原假设。**

- **链接: [https://arxiv.org/pdf/2602.02745v1](https://arxiv.org/pdf/2602.02745v1)**

> **作者:** Minyi Wang; Christoph Bartneck; Michael-John Turp; David Kaber
>
> **备注:** 27 pages, 3 figures
>
> **摘要:** The ethics of human-robot interaction (HRI) have been discussed extensively based on three traditional frameworks: deontology, consequentialism, and virtue ethics. We conducted a mixed within/between experiment to investigate Sparrow's proposed ethical asymmetry hypothesis in human treatment of robots. The moral permissibility of action (MPA) was manipulated as a subject grouping variable, and virtue type (prudence, justice, courage, and temperance) was controlled as a within-subjects factor. We tested moral stimuli using an online questionnaire with Perceived Moral Permissibility of Action (PMPA) and Perceived Virtue Scores (PVS) as response measures. The PVS measure was based on an adaptation of the established Questionnaire on Cardinal Virtues (QCV), while the PMPA was based on Malle et al. [39] work. We found that the MPA significantly influenced the PMPA and perceived virtue scores. The best-fitting model to describe the relationship between PMPA and PVS was cubic, which is symmetrical in nature. Our study did not confirm Sparrow's asymmetry hypothesis. The adaptation of the QCV is expected to have utility for future studies, pending additional psychometric property assessments.
>
---
#### [new 054] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文属于长期目标条件强化学习任务，旨在解决多实体复杂环境中的长时序问题。通过分层结构与子目标生成，提升算法在稀疏奖励下的性能。**

- **链接: [https://arxiv.org/pdf/2602.02722v1](https://arxiv.org/pdf/2602.02722v1)**

> **作者:** Dan Haramati; Carl Qi; Tal Daniel; Amy Zhang; Aviv Tamar; George Konidaris
>
> **备注:** ICLR 2026
>
> **摘要:** We propose a hierarchical entity-centric framework for offline Goal-Conditioned Reinforcement Learning (GCRL) that combines subgoal decomposition with factored structure to solve long-horizon tasks in domains with multiple entities. Achieving long-horizon goals in complex environments remains a core challenge in Reinforcement Learning (RL). Domains with multiple entities are particularly difficult due to their combinatorial complexity. GCRL facilitates generalization across goals and the use of subgoal structure, but struggles with high-dimensional observations and combinatorial state-spaces, especially under sparse reward. We employ a two-level hierarchy composed of a value-based GCRL agent and a factored subgoal-generating conditional diffusion model. The RL agent and subgoal generator are trained independently and composed post hoc through selective subgoal generation based on the value function, making the approach modular and compatible with existing GCRL algorithms. We introduce new variations to benchmark tasks that highlight the challenges of multi-entity domains, and show that our method consistently boosts performance of the underlying RL agent on image-based long-horizon tasks with sparse rewards, achieving over 150% higher success rates on the hardest task in our suite and generalizing to increasing horizons and numbers of entities. Rollout videos are provided at: https://sites.google.com/view/hecrl
>
---
#### [new 055] LEVIO: Lightweight Embedded Visual Inertial Odometry for Resource-Constrained Devices
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于视觉惯性里程计任务，解决资源受限设备实时定位问题。提出LEVIO系统，优化计算效率与内存使用，实现在低功耗平台上的六自由度实时感知。**

- **链接: [https://arxiv.org/pdf/2602.03294v1](https://arxiv.org/pdf/2602.03294v1)**

> **作者:** Jonas Kühne; Christian Vogt; Michele Magno; Luca Benini
>
> **备注:** This article has been accepted for publication in the IEEE Sensors Journal (JSEN)
>
> **摘要:** Accurate, infrastructure-less sensor systems for motion tracking are essential for mobile robotics and augmented reality (AR) applications. The most popular state-of-the-art visual-inertial odometry (VIO) systems, however, are too computationally demanding for resource-constrained hardware, such as micro-drones and smart glasses. This work presents LEVIO, a fully featured VIO pipeline optimized for ultra-low-power compute platforms, allowing six-degrees-of-freedom (DoF) real-time sensing. LEVIO incorporates established VIO components such as Oriented FAST and Rotated BRIEF (ORB) feature tracking and bundle adjustment, while emphasizing a computationally efficient architecture with parallelization and low memory usage to suit embedded microcontrollers and low-power systems-on-chip (SoCs). The paper proposes and details the algorithmic design choices and the hardware-software co-optimization approach, and presents real-time performance on resource-constrained hardware. LEVIO is validated on a parallel-processing ultra-low-power RISC-V SoC, achieving 20 FPS while consuming less than 100 mW, and benchmarked against public VIO datasets, offering a compelling balance between efficiency and accuracy. To facilitate reproducibility and adoption, the complete implementation is released as open-source.
>
---
#### [new 056] Towards Considerate Embodied AI: Co-Designing Situated Multi-Site Healthcare Robots from Abstract Concepts to High-Fidelity Prototypes
- **分类: cs.HC; cs.AI; cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决医疗场景中AI系统设计的落地问题。通过14周多学科协作，从抽象概念到高保真原型迭代，提出八项指导原则以提升AI系统的实用性与适应性。**

- **链接: [https://arxiv.org/pdf/2602.03054v1](https://arxiv.org/pdf/2602.03054v1)**

> **作者:** Yuanchen Bai; Ruixiang Han; Niti Parikh; Wendy Ju; Angelique Taylor
>
> **备注:** To appear in Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI 2026)
>
> **摘要:** Co-design is essential for grounding embodied artificial intelligence (AI) systems in real-world contexts, especially high-stakes domains such as healthcare. While prior work has explored multidisciplinary collaboration, iterative prototyping, and support for non-technical participants, few have interwoven these into a sustained co-design process. Such efforts often target one context and low-fidelity stages, limiting the generalizability of findings and obscuring how participants' ideas evolve. To address these limitations, we conducted a 14-week workshop with a multidisciplinary team of 22 participants, centered around how embodied AI can reduce non-value-added task burdens in three healthcare settings: emergency departments, long-term rehabilitation facilities, and sleep disorder clinics. We found that the iterative progression from abstract brainstorming to high-fidelity prototypes, supported by educational scaffolds, enabled participants to understand real-world trade-offs and generate more deployable solutions. We propose eight guidelines for co-designing more considerate embodied AI: attuned to context, responsive to social dynamics, mindful of expectations, and grounded in deployment. Project Page: https://byc-sophie.github.io/Towards-Considerate-Embodied-AI/
>
---
#### [new 057] QVLA: Not All Channels Are Equal in Vision-Language-Action Model's Quantization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于模型压缩任务，旨在解决VLA模型在机器人部署中的计算需求问题。提出QVLA框架，通过通道级量化提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.03782v1](https://arxiv.org/pdf/2602.03782v1)**

> **作者:** Yuhao Xu; Yantai Yang; Zhenyang Fan; Yufan Liu; Yuming Li; Bing Li; Zhipeng Zhang
>
> **备注:** ICLR2026
>
> **摘要:** The advent of Vision-Language-Action (VLA) models represents a significant leap for embodied intelligence, yet their immense computational demands critically hinder deployment on resource-constrained robotic platforms. Intuitively, low-bit quantization is a prevalent and preferred technique for large-scale model compression. However, we find that a systematic analysis of VLA model's quantization is fundamentally lacking. We argue that naively applying uniform-bit quantization from Large Language Models (LLMs) to robotics is flawed, as these methods prioritize passive data fidelity while ignoring how minor action deviations compound into catastrophic task failures. To bridge this gap, we introduce QVLA, the first action-centric quantization framework specifically designed for embodied control. In a sharp departure from the rigid, uniform-bit quantization of LLM-based methods, QVLA introduces a highly granular, channel-wise bit allocation strategy. Its core mechanism is to directly measure the final action-space sensitivity when quantizing each individual channel to various bit-widths. This process yields a precise, per-channel importance metric that guides a global optimization, which elegantly unifies quantization and pruning (0-bit) into a single, cohesive framework. Extensive evaluations on different baselines demonstrate the superiority of our approach. In the LIBERO, the quantization version of OpenVLA-OFT with our method requires only 29.2% of the original model's VRAM while maintaining 98.9% of its original performance and achieving a 1.49x speedup. This translates to a 22.6% performance improvement over the LLM-derived method SmoothQuant. Our work establishes a new, principled foundation for compressing VLA models in robotics, paving the way for deploying powerful, large-scale models on real-world hardware. Code will be released.
>
---
#### [new 058] Formulating Reinforcement Learning for Human-Robot Collaboration through Off-Policy Evaluation
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决人机协作中的状态空间和奖励函数选择问题。通过离线策略评估自动优化设计，提升RL在真实环境中的可行性与效率。**

- **链接: [https://arxiv.org/pdf/2602.02530v1](https://arxiv.org/pdf/2602.02530v1)**

> **作者:** Saurav Singh; Rodney Sanchez; Alexander Ororbia; Jamison Heard
>
> **摘要:** Reinforcement learning (RL) has the potential to transform real-world decision-making systems by enabling autonomous agents to learn from experience. Deploying RL in real-world settings, especially in the context of human-robot interaction, requires defining state representations and reward functions, which are critical for learning efficiency and policy performance. Traditional RL approaches often rely on domain expertise and trial-and-error, necessitating extensive human involvement as well as direct interaction with the environment, which can be costly and impractical, especially in complex and safety-critical applications. This work proposes a novel RL framework that leverages off-policy evaluation (OPE) for state space and reward function selection, using only logged interaction data. This approach eliminates the need for real-time access to the environment or human-in-the-loop feedback, greatly reducing the dependency on costly real-time interactions. The proposed approach systematically evaluates multiple candidate state representations and reward functions by training offline RL agents and applying OPE to estimate policy performance. The optimal state space and reward function are selected based on their ability to produce high-performing policies under OPE metrics. Our method is validated on two environments: the Lunar Lander environment by OpenAI Gym, which provides a controlled setting for assessing state space and reward function selection, and a NASA-MATB-II human subjects study environment, which evaluates the approach's real-world applicability to human-robot teaming scenarios. This work enhances the feasibility and scalability of offline RL for real-world environments by automating critical RL design decisions through a data-driven OPE-based evaluation, enabling more reliable, effective, and sustainable RL formulation for complex human-robot interaction settings.
>
---
#### [new 059] Sub-optimality bounds for certainty equivalent policies in partially observed systems
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文研究部分可观测系统中确定性等价策略的次优性界限，属于控制理论领域。旨在解决非线性系统下策略次优性分析问题，通过推导上界进行量化评估。**

- **链接: [https://arxiv.org/pdf/2602.02814v1](https://arxiv.org/pdf/2602.02814v1)**

> **作者:** Berk Bozkurt; Aditya Mahajan; Ashutosh Nayyar; Yi Ouyang
>
> **备注:** 12 pages, 0 figures
>
> **摘要:** In this paper, we present a generalization of the certainty equivalence principle of stochastic control. One interpretation of the classical certainty equivalence principle for linear systems with output feedback and quadratic costs is as follows: the optimal action at each time is obtained by evaluating the optimal state-feedback policy of the stochastic linear system at the minimum mean square error (MMSE) estimate of the state. Motivated by this interpretation, we consider certainty equivalent policies for general (non-linear) partially observed stochastic systems that allow for any state estimate rather than restricting to MMSE estimates. In such settings, the certainty equivalent policy is not optimal. For models where the cost and the dynamics are smooth in an appropriate sense, we derive upper bounds on the sub-optimality of certainty equivalent policies. We present several examples to illustrate the results.
>
---
#### [new 060] Simulating Human Audiovisual Search Behavior
- **分类: cs.HC; cs.AI; cs.RO**

- **简介: 该论文属于音频视觉搜索任务，旨在解决如何模拟人类在不确定环境下高效搜索目标的问题。研究提出Sensonaut模型，通过资源理性决策优化搜索策略。**

- **链接: [https://arxiv.org/pdf/2602.02790v1](https://arxiv.org/pdf/2602.02790v1)**

> **作者:** Hyunsung Cho; Xuejing Luo; Byungjoo Lee; David Lindlbauer; Antti Oulasvirta
>
> **备注:** 17 pages, 10 figures, CHI 2026
>
> **摘要:** Locating a target based on auditory and visual cues$\unicode{x2013}$such as finding a car in a crowded parking lot or identifying a speaker in a virtual meeting$\unicode{x2013}$requires balancing effort, time, and accuracy under uncertainty. Existing models of audiovisual search often treat perception and action in isolation, overlooking how people adaptively coordinate movement and sensory strategies. We present Sensonaut, a computational model of embodied audiovisual search. The core assumption is that people deploy their body and sensory systems in ways they believe will most efficiently improve their chances of locating a target, trading off time and effort under perceptual constraints. Our model formulates this as a resource-rational decision-making problem under partial observability. We validate the model against newly collected human data, showing that it reproduces both adaptive scaling of search time and effort under task complexity, occlusion, and distraction, and characteristic human errors. Our simulation of human-like resource-rational search informs the design of audiovisual interfaces that minimize search cost and cognitive load.
>
---
#### [new 061] Fast Near Time-Optimal Motion Planning for Holonomic Vehicles in Structured Environments
- **分类: math.OC; cs.RO**

- **简介: 该论文属于运动规划任务，解决结构化环境中全向车辆的近似时间最优轨迹生成问题。通过优化方法，利用运动基元和自由空间走廊，提高计算效率。**

- **链接: [https://arxiv.org/pdf/2602.02826v1](https://arxiv.org/pdf/2602.02826v1)**

> **作者:** Louis Callens; Bastiaan Vandewal; Ibrahim Ibrahim; Jan Swevers; Wilm Decré
>
> **摘要:** This paper proposes a novel and efficient optimization-based method for generating near time-optimal trajectories for holonomic vehicles navigating through complex but structured environments. The approach aims to solve the problem of motion planning for planar motion systems using magnetic levitation that can be used in assembly lines, automated laboratories or clean-rooms. In these applications, time-optimal trajectories that can be computed in real-time are required to increase productivity and allow the vehicles to be reactive if needed. The presented approach encodes the environment representation using free-space corridors and represents the motion of the vehicle through such a corridor using a motion primitive. These primitives are selected heuristically and define the trajectory with a limited number of degrees of freedom, which are determined in an optimization problem. As a result, the method achieves significantly lower computation times compared to the state-of-the-art, most notably solving a full Optimal Control Problem (OCP), OMG-tools or VP-STO without significantly compromising optimality within a fixed corridor sequence. The approach is benchmarked extensively in simulation and is validated on a real-world Beckhoff XPlanar system
>
---
## 更新

#### [replaced 001] Compact LED-Based Displacement Sensing for Robot Fingers
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉传感任务，旨在解决机器人手指位移检测问题。通过LED传感器实现高灵敏度位移感知，具备低成本、易集成等优势。**

- **链接: [https://arxiv.org/pdf/2410.03481v4](https://arxiv.org/pdf/2410.03481v4)**

> **作者:** Amr El-Azizi; Sharfin Islam; Pedro Piacenza; Kai Jiang; Ioannis Kymissis; Matei Ciocarlie
>
> **摘要:** In this paper, we introduce a sensor designed for integration in robot fingers, where it can provide information on the displacements induced by external contact. Our sensor uses LEDs to sense the displacement between two plates connected by a transparent elastomer; when a force is applied to the finger, the elastomer displaces and the LED signals change. We show that using LEDs as both light emitters an receivers in this context provides high sensitivity, allowing such an emitter and receiver pairs to detect very small displacements. We characterize the standalone performance of the sensor by testing the ability of a supervised learning model to predict complete force and torque data from its raw signals, and obtain a mean error between 0.05 and 0.07 N across the three directions of force applied to the finger. Our method allows for finger-size packaging with no amplification electronics, low cost manufacturing, easy integration into a complete hand, and high overload shear forces and bending torques, suggesting future applicability to complete manipulation tasks.
>
---
#### [replaced 002] TreeLoc: 6-DoF LiDAR Global Localization in Forests via Inter-Tree Geometric Matching
- **分类: cs.RO; cs.CV**

- **简介: 论文提出TreeLoc，用于森林中基于LiDAR的全局定位任务。解决GPS失效、场景重复复杂的问题，通过树干特征进行位姿估计与场景识别。**

- **链接: [https://arxiv.org/pdf/2602.01501v2](https://arxiv.org/pdf/2602.01501v2)**

> **作者:** Minwoo Jung; Nived Chebrolu; Lucas Carvalho de Lima; Haedam Oh; Maurice Fallon; Ayoung Kim
>
> **备注:** An 8-page paper with 7 tables and 8 figures, accepted to ICRA 2026
>
> **摘要:** Reliable localization is crucial for navigation in forests, where GPS is often degraded and LiDAR measurements are repetitive, occluded, and structurally complex. These conditions weaken the assumptions of traditional urban-centric localization methods, which assume that consistent features arise from unique structural patterns, necessitating forest-centric solutions to achieve robustness in these environments. To address these challenges, we propose TreeLoc, a LiDAR-based global localization framework for forests that handles place recognition and 6-DoF pose estimation. We represent scenes using tree stems and their Diameter at Breast Height (DBH), which are aligned to a common reference frame via their axes and summarized using the tree distribution histogram (TDH) for coarse matching, followed by fine matching with a 2D triangle descriptor. Finally, pose estimation is achieved through a two-step geometric verification. On diverse forest benchmarks, TreeLoc outperforms baselines, achieving precise localization. Ablation studies validate the contribution of each component. We also propose applications for long-term forest management using descriptors from a compact global tree database. TreeLoc is open-sourced for the robotics community at https://github.com/minwoo0611/TreeLoc.
>
---
#### [replaced 003] Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL
- **分类: cs.RO; cs.LG; cs.NE; eess.SY**

- **简介: 该论文属于自主驾驶控制任务，旨在解决预训练策略在真实环境中的性能下降问题。通过实时循环强化学习（RTRRL）和液态电阻电容RNN模型，实现策略的在线微调，提升驾驶表现。**

- **链接: [https://arxiv.org/pdf/2602.02236v2](https://arxiv.org/pdf/2602.02236v2)**

> **作者:** Julian Lemmel; Felix Resch; Mónika Farsang; Ramin Hasani; Daniela Rus; Radu Grosu
>
> **摘要:** Deploying pretrained policies in real-world applications presents substantial challenges that fundamentally limit the practical applicability of learning-based control systems. When autonomous systems encounter environmental changes in system dynamics, sensor drift, or task objectives, fixed policies rapidly degrade in performance. We show that employing Real-Time Recurrent Reinforcement Learning (RTRRL), a biologically plausible algorithm for online adaptation, can effectively fine-tune a pretrained policy to improve autonomous agents' performance on driving tasks. We further show that RTRRL synergizes with a recent biologically inspired recurrent network model, the Liquid-Resistance Liquid-Capacitance RNN. We demonstrate the effectiveness of this closed-loop approach in a simulated CarRacing environment and in a real-world line-following task with a RoboRacer car equipped with an event camera.
>
---
#### [replaced 004] LP-MPPI: Low-Pass Filtering for Efficient Model Predictive Path Integral Control
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制任务，解决MPPI控制中高频率噪声问题。通过引入低通滤波，提升控制平滑性和效率。**

- **链接: [https://arxiv.org/pdf/2503.11717v3](https://arxiv.org/pdf/2503.11717v3)**

> **作者:** Piotr Kicki
>
> **备注:** Accepted at International Conference on Robotics and Automation 2026 (ICRA 2026)
>
> **摘要:** Model Predictive Path Integral (MPPI) control is a widely used sampling-based approach for real-time control, valued for its flexibility in handling arbitrary dynamics and cost functions. However, it often suffers from high-frequency noise in the sampled control trajectories, which hinders the search for optimal controls and transfers to the applied controls, leading to actuator wear. In this work, we introduce Low-Pass Model Predictive Path Integral Control (LP-MPPI), which integrates low-pass filtering into the sampling process to eliminate detrimental high-frequency components and enhance the algorithm's efficiency. Unlike prior approaches, LP-MPPI provides direct and interpretable control over the frequency spectrum of sampled control trajectory perturbations, leading to more efficient sampling and smoother control. Through extensive evaluations in Gymnasium environments, simulated quadruped locomotion, and real-world F1TENTH autonomous racing, we demonstrate that LP-MPPI consistently outperforms state-of-the-art MPPI variants, achieving significant performance improvements while reducing control signal chattering.
>
---
#### [replaced 005] DDP-WM: Disentangled Dynamics Prediction for Efficient World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DDP-WM，解决世界模型计算效率低的问题，通过分解动态预测提升性能，适用于机器人导航与操作任务。**

- **链接: [https://arxiv.org/pdf/2602.01780v2](https://arxiv.org/pdf/2602.01780v2)**

> **作者:** Shicheng Yin; Kaixuan Yin; Weixing Chen; Yang Liu; Guanbin Li; Liang Lin
>
> **备注:** Codes will be available at https://github.com/HCPLab-SYSU/DDP-WM
>
> **摘要:** World models are essential for autonomous robotic planning. However, the substantial computational overhead of existing dense Transformerbased models significantly hinders real-time deployment. To address this efficiency-performance bottleneck, we introduce DDP-WM, a novel world model centered on the principle of Disentangled Dynamics Prediction (DDP). We hypothesize that latent state evolution in observed scenes is heterogeneous and can be decomposed into sparse primary dynamics driven by physical interactions and secondary context-driven background updates. DDP-WM realizes this decomposition through an architecture that integrates efficient historical processing with dynamic localization to isolate primary dynamics. By employing a crossattention mechanism for background updates, the framework optimizes resource allocation and provides a smooth optimization landscape for planners. Extensive experiments demonstrate that DDP-WM achieves significant efficiency and performance across diverse tasks, including navigation, precise tabletop manipulation, and complex deformable or multi-body interactions. Specifically, on the challenging Push-T task, DDP-WM achieves an approximately 9 times inference speedup and improves the MPC success rate from 90% to98% compared to state-of-the-art dense models. The results establish a promising path for developing efficient, high-fidelity world models. Codes will be available at https://github.com/HCPLab-SYSU/DDP-WM.
>
---
#### [replaced 006] SyNeT: Synthetic Negatives for Traversability Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主导航中的可行驶性估计任务，旨在解决缺乏明确负样本导致模型识别非可行驶区域能力不足的问题。通过构建合成负样本提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.00814v2](https://arxiv.org/pdf/2602.00814v2)**

> **作者:** Bomena Kim; Hojun Lee; Younsoo Park; Yaoyu Hu; Sebastian Scherer; Inwook Shim
>
> **摘要:** Reliable traversability estimation is crucial for autonomous robots to navigate complex outdoor environments safely. Existing self-supervised learning frameworks primarily rely on positive and unlabeled data; however, the lack of explicit negative data remains a critical limitation, hindering the model's ability to accurately identify diverse non-traversable regions. To address this issue, we introduce a method to explicitly construct synthetic negatives, representing plausible but non-traversable, and integrate them into vision-based traversability learning. Our approach is formulated as a training strategy that can be seamlessly integrated into both Positive-Unlabeled (PU) and Positive-Negative (PN) frameworks without modifying inference architectures. Complementing standard pixel-wise metrics, we introduce an object-centric FPR evaluation approach that analyzes predictions in regions where synthetic negatives are inserted. This evaluation provides an indirect measure of the model's ability to consistently identify non-traversable regions without additional manual labeling. Extensive experiments on both public and self-collected datasets demonstrate that our approach significantly enhances robustness and generalization across diverse environments. The source code and demonstration videos will be publicly available.
>
---
#### [replaced 007] SEMNAV: Enhancing Visual Semantic Navigation in Robotics through Semantic Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语义导航任务，旨在解决模拟与真实环境间泛化能力不足的问题。通过引入语义分割提升导航性能，提出SEMNAV方法及相应数据集，实验证明其效果优于现有模型。**

- **链接: [https://arxiv.org/pdf/2506.01418v2](https://arxiv.org/pdf/2506.01418v2)**

> **作者:** Rafael Flor-Rodríguez; Carlos Gutiérrez-Álvarez; Francisco Javier Acevedo-Rodríguez; Sergio Lafuente-Arroyo; Roberto J. López-Sastre
>
> **摘要:** Visual Semantic Navigation (VSN) is a fundamental problem in robotics, where an agent must navigate toward a target object in an unknown environment, mainly using visual information. Most state-of-the-art VSN models are trained in simulation environments, where rendered scenes of the real world are used, at best. These approaches typically rely on raw RGB data from the virtual scenes, which limits their ability to generalize to real-world environments due to domain adaptation issues. To tackle this problem, in this work, we propose SEMNAV, a novel approach that leverages semantic segmentation as the main visual input representation of the environment to enhance the agent's perception and decision-making capabilities. By explicitly incorporating this type of high-level semantic information, our model learns robust navigation policies that improve generalization across unseen environments, both in simulated and real world settings. We also introduce the SEMNAV dataset, a newly curated dataset designed for training semantic segmentation-aware navigation models like SEMNAV. Our approach is evaluated extensively in both simulated environments and with real-world robotic platforms. Experimental results demonstrate that SEMNAV outperforms existing state-of-the-art VSN models, achieving higher success rates in the Habitat 2.0 simulation environment, using the HM3D dataset. Furthermore, our real-world experiments highlight the effectiveness of semantic segmentation in mitigating the sim-to-real gap, making our model a promising solution for practical VSN-based robotic applications. The code and datasets are accessible at https://github.com/gramuah/semnav
>
---
#### [replaced 008] Spiking Neural-Invariant Kalman Fusion for Accurate Localization Using Low-Cost IMUs
- **分类: cs.RO**

- **简介: 该论文属于移动机器人定位任务，解决低成本IMU噪声导致的定位精度问题。通过融合脉冲神经网络与不变扩展卡尔曼滤波，提升定位鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.08248v2](https://arxiv.org/pdf/2601.08248v2)**

> **作者:** Yaohua Liu; Qiao Xu; Binkai Ou
>
> **摘要:** Low-cost inertial measurement units (IMUs) are widely utilized in mobile robot localization due to their affordability and ease of integration. However, their complex, nonlinear, and time-varying noise characteristics often lead to significant degradation in localization accuracy when applied directly for dead reckoning. To overcome this limitation, we propose a novel brain-inspired state estimation framework that combines a spiking neural network (SNN) with an invariant extended Kalman filter (InEKF). The SNN is designed to extract motion-related features from long sequences of IMU data affected by substantial random noise and is trained via a surrogate gradient descent algorithm to enable dynamic adaptation of the covariance noise parameter within the InEKF. By fusing the SNN output with raw IMU measurements, the proposed method enhances the robustness and accuracy of pose estimation. Extensive experiments conducted on the KITTI dataset and real-world data collected using a mobile robot equipped with a low-cost IMU demonstrate that the proposed approach outperforms state-of-the-art methods in localization accuracy and exhibits strong robustness to sensor noise, highlighting its potential for real-world mobile robot applications.
>
---
#### [replaced 009] Geometry-aware 4D Video Generation for Robot Manipulation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于4D视频生成任务，旨在解决多视角下视频时空一致性和几何一致性问题。通过几何监督学习3D场景表示，生成从新视角出发的稳定视频序列。**

- **链接: [https://arxiv.org/pdf/2507.01099v2](https://arxiv.org/pdf/2507.01099v2)**

> **作者:** Zeyi Liu; Shuang Li; Eric Cousineau; Siyuan Feng; Benjamin Burchfiel; Shuran Song
>
> **备注:** ICLR 2026; Project website: https://robot4dgen.github.io
>
> **摘要:** Understanding and predicting dynamics of the physical world can enhance a robot's ability to plan and interact effectively in complex environments. While recent video generation models have shown strong potential in modeling dynamic scenes, generating videos that are both temporally coherent and geometrically consistent across camera views remains a significant challenge. To address this, we propose a 4D video generation model that enforces multi-view 3D consistency of generated videos by supervising the model with cross-view pointmap alignment during training. Through this geometric supervision, the model learns a shared 3D scene representation, enabling it to generate spatio-temporally aligned future video sequences from novel viewpoints given a single RGB-D image per view, and without relying on camera poses as input. Compared to existing baselines, our method produces more visually stable and spatially aligned predictions across multiple simulated and real-world robotic datasets. We further show that the predicted 4D videos can be used to recover robot end-effector trajectories using an off-the-shelf 6DoF pose tracker, yielding robot manipulation policies that generalize well to novel camera viewpoints.
>
---
#### [replaced 010] A Low-Cost Vision-Based Tactile Gripper with Pretraining Learning for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决传统触觉传感器的局限性。提出一种低成本视觉触觉夹爪LVTG，结合视觉与触觉反馈，提升抓取稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2602.00514v2](https://arxiv.org/pdf/2602.00514v2)**

> **作者:** Yaohua Liu; Binkai Ou; Zicheng Qiu; Ce Hao; Hengjun Zhang
>
> **摘要:** Robotic manipulation in contact-rich environments remains challenging, particularly when relying on conventional tactile sensors that suffer from limited sensing range, reliability, and cost-effectiveness. In this work, we present LVTG, a low-cost visuo-tactile gripper designed for stable, robust, and efficient physical interaction. Unlike existing visuo-tactile sensors, LVTG enables more effective and stable grasping of larger and heavier everyday objects, thanks to its enhanced tactile sensing area and greater opening angle. Its surface skin is made of highly wear-resistant material, significantly improving durability and extending operational lifespan. The integration of vision and tactile feedback allows LVTG to provide rich, high-fidelity sensory data, facilitating reliable perception during complex manipulation tasks. Furthermore, LVTG features a modular design that supports rapid maintenance and replacement. To effectively fuse vision and touch, We adopt a CLIP-inspired contrastive learning objective to align tactile embeddings with their corresponding visual observations, enabling a shared cross-modal representation space for visuo-tactile perception. This alignment improves the performance of an Action Chunking Transformer (ACT) policy in contact-rich manipulation, leading to more efficient data collection and more effective policy learning. Compared to the original ACT method, the proposed LVTG with pretraining achieves significantly higher success rates in manipulation tasks.
>
---
#### [replaced 011] Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation
- **分类: cs.RO**

- **简介: 该论文提出Self-CriTeach框架，解决机器人规划中领域生成与推理监督问题。通过LLM自动生成规划域，提升规划成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.21543v2](https://arxiv.org/pdf/2509.21543v2)**

> **作者:** Jinbang Huang; Zhiyuan Li; Yuanzhao Hu; Zhanguang Zhang; Mark Coates; Xingyue Quan; Yingxue Zhang
>
> **备注:** 31 pages, 6 figures
>
> **摘要:** Large Language Models (LLMs) have recently shown strong promise for robotic task planning, particularly through automatic planning domain generation. Planning domains are brittle under imperfect logical states and perception noise; prior approaches largely treat generated planning domains as plan utilities, overlooking their potential as scalable sources of reasoning supervision and structured reward signals. At the same time, reasoning LLMs depend on chain-of-thought (CoT) supervision that is expensive to collect for robotic tasks, and reinforcement learning (RL) faces challenges in reward engineering. We propose Self-CriTeach, an LLM self-teaching and self-critiquing framework in which an LLM autonomously generates symbolic planning domains that serve a dual role: (i) enabling large-scale generation of robotic planning problem-plan pairs, and (ii) providing structured reward functions. First, the self-written domains enable large-scale generation of symbolic task plans, which are automatically transformed into extended CoT trajectories for supervised fine-tuning. Second, the self-written domains are reused as structured reward functions, providing dense feedback for reinforcement learning without manual reward engineering. This unified training pipeline yields a planning-enhanced LLM with higher planning success rates, stronger cross-task generalization, reduced inference cost, and improved robustness to imperfect logical states.
>
---
#### [replaced 012] MetaSym: A Symplectic Meta-learning Framework for Physical Intelligence
- **分类: cs.LG; cs.RO; physics.comp-ph; quant-ph**

- **简介: 该论文提出MetaSym框架，解决物理感知深度学习的泛化与适应问题。结合对称性先验和元注意力机制，提升物理系统建模的准确性与数据效率。**

- **链接: [https://arxiv.org/pdf/2502.16667v3](https://arxiv.org/pdf/2502.16667v3)**

> **作者:** Pranav Vaidhyanathan; Aristotelis Papatheodorou; Mark T. Mitchison; Natalia Ares; Ioannis Havoutis
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR), 10 + 18 pages, 9 figures, 10 tables
>
> **摘要:** Scalable and generalizable physics-aware deep learning has long been considered a significant challenge with various applications across diverse domains ranging from robotics to molecular dynamics. Central to almost all physical systems are symplectic forms, the geometric backbone that underpins fundamental invariants like energy and momentum. In this work, we introduce a novel deep learning framework, MetaSym. In particular, MetaSym combines a strong symplectic inductive bias obtained from a symplectic encoder, and an autoregressive decoder with meta-attention. This principled design ensures that core physical invariants remain intact, while allowing flexible, data efficient adaptation to system heterogeneities. We benchmark MetaSym with highly varied and realistic datasets, such as a high-dimensional spring-mesh system Otness et al. (2021), an open quantum system with dissipation and measurement backaction, and robotics-inspired quadrotor dynamics. Crucially, we fine-tune and deploy MetaSym on real-world quadrotor data, demonstrating robustness to sensor noise and real-world uncertainty. Across all tasks, MetaSym achieves superior few-shot adaptation and outperforms larger state-of-the-art (SOTA) models.
>
---
#### [replaced 013] L2M-Reg: Building-level Uncertainty-aware Registration of Outdoor LiDAR Point Clouds and Semantic 3D City Models
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于点云与三维模型配准任务，解决建筑级LiDAR与语义3D模型的精确对齐问题，提出L2M-Reg方法处理模型不确定性。**

- **链接: [https://arxiv.org/pdf/2509.16832v3](https://arxiv.org/pdf/2509.16832v3)**

> **作者:** Ziyang Xu; Benedikt Schwab; Yihui Yang; Thomas H. Kolbe; Christoph Holst
>
> **备注:** Accepted version by ISPRS Journal of Photogrammetry and Remote Sensing
>
> **摘要:** Accurate registration between LiDAR (Light Detection and Ranging) point clouds and semantic 3D city models is a fundamental topic in urban digital twinning and a prerequisite for downstream tasks, such as digital construction, change detection, and model refinement. However, achieving accurate LiDAR-to-Model registration at the individual building level remains challenging, particularly due to the generalization uncertainty in semantic 3D city models at the Level of Detail 2 (LoD2). This paper addresses this gap by proposing L2M-Reg, a plane-based fine registration method that explicitly accounts for model uncertainty. L2M-Reg consists of three key steps: establishing reliable plane correspondence, building a pseudo-plane-constrained Gauss-Helmert model, and adaptively estimating vertical translation. Overall, extensive experiments on five real-world datasets demonstrate that L2M-Reg is both more accurate and computationally efficient than current leading ICP-based and plane-based methods. Therefore, L2M-Reg provides a novel building-level solution regarding LiDAR-to-Model registration when model uncertainty is present. The datasets and code for L2M-Reg can be found: https://github.com/Ziyang-Geodesy/L2M-Reg.
>
---
#### [replaced 014] Toward Learning POMDPs Beyond Full-Rank Actions and State Observability
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究如何从部分可观测数据中学习POMDP模型，解决隐藏状态识别问题。通过PSR方法，学习观测和转移矩阵，实现不同目标的规划。**

- **链接: [https://arxiv.org/pdf/2601.18930v3](https://arxiv.org/pdf/2601.18930v3)**

> **作者:** Seiji Shaw; Travis Manderson; Chad Kessens; Nicholas Roy
>
> **备注:** Update abstract
>
> **摘要:** We are interested in enabling autonomous agents to learn and reason about systems with hidden states, such as locking mechanisms. We cast this problem as learning the parameters of a discrete Partially Observable Markov Decision Process (POMDP). The agent begins with knowledge of the POMDP's actions and observation spaces, but not its state space, transitions, or observation models. These properties must be constructed from a sequence of actions and observations. Spectral approaches to learning models of partially observable domains, such as Predictive State Representations (PSRs), learn representations of state that are sufficient to predict future outcomes. PSR models, however, do not have explicit transition and observation system models that can be used with different reward functions to solve different planning problems. Under a mild set of rankness assumptions on the products of transition and observation matrices, we show how PSRs learn POMDP matrices up to a similarity transform, and this transform may be estimated via tensor decomposition methods. Our method learns observation matrices and transition matrices up to a partition of states, where the states in a single partition have the same observation distributions corresponding to actions whose transition matrices are full-rank. Our experiments suggest that explicit observation and transition likelihoods can be leveraged to generate new plans for different goals and reward functions after the model has been learned. We also show that learning a POMDP beyond a partition of states is impossible from sequential data by constructing two POMDPs that agree on all observation distributions but differ in their transition dynamics.
>
---
#### [replaced 015] Mapping the Unseen: Unified Promptable Panoptic Mapping with Dynamic Labeling using Foundation Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉场景理解任务，解决开放词汇下语义分割与几何一致性问题。通过引入动态描述符和多尺度TSDF地图，实现高精度的统一语义映射。**

- **链接: [https://arxiv.org/pdf/2405.02162v5](https://arxiv.org/pdf/2405.02162v5)**

> **作者:** Mohamad Al Mdfaa; Raghad Salameh; Geesara Kulathunga; Sergey Zagoruyko; Gonzalo Ferrer
>
> **摘要:** Panoptic maps enable robots to reason about both geometry and semantics. However, open-vocabulary models repeatedly produce closely related labels that split panoptic entities and degrade volumetric consistency. The proposed UPPM advances open-world scene understanding by leveraging foundation models to introduce a panoptic Dynamic Descriptor that reconciles open-vocabulary labels with unified category structure and geometric size priors. The fusion for such dynamic descriptors is performed within a multi-resolution multi-TSDF map using language-guided open-vocabulary panoptic segmentation and semantic retrieval, resulting in a persistent and promptable panoptic map without additional model training. Based on our evaluation experiments, UPPM shows the best overall performance in terms of the map reconstruction accuracy and the panoptic segmentation quality. The ablation study investigates the contribution for each component of UPPM (custom NMS, blurry-frame filtering, and unified semantics) to the overall system performance. Consequently, UPPM preserves open-vocabulary interpretability while delivering strong geometric and panoptic accuracy.
>
---
#### [replaced 016] AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文提出AIR-VLA，针对空中操作系统的视觉-语言-动作任务，解决VLA模型在空中平台应用的挑战，通过构建数据集和仿真环境评估现有模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21602v2](https://arxiv.org/pdf/2601.21602v2)**

> **作者:** Jianli Sun; Bin Tian; Qiyao Zhang; Chengxiang Li; Zihan Song; Zhiyong Cui; Yisheng Lv; Yonglin Tian
>
> **摘要:** While Vision-Language-Action (VLA) models have achieved remarkable success in ground-based embodied intelligence, their application to Aerial Manipulation Systems (AMS) remains a largely unexplored frontier. The inherent characteristics of AMS, including floating-base dynamics, strong coupling between the UAV and the manipulator, and the multi-step, long-horizon nature of operational tasks, pose severe challenges to existing VLA paradigms designed for static or 2D mobile bases. To bridge this gap, we propose \textbf{AIR-VLA}, the first VLA benchmark specifically tailored for aerial manipulation. We construct a physics-based simulation environment and release a high-quality multimodal dataset comprising 3000 manually teleoperated demonstrations, covering base manipulation, object \& spatial understanding, semantic reasoning, and long-horizon planning. Leveraging this platform, we systematically evaluate mainstream VLA models and state-of-the-art VLM models. Our experiments not only validate the feasibility of transferring VLA paradigms to aerial systems but also, through multi-dimensional metrics tailored to aerial tasks, reveal the capabilities and boundaries of current models regarding UAV mobility, manipulator control, and high-level planning. \textbf{AIR-VLA} establishes a standardized testbed and data foundation for future research in general-purpose aerial robotics. The resource of AIR-VLA will be available at https://github.com/SpencerSon2001/AIR-VLA.
>
---
#### [replaced 017] Scene-Adaptive Motion Planning with Explicit Mixture of Experts and Interaction-Oriented Optimization
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶轨迹规划任务，解决复杂城市环境下的多模态轨迹生成与交互问题。提出EMoE-Planner，结合专家混合与交互优化，提升规划性能。**

- **链接: [https://arxiv.org/pdf/2505.12311v3](https://arxiv.org/pdf/2505.12311v3)**

> **作者:** Hongbiao Zhu; Liulong Ma; Xian Wu; Xin Deng; Xiaoyao Liang
>
> **备注:** Main text 10 pages with 7 figures
>
> **摘要:** Despite over a decade of development, autonomous driving trajectory planning in complex urban environments continues to encounter significant challenges. These challenges include the difficulty in accommodating the multi-modal nature of trajectories, the limitations of single expert model in managing diverse scenarios, and insufficient consideration of environmental interactions. To address these issues, this paper introduces the EMoE-Planner, which incorporates three innovative approaches. Firstly, the Explicit MoE (Mixture of Experts) dynamically selects specialized experts based on scenario-specific information through a shared scene router. Secondly, the planner utilizes scene-specific queries to provide multi-modal priors, directing the model's focus towards relevant target areas. Lastly, it enhances the prediction model and loss calculation by considering the interactions between the ego vehicle and other agents, thereby significantly boosting planning performance. Comparative experiments were conducted on the Nuplan dataset against the state-of-the-art methods. The simulation results demonstrate that our model consistently outperforms SOTA models across nearly all test scenarios. Our model is the first pure learning model to achieve performance surpassing rule-based algorithms in almost all Nuplan closed-loop simulations.
>
---
#### [replaced 018] LAGEA: Language Guided Embodied Agents for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出LAGEA框架，用于机器人操作任务中的自我反思与纠错。通过语言反馈指导强化学习，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2509.23155v2](https://arxiv.org/pdf/2509.23155v2)**

> **作者:** Abdul Monaf Chowdhury; Akm Moshiur Rahman Mazumder; Rabeya Akter; Safaeid Hossain Arib
>
> **摘要:** Robotic manipulation benefits from foundation models that describe goals, but today's agents still lack a principled way to learn from their own mistakes. We ask whether natural language can serve as feedback, an error-reasoning signal that helps embodied agents diagnose what went wrong and correct course. We introduce LAGEA (Language Guided Embodied Agents), a framework that turns episodic, schema-constrained reflections from a vision language model (VLM) into temporally grounded guidance for reinforcement learning. LAGEA summarizes each attempt in concise language, localizes the decisive moments in the trajectory, aligns feedback with visual state in a shared representation, and converts goal progress and feedback agreement into bounded, step-wise shaping rewards whose influence is modulated by an adaptive, failure-aware coefficient. This design yields dense signals early when exploration needs direction and gracefully recedes as competence grows. On the Meta-World MT10 and Robotic Fetch embodied manipulation benchmark, LAGEA improves average success over the state-of-the-art (SOTA) methods by 9.0% on random goals, 5.3% on fixed goals, and 17% on fetch tasks, while converging faster. These results support our hypothesis: language, when structured and grounded in time, is an effective mechanism for teaching robots to self-reflect on mistakes and make better choices.
>
---
#### [replaced 019] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于双臂机器人操作任务，旨在解决少样本学习与动态环境适应问题。通过视觉-语言锚定方法，实现技能复用与泛化，提升系统灵活性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.21723v3](https://arxiv.org/pdf/2509.21723v3)**

> **作者:** Huayi Zhou; Kui Jia
>
> **备注:** accepted by ICLR 2026. The project link is https://hnuzhy.github.io/projects/VLBiMan/
>
> **摘要:** Achieving generalizable bimanual manipulation requires systems that can learn efficiently from minimal human input while adapting to real-world uncertainties and diverse embodiments. Existing approaches face a dilemma: imitation policy learning demands extensive demonstrations to cover task variations, while modular methods often lack flexibility in dynamic scenes. We introduce VLBiMan, a framework that derives reusable skills from a single human example through task-aware decomposition, preserving invariant primitives as anchors while dynamically adapting adjustable components via vision-language grounding. This adaptation mechanism resolves scene ambiguities caused by background changes, object repositioning, or visual clutter without policy retraining, leveraging semantic parsing and geometric feasibility constraints. Moreover, the system inherits human-like hybrid control capabilities, enabling mixed synchronous and asynchronous use of both arms. Extensive experiments validate VLBiMan across tool-use and multi-object tasks, demonstrating: (1) a drastic reduction in demonstration requirements compared to imitation baselines, (2) compositional generalization through atomic skill splicing for long-horizon tasks, (3) robustness to novel but semantically similar objects and external disturbances, and (4) strong cross-embodiment transfer, showing that skills learned from human demonstrations can be instantiated on different robotic platforms without retraining. By bridging human priors with vision-language anchored adaptation, our work takes a step toward practical and versatile dual-arm manipulation in unstructured settings.
>
---
#### [replaced 020] Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究基于脉冲神经网络的连续控制任务，解决其在机器人操控中的应用难题。通过端到端训练，实现高效控制，验证了SNN在高维连续控制中的可行性与优势。**

- **链接: [https://arxiv.org/pdf/2509.05356v3](https://arxiv.org/pdf/2509.05356v3)**

> **作者:** Justus Huebotter; Pablo Lanillos; Marcel van Gerven; Serge Thill
>
> **摘要:** Despite recent progress in training spiking neural networks (SNNs) for classification, their application to continuous motor control remains limited. Here, we demonstrate that fully spiking architectures can be trained end-to-end to control robotic arms with multiple degrees of freedom in continuous environments. Our predictive-control framework combines Leaky Integrate-and-Fire dynamics with surrogate gradients, jointly optimizing a forward model for dynamics prediction and a policy network for goal-directed action. We evaluate this approach on both a planar 2D reaching task and a simulated 6-DOF Franka Emika Panda robot with torque control. In direct comparison to non-spiking recurrent baselines trained under the same predictive-control pipeline, the proposed SNN achieves comparable task performance while using substantially fewer parameters. An extensive ablation study highlights the role of initialization, learnable time constants, adaptive thresholds, and latent-space compression as key contributors to stable training and effective control. Together, these findings establish spiking neural networks as a viable and scalable substrate for high-dimensional continuous control, while emphasizing the importance of principled architectural and training design.
>
---
#### [replaced 021] Correspondence-Free, Function-Based Sim-to-Real Learning for Deformable Surface Control
- **分类: cs.RO**

- **简介: 该论文属于sim-to-real学习任务，解决变形表面控制问题。提出无需对应关系的函数方法，同时学习变形空间和置信图，实现从仿真到现实的迁移。**

- **链接: [https://arxiv.org/pdf/2509.00060v3](https://arxiv.org/pdf/2509.00060v3)**

> **作者:** Yingjun Tian; Guoxin Fang; Renbo Su; Aoran Lyu; Neelotpal Dutta; Weiming Wang; Simeon Gill; Andrew Weightman; Charlie C. L. Wang
>
> **备注:** arXiv admin note: text overlap with arXiv:2405.08935
>
> **摘要:** This paper presents a correspondence-free, function-based sim-to-real learning method for controlling deformable freeform surfaces. Unlike traditional sim-to-real transfer methods that strongly rely on marker points with full correspondences, our approach simultaneously learns a deformation function space and a confidence map -- both parameterized by a neural network -- to map simulated shapes to their real-world counterparts. As a result, the sim-to-real learning can be conducted by input from either a 3D scanner as point clouds (without correspondences) or a motion capture system as marker points (tolerating missed markers). The resultant sim-to-real transfer can be seamlessly integrated into a neural network-based computational pipeline for inverse kinematics and shape control. We demonstrate the versatility and adaptability of our method on both vision devices and across four pneumatically actuated soft robots: a deformable membrane, a robotic mannequin, and two soft manipulators.
>
---
#### [replaced 022] Multi-Agent Pathfinding Under Team-Connected Communication Constraint via Adaptive Path Expansion and Dynamic Leading
- **分类: cs.AI; cs.MA; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决团队通信连通性约束下的路径规划问题。提出一种结合自适应路径扩展和动态领头的框架，提升规划效率与成功率。**

- **链接: [https://arxiv.org/pdf/2501.02770v5](https://arxiv.org/pdf/2501.02770v5)**

> **作者:** Hoang-Dung Bui; Erion Plaku; Gregoy J. Stein
>
> **摘要:** This paper proposes a novel planning framework to handle a multi-agent pathfinding problem under team-connected communication constraint, where all agents must have a connected communication channel to the rest of the team during their entire movements. Standard multi-agent path finding approaches (e.g., priority-based search) have potential in this domain but fail when neighboring configurations at start and goal differ. Their single-expansion approach -- computing each agent's path from the start to the goal in just a single expansion -- cannot reliably handle planning under communication constraints for agents as their neighbors change during navigating. Similarly, leader-follower approaches (e.g., platooning) are effective at maintaining team communication, but fixing the leader at the outset of planning can cause planning to become stuck in dense-clutter environments, limiting their practical utility. To overcome this limitation, we propose a novel two-level multi-agent pathfinding framework that integrates two techniques: adaptive path expansion to expand agent paths to their goals in multiple stages; and dynamic leading technique that enables the reselection of the leading agent during each agent path expansion whenever progress cannot be made. Simulation experiments show the efficiency of our planners, which can handle up to 25 agents across five environment types under a limited communication range constraint and up to 11-12 agents on three environment types under line-of-sight communication constraint, exceeding 90% success-rate where baselines routinely fail.
>
---
#### [replaced 023] Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景重建任务，解决LiDAR与相机融合的复杂性和材料表示不足问题，提出仅用相机的重建方法，结合语义材料标签和物理材质属性。**

- **链接: [https://arxiv.org/pdf/2511.20348v3](https://arxiv.org/pdf/2511.20348v3)**

> **作者:** Andy Huynh; João Malheiro Silva; Holger Caesar; Tong Duy Son
>
> **备注:** 8 pages, 5 figures. Accepted to IEEE Intelligent Vehicles Symposium (IV) 2026. Revised version (v3) presents camera-ready publication
>
> **摘要:** 3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
>
---
#### [replaced 024] USS-Nav: Unified Spatio-Semantic Scene Graph for Lightweight UAV Zero-Shot Object Navigation
- **分类: cs.RO**

- **简介: 该论文属于无人机零样本目标导航任务，解决未知环境中语义推理与计算资源限制的矛盾。提出USS-Nav框架，构建统一时空场景图，提升导航效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.00708v2](https://arxiv.org/pdf/2602.00708v2)**

> **作者:** Weiqi Gai; Yuman Gao; Yuan Zhou; Yufan Xie; Zhiyang Liu; Yuze Wu; Xin Zhou; Fei Gao; Zhijun Meng
>
> **摘要:** Zero-Shot Object Navigation in unknown environments poses significant challenges for Unmanned Aerial Vehicles (UAVs) due to the conflict between high-level semantic reasoning requirements and limited onboard computational resources. To address this, we present USS-Nav, a lightweight framework that incrementally constructs a Unified Spatio-Semantic scene graph and enables efficient Large Language Model (LLM)-augmented Zero-Shot Object Navigation in unknown environments. Specifically, we introduce an incremental Spatial Connectivity Graph generation method utilizing polyhedral expansion to capture global geometric topology, which is dynamically partitioned into semantic regions via graph clustering. Concurrently, open-vocabulary object semantics are instantiated and anchored to this topology to form a hierarchical environmental representation. Leveraging this hierarchical structure, we present a coarse-to-fine exploration strategy: LLM grounded in the scene graph's semantics to determine global target regions, while a local planner optimizes frontier coverage based on information gain. Experimental results demonstrate that our framework outperforms state-of-the-art methods in terms of computational efficiency and real-time update frequency (15 Hz) on a resource-constrained platform. Furthermore, ablation studies confirm the effectiveness of our framework, showing substantial improvements in Success weighted by Path Length (SPL). The source code will be made publicly available to foster further research.
>
---
#### [replaced 025] RFS: Reinforcement learning with Residual flow steering for dexterous manipulation
- **分类: cs.RO**

- **简介: 该论文提出RFS框架，用于优化预训练的生成策略，解决机器人精细操作中策略泛化不足的问题。通过残差动作和潜在噪声联合优化，提升适应效率。**

- **链接: [https://arxiv.org/pdf/2602.01789v2](https://arxiv.org/pdf/2602.01789v2)**

> **作者:** Entong Su; Tyler Westenbroek; Anusha Nagabandi; Abhishek Gupta
>
> **摘要:** Imitation learning has emerged as an effective approach for bootstrapping sequential decision-making in robotics, achieving strong performance even in high-dimensional dexterous manipulation tasks. Recent behavior cloning methods further leverage expressive generative models, such as diffusion models and flow matching, to represent multimodal action distributions. However, policies pretrained in this manner often exhibit limited generalization and require additional fine-tuning to achieve robust performance at deployment time. Such adaptation must preserve the global exploration benefits of pretraining while enabling rapid correction of local execution errors. We propose Residual Flow Steering(RFS), a data-efficient reinforcement learning framework for adapting pretrained generative policies. RFS steers a pretrained flow-matching policy by jointly optimizing a residual action and a latent noise distribution, enabling complementary forms of exploration: local refinement through residual corrections and global exploration through latent-space modulation. This design allows efficient adaptation while retaining the expressive structure of the pretrained policy. We demonstrate the effectiveness of RFS on dexterous manipulation tasks, showing efficient fine-tuning in both simulation and real-world settings when adapting pretrained base policies. Project website:https://weirdlabuw.github.io/rfs.
>
---
#### [replaced 026] Fast Task Planning with Neuro-Symbolic Relaxation
- **分类: cs.RO**

- **简介: 该论文属于任务规划领域，解决复杂环境中长周期规划效率低的问题。通过结合神经网络与符号推理，提出Flax方法提升规划速度与可靠性。**

- **链接: [https://arxiv.org/pdf/2507.15975v2](https://arxiv.org/pdf/2507.15975v2)**

> **作者:** Qiwei Du; Bowen Li; Yi Du; Shaoshu Su; Taimeng Fu; Zitong Zhan; Zhipeng Zhao; Chen Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Real-world task planning requires long-horizon reasoning over large sets of objects with complex relationships and attributes, leading to a combinatorial explosion for classical symbolic planners. To prune the search space, recent methods prioritize searching on a simplified task only containing a few ``important" objects predicted by a neural network. However, such a simple neuro-symbolic (NeSy) integration risks omitting critical objects and wasting resources on unsolvable simplified tasks. To enable Fast and reliable planning, we introduce a NeSy relaxation strategy (Flax), combining neural importance prediction with symbolic expansion. Specifically, we first learn a graph neural network to predict object importance to create a simplified task and solve it with a symbolic planner. Then, we solve a rule-relaxed task to obtain a quick rough plan, and reintegrate all referenced objects into the simplified task to recover any overlooked but essential elements. Finally, we apply complementary rules to refine the updated task, keeping it both reliable and compact. Extensive experiments are conducted on both synthetic and real-world maze navigation benchmarks where a robot must traverse through a maze and interact with movable obstacles. The results show that Flax boosts the average success rate by 20.82\% and cuts mean wall-clock planning time by 17.65\% compared with the state-of-the-art NeSy baseline. We expect that Flax offers a practical path toward fast, scalable, long-horizon task planning in complex environments.
>
---
#### [replaced 027] CoFreeVLA: Collision-Free Dual-Arm Manipulation via Vision-Language-Action Model and Risk Estimation
- **分类: cs.RO**

- **简介: 该论文属于双臂机械臂操作任务，解决自碰撞安全问题。通过引入风险估计模块，提升VLA模型的碰撞规避能力，提高操作安全性与成功率。**

- **链接: [https://arxiv.org/pdf/2601.21712v2](https://arxiv.org/pdf/2601.21712v2)**

> **作者:** Xuanran Zhai; Binkai Ou; Qiaojun Yu; Ce Hao; Yaohua Liu
>
> **摘要:** Vision Language Action (VLA) models enable instruction following manipulation, yet dualarm deployment remains unsafe due to under modeled selfcollisions between arms and grasped objects. We introduce CoFreeVLA, which augments an endtoend VLA with a short horizon selfcollision risk estimator that predicts collision likelihood from proprioception, visual embeddings, and planned actions. The estimator gates risky commands, recovers to safe states via risk-guided adjustments, and shapes policy refinement for safer rollouts. It is pre-trained with model-based collision labels and posttrained on real robot rollouts for calibration. On five bimanual tasks with the PiPER robot arm, CoFreeVLA reduces selfcollisions and improves success rates versus RDT and APEX.
>
---
#### [replaced 028] What does really matter in image goal navigation?
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究图像目标导航任务，探讨是否可通过端到端强化学习解决。工作包括分析不同架构对相对位姿估计的影响，并验证其在真实环境中的迁移能力。**

- **链接: [https://arxiv.org/pdf/2507.01667v3](https://arxiv.org/pdf/2507.01667v3)**

> **作者:** Gianluca Monaci; Philippe Weinzaepfel; Christian Wolf
>
> **摘要:** Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In this large experimental study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extent. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill.
>
---
#### [replaced 029] Driving on Registers
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决端到端驾驶的高效性与准确性问题。提出DrivoR架构，利用视觉Transformer和相机感知的寄存器令牌压缩特征，提升计算效率并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.05083v2](https://arxiv.org/pdf/2601.05083v2)**

> **作者:** Ellington Kirby; Alexandre Boulch; Yihong Xu; Yuan Yin; Gilles Puy; Éloi Zablocki; Andrei Bursuc; Spyros Gidaris; Renaud Marlet; Florent Bartoccioni; Anh-Quan Cao; Nermin Samet; Tuan-Hung VU; Matthieu Cord
>
> **摘要:** We present DrivoR, a simple and efficient transformer-based architecture for end-to-end autonomous driving. Our approach builds on pretrained Vision Transformers (ViTs) and introduces camera-aware register tokens that compress multi-camera features into a compact scene representation, significantly reducing downstream computation without sacrificing accuracy. These tokens drive two lightweight transformer decoders that generate and then score candidate trajectories. The scoring decoder learns to mimic an oracle and predicts interpretable sub-scores representing aspects such as safety, comfort, and efficiency, enabling behavior-conditioned driving at inference. Despite its minimal design, DrivoR outperforms or matches strong contemporary baselines across NAVSIM-v1, NAVSIM-v2, and the photorealistic closed-loop HUGSIM benchmark. Our results show that a pure-transformer architecture, combined with targeted token compression, is sufficient for accurate, efficient, and adaptive end-to-end driving. Code and checkpoints will be made available via the project page.
>
---
#### [replaced 030] MapDream: Task-Driven Map Learning for Vision-Language Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决传统地图构建与导航策略脱节的问题。提出MapDream框架，通过生成式地图学习提升导航性能。**

- **链接: [https://arxiv.org/pdf/2602.00222v2](https://arxiv.org/pdf/2602.00222v2)**

> **作者:** Guoxin Lian; Shuo Wang; Yucheng Wang; Yongcai Wang; Maiyue Chen; Kaihui Wang; Bo Zhang; Zhizhong Su; Deying Li; Zhaoxin Fan
>
> **摘要:** Vision-Language Navigation (VLN) requires agents to follow natural language instructions in partially observed 3D environments, motivating map representations that aggregate spatial context beyond local perception. However, most existing approaches rely on hand-crafted maps constructed independently of the navigation policy. We argue that maps should instead be learned representations shaped directly by navigation objectives rather than exhaustive reconstructions. Based on this insight, we propose MapDream, a map-in-the-loop framework that formulates map construction as autoregressive bird's-eye-view (BEV) image synthesis. The framework jointly learns map generation and action prediction, distilling environmental context into a compact three-channel BEV map that preserves only navigation-critical affordances. Supervised pre-training bootstraps a reliable mapping-to-control interface, while the autoregressive design enables end-to-end joint optimization through reinforcement fine-tuning. Experiments on R2R-CE and RxR-CE achieve state-of-the-art monocular performance, validating task-driven generative map learning.
>
---
#### [replaced 031] MSACL: Multi-Step Actor-Critic Learning with Lyapunov Certificates for Exponentially Stabilizing Control
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文提出MSACL，用于安全关键控制任务，解决RL稳定性保障与高效探索的难题，通过引入Lyapunov证书和多步数据提升系统稳定性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.24955v2](https://arxiv.org/pdf/2512.24955v2)**

> **作者:** Yongwei Zhang; Yuanzhe Xing; Quanyi Liang; Quan Quan; Zhikun She
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** For safety-critical applications, model-free reinforcement learning (RL) faces numerous challenges, particularly the difficulty of establishing verifiable stability guarantees while maintaining high exploration efficiency. To address these challenges, we present Multi-Step Actor-Critic Learning with Lyapunov Certificates (MSACL), a novel approach that seamlessly integrates exponential stability with maximum entropy reinforcement learning (MERL). In contrast to existing methods that rely on complex reward engineering and single-step constraints, MSACL utilizes intuitive rewards and multi-step data for actor-critic learning. Specifically, we first introduce Exponential Stability Labels (ESLs) to categorize samples and propose a $λ$-weighted aggregation mechanism to learn Lyapunov certificates. Leveraging these certificates, we then develop a stability-aware advantage function to guide policy optimization, thereby ensuring rapid Lyapunov descent and robust state convergence. We evaluate MSACL across six benchmarks, comprising four stabilization and two high-dimensional tracking tasks. Experimental results demonstrate its consistent superiority over both standard RL baselines and state-of-the-art Lyapunov-based RL algorithms. Beyond rapid convergence, MSACL exhibits significant robustness against environmental uncertainties and remarkable generalization to unseen reference signals. The source code and benchmarking environments are available at \href{https://github.com/YuanZhe-Xing/MSACL}{https://github.com/YuanZhe-Xing/MSACL}.
>
---
#### [replaced 032] OptiPMB: Enhancing 3D Multi-Object Tracking with Optimized Poisson Multi-Bernoulli Filtering
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D多目标跟踪任务，旨在解决传统方法在数据关联和轨迹管理上的不足。提出OptiPMB方法，采用优化的泊松多伯努利滤波器提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2503.12968v2](https://arxiv.org/pdf/2503.12968v2)**

> **作者:** Guanhua Ding; Yuxuan Xia; Runwei Guan; Qinchen Wu; Tao Huang; Weiping Ding; Jinping Sun; Guoqiang Mao
>
> **摘要:** Accurate 3D multi-object tracking (MOT) is crucial for autonomous driving, as it enables robust perception, navigation, and planning in complex environments. While deep learning-based solutions have demonstrated impressive 3D MOT performance, model-based approaches remain appealing for their simplicity, interpretability, and data efficiency. Conventional model-based trackers typically rely on random vector-based Bayesian filters within the tracking-by-detection (TBD) framework but face limitations due to heuristic data association and track management schemes. In contrast, random finite set (RFS)-based Bayesian filtering handles object birth, survival, and death in a theoretically sound manner, facilitating interpretability and parameter tuning. In this paper, we present OptiPMB, a novel RFS-based 3D MOT method that employs an optimized Poisson multi-Bernoulli (PMB) filter while incorporating several key innovative designs within the TBD framework. Specifically, we propose a measurement-driven hybrid adaptive birth model for improved track initialization, employ adaptive detection probability parameters to effectively maintain tracks for occluded objects, and optimize density pruning and track extraction modules to further enhance overall tracking performance. Extensive evaluations on nuScenes and KITTI datasets show that OptiPMB achieves superior tracking accuracy compared with state-of-the-art methods, thereby establishing a new benchmark for model-based 3D MOT and offering valuable insights for future research on RFS-based trackers in autonomous driving.
>
---
#### [replaced 033] Task-Centric Policy Optimization from Misaligned Motion Priors
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决运动先验与任务目标不匹配的问题。提出TCMP框架，将模仿学习作为条件正则化，优先保证任务性能，提升控制稳定性与自然性。**

- **链接: [https://arxiv.org/pdf/2601.19411v2](https://arxiv.org/pdf/2601.19411v2)**

> **作者:** Ziang Zheng; Kai Feng; Yi Nie; Shentao Qin
>
> **备注:** Work requires further details and not complete yet
>
> **摘要:** Humanoid control often leverages motion priors from human demonstrations to encourage natural behaviors. However, such demonstrations are frequently suboptimal or misaligned with robotic tasks due to embodiment differences, retargeting errors, and task-irrelevant variations, causing naïve imitation to degrade task performance. Conversely, task-only reinforcement learning admits many task-optimal solutions, often resulting in unnatural or unstable motions. This exposes a fundamental limitation of linear reward mixing in adversarial imitation learning. We propose \emph{Task-Centric Motion Priors} (TCMP), a task-priority adversarial imitation framework that treats imitation as a conditional regularizer rather than a co-equal objective. TCMP maximizes task improvement while incorporating imitation signals only when they are compatible with task progress, yielding an adaptive, geometry-aware update that preserves task-feasible descent and suppresses harmful imitation under misalignment. We provide theoretical analysis of gradient conflict and task-priority stationary points, and validate our claims through humanoid control experiments demonstrating robust task performance with consistent motion style under noisy demonstrations.
>
---
