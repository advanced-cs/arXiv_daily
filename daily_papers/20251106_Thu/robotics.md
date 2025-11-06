# 机器人 cs.RO

- **最新发布 26 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] Motion Planning Under Temporal Logic Specifications In Semantically Unknown Environments
- **分类: cs.RO**

- **简介: 该论文研究在语义未知环境中基于scLTL规范的运动规划问题，提出一种基于产物自动机与奖励函数的自动化方法，结合值迭代实现在线重规划，以处理区域位置不确定的时序任务。**

- **链接: [http://arxiv.org/pdf/2511.03652v1](http://arxiv.org/pdf/2511.03652v1)**

> **作者:** Azizollah Taheri; Derya Aksaray
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This paper addresses a motion planning problem to achieve spatio-temporal-logical tasks, expressed by syntactically co-safe linear temporal logic specifications (scLTL\next), in uncertain environments. Here, the uncertainty is modeled as some probabilistic knowledge on the semantic labels of the environment. For example, the task is "first go to region 1, then go to region 2"; however, the exact locations of regions 1 and 2 are not known a priori, instead a probabilistic belief is available. We propose a novel automata-theoretic approach, where a special product automaton is constructed to capture the uncertainty related to semantic labels, and a reward function is designed for each edge of this product automaton. The proposed algorithm utilizes value iteration for online replanning. We show some theoretical results and present some simulations/experiments to demonstrate the efficacy of the proposed approach.
>
---
#### [new 002] Manifold-constrained Hamilton-Jacobi Reachability Learning for Decentralized Multi-Agent Motion Planning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对多智能体运动规划中的流形约束安全问题，提出一种流形约束的哈密顿-雅可比可达性学习框架，实现无需假设其他智能体策略的分布式安全轨迹规划，兼顾任务可行性与实时性。**

- **链接: [http://arxiv.org/pdf/2511.03591v1](http://arxiv.org/pdf/2511.03591v1)**

> **作者:** Qingyi Chen; Ruiqi Ni; Jun Kim; Ahmed H. Qureshi
>
> **摘要:** Safe multi-agent motion planning (MAMP) under task-induced constraints is a critical challenge in robotics. Many real-world scenarios require robots to navigate dynamic environments while adhering to manifold constraints imposed by tasks. For example, service robots must carry cups upright while avoiding collisions with humans or other robots. Despite recent advances in decentralized MAMP for high-dimensional systems, incorporating manifold constraints remains difficult. To address this, we propose a manifold-constrained Hamilton-Jacobi reachability (HJR) learning framework for decentralized MAMP. Our method solves HJR problems under manifold constraints to capture task-aware safety conditions, which are then integrated into a decentralized trajectory optimization planner. This enables robots to generate motion plans that are both safe and task-feasible without requiring assumptions about other agents' policies. Our approach generalizes across diverse manifold-constrained tasks and scales effectively to high-dimensional multi-agent manipulation problems. Experiments show that our method outperforms existing constrained motion planners and operates at speeds suitable for real-world applications. Video demonstrations are available at https://youtu.be/RYcEHMnPTH8 .
>
---
#### [new 003] Learning Natural and Robust Hexapod Locomotion over Complex Terrains via Motion Priors based on Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文面向六足机器人复杂地形行走任务，解决多腿协调难、运动不自然的问题，提出基于深度强化学习与运动先验的策略学习方法，通过对抗判别器引导生成自然稳健步态，并成功实现在无视觉信息下的真实机器人部署。**

- **链接: [http://arxiv.org/pdf/2511.03167v1](http://arxiv.org/pdf/2511.03167v1)**

> **作者:** Xin Liu; Jinze Wu; Yinghui Li; Chenkun Qi; Yufei Xue; Feng Gao
>
> **摘要:** Multi-legged robots offer enhanced stability to navigate complex terrains with their multiple legs interacting with the environment. However, how to effectively coordinate the multiple legs in a larger action exploration space to generate natural and robust movements is a key issue. In this paper, we introduce a motion prior-based approach, successfully applying deep reinforcement learning algorithms to a real hexapod robot. We generate a dataset of optimized motion priors, and train an adversarial discriminator based on the priors to guide the hexapod robot to learn natural gaits. The learned policy is then successfully transferred to a real hexapod robot, and demonstrate natural gait patterns and remarkable robustness without visual information in complex terrains. This is the first time that a reinforcement learning controller has been used to achieve complex terrain walking on a real hexapod robot.
>
---
#### [new 004] OneOcc: Semantic Occupancy Prediction for Legged Robots with a Single Panoramic Camera
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 论文提出OneOcc，面向腿足机器人，仅用全景相机实现360°语义占据预测，解决运动抖动与视角连续性难题，创新融合双投影、双网格、动态解码与步态补偿模块，并发布两个全景基准数据集，达到SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.03571v1](http://arxiv.org/pdf/2511.03571v1)**

> **作者:** Hao Shi; Ze Wang; Shangwei Guo; Mengfei Duan; Song Wang; Teng Chen; Kailun Yang; Lin Wang; Kaiwei Wang
>
> **备注:** Datasets and code will be publicly available at https://github.com/MasterHow/OneOcc
>
> **摘要:** Robust 3D semantic occupancy is crucial for legged/humanoid robots, yet most semantic scene completion (SSC) systems target wheeled platforms with forward-facing sensors. We present OneOcc, a vision-only panoramic SSC framework designed for gait-introduced body jitter and 360{\deg} continuity. OneOcc combines: (i) Dual-Projection fusion (DP-ER) to exploit the annular panorama and its equirectangular unfolding, preserving 360{\deg} continuity and grid alignment; (ii) Bi-Grid Voxelization (BGV) to reason in Cartesian and cylindrical-polar spaces, reducing discretization bias and sharpening free/occupied boundaries; (iii) a lightweight decoder with Hierarchical AMoE-3D for dynamic multi-scale fusion and better long-range/occlusion reasoning; and (iv) plug-and-play Gait Displacement Compensation (GDC) learning feature-level motion correction without extra sensors. We also release two panoramic occupancy benchmarks: QuadOcc (real quadruped, first-person 360{\deg}) and Human360Occ (H3O) (CARLA human-ego 360{\deg} with RGB, Depth, semantic occupancy; standardized within-/cross-city splits). OneOcc sets new state-of-the-art (SOTA): on QuadOcc it beats strong vision baselines and popular LiDAR ones; on H3O it gains +3.83 mIoU (within-city) and +8.08 (cross-city). Modules are lightweight, enabling deployable full-surround perception for legged/humanoid robots. Datasets and code will be publicly available at https://github.com/MasterHow/OneOcc.
>
---
#### [new 005] Source-Free Bistable Fluidic Gripper for Size-Selective and Stiffness-Adaptive Grasping
- **分类: cs.RO**

- **简介: 该论文提出一种无外部供能的双稳态流体夹持器，通过内部液体自 redistributing 实现尺寸选择性抓取与刚度自适应，解决传统软体夹持器依赖外源、续航差的问题，适用于水下与野外自主操作。**

- **链接: [http://arxiv.org/pdf/2511.03691v1](http://arxiv.org/pdf/2511.03691v1)**

> **作者:** Zhihang Qin; Yueheng Zhang; Wan Su; Linxin Hou; Shenghao Zhou; Zhijun Chen; Yu Jun Tan; Cecilia Laschi
>
> **摘要:** Conventional fluid-driven soft grippers typically depend on external sources, which limit portability and long-term autonomy. This work introduces a self-contained soft gripper with fixed size that operates solely through internal liquid redistribution among three interconnected bistable snap-through chambers. When the top sensing chamber deforms upon contact, the displaced liquid triggers snap-through expansion of the grasping chambers, enabling stable and size-selective grasping without continuous energy input. The internal hydraulic feedback further allows passive adaptation of gripping pressure to object stiffness. This source-free and compact design opens new possibilities for lightweight, stiffness-adaptive fluid-driven manipulation in soft robotics, providing a feasible approach for targeted size-specific sampling and operation in underwater and field environments.
>
---
#### [new 006] GUIDES: Guidance Using Instructor-Distilled Embeddings for Pre-trained Robot Policy Enhancement
- **分类: cs.RO**

- **简介: 论文提出GUIDES框架，通过微调视觉语言模型生成语义引导嵌入，注入预训练机器人策略的潜在空间，实现无需重构的性能提升，并引入大语言模型监控与推理机制增强鲁棒性，高效升级现有机器人策略。**

- **链接: [http://arxiv.org/pdf/2511.03400v1](http://arxiv.org/pdf/2511.03400v1)**

> **作者:** Minquan Gao; Xinyi Li; Qing Yan; Xiaojian Sun; Xiaopan Zhang; Chien-Ming Huang; Jiachen Li
>
> **备注:** 8 pages, 4 figures, Accepted by IEEE IROS 2025 Workshop WIR-M
>
> **摘要:** Pre-trained robot policies serve as the foundation of many validated robotic systems, which encapsulate extensive embodied knowledge. However, they often lack the semantic awareness characteristic of foundation models, and replacing them entirely is impractical in many situations due to high costs and the loss of accumulated knowledge. To address this gap, we introduce GUIDES, a lightweight framework that augments pre-trained policies with semantic guidance from foundation models without requiring architectural redesign. GUIDES employs a fine-tuned vision-language model (Instructor) to generate contextual instructions, which are encoded by an auxiliary module into guidance embeddings. These embeddings are injected into the policy's latent space, allowing the legacy model to adapt to this new semantic input through brief, targeted fine-tuning. For inference-time robustness, a large language model-based Reflector monitors the Instructor's confidence and, when confidence is low, initiates a reasoning loop that analyzes execution history, retrieves relevant examples, and augments the VLM's context to refine subsequent actions. Extensive validation in the RoboCasa simulation environment across diverse policy architectures shows consistent and substantial improvements in task success rates. Real-world deployment on a UR5 robot further demonstrates that GUIDES enhances motion precision for critical sub-tasks such as grasping. Overall, GUIDES offers a practical and resource-efficient pathway to upgrade, rather than replace, validated robot policies.
>
---
#### [new 007] Multi-User Personalisation in Human-Robot Interaction: Using Quantitative Bipolar Argumentation Frameworks for Preferences Conflict Resolution
- **分类: cs.RO; cs.AI; 68T40; I.2.9; I.2.4**

- **简介: 该论文面向多用户人机交互中的偏好冲突问题，提出MUP-QBAF框架，利用量化双极论证模型动态整合用户偏好与环境观测，实现透明、可解释的偏好协商，突破单用户适应局限。**

- **链接: [http://arxiv.org/pdf/2511.03576v1](http://arxiv.org/pdf/2511.03576v1)**

> **作者:** Aniol Civit; Antonio Andriella; Carles Sierra; Guillem Alenyà
>
> **备注:** Preprint submitted to a journal
>
> **摘要:** While personalisation in Human-Robot Interaction (HRI) has advanced significantly, most existing approaches focus on single-user adaptation, overlooking scenarios involving multiple stakeholders with potentially conflicting preferences. To address this, we propose the Multi-User Preferences Quantitative Bipolar Argumentation Framework (MUP-QBAF), a novel multi-user personalisation framework based on Quantitative Bipolar Argumentation Frameworks (QBAFs) that explicitly models and resolves multi-user preference conflicts. Unlike prior work in Argumentation Frameworks, which typically assumes static inputs, our approach is tailored to robotics: it incorporates both users' arguments and the robot's dynamic observations of the environment, allowing the system to adapt over time and respond to changing contexts. Preferences, both positive and negative, are represented as arguments whose strength is recalculated iteratively based on new information. The framework's properties and capabilities are presented and validated through a realistic case study, where an assistive robot mediates between the conflicting preferences of a caregiver and a care recipient during a frailty assessment task. This evaluation further includes a sensitivity analysis of argument base scores, demonstrating how preference outcomes can be shaped by user input and contextual observations. By offering a transparent, structured, and context-sensitive approach to resolving competing user preferences, this work advances the field of multi-user HRI. It provides a principled alternative to data-driven methods, enabling robots to navigate conflicts in real-world environments.
>
---
#### [new 008] 3D Cal: An Open-Source Software Library for Calibrating Tactile Sensors
- **分类: cs.RO**

- **简介: 论文提出开源库3D Cal，通过3D打印机自动化校准触觉传感器，解决人工校准耗时问题，利用生成的标注数据训练CNN重建深度图，并评估数据需求与泛化性能，推动触觉传感在机器人中的应用。**

- **链接: [http://arxiv.org/pdf/2511.03078v1](http://arxiv.org/pdf/2511.03078v1)**

> **作者:** Rohan Kota; Kaival Shah; J. Edward Colgate; Gregory Reardon
>
> **摘要:** Tactile sensing plays a key role in enabling dexterous and reliable robotic manipulation, but realizing this capability requires substantial calibration to convert raw sensor readings into physically meaningful quantities. Despite its near-universal necessity, the calibration process remains ad hoc and labor-intensive. Here, we introduce \libname{}, an open-source library that transforms a low-cost 3D printer into an automated probing device capable of generating large volumes of labeled training data for tactile sensor calibration. We demonstrate the utility of \libname{} by calibrating two commercially available vision-based tactile sensors, DIGIT and GelSight Mini, to reconstruct high-quality depth maps using the collected data and a custom convolutional neural network. In addition, we perform a data ablation study to determine how much data is needed for accurate calibration, providing practical guidelines for researchers working with these specific sensors, and we benchmark the trained models on previously unseen objects to evaluate calibration accuracy and generalization performance. By automating tactile sensor calibration, \libname{} can accelerate tactile sensing research, simplify sensor deployment, and promote the practical integration of tactile sensing in robotic platforms.
>
---
#### [new 009] A Collaborative Reasoning Framework for Anomaly Diagnostics in Underwater Robotics
- **分类: cs.RO**

- **简介: 论文提出AURA框架，解决水下机器人异常诊断中无人类参与的AI局限问题，通过LLM、数字孪生与人机协同，实现实时异常检测、根因推理与知识迭代，构建自适应的人机协作诊断系统。**

- **链接: [http://arxiv.org/pdf/2511.03075v1](http://arxiv.org/pdf/2511.03075v1)**

> **作者:** Markus Buchholz; Ignacio Carlucho; Yvan R. Petillot
>
> **备注:** Paper was submitted for ICRA 2026
>
> **摘要:** The safe deployment of autonomous systems in safety-critical settings requires a paradigm that combines human expertise with AI-driven analysis, especially when anomalies are unforeseen. We introduce AURA (Autonomous Resilience Agent), a collaborative framework for anomaly and fault diagnostics in robotics. AURA integrates large language models (LLMs), a high-fidelity digital twin (DT), and human-in-the-loop interaction to detect and respond to anomalous behavior in real time. The architecture uses two agents with clear roles: (i) a low-level State Anomaly Characterization Agent that monitors telemetry and converts signals into a structured natural-language problem description, and (ii) a high-level Diagnostic Reasoning Agent that conducts a knowledge-grounded dialogue with an operator to identify root causes, drawing on external sources. Human-validated diagnoses are then converted into new training examples that refine the low-level perceptual model. This feedback loop progressively distills expert knowledge into the AI, transforming it from a static tool into an adaptive partner. We describe the framework's operating principles and provide a concrete implementation, establishing a pattern for trustworthy, continually improving human-robot teams.
>
---
#### [new 010] Toward an Agricultural Operational Design Domain: A Framework
- **分类: cs.RO; cs.SE; cs.SY; eess.SY; I.2.9; I.1.4; J.7**

- **简介: 该论文提出农业操作设计域（Ag-ODD）框架，解决现有ODD不适用于农业复杂环境的问题。通过结构化描述、七层模型与迭代验证，实现农业自动驾驶系统运行边界的标准化与可验证性。**

- **链接: [http://arxiv.org/pdf/2511.02937v1](http://arxiv.org/pdf/2511.02937v1)**

> **作者:** Mirco Felske; Jannik Redenius; Georg Happich; Julius Schöning
>
> **备注:** 18 pages, 7 figures, 2 tables
>
> **摘要:** The agricultural sector increasingly relies on autonomous systems that operate in complex and variable environments. Unlike on-road applications, agricultural automation integrates driving and working processes, each of which imposes distinct operational constraints. Handling this complexity and ensuring consistency throughout the development and validation processes requires a structured, transparent, and verified description of the environment. However, existing Operational Design Domain (ODD) concepts do not yet address the unique challenges of agricultural applications. Therefore, this work introduces the Agricultural ODD (Ag-ODD) Framework, which can be used to describe and verify the operational boundaries of autonomous agricultural systems. The Ag-ODD Framework consists of three core elements. First, the Ag-ODD description concept, which provides a structured method for unambiguously defining environmental and operational parameters using concepts from ASAM Open ODD and CityGML. Second, the 7-Layer Model derived from the PEGASUS 6-Layer Model, has been extended to include a process layer to capture dynamic agricultural operations. Third, the iterative verification process verifies the Ag-ODD against its corresponding logical scenarios, derived from the 7-Layer Model, to ensure the Ag-ODD's completeness and consistency. Together, these elements provide a consistent approach for creating unambiguous and verifiable Ag-ODD. Demonstrative use cases show how the Ag-ODD Framework can support the standardization and scalability of environmental descriptions for autonomous agricultural systems.
>
---
#### [new 011] Learning-based Cooperative Robotic Paper Wrapping: A Unified Control Policy with Residual Force Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究人机协作的纸张包装任务，解决变形物体操作中动态不可预测与力控难的问题。提出基于LLM与START模型的统一策略，融合模仿与强化学习，实现高精度、长时序的包装控制，成功率97%。**

- **链接: [http://arxiv.org/pdf/2511.03181v1](http://arxiv.org/pdf/2511.03181v1)**

> **作者:** Rewida Ali; Cristian C. Beltran-Hernandez; Weiwei Wan; Kensuke Harada
>
> **摘要:** Human-robot cooperation is essential in environments such as warehouses and retail stores, where workers frequently handle deformable objects like paper, bags, and fabrics. Coordinating robotic actions with human assistance remains difficult due to the unpredictable dynamics of deformable materials and the need for adaptive force control. To explore this challenge, we focus on the task of gift wrapping, which exemplifies a long-horizon manipulation problem involving precise folding, controlled creasing, and secure fixation of paper. Success is achieved when the robot completes the sequence to produce a neatly wrapped package with clean folds and no tears. We propose a learning-based framework that integrates a high-level task planner powered by a large language model (LLM) with a low-level hybrid imitation learning (IL) and reinforcement learning (RL) policy. At its core is a Sub-task Aware Robotic Transformer (START) that learns a unified policy from human demonstrations. The key novelty lies in capturing long-range temporal dependencies across the full wrapping sequence within a single model. Unlike vanilla Action Chunking with Transformer (ACT), typically applied to short tasks, our method introduces sub-task IDs that provide explicit temporal grounding. This enables robust performance across the entire wrapping process and supports flexible execution, as the policy learns sub-goals rather than merely replicating motion sequences. Our framework achieves a 97% success rate on real-world wrapping tasks. We show that the unified transformer-based policy reduces the need for specialized models, allows controlled human supervision, and effectively bridges high-level intent with the fine-grained force control required for deformable object manipulation.
>
---
#### [new 012] Value Elicitation for a Socially Assistive Robot Addressing Social Anxiety: A Participatory Design Approach
- **分类: cs.RO**

- **简介: 该论文通过参与式设计工作坊，面向心理健康研究者， eliciting 与社交焦虑支持相关的机器人设计价值（如适应性、接受度、有效性），旨在推动以用户为中心的社交辅助机器人开发，解决现有支持不足的问题。**

- **链接: [http://arxiv.org/pdf/2511.03444v1](http://arxiv.org/pdf/2511.03444v1)**

> **作者:** Vesna Poprcova; Iulia Lefter; Martijn Warnier; Frances Brazier
>
> **备注:** Accepted at Value Engineering in AI (VALE) Workshop (ECAI 2025)
>
> **摘要:** Social anxiety is a prevalent mental health condition that can significantly impact overall well-being and quality of life. Despite its widespread effects, adequate support or treatment for social anxiety is often insufficient. Advances in technology, particularly in social robotics, offer promising opportunities to complement traditional mental health. As an initial step toward developing effective solutions, it is essential to understand the values that shape what is considered meaningful, acceptable, and helpful. In this study, a participatory design workshop was conducted with mental health academic researchers to elicit the underlying values that should inform the design of socially assistive robots for social anxiety support. Through creative, reflective, and envisioning activities, participants explored scenarios and design possibilities, allowing for systematic elicitation of values, expectations, needs, and preferences related to robot-supported interventions. The findings reveal rich insights into design-relevant values-including adaptivity, acceptance, and efficacy-that are core to support for individuals with social anxiety. This study highlights the significance of a research-led approach to value elicitation, emphasising user-centred and context-aware design considerations in the development of socially assistive robots.
>
---
#### [new 013] Collaborative Assembly Policy Learning of a Sightless Robot
- **分类: cs.RO; cs.HC; cs.SY; eess.SY**

- **简介: 该论文研究盲机器人与人类协作插板任务，解决传统力控难以估计人意图、强化学习因安全与稀疏奖励不适用的问题，提出融合人设计阻抗控制的强化学习方法，提升成功率与效率，降低人力负担。**

- **链接: [http://arxiv.org/pdf/2511.03189v1](http://arxiv.org/pdf/2511.03189v1)**

> **作者:** Zeqing Zhang; Weifeng Lu; Lei Yang; Wei Jing; Bowei Tang; Jia Pan
>
> **备注:** Accepted by IEEE ROBIO 2025
>
> **摘要:** This paper explores a physical human-robot collaboration (pHRC) task involving the joint insertion of a board into a frame by a sightless robot and a human operator. While admittance control is commonly used in pHRC tasks, it can be challenging to measure the force/torque applied by the human for accurate human intent estimation, limiting the robot's ability to assist in the collaborative task. Other methods that attempt to solve pHRC tasks using reinforcement learning (RL) are also unsuitable for the board-insertion task due to its safety constraints and sparse rewards. Therefore, we propose a novel RL approach that utilizes a human-designed admittance controller to facilitate more active robot behavior and reduce human effort. Through simulation and real-world experiments, we demonstrate that our approach outperforms admittance control in terms of success rate and task completion time. Additionally, we observed a significant reduction in measured force/torque when using our proposed approach compared to admittance control. The video of the experiments is available at https://youtu.be/va07Gw6YIog.
>
---
#### [new 014] Development of the Bioinspired Tendon-Driven DexHand 021 with Proprioceptive Compliance Control
- **分类: cs.RO; cs.AI**

- **简介: 论文提出轻量化1kg的腱驱动 DexHand 021 手，具19自由度，创新采用本体感觉力反馈自适应控制，提升抓取精度与安全性，实现33种抓取动作，显著降低关节力矩并增强力感知能力。**

- **链接: [http://arxiv.org/pdf/2511.03481v1](http://arxiv.org/pdf/2511.03481v1)**

> **作者:** Jianbo Yuan; Haohua Zhu; Jing Dai; Sheng Yi
>
> **备注:** 8 pages 18 fogures, IEEE RAL accept
>
> **摘要:** The human hand plays a vital role in daily life and industrial applications, yet replicating its multifunctional capabilities-including motion, sensing, and coordinated manipulation-with robotic systems remains a formidable challenge. Developing a dexterous robotic hand requires balancing human-like agility with engineering constraints such as complexity, size-to-weight ratio, durability, and force-sensing performance. This letter presents Dex-Hand 021, a high-performance, cable-driven five-finger robotic hand with 12 active and 7 passive degrees of freedom (DoFs), achieving 19 DoFs dexterity in a lightweight 1 kg design. We propose a proprioceptive force-sensing-based admittance control method to enhance manipulation. Experimental results demonstrate its superior performance: a single-finger load capacity exceeding 10 N, fingertip repeatability under 0.001 m, and force estimation errors below 0.2 N. Compared to PID control, joint torques in multi-object grasping are reduced by 31.19%, significantly improves force-sensing capability while preventing overload during collisions. The hand excels in both power and precision grasps, successfully executing 33 GRASP taxonomy motions and complex manipulation tasks. This work advances the design of lightweight, industrial-grade dexterous hands and enhances proprioceptive control, contributing to robotic manipulation and intelligent manufacturing.
>
---
#### [new 015] Unconscious and Intentional Human Motion Cues for Expressive Robot-Arm Motion Design
- **分类: cs.RO; cs.HC; H.5.2; I.2.9**

- **简介: 该论文研究如何利用人类无意识与有意动作 cues 设计富有表现力的机械臂运动，通过Geister游戏分析运动时序，构建相位依赖的机器人动作并评估其在实体与视频中的表现，揭示了撤回阶段时序与实体呈现对感知的关键影响。**

- **链接: [http://arxiv.org/pdf/2511.03676v1](http://arxiv.org/pdf/2511.03676v1)**

> **作者:** Taito Tashiro; Tomoko Yonezawa; Hirotake Yamazoe
>
> **备注:** 5 pages, 5 figures, HAI2025 Workshop on Socially Aware and Cooperative Intelligent Systems
>
> **摘要:** This study investigates how human motion cues can be used to design expressive robot-arm movements. Using the imperfect-information game Geister, we analyzed two types of human piece-moving motions: natural gameplay (unconscious tendencies) and instructed expressions (intentional cues). Based on these findings, we created phase-specific robot motions by varying movement speed and stop duration, and evaluated observer impressions under two presentation modalities: a physical robot and a recorded video. Results indicate that late-phase motion timing, particularly during withdrawal, plays an important role in impression formation and that physical embodiment enhances the interpretability of motion cues. These findings provide insights for designing expressive robot motions based on human timing behavior.
>
---
#### [new 016] Indicating Robot Vision Capabilities with Augmented Reality
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究人机协作中人类对机器人视野的误判问题，提出四种AR视野指示器，通过实验验证其提升认知对齐的效果，并给出六条设计指南，帮助用户准确理解机器人视觉能力。**

- **链接: [http://arxiv.org/pdf/2511.03550v1](http://arxiv.org/pdf/2511.03550v1)**

> **作者:** Hong Wang; Ridhima Phatak; James Ocampo; Zhao Han
>
> **摘要:** Research indicates that humans can mistakenly assume that robots and humans have the same field of view (FoV), possessing an inaccurate mental model of robots. This misperception may lead to failures during human-robot collaboration tasks where robots might be asked to complete impossible tasks about out-of-view objects. The issue is more severe when robots do not have a chance to scan the scene to update their world model while focusing on assigned tasks. To help align humans' mental models of robots' vision capabilities, we propose four FoV indicators in augmented reality (AR) and conducted a user human-subjects experiment (N=41) to evaluate them in terms of accuracy, confidence, task efficiency, and workload. These indicators span a spectrum from egocentric (robot's eye and head space) to allocentric (task space). Results showed that the allocentric blocks at the task space had the highest accuracy with a delay in interpreting the robot's FoV. The egocentric indicator of deeper eye sockets, possible for physical alteration, also increased accuracy. In all indicators, participants' confidence was high while cognitive load remained low. Finally, we contribute six guidelines for practitioners to apply our AR indicators or physical alterations to align humans' mental models with robots' vision capabilities.
>
---
#### [new 017] Flying Robotics Art: ROS-based Drone Draws the Record-Breaking Mural
- **分类: cs.RO; cs.CV; cs.SY; eess.SY; I.2.9; J.5**

- **简介: 该论文提出一种基于ROS的自主无人机系统，用于在户外恶劣条件下精准绘制超大壁画。通过融合IR与LiDAR定位、方向解耦控制与抗湍流喷漆机构，解决艺术精度与环境鲁棒性难题，实现高精度自主绘画。**

- **链接: [http://arxiv.org/pdf/2511.03651v1](http://arxiv.org/pdf/2511.03651v1)**

> **作者:** Andrei A. Korigodskii; Oleg D. Kalachev; Artem E. Vasiunik; Matvei V. Urvantsev; Georgii E. Bondar
>
> **摘要:** This paper presents the innovative design and successful deployment of a pioneering autonomous unmanned aerial system developed for executing the world's largest mural painted by a drone. Addressing the dual challenges of maintaining artistic precision and operational reliability under adverse outdoor conditions such as wind and direct sunlight, our work introduces a robust system capable of navigating and painting outdoors with unprecedented accuracy. Key to our approach is a novel navigation system that combines an infrared (IR) motion capture camera and LiDAR technology, enabling precise location tracking tailored specifically for largescale artistic applications. We employ a unique control architecture that uses different regulation in tangential and normal directions relative to the planned path, enabling precise trajectory tracking and stable line rendering. We also present algorithms for trajectory planning and path optimization, allowing for complex curve drawing and area filling. The system includes a custom-designed paint spraying mechanism, specifically engineered to function effectively amidst the turbulent airflow generated by the drone's propellers, which also protects the drone's critical components from paint-related damage, ensuring longevity and consistent performance. Experimental results demonstrate the system's robustness and precision in varied conditions, showcasing its potential for autonomous large-scale art creation and expanding the functional applications of robotics in creative fields.
>
---
#### [new 018] Multi-robot searching with limited sensing range for static and mobile intruders
- **分类: cs.RO; cs.CG; cs.CR; cs.MA**

- **简介: 该论文研究多机器人在正交多边形区域内搜索静止或移动入侵者的问题，因问题NP难，提出基于空间填充曲线、随机搜索等高效算法，并分析机器人数量与搜索时间的权衡。**

- **链接: [http://arxiv.org/pdf/2511.03622v1](http://arxiv.org/pdf/2511.03622v1)**

> **作者:** Swadhin Agrawal; Sujoy Bhore; Joseph S. B. Mitchell; P. B. Sujit; Aayush Gohil
>
> **摘要:** We consider the problem of searching for an intruder in a geometric domain by utilizing multiple search robots. The domain is a simply connected orthogonal polygon with edges parallel to the cartesian coordinate axes. Each robot has a limited sensing capability. We study the problem for both static and mobile intruders. It turns out that the problem of finding an intruder is NP-hard, even for a stationary intruder. Given this intractability, we turn our attention towards developing efficient and robust algorithms, namely methods based on space-filling curves, random search, and cooperative random search. Moreover, for each proposed algorithm, we evaluate the trade-off between the number of search robots and the time required for the robots to complete the search process while considering the geometric properties of the connected orthogonal search area.
>
---
#### [new 019] ROSBag MCP Server: Analyzing Robot Data with LLMs for Agentic Embodied AI Applications
- **分类: cs.RO; cs.AI; cs.SE**

- **简介: 该论文提出ROSBag MCP Server，解决机器人数据（如轨迹、激光雷达）与LLM/VLM交互分析的难题，构建支持自然语言查询的工具链，实现对ROS/ROS2数据的智能解析与可视化，并评估多模型工具调用能力。**

- **链接: [http://arxiv.org/pdf/2511.03497v1](http://arxiv.org/pdf/2511.03497v1)**

> **作者:** Lei Fu; Sahar Salimpour; Leonardo Militano; Harry Edelman; Jorge Peña Queralta; Giovanni Toffetti
>
> **摘要:** Agentic AI systems and Physical or Embodied AI systems have been two key research verticals at the forefront of Artificial Intelligence and Robotics, with Model Context Protocol (MCP) increasingly becoming a key component and enabler of agentic applications. However, the literature at the intersection of these verticals, i.e., Agentic Embodied AI, remains scarce. This paper introduces an MCP server for analyzing ROS and ROS 2 bags, allowing for analyzing, visualizing and processing robot data with natural language through LLMs and VLMs. We describe specific tooling built with robotics domain knowledge, with our initial release focused on mobile robotics and supporting natively the analysis of trajectories, laser scan data, transforms, or time series data. This is in addition to providing an interface to standard ROS 2 CLI tools ("ros2 bag list" or "ros2 bag info"), as well as the ability to filter bags with a subset of topics or trimmed in time. Coupled with the MCP server, we provide a lightweight UI that allows the benchmarking of the tooling with different LLMs, both proprietary (Anthropic, OpenAI) and open-source (through Groq). Our experimental results include the analysis of tool calling capabilities of eight different state-of-the-art LLM/VLM models, both proprietary and open-source, large and small. Our experiments indicate that there is a large divide in tool calling capabilities, with Kimi K2 and Claude Sonnet 4 demonstrating clearly superior performance. We also conclude that there are multiple factors affecting the success rates, from the tool description schema to the number of arguments, as well as the number of tools available to the models. The code is available with a permissive license at https://github.com/binabik-ai/mcp-rosbags.
>
---
#### [new 020] SENT Map - Semantically Enhanced Topological Maps with Foundation Models
- **分类: cs.RO**

- **简介: 论文提出SENT-Map，一种基于基础模型的语义增强拓扑地图，用于室内环境导航与操作。通过JSON格式融合语义信息，结合视觉与语言模型，实现人机可读的环境建模与高效规划，提升小模型在真实场景中的可行性。**

- **链接: [http://arxiv.org/pdf/2511.03165v1](http://arxiv.org/pdf/2511.03165v1)**

> **作者:** Raj Surya Rajendran Kathirvel; Zach A Chavis; Stephen J. Guy; Karthik Desingh
>
> **备注:** Accepted at ICRA 2025 Workshop on Foundation Models and Neuro-Symbolic AI for Robotics
>
> **摘要:** We introduce SENT-Map, a semantically enhanced topological map for representing indoor environments, designed to support autonomous navigation and manipulation by leveraging advancements in foundational models (FMs). Through representing the environment in a JSON text format, we enable semantic information to be added and edited in a format that both humans and FMs understand, while grounding the robot to existing nodes during planning to avoid infeasible states during deployment. Our proposed framework employs a two stage approach, first mapping the environment alongside an operator with a Vision-FM, then using the SENT-Map representation alongside a natural-language query within an FM for planning. Our experimental results show that semantic-enhancement enables even small locally-deployable FMs to successfully plan over indoor environments.
>
---
#### [new 021] Comprehensive Assessment of LiDAR Evaluation Metrics: A Comparative Study Using Simulated and Real Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文旨在评估LiDAR仿真与实测数据的相似性度量方法，解决虚拟测试环境真实性验证问题。通过对比多种指标，发现密度感知切比雪夫距离（DCD）最有效，并在真实数据构建的仿真环境中验证了其与感知性能的高度相关性。**

- **链接: [http://arxiv.org/pdf/2511.02994v1](http://arxiv.org/pdf/2511.02994v1)**

> **作者:** Syed Mostaquim Ali; Taufiq Rahman; Ghazal Farhani; Mohamed H. Zaki; Benoit Anctil; Dominique Charlebois
>
> **摘要:** For developing safe Autonomous Driving Systems (ADS), rigorous testing is required before they are deemed safe for road deployments. Since comprehensive conventional physical testing is impractical due to cost and safety concerns, Virtual Testing Environments (VTE) can be adopted as an alternative. Comparing VTE-generated sensor outputs against their real-world analogues can be a strong indication that the VTE accurately represents reality. Correspondingly, this work explores a comprehensive experimental approach to finding evaluation metrics suitable for comparing real-world and simulated LiDAR scans. The metrics were tested in terms of sensitivity and accuracy with different noise, density, distortion, sensor orientation, and channel settings. From comparing the metrics, we found that Density Aware Chamfer Distance (DCD) works best across all cases. In the second step of the research, a Virtual Testing Environment was generated using real LiDAR scan data. The data was collected in a controlled environment with only static objects using an instrumented vehicle equipped with LiDAR, IMU and cameras. Simulated LiDAR scans were generated from the VTEs using the same pose as real LiDAR scans. The simulated and LiDAR scans were compared in terms of model perception and geometric similarity. Actual and simulated LiDAR scans have a similar semantic segmentation output with a mIoU of 21\% with corrected intensity and an average density aware chamfer distance (DCD) of 0.63. This indicates a slight difference in the geometric properties of simulated and real LiDAR scans and a significant difference between model outputs. During the comparison, density-aware chamfer distance was found to be the most correlated among the metrics with perception methods.
>
---
#### [new 022] WorldPlanner: Monte Carlo Tree Search and MPC with Action-Conditioned Visual World Models
- **分类: cs.RO**

- **简介: 该论文提出WorldPlanner，基于无结构玩乐数据学习视觉世界模型，结合MCTS与MPC进行长程规划，解决机器人策略迁移难、数据采集贵的问题，显著超越行为克隆基线。**

- **链接: [http://arxiv.org/pdf/2511.03077v1](http://arxiv.org/pdf/2511.03077v1)**

> **作者:** R. Khorrambakht; Joaquim Ortiz-Haro; Joseph Amigo; Omar Mostafa; Daniel Dugas; Franziska Meier; Ludovic Righetti
>
> **摘要:** Robots must understand their environment from raw sensory inputs and reason about the consequences of their actions in it to solve complex tasks. Behavior Cloning (BC) leverages task-specific human demonstrations to learn this knowledge as end-to-end policies. However, these policies are difficult to transfer to new tasks, and generating training data is challenging because it requires careful demonstrations and frequent environment resets. In contrast to such policy-based view, in this paper we take a model-based approach where we collect a few hours of unstructured easy-to-collect play data to learn an action-conditioned visual world model, a diffusion-based action sampler, and optionally a reward model. The world model -- in combination with the action sampler and a reward model -- is then used to optimize long sequences of actions with a Monte Carlo Tree Search (MCTS) planner. The resulting plans are executed on the robot via a zeroth-order Model Predictive Controller (MPC). We show that the action sampler mitigates hallucinations of the world model during planning and validate our approach on 3 real-world robotic tasks with varying levels of planning and modeling complexity. Our experiments support the hypothesis that planning leads to a significant improvement over BC baselines on a standard manipulation test environment.
>
---
#### [new 023] OmniVLA: Unifiying Multi-Sensor Perception for Physically-Grounded Multimodal VLA
- **分类: cs.CV; cs.RO**

- **简介: 论文提出OmniVLA，一种融合红外、毫米波雷达和麦克风阵列的多模态视觉-语言-动作模型，通过传感器掩码图像统一多传感器数据，突破RGB-only限制，显著提升物理交互任务的成功率与学习效率。**

- **链接: [http://arxiv.org/pdf/2511.01210v1](http://arxiv.org/pdf/2511.01210v1)**

> **作者:** Heyu Guo; Shanmu Wang; Ruichun Ma; Shiqi Jiang; Yasaman Ghasempour; Omid Abari; Baining Guo; Lili Qi
>
> **摘要:** Vision-language-action (VLA) models have shown strong generalization for action prediction through large-scale vision-language pretraining. However, most existing models rely solely on RGB cameras, limiting their perception and, consequently, manipulation capabilities. We present OmniVLA, an omni-modality VLA model that integrates novel sensing modalities for physically-grounded spatial intelligence beyond RGB perception. The core of our approach is the sensor-masked image, a unified representation that overlays spatially grounded and physically meaningful masks onto the RGB images, derived from sensors including an infrared camera, a mmWave radar, and a microphone array. This image-native unification keeps sensor input close to RGB statistics to facilitate training, provides a uniform interface across sensor hardware, and enables data-efficient learning with lightweight per-sensor projectors. Built on this, we present a multisensory vision-language-action model architecture and train the model based on an RGB-pretrained VLA backbone. We evaluate OmniVLA on challenging real-world tasks where sensor-modality perception is needed to guide the manipulation. OmniVLA achieves an average task success rate of 84%, significantly outperforms both RGB-only and raw-sensor-input baseline models by 59% and 28% respectively, meanwhile showing higher learning efficiency and stronger generalization capability.
>
---
#### [new 024] Periodic Skill Discovery
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出周期性技能发现（PSD），解决强化学习中无监督技能发现忽视周期性行为的问题。通过将状态映射到环形潜空间，自动学习具有不同周期的周期性技能，适用于复杂机器人任务，提升下游任务性能与技能多样性。**

- **链接: [http://arxiv.org/pdf/2511.03187v1](http://arxiv.org/pdf/2511.03187v1)**

> **作者:** Jonghae Park; Daesol Cho; Jusuk Lee; Dongseok Shim; Inkyu Jang; H. Jin Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Unsupervised skill discovery in reinforcement learning (RL) aims to learn diverse behaviors without relying on external rewards. However, current methods often overlook the periodic nature of learned skills, focusing instead on increasing the mutual dependence between states and skills or maximizing the distance traveled in latent space. Considering that many robotic tasks -- particularly those involving locomotion -- require periodic behaviors across varying timescales, the ability to discover diverse periodic skills is essential. Motivated by this, we propose Periodic Skill Discovery (PSD), a framework that discovers periodic behaviors in an unsupervised manner. The key idea of PSD is to train an encoder that maps states to a circular latent space, thereby naturally encoding periodicity in the latent representation. By capturing temporal distance, PSD can effectively learn skills with diverse periods in complex robotic tasks, even with pixel-based observations. We further show that these learned skills achieve high performance on downstream tasks such as hurdling. Moreover, integrating PSD with an existing skill discovery method offers more diverse behaviors, thus broadening the agent's repertoire. Our code and demos are available at https://jonghaepark.github.io/psd/
>
---
#### [new 025] EvtSlowTV - A Large and Diverse Dataset for Event-Based Depth Estimation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出EvtSlowTV，首个大规模事件相机数据集，用于无监督深度估计。解决现有数据集规模小、场景受限问题，利用YouTube视频构建超13B事件数据，提升模型在复杂动态场景中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.02953v1](http://arxiv.org/pdf/2511.02953v1)**

> **作者:** Sadiq Layi Macaulay; Nimet Kaygusuz; Simon Hadfield
>
> **摘要:** Event cameras, with their high dynamic range (HDR) and low latency, offer a promising alternative for robust depth estimation in challenging environments. However, many event-based depth estimation approaches are constrained by small-scale annotated datasets, limiting their generalizability to real-world scenarios. To bridge this gap, we introduce EvtSlowTV, a large-scale event camera dataset curated from publicly available YouTube footage, which contains more than 13B events across various environmental conditions and motions, including seasonal hiking, flying, scenic driving, and underwater exploration. EvtSlowTV is an order of magnitude larger than existing event datasets, providing an unconstrained, naturalistic setting for event-based depth learning. This work shows the suitability of EvtSlowTV for a self-supervised learning framework to capitalise on the HDR potential of raw event streams. We further demonstrate that training with EvtSlowTV enhances the model's ability to generalise to complex scenes and motions. Our approach removes the need for frame-based annotations and preserves the asynchronous nature of event data.
>
---
#### [new 026] Optimizing Earth-Moon Transfer and Cislunar Navigation: Integrating Low-Energy Trajectories, AI Techniques and GNSS-R Technologies
- **分类: astro-ph.EP; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于综述研究，旨在解决地月转移成本高、导航覆盖不足问题。通过对比四种转移策略，融合AI用于轨迹优化与地形识别，并引入GNSS-R与PNT技术拓展地月空间导航与遥感能力，构建可持续的深空探索框架。**

- **链接: [http://arxiv.org/pdf/2511.03173v1](http://arxiv.org/pdf/2511.03173v1)**

> **作者:** Arsalan Muhammad; Wasiu Akande Ahmed; Omada Friday Ojonugwa; Paul Puspendu Biswas
>
> **摘要:** The rapid growth of cislunar activities, including lunar landings, the Lunar Gateway, and in-space refueling stations, requires advances in cost-efficient trajectory design and reliable integration of navigation and remote sensing. Traditional Earth-Moon transfers suffer from rigid launch windows and high propellant demands, while Earth-based GNSS systems provide little to no coverage beyond geostationary orbit. This limits autonomy and environmental awareness in cislunar space. This review compares four major transfer strategies by evaluating velocity requirements, flight durations, and fuel efficiency, and by identifying their suitability for both crewed and robotic missions. The emerging role of artificial intelligence and machine learning is highlighted: convolutional neural networks support automated crater recognition and digital terrain model generation, while deep reinforcement learning enables adaptive trajectory refinement during descent and landing to reduce risk and decision latency. The study also examines how GNSS-Reflectometry and advanced Positioning, Navigation, and Timing architectures can extend navigation capabilities beyond current limits. GNSS-R can act as a bistatic radar for mapping lunar ice, soil properties, and surface topography, while PNT systems support autonomous rendezvous, Lagrange point station-keeping, and coordinated satellite swarm operations. Combining these developments establishes a scalable framework for sustainable cislunar exploration and long-term human and robotic presence.
>
---
## 更新

#### [replaced 001] AURA: Autonomous Upskilling with Retrieval-Augmented Agents
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02507v3](http://arxiv.org/pdf/2506.02507v3)**

> **作者:** Alvin Zhu; Yusuke Tanaka; Andrew Goldberg; Dennis Hong
>
> **摘要:** Designing reinforcement learning curricula for agile robots traditionally requires extensive manual tuning of reward functions, environment randomizations, and training configurations. We introduce AURA (Autonomous Upskilling with Retrieval-Augmented Agents), a schema-validated curriculum reinforcement learning (RL) framework that leverages Large Language Models (LLMs) as autonomous designers of multi-stage curricula. AURA transforms user prompts into YAML workflows that encode full reward functions, domain randomization strategies, and training configurations. All files are statically validated before any GPU time is used, ensuring efficient and reliable execution. A retrieval-augmented feedback loop allows specialized LLM agents to design, execute, and refine curriculum stages based on prior training results stored in a vector database, enabling continual improvement over time. Quantitative experiments show that AURA consistently outperforms LLM-guided baselines in generation success rate, humanoid locomotion, and manipulation tasks. Ablation studies highlight the importance of schema validation and retrieval for curriculum quality. AURA successfully trains end-to-end policies directly from user prompts and deploys them zero-shot on a custom humanoid robot in multiple environments - capabilities that did not exist previously with manually designed controllers. By abstracting the complexity of curriculum design, AURA enables scalable and adaptive policy learning pipelines that would be complex to construct by hand. Project page: https://aura-research.org/
>
---
#### [replaced 002] Toward Humanoid Brain-Body Co-design: Joint Optimization of Control and Morphology for Fall Recovery
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.22336v2](http://arxiv.org/pdf/2510.22336v2)**

> **作者:** Bo Yue; Sheng Xu; Kui Jia; Guiliang Liu
>
> **摘要:** Humanoid robots represent a central frontier in embodied intelligence, as their anthropomorphic form enables natural deployment in humans' workspace. Brain-body co-design for humanoids presents a promising approach to realizing this potential by jointly optimizing control policies and physical morphology. Within this context, fall recovery emerges as a critical capability. It not only enhances safety and resilience but also integrates naturally with locomotion systems, thereby advancing the autonomy of humanoids. In this paper, we propose RoboCraft, a scalable humanoid co-design framework for fall recovery that iteratively improves performance through the coupled updates of control policy and morphology. A shared policy pretrained across multiple designs is progressively finetuned on high-performing morphologies, enabling efficient adaptation without retraining from scratch. Concurrently, morphology search is guided by human-inspired priors and optimization algorithms, supported by a priority buffer that balances reevaluation of promising candidates with the exploration of novel designs. Experiments show that RoboCraft achieves an average performance gain of 44.55% on seven public humanoid robots, with morphology optimization drives at least 40% of improvements in co-designing four humanoid robots, underscoring the critical role of humanoid co-design.
>
---
#### [replaced 003] Thor: Towards Human-Level Whole-Body Reactions for Intense Contact-Rich Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.26280v2](http://arxiv.org/pdf/2510.26280v2)**

> **作者:** Gangyang Li; Qing Shi; Youhao Hu; Jincheng Hu; Zhongyuan Wang; Xinlong Wang; Shaqi Luo
>
> **摘要:** Humanoids hold great potential for service, industrial, and rescue applications, in which robots must sustain whole-body stability while performing intense, contact-rich interactions with the environment. However, enabling humanoids to generate human-like, adaptive responses under such conditions remains a major challenge. To address this, we propose Thor, a humanoid framework for human-level whole-body reactions in contact-rich environments. Based on the robot's force analysis, we design a force-adaptive torso-tilt (FAT2) reward function to encourage humanoids to exhibit human-like responses during force-interaction tasks. To mitigate the high-dimensional challenges of humanoid control, Thor introduces a reinforcement learning architecture that decouples the upper body, waist, and lower body. Each component shares global observations of the whole body and jointly updates its parameters. Finally, we deploy Thor on the Unitree G1, and it substantially outperforms baselines in force-interaction tasks. Specifically, the robot achieves a peak pulling force of 167.7 N (approximately 48% of the G1's body weight) when moving backward and 145.5 N when moving forward, representing improvements of 68.9% and 74.7%, respectively, compared with the best-performing baseline. Moreover, Thor is capable of pulling a loaded rack (130 N) and opening a fire door with one hand (60 N). These results highlight Thor's effectiveness in enhancing humanoid force-interaction capabilities.
>
---
#### [replaced 004] Decentralized Aerial Manipulation of a Cable-Suspended Load using Multi-Agent Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.MA; I.2.9; I.2.11; I.2.6**

- **链接: [http://arxiv.org/pdf/2508.01522v3](http://arxiv.org/pdf/2508.01522v3)**

> **作者:** Jack Zeng; Andreu Matoses Gimenez; Eugene Vinitsky; Javier Alonso-Mora; Sihao Sun
>
> **摘要:** This paper presents the first decentralized method to enable real-world 6-DoF manipulation of a cable-suspended load using a team of Micro-Aerial Vehicles (MAVs). Our method leverages multi-agent reinforcement learning (MARL) to train an outer-loop control policy for each MAV. Unlike state-of-the-art controllers that utilize a centralized scheme, our policy does not require global states, inter-MAV communications, nor neighboring MAV information. Instead, agents communicate implicitly through load pose observations alone, which enables high scalability and flexibility. It also significantly reduces computing costs during inference time, enabling onboard deployment of the policy. In addition, we introduce a new action space design for the MAVs using linear acceleration and body rates. This choice, combined with a robust low-level controller, enables reliable sim-to-real transfer despite significant uncertainties caused by cable tension during dynamic 3D motion. We validate our method in various real-world experiments, including full-pose control under load model uncertainties, showing setpoint tracking performance comparable to the state-of-the-art centralized method. We also demonstrate cooperation amongst agents with heterogeneous control policies, and robustness to the complete in-flight loss of one MAV. Videos of experiments: https://autonomousrobots.nl/paper_websites/aerial-manipulation-marl
>
---
#### [replaced 005] Depth Matters: Multimodal RGB-D Perception for Robust Autonomous Agents
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16711v2](http://arxiv.org/pdf/2503.16711v2)**

> **作者:** Mihaela-Larisa Clement; Mónika Farsang; Felix Resch; Mihai-Teodor Stanusoiu; Radu Grosu
>
> **备注:** Submitted to ICRA 2025
>
> **摘要:** Autonomous agents that rely purely on perception to make real-time control decisions require efficient and robust architectures. In this work, we demonstrate that augmenting RGB input with depth information significantly enhances our agents' ability to predict steering commands compared to using RGB alone. We benchmark lightweight recurrent controllers that leverage the fused RGB-D features for sequential decision-making. To train our models, we collect high-quality data using a small-scale autonomous car controlled by an expert driver via a physical steering wheel, capturing varying levels of steering difficulty. Our models were successfully deployed on real hardware and inherently avoided dynamic and static obstacles, under out-of-distribution conditions. Specifically, our findings reveal that the early fusion of depth data results in a highly robust controller, which remains effective even with frame drops and increased noise levels, without compromising the network's focus on the task.
>
---
#### [replaced 006] ROADWork: A Dataset and Benchmark for Learning to Recognize, Observe, Analyze and Drive Through Work Zones
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.07661v3](http://arxiv.org/pdf/2406.07661v3)**

> **作者:** Anurag Ghosh; Shen Zheng; Robert Tamburo; Khiem Vuong; Juan Alvarez-Padilla; Hailiang Zhu; Michael Cardei; Nicholas Dunn; Christoph Mertz; Srinivasa G. Narasimhan
>
> **备注:** ICCV 2025 Accepted Paper
>
> **摘要:** Perceiving and autonomously navigating through work zones is a challenging and underexplored problem. Open datasets for this long-tailed scenario are scarce. We propose the ROADWork dataset to learn to recognize, observe, analyze, and drive through work zones. State-of-the-art foundation models fail when applied to work zones. Fine-tuning models on our dataset significantly improves perception and navigation in work zones. With ROADWork dataset, we discover new work zone images with higher precision (+32.5%) at a much higher rate (12.8$\times$) around the world. Open-vocabulary methods fail too, whereas fine-tuned detectors improve performance (+32.2 AP). Vision-Language Models (VLMs) struggle to describe work zones, but fine-tuning substantially improves performance (+36.7 SPICE). Beyond fine-tuning, we show the value of simple techniques. Video label propagation provides additional gains (+2.6 AP) for instance segmentation. While reading work zone signs, composing a detector and text spotter via crop-scaling improves performance +14.2% 1-NED). Composing work zone detections to provide context further reduces hallucinations (+3.9 SPICE) in VLMs. We predict navigational goals and compute drivable paths from work zone videos. Incorporating road work semantics ensures 53.6% goals have angular error (AE) < 0.5 (+9.9 %) and 75.3% pathways have AE < 0.5 (+8.1 %).
>
---
#### [replaced 007] An explicit construction of Kaleidocycles by elliptic theta functions
- **分类: nlin.SI; cs.RO; math.DG; 53A04, 53A70, 53A17, 70B15, 37K25, 37K10, 35Q53**

- **链接: [http://arxiv.org/pdf/2308.04977v3](http://arxiv.org/pdf/2308.04977v3)**

> **作者:** Shizuo Kaji; Kenji Kajiwara; Shota Shigetomi
>
> **摘要:** We consider the configuration space of ordered points on the two-dimensional sphere that satisfy a specific system of quadratic equations. We construct periodic orbits in this configuration space using elliptic theta functions and show that they simultaneously satisfy semi-discrete analogues of mKdV and sine-Gordon equations. The configuration space we investigate corresponds to the state space of a linkage mechanism known as the Kaleidocycle, and the constructed orbits describe the characteristic motion of the Kaleidocycle. A key consequence of our construction is the proof that Kaleidocycles exist for any number of tetrahedra greater than five. Our approach is founded on the relationship between the deformation of spatial curves and integrable systems, offering an intriguing example where an integrable system is explicitly solved to generate an orbit in the space of real solutions to polynomial equations defined by geometric constraints.
>
---
#### [replaced 008] Mastering Contact-rich Tasks by Combining Soft and Rigid Robotics with Imitation Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.07787v3](http://arxiv.org/pdf/2410.07787v3)**

> **作者:** Mariano Ramírez Montero; Ebrahim Shahabi; Giovanni Franzese; Jens Kober; Barbara Mazzolai; Cosimo Della Santina
>
> **备注:** Update with additional results and experiments
>
> **摘要:** Soft robots have the potential to revolutionize the use of robotic systems with their capability of establishing safe, robust, and adaptable interactions with their environment, but their precise control remains challenging. In contrast, traditional rigid robots offer high accuracy and repeatability but lack the flexibility of soft robots. We argue that combining these characteristics in a hybrid robotic platform can significantly enhance overall capabilities. This work presents a novel hybrid robotic platform that integrates a rigid manipulator with a fully developed soft arm. This system is equipped with the intelligence necessary to perform flexible and generalizable tasks through imitation learning autonomously. The physical softness and machine learning enable our platform to achieve highly generalizable skills, while the rigid components ensure precision and repeatability.
>
---
#### [replaced 009] NaviTrace: Evaluating Embodied Navigation of Vision-Language Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.26909v2](http://arxiv.org/pdf/2510.26909v2)**

> **作者:** Tim Windecker; Manthan Patel; Moritz Reuss; Richard Schwarzkopf; Cesar Cadena; Rudolf Lioutikov; Marco Hutter; Jonas Frey
>
> **备注:** 9 pages, 6 figures, under review at IEEE conference
>
> **摘要:** Vision-language models demonstrate unprecedented performance and generalization across a wide range of tasks and scenarios. Integrating these foundation models into robotic navigation systems opens pathways toward building general-purpose robots. Yet, evaluating these models' navigation capabilities remains constrained by costly real-world trials, overly simplified simulations, and limited benchmarks. We introduce NaviTrace, a high-quality Visual Question Answering benchmark where a model receives an instruction and embodiment type (human, legged robot, wheeled robot, bicycle) and must output a 2D navigation trace in image space. Across 1000 scenarios and more than 3000 expert traces, we systematically evaluate eight state-of-the-art VLMs using a newly introduced semantic-aware trace score. This metric combines Dynamic Time Warping distance, goal endpoint error, and embodiment-conditioned penalties derived from per-pixel semantics and correlates with human preferences. Our evaluation reveals consistent gap to human performance caused by poor spatial grounding and goal localization. NaviTrace establishes a scalable and reproducible benchmark for real-world robotic navigation. The benchmark and leaderboard can be found at https://leggedrobotics.github.io/navitrace_webpage/.
>
---
#### [replaced 010] RoboRAN: A Unified Robotics Framework for Reinforcement Learning-Based Autonomous Navigation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14526v2](http://arxiv.org/pdf/2505.14526v2)**

> **作者:** Matteo El-Hariry; Antoine Richard; Ricard M. Castan; Luis F. W. Batista; Matthieu Geist; Cedric Pradalier; Miguel Olivares-Mendez
>
> **备注:** Accepted at Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Autonomous robots must navigate and operate in diverse environments, from terrestrial and aquatic settings to aerial and space domains. While Reinforcement Learning (RL) has shown promise in training policies for specific autonomous robots, existing frameworks and benchmarks are often constrained to unique platforms, limiting generalization and fair comparisons across different mobility systems. In this paper, we present a multi-domain framework for training, evaluating and deploying RL-based navigation policies across diverse robotic platforms and operational environments. Our work presents four key contributions: (1) a scalable and modular framework, facilitating seamless robot-task interchangeability and reproducible training pipelines; (2) sim-to-real transfer demonstrated through real-world experiments with multiple robots, including a satellite robotic simulator, an unmanned surface vessel, and a wheeled ground vehicle; (3) the release of the first open-source API for deploying Isaac Lab-trained policies to real robots, enabling lightweight inference and rapid field validation; and (4) uniform tasks and metrics for cross-medium evaluation, through a unified evaluation testbed to assess performance of navigation tasks in diverse operational conditions (aquatic, terrestrial and space). By ensuring consistency between simulation and real-world deployment, RoboRAN lowers the barrier to developing adaptable RL-based navigation strategies. Its modular design enables straightforward integration of new robots and tasks through predefined templates, fostering reproducibility and extension to diverse domains. To support the community, we release RoboRAN as open-source.
>
---
#### [replaced 011] Hybrid Dynamics Modeling and Trajectory Planning for a Cable-Trailer System with a Quadruped Robot
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.12220v2](http://arxiv.org/pdf/2404.12220v2)**

> **作者:** Wentao Zhang; Shaohang Xu; Gewei Zuo; Bolin Li; Jingbo Wang; Lijun Zhu
>
> **备注:** 8 pages, 8 figures, Accept by RA-L 2025
>
> **摘要:** Inspired by sled-pulling dogs in transportation, we present a cable-trailer integrated with a quadruped robot system. The motion planning of this system faces challenges due to the interactions between the cable's state transitions, the trailer's nonholonomic constraints, and the system's underactuation. To address these challenges, we first develop a hybrid dynamics model that captures the cable's taut and slack states. A search algorithm is then introduced to compute a suboptimal trajectory while incorporating mode transitions. Additionally, we propose a novel collision avoidance constraint based on geometric polygons to formulate the trajectory optimization problem for the hybrid system. The proposed method is implemented on a Unitree A1 quadruped robot with a customized cable-trailer and validated through experiments. The real system demonstrates both agile and safe motion with cable mode transitions.
>
---
#### [replaced 012] mmE-Loc: Facilitating Accurate Drone Landing with Ultra-High-Frequency Localization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.09469v3](http://arxiv.org/pdf/2507.09469v3)**

> **作者:** Haoyang Wang; Jingao Xu; Xinyu Luo; Ting Zhang; Xuecheng Chen; Ruiyang Duan; Jialong Chen; Yunhao Liu; Jianfeng Zheng; Weijie Hong; Xinlei Chen
>
> **备注:** 17 pages, 34 figures. Journal extended version of arXiv:2502.14992
>
> **摘要:** For precise, efficient, and safe drone landings, ground platforms should real-time, accurately locate descending drones and guide them to designated spots. While mmWave sensing combined with cameras improves localization accuracy, lower sampling frequency of traditional frame cameras compared to mmWave radar creates bottlenecks in system throughput. In this work, we upgrade traditional frame camera with event camera, a novel sensor that harmonizes in sampling frequency with mmWave radar within ground platform setup, and introduce mmE-Loc, a high-precision, low-latency ground localization system designed for precise drone landings. To fully exploit the \textit{temporal consistency} and \textit{spatial complementarity} between these two modalities, we propose two innovative modules: \textit{(i)} the Consistency-instructed Collaborative Tracking module, which further leverages the drone's physical knowledge of periodic micro-motions and structure for accurate measurements extraction, and \textit{(ii)} the Graph-informed Adaptive Joint Optimization module, which integrates drone motion information for efficient sensor fusion and drone localization. Real-world experiments conducted in landing scenarios with a drone delivery company demonstrate that mmE-Loc significantly outperforms state-of-the-art methods in both accuracy and latency.
>
---
#### [replaced 013] Augmented Reality for RObots (ARRO): Pointing Visuomotor Policies Towards Visual Robustness
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.08627v2](http://arxiv.org/pdf/2505.08627v2)**

> **作者:** Reihaneh Mirjalili; Tobias Jülg; Florian Walter; Wolfram Burgard
>
> **摘要:** Visuomotor policies trained on human expert demonstrations have recently shown strong performance across a wide range of robotic manipulation tasks. However, these policies remain highly sensitive to domain shifts stemming from background or robot embodiment changes, which limits their generalization capabilities. In this paper, we present ARRO, a novel visual representation that leverages zero-shot open-vocabulary segmentation and object detection models to efficiently mask out task-irrelevant regions of the scene in real time without requiring additional training, modeling of the setup, or camera calibration. By filtering visual distractors and overlaying virtual guides during both training and inference, ARRO improves robustness to scene variations and reduces the need for additional data collection. We extensively evaluate ARRO with Diffusion Policy on a range of tabletop manipulation tasks in both simulation and real-world environments, and further demonstrate its compatibility and effectiveness with generalist robot policies, such as Octo and OpenVLA. Across all settings in our evaluation, ARRO yields consistent performance gains, allows for selective masking to choose between different objects, and shows robustness even to challenging segmentation conditions. Videos showcasing our results are available at: https://augmented-reality-for-robots.github.io/
>
---
#### [replaced 014] Autonomous Robotic Drilling System for Mice Cranial Window Creation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.14135v2](http://arxiv.org/pdf/2406.14135v2)**

> **作者:** Enduo Zhao; Murilo M. Marinho; Kanako Harada
>
> **备注:** 14 pages, 11 figures, accepted on T-ASE 2025
>
> **摘要:** Robotic assistance for experimental manipulation in the life sciences is expected to enable favorable outcomes, regardless of the skill of the scientist. Experimental specimens in the life sciences are subject to individual variability and hence require intricate algorithms for successful autonomous robotic control. As a use case, we are studying the cranial window creation in mice. This operation requires the removal of an 8-mm circular patch of the skull, which is approximately 300 um thick, but the shape and thickness of the mouse skull significantly varies depending on the strain of the mouse, sex, and age. In this work, we develop an autonomous robotic drilling system with no offline planning, consisting of a trajectory planner with execution-time feedback with drilling completion level recognition based on image and force information. In the experiments, we first evaluate the image-and-force-based drilling completion level recognition by comparing it with other state-of-the-art deep learning image processing methods and conduct an ablation study in eggshell drilling to evaluate the impact of each module on system performance. Finally, the system performance is further evaluated in postmortem mice, achieving a success rate of 70% (14/20 trials) with an average drilling time of 9.3 min.
>
---
#### [replaced 015] Deep Learning Warm Starts for Trajectory Optimization on the International Space Station
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05588v3](http://arxiv.org/pdf/2505.05588v3)**

> **作者:** Somrita Banerjee; Abhishek Cauligi; Marco Pavone
>
> **备注:** Accepted to 2025 International Conference on Space Robotics (iSpaRo). Presented at RSS 2025 Workshop on Space Robotics
>
> **摘要:** Trajectory optimization is a cornerstone of modern robot autonomy, enabling systems to compute trajectories and controls in real-time while respecting safety and physical constraints. However, it has seen limited usage in spaceflight applications due to its heavy computational demands that exceed the capability of most flight computers. In this work, we provide results on the first in-space demonstration of using machine learning-based warm starts for accelerating trajectory optimization for the Astrobee free-flying robot onboard the International Space Station (ISS). We formulate a data-driven optimal control approach that trains a neural network to learn the structure of the trajectory generation problem being solved using sequential convex programming (SCP). Onboard, this trained neural network predicts solutions for the trajectory generation problem and relies on using the SCP solver to enforce safety constraints for the system. Our trained network reduces the number of solver iterations required for convergence in cases including rotational dynamics by 60% and in cases with obstacles drawn from the training distribution of the warm start model by 50%. This work represents a significant milestone in the use of learning-based control for spaceflight applications and a stepping stone for future advances in the use of machine learning for autonomous guidance, navigation, & control.
>
---
#### [replaced 016] Human-Exoskeleton Kinematic Calibration to Improve Hand Tracking for Dexterous Teleoperation
- **分类: cs.RO; cs.HC; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.23592v2](http://arxiv.org/pdf/2507.23592v2)**

> **作者:** Haiyun Zhang; Stefano Dalla Gasperina; Saad N. Yousaf; Toshimitsu Tsuboi; Tetsuya Narita; Ashish D. Deshpande
>
> **备注:** 8 pages, 10 figures, 1 supplementary video, submitted to RA-L
>
> **摘要:** Hand exoskeletons are critical tools for dexterous teleoperation and immersive manipulation interfaces, but achieving accurate hand tracking remains a challenge due to user-specific anatomical variability and donning inconsistencies. These issues lead to kinematic misalignments that degrade tracking performance and limit applicability in precision tasks. We propose a subject-specific calibration framework for exoskeleton-based hand tracking that estimates virtual link parameters through residual-weighted optimization. A data-driven approach is introduced to empirically tune cost function weights using motion capture ground truth, enabling accurate and consistent calibration across users. Implemented on the Maestro hand exoskeleton with seven healthy participants, the method achieved substantial reductions in joint and fingertip tracking errors across diverse hand geometries. Qualitative visualizations using a Unity-based virtual hand further demonstrate improved motion fidelity. The proposed framework generalizes to exoskeletons with closed-loop kinematics and minimal sensing, laying the foundation for high-fidelity teleoperation and robot learning applications.
>
---
#### [replaced 017] Enhancing Fatigue Detection through Heterogeneous Multi-Source Data Integration and Cross-Domain Modality Imputation
- **分类: cs.RO; cs.AI; 62H30; I.2**

- **链接: [http://arxiv.org/pdf/2507.16859v3](http://arxiv.org/pdf/2507.16859v3)**

> **作者:** Luobin Cui; Yanlai Wu; Tang Ying; Weikai Li
>
> **备注:** 4figures,14pages
>
> **摘要:** Fatigue detection for human operators plays a key role in safety critical applications such as aviation, mining, and long haul transport. While numerous studies have demonstrated the effectiveness of high fidelity sensors in controlled laboratory environments, their performance often degrades when ported to real world settings due to noise, lighting conditions, and field of view constraints, thereby limiting their practicality. This paper formalizes a deployment oriented setting for real world fatigue detection, where high quality sensors are often unavailable in practical applications. To address this challenge, we propose leveraging knowledge from heterogeneous source domains, including high fidelity sensors that are difficult to deploy in the field but commonly used in controlled environments, to assist fatigue detection in the real world target domain. Building on this idea, we design a heterogeneous and multiple source fatigue detection framework that adaptively utilizes the available modalities in the target domain while exploiting diverse configurations in the source domains through alignment across domains and modality imputation. Our experiments, conducted using a field deployed sensor setup and two publicly available human fatigue datasets, demonstrate the practicality, robustness, and improved generalization of our approach across subjects and domains. The proposed method achieves consistent gains over strong baselines in sensor constrained scenarios.
>
---
#### [replaced 018] Multi-Agent Reinforcement Learning for Autonomous Multi-Satellite Earth Observation: A Realistic Case Study
- **分类: cs.AI; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15207v2](http://arxiv.org/pdf/2506.15207v2)**

> **作者:** Mohamad A. Hady; Siyi Hu; Mahardhika Pratama; Jimmy Cao; Ryszard Kowalczyk
>
> **摘要:** The exponential growth of Low Earth Orbit (LEO) satellites has revolutionised Earth Observation (EO) missions, addressing challenges in climate monitoring, disaster management, and more. However, autonomous coordination in multi-satellite systems remains a fundamental challenge. Traditional optimisation approaches struggle to handle the real-time decision-making demands of dynamic EO missions, necessitating the use of Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL). In this paper, we investigate RL-based autonomous EO mission planning by modelling single-satellite operations and extending to multi-satellite constellations using MARL frameworks. We address key challenges, including energy and data storage limitations, uncertainties in satellite observations, and the complexities of decentralised coordination under partial observability. By leveraging a near-realistic satellite simulation environment, we evaluate the training stability and performance of state-of-the-art MARL algorithms, including PPO, IPPO, MAPPO, and HAPPO. Our results demonstrate that MARL can effectively balance imaging and resource management while addressing non-stationarity and reward interdependency in multi-satellite coordination. The insights gained from this study provide a foundation for autonomous satellite operations, offering practical guidelines for improving policy learning in decentralised EO missions.
>
---
