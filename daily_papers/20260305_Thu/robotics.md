# 机器人 cs.RO

- **最新发布 62 篇**

- **更新 39 篇**

## 最新发布

#### [new 001] Self-adapting Robotic Agents through Online Continual Reinforcement Learning with World Model Feedback
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人强化学习任务，旨在解决机器人在运行中应对未知变化的能力问题。通过在线持续强化学习，实现机器人自主适应与优化。**

- **链接: [https://arxiv.org/pdf/2603.04029](https://arxiv.org/pdf/2603.04029)**

> **作者:** Fabian Domberg; Georg Schildbach
>
> **备注:** submitted to IROS 2026
>
> **摘要:** As learning-based robotic controllers are typically trained offline and deployed with fixed parameters, their ability to cope with unforeseen changes during operation is limited. Biologically inspired, this work presents a framework for online Continual Reinforcement Learning that enables automated adaptation during deployment. Building on DreamerV3, a model-based Reinforcement Learning algorithm, the proposed method leverages world model prediction residuals to detect out-of-distribution events and automatically trigger finetuning. Adaptation progress is monitored using both task-level performance signals and internal training metrics, allowing convergence to be assessed without external supervision and domain knowledge. The approach is validated on a variety of contemporary continuous control problems, including a quadruped robot in high-fidelity simulation, and a real-world model vehicle. Relevant metrics and their interpretation are presented and discussed, as well as resulting trade-offs described. The results sketch out how autonomous robotic agents could once move beyond static training regimes toward adaptive systems capable of self-reflection and -improvement during operation, just like their biological counterparts.
>
---
#### [new 002] GSeg3D: A High-Precision Grid-Based Algorithm for Safety-Critical Ground Segmentation in LiDAR Point Clouds
- **分类: cs.RO**

- **简介: 该论文属于点云地面分割任务，旨在解决安全关键场景下地面点与非地面点分离精度不足的问题，提出GSeg3D算法以提升检测可靠性。**

- **链接: [https://arxiv.org/pdf/2603.04208](https://arxiv.org/pdf/2603.04208)**

> **作者:** Muhammad Haider Khan Lodhi; Christoph Hertzberg
>
> **摘要:** Ground segmentation in point cloud data is the process of separating ground points from non-ground points. This task is fundamental for perception in autonomous driving and robotics, where safety and reliable operation depend on the precise detection of obstacles and navigable surfaces. Existing methods often fall short of the high precision required in safety-critical environments, leading to false detections that can compromise decision-making. In this work, we present a ground segmentation approach designed to deliver consistently high precision, supporting the stringent requirements of autonomous vehicles and robotic systems operating in real-world, safety-critical scenarios.
>
---
#### [new 003] Large-Language-Model-Guided State Estimation for Partially Observable Task and Motion Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人任务与运动规划领域，解决部分可观测环境下的状态估计问题。通过引入大语言模型引导的常识知识，提升规划效率。**

- **链接: [https://arxiv.org/pdf/2603.03704](https://arxiv.org/pdf/2603.03704)**

> **作者:** Yoonwoo Kim; Raghav Arora; Roberto Martín-Martín; Peter Stone; Ben Abbatematteo; Yoonchang Sung
>
> **摘要:** Robot planning in partially observable environments, where not all objects are known or visible, is a challenging problem, as it requires reasoning under uncertainty through partially observable Markov decision processes. During the execution of a computed plan, a robot may unexpectedly observe task-irrelevant objects, which are typically ignored by naive planners. In this work, we propose incorporating two types of common-sense knowledge: (1) certain objects are more likely to be found in specific locations; and (2) similar objects are likely to be co-located, while dissimilar objects are less likely to be found together. Manually engineering such knowledge is complex, so we explore leveraging the powerful common-sense reasoning capabilities of large language models (LLMs). Our planning and execution framework, CoCo-TAMP, introduces a hierarchical state estimation that uses LLM-guided information to shape the belief over task-relevant objects, enabling efficient solutions to long-horizon task and motion planning problems. In experiments, CoCo-TAMP achieves an average reduction of 62.7 in planning and execution time in simulation, and 72.6 in real-world demonstrations, compared to a baseline that does not incorporate either type of common-sense knowledge.
>
---
#### [new 004] X-Loco: Towards Generalist Humanoid Locomotion Control via Synergetic Policy Distillation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，旨在解决单一策略掌握多种运动技能的难题。通过X-Loco框架，融合多个专家策略，提升通用运动能力。**

- **链接: [https://arxiv.org/pdf/2603.03733](https://arxiv.org/pdf/2603.03733)**

> **作者:** Dewei Wang; Xinmiao Wang; Chenyun Zhang; Jiyuan Shi; Yingnan Zhao; Chenjia Bai; Xuelong Li
>
> **摘要:** While recent advances have demonstrated strong performance in individual humanoid skills such as upright locomotion, fall recovery and whole-body coordination, learning a single policy that masters all these skills remains challenging due to the diverse dynamics and conflicting control objectives involved. To address this, we introduce X-Loco, a framework for training a vision-based generalist humanoid locomotion policy. X-Loco trains multiple oracle specialist policies and adopts a synergetic policy distillation with a case-adaptive specialist selection mechanism, which dynamically leverages multiple specialist policies to guide a vision-based student policy. This design enables the student to acquire a broad spectrum of locomotion skills, ranging from fall recovery to terrain traversal and whole-body coordination skills. To the best of our knowledge, X-Loco is the first framework to demonstrate vision-based humanoid locomotion that jointly integrates upright locomotion, whole-body coordination and fall recovery, while operating solely under velocity commands without relying on reference motions. Experimental results show that X-Loco achieves superior performance, demonstrated by tasks such as fall recovery and terrain traversal. Ablation studies further highlight that our framework effectively leverages specialist expertise and enhances learning efficiency.
>
---
#### [new 005] Cognition to Control - Multi-Agent Learning for Human-Humanoid Collaborative Transport
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机协作任务，解决多智能体协作中长期决策与实时控制的融合问题。提出C2C框架，分层实现意图理解、协调决策与稳定控制，提升协作成功率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.03768](https://arxiv.org/pdf/2603.03768)**

> **作者:** Hao Zhang; Ding Zhao; H. Eric Tseng
>
> **摘要:** Effective human-robot collaboration (HRC) requires translating high-level intent into contact-stable whole-body motion while continuously adapting to a human partner. Many vision-language-action (VLA) systems learn end-to-end mappings from observations and instructions to actions, but they often emphasize reactive (System 1-like) behavior and leave under-specified how sustained System 2-style deliberation can be integrated with reliable, low-latency continuous control. This gap is acute in multi-agent HRC, where long-horizon coordination decisions and physical execution must co-evolve under contact, feasibility, and safety constraints. We address this limitation with cognition-to-control (C2C), a three-layer hierarchy that makes the deliberation-to-control pathway explicit: (i) a VLM-based grounding layer that maintains persistent scene referents and infers embodiment-aware affordances/constraints; (ii) a deliberative skill/coordination layer-the System 2 core-that optimizes long-horizon skill choices and sequences under human-robot coupling via decentralized MARL cast as a Markov potential game with a shared potential encoding task progress; and (iii) a whole-body control layer that executes the selected skills at high frequency while enforcing kinematic/dynamic feasibility and contact stability. The deliberative layer is realized as a residual policy relative to a nominal controller, internalizing partner dynamics without explicit role assignment. Experiments on collaborative manipulation tasks show higher success and robustness than single-agent and end-to-end baselines, with stable coordination and emergent leader-follower behaviors.
>
---
#### [new 006] HE-VPR: Height Estimation Enabled Aerial Visual Place Recognition Against Scale Variance
- **分类: cs.RO**

- **简介: 该论文属于视觉定位任务，解决高空拍摄图像的尺度变化问题。提出HE-VPR框架，结合高度估计与视觉定位，提升识别准确率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2603.04050](https://arxiv.org/pdf/2603.04050)**

> **作者:** Mengfan He; Xingyu Shao; Chunyu Li; Chao Chen; Liangzheng Sun; Ziyang Meng; Yuanqing Wu
>
> **摘要:** In this work, we propose HE-VPR, a visual place recognition (VPR) framework that incorporates height estimation. Our system decouples height inference from place recognition, allowing both modules to share a frozen DINOv2 backbone. Two lightweight bypass adapter branches are integrated into our system. The first estimates the height partition of the query image via retrieval from a compact height database, and the second performs VPR within the corresponding height-specific sub-database. The adaptation design reduces training cost and significantly decreases the search space of the database. We also adopt a center-weighted masking strategy to further enhance the robustness against scale differences. Experiments on two self-collected challenging multi-altitude datasets demonstrate that HE-VPR achieves up to 6.1\% Recall@1 improvement over state-of-the-art ViT-based baselines and reduces memory usage by up to 90\%. These results indicate that HE-VPR offers a scalable and efficient solution for height-aware aerial VPR, enabling practical deployment in GNSS-denied environments. All the code and datasets for this work have been released on this https URL.
>
---
#### [new 007] A Soft Robotic Demonstration in the Stratospher
- **分类: cs.RO**

- **简介: 该论文属于软体机器人技术领域，旨在解决极端环境下机器人适应性与耐久性问题。通过改进硅胶弹性体的交联机制，提升其在高温、低温及近真空条件下的性能，并成功应用于高空气球任务。**

- **链接: [https://arxiv.org/pdf/2603.04352](https://arxiv.org/pdf/2603.04352)**

> **作者:** Codrin Tugui; Tirth Thakar; Anatol Gogoj; Alexander White; Ang Leo Li; Alexander Yin; Edward Pomianek; Mihai Duduta
>
> **摘要:** Machines designed for operation in Space, as well as other extreme environments, need to be both resilient and adaptable when mission parameters change. Soft robots offer advantages in adaptability, but most lack resilience to the pressure and temperature extremes found as close as the Stratosphere. Dielectric elastomer actuators overcome some of those limitations when built as solid state compliant capacitors capable of converting electrical energy into mechanical work, but the elastomer resilience limits the device's operating window. Here we present a crosslinking mechanism for silicone elastomers under ultraviolet light using trimethyl(methylcyclopentadienyl)platinum(IV) as a catalyst to react hydrosilane to vinyl groups. The formation of carbon-carbon bonds enables fast processing under UV light and exceptional electro-mechanical performance in dielectric elastomer actuators. The material resilience advantage is demonstrated in controlled experiments at -40° and 120° C, as well as near vacuum, in comparison with state-of-the-art acrylic and silicone chemistries. Fully autonomous systems controlling grippers made with the novel silicone were integrated into payloads for high altitude balloon testing. Two stratospheric balloon missions were carried out and demonstrated DEAs as a viable soft robotic technology under space-like conditions (as high as 23.6 km elevation, at <0.05 atm and -55° C). The combinations of chemical building blocks and catalyst can be further expanded to address other challenges for silicones, including adhesion and additive manufacturing.
>
---
#### [new 008] AMP2026: A Multi-Platform Marine Robotics Dataset for Tracking and Mapping
- **分类: cs.RO**

- **简介: 该论文提出AMP2026数据集，用于海洋环境下的多平台跟踪与建图任务，解决动态海洋感知难题。工作包括数据采集、传感器配置及数据组织。**

- **链接: [https://arxiv.org/pdf/2603.04225](https://arxiv.org/pdf/2603.04225)**

> **作者:** Edwin Meriaux; Shuo Wen; David Widhalm; Zhizun Wang; Junming Shi; Mariana Sosa Guzmán; Kalvik Jakkala; Bennett Carley; Elias Sokolova; Yogesh Girdhar; Monika Roznere; Jason O'Kane; Junaed Sattar; Gregory Dudek
>
> **摘要:** Marine environments present significant challenges for perception and autonomy due to dynamic surfaces, limited visibility, and complex interactions between aerial, surface, and submerged sensing modalities. This paper introduces the Aerial Marine Perception Dataset (AMP2026), a multi-platform marine robotics dataset collected across multiple field deployments designed to support research in two primary areas: multi-view tracking and marine environment mapping. The dataset includes synchronized data from aerial drones, boat-mounted cameras, and submerged robotic platforms, along with associated localization and telemetry information. The goal of this work is to provide a publicly available dataset enabling research in marine perception and multi-robot observation scenarios. This paper describes the data collection methodology, sensor configurations, dataset organization, and intended research tasks supported by the dataset.
>
---
#### [new 009] MistyPilot: An Agentic Fast-Slow Thinking LLM Framework for Misty Social Robots
- **分类: cs.RO**

- **简介: 该论文提出MistyPilot框架，解决社交机器人用户指令理解与工具配置难题。属于智能交互任务，通过双代理机制提升任务执行效率与情感对齐。**

- **链接: [https://arxiv.org/pdf/2603.03640](https://arxiv.org/pdf/2603.03640)**

> **作者:** Xiao Wang; Lu Dong; Jingchen Sun; Ifeoma Nwogu; Srirangaraj Setlur; Venu Govindaraju
>
> **摘要:** With the availability of open APIs in social robots, it has become easier to customize general-purpose tools to meet users' needs. However, interpreting high-level user instructions, selecting and configuring appropriate tools, and executing them reliably remain challenging for users without programming experience. To address these challenges, we introduce MistyPilot, an agentic LLM-driven framework for autonomous tool selection, orchestration, and parameter configuration. MistyPilot comprises two core components: a Physically Interactive Agent (PIA) and a Socially Intelligent Agent (SIA). The PIA enables robust sensor-triggered and tool-driven task execution, while the SIA generates socially intelligent and emotionally aligned dialogue. MistyPilot further integrates a fast-slow thinking paradigm to capture user preferences, reduce latency, and improve task efficiency. To comprehensively evaluate MistyPilot, we contribute five benchmark datasets. Extensive experiments demonstrate the effectiveness of our framework in routing correctness, task completeness, fast-slow thinking retrieval efficiency, tool scalability,and emotion alignment. All code, datasets, and experimental videos will be made publicly available on the project webpage.
>
---
#### [new 010] HBRB-BoW: A Retrained Bag-of-Words Vocabulary for ORB-SLAM via Hierarchical BRB-KMeans
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决ORB-SLAM中二值化词袋词汇精度不足的问题。通过改进的分层BRB-KMeans方法，提升视觉词典的表达能力。**

- **链接: [https://arxiv.org/pdf/2603.04144](https://arxiv.org/pdf/2603.04144)**

> **作者:** Minjae Lee; Sang-Min Choi; Gun-Woo Kim; Suwon Lee
>
> **摘要:** In visual simultaneous localization and mapping (SLAM), the quality of the visual vocabulary is fundamental to the system's ability to represent environments and recognize locations. While ORB-SLAM is a widely used framework, its binary vocabulary, trained through the k-majority-based bag-of-words (BoW) approach, suffers from inherent precision loss. The inability of conventional binary clustering to represent subtle feature distributions leads to the degradation of visual words, a problem that is compounded as errors accumulate and propagate through the hierarchical tree structure. To address these structural deficiencies, this paper proposes hierarchical binary-to-real-and-back (HBRB)-BoW, a refined hierarchical binary vocabulary training algorithm. By integrating a global real-valued flow within the hierarchical clustering process, our method preserves high-fidelity descriptor information until the final binarization at the leaf nodes. Experimental results demonstrate that the proposed approach yields a more discriminative and well-structured vocabulary than traditional methods, significantly enhancing the representational integrity of the visual dictionary in complex environments. Furthermore, replacing the default ORB-SLAM vocabulary file with our HBRB-BoW file is expected to improve performance in loop closing and relocalization tasks.
>
---
#### [new 011] Sampling-Based Motion Planning with Scene Graphs Under Perception Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决高自由度机器人在多目标感知约束下的路径规划问题。通过引入场景图和感知成本，提升目标检测与跟踪效果。**

- **链接: [https://arxiv.org/pdf/2603.03514](https://arxiv.org/pdf/2603.03514)**

> **作者:** Qingxi Meng; Emiliano Flores; Thai Duong; Vaibhav Unhelkar; Lydia E. Kavraki
>
> **备注:** 8 pages, 5 figures, Accepted to R-AL
>
> **摘要:** It will be increasingly common for robots to operate in cluttered human-centered environments such as homes, workplaces, and hospitals, where the robot is often tasked to maintain perception constraints, such as monitoring people or multiple objects, for safety and reliability while executing its task. However, existing perception-aware approaches typically focus on low-degree-of-freedom (DoF) systems or only consider a single object in the context of high-DoF robots. This motivates us to consider the problem of perception-aware motion planning for high-DoF robots that accounts for multi-object monitoring constraints. We employ a scene graph representation of the environment, offering a great potential for incorporating long-horizon task and motion planning thanks to its rich semantic and spatial information. However, it does not capture perception-constrained information, such as the viewpoints the user prefers. To address these challenges, we propose MOPS-PRM, a roadmap-based motion planner, that integrates the perception cost of observing multiple objects or humans directly into motion planning for high-DoF robots. The perception cost is embedded to each object as part of a scene graph, and used to selectively sample configurations for roadmap construction, implicitly enforcing the perception constraints. Our method is extensively validated in both simulated and real-world experiments, achieving more than ~36% improvement in the average number of detected objects and ~17% better track rate against other perception-constrained baselines, with comparable planning times and path lengths.
>
---
#### [new 012] Overlapping Domain Decomposition for Distributed Pose Graph Optimization
- **分类: cs.RO**

- **简介: 该论文属于多机器人位姿图优化任务，解决分布式优化中的通信与计算效率问题。提出ROBO方法，通过重叠域分解实现高效收敛，减少迭代次数。**

- **链接: [https://arxiv.org/pdf/2603.03499](https://arxiv.org/pdf/2603.03499)**

> **作者:** Aneesa Sonawalla; Yulun Tian; Jonathan P. How
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** We present ROBO (Riemannian Overlapping Block Optimization), a distributed and parallel approach to multi-robot pose graph optimization (PGO) based on the idea of overlapping domain decomposition. ROBO offers a middle ground between centralized and fully distributed solvers, where the amount of pose information shared between robots at each optimization iteration can be set according to the available communication resources. Sharing additional pose information between neighboring robots effectively creates overlapping optimization blocks in the underlying pose graph, which substantially reduces the number of iterations required to converge. Through extensive experiments on benchmark PGO datasets, we demonstrate the applicability and feasibility of ROBO in different initialization scenarios, using various cost functions, and under different communication regimes. We also analyze the tradeoff between the increased communication and local computation required by ROBO's overlapping blocks and the resulting faster convergence. We show that overlaps with an average inter-robot data cost of only 36 Kb per iteration can converge 3.1$\times$ faster in terms of iterations than state-of-the-art distributed PGO approaches. Furthermore, we develop an asynchronous variant of ROBO that is robust to network delays and suitable for real-world robotic applications.
>
---
#### [new 013] Structural Action Transformer for 3D Dexterous Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决高自由度机械手跨实体技能迁移问题。提出结构化动作Transformer，通过3D结构视角提升样本效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03960](https://arxiv.org/pdf/2603.03960)**

> **作者:** Xiaohan Lei; Min Wang; Bohong Weng; Wengang Zhou; Houqiang Li
>
> **备注:** Accepted by CVPR
>
> **摘要:** Achieving human-level dexterity in robots via imitation learning from heterogeneous datasets is hindered by the challenge of cross-embodiment skill transfer, particularly for high-DoF robotic hands. Existing methods, often relying on 2D observations and temporal-centric action representation, struggle to capture 3D spatial relations and fail to handle embodiment heterogeneity. This paper proposes the Structural Action Transformer (SAT), a new 3D dexterous manipulation policy that challenges this paradigm by introducing a structural-centric perspective. We reframe each action chunk not as a temporal sequence, but as a variable-length, unordered sequence of joint-wise trajectories. This structural formulation allows a Transformer to natively handle heterogeneous embodiments, treating the joint count as a variable sequence length. To encode structural priors and resolve ambiguity, we introduce an Embodied Joint Codebook that embeds each joint's functional role and kinematic properties. Our model learns to generate these trajectories from 3D point clouds via a continuous-time flow matching objective. We validate our approach by pre-training on large-scale heterogeneous datasets and fine-tuning on simulation and real-world dexterous manipulation tasks. Our method consistently outperforms all baselines, demonstrating superior sample efficiency and effective cross-embodiment skill transfer. This structural-centric representation offers a new path toward scaling policies for high-DoF, heterogeneous manipulators.
>
---
#### [new 014] Gaussian Mixture-Based Inverse Perception Contract for Uncertainty-Aware Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决感知不确定性建模问题。提出GM-IPC，用高斯混合模型表示多模态不确定性，提升导航安全性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.04329](https://arxiv.org/pdf/2603.04329)**

> **作者:** Bingyao Du; Joonkyung Kim; Yiwei Lyu
>
> **备注:** 8 pages, 5 figures. Accepted to ACC 2026 (American Control Conference)
>
> **摘要:** Reliable navigation in cluttered environments requires perception outputs that are not only accurate but also equipped with uncertainty sets suitable for safe control. An inverse perception contract (IPC) provides such a connection by mapping perceptual estimates to sets that contain the ground truth with high confidence. Existing IPC formulations, however, instantiate uncertainty as a single ellipsoidal set and rely on deterministic trust scores to guide robot motion. Such a representation cannot capture the multi-modal and irregular structure of fine-grained perception errors, often resulting in over-conservative sets and degraded navigation performance. In this work, we introduce Gaussian Mixture-based Inverse Perception Contract (GM-IPC), which extends IPC to represent uncertainty with unions of ellipsoidal confidence sets derived from Gaussian mixture models. This design moves beyond deterministic single-set abstractions, enabling fine-grained, multi-modal, and non-convex error structures to be captured with formal guarantees. A learning framework is presented that trains GM-IPC to account for probabilistic inclusion, distribution matching, and empty-space penalties, ensuring both validity and compactness of the predicted sets. We further show that the resulting uncertainty characterizations can be leveraged in downstream planning frameworks for real-time safe navigation, enabling less conservative and more adaptive robot motion while preserving safety in a probabilistic manner.
>
---
#### [new 015] PRAM-R: A Perception-Reasoning-Action-Memory Framework with LLM-Guided Modality Routing for Adaptive Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PRAM-R框架，解决自动驾驶中多模态感知的计算效率问题。通过LLM引导的模态路由和分层记忆，实现高效、自适应的感知与决策。**

- **链接: [https://arxiv.org/pdf/2603.04222](https://arxiv.org/pdf/2603.04222)**

> **作者:** Yi Zhang; Xian Zhang; Saisi Zhao; Yinglei Song; Chengdong Wu; Nenad Petrovic; Alois Knoll
>
> **摘要:** Multimodal perception enables robust autonomous driving but incurs unnecessary computational cost when all sensors remain active. This paper presents PRAM-R, a unified Perception-Reasoning-Action-Memory framework with LLM-Guided Modality Routing for adaptive autonomous driving. PRAM-R adopts an asynchronous dual-loop design: a fast reactive loop for perception and control, and a slow deliberative loop for reasoning-driven modality selection and memory updates. An LLM router selects and weights modalities using environmental context and sensor diagnostics, while a hierarchical memory module preserves temporal consistency and supports long-term adaptation. We conduct a two-stage evaluation: (1) synthetic stress tests for stability analysis and (2) real-world validation on the nuScenes dataset. Synthetic stress tests confirm 87.2% reduction in routing oscillations via hysteresis-based stabilization. Real-world validation on nuScenes shows 6.22% modality reduction with 20% memory recall while maintaining comparable trajectory accuracy to full-modality baselines in complex urban scenarios. Our work demonstrates that LLM-augmented architectures with hierarchical memory achieve efficient, adaptive multimodal perception in autonomous driving.
>
---
#### [new 016] ArthroCut: Autonomous Policy Learning for Robotic Bone Resection in Knee Arthroplasty
- **分类: cs.RO**

- **简介: 该论文属于骨科手术机器人任务，旨在提升手术机器人的自主决策能力。通过构建多模态数据集并引入特定令牌，实现精准的骨切除操作。**

- **链接: [https://arxiv.org/pdf/2603.03957](https://arxiv.org/pdf/2603.03957)**

> **作者:** Xu Lu; Yiling Zhang; Wenquan Cheng; Longfei Ma; Fang Chen; Hongen Liao
>
> **备注:** Accepted for publication at the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Despite rapid commercialization of surgical robots, their autonomy and real-time decision-making remain limited in practice. To address this gap, we propose ArthroCut, an autonomous policy learning framework that upgrades knee arthroplasty robots from assistive execution to context-aware action generation. ArthroCut fine-tunes a Qwen--VL backbone on a self-built, time-synchronized multimodal dataset from 21 complete cases (23,205 RGB--D pairs), integrating preoperative CT/MR, intraoperative NDI tracking of bones and end effector, RGB--D surgical video, robot state, and textual intent. The method operates on two complementary token families -- Preoperative Imaging Tokens (PIT) to encode patient-specific anatomy and planned resection planes, and Time-Aligned Surgical Tokens (TAST) to fuse real-time visual, geometric, and kinematic evidence -- and emits an interpretable action grammar under grammar/safety-constrained decoding. In bench-top experiments on a knee prosthesis across seven trials, ArthroCut achieves an average success rate of 86% over the six standard resections, significantly outperforming strong baselines trained under the same protocol. Ablations show that TAST is the principal driver of reliability while PIT provides essential anatomical grounding, and their combination yields the most stable multi-plane execution. These results indicate that aligning preoperative geometry with time-aligned intraoperative perception and translating that alignment into tokenized, constrained actions is an effective path toward robust, interpretable autonomy in orthopedic robotic surgery.
>
---
#### [new 017] Whole-Body Safe Control of Robotic Systems with Koopman Neural Dynamics
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决非线性系统实时安全控制问题。通过学习Koopman模型并结合安全集算法，实现轨迹跟踪与避障的统一优化。**

- **链接: [https://arxiv.org/pdf/2603.03740](https://arxiv.org/pdf/2603.03740)**

> **作者:** Sebin Jung; Abulikemu Abuduweili; Jiaxing Li; Changliu Liu
>
> **摘要:** Controlling robots with strongly nonlinear, high-dimensional dynamics remains challenging, as direct nonlinear optimization with safety constraints is often intractable in real time. The Koopman operator offers a way to represent nonlinear systems linearly in a lifted space, enabling the use of efficient linear control. We propose a data-driven framework that learns a Koopman embedding and operator from data, and integrates the resulting linear model with the Safe Set Algorithm (SSA). This allows the tracking and safety constraints to be solved in a single quadratic program (QP), ensuring feasibility and optimality without a separate safety filter. We validate the method on a Kinova Gen3 manipulator and a Go2 quadruped, showing accurate tracking and obstacle avoidance.
>
---
#### [new 018] Navigating in Uncertain Environments with Heterogeneous Visibility
- **分类: cs.RO**

- **简介: 该论文研究不确定环境中导航任务，解决如何在路径成本与信息获取间平衡的问题。提出一种新算法，通过优化观察奖励与路径成本，提升导航效率。**

- **链接: [https://arxiv.org/pdf/2603.03495](https://arxiv.org/pdf/2603.03495)**

> **作者:** Jongann Lee; Melkior Ornik
>
> **摘要:** Navigating an environment with uncertain connectivity requires a strategic balance between minimizing the cost of traversal and seeking information to resolve map ambiguities. Unlike previous approaches that rely on local sensing, we utilize a framework where nodes possess varying visibility levels, allowing for observation of distant edges from certain vantage points. We propose a novel heuristic algorithm that balances the cost of detouring to high-visibility locations against the gain in information by optimizing the sum of a custom observation reward and the cost of traversal. We introduce a technique to sample the shortest path on numerous realizations of the environment, which we use to define an edge's utility for observation and to quickly estimate the path with the highest reward. Our approach can be easily adapted to a variety of scenarios by tuning a single hyperparameter that determines the importance of observation. We test our method on a variety of uncertain navigation tasks, including a map based on real-world topographical data. The method demonstrates lower mean cost of traversal compared to a shortest path baseline that does not consider observation and has exponentially lower computational overhead compared to an existing method for balancing observation with path cost minimization.
>
---
#### [new 019] Characterization and Correlation of Robotic Snake Scale Friction and Locomotion Speed
- **分类: cs.RO**

- **简介: 该论文属于机器人学任务，旨在研究仿生蛇形机器人运动与鳞片摩擦的关系。通过设计新型鳞片结构并测量不同角度和表面的摩擦特性，分析其对运动速度的影响。**

- **链接: [https://arxiv.org/pdf/2603.03735](https://arxiv.org/pdf/2603.03735)**

> **作者:** Umit Sen; Andri Mahegan; Gina Olson
>
> **备注:** Accepted for 9th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2026), 8 pages, 7 figures
>
> **摘要:** Snake robots are inspired by the ability of biological snakes to move over rock, grass, leaves, soil, up trees, along pavement and more. Their ability to move in multiple distinct environments is due to their legless locomotion strategy, which combines distinct gaits with a skin that exhibits frictional anisotropy. Designing soft robotic snakes with similar capabilities requires an understanding of how this underlying frictional anisotropy should be created in engineered systems, and how variances in the frictional anisotropy ratio affect locomotion speed and direction on different surfaces. While forward and backward frictional ratios have been characterized for previous scale designs, lateral friction and the associated ratios are often overlooked. In this paper, our contributions include: (i) the development of a novel articulated pseudo-skin design that is modular, easy to construct and has removable or replaceable scales; (ii) experimental measurement of the frictional characteristics of otherwise-identical scales at varying angles of attack (15°, 25°, 35°, 45°) on different surfaces of interest (grass, bark, smooth surface, carpet);(iii) separate measurements of locomotion speed for each angle and surface. Consequently, while we observed some consistent trends between frictional coefficients and scale angle, aligning with literature and intuition, we were not able to consistently identify expected correlations between frictional ratios and locomotion speed. We conclude that either frictional ratios alone are not sufficient to predict the observed speed of a snake robot, or that specific measurement approaches are required to accurately capture these ratios.
>
---
#### [new 020] Multi-Agent-Based Simulation of Archaeological Mobility in Uneven Landscapes
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于考古学模拟任务，旨在解决静态证据难以重建古代移动与互动的问题。通过多智能体模型，结合地形与代理特性，模拟不同运输方式在复杂地形中的移动效果。**

- **链接: [https://arxiv.org/pdf/2603.03390](https://arxiv.org/pdf/2603.03390)**

> **作者:** Chairi Kiourt; Vassilis Evangelidis; Dimitris Grigoropoulos
>
> **摘要:** Understanding mobility, movement, and interaction in archaeological landscapes is essential for interpreting past human behavior, transport strategies, and spatial organization, yet such processes are difficult to reconstruct from static archaeological evidence alone. This paper presents a multi-agent-based modeling framework for simulating archaeological mobility in uneven landscapes, integrating realistic terrain reconstruction, heterogeneous agent modeling, and adaptive navigation strategies. The proposed approach combines global path planning with local dynamic adaptation, through reinforcment learning, enabling agents to respond efficiently to dynamic obstacles and interactions without costly global replanning. Real-world digital elevation data are processed into high-fidelity three-dimensional environments, preserving slope and terrain constraints that directly influence agent movement. The framework explicitly models diverse agent types, including human groups and animal-based transport systems, each parameterized by empirically grounded mobility characteristics such as load, slope tolerance, and physical dimensions. Two archaeological-inspired use cases demonstrate the applicability of the approach: a terrain-aware pursuit and evasion scenario and a comparative transport analysis involving pack animals and wheeled carts. The results highlight the impact of terrain morphology, visibility, and agent heterogeneity on movement outcomes, while the proposed hybrid navigation strategy provides a computationally efficient and interpretable solution for large-scale, dynamic archaeological simulations.
>
---
#### [new 021] RoboLight: A Dataset with Linearly Composable Illumination for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出RoboLight数据集，用于机器人操作中的光照研究。解决光照变化对感知和控制的影响问题，通过真实与合成数据增强任务鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04249](https://arxiv.org/pdf/2603.04249)**

> **作者:** Shutong Jin; Jin Yang; Muhammad Zahid; Florian T. Pokorny
>
> **摘要:** In this paper, we introduce RoboLight, the first real-world robotic manipulation dataset capturing synchronized episodes under systematically varied lighting conditions. RoboLight consists of two components. (a) RoboLight-Real contains 2,800 real-world episodes collected in our custom Light Cube setup, a calibrated system equipped with eight programmable RGB LED lights. It includes structured illumination variation along three independently controlled dimensions: color, direction, and intensity. Each dimension is paired with a dedicated task featuring objects of diverse geometries and materials to induce perceptual challenges. All image data are recorded in high-dynamic-range (HDR) format to preserve radiometric accuracy. Leveraging the linearity of light transport, we introduce (b) RoboLight-Synthetic, comprising 196,000 episodes synthesized through interpolation in the HDR image space of RoboLight-Real. In principle, RoboLight-Synthetic can be arbitrarily expanded by refining the interpolation granularity. We further verify the dataset quality through qualitative analysis and real-world policy roll-outs, analyzing task difficulty, distributional diversity, and the effectiveness of synthesized data. We additionally demonstrate three representative use cases of the proposed dataset. The full dataset, along with the system software and hardware design, will be released as open-source to support continued research.
>
---
#### [new 022] SaFeR: Safety-Critical Scenario Generation for Autonomous Driving Test via Feasibility-Constrained Token Resampling
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶安全测试任务，解决生成安全关键场景时的冲突目标平衡问题。提出SaFeR方法，通过约束采样生成高真实性和可行性的场景。**

- **链接: [https://arxiv.org/pdf/2603.04071](https://arxiv.org/pdf/2603.04071)**

> **作者:** Jinlong Cui; Fenghua Liang; Guo Yang; Chengcheng Tang; Jianxun Cui
>
> **摘要:** Safety-critical scenario generation is crucial for evaluating autonomous driving systems. However, existing approaches often struggle to balance three conflicting objectives: adversarial criticality, physical feasibility, and behavioral realism. To bridge this gap, we propose SaFeR: safety-critical scenario generation for autonomous driving test via feasibility-constrained token resampling. We first formulate traffic generation as a discrete next token prediction problem, employing a Transformer-based model as a realism prior to capture naturalistic driving distributions. To capture complex interactions while effectively mitigating attention noise, we propose a novel differential attention mechanism within the realism prior. Building on this prior, SaFeR implements a novel resampling strategy that induces adversarial behaviors within a high-probability trust region to maintain naturalism, while enforcing a feasibility constraint derived from the Largest Feasible Region (LFR). By approximating the LFR via offline reinforcement learning, SaFeR effectively prevents the generation of theoretically inevitable collisions. Closed-loop experiments on the Waymo Open Motion Dataset and nuPlan demonstrate that SaFeR significantly outperforms state-of-the-art baselines, achieving a higher solution rate and superior kinematic realism while maintaining strong adversarial effectiveness.
>
---
#### [new 023] Perception-Aware Time-Optimal Planning for Quadrotor Waypoint Flight
- **分类: cs.RO**

- **简介: 该论文属于无人机路径规划任务，解决视觉感知与动态最优轨迹的耦合问题。通过引入感知约束，提升轨迹执行的可靠性与速度。**

- **链接: [https://arxiv.org/pdf/2603.04305](https://arxiv.org/pdf/2603.04305)**

> **作者:** Chao Qin; Jiaxu Xing; Rudolf Reiter; Angel Romero; Yifan Lin; Hugh H.-T. Liu; Davide Scaramuzza
>
> **摘要:** Agile quadrotor flight pushes the limits of control, actuation, and onboard perception. While time-optimal trajectory planning has been extensively studied, existing approaches typically neglect the tight coupling between vehicle dynamics, environmental geometry, and the visual requirements of onboard state estimation. As a result, trajectories that are dynamically feasible may fail in closed-loop execution due to degraded visual quality. This paper introduces a unified time-optimal trajectory optimization framework for vision-based quadrotors that explicitly incorporates perception constraints alongside full nonlinear dynamics, rotor actuation limits, aerodynamic effects, camera field-of-view constraints, and convex geometric gate representations. The proposed formulation solves minimum-time lap trajectories for arbitrary racetracks with diverse gate shapes and orientations, while remaining numerically robust and computationally efficient. We derive an information-theoretic position uncertainty metric to quantify visual state-estimation quality and integrate it into the planner through three perception objectives: position uncertainty minimization, sequential field-of-view constraints, and look-ahead alignment. This enables systematic exploration of the trade-offs between speed and perceptual reliability. To accurately track the resulting perception-aware trajectories, we develop a model predictive contouring tracking controller that separates lateral and progress errors. Experiments demonstrate real-world flight speeds up to 9.8 m/s with 0.07 m average tracking error, and closed-loop success rates improved from 55% to 100% on a challenging Split-S course. The proposed system provides a scalable benchmark for studying the fundamental limits of perception-aware, time-optimal autonomous flight.
>
---
#### [new 024] Compliant In-hand Rolling Manipulation Using Tactile Sensing
- **分类: cs.RO**

- **简介: 该论文研究多指机器人手的柔性滚动操作任务，通过触觉传感实现物体的精确控制，解决滚动过程中的接触力学与控制问题。**

- **链接: [https://arxiv.org/pdf/2603.04301](https://arxiv.org/pdf/2603.04301)**

> **作者:** Huan Weng; Yifei Chen; Kevin M. Lynch
>
> **摘要:** We investigate in-hand rolling manipulation using a multifingered robot hand, where each finger is compliant and equipped with a tactile fingertip providing contact location and wrench information. We derive the equations of motion for compliant quasistatic in-hand rolling manipulation and formulate a fingertip rolling manipulation controller for multiple fingers to achieve a desired object twist within a grasp. The contact mechanics are demonstrated in simulation and the controller is tested on an experimental robot system.
>
---
#### [new 025] Force-Aware Residual DAgger via Trajectory Editing for Precision Insertion with Impedance Control
- **分类: cs.RO**

- **简介: 该论文针对接触丰富的精密插入任务，解决模仿学习中的协变量偏移和持续专家监控问题，提出TER-DAgger框架，通过轨迹编辑和力感知机制提升成功率。**

- **链接: [https://arxiv.org/pdf/2603.04038](https://arxiv.org/pdf/2603.04038)**

> **作者:** Yiou Huang; Ma Ning; Weichu Zhao; Zinuo Liu; Jun Sun; Qiufeng Wang; Yaran Chen
>
> **摘要:** Imitation learning (IL) has shown strong potential for contact-rich precision insertion tasks. However, its practical deployment is often hindered by covariate shift and the need for continuous expert monitoring to recover from failures during execution. In this paper, we propose Trajectory Editing Residual Dataset Aggregation (TER-DAgger), a scalable and force-aware human-in-the-loop imitation learning framework that mitigates covariate shift by learning residual policies through optimization-based trajectory editing. This approach smoothly fuses policy rollouts with human corrective trajectories, providing consistent and stable supervision. Second, we introduce a force-aware failure anticipation mechanism that triggers human intervention only when discrepancies arise between predicted and measured end-effector forces, significantly reducing the requirement for continuous expert monitoring. Third, all learned policies are executed within a Cartesian impedance control framework, ensuring compliant and safe behavior during contact-rich interactions. Extensive experiments in both simulation and real-world precision insertion tasks show that TER-DAgger improves the average success rate by over 37\% compared to behavior cloning, human-guided correction, retraining, and fine-tuning baselines, demonstrating its effectiveness in mitigating covariate shift and enabling scalable deployment in contact-rich manipulation.
>
---
#### [new 026] UrbanHuRo: A Two-Layer Human-Robot Collaboration Framework for the Joint Optimization of Heterogeneous Urban Services
- **分类: cs.RO; cs.AI; cs.HC; cs.SI**

- **简介: 该论文属于城市服务优化任务，旨在解决异质服务协同优化问题。通过人机协作框架UrbanHuRo，提升感知覆盖与配送效率。**

- **链接: [https://arxiv.org/pdf/2603.03701](https://arxiv.org/pdf/2603.03701)**

> **作者:** Tonmoy Dey; Lin Jiang; Zheng Dong; Guang Wang
>
> **备注:** 8 pages, 15 figures. This paper has been accepted by ICRA'26 as a regular paper
>
> **摘要:** In the vision of smart cities, technologies are being developed to enhance the efficiency of urban services and improve residents' quality of life. However, most existing research focuses on optimizing individual services in isolation, without adequately considering reciprocal interactions among heterogeneous urban services that could yield higher efficiency and improved resource utilization. For example, human couriers could collect traffic and air quality data along their delivery routes, while sensing robots could assist with on-demand delivery during peak hours, enhancing both sensing coverage and delivery efficiency. However, the joint optimization of different urban services is challenging due to potentially conflicting objectives and the need for real-time coordination in dynamic environments. In this paper, we propose UrbanHuRo, a two-layer human-robot collaboration framework for joint optimization of heterogeneous urban services, demonstrated through crowdsourced delivery and urban sensing. UrbanHuRo includes two key designs: (i) a scalable distributed MapReduce-based K-submodular maximization module for efficient order dispatch, and (ii) a deep submodular reward reinforcement learning algorithm for sensing route planning. Experimental evaluations on real-world datasets from a food delivery platform demonstrate that UrbanHuRo improves sensing coverage by 29.7% and courier income by 39.2% on average in most settings, while also significantly reducing the number of overdue orders.
>
---
#### [new 027] OmniPlanner: Universal Exploration and Inspection Path Planning across Robot Morphologies
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决跨平台探索与检测问题。提出OmniPlanner框架，实现空中、地面和水下机器人的统一路径规划，提升通用性与效率。**

- **链接: [https://arxiv.org/pdf/2603.04284](https://arxiv.org/pdf/2603.04284)**

> **作者:** Angelos Zacharia; Mihir Dharmadhikari; Mohit Singh; Kostas Alexis
>
> **备注:** The code for this paper is open-sourced and released at: this https URL
>
> **摘要:** Autonomous robotic systems are increasingly deployed for mapping, monitoring, and inspection in complex and unstructured environments. However, most existing path planning approaches remain domain-specific (i.e., either on air, land, or sea), limiting their scalability and cross-platform applicability. This article presents OmniPlanner, a unified planning framework for autonomous exploration and inspection across aerial, ground, and underwater robots. The method integrates volumetric exploration and viewpoint-based inspection, alongside target reach behaviors within a single modular architecture, complemented by a platform abstraction layer that captures morphology-specific sensing, traversability and motion constraints. This enables the same planning strategy to generalize across distinct mobility domains with minimal retuning. The framework is validated through extensive simulation studies and field deployments in underground mines, industrial facilities, forests, submarine bunkers, and structured outdoor environments. Across these diverse scenarios, OmniPlanner demonstrates robust performance, consistent cross-domain generalization, and improved exploration and inspection efficiency compared to representative state-of-the-art baselines.
>
---
#### [new 028] GarmentPile++: Affordance-Driven Cluttered Garments Retrieval with Vision-Language Reasoning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于家务机器人任务，解决杂乱衣物精准抓取问题。通过视觉-语言推理与视觉可操作性感知结合，实现安全、准确的单件衣物检索。**

- **链接: [https://arxiv.org/pdf/2603.04158](https://arxiv.org/pdf/2603.04158)**

> **作者:** Mingleyang Li; Yuran Wang; Yue Chen; Tianxing Chen; Jiaqi Liang; Zishun Shen; Haoran Lu; Ruihai Wu; Hao Dong
>
> **备注:** ICRA2026 Accepted
>
> **摘要:** Garment manipulation has attracted increasing attention due to its critical role in home-assistant robotics. However, the majority of existing garment manipulation works assume an initial state consisting of only one garment, while piled garments are far more common in real-world settings. To bridge this gap, we propose a novel garment retrieval pipeline that can not only follow language instruction to execute safe and clean retrieval but also guarantee exactly one garment is retrieved per attempt, establishing a robust foundation for the execution of downstream tasks (e.g., folding, hanging, wearing). Our pipeline seamlessly integrates vision-language reasoning with visual affordance perception, fully leveraging the high-level reasoning and planning capabilities of VLMs alongside the generalization power of visual affordance for low-level actions. To enhance the VLM's comprehensive awareness of each garment's state within a garment pile, we employ visual segmentation model (SAM2) to execute object segmentation on the garment pile for aiding VLM-based reasoning with sufficient visual cues. A mask fine-tuning mechanism is further integrated to address scenarios where the initial segmentation results are suboptimal. In addition, a dual-arm cooperation framework is deployed to address cases involving large or long garments, as well as excessive garment sagging caused by incorrect grasping point determination, both of which are strenuous for a single arm to handle. The effectiveness of our pipeline are consistently demonstrated across diverse tasks and varying scenarios in both real-world and simulation environments. Project page: this https URL.
>
---
#### [new 029] Real-time loosely coupled GNSS and IMU integration via Factor Graph Optimization
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于定位与导航任务，解决GNSS与IMU融合的实时性问题。提出基于因子图优化的松耦合架构，提升服务可用性，分析精度与效率的权衡。**

- **链接: [https://arxiv.org/pdf/2603.03546](https://arxiv.org/pdf/2603.03546)**

> **作者:** Radu-Andrei Cioaca; Cristian Rusu; Paul Irofti; Gianluca Caparra; Andrei-Alexandru Marinache; Florin Stoican
>
> **摘要:** Accurate positioning, navigation, and timing (PNT) is fundamental to the operation of modern technologies and a key enabler of autonomous systems. A very important component of PNT is the Global Navigation Satellite System (GNSS) which ensures outdoor positioning. Modern research directions have pushed the performance of GNSS localization to new heights by fusing GNSS measurements with other sensory information, mainly measurements from Inertial Measurement Units (IMU). In this paper, we propose a loosely coupled architecture to integrate GNSS and IMU measurements using a Factor Graph Optimization (FGO) framework. Because the FGO method can be computationally challenging and often used as a post-processing method, our focus is on assessing its localization accuracy and service availability while operating in real-time in challenging environments (urban canyons). Experimental results on the UrbanNav-HK-MediumUrban-1 dataset show that the proposed approach achieves real-time operation and increased service availability compared to batch FGO methods. While this improvement comes at the cost of reduced positioning accuracy, the paper provides a detailed analysis of the trade-offs between accuracy, availability, and computational efficiency that characterize real-time FGO-based GNSS/IMU fusion.
>
---
#### [new 030] VANGUARD: Vehicle-Anchored Ground Sample Distance Estimation for UAVs in GPS-Denied Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主导航任务，解决GPS缺失环境下无人机的尺度估计问题。提出VANGUARD工具，通过车辆检测恢复地面采样距离，提升感知可靠性。**

- **链接: [https://arxiv.org/pdf/2603.04277](https://arxiv.org/pdf/2603.04277)**

> **作者:** Yifei Chen; Xupeng Chen; Feng Wang; Niangang Jiao; Jiayin Liu
>
> **摘要:** Autonomous aerial robots operating in GPS-denied or communication-degraded environments frequently lose access to camera metadata and telemetry, leaving onboard perception systems unable to recover the absolute metric scale of the scene. As LLM/VLM-based planners are increasingly adopted as high-level agents for embodied systems, their ability to reason about physical dimensions becomes safety-critical -- yet our experiments show that five state-of-the-art VLMs suffer from spatial scale hallucinations, with median area estimation errors exceeding 50%. We propose VANGUARD, a lightweight, deterministic Geometric Perception Skill designed as a callable tool that any LLM-based agent can invoke to recover Ground Sample Distance (GSD) from ubiquitous environmental anchors: small vehicles detected via oriented bounding boxes, whose modal pixel length is robustly estimated through kernel density estimation and converted to GSD using a pre-calibrated reference length. The tool returns both a GSD estimate and a composite confidence score, enabling the calling agent to autonomously decide whether to trust the measurement or fall back to alternative strategies. On the DOTA~v1.5 benchmark, VANGUARD achieves 6.87% median GSD error on 306~images. Integrated with SAM-based segmentation for downstream area measurement, the pipeline yields 19.7% median error on a 100-entry benchmark -- with 2.6x lower category dependence and 4x fewer catastrophic failures than the best VLM baseline -- demonstrating that equipping agents with deterministic geometric tools is essential for safe autonomous spatial reasoning.
>
---
#### [new 031] Interaction-Aware Whole-Body Control for Compliant Object Transport
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人协作搬运任务，解决复杂环境下力控不稳定问题。提出IO-WBC方法，分离上肢交互与下肢支撑，实现稳定物理交互与合规搬运。**

- **链接: [https://arxiv.org/pdf/2603.03751](https://arxiv.org/pdf/2603.03751)**

> **作者:** Hao Zhang; Yves Tseng; Ding Zhao; H. Eric Tseng
>
> **摘要:** Cooperative object transport in unstructured environments remains challenging for assistive humanoids because strong, time-varying interaction forces can make tracking-centric whole-body control unreliable, especially in close-contact support tasks. This paper proposes a bio-inspired, interaction-oriented whole-body control (IO-WBC) that functions as an artificial cerebellum - an adaptive motor agent that translates upstream (skill-level) commands into stable, physically consistent whole-body behavior under contact. This work structurally separates upper-body interaction execution from lower-body support control, enabling the robot to maintain balance while shaping force exchange in a tightly coupled robot-object system. A trajectory-optimized reference generator (RG) provides a kinematic prior, while a reinforcement learning (RL) policy governs body responses under heavy-load interactions and disturbances. The policy is trained in simulation with randomized payload mass/inertia and external perturbations, and deployed via asymmetric teacher-student distillation so that the student relies only on proprioceptive histories at runtime. Extensive experiments demonstrate that IO-WBC maintains stable whole-body behavior and physical interaction even when precise velocity tracking becomes infeasible, enabling compliant object transport across a wide range of scenarios.
>
---
#### [new 032] Sim2Sea: Sim-to-Real Policy Transfer for Maritime Vessel Navigation in Congested Waters
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主船舶导航任务，解决复杂水域中模拟到现实的策略迁移问题。提出Sim2Sea框架，通过仿真加速、双流策略和领域随机化提升导航安全与迁移效果。**

- **链接: [https://arxiv.org/pdf/2603.04057](https://arxiv.org/pdf/2603.04057)**

> **作者:** Xinyu Cui; Xuanfa Jin; Xue Yan; Yongcheng Zeng; Luoyang Sun; Siying Wei; Ruizhi Zhang; Jian Zhao; Haifeng Zhang; Jun Wang
>
> **摘要:** Autonomous navigation in congested maritime environments is a critical capability for a wide range of real-world applications. However, it remains an unresolved challenge due to complex vessel interactions and significant environmental uncertainties. Existing methods often fail in practical deployment due to a substantial sim-to-real gap, which stems from imprecise simulation, inadequate situational awareness, and unsafe exploration strategies. To address these, we propose \textbf{Sim2Sea}, a comprehensive framework designed to bridge simulation and real-world execution. Sim2Sea advances in three key aspects. First, we develop a GPU-accelerated parallel simulator for scalable and accurate maritime scenario simulation. Second, we design a dual-stream spatiotemporal policy that handles complex dynamics and multi-modal perception, augmented with a velocity-obstacle-guided action masking mechanism to ensure safe and efficient exploration. Finally, a targeted domain randomization scheme helps bridge the sim-to-real gap. Simulation results demonstrate that our method achieves faster convergence and safer trajectories than established baselines. In addition, our policy trained purely in simulation successfully transfers zero-shot to a 17-ton unmanned vessel operating in real-world congested waters. These results validate the effectiveness of Sim2Sea in achieving reliable sim-to-real transfer for practical autonomous maritime navigation.
>
---
#### [new 033] Swimming Under Constraints: A Safe Reinforcement Learning Framework for Quadrupedal Bio-Inspired Propulsion
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决水下四足机器人推进中的稳定性问题。通过安全强化学习框架，优化推力并减少扰动，提升运动效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.04073](https://arxiv.org/pdf/2603.04073)**

> **作者:** Xinyu Cui; Fei Han; Hang Xu; Yongcheng Zeng; Luoyang Sun; Ruizhi Zhang; Jian Zhao; Haifeng Zhang; Weikun Li; Hao Chen; Jun Wang; Dixia Fan
>
> **摘要:** Bio-inspired aquatic propulsion offers high thrust and maneuverability but is prone to destabilizing forces such as lift fluctuations, which are further amplified by six-degree-of-freedom (6-DoF) fluid coupling. We formulate quadrupedal swimming as a constrained optimization problem that maximizes forward thrust while minimizing destabilizing fluctuations. Our proposed framework, Accelerated Constrained Proximal Policy Optimization with a PID-regulated Lagrange multiplier (ACPPO-PID), enforces constraints with a PID-regulated Lagrange multiplier, accelerates learning via conditional asymmetric clipping, and stabilizes updates through cycle-wise geometric aggregation. Initialized with imitation learning and refined through on-hardware towing-tank experiments, ACPPO-PID produces control policies that transfer effectively to quadrupedal free-swimming trials. Results demonstrate improved thrust efficiency, reduced destabilizing forces, and faster convergence compared with state-of-the-art baselines, underscoring the importance of constraint-aware safe RL for robust and generalizable bio-inspired locomotion in complex fluid environments.
>
---
#### [new 034] Real-time tightly coupled GNSS and IMU integration via Factor Graph Optimization
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于定位任务，解决城市环境中GNSS信号弱的问题，通过实时紧耦合GNSS与IMU的因子图优化方法提升定位可靠性与精度。**

- **链接: [https://arxiv.org/pdf/2603.03556](https://arxiv.org/pdf/2603.03556)**

> **作者:** Radu-Andrei Cioaca; Paul Irofti; Cristian Rusu; Gianluca Caparra; Andrei-Alexandru Marinache; Florin Stoican
>
> **摘要:** Reliable positioning in dense urban environments remains challenging due to frequent GNSS signal blockage, multipath, and rapidly varying satellite geometry. While factor graph optimization (FGO)-based GNSS-IMU fusion has demonstrated strong robustness and accuracy, most formulations remain offline. In this work, we present a real-time tightly coupled GNSS-IMU FGO method that enables causal state estimation via incremental optimization with fixed-lag marginalization, and we evaluate its performance in a highly urbanized GNSS-degraded environment using the UrbanNav dataset.
>
---
#### [new 035] Impact of Localization Errors on Label Quality for Online HD Map Construction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于在线高精地图构建任务，研究定位误差对标签质量的影响。通过引入不同噪声类型，分析其对模型性能的影响，并提出基于距离的评估指标。**

- **链接: [https://arxiv.org/pdf/2603.03452](https://arxiv.org/pdf/2603.03452)**

> **作者:** Alexander Blumberg; Jonas Merkert; Richard Fehler; Fabian Immel; Frank Bieder; Jan-Hendrik Pauls; Christoph Stiller
>
> **备注:** Accepted for the 36th IEEE Intelligent Vehicles Symposium (IV 2025), 8 pages
>
> **摘要:** High-definition (HD) maps are crucial for autonomous vehicles, but their creation and maintenance is very costly. This motivates the idea of online HD map construction. To provide a continuous large-scale stream of training data, existing HD maps can be used as labels for onboard sensor data from consumer vehicle fleets. However, compared to current, well curated HD map perception datasets, this fleet data suffers from localization errors, resulting in distorted map labels. We introduce three kinds of localization errors, Ramp, Gaussian, and Perlin noise, to examine their influence on generated map labels. We train a variant of MapTRv2, a state-of-the-art online HD map construction model, on the Argoverse 2 dataset with various levels of localization errors and assess the degradation of model performance. Since localization errors affect distant labels more severely, but are also less significant to driving performance, we introduce a distance-based map construction metric. Our experiments reveal that localization noise affects the model performance significantly. We demonstrate that errors in heading angle exert a more substantial influence than position errors, as angle errors result in a greater distortion of labels as distance to the vehicle increases. Furthermore, we can demonstrate that the model benefits from non-distorted ground truth (GT) data and that the performance decreases more than linearly with the increase in noisy data. Our study additionally provides a qualitative evaluation of the extent to which localization errors influence the construction of HD maps.
>
---
#### [new 036] Lightweight Visual Reasoning for Socially-Aware Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉推理任务，旨在提升机器人对人类行为的理解与响应能力。通过引入轻量级语言到视觉反馈模块，增强视觉语言模型在多模态人机交互中的表现。**

- **链接: [https://arxiv.org/pdf/2603.03942](https://arxiv.org/pdf/2603.03942)**

> **作者:** Alessio Galatolo; Ronald Cumbal; Alexandros Rouchitsas; Katie Winkle; Didem Gürdür Broo; Ginevra Castellano
>
> **备注:** ICRA26
>
> **摘要:** Robots operating in shared human environments must not only navigate, interact, and detect their surroundings, they must also interpret and respond to dynamic, and often unpredictable, human behaviours. Although recent advances have shown promise in enhancing robotic perception and instruction-following using Vision-Language Models (VLMs), they remain limited in addressing the complexities of multimodal human-robot interactions (HRI). Motivated by this challenge, we introduce a lightweight language-to-vision feedback module that closes the loop between an LLM and the vision encoder in VLMs. The module projects image-token hidden states through a gated Multi-Layer Perceptron (MLP) back into the encoder input, prompting a second pass that reinterprets the scene under text context. We evaluate this approach on three robotics-centred tasks: navigation in a simulated environment (Habitat), sequential scene description (Mementos-Robotics), and human-intention recognition (our HRI dataset). Results show that our method improves Qwen 2.5 (7B) by $3.3\%$ (less distance), $+0.057$ description score, and $+2.93\%$ accuracy, with less than $3\%$ extra parameters; Gemma 3 (4B) and LLaVA OV 1.5 (4B) show mixed navigation results but gains $+0.111,+0.055$ and $+10.81\%,+4.79\%$ on the latter two tasks. Code is available at this https URL
>
---
#### [new 037] ManipulationNet: An Infrastructure for Benchmarking Real-World Robot Manipulation with Physical Skill Challenges and Embodied Multimodal Reasoning
- **分类: cs.RO**

- **简介: 该论文提出ManipulationNet，用于评估机器人真实世界操作能力。解决缺乏标准基准的问题，通过标准化硬件和软件进行可复现的性能评估，涵盖物理技能和具身推理两个任务。**

- **链接: [https://arxiv.org/pdf/2603.04363](https://arxiv.org/pdf/2603.04363)**

> **作者:** Yiting Chen; Kenneth Kimble; Edward H. Adelson; Tamim Asfour; Podshara Chanrungmaneekul; Sachin Chitta; Yash Chitambar; Ziyang Chen; Ken Goldberg; Danica Kragic; Hui Li; Xiang Li; Yunzhu Li; Aaron Prather; Nancy Pollard; Maximo A. Roa-Garzon; Robert Seney; Shuo Sha; Shihefeng Wang; Yu Xiang; Kaifeng Zhang; Yuke Zhu; Kaiyu Hang
>
> **备注:** 32 pages, 8 figures
>
> **摘要:** Dexterous manipulation enables robots to purposefully alter the physical world, transforming them from passive observers into active agents in unstructured environments. This capability is the cornerstone of physical artificial intelligence. Despite decades of advances in hardware, perception, control, and learning, progress toward general manipulation systems remains fragmented due to the absence of widely adopted standard benchmarks. The central challenge lies in reconciling the variability of the real world with the reproducibility and authenticity required for rigorous scientific evaluation. To address this, we introduce ManipulationNet, a global infrastructure that hosts real-world benchmark tasks for robotic manipulation. ManipulationNet delivers reproducible task setups through standardized hardware kits, and enables distributed performance evaluation via a unified software client that delivers real-time task instructions and collects benchmarking results. As a persistent and scalable infrastructure, ManipulationNet organizes benchmark tasks into two complementary tracks: 1) the Physical Skills Track, which evaluates low-level physical interaction skills, and 2) the Embodied Reasoning Track, which tests high-level reasoning and multimodal grounding abilities. This design fosters the systematic growth of an interconnected network of real-world abilities and skills, paving the path toward general robotic manipulation. By enabling comparable manipulation research in the real world at scale, this infrastructure establishes a sustainable foundation for measuring long-term scientific progress and identifying capabilities ready for real-world deployment.
>
---
#### [new 038] MEM: Multi-Scale Embodied Memory for Vision Language Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决长时序任务中多粒度记忆的问题。提出MEM架构，结合视频与文本记忆，提升机器人执行复杂任务的能力。**

- **链接: [https://arxiv.org/pdf/2603.03596](https://arxiv.org/pdf/2603.03596)**

> **作者:** Marcel Torne; Karl Pertsch; Homer Walke; Kyle Vedder; Suraj Nair; Brian Ichter; Allen Z. Ren; Haohuan Wang; Jiaming Tang; Kyle Stachowicz; Karan Dhabalia; Michael Equi; Quan Vuong; Jost Tobias Springenberg; Sergey Levine; Chelsea Finn; Danny Driess
>
> **备注:** Website: this https URL
>
> **摘要:** Conventionally, memory in end-to-end robotic learning involves inputting a sequence of past observations into the learned policy. However, in complex multi-stage real-world tasks, the robot's memory must represent past events at multiple levels of granularity: from long-term memory that captures abstracted semantic concepts (e.g., a robot cooking dinner should remember which stages of the recipe are already done) to short-term memory that captures recent events and compensates for occlusions (e.g., a robot remembering the object it wants to pick up once its arm occludes it). In this work, our main insight is that an effective memory architecture for long-horizon robotic control should combine multiple modalities to capture these different levels of abstraction. We introduce Multi-Scale Embodied Memory (MEM), an approach for mixed-modal long-horizon memory in robot policies. MEM combines video-based short-horizon memory, compressed via a video encoder, with text-based long-horizon memory. Together, they enable robot policies to perform tasks that span up to fifteen minutes, like cleaning up a kitchen, or preparing a grilled cheese sandwich. Additionally, we find that memory enables MEM policies to intelligently adapt manipulation strategies in-context.
>
---
#### [new 039] IROSA: Interactive Robot Skill Adaptation using Natural Language
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于机器人技能适应任务，旨在通过自然语言实现机器人技能的灵活调整。工作包括提出一个框架，利用预训练语言模型选择工具，无需微调即可完成速度、轨迹和避障等操作。**

- **链接: [https://arxiv.org/pdf/2603.03897](https://arxiv.org/pdf/2603.03897)**

> **作者:** Markus Knauer; Samuel Bustamante; Thomas Eiband; Alin Albu-Schäffer; Freek Stulp; João Silvério
>
> **备注:** Accepted IEEE Robotics and Automation Letters (RA-L) journal, 8 pages, 5 figures, 3 tables, 1 listing
>
> **摘要:** Foundation models have demonstrated impressive capabilities across diverse domains, while imitation learning provides principled methods for robot skill adaptation from limited data. Combining these approaches holds significant promise for direct application to robotics, yet this combination has received limited attention, particularly for industrial deployment. We present a novel framework that enables open-vocabulary skill adaptation through a tool-based architecture, maintaining a protective abstraction layer between the language model and robot hardware. Our approach leverages pre-trained LLMs to select and parameterize specific tools for adapting robot skills without requiring fine-tuning or direct model-to-robot interaction. We demonstrate the framework on a 7-DoF torque-controlled robot performing an industrial bearing ring insertion task, showing successful skill adaptation through natural language commands for speed adjustment, trajectory correction, and obstacle avoidance while maintaining safety, transparency, and interpretability.
>
---
#### [new 040] Map-Agnostic And Interactive Safety-Critical Scenario Generation via Multi-Objective Tree Search
- **分类: cs.RO; cs.GR**

- **简介: 该论文属于自动驾驶安全验证任务，旨在生成真实且多样的危险场景。通过多目标MCTS方法，提升场景的复杂性和现实性，以测试自动驾驶系统的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.03978](https://arxiv.org/pdf/2603.03978)**

> **作者:** Wenyun Li; Zejian Deng; Chen Sun
>
> **摘要:** Generating safety-critical scenarios is essential for validating the robustness of autonomous driving systems, yet existing methods often struggle to produce collisions that are both realistic and diverse while ensuring explicit interaction logic among traffic participants. This paper presents a novel framework for traffic-flow level safety-critical scenario generation via multi-objective Monte Carlo Tree Search (MCTS). We reframe trajectory feasibility and naturalistic behavior as optimization objectives within a unified evaluation function, enabling the discovery of diverse collision events without compromising realism. A hybrid Upper Confidence Bound (UCB) and Lower Confidence Bound (LCB) search strategy is introduced to balance exploratory efficiency with risk-averse decision-making. Furthermore, our method is map-agnostic and supports interactive scenario generation with each vehicle individually powered by SUMO's microscopic traffic models, enabling realistic agent behaviors in arbitrary geographic locations imported from OpenStreetMap. We validate our approach across four high-risk accident zones in Hong Kong's complex urban environments. Experimental results demonstrate that our framework achieves an 85\% collision failure rate while generating trajectories with superior feasibility and comfort metrics. The resulting scenarios exhibit greater complexity, as evidenced by increased vehicle mileage and CO\(_2\) emissions. Our work provides a principled solution for stress testing autonomous vehicles through the generation of realistic yet infrequent corner cases at traffic-flow level.
>
---
#### [new 041] Radar-based Pose Optimization for HD Map Generation from Noisy Multi-Drive Vehicle Fleet Data
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶地图生成任务，旨在解决多车数据中的定位噪声问题。通过雷达点云对齐与位姿图优化，提升地图精度和车道边界识别效果。**

- **链接: [https://arxiv.org/pdf/2603.03453](https://arxiv.org/pdf/2603.03453)**

> **作者:** Alexander Blumberg; Jonas Merkert; Christoph Stiller
>
> **备注:** Accepted for the 37th IEEE Intelligent Vehicles Symposium (IV 2026), 7 pages
>
> **摘要:** High-definition (HD) maps are important for autonomous driving, but their manual generation and maintenance is very expensive. This motivates the usage of an automated map generation pipeline. Fleet vehicles provide sufficient sensors for map generation, but their measurements are less precise, introducing noise into the mapping pipeline. This work focuses on mitigating the localization noise component through aligning radar measurements in terms of raw radar point clouds of vehicle poses of different drives and performing pose graph optimization to produce a globally optimized solution between all drives present in the dataset. Improved poses are first used to generate a global radar occupancy map, aimed to facilitate precise on-vehicle localization. Through qualitative analysis we show contrast-rich feature clarity, focusing on omnipresent guardrail posts as the main feature type observable in the map. Second, the improved poses can be used as a basis for an existing lane boundary map generation pipeline, majorly improving map output compared to its original pure line detection based optimization approach.
>
---
#### [new 042] Modeling and Control of a Pneumatic Soft Robotic Catheter Using Neural Koopman Operators
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决软性手术导管建模与控制难题。通过引入神经网络增强的Koopman算子框架，提升控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2603.04118](https://arxiv.org/pdf/2603.04118)**

> **作者:** Yiyao Yue; Noah Barnes; Lingyun Di; Olivia Young; Ryan D. Sochol; Jeremy D. Brown; Axel Krieger
>
> **备注:** 8 pages, 6 figures. Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Catheter-based interventions are widely used for the diagnosis and treatment of cardiac diseases. Recently, robotic catheters have attracted attention for their ability to improve precision and stability over conventional manual approaches. However, accurate modeling and control of soft robotic catheters remain challenging due to their complex, nonlinear behavior. The Koopman operator enables lifting the original system data into a linear "lifted space", offering a data-driven framework for predictive control; however, manually chosen basis functions in the lifted space often oversimplify system behaviors and degrade control performance. To address this, we propose a neural network-enhanced Koopman operator framework that jointly learns the lifted space representation and Koopman operator in an end-to-end manner. Moreover, motivated by the need to minimize radiation exposure during X-ray fluoroscopy in cardiac ablation, we investigate open-loop control strategies using neural Koopman operators to reliably reach target poses without continuous imaging feedback. The proposed method is validated in two experimental scenarios: interactive position control and a simulated cardiac ablation task using an atrium-like cavity. Our approach achieves average errors of 2.1 +- 0.4 mm in position and 4.9 +- 0.6 degrees in orientation, outperforming not only model-based baselines but also other Koopman variants in targeting accuracy and efficiency. These results highlight the potential of the proposed framework for advancing soft robotic catheter systems and improving catheter-based interventions.
>
---
#### [new 043] Right in Time: Reactive Reasoning in Regulated Traffic Spaces
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于智能交通系统任务，解决自动驾驶在复杂环境中实时合规推理问题。提出一种结合概率任务设计与反应式电路的框架，实现高效精确推理。**

- **链接: [https://arxiv.org/pdf/2603.03977](https://arxiv.org/pdf/2603.03977)**

> **作者:** Simon Kohaut; Benedict Flade; Julian Eggert; Kristian Kersting; Devendra Singh Dhami
>
> **摘要:** Exact inference in probabilistic First-Order Logic offers a promising yet computationally costly approach for regulating the behavior of autonomous agents in shared traffic spaces. While prior methods have combined logical and probabilistic data into decision-making frameworks, their application is often limited to pre-flight checks due to the complexity of reasoning across vast numbers of possible universes. In this work, we propose a reactive mission design framework that jointly considers uncertain environmental data and declarative, logical traffic regulations. By synthesizing Probabilistic Mission Design (ProMis) with reactive reasoning facilitated by Reactive Circuits (RC), we enable online, exact probabilistic inference over hybrid domains. Our approach leverages the Frequency of Change inherent in heterogeneous data streams to subdivide inference formulas into memoized, isolated tasks, ensuring that only the specific components affected by new sensor data are re-evaluated. In experiments involving both real-world vessel data and simulated drone traffic in dense urban scenarios, we demonstrate that our approach provides orders of magnitude in speedup over ProMis without reactive paradigms. This allows intelligent transportation systems, such as Unmanned Aircraft Systems (UAS), to actively assert safety and legal compliance during operations rather than relying solely on preparation procedures.
>
---
#### [new 044] SkillVLA: Tackling Combinatorial Diversity in Dual-Arm Manipulation via Skill Reuse
- **分类: cs.RO**

- **简介: 该论文属于双臂操作任务，解决组合多样性问题。通过提出SkillVLA框架，实现单臂技能的复用，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.03836](https://arxiv.org/pdf/2603.03836)**

> **作者:** Xuanran Zhai; Zekai Huang; Longyan Wu; Qianyou Zhao; Qiaojun Yu; Jieji Ren; Ce Hao; Harold Soh
>
> **备注:** 16 pages
>
> **摘要:** Recent progress in vision-language-action (VLA) models has demonstrated strong potential for dual-arm manipulation, enabling complex behaviors and generalization to unseen environments. However, mainstream bimanual VLA formulations largely overlook the critical challenge of combinatorial diversity. Different pairings of single-arm behaviors can induce qualitatively distinct task behaviors, yet existing models do not explicitly account for this structure. We argue that effective bimanual VLAs should support skill reuse - the ability to recombine previously learned single-arm skills across novel left-right pairings - thereby avoiding the need to separately learn every possible combination. Current VLA designs entangle skills across arms, preventing such recomposition and limiting scalability. To address this limitation, we propose SkillVLA, a framework explicitly designed to enable skill reuse in dual-arm manipulation. Extensive experiments demonstrate that SkillVLA substantially improves skill composition, increasing overall success rate from 0% to 51%, and achieves strong performance on cooperative and long-horizon tasks.
>
---
#### [new 045] RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出RoboCasa365，一个用于训练和评估通用机器人的大规模仿真基准，解决机器人泛化能力不足的问题。通过多样任务和数据提升机器人学习效果。**

- **链接: [https://arxiv.org/pdf/2603.04356](https://arxiv.org/pdf/2603.04356)**

> **作者:** Soroush Nasiriany; Sepehr Nasiriany; Abhiram Maddukuri; Yuke Zhu
>
> **备注:** ICLR 2026; First three authors contributed equally
>
> **摘要:** Recent advances in robot learning have accelerated progress toward generalist robots that can perform everyday tasks in human environments. Yet it remains difficult to gauge how close we are to this vision. The field lacks a reproducible, large-scale benchmark for systematic evaluation. To fill this gap, we present RoboCasa365, a comprehensive simulation benchmark for household mobile manipulation. Built on the RoboCasa platform, RoboCasa365 introduces 365 everyday tasks across 2,500 diverse kitchen environments, with over 600 hours of human demonstration data and over 1600 hours of synthetically generated demonstration data -- making it one of the most diverse and large-scale resources for studying generalist policies. RoboCasa365 is designed to support systematic evaluations for different problem settings, including multi-task learning, robot foundation model training, and lifelong learning. We conduct extensive experiments on this benchmark with state-of-the-art methods and analyze the impacts of task diversity, dataset scale, and environment variation on generalization. Our results provide new insights into what factors most strongly affect the performance of generalist robots and inform strategies for future progress in the field.
>
---
#### [new 046] Tendon Force Modeling for Sim2Real Transfer of Reinforcement Learning Policies for Tendon-Driven Robots
- **分类: cs.RO**

- **简介: 该论文属于强化学习在柔性机器人控制中的应用，旨在解决仿真到现实的迁移难题。通过建模肌腱力，提升控制策略的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.04351](https://arxiv.org/pdf/2603.04351)**

> **作者:** Valentin Yuryev; Josie Hughes
>
> **备注:** preprint
>
> **摘要:** Robots which make use of soft or compliant inter- actions often leverage tendon-driven actuation which enables actuators to be placed more flexibly, and compliance to be maintained. However, controlling complex tendon systems is challenging. Simulation paired with reinforcement learning (RL) could be enable more complex behaviors to be generated. Such methods rely on torque and force-based simulation roll- outs which are limited by the sim-to-real gap, stemming from the actuator and system dynamics, resulting in poor transfer of RL policies onto real robots. To address this, we propose a method to model the tendon forces produced by typical servo motors, focusing specifically on the transfer of RL policies for a tendon driven finger. Our approach extends existing data- driven techniques by leveraging contextual history and a novel data collection test-bench. This test-bench allows us to capture tendon forces undergo contact-rich interactions typical of real- world manipulation. We then utilize our force estimation model in a GPU-accelerated tendon force-driven rigid body simulation to train RL-based controllers. Our transformer-based model is capable of predicting tendon forces within 3% of the maximum motor force and is robot-agnostic. By integrating our learned model into simulation, we reduce the sim-to-real gap for test trajectories by 41%. RL-based controller trained with our model achieves a 50% improvement in fingertip pose tracking tasks on real tendon-driven robotic fingers. This approach is generalizable to different actuators and robot systems, and can enable RL policies to be used widely across tendon systems, advancing capabilities of dexterous manipulators and soft robots.
>
---
#### [new 047] Learning Surgical Robotic Manipulation with 3D Spatial Priors
- **分类: cs.RO**

- **简介: 该论文属于手术机器人视觉控制任务，旨在解决手术中3D空间感知不足的问题。提出SST模型，通过端到端方式直接从内镜图像中提取3D空间信息，提升操作精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03798](https://arxiv.org/pdf/2603.03798)**

> **作者:** Yu Sheng; Lidian Wang; Xiaomeng Chu; Jiajun Deng; Min Cheng; Yanyong Zhang; Bei Hua; Houqiang Li; Jianmin Ji
>
> **备注:** CVPR26
>
> **摘要:** Achieving 3D spatial awareness is crucial for surgical robotic manipulation, where precise and delicate operations are required. Existing methods either explicitly reconstruct the surgical scene prior to manipulation, or enhance multi-view features by adding wrist-mounted cameras to supplement the default stereo endoscopes. However, both paradigms suffer from notable limitations: the former easily leads to error accumulation and prevents end-to-end optimization due to its multi-stage nature, while the latter is rarely adopted in clinical practice since wrist-mounted cameras can interfere with the motion of surgical robot arms. In this work, we introduce the Spatial Surgical Transformer (SST), an end-to-end visuomotor policy that empowers surgical robots with 3D spatial awareness by directly exploring 3D spatial cues embedded in endoscopic images. First, we build Surgical3D, a large-scale photorealistic dataset containing 30K stereo endoscopic image pairs with accurate 3D geometry, addressing the scarcity of 3D data in surgical scenes. Based on Surgical3D, we finetune a powerful geometric transformer to extract robust 3D latent representations from stereo endoscopes images. These representations are then seamlessly aligned with the robot's action space via a lightweight multi-level spatial feature connector (MSFC), all within an endoscope-centric coordinate frame. Extensive real-robot experiments demonstrate that SST achieves state-of-the-art performance and strong spatial generalization on complex surgical tasks such as knot tying and ex-vivo organ dissection, representing a significant step toward practical clinical deployment. The dataset and code will be released.
>
---
#### [new 048] RVN-Bench: A Benchmark for Reactive Visual Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RVN-Bench，用于室内安全视觉导航的基准测试。解决现有基准不适用于室内或忽略碰撞的问题，通过构建包含多样化环境和碰撞感知评估的基准，支持有效训练与评估。**

- **链接: [https://arxiv.org/pdf/2603.03953](https://arxiv.org/pdf/2603.03953)**

> **作者:** Jaewon Lee; Jaeseok Heo; Gunmin Lee; Howoong Jun; Jeongwoo Oh; Songhwai Oh
>
> **摘要:** Safe visual navigation is critical for indoor mobile robots operating in cluttered environments. Existing benchmarks, however, often neglect collisions or are designed for outdoor scenarios, making them unsuitable for indoor visual navigation. To address this limitation, we introduce the reactive visual navigation benchmark (RVN-Bench), a collision-aware benchmark for indoor mobile robots. In RVN-Bench, an agent must reach sequential goal positions in previously unseen environments using only visual observations and no prior map, while avoiding collisions. Built on the Habitat 2.0 simulator and leveraging high-fidelity HM3D scenes, RVN-Bench provides large-scale, diverse indoor environments, defines a collision-aware navigation task and evaluation metrics, and offers tools for standardized training and benchmarking. RVN-Bench supports both online and offline learning by offering an environment for online reinforcement learning, a trajectory image dataset generator, and tools for producing negative trajectory image datasets that capture collision events. Experiments show that policies trained on RVN-Bench generalize effectively to unseen environments, demonstrating its value as a standardized benchmark for safe and robust visual navigation. Code and additional materials are available at: this https URL.
>
---
#### [new 049] TreeLoc++: Robust 6-DoF LiDAR Localization in Forests with a Compact Digital Forest Inventory
- **分类: cs.RO**

- **简介: 该论文属于6-DoF LiDAR定位任务，解决森林中依赖密集点云导致的存储与维护成本高的问题。工作是提出TreeLoc++，直接利用数字林相数据进行精准定位。**

- **链接: [https://arxiv.org/pdf/2603.03695](https://arxiv.org/pdf/2603.03695)**

> **作者:** Minwoo Jung; Dongjae Lee; Nived Chebrolu; Haedam Oh; Maurice Fallon; Ayoung Kim
>
> **备注:** 25 pages, 27 figures and 15 tables
>
> **摘要:** Reliable localization is essential for sustainable forest management, as it allows robots or sensor systems to revisit and monitor the status of individual trees over long periods. In modern forestry, this management is structured around Digital Forest Inventories (DFIs), which encode stems using compact geometric attributes rather than raw data. Despite their central role, DFIs have been overlooked in localization research, and most methods still rely on dense gigabyte-sized point clouds that are costly to store and maintain. To improve upon this, we propose TreeLoc++, a global localization framework that operates directly on DFIs as a discriminative representation, eliminating the need to use the raw point clouds. TreeLoc++ reduces false matches in structurally ambiguous forests and improves the reliability of full 6-DoF pose estimation. It augments coarse retrieval with a pairwise distance histogram that encodes local tree-layout context, subsequently refining candidates via DBH-based filtering and yaw-consistent inlier selection to further reduce mismatches. Furthermore, a constrained optimization leveraging tree geometry jointly estimates roll, pitch, and height, enhancing pose stability and enabling accurate localization without reliance on dense 3D point cloud data. Evaluations on 27 sequences recorded in forests across three datasets and four countries show that TreeLoc++ achieves precise localization with centimeter-level accuracy. We further demonstrate robustness to long-term change by localizing data recorded in 2025 against inventories built from 2023 data, spanning a two-year interval. The system represents 15 sessions spanning 7.98 km of trajectories using only 250KB of map data and outperforms both hand-crafted and learning-based baselines that rely on point cloud maps. This demonstrates the scalability of TreeLoc++ for long-term deployment.
>
---
#### [new 050] Passive Phase-Oriented Impedance Shaping for Rapid Acceleration in Soft Robotic Swimmers
- **分类: cs.RO**

- **简介: 该论文属于软体机器人推进任务，旨在提升水下机器人的快速加速能力。通过被动阻抗调节技术，优化力-速度相位关系，显著提高加速性能。**

- **链接: [https://arxiv.org/pdf/2603.03537](https://arxiv.org/pdf/2603.03537)**

> **作者:** Qimin Feng; Orion A. Roberts; Qiang Zhong
>
> **备注:** Submitted to the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Rapid acceleration and burst maneuvers in underwater robots depend less on maintaining precise resonance and more on force--velocity phase alignment during thrust generation. In this work, we investigate constrained-layer damping (CLD) as a passive mechanism for frequency-selective impedance shaping in soft robotic swimmers. Unlike conventional stiffness-tuning approaches, CLD selectively amplifies the dissipative component of bending impedance while preserving storage stiffness, passively shifting the impedance composition toward dissipative dominance as actuation frequency increases. We characterize this behavior through dry impedance measurements, demonstrate that CLD enhances thrust and alters force--motion phase relationships across Strouhal numbers in constrained propulsion tests, and validate that passive impedance shaping yields a nearly five-fold increase in peak acceleration and a three-fold increase in terminal velocity in unconstrained swimming trials. These results establish phase-oriented passive impedance modulation as a simple, control-free pathway for improving transient propulsion in soft robotic systems.
>
---
#### [new 051] HALyPO: Heterogeneous-Agent Lyapunov Policy Optimization for Human-Robot Collaboration
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机协作任务，解决机器人与人类行为多样性带来的学习稳定性问题。提出HALyPO方法，通过Lyapunov优化提升策略学习的稳定性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.03741](https://arxiv.org/pdf/2603.03741)**

> **作者:** Hao Zhang; Yaru Niu; Yikai Wang; Ding Zhao; H. Eric Tseng
>
> **摘要:** To improve generalization and resilience in human-robot collaboration (HRC), robots must handle the combinatorial diversity of human behaviors and contexts, motivating multi-agent reinforcement learning (MARL). However, inherent heterogeneity between robots and humans creates a rationality gap (RG) in the learning process-a variational mismatch between decentralized best-response dynamics and centralized cooperative ascent. The resulting learning problem is a general-sum differentiable game, so independent policy-gradient updates can oscillate or diverge without added structure. We propose heterogeneous-agent Lyapunov policy optimization (HALyPO), which establishes formal stability directly in the policy-parameter space by enforcing a per-step Lyapunov decrease condition on a parameter-space disagreement metric. Unlike Lyapunov-based safe RL, which targets state/trajectory constraints in constrained Markov decision processes, HALyPO uses Lyapunov certification to stabilize decentralized policy learning. HALyPO rectifies decentralized gradients via optimal quadratic projections, ensuring monotonic contraction of RG and enabling effective exploration of open-ended interaction spaces. Extensive simulations and real-world humanoid-robot experiments show that this certified stability improves generalization and robustness in collaborative corner cases.
>
---
#### [new 052] Touch2Insert: Zero-Shot Peg Insertion by Touching Intersections of Peg and Hole
- **分类: cs.RO**

- **简介: 该论文属于机器人连接器插入任务，解决工业连接器精准插入难题。通过触觉感知重建截面几何，实现无监督的零样本插入。**

- **链接: [https://arxiv.org/pdf/2603.03627](https://arxiv.org/pdf/2603.03627)**

> **作者:** Masaru Yajima; Yuma Shin; Rei Kawakami; Asako Kanezaki; Kei Ota
>
> **备注:** Accepted by ICRA 2026 (IEEE International Conference on Robotics and Automation)
>
> **摘要:** Reliable insertion of industrial connectors remains a central challenge in robotics, requiring sub-millimeter precision under uncertainty and often without full visual access. Vision-based approaches struggle with occlusion and limited generalization, while learning-based policies frequently fail to transfer to unseen geometries. To address these limitations, we leverage tactile sensing, which captures local surface geometry at the point of contact and thus provides reliable information even under occlusion and across novel connector shapes. Building on this capability, we present \emph{Touch2Insert}, a tactile-based framework for arbitrary peg insertion. Our method reconstructs cross-sectional geometry from high-resolution tactile images and estimates the relative pose of the hole with respect to the peg in a zero-shot manner. By aligning reconstructed shapes through registration, the framework enables insertion from a single contact without task-specific training. To evaluate its performance, we conducted experiments with three diverse connectors in both simulation and real-robot settings. The results indicate that Touch2Insert achieved sub-millimeter pose estimation accuracy for all connectors in simulation, and attained an average success rate of 86.7\% on the real robot, thereby confirming the robustness and generalizability of tactile sensing for real-world robotic connector insertion.
>
---
#### [new 053] Learning Hip Exoskeleton Control Policy via Predictive Neuromusculoskeletal Simulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决外骨骼控制器泛化能力不足的问题。通过物理仿真实现无需实测数据的控制策略训练，并成功转移到硬件。**

- **链接: [https://arxiv.org/pdf/2603.04166](https://arxiv.org/pdf/2603.04166)**

> **作者:** Ilseung Park; Changseob Song; Inseung Kang
>
> **摘要:** Developing exoskeleton controllers that generalize across diverse locomotor conditions typically requires extensive motion-capture data and biomechanical labeling, limiting scalability beyond instrumented laboratory settings. Here, we present a physics-based neuromusculoskeletal learning framework that trains a hip-exoskeleton control policy entirely in simulation, without motion-capture demonstrations, and deploys it on hardware via policy distillation. A reinforcement learning teacher policy is trained using a muscle-synergy action prior over a wide range of walking speeds and slopes through a two-stage curriculum, enabling direct comparison between assisted and no-exoskeleton conditions. In simulation, exoskeleton assistance reduces mean muscle activation by up to 3.4% and mean positive joint power by up to 7.0% on level ground and ramp ascent, with benefits increasing systematically with walking speed. On hardware, the assistance profiles learned in simulation are preserved across matched speed-slope conditions (r: 0.82, RMSE: 0.03 Nm/kg), providing quantitative evidence of sim-to-real transfer without additional hardware tuning. These results demonstrate that physics-based neuromusculoskeletal simulation can serve as a practical and scalable foundation for exoskeleton controller development, substantially reducing experimental burden during the design phase.
>
---
#### [new 054] LiteVLA-Edge: Quantized On-Device Multimodal Control for Embedded Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于嵌入式机器人任务，解决VLA模型在边缘设备部署困难的问题。通过量化和优化实现低延迟、离线的多模态控制。**

- **链接: [https://arxiv.org/pdf/2603.03380](https://arxiv.org/pdf/2603.03380)**

> **作者:** Justin Williams; Kishor Datta Gupta; Roy George; Mrinmoy Sarkar
>
> **摘要:** Vision-Language-Action (VLA) models provide a unified framework for perception, language conditioning, and action generation, but many existing systems remain difficult to deploy in embedded robotic settings because of their computational requirements and inference latency. In this paper, we present LiteVLA-Edge, a deployment-oriented VLA pipeline for fully on-device inference on Jetson Orin-class hardware. Our approach combines supervised image-to-action fine-tuning in FP32 with post-training 4-bit GGUF quantization and GPU-accelerated inference through the \texttt{this http URL} runtime. Under our deployment configuration, LiteVLA-Edge achieves a mean end-to-end latency of 150.5\,ms (approximately 6.6\,Hz) while operating entirely offline within a ROS~2-integrated perception--reasoning--action pipeline. Rather than introducing a new policy objective, our contribution is a practical systems path for executing compact multimodal control models locally on embedded hardware while preserving modular interfaces between perception, reasoning, and actuation. These results establish timing feasibility for reactive language-conditioned control and provide a reproducible baseline for future task-level evaluation of on-device VLAs in robotics.
>
---
#### [new 055] From Local Matches to Global Masks: Novel Instance Detection in Open-World Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于开放场景下的实例检测任务，解决在无约束环境中基于少量模板图像定位和分割新物体的问题。提出L2G-Det框架，通过局部匹配生成候选点并优化得到完整掩码。**

- **链接: [https://arxiv.org/pdf/2603.03577](https://arxiv.org/pdf/2603.03577)**

> **作者:** Qifan Zhang; Sai Haneesh Allu; Jikai Wang; Yangxiao Lu; Yu Xiang
>
> **摘要:** Detecting and segmenting novel object instances in open-world environments is a fundamental problem in robotic perception. Given only a small set of template images, a robot must locate and segment a specific object instance in a cluttered, previously unseen scene. Existing proposal-based approaches are highly sensitive to proposal quality and often fail under occlusion and background clutter. We propose L2G-Det, a local-to-global instance detection framework that bypasses explicit object proposals by leveraging dense patch-level matching between templates and the query image. Locally matched patches generate candidate points, which are refined through a candidate selection module to suppress false positives. The filtered points are then used to prompt an augmented Segment Anything Model (SAM) with instance-specific object tokens, enabling reliable reconstruction of complete instance masks. Experiments demonstrate improved performance over proposal-based methods in challenging open-world settings.
>
---
#### [new 056] Phys4D: Fine-Grained Physics-Consistent 4D Modeling from Video Diffusion
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Phys4D，解决视频生成中物理一致性不足的问题。通过三阶段训练提升4D场景的物理合理性，增强时空一致性与动态稳定性。**

- **链接: [https://arxiv.org/pdf/2603.03485](https://arxiv.org/pdf/2603.03485)**

> **作者:** Haoran Lu; Shang Wu; Jianshu Zhang; Maojiang Su; Guo Ye; Chenwei Xu; Lie Lu; Pranav Maneriker; Fan Du; Manling Li; Zhaoran Wang; Han Liu
>
> **摘要:** Recent video diffusion models have achieved impressive capabilities as large-scale generative world models. However, these models often struggle with fine-grained physical consistency, exhibiting physically implausible dynamics over time. In this work, we present \textbf{Phys4D}, a pipeline for learning physics-consistent 4D world representations from video diffusion models. Phys4D adopts \textbf{a three-stage training paradigm} that progressively lifts appearance-driven video diffusion models into physics-consistent 4D world representations. We first bootstrap robust geometry and motion representations through large-scale pseudo-supervised pretraining, establishing a foundation for 4D scene modeling. We then perform physics-grounded supervised fine-tuning using simulation-generated data, enforcing temporally consistent 4D dynamics. Finally, we apply simulation-grounded reinforcement learning to correct residual physical violations that are difficult to capture through explicit supervision. To evaluate fine-grained physical consistency beyond appearance-based metrics, we introduce a set of \textbf{4D world consistency evaluation} that probe geometric coherence, motion stability, and long-horizon physical plausibility. Experimental results demonstrate that Phys4D substantially improves fine-grained spatiotemporal and physical consistency compared to appearance-driven baselines, while maintaining strong generative performance. Our project page is available at this https URL
>
---
#### [new 057] RAGNav: A Retrieval-Augmented Topological Reasoning Framework for Multi-Goal Visual-Language Navigation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于多目标视觉语言导航任务，解决多实体识别与空间推理问题。提出RAGNav框架，结合语义与物理结构，提升导航准确性与效率。**

- **链接: [https://arxiv.org/pdf/2603.03745](https://arxiv.org/pdf/2603.03745)**

> **作者:** Ling Luo; Qiangian Bai
>
> **摘要:** Vision-Language Navigation (VLN) is evolving from single-point pathfinding toward the more challenging Multi-Goal VLN. This task requires agents to accurately identify multiple entities while collaboratively reasoning over their spatial-physical constraints and sequential execution order. However, generic Retrieval-Augmented Generation (RAG) paradigms often suffer from spatial hallucinations and planning drift when handling multi-object associations due to the lack of explicit spatial this http URL address these challenges, we propose RAGNav, a framework that bridges the gap between semantic reasoning and physical structure. The core of RAGNav is a Dual-Basis Memory system, which integrates a low-level topological map for maintaining physical connectivity with a high-level semantic forest for hierarchical environment abstraction. Building on this representation, the framework introduces an anchor-guided conditional retrieval and a topological neighbor score propagation mechanism. This approach facilitates the rapid screening of candidate targets and the elimination of semantic noise, while performing semantic calibration by leveraging the physical associations inherent in the topological this http URL mechanism significantly enhances the capability of inter-target reachability reasoning and the efficiency of sequential planning. Experimental results demonstrate that RAGNav achieves state-of-the-art (SOTA) performance in complex multi-goal navigation tasks.
>
---
#### [new 058] Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人策略学习任务，解决持续学习中的遗忘问题。研究发现预训练的视觉-语言-动作模型对遗忘有强抵抗力，简单经验回放即可有效防止遗忘。**

- **链接: [https://arxiv.org/pdf/2603.03818](https://arxiv.org/pdf/2603.03818)**

> **作者:** Huihan Liu; Changyeon Kim; Bo Liu; Minghuan Liu; Yuke Zhu
>
> **摘要:** Continual learning is a long-standing challenge in robot policy learning, where a policy must acquire new skills over time without catastrophically forgetting previously learned ones. While prior work has extensively studied continual learning in relatively small behavior cloning (BC) policy models trained from scratch, its behavior in modern large-scale pretrained Vision-Language-Action (VLA) models remains underexplored. In this work, we found that pretrained VLAs are remarkably resistant to forgetting compared with smaller policy models trained from scratch. Simple Experience Replay (ER) works surprisingly well on VLAs, sometimes achieving zero forgetting even with a small replay data size. Our analysis reveals that pretraining plays a critical role in downstream continual learning performance: large pretrained models mitigate forgetting with a small replay buffer size while maintaining strong forward learning capabilities. Furthermore, we found that VLAs can retain relevant knowledge from prior tasks despite performance degradation during learning new tasks. This knowledge retention enables rapid recovery of seemingly forgotten skills through finetuning. Together, these insights imply that large-scale pretraining fundamentally changes the dynamics of continual learning, enabling models to continually acquire new skills over time with simple replay. Code and more information can be found at this https URL
>
---
#### [new 059] DISC: Dense Integrated Semantic Context for Large-Scale Open-Set Semantic Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DISC，解决开放集语义映射中上下文缺失和计算成本高的问题，通过单次、距离加权的特征提取实现高效实时的语义表示。**

- **链接: [https://arxiv.org/pdf/2603.03935](https://arxiv.org/pdf/2603.03935)**

> **作者:** Felix Igelbrink; Lennart Niecksch; Martin Atzmueller; Joachim Hertzberg
>
> **摘要:** Open-set semantic mapping enables language-driven robotic perception, but current instance-centric approaches are bottlenecked by context-depriving and computationally expensive crop-based feature extraction. To overcome this fundamental limitation, we introduce DISC (Dense Integrated Semantic Context), featuring a novel single-pass, distance-weighted extraction mechanism. By deriving high-fidelity CLIP embeddings directly from the vision transformer's intermediate layers, our approach eliminates the latency and domain-shift artifacts of traditional image cropping, yielding pure, mask-aligned semantic representations. To fully leverage these features in large-scale continuous mapping, DISC is built upon a fully GPU-accelerated architecture that replaces periodic offline processing with precise, on-the-fly voxel-level instance refinement. We evaluate our approach on standard benchmarks (Replica, ScanNet) and a newly generated large-scale-mapping dataset based on Habitat-Matterport 3D (HM3DSEM) to assess scalability across complex scenes in multi-story buildings. Extensive evaluations demonstrate that DISC significantly surpasses current state-of-the-art zero-shot methods in both semantic accuracy and query retrieval, providing a robust, real-time capable framework for robotic deployment. The full source code, data generation and evaluation pipelines will be made available at this https URL.
>
---
#### [new 060] Long-Term Visual Localization in Dynamic Benthic Environments: A Dataset, Footprint-Based Ground Truth, and Visual Place Recognition Benchmark
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于长期水下视觉定位任务，旨在解决动态海底环境下的定位与地图构建问题。作者构建了首个多站点长期水下数据集，并提出基于足迹的真值方法，用于提升视觉位置识别的准确性。**

- **链接: [https://arxiv.org/pdf/2603.04056](https://arxiv.org/pdf/2603.04056)**

> **作者:** Martin Kvisvik Larsen; Oscar Pizarro
>
> **摘要:** Long-term visual localization has the potential to reduce cost and improve mapping quality in optical benthic monitoring with autonomous underwater vehicles (AUVs). Despite this potential, long-term visual localization in benthic environments remains understudied, primarily due to the lack of curated datasets for benchmarking. Moreover, limited georeferencing accuracy and image footprints necessitate precise geometric information for accurate ground-truthing. In this work, we address these gaps by presenting a curated dataset for long-term visual localization in benthic environments and a novel method to ground-truth visual localization results for near-nadir underwater imagery. Our dataset comprises georeferenced AUV imagery from five benthic reference sites, revisited over periods up to six years, and includes raw and color-corrected stereo imagery, camera calibrations, and sub-decimeter registered camera poses. To our knowledge, this is the first curated underwater dataset for long-term visual localization spanning multiple sites and photic-zone habitats. Our ground-truthing method estimates 3D seafloor image footprints and links camera views with overlapping footprints, ensuring that ground-truth links reflect shared visual content. Building on this dataset and ground truth, we benchmark eight state-of-the-art visual place recognition (VPR) methods and find that Recall@K is significantly lower on our dataset than on established terrestrial and underwater benchmarks. Finally, we compare our footprint-based ground truth to a traditional location-based ground truth and show that distance-threshold ground-truthing can overestimate VPR Recall@K at sites with rugged terrain and altitude variations. Together, the curated dataset, ground-truthing method, and VPR benchmark provide a stepping stone for advancing long-term visual localization in dynamic benthic environments.
>
---
#### [new 061] Soft Semi-active Back Support Device with Adaptive Force Profiles using Variable-elastic Actuation and Weight Feedback
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于康复工程任务，旨在解决传统背支持设备笨重或无法自适应的问题。通过结合可变刚度元件和人工肌肉，实现轻便且可调的背支持，提升 lifting 助力效果。**

- **链接: [https://arxiv.org/pdf/2603.03724](https://arxiv.org/pdf/2603.03724)**

> **作者:** Rohan Khatavkar; Bach Nguyen; Inseung Kang; Hyunglae Lee; Jiefeng Sun
>
> **备注:** 17 pages, 18 figures
>
> **摘要:** Portable active back support devices (BSDs) offer tunable assistance but are often bulky and heavy, limiting their usability. In contrast, passive BSDs are lightweight and compact but lack the ability to adapt their assistance to different back movements. We present a soft, lightweight, and compact BSD that combines a variable-stiffness passive element and an active element (an artificial muscle) in parallel. The device provides tunable assistance through discrete changes in stiffness values and active force levels. We validate the device's tuning capabilities through bench testing and on-body characterization. Further, we use the device's tuning capabilities to provide weight-adaptive object lifting and lowering assistance. We detect the weight handled by the user based on forearm force myography and upper-back inertial measurement unit data. Furthermore, electromyography analyses in five participants performing symmetric object lifting and lowering tasks showed reductions in back extensor activity. Preliminary results in one participant also indicated reduced muscle activity during asymmetric lifting.
>
---
#### [new 062] Lambdas at the Far Edge: a Tale of Flying Lambdas and Lambdas on Wheels
- **分类: cs.DC; cs.PL; cs.RO**

- **简介: 本文介绍一种用于分布式设备集体行为编程的模型XC及其C++实现FCPP，解决远端网络设备的协同问题。**

- **链接: [https://arxiv.org/pdf/2603.04008](https://arxiv.org/pdf/2603.04008)**

> **作者:** Giorgio Audrito; Daniele Bortoluzzi; Ferruccio Damiani; Giordano Scarso; Gianluca Torta; Andrea Basso; Monica Cochi; Lorenzo Gusman; Lorenzo Comba; Paolo Gay; Paola Dal Zovo; Giada Galati; Francesco Gallo; Aljaž Grdadolnik; Massimo Pescarollo; Paola Pisano
>
> **备注:** In Proceedings LTT 2026, arXiv:2603.02912
>
> **摘要:** Aggregate Programming (AP) is a paradigm for programming the collective behaviour of sets of distributed devices, possibly situated at the network far edge, by relying on asynchronous proximity-based interactions. The eXchange Calculus (XC), a recently proposed foundational model for AP, is essentially a typed lambda calculus extended with an operator (the exchange operator) providing an implicit communication mechanism between neighbour devices. This paper provides a gentle introduction to XC and to its implementation as a C++ library, called FCPP. The FCPP library and toolchain has been mainly developed at the Department of Computer Science of the University of Turin, where Stefano Berardi spent most of his academic career conducting outstanding research about logical foundation of computer science and transmitting his passion for research to students and young researchers, often exploiting typed lambda calculi. An FCCP program is essentially a typed lambda term, and FCPP has been used to write code that has been deployed on devices at the far edge of the network, including rovers and (soon) Uncrewed Aerial Vehicles (UAVs); hence the title of the paper.
>
---
## 更新

#### [replaced 001] ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL Problems
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出ELMUR，一种用于长视界强化学习的带更新/重写的外部层记忆架构，解决部分可观测环境下的长期依赖问题。通过结构化外部记忆提升决策能力，在多个任务中取得显著性能提升。**

- **链接: [https://arxiv.org/pdf/2510.07151](https://arxiv.org/pdf/2510.07151)**

> **作者:** Egor Cherepanov; Alexey K. Kovalev; Aleksandr I. Panov
>
> **备注:** 31 pages, 15 figures, 8 tables
>
> **摘要:** Real-world robotic agents must act under partial observability and long horizons, where key cues may appear long before they affect decision making. However, most modern approaches rely solely on instantaneous information, without incorporating insights from the past. Standard recurrent or transformer models struggle with retaining and leveraging long-term dependencies: context windows truncate history, while naive memory extensions fail under scale and sparsity. We propose ELMUR (External Layer Memory with Update/Rewrite), a transformer architecture with structured external memory. Each layer maintains memory embeddings, interacts with them via bidirectional cross-attention, and updates them through an Least Recently Used (LRU) memory module using replacement or convex blending. ELMUR extends effective horizons up to 100,000 times beyond the attention window and achieves a 100% success rate on a synthetic T-Maze task with corridors up to one million steps. In POPGym, it outperforms baselines on more than half of the tasks. On MIKASA-Robo sparse-reward manipulation tasks with visual observations, it nearly doubles the performance of strong baselines, achieving the best success rate on 21 out of 23 tasks and improving the aggregate success rate across all tasks by about 70% over the previous best baseline. These results demonstrate that structured, layer-local external memory offers a simple and scalable approach to decision making under partial observability. Code and project page: this https URL.
>
---
#### [replaced 002] VIGOR: Visual Goal-In-Context Inference for Unified Humanoid Fall Safety
- **分类: cs.RO**

- **简介: 该论文属于人形机器人防跌任务，解决复杂环境中跌倒安全问题。提出统一方法，通过视觉目标上下文推理实现高效跌倒恢复。**

- **链接: [https://arxiv.org/pdf/2602.16511](https://arxiv.org/pdf/2602.16511)**

> **作者:** Osher Azulay; Zhengjie Xu; Andrew Scheffer; Stella X. Yu
>
> **摘要:** Reliable fall recovery is critical for humanoids operating in cluttered environments. Unlike quadrupeds or wheeled robots, humanoids experience high-energy impacts, complex whole-body contact, and large viewpoint changes during a fall, making recovery essential for continued operation. Existing methods fragment fall safety into separate problems such as fall avoidance, impact mitigation, and stand-up recovery, or rely on end-to-end policies trained without vision through reinforcement learning or imitation learning, often on flat terrain. At a deeper level, fall safety is treated as monolithic data complexity, coupling pose, dynamics, and terrain and requiring exhaustive coverage, limiting scalability and generalization. We present a unified fall safety approach that spans all phases of fall recovery. It builds on two insights: 1) Natural human fall and recovery poses are highly constrained and transferable from flat to complex terrain through alignment, and 2) Fast whole-body reactions require integrated perceptual-motor representations. We train a privileged teacher using sparse human demonstrations on flat terrain and simulated complex terrains, and distill it into a deployable student that relies only on egocentric depth and proprioception. The student learns how to react by matching the teacher's goal-in-context latent representation, which combines the next target pose with the local terrain, rather than separately encoding what it must perceive and how it must act. Results in simulation and on a real Unitree G1 humanoid demonstrate robust, zero-shot fall safety across diverse non-flat environments without real-world fine-tuning. The project page is available at this https URL
>
---
#### [replaced 003] Segment-to-Act: Label-Noise-Robust Action-Prompted Video Segmentation Towards Embodied Intelligence
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **简介: 该论文属于视频目标分割任务，解决动作引导下的标签噪声问题。通过引入噪声类型、构建基准并提出PMHM机制，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.16677](https://arxiv.org/pdf/2509.16677)**

> **作者:** Wenxin Li; Kunyu Peng; Di Wen; Ruiping Liu; Mengfei Duan; Kai Luo; Kailun Yang
>
> **备注:** Accepted to ICRA 2026. The established benchmark and source code will be made publicly available at this https URL
>
> **摘要:** Embodied intelligence relies on accurately segmenting objects actively involved in interactions. Action-based video object segmentation addresses this by linking segmentation with action semantics, but it depends on large-scale annotations and prompts that are costly, inconsistent, and prone to multimodal noise such as imprecise masks and referential ambiguity. To date, this challenge remains unexplored. In this work, we take the first step by studying action-based video object segmentation under label noise, focusing on two sources: textual prompt noise (category flips and within-category noun substitutions) and mask annotation noise (perturbed object boundaries to mimic imprecise supervision). Our contributions are threefold. First, we introduce two types of label noises for the action-based video object segmentation task. Second, we build up the first action-based video object segmentation under a label noise benchmark ActiSeg-NL and adapt six label-noise learning strategies to this setting, and establish protocols for evaluating them under textual, boundary, and mixed noise. Third, we provide a comprehensive analysis linking noise types to failure modes and robustness gains, and we introduce a Parallel Mask Head Mechanism (PMHM) to address mask annotation noise. Qualitative evaluations further reveal characteristic failure modes, including boundary leakage and mislocalization under boundary perturbations, as well as occasional identity substitutions under textual flips. Our comparative analysis reveals that different learning strategies exhibit distinct robustness profiles, governed by a foreground-background trade-off where some achieve balanced performance while others prioritize foreground accuracy at the cost of background precision. The established benchmark and source code will be made publicly available at this https URL.
>
---
#### [replaced 004] A Bayesian Framework for Active Tactile Object Recognition, Pose Estimation and Shape Transfer Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人触觉感知任务，解决物体识别、位姿估计和形状迁移学习问题。通过贝叶斯框架结合粒子滤波与高斯过程隐面，实现物体形状重建与知识迁移。**

- **链接: [https://arxiv.org/pdf/2409.06912](https://arxiv.org/pdf/2409.06912)**

> **作者:** Haodong Zheng; Andrei Jalba; Raymond H. Cuijpers; Wijnand IJsselsteijn; Sanne Schoenmakers
>
> **摘要:** As humans can explore and understand the world through active touch, similar capability is desired for robots. In this paper, we address the problem of active tactile object recognition, pose estimation and shape transfer learning, where a customized particle filter (PF) and Gaussian process implicit surface (GPIS) is combined in a unified Bayesian framework. Upon new tactile input, the customized PF updates the joint distribution of the object class and object pose while tracking the novelty of the object. Once a novel object is identified, its shape will be reconstructed using GPIS. By grounding the prior of the GPIS with the maximum-a-posteriori (MAP) estimation from the PF, the knowledge about known shapes can be transferred to learn novel shapes. An exploration procedure based on global shape estimation is proposed to guide active data acquisition and terminate the exploration upon sufficient information. Through experiments in simulation, the proposed framework demonstrated its effectiveness and efficiency in estimating object class and pose for known objects and learning novel shapes. Furthermore, it can recognize previously learned shapes reliably.
>
---
#### [replaced 005] A Review of Reward Functions for Reinforcement Learning in the context of Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 本文综述了自动驾驶中强化学习的奖励函数设计，分析其在安全、舒适、进度和交通规则方面的挑战，提出未来需构建标准化、上下文感知的奖励框架。**

- **链接: [https://arxiv.org/pdf/2405.01440](https://arxiv.org/pdf/2405.01440)**

> **作者:** Ahmed Abouelazm; Jonas Michel; J. Marius Zoellner
>
> **备注:** Accepted at the 35th IEEE Intelligent Vehicles Symposium (IV 2024)
>
> **摘要:** Reinforcement learning has emerged as an important approach for autonomous driving. A reward function is used in reinforcement learning to establish the learned skill objectives and guide the agent toward the optimal policy. Since autonomous driving is a complex domain with partly conflicting objectives with varying degrees of priority, developing a suitable reward function represents a fundamental challenge. This paper aims to highlight the gap in such function design by assessing different proposed formulations in the literature and dividing individual objectives into Safety, Comfort, Progress, and Traffic Rules compliance categories. Additionally, the limitations of the reviewed reward functions are discussed, such as objectives aggregation and indifference to driving context. Furthermore, the reward categories are frequently inadequately formulated and lack standardization. This paper concludes by proposing future research that potentially addresses the observed shortcomings in rewards, including a reward validation framework and structured rewards that are context-aware and able to resolve conflicts.
>
---
#### [replaced 006] LaViRA: Language-Vision-Robot Actions Translation for Zero-Shot Vision Language Navigation in Continuous Environments
- **分类: cs.RO**

- **简介: 该论文提出LaViRA，解决零样本视觉语言导航任务中的泛化与推理问题。通过分层动作分解，结合多模态大模型优势，提升导航性能与实用性。**

- **链接: [https://arxiv.org/pdf/2510.19655](https://arxiv.org/pdf/2510.19655)**

> **作者:** Hongyu Ding; Ziming Xu; Yudong Fang; You Wu; Zixuan Chen; Jieqi Shi; Jing Huo; Yifan Zhang; Yang Gao
>
> **备注:** ICRA 2026
>
> **摘要:** LaViRA: Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) requires an agent to navigate unseen environments based on natural language instructions without any prior training. Current methods face a critical trade-off: either rely on environment-specific waypoint predictors that limit scene generalization, or underutilize the reasoning capabilities of large models during navigation. We introduce LaViRA, a simple yet effective zero-shot framework that addresses this dilemma by decomposing action into a coarse-to-fine hierarchy: Language Action for high-level planning, Vision Action for middle-level perceptual grounding, and Robot Action for low-level control. This modular decomposition allows us to leverage the distinct strengths of different scales of Multimodal Large Language Models (MLLMs) at each stage, creating a system that is powerful in its reasoning, grounding and practical control. LaViRA significantly outperforms existing state-of-the-art methods on the VLN-CE benchmark, demonstrating superior generalization capabilities in unseen environments, while maintaining transparency and efficiency for real-world deployment. Project page: this https URL
>
---
#### [replaced 007] Learning with pyCub: A Simulation and Exercise Framework for Humanoid Robotics
- **分类: cs.RO**

- **简介: 本文介绍pyCub，一个基于Python的人形机器人iCub仿真框架，用于教学。解决传统仿真需C++和YARP的问题，提供易用的编程接口和多种练习任务。**

- **链接: [https://arxiv.org/pdf/2506.01756](https://arxiv.org/pdf/2506.01756)**

> **作者:** Lukas Rustler; Matej Hoffmann
>
> **备注:** Accepted to 17th International Conference on Robotics in Education (RiE 2026)
>
> **摘要:** We present pyCub, an open-source physics-based simulation of the humanoid robot iCub, along with exercises to teach students the basics of humanoid robotics. Compared to existing iCub simulators (iCub SIM, iCub Gazebo), which require C++ code and YARP as middleware, pyCub works without YARP and with Python code. The complete robot with all articulations has been simulated, with two cameras in the eyes and the unique sensitive skin of the iCub comprising 4000 receptors on its body surface. The exercises range from basic control of the robot in velocity, joint, and Cartesian space to more complex tasks like gazing, grasping, or reactive control. The whole framework is written and controlled with Python, thus allowing to be used even by people with small or almost no programming practice. The exercises can be scaled to different difficulty levels. We tested the framework in two runs of a course on humanoid robotics. The simulation, exercises, documentation, Docker images, and example videos are publicly available at this https URL.
>
---
#### [replaced 008] Event-LAB: Towards Standardized Evaluation of Neuromorphic Localization Methods
- **分类: cs.RO**

- **简介: 该论文属于事件驱动的定位任务，旨在解决不同方法和数据集比较困难的问题。提出Event-LAB框架，统一管理依赖和数据格式，支持VPR和SLAM等方法的高效比较与分析。**

- **链接: [https://arxiv.org/pdf/2509.14516](https://arxiv.org/pdf/2509.14516)**

> **作者:** Adam D. Hines; Alejandro Fontan; Michael Milford; Tobias Fischer
>
> **备注:** 8 pages, 6 figures, accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Event-based localization research and datasets are a rapidly growing area of interest, with a tenfold increase in the cumulative total number of published papers on this topic over the past 10 years. Whilst the rapid expansion in the field is exciting, it brings with it an associated challenge: a growth in the variety of required code and package dependencies as well as data formats, making comparisons difficult and cumbersome for researchers to implement reliably. To address this challenge, we present Event-LAB: a new and unified framework for running several event-based localization methodologies across multiple datasets. Event-LAB is implemented using the Pixi package and dependency manager, that enables a single command-line installation and invocation for combinations of localization methods and datasets. To demonstrate the capabilities of the framework, we implement two common event-based localization pipelines: Visual Place Recognition (VPR) and Simultaneous Localization and Mapping (SLAM). We demonstrate the ability of the framework to systematically visualize and analyze the results of multiple methods and datasets, revealing key insights such as the association of parameters that control event collection counts and window sizes for frame generation to large variations in performance. The results and analysis demonstrate the importance of fairly comparing methodologies with consistent event image generation parameters. Our Event-LAB framework provides this ability for the research community, by contributing a streamlined workflow for easily setting up multiple conditions.
>
---
#### [replaced 009] VITA: Vision-to-Action Flow Matching Policy
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VITA，一种无需噪声和条件的视觉到动作策略框架，解决视觉与动作对齐难题，提升推理速度与性能。**

- **链接: [https://arxiv.org/pdf/2507.13231](https://arxiv.org/pdf/2507.13231)**

> **作者:** Dechen Gao; Boqi Zhao; Andrew Lee; Ian Chuang; Hanchu Zhou; Hang Wang; Zhe Zhao; Junshan Zhang; Iman Soltani
>
> **备注:** Project page: this https URL Code: this https URL
>
> **摘要:** Conventional flow matching and diffusion-based policies sample via iterative denoising from standard noise distributions (e.g., Gaussian), and require conditioning modules to repeatedly incorporate visual information during the generative process, incurring substantial time and memory overhead. To reduce the complexity, we develop VITA, VIsion-To-Action policy, a noise-free and conditioning-free flow matching policy learning framework that directly flows from visual representations to latent actions. Since the source of the flow is visually grounded, VITA eliminates the need for visual conditioning during generation. As expected, bridging vision and action is challenging, because actions are lower-dimensional, less structured, and sparser than visual representations; moreover, flow matching requires the source and target to have the same dimensionality. To overcome this, we introduce an action autoencoder that maps raw actions into a structured latent space aligned with visual latents, trained jointly with flow matching. To further prevent latent action space collapse during end-to-end training, we propose flow latent decoding, which anchors the latent generation process by backpropagating the action reconstruction loss through the flow matching ODE (ordinary differential equation) solving steps. We evaluate VITA on 9 simulation and 5 real-world tasks from ALOHA and Robomimic. VITA achieves 1.5x-2x faster inference compared to conventional methods with conditioning modules, while outperforming or matching state-of-the-art policies. Project page: this https URL.
>
---
#### [replaced 010] Fine-Tuning Robot Policies While Maintaining User Privacy
- **分类: cs.RO**

- **简介: 该论文属于人机交互领域，解决机器人个性化过程中用户隐私泄露问题。提出PRoP框架，通过密钥加密策略实现个性化与隐私保护。**

- **链接: [https://arxiv.org/pdf/2509.18311](https://arxiv.org/pdf/2509.18311)**

> **作者:** Benjamin A. Christie; Sagar Parekh; Dylan P. Losey
>
> **摘要:** Recent works introduce general-purpose robot policies. These policies provide a strong prior over how robots should behave -- e.g., how a robot arm should manipulate food items. But in order for robots to match an individual person's needs, users typically fine-tune these generalized policies -- e.g., showing the robot arm how to make their own preferred dinners. Importantly, during the process of personalizing robots, end-users leak data about their preferences, habits, and styles (e.g., the foods they prefer to eat). Other agents can simply roll-out the fine-tuned policy and see these personally-trained behaviors. This leads to a fundamental challenge: how can we develop robots that personalize actions while keeping learning private from external agents? We here explore this emerging topic in human-robot interaction and develop PRoP, a model-agnostic framework for personalized and private robot policies. Our core idea is to equip each user with a unique key; this key is then used to mathematically transform the weights of the robot's network. With the correct key, the robot's policy switches to match that user's preferences -- but with incorrect keys, the robot reverts to its baseline behaviors. We show the general applicability of our method across multiple model types in imitation learning, reinforcement learning, and classification tasks. PRoP is practically advantageous because it retains the architecture and behaviors of the original policy, and experimentally outperforms existing encoder-based approaches.
>
---
#### [replaced 011] Evolution 6.0: Robot Evolution through Generative Design
- **分类: cs.RO; cs.NE**

- **简介: 该论文提出Evolution 6.0，通过生成式AI实现机器人自主设计工具并完成任务，解决机器人适应性不足的问题，整合多模型实现工具生成与动作执行。**

- **链接: [https://arxiv.org/pdf/2502.17034](https://arxiv.org/pdf/2502.17034)**

> **作者:** Muhammad Haris Khan; Artyom Myshlyaev; Artem Lykov; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** Accepted to HRI
>
> **摘要:** We propose a new concept, Evolution 6.0, which represents the evolution of robotics driven by Generative AI. When a robot lacks the necessary tools to accomplish a task requested by a human, it autonomously designs the required instruments and learns how to use them to achieve the goal. Evolution 6.0 is an autonomous robotic system powered by Vision-Language Models (VLMs), Vision-Language Action (VLA) models, and Text-to-3D generative models for tool design and task execution. The system comprises two key modules: the Tool Generation Module, which fabricates task-specific tools from visual and textual data, and the Action Generation Module, which converts natural language instructions into robotic actions. It integrates QwenVLM for environmental understanding, OpenVLA for task execution, and Llama-Mesh for 3D tool generation. Evaluation results demonstrate a 90% success rate for tool generation with a 10-second inference time, and action generation achieving 83.5% in physical and visual generalization, 70% in motion generalization, and 37% in semantic generalization. Future improvements will focus on bimanual manipulation, expanded task capabilities, and enhanced environmental interpretation to improve real-world adaptability.
>
---
#### [replaced 012] Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决机器人在部分可观测环境中缺乏统一记忆评估基准的问题。提出MIKASA基准，包含分类框架和32个记忆密集型任务，以提升机器人记忆能力。**

- **链接: [https://arxiv.org/pdf/2502.10550](https://arxiv.org/pdf/2502.10550)**

> **作者:** Egor Cherepanov; Nikita Kachaev; Alexey K. Kovalev; Aleksandr I. Panov
>
> **备注:** 57 pages, 29 figures, 11 tables
>
> **摘要:** Memory is crucial for enabling agents to tackle complex tasks with temporal and spatial dependencies. While many reinforcement learning (RL) algorithms incorporate memory, the field lacks a universal benchmark to assess an agent's memory capabilities across diverse scenarios. This gap is particularly evident in tabletop robotic manipulation, where memory is essential for solving tasks with partial observability and ensuring robust performance, yet no standardized benchmarks exist. To address this, we introduce MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark for memory RL, with three key contributions: (1) we propose a comprehensive classification framework for memory-intensive RL tasks, (2) we collect MIKASA-Base -- a unified benchmark that enables systematic evaluation of memory-enhanced agents across diverse scenarios, and (3) we develop MIKASA-Robo (pip install mikasa-robo-suite) -- a novel benchmark of 32 carefully designed memory-intensive tasks that assess memory capabilities in tabletop robotic manipulation. Our work introduces a unified framework to advance memory RL research, enabling more robust systems for real-world use. MIKASA is available at this https URL.
>
---
#### [replaced 013] Category-Level Object Shape and Pose Estimation in Less Than a Millisecond
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于物体形状与位姿估计任务，解决如何快速准确估计物体形状和位置的问题。通过学习前端检测关键点，结合优化方法实现高效求解。**

- **链接: [https://arxiv.org/pdf/2509.18979](https://arxiv.org/pdf/2509.18979)**

> **作者:** Lorenzo Shaikewitz; Tim Nguyen; Luca Carlone
>
> **备注:** Accepted to ICRA 2026. This version contains appendices
>
> **摘要:** Object shape and pose estimation is a foundational robotics problem, supporting tasks from manipulation to scene understanding and navigation. We present a fast local solver for shape and pose estimation which requires only category-level object priors and admits an efficient certificate of global optimality. Given an RGB-D image of an object, we use a learned front-end to detect sparse, category-level semantic keypoints on the target object. We represent the target object's unknown shape using a linear active shape model and pose a maximum a posteriori optimization problem to solve for position, orientation, and shape simultaneously. Expressed in unit quaternions, this problem admits first-order optimality conditions in the form of an eigenvalue problem with eigenvector nonlinearities. Our primary contribution is to solve this problem efficiently with self-consistent field iteration, which only requires computing a 4-by-4 matrix and finding its minimum eigenvalue-vector pair at each iterate. Solving a linear system for the corresponding Lagrange multipliers gives a simple global optimality certificate. One iteration of our solver runs in about 100 microseconds, enabling fast outlier rejection. We test our method on synthetic data and a variety of real-world settings, including two public datasets and a drone tracking scenario. Code is released at this https URL.
>
---
#### [replaced 014] Metric, inertially aligned monocular state estimation via kinetodynamic priors
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决非刚体平台的精准定位问题。通过融合弹性模型与连续运动模型，实现鲁棒的单目里程计估计。**

- **链接: [https://arxiv.org/pdf/2511.20496](https://arxiv.org/pdf/2511.20496)**

> **作者:** Jiaxin Liu; Min Li; Wanting Xu; Liang Li; Jiaqi Yang; Laurent Kneip
>
> **摘要:** Accurate state estimation for flexible robotic systems poses significant challenges, particularly for platforms with dynamically deforming structures that invalidate rigid-body assumptions. This paper addresses this problem and enables the extension of existing rigid-body pose estimation methods to non-rigid systems. Our approach integrates two core components: first, we capture elastic properties using a deformation-force model, efficiently learned via a Multi-Layer Perceptron; second, we resolve the platform's inherently smooth motion using continuous-time B-spline kinematic models. By continuously applying Newton's Second Law, our method formulates the relationship between visually-derived trajectory acceleration and predicted deformation-induced acceleration. We demonstrate that our approach not only enables robust and accurate pose estimation on non-rigid platforms, but also demonstrates that the properly modeled platform physics allow for the recovery of inertial sensing properties. We validate this feasibility on a simple spring-camera system, showing how it robustly resolves the typically ill-posed problem of metric scale and gravity recovery in monocular visual odometry.
>
---
#### [replaced 015] CASSR: Continuous A-Star Search through Reachability for real time footstep planning
- **分类: cs.RO**

- **简介: 该论文提出CASSR框架，用于解决双足机器人实时步态规划问题。通过结合A*搜索与连续约束传播，提升规划效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.02989](https://arxiv.org/pdf/2603.02989)**

> **作者:** Jiayi Wang; Steve Tonneau
>
> **摘要:** Footstep planning involves a challenging combinatorial search. Traditional A* approaches require discretising reachability constraints, while Mixed-Integer Programming (MIP) supports continuous formulations but quickly becomes intractable, especially when rotations are included. We present CASSR, a novel framework that recursively propagates convex, continuous formulations of a robot's kinematic constraints within an A* search. Combined with a new cost-to-go heuristic based on the EPA algorithm, CASSR efficiently plans contact sequences of up to 30 footsteps in under 125 ms. Experiments on biped locomotion tasks demonstrate that CASSR outperforms traditional discretised A* by up to a factor of 100, while also surpassing a commercial MIP solver. These results show that CASSR enables fast, reliable, and real-time footstep planning for biped robots.
>
---
#### [replaced 016] FlowCorrect: Efficient Interactive Correction of Generative Flow Policies for Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出FlowCorrect，解决机器人操作中生成策略在分布偏移下的失败问题。通过少量人类修正，实时微调策略，提升成功率并保持原有性能。**

- **链接: [https://arxiv.org/pdf/2602.22056](https://arxiv.org/pdf/2602.22056)**

> **作者:** Edgar Welte; Yitian Shi; Rosa Wolf; Maximillian Gilles; Rania Rayyes
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Generative manipulation policies can fail catastrophically under deployment-time distribution shift, yet many failures are near-misses: the robot reaches almost-correct poses and would succeed with a small corrective motion. We propose FlowCorrect, a modular interactive imitation learning approach that enables deployment-time adaptation of flow-matching manipulation policies from sparse, relative human corrections without retraining. During execution, a human provides brief corrective pose nudges via a lightweight VR interface. FlowCorrect uses these sparse corrections to locally adapt the policy, improving actions without retraining the backbone while preserving the model performance on previously learned scenarios. We evaluate on a real-world robot across four tabletop tasks: pick-and-place, pouring, cup uprighting, and insertion. With a low correction budget, FlowCorrect achieves an 80% success rate on previously failed cases while preserving performance on previously solved scenarios. The results clearly demonstrate that FlowCorrect learns from very few demonstrations and enables fast, sample-efficient, incremental, human-in-the-loop corrections of generative visuomotor policies at deployment time in real-world robotics.
>
---
#### [replaced 017] Q-Guided Stein Variational Model Predictive Control via RL-informed Policy Prior
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于强化学习与控制任务，旨在解决传统MPC依赖精确模型和成本函数的问题。通过Q-SVMPC方法，利用RL引导的策略先验和软Q值指导，提升轨迹优化的多样性与稳定性。**

- **链接: [https://arxiv.org/pdf/2507.06625](https://arxiv.org/pdf/2507.06625)**

> **作者:** Shizhe Cai; Zeya Yin; Jayadeep Jacob; Fabio Ramos
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Model Predictive Control (MPC) enables reliable trajectory optimization under dynamics constraints, but often depends on accurate dynamics models and carefully hand-designed cost functions. Recent learning-based MPC methods aim to reduce these modeling and cost-design burdens by learning dynamics, priors, or value-related guidance signals. Yet many existing approaches still rely on deterministic gradient-based solvers (e.g., differentiable MPC) or parametric sampling-based updates (e.g., CEM/MPPI), which can lead to mode collapse and convergence to a single dominant solution. We propose Q-SVMPC, a Q-guided Stein variational MPC method with an RL-informed policy prior, which casts learning-based MPC as trajectory-level posterior inference and refines trajectory particles via SVGD under learned soft Q-value guidance to explicitly preserve diverse solutions. Experiments on navigation, robotic manipulation, and a real-world fruit-picking task show improved sample efficiency, stability, and robustness over MPC, model-free RL, and learning-based MPC baselines.
>
---
#### [replaced 018] Point2Act: Efficient 3D Distillation of Multimodal LLMs for Zero-Shot Context-Aware Grasping
- **分类: cs.RO**

- **简介: 该论文提出Point2Act，解决零样本情境感知抓取问题，通过3D知识蒸馏将多模态大模型的语义转化为精准3D动作点。**

- **链接: [https://arxiv.org/pdf/2508.03099](https://arxiv.org/pdf/2508.03099)**

> **作者:** Sang Min Kim; Hyeongjun Heo; Junho Kim; Yonghyeon Lee; Young Min Kim
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** We propose Point2Act, which directly retrieves the 3D action point relevant to a contextually described task, leveraging Multimodal Large Language Models (MLLMs). Foundation models opened the possibility for generalist robots that can perform a zero-shot task following natural language descriptions within an unseen environment. While the semantics obtained from large-scale image and language datasets provide contextual understanding in 2D images, the rich yet nuanced features deduce blurry 2D regions and struggle to find precise 3D locations for actions. Our proposed 3D relevancy fields bypass the high-dimensional features and instead efficiently imbue lightweight 2D point-level guidance tailored to the task-specific action. The multi-view aggregation effectively compensates for misalignments due to geometric ambiguities, such as occlusion, or semantic uncertainties inherent in the language descriptions. The output region is highly localized, reasoning fine-grained 3D spatial context that can directly transfer to an explicit position for physical action at the on-the-fly reconstruction of the scene. Our full-stack pipeline, which includes capturing, MLLM querying, 3D reconstruction, and grasp pose extraction, generates spatially grounded responses in under 20 seconds, facilitating practical manipulation tasks. Project page: this https URL
>
---
#### [replaced 019] CLASH: Collision Learning via Augmented Sim-to-real Hybridization to Bridge the Reality Gap
- **分类: cs.RO**

- **简介: 该论文属于机器人控制领域，旨在解决模拟到现实的差距问题。通过CLASH框架，利用少量真实碰撞数据提升模拟精度，增强策略迁移效果。**

- **链接: [https://arxiv.org/pdf/2602.18707](https://arxiv.org/pdf/2602.18707)**

> **作者:** Haotian He; Ning Guo; Siqi Shi; Qipeng Liu; Wenzhao Lian
>
> **摘要:** The sim-to-real gap, particularly in the inaccurate modeling of contact-rich dynamics like collisions, remains a primary obstacle to deploying robot policies trained in simulation. Conventional physics engines often trade accuracy for computational speed, leading to discrepancies that prevent direct policy transfer. To address this, we introduce Collision Learning via Augmented Sim-to-real Hybridization (CLASH), a data-efficient framework that learns a parameter-conditioned impulsive collision surrogate model and integrates it as a plug-in module within a standard simulator. CLASH first distills a base model from an imperfect simulator (MuJoCo) using large-scale simulated collisions to capture reusable physical priors. Given only a handful of real collisions (e.g., 10 samples), it then (i) performs gradient-based identification of key contact parameters and (ii) applies small-step, early-stopped fine-tuning to correct residual sim-to-real mismatches while avoiding overfitting. The resulting hybrid simulator not only achieves higher post-impact prediction accuracy but also reduces the wall-clock time of collision-heavy CMA-ES search by 42-48% compared to MuJoCo. We demonstrate that policies obtained with our hybrid simulator transfer more robustly to the real world, doubling the success rate in sequential pushing tasks with reinforcement learning and significantly increase the task performance with model-based control.
>
---
#### [replaced 020] Extremely Simple Multimodal Outlier Synthesis for Out-of-Distribution Detection and Segmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于OOD检测与分割任务，旨在解决多模态数据中缺乏监督信号导致的过自信预测问题。提出Feature Mixing方法，提升模型区分ID与OOD数据的能力，并构建了CARLA-OOD数据集。**

- **链接: [https://arxiv.org/pdf/2505.16985](https://arxiv.org/pdf/2505.16985)**

> **作者:** Moru Liu; Hao Dong; Jessica Kelly; Olga Fink; Mario Trapp
>
> **备注:** NeurIPS 2025
>
> **摘要:** Out-of-distribution (OOD) detection and segmentation are crucial for deploying machine learning models in safety-critical applications such as autonomous driving and robot-assisted surgery. While prior research has primarily focused on unimodal image data, real-world applications are inherently multimodal, requiring the integration of multiple modalities for improved OOD detection. A key challenge is the lack of supervision signals from unknown data, leading to overconfident predictions on OOD samples. To address this challenge, we propose Feature Mixing, an extremely simple and fast method for multimodal outlier synthesis with theoretical support, which can be further optimized to help the model better distinguish between in-distribution (ID) and OOD data. Feature Mixing is modality-agnostic and applicable to various modality combinations. Additionally, we introduce CARLA-OOD, a novel multimodal dataset for OOD segmentation, featuring synthetic OOD objects across diverse scenes and weather conditions. Extensive experiments on SemanticKITTI, nuScenes, CARLA-OOD datasets, and the MultiOOD benchmark demonstrate that Feature Mixing achieves state-of-the-art performance with a $10 \times$ to $370 \times$ speedup. Our source code and dataset will be available at this https URL.
>
---
#### [replaced 021] TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出TIGeR框架，解决机器人视觉-语言模型在几何推理中的精度不足问题，通过集成外部工具实现精确计算。**

- **链接: [https://arxiv.org/pdf/2510.07181](https://arxiv.org/pdf/2510.07181)**

> **作者:** Yi Han; Enshen Zhou; Shanyu Rong; Jingkun An; Pengwei Wang; Zhongyuan Wang; Cheng Chi; Lu Sheng; Shanghang Zhang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks.
>
---
#### [replaced 022] Safety Guardrails for LLM-Enabled Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人安全任务，旨在解决LLM在机器人应用中的安全风险。提出RoboGuard架构，通过上下文安全规则和逻辑控制合成，有效减少危险行为。**

- **链接: [https://arxiv.org/pdf/2503.07885](https://arxiv.org/pdf/2503.07885)**

> **作者:** Zachary Ravichandran; Alexander Robey; Vijay Kumar; George J. Pappas; Hamed Hassani
>
> **摘要:** Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the contextual vulnerabilities of LLMs, and current LLM safety approaches overlook the physical risks posed by robots operating in real-world environments. To ensure the safety of LLM-enabled robots, we propose RoboGuard, a two-stage guardrail architecture. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM. This LLM is shielded from malicious prompts and employs chain-of-thought (CoT) reasoning to generate context-dependent safety specifications, such as temporal logic constraints. RoboGuard then resolves conflicts between these contextual safety specifications and potentially unsafe plans using temporal logic control synthesis, ensuring compliance while minimally violating user preferences. In simulation and real-world experiments that consider worst-case jailbreaking attacks, RoboGuard reduces the execution of unsafe plans from over 92% to below 3% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and enhanced by its root-of-trust LLM's CoT reasoning. These results demonstrate the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots. We provide additional resources at this https URL.
>
---
#### [replaced 023] No Need to Look! Locating and Grasping Objects by a Robot Arm Covered with Sensitive Skin
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决无视觉输入下的物体定位与抓取问题。通过机器人全身触觉反馈实现物体搜索与精准抓取。**

- **链接: [https://arxiv.org/pdf/2508.17986](https://arxiv.org/pdf/2508.17986)**

> **作者:** Karel Bartunek; Lukas Rustler; Matej Hoffmann
>
> **备注:** Karel Bartunek, Lukas Rustler: Authors contributed equally Accepted to IEEE ICRA 2026
>
> **摘要:** Locating and grasping of objects by robots is typically performed using visual sensors. Haptic feedback from contacts with the environment is only secondary if present at all. In this work, we explored an extreme case of searching for and grasping objects in complete absence of visual input, relying on haptic feedback only. The main novelty lies in the use of contacts over the complete surface of a robot manipulator covered with sensitive skin. The search is divided into two phases: (1) coarse workspace exploration with the complete robot surface, followed by (2) precise localization using the end-effector equipped with a force/torque sensor. We systematically evaluated this method in simulation and on the real robot, demonstrating that diverse objects can be located, grasped, and put in a basket. The overall success rate on the real robot for one object was 85.7% with failures mainly while grasping specific objects. The method using whole-body contacts is six times faster compared to a baseline that uses haptic feedback only on the end-effector. We also show locating and grasping multiple objects on the table. This method is not restricted to our specific setup and can be deployed on any platform with the ability of sensing contacts over the entire body surface. This work holds promise for diverse applications in areas with challenging visual perception (due to lighting, dust, smoke, occlusion) such as in agriculture when fruits or vegetables need to be located inside foliage and picked.
>
---
#### [replaced 024] SoraNav: Adaptive UAV Task-Centric Navigation via Zeroshot VLM Reasoning
- **分类: cs.RO**

- **简介: 该论文属于无人机任务导向导航任务，解决语言驱动的3D导航中空间推理不足的问题。提出SoraNav框架，结合多模态视觉标注和自适应决策策略，提升导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2510.25191](https://arxiv.org/pdf/2510.25191)**

> **作者:** Hongyu Song; Rishabh Dev Yadav; Cheng Guo; Wei Pan
>
> **摘要:** Autonomous navigation under natural language instructions represents a crucial step toward embodied intelligence, enabling complex task execution in environments ranging from industrial facilities to domestic spaces. However, language-driven 3D navigation for Unmanned Aerial Vehicles (UAVs) requires precise spatial reasoning, a capability inherently lacking in current zero-shot Vision-Language Models (VLMs) which often generate ambiguous outputs and cannot guarantee geometric feasibility. Furthermore, existing Vision-Language Navigation (VLN) methods are predominantly tailored for 2.5D ground robots, rendering them unable to generalize to the unconstrained 3D spatial reasoning required for aerial tasks in small-scale, cluttered environments. In this paper, we present SoraNav, a novel framework enabling zero-shot VLM reasoning for UAV task-centric navigation. To address the spatial-semantic gap, we introduce Multi-modal Visual Annotation (MVA), which encodes 3D geometric priors directly into the VLM's 2D visual input. To mitigate hallucinated or infeasible commands, we propose an Adaptive Decision Making (ADM) strategy that validates VLM proposals against exploration history, seamlessly switching to geometry-based exploration to avoid dead-ends and redundant revisits. Deployed on a custom PX4-based micro-UAV, SoraNav demonstrates robust real-world performance. Quantitative results show our approach significantly outperforms state-of-the-art baselines, increasing Success Rate (SR) by 25.7% and navigation efficiency (SPL) by 17.3% in 2.5D scenarios, and achieving improvements of 39.3% (SR) and 24.7% (SPL) in complex 3D scenarios.
>
---
#### [replaced 025] RehearseVLA: Simulated Post-Training for VLAs with Physically-Consistent World Model
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的后训练任务，解决数据稀缺和安全问题。提出RehearseVLA，利用物理一致的世界模型进行虚拟训练，提升性能与安全性。**

- **链接: [https://arxiv.org/pdf/2509.24948](https://arxiv.org/pdf/2509.24948)**

> **作者:** Junjin Xiao; Yandan Yang; Xinyuan Chang; Ronghan Chen; Feng Xiong; Mu Xu; Wei-Shi Zheng; Qing Zhang
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Vision-Language-Action (VLA) models trained via imitation learning suffer from significant performance degradation in data-scarce scenarios due to their reliance on large-scale demonstration datasets. Although reinforcement learning (RL)-based post-training has proven effective in addressing data scarcity, its application to VLA models is hindered by the non-resettable nature of real-world environments. This limitation is particularly critical in high-risk domains such as industrial automation, where interactions often induce state changes that are costly or infeasible to revert. Furthermore, existing VLA approaches lack a reliable mechanism for detecting task completion, leading to redundant actions that reduce overall task success rates. To address these challenges, we propose RehearseVLA:, an RL-based post-training framework that replaces physical interaction with a low-cost world model-based virtual simulator. RehearseVLA: consists of two key components: (1) a physically-consistent world simulator that generates temporally consistent future visual observations, and (2) a vision-language model (VLM)-guided instant reflector that provides continuous reward signals and predicts action termination. This simulated environment enables VLA models to safely explore and generalize beyond their initial imitation learning distribution. Our method achieves notable performance gains with as few as five expert demonstrations per task. Experiments on complex robotic manipulation tasks demonstrate that RehearseVLA: effectively overcomes the data inefficiency, safety constraints, and inefficient execution of conventional VLA models that rely on real-world interaction, offering a practical and scalable solution for post-training in resource-constrained settings. Our code is available at this https URL.
>
---
#### [replaced 026] A Self-Supervised Learning Approach with Differentiable Optimization for UAV Trajectory Planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机轨迹规划任务，解决3D环境下的路径规划问题。提出一种自监督学习与可微优化结合的方法，提升轨迹的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2504.04289](https://arxiv.org/pdf/2504.04289)**

> **作者:** Yufei Jiang; Yuanzhu Zhan; Harsh Vardhan Gupta; Chinmay Borde; Junyi Geng
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** While Unmanned Aerial Vehicles (UAVs) have gained significant traction across various fields, path planning in 3D environments remains a critical challenge, particularly under size, weight, and power (SWAP) constraints. Traditional modular planning systems often introduce latency and suboptimal performance due to limited information sharing and local minima issues. End-to-end learning approaches streamline the pipeline by mapping sensory observations directly to actions but require large-scale datasets, face significant sim-to-real gaps, or lack dynamical feasibility. In this paper, we propose a self-supervised UAV trajectory planning pipeline that integrates a learning-based depth perception with differentiable trajectory optimization. A 3D cost map guides UAV behavior without expert demonstrations or human labels. Additionally, we incorporate a neural network-based time allocation strategy to improve the efficiency and optimality. The system thus combines robust learning-based perception with reliable physics-based optimization for improved generalizability and interpretability. Both simulation and real-world experiments validate our approach across various environments, demonstrating its effectiveness and robustness. Our method achieves a 31.33% improvement in position tracking error and 49.37% reduction in control effort compared to the state-of-the-art.
>
---
#### [replaced 027] Learning Agile Gate Traversal via Analytical Optimal Policy Gradient
- **分类: cs.RO**

- **简介: 该论文属于无人机敏捷飞行任务，解决窄门穿越问题。提出混合框架，结合神经网络与模型预测控制，提升穿越精度与抗扰能力。**

- **链接: [https://arxiv.org/pdf/2508.21592](https://arxiv.org/pdf/2508.21592)**

> **作者:** Tianchen Sun; Bingheng Wang; Nuthasith Gerdpratoom; Longbin Tang; Yichao Gao; Lin Zhao
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Traversing narrow gates presents a significant challenge and has become a standard benchmark for evaluating agile and precise quadrotor flight. Traditional modularized autonomous flight stacks require extensive design and parameter tuning, while end-to-end reinforcement learning (RL) methods often suffer from low sample efficiency, limited interpretability, and degraded disturbance rejection under unseen perturbations. In this work, we present a novel hybrid framework that adaptively fine-tunes model predictive control (MPC) parameters online using outputs from a neural network (NN) trained offline. The NN jointly predicts a reference pose and cost function weights, conditioned on the coordinates of the gate corners and the current drone state. To achieve efficient training, we derive analytical policy gradients not only for the MPC module but also for an optimization-based gate traversal detection module. Hardware experiments demonstrate agile and accurate gate traversal with peak accelerations of $30\ \mathrm{m/s^2}$, as well as recovery within $0.85\ \mathrm{s}$ following body-rate disturbances exceeding $1146\ \mathrm{deg/s}$.
>
---
#### [replaced 028] Fusion of Visual-Inertial Odometry with LiDAR Relative Localization for Cooperative Guidance of a Micro-Scale Aerial Vehicle
- **分类: cs.RO**

- **简介: 论文提出一种融合VIO与LiDAR的相对定位方法，用于微小型无人机的协同导航。解决轻量级无人机定位精度不足问题，通过协作提升导航可靠性与准确性。**

- **链接: [https://arxiv.org/pdf/2306.17544](https://arxiv.org/pdf/2306.17544)**

> **作者:** Václav Pritzl; Matouš Vrba; Petr Štěpán; Martin Saska
>
> **备注:** Accepted version
>
> **摘要:** A novel relative localization approach for guidance of a micro-scale Unmanned Aerial Vehicle (UAV) by a well-equipped aerial robot fusing Visual-Inertial Odometry (VIO) with Light Detection and Ranging (LiDAR) is proposed in this paper. LiDAR-based localization is accurate and robust to challenging environmental conditions, but 3D LiDARs are relatively heavy and require large UAV platforms, in contrast to lightweight cameras. However, visual-based self-localization methods exhibit lower accuracy and can suffer from significant drift with respect to the global reference frame. To benefit from both sensory modalities, we focus on cooperative navigation in a heterogeneous team of a primary LiDAR-equipped UAV and a secondary micro-scale camera-equipped UAV. We propose a novel cooperative approach combining LiDAR relative localization data with VIO output on board the primary UAV to obtain an accurate pose of the secondary UAV. The pose estimate is used to precisely and reliably guide the secondary UAV along trajectories defined in the primary UAV reference frame. The experimental evaluation has shown the superior accuracy of our method to the raw VIO output, reaching the average 3D Absolute Trajectory Error (ATE) of 0.28 m, and demonstrated its capability to guide the secondary UAV along desired trajectories while mitigating VIO drift. Thus, such a heterogeneous system can explore large areas with LiDAR precision, as well as visit locations inaccessible to the large LiDAR-carrying UAV platforms, as was showcased in a real-world cooperative mapping scenario.
>
---
#### [replaced 029] Dynamic-ICP: Doppler-Aware Iterative Closest Point Registration for Dynamic Scenes
- **分类: cs.RO**

- **简介: 该论文属于点云配准任务，解决动态场景中ICP方法失效的问题。通过引入多普勒信息，提升动态环境下的定位精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.20292](https://arxiv.org/pdf/2511.20292)**

> **作者:** Dong Wang; Daniel Casado Herraez; Stefan May; Andreas Nüchter
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Reliable odometry in highly dynamic environments remains challenging when it relies on ICP-based registration: ICP assumes near-static scenes and degrades in repetitive or low-texture geometry. We introduce Dynamic-ICP, a Doppler-aware registration framework. The method (i) estimates ego motion from per-point Doppler velocity via robust regression and builds a velocity filter, (ii) clusters dynamic objects and reconstructs object-wise translational velocities from ego-compensated radial measurements, (iii) predicts dynamic points with a constant-velocity model, and (iv) aligns scans using a compact objective that combines point-to-plane geometry residual with a translation-invariant, rotation-only Doppler residual. The approach requires no external sensors or sensor-vehicle calibration and operates directly on FMCW LiDAR range and Doppler velocities. We evaluate Dynamic-ICP on three datasets-HeRCULES, HeLiPR, AevaScenes-focusing on highly dynamic scenes. Dynamic-ICP consistently improves rotational stability and translation accuracy over the state-of-the-art methods. Our approach is also simple to integrate into existing pipelines, runs in real time, and provides a lightweight solution for robust registration in dynamic environments. To encourage further research, the code is available at: this https URL.
>
---
#### [replaced 030] A Novel Modular Cable-Driven Soft Robotic Arm with Multi-Segment Reconfigurability
- **分类: cs.RO**

- **简介: 论文提出一种可多段重构的模块化软体机械臂，解决传统结构适应性差的问题。通过模块化设计和材料刚度调节，提升灵活性与承载能力。**

- **链接: [https://arxiv.org/pdf/2603.02468](https://arxiv.org/pdf/2603.02468)**

> **作者:** Moeen Ul Islam; Cheng Ouyang; Xinda Qi; Azlan Zahid; Xiaobo Tan; Dong Chen
>
> **备注:** 6 pages, 8 figures
>
> **摘要:** This paper presents a novel, modular, cable-driven soft robotic arm featuring multi-segment reconfigurability. The proposed architecture enables a stackable system with independent segment control, allowing scalable adaptation to diverse structural and application requirements. The system is fabricated from soft silicone material and incorporates embedded tendon-routing channels with a protective dual-helical tendon structure. Experimental results showed that modular stacking substantially expanded the reachable workspace: relative to the single-segment arm, the three-segment configuration achieved up to a 13-fold increase in planar workspace area and a 38.9-fold increase in workspace volume. Furthermore, this study investigated the effect of silicone stiffness on actuator performance. The results revealed a clear trade-off between compliance and stiffness: softer silicone improved bending flexibility, while stiffer silicone improved structural rigidity and load-bearing stability. These results highlight the potential of stiffness tuning to balance compliance and strength for configuring scalable, reconfigurable soft robotic arms.
>
---
#### [replaced 031] Learning Physical Principles from Interaction: Self-Evolving Planning via Test-Time Memory
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决物体物理属性理解不足的问题。通过引入PhysMem框架，使模型在测试时学习物理原理，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.20323](https://arxiv.org/pdf/2602.20323)**

> **作者:** Haoyang Li; Yang You; Hao Su; Leonidas Guibas
>
> **摘要:** Reliable object manipulation requires understanding physical properties that vary across objects and environments. Vision-language model (VLM) planners can reason about friction and stability in general terms; however, they often cannot predict how a specific ball will roll on a particular surface or which stone will provide a stable foundation without direct experience. We present PhysMem, a memory framework that enables VLM robot planners to learn physical principles from interaction at test time, without updating model parameters. The system records experiences, generates candidate hypotheses, and verifies them through targeted interaction before promoting validated knowledge to guide future decisions. A central design choice is verification before application: the system tests hypotheses against new observations rather than applying retrieved experience directly, reducing rigid reliance on prior experience when physical conditions change. We evaluate PhysMem on three real-world manipulation tasks and simulation benchmarks across four VLM backbones. On a controlled brick insertion task, principled abstraction achieves 76% success compared to 23% for direct experience retrieval, and real-world experiments show consistent improvement over 30-minute deployment sessions.
>
---
#### [replaced 032] TOLEBI: Learning Fault-Tolerant Bipedal Locomotion via Online Status Estimation and Fallibility Rewards
- **分类: cs.RO**

- **简介: 该论文属于双足机器人运动控制任务，旨在解决硬件故障下的容错问题。通过在线状态估计和奖励机制，学习故障容忍的运动策略，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.05596](https://arxiv.org/pdf/2602.05596)**

> **作者:** Hokyun Lee; Woo-Jeong Baek; Junhyeok Cha; Jaeheung Park
>
> **备注:** Accepted for Publication at IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** With the growing employment of learning algorithms in robotic applications, research on reinforcement learning for bipedal locomotion has become a central topic for humanoid robotics. While recently published contributions achieve high success rates in locomotion tasks, scarce attention has been devoted to the development of methods that enable to handle hardware faults that may occur during the locomotion process. However, in real-world settings, environmental disturbances or sudden occurrences of hardware faults might yield severe consequences. To address these issues, this paper presents TOLEBI (A faulT-tOlerant Learning framEwork for Bipedal locomotIon) that handles faults on the robot during operation. Specifically, joint locking, power loss and external disturbances are injected in simulation to learn fault-tolerant locomotion strategies. In addition to transferring the learned policy to the real robot via sim-to-real transfer, an online joint status module incorporated. This module enables to classify joint conditions by referring to the actual observations at runtime under real-world conditions. The validation experiments conducted both in real-world and simulation with the humanoid robot TOCABI highlight the applicability of the proposed approach. To our knowledge, this manuscript provides the first learning-based fault-tolerant framework for bipedal locomotion, thereby fostering the development of efficient learning methods in this field.
>
---
#### [replaced 033] TPK: Trustworthy Trajectory Prediction Integrating Prior Knowledge For Interpretability and Kinematic Feasibility
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于轨迹预测任务，旨在提升预测的可信度。针对现有模型预测不物理可行、缺乏解释性的问题，提出融合先验知识的方法，增强交互可解释性和物理合理性。**

- **链接: [https://arxiv.org/pdf/2505.06743](https://arxiv.org/pdf/2505.06743)**

> **作者:** Marius Baden; Ahmed Abouelazm; Christian Hubschneider; Yin Wu; Daniel Slieter; J. Marius Zöllner
>
> **备注:** First and Second authors contributed equally; Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025) for oral presentation; Winner of the best paper award
>
> **摘要:** Trajectory prediction is crucial for autonomous driving, enabling vehicles to navigate safely by anticipating the movements of surrounding road users. However, current deep learning models often lack trustworthiness as their predictions can be physically infeasible and illogical to humans. To make predictions more trustworthy, recent research has incorporated prior knowledge, like the social force model for modeling interactions and kinematic models for physical realism. However, these approaches focus on priors that suit either vehicles or pedestrians and do not generalize to traffic with mixed agent classes. We propose incorporating interaction and kinematic priors of all agent classes--vehicles, pedestrians, and cyclists with class-specific interaction layers to capture agent behavioral differences. To improve the interpretability of the agent interactions, we introduce DG-SFM, a rule-based interaction importance score that guides the interaction layer. To ensure physically feasible predictions, we proposed suitable kinematic models for all agent classes with a novel pedestrian kinematic model. We benchmark our approach on the Argoverse 2 dataset, using the state-of-the-art transformer HPTR as our baseline. Experiments demonstrate that our method improves interaction interpretability, revealing a correlation between incorrect predictions and divergence from our interaction prior. Even though incorporating the kinematic models causes a slight decrease in accuracy, they eliminate infeasible trajectories found in the dataset and the baseline model. Thus, our approach fosters trust in trajectory prediction as its interaction reasoning is interpretable, and its predictions adhere to physics.
>
---
#### [replaced 034] CERNet: Class-Embedding Predictive-Coding RNN for Unified Robot Motion, Recognition, and Confidence Estimation
- **分类: cs.RO**

- **简介: 该论文提出CERNet模型，解决机器人运动生成、行为识别和置信度估计问题。通过统一的PC-RNN框架实现三项功能，提升机器人与人类交互能力。**

- **链接: [https://arxiv.org/pdf/2512.07041](https://arxiv.org/pdf/2512.07041)**

> **作者:** Hiroki Sawada; Alexandre Pitti; Mathias Quoy
>
> **备注:** Accepted for presentation at IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Robots interacting with humans must not only generate learned movements in real-time, but also infer the intent behind observed behaviors and estimate the confidence of their own inferences. This paper proposes a unified model that achieves all three capabilities within a single hierarchical predictive-coding recurrent neural network (PC-RNN) equipped with a class embedding vector, CERNet, which leverages a dynamically updated class embedding vector to unify motor generation and recognition. The model operates in two modes: generation and inference. In the generation mode, the class embedding constrains the hidden state dynamics to a class-specific subspace; in the inference mode, it is optimized online to minimize prediction error, enabling real-time recognition. Validated on a humanoid robot across 26 kinesthetically taught alphabets, our hierarchical model achieves 76% lower trajectory reproduction error than a parameter-matched single-layer baseline, maintains motion fidelity under external perturbations, and infers the demonstrated trajectory class online with 68% Top-1 and 81% Top-2 accuracy. Furthermore, internal prediction errors naturally reflect the model's confidence in its recognition. This integration of robust generation, real-time recognition, and intrinsic uncertainty estimation within a compact PC-RNN framework offers a compact and extensible approach to motor memory in physical robots, with potential applications in intent-sensitive human-robot collaboration.
>
---
#### [replaced 035] Agile Flight Emerges from Multi-Agent Competitive Racing
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文研究多智能体竞争飞行任务，解决如何通过稀疏奖励训练出敏捷且策略性强的飞行控制策略。工作包括模拟与现实验证，证明多智能体竞争优于单智能体训练。**

- **链接: [https://arxiv.org/pdf/2512.11781](https://arxiv.org/pdf/2512.11781)**

> **作者:** Vineet Pasumarti; Lorenzo Bianchi; Antonio Loquercio
>
> **摘要:** Through multi-agent competition and the sparse high-level objective of winning a race, we find that both agile flight (e.g., high-speed motion pushing the platform to its physical limits) and strategy (e.g., overtaking or blocking) emerge from agents trained with reinforcement learning. We provide evidence in both simulation and the real world that this approach outperforms the common paradigm of training agents in isolation with rewards that prescribe behavior, e.g., progress on the raceline, in particular when the complexity of the environment increases, e.g., in the presence of obstacles. Moreover, we find that multi-agent competition yields policies that transfer more reliably to the real world than policies trained with a single-agent progress-based reward, despite the two methods using the same simulation environment, randomization strategy, and hardware. In addition to improved sim-to-real transfer, the multi-agent policies also exhibit some degree of generalization to opponents unseen at training time. Overall, our work, following in the tradition of multi-agent competitive game-play in digital domains, shows that sparse task-level rewards are sufficient for training agents capable of advanced low-level control in the physical world. Code: this https URL
>
---
#### [replaced 036] Aerial Manipulation with Contact-Aware Onboard Perception and Hybrid Control
- **分类: cs.RO**

- **简介: 该论文属于空中操作任务，解决无外部定位的精确接触控制问题。提出基于视觉惯性里程计和混合控制的自主感知-控制流程，实现稳定接触与力控制。**

- **链接: [https://arxiv.org/pdf/2602.08251](https://arxiv.org/pdf/2602.08251)**

> **作者:** Yuanzhu Zhan; Yufei Jiang; Muqing Cao; Junyi Geng
>
> **备注:** 8 pages, 7 figures. Accepted by ICRA 2026
>
> **摘要:** Aerial manipulation (AM) promises to move Unmanned Aerial Vehicles (UAVs) beyond passive inspection to contact-rich tasks such as grasping, assembly, and in-situ maintenance. Most prior AM demonstrations rely on external motion capture (MoCap) and emphasize position control for coarse interactions, limiting deployability. We present a fully onboard perception-control pipeline for contact-rich AM that achieves accurate motion tracking and regulated contact wrenches without MoCap. The main components are (1) an augmented visual-inertial odometry (VIO) estimator with contact-consistency factors that activate only during interaction, tightening uncertainty around the contact frame and reducing drift, and (2) image-based visual servoing (IBVS) to mitigate perception-control coupling, together with a hybrid force-motion controller that regulates contact wrenches and lateral motion for stable contact. Experiments show that our approach closes the perception-to-wrench loop using only onboard sensing, yielding an velocity estimation improvement of 66.01% at contact, reliable target approach, and stable force holding-pointing toward deployable, in-the-wild aerial manipulation.
>
---
#### [replaced 037] Ask, Reason, Assist: Decentralized Robot Collaboration via Language and Logic
- **分类: cs.RO**

- **简介: 该论文研究机器人协作任务，解决异构机器人团队在冲突时的协同问题。通过语言与逻辑结合的框架，实现去中心化帮助请求与选择，提升任务效率。**

- **链接: [https://arxiv.org/pdf/2509.23506](https://arxiv.org/pdf/2509.23506)**

> **作者:** Dan BW Choe; Sundhar Vinodh Sangeetha; Steven Emanuel; Chih-Yuan Chiu; Samuel Coogan; Shreyas Kousik
>
> **摘要:** Increased robot deployment, such as in warehousing, has revealed a need for seamless collaboration among heterogeneous robot teams to resolve unforeseen conflicts. To address this challenge, we propose a novel decentralized framework that enables robots to request and provide help. The process begins when a robot detects a conflict and uses a Large Language Model (LLM) to decide whether external assistance is required. If so, it crafts and broadcasts a natural language (NL) help request. Potential helper robots reason over the request and respond with offers of assistance, including information about the effect on their ongoing tasks. Helper reasoning is implemented via an LLM grounded in Signal Temporal Logic (STL) using a Backus-Naur Form (BNF) grammar, ensuring syntactically valid NL-to-STL translations, which are then solved as a Mixed Integer Linear Program (MILP). Finally, the requester robot selects a helper by reasoning over the expected increase in system-level total task completion time. We evaluated our framework through experiments comparing different helper-selection strategies and found that considering multiple offers allows the requester to minimize added makespan. Our approach significantly outperforms heuristics such as selecting the nearest available candidate helper robot, and achieves performance comparable to a centralized "Oracle" baseline but without heavy information demands.
>
---
#### [replaced 038] H-WM: Robotic Task and Motion Planning Guided by Hierarchical World Model
- **分类: cs.RO**

- **简介: 该论文提出H-WM，融合逻辑与视觉状态预测，解决机器人长时序任务规划中误差累积问题，提升执行鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.11291](https://arxiv.org/pdf/2602.11291)**

> **作者:** Jinbang Huang; Wenyuan Chen; Zhiyuan Li; Oscar Pang; Xiao Hu; Lingfeng Zhang; Yuanzhao Hu; Zhanguang Zhang; Mark Coates; Tongtong Cao; Xingyue Quan; Yingxue Zhang
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** World models are becoming central to robotic planning and control as they enable prediction of future state transitions. Existing approaches often emphasize video generation or natural-language prediction, which are difficult to ground in robot actions and suffer from compounding errors over long horizons. Classic task and motion planning models world transitions in logical space, enabling robot-executable and robust long-horizon reasoning. However, they typically operate independently of visual perception, preventing synchronized symbolic and visual state prediction. We propose a Hierarchical World Model (H-WM) that jointly predicts logical and visual state transitions within a unified framework. H-WM combines a high-level logical world model with a low-level visual world model, integrating the long-horizon robustness of symbolic reasoning with visual grounding. The hierarchical outputs provide stable intermediate guidance for long-horizon tasks, mitigating error accumulation and enabling robust execution across extended task sequences. Experiments across multiple vision-language-action (VLA) control policies demonstrate the effectiveness and generality of H-WM's guidance.
>
---
#### [replaced 039] Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作学习任务，旨在提升扩散策略的训练效率。通过引入投影几何代数，构建混合扩散策略hPGA-DP，有效提升空间推理能力与训练速度。**

- **链接: [https://arxiv.org/pdf/2507.05695](https://arxiv.org/pdf/2507.05695)**

> **作者:** Xiatao Sun; Yuxuan Wang; Shuo Yang; Yinxing Chen; Daniel Rakita
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Diffusion policies are a powerful paradigm for robot learning, but their training is often inefficient. A key reason is that networks must relearn fundamental spatial concepts, such as translations and rotations, from scratch for every new task. To alleviate this redundancy, we propose embedding geometric inductive biases directly into the network architecture using Projective Geometric Algebra (PGA). PGA provides a unified algebraic framework for representing geometric primitives and transformations, allowing neural networks to reason about spatial structure more effectively. In this paper, we introduce hPGA-DP, a novel hybrid diffusion policy that capitalizes on these benefits. Our architecture leverages the Projective Geometric Algebra Transformer (P-GATr) as a state encoder and action decoder, while employing established U-Net or Transformer-based modules for the core denoising process. Through extensive experiments and ablation studies in both simulated and real-world environments, we demonstrate that hPGA-DP significantly improves task performance and training efficiency. Notably, our hybrid approach achieves substantially faster convergence compared to both standard diffusion policies and architectures that rely solely on P-GATr. The project website is available at: this https URL.
>
---
