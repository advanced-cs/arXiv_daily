# 机器人 cs.RO

- **最新发布 44 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Robots That Generate Planarity Through Geometry
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决传统系统依赖高精度组件的问题。通过几何反演实现自参照平面运动，提升精度并简化系统。**

- **链接: [https://arxiv.org/pdf/2602.06294v1](https://arxiv.org/pdf/2602.06294v1)**

> **作者:** Jakub F. Kowalewski; Abdulaziz O. Alrashed; Jacob Alpert; Rishi Ponnapalli; Lucas R. Meza; Jeffrey Ian Lipton
>
> **摘要:** Constraining motion to a flat surface is a fundamental requirement for equipment across science and engineering. Modern precision robotic motion systems, such as gantries, rely on the flatness of components, including guide rails and granite surface plates. However, translating this static flatness into motion requires precise internal alignment and tight-tolerance components that create long, error-sensitive reference chains. Here, we show that by using the geometric inversion of a sphere into a plane, we can produce robotic motion systems that derive planarity entirely from link lengths and connectivity. This allows planar motion to emerge from self-referencing geometric constraints, and without external metrology. We demonstrate these Flat-Plane Mechanisms (FPMs) from micron to meter scales and show that fabrication errors can be attenuated by an order of magnitude in the resulting flatness. Finally, we present a robotic FPM-based 3-axis positioning system that can be used for metrology surface scans ($\pm 12$-mm) and 3D printing inside narrow containers. This work establishes an alternative geometric foundation for planar motion that can be realized across size scales and opens new possibilities in metrology, fabrication, and micro-positioning.
>
---
#### [new 002] A Consistency-Improved LiDAR-Inertial Bundle Adjustment
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于SLAM任务，旨在解决LiDAR-Inertial系统中估计不一致的问题。通过改进参数化和优化方法，提升系统的精度与可观测性。**

- **链接: [https://arxiv.org/pdf/2602.06380v1](https://arxiv.org/pdf/2602.06380v1)**

> **作者:** Xinran Li; Shuaikang Zheng; Pengcheng Zheng; Xinyang Wang; Jiacheng Li; Zhitian Li; Xudong Zou
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) using 3D LiDAR has emerged as a cornerstone for autonomous navigation in robotics. While feature-based SLAM systems have achieved impressive results by leveraging edge and planar structures, they often suffer from the inconsistent estimator associated with feature parameterization and estimated covariance. In this work, we present a consistency-improved LiDAR-inertial bundle adjustment (BA) with tailored parameterization and estimator. First, we propose a stereographic-projection representation parameterizing the planar and edge features, and conduct a comprehensive observability analysis to support its integrability with consistent estimator. Second, we implement a LiDAR-inertial BA with Maximum a Posteriori (MAP) formulation and First-Estimate Jacobians (FEJ) to preserve the accurate estimated covariance and observability properties of the system. Last, we apply our proposed BA method to a LiDAR-inertial odometry.
>
---
#### [new 003] Bioinspired Kirigami Capsule Robot for Minimally Invasive Gastrointestinal Biopsy
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决传统胃肠活检方法侵入性强、取样不精准的问题。研究设计了一种仿 kirigami 结构的胶囊机器人，实现微创、可重复的组织采集。**

- **链接: [https://arxiv.org/pdf/2602.06207v1](https://arxiv.org/pdf/2602.06207v1)**

> **作者:** Ruizhou Zhao; Yichen Chu; Shuwei Zhao; Wenchao Yue; Raymond Shing-Yan Tang; Hongliang Ren
>
> **备注:** 8 pages, 11 figures, accepted to IEEE ICRA
>
> **摘要:** Wireless capsule endoscopy (WCE) has transformed gastrointestinal (GI) diagnostics by enabling noninvasive visualization of the digestive tract, yet its diagnostic yield remains constrained by the absence of biopsy capability, as histological analysis is still the gold standard for confirming disease. Conventional biopsy using forceps, needles, or rotating blades is invasive, limited in reach, and carries risks of perforation or mucosal trauma, while fluid- or microbiota-sampling capsules cannot provide structured tissue for pathology, leaving a critical gap in swallowable biopsy solutions. Here we present the Kiri-Capsule, a kirigami-inspired capsule robot that integrates deployable PI-film flaps actuated by a compact dual-cam mechanism to achieve minimally invasive and repeatable tissue collection. The kirigami surface remains flat during locomotion but transforms into sharp protrusions upon cam-driven stretching, enabling controlled penetration followed by rotary scraping, with specimens retained in internal fan-shaped cavities. Bench tests confirmed that PI films exhibit a Young's modulus of approximately 20 MPa and stable deployment angles (about 34$^\circ$ at 15% strain), while ex vivo porcine studies demonstrated shallow penetration depths (median $\sim$0.61 mm, range 0.46--0.66 mm) and biopsy yields comparable to standard forceps (mean $\sim$10.9 mg for stomach and $\sim$18.9 mg for intestine), with forces within safe ranges reported for GI biopsy. These findings demonstrate that the Kiri-Capsule bridges passive imaging and functional biopsy, providing a swallowable, depth-controlled, and histology-ready solution that advances capsule-based diagnostics toward safe and effective clinical application.
>
---
#### [new 004] SURE: Safe Uncertainty-Aware Robot-Environment Interaction using Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文提出SURE框架，解决机器人与环境接触时的轨迹优化问题，通过考虑接触时间不确定性提升任务鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06864v1](https://arxiv.org/pdf/2602.06864v1)**

> **作者:** Zhuocheng Zhang; Haizhou Zhao; Xudong Sun; Aaron M. Johnson; Majid Khadiv
>
> **摘要:** Robotic tasks involving contact interactions pose significant challenges for trajectory optimization due to discontinuous dynamics. Conventional formulations typically assume deterministic contact events, which limit robustness and adaptability in real-world settings. In this work, we propose SURE, a robust trajectory optimization framework that explicitly accounts for contact timing uncertainty. By allowing multiple trajectories to branch from possible pre-impact states and later rejoin a shared trajectory, SURE achieves both robustness and computational efficiency within a unified optimization framework. We evaluate SURE on two representative tasks with unknown impact times. In a cart-pole balancing task involving uncertain wall location, SURE achieves an average improvement of 21.6% in success rate when branch switching is enabled during control. In an egg-catching experiment using a robotic manipulator, SURE improves the success rate by 40%. These results demonstrate that SURE substantially enhances robustness compared to conventional nominal formulations.
>
---
#### [new 005] Towards Adaptive Environment Generation for Training Embodied Agents
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决 embodied agents 在新环境中泛化能力差的问题。通过闭环环境生成，根据代理表现动态调整难度，提升学习效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06366v1](https://arxiv.org/pdf/2602.06366v1)**

> **作者:** Teresa Yeo; Dulaj Weerakoon; Dulanga Weerakoon; Archan Misra
>
> **备注:** Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
>
> **摘要:** Embodied agents struggle to generalize to new environments, even when those environments share similar underlying structures to their training settings. Most current approaches to generating these training environments follow an open-loop paradigm, without considering the agent's current performance. While procedural generation methods can produce diverse scenes, diversity without feedback from the agent is inefficient. The generated environments may be trivially easy, providing limited learning signal. To address this, we present a proof-of-concept for closed-loop environment generation that adapts difficulty to the agent's current capabilities. Our system employs a controllable environment representation, extracts fine-grained performance feedback beyond binary success or failure, and implements a closed-loop adaptation mechanism that translates this feedback into environment modifications. This feedback-driven approach generates training environments that more challenging in the ways the agent needs to improve, enabling more efficient learning and better generalization to novel settings.
>
---
#### [new 006] Primary Experimental Feedback on a Co-manipulated Robotic System for Assisted Cervical Surgery
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在评估协作机器人系统在颈椎手术中的钻孔精度。通过实验分析系统误差，以提升手术安全性和效率。**

- **链接: [https://arxiv.org/pdf/2602.06541v1](https://arxiv.org/pdf/2602.06541v1)**

> **作者:** Seifeddine Sellemi; Abdelbadia Chaker; Tanguy Vendeuvre; Terence Essomba; Med Amine Laribi
>
> **摘要:** Robotic-assisted surgery has emerged as a promising approach to improve surgical ergonomics, precision, and workflow efficiency, particularly in complex procedures such as cervical spine surgery. In this study, we evaluate the performance of a collaborative robotic system designed to assist surgeons in drilling tasks by assessing its accuracy in executing predefined trajectories. A total of 14 drillings were performed by eight experienced cervical surgeons, utilizing a robotic-assisted setup aimed at ensuring stability and alignment. The primary objective of this study is to quantify the deviations in the position and orientation of the drilling tool relative to the planned trajectory, providing insights into the system's reliability and potential impact on clinical outcomes. While the primary function of robotic assistance in surgery is to enhance surgeon comfort and procedural guidance rather than solely optimizing precision, understanding the system's accuracy remains crucial for its effective integration into surgical practices part of this primary experimental feedback, the study offers an in-depth analysis of the co-manipulated robotic system's performance, focusing on the experimental setup and error evaluation methods. The findings of this study will contribute to the ongoing development of robotic-assisted cervical surgery, highlighting both its advantages and areas for improvement in achieving safer and more efficient surgical workflows
>
---
#### [new 007] Strategizing at Speed: A Learned Model Predictive Game for Multi-Agent Drone Racing
- **分类: cs.RO; cs.GT**

- **简介: 该论文属于多智能体无人机竞速任务，解决高速决策与策略制定问题。通过比较MPG与MPC，提出LMPG以平衡计算延迟与交互推理。**

- **链接: [https://arxiv.org/pdf/2602.06925v1](https://arxiv.org/pdf/2602.06925v1)**

> **作者:** Andrei-Carlo Papuc; Lasse Peters; Sihao Sun; Laura Ferranti; Javier Alonso-Mora
>
> **摘要:** Autonomous drone racing pushes the boundaries of high-speed motion planning and multi-agent strategic decision-making. Success in this domain requires drones not only to navigate at their limits but also to anticipate and counteract competitors' actions. In this paper, we study a fundamental question that arises in this domain: how deeply should an agent strategize before taking an action? To this end, we compare two planning paradigms: the Model Predictive Game (MPG), which finds interaction-aware strategies at the expense of longer computation times, and contouring Model Predictive Control (MPC), which computes strategies rapidly but does not reason about interactions. We perform extensive experiments to study this trade-off, revealing that MPG outperforms MPC at moderate velocities but loses its advantage at higher speeds due to latency. To address this shortcoming, we propose a Learned Model Predictive Game (LMPG) approach that amortizes model predictive gameplay to reduce latency. In both simulation and hardware experiments, we benchmark our approach against MPG and MPC in head-to-head races, finding that LMPG outperforms both baselines.
>
---
#### [new 008] Internalized Morphogenesis: A Self-Organizing Model for Growth, Replication, and Regeneration via Local Token Exchange in Modular Systems
- **分类: cs.RO; q-bio.QM**

- **简介: 该论文属于自组织系统研究，解决资源受限模块的形态生成问题。提出一种基于局部令牌交换的内部形态发生模型，实现自主生长、复制与再生。**

- **链接: [https://arxiv.org/pdf/2602.06296v1](https://arxiv.org/pdf/2602.06296v1)**

> **作者:** Takeshi Ishida
>
> **摘要:** This study presents an internalized morphogenesis model for autonomous systems, such as swarm robotics and micro-nanomachines, that eliminates the need for external spatial computation. Traditional self-organizing models often require calculations across the entire coordinate space, including empty areas, which is impractical for resource-constrained physical modules. Our proposed model achieves complex morphogenesis through strictly local interactions between adjacent modules within the "body." By extending the "Ishida token model," modules exchange integer values using an RD-inspired discrete analogue without solving differential equations. The internal potential, derived from token accumulation and aging, guides autonomous growth, shrinkage, and replication. Simulations on a hexagonal grid demonstrated the emergence of limb-like extensions, self-division, and robust regeneration capabilities following structural amputation. A key feature is the use of the body boundary as a natural sink for information entropy (tokens) to maintain a dynamic equilibrium. These results indicate that sophisticated morphological behaviors can emerge from minimal, internal-only rules. This framework offers a computationally efficient and biologically plausible approach to developing self-repairing, adaptive, and autonomous hardware.
>
---
#### [new 009] A Dialogue-Based Human-Robot Interaction Protocol for Wheelchair and Robotic Arm Integrated Control
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决传统辅助界面不直观的问题。提出对话式交互协议，通过自然语言理解用户意图，实现轮椅与机械臂的协同控制。**

- **链接: [https://arxiv.org/pdf/2602.06243v1](https://arxiv.org/pdf/2602.06243v1)**

> **作者:** Guangping Liu; Nicholas Hawkins; Billy Madden; Tipu Sultan; Madi Babaiasl
>
> **摘要:** People with lower and upper body disabilities can benefit from wheelchairs and robotic arms to improve mobility and independence. Prior assistive interfaces, such as touchscreens and voice-driven predefined commands, often remain unintuitive and struggle to capture complex user intent. We propose a natural, dialogue based human robot interaction protocol that simulates an intelligent agent capable of communicating with users to understand intent and execute assistive actions. In a pilot study, five participants completed five assistive tasks (cleaning, drinking, feeding, drawer opening, and door opening) through dialogue-based interaction with a wheelchair and robotic arm. As a baseline, participants were required to open a door using the manual control (a wheelchair joystick and a game controller for the arm) and complete a questionnaire to gather their feedback. By analyzing the post-study questionnaires, we found that most participants enjoyed the dialogue-based interaction and assistive robot autonomy.
>
---
#### [new 010] HiWET: Hierarchical World-Frame End-Effector Tracking for Long-Horizon Humanoid Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人长时程操作任务，解决动态稳定与精准末端跟踪问题。提出HiWET框架，通过分层强化学习实现世界坐标系下的末端跟踪。**

- **链接: [https://arxiv.org/pdf/2602.06341v1](https://arxiv.org/pdf/2602.06341v1)**

> **作者:** Zhanxiang Cao; Liyun Yan; Yang Zhang; Sirui Chen; Jianming Ma; Tianyue Zhan; Shengcheng Fu; Yufei Jia; Cewu Lu; Yue Gao
>
> **摘要:** Humanoid loco-manipulation requires executing precise manipulation tasks while maintaining dynamic stability amid base motion and impacts. Existing approaches typically formulate commands in body-centric frames, fail to inherently correct cumulative world-frame drift induced by legged locomotion. We reformulate the problem as world-frame end-effector tracking and propose HiWET, a hierarchical reinforcement learning framework that decouples global reasoning from dynamic execution. The high-level policy generates subgoals that jointly optimize end-effector accuracy and base positioning in the world frame, while the low-level policy executes these commands under stability constraints. We introduce a Kinematic Manifold Prior (KMP) that embeds the manipulation manifold into the action space via residual learning, reducing exploration dimensionality and mitigating kinematically invalid behaviors. Extensive simulation and ablation studies demonstrate that HiWET achieves precise and stable end-effector tracking in long-horizon world-frame tasks. We validate zero-shot sim-to-real transfer of the low-level policy on a physical humanoid, demonstrating stable locomotion under diverse manipulation commands. These results indicate that explicit world-frame reasoning combined with hierarchical control provides an effective and scalable solution for long-horizon humanoid loco-manipulation.
>
---
#### [new 011] Think Proprioceptively: Embodied Visual Reasoning for VLA Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决机器人如何有效利用本体感知提升操作性能。通过将本体信息转化为文本标记并早期融合，优化视觉推理与动作选择。**

- **链接: [https://arxiv.org/pdf/2602.06575v1](https://arxiv.org/pdf/2602.06575v1)**

> **作者:** Fangyuan Wang; Peng Zhou; Jiaming Qi; Shipeng Lyu; David Navarro-Alarcon; Guodong Guo
>
> **摘要:** Vision-language-action (VLA) models typically inject proprioception only as a late conditioning signal, which prevents robot state from shaping instruction understanding and from influencing which visual tokens are attended throughout the policy. We introduce ThinkProprio, which converts proprioception into a sequence of text tokens in the VLM embedding space and fuses them with the task instruction at the input. This early fusion lets embodied state participate in subsequent visual reasoning and token selection, biasing computation toward action-critical evidence while suppressing redundant visual tokens. In a systematic ablation over proprioception encoding, state entry point, and action-head conditioning, we find that text tokenization is more effective than learned projectors, and that retaining roughly 15% of visual tokens can match the performance of using the full token set. Across CALVIN, LIBERO, and real-world manipulation, ThinkProprio matches or improves over strong baselines while reducing end-to-end inference latency over 50%.
>
---
#### [new 012] RAPID: Reconfigurable, Adaptive Platform for Iterative Design
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RAPID平台，解决机器人操作策略迭代中硬件配置耗时的问题。通过模块化硬件和软件协同，实现快速重构与传感器热插拔支持，提升实验效率。**

- **链接: [https://arxiv.org/pdf/2602.06653v1](https://arxiv.org/pdf/2602.06653v1)**

> **作者:** Zi Yin; Fanhong Li; Shurui Zheng; Jia Liu
>
> **摘要:** Developing robotic manipulation policies is iterative and hypothesis-driven: researchers test tactile sensing, gripper geometries, and sensor placements through real-world data collection and training. Yet even minor end-effector changes often require mechanical refitting and system re-integration, slowing iteration. We present RAPID, a full-stack reconfigurable platform designed to reduce this friction. RAPID is built around a tool-free, modular hardware architecture that unifies handheld data collection and robot deployment, and a matching software stack that maintains real-time awareness of the underlying hardware configuration through a driver-level Physical Mask derived from USB events. This modular hardware architecture reduces reconfiguration to seconds and makes systematic multi-modal ablation studies practical, allowing researchers to sweep diverse gripper and sensing configurations without repeated system bring-up. The Physical Mask exposes modality presence as an explicit runtime signal, enabling auto-configuration and graceful degradation under sensor hot-plug events, so policies can continue executing when sensors are physically added or removed. System-centric experiments show that RAPID reduces the setup time for multi-modal configurations by two orders of magnitude compared to traditional workflows and preserves policy execution under runtime sensor hot-unplug events. The hardware designs, drivers, and software stack are open-sourced at https://rapid-kit.github.io/ .
>
---
#### [new 013] Dynamic Modeling, Parameter Identification and Numerical Analysis of Flexible Cables in Flexibly Connected Dual-AUV Systems
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于动态建模与参数识别任务，解决柔性连接双AUV系统中电缆非线性行为的建模问题，通过实验与模型结合的方法识别关键参数并分析其动态特性。**

- **链接: [https://arxiv.org/pdf/2602.06087v1](https://arxiv.org/pdf/2602.06087v1)**

> **作者:** Kuo Chen; Minghao Dou; Qianqi Liu; Yang An; Kai Ren; Zeming WU; Yu Tian; Jie Sun; Xinping Wang; Zhier Chen; Jiancheng Yu
>
> **摘要:** This research presents a dynamic modeling framework and parameter identification methods for describing the highly nonlinear behaviors of flexibly connected dual-AUV systems. The modeling framework is established based on the lumped mass method, integrating axial elasticity, bending stiffness, added mass and hydrodynamic forces, thereby accurately capturing the time-varying response of the forces and cable configurations. To address the difficulty of directly measuring material-related and hydrodynamic coefficients, this research proposes a parameter identification method that combines the physical model with experimental data. High-precision inversion of the equivalent Youngs modulus and hydrodynamic coefficients is performed through tension experiments under multiple configurations, effectively demonstrating that the identified model maintains predictive consistency in various operational conditions. Further numerical analysis indicates that the dynamic properties of flexible cable exhibit significant nonlinear characteristics, which are highly dependent on material property variations and AUV motion conditions. This nonlinear dynamic behavior results in two typical response states, slack and taut, which are jointly determined by boundary conditions and hydrodynamic effects, significantly affecting the cable configuration and endpoint loads. In this research, the dynamics of flexible cables under complex boundary conditions is revealed, providing a theoretical foundation for the design, optimization and further control research of similar systems.
>
---
#### [new 014] A High-Fidelity Robotic Manipulator Teleoperation Framework for Human-Centered Augmented Reality Evaluation
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于AR评估任务，解决人类运动不一致问题。设计ARBot系统，通过机器人精确复现人类动作，提升AR验证的准确性与可重复性。**

- **链接: [https://arxiv.org/pdf/2602.06273v1](https://arxiv.org/pdf/2602.06273v1)**

> **作者:** Harsh Chhajed; Tian Guo
>
> **摘要:** Validating Augmented Reality (AR) tracking and interaction models requires precise, repeatable ground-truth motion. However, human users cannot reliably perform consistent motion due to biomechanical variability. Robotic manipulators are promising to act as human motion proxies if they can mimic human movements. In this work, we design and implement ARBot, a real-time teleoperation platform that can effectively capture natural human motion and accurately replay the movements via robotic manipulators. ARBot includes two capture models: stable wrist motion capture via a custom CV and IMU pipeline, and natural 6-DOF control via a mobile application. We design a proactively-safe QP controller to ensure smooth, jitter-free execution of the robotic manipulator, enabling it to function as a high-fidelity record and replay physical proxy. We open-source ARBot and release a benchmark dataset of 132 human and synthetic trajectories captured using ARBot to support controllable and scalable AR evaluation.
>
---
#### [new 015] DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出DreamDojo，一个基于大规模人类视频的通用机器人世界模型，解决少数据和缺乏动作标签的问题。通过学习多样化交互与精细控制，提升机器人在复杂环境中的模拟能力。**

- **链接: [https://arxiv.org/pdf/2602.06949v1](https://arxiv.org/pdf/2602.06949v1)**

> **作者:** Shenyuan Gao; William Liang; Kaiyuan Zheng; Ayaan Malik; Seonghyeon Ye; Sihyun Yu; Wei-Cheng Tseng; Yuzhu Dong; Kaichun Mo; Chen-Hsuan Lin; Qianli Ma; Seungjun Nah; Loic Magne; Jiannan Xiang; Yuqi Xie; Ruijie Zheng; Dantong Niu; You Liang Tan; K. R. Zentner; George Kurian; Suneel Indupuru; Pooya Jannaty; Jinwei Gu; Jun Zhang; Jitendra Malik; Pieter Abbeel; Ming-Yu Liu; Yuke Zhu; Joel Jang; Linxi "Jim" Fan
>
> **备注:** Project page: https://dreamdojo-world.github.io/
>
> **摘要:** Being able to simulate the outcomes of actions in varied environments will revolutionize the development of generalist agents at scale. However, modeling these world dynamics, especially for dexterous robotics tasks, poses significant challenges due to limited data coverage and scarce action labels. As an endeavor towards this end, we introduce DreamDojo, a foundation world model that learns diverse interactions and dexterous controls from 44k hours of egocentric human videos. Our data mixture represents the largest video dataset to date for world model pretraining, spanning a wide range of daily scenarios with diverse objects and skills. To address the scarcity of action labels, we introduce continuous latent actions as unified proxy actions, enhancing interaction knowledge transfer from unlabeled videos. After post-training on small-scale target robot data, DreamDojo demonstrates a strong understanding of physics and precise action controllability. We also devise a distillation pipeline that accelerates DreamDojo to a real-time speed of 10.81 FPS and further improves context consistency. Our work enables several important applications based on generative world models, including live teleoperation, policy evaluation, and model-based planning. Systematic evaluation on multiple challenging out-of-distribution (OOD) benchmarks verifies the significance of our method for simulating open-world, contact-rich tasks, paving the way for general-purpose robot world models.
>
---
#### [new 016] DynaRetarget: Dynamically-Feasible Retargeting using Sampling-Based Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文提出DynaRetarget，解决人体动作到人形机器人控制策略的迁移问题。通过SBTO框架优化轨迹，提升动作的动态可行性，实现高效、通用的动作迁移。**

- **链接: [https://arxiv.org/pdf/2602.06827v1](https://arxiv.org/pdf/2602.06827v1)**

> **作者:** Victor Dhedin; Ilyass Taouil; Shafeef Omar; Dian Yu; Kun Tao; Angela Dai; Majid Khadiv
>
> **摘要:** In this paper, we introduce DynaRetarget, a complete pipeline for retargeting human motions to humanoid control policies. The core component of DynaRetarget is a novel Sampling-Based Trajectory Optimization (SBTO) framework that refines imperfect kinematic trajectories into dynamically feasible motions. SBTO incrementally advances the optimization horizon, enabling optimization over the entire trajectory for long-horizon tasks. We validate DynaRetarget by successfully retargeting hundreds of humanoid-object demonstrations and achieving higher success rates than the state of the art. The framework also generalizes across varying object properties, such as mass, size, and geometry, using the same tracking objective. This ability to robustly retarget diverse demonstrations opens the door to generating large-scale synthetic datasets of humanoid loco-manipulation trajectories, addressing a major bottleneck in real-world data collection.
>
---
#### [new 017] Consensus-based optimization (CBO): Towards Global Optimality in Robotics
- **分类: cs.RO**

- **简介: 该论文研究机器人轨迹优化任务，解决现有方法局部最优的问题，提出共识优化（CBO）方法，确保全局最优，实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2602.06868v1](https://arxiv.org/pdf/2602.06868v1)**

> **作者:** Xudong Sun; Armand Jordana; Massimo Fornasier; Jalal Etesami; Majid Khadiv
>
> **摘要:** Zero-order optimization has recently received significant attention for designing optimal trajectories and policies for robotic systems. However, most existing methods (e.g., MPPI, CEM, and CMA-ES) are local in nature, as they rely on gradient estimation. In this paper, we introduce consensus-based optimization (CBO) to robotics, which is guaranteed to converge to a global optimum under mild assumptions. We provide theoretical analysis and illustrative examples that give intuition into the fundamental differences between CBO and existing methods. To demonstrate the scalability of CBO for robotics problems, we consider three challenging trajectory optimization scenarios: (1) a long-horizon problem for a simple system, (2) a dynamic balance problem for a highly underactuated system, and (3) a high-dimensional problem with only a terminal cost. Our results show that CBO is able to achieve lower costs with respect to existing methods on all three challenging settings. This opens a new framework to study global trajectory optimization in robotics.
>
---
#### [new 018] SuReNav: Superpixel Graph-based Constraint Relaxation for Navigation in Over-constrained Environments
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于路径规划任务，解决过约束环境下的导航问题。提出SuReNav方法，通过超像素图和神经网络实现安全高效导航。**

- **链接: [https://arxiv.org/pdf/2602.06807v1](https://arxiv.org/pdf/2602.06807v1)**

> **作者:** Keonyoung Koh; Moonkyeong Jung; Samuel Seungsup Lee; Daehyung Park
>
> **备注:** Accepted by ICRA 2026. Code and videos are available at https://sure-nav.github.io/
>
> **摘要:** We address the over-constrained planning problem in semi-static environments. The planning objective is to find a best-effort solution that avoids all hard constraint regions while minimally traversing the least risky areas. Conventional methods often rely on pre-defined area costs, limiting generalizations. Further, the spatial continuity of navigation spaces makes it difficult to identify regions that are passable without overestimation. To overcome these challenges, we propose SuReNav, a superpixel graph-based constraint relaxation and navigation method that imitates human-like safe and efficient navigation. Our framework consists of three components: 1) superpixel graph map generation with regional constraints, 2) regional-constraint relaxation using graph neural network trained on human demonstrations for safe and efficient navigation, and 3) interleaving relaxation, planning, and execution for complete navigation. We evaluate our method against state-of-the-art baselines on 2D semantic maps and 3D maps from OpenStreetMap, achieving the highest human-likeness score of complete navigation while maintaining a balanced trade-off between efficiency and safety. We finally demonstrate its scalability and generalization performance in real-world urban navigation with a quadruped robot, Spot.
>
---
#### [new 019] Crowd-FM: Learned Optimal Selection of Conditional Flow Matching-generated Trajectories for Crowd Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决密集人群中的安全与拟人化导航问题。提出Crowd-FM方法，结合条件流匹配和人类轨迹评分，提升导航成功率与自然度。**

- **链接: [https://arxiv.org/pdf/2602.06698v1](https://arxiv.org/pdf/2602.06698v1)**

> **作者:** Antareep Singha; Laksh Nanwani; Mathai Mathew P.; Samkit Jain; Phani Teja Singamaneni; Arun Kumar Singh; K. Madhava Krishna
>
> **备注:** Accepted at IEEE ICRA 2026. Authors Antareep Singha and Laksh Nanwani have equal contributions
>
> **摘要:** Safe and computationally efficient local planning for mobile robots in dense, unstructured human crowds remains a fundamental challenge. Moreover, ensuring that robot trajectories are similar to how a human moves will increase the acceptance of the robot in human environments. In this paper, we present Crowd-FM, a learning-based approach to address both safety and human-likeness challenges. Our approach has two novel components. First, we train a Conditional Flow-Matching (CFM) policy over a dataset of optimally controlled trajectories to learn a set of collision-free primitives that a robot can choose at any given scenario. The chosen optimal control solver can generate multi-modal collision-free trajectories, allowing the CFM policy to learn a diverse set of maneuvers. Secondly, we learn a score function over a dataset of human demonstration trajectories that provides a human-likeness score for the flow primitives. At inference time, computing the optimal trajectory requires selecting the one with the highest score. Our approach improves the state-of-the-art by showing that our CFM policy alone can produce collision-free navigation with a higher success rate than existing learning-based baselines. Furthermore, when augmented with inference-time refinement, our approach can outperform even expensive optimisation-based planning approaches. Finally, we validate that our scoring network can select trajectories closer to the expert data than a manually designed cost function.
>
---
#### [new 020] ECO: Energy-Constrained Optimization with Reinforcement Learning for Humanoid Walking
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决人形机器人能耗高、效率低的问题。提出ECO框架，通过约束强化学习分离能量指标，提升行走的稳定性与能效。**

- **链接: [https://arxiv.org/pdf/2602.06445v1](https://arxiv.org/pdf/2602.06445v1)**

> **作者:** Weidong Huang; Jingwen Zhang; Jiongye Li; Shibowen Zhang; Jiayang Wu; Jiayi Wang; Hangxin Liu; Yaodong Yang; Yao Su
>
> **备注:** IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING. PREPRINT VERSION. ACCEPTED FEB, 2026
>
> **摘要:** Achieving stable and energy-efficient locomotion is essential for humanoid robots to operate continuously in real-world applications. Existing MPC and RL approaches often rely on energy-related metrics embedded within a multi-objective optimization framework, which require extensive hyperparameter tuning and often result in suboptimal policies. To address these challenges, we propose ECO (Energy-Constrained Optimization), a constrained RL framework that separates energy-related metrics from rewards, reformulating them as explicit inequality constraints. This method provides a clear and interpretable physical representation of energy costs, enabling more efficient and intuitive hyperparameter tuning for improved energy efficiency. ECO introduces dedicated constraints for energy consumption and reference motion, enforced by the Lagrangian method, to achieve stable, symmetric, and energy-efficient walking for humanoid robots. We evaluated ECO against MPC, standard RL with reward shaping, and four state-of-the-art constrained RL methods. Experiments, including sim-to-sim and sim-to-real transfers on the kid-sized humanoid robot BRUCE, demonstrate that ECO significantly reduces energy consumption compared to baselines while maintaining robust walking performance. These results highlight a substantial advancement in energy-efficient humanoid locomotion. All experimental demonstrations can be found on the project website: https://sites.google.com/view/eco-humanoid.
>
---
#### [new 021] Action Hallucination in Generative Visual-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，研究生成式视觉-语言-动作模型中的动作幻觉问题，分析其导致的物理约束违反和规划失败，提出改进可靠性的方向。**

- **链接: [https://arxiv.org/pdf/2602.06339v1](https://arxiv.org/pdf/2602.06339v1)**

> **作者:** Harold Soh; Eugene Lim
>
> **备注:** 22 pages
>
> **摘要:** Robot Foundation Models such as Vision-Language-Action models are rapidly reshaping how robot policies are trained and deployed, replacing hand-designed planners with end-to-end generative action models. While these systems demonstrate impressive generalization, it remains unclear whether they fundamentally resolve the long-standing challenges of robotics. We address this question by analyzing action hallucinations that violate physical constraints and their extension to plan-level failures. Focusing on latent-variable generative policies, we show that hallucinations often arise from structural mismatches between feasible robot behavior and common model architectures. We study three such barriers -- topological, precision, and horizon -- and show how they impose unavoidable tradeoffs. Our analysis provides mechanistic explanations for reported empirical failures of generative robot policies and suggests principled directions for improving reliability and trustworthiness, without abandoning their expressive power.
>
---
#### [new 022] Constraint Manifold Exploration for Efficient Continuous Coverage Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决工业机器人在复杂环境中实现表面全覆盖的可行性分析问题。提出一种基于采样的连续覆盖估计方法，探索配置空间中的可达区域。**

- **链接: [https://arxiv.org/pdf/2602.06749v1](https://arxiv.org/pdf/2602.06749v1)**

> **作者:** Robert Wilbrandt; Rüdiger Dillmann
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Many automated manufacturing processes rely on industrial robot arms to move process-specific tools along workpiece surfaces. In applications like grinding, sanding, spray painting, or inspection, they need to cover a workpiece fully while keeping their tools perpendicular to its surface. While there are approaches to generate trajectories for these applications, there are no sufficient methods for analyzing the feasibility of full surface coverage. This work proposes a sampling-based approach for continuous coverage estimation that explores reachable surface regions in the configuration space. We define an extended ambient configuration space that allows for the representation of tool position and orientation constraints. A continuation-based approach is used to explore it using two different sampling strategies. A thorough evaluation across different kinematics and environments analyzes their runtime and efficiency. This validates our ability to accurately and efficiently calculate surface coverage for complex surfaces in complicated environments.
>
---
#### [new 023] Force Generative Imitation Learning: Bridging Position Trajectory and Force Commands through Control Technique
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决接触密集任务中位置轨迹与力指令不匹配的问题。通过生成力指令模型结合反馈控制，提升机器人在未见过轨迹下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06620v1](https://arxiv.org/pdf/2602.06620v1)**

> **作者:** Hiroshi Sato; Sho Sakaino; Toshiaki Tsuji
>
> **备注:** Accepted for IEEE Access
>
> **摘要:** In contact-rich tasks, while position trajectories are often easy to obtain, appropriate force commands are typically unknown. Although it is conceivable to generate force commands using a pretrained foundation model such as Vision-Language-Action (VLA) models, force control is highly dependent on the specific hardware of the robot, which makes the application of such models challenging. To bridge this gap, we propose a force generative model that estimates force commands from given position trajectories. However, when dealing with unseen position trajectories, the model struggles to generate accurate force commands. To address this, we introduce a feedback control mechanism. Our experiments reveal that feedback control does not converge when the force generative model has memory. We therefore adopt a model without memory, enabling stable feedback control. This approach allows the system to generate force commands effectively, even for unseen position trajectories, improving generalization for real-world robot writing tasks.
>
---
#### [new 024] The Law of Task-Achieving Body Motion: Axiomatizing Success of Robot Manipulation Actions
- **分类: cs.RO**

- **简介: 该论文提出任务实现身体运动定律，解决机器人操作动作的正确性验证问题，通过分解任务为语义、因果和可行性三个谓词，支持运动合成与验证。**

- **链接: [https://arxiv.org/pdf/2602.06572v1](https://arxiv.org/pdf/2602.06572v1)**

> **作者:** Malte Huerkamp; Jonas Dech; Michael Beetz
>
> **备注:** 9 pages, 3 figures, submitted to the 2026 International Joint Conference on Artificial Intelligence (IJCAI)
>
> **摘要:** Autonomous agents that perform everyday manipulation actions need to ensure that their body motions are semantically correct with respect to a task request, causally effective within their environment, and feasible for their embodiment. In order to enable robots to verify these properties, we introduce the Law of Task-Achieving Body Motion as an axiomatic correctness specification for body motions. To that end we introduce scoped Task-Environment-Embodiment (TEE) classes that represent world states as Semantic Digital Twins (SDTs) and define applicable physics models to decompose task achievement into three predicates: SatisfiesRequest for semantic request satisfaction over SDT state evolution; Causes for causal sufficiency under the scoped physics model; and CanPerform for safety and feasibility verification at the embodiment level. This decomposition yields a reusable, implementation-independent interface that supports motion synthesis and the verification of given body motions. It also supports typed failure diagnosis (semantic, causal, embodiment and out-of-scope), feasibility across robots and environments, and counterfactual reasoning about robot body motions. We demonstrate the usability of the law in practice by instantiating it for articulated container manipulation in kitchen environments on three contrasting mobile manipulation platforms
>
---
#### [new 025] Beyond the Majority: Long-tail Imitation Learning for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决长尾分布下模仿学习性能下降的问题。通过分析传统方法的不足，提出APA方法提升尾部任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06512v1](https://arxiv.org/pdf/2602.06512v1)**

> **作者:** Junhong Zhu; Ji Zhang; Jingkuan Song; Lianli Gao; Heng Tao Shen
>
> **备注:** accept by IEEE International Conference on Robotics and Automation (ICRA 2026), 8 pages, 6 figures,
>
> **摘要:** While generalist robot policies hold significant promise for learning diverse manipulation skills through imitation, their performance is often hindered by the long-tail distribution of training demonstrations. Policies learned on such data, which is heavily skewed towards a few data-rich head tasks, frequently exhibit poor generalization when confronted with the vast number of data-scarce tail tasks. In this work, we conduct a comprehensive analysis of the pervasive long-tail challenge inherent in policy learning. Our analysis begins by demonstrating the inefficacy of conventional long-tail learning strategies (e.g., re-sampling) for improving the policy's performance on tail tasks. We then uncover the underlying mechanism for this failure, revealing that data scarcity on tail tasks directly impairs the policy's spatial reasoning capability. To overcome this, we introduce Approaching-Phase Augmentation (APA), a simple yet effective scheme that transfers knowledge from data-rich head tasks to data-scarce tail tasks without requiring external demonstrations. Extensive experiments in both simulation and real-world manipulation tasks demonstrate the effectiveness of APA. Our code and demos are publicly available at: https://mldxy.github.io/Project-VLA-long-tail/.
>
---
#### [new 026] Coupled Local and Global World Models for Efficient First Order RL
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习任务，旨在解决复杂环境下的操控问题。通过构建耦合的局部与全局世界模型，提升策略训练效率，避免依赖物理模拟器。**

- **链接: [https://arxiv.org/pdf/2602.06219v1](https://arxiv.org/pdf/2602.06219v1)**

> **作者:** Joseph Amigo; Rooholla Khorrambakht; Nicolas Mansard; Ludovic Righetti
>
> **摘要:** World models offer a promising avenue for more faithfully capturing complex dynamics, including contacts and non-rigidity, as well as complex sensory information, such as visual perception, in situations where standard simulators struggle. However, these models are computationally complex to evaluate, posing a challenge for popular RL approaches that have been successfully used with simulators to solve complex locomotion tasks but yet struggle with manipulation. This paper introduces a method that bypasses simulators entirely, training RL policies inside world models learned from robots' interactions with real environments. At its core, our approach enables policy training with large-scale diffusion models via a novel decoupled first-order gradient (FoG) method: a full-scale world model generates accurate forward trajectories, while a lightweight latent-space surrogate approximates its local dynamics for efficient gradient computation. This coupling of a local and global world model ensures high-fidelity unrolling alongside computationally tractable differentiation. We demonstrate the efficacy of our method on the Push-T manipulation task, where it significantly outperforms PPO in sample efficiency. We further evaluate our approach through an ego-centric object manipulation task with a quadruped. Together, these results demonstrate that learning inside data-driven world models is a promising pathway for solving hard-to-model RL tasks in image space without reliance on hand-crafted physics simulators.
>
---
#### [new 027] Humanoid Manipulation Interface: Humanoid Whole-Body Manipulation from Robot-Free Demonstrations
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于人形机器人操作任务，旨在解决传统方法依赖硬件和复杂奖励机制的问题。提出HuMI框架，通过无机器人数据采集实现高效学习，提升操作技能的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06643v1](https://arxiv.org/pdf/2602.06643v1)**

> **作者:** Ruiqian Nai; Boyuan Zheng; Junming Zhao; Haodong Zhu; Sicong Dai; Zunhao Chen; Yihang Hu; Yingdong Hu; Tong Zhang; Chuan Wen; Yang Gao
>
> **备注:** Website: https://humanoid-manipulation-interface.github.io
>
> **摘要:** Current approaches for humanoid whole-body manipulation, primarily relying on teleoperation or visual sim-to-real reinforcement learning, are hindered by hardware logistics and complex reward engineering. Consequently, demonstrated autonomous skills remain limited and are typically restricted to controlled environments. In this paper, we present the Humanoid Manipulation Interface (HuMI), a portable and efficient framework for learning diverse whole-body manipulation tasks across various environments. HuMI enables robot-free data collection by capturing rich whole-body motion using portable hardware. This data drives a hierarchical learning pipeline that translates human motions into dexterous and feasible humanoid skills. Extensive experiments across five whole-body tasks--including kneeling, squatting, tossing, walking, and bimanual manipulation--demonstrate that HuMI achieves a 3x increase in data collection efficiency compared to teleoperation and attains a 70% success rate in unseen environments.
>
---
#### [new 028] Perception-Control Coupled Visual Servoing for Textureless Objects Using Keypoint-Based EKF
- **分类: cs.RO**

- **简介: 该论文属于视觉伺服任务，旨在解决纹理缺失物体的视觉伺服问题。通过融合感知与控制，使用基于关键点的EKF估计6D位姿，并提出概率控制律提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06834v1](https://arxiv.org/pdf/2602.06834v1)**

> **作者:** Allen Tao; Jun Yang; Stanko Oparnica; Wenjie Xue
>
> **摘要:** Visual servoing is fundamental to robotic applications, enabling precise positioning and control. However, applying it to textureless objects remains a challenge due to the absence of reliable visual features. Moreover, adverse visual conditions, such as occlusions, often corrupt visual feedback, leading to reduced accuracy and instability in visual servoing. In this work, we build upon learning-based keypoint detection for textureless objects and propose a method that enhances robustness by tightly integrating perception and control in a closed loop. Specifically, we employ an Extended Kalman Filter (EKF) that integrates per-frame keypoint measurements to estimate 6D object pose, which drives pose-based visual servoing (PBVS) for control. The resulting camera motion, in turn, enhances the tracking of subsequent keypoints, effectively closing the perception-control loop. Additionally, unlike standard PBVS, we propose a probabilistic control law that computes both camera velocity and its associated uncertainty, enabling uncertainty-aware control for safe and reliable operation. We validate our approach on real-world robotic platforms using quantitative metrics and grasping experiments, demonstrating that our method outperforms traditional visual servoing techniques in both accuracy and practical application.
>
---
#### [new 029] MultiGraspNet: A Multitask 3D Vision Model for Multi-gripper Robotic Grasping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MultiGraspNet，解决多夹爪机器人抓取问题。通过统一框架同时预测平行与吸力夹爪的抓取位姿，提升抓取效率与适应性。**

- **链接: [https://arxiv.org/pdf/2602.06504v1](https://arxiv.org/pdf/2602.06504v1)**

> **作者:** Stephany Ortuno-Chanelo; Paolo Rabino; Enrico Civitelli; Tatiana Tommasi; Raffaello Camoriano
>
> **摘要:** Vision-based models for robotic grasping automate critical, repetitive, and draining industrial tasks. Existing approaches are typically limited in two ways: they either target a single gripper and are potentially applied on costly dual-arm setups, or rely on custom hybrid grippers that require ad-hoc learning procedures with logic that cannot be transferred across tasks, restricting their general applicability. In this work, we present MultiGraspNet, a novel multitask 3D deep learning method that predicts feasible poses simultaneously for parallel and vacuum grippers within a unified framework, enabling a single robot to handle multiple end effectors. The model is trained on the richly annotated GraspNet-1Billion and SuctionNet-1Billion datasets, which have been aligned for the purpose, and generates graspability masks quantifying the suitability of each scene point for successful grasps. By sharing early-stage features while maintaining gripper-specific refiners, MultiGraspNet effectively leverages complementary information across grasping modalities, enhancing robustness and adaptability in cluttered scenes. We characterize MultiGraspNet's performance with an extensive experimental analysis, demonstrating its competitiveness with single-task models on relevant benchmarks. We run real-world experiments on a single-arm multi-gripper robotic setup showing that our approach outperforms the vacuum baseline, grasping 16% percent more seen objects and 32% more of the novel ones, while obtaining competitive results for the parallel task.
>
---
#### [new 030] Nipping the Drift in the Bud: Retrospective Rectification for Robust Vision-Language Navigation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于视觉语言导航任务，解决模仿学习中的暴露偏差问题。提出BudVLN框架，通过在线学习和回溯修正提升导航鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06356v1](https://arxiv.org/pdf/2602.06356v1)**

> **作者:** Gang He; Zhenyang Liu; Kepeng Xu; Li Xu; Tong Qiao; Wenxin Yu; Chang Wu; Weiying Xie
>
> **摘要:** Vision-Language Navigation (VLN) requires embodied agents to interpret natural language instructions and navigate through complex continuous 3D environments. However, the dominant imitation learning paradigm suffers from exposure bias, where minor deviations during inference lead to compounding errors. While DAgger-style approaches attempt to mitigate this by correcting error states, we identify a critical limitation: Instruction-State Misalignment. Forcing an agent to learn recovery actions from off-track states often creates supervision signals that semantically conflict with the original instruction. In response to these challenges, we introduce BudVLN, an online framework that learns from on-policy rollouts by constructing supervision to match the current state distribution. BudVLN performs retrospective rectification via counterfactual re-anchoring and decision-conditioned supervision synthesis, using a geodesic oracle to synthesize corrective trajectories that originate from valid historical states, ensuring semantic consistency. Experiments on the standard R2R-CE and RxR-CE benchmarks demonstrate that BudVLN consistently mitigates distribution shift and achieves state-of-the-art performance in both Success Rate and SPL.
>
---
#### [new 031] MORPH Wheel: A Passive Variable-Radius Wheel Embedding Mechanical Behavior Logic for Input-Responsive Transformation
- **分类: cs.RO**

- **简介: 该论文属于机器人机械设计任务，旨在解决被动可变传动问题。通过机械结构实现扭矩响应的半径自适应，提升机器人在复杂环境中的运动性能。**

- **链接: [https://arxiv.org/pdf/2602.06265v1](https://arxiv.org/pdf/2602.06265v1)**

> **作者:** JaeHyung Jang; JuYeong Seo; Dae-Young Lee; Jee-Hwan Ryu
>
> **备注:** 14 pages, 16 figures. Under review at IEEE Transactions on Robotics
>
> **摘要:** This paper introduces the Mechacnially prOgrammed Radius-adjustable PHysical (MORPH) wheel, a fully passive variable-radius wheel that embeds mechanical behavior logic for torque-responsive transformation. Unlike conventional variable transmission systems relying on actuators, sensors, and active control, the MORPH wheel achieves passive adaptation solely through its geometry and compliant structure. The design integrates a torque-response coupler and spring-loaded connecting struts to mechanically adjust the wheel radius between 80 mm and 45 mm in response to input torque, without any electrical components. The MORPH wheel provides three unique capabilities rarely achieved simultaneously in previous passive designs: (1) bidirectional operation with unlimited rotation through a symmetric coupler; (2) high torque capacity exceeding 10 N with rigid power transmission in drive mode; and (3) precise and repeatable transmission ratio control governed by deterministic kinematics. A comprehensive analytical model was developed to describe the wheel's mechanical behavior logic, establishing threshold conditions for mode switching between direct drive and radius transformation. Experimental validation confirmed that the measured torque-radius and force-displacement characteristics closely follow theoretical predictions across wheel weights of 1.8-2.8kg. Robot-level demonstrations on varying loads (0-25kg), slopes, and unstructured terrains further verified that the MORPH wheel passively adjusts its radius to provide optimal transmission ratio. The MORPH wheel exemplifies a mechanically programmed structure, embedding intelligent, context-dependent behavior directly into its physical design. This approach offers a new paradigm for passive variable transmission and mechanical intelligence in robotic mobility systems operating in unpredictable or control-limited environments.
>
---
#### [new 032] Active Localization of Unstable Systems with Coarse Information
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究不稳定系统的主动定位问题，解决在粗略单比特传感下如何恢复初始状态。通过设计结合集估计和控制策略的算法，实现状态不确定性指数收缩。**

- **链接: [https://arxiv.org/pdf/2602.06191v1](https://arxiv.org/pdf/2602.06191v1)**

> **作者:** Ege Yuceel; Daniel Liberzon; Sayan Mitra
>
> **备注:** 10 pages, 4 figures, Accepted by International Conference on Hybrid Systems: Computation and Control (HSCC) 2026
>
> **摘要:** We study localization and control for unstable systems under coarse, single-bit sensing. Motivated by understanding the fundamental limitations imposed by such minimal feedback, we identify sufficient conditions under which the initial state can be recovered despite instability and extremely sparse measurements. Building on these conditions, we develop an active localization algorithm that integrates a set-based estimator with a control strategy derived from Voronoi partitions, which provably estimates the initial state while ensuring the agent remains in informative regions. Under the derived conditions, the proposed approach guarantees exponential contraction of the initial-state uncertainty, and the result is further supported by numerical experiments. These findings can offer theoretical insight into localization in robotics, where sensing is often limited to coarse abstractions such as keyframes, segmentations, or line-based features.
>
---
#### [new 033] World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决世界模型与策略学习的协同优化问题。提出World-VLA-Loop框架，通过闭环学习提升动作跟随精度和策略性能。**

- **链接: [https://arxiv.org/pdf/2602.06508v1](https://arxiv.org/pdf/2602.06508v1)**

> **作者:** Xiaokang Liu; Zechen Bai; Hai Ci; Kevin Yuchen Ma; Mike Zheng Shou
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Recent progress in robotic world models has leveraged video diffusion transformers to predict future observations conditioned on historical states and actions. While these models can simulate realistic visual outcomes, they often exhibit poor action-following precision, hindering their utility for downstream robotic learning. In this work, we introduce World-VLA-Loop, a closed-loop framework for the joint refinement of world models and Vision-Language-Action (VLA) policies. We propose a state-aware video world model that functions as a high-fidelity interactive simulator by jointly predicting future observations and reward signals. To enhance reliability, we introduce the SANS dataset, which incorporates near-success trajectories to improve action-outcome alignment within the world model. This framework enables a closed-loop for reinforcement learning (RL) post-training of VLA policies entirely within a virtual environment. Crucially, our approach facilitates a co-evolving cycle: failure rollouts generated by the VLA policy are iteratively fed back to refine the world model precision, which in turn enhances subsequent RL optimization. Evaluations across simulation and real-world tasks demonstrate that our framework significantly boosts VLA performance with minimal physical interaction, establishing a mutually beneficial relationship between world modeling and policy learning for general-purpose robotics. Project page: https://showlab.github.io/World-VLA-Loop/.
>
---
#### [new 034] A 26-Gram Butterfly-Inspired Robot Achieving Autonomous Tailless Flight
- **分类: cs.RO**

- **简介: 该论文属于微飞行器设计任务，旨在解决尾部缺失的双翼飞行器稳定控制问题。研究提出一种26克蝴蝶仿生飞行器，通过特定控制策略实现自主飞行。**

- **链接: [https://arxiv.org/pdf/2602.06811v1](https://arxiv.org/pdf/2602.06811v1)**

> **作者:** Weibin Gu; Chenrui Feng; Lian Liu; Chen Yang; Xingchi Jiao; Yuhe Ding; Xiaofei Shi; Chao Gao; Alessandro Rizzo; Guyue Zhou
>
> **摘要:** Flapping-wing micro air vehicles (FWMAVs) have demonstrated remarkable bio-inspired agility, yet tailless two-winged configurations remain largely unexplored due to their complex fluid-structure and wing-body coupling. Here we present \textit{AirPulse}, a 26-gram butterfly-inspired FWMAV that achieves fully onboard, closed-loop, untethered flight without auxiliary control surfaces. The AirPulse robot replicates key biomechanical traits of butterfly flight, including low wing aspect ratio, compliant carbon-fiber-reinforced wings, and low-frequency, high-amplitude flapping that induces cyclic variations in the center of gravity and moment of inertia, producing characteristic body undulation. We establish a quantitative mapping between flapping modulation parameters and force-torque generation, and introduce the Stroke Timing Asymmetry Rhythm (STAR) generator, enabling smooth, stable, and linearly parameterized wingstroke asymmetry for flapping control. Integrating these with an attitude controller, the AirPulse robot maintains pitch and yaw stability despite strong oscillatory dynamics. Free-flight experiments demonstrate stable climbing and turning maneuvers via either angle offset or stroke timing modulation, marking the first onboard controlled flight of the lightest two-winged, tailless butterfly-inspired FWMAV reported in peer-reviewed literature. This work corroborates a foundational platform for lightweight, collision-proof FWMAVs, bridging biological inspiration with practical aerial robotics. Their non-invasive maneuverability is ideally suited for real-world applications, such as confined-space inspection and ecological monitoring, inaccessible to traditional drones, while their biomechanical fidelity provides a physical model to decode the principles underlying the erratic yet efficient flight of real butterflies.
>
---
#### [new 035] Transformer-Based Reinforcement Learning for Autonomous Orbital Collision Avoidance in Partially Observable Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于轨道碰撞规避任务，解决部分可观测环境下的自主避撞问题。通过引入基于Transformer的POMDP框架，提升在观测不全情况下的决策能力。**

- **链接: [https://arxiv.org/pdf/2602.06088v1](https://arxiv.org/pdf/2602.06088v1)**

> **作者:** Thomas Georges; Adam Abdin
>
> **摘要:** We introduce a Transformer-based Reinforcement Learning framework for autonomous orbital collision avoidance that explicitly models the effects of partial observability and imperfect monitoring in space operations. The framework combines a configurable encounter simulator, a distance-dependent observation model, and a sequential state estimator to represent uncertainty in relative motion. A central contribution of this work is the use of transformer-based Partially Observable Markov Decision Process (POMDP) architecture, which leverage long-range temporal attention to interpret noisy and intermittent observations more effectively than traditional architectures. This integration provides a foundation for training collision avoidance agents that can operate more reliably under imperfect monitoring environments.
>
---
#### [new 036] User-Centric Object Navigation: A Benchmark with Integrated User Habits for Personalized Embodied Object Search
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决个性化家庭环境中物体定位问题。通过引入用户习惯基准UcON，提升导航代理的适应能力。**

- **链接: [https://arxiv.org/pdf/2602.06459v1](https://arxiv.org/pdf/2602.06459v1)**

> **作者:** Hongcheng Wang; Jinyu Zhu; Hao Dong
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** In the evolving field of robotics, the challenge of Object Navigation (ON) in household environments has attracted significant interest. Existing ON benchmarks typically place objects in locations guided by general scene priors, without accounting for the specific placement habits of individual users. This omission limits the adaptability of navigation agents in personalized household environments. To address this, we introduce User-centric Object Navigation (UcON), a new benchmark that incorporates user-specific object placement habits, referred to as user habits. This benchmark requires agents to leverage these user habits for more informed decision-making during navigation. UcON encompasses approximately 22,600 user habits across 489 object categories. UcON is, to our knowledge, the first benchmark that explicitly formalizes and evaluates habit-conditioned object navigation at scale and covers the widest range of target object categories. Additionally, we propose a habit retrieval module to extract and utilize habits related to target objects, enabling agents to infer their likely locations more effectively. Experimental results demonstrate that current SOTA methods exhibit substantial performance degradation under habit-driven object placement, while integrating user habits consistently improves success rates. Code is available at https://github.com/whcpumpkin/User-Centric-Object-Navigation.
>
---
#### [new 037] Now You See That: Learning End-to-End Humanoid Locomotion from Raw Pixels
- **分类: cs.RO**

- **简介: 该论文属于视觉驱动的人形机器人行走任务，解决sim-to-real感知噪声和地形适应问题。通过高保真传感器模拟和多 critic 学习，提升行走鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.06382v1](https://arxiv.org/pdf/2602.06382v1)**

> **作者:** Wandong Sun; Yongbo Su; Leoric Huang; Alex Zhang; Dwyane Wei; Mu San; Daniel Tian; Ellie Cao; Finn Yan; Ethan Xie; Zongwu Xie
>
> **摘要:** Achieving robust vision-based humanoid locomotion remains challenging due to two fundamental issues: the sim-to-real gap introduces significant perception noise that degrades performance on fine-grained tasks, and training a unified policy across diverse terrains is hindered by conflicting learning objectives. To address these challenges, we present an end-to-end framework for vision-driven humanoid locomotion. For robust sim-to-real transfer, we develop a high-fidelity depth sensor simulation that captures stereo matching artifacts and calibration uncertainties inherent in real-world sensing. We further propose a vision-aware behavior distillation approach that combines latent space alignment with noise-invariant auxiliary tasks, enabling effective knowledge transfer from privileged height maps to noisy depth observations. For versatile terrain adaptation, we introduce terrain-specific reward shaping integrated with multi-critic and multi-discriminator learning, where dedicated networks capture the distinct dynamics and motion priors of each terrain type. We validate our approach on two humanoid platforms equipped with different stereo depth cameras. The resulting policy demonstrates robust performance across diverse environments, seamlessly handling extreme challenges such as high platforms and wide gaps, as well as fine-grained tasks including bidirectional long-term staircase traversal.
>
---
#### [new 038] TFusionOcc: Student's t-Distribution Based Object-Centric Multi-Sensor Fusion Framework for 3D Occupancy Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，旨在解决多传感器融合中几何细节捕捉不足的问题。提出TFusionOcc框架，利用t分布和可变形超二次曲面提升预测精度。**

- **链接: [https://arxiv.org/pdf/2602.06400v1](https://arxiv.org/pdf/2602.06400v1)**

> **作者:** Zhenxing Ming; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **摘要:** 3D semantic occupancy prediction enables autonomous vehicles (AVs) to perceive fine-grained geometric and semantic structure of their surroundings from onboard sensors, which is essential for safe decision-making and navigation. Recent models for 3D semantic occupancy prediction have successfully addressed the challenge of describing real-world objects with varied shapes and classes. However, the intermediate representations used by existing methods for 3D semantic occupancy prediction rely heavily on 3D voxel volumes or a set of 3D Gaussians, hindering the model's ability to efficiently and effectively capture fine-grained geometric details in the 3D driving environment. This paper introduces TFusionOcc, a novel object-centric multi-sensor fusion framework for predicting 3D semantic occupancy. By leveraging multi-stage multi-sensor fusion, Student's t-distribution, and the T-Mixture model (TMM), together with more geometrically flexible primitives, such as the deformable superquadric (superquadric with inverse warp), the proposed method achieved state-of-the-art (SOTA) performance on the nuScenes benchmark. In addition, extensive experiments were conducted on the nuScenes-C dataset to demonstrate the robustness of the proposed method in different camera and lidar corruption scenarios. The code will be available at: https://github.com/DanielMing123/TFusionOcc
>
---
#### [new 039] Efficient and Robust Modeling of Nonlinear Mechanical Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于系统与控制工程领域，旨在提升非线性机械系统的动态建模效率与鲁棒性。提出一种新模型及自动生成方法，解决传统方法在噪声干扰和计算速度上的不足。**

- **链接: [https://arxiv.org/pdf/2602.06639v1](https://arxiv.org/pdf/2602.06639v1)**

> **作者:** Davide Tebaldi; Roberto Zanasi
>
> **摘要:** The development of efficient and robust dynamic models is fundamental in the field of systems and control engineering. In this paper, a new formulation for the dynamic model of nonlinear mechanical systems, that can be applied to different automotive and robotic case studies, is proposed, together with a modeling procedure allowing to automatically obtain the model formulation. Compared with the Euler-Lagrange formulation, the proposed model is shown to give superior performances in terms of robustness against measurement noise for systems exhibiting dependence on some external variables, as well as in terms of execution time when computing the inverse dynamics of the system.
>
---
#### [new 040] LIBERO-X: Robustness Litmus for Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出LIBERO-X基准，用于评估视觉-语言-动作模型的鲁棒性。针对现有基准评估不足的问题，设计分层评估协议和多样化数据集，提升模型测试可靠性。**

- **链接: [https://arxiv.org/pdf/2602.06556v1](https://arxiv.org/pdf/2602.06556v1)**

> **作者:** Guodong Wang; Chenkai Zhang; Qingjie Liu; Jinjin Zhang; Jiancheng Cai; Junjie Liu; Xinmin Liu
>
> **备注:** 19 pages, 14 figures and 8 tables
>
> **摘要:** Reliable benchmarking is critical for advancing Vision-Language-Action (VLA) models, as it reveals their generalization, robustness, and alignment of perception with language-driven manipulation tasks. However, existing benchmarks often provide limited or misleading assessments due to insufficient evaluation protocols that inadequately capture real-world distribution shifts. This work systematically rethinks VLA benchmarking from both evaluation and data perspectives, introducing LIBERO-X, a benchmark featuring: 1) A hierarchical evaluation protocol with progressive difficulty levels targeting three core capabilities: spatial generalization, object recognition, and task instruction understanding. This design enables fine-grained analysis of performance degradation under increasing environmental and task complexity; 2) A high-diversity training dataset collected via human teleoperation, where each scene supports multiple fine-grained manipulation objectives to bridge the train-evaluation distribution gap. Experiments with representative VLA models reveal significant performance drops under cumulative perturbations, exposing persistent limitations in scene comprehension and instruction grounding. By integrating hierarchical evaluation with diverse training data, LIBERO-X offers a more reliable foundation for assessing and advancing VLA development.
>
---
#### [new 041] Addressing the Waypoint-Action Gap in End-to-End Autonomous Driving via Vehicle Motion Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决waypoint与action之间的差距问题。通过引入可微车辆模型框架，使基于动作的策略能在基于waypoint的基准中训练和评估。**

- **链接: [https://arxiv.org/pdf/2602.06214v1](https://arxiv.org/pdf/2602.06214v1)**

> **作者:** Jorge Daniel Rodríguez-Vidal; Gabriel Villalonga; Diego Porres; Antonio M. López Peña
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** End-to-End Autonomous Driving (E2E-AD) systems are typically grouped by the nature of their outputs: (i) waypoint-based models that predict a future trajectory, and (ii) action-based models that directly output throttle, steer and brake. Most recent benchmark protocols and training pipelines are waypoint-based, which makes action-based policies harder to train and compare, slowing their progress. To bridge this waypoint-action gap, we propose a novel, differentiable vehicle-model framework that rolls out predicted action sequences to their corresponding ego-frame waypoint trajectories while supervising in waypoint space. Our approach enables action-based architectures to be trained and evaluated, for the first time, within waypoint-based benchmarks without modifying the underlying evaluation protocol. We extensively evaluate our framework across multiple challenging benchmarks and observe consistent improvements over the baselines. In particular, on NAVSIM \texttt{navhard} our approach achieves state-of-the-art performance. Our code will be made publicly available upon acceptance.
>
---
#### [new 042] AnyThermal: Towards Learning Universal Representations for Thermal Perception
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出AnyThermal，解决热成像特征提取任务，通过多环境数据训练，提升模型泛化能力，适用于多种下游任务。**

- **链接: [https://arxiv.org/pdf/2602.06203v1](https://arxiv.org/pdf/2602.06203v1)**

> **作者:** Parv Maheshwari; Jay Karhade; Yogesh Chawla; Isaiah Adu; Florian Heisen; Andrew Porco; Andrew Jong; Yifei Liu; Santosh Pitla; Sebastian Scherer; Wenshan Wang
>
> **备注:** Accepted at IEEE ICRA (International Conference on Robotics & Automation) 2026
>
> **摘要:** We present AnyThermal, a thermal backbone that captures robust task-agnostic thermal features suitable for a variety of tasks such as cross-modal place recognition, thermal segmentation, and monocular depth estimation using thermal images. Existing thermal backbones that follow task-specific training from small-scale data result in utility limited to a specific environment and task. Unlike prior methods, AnyThermal can be used for a wide range of environments (indoor, aerial, off-road, urban) and tasks, all without task-specific training. Our key insight is to distill the feature representations from visual foundation models such as DINOv2 into a thermal encoder using thermal data from these multiple environments. To bridge the diversity gap of the existing RGB-Thermal datasets, we introduce the TartanRGBT platform, the first open-source data collection platform with synced RGB-Thermal image acquisition. We use this payload to collect the TartanRGBT dataset - a diverse and balanced dataset collected in 4 environments. We demonstrate the efficacy of AnyThermal and TartanRGBT, achieving state-of-the-art results with improvements of up to 36% across diverse environments and downstream tasks on existing datasets.
>
---
#### [new 043] DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DriveWorld-VLA，解决自动驾驶中场景演化与动作规划统一的问题。通过融合视觉-语言-动作与世界模型，在潜在空间实现联合建模与决策，提升感知与规划效果。**

- **链接: [https://arxiv.org/pdf/2602.06521v1](https://arxiv.org/pdf/2602.06521v1)**

> **作者:** Feiyang jia; Lin Liu; Ziying Song; Caiyan Jia; Hangjun Ye; Xiaoshuai Hao; Long Chen
>
> **备注:** 20 pages, 7 tables, 12 figures
>
> **摘要:** End-to-end (E2E) autonomous driving has recently attracted increasing interest in unifying Vision-Language-Action (VLA) with World Models to enhance decision-making and forward-looking imagination. However, existing methods fail to effectively unify future scene evolution and action planning within a single architecture due to inadequate sharing of latent states, limiting the impact of visual imagination on action decisions. To address this limitation, we propose DriveWorld-VLA, a novel framework that unifies world modeling and planning within a latent space by tightly integrating VLA and world models at the representation level, which enables the VLA planner to benefit directly from holistic scene-evolution modeling and reducing reliance on dense annotated supervision. Additionally, DriveWorld-VLA incorporates the latent states of the world model as core decision-making states for the VLA planner, facilitating the planner to assess how candidate actions impact future scene evolution. By conducting world modeling entirely in the latent space, DriveWorld-VLA supports controllable, action-conditioned imagination at the feature level, avoiding expensive pixel-level rollouts. Extensive open-loop and closed-loop evaluations demonstrate the effectiveness of DriveWorld-VLA, which achieves state-of-the-art performance with 91.3 PDMS on NAVSIMv1, 86.8 EPDMS on NAVSIMv2, and 0.16 3-second average collision rate on nuScenes. Code and models will be released in https://github.com/liulin815/DriveWorld-VLA.git.
>
---
#### [new 044] Bridging the Indoor-Outdoor Gap: Vision-Centric Instruction-Guided Embodied Navigation for the Last Meters
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于室内外导航任务，解决室外到室内无缝过渡问题。提出无需外部先验的视觉引导导航框架，并构建首个相关数据集，提升导航精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.06427v1](https://arxiv.org/pdf/2602.06427v1)**

> **作者:** Yuxiang Zhao; Yirong Yang; Yanqing Zhu; Yanfen Shen; Chiyu Wang; Zhining Gu; Pei Shi; Wei Guo; Mu Xu
>
> **摘要:** Embodied navigation holds significant promise for real-world applications such as last-mile delivery. However, most existing approaches are confined to either indoor or outdoor environments and rely heavily on strong assumptions, such as access to precise coordinate systems. While current outdoor methods can guide agents to the vicinity of a target using coarse-grained localization, they fail to enable fine-grained entry through specific building entrances, critically limiting their utility in practical deployment scenarios that require seamless outdoor-to-indoor transitions. To bridge this gap, we introduce a novel task: out-to-in prior-free instruction-driven embodied navigation. This formulation explicitly eliminates reliance on accurate external priors, requiring agents to navigate solely based on egocentric visual observations guided by instructions. To tackle this task, we propose a vision-centric embodied navigation framework that leverages image-based prompts to drive decision-making. Additionally, we present the first open-source dataset for this task, featuring a pipeline that integrates trajectory-conditioned video synthesis into the data generation process. Through extensive experiments, we demonstrate that our proposed method consistently outperforms state-of-the-art baselines across key metrics including success rate and path efficiency.
>
---
## 更新

#### [replaced 001] Observability-Aware Control for Quadrotor Formation Flight with Range-only Measurement
- **分类: cs.RO**

- **简介: 该论文属于无人机编队控制任务，解决基于单距离测量的定位可观测性问题。提出STLOG和OPC方法，提升定位精度与系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2411.03747v3](https://arxiv.org/pdf/2411.03747v3)**

> **作者:** H S Helson Go; Ching Lok Chong; Longhao Qian; Hugh H. -T. Liu
>
> **备注:** 37 pages, 5 figures
>
> **摘要:** Cooperative Localization is a promising approach to achieving safe quadrotor formation flight through precise positioning via low-cost inter-drone sensors. This paper develops an observability-aware control principle tailored to quadrotor formation flight with range-only inter-drone measurements. The control principle is based on a novel approximation of the local observability Gramian (LOG), which we name the Short-Term Local Observability Gramian (STLOG). The validity of STLOG is established by proving its link to directional estimation precision in nonlinear systems. We propose the Observability Predictive Controller (OPC), a receding-horizon controller that generates optimal inputs to enhance information gain in weakly observable state directions by maximizing the minimum eigenvalue of the STLOG. This reduces the risk of estimator divergence due to the unbounded growth of uncertainty in weakly observed state components. Monte Carlo simulations and flight experiments are conducted with quadrotors in a GNSS-denied ferrying mission, showing that the OPC improves positioning confidence and estimator robustness.
>
---
#### [replaced 002] Right-Side-Out: Learning Zero-Shot Sim-to-Real Garment Reversal
- **分类: cs.RO**

- **简介: 该论文研究衣物翻转任务，解决真实环境中动态、遮挡等问题。通过任务分解与高保真仿真，实现零样本模拟到现实的控制。**

- **链接: [https://arxiv.org/pdf/2509.15953v2](https://arxiv.org/pdf/2509.15953v2)**

> **作者:** Chang Yu; Siyu Ma; Wenxin Du; Zeshun Zong; Han Xue; Wendi Chen; Cewu Lu; Yin Yang; Xuchen Han; Joseph Masterjohn; Alejandro Castro; Chenfanfu Jiang
>
> **备注:** More details and supplementary material are on the website: https://right-side-out.github.io
>
> **摘要:** Turning garments right-side out is a challenging manipulation task: it is highly dynamic, entails rapid contact changes, and is subject to severe visual occlusion. We introduce Right-Side-Out, a zero-shot sim-to-real framework that effectively solves this challenge by exploiting task structures. We decompose the task into Drag/Fling to create and stabilize an access opening, followed by Insert&Pull to invert the garment. Each step uses a depth-inferred, keypoint-parameterized bimanual primitive that sharply reduces the action space while preserving robustness. Efficient data generation is enabled by our custom-built, high-fidelity, GPU-parallel Material Point Method (MPM) simulator that models thin-shell deformation and provides robust and efficient contact handling for batched rollouts. Built on the simulator, our fully automated pipeline scales data generation by randomizing garment geometry, material parameters, and viewpoints, producing depth, masks, and per-primitive keypoint labels without any human annotations. With a single depth camera, policies trained entirely in simulation deploy zero-shot on real hardware, achieving up to 81.3% success rate. By employing task decomposition and high fidelity simulation, our framework enables tackling highly dynamic, severely occluded tasks without laborious human demonstrations.
>
---
#### [replaced 003] Encoding Tactile Stimuli for Braille Recognition with Organoids
- **分类: cs.NE; cs.ET; cs.RO**

- **简介: 该论文属于生物混合计算任务，旨在通过电刺激编码实现触觉信息的脑机接口应用。研究提出一种编码策略，利用器官oids进行盲文分类，提升识别准确率与抗噪能力。**

- **链接: [https://arxiv.org/pdf/2508.20850v2](https://arxiv.org/pdf/2508.20850v2)**

> **作者:** Tianyi Liu; Hemma Philamore; Benjamin Ward-Cherrier
>
> **摘要:** This study proposes a transferable encoding strategy that maps tactile sensor data to electrical stimulation patterns, enabling neural organoids to perform an open-loop artificial tactile Braille classification task. Human forebrain organoids cultured on a low-density microelectrode array (MEA) are systematically stimulated to characterize the relationship between electrical stimulation parameters (number of pulse, phase amplitude, phase duration, and trigger delay) and organoid responses, measured as spike activity and spatial displacement of the center of activity. Implemented on event-based tactile inputs recorded from the Evetac sensor, our system achieved an average Braille letter classification accuracy of 61% with a single organoid, which increased significantly to 83% when responses from a three-organoid ensemble were combined. Additionally, the multi-organoid configuration demonstrated enhanced robustness against various types of artificially introduced noise. This research demonstrates the potential of organoids as low-power, adaptive bio-hybrid computational elements and provides a foundational encoding framework for future scalable bio-hybrid computing architectures.
>
---
#### [replaced 004] GNSS-based Lunar Orbit and Clock Estimation With Stochastic Cloning UD Filter
- **分类: cs.RO**

- **简介: 该论文属于卫星轨道与钟差估计任务，解决月球距离下观测条件差的问题，提出一种改进的滤波方法，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2601.16393v2](https://arxiv.org/pdf/2601.16393v2)**

> **作者:** Keidai Iiyama; Grace Gao
>
> **备注:** Submitted to the Journal of Guidance, Control, and Dynamics
>
> **摘要:** This paper presents a terrestrial GNSS-based orbit and clock estimation framework for lunar navigation satellites. To enable high-precision estimation under the low-observability conditions encountered at lunar distances, we develop a stochastic-cloning UD-factorized filter and delayed-state smoother that provide enhanced numerical stability when processing precise time-differenced carrier phase (TDCP) measurements. A comprehensive dynamics and measurement model is formulated, explicitly accounting for relativistic coupling between orbital and clock states, lunar time-scale transformations, and signal propagation delays including ionospheric, plasmaspheric, and Shapiro effects. The proposed approach is evaluated using high-fidelity Monte-Carlo simulations incorporating realistic multi-constellation GNSS geometry, broadcast ephemeris errors, lunar satellite dynamics, and ionospheric and plasmaspheric delay computed from empirical electron density models. Simulation results demonstrate that combining ionosphere-free pseudorange and TDCP measurements achieves meter-level orbit accuracy and sub-millimeter-per-second velocity accuracy, satisfying the stringent signal-in-space error requirements of future Lunar Augmented Navigation Services (LANS).
>
---
#### [replaced 005] A Taxonomy for Evaluating Generalist Robot Manipulation Policies
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作领域，旨在解决通用化策略评估难题。提出STAR-Gen分类体系，用于系统衡量视觉、语义和行为泛化能力。**

- **链接: [https://arxiv.org/pdf/2503.01238v3](https://arxiv.org/pdf/2503.01238v3)**

> **作者:** Jensen Gao; Suneel Belkhale; Sudeep Dasari; Ashwin Balakrishna; Dhruv Shah; Dorsa Sadigh
>
> **备注:** IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Machine learning for robot manipulation promises to unlock generalization to novel tasks and environments. But how should we measure the progress of these policies towards generalization? Evaluating and quantifying generalization is the Wild West of modern robotics, with each work proposing and measuring different types of generalization in their own, often difficult to reproduce settings. In this work, our goal is (1) to outline the forms of generalization we believe are important for robot manipulation in a comprehensive and fine-grained manner, and (2) to provide reproducible guidelines for measuring these notions of generalization. We first propose STAR-Gen, a taxonomy of generalization for robot manipulation structured around visual, semantic, and behavioral generalization. Next, we instantiate STAR-Gen with two case studies on real-world benchmarking: one based on open-source models and the Bridge V2 dataset, and another based on the bimanual ALOHA 2 platform that covers more dexterous and longer horizon tasks. Our case studies reveal many interesting insights: for example, we observe that open-source vision-language-action models often struggle with semantic generalization, despite pre-training on internet-scale language datasets. We provide videos and other supplementary material at stargen-taxonomy.github.io.
>
---
#### [replaced 006] CRISP -- Compliant ROS2 Controllers for Learning-Based Manipulation Policies and Teleoperation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决学习型策略与机械臂的兼容性问题。提出CRISP控制器，实现平滑参考跟踪和实时 compliant 行为。**

- **链接: [https://arxiv.org/pdf/2509.06819v2](https://arxiv.org/pdf/2509.06819v2)**

> **作者:** Daniel San José Pro; Oliver Hausdörfer; Ralf Römer; Maximilian Dösch; Martin Schuck; Angela P. Schoellig
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Learning-based controllers, such as diffusion policies and vision-language action models, often generate low-frequency or discontinuous robot state changes. Achieving smooth reference tracking requires a low-level controller that converts high-level targets commands into joint torques, enabling compliant behavior during contact interactions. We present CRISP, a lightweight C++ implementation of compliant Cartesian and joint-space controllers for the ROS2 control standard, designed for seamless integration with high-level learning-based policies as well as teleoperation. The controllers are compatible with any manipulator that exposes a joint-torque interface. Through our Python and Gymnasium interfaces, CRISP provides a unified pipeline for recording data from hardware and simulation and deploying high-level learning-based policies seamlessly, facilitating rapid experimentation. The system has been validated on hardware with the Franka Robotics FR3 and in simulation with the Kuka IIWA14 and Kinova Gen3. Designed for rapid integration, flexible deployment, and real-time performance, our implementation provides a unified pipeline for data collection and policy execution, lowering the barrier to applying learning-based methods on ROS2-compatible manipulators. Detailed documentation is available at the project website - https://utiasDSL.github.io/crisp_controllers.
>
---
#### [replaced 007] REACT: Real-time Entanglement-Aware Coverage Path Planning for Tethered Underwater Vehicles
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于路径规划任务，解决水下机器人 tether entanglement 问题。提出 REACT 框架，实现实时避障覆盖路径规划，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2507.10204v2](https://arxiv.org/pdf/2507.10204v2)**

> **作者:** Abdelhakim Amer; Mohit Mehindratta; Yury Brodskiy; Bilal Wehbe; Erdal Kayacan
>
> **备注:** Accepted for publication at International Conference on Robotics & Automation 2026
>
> **摘要:** Inspection of underwater structures with tethered underwater vehicles is often hindered by the risk of tether entanglement. We propose REACT (real-time entanglement-aware coverage path planning for tethered underwater vehicles), a framework designed to overcome this limitation. REACT comprises a computationally efficient geometry-based tether model using the signed distance field (SDF) map for accurate, real-time simulation of taut tether configurations around arbitrary structures in 3D. This model enables an efficient online replanning strategy by enforcing a maximum tether length constraint, thereby actively preventing entanglement. By integrating REACT into a coverage path planning framework, we achieve safe and entanglement-free inspection paths, previously challenging due to tether constraints. The complete REACT framework's efficacy is validated in a pipe inspection scenario, demonstrating safe navigation and full coverage inspection. Simulation results show that REACT achieves complete coverage while maintaining tether constraints and completing the total mission 20% faster than conventional planners, despite a longer inspection time due to proactive avoidance of entanglement that eliminates extensive post-mission disentanglement. Real-world experiments confirm these benefits, where REACT completes the full mission, while the baseline planner fails due to physical tether entanglement.
>
---
#### [replaced 008] HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出HyPlan，用于自动驾驶中的安全路径规划任务，解决部分可观测交通环境下的避撞问题。结合行为预测、深度强化学习和在线POMDP规划，提升安全性与效率。**

- **链接: [https://arxiv.org/pdf/2510.07210v2](https://arxiv.org/pdf/2510.07210v2)**

> **作者:** Donald Pfaffmann; Matthias Klusch; Marcel Steinmetz
>
> **摘要:** We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners.
>
---
#### [replaced 009] Sampling for Model Predictive Trajectory Planning in Autonomous Driving using Normalizing Flows
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自主驾驶轨迹规划任务，旨在提升采样效率。通过引入归一化流模型生成更优采样分布，优化轨迹生成过程。**

- **链接: [https://arxiv.org/pdf/2404.09657v4](https://arxiv.org/pdf/2404.09657v4)**

> **作者:** Georg Rabenstein; Lars Ullrich; Knut Graichen
>
> **备注:** Accepted to be published as part of the 2024 IEEE Intelligent Vehicles Symposium (IV), Jeju Shinhwa World, Jeju Island, Korea, June 2-5, 2024
>
> **摘要:** Alongside optimization-based planners, sampling-based approaches are often used in trajectory planning for autonomous driving due to their simplicity. Model predictive path integral control is a framework that builds upon optimization principles while incorporating stochastic sampling of input trajectories. This paper investigates several sampling approaches for trajectory generation. In this context, normalizing flows originating from the field of variational inference are considered for the generation of sampling distributions, as they model transformations of simple to more complex distributions. Accordingly, learning-based normalizing flow models are trained for a more efficient exploration of the input domain for the task at hand. The developed algorithm and the proposed sampling distributions are evaluated in two simulation scenarios.
>
---
#### [replaced 010] SPIDER: Scalable Physics-Informed Dexterous Retargeting
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SPIDER框架，解决机器人控制中数据稀缺与人类演示转换的问题。通过物理引导的重定向方法，将人类运动数据转化为动态可行的机器人轨迹，提升任务成功率并加速策略学习。**

- **链接: [https://arxiv.org/pdf/2511.09484v2](https://arxiv.org/pdf/2511.09484v2)**

> **作者:** Chaoyi Pan; Changhao Wang; Haozhi Qi; Zixi Liu; Homanga Bharadhwaj; Akash Sharma; Tingfan Wu; Guanya Shi; Jitendra Malik; Francois Hogan
>
> **备注:** Project website: https://jc-bao.github.io/spider-project/
>
> **摘要:** Learning dexterous and agile policy for humanoid and dexterous hand control requires large-scale demonstrations, but collecting robot-specific data is prohibitively expensive. In contrast, abundant human motion data is readily available from motion capture, videos, and virtual reality, which could help address the data scarcity problem. However, due to the embodiment gap and missing dynamic information like force and torque, these demonstrations cannot be directly executed on robots. To bridge this gap, we propose Scalable Physics-Informed DExterous Retargeting (SPIDER), a physics-based retargeting framework to transform and augment kinematic-only human demonstrations to dynamically feasible robot trajectories at scale. Our key insight is that human demonstrations should provide global task structure and objective, while large-scale physics-based sampling with curriculum-style virtual contact guidance should refine trajectories to ensure dynamical feasibility and correct contact sequences. SPIDER scales across diverse 9 humanoid/dexterous hand embodiments and 6 datasets, improving success rates by 18% compared to standard sampling, while being 10X faster than reinforcement learning (RL) baselines, and enabling the generation of a 2.4M frames dynamic-feasible robot dataset for policy learning. As a universal physics-based retargeting method, SPIDER can work with diverse quality data and generate diverse and high-quality data to enable efficient policy learning with methods like RL.
>
---
#### [replaced 011] DarkEQA: Benchmarking Vision-Language Models for Embodied Question Answering in Low-Light Indoor Environments
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉语言模型在低光环境下的问答任务，旨在解决现有基准未覆盖低光条件的问题。工作包括构建DarkEQA基准，模拟真实低光场景，评估模型性能。**

- **链接: [https://arxiv.org/pdf/2512.24985v3](https://arxiv.org/pdf/2512.24985v3)**

> **作者:** Yohan Park; Hyunwoo Ha; Wonjun Jo; Tae-Hyun Oh
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Vision Language Models (VLMs) are increasingly adopted as central reasoning modules for embodied agents. Existing benchmarks evaluate their capabilities under ideal, well-lit conditions, yet robust 24/7 operation demands performance under a wide range of visual degradations, including low-light conditions at night or in dark environments--a core necessity that has been largely overlooked. To address this underexplored challenge, we present DarkEQA, an open-source benchmark for evaluating EQA-relevant perceptual primitives under multi-level low-light conditions. DarkEQA isolates the perception bottleneck by evaluating question answering from egocentric observations under controlled degradations, enabling attributable robustness analysis. A key design feature of DarkEQA is its physical fidelity: visual degradations are modeled in linear RAW space, simulating physics-based illumination drop and sensor noise followed by an ISP-inspired rendering pipeline. We demonstrate the utility of DarkEQA by evaluating a wide range of state-of-the-art VLMs and Low-Light Image Enhancement (LLIE) models. Our analysis systematically reveals VLMs' limitations when operating under these challenging visual conditions. Project website: https://darkeqa-benchmark.github.io/
>
---
#### [replaced 012] Constrained Group Relative Policy Optimization
- **分类: cs.LG; cs.CL; cs.RO**

- **简介: 该论文提出Constrained GRPO，解决受限策略优化问题。针对GRPO在约束环境下的不足，通过拉格朗日方法引入约束，改进优势估计以稳定约束控制，提升任务成功率与约束满足度。**

- **链接: [https://arxiv.org/pdf/2602.05863v2](https://arxiv.org/pdf/2602.05863v2)**

> **作者:** Roger Girgis; Rodrigue de Schaetzen; Luke Rowe; Azalée Robitaille; Christopher Pal; Liam Paull
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** While Group Relative Policy Optimization (GRPO) has emerged as a scalable framework for critic-free policy learning, extending it to settings with explicit behavioral constraints remains underexplored. We introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. Constraints are specified via indicator cost functions, enabling direct optimization of violation rates through a Lagrangian relaxation. We show that a naive multi-component treatment in advantage estimation can break constrained learning: mismatched component-wise standard deviations distort the relative importance of the different objective terms, which in turn corrupts the Lagrangian signal and prevents meaningful constraint enforcement. We formally derive this effect to motivate our scalarized advantage construction that preserves the intended trade-off between reward and constraint terms. Experiments in a toy gridworld confirm the predicted optimization pathology and demonstrate that scalarizing advantages restores stable constraint control. In addition, we evaluate Constrained GRPO on robotics tasks, where it improves constraint satisfaction while increasing task success, establishing a simple and effective recipe for constrained policy optimization in embodied AI domains that increasingly rely on large multimodal foundation models.
>
---
#### [replaced 013] Less Is More: Scalable Visual Navigation from Limited Data
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决数据稀缺下的导航问题。通过结合几何规划器生成合成轨迹，提升基于模仿学习的导航性能。**

- **链接: [https://arxiv.org/pdf/2601.17815v2](https://arxiv.org/pdf/2601.17815v2)**

> **作者:** Yves Inglin; Jonas Frey; Changan Chen; Marco Hutter
>
> **备注:** v2: Minor text edits, reference formatting fixes, and project page link added
>
> **摘要:** Imitation learning provides a powerful framework for goal-conditioned visual navigation in mobile robots, enabling obstacle avoidance while respecting human preferences and social norms. However, its effectiveness depends critically on the quality and diversity of training data. In this work, we show how classical geometric planners can be leveraged to generate synthetic trajectories that complement costly human demonstrations. We train Less is More (LiMo), a transformer-based visual navigation policy that predicts goal-conditioned SE(2) trajectories from a single RGB observation, and find that augmenting limited expert demonstrations with planner-generated supervision yields substantial performance gains. Through ablations and complementary qualitative and quantitative analyses, we characterize how dataset scale and diversity affect planning performance. We demonstrate real-robot deployment and argue that robust visual navigation is enabled not by simply collecting more demonstrations, but by strategically curating diverse, high-quality datasets. Our results suggest that scalable, embodiment-specific geometric supervision is a practical path toward data-efficient visual navigation.
>
---
#### [replaced 014] Robust Meta-Learning of Vehicle Yaw Rate Dynamics via Conditional Neural Processes
- **分类: cs.RO**

- **简介: 该论文属于车辆动力学预测任务，旨在解决物理模型在 yaw rate 预测中的误差与复杂性问题，通过条件神经过程实现更准确、鲁棒的预测。**

- **链接: [https://arxiv.org/pdf/2407.06605v2](https://arxiv.org/pdf/2407.06605v2)**

> **作者:** Lars Ullrich; Andreas Völz; Knut Graichen
>
> **备注:** Published in 2023 62nd IEEE IEEE Conference on Decision and Control (CDC), Singapore, Singapore, December 13 - 15, 2023
>
> **摘要:** Trajectory planners of autonomous vehicles usually rely on physical models to predict the vehicle behavior. However, despite their suitability, physical models have some shortcomings. On the one hand, simple models suffer from larger model errors and more restrictive assumptions. On the other hand, complex models are computationally more demanding and depend on environmental and operational parameters. In each case, the drawbacks can be associated to a certain degree to the physical modeling of the yaw rate dynamics. Therefore, this paper investigates the yaw rate prediction based on conditional neural processes (CNP), a data-driven meta-learning approach, to simultaneously achieve low errors, adequate complexity and robustness to varying parameters. Thus, physical models can be enhanced in a targeted manner to provide accurate and computationally efficient predictions to enable safe planning in autonomous vehicles. High fidelity simulations for a variety of driving scenarios and different types of cars show that CNP makes it possible to employ and transfer knowledge about the yaw rate based on current driving dynamics in a human-like manner, yielding robustness against changing environmental and operational conditions.
>
---
