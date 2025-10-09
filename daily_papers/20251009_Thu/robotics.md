# 机器人 cs.RO

- **最新发布 36 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] Distributed 3D Source Seeking via SO(3) Geometric Control of Robot Swarms
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决三维空间中机器人集群寻找信号源的问题。作者提出了一种基于SO(3)流形的几何控制方法，避免了传统表示的奇异性问题，并设计了自适应控制器以实现快速对齐和稳定编队，通过仿真验证了方法有效性。**

- **链接: [http://arxiv.org/pdf/2510.06836v1](http://arxiv.org/pdf/2510.06836v1)**

> **作者:** Jesús Bautista; Héctor García de Marina
>
> **备注:** 7 pages, 3 figures. Submitted for presentation at the IFAC World Congress 2026
>
> **摘要:** This paper presents a geometric control framework on the Lie group SO(3) for 3D source-seeking by robots with first-order attitude dynamics and constant translational speed. By working directly on SO(3), the approach avoids Euler-angle singularities and quaternion ambiguities, providing a unique, intrinsic representation of orientation. We design a proportional feed-forward controller that ensures exponential alignment of each agent to an estimated ascending direction toward a 3D scalar field source. The controller adapts to bounded unknown variations and preserves well-posed swarm formations. Numerical simulations demonstrate the effectiveness of the method, with all code provided open source for reproducibility.
>
---
#### [new 002] A Formal gatekeeper Framework for Safe Dual Control with Active Exploration
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决模型不确定下安全轨迹规划问题。现有方法或忽视不确定性缩减，或缺乏安全保证。作者提出一种结合鲁棒规划与主动探索的框架，在确保安全前提下，仅在有益时进行探索，以减少不确定性并降低任务成本。通过扩展“gatekeeper”架构实现安全与信息性轨迹生成，并在四旋翼飞行器仿真中验证效果。**

- **链接: [http://arxiv.org/pdf/2510.06351v1](http://arxiv.org/pdf/2510.06351v1)**

> **作者:** Kaleb Ben Naveed; Devansh R. Agrawal; Dimitra Panagou
>
> **备注:** Submitted to American Control Conference (ACC) 2026
>
> **摘要:** Planning safe trajectories under model uncertainty is a fundamental challenge. Robust planning ensures safety by considering worst-case realizations, yet ignores uncertainty reduction and leads to overly conservative behavior. Actively reducing uncertainty on-the-fly during a nominal mission defines the dual control problem. Most approaches address this by adding a weighted exploration term to the cost, tuned to trade off the nominal objective and uncertainty reduction, but without formal consideration of when exploration is beneficial. Moreover, safety is enforced in some methods but not in others. We propose a framework that integrates robust planning with active exploration under formal guarantees as follows: The key innovation and contribution is that exploration is pursued only when it provides a verifiable improvement without compromising safety. To achieve this, we utilize our earlier work on gatekeeper as an architecture for safety verification, and extend it so that it generates both safe and informative trajectories that reduce uncertainty and the cost of the mission, or keep it within a user-defined budget. The methodology is evaluated via simulation case studies on the online dual control of a quadrotor under parametric uncertainty.
>
---
#### [new 003] Assist-As-Needed: Adaptive Multimodal Robotic Assistance for Medication Management in Dementia Care
- **分类: cs.RO**

- **简介: 该论文属于人机交互与医疗辅助机器人任务，旨在解决痴呆患者用药管理中辅助技术无法自适应的问题。作者设计了一个基于Pepper机器人的多模态自适应辅助框架，根据实时评估提供分级帮助。论文贡献包括：理论框架、系统实现与初步实验验证。**

- **链接: [http://arxiv.org/pdf/2510.06633v1](http://arxiv.org/pdf/2510.06633v1)**

> **作者:** Kruthika Gangaraju; Tanmayi Inaparthy; Jiaqi Yang; Yihao Zheng; Fengpei Yuan
>
> **摘要:** People living with dementia (PLWDs) face progressively declining abilities in medication management-from simple forgetfulness to complete task breakdown-yet most assistive technologies fail to adapt to these changing needs. This one-size-fits-all approach undermines autonomy, accelerates dependence, and increases caregiver burden. Occupational therapy principles emphasize matching assistance levels to individual capabilities: minimal reminders for those who merely forget, spatial guidance for those who misplace items, and comprehensive multimodal support for those requiring step-by-step instruction. However, existing robotic systems lack this adaptive, graduated response framework essential for maintaining PLWD independence. We present an adaptive multimodal robotic framework using the Pepper robot that dynamically adjusts assistance based on real-time assessment of user needs. Our system implements a hierarchical intervention model progressing from (1) simple verbal reminders, to (2) verbal + gestural cues, to (3) full multimodal guidance combining physical navigation to medication locations with step-by-step verbal and gestural instructions. Powered by LLM-driven interaction strategies and multimodal sensing, the system continuously evaluates task states to provide just-enough assistance-preserving autonomy while ensuring medication adherence. We conducted a preliminary study with healthy adults and dementia care stakeholders in a controlled lab setting, evaluating the system's usability, comprehensibility, and appropriateness of adaptive feedback mechanisms. This work contributes: (1) a theoretically grounded adaptive assistance framework translating occupational therapy principles into HRI design, (2) a multimodal robotic implementation that preserves PLWD dignity through graduated support, and (3) empirical insights into stakeholder perceptions of adaptive robotic care.
>
---
#### [new 004] A Narwhal-Inspired Sensing-to-Control Framework for Small Fixed-Wing Aircraft
- **分类: cs.RO**

- **简介: 该论文旨在提升小型固定翼无人机在低速时的敏捷性。通过仿生硬件设计（如仿独角鲸牙的探针）、物理引导的动力学学习及凸优化控制分配，构建端到端的感知-控制框架。利用多孔探针与机翼压力传感器测量气流，结合数据驱动校准与正则化方法，改善气流估计与控制性能，实现更平稳、精准的飞行控制。**

- **链接: [http://arxiv.org/pdf/2510.07160v1](http://arxiv.org/pdf/2510.07160v1)**

> **作者:** Fengze Xie; Xiaozhou Fan; Jacob Schuster; Yisong Yue; Morteza Gharib
>
> **摘要:** Fixed-wing unmanned aerial vehicles (UAVs) offer endurance and efficiency but lack low-speed agility due to highly coupled dynamics. We present an end-to-end sensing-to-control pipeline that combines bio-inspired hardware, physics-informed dynamics learning, and convex control allocation. Measuring airflow on a small airframe is difficult because near-body aerodynamics, propeller slipstream, control-surface actuation, and ambient gusts distort pressure signals. Inspired by the narwhal's protruding tusk, we mount in-house multi-hole probes far upstream and complement them with sparse, carefully placed wing pressure sensors for local flow measurement. A data-driven calibration maps probe pressures to airspeed and flow angles. We then learn a control-affine dynamics model using the estimated airspeed/angles and sparse sensors. A soft left/right symmetry regularizer improves identifiability under partial observability and limits confounding between wing pressures and flaperon inputs. Desired wrenches (forces and moments) are realized by a regularized least-squares allocator that yields smooth, trimmed actuation. Wind-tunnel studies across a wide operating range show that adding wing pressures reduces force-estimation error by 25-30%, the proposed model degrades less under distribution shift (about 12% versus 44% for an unstructured baseline), and force tracking improves with smoother inputs, including a 27% reduction in normal-force RMSE versus a plain affine model and 34% versus an unstructured baseline.
>
---
#### [new 005] TrackVLA++: Unleashing Reasoning and Memory Capabilities in VLA Models for Embodied Visual Tracking
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型在具身视觉跟踪（EVT）任务中的应用。解决现有方法在遮挡或干扰物下跟踪失败的问题，提出TrackVLA++模型，引入空间推理机制（Polar-CoT）和目标识别记忆模块（TIM），提升跟踪的时空连续性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.07134v1](http://arxiv.org/pdf/2510.07134v1)**

> **作者:** Jiahang Liu; Yunpeng Qi; Jiazhao Zhang; Minghan Li; Shaoan Wang; Kui Wu; Hanjing Ye; Hong Zhang; Zhibo Chen; Fangwei Zhong; Zhizheng Zhang; He Wang
>
> **备注:** Project page: https://pku-epic.github.io/TrackVLA-plus-plus-Web/
>
> **摘要:** Embodied Visual Tracking (EVT) is a fundamental ability that underpins practical applications, such as companion robots, guidance robots and service assistants, where continuously following moving targets is essential. Recent advances have enabled language-guided tracking in complex and unstructured scenes. However, existing approaches lack explicit spatial reasoning and effective temporal memory, causing failures under severe occlusions or in the presence of similar-looking distractors. To address these challenges, we present TrackVLA++, a novel Vision-Language-Action (VLA) model that enhances embodied visual tracking with two key modules, a spatial reasoning mechanism and a Target Identification Memory (TIM). The reasoning module introduces a Chain-of-Thought paradigm, termed Polar-CoT, which infers the target's relative position and encodes it as a compact polar-coordinate token for action prediction. Guided by these spatial priors, the TIM employs a gated update strategy to preserve long-horizon target memory, ensuring spatiotemporal consistency and mitigating target loss during extended occlusions. Extensive experiments show that TrackVLA++ achieves state-of-the-art performance on public benchmarks across both egocentric and multi-camera settings. On the challenging EVT-Bench DT split, TrackVLA++ surpasses the previous leading approach by 5.1 and 12, respectively. Furthermore, TrackVLA++ exhibits strong zero-shot generalization, enabling robust real-world tracking in dynamic and occluded scenarios.
>
---
#### [new 006] HyPlan: Hybrid Learning-Assisted Planning Under Uncertainty for Safe Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 论文提出HyPlan，一种混合学习辅助规划方法，用于自动驾驶在部分可观测交通环境中实现无碰撞导航。该研究属于自动驾驶决策规划任务，旨在解决复杂交通场景下的安全路径规划问题。方法融合多智能体行为预测、深度强化学习与在线POMDP规划，提升安全性与计算效率。**

- **链接: [http://arxiv.org/pdf/2510.07210v1](http://arxiv.org/pdf/2510.07210v1)**

> **作者:** Donald Pfaffmann; Matthias Klusch; Marcel Steinmetz
>
> **摘要:** We present a novel hybrid learning-assisted planning method, named HyPlan, for solving the collision-free navigation problem for self-driving cars in partially observable traffic environments. HyPlan combines methods for multi-agent behavior prediction, deep reinforcement learning with proximal policy optimization and approximated online POMDP planning with heuristic confidence-based vertical pruning to reduce its execution time without compromising safety of driving. Our experimental performance analysis on the CARLA-CTS2 benchmark of critical traffic scenarios with pedestrians revealed that HyPlan may navigate safer than selected relevant baselines and perform significantly faster than considered alternative online POMDP planners.
>
---
#### [new 007] A Digital Twin Framework for Metamorphic Testing of Autonomous Driving Systems Using Generative Model
- **分类: cs.RO; cs.AI**

- **简介: 论文提出了一种基于数字孪生和生成模型的自动驾驶系统**测试框架**，旨在解决**安全验证难、场景覆盖不全**的问题。通过构建虚拟环境，结合Stable Diffusion生成多样化驾驶场景，定义三种**变异关系**进行测试，提升了测试覆盖率和效果，在Udacity模拟器中验证有效。**

- **链接: [http://arxiv.org/pdf/2510.07133v1](http://arxiv.org/pdf/2510.07133v1)**

> **作者:** Tony Zhang; Burak Kantarci; Umair Siddique
>
> **摘要:** Ensuring the safety of self-driving cars remains a major challenge due to the complexity and unpredictability of real-world driving environments. Traditional testing methods face significant limitations, such as the oracle problem, which makes it difficult to determine whether a system's behavior is correct, and the inability to cover the full range of scenarios an autonomous vehicle may encounter. In this paper, we introduce a digital twin-driven metamorphic testing framework that addresses these challenges by creating a virtual replica of the self-driving system and its operating environment. By combining digital twin technology with AI-based image generative models such as Stable Diffusion, our approach enables the systematic generation of realistic and diverse driving scenes. This includes variations in weather, road topology, and environmental features, all while maintaining the core semantics of the original scenario. The digital twin provides a synchronized simulation environment where changes can be tested in a controlled and repeatable manner. Within this environment, we define three metamorphic relations inspired by real-world traffic rules and vehicle behavior. We validate our framework in the Udacity self-driving simulator and demonstrate that it significantly enhances test coverage and effectiveness. Our method achieves the highest true positive rate (0.719), F1 score (0.689), and precision (0.662) compared to baseline approaches. This paper highlights the value of integrating digital twins with AI-powered scenario generation to create a scalable, automated, and high-fidelity testing solution for autonomous vehicle safety.
>
---
#### [new 008] COMPAct: Computational Optimization and Automated Modular design of Planetary Actuators
- **分类: cs.RO**

- **简介: 该论文属于机械设计与自动化任务，旨在优化行星减速器参数并实现自动化建模。研究提出COMPAct框架，针对四种行星减速器类型进行参数优化，以最小化质量与宽度并最大化效率。同时生成可直接3D打印的CAD模型，并通过实验验证设计性能。**

- **链接: [http://arxiv.org/pdf/2510.07197v1](http://arxiv.org/pdf/2510.07197v1)**

> **作者:** Aman Singh; Deepak Kapa; Suryank Joshi; Shishir Kolathaya
>
> **备注:** 8 pages, 9 Figures, 2 tables, first two authors contributed equally
>
> **摘要:** The optimal design of robotic actuators is a critical area of research, yet limited attention has been given to optimizing gearbox parameters and automating actuator CAD. This paper introduces COMPAct: Computational Optimization and Automated Modular Design of Planetary Actuators, a framework that systematically identifies optimal gearbox parameters for a given motor across four gearbox types, single-stage planetary gearbox (SSPG), compound planetary gearbox (CPG), Wolfrom planetary gearbox (WPG), and double-stage planetary gearbox (DSPG). The framework minimizes mass and actuator width while maximizing efficiency, and further automates actuator CAD generation to enable direct 3D printing without manual redesign. Using this framework, optimal gearbox designs are explored over a wide range of gear ratios, providing insights into the suitability of different gearbox types across various gear ratio ranges. In addition, the framework is used to generate CAD models of all four gearbox types with varying gear ratios and motors. Two actuator types are fabricated and experimentally evaluated through power efficiency, no-load backlash, and transmission stiffness tests. Experimental results indicate that the SSPG actuator achieves a mechanical efficiency of 60-80 %, a no-load backlash of 0.59 deg, and a transmission stiffness of 242.7 Nm/rad, while the CPG actuator demonstrates 60 % efficiency, 2.6 deg backlash, and a stiffness of 201.6 Nm/rad. Code available at: https://anonymous.4open.science/r/COMPAct-SubNum-3408 Video: https://youtu.be/99zOKgxsDho
>
---
#### [new 009] Constrained Natural Language Action Planning for Resilient Embodied Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人任务规划领域，旨在解决大语言模型（LLM）在真实环境中因幻觉和不可重复性导致的可靠性问题。论文提出一种结合LLM与符号规划的新方法，增强系统的可靠性、可重复性和透明度，同时保留LLM的灵活性与泛化能力。实验表明其在模拟和真实世界任务中表现优异。**

- **链接: [http://arxiv.org/pdf/2510.06357v1](http://arxiv.org/pdf/2510.06357v1)**

> **作者:** Grayson Byrd; Corban Rivera; Bethany Kemp; Meghan Booker; Aurora Schmidt; Celso M de Melo; Lalithkumar Seenivasan; Mathias Unberath
>
> **摘要:** Replicating human-level intelligence in the execution of embodied tasks remains challenging due to the unconstrained nature of real-world environments. Novel use of large language models (LLMs) for task planning seeks to address the previously intractable state/action space of complex planning tasks, but hallucinations limit their reliability, and thus, viability beyond a research context. Additionally, the prompt engineering required to achieve adequate system performance lacks transparency, and thus, repeatability. In contrast to LLM planning, symbolic planning methods offer strong reliability and repeatability guarantees, but struggle to scale to the complexity and ambiguity of real-world tasks. We introduce a new robotic planning method that augments LLM planners with symbolic planning oversight to improve reliability and repeatability, and provide a transparent approach to defining hard constraints with considerably stronger clarity than traditional prompt engineering. Importantly, these augmentations preserve the reasoning capabilities of LLMs and retain impressive generalization in open-world environments. We demonstrate our approach in simulated and real-world environments. On the ALFWorld planning benchmark, our approach outperforms current state-of-the-art methods, achieving a near-perfect 99% success rate. Deployment of our method to a real-world quadruped robot resulted in 100% task success compared to 50% and 30% for pure LLM and symbolic planners across embodied pick and place tasks. Our approach presents an effective strategy to enhance the reliability, repeatability and transparency of LLM-based robot planners while retaining their key strengths: flexibility and generalizability to complex real-world environments. We hope that this work will contribute to the broad goal of building resilient embodied intelligent systems.
>
---
#### [new 010] Vi-TacMan: Articulated Object Manipulation via Vision and Touch
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决无明确运动学模型的关节点物体自主操作问题。通过结合视觉提供全局引导与触觉实现局部精准控制，提出Vi-TacMan框架，利用视觉生成抓取和方向建议，并通过触觉反馈实时调整，提升操作精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06339v1](http://arxiv.org/pdf/2510.06339v1)**

> **作者:** Leiyao Cui; Zihang Zhao; Sirui Xie; Wenhuan Zhang; Zhi Han; Yixin Zhu
>
> **摘要:** Autonomous manipulation of articulated objects remains a fundamental challenge for robots in human environments. Vision-based methods can infer hidden kinematics but can yield imprecise estimates on unfamiliar objects. Tactile approaches achieve robust control through contact feedback but require accurate initialization. This suggests a natural synergy: vision for global guidance, touch for local precision. Yet no framework systematically exploits this complementarity for generalized articulated manipulation. Here we present Vi-TacMan, which uses vision to propose grasps and coarse directions that seed a tactile controller for precise execution. By incorporating surface normals as geometric priors and modeling directions via von Mises-Fisher distributions, our approach achieves significant gains over baselines (all p<0.0001). Critically, manipulation succeeds without explicit kinematic models -- the tactile controller refines coarse visual estimates through real-time contact regulation. Tests on more than 50,000 simulated and diverse real-world objects confirm robust cross-category generalization. This work establishes that coarse visual cues suffice for reliable manipulation when coupled with tactile feedback, offering a scalable paradigm for autonomous systems in unstructured environments.
>
---
#### [new 011] Safe Obstacle-Free Guidance of Space Manipulators in Debris Removal Missions via Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于空间机械臂控制任务，旨在解决非合作目标碎片清除中的安全轨迹规划问题。作者采用深度强化学习方法，设计多评论家网络与经验回放缓冲，实现机械臂避障、防碰撞与稳定追踪目标。**

- **链接: [http://arxiv.org/pdf/2510.06566v1](http://arxiv.org/pdf/2510.06566v1)**

> **作者:** Vincent Lam; Robin Chhabra
>
> **摘要:** The objective of this study is to develop a model-free workspace trajectory planner for space manipulators using a Twin Delayed Deep Deterministic Policy Gradient (TD3) agent to enable safe and reliable debris capture. A local control strategy with singularity avoidance and manipulability enhancement is employed to ensure stable execution. The manipulator must simultaneously track a capture point on a non-cooperative target, avoid self-collisions, and prevent unintended contact with the target. To address these challenges, we propose a curriculum-based multi-critic network where one critic emphasizes accurate tracking and the other enforces collision avoidance. A prioritized experience replay buffer is also used to accelerate convergence and improve policy robustness. The framework is evaluated on a simulated seven-degree-of-freedom KUKA LBR iiwa mounted on a free-floating base in Matlab/Simulink, demonstrating safe and adaptive trajectory generation for debris removal missions.
>
---
#### [new 012] TIGeR: Tool-Integrated Geometric Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-语言模型在机器人领域的应用任务，旨在解决现有模型在几何推理中精度不足的问题。论文提出了TIGeR框架，通过集成外部工具实现精确几何计算，提升了机器人操作的厘米级精度。**

- **链接: [http://arxiv.org/pdf/2510.07181v1](http://arxiv.org/pdf/2510.07181v1)**

> **作者:** Yi Han; Cheng Chi; Enshen Zhou; Shanyu Rong; Jingkun An; Pengwei Wang; Zhongyuan Wang; Lu Sheng; Shanghang Zhang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Vision-Language Models (VLMs) have shown remarkable capabilities in spatial reasoning, yet they remain fundamentally limited to qualitative precision and lack the computational precision required for real-world robotics. Current approaches fail to leverage metric cues from depth sensors and camera calibration, instead reducing geometric problems to pattern recognition tasks that cannot deliver the centimeter-level accuracy essential for robotic manipulation. We present TIGeR (Tool-Integrated Geometric Reasoning), a novel framework that transforms VLMs from perceptual estimators to geometric computers by enabling them to generate and execute precise geometric computations through external tools. Rather than attempting to internalize complex geometric operations within neural networks, TIGeR empowers models to recognize geometric reasoning requirements, synthesize appropriate computational code, and invoke specialized libraries for exact calculations. To support this paradigm, we introduce TIGeR-300K, a comprehensive tool-invocation-oriented dataset covering point transformations, pose estimation, trajectory generation, and spatial compatibility verification, complete with tool invocation sequences and intermediate computations. Through a two-stage training pipeline combining supervised fine-tuning (SFT) and reinforcement fine-tuning (RFT) with our proposed hierarchical reward design, TIGeR achieves SOTA performance on geometric reasoning benchmarks while demonstrating centimeter-level precision in real-world robotic manipulation tasks.
>
---
#### [new 013] DPL: Depth-only Perceptive Humanoid Locomotion via Realistic Depth Synthesis and Cross-Attention Terrain Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于人形机器人地形感知与运动控制任务，旨在解决复杂地形下机器人运动的感知效率与适应性问题。论文提出DPL框架，结合盲策略、跨模态注意力模型与真实深度合成方法，提升地形重建与运动控制性能。**

- **链接: [http://arxiv.org/pdf/2510.07152v1](http://arxiv.org/pdf/2510.07152v1)**

> **作者:** Jingkai Sun; Gang Han; Pihai Sun; Wen Zhao; Jiahang Cao; Jiaxu Wang; Yijie Guo; Qiang Zhang
>
> **摘要:** Recent advancements in legged robot perceptive locomotion have shown promising progress. However, terrain-aware humanoid locomotion remains largely constrained to two paradigms: depth image-based end-to-end learning and elevation map-based methods. The former suffers from limited training efficiency and a significant sim-to-real gap in depth perception, while the latter depends heavily on multiple vision sensors and localization systems, resulting in latency and reduced robustness. To overcome these challenges, we propose a novel framework that tightly integrates three key components: (1) Terrain-Aware Locomotion Policy with a Blind Backbone, which leverages pre-trained elevation map-based perception to guide reinforcement learning with minimal visual input; (2) Multi-Modality Cross-Attention Transformer, which reconstructs structured terrain representations from noisy depth images; (3) Realistic Depth Images Synthetic Method, which employs self-occlusion-aware ray casting and noise-aware modeling to synthesize realistic depth observations, achieving over 30\% reduction in terrain reconstruction error. This combination enables efficient policy training with limited data and hardware resources, while preserving critical terrain features essential for generalization. We validate our framework on a full-sized humanoid robot, demonstrating agile and adaptive locomotion across diverse and challenging terrains.
>
---
#### [new 014] UniFField: A Generalizable Unified Neural Feature Field for Visual, Semantic, and Spatial Uncertainties in Any Scene
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人感知任务，旨在解决复杂环境中3D场景理解与不确定性建模问题。作者提出UniFField，一种统一的神经特征场，融合视觉、语义与几何特征，并预测各模态的不确定性，支持零样本迁移与增量式场景建模，提升机器人决策鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.06754v1](http://arxiv.org/pdf/2510.06754v1)**

> **作者:** Christian Maurer; Snehal Jauhri; Sophie Lueth; Georgia Chalvatzaki
>
> **备注:** Project website: https://sites.google.com/view/uniffield
>
> **摘要:** Comprehensive visual, geometric, and semantic understanding of a 3D scene is crucial for successful execution of robotic tasks, especially in unstructured and complex environments. Additionally, to make robust decisions, it is necessary for the robot to evaluate the reliability of perceived information. While recent advances in 3D neural feature fields have enabled robots to leverage features from pretrained foundation models for tasks such as language-guided manipulation and navigation, existing methods suffer from two critical limitations: (i) they are typically scene-specific, and (ii) they lack the ability to model uncertainty in their predictions. We present UniFField, a unified uncertainty-aware neural feature field that combines visual, semantic, and geometric features in a single generalizable representation while also predicting uncertainty in each modality. Our approach, which can be applied zero shot to any new environment, incrementally integrates RGB-D images into our voxel-based feature representation as the robot explores the scene, simultaneously updating uncertainty estimation. We evaluate our uncertainty estimations to accurately describe the model prediction errors in scene reconstruction and semantic feature prediction. Furthermore, we successfully leverage our feature predictions and their respective uncertainty for an active object search task using a mobile manipulator robot, demonstrating the capability for robust decision-making.
>
---
#### [new 015] Real-Time Glass Detection and Reprojection using Sensor Fusion Onboard Aerial Robots
- **分类: cs.RO; cs.CV; cs.SY; eess.SY**

- **简介: 论文任务为透明障碍物实时检测与地图构建。针对无人机在透明物体前导航困难的问题，提出融合ToF相机与超声波传感器的轻量级方案，实现低功耗无人机上实时透明障碍物感知。**

- **链接: [http://arxiv.org/pdf/2510.06518v1](http://arxiv.org/pdf/2510.06518v1)**

> **作者:** Malakhi Hopkins; Varun Murali; Vijay Kumar; Camillo J Taylor
>
> **备注:** 8 pages, 8 figures, submitted to ICRA 2026
>
> **摘要:** Autonomous aerial robots are increasingly being deployed in real-world scenarios, where transparent obstacles present significant challenges to reliable navigation and mapping. These materials pose a unique problem for traditional perception systems because they lack discernible features and can cause conventional depth sensors to fail, leading to inaccurate maps and potential collisions. To ensure safe navigation, robots must be able to accurately detect and map these transparent obstacles. Existing methods often rely on large, expensive sensors or algorithms that impose high computational burdens, making them unsuitable for low Size, Weight, and Power (SWaP) robots. In this work, we propose a novel and computationally efficient framework for detecting and mapping transparent obstacles onboard a sub-300g quadrotor. Our method fuses data from a Time-of-Flight (ToF) camera and an ultrasonic sensor with a custom, lightweight 2D convolution model. This specialized approach accurately detects specular reflections and propagates their depth into corresponding empty regions of the depth map, effectively rendering transparent obstacles visible. The entire pipeline operates in real-time, utilizing only a small fraction of a CPU core on an embedded processor. We validate our system through a series of experiments in both controlled and real-world environments, demonstrating the utility of our method through experiments where the robot maps indoor environments containing glass. Our work is, to our knowledge, the first of its kind to demonstrate a real-time, onboard transparent obstacle mapping system on a low-SWaP quadrotor using only the CPU.
>
---
#### [new 016] Vision-Language-Action Models for Robotics: A Review Towards Real-World Applications
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人领域的综述任务，旨在解决如何将视觉、语言和动作统一应用于机器人系统以提升泛化能力的问题。论文系统回顾了VLA模型的架构、学习方法、数据策略及硬件平台，提供实际部署指导，并整理了相关资源，助力真实场景应用。**

- **链接: [http://arxiv.org/pdf/2510.07077v1](http://arxiv.org/pdf/2510.07077v1)**

> **作者:** Kento Kawaharazuka; Jihoon Oh; Jun Yamada; Ingmar Posner; Yuke Zhu
>
> **备注:** Accepted to IEEE Access, website: https://vla-survey.github.io
>
> **摘要:** Amid growing efforts to leverage advances in large language models (LLMs) and vision-language models (VLMs) for robotics, Vision-Language-Action (VLA) models have recently gained significant attention. By unifying vision, language, and action data at scale, which have traditionally been studied separately, VLA models aim to learn policies that generalise across diverse tasks, objects, embodiments, and environments. This generalisation capability is expected to enable robots to solve novel downstream tasks with minimal or no additional task-specific data, facilitating more flexible and scalable real-world deployment. Unlike previous surveys that focus narrowly on action representations or high-level model architectures, this work offers a comprehensive, full-stack review, integrating both software and hardware components of VLA systems. In particular, this paper provides a systematic review of VLAs, covering their strategy and architectural transition, architectures and building blocks, modality-specific processing techniques, and learning paradigms. In addition, to support the deployment of VLAs in real-world robotic applications, we also review commonly used robot platforms, data collection strategies, publicly available datasets, data augmentation methods, and evaluation benchmarks. Throughout this comprehensive survey, this paper aims to offer practical guidance for the robotics community in applying VLAs to real-world robotic systems. All references categorized by training approach, evaluation method, modality, and dataset are available in the table on our project website: https://vla-survey.github.io .
>
---
#### [new 017] Tailoring materials into kirigami robots
- **分类: cs.RO; cond-mat.soft**

- **简介: 论文探讨将剪纸艺术应用于机器人，设计 kirigami 结构以实现轻量化、多功能的机器人组件。任务是解决如何通过优化剪切图案提升机器人性能，如驱动、传感与储能。工作包括开发 kirigami 执行器、传感器及电池，并探索其在抓取、运动与可穿戴设备中的应用。**

- **链接: [http://arxiv.org/pdf/2510.07027v1](http://arxiv.org/pdf/2510.07027v1)**

> **作者:** Saravana Prashanth Murali Babu; Aida Parvaresh; Ahmad Rafsanjani
>
> **摘要:** Kirigami, the traditional paper-cutting craft, holds immense potential for revolutionizing robotics by providing multifunctional, lightweight, and adaptable solutions. Kirigami structures, characterized by their bending-dominated deformation, offer resilience to tensile forces and facilitate shape morphing under small actuation forces. Kirigami components such as actuators, sensors, batteries, controllers, and body structures can be tailored to specific robotic applications by optimizing cut patterns. Actuators based on kirigami principles exhibit complex motions programmable through various energy sources, while kirigami sensors bridge the gap between electrical conductivity and compliance. Kirigami-integrated batteries enable energy storage directly within robot structures, enhancing flexibility and compactness. Kirigami-controlled mechanisms mimic mechanical computations, enabling advanced functionalities such as shape morphing and memory functions. Applications of kirigami-enabled robots include grasping, locomotion, and wearables, showcasing their adaptability to diverse environments and tasks. Despite promising opportunities, challenges remain in the design of cut patterns for a given function and streamlining fabrication techniques.
>
---
#### [new 018] Sampling Strategies for Robust Universal Quadrupedal Locomotion Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人控制策略研究任务，旨在解决四足机器人运动策略在仿真与现实间泛化能力不足的问题。通过对比不同关节增益采样策略，探索如何提升策略鲁棒性，实现跨配置与跨平台的零样本迁移。**

- **链接: [http://arxiv.org/pdf/2510.07094v1](http://arxiv.org/pdf/2510.07094v1)**

> **作者:** David Rytz; Kim Tien Ly; Ioannis Havoutis
>
> **摘要:** This work focuses on sampling strategies of configuration variations for generating robust universal locomotion policies for quadrupedal robots. We investigate the effects of sampling physical robot parameters and joint proportional-derivative gains to enable training a single reinforcement learning policy that generalizes to multiple parameter configurations. Three fundamental joint gain sampling strategies are compared: parameter sampling with (1) linear and polynomial function mappings of mass-to-gains, (2) performance-based adaptive filtering, and (3) uniform random sampling. We improve the robustness of the policy by biasing the configurations using nominal priors and reference models. All training was conducted on RaiSim, tested in simulation on a range of diverse quadrupeds, and zero-shot deployed onto hardware using the ANYmal quadruped robot. Compared to multiple baseline implementations, our results demonstrate the need for significant joint controller gains randomization for robust closing of the sim-to-real gap.
>
---
#### [new 019] Bring the Apple, Not the Sofa: Impact of Irrelevant Context in Embodied AI Commands on VLA Models
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言-动作（VLA）模型在具身AI任务中对指令语言变化的鲁棒性。主要问题是如何应对语言指令中的无关上下文和改写带来的性能下降。作者系统评估了两种语言干扰类型的影响，并提出一种基于大语言模型的过滤框架，以提取核心指令，显著恢复模型性能。**

- **链接: [http://arxiv.org/pdf/2510.07067v1](http://arxiv.org/pdf/2510.07067v1)**

> **作者:** Daria Pugacheva; Andrey Moskalenko; Denis Shepelev; Andrey Kuznetsov; Vlad Shakhuro; Elena Tutubalina
>
> **摘要:** Vision Language Action (VLA) models are widely used in Embodied AI, enabling robots to interpret and execute language instructions. However, their robustness to natural language variability in real-world scenarios has not been thoroughly investigated. In this work, we present a novel systematic study of the robustness of state-of-the-art VLA models under linguistic perturbations. Specifically, we evaluate model performance under two types of instruction noise: (1) human-generated paraphrasing and (2) the addition of irrelevant context. We further categorize irrelevant contexts into two groups according to their length and their semantic and lexical proximity to robot commands. In this study, we observe consistent performance degradation as context size expands. We also demonstrate that the model can exhibit relative robustness to random context, with a performance drop within 10%, while semantically and lexically similar context of the same length can trigger a quality decline of around 50%. Human paraphrases of instructions lead to a drop of nearly 20%. To mitigate this, we propose an LLM-based filtering framework that extracts core commands from noisy inputs. Incorporating our filtering step allows models to recover up to 98.5% of their original performance under noisy conditions.
>
---
#### [new 020] What You Don't Know Can Hurt You: How Well do Latent Safety Filters Understand Partially Observable Safety Constraints?
- **分类: cs.RO**

- **简介: 该论文研究机器人安全控制，探讨在部分可观测条件下，如何通过学习潜在状态空间实现安全策略。现有方法假设潜在状态包含所有安全信息，但作者发现仅依赖视觉可能引发短视安全行为。为此，作者提出信息度量方法预测潜在状态不足，并设计多模态训练策略提升安全性，最终在模拟和真实机械臂任务中验证效果。**

- **链接: [http://arxiv.org/pdf/2510.06492v1](http://arxiv.org/pdf/2510.06492v1)**

> **作者:** Matthew Kim; Kensuke Nakamura; Andrea Bajcsy
>
> **备注:** 8 tables 6 figures
>
> **摘要:** Safe control techniques, such as Hamilton-Jacobi reachability, provide principled methods for synthesizing safety-preserving robot policies but typically assume hand-designed state spaces and full observability. Recent work has relaxed these assumptions via latent-space safe control, where state representations and dynamics are learned jointly through world models that reconstruct future high-dimensional observations (e.g., RGB images) from current observations and actions. This enables safety constraints that are difficult to specify analytically (e.g., spilling) to be framed as classification problems in latent space, allowing controllers to operate directly from raw observations. However, these methods assume that safety-critical features are observable in the learned latent state. We ask: when are latent state spaces sufficient for safe control? To study this, we examine temperature-based failures, comparable to overheating in cooking or manufacturing tasks, and find that RGB-only observations can produce myopic safety behaviors, e.g., avoiding seeing failure states rather than preventing failure itself. To predict such behaviors, we introduce a mutual information-based measure that identifies when observations fail to capture safety-relevant features. Finally, we propose a multimodal-supervised training strategy that shapes the latent state with additional sensory inputs during training, but requires no extra modalities at deployment, and validate our approach in simulation and on hardware with a Franka Research 3 manipulator preventing a pot of wax from overheating.
>
---
#### [new 021] Active Next-Best-View Optimization for Risk-Averse Path Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人路径规划任务，旨在解决不确定环境中安全导航问题。通过结合风险规避与主动感知，提出一种统一框架：利用3D高斯辐射场构建风险地图，并在SE(3)流形上优化下一最佳视角，以提升路径安全性与感知效率。**

- **链接: [http://arxiv.org/pdf/2510.06481v1](http://arxiv.org/pdf/2510.06481v1)**

> **作者:** Amirhossein Mollaei Khass; Guangyi Liu; Vivek Pandey; Wen Jiang; Boshu Lei; Kostas Daniilidis; Nader Motee
>
> **摘要:** Safe navigation in uncertain environments requires planning methods that integrate risk aversion with active perception. In this work, we present a unified framework that refines a coarse reference path by constructing tail-sensitive risk maps from Average Value-at-Risk statistics on an online-updated 3D Gaussian-splat Radiance Field. These maps enable the generation of locally safe and feasible trajectories. In parallel, we formulate Next-Best-View (NBV) selection as an optimization problem on the SE(3) pose manifold, where Riemannian gradient descent maximizes an expected information gain objective to reduce uncertainty most critical for imminent motion. Our approach advances the state-of-the-art by coupling risk-averse path refinement with NBV planning, while introducing scalable gradient decompositions that support efficient online updates in complex environments. We demonstrate the effectiveness of the proposed framework through extensive computational studies.
>
---
#### [new 022] RAISE: A self-driving laboratory for interfacial property formulation discovery
- **分类: cs.RO**

- **简介: 该论文设计了一个名为RAISE的自动驾驶实验室系统，用于自主探索液体配方与表面润湿性之间的关系。该系统通过贝叶斯优化实现多目标配方优化，以达到指定接触角、减少表面活性剂使用和成本的目标。论文属于自动化实验与优化任务，旨在解决液体配方优化与表面润湿性评估的高效结合问题。**

- **链接: [http://arxiv.org/pdf/2510.06546v1](http://arxiv.org/pdf/2510.06546v1)**

> **作者:** Mohammad Nazeri; Sheldon Mei; Jeffrey Watchorn; Alex Zhang; Erin Ng; Tao Wen; Abhijoy Mandal; Kevin Golovin; Alan Aspuru-Guzik; Frank Gu
>
> **备注:** Mohammad Nazeri, Sheldon Mei, and Jeffrey Watchorn contributed equally to this work. *Corresponding author: Frank Gu (f.gu@utoronto.ca)
>
> **摘要:** Surface wettability is a critical design parameter for biomedical devices, coatings, and textiles. Contact angle measurements quantify liquid-surface interactions, which depend strongly on liquid formulation. Herein, we present the Robotic Autonomous Imaging Surface Evaluator (RAISE), a closed-loop, self-driving laboratory that is capable of linking liquid formulation optimization with surface wettability assessment. RAISE comprises a full experimental orchestrator with the ability of mixing liquid ingredients to create varying formulation cocktails, transferring droplets of prepared formulations to a high-throughput stage, and using a pick-and-place camera tool for automated droplet image capture. The system also includes an automated image processing pipeline to measure contact angles. This closed loop experiment orchestrator is integrated with a Bayesian Optimization (BO) client, which enables iterative exploration of new formulations based on previous contact angle measurements to meet user-defined objectives. The system operates in a high-throughput manner and can achieve a measurement rate of approximately 1 contact angle measurement per minute. Here we demonstrate RAISE can be used to explore surfactant wettability and how surfactant combinations create tunable formulations that compensate for purity-related variations. Furthermore, multi-objective BO demonstrates how precise and optimal formulations can be reached based on application-specific goals. The optimization is guided by a desirability score, which prioritizes formulations that are within target contact angle ranges, minimize surfactant usage and reduce cost. This work demonstrates the capabilities of RAISE to autonomously link liquid formulations to contact angle measurements in a closed-loop system, using multi-objective BO to efficiently identify optimal formulations aligned with researcher-defined criteria.
>
---
#### [new 023] RLinf-VLA: A Unified and Efficient Framework for VLA+RL Training
- **分类: cs.RO**

- **简介: 论文提出RLinf-VLA，旨在解决视觉-语言-动作（VLA）模型在具身智能中的训练问题。现有方法依赖监督微调，泛化能力差，而强化学习（RL）缺乏统一平台。RLinf-VLA统一支持多种模型、算法和仿真器，提升训练效率，并在模拟和真实机器人任务中表现出强泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06710v1](http://arxiv.org/pdf/2510.06710v1)**

> **作者:** Hongzhi Zang; Mingjie Wei; Si Xu; Yongji Wu; Zhen Guo; Yuanqing Wang; Hao Lin; Liangzhi Shi; Yuqing Xie; Zhexuan Xu; Zhihao Liu; Kang Chen; Wenhao Tang; Quanlu Zhang; Weinan Zhang; Chao Yu; Yu Wang
>
> **备注:** This is the technical report of the RLinf Team, focusing on the algorithm side. For the system-level design, please refer to arXiv:2509.15965. The open-sourced code link: https://github.com/RLinf/RLinf
>
> **摘要:** Recent progress in vision and language foundation models has significantly advanced multimodal understanding, reasoning, and generation, inspiring a surge of interest in extending such capabilities to embodied settings through vision-language-action (VLA) models. Yet, most VLA models are still trained with supervised fine-tuning (SFT), which struggles to generalize under distribution shifts due to error accumulation. Reinforcement learning (RL) offers a promising alternative by directly optimizing task performance through interaction, but existing attempts remain fragmented and lack a unified platform for fair and systematic comparison across model architectures and algorithmic designs. To address this gap, we introduce RLinf-VLA, a unified and efficient framework for scalable RL training of VLA models. The system adopts a highly flexible resource allocation design that addresses the challenge of integrating rendering, training, and inference in RL+VLA training. In particular, for GPU-parallelized simulators, RLinf-VLA implements a novel hybrid fine-grained pipeline allocation mode, achieving a 1.61x-1.88x speedup in training. Through a unified interface, RLinf-VLA seamlessly supports diverse VLA architectures (e.g., OpenVLA, OpenVLA-OFT), multiple RL algorithms (e.g., PPO, GRPO), and various simulators (e.g., ManiSkill, LIBERO). In simulation, a unified model achieves 98.11\% across 130 LIBERO tasks and 97.66\% across 25 ManiSkill tasks. Beyond empirical performance, our study distills a set of best practices for applying RL to VLA training and sheds light on emerging patterns in this integration. Furthermore, we present preliminary deployment on a real-world Franka robot, where RL-trained policies exhibit stronger generalization than those trained with SFT. We envision RLinf-VLA as a foundation to accelerate and standardize research on embodied intelligence.
>
---
#### [new 024] Diffusing Trajectory Optimization Problems for Recovery During Multi-Finger Manipulation
- **分类: cs.RO**

- **简介: 该论文属于多指灵巧操作任务，旨在解决操作中因环境扰动或执行错误导致的任务中断问题。作者利用扩散模型构建框架，自动识别需恢复的状态，并优化接触丰富的恢复轨迹。提出的新方法能高效扩散轨迹优化问题的全参数化，显著提升在线执行效率。实验表明其恢复方法使任务性能提高96%，优于强化学习等基线方法。**

- **链接: [http://arxiv.org/pdf/2510.07030v1](http://arxiv.org/pdf/2510.07030v1)**

> **作者:** Abhinav Kumar; Fan Yang; Sergio Aguilera Marinovic; Soshi Iba; Rana Soltani Zarrin; Dmitry Berenson
>
> **摘要:** Multi-fingered hands are emerging as powerful platforms for performing fine manipulation tasks, including tool use. However, environmental perturbations or execution errors can impede task performance, motivating the use of recovery behaviors that enable normal task execution to resume. In this work, we take advantage of recent advances in diffusion models to construct a framework that autonomously identifies when recovery is necessary and optimizes contact-rich trajectories to recover. We use a diffusion model trained on the task to estimate when states are not conducive to task execution, framed as an out-of-distribution detection problem. We then use diffusion sampling to project these states in-distribution and use trajectory optimization to plan contact-rich recovery trajectories. We also propose a novel diffusion-based approach that distills this process to efficiently diffuse the full parameterization, including constraints, goal state, and initialization, of the recovery trajectory optimization problem, saving time during online execution. We compare our method to a reinforcement learning baseline and other methods that do not explicitly plan contact interactions, including on a hardware screwdriver-turning task where we show that recovering using our method improves task performance by 96% and that ours is the only method evaluated that can attempt recovery without causing catastrophic task failure. Videos can be found at https://dtourrecovery.github.io/.
>
---
#### [new 025] SanDRA: Safe Large-Language-Model-Based Decision Making for Automated Vehicles Using Reachability Analysis
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶决策任务，旨在解决大语言模型在自动驾驶中决策安全性不足的问题。通过结合可达性分析与形式化交通规则，提出SanDRA框架，过滤不安全驾驶行为，确保决策安全且合规。**

- **链接: [http://arxiv.org/pdf/2510.06717v1](http://arxiv.org/pdf/2510.06717v1)**

> **作者:** Yuanfei Lin; Sebastian Illing; Matthias Althoff
>
> **备注:** @2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Large language models have been widely applied to knowledge-driven decision-making for automated vehicles due to their strong generalization and reasoning capabilities. However, the safety of the resulting decisions cannot be ensured due to possible hallucinations and the lack of integrated vehicle dynamics. To address this issue, we propose SanDRA, the first safe large-language-model-based decision making framework for automated vehicles using reachability analysis. Our approach starts with a comprehensive description of the driving scenario to prompt large language models to generate and rank feasible driving actions. These actions are translated into temporal logic formulas that incorporate formalized traffic rules, and are subsequently integrated into reachability analysis to eliminate unsafe actions. We validate our approach in both open-loop and closed-loop driving environments using off-the-shelf and finetuned large language models, showing that it can provide provably safe and, where possible, legally compliant driving actions, even under high-density traffic conditions. To ensure transparency and facilitate future research, all code and experimental setups are publicly available at github.com/CommonRoad/SanDRA.
>
---
#### [new 026] Temporal-Prior-Guided View Planning for Periodic 3D Plant Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于三维重建任务，旨在解决周期性植物重建中重复采集导致资源浪费的问题。作者提出一种基于时间先验的视角规划方法，利用先前模型与新观测对齐，生成当前几何近似并优化视角选择，减少采集次数且保持覆盖效果。**

- **链接: [http://arxiv.org/pdf/2510.07028v1](http://arxiv.org/pdf/2510.07028v1)**

> **作者:** Sicong Pan; Xuying Huang; Maren Bennewitz
>
> **备注:** Accepted to the Active Perception Workshop at IROS 2025
>
> **摘要:** Periodic 3D reconstruction is essential for crop monitoring, but costly when each cycle restarts from scratch, wasting resources and ignoring information from previous captures. We propose temporal-prior-guided view planning for periodic plant reconstruction, in which a previously reconstructed model of the same plant is non-rigidly aligned to a new partial observation to form an approximation of the current geometry. To accommodate plant growth, we inflate this approximation and solve a set covering optimization problem to compute a minimal set of views. We integrated this method into a complete pipeline that acquires one additional next-best view before registration for robustness and then plans a globally shortest path to connect the planned set of views and outputs the best view sequence. Experiments on maize and tomato under hemisphere and sphere view spaces show that our system maintains or improves surface coverage while requiring fewer views and comparable movement cost compared to state-of-the-art baselines.
>
---
#### [new 027] Three-dimensional Integrated Guidance and Control for Leader-Follower Flexible Formation of Fixed Wing UAVs
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; math.DS**

- **简介: 该论文属于无人机编队飞行控制任务，旨在解决固定翼无人机在复杂动态环境下灵活编队问题。论文提出了一种三维非线性一体化制导与控制方法，使跟随者无人机能在领导者后方半球区域内灵活调整位置，适应领导者剧烈机动，并通过动态面控制和李雅普诺夫屏障函数确保系统稳定与约束满足。**

- **链接: [http://arxiv.org/pdf/2510.06394v1](http://arxiv.org/pdf/2510.06394v1)**

> **作者:** Praveen Kumar Ranjan; Abhinav Sinha; Yongcan Cao
>
> **摘要:** This paper presents a nonlinear integrated guidance and control (IGC) approach for flexible leader-follower formation flight of fixed-wing unmanned aerial vehicles (UAVs) while accounting for high-fidelity aerodynamics and thrust dynamics. Unlike conventional leader-follower schemes that fix the follower's position relative to the leader, the follower is steered to maintain range and bearing angles (which is the angle between its velocity vector and its line-of-sight (LOS) with respect to the leader) arbitrarily close to the prescribed values, enabling the follower to maintain formation on a hemispherical region behind the leader. The proposed IGC framework directly maps leader-follower relative range dynamics to throttle commands, and the follower's velocity orientation relative to the LOS to aerodynamic control surface deflections. This enables synergism between guidance and control subsystems. The control design uses a dynamic surface control-based backstepping approach to achieve convergence to the desired formation set, where Lyapunov barrier functions are incorporated to ensure the follower's bearing angle is constrained within specified bounds. Rigorous stability analysis guarantees uniform ultimate boundedness of all error states and strict constraint satisfaction in the presence of aerodynamic nonlinearities. The proposed flexible formation scheme allows the follower to have an orientation mismatch relative to the leader to execute anticipatory reconfiguration by transitioning between the relative positions in the admissible formation set when the leader aggressively maneuvers. The proposed IGC law relies only on relative information and onboard sensors without the information about the leader's maneuver, making it suitable for GPS-denied or non-cooperative scenarios. Finally, we present simulation results to vindicate the effectiveness and robustness of our approach.
>
---
#### [new 028] Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于人工智能与机器人领域任务，旨在提升人形机器人对未来状态的预测能力。论文提出了1X世界模型挑战，包含未来图像帧采样与离散潜在码压缩两个赛道。研究团队分别采用视频生成模型与时空变换模型完成任务，并通过技术改进取得优异成绩。**

- **链接: [http://arxiv.org/pdf/2510.07092v1](http://arxiv.org/pdf/2510.07092v1)**

> **作者:** Riccardo Mereu; Aidan Scannell; Yuxin Hou; Yi Zhao; Aditya Jitta; Antonio Dominguez; Luigi Acerbi; Amos Storkey; Paul Chang
>
> **备注:** 6 pages, 3 figures, 1X world model challenge technical report
>
> **摘要:** World models are a powerful paradigm in AI and robotics, enabling agents to reason about the future by predicting visual observations or compact latent states. The 1X World Model Challenge introduces an open-source benchmark of real-world humanoid interaction, with two complementary tracks: sampling, focused on forecasting future image frames, and compression, focused on predicting future discrete latent codes. For the sampling track, we adapt the video generation foundation model Wan-2.2 TI2V-5B to video-state-conditioned future frame prediction. We condition the video generation on robot states using AdaLN-Zero, and further post-train the model using LoRA. For the compression track, we train a Spatio-Temporal Transformer model from scratch. Our models achieve 23.0 dB PSNR in the sampling task and a Top-500 CE of 6.6386 in the compression task, securing 1st place in both challenges.
>
---
#### [new 029] WristWorld: Generating Wrist-Views via 4D World Models for Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉生成与机器人操控任务，旨在解决腕视图数据稀缺问题。通过提出WristWorld模型，利用锚视图生成几何一致的腕视图视频，提升VLA模型的操作性能。**

- **链接: [http://arxiv.org/pdf/2510.07313v1](http://arxiv.org/pdf/2510.07313v1)**

> **作者:** Zezhong Qian; Xiaowei Chi; Yuming Li; Shizun Wang; Zhiyuan Qin; Xiaozhu Ju; Sirui Han; Shanghang Zhang
>
> **摘要:** Wrist-view observations are crucial for VLA models as they capture fine-grained hand-object interactions that directly enhance manipulation performance. Yet large-scale datasets rarely include such recordings, resulting in a substantial gap between abundant anchor views and scarce wrist views. Existing world models cannot bridge this gap, as they require a wrist-view first frame and thus fail to generate wrist-view videos from anchor views alone. Amid this gap, recent visual geometry models such as VGGT emerge with geometric and cross-view priors that make it possible to address extreme viewpoint shifts. Inspired by these insights, we propose WristWorld, the first 4D world model that generates wrist-view videos solely from anchor views. WristWorld operates in two stages: (i) Reconstruction, which extends VGGT and incorporates our Spatial Projection Consistency (SPC) Loss to estimate geometrically consistent wrist-view poses and 4D point clouds; (ii) Generation, which employs our video generation model to synthesize temporally coherent wrist-view videos from the reconstructed perspective. Experiments on Droid, Calvin, and Franka Panda demonstrate state-of-the-art video generation with superior spatial consistency, while also improving VLA performance, raising the average task completion length on Calvin by 3.81% and closing 42.4% of the anchor-wrist view gap.
>
---
#### [new 030] DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文任务是交通行为仿真，旨在解决现有模仿学习方法难以建模真实交通行为的问题。作者提出DecompGAIL，通过分解真实感、过滤误导交互，并引入社会PPO目标，提升多智能体设置下的稳定性和仿真真实性。**

- **链接: [http://arxiv.org/pdf/2510.06913v1](http://arxiv.org/pdf/2510.06913v1)**

> **作者:** Ke Guo; Haochen Liu; Xiaojun Wu; Chen Lv
>
> **摘要:** Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability: irrelevant interaction misguidance, where a discriminator penalizes an ego vehicle's realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego-map and ego-neighbor components, filtering out misleading neighbor: neighbor and neighbor: map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark.
>
---
#### [new 031] Introspection in Learned Semantic Scene Graph Localisation
- **分类: cs.LG; cs.AI; cs.CV; cs.RO; I.2.10; I.2.9; I.4.8; I.5.2; I.5.1**

- **简介: 该论文属于语义定位任务，旨在研究语义如何影响定位性能与鲁棒性。作者在自监督对比学习框架下训练定位网络，并通过事后分析探究模型是否过滤环境噪声、关注显著地标。他们验证了解释性方法的可靠性，发现集成梯度与注意力权重最有效，并揭示模型隐式弱化常见物体的权重，最终实现对场景定义的可解释、鲁棒定位。**

- **链接: [http://arxiv.org/pdf/2510.07053v1](http://arxiv.org/pdf/2510.07053v1)**

> **作者:** Manshika Charvi Bissessur; Efimia Panagiotaki; Daniele De Martini
>
> **备注:** IEEE IROS 2025 Workshop FAST
>
> **摘要:** This work investigates how semantics influence localisation performance and robustness in a learned self-supervised, contrastive semantic localisation framework. After training a localisation network on both original and perturbed maps, we conduct a thorough post-hoc introspection analysis to probe whether the model filters environmental noise and prioritises distinctive landmarks over routine clutter. We validate various interpretability methods and present a comparative reliability analysis. Integrated gradients and Attention Weights consistently emerge as the most reliable probes of learned behaviour. A semantic class ablation further reveals an implicit weighting in which frequent objects are often down-weighted. Overall, the results indicate that the model learns noise-robust, semantically salient relations about place definition, thereby enabling explainable registration under challenging visual and structural variations.
>
---
#### [new 032] Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云语义分割任务，旨在解决地面激光扫描数据手动标注成本高的问题。作者提出了一种半自动化、结合不确定性评估的标注流程，通过球面投影、特征增强、集成学习与可视化工具，减少标注工作量并保持精度。同时构建了Mangrove3D数据集，并验证了方法在多个数据集上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.06582v1](http://arxiv.org/pdf/2510.06582v1)**

> **作者:** Fei Zhang; Rob Chancia; Josie Clapp; Amirhossein Hassanzadeh; Dimah Dera; Richard MacKenzie; Jan van Aardt
>
> **摘要:** Accurate semantic segmentation of terrestrial laser scanning (TLS) point clouds is limited by costly manual annotation. We propose a semi-automated, uncertainty-aware pipeline that integrates spherical projection, feature enrichment, ensemble learning, and targeted annotation to reduce labeling effort, while sustaining high accuracy. Our approach projects 3D points to a 2D spherical grid, enriches pixels with multi-source features, and trains an ensemble of segmentation networks to produce pseudo-labels and uncertainty maps, the latter guiding annotation of ambiguous regions. The 2D outputs are back-projected to 3D, yielding densely annotated point clouds supported by a three-tier visualization suite (2D feature maps, 3D colorized point clouds, and compact virtual spheres) for rapid triage and reviewer guidance. Using this pipeline, we build Mangrove3D, a semantic segmentation TLS dataset for mangrove forests. We further evaluate data efficiency and feature importance to address two key questions: (1) how much annotated data are needed and (2) which features matter most. Results show that performance saturates after ~12 annotated scans, geometric features contribute the most, and compact nine-channel stacks capture nearly all discriminative power, with the mean Intersection over Union (mIoU) plateauing at around 0.76. Finally, we confirm the generalization of our feature-enrichment strategy through cross-dataset tests on ForestSemantic and Semantic3D. Our contributions include: (i) a robust, uncertainty-aware TLS annotation pipeline with visualization tools; (ii) the Mangrove3D dataset; and (iii) empirical guidance on data efficiency and feature importance, thus enabling scalable, high-quality segmentation of TLS point clouds for ecological monitoring and beyond. The dataset and processing scripts are publicly available at https://fz-rit.github.io/through-the-lidars-eye/.
>
---
#### [new 033] ELMUR: External Layer Memory with Update/Rewrite for Long-Horizon RL
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文提出ELMUR，一种带外部记忆的Transformer架构，解决长视野、部分可观测环境中的强化学习问题。通过分层记忆嵌入与双向交叉注意力，结合LRU更新机制，显著延长有效视野，提升决策性能。实验显示其在合成任务和机器人控制任务中均表现优异。**

- **链接: [http://arxiv.org/pdf/2510.07151v1](http://arxiv.org/pdf/2510.07151v1)**

> **作者:** Egor Cherepanov; Alexey K. Kovalev; Aleksandr I. Panov
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** Real-world robotic agents must act under partial observability and long horizons, where key cues may appear long before they affect decision making. However, most modern approaches rely solely on instantaneous information, without incorporating insights from the past. Standard recurrent or transformer models struggle with retaining and leveraging long-term dependencies: context windows truncate history, while naive memory extensions fail under scale and sparsity. We propose ELMUR (External Layer Memory with Update/Rewrite), a transformer architecture with structured external memory. Each layer maintains memory embeddings, interacts with them via bidirectional cross-attention, and updates them through an Least Recently Used (LRU) memory module using replacement or convex blending. ELMUR extends effective horizons up to 100,000 times beyond the attention window and achieves a 100% success rate on a synthetic T-Maze task with corridors up to one million steps. In POPGym, it outperforms baselines on more than half of the tasks. On MIKASA-Robo sparse-reward manipulation tasks with visual observations, it nearly doubles the performance of strong baselines. These results demonstrate that structured, layer-local external memory offers a simple and scalable approach to decision making under partial observability.
>
---
#### [new 034] Terrain-Aided Navigation Using a Point Cloud Measurement Sensor
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 论文研究基于点云测量的地形辅助导航，旨在提升惯性导航系统的状态估计精度。通过对比两种点云测量模型（光线投射与滑动网格），分析其可观测性，并以雷达高度计为基准验证性能，结果显示点云测量精度更高，适用模型取决于计算资源。**

- **链接: [http://arxiv.org/pdf/2510.06470v1](http://arxiv.org/pdf/2510.06470v1)**

> **作者:** Abdülbaki Şanlan; Fatih Erol; Murad Abu-Khalaf; Emre Koyuncu
>
> **摘要:** We investigate the use of a point cloud measurement in terrain-aided navigation. Our goal is to aid an inertial navigation system, by exploring ways to generate a useful measurement innovation error for effective nonlinear state estimation. We compare two such measurement models that involve the scanning of a digital terrain elevation model: a) one that is based on typical ray-casting from a given pose, that returns the predicted point cloud measurement from that pose, and b) another computationally less intensive one that does not require raycasting and we refer to herein as a sliding grid. Besides requiring a pose, it requires the pattern of the point cloud measurement itself and returns a predicted point cloud measurement. We further investigate the observability properties of the altitude for both measurement models. As a baseline, we compare the use of a point cloud measurement performance to the use of a radar altimeter and show the gains in accuracy. We conclude by showing that a point cloud measurement outperforms the use of a radar altimeter, and the point cloud measurement model to use depends on the computational resources
>
---
#### [new 035] HARP-NeXt: High-Speed and Accurate Range-Point Fusion Network for 3D LiDAR Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 本文属于3D LiDAR语义分割任务，旨在解决现有方法在准确性和速度之间的权衡问题。作者提出HARP-NeXt网络，通过高效预处理、Conv-SE-NeXt模块和多尺度range-point融合结构，在保持高精度的同时显著提升推理速度，适用于资源受限的嵌入式系统。**

- **链接: [http://arxiv.org/pdf/2510.06876v1](http://arxiv.org/pdf/2510.06876v1)**

> **作者:** Samir Abou Haidar; Alexandre Chariot; Mehdi Darouich; Cyril Joly; Jean-Emmanuel Deschaud
>
> **备注:** Accepted at IROS 2025 (IEEE/RSJ International Conference on Intelligent Robots and Systems)
>
> **摘要:** LiDAR semantic segmentation is crucial for autonomous vehicles and mobile robots, requiring high accuracy and real-time processing, especially on resource-constrained embedded systems. Previous state-of-the-art methods often face a trade-off between accuracy and speed. Point-based and sparse convolution-based methods are accurate but slow due to the complexity of neighbor searching and 3D convolutions. Projection-based methods are faster but lose critical geometric information during the 2D projection. Additionally, many recent methods rely on test-time augmentation (TTA) to improve performance, which further slows the inference. Moreover, the pre-processing phase across all methods increases execution time and is demanding on embedded platforms. Therefore, we introduce HARP-NeXt, a high-speed and accurate LiDAR semantic segmentation network. We first propose a novel pre-processing methodology that significantly reduces computational overhead. Then, we design the Conv-SE-NeXt feature extraction block to efficiently capture representations without deep layer stacking per network stage. We also employ a multi-scale range-point fusion backbone that leverages information at multiple abstraction levels to preserve essential geometric details, thereby enhancing accuracy. Experiments on the nuScenes and SemanticKITTI benchmarks show that HARP-NeXt achieves a superior speed-accuracy trade-off compared to all state-of-the-art methods, and, without relying on ensemble models or TTA, is comparable to the top-ranked PTv3, while running 24$\times$ faster. The code is available at https://github.com/SamirAbouHaidar/HARP-NeXt
>
---
#### [new 036] Artists' Views on Robotics Involvement in Painting Productions
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究艺术家对机器人参与绘画创作的看法，属于人机协作任务。它探讨机器人技术如何影响艺术创作，通过让八位艺术家分别与人类和机器人合作绘画并进行访谈分析，揭示了人机共创的体验特点及价值。**

- **链接: [http://arxiv.org/pdf/2510.07063v1](http://arxiv.org/pdf/2510.07063v1)**

> **作者:** Francesca Cocchella; Nilay Roy Choudhury; Eric Chen; Patrícia Alves-Oliveira
>
> **备注:** 10 pages, 9 figures, submitted to RAM special issue: Arts and Robotics
>
> **摘要:** As robotic technologies evolve, their potential in artistic creation becomes an increasingly relevant topic of inquiry. This study explores how professional abstract artists perceive and experience co-creative interactions with an autonomous painting robotic arm. Eight artists engaged in six painting sessions -- three with a human partner, followed by three with the robot -- and subsequently participated in semi-structured interviews analyzed through reflexive thematic analysis. Human-human interactions were described as intuitive, dialogic, and emotionally engaging, whereas human-robot sessions felt more playful and reflective, offering greater autonomy and prompting for novel strategies to overcome the system's limitations. This work offers one of the first empirical investigations into artists' lived experiences with a robot, highlighting the value of long-term engagement and a multidisciplinary approach to human-robot co-creation.
>
---
## 更新

#### [replaced 001] Estimating the Joint Probability of Scenario Parameters with Gaussian Mixture Copula Models
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.10098v2](http://arxiv.org/pdf/2506.10098v2)**

> **作者:** Christian Reichenbächer; Philipp Rank; Jochen Hipp; Oliver Bringmann
>
> **备注:** 8 pages, 4 figures; This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper presents the first application of Gaussian Mixture Copula Models to the statistical modeling of driving scenarios for the safety validation of automated driving systems. Knowledge of the joint probability distribution of scenario parameters is essential for scenario-based safety assessment, where risk quantification depends on the likelihood of concrete parameter combinations. Gaussian Mixture Copula Models bring together the multimodal expressivity of Gaussian Mixture Models and the flexibility of copulas, enabling separate modeling of marginal distributions and dependencies. We benchmark Gaussian Mixture Copula Models against previously proposed approaches - Gaussian Mixture Models and Gaussian Copula Models - using real-world driving data drawn from scenarios defined in United Nations Regulation No. 157. Our evaluation across approximately 18 million scenario instances demonstrates that Gaussian Mixture Copula Models consistently surpass Gaussian Copula Models and perform better than, or at least comparably to, Gaussian Mixture Models, as measured by both log-likelihood and Sinkhorn distance. These results are promising for the adoption of Gaussian Mixture Copula Models as a statistical foundation for future scenario-based validation frameworks.
>
---
#### [replaced 002] EffiTune: Diagnosing and Mitigating Training Inefficiency for Parameter Tuner in Robot Navigation System
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.10832v2](http://arxiv.org/pdf/2409.10832v2)**

> **作者:** Shiwei Feng; Xuan Chen; Zikang Xiong; Zhiyuan Cheng; Yifei Gao; Siyuan Cheng; Sayali Kate; Xiangyu Zhang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Robot navigation systems are critical for various real-world applications such as delivery services, hospital logistics, and warehouse management. Although classical navigation methods provide interpretability, they rely heavily on expert manual tuning, limiting their adaptability. Conversely, purely learning-based methods offer adaptability but often lead to instability and erratic robot behaviors. Recently introduced parameter tuners aim to balance these approaches by integrating data-driven adaptability into classical navigation frameworks. However, the parameter tuning process currently suffers from training inefficiencies and redundant sampling, with critical regions in environment often underrepresented in training data. In this paper, we propose EffiTune, a novel framework designed to diagnose and mitigate training inefficiency for parameter tuners in robot navigation systems. EffiTune first performs robot-behavior-guided diagnostics to pinpoint critical bottlenecks and underrepresented regions. It then employs a targeted up-sampling strategy to enrich the training dataset with critical samples, significantly reducing redundancy and enhancing training efficiency. Our comprehensive evaluation demonstrates that EffiTune achieves more than a 13.5% improvement in navigation performance, enhanced robustness in out-of-distribution scenarios, and a 4x improvement in training efficiency within the same computational budget.
>
---
#### [replaced 003] Context Matters! Relaxing Goals with LLMs for Feasible 3D Scene Planning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.15828v2](http://arxiv.org/pdf/2506.15828v2)**

> **作者:** Emanuele Musumeci; Michele Brienza; Francesco Argenziano; Abdel Hakim Drid; Vincenzo Suriani; Daniele Nardi; Domenico D. Bloisi
>
> **摘要:** Embodied agents need to plan and act reliably in real and complex 3D environments. Classical planning (e.g., PDDL) offers structure and guarantees, but in practice it fails under noisy perception and incorrect predicate grounding. On the other hand, Large Language Models (LLMs)-based planners leverage commonsense reasoning, yet frequently propose actions that are unfeasible or unsafe. Following recent works that combine the two approaches, we introduce ContextMatters, a framework that fuses LLMs and classical planning to perform hierarchical goal relaxation: the LLM helps ground symbols to the scene and, when the target is unreachable, it proposes functionally equivalent goals that progressively relax constraints, adapting the goal to the context of the agent's environment. Operating on 3D Scene Graphs, this mechanism turns many nominally unfeasible tasks into tractable plans and enables context-aware partial achievement when full completion is not achievable. Our experimental results show a +52.45% Success Rate improvement over state-of-the-art LLMs+PDDL baseline, demonstrating the effectiveness of our approach. Moreover, we validate the execution of ContextMatter in a real world scenario by deploying it on a TIAGo robot. Code, dataset, and supplementary materials are available to the community at https://lab-rococo-sapienza.github.io/context-matters/.
>
---
#### [replaced 004] Development of a magnetorheological hand exoskeleton featuring a high force-to-power ratio for enhanced grip endurance
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.15915v3](http://arxiv.org/pdf/2503.15915v3)**

> **作者:** Wenbo Li; Xianlong Mai; Ying Li; Weihua Li; Shiwu Zhang; Lei Deng; Shuaishuai Sun
>
> **摘要:** Hand exoskeletons have significant potential in labor-intensive fields by mitigating hand grip fatigue, enhancing hand strength, and preventing injuries. However, most of the traditional hand exoskeletons are driven by motors, whose output force is limited in the constrained installation conditions. Besides, they also come with the disadvantages of high power consumption, complex and bulky assistive systems, and high instability. In this work, we develop a novel hand exoskeleton integrated with innovative magnetorheological (MR) clutches that offers a high force-to-power ratio to improve grip endurance. The clutch features an enhanced structure design, a micro roller enhancing structure, which can significantly boost output forces. The experimental data demonstrate that, when it is supplied with 2 V, the clutch can deliver a peak holding force of 381.15 N-55 times that when no voltage is provided (7 N). In this scenario, it only consumes 1.38 W, yielding a force-to-power ratio of 256.75N/W, which is 2.35 times higher than the best-reported actuator used for hand exoskeletons. This capability enables the designed MRHE to provide approximately 419.79 N support force for gripping. The designed MR hand exoskeleton is highly integrated, comprising an exoskeleton frame, MR clutches, a control unit, and a battery. Evaluations through static grip endurance tests and dynamic carrying and lifting tests confirm that the MR hand exoskeleton can effectively reduce muscle fatigue, extend grip endurance, and minimize injuries. These findings highlight its strong potential for practical applications in repetitive tasks such as carrying and lifting in industrial settings.
>
---
#### [replaced 005] P2 Explore: Efficient Exploration in Unknown Cluttered Environment with Floor Plan Prediction
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.10878v4](http://arxiv.org/pdf/2409.10878v4)**

> **作者:** Kun Song; Gaoming Chen; Masayoshi Tomizuka; Wei Zhan; Zhenhua Xiong; Mingyu Ding
>
> **备注:** 7 pages, Accepted by IROS 2025, Open-sourced at https://github.com/song-kun/P2Explore
>
> **摘要:** Robot exploration aims at the reconstruction of unknown environments, and it is important to achieve it with shorter paths. Traditional methods focus on optimizing the visiting order of frontiers based on current observations, which may lead to local-minimal results. Recently, by predicting the structure of the unseen environment, the exploration efficiency can be further improved. However, in a cluttered environment, due to the randomness of obstacles, the ability to predict is weak. Moreover, this inaccuracy will lead to limited improvement in exploration. Therefore, we propose FPUNet which can be efficient in predicting the layout of noisy indoor environments. Then, we extract the segmentation of rooms and construct their topological connectivity based on the predicted map. The visiting order of these predicted rooms is optimized which can provide high-level guidance for exploration. The FPUNet is compared with other network architectures which demonstrates it is the SOTA method for this task. Extensive experiments in simulations show that our method can shorten the path length by 2.18% to 34.60% compared to the baselines.
>
---
#### [replaced 006] Generating and Optimizing Topologically Distinct Guesses for Mobile Manipulator Path Planning with Path Constraints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.20635v2](http://arxiv.org/pdf/2410.20635v2)**

> **作者:** Rufus Cheuk Yin Wong; Mayank Sewlia; Adrian Wiltz; Dimos V. Dimarogonas
>
> **摘要:** Optimal path planning is prone to convergence to local, rather than global, optima. This is often the case for mobile manipulators due to nonconvexities induced by obstacles, robot kinematics and constraints. This paper focuses on planning under end effector path constraints and attempts to circumvent the issue of converging to a local optimum. We propose a pipeline that first discovers multiple homotopically distinct paths, and then optimizes them to obtain multiple distinct local optima. The best out of these distinct local optima is likely to be close to the global optimum. We demonstrate the effectiveness of our pipeline in the optimal path planning of mobile manipulators in the presence of path and obstacle constraints.
>
---
#### [replaced 007] Interleave-VLA: Enhancing Robot Manipulation with Interleaved Image-Text Instructions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.02152v2](http://arxiv.org/pdf/2505.02152v2)**

> **作者:** Cunxin Fan; Xiaosong Jia; Yihang Sun; Yixiao Wang; Jianglan Wei; Ziyang Gong; Xiangyu Zhao; Masayoshi Tomizuka; Xue Yang; Junchi Yan; Mingyu Ding
>
> **摘要:** The rise of foundation models paves the way for generalist robot policies in the physical world. Existing methods relying on text-only instructions often struggle to generalize to unseen scenarios. We argue that interleaved image-text inputs offer richer and less biased context and enable robots to better handle unseen tasks with more versatile human-robot interaction. Building on this insight, Interleave-VLA, the first robot learning paradigm capable of comprehending interleaved image-text instructions and directly generating continuous action sequences in the physical world, is introduced. It offers a natural, flexible, and model-agnostic paradigm that extends state-of-the-art vision-language-action (VLA) models with minimal modifications while achieving strong zero-shot generalization. Interleave-VLA also includes an automatic pipeline that converts text instructions from Open X-Embodiment into interleaved image-text instructions, resulting in a large-scale real-world interleaved embodied dataset with 210k episodes. Comprehensive evaluation in simulation and the real world shows that Interleave-VLA offers two major benefits: (1) improves out-of-domain generalization to unseen objects by 2x compared to text input baselines, (2) supports flexible task interfaces and diverse instructions in a zero-shot manner, such as hand-drawn sketches. We attribute Interleave-VLA's strong zero-shot capability to the use of instruction images, which effectively mitigate hallucinations, and the inclusion of heterogeneous multimodal datasets, enriched with Internet-sourced images, offering potential for scalability. More information is available at https://interleave-vla.github.io/Interleave-VLA-Anonymous/
>
---
#### [replaced 008] Mitigating Cross-Modal Distraction and Ensuring Geometric Feasibility via Affordance-Guided and Self-Consistent MLLMs for Task Planning in Instruction-Following Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13055v2](http://arxiv.org/pdf/2503.13055v2)**

> **作者:** Yu-Hong Shen; Chuan-Yu Wu; Yi-Ru Yang; Yen-Ling Tai; Yi-Ting Chen
>
> **摘要:** We investigate the use of Multimodal Large Language Models (MLLMs) with in-context learning for closed-loop task planning in instruction-following manipulation. We identify four essential requirements for successful task planning: quantity estimation, reachability analysis, relative positioning, and collision avoidance. However, existing benchmarks fail to support holistic evaluation across all these aspects. To address this gap, we introduce \textbf{QuARC} (Quantity, Analysis, Relative positioning, Collision), a new benchmark based on a food preparation scenario that integrates all four challenges. Using QuARC, we reveal two major limitations of current MLLMs: cross-modal distraction and geometric infeasibility. To tackle these, we adapt Chain-of-Thought with Self-Consistency to mitigate reasoning loss from cross-modal distractions and incorporate an affordance predictor to guide planning based on geometric feasibility. Our comprehensive evaluation analyzes performance across multiple baselines and explains sources of improvement. Our method achieves a 76.7\% success rate on the benchmark, significantly outperforming the ViLa baseline (36.7\%), without requiring additional finetuning. Code and dataset are available at https://hcis-lab.github.io/Affordance-Guided-Self-Consistent-MLLM.
>
---
#### [replaced 009] BIM Informed Visual SLAM for Construction Monitoring
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13972v2](http://arxiv.org/pdf/2509.13972v2)**

> **作者:** Asier Bikandi-Noya; Miguel Fernandez-Cortizas; Muhammad Shaheer; Ali Tourani; Holger Voos; Jose Luis Sanchez-Lopez
>
> **备注:** 8 pages, 5 tables, 4 figures
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) is a key tool for monitoring construction sites, where aligning the evolving as-built state with the as-planned design enables early error detection and reduces costly rework. LiDAR-based SLAM achieves high geometric precision, but its sensors are typically large and power-demanding, limiting their use on portable platforms. Visual SLAM offers a practical alternative with lightweight cameras already embedded in most mobile devices. however, visually mapping construction environments remains challenging: repetitive layouts, occlusions, and incomplete or low-texture structures often cause drift in the trajectory map. To mitigate this, we propose an RGB-D SLAM system that incorporates the Building Information Model (BIM) as structural prior knowledge. Instead of relying solely on visual cues, our system continuously establishes correspondences between detected wall and their BIM counterparts, which are then introduced as constraints in the back-end optimization. The proposed method operates in real time and has been validated on real construction sites, reducing trajectory error by an average of 23.71% and map RMSE by 7.14% compared to visual SLAM baselines. These results demonstrate that BIM constraints enable reliable alignment of the digital plan with the as-built scene, even under partially constructed conditions.
>
---
#### [replaced 010] Control of Humanoid Robots with Parallel Mechanisms using Differential Actuation Models
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.22459v2](http://arxiv.org/pdf/2503.22459v2)**

> **作者:** Victor Lutz; Ludovic de Matteis; Virgile Batto; Nicolas Mansard
>
> **摘要:** Several recently released humanoid robots, inspired by the mechanical design of Cassie, employ actuator configurations in which the motors are displaced from the joints to reduce leg inertia. While studies accounting for the full kinematic complexity have demonstrated the benefits of these designs, the associated loop-closure constraints greatly increase computational cost and limit their use in control and learning. As a result, the non-linear transmission is often approximated by a constant reduction ratio, preventing exploitation of the mechanism's full capabilities. This paper introduces a compact analytical formulation for the two standard knee and ankle mechanisms that captures the exact non-linear transmission while remaining computationally efficient. The model is fully differentiable up to second order with a minimal formulation, enabling low-cost evaluation of dynamic derivatives for trajectory optimization and of the apparent transmission impedance for reinforcement learning. We integrate this formulation into trajectory optimization and locomotion policy learning, and compare it against simplified constant-ratio approaches. Hardware experiments demonstrate improved accuracy and robustness, showing that the proposed method provides a practical means to incorporate parallel actuation into modern control algorithms.
>
---
#### [replaced 011] Automating RT Planning at Scale: High Quality Data For AI Training
- **分类: cs.HC; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.11803v4](http://arxiv.org/pdf/2501.11803v4)**

> **作者:** Riqiang Gao; Mamadou Diallo; Han Liu; Anthony Magliari; Jonathan Sackett; Wilko Verbakel; Sandra Meyers; Rafe Mcbeth; Masoud Zarepisheh; Simon Arberet; Martin Kraus; Florin C. Ghesu; Ali Kamen
>
> **备注:** radiotherapy planning, data for AI training
>
> **摘要:** Radiotherapy (RT) planning is complex, subjective, and time-intensive. Advances with artificial intelligence (AI) promise to improve its precision and efficiency, but progress is often limited by the scarcity of large, standardized datasets. To address this, we introduce the Automated Iterative RT Planning (AIRTP) system, a scalable solution for generating high-quality treatment plans. This scalable solution is designed to generate substantial volumes of consistently high-quality treatment plans, overcoming a key obstacle in the advancement of AI-driven RT planning. Our AIRTP pipeline adheres to clinical guidelines and automates essential steps, including organ-at-risk (OAR) contouring, helper structure creation, beam setup, optimization, and plan quality improvement, using AI integrated with RT planning software like Varian Eclipse. Furthermore, a novel approach for determining optimization parameters to reproduce 3D dose distributions, i.e. a method to convert dose predictions to deliverable treatment plans constrained by machine limitations is proposed. A comparative analysis of plan quality reveals that our automated pipeline produces treatment plans of quality comparable to those generated manually, which traditionally require several hours of labor per plan. Committed to public research, the first data release of our AIRTP pipeline includes nine cohorts covering head-and-neck and lung cancer sites to support an AAPM 2025 challenge. To our best knowledge, this dataset features more than 10 times number of plans compared to the largest existing well-curated public dataset. Repo: https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge.
>
---
#### [replaced 012] Online Hybrid-Belief POMDP with Coupled Semantic-Geometric Models
- **分类: cs.RO; None**

- **链接: [http://arxiv.org/pdf/2501.11202v5](http://arxiv.org/pdf/2501.11202v5)**

> **作者:** Tuvy Lemberg; Vadim Indelman
>
> **备注:** 20 pages, 9 figures
>
> **摘要:** Robots operating in complex and unknown environments frequently require geometric-semantic representations of the environment to safely perform their tasks. While inferring the environment, they must account for many possible scenarios when planning future actions. Since objects' class types are discrete and the robot's self-pose and the objects' poses are continuous, the environment can be represented by a hybrid discrete-continuous belief which is updated according to models and incoming data. Prior probabilities and observation models representing the environment can be learned from data using deep learning algorithms. Such models often couple environmental semantic and geometric properties. As a result, semantic variables are interconnected, causing semantic state space dimensionality to increase exponentially. In this paper, we consider planning under uncertainty using partially observable Markov decision processes (POMDPs) with hybrid semantic-geometric beliefs. The models and priors consider the coupling between semantic and geometric variables. Within POMDP, we introduce the concept of semantically aware safety. Obtaining representative samples of the theoretical hybrid belief, required for estimating the value function, is very challenging. As a key contribution, we develop a novel form of the hybrid belief and leverage it to sample representative samples. We show that under certain conditions, the value function and probability of safety can be calculated efficiently with an explicit expectation over all possible semantic mappings. Our simulations show that our estimates of the objective function and probability of safety achieve similar levels of accuracy compared to estimators that run exhaustively on the entire semantic state-space using samples from the theoretical hybrid belief. Nevertheless, the complexity of our estimators is polynomial rather than exponential.
>
---
#### [replaced 013] Diffusion Trajectory-guided Policy for Long-horizon Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.10040v2](http://arxiv.org/pdf/2502.10040v2)**

> **作者:** Shichao Fan; Quantao Yang; Yajie Liu; Kun Wu; Zhengping Che; Qingjie Liu; Min Wan
>
> **备注:** 8 pages, 5 figures, accepted to IEEE Robotics and Automation Letters (RAL)
>
> **摘要:** Recently, Vision-Language-Action models (VLA) have advanced robot imitation learning, but high data collection costs and limited demonstrations hinder generalization and current imitation learning methods struggle in out-of-distribution scenarios, especially for long-horizon tasks. A key challenge is how to mitigate compounding errors in imitation learning, which lead to cascading failures over extended trajectories. To address these challenges, we propose the Diffusion Trajectory-guided Policy (DTP) framework, which generates 2D trajectories through a diffusion model to guide policy learning for long-horizon tasks. By leveraging task-relevant trajectories, DTP provides trajectory-level guidance to reduce error accumulation. Our two-stage approach first trains a generative vision-language model to create diffusion-based trajectories, then refines the imitation policy using them. Experiments on the CALVIN benchmark show that DTP outperforms state-of-the-art baselines by 25% in success rate, starting from scratch without external pretraining. Moreover, DTP significantly improves real-world robot performance.
>
---
#### [replaced 014] Robot Learning from Any Images
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.22970v2](http://arxiv.org/pdf/2509.22970v2)**

> **作者:** Siheng Zhao; Jiageng Mao; Wei Chow; Zeyu Shangguan; Tianheng Shi; Rong Xue; Yuxi Zheng; Yijia Weng; Yang You; Daniel Seita; Leonidas Guibas; Sergey Zakharov; Vitor Guizilini; Yue Wang
>
> **备注:** CoRL 2025 camera ready
>
> **摘要:** We introduce RoLA, a framework that transforms any in-the-wild image into an interactive, physics-enabled robotic environment. Unlike previous methods, RoLA operates directly on a single image without requiring additional hardware or digital assets. Our framework democratizes robotic data generation by producing massive visuomotor robotic demonstrations within minutes from a wide range of image sources, including camera captures, robotic datasets, and Internet images. At its core, our approach combines a novel method for single-view physical scene recovery with an efficient visual blending strategy for photorealistic data collection. We demonstrate RoLA's versatility across applications like scalable robotic data generation and augmentation, robot learning from Internet images, and single-image real-to-sim-to-real systems for manipulators and humanoids. Video results are available at https://sihengz02.github.io/RoLA .
>
---
#### [replaced 015] Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.05516v2](http://arxiv.org/pdf/2506.05516v2)**

> **作者:** Boyuan Deng; Luca Rossini; Jin Wang; Weijie Wang; Dimitrios Kanoulas; Nikolaos Tsagarakis
>
> **摘要:** Adaptive recovery from fall incidents are essential skills for the practical deployment of wheeled-legged robots, which uniquely combine the agility of legs with the speed of wheels for rapid recovery. However, traditional methods relying on preplanned recovery motions, simplified dynamics or sparse rewards often fail to produce robust recovery policies. This paper presents a learning-based framework integrating Episode-based Dynamic Reward Shaping and curriculum learning, which dynamically balances exploration of diverse recovery maneuvers with precise posture refinement. An asymmetric actor-critic architecture accelerates training by leveraging privileged information in simulation, while noise-injected observations enhance robustness against uncertainties. We further demonstrate that synergistic wheel-leg coordination reduces joint torque consumption by 15.8% and 26.2% and improves stabilization through energy transfer mechanisms. Extensive evaluations on two distinct quadruped platforms achieve recovery success rates up to 99.1% and 97.8% without platform-specific tuning. The supplementary material is available at https://boyuandeng.github.io/L2R-WheelLegCoordination/
>
---
#### [replaced 016] M^3RS: Multi-robot, Multi-objective, and Multi-mode Routing and Scheduling
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.16275v3](http://arxiv.org/pdf/2403.16275v3)**

> **作者:** Ishaan Mehta; Junseo Kim; Sharareh Taghipour; Sajad Saeedi
>
> **备注:** Under review
>
> **摘要:** Task execution quality significantly impacts multi-robot missions, yet existing task allocation frameworks rarely consider quality of service as a decision variable, despite its importance in applications like robotic disinfection and cleaning. We introduce the multi-robot, multi-objective, and multi-mode routing and scheduling (M3RS) problem, designed for time-constrained missions. In M3RS, each task offers multiple execution modes with varying resource needs, durations, and quality levels, allowing trade-offs across mission objectives. M3RS is modeled as a mixed-integer linear programming (MIP) problem and optimizes task sequencing and execution modes for each agent. We apply M3RS to multi-robot disinfection in healthcare and public spaces, optimizing disinfection quality and task completion rates. Through synthetic case studies, M3RS demonstrates 3-46$\%$ performance improvements over the standard task allocation method across various metrics. Further, to improve compute time, we propose a clustering-based column generation algorithm that achieves solutions comparable to or better than the baseline MIP solver while reducing computation time by 60$\%$. We also conduct case studies with simulated and real robots. Experimental videos are available on the project page: \href{https://sites.google.com/view/g-robot/m3rs/}{https://sites.google.com/view/g-robot/m3rs/}.
>
---
#### [replaced 017] Touch Speaks, Sound Feels: A Multimodal Approach to Affective and Social Touch from Robots to Humans
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07839v2](http://arxiv.org/pdf/2508.07839v2)**

> **作者:** Qiaoqiao Ren; Tony Belpaeme
>
> **摘要:** Affective tactile interaction constitutes a fundamental component of human communication. In natural human-human encounters, touch is seldom experienced in isolation; rather, it is inherently multisensory. Individuals not only perceive the physical sensation of touch but also register the accompanying auditory cues generated through contact. The integration of haptic and auditory information forms a rich and nuanced channel for emotional expression. While extensive research has examined how robots convey emotions through facial expressions and speech, their capacity to communicate social gestures and emotions via touch remains largely underexplored. To address this gap, we developed a multimodal interaction system incorporating a 5*5 grid of 25 vibration motors synchronized with audio playback, enabling robots to deliver combined haptic-audio stimuli. In an experiment involving 32 Chinese participants, ten emotions and six social gestures were presented through vibration, sound, or their combination. Participants rated each stimulus on arousal and valence scales. The results revealed that (1) the combined haptic-audio modality significantly enhanced decoding accuracy compared to single modalities; (2) each individual channel-vibration or sound-effectively supported certain emotions recognition, with distinct advantages depending on the emotional expression; and (3) gestures alone were generally insufficient for conveying clearly distinguishable emotions. These findings underscore the importance of multisensory integration in affective human-robot interaction and highlight the complementary roles of haptic and auditory cues in enhancing emotional communication.
>
---
#### [replaced 018] UltraHiT: A Hierarchical Transformer Architecture for Generalizable Internal Carotid Artery Robotic Ultrasonography
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13832v2](http://arxiv.org/pdf/2509.13832v2)**

> **作者:** Teng Wang; Haojun Jiang; Yuxuan Wang; Zhenguo Sun; Xiangjie Yan; Xiang Li; Gao Huang
>
> **摘要:** Carotid ultrasound is crucial for the assessment of cerebrovascular health, particularly the internal carotid artery (ICA). While previous research has explored automating carotid ultrasound, none has tackled the challenging ICA. This is primarily due to its deep location, tortuous course, and significant individual variations, which greatly increase scanning complexity. To address this, we propose a Hierarchical Transformer-based decision architecture, namely UltraHiT, that integrates high-level variation assessment with low-level action decision. Our motivation stems from conceptualizing individual vascular structures as morphological variations derived from a standard vascular model. The high-level module identifies variation and switches between two low-level modules: an adaptive corrector for variations, or a standard executor for normal cases. Specifically, both the high-level module and the adaptive corrector are implemented as causal transformers that generate predictions based on the historical scanning sequence. To ensure generalizability, we collected the first large-scale ICA scanning dataset comprising 164 trajectories and 72K samples from 28 subjects of both genders. Based on the above innovations, our approach achieves a 95% success rate in locating the ICA on unseen individuals, outperforming baselines and demonstrating its effectiveness. Our code will be released after acceptance.
>
---
#### [replaced 019] ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.05070v2](http://arxiv.org/pdf/2510.05070v2)**

> **作者:** Siheng Zhao; Yanjie Ze; Yue Wang; C. Karen Liu; Pieter Abbeel; Guanya Shi; Rocky Duan
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Humanoid whole-body loco-manipulation promises transformative capabilities for daily service and warehouse tasks. While recent advances in general motion tracking (GMT) have enabled humanoids to reproduce diverse human motions, these policies lack the precision and object awareness required for loco-manipulation. To this end, we introduce ResMimic, a two-stage residual learning framework for precise and expressive humanoid control from human motion data. First, a GMT policy, trained on large-scale human-only motion, serves as a task-agnostic base for generating human-like whole-body movements. An efficient but precise residual policy is then learned to refine the GMT outputs to improve locomotion and incorporate object interaction. To further facilitate efficient training, we design (i) a point-cloud-based object tracking reward for smoother optimization, (ii) a contact reward that encourages accurate humanoid body-object interactions, and (iii) a curriculum-based virtual object controller to stabilize early training. We evaluate ResMimic in both simulation and on a real Unitree G1 humanoid. Results show substantial gains in task success, training efficiency, and robustness over strong baselines. Videos are available at https://resmimic.github.io/ .
>
---
#### [replaced 020] Interpretable Robot Control via Structured Behavior Trees and Large Language Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.09621v2](http://arxiv.org/pdf/2508.09621v2)**

> **作者:** Ingrid Maéva Chekam; Ines Pastor-Martinez; Ali Tourani; Jose Andres Millan-Romera; Laura Ribeiro; Pedro Miguel Bastos Soares; Holger Voos; Jose Luis Sanchez-Lopez
>
> **备注:** 15 pages, 5 figures, 3 tables
>
> **摘要:** As intelligent robots become more integrated into human environments, there is a growing need for intuitive and reliable Human-Robot Interaction (HRI) interfaces that are adaptable and more natural to interact with. Traditional robot control methods often require users to adapt to interfaces or memorize predefined commands, limiting usability in dynamic, unstructured environments. This paper presents a novel framework that bridges natural language understanding and robotic execution by combining Large Language Models (LLMs) with Behavior Trees. This integration enables robots to interpret natural language instructions given by users and translate them into executable actions by activating domain-specific plugins. The system supports scalable and modular integration, with a primary focus on perception-based functionalities, such as person tracking and hand gesture recognition. To evaluate the system, a series of real-world experiments was conducted across diverse environments. Experimental results demonstrate that the proposed approach is practical in real-world scenarios, with an average cognition-to-execution accuracy of approximately 94%, making a significant contribution to HRI systems and robots. The complete source code of the framework is publicly available at https://github.com/snt-arg/robot_suite.
>
---
#### [replaced 021] NAR-*ICP: Neural Execution of Classical ICP-based Pointcloud Registration Algorithms
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.11031v3](http://arxiv.org/pdf/2410.11031v3)**

> **作者:** Efimia Panagiotaki; Daniele De Martini; Lars Kunze; Paul Newman; Petar Veličković
>
> **备注:** 19 pages, 16 tables, 7 figures
>
> **摘要:** This study explores the intersection of neural networks and classical robotics algorithms through the Neural Algorithmic Reasoning (NAR) blueprint, enabling the training of neural networks to reason like classical robotics algorithms by learning to execute them. Algorithms are integral to robotics and safety-critical applications due to their predictable and consistent performance through logical and mathematical principles. In contrast, while neural networks are highly adaptable, handling complex, high-dimensional data and generalising across tasks, they often lack interpretability and transparency in their internal computations. To bridge the two, we propose a novel Graph Neural Network (GNN)-based framework, NAR-*ICP, that learns the intermediate computations of classical ICP-based registration algorithms, extending the CLRS Benchmark. We evaluate our approach across real-world and synthetic datasets, demonstrating its flexibility in handling complex inputs, and its potential to be used within larger learning pipelines. Our method achieves superior performance compared to the baselines, even surpassing the algorithms it was trained on, further demonstrating its ability to generalise beyond the capabilities of traditional algorithms.
>
---
#### [replaced 022] Gaze Estimation for Human-Robot Interaction: Analysis Using the NICO Platform
- **分类: cs.CV; cs.RO; I.4.9**

- **链接: [http://arxiv.org/pdf/2509.24001v2](http://arxiv.org/pdf/2509.24001v2)**

> **作者:** Matej Palider; Omar Eldardeer; Viktor Kocur
>
> **备注:** Code available at http://github.com/kocurvik/nico_gaze
>
> **摘要:** This paper evaluates the current gaze estimation methods within an HRI context of a shared workspace scenario. We introduce a new, annotated dataset collected with the NICO robotic platform. We evaluate four state-of-the-art gaze estimation models. The evaluation shows that the angular errors are close to those reported on general-purpose benchmarks. However, when expressed in terms of distance in the shared workspace the best median error is 16.48 cm quantifying the practical limitations of current methods. We conclude by discussing these limitations and offering recommendations on how to best integrate gaze estimation as a modality in HRI systems.
>
---
#### [replaced 023] BIM-Constrained Optimization for Accurate Localization and Deviation Correction in Construction Monitoring
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.17693v2](http://arxiv.org/pdf/2504.17693v2)**

> **作者:** Asier Bikandi-Noya; Muhammad Shaheer; Hriday Bavle; Jayan Jevanesan; Holger Voos; Jose Luis Sanchez-Lopez
>
> **摘要:** Augmented reality (AR) applications for construction monitoring rely on real-time environmental tracking to visualize architectural elements. However, construction sites present significant challenges for traditional tracking methods due to featureless surfaces, dynamic changes, and drift accumulation, leading to misalignment between digital models and the physical world. This paper proposes a BIM-aware drift correction method to address these challenges. Instead of relying solely on SLAM-based localization, we align ``as-built" detected planes from the real-world environment with ``as-planned" architectural planes in BIM. Our method performs robust plane matching and computes a transformation (TF) between SLAM (S) and BIM (B) origin frames using optimization techniques, minimizing drift over time. By incorporating BIM as prior structural knowledge, we can achieve improved long-term localization and enhanced AR visualization accuracy in noisy construction environments. The method is evaluated through real-world experiments, showing significant reductions in drift-induced errors and optimized alignment consistency. On average, our system achieves a reduction of 52.24% in angular deviations and a reduction of 60.8% in the distance error of the matched walls compared to the initial manual alignment by the user.
>
---
