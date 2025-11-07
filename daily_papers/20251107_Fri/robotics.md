# 机器人 cs.RO

- **最新发布 24 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
- **分类: cs.RO; cs.CV**

- **简介: Evo-1提出一种轻量级视觉-语言-动作模型，无需机器人数据预训练，在保持语义对齐下显著降低参数量与计算开销，提升部署效率，在多个基准上超越现有模型。**

- **链接: [http://arxiv.org/pdf/2511.04555v1](http://arxiv.org/pdf/2511.04555v1)**

> **作者:** Tao Lin; Yilei Zhong; Yuxin Du; Jingjing Zhang; Jiting Liu; Yinxinyu Chen; Encheng Gu; Ziyan Liu; Hongyi Cai; Yanwen Zou; Lixing Zou; Zhaoye Zhou; Gen Li; Bo Zhao
>
> **备注:** Github: https://github.com/MINT-SJTU/Evo-1
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful framework that unifies perception, language, and control, enabling robots to perform diverse tasks through multimodal understanding. However, current VLA models typically contain massive parameters and rely heavily on large-scale robot data pretraining, leading to high computational costs during training, as well as limited deployability for real-time inference. Moreover, most training paradigms often degrade the perceptual representations of the vision-language backbone, resulting in overfitting and poor generalization to downstream tasks. In this work, we present Evo-1, a lightweight VLA model that reduces computation and improves deployment efficiency, while maintaining strong performance without pretraining on robot data. Evo-1 builds on a native multimodal Vision-Language model (VLM), incorporating a novel cross-modulated diffusion transformer along with an optimized integration module, together forming an effective architecture. We further introduce a two-stage training paradigm that progressively aligns action with perception, preserving the representations of the VLM. Notably, with only 0.77 billion parameters, Evo-1 achieves state-of-the-art results on the Meta-World and RoboTwin suite, surpassing the previous best models by 12.4% and 6.9%, respectively, and also attains a competitive result of 94.8% on LIBERO. In real-world evaluations, Evo-1 attains a 78% success rate with high inference frequency and low memory overhead, outperforming all baseline methods. We release code, data, and model weights to facilitate future research on lightweight and efficient VLA models.
>
---
#### [new 002] Real-to-Sim Robot Policy Evaluation with Gaussian Splatting Simulation of Soft-Body Interactions
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出一种基于高斯泼溅的实到虚机器人策略评估框架，通过从真实视频重建软体数字孪生体，实现高保真物理与视觉仿真，解决软体操作任务中真实评估成本高、仿真失真的问题。**

- **链接: [http://arxiv.org/pdf/2511.04665v1](http://arxiv.org/pdf/2511.04665v1)**

> **作者:** Kaifeng Zhang; Shuo Sha; Hanxiao Jiang; Matthew Loper; Hyunjong Song; Guangyan Cai; Zhuo Xu; Xiaochen Hu; Changxi Zheng; Yunzhu Li
>
> **备注:** Website: https://real2sim-eval.github.io/
>
> **摘要:** Robotic manipulation policies are advancing rapidly, but their direct evaluation in the real world remains costly, time-consuming, and difficult to reproduce, particularly for tasks involving deformable objects. Simulation provides a scalable and systematic alternative, yet existing simulators often fail to capture the coupled visual and physical complexity of soft-body interactions. We present a real-to-sim policy evaluation framework that constructs soft-body digital twins from real-world videos and renders robots, objects, and environments with photorealistic fidelity using 3D Gaussian Splatting. We validate our approach on representative deformable manipulation tasks, including plush toy packing, rope routing, and T-block pushing, demonstrating that simulated rollouts correlate strongly with real-world execution performance and reveal key behavioral patterns of learned policies. Our results suggest that combining physics-informed reconstruction with high-quality rendering enables reproducible, scalable, and accurate evaluation of robotic manipulation policies. Website: https://real2sim-eval.github.io/
>
---
#### [new 003] Design and Control of a Coaxial Dual-rotor Reconfigurable Tailsitter UAV Based on Swashplateless Mechanism
- **分类: cs.RO**

- **简介: 该论文设计了一种基于无挥舞机构的同轴双转子可重构倾转旋翼无人机，解决多旋翼模式下机身受风扰大的问题，通过收放机翼与优化无挥舞控制机构，提升效率与稳定性，并完成全包线过渡飞行验证。**

- **链接: [http://arxiv.org/pdf/2511.04251v1](http://arxiv.org/pdf/2511.04251v1)**

> **作者:** Jinfeng Liang; Haocheng Guo; Ximin Lyu
>
> **备注:** 8 pages 12 figures
>
> **摘要:** The tailsitter vertical takeoff and landing (VTOL) UAV is widely used due to its lower dead weight, which eliminates the actuators and mechanisms for tilting. However, the tailsitter UAV is susceptible to wind disturbances in multi-rotor mode, as it exposes a large frontal fuselage area. To address this issue, our tailsitter UAV features a reconfigurable wing design, allowing wings to retract in multi-rotor mode and extend in fixed- wing mode. Considering power efficiency, we design a coaxial heterogeneous dual-rotor configuration, which significantly re- duces the total power consumption. To reduce structural weight and simplify structural complexity, we employ a swashplateless mechanism with an improved design to control pitch and roll in multi-rotor mode. We optimize the structure of the swashplateless mechanism by adding flapping hinges, which reduces vibration during cyclic acceleration and deceleration. Finally, we perform comprehensive transition flight tests to validate stable flight performance across the entire flight envelope of the tailsitter UAV.
>
---
#### [new 004] BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出BFM-Zero，一种基于无监督强化学习的可提示行为基础模型，用于人形机器人全身控制。通过统一的潜在空间实现零样本与少样本任务泛化，解决多任务专用策略的局限性，并成功实现在真实机器人上的跨任务迁移。**

- **链接: [http://arxiv.org/pdf/2511.04131v1](http://arxiv.org/pdf/2511.04131v1)**

> **作者:** Yitang Li; Zhengyi Luo; Tonghe Zhang; Cunxi Dai; Anssi Kanervisto; Andrea Tirinzoni; Haoyang Weng; Kris Kitani; Mateusz Guzek; Ahmed Touati; Alessandro Lazaric; Matteo Pirotta; Guanya Shi
>
> **摘要:** Building Behavioral Foundation Models (BFMs) for humanoid robots has the potential to unify diverse control tasks under a single, promptable generalist policy. However, existing approaches are either exclusively deployed on simulated humanoid characters, or specialized to specific tasks such as tracking. We propose BFM-Zero, a framework that learns an effective shared latent representation that embeds motions, goals, and rewards into a common space, enabling a single policy to be prompted for multiple downstream tasks without retraining. This well-structured latent space in BFM-Zero enables versatile and robust whole-body skills on a Unitree G1 humanoid in the real world, via diverse inference methods, including zero-shot motion tracking, goal reaching, and reward optimization, and few-shot optimization-based adaptation. Unlike prior on-policy reinforcement learning (RL) frameworks, BFM-Zero builds upon recent advancements in unsupervised RL and Forward-Backward (FB) models, which offer an objective-centric, explainable, and smooth latent representation of whole-body motions. We further extend BFM-Zero with critical reward shaping, domain randomization, and history-dependent asymmetric learning to bridge the sim-to-real gap. Those key design choices are quantitatively ablated in simulation. A first-of-its-kind model, BFM-Zero establishes a step toward scalable, promptable behavioral foundation models for whole-body humanoid control.
>
---
#### [new 005] Can Context Bridge the Reality Gap? Sim-to-Real Transfer of Context-Aware Policies
- **分类: cs.RO**

- **简介: 该论文研究机器人强化学习中的仿真到现实迁移问题，提出通过引入动力学上下文估计来增强策略泛化能力。实验表明，上下文感知策略显著优于传统无上下文方法，提升跨域迁移性能。**

- **链接: [http://arxiv.org/pdf/2511.04249v1](http://arxiv.org/pdf/2511.04249v1)**

> **作者:** Marco Iannotta; Yuxuan Yang; Johannes A. Stork; Erik Schaffernicht; Todor Stoyanov
>
> **摘要:** Sim-to-real transfer remains a major challenge in reinforcement learning (RL) for robotics, as policies trained in simulation often fail to generalize to the real world due to discrepancies in environment dynamics. Domain Randomization (DR) mitigates this issue by exposing the policy to a wide range of randomized dynamics during training, yet leading to a reduction in performance. While standard approaches typically train policies agnostic to these variations, we investigate whether sim-to-real transfer can be improved by conditioning the policy on an estimate of the dynamics parameters -- referred to as context. To this end, we integrate a context estimation module into a DR-based RL framework and systematically compare SOTA supervision strategies. We evaluate the resulting context-aware policies in both a canonical control benchmark and a real-world pushing task using a Franka Emika Panda robot. Results show that context-aware policies outperform the context-agnostic baseline across all settings, although the best supervision strategy depends on the task.
>
---
#### [new 006] Enhancing Fault-Tolerant Space Computing: Guidance Navigation and Control (GNC) and Landing Vision System (LVS) Implementations on Next-Gen Multi-Core Processors
- **分类: cs.RO**

- **简介: 该论文面向深空探测任务，解决传统航天硬件算力不足与容错性差的问题，提出在多核处理器上部署GNC与LVS算法，并设计ARBITER容错机制，实现显著性能提升与实时错误恢复。**

- **链接: [http://arxiv.org/pdf/2511.04052v1](http://arxiv.org/pdf/2511.04052v1)**

> **作者:** Kyongsik Yun; David Bayard; Gerik Kubiak; Austin Owens; Andrew Johnson; Ryan Johnson; Dan Scharf; Thomas Lu
>
> **摘要:** Future planetary exploration missions demand high-performance, fault-tolerant computing to enable autonomous Guidance, Navigation, and Control (GNC) and Lander Vision System (LVS) operations during Entry, Descent, and Landing (EDL). This paper evaluates the deployment of GNC and LVS algorithms on next-generation multi-core processors--HPSC, Snapdragon VOXL2, and AMD Xilinx Versal--demonstrating up to 15x speedup for LVS image processing and over 250x speedup for Guidance for Fuel-Optimal Large Divert (GFOLD) trajectory optimization compared to legacy spaceflight hardware. To ensure computational reliability, we present ARBITER (Asynchronous Redundant Behavior Inspection for Trusted Execution and Recovery), a Multi-Core Voting (MV) mechanism that performs real-time fault detection and correction across redundant cores. ARBITER is validated in both static optimization tasks (GFOLD) and dynamic closed-loop control (Attitude Control System). A fault injection study further identifies the gradient computation stage in GFOLD as the most sensitive to bit-level errors, motivating selective protection strategies and vector-based output arbitration. This work establishes a scalable and energy-efficient architecture for future missions, including Mars Sample Return, Enceladus Orbilander, and Ceres Sample Return, where onboard autonomy, low latency, and fault resilience are critical.
>
---
#### [new 007] Integrating Ergonomics and Manipulability for Upper Limb Postural Optimization in Bimanual Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文针对双臂人机协同搬运任务，提出融合人体工效与操作性的上肢姿态优化方法，通过最小化成本函数优化关节角度，并结合机器人参考位姿引导与MPIC控制器，显著改善肌肉负荷。**

- **链接: [http://arxiv.org/pdf/2511.04009v1](http://arxiv.org/pdf/2511.04009v1)**

> **作者:** Chenzui Li; Yiming Chen; Xi Wu; Giacinto Barresi; Fei Chen
>
> **备注:** 7 pages, 7 figures, IROS 2025 accepted
>
> **摘要:** This paper introduces an upper limb postural optimization method for enhancing physical ergonomics and force manipulability during bimanual human-robot co-carrying tasks. Existing research typically emphasizes human safety or manipulative efficiency, whereas our proposed method uniquely integrates both aspects to strengthen collaboration across diverse conditions (e.g., different grasping postures of humans, and different shapes of objects). Specifically, the joint angles of a simplified human skeleton model are optimized by minimizing the cost function to prioritize safety and manipulative capability. To guide humans towards the optimized posture, the reference end-effector poses of the robot are generated through a transformation module. A bimanual model predictive impedance controller (MPIC) is proposed for our human-like robot, CURI, to recalibrate the end effector poses through planned trajectories. The proposed method has been validated through various subjects and objects during human-human collaboration (HHC) and human-robot collaboration (HRC). The experimental results demonstrate significant improvement in muscle conditions by comparing the activation of target muscles before and after optimization.
>
---
#### [new 008] Studying the Effect of Explicit Interaction Representations on Learning Scene-level Distributions of Human Trajectories
- **分类: cs.RO**

- **简介: 该论文研究人类轨迹的场景级分布学习，聚焦交互表示方式对预测性能的影响。对比隐式与显式交互建模，发现显式定义交互（如过路口顺序）显著提升模型效果，解决自动驾驶中交互表征不明确的问题。**

- **链接: [http://arxiv.org/pdf/2511.04375v1](http://arxiv.org/pdf/2511.04375v1)**

> **作者:** Anna Mészáros; Javier Alonso-Mora; Jens Kober
>
> **摘要:** Effectively capturing the joint distribution of all agents in a scene is relevant for predicting the true evolution of the scene and in turn providing more accurate information to the decision processes of autonomous vehicles. While new models have been developed for this purpose in recent years, it remains unclear how to best represent the joint distributions particularly from the perspective of the interactions between agents. Thus far there is no clear consensus on how best to represent interactions between agents; whether they should be learned implicitly from data by neural networks, or explicitly modeled using the spatial and temporal relations that are more grounded in human decision-making. This paper aims to study various means of describing interactions within the same network structure and their effect on the final learned joint distributions. Our findings show that more often than not, simply allowing a network to establish interactive connections between agents based on data has a detrimental effect on performance. Instead, having well defined interactions (such as which agent of an agent pair passes first at an intersection) can often bring about a clear boost in performance.
>
---
#### [new 009] An LLM-based Framework for Human-Swarm Teaming Cognition in Disaster Search and Rescue
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种基于LLM的框架，解决灾难搜救中人类操作员因认知负荷过高导致的“意图-动作鸿沟”问题。通过自然交互与LLM实现意图理解、任务分解与 swarm 协同规划，显著提升任务效率与人机协作体验。**

- **链接: [http://arxiv.org/pdf/2511.04042v1](http://arxiv.org/pdf/2511.04042v1)**

> **作者:** Kailun Ji; Xiaoyu Hu; Xinyu Zhang; Jun Chen
>
> **摘要:** Large-scale disaster Search And Rescue (SAR) operations are persistently challenged by complex terrain and disrupted communications. While Unmanned Aerial Vehicle (UAV) swarms offer a promising solution for tasks like wide-area search and supply delivery, yet their effective coordination places a significant cognitive burden on human operators. The core human-machine collaboration bottleneck lies in the ``intention-to-action gap'', which is an error-prone process of translating a high-level rescue objective into a low-level swarm command under high intensity and pressure. To bridge this gap, this study proposes a novel LLM-CRF system that leverages Large Language Models (LLMs) to model and augment human-swarm teaming cognition. The proposed framework initially captures the operator's intention through natural and multi-modal interactions with the device via voice or graphical annotations. It then employs the LLM as a cognitive engine to perform intention comprehension, hierarchical task decomposition, and mission planning for the UAV swarm. This closed-loop framework enables the swarm to act as a proactive partner, providing active feedback in real-time while reducing the need for manual monitoring and control, which considerably advances the efficacy of the SAR task. We evaluate the proposed framework in a simulated SAR scenario. Experimental results demonstrate that, compared to traditional order and command-based interfaces, the proposed LLM-driven approach reduced task completion time by approximately $64.2\%$ and improved task success rate by $7\%$. It also leads to a considerable reduction in subjective cognitive workload, with NASA-TLX scores dropping by $42.9\%$. This work establishes the potential of LLMs to create more intuitive and effective human-swarm collaborations in high-stakes scenarios.
>
---
#### [new 010] GraspView: Active Perception Scoring and Best-View Optimization for Robotic Grasping in Cluttered Environments
- **分类: cs.RO**

- **简介: GraspView提出一种仅用RGB相机的机器人抓取框架，解决复杂场景中遮挡、透明物体和无深度传感器导致的抓取失败问题。通过多视图重建、主动感知选优和尺度校准，实现高鲁棒性抓取。**

- **链接: [http://arxiv.org/pdf/2511.04199v1](http://arxiv.org/pdf/2511.04199v1)**

> **作者:** Shenglin Wang; Mingtong Dai; Jingxuan Su; Lingbo Liu; Chunjie Chen; Xinyu Wu; Liang Lin
>
> **摘要:** Robotic grasping is a fundamental capability for autonomous manipulation, yet remains highly challenging in cluttered environments where occlusion, poor perception quality, and inconsistent 3D reconstructions often lead to unstable or failed grasps. Conventional pipelines have widely relied on RGB-D cameras to provide geometric information, which fail on transparent or glossy objects and degrade at close range. We present GraspView, an RGB-only robotic grasping pipeline that achieves accurate manipulation in cluttered environments without depth sensors. Our framework integrates three key components: (i) global perception scene reconstruction, which provides locally consistent, up-to-scale geometry from a single RGB view and fuses multi-view projections into a coherent global 3D scene; (ii) a render-and-score active perception strategy, which dynamically selects next-best-views to reveal occluded regions; and (iii) an online metric alignment module that calibrates VGGT predictions against robot kinematics to ensure physical scale consistency. Building on these tailor-designed modules, GraspView performs best-view global grasping, fusing multi-view reconstructions and leveraging GraspNet for robust execution. Experiments on diverse tabletop objects demonstrate that GraspView significantly outperforms both RGB-D and single-view RGB baselines, especially under heavy occlusion, near-field sensing, and with transparent objects. These results highlight GraspView as a practical and versatile alternative to RGB-D pipelines, enabling reliable grasping in unstructured real-world environments.
>
---
#### [new 011] Temporal Action Selection for Action Chunking
- **分类: cs.RO**

- **简介: 该论文针对学习示范中的动作分块问题，提出Temporal Action Selector（TAS），通过缓存多时间步动作块并动态选择最优动作，同步提升反应性、决策一致性和运动连贯性，显著提高任务成功率与强化学习效率。**

- **链接: [http://arxiv.org/pdf/2511.04421v1](http://arxiv.org/pdf/2511.04421v1)**

> **作者:** Yueyang Weng; Xiaopeng Zhang; Yongjin Mu; Yingcong Zhu; Yanjie Li; Qi Liu
>
> **摘要:** Action chunking is a widely adopted approach in Learning from Demonstration (LfD). By modeling multi-step action chunks rather than single-step actions, action chunking significantly enhances modeling capabilities for human expert policies. However, the reduced decision frequency restricts the utilization of recent observations, degrading reactivity - particularly evident in the inadequate adaptation to sensor noise and dynamic environmental changes. Existing efforts to address this issue have primarily resorted to trading off reactivity against decision consistency, without achieving both. To address this limitation, we propose a novel algorithm, Temporal Action Selector (TAS), which caches predicted action chunks from multiple timesteps and dynamically selects the optimal action through a lightweight selector network. TAS achieves balanced optimization across three critical dimensions: reactivity, decision consistency, and motion coherence. Experiments across multiple tasks with diverse base policies show that TAS significantly improves success rates - yielding an absolute gain of up to 73.3%. Furthermore, integrating TAS as a base policy with residual reinforcement learning (RL) substantially enhances training efficiency and elevates the performance plateau. Experiments in both simulation and physical robots confirm the method's efficacy.
>
---
#### [new 012] GentleHumanoid: Learning Upper-body Compliance for Contact-rich Human and Object Interaction
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 论文提出GentleHumanoid框架，通过统一弹簧模型将阻抗控制融入全身运动策略，实现人形机器人上肢的自然柔顺交互，解决传统RL策略僵硬、缺乏接触顺应性的问题，提升安全协作能力。**

- **链接: [http://arxiv.org/pdf/2511.04679v1](http://arxiv.org/pdf/2511.04679v1)**

> **作者:** Qingzhou Lu; Yao Feng; Baiyu Shi; Michael Piseno; Zhenan Bao; C. Karen Liu
>
> **备注:** Home page: https://gentle-humanoid.axell.top
>
> **摘要:** Humanoid robots are expected to operate in human-centered environments where safe and natural physical interaction is essential. However, most recent reinforcement learning (RL) policies emphasize rigid tracking and suppress external forces. Existing impedance-augmented approaches are typically restricted to base or end-effector control and focus on resisting extreme forces rather than enabling compliance. We introduce GentleHumanoid, a framework that integrates impedance control into a whole-body motion tracking policy to achieve upper-body compliance. At its core is a unified spring-based formulation that models both resistive contacts (restoring forces when pressing against surfaces) and guiding contacts (pushes or pulls sampled from human motion data). This formulation ensures kinematically consistent forces across the shoulder, elbow, and wrist, while exposing the policy to diverse interaction scenarios. Safety is further supported through task-adjustable force thresholds. We evaluate our approach in both simulation and on the Unitree G1 humanoid across tasks requiring different levels of compliance, including gentle hugging, sit-to-stand assistance, and safe object manipulation. Compared to baselines, our policy consistently reduces peak contact forces while maintaining task success, resulting in smoother and more natural interactions. These results highlight a step toward humanoid robots that can safely and effectively collaborate with humans and handle objects in real-world environments.
>
---
#### [new 013] CBMC-V3: A CNS-inspired Control Framework Towards Manipulation Agility with SNN
- **分类: cs.RO**

- **简介: 论文提出一种受中枢神经系统启发的SNN控制框架CBMC-V3，解决机器人臂在动态环境中敏捷操控难题，通过五模块分级架构实现反馈与前馈协同控制，显著超越传统工业控制方法。**

- **链接: [http://arxiv.org/pdf/2511.04109v1](http://arxiv.org/pdf/2511.04109v1)**

> **作者:** Yanbo Pang; Qingkai Li; Mingguo Zhao
>
> **摘要:** As robotic arm applications extend beyond industrial settings into healthcare, service, and daily life, existing control algorithms struggle to achieve the agile manipulation required for complex environments with dynamic trajectories, unpredictable interactions, and diverse objects. This paper presents a biomimetic control framework based on Spiking Neural Networks (SNN), inspired by the human Central Nervous System (CNS), to achieve agile control in such environments. The proposed framework features five control modules (cerebral cortex, cerebellum, thalamus, brainstem, spinal cord), three hierarchical control levels (first-order, second-order, third-order), and two information pathways (ascending, descending). Each module is fully implemented using SNN. The spinal cord module uses spike encoding and Leaky Integrate-and-Fire (LIF) neurons for feedback control. The brainstem module employs a network of LIF and non-spiking LIF neurons to dynamically adjust spinal cord parameters via reinforcement learning. The thalamus module similarly adjusts the cerebellum's torque outputs. The cerebellum module uses a recurrent SNN to learn the robotic arm's dynamics through regression, providing feedforward gravity compensation torques. The framework is validated both in simulation and on real-world robotic arm platform under various loads and trajectories. Results demonstrate that our method outperforms the industrial-grade position control in manipulation agility.
>
---
#### [new 014] SAFe-Copilot: Unified Shared Autonomy Framework
- **分类: cs.RO**

- **简介: 论文提出SAFe-Copilot，一种基于视觉语言模型的统一共享自动驾驶框架，通过高层语义意图推理协调人机决策，解决传统方法仅优化轨迹而忽视驾驶意图的问题，显著提升安全性与人机一致性。**

- **链接: [http://arxiv.org/pdf/2511.04664v1](http://arxiv.org/pdf/2511.04664v1)**

> **作者:** Phat Nguyen; Erfan Aasi; Shiva Sreeram; Guy Rosman; Andrew Silva; Sertac Karaman; Daniela Rus
>
> **摘要:** Autonomous driving systems remain brittle in rare, ambiguous, and out-of-distribution scenarios, where human driver succeed through contextual reasoning. Shared autonomy has emerged as a promising approach to mitigate such failures by incorporating human input when autonomy is uncertain. However, most existing methods restrict arbitration to low-level trajectories, which represent only geometric paths and therefore fail to preserve the underlying driving intent. We propose a unified shared autonomy framework that integrates human input and autonomous planners at a higher level of abstraction. Our method leverages Vision Language Models (VLMs) to infer driver intent from multi-modal cues -- such as driver actions and environmental context -- and to synthesize coherent strategies that mediate between human and autonomous control. We first study the framework in a mock-human setting, where it achieves perfect recall alongside high accuracy and precision. A human-subject survey further shows strong alignment, with participants agreeing with arbitration outcomes in 92% of cases. Finally, evaluation on the Bench2Drive benchmark demonstrates a substantial reduction in collision rate and improvement in overall performance compared to pure autonomy. Arbitration at the level of semantic, language-based representations emerges as a design principle for shared autonomy, enabling systems to exercise common-sense reasoning and maintain continuity with human intent.
>
---
#### [new 015] MacroNav: Multi-Task Context Representation Learning Enables Efficient Navigation in Unknown Environments
- **分类: cs.RO**

- **简介: MacroNav面向未知环境自主导航，解决上下文表征与效率平衡难题，提出多任务自监督编码器与图推理强化学习策略，实现高效高精度导航，显著提升SR和SPL指标。**

- **链接: [http://arxiv.org/pdf/2511.04320v1](http://arxiv.org/pdf/2511.04320v1)**

> **作者:** Kuankuan Sima; Longbin Tang; Haozhe Ma; Lin Zhao
>
> **摘要:** Autonomous navigation in unknown environments requires compact yet expressive spatial understanding under partial observability to support high-level decision making. Existing approaches struggle to balance rich contextual representation with navigation efficiency. We present MacroNav, a learning-based navigation framework featuring two key components: (1) a lightweight context encoder trained via multi-task self-supervised learning to capture multi-scale, navigation-centric spatial representations; and (2) a reinforcement learning policy that seamlessly integrates these representations with graph-based reasoning for efficient action selection. Extensive experiments demonstrate the context encoder's efficient and robust environmental understanding. Real-world deployments further validate MacroNav's effectiveness, yielding significant gains over state-of-the-art navigation methods in both Success Rate (SR) and Success weighted by Path Length (SPL), while maintaining low computational cost. Code will be released upon acceptance.
>
---
#### [new 016] Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文面向人形机器人足球任务，解决感知-动作解耦导致的响应延迟与行为不协调问题，提出一种基于强化学习的统一控制器，通过视觉感知与运动控制直接融合，并引入虚拟感知系统提升鲁棒性，实现高效反应的足球技能。**

- **链接: [http://arxiv.org/pdf/2511.03996v1](http://arxiv.org/pdf/2511.03996v1)**

> **作者:** Yushi Wang; Changsheng Luo; Penghui Chen; Jianran Liu; Weijian Sun; Tong Guo; Kechang Yang; Biao Hu; Yangang Zhang; Mingguo Zhao
>
> **备注:** Project page: https://humanoid-kick.github.io
>
> **摘要:** Humanoid soccer poses a representative challenge for embodied intelligence, requiring robots to operate within a tightly coupled perception-action loop. However, existing systems typically rely on decoupled modules, resulting in delayed responses and incoherent behaviors in dynamic environments, while real-world perceptual limitations further exacerbate these issues. In this work, we present a unified reinforcement learning-based controller that enables humanoid robots to acquire reactive soccer skills through the direct integration of visual perception and motion control. Our approach extends Adversarial Motion Priors to perceptual settings in real-world dynamic environments, bridging motion imitation and visually grounded dynamic control. We introduce an encoder-decoder architecture combined with a virtual perception system that models real-world visual characteristics, allowing the policy to recover privileged states from imperfect observations and establish active coordination between perception and action. The resulting controller demonstrates strong reactivity, consistently executing coherent and robust soccer behaviors across various scenarios, including real RoboCup matches.
>
---
#### [new 017] X-Diffusion: Training Diffusion Policies on Cross-Embodiment Human Demonstrations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: X-Diffusion提出一种跨体态扩散策略，利用人类视频数据训练机器人策略，通过加噪消除人-机动作执行差异，保留高层任务指导，提升机器人操作成功率，解决人类数据直接迁移导致的物理不可行问题。**

- **链接: [http://arxiv.org/pdf/2511.04671v1](http://arxiv.org/pdf/2511.04671v1)**

> **作者:** Maximus A. Pace; Prithwish Dan; Chuanruo Ning; Atiksh Bhardwaj; Audrey Du; Edward W. Duan; Wei-Chiu Ma; Kushal Kedia
>
> **摘要:** Human videos can be recorded quickly and at scale, making them an appealing source of training data for robot learning. However, humans and robots differ fundamentally in embodiment, resulting in mismatched action execution. Direct kinematic retargeting of human hand motion can therefore produce actions that are physically infeasible for robots. Despite these low-level differences, human demonstrations provide valuable motion cues about how to manipulate and interact with objects. Our key idea is to exploit the forward diffusion process: as noise is added to actions, low-level execution differences fade while high-level task guidance is preserved. We present X-Diffusion, a principled framework for training diffusion policies that maximally leverages human data without learning dynamically infeasible motions. X-Diffusion first trains a classifier to predict whether a noisy action is executed by a human or robot. Then, a human action is incorporated into policy training only after adding sufficient noise such that the classifier cannot discern its embodiment. Actions consistent with robot execution supervise fine-grained denoising at low noise levels, while mismatched human actions provide only coarse guidance at higher noise levels. Our experiments show that naive co-training under execution mismatches degrades policy performance, while X-Diffusion consistently improves it. Across five manipulation tasks, X-Diffusion achieves a 16% higher average success rate than the best baseline. The project website is available at https://portal-cornell.github.io/X-Diffusion/.
>
---
#### [new 018] ForeRobo: Unlocking Infinite Simulation Data for 3D Goal-driven Robotic Manipulation
- **分类: cs.RO**

- **简介: ForeRobo提出一种生成式机器人框架，通过模拟自动生成目标状态数据，训练ForeFormer模型预测3D目标位姿，结合经典控制实现高效、可解释的零样本仿真实现迁移，显著提升机器人操作泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.04381v1](http://arxiv.org/pdf/2511.04381v1)**

> **作者:** Dexin wang; Faliang Chang; Chunsheng Liu
>
> **摘要:** Efficiently leveraging simulation to acquire advanced manipulation skills is both challenging and highly significant. We introduce \textit{ForeRobo}, a generative robotic agent that utilizes generative simulations to autonomously acquire manipulation skills driven by envisioned goal states. Instead of directly learning low-level policies, we advocate integrating generative paradigms with classical control. Our approach equips a robotic agent with a self-guided \textit{propose-generate-learn-actuate} cycle. The agent first proposes the skills to be acquired and constructs the corresponding simulation environments; it then configures objects into appropriate arrangements to generate skill-consistent goal states (\textit{ForeGen}). Subsequently, the virtually infinite data produced by ForeGen are used to train the proposed state generation model (\textit{ForeFormer}), which establishes point-wise correspondences by predicting the 3D goal position of every point in the current state, based on the scene state and task instructions. Finally, classical control algorithms are employed to drive the robot in real-world environments to execute actions based on the envisioned goal states. Compared with end-to-end policy learning methods, ForeFormer offers superior interpretability and execution efficiency. We train and benchmark ForeFormer across a variety of rigid-body and articulated-object manipulation tasks, and observe an average improvement of 56.32\% over the state-of-the-art state generation models, demonstrating strong generality across different manipulation patterns. Moreover, in real-world evaluations involving more than 20 robotic tasks, ForeRobo achieves zero-shot sim-to-real transfer and exhibits remarkable generalization capabilities, attaining an average success rate of 79.28\%.
>
---
#### [new 019] GraSP-VLA: Graph-based Symbolic Action Representation for Long-Horizon Planning with VLA Policies
- **分类: cs.RO; cs.CV**

- **简介: 论文提出GraSP-VLA，一种神经符号框架，通过连续场景图将人类示范转化为符号动作表示，以协调低层VLA策略，解决视觉语言模型在长程任务中规划能力不足与符号方法泛化性差的问题。**

- **链接: [http://arxiv.org/pdf/2511.04357v1](http://arxiv.org/pdf/2511.04357v1)**

> **作者:** Maëlic Neau; Zoe Falomir; Paulo E. Santos; Anne-Gwenn Bosser; Cédric Buche
>
> **摘要:** Deploying autonomous robots that can learn new skills from demonstrations is an important challenge of modern robotics. Existing solutions often apply end-to-end imitation learning with Vision-Language Action (VLA) models or symbolic approaches with Action Model Learning (AML). On the one hand, current VLA models are limited by the lack of high-level symbolic planning, which hinders their abilities in long-horizon tasks. On the other hand, symbolic approaches in AML lack generalization and scalability perspectives. In this paper we present a new neuro-symbolic approach, GraSP-VLA, a framework that uses a Continuous Scene Graph representation to generate a symbolic representation of human demonstrations. This representation is used to generate new planning domains during inference and serves as an orchestrator for low-level VLA policies, scaling up the number of actions that can be reproduced in a row. Our results show that GraSP-VLA is effective for modeling symbolic representations on the task of automatic planning domain generation from observations. In addition, results on real-world experiments show the potential of our Continuous Scene Graph representation to orchestrate low-level VLA policies in long-horizon tasks.
>
---
#### [new 020] PUL-SLAM: Path-Uncertainty Co-Optimization with Lightweight Stagnation Detection for Efficient Robotic Exploration
- **分类: cs.RO**

- **简介: 该论文提出PUL-SLAM，用于机器人自主探索任务，解决传统方法探索慢、路径低效问题。通过深度强化学习协同优化路径与地图不确定性，并引入轻量级停滞检测，显著提升探索效率与实用性。**

- **链接: [http://arxiv.org/pdf/2511.04180v1](http://arxiv.org/pdf/2511.04180v1)**

> **作者:** Yizhen Yin; Dapeng Feng; Hongbo Chen; Yuhua Qi
>
> **摘要:** Existing Active SLAM methodologies face issues such as slow exploration speed and suboptimal paths. To address these limitations, we propose a hybrid framework combining a Path-Uncertainty Co-Optimization Deep Reinforcement Learning framework and a Lightweight Stagnation Detection mechanism. The Path-Uncertainty Co-Optimization framework jointly optimizes travel distance and map uncertainty through a dual-objective reward function, balancing exploration and exploitation. The Lightweight Stagnation Detection reduces redundant exploration through Lidar Static Anomaly Detection and Map Update Stagnation Detection, terminating episodes on low expansion rates. Experimental results show that compared with the frontier-based method and RRT method, our approach shortens exploration time by up to 65% and reduces path distance by up to 42%, significantly improving exploration efficiency in complex environments while maintaining reliable map completeness. Ablation studies confirm that the collaborative mechanism accelerates training convergence. Empirical validation on a physical robotic platform demonstrates the algorithm's practical applicability and its successful transferability from simulation to real-world environments.
>
---
#### [new 021] Dynamic Shape Control of Soft Robots Enabled by Data-Driven Model Reduction
- **分类: cs.RO**

- **简介: 该论文研究软体机器人动态形变控制问题，提出基于数据驱动模型降阶的线性建模方法，对比三种算法在预测控制中的表现，发现Lagrangian算子推断（LOpInf）在轨迹跟踪任务中误差最低。**

- **链接: [http://arxiv.org/pdf/2511.03931v1](http://arxiv.org/pdf/2511.03931v1)**

> **作者:** Iman Adibnazari; Harsh Sharma; Myungsun Park; Jacobo Cervera-Torralba; Boris Kramer; Michael T. Tolley
>
> **备注:** 20 Pages, 8 Figures
>
> **摘要:** Soft robots have shown immense promise in settings where they can leverage dynamic control of their entire bodies. However, effective dynamic shape control requires a controller that accounts for the robot's high-dimensional dynamics--a challenge exacerbated by a lack of general-purpose tools for modeling soft robots amenably for control. In this work, we conduct a comparative study of data-driven model reduction techniques for generating linear models amendable to dynamic shape control. We focus on three methods--the eigensystem realization algorithm, dynamic mode decomposition with control, and the Lagrangian operator inference (LOpInf) method. Using each class of model, we explored their efficacy in model predictive control policies for the dynamic shape control of a simulated eel-inspired soft robot in three experiments: 1) tracking simulated reference trajectories guaranteed to be feasible, 2) tracking reference trajectories generated from a biological model of eel kinematics, and 3) tracking reference trajectories generated by a reduced-scale physical analog. In all experiments, the LOpInf-based policies generated lower tracking errors than policies based on other models.
>
---
#### [new 022] Investigating Robot Control Policy Learning for Autonomous X-ray-guided Spine Procedures
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究基于模仿学习的机器人控制策略，用于X光引导的脊柱穿刺手术。通过构建高仿真模拟环境与数据集，训练视觉驱动的开环控制策略，实现精准 Cannula 定位，验证了模拟训练策略在真实X光下的泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.03882v1](http://arxiv.org/pdf/2511.03882v1)**

> **作者:** Florence Klitzner; Blanca Inigo; Benjamin D. Killeen; Lalithkumar Seenivasan; Michelle Song; Axel Krieger; Mathias Unberath
>
> **摘要:** Imitation learning-based robot control policies are enjoying renewed interest in video-based robotics. However, it remains unclear whether this approach applies to X-ray-guided procedures, such as spine instrumentation. This is because interpretation of multi-view X-rays is complex. We examine opportunities and challenges for imitation policy learning in bi-plane-guided cannula insertion. We develop an in silico sandbox for scalable, automated simulation of X-ray-guided spine procedures with a high degree of realism. We curate a dataset of correct trajectories and corresponding bi-planar X-ray sequences that emulate the stepwise alignment of providers. We then train imitation learning policies for planning and open-loop control that iteratively align a cannula solely based on visual information. This precisely controlled setup offers insights into limitations and capabilities of this method. Our policy succeeded on the first attempt in 68.5% of cases, maintaining safe intra-pedicular trajectories across diverse vertebral levels. The policy generalized to complex anatomy, including fractures, and remained robust to varied initializations. Rollouts on real bi-planar X-rays further suggest that the model can produce plausible trajectories, despite training exclusively in simulation. While these preliminary results are promising, we also identify limitations, especially in entry point precision. Full closed-look control will require additional considerations around how to provide sufficiently frequent feedback. With more robust priors and domain knowledge, such models may provide a foundation for future efforts toward lightweight and CT-free robotic intra-operative spinal navigation.
>
---
#### [new 023] HACI: A Haptic-Audio Code Interface to Improve Educational Outcomes for Visually Impaired Introductory Programming Students
- **分类: cs.HC; cs.RO**

- **简介: 论文提出HACI，一种融合触觉与音频反馈的编程教学工具，旨在帮助视障学生非视觉化理解与调试代码，解决视觉缺失导致的编程学习障碍，通过可穿戴设备与Web应用实现交互式编程教育。**

- **链接: [http://arxiv.org/pdf/2511.03733v1](http://arxiv.org/pdf/2511.03733v1)**

> **作者:** Pratham Gandhi
>
> **摘要:** This thesis introduces the Haptic-Audio Code Interface (HACI), an educational tool designed to enhance programming education for visually impaired (VI) students by integrating haptic and audio feedback to compensate for the absence of visual cues. HACI consists of a non-resource-intensive web application supporting JavaScript program development, execution, and debugging, connected via a cable to an Arduino-powered glove with six integrated haptic motors to provide physical feedback to VI programmers. Motivated by the need to provide equitable educational opportunities in computer science, HACI aims to improve non-visual code navigation, comprehension, summarizing, editing, and debugging for students with visual impairments while minimizing cognitive load. This work details HACI's design principles, technical implementation, and a preliminary evaluation through a pilot study conducted with undergraduate Computer Science students. Findings indicate that HACI aids in the non-visual navigation and understanding of programming constructs, although challenges remain in refining feedback mechanisms to ensure consistency and reliability, as well as supplementing the current functionality with a more feature-reach and customizable accessible learning experience which will allow visually impaired students to fully utilize interleaved haptic and audio feedback. The study underscores the transformative potential of haptic and audio feedback in educational practices for the visually impaired, setting a foundation for future research and development in accessible programming education. This thesis contributes to the field of accessible technology by demonstrating how tactile and auditory feedback can be effectively integrated into educational tools, thereby broadening accessibility in STEM education.
>
---
#### [new 024] BoRe-Depth: Self-supervised Monocular Depth Estimation with Boundary Refinement for Embedded Systems
- **分类: cs.CV; cs.RO**

- **简介: 论文提出BoRe-Depth，一种轻量级单目深度估计算法，面向嵌入式系统，通过增强特征融合与语义引导提升边界精度，在Jetson Orin上达50.7 FPS，显著优于现有轻量模型。**

- **链接: [http://arxiv.org/pdf/2511.04388v1](http://arxiv.org/pdf/2511.04388v1)**

> **作者:** Chang Liu; Juan Li; Sheng Zhang; Chang Liu; Jie Li; Xu Zhang
>
> **备注:** 8 pages, 5 figures, published to IROS 2025
>
> **摘要:** Depth estimation is one of the key technologies for realizing 3D perception in unmanned systems. Monocular depth estimation has been widely researched because of its low-cost advantage, but the existing methods face the challenges of poor depth estimation performance and blurred object boundaries on embedded systems. In this paper, we propose a novel monocular depth estimation model, BoRe-Depth, which contains only 8.7M parameters. It can accurately estimate depth maps on embedded systems and significantly improves boundary quality. Firstly, we design an Enhanced Feature Adaptive Fusion Module (EFAF) which adaptively fuses depth features to enhance boundary detail representation. Secondly, we integrate semantic knowledge into the encoder to improve the object recognition and boundary perception capabilities. Finally, BoRe-Depth is deployed on NVIDIA Jetson Orin, and runs efficiently at 50.7 FPS. We demonstrate that the proposed model significantly outperforms previous lightweight models on multiple challenging datasets, and we provide detailed ablation studies for the proposed methods. The code is available at https://github.com/liangxiansheng093/BoRe-Depth.
>
---
## 更新

#### [replaced 001] Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11234v4](http://arxiv.org/pdf/2506.11234v4)**

> **作者:** Luke Rowe; Rodrigue de Schaetzen; Roger Girgis; Christopher Pal; Liam Paull
>
> **摘要:** Maintaining good driving behavior in out-of-distribution scenarios remains a critical challenge in autonomous driving. A promising direction is to leverage the generalist knowledge and reasoning capabilities of large-language models by treating unusual driving scenarios as a logical reasoning task. In this work, we present Poutine, a method that uses an off-the-shelf 3B-parameter vision-language model (VLM) - without any additional components - to achieve robust end-to-end autonomous driving via a simple and scalable training recipe. To learn strong base driving capabilities, we first train Poutine-Base using self-supervised next-token prediction over vision, language, and trajectory (VLT) tokens, leveraging both nominal and long-tail driving data. In the second stage, we fine-tune Poutine-Base using Group Relative Policy Optimization (GRPO) with a small set of human preference-labeled examples. We evaluated our approach on the Waymo end-to-end driving benchmark curated for long-tail scenarios. The final Poutine model achieves an RFS of 7.99 on the test set, placing 1st in the 2025 Waymo Vision-Based End-to-End Driving Challenge by a significant margin. Our results suggest that handcrafted tokenizers or custom architectural components added to base VLMs in prior work are not necessary to achieve strong driving performance. Instead, this work highlights the potential of scalable VLT pretraining combined with lightweight RL fine-tuning to enable robust and generalizable autonomous driving.
>
---
#### [replaced 002] Scalable Multi-Robot Motion Planning Using Workspace Guidance-Informed Hypergraphs
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2311.10176v4](http://arxiv.org/pdf/2311.10176v4)**

> **作者:** Courtney McBeth; James Motes; Isaac Ngui; Marco Morales; Nancy M. Amato
>
> **备注:** This work has been submitted for review
>
> **摘要:** In this work, we propose a method for multiple mobile robot motion planning that efficiently plans for robot teams up to 128 robots (an order of magnitude larger than existing state-of-the-art methods) in congested settings with narrow passages in the environment. We achieve this improvement in scalability by extending the state-of-the-art Decomposable State Space Hypergraph (DaSH) multi-robot planning framework to support mobile robot motion planning in congested environments. This is a problem that DaSH cannot be directly applied to because it lacks a highly structured, easily discretizable task space and features kinodynamic constraints. We accomplish this by exploiting knowledge about the workspace topology to limit exploration of the planning space and through modifying DaSH's conflict resolution scheme. This guidance captures when coordination between robots is necessary, allowing us to decompose the intractably large multi-robot search space while limiting risk of inter-robot conflicts by composing relevant robot groups together while planning.
>
---
#### [replaced 003] SLAM&Render: A Benchmark for the Intersection Between Neural Rendering, Gaussian Splatting and SLAM
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.13713v3](http://arxiv.org/pdf/2504.13713v3)**

> **作者:** Samuel Cerezo; Gaetano Meli; Tomás Berriel Martins; Kirill Safronov; Javier Civera
>
> **备注:** 9 pages, 8 figures, submitted to The International Journal of Robotics Research (IJRR)
>
> **摘要:** Models and methods originally developed for Novel View Synthesis and Scene Rendering, such as Neural Radiance Fields (NeRF) and Gaussian Splatting, are increasingly being adopted as representations in Simultaneous Localization and Mapping (SLAM). However, existing datasets fail to include the specific challenges of both fields, such as sequential operations and, in many settings, multi-modality in SLAM or generalization across viewpoints and illumination conditions in neural rendering. Additionally, the data are often collected using sensors which are handheld or mounted on drones or mobile robots, which complicates the accurate reproduction of sensor motions. To bridge these gaps, we introduce SLAM&Render, a novel dataset designed to benchmark methods in the intersection between SLAM, Novel View Rendering and Gaussian Splatting. Recorded with a robot manipulator, it uniquely includes 40 sequences with time-synchronized RGB-D images, IMU readings, robot kinematic data, and ground-truth pose streams. By releasing robot kinematic data, the dataset also enables the assessment of recent integrations of SLAM paradigms within robotic applications. The dataset features five setups with consumer and industrial objects under four controlled lighting conditions, each with separate training and test trajectories. All sequences are static with different levels of object rearrangements and occlusions. Our experimental results, obtained with several baselines from the literature, validate SLAM&Render as a relevant benchmark for this emerging research area.
>
---
#### [replaced 004] Swarmodroid & AMPy: Reconfigurable Bristle-Bots and Software Package for Robotic Active Matter Studies
- **分类: cond-mat.soft; cond-mat.stat-mech; cs.RO**

- **链接: [http://arxiv.org/pdf/2305.13510v2](http://arxiv.org/pdf/2305.13510v2)**

> **作者:** Alexey A. Dmitriev; Vadim A. Porvatov; Alina D. Rozenblit; Mikhail K. Buzakov; Anastasia A. Molodtsova; Daria V. Sennikova; Vyacheslav A. Smirnov; Oleg I. Burmistrov; Timur I. Karimov; Ekaterina M. Puhtina; Nikita A. Olekhno
>
> **备注:** 17 pages, 6 figures, 1 table + Supplementary Information. Comments are welcome
>
> **摘要:** Large assemblies of extremely simple robots capable only of basic motion activities (like propelling forward or self-rotating) are often applied to study swarming behavior or implement various phenomena characteristic of active matter composed of non-equilibrium particles that convert their energy to a directed motion. As a result, a great abundance of compact swarm robots have been developed. The simplest are bristle-bots that self-propel via converting their vibration with the help of elastic bristles. However, many platforms are optimized for a certain class of studies, are not always made open-source, or have limited customization potential. To address these issues, we develop the open-source Swarmodroid 1.0 platform based on bristle-bots with reconfigurable 3D printed bodies and simple electronics that possess external control of motion velocity and demonstrate basic capabilities of trajectory adjustment. Then, we perform a detailed analysis of individual Swarmodroids' motion characteristics and their kinematics. In addition, we introduce the AMPy software package in Python that features OpenCV-based extraction of robotic swarm kinematics accompanied by the evaluation of key physical quantities describing the collective dynamics. Finally, we discuss potential applications as well as further directions for fundamental studies and Swarmodroid 1.0 platform development.
>
---
#### [replaced 005] When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2511.00783v2](http://arxiv.org/pdf/2511.00783v2)**

> **作者:** Jingzehua Xu; Weihang Zhang; Yangyang Li; Hongmiaoyi Zhang; Guanwen Xie; Jiwei Tang; Shuai Zhang; Yi Li
>
> **备注:** This paper has been submitted to IEEE Transactions on Mobile Computing. Jingzehua Xu, Weihang Zhang, and Yangyang Li contributed equally to this work and are recognized as the co-first authors of the paper
>
> **摘要:** Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.
>
---
#### [replaced 006] Team Xiaomi EV-AD VLA: Caption-Guided Retrieval System for Cross-Modal Drone Navigation -- Technical Report for IROS 2025 RoboSense Challenge Track 4
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.02728v2](http://arxiv.org/pdf/2510.02728v2)**

> **作者:** Lingfeng Zhang; Erjia Xiao; Yuchen Zhang; Haoxiang Fu; Ruibin Hu; Yanbiao Ma; Wenbo Ding; Long Chen; Hangjun Ye; Xiaoshuai Hao
>
> **摘要:** Cross-modal drone navigation remains a challenging task in robotics, requiring efficient retrieval of relevant images from large-scale databases based on natural language descriptions. The RoboSense 2025 Track 4 challenge addresses this challenge, focusing on robust, natural language-guided cross-view image retrieval across multiple platforms (drones, satellites, and ground cameras). Current baseline methods, while effective for initial retrieval, often struggle to achieve fine-grained semantic matching between text queries and visual content, especially in complex aerial scenes. To address this challenge, we propose a two-stage retrieval refinement method: Caption-Guided Retrieval System (CGRS) that enhances the baseline coarse ranking through intelligent reranking. Our method first leverages a baseline model to obtain an initial coarse ranking of the top 20 most relevant images for each query. We then use Vision-Language-Model (VLM) to generate detailed captions for these candidate images, capturing rich semantic descriptions of their visual content. These generated captions are then used in a multimodal similarity computation framework to perform fine-grained reranking of the original text query, effectively building a semantic bridge between the visual content and natural language descriptions. Our approach significantly improves upon the baseline, achieving a consistent 5\% improvement across all key metrics (Recall@1, Recall@5, and Recall@10). Our approach win TOP-2 in the challenge, demonstrating the practical value of our semantic refinement strategy in real-world robotic navigation scenarios.
>
---
#### [replaced 007] XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.00097v2](http://arxiv.org/pdf/2508.00097v2)**

> **作者:** Zhigen Zhao; Liuchuan Yu; Ke Jing; Ning Yang
>
> **备注:** 6 pages, 6 figures, accepted at The 2026 IEEE/SICE International Symposium on System Integration, project link: http://xr-robotics.github.io/
>
> **摘要:** The rapid advancement of Vision-Language-Action models has created an urgent need for large-scale, high-quality robot demonstration datasets. Although teleoperation is the predominant method for data collection, current approaches suffer from limited scalability, complex setup procedures, and suboptimal data quality. This paper presents XRoboToolkit, a cross-platform framework for extended reality based robot teleoperation built on the OpenXR standard. The system features low-latency stereoscopic visual feedback, optimization-based inverse kinematics, and support for diverse tracking modalities including head, controller, hand, and auxiliary motion trackers. XRoboToolkit's modular architecture enables seamless integration across robotic platforms and simulation environments, spanning precision manipulators, mobile robots, and dexterous hands. We demonstrate the framework's effectiveness through precision manipulation tasks and validate data quality by training VLA models that exhibit robust autonomous performance.
>
---
#### [replaced 008] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.11321v2](http://arxiv.org/pdf/2510.11321v2)**

> **作者:** Ruizhe Liu; Pei Zhou; Qian Luo; Li Sun; Jun Cen; Yibing Song; Yanchao Yang
>
> **备注:** Accepted at 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Effective generalization in robotic manipulation requires representations that capture invariant patterns of interaction across environments and tasks. We present a self-supervised framework for learning hierarchical manipulation concepts that encode these invariant patterns through cross-modal sensory correlations and multi-level temporal abstractions without requiring human annotation. Our approach combines a cross-modal correlation network that identifies persistent patterns across sensory modalities with a multi-horizon predictor that organizes representations hierarchically across temporal scales. Manipulation concepts learned through this dual structure enable policies to focus on transferable relational patterns while maintaining awareness of both immediate actions and longer-term goals. Empirical evaluation across simulated benchmarks and real-world deployments demonstrates significant performance improvements with our concept-enhanced policies. Analysis reveals that the learned concepts resemble human-interpretable manipulation primitives despite receiving no semantic supervision. This work advances both the understanding of representation learning for manipulation and provides a practical approach to enhancing robotic performance in complex scenarios.
>
---
#### [replaced 009] OmniVLA: Physically-Grounded Multimodal VLA with Unified Multi-Sensor Perception for Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2511.01210v2](http://arxiv.org/pdf/2511.01210v2)**

> **作者:** Heyu Guo; Shanmu Wang; Ruichun Ma; Shiqi Jiang; Yasaman Ghasempour; Omid Abari; Baining Guo; Lili Qiu
>
> **摘要:** Vision-language-action (VLA) models have shown strong generalization for robotic action prediction through large-scale vision-language pretraining. However, most existing models rely solely on RGB cameras, limiting their perception and, consequently, manipulation capabilities. We present OmniVLA, an omni-modality VLA model that integrates novel sensing modalities for physically-grounded spatial intelligence beyond RGB perception. The core of our approach is the sensor-masked image, a unified representation that overlays spatially grounded and physically meaningful masks onto the RGB images, derived from sensors including an infrared camera, a mmWave radar, and a microphone array. This image-native unification keeps sensor input close to RGB statistics to facilitate training, provides a uniform interface across sensor hardware, and enables data-efficient learning with lightweight per-sensor projectors. Built on this, we present a multisensory vision-language-action model architecture and train the model based on an RGB-pretrained VLA backbone. We evaluate OmniVLA on challenging real-world tasks where sensor-modality perception guides the robotic manipulation. OmniVLA achieves an average task success rate of 84%, significantly outperforms both RGB-only and raw-sensor-input baseline models by 59% and 28% respectively, meanwhile showing higher learning efficiency and stronger generalization capability.
>
---
#### [replaced 010] Application Management in C-ITS: Orchestrating Demand-Driven Deployments and Reconfigurations
- **分类: cs.RO; cs.MA; cs.SE**

- **链接: [http://arxiv.org/pdf/2509.18793v2](http://arxiv.org/pdf/2509.18793v2)**

> **作者:** Lukas Zanger; Bastian Lampe; Lennart Reiher; Lutz Eckstein
>
> **备注:** 7 pages, 2 figures, 2 tables; Accepted to be published as part of the 2025 IEEE International Conference on Intelligent Transportation Systems (ITSC 2025), Gold Coast, Australia, November 18-21, 2025
>
> **摘要:** Vehicles are becoming increasingly automated and interconnected, enabling the formation of cooperative intelligent transport systems (C-ITS) and the use of offboard services. As a result, cloud-native techniques, such as microservices and container orchestration, play an increasingly important role in their operation. However, orchestrating applications in a large-scale C-ITS poses unique challenges due to the dynamic nature of the environment and the need for efficient resource utilization. In this paper, we present a demand-driven application management approach that leverages cloud-native techniques - specifically Kubernetes - to address these challenges. Taking into account the demands originating from different entities within the C-ITS, the approach enables the automation of processes, such as deployment, reconfiguration, update, upgrade, and scaling of microservices. Executing these processes on demand can, for example, reduce computing resource consumption and network traffic. A demand may include a request for provisioning an external supporting service, such as a collective environment model. The approach handles changing and new demands by dynamically reconciling them through our proposed application management framework built on Kubernetes and the Robot Operating System (ROS 2). We demonstrate the operation of our framework in the C-ITS use case of collective environment perception and make the source code of the prototypical framework publicly available at https://github.com/ika-rwth-aachen/application_manager.
>
---
#### [replaced 011] MLP-SLAM: Multilayer Perceptron-Based Simultaneous Localization and Mapping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.10669v2](http://arxiv.org/pdf/2410.10669v2)**

> **作者:** Taozhe Li; Wei Sun
>
> **备注:** Dynamic SLAM
>
> **摘要:** The Visual Simultaneous Localization and Mapping (V-SLAM) system has seen significant development in recent years, demonstrating high precision in environments with limited dynamic objects. However, their performance significantly deteriorates when deployed in settings with a higher presence of movable objects, such as environments with pedestrians, cars, and buses, which are common in outdoor scenes. To address this issue, we propose a Multilayer Perceptron (MLP)-based real-time stereo SLAM system that leverages complete geometry information to avoid information loss. Moreover, there is currently no publicly available dataset for directly evaluating the effectiveness of dynamic and static feature classification methods, and to bridge this gap, we have created a publicly available dataset containing over 50,000 feature points. Experimental results demonstrate that our MLP-based dynamic and static feature point discriminator has achieved superior performance compared to other methods on this dataset. Furthermore, the MLP-based real-time stereo SLAM system has shown the highest average precision and fastest speed on the outdoor KITTI tracking datasets compared to other dynamic SLAM systems.The open-source code and datasets are available at https://github.com/TaozheLi/MLP-SLAM.
>
---
#### [replaced 012] A Framework for Human-Reason-Aligned Trajectory Evaluation in Automated Vehicles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.23324v2](http://arxiv.org/pdf/2507.23324v2)**

> **作者:** Lucas Elbert Suryana; Saeed Rahmani; Simeon Craig Calvert; Arkady Zgonnikov; Bart van Arem
>
> **备注:** This version incorporates revisions based on peer-review feedback from a new submission. The work has been accepted and is being prepared for publication
>
> **摘要:** One major challenge for the adoption and acceptance of automated vehicles (AVs) is ensuring that they can make sound decisions in everyday situations that involve ethical tension. Much attention has focused on rare, high-stakes dilemmas such as trolley problems. Yet similar conflicts arise in routine driving when human considerations, such as legality, efficiency, and comfort, come into conflict. Current AV planning systems typically rely on rigid rules, which struggle to balance these competing considerations and often lead to behaviour that misaligns with human expectations. This paper introduces a reasons-based trajectory evaluation framework that operationalises the tracking condition of Meaningful Human Control (MHC). The framework represents human agents reasons (e.g., regulatory compliance) as quantifiable functions and evaluates how well candidate trajectories align with them. It assigns adjustable weights to agent priorities and includes a balance function to discourage excluding any agent. To demonstrate the approach, we use a real-world-inspired overtaking scenario, which highlights tensions between compliance, efficiency, and comfort. Our results show that different trajectories emerge as preferable depending on how agents reasons are weighted, and small shifts in priorities can lead to discrete changes in the selected action. This demonstrates that everyday ethical decisions in AV driving are highly sensitive to the weights assigned to the reasons of different human agents.
>
---
#### [replaced 013] Action Deviation-Aware Inference for Low-Latency Wireless Robots
- **分类: cs.RO; cs.DC**

- **链接: [http://arxiv.org/pdf/2510.02851v2](http://arxiv.org/pdf/2510.02851v2)**

> **作者:** Jeyoung Park; Yeonsub Lim; Seungeun Oh; Jihong Park; Jinho Choi; Seong-Lyun Kim
>
> **摘要:** To support latency-sensitive AI applications ranging from autonomous driving to industrial robot manipulation, 6G envisions distributed ML with computational resources in mobile, edge, and cloud connected over hyper-reliable low-latency communication (HRLLC). In this setting, speculative decoding can facilitate collaborative inference of models distributively deployed: a lightweight on-device model locally generates drafts while a more capable remote target model on a server verifies and corrects them in parallel with speculative sampling, thus resulting in lower latency without compromising accuracy. However, unlike autoregressive text generation, behavior cloning policies, typically used for embodied AI applications, cannot parallelize verification and correction for multiple drafts as each generated action depends on observation updated by a previous action. To this end, we propose Action Deviation-Aware Hybrid Inference (ADAHI), wherein drafts are selectively transmitted and verified based on action deviation, which has a strong correlation with action's rejection probability by the target model. By invoking server operation only when necessary, communication and computational overhead can be reduced while accuracy gain from speculative sampling is preserved. Experiments on our testbed show that ADAHI reduces transmission and server operations by approximately 40%, lowers end-to-end latency by 39.2%, and attains up to 97.2% of the task-success rate of baseline that invokes speculative sampling for every draft embedding vector.
>
---
#### [replaced 014] Modeling Elastic-Body Dynamics of Robotic Fish Using a Variational Framework
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.16145v2](http://arxiv.org/pdf/2509.16145v2)**

> **作者:** Zhiheng Chen; Wei Wang
>
> **备注:** Under review at IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Fish-inspired aquatic robots are gaining increasing attention in marine robot communities due to their high swimming speeds and efficient propulsion enabled by flexible bodies that generate undulatory motions. To support the design optimization and control of such systems, accurate, interpretable, and computationally tractable modeling of the underlying swimming dynamics is indispensable. In this letter, we present a full-body dynamics model for motor-actuated robotic fish, rigorously derived from Hamilton's principle. The model captures the continuously distributed elasticity of a deformable fish body undergoing large deformations and incorporates fluid-structure coupling effects, enabling self-propelled motion without prescribing kinematics. Preliminary open-loop simulations examine how actuation frequency and body stiffness influence the swimming speed and energy efficiency of the robotic fish. Closed-loop simulations further assess how stiffness distribution impacts the controller's velocity-tracking performance and energy efficiency. The results demonstrate the model's potential for performance evaluation and control optimization of soft robotic swimmers when stiffness is treated as a design variable.
>
---
#### [replaced 015] The Mini Wheelbot: A Testbed for Learning-based Balancing, Flips, and Articulated Driving
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2502.04582v2](http://arxiv.org/pdf/2502.04582v2)**

> **作者:** Henrik Hose; Jan Weisgerber; Sebastian Trimpe
>
> **摘要:** The Mini Wheelbot is a balancing, reaction wheel unicycle robot designed as a testbed for learning-based control. It is an unstable system with highly nonlinear yaw dynamics, non-holonomic driving, and discrete contact switches in a small, powerful, and rugged form factor. The Mini Wheelbot can use its wheels to stand up from any initial orientation - enabling automatic environment resets in repetitive experiments and even challenging half flips. We illustrate the effectiveness of the Mini Wheelbot as a testbed by implementing two popular learning-based control algorithms. First, we showcase Bayesian optimization for tuning the balancing controller. Second, we use imitation learning from an expert nonlinear MPC that uses gyroscopic effects to reorient the robot and can track higher-level velocity and orientation commands. The latter allows the robot to drive around based on user commands - for the first time in this class of robots. The Mini Wheelbot is not only compelling for testing learning-based control algorithms, but it is also just fun to work with, as demonstrated in the video of our experiments.
>
---
#### [replaced 016] Learning to Navigate Socially Through Proactive Risk Perception
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07871v2](http://arxiv.org/pdf/2510.07871v2)**

> **作者:** Erjia Xiao; Lingfeng Zhang; Yingbo Tang; Hao Cheng; Renjing Xu; Wenbo Ding; Lei Zhou; Long Chen; Hangjun Ye; Xiaoshuai Hao
>
> **摘要:** In this report, we describe the technical details of our submission to the IROS 2025 RoboSense Challenge Social Navigation Track. This track focuses on developing RGBD-based perception and navigation systems that enable autonomous agents to navigate safely, efficiently, and socially compliantly in dynamic human-populated indoor environments. The challenge requires agents to operate from an egocentric perspective using only onboard sensors including RGB-D observations and odometry, without access to global maps or privileged information, while maintaining social norm compliance such as safe distances and collision avoidance. Building upon the Falcon model, we introduce a Proactive Risk Perception Module to enhance social navigation performance. Our approach augments Falcon with collision risk understanding that learns to predict distance-based collision risk scores for surrounding humans, which enables the agent to develop more robust spatial awareness and proactive collision avoidance behaviors. The evaluation on the Social-HM3D benchmark demonstrates that our method improves the agent's ability to maintain personal space compliance while navigating toward goals in crowded indoor scenes with dynamic human agents, achieving 2nd place among 16 participating teams in the challenge.
>
---
#### [replaced 017] Generalized Nash Equilibrium Solutions in Dynamic Games With Shared Constraints
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.19569v3](http://arxiv.org/pdf/2502.19569v3)**

> **作者:** Mark Pustilnik; Francesco Borrelli
>
> **摘要:** In dynamic games with shared constraints, Generalized Nash Equilibria (GNE) are often computed using the normalized solution concept, which assumes identical Lagrange multipliers for shared constraints across all players. While widely used, this approach excludes other potentially valuable GNE. This paper presents a novel method based on the Mixed Complementarity Problem (MCP) formulation to compute non-normalized GNE, expanding the solution space. We also propose a systematic approach for selecting the optimal GNE based on predefined criteria, enhancing practical flexibility. Numerical examples illustrate the methods effectiveness, offering an alternative to traditional normalized solutions.
>
---
#### [replaced 018] Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.15680v2](http://arxiv.org/pdf/2506.15680v2)**

> **作者:** Kaifeng Zhang; Baoyu Li; Kris Hauser; Yunzhu Li
>
> **备注:** Project page: https://kywind.github.io/pgnd
>
> **摘要:** Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .
>
---
#### [replaced 019] SafeVLA: Towards Safety Alignment of Vision-Language-Action Model via Constrained Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.03480v3](http://arxiv.org/pdf/2503.03480v3)**

> **作者:** Borong Zhang; Yuhao Zhang; Jiaming Ji; Yingshan Lei; Josef Dai; Yuanpei Chen; Yaodong Yang
>
> **备注:** Accepted by NeurIPS 2025 Spotlight Presentation
>
> **摘要:** Vision-language-action models (VLAs) show potential as generalist robot policies. However, these models pose extreme safety challenges during real-world deployment, including the risk of harm to the environment, the robot itself, and humans. How can safety constraints be explicitly integrated into VLAs? We address this by exploring an integrated safety approach (ISA), systematically modeling safety requirements, then actively eliciting diverse unsafe behaviors, effectively constraining VLA policies via safe reinforcement learning, and rigorously assuring their safety through targeted evaluations. Leveraging the constrained Markov decision process (CMDP) paradigm, ISA optimizes VLAs from a min-max perspective against elicited safety risks. Thus, policies aligned through this comprehensive approach achieve the following key features: (I) effective safety-performance trade-offs, reducing the cumulative cost of safety violations by 83.58% compared to the state-of-the-art method, while also maintaining task success rate (+3.85%). (II) strong safety assurance, with the ability to mitigate long-tail risks and handle extreme failure scenarios. (III) robust generalization of learned safety behaviors to various out-of-distribution perturbations. The effectiveness is evaluated on long-horizon mobile manipulation tasks. Our data, models and newly proposed benchmark environment are available at https://pku-safevla.github.io.
>
---
