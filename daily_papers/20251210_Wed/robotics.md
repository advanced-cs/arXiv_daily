# 机器人 cs.RO

- **最新发布 42 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Ergodic Trajectory Planning with Dynamic Sensor Footprints
- **分类: cs.RO**

- **简介: 该论文研究信息采集中的轨迹规划问题，针对传感器视野动态变化的特性，提出考虑动态感知范围的新度量方法，优化轨迹以实现更优的遍历性，兼顾探索与利用，显著提升信息采集效率，并在多无人机系统中验证。**

- **链接: [https://arxiv.org/pdf/2512.08661v1](https://arxiv.org/pdf/2512.08661v1)**

> **作者:** Ziyue Zheng; Yongce Liu; Hesheng Wang; Zhongqiang Ren
>
> **备注:** 12 figures
>
> **摘要:** This paper addresses the problem of trajectory planning for information gathering with a dynamic and resolution-varying sensor footprint. Ergodic planning offers a principled framework that balances exploration (visiting all areas) and exploitation (focusing on high-information regions) by planning trajectories such that the time spent in a region is proportional to the amount of information in that region. Existing ergodic planning often oversimplifies the sensing model by assuming a point sensor or a footprint with constant shape and resolution. In practice, the sensor footprint can drastically change over time as the robot moves, such as aerial robots equipped with downward-facing cameras, whose field of view depends on the orientation and altitude. To overcome this limitation, we propose a new metric that accounts for dynamic sensor footprints, analyze the theoretic local optimality conditions, and propose numerical trajectory optimization algorithms. Experimental results show that the proposed approach can simultaneously optimize both the trajectories and sensor footprints, with up to an order of magnitude better ergodicity than conventional methods. We also deploy our approach in a multi-drone system to ergodically cover an object in 3D space.
>
---
#### [new 002] Optimized Area Coverage in Disaster Response Utilizing Autonomous UAV Swarm Formations
- **分类: cs.RO**

- **简介: 该论文研究无人机 swarm 在灾害响应中的区域覆盖任务，旨在解决飞行中避障、编队保持与高效覆盖问题。提出基于局部ESDF的自主导航框架和改进TSP路径规划，优化覆盖路径并优先探测高价值兴趣点，通过仿真验证了多机协同覆盖的有效性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.08028v1](https://arxiv.org/pdf/2512.08028v1)**

> **作者:** Lampis Papakostas; Aristeidis Geladaris; Athanasios Mastrogeorgiou; Jim Sharples; Gautier Hattenberger; Panagiotis Chatzakos; Panagiotis Polygerinos
>
> **摘要:** This paper presents a UAV swarm system designed to assist first responders in disaster scenarios like wildfires. By distributing sensors across multiple agents, the system extends flight duration and enhances data availability, reducing the risk of mission failure due to collisions. To mitigate this risk further, we introduce an autonomous navigation framework that utilizes a local Euclidean Signed Distance Field (ESDF) map for obstacle avoidance while maintaining swarm formation with minimal path deviation. Additionally, we incorporate a Traveling Salesman Problem (TSP) variant to optimize area coverage, prioritizing Points of Interest (POIs) based on preassigned values derived from environmental behavior and critical infrastructure. The proposed system is validated through simulations with varying swarm sizes, demonstrating its ability to maximize coverage while ensuring collision avoidance between UAVs and obstacles.
>
---
#### [new 003] Chat with UAV -- Human-UAV Interaction Based on Large Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种基于大语言模型的双智能体人机交互框架，旨在解决无人机与用户间自然语言交互难、任务规划与执行适应性差的问题，提升交互流畅性与任务灵活性，满足个性化需求。**

- **链接: [https://arxiv.org/pdf/2512.08145v1](https://arxiv.org/pdf/2512.08145v1)**

> **作者:** Haoran Wang; Zhuohang Chen; Guang Li; Bo Ma; Chuanghuang Li
>
> **摘要:** The future of UAV interaction systems is evolving from engineer-driven to user-driven, aiming to replace traditional predefined Human-UAV Interaction designs. This shift focuses on enabling more personalized task planning and design, thereby achieving a higher quality of interaction experience and greater flexibility, which can be used in many fileds, such as agriculture, aerial photography, logistics, and environmental monitoring. However, due to the lack of a common language between users and the UAVs, such interactions are often difficult to be achieved. The developments of Large Language Models possess the ability to understand nature languages and Robots' (UAVs') behaviors, marking the possibility of personalized Human-UAV Interaction. Recently, some HUI frameworks based on LLMs have been proposed, but they commonly suffer from difficulties in mixed task planning and execution, leading to low adaptability in complex scenarios. In this paper, we propose a novel dual-agent HUI framework. This framework constructs two independent LLM agents (a task planning agent, and an execution agent) and applies different Prompt Engineering to separately handle the understanding, planning, and execution of tasks. To verify the effectiveness and performance of the framework, we have built a task database covering four typical application scenarios of UAVs and quantified the performance of the HUI framework using three independent metrics. Meanwhile different LLM models are selected to control the UAVs with compared performance. Our user study experimental results demonstrate that the framework improves the smoothness of HUI and the flexibility of task execution in the tasks scenario we set up, effectively meeting users' personalized needs.
>
---
#### [new 004] Robust Finetuning of Vision-Language-Action Robot Policies via Parameter Merging
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究视觉-语言-动作机器人策略的鲁棒微调。针对微调时易过拟合并遗忘原有能力的问题，提出通过插值微调模型与预训练模型权重的融合方法，使单一策略既能学会新任务，又保留泛化能力，并支持持续学习。**

- **链接: [https://arxiv.org/pdf/2512.08333v1](https://arxiv.org/pdf/2512.08333v1)**

> **作者:** Yajat Yadav; Zhiyuan Zhou; Andrew Wagenmaker; Karl Pertsch; Sergey Levine
>
> **摘要:** Generalist robot policies, trained on large and diverse datasets, have demonstrated the ability to generalize across a wide spectrum of behaviors, enabling a single policy to act in varied real-world environments. However, they still fall short on new tasks not covered in the training data. When finetuned on limited demonstrations of a new task, these policies often overfit to the specific demonstrations--not only losing their prior abilities to solve a wide variety of generalist tasks but also failing to generalize within the new task itself. In this work, we aim to develop a method that preserves the generalization capabilities of the generalist policy during finetuning, allowing a single policy to robustly incorporate a new skill into its repertoire. Our goal is a single policy that both learns to generalize to variations of the new task and retains the broad competencies gained from pretraining. We show that this can be achieved through a simple yet effective strategy: interpolating the weights of a finetuned model with that of the pretrained model. We show, across extensive simulated and real-world experiments, that such model merging produces a single model that inherits the generalist abilities of the base model and learns to solve the new task robustly, outperforming both the pretrained and finetuned model on out-of-distribution variations of the new task. Moreover, we show that model merging enables continual acquisition of new skills in a lifelong learning setting, without sacrificing previously learned generalist abilities.
>
---
#### [new 005] Data-Driven Dynamic Parameter Learning of manipulator robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机械臂动态参数估计，旨在缩小仿真到现实的差距。针对传统方法难以准确建模的问题，提出基于Transformer的方法，利用自动生成的多样化机器人数据和运动学特征，实现对质量、惯性等参数的高精度估计，提升模型泛化与控制性能。**

- **链接: [https://arxiv.org/pdf/2512.08767v1](https://arxiv.org/pdf/2512.08767v1)**

> **作者:** Mohammed Elseiagy; Tsige Tadesse Alemayoh; Ranulfo Bezerra; Shotaro Kojima; Kazunori Ohno
>
> **备注:** Accepted for publication at SII 2026. 6 pages, 7 figures. Code is available at: https://github.com/MohamedAlsiagy/dynamic_parameter_est
>
> **摘要:** Bridging the sim-to-real gap remains a fundamental challenge in robotics, as accurate dynamic parameter estimation is essential for reliable model-based control, realistic simulation, and safe deployment of manipulators. Traditional analytical approaches often fall short when faced with complex robot structures and interactions. Data-driven methods offer a promising alternative, yet conventional neural networks such as recurrent models struggle to capture long-range dependencies critical for accurate estimation. In this study, we propose a Transformer-based approach for dynamic parameter estimation, supported by an automated pipeline that generates diverse robot models and enriched trajectory data using Jacobian-derived features. The dataset consists of 8,192 robots with varied inertial and frictional properties. Leveraging attention mechanisms, our model effectively captures both temporal and spatial dependencies. Experimental results highlight the influence of sequence length, sampling rate, and architecture, with the best configuration (sequence length 64, 64 Hz, four layers, 32 heads) achieving a validation R2 of 0.8633. Mass and inertia are estimated with near-perfect accuracy, Coulomb friction with moderate-to-high accuracy, while viscous friction and distal link center-of-mass remain more challenging. These results demonstrate that combining Transformers with automated dataset generation and kinematic enrichment enables scalable, accurate dynamic parameter estimation, contributing to improved sim-to-real transfer in robotic systems
>
---
#### [new 006] Heterogeneity in Multi-Robot Environmental Monitoring for Resolving Time-Conflicting Tasks
- **分类: cs.RO**

- **简介: 该论文研究多机器人环境监测中巡逻与紧急信号搜索任务的时间冲突问题。通过行为与感知异质性设计，分析不同团队构成的权衡表现，发现角色和感知专业化可优化系统性能，支持任务灵活调配并降低传感器部署成本。**

- **链接: [https://arxiv.org/pdf/2512.08813v1](https://arxiv.org/pdf/2512.08813v1)**

> **作者:** Connor York; Zachary R Madin; Paul O'Dowd; Edmund R Hunt
>
> **备注:** Accepted to SAC '26. To appear, DOI: https://doi.org/10.1145/3748522.3779970
>
> **摘要:** Multi-robot systems performing continuous tasks face a performance trade-off when interrupted by urgent, time-critical sub-tasks. We investigate this trade-off in a scenario where a team must balance area patrolling with locating an anomalous radio signal. To address this trade-off, we evaluate both behavioral heterogeneity through agent role specialization ("patrollers" and "searchers") and sensing heterogeneity (i.e., only the searchers can sense the radio signal). Through simulation, we identify the Pareto-optimal trade-offs under varying team compositions, with behaviorally heterogeneous teams demonstrating the most balanced trade-offs in the majority of cases. When sensing capability is restricted, heterogeneous teams with half of the sensing-capable agents perform comparably to homogeneous teams, providing cost-saving rationale for restricting sensor payload deployment. Our findings demonstrate that pre-deployment role and sensing specialization are powerful design considerations for multi-robot systems facing time-conflicting tasks, where varying the degree of behavioral heterogeneity can tune system performance toward either task.
>
---
#### [new 007] Non Normalized Shared-Constraint Dynamic Games for Human-Robot Collaboration with Asymmetric Responsibility
- **分类: cs.RO**

- **简介: 该论文研究人机协作导航任务，解决非对称责任下共享安全约束的分配问题。提出非归一化共享约束动态博弈框架，使双方以不同努力程度满足安全要求，并嵌入预测控制方案实现协同避障与任务执行。**

- **链接: [https://arxiv.org/pdf/2512.08688v1](https://arxiv.org/pdf/2512.08688v1)**

> **作者:** Mark Pustilnik; Francesco Borrelli
>
> **摘要:** This paper proposes a dynamic game formulation for cooperative human-robot navigation in shared workspaces with obstacles, where the human and robot jointly satisfy shared safety constraints while pursuing a common task. A key contribution is the introduction of a non-normalized equilibrium structure for the shared constraints. This structure allows the two agents to contribute different levels of effort towards enforcing safety requirements such as collision avoidance and inter-players spacing. We embed this non-normalized equilibrium into a receding-horizon optimal control scheme.
>
---
#### [new 008] Semantic-Metric Bayesian Risk Fields: Learning Robot Safety from Human Videos with a VLM Prior
- **分类: cs.RO**

- **简介: 该论文提出语义度量贝叶斯风险场，从人类视频中学习连续、上下文相关的隐式安全模型。利用VLM先验与视觉Transformer后验，生成像素级风险图，用于机器人规划，实现类人风险感知与决策。**

- **链接: [https://arxiv.org/pdf/2512.08233v1](https://arxiv.org/pdf/2512.08233v1)**

> **作者:** Timothy Chen; Marcus Dominguez-Kuhne; Aiden Swann; Xu Liu; Mac Schwager
>
> **摘要:** Humans interpret safety not as a binary signal but as a continuous, context- and spatially-dependent notion of risk. While risk is subjective, humans form rational mental models that guide action selection in dynamic environments. This work proposes a framework for extracting implicit human risk models by introducing a novel, semantically-conditioned and spatially-varying parametrization of risk, supervised directly from safe human demonstration videos and VLM common sense. Notably, we define risk through a Bayesian formulation. The prior is furnished by a pretrained vision-language model. In order to encourage the risk estimate to be more human aligned, a likelihood function modulates the prior to produce a relative metric of risk. Specifically, the likelihood is a learned ViT that maps pretrained features, to pixel-aligned risk values. Our pipeline ingests RGB images and a query object string, producing pixel-dense risk images. These images that can then be used as value-predictors in robot planning tasks or be projected into 3D for use in conventional trajectory optimization to produce human-like motion. This learned mapping enables generalization to novel objects and contexts, and has the potential to scale to much larger training datasets. In particular, the Bayesian framework that is introduced enables fast adaptation of our model to additional observations or common sense rules. We demonstrate that our proposed framework produces contextual risk that aligns with human preferences. Additionally, we illustrate several downstream applications of the model; as a value learner for visuomotor planners or in conjunction with a classical trajectory optimization algorithm. Our results suggest that our framework is a significant step toward enabling autonomous systems to internalize human-like risk. Code and results can be found at https://riskbayesian.github.io/bayesian_risk/.
>
---
#### [new 009] DIJIT: A Robotic Head for an Active Observer
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DIJIT——一种具有九个机械自由度的双目机器人头，用于主动视觉研究。旨在模拟人类眼-头运动，探究其在视觉任务中的作用，并开发接近人类扫视运动的相机控制方法。**

- **链接: [https://arxiv.org/pdf/2512.07998v1](https://arxiv.org/pdf/2512.07998v1)**

> **作者:** Mostafa Kamali Tabrizi; Mingshi Chi; Bir Bikram Dey; Yu Qing Yuan; Markus D. Solbach; Yiqian Liu; Michael Jenkin; John K. Tsotsos
>
> **摘要:** We present DIJIT, a novel binocular robotic head expressly designed for mobile agents that behave as active observers. DIJIT's unique breadth of functionality enables active vision research and the study of human-like eye and head-neck motions, their interrelationships, and how each contributes to visual ability. DIJIT is also being used to explore the differences between how human vision employs eye/head movements to solve visual tasks and current computer vision methods. DIJIT's design features nine mechanical degrees of freedom, while the cameras and lenses provide an additional four optical degrees of freedom. The ranges and speeds of the mechanical design are comparable to human performance. Our design includes the ranges of motion required for convergent stereo, namely, vergence, version, and cyclotorsion. The exploration of the utility of these to both human and machine vision is ongoing. Here, we present the design of DIJIT and evaluate aspects of its performance. We present a new method for saccadic camera movements. In this method, a direct relationship between camera orientation and motor values is developed. The resulting saccadic camera movements are close to human movements in terms of their accuracy.
>
---
#### [new 010] Learning Robot Manipulation from Audio World Models
- **分类: cs.RO**

- **简介: 该论文研究机器人操作任务中的音频模态建模，旨在通过预测未来音频状态来提升策略推理能力。提出一种生成式潜变量流匹配模型，利用音频的时序与节奏特征实现长时程动作规划，在需感知真实音频或音乐的任务中优于无前瞻方法。**

- **链接: [https://arxiv.org/pdf/2512.08405v1](https://arxiv.org/pdf/2512.08405v1)**

> **作者:** Fan Zhang; Michael Gienger
>
> **摘要:** World models have demonstrated impressive performance on robotic learning tasks. Many such tasks inherently demand multimodal reasoning; for example, filling a bottle with water will lead to visual information alone being ambiguous or incomplete, thereby requiring reasoning over the temporal evolution of audio, accounting for its underlying physical properties and pitch patterns. In this paper, we propose a generative latent flow matching model to anticipate future audio observations, enabling the system to reason about long-term consequences when integrated into a robot policy. We demonstrate the superior capabilities of our system through two manipulation tasks that require perceiving in-the-wild audio or music signals, compared to methods without future lookahead. We further emphasize that successful robot action learning for these tasks relies not merely on multi-modal input, but critically on the accurate prediction of future audio states that embody intrinsic rhythmic patterns.
>
---
#### [new 011] OSMO: Open-Source Tactile Glove for Human-to-Robot Skill Transfer
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出OSMO，一种开源触觉手套，用于人体到机器人技能迁移。针对视频无法捕捉接触信息的问题，通过12个三轴传感器采集触觉信号，结合手部追踪，实现无需真实机器人数据的策略训练，在擦拭任务中成功率达72%。**

- **链接: [https://arxiv.org/pdf/2512.08920v1](https://arxiv.org/pdf/2512.08920v1)**

> **作者:** Jessica Yin; Haozhi Qi; Youngsun Wi; Sayantan Kundu; Mike Lambeta; William Yang; Changhao Wang; Tingfan Wu; Jitendra Malik; Tess Hellebrekers
>
> **备注:** Project website: https://jessicayin.github.io/osmo_tactile_glove/
>
> **摘要:** Human video demonstrations provide abundant training data for learning robot policies, but video alone cannot capture the rich contact signals critical for mastering manipulation. We introduce OSMO, an open-source wearable tactile glove designed for human-to-robot skill transfer. The glove features 12 three-axis tactile sensors across the fingertips and palm and is designed to be compatible with state-of-the-art hand-tracking methods for in-the-wild data collection. We demonstrate that a robot policy trained exclusively on human demonstrations collected with OSMO, without any real robot data, is capable of executing a challenging contact-rich manipulation task. By equipping both the human and the robot with the same glove, OSMO minimizes the visual and tactile embodiment gap, enabling the transfer of continuous shear and normal force feedback while avoiding the need for image inpainting or other vision-based force inference. On a real-world wiping task requiring sustained contact pressure, our tactile-aware policy achieves a 72% success rate, outperforming vision-only baselines by eliminating contact-related failure modes. We release complete hardware designs, firmware, and assembly instructions to support community adoption.
>
---
#### [new 012] VLD: Visual Language Goal Distance for Reinforcement Learning Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉语言导航任务，旨在解决端到端策略学习中仿真到现实的迁移难题及标注数据不足问题。提出VLD框架，通过解耦感知与策略学习，利用互联网视频自监督训练距离预测器，再在仿真中训练强化学习策略，实现对图像和文本目标的灵活导航。**

- **链接: [https://arxiv.org/pdf/2512.07976v1](https://arxiv.org/pdf/2512.07976v1)**

> **作者:** Lazar Milikic; Manthan Patel; Jonas Frey
>
> **摘要:** Training end-to-end policies from image data to directly predict navigation actions for robotic systems has proven inherently difficult. Existing approaches often suffer from either the sim-to-real gap during policy transfer or a limited amount of training data with action labels. To address this problem, we introduce Vision-Language Distance (VLD) learning, a scalable framework for goal-conditioned navigation that decouples perception learning from policy learning. Instead of relying on raw sensory inputs during policy training, we first train a self-supervised distance-to-goal predictor on internet-scale video data. This predictor generalizes across both image- and text-based goals, providing a distance signal that can be minimized by a reinforcement learning (RL) policy. The RL policy can be trained entirely in simulation using privileged geometric distance signals, with injected noise to mimic the uncertainty of the trained distance predictor. At deployment, the policy consumes VLD predictions, inheriting semantic goal information-"where to go"-from large-scale visual training while retaining the robust low-level navigation behaviors learned in simulation. We propose using ordinal consistency to assess distance functions directly and demonstrate that VLD outperforms prior temporal distance approaches, such as ViNT and VIP. Experiments show that our decoupled design achieves competitive navigation performance in simulation while supporting flexible goal modalities, providing an alternative and, most importantly, scalable path toward reliable, multimodal navigation policies.
>
---
#### [new 013] Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文研究离线决策任务，解决生成轨迹动态不可行的问题。提出MPDiffuser框架，结合规划、动力学建模与排序模块，通过交替扩散采样提升轨迹合理性与任务对齐性，并在多类基准及真实四足机器人上验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.08280v1](https://arxiv.org/pdf/2512.08280v1)**

> **作者:** Haldun Balim; Na Li; Yilun Du
>
> **摘要:** Offline decision-making requires synthesizing reliable behaviors from fixed datasets without further interaction, yet existing generative approaches often yield trajectories that are dynamically infeasible. We propose Model Predictive Diffuser (MPDiffuser), a compositional model-based diffusion framework consisting of: (i) a planner that generates diverse, task-aligned trajectories; (ii) a dynamics model that enforces consistency with the underlying system dynamics; and (iii) a ranker module that selects behaviors aligned with the task objectives. MPDiffuser employs an alternating diffusion sampling scheme, where planner and dynamics updates are interleaved to progressively refine trajectories for both task alignment and feasibility during the sampling process. We also provide a theoretical rationale for this procedure, showing how it balances fidelity to data priors with dynamics consistency. Empirically, the compositional design improves sample efficiency, as it leverages even low-quality data for dynamics learning and adapts seamlessly to novel dynamics. We evaluate MPDiffuser on both unconstrained (D4RL) and constrained (DSRL) offline decision-making benchmarks, demonstrating consistent gains over existing approaches. Furthermore, we present a preliminary study extending MPDiffuser to vision-based control tasks, showing its potential to scale to high-dimensional sensory inputs. Finally, we deploy our method on a real quadrupedal robot, showcasing its practicality for real-world control.
>
---
#### [new 014] Learning Spatiotemporal Tubes for Temporal Reach-Avoid-Stay Tasks using Physics-Informed Neural Networks
- **分类: cs.RO**

- **简介: 该论文研究时变可达-避障-保持控制任务，针对未知动态的非线性系统，提出基于物理信息神经网络学习时空管状约束的方法，实现无需模型逼近的闭环控制，并验证了其在机器人和飞行器复杂环境导航中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.08248v1](https://arxiv.org/pdf/2512.08248v1)**

> **作者:** Ahan Basu; Ratnangshu Das; Pushpak Jagtap
>
> **摘要:** This paper presents a Spatiotemporal Tube (STT)-based control framework for general control-affine MIMO nonlinear pure-feedback systems with unknown dynamics to satisfy prescribed time reach-avoid-stay tasks under external disturbances. The STT is defined as a time-varying ball, whose center and radius are jointly approximated by a Physics-Informed Neural Network (PINN). The constraints governing the STT are first formulated as loss functions of the PINN, and a training algorithm is proposed to minimize the overall violation. The PINN being trained on certain collocation points, we propose a Lipschitz-based validity condition to formally verify that the learned PINN satisfies the conditions over the continuous time horizon. Building on the learned STT representation, an approximation-free closed-form controller is defined to guarantee satisfaction of the T-RAS specification. Finally, the effectiveness and scalability of the framework are validated through two case studies involving a mobile robot and an aerial vehicle navigating through cluttered environments.
>
---
#### [new 015] RVC-NMPC: Nonlinear Model Predictive Control with Reciprocal Velocity Constraints for Mutual Collision Avoidance in Agile UAV Flight
- **分类: cs.RO**

- **简介: 该论文研究无人机敏捷飞行中的互碰撞规避问题，提出一种基于非线性模型预测控制与时间依赖型互逆速度约束的方法。仅用可观测信息，无需频繁通信，实现实时高效避障，100 Hz运行频率下显著缩短飞行时间并保持零碰撞。**

- **链接: [https://arxiv.org/pdf/2512.08574v1](https://arxiv.org/pdf/2512.08574v1)**

> **作者:** Vit Kratky; Robert Penicka; Parakh M. Gupta; Ondrej Prochazka; Martin Saska
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** This paper presents an approach to mutual collision avoidance based on Nonlinear Model Predictive Control (NMPC) with time-dependent Reciprocal Velocity Constraints (RVCs). Unlike most existing methods, the proposed approach relies solely on observable information about other robots, eliminating the necessity of excessive communication use. The computationally efficient algorithm for computing RVCs, together with the direct integration of these constraints into NMPC problem formulation on a controller level, allows the whole pipeline to run at 100 Hz. This high processing rate, combined with modeled nonlinear dynamics of the controlled Uncrewed Aerial Vehicles (UAVs), is a key feature that facilitates the use of the proposed approach for an agile UAV flight. The proposed approach was evaluated through extensive simulations emulating real-world conditions in scenarios involving up to 10 UAVs and velocities of up to 25 m/s, and in real-world experiments with accelerations up to 30 m/s$^2$. Comparison with state of the art shows 31% improvement in terms of flight time reduction in challenging scenarios, while maintaining a collision-free navigation in all trials.
>
---
#### [new 016] Bridging Scale Discrepancies in Robotic Control via Language-Based Action Representations
- **分类: cs.RO; cs.AI**

- **简介: 该论文属机器人操控任务，旨在解决跨平台动作数据分布差异大导致的知识迁移困难。作者提出基于语言的动作表示方法，忽略数值尺度、强调运动方向，以增强预训练的泛化性和模态一致性，提升多任务迁移性能。**

- **链接: [https://arxiv.org/pdf/2512.08548v1](https://arxiv.org/pdf/2512.08548v1)**

> **作者:** Yuchi Zhang; Churui Sun; Shiqi Liang; Diyuan Liu; Chao Ji; Wei-Nan Zhang; Ting Liu
>
> **摘要:** Recent end-to-end robotic manipulation research increasingly adopts architectures inspired by large language models to enable robust manipulation. However, a critical challenge arises from severe distribution shifts between robotic action data, primarily due to substantial numerical variations in action commands across diverse robotic platforms and tasks, hindering the effective transfer of pretrained knowledge. To address this limitation, we propose a semantically grounded linguistic representation to normalize actions for efficient pretraining. Unlike conventional discretized action representations that are sensitive to numerical scales, the motion representation specifically disregards numeric scale effects, emphasizing directionality instead. This abstraction mitigates distribution shifts, yielding a more generalizable pretraining representation. Moreover, using the motion representation narrows the feature distance between action tokens and standard vocabulary tokens, mitigating modality gaps. Multi-task experiments on two benchmarks demonstrate that the proposed method significantly improves generalization performance and transferability in robotic manipulation tasks.
>
---
#### [new 017] Embodied Tree of Thoughts: Deliberate Manipulation Planning with Embodied World Model
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对机器人操作规划中世界模型物理不准确的问题，提出Embodied Tree of Thoughts（EToT）框架。它结合物理仿真与视觉语言模型，通过先验与反思分支机制进行树搜索规划，提升长视野任务中的物理一致性与容错能力。**

- **链接: [https://arxiv.org/pdf/2512.08188v1](https://arxiv.org/pdf/2512.08188v1)**

> **作者:** Wenjiang Xu; Cindy Wang; Rui Fang; Mingkang Zhang; Lusong Li; Jing Xu; Jiayuan Gu; Zecui Zeng; Rui Chen
>
> **备注:** Website at https://embodied-tree-of-thoughts.github.io
>
> **摘要:** World models have emerged as a pivotal component in robot manipulation planning, enabling agents to predict future environmental states and reason about the consequences of actions before execution. While video-generation models are increasingly adopted, they often lack rigorous physical grounding, leading to hallucinations and a failure to maintain consistency in long-horizon physical constraints. To address these limitations, we propose Embodied Tree of Thoughts (EToT), a novel Real2Sim2Real planning framework that leverages a physics-based interactive digital twin as an embodied world model. EToT formulates manipulation planning as a tree search expanded through two synergistic mechanisms: (1) Priori Branching, which generates diverse candidate execution paths based on semantic and spatial analysis; and (2) Reflective Branching, which utilizes VLMs to diagnose execution failures within the simulator and iteratively refine the planning tree with corrective actions. By grounding high-level reasoning in a physics simulator, our framework ensures that generated plans adhere to rigid-body dynamics and collision constraints. We validate EToT on a suite of short- and long-horizon manipulation tasks, where it consistently outperforms baselines by effectively predicting physical dynamics and adapting to potential failures. Website at https://embodied-tree-of-thoughts.github.io .
>
---
#### [new 018] Prospect Theory in Physical Human-Robot Interaction: A Pilot Study of Probability Perception
- **分类: cs.RO**

- **简介: 该论文研究物理人机交互中人类对不确定性的行为响应，发现个体在概率感知上存在偏差，表现为不同决策模式。为更准确建模此类行为，提出引入累积前景理论，以改进传统最优控制框架，提升机器人自适应控制设计。**

- **链接: [https://arxiv.org/pdf/2512.08481v1](https://arxiv.org/pdf/2512.08481v1)**

> **作者:** Yixiang Lin; Tiancheng Yang; Jonathan Eden; Ying Tan
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Understanding how humans respond to uncertainty is critical for designing safe and effective physical human-robot interaction (pHRI), as physically working with robots introduces multiple sources of uncertainty, including trust, comfort, and perceived safety. Conventional pHRI control frameworks typically build on optimal control theory, which assumes that human actions minimize a cost function; however, human behavior under uncertainty often departs from such optimal patterns. To address this gap, additional understanding of human behavior under uncertainty is needed. This pilot study implemented a physically coupled target-reaching task in which the robot delivered assistance or disturbances with systematically varied probabilities (10\% to 90\%). Analysis of participants' force inputs and decision-making strategies revealed two distinct behavioral clusters: a "trade-off" group that modulated their physical responses according to disturbance likelihood, and an "always-compensate" group characterized by strong risk aversion irrespective of probability. These findings provide empirical evidence that human decision-making in pHRI is highly individualized and that the perception of probability can differ to its true value. Accordingly, the study highlights the need for more interpretable behavioral models, such as cumulative prospect theory (CPT), to more accurately capture these behaviors and inform the design of future adaptive robot controllers.
>
---
#### [new 019] A Multi-Agent LLM Framework for Design Space Exploration in Autonomous Driving Systems
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶系统设计空间探索（DSE）中多模态输出复杂、依赖人工评估的问题，提出一种基于多智能体大语言模型（LLM）的自动化框架。通过集成3D仿真与分析工具，实现设计生成、执行调度与结果理解的全自动化，提升了Pareto最优解的发现效率。**

- **链接: [https://arxiv.org/pdf/2512.08476v1](https://arxiv.org/pdf/2512.08476v1)**

> **作者:** Po-An Shih; Shao-Hua Wang; Yung-Che Li; Chia-Heng Tu; Chih-Han Chang
>
> **摘要:** Designing autonomous driving systems requires efficient exploration of large hardware/software configuration spaces under diverse environmental conditions, e.g., with varying traffic, weather, and road layouts. Traditional design space exploration (DSE) approaches struggle with multi-modal execution outputs and complex performance trade-offs, and often require human involvement to assess correctness based on execution outputs. This paper presents a multi-agent, large language model (LLM)-based DSE framework, which integrates multi-modal reasoning with 3D simulation and profiling tools to automate the interpretation of execution outputs and guide the exploration of system designs. Specialized LLM agents are leveraged to handle user input interpretation, design point generation, execution orchestration, and analysis of both visual and textual execution outputs, which enables identification of potential bottlenecks without human intervention. A prototype implementation is developed and evaluated on a robotaxi case study (an SAE Level 4 autonomous driving application). Compared with a genetic algorithm baseline, the proposed framework identifies more Pareto-optimal, cost-efficient solutions with reduced navigation time under the same exploration budget. Experimental results also demonstrate the efficiency of the adoption of the LLM-based approach for DSE. We believe that this framework paves the way to the design automation of autonomous driving systems.
>
---
#### [new 020] An Introduction to Deep Reinforcement and Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文介绍面向具身智能体的深度强化学习与模仿学习，旨在解决复杂序列决策任务中控制器设计难题。通过讲解基础算法如PPO、DAgger和GAIL，提供自包含的理论与方法导论，侧重深入理解而非广泛综述。**

- **链接: [https://arxiv.org/pdf/2512.08052v1](https://arxiv.org/pdf/2512.08052v1)**

> **作者:** Pedro Santana
>
> **摘要:** Embodied agents, such as robots and virtual characters, must continuously select actions to execute tasks effectively, solving complex sequential decision-making problems. Given the difficulty of designing such controllers manually, learning-based approaches have emerged as promising alternatives, most notably Deep Reinforcement Learning (DRL) and Deep Imitation Learning (DIL). DRL leverages reward signals to optimize behavior, while DIL uses expert demonstrations to guide learning. This document introduces DRL and DIL in the context of embodied agents, adopting a concise, depth-first approach to the literature. It is self-contained, presenting all necessary mathematical and machine learning concepts as they are needed. It is not intended as a survey of the field; rather, it focuses on a small set of foundational algorithms and techniques, prioritizing in-depth understanding over broad coverage. The material ranges from Markov Decision Processes to REINFORCE and Proximal Policy Optimization (PPO) for DRL, and from Behavioral Cloning to Dataset Aggregation (DAgger) and Generative Adversarial Imitation Learning (GAIL) for DIL.
>
---
#### [new 021] Sparse Variable Projection in Robotic Perception: Exploiting Separable Structure for Efficient Nonlinear Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人感知中的非线性优化问题，利用变量可分性与稀疏性，提出一种适配规范对称的变量投影方法，构建无需显式存储的Schur补算子，显著加速求解过程，兼顾效率与精度。**

- **链接: [https://arxiv.org/pdf/2512.07969v1](https://arxiv.org/pdf/2512.07969v1)**

> **作者:** Alan Papalia; Nikolas Sanderson; Haoyu Han; Heng Yang; Hanumant Singh; Michael Everett
>
> **备注:** 8 pages, submitted for review
>
> **摘要:** Robotic perception often requires solving large nonlinear least-squares (NLS) problems. While sparsity has been well-exploited to scale solvers, a complementary and underexploited structure is \emph{separability} -- where some variables (e.g., visual landmarks) appear linearly in the residuals and, for any estimate of the remaining variables (e.g., poses), have a closed-form solution. Variable projection (VarPro) methods are a family of techniques that exploit this structure by analytically eliminating the linear variables and presenting a reduced problem in the remaining variables that has favorable properties. However, VarPro has seen limited use in robotic perception; a major challenge arises from gauge symmetries (e.g., cost invariance to global shifts and rotations), which are common in perception and induce specific computational challenges in standard VarPro approaches. We present a VarPro scheme designed for problems with gauge symmetries that jointly exploits separability and sparsity. Our method can be applied as a one-time preprocessing step to construct a \emph{matrix-free Schur complement operator}. This operator allows efficient evaluation of costs, gradients, and Hessian-vector products of the reduced problem and readily integrates with standard iterative NLS solvers. We provide precise conditions under which our method applies, and describe extensions when these conditions are only partially met. Across synthetic and real benchmarks in SLAM, SNL, and SfM, our approach achieves up to \textbf{2$\times$--35$\times$ faster runtimes} than state-of-the-art methods while maintaining accuracy. We release an open-source C++ implementation and all datasets from our experiments.
>
---
#### [new 022] Mind to Hand: Purposeful Robotic Control via Embodied Reasoning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Lumo-1模型，属于具身推理与机器人控制任务，旨在解决AI推理与物理动作脱节的问题。通过三阶段预训练和强化学习，实现从视觉语言推理到动作的连贯映射，提升机器人在复杂任务中的推理与执行能力。**

- **链接: [https://arxiv.org/pdf/2512.08580v1](https://arxiv.org/pdf/2512.08580v1)**

> **作者:** Peijun Tang; Shangjin Xie; Binyan Sun; Baifu Huang; Kuncheng Luo; Haotian Yang; Weiqi Jin; Jianan Wang
>
> **备注:** 49 pages, 25 figures
>
> **摘要:** Humans act with context and intention, with reasoning playing a central role. While internet-scale data has enabled broad reasoning capabilities in AI systems, grounding these abilities in physical action remains a major challenge. We introduce Lumo-1, a generalist vision-language-action (VLA) model that unifies robot reasoning ("mind") with robot action ("hand"). Our approach builds upon the general multi-modal reasoning capabilities of pre-trained vision-language models (VLMs), progressively extending them to embodied reasoning and action prediction, and ultimately towards structured reasoning and reasoning-action alignment. This results in a three-stage pre-training pipeline: (1) Continued VLM pre-training on curated vision-language data to enhance embodied reasoning skills such as planning, spatial understanding, and trajectory prediction; (2) Co-training on cross-embodiment robot data alongside vision-language data; and (3) Action training with reasoning process on trajectories collected on Astribot S1, a bimanual mobile manipulator with human-like dexterity and agility. Finally, we integrate reinforcement learning to further refine reasoning-action consistency and close the loop between semantic inference and motor control. Extensive experiments demonstrate that Lumo-1 achieves significant performance improvements in embodied vision-language reasoning, a critical component for generalist robotic control. Real-world evaluations further show that Lumo-1 surpasses strong baselines across a wide range of challenging robotic tasks, with strong generalization to novel objects and environments, excelling particularly in long-horizon tasks and responding to human-natural instructions that require reasoning over strategy, concepts and space.
>
---
#### [new 023] SensHRPS: Sensing Comfortable Human-Robot Proxemics and Personal Space With Eye-Tracking
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文研究人与机器人交互中的舒适距离感知任务，旨在通过眼动追踪识别用户对机器人空间距离的舒适度。实验结合主观报告与眼动数据，使用机器学习模型分析，发现决策树效果最佳，最小瞳孔直径为关键指标，表明人机舒适机制不同于人际。**

- **链接: [https://arxiv.org/pdf/2512.08518v1](https://arxiv.org/pdf/2512.08518v1)**

> **作者:** Nadezhda Kushina; Ko Watanabe; Aarthi Kannan; Ashita Ashok; Andreas Dengel; Karsten Berns
>
> **摘要:** Social robots must adjust to human proxemic norms to ensure user comfort and engagement. While prior research demonstrates that eye-tracking features reliably estimate comfort in human-human interactions, their applicability to interactions with humanoid robots remains unexplored. In this study, we investigate user comfort with the robot "Ameca" across four experimentally controlled distances (0.5 m to 2.0 m) using mobile eye-tracking and subjective reporting (N=19). We evaluate multiple machine learning and deep learning models to estimate comfort based on gaze features. Contrary to previous human-human studies where Transformer models excelled, a Decision Tree classifier achieved the highest performance (F1-score = 0.73), with minimum pupil diameter identified as the most critical predictor. These findings suggest that physiological comfort thresholds in human-robot interaction differ from human-human dynamics and can be effectively modeled using interpretable logic.
>
---
#### [new 024] A Sensor-Aware Phenomenological Framework for Lidar Degradation Simulation and SLAM Robustness Evaluation
- **分类: cs.RO**

- **简介: 该论文针对激光SLAM系统在复杂环境下的鲁棒性评估问题，提出一种感知真实传感器特性的退化模拟框架。通过在真实点云上施加可解释的物理退化（如遮挡、噪声等），实现对SLAM系统的可控、可重复压力测试，并支持多传感器与实时评估。**

- **链接: [https://arxiv.org/pdf/2512.08653v1](https://arxiv.org/pdf/2512.08653v1)**

> **作者:** Doumegna Mawuto Koudjo Felix; Xianjia Yu; Zhuo Zou; Tomi Westerlund
>
> **摘要:** Lidar-based SLAM systems are highly sensitive to adverse conditions such as occlusion, noise, and field-of-view (FoV) degradation, yet existing robustness evaluation methods either lack physical grounding or do not capture sensor-specific behavior. This paper presents a sensor-aware, phenomenological framework for simulating interpretable lidar degradations directly on real point clouds, enabling controlled and reproducible SLAM stress testing. Unlike image-derived corruption benchmarks (e.g., SemanticKITTI-C) or simulation-only approaches (e.g., lidarsim), the proposed system preserves per-point geometry, intensity, and temporal structure while applying structured dropout, FoV reduction, Gaussian noise, occlusion masking, sparsification, and motion distortion. The framework features autonomous topic and sensor detection, modular configuration with four severity tiers (light--extreme), and real-time performance (less than 20 ms per frame) compatible with ROS workflows. Experimental validation across three lidar architectures and five state-of-the-art SLAM systems reveals distinct robustness patterns shaped by sensor design and environmental context. The open-source implementation provides a practical foundation for benchmarking lidar-based SLAM under physically meaningful degradation scenarios.
>
---
#### [new 025] Zero-Splat TeleAssist: A Zero-Shot Pose Estimation Framework for Semantic Teleoperation
- **分类: cs.RO; cs.CV; cs.LG; eess.IV**

- **简介: 该论文提出Zero-Splat TeleAssist，属于机器人遥操作中的姿态估计任务，旨在无需标记物和深度传感器的情况下，实现多机器人6自由度位姿实时估计。通过融合视觉-语言分割、单目深度与3D高斯溅射，构建共享世界模型，支持多方协同遥操作。**

- **链接: [https://arxiv.org/pdf/2512.08271v1](https://arxiv.org/pdf/2512.08271v1)**

> **作者:** Srijan Dokania; Dharini Raghavan
>
> **备注:** Published and Presented at 3rd Workshop on Human-Centric Multilateral Teleoperation in ICRA 2025
>
> **摘要:** We introduce Zero-Splat TeleAssist, a zero-shot sensor-fusion pipeline that transforms commodity CCTV streams into a shared, 6-DoF world model for multilateral teleoperation. By integrating vision-language segmentation, monocular depth, weighted-PCA pose extraction, and 3D Gaussian Splatting (3DGS), TeleAssist provides every operator with real-time global positions and orientations of multiple robots without fiducials or depth sensors in an interaction-centric teleoperation setup.
>
---
#### [new 026] Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言导航（VLN）中动作碎片化、延迟高和动态避障难的问题，提出DualVLN双系统模型。其分离高层推理与低层执行，实现兼顾泛化能力、实时性与鲁棒性的长距离导航。**

- **链接: [https://arxiv.org/pdf/2512.08186v1](https://arxiv.org/pdf/2512.08186v1)**

> **作者:** Meng Wei; Chenyang Wan; Jiaqi Peng; Xiqian Yu; Yuqiang Yang; Delin Feng; Wenzhe Cai; Chenming Zhu; Tai Wang; Jiangmiao Pang; Xihui Liu
>
> **摘要:** While recent large vision-language models (VLMs) have improved generalization in vision-language navigation (VLN), existing methods typically rely on end-to-end pipelines that map vision-language inputs directly to short-horizon discrete actions. Such designs often produce fragmented motions, incur high latency, and struggle with real-world challenges like dynamic obstacle avoidance. We propose DualVLN, the first dual-system VLN foundation model that synergistically integrates high-level reasoning with low-level action execution. System 2, a VLM-based global planner, "grounds slowly" by predicting mid-term waypoint goals via image-grounded reasoning. System 1, a lightweight, multi-modal conditioning Diffusion Transformer policy, "moves fast" by leveraging both explicit pixel goals and latent features from System 2 to generate smooth and accurate trajectories. The dual-system design enables robust real-time control and adaptive local decision-making in complex, dynamic environments. By decoupling training, the VLM retains its generalization, while System 1 achieves interpretable and effective local navigation. DualVLN outperforms prior methods across all VLN benchmarks and real-world experiments demonstrate robust long-horizon planning and real-time adaptability in dynamic environments.
>
---
#### [new 027] Multi-Task Bayesian Optimization for Tuning Decentralized Trajectory Generation in Multi-UAV Systems
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多无人机系统中去中心化轨迹生成算法的参数调优问题，属于自动化与优化领域。利用多任务贝叶斯优化，通过多任务高斯过程建模不同场景间的关联，提升调优效率，并比较不同优化策略的效果。**

- **链接: [https://arxiv.org/pdf/2512.08630v1](https://arxiv.org/pdf/2512.08630v1)**

> **作者:** Marta Manzoni; Alessandro Nazzari; Roberto Rubinacci; Marco Lovera
>
> **摘要:** This paper investigates the use of Multi-Task Bayesian Optimization for tuning decentralized trajectory generation algorithms in multi-drone systems. We treat each task as a trajectory generation scenario defined by a specific number of drone-to-drone interactions. To model relationships across scenarios, we employ Multi-Task Gaussian Processes, which capture shared structure across tasks and enable efficient information transfer during optimization. We compare two strategies: optimizing the average mission time across all tasks and optimizing each task individually. Through a comprehensive simulation campaign, we show that single-task optimization leads to progressively shorter mission times as swarm size grows, but requires significantly more optimization time than the average-task approach.
>
---
#### [new 028] High-Performance Dual-Arm Task and Motion Planning for Tabletop Rearrangement
- **分类: cs.RO**

- **简介: 该论文研究双臂机器人在桌面物体重排中的任务与运动协同规划问题，提出SDAR框架，通过紧耦合的任务分解与同步运动规划，解决高纠缠、非单调复杂场景下的高效求解难题，实现100%成功率并优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.08206v1](https://arxiv.org/pdf/2512.08206v1)**

> **作者:** Duo Zhang; Junshan Huang; Jingjin Yu
>
> **备注:** ICRA 2026 Submission
>
> **摘要:** We propose Synchronous Dual-Arm Rearrangement Planner (SDAR), a task and motion planning (TAMP) framework for tabletop rearrangement, where two robot arms equipped with 2-finger grippers must work together in close proximity to rearrange objects whose start and goal configurations are strongly entangled. To tackle such challenges, SDAR tightly knit together its dependency-driven task planner (SDAR-T) and synchronous dual-arm motion planner (SDAR-M), to intelligently sift through a large number of possible task and motion plans. Specifically, SDAR-T applies a simple yet effective strategy to decompose the global object dependency graph induced by the rearrangement task, to produce more optimal dual-arm task plans than solutions derived from optimal task plans for a single arm. Leveraging state-of-the-art GPU SIMD-based motion planning tools, SDAR-M employs a layered motion planning strategy to sift through many task plans for the best synchronous dual-arm motion plan while ensuring high levels of success rate. Comprehensive evaluation demonstrates that SDAR delivers a 100% success rate in solving complex, non-monotone, long-horizon tabletop rearrangement tasks with solution quality far exceeding the previous state-of-the-art. Experiments on two UR-5e arms further confirm SDAR directly and reliably transfers to robot hardware.
>
---
#### [new 029] IPPO Learns the Game, Not the Team: A Study on Generalization in Heterogeneous Agent Teams
- **分类: cs.RO**

- **简介: 该论文研究异构多智能体团队中IPPO算法的泛化能力，探讨其在自博弈训练下是否学习到游戏本质策略而非过拟合队友行为。通过引入轮换策略训练（RPT）并在HeMAC环境中实验，发现IPPO能有效泛化至未见过的队友策略。**

- **链接: [https://arxiv.org/pdf/2512.08877v1](https://arxiv.org/pdf/2512.08877v1)**

> **作者:** Ryan LeRoy; Jack Kolb
>
> **备注:** 4 pages, 3 figures, appendix
>
> **摘要:** Multi-Agent Reinforcement Learning (MARL) is commonly deployed in settings where agents are trained via self-play with homogeneous teammates, often using parameter sharing and a single policy architecture. This opens the question: to what extent do self-play PPO agents learn general coordination strategies grounded in the underlying game, compared to overfitting to their training partners' behaviors? This paper investigates the question using the Heterogeneous Multi-Agent Challenge (HeMAC) environment, which features distinct Observer and Drone agents with complementary capabilities. We introduce Rotating Policy Training (RPT), an approach that rotates heterogeneous teammate policies of different learning algorithms during training, to expose the agent to a broader range of partner strategies. When playing alongside a withheld teammate policy (DDQN), we find that RPT achieves similar performance to a standard self-play baseline, IPPO, where all agents were trained sharing a single PPO policy. This result indicates that in this heterogeneous multi-agent setting, the IPPO baseline generalizes to novel teammate algorithms despite not experiencing teammate diversity during training. This shows that a simple IPPO baseline may possess the level of generalization to novel teammates that a diverse training regimen was designed to achieve.
>
---
#### [new 030] RAVES-Calib: Robust, Accurate and Versatile Extrinsic Self Calibration Using Optimal Geometric Features
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对无标定物环境下LiDAR-相机外参自标定任务，解决初始位姿未知和特征质量不均导致的精度与鲁棒性问题。提出RAVES-Calib，利用Gluestick建立2D-3D点线特征对应，自适应加权优化外参，无需初值且兼容多传感器，实现高精度、强鲁棒的标定。**

- **链接: [https://arxiv.org/pdf/2512.08170v1](https://arxiv.org/pdf/2512.08170v1)**

> **作者:** Haoxin Zhang; Shuaixin Li; Xiaozhou Zhu; Hongbo Chen; Wen Yao
>
> **摘要:** In this paper, we present a user-friendly LiDAR-camera calibration toolkit that is compatible with various LiDAR and camera sensors and requires only a single pair of laser points and a camera image in targetless environments. Our approach eliminates the need for an initial transform and remains robust even with large positional and rotational LiDAR-camera extrinsic parameters. We employ the Gluestick pipeline to establish 2D-3D point and line feature correspondences for a robust and automatic initial guess. To enhance accuracy, we quantitatively analyze the impact of feature distribution on calibration results and adaptively weight the cost of each feature based on these metrics. As a result, extrinsic parameters are optimized by filtering out the adverse effects of inferior features. We validated our method through extensive experiments across various LiDAR-camera sensors in both indoor and outdoor settings. The results demonstrate that our method provides superior robustness and accuracy compared to SOTA techniques. Our code is open-sourced on GitHub to benefit the community.
>
---
#### [new 031] Sim2Swim: Zero-Shot Velocity Control for Agile AUV Maneuvering in 3 Minutes
- **分类: cs.RO**

- **简介: 该论文研究自主水下航行器（AUV）的敏捷控制问题，提出一种基于零样本仿真到现实迁移的深度强化学习速度控制器Sim2Swim，实现无需调参的6自由度敏捷机动与路径跟踪，仅需3分钟训练即可部署。**

- **链接: [https://arxiv.org/pdf/2512.08656v1](https://arxiv.org/pdf/2512.08656v1)**

> **作者:** Lauritz Rismark Fosso; Herman Biørn Amundsen; Marios Xanthidis; Sveinung Johan Ohrem
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Holonomic autonomous underwater vehicles (AUVs) have the hardware ability for agile maneuvering in both translational and rotational degrees of freedom (DOFs). However, due to challenges inherent to underwater vehicles, such as complex hydrostatics and hydrodynamics, parametric uncertainties, and frequent changes in dynamics due to payload changes, control is challenging. Performance typically relies on carefully tuned controllers targeting unique platform configurations, and a need for re-tuning for deployment under varying payloads and hydrodynamic conditions. As a consequence, agile maneuvering with simultaneous tracking of time-varying references in both translational and rotational DOFs is rarely utilized in practice. To the best of our knowledge, this paper presents the first general zero-shot sim2real deep reinforcement learning-based (DRL) velocity controller enabling path following and agile 6DOF maneuvering with a training duration of just 3 minutes. Sim2Swim, the proposed approach, inspired by state-of-the-art DRL-based position control, leverages domain randomization and massively parallelized training to converge to field-deployable control policies for AUVs of variable characteristics without post-processing or tuning. Sim2Swim is extensively validated in pool trials for a variety of configurations, showcasing robust control for highly agile motions.
>
---
#### [new 032] A Multi-Robot Platform for Robotic Triage Combining Onboard Sensing and Foundation Models
- **分类: cs.RO**

- **简介: 该论文针对灾害现场伤员分诊任务，解决传统救援中人员风险高、评估不全的问题。提出一种空地协同多机器人系统，结合机载感知与基础模型，实现伤员定位、生命体征检测、伤情分级与数据整合，完成全流程远程分诊。**

- **链接: [https://arxiv.org/pdf/2512.08754v1](https://arxiv.org/pdf/2512.08754v1)**

> **作者:** Jason Hughes; Marcel Hussing; Edward Zhang; Shenbagaraj Kannapiran; Joshua Caswell; Kenneth Chaney; Ruichen Deng; Michaela Feehery; Agelos Kratimenos; Yi Fan Li; Britny Major; Ethan Sanchez; Sumukh Shrote; Youkang Wang; Jeremy Wang; Daudi Zein; Luying Zhang; Ruijun Zhang; Alex Zhou; Tenzi Zhouga; Jeremy Cannon; Zaffir Qasim; Jay Yelon; Fernando Cladera; Kostas Daniilidis; Camillo J. Taylor; Eric Eaton
>
> **备注:** Technical Report for the DARPA Triage Challenge PRONTO team
>
> **摘要:** This report presents a heterogeneous robotic system designed for remote primary triage in mass-casualty incidents (MCIs). The system employs a coordinated air-ground team of unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs) to locate victims, assess their injuries, and prioritize medical assistance without risking the lives of first responders. The UAV identify and provide overhead views of casualties, while UGVs equipped with specialized sensors measure vital signs and detect and localize physical injuries. Unlike previous work that focused on exploration or limited medical evaluation, this system addresses the complete triage process: victim localization, vital sign measurement, injury severity classification, mental status assessment, and data consolidation for first responders. Developed as part of the DARPA Triage Challenge, this approach demonstrates how multi-robot systems can augment human capabilities in disaster response scenarios to maximize lives saved.
>
---
#### [new 033] vEDGAR -- Can CARLA Do HiL?
- **分类: cs.RO**

- **简介: 该论文旨在解决CARLA模拟器无法支持硬件在环（HiL）测试的问题。作者提出了vEDGAR框架，实现了基于CARLA的实时全栈自动驾驶系统硬件测试，验证了其在HiL场景中的可行性，推动了开源自动驾驶开发流程的一致性评估。**

- **链接: [https://arxiv.org/pdf/2512.08541v1](https://arxiv.org/pdf/2512.08541v1)**

> **作者:** Nils Gehrke; David Brecht; Dominik Kulmer; Dheer Patel; Frank Diermeyer
>
> **摘要:** Simulation offers advantages throughout the development process of automated driving functions, both in research and product development. Common open-source simulators like CARLA are extensively used in training, evaluation, and software-in-the-loop testing of new automated driving algorithms. However, the CARLA simulator lacks an evaluation where research and automated driving vehicles are simulated with their entire sensor and actuation stack in real time. The goal of this work is therefore to create a simulation framework for testing the automation software on its dedicated hardware and identifying its limits. Achieving this goal would greatly benefit the open-source development workflow of automated driving functions, designating CARLA as a consistent evaluation tool along the entire development process. To achieve this goal, in a first step, requirements are derived, and a simulation architecture is specified and implemented. Based on the formulated requirements, the proposed vEDGAR software is evaluated, resulting in a final conclusion on the applicability of CARLA for HiL testing of automated vehicles. The tool is available open source: Modified CARLA fork: https://github.com/TUMFTM/carla, vEDGAR Framework: https://github.com/TUMFTM/vEDGAR
>
---
#### [new 034] SDT-6D: Fully Sparse Depth-Transformer for Staged End-to-End 6D Pose Estimation in Industrial Multi-View Bin Picking
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究工业多视角抓取中的6D姿态估计任务，旨在解决密集遮挡、无纹理等问题。提出全稀疏深度Transformer（SDT-6D），融合多视角深度图，通过分阶段热图机制与稀疏注意力，实现高分辨率下高效、精确的多物体位姿估计。**

- **链接: [https://arxiv.org/pdf/2512.08430v1](https://arxiv.org/pdf/2512.08430v1)**

> **作者:** Nico Leuze; Maximilian Hoh; Samed Doğan; Nicolas R. -Peña; Alfred Schoettl
>
> **备注:** Accepted to WACV 2026. Preprint version
>
> **摘要:** Accurately recovering 6D poses in densely packed industrial bin-picking environments remain a serious challenge, owing to occlusions, reflections, and textureless parts. We introduce a holistic depth-only 6D pose estimation approach that fuses multi-view depth maps into either a fine-grained 3D point cloud in its vanilla version, or a sparse Truncated Signed Distance Field (TSDF). At the core of our framework lies a staged heatmap mechanism that yields scene-adaptive attention priors across different resolutions, steering computation toward foreground regions, thus keeping memory requirements at high resolutions feasible. Along, we propose a density-aware sparse transformer block that dynamically attends to (self-) occlusions and the non-uniform distribution of 3D data. While sparse 3D approaches has proven effective for long-range perception, its potential in close-range robotic applications remains underexplored. Our framework operates fully sparse, enabling high-resolution volumetric representations to capture fine geometric details crucial for accurate pose estimation in clutter. Our method processes the entire scene integrally, predicting the 6D pose via a novel per-voxel voting strategy, allowing simultaneous pose predictions for an arbitrary number of target objects. We validate our method on the recently published IPD and MV-YCB multi-view datasets, demonstrating competitive performance in heavily cluttered industrial and household bin picking scenarios.
>
---
#### [new 035] LiDAS: Lighting-driven Dynamic Active Sensing for Nighttime Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究夜间视觉感知任务，解决光照不足导致的感知性能下降问题。提出LiDAS系统，通过动态调控车灯照明增强关键区域，提升检测与分割效果，实现无需训练的零样本泛化，在同等或更低功耗下显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.08912v1](https://arxiv.org/pdf/2512.08912v1)**

> **作者:** Simon de Moreau; Andrei Bursuc; Hafid El-Idrissi; Fabien Moutarde
>
> **备注:** Preprint. 12 pages, 9 figures. Project page: https://simondemoreau.github.io/LiDAS/
>
> **摘要:** Nighttime environments pose significant challenges for camera-based perception, as existing methods passively rely on the scene lighting. We introduce Lighting-driven Dynamic Active Sensing (LiDAS), a closed-loop active illumination system that combines off-the-shelf visual perception models with high-definition headlights. Rather than uniformly brightening the scene, LiDAS dynamically predicts an optimal illumination field that maximizes downstream perception performance, i.e., decreasing light on empty areas to reallocate it on object regions. LiDAS enables zero-shot nighttime generalization of daytime-trained models through adaptive illumination control. Trained on synthetic data and deployed zero-shot in real-world closed-loop driving scenarios, LiDAS enables +18.7% mAP50 and +5.0% mIoU over standard low-beam at equal power. It maintains performances while reducing energy use by 40%. LiDAS complements domain-generalization methods, further strengthening robustness without retraining. By turning readily available headlights into active vision actuators, LiDAS offers a cost-effective solution to robust nighttime perception.
>
---
#### [new 036] Using reinforcement learning to probe the role of feedback in skill acquisition
- **分类: cs.AI; cs.LG; cs.RO; eess.SY**

- **简介: 该论文研究技能学习中反馈的作用，利用强化学习控制水槽中的旋转圆柱以调控阻力。实验表明，执行策略无需反馈，但学习过程需丰富信息，且依赖任务目标：最大化阻力需反馈，最小化则否。**

- **链接: [https://arxiv.org/pdf/2512.08463v1](https://arxiv.org/pdf/2512.08463v1)**

> **作者:** Antonio Terpin; Raffaello D'Andrea
>
> **备注:** Website: https://antonioterpin.com/fluids-control
>
> **摘要:** Many high-performance human activities are executed with little or no external feedback: think of a figure skater landing a triple jump, a pitcher throwing a curveball for a strike, or a barista pouring latte art. To study the process of skill acquisition under fully controlled conditions, we bypass human subjects. Instead, we directly interface a generalist reinforcement learning agent with a spinning cylinder in a tabletop circulating water channel to maximize or minimize drag. This setup has several desirable properties. First, it is a physical system, with the rich interactions and complex dynamics that only the physical world has: the flow is highly chaotic and extremely difficult, if not impossible, to model or simulate accurately. Second, the objective -- drag minimization or maximization -- is easy to state and can be captured directly in the reward, yet good strategies are not obvious beforehand. Third, decades-old experimental studies provide recipes for simple, high-performance open-loop policies. Finally, the setup is inexpensive and far easier to reproduce than human studies. In our experiments we find that high-dimensional flow feedback lets the agent discover high-performance drag-control strategies with only minutes of real-world interaction. When we later replay the same action sequences without any feedback, we obtain almost identical performance. This shows that feedback, and in particular flow feedback, is not needed to execute the learned policy. Surprisingly, without flow feedback during training the agent fails to discover any well-performing policy in drag maximization, but still succeeds in drag minimization, albeit more slowly and less reliably. Our studies show that learning a high-performance skill can require richer information than executing it, and learning conditions can be kind or wicked depending solely on the goal, not on dynamics or policy complexity.
>
---
#### [new 037] RLCNet: An end-to-end deep learning framework for simultaneous online calibration of LiDAR, RADAR, and Camera
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多传感器标定任务，旨在解决自动驾驶中LiDAR、RADAR与相机因振动和漂移导致的外参失准问题。提出RLCNet——端到端可训练的深度学习框架，实现三传感器在线联合标定，具备实时性、抗噪性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.08262v1](https://arxiv.org/pdf/2512.08262v1)**

> **作者:** Hafeez Husain Cholakkal; Stefano Arrigoni; Francesco Braghin
>
> **摘要:** Accurate extrinsic calibration of LiDAR, RADAR, and camera sensors is essential for reliable perception in autonomous vehicles. Still, it remains challenging due to factors such as mechanical vibrations and cumulative sensor drift in dynamic environments. This paper presents RLCNet, a novel end-to-end trainable deep learning framework for the simultaneous online calibration of these multimodal sensors. Validated on real-world datasets, RLCNet is designed for practical deployment and demonstrates robust performance under diverse conditions. To support real-time operation, an online calibration framework is introduced that incorporates a weighted moving average and outlier rejection, enabling dynamic adjustment of calibration parameters with reduced prediction noise and improved resilience to drift. An ablation study highlights the significance of architectural choices, while comparisons with existing methods demonstrate the superior accuracy and robustness of the proposed approach.
>
---
#### [new 038] Decoupled Design of Time-Varying Control Barrier Functions via Equivariances
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究时间变控制屏障函数（CBF）的设计问题，旨在高效处理含时变约束与输入限制的系统。提出一种解耦设计方法，利用系统动力学的等变性，将时不变CBF与设计的时变变换结合，降低计算成本，适用于不确定环境。**

- **链接: [https://arxiv.org/pdf/2512.08607v1](https://arxiv.org/pdf/2512.08607v1)**

> **作者:** Adrian Wiltz; Dimos V. Dimarogonas
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** This article presents a systematic method for designing time-varying Control Barrier Functions (CBF) composed of a time-invariant component and multiple time-dependent components, leveraging structural properties of the system dynamics. The method involves the construction of a specific class of time-invariant CBFs that encode the system's dynamic capabilities with respect to a given constraint, and augments them subsequently with appropriately designed time-dependent transformations. While transformations uniformly varying the time-invariant CBF can be applied to arbitrary systems, transformations exploiting structural properties in the dynamics - equivariances in particular - enable the handling of a broader and more expressive class of time-varying constraints. The article shows how to leverage such properties in the design of time-varying CBFs. The proposed method decouples the design of time variations from the computationally expensive construction of the underlying CBFs, thereby providing a computationally attractive method to the design of time-varying CBFs. The method accounts for input constraints and under-actuation, and requires only qualitative knowledge on the time-variation of the constraints making it suitable to the application in uncertain environments.
>
---
#### [new 039] Geometry-Aware Sparse Depth Sampling for High-Fidelity RGB-D Depth Completion in Robotic Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究RGB-D深度补全任务，旨在解决现有方法中稀疏深度采样不真实的问题。作者提出一种几何感知的稀疏深度采样策略，利用PCA估计表面法向，生成更符合实际传感器特性的稀疏深度分布，并结合Marigold-DC模型提升补全精度与训练真实性。**

- **链接: [https://arxiv.org/pdf/2512.08229v1](https://arxiv.org/pdf/2512.08229v1)**

> **作者:** Tony Salloom; Dandi Zhou; Xinhai Sun
>
> **摘要:** Accurate three-dimensional perception is essential for modern industrial robotic systems that perform manipulation, inspection, and navigation tasks. RGB-D and stereo vision sensors are widely used for this purpose, but the depth maps they produce are often noisy, incomplete, or biased due to sensor limitations and environmental conditions. Depth completion methods aim to generate dense, reliable depth maps from RGB images and sparse depth input. However, a key limitation in current depth completion pipelines is the unrealistic generation of sparse depth: sparse pixels are typically selected uniformly at random from dense ground-truth depth, ignoring the fact that real sensors exhibit geometry-dependent and spatially nonuniform reliability. In this work, we propose a normal-guided sparse depth sampling strategy that leverages PCA-based surface normal estimation on the RGB-D point cloud to compute a per-pixel depth reliability measure. The sparse depth samples are then drawn according to this reliability distribution. We integrate this sampling method with the Marigold-DC diffusion-based depth completion model and evaluate it on NYU Depth v2 using the standard metrics. Experiments show that our geometry-aware sparse depth improves accuracy, reduces artifacts near edges and discontinuities, and produces more realistic training conditions that better reflect real sensor behavior.
>
---
#### [new 040] Accelerated Rotation-Invariant Convolution for UAV Image Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对无人机图像分割中目标方向多变的问题，提出一种GPU优化的旋转不变卷积方法。通过共享旋转滤波器的数据，减少计算冗余和内存访问，提升效率与能效，并集成到U-Net中，在保持精度的同时显著加速训练并降低能耗。**

- **链接: [https://arxiv.org/pdf/2512.08888v1](https://arxiv.org/pdf/2512.08888v1)**

> **作者:** Manduhu Manduhu; Alexander Dow; Gerard Dooly; James Riordan
>
> **摘要:** Rotation invariance is essential for precise, object-level segmentation in UAV aerial imagery, where targets can have arbitrary orientations and exhibit fine-scale details. Conventional segmentation architectures like U-Net rely on convolution operators that are not rotation-invariant, leading to degraded segmentation accuracy across varying viewpoints. Rotation invariance can be achieved by expanding the filter bank across multiple orientations; however, this will significantly increase computational cost and memory traffic. In this paper, we introduce a GPU-optimized rotation-invariant convolution framework that eliminates the traditional data-lowering (im2col) step required for matrix-multiplication-based convolution. By exploiting structured data sharing among symmetrically rotated filters, our method achieves multi-orientation convolution with greatly reduced memory traffic and computational redundancy. We further generalize the approach to accelerate convolution with arbitrary (non-symmetric) rotation angles. Across extensive benchmarks, the proposed convolution achieves 20--55% faster training and 15--45% lower energy consumption than CUDNN, while maintaining accuracy comparable to state-of-the-art rotation-invariant methods. In the eight-orientation setting, our approach achieves up to 45% speedup and 41% energy savings on 256\(\times\)256 inputs, and 32% speedup and 23% lower energy usage on 1024\(\times\)1024 inputs. Integrated into a U-Net segmentation model, the framework yields up to 6% improvement in accuracy over the non-rotation-aware baseline. These results demonstrate that the proposed method provides an effective and highly efficient alternative to existing rotation-invariant CNN frameworks.
>
---
#### [new 041] Prismatic World Model: Learning Compositional Dynamics for Planning in Hybrid Systems
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究模型-based规划中的混合动力系统建模问题，提出Prismatic World Model（PRISM-WM），通过上下文感知的专家混合架构与正交化约束，分解并精确建模离散模式切换，提升长视野规划的准确性。**

- **链接: [https://arxiv.org/pdf/2512.08411v1](https://arxiv.org/pdf/2512.08411v1)**

> **作者:** Mingwei Li; Xiaoyuan Zhang; Chengwei Yang; Zilong Zheng; Yaodong Yang
>
> **摘要:** Model-based planning in robotic domains is fundamentally challenged by the hybrid nature of physical dynamics, where continuous motion is punctuated by discrete events such as contacts and impacts. Conventional latent world models typically employ monolithic neural networks that enforce global continuity, inevitably over-smoothing the distinct dynamic modes (e.g., sticking vs. sliding, flight vs. stance). For a planner, this smoothing results in catastrophic compounding errors during long-horizon lookaheads, rendering the search process unreliable at physical boundaries. To address this, we introduce the Prismatic World Model (PRISM-WM), a structured architecture designed to decompose complex hybrid dynamics into composable primitives. PRISM-WM leverages a context-aware Mixture-of-Experts (MoE) framework where a gating mechanism implicitly identifies the current physical mode, and specialized experts predict the associated transition dynamics. We further introduce a latent orthogonalization objective to ensure expert diversity, effectively preventing mode collapse. By accurately modeling the sharp mode transitions in system dynamics, PRISM-WM significantly reduces rollout drift. Extensive experiments on challenging continuous control benchmarks, including high-dimensional humanoids and diverse multi-task settings, demonstrate that PRISM-WM provides a superior high-fidelity substrate for trajectory optimization algorithms (e.g., TD-MPC), proving its potential as a powerful foundational model for next-generation model-based agents.
>
---
#### [new 042] Disturbance-Free Surgical Video Generation from Multi-Camera Shadowless Lamps for Open Surgery
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文致力于自动化生成无遮挡的开放手术视频。针对多摄像头无影灯系统中因调整灯光导致的图像错位问题，提出自动校正帧对齐并选择最优无遮挡视角的方法，生成稳定、舒适、高质量的手术视频，并通过用户研究验证其优越性。**

- **链接: [https://arxiv.org/pdf/2512.08577v1](https://arxiv.org/pdf/2512.08577v1)**

> **作者:** Yuna Kato; Shohei Mori; Hideo Saito; Yoshifumi Takatsume; Hiroki Kajita; Mariko Isogawa
>
> **摘要:** Video recordings of open surgeries are greatly required for education and research purposes. However, capturing unobstructed videos is challenging since surgeons frequently block the camera field of view. To avoid occlusion, the positions and angles of the camera must be frequently adjusted, which is highly labor-intensive. Prior work has addressed this issue by installing multiple cameras on a shadowless lamp and arranging them to fully surround the surgical area. This setup increases the chances of some cameras capturing an unobstructed view. However, manual image alignment is needed in post-processing since camera configurations change every time surgeons move the lamp for optimal lighting. This paper aims to fully automate this alignment task. The proposed method identifies frames in which the lighting system moves, realigns them, and selects the camera with the least occlusion to generate a video that consistently presents the surgical field from a fixed perspective. A user study involving surgeons demonstrated that videos generated by our method were superior to those produced by conventional methods in terms of the ease of confirming the surgical area and the comfort during video viewing. Additionally, our approach showed improvements in video quality over existing techniques. Furthermore, we implemented several synthesis options for the proposed view-synthesis method and conducted a user study to assess surgeons' preferences for each option.
>
---
## 更新

#### [replaced 001] Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion
- **分类: cs.RO**

- **简介: 该论文研究多传感器融合定位任务，旨在解决单相机LIVO系统无法充分利用LiDAR几何信息的问题。作者提出Omni-LIVO，通过多相机系统实现跨视图直接对齐，并引入改进的ESIKF融合框架，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.15673v2](https://arxiv.org/pdf/2509.15673v2)**

> **作者:** Yinong Cao; Xin He; Yuwei Chen; Chenyang Zhang; Chengyu Pu; Bingtao Wang; Kaile Wu; Shouzheng Zhu; Fei Han; Shijie Liu; Chunlai Li; Jianyu Wang
>
> **摘要:** Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but existing LiDAR-inertial-visual odometry (LIVO) systems generally rely on a single camera, limiting their ability to fully exploit LiDAR-derived depth for photometric alignment and scene colorization. We present Omni-LIVO, a tightly coupled multi-camera LIVO system that leverages multi-view observations to comprehensively utilize LiDAR geometric information across extended spatial regions. Omni-LIVO introduces a Cross-View direct alignment strategy that maintains photometric consistency across non-overlapping views, and extends the Error-State Iterated Kalman Filter (ESIKF) with multi-view updates and adaptive covariance. The system is evaluated on public benchmarks and our custom dataset, showing improved accuracy and robustness over state-of-the-art LIVO, LIO, and visual-inertial SLAM baselines. Code and dataset will be released upon publication.
>
---
#### [replaced 002] Enhancing the NAO: Extending Capabilities of Legacy Robots for Long-Term Research
- **分类: cs.RO; cs.HC; eess.AS**

- **简介: 该论文致力于延长老旧机器人研究寿命，解决因厂商停支持导致的功能落后问题。作者改造NAO机器人，集成新型传感器与计算单元，提升感知与对话能力，保持原有交互特性，在不增加延迟的前提下显著提高对话质量与用户偏好。**

- **链接: [https://arxiv.org/pdf/2509.17760v3](https://arxiv.org/pdf/2509.17760v3)**

> **作者:** Austin Wilson; Sahar Kapasi; Zane Greene; Alexis E. Block
>
> **摘要:** Legacy (unsupported) robotic platforms often lose research utility when manufacturer support ends, preventing integration of modern sensing, speech, and interaction capabilities. We present the Enhanced NAO, a revitalized version of Aldebaran's NAO robot featuring upgraded beamforming microphones, RGB-D and thermal cameras, and additional compute resources in a fully self-contained package. This system combines cloud-based and local models for perception and dialogue, while preserving the NAO's expressive body and behaviors. In a pilot user study validating conversational performance, the Enhanced NAO delivered significantly higher conversational quality and elicited stronger user preference compared to the NAO AI Edition, without increasing response latency. The added visual and thermal sensing modalities established a foundation for future perception-driven interaction. Beyond this implementation, our framework provides a platform-agnostic strategy for extending the lifespan and research utility of legacy robots, ensuring they remain valuable tools for human-robot interaction.
>
---
#### [replaced 003] Fitts' List Revisited: An Empirical Study on Function Allocation in a Two-Agent Physical Human-Robot Collaborative Position/Force Task
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究物理人机协作中的功能分配问题，验证Fitts原则在位置/力控制分配中的适用性。通过用户实验比较四种分配方式，发现人控位置、机控力时性能更优，且用户体验更好，支持Fitts原则在静态分配中的应用，并揭示了自主感知的关键影响。**

- **链接: [https://arxiv.org/pdf/2505.04722v2](https://arxiv.org/pdf/2505.04722v2)**

> **作者:** Nicky Mol; J. Micah Prendergast; David A. Abbink; Luka Peternel
>
> **备注:** 8 pages, 6 figures, published in IEEE Robotics and Automation Letters, col. 11, no. 1, January 2026
>
> **摘要:** In this letter, we investigate whether classical function allocation-the principle of assigning tasks to either a human or a machine-holds for physical Human-Robot Collaboration, which is important for providing insights for Industry 5.0 to guide how to best augment rather than replace workers. This study empirically tests the applicability of Fitts' List within physical Human-Robot Collaboration, by conducting a user study (N=26, within-subject design) to evaluate four distinct allocations of position/force control between human and robot in an abstract blending task. We hypothesize that the function in which humans control the position achieves better performance and receives higher user ratings. When allocating position control to the human and force control to the robot, compared to the opposite case, we observed a significant improvement in preventing overblending. This was also perceived better in terms of physical demand and overall system acceptance, while participants experienced greater autonomy, more engagement and less frustration. An interesting insight was that the supervisory role (when the robot controls both position and force) was rated second best in terms of subjective acceptance. Another surprising insight was that if position control was delegated to the robot, the participants perceived much lower autonomy than when the force control was delegated to the robot. These findings empirically support applying Fitts' principles to static function allocation for physical collaboration, while also revealing important nuanced user experience trade-offs, particularly regarding perceived autonomy when delegating position control.
>
---
#### [replaced 004] 2Fast-2Lamaa: Large-Scale Lidar-Inertial Localization and Mapping with Continuous Distance Fields
- **分类: cs.RO**

- **简介: 该论文提出2Fast-2Lamaa，属于激光-惯性里程计与定位建图任务，旨在解决运动失真校正与长期几何一致性问题。通过IMU连续预积分实现无先验的扫描校正，并结合高斯过程构建连续距离场地图，提升定位建图精度。**

- **链接: [https://arxiv.org/pdf/2410.05433v2](https://arxiv.org/pdf/2410.05433v2)**

> **作者:** Cedric Le Gentil; Raphael Falque; Daniil Lisus; Timothy D. Barfoot
>
> **摘要:** This paper introduces 2Fast-2Lamaa, a lidar-inertial state estimation framework for odometry, mapping, and localization. Its first key component is the optimization-based undistortion of lidar scans, which uses continuous IMU preintegration to model the system's pose at every lidar point timestamp. The continuous trajectory over 100-200ms is parameterized only by the initial scan conditions (linear velocity and gravity orientation) and IMU biases, yielding eleven state variables. These are estimated by minimizing point-to-line and point-to-plane distances between lidar-extracted features without relying on previous estimates, resulting in a prior-less motion-distortion correction strategy. Because the method performs local state estimation, it directly provides scan-to-scan odometry. To maintain geometric consistency over longer periods, undistorted scans are used for scan-to-map registration. The map representation employs Gaussian Processes to form a continuous distance field, enabling point-to-surface distance queries anywhere in space. Poses of the undistorted scans are refined by minimizing these distances through non-linear least-squares optimization. For odometry and mapping, the map is built incrementally in real time; for pure localization, existing maps are reused. The incremental map construction also includes mechanisms for removing dynamic objects. We benchmark 2Fast-2Lamaa on 250km (over 10h) of public and self-collected datasets from both automotive and handheld systems. The framework achieves state-of-the-art performance across diverse and challenging scenarios, reaching odometry and localization errors as low as 0.27% and 0.06 m, respectively. The real-time implementation is publicly available at https://github.com/clegenti/2fast2lamaa.
>
---
#### [replaced 005] Khalasi: Energy-Efficient Navigation for Surface Vehicles in Vortical Flow Fields
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究自主水面航行器在涡旋流场中的节能导航问题。针对传统方法因流场部分可观而失效的难题，提出基于软演员-评论家算法的强化学习框架，仅用局部速度测量实现高效路径规划，显著降低能耗30%-50%。**

- **链接: [https://arxiv.org/pdf/2512.06912v2](https://arxiv.org/pdf/2512.06912v2)**

> **作者:** Rushiraj Gadhvi; Sandeep Manjanna
>
> **备注:** Under Review for International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** For centuries, khalasi (Gujarati for sailor) have skillfully harnessed ocean currents to navigate vast waters with minimal effort. Emulating this intuition in autonomous systems remains a significant challenge, particularly for Autonomous Surface Vehicles tasked with long duration missions under strict energy budgets. In this work, we present a learning-based approach for energy-efficient surface vehicle navigation in vortical flow fields, where partial observability often undermines traditional path-planning methods. We present an end to end reinforcement learning framework based on Soft Actor Critic that learns flow-aware navigation policies using only local velocity measurements. Through extensive evaluation across diverse and dynamically rich scenarios, our method demonstrates substantial energy savings and robust generalization to previously unseen flow conditions, offering a promising path toward long term autonomy in ocean environments. The navigation paths generated by our proposed approach show an improvement in energy conservation 30 to 50 percent compared to the existing state of the art techniques.
>
---
#### [replaced 006] Haptic-based Complementary Filter for Rigid Body Rotations
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究3D刚体旋转的位姿估计问题，解决传统方法难以融合触觉信息的挑战。提出一种基于触觉与视觉的SO(3)互补滤波框架，利用超二次曲面几何与群对称性，实现稳定、鲁棒的旋转估计。**

- **链接: [https://arxiv.org/pdf/2504.14570v2](https://arxiv.org/pdf/2504.14570v2)**

> **作者:** Amit Kumar; Domenico Campolo; Ravi N. Banavar
>
> **备注:** 7 pages, 7 figures; Updated filter design; Submitted to IFAC for possible publication
>
> **摘要:** The non-commutative nature of 3D rotations poses well-known challenges in generalizing planar problems to three-dimensional ones, even more so in contact-rich tasks where haptic information (i.e., forces/torques) is involved. In this sense, not all learning-based algorithms that are currently available generalize to 3D orientation estimation. Non-linear filters defined on $\mathbf{\mathbb{SO}(3)}$ are widely used with inertial measurement sensors; however, none of them have been used with haptic measurements. This paper presents a unique complementary filtering framework that interprets the geometric shape of objects in the form of superquadrics, exploits the symmetry of $\mathbf{\mathbb{SO}(3)}$, and uses force and vision sensors as measurements to provide an estimate of orientation. The framework's robustness and almost global stability are substantiated by a set of experiments on a dual-arm robotic setup.
>
---
#### [replaced 007] DIVER: Reinforced Diffusion Breaks Imitation Bottlenecks in End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对端到端自动驾驶中模仿学习因单一专家示范导致的保守与多样性不足问题，提出DIVER框架，结合强化学习与扩散模型生成多样化可行轨迹，并设计新多样性指标评估，提升复杂场景下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.04049v3](https://arxiv.org/pdf/2507.04049v3)**

> **作者:** Ziying Song; Lin Liu; Hongyu Pan; Bencheng Liao; Mingzhe Guo; Lei Yang; Yongchang Zhang; Shaoqing Xu; Caiyan Jia; Yadan Luo
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Most end-to-end autonomous driving methods rely on imitation learning from single expert demonstrations, often leading to conservative and homogeneous behaviors that limit generalization in complex real-world scenarios. In this work, we propose DIVER, an end-to-end driving framework that integrates reinforcement learning with diffusion-based generation to produce diverse and feasible trajectories. At the core of DIVER lies a reinforced diffusion-based generation mechanism. First, the model conditions on map elements and surrounding agents to generate multiple reference trajectories from a single ground-truth trajectory, alleviating the limitations of imitation learning that arise from relying solely on single expert demonstrations. Second, reinforcement learning is employed to guide the diffusion process, where reward-based supervision enforces safety and diversity constraints on the generated trajectories, thereby enhancing their practicality and generalization capability. Furthermore, to address the limitations of L2-based open-loop metrics in capturing trajectory diversity, we propose a novel Diversity metric to evaluate the diversity of multi-mode predictions.Extensive experiments on the closed-loop NAVSIM and Bench2Drive benchmarks, as well as the open-loop nuScenes dataset, demonstrate that DIVER significantly improves trajectory diversity, effectively addressing the mode collapse problem inherent in imitation learning.
>
---
#### [replaced 008] Humanoid Whole-Body Badminton via Multi-Stage Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文研究人形机器人打羽毛球任务，解决动态环境中全身协调控制问题。提出多阶段强化学习框架，无需先验动作或专家示范，实现脚步移动与挥拍击球协同，并结合轨迹预测或预测-free方法部署，在仿真与真实场景中均实现高速精准回击。**

- **链接: [https://arxiv.org/pdf/2511.11218v2](https://arxiv.org/pdf/2511.11218v2)**

> **作者:** Chenhao Liu; Leyun Jiang; Yibo Wang; Kairan Yao; Jinchen Fu; Xiaoyu Ren
>
> **备注:** Project Page: https://humanoid-badminton.github.io/Humanoid-Whole-Body-Badminton-via-Multi-Stage-Reinforcement-Learning
>
> **摘要:** Humanoid robots have demonstrated strong capabilities for interacting with static scenes across locomotion, manipulation, and more challenging loco-manipulation tasks. Yet the real world is dynamic, and quasi-static interactions are insufficient to cope with diverse environmental conditions. As a step toward more dynamic interaction scenarios, we present a reinforcement-learning-based training pipeline that produces a unified whole-body controller for humanoid badminton, enabling coordinated lower-body footwork and upper-body striking without motion priors or expert demonstrations. Training follows a three-stage curriculum: first footwork acquisition, then precision-guided racket swing generation, and finally task-focused refinement, yielding motions in which both legs and arms serve the hitting objective. For deployment, we incorporate an Extended Kalman Filter (EKF) to estimate and predict shuttlecock trajectories for target striking. We also introduce a prediction-free variant that dispenses with EKF and explicit trajectory prediction. To validate the framework, we conduct five sets of experiments in both simulation and the real world. In simulation, two robots sustain a rally of 21 consecutive hits. Moreover, the prediction-free variant achieves successful hits with comparable performance relative to the target-known policy. In real-world tests, both prediction and controller modules exhibit high accuracy, and on-court hitting achieves an outgoing shuttle speed up to 19.1 m/s with a mean return landing distance of 4 m. These experimental results show that our proposed training scheme can deliver highly dynamic while precise goal striking in badminton, and can be adapted to more dynamics-critical domains.
>
---
#### [replaced 009] Control Your Robot: A Unified System for Robot Control and Policy Deployment
- **分类: cs.RO**

- **简介: 该论文聚焦机器人跨平台控制难题，提出统一框架Control Your Robot。通过标准化流程、模块化设计和统一API，实现多平台数据采集与策略部署的集成，支持遥操作、轨迹回放及多模态感知，验证了其在低延迟数据收集和策略学习中的有效性。**

- **链接: [https://arxiv.org/pdf/2509.23823v2](https://arxiv.org/pdf/2509.23823v2)**

> **作者:** Tian Nian; Weijie Ke; Shaolong Zhu; Bingshan Hu
>
> **备注:** Code: https://github.com/Tian-Nian/control_your_robot
>
> **摘要:** Cross-platform robot control remains difficult because hardware interfaces, data formats, and control paradigms vary widely, which fragments toolchains and slows deployment. To address this, we present Control Your Robot, a modular, general-purpose framework that unifies data collection and policy deployment across diverse platforms. The system reduces fragmentation through a standardized workflow with modular design, unified APIs, and a closed-loop architecture. It supports flexible robot registration, dual-mode control with teleoperation and trajectory playback, and seamless integration from multimodal data acquisition to inference. Experiments on single-arm and dual-arm systems show efficient, low-latency data collection and effective support for policy learning with imitation learning and vision-language-action models. Policies trained on data gathered by Control Your Robot match expert demonstrations closely, indicating that the framework enables scalable and reproducible robot learning across platforms.
>
---
#### [replaced 010] Training-Time Action Conditioning for Efficient Real-Time Chunking
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究实时机器人控制中的动作分块任务，旨在降低推理时因图像修复导致的计算延迟。作者提出在训练时模拟延迟并直接条件于历史动作，避免推理时开销，方法简单高效，实验证明其在仿真和真实任务中均优于传统方法。**

- **链接: [https://arxiv.org/pdf/2512.05964v2](https://arxiv.org/pdf/2512.05964v2)**

> **作者:** Kevin Black; Allen Z. Ren; Michael Equi; Sergey Levine
>
> **摘要:** Real-time chunking (RTC) enables vision-language-action models (VLAs) to generate smooth, reactive robot trajectories by asynchronously predicting action chunks and conditioning on previously committed actions via inference-time inpainting. However, this inpainting method introduces computational overhead that increases inference latency. In this work, we propose a simple alternative: simulating inference delay at training time and conditioning on action prefixes directly, eliminating any inference-time overhead. Our method requires no modifications to the model architecture or robot runtime, and can be implemented with only a few additional lines of code. In simulated experiments, we find that training-time RTC outperforms inference-time RTC at higher inference delays. In real-world experiments on box building and espresso making tasks with the $π_{0.6}$ VLA, we demonstrate that training-time RTC maintains both task performance and speed parity with inference-time RTC while being computationally cheaper. Our results suggest that training-time action conditioning is a practical drop-in replacement for inference-time inpainting in real-time robot control.
>
---
#### [replaced 011] RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation
- **分类: cs.RO**

- **简介: 该论文针对机器人在线装箱任务，解决因标准不一、数据失真和评估不足导致的研究瓶颈。提出RoboBPP，集成物理仿真、真实工业数据集、新评估指标及开源平台，支持可复现与可扩展的算法评测。**

- **链接: [https://arxiv.org/pdf/2512.04415v2](https://arxiv.org/pdf/2512.04415v2)**

> **作者:** Zhoufeng Wang; Hang Zhao; Juzhan Xu; Shishun Zhang; Zeyu Xiong; Ruizhen Hu; Chenyang Zhu; Kai Xu
>
> **备注:** Under review at the International Journal of Robotics Research (IJRR)
>
> **摘要:** Physical feasibility in 3D bin packing is a key requirement in modern industrial logistics and robotic automation. With the growing adoption of industrial automation, online bin packing has gained increasing attention. However, inconsistencies in problem settings, test datasets, and evaluation metrics have hindered progress in the field, and there is a lack of a comprehensive benchmarking system. Direct testing on real hardware is costly, and building a realistic simulation environment is also challenging. To address these limitations, we introduce RoboBPP, a benchmarking system designed for robotic online bin packing. RoboBPP integrates a physics-based simulator to assess physical feasibility. In our simulation environment, we introduce a robotic arm and boxes at real-world scales to replicate real industrial packing workflows. By simulating conditions that arise in real industrial applications, we ensure that evaluated algorithms are practically deployable. In addition, prior studies often rely on synthetic datasets whose distributions differ from real-world industrial data. To address this issue, we collect three datasets from real industrial workflows, including assembly-line production, logistics packing, and furniture manufacturing. The benchmark comprises three carefully designed test settings and extends existing evaluation metrics with new metrics for structural stability and operational safety. We design a scoring system and derive a range of insights from the evaluation results. RoboBPP is fully open-source and is equipped with visualization tools and an online leaderboard, providing a reproducible and extensible foundation for future research and industrial applications (https://robot-bin-packing-benchmark.github.io).
>
---
#### [replaced 012] CDKFormer: Contextual Deviation Knowledge-Based Transformer for Long-Tail Trajectory Prediction
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究自动驾驶中的长尾轨迹预测任务，旨在提升罕见场景下的预测鲁棒性。通过分析异常运动特征，提出CDKFormer模型，融合场景上下文与动态偏差信息，实现更精准的多模态轨迹预测。**

- **链接: [https://arxiv.org/pdf/2503.12695v2](https://arxiv.org/pdf/2503.12695v2)**

> **作者:** Yuansheng Lian; Ke Zhang; Meng Li
>
> **摘要:** Predicting the future movements of surrounding vehicles is essential for ensuring the safe operation and efficient navigation of autonomous vehicles (AVs) in urban traffic environments. Existing vehicle trajectory prediction methods primarily focus on improving overall performance, yet they struggle to address long-tail scenarios effectively. This limitation often leads to poor predictions in rare cases, significantly increasing the risk of safety incidents. Taking Argoverse 2 motion forecasting dataset as an example, we first investigate the long-tail characteristics in trajectory samples from two perspectives, individual motion and group interaction, and deriving deviation features to distinguish abnormal from regular scenarios. On this basis, we propose CDKFormer, a Contextual Deviation Knowledge-based Transformer model for long-tail trajectory prediction. CDKFormer integrates an attention-based scene context fusion module to encode spatiotemporal interaction and road topology. An additional deviation feature fusion module is proposed to capture the dynamic deviations in the target vehicle status. We further introduce a dual query-based decoder, supported by a multi-stream decoder block, to sequentially decode heterogeneous scene deviation features and generate multimodal trajectory predictions. Extensive experiments demonstrate that CDKFormer achieves state-of-the-art performance, significantly enhancing prediction accuracy and robustness for long-tailed trajectories compared to existing methods, thus advancing the reliability of AVs in complex real-world environments.
>
---
#### [replaced 013] Obstacle Avoidance of UAV in Dynamic Environments Using Direction and Velocity-Adaptive Artificial Potential Field
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究无人机在动态环境中的避障任务，旨在解决传统人工势场法易陷入局部极小及忽略障碍物运动状态的问题。提出方向与相对速度加权的人工势场，并结合模型预测控制实现安全、平滑的避障。**

- **链接: [https://arxiv.org/pdf/2512.07609v2](https://arxiv.org/pdf/2512.07609v2)**

> **作者:** Nikita Vaibhav Pavle; Shrreya Rajneesh; Rakesh Kumar Sahoo; Manoranjan Sinha
>
> **摘要:** The conventional Artificial Potential Field (APF) is fundamentally limited by the local minima issue and its inability to account for the kinematics of moving obstacles. This paper addresses the critical challenge of autonomous collision avoidance for Unmanned Aerial Vehicles (UAVs) operating in dynamic and cluttered airspace by proposing a novel Direction and Relative Velocity Weighted Artificial Potential Field (APF). In this approach, a bounded weighting function, $ω(θ,v_{e})$, is introduced to dynamically scale the repulsive potential based on the direction and velocity of the obstacle relative to the UAV. This robust APF formulation is integrated within a Model Predictive Control (MPC) framework to generate collision-free trajectories while adhering to kinematic constraints. Simulation results demonstrate that the proposed method effectively resolves local minima and significantly enhances safety by enabling smooth, predictive avoidance maneuvers. The system ensures superior path integrity and reliable performance, confirming its viability for autonomous navigation in complex environments.
>
---
#### [replaced 014] db-LaCAM: Fast and Scalable Multi-Robot Kinodynamic Motion Planning with Discontinuity-Bounded Search and Lightweight MAPF
- **分类: cs.RO**

- **简介: 该论文研究多机器人运动规划，旨在解决现有动力学规划器扩展性差、速度慢的问题。提出db-LaCAM方法，结合轻量级MAPF与动力学感知的运动原语搜索，在保证解质量的同时显著提升规模与效率，支持多种机器人动态并验证于实物实验。**

- **链接: [https://arxiv.org/pdf/2512.06796v2](https://arxiv.org/pdf/2512.06796v2)**

> **作者:** Akmaral Moldagalieva; Keisuke Okumura; Amanda Prorok; Wolfgang Hönig
>
> **摘要:** State-of-the-art multi-robot kinodynamic motion planners struggle to handle more than a few robots due to high computational burden, which limits their scalability and results in slow planning time. In this work, we combine the scalability and speed of modern multi-agent path finding (MAPF) algorithms with the dynamic-awareness of kinodynamic planners to address these limitations. To this end, we propose discontinuity-Bounded LaCAM (db-LaCAM), a planner that utilizes a precomputed set of motion primitives that respect robot dynamics to generate horizon-length motion sequences, while allowing a user-defined discontinuity between successive motions. The planner db-LaCAM is resolution-complete with respect to motion primitives and supports arbitrary robot dynamics. Extensive experiments demonstrate that db-LaCAM scales efficiently to scenarios with up to 50 robots, achieving up to ten times faster runtime compared to state-of-the-art planners, while maintaining comparable solution quality. The approach is validated in both 2D and 3D environments with dynamics such as the unicycle and 3D double integrator. We demonstrate the safe execution of trajectories planned with db-LaCAM in two distinct physical experiments involving teams of flying robots and car-with-trailer robots.
>
---
#### [replaced 015] Incremental Generalized Hybrid A*
- **分类: cs.RO**

- **简介: 该论文属路径规划任务，旨在解决复杂动力学下实时运动规划效率低的问题。提出Incremental Generalized Hybrid A*（IGHA*），通过动态组织节点扩展，避免刚性剪枝，提升搜索效率，较Hybrid A*显著减少扩展次数，实现快速鲁棒的实时规划。**

- **链接: [https://arxiv.org/pdf/2508.13392v3](https://arxiv.org/pdf/2508.13392v3)**

> **作者:** Sidharth Talia; Oren Salzman; Siddhartha Srinivasa
>
> **备注:** 8 pages, 7 figures, Accepted to IEEE RA-L, Nov 2025
>
> **摘要:** We address the problem of efficiently organizing search over very large trees, which arises in many applications ranging from autonomous driving to aerial vehicles. Here, we are motivated by off-road autonomy, where real-time planning is essential. Classical approaches use graphs of motion primitives and exploit dominance to mitigate the curse of dimensionality and prune expansions efficiently. However, for complex dynamics, repeatedly solving two-point boundary-value problems makes graph construction too slow for fast kinodynamic planning. Hybrid A* (HA*) addressed this challenge by searching over a tree of motion primitives and introducing approximate pruning using a grid-based dominance check. However, choosing the grid resolution is difficult: too coarse risks failure, while too fine leads to excessive expansions and slow planning. We propose Incremental Generalized Hybrid A* (IGHA*), an anytime tree-search framework that dynamically organizes vertex expansions without rigid pruning. IGHA* provably matches or outperforms HA*. For both on-road kinematic and off-road kinodynamic planning queries for a car-like robot, variants of IGHA* use 6x fewer expansions to the best solution compared to an optimized version of HA* (HA*M, an internal baseline). In simulated off-road experiments in a high-fidelity simulator, IGHA* outperforms HA*M when both are used in the loop with a model predictive controller. We demonstrate real-time performance both in simulation and on a small-scale off-road vehicle, enabling fast, robust planning under complex dynamics. Website: https://personalrobotics.github.io/IGHAStar/
>
---
#### [replaced 016] Unifying Entropy Regularization in Optimal Control: From and Back to Classical Objectives via Iterated Soft Policies and Path Integral Solutions
- **分类: math.OC; cs.LG; cs.RO; eess.SY**

- **简介: 该论文研究熵正则化在最优控制中的统一框架，旨在通过KL正则化连接经典控制问题。提出广义优化模型，分离策略与状态转移的KL惩罚项，可退化为SOC、RSOC等经典问题，并揭示软策略版本可通过迭代收敛至原问题解，特定参数下具备路径积分解与线性Bellman方程等优良性质。**

- **链接: [https://arxiv.org/pdf/2512.06109v2](https://arxiv.org/pdf/2512.06109v2)**

> **作者:** Ajinkya Bhole; Mohammad Mahmoudi Filabadi; Guillaume Crevecoeur; Tom Lefebvre
>
> **备注:** Corrected "DRO" to "DRC" and fixed theorem numbering throughout paper
>
> **摘要:** This paper develops a unified perspective on several stochastic optimal control formulations through the lens of Kullback-Leibler regularization. We propose a central problem that separates the KL penalties on policies and transitions, assigning them independent weights, thereby generalizing the standard trajectory-level KL-regularization commonly used in probabilistic and KL-regularized control. This generalized formulation acts as a generative structure allowing to recover various control problems. These include the classical Stochastic Optimal Control (SOC), Risk-Sensitive Optimal Control (RSOC), and their policy-based KL-regularized counterparts. The latter we refer to as soft-policy SOC and RSOC, facilitating alternative problems with tractable solutions. Beyond serving as regularized variants, we show that these soft-policy formulations majorize the original SOC and RSOC problem. This means that the regularized solution can be iterated to retrieve the original solution. Furthermore, we identify a structurally synchronized case of the risk-seeking soft-policy RSOC formulation, wherein the policy and transition KL-regularization weights coincide. Remarkably, this specific setting gives rise to several powerful properties such as a linear Bellman equation, path integral solution, and, compositionality, thereby extending these computationally favourable properties to a broad class of control problems.
>
---
#### [replaced 017] Towards Task-Oriented Flying: Framework, Infrastructure, and Principles
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对无人机在非结构化环境中的自主飞行任务，解决端到端学习控制缺乏系统设计与统一基础设施的问题。提出任务导向框架，整合仿真、训练与部署，支持可复现的深度强化学习，实现强鲁棒性和真实环境迁移。**

- **链接: [https://arxiv.org/pdf/2504.15129v2](https://arxiv.org/pdf/2504.15129v2)**

> **作者:** Kangyao Huang; Hao Wang; Jingyu Chen; Jintao Chen; Yu Luo; Di Guo; Xiangkui Zhang; Xiangyang Ji; Huaping Liu
>
> **摘要:** Deploying robot learning methods to aerial robots in unstructured environments remains both challenging and promising. While recent advances in deep reinforcement learning (DRL) have enabled end-to-end flight control, the field still lacks systematic design guidelines and a unified infrastructure to support reproducible training and real-world deployment. We present a task-oriented framework for end-to-end DRL in quadrotors that integrates design principles for complex task specification and reveals the interdependencies among simulated task definition, training design principles, and physical deployment. Our framework involves software infrastructure, hardware platforms, and open-source firmware to support a full-stack learning infrastructure and workflow. Extensive empirical results demonstrate robust flight and sim-to-real generalization under real-world disturbances. By reducing the entry barrier for deploying learning-based controllers on aerial robots, our work lays a practical foundation for advancing autonomous flight in dynamic and unstructured environments.
>
---
#### [replaced 018] Botany Meets Robotics in Alpine Scree Monitoring
- **分类: cs.RO**

- **简介: 该论文提出利用足式机器人ANYmal C辅助高山碎石坡植被监测，解决传统人工调查耗时、危险且低效的问题。通过两年两次野外实验，结合深度学习实现植物识别，提升数据采集效率与频率，推动生态监测自动化。**

- **链接: [https://arxiv.org/pdf/2511.12526v2](https://arxiv.org/pdf/2511.12526v2)**

> **作者:** Davide De Benedittis; Giovanni Di Lorenzo; Franco Angelini; Barbara Valle; Marina Serena Borgatti; Paolo Remagnino; Marco Caccianiga; Manolo Garabini
>
> **备注:** 19 pages, 13 figures
>
> **摘要:** According to the European Union's Habitat Directive, habitat monitoring plays a critical role in response to the escalating problems posed by biodiversity loss and environmental degradation. Scree habitats, hosting unique and often endangered species, face severe threats from climate change due to their high-altitude nature. Traditionally, their monitoring has required highly skilled scientists to conduct extensive fieldwork in remote, potentially hazardous locations, making the process resource-intensive and time-consuming. This paper presents a novel approach for scree habitat monitoring using a legged robot to assist botanists in data collection and species identification. Specifically, we deployed the ANYmal C robot in the Italian Alpine bio-region in two field campaigns spanning two years and leveraged deep learning to detect and classify key plant species of interest. Our results demonstrate that agile legged robots can navigate challenging terrains and increase the frequency and efficiency of scree monitoring. When paired with traditional phytosociological surveys performed by botanists, this robotics-assisted protocol not only streamlines field operations but also enhances data acquisition, storage, and usage. The outcomes of this research contribute to the evolving landscape of robotics in environmental science, paving the way for a more comprehensive and sustainable approach to habitat monitoring and preservation.
>
---
#### [replaced 019] MARL Warehouse Robots
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究多智能体强化学习（MARL）在协作式仓库机器人中的应用，旨在解决多机器人协同任务效率问题。比较了QMIX与IPPO算法，在RWARE和Unity 3D环境中验证QMIX性能更优但需精细调参，成功实现1M步内的稳定包裹递送。**

- **链接: [https://arxiv.org/pdf/2512.04463v2](https://arxiv.org/pdf/2512.04463v2)**

> **作者:** Price Allman; Lian Thang; Dre Simmons; Salmon Riaz
>
> **备注:** 5 pages.Project documentation: https://pallman14.github.io/MARL-QMIX-Warehouse-Robots/
>
> **摘要:** We present a comparative study of multi-agent reinforcement learning (MARL) algorithms for cooperative warehouse robotics. We evaluate QMIX and IPPO on the Robotic Warehouse (RWARE) environment and a custom Unity 3D simulation. Our experiments reveal that QMIX's value decomposition significantly outperforms independent learning approaches (achieving 3.25 mean return vs. 0.38 for advanced IPPO), but requires extensive hyperparameter tuning -- particularly extended epsilon annealing (5M+ steps) for sparse reward discovery. We demonstrate successful deployment in Unity ML-Agents, achieving consistent package delivery after 1M training steps. While MARL shows promise for small-scale deployments (2-4 robots), significant scaling challenges remain. Code and analyses: https://pallman14.github.io/MARL-QMIX-Warehouse-Robots/
>
---
#### [replaced 020] Capability-Driven Skill Generation with LLMs: A RAG-Based Approach for Reusing Existing Libraries and Interfaces
- **分类: cs.AI; cs.RO; cs.SE**

- **简介: 该论文属于自动化系统中的技能生成任务，旨在解决技能实现开发耗时且困难的问题。作者提出一种基于RAG的LLM方法，将能力作为合同，结合自然语言输入与现有库和接口，自动生成跨语言的可执行技能代码，并在ROS 2机器人上验证了可行性。**

- **链接: [https://arxiv.org/pdf/2505.03295v2](https://arxiv.org/pdf/2505.03295v2)**

> **作者:** Luis Miguel Vieira da Silva; Aljosha Köcher; Nicolas König; Felix Gehlhoff; Alexander Fay
>
> **备注:** \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Modern automation systems increasingly rely on modular architectures, with capabilities and skills as one solution approach. Capabilities define the functions of resources in a machine-readable form and skills provide the concrete implementations that realize those capabilities. However, the development of a skill implementation conforming to a corresponding capability remains a time-consuming and challenging task. In this paper, we present a method that treats capabilities as contracts for skill implementations and leverages large language models to generate executable code based on natural language user input. A key feature of our approach is the integration of existing software libraries and interface technologies, enabling the generation of skill implementations across different target languages. We introduce a framework that allows users to incorporate their own libraries and resource interfaces into the code generation process through a retrieval-augmented generation architecture. The proposed method is evaluated using an autonomous mobile robot controlled via Python and ROS 2, demonstrating the feasibility and flexibility of the approach.
>
---
