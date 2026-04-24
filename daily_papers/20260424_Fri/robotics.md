# 机器人 cs.RO

- **最新发布 37 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] Neuro-Symbolic Manipulation Understanding with Enriched Semantic Event Chains
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作理解任务，旨在提升动作推理的准确性与鲁棒性。通过改进的eSEC-LAM框架，将语义事件链转化为更丰富的符号状态，增强决策与解释能力。**

- **链接: [https://arxiv.org/pdf/2604.21053](https://arxiv.org/pdf/2604.21053)**

> **作者:** Fatemeh Ziaeetabar
>
> **摘要:** Robotic systems operating in human environments must reason about how object interactions evolve over time, which actions are currently being performed, and what manipulation step is likely to follow. Classical enriched Semantic Event Chains (eSECs) provide an interpretable relational description of manipulation, but remain primarily descriptive and do not directly support uncertainty-aware decision making. In this paper, we propose eSEC-LAM, a neuro-symbolic framework that transforms eSECs into an explicit event-level symbolic state for manipulation understanding. The proposed formulation augments classical eSECs with confidence-aware predicates, functional object roles, affordance priors, primitive-level abstraction, and saliency-guided explanation cues. These enriched symbolic states are derived from a foundation-model-based perception front-end through deterministic predicate extraction, while current-action inference and next-primitive prediction are performed using lightweight symbolic reasoning over primitive pre- and post-conditions. We evaluate the proposed framework on EPIC-KITCHENS-100, EPIC-KITCHENS VISOR, and Assembly101 across action recognition, next-primitive prediction, robustness to perception noise, and explanation consistency. Experimental results show that eSEC-LAM achieves competitive action recognition, substantially improves next-primitive prediction, remains more robust under degraded perceptual conditions than both classical symbolic and end-to-end video baselines, and provides temporally consistent explanation traces grounded in explicit relational evidence. These findings demonstrate that enriched Semantic Event Chains can serve not only as interpretable descriptors of manipulation, but also as effective internal states for neuro-symbolic action reasoning.
>
---
#### [new 002] Self-Predictive Representation for Autonomous UAV Object-Goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于自主无人机目标导航任务，旨在解决数据效率低和目标识别难的问题。提出自预测模型AmelPredSto提升状态表示学习，增强强化学习效果。**

- **链接: [https://arxiv.org/pdf/2604.21130](https://arxiv.org/pdf/2604.21130)**

> **作者:** Angel Ayala; Donling Sui; Francisco Cruz; Mitchell Torok; Mohammad Deghat; Bruno J. T. Fernandes
>
> **备注:** Submitted to T-RO
>
> **摘要:** Autonomous Unmanned Aerial Vehicles (UAVs) have revolutionized industries through their versatility with applications including aerial surveillance, search and rescue, agriculture, and delivery. Their autonomous capabilities offer unique advantages, such as operating in large open space environments. Reinforcement Learning (RL) empowers UAVs to learn intricate navigation policies, enabling them to optimize flight behavior autonomously. However, one of its main challenge is the inefficiency in using data sample to achieve a good policy. In object-goal navigation (OGN) settings, target recognition arises as an extra challenge. Most UAV-related approaches use relative or absolute coordinates to move from an initial position to a predefined location, rather than to find the target directly. This study addresses the data sample efficiency issue in solving a 3D OGN problem, in addition to, the formalization of the unknown target location setting as a Markov decision process. Experiments are conducted to analyze the interplay of different state representation learning (SRL) methods for perception with a model-free RL algorithm for planning in an autonomous navigation system. The main contribution of this study is the development of the perception module, featuring a novel self-predictive model named AmelPred. Empirical results demonstrate that its stochastic version, AmelPredSto, is the best-performing SRL model when combined with actor-critic RL algorithms. The obtained results show substantial improvement in RL algorithms' efficiency by using AmelPredSto in solving the OGN problem.
>
---
#### [new 003] Full-Body Dynamic Safety for Robot Manipulators: 3D Poisson Safety Functions for CBF-Based Safety Filters
- **分类: cs.RO**

- **简介: 该论文属于机器人安全控制任务，解决动态环境下机械臂全身体碰撞避免问题。通过3D泊松安全函数构建全局平滑的CBF，实现高效安全过滤。**

- **链接: [https://arxiv.org/pdf/2604.21189](https://arxiv.org/pdf/2604.21189)**

> **作者:** Meg Wilkinson; Gilbert Bahati; Ryan M. Bena; Emily Fourney; Joel W. Burdick; Aaron D. Ames
>
> **摘要:** Collision avoidance for robotic manipulators requires enforcing full-body safety constraints in high-dimensional configuration spaces. Control Barrier Function (CBF) based safety filters have proven effective in enabling safe behaviors, but enforcing the high number of constraints needed for safe manipulation leads to theoretic and computational challenges. This work presents a framework for full-body collision avoidance for manipulators in dynamic environments by leveraging 3D Poisson Safety Functions (PSFs). In particular, given environmental occupancy data, we sample the manipulator surface at a prescribed resolution and shrink free space via a Pontryagin difference according to this resolution. On this buffered domain, we synthesize a globally smooth CBF by solving Poisson's equation, yielding a single safety function for the entire environment. This safety function, evaluated at each sampled point, yields task-space CBF constraints enforced by a real-time safety filter via a multi-constraint quadratic program. We prove that keeping the sample points safe in the buffered region guarantees collision avoidance for the entire continuous robot surface. The framework is validated on a 7-degree-of-freedom manipulator in dynamic environments.
>
---
#### [new 004] Impact-Aware Model Predictive Control for UAV Landing on a Heaving Platform
- **分类: cs.RO**

- **简介: 该论文属于无人机着陆任务，解决在波动平台上着陆时冲击力大、易反弹的问题。通过构建考虑碰撞的模型预测控制框架，提升着陆稳定性。**

- **链接: [https://arxiv.org/pdf/2604.21078](https://arxiv.org/pdf/2604.21078)**

> **作者:** Jess Stephenson; Melissa Greeff
>
> **备注:** To be published in the proceedings of International Federation of Automatic Control (IFAC) World Congress 2026
>
> **摘要:** Landing UAVs on heaving marine platforms is challenging because relative vertical motion can generate large impact forces and cause rebound on touchdown. To address this, we develop an impact-aware Model Predictive Control (MPC) framework that models landing as a velocity-level rigid-body impact governed by Newton's restitution law. We embed this as a linear complementarity problem (LCP) within the MPC dynamics to predict the discontinuous post-impact velocity and suppress rebound. In simulation, restitution-aware prediction reduces pre-impact relative velocity and improves landing robustness. Experiments on a heaving-deck testbed show an 86.2% reduction in post-impact deflection compared to a tracking MPC.
>
---
#### [new 005] A Bayesian Reasoning Framework for Robotic Systems in Autonomous Casualty Triage
- **分类: cs.RO**

- **简介: 该论文属于自主救援任务，解决MCI中机器人准确评估伤者状况的问题。通过融合多视觉算法与贝叶斯网络，提升伤者分类的准确性与覆盖范围。**

- **链接: [https://arxiv.org/pdf/2604.21568](https://arxiv.org/pdf/2604.21568)**

> **作者:** Szymon Rusiecki; Cecilia Morales; Pia Störy; Kimberly Elenberg; Leonard Weiss; Artur Dubrawski
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Autonomous robots deployed in mass casualty incidents (MCI) face the challenge of making critical decisions based on incomplete and noisy perceptual data. We present an autonomous robotic system for casualty assessment that fuses outputs from multiple vision-based algorithms, estimating signs of severe hemorrhage, visible trauma, or physical alertness, into a coherent triage assessment. At the core of our system is a Bayesian network, constructed from expert-defined rules, which enables probabilistic reasoning about a casualty's condition even with missing or conflicting sensory inputs. The system, evaluated during the DARPA Triage Challenge (DTC) in realistic MCI scenarios involving 11 and 9 casualties, demonstrated a nearly three-fold improvement in physiological assessment accuracy (from 15\% to 42\% and 19\% to 46\%) compared to a vision-only baseline. More importantly, overall triage accuracy increased from 14\% to 53\%, while the diagnostic coverage of the system expanded from 31\% to 95\% of cases. These results demonstrate that integrating expert-guided probabilistic reasoning with advanced vision-based sensing can significantly enhance the reliability and decision-making capabilities of autonomous systems in critical real-world applications.
>
---
#### [new 006] A Replicable Robotics Awareness Method Using LLM-Enabled Robotics Interaction: Evidence from a Corporate Challenge
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决非专业用户在组织中接触机器人技术的问题。通过LLM驱动的机器人活动，评估其在提升机器人意识与协作理解方面的有效性。**

- **链接: [https://arxiv.org/pdf/2604.21377](https://arxiv.org/pdf/2604.21377)**

> **作者:** S. A. Prieto; M. A. Gopee; Y. Ben Arab; B. García de Soto; J. Esteba; P. Olivera Brizzio
>
> **备注:** 10 pages, 8 Figures, to be submitted for journal per-review
>
> **摘要:** Large language models are increasingly being explored as interfaces between humans and robotic systems, yet there remains limited evidence on how such technologies can be used not only for interaction, but also as a structured means of introducing robotics to non-specialist users in real organizational settings. This paper introduces and evaluates a challenge-based method for robotics awareness, implemented through an LLM-enabled humanoid robot activity conducted with employees of AD Ports Group in the United Arab Emirates. In the event, participants engaged with a humanoid robot in a logistics-inspired task environment using voice commands interpreted through an LLM-based control framework. The activity was designed as a team-based, role-driven experience intended to expose participants to embodied AI and human-robot collaboration without requiring prior robotics expertise. To evaluate the approach, a post-event survey remained open for 16 days and collected 102 responses. Results indicate strong overall reception, with high satisfaction (8.46/10), increased interest in robotics and AI (4.47/5), and improved understanding of emerging forms of human-robot collaboration (4.45/5). Participants who interacted directly with the robot also reported natural interaction (4.37/5) and a strong sense that interaction became easier as the activity progressed (4.74/5). At the same time, lower ratings for reliability and predictability point to important technical and design challenges for future iterations. The findings suggest that challenge-based, LLM-enabled humanoid interaction can serve as a promising and replicable method for robotics awareness in industrial and operational environments.
>
---
#### [new 007] RPG: Robust Policy Gating for Smooth Multi-Skill Transitions in Humanoid Fighting
- **分类: cs.RO**

- **简介: 该论文属于人形机器人战斗控制任务，解决多技能平滑过渡的稳定性问题。提出RPG框架，通过统一策略实现稳定、流畅的技能切换。**

- **链接: [https://arxiv.org/pdf/2604.21355](https://arxiv.org/pdf/2604.21355)**

> **作者:** Yucheng Xin; Jiacheng Bao; Yubo Dong; Xueqian Wang; Bin Zhao; Xuelong Li; Junbo Tan; Dong Wang
>
> **摘要:** Humanoid robots have demonstrated impressive motor skills in a wide range of tasks, yet whole-body control for humanlike long-time, dynamic fighting remains particularly challenging due to the stringent requirements on agility and stability. While imitation learning enables robots to execute human-like fighting skills, existing approaches often rely on switching among multiple single-skill policies or employing a general policy to imitate input reference motions. These strategies suffer from instability when transitioning between skills, as the mismatch of initial and terminal states across skills or reference motions introduces out-of-domain disturbances, resulting in unsmooth or unstable behaviors. In this work, we propose RPG, a hybrid expert policy framework, for smooth and stable humanoid multi-skills transition. Our approach incorporates motion transition randomization and temporal randomization to train a unified policy that generates agile fighting actions with stability and smoothness during skill transitions. Furthermore, we design a control pipeline that integrates walking/running locomotion with fighting skills, allowing humanlike long-time combat of arbitrary duration that can be seamlessly interrupted or transit action policies at any time. Extensive experiments in simulation demonstrate the effectiveness of the proposed framework, and real-world deployment on the Unitree G1 humanoid robot further validates its robustness and applicability.
>
---
#### [new 008] Long-Horizon Manipulation via Trace-Conditioned VLA Planning
- **分类: cs.RO**

- **简介: 该论文研究长时段操作任务，解决视觉-语言-动作策略在多步骤、易出错环境中的性能问题。提出LoHo-Manip框架，通过任务管理器和执行器协作，提升长时序操作的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.21924](https://arxiv.org/pdf/2604.21924)**

> **作者:** Isabella Liu; An-Chieh Cheng; Rui Yan; Geng Chen; Ri-Zhao Qiu; Xueyan Zou; Sha Yi; Hongxu Yin; Xiaolong Wang; Sifei Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** Long-horizon manipulation remains challenging for vision-language-action (VLA) policies: real tasks are multi-step, progress-dependent, and brittle to compounding execution errors. We present LoHo-Manip, a modular framework that scales short-horizon VLA execution to long-horizon instruction following via a dedicated task-management VLM. The manager is decoupled from the executor and is invoked in a receding-horizon manner: given the current observation, it predicts a progress-aware remaining plan that combines (i) a subtask sequence with an explicit done + remaining split as lightweight language memory, and (ii) a visual trace -- a compact 2D keypoint trajectory prompt specifying where to go and what to approach next. The executor VLA is adapted to condition on the rendered trace, thereby turning long-horizon decision-making into repeated local control by following the trace. Crucially, predicting the remaining plan at each step yields an implicit closed loop: failed steps persist in subsequent outputs, and traces update accordingly, enabling automatic continuation and replanning without hand-crafted recovery logic or brittle visual-history buffers. Extensive experiments spanning embodied planning, long-horizon reasoning, trajectory prediction, and end-to-end manipulation in simulation and on a real Franka robot demonstrate strong gains in long-horizon success, robustness, and out-of-distribution generalization. Project page: this https URL
>
---
#### [new 009] Open-H-Embodiment: A Large-Scale Dataset for Enabling Foundation Models in Medical Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Open-H-Embodiment数据集，解决医疗机器人领域数据不足问题，支持基础模型研究，提升机器人自主能力。**

- **链接: [https://arxiv.org/pdf/2604.21017](https://arxiv.org/pdf/2604.21017)**

> **作者:** Open-H-Embodiment Consortium; Nigel Nelson; Juo-Tung Chen; Jesse Haworth; Xinhao Chen; Lukas Zbinden; Dianye Huang; Alaa Eldin Abdelaal; Alberto Arezzo; Ayberk Acar; Farshid Alambeigi; Carlo Alberto Ammirati; Yunke Ao; Pablo David Aranda Rodriguez; Soofiyan Atar; Mattia Ballo; Noah Barnes; Federica Barontini; Filip Binkiewicz; Peter Black; Sebastian Bodenstedt; Leonardo Borgioli; Nikola Budjak; Benjamin Calmé; Fabio Carrillo; Nicola Cavalcanti; Changwei Chen; Haoxin Chen; Sihang Chen; Qihan Chen; Zhongyu Chen; Ziyang Chen; Shing Shin Cheng; Meiqing Cheng; Min Cheng; Zih-Yun Sarah Chiu; Xiangyu Chu; Camilo Correa-Gallego; Giulio Dagnino; Anton Deguet; Jacob Delgado; Jonathan C. DeLong; Kaizhong Deng; Alexander Dimitrakakis; Qingpeng Ding; Hao Ding; Giovanni Distefano; Daniel Donoho; Anqing Duan; Marco Esposito; Shane Farritor; Jad Fayad; Zahi Fayad; Mario Ferradosa; Filippo Filicori; Chelsea Finn; Philipp Fürnstahl; Jiawei Ge; Stamatia Giannarou; Xavier Giralt Ludevid; Frederic Giraud; Aditya Amit Godbole; Ken Goldberg; Antony Goldenberg; Diego Granero Marana; Xiaoqing Guo; Tamás Haidegger; Evan Hailey; Pascal Hansen; Ziyi Hao; Kush Hari; Kengo Hayashi; Jonathon Hawkins; Shelby Haworth; Ortrun Hellig; S. Duke Herrell; Zhouyang Hong; Andrew Howe; Junlei Hu; Ria Jain; Mohammad Rafiee Javazm; Howard Ji; Rui Ji; Jianmin Ji; Zhongliang Jiang; Dominic Jones; Jeffrey Jopling; Britton Jordan; Ran Ju; Michael Kam; Luoyao Kang; Fausto Kang; Siddhartha Kapuria; Peter Kazanzides; Sonika Kiehler; Ethan Kilmer; Ji Woong; Przemysław Korzeniowski; Chandra Kuchi; Nithesh Kumar
>
> **备注:** Project website: this https URL
>
> **摘要:** Autonomous medical robots hold promise to improve patient outcomes, reduce provider workload, democratize access to care, and enable superhuman precision. However, autonomous medical robotics has been limited by a fundamental data problem: existing medical robotic datasets are small, single-embodiment, and rarely shared openly, restricting the development of foundation models that the field needs to advance. We introduce Open-H-Embodiment, the largest open dataset of medical robotic video with synchronized kinematics to date, spanning more than 49 institutions and multiple robotic platforms including the CMR Versius, Intuitive Surgical's da Vinci, da Vinci Research Kit (dVRK), Rob Surgical BiTrack, Virtual Incision's MIRA, Moon Surgical Maestro, and a variety of custom systems, spanning surgical manipulation, robotic ultrasound, and endoscopy procedures. We demonstrate the research enabled by this dataset through two foundation models. GR00T-H is the first open foundation vision-language-action model for medical robotics, which is the only evaluated model to achieve full end-to-end task completion on a structured suturing benchmark (25% of trials vs. 0% for all others) and achieves 64% average success across a 29-step ex vivo suturing sequence. We also train Cosmos-H-Surgical-Simulator, the first action-conditioned world model to enable multi-embodiment surgical simulation from a single checkpoint, spanning nine robotic platforms and supporting in silico policy evaluation and synthetic data generation for the medical domain. These results suggest that open, large-scale medical robot data collection can serve as critical infrastructure for the research community, enabling advances in robot learning, world modeling, and beyond.
>
---
#### [new 010] Hi-WM: Human-in-the-World-Model for Scalable Robot Post-Training
- **分类: cs.RO**

- **简介: 该论文提出Hi-WM框架，解决机器人后训练中依赖真实世界交互的问题。通过世界模型实现高效策略修正，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.21741](https://arxiv.org/pdf/2604.21741)**

> **作者:** Yaxuan Li; Zhongyi Zhou; Yefei Chen; Yanjiang Guo; Jiaming Liu; Shanghang Zhang; Jianyu Chen; Yichen Zhu
>
> **备注:** Project Page: this https URL
>
> **摘要:** Post-training is essential for turning pretrained generalist robot policies into reliable task-specific controllers, but existing human-in-the-loop pipelines remain tied to physical execution: each correction requires robot time, scene setup, resets, and operator supervision in the real world. Meanwhile, action-conditioned world models have been studied mainly for imagination, synthetic data generation, and policy evaluation. We propose \textbf{Human-in-the-World-Model (Hi-WM)}, a post-training framework that uses a learned world model as a reusable corrective substrate for failure-targeted policy improvement. A policy is first rolled out in closed loop inside the world model; when the rollout becomes incorrect or failure-prone, a human intervenes directly in the model to provide short corrective actions. Hi-WM caches intermediate states and supports rollback and branching, allowing a single failure state to be reused for multiple corrective continuations and yielding dense supervision around behaviors that the base policy handles poorly. The resulting corrective trajectories are then added back to the training set for post-training. We evaluate Hi-WM on three real-world manipulation tasks spanning both rigid and deformable object interaction, and on two policy backbones. Hi-WM improves real-world success by 37.9 points on average over the base policy and by 19.0 points over a world-model closed-loop baseline, while world-model evaluation correlates strongly with real-world performance (r = 0.953). These results suggest that world models can serve not only as generators or evaluators, but also as effective corrective substrates for scalable robot post-training.
>
---
#### [new 011] A Survey of Legged Robotics in Non-Inertial Environments: Past, Present, and Future
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动控制任务，解决腿式机器人在非惯性环境中的可靠性问题，通过综述建模、状态估计与控制方法，分析挑战并提出未来方向。**

- **链接: [https://arxiv.org/pdf/2604.20990](https://arxiv.org/pdf/2604.20990)**

> **作者:** I-Chia Chang; Xinyan Huang; Tzu-Yuan Lin; Sangli Teng; Wenjing Li; Maani Ghaffari; Jingang Yi; Yan Gu
>
> **摘要:** Legged robots have demonstrated remarkable agility on rigid, stationary ground, but their locomotion reliability remains limited in non-inertial environments, where the supporting ground moves, tilts, or accelerates. Such conditions arise in ground transportation, maritime platforms, and aerospace settings, and they introduce persistent time-varying disturbances that break the stationary-ground assumptions underlying conventional legged locomotion. This survey reviews the state of the art in modeling, state estimation, and control for legged robots in non-inertial environments. We summarize representative application domains and motion characteristics, analyze the root causes of locomotion performance degradation, and review existing methods together with their key assumptions and limitations. We further identify open problems in robot-environment coupling, observability, robustness, and experimental validation, and discuss future directions in autonomy, system-level design, bio-inspired strategies, safety, and testing. The survey aims to clarify the technical foundations of this emerging area and support the development of reliable legged robots for real-world dynamic environments.
>
---
#### [new 012] VistaBot: View-Robust Robot Manipulation via Spatiotemporal-Aware View Synthesis
- **分类: cs.RO**

- **简介: 该论文提出VistaBot，解决机器人操作中视角变化导致的鲁棒性问题。通过整合几何模型与视频扩散模型，提升跨视角泛化能力，适用于仿真和真实任务。**

- **链接: [https://arxiv.org/pdf/2604.21914](https://arxiv.org/pdf/2604.21914)**

> **作者:** Songen Gu; Yuhang Zheng; Weize Li; Yupeng Zheng; Yating Feng; Xiang Li; Yilun Chen; Pengfei Li; Wenchao Ding
>
> **备注:** This paper has been accepted to ICRA 2026
>
> **摘要:** Recently, end-to-end robotic manipulation models have gained significant attention for their generalizability and scalability. However, they often suffer from limited robustness to camera viewpoint changes when training with a fixed camera. In this paper, we propose VistaBot, a novel framework that integrates feed-forward geometric models with video diffusion models to achieve view-robust closed-loop manipulation without requiring camera calibration at test time. Our approach consists of three key components: 4D geometry estimation, view synthesis latent extraction, and latent action learning. VistaBot is integrated into both action-chunking (ACT) and diffusion-based ($\pi_0$) policies and evaluated across simulation and real-world tasks. We further introduce the View Generalization Score (VGS) as a new metric for comprehensive evaluation of cross-view generalization. Results show that VistaBot improves VGS by 2.79$\times$ and 2.63$\times$ over ACT and $\pi_0$, respectively, while also achieving high-quality novel view synthesis. Our contributions include a geometry-aware synthesis model, a latent action planner, a new benchmark metric, and extensive validation across diverse environments. The code and models will be made publicly available.
>
---
#### [new 013] Task-Driven Co-Design of Heterogeneous Multi-Robot Systems
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人系统设计任务，旨在解决异构多机器人系统的协同设计问题。通过构建形式化框架，实现机器人设计、编队和规划的联合优化。**

- **链接: [https://arxiv.org/pdf/2604.21894](https://arxiv.org/pdf/2604.21894)**

> **作者:** Maximilian Stralz; Meshal Alharbi; Yujun Huang; Gioele Zardini
>
> **摘要:** Designing multi-agent robotic systems requires reasoning across tightly coupled decisions spanning heterogeneous domains, including robot design, fleet composition, and planning. Much effort has been devoted to isolated improvements in these domains, whereas system-level co-design considering trade-offs and task requirements remains underexplored. In this work, we present a formal and compositional framework for the task-driven co-design of heterogeneous multi-robot systems. Building on a monotone co-design theory, we introduce general abstractions of robots, fleets, planners, executors, and evaluators as interconnected design problems with well-defined interfaces that are agnostic to both implementations and tasks. This structure enables efficient joint optimization of robot design, fleet composition, and planning under task-specific performance constraints. A series of case studies demonstrates the capabilities of the framework. Various component models can be seamlessly incorporated, including new robot types, task profiles, and probabilistic sensing objectives, while non-obvious design alternatives are systematically uncovered with optimality guarantees. The results highlight the flexibility, scalability, and interpretability of the proposed approach, and illustrate how formal co-design enables principled reasoning about complex heterogeneous multi-robot systems.
>
---
#### [new 014] From Noise to Intent: Anchoring Generative VLA Policies with Residual Bridges
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决语义理解与物理控制间的尺度不匹配问题。提出ResVLA架构，通过分解意图与动态，提升控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.21391](https://arxiv.org/pdf/2604.21391)**

> **作者:** Yiming Zhong; Yaoyu He; Zemin Yang; Pengfei Tian; Yifan Huang; Qingqiu Huang; Xinge Zhu; Yuexin Ma
>
> **摘要:** Bridging high-level semantic understanding with low-level physical control remains a persistent challenge in embodied intelligence, stemming from the fundamental spatiotemporal scale mismatch between cognition and action. Existing generative VLA policies typically adopt a "Generation-from-Noise" paradigm, which disregards this disparity, leading to representation inefficiency and weak condition alignment during optimization. In this work, we propose ResVLA, an architecture that shifts the paradigm to "Refinement-from-Intent." Recognizing that robotic motion naturally decomposes into global intent and local dynamics, ResVLA utilizes spectral analysis to decouple control into a deterministic low-frequency anchor and a stochastic high-frequency residual. By anchoring the generative process on the predicted intent, our model focuses strictly on refining local dynamics via a residual diffusion bridge. Extensive simulation experiments show that ResVLA achieves competitive performance, strong robustness to language and robot embodiment perturbations, and faster convergence than standard generative baselines. It also demonstrates strong performance in real-world robot experiments.
>
---
#### [new 015] PREVENT-JACK: Context Steering for Swarms of Long Heavy Articulated Vehicles
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究群体机器人中长重挂车的协同控制问题，提出Prevent-Jack方法解决jackknifing和碰撞问题，通过仿真评估不同规模群体的表现。**

- **链接: [https://arxiv.org/pdf/2604.21337](https://arxiv.org/pdf/2604.21337)**

> **作者:** Adrian Baruck; Michael Dubé; Christoph Steup; Sanaz Mostaghim
>
> **备注:** 32 pages, 7 figures, 4 videos; submitted to the Swarm Robotics collection of the Nature Portfolio Journal Robotics (NPJ Robot)
>
> **摘要:** In this paper, we aim to extend the traditional point-mass-like robot representation in swarm robotics and instead study a swarm of long Heavy Articulated Vehicles (HAVs). HAVs are kinematically constrained, elongated, and articulated, introducing unique challenges. Local, decentralized coordination of these vehicles is motivated by many real-world applications. Our approach, Prevent-Jack, introduces the sparsely covered context steering framework in robotics. It fuses six local behaviors, providing guarantees against jackknifing and collisions at the cost of potential dead- and livelocks, tested for vehicles with up to ten trailers. We highlight the importance of the Evade Attraction behavior for deadlock prevention using a parameter study, and use 15,000 simulations to evaluate the swarm performance. Our extensive experiments and the results show that both the dead- and livelocks occur more frequently in larger swarms and denser scenarios, affecting a peak average of 27%/31% of vehicles. We observe that larger swarms exhibit increased waiting, while smaller swarms show increased evasion.
>
---
#### [new 016] SLAM as a Stochastic Control Problem with Partial Information: Optimal Solutions and Rigorous Approximations
- **分类: cs.RO; math.OC**

- **简介: 该论文将SLAM建模为部分可观测马尔可夫决策过程，解决机器人在未知环境中同时定位与建图的问题，提出近似最优解并进行数值验证。**

- **链接: [https://arxiv.org/pdf/2604.21693](https://arxiv.org/pdf/2604.21693)**

> **作者:** Ilir Gusija; Fady Alajaji; Serdar Yüksel
>
> **摘要:** Simultaneous localization and mapping (SLAM) is a foundational state estimation problem in robotics in which a robot accurately constructs a map of its environment while also localizing itself within this construction. We study the active SLAM problem through the lens of optimal stochastic control, thereby recasting it as a decision-making problem under partial information. After reviewing several commonly studied models, we present a general stochastic control formulation of active SLAM together with a rigorous treatment of motion, sensing, and map representation. We introduce a new exploration stage cost that encodes the geometry of the state when evaluating information-gathering actions. This formulation, constructed as a nonstandard partially observable Markov decision process (POMDP), is then analyzed to derive rigorously justified approximate solutions that are near-optimal. To enable this analysis, the associated regularity conditions are studied under general assumptions that apply to a wide range of robotics applications. For a particular case, we conduct an extensive numerical study in which standard learning algorithms are used to learn near-optimal policies.
>
---
#### [new 017] Reasoning About Traversability: Language-Guided Off-Road 3D Trajectory Planning
- **分类: cs.RO**

- **简介: 该论文属于越野路径规划任务，解决语言标注与车辆动作不匹配的问题。通过构建动作对齐的标注和地形感知优化策略，提升轨迹准确性与地形一致性。**

- **链接: [https://arxiv.org/pdf/2604.21249](https://arxiv.org/pdf/2604.21249)**

> **作者:** Byounggun Park; Soonmin Hwang
>
> **摘要:** While Vision-Language Models (VLMs) enable high-level semantic reasoning for end-to-end autonomous driving, particularly in unstructured environments, existing off-road datasets suffer from language annotations that are weakly aligned with vehicle actions and terrain geometry. To address this misalignment, we propose a language refinement framework that restructures annotations into action-aligned pairs, enabling a VLM to generate refined scene descriptions and 3D future trajectories directly from a single image. To further encourage terrain-aware planning, we introduce a preference optimization strategy that constructs geometry-aware hard negatives and explicitly penalizes trajectories inconsistent with local elevation profiles. Furthermore, we propose off-road-specific metrics to quantify traversability compliance and elevation consistency, addressing the limitations of conventional on-road evaluation. Experiments on the ORAD-3D benchmark demonstrate that our approach reduces average trajectory error from 1.01m to 0.97m, improves traversability compliance from 0.621 to 0.644, and decreases elevation inconsistency from 0.428 to 0.322, highlighting the efficacy of action-aligned supervision and terrain-aware optimization for robust off-road driving.
>
---
#### [new 018] A Compact Peristaltic Pump Based on Magneto-Elastic Hysteresis with Single Pneumatic Control
- **分类: cs.RO**

- **简介: 该论文属于流体输送技术领域，旨在解决传统泵结构复杂的问题。通过单气动控制结合磁弹性滞后效应，设计了一种紧凑的蠕动泵。**

- **链接: [https://arxiv.org/pdf/2604.21729](https://arxiv.org/pdf/2604.21729)**

> **作者:** Minjo Park; Metin Sitti
>
> **备注:** 5 pages
>
> **摘要:** Pumping fluids is fundamental to a wide range of industrial, environmental, and biomedical applications. Among various pumping mechanisms, peristaltic pumps enable efficient and safe fluid transport by deforming an elastic tube without direct contact with the working fluid. Although previous studies have introduced mechanical, pneumatic, or magnetic actuations to drive membrane deformation, these approaches often lead to complex pump architectures and control schemes. In this study, we present a soft membrane pump that achieves peristaltic motion through a single pneumatic input combined with an embedded passive magnet. The actuation mechanism and system dynamics were analyzed and simplified through modeling. Numerical simulations were conducted to predict the internal fluid flow, and the magneto-elastic hysteresis behavior observed in the simulations was successfully validated by experiments with a proof-of-concept prototype.
>
---
#### [new 019] Effects of Swarm Size Variability on Operator Workload
- **分类: cs.RO**

- **简介: 该论文属于人机协同任务，研究 swarm 大小变化对操作员负荷的影响，通过实验分析不同变化方向和幅度对工作负荷的作用，为动态系统中的负荷管理提供指导。**

- **链接: [https://arxiv.org/pdf/2604.21707](https://arxiv.org/pdf/2604.21707)**

> **作者:** William Hunt; Aleksandra Landowska; Horia A. Maior; Sarvapali D. Ramchurn; Mohammad Soorati
>
> **摘要:** Real-world deployments of human--swarm teams depend on balancing operator workload to leverage human strengths without inducing overload. A key challenge is that swarm size is often dynamic: robots may join or leave the mission due to failures or redeployment, causing abrupt workload fluctuations. Understanding how such changes affect human workload and performance is critical for robust human--swarm interaction design. This paper investigates how the magnitude and direction of changes in swarm size influence operator workload. Drawing on the concept of workload history, we test three hypotheses: (1) workload remains elevated following decreases in swarm size, (2) small increases are more manageable than large jumps, and (3) sufficiently large changes override these effects by inducing a cognitive reset. We conducted two studies (N = 34) using a monitoring task with simulated drone swarms of varying sizes. By varying the swarm size between episodes, we measured perceived workload relative to swarm size changes. Results show that objective performance is largely unaffected by small changes in swarm size, while subjective workload is sensitive to both change direction and magnitude. Small increases preserve lower workload, whereas small decreases leave workload elevated, indicating workload residue; large changes in either direction attenuate these effects, suggesting a reset response. These findings offer actionable guidance for managing swarm-size transitions to support operator workload in dynamic human--swarm systems.
>
---
#### [new 020] How VLAs (Really) Work In Open-World Environments
- **分类: cs.RO; cs.AI**

- **简介: 论文研究视觉-语言-动作模型在开放环境中的表现，指出现有评估方法忽视安全性和任务意识，提出更全面的评估协议以提升真实场景下的性能衡量。**

- **链接: [https://arxiv.org/pdf/2604.21192](https://arxiv.org/pdf/2604.21192)**

> **作者:** Amir Rasouli; Yangzheng Wu; Zhiyuan Li; Rui Heng Yang; Xuan Zhao; Charles Eret; Sajjad Pakdamansavoji
>
> **备注:** 8 pages, 7 figures, 2 tables
>
> **摘要:** Vision-language-action models (VLAs) have been extensively used in robotics applications, achieving great success in various manipulation problems. More recently, VLAs have been used in long-horizon tasks and evaluated on benchmarks, such as BEHAVIOR1K (B1K), for solving complex household chores. The common metric for measuring progress in such benchmarks is success rate or partial score based on satisfaction of progress-agnostic criteria, meaning only the final states of the objects are considered, regardless of the events that lead to such states. In this paper, we argue that using such evaluation protocols say little about safety aspects of operation and can potentially exaggerate reported performance, undermining core challenges for future real-world deployment. To this end, we conduct a thorough analysis of state-of-the-art models on the B1K Challenge and evaluate policies in terms of robustness via reproducibility and consistency of performance, safety aspects of policies operations, task awareness, and key elements leading to the incompletion of tasks. We then propose evaluation protocols to capture safety violations to better measure the true performance of the policies in more complex and interactive scenarios. At the end, we discuss the limitations of the existing VLAs and motivate future research.
>
---
#### [new 021] A Tendon-Driven Wrist Abduction-Adduction Joint Improves Performance of a 5 DoF Upper Limb Exoskeleton -- Implementation and Experimental Evaluation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于康复工程任务，旨在提升上肢外骨骼在日常活动中的功能表现。通过引入主动腕关节外展-内收模块，改善了任务完成效果。**

- **链接: [https://arxiv.org/pdf/2604.20898](https://arxiv.org/pdf/2604.20898)**

> **作者:** Juwairiya S. Khan; Mostafa Mohammadi; Alexander L. Ammitzbøll; Ellen-Merete Hagen; Jakob Blicher; Izabella Obál; Ana S. S. Cardoso; Oguzhan Kirtas; Rasmus L. Kæseler; John Rasmussen; Lotte N.S. Andreasen Struijk
>
> **备注:** 9 pages, 5 figures and 1 table. Submitted to IEEE Transactions on Biomedical Engineering as invited IEEE EMBC special issue paper. Under review after first revision
>
> **摘要:** Wrist function is essential in performing activities of daily living (ADLs). However, there is limited experimental evidence on the functional impact of wrist Abduction-Adduction (Ab-Ad) joint assistance in upper limb exoskeletons (ULEs) for rehabilitation. This study evaluates the effect of implementing an active wrist Ab-Ad joint in a five degree of freedom (DoF) ULE, EXOTIC2 exoskeleton, to support individuals with severe motor impairments. Methods: A compact, lightweight wrist module with tendon-driven abduction and spring-driven adduction was integrated into the EXOTIC exoskeleton. Eight adults with no motor disabilities completed drinking and scratching tasks under randomized wrist-enabled and wrist-locked conditions along with a preliminary feasibility test in one individual with Amyotrophic lateral sclerosis (ALS). Kinematic and task performance metrics including wrist range of motion, task completion time, spillage and leveling metrics were assessed. Results: Implementing the wrist Ab-Ad DoF improved task success metrics. Spill incidence during the drinking task decreased from 56% to 3%, and leveling success for scratching task improved from 28% to 75%. Conclusion: Integrating wrist Ab-Ad assistance improved key functional task outcomes without increasing execution time. Significance: The study provides the experimental evidence that active wrist Ab-Ad control enhances task-level performance in exoskeleton-assisted ADLs.
>
---
#### [new 022] CorridorVLA: Explicit Spatial Constraints for Generative Action Heads via Sparse Anchors
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在通过显式空间约束提升动作生成效果。提出CorridorVLA模型，利用稀疏空间锚点定义容错区域，增强动作策略的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.21241](https://arxiv.org/pdf/2604.21241)**

> **作者:** Dachong Li; ZhuangZhuang Chen; Jin Zhang; Jianqiang Li
>
> **摘要:** Vision--Language--Action (VLA) models often use intermediate representations to connect multimodal inputs with continuous control, yet spatial guidance is often injected implicitly through latent features. We propose $CorridorVLA$, which predicts sparse spatial anchors as incremental physical changes (e.g., $\Delta$-positions) and uses them to impose an explicit tolerance region in the training objective for action generation. The anchors define a corridor that guides a flow-matching action head: trajectories whose implied spatial evolution falls outside it receive corrective gradients, while minor deviations from contacts and execution noise are permitted. On the more challenging LIBERO-Plus benchmark, CorridorVLA yields consistent gains across both SmolVLA and GR00T, improving success rate by $3.4\%$--$12.4\%$ over the corresponding baselines; notably, our GR00T-Corr variant reaches a success rate of $83.21\%$. These results indicate that action-aligned physical cues can provide direct and interpretable constraints for generative action policies, complementing spatial guidance encoded in visual or latent forms. Code is available at this https URL.
>
---
#### [new 023] Navigating the Clutter: Waypoint-Based Bi-Level Planning for Multi-Robot Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人系统控制任务，解决复杂环境中机器人路径规划问题。通过引入航点和强化学习方法，实现任务与运动规划的联合优化，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.21138](https://arxiv.org/pdf/2604.21138)**

> **作者:** Jiabao Ji; Yongchao Chen; Yang Zhang; Ramana Rao Kompella; Chuchu Fan; Gaowen Liu; Shiyu Chang
>
> **摘要:** Multi-robot control in cluttered environments is a challenging problem that involves complex physical constraints, including robot-robot collisions, robot-obstacle collisions, and unreachable motions. Successful planning in such settings requires joint optimization over high-level task planning and low-level motion planning, as violations of physical constraints may arise from failures at either level. However, jointly optimizing task and motion planning is difficult due to the complex parameterization of low-level motion trajectories and the ambiguity of credit assignment across the two planning levels. In this paper, we propose a hybrid multi-robot control framework that jointly optimizes task and motion planning. To enable effective parameterization of low-level planning, we introduce waypoints, a simple yet expressive representation for motion trajectories. To address the credit assignment challenge, we adopt a curriculum-based training strategy with a modified RLVR algorithm that propagates motion feasibility feedback from the motion planner to the task planner. Experiments on BoxNet3D-OBS, a challenging multi-robot benchmark with dense obstacles and up to nine robots, show that our approach consistently improves task success over motion-agnostic and VLA-based baselines. Our code is available at this https URL
>
---
#### [new 024] MISTY: High-Throughput Motion Planning via Mixer-based Single-step Drifting
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶中的运动规划任务，解决传统方法推理延迟高的问题。提出MISTY模型，通过单步推理实现高效轨迹生成。**

- **链接: [https://arxiv.org/pdf/2604.21489](https://arxiv.org/pdf/2604.21489)**

> **作者:** Yining Xing; Zehong Ke; Yiqian Tu; Zhiyuan Liu; Wenhao Yu; Jianqiang Wang
>
> **备注:** 8 pages, 4 figures, 3 tables. Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Multi-modal trajectory generation is essential for safe autonomous driving, yet existing diffusion-based planners suffer from high inference latency due to iterative neural function evaluations. This paper presents MISTY (Mixer-based Inference for Single-step Trajectory-drifting Yield), a high-throughput generative motion planner that achieves state-of-the-art closed-loop performance with pure single-step inference. MISTY integrates a vectorized Sub-Graph encoder to capture environment context, a Variational Autoencoder to structure expert trajectories into a compact 32-dimensional latent manifold, and an ultra-lightweight MLP-Mixer decoder to eliminate quadratic attention complexity. Importantly, we introduce a latent-space drifting loss that shifts the complex distribution evolution entirely to the training phase. By formulating explicit attractive and repulsive forces, this mechanism empowers the model to synthesize novel, proactive maneuvers, such as active overtaking, that are virtually absent from the raw expert demonstrations. Extensive evaluations on the nuPlan benchmark demonstrate that MISTY achieves state-of-the-art results on the challenging Test14-hard split, with comprehensive scores of 80.32 and 82.21 in non-reactive and reactive settings, respectively. Operating at over 99 FPS with an end-to-end latency of 10.1 ms, MISTY offers an order-of-magnitude speedup over iterative diffusion planners while while achieving significantly robust generation.
>
---
#### [new 025] Clinical Evaluation of a Tongue-Controlled Wrist Abduction-Adduction Assistance in a 6-DoF Upper-Limb Exoskeleton for Individuals with ALS and SCI
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于康复工程任务，旨在评估手腕外展-内收辅助功能对上肢外骨骼使用的影响。研究通过实验验证该功能提升任务成功率，改善用户体验，为辅助设备设计提供依据。**

- **链接: [https://arxiv.org/pdf/2604.20967](https://arxiv.org/pdf/2604.20967)**

> **作者:** Juwairiya S. Khan; Mostafa Mohammadi; Alexander L. Ammitzbøll; Ellen-Merete Hagen; Jakob Blicher Izabella Obál; Ana S. S. Cardoso; Oguzhan Kirtas; Rasmus L. Kæseler; John Rasmussen; Lotte N.S. Andreasen Struijk
>
> **备注:** 9 pages, 7 figures and 2 tables. This work has been submitted to the IEEE Transactions on Neural Systems and Rehabilitation Engineering
>
> **摘要:** Upper-limb exoskeletons (ULEs) have the potential to restore functional independence in individuals with severe motor impairments; however, the clinical relevance of wrist degrees of freedom (DoF), particularly abduction-adduction (Ab-Ad), remains insufficiently evaluated. This study investigates the functional and user-perceived impact of wrist Ab-Ad assistance during two activities of daily living (ADLs). Wrist Ab-Ad assistance in a tongue-controlled 6-DoF ULE, EXOTIC2, was evaluated in a within-subject study involving one individual with amyotrophic lateral sclerosis and five individuals with spinal cord injury. Participants performed drinking and scratch stick leveling tasks with EXOTIC2 under two conditions: with and without wrist Ab-Ad assistance. Outcome measure included task success, task completion time, kinematic measures, and a usability questionnaire capturing comfort, functional perception, and acceptance. Enabling wrist Ab-Ad improved task success rates across both ADLs, with consistent reductions in spillage (from 77.8% spillages to 22.2%) and failed placements (from 66.7% to 16.7%). Participants utilized task-specific subsets of the available wrist range of motion, indicating that effective control within functional ranges was more critical than maximal joint excursion. Questionnaire responses indicated no increase in discomfort with the additional DoF and reflected perceived improvements in task performance. In conclusion, wrist Ab-Ad assistance enhances functional task performance in assistive exoskeleton use without compromising user comfort. However, its effectiveness depends on task context, control usability, and individual user strategies. This study provides clinically relevant, user-centered evidence supporting the inclusion of wrist Ab-Ad in ULEs, emphasizing the importance of balancing functional capability with usability in assistive device design.
>
---
#### [new 026] Design, Modelling and Experimental Evaluation of a Tendon-driven Wrist Abduction-Adduction Mechanism for an upper limb exoskeleton
- **分类: cs.RO**

- **简介: 该论文属于上肢外骨骼设计任务，旨在解决传统驱动方式带来的重量和复杂性问题。提出一种单缆扭簧驱动机制，并通过仿真优化参数，实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.20893](https://arxiv.org/pdf/2604.20893)**

> **作者:** Juwairiya S. Khan; Mostafa Mohammadi; John Rasmussen; Lotte N.S. Andreasen Struijk
>
> **备注:** 8 pages and 8 figures. Submitted to IEEE/ASME Transactions on Mechatronics. Includes experimental validation on human participants
>
> **摘要:** Wrist exoskeletons play a vital role in rehabilitation and assistive applications, yet conventional actuation mechanisms such as electric motors or pneumatics often introduce undesirable weight, friction, and complexity. This paper presents a novel single-cable (tendon), torsional-spring-assisted actuation mechanism for wrist abduction-adduction, and a simulation-based method for selecting its stiffness parameters. The mechanism employs a single Bowden cable passively tensioned by a spiral torsional spring (clock spring) to maintain continuous cable tension without antagonistic actuation. Kinematic and dynamic modeling of the mechanism was performed to estimate the required torque and identify optimal spring parameters. These simulation-derived parameters guided the design of a functional prototype, which was experimentally evaluated with five participants with no motor disabilities (NMD) under varying arm positions and loading conditions using three spring configurations to account for user variability and modeling uncertainties. Experimental results show consistent agreement with simulation-derived trends, with the nominal spring configuration achieving balanced motion range, torque demand, and repeatability. The results demonstrate that simulation-informed stiffness selection can effectively guide the design of compact, cable-driven wrist exoskeletons while reducing reliance on empirical tuning.
>
---
#### [new 027] X2-N: A Transformable Wheel-legged Humanoid Robot with Dual-mode Locomotion and Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决轮腿机器人稳定性差和操作能力弱的问题。研究提出X2-N机器人，具备双模式运动与操作能力，提升地形适应性和任务灵活性。**

- **链接: [https://arxiv.org/pdf/2604.21541](https://arxiv.org/pdf/2604.21541)**

> **作者:** Yan Ning; Xingzhou Chen; Delong Li; Hao Zhang; Hanfu Gai; Tongyuan Li; Cheng Zhang; Zhihui Peng; Ling Shi
>
> **摘要:** Wheel-legged robots combine the efficiency of wheeled locomotion with the versatility of legged systems, enabling rapid traversal over both continuous and discrete terrains. However, conventional designs typically employ fixed wheels as feet and limited degrees of freedom (DoFs) at the hips, resulting in reduced stability and mobility during legged locomotion compared to humanoids with flat feet. In addition, most existing platforms lack a full upper body with arms, which limits their ability to perform dexterous manipulation tasks. In this letter, we present X2-N, a high-DoF transformable robot with dual-mode locomotion and manipulation. X2-N can operate in both humanoid and wheel-legged forms and transform seamlessly between them through joint reconfiguration. We further propose a reinforcement learning (RL)-based whole-body control framework tailored to this morphology, enabling unified control across hybrid locomotion, transformation, and manipulation. We validate X2-N in a range of challenging locomotion and manipulation tasks, including dynamic skating-like motion, stair climbing and package delivery. Results demonstrate high locomotion efficiency, strong terrain adaptability, and stable loco-manipulation performance of X2-N, highlighting its potential for real-world deployment.
>
---
#### [new 028] Ufil: A Unified Framework for Infrastructure-based Localization
- **分类: cs.RO**

- **简介: 该论文提出Ufil框架，解决基础设施定位中组件耦合问题，通过标准化模型和可复用跟踪组件，实现多源数据融合定位。**

- **链接: [https://arxiv.org/pdf/2604.21471](https://arxiv.org/pdf/2604.21471)**

> **作者:** Simon Schäfer; Lucas Hegerath; Marius Molz; Massimo Marcon; Bassam Alrifaee
>
> **备注:** 8 pages, 6 figures, this work was submitted to IEEE International Conference on Intelligent Transportation Systems (ITSC) 2026
>
> **摘要:** Infrastructure-based localization enhances road safety and traffic management by providing state estimates of road users. Development is hindered by fragmented, application-specific stacks that tightly couple perception, tracking, and middleware. We introduce Ufil, a Unified Framework for Infrastructure-Based Localization with a standardized object model and reusable multi-object tracking components. Ufil offers interfaces and reference implementations for prediction, detection, association, state update, and track management, allowing researchers to improve components without reimplementing the pipeline. Ufil is open-source C++/ROS 2 software with documentation and executable examples. We demonstrate Ufil by integrating three heterogeneous data sources into a single localization pipeline combining (i) vehicle onboard units broadcasting ETSI ITS-G5 Cooperative Awareness Messages, (ii) a lidar-based roadside sensor node, and (iii) an in-road sensitive surface layer. The pipeline runs unchanged in the CARLA simulator and a small-scale CAV testbed, demonstrating Ufil's scale-independent execution model. In a three-lane highway scenario with 423 and 355 vehicles in simulation and testbed, respectively, the fused system achieves lane-level lateral accuracy with mean lateral position RMSEs of 0.31 m in CARLA and 0.29 m in the CPM Lab, and mean absolute orientation errors around 2.2°. Median end-to-end latencies from sensing to fused output remain below 100 ms across all modalities in both environments.
>
---
#### [new 029] Learn Weightlessness: Imitate Non-Self-Stabilizing Motions on Humanoid Robot
- **分类: cs.RO**

- **简介: 该论文属于人形机器人控制任务，旨在解决环境依赖运动中的稳定性问题。通过模仿人类的“失重”状态，设计了一种动态调整关节放松程度的方法，提升机器人与环境的交互能力。**

- **链接: [https://arxiv.org/pdf/2604.21351](https://arxiv.org/pdf/2604.21351)**

> **作者:** Yucheng Xin; Jiacheng Bao; Haoran Yang; Wenqiang Que; Dong Wang; Junbo Tan; Xueqian Wang; Bin Zhao; Xuelong Li
>
> **摘要:** The integration of imitation and reinforcement learning has enabled remarkable advances in humanoid whole-body control, facilitating diverse human-like behaviors. However, research on environment-dependent motions remains limited. Existing methods typically enforce rigid trajectory tracking while neglecting physical interactions with the environment. We observe that humans naturally exploit a "weightless" state during non-self-stabilizing (NSS) motions--selectively relaxing specific joints to allow passive body--environment contact, thereby stabilizing the body and completing the motion. Inspired by this biological mechanism, we design a weightlessness-state auto-labeling strategy for dataset annotation; and we propose the Weightlessness Mechanism (WM), a method that dynamically determines which joints to relax and to what level, together enabling effective environmental interaction while executing target motions. We evaluate our approach on 3 representative NSS tasks: sitting on chairs of varying heights, lying down on beds with different inclinations, and leaning against walls via shoulder or elbow. Extensive experiments in simulation and on the Unitree G1 robot demonstrate that our WM method, trained on single-action demonstrations without any task-specific tuning, achieves strong generalization across diverse environmental configurations while maintaining motion stability. Our work bridges the gap between precise trajectory tracking and adaptive environmental interaction, offering a biologically-inspired solution for contact-rich humanoid control.
>
---
#### [new 030] A Deployable Embodied Vision-Language Navigation System with Hierarchical Cognition and Context-Aware Exploration
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决智能机器人在资源受限下高效推理与部署的问题。提出分模块系统，结合认知图优化导航效率。**

- **链接: [https://arxiv.org/pdf/2604.21363](https://arxiv.org/pdf/2604.21363)**

> **作者:** Kuan Xu; Ruimeng Liu; Yizhuo Yang; Denan Liang; Tongxing Jin; Shenghai Yuan; Chen Wang; Lihua Xie
>
> **备注:** 10 pages, 5 figures,
>
> **摘要:** Bridging the gap between embodied intelligence and embedded deployment remains a key challenge in intelligent robotic systems, where perception, reasoning, and planning must operate under strict constraints on computation, memory, energy, and real-time execution. In vision-language navigation (VLN), existing approaches often face a fundamental trade-off between strong reasoning capabilities and efficient deployment on real-world platforms. In this paper, we present a deployable embodied VLN system that achieves both high efficiency and robust high-level reasoning on real-world robotic platforms. To achieve this, we decouple the system into three asynchronous modules: a real-time perception module for continuous environment sensing, a memory integration module for spatial-semantic aggregation, and a reasoning module for high-level decision making. We incrementally construct a cognitive memory graph to encode scene information, which is further decomposed into subgraphs to enable reasoning with a vision-language model (VLM). To further improve navigation efficiency and accuracy, we also leverage the cognitive memory graph to formulate the exploration problem as a context-aware Weighted Traveling Repairman Problem (WTRP), which minimizes the weighted waiting time of viewpoints. Extensive experiments in both simulation and real-world robotic platforms demonstrate improved navigation success and efficiency over existing VLN approaches, while maintaining real-time performance on resource-constrained hardware.
>
---
#### [new 031] FingerViP: Learning Real-World Dexterous Manipulation with Fingertip Visual Perception
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决单视角感知受限的问题。通过在指尖安装摄像头，实现多视角视觉感知，提升灵巧操作的性能与适应性。**

- **链接: [https://arxiv.org/pdf/2604.21331](https://arxiv.org/pdf/2604.21331)**

> **作者:** Zhen Zhang; Weinan Wang; Hejia Sun; Qingpeng Ding; Xiangyu Chu; Guoxin Fang; K. W. Samuel Au
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** The current practice of dexterous manipulation generally relies on a single wrist-mounted view, which is often occluded and limits performance on tasks requiring multi-view perception. In this work, we present FingerViP, a learning system that utilizes a visuomotor policy with fingertip visual perception for dexterous manipulation. Specifically, we design a vision-enhanced fingertip module with an embedded miniature camera and install the modules on each finger of a multi-fingered hand. The fingertip cameras substantially improve visual perception by providing comprehensive, multi-view feedback of both the hand and its surrounding environment. Building on the integrated fingertip modules, we develop a diffusion-based whole-body visuomotor policy conditioned on a third-view camera and multi-view fingertip vision, which effectively learns complex manipulation skills directly from human demonstrations. To improve view-proprioception alignment and contact awareness, each fingertip visual feature is augmented with its corresponding camera pose encoding and per-finger joint-current encoding. We validate the effectiveness of the multi-view fingertip vision and demonstrate the robustness and adaptability of FingerViP on various challenging real-world tasks, including pressing buttons inside a confined box, retrieving sticks from an unstable support, retrieving objects behind an occluding curtain, and performing long-horizon cabinet opening and object retrieval, achieving an overall success rate of 80.8%. All hardware designs and code will be fully open-sourced.
>
---
#### [new 032] Tempered Sequential Monte Carlo for Trajectory and Policy Optimization with Differentiable Dynamics
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出一种基于采样的轨迹与策略优化框架，解决不同微分动力系统下的控制问题。通过将控制器设计转化为推理任务，利用TSMC方法高效采样最优解。**

- **链接: [https://arxiv.org/pdf/2604.21456](https://arxiv.org/pdf/2604.21456)**

> **作者:** Heng Yang
>
> **摘要:** We propose a sampling-based framework for finite-horizon trajectory and policy optimization under differentiable dynamics by casting controller design as inference. Specifically, we minimize a KL-regularized expected trajectory cost, which yields an optimal "Boltzmann-tilted" distribution over controller parameters that concentrates on low-cost solutions as temperature decreases. To sample efficiently from this sharp, potentially multimodal target, we introduce tempered sequential Monte Carlo (TSMC): an annealing scheme that adaptively reweights and resamples particles along a tempering path from a prior to the target distribution, while using Hamiltonian Monte Carlo rejuvenation to maintain diversity and exploit exact gradients obtained by differentiating through trajectory rollouts. For policy optimization, we extend TSMC via (i) a deterministic empirical approximation of the initial-state distribution and (ii) an extended-space construction that treats rollout randomness as auxiliary variables. Experiments across trajectory- and policy-optimization benchmarks show that TSMC is broadly applicable and compares favorably to state-of-the-art baselines.
>
---
#### [new 033] A Case Study in Recovery of Drones using Discrete-Event Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于 swarm robotics 任务，解决无人机丢失后的恢复问题。通过结合离散事件系统与连续控制器，实现无人机安全恢复和重新编队。**

- **链接: [https://arxiv.org/pdf/2604.21740](https://arxiv.org/pdf/2604.21740)**

> **作者:** Liam P. Burns; Dayse M. Cavalcanti; Felipe G. Cabral; Max H. de Queiroz; Melissa Greeff; Publio M. M. Lima; Karen Rudie
>
> **备注:** Accepted for publication at WODES 2026; final version will appear in IEEE Xplore
>
> **摘要:** Discrete-event systems and supervisory control theory provide a rigorous framework for specifying correct-by-construction behavior. However, their practical application to swarm robotics remains largely underexplored. In this paper, we investigate a topological recovery method based on discrete-event-systems within a swarm robotics context. We propose a hybrid architecture that combines a high-level discrete event systems supervisor with a low-level continuous controller, allowing lost drones to safely recover from fault or attack events and re-enter a controlled region. The method is demonstrated using ten simulated UAVs in the py-bullet-drones framework. We show recovery performance across four distinct scenarios, each with varying initial state estimates. Additionally, we introduce a secondary recovery supervisor that manages the regrouping process for a drone after it has re-entered the operational region.
>
---
#### [new 034] Spectral Kernel Dynamics for Planetary Surface Graphs: Distinction Dynamics and Topological Conservation
- **分类: math.DS; astro-ph.EP; cs.LG; cs.RO**

- **简介: 该论文研究行星表面图谱的谱核动力学，解决拓扑守恒问题。提出区别动力学方程，证明模式保留条件，用于分析排水网络。**

- **链接: [https://arxiv.org/pdf/2604.20887](https://arxiv.org/pdf/2604.20887)**

> **作者:** Jnaneshwar Das
>
> **备注:** 17 pages, 0 figures
>
> **摘要:** The spectral kernel field equation R[k] = T[k] lacks a conservation-law analog. We prove (i) the fixed-point flow is strictly volume-expanding (tr DF > 0), precluding automatic conservation, and (ii) the conservation deficit per mode equals the Hessian stability margin exactly: D_m = -Delta'. Closing the deficit requires a scene-side compensating contribution, which we formalise as the distinction dynamics equation dc/dt = G[c, h_t], with MaxCal-optimal realisation G_opt. On fixed-topology 3D surface graphs we derive a conditional topology-preserving compression theorem: retaining k >= beta_0 + beta_1 modes (under a spectral-ordering assumption) preserves all Betti-number charges; we include a worked short-cycle counterexample (figure-eight) calibrating when the assumption fails. A triple necessary spectral diagnostic -- Fiedler-mode concentration, elevated curl energy, anomalous beta_1 -- is derived for planetary drainage networks at O(N) cost. Two internal real-data sequences serve as preliminary consistency checks; full benchmarks and adaptive-topology extensions are deferred.
>
---
#### [new 035] A Systematic Review and Taxonomy of Reinforcement Learning-Model Predictive Control Integration for Linear Systems
- **分类: eess.SY; cs.AI; cs.RO; math.OC**

- **简介: 该论文属于系统综述任务，旨在整合强化学习与模型预测控制，解决线性系统中的控制问题。通过分类和分析现有研究，识别设计模式与挑战。**

- **链接: [https://arxiv.org/pdf/2604.21030](https://arxiv.org/pdf/2604.21030)**

> **作者:** Mohsen Jalaeian Farimani; Roya Khalili Amirabadi; Davoud Nikkhouy; Malihe Abdolbaghi; Mahshad Rastegarmoghaddam; Shima Samadzadeh
>
> **摘要:** The integration of Model Predictive Control (MPC) and Reinforcement Learning (RL) has emerged as a promising paradigm for constrained decision-making and adaptive control. MPC offers structured optimization, explicit constraint handling, and established stability tools, whereas RL provides data-driven adaptation and performance improvement in the presence of uncertainty and model mismatch. Despite the rapid growth of research on RL--MPC integration, the literature remains fragmented, particularly for control architectures built on linear or linearized predictive models. This paper presents a comprehensive Systematic Literature Review (SLR) of RL--MPC integrations for linear and linearized systems, covering peer-reviewed and formally indexed studies published until 2025. The reviewed studies are organized through a multi-dimensional taxonomy covering RL functional roles, RL algorithm classes, MPC formulations, cost-function structures, and application domains. In addition, a cross-dimensional synthesis is conducted to identify recurring design patterns and reported associations among these dimensions within the reviewed corpus. The review highlights methodological trends, commonly adopted integration strategies, and recurring practical challenges, including computational burden, sample efficiency, robustness, and closed-loop guarantees. The resulting synthesis provides a structured reference for researchers and practitioners seeking to design or analyze RL--MPC architectures based on linear or linearized predictive control formulations.
>
---
#### [new 036] Planetary Exploration 3.0: A Roadmap for Software-Defined, Radically Adaptive Space Systems
- **分类: astro-ph.IM; astro-ph.EP; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于行星探测任务，旨在解决外太阳系探索难题。提出PE 3.0新范式，通过软件定义系统实现适应性探测，开展三项任务概念设计。**

- **链接: [https://arxiv.org/pdf/2604.20910](https://arxiv.org/pdf/2604.20910)**

> **作者:** Masahiro Ono; Daniel Selva; Morgan L. Cable; Marie Ethvignot; Margaret Hansen; Andreas M. Hein; Elena-Sorina Lupu; Zachary Manchester; David Murrow; Chad Pozarycki; Pascal Spino; Amanda Stockton; Mathieu Choukroun; Soon-Jo Chung; John Day; Alexander Demagall; Anthony Freeman; Chloe Gentgen; Michel D. Ingham; Charity M. Phillips-Lander; Richard Rieber; Alejandro Salado; Maria Sakovsky; Lori R. Shiraishi; Yisong Yue; Kris Zacny
>
> **摘要:** The surface and subsurface of worlds beyond Mars remain largely unexplored. Yet these worlds hold keys to fundamental questions in planetary science - from potentially habitable subsurface oceans on icy moons to ancient records preserved in Kuiper Belt objects. NASA's success in Mars exploration was achieved through incrementalism: 22 progressively sophisticated missions over decades. This paradigm, which we call Planetary Exploration 2.0 (PE 2.0), is untenable for the outer Solar System, where cruise times of a decade or more make iterative missions infeasible. We propose Planetary Exploration 3.0 (PE 3.0): a paradigm in which unvisited worlds are explored by a single or a few missions with radically adaptive space systems. A PE 3.0 mission conducts both initial exploratory science and follow-on hypothesis-driven science based on its own in situ data returns, evolving spacecraft capabilities to work resiliently in previously unseen environments. The key enabler of PE 3.0 is software-defined space systems (SDSSs) - systems that can adapt their functions at all levels through software updates. This paper presents findings from a Keck Institute for Space Studies (KISS) workshop on PE 3.0, covering: (1) PE 3.0 systems engineering including science definition, architecture, design methods, and verification & validation; (2) software-defined space system technologies including reconfigurable hardware, multi-functionality, and modularity; (3) onboard intelligence including autonomous science, navigation, controls, and embodied AI; and (4) three PE 3.0 mission concepts: a Neptune/Triton smart flyby, an ocean world explorer, and an Oort cloud reconnaissance mission.
>
---
#### [new 037] Task-specific Subnetwork Discovery in Reinforcement Learning for Autonomous Underwater Navigation
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究多任务强化学习中的任务特定子网络，旨在提升水下自主导航的透明度与可解释性。通过分析预训练网络结构，发现仅1.5%权重用于任务区分，其中85%连接上下文变量，为模型优化提供依据。**

- **链接: [https://arxiv.org/pdf/2604.21640](https://arxiv.org/pdf/2604.21640)**

> **作者:** Yi-Ling Liu; Melvin Laux; Mariela De Lucas Alvarez; Frank Kirchner; Rebecca Adam
>
> **备注:** To be published in IEEE OCEANS 2026 (Sanya) conference proceedings
>
> **摘要:** Autonomous underwater vehicles are required to perform multiple tasks adaptively and in an explainable manner under dynamic, uncertain conditions and limited sensing, challenges that classical controllers struggle to address. This demands robust, generalizable, and inherently interpretable control policies for reliable long-term monitoring. Reinforcement learning, particularly multi-task RL, overcomes these limitations by leveraging shared representations to enable efficient adaptation across tasks and environments. However, while such policies show promising results in simulation and controlled experiments, they yet remain opaque and offer limited insight into the agent's internal decision-making, creating gaps in transparency, trust, and safety that hinder real-world deployment. The internal policy structure and task-specific specialization remain poorly understood. To address these gaps, we analyze the internal structure of a pretrained multi-task reinforcement learning network in the HoloOcean simulator for underwater navigation by identifying and comparing task-specific subnetworks responsible for navigating toward different species. We find that in a contextual multi-task reinforcement learning setting with related tasks, the network uses only about 1.5% of its weights to differentiate between tasks. Of these, approximately 85% connect the context-variable nodes in the input layer to the next hidden layer, highlighting the importance of context variables in such settings. Our approach provides insights into shared and specialized network components, useful for efficient model editing, transfer learning, and continual learning for underwater monitoring through a contextual multi-task reinforcement learning method.
>
---
## 更新

#### [replaced 001] Demystifying Action Space Design for Robotic Manipulation Policies
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控策略学习任务，旨在解决动作空间设计对策略学习的影响问题。通过大量实验，分析了不同动作表示方式的优劣，提出了优化动作空间设计的方法。**

- **链接: [https://arxiv.org/pdf/2602.23408](https://arxiv.org/pdf/2602.23408)**

> **作者:** Yuchun Feng; Jinliang Zheng; Zhihao Wang; Dongxiu Liu; Jianxiong Li; Jiangmiao Pang; Tai Wang; Xianyuan Zhan
>
> **摘要:** The specification of the action space plays a pivotal role in imitation-based robotic manipulation policy learning, fundamentally shaping the optimization landscape of policy learning. While recent advances have focused heavily on scaling training data and model capacity, the choice of action space remains guided by ad-hoc heuristics or legacy designs, leading to an ambiguous understanding of robotic policy design philosophies. To address this ambiguity, we conducted a large-scale and systematic empirical study, confirming that the action space does have significant and complex impacts on robotic policy learning. We dissect the action design space along temporal and spatial axes, facilitating a structured analysis of how these choices govern both policy learnability and control stability. Based on 13,000+ real-world rollouts on a bimanual robot and evaluation on 500+ trained models over four scenarios, we examine the trade-offs between absolute vs. delta representations, and joint-space vs. task-space parameterizations. Our large-scale results suggest that properly designing the policy to predict delta actions consistently improves performance, while joint-space and task-space representations offer complementary strengths, favoring control stability and generalization, respectively.
>
---
#### [replaced 002] Language-Conditioned Safe Trajectory Generation for Spacecraft Rendezvous
- **分类: cs.RO; cs.AI; math.OC**

- **简介: 该论文属于航天自主导航任务，解决复杂任务中轨迹生成依赖专家输入的问题，提出SAGES框架，通过自然语言生成符合约束的轨迹。**

- **链接: [https://arxiv.org/pdf/2512.09111](https://arxiv.org/pdf/2512.09111)**

> **作者:** Yuji Takubo; Arpit Dwivedi; Sukeerth Ramkumar; Luis A. Pabon; Daniele Gammelli; Marco Pavone; Simone D'Amico
>
> **备注:** 42 pages, 12 figures. Submitted to AIAA Journal of Guidance, Control, and Dynamics
>
> **摘要:** Reliable real-time trajectory generation is essential for future autonomous spacecraft. While recent progress in nonconvex guidance and control is paving the way for onboard autonomous trajectory optimization, these methods still rely on extensive expert input (e.g., waypoints, constraints, mission timelines, etc.), which limits operational scalability in complex missions such as rendezvous and proximity operations. This paper introduces SAGES (Semantic Autonomous Guidance Engine for Space), a trajectory-generation framework that translates natural-language commands into spacecraft trajectories that reflect high-level intent while respecting nonconvex constraints. Experiments in two settings (fault-tolerant proximity operations with continuous-time constraint enforcement and a free-flying robotic platform) demonstrate that SAGES reliably produces trajectories aligned with human commands, achieving over 90% semantic-behavioral consistency across diverse behavior modes. Ultimately, this work marks an initial step toward language-conditioned, constraint-aware spacecraft trajectory generation, enabling operators to interactively guide both safety and behavior through intuitive natural-language commands with reduced expert burden.
>
---
#### [replaced 003] PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景理解任务，旨在解决语义表示在2D与3D间语言对齐及冗余问题。提出PLAF框架，实现像素级语义对齐与高效存储查询。**

- **链接: [https://arxiv.org/pdf/2604.15770](https://arxiv.org/pdf/2604.15770)**

> **作者:** Junjie Wen; Junlin He; Fei Ma; Jinqiang Cui
>
> **备注:** Accepted by ICCA 2026
>
> **摘要:** Accurate open-vocabulary 3D scene understanding requires semantic representations that are both language-aligned and spatially precise at the pixel level, while remaining scalable when lifted to 3D space. However, existing representations struggle to jointly satisfy these requirements, and densely propagating pixel-wise semantics to 3D often results in substantial redundancy, leading to inefficient storage and querying in large-scale scenes. To address these challenges, we present \emph{PLAF}, a Pixel-wise Language-Aligned Feature extraction framework that enables dense and accurate semantic alignment in 2D without sacrificing open-vocabulary expressiveness. Building upon this representation, we further design an efficient semantic storage and querying scheme that significantly reduces redundancy across both 2D and 3D domains. Experimental results show that \emph{PLAF} provides a strong semantic foundation for accurate and efficient open-vocabulary 3D scene understanding. The codes are publicly available at this https URL.
>
---
#### [replaced 004] Fake or Real, Can Robots Tell? Evaluating VLM Robustness to Domain Shift in Single-View Robotic Scene Understanding
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文研究机器人场景理解任务，解决VLM在领域迁移下的鲁棒性问题。通过对比真实与3D打印物体，评估模型性能及评价指标的可靠性。**

- **链接: [https://arxiv.org/pdf/2506.19579](https://arxiv.org/pdf/2506.19579)**

> **作者:** Federico Tavella; Amber Drinkwater; Angelo Cangelosi
>
> **摘要:** Robotic scene understanding increasingly relies on Vision-Language Models (VLMs) to generate natural language descriptions of the environment. In this work, we systematically evaluate single-view object captioning for tabletop scenes captured by a robotic manipulator, introducing a controlled physical domain shift that contrasts real-world tools with geometrically similar 3D-printed counterparts that differ in texture, colour, and material. We benchmark a suite of state-of-the-art, locally deployable VLMs across multiple metrics to assess semantic alignment and factual grounding. Our results demonstrate that while VLMs describe common real-world objects effectively, performance degrades markedly on 3D-printed items despite their structurally familiar forms. We further expose critical vulnerabilities in standard evaluation metrics, showing that some fail to detect domain shifts entirely or reward fluent but factually incorrect captions. These findings highlight the limitations of deploying foundation models for embodied agents and the need for more robust architectures and evaluation protocols in physical robotic applications.
>
---
#### [replaced 005] Rectified Schrödinger Bridge Matching for Few-Step Visual Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉导航任务，解决生成策略需多步积分的问题。提出RSBM框架，通过调节熵正则化参数，在少步内实现高精度导航。**

- **链接: [https://arxiv.org/pdf/2604.05673](https://arxiv.org/pdf/2604.05673)**

> **作者:** Wuyang Luan; Junhui Li; Weiguang Zhao; Wenjian Zhang; Tieru Wu; Rui Ma
>
> **备注:** 18 pages, 7 figures, 10 tables. Code available at this https URL
>
> **摘要:** Visual navigation is a core challenge in Embodied AI, requiring autonomous agents to translate high-dimensional sensory observations into continuous, long-horizon action trajectories. While generative policies based on diffusion models and Schrödinger Bridges (SB) effectively capture multimodal action distributions, they require dozens of integration steps due to high-variance stochastic transport, posing a critical barrier for real-time robotic control. We propose Rectified Schrödinger Bridge Matching (RSBM), a framework that exploits a shared velocity-field structure between standard Schrödinger Bridges ($\varepsilon=1$, maximum-entropy transport) and deterministic Optimal Transport ($\varepsilon\to 0$, as in Conditional Flow Matching), controlled by a single entropic regularization parameter $\varepsilon$. We prove two key results: (1) the conditional velocity field's functional form is invariant across the entire $\varepsilon$-spectrum (Velocity Structure Invariance), enabling a single network to serve all regularization strengths; and (2) reducing $\varepsilon$ linearly decreases the conditional velocity variance, enabling more stable coarse-step ODE integration. Anchored to a learned conditional prior that shortens transport distance, RSBM operates at an intermediate $\varepsilon$ that balances multimodal coverage and path straightness. Empirically, while standard bridges require $\geq 10$ steps to converge, RSBM achieves over 94% cosine similarity and 92% success rate in merely 3 integration steps -- without distillation or multi-stage training -- substantially narrowing the gap between high-fidelity generative policies and the low-latency demands of Embodied AI.
>
---
#### [replaced 006] Reinforcement Learning with Foundation Priors: Let the Embodied Agent Efficiently Learn on Its Own
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人强化学习任务，旨在解决真实世界中RL数据消耗大、奖励函数设计困难的问题。提出RLFP框架，利用基础模型提升学习效率与自主性。**

- **链接: [https://arxiv.org/pdf/2310.02635](https://arxiv.org/pdf/2310.02635)**

> **作者:** Weirui Ye; Yunsheng Zhang; Haoyang Weng; Xianfan Gu; Shengjie Wang; Tong Zhang; Mengchen Wang; Pieter Abbeel; Yang Gao
>
> **备注:** CoRL 2024 (Oral)
>
> **摘要:** Reinforcement learning (RL) is a promising approach for solving robotic manipulation tasks. However, it is challenging to apply the RL algorithms directly in the real world. For one thing, RL is data-intensive and typically requires millions of interactions with environments, which are impractical in real scenarios. For another, it is necessary to make heavy engineering efforts to design reward functions manually. To address these issues, we leverage foundation models in this paper. We propose Reinforcement Learning with Foundation Priors (RLFP) to utilize guidance and feedback from policy, value, and success-reward foundation models. Within this framework, we introduce the Foundation-guided Actor-Critic (FAC) algorithm, which enables embodied agents to explore more efficiently with automatic reward functions. The benefits of our framework are threefold: (1) \textit{sample efficient}; (2) \textit{minimal and effective reward engineering}; (3) \textit{agnostic to foundation model forms and robust to noisy priors}. Our method achieves remarkable performances in various manipulation tasks on both real robots and in simulation. Across 5 dexterous tasks with real robots, FAC achieves an average success rate of 86\% after one hour of real-time learning. Across 8 tasks in the simulated Meta-world, FAC achieves 100\% success rates in 7/8 tasks under less than 100k frames (about 1-hour training), outperforming baseline methods with manual-designed rewards in 1M frames. We believe the RLFP framework can enable future robots to explore and learn autonomously in the physical world for more tasks. Visualizations and code are available at this https URL.
>
---
#### [replaced 007] JoyAI-RA 0.1: A Foundation Model for Robotic Autonomy
- **分类: cs.RO**

- **简介: 该论文提出JoyAI-RA，一个用于机器人自主的视觉-语言-动作基础模型，解决数据多样性不足和跨体感泛化差的问题。通过多源预训练提升机器人操作的通用性。**

- **链接: [https://arxiv.org/pdf/2604.20100](https://arxiv.org/pdf/2604.20100)**

> **作者:** Tianle Zhang; Zhihao Yuan; Dafeng Chi; Peidong Liu; Dongwei Li; Kejun Hu; Likui Zhang; Junnan Nie; Ziming Wei; Zengjue Chen; Yili Tang; Jiayi Li; Zhiyuan Xiang; Mingyang Li; Tianci Luo; Hanwen Wan; Ao Li; Linbo Zhai; Zhihao Zhan; Xiaodong Bai; Jiakun Cai; Peng Cao; Kangliang Chen; Siang Chen; Yixiang Dai; Shuai Di; Yicheng Gong; Chenguang Gui; Yucheng Guo; Peng Hao; Qingrong He; Haoyang Huang; Kunrui Huang; Zhixuan Huang; Shibo Jin; Yixiang Jin; Anson Li; Dongjiang Li; Jiawei Li; Ruodai Li; Yihang Li; Yuzhen Li; Jiaming Liang; Fangsheng Liu; Jing Long; Mingxi Luo; Xing Pan; Hui Shen; Xiaomeng Tian; Daming Wang; Song Wang; Junwu Xiong; Hang Xu; Wanting Xu; Zhengcheng Yu; He Zhang; Jiyao Zhang; Lin Zhao; Chen Zhou; Nan Duan; Yuzheng Zhuang; Liang Lin
>
> **摘要:** Robotic autonomy in open-world environments is fundamentally limited by insufficient data diversity and poor cross-embodiment generalization. Existing robotic datasets are often limited in scale and task coverage, while relatively large differences across robot embodiments impede effective behavior knowledge transfer. To address these challenges, we propose JoyAI-RA, a vision-language-action (VLA) embodied foundation model tailored for generalizable robotic manipulation. JoyAI-RA presents a multi-source multi-level pretraining framework that integrates web data, large-scale egocentric human manipulation videos, simulation-generated trajectories, and real-robot data. Through training on heterogeneous multi-source data with explicit action-space unification, JoyAI-RA effectively bridges embodiment gaps, particularly between human manipulation and robotic control, thereby enhancing cross-embodiment behavior learning. JoyAI-RA outperforms state-of-the-art methods in both simulation and real-world benchmarks, especially on diverse tasks with generalization demands.
>
---
#### [replaced 008] MOMO: A framework for seamless physical, verbal, and graphical robot skill learning and adaptation
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出MOMO框架，解决工业机器人技能适应问题。通过触觉、语言和图形界面实现灵活控制，提升非专家用户操作效率。**

- **链接: [https://arxiv.org/pdf/2604.20468](https://arxiv.org/pdf/2604.20468)**

> **作者:** Markus Knauer; Edoardo Fiorini; Maximilian Mühlbauer; Stefan Schneyer; Promwat Angsuratanawech; Florian Samuel Lay; Timo Bachmann; Samuel Bustamante; Korbinian Nottensteiner; Freek Stulp; Alin Albu-Schäffer; João Silvério; Thomas Eiband
>
> **备注:** 15 pages, 13 figures, 3 tables
>
> **摘要:** Industrial robot applications require increasingly flexible systems that non-expert users can easily adapt for varying tasks and environments. However, different adaptations benefit from different interaction modalities. We present an interactive framework that enables robot skill adaptation through three complementary modalities: kinesthetic touch for precise spatial corrections, natural language for high-level semantic modifications, and a graphical web interface for visualizing geometric relations and trajectories, inspecting and adjusting parameters, and editing via-points by drag-and-drop. The framework integrates five components: energy-based human-intention detection, a tool-based LLM architecture (where the LLM selects and parameterizes predefined functions rather than generating code) for safe natural language adaptation, Kernelized Movement Primitives (KMPs) for motion encoding, probabilistic Virtual Fixtures for guided demonstration recording, and ergodic control for surface finishing. We demonstrate that this tool-based LLM architecture generalizes skill adaptation from KMPs to ergodic control, enabling voice-commanded surface finishing. Validation on a 7-DoF torque-controlled robot at the Automatica 2025 trade fair demonstrates the practical applicability of our approach in industrial settings.
>
---
#### [replaced 009] ExpressMM: Expressive Mobile Manipulation Behaviors in Human-Robot Interactions
- **分类: cs.RO**

- **简介: 该论文提出ExpressMM框架，解决人机协作中机器人表达性行为的问题。通过语言引导的规划与视觉-语言-动作策略，实现自然、可中断的人机交互。**

- **链接: [https://arxiv.org/pdf/2604.05320](https://arxiv.org/pdf/2604.05320)**

> **作者:** Souren Pashangpour; Haitong Wang; Matthew Lisondra; Goldie Nejat
>
> **摘要:** Mobile manipulators are increasingly deployed in human-centered environments to perform tasks. While completing such tasks, they should also be able to communicate their intent to the people around them using expressive robot behaviors. Prior work on expressive robot behaviors has used preprogrammed or learning-from-demonstration-based expressive motions and large language model generated high-level interactions. The majority of these existing approaches have not considered human-robot interactions (HRI) where users may interrupt, modify, or redirect a robot's actions during task execution. In this paper, we develop the novel ExpressMM framework that integrates a high-level language-guided planner based on a vision-language model for perception and conversational reasoning with a low-level vision-language-action policy to generate expressive robot behaviors during collaborative HRI tasks. Furthermore, ExpressMM supports interruptible interactions to accommodate updated or redirecting instructions by users. We demonstrate ExpressMM on a mobile manipulator assisting a human in a collaborative assembly scenario and conduct audience-based evaluation of live HRI demonstrations. Questionnaire results show that the ExpressMM-enabled expressive behaviors helped observers clearly interpret the robot's actions and intentions while supporting socially appropriate and understandable interactions. Participants also reported that the robot was useful for collaborative tasks and behaved in a predictable and safe manner during the demonstrations, fostering positive perceptions of the robot's usefulness, safety, and predictability during the collaborative tasks.
>
---
#### [replaced 010] Situationally-aware Path Planning Exploiting 3D Scene Graphs
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，旨在提升基于3D场景图的路径规划效率与可解释性。通过构建语义图进行分阶段规划，并引入重规划机制，显著减少规划时间。**

- **链接: [https://arxiv.org/pdf/2508.06283](https://arxiv.org/pdf/2508.06283)**

> **作者:** Saad Ejaz; Marco Giberna; Muhammad Shaheer; Jose Andres Millan-Romera; Ali Tourani; Paul Kremer; Holger Voos; Jose Luis Sanchez-Lopez
>
> **摘要:** 3D Scene Graphs integrate both metric and semantic information, yet their structure remains underutilized for improving path planning efficiency and interpretability. In this work, we present S-Path, a situationally-aware path planner that leverages the metric-semantic structure of indoor 3D Scene Graphs to significantly enhance planning efficiency. S-Path follows a two-stage process: it first performs a search over a semantic graph derived from the scene graph to yield a human-understandable high-level path. This also identifies relevant regions for planning, which later allows the decomposition of the problem into smaller, independent subproblems that can be solved in parallel. We also introduce a replanning mechanism that, in the event of an infeasible path, reuses information from previously solved subproblems to update semantic heuristics and prioritize reuse to further improve the efficiency of future planning attempts. Extensive experiments on both real-world and simulated environments show that S-Path achieves average reductions of 6x in planning time while maintaining comparable path optimality to classical sampling-based planners and surpassing them in complex scenarios, making it an efficient and interpretable path planner for environments represented by indoor 3D Scene Graphs. Code available at: this https URL
>
---
#### [replaced 011] FingerEye: Continuous and Unified Vision-Tactile Sensing for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文提出FingerEye传感器，解决机器人操作中连续感知问题，通过视觉-触觉融合实现精确操控。**

- **链接: [https://arxiv.org/pdf/2604.20689](https://arxiv.org/pdf/2604.20689)**

> **作者:** Zhixuan Xu; Yichen Li; Xuanye Wu; Tianyu Qiu; Lin Shao
>
> **摘要:** Dexterous robotic manipulation requires comprehensive perception across all phases of interaction: pre-contact, contact initiation, and post-contact. Such continuous feedback allows a robot to adapt its actions throughout interaction. However, many existing tactile sensors, such as GelSight and its variants, only provide feedback after contact is established, limiting a robot's ability to precisely initiate contact. We introduce FingerEye, a compact and cost-effective sensor that provides continuous vision-tactile feedback throughout the interaction process. FingerEye integrates binocular RGB cameras to provide close-range visual perception with implicit stereo depth. Upon contact, external forces and torques deform a compliant ring structure; these deformations are captured via marker-based pose estimation and serve as a proxy for contact wrench sensing. This design enables a perception stream that smoothly transitions from pre-contact visual cues to post-contact tactile feedback. Building on this sensing capability, we develop a vision-tactile imitation learning policy that fuses signals from multiple FingerEye sensors to learn dexterous manipulation behaviors from limited real-world data. We further develop a digital twin of our sensor and robot platform to improve policy generalization. By combining real demonstrations with visually augmented simulated observations for representation learning, the learned policies become more robust to object appearance variations. Together, these design aspects enable dexterous manipulation across diverse object properties and interaction regimes, including coin standing, chip picking, letter retrieving, and syringe manipulation. The hardware design, code, appendix, and videos are available on our project website: this https URL
>
---
#### [replaced 012] Geometry-aided Vision-based Localization of Future Mars Helicopters in Challenging Illumination Conditions
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决火星直升机在光照变化下的定位问题。提出Geo-LoFTR模型，提升图像匹配鲁棒性。**

- **链接: [https://arxiv.org/pdf/2502.09795](https://arxiv.org/pdf/2502.09795)**

> **作者:** Dario Pisanti; Robert Hewitt; Roland Brockers; Georgios Georgakis
>
> **摘要:** Planetary exploration using aerial assets has the potential for unprecedented scientific discoveries on Mars. While NASA's Mars helicopter Ingenuity proved flight in Martian atmosphere is possible, future Mars rotorcraft will require advanced navigation capabilities for long-range flights. One such critical capability is Map-based Localization (MbL) which registers an onboard image to a reference map during flight to mitigate cumulative drift from visual odometry. However, significant illumination differences between rotorcraft observations and a reference map prove challenging for traditional MbL systems, restricting the operational window of the vehicle. In this work, we investigate a new MbL system and propose Geo-LoFTR, a geometry-aided deep learning model for image registration that is more robust under large illumination differences than prior models. The system is supported by a custom simulation framework that uses real orbital maps to produce large amounts of realistic images of the Martian terrain. Comprehensive evaluations show that our proposed system outperforms prior MbL efforts in terms of localization accuracy under significant lighting and scale variations. Furthermore, we demonstrate the validity of our approach across a simulated Martian day and on real Mars imagery. Code and datasets are available at: this https URL.
>
---
#### [replaced 013] Scensory: Real-Time Robotic Olfactory Perception for Joint Identification and Source Localization
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于机器人嗅觉感知任务，解决从扩散化学信号中识别真菌种类并定位来源的问题。通过学习框架Scensory实现高效、实时的环境监测。**

- **链接: [https://arxiv.org/pdf/2509.19318](https://arxiv.org/pdf/2509.19318)**

> **作者:** Yanbaihui Liu; Erica Babusci; Claudia K. Gunsch; Boyuan Chen
>
> **备注:** Our project website is at: this http URL
>
> **摘要:** While robotic perception has advanced rapidly in vision and touch, enabling robots to reason about indoor fungal contamination from weak, diffusion-dominated chemical signals remains an open challenge. We introduce Scensory, a learning-based robotic olfaction framework that simultaneously identifies fungal species and localizes their source from short time series measured by affordable, cross-sensitive VOC sensor arrays. Temporal VOC dynamics encode both chemical and spatial signatures, which we decode through neural networks trained on robot-automated data collection with spatial supervision. Across five fungal species, Scensory achieves up to 89.85% species accuracy and 87.31% source localization accuracy under ambient conditions with 3-7s sensor inputs. These results demonstrate real-time, spatially grounded perception from diffusion-dominated chemical signals, enabling scalable and low-cost source localization for robotic indoor environmental monitoring.
>
---
#### [replaced 014] Certified Coil Geometry Learning for Short-Range Magnetic Actuation and Spacecraft Docking Application
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于磁场建模任务，旨在解决多卫星对接中磁场计算精度与效率的问题。通过学习框架近似精确磁场模型，提升计算效率并保证精度。**

- **链接: [https://arxiv.org/pdf/2507.03806](https://arxiv.org/pdf/2507.03806)**

> **作者:** Yuta Takahashi; Hayate Tajima; Shin-ichiro Sakai
>
> **备注:** IEEE Robotics and Automation Letters. Preprint Version. Accepted March, 2026 (DOI: this https URL)
>
> **摘要:** This paper presents a learning-based framework for approximating an exact magnetic-field interaction model, supported by both numerical and experimental validation. High-fidelity magnetic-field interaction modeling is essential for achieving exceptional accuracy and responsiveness across a wide range of fields, including transportation, energy systems, medicine, biomedical robotics, and aerospace robotics. In aerospace engineering, magnetic actuation has been investigated as a fuel-free solution for multi-satellite attitude and formation control. Although the exact magnetic field can be computed from the Biot-Savart law, the associated computational cost is prohibitive, and prior studies have therefore relied on dipole approximations to improve efficiency. However, these approximations lose accuracy during proximity operations, leading to unstable behavior and even collisions. To address this limitation, we develop a learning-based approximation framework that faithfully reproduces the exact field while dramatically reducing computational cost. This framework directly derives a coefficient matrix that maps inter-satellite current vectors to the resulting forces and torques, enabling efficient computation of control current commands. The proposed method additionally provides a certified error bound, derived from the number of training samples, ensuring reliable prediction accuracy. The learned model can also accommodate interactions between coils of different sizes through appropriate geometric transformations, without retraining. To verify the effectiveness of the proposed framework under challenging conditions, a spacecraft docking scenario is examined through both numerical simulations and experimental validation.
>
---
#### [replaced 015] Learning Physics from Pretrained Video Models: A Multimodal Continuous and Sequential World Interaction Models for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决数据稀缺问题。通过利用预训练视频模型，提出PhysGen框架，将视频与动作统一为物理标记，提升机器人操作能力。**

- **链接: [https://arxiv.org/pdf/2603.00110](https://arxiv.org/pdf/2603.00110)**

> **作者:** Zijian Song; Qichang Li; Sihan Qin; Yuhao Chen; Tianshui Chen; Liang Lin; Guangrun Wang
>
> **备注:** 11 pages, 6 figures. arXiv admin note: text overlap with arXiv:2508.09822
>
> **摘要:** The scarcity of large-scale robotic data has motivated the repurposing of foundation models from other modalities for policy learning. In this work, we introduce PhysGen (Learning Physics from Pretrained Video Generation Models), a scalable continuous and sequential world interaction framework that leverages autoregressive video generation to solve robotic manipulation tasks. By treating the pretrained video model as a proxy for a physics simulator, PhysGen models the dynamic interplay between the external environment and robot actions. We introduce a multimodal continuous representation that unifies video and action into shared physical tokens, bridging the gap between discrete video generation and continuous robotic control. This approach enables the seamless transfer of implicit physical knowledge-such as object permanence and dynamics-from video pretraining to downstream this http URL ensure efficient convergence, we incorporate causal masking, inverse kinematics, Lookahead Multi-Token Prediction (L-MTP), and key-value (KV) caching. Experimental results on the Libero and ManiSkill benchmarks demonstrate that PhysGen consistently outperforms robust baselines, surpassing OpenVLA and WorldVLA by margins of 13.8% and 8.8%, respectively. Notably, in real-world scenarios, PhysGen matches the performance of large-scale action-pretrained models like $\pi_0$ without requiring prior action-specific pretraining, demonstrating superior capability in physically complex tasks such as grasping transparent objects. These findings validate the potential of extracting physical intuition from pretrained video generators to facilitate generalizable robotic manipulation.
>
---
#### [replaced 016] ZipFold: Modular Actuators for Scaleable Adaptive Robots
- **分类: cs.RO; cond-mat.soft; cs.HC**

- **简介: 该论文属于机器人适应性设计任务，旨在解决传统形状变化系统难以扩展和重构的问题。提出一种可折叠变形的致动器，实现模块化、可调节的机器人结构。**

- **链接: [https://arxiv.org/pdf/2604.05260](https://arxiv.org/pdf/2604.05260)**

> **作者:** Niklas Hagemann; Daniela Rus
>
> **摘要:** There is a growing need for robots that can change their shape, size and mechanical properties to adapt to evolving tasks and environments. However, current shape-changing systems generally utilize bespoke, system-specific mechanisms that can be difficult to scale, reconfigure or translate from one application to another. This paper introduces a compact, easy-to-fabricate deployable actuator that achieves reversible scale and stiffness transformations through compound folding and zipping of flexible 3D-printed plastic strips into square-section deployable beams. The simple actuation method allows for smooth, continuous transitions between compact (flexible) and expanded (quasi-rigid) states, facilitating diverse shape and stiffness transformations when modules are combined into larger assemblies. The actuator's mechanical performance is characterized and an integrated system involving a four-module adaptive walking robot is demonstrated.
>
---
#### [replaced 017] Stratified Topological Autonomy for Long-Range Coordination (STALC)
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出STALC，用于多机器人长距离协调的分层规划方法，解决复杂环境下的多机器人路径规划问题。通过拓扑图与混合整数规划结合，实现高效协同。**

- **链接: [https://arxiv.org/pdf/2503.10475](https://arxiv.org/pdf/2503.10475)**

> **作者:** Cora A. Duggan; Adam Goertz; Adam Polevoy; Mark Gonzales; Kevin C. Wolfe; Bradley Woosley; John G. Rogers III; Joseph Moore
>
> **备注:** ©2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** In this paper, we present Stratified Topological Autonomy for Long-Range Coordination (STALC), a hierarchical planning approach for multi-robot coordination in real-world environments with significant inter-robot spatial and temporal dependencies. At its core, STALC consists of a multi-robot graph-based planner which combines a topological graph with a novel, computationally efficient mixed-integer programming formulation to generate highly-coupled multi-robot plans in seconds. To enable autonomous planning across different spatial and temporal scales, we construct our graphs so that they capture connectivity between free-space regions and other problem-specific features, such as traversability or risk. We then use receding-horizon planners to achieve local collision avoidance and formation control. To evaluate our approach, we consider a multi-robot reconnaissance scenario where robots must autonomously coordinate to navigate through an environment while minimizing the risk of detection by observers. Through simulation-based experiments, we show that our approach is able to scale to address complex multi-robot planning scenarios. Through hardware experiments, we demonstrate our ability to generate graphs from real-world data and successfully plan across the entire hierarchy to achieve shared objectives.
>
---
#### [replaced 018] Efficient Emotion-Aware Iconic Gesture Prediction for Robot Co-Speech
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于情感感知的图标手势预测任务，旨在提升机器人共言手势的语义表达。解决现有系统生成手势缺乏语义重点的问题，通过轻量级Transformer模型，仅凭文本和情感信息生成手势，无需音频输入。**

- **链接: [https://arxiv.org/pdf/2604.11417](https://arxiv.org/pdf/2604.11417)**

> **作者:** Edwin C. Montiel-Vazquez; Christian Arzate Cruz; Stefanos Gkikas; Thomas Kassiotis; Giorgos Giannakakis; Randy Gomez
>
> **摘要:** Co-speech gestures increase engagement and improve speech understanding. Most data-driven robot systems generate rhythmic beat-like motion, yet few integrate semantic emphasis. To address this, we propose a lightweight transformer that derives iconic gesture placement and intensity from text and emotion alone, requiring no audio input at inference time. The model outperforms GPT-4o in both semantic gesture placement classification and intensity regression on the BEAT2 dataset, while remaining computationally compact and suitable for real-time deployment on embodied agents.
>
---
#### [replaced 019] Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于机器人定位任务，解决农业环境中基于LiDAR的场景识别问题。提出MinkUNeXt-VINE方法，采用多损失结构提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2601.18714](https://arxiv.org/pdf/2601.18714)**

> **作者:** Judith Vilella-Cantos; Mauro Martini; Marcello Chiaberge; Mónica Ballesta; David Valiente
>
> **摘要:** Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks. Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art. In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach. Our method prioritizes enhanced performance with low-cost, sparse LiDAR inputs and lower-dimensionality outputs to ensure high efficiency in real-time scenarios. Additionally, we present a comprehensive ablation study of the results on various evaluation cases and two extensive long-term vineyard datasets employing different LiDAR sensors. The results demonstrate the efficiency of the trade-off output produced by this approach, as well as its robust performance on low-cost and low-resolution input data. The code is publicly available for reproduction.
>
---
