# 机器人 cs.RO

- **最新发布 32 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] Plant-Inspired Robot Design Metaphors for Ambient HRI
- **分类: cs.RO; cs.HC**

- **简介: 论文探讨植物作为人机交互设计隐喻的潜力，旨在解决传统拟人化交互方式过于显性的问题。通过设计研究方法，开发植物启发的机器人原型，分析其感知与表现形式。**

- **链接: [https://arxiv.org/pdf/2601.22387v1](https://arxiv.org/pdf/2601.22387v1)**

> **作者:** Victor Nikhil Antony; Adithya R N; Sarah Derrick; Zhili Gong; Peter M. Donley; Chien-Ming Huang
>
> **摘要:** Plants offer a paradoxical model for interaction: they are ambient, low-demand presences that nonetheless shape atmosphere, routines, and relationships through temporal rhythms and subtle expressions. In contrast, most human-robot interaction (HRI) has been grounded in anthropomorphic and zoomorphic paradigms, producing overt, high-demand forms of engagement. Using a Research through Design (RtD) methodology, we explore plants as metaphoric inspiration for HRI; we conducted iterative cycles of ideation, prototyping, and reflection to investigate what design primitives emerge from plant metaphors and morphologies, and how these primitives can be combined into expressive robotic forms. We present a suite of speculative, open-source prototypes that help probe plant-inspired presence, temporality, form, and gestures. We deepened our learnings from design and prototyping through prototype-centered workshops that explored people's perceptions and imaginaries of plant-inspired robots. This work contributes: (1) Set of plant-inspired robotic artifacts; (2) Designerly insights on how people perceive plant-inspired robots; and (3) Design consideration to inform how to use plant metaphors to reshape HRI.
>
---
#### [new 002] MTDrive: Multi-turn Interactive Reinforcement Learning for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自动驾驶任务，解决轨迹规划中长尾场景的复杂问题。提出MTDrive框架，通过多轮交互强化学习实现轨迹迭代优化，提升安全性和舒适性。**

- **链接: [https://arxiv.org/pdf/2601.22930v1](https://arxiv.org/pdf/2601.22930v1)**

> **作者:** Xidong Li; Mingyu Guo; Chenchao Xu; Bailin Li; Wenjing Zhu; Yangang Zou; Rui Chen; Zehuan Wang
>
> **摘要:** Trajectory planning is a core task in autonomous driving, requiring the prediction of safe and comfortable paths across diverse scenarios. Integrating Multi-modal Large Language Models (MLLMs) with Reinforcement Learning (RL) has shown promise in addressing "long-tail" scenarios. However, existing methods are constrained to single-turn reasoning, limiting their ability to handle complex tasks requiring iterative refinement. To overcome this limitation, we present MTDrive, a multi-turn framework that enables MLLMs to iteratively refine trajectories based on environmental feedback. MTDrive introduces Multi-Turn Group Relative Policy Optimization (mtGRPO), which mitigates reward sparsity by computing relative advantages across turns. We further construct an interactive trajectory understanding dataset from closed-loop simulation to support multi-turn training. Experiments on the NAVSIM benchmark demonstrate superior performance compared to existing methods, validating the effectiveness of our multi-turn reasoning paradigm. Additionally, we implement system-level optimizations to reduce data transfer overhead caused by high-resolution images and multi-turn sequences, achieving 2.5x training throughput. Our data, models, and code will be made available soon.
>
---
#### [new 003] Exo-Plore: Exploring Exoskeleton Control Space through Human-aligned Simulation
- **分类: cs.RO; cs.GR; cs.LG**

- **简介: 该论文属于人机协同控制任务，旨在解决外骨骼辅助优化难题。通过仿真与深度强化学习，无需真实人体实验即可优化髋关节外骨骼控制策略。**

- **链接: [https://arxiv.org/pdf/2601.22550v1](https://arxiv.org/pdf/2601.22550v1)**

> **作者:** Geonho Leem; Jaedong Lee; Jehee Lee; Seungmoon Song; Jungdam Won
>
> **备注:** 10 pages, 9 figures, ICLR 2026 accepted
>
> **摘要:** Exoskeletons show great promise for enhancing mobility, but providing appropriate assistance remains challenging due to the complexity of human adaptation to external forces. Current state-of-the-art approaches for optimizing exoskeleton controllers require extensive human experiments in which participants must walk for hours, creating a paradox: those who could benefit most from exoskeleton assistance, such as individuals with mobility impairments, are rarely able to participate in such demanding procedures. We present Exo-plore, a simulation framework that combines neuromechanical simulation with deep reinforcement learning to optimize hip exoskeleton assistance without requiring real human experiments. Exo-plore can (1) generate realistic gait data that captures human adaptation to assistive forces, (2) produce reliable optimization results despite the stochastic nature of human gait, and (3) generalize to pathological gaits, showing strong linear relationships between pathology severity and optimal assistance.
>
---
#### [new 004] Robust and Generalized Humanoid Motion Tracking
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决 humanoid 机器人在噪声和动态环境中的运动跟踪问题。提出一种动态条件命令聚合框架，提升运动鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.23080v1](https://arxiv.org/pdf/2601.23080v1)**

> **作者:** Yubiao Ma; Han Yu; Jiayin Xie; Changtai Lv; Qiang Luo; Chi Zhang; Yunpeng Yin; Boyang Xing; Xuemei Ren; Dongdong Zheng
>
> **摘要:** Learning a general humanoid whole-body controller is challenging because practical reference motions can exhibit noise and inconsistencies after being transferred to the robot domain, and local defects may be amplified by closed-loop execution, causing drift or failure in highly dynamic and contact-rich behaviors. We propose a dynamics-conditioned command aggregation framework that uses a causal temporal encoder to summarize recent proprioception and a multi-head cross-attention command encoder to selectively aggregate a context window based on the current dynamics. We further integrate a fall recovery curriculum with random unstable initialization and an annealed upward assistance force to improve robustness and disturbance rejection. The resulting policy requires only about 3.5 hours of motion data and supports single-stage end-to-end training without distillation. The proposed method is evaluated under diverse reference inputs and challenging motion regimes, demonstrating zero-shot transfer to unseen motions as well as robust sim-to-real transfer on a physical humanoid robot.
>
---
#### [new 005] FlyAware: Inertia-Aware Aerial Manipulation via Vision-Based Estimation and Post-Grasp Adaptation
- **分类: cs.RO**

- **简介: 该论文属于空中机械臂控制任务，旨在解决动态惯性参数带来的操控难题。通过视觉估计与后抓适应机制，实现稳定抓取与控制。**

- **链接: [https://arxiv.org/pdf/2601.22686v1](https://arxiv.org/pdf/2601.22686v1)**

> **作者:** Biyu Ye; Na Fan; Zhengping Fan; Weiliang Deng; Hongming Chen; Qifeng Chen; Ximin Lyu
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Aerial manipulators (AMs) are gaining increasing attention in automated transportation and emergency services due to their superior dexterity compared to conventional multirotor drones. However, their practical deployment is challenged by the complexity of time-varying inertial parameters, which are highly sensitive to payload variations and manipulator configurations. Inspired by human strategies for interacting with unknown objects, this letter presents a novel onboard framework for robust aerial manipulation. The proposed system integrates a vision-based pre-grasp inertia estimation module with a post-grasp adaptation mechanism, enabling real-time estimation and adaptation of inertial dynamics. For control, we develop an inertia-aware adaptive control strategy based on gain scheduling, and assess its robustness via frequency-domain system identification. Our study provides new insights into post-grasp control for AMs, and real-world experiments validate the effectiveness and feasibility of the proposed framework.
>
---
#### [new 006] Postural Virtual Fixtures for Ergonomic Physical Interactions with Supernumerary Robotic Bodies
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决物理交互中用户姿势不 ergonomic 的问题。通过引入虚拟夹具和实时评估框架，提供阻力反馈以引导正确姿势。**

- **链接: [https://arxiv.org/pdf/2601.22672v1](https://arxiv.org/pdf/2601.22672v1)**

> **作者:** Theodora Kastritsi; Marta Lagomarsino; Arash Ajoudani
>
> **备注:** Published in The International Journal of Robotics Research
>
> **摘要:** Conjoined collaborative robots, functioning as supernumerary robotic bodies (SRBs), can enhance human load tolerance abilities. However, in tasks involving physical interaction with humans, users may still adopt awkward, non-ergonomic postures, which can lead to discomfort or injury over time. In this paper, we propose a novel control framework that provides kinesthetic feedback to SRB users when a non-ergonomic posture is detected, offering resistance to discourage such behaviors. This approach aims to foster long-term learning of ergonomic habits and promote proper posture during physical interactions. To achieve this, a virtual fixture method is developed, integrated with a continuous, online ergonomic posture assessment framework. Additionally, to improve coordination between the operator and the SRB, which consists of a robotic arm mounted on a floating base, the position of the floating base is adjusted as needed. Experimental results demonstrate the functionality and efficacy of the ergonomics-driven control framework, including two user studies involving practical loco-manipulation tasks with 14 subjects, comparing the proposed framework with a baseline control framework that does not account for human ergonomics.
>
---
#### [new 007] IRL-DAL: Safe and Adaptive Trajectory Planning for Autonomous Driving via Energy-Guided Diffusion Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出IRL-DAL框架，用于自动驾驶轨迹规划，解决安全与适应性问题。通过扩散模型和强化学习，提升路径规划的稳定性和安全性。**

- **链接: [https://arxiv.org/pdf/2601.23266v1](https://arxiv.org/pdf/2601.23266v1)**

> **作者:** Seyed Ahmad Hosseini Miangoleh; Amin Jalal Aghdasian; Farzaneh Abdollahi
>
> **摘要:** This paper proposes a novel inverse reinforcement learning framework using a diffusion-based adaptive lookahead planner (IRL-DAL) for autonomous vehicles. Training begins with imitation from an expert finite state machine (FSM) controller to provide a stable initialization. Environment terms are combined with an IRL discriminator signal to align with expert goals. Reinforcement learning (RL) is then performed with a hybrid reward that combines diffuse environmental feedback and targeted IRL rewards. A conditional diffusion model, which acts as a safety supervisor, plans safe paths. It stays in its lane, avoids obstacles, and moves smoothly. Then, a learnable adaptive mask (LAM) improves perception. It shifts visual attention based on vehicle speed and nearby hazards. After FSM-based imitation, the policy is fine-tuned with Proximal Policy Optimization (PPO). Training is run in the Webots simulator with a two-stage curriculum. A 96\% success rate is reached, and collisions are reduced to 0.05 per 1k steps, marking a new benchmark for safe navigation. By applying the proposed approach, the agent not only drives in lane but also handles unsafe conditions at an expert level, increasing robustness.We make our code publicly available.
>
---
#### [new 008] End-to-end Optimization of Belief and Policy Learning in Shared Autonomy Paradigms
- **分类: cs.RO; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于共享自主任务，解决用户意图推理与辅助水平决策问题。提出BRACE框架，实现端到端优化，提升复杂环境下的控制效果。**

- **链接: [https://arxiv.org/pdf/2601.23285v1](https://arxiv.org/pdf/2601.23285v1)**

> **作者:** MH Farhadi; Ali Rabiee; Sima Ghafoori; Anna Cetera; Andrew Fisher; Reza Abiri
>
> **摘要:** Shared autonomy systems require principled methods for inferring user intent and determining appropriate assistance levels. This is a central challenge in human-robot interaction, where systems must be successful while being mindful of user agency. Previous approaches relied on static blending ratios or separated goal inference from assistance arbitration, leading to suboptimal performance in unstructured environments. We introduce BRACE (Bayesian Reinforcement Assistance with Context Encoding), a novel framework that fine-tunes Bayesian intent inference and context-adaptive assistance through an architecture enabling end-to-end gradient flow between intent inference and assistance arbitration. Our pipeline conditions collaborative control policies on environmental context and complete goal probability distributions. We provide analysis showing (1) optimal assistance levels should decrease with goal uncertainty and increase with environmental constraint severity, and (2) integrating belief information into policy learning yields a quadratic expected regret advantage over sequential approaches. We validated our algorithm against SOTA methods (IDA, DQN) using a three-part evaluation progressively isolating distinct challenges of end-effector control: (1) core human-interaction dynamics in a 2D human-in-the-loop cursor task, (2) non-linear dynamics of a robotic arm, and (3) integrated manipulation under goal ambiguity and environmental constraints. We demonstrate improvements over SOTA, achieving 6.3% higher success rates and 41% increased path efficiency, and 36.3% success rate and 87% path efficiency improvement over unassisted control. Our results confirmed that integrated optimization is most beneficial in complex, goal-ambiguous scenarios, and is generalizable across robotic domains requiring goal-directed assistance, advancing the SOTA for adaptive shared autonomy.
>
---
#### [new 009] CARE: Multi-Task Pretraining for Latent Continuous Action Representation in Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出CARE框架，解决机器人控制中依赖动作标注导致的可扩展性问题。通过视频-文本对进行多任务预训练，学习连续动作表示，提升控制效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.22467v1](https://arxiv.org/pdf/2601.22467v1)**

> **作者:** Jiaqi Shi; Xulong Zhang; Xiaoyang Qu; Jianzong Wang
>
> **备注:** Accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models have shown promise for robot control, but their dependence on action supervision limits scalability and generalization. To address this challenge, we introduce CARE, a novel framework designed to train VLA models for robotic task execution. Unlike existing methods that depend on action annotations during pretraining, CARE eliminates the need for explicit action labels by leveraging only video-text pairs. These weakly aligned data sources enable the model to learn continuous latent action representations through a newly designed multi-task pretraining objective. During fine-tuning, a small set of labeled data is used to train the action head for control. Experimental results across various simulation tasks demonstrate CARE's superior success rate, semantic interpretability, and ability to avoid shortcut learning. These results underscore CARE's scalability, interpretability, and effectiveness in robotic control with weak supervision.
>
---
#### [new 010] Robust Rigid Body Assembly via Contact-Implicit Optimal Control with Exact Second-Order Derivatives
- **分类: cs.RO; math.OC**

- **简介: 该论文属于机器人装配任务，解决装配运动规划问题。提出一种基于二阶导数的鲁棒最优控制方法，减少物理仿真次数，提升成功率。**

- **链接: [https://arxiv.org/pdf/2601.22849v1](https://arxiv.org/pdf/2601.22849v1)**

> **作者:** Christian Dietz; Sebastian Albrecht; Gianluca Frison; Moritz Diehl; Armin Nurkanović
>
> **备注:** Submitted to Transactions on Robotics
>
> **摘要:** Efficient planning of assembly motions is a long standing challenge in the field of robotics that has been primarily tackled with reinforcement learning and sampling-based methods by using extensive physics simulations. This paper proposes a sample-efficient robust optimal control approach for the determination of assembly motions, which requires significantly less physics simulation steps during planning through the efficient use of derivative information. To this end, a differentiable physics simulation is constructed that provides second-order analytic derivatives to the numerical solver and allows one to traverse seamlessly from informative derivatives to accurate contact simulation. The solution of the physics simulation problem is made differentiable by using smoothing inspired by interior-point methods applied to both the collision detection as well as the contact resolution problem. We propose a modified variant of an optimization-based formulation of collision detection formulated as a linear program and present an efficient implementation for the nominal evaluation and corresponding first- and second-order derivatives. Moreover, a multi-scenario-based trajectory optimization problem that ensures robustness with respect to sim-to-real mismatches is derived. The capability of the considered formulation is illustrated by results where over 99\% successful executions are achieved in real-world experiments. Thereby, we carefully investigate the effect of smooth approximations of the contact dynamics and robust modeling on the success rates. Furthermore, the method's capability is tested on different peg-in-hole problems in simulation to show the benefit of using exact Hessians over commonly used Hessian approximations.
>
---
#### [new 011] RoboStriker: Hierarchical Decision-Making for Autonomous Humanoid Boxing
- **分类: cs.RO**

- **简介: 该论文提出RoboStriker，解决人形机器人拳击中的自主决策问题。通过分层框架，将策略与执行分离，提升竞技表现与物理可行性。**

- **链接: [https://arxiv.org/pdf/2601.22517v1](https://arxiv.org/pdf/2601.22517v1)**

> **作者:** Kangning Yin; Zhe Cao; Wentao Dong; Weishuai Zeng; Tianyi Zhang; Qiang Zhang; Jingbo Wang; Jiangmiao Pang; Ming Zhou; Weinan Zhang
>
> **摘要:** Achieving human-level competitive intelligence and physical agility in humanoid robots remains a major challenge, particularly in contact-rich and highly dynamic tasks such as boxing. While Multi-Agent Reinforcement Learning (MARL) offers a principled framework for strategic interaction, its direct application to humanoid control is hindered by high-dimensional contact dynamics and the absence of strong physical motion priors. We propose RoboStriker, a hierarchical three-stage framework that enables fully autonomous humanoid boxing by decoupling high-level strategic reasoning from low-level physical execution. The framework first learns a comprehensive repertoire of boxing skills by training a single-agent motion tracker on human motion capture data. These skills are subsequently distilled into a structured latent manifold, regularized by projecting the Gaussian-parameterized distribution onto a unit hypersphere. This topological constraint effectively confines exploration to the subspace of physically plausible motions. In the final stage, we introduce Latent-Space Neural Fictitious Self-Play (LS-NFSP), where competing agents learn competitive tactics by interacting within the latent action space rather than the raw motor space, significantly stabilizing multi-agent training. Experimental results demonstrate that RoboStriker achieves superior competitive performance in simulation and exhibits sim-to-real transfer. Our website is available at RoboStriker.
>
---
#### [new 012] Learning Geometrically-Grounded 3D Visual Representations for View-Generalizable Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决单视角下3D视觉表示不足和泛化能力差的问题。提出MethodName框架，通过单视角3D预训练和多步蒸馏提升操作性能与视图泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.22988v1](https://arxiv.org/pdf/2601.22988v1)**

> **作者:** Di Zhang; Weicheng Duan; Dasen Gu; Hongye Lu; Hai Zhang; Hang Yu; Junqiao Zhao; Guang Chen
>
> **摘要:** Real-world robotic manipulation demands visuomotor policies capable of robust spatial scene understanding and strong generalization across diverse camera viewpoints. While recent advances in 3D-aware visual representations have shown promise, they still suffer from several key limitations, including reliance on multi-view observations during inference which is impractical in single-view restricted scenarios, incomplete scene modeling that fails to capture holistic and fine-grained geometric structures essential for precise manipulation, and lack of effective policy training strategies to retain and exploit the acquired 3D knowledge. To address these challenges, we present MethodName, a unified representation-policy learning framework for view-generalizable robotic manipulation. MethodName introduces a single-view 3D pretraining paradigm that leverages point cloud reconstruction and feed-forward gaussian splatting under multi-view supervision to learn holistic geometric representations. During policy learning, MethodName performs multi-step distillation to preserve the pretrained geometric understanding and effectively transfer it to manipulation skills. We conduct experiments on 12 RLBench tasks, where our approach outperforms the previous state-of-the-art method by 12.7% in average success rate. Further evaluation on six representative tasks demonstrates strong zero-shot view generalization, with success rate drops of only 22.0% and 29.7% under moderate and large viewpoint shifts respectively, whereas the state-of-the-art method suffers larger decreases of 41.6% and 51.5%.
>
---
#### [new 013] Advanced techniques and applications of LiDAR Place Recognition in Agricultural Environments: A Comprehensive Survey
- **分类: cs.RO; cs.AI; cs.ET**

- **简介: 该论文属于农业环境中的LiDAR位姿识别任务，旨在解决农业场景下定位困难的问题。通过综述最新深度学习方法和评估指标，分析现有技术及未来方向。**

- **链接: [https://arxiv.org/pdf/2601.22198v1](https://arxiv.org/pdf/2601.22198v1)**

> **作者:** Judith Vilella-Cantos; Mónica Ballesta; David Valiente; María Flores; Luis Payá
>
> **摘要:** An optimal solution to the localization problem is essential for developing autonomous robotic systems. Apart from autonomous vehicles, precision agriculture is one of the elds that can bene t most from these systems. Although LiDAR place recognition is a widely used technique in recent years to achieve accurate localization, it is mostly used in urban settings. However, the lack of distinctive features and the unstructured nature of agricultural environments make place recognition challenging. This work presents a comprehensive review of state-of-the-art the latest deep learning applications for agricultural environments and LPR techniques. We focus on the challenges that arise in these environments. We analyze the existing approaches, datasets, and metrics used to evaluate LPR system performance and discuss the limitations and future directions of research in this eld. This is the rst survey that focuses on LiDAR based localization in agricultural settings, with the aim of providing a thorough understanding and fostering further research in this specialized domain.
>
---
#### [new 014] Accurate Pedestrian Tracking in Urban Canyons: A Multi-Modal Fusion Approach
- **分类: cs.RO**

- **简介: 论文提出一种多模态融合方法，用于提升城市峡谷中行人定位精度。解决GNSS信号弱环境下定位不准确的问题，结合惯性数据与地图先验，提高盲人等用户导航可靠性。**

- **链接: [https://arxiv.org/pdf/2601.22406v1](https://arxiv.org/pdf/2601.22406v1)**

> **作者:** Shahar Dubiner; Peng Ren; Roberto Manduchi
>
> **摘要:** The contribution describes a pedestrian navigation approach designed to improve localization accuracy in urban environments where GNSS performance is degraded, a problem that is especially critical for blind or low-vision users who depend on precise guidance such as identifying the correct side of a street. To address GNSS limitations and the impracticality of camera-based visual positioning, the work proposes a particle filter based fusion of GNSS and inertial data that incorporates spatial priors from maps, such as impassable buildings and unlikely walking areas, functioning as a probabilistic form of map matching. Inertial localization is provided by the RoNIN machine learning method, and fusion with GNSS is achieved by weighting particles based on their consistency with GNSS estimates and uncertainty. The system was evaluated on six challenging walking routes in downtown San Francisco using three metrics related to sidewalk correctness and localization error. Results show that the fused approach (GNSS+RoNIN+PF) significantly outperforms GNSS only localization on most metrics, while inertial-only localization with particle filtering also surpasses GNSS alone for critical measures such as sidewalk assignment and across street error.
>
---
#### [new 015] Self-Imitated Diffusion Policy for Efficient and Robust Visual Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决扩散策略训练效率低和依赖后处理的问题。提出SIDP框架，通过自模仿机制提升导航效果与效率。**

- **链接: [https://arxiv.org/pdf/2601.22965v1](https://arxiv.org/pdf/2601.22965v1)**

> **作者:** Runhua Zhang; Junyi Hou; Changxu Cheng; Qiyi Chen; Tao Wang; Wuyue Zhao
>
> **备注:** Preprint
>
> **摘要:** Diffusion policies (DP) have demonstrated significant potential in visual navigation by capturing diverse multi-modal trajectory distributions. However, standard imitation learning (IL), which most DP methods rely on for training, often inherits sub-optimality and redundancy from expert demonstrations, thereby necessitating a computationally intensive "generate-then-filter" pipeline that relies on auxiliary selectors during inference. To address these challenges, we propose Self-Imitated Diffusion Policy (SIDP), a novel framework that learns improved planning by selectively imitating a set of trajectories sampled from itself. Specifically, SIDP introduces a reward-guided self-imitation mechanism that encourages the policy to consistently produce high-quality trajectories efficiently, rather than outputs of inconsistent quality, thereby reducing reliance on extensive sampling and post-filtering. During training, we employ a reward-driven curriculum learning paradigm to mitigate inefficient data utility, and goal-agnostic exploration for trajectory augmentation to improve planning robustness. Extensive evaluations on a comprehensive simulation benchmark show that SIDP significantly outperforms previous methods, with real-world experiments confirming its effectiveness across multiple robotic platforms. On Jetson Orin Nano, SIDP delivers a 2.5$\times$ faster inference than the baseline NavDP, i.e., 110ms VS 273ms, enabling efficient real-time deployment.
>
---
#### [new 016] Lantern: A Minimalist Robotic Object Platform
- **分类: cs.RO; cs.HC**

- **简介: 该论文介绍了一种低成本的机器人平台Lantern，用于人机交互研究。旨在降低HRI研究门槛，通过简单形态引发用户互动，验证其在多种场景中的应用潜力。**

- **链接: [https://arxiv.org/pdf/2601.22381v1](https://arxiv.org/pdf/2601.22381v1)**

> **作者:** Victor Nikhil Antony; Zhili Gong; Guanchen Li; Clara Jeon; Chien-Ming Huang
>
> **摘要:** Robotic objects are simple actuated systems that subtly blend into human environments. We design and introduce Lantern, a minimalist robotic object platform to enable building simple robotic artifacts. We conducted in-depth design and engineering iterations of Lantern's mechatronic architecture to meet specific design goals while maintaining a low build cost (~40 USD). As an extendable, open-source platform, Lantern aims to enable exploration of a range of HRI scenarios by leveraging human tendency to assign social meaning to simple forms. To evaluate Lantern's potential for HRI, we conducted a series of explorations: 1) a co-design workshop, 2) a sensory room case study, 3) distribution to external HRI labs, 4) integration into a graduate-level HRI course, and 5) public exhibitions with older adults and children. Our findings show that Lantern effectively evokes engagement, can support versatile applications ranging from emotion regulation to focused work, and serves as a viable platform for lowering barriers to HRI as a field.
>
---
#### [new 017] Temporally Coherent Imitation Learning via Latent Action Flow Matching for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决长时序操作的稳定性与效率问题。提出LG-Flow Policy，通过潜在动作空间的流匹配实现高效且稳定的轨迹模仿学习。**

- **链接: [https://arxiv.org/pdf/2601.23087v1](https://arxiv.org/pdf/2601.23087v1)**

> **作者:** Wu Songwei; Jiang Zhiduo; Xie Guanghu; Liu Yang; Liu Hong
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Learning long-horizon robotic manipulation requires jointly achieving expressive behavior modeling, real-time inference, and stable execution, which remains challenging for existing generative policies. Diffusion-based approaches provide strong modeling capacity but typically incur high inference latency, while flow matching enables fast one-step generation yet often leads to unstable execution when applied directly in the raw action space. We propose LG-Flow Policy, a trajectory-level imitation learning framework that performs flow matching in a continuous latent action space. By encoding action sequences into temporally regularized latent trajectories and learning an explicit latent-space flow, the proposed approach decouples global motion structure from low-level control noise, resulting in smooth and reliable long-horizon execution. LG-Flow Policy further incorporates geometry-aware point cloud conditioning and execution-time multimodal modulation, with visual cues evaluated as a representative modality in real-world settings. Experimental results in simulation and on physical robot platforms demonstrate that LG-Flow Policy achieves near single-step inference, substantially improves trajectory smoothness and task success over flow-based baselines operating in the raw action space, and remains significantly more efficient than diffusion-based policies.
>
---
#### [new 018] Toward Fully Autonomous Driving: AI, Challenges, Opportunities, and Needs
- **分类: cs.RO; cs.ET**

- **简介: 论文探讨AI在自动驾驶中的应用，分析其挑战与机遇，旨在推动完全自动驾驶的发展。任务是评估AI对自动驾驶的影响，解决安全与迁移性问题，提出研究方向。**

- **链接: [https://arxiv.org/pdf/2601.22927v1](https://arxiv.org/pdf/2601.22927v1)**

> **作者:** Lars Ullrich; Michael Buchholz; Klaus Dietmayer; Knut Graichen
>
> **备注:** Published in IEEE Access, 29 January 2026
>
> **摘要:** Automated driving (AD) is promising, but the transition to fully autonomous driving is, among other things, subject to the real, ever-changing open world and the resulting challenges. However, research in the field of AD demonstrates the ability of artificial intelligence (AI) to outperform classical approaches, handle higher complexities, and reach a new level of autonomy. At the same time, the use of AI raises further questions of safety and transferability. To identify the challenges and opportunities arising from AI concerning autonomous driving functionalities, we have analyzed the current state of AD, outlined limitations, and identified foreseeable technological possibilities. Thereby, various further challenges are examined in the context of prospective developments. In this way, this article reconsiders fully autonomous driving with respect to advancements in the field of AI and carves out the respective needs and resulting research questions.
>
---
#### [new 019] ReloPush-BOSS: Optimization-guided Nonmonotone Rearrangement Planning for a Car-like Robot Pusher
- **分类: cs.RO**

- **简介: 该论文属于多物体重新排列任务，解决密集环境下的非单调操作问题。通过优化预移位和路径规划，提升机器人推动物体的效率与可行性。**

- **链接: [https://arxiv.org/pdf/2601.22289v1](https://arxiv.org/pdf/2601.22289v1)**

> **作者:** Jeeho Ahn; Christoforos Mavrogiannis
>
> **备注:** Preprint of final version, accepted to RA-L 2026
>
> **摘要:** We focus on multi-object rearrangement planning in densely cluttered environments using a car-like robot pusher. The combination of kinematic, geometric and physics constraints underlying this domain results in challenging nonmonotone problem instances which demand breaking each manipulation action into multiple parts to achieve a desired object rearrangement. Prior work tackles such instances by planning prerelocations, temporary object displacements that enable constraint satisfaction, but deciding where to prerelocate remains difficult due to local minima leading to infeasible or high-cost paths. Our key insight is that these minima can be avoided by steering a prerelocation optimization toward low-cost regions informed by Dubins path classification. These optimized prerelocations are integrated into an object traversability graph that encodes kinematic, geometric, and pushing constraints. Searching this graph in a depth-first fashion results in efficient, feasible rearrangement sequences. Across a series of densely cluttered scenarios with up to 13 objects, our framework, ReloPush-BOSS, exhibits consistently highest success rates and shortest pushing paths compared to state-of-the-art baselines. Hardware experiments on a 1/10 car-like pusher demonstrate the robustness of our approach. Code and footage from our experiments can be found at: https://fluentrobotics.com/relopushboss.
>
---
#### [new 020] High-Definition 5MP Stereo Vision Sensing for Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉感知任务，旨在解决高分辨率立体视觉系统在机器人中的校准与实时处理问题。通过改进校准和匹配方法，提升点云质量与处理速度。**

- **链接: [https://arxiv.org/pdf/2601.22445v1](https://arxiv.org/pdf/2601.22445v1)**

> **作者:** Leaf Jiang; Matthew Holzel; Bernhard Kaplan; Hsiou-Yuan Liu; Sabyasachi Paul; Karen Rankin; Piotr Swierczynski
>
> **摘要:** High-resolution (5MP+) stereo vision systems are essential for advancing robotic capabilities, enabling operation over longer ranges and generating significantly denser and accurate 3D point clouds. However, realizing the full potential of high-angular-resolution sensors requires a commensurately higher level of calibration accuracy and faster processing -- requirements often unmet by conventional methods. This study addresses that critical gap by processing 5MP camera imagery using a novel, advanced frame-to-frame calibration and stereo matching methodology designed to achieve both high accuracy and speed. Furthermore, we introduce a new approach to evaluate real-time performance by comparing real-time disparity maps with ground-truth disparity maps derived from more computationally intensive stereo matching algorithms. Crucially, the research demonstrates that high-pixel-count cameras yield high-quality point clouds only through the implementation of high-accuracy calibration.
>
---
#### [new 021] Game-Based and Gamified Robotics Education: A Comparative Systematic Review and Design Guidelines
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于教育技术领域，旨在比较游戏化与游戏教学在机器人教育中的效果。通过系统综述分析95项研究，提出设计建议。**

- **链接: [https://arxiv.org/pdf/2601.22199v1](https://arxiv.org/pdf/2601.22199v1)**

> **作者:** Syed T. Mubarrat; Byung-Cheol Min; Tianyu Shao; E. Cho Smith; Bedrich Benes; Alejandra J. Magana; Christos Mousas; Dominic Kao
>
> **备注:** Accepted for publication at Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems. 26 pages, 14 figures, 7 tables;
>
> **摘要:** Robotics education fosters computational thinking, creativity, and problem-solving, but remains challenging due to technical complexity. Game-based learning (GBL) and gamification offer engagement benefits, yet their comparative impact remains unclear. We present the first PRISMA-aligned systematic review and comparative synthesis of GBL and gamification in robotics education, analyzing 95 studies from 12,485 records across four databases (2014-2025). We coded each study's approach, learning context, skill level, modality, pedagogy, and outcomes (k = .918). Three patterns emerged: (1) approach-context-pedagogy coupling (GBL more prevalent in informal settings, while gamification dominated formal classrooms [p < .001] and favored project-based learning [p = .009]); (2) emphasis on introductory programming and modular kits, with limited adoption of advanced software (~17%), advanced hardware (~5%), or immersive technologies (~22%); and (3) short study horizons, relying on self-report. We propose eight research directions and a design space outlining best practices and pitfalls, offering actionable guidance for robotics education.
>
---
#### [new 022] MOSAIC: Modular Scalable Autonomy for Intelligent Coordination of Heterogeneous Robotic Teams
- **分类: cs.RO**

- **简介: 该论文提出MOSAIC框架，解决多机器人协作探索任务中的自主性与操作员干预问题。通过POI抽象和分层自主，实现高效任务分配与持续运行。**

- **链接: [https://arxiv.org/pdf/2601.23038v1](https://arxiv.org/pdf/2601.23038v1)**

> **作者:** David Oberacker; Julia Richer; Philip Arm; Marvin Grosse Besselmann; Lennart Puck; William Talbot; Maximilian Schik; Sabine Bellmann; Tristan Schnell; Hendrik Kolvenbach; Rüdiger Dillmann; Marco Hutter; Arne Roennau
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Mobile robots have become indispensable for exploring hostile environments, such as in space or disaster relief scenarios, but often remain limited to teleoperation by a human operator. This restricts the deployment scale and requires near-continuous low-latency communication between the operator and the robot. We present MOSAIC: a scalable autonomy framework for multi-robot scientific exploration using a unified mission abstraction based on Points of Interest (POIs) and multiple layers of autonomy, enabling supervision by a single operator. The framework dynamically allocates exploration and measurement tasks based on each robot's capabilities, leveraging team-level redundancy and specialization to enable continuous operation. We validated the framework in a space-analog field experiment emulating a lunar prospecting scenario, involving a heterogeneous team of five robots and a single operator. Despite the complete failure of one robot during the mission, the team completed 82.3% of assigned tasks at an Autonomy Ratio of 86%, while the operator workload remained at only 78.2%. These results demonstrate that the proposed framework enables robust, scalable multi-robot scientific exploration with limited operator intervention. We further derive practical lessons learned in robot interoperability, networking architecture, team composition, and operator workload management to inform future multi-robot exploration missions.
>
---
#### [new 023] Adapting Reinforcement Learning for Path Planning in Constrained Parking Scenarios
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于路径规划任务，解决受限环境中实时路径规划问题。提出一种深度强化学习框架，无需理想感知，实现高效、实用的停车路径规划。**

- **链接: [https://arxiv.org/pdf/2601.22545v1](https://arxiv.org/pdf/2601.22545v1)**

> **作者:** Feng Tao; Luca Paparusso; Chenyi Gu; Robin Koehler; Chenxu Wu; Xinyu Huang; Christian Juette; David Paz; Ren Liu
>
> **摘要:** Real-time path planning in constrained environments remains a fundamental challenge for autonomous systems. Traditional classical planners, while effective under perfect perception assumptions, are often sensitive to real-world perception constraints and rely on online search procedures that incur high computational costs. In complex surroundings, this renders real-time deployment prohibitive. To overcome these limitations, we introduce a Deep Reinforcement Learning (DRL) framework for real-time path planning in parking scenarios. In particular, we focus on challenging scenes with tight spaces that require a high number of reversal maneuvers and adjustments. Unlike classical planners, our solution does not require ideal and structured perception, and in principle, could avoid the need for additional modules such as localization and tracking, resulting in a simpler and more practical implementation. Also, at test time, the policy generates actions through a single forward pass at each step, which is lightweight enough for real-time deployment. The task is formulated as a sequential decision-making problem grounded in a bicycle model dynamics, enabling the agent to directly learn navigation policies that respect vehicle kinematics and environmental constraints in the closed-loop setting. A new benchmark is developed to support both training and evaluation, capturing diverse and challenging scenarios. Our approach achieves state-of-the-art success rates and efficiency, surpassing classical planner baselines by +96% in success rate and +52% in efficiency. Furthermore, we release our benchmark as an open-source resource for the community to foster future research in autonomous systems. The benchmark and accompanying tools are available at https://github.com/dqm5rtfg9b-collab/Constrained_Parking_Scenarios.
>
---
#### [new 024] A Comparative Evaluation of Large Vision-Language Models for 2D Object Detection under SOTIF Conditions
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的环境感知任务，旨在解决SOTIF条件下2D目标检测的可靠性问题。通过对比十种大型视觉语言模型与传统YOLO检测器，评估其在复杂场景下的性能表现。**

- **链接: [https://arxiv.org/pdf/2601.22830v1](https://arxiv.org/pdf/2601.22830v1)**

> **作者:** Ji Zhou; Yilin Ding; Yongqi Zhao; Jiachen Xu; Arno Eichberger
>
> **备注:** 6 pages, 11 figures
>
> **摘要:** Reliable environmental perception remains one of the main obstacles for safe operation of automated vehicles. Safety of the Intended Functionality (SOTIF) concerns safety risks from perception insufficiencies, particularly under adverse conditions where conventional detectors often falter. While Large Vision-Language Models (LVLMs) demonstrate promising semantic reasoning, their quantitative effectiveness for safety-critical 2D object detection is underexplored. This paper presents a systematic evaluation of ten representative LVLMs using the PeSOTIF dataset, a benchmark specifically curated for long-tail traffic scenarios and environmental degradations. Performance is quantitatively compared against the classical perception approach, a YOLO-based detector. Experimental results reveal a critical trade-off: top-performing LVLMs (e.g., Gemini 3, Doubao) surpass the YOLO baseline in recall by over 25% in complex natural scenarios, exhibiting superior robustness to visual degradation. Conversely, the baseline retains an advantage in geometric precision for synthetic perturbations. These findings highlight the complementary strengths of semantic reasoning versus geometric regression, supporting the use of LVLMs as high-level safety validators in SOTIF-oriented automated driving systems.
>
---
#### [new 025] Offline Reinforcement Learning of High-Quality Behaviors Under Robust Style Alignment
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于离线强化学习任务，旨在解决风格与任务性能冲突的问题。提出SCIQL框架，结合隐式Q学习和风格约束，提升行为质量与风格一致性。**

- **链接: [https://arxiv.org/pdf/2601.22823v1](https://arxiv.org/pdf/2601.22823v1)**

> **作者:** Mathieu Petitbois; Rémy Portelas; Sylvain Lamprier
>
> **摘要:** We study offline reinforcement learning of style-conditioned policies using explicit style supervision via subtrajectory labeling functions. In this setting, aligning style with high task performance is particularly challenging due to distribution shift and inherent conflicts between style and reward. Existing methods, despite introducing numerous definitions of style, often fail to reconcile these objectives effectively. To address these challenges, we propose a unified definition of behavior style and instantiate it into a practical framework. Building on this, we introduce Style-Conditioned Implicit Q-Learning (SCIQL), which leverages offline goal-conditioned RL techniques, such as hindsight relabeling and value learning, and combine it with a new Gated Advantage Weighted Regression mechanism to efficiently optimize task performance while preserving style alignment. Experiments demonstrate that SCIQL achieves superior performance on both objectives compared to prior offline methods. Code, datasets and visuals are available in: https://sciql-iclr-2026.github.io/.
>
---
#### [new 026] Assistive Robots and Reasonable Work Assignment Reduce Perceived Stigma toward Persons with Disabilities
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于社会心理学研究，旨在解决工作场所中残疾人面临的认知偏见问题。通过实验探讨辅助机器人和合理工作分配对减少偏见的影响。**

- **链接: [https://arxiv.org/pdf/2601.22689v1](https://arxiv.org/pdf/2601.22689v1)**

> **作者:** Stina Klein; Birgit Prodinger; Elisabeth André; Lars Mikelsons; Nils Mandischer
>
> **备注:** 5 pages, 2 figures, Companion Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction
>
> **摘要:** Robots are becoming more prominent in assisting persons with disabilities (PwD). Whilst there is broad consensus that robots can assist in mitigating physical impairments, the extent to which they can facilitate social inclusion remains equivocal. In fact, the exposed status of assisted workers could likewise lead to reduced or increased perceived stigma by other workers. We present a vignette study on the perceived cognitive and behavioral stigma toward PwD in the workplace. We designed four experimental conditions depicting a coworker with an impairment in work scenarios: overburdened work, suitable work, and robot-assisted work only for the coworker, and an offer of robot-assisted work for everyone. Our results show that cognitive stigma is significantly reduced when the work task is adapted to the person's abilities or augmented by an assistive robot. In addition, offering robot-assisted work for everyone, in the sense of universal design, further reduces perceived cognitive stigma. Thus, we conclude that assistive robots reduce perceived cognitive stigma, thereby supporting the use of collaborative robots in work scenarios involving PwDs.
>
---
#### [new 027] FlowCalib: LiDAR-to-Vehicle Miscalibration Detection using Scene Flows
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出FlowCalib，用于检测LiDAR与车辆的角偏移问题。通过分析静态物体的场景流，实现无需额外传感器的校准检测。**

- **链接: [https://arxiv.org/pdf/2601.23107v1](https://arxiv.org/pdf/2601.23107v1)**

> **作者:** Ilir Tahiraj; Peter Wittal; Markus Lienkamp
>
> **摘要:** Accurate sensor-to-vehicle calibration is essential for safe autonomous driving. Angular misalignments of LiDAR sensors can lead to safety-critical issues during autonomous operation. However, current methods primarily focus on correcting sensor-to-sensor errors without considering the miscalibration of individual sensors that cause these errors in the first place. We introduce FlowCalib, the first framework that detects LiDAR-to-vehicle miscalibration using motion cues from the scene flow of static objects. Our approach leverages the systematic bias induced by rotational misalignment in the flow field generated from sequential 3D point clouds, eliminating the need for additional sensors. The architecture integrates a neural scene flow prior for flow estimation and incorporates a dual-branch detection network that fuses learned global flow features with handcrafted geometric descriptors. These combined representations allow the system to perform two complementary binary classification tasks: a global binary decision indicating whether misalignment is present and separate, axis-specific binary decisions indicating whether each rotational axis is misaligned. Experiments on the nuScenes dataset demonstrate FlowCalib's ability to robustly detect miscalibration, establishing a benchmark for sensor-to-vehicle miscalibration detection.
>
---
#### [new 028] PoSafeNet: Safe Learning with Poset-Structured Neural Nets
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于安全强化学习任务，解决多约束下安全执行的问题。通过构建偏序集结构，提出PoSafeNet实现自适应安全控制。**

- **链接: [https://arxiv.org/pdf/2601.22356v1](https://arxiv.org/pdf/2601.22356v1)**

> **作者:** Kiwan Wong; Wei Xiao; Daniela Rus
>
> **摘要:** Safe learning is essential for deploying learningbased controllers in safety-critical robotic systems, yet existing approaches often enforce multiple safety constraints uniformly or via fixed priority orders, leading to infeasibility and brittle behavior. In practice, safety requirements are heterogeneous and admit only partial priority relations, where some constraints are comparable while others are inherently incomparable. We formalize this setting as poset-structured safety, modeling safety constraints as a partially ordered set and treating safety composition as a structural property of the policy class. Building on this formulation, we propose PoSafeNet, a differentiable neural safety layer that enforces safety via sequential closed-form projection under poset-consistent constraint orderings, enabling adaptive selection or mixing of valid safety executions while preserving priority semantics by construction. Experiments on multi-obstacle navigation, constrained robot manipulation, and vision-based autonomous driving demonstrate improved feasibility, robustness, and scalability over unstructured and differentiable quadratic program-based safety layers.
>
---
#### [new 029] Do Open-Vocabulary Detectors Transfer to Aerial Imagery? A Comparative Evaluation
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，研究开放词汇检测器在航空影像中的迁移能力。通过实验评估五种模型，发现其在航空影像中表现不佳，主要受限于语义混淆，提出需改进域适应方法。**

- **链接: [https://arxiv.org/pdf/2601.22164v1](https://arxiv.org/pdf/2601.22164v1)**

> **作者:** Christos Tsourveloudis
>
> **摘要:** Open-vocabulary object detection (OVD) enables zero-shot recognition of novel categories through vision-language models, achieving strong performance on natural images. However, transferability to aerial imagery remains unexplored. We present the first systematic benchmark evaluating five state-of-the-art OVD models on the LAE-80C aerial dataset (3,592 images, 80 categories) under strict zero-shot conditions. Our experimental protocol isolates semantic confusion from visual localization through Global, Oracle, and Single-Category inference modes. Results reveal severe domain transfer failure: the best model (OWLv2) achieves only 27.6% F1-score with 69% false positive rate. Critically, reducing vocabulary size from 80 to 3.2 classes yields 15x improvement, demonstrating that semantic confusion is the primary bottleneck. Prompt engineering strategies such as domain-specific prefixing and synonym expansion, fail to provide meaningful performance gains. Performance varies dramatically across datasets (F1: 0.53 on DIOR, 0.12 on FAIR1M), exposing brittleness to imaging conditions. These findings establish baseline expectations and highlight the need for domain-adaptive approaches in aerial OVD.
>
---
#### [new 030] Aligning Microscopic Vehicle and Macroscopic Traffic Statistics: Reconstructing Driving Behavior from Partial Data
- **分类: cs.MA; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决从部分数据中重建驾驶行为的问题。通过融合微观与宏观交通数据，学习符合交通统计的驾驶策略。**

- **链接: [https://arxiv.org/pdf/2601.22242v1](https://arxiv.org/pdf/2601.22242v1)**

> **作者:** Zhihao Zhang; Keith Redmill; Chengyang Peng; Bowen Weng
>
> **摘要:** A driving algorithm that aligns with good human driving practices, or at the very least collaborates effectively with human drivers, is crucial for developing safe and efficient autonomous vehicles. In practice, two main approaches are commonly adopted: (i) supervised or imitation learning, which requires comprehensive naturalistic driving data capturing all states that influence a vehicle's decisions and corresponding actions, and (ii) reinforcement learning (RL), where the simulated driving environment either matches or is intentionally more challenging than real-world conditions. Both methods depend on high-quality observations of real-world driving behavior, which are often difficult and costly to obtain. State-of-the-art sensors on individual vehicles can gather microscopic data, but they lack context about the surrounding conditions. Conversely, roadside sensors can capture traffic flow and other macroscopic characteristics, but they cannot associate this information with individual vehicles on a microscopic level. Motivated by this complementarity, we propose a framework that reconstructs unobserved microscopic states from macroscopic observations, using microscopic data to anchor observed vehicle behaviors, and learns a shared policy whose behavior is microscopically consistent with the partially observed trajectories and actions and macroscopically aligned with target traffic statistics when deployed population-wide. Such constrained and regularized policies promote realistic flow patterns and safe coordination with human drivers at scale.
>
---
#### [new 031] RN-D: Discretized Categorical Actors with Regularized Networks for On-Policy Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，旨在解决连续控制中的策略优化问题。通过引入离散化分类策略和正则化网络，提升策略更新的稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2601.23075v1](https://arxiv.org/pdf/2601.23075v1)**

> **作者:** Yuexin Bian; Jie Feng; Tao Wang; Yijiang Li; Sicun Gao; Yuanyuan Shi
>
> **摘要:** On-policy deep reinforcement learning remains a dominant paradigm for continuous control, yet standard implementations rely on Gaussian actors and relatively shallow MLP policies, often leading to brittle optimization when gradients are noisy and policy updates must be conservative. In this paper, we revisit policy representation as a first-class design choice for on-policy optimization. We study discretized categorical actors that represent each action dimension with a distribution over bins, yielding a policy objective that resembles a cross-entropy loss. Building on architectural advances from supervised learning, we further propose regularized actor networks, while keeping critic design fixed. Our results show that simply replacing the standard actor network with our discretized regularized actor yields consistent gains and achieve the state-of-the-art performance across diverse continuous-control benchmarks.
>
---
#### [new 032] About an Automating Annotation Method for Robot Markers
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决机器人标记识别中手动标注耗时且易错的问题。通过自动标注ArUco标记图像，提升深度学习模型的识别性能。**

- **链接: [https://arxiv.org/pdf/2601.22982v1](https://arxiv.org/pdf/2601.22982v1)**

> **作者:** Wataru Uemura; Takeru Nagashima
>
> **摘要:** Factory automation has become increasingly important due to labor shortages, leading to the introduction of autonomous mobile robots for tasks such as material transportation. Markers are commonly used for robot self-localization and object identification. In the RoboCup Logistics League (RCLL), ArUco markers are employed both for robot localization and for identifying processing modules. Conventional recognition relies on OpenCV-based image processing, which detects black-and-white marker patterns. However, these methods often fail under noise, motion blur, defocus, or varying illumination conditions. Deep-learning-based recognition offers improved robustness under such conditions, but requires large amounts of annotated data. Annotation must typically be done manually, as the type and position of objects cannot be detected automatically, making dataset preparation a major bottleneck. In contrast, ArUco markers include built-in recognition modules that provide both ID and positional information, enabling automatic annotation. This paper proposes an automated annotation method for training deep-learning models on ArUco marker images. By leveraging marker detection results obtained from the ArUco module, the proposed approach eliminates the need for manual labeling. A YOLO-based model is trained using the automatically annotated dataset, and its performance is evaluated under various conditions. Experimental results demonstrate that the proposed method improves recognition performance compared with conventional image-processing techniques, particularly for images affected by blur or defocus. Automatic annotation also reduces human effort and ensures consistent labeling quality. Future work will investigate the relationship between confidence thresholds and recognition performance.
>
---
## 更新

#### [replaced 001] A Practical Framework of Key Performance Indicators for Multi-Robot Lunar and Planetary Field Tests
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人行星探测任务，旨在解决不同实验间性能评估不一致的问题。提出了一套基于场景的KPI框架，以实现高效、稳健和精确的评估。**

- **链接: [https://arxiv.org/pdf/2601.20529v2](https://arxiv.org/pdf/2601.20529v2)**

> **作者:** Julia Richter; David Oberacker; Gabriela Ligeza; Valentin T. Bickel; Philip Arm; William Talbot; Marvin Grosse Besselmann; Florian Kehl; Tristan Schnell; Hendrik Kolvenbach; Rüdiger Dillmann; Arne Roennau; Marco Hutter
>
> **摘要:** Robotic prospecting for critical resources on the Moon, such as ilmenite, rare earth elements, and water ice, requires robust exploration methods given the diverse terrain and harsh environmental conditions. Although numerous analog field trials address these goals, comparing their results remains challenging because of differences in robot platforms and experimental setups. These missions typically assess performance using selected, scenario-specific engineering metrics that fail to establish a clear link between field performance and science-driven objectives. In this paper, we address this gap by deriving a structured framework of KPI from three realistic multi-robot lunar scenarios reflecting scientific objectives and operational constraints. Our framework emphasizes scenario-dependent priorities in efficiency, robustness, and precision, and is explicitly designed for practical applicability in field deployments. We validated the framework in a multi-robot field test and found it practical and easy to apply for efficiency- and robustness-related KPI, whereas precision-oriented KPI require reliable ground-truth data that is not always feasible to obtain in outdoor analog environments. Overall, we propose this framework as a common evaluation standard enabling consistent, goal-oriented comparison of multi-robot field trials and supporting systematic development of robotic systems for future planetary exploration.
>
---
#### [replaced 002] SuperPoint-SLAM3: Augmenting ORB-SLAM3 with Deep Features, Adaptive NMS, and Learning-Based Loop Closure
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在提升ORB-SLAM3在极端条件下的精度。通过引入深度特征、自适应NMS和学习式回环检测，显著降低了定位误差。**

- **链接: [https://arxiv.org/pdf/2506.13089v2](https://arxiv.org/pdf/2506.13089v2)**

> **作者:** Shahram Najam Syed; Ishir Roongta; Kavin Ravie; Gangadhar Nageswar
>
> **备注:** 10 pages, 6 figures, code at https://github.com/shahram95/SuperPointSLAM3
>
> **摘要:** Visual simultaneous localization and mapping (SLAM) must remain accurate under extreme viewpoint, scale and illumination variations. The widely adopted ORB-SLAM3 falters in these regimes because it relies on hand-crafted ORB keypoints. We introduce SuperPoint-SLAM3, a drop-in upgrade that (i) replaces ORB with the self-supervised SuperPoint detector--descriptor, (ii) enforces spatially uniform keypoints via adaptive non-maximal suppression (ANMS), and (iii) integrates a lightweight NetVLAD place-recognition head for learning-based loop closure. On the KITTI Odometry benchmark SuperPoint-SLAM3 reduces mean translational error from 4.15% to 0.34% and mean rotational error from 0.0027 deg/m to 0.0010 deg/m. On the EuRoC MAV dataset it roughly halves both errors across every sequence (e.g., V2\_03: 1.58% -> 0.79%). These gains confirm that fusing modern deep features with a learned loop-closure module markedly improves ORB-SLAM3 accuracy while preserving its real-time operation. Implementation, pretrained weights and reproducibility scripts are available at https://github.com/shahram95/SuperPointSLAM3.
>
---
#### [replaced 003] VAT: Vision Action Transformer by Unlocking Full Representation of ViT
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VAT模型，解决机器人学习中视觉感知与动作生成融合不足的问题。通过利用ViT全层次特征，提升模仿学习效果，在模拟任务中取得优异成绩。**

- **链接: [https://arxiv.org/pdf/2512.06013v2](https://arxiv.org/pdf/2512.06013v2)**

> **作者:** Wenhao Li; Chengwei Ma; Weixin Mao
>
> **摘要:** In robot learning, Vision Transformers (ViTs) are standard for visual perception, yet most methods discard valuable information by using only the final layer's features. We argue this provides an insufficient representation and propose the Vision Action Transformer (VAT), a novel architecture that is extended from ViT and unlocks the full feature hierarchy of ViT. VAT processes specialized action tokens with visual features across all transformer layers, enabling a deep and progressive fusion of perception and action generation. On a suite of simulated manipulation tasks, VAT achieves a 98.15\% average success rate across four LIBERO benchmarks, establishing a new state-of-the-art by outperforming prior methods like OpenVLA-OFT. Our work presents not only a powerful model for imitation learning but also demonstrates the critical importance of leveraging the complete ''representation trajectory'' of vision models to advance robotic policy. The GitHub URL for the project code is https://github.com/sellerbubble/VAT.
>
---
#### [replaced 004] ReGlove: A Soft Pneumatic Glove for Activities of Daily Living Assistance via Wrist-Mounted Vision
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ReGlove，一种基于视觉的低成本辅助手套系统，用于上肢功能障碍患者的日常活动帮助。任务是解决传统设备成本高或依赖不可靠生物信号的问题，通过集成摄像头和边缘计算实现精准抓握控制。**

- **链接: [https://arxiv.org/pdf/2512.11824v2](https://arxiv.org/pdf/2512.11824v2)**

> **作者:** Rosh Ho; Jian Zhang
>
> **摘要:** This paper presents ReGlove, a system that converts low-cost commercial pneumatic rehabilitation gloves into vision-guided assistive orthoses. Chronic upper-limb impairment affects millions worldwide, yet existing assistive technologies remain prohibitively expensive or rely on unreliable biological signals. Our platform integrates a wrist-mounted camera with an edge-computing inference engine (Raspberry Pi 5) to enable context-aware grasping without requiring reliable muscle signals. By adapting real-time YOLO-based computer vision models, the system achieves 96.73% grasp classification accuracy with sub-40.00 millisecond end-to-end latency. Physical validation using standardized benchmarks shows 82.71% success on YCB object manipulation and reliable performance across 27 Activities of Daily Living (ADL) tasks. With a total cost under $250 and exclusively commercial components, ReGlove provides a technical foundation for accessible, vision-based upper-limb assistance that could benefit populations excluded from traditional EMG-controlled devices.
>
---
#### [replaced 005] Monocular pose estimation of articulated open surgery tools -- in the wild
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 论文提出一种单目6D姿态估计框架，用于开放手术中可动手术工具的定位，解决对象关节、反光、遮挡及域适应问题。**

- **链接: [https://arxiv.org/pdf/2407.12138v3](https://arxiv.org/pdf/2407.12138v3)**

> **作者:** Robert Spektor; Tom Friedman; Itay Or; Gil Bolotin; Shlomi Laufer
>
> **备注:** Author Accepted Manuscript (AAM)
>
> **摘要:** This work presents a framework for monocular 6D pose estimation of surgical instruments in open surgery, addressing challenges such as object articulations, specularity, occlusions, and synthetic-to-real domain adaptation. The proposed approach consists of three main components: $(1)$ synthetic data generation pipeline that incorporates 3D scanning of surgical tools with articulation rigging and physically-based rendering; $(2)$ a tailored pose estimation framework combining tool detection with pose and articulation estimation; and $(3)$ a training strategy on synthetic and real unannotated video data, employing domain adaptation with automatically generated pseudo-labels. Evaluations conducted on real data of open surgery demonstrate the good performance and real-world applicability of the proposed framework, highlighting its potential for integration into medical augmented reality and robotic systems. The approach eliminates the need for extensive manual annotation of real surgical data.
>
---
#### [replaced 006] Reinforcement Learning for Ballbot Navigation in Uneven Terrain
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人导航任务，旨在解决Ballbot在不平整地形中的导航问题。通过强化学习方法，结合观测数据和奖励设计，实现有效导航。**

- **链接: [https://arxiv.org/pdf/2505.18417v2](https://arxiv.org/pdf/2505.18417v2)**

> **作者:** Achkan Salehi
>
> **备注:** 6 pages, 9 figures, 2 tables. Version two corrects figure 4 and adds some experiments
>
> **摘要:** Ballbot (i.e. Ball balancing robot) navigation usually relies on methods rooted in control theory (CT), and works that apply Reinforcement learning (RL) to the problem remain rare while generally being limited to specific subtasks (e.g. balance recovery). Unlike CT based methods, RL does not require (simplifying) assumptions about environment dynamics (e.g. the absence of slippage between the ball and the floor). In addition to this increased accuracy in modeling, RL agents can easily be conditioned on additional observations such as depth-maps without the need for explicit formulations from first principles, leading to increased adaptivity. Despite those advantages, there has been little to no investigation into the capabilities, data-efficiency and limitations of RL based methods for ballbot control and navigation. Furthermore, there is a notable absence of an open-source, RL-friendly simulator for this task. In this paper, we present an open-source ballbot simulation based on MuJoCo, and show that with appropriate conditioning on exteroceptive observations as well as reward shaping, policies learned by classical model-free RL methods are capable of effectively navigating through randomly generated uneven terrain, using a reasonable amount of data (four to five hours on a system operating at 500hz). Our code is made publicly available.
>
---
#### [replaced 007] Multi-agent Coordination via Flow Matching
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，解决协调与计算效率的平衡问题。提出MAC-Flow框架，通过流模型学习联合行为表示，并转化为高效策略。**

- **链接: [https://arxiv.org/pdf/2511.05005v2](https://arxiv.org/pdf/2511.05005v2)**

> **作者:** Dongsu Lee; Daehee Lee; Amy Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** This work presents MAC-Flow, a simple yet expressive framework for multi-agent coordination. We argue that requirements of effective coordination are twofold: (i) a rich representation of the diverse joint behaviors present in offline data and (ii) the ability to act efficiently in real time. However, prior approaches often sacrifice one for the other, i.e., denoising diffusion-based solutions capture complex coordination but are computationally slow, while Gaussian policy-based solutions are fast but brittle in handling multi-agent interaction. MAC-Flow addresses this trade-off by first learning a flow-based representation of joint behaviors, and then distilling it into decentralized one-step policies that preserve coordination while enabling fast execution. Across four different benchmarks, including $12$ environments and $34$ datasets, MAC-Flow alleviates the trade-off between performance and computational cost, specifically achieving about $\boldsymbol{\times14.5}$ faster inference compared to diffusion-based MARL methods, while maintaining good performance. At the same time, its inference speed is similar to that of prior Gaussian policy-based offline multi-agent reinforcement learning (MARL) methods.
>
---
#### [replaced 008] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MemoryVLA，解决机器人操作中的长期时序依赖问题。通过结合感知-认知记忆机制，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2508.19236v2](https://arxiv.org/pdf/2508.19236v2)**

> **作者:** Hao Shi; Bin Xie; Yingfei Liu; Lin Sun; Fengrong Liu; Tiancai Wang; Erjin Zhou; Haoqiang Fan; Xiangyu Zhang; Gao Huang
>
> **备注:** ICLR 2026 | The project is available at https://shihao1895.github.io/MemoryVLA
>
> **摘要:** Temporal context is essential for robotic manipulation because such tasks are inherently non-Markovian, yet mainstream VLA models typically overlook it and struggle with long-horizon, temporally dependent tasks. Cognitive science suggests that humans rely on working memory to buffer short-lived representations for immediate control, while the hippocampal system preserves verbatim episodic details and semantic gist of past experience for long-term memory. Inspired by these mechanisms, we propose MemoryVLA, a Cognition-Memory-Action framework for long-horizon robotic manipulation. A pretrained VLM encodes the observation into perceptual and cognitive tokens that form working memory, while a Perceptual-Cognitive Memory Bank stores low-level details and high-level semantics consolidated from it. Working memory retrieves decision-relevant entries from the bank, adaptively fuses them with current tokens, and updates the bank by merging redundancies. Using these tokens, a memory-conditioned diffusion action expert yields temporally aware action sequences. We evaluate MemoryVLA on 150+ simulation and real-world tasks across three robots. On SimplerEnv-Bridge, Fractal, LIBERO-5 suites and Mikasa-Robo, it achieves 71.9%, 72.7%, 96.5%, and 41.2% success rates, respectively, all outperforming state-of-the-art baselines CogACT and pi-0, with a notable +14.6 gain on Bridge and +11.8 gain on Mikasa-Robo. On 12 real-world tasks spanning general skills and long-horizon temporal dependencies, MemoryVLA achieves 84.0% success rate, with long-horizon tasks showing a +26 improvement over state-of-the-art baseline. Project Page: https://shihao1895.github.io/MemoryVLA
>
---
#### [replaced 009] TaF-VLA: Tactile-Force Alignment in Vision-Language-Action Models for Force-aware Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉语言动作模型在需要精确力控制的接触任务中缺乏物理直觉的问题。通过引入触觉-力对齐机制，提升模型的力感知与物理推理能力。**

- **链接: [https://arxiv.org/pdf/2601.20321v2](https://arxiv.org/pdf/2601.20321v2)**

> **作者:** Yuzhe Huang; Pei Lin; Wanlin Li; Daohan Li; Jiajun Li; Jiaming Jiang; Chenxi Xiao; Ziyuan Jiao
>
> **备注:** 17pages,9fig
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as powerful generalists for robotic manipulation. However, due to their predominant reliance on visual modalities, they fundamentally lack the physical intuition required for contact-rich tasks that require precise force regulation and physical reasoning. Existing attempts to incorporate vision-based tactile sensing into VLA models typically treat tactile inputs as auxiliary visual textures, thereby overlooking the underlying correlation between surface deformation and interaction dynamics. To bridge this gap, we propose a paradigm shift from tactile-vision alignment to tactile-force alignment. Here, we introduce TaF-VLA, a framework that explicitly grounds high-dimensional tactile observations in physical interaction forces. To facilitate this, we develop an automated tactile-force data acquisition device and curate the TaF-Dataset, comprising over 10 million synchronized tactile observations, 6-axis force/torque, and matrix force map. To align sequential tactile observations with interaction forces, the central component of our approach is the Tactile-Force Adapter (TaF-Adapter), a tactile sensor encoder that extracts discretized latent information for encoding tactile observations. This mechanism ensures that the learned representations capture history-dependent, noise-insensitive physical dynamics rather than static visual textures. Finally, we integrate this force-aligned encoder into a VLA backbone. Extensive real-world experiments demonstrate that TaF-VLA policy significantly outperforms state-of-the-art tactile-vision-aligned and vision-only baselines on contact-rich tasks, verifying its ability to achieve robust, force-aware manipulation through cross-modal physical reasoning.
>
---
#### [replaced 010] CaLiV: LiDAR-to-Vehicle Calibration of Arbitrary Sensor Setups
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于传感器标定任务，解决多LiDAR系统在非重叠视场下的标定问题，提出CaLiV算法实现高精度传感器间及传感器到车辆的标定。**

- **链接: [https://arxiv.org/pdf/2504.01987v3](https://arxiv.org/pdf/2504.01987v3)**

> **作者:** Ilir Tahiraj; Markus Edinger; Dominik Kulmer; Markus Lienkamp
>
> **摘要:** In autonomous systems, sensor calibration is essential for safe and efficient navigation in dynamic environments. Accurate calibration is a prerequisite for reliable perception and planning tasks such as object detection and obstacle avoidance. Many existing LiDAR calibration methods require overlapping fields of view, while others use external sensing devices or postulate a feature-rich environment. In addition, Sensor-to-Vehicle calibration is not supported by the vast majority of calibration algorithms. In this work, we propose a novel target-based technique for extrinsic Sensor-to-Sensor and Sensor-to-Vehicle calibration of multi-LiDAR systems called CaLiV. This algorithm works for non-overlapping fields of view and does not require any external sensing devices. First, we apply motion to produce field of view overlaps and utilize a simple Unscented Kalman Filter to obtain vehicle poses. Then, we use the Gaussian mixture model-based registration framework GMMCalib to align the point clouds in a common calibration frame. Finally, we reduce the task of recovering the sensor extrinsics to a minimization problem. We show that both translational and rotational Sensor-to-Sensor errors can be solved accurately by our method. In addition, all Sensor-to-Vehicle rotation angles can also be calibrated with high accuracy. We validate the simulation results in real-world experiments. The code is open-source and available on https://github.com/TUMFTM/CaLiV.
>
---
#### [replaced 011] Model-Based Diffusion Sampling for Predictive Control in Offline Decision Making
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于离线决策任务，解决扩散模型生成轨迹与系统动力学不匹配的问题。提出MPDiffuser框架，结合规划器与动力学模型，生成符合任务目标的动态合理轨迹。**

- **链接: [https://arxiv.org/pdf/2512.08280v2](https://arxiv.org/pdf/2512.08280v2)**

> **作者:** Haldun Balim; Na Li; Yilun Du
>
> **摘要:** Offline decision-making via diffusion models often produces trajectories that are misaligned with system dynamics, limiting their reliability for control. We propose Model Predictive Diffuser (MPDiffuser), a compositional diffusion framework that combines a diffusion planner with a dynamics diffusion model to generate task-aligned and dynamically plausible trajectories. MPDiffuser interleaves planner and dynamics updates during sampling, progressively correcting feasibility while preserving task intent. A lightweight ranking module then selects trajectories that best satisfy task objectives. The compositional design improves sample efficiency and adaptability by enabling the dynamics model to leverage diverse and previously unseen data independently of the planner. Empirically, we demonstrate consistent improvements over prior diffusion-based methods on unconstrained (D4RL) and constrained (DSRL) benchmarks, and validate practicality through deployment on a real quadrupedal robot.
>
---
#### [replaced 012] PocketDP3: Efficient Pocket-Scale 3D Visuomotor Policy
- **分类: cs.RO**

- **简介: 该论文提出PocketDP3，解决3D视觉-动作策略的参数效率问题。通过轻量级架构提升推理速度与部署实用性，适用于机器人操作任务。**

- **链接: [https://arxiv.org/pdf/2601.22018v2](https://arxiv.org/pdf/2601.22018v2)**

> **作者:** Jinhao Zhang; Zhexuan Zhou; Huizhe Li; Yichen Lai; Wenlong Xia; Haoming Song; Youmin Gong; Jie Mei
>
> **摘要:** Recently, 3D vision-based diffusion policies have shown strong capability in learning complex robotic manipulation skills. However, a common architectural mismatch exists in these models: a tiny yet efficient point-cloud encoder is often paired with a massive decoder. Given a compact scene representation, we argue that this may lead to substantial parameter waste in the decoder. Motivated by this observation, we propose PocketDP3, a pocket-scale 3D diffusion policy that replaces the heavy conditional U-Net decoder used in prior methods with a lightweight Diffusion Mixer (DiM) built on MLP-Mixer blocks. This architecture enables efficient fusion across temporal and channel dimensions, significantly reducing model size. Notably, without any additional consistency distillation techniques, our method supports two-step inference without sacrificing performance, improving practicality for real-time deployment. Across three simulation benchmarks--RoboTwin2.0, Adroit, and MetaWorld--PocketDP3 achieves state-of-the-art performance with fewer than 1% of the parameters of prior methods, while also accelerating inference. Real-world experiments further demonstrate the practicality and transferability of our method in real-world settings. Code will be released.
>
---
#### [replaced 013] Emergent morphogenesis via planar fabrication enabled by a reduced model of composites
- **分类: cs.GR; cs.RO**

- **简介: 该论文属于结构设计任务，旨在解决从平面材料高效生成三维形态的问题。通过简化多层复合结构为单层模型，实现3D形状的可编程控制与制造。**

- **链接: [https://arxiv.org/pdf/2508.08198v2](https://arxiv.org/pdf/2508.08198v2)**

> **作者:** Yupeng Zhang; Adam Alon; M. Khalid Jawed
>
> **备注:** GitHub repository: https://github.com/StructuresComp/discrete-shells-shrinky-dink/
>
> **摘要:** The ability to engineer complex three-dimensional shapes from planar sheets with precise, programmable control underpins emerging technologies in soft robotics, reconfigurable devices, and functional materials. Here, we present a reduced-order numerical and experimental framework for a bilayer system consisting of a stimuli-responsive thermoplastic sheet (Shrinky Dink) bonded to a kirigami-patterned, inert plastic layer. Upon uniform heating, the active layer contracts while the patterned layer constrains in-plane stretch but allows out-of-plane bending, yielding programmable 3D morphologies from simple planar precursors. Our approach enables efficient computational design and scalable manufacturing of 3D forms with a single-layer reduced model that captures the coupled mechanics of stretching and bending. Unlike traditional bilayer modeling, our framework collapses the multilayer composite into a single layer of nodes and elements, reducing the degrees of freedom and enabling simulation on a 2D geometry. This is achieved by introducing a novel energy formulation that captures the coupling between in-plane stretch mismatch and out-of-plane bending - extending beyond simple isotropic linear elastic models. Experimentally, we establish a fully planar, repeatable fabrication protocol using a stimuli-responsive thermoplastic and a laser-cut inert plastic layer. The programmed strain mismatch drives an array of 3D morphologies, such as bowls, canoes, and flower petals, all verified by both simulation and physical prototypes.
>
---
#### [replaced 014] RoboArmGS: High-Quality Robotic Arm Splatting via Bézier Curve Refinement
- **分类: cs.RO**

- **简介: 该论文属于机器人数字资产构建任务，旨在解决URDF运动与真实运动不匹配导致的渲染问题。提出RoboArmGS，通过Bézier曲线优化运动模型，提升3D高斯的精度和一致性。**

- **链接: [https://arxiv.org/pdf/2511.17961v2](https://arxiv.org/pdf/2511.17961v2)**

> **作者:** Hao Wang; Xiaobao Wei; Ying Li; Qingpo Wuwu; Dongli Wu; Jiajun Cao; Ming Lu; Wenzhao Zheng; Shanghang Zhang
>
> **摘要:** Constructing photorealistic and controllable robotic arm digital assets from real observations is fundamental to robotic applications. Current approaches naively bind static 3D Gaussians according to URDF links, forcing them to follow an URDF-rigged motion passively. However, the idealized URDF-rigged motion cannot accurately model the actual motion captured in real-world observations, leading to severe rendering artifacts in 3D Gaussians. To address these challenges, we propose RoboArmGS, a novel hybrid representation that refines the URDF-rigged motion with learnable Bézier curves, enabling more accurate real-world motion modeling. To be more specific, we present a learnable Bézier Curve motion refiner that corrects per-joint residuals to address mismatches between real-world motion and URDF-rigged motion. RoboArmGS enables the learning of more accurate real-world motion while achieving a coherent binding of 3D Gaussians across arm parts. To support future research, we contribute a carefully collected dataset named RoboArm4D, which comprises several widely used robotic arms for evaluating the quality of building high-quality digital assets. We evaluate our approach on RoboArm4D, and RoboArmGS achieves state-of-the-art performance in real-world motion modeling and rendering quality. The code and dataset will be released.
>
---
#### [replaced 015] On Your Own: Pro-level Autonomous Drone Racing in Uninstrumented Arenas
- **分类: cs.RO**

- **简介: 该论文属于自主无人机竞速任务，旨在提升无人机在非结构化环境中的自主导航能力。工作包括在受控与未装备环境中的性能验证，证明系统可媲美人类飞行员。**

- **链接: [https://arxiv.org/pdf/2510.13644v2](https://arxiv.org/pdf/2510.13644v2)**

> **作者:** Michael Bosello; Flavio Pinzarrone; Sara Kiade; Davide Aguiari; Yvo Keuter; Aaesha AlShehhi; Gyordan Caminati; Kei Long Wong; Ka Seng Chou; Junaid Halepota; Fares Alneyadi; Jacopo Panerati; Giovanni Pau
>
> **摘要:** Drone technology is proliferating in many industries, including agriculture, logistics, defense, infrastructure, and environmental monitoring. Vision-based autonomy is one of its key enablers, particularly for real-world applications. This is essential for operating in novel, unstructured environments where traditional navigation methods may be unavailable. Autonomous drone racing has become the de facto benchmark for such systems. State-of-the-art research has shown that autonomous systems can surpass human-level performance in racing arenas. However, the direct applicability to commercial and field operations is still limited, as current systems are often trained and evaluated in highly controlled environments. In our contribution, the system's capabilities are analyzed within a controlled environment -- where external tracking is available for ground-truth comparison -- but also demonstrated in a challenging, uninstrumented environment -- where ground-truth measurements were never available. We show that our approach can match the performance of professional human pilots in both scenarios.
>
---
#### [replaced 016] EquiContact: A Hierarchical SE(3) Vision-to-Force Equivariant Policy for Spatially Generalizable Contact-rich Tasks
- **分类: cs.RO**

- **简介: 该论文研究机器人在接触丰富的操作任务中实现空间泛化的问题。提出EquiContact框架，结合视觉规划与顺应控制，提升 peg-in-hole 等任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.10961v4](https://arxiv.org/pdf/2507.10961v4)**

> **作者:** Joohwan Seo; Arvind Kruthiventy; Soomi Lee; Megan Teng; Seoyeon Choi; Xiang Zhang; Jongeun Choi; Roberto Horowitz
>
> **备注:** Submitted to RSS
>
> **摘要:** This paper presents a framework for learning vision-based robotic policies for contact-rich manipulation tasks that generalize spatially across task configurations. We focus on achieving robust spatial generalization of the policy for the peg-in-hole (PiH) task trained from a small number of demonstrations. We propose EquiContact, a hierarchical policy composed of a high-level vision planner (Diffusion Equivariant Descriptor Field, Diff-EDF) and a novel low-level compliant visuomotor policy (Geometric Compliant ACT, G-CompACT). G-CompACT operates using only localized observations (geometrically consistent error vectors (GCEV), force-torque readings, and wrist-mounted RGB images) and produces actions defined in the end-effector frame. Through these design choices, we show that the entire EquiContact pipeline is SE(3)-equivariant, from perception to force control. We also outline three key components for spatially generalizable contact-rich policies: compliance, localized policies, and induced equivariance. Real-world experiments on PiH, screwing, and surface wiping tasks demonstrate a near-perfect success rate and robust generalization to unseen spatial configurations, validating the proposed framework and principles. The experimental videos and more details can be found on the project website: https://equicontact.github.io/EquiContact-website/
>
---
#### [replaced 017] GMOR: A Lightweight Robust Point Cloud Registration Framework via Geometric Maximum Overlapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云配准任务，解决高噪声下配准精度与效率问题。提出GMOR框架，通过旋转BnB搜索和几何重叠优化，实现高效准确的点云配准。**

- **链接: [https://arxiv.org/pdf/2508.17427v2](https://arxiv.org/pdf/2508.17427v2)**

> **作者:** Zhao Zheng; Jingfan Fan; Long Shao; Hong Song; Danni Ai; Tianyu Fu; Deqiang Xiao; Yongtian Wang; Jian Yang
>
> **摘要:** Point cloud registration based on correspondences computes the rigid transformation that maximizes the number of inliers constrained within the noise threshold. Current state-of-the-art (SOTA) methods employing spatial compatibility graphs or branch-and-bound (BnB) search mainly focus on registration under high outlier ratios. However, graph-based methods require at least quadratic space and time complexity for graph construction, while multi-stage BnB search methods often suffer from inaccuracy due to local optima between decomposed stages. This paper proposes a geometric maximum overlapping registration framework via rotation-only BnB search. The rigid transformation is decomposed using Chasles' theorem into a translation along rotation axis and a 2D rigid transformation. The optimal rotation axis and angle are searched via BnB, with residual parameters formulated as range maximum query (RMQ) problems. Firstly, the top-k candidate rotation axes are searched within a hemisphere parameterized by cube mapping, and the translation along each axis is estimated through interval stabbing of the correspondences projected onto that axis. Secondly, the 2D registration is relaxed to 1D rotation angle search with 2D RMQ of geometric overlapping for axis-aligned rectangles, which is solved deterministically in polynomial time using sweep line algorithm with segment tree. Experimental results on indoor 3DMatch/3DLoMatch scanning and outdoor KITTI LiDAR datasets demonstrate superior accuracy and efficiency over SOTA methods, while the time complexity is polynomial and the space complexity increases linearly with the number of points, even in the worst case.
>
---
#### [replaced 018] TwinBrainVLA: Unleashing the Potential of Generalist VLMs for Embodied Tasks via Asymmetric Mixture-of-Transformers
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出TwinBrainVLA模型，解决机器人任务中VLM因微调导致的泛化能力下降问题。通过双路径架构保留通用视觉理解，提升复杂操作任务性能。**

- **链接: [https://arxiv.org/pdf/2601.14133v2](https://arxiv.org/pdf/2601.14133v2)**

> **作者:** Bin Yu; Shijie Lian; Xiaopeng Lin; Yuliang Wei; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Xinming Wang; Bailing Wang; Cong Huang; Kai Chen
>
> **备注:** GitHub: https://github.com/ZGC-EmbodyAI/TwinBrainVLA
>
> **摘要:** The fundamental premise of Vision-Language-Action (VLA) models is to harness the extensive general capabilities of pre-trained Vision-Language Models (VLMs) for generalized embodied intelligence. However, standard robotic fine-tuning inevitably disrupts the pre-trained feature space, leading to "catastrophic forgetting" that compromises the general visual understanding we aim to leverage. To effectively utilize the uncorrupted general capabilities of VLMs for robotic tasks, we propose TwinBrainVLA, which coordinates two isomorphic VLM pathways: a frozen generalist (also called "Left Brain") and a trainable specialist (also called "Right Brain"). Our architecture utilizes a Asymmetric Mixture-of-Transformers (AsyMoT) mechanism, enabling the Right Brain to dynamically query and fuse intact semantic knowledge from the Left Brain with proprioceptive states. This fused representation conditions a flow-matching action expert for precise continuous control. Empirical results on SimplerEnv and RoboCasa benchmarks demonstrate that by explicitly retaining general capabilities, TwinBrainVLA achieves substantial performance gains over baseline models in complex manipulation tasks.
>
---
#### [replaced 019] Joint Learning of Depth, Pose, and Local Radiance Field for Large Scale Monocular 3D Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目3D重建任务，解决大场景下深度、位姿和辐射场分离导致的几何模糊、位姿漂移和内容不足问题。通过联合学习框架，提升重建精度与覆盖范围。**

- **链接: [https://arxiv.org/pdf/2512.18237v2](https://arxiv.org/pdf/2512.18237v2)**

> **作者:** Shahram Najam Syed; Yitian Hu; Yuchao Yao
>
> **备注:** 8 pages, 2 figures, 2 tables
>
> **摘要:** Photorealistic 3-D reconstruction from monocular video collapses in large-scale scenes when depth, pose, and radiance are solved in isolation: scale-ambiguous depth yields ghost geometry, long-horizon pose drift corrupts alignment, and a single global NeRF cannot model hundreds of metres of content. We introduce a joint learning framework that couples all three factors and demonstrably overcomes each failure case. Our system begins with a Vision-Transformer (ViT) depth network trained with metric-scale supervision, giving globally consistent depths despite wide field-of-view variations. A multi-scale feature bundle-adjustment (BA) layer refines camera poses directly in feature space--leveraging learned pyramidal descriptors instead of brittle keypoints--to suppress drift on unconstrained trajectories. For scene representation, we deploy an incremental local-radiance-field hierarchy: new hash-grid NeRFs are allocated and frozen on-the-fly when view overlap falls below a threshold, enabling city-block-scale coverage on a single GPU. Evaluated on the Tanks and Temples benchmark, our method reduces Absolute Trajectory Error to 0.001-0.021 m across eight indoor-outdoor sequences--up to 18x lower than BARF and 2x lower than NoPe-NeRF--while maintaining sub-pixel Relative Pose Error. These results demonstrate that metric-scale, drift-free 3-D reconstruction and high-fidelity novel-view synthesis are achievable from a single uncalibrated RGB camera.
>
---
#### [replaced 020] HAFO: A Force-Adaptive Control Framework for Humanoid Robots in Intense Interaction Environments
- **分类: cs.RO**

- **简介: 该论文提出HAFO框架，解决人形机器人在强力交互环境中的控制问题。通过双智能体强化学习，实现稳定运动与精准操作，提升力控性能。**

- **链接: [https://arxiv.org/pdf/2511.20275v4](https://arxiv.org/pdf/2511.20275v4)**

> **作者:** Chenhui Dong; Haozhe Xu; Wenhao Feng; Zhipeng Wang; Yanmin Zhou; Yifei Zhao; Bin He
>
> **摘要:** Reinforcement learning (RL) controllers have made impressive progress in humanoid locomotion and light-weight object manipulation. However, achieving robust and precise motion control with intense force interaction remains a significant challenge. To address these limitations, this paper proposes HAFO, a dual-agent reinforcement learning framework that concurrently optimizes both a robust locomotion strategy and a precise upper-body manipulation strategy via coupled training. We employ a constrained residual action space to improve dual-agent training stability and sample efficiency. The external tension disturbances are explicitly modeled using a spring-damper system, allowing for fine-grained force control through manipulation of the virtual spring. In this process, the reinforcement learning policy autonomously generates a disturbance-rejection response by utilizing environmental feedback. The experimental results demonstrate that HAFO achieves whole-body control for humanoid robots across diverse force-interaction environments using a single dual-agent policy, delivering outstanding performance under load-bearing and thrust-disturbance conditions, while maintaining stable operation even under rope suspension state.
>
---
