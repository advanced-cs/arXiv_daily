# 机器人 cs.RO

- **最新发布 66 篇**

- **更新 27 篇**

## 最新发布

#### [new 001] The Unified Autonomy Stack: Toward a Blueprint for Generalizable Robot Autonomy
- **分类: cs.RO**

- **简介: 该论文提出统一自主架构，解决多类型机器人在复杂环境中的自主导航问题，集成感知、规划与安全导航模块，实现高效任务执行。**

- **链接: [https://arxiv.org/pdf/2605.12735](https://arxiv.org/pdf/2605.12735)**

> **作者:** Mihir Dharmadhikari; Nikhil Khedekar; Mihir Kulkarni; Morten Nissov; Martin Jacquet; Angelos Zacharia; Marvin Harms; Albert Gassol Puigjaner; Philipp Weiss; Kostas Alexis
>
> **备注:** 35 pages, 22 figures, 8 tables
>
> **摘要:** We introduce and open-source the Unified Autonomy Stack, a system-level solution that enables resilient autonomy across diverse aerial and ground robot morphologies. The architecture centers on three synergistic modules -- multi-modal perception, multi-behavior planning, and multi-layered safe navigation -- that together deliver comprehensive mission autonomy. The stack fuses data from LiDAR, radar, vision, and inertial sensing, enabling (a) robust localization and mapping through factor graph-based fusion, (b) semantic scene understanding, (c) motion and informative path planning through sampling-based techniques adaptive across spatial scales, as well as (d) multi-layered safe navigation both through planning on the online reconstructed map and deep learning-driven exteroceptive policies alongside last-resort safety filters using control barrier functions. The resulting behaviors include safe GNSS-denied navigation into unknown and perceptually-degraded regions, exploration of complex environments, object discovery, and efficient inspection planning. The stack has been field-tested and validated on both aerial (rotorcraft) and ground (legged) robots operating in a host of demanding environments, including self-similar and smoke-filled settings, with complex geometries and high obstacle clutter. These tests demonstrate resilient performance in challenging conditions. To facilitate ease of adoption, we open-source the implementation alongside supporting documentation, validation, and evaluation datasets this https URL. A video giving the overview of the paper and the field experiments is available at this https URL.
>
---
#### [new 002] When Absolute State Fails: Evaluating Proprioceptive Encodings for Robust Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操控任务，旨在解决机器人在不同参考帧下性能下降的问题。通过研究本体感觉编码策略，提出一种相对帧表示，提升模型在分布内外的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.13067](https://arxiv.org/pdf/2605.13067)**

> **作者:** Maxime Alvarez; Ryo Watanabe; Paul Crook; Afshin Zeinaddini Meymand; Suvin Kurian; Pablo Ferreiro; Genki Sano
>
> **备注:** Accepted to ICRA 2026 Workshop: From Data to Decisions
>
> **摘要:** As end-to-end robotic policies are progressively deployed in the real world to solve real tasks, they face a gap between the training and inference conditions. Scaling the amount and diversity of the training data has shown some success in improving zero-shot generalization, yet robots still fail when faced with new, unseen test conditions. For instance, while robots with fixed frames of reference are common, those with moving frames pose a greater challenge for deployment. To address this specific instance of the issue, we present a study of strategies for encoding the robot's proprioceptive state to improve both in- and out-of-distribution performance at test time. Through a systematic study of joint representations, we find that a simple episode-wise relative frame provides the best trade-off between task performance and robustness, outperforming the baselines in extensive real-robot experiments conducted in a realistic test environment. The results suggest a practical path to leveraging data collected by robots with varying frames of reference and deployment to unseen test configurations.
>
---
#### [new 003] CUBic: Coordinated Unified Bimanual Perception and Control Framework
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出CUBic框架，解决双臂协作中的感知与控制统一问题，通过共享表示实现自主协调，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.13452](https://arxiv.org/pdf/2605.13452)**

> **作者:** Xingyu Wang; Pengxiang Ding; Jingkai Xu; Donglin Wang; Zhaoxin Fan
>
> **摘要:** Recent advances in visuomotor policy learning have enabled robots to perform control directly from visual inputs. Yet, extending such end-to-end learning from single-arm to bimanual manipulation remains challenging due to the need for both independent perception and coordinated interaction between arms. Existing methods typically favor one side -- either decoupling the two arms to avoid interference or enforcing strong cross-arm coupling for coordination -- thus lacking a unified treatment. We propose CUBic, a Coordinated and Unified framework for Bimanual perception and control that reformulates bimanual coordination as a unified perceptual modeling problem. CUBic learns a shared tokenized representation bridging perception and control, where independence and coordination emerge intrinsically from structure rather than from hand-crafted coupling. Our approach integrates three components: unidirectional perception aggregation, bidirectional perception coordination through two codebooks with shared mapping, and a unified perception-to-control diffusion policy. Extensive experiments on the RoboTwin benchmark show that CUBic consistently surpasses standard baselines, achieving marked improvements in coordination accuracy and task success rates over state-of-the-art visuomotor baselines.
>
---
#### [new 004] Guide, Think, Act: Interactive Embodied Reasoning in Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GTA-VLA框架，解决机器人在视觉-语言-动作任务中应对分布外变化和错误恢复的问题。通过引入用户空间引导，增强模型的推理与决策能力。**

- **链接: [https://arxiv.org/pdf/2605.13632](https://arxiv.org/pdf/2605.13632)**

> **作者:** Yiran Ling; Qing Lian; Jinghang Li; Qing Jiang; Tianming Zhang; Xiaoke Jiang; Chuanxiu Liu; Jie Liu; Lei Zhang
>
> **摘要:** In this paper, we propose GTA-VLA(Guide, Think, Act), an interactive Vision-Language-Action (VLA) framework that enables spatially steerable embodied reasoning by allowing users to guide robot policies with explicit visual cues. Existing VLA models learn a direct "Sense-to-Act" mapping from multimodal observations to robot actions. While effective within the training distribution, such tightly coupled policies are brittle under out-of-domain (OOD) shifts and difficult to correct when failures occur. Although recent embodied Chain-of-Thought (CoT) approaches expose intermediate reasoning, they still lack a mechanism for incorporating human spatial guidance, limiting their ability to resolve visual ambiguities or recover from mistakes. To address this gap, our framework allows users to optionally guide the policy with spatial priors, such as affordance points, boxes, and traces, which the subsequent reasoning process can directly condition on. Based on these inputs, the model generates a unified spatial-visual Chain-of-Thought that integrates external guidance with internal task planning, aligning human visual intent with autonomous decision-making. For practical deployment, we further couple the reasoning module with a lightweight reactive action head for efficient action execution. Extensive experiments demonstrate the effectiveness of our approach. On the in-domain SimplerEnv WidowX benchmark, our framework achieves a state-of-the-art 81.2% success rate. Under OOD visual shifts and spatial ambiguities, a single visual interaction substantially improves task success over existing methods, highlighting the value of interactive reasoning for failure recovery in embodied control. Details of the project can be found here: this https URL
>
---
#### [new 005] SECOND-Grasp: Semantic Contact-guided Dexterous Grasping
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，解决物理稳定与语义理解结合的问题。提出SECOND-Grasp框架，通过语义引导和几何一致性优化，提升抓取成功率和意图感知能力。**

- **链接: [https://arxiv.org/pdf/2605.13117](https://arxiv.org/pdf/2605.13117)**

> **作者:** Han Yi Shin; Heeju Ko; Jaewon Mun; Qixing Huang; Jaehyeok Lee; Sung June Kim; Honglak Lee; Sujin Jang; Sangpil Kim
>
> **摘要:** Achieving reliable robotic manipulation, such as dexterous grasping, requires a synergy between physically stable interactions and semantic task guidance, yet these objectives are often treated as separate, disjoint goals. In this paper, we investigate how to integrate dexterous grasping techniques, i.e., physically stable grasps for object lifting and language-guided grasp generation, to achieve both physical stability and semantic understanding. To this end, we propose SECOND-Grasp (SEmantic CONtact-guided Dexterous Grasping), a unified framework that enables robotic hands to dynamically adjust grasping strategies based on semantic reasoning while ensuring physical feasibility. We begin by obtaining coarse contact proposals through vision-language reasoning to infer where contacts should occur based on object properties, followed by segmentation to localize these regions across views. To further ensure consistency across multiple viewpoints, we introduce Semantic-Geometric Consistency Refinement (SGCR), which refines initial contact predictions by enforcing semantic consistency across views and removing geometrically invalid regions, yielding reliable 3D contact maps. Then, we derive a feasible hand pose for each contact map via inverse kinematics, generating a supervision signal for policy learning. Our approach, trained on DexGraspNet, consistently outperforms baselines in lifting success rate on both seen and unseen categories, achieving 98.2% and 97.7%, respectively, while also improving intent-aware grasping by 12.8% and 26.2%. We further show promising results on additional datasets and robotic hands, including Shadow Hand and Allegro Hand.
>
---
#### [new 006] Exploring Human-Robot Collaboration: Analysis of Interaction Modalities in Challenging Tasks
- **分类: cs.RO**

- **简介: 论文研究人机协作中的交互模式，比较被动、反应和主动三种方式。任务是记忆搭建七层彩色塔，解决如何提升协作体验的问题。通过实验验证主动支持更受用户欢迎。**

- **链接: [https://arxiv.org/pdf/2605.13380](https://arxiv.org/pdf/2605.13380)**

> **作者:** Simone Arreghini; Cristina Iani; Alessandro Giusti; Valeria Villani; Lorenzo Sabattini; Antonio Paolillo
>
> **摘要:** This work compares three interaction modalities for human-robot collaboration: passive, reactive, and proactive. We studied 18 participants assembling a seven-layer colored tower from memory while using nearby and distant blocks. In the passive modality participants worked alone; in the reactive modality a mobile robot helped only upon request; in the proactive modality it initiated brick delivery and error signaling without explicit requests. Although robot assistance increased completion time, most participants preferred collaboration: 67% preferred proactive behavior and 78% judged it most useful. These results suggest that timely proactive support can improve user experience in controlled collaborative tasks.
>
---
#### [new 007] Uncertainty-Aware 3D Position Refinement for Multi-UAV Systems
- **分类: cs.RO; cs.CR**

- **简介: 该论文属于多无人机系统定位任务，解决GNSS受限环境下的3D定位问题。通过融合局部估计与邻近约束，提升定位鲁棒性与抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2605.13500](https://arxiv.org/pdf/2605.13500)**

> **作者:** Hosam Alamleh; Damir Pulatov
>
> **摘要:** Reliable real-time 3D localization is essential for multi-UAV navigation, collision avoidance, and coordinated flight, yet onboard estimates can degrade under GNSS multipath, non-line-of-sight reception, vertical drift, and intentional interference. This paper presents a decentralized, lightweight 3D position-refinement layer that improves robustness by fusing each Unmanned Aerial Vehicle (UAV)'s local estimate with neighbor-shared state summaries and inter-UAV range or proximity constraints. The method performs uncertainty-aware neighborhood fusion by weighting each UAV's prior according to its reported covariance and weighting neighbor constraints according to link quality, ranging uncertainty, and a learned trust score. To support practical deployment, the framework explicitly handles cold start and temporary localization loss by inflating or substituting weak priors, allowing trusted neighborhood constraints to bootstrap and stabilize estimates until absolute sensing recovers. To mitigate the impact of faulty or malicious participants, each UAV applies a local range-consistency check, smoothed over time, to down-weight or exclude neighbors whose reported positions are incompatible with observed inter-UAV distances. Simulation experiments with 10 UAVs in a 3D volume show that the proposed refinement substantially reduces mean localization error during cold start, remains competitive after local estimators stabilize, and maintains lower error as the fraction of malicious nodes increases compared with fusion without trust. These results suggest that the approach can serve as a practical resilience layer for swarm operation in challenging environments.
>
---
#### [new 008] Driving Intents Amplify Planning-Oriented Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决连续动作策略在单次示范下模式崩溃的问题。提出DIAL框架，通过意图扩增和多意图优化，提升偏好强化学习性能。**

- **链接: [https://arxiv.org/pdf/2605.12625](https://arxiv.org/pdf/2605.12625)**

> **作者:** Hengtong Lu; Victor Shea-Jay Huang; Chengmin Yang; Pengfei Jing; Jifeng Dai; Yan Xie; Benjin Zhu
>
> **备注:** Work in progress. Project page: this https URL
>
> **摘要:** Continuous-action policies trained on a single demonstrated trajectory per scene suffer from mode collapse: samples cluster around the demonstrated maneuver and the policy cannot represent semantically distinct alternatives. Under preference-based evaluation, this caps best-of-N performance -- even oracle selection cannot recover what the sampling distribution does not contain. We introduce DIAL, a two-stage Driving-Intent-Amplified reinforcement Learning framework for preference-aligned continuous-action driving policies. In the first stage, DIAL conditions the flow-matching action head on a discrete intent label with classifier-free guidance (CFG), which expands the sampling distribution along distinct maneuver modes and breaks single-demonstration mode collapse. In the second stage, DIAL carries this expanded distribution into preference RL through multi-intent GRPO, which spans all intent classes within every preference group and prevents fine-tuning from re-collapsing around the currently preferred mode. Instantiated for end-to-end driving with eight rule-derived intents and evaluated on WOD-E2E: competitive Vision-to-Action (VA) and Vision-Language-Action (VLA) Supervised Finetuning (SFT) baselines plateau below the human-driven demonstration at best-of-128, with the strongest prior (RAP) capping at Rater Feedback Score (RFS) 8.5 even with best-of-64; intent-CFG sampling lifts this ceiling to RFS 9.14 at best-of-128, surpassing both the prior best (RAP 8.5) and the human-driven demonstration (8.13) for the first time; and multi-intent GRPO improves held-out RFS from 7.681 to 8.211, while every single-intent baseline peaks lower and degrades by training end. These results suggest that the bottleneck of preference RL on continuous-action policies trained from demonstrations is not only how to update the policy, but to expand and preserve the sampling distribution being optimized.
>
---
#### [new 009] Emotional Expression in Low-Degrees-of-Freedom Robots: Assessing Perception with Reachy Mini
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究低自由度机器人情感表达的感知问题，通过实验分析用户对Reachy Mini情感表达的识别与评价，探讨其在情感沟通中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.12786](https://arxiv.org/pdf/2605.12786)**

> **作者:** Amit Rogel; Elmira Yadollahi; Guy Laban
>
> **摘要:** Emotion expression is central to human--robot interaction, yet little is known about how people interpret affect on robots with sparse, non-anthropomorphic expressive capabilities. This study examined how people perceive emotional expressions displayed by Reachy Mini (Pollen Robotics and Hugging Face), a low-degree-of-freedom (low-DoF) robot with a constrained and distinctly non-human expressive repertoire. In an online within-subjects study, 100 participants viewed 10 short video clips of Reachy Mini expressing different emotions and, for each clip, identified the perceived emotion, rated its valence and arousal, and evaluated the robot on social-perception traits. Exact emotion recognition was modest overall and varied considerably across expressions, with anger, sadness, and interest recognized more reliably than emotions such as love, pleasure, shame, and disgust. However, participants were generally more successful at recovering broader affective meaning than exact emotion labels, particularly along valence and arousal dimensions. Emotional expressions also shaped social evaluation, as positive expressions were perceived as warmer and more sociable than negative ones, and animacy varied less across conditions. These findings suggest that even constrained robotic expressions can communicate affective meaning and influence social impressions, positioning Reachy Mini as a useful benchmark for studying affective communication in low-DoF robots.
>
---
#### [new 010] Local Conformal Calibration of Dynamics Uncertainty from Semantic Images
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于不确定性量化任务，解决动态模型在未知环境中的不确定性校准问题。通过引入OCULAR算法，利用感知信息实现对动态模型的局部校准，提供概率保障。**

- **链接: [https://arxiv.org/pdf/2605.13028](https://arxiv.org/pdf/2605.13028)**

> **作者:** Luís Marques; Dmitry Berenson
>
> **备注:** 26 pages, 8 figures. Accepted to the 17th World Symposium on the Algorithmic Foundations of Robotics (WAFR) 2026
>
> **摘要:** We introduce Observation-aware Conformal Uncertainty Local-Calibration (OCULAR), a conformal prediction-based algorithm that uses perception information to provide uncertainty quantification guarantees for unseen test-time environments. While previous conformal approaches lack the ability to discriminate between state-action space regions leading to higher or lower model mismatch, and require environment-specific data, our method uses data collected from visually similar environments to provably calibrate a given linear Gaussian dynamics model of arbitrary fidelity. The prediction regions generated from OCULAR are guaranteed to contain the future system states with, at least, a user-set likelihood, despite both aleatoric and epistemic uncertainty -- i.e., uncertainty arising from both stochastic disturbances and lack of data. Our guarantees are non-asymptotic and distribution-free, not requiring strong assumptions about the unknown real system dynamics. Our calibration procedure enables distinguishing between observation-velocity-action inputs leading to higher and lower next-state-uncertainty, which is helpful for probabilistically-safe planning. We numerically validate our algorithm on a double-integrator system subject to random perturbations and significant model mismatch, using both a simplified sensor and a more realistic simulated camera. Our approach appropriately quantifies uncertainty both when in-distribution and out-of-distribution, being comparatively volume-efficient to baselines requiring environment-specific data.
>
---
#### [new 011] HCSG: Human-Centric Semantic-Geometric Reasoning for Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决动态环境中机器人与人类互动的问题。提出HCSG框架，融合语义与几何推理，提升导航安全性与社会智能。**

- **链接: [https://arxiv.org/pdf/2605.13321](https://arxiv.org/pdf/2605.13321)**

> **作者:** Haoxuan Xu; Tianfu Li; Wenbo Chen; Yi Liu; Jin Wu; Huashuo Lei; Yunfan Lou; Lujia Wang; Hesheng Wang; Haoang Li
>
> **摘要:** VLN has achieved remarkable progress by scaling data and model capacity. However, the assumption of a static environment breaks down in real-world indoor scenarios, where robots inevitably encounter dynamic pedestrians. Existing human-aware approaches typically treat humans merely as moving obstacles based on implicit visual cues, lacking the explicit reasoning required to interpret human intentions or maintain social norms. To address this, we propose HCSG, the first human-centric framework for VLN. This framework provides a robust foundation for safe, socially intelligent navigation in dynamic human-robot environments that shifts the paradigm from passive collision avoidance to active human behavior understanding. Specifically, HCSG introduces a unified Human Understanding Module that synergizes two key capabilities: (i) geometric forecasting, which predicts human pose and trajectory to anticipate future motion dynamics; and (ii) semantic interpretation, which leverages a Vision-Language Model (VLM) to generate natural language descriptions of human actions and intentions. These semantic-geometric representations are fused into the agent's topological map for instruction-conditioned planning. Furthermore, a social distance loss is introduced to enforce socially compliant interaction distances. Extensive experiments on the HA-VLNCE benchmark demonstrate that HCSG significantly outperforms state-of-the-art methods, achieving a 14% improvement in Success Rate and a 34% reduction in Collision Rate. Our project can be seen at this https URL.
>
---
#### [new 012] Distributionally Robust Safety Under Arbitrary Uncertainties: A Safety Filtering Approach
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于安全控制任务，解决非线性系统在分布不确定性下的概率安全性问题。通过安全过滤框架，结合Wasserstein鲁棒方法，实现高效的安全验证与决策。**

- **链接: [https://arxiv.org/pdf/2605.12974](https://arxiv.org/pdf/2605.12974)**

> **作者:** Daniel M. Cherenson; Haejoon Lee; Taekyung Kim; Dimitra Panagou
>
> **备注:** 10 pages, 4 figures, submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** In this work, we study how to ensure probabilistic safety for nonlinear systems under distributional ambiguity. Our approach builds on a backup-based safety filtering framework that switches between a high-performance nominal policy and a certified backup policy to ensure safety. To handle arbitrary uncertainties from ambiguous distributions, i.e., where the distribution is not of specific structure and the true distribution is unknown, we adopt a distributionally robust (DR) formulation using Wasserstein ambiguity sets. Rather than solving a high-dimensional DR trajectory optimization problem online, we exploit the structure of backup-based safety filtering to reduce safety certification to a one-dimensional search over the switching time between nominal and backup policies. We then develop a sampling-based certification procedure with finite-sample guarantees, where empirical failure probabilities are compared against a Wasserstein-inflated threshold. We validate our method through simulations across three systems, from a Dubins vehicle to a high-speed racing car and a fighter jet, demonstrating the broad applicability and computational efficiency.
>
---
#### [new 013] Multi-Depth Uniform Coverage Path Planning for Unmanned Surface Vehicle Surveying
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决水下地形测绘中因深度变化导致覆盖不均的问题。通过引入深度信息优化路径和传感器配置，提升覆盖率。**

- **链接: [https://arxiv.org/pdf/2605.13123](https://arxiv.org/pdf/2605.13123)**

> **作者:** Maider Larrazabal; Tong Yang; Izaro Goienetxea; Jaime Valls Miro
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** This paper introduces a novel automatic coverage path planning algorithm for bathymetry surveying with unmanned surface vehicles. The detection range of the mapping sensor employed - a multibeam echo sounder - is heavily influenced by local seafloor depths. Hence, a path designed to uniformly cover the sea surface does not guarantee uniform coverage of the seafloor. Yet this is currently the typical process for bathymetric surveys, with the simplistic boustrophedon scheme along manually selected waypoints at constant depths being the most widespread planner used. The proposed scheme incorporates coarse prior depth information to pre-process the target region and adaptively guide path generation and sensing range configuration. By explicitly accounting for depth variations, the proposed algorithm designs a coverage path with optimised spacing between survey passes that adjusts the sensing beam aperture to achieve more consistent seafloor coverage. The proposed method is shown to offer significant improvements in both synthetic and real-world scenarios. Validations in challenging synthetic terrains achieves coverage ratios beyond 99%, a marked improvement when compared with traditional boustrophedon paths revealing a maximum 75% coverage. The same trend appears in realistic simulations using real bathymetric data from a coastal harbour, with coverage reaching over 92%, and significantly surpassing boustrophedon sweeps with coverage rates below 65%. Beyond improved performance, the scheme also brings a fully automated design, suitable for autonomous marine vehicles, thus offering practical utilities for real-world applications.
>
---
#### [new 014] FrameSkip: Learning from Fewer but More Informative Frames in VLA Training
- **分类: cs.RO**

- **简介: 该论文提出FrameSkip框架，解决VLA训练中帧监督不平衡问题，通过选择关键帧提升训练效果。属于机器人视觉-语言-动作学习任务。**

- **链接: [https://arxiv.org/pdf/2605.13757](https://arxiv.org/pdf/2605.13757)**

> **作者:** Bin Yu; Shijie Lian; Xiaopeng Lin; Zhaolong Shen; Yuliang Wei; Changti Wu; Hang Yuan; Haishan Liu; Bailing Wang; Cong Huang; Kai Chen
>
> **备注:** GitHub: this https URL
>
> **摘要:** Vision-Language-Action (VLA) policies are commonly trained from dense robot demonstration trajectories, often collected through teleoperation, by sampling every recorded frame as if it provided equally useful supervision. We argue that this convention creates a temporal supervision imbalance: long low-change segments dominate the training stream, while manipulation-critical transitions such as alignment, contact, grasping, and release appear only sparsely. We introduce FrameSkip, a data-layer frame selection framework that scores trajectory frames using action variation, visual-action coherence, task-progress priors, and gripper-transition preservation, then remaps training samples toward high-importance frames under a target retention ratio. Because FrameSkip operates only in the dataloader, it leaves the VLA architecture, action head, training objective, and inference procedure unchanged. Across RoboCasa-GR1, SimplerEnv, and LIBERO, FrameSkip improves the success-retention trade-off over full-frame training and simpler frame selection variants, achieving a macro-average success rate of 76.15% across the three benchmarks compared with 66.50% for full-frame training while using a compressed trajectory view that retains 20% of unique frames in the main setting.
>
---
#### [new 015] TouchAnything: A Dataset and Framework for Bimanual Tactile Estimation from Egocentric Video
- **分类: cs.RO**

- **简介: 该论文提出EgoTouch数据集和TouchAnything框架，解决从视觉推断触觉的问题，用于增强人机交互的物理感知。**

- **链接: [https://arxiv.org/pdf/2605.13083](https://arxiv.org/pdf/2605.13083)**

> **作者:** Jianyi Zhou; Ziteng Gao; Feiyang Hong; Zirui Liu; Guannan Zhang; Weisheng Dai; Ruichen Zhen; Chuqiao Lyu; Haotian Wu; Yinian Mao; Xushi Wang; Yuxiang Jiang; Wenbo Ding; Shuo Yang
>
> **摘要:** Egocentric human video data, which captures rich human-environment interactions and can be collected at scale, has become a key driver of embodied intelligence research. However, existing egocentric datasets typically lack tactile sensing, a critical modality that provides direct cues about contact, force, and pressure in human-object interaction. Without such signals, models struggle to learn physically grounded representations of real-world interaction dynamics. While tactile sensors provide these cues, deploying high-quality tactile hardware at scale remains expensive and cumbersome. This raises a central question: can tactile feedback be inferred directly from visual observations, enabling scalable tactile supervision for egocentric video data and supporting physically grounded embodied learning? To enable research in this direction, we introduce EgoTouch, a large-scale multi-view egocentric dataset with dense tactile supervision for bimanual hand-object interaction. EgoTouch comprises 208 manipulation tasks spanning 1,891 episodes in diverse indoor and outdoor environments, with synchronized multi-view RGB (head-mounted egocentric and dual wrist-mounted cameras), bimanual 3D hand pose, and continuous pressure maps from wearable tactile sensors. Building on EgoTouch, we introduce TouchAnything, a baseline multi-view vision-to-touch prediction framework that uses the egocentric view as the primary input and flexibly leverages available wrist-mounted views at inference time. Experiments show that incorporating wrist-mounted views generally improves tactile prediction over egocentric-only input, achieving up to 5.0% relative improvement in Contact IoU and 6.1% relative improvement in Volumetric IoU. We will publicly release the dataset, code, and benchmark.
>
---
#### [new 016] Integration of an Agent Model into an Open Simulation Architecture for Scenario-Based Testing of Automated Vehicles
- **分类: cs.RO; cs.SE**

- **简介: 该论文属于自动驾驶仿真任务，解决交通代理模型在不同仿真环境中的兼容性问题。通过构建标准化架构，实现模型的模块化集成与跨平台互操作。**

- **链接: [https://arxiv.org/pdf/2605.13539](https://arxiv.org/pdf/2605.13539)**

> **作者:** Christian Geller; Daniel Becker; Jobst Beckmann; Lutz Eckstein
>
> **摘要:** Simulative and scenario-based testing are crucial methods in the safety assurance for automated driving systems. To ensure that simulation results are reliable, the real world must be modeled with sufficient fidelity, including not only the static environment but also the surrounding traffic of a vehicle under test. Thus, the availability of traffic agent models is of common interest to model naturalistic and parameterizable behavior, similar to human drivers. The interchangeability of agent models across different simulation environments represents a major challenge and necessitates harmonization and standardization. To address this challenge, we present a standardized and modular simulation integration architecture that enables the tool-independent integration of traffic agent models. The architecture builds upon the Open Simulation Interface (OSI) as a structured message format and the Functional Mock-up Interface (FMI) for dynamic model exchange. Rather than introducing yet another model or simulation tool, we provide a reusable reference implementation that translates these standards into a practical integration blueprint, including clear interfaces, data mappings, and execution semantics. The generic nature of the architecture is demonstrated by integrating an exemplary agent model into three widely used simulation environments: OpenPASS, CARLA, and CarMaker. As part of the evaluation, we show that the model yields consistent behavior in all simulation platforms, thereby validating the interoperability, modularity, and standard compliance of the proposed architecture. The reference implementation lowers integration barriers, serves as a foundation for future research, and is made publicly available at this http URL
>
---
#### [new 017] DynoJEPP: Joint Estimation, Prediction and Planning in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文提出DynoJEPP框架，解决动态环境中估计、预测与规划的联合优化问题，防止信息污染，提升安全性。**

- **链接: [https://arxiv.org/pdf/2605.12897](https://arxiv.org/pdf/2605.12897)**

> **作者:** Mikolaj Kliniewski; Jesse Morris; Yiduo Wang; Ian R. Manchester; Viorela Ila
>
> **摘要:** DynoJEPP is a factor-graph-based framework that jointly formulates and simultaneously optimizes estimation, prediction, and planning in dynamic environments. In conventional factor-graph-based approaches that jointly formulate estimation, prediction, and planning, information from prediction and planning feeds back into state estimation, yielding corrupted estimates, undesired behaviors, and unsafe plans. To address this, DynoJEPP introduces a novel directed factor that enforces directional information flow within the factor graph, preventing prediction and planning from corrupting state estimation. We evaluate the impact of directed factors on inter-module interactions during navigation in both static and dynamic environments. Our results demonstrate that these factors are critical for safe operation, as without them, the robot collides in the majority of experiments. Building on this, we further introduce Cooperative DynoJEPP, which enables the ego robot to incorporate cooperative object behavior into its prediction and trajectory planning.
>
---
#### [new 018] Calibration-Free Gas Source Localization with Mobile Robots: Source Term Estimation Based on Concentration Measurement Ranking
- **分类: cs.RO**

- **简介: 该论文属于气体源定位任务，旨在解决移动机器人在复杂环境中准确定位气体源的问题。通过分析浓度测量排名，无需校准传感器即可实现有效定位。**

- **链接: [https://arxiv.org/pdf/2605.13208](https://arxiv.org/pdf/2605.13208)**

> **作者:** Wanting Jin; Agatha Duranceau; İzzet Kağan Erünsal; Alcherio Martinoli
>
> **备注:** This paper has been accepted for publication in the IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** Efficient Gas Source Localization (GSL) in real-world settings is crucial, especially in emergency scenarios. Mobile robots equipped with low-cost, in-situ gas sensors offer a safer alternative to human inspection in hazardous environments. Probabilistic algorithms enhance GSL efficiency with scattered gas measurements by comparing gas concentration measurements gathered by robots to physical dispersion models. However, accurately deriving gas concentrations from data acquired with low-cost sensors is challenging due to the nonlinear sensor response, environmental dependencies (e.g., humidity, temperature, and other gas influences), and robot motion. Mitigating these disturbance factors requires frequent sensor calibration in controlled environments, which is often impractical for real-world deployments. To overcome these issues, we propose a novel feature extraction algorithm that leverages the relative ranking of gas measurements within the dynamically accumulated dataset. By comparing the rank differences between gathered and modeled values, we estimate the probabilistic distribution of source locations across the entire environment. We validate our approach in high-fidelity simulations and physical experiments, demonstrating consistent localization accuracy with uncalibrated gas sensors. Compared to existing methods, our technique eliminates the need for gas sensor calibration, making it well-suited for real-world applications.
>
---
#### [new 019] Occlusion-Based Object Transportation Around Obstacles With a Swarm of Miniature Robots
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于群体机器人运输任务，解决物体绕过障碍物运输问题。通过建立子目标链，扩展了基于遮挡的策略，实现无通信的分布式运输。**

- **链接: [https://arxiv.org/pdf/2605.13006](https://arxiv.org/pdf/2605.13006)**

> **作者:** Breno Cunha Queiroz; Daniel MacRae
>
> **备注:** 25 pages, 9 figures, 6 tables. Accepted for publication in the journal Swarm Intelligence
>
> **摘要:** Swarm robotics utilises decentralised self-organising systems to form complex collective behaviours built from the bottom-up using individuals that have limited capabilities. Previous work has shown that simple occlusion-based strategies can be effective in using swarm robotics for the task of transporting objects to a goal position. However, this strategy requires a clear line-of-sight between the object and the goal. In this paper, we extend this strategy by allowing robots to form sub-goals; enabling any member of the swarm to establish a wider range of visibility of the goal, ultimately forming a chain of sub-goals between the object and the goal position. We do so while preserving the fully decentralised and communication-free nature of the original strategy, while maintaining performance in object-free scenarios. In five sets of simulated experiments, we demonstrate the generalisability of our proposed strategy. Our finite-state machine allows a sufficiently large swarm to transport objects around obstacles that block the goal. The method is robust to varying starting positions and can handle both concave and convex shapes.
>
---
#### [new 020] Towards Long-horizon Embodied Agents with Tool-Aligned Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决长时序任务中VLA模型的局限性。通过分层结构和工具对齐训练，提升任务成功率和执行精度。**

- **链接: [https://arxiv.org/pdf/2605.13119](https://arxiv.org/pdf/2605.13119)**

> **作者:** Zixing Lei; Changxing Liu; Yichen Xiong; Minhao Xiong; Yuanzhuo Ding; Zhipeng Zhang; Weixin Li; Siheng Chen
>
> **摘要:** Vision-language-action (VLA) models are effective robot action executors, but they remain limited on long-horizon tasks due to the dual burden of extended closed-loop planning and diverse physical operations. We therefore propose VLAs-as-Tools, a strategy that distributes this burden across a high-level vision language model (VLM) agent for temporal reasoning and a family of specialized VLA tools for diverse local physical operations. The VLM handles scene analysis, global planning, and recovery, while each VLA tool executes a bounded subtask. To tightly couple agent planning with VLA tool execution in long-horizon tasks, we introduce a VLA tool-family interface that exposes explicit tool selection and in-execution progress feedback, enabling efficient event-triggered agent replanning without continuous agent polling. To obtain diverse specialized VLA tools that faithfully follow agent invocations, we further propose Tool-Aligned Post-Training (TAPT), which constructs invocation-aligned training units for instruction following and adopts tool-family residual adapters for efficient tool specialization. Experiments show that VLAs-as-Tools improves the success rate of $\pi_{0.5}$ by 4.8 points on LIBERO-Long and 23.1 points on RoboTwin, and further enhances invocation fidelity by 15.0 points as measured by Non-biased Rate. Code will be released.
>
---
#### [new 021] Dynamics Computation of Soft-Rigid Hybrid-Link System and Its Application to Motion Analysis of an Athlete Wearing Sport Prosthesis
- **分类: cs.RO**

- **简介: 该论文属于运动分析任务，解决运动员穿戴柔性假肢时的运动建模问题，通过构建软-刚混合系统，实现运动重建与力估算。**

- **链接: [https://arxiv.org/pdf/2605.13192](https://arxiv.org/pdf/2605.13192)**

> **作者:** Sunghee Kim; Yuta Shimane; Taiki Ishigaki; Ko Yamamoto
>
> **摘要:** This paper presents a motion analysis framework for an athlete wearing sport-specific flexible prosthesis based on the soft-rigid hybrid-link system. Such a motion analysis is a challenging problem because we need to consider the interaction force between the rigid human skeleton system and a flexible prosthesis. However, most of human musculoskeletal models are based on the computation framework of a rigid-body multi-link system. Recently in soft robotics research field, fast and efficient modeling methods were developed for a flexible rod deformation, which allows us to build a hybrid-link system that integrates rigid-link and soft-bodies in a unified formulation. We apply inverse kinematics of the hybrid-link system to motion reconstruction from a motion captured data, and also present the estimation of the joint torques and ground reaction force by inverse dynamics. Through a human subject experiment, we show that the inverse dynamics achieved approximately 12% error on the ground reaction force estimation. Furthermore, we provide the muscle force estimation considering muscle amputation and interaction force with the prosthesis leg deformation.
>
---
#### [new 022] MindVLA-U1: VLA Beats VA with Unified Streaming Architecture for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MindVLA-U1，解决自动驾驶中视觉-语言-动作模型的统一问题，通过流式架构提升规划质量与效率。**

- **链接: [https://arxiv.org/pdf/2605.12624](https://arxiv.org/pdf/2605.12624)**

> **作者:** Yuzhou Huang; Benjin Zhu; Hengtong Lu; Victor Shea-Jay Huang; Haiming Zhang; Wei Chen; Jifeng Dai; Yan Xie; Hongsheng Li
>
> **备注:** Work in progress. Project page: this https URL
>
> **摘要:** Autonomous driving has progressed from modular pipelines toward end-to-end unification, and Vision-Language-Action (VLA) models are a natural extension of this journey beyond Vision-to-Action (VA). In practice, driving VLAs have often trailed VA on planning quality, suggesting that the difficulty is not simply model scale but the interface through which semantic reasoning, temporal context, and continuous control are combined. We argue that this gap reflects how VLA has been built -- as isolated subtask improvements that fail to compose into coherent driving capabilities -- rather than what VLA is. We present MindVLA-U1, the first unified streaming VLA architecture for autonomous driving. A unified VLM backbone produces autoregressive language tokens and flow-matching continuous action trajectories in a single forward pass over one shared representation, preserving the natural output form of each modality. A streaming design processes the driving video framewise rather than as fixed video-action chunks, while a learned memory channel carries temporal context across frames so planned trajectories evolve smoothly without redundant multi-frame VLM modeling. The unified architecture admits fast/slow execution on dense/sparse Mixture-of-Transformers (MoT) backbones via flexible self-attention context management, and exposes a measurable language-to-action route: a language-predicted driving intent steers action diffusion through classifier-free guidance (CFG), turning language-side intent into a control signal for continuous trajectory generation. On the long-tail WOD-E2E benchmark, MindVLA-U1 surpasses experienced human drivers for the first time (8.20 RFS vs. 8.13 GT RFS) with 2 diffusion steps, achieves state-of-the-art planning ADEs over prior VA/VLA methods by large margins, and matches VA-class throughput (16 FPS vs. RAP-DINO's 18 FPS) while preserving natural-language interfaces.
>
---
#### [new 023] Identification of Non-Transversal Bifurcations of Linkages
- **分类: cs.RO**

- **简介: 该论文属于机构运动分析任务，旨在解决非横截分支区分问题。通过改进运动切锥算法，实现对不同运动分支的有效分离。**

- **链接: [https://arxiv.org/pdf/2605.13094](https://arxiv.org/pdf/2605.13094)**

> **作者:** Andreas Mueller; P. C. López Custodio; J. S. Dai
>
> **备注:** Paper No: DETC2020-22301, V010T10A090; 8 pages
>
> **摘要:** The local analysis is an established approach to the study of singularities and mobility of linkages. Key result of such analyses is a local picture of the finite motion through a configuration. This reveals the finite mobility at that point and the tangents to smooth motion curves. It does, however, not immediately allow to distinguish between motion branches that do not intersect transversally (which is a rather uncommon situation that has only recently been discussed in the literature). The mathematical framework for such a local analysis is the kinematic tangent cone. It is shown in this paper that the constructive definition of the kinematic tangent cone already involves all information necessary to separate different motion branches. A computational method is derived by amending the algorithmic framework reported in previous publications.
>
---
#### [new 024] Few-Shot Physics-Informed Neural Network for Shape Reconstruction of Concentric-Tube Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人建模任务，旨在解决CTRs形状重建的精度与数据需求问题。通过融合物理方程与神经网络，提出一种少样本物理信息模型，实现高精度、实时的运动状态估计。**

- **链接: [https://arxiv.org/pdf/2605.12790](https://arxiv.org/pdf/2605.12790)**

> **作者:** Navid Feizi; Filipe C. Pedrosa; Rajni V. Patel; Jagadeesan Jayender
>
> **备注:** to be published in 2026 IEEE International Conference on Robotics & Automation proceedings
>
> **摘要:** Modeling concentric tube robots (CTRs) involves complex nonlinear continuum mechanics, and despite recent progress, physics-based models often lack an accurate representation of the experimental setups. To overcome these limitations, deep neural network-based models have been explored as alternatives with superior accuracy; however, they often overlook known mechanics, require large training datasets, and typically discard shape estimation of the robot. We present a physics-informed neural network (PINN) for kinematic modeling of a 6-DoF CTR with three pre-curved tubes that embeds the Cosserat rod differential equations and learns from few-shot observational data, balancing physics priors with data-driven fitting. PINN enables full-state estimation of shape, twist angle, torsional strain, bending moment, and orientation. Benchmark tests show a mean shape error below 1% of the robot length and accurately recovered other kinematic states, outperforming a purely physics-based Cosserat rod model baseline while using a minimal training set. The resulting model is also computationally efficient and robust, making it well-suited for real-time control applications.
>
---
#### [new 025] Design of Magnetic Continuum Robots with Tunable Force Response Using Rotational Ring Pairs
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决传统磁控连续机器人无法在线调节末端力响应的问题。通过旋转环对结构，实现磁响应的动态调整，提升操控灵活性与临床适用性。**

- **链接: [https://arxiv.org/pdf/2605.13613](https://arxiv.org/pdf/2605.13613)**

> **作者:** Alex Sayres; Giovanni Pittiglio
>
> **备注:** 7 pages, 6 figures, Accepted to ISMR 2026
>
> **摘要:** In this paper, we discuss a novel continuum robot design that enables the online tuning of the magnetic response at its tip. The proposed method allows for the change of both effective magnetic direction and intensity, introducing steering DOF without the need to control the external fields. This is unattainable with classical designs, which rely on fixed internal magnetic content and steer solely under the effect of a controllable magnetic field. The proposed robot design can be used in both controllable and fixed magnetic fields, potentially widening the clinical applicability of these robots. We experimentally show a max tip deflection of 33.8 mm from the resting state (23 % of the length of the robot). We discuss a model based on modified beam theory that captures the mechanical behavior of the continuum robot, with a mean absolute tip tracking error of 1.86 mm (1.2 % of the length) and maximum errors of less than 4.8 mm (3.2 % of the length) for all experimental points.
>
---
#### [new 026] Robot Squid Game: Quadrupedal Locomotion for Traversing Narrow Tunnels
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决四足机器人在狭窄隧道中的自主移动问题。通过强化学习与策略蒸馏方法，提升机器人在复杂环境中的适应能力。**

- **链接: [https://arxiv.org/pdf/2605.13665](https://arxiv.org/pdf/2605.13665)**

> **作者:** Amir Hossain Raj; Dibyendu Das; Xuesu Xiao
>
> **摘要:** Quadruped robots demonstrate exceptional potential for navigating complex terrain in critical applications such as search and rescue missions and infrastructure inspection However autonomous traversal of confined 3D environments including tunnels caves and collapsed structures remains a significant challenge Existing methods often struggle with rigid gait patterns limited adaptability to diverse geometries and reliance on oversimplified environmental assumptions This paper introduces a Reinforcement Learning RL framework that combines procedural environment generation with policy distillation to enable robust locomotion across various tunnel configurations Our approach leverages a teacher student training paradigm where specialized expert policies trained on procedurally generated tunnel geometries transfer their knowledge to a unified student policy This strategy eliminates the need for complex reward shaping in end-to-end RL training simplifying the process by breaking down complicated tasks into smaller more manageable components that are easier for the robot to learn By synthesizing diverse tunnel structures during training and distilling navigation strategies into a generalizable policy our method achieves consistent traversal across complex spatial constraints where conventional approaches fail We demonstrate through both simulation and real world experiments that our method enables quadruped robots to successfully traverse challenging confined tunnel environments
>
---
#### [new 027] Realtime-VLA FLASH: Speculative Inference Framework for Diffusion-based VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型的实时推理任务，旨在解决dVLA高延迟问题。通过引入轻量级草稿模型和验证机制，提升推理效率，降低延迟。**

- **链接: [https://arxiv.org/pdf/2605.13778](https://arxiv.org/pdf/2605.13778)**

> **作者:** Jiahui Niu; Kefan Gu; Yucheng Zhao; Shengwen Liang; Tiancai Wang; Xing Hu; Ying Wang; Huawei Li
>
> **摘要:** Diffusion-based vision-language-action models (dVLAs) are promising for embodied intelligence but are fundamentally limited in real-time deployment by the high latency of full inference. We propose Realtime-VLA FLASH, a speculative inference framework that eliminates most full inference calls during replanning by introducing a lightweight draft model with parallel verification via the main model's Action Expert and a phase-aware fallback mechanism that reverts to the full inference pipeline when needed. This design enables low-latency, high-frequency replanning without sacrificing reliability. Experiments show that on LIBERO, FLASH largely preserves task performance by replacing many 58.0 ms full-inference rounds with speculative rounds as fast as 7.8 ms, lowering task-level average inference latency to 19.1 ms (3.04x speedup). We additionally demonstrate effectiveness on real-world conveyor-belt sorting, highlighting its practical impact for latency-critical embodied tasks.
>
---
#### [new 028] Causality-Aware End-to-End Autonomous Driving via Ego-Centric Joint Scene Modeling
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决传统方法忽略车辆与周围交通参与者因果关系的问题。提出CaAD框架，通过联合建模和策略对齐提升轨迹预测可靠性。**

- **链接: [https://arxiv.org/pdf/2605.13646](https://arxiv.org/pdf/2605.13646)**

> **作者:** Seokha Moon; Minseung Lee; Joon Seo; Jinkyu Kim; Jungbeom Lee
>
> **摘要:** End-to-end autonomous driving, which bypasses traditional modular pipelines by directly predicting future trajectories from sensor inputs, has recently achieved substantial progress. However, existing methods often overlook the causal inter-dependencies in ego-vehicle planning, ignoring the reciprocal relations between the ego vehicle and surrounding agents. This causal oversight leads to inconsistent and unreliable trajectory predictions, especially in interaction-critical scenarios where ego decisions and neighboring agent behaviors must be reasoned about jointly. To address this limitation, we propose CaAD, a Causality-aware end-to-end Autonomous Driving framework that captures these dependencies within a shared latent scene representation. First, we propose a ego-centric joint-causal modeling module that builds on the marginal prediction branch, and learns causal dependencies between the ego vehicle and interaction-relevant agents. Second, we employ a causality-aware policy alignment stage implemented with joint-mode embeddings to align the stochastic ego policy with planning-oriented closed-loop feedback computed from surrounding traffic and map context. On the Bench2Drive and NAVSIM benchmarks, CaAD demonstrates strong closed-loop planning performance, achieving a Driving Score of 87.53 and Success Rate of 71.81 on Bench2Drive, and a PDMS of 91.1 on NAVSIM.
>
---
#### [new 029] AttenA+: Rectifying Action Inequality in Robotic Foundation Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决现有模型对动作信息均匀处理导致的性能限制。通过引入AttenA+框架，基于速度调整训练权重，提升模型在复杂任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.13548](https://arxiv.org/pdf/2605.13548)**

> **作者:** Daojie Peng; Fulong Ma; Jiahang Cao; Qiang Zhang; Xupeng Xie; Jian Guo; Ping Luo; Andrew F. Luo; Boyu Zhou; Jun Ma
>
> **摘要:** Existing robotic foundation models, while powerful, are predicated on an implicit assumption of temporal homogeneity: treating all actions as equally informative during optimization. This "flat" training paradigm, inherited from language modeling, remains indifferent to the underlying physical hierarchy of manipulation. In reality, robot trajectories are fundamentally heterogeneous, where low-velocity segments often dictate task success through precision-demanding interactions, while high-velocity motions serve as error-tolerant transitions. Such a misalignment between uniform loss weighting and physical criticality fundamentally limits the performance of current Vision-Language-Action (VLA) models and World-Action Models (WAM) in complex, long-horizon tasks. To rectify this, we introduce AttenA+, an architecture-agnostic framework that prioritizes kinematically critical segments via velocity-driven action attention. By reweighting the training objective based on the inverse velocity field, AttenA+ naturally aligns the model's learning capacity with the physical demands of manipulation. As a plug-and-play enhancement, AttenA+ can be integrated into existing backbones without structural modifications or additional parameters. Extensive experiments demonstrate that AttenA+ significantly elevates the ceilings of current state-of-the-art models. Specifically, it improves OpenVLA-OFT to 98.6% (+1.5%) on the Libero benchmark and pushes FastWAM to 92.4% (+0.6%) on RoboTwin 2.0. Real-world validation on a Franka manipulator further showcases its robustness and cross-task generalization. Our work suggests that mining the intrinsic structural priors of action sequences offers a highly efficient, physics-aware complement to standard scaling laws, paving a new path for general-purpose robotic control.
>
---
#### [new 030] LMPath: Language-Mediated Priors and Path Generation for Aerial Exploration
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机搜索任务，解决传统方法忽略语义信息导致效率低的问题。通过语言模型和视觉模型生成探索先验，优化无人机路径规划。**

- **链接: [https://arxiv.org/pdf/2605.13782](https://arxiv.org/pdf/2605.13782)**

> **作者:** Jonathan A. Diller; Fernando Cladera; Camillo J. Taylor; Vijay Kumar
>
> **备注:** Poster at 2026 AI-Driven Safe Aerial Robotics Workshop
>
> **摘要:** Traditional autonomous UAV search missions rely on geometric coverage patterns that ignore the semantic context of the target, leading to significant time waste in large-scale environments. In this paper we present LMPath, a pipeline for generating language-mediated exploration priors for Unmanned Aerial Vehicle (UAV) search missions that leverages semantics. Given a basic geofence and an object of interest prompt, LMPath uses generative language models to determine what regions of the environment should contain that object and a foundation vision model ran over satellite imagery to segment sub-regions that form the exploration prior. This prior can then be used to generate UAV paths with various objectives, such as minimizing the expected time to locate the object of interest, maximizing the probability that the object is found given a limited travel distance, or narrowing down the search space to sub-regions that are most likely to contain the object. To demonstrate it's capabilities, we used LMPath to generate various UAV paths and ran them using a real UAV over large-scale environments. We also ran simulations to demonstrate how paths generated using LMPath outperform traditional path planning approaches for search missions.
>
---
#### [new 031] MUJICA: Multi-skill Unified Joint Integration of Control Architecture for Wheeled-Legged Robots
- **分类: cs.RO**

- **简介: 该论文提出MUJICA框架，解决轮腿机器人在复杂环境中实现多技能自适应控制的问题，通过统一策略整合多种运动技能，提升仿生机器人的环境适应能力。**

- **链接: [https://arxiv.org/pdf/2605.13058](https://arxiv.org/pdf/2605.13058)**

> **作者:** Yuqi Li; Peng Zhai; Yueqi Zhang; Xiaoyi Wei; Quancheng Qian; Zhengxu He; Qianxiang Yu; Lihua Zhang
>
> **摘要:** Wheeled-legged robots hold promise for traversing complex terrains and offer superior mobility compared to legged robots. However, wheeled-legged robots must effectively balance both wheeled driving and legged control. Furthermore, due to noisy proprioceptive sensing and real-world motor constraints, realizing robust and adaptive locomotion at peak performance of motors remains challenging. We propose the Multi-skill Unified Joint Integration of Control Architecture (MUJICA), a unified, fully proprioceptive control framework for wheeled-legged robots that integrates diverse low-level skills-including omnidirectional moving, high platform climbing, and fall recovery-within a single policy. All skills, distinguished by unique indicator variables, are trained jointly with accurate DC-motor constraint modeling. Additionally, a high-level skill selector is learned to dynamically choose the optimal skill based solely on proprioceptions, enabling adaptive responses to the surrounding environment. Therefore, MUJICA enhances sim-to-real robustness and enables seamless transitions across diverse locomotion modes, facilitating autonomous adjustment to the environment. We validate our framework in both simulation and real-world experiments on the Unitree Go2-W robot, demonstrating significant improvements in adaptability and task success in unstructured environments.
>
---
#### [new 032] LEXI-SG: Monocular 3D Scene Graph Mapping with Room-Guided Feed-Forward Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LEXI-SG，用于单目RGB相机构建开放词汇3D场景图。解决无深度传感器下场景图构建问题，通过房间分割和全局对齐实现高效、准确的三维重建与语义理解。**

- **链接: [https://arxiv.org/pdf/2605.13741](https://arxiv.org/pdf/2605.13741)**

> **作者:** Christina Kassab; Hyeonjae Gil; Matías Mattamala; Ayoung Kim; Maurice Fallon
>
> **摘要:** Scene graphs are becoming a standard representation for robot navigation, providing hierarchical geometric and semantic scene understanding. However, most scene graph mapping methods rely on depth cameras or LiDAR sensors. In this work, we present LEXI-SG, the first dense monocular visual mapping system for open-vocabulary 3D scene graphs using only RGB camera input. Our approach exploits the semantic priors of open-vocabulary foundation models to partition the scene into rooms, deferring feed-forward reconstruction to when each room is fully observed -- enabling scalable dense mapping without sliding-window scale inconsistencies. We propose a room-based factor graph formulation to globally align room reconstructions while preserving local map consistency and naturally imposing the semantic scene graph hierarchy. Within each room, we further support open-vocabulary object segmentation and tracking. We validate LEXI-SG on indoor scenes from the Habitat-Matterport 3D and self-collected egocentric office sequences. We evaluate its performance against existing feed-forward SLAM methods, as well as established scene graphs baselines. We demonstrate improved trajectory estimation and dense reconstruction, as well as, competitive performance in open-vocabulary segmentation. LEXI-SG shows that accurate, scalable, open-vocabulary 3D scene graphs can be achieved from monocular RGB alone. Our project page and office sequences are available here: this https URL.
>
---
#### [new 033] Multistep Belief Space Dynamics Learning For Risk-Aware Control
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶任务，解决风险感知控制问题。通过学习分布动态模型，优化实时预测与控制，提升车辆在复杂环境中的智能行为。**

- **链接: [https://arxiv.org/pdf/2605.12628](https://arxiv.org/pdf/2605.12628)**

> **作者:** Jason Gibson; Bogdan Vlahov; Patrick Spieler; Evangelos A. Theodorou
>
> **摘要:** As autonomous vehicles move from a simplified research setting to practical use, there exists a large gap between the dynamic behavior of a human driving and an autonomous system. Risk-aware behavior needs to naturally develop in order to scale to the demands of the real world. A major issue for risk-aware planning and control has been predicting how dynamical uncertainty evolves through time and optimizing plans that account for this without being overly conservative. Here, we present a learning framework to predict distributional dynamics that can be optimized in real time for Model Predictive Control (MPC). We explore the importance of structure when learning distributional dynamics for use in MPC. A rigorous ablation study is conducted on a large dataset of real world off-road driving that shows the impact of deviations from our proposed structure. Furthermore, we deploy our learned model and planning stack on a full sized vehicle in challenging off-road conditions. Our planning architecture is able to naturally regulate the speed of the vehicle based on the environment and consistently demonstrates intelligent behavior over miles of diverse terrain.
>
---
#### [new 034] Loiter UAV Reinsertion Guidance for Fixed-wing UAV Corridors
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机路径规划任务，解决固定航线中无人机从环形等待区安全返回主航线的问题。通过算法计算无人机速度以确保安全再插入。**

- **链接: [https://arxiv.org/pdf/2605.13822](https://arxiv.org/pdf/2605.13822)**

> **作者:** Pradeep J; Kedarisetty Siddhardha; Ashwini Ratnoo
>
> **摘要:** This paper considers fixed-wing unmanned aerial vehicle (UAV) corridors comprising a main lane, a circular loiter lane for managing traffic congestion, and transit lanes connecting the two. In particular, we address the problem of conflict-free reinsertion of UAVs from the loiter lane back into the main lane. The loiter lane contains a fixed number of equidistant virtual slots that UAVs can occupy. Reinsertion of loiter UAVs into the main lane becomes essential either due to reduced traffic in the main lane or due to a loiter UAV needing to reach its destination urgently. Given the total number of loiter slots, UAV speed limits, and the minimum safety distance, a guidance algorithm is developed to compute the required speed of a loiter UAV in the transit lane to ensure safe reinsertion. The proposed guidance and automation strategies are validated through numerical simulations.
>
---
#### [new 035] Galilean State Estimation for Inertial Navigation Systems with Unknown Time Delay
- **分类: cs.RO**

- **简介: 该论文属于导航系统任务，解决GNSS时间延迟影响INS精度的问题。提出一种基于伽利略对称性的等变滤波器，实现导航状态与时间延迟的联合估计。**

- **链接: [https://arxiv.org/pdf/2605.13266](https://arxiv.org/pdf/2605.13266)**

> **作者:** Giulio Delama; Martin Scheiber; Yixiao Ge; Tarek Hamel; Stephan Weiss; Robert Mahony
>
> **摘要:** Many Inertial Navigation Systems (INS) use Global Navigation Satellite System (GNSS) position as the primary measurement to drive filter performance and bound error growth. However, commercial-grade GNSS receivers introduce unknown measurement delays ranging from 50 ms to 300 ms depending on sensor quality and operating mode. Such time delays can significantly degrade INS performance unless they are explicitly compensated for. Existing algorithms commonly estimate this delay offline, run the filter concurrently with GNSS measurements using buffered Inertial Measurement Unit (IMU) data, and predict the current state by forward-integrating buffered inertial measurements via IMU preintegration. The state-of-the-art online method is an Extended Kalman Filter (EKF) that explicitly models the time delay as a state parameter, which defines the preintegration duration. This paper introduces a novel geometric framework for modeling time-delayed INS, in which Galilean symmetry is leveraged to provide a joint representation of space and time for consistent state estimation. An Equivariant Filter (EqF) is derived for the coupled estimation of navigation states and time delay. Validation is performed on two fixed-wing Uncrewed Aerial Vehicles (UAV) with GNSS time lags of 90 ms and 120 ms. The test flights last two to three minutes. Simulations further investigate delays up to 500 ms and provide a statistical comparison against the state-of-the-art EKF. Results show that the EqF preserves accuracy and consistency, while the EKF lacks consistency and its performance degrades significantly with increasing measurement delays.
>
---
#### [new 036] Action Emergence from Streaming Intent
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决动作生成问题。提出Streaming Intent方法，通过场景推理生成符合意图的驾驶动作，提升路径规划的多样性和可控性。**

- **链接: [https://arxiv.org/pdf/2605.12622](https://arxiv.org/pdf/2605.12622)**

> **作者:** Pengfei Jing; Victor Shea-Jay Huang; Hengtong Lu; Jifeng Dai; Xie Yan; Benjin Zhu
>
> **备注:** Work in progress. Project page: this https URL
>
> **摘要:** We formalize action emergence as a target capability for end-to-end autonomous driving: the ability to generate physically feasible, semantically appropriate, and safety-compliant actions in arbitrary, long-tail traffic scenes through scene-conditioned reasoning rather than retrieval or interpolation of learned scene-action mappings. We show that previous paradigms cannot deliver action emergence: autoregressive trajectory decoders collapse the inherently multimodal future into a single averaged output, while diffusion and flow-matching generators express multimodality but are not steerable by reasoned intent. We propose Streaming Intent as a concrete way to approach action emergence: a mechanism that makes driving intent (i) semantically streamed through a continuous chain-of-thought that causally derives the intent from scene understanding, and (ii) temporally streamed across clips so that intent commitments remain coherent along the driving horizon. We realize Streaming Intent in a VLA model we call SI (Streaming Intent). SI autoregressively decodes a four-step chain-of-thought and emits an intent token; the decoded intent then drives classifier-free guidance (CFG) on a flow-matching action head, requiring only two denoising steps to generate the final trajectory. On the Waymo End-to-End benchmark, SI achieves competitive aggregate performance, with an RFS score of 7.96 on the validation set and 7.74 on the test set. Beyond aggregate metrics, the model demonstrates -- to our knowledge for the first time in a fully end-to-end VLA -- intent-faithful controllability: for a fixed scene, varying the intent class at inference yields qualitatively distinct yet consistently high-quality plans, arising purely from data-driven learning without any pre-built trajectory bank or hand-coded post-hoc selector.
>
---
#### [new 037] What Limits Vision-and-Language Navigation ?
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉-语言导航任务，旨在解决模拟到现实部署中的性能下降问题。提出StereoNav框架，通过空间定位和立体视觉提升导航鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.13328](https://arxiv.org/pdf/2605.13328)**

> **作者:** Yunheng Wang; Yuetong Fang; Taowen Wang; Lusong Li; Kun Liu; Junzhe Xu; Zizhao Yuan; Yixiao Feng; Jiaxi Zhang; Wei Lu; Zecui Zeng; Renjing Xu
>
> **摘要:** Vision-and-Language Navigation (VLN) is a cornerstone of embodied intelligence. However, current agents often suffer from significant performance degradation when transitioning from simulation to real-world deployment, primarily due to perceptual instability (e.g., lighting variations and motion blur) and under-specified instructions. While existing methods attempt to bridge this gap by scaling up model size and training data, we argue that the bottleneck lies in the lack of robust spatial grounding and cross-domain priors. In this paper, we propose StereoNav, a robust Vision-Language-Action framework designed to enhance real-world navigation consistency. To address the inherent gap between synthetic training and physical execution, we introduce Target-Location Priors as a persistent bridge. These priors provide stable visual guidance that remains invariant across domains, effectively grounding the agent even when instructions are vague. Furthermore, to mitigate visual disturbances like motion blur and illumination shifts, StereoNav leverages stereo vision to construct a unified representation of semantics and geometry, enabling precise action prediction through enhanced depth awareness. Extensive experiments on R2R-CE and RxR-CE demonstrate that StereoNav achieves state-of-the-art egocentric RGB performance, with SR and SPL scores of 81.1% and 68.3%, and 67.5% and 52.0%, respectively, while using significantly fewer parameters and less training data than prior scaling-based approaches. More importantly, real-world robotic deployments confirm that StereoNav substantially improves navigation reliability in complex, unstructured environments. Project page: this https URL.
>
---
#### [new 038] TinySDP: Real Time Semidefinite Optimization for Certifiable and Agile Edge Robotics
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动规划任务，解决嵌入式系统中实时求解非凸约束的问题。提出TinySDP，结合半定规划与ADMM算法，实现快速、安全的路径规划。**

- **链接: [https://arxiv.org/pdf/2605.13748](https://arxiv.org/pdf/2605.13748)**

> **作者:** Ishaan Mahajan; Jon Arrizabalaga; Andrea Grillo; Fausto Vega; James Anderson; Zachary Manchester; Brian Plancher
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026. 11 pages, 5 figures, 2 tables. Project website: this https URL
>
> **摘要:** Semidefinite programming (SDP) provides a principled framework for convex relaxations of nonconvex geometric constraints in motion planning, yet existing solvers are too computationally expensive for real-time control, particularly on resource-constrained embedded systems. To address this gap, we introduce TinySDP, the first semidefinite programming solver designed for embedded systems, enabling real-time model-predictive control (MPC) on microcontrollers for problems with nonconvex obstacle constraints. Our approach integrates positive-semidefinite cone projections into a cached-Riccati-based ADMM solver, leveraging computational structure for embedded tractability. We pair this solver with an a posteriori rank-1 certificate that converts relaxed solutions into explicit geometric guarantees at each timestep. On challenging benchmarks, e.g., cul-de-sac and dynamic obstacle avoidance scenarios that induce failures in local methods, TinySDP achieves collision-free navigation with up to 73% shorter paths than state-of-the-art baselines. We validate our approach on a Crazyflie quadrotor, demonstrating that semidefinite constraints can be enforced at real-time rates for agile embedded robotics.
>
---
#### [new 039] A Five-Layer MLOps Architecture for Connected Automated Driving
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自动驾驶领域，旨在解决ADS持续安全与性能保障问题，提出五层MLOps架构以支持集体学习和边缘案例检测。**

- **链接: [https://arxiv.org/pdf/2605.12719](https://arxiv.org/pdf/2605.12719)**

> **作者:** Bastian Lampe; Lutz Eckstein
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** The continual assurance of safety and performance of automated driving systems (ADSs) poses significant challenges. ADSs operate in complex, dynamic, open-world environments allowing a wide range of scenarios, including ones that are rare or not foreseen during initial development. While the incorporation of artificial intelligence (AI) and machine learning (ML) technology allows ADSs to learn from data gathered during operation and thus enables them to adapt over time, these approaches come with their own challenges. A key advantage of ADSs compared to human drivers is their greater ability to gather data collectively across a fleet of vehicles, or even across multiple fleets operated by different entities, and to learn from this data collectively. Vehicles can share and combine their data to identify additional learning opportunities otherwise missed by individual vehicles. This creates new opportunities to tackle the challenges of continual assurance of safety and performance, but requires the implementation of architectures that leverage the collective learning potential. Based on established MLOps principles and existing work in the field of connected automated driving, this paper presents a five-layer architecture for collective learning-enabled MLOps processes for ADSs. The goal of this architecture is to provide a conceptual blueprint for the design and implementation of MLOps processes by fleet operators and other relevant stakeholders. The paper describes the main responsibilities of each layer, their interactions, and how multi-level self-assessments enabled by the architecture can support the detection and reduction of edge cases including black swan events.
>
---
#### [new 040] Object Manipulation of the Variable Topology Truss system
- **分类: cs.RO**

- **简介: 该论文研究VTT系统的物体操作问题，提出一种混合控制框架，解决位置与力的协同控制难题，通过实验验证了操作的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.13086](https://arxiv.org/pdf/2605.13086)**

> **作者:** Andrew Jang-Ho Bae; Myeongjin Choi; Haorui Li; Mark Yim; TaeWon Seo
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** This paper presents an object manipulation strategy for the Variable Topology Truss (VTT) system, a truss robot that comprises actuated truss members connected by passive spherical joints. Although truss robots were originally proposed as rapidly deployable manipulators, manipulation strategy has not been studied thoroughly. To enable manipulation, we introduce a hybrid control framework that regulates position and force concurrently without explicit decoupling. At the actuator level, each member employs a sensor-based force feedback controller to generate the desired axial forces despite high actuator friction. At the task level, the forces applied at the end-effector nodes are produced by computing the required member forces using a static model of the VTT. We evaluate force-tracking performance through experiments on both a single member module and the full VTT system. Finally, we demonstrate object manipulation using two representative configurations and quantitatively assess combined position and force tracking performance. Experimental results confirm that the proposed approach enables consistent and reliable object manipulation with the VTT system.
>
---
#### [new 041] COSMIC: Concurrent Optimization of Structure, Material, and Integrated Control for robotic systems
- **分类: cs.RO**

- **简介: 该论文属于机器人协同设计任务，旨在解决结构、材料与控制分离导致的优化不足问题。通过构建联合优化框架，同时优化三者以提升机器人性能。**

- **链接: [https://arxiv.org/pdf/2605.12654](https://arxiv.org/pdf/2605.12654)**

> **作者:** Qinsong Guo; Liwei Wang
>
> **摘要:** Replicating and surpassing the autonomy of natural organisms remains a long-standing goal in robotics. Yet most robotic systems have their structure, materials, and control designed separately, in sharp contrast to the co-evolution in nature. This separation often leads to suboptimal designs, and we still have a limited understanding of the individual and collective contributions of these design entities. In this work, we propose a gradient-based co-design framework that simultaneously optimizes the topology, material distribution, and control policy of a truss-lattice robot. The framework embeds mixed-type topological and material variables into a continuous design space and integrates a neural network controller within a differentiable simulator, capturing their interactions and enabling efficient gradient calculation via automatic differentiation. Furthermore, we develop a constrained optimization to navigate the highly non-convex design landscape and jointly optimize all design entities. Case studies demonstrate that the proposed framework consistently discovers diverse locomotion strategies that outperform baselines obtained through separated design. The framework is also flexible to accommodate different functional requirements and boundary conditions. Using this framework, we further extract design insights that reveal the individual and collective effects of different entities on robotic performance. The proposed framework provides a computational foundation for the autonomous co-design of robotic systems, capable of reconfiguration, locomotion, and other complex autonomous behaviors.
>
---
#### [new 042] Lifelong Learning in Vision-Language Models: Enhanced EWC with Cross-Modal Knowledge Retention
- **分类: cs.RO**

- **简介: 该论文属于多模态AI的持续学习任务，旨在解决模型在连续学习新任务时的灾难性遗忘问题。通过改进EWC方法，提升跨模态知识保留能力。**

- **链接: [https://arxiv.org/pdf/2605.12789](https://arxiv.org/pdf/2605.12789)**

> **作者:** Hamza Ahmed Durrani; Rafay Suleman Durrani
>
> **备注:** 8 pages, 5 figures, 1 table. Applications in autonomous driving, intelligent robotic assistants, and adaptive robotics systems
>
> **摘要:** Large language-vision models (LVLMs) such as CLIP, Flamingo, and BLIP have revolutionized AI by enabling understanding across textual and visual modalities. These models excel at tasks like image captioning, visual question answering, and cross-modal retrieval. However, they face catastrophic forgetting when learning new tasks sequentially, particularly challenging in multi-modal settings where preserving cross-modal alignments adds complexity to the learning process. This paper presents a comprehensive continual learning framework for LVLMs that combines enhanced Elastic Weight Consolidation (EWC) with parameter-efficient fine-tuning techniques. We integrate multi-modal Fisher Information Matrix calculation, consistency preservation across modalities, and adaptive regularization that considers dependencies across visual and textual encoders. The framework achieves a 78% reduction in forgetting rates relative to naive sequential training approaches through extensive evaluation testing. The framework also preserves alignment between modalities during sequential learning with only 15% additional computational cost. This work advances the state of the art in lifelong learning for multi-modal AI systems, with direct applications to autonomous driving, intelligent robotic assistants, and adaptive robotic systems that must continuously learn in dynamic real-world environments.
>
---
#### [new 043] RoboEvolve: Co-Evolving Planner-Simulator for Robotic Manipulation with Limited Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboEvolve框架，解决机器人操作中数据稀缺问题。通过联合视觉语言模型与视频生成模型，实现高效数据生成与策略优化。**

- **链接: [https://arxiv.org/pdf/2605.13775](https://arxiv.org/pdf/2605.13775)**

> **作者:** Harold Haodong Chen; Sirui Chen; Yingjie Xu; Wenhang Ge; Ying-Cong Chen
>
> **备注:** On-going work
>
> **摘要:** The scalability of robotic manipulation is fundamentally bottlenecked by the scarcity of task-aligned physical interaction data. While vision-language models (VLMs) and video generation models (VGMs) hold promise for autonomous data synthesis, they suffer from semantic-spatial misalignment and physical hallucinations, respectively. To bridge this gap, we introduce RoboEvolve, a novel framework that couples a VLM planner and a VGM simulator into a mutually reinforcing co-evolutionary loop. Operating purely on unlabeled seed images, RoboEvolve leverages a cognitive-inspired dual-phase mechanism: (i) daytime exploration fosters physically grounded behavioral discovery through a semantic-controlled multi-granular reward, and (ii) nighttime consolidation mines "near-miss" failures to stabilize policy optimization. Guided by an autonomous progressive curriculum, the system naturally scales from simple atomic actions to complex tasks. Extensive experiments demonstrate that RoboEvolve (I) achieves superior effectiveness, elevating base planners by 30 absolute points and amplifying simulator success by 48% on average; (II) exhibits extreme data efficiency, surpassing fully supervised baselines with merely 500 unlabeled seeds--a 50x reduction; and (III) demonstrates robust continual learning without catastrophic forgetting.
>
---
#### [new 044] RotVLA: Rotational Latent Action for Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RotVLA，解决VLA模型中动作表示不足的问题。通过连续旋转潜行动作表示，提升动作生成效果，适用于机器人控制任务。**

- **链接: [https://arxiv.org/pdf/2605.13403](https://arxiv.org/pdf/2605.13403)**

> **作者:** Qiwei Li; Xicheng Gong; Xinghang Li; Peiyan Li; Quanyun Zhou; Hangjun Ye; Jiahuan Zhou; Yadong Mu
>
> **摘要:** Latent Action Models (LAMs) have emerged as an effective paradigm for handling heterogeneous datasets during Vision-Language-Action (VLA) model pretraining, offering a unified action space across embodiments. However, existing LAMs often rely on discrete quantization encode and decode pipelines, which can lead to trivial frame reconstruction behavior, limited representational capacity, and a lack of physically meaningful structure. We introduce RotVLA, a VLA framework built on a continuous rotational latent action representation. Latent actions are modeled as elements of SO(n), providing continuity, compositionality, and structured geometry aligned with real-world action dynamics. A triplet frame learning framework further enforces meaningful temporal dynamics while avoiding degeneration. RotVLA consists of a VLM backbone and a flow-matching action head, pretrained on large-scale cross-embodiment robotic datasets and human videos with latent-action supervision. For downstream robot control, the flow-matching head is extended into a unified action expert that jointly denoises latent and robot actions. Here, latent actions serve as a latent planner, providing high-level guidance that conditions action generation. With only 1.7B parameters and 1700+ hours of pretraining data, RotVLA achieves 98.2% on LIBERO and 89.6% / 88.5% on RoboTwin2.0 under clean and randomized settings, respectively. It also demonstrates strong real-world performance on manipulation tasks, consistently outperforming existing VLA models.
>
---
#### [new 045] Asymptotically Optimal Ergodic Coverage on Generalized Motion Fields
- **分类: cs.RO**

- **简介: 该论文属于机器人自主探索任务，解决动态环境中覆盖问题。提出基于遍历性的覆盖方法，适应流场变化，确保有效数据采集。**

- **链接: [https://arxiv.org/pdf/2605.13442](https://arxiv.org/pdf/2605.13442)**

> **作者:** Christian Hughes; Yilang Liu; Yanis Lahrach; Julia Engdahl; Houston Warren; Darrick Lee; Fabio Ramos; Travis Miles; Ian Abraham
>
> **备注:** 13 pages, 9 figures, 6 tables, Robotics: Science and Systems 2026
>
> **摘要:** Autonomous robotic exploration in remote and extreme environments allows scientists to model complex transport phenomena and collective behaviors described by continuously deforming flow fields. Although these environments are naturally modeled as time-varying domains, most adaptive exploration methods assume static environments and fail to provide adequate coverage or satisfy any formal guarantees. This is especially the case in oceanography where autonomous underwater systems (UxS) have highly restrictive compute and payload requirements that necessitate path planning methods that yield robust data collection strategies in open-loop and underactuated settings. In this work, to address the aforementioned issues, we propose to formulate adaptive search as an ergodic coverage problem and investigate certifying coverage in the ergodic sense over evolving domains with flow-induced dynamics. We expand upon recent work demonstrating maximum mean discrepancy (MMD) as a functional ergodic metric, and derive a flow-adaptive formulation that explicitly accounts for domain evolution within the coverage objective. We show that this approach preserves ergodic coverage guarantees in ambient flows and enables effective exploration in under-actuated, and even open-loop planning settings by integrating environment dynamics. Experiments validate that our method generalizes to diverse spatiotemporal processes including ocean exploration, and tracking human and cattle movement. Physical experiments on aerial and legged robotic platforms validate our ability to obtain ergodic coverage in non-convex, flow-restricted environments while respecting robot dynamics.
>
---
#### [new 046] BiPneu: Design and Control of a Bipolar-Pressure Pneumatic System for Soft Robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决双压调节难题。提出BiPneu系统与DM-SMC控制器，实现高精度、快速响应的压力控制。**

- **链接: [https://arxiv.org/pdf/2605.12804](https://arxiv.org/pdf/2605.12804)**

> **作者:** Yu Mei; Xinyu Zhou; Vedant Naik; Alan Gao; Xiaobo Tan
>
> **备注:** Accepted by IEEE/ASME TMECH
>
> **摘要:** Positive-negative pressure regulation is critical to soft robotic actuators, enabling large motion ranges and versatile actuation modes. However, achieving high-performance regulation across both pressure polarities remains challenging due to asymmetric inflation-deflation dynamics, valve nonlinearities, and switching-induced flow disturbances. This paper presents BiPneu, a scalable and cost-efficient multi-channel bipolar-pressure pneumatic system for soft robots that enables wide-range, accurate, and responsive pressure regulation while providing seamless compatibility with high-level software ecosystems. A dual-mode sliding-mode controller (DM-SMC) with hysteresis-supervised mode selection is proposed based on a hybrid electro-pneumatic model. Extensive simulation and experiments demonstrate the superior performance of DM-SMC in tracking step and sinusoidal pressure references compared with both advanced model predictive controllers and well-tuned PID controllers. Experimental results show average absolute errors of 1.44 kPa in multi-step tests and 4.23 kPa in sinusoidal tracking, corresponding to reductions of 11.9% and 35.6% relative to PID control, along with improved control effort, valve switching rate, and transient response. Robustness of DM-SMC is further verified on a bellow actuator with pressure-dependent volume. Finally, BiPneu's capability is demonstrated via two soft robotic examples, quick ball-maneuvering with a soft parallel manipulator and real-time finite element method (FEM)-based teleoperation of a soft bellows actuator.
>
---
#### [new 047] 3D RL-DWA: A Hybrid Reinforcement Learning and Dynamic Window Approach for Goal-Directed Local Navigation in Multi-DoF Robots
- **分类: cs.RO**

- **简介: 该论文属于多自由度机器人3D局部导航任务，旨在解决复杂环境下的路径规划与变形控制问题。通过融合强化学习与动态窗口方法，提升导航效率与适应性。**

- **链接: [https://arxiv.org/pdf/2605.12689](https://arxiv.org/pdf/2605.12689)**

> **作者:** Chiara Castellani; Enrico Turco; Domenico Prattichizzo
>
> **备注:** Accepted to IEEE/ASME International Conference on Advanced Intelligent Mechatronics, AIM2026
>
> **摘要:** In this paper, we present a novel hybrid approach that combines Reinforcement Learning (RL) with Dynamic Window Approach (DWA) for adaptive 3D local navigation of high-degree-of-freedom robotic systems. Our method leverages sparse point cloud data to dynamically adjust both the motion and the shape of a deformable microrobot, enabling the system to navigate toward a goal in complex, constrained environments while maximizing the occupied volume. We evaluate our framework in a simulated vascular network. Experimental results, based on 1080 trials, indicate that integrating RL with a DWA-based local planner significantly enhances both deformation and navigation capabilities compared to a pure RL and a model-based methods. In particular, the proposed autonomous controller consistently achieves high deformation and near-perfect path completion during training and maintains robust performance in unseen scenarios. These findings highlight the potential of hybrid planning strategies for efficient and adaptive 3D navigation under sparse sensory conditions.
>
---
#### [new 048] BlockVLA: Accelerating Autoregressive VLA via Block Diffusion Finetuning
- **分类: cs.RO**

- **简介: 该论文提出BlockVLA，解决AR-VLA模型推理慢和误差累积问题，通过块扩散机制提升机器人控制效率。**

- **链接: [https://arxiv.org/pdf/2605.13382](https://arxiv.org/pdf/2605.13382)**

> **作者:** Ruiheng Wang; Shuanghao Bai; Haoran Zhang; Badong Chen; Xiangyu Xu
>
> **摘要:** While autoregressive (AR) Vision-Language-Action (VLA) models have demonstrated formidable reasoning capabilities in robotic tasks, their sequential decoding process often incurs high inference latency and may amplify error accumulation during long-horizon execution. Discrete Diffusion Language Models (dLLMs) provide a promising alternative through parallel token refinement, but their practical deployment in robotics remains limited by repeated denoising function evaluations (NFEs) and the difficulty of directly applying standard KV caching to bidirectional iterative decoding. To bridge these paradigms, we propose BlockVLA, a framework that adapts pretrained AR backbones into an efficient discrete diffusion policy through a block diffusion paradigm. BlockVLA maintains autoregressive dependencies at the block level while enabling parallel denoising within each block, thereby combining global causal coherence with local parallel generation. This design enables prefix KV-cache reuse across completed blocks, reduces the effective cost of iterative denoising, and provides a smoother transition from AR pretraining to diffusion-based policy fine-tuning. We conduct extensive evaluations on the LIBERO and SimplerEnv benchmarks. Experimental results demonstrate that our BlockVLA achieves a 3.3$\times$ inference acceleration over standard discrete diffusion baselines. Furthermore, our model exhibits superior training efficiency, with success rates converging substantially faster than baselines, a gain that is particularly pronounced in complex, long-horizon tasks, where BlockVLA achieves significant performance gains in the early stages of training. This work establishes Block Diffusion as a robust bridge between large-scale pretrained AR models and efficient, high-frequency real-time robotic control.
>
---
#### [new 049] What to Ignore, What to React: Visually Robust RL Fine-Tuning of VLA Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决视觉变化下的强化学习微调问题。提出PAIR-VLA框架，通过引入不变性和敏感性目标，提升模型对视觉扰动的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.13105](https://arxiv.org/pdf/2605.13105)**

> **作者:** Yuanfang Peng; Jingjing Fu; Chuheng Zhang; Li Zhao; Jiang Bian; Mingyu Liu; Ling Zhang; Jun Zhang; Rui Wang
>
> **摘要:** Reinforcement learning (RL) fine-tuning has shown promise for Vision-Language-Action (VLA) models in robotic manipulation, but deployment-time visual shifts pose practical challenges. A key difficulty is that standard task rewards supervise task success, but offer limited guidance on whether a visual change is task-irrelevant or changes the behavior required for manipulation. We propose PAIR-VLA (Paired Action Invariance & Sensitivity for Visually Robust VLA), an RL fine-tuning framework to address this difficulty by adding two auxiliary objectives over paired visual variants during PPO optimization: an invariance term that reduces the discrepancy between action distributions for a task-preserving pair (e.g., different distractors), and a sensitivity objective that encourages separable action distributions for a task-altering pair (e.g., target object in a different pose). Together, these objectives turn visual variants from mere observation diversity into behavior-level guidance on policy responses during RL fine-tuning. We evaluate on ManiSkill3 across two representative VLA architectures, OpenVLA and $\pi_{0.5}$, under diverse out-of-distribution visual shifts including unseen distractors, texture changes, target object pose variation, viewpoint shifts, and lighting changes. Our method consistently improves over standard PPO, achieving average improvements of 16.62% on $\pi_{0.5}$ and 9.10% on OpenVLA. Notably, ablations further show generalization across visual shifts: invariance guidance learned from distractor and texture variants transfers to target-pose and lighting shifts, while adding sensitivity guidance on target-pose variants further improves robustness to nuisance shifts, highlighting the broader transferability of behavior-level RL guidance.
>
---
#### [new 050] SID: Sliding into Distribution for Robust Few-Demonstration Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决少样本下泛化能力不足的问题。提出SID框架，通过学习物体中心运动场，提升在分布外情况下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.13428](https://arxiv.org/pdf/2605.13428)**

> **作者:** Yicheng Ma; Wei Yu; Zhian Su; Xidan Zhang; Huixu Dong
>
> **备注:** 20 pages, 14 figures. Project website: this https URL
>
> **摘要:** Generalizing robotic manipulation across object poses, viewpoints, and dynamic disturbances is difficult, especially with only a few demonstrations. End-to-end visuomotor policies are expressive but data-hungry, while planning and optimization satisfy explicit constraints but do not directly capture the interaction strategies demonstrated by humans. We propose Sliding into Distribution (SID), a structured framework that learns an object-centric motion field from canonicalized demonstrations to iteratively slide the system toward the demonstrated manifold and into the reliable operating region of a lightweight egocentric execution policy, mitigating out-of-distribution (OOD) execution. The motion field provides large corrective motions when far from the demonstration manifold and naturally vanishes near convergence, enabling robust reaching under substantial pose and viewpoint shifts. Within the reached regime, an egocentric policy trained with conditioned flow matching performs task-specific manipulation, supported by kinematically consistent point-cloud reprojection augmentation that preserves action-observation consistency. Across six real-world tasks, SID achieves approximately 90% success under OOD initializations with only two demonstrations, with under a 10% drop under distractors and external disturbances. Overall, SID provides a new paradigm for few-shot manipulation: explicitly managing distribution shift via online distribution recovery.
>
---
#### [new 051] Adaptive Smooth Tchebycheff Attention for Multi-Objective Policy Optimization
- **分类: cs.RO; cs.AI; cs.LG; eess.SY; math.OC**

- **简介: 该论文属于多目标强化学习任务，旨在解决机器人领域中非凸帕累托前沿的优化问题。提出自适应平滑Tchebycheff方法，动态调节优化曲率，提升稳定性与非凸解的获取能力。**

- **链接: [https://arxiv.org/pdf/2605.12771](https://arxiv.org/pdf/2605.12771)**

> **作者:** Alejandro Murillo-Gonzalez; Mahmoud Ali; Lantao Liu
>
> **备注:** To appear in the Proceedings of Robotics: Science and Systems (RSS) 2026
>
> **摘要:** Multi-objective reinforcement learning in robotic domains requires balancing complex, non-convex trade-offs between conflicting objectives. While linear scalarization methods provide stability, they are theoretically incapable of recovering solutions within non-convex regions of the Pareto front. Conversely, static non-linear scalarizations (e.g., Tchebycheff) can theoretically access these regions but often suffer from severe gradient variance and optimization instability in deep RL. In this work, we propose an Adaptive Smooth Tchebycheff framework that resolves this tension by dynamically modulating the curvature of the optimization landscape. We introduce a novel conflict-driven controller that regulates the optimization smoothness based on real-time gradient interference. This allows the agent to anneal toward precise, non-convex scalarization when objectives align, while elastically reverting to stable, smooth approximations when destructive gradient conflicts emerge. We validate our approach on a challenging robotic stealth visual search task -- a proxy for monitoring of protected/fragile ecosystems -- where an agent must balance search, exposure/interference minimization and exploration speed. Extensive ablations confirm that our conflict-aware adaptation enables the robust discovery of Pareto-optimal policies in non-convex regions inaccessible to linear baselines and unstable for static non-linear methods. Website: this https URL
>
---
#### [new 052] Belief-Space Residual Risk for Automated Driving under Localization Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，解决定位不确定性下的残余风险计算问题。通过将自车位姿不确定性建模为高斯分布，扩展了残余风险的定义，并改进了碰撞概率计算方法。**

- **链接: [https://arxiv.org/pdf/2605.12710](https://arxiv.org/pdf/2605.12710)**

> **作者:** Nijinshan Karunainayagam; Nils Gehrke; Frank Diermeyer
>
> **备注:** 7 Pages, this work has been accepted for publication in IEEE Intelligent Transportation Systems (ITSC) 2026. The final published version will be available via IEEE Xplore
>
> **摘要:** Residual risk metrics have recently been introduced to assess the safety implications of automated driving systems. Existing approaches typically assume a deterministic ego pose and concentrate mainly on perception errors related to surrounding objects and latency effects. In practice, however, automated vehicles operate under considerable localization uncertainty, especially in complex urban settings and in adverse weather conditions. This work extends the spatial residual risk formulation to the belief space by explicitly modeling ego pose uncertainty as a Gaussian distribution. Residual risk is reformulated as the expected degradation-induced risk over the ego pose belief distribution. Within a particle-based risk estimation framework, localization uncertainty is incorporated into the computation of collision probabilities through covariance fusion of ego and object uncertainties.
>
---
#### [new 053] Learning Responsibility-Attributed Adversarial Scenarios for Testing Autonomous Vehicles
- **分类: cs.RO; cs.SE; eess.SY**

- **简介: 该论文属于自动驾驶安全测试任务，旨在区分系统缺陷与交通冲突导致的故障。提出CARS框架，结合责任归属生成可解释的碰撞场景，提升安全验证效果。**

- **链接: [https://arxiv.org/pdf/2605.13751](https://arxiv.org/pdf/2605.13751)**

> **作者:** Yizhuo Xiao; Haotian Yan; Ying Wang; Zhongpan Zhu; Yuxin Zhang; Xintao Yan; Mustafa Suphi Erden; Cheng Wang
>
> **摘要:** Establishing trustworthy safety assurance for autonomous driving systems (ADSs) requires evidence that failures arise from avoidable system deficiencies rather than unavoidable traffic conflicts. Current adversarial simulation methods can efficiently expose collisions, but generally lack mechanisms to distinguish these fundamentally different failure modes. Here we present CARS (Context-Aware, Responsibility-attributed Scenario generation), a framework that integrates responsibility attribution directly into adversarial scenario generation. CARS combines context-aware adversary selection with a generative adversarial policy optimized in closed-loop simulation to construct collision scenarios that are both physically feasible and diagnostically attributable. Across benchmark datasets spanning heterogeneous national traffic environments, CARS consistently discovers feasible collision scenarios with high attribution rates under multiple regulation-prescribed careful and competent driver models. By coupling adversarial generation with normative responsibility assessment, CARS moves simulation testing beyond collision discovery toward the construction of interpretable, regulation-aligned safety evidence for scalable ADS validation.
>
---
#### [new 054] Manipulation Planning for Construction Activities with Repetitive Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决重复性施工动作的技能获取问题。通过VR演示和螺钉运动模型，生成精确的机械臂运动规划，实现高效施工。**

- **链接: [https://arxiv.org/pdf/2605.13754](https://arxiv.org/pdf/2605.13754)**

> **作者:** Wangyi Liu; Dasharadhan Mahalingam; Fanru Gao; Ci-Jyun Liang; Nilanjan Chakraborty
>
> **摘要:** In this paper, we study the problem of manipulation skill acquisition for performing construction activities consisting of repetitive tasks (e.g., building a wall or installing ceiling tiles). Our approach involves setting up a simulated construction activity in a Virtual Reality (VR) environment, where the user can provide demonstrations of the object manipulation skills needed to perform the construction activity. We then exploit the screw geometry of motion to approximate the demonstrated motion as a sequence of constant screw motions. For performing the construction activity, we generate the sequence of manipulation task instances and then compute the joint space motion plan corresponding to each instance using Screw Linear Interpolation (ScLERP) and Resolved Motion Rate Control (RMRC). We evaluate our framework by executing two representative construction tasks: constructing brick walls and installing multiple ceiling tiles. Each task is performed using only a single demonstration, a pick-and-place action for the bricks, and a single ceiling tile installation. Our experiments with a 7-DoF robot in both simulation and hardware demonstrate that the approach generalizes robustly to arbitrarily long construction activities that involve repetitive motions and demand precision, even when provided with just one demonstration. For instance, we can construct walls of arbitrary layout and length by leveraging a single demonstration of placing one brick on top of another.
>
---
#### [new 055] MoCCA: A Movable Circle Probability of Collision Approximation
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的碰撞风险评估任务，解决传感器不确定性下的碰撞概率估算问题。提出MoCCA算法，用优化单圆模型提高计算效率并降低保守性。**

- **链接: [https://arxiv.org/pdf/2605.13125](https://arxiv.org/pdf/2605.13125)**

> **作者:** Tobias Kern; Christian Birkner
>
> **备注:** Accepted at ITSC 2026
>
> **摘要:** In automated driving, crash mitigation is crucial to ensure passenger safety. Accurate avoidance requires precise knowledge of the object's position and orientation. However, sensor noise and occlusions often result in tracking and prediction uncertainties. To account for these uncertainties, estimating the Probability of Collision (POC) is a critical requirement. While Monte Carlo sampling is a common estimation technique, its high computational demand and stochastic nature often render it unsuitable for real-time applications. Analytical POC calculations are simplified by approximating vehicle geometries using circular bounds. While multi-circle approximations offer higher fidelity than a single circumscribed circle, they significantly increase computational complexity. This paper proposes a shape approximation algorithm, MoCCA, which utilizes a single circle for each vehicle, optimized to minimize the relative distance between them. MoCCA maintains a computational efficiency comparable to standard single-circle techniques while reducing over-conservatism. To address the potential underestimation of POC inherent in partial coverage, we establish an upper bound for the approximation error, demonstrating that it depends primarily on inter-vehicle distance and orientation variance. Furthermore, we introduce a safety distance margin that can be calibrated solely based on orientation variance.
>
---
#### [new 056] Relative Pose-Velocity Estimation Using Dual IMU Measurements and Relative Position Sensing
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于相对姿态-速度估计任务，解决车辆与运动目标间相对状态估计问题，通过双IMU和相对位置测量设计观测器和滤波器，确保估计稳定性和准确性。**

- **链接: [https://arxiv.org/pdf/2605.13031](https://arxiv.org/pdf/2605.13031)**

> **作者:** Alessandro Melis; Tarek Bouazza; Soulaimane Berkane; Tarek Hamel
>
> **摘要:** This paper addresses the problem of estimating the relative pose (position and orientation) and velocity of a vehicle with respect to a moving target, where both are equipped with Inertial Measurement Units (IMUs), assuming the availability of relative position or bearing measurements. The body-target relative dynamics are formulated on $\mathbf{SE}_2(3)$ and recast into a linear time-varying (LTV) model in the ambient space $\mathbb{R}^{15}$, on which a deterministic Riccati observer is designed. We analyze the uniform observability (UO) conditions required to guarantee global exponential convergence of the estimation error in the ambient space for both measurement cases. In the case of relative position measurements, UO requires only a persistence-of-excitation condition on the target acceleration, whereas for bearing measurements, additional conditions are required. Building on this, a nonlinear complementary filter on $\mathbf{SO}(3)$ is designed to provide a smooth estimate of the orientation component of the state with almost global asymptotic stability. Finally, simulation results are provided to validate the proposed solution.
>
---
#### [new 057] Beyond VMAF: Towards Application-Specific Metrics for Teleoperation Video
- **分类: cs.HC; cs.RO**

- **简介: 论文研究如何改进视频质量评估指标以适应远程驾驶场景。针对 teleoperation 中视频质量对安全与任务表现的影响，通过调整 VMAF 模型提升其预测准确性，但发现仍存在异常情况。**

- **链接: [https://arxiv.org/pdf/2605.13525](https://arxiv.org/pdf/2605.13525)**

> **作者:** Ines Trautmannsheimer; Richard Grauberger; Frank Diermeyer
>
> **备注:** Preprint ITSC 2026
>
> **摘要:** Automated driving has made remarkable progress, yet situations still arise where human intervention is necessary. Teleoperation provides a scalable solution to address such cases, enabling remote operators to support vehicles without being physically present. In this context, video transmission forms the operator's primary source of situational awareness, making video quality a decisive factor for both safety and task performance. In an online study, participants rated compressed video sequences from the Zenseact Dataset and provided subjective quality ratings. These ratings were then used to retrain the Video Multi-Method Assessment Fusion (VMAF) model, yielding an adapted variant tailored to teleoperation. The retrained model demonstrated improved alignment with human ratings compared to the original 4K VMAF. In particular, RMSE decreased from 10.36 to 8.83, and MAD from 8.71 to 6.38, corresponding to improvements of 15% and 27%, respectively. These results highlight that incorporating domain-specific data can enhance the predictive power of established quality metrics in safety-critical applications. At the same time, Outlier cases emerged in which videos received high objective scores despite noticeable degradations in regions critical for the driving task.
>
---
#### [new 058] OmniLiDAR: A Unified Diffusion Framework for Multi-Domain 3D LiDAR Generation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OmniLiDAR，解决多领域3D LiDAR生成问题，通过统一框架实现跨域可控生成。**

- **链接: [https://arxiv.org/pdf/2605.13815](https://arxiv.org/pdf/2605.13815)**

> **作者:** Youquan Liu; Weidong Yang; Ao Liang; Xiang Xu; Lingdong Kong; Yang Wu; Dekai Zhu; Xin Li; Runnan Chen; Ben Fei; Tongliang Liu; Wanli Ouyang
>
> **备注:** Preprint; 12 pages, 7 figures, 10 tables
>
> **摘要:** LiDAR scene generation is increasingly important for scalable simulation and synthetic data creation, especially under diverse sensing conditions that are costly to capture at scale. Typically, diffusion-based LiDAR generators are developed under single-domain settings, requiring separate models for different datasets or sensing conditions and hindering unified, controllable synthesis under heterogeneous distribution shifts. To this end, we present OmniLiDAR, a unified text-conditioned diffusion framework that generates LiDAR scans in a shared range-image representation across eight representative domains spanning three shift types: adverse weather, sensor-configuration changes (e.g., reduced beams), and cross-platform acquisition (vehicle, drone, and quadruped). To enable training a single model over heterogeneous domains without isolating optimization by domain, we introduce a Cross-Domain Training Strategy (CDTS) that mixes domains within each mini-batch and leverages conditioning to steer generation. We further propose Cross-Domain Feature Modeling (CDFM), which captures directional dependencies along azimuth and elevation axes to reflect the anisotropic scanning structure of range images, and Domain-Adaptive Feature Scaling (DAFS) as a lightweight modulation to account for structured domain-dependent feature shifts during denoising. In the absence of a public consolidated benchmark, we construct an 8-domain dataset by combining real-world scans with physically based weather simulation and systematic beam reduction while following official splits. Extensive experiments demonstrate strong generation fidelity and consistent gains in downstream use cases, including generative data augmentation for LiDAR semantic segmentation and 3D object detection, as well as robustness evaluation under corruptions, with consistent benefits in limited-label regimes.
>
---
#### [new 059] D-VLA: A High-Concurrency Distributed Asynchronous Reinforcement Learning Framework for Vision-Language-Action Models
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出D-VLA框架，解决大规模视觉-语言-动作模型在分布式强化学习中的高并发与资源冲突问题，通过架构优化提升训练效率和稳定性。**

- **链接: [https://arxiv.org/pdf/2605.13276](https://arxiv.org/pdf/2605.13276)**

> **作者:** Yucheng Guo; Yongjian Guo; Zhong Guan; Wen Huang; Haoran Sun; Haodong Yue; Xiaolong Xiang; Shuai Di; Zhen Sun; Luqiao Wang; Junwu Xiong; Yicheng Gong
>
> **摘要:** The rapid evolution of Embodied AI has enabled Vision-Language-Action (VLA) models to excel in multimodal perception and task execution. However, applying Reinforcement Learning (RL) to these massive models in large-scale distributed environments faces severe systemic bottlenecks, primarily due to the resource conflict between high-fidelity physical simulation and the intensive VRAM/bandwidth demands of deep learning. This conflict often leaves overall throughput constrained by execution-phase inefficiencies. To address these challenges, we propose D-VLA, a high-concurrency, low-latency distributed RL framework for large-scale embodied foundation models. D-VLA introduces "Plane Decoupling," physically isolating high-frequency training data from low-frequency weight control to eliminate interference between simulation and optimization. We further design a four-thread asynchronous "Swimlane" pipeline, enabling full parallel overlap of sampling, inference, gradient computation, and parameter distribution. Additionally, a dual-pool VRAM management model and topology-aware replication resolve memory fragmentation and optimize communication efficiency. Experiments on benchmarks like LIBERO show that D-VLA significantly outperforms mainstream RL frameworks in throughput and sampling efficiency for billion-parameter VLA models. In trillion-parameter scalability tests, our framework maintains exceptional stability and linear speedup, providing a robust system for high-performance general-purpose embodied agents.
>
---
#### [new 060] Trajectory-Level Data Augmentation for Offline Reinforcement Learning
- **分类: cs.LG; cs.RO; stat.ML**

- **简介: 该论文属于强化学习领域，针对离线强化学习中数据量少、质量低的问题，提出一种基于轨迹的数据增强方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.13401](https://arxiv.org/pdf/2605.13401)**

> **作者:** Tobias Schmähling; Matthias Burkhardt; Tobias Windisch
>
> **备注:** 26 pages, 25 figures, Accepted at ICML 2026
>
> **摘要:** We propose a data augmentation method for offline reinforcement learning, motivated by active positioning problems. Particularly, our approach enables the training of off-policy models from a limited number of suboptimal trajectories. We introduce a trajectory-based augmentation technique that exploits task structure and the geometric relationship between rewards, value functions, and mathematical properties of logging policies. During data collection, our augmentation supports suboptimal logging policies, leading to higher data quality and improved offline reinforcement learning performance. We provide theoretical justification for these strategies and validate them empirically across positioning tasks of varying dimensionality and under partial observability.
>
---
#### [new 061] Scale-Gest: Scalable Model-Space Synthesis and Runtime Selection for On-Device Gesture Detection
- **分类: cs.CV; cs.AI; cs.HC; cs.RO; eess.IV**

- **简介: 该论文属于移动端手势检测任务，解决设备资源受限下的实时性与能效问题。提出Scale-Gest框架，通过动态选择模型和优化检测区域，提升性能并降低能耗。**

- **链接: [https://arxiv.org/pdf/2605.12506](https://arxiv.org/pdf/2605.12506)**

> **作者:** Abdul Basit; Saim Rehman; Muhammad Shafique
>
> **备注:** 7 pages, 11 figures, Accepted to DAC 2026
>
> **摘要:** Realizing on-device ML-based gesture detection under tight real-time performance, energy and memory constraints is challenging, especially when considering mobile devices with varying battery-power levels. Existing EdgeAI deployments typically rely on a single fixed detector, limiting optimization opportunities. We present Scale-Gest, a novel run-time adaptive gesture detection framework that expands the detector space into a dense family of tiny-YOLO architectures. We introduce multiple novel device-calibrated ACE (Accuracy-Complexity-Energy) profiles by analyzing different model-resolution-stride operating points. A lightweight run-time controller selects an appropriate ACE mode under user-defined and battery constraints, while a motion-aware hand-gesture-tracking ROI gate crops the input for reduced complexity detection. To evaluate performance of our system in real-world car driving scenarios, we introduce a temporally-annotated Driver Simulated Gesture (DSG-18) dataset. Scale-Gest maintains event-level F1 while significantly reducing energy and latency compared to single-detector approaches. On a battery-powered laptop running gesture streams, our ACE controller reduces per-frame energy by 4x (from 6.9 mJ to 1.6 mJ) while maintaining high gesture-detection performance (event-level F1 = 0.8-0.9) and low mean latency (6 ms).
>
---
#### [new 062] Bounded-Input True Proportional Navigation for Impact-Time Control
- **分类: eess.SY; cs.RO; math.DS**

- **简介: 该论文属于制导任务，解决如何在控制输入受限下精确拦截目标的问题。提出一种非线性制导策略，结合滑模控制实现时间约束拦截。**

- **链接: [https://arxiv.org/pdf/2605.13669](https://arxiv.org/pdf/2605.13669)**

> **作者:** Lohitvel Gopikannan; Shashi Ranjan Kumar; Abhinav Sinha
>
> **备注:** Preprint; Accepted for presentation at the 15th Asian Control Conference, June 17th-21st, 2026, Indonesia
>
> **摘要:** This paper proposes a nonlinear guidance strategy capable of intercepting a constant-velocity, non-maneuvering target while strictly satisfying the prescribed bounds on the control input (commanded acceleration). Unlike conventional strategies that estimate time-to-go using linearization or small-angle approximations, the proposed strategy employs true proportional-navigation guidance (TPNG) as a baseline, which utilizes an exact time-to-go formulation and is applicable over a wide range of target motions. In contrast to most existing strategies, which do not incorporate control input bounds into the guidance design, the proposed approach explicitly accounts for these limits by modeling the interceptor acceleration as a dynamic variable. Based on the sliding mode control technique, an effective guidance law that achieves time-constrained interception while accounting for bounded input is then derived. The performance of the proposed strategy is evaluated for various engagement scenarios.
>
---
#### [new 063] Learning When to Act: Communication-Efficient Reinforcement Learning via Run-Time Assurance
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于安全强化学习任务，解决如何在保证安全的前提下提高通信效率的问题。通过结合Lyapunov安全屏障与自适应决策机制，实现更高效的控制策略。**

- **链接: [https://arxiv.org/pdf/2605.12561](https://arxiv.org/pdf/2605.12561)**

> **作者:** Adam Haroon; Erick J. Rodríguez-Seda; Cody Fleming; Tristan Schuler
>
> **备注:** 27 pages, 6 figures
>
> **摘要:** Safe reinforcement learning (RL) typically asks $\textit{what}$ an agent should do. We ask $\textit{when}$ it needs to act, and show that a single policy can jointly learn control inputs and communication-efficient timing decisions under a pointwise Lyapunov safety shield. We focus on stabilization around a known equilibrium, where CARE-based LQR backups, Lyapunov certificates, and classical Lyapunov-STC are well defined, enabling clean comparison against analytical baselines. A run-time assurance (RTA) layer overrides the policy via a one-step-ahead Lyapunov prediction and a precomputed LQR backup, providing a strictly stronger guarantee than constrained MDP methods that enforce safety only in expectation. On an inverted pendulum, cart--pole, and planar quadrotor, the learned policy achieves $1.91\times$, $1.45\times$, and $3.51\times$ higher mean inter-sample interval (MSI) than a Lyapunov-triggered baseline; a fixed LQR controller at the same average rate is unstable on all three plants, showing that adaptive timing, not a lower average rate, makes sparsity safe. A CARE-derived Lyapunov reward transfers across environments without redesign, with a single weight $w_c$ controlling the stability--communication tradeoff; ablations confirm the RTA shield is essential, with its removal reducing MSI by $1.27$--$1.84\times$ and degrading state norms. A preference-conditioned extension recovers the full tradeoff frontier from one model at $\tfrac{2}{11}$ of training compute, and SAC experiments show the results are algorithm-agnostic across discrete and continuous domains. A 12-state 3D quadrotor case study extends the framework to higher-dimensional systems where classical STC is intractable, and robustness to $\pm30\%$ mass variation and disturbances shows graceful degradation, with the RTA absorbing what the learned policy cannot.
>
---
#### [new 064] Revealing Interpretable Failure Modes of VLMs
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于VLM安全研究任务，旨在发现模型的可解释失败模式。通过REVELIO框架，系统识别VLM在特定场景下的错误行为，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2605.12674](https://arxiv.org/pdf/2605.12674)**

> **作者:** Isha Chaudhary; Vedaant V Jain; Kavya Sachdeva; Sayan Ranu; Gagandeep Singh
>
> **摘要:** Vision-Language Models (VLMs) are increasingly used in safety-critical applications because of their broad reasoning capabilities and ability to generalize with minimal task-specific engineering. Despite these advantages, they can exhibit catastrophic failures in specific real-world situations, constituting failure modes. We introduce REVELIO, a framework for systematically uncovering interpretable failure modes in VLMs. We define a failure mode as a composition of interpretable, domain-relevant concepts-such as pedestrian proximity or adverse weather conditions-under which a target VLM consistently behaves incorrectly. Identifying such failures requires searching over an exponentially large discrete combinatorial space. To address this challenge, REVELIO combines two search procedures: a diversity-aware beam search that efficiently maps the failure landscape, and a Gaussian-process Thompson Sampling strategy that enables broader exploration of complex failure modes. We apply REVELIO to autonomous driving and indoor robotics domains, uncovering previously unreported vulnerabilities in state-of-the-art VLMs. In driving environments, the models often demonstrate weak spatial grounding and fail to account for major obstructions, leading to recommendations that would result in simulated crashes. In indoor robotics tasks, VLMs either miss safety hazards or behave excessively conservatively, producing false alarms and reducing operational efficiency. By identifying structured and interpretable failure modes, REVELIO offers actionable insights that can support targeted VLM safety improvements.
>
---
#### [new 065] EvObj: Learning Evolving Object-centric Representations for 3D Instance Segmentation without Scene Supervision
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出EvObj，用于无监督的3D实例分割任务，解决合成数据与真实点云间的几何域差距问题，通过动态优化对象先验和补全部分几何实现有效分割。**

- **链接: [https://arxiv.org/pdf/2605.13152](https://arxiv.org/pdf/2605.13152)**

> **作者:** Jiahao Chen; Zihui Zhang; Yafei Yang; Jinxi Li; Shenxing Wei; Zhixuan Sun; Bo Yang
>
> **备注:** CVPR 2026. Code and data are available at: this https URL
>
> **摘要:** We introduce EvObj for unsupervised 3D instance segmentation that bridges the geometric domain gap between synthetic pretraining data and real-world point clouds. Current methods suffer from structural discrepancies when transferring object priors from synthetic datasets (e.g., ShapeNet) to real scans (e.g., ScanNet), particularly due to morphological variations and occlusion artifacts. To address this, EvObj integrates two innovative modules: (1) An object discerning module that dynamically refines object candidates, enabling continuous adaptation of object priors to target domains; and (2) An object completion module that reconstructs partial geometries after discovering objects. We conduct extensive experiments on both real-world and synthetic datasets, demonstrating superior 3D object segmentation performance over all baselines while achieving state-of-the-art results.
>
---
#### [new 066] ERPPO: Entropy Regularization-based Proximal Policy Optimization
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，解决多维环境下的策略优化问题。通过引入熵正则化机制，提升目标定位准确性与搜索效率。**

- **链接: [https://arxiv.org/pdf/2605.13131](https://arxiv.org/pdf/2605.13131)**

> **作者:** Changha Lee; Gyusang Cho
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Multi-Agent Proximal Policy Optimization (MAPPO) is a variant of the Proximal Policy Optimization (PPO) algorithm, specifically tailored for multi-agent reinforcement learning (MARL). MAPPO optimizes cooperative multi-agent settings by employing a centralized critic with decentralized actors. However, in case of multi-dimensional environment, MAPPO can not extract optimal policy due to non-stationary agent observation. To overcome this problem, we introduce a novel approach, Entropy Regularization-based Proximal Policy Optimization (ERPPO). For the policy optimization, we first define the object detection ambiguity under multi-dimensional observation environment. Distributional Spatiotemporal Ambiguity (DSA) learner is trained to estimate object detection uncertainty in non-stationary constraints. Then, we enhance PPO with a novel Entropy Regularization term. This regularization dynamically adjusts the policy update by applying a stronger (L1) regularization in high-ambiguity observation to encourage significant exploratory actions and a weaker (L2) regularization in low-ambiguity observation to stabilize the proximal policy optimization. This approach is designed to enhance the probability of successful object localization in time-critical operations by reducing detection failures and optimizing search policy. Experiments on a testbed with AirSim-based maritime searching scenarios show that the proposed ERPPO improves accuracy performance. Our proposed method improves higher gradient than MAPPO. Qualitative results confirm that ERPPO effectiveness in terms of suppressing false detection in visually uncertain conditions.
>
---
## 更新

#### [replaced 001] AdaptNC: Adaptive Nonconformity Scores for Conformal Prediction under Distribution Shift
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于不确定性量化任务，解决分布偏移下预测区域体积过大问题。提出AdaptNC框架，联合调整非一致性评分和阈值，提升预测效率。**

- **链接: [https://arxiv.org/pdf/2602.01629](https://arxiv.org/pdf/2602.01629)**

> **作者:** Renukanandan Tumu; Aditya Singh; Rahul Mangharam
>
> **摘要:** Rigorous uncertainty quantification is essential for the safe deployment of autonomous systems in unconstrained environments. Conformal Prediction (CP) provides a distribution-free framework for this task, yet its standard formulations rely on exchangeability assumptions that are violated by the distribution shifts inherent in real-world robotics. Existing online CP methods maintain target coverage by adaptively scaling the conformal threshold, but typically employ a static nonconformity score function. We show that this fixed geometry leads to highly conservative, volume-inefficient prediction regions when environments undergo structural shifts. To address this, we propose $\textbf{AdaptNC}$, a framework for the joint online adaptation of both the nonconformity score parameters and the conformal threshold. AdaptNC leverages an adaptive reweighting scheme to optimize score functions, and introduces a replay buffer mechanism to mitigate the coverage instability that occurs during score transitions. We evaluate AdaptNC on diverse robotic benchmarks involving multi-agent policy changes, environmental changes and sensor degradation. Our results demonstrate that AdaptNC significantly reduces prediction region volume compared to state-of-the-art threshold-only baselines while maintaining target coverage levels.
>
---
#### [replaced 002] Prismatic World Model: Learning Compositional Dynamics for Planning in Hybrid Systems
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人模型预测控制任务，旨在解决混合系统中连续与离散动态建模问题。提出PRISM-WM结构化模型，分解动态为可组合原语，提升长时程规划可靠性。**

- **链接: [https://arxiv.org/pdf/2512.08411](https://arxiv.org/pdf/2512.08411)**

> **作者:** Mingwei Li; Xiaoyuan Zhang; Chengwei Yang; Zilong Zheng; Yaodong Yang
>
> **摘要:** Model-based planning in robotic domains is challenged by the hybrid nature of physical dynamics, where continuous motion is punctuated by discrete events such as contacts and impacts. Conventional latent world models typically employ monolithic neural networks that enforce global continuity, which over-smooths distinct dynamic modes (e.g., sticking vs. sliding, flight vs. stance). For a planner, this smoothing results in compounding errors during long-horizon lookaheads, rendering the search process unreliable at physical boundaries. To address this, we introduce the Prismatic World Model (PRISM-WM), a structured architecture designed to decompose complex hybrid dynamics into composable primitives. PRISM-WM uses a context-aware Mixture-of-Experts (MoE) framework where a gating mechanism implicitly identifies the current physical mode, and specialized experts predict the associated transition dynamics. We further introduce a latent orthogonalization objective to ensure expert diversity, preventing mode collapse. By modeling the mode transitions in system dynamics, PRISM-WM reduces rollout drift. Experiments on continuous control benchmarks, including high-dimensional humanoids and multi-task settings, demonstrate that PRISM-WM provides a high-fidelity substrate for trajectory optimization algorithms (e.g., TD-MPC), indicating its potential as a foundational model for model-based agents.
>
---
#### [replaced 003] Follow-Bench: A Unified Motion Planning Benchmark for Socially-Aware Robot Person Following
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决社交感知的跟随问题。提出Follow-Bench基准，评估安全与舒适性，分析现有方法并指出未来方向。**

- **链接: [https://arxiv.org/pdf/2509.10796](https://arxiv.org/pdf/2509.10796)**

> **作者:** Hanjing Ye; Weixi Situ; Jianwei Peng; Yu Zhan; Bingyi Xia; Kuanqi Cai; Hong Zhang
>
> **备注:** Project page: this https URL
>
> **摘要:** Robot person following (RPF) -- mobile robots that follow and assist a specific person -- has emerging applications in personal assistance, security patrols, eldercare, and logistics. To be effective, such robots must follow the target while ensuring safety and comfort for both the target and surrounding people. In this work, we present the first comprehensive study of RPF, which (i) surveys representative scenarios, motion-planning methods, and evaluation metrics with a focus on safety and comfort; (ii) introduces Follow-Bench, a unified benchmark simulating diverse scenarios, including various target trajectory patterns, crowd dynamics, and environmental layouts; and (iii) re-implements eight representative RPF planners, ensuring that both safety and comfort are systematically considered. Moreover, we evaluate the two best-performing planners from our benchmark on a differential-drive robot to provide insights into real-world deployment of RPF planners. Extensive simulation and real-world experiments provide quantitative study of the safety-comfort trade-offs of existing planners, while revealing open challenges and future research directions.
>
---
#### [replaced 004] Self-CriTeach: LLM Self-Teaching and Self-Critiquing for Improving Robotic Planning via Automated Domain Generation
- **分类: cs.RO**

- **简介: 该论文提出Self-CriTeach框架，用于提升机器人规划能力。解决LLM在不完美状态下的脆弱性问题，通过自动生成领域实现自教与自评，增强泛化性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.21543](https://arxiv.org/pdf/2509.21543)**

> **作者:** Jinbang Huang; Zhiyuan Li; Yuanzhao Hu; Zhanguang Zhang; Mark Coates; Xingyue Quan; Yingxue Zhang
>
> **备注:** International Conference on Machine Learning (ICML) 2026
>
> **摘要:** Large Language Models (LLMs) have shown strong promise for robotic task planning, particularly through the automatic generation of symbolic planning domains. However, prior work mainly treats generated domains as planning utilities. Such pipelines remain brittle under imperfect logical states and perception noise, while overlooking the potential of generated domains as scalable sources of reasoning supervision and structured reward signals. At the same time, reasoning LLMs depend on chain-of-thought (CoT) supervision, which is expensive to collect for robotic tasks, and reinforcement learning (RL) faces challenges in reward engineering. We propose Self-CriTeach, an LLM self-teaching and self-critiquing framework in which an LLM autonomously generates symbolic planning domains that serve a dual role: (1) In the self-teaching stage, generated domains are used to produce large-scale robotic planning problem--plan pairs, which are automatically converted into extended CoT trajectories for supervised fine-tuning. (2) In the self-critiquing stage, the same domains are reused as structured reward functions, providing dense feedback for reinforcement learning without manual reward engineering. This unified training pipeline yields a planning-enhanced LLM with higher planning success rates, stronger cross-task generalization, reduced inference cost, and improved resistance to imperfect logical states. GitHub Page: this https URL
>
---
#### [replaced 005] Robust and Safe Multi-Agent Reinforcement Learning with Communication for Autonomous Vehicles: From Simulation to Hardware
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多智能体强化学习任务，解决模拟到硬件的零样本迁移问题。提出RSR-RSMARL框架，结合通信与安全机制，提升自动驾驶车辆的安全性和协作性。**

- **链接: [https://arxiv.org/pdf/2506.00982](https://arxiv.org/pdf/2506.00982)**

> **作者:** Keshawn Smith; Zhili Zhang; H M Sabbir Ahmad; Ehsan Sabouni; Mainak Mondal; Song Han; Wenchao Li; Fei Miao
>
> **备注:** 15 pages, 5 Figures
>
> **摘要:** Deep multi-agent reinforcement learning (MARL) has been demonstrated effectively in simulations for multi-robot problems. For autonomous vehicles, the development of vehicle-to-vehicle (V2V) communication technologies provide opportunities to further enhance system safety. However, zero-shot transfer of simulator-trained MARL policies to dynamic hardware systems remains challenging, and how to leverage communication and shared information for MARL has limited demonstrations on hardware. This problem is challenged by discrepancies between simulated and physical states, system state and model uncertainties, practical shared information design, and the need for safety guarantees in both simulation and hardware. This paper designs RSR-RSMARL, a novel Robust and Safe MARL framework that supports Real-Sim-Real (RSR) policy adaptation for multi-agent systems with communication among agents, with both simulation and hardware demonstrations. RSR-RSMARL leverages state (includes shared state information among agents) and action representations considering real system complexities for MARL formulation. The MARL policy is trained with robust MARL algorithm to enable zero-shot transfer to hardware considering the sim-to-real gap. A safety shield module using Control Barrier Functions (CBFs) provides safety guarantee for each individual agent. Experimental results on 1/10th-scale autonomous vehicles with V2V communication demonstrate the ability of RSR-RSMARL framework to enhance driving safety and coordination across multiple configurations. These findings emphasize the importance of jointly designing robust policy representations and modular safety architectures to enable scalable, generalizable RSR transfer in multi-agent autonomy.
>
---
#### [replaced 006] Compact 3D Gaussian Splatting For Dense Visual SLAM
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决3D高斯溅射方法内存占用高、训练慢的问题。通过减少高斯椭球数量和参数大小，提升效率并保持重建质量。**

- **链接: [https://arxiv.org/pdf/2403.11247](https://arxiv.org/pdf/2403.11247)**

> **作者:** Tianchen Deng; Chang Nie; Shuhong Liu; Wenhua Wu; Jianfei Yang; Shenghai Yuan; Jiuming Liu; Danwei Wang; Hesheng Wang
>
> **备注:** Accepted by IJCV 2026
>
> **摘要:** Recent work has shown that 3D Gaussian-based SLAM enables high-quality reconstruction, accurate pose estimation, and real-time rendering of scenes. However, these approaches are built on a tremendous number of redundant 3D Gaussian ellipsoids, leading to high memory and storage costs, and slow training speed. To address the limitation, we propose a compact 3D Gaussian Splatting SLAM system that reduces the number and the parameter size of Gaussian ellipsoids. A sliding window-based masking strategy is first proposed to reduce the redundant ellipsoids. Then we observe that the covariance matrix (geometry) of most 3D Gaussian ellipsoids are extremely similar, which motivates a novel geometry codebook to compress 3D Gaussian geometric attributes, i.e., the parameters. Robust and accurate pose estimation is achieved by a global bundle adjustment method with reprojection loss. Extensive experiments demonstrate that our method achieves faster training and rendering speed while maintaining the state-of-the-art (SOTA) quality of the scene representation.
>
---
#### [replaced 007] QuickLAP: Quick Language-Action Preference Learning for Semi-Autonomous Systems
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出QuickLAP，用于半自主系统的学习任务，解决物理与语言反馈融合问题。通过贝叶斯框架结合两者，提升奖励函数学习效果。**

- **链接: [https://arxiv.org/pdf/2511.17855](https://arxiv.org/pdf/2511.17855)**

> **作者:** Jordan Abi Nader; David Lee; Nathaniel Dennler; Andreea Bobu
>
> **摘要:** Robots must learn from both what people do and what they say, but either modality alone is often incomplete: physical corrections are grounded but ambiguous in intent, while language expresses high-level goals but lacks physical grounding. We introduce QuickLAP: Quick Language-Action Preference learning, a Bayesian framework that fuses physical and language feedback to infer reward functions in real time. Our key insight is to treat language as a probabilistic observation over the user's latent preferences, clarifying which reward features matter and how physical corrections should be interpreted. QuickLAP uses Large Language Models (LLMs) to extract reward feature attention masks and preference shifts from free-form utterances, which it integrates with physical feedback in a closed-form update rule. This enables fast, real-time, and robust reward learning that handles ambiguous feedback. In a semi-autonomous driving simulator, QuickLAP reduces reward learning error by over 70% compared to physical-only and heuristic multimodal baselines. A 15-participant user study further validates our approach: participants found QuickLAP significantly more understandable and collaborative, and preferred its learned behavior over baselines. Code is available at this https URL.
>
---
#### [replaced 008] An Efficient Insect-inspired Approach for Visual Point-goal Navigation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于视觉点目标导航任务，旨在解决高效路径规划问题。通过模仿昆虫大脑结构，提出一种计算成本低的导航模型，表现出与先进模型相当的性能。**

- **链接: [https://arxiv.org/pdf/2601.16806](https://arxiv.org/pdf/2601.16806)**

> **作者:** Yihe Lu; Barbara Webb
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** In this work we develop a novel insect-inspired model for visual point-goal navigation. This combines abstracted models of two insect brain structures that have been implicated, respectively, in associative learning and path integration. We draw an analogy between the formal benchmark of the Habitat point-goal navigation task and the ability of insects to discover, learn, and refine visually guided paths around obstacles between a discovered food location and their nest. We demonstrate that the simple insect-inspired model exhibits performance comparable to recent state-of-the-art models at many orders of magnitude less computational cost. Testing in a more realistic simulated environment shows the approach is robust to perturbations.
>
---
#### [replaced 009] OptMap: Geometric Map Distillation via Submodular Maximization
- **分类: cs.RO**

- **简介: 该论文提出OptMap，用于在线生成应用特定的几何地图。解决多尺度感知与规划中地图选择的NP难问题，通过子模最大化实现高效、近优的地图压缩与生成。**

- **链接: [https://arxiv.org/pdf/2512.07775](https://arxiv.org/pdf/2512.07775)**

> **作者:** David Thorne; Nathan Chan; Christa S. Robison; Philip R. Osteen; Brett T. Lopez
>
> **摘要:** Autonomous robots rely on geometric maps to inform a diverse set of perception and decision-making algorithms. As autonomy requires reasoning and planning on multiple scales, each algorithm may require a different map for optimal performance. LiDAR sensors generate an abundance of geometric data (up to 50 MB per second) to satisfy these diverse requirements. However, the point-based operations required to process perception data are both memory and computationally expensive. Such operations can be bypassed via learned representations that encode similarity, but selecting informative, size-constrained maps remains an NP-hard combinatorial problem. In this work we present OptMap: a geometric map distillation algorithm which achieves online, application-specific map generation via multiple theoretical and algorithmic innovations. A central feature is the maximization of set functions that exhibit diminishing returns, i.e., submodularity, using polynomial-time algorithms with provably near-optimal solutions. We formulate a novel submodular reward function which quantifies informativeness, reduces input set sizes, and minimizes solution bias. Further, we propose a dynamically reordered streaming submodular algorithm which improves empirical solution quality and addresses input order bias via an online approximation of the value of all scans. Testing was conducted on open-source and custom datasets with an emphasis on long-duration mapping sessions, highlighting OptMap's minimal computation requirements. OptMap's practical value is then illustrated through its application to online geometric change detection. Open-source ROS1 and ROS2 packages are available and can be used alongside any LiDAR odometry algorithm.
>
---
#### [replaced 010] An Overtaking Trajectory Planning Framework Based on Spatio-temporal Topology and Reachable Set Analysis Ensuring Time Efficiency
- **分类: cs.RO**

- **简介: 该论文属于自主车辆变道轨迹规划任务，解决高速场景下轨迹生成效率低和易陷入局部最优的问题。提出SROP框架，结合时空拓扑与可达集分析，提升轨迹平滑度并加速计算。**

- **链接: [https://arxiv.org/pdf/2410.22643](https://arxiv.org/pdf/2410.22643)**

> **作者:** Wule Mao; Zhouheng Li; Entao Sun; Lei Xie; Hongye Su
>
> **摘要:** Generating overtaking trajectories in high-speed scenarios is typically addressed through hierarchical planning, which often suffers from local optima due to single initial solutions and low computational efficiency during numerical optimization. To overcome these limitations, this paper proposes a Spatio-temporal topology and Reachable set analysis enhanced Overtaking trajectory Planning framework (SROP). Specifically, by introducing topological classes to represent distinct overtaking behaviors, the upper-layer planner performs a spatio-temporal search to extract diverse initial paths, effectively preventing local optima. Subsequently, a lower-layer planner conducts parallel trajectory evaluation using reachable sets, which decouples vehicle kinematic constraints from the optimization process to ensure feasibility and significantly accelerate computation. Numerical experiments demonstrate that SROP improves trajectory smoothness by 66.8% and reduces computation time by 62.9% compared to state-of-the-art methods. Furthermore, by seamlessly integrating the method into the F1TENTH autonomous racing simulation platform, a 100-lap sensitivity analysis demonstrates high overtaking success rates in challenging scenarios, thereby validating its practical utility, real-time efficiency, and robustness.
>
---
#### [replaced 011] Cycle-resolved Cephalopod-Inspired Pulsed-Jet Robot With High-Volume Expulsion and Drag-Reduced Gliding
- **分类: cs.RO; physics.flu-dyn**

- **简介: 该论文属于仿生机器人任务，旨在解决水下推进效率问题。通过模仿章鱼喷射运动，设计具有可变形外壳的机器人，提升速度与能效。**

- **链接: [https://arxiv.org/pdf/2605.05875](https://arxiv.org/pdf/2605.05875)**

> **作者:** Yiyuan Zhang; Anye Zhong; Junkai Chen; Wenci Xin
>
> **备注:** Updated author list; no changes to the scientific content
>
> **摘要:** Cephalopod pulsed-jet locomotion is not a single isolated expulsion event, but a coordinated cycle involving jet expulsion, passive gliding, and mantle refilling. Inspired by this cycle-resolved biological strategy, this paper presents a cephalopod-inspired pulsed-jet robot with a rigid-soft hybrid origami mantle that enables large, actively driven, and geometry-guided body deformation. The proposed mantle integrates rigid folding panels with a compliant silicone framework, allowing a 75% effective cavity-volume reduction during expulsion and reducing the projected cross-sectional drag area by approximately 75.7% in the contracted gliding configuration. Using this platform, we formulate a cycle-resolved framework to separately investigate how expelled volume, glide duration, and refill pathway influence whole-cycle locomotion performance. Experiments show that the robot reaches a peak speed of approximately 0.5 m/s (3.8 BL/s) and an average speed exceeding 0.2 m/s (1.5 BL/s) within the first jetting cycle. The results further demonstrate the roles of high expelled-volume-ratio contraction in speed generation, reduced-drag-area gliding under different glide durations, and mantle-aperture-inspired passive inlet valves in assisting refill. This work provides both a robotic implementation of actively deformable cephalopod-like jet propulsion and a unified experimental platform for studying expulsion-gliding-refilling dynamics in pulsed-jet locomotion.
>
---
#### [replaced 012] UniJEPA: Enhancing Robot Policy via Unified Continuous and Discrete Representation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人策略学习任务，旨在提升机器人在开放环境中的泛化能力。通过统一的连续与离散表征学习，构建更高效的机器人策略。**

- **链接: [https://arxiv.org/pdf/2510.10642](https://arxiv.org/pdf/2510.10642)**

> **作者:** Jianke Zhang; Yucheng Hu; Yanjiang Guo; Xiaoyu Chen; Yichen Liu; Wenna Chen; Chaochao Lu; Jianyu Chen
>
> **摘要:** Building generalist robot policies that can handle diverse tasks in open-ended environments is a central challenge in robotics. To leverage knowledge from large-scale pretraining, prior work (VLA) has typically built generalist policies either on top of vision-language understanding models (VLMs) or generative models. However, both semantic understanding from vision-language pretraining and visual dynamics modeling from visual-generation pretraining are crucial for embodied robots. Recent unified models of generation and understanding have demonstrated strong capabilities in both comprehension and generation through large-scale pretraining. We posit that robotic policy learning can likewise benefit from the combined strengths of understanding, planning, and continuous future representation learning. Building on this insight, we introduce UniJEPA, which acquires the ability to dynamically model high-dimensional visual features through pretraining on over 1M internet-scale instructional manipulation videos. Subsequently, UniJEPA is fine-tuned on data collected from the robot embodiment, enabling the learning of mappings from predictive representations to action tokens. Extensive experiments show our approach consistently outperforms baseline methods in terms of 9\% and 12\% across simulation environments and real-world out-of-distribution tasks.
>
---
#### [replaced 013] ALAM: Algebraically Consistent Latent Action Model for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ALAM模型，解决VLA模型中动作数据稀缺的问题。通过视频学习结构化潜在动作，提升策略生成效果。**

- **链接: [https://arxiv.org/pdf/2605.10819](https://arxiv.org/pdf/2605.10819)**

> **作者:** Zuojin Tang; Haoyun Liu; Xinyuan Chang; Changjie Wu; Dongjie Huo; Yandan Yang; Bin Liu; Zhejia Cai; Feng Xiong; Mu Xu; jiachen Luo; De Ma; Zhiheng Ma; Gang Pan
>
> **摘要:** Vision-language-action (VLA) models remain constrained by the scarcity of action-labeled robot data, whereas action-free videos provide abundant evidence of how the physical world changes. Latent action models offer a promising way to extract such priors from videos, but reconstruction-trained latent codes are not necessarily suitable for policy generation: they may predict future observations while lacking the structure needed to be reused or generated coherently with robot actions. We introduce ALAM (Algebraic Latent Action Model), an Algebraically Consistent Latent Action Model that turns temporal relations in action-free video into structural supervision. Given frame triplets, ALAM learns latent transitions that are grounded by reconstruction while being regularized by composition and reversal consistency, encouraging a locally additive transition space. For downstream VLA learning, we freeze the pretrained encoder and use its latent transition sequences as auxiliary generative targets, co-generated with robot actions under a joint flow-matching objective. This couples structured latent transitions with flow-based policy generation, allowing the policy to exploit ALAM's locally consistent transition geometry without requiring latent-to-action decoding. Representation probes show that ALAM reduces additivity and reversibility errors by 25-85 times over unstructured latent-action baselines and improves long-horizon cumulative reconstruction. When transferred to VLA policies, ALAM raises the average success rate from 47.9% to 85.0% on MetaWorld MT50 and from 94.1% to 98.1% on LIBERO, with consistent gains on real-world manipulation tasks. Ablations further confirm that the strongest improvements arise from the synergy between algebraically structured latent transitions and joint flow matching.
>
---
#### [replaced 014] TeleGate: Whole-Body Humanoid Teleoperation via Gated Expert Selection with Motion Prior
- **分类: cs.RO**

- **简介: 该论文属于人形机器人全身遥操作任务，解决复杂动态动作跟踪精度不足问题。提出TeleGate框架，通过门控专家选择和运动先验模块实现高精度实时控制。**

- **链接: [https://arxiv.org/pdf/2602.09628](https://arxiv.org/pdf/2602.09628)**

> **作者:** Jie Li; Bing Tang; Feng Wu
>
> **备注:** Accepted by RSS 2026. Project page: this https URL
>
> **摘要:** Real-time whole-body teleoperation is a critical method for humanoid robots to perform complex tasks in unstructured environments. However, developing a unified controller that robustly supports diverse human motions remains a significant challenge. Existing methods typically distill multiple expert policies into a single general policy, which often inevitably leads to performance degradation, particularly on highly dynamic motions. This paper presents TeleGate, a unified whole-body teleoperation framework for humanoid robots that achieves high-precision tracking across various motions while avoiding the performance loss inherent in knowledge distillation. Our key idea is to preserve the full capability of domain-specific expert policies by training a lightweight gating network, which dynamically activates experts in real-time based on proprioceptive states and reference trajectories. Furthermore, to compensate for the absence of future reference trajectories in real-time teleoperation, we introduce a VAE-based motion prior module that extracts implicit future motion intent from historical observations, enabling anticipatory control for motions requiring prediction such as jumping and standing up. We conducted empirical evaluations in simulation and also deployed our technique on the Unitree G1 humanoid robot. Using only 2.5 hours of motion capture data for training, our TeleGate achieves high-precision real-time teleoperation across diverse dynamic motions (e.g., running, fall recovery, and jumping), significantly outperforming the baseline methods in both tracking accuracy and success rate.
>
---
#### [replaced 015] Unify Robot Actions in Camera Frame
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，解决跨平台动作表示不一致问题。提出CalibAll方法，统一相机帧下的动作表示，提升跨机器人预训练效果。**

- **链接: [https://arxiv.org/pdf/2511.17001](https://arxiv.org/pdf/2511.17001)**

> **作者:** Sicheng Xie; Lingchen Meng; Zijie Diao; Haidong Cao; Zhiying Du; Shuyuan Tu; Jiaqi Leng; Qiuyue Wang; Mingsheng Li; Shuai Bai; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Cross-embodiment robot learning requires a unified action representation with consistent semantics across robot platforms. Existing representations suffer from platform-specific inconsistencies, while current solutions either maintain embodiment-specific action heads or learn latent action spaces, without fundamentally resolving the mismatch. We propose to unify robot actions in the camera frame using camera extrinsics, so that actions share consistent geometric semantics across different robot embodiments, including both single-arm and bimanual robots. However, most existing datasets lack camera extrinsic annotations, and existing offline calibration methods either suffer from local minima or require robot-specific training data. To address this gap, we present CalibAll, a training-free, robot-independent annotation pipeline that estimates camera extrinsics for offline datasets and converts heterogeneous robot actions into standardized camera-frame actions. CalibAll follows a coarse-to-fine calibration strategy: temporal PnP provides a stable initialization, followed by differentiable rendering-based refinement for high precision. Beyond extrinsics, CalibAll produces standardized TCP-pose actions and auxiliary annotations. We apply CalibAll to 16 datasets across 4 robot platforms, producing approximately 97K calibrated data episodes. Downstream simulation and real-robot experiments show that cross-embodiment pretraining with camera-frame actions achieves state-of-the-art performance.
>
---
#### [replaced 016] Multi-Modal World Model for Physical Robot Interactions: Simultaneous Visual and Tactile Predictions for Enhanced Accuracy
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人交互任务，旨在解决复杂环境中物理互动预测问题。通过融合视觉与触觉信息，提升预测准确性与鲁棒性，尤其在物理模糊场景中效果显著。**

- **链接: [https://arxiv.org/pdf/2304.11193](https://arxiv.org/pdf/2304.11193)**

> **作者:** Willow Mandil; Amir Ghalamzan-E
>
> **备注:** This paper is accepted for publication in Robotics and Autonomous Systems
>
> **摘要:** Predicting the outcomes of robotic actions, often referred to as learning a world model, in complex environments remains a fundamental challenge in robotics. Existing approaches primarily rely on visual observations and action inputs to generate video-based predictions, frequently overlooking the critical role of tactile feedback in understanding physical interactions. In this work, we investigate the integration of tactile and visual information within predictive perception systems for physical robot interaction. We demonstrate that visuo-tactile prediction provides the greatest benefits in physically ambiguous interaction regimes, while improvements are naturally limited when object dynamics are visually inferable. Furthermore, we introduce two novel robot-pushing datasets collected using a magnetic-based tactile sensor for unsupervised learning. The first dataset comprises visually identical objects with varying physical properties, explicitly isolating physical ambiguity, while the second mirrors existing robot-pushing benchmarks involving clusters of household objects. Our results show that tactile-visual integration improves prediction accuracy and robustness under physical ambiguity, while offering limited gains in visually unambiguous settings. Code and datasets are publicly available.
>
---
#### [replaced 017] When Backdoors Meet Partial Observability: Attacking Real-World Reinforcement Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于强化学习安全任务，解决真实环境中后门攻击问题。针对部分可观测场景，提出DGBA框架，通过视觉触发和优势中毒策略实现有效攻击。**

- **链接: [https://arxiv.org/pdf/2601.14104](https://arxiv.org/pdf/2601.14104)**

> **作者:** Tairan Huang; Qingqing Ye; Yulin Jin; Jiawei Lian; Yaxin Xiao; Yi Wang; Haibo Hu
>
> **摘要:** Backdoor attacks can cause reinforcement learning (RL) policies to behave normally under clean inputs while executing malicious behaviors when triggers are present. Existing RL backdoor attacks are primarily studied in simulation and often assume that attackers can reliably manipulate the observations driving policy decisions. This assumption becomes fragile in real-world deployment, where RL policies commonly rely on multimodal observations. Attackers can manipulate visual inputs through physical triggers, but auxiliary states such as LiDAR and odometry signals remain uncontrollable and vary across trajectories. We study this overlooked challenge and propose a diffusion-guided backdoor attack framework (DGBA) for real-world RL. DGBA uses small printable visual patches as triggers and learns a stochastic trigger distribution via conditional diffusion to maintain consistent attack activation under varying uncontrollable states. We further introduce an advantage-based poisoning strategy that injects triggers only at decision-critical training states. Experiments on a physical TurtleBot3 platform show that DGBA consistently outperforms prior RL backdoor attacks while preserving normal task performance. Demo videos and code are available in the supplementary material.
>
---
#### [replaced 018] Unifying Entropy Regularization in Optimal Control: From and Back to Classical Objectives via Iterated Soft Policies and Path Integral Solutions
- **分类: math.OC; cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于最优控制领域，旨在统一多种控制框架。通过KL正则化，提出一种新范式，解决传统方法的局限性，并实现更高效的路径积分解法。**

- **链接: [https://arxiv.org/pdf/2512.06109](https://arxiv.org/pdf/2512.06109)**

> **作者:** Ajinkya Bhole; Mohammad Mahmoudi Filabadi; Guillaume Crevecoeur; Tom Lefebvre
>
> **备注:** refurbished introduction, added a few remarks, reduced size
>
> **摘要:** This paper develops a unified perspective on several optimal control formulations through the lens of Kullback-Leibler (KL) regularization. We propose a central problem that separates the KL penalties on policies and transitions with independent weights, thus generalizing the standard trajectory-level KL-regularization used in probabilistic optimal control. This umbrella formulation recovers various control problems: the classical Stochastic Optimal Control (SOC), Risk-Sensitive Stochastic Optimal Control (RSOC), and their policy-based KL-regularized counterparts, termed soft-policy SOC and RSOC, which yield tractable surrogates. Beyond being regularized variants, these soft-policy formulations majorize the original SOC and RSOC, thus, iterating their solutions recovers the original objectives. We further identify a synchronized case of soft-policy RSOC where the policy and transition KL weights coincide, yielding a linear Bellman operator, path-integral solution, and compositionality -- extending these computationally favourable properties to a broad class of control problems.
>
---
#### [replaced 019] Block-wise Adaptive Caching for Accelerating Diffusion Policy
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决扩散策略计算成本高的问题。通过块级自适应缓存技术（BAC），提升推理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2506.13456](https://arxiv.org/pdf/2506.13456)**

> **作者:** Kangye Ji; Yuan Meng; Hanyun Cui; Ye Li; Jianbo Zhou; Shengjia Hua; Lei Chen; Zhi Wang
>
> **摘要:** Diffusion Policy has demonstrated strong visuomotor modeling capabilities, but its high computational cost renders it impractical for real-time robotic control. Despite huge redundancy across repetitive denoising steps, existing diffusion acceleration techniques fail to generalize to Diffusion Policy due to fundamental architectural and data divergences. In this paper, we propose $\textbf{B}$lock-wise $\textbf{A}$daptive $\textbf{C}$aching ($\textbf{BAC}$), a method to accelerate Diffusion Policy by caching intermediate action features. BAC achieves lossless action generation acceleration by adaptively updating and reusing cached features at the block level, based on a key observation that feature similarities exhibit non-uniform temporal dynamics and distinct block-specific patterns. To operationalize this insight, we first design an Adaptive Caching Scheduler to identify optimal update timesteps by maximizing the global feature similarities between cached and skipped features. However, applying this scheduler for each block leads to significant error surges due to the inter-block propagation of caching errors, particularly within Feed-Forward Network (FFN) blocks. To mitigate this issue, we develop the Bubbling Union Algorithm, which truncates these errors by updating the upstream blocks with significant caching errors before downstream FFNs. As a training-free plugin, BAC is readily integrable with existing transformer-based Diffusion Policy and vision-language-action models. Extensive experiments on multiple robotic benchmarks demonstrate that BAC achieves up to 3$\times$ inference speedup for free. Project page: this https URL.
>
---
#### [replaced 020] ZeD-MAP: Bundle Adjustment Guided Zero-Shot Depth Maps for Real-Time Aerial Imaging
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于实时航空影像深度重建任务，解决大尺度、低纹理等挑战，通过结合扩散模型与束调整技术，实现高精度、实时的三维地图生成。**

- **链接: [https://arxiv.org/pdf/2604.04667](https://arxiv.org/pdf/2604.04667)**

> **作者:** Selim Ahmet Iz; Francesco Nex; Norman Kerle; Henry Meissner; Ralf Berger
>
> **摘要:** Real-time depth reconstruction from ultra-high-resolution UAV imagery is essential for time-critical geospatial tasks such as disaster response, yet remains challenging due to wide-baseline parallax, large image sizes, low-texture or specular surfaces, occlusions, and strict computational constraints. Recent zero-shot diffusion models offer fast per-image dense predictions without task-specific retraining, and require fewer labelled datasets than transformer-based predictors while avoiding the rigid capture geometry requirement of classical multi-view stereo. However, their probabilistic inference prevents reliable metric accuracy and temporal consistency across sequential frames and overlapping tiles. We present ZeD-MAP, a cluster-level framework that converts a test-time diffusion depth model into a metrically consistent, SLAM-like mapping pipeline by integrating incremental cluster-based bundle adjustment (BA). Streamed UAV frames are grouped into overlapping clusters; periodic BA produces metrically consistent poses and sparse 3D tie-points, which are reprojected into selected frames and used as metric guidance for diffusion-based depth estimation. Validation on ground-marker flights captured at approximately 50 m altitude (GSD is approximately 0.85 cm/px, corresponding to 2,650 square meters ground coverage per frame) with the DLR Modular Aerial Camera System (MACS) shows that our method achieves sub-meter accuracy, with approximately 0.87 m error in the horizontal (XY) plane and 0.12 m in the vertical (Z) direction, while maintaining per-image runtimes between 1.47 and 4.91 seconds. Results are subject to minor noise from manual point-cloud annotation. These findings show that BA-based metric guidance provides consistency comparable to classical photogrammetric methods while significantly accelerating processing, enabling real-time 3D map generation.
>
---
#### [replaced 021] When to Act, Ask, or Learn: Uncertainty-Aware Policy Steering
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决部署时策略适应问题。针对VLM过自信导致的性能下降，提出UPS框架，联合处理语义不确定性和动作可行性，选择执行、询问或干预策略，提升系统持续学习能力。**

- **链接: [https://arxiv.org/pdf/2602.22474](https://arxiv.org/pdf/2602.22474)**

> **作者:** Jessie Yuan; Yilin Wu; Andrea Bajcsy
>
> **备注:** To appear in Robotics: Science and Systems 2026
>
> **摘要:** Policy steering is an emerging way to adapt robot behaviors at deployment-time: a learned verifier analyzes low-level action samples proposed by a pre-trained policy (e.g., diffusion policy) and selects only those aligned with the task. While Vision-Language Models (VLMs) are promising general-purpose verifiers due to their reasoning capabilities, existing frameworks often assume these models are well-calibrated. In practice, the overconfident judgment from VLM can degrade the steering performance under both high-level semantic uncertainty in task specifications and low-level action uncertainty or incapability of the pre-trained policy. We propose uncertainty-aware policy steering (UPS), a framework that jointly reasons about semantic task uncertainty and low-level action feasibility, and selects an uncertainty resolution strategy: execute a high-confidence action, clarify task ambiguity via natural language queries, or ask for action interventions to correct the low-level policy when it is deemed incapable at the task. We leverage conformal prediction to calibrate the composition of the VLM and the pre-trained base policy, providing statistical assurances that the verifier selects the correct strategy. After collecting interventions during deployment, we employ residual learning to improve the capability of the pre-trained policy, enabling the system to learn continually but with minimal expensive human feedback. We demonstrate our framework through experiments in simulation and on hardware, showing that UPS can disentangle confident, ambiguous, and incapable scenarios and minimizes expensive user interventions compared to uncalibrated baselines and prior human- or robot-gated continual learning approaches. Videos can be found at this https URL
>
---
#### [replaced 022] Saturation-Aware Angular Velocity Estimation: Extending the Robustness of SLAM to Aggressive Motions
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决陀螺仪在剧烈运动中饱和导致的定位误差问题。通过加速度计估计角速度，提升SLAM鲁棒性。**

- **链接: [https://arxiv.org/pdf/2310.07844](https://arxiv.org/pdf/2310.07844)**

> **作者:** Simon-Pierre Deschênes; Dominic Baril; Matěj Boxan; Johann Laconte; Philippe Giguère; François Pomerleau
>
> **备注:** 7 pages, 7 figures, published in 2024 IEEE International Conference on Robotics and Automation (ICRA), Yokohama, Japan
>
> **摘要:** We propose a novel angular velocity estimation method to increase the robustness of Simultaneous Localization And Mapping (SLAM) algorithms against gyroscope saturations induced by aggressive motions. Field robotics expose robots to various hazards, including steep terrains, landslides, and staircases, where substantial accelerations and angular velocities can occur if the robot loses stability and tumbles. These extreme motions can saturate sensor measurements, especially gyroscopes, which are the first sensors to become inoperative. While the structural integrity of the robot is at risk, the robustness of the SLAM framework is oftentimes given little consideration. Consequently, even if the robot is physically capable of continuing the mission, its operation will be compromised due to a corrupted representation of the world. Regarding this problem, we propose a method to estimate the angular velocity using accelerometers during extreme rotations caused by tumbling. We show that our method reduces the median localization error by 71.5 % in translation and 65.5 % in rotation and is robust to mapping failures, which occurred in 37.5 % of the experiments without our method. We also propose the Tumbling-Induced Gyroscope Saturation (TIGS) dataset, which consists of outdoor experiments recording the motion of a mechanical lidar subject to angular velocities four times higher than other similar datasets available. The dataset is available online at this https URL.
>
---
#### [replaced 023] Simulation-Ready Cluttered Scene Estimation via Physics-aware Joint Shape and Pose Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于真实到仿真场景重建任务，解决杂乱环境中物体形状与位姿估计问题。提出联合优化方法，结合物理约束与结构稀疏性，提升重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.20150](https://arxiv.org/pdf/2602.20150)**

> **作者:** Wei-Cheng Huang; Jiaheng Han; Xiaohan Ye; Zherong Pan; Kris Hauser
>
> **备注:** Accepted to RSS 2026, camera-ready version; 17 pages, 15 figures
>
> **摘要:** Estimating simulation-ready scenes from real-world observations is crucial for downstream planning and policy learning tasks. Regretfully, existing methods struggle in cluttered environments, often exhibiting prohibitive computational cost, poor robustness, and restricted generality when scaling to multiple interacting objects. We propose a unified optimization-based formulation for real-to-sim scene estimation that jointly recovers the shapes and poses of multiple rigid objects under physical constraints. Our method is built on two key technical innovations. First, we leverage the recently introduced shape-differentiable contact model, whose global differentiability permits joint optimization over object geometry and pose while modeling inter-object contacts. Second, we exploit the structured sparsity of the augmented Lagrangian Hessian to derive an efficient linear system solver whose computational cost scales favorably with scene complexity. Building on this formulation, we develop an end-to-end Simulation-ready Physics-Aware Reconstruction for Cluttered Scenes (SPARCS) pipeline, which integrates learning-based object initialization, physics-constrained joint shape-pose optimization, and differentiable texture refinement. Experiments on cluttered scenes with up to 5 objects and 22 convex hulls demonstrate that our approach robustly reconstructs physically valid, simulation-ready object shapes and poses. Project webpage: this https URL.
>
---
#### [replaced 024] SCU-Hand with Integrated Single-Sheet Valve: A Funnel-Shaped Robotic Hand for Milligram-Scale Powder Handling
- **分类: cs.RO**

- **简介: 该论文属于实验室自动化任务，旨在解决毫克级粉末精准处理难题。提出SCU-Hand-SV机械手，集成单片阀实现精确定量输送，提升自动化效率与准确性。**

- **链接: [https://arxiv.org/pdf/2512.07091](https://arxiv.org/pdf/2512.07091)**

> **作者:** Tomoya Takahashi; Yusaku Nakajima; Cristian Camilo Beltran-Hernandez; Yuki Kuroda; Kazutoshi Tanaka; Masashi Hamaya; Kanta Ono; Yoshitaka Ushiku
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Laboratory Automation (LA) has the potential to accelerate solid-state materials discovery by enabling continuous robotic operation without human intervention. While robotic systems have been developed for tasks such as powder grinding and X-ray diffraction (XRD) analysis, fully automating powder handling at the milligram scale remains a significant challenge due to the complex flow dynamics of powders and the diversity of laboratory tasks. To address this challenge, this study proposes the SCU-Hand-SV (Soft Conical Universal Robotic Hand with Single-sheet Valve), which preserves the softness and conical sheet designs in prior work while incorporating a controllable valve at the cone apex to enable precise, incremental dispensing of milligram-scale powder quantities. The SCU-Hand-SV is integrated with an external balance through a feedback control system based on a model of powder flow and online parameter identification. Experimental evaluations with glass beads, monosodium glutamate, and titanium dioxide demonstrated that 80% of the trials achieved an error within -2 mg to +2 mg, and the maximum error observed was approximately 20 mg across a target range of 20 mg to 3 g. In addition, by incorporating flow prediction models commonly used for hoppers and performing online parameter identification, the system is able to adapt to variations in powder dynamics. Compared to direct PID control, the proposed model-based control significantly improved both accuracy and convergence speed. These results highlight the potential of the proposed system to enable efficient and flexible powder weighing, with scalability toward larger quantities and applicability to a broad range of laboratory automation tasks.
>
---
#### [replaced 025] ViTacFormer: Learning Cross-Modal Representation for Visuo-Tactile Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文提出ViTacFormer，用于视觉与触觉的跨模态表示学习，解决机器人精细操作问题，通过融合视觉与触觉信息提升操控精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.15953](https://arxiv.org/pdf/2506.15953)**

> **作者:** Liang Heng; Haoran Geng; Kaifeng Zhang; Pieter Abbeel; Jitendra Malik
>
> **摘要:** Dexterous manipulation is a cornerstone capability for robotic systems aiming to interact with the physical world in a human-like manner. Although vision-based methods have advanced rapidly, tactile sensing remains crucial for fine-grained control, particularly in unstructured or visually occluded settings. We present ViTacFormer, a representation-learning approach that couples a cross-attention encoder to fuse high-resolution vision and touch with an autoregressive tactile prediction head that anticipates future contact signals. Building on this architecture, we devise an easy-to-challenging curriculum that steadily refines the visual-tactile latent space, boosting both accuracy and robustness. The learned cross-modal representation drives imitation learning for multi-fingered hands, enabling precise and adaptive manipulation. Across a suite of challenging real-world benchmarks, our method achieves approximately 50% higher success rates than prior state-of-the-art systems. To our knowledge, it is also the first to autonomously complete long-horizon dexterous manipulation tasks that demand highly precise control with an anthropomorphic hand, successfully executing up to 11 sequential stages and sustaining continuous operation for 2.5 minutes.
>
---
#### [replaced 026] Perception with Guarantees: Certified Pose Estimation via Reachability Analysis
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于姿态估计任务，解决安全关键场景下姿态估计的可靠性问题。通过摄像头和已知目标几何，结合可达性分析和神经网络验证，实现可信姿态估计。**

- **链接: [https://arxiv.org/pdf/2602.10032](https://arxiv.org/pdf/2602.10032)**

> **作者:** Tobias Ladner; Yasser Shoukry; Matthias Althoff
>
> **备注:** Accepted at Computed Aided Verification (CAV'2026)
>
> **摘要:** Agents in cyber-physical systems are increasingly entrusted with safety-critical tasks. Ensuring safety of these agents often requires localizing the pose for subsequent actions. Pose estimates can, e.g., be obtained from various combinations of lidar sensors, cameras, and external services such as GPS. Crucially, in safety-critical domains, a rough estimate is insufficient to formally determine safety, i.e., guaranteeing safety even in the worst-case scenario, and external services might additionally not be trustworthy. We address this problem by presenting a certified pose estimation in 3D solely from a camera image and a well-known target geometry. This is realized by formally bounding the pose, which is computed by leveraging recent results from reachability analysis and formal neural network verification. Our experiments demonstrate that our approach efficiently and accurately localizes agents in both synthetic and real-world experiments.
>
---
#### [replaced 027] Robotic Manipulation by Imitating Generated Videos Without Physical Demonstrations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RIGVid系统，通过模仿AI生成视频实现机器人操作，解决无需物理演示的复杂任务控制问题。工作包括视频生成、过滤、姿态跟踪与迁移。**

- **链接: [https://arxiv.org/pdf/2507.00990](https://arxiv.org/pdf/2507.00990)**

> **作者:** Shivansh Patel; Shraddhaa Mohan; Hanlin Mai; Unnat Jain; Svetlana Lazebnik; Yunzhu Li
>
> **备注:** In ICLR 2026. Website: this https URL
>
> **摘要:** This work introduces Robots Imitating Generated Videos (RIGVid), a system that enables robots to perform complex manipulation tasks--such as pouring, wiping, and mixing--purely by imitating AI-generated videos, without requiring any physical demonstrations or robot-specific training. Given a language command and an initial scene image, a video diffusion model generates potential demonstration videos, and a vision-language model (VLM) automatically filters out results that do not follow the command. A 6D pose tracker then extracts object trajectories from the video, and the trajectories are retargeted to the robot in an embodiment-agnostic fashion. Through extensive real-world evaluations, we show that filtered generated videos are as effective as real demonstrations, and that performance improves with generation quality. We also show that relying on generated videos outperforms more compact alternatives such as keypoint prediction using VLMs, and that strong 6D pose tracking outperforms other ways to extract trajectories, such as dense feature point tracking. These findings suggest that videos produced by a state-of-the-art off-the-shelf model can offer an effective source of supervision for robotic manipulation.
>
---
