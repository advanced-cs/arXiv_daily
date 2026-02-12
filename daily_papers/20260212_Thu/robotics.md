# 机器人 cs.RO

- **最新发布 48 篇**

- **更新 25 篇**

## 最新发布

#### [new 001] LAP: Language-Action Pre-Training Enables Zero-shot Cross-Embodiment Transfer
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出LAP方法，解决机器人零样本跨实体迁移问题。通过自然语言表示动作，实现无需微调的高效迁移，提升机器人泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10556v1](https://arxiv.org/pdf/2602.10556v1)**

> **作者:** Lihan Zha; Asher J. Hancock; Mingtong Zhang; Tenny Yin; Yixuan Huang; Dhruv Shah; Allen Z. Ren; Anirudha Majumdar
>
> **备注:** Project website: https://lap-vla.github.io
>
> **摘要:** A long-standing goal in robotics is a generalist policy that can be deployed zero-shot on new robot embodiments without per-embodiment adaptation. Despite large-scale multi-embodiment pre-training, existing Vision-Language-Action models (VLAs) remain tightly coupled to their training embodiments and typically require costly fine-tuning. We introduce Language-Action Pre-training (LAP), a simple recipe that represents low-level robot actions directly in natural language, aligning action supervision with the pre-trained vision-language model's input-output distribution. LAP requires no learned tokenizer, no costly annotation, and no embodiment-specific architectural design. Based on LAP, we present LAP-3B, which to the best of our knowledge is the first VLA to achieve substantial zero-shot transfer to previously unseen robot embodiments without any embodiment-specific fine-tuning. Across multiple novel robots and manipulation tasks, LAP-3B attains over 50% average zero-shot success, delivering roughly a 2x improvement over the strongest prior VLAs. We further show that LAP enables efficient adaptation and favorable scaling, while unifying action prediction and VQA in a shared language-action format that yields additional gains through co-training.
>
---
#### [new 002] Multi-Task Reinforcement Learning of Drone Aerobatics by Exploiting Geometric Symmetries
- **分类: cs.RO**

- **简介: 该论文属于多任务强化学习领域，旨在解决微飞行器在复杂机动动作中的控制问题。通过利用几何对称性，提出GEAR框架，提升数据效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10997v1](https://arxiv.org/pdf/2602.10997v1)**

> **作者:** Zhanyu Guo; Zikang Yin; Guobin Zhu; Shiliang Guo; Shiyu Zhao
>
> **摘要:** Flight control for autonomous micro aerial vehicles (MAVs) is evolving from steady flight near equilibrium points toward more aggressive aerobatic maneuvers, such as flips, rolls, and Power Loop. Although reinforcement learning (RL) has shown great potential in these tasks, conventional RL methods often suffer from low data efficiency and limited generalization. This challenge becomes more pronounced in multi-task scenarios where a single policy is required to master multiple maneuvers. In this paper, we propose a novel end-to-end multi-task reinforcement learning framework, called GEAR (Geometric Equivariant Aerobatics Reinforcement), which fully exploits the inherent SO(2) rotational symmetry in MAV dynamics and explicitly incorporates this property into the policy network architecture. By integrating an equivariant actor network, FiLM-based task modulation, and a multi-head critic, GEAR achieves both efficiency and flexibility in learning diverse aerobatic maneuvers, enabling a data-efficient, robust, and unified framework for aerobatic control. GEAR attains a 98.85\% success rate across various aerobatic tasks, significantly outperforming baseline methods. In real-world experiments, GEAR demonstrates stable execution of multiple maneuvers and the capability to combine basic motion primitives to complete complex aerobatics.
>
---
#### [new 003] A Human-in-the-Loop Confidence-Aware Failure Recovery Framework for Modular Robot Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人故障恢复任务，解决模块化机器人在不确定环境中的失败恢复问题。通过结合模块不确定性与人类干预成本，优化人机协作恢复过程，提升恢复效率并降低用户负担。**

- **链接: [https://arxiv.org/pdf/2602.10289v1](https://arxiv.org/pdf/2602.10289v1)**

> **作者:** Rohan Banerjee; Krishna Palempalli; Bohan Yang; Jiaying Fang; Alif Abdullah; Tom Silver; Sarah Dean; Tapomayukh Bhattacharjee
>
> **备注:** The second and third authors contributed equally. The last two authors advised equally
>
> **摘要:** Robots operating in unstructured human environments inevitably encounter failures, especially in robot caregiving scenarios. While humans can often help robots recover, excessive or poorly targeted queries impose unnecessary cognitive and physical workload on the human partner. We present a human-in-the-loop failure-recovery framework for modular robotic policies, where a policy is composed of distinct modules such as perception, planning, and control, any of which may fail and often require different forms of human feedback. Our framework integrates calibrated estimates of module-level uncertainty with models of human intervention cost to decide which module to query and when to query the human. It separates these two decisions: a module selector identifies the module most likely responsible for failure, and a querying algorithm determines whether to solicit human input or act autonomously. We evaluate several module-selection strategies and querying algorithms in controlled synthetic experiments, revealing trade-offs between recovery efficiency, robustness to system and user variables, and user workload. Finally, we deploy the framework on a robot-assisted bite acquisition system and demonstrate, in studies involving individuals with both emulated and real mobility limitations, that it improves recovery success while reducing the workload imposed on users. Our results highlight how explicitly reasoning about both robot uncertainty and human effort can enable more efficient and user-centered failure recovery in collaborative robots. Supplementary materials and videos can be found at: http://emprise.cs.cornell.edu/modularhil
>
---
#### [new 004] APEX: Learning Adaptive High-Platform Traversal for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出APEX系统，解决人形机器人高平台行走问题。通过深度强化学习，实现攀爬、行走和姿态调整等技能的自主切换与稳定执行。**

- **链接: [https://arxiv.org/pdf/2602.11143v1](https://arxiv.org/pdf/2602.11143v1)**

> **作者:** Yikai Wang; Tingxuan Leng; Changyi Lin; Shiqi Liu; Shir Simon; Bingqing Chen; Jonathan Francis; Ding Zhao
>
> **备注:** Project Website: https://apex-humanoid.github.io/
>
> **摘要:** Humanoid locomotion has advanced rapidly with deep reinforcement learning (DRL), enabling robust feet-based traversal over uneven terrain. Yet platforms beyond leg length remain largely out of reach because current RL training paradigms often converge to jumping-like solutions that are high-impact, torque-limited, and unsafe for real-world deployment. To address this gap, we propose APEX, a system for perceptive, climbing-based high-platform traversal that composes terrain-conditioned behaviors: climb-up and climb-down at vertical edges, walking or crawling on the platform, and stand-up and lie-down for posture reconfiguration. Central to our approach is a generalized ratchet progress reward for learning contact-rich, goal-reaching maneuvers. It tracks the best-so-far task progress and penalizes non-improving steps, providing dense yet velocity-free supervision that enables efficient exploration under strong safety regularization. Based on this formulation, we train LiDAR-based full-body maneuver policies and reduce the sim-to-real perception gap through a dual strategy: modeling mapping artifacts during training and applying filtering and inpainting to elevation maps during deployment. Finally, we distill all six skills into a single policy that autonomously selects behaviors and transitions based on local geometry and commands. Experiments on a 29-DoF Unitree G1 humanoid demonstrate zero-shot sim-to-real traversal of 0.8 meter platforms (approximately 114% of leg length), with robust adaptation to platform height and initial pose, as well as smooth and stable multi-skill transitions.
>
---
#### [new 005] From Representational Complementarity to Dual Systems: Synergizing VLM and Vision-Only Backbones for End-to-End Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在提升端到端驾驶规划性能。通过结合视觉语言模型与纯视觉模型，探索两者互补性，提出混合与双策略系统以提高表现。**

- **链接: [https://arxiv.org/pdf/2602.10719v1](https://arxiv.org/pdf/2602.10719v1)**

> **作者:** Sining Ang; Yuguang Yang; Chenxu Dang; Canyu Chen; Cheng Chi; Haiyan Liu; Xuanyao Mao; Jason Bao; Xuliang; Bingchuan Sun; Yan Wang
>
> **备注:** 22 pages (10 pages main text + 12 pages appendix), 18 figures
>
> **摘要:** Vision-Language-Action (VLA) driving augments end-to-end (E2E) planning with language-enabled backbones, yet it remains unclear what changes beyond the usual accuracy--cost trade-off. We revisit this question with 3--RQ analysis in RecogDrive by instantiating the system with a full VLM and vision-only backbones, all under an identical diffusion Transformer planner. RQ1: At the backbone level, the VLM can introduce additional subspaces upon the vision-only backbones. RQ2: This unique subspace leads to a different behavioral in some long-tail scenario: the VLM tends to be more aggressive whereas ViT is more conservative, and each decisively wins on about 2--3% of test scenarios; With an oracle that selects, per scenario, the better trajectory between the VLM and ViT branches, we obtain an upper bound of 93.58 PDMS. RQ3: To fully harness this observation, we propose HybridDriveVLA, which runs both ViT and VLM branches and selects between their endpoint trajectories using a learned scorer, improving PDMS to 92.10. Finally, DualDriveVLA implements a practical fast--slow policy: it runs ViT by default and invokes the VLM only when the scorer's confidence falls below a threshold; calling the VLM on 15% of scenarios achieves 91.00 PDMS while improving throughput by 3.2x. Code will be released.
>
---
#### [new 006] Safe mobility support system using crowd mapping and avoidance route planning using VLM
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决动态环境中安全移动问题。通过融合VLM和GPR生成动态人群密度图，实现避障与避 crowd 的路径规划。**

- **链接: [https://arxiv.org/pdf/2602.10910v1](https://arxiv.org/pdf/2602.10910v1)**

> **作者:** Sena Saito; Kenta Tabata; Renato Miyagusuku; Koichi Ozaki
>
> **摘要:** Autonomous mobile robots offer promising solutions for labor shortages and increased operational efficiency. However, navigating safely and effectively in dynamic environments, particularly crowded areas, remains challenging. This paper proposes a novel framework that integrates Vision-Language Models (VLM) and Gaussian Process Regression (GPR) to generate dynamic crowd-density maps (``Abstraction Maps'') for autonomous robot navigation. Our approach utilizes VLM's capability to recognize abstract environmental concepts, such as crowd densities, and represents them probabilistically via GPR. Experimental results from real-world trials on a university campus demonstrated that robots successfully generated routes avoiding both static obstacles and dynamic crowds, enhancing navigation safety and adaptability.
>
---
#### [new 007] ContactGaussian-WM: Learning Physics-Grounded World Model from Videos
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人感知与建模任务，旨在解决数据稀缺下复杂物理交互建模问题。提出ContactGaussian-WM框架，通过视觉与物理联合学习，提升环境建模精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.11021v1](https://arxiv.org/pdf/2602.11021v1)**

> **作者:** Meizhong Wang; Wanxin Jin; Kun Cao; Lihua Xie; Yiguang Hong
>
> **摘要:** Developing world models that understand complex physical interactions is essential for advancing robotic planning and simulation.However, existing methods often struggle to accurately model the environment under conditions of data scarcity and complex contact-rich dynamic motion.To address these challenges, we propose ContactGaussian-WM, a differentiable physics-grounded rigid-body world model capable of learning intricate physical laws directly from sparse and contact-rich video sequences.Our framework consists of two core components: (1) a unified Gaussian representation for both visual appearance and collision geometry, and (2) an end-to-end differentiable learning framework that differentiates through a closed-form physics engine to infer physical properties from sparse visual observations.Extensive simulations and real-world evaluations demonstrate that ContactGaussian-WM outperforms state-of-the-art methods in learning complex scenarios, exhibiting robust generalization capabilities.Furthermore, we showcase the practical utility of our framework in downstream applications, including data synthesis and real-time MPC.
>
---
#### [new 008] RISE: Self-Improving Robot Policy with Compositional World Model
- **分类: cs.RO**

- **简介: 该论文提出RISE框架，解决机器人在动态操作任务中因执行偏差导致的失败问题。通过组合世界模型和想象强化学习，提升政策性能。**

- **链接: [https://arxiv.org/pdf/2602.11075v1](https://arxiv.org/pdf/2602.11075v1)**

> **作者:** Jiazhi Yang; Kunyang Lin; Jinwei Li; Wencong Zhang; Tianwei Lin; Longyan Wu; Zhizhong Su; Hao Zhao; Ya-Qin Zhang; Li Chen; Ping Luo; Xiangyu Yue; Hongyang Li
>
> **备注:** Project page: https://opendrivelab.com/kai0-rl/
>
> **摘要:** Despite the sustained scaling on model capacity and data acquisition, Vision-Language-Action (VLA) models remain brittle in contact-rich and dynamic manipulation tasks, where minor execution deviations can compound into failures. While reinforcement learning (RL) offers a principled path to robustness, on-policy RL in the physical world is constrained by safety risk, hardware cost, and environment reset. To bridge this gap, we present RISE, a scalable framework of robotic reinforcement learning via imagination. At its core is a Compositional World Model that (i) predicts multi-view future via a controllable dynamics model, and (ii) evaluates imagined outcomes with a progress value model, producing informative advantages for the policy improvement. Such compositional design allows state and value to be tailored by best-suited yet distinct architectures and objectives. These components are integrated into a closed-loop self-improving pipeline that continuously generates imaginary rollouts, estimates advantages, and updates the policy in imaginary space without costly physical interaction. Across three challenging real-world tasks, RISE yields significant improvement over prior art, with more than +35% absolute performance increase in dynamic brick sorting, +45% for backpack packing, and +35% for box closing, respectively.
>
---
#### [new 009] Say, Dream, and Act: Learning Video World Models for Instruction-Driven Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决环境预测与动作规划问题。通过视频生成和对抗蒸馏技术，提升视频预测的时空一致性，增强机器人操作精度。**

- **链接: [https://arxiv.org/pdf/2602.10717v1](https://arxiv.org/pdf/2602.10717v1)**

> **作者:** Songen Gu; Yunuo Cai; Tianyu Wang; Simo Wu; Yanwei Fu
>
> **摘要:** Robotic manipulation requires anticipating how the environment evolves in response to actions, yet most existing systems lack this predictive capability, often resulting in errors and inefficiency. While Vision-Language Models (VLMs) provide high-level guidance, they cannot explicitly forecast future states, and existing world models either predict only short horizons or produce spatially inconsistent frames. To address these challenges, we propose a framework for fast and predictive video-conditioned action. Our approach first selects and adapts a robust video generation model to ensure reliable future predictions, then applies adversarial distillation for fast, few-step video generation, and finally trains an action model that leverages both generated videos and real observations to correct spatial errors. Extensive experiments show that our method produces temporally coherent, spatially accurate video predictions that directly support precise manipulation, achieving significant improvements in embodiment consistency, spatial referring ability, and task completion over existing baselines. Codes & Models will be released.
>
---
#### [new 010] Design, Development, and Use of Maya Robot as an Assistant for the Therapy/Education of Children with Cancer: a Pilot Study
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于医疗辅助机器人研究，旨在通过Maya机器人减轻儿童癌症治疗中的疼痛和焦虑。工作包括设计机器人、进行实验验证其效果。**

- **链接: [https://arxiv.org/pdf/2602.10942v1](https://arxiv.org/pdf/2602.10942v1)**

> **作者:** Alireza Taheri; Minoo Alemi; Elham Ranjkar; Raman Rafatnejad; Ali F. Meghdari
>
> **摘要:** This study centers around the design and implementation of the Maya Robot, a portable elephant-shaped social robot, intended to engage with children undergoing cancer treatment. Initial efforts were devoted to enhancing the robot's facial expression recognition accuracy, achieving a 98% accuracy through deep neural networks. Two subsequent preliminary exploratory experiments were designed to advance the study's objectives. The first experiment aimed to compare pain levels experienced by children during the injection process, with and without the presence of the Maya robot. Twenty-five children, aged 4 to 9, undergoing cancer treatment participated in this counterbalanced study. The paired T-test results revealed a significant reduction in perceived pain when the robot was actively present in the injection room. The second experiment sought to assess perspectives of hospitalized children and their mothers during engagement with Maya through a game. Forty participants, including 20 children aged 4 to 9 and their mothers, were involved. Post Human-Maya Interactions, UTAUT questionnaire results indicated that children experienced significantly less anxiety than their parents during the interaction and game play. Notably, children exhibited higher trust levels in both the robot and the games, presenting a statistically significant difference in trust levels compared to their parents (P-value < 0.05). This preliminary exploratory study highlights the positive impact of utilizing Maya as an assistant for therapy/education in a clinical setting, particularly benefiting children undergoing cancer treatment. The findings underscore the potential of social robots in pediatric healthcare contexts, emphasizing improved pain management and emotional well-being among young patients.
>
---
#### [new 011] Scaling World Model for Hierarchical Manipulation Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在分布外场景下的泛化问题。通过引入分层框架，利用预训练世界模型生成视觉目标，提升低层策略的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10983v1](https://arxiv.org/pdf/2602.10983v1)**

> **作者:** Qian Long; Yueze Wang; Jiaxi Song; Junbo Zhang; Peiyan Li; Wenxuan Wang; Yuqi Wang; Haoyang Li; Shaoxuan Xie; Guocai Yao; Hanbo Zhang; Xinlong Wang; Zhongyuan Wang; Xuguang Lan; Huaping Liu; Xinghang Li
>
> **摘要:** Vision-Language-Action (VLA) models are promising for generalist robot manipulation but remain brittle in out-of-distribution (OOD) settings, especially with limited real-robot data. To resolve the generalization bottleneck, we introduce a hierarchical Vision-Language-Action framework \our{} that leverages the generalization of large-scale pre-trained world model for robust and generalizable VIsual Subgoal TAsk decomposition VISTA. Our hierarchical framework \our{} consists of a world model as the high-level planner and a VLA as the low-level executor. The high-level world model first divides manipulation tasks into subtask sequences with goal images, and the low-level policy follows the textual and visual guidance to generate action sequences. Compared to raw textual goal specification, these synthesized goal images provide visually and physically grounded details for low-level policies, making it feasible to generalize across unseen objects and novel scenarios. We validate both visual goal synthesis and our hierarchical VLA policies in massive out-of-distribution scenarios, and the performance of the same-structured VLA in novel scenarios could boost from 14% to 69% with the guidance generated by the world model. Results demonstrate that our method outperforms previous baselines with a clear margin, particularly in out-of-distribution scenarios. Project page: \href{https://vista-wm.github.io/}{https://vista-wm.github.io}
>
---
#### [new 012] Pitch Angle Control of a Magnetically Actuated Capsule Robot with Nonlinear FEA-based MPC and EKF Multisensory Fusion
- **分类: cs.RO**

- **简介: 该论文属于磁控胶囊机器人控制任务，解决其俯仰角控制问题。通过非线性模型预测控制与传感器融合，实现稳定、快速的俯仰调节。**

- **链接: [https://arxiv.org/pdf/2602.10610v1](https://arxiv.org/pdf/2602.10610v1)**

> **作者:** Chongxun Wang; Zikang Shen; Apoorav Rathore; Akanimoh Udombeh; Harrison Teng; Fangzhou Xia
>
> **备注:** This version is submitted for review at IEEE/ASME Transactions on Mechatronics
>
> **摘要:** Magnetically actuated capsule robots promise minimally invasive diagnosis and therapy in the gastrointestinal (GI) tract, but existing systems largely neglect control of capsule pitch, a degree of freedom critical for contact-rich interaction with inclined gastric walls. This paper presents a nonlinear, model-based framework for magnetic pitch control of an ingestible capsule robot actuated by a four-coil electromagnetic array. Angle-dependent magnetic forces and torques acting on embedded permanent magnets are characterized using three-dimensional finite-element simulations and embedded as lookup tables in a control-oriented rigid-body pitching model with rolling contact and actuator dynamics. A constrained model predictive controller (MPC) is designed to regulate pitch while respecting hardware-imposed current and slew-rate limits. Experiments on a compliant stomach-inspired surface demonstrate robust pitch reorientation from both horizontal and upright configurations, achieving about three to five times faster settling and reduced oscillatory motion than on-off control. Furthermore, an extended Kalman filter (EKF) fusing inertial sensing with intermittent visual measurements enables stable closed-loop control when the camera update rate is reduced from 30 Hz to 1 Hz, emulating clinically realistic imaging constraints. These results establish finite-element-informed MPC with sensor fusion as a scalable strategy for pitch regulation, controlled docking, and future multi-degree-of-freedom capsule locomotion.
>
---
#### [new 013] Biomimetic Mantaray robot toward the underwater autonomous -- Experimental verification of swimming and diving by flapping motion -
- **分类: cs.RO**

- **简介: 该论文属于水下机器人研究，旨在开发仿生魔鬼鱼机器人以实现高效、低扰动的自主探索。通过实验验证其游泳与下潜能力。**

- **链接: [https://arxiv.org/pdf/2602.10904v1](https://arxiv.org/pdf/2602.10904v1)**

> **作者:** Kenta Tabata; Ryosuke Oku; Jun Ito; Renato Miyagusuku; Koichi Ozaki
>
> **摘要:** This study presents the development and experimental verification of a biomimetic manta ray robot for underwater autonomous exploration. Inspired by manta rays, the robot uses flapping motion for propulsion to minimize seabed disturbance and enhance efficiency compared to traditional screw propulsion. The robot features pectoral fins driven by servo motors and a streamlined control box to reduce fluid resistance. The control system, powered by a Raspberry Pi 3B, includes an IMU and pressure sensor for real-time monitoring and control. Experiments in a pool assessed the robot's swimming and diving capabilities. Results show stable swimming and diving motions with PD control. The robot is suitable for applications in environments like aquariums and fish nurseries, requiring minimal disturbance and efficient maneuverability. Our findings demonstrate the potential of bio-inspired robotic designs to improve ecological monitoring and underwater exploration.
>
---
#### [new 014] RADAR: Benchmarking Vision-Language-Action Generalization via Real-World Dynamics, Spatial-Physical Intelligence, and Autonomous Evaluation
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型评估任务，旨在解决真实世界泛化能力不足的问题。通过提出RADAR基准，系统评估模型在物理动态、空间推理和自主评价方面的表现。**

- **链接: [https://arxiv.org/pdf/2602.10980v1](https://arxiv.org/pdf/2602.10980v1)**

> **作者:** Yuhao Chen; Zhihao Zhan; Xiaoxin Lin; Zijian Song; Hao Liu; Qinhan Lyu; Yubo Zu; Xiao Chen; Zhiyuan Liu; Tao Pu; Tianshui Chen; Keze Wang; Liang Lin; Guangrun Wang
>
> **备注:** 12 pages, 11 figures, 3 tables
>
> **摘要:** VLA models have achieved remarkable progress in embodied intelligence; however, their evaluation remains largely confined to simulations or highly constrained real-world settings. This mismatch creates a substantial reality gap, where strong benchmark performance often masks poor generalization in diverse physical environments. We identify three systemic shortcomings in current benchmarking practices that hinder fair and reliable model comparison. (1) Existing benchmarks fail to model real-world dynamics, overlooking critical factors such as dynamic object configurations, robot initial states, lighting changes, and sensor noise. (2) Current protocols neglect spatial--physical intelligence, reducing evaluation to rote manipulation tasks that do not probe geometric reasoning. (3) The field lacks scalable fully autonomous evaluation, instead relying on simplistic 2D metrics that miss 3D spatial structure or on human-in-the-loop systems that are costly, biased, and unscalable. To address these limitations, we introduce RADAR (Real-world Autonomous Dynamics And Reasoning), a benchmark designed to systematically evaluate VLA generalization under realistic conditions. RADAR integrates three core components: (1) a principled suite of physical dynamics; (2) dedicated tasks that explicitly test spatial reasoning and physical understanding; and (3) a fully autonomous evaluation pipeline based on 3D metrics, eliminating the need for human supervision. We apply RADAR to audit multiple state-of-the-art VLA models and uncover severe fragility beneath their apparent competence. Performance drops precipitously under modest physical dynamics, with the expectation of 3D IoU declining from 0.261 to 0.068 under sensor noise. Moreover, models exhibit limited spatial reasoning capability. These findings position RADAR as a necessary bench toward reliable and generalizable real-world evaluation of VLA models.
>
---
#### [new 015] YOR: Your Own Mobile Manipulator for Generalizable Robotics
- **分类: cs.RO; cs.LG**

- **简介: 该论文介绍了一种低成本、模块化的移动操作机器人YOR，旨在解决移动操作平台成本高、通用性差的问题。通过集成多种部件，实现灵活的全身体操控与自主导航。**

- **链接: [https://arxiv.org/pdf/2602.11150v1](https://arxiv.org/pdf/2602.11150v1)**

> **作者:** Manan H Anjaria; Mehmet Enes Erciyes; Vedant Ghatnekar; Neha Navarkar; Haritheja Etukuru; Xiaole Jiang; Kanad Patel; Dhawal Kabra; Nicholas Wojno; Radhika Ajay Prayage; Soumith Chintala; Lerrel Pinto; Nur Muhammad Mahi Shafiullah; Zichen Jeff Cui
>
> **摘要:** Recent advances in robot learning have generated significant interest in capable platforms that may eventually approach human-level competence. This interest, combined with the commoditization of actuators, has propelled growth in low-cost robotic platforms. However, the optimal form factor for mobile manipulation, especially on a budget, remains an open question. We introduce YOR, an open-source, low-cost mobile manipulator that integrates an omnidirectional base, a telescopic vertical lift, and two arms with grippers to achieve whole-body mobility and manipulation. Our design emphasizes modularity, ease of assembly using off-the-shelf components, and affordability, with a bill-of-materials cost under 10,000 USD. We demonstrate YOR's capability by completing tasks that require coordinated whole-body control, bimanual manipulation, and autonomous navigation. Overall, YOR offers competitive functionality for mobile manipulation research at a fraction of the cost of existing platforms. Project website: https://www.yourownrobot.ai/
>
---
#### [new 016] SQ-CBF: Signed Distance Functions for Numerically Stable Superquadric-Based Safety Filtering
- **分类: cs.RO**

- **简介: 该论文属于机器人安全过滤任务，解决传统方法因几何表示不准确导致的安全性不足问题。提出基于超二次曲面的稳定安全过滤框架，使用符号距离函数提升实时安全性。**

- **链接: [https://arxiv.org/pdf/2602.11049v1](https://arxiv.org/pdf/2602.11049v1)**

> **作者:** Haocheng Zhao; Lukas Brunke; Oliver Lagerquist; Siqi Zhou; Angela P. Schoellig
>
> **摘要:** Ensuring safe robot operation in cluttered and dynamic environments remains a fundamental challenge. While control barrier functions provide an effective framework for real-time safety filtering, their performance critically depends on the underlying geometric representation, which is often simplified, leading to either overly conservative behavior or insufficient collision coverage. Superquadrics offer an expressive way to model complex shapes using a few primitives and are increasingly used for robot safety. To integrate this representation into collision avoidance, most existing approaches directly use their implicit functions as barrier candidates. However, we identify a critical but overlooked issue in this practice: the gradients of the implicit SQ function can become severely ill-conditioned, potentially rendering the optimization infeasible and undermining reliable real-time safety filtering. To address this issue, we formulate an SQ-based safety filtering framework that uses signed distance functions as barrier candidates. Since analytical SDFs are unavailable for general SQs, we compute distances using the efficient Gilbert-Johnson-Keerthi algorithm and obtain gradients via randomized smoothing. Extensive simulation and real-world experiments demonstrate consistent collision-free manipulation in cluttered and unstructured scenes, showing robustness to challenging geometries, sensing noise, and dynamic disturbances, while improving task efficiency in teleoperation tasks. These results highlight a pathway toward safety filters that remain precise and reliable under the geometric complexity of real-world environments.
>
---
#### [new 017] Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型在持续学习中的数据依赖和遗忘问题。提出LifeLong-RFT方法，通过强化微调实现高效多任务学习。**

- **链接: [https://arxiv.org/pdf/2602.10503v1](https://arxiv.org/pdf/2602.10503v1)**

> **作者:** Yuan Liu; Haoran Li; Shuai Tian; Yuxing Qin; Yuhui Chen; Yupeng Zheng; Yongzhen Huang; Dongbin Zhao
>
> **摘要:** Pretrained on large-scale and diverse datasets, VLA models demonstrate strong generalization and adaptability as general-purpose robotic policies. However, Supervised Fine-Tuning (SFT), which serves as the primary mechanism for adapting VLAs to downstream domains, requires substantial amounts of task-specific data and is prone to catastrophic forgetting. To address these limitations, we propose LifeLong-RFT, a simple yet effective Reinforcement Fine-Tuning (RFT) strategy for VLA models independent of online environmental feedback and pre-trained reward models. By integrating chunking-level on-policy reinforcement learning with the proposed Multi-Dimensional Process Reward (MDPR) mechanism, LifeLong-RFT quantifies the heterogeneous contributions of intermediate action chunks across three dimensions to facilitate policy optimization. Specifically, (1) the Quantized Action Consistency Reward (QACR) ensures accurate action prediction within the discrete action space; (2) the Continuous Trajectory Alignment Reward (CTAR) aligns decoded continuous action chunks with reference trajectories to ensure precise control; (3) the Format Compliance Reward (FCR) guarantees the structural validity of outputs. Comprehensive experiments across SimplerEnv, LIBERO, and real-world tasks demonstrate that LifeLong-RFT exhibits strong performance in multi-task learning. Furthermore, for continual learning on the LIBERO benchmark, our method achieves a 22% gain in average success rate over SFT, while effectively adapting to new tasks using only 20% of the training data. Overall, our method provides a promising post-training paradigm for VLAs.
>
---
#### [new 018] Adaptive Time Step Flow Matching for Autonomous Driving Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶轨迹规划任务，解决实时性与轨迹质量问题。提出自适应时间步流匹配框架，联合预测周围车辆运动并生成 ego 轨迹，提升平滑性和动态约束遵守。**

- **链接: [https://arxiv.org/pdf/2602.10285v1](https://arxiv.org/pdf/2602.10285v1)**

> **作者:** Ananya Trivedi; Anjian Li; Mohamed Elnoor; Yusuf Umut Ciftci; Avinash Singh; Jovin D'sa; Sangjae Bae; David Isele; Taskin Padir; Faizan M. Tariq
>
> **备注:** Accepted to Intelligent Vehicles Symposium 2026
>
> **摘要:** Autonomous driving requires reasoning about interactions with surrounding traffic. A prevailing approach is large-scale imitation learning on expert driving datasets, aimed at generalizing across diverse real-world scenarios. For online trajectory generation, such methods must operate at real-time rates. Diffusion models require hundreds of denoising steps at inference, resulting in high latency. Consistency models mitigate this issue but rely on carefully tuned noise schedules to capture the multimodal action distributions common in autonomous driving. Adapting the schedule, typically requires expensive retraining. To address these limitations, we propose a framework based on conditional flow matching that jointly predicts future motions of surrounding agents and plans the ego trajectory in real time. We train a lightweight variance estimator that selects the number of inference steps online, removing the need for retraining to balance runtime and imitation learning performance. To further enhance ride quality, we introduce a trajectory post-processing step cast as a convex quadratic program, with negligible computational overhead. Trained on the Waymo Open Motion Dataset, the framework performs maneuvers such as lane changes, cruise control, and navigating unprotected left turns without requiring scenario-specific tuning. Our method maintains a 20 Hz update rate on an NVIDIA RTX 3070 GPU, making it suitable for online deployment. Compared to transformer, diffusion, and consistency model baselines, we achieve improved trajectory smoothness and better adherence to dynamic constraints. Experiment videos and code implementations can be found at https://flow-matching-self-driving.github.io/.
>
---
#### [new 019] Developing Neural Network-Based Gaze Control Systems for Social Robots
- **分类: cs.RO**

- **简介: 该论文属于社会机器人 gaze 控制任务，旨在通过深度学习预测人类在不同社交场景中的注视方向，提升机器人互动能力。**

- **链接: [https://arxiv.org/pdf/2602.10946v1](https://arxiv.org/pdf/2602.10946v1)**

> **作者:** Ramtin Tabatabaei; Alireza Taheri
>
> **摘要:** During multi-party interactions, gaze direction is a key indicator of interest and intent, making it essential for social robots to direct their attention appropriately. Understanding the social context is crucial for robots to engage effectively, predict human intentions, and navigate interactions smoothly. This study aims to develop an empirical motion-time pattern for human gaze behavior in various social situations (e.g., entering, leaving, waving, talking, and pointing) using deep neural networks based on participants' data. We created two video clips-one for a computer screen and another for a virtual reality headset-depicting different social scenarios. Data were collected from 30 participants: 15 using an eye-tracker and 15 using an Oculus Quest 1 headset. Deep learning models, specifically Long Short-Term Memory (LSTM) and Transformers, were used to analyze and predict gaze patterns. Our models achieved 60% accuracy in predicting gaze direction in a 2D animation and 65% accuracy in a 3D animation. Then, the best model was implemented onto the Nao robot; and 36 new participants evaluated its performance. The feedback indicated overall satisfaction, with those experienced in robotics rating the models more favorably.
>
---
#### [new 020] Morphogenetic Assembly and Adaptive Control for Heterogeneous Modular Robots
- **分类: cs.RO**

- **简介: 该论文属于模块化机器人任务，解决异构机器人动态组装与自适应控制问题。提出闭环框架，包含分层规划和GPU加速控制器，实现高效配置生成与实时运动控制。**

- **链接: [https://arxiv.org/pdf/2602.10561v1](https://arxiv.org/pdf/2602.10561v1)**

> **作者:** Chongxi Meng; Da Zhao; Yifei Zhao; Minghao Zeng; Yanmin Zhou; Zhipeng Wang; Bin He
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** This paper presents a closed-loop automation framework for heterogeneous modular robots, covering the full pipeline from morphological construction to adaptive control. In this framework, a mobile manipulator handles heterogeneous functional modules including structural, joint, and wheeled modules to dynamically assemble diverse robot configurations and provide them with immediate locomotion capability. To address the state-space explosion in large-scale heterogeneous reconfiguration, we propose a hierarchical planner: the high-level planner uses a bidirectional heuristic search with type-penalty terms to generate module-handling sequences, while the low level planner employs A* search to compute optimal execution trajectories. This design effectively decouples discrete configuration planning from continuous motion execution. For adaptive motion generation of unknown assembled configurations, we introduce a GPU accelerated Annealing-Variance Model Predictive Path Integral (MPPI) controller. By incorporating a multi stage variance annealing strategy to balance global exploration and local convergence, the controller enables configuration-agnostic, real-time motion control. Large scale simulations show that the type-penalty term is critical for planning robustness in heterogeneous scenarios. Moreover, the greedy heuristic produces plans with lower physical execution costs than the Hungarian heuristic. The proposed annealing-variance MPPI significantly outperforms standard MPPI in both velocity tracking accuracy and control frequency, achieving real time control at 50 Hz. The framework validates the full-cycle process, including module assembly, robot merging and splitting, and dynamic motion generation.
>
---
#### [new 021] Omnidirectional Dual-Arm Aerial Manipulator with Proprioceptive Contact Localization for Landing on Slanted Roofs
- **分类: cs.RO**

- **简介: 该论文属于无人机着陆任务，解决复杂屋顶倾斜面精准检测问题。提出双臂无人机结构与基于力矩观测的接触定位方法，实现无视觉条件下的倾斜面盲降。**

- **链接: [https://arxiv.org/pdf/2602.10703v1](https://arxiv.org/pdf/2602.10703v1)**

> **作者:** Martijn B. J. Brummelhuis; Nathan F. Lepora; Salua Hamaza
>
> **备注:** Accepted into 2026 International Conference on Robotics and Automation (ICRA) in Vienna
>
> **摘要:** Operating drones in urban environments often means they need to land on rooftops, which can have different geometries and surface irregularities. Accurately detecting roof inclination using conventional sensing methods, such as vision-based or acoustic techniques, can be unreliable, as measurement quality is strongly influenced by external factors including weather conditions and surface materials. To overcome these challenges, we propose a novel unmanned aerial manipulator morphology featuring a dual-arm aerial manipulator with an omnidirectional 3D workspace and extended reach. Building on this design, we develop a proprioceptive contact detection and contact localization strategy based on a momentum-based torque observer. This enables the UAM to infer the inclination of slanted surfaces blindly - through physical interaction - prior to touchdown. We validate the approach in flight experiments, demonstrating robust landings on surfaces with inclinations of up to 30.5 degrees and achieving an average surface inclination estimation error of 2.87 degrees over 9 experiments at different incline angles.
>
---
#### [new 022] SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes
- **分类: cs.RO; cs.AI; cs.CV; cs.GR**

- **简介: 该论文提出SceneSmith，用于生成逼真室内场景的模拟环境，解决现有场景缺乏多样性和物理复杂性的问题。通过自然语言生成高质量仿真场景，提升机器人训练效果。**

- **链接: [https://arxiv.org/pdf/2602.09153v1](https://arxiv.org/pdf/2602.09153v1)**

> **作者:** Nicholas Pfaff; Thomas Cohn; Sergey Zakharov; Rick Cory; Russ Tedrake
>
> **备注:** Project page: https://scenesmith.github.io/
>
> **摘要:** Simulation has become a key tool for training and evaluating home robots at scale, yet existing environments fail to capture the diversity and physical complexity of real indoor spaces. Current scene synthesis methods produce sparsely furnished rooms that lack the dense clutter, articulated furniture, and physical properties essential for robotic manipulation. We introduce SceneSmith, a hierarchical agentic framework that generates simulation-ready indoor environments from natural language prompts. SceneSmith constructs scenes through successive stages$\unicode{x2013}$from architectural layout to furniture placement to small object population$\unicode{x2013}$each implemented as an interaction among VLM agents: designer, critic, and orchestrator. The framework tightly integrates asset generation through text-to-3D synthesis for static objects, dataset retrieval for articulated objects, and physical property estimation. SceneSmith generates 3-6x more objects than prior methods, with <2% inter-object collisions and 96% of objects remaining stable under physics simulation. In a user study with 205 participants, it achieves 92% average realism and 91% average prompt faithfulness win rates against baselines. We further demonstrate that these environments can be used in an end-to-end pipeline for automatic robot policy evaluation.
>
---
#### [new 023] ReSPEC: A Framework for Online Multispectral Sensor Reconfiguration in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文提出ReSPEC框架，解决动态环境中多光谱传感器配置静态的问题，通过强化学习实现传感器实时自适应调整，优化资源使用。**

- **链接: [https://arxiv.org/pdf/2602.10547v1](https://arxiv.org/pdf/2602.10547v1)**

> **作者:** Yanchen Liu; Yuang Fan; Minghui Zhao; Xiaofan Jiang
>
> **备注:** 8 pages, 4 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Multi-sensor fusion is central to robust robotic perception, yet most existing systems operate under static sensor configurations, collecting all modalities at fixed rates and fidelity regardless of their situational utility. This rigidity wastes bandwidth, computation, and energy, and prevents systems from prioritizing sensors under challenging conditions such as poor lighting or occlusion. Recent advances in reinforcement learning (RL) and modality-aware fusion suggest the potential for adaptive perception, but prior efforts have largely focused on re-weighting features at inference time, ignoring the physical cost of sensor data collection. We introduce a framework that unifies sensing, learning, and actuation into a closed reconfiguration loop. A task-specific detection backbone extracts multispectral features (e.g. RGB, IR, mmWave, depth) and produces quantitative contribution scores for each modality. These scores are passed to an RL agent, which dynamically adjusts sensor configurations, including sampling frequency, resolution, sensing range, and etc., in real time. Less informative sensors are down-sampled or deactivated, while critical sensors are sampled at higher fidelity as environmental conditions evolve. We implement and evaluate this framework on a mobile rover, showing that adaptive control reduces GPU load by 29.3\% with only a 5.3\% accuracy drop compared to a heuristic baseline. These results highlight the potential of resource-aware adaptive sensing for embedded robotic platforms.
>
---
#### [new 024] A receding-horizon multi-contact motion planner for legged robots in challenging environments
- **分类: cs.RO**

- **简介: 该论文属于腿式机器人运动规划任务，解决复杂环境中多接触点路径规划问题。提出一种滚动时域方法，可实时重规划并同步生成接触点与全身轨迹，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2602.11113v1](https://arxiv.org/pdf/2602.11113v1)**

> **作者:** Daniel S. J. Derwent; Simon Watson; Bruno V. Adorno
>
> **备注:** Submitted to Robotics and Autonomous Systems For supplementary video, see https://www.youtube.com/watch?v=RJp8DCmhDa4
>
> **摘要:** We present a novel receding-horizon multi-contact motion planner for legged robots in challenging scenarios, able to plan motions such as chimney climbing, navigating very narrow passages or crossing large gaps. Our approach adds new capabilities to the state of the art, including the ability to reactively re-plan in response to new information, and planning contact locations and whole-body trajectories simultaneously, simplifying the implementation and removing the need for post-processing or complex multi-stage approaches. Our method is more resistant to local minima problems than other potential field based approaches, and our quadratic-program-based posture generator returns nodes more quickly than those of existing algorithms. Rigorous statistical analysis shows that, with short planning horizons (e.g., one step ahead), our planner is faster than the state-of-the-art across all scenarios tested (between 45% and 98% faster on average, depending on the scenario), while planning less efficient motions (requiring 5% fewer to 700% more stance changes on average). In all but one scenario (Chimney Walking), longer planning horizons (e.g., four steps ahead) extended the average planning times (between 73% faster and 400% slower than the state-of-the-art) but resulted in higher quality motion plans (between 8% more and 47% fewer stance changes than the state-of-the-art).
>
---
#### [new 025] Stability Analysis of Geometric Control for a Canonical Class of Underactuated Aerial Vehicles with Spurious Forces
- **分类: cs.RO; math.OC**

- **简介: 该论文属于控制理论任务，旨在解决受干扰力矩影响的飞行器稳定性问题。通过构建模型并进行李雅普诺夫分析，证明了悬停平衡点的局部指数稳定。**

- **链接: [https://arxiv.org/pdf/2602.10961v1](https://arxiv.org/pdf/2602.10961v1)**

> **作者:** Simone Orelli; Mirko Mizzoni; Antonio Franchi
>
> **摘要:** Standard geometric control relies on force-moment decoupling, an assumption that breaks down in many aerial platforms due to spurious forces naturally induced by control moments. While strategies for such coupled systems have been validated experimentally, a rigorous theoretical certification of their stability is currently missing. This work fills this gap by providing the first formal stability analysis for a generic class of floating rigid bodies subject to spurious forces. We introduce a canonical model and construct a Lyapunov-based proof establishing local exponential stability of the hovering equilibrium. Crucially, the analysis explicitly addresses the structural challenges - specifically the induced non-minimum-phase behavior - that prevent the application of standard cascade arguments.
>
---
#### [new 026] Flow-Enabled Generalization to Human Demonstrations in Few-Shot Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于少样本模仿学习任务，旨在解决机器人从少量人类示范中学习技能的问题。通过引入场景流和点云条件策略，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10594v1](https://arxiv.org/pdf/2602.10594v1)**

> **作者:** Runze Tang; Penny Sweetser
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Imitation Learning (IL) enables robots to learn complex skills from demonstrations without explicit task modeling, but it typically requires large amounts of demonstrations, creating significant collection costs. Prior work has investigated using flow as an intermediate representation to enable the use of human videos as a substitute, thereby reducing the amount of required robot demonstrations. However, most prior work has focused on the flow, either on the object or on specific points of the robot/hand, which cannot describe the motion of interaction. Meanwhile, relying on flow to achieve generalization to scenarios observed only in human videos remains limited, as flow alone cannot capture precise motion details. Furthermore, conditioning on scene observation to produce precise actions may cause the flow-conditioned policy to overfit to training tasks and weaken the generalization indicated by the flow. To address these gaps, we propose SFCrP, which includes a Scene Flow prediction model for Cross-embodiment learning (SFCr) and a Flow and Cropped point cloud conditioned Policy (FCrP). SFCr learns from both robot and human videos and predicts any point trajectories. FCrP follows the general flow motion and adjusts the action based on observations for precision tasks. Our method outperforms SOTA baselines across various real-world task settings, while also exhibiting strong spatial and instance generalization to scenarios seen only in human videos.
>
---
#### [new 027] LocoVLM: Grounding Vision and Language for Adapting Versatile Legged Locomotion Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决腿式机器人无法理解高级语义指令的问题。通过融合视觉语言模型和大语言模型，实现环境语义的实时适应与技能生成。**

- **链接: [https://arxiv.org/pdf/2602.10399v1](https://arxiv.org/pdf/2602.10399v1)**

> **作者:** I Made Aswin Nahrendra; Seunghyun Lee; Dongkyu Lee; Hyun Myung
>
> **备注:** Project page: https://locovlm.github.io
>
> **摘要:** Recent advances in legged locomotion learning are still dominated by the utilization of geometric representations of the environment, limiting the robot's capability to respond to higher-level semantics such as human instructions. To address this limitation, we propose a novel approach that integrates high-level commonsense reasoning from foundation models into the process of legged locomotion adaptation. Specifically, our method utilizes a pre-trained large language model to synthesize an instruction-grounded skill database tailored for legged robots. A pre-trained vision-language model is employed to extract high-level environmental semantics and ground them within the skill database, enabling real-time skill advisories for the robot. To facilitate versatile skill control, we train a style-conditioned policy capable of generating diverse and robust locomotion skills with high fidelity to specified styles. To the best of our knowledge, this is the first work to demonstrate real-time adaptation of legged locomotion using high-level reasoning from environmental semantics and instructions with instruction-following accuracy of up to 87% without the need for online query to on-the-cloud foundation models.
>
---
#### [new 028] 3D-Printed Anisotropic Soft Magnetic Coating for Directional Rolling of a Magnetically Actuated Capsule Robot
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决传统胶囊机器人空间受限的问题。通过3D打印软磁涂层实现定向滚动，提升其在体内的运动能力与功能性。**

- **链接: [https://arxiv.org/pdf/2602.10688v1](https://arxiv.org/pdf/2602.10688v1)**

> **作者:** Jin Zhou; Chongxun Wang; Zikang Shen; Fangzhou Xia
>
> **备注:** Submitted for review at IEEE/ASME Advanced Intelligenet Mechatronics Conference (2026)
>
> **摘要:** Capsule robots are promising tools for minimally invasive diagnostics and therapy, with applications from gastrointestinal endoscopy to targeted drug delivery and biopsy sampling. Conventional magnetic capsule robots embed bulky permanent magnets at both ends, reducing the usable cavity by about 10-20 mm and limiting integration of functional modules. We propose a compact, 3D-printed soft capsule robot with a magnetic coating that replaces internal magnets, enabling locomotion via a thin, functional shell while preserving the entire interior cavity as a continuous volume for medical payloads. The compliant silicone-magnetic composite also improves swallowability, even with a slightly larger capsule size. Magnetostatic simulations and experiments confirm that programmed NSSN/SNNS pole distributions provide strong anisotropy and reliable torque generation, enabling stable bidirectional rolling, omnidirectional steering, climbing on 7.5 degree inclines, and traversal of 5 mm protrusions. Rolling motion is sustained when the magnetic field at the capsule reaches at least 0.3 mT, corresponding to an effective actuation depth of 30 mm in our setup. Future work will optimize material composition, coating thickness, and magnetic layouts to enhance force output and durability, while next-generation robotic-arm-based field generators with closed-loop feedback will address nonlinearities and expand maneuverability. Together, these advances aim to transition coating-based capsule robots toward reliable clinical deployment and broaden their applications in minimally invasive diagnostics and therapy.
>
---
#### [new 029] Data-Efficient Hierarchical Goal-Conditioned Reinforcement Learning via Normalizing Flows
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于强化学习领域，解决长周期任务中数据效率低和策略表达能力弱的问题。提出NF-HIQL框架，利用归一化流提升策略表达，增强泛化与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.11142v1](https://arxiv.org/pdf/2602.11142v1)**

> **作者:** Shaswat Garg; Matin Moezzi; Brandon Da Silva
>
> **备注:** 9 pages, 3 figures, IEEE International Conference on Robotics and Automation 2026
>
> **摘要:** Hierarchical goal-conditioned reinforcement learning (H-GCRL) provides a powerful framework for tackling complex, long-horizon tasks by decomposing them into structured subgoals. However, its practical adoption is hindered by poor data efficiency and limited policy expressivity, especially in offline or data-scarce regimes. In this work, Normalizing flow-based hierarchical implicit Q-learning (NF-HIQL), a novel framework that replaces unimodal gaussian policies with expressive normalizing flow policies at both the high- and low-levels of the hierarchy is introduced. This design enables tractable log-likelihood computation, efficient sampling, and the ability to model rich multimodal behaviors. New theoretical guarantees are derived, including explicit KL-divergence bounds for Real-valued non-volume preserving (RealNVP) policies and PAC-style sample efficiency results, showing that NF-HIQL preserves stability while improving generalization. Empirically, NF-HIQL is evaluted across diverse long-horizon tasks in locomotion, ball-dribbling, and multi-step manipulation from OGBench. NF-HIQL consistently outperforms prior goal-conditioned and hierarchical baselines, demonstrating superior robustness under limited data and highlighting the potential of flow-based architectures for scalable, data-efficient hierarchical reinforcement learning.
>
---
#### [new 030] Digging for Data: Experiments in Rock Pile Characterization Using Only Proprioceptive Sensing in Excavation
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于岩石碎片化分析任务，旨在通过挖掘过程中的本体感知数据估计岩堆颗粒大小，解决传统依赖外部传感器的局限性。**

- **链接: [https://arxiv.org/pdf/2602.11082v1](https://arxiv.org/pdf/2602.11082v1)**

> **作者:** Unal Artan; Martin Magnusson; Joshua A. Marshall
>
> **备注:** Accepted for publication in the IEEE Transactions on Field Robotics
>
> **摘要:** Characterization of fragmented rock piles is a fundamental task in the mining and quarrying industries, where rock is fragmented by blasting, transported using wheel loaders, and then sent for further processing. This field report studies a novel method for estimating the relative particle size of fragmented rock piles from only proprioceptive data collected while digging with a wheel loader. Rather than employ exteroceptive sensors (e.g., cameras or LiDAR sensors) to estimate rock particle sizes, the studied method infers rock fragmentation from an excavator's inertial response during excavation. This paper expands on research that postulated the use of wavelet analysis to construct a unique feature that is proportional to the level of rock fragmentation. We demonstrate through extensive field experiments that the ratio of wavelet features, constructed from data obtained by excavating in different rock piles with different size distributions, approximates the ratio of the mean particle size of the two rock piles. Full-scale excavation experiments were performed with a battery electric, 18-tonne capacity, load-haul-dump (LHD) machine in representative conditions in an operating quarry. The relative particle size estimates generated with the proposed sensing methodology are compared with those obtained from both a vision-based fragmentation analysis tool and from sieving of sampled materials.
>
---
#### [new 031] A Unified Experimental Architecture for Informative Path Planning: from Simulation to Deployment with GuadalPlanner
- **分类: cs.RO; cs.LG; cs.SE**

- **简介: 该论文属于自主车辆路径规划任务，解决仿真与实际部署间迁移性差的问题，提出统一架构GuadalPlanner，实现算法在不同环境下的一致评估与部署。**

- **链接: [https://arxiv.org/pdf/2602.10702v1](https://arxiv.org/pdf/2602.10702v1)**

> **作者:** Alejandro Mendoza Barrionuevo; Dame Seck Diop; Alejandro Casado Pérez; Daniel Gutiérrez Reina; Sergio L. Toral Marín; Samuel Yanes Luis
>
> **摘要:** The evaluation of informative path planning algorithms for autonomous vehicles is often hindered by fragmented execution pipelines and limited transferability between simulation and real-world deployment. This paper introduces a unified architecture that decouples high-level decision-making from vehicle-specific control, enabling algorithms to be evaluated consistently across different abstraction levels without modification. The proposed architecture is realized through GuadalPlanner, which defines standardized interfaces between planning, sensing, and vehicle execution. It is an open and extensible research tool that supports discrete graph-based environments and interchangeable planning strategies, and is built upon widely adopted robotics technologies, including ROS2, MAVLink, and MQTT. Its design allows the same algorithmic logic to be deployed in fully simulated environments, software-in-the-loop configurations, and physical autonomous vehicles using an identical execution pipeline. The approach is validated through a set of experiments, including real-world deployment on an autonomous surface vehicle performing water quality monitoring with real-time sensor feedback.
>
---
#### [new 032] Free-Flying Crew Cooperative Robots on the ISS: A Joint Review of Astrobee, CIMON, and Int-Ball Operations
- **分类: cs.RO**

- **简介: 论文综述国际空间站上的自由飞行机器人Astrobee、CIMON和Int-Ball，分析其设计与操作经验，旨在为未来太空机器人开发提供参考。**

- **链接: [https://arxiv.org/pdf/2602.10686v1](https://arxiv.org/pdf/2602.10686v1)**

> **作者:** Seiko Piotr Yamaguchi; Andres Mora Vargas; Till Eisenberg; Christian Rogon; Tatsuya Yamamoto; Shona Inoue; Christoph Kössl; Brian Coltin; Trey Smith; Jose V. Benavides
>
> **备注:** Author's version of a manuscript accepted at the 2025 International Conference on Space Robotics (iSpaRo25). (c)IEEE
>
> **摘要:** Intra-vehicular free-flying robots are anticipated to support various work in human spaceflight while working side-by-side with astronauts. Such example of robots includes NASA's Astrobee, DLR's CIMON, and JAXA's Int-Ball, which are deployed on the International Space Station. This paper presents the first joint analyses of these robot's shared experiences, co-authored by their development and operation team members. Despite the different origins and design philosophies, the development and operations of these platforms encountered various convergences. Hence, this paper presents a detailed overview of these robots, presenting their objectives, design, and onboard operations. Hence, joint lessons learned across the lifecycle are presented, from design to on-orbit operations. These lessons learned are anticipated to serve for future development and research as design recommendations.
>
---
#### [new 033] Solving Geodesic Equations with Composite Bernstein Polynomials for Trajectory Planning
- **分类: cs.RO; math.OC**

- **简介: 该论文属于轨迹规划任务，解决复杂环境中自主系统路径生成问题。采用复合伯恩斯坦多项式方法，实现连续、安全且高效的轨迹优化。**

- **链接: [https://arxiv.org/pdf/2602.10365v1](https://arxiv.org/pdf/2602.10365v1)**

> **作者:** Nick Gorman; Gage MacLin; Maxwell Hammond; Venanzio Cichella
>
> **备注:** Accepted for the 2026 IEEE Aerospace Conference
>
> **摘要:** This work presents a trajectory planning method based on composite Bernstein polynomials for autonomous systems navigating complex environments. The method is implemented in a symbolic optimization framework that enables continuous paths and precise control over trajectory shape. Trajectories are planned over a cost surface that encodes obstacles as continuous fields rather than discrete boundaries. Regions near obstacles are assigned higher costs, naturally encouraging the trajectory to maintain a safe distance while still allowing efficient routing through constrained spaces. The use of composite Bernstein polynomials preserves continuity while enabling fine control over local curvature to satisfy geodesic constraints. The symbolic representation supports exact derivatives, improving optimization efficiency. The method applies to both two- and three-dimensional environments and is suitable for ground, aerial, underwater, and space systems. In spacecraft trajectory planning, for example, it enables the generation of continuous, dynamically feasible trajectories with high numerical efficiency, making it well suited for orbital maneuvers, rendezvous and proximity operations, cluttered gravitational environments, and planetary exploration missions with limited onboard computational resources. Demonstrations show that the approach efficiently generates smooth, collision-free paths in scenarios with multiple obstacles, maintaining clearance without extensive sampling or post-processing. The optimization incorporates three constraint types: (1) a Gaussian surface inequality enforcing minimum obstacle clearance; (2) geodesic equations guiding the path along locally efficient directions on the cost surface; and (3) boundary constraints enforcing fixed start and end conditions. The method can serve as a standalone planner or as an initializer for more complex motion planning problems.
>
---
#### [new 034] Co-jump: Cooperative Jumping with Quadrupedal Robots via Multi-Agent Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究合作跳跃任务，解决单个机器人跳跃能力受限的问题，通过多智能体强化学习实现两足机器人协同跳跃，提升整体性能。**

- **链接: [https://arxiv.org/pdf/2602.10514v1](https://arxiv.org/pdf/2602.10514v1)**

> **作者:** Shihao Dong; Yeke Chen; Zeren Luo; Jiahui Zhang; Bowen Xu; Jinghan Lin; Yimin Han; Ji Ma; Zhiyou Yu; Yudong Zhao; Peng Lu
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** While single-agent legged locomotion has witnessed remarkable progress, individual robots remain fundamentally constrained by physical actuation limits. To transcend these boundaries, we introduce Co-jump, a cooperative task where two quadrupedal robots synchronize to execute jumps far beyond their solo capabilities. We tackle the high-impulse contact dynamics of this task under a decentralized setting, achieving synchronization without explicit communication or pre-specified motion primitives. Our framework leverages Multi-Agent Proximal Policy Optimization (MAPPO) enhanced by a progressive curriculum strategy, which effectively overcomes the sparse-reward exploration challenges inherent in mechanically coupled systems. We demonstrate robust performance in simulation and successful transfer to physical hardware, executing multi-directional jumps onto platforms up to 1.5 m in height. Specifically, one of the robots achieves a foot-end elevation of 1.1 m, which represents a 144% improvement over the 0.45 m jump height of a standalone quadrupedal robot, demonstrating superior vertical performance. Notably, this precise coordination is achieved solely through proprioceptive feedback, establishing a foundation for communication-free collaborative locomotion in constrained environments.
>
---
#### [new 035] Lie Group Variational Integrator for the Geometrically Exact Rod with Circular Cross-Section Incorporating Cross-Sectional Deformation
- **分类: eess.SY; cs.RO; math.NA**

- **简介: 该论文属于计算力学任务，旨在解决几何精确杆的运动建模问题。通过引入Lie群变分积分方法，建立考虑截面变形的离散模型，确保体积守恒与能量稳定性。**

- **链接: [https://arxiv.org/pdf/2602.10963v1](https://arxiv.org/pdf/2602.10963v1)**

> **作者:** Srishti Siddharth; Vivek Natarajan; Ravi N. Banavar
>
> **备注:** Submitted to: Computers and Mathematics with Applications
>
> **摘要:** In this paper, we derive the continuous space-time equations of motion of a three-dimensional geometrically exact rod, or the Cosserat rod, incorporating planar cross-sectional deformation. We then adopt the Lie group variational integrator technique to obtain a discrete model of the rod incorporating both rotational motion and cross-sectional deformation as well. The resulting discrete model possesses several desirable features: it ensures volume conservation of the discrete elements by considering cross-sectional deformation through a local dilatation factor, it demonstrates the beneficial properties associated with the variational integrator technique, such as the preservation of the rotational configuration, and energy conservation with a bounded error. An exhaustive set of numerical results under various initial conditions of the rod demonstrates the efficacy of the model in replicating the physics of the system.
>
---
#### [new 036] RadarEye: Robust Liquid Level Tracking Using mmWave Radar in Robotic Pouring
- **分类: eess.SP; cs.RO**

- **简介: 该论文提出RadarEye，用于机器人倒液过程中的液体水平跟踪。解决视觉系统在透明液体操作中的感知难题，通过毫米波雷达实现高精度、实时的液位估计与跟踪。**

- **链接: [https://arxiv.org/pdf/2602.10417v1](https://arxiv.org/pdf/2602.10417v1)**

> **作者:** Hongyu Deng; He Chen
>
> **备注:** To appear in IEEE ICASSP 2026
>
> **摘要:** Transparent liquid manipulation in robotic pouring remains challenging for perception systems: specular/refraction effects and lighting variability degrade visual cues, undermining reliable level estimation. To address this challenge, we introduce RadarEye, a real-time mmWave radar signal processing pipeline for robust liquid level estimation and tracking during the whole pouring process. RadarEye integrates (i) a high-resolution range-angle beamforming module for liquid level sensing and (ii) a physics-informed mid-pour tracker that suppresses multipath to maintain lock on the liquid surface despite stream-induced clutter and source container reflections. The pipeline delivers sub-millisecond latency. In real-robot water-pouring experiments, RadarEye achieves a 0.35 cm median absolute height error at 0.62 ms per update, substantially outperforming vision and ultrasound baselines.
>
---
#### [new 037] Min-Sum Uniform Coverage Problem by Autonomous Mobile Robots
- **分类: cs.DC; cs.RO**

- **简介: 该论文研究自主机器人在直线和圆上实现最小总移动距离的均匀覆盖问题，提出分布式算法解决此任务。**

- **链接: [https://arxiv.org/pdf/2602.11125v1](https://arxiv.org/pdf/2602.11125v1)**

> **作者:** Animesh Maiti; Abhinav Chakraborty; Bibhuti Das; Subhash Bhagat; Krishnendu Mukhopadhyaya
>
> **摘要:** We study the \textit{min-sum uniform coverage} problem for a swarm of $n$ mobile robots on a given finite line segment and on a circle having finite positive radius, where the circle is given as an input. The robots must coordinate their movements to reach a uniformly spaced configuration that minimizes the total distance traveled by all robots. The robots are autonomous, anonymous, identical, and homogeneous, and operate under the \textit{Look-Compute-Move} (LCM) model with \textit{non-rigid} motion controlled by a fair asynchronous scheduler. They are oblivious and silent, possessing neither persistent memory nor a means of explicit communication. In the \textbf{line-segment setting}, the \textit{min-sum uniform coverage} problem requires placing the robots at uniformly spaced points along the segment so as to minimize the total distance traveled by all robots. In the \textbf{circle setting} for this problem, the robots have to arrange themselves uniformly around the given circle to form a regular $n$-gon. There is no fixed orientation or designated starting vertex, and the goal is to minimize the total distance traveled by all the robots. We present a deterministic distributed algorithm that achieves uniform coverage in the line-segment setting with minimum total movement cost. For the circle setting, we characterize all initial configurations for which the \textit{min-sum uniform coverage} problem is deterministically unsolvable under the considered robot model. For all the other remaining configurations, we provide a deterministic distributed algorithm that achieves uniform coverage while minimizing the total distance traveled. These results characterize the deterministic solvability of min-sum coverage for oblivious robots and achieve optimal cost whenever solvable.
>
---
#### [new 038] From Steering to Pedalling: Do Autonomous Driving VLMs Generalize to Cyclist-Assistive Spatial Perception and Planning?
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于交通感知与决策任务，旨在解决自动驾驶模型在骑行者辅助场景下的泛化能力问题。通过构建CyclingVQA基准，评估模型在骑行者视角下的空间理解和交通规则推理能力。**

- **链接: [https://arxiv.org/pdf/2602.10771v1](https://arxiv.org/pdf/2602.10771v1)**

> **作者:** Krishna Kanth Nakka; Vedasri Nakka
>
> **备注:** Preprint
>
> **摘要:** Cyclists often encounter safety-critical situations in urban traffic, highlighting the need for assistive systems that support safe and informed decision-making. Recently, vision-language models (VLMs) have demonstrated strong performance on autonomous driving benchmarks, suggesting their potential for general traffic understanding and navigation-related reasoning. However, existing evaluations are predominantly vehicle-centric and fail to assess perception and reasoning from a cyclist-centric viewpoint. To address this gap, we introduce CyclingVQA, a diagnostic benchmark designed to probe perception, spatio-temporal understanding, and traffic-rule-to-lane reasoning from a cyclist's perspective. Evaluating 31+ recent VLMs spanning general-purpose, spatially enhanced, and autonomous-driving-specialized models, we find that current models demonstrate encouraging capabilities, while also revealing clear areas for improvement in cyclist-centric perception and reasoning, particularly in interpreting cyclist-specific traffic cues and associating signs with the correct navigational lanes. Notably, several driving-specialized models underperform strong generalist VLMs, indicating limited transfer from vehicle-centric training to cyclist-assistive scenarios. Finally, through systematic error analysis, we identify recurring failure modes to guide the development of more effective cyclist-assistive intelligent systems.
>
---
#### [new 039] Enhancing Predictability of Multi-Tenant DNN Inference for Autonomous Vehicles' Perception
- **分类: cs.CV; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶感知任务，旨在提升多租户DNN推理的可预测性。通过动态选择关键帧和感兴趣区域，减少计算量并保持精度，解决资源受限下的实时性问题。**

- **链接: [https://arxiv.org/pdf/2602.11004v1](https://arxiv.org/pdf/2602.11004v1)**

> **作者:** Liangkai Liu; Kang G. Shin; Jinkyu Lee; Chengmo Yang; Weisong Shi
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Autonomous vehicles (AVs) rely on sensors and deep neural networks (DNNs) to perceive their surrounding environment and make maneuver decisions in real time. However, achieving real-time DNN inference in the AV's perception pipeline is challenging due to the large gap between the computation requirement and the AV's limited resources. Most, if not all, of existing studies focus on optimizing the DNN inference time to achieve faster perception by compressing the DNN model with pruning and quantization. In contrast, we present a Predictable Perception system with DNNs (PP-DNN) that reduce the amount of image data to be processed while maintaining the same level of accuracy for multi-tenant DNNs by dynamically selecting critical frames and regions of interest (ROIs). PP-DNN is based on our key insight that critical frames and ROIs for AVs vary with the AV's surrounding environment. However, it is challenging to identify and use critical frames and ROIs in multi-tenant DNNs for predictable inference. Given image-frame streams, PP-DNN leverages an ROI generator to identify critical frames and ROIs based on the similarities of consecutive frames and traffic scenarios. PP-DNN then leverages a FLOPs predictor to predict multiply-accumulate operations (MACs) from the dynamic critical frames and ROIs. The ROI scheduler coordinates the processing of critical frames and ROIs with multiple DNN models. Finally, we design a detection predictor for the perception of non-critical frames. We have implemented PP-DNN in an ROS-based AV pipeline and evaluated it with the BDD100K and the nuScenes dataset. PP-DNN is observed to significantly enhance perception predictability, increasing the number of fusion frames by up to 7.3x, reducing the fusion delay by >2.6x and fusion-delay variations by >2.3x, improving detection completeness by 75.4% and the cost-effectiveness by up to 98% over the baseline.
>
---
#### [new 040] An Ontology-driven Dynamic Knowledge Base for Uninhabited Ground Vehicles
- **分类: cs.MA; cs.DB; cs.RO**

- **简介: 该论文属于智能系统任务，旨在解决UGV在动态环境中信息不足的问题。通过构建本体驱动的动态知识库，实现任务中的实时情境数据更新，提升自主决策与态势感知能力。**

- **链接: [https://arxiv.org/pdf/2602.10555v1](https://arxiv.org/pdf/2602.10555v1)**

> **作者:** Hsan Sandar Win; Andrew Walters; Cheng-Chew Lim; Daniel Webber; Seth Leslie; Tan Doan
>
> **备注:** 10 pages, 11 figures, 2025 Australasian Conference on Robotics and Automation (ACRA 2025)
>
> **摘要:** In this paper, the concept of Dynamic Contextual Mission Data (DCMD) is introduced to develop an ontology-driven dynamic knowledge base for Uninhabited Ground Vehicles (UGVs) at the tactical edge. The dynamic knowledge base with DCMD is added to the UGVs to: support enhanced situation awareness; improve autonomous decision making; and facilitate agility within complex and dynamic environments. As UGVs are heavily reliant on the a priori information added pre-mission, unexpected occurrences during a mission can cause identification ambiguities and require increased levels of user input. Updating this a priori information with contextual information can help UGVs realise their full potential. To address this, the dynamic knowledge base was designed using an ontology-driven representation, supported by near real-time information acquisition and analysis, to provide in-mission on-platform DCMD updates. This was implemented on a team of four UGVs that executed a laboratory based surveillance mission. The results showed that the ontology-driven dynamic representation of the UGV operational environment was machine actionable, producing contextual information to support a successful and timely mission, and contributed directly to the situation awareness.
>
---
#### [new 041] Multi-UAV Trajectory Optimization for Bearing-Only Localization in GPS Denied Environments
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于多无人机协同定位任务，解决GPS拒止环境下目标定位问题。通过优化轨迹，使用固定相机的无人机与水面舰船协作，提升定位精度并降低系统复杂度。**

- **链接: [https://arxiv.org/pdf/2602.11116v1](https://arxiv.org/pdf/2602.11116v1)**

> **作者:** Alfonso Sciacchitano; Liraz Mudrik; Sean Kragelund; Isaac Kaminer
>
> **备注:** 38 pages, 7 figure, and 6 tables
>
> **摘要:** Accurate localization of maritime targets by unmanned aerial vehicles (UAVs) remains challenging in GPS-denied environments. UAVs equipped with gimballed electro-optical sensors are typically used to localize targets, however, reliance on these sensors increases mechanical complexity, cost, and susceptibility to single-point failures, limiting scalability and robustness in multi-UAV operations. This work presents a new trajectory optimization framework that enables cooperative target localization using UAVs with fixed, non-gimballed cameras operating in coordination with a surface vessel. This estimation-aware optimization generates dynamically feasible trajectories that explicitly account for mission constraints, platform dynamics, and out-of-frame events. Estimation-aware trajectories outperform heuristic paths by reducing localization error by more than a factor of two, motivating their use in cooperative operations. Results further demonstrate that coordinated UAVs with fixed, non-gimballed cameras achieve localization accuracy that meets or exceeds that of single gimballed systems, while substantially lowering system complexity and cost, enabling scalability, and enhancing mission resilience.
>
---
#### [new 042] End-to-End LiDAR optimization for 3D point cloud registration
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D点云配准任务，解决LiDAR设计与下游任务脱节的问题。通过动态调整传感器参数，联合优化采集与配准过程，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.10492v1](https://arxiv.org/pdf/2602.10492v1)**

> **作者:** Siddhant Katyan; Marc-André Gardner; Jean-François Lalonde
>
> **备注:** 36th British Machine Vision Conference 2025, {BMVC} 2025, Sheffield, UK, November 24-27, 2025. Project page: https://lvsn.github.io/e2e-lidar-registration/
>
> **摘要:** LiDAR sensors are a key modality for 3D perception, yet they are typically designed independently of downstream tasks such as point cloud registration. Conventional registration operates on pre-acquired datasets with fixed LiDAR configurations, leading to suboptimal data collection and significant computational overhead for sampling, noise filtering, and parameter tuning. In this work, we propose an adaptive LiDAR sensing framework that dynamically adjusts sensor parameters, jointly optimizing LiDAR acquisition and registration hyperparameters. By integrating registration feedback into the sensing loop, our approach optimally balances point density, noise, and sparsity, improving registration accuracy and efficiency. Evaluations in the CARLA simulation demonstrate that our method outperforms fixed-parameter baselines while retaining generalization abilities, highlighting the potential of adaptive LiDAR for autonomous perception and robotic applications.
>
---
#### [new 043] Assessing Vision-Language Models for Perception in Autonomous Underwater Robotic Software
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于自主水下机器人感知任务，旨在解决 underwater 环境中感知模块可靠性问题，通过评估 VLM 的性能与不确定性，为软件工程提供选择依据。**

- **链接: [https://arxiv.org/pdf/2602.10655v1](https://arxiv.org/pdf/2602.10655v1)**

> **作者:** Muhammad Yousaf; Aitor Arrieta; Shaukat Ali; Paolo Arcaini; Shuai Wang
>
> **备注:** 10 pages, 5 figures, submitted to ICST 2026
>
> **摘要:** Autonomous Underwater Robots (AURs) operate in challenging underwater environments, including low visibility and harsh water conditions. Such conditions present challenges for software engineers developing perception modules for the AUR software. To successfully carry out these tasks, deep learning has been incorporated into the AUR software to support its operations. However, the unique challenges of underwater environments pose difficulties for deep learning models, which often rely on labeled data that is scarce and noisy. This may undermine the trustworthiness of AUR software that relies on perception modules. Vision-Language Models (VLMs) offer promising solutions for AUR software as they generalize to unseen objects and remain robust in noisy conditions by inferring information from contextual cues. Despite this potential, their performance and uncertainty in underwater environments remain understudied from a software engineering perspective. Motivated by the needs of an industrial partner in assurance and risk management for maritime systems to assess the potential use of VLMs in this context, we present an empirical evaluation of VLM-based perception modules within the AUR software. We assess their ability to detect underwater trash by computing performance, uncertainty, and their relationship, to enable software engineers to select appropriate VLMs for their AUR software.
>
---
#### [new 044] Transforming Policy-Car Swerving for Mitigating Stop-and-Go Traffic Waves: A Practice-Oriented Jam-Absorption Driving Strategy
- **分类: physics.soc-ph; cs.AI; cs.RO**

- **简介: 该论文属于交通管理任务，旨在解决Stop-and-Go交通波传播问题，提出一种实用的Jam-Absorption Driving策略。**

- **链接: [https://arxiv.org/pdf/2602.10234v1](https://arxiv.org/pdf/2602.10234v1)**

> **作者:** Zhengbing He
>
> **摘要:** Stop-and-go waves, as a major form of freeway traffic congestion, cause severe and long-lasting adverse effects, including reduced traffic efficiency, increased driving risks, and higher vehicle emissions. Amongst the highway traffic management strategies, jam-absorption driving (JAD), in which a dedicated vehicle performs "slow-in" and "fast-out" maneuvers before being captured by a stop-and-go wave, has been proposed as a potential method for preventing the propagation of such waves. However, most existing JAD strategies remain impractical mainly due to the lack of discussion regarding implementation vehicles and operational conditions. Inspired by real-world observations of police-car swerving behavior, this paper first introduces a Single-Vehicle Two-Detector Jam-Absorption Driving (SVDD-JAD) problem, and then proposes a practical JAD strategy that transforms such behavior into a maneuver capable of suppressing the propagation of an isolated stop-and-go wave. Five key parameters that significantly affect the proposed strategy, namely, JAD speed, inflow traffic speed, wave width, wave speed, and in-wave speed, are identified and systematically analyzed. Using a SUMO-based simulation as an illustrative example, we further demonstrate how these parameters can be measured in practice with two stationary roadside traffic detectors. The results show that the proposed JAD strategy successfully suppresses the propagation of a stop-and-go wave, without triggering a secondary wave. This paper is expected to take a significant step toward making JAD practical, advancing it from a theoretical concept to a feasible and implementable strategy. To promote reproducibility in the transportation domain, we have also open-sourced all the code on our GitHub repository https://github.com/gotrafficgo.
>
---
#### [new 045] Semi-Supervised Cross-Domain Imitation Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究跨域模仿学习任务，解决专家数据获取成本高的问题。提出半监督方法SS-CDIL，利用少量目标专家数据和大量未标注轨迹，提升策略学习的稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2602.10793v1](https://arxiv.org/pdf/2602.10793v1)**

> **作者:** Li-Min Chu; Kai-Siang Ma; Ming-Hong Chen; Ping-Chun Hsieh
>
> **备注:** Published in Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Cross-domain imitation learning (CDIL) accelerates policy learning by transferring expert knowledge across domains, which is valuable in applications where the collection of expert data is costly. Existing methods are either supervised, relying on proxy tasks and explicit alignment, or unsupervised, aligning distributions without paired data, but often unstable. We introduce the Semi-Supervised CDIL (SS-CDIL) setting and propose the first algorithm for SS-CDIL with theoretical justification. Our method uses only offline data, including a small number of target expert demonstrations and some unlabeled imperfect trajectories. To handle domain discrepancy, we propose a novel cross-domain loss function for learning inter-domain state-action mappings and design an adaptive weight function to balance the source and target knowledge. Experiments on MuJoCo and Robosuite show consistent gains over the baselines, demonstrating that our approach achieves stable and data-efficient policy learning with minimal supervision. Our code is available at~ https://github.com/NYCU-RL-Bandits-Lab/CDIL.
>
---
#### [new 046] Confounding Robust Continuous Control via Automatic Reward Shaping
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决复杂连续控制中奖励函数设计的问题。通过自动学习奖励函数，提升算法在存在混杂变量时的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.10305v1](https://arxiv.org/pdf/2602.10305v1)**

> **作者:** Mateo Juliani; Mingxuan Li; Elias Bareinboim
>
> **备注:** Mateo Juliani and Mingxuan Li contributed equally to this work; accepted in AAMAS 2026
>
> **摘要:** Reward shaping has been applied widely to accelerate Reinforcement Learning (RL) agents' training. However, a principled way of designing effective reward shaping functions, especially for complex continuous control problems, remains largely under-explained. In this work, we propose to automatically learn a reward shaping function for continuous control problems from offline datasets, potentially contaminated by unobserved confounding variables. Specifically, our method builds upon the recently proposed causal Bellman equation to learn a tight upper bound on the optimal state values, which is then used as the potentials in the Potential-Based Reward Shaping (PBRS) framework. Our proposed reward shaping algorithm is tested with Soft-Actor-Critic (SAC) on multiple commonly used continuous control benchmarks and exhibits strong performance guarantees under unobserved confounders. More broadly, our work marks a solid first step towards confounding robust continuous control from a causal perspective. Code for training our reward shaping functions can be found at https://github.com/mateojuliani/confounding_robust_cont_control.
>
---
#### [new 047] Towards Learning a Generalizable 3D Scene Representation from 2D Observations
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景重建任务，旨在从2D观测中学习可泛化的3D占用表示。解决传统方法在全局坐标系下泛化能力不足的问题，通过整合多视角信息实现无需微调的场景预测。**

- **链接: [https://arxiv.org/pdf/2602.10943v1](https://arxiv.org/pdf/2602.10943v1)**

> **作者:** Martin Gromniak; Jan-Gerrit Habekost; Sebastian Kamp; Sven Magg; Stefan Wermter
>
> **备注:** Paper accepted at ESANN 2026
>
> **摘要:** We introduce a Generalizable Neural Radiance Field approach for predicting 3D workspace occupancy from egocentric robot observations. Unlike prior methods operating in camera-centric coordinates, our model constructs occupancy representations in a global workspace frame, making it directly applicable to robotic manipulation. The model integrates flexible source views and generalizes to unseen object arrangements without scene-specific finetuning. We demonstrate the approach on a humanoid robot and evaluate predicted geometry against 3D sensor ground truth. Trained on 40 real scenes, our model achieves 26mm reconstruction error, including occluded regions, validating its ability to infer complete 3D occupancy beyond traditional stereo vision methods.
>
---
#### [new 048] (MGS)$^2$-Net: Unifying Micro-Geometric Scale and Macro-Geometric Structure for Cross-View Geo-Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于跨视角地理定位任务，解决航拍图像与卫星图间几何错位问题。提出(MGS)$^2$框架，融合宏观结构与微观尺度，提升定位精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.10704v1](https://arxiv.org/pdf/2602.10704v1)**

> **作者:** Minglei Li; Mengfan He; Chao Chen; Ziyang Meng
>
> **摘要:** Cross-view geo-localization (CVGL) is pivotal for GNSS-denied UAV navigation but remains brittle under the drastic geometric misalignment between oblique aerial views and orthographic satellite references. Existing methods predominantly operate within a 2D manifold, neglecting the underlying 3D geometry where view-dependent vertical facades (macro-structure) and scale variations (micro-scale) severely corrupt feature alignment. To bridge this gap, we propose (MGS)$^2$, a geometry-grounded framework. The core of our innovation is the Macro-Geometric Structure Filtering (MGSF) module. Unlike pixel-wise matching sensitive to noise, MGSF leverages dilated geometric gradients to physically filter out high-frequency facade artifacts while enhancing the view-invariant horizontal plane, directly addressing the domain shift. To guarantee robust input for this structural filtering, we explicitly incorporate a Micro-Geometric Scale Adaptation (MGSA) module. MGSA utilizes depth priors to dynamically rectify scale discrepancies via multi-branch feature fusion. Furthermore, a Geometric-Appearance Contrastive Distillation (GACD) loss is designed to strictly discriminate against oblique occlusions. Extensive experiments demonstrate that (MGS)$^2$ achieves state-of-the-art performance, recording a Recall@1 of 97.5\% on University-1652 and 97.02\% on SUES-200. Furthermore, the framework exhibits superior cross-dataset generalization against geometric ambiguity. The code is available at: \href{https://github.com/GabrielLi1473/MGS-Net}{https://github.com/GabrielLi1473/MGS-Net}.
>
---
## 更新

#### [replaced 001] First Multi-Constellation Observations of Navigation Satellite Signals in the Lunar Domain by Post-Processing L1/L5 IQ Snapshots
- **分类: physics.space-ph; astro-ph.IM; cs.RO; eess.SP**

- **简介: 该论文属于深空导航任务，旨在提升月球及近月空间的航天器自主性。通过处理LuGRE接收器的IQ数据，验证多星座信号在远距离下的可用性，提升定位精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.06081v2](https://arxiv.org/pdf/2601.06081v2)**

> **作者:** Lorenzo Sciacca; Alex Minetto; Andrea Nardin; Fabio Dovis; Luca Canzian; Mario Musmeci; Claudia Facchinetti; Giancarlo Varacalli
>
> **备注:** 13 pages, 9 figures, IEEE Transactions on Aerospace and Electronic Systems
>
> **摘要:** The use of Global Navigation Satellite Systems (GNSS) to increase spacecraft autonomy for orbit determination has gained renewed momentum following the Lunar GNSS Receiver Experiment (LuGRE), which demonstrated feasible onboard GPS and Galileo signal reception and tracking at lunar distances. This work processes in-phase and quadrature (IQ) snapshots collected by the LuGRE receiver in cis-lunar space and on the lunar surface to assess multi-frequency, multi-constellation signal availability. Signals from additional systems beyond GPS and Galileo, including RNSS and SBAS constellations, are observable and successfully acquired exclusively in the recorded IQ snapshots. These observations provide the first experimental evidence that signals from multiple constellations, including systems not supported by LuGRE realtime operations, are detectable at unprecedented distances from Earth. Useful observables can be extracted from the IQ snapshots, despite minimal sampling rates, 4-bit quantization, and short durations (200 ms-2 s), through a hybrid coherent/non-coherent acquisition stage compensating for code Doppler. These observations are exploited to tune simulation tools and to perform extended simulation campaigns, showing that the inclusion of additional constellations significantly improves availability; for a 26 dB-Hz acquisition threshold, the fraction of epochs with at least four visible satellites increases from 11% to 46% of the total epoch count. These findings indicate that BeiDou, RNSS, and SBAS signals can substantially enhance GNSS-based autonomy for lunar and cislunar missions.
>
---
#### [replaced 002] Robotic Depowdering for Additive Manufacturing Via Pose Tracking
- **分类: cs.RO**

- **简介: 该论文属于机器人自动化任务，旨在解决3D打印后去除未熔粉末的问题。通过视觉系统实现对零件的实时位姿跟踪和去粉进度估计，实现高效无损去粉。**

- **链接: [https://arxiv.org/pdf/2207.04196v3](https://arxiv.org/pdf/2207.04196v3)**

> **作者:** Zhenwei Liu; Junyi Geng; Xikai Dai; Tomasz Swierzewski; Kenji Shimada
>
> **备注:** Github link: https://github.com/zhenweil/Robotic-Depowdering-for-Additive-Manufacturing-Via-Pose-Tracking Video link: https://www.youtube.com/watch?v=AUIkyULAhqM
>
> **摘要:** With the rapid development of powder-based additive manufacturing, depowdering, a process of removing unfused powder that covers 3D-printed parts, has become a major bottleneck to further improve its productiveness. Traditional manual depowdering is extremely time-consuming and costly, and some prior automated systems either require pre-depowdering or lack adaptability to different 3D-printed parts. To solve these problems, we introduce a robotic system that automatically removes unfused powder from the surface of 3D-printed parts. The key component is a visual perception system, which consists of a pose-tracking module that tracks the 6D pose of powder-occluded parts in real-time, and a progress estimation module that estimates the depowdering completion percentage. The tracking module can be run efficiently on a laptop CPU at up to 60 FPS. Experiments show that our depowdering system can remove unfused powder from the surface of various 3D-printed parts without causing any damage. To the best of our knowledge, this is one of the first vision-based robotic depowdering systems that adapt to parts with various shapes without the need for pre-depowdering.
>
---
#### [replaced 003] Proactive Local-Minima-Free Robot Navigation: Blending Motion Prediction with Safe Control
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中安全高效避障问题。通过融合运动预测与安全控制，提出在线学习障碍物屏障函数的方法，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2601.10233v2](https://arxiv.org/pdf/2601.10233v2)**

> **作者:** Yifan Xue; Ze Zhang; Knut Åkesson; Nadia Figueroa
>
> **备注:** Co-first authors: Yifan Xue and Ze Zhang; Accepted by IEEE RA-L 2026
>
> **摘要:** This work addresses the challenge of safe and efficient mobile robot navigation in complex dynamic environments with concave moving obstacles. Reactive safe controllers like Control Barrier Functions (CBFs) design obstacle avoidance strategies based only on the current states of the obstacles, risking future collisions. To alleviate this problem, we use Gaussian processes to learn barrier functions online from multimodal motion predictions of obstacles generated by neural networks trained with energy-based learning. The learned barrier functions are then fed into quadratic programs using modulated CBFs (MCBFs), a local-minimum-free version of CBFs, to achieve safe and efficient navigation. The proposed framework makes two key contributions. First, it develops a prediction-to-barrier function online learning pipeline. Second, it introduces an autonomous parameter tuning algorithm that adapts MCBFs to deforming, prediction-based barrier functions. The framework is evaluated in both simulations and real-world experiments, consistently outperforming baselines and demonstrating superior safety and efficiency in crowded dynamic environments.
>
---
#### [replaced 004] MOSAIC: Bridging the Sim-to-Real Gap in Generalist Humanoid Motion Tracking and Teleoperation with Rapid Residual Adaptation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动跟踪与遥操作任务，旨在解决仿真到现实的接口误差问题。提出MOSAIC系统，通过快速残差适应提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.08594v2](https://arxiv.org/pdf/2602.08594v2)**

> **作者:** Zhenguo Sun; Bo-Sheng Huang; Yibo Peng; Xukun Li; Jingyu Ma; Yu Sun; Zhe Li; Haojun Jiang; Biao Gao; Zhenshan Bing; Xinlong Wang; Alois Knoll
>
> **备注:** add project page: training codes and data are open sourced
>
> **摘要:** Generalist humanoid motion trackers have recently achieved strong simulation metrics by scaling data and training, yet often remain brittle on hardware during sustained teleoperation due to interface- and dynamics-induced errors. We present MOSAIC, an open-source, full-stack system for humanoid motion tracking and whole-body teleoperation across multiple interfaces. MOSAIC first learns a teleoperation-oriented general motion tracker via RL on a multi-source motion bank with adaptive resampling and rewards that emphasize world-frame motion consistency, which is critical for mobile teleoperation. To bridge the sim-to-real interface gap without sacrificing generality, MOSAIC then performs rapid residual adaptation: an interface-specific policy is trained using minimal interface-specific data, and then distilled into the general tracker through an additive residual module, outperforming naive fine-tuning or continual learning. We validate MOSAIC with systematic ablations, out-of-distribution benchmarking, and real-robot experiments demonstrating robust offline motion replay and online long-horizon teleoperation under realistic latency and noise. Project page: baai-humanoid.github.io/MOSAIC.
>
---
#### [replaced 005] First-order friction models with bristle dynamics: lumped and distributed formulations
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于摩擦建模任务，旨在建立更物理可解释的动态摩擦模型。通过引入一阶摩擦模型和分布参数模型，解决传统经验模型缺乏物理基础的问题。**

- **链接: [https://arxiv.org/pdf/2602.09429v2](https://arxiv.org/pdf/2602.09429v2)**

> **作者:** Luigi Romano; Ole Morten Aamo; Jan Åslund; Erik Frisk
>
> **备注:** 15 pages, 9 figures. Under review at IEEE Transactions on Control Systems Technology
>
> **摘要:** Dynamic models, particularly rate-dependent models, have proven effective in capturing the key phenomenological features of frictional processes, whilst also possessing important mathematical properties that facilitate the design of control and estimation algorithms. However, many rate-dependent formulations are built on empirical considerations, whereas physical derivations may offer greater interpretability. In this context, starting from fundamental physical principles, this paper introduces a novel class of first-order dynamic friction models that approximate the dynamics of a bristle element by inverting the friction characteristic. Amongst the developed models, a specific formulation closely resembling the LuGre model is derived using a simple rheological equation for the bristle element. This model is rigorously analyzed in terms of stability and passivity -- important properties that support the synthesis of observers and controllers. Furthermore, a distributed version, formulated as a hyperbolic partial differential equation (PDE), is presented, which enables the modeling of frictional processes commonly encountered in rolling contact phenomena. The tribological behavior of the proposed description is evaluated through classical experiments and validated against the response predicted by the LuGre model, revealing both notable similarities and key differences.
>
---
#### [replaced 006] Sim2real Image Translation Enables Viewpoint-Robust Policies from Fixed-Camera Datasets
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉机器人控制任务，解决相机视角变化导致的策略脆弱问题。通过提出MANGO方法，实现模拟到现实的图像翻译，提升策略的视角鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.09605v2](https://arxiv.org/pdf/2601.09605v2)**

> **作者:** Jeremiah Coholich; Justin Wit; Robert Azarcon; Zsolt Kira
>
> **摘要:** Vision-based policies for robot manipulation have achieved significant recent success, but are still brittle to distribution shifts such as camera viewpoint variations. Robot demonstration data is scarce and often lacks appropriate variation in camera viewpoints. Simulation offers a way to collect robot demonstrations at scale with comprehensive coverage of different viewpoints, but presents a visual sim2real challenge. To bridge this gap, we propose MANGO -- an unpaired image translation method with a novel segmentation-conditioned InfoNCE loss, a highly-regularized discriminator design, and a modified PatchNCE loss. We find that these elements are crucial for maintaining viewpoint consistency during sim2real translation. When training MANGO, we only require a small amount of fixed-camera data from the real world, but show that our method can generate diverse unseen viewpoints by translating simulated observations. In this domain, MANGO outperforms all other image translation methods we tested. Imitation-learning policies trained on data augmented by MANGO are able to achieve success rates as high as 60% on views that the non-augmented policy fails completely on.
>
---
#### [replaced 007] Provably Optimal Reinforcement Learning under Safety Filtering
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于安全强化学习任务，解决安全与性能的权衡问题。通过定义安全关键MDP和过滤MDP，证明安全过滤不影响最优性能，验证了理论有效性。**

- **链接: [https://arxiv.org/pdf/2510.18082v2](https://arxiv.org/pdf/2510.18082v2)**

> **作者:** Donggeon David Oh; Duy P. Nguyen; Haimin Hu; Jaime F. Fisac
>
> **备注:** Accepted for publication in the proceedings of The International Association for Safe & Ethical AI (IASEAI) 2026; 17 pages, 3 figures
>
> **摘要:** Recent advances in reinforcement learning (RL) enable its use on increasingly complex tasks, but the lack of formal safety guarantees still limits its application in safety-critical settings. A common practical approach is to augment the RL policy with a safety filter that overrides unsafe actions to prevent failures during both training and deployment. However, safety filtering is often perceived as sacrificing performance and hindering the learning process. We show that this perceived safety-performance tradeoff is not inherent and prove, for the first time, that enforcing safety with a sufficiently permissive safety filter does not degrade asymptotic performance. We formalize RL safety with a safety-critical Markov decision process (SC-MDP), which requires categorical, rather than high-probability, avoidance of catastrophic failure states. Additionally, we define an associated filtered MDP in which all actions result in safe effects, thanks to a safety filter that is considered to be a part of the environment. Our main theorem establishes that (i) learning in the filtered MDP is safe categorically, (ii) standard RL convergence carries over to the filtered MDP, and (iii) any policy that is optimal in the filtered MDP-when executed through the same filter-achieves the same asymptotic return as the best safe policy in the SC-MDP, yielding a complete separation between safety enforcement and performance optimization. We validate the theory on Safety Gymnasium with representative tasks and constraints, observing zero violations during training and final performance matching or exceeding unfiltered baselines. Together, these results shed light on a long-standing question in safety-filtered learning and provide a simple, principled recipe for safe RL: train and deploy RL policies with the most permissive safety filter that is available.
>
---
#### [replaced 008] OAT: Ordered Action Tokenization
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学习任务，解决连续动作离散化问题。提出OAT方法，实现高效、可解码的动作分词，提升自回归策略性能。**

- **链接: [https://arxiv.org/pdf/2602.04215v2](https://arxiv.org/pdf/2602.04215v2)**

> **作者:** Chaoqi Liu; Xiaoshen Han; Jiawei Gao; Yue Zhao; Haonan Chen; Yilun Du
>
> **摘要:** Autoregressive policies offer a compelling foundation for scalable robot learning by enabling discrete abstraction, token-level reasoning, and flexible inference. However, applying autoregressive modeling to continuous robot actions requires an effective action tokenization scheme. Existing approaches either rely on analytical discretization methods that produce prohibitively long token sequences, or learned latent tokenizers that lack structure, limiting their compatibility with next-token prediction. In this work, we identify three desiderata for action tokenization - high compression, total decodability, and a left-to-right causally ordered token space - and introduce Ordered Action Tokenization (OAT), a learned action tokenizer that satisfies all three. OAT discretizes action chunks into an ordered sequence of tokens using transformer with registers, finite scalar quantization, and ordering-inducing training mechanisms. The resulting token space aligns naturally with autoregressive generation and enables prefix-based detokenization, yielding an anytime trade-off between inference cost and action fidelity. Across more than 20 tasks spanning four simulation benchmarks and real-world settings, autoregressive policies equipped with OAT consistently outperform prior tokenization schemes and diffusion-based baselines, while offering significantly greater flexibility at inference time.
>
---
#### [replaced 009] Lateral tracking control of all-wheel steering vehicles with intelligent tires
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于车辆控制任务，旨在解决全轮转向车辆的横向跟踪控制问题。通过结合智能轮胎技术和偏微分方程模型，提出一种新型控制策略，提升路径跟随与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.09427v2](https://arxiv.org/pdf/2602.09427v2)**

> **作者:** Luigi Romano; Ole Morten Aamo; Jan Åslund; Erik Frisk
>
> **备注:** 16 pages, 12 figures. Under review at IEEE Transactions on Intelligent Vehicles
>
> **摘要:** The accurate characterization of tire dynamics is critical for advancing control strategies in autonomous road vehicles, as tire behavior significantly influences handling and stability through the generation of forces and moments at the tire-road interface. Smart tire technologies have emerged as a promising tool for sensing key variables such as road friction, tire pressure, and wear states, and for estimating kinematic and dynamic states like vehicle speed and tire forces. However, most existing estimation and control algorithms rely on empirical correlations or machine learning approaches, which require extensive calibration and can be sensitive to variations in operating conditions. In contrast, model-based techniques, which leverage infinite-dimensional representations of tire dynamics using partial differential equations (PDEs), offer a more robust approach. This paper proposes a novel model-based, output-feedback lateral tracking control strategy for all-wheel steering vehicles that integrates distributed tire dynamics with smart tire technologies. The primary contributions include the suppression of micro-shimmy phenomena at low speeds and path-following via force control, achieved through the estimation of tire slip angles, vehicle kinematics, and lateral tire forces. The proposed controller and observer are based on formulations using ODE-PDE systems, representing rigid body dynamics and distributed tire behavior. This work marks the first rigorous control strategy for vehicular systems equipped with distributed tire representations in conjunction with smart tire technologies.
>
---
#### [replaced 010] Crane Lowering Guidance Using a Attachable Camera Module for Driver Vision Support
- **分类: cs.RO**

- **简介: 该论文属于起重机操作辅助任务，旨在解决负载遮挡操作员视线的问题。通过安装可附加的摄像头模块，实现实时图像传输与定位引导，提升施工安全。**

- **链接: [https://arxiv.org/pdf/2601.11026v2](https://arxiv.org/pdf/2601.11026v2)**

> **作者:** HyoJae Kang; SunWoo Ahn; InGyu Choi; GeonYeong Go; KunWoo Son; Min-Sung Kang
>
> **备注:** Published in the Proceedings of ICCR 2025 (IEEE)
>
> **摘要:** Cranes have long been essential equipment for lifting and placing heavy loads in construction projects. This study focuses on the lowering phase of crane operation, the stage in which the load is moved to the desired location. During this phase, a constant challenge exists: the load obstructs the operator's view of the landing point. As a result, operators traditionally have to rely on verbal or gestural instructions from ground personnel, which significantly impacts site safety. To alleviate this constraint, the proposed system incorporates a attachable camera module designed to be attached directly to the load via a suction cup. This module houses a single-board computer, battery, and compact camera. After installation, it streams and processes images of the ground directly below the load in real time to generate installation guidance. Simultaneously, this guidance is transmitted to and monitored by a host computer. Preliminary experiments were conducted by attaching this module to a test object, confirming the feasibility of real-time image acquisition and transmission. This approach has the potential to significantly improve safety on construction sites by providing crane operators with an instant visual reference of hidden landing zones.
>
---
#### [replaced 011] CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents
- **分类: cs.AI; cs.CE; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出CostNav，一个用于评估物理AI代理经济成本的导航基准，解决传统导航任务忽略实际经济约束的问题。通过真实数据与仿真结合，揭示现有方法在商业可行性上的不足。**

- **链接: [https://arxiv.org/pdf/2511.20216v4](https://arxiv.org/pdf/2511.20216v4)**

> **作者:** Haebin Seong; Sungmin Kim; Yongjun Cho; Myunchul Joe; Geunwoo Kim; Yubeen Park; Sunhoo Kim; Yoonshik Kim; Suhwan Choi; Jaeyoon Jung; Jiyong Youn; Jinmyung Kwak; Sunghee Ahn; Jaemin Lee; Younggil Do; Seungyeop Yi; Woojin Cheong; Minhyeok Oh; Minchan Kim; Seongjae Kang; Samwoo Seong; Youngjae Yu; Yunsung Lee
>
> **摘要:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data - such as SEC filings and AIS injury reports - with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Our evaluation of rule-based Nav2 navigation shows that current approaches are not economically viable: the contribution margin is -22.81/run (AMCL) and -12.87/run (GPS), resulting in no break-even point. We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on the metric of cost rather than the underlying architecture. All resources are available at https://github.com/worv-ai/CostNav.
>
---
#### [replaced 012] ZebraPose: Zebra Detection and Pose Estimation using only Synthetic Data
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于动物检测与姿态估计任务，解决真实数据收集困难及合成数据泛化性差的问题。通过生成高质量合成数据，实现无需真实数据的准确检测与姿态估计。**

- **链接: [https://arxiv.org/pdf/2408.10831v2](https://arxiv.org/pdf/2408.10831v2)**

> **作者:** Elia Bonetto; Aamir Ahmad
>
> **备注:** 17 pages, 5 tables, 13 figures. Published in WACV 2026
>
> **摘要:** Collecting and labeling large real-world wild animal datasets is impractical, costly, error-prone, and labor-intensive. For animal monitoring tasks, as detection, tracking, and pose estimation, out-of-distribution viewpoints (e.g. aerial) are also typically needed but rarely found in publicly available datasets. To solve this, existing approaches synthesize data with simplistic techniques that then necessitate strategies to bridge the synthetic-to-real gap. Therefore, real images, style constraints, complex animal models, or pre-trained networks are often leveraged. In contrast, we generate a fully synthetic dataset using a 3D photorealistic simulator and demonstrate that it can eliminate such needs for detecting and estimating 2D poses of wild zebras. Moreover, existing top-down 2D pose estimation approaches using synthetic data assume reliable detection models. However, these often fail in out-of-distribution scenarios, e.g. those that include wildlife or aerial imagery. Our method overcomes this by enabling the training of both tasks using the same synthetic dataset. Through extensive benchmarks, we show that models trained from scratch exclusively on our synthetic data generalize well to real images. We perform these using multiple real-world and synthetic datasets, pre-trained and randomly initialized backbones, and different image resolutions. Code, results, models, and data can be found athttps://zebrapose.is.tue.mpg.de/.
>
---
#### [replaced 013] BagelVLA: Enhancing Long-Horizon Manipulation via Interleaved Vision-Language-Action Generation
- **分类: cs.RO**

- **简介: 该论文提出BagelVLA，解决长周期操作任务中的多模态协同问题，整合语言规划、视觉预测与动作生成，提升复杂任务性能。**

- **链接: [https://arxiv.org/pdf/2602.09849v2](https://arxiv.org/pdf/2602.09849v2)**

> **作者:** Yucheng Hu; Jianke Zhang; Yuanfei Luo; Yanjiang Guo; Xiaoyu Chen; Xinshu Sun; Kun Feng; Qingzhou Lu; Sheng Chen; Yangang Zhang; Wei Li; Jianyu Chen
>
> **摘要:** Equipping embodied agents with the ability to reason about tasks, foresee physical outcomes, and generate precise actions is essential for general-purpose manipulation. While recent Vision-Language-Action (VLA) models have leveraged pre-trained foundation models, they typically focus on either linguistic planning or visual forecasting in isolation. These methods rarely integrate both capabilities simultaneously to guide action generation, leading to suboptimal performance in complex, long-horizon manipulation tasks. To bridge this gap, we propose BagelVLA, a unified model that integrates linguistic planning, visual forecasting, and action generation within a single framework. Initialized from a pretrained unified understanding and generative model, BagelVLA is trained to interleave textual reasoning and visual prediction directly into the action execution loop. To efficiently couple these modalities, we introduce Residual Flow Guidance (RFG), which initializes from current observation and leverages single-step denoising to extract predictive visual features, guiding action generation with minimal latency. Extensive experiments demonstrate that BagelVLA outperforms existing baselines by a significant margin on multiple simulated and real-world benchmarks, particularly in tasks requiring multi-stage reasoning.
>
---
#### [replaced 014] HOGraspFlow: Taxonomy-Aware Hand-Object Retargeting for Multi-Modal SE(3) Grasp Generation
- **分类: cs.RO**

- **简介: 该论文提出HOGraspFlow，用于多模态SE(3)抓取生成，解决从RGB图像中生成精确抓取姿态的问题。通过视觉语义、接触重建和分类先验，实现无需物体几何信息的高精度抓取合成。**

- **链接: [https://arxiv.org/pdf/2509.16871v2](https://arxiv.org/pdf/2509.16871v2)**

> **作者:** Yitian Shi; Zicheng Guo; Rosa Wolf; Edgar Welte; Rania Rayyes
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** We propose Hand-Object\emph{(HO)GraspFlow}, an affordance-centric approach that retargets a single RGB with hand-object interaction (HOI) into multi-modal executable parallel jaw grasps without explicit geometric priors on target objects. Building on foundation models for hand reconstruction and vision, we synthesize $SE(3)$ grasp poses with denoising flow matching (FM), conditioned on the following three complementary cues: RGB foundation features as visual semantics, HOI contact reconstruction, and taxonomy-aware prior on grasp types. Our approach demonstrates high fidelity in grasp synthesis without explicit HOI contact input or object geometry, while maintaining strong contact and taxonomy recognition. Another controlled comparison shows that \emph{HOGraspFlow} consistently outperforms diffusion-based variants (\emph{HOGraspDiff}), achieving high distributional fidelity and more stable optimization in $SE(3)$. We demonstrate a reliable, object-agnostic grasp synthesis from human demonstrations in real-world experiments, where an average success rate of over $83\%$ is achieved. Code: https://github.com/YitianShi/HOGraspFlow
>
---
#### [replaced 015] Localized Graph-Based Neural Dynamics Models for Terrain Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人地形操控任务，解决高维地形状态表示难题。通过图神经动力模型，仅关注局部活动区域，提升预测效率与准确性。**

- **链接: [https://arxiv.org/pdf/2503.23270v3](https://arxiv.org/pdf/2503.23270v3)**

> **作者:** Chaoqi Liu; Yunzhu Li; Kris Hauser
>
> **摘要:** Predictive models can be particularly helpful for robots to effectively manipulate terrains in construction sites and extraterrestrial surfaces. However, terrain state representations become extremely high-dimensional especially to capture fine-resolution details and when depth is unknown or unbounded. This paper introduces a learning-based approach for terrain dynamics modeling and manipulation, leveraging the Graph-based Neural Dynamics (GBND) framework to represent terrain deformation as motion of a graph of particles. Based on the principle that the moving portion of a terrain is usually localized, our approach builds a large terrain graph (potentially millions of particles) but only identifies a very small active subgraph (hundreds of particles) for predicting the outcomes of robot-terrain interaction. To minimize the size of the active subgraph we introduce a learning-based approach that identifies a small region of interest (RoI) based on the robot's control inputs and the current scene. We also introduce a novel domain boundary feature encoding that allows GBNDs to perform accurate dynamics prediction in the RoI interior while avoiding particle penetration through RoI boundaries. Our proposed method is both orders of magnitude faster than naive GBND and it achieves better overall prediction accuracy. We further evaluated our framework on excavation and shaping tasks on terrain with different granularity.
>
---
#### [replaced 016] Towards Safe Path Tracking Using the Simplex Architecture
- **分类: cs.RO**

- **简介: 该论文属于机器人路径跟踪任务，旨在解决复杂环境中安全与性能的平衡问题。提出基于Simplex架构的控制器，结合强化学习与高保障控制，确保安全同时保持性能。**

- **链接: [https://arxiv.org/pdf/2503.10559v2](https://arxiv.org/pdf/2503.10559v2)**

> **作者:** Georg Jäger; Nils-Jonathan Friedrich; Hauke Petersen; Benjamin Noack
>
> **摘要:** Robot navigation in complex environments necessitates controllers that prioritize safety while remaining performant and adaptable. Traditional controllers like Regulated Pure Pursuit, Dynamic Window Approach, and Model-Predictive Path Integral, while reliable, struggle to adapt to dynamic conditions. Reinforcement Learning offers adaptability but state-wise safety guarantees remain challenging and often absent in practice. To address this, we propose a path tracking controller leveraging the Simplex architecture. It combines a Reinforcement Learning controller for adaptiveness and performance with a high-assurance controller providing safety and stability. Our main goal is to provide a safe testbed for the design and evaluation of path-planning algorithms, including machine-learning-based planners. Our contribution is twofold. We firstly discuss general stability and safety considerations for designing controllers using the Simplex architecture. Secondly, we present a Simplex-based path tracking controller. Our simulation results, supported by preliminary in-field tests, demonstrate the controller's effectiveness in maintaining safety while achieving comparable performance to state-of-the-art methods.
>
---
#### [replaced 017] Discrete Variational Autoencoding via Policy Search
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于深度学习任务，解决离散变分自编码器训练难题，通过策略搜索方法提升高维数据重建效果。**

- **链接: [https://arxiv.org/pdf/2509.24716v3](https://arxiv.org/pdf/2509.24716v3)**

> **作者:** Michael Drolet; Firas Al-Hafez; Aditya Bhatt; Jan Peters; Oleg Arenz
>
> **摘要:** Discrete latent bottlenecks in variational autoencoders (VAEs) offer high bit efficiency and can be modeled with autoregressive discrete distributions, enabling parameter-efficient multimodal search with transformers. However, discrete random variables do not allow for exact differentiable parameterization; therefore, discrete VAEs typically rely on approximations, such as Gumbel-Softmax reparameterization or straight-through gradient estimates, or employ high-variance gradient-free methods such as REINFORCE that have had limited success on high-dimensional tasks such as image reconstruction. Inspired by popular techniques in policy search, we propose a training framework for discrete VAEs that leverages the natural gradient of a non-parametric encoder to update the parametric encoder without requiring reparameterization. Our method, combined with automatic step size adaptation and a transformer-based encoder, scales to challenging datasets such as ImageNet and outperforms both approximate reparameterization methods and quantization-based discrete autoencoders in reconstructing high-dimensional data from compact latent spaces.
>
---
#### [replaced 018] RoboSubtaskNet: Temporal Sub-task Segmentation for Human-to-Robot Skill Transfer in Real-World Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboSubtaskNet，解决真实环境中人类到机器人的技能转移问题，通过时间子任务分割实现精准操作指令理解与执行。**

- **链接: [https://arxiv.org/pdf/2602.10015v2](https://arxiv.org/pdf/2602.10015v2)**

> **作者:** Dharmendra Sharma; Archit Sharma; John Rebeiro; Vaibhav Kesharwani; Peeyush Thakur; Narendra Kumar Dhar; Laxmidhar Behera
>
> **摘要:** Temporally locating and classifying fine-grained sub-task segments in long, untrimmed videos is crucial to safe human-robot collaboration. Unlike generic activity recognition, collaborative manipulation requires sub-task labels that are directly robot-executable. We present RoboSubtaskNet, a multi-stage human-to-robot sub-task segmentation framework that couples attention-enhanced I3D features (RGB plus optical flow) with a modified MS-TCN employing a Fibonacci dilation schedule to capture better short-horizon transitions such as reach-pick-place. The network is trained with a composite objective comprising cross-entropy and temporal regularizers (truncated MSE and a transition-aware term) to reduce over-segmentation and to encourage valid sub-task progressions. To close the gap between vision benchmarks and control, we introduce RoboSubtask, a dataset of healthcare and industrial demonstrations annotated at the sub-task level and designed for deterministic mapping to manipulator primitives. Empirically, RoboSubtaskNet outperforms MS-TCN and MS-TCN++ on GTEA and our RoboSubtask benchmark (boundary-sensitive and sequence metrics), while remaining competitive on the long-horizon Breakfast benchmark. Specifically, RoboSubtaskNet attains F1 @ 50 = 79.5%, Edit = 88.6%, Acc = 78.9% on GTEA; F1 @ 50 = 30.4%, Edit = 52.0%, Acc = 53.5% on Breakfast; and F1 @ 50 = 94.2%, Edit = 95.6%, Acc = 92.2% on RoboSubtask. We further validate the full perception-to-execution pipeline on a 7-DoF Kinova Gen3 manipulator, achieving reliable end-to-end behavior in physical trials (overall task success approx 91.25%). These results demonstrate a practical path from sub-task level video understanding to deployed robotic manipulation in real-world settings.
>
---
#### [replaced 019] Neural-Augmented Kelvinlet for Real-Time Soft Tissue Deformation Modeling
- **分类: cs.GR; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于手术仿真任务，旨在解决软组织实时变形建模问题。通过结合物理先验与神经网络，提升预测精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2506.08043v4](https://arxiv.org/pdf/2506.08043v4)**

> **作者:** Ashkan Shahbazi; Kyvia Pereira; Jon S. Heiselman; Elaheh Akbari; Annie C. Benson; Sepehr Seifi; Xinyuan Liu; Garrison L. Johnston; Jie Ying Wu; Nabil Simaan; Michael I. Miga; Soheil Kolouri
>
> **摘要:** Accurate and efficient modeling of soft-tissue interactions is fundamental for advancing surgical simulation, surgical robotics, and model-based surgical automation. To achieve real-time latency, classical Finite Element Method (FEM) solvers are often replaced with neural approximations; however, naively training such models in a fully data-driven manner without incorporating physical priors frequently leads to poor generalization and physically implausible predictions. We present a novel physics-informed neural simulation framework that enables real-time prediction of soft-tissue deformations under complex single- and multi-grasper interactions. Our approach integrates Kelvinlet-based analytical priors with large-scale FEM data, capturing both linear and nonlinear tissue responses. This hybrid design improves predictive accuracy and physical plausibility across diverse neural architectures while maintaining the low-latency performance required for interactive applications. We validate our method on challenging surgical manipulation tasks involving standard laparoscopic grasping tools, demonstrating substantial improvements in deformation fidelity and temporal stability over existing baselines. These results establish Kelvinlet-augmented learning as a principled and computationally efficient paradigm for real-time, physics-aware soft-tissue simulation in surgical AI.
>
---
#### [replaced 020] WorldArena: A Unified Benchmark for Evaluating Perception and Functional Utility of Embodied World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出WorldArena，一个统一基准，用于评估具身世界模型的感知质量和功能实用性，解决当前评价碎片化问题。**

- **链接: [https://arxiv.org/pdf/2602.08971v2](https://arxiv.org/pdf/2602.08971v2)**

> **作者:** Yu Shang; Zhuohang Li; Yiding Ma; Weikang Su; Xin Jin; Ziyou Wang; Lei Jin; Xin Zhang; Yinzhou Tang; Haisheng Su; Chen Gao; Wei Wu; Xihui Liu; Dhruv Shah; Zhaoxiang Zhang; Zhibo Chen; Jun Zhu; Yonghong Tian; Tat-Seng Chua; Wenwu Zhu; Yong Li
>
> **摘要:** While world models have emerged as a cornerstone of embodied intelligence by enabling agents to reason about environmental dynamics through action-conditioned prediction, their evaluation remains fragmented. Current evaluation of embodied world models has largely focused on perceptual fidelity (e.g., video generation quality), overlooking the functional utility of these models in downstream decision-making tasks. In this work, we introduce WorldArena, a unified benchmark designed to systematically evaluate embodied world models across both perceptual and functional dimensions. WorldArena assesses models through three dimensions: video perception quality, measured with 16 metrics across six sub-dimensions; embodied task functionality, which evaluates world models as data engines, policy evaluators, and action planners integrating with subjective human evaluation. Furthermore, we propose EWMScore, a holistic metric integrating multi-dimensional performance into a single interpretable index. Through extensive experiments on 14 representative models, we reveal a significant perception-functionality gap, showing that high visual quality does not necessarily translate into strong embodied task capability. WorldArena benchmark with the public leaderboard is released at https://world-arena.ai, providing a framework for tracking progress toward truly functional world models in embodied AI.
>
---
#### [replaced 021] PA-MPPI: Perception-Aware Model Predictive Path Integral Control for Quadrotor Navigation in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文提出PA-MPPI方法，用于解决未知环境中四旋翼飞行器的路径规划问题，通过引入感知成本提升探索能力，增强导航鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.14978v2](https://arxiv.org/pdf/2509.14978v2)**

> **作者:** Yifan Zhai; Rudolf Reiter; Davide Scaramuzza
>
> **摘要:** Quadrotor navigation in unknown environments is critical for practical missions such as search-and-rescue. Solving this problem requires addressing three key challenges: path planning in non-convex free space due to obstacles, satisfying quadrotor-specific dynamics and objectives, and exploring unknown regions to expand the map. Recently, the Model Predictive Path Integral (MPPI) method has emerged as a promising solution to the first two challenges. By leveraging sampling-based optimization, it can effectively handle non-convex free space while directly optimizing over the full quadrotor dynamics, enabling the inclusion of quadrotor-specific costs such as energy consumption. However, MPPI has been limited to tracking control that optimizes trajectories only within a small neighborhood around a reference trajectory, as it lacks the ability to explore unknown regions and plan alternative paths when blocked by large obstacles. To address this limitation, we introduce Perception-Aware MPPI (PA-MPPI). In this approach, perception-awareness is characterized by planning and adapting the trajectory online based on perception objectives. Specifically, when the goal is occluded, PA-MPPI incorporates a perception cost that biases trajectories toward those that can observe unknown regions. This expands the mapped traversable space and increases the likelihood of finding alternative paths to the goal. Through hardware experiments, we demonstrate that PA-MPPI, running at 50 Hz, performs on par with the state-of-the-art quadrotor navigation planner for unknown environments in challenging test scenarios. Furthermore, we show that PA-MPPI can serve as a safe and robust action policy for navigation foundation models, which often provide goal poses that are not directly reachable.
>
---
#### [replaced 022] HuMam: Humanoid Motion Control via End-to-End Deep Reinforcement Learning with Mamba
- **分类: cs.RO; cs.AI; cs.ET; eess.SP; eess.SY**

- **简介: 该论文提出HuMam，一种基于Mamba的端到端强化学习框架，用于双足机器人运动控制。解决训练不稳定、特征融合低效和能耗高的问题，通过状态融合与优化奖励函数提升性能。**

- **链接: [https://arxiv.org/pdf/2509.18046v2](https://arxiv.org/pdf/2509.18046v2)**

> **作者:** Yinuo Wang; Yuanyang Qi; Jinzhao Zhou; Pengxiang Meng; Xiaowen Tao
>
> **备注:** 12 pages
>
> **摘要:** End-to-end reinforcement learning (RL) for humanoid locomotion is appealing for its compact perception-action mapping, yet practical policies often suffer from training instability, inefficient feature fusion, and high actuation cost. We present HuMam, a state-centric end-to-end RL framework that employs a single-layer Mamba encoder to fuse robot-centric states with oriented footstep targets and a continuous phase clock. The policy outputs joint position targets tracked by a low-level PD loop and is optimized with PPO. A concise six-term reward balances contact quality, swing smoothness, foot placement, posture, and body stability while implicitly promoting energy saving. On the JVRC-1 humanoid in mc-mujoco, HuMam consistently improves learning efficiency, training stability, and overall task performance over a strong feedforward baseline, while reducing power consumption and torque peaks. To our knowledge, this is the first end-to-end humanoid RL controller that adopts Mamba as the fusion backbone, demonstrating tangible gains in efficiency, stability, and control economy.
>
---
#### [replaced 023] Self-Augmented Robot Trajectory: Efficient Imitation Learning via Safe Self-augmentation with Demonstrator-annotated Precision
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出SART框架，解决机器人模仿学习中数据采集效率低、安全性差的问题。通过单次人类示范和安全自增强生成多样化轨迹，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2509.09893v2](https://arxiv.org/pdf/2509.09893v2)**

> **作者:** Hanbit Oh; Masaki Murooka; Tomohiro Motoda; Ryoichi Nakajo; Yukiyasu Domae
>
> **备注:** 21 pages, 10 figures, Advanced Robotics accepted 2026.02.03
>
> **摘要:** Imitation learning is a promising paradigm for training robot agents; however, standard approaches typically require substantial data acquisition -- via numerous demonstrations or random exploration -- to ensure reliable performance. Although exploration reduces human effort, it lacks safety guarantees and often results in frequent collisions -- particularly in clearance-limited tasks (e.g., peg-in-hole) -- thereby, necessitating manual environmental resets and imposing additional human burden. This study proposes Self-Augmented Robot Trajectory (SART), a framework that enables policy learning from a single human demonstration, while safely expanding the dataset through autonomous augmentation. SART consists of two stages: (1) human teaching only once, where a single demonstration is provided and precision boundaries -- represented as spheres around key waypoints -- are annotated, followed by one environment reset; (2) robot self-augmentation, where the robot generates diverse, collision-free trajectories within these boundaries and reconnects to the original demonstration. This design improves the data collection efficiency by minimizing human effort while ensuring safety. Extensive evaluations in simulation and real-world manipulation tasks show that SART achieves substantially higher success rates than policies trained solely on human-collected demonstrations. Video results available at https://sites.google.com/view/sart-il .
>
---
#### [replaced 024] Multi-Momentum Observer Contact Estimation for Bipedal Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人接触状态估计任务，旨在无需接触传感器准确判断双足机器人脚部与地面的接触状态。通过多动量观测器方法，提高控制可靠性与稳定性。**

- **链接: [https://arxiv.org/pdf/2412.03462v2](https://arxiv.org/pdf/2412.03462v2)**

> **作者:** J. Joe Payne; Daniel A. Hagen; Denis Garagić; Aaron M. Johnson
>
> **摘要:** As bipedal robots become more and more popular in commercial and industrial settings, the ability to control them with a high degree of reliability is critical. To that end, this paper considers how to accurately estimate which feet are currently in contact with the ground so as to avoid improper control actions that could jeopardize the stability of the robot. Additionally, modern algorithms for estimating the position and orientation of a robot's base frame rely heavily on such contact mode estimates. Dedicated contact sensors on the feet can be used to estimate this contact mode, but these sensors are prone to noise, time delays, damage/yielding from repeated impacts with the ground, and are not available on every robot. To overcome these limitations, we propose a momentum observer based method for contact mode estimation that does not rely on such contact sensors. Often, momentum observers assume that the robot's base frame can be treated as an inertial frame. However, since many humanoids' legs represent a significant portion of the overall mass, the proposed method instead utilizes multiple simultaneous dynamic models. Each of these models assumes a different contact condition. A given contact assumption is then used to constrain the full dynamics in order to avoid assuming that either the body is an inertial frame or that a fully accurate estimate of body velocity is known. The (dis)agreement between each model's estimates and measurements is used to determine which contact mode is most likely using a Markov-style fusion method. The proposed method produces contact detection accuracy of up to 98.44% with a low noise simulation and 77.12% when utilizing data collect on the Sarcos Guardian XO robot (a hybrid humanoid/exoskeleton).
>
---
#### [replaced 025] Occlusion-Aware Consistent Model Predictive Control for Robot Navigation in Occluded Obstacle-Dense Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人导航任务，解决遮挡环境下障碍物密集区域的安全与运动一致性问题。提出一种考虑遮挡的CMPC策略，通过动态风险边界和多轨迹分支实现安全高效导航。**

- **链接: [https://arxiv.org/pdf/2503.04563v5](https://arxiv.org/pdf/2503.04563v5)**

> **作者:** Minzhe Zheng; Lei Zheng; Lei Zhu; Jun Ma
>
> **摘要:** Ensuring safety and motion consistency for robot navigation in occluded, obstacle-dense environments is a critical challenge. In this context, this study presents an occlusion-aware Consistent Model Predictive Control (CMPC) strategy. To account for the occluded obstacles, it incorporates adjustable risk regions that represent their potential future locations. Subsequently, dynamic risk boundary constraints are developed online to enhance safety. Based on these constraints, the CMPC constructs multiple locally optimal trajectory branches (each tailored to different risk regions) to strike a balance between safety and performance. A shared consensus segment is generated to ensure smooth transitions between branches without significant velocity fluctuations, preserving motion consistency. To facilitate high computational efficiency and ensure coordination across local trajectories, we use the alternating direction method of multipliers (ADMM) to decompose the CMPC into manageable sub-problems for parallel solving. The proposed strategy is validated through simulations and real-world experiments on an Ackermann-steering robot platform. The results demonstrate the effectiveness of the proposed CMPC strategy through comparisons with baseline approaches in occluded, obstacle-dense environments.
>
---
