# 机器人 cs.RO

- **最新发布 40 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] PhyPush: One Push is All You Need for Sensorless Physical Property Estimation with Physics-Guided Transformers
- **分类: cs.RO**

- **简介: 该论文属于物理属性估计任务，旨在无需传感器仅通过一次推动作业准确估算物体质量与摩擦系数。提出PhyPush框架，结合物理定律提升估计精度。**

- **链接: [https://arxiv.org/pdf/2605.26284](https://arxiv.org/pdf/2605.26284)**

> **作者:** Koyo Fujii; Luis Figueredo; Praminda Caleb-Solly; Ivan Boschi; Edoardo Ida'; Marco Carricato; Aly Magassouba
>
> **备注:** Submitted to 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems
>
> **摘要:** Accurately estimating object mass and friction is fundamental to achieving reliable and adaptive robotic manipulation. Although interactive perception provides a powerful mechanism for inferring such properties, most existing approaches depend on specialized hardware such as force/torque sensors, tactile arrays, or multi-camera motion-capture systems, limiting scalability and deployment. This paper presents PhyPush, a physics-guided Transformer framework that estimates an object's mass and friction coefficient using only kinematically derived end-effector velocity from a single push. This typically requires data available on standard robotic arms. The model incorporates constraints from Newton's second law and the Coulomb friction model through a physics-guided loss, improving physical consistency and generalization to unseen objects and surfaces. Across diverse simulation and real-world setups, PhyPush consistently achieves more accurate mass and friction estimation in challenging out-of-domain conditions. In simulation, it reduces error by over 10% compared with a baseline that has privileged access to full force information, while in real-world experiments, it outperforms a data-driven loss approach. Overall, the results demonstrate that physics-guided learning can enable low-cost, sensor-efficient estimation of physical properties, relying solely on a single push and readily available kinematic data.
>
---
#### [new 002] Multi-Robot Box Transport over Different Surfaces with Decentralized Role-based Proportional Control
- **分类: cs.RO**

- **简介: 该论文研究多机器人协作搬运任务，解决不同表面条件下物体运输的挑战。提出R2P2方法，通过角色分配和比例控制实现有效运输。**

- **链接: [https://arxiv.org/pdf/2605.26430](https://arxiv.org/pdf/2605.26430)**

> **作者:** Aditya Bhatt; Himavarshini Yarragangu; Urvish Shah; Venkata Sai Yaswanth Mohan Thota; Souma Chowdhury
>
> **摘要:** Collaborative transport of objects via pushing by multiple robots has many applications, ranging from construction and warehouse environments to post disaster debris clean-up. Achieving collaborative transport over surfaces with different inclination and friction properties however poses unique challenges. To address these challenges, this paper presents an asynchronous decentralized task and motion planning approach for transporting rectangular boxes of varying mass over flat, uphill and downhill terrain. Such a decentralized approach alleviates communication, synchronization and consensus needs and mitigates single point of failure issues. Our approach, called R2P2 or Roles with Rules and Proportional-control Primitive, assigns roles (e.g., push, support and prevent) to robots based on rules cognizant of the mode of manipulation needed (box rotation vs translation); this is followed by either rule-based control or proportional control of robot velocity based on the roles. Each robot is assumed to observe the location and heading of self and the box in executing the role and controls. R2P2 is evaluated with a six-robot team deployed in a simulator built using NVIDIA IsaacSim -- demonstrating generalizability across different surface friction/inclination and box mass scenarios, and better success rate compared to a standard virtual-leader-follower method. R2P2 is also successfully validated with a physical experiment, where it is executed onboard four turtlebots tasked with moving a 1.2 kg box.
>
---
#### [new 003] Manipulating Tangible Virtual Object Dynamics to Promote Learning of Precision Force Generation
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于康复训练任务，旨在提升精准力控制能力。通过操控虚拟物体动力学，测试不同模型对力控制训练效果的影响。**

- **链接: [https://arxiv.org/pdf/2605.26782](https://arxiv.org/pdf/2605.26782)**

> **作者:** Alberto Garzás-Villar; Alba Riera-Cardona; Alexis Derumigny; J. Micah Prendergast; Jane Murray Cramm; Laura Marchal-Crespo
>
> **摘要:** Robotic haptic devices combined with virtual reality offer novel opportunities to train fine force generation, an essential yet overlooked component of post-stroke rehabilitation. This study proposes that manipulating the rendered dynamics of tangible virtual objects can be leveraged to train precise force control while engaging the somatosensory system. We conducted an experiment with fifty healthy participants who performed a curling-inspired task in which they had to stretch a virtual spring to generate a target release force to propel the stone to a predefined location on the ice sheet. During training, the spring's force-elongation relationship was modeled as either a linear or non-linear function, i.e., a Gaussian or antisymmetric Gaussian (AS-Gaussian) function with zero derivative at the release target force. Results indicate that the AS-Gaussian group consistently achieved higher force accuracy during training than the linear group, while the Gaussian group only outperformed the linear group toward the end of training. Analysis of personality traits revealed that higher Free Spirit scores were associated with poorer performance and reduced task exploration under Gaussian dynamics, whereas higher Transform-of-Challenge scores correlated with increased exploration. Despite these training effects, no significant differences in long-term retention were found across spring types or personality traits. Participants primarily relied on learned target elongation rather than target force, as evidenced by performance in a transfer task with a different stiffness but the same target force. While promising for somatosensory neurorehabilitation, these methods require refinement to reduce reliance on proprioceptive cues before testing with neurological patients.
>
---
#### [new 004] Learning Compositional Symbolic Task Rules from Demonstrations with Inductive Logic Programming
- **分类: cs.RO**

- **简介: 该论文属于任务学习领域，旨在通过归纳逻辑编程从示范中学习可解释的符号化任务规则。工作包括分解任务、推理规则并实现知识复用，以提升任务结构的可理解性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.26828](https://arxiv.org/pdf/2605.26828)**

> **作者:** Oleh Borys; Karla Stepanova
>
> **备注:** In: ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy: From Environment Understanding and Reasoning to Safe Interaction, Vienna, 2026 In: ICRA 2026, International Joint Workshop on Ontologies, Semantic Maps and Autonomous Robotics Standardization (J-WOSMARS 2026), Vienna, 2026
>
> **摘要:** Learning from Demonstration~(LfD) should capture not only how a task is executed, but also its high-level task structure that explains the demonstrated behavior. As robots become more autonomous, such task representations must be inspectable, reusable, and human-interpretable. To address this, we study how to represent and learn robotic tasks with inductive logic programming~(ILP) by decomposing a complex task into a series of simpler learning objectives at different abstraction (ontological) levels. The system infers symbolic rules from demonstrations and prior (domain) knowledge, and reuses learned rules when learning higher-level task structure. We evaluate the approach in a synthetic block-assembly scenario and show that the learned abstractions are interpretable and support strong generalization to harder, held-out tasks with unseen objects. These results provide preliminary evidence that decomposed ILP is a feasible approach to task-level LfD.
>
---
#### [new 005] Learning to Balance Motor Thermal Safety and Quadrupedal Locomotion Performance with Residual Policy
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决电机过热限制四足机器人长时间运动的问题。通过强化学习框架，结合热模型训练出兼顾热安全与运动性能的策略。**

- **链接: [https://arxiv.org/pdf/2605.27046](https://arxiv.org/pdf/2605.27046)**

> **作者:** Yuhang Wan; Weixian Lin; Letian Qian; Yiqi Zou; Weiwei Wu; Shengwei Wu; Chuanlin Zhao; Xin Luo
>
> **摘要:** Motor thermal management is often overlooked in the context of electrically-actuated robots, particularly legged robots, but motor overheating is a key factor that limits long-duration locomotion especially under payload conditions. This paper integrates a whole-body thermal model of a quadruped robot into the reinforcement learning pipeline to update motor temperatures, and proposes a two-stage training framework for motor thermal management. In this framework, a nominal policy is first pre-trained as a locomotion baseline capable of traversing diverse terrains. A residual policy is then trained on top of the nominal policy to provide corrective actions based on the robot's thermal state, ensuring high performance under low-temperature conditions and preventing motor overheating under high-temperature conditions. Simulation results demonstrate that the proposed policy achieves an effective balance between motor thermal safety and locomotion performance. Real-world experiments on a Unitree A1 quadruped robot further validate the approach: under a 3 kg payload, the robot achieves stable locomotion across multiple terrains for over 13 minutes, while the nominal policy alone leads to motor overheating in about 5 minutes.
>
---
#### [new 006] Robust Koopman Control Barrier Filters for Safe Actor-Critic Reinforcement Learning
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于安全强化学习任务，旨在解决机器人系统在训练和部署中满足状态和输入约束的问题。通过结合Koopman框架与控制屏障函数，构建安全过滤器，提升RL的安全性与性能。**

- **链接: [https://arxiv.org/pdf/2605.26452](https://arxiv.org/pdf/2605.26452)**

> **作者:** Dhruv S. Kushwaha; Zoleikha A. Biron
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Safe reinforcement learning (RL) for robotic systems requires policies that improve task performance while satisfying state and input constraints during both training and deployment. Control barrier functions (CBFs) provide a principled mechanism for enforcing forward invariance through minimally invasive safety filters, but their use in model-free RL is limited by the need for accurate dynamics and hand-designed barrier certificates. We propose Robust Koopman-CBF SAC, a safety-filtered actor--critic framework that learns a finite-dimensional Koopman predictor from data, constructs affine CBF constraints in the lifted space, and enforces them through a quadratic-program safety layer. To account for finite-dimensional Koopman approximation error, the CBF condition is tightened using a projected residual margin estimated from held-out rollout data. The critic is trained on the executed safe action, while the actor is regularized toward the Koopman-CBF feasible set, reducing dependence on the filter over training. Across safe-control benchmarks, the method achieves zero constraint violations on CartPole stabilization and tracking while matching or exceeding unconstrained SAC returns. On high-dimensional Safety Gymnasium locomotion tasks, the method reduces violations in some settings but also exposes important limitations of first-order velocity barriers and linear EDMD models, motivating high-order and multi-step Koopman-CBF extensions. These results suggest that robust Koopman-CBF filters are a promising bridge between model-free RL and certifiable safety, while clarifying the structural conditions under which such filters remain effective. All code is available at \href{this https URL}{Github Repository}.
>
---
#### [new 007] Towards Drone-based Mapping of Volcanic Gases using Gas Tomography
- **分类: cs.RO**

- **简介: 该论文属于火山气体监测任务，解决无人机测量气体受气流干扰的问题。通过模型化气体断层成像技术，实现准确的火山气体分布映射。**

- **链接: [https://arxiv.org/pdf/2605.27180](https://arxiv.org/pdf/2605.27180)**

> **作者:** Marius Schaab; Niklas Karbach; Antonia Rabe; Thomas Wiedemann; Patrick Hinsen; Dmitriy Shutin; Thorsten Hoffmann; Achim J. Lilienthal
>
> **摘要:** Volcanoes emit large amounts of CO2, directly influencing human lives. Mapping volcanic gas emissions helps to forecast eruptions and understand the impact of volcanoes on climate and the environment. Drone-based gas sensing significantly reduces risks in volcanic monitoring but faces technical limitations when measuring gas, as rotor downwash disperses the gas plume before detection. Gas Tomography using remote gas sensing addresses this challenge. At the Salinelle dei Cappuccini mud volcanoes, we demonstrate that while drone-mounted in-situ sensors failed to detect CO2 emissions due to aerodynamic disturbance, open-path sensing successfully enabled remote gas distribution mapping. We present a novel model-based gas tomographic reconstruction approach that incorporates a Lagrangian model to compensate for wind-induced advection. The resulting gas distribution maps align with manually collected in-situ measurements, confirming that model-based gas tomography effectively overcomes downwash limitations and enables accurate mapping of volcanic emissions.
>
---
#### [new 008] Object Pose and Shape Estimation for Grasping: Does it Work?
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究物体位姿与形状估计在抓取中的应用，对比模块化方法与端到端方法的性能，探讨其在单视角RGB-D图像下的有效性。**

- **链接: [https://arxiv.org/pdf/2605.26944](https://arxiv.org/pdf/2605.26944)**

> **作者:** Pavan Karke; Kushal Shah; Gaurav Singh; Md Faizal Karim; K Madhava Krishna; Rajat Talak
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** The problem of object pose and shape estimation has seen key advancements lately. Encoder-decoder (e.g., SAM3D, LRM, CRISP) and diffusion-based models (e.g., InstantMesh, Zero123, SceneComplete) have shown category-agnostic shape encoding capacity and open-set generalizability. In this work, we ask the question: Are the object pose and shape estimation methods mature enough, such that when used with antipodal grasp sampling, can outperform the end-to-end grasp synthesis methods? We explore this question in detail by scoping our study to parallel jaw grippers, 7-DoF grasps, and single-view RGB(-D) image as input. We implement and compare a state-of-the-art, end-to-end grasp synthesis method and three modular methods, which first estimate the object pose and shape for all objects in the scene, and generate grasps using antipodal sampling. We observe that the modular methods outperform the end-to-end method in all our experiments. The modular methods are able to synthesize plenty of grasps, even for small objects, where the end-to-end methods fail. The effectiveness of the modular methods is contingent on the accuracy of the pose and shape estimation, and suffers partial degradation in cluttered scenes - a limitation of the existing pose and shape estimation methods. We also analyze the failure modes and run-times for the three modular methods, which use two different ways of object pose and shape estimation: one based on an encoder-decoder model, while another a diffusion model. Finally, we demonstrate that the single-view object pose and shape estimation methods can be augmented with vision-language models to yield language-conditioned grasps from just single-view RGB-D image as input. We notice comparable performance to the state-of-the-art LERF-TOGO baseline.
>
---
#### [new 009] FineVLA: Fine-Grained Instruction Alignment for Steerable Vision-Language-Action Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出FineVLA，解决机器人任务执行中语言指令与动作对齐问题，通过细粒度监督提升策略可操控性。**

- **链接: [https://arxiv.org/pdf/2605.27284](https://arxiv.org/pdf/2605.27284)**

> **作者:** Xintong Hu; Xuhong Huang; Jinyu Zhang; Yutong Yao; Yuchong Sun; Qiuyue Wang; Mingsheng Li; Sicheng Xie; Yitao Liu; Junhao Chen; Yixuan Chen; Yingming Zheng; Shuai Bai; Tao Yu
>
> **备注:** 26 pages, 7 figures, 25 tables
>
> **摘要:** Vision-Language-Action (VLA) models are increasingly expected to not only complete robot tasks, but also follow human instructions about how those tasks should be executed. However, existing robot datasets usually pair trajectories with coarse goal-level language, leaving execution-critical details such as active arm, approach direction, and contact region unspecified. This limits steerable policy learning and robotic video understanding. We introduce FineVLA, an open framework for action-aligned fine-grained VLA supervision. The framework includes: (1) a data construction tool that unifies 972,247 trajectories across 85K tasks from 10 open-source robot datasets and builds FineVLA-Data, a human-verified dataset of 47,159 fine-grained trajectories; (2) a held-out benchmark with 500 videos, 10,816 atomic facts, and 1,030 VQA questions; (3) a robotics-specialized VLM annotator for scalable fine-grained annotation; and (4) a steerable VLA policy trained with controlled mixtures of fine-grained and raw goal-level instructions. Our experiments yield three findings. First, fine-grained supervision does not sacrifice goal-level success: FG-only improves over Raw-only by +1.4 to +8.1 success-rate points across settings. Second, fine-grained and raw instructions are complementary, following a consistent inverted-U trend peaking at FG:Raw = 1:2 to 1:1. The best mixed setting reaches 86.8%/82.5% in RoboTwin simulation and 62.7/100 in real-world dual-arm manipulation (vs. 49.9 Raw-only). Third, fine-grained supervision improves steerable control: the largest real-world gains appear on pose (+23), color (+18), and approach direction (+18)--factors where goal-level instructions provide no guidance. Overall, fine-grained language should augment goal-level instructions: specifying how to execute alongside what to achieve. Project page: this https URL
>
---
#### [new 010] On the Generalization Capabilities, Design Choices and Limitations of Keypoint Imitation Learning
- **分类: cs.RO**

- **简介: 该论文研究机器人操作中的关键点模仿学习，解决数据效率与泛化能力问题。通过结合视觉基础模型，提升模仿学习性能，并评估其在不同任务中的表现。**

- **链接: [https://arxiv.org/pdf/2605.26649](https://arxiv.org/pdf/2605.26649)**

> **作者:** Thomas Lips; Marco Moletta; Michael C. Welle; Danica Kragic; Francis wyffels
>
> **备注:** This version was submitted to IROS 2026
>
> **摘要:** RGB-based imitation learning requires many demonstrations to generalize to unseen objects or scenes, motivating research into intermediate representations to improve generalization for robotic manipulation. Visual foundation models enable one-shot extraction of keypoints to provide such representation. However, it remains unclear how to integrate them into imitation learning optimally and when they outperform alternative representations. We combine approaches from previous works on keypoint imitation learning (KIL) and investigate several design choices to provide practical guidelines. Using over 2000 real-world rollouts, we also assess the generalization capabilities of KIL to unseen objects and scene variations. KIL achieves a 75% overall success rate across five tasks, significantly outperforming the RGB baseline (47%) and performing on par with S2-diffusion (73%). Finally, we explore the limitations of the foundation models used for keypoint extraction and extend KIL to tasks with multiple object instances. Our results confirm KIL as a data-efficient approach for robot learning, though it does not outperform alternative representations and inherits limitations of the foundation models used for keypoint extraction. All rollout videos, demonstrations, and results are available at this https URL.
>
---
#### [new 011] When Does Adaptive Guidance Help? Belief-Aware Privileged Distillation for Autonomous Driving Under Partial Observability
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自主驾驶任务，解决部分可观测环境下的知识蒸馏问题。提出BA-GSAC方法，通过集成分歧调整蒸馏系数，研究自适应引导的有效性。**

- **链接: [https://arxiv.org/pdf/2605.26155](https://arxiv.org/pdf/2605.26155)**

> **作者:** Mehmet Haklidir
>
> **备注:** 9 pages, 3 figures, 7 tables. Accepted at CVPR 2026 Workshop on Autonomous Driving (WAD)
>
> **摘要:** Guided Soft Actor-Critic (GSAC) distills knowledge from a privileged full-state teacher to a partial-observation student for autonomous driving, but uses a fixed distillation coefficient lambda regardless of the agent's uncertainty. We present Belief-Aware GSAC (BA-GSAC), which modulates lambda via ensemble disagreement, and use it as a testbed for a systematic empirical study asking: when does adaptive guidance actually help? Evaluating five strategies (fixed lambda in {0.01, 0.1}, adaptive, linear decay, and vanilla SAC) across three POMDP difficulty levels on Highway-Env, we find that preliminary single-seed runs suggest benefits under mild and moderate partial observability, but under severe occlusion (evaluated with 3 seeds for all methods) the adaptive coefficient collapses to lambda_min within about 3K steps. We trace this to an observability blindness phenomenon: because the ensemble predicts partial observations, it achieves low disagreement even under heavy occlusion, modeling what is visible but unable to detect what is missing. We diagnose the root cause and propose an architectural fix (training the ensemble on full-state predictions using the guiding actor's privileged access); while not validated here, we show that even with current limitations, the warmup phase provides measurable stabilization (CV=13.3% vs. 29.8% for constant lambda=0.01). In fact, a simple deterministic linear decay schedule achieves the best severe-POMDP performance across all metrics (mean 116.5, CV=8.9%), suggesting that the scheduling effect, not the ensemble, drives the stability benefit. These findings provide practical guidance for designing uncertainty-aware teacher-student frameworks and highlight ensemble prediction targets as an important design choice.
>
---
#### [new 012] A Bioinspired Underwater Robot with a Latch-Mediated Soft Bistable Mechanism
- **分类: cs.RO**

- **简介: 该论文属于水下机器人设计任务，旨在解决传统动力源能量密度低的问题。通过仿生学设计，提出一种具有锁扣机制的软双稳驱动器，实现高效推进与灵活运动。**

- **链接: [https://arxiv.org/pdf/2605.26936](https://arxiv.org/pdf/2605.26936)**

> **作者:** Chongze Bi; Wenjie Wu; Zonghao Zuo; Li Wen
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Underwater robotics has advanced significantly over recent decades. however, the development of miniaturized underwater robots remains limited by low energy densities of traditional power sources. Nature offers compelling solutions-organisms like mantis shrimps and fleas utilize latch-mediated spring actuation (LaMSA) systems that achieve rapid movements through a decoupled energy storage and release mechanism. Despite extensive studies of LaMSA, replicating such rapid, asymmetric actuation within simple, compact structures remains challenging. In this work, we introduce a bioinspired, soft bistable actuator with an integrated latch mechanism that enables asymmetric energy input and release using a single motor. Coupled with fin structures, this design facilitates efficient underwater propulsion and maneuverability. Experimental results demonstrate stable periodic flapping, precise steering, and a maximum thrust of 0.528 N, impulse of 0.147 Ns, and vertical displacement of 30 mm. By modulating fin angles, the robot achieves versatile motions, including vertical ascent, diagonal forward movement, and lateral translation. This study presents a novel, energy-efficient approach for controlling motion in compact underwater robots, paving the way for advanced biomimetic designs with potential applications in exploration, environmental monitoring, and inspection.
>
---
#### [new 013] Trust, Geometry, and Rules: A Credibility-Aware Reinforcement Learning Framework for Safe USV Navigation under Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于USV安全导航任务，解决动态海况下感知不确定性导致的导航不安全问题。提出融合可信度、几何安全和规则感知的强化学习框架，提升导航鲁棒性和合规性。**

- **链接: [https://arxiv.org/pdf/2605.26974](https://arxiv.org/pdf/2605.26974)**

> **作者:** Yuhang Zhang; Shuqi Chai; Yukang Zhang; Liusha Yang; Mingchuan Zhang; Wei Wang; Qingjiang Shi; Quanbo Ge
>
> **摘要:** Autonomous navigation of Unmanned Surface Vehicles (USVs) that is safe and compliant with the International Regulations for Preventing Collisions at Sea (COLREGs) remains a formidable challenge in dynamic maritime environments, particularly when perception systems exhibit miscalibrated uncertainty. Existing Reinforcement Learning (RL)-based methods often falter because state-estimation errors induce unreliable belief states that mislead the value function, while discrete traffic rules introduce discontinuity in the learning objective. To address these challenges, we propose a framework integrating credibility-aware learning, geometric safety shielding, and continuous rule-aware embedding. First, Credibility-Weighted Value Learning (CW-VL) introduces a dynamic trust factor derived from the discrepancy between filter-estimated covariance and empirical error statistics to modulate the critic's heteroscedastic loss, preventing policy overfitting to noisy samples. Second, the Covariance-Inflated Velocity Obstacle (CI-VO) maps position-estimation uncertainty into set-wise angular margins, forming a conservative geometric shield that overrides hazardous exploratory actions. Third, Risk-Aware COLREGs Duty Embedding relaxes binary encounter duties into continuous rule-aware signals, providing smooth sector-transition information and suppressing oscillation from sparse rule rewards. Simulated encounter studies demonstrate improved training robustness against perceptual inconsistency and superior collision avoidance and COLREGs compliance over baselines.
>
---
#### [new 014] Towards Shared Embodied Intelligence in Humanoid Robots through Optimization Development and Testing of the Human Aware ergoCub Robot
- **分类: cs.RO**

- **简介: 论文提出一种融合共享智能与具身认知的机器人架构，解决人机协作安全有效的问题。通过优化硬件和控制参数，实现人机协同，应用于工业与辅助机器人。**

- **链接: [https://arxiv.org/pdf/2605.26991](https://arxiv.org/pdf/2605.26991)**

> **作者:** Carlotta Sartore; Mohamed Elobaid; Lorenzo Rapetti; Giulio Romualdi; Stefano Dafarra; Nicola A. Piga; Ines Sorrentino; Paolo Maria Vicecone; Silvio Traversaro; Ugo Pattacini; Luca Fiorio; Francesco Draicchio; Giovanna Tranfo; Lorenzo Natale; Marco Maggiali; Daniele Pucci
>
> **摘要:** Collaboration is central to human behavior, enabling tasks beyond individual capability. This ability arises from coordinating actions through internal representations of others, a concept known as shared intelligence. Additionally, humans are characterized by physical bodies and cognitive abilities that are optimized in response to their environment, a phenomenon referred to as embodied cognition. Designing humanoid robots that collaborate safely and effectively with people requires unifying these principles. Here we propose an architecture that integrates shared intelligence and embodied cognition to enable robots to physically collaborate with humans, where robot hardware and control are optimized for human metrics, using representations of the human body and motion intelligence. The ultimate goal is to achieve a form of shared embodied intelligence. Specifically, our architecture optimizes robot hardware and physical intelligence parameters with respect to human ergonomic metrics. This is accomplished by modeling human-robot interaction as a function of hardware configurations and embedding human models into the robot's physical intelligence. As a concrete implementation, we present the humanoid robot ergoCub, whose morphology and control have been optimized for collaborative tasks with humans. Our approach provides a framework for designing humanoid robots that prioritize human ergonomics at both the hardware and physical intelligence levels, with applications in industrial and assistive robotics.
>
---
#### [new 015] Closing the Loop in Teleoperation: Episode-Level Data Quality Assessment and Feedback for High-Quality Demonstration Collection
- **分类: cs.RO**

- **简介: 该论文属于机器人示范数据收集任务，旨在解决 teleoperation 中示范质量低的问题。提出 DQAF 框架，通过实时反馈提升示范质量。**

- **链接: [https://arxiv.org/pdf/2605.26349](https://arxiv.org/pdf/2605.26349)**

> **作者:** Gokul Narayanan; Yash Shahapurkar; Melih Erdogan; Brian Zhu; Eugen Solowjow
>
> **摘要:** Industrial automation is at a pivotal moment, as Physical AI is driving a transition from rigid, hand-engineered automation systems toward more flexible and adaptive systems. This shift has created a growing demand for large-scale, real-world robot demonstration data, making teleoperation an increasingly important mechanism for data collection. However, high-quality teleoperated demonstrations remain difficult to obtain in practice, as novice operators often produce episodes that are task-successful but suboptimal for downstream use due to inefficient motion, repeated corrections, or operation near robot joint limits. We present a Data Quality Assessment and Feedback (DQAF) framework that closes the loop in teleoperation by providing immediate post-episode feedback grounded in semantic task progress and robot telemetry. The framework extracts quality relevant signals such as sub-task progress, motion smoothness, stalls, kinematic limits and converts them into structured quality assessments and actionable natural-language feedback. Unlike binary success or failure feedback, the proposed system explains why an episode is suboptimal and highlights specific behaviors to correct in the next trial. We evaluate the framework through a diagnostic validation study and a pilot user study. In the validation study, the system is compared with a human reviewer during dataset curation, producing rejection reasons and actionable feedback for improvement. In the pilot study with three novice operators across two manipulation tasks, the operator who received the systems immediate, automated post-episode feedback improved faster than those who did not, producing higher-quality demonstrations sooner.
>
---
#### [new 016] VR-DAgger: Immersive VR for Dexterous Data Collection and Uncertainty-Guided On-Policy Correction
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决数据收集不足和分布偏移导致的性能下降问题，提出VR-DAgger方法进行高效数据收集与策略修正。**

- **链接: [https://arxiv.org/pdf/2605.27114](https://arxiv.org/pdf/2605.27114)**

> **作者:** René Zurbrügg; Tifanny Portela; Arjun Bhardwaj; Aravind Elanjimattathil Vijayan; Maximum Wilder-Smith; Marco Hutter
>
> **摘要:** Learning from demonstrations is effective for robotic manipulation, but collecting sufficient task-specific data remains a major bottleneck. Under distribution shift, small errors compound, performance degrades, and expert time is often spent on redundant, low-value corrections instead of the few critical failure cases.
>
---
#### [new 017] Provably Safe Motion Planning Under Unknown Disturbances
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动规划任务，解决在未知扰动下保证安全的问题。通过学习Wasserstein模糊带，构建满足约束的采样规划树，提升算法安全性和效率。**

- **链接: [https://arxiv.org/pdf/2605.26625](https://arxiv.org/pdf/2605.26625)**

> **作者:** Ibon Gracia; Qi Heng Ho; Luca Laurenti; Morteza Lahijanian
>
> **摘要:** We present a provably safe sampling-based motion planning algorithm for robotic systems affected by random disturbances of unknown distribution. We consider systems with linear or linearizable dynamics evolving in workspace with arbitrary-shaped obstacles subject to state and control constraints. Safety requirements are formulated as chance-constraints. Our approach leverages data from trajectories of the system to learn a Wasserstein ambiguity tube, i.e., a sequence of ambiguity sets, which contains the trajectory of the system's state distribution with high confidence. This ambiguity tube is then used in a probabilistically complete algorithm to grow a sampling-based motion planning tree that respects the constraints of the problem. We show that learning several lower-dimensional ambiguity tubes instead of a single high-dimensional one effectively reduces the conservatism and boosts scalability. Additionally, we design an efficient bandit-based validity checker that remarkably increases the empirical performance of our approach without sacrificing probabilistic completeness. Case studies show our algorithm finds valid plans in cluttered environments under strict safety thresholds, outperforming state-of-the-art methods.
>
---
#### [new 018] RCSP: Risk-Sensitive Conjectural Scenario Planning for Safe Dynamic Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人动态导航任务，解决预测性近撞风险问题。提出RCSP方法，通过评估障碍物未来状态，提升导航安全性与路径质量。**

- **链接: [https://arxiv.org/pdf/2605.26348](https://arxiv.org/pdf/2605.26348)**

> **作者:** Zhengye Han; Quanyan Zhu
>
> **摘要:** Mobile robots can fail before they collide: a velocity that is safe now may commit the robot to a passage that moving obstacles will soon close. We study this predictive near-miss commitment problem and propose Risk-Sensitive Conjectural Scenario Planning (RCSP), a planning layer that evaluates candidate commands against plausible short-horizon obstacle futures. RCSP maintains a lightweight belief over local motion conjectures, samples future interactions, penalizes high-risk tails, and executes through a local safety check. In controlled MuJoCo bottleneck tasks, the RCSP planner reaches the goal without collisions and yields higher secondary safety and path-quality point estimates than a non-adaptive predictor, with additional latency. In ROS2/Gazebo, adding the local safety layer to a standard Nav2 stack reduces dynamic near-miss failures. On official DynaBARN/Jackal transfer, tuned DWA and TEB remain stronger on strict benchmark success, revealing the boundary of the approach. These simulation results position RCSP as a predictive-risk module that complements existing navigation stacks in dynamic bottleneck regimes.
>
---
#### [new 019] HyperSim: A Holistic Sim-To-Real Framework For Robust Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出HyperSim框架，解决机器人操作中的sim-to-real迁移问题。通过合成数据、对抗轨迹生成和联合训练，提升策略在真实环境中的表现与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26638](https://arxiv.org/pdf/2605.26638)**

> **作者:** Junyi Dong; Haotian Luo; Ziwei Xu; Shengwei Bian; Heng Zhang; Sitong Mao; Jingyi Guo; Yang Xu; Wenhao Chen; Qiuyu Feng; Yao Mu; Ping Luo; Shunbo Zhou; Xiaodong Wu
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Scaling data volume and diversity is critical for generalizing embodied intelligence. While synthetic data generation offers a scalable alternative to expensive physical data acquisition, transferring robotic manipulation policies from simulation to the real world (sim-to-real) remains a formidable challenge due to the domain gap. This paper presents HyperSim, a holistic framework spanning from synthetic data generation to policy training and seamless real-world deployment. To systematically bridge the sim-to-real gap, HyperSim is realized through three core pillars: high-fidelity environment synthesis, adversarial trajectory generation, and sim-and-real co-training. Collectively, these modules address domain discrepancies by enhancing visual fidelity, expanding data coverage, and enforcing domain-invariant representations. We rigorously validate HyperSim through a large-scale empirical study involving 400 real-world task executions across two representative manipulation models. Assessed across three fine-grained metrics, our complete pipeline achieves remarkable sim-to-real success rates of 80% and 95% with ACT and \pi_{0}, respectively. Furthermore, policies trained on our adversarial trajectories exhibit significantly enhanced robustness against dynamic uncertainties, achieving a 35% higher completion rate under physical perturbations.
>
---
#### [new 020] Efficient On-policy Visual-RL via Stochastic Decoupled Policy Gradient
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; eess.SY**

- **简介: 该论文提出SDPG，一种轻量级视觉强化学习方法，解决高效训练视觉-运动控制策略的问题，通过随机扰动轨迹估计梯度，提升训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2605.26478](https://arxiv.org/pdf/2605.26478)**

> **作者:** Haoxiang You; Yilang Liu; Davis Zong; Qian Wang; Teeratham Vitchutripop; Qi Wang; Daniel Rakita; Ian Abraham
>
> **摘要:** We present the stochastic decoupled policy gradient (SDPG), a lightweight visual reinforcement learning (RL) method that trains diverse visuomotor control policies end-to-end within a few hours on a single NVIDIA RTX 4080 GPU. SDPG estimates policy gradients via random perturbations of trajectory rollouts, requiring orders of magnitude fewer batch-rendered environments and substantially reducing compute and memory overhead. On visual MuJoCo benchmarks, SDPG consistently outperforms baseline methods in training time, memory usage, and rewards. Finally, to support future research, we introduce a suite of realistic visual robotics benchmarks spanning dexterous manipulation, challenging locomotion, and demonstrate effective sim-to-real transfer on physical hardware.
>
---
#### [new 021] Riding the Shifting Potential: When Reactive Control Suffices for Multi-Goal Behavior
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究多目标行为中的反应式控制问题，解决冲突目标导致局部最优的难题。通过动态优先级和空域投影方法，提升导航与物体推动任务的性能。**

- **链接: [https://arxiv.org/pdf/2605.27314](https://arxiv.org/pdf/2605.27314)**

> **作者:** Vito Mengers; Oliver Brock
>
> **摘要:** Reactive control is often considered insufficient for multi-objective tasks because conflicting objectives give rise to local minima. We argue this limitation is not inherent but arises from static encodings that fail to reflect how objectives currently interact. We exploit the interaction structure encoded in a graph-based world model by extending it with nullspace projections: conflicts are resolved where they arise by projecting lower-priority gradients into the nullspace of higher-priority ones, with priorities determined continuously from the current state. We demonstrate this in two domains where conflicts between objectives are central: navigation around non-convex obstacles, where static potential fields fundamentally fail, and planar pushing of non-convex objects, where our method achieves $100\%$ success across one-hundred configurations versus $0\%$ for the steepest-descent baseline and ${\sim}55\%$ for diffusion policy, without demonstrations or retraining. The same formulation transfers directly to a real robot with additional perceptual and kinematic constraints, accommodating them through the same mechanism.
>
---
#### [new 022] L-Learning : A Lyapunov-Based Approach Leveraging Lagrangian Mechanics for Efficient and Stable Robot Tracking
- **分类: cs.RO**

- **简介: 该论文提出L-Learning，结合李雅普诺夫稳定性理论与拉格朗日力学，解决机器人轨迹跟踪问题，提升控制精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.26648](https://arxiv.org/pdf/2605.26648)**

> **作者:** Quan Quan; Hao Li
>
> **备注:** 9 pages, 4 figures, 4 tables
>
> **摘要:** This paper presents L-Learning, a novel data-driven control framework for robotics that integrates Lyapunov stability theory with Lagrangian mechanics to enhance trajectory tracking performance. While traditional control methods often suffer from performance degradation in dynamic and uncertain environments, data-driven approaches, while more adaptable, are frequently limited by high sample complexity and a lack of rigorous stability guarantees. L-Learning mitigates these challenges by explicitly learning the system's energy function from data, thereby optimizing performance while ensuring closed-loop stability intrinsically. Characterized by superior control accuracy, theoretical stability guarantees, and high sample efficiency, L-Learning represents a promising solution for practical robotic applications.
>
---
#### [new 023] Heterogeneous AAV Logistics Task Allocation: A Reinforcement Learning Enhanced Overlapping Coalition Formation Game Approach
- **分类: cs.RO**

- **简介: 该论文属于动态物流任务分配问题，针对异构AAVs在随机任务下的优化难题，提出一种融合强化学习的重叠联盟形成方法，提升任务分配效率。**

- **链接: [https://arxiv.org/pdf/2605.26471](https://arxiv.org/pdf/2605.26471)**

> **作者:** Yuze Zhou; Jingliang Sun; Junzhi Li; Jianxin Zhong; Zihan Wang; Teng Long
>
> **备注:** 12 pages
>
> **摘要:** In dynamic urban logistics, the stochastic emergence of time-sensitive tasks poses a significant optimality challenge for heterogeneous AAVs logistics task allocation. To address this problem, a reinforcement learning enhanced overlapping coalition formation game approach is proposed. A dynamic task allocation model is established, where global optimality is mathematically quantified by a generalized logistics cost coupling service quality and resource consumption. To deal with the time-varying task sets induced by stochastic order arrivals, a transformer-based soft actor-critic network is designed. By leveraging multi-head self-attention to encode variable-length logistics states and capture task-wise spatiotemporal dependencies, the learned policy adaptively guides coalition updates, replacing heuristic rules in the overlapping coalition formation game. On this basis, heterogeneous AAVs can form more efficient overlapping coalitions for dynamic logistics tasks. The resulting coalition formation process is proven to constitute an exact potential game, which guarantees convergence to a Nash-stable equilibrium within a finite number of iterations. Numerical simulations demonstrate that the proposed algorithm effectively improves the optimality of task allocation under the generalized logistics cost criterion. In a scenario with 32 AAVs and 80 tasks, our algorithm achieves a 39.76% cost reduction compared with the heuristic OCF baseline. Indoor flight experiments further validate its practicality.
>
---
#### [new 024] Look Further: Socially-Compliant Navigation System in Residential Buildings
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在改善人机交互体验。通过延长反应距离和引入PLC模式，提升用户对机器人行为的安全感、流畅性和礼貌性感知。**

- **链接: [https://arxiv.org/pdf/2605.26710](https://arxiv.org/pdf/2605.26710)**

> **作者:** Akira Shiba; Marina Obata; Nathan Kau; Zoltan Beck; Rishi Shah; Michael Sudano; Sabrina Lee
>
> **备注:** 2025 ACM/IEEE International Conference on Human-Robot Interaction
>
> **摘要:** The distance at which a mobile robot reacts to a person strongly impacts various qualities of the human-robot interaction. In this paper, we focus on the navigation of a mobile delivery robot platform in a residential indoor hallway environment. Social navigation methods typically focus on avoiding uncomfortable human-robot interactions, such as when a robot encroaches on someone's personal space. Since personal space has been shown to be in the range of just a few meters, social navigation methods typically focus on deconflicting and resolving these short-range interactions. In this work, however, we demonstrate that by extending the reaction distance to over eight meters, far beyond the typical interaction distance, we can improve the human's perception of the robot's motion. We introduce the Proactive Lane-Changing (PLC) motion pattern and a navigation system that leverages it to react to people at an increased distance. This pattern consists of changing the robot's lateral position as it navigates down the hallway from the center to the side at an eight-meter distance from an oncoming person. We conducted a user study with 42 participants to assess their impressions of the delivery robot based on three service objectives: safety, smoothness, and politeness. In the straight hallway scenario (Frontal Approach), results showed significant improvement in each of these three objectives compared to typical motion patterns found in the literature: slowing down, stopping, and reactive collision avoidance in the proximity of a person. In contrast, in the intersection (Blind Corner) scenarios, none of the approaches performed significantly better than any other, with participants having a diverse range of preferences among robot motion patterns.
>
---
#### [new 025] SteelDS: A High-Resolution Video Dataset of E40 Steel Scrap for Object Detection and Instance Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SteelDS数据集，用于钢渣中铜杂质的检测与分割。解决工业自动分拣中的材料识别问题，包含高分辨率标注视频数据。**

- **链接: [https://arxiv.org/pdf/2605.26682](https://arxiv.org/pdf/2605.26682)**

> **作者:** Melanie Neubauer; Christian Rauch; Gerald Koinig; Alexia Tischberger-Aldrian; Roland Pomberger; Elmar Rueckert
>
> **摘要:** This dataset provides high-resolution, annotated video sequences of shredded E40-grade steel and copper scrap on a conveyor belt. Captured in a controlled laboratory environment, the data reflects the industrial post-magnetic sorting stage, where manual intervention is typically required to remove copper contaminants. The dataset comprises 24,297 labeled frames across five subsets, featuring 396 steel and 101 copper objects categorized by size. It supports the development of machine learning models for material classification, object detection, and instance segmentation. Variations in object spacing and density are included to simulate realistic industrial sorting conditions. Ground truth annotations include pixel-wise segmentation masks and material classes. This dataset serves as a benchmark for evaluating automated sorting algorithms aiming to identify copper impurities within complex, heterogeneous steel scrap streams.
>
---
#### [new 026] Can VLA Models Learn from Real-World Data Continually without Forgetting?
- **分类: cs.RO**

- **简介: 该论文属于机器人持续学习任务，旨在解决VLA模型在真实世界中不断学习新技能时的遗忘问题。通过构建真实数据集并测试经验回放方法，提出有效解决方案。**

- **链接: [https://arxiv.org/pdf/2605.26820](https://arxiv.org/pdf/2605.26820)**

> **作者:** Jiarun Zhu; Yijun Hong; Xiaoquan Sun; Zetian Xu; Mingqi Yuan; Zhiyong Wang; Wenjun Zeng; Jiayu Chen
>
> **摘要:** Vision-language-action (VLA) models provide a promising foundation for general-purpose robotics. However, their successful deployment in real-world scenarios requires the ability to continually acquire new skills while retaining previously learned behaviors. While pioneering research has studied the continual learning of VLA models in narrowly simulated environments, this challenge remains largely unexplored under realistic conditions. To address this limitation, we construct a real-world continual learning dataset comprising four sequential manipulation tasks, spanning rigid-object pick-and-place, contact-rich pressing, and deformable-object folding. Using this dataset, we conduct comprehensive experiments and find that VLA models suffer significant catastrophic forgetting when continually learning from heterogeneous real-world demonstrations. We then systematically evaluate experience replay and uncover key implementation factors that govern its success. In summary, this work provides the first empirical study of real-world continual VLA learning and offers practical guidance for deploying long-lived robot policies.
>
---
#### [new 027] Enabling Extensible Embodied Capabilities with Tools
- **分类: cs.RO**

- **简介: 该论文属于具身智能任务，旨在解决单一模型难以有效整合感知、推理等异构能力的问题。通过提出ETP协议和EmbodiedToolBench，将能力外部化为独立工具，提升具身性能。**

- **链接: [https://arxiv.org/pdf/2605.26637](https://arxiv.org/pdf/2605.26637)**

> **作者:** Xueyang Zhou; Zijia Wang; Qianjiang Li; Yibo Hu; Guiyao Tie; Li Wan; Yidan Liu; Pan Zhou; Lichao Sun; Yongchao Chen
>
> **备注:** 51 pages, 20 figures,
>
> **摘要:** Most existing embodied intelligence methods formulate perception, reasoning, planning, and control within a unified parameterized policy. Yet these capabilities are inherently hierarchical and heterogeneous, making them difficult to reliably learn and modularize within a single model. We propose a capability externalization approach that decouples heterogeneous capabilities into independently optimized tools, dynamically invoked at inference time. To this end, we introduce Embodied Tool Protocol (ETP), a standardized protocol for embodied tool registration, discovery, invocation, and execution, and curate 100+ validated tools spanning perception, cognition, reasoning, and execution as the tool base. Building on this, we construct EmbodiedToolBench to evaluate both whether tool augmentation improves embodied performance and how well current models use tools across tool-necessity recognition, tool selection, tool execution, and tool-chain composition. Experiments across simulation and real-world platforms confirm that capability externalization consistently improves embodied performance (avg. gain 31% on EB-ALFRED and 36% on EB-Navigation), yet reveal a clear boundary: gains are substantial for cognition and perception but are limited for execution-type capabilities. Moreover, our analysis reveals that knowing when, which, and how to invoke tools remains a persistent challenge across all models, thereby highlighting embodied tool competence as a critical direction for future research.
>
---
#### [new 028] TPS-Drive: Task-Guided Representation Purification for VLM-based Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出TPS-Drive，解决VLM在自动驾驶中语义推理与3D空间预测的难题，通过任务引导的表示净化提升空间建模能力。**

- **链接: [https://arxiv.org/pdf/2605.27038](https://arxiv.org/pdf/2605.27038)**

> **作者:** Jiaxiang Li; Yumao Liu; Ke Ma
>
> **摘要:** Vision-Language Models (VLMs) provide a promising foundation for autonomous driving planning, yet bridging semantic reasoning and precise 3D spatial forecasting remains a critical challenge. Existing representation strategies generally follow two paths: text-aligned methods flatten continuous spatial states into symbols, which compromises geometric structure and induces "spatial hallucinations"; dense visual methods preserve spatial topology but overwhelm standard tokenizers with redundant background textures, leading to "representation interference". To address these limitations, we introduce TPS-Drive, a novel framework centered on Task-Guided Representation Purification that empowers VLMs to Think in Purified Space. At its core, an Agent-Centric Tokenizer utilizes a task-guided vector quantization mechanism supervised by a frozen 3D detection head, which explicitly reallocates limited codebook capacity from pervasive static backgrounds to critical dynamic agents and effectively isolates spatial redundancy. Leveraging this purified spatial vocabulary, TPS-Drive employs a decoupled reasoning pipeline that sequentially performs scene understanding, future forecasting, and action generation. The framework is optimized via a progressive three-stage training paradigm, culminating in reward-driven refinement that surpasses pure imitation learning. Extensive experiments validate our approach: TPS-Drive achieves accurate agent spatial state forecasting and reduces collision rates in open-loop nuScenes evaluations, while establishing new safety records on the rigorous closed-loop NAVSIMv1 and NAVSIMv2 benchmarks.
>
---
#### [new 029] Collaborative Navigation and Exploration with $β$-Sparse Gaussian Processes
- **分类: cs.RO**

- **简介: 该论文研究多机器人协同导航与探索任务，解决未知环境中信息传输和计算限制问题。提出β-Sparse Gaussian Processes模型，优化地图点选择与导航策略，提升探索效率。**

- **链接: [https://arxiv.org/pdf/2605.26304](https://arxiv.org/pdf/2605.26304)**

> **作者:** Evangelos Psomiadis; Dipankar Maity; Panagiotis Tsiotras
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Collaborative navigation of heterogeneous robots in unknown environments poses significant challenges due to sensing, communication, and computational limitations. In this work, a lead robot navigates toward a target while a mobile sensor robot (e.g., a drone) assists by transmitting information about its locally observed environment under bandwidth constraints. We propose a framework that enables the sensor to jointly select its transmitted map points and navigation actions online, while also predicting unexplored regions of the environment. To this end, we present $\beta$-Sparse Gaussian Processes, a novel and robust variational sparse Gaussian Process model for task-aware inducing point selection. Furthermore, we develop an action-selection strategy that balances task relevance with exploration. Simulations on Mars and Earth maps show that the framework can reduce path cost by 18% relative to no communication and decrease transmitted information by 76% compared to raw-data transmission baselines.
>
---
#### [new 030] NightSight: Passive Computation for Navigation in Dark Using Events
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决黑暗环境中小型无人机的感知问题。通过结合事件相机、编码光圈和红外点投影，实现高精度深度估计。**

- **链接: [https://arxiv.org/pdf/2605.26330](https://arxiv.org/pdf/2605.26330)**

> **作者:** Deepak Singh; Brijan Vaghasiya; Shreyas Khobragade; Nitin Sanket
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Small aerial robots are particularly well-suited for search and rescue in confined and hazardous environments due to their agility, low cost, and ability to traverse through cluttered spaces that are inaccessible to larger platforms. However, enabling autonomous navigation in complete darkness remains a significant challenge, because small aerial robots cannot easily accommodate perception systems that demand substantial payload, power, or computation. In this work, we present a lightweight perception approach that combines a monocular event camera, a coded aperture lens, and an infrared dot projector to enable navigation in such conditions. The projected pattern, when imaged through the coded aperture, produces depth dependent blur signatures that implicitly encode scene geometry. We train a convolutional neural network to decode these signatures into dense depth maps using only synthetic data generated from a simple planar wall setup. Despite this minimal training regime, the model generalizes zero-shot to complex real-world scenes. Our system operates in real time at 20 Hz on a NVIDIA Jetson Orin Nano, demonstrating suitability for resource-constrained platforms. We further analyze the impact of different coded aperture designs on depth estimation performance. Our approach gives high accuracy (l1 error 7.0cm) upto 2.5m range (2.80% error). These results highlight the potential of combining structured illumination, coded optics, and event-based sensing for enabling robust perception and navigation in complete darkness.
>
---
#### [new 031] TCBiRRT: Rapid Motion Planning for Tightly Coupled Dual-arm Space Manipulator Using Task-space Random Expansion
- **分类: cs.RO**

- **简介: 该论文属于空间机械臂运动规划任务，解决紧耦合双臂在闭环约束下的路径规划问题。提出TCBiRRT算法，在任务空间进行采样与扩展，提升规划效率与成功率。**

- **链接: [https://arxiv.org/pdf/2605.27167](https://arxiv.org/pdf/2605.27167)**

> **作者:** Jiawei Zhang; Xinhao Miao; Jifeng Guo; Qinghua Li; Chengchao Bai
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Planning the motion path for a tightly coupled dual-arm space manipulator under closed-chain constraints is a fundamental yet challenging problem in on-orbit assembly of large-scale space structures. The closed-chain constraints significantly reduce the feasible configuration space, making it difficult for existing planners to efficiently generate collision-free motions, especially in cluttered environments. To address this issue, this paper proposes a task-space constrained bidirectional rapidly-exploring random tree algorithm, termed TCBiRRT. Unlike conventional methods that operate in the high-dimensional configuration space, the proposed approach performs random sampling and node expansion directly in the task space defined by the manipulated object pose. A task-space node expansion strategy is developed to generate candidate object motions, which are then mapped to continuous joint paths using a path inverse kinematics algorithm. The method is further integrated with a bidirectional RRT framework and a regrasp mechanism to efficiently connect two random trees. Extensive simulations are conducted in representative on-orbit assembly scenarios with varying levels of environmental complexity. The results demonstrate that TCBiRRT achieves significantly higher success rates and orders-of-magnitude improvements in planning time compared to state-of-the-art planners. The proposed method provides an efficient and robust solution for motion planning of tightly coupled dual-arm space manipulators.
>
---
#### [new 032] Towards Real-World Identification of Fatigued Muscle Groups via Musculoskeletal Simulation
- **分类: physics.med-ph; cs.RO**

- **简介: 该论文属于肌肉疲劳识别任务，旨在通过无接触方式诊断上肢肌肉群疲劳。工作包括提出算法，比较真实运动与模拟数据，实现非侵入式诊断。**

- **链接: [https://arxiv.org/pdf/2605.26151](https://arxiv.org/pdf/2605.26151)**

> **作者:** Jenishkumar Chauhan; Samarth Brahmbhatt; Vineet Vashista
>
> **备注:** Video File: this https URL
>
> **摘要:** Contactless diagnosis of musculoskeletal disorders can potentially improve population health as well as robot behaviours in collaborative settings. However, current diagnosis methods require an in-person physical examination in which a trained physician senses, through contact, the force applied by various muscles. Simulation tools exist, but their use for diagnosis with real data is under-explored. In this paper, we propose an algorithm for identifying which upper-limb muscle group is fatigued. Our algorithm compares the realworld free-space motion of the subject with that of a simulated musculoskeletal model, and is therefore contactless: preventing the need for invasive sensing or in-person assessment. Our algorithm simulates various fatigue conditions using a physics-based musculoskeletal model and extracts diagnostic motion features from both real and simulated data, which are compared for diagnosis. Experimental results on real data demonstrate that the proposed method can reliably distinguish between multiple muscle-groups of fatigue. Additionally, through comprehensive performance comparisons, we show how recent advanced musculoskeletal simulators can be properly configured to address the sim-to-real gap in the context of the fatigue diagnosis task. Our approach can potentially spur further research in remote and automated diagnosis, significantly lowering the barrier to large-scale and early detection.
>
---
#### [new 033] Breaking the Epistemic Trap: Active Perception Under Compound Uncertainty
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于强化学习任务，解决安全关键领域中因复合不确定性导致的决策失败问题。提出主动感知框架，通过量化不确定性并动态调整安全约束，提升系统在未知环境中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26627](https://arxiv.org/pdf/2605.26627)**

> **作者:** Chayan Banerjee; Ethan Goan
>
> **摘要:** Deploying reinforcement learning in safety critical domains, from autonomous vehicles to medical decision support, is constrained by failures arising when systems encounter unfamiliar conditions. We argue that the fundamental bottleneck is not individual challenges like changing dynamics or incomplete observations, but their synergistic interaction, which we term the Epistemic Trap: agents cannot estimate their state without knowing system dynamics, nor learn dynamics without accurate state information. Proof-of-concept experiments in simulated locomotion reveal that combining these uncertainties causes failures far worse than either challenge alone, a 77% performance degradation against the 46% by adding the individual effects, demonstrating compounding failure modes that conventional methods overlook. Such approaches adopt a passive epistemic stance that cannot resolve this coupled uncertainty. We propose reframing safety as an information problem, introducing an Adaptive Safety Architecture built around three contributions: the Compound Uncertainty Coefficient ($\kappa$), a mutual information based metric that quantifies state dynamics coupling and is computable online without full joint belief inference; information seeking policies governed by a MaxInfoRL objective that actively probe system dynamics; and regime-adaptive safety constraints that tighten as epistemic coupling rises. This paradigm shift, from passive robustness to active perception, offers a principled path toward decision making systems that operate under uncertainty, recognize their own ignorance, and act strategically to resolve it.
>
---
#### [new 034] Trust Region Q Adjoint Matching
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决预训练流策略在离线强化学习中的稳定性问题。提出TRQAM算法，通过控制路径空间KL散度实现稳定微调。**

- **链接: [https://arxiv.org/pdf/2605.27079](https://arxiv.org/pdf/2605.27079)**

> **作者:** Yonghoon Dong; Kyungmin Lee; Changyeon Kim; Jaehyuk Kim; Jinwoo Shin
>
> **摘要:** Off-policy reinforcement learning of pretrained flow policies remains challenging due to the instability of optimization arising from the multi-step sampling process. Recently, Q-learning with Adjoint Matching (QAM) addressed this issue by reformulating into a memoryless stochastic optimal control (SOC) problem with a learned critic. However, QAM inherits a fundamental fragility of critic-guided improvement: small critic errors are amplified when critics are ill-conditioned, often leading to model collapse. This paper introduces Trust Region Q-Adjoint Matching (TRQAM), a stable off-policy fine-tuning algorithm that adaptively controls the path-space KL with pretrained flow policies through projected dual descent. Specifically, we optimize the trust-region parameter $\lambda$ in SOC dynamics, and theoretically show that the path-space KL can be represented by a closed-form function of $\lambda$. As a result, our method can precisely control the exact deviation from pretrained flow policies, achieving stable off-policy RL. Through experiments on 50 OGBench tasks, TRQAM consistently outperforms prior arts in both offline RL and offline-to-online RL. In particular, TRQAM achieves an overall success rate of 68% in offline RL, substantially improves the strongest baseline at 46%.
>
---
#### [new 035] Decoupled Delay Compensation: Enhancing Pre-trained MARL Policies via Learned Dynamics Filtering
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，解决通信延迟和数据丢失导致的性能下降问题。通过引入模块化状态估计层，提升预训练策略的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.26286](https://arxiv.org/pdf/2605.26286)**

> **作者:** Maxim Mednikov; Oren Gal
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Real-world multi-agent reinforcement learning (MARL) systems must often operate under stale observations, stochastic communication delays, and intermittent packet loss. Policies trained under idealized synchronous conditions frequently exhibit significant performance degradation in these regimes because they act on outdated feedback. We propose a modular execution-stage state-estimation layer that replaces delayed communicated observations with current belief-state estimates. The framework integrates a learned Gated transition model with a recursive Kalman filtering layer to estimate instantaneous states from asynchronous measurements. A primary advantage of this approach is its modularity, The estimator serves as a plug-in for pre-trained policies, requiring no modifications to the original MARL training algorithm, architecture, or reward structure. Evaluation across diverse multi-agent and continuous-control benchmarks demonstrates that the proposed layer consistently enhances robustness to communication latency and message loss. The most significant performance gains are observed in coordination-intensive and dynamically unstable tasks where temporal consistency is critical for control.
>
---
#### [new 036] LocateAnything: Fast and High-Quality Vision-Language Grounding with Parallel Box Decoding
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出LocateAnything，解决视觉-语言定位与检测任务中的效率与精度问题。通过并行框解码提升推理速度和定位准确性。**

- **链接: [https://arxiv.org/pdf/2605.27365](https://arxiv.org/pdf/2605.27365)**

> **作者:** Shihao Wang; Shilong Liu; Yuanguo Kuang; Xinyu Wei; Yangzhou Liu; Zhiqi Li; Yunze Man; Guo Chen; Andrew Tao; Guilin Liu; Jan Kautz; Lei Zhang; Zhiding Yu
>
> **摘要:** Vision-language models (VLMs) commonly formulate visual grounding and detection as a coordinate-token generation problem, serializing each 2D box into multiple 1D tokens that are learned and decoded largely independently. This token-by-token decoding mismatches the coupled structure of box geometry and creates a practical inference bottleneck due to strictly sequential generation. We introduce LocateAnything, a unified generative grounding and detection framework based on Parallel Box Decoding (PBD). By decoding geometric elements such as bounding boxes and points as atomic units in a single step, LocateAnything preserves intra-box geometric coherence and unlocks substantial parallelism. We show that PBD improves both decoding throughput and localization accuracy. We further develop a scalable data engine and curate LocateAnything-Data, a large-scale dataset with more than 138 million training samples, substantially increasing data diversity for high-precision localization. Extensive evaluations show that LocateAnything advances the speed-accuracy frontier, achieving significantly higher decoding throughput while improving high-IoU localization quality across diverse benchmarks. The results highlight the complementary benefits of Parallel Box Decoding and large-scale training data in enabling efficient and precise unified visual grounding and detection.
>
---
#### [new 037] OSMa-Bench++: Toward Open-Ended Benchmarking of Semantic Mapping for Manipulation with Prompt-Generated Synthetic Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决语义映射评估不足的问题。通过生成合成场景扩展OSMa-Bench，提升基准测试的灵活性与实用性。**

- **链接: [https://arxiv.org/pdf/2605.26831](https://arxiv.org/pdf/2605.26831)**

> **作者:** Regina Kurkova; Maxim Popov; Sergey Kolyubin
>
> **备注:** Code: this https URL
>
> **摘要:** Semantic mapping methods are increasingly used as intermediate scene representations for downstream robotic reasoning and manipulation, yet their evaluation is still largely tied to fixed benchmark datasets with limited coverage of manipulation-relevant corner cases. In this work, we extend OSMa-Bench toward controllable benchmarking with prompt-generated synthetic indoor scenes. Our pipeline automatically generates scene descriptions, synthesizes corresponding environments with SceneSmith, and adapts the resulting assets into an OSMa-Bench-compatible simulation format. This adaptation requires a nontrivial intermediate layer, including semantic normalization, material and texture repair, shader fallback policies, floor handling, navigation setup, and controlled lighting configuration. A key advantage of the proposed setup is that the original scene-generation prompt is known in advance and can therefore serve as an auxiliary semantic specification of the intended scene. We use this property to extend the VQA component of OSMa-Bench with a prompt-grounded question category. The resulting framework supports targeted stress-testing of semantic scene representations under conditions such as clutter, small objects, partial occlusions, and lighting variation, and makes benchmarking more extensible and better aligned with downstream manipulation requirements. Our code is available at this https URL.
>
---
#### [new 038] FoundObj: Self-supervised Foundation Models as Rewards for Label-free 3D Object Segmentation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于3D物体分割任务，解决无标注点云中复杂场景的物体分割问题。提出FoundObj框架，利用自监督模型提供语义和几何奖励，实现多类物体的鲁棒识别。**

- **链接: [https://arxiv.org/pdf/2605.27178](https://arxiv.org/pdf/2605.27178)**

> **作者:** Zihui Zhang; Zhixuan Sun; Yafei Yang; Jinxi Li; Jiahao Chen; Bo Yang
>
> **备注:** ICML 2026. Zihui and Zhixuan are co-first authors. Code and data are available at: this https URL
>
> **摘要:** We address the challenging task of 3D object segmentation in complex scene point clouds without relying on any scene-level human annotations during training. Existing methods are typically constrained to identifying simple objects, primarily due to insufficient object priors in the learning process. In this paper, we present FoundObj, a novel framework featuring a superpoint-based object discovery agent that incrementally merges suitable neighboring superpoints, guided by our innovative semantic and geometric reward modules. These modules synergistically leverage semantic and geometric priors from self-supervised 2D/3D foundation models, providing complementary feedback to the object discovery agent and enabling robust identification of multi-class objects through reinforcement learning. Extensive experiments on diverse benchmarks demonstrate that our approach consistently outperforms existing baselines. Notably, our method exhibits strong generalization in zero-shot and long-tail scenarios, underscoring its potential for scalable, label-free 3D object segmentation.
>
---
#### [new 039] The Sensation Modulating Network:Haltability as the architectural ground for object-directed phenomenology
- **分类: q-bio.NC; cs.AI; cs.RO**

- **简介: 该论文提出Sensation Modulating Network（SMN）模型，解决认知科学中符号意义与身体体验的矛盾，通过 opponency 和 haltability 等机制整合认知与具身理论。**

- **链接: [https://arxiv.org/pdf/2605.26856](https://arxiv.org/pdf/2605.26856)**

> **作者:** G. Nagarjuna; Durgaprasad Karnam
>
> **备注:** 64 pages, main body 38 pages + References 6, Appendices 20 pages, Tables 3, and Figures 21
>
> **摘要:** Cognitive science remains split between cognitivism - which accounts for recursion and language but cannot ground formal symbols in meaning - and 4E approaches - which ground cognition in the body but rarely specify the body's architecture in enough detail to support generativity. We argue the impasse stems from an incomplete account of the embodied agent's architecture, and propose one: the Sensation Modulating Network (SMN), the cognitive agent conceived as the whole body, organized at every anatomical scale by opponent dynamics, built from Sensation Modulators that sense and act through one substrate, paired into Coordinated Action Zones routed by a body-wide broadcast network. Three commitments give the SMN its purchase. Haltability - the recruitment of antagonistic affordance into co-activated equilibrium - provides the architectural locus that object-directed phenomenology, in Husserl's sense, requires: opponency enables co-activation, co-activation enables halt, halt enables attention, attention enables intentional directedness, with no module added on top. The dual-signal property of self-modulatable action patterns (SMAPs) makes the self/world distinction a structural feature of the wiring rather than a category the agent applies. And a four-level action-pattern hierarchy - Basal, Haltable, Negotiable, Transactional - gives a single trajectory from autonomic regularity to public conventionalization, locating the conditions for grammar-grounded generativity as architectural transitions. The SMN reconciles the cognitivism-4E debate: recursion lives in the modifiable dynamics of Negotiable Action Patterns, embodiment in the opponent substrate that supports them. A tentative formalism and eight predicted registers (seven testable, one hypothetical), with reference simulations, are given in an appendix.
>
---
#### [new 040] YOLO26-RipeLoc Lite: A lightweight architecture for tomato ripeness detection and picking point localization in greenhouse robotic harvesting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于温室番茄成熟度检测与采摘点定位任务，提出YOLO26-RipeLoc Lite模型，解决精准检测与定位问题。**

- **链接: [https://arxiv.org/pdf/2605.27129](https://arxiv.org/pdf/2605.27129)**

> **作者:** Rajmeet Singh; Manveen Kaur; Shahpour Alirezaee; Irfan Hussain
>
> **摘要:** In greenhouse tomato production, automated harvesting requires accurate detection of ripe tomatoes, ripeness classification, and precise picking-point localization for robotic end-effectors. This paper proposes YOLO26-RipeLoc Lite, a lightweight deep learning architecture based on YOLO26 for simultaneous detection, ripeness classification, and center-point localization of greenhouse tomatoes. The model introduces three modifications: (1) a Lightweight Feature Pyramid Network (LFPN) with depthwise separable convolutions for efficient multi-scale fusion, (2) a Ripeness-Aware Attention Module (RAAM) with dual pooling and a learnable ripeness bias vector for enhanced color-texture discrimination, and (3) a Compact Detection Head (CDH) with shared convolutions and an integrated center-point regression branch for direct grasp planning. The model is evaluated on a custom dataset of 1,500 images with 6,227 instances (3,566 ripe, 2,661 unripe) from the SILAL greenhouse, Abu Dhabi, UAE. YOLO26-RipeLoc Lite achieves mAP@0.5 of 92.9% (95.2% ripe, 90.6% unripe) with the highest precision (95.2%) among all evaluated architectures using only 2.38M parameters. Post-training BatchNorm pruning at 30% reduces parameters to ~1.8M with negligible accuracy loss. Ablation studies confirm that greenhouse-aware HSV augmentation provides the largest improvement (+2.02 pp mAP@50), backbone freezing achieves peak precision (93.8%), and 3-phase progressive unfreezing yields the best localization quality (mAP@50:95 of 64.6%). Comparisons with YOLOv8n/s, YOLO11n/s, YOLO12n/s, and YOLO26s confirm superior accuracy-efficiency: 2.9 pp higher precision than YOLO12n with 7.0% fewer parameters and integrated center-point localization for robotic end-effector guidance.
>
---
## 更新

#### [replaced 001] AdaMorph: Unified Motion Retargeting via Embodiment-Aware Adaptive Transformers
- **分类: cs.RO**

- **简介: 该论文属于机器人运动重定向任务，解决不同机器人形态间运动迁移问题。提出AdaMorph框架，通过统一模型适配多种机器人，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.07284](https://arxiv.org/pdf/2601.07284)**

> **作者:** Haoyu Zhang; Shibo Jin; Lusong Li; Jun Li; Liang Lin; Xiaodong He; Zecui Zeng
>
> **摘要:** Retargeting human motion to heterogeneous robots is a fundamental challenge in robotics, primarily due to the severe kinematic and dynamic discrepancies between varying embodiments. Existing solutions typically resort to training embodiment-specific models, which scales poorly and fails to exploit shared motion semantics. To address this, we present AdaMorph, a unified neural retargeting framework that enables a single model to adapt human motion to diverse robot morphologies. Our approach treats retargeting as a conditional generation task. We map human motion into a morphology-agnostic latent intent space and utilize a dual-purpose prompting mechanism to condition the generation. Instead of simple input concatenation, we leverage Adaptive Layer Normalization (AdaLN) to dynamically modulate the decoder's feature space based on embodiment constraints. Furthermore, we enforce physical plausibility through a curriculum-based training objective that ensures orientation and trajectory consistency via integration. Experimental results on 12 distinct humanoid robots demonstrate that AdaMorph effectively unifies control across heterogeneous topologies, exhibiting strong zero-shot generalization to unseen complex motions while preserving the dynamic essence of the source behaviors.
>
---
#### [replaced 002] Synergetic Empowerment: Wireless Communications Meets Embodied Intelligence
- **分类: cs.NI; cs.RO; eess.SP; eess.SY**

- **简介: 该论文属于人工智能与通信融合领域，探讨无线通信与具身智能的协同增强，解决二者如何共同提升代理系统的问题，提出PCE循环框架实现系统优化。**

- **链接: [https://arxiv.org/pdf/2509.10481](https://arxiv.org/pdf/2509.10481)**

> **作者:** Hongtao Liang; Yihe Diao; YuHang Wu; Fuhui Zhou; Qihui Wu
>
> **备注:** Accepted by IEEE Communications Magazine
>
> **摘要:** Wireless communication is evolving into an agent era, where large-scale agents with inherent embodied intelligence are not just users but active participants. The perfect combination of wireless communication and embodied intelligence can achieve a synergetic empowerment and greatly facilitate the development of agent communication. An overview of this synergetic empowerment is presented, framing it as a co-evolutionary process that transforms wireless communication from a simple utility into the digital nervous system of a collective intelligence, while simultaneously elevating isolated agents into a unified superorganism with emergent capabilities far exceeding individual contributions. Moreover, we elaborate how embodied intelligence and wireless communication mutually benefit each other through the lens of the perception-cognition-execution (PCE) loop, revealing a fundamental duality where each PCE stage both challenges network capacity and creates unprecedented opportunities for system-wide optimization. Furthermore, critical open issues and future research directions are identified.
>
---
#### [replaced 003] Polymander II: an amphibious salamander-inspired robot with contact and flow sensors
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决 amphibious 机器人在陆地和水域中感知交互力的问题。通过使用霍尔传感器，实现高频率的接触与流体力感知，提升机器人适应复杂地形的能力。**

- **链接: [https://arxiv.org/pdf/2605.24465](https://arxiv.org/pdf/2605.24465)**

> **作者:** Qiyuan Fu; Sudong Lee; Andrea Grillo; Jonathan Arreguit; Louis Gevers; Josie Hughes; Auke J. Ijspeert
>
> **备注:** This work has been accepted for publication in the 2026 International Conference on Robotics and Automation (ICRA), Vienna, Austria
>
> **摘要:** Robots benefit from sensory information to coordinate body movement, gain robustness against perturbations, and transition between different modes to adapt to various terrains. However, few amphibious robots can sense interactions with both terrestrial and aquatic environments. In this paper, we present a solution that uses Hall-effect sensors to sense foot contact forces and lateral hydrodynamic forces on a salamander-inspired amphibious robot. With two bus lines, the robot can simultaneously acquire this exteroceptive information at more than 500 Hz and proprioceptive information, such as joint positions and loads, at 100 Hz. The Hall-effect sensors used are compact, making them suitable for embedding in multiple positions within a robot, and exhibit high sensitivity to small forces. Moreover, because the sensor can be positioned separately from the measured object, waterproofing can be implemented with relative ease. Our tests demonstrate the robot's capabilities in traversing amphibious environments and its potential in using feedback control for more complex locomotion tasks.
>
---
#### [replaced 004] Zero-Shot MARL Benchmark in the Cyber-Physical Mobility Lab
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多智能体强化学习（MARL）的模拟到现实迁移任务，旨在评估CAV运动规划策略的跨域性能。工作包括构建一个集成仿真、数字孪生和物理测试平台的基准，分析性能下降原因。**

- **链接: [https://arxiv.org/pdf/2601.16578](https://arxiv.org/pdf/2601.16578)**

> **作者:** Julius Beerwerth; Jianye Xu; Simon Schäfer; Fynn Belderink; Bassam Alrifaee
>
> **摘要:** We present a reproducible benchmark for evaluating sim-to-real transfer of Multi-Agent Reinforcement Learning (MARL) policies for Connected and Automated Vehicles (CAVs). The platform, based on the Cyber-Physical Mobility Lab (CPM Lab) [1], integrates simulation, a high-fidelity digital twin, and a physical testbed, enabling structured zero-shot evaluation of MARL motion-planning policies. We demonstrate its use by deploying a SigmaRL-trained policy [2] across all three domains, revealing two complementary sources of performance degradation: architectural differences between simulation and hardware control stacks, and the sim-to-real gap induced by increasing environmental realism. The open-source setup enables systematic analysis of sim-to-real challenges in MARL under realistic, reproducible conditions.
>
---
#### [replaced 005] Multi-Agent Reinforcement Learning for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty
- **分类: cs.LG; cs.AI; cs.HC; cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，解决行人行为不确定性带来的安全测试不足问题。通过多智能体强化学习联合训练自动驾驶车辆与行人，提升交互场景的真实性与安全性。**

- **链接: [https://arxiv.org/pdf/2605.20255](https://arxiv.org/pdf/2605.20255)**

> **作者:** Prakash Aryan; Kaushik Raghupathruni; Timo Kehrer; Sebastiano Panichella
>
> **备注:** Accepted to ICRA 2026 Workshop "8th Workshop on Long-term Human Motion Prediction"
>
> **摘要:** Simulation-based testing of self-driving cars (SDCs) typically relies on scripted pedestrian models that do not capture the heterogeneity and uncertainty of real crossing behavior, limiting the realism of safety assessments, especially for jaywalking, which is governed by latent personality traits the vehicle cannot observe. We hypothesize that jointly training pedestrians and the SDC with multi-agent reinforcement learning (MARL) yields more realistic interaction scenarios than training against fixed pedestrian policies, and that the behavior gap between predictable and unpredictable crossings can be measured directly from trajectories. We co-train an SDC and 12 pedestrians using Multi-Agent Proximal Policy Optimization (MAPPO): pedestrian locomotion follows scripted Dijkstra pathfinding while an RL policy controls high-level go/wait decisions, and jaywalking probability depends on a per-pedestrian trait sampled at episode start and hidden from the SDC. In 500-episode evaluations, the co-trained SDC reached 78% of goals with a 14% collision rate, versus 35%/33% for the best rule-based baseline. A speed differential metric shows the SDC traveled 2.65 m/s faster near jaywalkers than near crosswalk users at close range (0-3 m), indicating jaywalking encounters were not anticipated. Jaywalking was 13% of crossing events but 62% of collisions, and co-training reduced collisions by 30% relative to single-agent RL as pedestrians learned to wait when the SDC approached at speed.
>
---
#### [replaced 006] Adapting Dijkstra for Buffers and Unlimited Transfers
- **分类: cs.DS; cs.AI; cs.RO**

- **简介: 该论文属于公共交通路径规划任务，解决无限换乘下的最优路径问题。通过改进Dijkstra算法，提出TAD方法，在考虑缓冲时间的情况下提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.11729](https://arxiv.org/pdf/2603.11729)**

> **作者:** Denys Katkalo; Andrii Rohovyi; Toby Walsh
>
> **备注:** v4: clarified RAPTOR description in the Background section
>
> **摘要:** In recent years, RAPTOR based algorithms have been considered the state-of-the-art for path-finding with unlimited transfers without preprocessing. However, this status largely stems from the evolution of routing research, where Dijkstra-based solutions were superseded by timetable-based algorithms without a systematic comparison. In this work, we revisit classical Dijkstra-based approaches for public transit routing with unlimited transfers and demonstrate that Time-Dependent Dijkstra (TD-Dijkstra) outperforms MR. However, efficient TD-Dijkstra implementations rely on filtering dominated connections during preprocessing, which assumes passengers can always switch to a faster connection. We show that this filtering is unsound when stops have buffer times, as it cannot distinguish between seated passengers who may continue without waiting and transferring passengers who must respect the buffer. To address this limitation, we introduce Transfer Aware Dijkstra (TAD), a modification that scans entire trip sequences rather than individual edges, correctly handling buffer times while maintaining performance advantages over MR. Our experiments on London and Switzerland networks show that we can achieve a greater than two time speed-up over MR while producing optimal results on both networks with and without buffer times.
>
---
#### [replaced 007] Governed Capability Evolution: Lifecycle-Time Compatibility Checking and Rollback for AI-Component-Based Systems, with Embodied Agents as Case Study
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于AI系统生命周期管理任务，解决AI组件升级中的兼容性与安全问题。提出分阶段升级框架，包含四类检查和七步流程，确保升级安全可靠。**

- **链接: [https://arxiv.org/pdf/2604.08059](https://arxiv.org/pdf/2604.08059)**

> **作者:** Xue Qin; Simin Luan; John See; Zeyd Boukhers; Cong Yang; Zhijun Li
>
> **备注:** 42 pages, 7 figures, 12 tables
>
> **摘要:** Software systems built from versioned AI components increasingly need lifecycle-time governance: when a capability module evolves into a new version, the hosting system must decide whether the new version may be activated safely, under what deployment conditions it should run, how it must be monitored, and when it should be rolled back. Existing software-deployment patterns (canary release, blue-green, feature flags, and MLOps pipelines) address parts of this loop but were designed for stateless web services rather than for stateful, policy-constrained runtimes that drive AI components in the field. We formulate governed capability evolution as a first-class software-lifecycle problem for AI-component-based systems and propose a staged upgrade framework in which every new capability version is treated as a governed deployment candidate rather than an immediately executable replacement. The framework introduces four upgrade compatibility checks (interface, policy, behavioral, recovery) and organizes them into a seven-stage pipeline (candidate validation, sandbox evaluation, shadow deployment, gated activation, online monitoring, rollback, audit). We implement a reference prototype on a PyBullet manipulation testbed with ROS 2 middleware and evaluate it over 6 rounds of capability upgrade with 15 random seeds. Naive upgrade achieves 72.9% task success but drives unsafe activation to 60% by the final round; governed upgrade retains comparable success (67.4%) while maintaining zero unsafe activations across all rounds (Wilcoxon p=0.003). Shadow deployment reveals 40% of upgrade regressions invisible to sandbox evaluation alone, and rollback succeeds in 79.8% of post-activation drift scenarios.
>
---
#### [replaced 008] LAD-VF: LLM-Automatic Differentiation Enables Fine-Tuning-Free Robot Planning from Formal Methods Feedback
- **分类: cs.RO; cs.FL**

- **简介: 该论文属于机器人规划任务，解决LLM在物理世界中因违反安全约束而失效的问题。提出LAD-VF框架，通过形式化验证反馈优化提示，无需微调模型。**

- **链接: [https://arxiv.org/pdf/2509.18384](https://arxiv.org/pdf/2509.18384)**

> **作者:** Yunhao Yang; Junyuan Hong; Gabriel Jacob Perin; Zhiwen Fan; Li Yin; Zhangyang Wang; Ufuk Topcu
>
> **备注:** Presented at ICRA 2026
>
> **摘要:** Large language models (LLMs) can translate natural language instructions into executable action plans for robotics, autonomous driving, and other domains. Yet, deploying LLM-driven planning in the physical world demands strict adherence to safety and regulatory constraints, which current models often violate due to hallucination or weak alignment. Traditional data-driven alignment methods, such as Direct Preference Optimization (DPO), require costly human labeling, while recent formal-feedback approaches still depend on resource-intensive fine-tuning. In this paper, we propose LAD-VF, a fine-tuning-free framework that leverages formal verification feedback for automated prompt engineering. By introducing a formal-verification-informed text loss integrated with LLM-AutoDiff, LAD-VF iteratively refines prompts rather than model parameters. This yields three key benefits: (i) scalable adaptation without fine-tuning; (ii) compatibility with modular LLM architectures; and (iii) interpretable refinement via auditable prompts. Experiments in robot navigation and manipulation tasks demonstrate that LAD-VF substantially enhances specification compliance, improving success rates from 60% to over 90%. Our method thus presents a scalable and interpretable pathway toward trustworthy, formally-verified LLM-driven control systems.
>
---
#### [replaced 009] Equivariant Filter for Relative Attitude and Target's Angular Velocity Estimation
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于航天器相对姿态与角速度估计任务，解决如何在噪声环境下准确估计目标物体的相对姿态和角速度问题，提出一种等变滤波器（EqF）并进行仿真与实验验证。**

- **链接: [https://arxiv.org/pdf/2506.06016](https://arxiv.org/pdf/2506.06016)**

> **作者:** Gil Serrano; Bruno J. Guerreiro; Pedro Lourenço; Rita Cunha
>
> **备注:** Published in the IEEE Transactions on Aerospace and Electronic Systems, 2026. Open Access article under CC BY 4.0
>
> **摘要:** Accurate estimation of the relative attitude and angular velocity between two rigid bodies is fundamental in aerospace applications such as spacecraft rendezvous and docking. In these scenarios, a chaser vehicle must determine the orientation and angular velocity of a target object using onboard sensors. This work addresses the challenge of designing an Equivariant Filter (EqF) that can reliably estimate both the relative attitude and the target angular velocity using noisy observations of two known, non-collinear vectors fixed in the target frame. To derive the EqF, a symmetry for the system is proposed and an equivariant lift onto the symmetry group is calculated. Observability and convergence properties are analyzed. Simulations demonstrate the filter's performance, with Monte Carlo runs yielding statistically significant results. The impact of low-rate measurements is also examined and a strategy to mitigate this effect is proposed. Experimental results, using fiducial markers and both conventional and event cameras for measurement acquisition, further validate the approach, confirming its effectiveness in a realistic setting.
>
---
#### [replaced 010] Physically Native World Models: A Hamiltonian Perspective on Generative World Modeling
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于世界建模任务，旨在解决物理可靠性与长期稳定预测问题。提出哈密顿世界模型，通过物理动力学框架提升预测能力。**

- **链接: [https://arxiv.org/pdf/2605.00412](https://arxiv.org/pdf/2605.00412)**

> **作者:** Sen Cui; Jingheng Ma
>
> **摘要:** World models have recently re-emerged as a central paradigm for embodied intelligence, robotics, autonomous driving, and model-based reinforcement learning. However, current world model research is often dominated by three partially separated routes: 2D video-generative models that emphasize visual future synthesis, 3D scene-centric models that emphasize spatial reconstruction, and JEPA-like latent models that emphasize abstract predictive representations. While each route has made important progress, they still struggle to provide physically reliable, action-controllable, and long-horizon stable predictions for embodied decision making. In this paper, we argue that the bottleneck of world models is no longer only whether they can generate realistic futures, but whether those futures are physically meaningful and useful for action. We propose \emph{Hamiltonian World Models} as a physically grounded perspective on world modeling. The key idea is to encode observations into a structured latent phase space, evolve the latent state through Hamiltonian-inspired dynamics with control, dissipation, and residual terms, decode the predicted trajectory into future observations, and use the resulting rollouts for planning. We discuss how Hamiltonian structure may improve interpretability, data efficiency, and long-horizon stability, while also noting practical challenges in real-world robotic scenes involving friction, contact, non-conservative forces, and deformable objects.
>
---
#### [replaced 011] SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SOLE-R1，用于机器人强化学习的视频-语言推理奖励模型，解决无监督任务学习问题，通过自然语言目标生成密集奖励信号。**

- **链接: [https://arxiv.org/pdf/2603.28730](https://arxiv.org/pdf/2603.28730)**

> **作者:** Philip Schroeder; Thomas Weng; Karl Schmeckpeper; Eric Rosen; Stephen Hart; Ondrej Biza
>
> **摘要:** Vision-language models (VLMs) have shown impressive capabilities across diverse tasks, motivating efforts to leverage these models to supervise robot learning. However, when used as evaluators in reinforcement learning (RL), today's strongest models often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors rather than solve the task. We introduce SOLE-R1 (Self-Observing LEarner), a video-language reasoning model explicitly designed to serve as the sole reward signal for online RL. Given only raw video observations and a natural-language goal, SOLE-R1 performs per-timestep spatiotemporal chain-of-thought (CoT) reasoning and produces dense estimates of task progress that can be used directly as rewards. To train SOLE-R1, we develop a large-scale video trajectory and reasoning synthesis pipeline that generates temporally grounded CoT traces aligned with continuous progress supervision. This data is combined with foundational spatial and multi-frame temporal reasoning, and used to train the model with a hybrid framework that couples supervised fine-tuning with RL from verifiable rewards. Across four different simulation environments and a real-robot setting, SOLE-R1 enables zero-shot online RL from random initialization: robots learn previously unseen manipulation tasks without ground-truth rewards, success indicators, demonstrations, or task-specific tuning. SOLE-R1 succeeds on 24 unseen tasks and substantially outperforms strong vision-language rewarders, including Robometer, RoboReward, ReWiND, GPT-5, and Gemini-3-Pro, while exhibiting markedly greater robustness to reward hacking. We release all models, data, code, and demos at the anonymous page: this https URL
>
---
#### [replaced 012] Early Pruning for Public Transport Routing
- **分类: cs.DS; cs.AI; cs.RO**

- **简介: 该论文属于公共交通路径规划任务，解决转移阶段效率低的问题。通过早期剪枝技术，在不影响最优性的前提下提升算法效率。**

- **链接: [https://arxiv.org/pdf/2603.12592](https://arxiv.org/pdf/2603.12592)**

> **作者:** Andrii Rohovyi; Abdallah Abuaisha; Toby Walsh
>
> **摘要:** Routing algorithms for public transport, particularly the widely used RAPTOR and its variants, often face performance bottlenecks during the transfer relaxation phase, especially on dense transfer graphs, when supporting unlimited transfers. This inefficiency arises from iterating over many potential inter-stop connections (walks, bikes, e-scooters, etc.). To maintain acceptable performance, practitioners often limit transfer distances or exclude certain transfer options, which can reduce path optimality and restrict the multimodal options presented to travellers. This paper introduces Early Pruning, a low-overhead technique that accelerates routing algorithms without compromising optimality. By pre-sorting transfer connections by duration and applying a pruning rule within the transfer loop, the method discards longer transfers at a stop once they cannot yield an earlier arrival than the current best solution. Early Pruning can be integrated with minimal changes to existing codebases and requires only a one-time preprocessing step. The technique preserves Pareto-optimality in extended-criteria settings whenever the additional optimization criteria are monotonically non-decreasing in transfer duration. Across multiple state-of-the-art RAPTOR-based solutions, including RAPTOR, ULTRA-RAPTOR, McRAPTOR, BM-RAPTOR, ULTRA-McRAPTOR, and UBM-RAPTOR and tested on the Switzerland and London transit networks, we achieved query time reductions of up to 57\%. This approach provides a generalizable improvement to the efficiency of transit pathfinding algorithms.
>
---
#### [replaced 013] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboMME，一个用于评估和提升机器人通用策略记忆能力的基准。解决长期任务中记忆机制不足的问题，通过设计16个任务和14种变体模型进行系统研究。**

- **链接: [https://arxiv.org/pdf/2603.04639](https://arxiv.org/pdf/2603.04639)**

> **作者:** Yinpei Dai; Hongze Fu; Jayjun Lee; Yuejiang Liu; Haoran Zhang; Jianing Yang; Chelsea Finn; Nima Fazeli; Joyce Chai
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Memory is critical for long-horizon and history-dependent robotic manipulation. Such tasks often involve counting repeated actions or manipulating objects that become temporarily occluded. Recent vision-language-action (VLA) models have begun to incorporate memory mechanisms; however, their evaluations remain confined to narrow, non-standardized settings. This limits systematic understanding, comparison, and progress measurement. To address these challenges, we introduce RoboMME: a large-scale standardized benchmark for evaluating and advancing VLA models in long-horizon, history-dependent scenarios. Our benchmark comprises 16 manipulation tasks constructed under a carefully designed taxonomy that evaluates temporal, spatial, object, and procedural memory. We further develop a suite of 14 memory-augmented VLA variants built on the {\pi}0.5 backbone to systematically explore different memory representations across multiple integration strategies. Experimental results show that the effectiveness of memory representations is highly task-dependent, with each design offering distinct advantages and limitations across different tasks. Videos and code can be found at our website this https URL.
>
---
#### [replaced 014] Drive-P2D: A Progressive Perception-to-Decision Benchmark for VLMs in Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出Drive-P2D基准，解决VLMs在自动驾驶中感知与决策联合评估的问题，通过多层级测试和错误分析提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14702](https://arxiv.org/pdf/2601.14702)**

> **作者:** Zecong Tang; Zixu Wang; Yifei Wang; Weitong Lian; Tianjian Gao; Haoran Li; Tengju Ru; Lingyi Meng; Zhejun Cui; Yichen Zhu; Qi Kang; Kaixuan Wang; Yu Zhang
>
> **摘要:** Autonomous driving requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks often evaluate perception and decision-making separately, limit failure analysis with choice-only formats, or introduce evaluation bias through LLM-scored long-form outputs. To address these issues, we present Drive-P2D, a progressive perception-to-decision benchmark with 6,650 questions across Object, Scene, and Decision levels. Drive-P2D adopts a separated reasoning-and-answer protocol: final answers are scored objectively, while reasoning is analyzed to identify error modes exposed along the progressive perception-to-decision chain. We evaluate mainstream VLMs across all and high-risk scenarios, and further characterize the perception-to-decision capability boundary through correlation analysis and similar-scene robustness testing. Reasoning further exposes failure modes such as logical reasoning errors and semantic feature omissions, and we train a lightweight analyzer model to automate large-scale error-mode annotation of reasoning. Together, these designs provide practical insights for building safer and more reliable VLMs for real-world autonomous driving.
>
---
#### [replaced 015] ParkourFormer: Integrating Predictive Supervision and Sequence Modeling into Parkour Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决复杂地形下人体机器人敏捷移动问题。提出ParkourFormer框架，结合预测监督与序列建模，提升运动规划的预见性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.25782](https://arxiv.org/pdf/2605.25782)**

> **作者:** Yanheng Mai; Wenhao Xu; Zirui Huang; Yifei Fu; Shengwei Dong; Xinjue Wang; Kailun Huang; Yanzhe Xie; Renjing Xu
>
> **摘要:** Humanoid parkour requires locomotion policies to coordinate whole-body dynamics across rapidly changing terrains such as stairs, gaps, slopes, and obstacles. Existing reinforcement learning policies are largely reactive, mapping observations directly to actions without explicitly modeling future body states. Such modeling becomes critical in agile locomotion tasks where successful motion execution depends strongly on anticipating upcoming contact transitions and body dynamics. We present ParkourFormer, a Transformer-based sequence modeling framework that reformulates humanoid locomotion as a future-conditioned decision-making problem. The current robot state queries historical sensorimotor trajectories through cross-attention, while a lightweight prediction head forecasts short-horizon future proprioceptive states. The predicted future states, trained with supervised signals, are fused with temporal features to generate actions, enabling the policy to jointly reason over motion history and anticipated future dynamics. We evaluate ParkourFormer on a diverse multi-terrain humanoid parkour benchmark including stairs, gaps, slopes, rough terrain, and obstacle traversal. Experiments in simulation and on a real humanoid robot show that ParkourFormer achieves a 93.85% average traversal success rate on highly challenging terrains, with improvements of up to 42.73% over strong MLP, MoE-based MLP, and vanilla Transformer baselines, while maintaining a single unified policy across all terrain types. These results demonstrate that explicit future-state modeling significantly improves robustness and generalization for agile whole-body locomotion.
>
---
#### [replaced 016] Continual Model-Based Reinforcement Learning with Hypernetworks
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于持续模型强化学习任务，解决动态模型更新效率低的问题。提出HyperCRL方法，使用任务条件超网络持续学习动态，避免重复训练历史数据。**

- **链接: [https://arxiv.org/pdf/2009.11997](https://arxiv.org/pdf/2009.11997)**

> **作者:** Yizhou Huang; Kevin Xie; Homanga Bharadhwaj; Florian Shkurti
>
> **备注:** Updated link to project website in the abstract. 7 pages (+2 pages in appendix), 8 figures. In proceedings of the 2021 IEEE International Conference on Robotics and Automation
>
> **摘要:** Effective planning in model-based reinforcement learning (MBRL) and model-predictive control (MPC) relies on the accuracy of the learned dynamics model. In many instances of MBRL and MPC, this model is assumed to be stationary and is periodically re-trained from scratch on state transition experience collected from the beginning of environment interactions. This implies that the time required to train the dynamics model - and the pause required between plan executions - grows linearly with the size of the collected experience. We argue that this is too slow for lifelong robot learning and propose HyperCRL, a method that continually learns the encountered dynamics in a sequence of tasks using task-conditional hypernetworks. Our method has three main attributes: first, it includes dynamics learning sessions that do not revisit training data from previous tasks, so it only needs to store the most recent fixed-size portion of the state transition experience; second, it uses fixed-capacity hypernetworks to represent non-stationary and task-aware dynamics; third, it outperforms existing continual learning alternatives that rely on fixed-capacity networks, and does competitively with baselines that remember an ever increasing coreset of past experience. We show that HyperCRL is effective in continual model-based reinforcement learning in robot locomotion and manipulation scenarios, such as tasks involving pushing and door opening. Our project website with videos is at this link this https URL
>
---
#### [replaced 017] Modernising Reinforcement Learning-Based Navigation for Embodied Semantic Scene Graph Generation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于视觉导航任务，旨在提升基于强化学习的语义场景图生成。通过优化策略和动作表示，解决有限行动预算下的信息获取与导航效率问题。**

- **链接: [https://arxiv.org/pdf/2603.25415](https://arxiv.org/pdf/2603.25415)**

> **作者:** Roman Küble; Marco Hüller; Mrunmai Phatak; Rainer Lienhart; Jörg Hähner
>
> **摘要:** Semantic world models enable embodied agents to reason about objects, relations, and spatial context beyond purely geometric representations. In Organic Computing, such models are a key enabler for objective-driven self-adaptation under uncertainty and resource constraints. The core challenge is to acquire observations maximising model quality and downstream usefulness within a limited action budget. Semantic scene graphs (SSGs) provide a structured and compact representation for this purpose. However, constructing them within a finite action horizon requires exploration strategies that trade off information gain against navigation cost and decide when additional actions yield diminishing returns. This work presents a modular navigation component for Embodied Semantic Scene Graph Generation and modernises its decision-making by replacing the policy-optimisation method and revisiting the discrete action formulation. We study compact and finer-grained, larger discrete motion sets and compare a single-head policy over atomic actions with a factorised multi-head policy over action components. We evaluate curriculum learning and optional depth-based collision supervision, and assess SSG completeness, execution safety, and navigation behaviour. Results show that replacing the optimisation algorithm alone improves SSG completeness by 21\% relative to the baseline under identical reward shaping. Depth mainly affects execution safety (collision-free motion), while completeness remains largely unchanged. Combining modern optimisation with a finer-grained, factorised action representation yields the strongest overall completeness--efficiency trade-off.
>
---
#### [replaced 018] ParkingWorld: End-to-End Autonomous Parking Reinforcement Learning from Corrective Experience in 3DGS Simulation
- **分类: cs.RO**

- **简介: 该论文属于自主泊车任务，解决传统方法在训练效率和泛化能力上的不足。提出一种基于修正经验的强化学习框架，提升泊车成功率与安全性。**

- **链接: [https://arxiv.org/pdf/2605.25029](https://arxiv.org/pdf/2605.25029)**

> **作者:** Zhengcheng Yu; Changze Li; Haoran Liu; Tong Qin
>
> **备注:** 9 pages(including 1 page of Appendix), 6 figures. Will be submitted to RA-L 2026
>
> **摘要:** Autonomous parking demands precise low-speed maneuvering within narrow, cluttered, and highly constrained environments, where vehicles must navigate tight spaces while avoiding static obstacles and complex geometric boundaries. Unlike imitation learning, which typically requires massive volumes of high-quality expert demonstrations to converge to a stable policy and often suffers from limited generalization to unseen scenarios, traditional reinforcement learning (RL) methods face persistent challenges including excessive training overhead, inefficient exploration, and even failure to learn viable parking strategies in challenging settings. To address these limitations, this paper presents a correction-in-the-loop sample-efficient reinforcement learning (CIL-SERL) framework for end-to-end autonomous parking, which is entirely trained in a photorealistic 3D Gaussian Splatting (3DGS) parking simulator that enables high-fidelity digital reconstruction of real-world scenes. Inspired by error-correction notebooks used in learning practice, we design a novel multi-level replay buffer mechanism. These buffers hierarchically organize and store standard RL rollouts, human corrective interventions, failed exploration trajectories, and rollback-based correction segments in separate yet interconnected memory regions, facilitating structured sampling and targeted learning during training. The proposed framework is systematically evaluated in both the 3DGS simulation environment and a physical vehicle platform. Extensive experimental results demonstrate that our method achieves substantial improvements in parking success rate, operational efficiency, and safety performance across diverse scenarios, validating the effectiveness and practical applicability of the proposed CIL-SERL-based end-to-end autonomous parking solution.
>
---
#### [replaced 019] Vector Fields for Path Following on Lie Groups with Application in Robot Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究机器人路径跟踪问题，针对李群上的路径跟随提出向量场框架，解决机器人在3D空间中连续姿态跟踪的控制问题。**

- **链接: [https://arxiv.org/pdf/2602.21450](https://arxiv.org/pdf/2602.21450)**

> **作者:** Felipe Bartelt; Luciano C. A. Pimenta; Weijia Yao; Vinicius M. Gonçalves
>
> **备注:** Manuscript revised: new title, reframed abstract and introduction for robotics, and added a coauthor
>
> **摘要:** Many robotic systems allow independent control of position and orientation (pose), including omnidirectional aerial vehicles, underwater robots, and manipulator end-effectors. In many applications, these systems must follow a continuous sequence of poses, leading to either trajectory-tracking or path following formulations. Compared to trajectory-tracking, path following offers important practical advantages. In particular, we focus on the problem of path following on Lie groups. Considering the robots as rigid bodies moving in the 3D space, this path-following problem can be posed as a problem of designing guiding vector fields on the matrix Lie group SE(3). In this paper, we develop a general vector-field framework for path following on connected matrix Lie groups, of which SE(3) is a prominent special case. The proposed vector field guarantees convergence to a desired parametric curve from almost all initial conditions while ensuring continuous motion along the path. Furthermore, another interesting feature is that, as opposed to previous works, the control input is "minimal" in terms of representation and closer to the engineering application (e.g., the body twist in the case SE(3)). After establishing the general case, the framework is then specialized to SE(3), of special interest in robotics, yielding an efficient algorithm suitable for real-time robotic control. Experiments with a robotic manipulator tracking complex pose paths demonstrate the effectiveness of the approach. An open-source implementation is also provided.
>
---
