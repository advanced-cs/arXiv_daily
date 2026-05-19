# 机器人 cs.RO

- **最新发布 91 篇**

- **更新 70 篇**

## 最新发布

#### [new 001] Support-Safe Variational Hybrid Filtering for Contact-Mode and Sparse-Law Recovery
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VHYDRO，解决机器人接触动态中的状态和模式推断问题，通过变分方法确保过滤稳定性与物理模型恢复。**

- **链接: [https://arxiv.org/pdf/2605.16398](https://arxiv.org/pdf/2605.16398)**

> **作者:** Marios Papamichalis; Regina Ruane
>
> **摘要:** Contact-rich robot dynamics are hybrid: a single observation can match several latent states and contact regimes (free, impact, stick--slip). A standard amortized filter that places no probability on a feasible contact transition will permanently lose the branch the robot actually follows. We introduce VHYDRO, a variational hybrid dynamics learner that prevents this branch loss. At each step, VHYDRO mixes the learned proposal with a feasible transition law before sampling and importance weighting, ensuring that every transition retained by the model-feasible carrier remains covered. VHYDRO jointly infers a continuous latent state and a discrete contact mode, and fits a sparse port-Hamiltonian law to each recovered regime. On top of this, three guarantees connect: support coverage stabilizes filtering, the stabilized filter concentrates the discrete contact posterior on coherent regimes, and mode-pure segments admit sparse port-Hamiltonian recovery. The recovery error separates cleanly into filtering, derivative, mode-impurity, and physics-residual parts. Three empirical findings track the same mechanism. Under heavy occlusion the support-safe filter stays usable while a non-defensive proposal collapses. On ManiSkill demonstrations and on four Sawyer/BridgeData task families the discrete state forms temporally coherent contact-regime segments that the discrete state yields a stronger joint profile across ARI, change-point F1, and segment purity than post-hoc and mode-free baselines. On hybrid systems with known equations the mode-conditioned sparse fit recovers the active physical terms; purely predictive baselines do not.
>
---
#### [new 002] Optimal Knock-Pick Planning for Tightly Packed Tabletop Blocks With Parallel Grippers
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究密集排列物体的最优敲击抓取问题，解决平行夹爪无法操作时的重排难题。通过引入方向性敲击原语和图匹配方法，实现高效动作规划。**

- **链接: [https://arxiv.org/pdf/2605.17800](https://arxiv.org/pdf/2605.17800)**

> **作者:** Hao Lu; Rahul Shome
>
> **备注:** Accepted by WAFR 2026, 18 pages, 6 figures
>
> **摘要:** Rearranging densely packed tabletop objects is challenging when parallel-gripper picks are infeasible without sufficient clearance around an object. This work studies the problem characteristics for practically motivated settings with uniformly sized blocks placed at planar tabletop grid locations. Since purely prehensile removal can become infeasible, a directional knock primitive is therefore introduced and the optimal knock-pick variant of the problem is formulated. The work proposes a series of abstractions wherein minimal constraining gadgets are covered to identify the necessary knocks. Utilizing a maximum-weight perfect matching on a graphical abstraction yields efficient polynomial-time computation of the optimal plan that minimizes the number of actions. Experiments are reported for increasing grid sizes in synthetic settings as well as in IsaacSim. The theoretical observations provide a promising stepping stone towards rigorously building efficient manipulation strategies that interleave prehensile and non-prehensile actions.
>
---
#### [new 003] FUSE: A Framework for Unified State Estimation in Robotic SLAM Systems
- **分类: cs.RO**

- **简介: 该论文属于机器人SLAM任务，旨在解决多速率传感下的状态估计问题。提出FUSE框架，统一处理观测、传播、更新和查询，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.18047](https://arxiv.org/pdf/2605.18047)**

> **作者:** Wei Wu; Honglin Chen; Wenhan Cao; Yao Lyu; Jiangtao Li; Tao Zhang; Shengbo Eben Li
>
> **摘要:** Tightly coupled SLAM formulations under mixed-rate sensing often bind temporal processing, local geometric association, estimator formulation, and map-update policy into method-specific designs. Such binding makes it difficult to vary one design choice without re-engineering the rest of the state-estimation process. This paper presents FUSE, a framework for unified state estimation in robotic SLAM systems. FUSE organizes the state-estimation interface around observation ingestion, propagation, update, and state query, and uses this interface to separate temporal processing, residual-ready local geometric association, estimator formulation, and map-update policy. A LiDAR--IMU instantiation is developed to examine the framework under mixed-rate sensing and directional degeneracy, where high-rate inertial propagation, LiDAR-triggered geometric update, residual screening, and degeneracy-aware correction operate through the same interface boundaries. On a 418 m loop-corridor sequence, the instantiation reports a 1.626~m end-to-end trajectory error, corresponding to a 7.9% relative error reduction compared with Faster-LIO, the lowest-error baseline on this sequence. The results support FUSE as a framework for organizing state-estimation design choices and show how the evaluated instantiation regularizes updates along weakly observable directions.
>
---
#### [new 004] Nori Bot: A Sub-$1,000 Floor-to-Counter Mobile Manipulator
- **分类: cs.RO**

- **简介: 该论文介绍了一款低成本的双臂移动操作机器人Nori Bot，解决固定工作空间、反应控制和伺服烧毁问题。属于移动操作任务，通过创新设计实现更灵活、安全的自主操作。**

- **链接: [https://arxiv.org/pdf/2605.16537](https://arxiv.org/pdf/2605.16537)**

> **作者:** Antonio Li; Sungjoon Park; Wen Ni Chew
>
> **备注:** 7 pages, 3 figures, 2 tables. Columbia University Deep Learning Robot Manipulation course project, Spring 2026
>
> **摘要:** Open-source mobile manipulators have reached $660 (XLeRobot) but every sub-$1,000 platform shares three limitations: a fixed-height workspace, reactive-only control, and no protection against the stall-induced burn-out that destroys cheap Feetech servos. We present Nori Bot, a 17-DoF dual-arm mobile manipulator at $947 (~3% the cost of comparable commercial platforms) that addresses all three: (1) a 600mm Z-axis lift on the existing servo bus for floor-to-counter reach; (2) a thin-client Raspberry Pi 4 paired with the OpenClaw proactive agent runtime so cron jobs and hooks trigger physical tasks autonomously; and (3) a software safety stack with sensorless grip-force feedback via motor current on a soft TPU finger. Code, CAD, and the skill manifest will be released.
>
---
#### [new 005] From a Single Demonstration to a General Policy for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决多阶段接触密集操作的一次性泛化问题。通过利用环境约束作为归纳偏置，构建通用策略，实现高效泛化。**

- **链接: [https://arxiv.org/pdf/2605.17601](https://arxiv.org/pdf/2605.17601)**

> **作者:** Xing Li; Oliver Brock
>
> **备注:** 21 pages, 22 figures, 7 tables
>
> **摘要:** We present a Learning from Demonstration (LfD) framework that achieves one-shot generalization in multi-stage, contact-rich manipulation tasks. Central to our approach is the utilization of environmental constraints as the inductive bias. By representing a demonstration as a sequence of behaviors that exploit environmental constraints, the robot separates task-general structure -- the constraint types and their transitions -- from instance-specific details such as exact demonstration trajectories, poses, and local geometries. Our four-stage pipeline builds a complete policy on this representation: the robot first abstracts a single demonstration into environmental-constraint primitives, then disambiguates them through self-guided exploration, next assimilates targeted human corrections that handle out-of-distribution variations, and finally recovers the abstracted-away details online through compliant interaction. Because the resulting policy follows constraints rather than mimics trajectories, it generalizes across object poses, local geometries, and unmodeled contact dynamics. We validate our approach on seven real-world multi-stage contact-rich manipulation tasks and achieve over 90% success. These extensive experimental results establish environmental constraints as fundamental building blocks for efficient generalization in learning from demonstration.
>
---
#### [new 006] DriveSafer: End-to-End Autonomous Driving with Safety Guidance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决E2E模型在安全关键场景中的灾难性失败问题。提出DriveSafer框架，通过安全约束和引导提升安全性。**

- **链接: [https://arxiv.org/pdf/2605.16737](https://arxiv.org/pdf/2605.16737)**

> **作者:** Shounak Sural; Raj Rajkumar
>
> **摘要:** End-to-End (E2E) autonomous driving models have shown growing capability in recent years, with performance improving on increasingly challenging benchmarks. However, modern generative E2E planners still suffer from a substantial number of catastrophic failures in safety-critical scenarios. We find that many such failures arise from violations of physical constraints and safety requirements, leading to unsafe behavior. Motivated by this finding, in this paper, we focus on improving safety outcomes in generative end-to-end driving with a targeted reduction of catastrophic planning failures, instead of enhancing average planning quality. Towards this end, we propose DriveSafer, a failure-aware safety framework for end-to-end planners. DriveSafer explicitly steers generative planners towards safe behaviors leveraging both training-time safety constraints and inference-time safety guidance. Compared to the state-of-the-art DiffusionDrive model, on the NAVSIM benchmark, DriveSafer reduces the number of catastrophic failures (PDMS=0) by 48%, with over 65% reduction in drivable-area compliance failures.
>
---
#### [new 007] Scenario Generation in Roundabouts with Adjustable Interaction Intensity
- **分类: cs.RO**

- **简介: 该论文属于智能驾驶安全测试任务，旨在解决场景生成中交互强度难以控制的问题。通过调整yield代码的缩放因子λ，实现交互强度的连续调节，提升测试的可控性与有效性。**

- **链接: [https://arxiv.org/pdf/2605.18026](https://arxiv.org/pdf/2605.18026)**

> **作者:** Li Li; Till Temmen; Tobias Brinkmann; Björn Krautwig; Markus Eisenbarth; Jakob Andert
>
> **摘要:** Roundabouts, characterized by frequent merging and yielding interactions, remain a safety-critical corner case for the development and testing of intelligent driving functions. However, extracting sufficient near-critical scenarios from naturalistic data is inefficient. Most existing scenario generation methods provide limited controllability over interaction intensity and criticality, making systematic safety testing and detailed analysis difficult. This paper presents an interaction-aware roundabout scenario generator with continuously adjustable interaction intensity. Geometric routes and temporal progress profiles are first decoupled and mapped to latent codes using pretrained autoencoders. Conditional latent generation is then performed with Wasserstein Generative Adversarial Networks (WGAN) to generate scenarios. Yielding is modeled as a controllable timing intervention via a compact yield code during the approach-to-entry segment, where interaction intensity is modulated by scaling the code with a factor $\lambda$. Results demonstrate enhanced timing-latent fidelity and plausible interaction responses compared to a baseline model. Under criticality-calibrated scaling, increasing $\lambda$ expands the safety margin, providing a scalable and controlled testing mechanism.
>
---
#### [new 008] ManiSoft: Towards Vision-Language Manipulation for Soft Continuum Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ManiSoft，一个用于软体机械臂视觉-语言操作的基准。解决软体机器人在复杂环境中的控制难题，通过模拟与策略训练提升其适应能力。**

- **链接: [https://arxiv.org/pdf/2605.18617](https://arxiv.org/pdf/2605.18617)**

> **作者:** Ziyu Wei; Luting Wang; Chen Gao; Li Wen; Si Liu
>
> **备注:** Accepted in ICML 2026
>
> **摘要:** Most existing vision-language manipulation research targets rigid robotic arms, whose fixed morphology limits adaptability in cluttered or confined spaces. Soft robotic arms offer an appealing alternative due to their deformability, but confront challenges such as unreliable proprioception and distributed low-level actuation. To investigate these challenges, we introduce \ManiSoft, a benchmark for vision-language manipulation with soft arms. ManiSoft features a tailored simulator that couples realistic soft-body dynamics with contact-rich interactions via an elastic force constraint. On this basis, ManiSoft defines four tasks, each highlighting distinct aspects of deformable control, from basic end-effector coordination to obstacle avoidance. To support policy training and evaluation, \ManiSoft{} includes an automated pipeline that generates $6{,}300$ diverse scenes and corresponding expert trajectories. To produce high-quality trajectories at scale, we first employ a high-level planner to decompose each task into a sequence of waypoints, followed by a low-level reinforcement learning policy that generates torque commands to track waypoints. Benchmarking three representative policy models shows relatively promising results in clean scenes but substantial performance drop under randomization. Visualization analysis indicates that failures stem primarily from inaccurate visual estimation of proprioceptive state and limited exploitation of deformability for adaptive obstacle avoiding. We anticipate ManiSoft to serve as a valuable testbed, bridging the gap between rigid and soft arms in the context of vision-language manipulation. Out codes and datasets are released at this https URL.
>
---
#### [new 009] Stretch-ICP: A Continuous-Trajectory Registration and Deskewing Algorithm in Scenarios of Aggressive Motions
- **分类: cs.RO**

- **简介: 该论文属于机器人SLAM任务，解决高速运动下传感器数据失真导致的定位不准问题。提出SAAVE和Stretch-ICP算法，提升状态估计的鲁棒性与一致性。**

- **链接: [https://arxiv.org/pdf/2605.17264](https://arxiv.org/pdf/2605.17264)**

> **作者:** Simon-Pierre Deschênes; Veronica Vannini; Philippe Giguère; François Pomerleau
>
> **备注:** 29 pages, 16 figures, published in Sensors 2026, 26(8), 2567, special issue "New Challenges and Sensor Techniques in Robot Positioning"
>
> **摘要:** Robust robotic autonomy remains challenging in complex environments, where loss of stability on uneven or slippery terrain can induce extreme accelerations and angular velocities. Such motions corrupt sensor measurements and degrade state estimation, motivating the need for improved algorithmic robustness. To investigate this issue, we introduce the Tumbling-Induced Gyroscope Saturation (TIGS) dataset, which consists of recordings from a mechanical lidar and an Inertial Measurement Unit (IMU) tumbling down a hill. The dataset contains angular speeds up to four times higher than those in similar datasets and is publicly available. We then propose two complementary methods to improve Simultaneous Localization And Mapping (SLAM) robustness and evaluate them on TIGS. First, Saturation-Aware Angular Velocity Estimation (SAAVE) estimates angular velocities when gyroscope measurements become saturated during aggressive motions, reducing angular speed estimation error by 83.4%. Second, Stretch-ICP, a novel registration and deskewing algorithm, enables reconstruction of smoother 6-Degrees Of Freedom (DOF) trajectories under aggressive motions compared to classical Iterative Closest Point (ICP). Stretch-ICP reduces linear and angular velocity errors by 95.2% and 94.8%, respectively, at scan boundaries. Together, these contributions improve the robustness and consistency of lidar-inertial state estimation under aggressive motions.
>
---
#### [new 010] Geometry-Aware Surrogate for Real-Time Hydrodynamics Estimation of Autonomous Ground Vehicles in Amphibious Environments
- **分类: cs.RO**

- **简介: 该论文属于自主地面车辆在水陆环境中的实时流体动力学估计任务，旨在解决现有模型精度不足或计算成本过高的问题。通过构建基于几何的神经网络代理模型，实现快速准确的流体力预测。**

- **链接: [https://arxiv.org/pdf/2605.18543](https://arxiv.org/pdf/2605.18543)**

> **作者:** Ammar Waheed; Luke Gallantree; Zohaib Hasnain
>
> **摘要:** Autonomous ground vehicles operating in shallow water or flood-prone terrains require dynamic models that account for hydrodynamic forces. However, the simulation and planning tools currently available either lack the physical fidelity or are too computationally expensive to run in real time. This work presents a per-surface neural network surrogate that bridges this gap by predicting geometry-resolved hydrodynamic forces at real-time rates, trained entirely on high-fidelity CFD data from two geometrically distinct vehicles. A vehicle specific Signed Distance Field (SDF) provides per-surface submergence inputs, allowing the model to resolve how loading varies with vehicle geometry, depth, and flow direction. On held-out CFD data, the surrogate achieves a longitudinal-force symmetric MAPE (sMAPE) of 13\% and a vertical-force sMAPE of 3-12\%, with inference running under 0.9\,ms per sample. To evaluate the model under real-world conditions, water wading trials of a full-scale vehicle at different submersion depths are used. Motion capture derived kinematics serve as the surrogate inputs, and the resulting predictions are tested to reproduce known physical relationships between force, speed, and depth. The predicted drag follows quadratic speed scaling ($R^2 \geq 0.97$) and the buoyancy intercepts scale linearly with depth ($R^2 = 0.973$). Neither relationship is encoded in the model training loss, both emerge from the per-surface architecture summing individually predicted surface forces. The resulting framework provides a pathway for embedding physically grounded hydrodynamics into the simulation and planning loops that autonomous ground vehicles depend on in amphibious environments.
>
---
#### [new 011] SCAR: Self-Supervised Continuous Action Representation Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SCAR，解决跨实体动作表示学习问题，通过联合逆向与正向动力学模型，学习可迁移的潜在动作表示，提升世界建模的泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.16412](https://arxiv.org/pdf/2605.16412)**

> **作者:** Hongjia Liu; Fan Feng; Minghao Fu; Xinyue Wang; Haofei Lu; Biwei Huang
>
> **摘要:** Despite the central role of action in embodied intelligence, learning transferable action representations from visual transitions remains a fundamental challenge, particularly when world models must generalize across embodiments under limited data. We argue that action is not merely an auxiliary conditioning signal, but a distinct representational factor that decouples the controllable change from embodiment-specific actuation. In this work, we propose SCAR, a joint inverse-forward dynamics framework for learning unified action representations across embodiments from visual transitions. Built on a pretrained generative backbone, SCAR uses an inverse dynamics model (IDM) to infer latent actions from latent observation pairs and a forward dynamics model (FDM) to predict future dynamics conditioned on them. To make the latent space transferable rather than a generic visual bottleneck, we regularize the latent action posterior toward a standard Gaussian prior to limit arbitrary visual encoding, and introduce adversarial invariance to suppress embodiment- and environment-specific nuisance factors. Experiments on the Procgen and Robotwin dataset show that the learned unified latent action representation serves as a stronger conditioning interface for world modeling than embodiment-specific raw actions, yielding improved cross-embodiment low-data adaptation and cross-task transfer. Taken together, these results suggest that action can be learned as a shared representation of controllable change across embodiments, providing an interface for more transferable and generalizable world models.
>
---
#### [new 012] "I'm Not Mad, Just Focused'': Understanding Human Emotions in Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文属于情感识别任务，旨在提升人机协作中情绪理解的准确性。针对现有模型依赖表演数据和单一模态的问题，提出基于视觉语言模型的ER系统，并通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.16816](https://arxiv.org/pdf/2605.16816)**

> **作者:** Seung Chan Hong; Dana Kulić; Leimin Tian
>
> **摘要:** Human-robot collaboration (HRC) can benefit from robots' abilities to interpret human emotional states. However, current emotion recognition (ER) models in HRC often fall short, particularly due to their reliance on acted datasets and single-modality inputs like facial expressions. We propose a novel vision language model (VLM)-based ER system that leverages contextual understanding to improve emotion interpretation in HRC. We first evaluate the VLM-ER system by assessing its semantic and sentiment similarity with human annotations on an existing HRC dataset. Then, in a user study with a service robot in a collaborative delivery task, we evaluate the effects of modulating the robot's behaviour based on the user's emotional state inferred by the VLM-ER system. The results show that the proposed VLM-ER system achieves higher semantic similarity and positive sentiment alignment with human annotations compared to a baseline convolutional neural network-based system. Further, participants in the user study preferred emotion-adaptive robot behaviour facilitated by the VLM-ER system.
>
---
#### [new 013] Rapid Vibration Suppression and Trajectory Tracking of a Serial Manipulator with Multi-Flexible Links
- **分类: cs.RO**

- **简介: 该论文属于控制任务，旨在解决多柔体机械臂的振动抑制与轨迹跟踪问题。通过构建后推输出反馈框架和引入DeepONet神经算子，实现快速控制与高效部署。**

- **链接: [https://arxiv.org/pdf/2605.17477](https://arxiv.org/pdf/2605.17477)**

> **作者:** Chengyi Wang; Yilong Huang; Ji Wang
>
> **摘要:** Flexible robotic manipulators (FRMs) offer advantages in lightweight design and large workspace, but their structural flexibility induces vibrations, accelerates fatigue, degrades tracking performance, and limits operational speed. These challenges are further amplified in multi-link serial manipulators, where increased overall length leads to greater structural flexibility. This article presents a backstepping output-feedback framework for fast vibration suppression and tip tracking of an n-degree-of-freedom serial flexible manipulator robot (nDSFMR), with a DeepONet-based approximation for practical deployment. Each link-joint is modeled as a Timoshenko beam coupled with an ODE and transformed into a canonical hyperbolic PDE with boundary dynamics. A backstepping-based boundary controller at the joint is developed to equivalently inject distributed damping along the beam, enabling rapid vibration suppression and trajectory tracking, only using available boundary measurements. To enable real-time implementation and scalability, a DeepONet neural operator is introduced to approximate the backstepping kernels, significantly reducing computational cost and facilitating fast controller updates under varying operating conditions. Experiments on a two-link flexible manipulator demonstrate faster vibration suppression and convergence of the end-effector to the desired trajectory, compared with a linear quadratic regulator (LQR) with feedforward control.
>
---
#### [new 014] Confidence-Gated Robot Autonomy: When Does Uncertainty Actually Help?
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人自主性决策问题，探讨不确定性在决定是否自主执行任务中的作用。通过实验验证不同方法在不同数据条件下的表现，发现当模型足够可靠时，简单不确定性代理即可有效用于选择性决策。**

- **链接: [https://arxiv.org/pdf/2605.18045](https://arxiv.org/pdf/2605.18045)**

> **作者:** Johannes A. Gaus; Jhon P.F. Charaja; Daniel Haeufle
>
> **备注:** ICRA 2026 workshop paper
>
> **摘要:** Robotic systems often use predictive uncertainty to decide whether to act autonomously or defer to a fallback policy. In threshold-gated autonomy, uncertainty matters mainly through its ability to rank likely errors. Standard metrics such as expected calibration error and AUROC do not directly test whether uncertainty changes act/defer decisions. We therefore evaluate uncertainty using Spearman rank correlation, paired bootstrap equivalence testing, and act/defer agreement. Across three temporal activity-recognition benchmarks, we find a dataset-dependent competence regime below which uncertainty provides a weak and unstable error ranking. Above this regime, softmax heuristics, MC Dropout, and ensembles produce similar gating behavior, while threshold choice has a much larger effect on execution outcomes. A multi-seed embodied simulation shows the same pattern for collision rate and cost once realized autonomy is matched. Under temporal covariate shift, ranking quality remains stable, but fine grained semantic OOD detection remains near chance. These results suggest that simple uncertainty proxies can suffice for selective gating once the base model is competent, but not for semantic novelty detection.
>
---
#### [new 015] Policy Library CBF: Finite-Horizon Safety at Runtime via Parallel Rollouts
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于安全控制任务，解决在线安全认证难题。提出PL-CBF方法，通过并行滚动评估多策略，确保运行时安全。**

- **链接: [https://arxiv.org/pdf/2605.16588](https://arxiv.org/pdf/2605.16588)**

> **作者:** Taekyung Kim; Hideki Okamoto; Bardh Hoxha; Georgios Fainekos; Dimitra Panagou
>
> **备注:** Project page: this https URL
>
> **摘要:** Safety-critical autonomy in unstructured environments poses significant challenges for online safety certification under evolving constraints. We propose Policy Library Control Barrier Function~(PL-CBF), a runtime safety filter that evaluates a library of fallback policies via parallel finite-horizon rollouts, selects the least invasive safe mode, and enforces safety by solving a quadratic program that minimally modifies a nominal policy. We provide a theoretical analysis based on a finite-horizon language metric over closed-loop behaviors, characterizing policy-library coverage requirements for certifying finite-horizon safety. Simulations on a planar double-integrator (4 states), highway driving with abrupt friction changes using a realistic nonlinear vehicle model (8 states), and 3D quadrotor navigation in crowded dynamic environments (12 states) demonstrate improved safety coverage over single-policy safety filters while retaining millisecond-level runtime.
>
---
#### [new 016] Bidirectional Optical sensors for Actuation Tracking (BOAT) in soft lattice systems
- **分类: cs.RO**

- **简介: 该论文属于软体机器人传感任务，旨在解决软晶格系统变形监测问题。提出一种双向光学传感器（BOAT），通过波导弯曲实现压缩与拉伸状态的区分。**

- **链接: [https://arxiv.org/pdf/2605.18482](https://arxiv.org/pdf/2605.18482)**

> **作者:** Petr Trunin; Carolina Gay; Anderson Brazil Nardin; Trevor Exley; Diana Cafiso; Lucia Beccai
>
> **摘要:** The growing adoption of lattice-based structures in soft robotics creates a need for advanced sensing solutions capable of monitoring their global deformation, particularly compression and extension. In this work, we address this challenge by introducing a novel optical sensor based on two patterned waveguides arranged in an ellipsoidal geometry. This Bidirectional Optical sensor for Actuation Tracking (BOAT) is seamlessly co-printed with a lattice structure actuated by an embedded pneumatic artificial muscle (PAM), and its performance is assessed. During PAM elongation or contraction, the bending of the embedded BOAT waveguides induces output signal variations that enable a clear discrimination between compression and extension states. The designs of both each specific waveguide structure (by surface patterning) and of the sensorized lattice-based unit embedding two BOATs are supported by numerical simulations. Experimental calibration over 100 consecutive pressure cycles ranging from +50 kPa to $-$40 kPa demonstrates a highly repeatable response, allowing a reliable distinction between extension and compression. Finally, sensor feedback is used to implement a digital shadow, enabling continuous synchronization between the whole sensorized unit and its virtual counterpart. These results establish BOAT as a powerful and reliable approach for deformation monitoring in soft lattice-based robotic systems.
>
---
#### [new 017] Efficient Feature-Free Initialization for Monocular Visual-Inertial Systems Using a Feed-Forward 3D Model
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉-惯性导航任务，解决单目系统初始化效率低的问题。提出无需特征点的初始化方法，利用3D模型预测点云，提升速度与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.17327](https://arxiv.org/pdf/2605.17327)**

> **作者:** Yuantai Zhang; Jiaqi Yang; Huajian Zeng; Changhao Chen; Haoang Li; Liang Li; Dezhen Song; Xingxing Zuo
>
> **摘要:** Fast and reliable initialization is critical for monocular visual-inertial navigation systems (VINS), as it establishes the starting conditions for subsequent state estimation. Despite steady progress, most existing methods heavily rely on visual feature correspondences and require 3-4 seconds of sensory data for successful initialization, which limits their applicability and efficiency. With the advent of feed-forward 3D models that can directly predict point clouds from images, we revisit the visual-inertial initialization problem from a concise perspective. In this work, we propose a feature-free initialization framework that leverages up-to-scale point clouds predicted by a feed-forward 3D model, thereby obviating the need for visual feature tracking and estimation. This design substantially reduces system complexity and improves the reliability of initialization. Experiments on public datasets demonstrate that the proposed feature-free initialization method achieves the highest success rate, exceeding 90%, and significantly reduces the data duration required for successful initialization, typically to under 1.2 s. We further validate our method on a self-collected dataset covering various indoor and outdoor scenarios, demonstrating robust performance, particularly in visually degraded environments where existing methods often fail. The code and dataset are available at this https URL.
>
---
#### [new 018] DyGRO-VLA: Cross-Task Scaling of Vision-Language-Action Models via Dynamic Grouped Residual Optimization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型优化任务，旨在解决VLA模型在多任务中的泛化能力不足问题。提出DyGRO-VLA框架，提升跨任务特征表示与策略优化效果。**

- **链接: [https://arxiv.org/pdf/2605.17486](https://arxiv.org/pdf/2605.17486)**

> **作者:** Sixu Lin; Yunpeng Qing; Litao Liu; Ming Zhou; Ruixing Jin; Xiaoyi Fan; Guiliang Liu
>
> **摘要:** Recent progress in Reinforcement Learning (RL) provides a principled approach to optimizing Vision-Language-Action (VLA) models, facilitating a shift from trajectory imitation to active learning in the task environment. Despite improvements in control precision, most RL optimizers remain task-specific, which reduces VLA models from generalist controllers to policies that overfit to a narrow set of tasks. In this study, we conduct an in-depth analysis of this phenomenon and highlight the importance of cross-task feature representations for improving the generalizability of VLA models. Motivated by this finding, we introduce DyGRO-VLA, a two-stage optimization framework that 1) effectively captures cross-task latent representations based on information-theoretic principles, and 2) dynamically refines policy optimization via a mixture-of-RL-residuals. DyGRO-VLA enables the RL optimizer to exploit task-relevant latent information while strategically mitigating adverse interference on the learned representations throughout the optimization process. We evaluate our approach on LIBERO, RoboTwin2 benchmarks, and further validate it on real world, demonstrating consistent improvements over strong baselines under multi-task training and distribution shift.
>
---
#### [new 019] Task Capability Improvement Algorithm for Collaborative Manipulators
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究协作机械臂的任务能力提升，解决因力作用点偏离质心产生的不利力矩问题。通过引入附加力矩，提高单个机械臂及协作群体的任务能力。**

- **链接: [https://arxiv.org/pdf/2605.17293](https://arxiv.org/pdf/2605.17293)**

> **作者:** Keshab Patra; Arpita Sinha; Anirban Guha
>
> **摘要:** This work introduces a cooperative task capability improvement utilizing additional moments. The manipulators apply forces at the object's grasp point. Applying forces at a point other than the object's center of gravity produces undesired moments. The undesired moment acts as an additional moment. It improves the capability of an individual manipulator and, hence, the entire collaborative group. Any improvements in task capability directly add up to the object and transportation capability. The group's enhanced capability also helps achieve optimal capability, optimal resource allocation, and maximum fault tolerance in object manipulation. Our simulation results show an improvement in the capability of 5.86 \% compared to when no moment is used to enhance the capability of the manipulators.
>
---
#### [new 020] Plan First, Diffuse Later: Extrinsic Graph Guidance for Long-Horizon Diffusion Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于长期规划任务，解决长序列生成中全局结构不连贯的问题。通过引入外部搜索，在扩散模型前生成路径引导，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2605.16863](https://arxiv.org/pdf/2605.16863)**

> **作者:** Yaniv Hassidof; Adir Morgan; Yilun Du; Kiril Solovey
>
> **摘要:** Compositional diffusion models offer a promising route to long-horizon planning by denoising multiple overlapping sub-trajectories while ensuring that together they constitute a global solution. However, enforcing local behavior over long chains is often insufficient for a coherent global structure to emerge. Recent works tackle this limitation through intrinsic search, which explores multiple paths during the denoising process. While intrinsic search improves global coherence, it comes at the cost of repeated evaluations of an already compute-heavy model. In this work, we argue that extrinsic search, performed outside the denoising process, offers a more effective mode of exploration for long-horizon planning while naturally enabling the use of classical algorithms to solve unseen combinatorial tasks at test time. Our eXtrinsic search-guided Diffuser (XDiffuser) first computes a plan over a state-space graph -- serving as a lightweight local connectivity oracle for the diffusion model. The plan is then used to guide denoising for a single trajectory, effectively offloading the burden of exploration. XDiffuser outperforms diffusion-based baselines on long-horizon tasks, with particularly large gains in the low-quality data regime and on unseen tasks beyond goal-reaching, including multi-agent coordination and TSP-style reasoning. Project website: this https URL
>
---
#### [new 021] Pedestrian-Aware LLM-Driven Behavioral Planning for Autonomous Vehicles
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶行为规划任务，旨在解决复杂行人交互下的决策问题。通过引入LLM框架，提升AV对行人意图的感知与安全决策能力。**

- **链接: [https://arxiv.org/pdf/2605.16858](https://arxiv.org/pdf/2605.16858)**

> **作者:** Aidana Baimbetova; Haruki Yonekura; Hamada Rizk; Hirozumi Yamaguchi
>
> **备注:** This paper has been accepted for presentation at the 29th IEEE International Conference on Intelligent Transportation Systems (ITSC)
>
> **摘要:** Autonomous Vehicles (AVs) must make reliable decisions in dense urban environments where pedestrian behavior is variable, sometimes abnormal, and often unseen during training. Reinforcement learning (RL)-based AV control systems perform well in structured traffic but struggle to generalize to unpredictable pedestrian interactions and out-of-distribution scenarios. Their reliance on handcrafted rewards and opaque decisions further limits their suitability for safety-critical, pedestrian-rich environments. To address these limitations, we introduce a Large Language Model (LLM)-based decision-making framework for pedestrian-aware behavioral planning. The system converts structured scene observations into natural-language reasoning prompts, enabling the LLM to infer pedestrian intent, anticipate risk, and generate cautious tactical driving decisions. These decisions are executed by a motion planner that ensures smooth, kinematically feasible control. We evaluate the framework in SUMO across multiple pedestrian-interaction scenarios, including unexpected jaywalking, turn-back crossing, hesitation, and bidirectional crossing. In zero-shot evaluation, the LLM-based agent achieves a 68% collision-free success rate, substantially outperforming deep RL baselines (17.7%). With few-shot episodic memory in a single-pedestrian scenario, performance increases to 96.0%, exceeding a custom DQN controller (82.0%). Cross-behavior evaluation further shows that memory derived from turn-back interactions transfers to unseen hesitation and bidirectional crossing scenarios, achieving 82.0% and 90.0% success, respectively. The system consistently initiates earlier responses, maintains wider safety buffers, and produces interpretable, human-aligned decisions.
>
---
#### [new 022] RGB-only Active 3D Scene Graph Generation for Indoor Mobile Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于3D场景图生成任务，解决无深度传感器下构建3D场景图的问题。提出仅使用RGB输入的主动框架，实现高效场景理解与探索。**

- **链接: [https://arxiv.org/pdf/2605.18197](https://arxiv.org/pdf/2605.18197)**

> **作者:** Giorgia Modi; Davide Buoso; Giuseppe Averta; Daniele De Martini
>
> **摘要:** Current approaches to 3D scene graph generation rely on dedicated depth sensors, such as LiDAR or RGB-D cameras, for metric 3D reconstruction. This limits deployment to specialized robotic platforms and excludes settings where only RGB cameras are available, such as fixed external infrastructure. Existing pipelines also typically operate on passively collected observation trajectories, rather than selecting viewpoints based on the partially built scene representation, and therefore fail to effectively exploit the semantic and spatial information encoded within the graph during exploration. This paper presents a fully visual framework for the active, incremental construction of 3D scene graphs from RGB input only, addressing both limitations. The proposed approach unifies perception and planning around a shared structured representation that captures object semantics, 3D geometry, relational context, and information from multiple viewpoints. Because the framework is hardware-agnostic and relies only on RGB observations, it can incorporate inputs from both onboard robot cameras and fixed external cameras within the same representation. Experiments on the Replica dataset show that the RGB-only pipeline achieves F1-score parity with baselines using ground-truth depth. Active exploration experiments on ReplicaCAD further show that semantic-driven viewpoint selection detects more than twice as many objects as a geometric frontier-based baseline under the same exploration budget. Finally, the external-camera setting demonstrates that complementary RGB views can effectively bootstrap the scene graph and improve contextual understanding at no additional exploration cost.
>
---
#### [new 023] A Dexterous and Compliant Gripper With Soft Hydraulic Actuation for Microgravity Manipulation
- **分类: cs.RO**

- **简介: 该论文属于微重力环境下的机械臂操作任务，旨在解决现有抓取器操控能力不足的问题。研究提出一种六自由度柔性液压抓取器，提升精准操作能力并减少对基座的干扰。**

- **链接: [https://arxiv.org/pdf/2605.17851](https://arxiv.org/pdf/2605.17851)**

> **作者:** William Su; Jordan Kam; Yixiao Wang; Jianshu Zhou
>
> **备注:** Accepted to the IEEE ICRA 2026 Space Robotics Workshop (SRW). 4 pages, 3 figures
>
> **摘要:** Astrobee's existing one-degree-of-freedom (DOF) underactuated compliant claw gripper enables perching on the International Space Station (ISS), but provides limited capability for continuous dexterous manipulation. More complex microgravity tasks require an end-effector that can maintain stable contact while limiting disturbance to the free-flying base, since contact forces directly couple into base motion. This article presents the integration of DexCoHand, a dexterous and compliant two-finger, 6-DOF gripper, with the Astrobee free-flying robot for microgravity manipulation. The system is evaluated in MuJoCo using Astrobee's standard handrail perching sequence, including approach, perching, and subsequent pan and tilt motions. Compared with Astrobee's existing gripper, DexCoHand preserves the commanded pan and tilt motions while reducing unintended cross-axis base motion. Hardware experiments on Earth further demonstrate DexCoHand's dexterous manipulation capabilities and its potential for more adaptable intelligent manipulation tasks.
>
---
#### [new 024] WorldArena 2.0: Extending Embodied World Model Benchmarking on Modality, Functionality and Platform
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人感知与决策任务，旨在解决现有基准不足的问题。通过扩展模态、功能和平台维度，构建更全面的评估体系。**

- **链接: [https://arxiv.org/pdf/2605.17912](https://arxiv.org/pdf/2605.17912)**

> **作者:** Yu Shang; Yinzhou Tang; Yiding Ma; Zhuohang Li; Lei Jin; Weikang Su; Xin Jin; Zhaolu Wang; Ziyou Wang; Xin Zhang; Haisheng Su; Weizhen He; Wei Wu; Haoyi Duan; Gordon Wetzstein; Xihui Liu; Dhruv Shah; Zhaoxiang Zhang; Zhibo Chen; Jun Zhu; Yonghong Tian; Tat-Seng Chua; Wenwu Zhu; Chen Gao; Yong Li
>
> **摘要:** World models have emerged as a central paradigm for embodied intelligence, enabling agents to predict action-conditioned future and reason about environmental dynamics. However, existing embodied world model benchmarks are still largely confined to vision-only prediction, offline embodied applications, and simulator-based evaluation, making them insufficient for assessing increasingly comprehensive world models. In this work, we introduce WorldArena 2.0, an expanded benchmark that systematically broadens embodied world model evaluation along three dimensions: modality, functionality, and platform. Along the modality dimension, WorldArena 2.0 extends evaluation from vision-only to visuotactile modalities, enabling assessment of multimodal perception and prediction. Along the functionality dimension, it extends beyond policy evaluation and planning to assess world models as interactive RL environments for policy optimization. Along the platform dimension, it moves beyond simulator-only evaluation to a diverse suite of simulated and real-world robotic settings across multiple embodiments. Under a standardized protocol, WorldArena 2.0 comprehensively evaluates perceptual quality, interactive utility, and cross-platform performance, providing a comprehensive testbed for tracking progress toward embodied world models. The benchmark is available at: this https URL.
>
---
#### [new 025] Assessing Localization Technologies for Pedestrian Collision Avoidance
- **分类: cs.RO**

- **简介: 论文评估UWB和蓝牙6.0在行人定位中的性能，解决智能交通系统中行人碰撞预警问题，通过实验对比其定位精度与卫星系统。**

- **链接: [https://arxiv.org/pdf/2605.18295](https://arxiv.org/pdf/2605.18295)**

> **作者:** Joshua Varughese; Joseba Gorospe; Novel Certad; Cristina Olaverri-Monreal
>
> **摘要:** Robust pedestrian safety is crucial to the next-generation of intelligent transportation systems. Such systems rely on active pedestrian localization and predictive collision alerts. Pedestrian localization can be supported by Ultra-Wideband technology and Bluetooth 6.0, which offer high-precision ranging and low-latency communication, making them promising candidates for vehicular collision warning systems. This paper assesses the localization accuracy of these technologies for pedestrian alerting and benchmarks their performance against Global Navigation Satellite Systems. Experimental evaluations performed in this paper focused on key performance metrics, including localization accuracy and robustness to environmental conditions. Preliminary results suggest that Ultra-Wideband and Bluetooth 6.0 can serve as viable alternatives or complements to Global Navigation Satellite Systems in certain scenarios, improving situational awareness and enabling timely pedestrian alerts.
>
---
#### [new 026] CosFly-Track: A Large-Scale Multi-Modal Dataset for UAV Visual Tracking via Multi-Constraint Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文提出CosFlyTrack数据集，解决UAV视觉跟踪缺乏专用训练数据的问题。通过多约束轨迹优化生成高质量轨迹，提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2605.17776](https://arxiv.org/pdf/2605.17776)**

> **作者:** Xiangyue Wang; Hanxuan Chen; Songsheng Cheng; Ruilong Ren; Jie Zheng; Shuai Yuan; Tianle Zeng; Hanzhong Guo; Kangli Wang; Ji Pei
>
> **摘要:** Recent aerial vision-language navigation (VLN) datasets have grown rapidly, but they primarily address goal-oriented navigation to static destinations, leaving UAV visual tracking -- continuously following a moving target while maintaining visibility -- largely without dedicated training data. We introduce CosFlyTrack, a large-scale multi-modal dataset and scalable generation pipeline for UAV visual tracking in urban environments. The dataset provides approximately 12,000 expert and perturbed UAV trajectories generated from 6,000 pedestrian paths, comprising 2.4 million timesteps (approximately 334 hours) with seven aligned data channels: RGB, metric depth, semantic segmentation, six-degree-of-freedom drone pose, target state with visibility flag, bilingual (Chinese-English) instructions, and trajectory-pair metadata. To generate high-quality expert trajectories, we develop MuCO, a multi-constraint optimizer that plans directly in continuous three-dimensional space with BVH-accelerated collision and visibility queries, jointly enforcing target visibility, viewpoint quality, collision avoidance, smoothness, and kinematic feasibility, avoiding the discretization artifacts and post-hoc smoothing of grid-based planners. Fine-tuning experiments on seven vision-language models show that CosFlyTrack improves tracking performance to 78.3 to 95.6 percent SR@1 meter, a 53 to 69 percentage point gain over zero-shot baselines, supporting the dataset as a training resource for dynamic target-following agents. The dataset is publicly available at this https URL evaluation scripts and pre-trained checkpoints are hosted at this https URL.
>
---
#### [new 027] Motion-Uncertainty-Aware Next-Best-View Planning for Moving Object Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于3D重建任务，解决移动物体在运动不确定性下的视点选择问题。提出一种考虑运动不确定性的下最佳视点规划方法，提升重建完整性。**

- **链接: [https://arxiv.org/pdf/2605.17593](https://arxiv.org/pdf/2605.17593)**

> **作者:** Karen Li; Mattia Mantovani; Robert J. Wood; Lorenzo Sabattini; Stephanie Gil
>
> **备注:** This paper is accepted for publication for Robotics: Science and Systems (RSS) 2026
>
> **摘要:** Active 3D reconstruction of moving objects requires selecting informative viewpoints while accounting for object motion uncertainty during the decision-to-execution delay. Existing methods address only parts of this problem: next-best-view (NBV) planners for object reconstruction typically optimize surface coverage but assume static objects, while motion-aware active perception for moving targets accounts for target motion but prioritizes tracking or visibility over reconstruction coverage. This work presents a motion-uncertainty-aware NBV framework for reconstructing an unknown rigid object undergoing planar motion, using noisy planar position measurements of the object and depth observations from a mobile robot. The key idea is to evaluate each candidate viewpoint by its expected observation quality over plausible future object states induced by motion and measurement uncertainty, rather than at a single predicted object pose. To obtain this predictive belief, a fixed-lag Gaussian Process smoother estimates and predicts the object state from noisy position measurements. The resulting belief is used to generate candidate viewpoints around the predicted object location, filter them by reachability, and estimate their expected coverage-driven scores. Simulation and real-world experiments demonstrate improved reconstruction completeness over non-predictive NBV and prediction-only tracking methods, bridging coverage-driven active reconstruction and prediction-driven tracking.
>
---
#### [new 028] OrbiSim: World Models as Differentiable Physics Engines for Embodied Intelligence
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出OrbiSim，一种用于具身智能的可微物理引擎，解决传统模拟器在物理建模和策略优化上的不足。**

- **链接: [https://arxiv.org/pdf/2605.16395](https://arxiv.org/pdf/2605.16395)**

> **作者:** Jiajian Li; Jingyuan Huang; Junru Gong; Qi Wang; Xiaokang Yang; Yunbo Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** We present OrbiSim, a novel robotic simulation paradigm that redefines world models as a fully differentiable physics engine for embodied intelligence. Unlike prior world models that focus on unconstrained imagination in latent or visual domains, OrbiSim establishes a unified, physically-grounded pathway that bridges structured scene assets, neural dynamics, and downstream reinforcement learning. By enabling end-to-end differentiability throughout the entire simulation loop -- spanning from explicit state transitions to visual observation generation -- OrbiSim supports tasks traditionally intractable for classical simulators, such as differentiable contact modeling, gradient-based policy optimization under sparse rewards, and intuitive physical inference. Empirical results demonstrate that OrbiSim significantly outperforms state-of-the-art world models in both predictive fidelity and control performance. Furthermore, its consistent responsiveness to asset configurations and physical parameters suggests its potential as a differentiable tool for enhancing robot simulation and policy training.
>
---
#### [new 029] MR-SLAM: Immersive Spatial Supervision for Multi-Robot Mapping via Mixed Reality
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文提出MR-SLAM，解决多机器人SLAM中操作员空间感知困难的问题。通过混合现实技术实现多机器人地图同步与监控。**

- **链接: [https://arxiv.org/pdf/2605.16432](https://arxiv.org/pdf/2605.16432)**

> **作者:** Prakash Aryan; Cem Erdogdu; Kavinaya Kumarchokkappan; Timo Kehrer; Sebastiano Panichella
>
> **备注:** Accepted to ICRA 2026 Workshop "MM-SpatialAI Workshop: Multi-Modal Spatial AI for Robust Navigation and Open-World Understanding"
>
> **摘要:** Operating a multi-robot fleet for simultaneous localization and mapping (SLAM) in applications such as building inspection or warehouse-aisle monitoring requires the operator to maintain spatial awareness of each robot's position and mapping state, a task that scales poorly on conventional 2D interfaces. We present MR-SLAM, a mixed reality (MR) system in which an operator wearing a Meta Quest 3 headset teleoperates three simulated TurtleBot3 robots through a passthrough view with real-world occlusion, while spatially anchored dashboard panels report mapping progress in situ. Each robot runs an independent SLAM Toolbox instance whose occupancy grid is merged in real time on a Robot Operating System 2 (ROS 2) back end. Across five 9-minute evaluation sessions, the system delivered scans at 8.83 +/- 0.16 Hz, mapped 17.9 +/- 0.8 m^2 of merged occupancy, and reached 94.7 +/- 0.5% cross-instance occupancy consistency across robot pairs. An additional session recorded 6.3 ms median transform jitter and 26.7 m^2 coverage of a 41 m^2 grid. We position MR-SLAM as a reference implementation for combining passthrough mixed reality supervision with multi-robot SLAM on consumer hardware.
>
---
#### [new 030] MORN: Metacognitive Object-Goal Regulation for Resource-Rational Long-Horizon Navigation
- **分类: cs.RO**

- **简介: 该论文属于长期目标导航任务，解决机器人在资源受限下因局部决策导致资源浪费的问题。提出MORN架构，通过元认知机制动态调节任务进度，提升导航效率。**

- **链接: [https://arxiv.org/pdf/2605.16932](https://arxiv.org/pdf/2605.16932)**

> **作者:** Xi Lin; Jiayi Li; Kangyi Wu; Jiaqiao Tang; Qingrong He; Lin Zhao
>
> **摘要:** Robots deployed in unstructured human environments must frequently execute long-horizon missions, such as find the mug, then the chair, then the printer, under strict operational constraints. While contemporary zero-shot Object Navigation (ObjectNav) agents leverage Vision-Language Models (VLMs) to effectively localize semantic targets, they operate as purely reactive systems that inherently lack global resource awareness. Consequently, these agents inadvertently exhaust critical budgets, including time and battery, on infeasible subgoals due to partial observability, failing to balance local exploration with global mission viability. To bridge this gap by injecting resource-rationality into the navigation loop, we present MORN (Metacognitive Object-goal Regulation Navigation), an executive architecture inspired by Dual-Process Theory in cognitive science. MORN augments frozen navigation backbones with a System 2 meta-controller that continuously monitors the System 1 locomotor. By formalizing three neuro-cognitive states, Potentiality Index, Persistence Gating, and Evidence Accumulation, MORN dynamically regulates the mission schedule based on online estimates of progress velocity and perceptual uncertainty. This mechanism effectively neutralizes the Sunk Cost Fallacy, enabling agents to abort zombie goals early and decisively commit to achievable ones. Extensive experiments on the HM3D dataset demonstrate that MORN improves Goal Completion Rate (CR) from 0.23 to 0.30 and reduces Wasted Step Fraction (WSF) from 0.90 to 0.70, establishing that in resource-constrained autonomy, the metacognitive awareness of global resources is as critical as the reactive ability to navigate.
>
---
#### [new 031] Contrastive Conceptor Activation Steering (COAST): Unlocking Vision-Language-Action Models through Hidden States
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出COAST方法，解决VLA模型在机器人任务中表现脆弱的问题。通过概念器激活引导，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.17144](https://arxiv.org/pdf/2605.17144)**

> **作者:** Miranda Muqing Miao; Subin Kim; Brandon Yang; Lyle Ungar
>
> **备注:** Submitted to NeurIPS 2026
>
> **摘要:** Vision-Language-Action (VLA) models leverage powerful perceptual priors from web-scale Vision-Language Model (VLM) pre-training, yet they remain surprisingly brittle in practice, frequently failing at simple robotic tasks. To mitigate this, we propose Contrastive Conceptor Activation Steering (COAST). COAST builds on the notion of a "conceptor", a linear operator that soft-projects data into the principal components of a target distribution. COAST uses conceptors to identify success-critical subspaces for a target robotic task from a few examples of success and failure rollouts. At inference time, it steers VLA latents into these identified success subspaces to improve task outcomes. Across three architecturally distinct neural policies (flow-matching VLA, autoregressive VLA, and Diffusion Policy), COAST improves absolute mean simulation and real-robot task success rate by over 20 and 40% respectively. The activation subspace geometry reveals that failure modes share substantial structure across tasks while success representations remain largely task-specific. When tasks share similar failure modes, this structure enables previously fitted conceptors to improve performance on new tasks without refitting. Ultimately, our results suggest that current VLAs retain substantial task-relevant knowledge in their latent representations, and that the action expert's decoding bottleneck could be mitigated by steering its residual stream toward task-relevant subspaces. COAST provides a lightweight, training-free path to unlocking these latent capabilities by steering the model towards its own "success" distributions.
>
---
#### [new 032] AffordVLA: Injecting Affordance Representations into Vision-Language-Action Models via Implicit Feature Alignment
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型视觉表示不足的问题。通过隐式对齐注入操作性感知，提升模型在非结构化环境中的鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.17517](https://arxiv.org/pdf/2605.17517)**

> **作者:** Weijie Kong; Zhian Su; Wei Yu; Huixu Dong
>
> **备注:** 13pages, 10figures
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models have shown strong potential for general-purpose robotic manipulation. However, the visual representations of most VLA models are often dominated by global object appearance and struggle to focus on task-relevant functional interaction regions, which limits their robustness in unstructured environments. Existing affordance-based methods typically rely on explicit mask injection or external perception modules, requiring additional annotations while introducing cascading perception errors and inference overhead. To address these limitations, we propose AffordVLA, an affordance-enhanced VLA framework that internalizes manipulation-centric affordance perception into VLA visual representations through implicit representation alignment. Specifically, we construct a zero-shot affordance teacher to extract task-conditioned affordance visual representations from RGB observations and language instructions. AffordVLA aligns the intermediate visual representations of the VLA with the affordance visual representations extracted by the teacher, thereby implicitly injecting manipulation-centric affordance perception into VLA visual representations and improving action accuracy. Extensive simulation and real-world experiments demonstrate that AffordVLA and its affordance teacher achieve state-of-the-art performance and outperform strong baselines. Ablation analyses show that AffordVLA effectively reshapes VLA visual representations while preserving inference efficiency, leading to improved manipulation success rates and training efficiency.
>
---
#### [new 033] MUSE: Multimodal Uncertainty Quantification of State Estimation
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决视觉-惯性里程计中不确定性量化问题。提出MUSE框架，利用Mamba模型实现多传感器实时不确定性估计。**

- **链接: [https://arxiv.org/pdf/2605.17421](https://arxiv.org/pdf/2605.17421)**

> **作者:** Minkyung Kim; Henry Che; Bhargav Chandaka; Bhumsitt Pramuanpornsatid; Chengyu Yang; Sheng Cheng; Xiaofeng Wang; Naira Hovakimyan; Shenlong Wang
>
> **备注:** Code and dataset: this https URL
>
> **摘要:** Accurate visual state estimation has been a central topic in robotics with a wide range of applications in robot navigation, autonomous driving, and autonomous flight. Recent advances in robot perception have led to significant improvements in the accuracy and robustness of state estimation, yet a fundamental challenge remains in how to quantify and calibrate its precision, i.e., how confident we are in an estimate and whether failures can be detected. This issue is particularly pronounced in visual-inertial odometry (VIO), where the heteroscedastic and multimodal nature of the problem makes uncertainty quantification especially difficult. This paper introduces MUSE (Multimodal Uncertainty Quantification of State Estimation), a novel real-time learning-based framework that leverages the strong and efficient sequential modeling capacity of Mamba to estimate localization uncertainty from multiple asynchronous sensor streams. Experiments on both public and in-house datasets demonstrate that MUSE achieves superior reliability and robustness compared to existing uncertainty quantification methods, and ablation studies justify the benefits of its key design choices.
>
---
#### [new 034] On Improving Multimodal Pedestrian Trajectory Prediction with CVAE: A Study on Benchmark and Robot Data
- **分类: cs.RO**

- **简介: 该论文属于行人轨迹预测任务，旨在提升多模态轨迹预测的准确性与多样性。通过引入CVAE改进Social-STGCNN模型，在真实数据中验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2605.18262](https://arxiv.org/pdf/2605.18262)**

> **作者:** Yuzhou Liu; Cristina Olaverri-Monreal
>
> **摘要:** Accurate pedestrian trajectory prediction is crucial for autonomous systems operating in complex environments, such as modular buses and delivery robots in suburban or semi-structured areas. Social Spatio-Temporal Graph Convolutional Neural Networks (Social-STGCNN) have shown strong performance by modeling social interactions; however, producing diverse and well-calibrated future trajectories remains challenging. In this work, we build on a Social-STGCNN backbone and introduce a Conditional Variational Autoencoder (CVAE)-based probabilistic formulation to explicitly model multimodal future trajectories. We evaluate the method on the ETH and UCY pedestrian trajectory datasets as well as on a real-world pedestrian dataset collected by a mobile robot. Results show moderate gains on public benchmarks, but more consistent endpoint accuracy and improved trajectory diversity across different crowd configurations. Evaluation on robot-collected data further demonstrates the approach's effectiveness beyond curated benchmarks and supports its applicability in practical deployments.
>
---
#### [new 035] Bayesian Networks for Path-Based Sensors: Gathering Information and Path Planning in Communication Denied Environments
- **分类: cs.RO**

- **简介: 论文研究在通信受限环境下，利用路径传感器进行环境感知与路径规划。任务是通过贝叶斯网络更新信念图，提升信息获取效率，解决路径规划中信息不足的问题。**

- **链接: [https://arxiv.org/pdf/2605.16673](https://arxiv.org/pdf/2605.16673)**

> **作者:** Alkesh K. Srivastava; George P. Kontoudis; Donald Sofge; Michael Otte
>
> **备注:** This paper has been accepted for presentation at 17th World Symposium on the Algorithmic Foundations of Robotics (WAFR 2026)
>
> **摘要:** A "path-based sensor" produces a single observation along a continuous path. For example, a boolean path-based sensor returns a single "1" if an event of interest is detected at any point along the path and a "0" otherwise. Notably, a "1" provides no direct information about where along the path the event(s) may have occurred. Previous work has demonstrated that observations from multiple path-based sensors can be fused to create a Bayesian belief map over the spatial locations of the underlying event or phenomenon. Moreover, path planning can employ Shannon information theory to accelerate the rate of convergence of the belief map. In this paper, we present a new method to update the belief map based on a path-based sensor observation, and then plan paths to increase information gain. In contrast to prior work that approximates the posterior by averaging over the alternative event histories, we introduce a Bayesian Network (BN) formulation that models the probabilistic relationships between the latent variables and path-based sensor measurements, enabling a more principled Bayesian belief update. We consider static hazard detection in a communication-denied environment as a representative problem setting. The event of a robot returning from its path corresponds to a path-based hazard sensor reading of "0" (hazard not detected), while a robot failing to return corresponds to a reading of "1" (hazard detected). We consider false positives and false negatives. We find that the new method leads to quicker convergence of the belief map than prior work in both single- and multi-robot cases.
>
---
#### [new 036] SEDualVLN: A Spatially-Enhanced Dual-System for Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文提出SEDualVLN，解决视觉语言导航（VLN）中的空间感知与动态推理问题。通过双系统架构提升导航性能。**

- **链接: [https://arxiv.org/pdf/2605.17249](https://arxiv.org/pdf/2605.17249)**

> **作者:** Jingzhi Huang; Junkai Huang; Wenxuan Song; Haoyang Yang; Hailong Huang; Haoang Li; Yi Wang
>
> **摘要:** Vision-Language Navigation (VLN) approaches have currently followed two primary paradigms: the end-to-end Vision-Language Model (VLM) policy fine-tuned on navigation trajectories to directly predict actions, and the zero-shot modular pipeline integrating pre-trained Multimodal Large Language Model (MLLM) for training-free generalization to unseen environments. However, end-to-end methods struggle with long-horizon navigation and lack dynamic reasoning, whereas zero-shot methods are constrained by limited spatial grounding for reliable planning and also require substantial reasoning time. To bridge this gap, we introduce SEDualVLN, a spatially-enhanced dual-system VLN framework. System 1 is a VLM model enhanced with both global and local spatial awareness, used for action generation. System 2 integrates a general MLLM with a mapping module, wherein the MLLM plans waypoints by leveraging top-down views of the real-time 3D map alongside streams of rendered path images. Both systems leverage different forms of spatial enhancement to cultivate the agent's sense of direction in VLN tasks. Ultimately, they cooperate to complete the navigation task through a fast-slow coordinated approach. SEDualVLN achieves state-of-the-art performance on VLN-CE benchmarks, and further ablation studies demonstrate the effectiveness of each system and module.
>
---
#### [new 037] HCLM: A Hierarchical Framework for Cooperative Loco-Manipulation with Dual Quadrupeds
- **分类: cs.RO**

- **简介: 该论文提出HCLM框架，解决双四足机器人协作操作中的协调问题。通过分层架构实现空间协同与运动执行的解耦，提升任务执行的鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2605.17300](https://arxiv.org/pdf/2605.17300)**

> **作者:** Qixuan Li; Chen Le; Jincheng Yu; Xinlei Chen
>
> **摘要:** We introduce HCLM, a hierarchical framework for general-purpose cooperative loco-manipulation with dual quadrupedal systems. Coordinating multi-robot collaborative manipulation across floating bases is highly challenging due to the conflicting demands of spatial coordination, robust locomotion, and closed-chain physical interactions. To resolve this, our architecture systematically decouples high-level collaborative reasoning from low-level robust motion execution. At the high level, a centralized Joint Diffusion Policy leverages an SE(3)-invariant task-space representation to learn coordinate-agnostic spatial coordination patterns. To translate these frame-agnostic references into physical motion, a task-centric hybrid Whole-Body Controller synergizes a proactive kinematic Model Predictive Control for collision-free velocity distribution with a reactive execution layer. Crucially, this reactive layer guarantees rapid responsiveness for precise end-effector tracking, while concurrently integrating active force regulation via a cooperative admittance scheme to safely resolve kinematic conflicts and strictly regulate internal stresses during closed-chain interactions. We validate the framework across progressively challenging simulated scenarios, including cooperative carrying, packing and handovers, and successfully deploy the latter in the real world. The results demonstrate reliable task execution, strict configuration agnosticism, and exceptional resilience against severe physical perturbations, offering a highly robust pathway for multi-robot embodied coordination.
>
---
#### [new 038] Fixed External Cameras as Common Prior Maps for Active 3D Scene Graph Generation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于3D场景图生成任务，解决机器人在未知环境中高效构建场景图的问题。通过引入固定外置摄像头作为先验地图，提升初始物体召回率并优化后续探索效率。**

- **链接: [https://arxiv.org/pdf/2605.18184](https://arxiv.org/pdf/2605.18184)**

> **作者:** Giorgia Modi; Davide Buoso; Giuseppe Averta; Daniele De Martini
>
> **摘要:** Commonly available prior information, such as BIM models, floor plans, and remote sensing images, can provide valuable geometric and semantic context for autonomous robotic systems. In this paper, we treat observations from fixed external RGB cameras as Common Prior Maps (CPMs): wide-field views of the environment that initialize a semantic and geometric scene prior before any robot motion begins. We present an RGB-only framework for active, incremental 3D scene graph (3DSG) generation that seamlessly fuses observations from both onboard robot cameras and fixed external cameras within a single hardware-agnostic pipeline. By relying solely on RGB observations processed by a feed-forward 3D reconstruction model, the system treats all cameras - onboard or external - identically, requiring no hardware modifications. A graph-based active semantic exploration framework then directly leverages the partial scene graph to guide the robot toward regions of high semantic uncertainty, progressively completing and refining the prior. Experiments demonstrate that bootstrapping the scene graph with even a single external camera increases initial object recall by up to +79%, and that the richer context of the prior significantly improves the efficiency of subsequent active exploration.
>
---
#### [new 039] RoboFlow4D: A Lightweight Flow World Model Toward Real-Time Flow-Guided Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出RoboFlow4D，解决实时3D机器人操作中的规划与执行问题。通过统一感知与规划，实现高效流引导的机械臂控制。**

- **链接: [https://arxiv.org/pdf/2605.17522](https://arxiv.org/pdf/2605.17522)**

> **作者:** Sixu Lin; Junliang Chen; Huaiyuan Xu; Zhuohao Li; Guangming Wang; Yixiong Jing; Sheng Xu; Runyi Zhao; Brian Sheil; Lap-Pui Chau; Guiliang Liu
>
> **摘要:** Planning and acting in 3D environments is a fundamental capability for robotic manipulation in the real world. Although prior work has explored predictive flow planners to guide 3D manipulation, existing approaches often rely on modular pipelines stacking multiple submodels, resulting in high computational overhead and limited real-time performance. To address these challenges, we introduce RoboFlow4D, a lightweight flow world model that unifies perception and planning by estimating temporal motion in physical 3D space. As an end-to-end framework, RoboFlow4D directly predicts multi-frame 3D flows from visual observations and textual instructions, providing explicit flow-based planning to guide action generation. This design allows seamless integration with general action policies, forming an efficient observation-planning-execution closed loop. Through slow-fast collaboration between flow prediction and action control, RoboFlow4D enables real-time and resource-efficient manipulation. Extensive experiments in both simulation and real-world settings demonstrate that RoboFlow4D consistently improves manipulation success rates and computational efficiency, advancing flow-guided planning for embodied intelligence.
>
---
#### [new 040] Key-Gram: Extensible World Knowledge for Embodied Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Key-Gram，解决具身控制中语言与视觉耦合导致的模态竞争问题。通过分离语言知识与视觉推理，提升任务执行效果。属于具身操控任务。**

- **链接: [https://arxiv.org/pdf/2605.18556](https://arxiv.org/pdf/2605.18556)**

> **作者:** Jingjing Fan; Siyuan Li; Botao Ren; Zhidong Deng
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Embodied control increasingly requires models to follow compositional language instructions while reasoning over dynamic visual states. However, current vision-language-action policies and world-action models often couple linguistic knowledge with visual computation in a shared backbone or conditioning pathway, leading to modality competition and making knowledge extension dependent on backbone updates. In this paper, we introduce Key-Gram, a conditional-memory framework that separates language-derived world knowledge from visual-state reasoning for embodied control. At its core is a memory module that decomposes an instruction into task-specific key-grams, retrieves static linguistic priors through deterministic hashed lookup, and injects the retrieved entries into selected hidden layers through context-aware gating and lightweight convolutional fusion. This design allows the backbone to devote its main capacity to visual reasoning and action inference, while reusable instruction knowledge is stored in an extensible external memory. The logical memory table can be conveniently partitioned during training and, due to its $O(1)$ lookup pattern, efficiently placed on host memory during inference. Across RoboTwin2.0, LIBERO/LIBERO-Plus, and real-world dual-arm manipulation, Key-Gram consistently improves both $\pi_{0}$ and $\pi_{0.5}$ backbones, with average relative gains of $29.5\%/9.9\%$ on RoboTwin2.0, $35.8\%/4.5\%$ on LIBERO-Plus transfer without target-domain fine-tuning, and $15.4\%/8.1\%$ on real-world long-horizon tasks. These results demonstrate that externalized linguistic memory provides an effective and extensible mechanism for improving compositional grounding, transfer, and real-world manipulation.
>
---
#### [new 041] Hierarchical Two-Stage Framework for Environment-Aware Long-Horizon Vessel Trajectory Prediction
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于船舶轨迹预测任务，旨在解决长时距预测中环境因素影响和时间依赖性问题。提出分层两阶段框架，结合长期与短期预测模块，提升预测精度。**

- **链接: [https://arxiv.org/pdf/2605.16442](https://arxiv.org/pdf/2605.16442)**

> **作者:** Ganeshaaraj Gnanavel; Tharindu Fernando; Sridha Sridharan; Clinton Fookes
>
> **摘要:** Long-horizon vessel trajectory forecasting under real ocean conditions is critical for collision avoidance, traffic management, and route planning. However, achieving accurate predictions is challenging due to long-range temporal dependencies and dynamic environmental factors such as currents, wind, and waves. To address these issues, we propose a hierarchical two-stage framework that combines a coarse long-term predictor with a grid-aware short-term predictor through a hierarchical fusion mechanism. The short-term branch leverages a Spatio-Temporal Graph Transformer on discretized maritime cells to capture localized dynamics, while the long-term branch encodes overarching navigational intent. An integrated environmental module incorporates oceanographic parameters, including surface currents, wind vectors, and significant wave height, using cross-modal attention and feature-wise modulation for adaptive response to varying sea conditions. Additionally, a learnable Savitzky-Golay smoothing layer enhances temporal coherence in fused trajectories. We evaluate our approach on Australian Craft Tracking System (CTS) data from the North West region, aligned with Copernicus Marine Service products, using a 3-hour input and a 10-hour prediction horizon. Experimental results show that our framework outperforms the state-of-the-art by 25% in Average Displacement Error (ADE) and 17% in Final Displacement Error (FDE). Ablation studies further validate the contribution of each component.
>
---
#### [new 042] Dynamic robotic cloth folding with efficient Koopman operator-based model predictive control
- **分类: cs.RO; cs.LG; math.DS; math.OC**

- **简介: 该论文属于机器人动态布料折叠任务，旨在解决快速折叠中布料动力学复杂导致的轨迹规划难题。通过结合Koopman算子回归与模型预测控制，实现高效准确的折叠轨迹生成。**

- **链接: [https://arxiv.org/pdf/2605.18373](https://arxiv.org/pdf/2605.18373)**

> **作者:** Edoardo Caldarelli; Franco Coltraro; Adrià Colomé; Lorenzo Rosasco; Carme Torras
>
> **备注:** Accepted for presentation at the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Robotic cloth folding is a challenging task, particularly when considering dynamic folding tasks, which aim at folding cloth by fast motions that leverage its dynamics. When subject to such fast motions, the complexity of cloth dynamics hinders both system identification and planning of folding trajectories, resulting in a difficult simulation-to-reality transfer when using physical models of cloth. Compared to the dexterity that humans exhibit when performing folding tasks, robotic approaches usually employ small garments with quite rigid dynamics, and are either too slow, or fast but imprecise, requiring several attempts to achieve a reasonably good fold. In this paper, we tackle these challenges by generating fast folding trajectories with a novel model predictive controller, integrating physics-based simulation of cloth dynamics and efficient, kernel-based Koopman operator regression. Koopman operator regression, an increasingly popular machine learning technique for nonlinear system identification, is used to obtain a linear model for the cloth being folded. Such a surrogate model, trained with data from a high-fidelity, physics-based cloth simulator, can then be employed within a suitable model predictive control algorithm, in place of the costly, nonlinear one, to efficiently generate folding trajectories to be executed by a robotic manipulator. Both in simulated and real-robot experiments, we show how the linearization supplied by the Koopman operator-based model can be employed to efficiently generate fast folding trajectories to unseen poses, without sacrificing folding accuracy.
>
---
#### [new 043] No Plan, Yet Human: A Reactive Robotics Model Predicts Human Planning Failures on a Clinical Task
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于认知任务研究，旨在理解人类规划困难。通过应用AICON模型，模拟人类在塔罗伦敦测试中的表现，揭示规划能力下降时的行为模式。**

- **链接: [https://arxiv.org/pdf/2605.16514](https://arxiv.org/pdf/2605.16514)**

> **作者:** Michael Migacev; Vito Mengers; Antonia Köngeter; Oliver Brock
>
> **摘要:** Understanding why some sequential planning problems are harder than others requires models that go beyond average performance. They should capture the specific pattern of which problems are hard, and ideally fail in the same way people do when planning capacity is reduced. We apply AICON, a reactive gradient-descent framework developed for robotic manipulation, to the Tower of London test, a cognitive test used to assess planning in Parkinson's disease, mild cognitive impairment, and stroke. Without any lookahead planning or knowledge of human cognition, AICON reproduces the fine-grained human difficulty ordering across 24 problems better than structural task parameters and generalizes to held-out problems in a leave-two-out evaluation. Crucially, AICON outperforms a planning baseline for groups with reduced planning capacity while the planning baseline better captures healthy controls. This dissociation was predicted by the original AICON paper, which noted that the model's failure modes resemble those of Parkinson's patients who struggle with goal hierarchies but not move counts. This suggests that as planning capacity is reduced, human behavior shifts toward the reactive mode AICON models. The finding extends a broader pattern: AICON, originally built for robotics, now captures aspects of biological behavior across perception, eye movements, and sequential planning, suggesting its core abstraction reflects something real about how biological systems are organized.
>
---
#### [new 044] Active Defense Against False Data Injection Attacks in Robotic Manipulators
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全任务，解决FDIA导致的传感器信号篡改问题，提出两种防御方法以提升机械臂的抗攻击能力。**

- **链接: [https://arxiv.org/pdf/2605.17950](https://arxiv.org/pdf/2605.17950)**

> **作者:** Gabriele Gualandi; Carl Mikael Larsson; Alessandro V. Papadopoulos
>
> **备注:** Extended 8-page version containing full proofs. An abridged 6-page version has been accepted for publication in the Proceedings of the 23rd IFAC World Congress (2026)
>
> **摘要:** Robotic systems are vulnerable to False Data Injection Attacks (FDIAs), where adversaries corrupt sensor signals to gain malicious control. Feedback linearization exposes robotic systems to integrator vulnerability, making them susceptible to stealthy attacks that can cause significant deviations in end-effector behavior without raising alarms. This paper addresses the resilience of manipulators against finite-horizon FDIAs by formalizing two defense methods, namely anomaly-aware virtual damping and manipulability reduction, with probabilistic guarantees on nominal task execution. Simulations on a 7-DOF redundant manipulator show that the proposed defenses substantially reduce the impact of FDIA compared to using solely a threshold-based ADS like the Chi-squared, while preserving nominal task performance in the absence of attack.
>
---
#### [new 045] Generalizable and Actionable Parts Pose Estimation with Symmetry Annotation-Free Learning Strategy
- **分类: cs.RO**

- **简介: 该论文属于物体部件姿态估计任务，解决数据匮乏场景下的对称性问题。提出SAFAG框架，无需对称标注，通过自监督学习提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.17033](https://arxiv.org/pdf/2605.17033)**

> **作者:** Wenxiao Chen; Xueyu Yuan; Liu Liu; Di Wu; Dan Guo
>
> **备注:** Accepted as a poster at the Forty-third International Conference on Machine Learning (ICML 2026)
>
> **摘要:** Urgently needed generalizable robot object interaction and manipulation requires high-quality Cross-Category object perception. As a pioneer of this area, Generalizable and Actionable Parts (GAParts) understanding has attracted increasing attention from relevant researchers. However, most recent works either have insufficient design regarding the symmetry issue or require rich symmetry annotation, which severely impedes precise GAPart pose estimation in data-lacking scenarios. In this paper, we propose SAFAG, a novel Symmetry Annotation-Free framework for Generalizable and Actionable Parts Pose Estimation. Specifically, we suggest a stepwise refinement two-stage framework for candidate-to-final quaternion regression, and tackle the symmetry prediction as a probability distribution problem with self-supervised learning strategy. The experimental results demonstrate the superior performance and robustness of our SAFAG. We believe that our work has the enormous potential to be applied in many areas of embodied AI system.
>
---
#### [new 046] NORM-Nav: Zero-Shot Mobile Robot Navigation with Natural Language Behavioral Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决传统方法忽视社会行为规范的问题。通过整合自然语言约束，生成符合人类行为习惯的导航路径。**

- **链接: [https://arxiv.org/pdf/2605.16979](https://arxiv.org/pdf/2605.16979)**

> **作者:** Dongjie Huo; Junhui Wang; Chao Gao; Yan Qiao; Dong Zhang; Guyue Zhou
>
> **摘要:** Mobile robots operating in human-centered environments must generate not only collision-free paths but also trajectories that follow local behavioral conventions. Conventional costmap-based navigation emphasizes geometric feasibility and often overlooks such requirements, which can result in socially inappropriate behaviors. This paper presents NORM-Nav, a zero-shot framework that integrates natural language behavioral constraints into costmap-based planning. An LLM parses each instruction into structured constraints and grounds them using real-time vision--LiDAR perception. These constraints are encoded as multi-layer costmaps that represent geometric, semantic, directional, and velocity cues and are directly compatible with standard grid-based planners. Simulation and real-world experiments indicate that NORM-Nav improves task success rates and produces trajectories closer to human references than representative baselines. The project website is available at this https URL.
>
---
#### [new 047] A Mechanistic Model for Collective Motion from Sensorimotor Regularities
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于群体行为建模任务，旨在解决集体行为生成机制问题。通过构建基于传感器和运动的机械模型，解释动物群体行为的产生过程。**

- **链接: [https://arxiv.org/pdf/2605.16522](https://arxiv.org/pdf/2605.16522)**

> **作者:** Vito Mengers; Bao Duc Cao; Oliver Brock
>
> **摘要:** Collective behavior in animals has long been modeled through self-propelled particle models, which reproduce striking group-level phenomena through abstract interaction forces. Yet these models are fundamentally descriptive: they leave open the question of how collective behavior is actually produced. Recent empirical work makes this gap concrete: locusts do not align with neighbors, sensory and cognitive mechanisms mediate interaction instead. A mechanistic model must therefore operate at the sensorimotor level, grounded in what individual organisms can actually perceive, estimate, and physically execute. We present such a model based on a modeling framework from robotics, extended here to collective motion. Each agent perceives neighbors through bearing and apparent-size cues within a limited field of view, maintains uncertain internal state estimates, and selects actions through gradient descent on a desired social distance -- without any prescribed interaction forces. This simple model produces diverse collective behaviors including polarized motion, milling, ring formations, and subgroup fragmentation. A global sensitivity analysis shows that behavioral transitions are governed by sensorimotor parameters corresponding to measurable biological quantities: field of view geometry, sensory noise, turning agility, and memory. Collective behavior can therefore be understood as the emergent outcome of interacting sensorimotor regularities, and differences across species as the emergent outcome of differences in embodiment and environment.
>
---
#### [new 048] Tactile-based Multimodal Fusion in Embodied Intelligence: A Survey of Vision, Language, and Contact-Driven Paradigms
- **分类: cs.RO; cs.CV; eess.SP**

- **简介: 本文综述多模态触觉融合研究，旨在解决 embodied intelligence 中感知与语义理解的结合问题。工作包括分类数据集、方法，并探讨挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2605.17336](https://arxiv.org/pdf/2605.17336)**

> **作者:** Zhixiang Cao; Di Tian; Runwei Guan; Yanzhou Mu; Xiaolou Sun; Shaofeng Liang; Daizong Liu; Tao Huang; Yutao Yue; Henghui Ding; Bin Fang; Alex Zhou; Qing-Long Han; Hui Xiong
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Tactile sensing is a fundamental modality for embodied intelligence, offering unique and direct feedback on contact geometry, material properties, and interaction dynamics that remote sensors cannot replace. However, unimodal tactile perception is inherently limited by its sparse spatial coverage and lack of global semantic context. With the recent explosion in deep learning and large language models, integrating tactile with vision and language has become essential to bridge physical interaction with semantic reasoning, leading to the emergence of Multimodal Tactile Fusion. Despite rapid progress, the existing researches remain fragmented across disparate datasets, sensing modalities, and tasks, lacking a unified theoretical framework. To address this gap, this paper provides a comprehensive survey of multimodal tactile fusion research up to the first quarter of 2026. We propose a hierarchical taxonomy that organizes the field into two primary dimensions: multimodal datasets and multimodal methods. On the data side, we categorize resources ranging from Tactile-Vision datasets, Tactile-Language datasets, Tactile-Vision-Language datasets, and Tactile-Vision-Other datasets. On the method side, we structure prior work into three core pillars: (1) Multimodal Perception and Recognition, which focuses on object understanding and grasp prediction; (2) Cross-Modal Generation, focusing on bidirectional translation between tactile, vision, and text; and (3) Multimodal Interaction, emphasizing feedback control and language-guided manipulation. Furthermore, we summarize representative tactile sensing hardware, review commonly used evaluation metrics and benchmark settings, and discuss current challenges and promising future directions.
>
---
#### [new 049] Event-Grounded Sparse Autoencoders for Vision-Language-Action Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作策略任务，旨在提升VLA模型的可解释性。通过事件对齐的稀疏自编码器分析，将特征与行为事件关联，解决传统方法难以直接评估动作输出的问题。**

- **链接: [https://arxiv.org/pdf/2605.17204](https://arxiv.org/pdf/2605.17204)**

> **作者:** Xinchen Jin; Aditya Chatterjee; Pranav Kumar; Rohan Paleja
>
> **摘要:** Vision-Language-Action (VLA) policies translate language and visual inputs into robot actions, where their hidden representations directly shape closed-loop behavior. However, mechanistic interpretability tools from language and vision-language models do not transfer cleanly to VLAs: outputs are robot actions rather than human-readable tokens, and interventions can only be tested via expensive closed-loop rollouts. We propose an event-grounded interpretability pipeline that anchors SAE feature analysis to behavioral events rather than text contexts. End-effector keyframes are clustered within each task using visual, state, and temporal cues, linking SAE features to behaviorally salient events and, via optional VLM annotations, to semantic context. To our knowledge, our pipeline is among the first to ground SAE-based VLA analysis in closed-loop behavioral events. Across two simulation architectures and a real-robot study, event-grounded ranking yields the strongest causal effects on OpenVLA and transfers to the continuous action chunks of $\pi_{0.5}$. SAE is a sparse but imperfect intervention basis: usability varies with architecture and intervention site, and aggressive intervention reveals safety and interpretability limits. Overall, event-grounded SAE analysis emerges as a practical starting point for behavior-anchored VLA interpretability, motivating future work on SAE features beyond action-aligned coordinates, finer-grained closed-loop evaluation, and safe interventions for high-stakes VLA deployments. Code is available at \url{this https URL}.
>
---
#### [new 050] Bench2Drive-Robust: Benchmarking Closed-Loop Autonomous Driving under Deployment Perturbations
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决闭环系统在部署扰动下的鲁棒性问题。通过构建Bench2Drive-Robust基准，评估摄像头故障、状态估计误差和计算延迟等系统级扰动的影响。**

- **链接: [https://arxiv.org/pdf/2605.18059](https://arxiv.org/pdf/2605.18059)**

> **作者:** Zhiyuan Zhang; Zhenghao Jin; Yanlun Peng; Xianda Guo; Haoran Liu; Shaofeng Zhang; Xingjun Ma; Zuxuan Wu; Junchi Yan; Xiaosong Jia; Yu-Gang Jiang
>
> **摘要:** Robustness is a critical requirement for deploying autonomous driving systems in the real world. Existing robustness benchmarks for autonomous driving have made important progress in studying the effects of image-level corruptions, such as adverse weather or camera degradation, on perception modules and open-loop planning outputs. However, deployment can also involve system-level imperfections, such as inference latency and ego-state estimation errors, which remain less studied in closed-loop E2E-AD evaluation. These imperfections can accumulate through the feedback loop and destabilize control. In this work, we present Bench2Drive-Robust, to our knowledge the first device-centric robustness benchmark for closed-loop end-to-end autonomous driving under realistic deployment perturbations. We systematically evaluate deployment-oriented perturbations arising from three major sources: camera-stream failures (frame drop, partial observation), ego-state estimation errors (GPS noise, and speed or odometry errors), and compute-induced control delay (model inference delay). We evaluate representative end-to-end driving methods and analyze their robustness under different perturbation severities. Our results show that these deployment-related perturbations can substantially degrade closed-loop driving performance, revealing robustness challenges that are not fully captured by conventional image-level corruption evaluations. By establishing a closed-loop evaluation protocol and demonstrating the substantial impact of these deployment-oriented perturbations, Bench2Drive-Robust defines practical robustness problems for end-to-end autonomous driving and encourages further research on deployment-aware robust driving systems.
>
---
#### [new 051] DexHoldem: Playing Texas Hold'em with Dexterous Embodied System
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DexHoldem，用于评估具身系统在德州扑克中的操作能力。任务是解决真实场景下的感知、决策与执行问题，通过基准测试和案例研究验证系统性能。**

- **链接: [https://arxiv.org/pdf/2605.18727](https://arxiv.org/pdf/2605.18727)**

> **作者:** Feng Chen; Tianzhe Chu; Li Sun; Pei Zhou; Zhuxiu Xu; Shenghua Gao; Yuexiang Zhai; Yanchao Yang; Yi Ma
>
> **备注:** 30 Pages
>
> **摘要:** Evaluating embodied systems on real dexterous hardware requires more than isolated primitive skills: an agent must perceive a changing tabletop scene, choose a context-appropriate action, execute it with a dexterous hand, and leave the scene usable for later decisions. We introduce DexHoldem, a real-world system-level benchmark built around Texas Hold'em dexterous manipulation with a ShadowHand. DexHoldem provides 1,470 teleoperated demonstrations across 14 Texas Hold'em manipulation primitives, a standardized physical policy benchmark, and an agentic perception benchmark that tests whether agents can recover the structured game state needed for embodied decision making. On primitive execution, $\pi_{0.5}$ obtains the highest task completion rate ($61.2\%$), while $\pi_{0.5}$ and $\pi_0$ tie on scene-preserving success rate ($47.5\%$). On agentic perception, Opus 4.7 obtains the best strict problem-level accuracy ($34.3\%$), while GPT 5.5 obtains the best average field-wise accuracy ($66.8\%$), exposing a gap between isolated visual sub-capabilities and complete routing-relevant state recovery. Finally, we instantiate the full embodied-agent loop in three case studies, where waiting, recovery dispatches, human-help requests, and repeated primitive execution reveal how perception and policy errors accumulate during closed-loop deployment. DexHoldem therefore evaluates dexterous tabletop execution, agentic perception, and embodied decision routing in a shared physical setting. Project page: this https URL.
>
---
#### [new 052] Learning-Based Adaptive Control for Surgical Robotic Exposure Task on Deformable Tissues
- **分类: cs.RO**

- **简介: 该论文属于手术机器人暴露任务，解决软组织遮挡下的ROI暴露问题。通过学习式自适应控制框架，实现自主抓取与暴露，提升手术辅助能力。**

- **链接: [https://arxiv.org/pdf/2605.17927](https://arxiv.org/pdf/2605.17927)**

> **作者:** Jiayi Liu; Kaiqi Wei; Yiwei Wang; Huan Zhao; Han Ding
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026. 12 pages, 9 figures
>
> **摘要:** In various surgical procedures, regions of interest (ROIs) such as organs or lesions are often occluded by overlying tissues, requiring surgeons to achieve adequate exposure for precise intervention. However, the irregular geometry, nonlinear biomechanical properties of overlying tissues, and limited intraoperative visibility of the ROI pose significant challenges to the autonomous execution of tissue retraction. To address this, we formulate a realistic model of the tissue retraction task and propose a learning-based adaptive control framework for achieving ROI exposure. The method optimizes control inputs online by monitoring changes in the visual boundary of the tissue, while leveraging a deep deformation estimation model trained on simulation data to identify the optimal grasping point and ensure the convergence and safety of the adaptive controller. Through simulations and real-world experiments on different deformable materials, it has been demonstrated that this framework exhibits zero-shot adaptation to similar tasks and can complete the autonomous retraction process, from initial grasp selection to full ROI exposure. Therefore, it has the potential to be applied in actual surgical assistance scenarios.
>
---
#### [new 053] Generating Realistic Safety-Critical Scenarios for Vehicle-Pedestrian Interactions
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶安全验证任务，旨在解决真实高风险场景稀缺与仿真行为不真实的问题。通过三阶段框架生成高保真安全关键场景数据集。**

- **链接: [https://arxiv.org/pdf/2605.17229](https://arxiv.org/pdf/2605.17229)**

> **作者:** Qingwen Pu; Kun Xie; Yuan Zhu; Guocong Zhai
>
> **备注:** 49 pages, 13 figures, 11 table
>
> **摘要:** Automated driving system deployment requires rigorous validation across safety-critical vehicle-pedestrian interactions, yet real-world datasets rarely capture high-risk scenarios while simulation platforms lack realistic behavior. In response, this study proposes a three-stage framework that combines real-world grounding with adaptive simulation to generate behaviorally realistic safety-critical scenarios at scale. Stage 1 pre-trains multi-agent state-space Transformer-enhanced DDPG (MA-SST-DDPG) agents on real-world safety-critical data to learn human-like interactive evasive behaviors through data-driven learning. Stage 2 deploys pre-trained multi-agents in CARLA for online reinforcement learning to generalize across diverse scenarios, integrating real-world knowledge with simulation experience to produce a refined MA-SST-DDPG model. Stage 3 uses CARLA with the refined model to generate over 198,000 high-resolution interaction episodes from eight intersection scenarios, culminating in the Vehicle-Pedestrian Safety-Critical Interaction (VPSCI) dataset. The Refined MA-SST-DDPG model outperformed baseline methods in reproducing realistic evasive behaviors, achieving the lowest trajectory errors (ADE = 0.072 m, FDE = 0.142 m). Statistical comparison confirmed distributional equivalence between the generated and real-world data in both conflict severity and behavioral response. A Turing test confirmed that the three-stage framework generated evasive behaviors were indistinguishable from real-world interactions. These results demonstrate the framework's effectiveness in producing high-fidelity safety-critical data, offering valuable sources for the development of ADS and simulation-based safety evaluations.
>
---
#### [new 054] Virtues of Ordered Chaos: Planning with Topple Actions in Tabletop Stack Rearrangement
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究桌面上堆叠物的重新排列任务，解决如何高效进行物体操作的问题。通过引入翻倒动作，提升规划效率，验证了其在物理仿真中的优势。**

- **链接: [https://arxiv.org/pdf/2605.17815](https://arxiv.org/pdf/2605.17815)**

> **作者:** Hao Lu; Rahul Shome
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Efficient object manipulation strategies have significant impact in automation applications. In this work, the stack rearrangement in tabletop settings is studied, with a focus on augmenting the task planning domain with richer nonprehensile aggregating actions, in particular the toppling of objects from a stack to the table. Toppling can compress long sequences of intermediate relocations. Computed plans need to interleave pick-and-place actions with topple throughout its plan based on the problem. In order to generate the task plan and model an abstraction to compute solutions that include both pick-and-place and topple actions, a novel aggregating gadget for topple is introduced. Using this directed graphical abstraction, candidate task plan computation becomes a variant of the pebble motion problem, treating objects as pebbles. Benchmarks are then reported in a IsaacSim-based physics simulation. Results highlight clear benefits of achieving faster execution than solely using pick-and-place actions. Though this work primarily investigates the topple action, we demonstrate that similar abstractions can model other aggregating actions of interest, like scoop. The current work provides a preliminary, strong indication of the promising benefits of abstractions for rich object interactions in manipulation applications.
>
---
#### [new 055] How to Instruct Your Robot: Dense Language Annotations Power Robot Policy Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决数据收集成本高的问题。通过密集语言标注提升机器人策略学习效果，提出DeMiAn方法优化任务执行。**

- **链接: [https://arxiv.org/pdf/2605.17077](https://arxiv.org/pdf/2605.17077)**

> **作者:** Bosung Kim; Ruiyi Wang; David Acuna; Jaehun Jung; Alexander Trevithick; Brandon Cui; Yejin Choi; Prithviraj Ammanabrolu
>
> **摘要:** Scaling robot policy learning is bottlenecked by the cost of collecting demonstrations, while language annotations for existing demonstrations are comparatively cheap. We study language density as a lever for extracting more signal from a fixed robot or egocentric-video corpus. We introduce DeMiAn (Dense Multi-aspect Annotation), a two-stage approach that first re-labels demonstration segments with VLM-generated annotations along four complementary aspects: physical motion, scene composition, arm pose, and reasoning. A learned instructor then maps a task description and initial scene snapshot to a task-appropriate annotation at deployment, running asynchronously so generation latency is hidden behind policy execution. Across over 1M robot manipulation clips and 50K EgoVerse human-egocentric videos, DeMiAn improves both a vision-language-action policy and a video-based world-action model without collecting new demonstrations. On RoboCasa, the instructor raises success by 5 points over a task-only baseline and comes within 3 points of a per-task oracle. No fixed annotation aspect dominates across tasks, showing that selecting the right dense language matters. DeMiAn also improves composite-task and out-of-distribution performance, and shifts the compute-performance frontier in both mid-training and post-training after accounting for annotation-generation FLOPs. These results position dense re-annotation as a practical scaling lever for robot policy learning.
>
---
#### [new 056] Visual Sculpting: Visually-Aligned Planning Representations for Long-Horizon Robot Clay Sculpting
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人长期规划的陶土雕塑任务，解决变形物体操作中动态建模与视觉对齐的问题。提出一种视觉对齐的动态模型，支持长序列操作和视觉规划。**

- **链接: [https://arxiv.org/pdf/2605.17556](https://arxiv.org/pdf/2605.17556)**

> **作者:** Peter Schaldenbrand; Jean Oh
>
> **备注:** 8 pages, 14 figures. Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Clay sculpting is a nuanced, artistic task involving dexterous manipulation with long-horizon planning to achieve high-level goals. As a robotics problem, we formulate clay sculpting as a shape-to-shape matching challenge. Prior deformable object manipulation work either requires retraining a policy per goal or relies on dynamics models which represent state as sparse point clouds which do not capture important clay features, such as textures, well. We present a method for modeling the dynamics of deformable materials and planning for robotic sculpting in a representation that is visually-aligned, capturing lighting and texture features. With three different deformable materials and various end-effectors, we demonstrate that our dynamics model is comparable in performance to the state-of-the-art with the added benefit of being compatible with visual planning. Our actions are represented as parametrized pushes into clay with a single end-effector, which proved to be suitable for long-horizon (>100 actions) clay relief sculptures. Lastly, we show the benefits of planning in a visually-aligned representation, but also provide analysis providing evidence as to why this representation is challenging to plan in compared to 3D representations.
>
---
#### [new 057] Towards Ubiquitous Mapping and Localization for Dynamic Indoor Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于室内动态环境下的定位与建图任务，旨在解决传统SLAM系统对环境变化敏感的问题。通过部署固定RGB-D相机网络，实现高效、实时的全局地图构建与定位。**

- **链接: [https://arxiv.org/pdf/2605.18385](https://arxiv.org/pdf/2605.18385)**

> **作者:** Halim Djerroud; Nico Steyn; Olivier Rabreau; Patrick Bonnin; Abderraouf Benali
>
> **摘要:** We present UbiSLAM, an innovative solution for real-time mapping and localization in dynamic indoor environments. By deploying a network of fixed RGB-D cameras strategically throughout the workspace, UbiSLAM addresses limitations commonly encountered in traditional SLAM systems, such as sensitivity to environmental changes and reliance on mobile unit sensors. This fixed-sensor approach enables real-time, comprehensive mapping, enhancing the localization accuracy and responsiveness of robots operating within the environment. The centralized map generated by UbiSLAM is continuously updated, providing robots with an accurate global view, which improves navigation, minimizes collisions, and facilitates smoother human-robot interactions in shared spaces. Beyond its advantages, UbiSLAM faces challenges, particularly in ensuring complete spatial coverage and managing blind spots, which necessitate data integration from the robots themselves. In this paper we discuss potential solutions, such as automatic calibration for optimal camera placement and orientation, along with enhanced communication protocols for real-time data sharing. The proposed model reduces the computational load on individual robotic units, allowing less complex robotic platforms to operate effectively while enhancing the robustness of the overall system.
>
---
#### [new 058] LACE: Latent Visual Representation for Cross-Embodiment Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，解决人类与机器人视觉表征不匹配的问题。通过LACE框架，在潜在空间对齐两者视觉表示，提升机器人政策性能。**

- **链接: [https://arxiv.org/pdf/2605.16743](https://arxiv.org/pdf/2605.16743)**

> **作者:** Yoo Sung Jang; Kanchana Ranasinghe; Cristina Mata; Yichi Zhang; Jorge Mendez-Mendez; Michael S. Ryoo
>
> **摘要:** Cross-embodiment learning from human demonstrations is hindered by the visual gap between human and robot embodiments. While self-supervised learning (SSL) backbones encode rich inter-class semantics of general objects, we show they fail to establish correspondence between human and robot hands. We propose LACE, a framework that aligns human and robot visual representations in the latent space of these backbones by leveraging correspondences between shared body parts across embodiments as sparse supervision. These annotations can be automatically obtained via forward kinematics, and single robot demonstration is sufficient to train the model. Our semantic alignment loss matches distributions incurred by corresponding features, lifting patch-level supervision to semantic-level alignment, while a Gram loss preserves pretrained feature quality. This alignment enables robot policies to leverage abundant human data when robot demonstrations are scarce: in zero-shot transfer, policies using LACE-DINO outperform those using DINO by a large margin (65\%), with consistent gains in low-data regimes and out-of-distribution environments.
>
---
#### [new 059] REACT: Environment-Adaptive Architecture for Continuous Formation Navigation of Wheeled Mobile Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多机器人路径规划任务，解决复杂环境中机器人编队导航适应性问题。提出REACT架构，结合生成与维护机制，实现无冲突编队切换和连续导航。**

- **链接: [https://arxiv.org/pdf/2605.18441](https://arxiv.org/pdf/2605.18441)**

> **作者:** Jianghong Dong; Yifeng Zhang; Jiawei Wang; Mengchi Cai; Keqiang Li; Guillaume Sartoretti
>
> **摘要:** Formation control of wheeled mobile robots (WMRs) has been extensively studied due to its broad applications in fields such as logistics transportation, environmental monitoring, and search and rescue. However, most existing works mainly focus on tracking predefined formations, which limits their adaptability to complex real-world environments. To address this, we propose REACT (Real-time Environment-Adaptive architecture for Continuous formation navigaTion), a hierarchical architecture integrating centralized formation generation and distributed formation maintenance. Specifically, our upper layer generates new environment-adaptive formations when necessary and uses our proposed TCF-R2T (Trajectory-Conflict-Free Robot-to-Target assignment) algorithm to compute conflict-free WMR-to-target assignments in polynomial time, enabling timely formation transitions without trajectory conflicts. At the lower layer, each WMR executes our developed JSTP (Joint Spatio-Temporal trajectory Planning) method to maintain the generated formation by simultaneously optimizing spatial positions and temporal durations, thereby enhancing coordination among WMRs and enabling continuous navigation in obstacle-rich environments and dynamic-obstacle scenarios. Both simulation and real-world experiments validate the effectiveness and practical applicability of REACT. Experimental videos are available on our project website: this https URL.
>
---
#### [new 060] TacSE3: Equivariant SE(3) Motion Estimation from Low-Texture Visuotactile Images for In-Gripper Tracking and Compensation
- **分类: cs.RO**

- **简介: 该论文提出TacSE3，用于解决低纹理触觉图像下的机械臂抓取运动估计问题，通过分解力场并估计SE(3)位姿，提升抓取中的跟踪与补偿能力。**

- **链接: [https://arxiv.org/pdf/2605.17929](https://arxiv.org/pdf/2605.17929)**

> **作者:** Zhongyuan Liao; Junzhe Wang; Qingyang Liu; Zhenmin Huang; Jun Ma; Yi Cai; Fei Meng; Haobo Liang; Michael Yu Wang
>
> **摘要:** Robotic in-hand manipulation requires reliable object-motion tracking under frequent visual occlusion, yet low-texture visuotactile images provide few stable correspondences for conventional image- or geometry-matching methods. This paper presents TacSE3, a tactile motion-estimation pipeline that converts low-texture visuotactile observations into a decoupled three-dimensional force field and estimates incremental rigid-body motion on SE(3). The method derives planar translation from contact-centroid motion and estimates rotation primarily from shear-related tactile responses, yielding a physically interpretable signal for in-gripper tracking and compensation. Experiments with paired DM-Tac fingertip sensors show that dual-sensor sensing reduces translation-rotation ambiguity, supports rotation tracking across axes and object geometries, and provides a lightweight compensation signal that improves disturbance tolerance in downstream manipulation tasks without retraining the base policy.
>
---
#### [new 061] PRIME: Physically-consistent Robotic Inertial and Motion Estimation for Legged and Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出PRIME方法，解决腿式和人形机器人运动估计问题，通过融合动力学约束，提升轨迹一致性与惯性参数估计精度。**

- **链接: [https://arxiv.org/pdf/2605.17681](https://arxiv.org/pdf/2605.17681)**

> **作者:** Jiarong Kang; Kunzhao Ren; Tao Pang; Xiaobin Xiong
>
> **备注:** Robotics: Science and Systems 2026
>
> **摘要:** Humanoid and legged robots interact with the environment through intermittent contacts, making accurate motion estimation fundamentally dependent on reasoning about contact dynamics. However, standard sensing pipelines-whether based on onboard proprioception with Extended Kalman Filters (EKFs) or external motion capture systems-recover only kinematics, while contact forces, contact timing, and inertial parameters remain unobserved. As a result, purely kinematic reconstructions often violate rigid-body dynamics, particularly during contact-rich motions. To enable accurate motion estimation from onboard kinematics in real-world deployment, we propose PRIME (Physically-consistent Robotic Inertial and Motion Estimation), a Maximum A Posteriori (MAP) formulation that refines measured kinematics and actuator commands into a dynamically consistent trajectory while jointly estimating frictional contact forces and physically consistent inertial parameters. Our approach incorporates differentiable contact dynamics with smoothed complementarity constraints and an Anitescu-style friction model, yielding a smooth optimization problem that remains tractable across versatile contact transitions. We evaluate PRIME on contact-rich locomotion with quadrupedal robots and the Unitree G1 humanoid, demonstrating improved trajectory consistency and accurate inertial parameter identification. Beyond improving state estimation and feedback control with calibrated inertial parameters, PRIME produces force- and contact-annotated motion reconstructions from real robots in deployment, which can be used to provide high-quality data for downstream learning applications, including large-scale behavior modeling and robot foundation models.
>
---
#### [new 062] REBAR: Reference Ethical Benchmark for Autonomy Readiness
- **分类: cs.RO; cs.CY**

- **简介: 该论文属于自主系统伦理评估任务，旨在解决缺乏量化评估标准的问题。提出REBAR框架，通过测试和仿真生成可计算的伦理性能指标。**

- **链接: [https://arxiv.org/pdf/2605.18423](https://arxiv.org/pdf/2605.18423)**

> **作者:** Jonathan Diller; David Barnes; Rebekah Bogdanoff; Rhett Collier; Roddy Collins; Keith Fieldhouse; Yonatan Gefen; Cameron Johnson; Anuriha Kodali; Brad Kriel; Varun Murali; James Niehaus; Mish Sukharev; Joseph VanPelt; Anthony Hoogs; Vijay Kumar; Arslan Basharat
>
> **备注:** To be presented at the 2026 Workshop on Robot Ethics - Ethical, Legal and User Perspectives in Robotics and Automation (WOROBET)
>
> **摘要:** As autonomous systems grow more advanced, objective metrics to evaluate their ethical and legal compliance are critical for informing end users of their limitations and ensuring accountability of those who misuse them. Current ethical embodied AI frameworks remain mostly qualitative, focusing on system design (through safety guardrails or targeted red teaming), and the realized guardrails often directly disallow unsafe behavior without providing the user with an override or interpretable reason. Instead, there is a need for computable metrics through rigorous testing that allow a user to determine the applicability of the system to the task. To address this gap, we introduce the Reference Ethical Benchmark for Autonomy Readiness (REBAR), a quantitative test and evaluation framework for autonomous systems. REBAR maps operating metrics into a computable Autonomy Readiness Level (ARL) rubric that can quantify ethical performance. Key innovations of the framework include a neuro-symbolic Large Language Model (LLM) approach to calculate and explain the ethical difficulty of scenarios, LLM-driven at-scale generation of test instances, and a versatile, photorealistic simulation environment. By evaluating white-box autonomy solutions through this rigorous testing pipeline, REBAR delivers an objective and repeatable benchmark score, bridging the gap between abstract principles and verifiable, accountable autonomy.
>
---
#### [new 063] Transfer Learning for Customized Car Racing Environments
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于深度强化学习任务，旨在通过迁移学习实现赛车环境中的快速圈速。研究比较了模型基础与模型无关方法，验证了迁移学习在不同环境中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.17928](https://arxiv.org/pdf/2605.17928)**

> **作者:** Benedict Florance Arockiaraj; Richard Chang; Wesley Yee
>
> **摘要:** Transfer Learning, a technique where a model/agent can use the knowledge/expertise that it gained from one task and exploit that to solve another closely-related task, is often used in tackling problems in deep learning. Through this project, we explore transfer learning in the purview of deep reinforcement learning. Specifically, we want to use transfer learning to achieve the fast lap times in OpenAI's Car racing environment by training the agent on one circuit, and racing it on other customized target environments by zero-shot transfer or by additional fine-tuning. In addition, we compare the performance of model-based and model-free approaches, and observe that model-based approaches dominate in performance and converge faster than model-free approaches in this environment. We observe that transfer learning in most setups not only boosts the performance on the target domain, but also shows high performance ability during learning.
>
---
#### [new 064] Robo-Cortex: A Self-Evolving Embodied Agent via Dual-Grain Cognitive Memory and Autonomous Knowledge Induction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Robo-Cortex，解决机器人在未知环境中导航的“经验遗忘”问题，通过自适应认知记忆和知识归纳实现自主策略进化。**

- **链接: [https://arxiv.org/pdf/2605.18729](https://arxiv.org/pdf/2605.18729)**

> **作者:** Nga Teng Chan; Yi Zhang; Yechi Liu; Renwen Cui; Fanhu Zeng; Zeyuan Ding; Xiancong Ren; Zhang Zhang; Qifeng Chen; Jian Liu; Yong Dai; Xiaozhu Ju
>
> **摘要:** The ability to navigate and interact with complex environments is central to real-world embodied agents, yet navigation in unseen environments remains challenging due to "experiential amnesia," where existing trajectory-driven or reactive policies fail to synthesize generalizable strategies from past interactions. We propose Robo-Cortex, a self-evolving framework that enables robots to autonomously induce navigation heuristics and refine cognitive strategies through a continuous reflection-adaptation loop. By abstracting success patterns and failure pitfalls into natural-language heuristics, Robo-Cortex enables a transition from passive execution to active strategy evolution. Our core innovation is an Autonomous Knowledge Induction (AKI) mechanism that distills multimodal trajectories into a structured Navigation Heuristic Library for knowledge generalization. The architecture further incorporates a Dual-Grain Cognitive Memory system, comprising a Short-term Reflective Memory (SRM) for real-time local progress analysis, and a Long-term Principle Memory (LPM) that abstracts past trajectories into reusable guiding and cautionary principles. To ensure robust decision-making, we introduce a multimodal Imagine-then-Verify loop, where a world model simulates potential outcomes and a VLM-based evaluator validates action plans. Extensive evaluations on IGNav, AR, and AEQA show that Robo-Cortex consistently outperforms strong baselines in both task success and exploration efficiency, with gains of up to +4.16% SPL over the strongest prior method and up to +15.30% SPL under heuristic transfer to unseen environments. Preliminary real-world robotic experiments further support the effectiveness of Robo-Cortex in physical settings.
>
---
#### [new 065] Mono-Hydra++: Real-Time Monocular Scene Graph Construction with Multi-Task Learning for 3D Indoor Mapping
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Mono-Hydra++，解决资源受限机器人实时3D场景图构建问题，采用单目RGB+IMU实现语义地图与场景关系建模。**

- **链接: [https://arxiv.org/pdf/2605.17661](https://arxiv.org/pdf/2605.17661)**

> **作者:** U. V. B. L. Udugama; George Vosselman; Francesco Nex
>
> **备注:** Submitted to ISPRS Journal of Photogrammetry and Remote Sensing. 50 pages, figures and tables included. Code: this https URL
>
> **摘要:** Autonomous agile robots need more than metric geometry: they must understand objects, rooms, places, and spatial relations for search, inspection, exploration, and human robot interaction. Conventional metric maps support localization and collision avoidance, but do not provide this semantic and relational structure. 3D scene graphs address this gap by connecting geometry with object level and room level understanding. Building such representations on agile platforms remains difficult because aerial and lightweight robots operate under strict payload, power, and compute limits, making RGB-D cameras and LiDAR sensors impractical for many onboard settings. We present Mono-Hydra++, a real time monocular RGB plus IMU pipeline for indoor metric semantic mapping and hierarchical 3D scene graph construction. The system combines M2H-MX, a DINOv3 based multi-task model for depth and semantics, with a deep feature visual inertial odometry front end, sparse predicted depth constraints in the VIO derived pose graph, semantic masking for dynamic regions, and pose aware temporal alignment before volumetric fusion in the Mono-Hydra backend. On the Go-SLAM ScanNet evaluation subset, Mono-Hydra++ achieves 1.6% lower average trajectory error than the strongest RGB-D baseline in our comparison, while using only monocular RGB plus IMU input. On calibrated 7-Scenes, it improves average ATE by 29.8% over the strongest competing calibrated baseline. We further validate Mono-Hydra++ in a real ITC building deployment using RealSense RGB plus IMU and demonstrate embedded feasibility by deploying the ONNX/TensorRT FP16 M2H-MX-L perception model at 25.53 FPS on a Jetson Orin NX 16GB. These results show that Mono-Hydra++ can provide real time metric semantic mapping and scene graph construction for resource constrained robotic platforms without relying on active depth sensors.
>
---
#### [new 066] Beyond Geometry: Efficient Topologically-Grounded Navigation in Complex 3D Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决复杂3D环境中几何模糊和计算成本高的问题，通过构建物理可达状态空间提升导航效率。**

- **链接: [https://arxiv.org/pdf/2605.17302](https://arxiv.org/pdf/2605.17302)**

> **作者:** Yifan Du; Chengwei Zhang; Siyu Liao; Zhongfeng Wang
>
> **摘要:** Ground robot navigation in complex 3D environments is often hindered by geometric ambiguity, where non-traversable structures such as furniture share local geometric properties with navigable ground. Furthermore, the computational cost of searching massive voxel spaces remains a significant challenge. To address these issues, we present a surface extraction framework that constructs a reduced state space of physically reachable standing positions by enforcing ground support, overhead clearance, and seed-based connectivity constraints. Evaluation across five Matterport3D indoor scenes and three PCT benchmark scenes demonstrates over 80\% state space reduction and sub-millisecond A* search on the Matterport3D scenes, with 100\% planning success across all 300 tested queries.
>
---
#### [new 067] Unified Walking, Running, and Recovery for Humanoids via State-Dependent Adversarial Motion Priors
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决人形机器人统一行走、奔跑与跌倒恢复问题。通过状态依赖的对抗运动先验框架，实现单一策略的多种运动模式切换。**

- **链接: [https://arxiv.org/pdf/2605.18611](https://arxiv.org/pdf/2605.18611)**

> **作者:** Yidan Lu; Yichao Zhong; Liu Zhao; Wanyue Li; Peng Lu
>
> **摘要:** We propose a unified reinforcement learning framework that enables a single policy to perform walking, running, and fall recovery on the Unitree G1 humanoid robot, validated on physical hardware without any explicit mode-switching command at deployment. The framework extends Adversarial Motion Priors (AMP) by replacing the conventional global reference distribution with a state-dependent gate that routes each training transition to one of two discriminators: a dedicated recovery discriminator and a velocity-conditioned locomotion discriminator that jointly covers walking and running. The gate is defined by a single fixed threshold on projected gravity: the recovery discriminator is activated when body tilt exceeds approximately $37^\circ$ from vertical ($|g_z+1|>0.6$); otherwise the locomotion discriminator is used, with the normalized commanded velocity serving as a condition that selects the appropriate reference trajectory between walk and run clips. Only three LAFAN1 reference clips are required to regularize the complete behavior set. At deployment, a single frozen ONNX policy executes at 50\,Hz with no runtime mode logic; hardware experiments demonstrate successful recovery from both prone and supine falls and smooth walk-to-run transitions under the same controller.
>
---
#### [new 068] Beyond Safety Filtering: Control Barrier Function-Informed Reinforcement Learning for Connected and Automated Vehicles
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自动驾驶任务，解决RL中奖励设计难调的问题。提出基于CBF的奖励设计，提升安全性和性能。**

- **链接: [https://arxiv.org/pdf/2605.16894](https://arxiv.org/pdf/2605.16894)**

> **作者:** Jianye Xu; Bassam Alrifaee
>
> **备注:** This paper has been accepted for publication in the Proceedings of the 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC 2026)
>
> **摘要:** Reinforcement Learning (RL) uses rewards to guide learning, yet reward design is typically hand-crafted using heuristics that can be difficult to tune. We propose a Control Barrier Function (CBF)-informed reward design for Multi-Agent RL (MARL) that converts CBF constraint values under joint MARL actions into a reward signal that explicitly guides safe learning. We compare against two heuristic reward baselines in a four-way multi-lane intersection with connected and automated vehicles. Results show that our method achieves the highest task performance and is less sensitive to reward hyperparameters, yielding consistently strong performance across the tested hyperparameter range. Code for reproducing the experimental results and a video demonstration are available at this https URL.
>
---
#### [new 069] 4DLidarOpen: An Open 4D FMCW Lidar Dataset for Motion-Aware Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出4DLidarOpen数据集，用于运动感知的自动驾驶。解决多传感器融合与动态场景理解问题，包含4D FMCW Lidar数据及多模态信息，支持目标检测、分割和运动预测任务。**

- **链接: [https://arxiv.org/pdf/2605.18074](https://arxiv.org/pdf/2605.18074)**

> **作者:** Kane Qian; Xin Zhao; Yining Shi; Rujun Yan; Zhengqing Pan; Kaojin Zhu; Mengmeng Yang; Kai Sun; Diange Yang; Kun Jiang
>
> **备注:** 15pages, 9 figures
>
> **摘要:** We present 4DLidarOpen, a large-scale open multi-modal dataset for autonomous driving, centered on 4D frequency-modulated continuous-wave (FMCW) Lidar sensing. Unlike conventional time-of-flight Lidar datasets that mainly provide geometric measurements, 4DLidarOpen includes point-wise radial velocity measurements from a forward-facing 4D FMCW Lidar, together with multiple Lidars of different types, including rotating, solid-state, and blind-spot variants, surround-view cameras, and 6-DOF ego-vehicle poses. The dataset was collected in complex urban environments in Beijing and covers dense pedestrian interactions, congested traffic, high-speed driving, and unprotected maneuvers. 4DLidarOpen provides synchronized multi-sensor data and 3D bounding-box annotations with persistent track IDs across five object categories. A hybrid annotation strategy is adopted, where large-scale auto-labeled data support scalable training and human experts refine annotations for the human-annotated training and validation sets. Based on this dataset, we establish benchmarks for 3D object detection, birds-eye view (BEV) segmentation and flow prediction, and motion forecasting with planning. Extensive experiments show that direct velocity measurements from 4D FMCW Lidar provide complementary motion cues for dynamic-scene understanding. Compared with geometric-only sensing, the velocity-aware representation improves motion-related perception and downstream forecasting and planning, especially in scenarios involving vulnerable road users and fast-moving objects. These results indicate that 4D FMCW Lidar is a promising sensing modality for motion-aware autonomous driving. The dataset and evaluation toolkit are publicly released to support research on 4D scene understanding, multi-Lidar fusion, and velocity-aware perception and planning.
>
---
#### [new 070] Data-Driven Dynamic Modeling of a Tendon-Actuated Continuum Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人动力学建模任务，旨在解决腱驱动连续机器人的动态建模难题。通过数据驱动方法建立简化模型，并用于实时控制设计。**

- **链接: [https://arxiv.org/pdf/2605.18720](https://arxiv.org/pdf/2605.18720)**

> **作者:** Harald Minde Hansen; Bjørn Kåre Sæbø; Kristin Y. Pettersen; Jan Tommy Gravdahl; Mario Di Castro
>
> **摘要:** Developing dynamic models for tendon-driven continuum robots is challenging due to their nonlinear, high-dimensional, and friction-dominated dynamics. This paper presents a comparative study of data-driven system identification methods, including N4SID, ARX, and SINDYc, for modeling a tendon-actuated continuum robot with rolling joints developed at CERN. Despite the high number of joints of the robot, experimental analysis reveals that a two-degree-of-freedom dynamic model can accurately capture the system dynamics, owing to strong kinematic dependencies between the joints. The models are validated against experimental data, and used in the design of a model predictive controller, demonstrating their feasibility for real-time control.
>
---
#### [new 071] SADP: Subgoal-Aware Diffusion Policy for Explainable Robots Learned from Foundation Model Generated Demonstrations
- **分类: cs.RO**

- **简介: 该论文提出SADP框架，解决机器人任务执行中缺乏可解释性的问题。通过生成带子目标的示范数据，训练扩散策略，使机器人能解释其执行阶段，提升任务成功率与可解释性。**

- **链接: [https://arxiv.org/pdf/2605.16871](https://arxiv.org/pdf/2605.16871)**

> **作者:** Site Hu; Takato Horii
>
> **摘要:** Explainable robots require not only successful task execution but also the ability to expose internal decision-making process in a user-friendly manner. However, most imitation learning methods are trained solely on task-level demonstrations, without explicitly modeling subgoal structure or execution progress. This limitation is further exacerbated by the scarcity of subgoal-level supervision in standard robot learning datasets, which restricts the development of robots that can convey the subtasks they are executing during long-horizon manipulation. To address this issue, this paper proposes Subgoal-Aware Diffusion Policy (SADP), a framework that leverages foundation models to autonomously generate subgoal-annotated demonstrations and trains diffusion policies on these datasets. SADP structures policy execution around human-interpretable subgoals by conditioning action generation on both task-level and subgoal-level descriptions. A lightweight auxiliary head further predicts subgoal completion states, allowing the robot to expose its current execution stage and monitor subgoal progression. Experiments in RLBench simulations and real-world evaluations on a UR5e robot demonstrate that SADP achieves higher task success rates than strong task-conditioned diffusion baselines, while providing subgoal-level execution signals for monitoring progress and diagnosing failures. These results highlight that built-in, rather than post-hoc, interpretability can coexist with high task performance.
>
---
#### [new 072] SSTL: Self-Sensing Tendon Loop for Hysteresis Modeling and Compensation in Tendon-Sheath Mechanisms
- **分类: cs.RO**

- **简介: 该论文属于控制任务，解决 tendon-sheath 机制中的滞回问题。提出 SSTL 结构，通过测量张力实现滞回补偿，提升控制精度。**

- **链接: [https://arxiv.org/pdf/2605.16870](https://arxiv.org/pdf/2605.16870)**

> **作者:** Myeongbo Park; Junhyun Park; Ihsan Ullah; Chunggil An; Minho Hwang
>
> **备注:** 8 pages, 7 figures, 4 tables
>
> **摘要:** Flexible endoscopic robots enable minimally invasive access through natural orifices, but their control accuracy is limited by configuration-dependent hysteresis in the tendon-sheath mechanisms (TSMs). Tendon-sheath friction and tendon elasticity induce a systematic discrepancy between the proximal actuation input and distal output, and this discrepancy varies with the insertion tube configuration. To address this challenge, this paper proposes the Self-Sensing Tendon Loop (SSTL), a double-pass tendon loop routed through the insertion tube and wrapped around a distal pulley, and returned to the proximal end. The loop structure allows both the input and output tensions of the SSTL to be measured proximally, thereby providing an input-output tension profile without requiring distal force or fiber-optic sensors. Because the SSTL shares the same routing path as the actuation TSM, the two TSMs exhibit strongly correlated hysteresis behaviors. From the SSTL tension profile, a learning-based mapping estimates the configuration-dependent hysteresis parameters of the actuation TSM, which are then used by a feedforward controller to compensate for actuation hysteresis. We validate the proposed method by tracking actuation tendon tension under three different insertion tube configurations. Across sinusoidal and random trajectories, the proposed method reduces average RMSE by 88.1% compared with the uncompensated baseline, achieving 97.8% of the performance of direct identification, which requires direct measurement of the input and output tension profile of the actuation TSM.
>
---
#### [new 073] Dexora: Open-source VLA for High-DoF Bimanual Dexterity
- **分类: cs.RO**

- **简介: 该论文提出Dexora，首个开源视觉-语言-动作系统，解决高自由度双臂操作问题。通过混合遥控流程和高质量数据训练，提升操作精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.18722](https://arxiv.org/pdf/2605.18722)**

> **作者:** Zongzheng Zhang; Jingrui Pang; Zhuo Yang; Kun Li; Minwen Liao; Saining Zhang; Guoxuan Chi; Jinbang Guo; Huan-ang Gao; Modi Shi; Dongyun Ge; Yao Mu; Jiayuan Gu; Rui Chen; Hao Dong; Huazhe Xu; Li Yi; Yixin Zhu; Hang Zhao; Pengwei Wang; Shanghang Zhang; Guocai Yao; Jianyu Chen; Hongyang Li; Hao Zhao
>
> **备注:** Accpeted by ICRA 2026
>
> **摘要:** Vision-Language-Action (VLA) models have recently become a central direction in embodied AI, but current systems are restricted to either dual-gripper control or single-arm dexterous hand manipulation. While low-dimensional gripper control can often be handled with simpler methods, high-dimensional dexterous hand control benefits greatly from full end-to-end VLA learning. In this work, we introduce Dexora, the first open-source VLA system that natively targets dual-arm, dual-hand high-DoF manipulation. We design a hybrid teleoperation pipeline that decouples gross arm kinematics (captured with a custom exoskeleton backpack) from fine finger motion (markerless hand tracking via Apple Vision Pro), and that drives both a physical dual-arm dual-hand platform and an identical MuJoCo digital twin. Using that interface, we assemble a large training corpus: an embodiment-matched synthetic corpus (100K simulated trajectories, 6.5M frames) and a real-world dataset of 10K teleoperated episodes (2.92M frames). To mitigate noisy teleoperation demonstrations, we propose a data-quality-aware training recipe: an offline discriminator provides clip-level weights for diffusion-transformer policy training, down-weighting low-quality demonstrations. Empirically, Dexora outperforms competitive VLA baselines on both basic and dexterous benchmarks (e.g., average dexterous success 66.7% vs. 51.7%), attains 90% success on basic tasks, and shows robust out-of-distribution and cross-embodiment generalization. Ablations confirm the importance of real data and the discriminator for dexterity.
>
---
#### [new 074] Haptic Rendering of Fractional-Order Viscoelasticity: Passivity and Rendering Fidelity
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于虚拟触觉渲染任务，解决分数阶粘弹性材料的稳定性与真实感问题。通过分析和实验，建立统一的理论框架以确保渲染的无源性和精度。**

- **链接: [https://arxiv.org/pdf/2605.16389](https://arxiv.org/pdf/2605.16389)**

> **作者:** Gorkem Gemalmaz; Harun Tolasa; Volkan Patoglu
>
> **备注:** Under review for publication in IEEE Transactions on Robotics
>
> **摘要:** Haptic rendering of viscoelastic materials that exhibit creep and stress relaxation is crucial for many applications, such as medical training with realistic biological tissue models. Fractional-order viscoelastic models provide an effective means of describing intrinsically time-dependent dynamics with few parameters, as these models can naturally capture memory effects. In this study, we present analyses of passivity and rendering performance for fractional-order viscoelastic models under finite-memory discretization. We derive closed-form expressions to ensure the passivity of haptic rendering with a fractional-order (FO) standard linear solid (SLS) model based on Grunwald-Letnikov derivative under short-memory discretization. We also provide symbolic expressions for the effective stiffness and damping of such FO-SLS models. The resulting passivity conditions constitute a unified framework that generalizes previously reported results for integer-order Kelvin-Voigt, Maxwell, and SLS models, since these results are special cases of the newly derived condition. Furthermore, we provide experimental validations of the theoretical passivity bounds and human-subject evaluations of perceived realism of FO-SLS models. Overall, this study establishes a unified theoretical framework and experimental evaluations for FO viscoelastic rendering under short-memory discretization.
>
---
#### [new 075] ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文提出ESI-Bench，用于评估具身空间智能，解决感知-行动闭环问题，通过主动探索提升任务表现。**

- **链接: [https://arxiv.org/pdf/2605.18746](https://arxiv.org/pdf/2605.18746)**

> **作者:** Yining Hong; Jiageng Liu; Han Yin; Manling Li; Leonidas Guibas; Li Fei-Fei; Jiajun Wu; Yejin Choi
>
> **备注:** this https URL
>
> **摘要:** Spatial intelligence unfolds through a perception-action loop: agents act to acquire observations, and reason about how observations vary as a function of action. Rather than passively processing what is seen, they actively uncover what is unseen - occluded structure, dynamics, containment, and functionality that cannot be resolved from passive sensing alone. We move beyond prior formulations of spatial intelligence that assume oracle observations by recasting the observer as an actor. We introduce ESI-BENCH, a comprehensive benchmark for embodied spatial intelligence spanning 10 task categories and 29 subcategories built on OmniGibson, grounded in Spelke's core knowledge systems. Agents must decide what abilities to deploy - perception, locomotion, and manipulation - and how to sequence them to actively accumulate task-relevant evidence. We conduct extensive experiments on state-of-the-art MLLMs and find that active exploration substantially outperforms passive counterparts, with agents spontaneously discovering emergent spatial strategies without explicit instructions, while random multi-view often adds noise rather than signal despite consuming far more images. Most failures stem not from weak perception but from action blindness: poor action choices lead to poor observations, which in turn drive cascading errors. While explicit 3D grounding stabilizes reasoning on depth-sensitive tasks, imperfect 3D representation proves more harmful than 2D baselines by distorting spatial relations. Human studies further reveal that unlike humans who seek falsifying viewpoints and revise beliefs under contradiction, models commit prematurely with high confidence regardless of evidence quality, exposing a metacognitive gap that neither better perception nor more embodied interaction alone can close.
>
---
#### [new 076] Consent Chain Degradation in Embodied Multi-Agent Systems: Bridging the Gap Between AI Agent Governance and Robot Ethics
- **分类: cs.CY; cs.AI; cs.MA; cs.RO**

- **简介: 该论文属于AI伦理与机器人治理领域，解决多智能体系统中授权链导致的同意退化问题，提出CoRVE框架以监控和评估同意的传播与失效。**

- **链接: [https://arxiv.org/pdf/2605.16300](https://arxiv.org/pdf/2605.16300)**

> **作者:** Mehmet Haklidir
>
> **备注:** Accepted for oral presentation at the 2nd Workshop on Robot Ethics (WoRoBet), ICRA 2026, Vienna, Austria, June 1, 2026. 6 pages, 3 tables, 1 figure
>
> **摘要:** Robotic systems are moving from isolated platforms to interconnected multi-agent ecosystems that operate in human environments. This shift raises a governance problem that existing frameworks do not address: how does consent propagate, degrade, and break down across chains of delegation between embodied autonomous agents? The AI ethics community has begun to study consent for digital software agents, and the HRI community has examined consent in dyadic human-robot encounters. Neither body of work covers what happens when physical robots delegate tasks to other robots in ways that affect humans. This paper introduces consent chain degradation (CCD), a conceptual framework for analyzing how the specificity, validity, and scope of human consent erodes as authority passes through multi-robot delegation chains. We propose a three-layer governance architecture, the Consent Runtime Verification Framework for Embodied Agents (CoRVE), which integrates consent scope modeling, delegation chain tracking, and physical irreversibility assessment. Three scenarios in healthcare, domestic, and industrial robotics show how CCD arises in practice, including a worked numerical example. A regulatory gap analysis covering the EU AI Act, the GDPR, the Machinery Regulation, and the Revised Product Liability Directive shows that all four instruments leave core CCD dimensions unaddressed.
>
---
#### [new 077] EfficientTDMPC: Improved MPC Objectives for Sample-Efficient Continuous Control
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出EfficientTDMPC，用于连续控制的高效模型预测控制方法，解决样本效率低的问题。通过集成模型和不确定性惩罚提升性能。**

- **链接: [https://arxiv.org/pdf/2605.16692](https://arxiv.org/pdf/2605.16692)**

> **作者:** Thomas Evers; Cristian Meo; Wendelin Bohmer; Justin Dauwels; Yaniv Oren
>
> **摘要:** We introduce EfficientTDMPC, a sample-efficient model-based reinforcement learning method for continuous control built on the TD-MPC family of algorithms. Central to this family is a planner that aims to find an action sequence that maximizes the estimated return. The return is estimated using a learned model and value networks, each of which can introduce error. EfficientTDMPC proposes to reduce this error in two ways. First, it introduces an ensemble of dynamics models and averages the return estimates across those models and across different rollout depths. Second, it adds the option to apply an uncertainty penalty to the planner objective, yielding a planner that avoids actions with uncertain return estimates. It then adds practical improvements which increase buffer data freshness and reduce compute. Lastly, we find that our contributions enable EfficientTDMPC to benefit more from a higher update-to-data (UTD) ratio, further improving sample efficiency. To the best of our knowledge, in the low data regime of each benchmark, EfficientTDMPC achieves state-of-the-art (SOTA) in terms of sample efficiency on HumanoidBench-Hard and DMC hard, while matching SOTA on DMC easy.
>
---
#### [new 078] Agentic Pipeline for Self-Synchronized Multiview Joint Angle Monitoring in Uncalibrated Environments
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于多视角运动捕捉任务，解决非标定环境下关节角度监测问题。通过自同步机制和智能选择策略，实现无标定的精准关节角度估计。**

- **链接: [https://arxiv.org/pdf/2605.16419](https://arxiv.org/pdf/2605.16419)**

> **作者:** Juncheng Yu; Lusi A; Haoxuan Xie; Weiming Wang
>
> **备注:** Accepted by EMBC 2026. 7 pages, 3 figures
>
> **摘要:** Kinematic monitoring plays a critical role in long-term rehabilitation for patients with spinal cord injury (SCI), where multi-view markerless motion capture methods have shown significant potential. However, owing to the reliance on calibration and the difficulty of achieving multi-view synchronization, their deployment in patient self-deployed environments remains challenging. In this work, we propose an agentic pipeline for self-synchronized multi-view joint angle monitoring in uncalibrated environments using two cameras without hardware triggers. The Multimodal large language models enable automatic video synchronization and agent-driven self-verification. State-of-the-art monocular 2D pose estimation models are employed to extract candidate poses, where an agent-based selection mechanism is then applied to automatically identify and track the target subject, thereby producing consistent 2D poses in the presence of multiple individuals and occlusions. Such 2D poses are optimized to estimate joint angles from uncalibrated multi-view pose sequences, ensuring interpretability through explicit geometric modeling. Validation against Vicon system demonstrated the strong performance, achieving an MAE of $5.97^\circ \pm 2.36^\circ$ and a Pearson correlation coefficient of $0.962 \pm 0.014$. The proposed method is expected to provide a practical, patient self-deployable system to perform daily kinematic monitoring in uncalibrated home environments.
>
---
#### [new 079] CLAP: Contrastive Latent-space Prompt Optimization for End-to-end Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决长尾安全场景下的规划问题。通过CLAP框架，优化VLA模型的潜在空间提示，提升复杂场景性能，同时保持常规场景表现。**

- **链接: [https://arxiv.org/pdf/2605.17284](https://arxiv.org/pdf/2605.17284)**

> **作者:** Ruiyang Zhu; Yuehan He; Boyuan Zheng; Zesen Zhao; Ahmad Chalhoub; Qingzhao Zhang; Z. Morley Mao
>
> **备注:** 9 pages + appendix
>
> **摘要:** End-to-end autonomous driving systems powered by Vision-Language-Action (VLA) models achieve strong performance on common driving scenarios, yet remain brittle in rare but safety-critical long-tail situations such as active construction zones and complex yielding geometries. In this paper, we present a method that addresses the long-tail challenging scenes beyond data scaling and model training. We introduce CLAP (Contrastive Latent-space Prompt optimization), a location-aware adaptation framework that augments a frozen VLA driving model with per-roadblock soft prompts, optimized from crowdsourced data and retrieved on demand via Vehicle-to-Everything (V2X) communication. Our approach rests on two observations from VLAs' latent space: (i) at the VLA's hidden-state layer, scenarios from the same roadblock cluster tightly and occupy compact regions of the latent space; and (ii) within a single roadblock, long-tail and normal frames are heavily intermixed in the latent representation, making it difficult to improve one without disturbing the other. CLAP addresses this via a two-stage pipeline: supervised contrastive learning to discover a roadblock-specific hard-scene direction, followed by directionally regularized prompt optimization that selectively improves challenging frames while preserving normal frame performance. On the NAVSIM benchmark with various state-of-the-art VLA backbones, CLAP reduces challenging scenario planning error by 24% with no regression on normal frames, significantly improving planning performance.
>
---
#### [new 080] StableVLA: Towards Robust Vision-Language-Action Models without Extra Data
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决模型在未见视觉干扰下的鲁棒性问题。提出IB-Adapter模块，无需额外数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2605.18287](https://arxiv.org/pdf/2605.18287)**

> **作者:** Yiyang Fu; Chubin Zhang; Shukai Gong; Yufan Deng; Kaiwei Sun; Qiyang Min; Qibin Hou; Yansong Tang; Jianan Wang; Daquan Zhou
>
> **备注:** Accepted by ICML 2026. Code: this https URL. Project website: this https URL
>
> **摘要:** It is infeasible to encompass all possible disturbances within the training dataset. This raises a critical question regarding the robustness of Vision-Language-Action (VLA) models when encountering unseen real-world visual disturbances, particularly under imperfect visual conditions. In this work, we conduct a systematic study based on recent state-of-the-art VLA models and reveal a significant performance drop when visual disturbances absent from the training data are introduced. To mitigate this issue, we propose a lightweight adapter module grounded in information theory, termed the Information Bottleneck Adapter (IB-Adapter), which selectively filters potential noise from visual inputs. Without requiring any extra data or augmentation strategies, IB-Adapter consistently improves over the baseline by an average of 30%, while adding fewer than 10M parameters, demonstrating notable efficiency and effectiveness. Furthermore, even with a 14x smaller backbone (0.5B parameters) and no pre-training on the Open X-Embodiment dataset, our model StableVLA achieves robustness competitive with 7B-scale state-of-the-art VLAs. With negligible parameter overhead (<10M), our approach maintains accuracy on long-horizon tasks and surpasses OpenPi under both synthetic and physical visual corruptions.
>
---
#### [new 081] Not What You Asked For: Typographic Attacks in Household Robot Manipulation
- **分类: cs.CR; cs.AI; cs.RO**

- **简介: 该论文研究家庭机器人操作中的字体攻击问题，属于机器人感知与安全任务。针对视觉语言模型的脆弱性，提出对抗贴纸攻击，验证其对机器人抓取任务的影响。**

- **链接: [https://arxiv.org/pdf/2605.18593](https://arxiv.org/pdf/2605.18593)**

> **作者:** Ali Iranmanesh; Peng Liu
>
> **备注:** 10 pages, 1 figure, IEEE conference format
>
> **摘要:** Open-vocabulary embodied AI agents increasingly rely on vision-language models such as CLIP for object perception and task grounding. However, the shared embedding space that enables this flexibility introduces a structural vulnerability to typographic attacks, where printed text in a physical scene semantically overrides visual judgment. While prior work has quantified this threat in static 2D benchmarks and 3D navigation tasks, its impact on the full Sense-Plan-Act pipeline of household robot manipulation remains unexplored. This work evaluates typographic attacks in a Habitat-based simulation using the HomeRobot benchmark. We introduce a decoupled perception architecture that exposes a frozen CLIP encoder to adversarial stickers while maintaining geometric grounding via DETIC. In a controlled evaluation pool of 59 attributable episodes, the attack achieves an overall Attack Success Rate (ASR) of 67.8%, rising to 70.0% among fully successful episodes, under uncontrolled viewing angles and occlusion with no perceptual optimization. Critically, we find that perceptual errors propagate through the persistent 3D semantic map to produce kinetic failures, defined here as physically executed grasping and transport of the wrong object driven by an adversarially poisoned semantic state. In these cases, the robot physically grasps and delivers the wrong object to a target receptacle. These results establish typographic misclassification as a real, measurable, and physically consequential threat to the safety of modular manipulation pipelines that prior typographic attack research has left unexamined.
>
---
#### [new 082] From Prompts to Protocols: An AI Agent for Laboratory Automation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于实验室自动化任务，旨在解决科学家编程复杂的问题。通过集成大语言模型与实验系统，实现自然语言控制实验流程，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2605.16552](https://arxiv.org/pdf/2605.16552)**

> **作者:** Angelos Angelopoulos; James F. Cahoon; Ron Alterovitz
>
> **摘要:** Automating science laboratories enables faster, safer, more accurate, and more reproducible execution of protocols, accelerating the discovery and testing of new materials, drugs, and more. However, setting up and running autonomous labs requires coordinating numerous instruments and robots, forcing scientists to write code, manage configuration files, and navigate complex software infrastructure. We present an AI agent architecture that integrates large language models with laboratory orchestration, enabling scientists to interactively create and monitor automated lab protocols using natural language. Integrated into the Experiment Orchestration System (EOS), the AI agent operates under an agentic loop with automated validation and error correction, and supports the complete experimental lifecycle: creating protocols, running and monitoring both protocols and closed-loop optimization campaigns, and analyzing results. A visual graph editor renders protocols as interactive node-based diagrams synchronized with the AI agent's protocol representation, enabling seamless alternation between AI-assisted and manual protocol construction. Evaluated on three simulated automated labs spanning chemistry, biology, and materials science, the AI agent achieves a 97% first-attempt protocol generation success rate and an order of magnitude reduction in required interface actions.
>
---
#### [new 083] TaskGround: Structured Executable Task Inference for Full-Scene Household Reasoning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出TaskGround，解决全场景家庭推理中的任务结构推断问题，通过结构化接地提升模型效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.18109](https://arxiv.org/pdf/2605.18109)**

> **作者:** ZhiYuan Feng; Yu Deng; Ruichuan An; Zhenhua Liu; Qixiu Li; Keming Wu; Zhiying Du; Weijie Wang; Haoxiao Wang; Shuang Chen; Sicheng Xu; Yaobo Liang; Jiaolong Yang; Baining Guo
>
> **备注:** Project page: this https URL
>
> **摘要:** In real home deployments, household agents must often operate from a complete household scene and a situated household request, rather than from a clean task specification. Such requests require agents to identify task-relevant entities, recover intended task conditions, and resolve ordering constraints from the surrounding scene context. We formalize this capability as full-scene household reasoning: given a complete household scene and a situated household request, an agent must infer executable task structure before producing a grounded skill-level action sequence. This setting is challenging because complete household scenes contain substantial task-irrelevant information, making direct complete-scene prompting inefficient and error-prone. In practical deployment, this challenge is further amplified by privacy and local compute constraints, which favor compact open-weight models with limited long-context reasoning ability. We propose TaskGround, a training-free and model-agnostic Ground-Infer-Execute framework that grounds complete scenes into compact task-relevant scene slices, infers executable task structure, and compiles it into grounded skill-level action sequences. To evaluate this setting, we introduce FullHome, a human-validated evaluation suite of 400 household tasks spanning diverse home-scale environments and both goal-oriented and process-constrained requirements. On FullHome, TaskGround improves task success rates by large margins across both proprietary and open-weight models. Notably, it makes Qwen3.5-9B competitive with GPT-5 under direct complete-scene prompting while reducing total input-token cost by up to 18x. Our results identify executable task-structure inference as a central bottleneck in full-scene household reasoning and show that structured grounding can make compact local models substantially more effective for practical household deployment.
>
---
#### [new 084] Distributed 3D Leader-Follower Formation Control with Field-of-View Safety via Control Barrier Functions
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多无人机编队控制任务，解决3D编队中视觉感知安全问题。通过设计感知-aware 控制架构，确保领航机在跟随者视野内，保障编队跟踪与视觉安全。**

- **链接: [https://arxiv.org/pdf/2605.17533](https://arxiv.org/pdf/2605.17533)**

> **作者:** Immanuel R. Santjoko; Richie R. Suganda; Miao Pan; Bin Hu
>
> **备注:** 9 page
>
> **摘要:** This letter proposes a distributed 3D leader-follower formation (3D-LFF) control framework for multi-UAV systems that achieves formation tracking while enforcing perception safety constraints. Maintaining safe, vision-based 3D-LFF is challenging because onboard cameras impose strict Field-of-View (FOV) limitations, and demanding formation commands can drive the leader outside the follower's camera frustum, resulting in loss of visibility. To address this issue, we develop a perception-aware safe control architecture that guarantees visibility by construction. First, we derive a relative kinematic model in a line-of-sight coordinate representation and design a distributed 3D-LFF tracking controller using only locally available relative states. Next, we embed the nominal formation controller within a Control Barrier Function-based Quadratic Program (CBF-QP) safety filter that minimally modifies the commanded velocities to maintain the leader inside the follower's camera frustum while preserving formation tracking whenever feasible. Gazebo simulations and Crazyflie hardware experiments validate the proposed approach, demonstrating accurate formation tracking and effective FOV enforcement, including scenarios in which the nominal desired formation conflicts with visibility constraints.
>
---
#### [new 085] ATRACT: A Trustworthy Robotic Autonomous system to support Casualty Triage
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出ATRACT系统，用于战场伤员分类。任务是解决伤员优先级评估问题，通过无人机和可穿戴设备多模态数据融合，提升远程 triage 的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.17123](https://arxiv.org/pdf/2605.17123)**

> **作者:** Tasweer Ahmad; Rafael Pina; Sandip Pradhan; Arindam Sikdar; Mindula Illeperuma; Khizer Saeed; Peter Lee; Varuna De Silva; Ardhendu Behera
>
> **摘要:** At a time when drones are increasingly associated with hostile operations, we re-purpose them for humanitarian and life-saving applications. However, adapting search and rescue drones for battlefield triage remains extremely challenging; the technology must perform reliably to support frontline medics who are forced to operate under extreme uncertainty, restricted access, and significant personal risk. Due to growing vulnerabilities of casualty evacuation in conflicting zones, this paper presents ATRACT (A Trustworthy Robotic Autonomous system to support Casualty Triage), a novel human-in-the-loop decision support system to enable early battlefield triage during the critical post-trauma period. ATRACT integrates drone-captured video with wearable sensor input for multi-modal learning to support casualty-state assessment, thereby addressing the limitations of existing systems. Drone video captures fine-grained behavioural cues, such as pose, posture, while body-worn sensors provide complementary physiological signals, including heart rate, breathing rate, and movement. By combining two modalities, ATRACT provides evidence to support the early judgement of medics when direct access to the casualty is delayed, risky, or restricted. To mitigate the data realism gap pertaining to injured actions, a conditional variational autoencoder is devised for data augmentation. Experimental results on our drone captured dataset show that proposed pipeline achieves 85.7% accuracy for action classification; while our lightweight CNN visual encoder remains competitive with stronger pre-trained video backbones. Overall, the results support ATRACT as a practically meaningful step towards remote triage in contested environments, where multi-modal sensing, human oversight and trustworthy decision support can improve casualty prioritisation, and lessen the exposure of frontline medics.
>
---
#### [new 086] Overcoming the Intrinsic Performance Limitations of MEMS IMU via Diffusion-Based Generative Learning
- **分类: eess.SP; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于导航任务，旨在解决低精度IMU性能受限问题。通过构建扩散生成模型，将低成本IMU数据转化为高精度虚拟数据，提升定位与姿态估计效果。**

- **链接: [https://arxiv.org/pdf/2605.16391](https://arxiv.org/pdf/2605.16391)**

> **作者:** Jiarui Lv; Feng Zhu; Xiaohong Zhang
>
> **摘要:** Inertial measurement units (IMUs) are fundamental sensing components in multi-source integrated navigation systems, and their performance directly determines the accuracy and reliability of solutions. However, the precision of low-cost IMUs is inherently constrained by hardware limitations. Recently, generative artificial intelligence has demonstrated remarkable capability in modeling complex data distributions and reconstructing high-fidelity signals. Motivated by this, we propose a diffusion-based generative learning framework for synthesizing high-fidelity virtual IMU data from low-cost IMU measurements. Specifically, a conditional diffusion model based on a U-Net architecture is constructed, where high-grade IMU measurements are utilized as ground-truth priors and low-cost IMU measurements are employed as conditional inputs. The virtual IMU data generated by the model is used for subsequent navigation and localization tasks. Experimental results demonstrate that the generated virtual IMU data significantly outperform the original low-cost IMU measurements in both positioning and attitude estimation. Furthermore, we transfer the model to airborne mapping experiments, where the proposed method produces thinner and more consistent point clouds. Overall, the proposed framework breaks the performance limits of low-cost IMU and demonstrates the potential of diffusion-based generative learning for virtual high-grade IMU data.
>
---
#### [new 087] See Silhouettes in Motion with Neuromorphic Vision
- **分类: eess.IV; cs.CV; cs.RO**

- **简介: 该论文属于图像处理任务，解决动态场景下运动模糊问题。通过结合帧与事件数据，实现高效实时二值化，提升边缘设备的视觉感知能力。**

- **链接: [https://arxiv.org/pdf/2605.17984](https://arxiv.org/pdf/2605.17984)**

> **作者:** Pei Zhang; Shijie Lin; Zhou Ge; Jinpeng Chen; Wei Pu
>
> **备注:** 12 pages, 12 figures, and 3 tables. This work is under review. Project page: this https URL
>
> **摘要:** Quasi-bimodal objects, such as text, road signs, and barcodes, play a basic yet vital role in daily visual communication. By boiling these down to clear silhouettes, binarization uses a minimal language to convey essential vision cues for maximum downstream efficiency. The catch is that frame-based imaging often struggles on mobile platforms like drones, self-driving cars, and underwater vehicles. In these dynamic scenes, rapid motion and harsh lighting can make it blind, causing severe motion blur and erasing crucial details. To overcome the limits, neuromorphic vision via event cameras, featuring microsecond-level temporal resolution and high dynamic range, steps in as a natural solution. Building upon this event-driven sensing paradigm, we introduce a simple yet effective dual-modal approach that harnesses the synergy between frames and events to achieve real-time, high-frame-rate binarization on CPU-only devices. Extensive evaluations present that it earns competitive performance against leading techniques in reducing motion blur, while delivering impressive improvements under challenging illumination. Besides, our asynchronous workflow bypasses event scarcity that breaks traditional time-binning reconstruction, maintaining clear target shapes even at extreme kilohertz frame rates. Its binary results further serve as reliable representations that facilitate a range of downstream tasks. This work paves the way towards lightweight perception and interaction in embodied intelligence on resource-constrained edge platforms.
>
---
#### [new 088] Qumus: Realization of An Embodied AI Quantum Material Experimentalist
- **分类: cond-mat.mes-hall; cond-mat.mtrl-sci; cs.AI; cs.RO**

- **简介: 该论文介绍Qumus，首个能进行量子材料实验的具身AI系统，解决现实科学发现难题。它自主完成材料制备与分析，实现AI创建石墨烯和纳米器件。**

- **链接: [https://arxiv.org/pdf/2605.18407](https://arxiv.org/pdf/2605.18407)**

> **作者:** Lihan Shi; Zhaoyi Joy Zheng; Xinzhe Juan; Yimin Wang; Ming Yin; Mayank Sengupta; Kristina Wolinski; Yanyu Jia; Jingzhi Shi; Derek Saucedo; Neill Saggi; Haosen Guan; Kenji Watanabe; Takashi Taniguchi; Ali Yazdani; Mengdi Wang; Sanfeng Wu
>
> **备注:** 29 Pages in total. Supplementary Demo Videos are available at this https URL
>
> **摘要:** While modern Large Language Models (LLMs) and agentic artificial intelligence (AI) have demonstrated transformative capabilities in digital domains, the realization of embodied AI capable of real-world scientific discovery remains a difficult frontier. The advancements are hindered by the inherent complexity of integrating high-level reasoning, multimodal information processing and real-time physical execution. Here we introduce Qumus, the first AI quantum materials experimentalist. Physically embodied within a robotic mini-laboratory, Qumus is an intelligent, multimodal, and multi-agent system designed for the creation and nano-processing of atomically thin two-dimensional (2D) materials and stacked van der Waals (vdW) structures. Qumus autonomously navigates the full scientific cycle, from hypothesis generation and protocol planning to multi-step experimental execution, result analysis and reporting, acting as an experimentalist. Markedly, the system has achieved, for the first time, the AI-creation of graphene, as well as the first AI-fabrication of complex nanodevices including atomically thin field-effect transistors via vdW stacking. Qumus excels at these tasks by demonstrating autonomous error correction and closed-loop experimentation. Our results establish a generalizable framework for self-improving embodied AI systems that learn directly from the quantum world, opening a pathway toward accelerated discovery in quantum materials, electronics and beyond.
>
---
#### [new 089] Is VLA Reasoning Faithful? Probing Safety of Chain-of-Causation
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文研究视觉-语言-动作（VLA）驾驶模型的推理忠实性问题，分析其在物理场景中的表现，揭示模型推理与现实不一致的现象，提出安全架构以提升可靠性。**

- **链接: [https://arxiv.org/pdf/2605.17268](https://arxiv.org/pdf/2605.17268)**

> **作者:** Nicanor Mayumu; Xiaoheng Deng; Patrick Mukala
>
> **备注:** Accept (Poster), CVPR 2026 Workshop DriveX NonArchival Track
>
> **摘要:** We present the first systematic study of faithfulness in Vision-Language-Action (VLA) driving models, analyzing 300 Alpamayo-R1-10B inferences across 100 diverse PhysicalAI-AV scenarios. Our main finding is that output natural-language rationales with trajectories may be significantly unfaithful: (i) overall reasoning fidelity is only 42.5%, with Chain-of-Causation matching scene reality less than half the time; (ii) 94 missed pedestrians in one-third of pedestrian-relevant scenes; (iii) 97.7% trajectory fragility under mild visual perturbations; and (iv) only 48.3% mean reasoning-action consistency, with 53.3% of inferences exhibiting low consistency, including 37.9% of stop-claimed cases where the model continues instead. We formalize faithfulness information-theoretically, define entity and action fidelity with verification criteria, and outline a four-component safety architecture aligned with these results.
>
---
#### [new 090] EgoKit: Towards Unified Low-Cost Egocentric Data Collection with Heterogeneous Devices
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EgoKit，解决多设备协同采集第一视角数据的问题，实现统一的录制流程和日志格式，支持多种设备及附加摄像头。**

- **链接: [https://arxiv.org/pdf/2605.16797](https://arxiv.org/pdf/2605.16797)**

> **作者:** Liuchuan Yu; Erdem Murat; Beichen Wang; Yan Zeng; Tingting Luo; Huizhen Zhou; Shanghao Li; Huining Feng; Zhigen Zhao; Ning Yang; Ke Jing; Yunhao Liu; Ruoya Sheng
>
> **摘要:** Egocentric video is increasingly used as a data source for robot learning, activity understanding, and embodied AI research, but collecting it at scale remains fragmented in practice: each candidate host device, such as an Android phone, iPhone, iPad, smart glasses, or extended reality (XR) headset, exposes a different SDK, a different policy on raw camera access, and different limitations on external USB cameras and on-device tracking. Synchronized ego-view and wrist-view capture is therefore typically obtained by either committing to a single proprietary platform or building one-off rigs that do not transfer across devices. To address this gap, we present EgoKit, a toolkit that exposes the same egocentric recording workflow across six heterogeneous host devices. Across all supported devices, EgoKit presents the same recording interaction and produces locally stored video with a uniform log format; on XR headsets, it additionally logs head pose and OpenXR-standard 26-joint hand tracking aligned to the video streams. The companion accessories, including two wrist cameras with mounts, a head strap, and a USB-C hub, add wrist-view capture to any supported host without custom hardware fabrication. EgoKit is available at \url{this https URL}.
>
---
#### [new 091] PH-Dreamer: A Physics-Driven World Model via Port-Hamiltonian Generative Dynamics
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决世界模型物理不结构化问题。通过引入基于端口哈密顿的框架，提升模型的物理合理性与控制效率。**

- **链接: [https://arxiv.org/pdf/2605.18303](https://arxiv.org/pdf/2605.18303)**

> **作者:** Xueyu Luan; Chenwei Shi
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** World models built on recurrent state space architectures enable efficient latent imagination, yet remain physically unstructured, producing dynamics that violate conservation and dissipative principles. We introduce a unified Port-Hamiltonian framework that remedies this through three synergistic mechanisms. First, we embed implicit physical priors into recurrent transitions by modeling projected latent evolution as action controlled energy routing governed by flow and dissipation, biasing the projected PH phase space toward a more compact and physically structured representation. Second, we develop a kinematics aware energy world model that estimates the Hamiltonian and power balance from proprioceptive observations, providing an explicit physical signal for thermodynamic reasoning. Third, leveraging these energy gradients, we establish an energy guided Actor-Critic that uses Lagrangian multipliers to regularize policy optimization toward lower energy and smoother control. Across visual control benchmarks, this paradigm not only attains superior asymptotic returns but also elevates internal simulator fidelity by establishing a tighter, lower variance alignment between imagined and real rewards, all while reducing latent phase space volume by 4.18-8.41%, energy consumption by up to 7.80%, and mean squared jerk by up to 9.38%.
>
---
## 更新

#### [replaced 001] CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D affordance任务，解决多对象中意图驱动的可操作性定位问题。构建了CompassAD基准，提出CompassNet框架，提升复杂场景下的物体识别与操作能力。**

- **链接: [https://arxiv.org/pdf/2604.02060](https://arxiv.org/pdf/2604.02060)**

> **作者:** Jingliang Li; Jindou Jia; Tuo An; Chuhao Zhou; Xiangyu Chen; Shilin Shan; Boyu Ma; Bofan Lyu; Gen Li; Jianfei Yang
>
> **摘要:** When told to "cut the cake," a robot must choose the knife over nearby scissors, despite both objects affording the same cutting function. In real-world scenes, multiple objects may share identical affordances, yet only one is appropriate under the given task context. We call such cases confusing pairs. However, existing 3D affordance methods largely sidestep this challenge by evaluating isolated single objects, often with explicit category names provided in the query. We formalize Intent-Driven Confusable Affordance Grounding, a new 3D affordance setting that requires predicting a per-point affordance mask on the correct object within a multi-object point cloud, conditioned on implicit natural language intent. To study this problem, we construct CompassAD, the first benchmark centered on implicit intent in confusing multi-object compositions. It comprises 30 confusing object pairs spanning 16 affordance types, 6,422 compositions, and 88K+ query-answer pairs. Furthermore, we propose CompassNet, a framework that incorporates two dedicated modules tailored to this task. Instance-bounded Cross Injection (ICI) constrains language-geometry alignment within object boundaries to prevent cross-object semantic leakage. Bi-level Contrastive Refinement (BCR) enforces discrimination at both geometric-group and point levels, sharpening distinctions between target and confusable surfaces. Extensive experiments demonstrate state-of-the-art results on both seen and unseen queries, and deployment on a robotic manipulator confirms effective transfer to real-world grasping in confusing multi-object compositions.
>
---
#### [replaced 002] DexWild: Dexterous Human Interactions for In-the-Wild Robot Policies
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; eess.SY**

- **简介: 该论文提出DexWild，用于提升机器人在复杂环境中的操作能力。任务是增强机器人泛化能力，解决数据获取成本高和泛化性差的问题。通过人类手部交互收集数据，并结合机器人数据训练，提升性能。**

- **链接: [https://arxiv.org/pdf/2505.07813](https://arxiv.org/pdf/2505.07813)**

> **作者:** Tony Tao; Mohan Kumar Srirama; Jason Jingzhou Liu; Kenneth Shaw; Deepak Pathak
>
> **备注:** In RSS 2025. Website at this https URL
>
> **摘要:** Large-scale, diverse robot datasets have emerged as a promising path toward enabling dexterous manipulation policies to generalize to novel environments, but acquiring such datasets presents many challenges. While teleoperation provides high-fidelity datasets, its high cost limits its scalability. Instead, what if people could use their own hands, just as they do in everyday life, to collect data? In DexWild, a diverse team of data collectors uses their hands to collect hours of interactions across a multitude of environments and objects. To record this data, we create DexWild-System, a low-cost, mobile, and easy-to-use device. The DexWild learning framework co-trains on both human and robot demonstrations, leading to improved performance compared to training on each dataset individually. This combination results in robust robot policies capable of generalizing to novel environments, tasks, and embodiments with minimal additional robot-specific data. Experimental results demonstrate that DexWild significantly improves performance, achieving a 68.5% success rate in unseen environments-nearly four times higher than policies trained with robot data only-and offering 5.8x better cross-embodiment generalization. Video results, codebases, and instructions at this https URL
>
---
#### [replaced 003] DECODE: Domain-aware Continual Domain Expansion for Motion Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于运动预测任务，解决自动驾驶中模型更新与适应新场景的问题。提出DECODE框架，实现持续学习与模型动态调整。**

- **链接: [https://arxiv.org/pdf/2411.17917](https://arxiv.org/pdf/2411.17917)**

> **作者:** Boqi Li; Haojie Zhu; Henry X. Liu
>
> **备注:** This work has been published in IEEE TPAMI Early Access
>
> **摘要:** Motion prediction is critical for autonomous vehicles to effectively navigate complex environments and accurately anticipate the behaviors of other traffic participants. As autonomous driving continues to evolve, the need to assimilate new and varied driving scenarios necessitates frequent model updates through retraining. To address these demands, we introduce DECODE, a novel continual learning framework that begins with a pre-trained generalized model and incrementally develops specialized models for distinct domains. Unlike existing continual learning approaches that attempt to develop a unified model capable of generalizing across diverse scenarios, DECODE uniquely balances specialization with generalization, dynamically adjusting to real-time demands. The proposed framework leverages a hypernetwork to generate model parameters, significantly reducing storage requirements, and incorporates a normalizing flow mechanism for real-time model selection based on likelihood estimation. Furthermore, DECODE merges outputs from the most relevant specialized and generalized models using deep Bayesian uncertainty estimation techniques. This integration ensures optimal performance in familiar conditions while maintaining robustness in unfamiliar scenarios. Extensive evaluations confirm the effectiveness of the framework, achieving a notably low forgetting rate of 0.044 and an average minADE of 0.584 m, significantly surpassing traditional learning strategies and demonstrating adaptability across a wide range of driving conditions.
>
---
#### [replaced 004] FASTER: Rethinking Real-Time Flow VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型的实时执行任务，旨在解决反应延迟问题。通过提出FASTER方法，优化动作采样策略，显著降低反应时间，提升机器人实时响应能力。**

- **链接: [https://arxiv.org/pdf/2603.19199](https://arxiv.org/pdf/2603.19199)**

> **作者:** Yuxiang Lu; Zhe Liu; Xianzhe Fan; Zhenya Yang; Jinghua Hou; Junyi Li; Kaixin Ding; Hengshuang Zhao
>
> **备注:** Project page: this https URL
>
> **摘要:** Real-time execution is crucial for deploying Vision-Language-Action (VLA) models in the physical world. Existing asynchronous inference methods primarily optimize trajectory smoothness, but neglect the critical latency in reacting to environmental changes. By rethinking the notion of reaction in action chunking policies, this paper presents a systematic analysis of the factors governing reaction time. We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon. Moreover, we reveal that the standard practice of applying a constant schedule in flow-based VLAs can be inefficient and forces the system to complete all sampling steps before any movement can start, forming the bottleneck in reaction latency. To overcome this issue, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER). By introducing a Horizon-Aware Schedule, FASTER adaptively prioritizes near-term actions during flow sampling, compressing the denoising of the immediate reaction by tenfold (e.g., in $\pi_{0.5}$ and X-VLA) into a single step, while preserving the quality of long-horizon trajectory. Coupled with a streaming client-server pipeline, FASTER substantially reduces the effective reaction latency on real robots, especially when deployed on consumer-grade GPUs. Real-world experiments, including a highly dynamic table tennis task, prove that FASTER unlocks substantially improved real-time responsiveness for generalist policies, enabling rapid generation of accurate and smooth trajectories.
>
---
#### [replaced 005] Weather-Robust Cross-View Geo-Localization via Prototype-Based Semantic Part Discovery
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于跨视角地理定位任务，解决无人机在无GNSS信号时的定位问题。提出SkyPart模型，通过语义部件发现提升定位鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.11654](https://arxiv.org/pdf/2605.11654)**

> **作者:** Chi-Nguyen Tran; Dao Sy Duy Minh; Huynh Trung Kiet; Nguyen Lam Phu Quy; Phu-Hoa Pham; Long Tran-Thanh
>
> **备注:** 37 pages, 7 figures, 6 tables
>
> **摘要:** Cross-view geo-localization (CVGL), which matches an oblique drone view to a geo-referenced satellite tile, has emerged as a key alternative for autonomous drone navigation when GNSS signals are jammed, spoofed, or unavailable. Despite strong recent progress, three limitations persist: (1) global-descriptor designs compress the patch grid into a single vector without separating layout from texture across the view gap; (2) altitude-related scale variation is retained in the learned embedding rather than marginalized; and (3) multi-objective training relies on hand-tuned scalars over losses on incompatible gradient scales. We propose SkyPart, a lightweight swappable head for patch-based vision transformers (ViTs) that institutes explicit part grouping over the patch grid. SkyPart has four theory-grounded components: (i) learnable prototypes competing for patch tokens via single-pass cosine assignment; (ii) altitude-conditioned linear modulation applied only during training, making the retrieval embedding altitude-free at inference; (iii) a graph-attention readout over active prototypes; and (iv) a Kendall uncertainty-weighted multi-objective loss whose stationary points are Pareto-stationary. At 26.95M parameters and 22.14 GFLOPs, SkyPart is the smallest among top-performing methods and sets a new state of the art on SUES-200, University-1652, and DenseUAV under a single-pass, no-re-ranking, no-TTA protocol. Its advantage over the strongest baseline widens under the ten-condition WeatherPrompt corruption benchmark.
>
---
#### [replaced 006] Therapist-Exoskeleton-Patient Interaction for Gait Therapy
- **分类: cs.RO**

- **简介: 该论文属于康复机器人任务，旨在解决传统康复训练效率低、 therapist负担重的问题。提出一种新型人机交互方式，通过外骨骼实现治疗师与患者的双向互动，提升康复效果。**

- **链接: [https://arxiv.org/pdf/2507.16059](https://arxiv.org/pdf/2507.16059)**

> **作者:** Emek Barış Küçüktabak; Matthew R. Short; Lorenzo Vianello; Daniel Ludvig; Levi Hargrove; Kevin Lynch; Jose Pons
>
> **摘要:** Following a stroke, individuals often experience mobility and balance impairments due to lower-limb weakness and loss of independent joint control. Gait recovery is a key goal of rehabilitation, traditionally achieved through high-intensity therapist-led training. However, manual assistance can be physically demanding and limits the therapist's ability to interact with multiple joints simultaneously. Robotic exoskeletons offer multi-joint support, reduce therapist strain, and provide objective feedback, but current control strategies often limit therapist involvement and adaptability. We present a novel gait rehabilitation paradigm based on physical Human-Robot-Human Interaction (pHRHI), where both the therapist and the post-stroke individual wear lower-limb exoskeletons virtually connected at the hips and knees via spring-damper elements. This enables bidirectional interaction, allowing the therapist to guide movement and receive haptic feedback. In a study with eight chronic stroke patients, pHRHI training outperformed conventional therapist-guided treadmill walking, leading to increased joint range of motion, step metrics, muscle activation, and motivation. These results highlight pHRHI's potential to combine robotic precision with therapist intuition for improved rehabilitation outcomes.
>
---
#### [replaced 007] COLSON: Controllable Learning-Based Social Navigation via Diffusion-Based Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于社会导航任务，解决动态环境中机器人自主导航问题。采用基于扩散的强化学习方法，提升动作灵活性并适应新场景。**

- **链接: [https://arxiv.org/pdf/2503.13934](https://arxiv.org/pdf/2503.13934)**

> **作者:** Kohei Matsumoto; Yuki Tomita; Yuki Hyodo; Ryo Kurazume
>
> **备注:** ICRA 2026
>
> **摘要:** Mobile robot navigation in dynamic environments with pedestrian traffic is a key challenge in the development of autonomous mobile service robots. Recently, deep reinforcement learning-based methods have been actively studied and have outperformed traditional rule-based approaches owing to their optimization capabilities. Among these methods, those that assume continuous action spaces typically rely on Gaussian distributions, which limit the flexibility of the generated actions. In contrast, the application of diffusion models to reinforcement learning has advanced, enabling more flexible action distributions than Gaussian policy-based approaches. In this study, we apply a diffusion-based reinforcement learning approach to social navigation and validate its effectiveness. Furthermore, by exploiting the characteristics of diffusion models, we propose extensions that enable adaptation to previously unseen scenarios without additional training. As concrete scenario examples, we demonstrate adaptability to scenarios in which static obstacles exist in the environment that were not present during training, as well as scenarios in which the objective differs from training, such as accompanying target pedestrians while avoiding others to reach the destination.
>
---
#### [replaced 008] PLATO Hand: Shaping Contact Behavior with Fingernails for Precise Manipulation
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出PLATO手，通过结构化接触界面解决精确操作问题。融合刚性指甲与柔性指腹，提升抓取稳定性与任务执行能力。**

- **链接: [https://arxiv.org/pdf/2602.05156](https://arxiv.org/pdf/2602.05156)**

> **作者:** Dong Ho Kang; Aaron Kim; Mingyo Seo; Kazuto Yokoyama; Tetsuya Narita; Luis Sentis
>
> **摘要:** We present the PLATO Hand, a dexterous robotic hand with a hybrid fingertip that combines a rigid fingernail, embedded distal phalanx, and compliant pulp to shape contact behavior during manipulation. \rrev{By mechanically organizing how contact is initiated, supported, and transmitted at the fingertip, this structure creates stable and task-relevant contact conditions across diverse object geometries and grasp orientations.} We develop a strain-energy-based bending--indentation model to guide the fingertip design and to explain how material stiffness and contact geometry govern deformation partitioning within the fingertip. \rrev{Experiments show improved pinch stability, improved fingernail-mediated dorsal-contact force transmission and proprioceptive observability}, and successful execution of edge-sensitive manipulation tasks, including paper singulation, card picking, and orange peeling. These results show that coupling a mechanically structured contact interface with a force-motion-transparent finger mechanism provides a principled approach to precise manipulation. Our project page is at: this https URL
>
---
#### [replaced 009] Geometry-aware 4D Video Generation for Robot Manipulation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于4D视频生成任务，旨在解决多视角下视频时空一致性和几何一致性问题。通过几何监督学习3D场景表示，生成多视角未来视频序列。**

- **链接: [https://arxiv.org/pdf/2507.01099](https://arxiv.org/pdf/2507.01099)**

> **作者:** Zeyi Liu; Shuang Li; Eric Cousineau; Siyuan Feng; Benjamin Burchfiel; Shuran Song
>
> **备注:** ICLR 2026; Project website: this https URL
>
> **摘要:** Understanding and predicting dynamics of the physical world can enhance a robot's ability to plan and interact effectively in complex environments. While recent video generation models have shown strong potential in modeling dynamic scenes, generating videos that are both temporally coherent and geometrically consistent across camera views remains a significant challenge. To address this, we propose a 4D video generation model that enforces multi-view 3D consistency of generated videos by supervising the model with cross-view pointmap alignment during training. Through this geometric supervision, the model learns a shared 3D scene representation, enabling it to generate spatio-temporally aligned future video sequences from novel viewpoints given a single RGB-D image per view, and without relying on camera poses as input. Compared to existing baselines, our method produces more visually stable and spatially aligned predictions across multiple simulated and real-world robotic datasets. We further show that the predicted 4D videos can be used to recover robot end-effector trajectories using an off-the-shelf 6DoF pose tracker, yielding robot manipulation policies that generalize well to novel camera viewpoints.
>
---
#### [replaced 010] DynoSLAM: Dynamic SLAM with Generative Graph Neural Networks for Real-World Social Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DynoSLAM，解决动态环境中机器人定位与建图问题，通过融合生成式图神经网络，提升复杂人群场景下的导航安全性与准确性。**

- **链接: [https://arxiv.org/pdf/2605.02759](https://arxiv.org/pdf/2605.02759)**

> **作者:** Danil Tokhchukov; Veronika Morozova; Gonzalo Ferrer
>
> **备注:** Code & Project page at this https URL
>
> **摘要:** Traditional Simultaneous Localization and Mapping (SLAM) algorithms rely heavily on the static environment assumption, which severely limits their applicability in real-world spaces populated by moving entities, such as pedestrians. In this work, we propose DynoSLAM, a tightly-coupled Dynamic GraphSLAM architecture that integrates socially-aware Graph Neural Networks (GNNs) directly into the factor graph optimization. Unlike conventional approaches that use rigid constant-velocity heuristics or deterministic single-agent neural priors, our framework formulates pedestrian motion forecasting as a stochastic World Model. By utilizing Monte Carlo rollouts from a trained GNN, we capture the multimodal epistemic uncertainty of human interactions and embed it into the SLAM graph via a dynamic Mahalanobis distance factor. We demonstrate through extensive simulated experiments that this stochastic formulation not only maintains highly accurate retrospective tracking but also prevents the optimization failures caused by the deterministic "argmax problem". Ultimately, extracting the empirical mean and covariance matrices of future pedestrian states provides a mathematically rigorous, probabilistic safety envelope for downstream local planners, enabling anticipatory and collision-free robot navigation in densely crowded environments.
>
---
#### [replaced 011] InFeR: Informed Failure Resilience in Learned Visual Navigation Control
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决IL策略在OOD场景下的故障问题。提出InFeR框架，通过VIB和Grad-CAM实现故障检测与恢复，无需额外数据。**

- **链接: [https://arxiv.org/pdf/2510.24680](https://arxiv.org/pdf/2510.24680)**

> **作者:** Zishuo Wang; Joel Loo; David Hsu
>
> **摘要:** While imitation learning (IL) has enabled successful visual navigation in many common environments, IL policies are prone to unpredictable failures under out-of-distribution (OOD) scenarios. This necessitates failure-resilient policies, which not only detect failures, but also recognise their sources and recover from them autonomously. We propose InFeR, a general framework for building IL policies with informed failure resilience without failure or recovery demonstrations. InFeR retrains an IL policy with a Variational Information Bottleneck (VIB) loss to structure its latent space for OOD failure detection. It applies a visual explainability technique, Grad-CAM, to localise an image region as the source of failure and inform a heuristic policy for recovery. All these are achieved without requiring additional training data. Real-world experiments show that InFeR enables informed failure recovery across two different policy architectures, yielding robust long-range navigation in complex environments.
>
---
#### [replaced 012] Real2Sim via Active Perception with Behavior Trees Automatically Generated by VLMs
- **分类: cs.RO**

- **简介: 该论文提出一种自主的Real2Sim框架，利用VLM生成行为树，解决物理仿真参数缺失问题，提升效率与安全性。**

- **链接: [https://arxiv.org/pdf/2601.08454](https://arxiv.org/pdf/2601.08454)**

> **作者:** Alessandro Adami; Sebastian Zudaire; Ruggero Carli; Pietro Falco
>
> **摘要:** Constructing physically accurate simulation environments (Real2Sim) traditionally relies on manual system identification or rigid, exhaustive exploration routines. These task-agnostic pipelines often fail to leverage semantic scene context, leading to redundant physical interactions and inefficient data acquisition. In this paper, we present an autonomous, intent-driven Real2Sim framework that leverages Vision-Language Models (VLMs) for Semantic Task Decomposition. Given a high-level natural language request, an incomplete simulation description, and a visual observation, the framework autonomously identifies the minimal subset of missing physical parameters required for the simulation task. It then generates a reactive Behavior Tree (BT) composed of atomic motion and sensing primitives to selectively acquire these parameters through contact-rich robotic interaction. Extensive real-world experiments on a torque-controlled Franka Emika Panda demonstrate that our approach accurately estimates object mass, surface geometry, and derived parameters such as friction. Quantitative evaluations reveal significant operational efficiency gains compared to exhaustive baseline methods, while ablation studies confirm the robustness of the prompt architecture across different state-of-the-art VLMs. Furthermore, the reactive hierarchy of the BT acts as a deterministic safety filter, successfully mitigating generative VLM hallucinations and preventing unsafe physical anomalies. Ultimately, this work provides a scalable, efficient, and interpretable pipeline for building physics-aware digital twins directly from unstructured human intent.
>
---
#### [replaced 013] FUNCanon: Learning Pose-Aware Action Primitives via Functional Object Canonicalization for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决技能泛化不足的问题。通过功能物体归一化和动作片段学习，提升策略的可组合性和跨任务适应能力。**

- **链接: [https://arxiv.org/pdf/2509.19102](https://arxiv.org/pdf/2509.19102)**

> **作者:** Hongli Xu; Lei Zhang; Xiaoyue Hu; Boyang Zhong; Kaixin Bai; Zoltán-Csaba Márton; Zhenshan Bing; Zhaopeng Chen; Alois Christian Knoll; Jianwei Zhang
>
> **备注:** project website: this https URL, 11 pages
>
> **摘要:** General-purpose robotic skills from end-to-end demonstrations often leads to task-specific policies that fail to generalize beyond the training distribution. Therefore, we introduce FunCanon, a framework that converts long-horizon manipulation tasks into sequences of action chunks, each defined by an actor, verb, and object. These chunks focus policy learning on the actions themselves, rather than isolated tasks, enabling compositionality and reuse. To make policies pose-aware and category-general, we perform functional object canonicalization for functional alignment and automatic manipulation trajectory transfer, mapping objects into shared functional frames using affordance cues from large vision language models. An object centric and action centric diffusion policy FuncDiffuser trained on this aligned data naturally respects object affordances and poses, simplifying learning and improving generalization ability. Experiments on simulated and real-world benchmarks demonstrate category-level generalization, cross-task behavior reuse, and robust sim2real deployment, showing that functional canonicalization provides a strong inductive bias for scalable imitation learning in complex manipulation domains. Details of the demo and supplemental material are available on our project website this https URL.
>
---
#### [replaced 014] Bio-Inspired Event-Based Visual Servoing for Ground Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉伺服任务，解决地面机器人在结构化环境中高效感知与控制问题。通过生物启发的事件触发机制，实现低延迟、高效率的状态反馈控制。**

- **链接: [https://arxiv.org/pdf/2603.23672](https://arxiv.org/pdf/2603.23672)**

> **作者:** Maral Mordad; Kian Behzad; Debojyoti Biswas; Noah J. Cowan; Milad Siami
>
> **摘要:** Biological sensory systems are inherently adaptive, filtering out constant stimuli and prioritizing relative changes, likely enhancing computational and metabolic efficiency. Inspired by active sensing behaviors across a wide range of animals, this paper introduces a principled 1D event-based visual servoing framework for ground robots operating in structured environments. Utilizing a Dynamic Vision Sensor (DVS), we demonstrate that by applying a fixed spatial kernel to the asynchronous event stream generated from structured logarithmic intensity-change patterns, the resulting net event flux analytically isolates specific combinations of kinematic states. We establish a generalized theoretical bound for this event rate estimator and show that linear and quadratic spatial profiles isolate the robot's velocity and position-velocity product, respectively. Leveraging these properties, we employ a multi-pattern stimulus to directly synthesize a nonlinear state feedback term entirely without traditional state estimation. To overcome the inescapable loss of linear observability at equilibrium inherent in event sensing, we propose a bio-inspired active sensing limit-cycle controller. Experimental validation on a 1/10-scale autonomous ground vehicle confirms the efficacy, extreme low-latency, and computational efficiency of the proposed direct-sensing approach.
>
---
#### [replaced 015] LiPS: Lightweight Panoptic Segmentation for Resource-Constrained Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉感知任务，解决资源受限机器人中高效全景分割的问题。提出轻量级模型LiPS，在保持性能的同时降低计算需求。**

- **链接: [https://arxiv.org/pdf/2604.00634](https://arxiv.org/pdf/2604.00634)**

> **作者:** Calvin Galagain; Martyna Poreba; François Goulette; Cyrill Stachniss
>
> **备注:** Accepted to IEEE International Conference on Image Processing (ICIP) 2026, Paper #2070
>
> **摘要:** Panoptic segmentation is a key enabler for robotic perception, as it unifies semantic understanding with object-level reasoning. However, the increasing complexity of state-of-the-art models makes them unsuitable for deployment on resource-constrained platforms such as mobile robots. We propose a novel approach called LiPS that addresses the challenge of efficient-to-compute panoptic segmentation with a lightweight design that retains query-based decoding while introducing a streamlined feature extraction and fusion pathway. It aims at providing a strong panoptic segmentation performance while substantially lowering the computational demands. Evaluations on standard benchmarks demonstrate that LiPS attains accuracy comparable to much heavier baselines, while providing up to 4.5 higher throughput, measured in frames per second, and requiring nearly 6.8 times fewer computations. This efficiency makes LiPS a highly relevant bridge between modern panoptic models and real-world robotic applications.
>
---
#### [replaced 016] SonarSweep: Fusing Sonar and Vision for Robust 3D Reconstruction via Plane Sweeping
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于3D重建任务，旨在解决水下视觉退化环境中的深度估计问题。通过融合声呐与视觉数据，提出SonarSweep框架，提升重建精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.00392](https://arxiv.org/pdf/2511.00392)**

> **作者:** Lingpeng Chen; Jiakun Tang; Apple Pui-Yi Chui; Ziyang Hong; Junfeng Wu
>
> **备注:** 8 pages, 9 figures, conference
>
> **摘要:** Accurate 3D reconstruction in visually-degraded underwater environments remains a formidable challenge. Single-modality approaches are insufficient: vision-based methods fail due to poor visibility and geometric constraints, while sonar is crippled by inherent elevation ambiguity and low resolution. Consequently, prior fusion technique relies on heuristics and flawed geometric assumptions, leading to significant artifacts and an inability to model complex scenes. In this paper, we introduce SonarSweep, a novel, end-to-end deep learning framework that overcomes these limitations by adapting the principled plane sweep algorithm for cross-modal fusion between sonar and visual data. Extensive experiments in both high-fidelity simulation and real-world environments demonstrate that SonarSweep consistently generates dense and accurate depth maps, significantly outperforming state-of-the-art methods across challenging conditions, particularly in high turbidity. To foster further research, we will publicly release our code and a novel dataset featuring synchronized stereo-camera and sonar data, the first of its kind.
>
---
#### [replaced 017] State-Conditional Adversarial Learning: An Off-Policy Visual Domain Transfer Method for End-to-End Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于模仿学习中的视觉域迁移任务，解决目标域数据稀缺且非策略的问题。提出SCAL方法，通过状态条件对抗学习实现高效迁移。**

- **链接: [https://arxiv.org/pdf/2512.05335](https://arxiv.org/pdf/2512.05335)**

> **作者:** Yuxiang Liu; Shengfan Cao
>
> **摘要:** We study visual domain transfer for end-to-end imitation learning in a realistic and challenging setting where target-domain data are strictly off-policy, expert-free, and scarce. We first provide a theoretical analysis showing that the target-domain imitation loss can be upper bounded by the source-domain loss plus a state-conditional latent KL divergence between source and target observation models. Guided by this result, we propose State- Conditional Adversarial Learning, an off-policy adversarial framework that aligns latent distributions conditioned on system state using a discriminator-based estimator of the conditional KL term. Experiments on visually diverse autonomous driving environments built on the BARC-CARLA simulator demonstrate that SCAL achieves robust transfer and strong sample efficiency.
>
---
#### [replaced 018] Motion Planning of Cooperative Nonholonomic Mobile Manipulators
- **分类: cs.RO; cs.MA; eess.SY; math.OC**

- **简介: 该论文属于协作移动机械臂的运动规划任务，解决动态环境中物体协同运输的路径规划问题，提出一种基于NMPC的实时规划方法。**

- **链接: [https://arxiv.org/pdf/2502.05462](https://arxiv.org/pdf/2502.05462)**

> **作者:** Keshab Patra; Arpita Sinha; Anirban Guha
>
> **备注:** Published in ASME Letters in Translational Robotics. This includes supplementary materials
>
> **摘要:** We propose a real-time implementable motion planning framework for cooperative object transportation by nonholonomic mobile manipulator robots (MMRs) in dynamic environments. Our global planner finds a path from start to goal through the static, obstacle-free regions in the environment and generates a set of convex, static, obstacle-free regions around the path using a novel, fast, and computationally lightweight ellipse-based technique. We introduce a nonlinear Model Predictive Control (NMPC) based real-time implementable planning technique that jointly plans feasible motion for the mobile base and the manipulator's arm and generates a kinodynamic feasible, collision-free trajectory for cooperative object transportation. Simulation and hardware experiments validate the efficiency of our proposed planning framework.
>
---
#### [replaced 019] Gesture First, LLM-Assisted Voice Complement: Exploring Multimodal Robot 'Puppeteer' Teleoperation Via Virtual Counterpart in Augmented Reality
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究AR辅助的机器人遥控任务，探讨手势与语音结合的交互方式。旨在提升人机交互的直观性，通过实验比较不同交互模式的效率与用户体验。**

- **链接: [https://arxiv.org/pdf/2506.13189](https://arxiv.org/pdf/2506.13189)**

> **作者:** Yuchong Zhang; Bastian Orthmann; Shichen Ji; Michael Welle; Jonne Van Haastregt; Danica Kragic
>
> **备注:** This work is under peer review
>
> **摘要:** Robot teleoperation via augmented reality (AR) offers a promising path toward more intuitive human-robot interaction (HRI). We present a head-mounted AR 'puppeteer' system in which users control a physical robot by interacting with its virtual counterpart robot using large language model (LLM)-assisted voice commands and hand-gesture interaction on the Meta Quest 3. In a within-subject user study with 42 participants performing an AR-based robotic pick-and-place pattern-matching task, we empirically compare two interaction conditions: gesture-only (GO) and combined voice+gesture (VG) on performance and user experience (UX). In VG, voice and gesture operate in a sequential role-allocated manner, with voice handling high-level navigation and gesture handling fine manipulation. Our results show that GO currently provides more reliable and efficient control for this time-critical task, while VG introduces additional flexibility but also latency and recognition issues that can increase workload. We additionally analyze how prior robotics expertise differentiates performance and UX across conditions. Based on these findings, we distill a set of design guidelines for AR 'puppeteer' metaphoric robot teleoperation, framing multimodality as an adaptive strategy that must balance efficiency, robustness, and user expertise rather than assuming that additional modalities are universally beneficial.
>
---
#### [replaced 020] GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics
- **分类: cs.RO**

- **简介: 该论文提出GRaD-Nav++，解决无人机在非结构化环境中基于自然语言指令的自主导航问题。通过轻量VLA框架和DiffRL训练，实现高效、实时的视觉-语言-动作控制。**

- **链接: [https://arxiv.org/pdf/2506.14009](https://arxiv.org/pdf/2506.14009)**

> **作者:** Qianzhong Chen; Naixiang Gao; Suning Huang; JunEn Low; Timothy Chen; Jiankai Sun; Mac Schwager
>
> **备注:** Published in: IEEE Robotics and Automation Letters ( Volume: 11, Issue: 2, February 2026)
>
> **摘要:** Autonomous drones capable of interpreting and executing high-level language instructions in unstructured environments remain a long-standing goal. Yet existing approaches are constrained by their dependence on hand-crafted skills, extensive parameter tuning, or computationally intensive models unsuitable for onboard use. We introduce GRaD-Nav++, a lightweight Vision-Language-Action (VLA) framework that runs fully onboard and follows natural-language commands in real time. Our policy is trained in a photorealistic 3D Gaussian Splatting (3DGS) simulator via Differentiable Reinforcement Learning (DiffRL), enabling efficient learning of low-level control from visual and linguistic inputs. At its core is a Mixture-of-Experts (MoE) action head, which adaptively routes computation to improve generalization while mitigating forgetting. In multi-task generalization experiments, GRaD-Nav++ achieves a success rate of 83% on trained tasks and 75% on unseen tasks in simulation. When deployed on real hardware, it attains 67% success on trained tasks and 50% on unseen ones. In multi-environment adaptation experiments, GRaD-Nav++ achieves an average success rate of 81% across diverse simulated environments and 67% across varied real-world settings. These results establish a new benchmark for fully onboard Vision-Language-Action (VLA) flight and demonstrate that compact, efficient models can enable reliable, language-guided navigation without relying on external infrastructure.
>
---
#### [replaced 021] SuReNav: Superpixel Graph-based Constraint Relaxation for Navigation in Over-constrained Environments
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于路径规划任务，解决过约束环境下的导航问题。提出SuReNav方法，通过超像素图和神经网络实现安全高效导航，提升人机一致性。**

- **链接: [https://arxiv.org/pdf/2602.06807](https://arxiv.org/pdf/2602.06807)**

> **作者:** Keonyoung Koh; Moonkyeong Jung; Samuel Seungsup Lee; Daehyung Park
>
> **备注:** Accepted by ICRA 2026. Code and videos are available at this https URL
>
> **摘要:** We address the over-constrained planning problem in semi-static environments. The planning objective is to find a best-effort solution that avoids all hard constraint regions while minimally traversing the least risky areas. Conventional methods often rely on pre-defined area costs, limiting generalizations. Further, the spatial continuity of navigation spaces makes it difficult to identify regions that are passable without overestimation. To overcome these challenges, we propose SuReNav, a superpixel graph-based constraint relaxation and navigation method that imitates human-like safe and efficient navigation. Our framework consists of three components: 1) superpixel graph map generation with regional constraints, 2) regional-constraint relaxation using graph neural network trained on human demonstrations for safe and efficient navigation, and 3) interleaving relaxation, planning, and execution for complete navigation. We evaluate our method against state-of-the-art baselines on 2D semantic maps and 3D maps from OpenStreetMap, achieving the highest human-likeness score of complete navigation while maintaining a balanced trade-off between efficiency and safety. We finally demonstrate its scalability and generalization performance in real-world urban navigation with a quadruped robot, Spot. Code and Videos are available at this https URL.
>
---
#### [replaced 022] Do Robots Really Need Anthropomorphic Hands? A Comparison of Human and Robotic Hands
- **分类: cs.RO**

- **简介: 论文探讨机器人是否需要仿人手，分析了人类与机器手的差异。研究比较了手部设计与操作能力的关系，旨在解决机器人手设计优化问题。**

- **链接: [https://arxiv.org/pdf/2508.05415](https://arxiv.org/pdf/2508.05415)**

> **作者:** Alexander Fabisch; Wadhah Zai El Amri; Chandandeep Singh; Nicolás Navarro-Guerrero
>
> **摘要:** Human manipulation skills represent a pinnacle of their voluntary motor functions, requiring the coordination of many degrees of freedom and processing of high-dimensional sensor input to achieve remarkable dexterity. Thus, we set out to answer whether the human hand, with its associated biomechanical properties, sensors, and control mechanisms, is an ideal that we should strive for in robotics. Do robots need anthropomorphic hands? We start by extracting characteristics of the human hand in terms of biomechanics and perception to compare them with currently commercially available robotic hands. From this comparison, we derive our research questions that connect manipulation system complexity to skill repertoire size and dexterity. We attempt to answer these with a systematic literature review, in which we analyze the manipulation capabilities demonstrated in 125 papers from 2019-2025. Although complex five-fingered hands are often considered the ultimate goal for robotic manipulators, they are not necessary for all tasks. We find that in-hand manipulation does not benefit from anthropomorphic hand design as simpler mechanisms are sufficient, but mechanism complexity correlates with the breadth of manipulation tasks a hand can perform. Sensor integration and intelligent manipulation strategies remain underexplored, which may be because of a misalignment with hand design: instead of replicating the number of fingers and degrees of freedom, focusing on robustness and softness would allow more intelligent control and learning to exploit environmental contacts and integrate more sensors. Finally, we argue for standardized evaluation criteria to enable systematic comparison of hand designs and manipulation systems.
>
---
#### [replaced 023] Teaching Robots to Interpret Social Interactions through Lexically-guided Dynamic Graph Learning
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于社会交互理解任务，旨在让机器人解析用户行为并预测其未来动作。工作包括提出SocialLDG框架，通过动态图学习建模状态间关系，提升任务可扩展性与解释性。**

- **链接: [https://arxiv.org/pdf/2604.10895](https://arxiv.org/pdf/2604.10895)**

> **作者:** Tongfei Bian; Mathieu Chollet; Tanaya Guha
>
> **备注:** submitted to ACM MM 26
>
> **摘要:** For a robot to be called socially intelligent, it must be able to infer users internal states from their current behaviour, predict the users future behaviour, and if required, respond appropriately. In this work, we investigate how robots can be endowed with such social intelligence by modelling the dynamic relationship between user's internal states (latent) and actions (observable state). Our premise is that these states arise from the same underlying socio-cognitive process and influence each other dynamically. Drawing inspiration from theories in Cognitive Science, we propose a novel multi-task learning framework, termed as \textbf{SocialLDG} that explicitly models the dynamic relationship among the states represent as six distinct tasks. Our framework uses a language model to introduce lexical priors for each task and employs dynamic graph learning to model task affinity evolving with time. SocialLDG has three advantages: First, it achieves state-of-the-art performance on two challenging human-robot social interaction datasets available publicly. Second, it supports strong task scalability by learning new tasks seamlessly without catastrophic forgetting. Finally, benefiting from explicit modelling task affinity, it offers insights on how different interactions unfolds in time and how the internal states and observable actions influence each other in human decision making.
>
---
#### [replaced 024] General-purpose LLMs as Models of Human Driver Behavior: The Case of Simplified Merging
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶领域，旨在评估通用大语言模型作为人类驾驶员行为模型的适用性。通过实验对比分析，探讨其在模拟驾驶行为中的表现与局限。**

- **链接: [https://arxiv.org/pdf/2604.09609](https://arxiv.org/pdf/2604.09609)**

> **作者:** Samir H.A. Mohammad; Wouter Mooi; Arkady Zgonnikov
>
> **备注:** To be published in proceedings of IEEE ITSC 2026
>
> **摘要:** Human behavior models are essential as behavior references and for simulating human agents in virtual safety assessment of automated vehicles (AVs), yet current models face a trade-off between interpretability and flexibility. General-purpose large language models (LLMs) offer a promising alternative: a single model potentially deployable without parameter fitting across diverse scenarios. However, what LLMs can and cannot capture about human driving behavior remains poorly understood. We address this gap by embedding two general-purpose LLMs (OpenAI o3 and Google Gemini 2.5 Pro) as standalone, closed-loop driver agents in a simplified one-dimensional merging scenario and comparing their behavior against human data using quantitative and qualitative analyses. Both models reproduce human-like intermittent operational control and tactical dependencies on spatial cues. However, neither consistently captures the human response to dynamic velocity cues, and safety performance diverges sharply between models. A systematic prompt ablation study reveals that prompt components act as model-specific inductive biases that do not transfer across LLMs. These findings suggest that general-purpose LLMs could potentially serve as standalone, ready-to-use human behavior models in AV evaluation pipelines, but future research is needed to better understand their failure modes and ensure their validity as models of human driving behavior.
>
---
#### [replaced 025] AT-VLA: Adaptive Tactile Injection for Enhanced Feedback Reaction in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出AT-VLA模型，解决VLA在接触密集操作中反馈反应慢和多模态干扰问题，通过自适应触觉注入和双流机制提升实时性与精度。**

- **链接: [https://arxiv.org/pdf/2605.07308](https://arxiv.org/pdf/2605.07308)**

> **作者:** Xiaoqi Li; Muhe Cai; Jiadong Xu; Juan Zhu; Hongwei Fan; Yan Shen; Guangrui Ren; Hao Dong
>
> **摘要:** Vision-Language-Action (VLA) models have significantly advanced the capabilities of robotic agents in executing diverse tasks; however, they still face challenges in contact-rich manipulation scenarios that require precise physical interactions. To address this limitation, recent studies have attempted to incorporate tactile signals during downstream tasks, enabling pretrained VLAs to interpret tactile feedback. Nevertheless, introducing new modalities during finetuning, which are rarely present in the pretrain stage, may disrupt the pretrained capabilities of VLAs. In addition, the inherently slow inference speed of VLAs hampers real-time responsiveness and limits the effective utilization of tactile feedback for action adjustment. To overcome these challenges, we propose Adaptive Tactile Vision-Language-Action (AT-VLA), which introduces a novel Adaptive Tactile Injection mechanism. This mechanism dynamically determines the appropriate timing and locations for tactile injection, incorporating only when it significantly contributes to action generation, thereby minimizing interference with pretrained representations. Furthermore, to enable rapid and accurate tactile responses, we propose a Tactile Reaction Dual-Stream mechanism, which decouples sensory processing into a slow visual-language stream for low-frequency perceptual reasoning and a fast tactile control stream for high-frequency physical interaction understanding, achieving real-time close-loop responses within 0.04 s. Real-world experiments thoroughly validate the effectiveness of AT-VLA in contact-rich manipulation tasks. The project page is available at: this https URL.
>
---
#### [replaced 026] CoLA-Flow Policy: Temporally Coherent Imitation Learning via Continuous Latent Action Flow Matching for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决长时序控制的稳定性与效率问题。提出CoLA-Flow Policy，通过连续潜在动作空间实现高效、平滑的轨迹模仿学习。**

- **链接: [https://arxiv.org/pdf/2601.23087](https://arxiv.org/pdf/2601.23087)**

> **作者:** Wu Songwei; Jiang Zhiduo; Sun Wandong; Xie Guanghu; Zhao Rui; Liu Hong; Liu Yang
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Learning long-horizon robotic manipulation requires jointly achieving expressive behavior modeling, real-time inference, and stable execution, which remains challenging for existing generative policies. Diffusion-based approaches offer strong modeling capacity but incur high inference latency, while flow matching enables fast, near-single-step generation yet often suffers from unstable execution when operating directly in the raw action space. We propose Continuous Latent Action Flow Policy (CoLA-Flow Policy), a trajectory-level imitation learning framework that performs flow matching in a continuous latent action space. By encoding action sequences into temporally coherent latent trajectories and learning an explicit latent-space flow, CoLA-Flow Policy decouples global motion structure from low-level control noise, enabling smooth and reliable long-horizon execution. The framework further integrates geometry-aware point cloud conditioning and execution-time multimodal modulation, using visual cues as a representative modality to enhance real-world robustness. Experiments in simulation and on real robots show that CoLA-Flow Policy achieves near-single-step inference, improves trajectory smoothness by up to 93.7% and task success by up to 25 percentage points over raw action-space flow baselines, while remaining significantly faster than diffusion-based policies.
>
---
#### [replaced 027] Learning Native Continuation for Action Chunking Flow Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉语言动作任务，解决动作分块执行中的连续性问题。提出Legato方法，在训练时引入延续机制，提升轨迹平滑度和任务效率。**

- **链接: [https://arxiv.org/pdf/2602.12978](https://arxiv.org/pdf/2602.12978)**

> **作者:** Yufeng Liu; Hang Yu; Juntu Zhao; Bocheng Li; Di Zhang; Mingzhu Li; Wenxuan Wu; Yingdong Hu; Junyuan Xie; Junliang Guo; Dequan Wang; Yang Gao
>
> **备注:** Accepted by Robotics: Science and Systems 2026 (RSS 2026). Project page: this https URL
>
> **摘要:** Action chunking enables Vision Language Action (VLA) models to run in real time, but naive chunked execution often exhibits discontinuities at chunk boundaries. Real-Time Chunking (RTC) alleviates this issue but is external to the policy, leading to spurious multimodal switching and trajectories that are not intrinsically smooth. We propose Legato, a training-time continuation method for action-chunked flow-based VLA policies. Specifically, Legato initializes denoising from a schedule-shaped mixture of known actions and noise, exposing the model to partial action information. Moreover, Legato reshapes the learned flow dynamics to ensure that the denoising process remains consistent between training and inference under per-step guidance. Legato further uses randomized schedule condition during training to support varying inference delays and achieve controllable smoothness. Empirically, Legato produces smoother trajectories and reduces spurious multimodal switching during execution, leading to less hesitation and shorter task completion time. Extensive real-world experiments show that Legato consistently outperforms RTC across five manipulation tasks, achieving approximately 10% improvements in both trajectory smoothness and task completion time.
>
---
#### [replaced 028] DriveMoE: Mixture-of-Experts for Vision-Language-Action Model in End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于端到端自动驾驶任务，旨在解决多视角数据处理和复杂场景应对问题。通过引入视觉和动作专家混合模型，提升模型性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.16278](https://arxiv.org/pdf/2505.16278)**

> **作者:** Zhenjie Yang; Yilin Chai; Xiaosong Jia; Qifeng Li; Yuqian Shao; Xuekai Zhu; Haisheng Su; Junchi Yan
>
> **备注:** Accepted by CVPR 2026, Project Page: this https URL
>
> **摘要:** End-to-end autonomous driving (E2E-AD) demands effective processing of multi-view sensory data and robust handling of diverse and complex driving scenarios, particularly rare maneuvers such as aggressive turns. Recent success of Mixture-of-Experts (MoE) architecture in Large Language Models (LLMs) demonstrates that specialization of parameters enables strong scalability. In this work, we propose DriveMoE, a novel MoE-based E2E-AD framework, with a Scene-Specialized Vision MoE and a Skill-Specialized Action MoE. DriveMoE is built upon our $\pi_0$ Vision-Language-Action (VLA) baseline (originally from the embodied AI field), called Drive-$\pi_0$. Specifically, we add Vision MoE to Drive-$\pi_0$ by training a router to select relevant cameras according to the driving context dynamically. This design mirrors human driving cognition, where drivers selectively attend to crucial visual cues rather than exhaustively processing all visual information. In addition, we add Action MoE by training another router to activate specialized expert modules for different driving behaviors. Through explicit behavioral specialization, DriveMoE is able to handle diverse scenarios without suffering from modes averaging like existing models. In Bench2Drive closed-loop evaluation experiments, DriveMoE achieves state-of-the-art (SOTA) performance, demonstrating the effectiveness of combining vision and action MoE in autonomous driving tasks. We will release our code and models of DriveMoE and Drive-$\pi_0$.
>
---
#### [replaced 029] What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?
- **分类: cs.AI; cs.LG; cs.RO; stat.ML**

- **简介: 该论文研究物理任务规划中的联合嵌入预测世界模型（JEPA-WMs），旨在提升规划效率与泛化能力。通过分析模型结构、训练目标和规划算法，提出更优方法，在导航和操作任务中优于现有基准。**

- **链接: [https://arxiv.org/pdf/2512.24497](https://arxiv.org/pdf/2512.24497)**

> **作者:** Basile Terver; Tsung-Yen Yang; Jean Ponce; Adrien Bardes; Yann LeCun
>
> **备注:** V2 of the article: - Added AdaLN-zero - Added table comparing JEPA-WMs with baselines with std translating per-seed variability only, no variability across epochs - Reordered figures in main body of the paper V3: added data scaling experiments, theoretical appendix section on autoregressive rollout, acceptance at TMLR
>
> **摘要:** A long-standing challenge in AI is to develop agents capable of solving a wide range of physical tasks and generalizing to new, unseen tasks and environments. A popular recent approach involves training a world model from state-action trajectories and subsequently use it with a planning algorithm to solve new tasks. Planning is commonly performed in the input space, but a recent family of methods has introduced planning algorithms that optimize in the learned representation space of the world model, with the promise that abstracting irrelevant details yields more efficient planning. In this work, we characterize models from this family as JEPA-WMs and investigate the technical choices that make algorithms from this class work. We propose a comprehensive study of several key components with the objective of finding the optimal approach within the family. We conducted experiments using both simulated environments and real-world robotic data, and studied how the model architecture, the training objective, and the planning algorithm affect planning success. We combine our findings to propose a model that outperforms two established baselines, DINO-WM and V-JEPA-2-AC, in both navigation and manipulation tasks. Code, data and checkpoints are available at this https URL.
>
---
#### [replaced 030] Bundle Adjustment in the Eager Mode
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，解决传统BA库与深度学习框架不兼容的问题，提出一种与PyTorch无缝集成的高效GPU加速BA方法。**

- **链接: [https://arxiv.org/pdf/2409.12190](https://arxiv.org/pdf/2409.12190)**

> **作者:** Zitong Zhan; Huan Xu; Zihang Fang; Xinpeng Wei; Yaoyu Hu; Chen Wang
>
> **摘要:** Bundle adjustment (BA) is a critical technique in various robotic applications such as simultaneous localization and mapping (SLAM), augmented reality (AR), and photogrammetry. BA optimizes parameters such as camera poses and 3D landmarks to align them with observations. With the growing importance of deep learning in perception systems, there is an increasing need to integrate BA with deep learning frameworks for enhanced reliability and performance. However, widely-used C++-based BA libraries, such as GTSAM, g$^2$o, and Ceres Solver, lack native integration with modern deep learning libraries like PyTorch. This limitation affects their flexibility, ease of debugging, and overall implementation efficiency. To address this gap, we introduce an eager-mode BA library seamlessly integrated with PyTorch with high efficiency. Our approach includes a sparsity-aware auto-differentiation design and GPU-accelerated sparse operations designed for 2nd-order optimization. Our eager-mode BA on GPU demonstrates substantial runtime efficiency, achieving an average speedup of 18.5$\times$, 22$\times$, and 23$\times$ across all benchmarks compared to GTSAM, g$^2$o, and Ceres, respectively.
>
---
#### [replaced 031] EvoQRE: Modeling Bounded Rationality in Safety-Critical Traffic Simulation via Evolutionary Quantal Response Equilibrium
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于交通仿真任务，旨在解决自动驾驶中人类驾驶员有限理性建模问题。提出EvoQRE框架，结合QRE与进化动态，提升仿真 realism 和安全性。**

- **链接: [https://arxiv.org/pdf/2601.05653](https://arxiv.org/pdf/2601.05653)**

> **作者:** Phu-Hoa Pham; Chi-Nguyen Tran; Duy-Minh Dao-Sy; Phu-Quy Nguyen-Lam; Trung-Kiet Huynh
>
> **备注:** This article is being withdrawn due to identified issues in the experimental evaluation and theoretical assumptions that may affect the validity of some reported conclusions. The authors plan to revise the methodology and provide a corrected version in future work.
>
> **摘要:** Existing traffic simulation frameworks for autonomous vehicles typically rely on imitation learning or game-theoretic approaches that solve for Nash or coarse correlated equilibria, implicitly assuming perfectly rational agents. However, human drivers exhibit bounded rationality, making approximately optimal decisions under cognitive and perceptual constraints. We propose EvoQRE, a principled framework for modeling safety-critical traffic interactions as general-sum Markov games solved via Quantal Response Equilibrium (QRE) and evolutionary game dynamics. EvoQRE integrates a pre-trained generative world model with entropy-regularized replicator dynamics, capturing stochastic human behavior while maintaining equilibrium structure. We provide rigorous theoretical results, proving that the proposed dynamics converge to Logit-QRE under a two-timescale stochastic approximation with an explicit convergence rate of O(log k / k^{1/3}) under weak monotonicity assumptions. We further extend QRE to continuous action spaces using mixture-based and energy-based policy representations. Experiments on the Waymo Open Motion Dataset and nuPlan benchmark demonstrate that EvoQRE achieves state-of-the-art realism, improved safety metrics, and controllable generation of diverse safety-critical scenarios through interpretable rationality parameters.
>
---
#### [replaced 032] One Hand to Rule Them All: Canonical Representations for Unified Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决灵活手抓取策略泛化问题。通过构建统一的规范表示，实现不同结构手的跨模态策略学习与迁移。**

- **链接: [https://arxiv.org/pdf/2602.16712](https://arxiv.org/pdf/2602.16712)**

> **作者:** Zhenyu Wei; Yunchao Yao; Mingyu Ding
>
> **备注:** Accepted at RSS 2026
>
> **摘要:** Dexterous manipulation policies today largely assume fixed hand designs, severely restricting their generalization to new embodiments with varied kinematic and structural layouts. To overcome this limitation, we introduce a parameterized canonical representation that unifies a broad spectrum of dexterous hand architectures. It comprises a unified parameter space and a canonical URDF format, offering three key advantages. 1) The parameter space captures essential morphological and kinematic variations for effective conditioning in learning algorithms. 2) A structured latent manifold can be learned over our space, where interpolations between embodiments yield smooth and physically meaningful morphology transitions. 3) The canonical URDF standardizes the action space while preserving dynamic and functional properties of the original URDFs, enabling efficient and reliable cross-embodiment policy learning. We validate these advantages through extensive analysis and experiments, including grasp policy replay, VAE latent encoding, and cross-embodiment zero-shot transfer. Specifically, we train a VAE on the unified representation to obtain a compact, semantically rich latent embedding, and develop a grasping policy conditioned on the canonical representation that generalizes across dexterous hands. We demonstrate, through simulation and real-world tasks on unseen morphologies (e.g., 81.9% zero-shot success rate on 3-finger LEAP Hand), that our framework unifies both the representational and action spaces of structurally diverse hands, providing a scalable foundation for cross-hand learning toward universal dexterous manipulation. Project Page: this https URL
>
---
#### [replaced 033] A Deployable Embodied Vision-Language Navigation System with Hierarchical Cognition and Context-Aware Exploration
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决机器人在资源受限下高效推理与实时执行的矛盾。提出分层系统，结合快速感知与深度推理，提升导航效率与成功率。**

- **链接: [https://arxiv.org/pdf/2604.21363](https://arxiv.org/pdf/2604.21363)**

> **作者:** Kuan Xu; Ruimeng Liu; Yizhuo Yang; Denan Liang; Tongxing Jin; Shenghai Yuan; Chen Wang; Lihua Xie
>
> **备注:** 10 pages, 5 figures,
>
> **摘要:** Bridging the gap between embodied intelligence and embedded deployment remains a key challenge in intelligent robotic systems, where perception, reasoning, and planning must operate under strict constraints on computation, memory, energy, and real-time execution. In vision-and-language navigation (VLN), existing approaches often face a trade-off between reasoning capability and deployment efficiency on real-world platforms. In this paper, we present a deployable embodied VLN system that achieves both high efficiency and strong high-level reasoning on real-world robots. The system is decomposed into a fast perception-action layer and a deep reasoning layer running asynchronously at different time scales, with a shared memory layer enabling efficient interaction between them. To support long-horizon reasoning, we incrementally construct a compact memory graph and progressively feed decomposed subgraphs into a vision-language model (VLM). Furthermore, we formulate exploration as a Weighted Traveling Repairman Problem (WTRP) by jointly considering reasoning outcomes and the spatial distribution of candidate regions. Extensive experiments in simulation and real-world environments demonstrate improved navigation success and efficiency over existing VLN approaches while maintaining real-time performance on resource-constrained hardware. Code and additional real-world experiments are available at this https URL.
>
---
#### [replaced 034] Action-Gradient Monte Carlo Tree Search for Non-Parametric Continuous (PO)MDPs
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于强化学习领域，解决连续状态和动作空间中的在线规划问题。提出AGMCTS框架，结合树搜索与梯度优化，提升求解效率与效果。**

- **链接: [https://arxiv.org/pdf/2503.12181](https://arxiv.org/pdf/2503.12181)**

> **作者:** Idan Lev-Yehudi; Michael Novitsky; Moran Barenboim; Ron Benchetrit; Vadim Indelman
>
> **摘要:** Online planning in continuous state, action, and observation spaces remains challenging for autonomous systems. While Monte Carlo Tree Search (MCTS) scales effectively via sampling, most continuous (PO)MDP solvers do not exploit gradient-based action optimization. We propose Action-Gradient MCTS (AGMCTS), a framework that combines global tree search with local gradient-based action refinement, while maintaining consistent value estimates. We provide three key theoretical contributions: (1) an action score gradient theorem for particle belief states; (2) the Multiple Importance Sampling (MIS) Tree that supports frequent action-branch updates by reusing prior samples without introducing estimator drift; and (3) tractable action score gradients for smooth generative models using the Area Formula. Empirical results demonstrate that AGMCTS outperforms state-of-the-art sample-based solvers in multiple challenging continuous MDP and POMDP benchmarks.
>
---
#### [replaced 035] Unleashing the Potential of Diffusion Models for End-to-End Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究扩散模型在端到端自动驾驶中的应用，解决真实场景下规划性能不足的问题。通过大量实车数据和测试，提出HDP框架，提升驾驶性能10倍。**

- **链接: [https://arxiv.org/pdf/2602.22801](https://arxiv.org/pdf/2602.22801)**

> **作者:** Yinan Zheng; Tianyi Tan; Bin Huang; Enguang Liu; Ruiming Liang; Jianlin Zhang; Jianwei Cui; Guang Chen; Kun Ma; Hangjun Ye; Long Chen; Ya-Qin Zhang; Xianyuan Zhan; Jingjing Liu
>
> **摘要:** Diffusion models have become a popular choice for decision-making tasks in robotics, and more recently, are also being considered for solving autonomous driving tasks. However, their applications and evaluations in autonomous driving remain limited to simulation-based or laboratory settings. The full strength of diffusion models for large-scale, complex real-world settings, such as End-to-End Autonomous Driving (E2E AD), remains underexplored. In this study, we conducted a systematic and large-scale investigation to unleash the potential of the diffusion models as planners for E2E AD, based on a tremendous amount of real-vehicle data and road testing. Through comprehensive and carefully controlled studies, we identify key insights into the diffusion loss space, trajectory representation, and data scaling that significantly impact E2E planning performance. Moreover, we also provide an effective reinforcement learning post-training strategy to further enhance the safety and robustness of the learned planner. The resulting diffusion-based learning framework, Hyper Diffusion Planner (HDP), is deployed on a real-vehicle platform and evaluated across 6 urban driving scenarios and 200 km of real-world testing, achieving a notable 10x performance improvement over the base model. Our work demonstrates that diffusion models, when properly designed and trained, can serve as effective and scalable E2E AD planners for complex, real-world autonomous driving tasks.
>
---
#### [replaced 036] Early Pruning for Public Transport Routing
- **分类: cs.DS; cs.AI; cs.RO**

- **简介: 该论文属于公共交通路径规划任务，解决转移阶段效率低的问题。通过早期剪枝技术，在不牺牲最优性的前提下提升算法效率。**

- **链接: [https://arxiv.org/pdf/2603.12592](https://arxiv.org/pdf/2603.12592)**

> **作者:** Andrii Rohovyi; Abdallah Abuaisha; Toby Walsh
>
> **摘要:** Routing algorithms for public transport, particularly the widely used RAPTOR and its variants, often face performance bottlenecks during the transfer relaxation phase, especially on dense transfer graphs, when supporting unlimited transfers. This inefficiency arises from iterating over many potential inter-stop connections (walks, bikes, e-scooters, etc.). To maintain acceptable performance, practitioners often limit transfer distances or exclude certain transfer options, which can reduce path optimality and restrict the multimodal options presented to travellers. This paper introduces Early Pruning, a low-overhead technique that accelerates routing algorithms without compromising optimality. By pre-sorting transfer connections by duration and applying a pruning rule within the transfer loop, the method discards longer transfers at a stop once they cannot yield an earlier arrival than the current best solution. Early Pruning can be integrated with minimal changes to existing codebases and requires only a one-time preprocessing step. The technique preserves Pareto-optimality in extended-criteria settings whenever the additional optimization criteria are monotonically non-decreasing in transfer duration. Across multiple state-of-the-art RAPTOR-based solutions, including RAPTOR, ULTRA-RAPTOR, McRAPTOR, BM-RAPTOR, ULTRA-McRAPTOR, and UBM-RAPTOR and tested on the Switzerland and London transit networks, we achieved query time reductions of up to 57\%. This approach provides a generalizable improvement to the efficiency of transit pathfinding algorithms.
>
---
#### [replaced 037] Real-to-Sim for Highly Cluttered Environments via Physics-Consistent Inter-Object Reasoning
- **分类: cs.RO**

- **简介: 该论文属于3D场景重建任务，旨在解决单视角下物理一致性不足的问题。通过物理约束的优化管道，重建高精度物理有效的3D场景，提升机器人操作的可靠性。**

- **链接: [https://arxiv.org/pdf/2602.12633](https://arxiv.org/pdf/2602.12633)**

> **作者:** Tianyi Xiang; Jiahang Cao; Sikai Guo; Guoyang Zhao; Andrew F. Luo; Jun Ma
>
> **备注:** Project page: this https URL
>
> **摘要:** Reconstructing physically valid 3D scenes from single-view observations is a prerequisite for bridging the gap between visual perception and robotic control. However, in scenarios requiring precise contact reasoning, such as robotic manipulation in highly cluttered environments, geometric fidelity alone is insufficient. Standard perception pipelines often neglect physical constraints, resulting in invalid states, e.g., floating objects or severe inter-penetration, rendering downstream simulation unreliable. To address these limitations, we propose a novel physics-constrained Real-to-Sim pipeline that reconstructs physically consistent 3D scenes from single-view RGB-D data. Central to our approach is a differentiable optimization pipeline that explicitly models spatial dependencies via a contact graph, jointly refining object poses and physical properties through differentiable rigid-body simulation. Extensive evaluations in both simulation and real-world settings demonstrate that our reconstructed scenes achieve high physical fidelity and faithfully replicate real-world contact dynamics, enabling stable and reliable contact-rich manipulation.
>
---
#### [replaced 038] Encirclement Guaranteed Finite-Time Capture against Unknown Evader Strategies
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体协同控制任务，解决未知策略下追捕者包围并有限时间捕捉逃逸者的问题。提出策略确保包围与捕获，且给出时间上界。**

- **链接: [https://arxiv.org/pdf/2603.15278](https://arxiv.org/pdf/2603.15278)**

> **作者:** Dinesh Patra; Prajakta Surve; Ashish R. Hota; Shaunak D. Bopardikar
>
> **摘要:** We consider a pursuit-evasion scenario involving a group of pursuers and a single evader in a two-dimensional unbounded environment. The pursuers aim to capture the evader in finite time while ensuring the evader remains enclosed within the convex hull of their positions until capture, without knowledge of the evader's heading angle. Prior works have addressed the problem of encirclement and capture separately in different contexts. In this paper, we present a class of strategies for the pursuers that guarantee capture in finite time while maintaining encirclement, irrespective of the evader's strategy. Furthermore, we derive an upper bound on the time to capture. Numerical results highlight the effectiveness of the proposed framework against a range of evader strategies.
>
---
#### [replaced 039] Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在推理效率和鲁棒性上的问题。提出OptimusVLA框架，引入全局先验和局部一致性记忆，提升成功率与速度。**

- **链接: [https://arxiv.org/pdf/2602.20200](https://arxiv.org/pdf/2602.20200)**

> **作者:** Zaijing Li; Bing Hu; Rui Shao; Gongwei Chen; Dongmei Jiang; Pengwei Xie; Jianye Hao; Liqiang Nie
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** Hierarchical Vision-Language-Action (VLA) models have rapidly become a dominant paradigm for robotic manipulation. It typically comprising a Vision-Language backbone for perception and understanding, together with a generative policy for action generation. However, its performance is increasingly bottlenecked by the action generation proceess. (i) Low inference efficiency. A pronounced distributional gap between isotropic noise priors and target action distributions, which increases denoising steps and the incidence of infeasible samples. (ii) Poor robustness. Existing policies condition solely on the current observation, neglecting the constraint of history sequence and thus lacking awareness of task progress and temporal consistency. To address these issues, we introduce OptimusVLA, a dual-memory VLA framework with Global Prior Memory (GPM) and Local Consistency Memory (LCM). GPM replaces Gaussian noise with task-level priors retrieved from semantically similar trajectories, thereby shortening the generative path and reducing the umber of function evaluations (NFE). LCM dynamically models executed action sequence to infer task progress and injects a learned consistency constraint that enforces temporal coherence and smoothness of trajectory. Across three simulation benchmarks, OptimusVLA consistently outperforms strong baselines: it achieves 98.6% average success rate on LIBERO, improves over pi_0 by 13.5% on CALVIN, and attains 38% average success rate on RoboTwin 2.0 Hard. In Real-World evaluation, OptimusVLA ranks best on Generalization and Long-horizon suites, surpassing pi_0 by 42.9% and 52.4%, respectively, while delivering 2.9x inference speedup.
>
---
#### [replaced 040] cuNRTO: GPU-Accelerated Nonlinear Robust Trajectory Optimization
- **分类: cs.RO; cs.DC; eess.SY**

- **简介: 该论文属于轨迹优化任务，解决不确定环境下自主系统安全控制问题。提出cuNRTO框架，利用GPU加速SOCP求解，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.02642](https://arxiv.org/pdf/2603.02642)**

> **作者:** Jiawei Wang; Arshiya Taj Abdul; Evangelos A. Theodorou
>
> **摘要:** Robust trajectory optimization enables autonomous systems to operate safely under uncertainty by computing control policies that satisfy the constraints for all bounded disturbances. However, these problems often lead to large Second Order Conic Programming (SOCP) constraints, which are computationally expensive. In this work, we propose the CUDA Nonlinear Robust Trajectory Optimization (cuNRTO) framework by introducing two dynamic optimization architectures that have direct application to robust decision-making and are implemented on CUDA. The first architecture, NRTO-DR, leverages the Douglas-Rachford (DR) splitting method to solve the SOCP inner subproblems of NRTO, thereby significantly reducing the computational burden through parallel SOCP projections and sparse direct solves. The second architecture, NRTO-FullADMM, is a novel variant that further exploits the problem structure to improve scalability using the Alternating Direction Method of Multipliers (ADMM). Finally, we provide GPU implementations of the proposed methodologies using custom CUDA kernels for SOC projection steps and cuBLAS GEMM chains for feedback gain updates. We validate the performance of cuNRTO through simulated experiments on unicycle, quadcopter, and Franka manipulator models, demonstrating speedups of up to 139.6$\times$. More details are available at this https URL.
>
---
#### [replaced 041] Towards Long-Lived Robots: Continual Learning VLA Models via Reinforcement Fine-Tuning
- **分类: cs.RO**

- **简介: 该论文属于机器人持续学习任务，旨在解决VLA模型在适应新任务时数据需求高和遗忘问题。提出LifeLong-RFT方法，通过多维奖励机制提升性能。**

- **链接: [https://arxiv.org/pdf/2602.10503](https://arxiv.org/pdf/2602.10503)**

> **作者:** Yuan Liu; Haoran Li; Shuai Tian; Yuxing Qin; Yuhui Chen; Yupeng Zheng; Yongzhen Huang; Dongbin Zhao
>
> **摘要:** Pretrained on large-scale and diverse datasets, VLA models demonstrate strong generalization and adaptability as general-purpose robotic policies. However, Supervised Fine-Tuning (SFT), which serves as the primary mechanism for adapting VLAs to downstream domains, requires substantial amounts of task-specific data and is prone to catastrophic forgetting. To address these limitations, we propose LifeLong-RFT, a simple yet effective Reinforcement Fine-Tuning (RFT) strategy for VLA models independent of online environmental feedback and pre-trained reward models. By integrating chunking-level on-policy reinforcement learning with the proposed multi-dimensional process reward mechanism, LifeLong-RFT quantifies the heterogeneous contributions of intermediate action chunks across three dimensions to facilitate policy optimization. Specifically, (1) the Quantized Action Consistency Reward (QACR) ensures accurate action prediction within the discrete action space; (2) the Continuous Trajectory Alignment Reward (CTAR) aligns decoded continuous action chunks with reference trajectories to ensure precise control; (3) the Format Compliance Reward (FCR) guarantees the structural validity of outputs. Comprehensive experiments across SimplerEnv, LIBERO, and real-world tasks demonstrate that LifeLong-RFT exhibits strong performance in multi-task learning. Furthermore, for continual learning on the LIBERO benchmark, our method achieves a 22% gain in average success rate over SFT, while effectively adapting to new tasks using only 20% of the training data. Overall, our method provides a promising post-training paradigm for VLAs. The project page is available at <this https URL>.
>
---
#### [replaced 042] HandelBot: Real-World Piano Playing via Fast Adaptation of Dexterous Robot Policies
- **分类: cs.RO**

- **简介: 该论文提出HandelBot，解决多指机器人高精度操作问题，通过模拟策略与快速适应结合，实现精确钢琴演奏。**

- **链接: [https://arxiv.org/pdf/2603.12243](https://arxiv.org/pdf/2603.12243)**

> **作者:** Amber Xie; Haozhi Qi; Dorsa Sadigh
>
> **备注:** Website: this https URL
>
> **摘要:** Mastering dexterous manipulation with multi-fingered hands has been a grand challenge in robotics for decades. Despite its potential, the difficulty of collecting high-quality data remains a primary bottleneck for high-precision tasks. While reinforcement learning and simulation-to-real-world transfer offer a promising alternative, the transferred policies often fail for tasks demanding millimeter-scale precision, such as bimanual piano playing. In this work, we introduce HandelBot, a framework that combines a simulation policy and rapid adaptation through a two-stage pipeline. Starting from a simulation-trained policy, we first apply a structured refinement stage to correct spatial alignments by adjusting lateral finger joints based on physical rollouts. Next, we use residual reinforcement learning to autonomously learn fine-grained corrective actions. Through extensive hardware experiments across five recognized songs, we demonstrate that HandelBot can successfully perform precise bimanual piano playing. Our system outperforms direct simulation deployment by a factor of 1.8x and requires only 30 minutes of physical interaction data.
>
---
#### [replaced 043] Multi-Source Human-in-the-Loop Digital Twin Testbed for Connected and Autonomous Vehicles in Mixed Traffic Flow
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于智能交通领域，解决CAV与HDV在混合交通中的交互问题。提出MSH-MCCT测试平台，融合物理、虚拟与混合环境，实现CAV与人类驾驶员的实时协同测试。**

- **链接: [https://arxiv.org/pdf/2603.17751](https://arxiv.org/pdf/2603.17751)**

> **作者:** Jianghong Dong; Chunying Yang; Mengchi Cai; Chaoyi Chen; Qing Xu; Jianqiang Wang; Jiawei Wang; Keqiang Li
>
> **摘要:** In the emerging mixed traffic environments, Connected and Autonomous Vehicles (CAVs) have to interact with surrounding human-driven vehicles (HDVs). This paper introduces MSH-MCCT (Multi-Source Human-in-the-Loop Mixed Cloud Control Testbed), a novel CAV testbed that captures complex interactions between various CAVs and HDVs. Utilizing the Mixed Digital Twin concept, which combines Mixed Reality with Digital Twin, MSH-MCCT integrates physical, virtual, and mixed platforms, along with multi-source control inputs. Bridged by the mixed platform, MSH-MCCT allows human drivers and CAV algorithms to operate both physical and virtual vehicles within multiple fields of view. Particularly, this testbed facilitates the coexistence and real-time interaction of physical and virtual CAVs \& HDVs, significantly enhancing the experimental flexibility and scalability. Experiments on vehicle platooning in mixed traffic showcase the potential of MSH-MCCT to conduct CAV testing with multi-source real human drivers in the loop through driving simulators of diverse fidelity. The videos for the experiments are available at our project website: this https URL.
>
---
#### [replaced 044] Robust and Resilient Soft Robotic Object Insertion with Compliance-Enabled Contact Formation and Failure Recovery
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决对象插入中的失败问题。通过软腕实现安全接触与自动恢复，提升插入的鲁棒性与适应性。**

- **链接: [https://arxiv.org/pdf/2509.17666](https://arxiv.org/pdf/2509.17666)**

> **作者:** Mimo Shirasaka; Cristian C. Beltran-Hernandez; Masashi Hamaya; Yoshitaka Ushiku
>
> **摘要:** Object insertion tasks are prone to failure under pose uncertainty and environmental variation, often requiring manual fine-tuning or controller retraining. We present a novel approach for robust and resilient object insertion using a passively compliant soft wrist that enables safe contact absorption through large deformations, without high-frequency control or force sensing. Our method structures insertion as compliance-enabled contact formations, sequential contact states that progressively constrain degrees of freedom, and integrates automated failure recovery strategies. Our key insight is that wrist compliance permits safe, repeated recovery attempts; hence, we refer to it as compliance-enabled failure recovery. We employ a pre-trained vision-language model (VLM) that assesses each skill execution from terminal poses and images, identifies failure modes, and proposes recovery actions by selecting skills and updating goals. In simulation, our method achieved an 83% success rate, recovering from failures induced by randomized conditions, including grasp misalignments up to 5 degrees, hole-pose errors up to 20 mm, fivefold increases in friction, and unseen square/rectangular pegs, and we further validated the approach on a real robot. Project page is available at this https URL.
>
---
#### [replaced 045] Propagating Unsafe Actions in LLM Controlled Multi-Robot Collaboration via Single Robot Compromise
- **分类: cs.RO; cs.CR**

- **简介: 该论文研究LLM控制下的多机器人协作安全问题，提出一种通过单个机器人传播恶意指令的攻击方法，揭示了系统中的安全漏洞。**

- **链接: [https://arxiv.org/pdf/2605.15641](https://arxiv.org/pdf/2605.15641)**

> **作者:** Zhen Huang; Zhihuang Liu; Mengxuan Luo; Weishang Wu; Zhiping Cai
>
> **备注:** Accepted by the 35th International Joint Conference on Artificial Intelligence (IJCAI 2026). 9 pages, 4 figures, 3 tables
>
> **摘要:** Large language models (LLMs) are increasingly used as general planners in embodied intelligence, enabling high level coordination and low level task planning for both single robot and multi-robot collaboration. This increasing reliance on embodied LLM planners also raises critical security concerns, since misaligned or manipulated instructions can be translated into physical actions. Prior work has studied such threats in single robot settings, while security risks in LLM controlled multi-robot collaboration, especially those propagated through inter robot communication, remain largely unexplored. To bridge this gap, we propose a novel attack paradigm for multi-robot system in which the adversary interacts with only a single entry robot. The compromised robot then propagates malicious intent through peer communication, leading to coordinated unsafe actions across the system. Our evaluation, covering high risk dimensions of dereliction of duty, privacy compromise, and public safety hazards, reveals a persistent safety alignment gap in multi-robot planners. We quantify this process with three metrics, obedience, infectiousness, and stealthiness. Experiments demonstrate both persistent attacker control and rapid propagation: obedience reaches 1.00 in the strongest cases, and infectiousness rises to 0.90. Notably, the attack is highly efficient, requiring as few as 3.0 rounds to compromise all the robots while maintaining a stealthiness score of 0.81. Such risks are amplified when robots must resolve trade offs in critical situations, such as emergencies or conflicts of rights, because the coordination mechanism can unintentionally allow adversarial instructions to override safety requirements. The code is available at this https URL.
>
---
#### [replaced 046] Guided Reinforcement Learning for Omnidirectional 3D Jumping in Quadruped Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于四足机器人跳跃任务，解决传统方法效率低、依赖参数和安全性差的问题，提出一种结合物理模型的引导强化学习方法。**

- **链接: [https://arxiv.org/pdf/2507.16481](https://arxiv.org/pdf/2507.16481)**

> **作者:** Riccardo Bussola; Michele Focchi; Giulio Turrisi; Claudio Semini; Luigi Palopoli
>
> **摘要:** Jumping poses a significant challenge for quadruped robots, despite being crucial for many operational scenarios. While optimisation methods exist for controlling such motions, they are often time-consuming and demand extensive knowledge of robot and terrain parameters, making them less robust in real-world scenarios. Reinforcement learning (RL) is emerging as a viable alternative, yet conventional end-to-end approaches lack efficiency in terms of sample complexity, requiring extensive training in simulations, and predictability of the final motion, which makes it difficult to certify the safety of the final motion. To overcome these limitations, this paper introduces a novel guided reinforcement learning approach that leverages physical intuition for efficient and explainable jumping, by combining Bézier curves with a Uniformly Accelerated Rectilinear Motion (UARM) model. Extensive simulation and experimental results clearly demonstrate the advantages of our approach over existing alternatives.
>
---
#### [replaced 047] A Visual Reinforcement Learning-Based Separate Primitive Policy for Peg-in-Hole Tasks
- **分类: cs.RO**

- **简介: 该论文针对插孔任务，提出S2P方法，通过视觉强化学习同时学习定位与插入策略，提升样本效率和成功率。**

- **链接: [https://arxiv.org/pdf/2504.14820](https://arxiv.org/pdf/2504.14820)**

> **作者:** Zichun Xu; Zhaomin Wang; Yuntao Li; Lei Zhuang; Zhiyuan Zhao; Guocai Yang; Jingdong Zhao
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** For peg-in-hole tasks, humans rely on binocular visual perception to locate the peg above the hole surface and then proceed with insertion. This paper draws insights from this behavior to enable agents to learn efficient assembly strategies through visual reinforcement learning. Hence, we propose a Separate Primitive Policy (S2P) to learn how to derive location and insertion actions simultaneously. S2P is compatible with model-free reinforcement learning algorithms. Ten insertion tasks featuring different polygons are developed as benchmarks for evaluations. Simulation experiments show that S2P can boost the sample efficiency and success rate even with force constraints. Real-world experiments are also performed to verify the feasibility of S2P. Ablations are finally given to discuss the generalizability of S2P and some factors that affect its performance.
>
---
#### [replaced 048] Quality-guided UAV Surface Exploration for 3D Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于3D重建任务，解决自主机器人在未知环境中高效探索与建模的问题。提出一种基于重建质量的NBV规划框架，提升建图覆盖率和效率。**

- **链接: [https://arxiv.org/pdf/2511.20353](https://arxiv.org/pdf/2511.20353)**

> **作者:** Benjamin Sportich; Kenza Boubakri; Olivier Simonin; Alessandro Renzaglia
>
> **摘要:** Reasons for mapping an unknown environment with autonomous robots are wide-ranging, but in practice, they are often overlooked when developing planning strategies. Rapid information gathering and comprehensive structural assessment of buildings have different requirements and therefore necessitate distinct methodologies. In this paper, we propose a novel modular Next-Best-View (NBV) planning framework for aerial robots that explicitly uses a reconstruction quality objective to guide the exploration planning. In particular, our approach introduces new and efficient methods for view generation and selection of viewpoint candidates that are adaptive to the user-defined quality requirements, fully exploiting the uncertainty encoded in a Truncated Signed Distance field (TSDF) representation of the environment. This results in informed and efficient exploration decisions tailored towards the predetermined objective. Finally, we validate our method via extensive simulations in realistic environments. We demonstrate that it successfully adjusts its behavior to the user goal while consistently outperforming conventional NBV strategies in terms of coverage, quality of the final 3D map and path efficiency.
>
---
#### [replaced 049] Universal Pose Pretraining for Generalizable Vision-Language-Action Policies
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决模型特征崩溃和训练效率低的问题。通过分离预训练与微调阶段，引入姿态令牌提升空间对齐与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.19710](https://arxiv.org/pdf/2602.19710)**

> **作者:** Haitao Lin; Hanyang Yu; Jingshun Huang; He Zhang; Yonggen Ling; Ping Tan; Xiangyang Xue; Yanwei Fu
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026. Project website: this https URL
>
> **摘要:** Existing Vision-Language-Action (VLA) models often suffer from feature collapse and low training efficiency because they entangle high-level perception with sparse, embodiment-specific action supervision. Since these models typically rely on VLM backbones optimized for Visual Question Answering (VQA), they excel at semantic identification but often overlook subtle 3D state variations that dictate distinct action patterns. To resolve these misalignments, we propose Pose-VLA, a decoupled paradigm that separates VLA training into a pre-training phase for extracting universal 3D spatial priors in a unified camera-centric space, and a post-training phase for efficient embodiment alignment within robot-specific action space. By introducing discrete pose tokens as a universal representation, Pose-VLA seamlessly integrates spatial grounding from diverse 3D datasets with geometry-level trajectories from robotic demonstrations. Our framework follows a two-stage pre-training pipeline, establishing fundamental spatial grounding via poses followed by motion alignment through trajectory supervision. Extensive evaluations demonstrate that Pose-VLA achieves state-of-the-art results on RoboTwin 2.0 with a 79.5% average success rate and competitive performance on LIBERO at 96.0%. Real-world experiments further showcase robust generalization across diverse objects using only 100 demonstrations per task, validating the efficiency of our pre-training paradigm.
>
---
#### [replaced 050] A Novel Model for 3D Motion Planning for a Generalized Dubins Vehicle with Pitch and Yaw Rate Constraints
- **分类: cs.RO; math.OC**

- **简介: 该论文属于3D运动规划任务，解决固定翼无人机在姿态约束下的最短路径问题。通过引入完整姿态模型和双控制输入，构建更准确的路径规划方法。**

- **链接: [https://arxiv.org/pdf/2509.24143](https://arxiv.org/pdf/2509.24143)**

> **作者:** Deepak Prakash Kumar; Swaroop Darbha; Satyanarayana Gupta Manyam; David W. Casbeer
>
> **备注:** The code for this paper is available at this https URL
>
> **摘要:** In this paper, we propose a new modeling approach and a fast algorithm for 3D motion planning, applicable for fixed-wing unmanned aerial vehicles. The goal is to construct the shortest path connecting given initial and final configurations subject to motion constraints. Our work differs from existing literature in two ways. First, we consider full vehicle orientation using a body-attached frame, which includes roll, pitch, and yaw angles. However, existing work uses only pitch and/or heading angle, which is insufficient to uniquely determine orientation. Second, we use two control inputs to represent bounded pitch and yaw rates, reflecting control by two separate actuators. In contrast, most previous methods rely on a single input, such as path curvature, which is insufficient for accurately modeling the vehicle's kinematics in 3D. We use a rotation minimizing frame to describe the vehicle's configuration and its evolution, and construct paths by concatenating optimal Dubins paths on spherical, cylindrical, or planar surfaces. Numerical simulations show our approach generates feasible paths within 10 seconds on average and yields shorter paths than existing methods in most cases.
>
---
#### [replaced 051] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉-语言-动作任务，旨在解决长时序、依赖历史的机械操作问题。提出RoboMME基准，评估和提升VLA模型的记忆能力。**

- **链接: [https://arxiv.org/pdf/2603.04639](https://arxiv.org/pdf/2603.04639)**

> **作者:** Yinpei Dai; Hongze Fu; Jayjun Lee; Yuejiang Liu; Haoran Zhang; Jianing Yang; Chelsea Finn; Nima Fazeli; Joyce Chai
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Memory is critical for long-horizon and history-dependent robotic manipulation. Such tasks often involve counting repeated actions or manipulating objects that become temporarily occluded. Recent vision-language-action (VLA) models have begun to incorporate memory mechanisms; however, their evaluations remain confined to narrow, non-standardized settings. This limits their systematic understanding, comparison, and progress measurement. To address these challenges, we introduce RoboMME: a large-scale standardized benchmark for evaluating and advancing VLA models in long-horizon, history-dependent scenarios. Our benchmark comprises 16 manipulation tasks constructed under a carefully designed taxonomy that evaluates temporal, spatial, object, and procedural memory. We further develop a suite of 14 memory-augmented VLA variants built on the {\pi}0.5 backbone to systematically explore different memory representations across multiple integration strategies. Experimental results show that the effectiveness of memory representations is highly task-dependent, with each design offering distinct advantages and limitations across different tasks. Videos and code can be found at our website this https URL.
>
---
#### [replaced 052] Adaptive Outer-Loop Control of Quadrotors via Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于无人机控制任务，解决DRL在真实环境中性能不足的问题。通过引入自适应控制架构和残差动力预测器，提升无人机对动态扰动的响应能力。**

- **链接: [https://arxiv.org/pdf/2605.16015](https://arxiv.org/pdf/2605.16015)**

> **作者:** Vishnu Saj; Sushil Vemuri; Dileep Kalathil; Moble Benedict
>
> **摘要:** Deep Reinforcement Learning (DRL) for quadrotor flight control typically relies on Domain Randomization (DR) for sim-to-real transfer, resulting in overly conservative policies that struggle with dynamic disturbances. To overcome this, we propose a novel adaptive control architecture that actively perceives and reacts to instantaneous perturbations. First, we train an optimal outer-loop policy, then replace its reliance on ground-truth disturbance data with a Residual Dynamics Predictor (RDP). The RDP estimates the external forces and moments acting on the aircraft in flight online using only the history of states and control actions. For seamless hardware transfer, we introduce a data-efficient linear calibration bridge and an online thrust correction mechanism that align the simulated latent space with reality using mere seconds of flight data. Real-world validations on a Crazyflie micro-quadrotor demonstrate that our adaptive controller significantly outperforms baselines, maintaining precise trajectory tracking under severe uncertainties including mass variations, asymmetric payloads, and dynamic slung loads
>
---
#### [replaced 053] A Sliced Learning Framework for Online Disturbance Identification in Quadrotor SO(3) Attitude Control
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于无人机姿态控制任务，旨在解决在线扰动识别问题。提出Sliced Learning框架，通过误差分解实现轻量级神经自适应，提升控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2508.14422](https://arxiv.org/pdf/2508.14422)**

> **作者:** Tianhua Gao; Masashi Izumita; Kohji Tomita; Akiya Kamimura
>
> **备注:** v4: This version has been accepted for publication in IEEE/ASME Transactions on Mechatronics (TMECH). Supplementary video links have also been added
>
> **摘要:** This paper introduces a dimension-decomposed geometric learning framework called Sliced Learning for disturbance identification in quadrotor geometric attitude control. Instead of conventional learning-from-states, this framework adopts a learning-from-error strategy by using the Lie-algebraic error representation as the input feature, enabling axis-wise space decomposition (``slicing") while preserving the SO(3) structure. This is highly consistent with the geometric mechanism of cognitive control observed in neuroscience, where neural systems organize adaptive representations within structured subspaces to enable cognitive flexibility and efficiency. Based on this framework, we develop a lightweight and structurally interpretable Sliced Adaptive-Neuro Mapping (SANM) module. The high-dimensional mapping for online identification is axially ``sliced" into multiple low-dimensional submappings (``slices"), implemented by shallow neural networks and adaptive laws. These neural networks and adaptive laws are updated online via Lyapunov-based adaptation within their respective shared subspaces. To enhance interpretability, we prove exponential convergence despite time-varying disturbances and inertia uncertainties. To our knowledge, Sliced Learning is among the first frameworks to demonstrate lightweight online neural adaptation at 400 Hz on resource-constrained microcontroller units (MCUs), such as STM32, with real-world experimental validation.
>
---
#### [replaced 054] GeoWorld: Geometric World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GeoWorld，解决视觉规划中的长期预测问题。通过超球JEPA保留几何结构，提升多步规划性能。**

- **链接: [https://arxiv.org/pdf/2602.23058](https://arxiv.org/pdf/2602.23058)**

> **作者:** Zeyu Zhang; Danning Li; Ian Reid; Richard Hartley
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Energy-based predictive world models provide a powerful approach for multi-step visual planning by reasoning over latent energy landscapes rather than generating pixels. However, existing approaches face two major challenges: (i) their latent representations are typically learned in Euclidean space, neglecting the underlying geometric and hierarchical structure among states, and (ii) they struggle with long-horizon prediction, which leads to rapid degradation across extended rollouts. To address these challenges, we introduce GeoWorld, a geometric world model that preserves geometric structure and hierarchical relations through a Hyperbolic JEPA, which maps latent representations from Euclidean space onto hyperbolic manifolds. We further introduce Geometric Reinforcement Learning for energy-based optimization, enabling stable multi-step planning in hyperbolic latent space. Extensive experiments on CrossTask and COIN demonstrate around 3% SR improvement in 3-step planning and 2% SR improvement in 4-step planning compared to the state-of-the-art V-JEPA 2. Project website: this https URL.
>
---
#### [replaced 055] SG-CADVLM: A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全测试任务，旨在解决真实事故场景生成的高成本与低精度问题。通过引入上下文感知解码框架SG-CADVLM，提升场景生成的准确性和真实性。**

- **链接: [https://arxiv.org/pdf/2601.18442](https://arxiv.org/pdf/2601.18442)**

> **作者:** Hongyi Zhao; Shuo Wang; Qijie He; Ziyuan Pu
>
> **摘要:** Autonomous Vehicle (AV) requires rigorous testing in safety-critical scenarios for safety validation, yet its validation is hindered by the high cost of field testing and the lack of fidelity in current simulations for rare safety-critical events. Crash reports offer rich and authentic specifications of real-world accident dynamics, making them a promising resource for Large Language Models and Vision-Language models to generate high-fidelity scenarios. However, the existing models frequently deviate from actual accident characteristics due to context suppression. To address these limitations, this paper presents SG-CADVLM, a framework integrateing Context-Aware Decoding with multimodal input processing to generate safety-critical scenarios from crash reports. The framework mitigates the hallucination of VLMs while generating road geometry and vehicle trajectories simultaneously. The experimental results demonstrate that SG-CADVLM generates combined critical and high-risk scenarios at a rate of 88.1% compared to 31.2% for the baseline methods, representing a 182% improvement, while producing executable simulations for autonomous vehicle testing.
>
---
#### [replaced 056] OxyGen: Unified KV Cache Management for VLA Inference under Multi-Task Parallelism
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出OxyGen，解决多任务并行推理中的KV缓存管理问题，提升VLA模型效率。属于AI推理优化任务。**

- **链接: [https://arxiv.org/pdf/2603.14371](https://arxiv.org/pdf/2603.14371)**

> **作者:** Xiangyu Li; Huaizhi Tang; Xin Ding; Weijun Wang; Ting Cao; Yunxin Liu
>
> **备注:** Preprint
>
> **摘要:** Embodied AI agents increasingly require parallel execution of multiple tasks, such as manipulation, conversation, and memory construction, from shared observations under distinct time constraints. Recent Mixture-of-Transformers (MoT) Vision-Language-Action Models (VLAs) architecturally support such heterogeneous outputs, yet existing inference systems fail to achieve efficient multi-task parallelism for on-device deployment because of redundant computation and resource contention. We identify isolated KV cache management as the root cause. To address this, we propose unified KV cache management, an inference design that treats the KV cache as a first-class shared resource across tasks and over time. This abstraction enables two key optimizations: cross-task KV sharing eliminates redundant prefill of shared observations, while cross-frame continuous batching decouples variable-length language decoding from fixed-rate action generation across control cycles. We implement this design for $\pi_{0.5}$, a popular MoT VLA, and evaluate it on both NVIDIA GeForce RTX 4090 and Jetson AGX Thor, two representative platforms for on-device VLA inference. OxyGen achieves up to 3.7$\times$ speedup over isolated execution, delivering over 200 tokens/s language throughput and 70 Hz action frequency simultaneously without degrading action quality, and we further validate the gains on a real humanoid robot with on-board Jetson AGX Thor.
>
---
#### [replaced 057] Efficient Emotion-Aware Iconic Gesture Prediction for Robot Co-Speech
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于情感感知的图标手势预测任务，旨在提升机器人共言手势的语义准确性。通过轻量级Transformer模型，从文本和情感中生成手势，解决现有系统缺乏语义强调的问题。**

- **链接: [https://arxiv.org/pdf/2604.11417](https://arxiv.org/pdf/2604.11417)**

> **作者:** Edwin C. Montiel-Vazquez; Christian Arzate Cruz; Stefanos Gkikas; Thomas Kassiotis; Giorgos Giannakakis; Randy Gomez
>
> **摘要:** Co-speech gestures increase engagement and improve speech understanding. Most data-driven robot systems generate rhythmic beat-like motion, yet few integrate semantic emphasis. To address this, we propose a lightweight transformer that derives iconic gesture placement and intensity from text and emotion alone, requiring no audio input at inference time. The model outperforms GPT-4o in both semantic gesture placement classification and intensity regression on the BEAT2 dataset, while remaining computationally compact and suitable for real-time deployment on embodied agents.
>
---
#### [replaced 058] ORION: Option-Regularized Deep Reinforcement Learning for Cooperative Multi-Agent Online Navigation
- **分类: cs.RO**

- **简介: 该论文属于多智能体协作导航任务，解决部分已知环境中路径优化与信息共享问题。提出ORION框架，通过深度强化学习实现高效协作导航。**

- **链接: [https://arxiv.org/pdf/2601.01155](https://arxiv.org/pdf/2601.01155)**

> **作者:** Shizhe Zhang; Jingsong Liang; Zhitao Zhou; Shuhan Ye; Yizhuo Wang; Ming Siang Derek Tan; Jimmy Chiun; Yuhong Cao; Guillaume Sartoretti
>
> **摘要:** Existing methods for multi-agent navigation typically assume fully known environments, offering limited support for partially known scenarios with outdated or imperfect prior maps, such as warehouses or factory floors. There, agents need to balance path optimality with collecting and sharing environmental information to help teammates reach their own targets. To these ends, we propose ORION, a novel deep reinforcement learning framework for cooperative multi-agent online navigation in partially known environments. Starting from an imperfect prior map, ORION trains agents to make decentralized decisions, coordinate toward individual targets, and actively reduce task-relevant map uncertainty through online observation sharing in a closed perception-action loop. We first design a shared graph encoder that fuses prior map with online perception into a unified representation, providing robust state embeddings under environmental discrepancies. At the core of ORION is an option-critic framework that learns high-level cooperative modes translated into sequences of low-level actions, enabling adaptive switching between individual navigation and team-level exploration. We further introduce a dual-stage cooperation strategy that allows agents to assist teammates under map uncertainty, thereby reducing the overall makespan. Across extensive maze-like maps and large-scale warehouse environments, ORION achieves high-quality real-time decentralized cooperation while scaling to up to 10 robots, outperforming state-of-the-art classical and learning-based baselines. Finally, we validate ORION on physical robot teams, demonstrating its robustness and practicality for real-world cooperative navigation.
>
---
#### [replaced 059] TACO: Temporal Consensus Optimization for Continual Neural Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中持续学习的神经映射问题。提出TACO框架，通过时间共识优化实现无需回放的历史知识利用，提升映射的适应性与内存效率。**

- **链接: [https://arxiv.org/pdf/2602.04516](https://arxiv.org/pdf/2602.04516)**

> **作者:** Xunlan Zhou; Hongrui Zhao; Negar Mehr
>
> **备注:** In: Robotics: Science and Systems (RSS 2026)
>
> **摘要:** Neural implicit mapping has emerged as a powerful paradigm for robotic navigation and scene understanding. However, real-world robotic deployment requires continual adaptation to changing environments under strict memory and computation constraints, which existing mapping systems fail to support. Most prior methods rely on replaying historical observations to preserve consistency and assume static scenes. As a result, they cannot adapt to continual learning in dynamic robotic settings. To address these challenges, we propose TACO (TemporAl Consensus Optimization), a replay-free framework for continual neural mapping. We reformulate mapping as a temporal consensus optimization problem, where we treat past model snapshots as temporal neighbors. Intuitively, our approach resembles a model consulting its own past knowledge. We update the current map by enforcing weighted consensus with historical representations. Our method allows reliable past geometry to constrain optimization while permitting unreliable or outdated regions to be revised in response to new observations. TACO achieves a balance between memory efficiency and adaptability without storing or replaying previous data. Through extensive simulated and real-world experiments, we show that TACO robustly adapts to scene changes, and consistently outperforms other continual learning baselines.
>
---
#### [replaced 060] Constrained Policy Optimization via Sampling-Based Weight-Space Projection
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于安全强化学习任务，解决在不破坏安全约束的情况下优化策略的问题。提出SCPO方法，在参数空间直接约束策略更新，确保安全性和稳定性。**

- **链接: [https://arxiv.org/pdf/2512.13788](https://arxiv.org/pdf/2512.13788)**

> **作者:** Shengfan Cao; Francesco Borrelli; Eunhyek Joa
>
> **备注:** Accepted for publication at IFAC World Congress 2026
>
> **摘要:** Safety-critical learning requires policies that improve performance without leaving the safe operating regime. We study constrained policy learning where model parameters must satisfy rollout-based safety constraints that can be evaluated but not differentiated analytically. We propose SCPO, a sampling-based weight-space projection method that enforces safety directly in parameter space without requiring gradient access to the constraint functions. SCPO constructs a local safe region by combining rollout-based safety evaluations with smoothness bounds relating parameter perturbations to changes in safety metrics, and projects each gradient update via a convex SOCP. We establish a safe-by-induction guarantee: starting from any safe initialization, all intermediate policies remain safe given feasible projections. In constrained control settings with a stabilizing backup policy, SCPO further ensures closed-loop stability while enabling safe adaptation beyond the conservative backup. Experiments on constrained regression with harmful supervision and double-integrator imitation with a malicious expert show that SCPO rejects unsafe updates, maintains feasibility throughout training, and achieves meaningful objective improvement.
>
---
#### [replaced 061] See What Matters: Differentiable Grid Sample Pruning for Generalizable Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型优化任务，旨在解决高计算成本与压缩导致性能下降的矛盾。提出GridS模块，通过几何感知的连续采样实现高效压缩，保持性能。**

- **链接: [https://arxiv.org/pdf/2605.11817](https://arxiv.org/pdf/2605.11817)**

> **作者:** Yixu Feng; Zinan Zhao; Yanxiang Ma; Chenghao Xia; Chengbin Du; Yunke Wang; Chang Xu
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable promise in robotics manipulation, yet their high computational cost hinders real-time deployment. Existing token pruning methods suffer from a fundamental trade-off: aggressive compression using pruning inevitably discards critical geometric details like contact points, leading to severe performance degradation. This forces a compromise, limiting the achievable compression rate and thus the potential speedup. We argue that breaking this trade-off requires rethinking compression as a geometry-aware, continuous token resampling in the vision encoder. To this end, we propose the Differentiable Grid Sampler (GridS), a plug-and-play module that performs task-aware, continuous resampling of visual tokens in VLA. By adaptively predicting a minimal set of salient coordinates and extracting features via differentiable interpolation, GridS preserves essential spatial information while achieving drastic compression (with fewer than 10% original visual tokens). Experiments on both LIBERO benchmark and a real robotic platform demonstrate that validating the lowest feasible visual token count reported to date, GridS achieves a 76% reduction in FLOPs with no degradation in the success rate. The code is available at this https URL.
>
---
#### [replaced 062] FAM-HRI: Foundation-Model Assisted Multi-Modal Human-Robot Interaction Combining Gaze and Speech
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决传统交互方式效率低、易混淆的问题。通过融合语言与视觉信息，提出FAM-HRI框架，提升机器人操作的准确性与便捷性。**

- **链接: [https://arxiv.org/pdf/2503.16492](https://arxiv.org/pdf/2503.16492)**

> **作者:** Yuzhi Lai; Shenghai Yuan; Peizheng Li; Boya Zhang; Benjamin Kiefer; Tianchen Deng; Andreas Zell
>
> **备注:** This work has been accepted for publication in IEEE Transactions on Automation Science and Engineering @ 2026 IEEE
>
> **摘要:** ffective Human-Robot Interaction (HRI) is crucial for enhancing accessibility and usability in real-world robotics applications. However, existing solutions often rely on gesture- only or language-only commands, making interaction inefficient and ambiguous, particularly for users with physical impairments. In this paper, we introduce FAM-HRI, an efficient multimodal framework for HRI that integrates language and gaze inputs via foundation models. By leveraging lightweight Meta ARIA glasses, our system captures real-time multimodal signals and utilizes large language models (LLMs) to fuse user intention with scene context, enabling intuitive and precise robot manipulation. Our method accurately determines the gaze fixation time interval, reducing noise caused by the gaze dynamic nature. Experimental evaluations demonstrate that FAM-HRI achieves a high success rate in task execution while maintaining a low interaction time, providing a practical solution for individuals with limited physical mobility or motor impairments. To support the community, we have released our system design, algorithms, and solutions at this https URL.
>
---
#### [replaced 063] SutureFormer: Learning Surgical Trajectories via Goal-conditioned Offline RL in Pixel Space
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于手术轨迹预测任务，解决视觉引导下缝合针轨迹预测问题，提出SutureFormer框架，通过目标条件离线强化学习实现像素级序列动作学习。**

- **链接: [https://arxiv.org/pdf/2603.26720](https://arxiv.org/pdf/2603.26720)**

> **作者:** Huanrong Liu; Chunlin Tian; Tongyu Jia; Tailai Zhou; Qin Liu; Yu Gao; Yutong Ban; Yun Gu; Guy Rosman; Xin Ma; Qingbiao Li
>
> **摘要:** Predicting surgical needle trajectories from endoscopic video is critical for robot-assisted suturing, enabling anticipatory planning, real-time guidance, and safer motion execution. Existing methods that directly learn motion distributions from visual observations tend to overlook the sequential dependency among adjacent motion steps. Moreover, sparse waypoint annotations often fail to provide sufficient supervision, further increasing the difficulty of supervised or imitation learning methods. To address these challenges, we formulate image-based needle trajectory prediction as a sequential decision-making problem, in which the needle tip is treated as an agent that moves step by step in pixel space. This formulation naturally captures the continuity of needle motion and enables the explicit modeling of physically plausible pixel-wise state transitions over time. From this perspective, we propose SutureFormer, a goal-conditioned offline reinforcement learning framework that leverages sparse annotations to dense reward signals via cubic spline interpolation, encouraging the policy to exploit limited expert guidance while exploring plausible future motion paths. SutureFormer encodes variable-length clips using an observation encoder to capture both local spatial cues and long-range temporal dynamics, and autoregressively predicts future waypoints through actions composed of discrete directions and continuous magnitudes. To enable stable offline policy optimization from expert demonstrations, we adopt Conservative Q-Learning with Behavioral Cloning regularization. Experiments on a new kidney wound suturing dataset containing 1,158 trajectories from 50 patients show that SutureFormer reduces Average Displacement Error by 58.6% compared with the strongest baseline, demonstrating the effectiveness of modeling needle trajectory prediction as pixel-level sequential action learning.
>
---
#### [replaced 064] Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization
- **分类: cs.RO**

- **简介: 该论文属于自主赛车轨迹优化任务，旨在解决传统初始化方法收敛慢、效果差的问题。通过学习Formula 1数据生成优化初始轨迹，提升求解效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.07126](https://arxiv.org/pdf/2603.07126)**

> **作者:** Samir Shehadeh; Lukas Kutsch; Nils Dengler; Sicong Pan; Maren Bennewitz
>
> **摘要:** Trajectory optimization is a central component of fast and efficient autonomous racing. However practical optimization pipelines remain highly sensitive to initialization and may converge slowly or to suboptimal local solutions when seeded with heuristic trajectories such as the centerline or minimum-curvature paths. To address this limitation, we leverage expert driving behavior as a initialization prior and propose a learning-informed initialization strategy based on real-world Formula~1 telemetry. To this end, we first construct a multi-track Formula~1 trajectory dataset by reconstructing and aligning noisy GPS telemetry to a standardized reference-line representation across 17 tracks. Building on this, we present a neural network that predicts an expert-like raceline offset directly from local track geometry, without explicitly modeling vehicle dynamics or forces. The predicted raceline is then used as an informed seed for a minimum-time optimal control solver. Experiments on all 17 tracks demonstrate that the learned initialization accelerates solver convergence and significantly reduces runtime compared to traditional geometric baselines, while preserving the final optimized lap time.
>
---
#### [replaced 065] VISOR: A Vision-Language Model-based Test Oracle for Testing Robots
- **分类: cs.SE; cs.RO**

- **简介: 该论文提出VISOR，一种基于视觉-语言模型的测试预言机方法，用于自动化评估机器人任务的正确性和质量，解决传统测试方法依赖人工和任务特定符号预言机的问题。**

- **链接: [https://arxiv.org/pdf/2605.10408](https://arxiv.org/pdf/2605.10408)**

> **作者:** Prasun Saurabh; Pablo Valle; Aitor Arrieta; Shaukat Ali; Paolo Arcaini
>
> **摘要:** Testing robots requires assessing whether they perform their intended tasks correctly, dependably, and with high quality, a challenge known as the test oracle problem in software testing. Traditionally, this assessment relies on task-specific symbolic oracles for task correctness and on human manual evaluation of robot behavior, which is time-consuming, subjective, and error-prone. To address this, we propose VISOR, a Vision-Language Model (VLM)-based approach for automated test oracle assessment that eliminates the need of expensive human evaluations. VISOR performs automated evaluation of task correctness and quality, addressing the limitations of existing symbolic test oracles, which are task-specific and provide pass/fail judgments without explicitly quantifying task quality. Given the inherent uncertainty in VLMs, VISOR also explicitly quantifies its own uncertainty during test assessments. We evaluated VISOR using two VLMs, i.e., GPT and Gemini, across four robotic tasks on over 1,000 videos. Results show that Gemini achieves higher recall while GPT achieves higher precision. However, both models show low correlation between uncertainty and correctness, which prevents using uncertainty as a correctness predictor.
>
---
#### [replaced 066] Self-Supervised Bootstrapping of Action-Predictive Embodied Reasoning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型任务，解决传统方法依赖固定模板导致的推理与控制不匹配问题。提出R&B-EnCoRe方法，通过自监督学习提升动作预测的推理能力。**

- **链接: [https://arxiv.org/pdf/2602.08167](https://arxiv.org/pdf/2602.08167)**

> **作者:** Milan Ganai; Katie Luo; Jonas Frey; Clark Barrett; Marco Pavone
>
> **备注:** Robotics: Science and Systems (RSS) 2026
>
> **摘要:** Embodied Chain-of-Thought (CoT) reasoning has significantly enhanced Vision-Language-Action (VLA) models, yet current methods rely on rigid templates to specify reasoning primitives (e.g., objects in the scene, high-level plans, structural affordances). These templates can force policies to process irrelevant information that distracts from critical action-prediction signals. This creates a bottleneck: without successful policies, we cannot verify reasoning quality; without quality reasoning, we cannot build robust policies. We introduce R&B-EnCoRe, which enables models to bootstrap embodied reasoning from internet-scale knowledge through self-supervised refinement. By treating reasoning as a latent variable within importance-weighted variational inference, models can generate and distill a refined reasoning training dataset of embodiment-specific strategies without external rewards, verifiers, or human annotation. We validate R&B-EnCoRe across manipulation (Franka Panda in simulation, WidowX in hardware), legged navigation (bipedal, wheeled, bicycle, quadruped), and autonomous driving embodiments using various VLA architectures with 1B, 4B, 7B, and 30B parameters. Our approach achieves 28% gains in manipulation success, 101% improvement in navigation scores, and 21% reduction in collision-rate metric over models that indiscriminately reason about all available primitives. R&B-EnCoRe enables models to distill reasoning that is predictive of successful control, bypassing manual annotation engineering while grounding internet-scale knowledge in physical execution.
>
---
#### [replaced 067] First Experimental Demonstration of Natural Hovering Extremum Seeking: A New Paradigm in Flapping Flight Physics
- **分类: cs.RO; math.OC**

- **简介: 该论文属于飞行控制任务，旨在解决自主悬停与稳定问题。通过实验验证一种无需模型的自然悬停极值搜索方法，实现对光源的稳定悬停。**

- **链接: [https://arxiv.org/pdf/2508.20836](https://arxiv.org/pdf/2508.20836)**

> **作者:** Ahmed A. Elgohary; Rohan Palanikumar; Simone Martini; Sameh A. Eisa
>
> **摘要:** In this letter, we report the first experimental demonstration of the recently emerged new paradigm in hovering and flapping flight physics called (Natural Hovering Extremum Seeking (NH-ES)) [this http URL], which theorized that stable hovering flight physics observed in nature by flapping insects and hummingbirds can be generated via a model-free, real-time, computationally-basic, sensory-based feedback mechanism that only needs the built-in natural oscillations of the flapping wing as both the control and the propulsive input. We run experiments of moth-like, light source-seeking, on a flapping-wing body in a total model-free setting that is agnostic to morphological parameters and body/aerodynamic models. We show that the flapping body using NH-ES gains altitude and stabilizes autonomously the servos responsible for flapping, including with pitching dynamics (believed in literature to be a main reason of instability in open-loop hovering). The flapping body effectively/stably hovers about the light source, needing only feedback of local measurements of light intensity. Our results were also achieved under delay/noise effects, supporting earlier observations that NH-ES is robust against potential processing delays and noisy-sensations.
>
---
#### [replaced 068] Adaptive Control in Autonomous Driving via Real-Time Recurrent RL
- **分类: cs.RO; cs.LG; cs.NE; eess.SY**

- **简介: 论文研究自动驾驶中的在线策略微调，使用实时循环强化学习（RTRRL）适应分布变化。结合行为克隆与在线微调，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2602.02236](https://arxiv.org/pdf/2602.02236)**

> **作者:** Julian Lemmel; Felix Resch; Mónika Farsang; Ramin Hasani; Daniela Rus; Radu Grosu
>
> **摘要:** We study online fine-tuning of pretrained control policies for autonomous driving using Real-Time Recurrent Reinforcement Learning (RTRRL), a memory-efficient algorithm that updates policy parameters at every time step without backpropagation through time. We extend RTRRL to support LrcSSM, a recently proposed nonlinear diagonal state-space model, and combine offline behavioral cloning with online RTRRL fine-tuning to adapt policies to distribution shifts at deployment. We validate the approach in the CarRacing simulation and on a 1:10-scale RoboRacer platform equipped with an event camera, where a pretrained policy is fine-tuned online during real-world line-following. To our knowledge, this is the first demonstration of online RL fine-tuning with event-camera observations on standard (non-spiking) hardware in closed-loop control. LrcSSM-based policies improve fastest and most consistently across both settings.
>
---
#### [replaced 069] QuickLAP: Quick Language-Action Preference Learning for Semi-Autonomous Agents
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出QuickLAP，解决半自主代理的奖励函数学习问题。融合物理反馈与语言信息，提升奖励学习的准确性和实时性。**

- **链接: [https://arxiv.org/pdf/2511.17855](https://arxiv.org/pdf/2511.17855)**

> **作者:** Jordan Abi Nader; David Lee; Nathaniel Dennler; Andreea Bobu
>
> **摘要:** Robots must learn from both what people do and what they say, but either modality alone is often incomplete: physical corrections are grounded but ambiguous in intent, while language expresses high-level goals but lacks physical grounding. We introduce QuickLAP: Quick Language-Action Preference learning, a Bayesian framework that fuses physical and language feedback to infer reward functions in real time. Our key insight is to treat language as a probabilistic observation over the user's latent preferences, clarifying which reward features matter and how physical corrections should be interpreted. QuickLAP uses Large Language Models (LLMs) to extract reward feature attention masks and preference shifts from free-form utterances, which it integrates with physical feedback in a closed-form update rule. This enables fast, real-time, and robust reward learning that handles ambiguous feedback. In a semi-autonomous driving simulator, QuickLAP reduces reward learning error by over 70% compared to physical-only and heuristic multimodal baselines. A 15-participant user study further validates our approach: participants found QuickLAP significantly more understandable and collaborative, and preferred its learned behavior over baselines. Code is available at this https URL.
>
---
#### [replaced 070] Beyond Policy Optimization: A Data Curation Flywheel for Sparse-Reward Long-Horizon Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于智能体规划任务，解决稀疏奖励和长周期规划中的问题。提出BPO框架，通过数据循环优化提升模型性能。**

- **链接: [https://arxiv.org/pdf/2508.03018](https://arxiv.org/pdf/2508.03018)**

> **作者:** Yutong Wang; Pengliang Ji; Kaixin Li; Baolong Bi; Tao Feng; Guillaume Sartoretti
>
> **摘要:** Large Language Reasoning Models have demonstrated remarkable success on static tasks, yet their application to multi-round agentic planning in interactive environments faces two fundamental challenges. First, the intractable credit assignment problem renders conventional reinforcement learning ineffective in sparse-reward settings. Second, the computational overhead of verbose, step-by-step reasoning histories is prohibitive. To address these challenges, we propose BPO, a three-stage framework (bootstrapping, extrapolation, and refinement) that establishes a self-improving data flywheel to develop robust reasoning models for long-horizon, sparse-reward environments. Our framework first bootstraps efficient reasoning using the proposed planning quaternions with long-short chain-of-thought fusion. It then extrapolates to out-of-distribution tasks through complexity-stratified curriculum learning. Finally, the model iteratively refines itself by learning exclusively on experiences selected via reward-gated rejection sampling. Experiments on ALFWorld, ScienceWorld, and WebShop demonstrate that our approach achieves state-of-the-art with significant token efficiency, providing a new recipe for reasoning models in agentic planning.
>
---
