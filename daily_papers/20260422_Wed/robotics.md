# 机器人 cs.RO

- **最新发布 37 篇**

- **更新 27 篇**

## 最新发布

#### [new 001] HALO: Hybrid Auto-encoded Locomotion with Learned Latent Dynamics, Poincaré Maps, and Regions of Attraction
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出HALO框架，用于学习腿部机器人的低维简化模型，解决如何从数据中提取稳定动力学结构的问题。**

- **链接: [https://arxiv.org/pdf/2604.18887](https://arxiv.org/pdf/2604.18887)**

> **作者:** Blake Werner; Sergio A. Esteban; Massimiliano De Sa; Max H. Cohen; Aaron D. Ames
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Reduced-order models are powerful for analyzing and controlling high-dimensional dynamical systems. Yet constructing these models for complex hybrid systems such as legged robots remains challenging. Classical approaches rely on hand-designed template models (e.g., LIP, SLIP), which, though insightful, only approximate the underlying dynamics. In contrast, data-driven methods can extract more accurate low-dimensional representations, but it remains unclear when stability and safety properties observed in the latent space meaningfully transfer back to the full-order system. To bridge this gap, we introduce HALO (Hybrid Auto-encoded Locomotion), a framework for learning latent reduced-order models of periodic hybrid dynamics directly from trajectory data. HALO employs an autoencoder to identify a low-dimensional latent state together with a learned latent Poincaré map that captures step-to-step locomotion dynamics. This enables Lyapunov analysis and the construction of an associated region of attraction in the latent space, both of which can be lifted back to the full-order state space through the decoder. Experiments on a simulated hopping robot and full-body humanoid locomotion demonstrate that HALO yields low-dimensional models that retain meaningful stability structure and predict full-order region-of-attraction boundaries.
>
---
#### [new 002] M$^{2}$GRPO: Mamba-based Multi-Agent Group Relative Policy Optimization for Biomimetic Underwater Robots Pursuit
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体协作追踪任务，解决生物仿生水下机器人在长时序、部分可观测环境下的协同追踪问题。提出M$^{2}$GRPO框架，提升追踪成功率与效率。**

- **链接: [https://arxiv.org/pdf/2604.19404](https://arxiv.org/pdf/2604.19404)**

> **作者:** Yukai Feng; Zhiheng Wu; Zhengxing Wu; Junwen Gu; Junzhi Yu
>
> **摘要:** Traditional policy learning methods in cooperative pursuit face fundamental challenges in biomimetic underwater robots, where long-horizon decision making, partial observability, and inter-robot coordination require both expressiveness and stability. To address these issues, a novel framework called Mamba-based multi-agent group relative policy optimization (M$^{2}$GRPO) is proposed, which integrates a selective state-space Mamba policy with group-relative policy optimization under the centralized-training and decentralized-execution (CTDE) paradigm. Specifically, the Mamba-based policy leverages observation history to capture long-horizon temporal dependencies and exploits attention-based relational features to encode inter-agent interactions, producing bounded continuous actions through normalized Gaussian sampling. To further improve credit assignment without sacrificing stability, the group-relative advantages are obtained by normalizing rewards across agents within each episode and optimized through a multi-agent extension of GRPO, significantly reducing the demand for training resources while enabling stable and scalable policy updates. Extensive simulations and real-world pool experiments across team scales and evader strategies demonstrate that M$^{2}$GRPO consistently outperforms MAPPO and recurrent baselines in both pursuit success rate and capture efficiency. Overall, the proposed framework provides a practical and scalable solution for cooperative underwater pursuit with biomimetic robot systems.
>
---
#### [new 003] Reinforcement Learning Enabled Adaptive Multi-Task Control for Bipedal Soccer Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于双足足球机器人控制任务，解决动态环境中运动稳定性与多任务切换问题。提出模块化强化学习框架，实现自适应多任务控制。**

- **链接: [https://arxiv.org/pdf/2604.19104](https://arxiv.org/pdf/2604.19104)**

> **作者:** Yulai Zhang; Yinrong Zhang; Ting Wu; Linqi Ye
>
> **摘要:** Developing bipedal football robots in dynamiccombat environments presents challenges related to motionstability and deep coupling of multiple tasks, as well ascontrol switching issues between different states such as up-right walking and fall recovery. To address these problems,this paper proposes a modular reinforcement learning (RL)framework for achieving adaptive multi-task control. Firstly,this framework combines an open-loop feedforward oscilla-tor with a reinforcement learning-based feedback residualstrategy, effectively separating the generation of basic gaitsfrom complex football actions. Secondly, a posture-driven statemachine is introduced, clearly switching between the ballseeking and kicking network (BSKN) and the fall recoverynetwork (FRN), fundamentally preventing state this http URL FRN is efficiently trained through a progressive forceattenuation curriculum learning strategy. The architecture wasverified in Unity simulations of bipedal robots, demonstratingexcellent spatial adaptability-reliably finding and kicking theball even in restricted corner scenarios-and rapid autonomousfall recovery (with an average recovery time of 0.715 seconds).This ensures seamless and stable operation in complex multi-task environments.
>
---
#### [new 004] Thrust Regulation Through Wing Linkage Modulation on the Aerobat Platform: Piezoelectric Slip-Stick Actuated Regulator Development
- **分类: cs.RO**

- **简介: 论文研究如何通过调节翼连杆长度实现飞行动作的推力控制。针对传统设计中双翼耦合导致无法独立控制的问题，提出使用压电滑移驱动器改变连杆长度，以实现独立推力调节。**

- **链接: [https://arxiv.org/pdf/2604.18900](https://arxiv.org/pdf/2604.18900)**

> **作者:** Luca Ciampaglia
>
> **摘要:** Aerobat is a bat-inspired flapping-wing robot with a wing gait generate by the computational structure, a planar linkage of carbon fiber links driven by a single motor. This design minimizes weight but couples both wings to a shared input motor, eliminating independent thrust control and preventing asymmetric maneuvers. This thesis investigates thrust regulation by modifying the effective length of the first radius link $R_1$ in the computational structure. Static experiments using FDM-printed $R_1$ links at three lengths (28.58, 29.33, and 30.08 mm) across 3,4, and 5 Hz flapping frequencies demonstrated that a 1.5 mm length increase produced a 37% increase in peak lift force and shifted peak force timing within the downstroke. An additional experiment using a string-actuated regulator mechanism was performed. Further actuation methods were evaluated: sub-gram micro-servo and piezoelectric slip-stick. After both the string-tension and micro-servo actuation methods failed due to structural member compliance and motor fragility respectively, a TULA-50 piezoelectric slip-stick actuator was selected. Multiple force-amplifying mechanisms were prototyped, resulting in a direct-drive variable-length mechanism. This final mechanism was demonstrated in a preliminary bench-top test, though insufficient force output prevented dynamic testing during flapping. This work establishes linkage-length modulation via embedded slip-stick actuation as a viable approach to independent wing thrust control.
>
---
#### [new 005] Multi-Cycle Spatio-Temporal Adaptation in Human-Robot Teaming
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机协作任务，解决人机计划优化问题。通过建模个体空间与时间行为，提出RAPIDDS框架，联合调整任务调度和机器人运动，提升效率与安全性。**

- **链接: [https://arxiv.org/pdf/2604.19670](https://arxiv.org/pdf/2604.19670)**

> **作者:** Alex Cuellar; Michael Hagenow; Julie Shah
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Effective human-robot teaming is crucial for the practical deployment of robots in human workspaces. However, optimizing joint human-robot plans remains a challenge due to the difficulty of modeling individualized human capabilities and preferences. While prior research has leveraged the multi-cycle structure of domains like manufacturing to learn an individual's tendencies and adapt plans over repeated interactions, these techniques typically consider task-level and motion-level adaptation in isolation. Task-level methods optimize allocation and scheduling but often ignore spatial interference in close-proximity scenarios; conversely, motion-level methods focus on collision avoidance while ignoring the broader task context. This paper introduces RAPIDDS, a framework that unifies these approaches by modeling an individual's spatial behavior (motion paths) and temporal behavior (time required to complete tasks) over multiple cycles. RAPIDDS then jointly adapts task schedules and steers diffusion models of robot motions to maximize efficiency and minimize proximity accounting for these individualized models. We demonstrate the importance of this dual adaptation through an ablation study in simulation and a physical robot scenario using a 7-DOF robot arm. Finally, we present a user study (n=32) showing significant plan improvement compared to non-adaptive systems across both objective metrics, such as efficiency and proximity, and subjective measures, including fluency and user preference. See this paper's companion video at: this https URL.
>
---
#### [new 006] Gated Memory Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Gated Memory Policy（GMP），解决机器人操作任务中的记忆需求问题，通过记忆门控和注意力机制提升性能，适用于非马尔可夫任务。**

- **链接: [https://arxiv.org/pdf/2604.18933](https://arxiv.org/pdf/2604.18933)**

> **作者:** Yihuai Gao; Jinyun Liu; Shuang Li; Shuran Song
>
> **摘要:** Robotic manipulation tasks exhibit varying memory requirements, ranging from Markovian tasks that require no memory to non-Markovian tasks that depend on historical information spanning single or multiple interaction trials. Surprisingly, simply extending observation histories of a visuomotor policy often leads to a significant performance drop due to distribution shift and overfitting. To address these issues, we propose Gated Memory Policy (GMP), a visuomotor policy that learns both when to recall memory and what to recall. To learn when to recall memory, GMP employs a learned memory gate mechanism that selectively activates history context only when necessary, improving robustness and reactivity. To learn what to recall efficiently, GMP introduces a lightweight cross-attention module that constructs effective latent memory representations. To further enhance robustness, GMP injects diffusion noise into historical actions, mitigating sensitivity to noisy or inaccurate histories during both training and inference. On our proposed non-Markovian benchmark MemMimic, GMP achieves a 30.1% average success rate improvement over long-history baselines, while maintaining competitive performance on Markovian tasks in RoboMimic. All code, data and in-the-wild deployment instructions are available on our project website this https URL.
>
---
#### [new 007] LiveVLN: Breaking the Stop-and-Go Loop in Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决实时部署中因感知-推理-执行循环导致的停顿问题。通过引入多步动作延续机制，提升导航连续性与效率。**

- **链接: [https://arxiv.org/pdf/2604.19536](https://arxiv.org/pdf/2604.19536)**

> **作者:** Xiangchen Wang; Weiye Zhu; Teng Wang; TianTian Geng; Zekai Zhang; Zhiyuan Qi; Jinyu Yang; Feng Zheng
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Recent navigation systems achieve strong benchmark results, yet real-world deployment often remains visibly stop-and-go. This bottleneck arises because the sense-inference-execution loop is still blocking: after each new observation, the controller must wait for sensing, transmission, and inference before motion can continue. Reducing action-generation cost alone therefore does not remove redundant waiting. To address this issue, we present LiveVLN, a training-free framework for more continuous embodied navigation by augmenting pretrained VLM navigators with multi-step action continuation. Instead of pausing for each full sense-and-inference round, LiveVLN overlaps execution with the processing of newly arrived observations, allowing refreshed future actions to be handed off before the current executable prefix is exhausted. This design keeps actions continuously available during motion, reducing idle waiting and enabling smoother online execution. The framework operates at runtime and can be integrated with compatible pretrained VLM navigators. Across R2R and RxR, LiveVLN preserves benchmark performance while reducing waiting time and improving action availability. In real-world deployments, it cuts average episode waiting time by up to $77.7\%$ and shortens wall-clock episode time by $12.6\%$ on StreamVLN and $19.6\%$ on NaVIDA, yielding more coherent execution during deployment. Code is available at this https URL.
>
---
#### [new 008] Learning Hybrid-Control Policies for High-Precision In-Contact Manipulation Under Uncertainty
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究高精度接触操作中的混合控制策略，解决不确定环境下力与位置控制的协同问题。通过引入MATCH方法，提升政策在复杂任务中的成功率与安全性。**

- **链接: [https://arxiv.org/pdf/2604.19677](https://arxiv.org/pdf/2604.19677)**

> **作者:** Hunter L. Brown; Geoffrey Hollinger; Stefan Lee
>
> **摘要:** Reinforcement learning-based control policies have been frequently demonstrated to be more effective than analytical techniques for many manipulation tasks. Commonly, these methods learn neural control policies that predict end-effector pose changes directly from observed state information. For tasks like inserting delicate connectors which induce force constraints, pose-based policies have limited explicit control over force and rely on carefully tuned low-level controllers to avoid executing damaging actions. In this work, we present hybrid position-force control policies that learn to dynamically select when to use force or position control in each control dimension. To improve learning efficiency of these policies, we introduce Mode-Aware Training for Contact Handling (MATCH) which adjusts policy action probabilities to explicitly mirror the mode selection behavior in hybrid control. We validate MATCH's learned policy effectiveness using fragile peg-in-hole tasks under extreme localization uncertainty. We find MATCH substantially outperforms pose-control policies -- solving these tasks with up to 10% higher success rates and 5x fewer peg breaks than pose-only policies under common types of state estimation error. MATCH also demonstrates data efficiency equal to pose-control policies, despite learning in a larger and more complex action space. In over 1600 sim-to-real experiments, we find MATCH succeeds twice as often as pose policies in high noise settings (33% vs.~68%) and applies ~30% less force on average compared to variable impedance policies on a Franka FR3 in laboratory conditions.
>
---
#### [new 009] Achieving Interaction Fluidity in a Wizard-of-Oz Robotic System: A Prototype for Fluid Error-Correction
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人语音交互不流畅的问题。提出中断与修正、延迟优化等关键标准，并开发了一个符合这些标准的VR仿真环境。**

- **链接: [https://arxiv.org/pdf/2604.19374](https://arxiv.org/pdf/2604.19374)**

> **作者:** Carlos Baptista De Lima; Julian Hough; Frank Förster; Patrick Holthaus; Yongjun Zheng
>
> **备注:** 5 pages, 1 figure, Workshop on Errors, Mistakes, and Failures in Humans and Robots at 2026 ACM/IEEE International Conference on Human-Robot Interaction
>
> **摘要:** Achieving truly fluid interaction with robots with speech interfaces remains a hard problem, and the experience of current Human-Robot Interaction (HRI) remains laboured and frustrating. Some of the barriers to fluid interaction stem from a lack of a suitable development platform for HRI for improving interaction, even in robotic Wizard-of-Oz (WoZ) modes of operation used for data collection and prototyping. Based on previous systems, we propose the properties of interruptibility and correction (IaC), pollability, latency measurement and optimisation and time-accurate reproducibility of actions from logging data as key criteria for a fluid WoZ system to support fluid error correction. We finish by presenting a Virtual Reality (VR) HRI simulation environment for mobile manipulators which meets these criteria.
>
---
#### [new 010] Assessing VLM-Driven Semantic-Affordance Inference for Non-Humanoid Robot Morphologies
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于机器人感知任务，旨在解决非人形机器人使用VLM进行语义可操作性推理的问题。通过构建混合数据集并分析VLM性能，发现其在不同物体领域表现不一，存在高误漏率但低误报率的特点。**

- **链接: [https://arxiv.org/pdf/2604.19509](https://arxiv.org/pdf/2604.19509)**

> **作者:** Jess Jones; Raul Santos-Rodriguez; Sabine Hauert
>
> **备注:** AAMAS 2026 (main track), 9 pages, 4 figures
>
> **摘要:** Vision-language models (VLMs) have demonstrated remarkable capabilities in understanding human-object interactions, but their application to robotic systems with non-humanoid morphologies remains largely unexplored. This work investigates whether VLMs can effectively infer affordances for robots with fundamentally different embodiments than humans, addressing a critical gap in the deployment of these models for diverse robotic applications. We introduce a novel hybrid dataset that combines annotated real-world robotic affordance-object relations with VLM-generated synthetic scenarios, and perform an empirical analysis of VLM performance across multiple object categories and robot morphologies, revealing significant variations in affordance inference capabilities. Our experiments demonstrate that while VLMs show promising generalisation to non-humanoid robot forms, their performance is notably inconsistent across different object domains. Critically, we identify a consistent pattern of low false positive rates but high false negative rates across all morphologies and object categories, indicating that VLMs tend toward conservative affordance predictions. Our analysis reveals that this pattern is particularly pronounced for novel tool use scenarios and unconventional object manipulations, suggesting that effective integration of VLMs in robotic systems requires complementary approaches to mitigate over-conservative behaviour while preserving the inherent safety benefits of low false positive rates.
>
---
#### [new 011] Forward Dynamics of Variable Topology Mechanisms - The Case of Constraint Activation
- **分类: cs.RO; math.DG; math.DS; math.NA; physics.class-ph**

- **简介: 该论文属于机械动力学任务，解决可变拓扑机构的前向动力学问题，提出拓扑切换时的物理过渡条件，并通过两种方法进行计算验证。**

- **链接: [https://arxiv.org/pdf/2604.19419](https://arxiv.org/pdf/2604.19419)**

> **作者:** Andreas Mueller
>
> **摘要:** Many mechanical systems exhibit changes in their kinematic topology altering the mobility. Ideal contact is the best known cause, but also stiction and controlled locking of parts of a mechanism lead to topology changes. The latter is becoming an important issue in human-machine interaction. Anticipating the dynamic behavior of variable topology mechanisms requires solving a non-smooth dynamic problem. The core challenge is a physically meaningful transition condition at the topology switching events. Such a condition is presented in this paper. Two versions are reported, one using projected motion equations in terms of redundant coordinates, and another one using the Voronets equations in terms of minimal coordinates. Their computational properties are discussed. Results are shown for joint locking of a planar 3R mechanisms and a 6DOF industrial manipulator.
>
---
#### [new 012] VLA Foundry: A Unified Framework for Training Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SE**

- **简介: 该论文提出VLA Foundry框架，统一训练视觉-语言-动作模型，解决多模态模型训练不兼容问题，支持从头训练和预训练模型微调。**

- **链接: [https://arxiv.org/pdf/2604.19728](https://arxiv.org/pdf/2604.19728)**

> **作者:** Jean Mercat; Sedrick Keh; Kushal Arora; Isabella Huang; Paarth Shah; Haruki Nishimura; Shun Iwase; Katherine Liu
>
> **备注:** 32 pages, 16 figures, technical report
>
> **摘要:** We present VLA Foundry, an open-source framework that unifies LLM, VLM, and VLA training in a single codebase. Most open-source VLA efforts specialize on the action training stage, often stitching together incompatible pretraining pipelines. VLA Foundry instead provides a shared training stack with end-to-end control, from language pretraining to action-expert fine-tuning. VLA Foundry supports both from-scratch training and pretrained backbones from Hugging Face. To demonstrate the utility of our framework, we train and release two types of models: the first trained fully from scratch through our LLM-->VLM-->VLA pipeline and the second built on the pretrained Qwen3-VL backbone. We evaluate closed-loop policy performance of both models on LBM Eval, an open-data, open-source simulator. We also contribute usability improvements to the simulator and the STEP analysis tools for easier public use. In the nominal evaluation setting, our fully-open from-scratch model is on par with our prior closed-source work and substituting in the Qwen3-VL backbone leads to a strong multi-task table top manipulation policy outperforming our baseline by a wide margin. The VLA Foundry codebase is available at this https URL and all multi-task model weights are released on this https URL. Additional qualitative videos are available on the project website this https URL.
>
---
#### [new 013] RoboWM-Bench: A Benchmark for Evaluating World Models in Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboWM-Bench，用于评估机器人操作中视频世界模型的物理可执行性。解决视频生成与实际动作执行间的差距问题，通过机器人执行验证行为有效性。**

- **链接: [https://arxiv.org/pdf/2604.19092](https://arxiv.org/pdf/2604.19092)**

> **作者:** Feng Jiang; Yang Chen; Kyle Xu; Yuchen Liu; Haifeng Wang; Zhenhao Shen; Jasper Lu; Shengze Huang; Yuanfei Wang; Chen Xie; Ruihai Wu
>
> **摘要:** Recent advances in large-scale video world models have enabled increasingly realistic future prediction, raising the prospect of leveraging imagined videos for robot learning. However, visual realism does not imply physical plausibility, and behaviors inferred from generated videos may violate dynamics and fail when executed by embodied agents. Existing benchmarks begin to incorporate notions of physical plausibility, but they largely remain perception- or diagnostic-oriented and do not systematically evaluate whether predicted behaviors can be translated into executable actions that complete the intended task. To address this gap, we introduce RoboWM-Bench, a manipulation-centric benchmark for embodiment-grounded evaluation of video world models. RoboWM-Bench converts generated behaviors from both human-hand and robotic manipulation videos into embodied action sequences and validates them through robotic execution. The benchmark spans diverse manipulation scenarios and establishes a unified protocol for consistent and reproducible evaluation. Using RoboWM-Bench, we evaluate state-of-the-art video world models and find that reliably generating physically executable behaviors remains an open challenge. Common failure modes include errors in spatial reasoning, unstable contact prediction, and non-physical deformations. While finetuning on manipulation data yields improvements, physical inconsistencies still persist, suggesting opportunities for more physically grounded video generation for robots.
>
---
#### [new 014] Multimodal embodiment-aware navigation transformer
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决环境变化下碰撞避让能力下降的问题。通过多模态融合与轨迹排序机制提升导航鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.19267](https://arxiv.org/pdf/2604.19267)**

> **作者:** Louis Dezons; Quentin Picard; Rémi Marsal; François Goulette; David Filliat
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Goal-conditioned navigation models for ground robots trained using supervised learning show promising zero-shot transfer, but their collision-avoidance capability nevertheless degrades under distribution shift, i.e. environmental, robot or sensor configuration changes. We propose ViLiNT a multimodal, attention-based policy for goal navigation, trained on heterogeneous data from multiple platforms and environments, which improves robustness with two key features. First, we fuse RGB images, 3D LiDAR point clouds, a goal embedding and a robot's embodiment descriptor with a transformer architecture to capture complementary geometry and appearance cues. The transformer's output is used to condition a diffusion model that generates navigable trajectories. Second, using automatically generated offline labels, we train a path clearance prediction head for scoring and ranking trajectories produced by the diffusion model. The diffusion conditioning as well as the trajectory ranking head depend on a robot's embodiment token that allows our model to generate and select trajectories with respect to the robot's dimensions. Across three simulated environments, ViLiNT improves Success Rate on average by 166\% over equivalent state-of-the-art vision-only baseline (NoMaD). This increase in performance is confirmed through real-world deployments of a rover navigating in obstacle fields. These results highlight that combining multimodal fusion with our collision prediction mechanism leads to improved off-road navigation robustness.
>
---
#### [new 015] Wrench-Aware Admittance Control for Unknown-Payload Manipulation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人抓取与放置任务，解决未知负载影响定位精度的问题。通过力矩测量和自适应控制，提升搬运与堆叠性能。**

- **链接: [https://arxiv.org/pdf/2604.19469](https://arxiv.org/pdf/2604.19469)**

> **作者:** Hossein Gholampour; Logan E. Beaver
>
> **摘要:** Unknown payloads can strongly affect compliant robotic manipulation, especially when the payload center of mass is not aligned with the tool center point. In this case, the payload generates an offset wrench at the robot wrist. During motion, this wrench is not only related to payload weight, but also to payload inertia. If it is not modeled, the compliant controller can interpret it as an external interaction wrench, which causes unintended compliant motion, larger tracking error, and reduced transport accuracy. This paper presents a wrench-aware admittance control framework for unknown-payload pick-and-place using a UR5e robot. The method uses force-torque measurements in two different roles. First, a three-axis translational excitation term is used to reduce payload-induced force effects during transport without making the robot excessively stiff. Second, after grasping, the controller first estimates payload mass for transport compensation and then estimates the payload CoM offset relative to the TCP using wrist force-torque measurements collected during the subsequent translational motion. This helps improve object placement and stacking behavior. Experimental results show improved transport and placement performance compared with uncorrected placement while preserving compliant motion.
>
---
#### [new 016] RoomRecon: High-Quality Textured Room Layout Reconstruction on Mobile Devices
- **分类: cs.RO**

- **简介: 该论文属于3D室内场景重建任务，旨在解决纹理质量差、无法实时更新和缺乏语义理解的问题。提出RoomRecon系统，结合AR与AI提升纹理效果和计算效率。**

- **链接: [https://arxiv.org/pdf/2604.19025](https://arxiv.org/pdf/2604.19025)**

> **作者:** Seok Joon Kim; Dinh Duc Cao; Federica Spinola; Se Jin Lee; Kyu Sung Cho
>
> **备注:** 23 pages, including supplementary material. Accepted to the 2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). Best Paper Nominee
>
> **摘要:** Widespread RGB-Depth (RGB-D) sensors and advanced 3D reconstruction technologies facilitate the capture of indoor spaces, improving the fields of augmented reality (AR), virtual reality (VR), and extended reality (XR). Nevertheless, current technologies still face limitations, such as the inability to reflect minor scene changes without a complete recapture, the lack of semantic scene understanding, and various texturing challenges that affect the 3D model's visual quality. These issues affect the realism required for VR experiences and other applications such as in interior design and real estate. To address these challenges, we introduce RoomRecon, an interactive, real-time scanning and texturing pipeline for 3D room models. We propose a two-phase texturing pipeline that integrates AR-guided image capturing for texturing and generative AI models to improve texturing quality and provide better replicas of indoor spaces. Moreover, we suggest focusing only on permanent room elements such as walls, floors, and ceilings, to allow for easily customizable 3D models. We conduct experiments in a variety of indoor spaces to assess the texturing quality and speed of our method. The quantitative results and user study demonstrate that RoomRecon surpasses state-of-the-art methods in terms of texturing quality and on-device computation time.
>
---
#### [new 017] Differentiable Satellite Constellation Configuration via Relaxed Coverage and Revisit Objectives
- **分类: cs.RO**

- **简介: 该论文属于卫星星座设计任务，解决覆盖与重访优化问题。通过引入连续松弛方法，实现梯度优化，提升设计效率与效果。**

- **链接: [https://arxiv.org/pdf/2604.19062](https://arxiv.org/pdf/2604.19062)**

> **作者:** Shreeyam Kacker; Kerri Cahoy
>
> **摘要:** Satellite constellation design requires optimizing orbital parameters across multiple satellites to maximize mission specific metrics. For many types of mission, it is desirable to maximize coverage and minimize revisit gaps over ground targets. Existing approaches to constellation design either restrict the design space to symmetric parametric families such as Walker constellations, or rely on metaheuristic methods that require significant compute and many iterations. Gradient-based optimization has been considered intractable due to the non-differentiability of coverage and revisit metrics, which involve binary visibility indicators and discrete max operations. We introduce four continuous relaxations: soft sigmoid visibility, noisy-OR multi-satellite aggregation, leaky integrator revisit gap tracking, and LogSumExp soft-maximum, which when composed with the $\partial$SGP4 differentiable orbit propagator, yield a fully differentiable pipeline from orbital elements to mission-level objectives. We show that this scheme can recover Walker-Delta geometry from irregular initializations, and discovers elliptical Molniya-like orbits with apogee dwell over extreme latitudes from only gradients. Compared to simulated annealing (SA), genetic algorithm (GA), and differential evolution (DE) baselines, our gradient-based method recovers Walker-equivalent geometry within ${\sim}750$ evaluations, whereas the three black-box baselines plateau at with significantly worse revisit even with roughly four times the evaluation budget.
>
---
#### [new 018] GenerativeMPC: VLM-RAG-guided Whole-Body MPC with Virtual Impedance for Bimanual Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GenerativeMPC，解决双臂移动操作中语义理解与物理控制的融合问题，通过VLM-RAG生成控制参数，提升人机交互的安全性与适应性。**

- **链接: [https://arxiv.org/pdf/2604.19522](https://arxiv.org/pdf/2604.19522)**

> **作者:** Marcelino Julio Fernando; Miguel Altamirano Cabrera; Jeffrin Sam; Yara Mahmoud; Konstantin Gubernatorov; Dzmitry Tsetserukou
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Bimanual mobile manipulation requires a seamless integration between high-level semantic reasoning and safe, compliant physical interaction - a challenge that end-to-end models approach opaquely and classical controllers lack the context to address. This paper presents GenerativeMPC, a hierarchical cyber-physical framework that explicitly bridges semantic scene understanding with physical control parameters for bimanual mobile manipulators. The system utilizes a Vision-Language Model with Retrieval-Augmented Generation (VLM-RAG) to translate visual and linguistic context into grounded control constraints, specifically outputting dynamic velocity limits and safety margins for a Whole-Body Model Predictive Controller (MPC). Simultaneously, the VLM-RAG module modulates virtual stiffness and damping gains for a unified impedance-admittance controller, enabling context-aware compliance during human-robot interaction. Our framework leverages an experience-driven vector database to ensure consistent parameter grounding without retraining. Experimental results in MuJoCo, IsaacSim, and on a physical bimanual platform confirm a 60% speed reduction near humans and safe, socially-aware navigation and manipulation through semantic-to-physical parameter grounding. This work advances the field of human-centric cybernetics by grounding large-scale cognitive models into predictable, high-frequency physical control loops.
>
---
#### [new 019] AeroBridge-TTA: Test-Time Adaptive Language-Conditioned Control for UAVs
- **分类: cs.RO**

- **简介: 该论文提出AeroBridge-TTA，解决UAV语言控制中的执行不匹配问题。通过测试时自适应机制提升控制性能，显著改善分布外表现。**

- **链接: [https://arxiv.org/pdf/2604.19059](https://arxiv.org/pdf/2604.19059)**

> **作者:** Lingxue Lyu
>
> **摘要:** Language-guided unmanned aerial vehicles (UAVs) often fail not from bad reasoning or perception, but from execution mismatch: the gap between a planned trajectory and the controller's ability to track it when the real dynamics differ from training (mass changes, drag shifts, actuator delay, wind). We propose AeroBridge-TTA, a language-conditioned control pipeline that targets this gap with test-time adaptation. It has three parts: a language encoder that maps the command into a subgoal, an adaptive policy conditioned on the subgoal and a learned latent, and a test-time adaptation (TTA) module that updates the latent online from observed transitions. On five language-conditioned UAV tasks under 13 mismatch conditions with the same domain randomization, AeroBridge-TTA ties a strong PPO-MLP baseline in-distribution and wins all 5 out-of-distribution (OOD) conditions, +22.0 pts on average (62.7% vs. 40.7%); the +8.5 pt overall gain comes entirely from the OOD regime. A same-weights ablation that only changes the step size $\alpha$ shows the latent update itself is responsible for a $4.6\times$ OOD lift.
>
---
#### [new 020] Quadruped Parkour Learning: Sparsely Gated Mixture of Experts with Visual Input
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在提升视觉引导的四足机器人越障能力。通过引入稀疏门控专家混合架构，提升性能与计算效率。**

- **链接: [https://arxiv.org/pdf/2604.19344](https://arxiv.org/pdf/2604.19344)**

> **作者:** Michael Ziegltrum; Jianhao Jiao; Tianhu Peng; Chengxu Zhou; Dimitrios Kanoulas
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Robotic parkour provides a compelling benchmark for advancing locomotion over highly challenging terrain, including large discontinuities such as elevated steps. Recent approaches have demonstrated impressive capabilities, including dynamic climbing and jumping, but typically rely on sequential multilayer perceptron (MLP) architectures with densely activated layers. In contrast, sparsely gated mixture-of-experts (MoE) architectures have emerged in the large language model domain as an effective paradigm for improving scalability and performance by activating only a subset of parameters at inference time. In this work, we investigate the application of sparsely gated MoE architectures to vision-based robotic parkour. We compare control policies based on standard MLPs and MoE architectures under a controlled setting where the number of active parameters at inference time is matched. Experimental results on a real Unitree Go2 quadruped robot demonstrate clear performance gains, with the MoE policy achieving double the number of successful trials in traversing large obstacles compared to a standard MLP baseline. We further show that achieving comparable performance with a standard MLP requires scaling its parameter count to match that of the total MoE model, resulting in a 14.3\% increase in computation time. These results highlight that sparsely gated MoE architectures provide a favorable trade-off between performance and computational efficiency, enabling improved scaling of control policies for vision-based robotic parkour. An anonymized link to the codebase is this https URL.
>
---
#### [new 021] Task-Adaptive Admittance Control for Human-Quadrotor Cooperative Load Transportation with Dynamic Cable-Length Regulation
- **分类: cs.RO**

- **简介: 该论文研究人机协同运输任务，解决动态绳长下安全高效运输问题。提出一种自适应阻抗控制器，提升系统响应与平滑性。**

- **链接: [https://arxiv.org/pdf/2604.18905](https://arxiv.org/pdf/2604.18905)**

> **作者:** Shuai Li; Ton T. H. Duong; Damiano Zanotto
>
> **备注:** Preprint of accepted manuscript to be published in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** The collaboration between humans and robots is critical in many robotic applications, especially in those requiring physical human-robot interaction (pHRI). Previous research in pHRI has largely focused on robotic manipulators, employing impedance or admittance control to maintain operational safety. Conversely, research in human-quadrotor cooperative load transportation (CLT) is still in its infancy. This letter introduces a novel admittance controller designed for safe and effective human-quadrotor CLT using a quadrotor equipped with an actively-controlled winch. The proposed method accounts for the system's coupled dynamics, allowing the quadrotor and its cable to dynamically adapt to contact forces during CLT tasks, thereby enhancing responsiveness. We experimentally validated the task-adaptive capability of the controller across the entire CLT process, including in-place loading/unloading and load transporting tasks. To this end, we compared the system performances against a conventional approach, using both variable and fixed cable lengths under low- and high-stiffness conditions. Results demonstrate that the proposed method outperforms the conventional approach in terms of system responsiveness and motion smoothness, leading to improved CLT capabilities.
>
---
#### [new 022] AI-Enabled Image-Based Hybrid Vision/Force Control of Tendon-Driven Aerial Continuum Manipulators
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机械臂控制任务，解决空中连续机械臂在复杂环境中的精确控制问题。通过AI融合视觉与力反馈，实现自主交互与误差稳定。**

- **链接: [https://arxiv.org/pdf/2604.18961](https://arxiv.org/pdf/2604.18961)**

> **作者:** Shayan Sepahvand; Farrokh Janabi-Sharifi; Farhad Aghili
>
> **摘要:** This paper presents an AI-enabled cascaded hybrid vision/force control framework for tendon-driven aerial continuum manipulators based on constant-strain modeling in $SE(3)$ as a coupled system. The proposed controller is designed to enable autonomous, physical interaction with a static environment while stabilizing the image feature error. The developed strategy combines the cascaded fast fixed-time sliding mode control and a radial basis function neural network to cope with the uncertainties in the image acquired by the eye-in-hand monocular camera and the measurements from the force sensing apparatus. This ensures rapid, online learning of the vision- and force-related uncertainties without requiring offline training. Furthermore, the features are extracted via a state-of-the-art graph neural network architecture employed by a visual servoing framework using line features, rather than relying on heuristic geometric line extractors, to concurrently contribute to tracking the desired normal interaction force during contact and regulating the image feature error. A comparative study benchmarks the proposed controller against established rigid-arm aerial manipulation methods, evaluating robustness across diverse scenarios and feature extraction strategies. The simulation and experimental results showcase the effectiveness of the proposed methodology under various initial conditions and demonstrate robust performance in executing manipulation tasks.
>
---
#### [new 023] UniT: Toward a Unified Physical Language for Human-to-Humanoid Policy Learning and World Modeling
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出UniT，解决人类与人形机器人之间运动差异带来的数据迁移问题，通过统一物理语言实现高效策略学习和世界建模。**

- **链接: [https://arxiv.org/pdf/2604.19734](https://arxiv.org/pdf/2604.19734)**

> **作者:** Boyu Chen; Yi Chen; Lu Qiu; Jerry Bai; Yuying Ge; Yixiao Ge
>
> **备注:** Project page: this https URL
>
> **摘要:** Scaling humanoid foundation models is bottlenecked by the scarcity of robotic data. While massive egocentric human data offers a scalable alternative, bridging the cross-embodiment chasm remains a fundamental challenge due to kinematic mismatches. We introduce UniT (Unified Latent Action Tokenizer via Visual Anchoring), a framework that establishes a unified physical language for human-to-humanoid transfer. Grounded in the philosophy that heterogeneous kinematics share universal visual consequences, UniT employs a tri-branch cross-reconstruction mechanism: actions predict vision to anchor kinematics to physical outcomes, while vision reconstructs actions to filter out irrelevant visual confounders. Concurrently, a fusion branch synergies these purified modalities into a shared discrete latent space of embodiment-agnostic physical intents. We validate UniT across two paradigms: 1) Policy Learning (VLA-UniT): By predicting these unified tokens, it effectively leverages diverse human data to achieve state-of-the-art data efficiency and robust out-of-distribution (OOD) generalization on both humanoid simulation benchmark and real-world deployments, notably demonstrating zero-shot task transfer. 2) World Modeling (WM-UniT): By aligning cross-embodiment dynamics via unified tokens as conditions, it realizes direct human-to-humanoid action transfer. This alignment ensures that human data seamlessly translates into enhanced action controllability for humanoid video generation. Ultimately, by inducing a highly aligned cross-embodiment representation (empirically verified by t-SNE visualizations revealing the convergence of human and humanoid features into a shared manifold), UniT offers a scalable path to distill vast human knowledge into general-purpose humanoid capabilities.
>
---
#### [new 024] Autonomous UAV Pipeline Near-proximity Inspection via Disturbance-Aware Predictive Visual Servoing
- **分类: cs.RO**

- **简介: 该论文属于无人机自主管道近距检测任务，解决复杂环境下可靠自主检测问题。提出基于视觉伺服的预测控制框架，融合动态模型与图像特征，提升检测精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.19618](https://arxiv.org/pdf/2604.19618)**

> **作者:** Wen Li; Hui Wang; Jinya Su; Cunjia Liu; Wen-Hua Chen; Shihua Li
>
> **备注:** 11 pages, 12 figures, Under Review
>
> **摘要:** Reliable pipeline inspection is critical to safe energy transportation, but is constrained by long distances, complex terrain, and risks to human inspectors. Unmanned aerial vehicles provide a flexible sensing platform, yet reliable autonomous inspection remains challenging. This paper presents an autonomous quadrotor near-proximity pipeline inspection framework for three-dimensional scenarios based on image-based visual servoing model predictive control (VMPC). A unified predictive model couples quadrotor dynamics with image feature kinematics, enabling direct image-space prediction within the control loop. To address low-rate visual updates, measurement noise, and environmental uncertainties, an extended-state Kalman filtering scheme with image feature prediction (ESKF-PRE) is developed, and the estimated lumped disturbances are incorporated into the VMPC prediction model, yielding the ESKF-PRE-VMPC framework. A terrain-adaptive velocity design is introduced to maintain the desired cruising speed while generating vertical velocity references over unknown terrain slopes without prior terrain information. The framework is validated in high-fidelity Gazebo simulations and real-world experiments. In real-world tests, the proposed method reduces RMSE by 52.63% and 75.04% in pipeline orientation and lateral deviation in the image, respectively, for straight-pipeline inspection without wind, and successfully completes both wind-disturbance and bend-pipeline tasks where baseline method fails. An open-source nano quadrotor is modified for indoor experimentation.
>
---
#### [new 025] Mask World Model: Predicting What Matters for Robust Robot Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决传统世界模型因关注无关视觉因素导致泛化能力差的问题。提出Mask World Model，通过预测语义掩码提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.19683](https://arxiv.org/pdf/2604.19683)**

> **作者:** Yunfan Lou; Xiaowei Chi; Xiaojie Zhang; Zezhong Qian; Chengxuan Li; Rongyu Zhang; Yaoxu Lyu; Guoyu Song; Chuyao Fu; Haoxuan Xu; Pengwei Wang; Shanghang Zhang
>
> **备注:** 16 pages,5 figures
>
> **摘要:** World models derived from large-scale video generative pre-training have emerged as a promising paradigm for generalist robot policy learning. However, standard approaches often focus on high-fidelity RGB video prediction, this can result in overfitting to irrelevant factors, such as dynamic backgrounds and illumination changes. These distractions reduce the model's ability to generalize, ultimately leading to unreliable and fragile control policies. To address this, we introduce the Mask World Model (MWM), which leverages video diffusion architectures to predict the evolution of semantic masks instead of pixels. This shift imposes a geometric information bottleneck, forcing the model to capture essential physical dynamics and contact relations while filtering out visual noise. We seamlessly integrate this mask dynamics backbone with a diffusion-based policy head to enable robust end-to-end control. Extensive evaluations demonstrate the superiority of MWM on the LIBERO and RLBench simulation benchmarks, significantly outperforming the state-of-the-art RGB-based world models. Furthermore, real-world experiments and robustness evaluation (via random token pruning) reveal that MWM exhibits superior generalization capabilities and robust resilience to texture information loss.
>
---
#### [new 026] Multi-Gait Learning for Humanoid Robots Using Reinforcement Learning with Selective Adversarial Motion Prior
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，解决 humanoid 机器人在统一框架下学习多种步态的问题。通过引入选择性对抗运动先验策略，提升学习效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.19102](https://arxiv.org/pdf/2604.19102)**

> **作者:** Yuanye Wu; Keyi Wang; Linqi Ye; Boyang Xing
>
> **摘要:** Learning diverse locomotion skills for humanoid robots in a unified reinforcement learning framework remains challenging due to the conflicting requirements of stability and dynamic expressiveness across different gaits. We present a multi-gait learning approach that enables a humanoid robot to master five distinct gaits -- walking, goose-stepping, running, stair climbing, and jumping -- using a consistent policy structure, action space, and reward formulation. The key contribution is a selective Adversarial Motion Prior (AMP) strategy: AMP is applied to periodic, stability-critical gaits (walking, goose-stepping, stair climbing) where it accelerates convergence and suppresses erratic behavior, while being deliberately omitted for highly dynamic gaits (running, jumping) where its regularization would over-constrain the motion. Policies are trained via PPO with domain randomization in simulation and deployed on a physical 12-DOF humanoid robot through zero-shot sim-to-real transfer. Quantitative comparisons demonstrate that selective AMP outperforms a uniform AMP policy across all five gaits, achieving faster convergence, lower tracking error, and higher success rates on stability-focused gaits without sacrificing the agility required for dynamic ones.
>
---
#### [new 027] Warmth and Competence in the Swarm: Designing Effective Human-Robot Teams
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究人机协作中机器人团队的社会感知，探讨如何通过设计提升人类对机器人群体的温暖与能力评价，以优化人机合作效果。**

- **链接: [https://arxiv.org/pdf/2604.19270](https://arxiv.org/pdf/2604.19270)**

> **作者:** Genki Miyauchi; Roderich Groß; Chaona Chen
>
> **备注:** 15 pages, 4 figures, camera-ready version for ANTS 2026
>
> **摘要:** As groups of robots increasingly collaborate with humans, understanding how humans perceive them is critical for designing effective human-robot teams. While prior research examined how humans interpret and evaluate the abilities and intentions of individual agents, social perception of robot teams remains relatively underexplored. Drawing on the competence-warmth framework, we conducted two studies manipulating swarm behaviors in completing a collective search task and measured the social perception of swarm behaviors when human participants are either observers (Study 1) and operators (Study 2). Across both studies, our results show that variations in swarm behaviors consistently influenced participants' perceptions of warmth and competence. Notably, longer broadcast durations increased perceived warmth; larger separation distances increased perceived competence. Interestingly, individual robot speed had no effect on either of the perceptions. Furthermore, our results show that these social perceptions predicted participants' team preferences more strongly than task performance. Participants preferred robot teams that were both warm and competent, not those that completed tasks most quickly. These findings demonstrate that human-robot interaction dynamically shapes social perception, underscoring the importance of integrating both technical and social considerations when designing robot swarms for effective human-robot collaboration.
>
---
#### [new 028] Multi-Step Gaussian Process Propagation for Adaptive Path Planning
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决机器人在不确定环境中的自适应路径问题。提出一种基于高斯过程的方法，优化未来路径点，提升对藻华的识别精度。**

- **链接: [https://arxiv.org/pdf/2604.19148](https://arxiv.org/pdf/2604.19148)**

> **作者:** Alex Beaudin; Bjørn Andreas Kristiansen; Kristoffer Gryte; Corrado Chiatante; Morten Omholt Alver; Murat Arcak; Tor Arne Johansen
>
> **摘要:** Efficient and robust path planning hinges on combining all accessible information sources. In particular, the task of path planning for robotic environmental exploration and monitoring depends highly on the current belief of the world. To capture the uncertainty in the belief, we present a Gaussian process based path planning method that adapts to multi-modal environmental sensing data and incorporates state and input constraints. To solve the path planning problem, we optimize over future waypoints in a receding horizon fashion, and our cost is thus a function of the Gaussian process posterior over all these waypoints. We demonstrate this method, dubbed OLAhGP, on an autonomous surface vessel using oceanic algal bloom data from both a high-fidelity model and in-situ sensing data in a monitoring scenario. Our simulated and experimental results demonstrate significant improvement over existing methods. With the same number of samples, our method generates more informative paths and achieves greater accuracy in identifying algal blooms in chlorophyll a rich waters, measured with respect to total misclassification probability and binary misclassification rate over the domain of interest.
>
---
#### [new 029] A Gesture-Based Visual Learning Model for Acoustophoretic Interactions using a Swarm of AcoustoBots
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决AcoustoBots缺乏直观控制的问题。通过手势识别与视觉学习模型实现多模态交互，提升控制效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.19643](https://arxiv.org/pdf/2604.19643)**

> **作者:** Alex Lin; Lei Gao; Narsimlu Kemsaram; Sriram Subramanian
>
> **备注:** This paper has been accepted for publication in the Proceedings of the 2026 4th International Conference on Robotics, Control and Vision Engineering (RCVE 2026)
>
> **摘要:** AcoustoBots are mobile acoustophoretic robots capable of delivering mid-air haptics, directional audio, and acoustic levitation, but existing implementations rely on scripted commands and lack an intuitive interface for real-time human control. This work presents a gesture-based visual learning framework for contactless human-swarm interaction with a multimodal AcoustoBot platform. The system combines ESP32-CAM gesture capture, PhaseSpace motion tracking, centralized processing, and an OpenCLIP-based visual learning model (VLM) with linear probing to classify three hand gestures and map them to haptics, audio, and levitation modalities. Validation accuracy improved from about 67% with a small dataset to nearly 98% with the largest dataset. In integrated experiments with two AcoustoBots, the system achieved an overall gesture-to-modality switching accuracy of 87.8% across 90 trials, with an average end-to-end latency of 3.95 seconds. These results demonstrate the feasibility of using a vision-language-model-based gesture interface for multimodal human-swarm interaction. While the current system is limited by centralized processing, a static gesture set, and controlled-environment evaluation, it establishes a foundation for more expressive, scalable, and accessible swarm robotic interfaces.
>
---
#### [new 030] Vision-Based Human Awareness Estimation for Enhanced Safety and Efficiency of AMRs in Industrial Warehouses
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人机协作任务，旨在提升工业仓库中AMR的安全与效率。通过视觉估计人类对AMR的意识，解决AMR因误判人类状态而过度避让的问题。**

- **链接: [https://arxiv.org/pdf/2604.18627](https://arxiv.org/pdf/2604.18627)**

> **作者:** Maximilian Haug; Christian Stippel; Lukas Pscherer; Benjamin Schwendinger; Ralph Hoch; Angel Gaydarov; Sebastian Schlund; Thilo Sauter
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Ensuring human safety is of paramount importance in warehouse environments that feature mixed traffic of human workers and autonomous mobile robots (AMRs). Current approaches often treat humans as generic dynamic obstacles, leading to conservative AMR behaviors like slowing down or detouring, even when workers are fully aware and capable of safely sharing space. This paper presents a real-time vision-based method to estimate human awareness of an AMR using a single RGB camera. We integrate state-of-the-art 3D human pose lifting with head orientation estimation to ascertain a human's position relative to the AMR and their viewing cone, thereby determining if the human is aware of the AMR. The entire pipeline is validated using synthetically generated data within NVIDIA Isaac Sim, a robust physics-accurate robotics simulation environment. Experimental results confirm that our system reliably detects human positions and their attention in real time, enabling AMRs to safely adapt their motion based on human awareness. This enhancement is crucial for improving both safety and operational efficiency in industrial and factory automation settings.
>
---
#### [new 031] Mind2Drive: Predicting Driver Intentions from EEG in Real-world On-Road Driving
- **分类: cs.CV; cs.HC; cs.LG; cs.RO**

- **简介: 该论文属于驾驶员意图预测任务，旨在通过EEG信号提升驾驶安全。研究提出一个基于EEG的预测框架，解决真实道路中信号不稳定和认知复杂性问题，验证了多种深度学习模型并发现最佳性能区间。**

- **链接: [https://arxiv.org/pdf/2604.19368](https://arxiv.org/pdf/2604.19368)**

> **作者:** Ghadah Alosaimi; Hanadi Alhamdan; Wenke E; Stamos Katsigiannis; Amir Atapour-Abarghouei; Toby P. Breckon
>
> **备注:** 8 pages, 4 figures, 6 tables, conference
>
> **摘要:** Predicting driver intention from neurophysiological signals offers a promising pathway for enhancing proactive safety in advanced driver assistance systems, yet remains challenging in real-world driving due to EEG signal non-stationarity and the complexity of cognitive-motor preparation. This study proposes and evaluates an EEG-based driver intention prediction framework using a synchronised multi-sensor platform integrated into a real electric vehicle. A real-world on-road dataset was collected across 32 driving sessions, and 12 deep learning architectures were evaluated under consistent experimental conditions. Among the evaluated architectures, TSCeption achieved the highest average accuracy (0.907) and Macro-F1 score (0.901). The proposed framework demonstrates strong temporal stability, maintaining robust decoding performance up to 1000 ms before manoeuvre execution with minimal degradation. Furthermore, additional analyses reveal that minimal EEG preprocessing outperforms artefact-handling pipelines, and prediction performance peaks within a 400-600 ms interval, corresponding to a critical neural preparatory phase preceding driving manoeuvres. Overall, these findings support the feasibility of early and stable EEG-based driver intention decoding under real-world on-road conditions. Code: this https URL.
>
---
#### [new 032] Feasibility of Indoor Frame-Wise Lidar Semantic Segmentation via Distillation from Visual Foundation Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究室内激光雷达帧级语义分割任务，解决标注数据稀缺问题。通过知识蒸馏，利用视觉基础模型提升激光雷达分割效果。**

- **链接: [https://arxiv.org/pdf/2604.18831](https://arxiv.org/pdf/2604.18831)**

> **作者:** Haiyang Wu; Juan J. Gonzales Torres; George Vosselman; Ville Lehtola
>
> **摘要:** Frame-wise semantic segmentation of indoor lidar scans is a fundamental step toward higher-level 3D scene understanding and mapping applications. However, acquiring frame-wise ground truth for training deep learning models is costly and time-consuming. This challenge is largely addressed, for imagery, by Visual Foundation Models (VFMs) which segment image frames. The same VFMs may be used to train a lidar scan frame segmentation model via a 2D-to-3D distillation pipeline. The success of such distillation has been shown for autonomous driving scenes, but not yet for indoor scenes. Here, we study the feasibility of repeating this success for indoor scenes, in a frame-wise distillation manner by coupling each lidar scan with a VFM-processed camera image. The evaluation is done using indoor SLAM datasets, where pseudo-labels are used for downstream evaluation. Also, a small manually annotated lidar dataset is provided for validation, as there are no other lidar frame-wise indoor datasets with semantics. Results show that the distilled model achieves up to 56% mIoU under pseudo-label evaluation and around 36% mIoU with real-label, demonstrating the feasibility of cross-modal distillation for indoor lidar semantic segmentation without manual annotations.
>
---
#### [new 033] Localization-Guided Foreground Augmentation in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶感知任务，解决恶劣天气下场景几何信息缺失问题。提出LG-FA模块，通过增强几何上下文提升前景感知与定位精度。**

- **链接: [https://arxiv.org/pdf/2604.18940](https://arxiv.org/pdf/2604.18940)**

> **作者:** Jiawei Yong; Deyuan Qu; Qi Chen; Kentaro Oguchi; Shintaro Fukushima
>
> **摘要:** Autonomous driving systems often degrade under adverse visibility conditions-such as rain, nighttime, or snow-where online scene geometry (e.g., lane dividers, road boundaries, and pedestrian crossings) becomes sparse or fragmented. While high-definition (HD) maps can provide missing structural context, they are costly to construct and maintain at scale. We propose Localization-Guided Foreground Augmentation (LG-FA), a lightweight and plug-and-play inference module that enhances foreground perception by enriching geometric context online. LG-FA: (i) incrementally constructs a sparse global vector layer from per-frame Bird's-Eye View (BEV) predictions; (ii) estimates ego pose via class-constrained geometric alignment, jointly improving localization and completing missing local topology; and (iii) reprojects the augmented foreground into a unified global frame to improve per-frame predictions. Experiments on challenging nuScenes sequences demonstrate that LG-FA improves the geometric completeness and temporal stability of BEV representations, reduces localization error, and produces globally consistent lane and topology reconstructions. The module can be seamlessly integrated into existing BEV-based perception systems without backbone modification. By providing a reliable geometric context prior, LG-FA enhances temporal consistency and supplies stable structural support for downstream modules such as tracking and decision-making.
>
---
#### [new 034] SafetyALFRED: Evaluating Safety-Conscious Planning of Multimodal Large Language Models
- **分类: cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于多模态大模型安全评估任务，旨在解决模型在真实环境中的安全风险应对问题。通过构建SafetyALFRED数据集，评估模型在识别与主动规避危险方面的能力。**

- **链接: [https://arxiv.org/pdf/2604.19638](https://arxiv.org/pdf/2604.19638)**

> **作者:** Josue Torres-Fonseca; Naihao Deng; Yinpei Dai; Shane Storks; Yichi Zhang; Rada Mihalcea; Casey Kennington; Joyce Chai
>
> **备注:** Work accepted at ACL 2026 Findings
>
> **摘要:** Multimodal Large Language Models are increasingly adopted as autonomous agents in interactive environments, yet their ability to proactively address safety hazards remains insufficient. We introduce SafetyALFRED, built upon the embodied agent benchmark ALFRED, augmented with six categories of real-world kitchen hazards. While existing safety evaluations focus on hazard recognition through disembodied question answering (QA) settings, we evaluate eleven state-of-the-art models from the Qwen, Gemma, and Gemini families on not only hazard recognition, but also active risk mitigation through embodied planning. Our experimental results reveal a significant alignment gap: while models can accurately recognize hazards in QA settings, average mitigation success rates for these hazards are low in comparison. Our findings demonstrate that static evaluations through QA are insufficient for physical safety, thus we advocate for a paradigm shift toward benchmarks that prioritize corrective actions in embodied contexts. We open-source our code and dataset under this https URL
>
---
#### [new 035] Scheduling Analysis of UAV Flight Control Workloads using Raspberry Pi 5 Using PREEMPT_RT Linux
- **分类: eess.SY; cs.OS; cs.RO**

- **简介: 该论文属于实时系统任务，解决UAV控制延迟问题。通过分析PREEMPT_RT Linux在Raspberry Pi 5上的调度性能，验证其降低延迟的有效性。**

- **链接: [https://arxiv.org/pdf/2604.19275](https://arxiv.org/pdf/2604.19275)**

> **作者:** Luiz Giacomossi; Håkan Forsberg; Ivan Tomasic; Baran Çürüklü; Tommaso Cucinotta
>
> **备注:** 9 pages, 8 figures, conference
>
> **摘要:** Modern UAV architectures increasingly aim to unify high-level autonomy and low-level flight control on a single General-Purpose Operating System (GPOS). However, complex multi-core System-on-Chips (SoCs) introduce significant timing indeterminism due to shared resource contention. This paper performs an architectural analysis of the PREEMPT RT Linux kernel on a Raspberry Pi 5, specifically isolating the impact of kernel activation paths (deferred execution SoftIRQs versus real-time direct activation) on a 250 Hz control loop. Results show that under heavy stress, the standard kernel is unsuitable, exhibiting worst-case latencies exceeding 9 ms. In contrast, PREEMPT RT reduced the worst-case latency by nearly 88 percent to under 225 microseconds, enforcing a direct wake-up path that mitigates OS noise. These findings demonstrate that while PREEMPT RT resolves scheduling variance, the residual jitter on modern SoCs is primarily driven by hardware memory contention.
>
---
#### [new 036] Accelerating trajectory optimization with Sobolev-trained diffusion policies
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于轨迹优化任务，旨在提升优化效率。通过学习扩散策略作为初始猜测，解决传统方法收敛慢、依赖初始轨迹的问题。引入一阶信息提升预测精度，减少求解时间。**

- **链接: [https://arxiv.org/pdf/2604.19011](https://arxiv.org/pdf/2604.19011)**

> **作者:** Théotime Le Hellard; Franki Nguimatsia Tiofack; Quentin Le Lidec; Justin Carpentier
>
> **摘要:** Trajectory Optimization (TO) solvers exploit known system dynamics to compute locally optimal trajectories through iterative improvements. A downside is that each new problem instance is solved independently; therefore, convergence speed and quality of the solution found depend on the initial trajectory proposed. To improve efficiency, a natural approach is to warm-start TO with initial guesses produced by a learned policy trained on trajectories previously generated by the solver. Diffusion-based policies have recently emerged as expressive imitation learning models, making them promising candidates for this role. Yet, a counterintuitive challenge comes from the local optimality of TO demonstrations: when a policy is rolled out, small non-optimal deviations may push it into situations not represented in the training data, triggering compounding errors over long horizons. In this work, we focus on learning-based warm-starting for gradient-based TO solvers that also provide feedback gains. Exploiting this specificity, we derive a first-order loss for Sobolev learning of diffusion-based policies using both trajectories and feedback gains. Through comprehensive experiments, we demonstrate that the resulting policy avoids compounding errors, and so can learn from very few trajectories to provide initial guesses reducing solving time by $2\times$ to $20 \times$. Incorporating first-order information enables predictions with fewer diffusion steps, reducing inference latency.
>
---
#### [new 037] SynAgent: Generalizable Cooperative Humanoid Manipulation via Solo-to-Cooperative Agent Synergy
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于多智能体协作操作任务，解决数据稀缺与跨物体泛化问题。提出SynAgent框架，通过单到多智能体协同迁移技能，实现可控的协作操作。**

- **链接: [https://arxiv.org/pdf/2604.18557](https://arxiv.org/pdf/2604.18557)**

> **作者:** Wei Yao; Haohan Ma; Hongwen Zhang; Yunlian Sun; Liangjun Xing; Zhile Yang; Yuanjun Guo; Yebin Liu; Jinhui Tang
>
> **摘要:** Controllable cooperative humanoid manipulation is a fundamental yet challenging problem for embodied intelligence, due to severe data scarcity, complexities in multi-agent coordination, and limited generalization across objects. In this paper, we present SynAgent, a unified framework that enables scalable and physically plausible cooperative manipulation by leveraging Solo-to-Cooperative Agent Synergy to transfer skills from single-agent human-object interaction to multi-agent human-object-human scenarios. To maintain semantic integrity during motion transfer, we introduce an interaction-preserving retargeting method based on an Interact Mesh constructed via Delaunay tetrahedralization, which faithfully maintains spatial relationships among humans and objects. Building upon this refined data, we propose a single-agent pretraining and adaptation paradigm that bootstraps synergistic collaborative behaviors from abundant single-human data through decentralized training and multi-agent PPO. Finally, we develop a trajectory-conditioned generative policy using a conditional VAE, trained via multi-teacher distillation from motion imitation priors to achieve stable and controllable object-level trajectory execution. Extensive experiments demonstrate that SynAgent significantly outperforms existing baselines in both cooperative imitation and trajectory-conditioned control, while generalizing across diverse object geometries. Codes and data will be available after publication. Project Page: this http URL
>
---
## 更新

#### [replaced 001] ARM: Advantage Reward Modeling for Long-Horizon Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ARM框架，解决长时域机械臂操作中的稀疏奖励问题，通过相对优势建模和三态标注策略提升数据效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.03037](https://arxiv.org/pdf/2604.03037)**

> **作者:** Yiming Mao; Zixi Yu; Weixin Mao; Yinhao Li; Qirui Hu; Zihan Lan; Minzhao Zhu; Hua Chen
>
> **摘要:** Long-horizon robotic manipulation remains challenging for reinforcement learning (RL) because sparse rewards provide limited guidance for credit assignment. Practical policy improvement thus relies on richer intermediate supervision, such as dense progress rewards, which are costly to obtain and ill-suited to non-monotonic behaviors such as backtracking and recovery. To address this, we propose Advantage Reward Modeling (ARM), a framework that shifts from hard-to-quantify absolute progress to estimating relative advantage. We introduce a cost-effective tri-state labeling strategy -- Progressive, Regressive, and Stagnant -- that reduces human cognitive overhead while ensuring high cross-annotator consistency. By training on these intuitive signals, ARM enables automated progress annotation for both complete demonstrations and fragmented DAgger-style data. Integrating ARM into an offline RL pipeline allows for adaptive action-reward reweighting, effectively filtering suboptimal samples. Our approach achieves a 99.4% success rate on a challenging long-horizon towel-folding task, demonstrating improved stability and data efficiency over current VLA baselines with near-zero human intervention during policy training.
>
---
#### [replaced 002] Generative Models and Connected and Automated Vehicles: A Survey in Exploring the Intersection of Transportation and AI
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于综述任务，探讨生成模型与自动驾驶车辆的结合，解决如何提升自动驾驶预测、模拟和决策的问题，分析其优势与挑战。**

- **链接: [https://arxiv.org/pdf/2403.10559](https://arxiv.org/pdf/2403.10559)**

> **作者:** Bo Shu; Yiting Zhang; Saisai Hu; Dong Shu
>
> **摘要:** This report investigates the history and impact of Generative Models and Connected and Automated Vehicles (CAVs), two groundbreaking forces pushing progress in technology and transportation. By focusing on the application of generative models within the context of CAVs, the study aims to unravel how this integration could enhance predictive modeling, simulation accuracy, and decision-making processes in autonomous vehicles. This thesis discusses the benefits and challenges of integrating generative models and CAV technology in transportation. It aims to highlight the progress made, the remaining obstacles, and the potential for advancements in safety and innovation.
>
---
#### [replaced 003] Hybrid Task and Motion Planning with Reactive Collision Handling for Multi-Robot Disassembly of Complex Products: Application to EV Batteries
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同任务，解决复杂产品拆解中的路径规划与避障问题。提出一种融合预测与反应式避障的框架，提升拆解效率与安全性。**

- **链接: [https://arxiv.org/pdf/2509.21020](https://arxiv.org/pdf/2509.21020)**

> **作者:** Abdelaziz Shaarawy; Cansu Erdogan; Rustam Stolkin; Alireza Rastegarpanah
>
> **摘要:** This paper addresses the problem of multi-robot coordination for complex manipulation task sequences. We present a vision-driven task-and-motion planning (TAMP) framework for a real dual-agent platform that integrates task decomposition and allocation with a learning-based RRT planner. A GMM-informed motion planner is coupled with a hybrid safety layer that combines predictive collision checking in a MoveIt/FCL digital twin with reactive vision-based avoidance and replanning. This integration is challenging as the system jointly satisfies task precedence, geometric feasibility, dynamic obstacle avoidance, and dual-arm coordination constraints. The framework operates in closed loop by updating the remaining task sequence from repeated scene scans and completion-state tracking rather than executing a fixed open-loop plan. In EV battery disassembly experiments, compared with Default-RRTConnect under identical perception and task assignments, the proposed system reduces cumulative end-effector path length from 48.8 to 17.9~m ($-63.3\%$), improves makespan from 467.9 to 429.8~s ($-8.1\%$), and reduces swept volumes (R1: $0.583\rightarrow0.139\,\mathrm{m}^3$, R2: $0.696\rightarrow0.252\,\mathrm{m}^3$) and overlap ($0.064\rightarrow0.034\,\mathrm{m}^3$). These results show that combining predictive planning and reactive collision avoidance in a real dual-arm disassembly scenario improves motion compactness, safety, and scalability to broader multi-robot sequential manipulation tasks.
>
---
#### [replaced 004] Memory Over Maps: 3D Object Localization Without Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人目标定位任务，解决传统3D重建耗时费力的问题，提出无需构建3D场景的视觉记忆方法，通过2D图像直接定位目标。**

- **链接: [https://arxiv.org/pdf/2603.20530](https://arxiv.org/pdf/2603.20530)**

> **作者:** Rui Zhou; Xander Yap; Jianwen Cao; Allison Lau; Boyang Sun; Marc Pollefeys
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Target localization is a prerequisite for embodied tasks such as navigation and manipulation. Conventional approaches rely on constructing explicit 3D scene representations to enable target localization, such as point clouds, voxel grids, or scene graphs. While effective, these pipelines incur substantial mapping time, storage overhead, and scalability limitations. Recent advances in vision-language models suggest that rich semantic reasoning can be performed directly on 2D observations, raising a fundamental question: is a complete 3D scene reconstruction necessary for object localization? In this work, we revisit object localization and propose a map-free pipeline that stores only posed RGB-D keyframes as a lightweight visual memory--without constructing any global 3D representation of the scene. At query time, our method retrieves candidate views, re-ranks them with a vision-language model, and constructs a sparse, on-demand 3D estimate of the queried target through depth backprojection and multi-view fusion. Compared to reconstruction-based pipelines, this design drastically reduces preprocessing cost, enabling scene indexing that is over two orders of magnitude faster to build while using substantially less storage. We further validate the localized targets on downstream object-goal navigation tasks. Despite requiring no task-specific training, our approach achieves strong performance across multiple benchmarks, demonstrating that direct reasoning over image-based scene memory can effectively replace dense 3D reconstruction for object-centric robot navigation. Project page: this https URL
>
---
#### [replaced 005] On the Derivation of Tightly-Coupled LiDAR-Inertial Odometry with VoxelMap
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于定位与导航任务，解决LiDAR与惯性数据融合问题，通过VoxelMap和卡尔曼滤波实现紧耦合的里程计系统。**

- **链接: [https://arxiv.org/pdf/2603.15471](https://arxiv.org/pdf/2603.15471)**

> **作者:** Zhihao Zhan
>
> **摘要:** This note presents a concise mathematical formulation of tightly-coupled LiDAR-Inertial Odometry within an iterated error-state Kalman filter framework using a VoxelMap representation. Rather than proposing a new algorithm, it provides a clear and self-contained derivation that unifies the geometric modeling and probabilistic state estimation through consistent notation and explicit formulations. The document is intended to serve both as a technical reference and as an accessible entry point for a foundational understanding of the system architecture and estimation principles.
>
---
#### [replaced 006] Personalized Embodied Navigation for Portable Object Finding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于智能导航任务，解决动态环境中非固定目标的寻找问题。通过引入TAP方法，提升代理在真实和模拟环境中的导航性能。**

- **链接: [https://arxiv.org/pdf/2403.09905](https://arxiv.org/pdf/2403.09905)**

> **作者:** Vishnu Sashank Dorbala; Bhrij Patel; Amrit Singh Bedi; Dinesh Manocha
>
> **备注:** 10 pages
>
> **摘要:** Embodied navigation methods commonly operate in static environments with stationary objects. In this work, we present approaches for tackling navigation in dynamic scenarios with non-stationary targets. In an indoor environment, we assume that these objects are everyday portable items moved by human intervention. We therefore formalize the problem as a personalized habit learning problem. To learn these habits, we introduce two Transit-Aware Planning (TAP) approaches that enrich embodied navigation policies with object path information. TAP improves performance in portable object finding by rewarding agents that learn to synchronize their routes with target routes. TAPs are evaluated on Dynamic Object Maps (DOMs), a dynamic variant of node-attributed topological graphs with structured object transitions. DOMs mimic human habits to simulate realistic object routes on a graph. We test TAP agents both in simulation as well as the real-world. In the MP3D simulator, TAP improves the success of a vanilla agent by 21.1% in finding non-stationary targets, while also generalizing better from static environments by 44.5% when measured by Relative Change in Success. In the real-world, we note a similar 18.3% increase on average, in multiple transit scenarios. We present qualitative inferences of TAP-agents deployed in the real world, showing them to be especially better at providing personalized assistance by finding targets in positions that they are usually not expected to be in (a toothbrush in a workspace). We also provide details of our real-to-sim pipeline, which allows researchers to generate simulations of their own physical environments for TAP, aiming to foster research in this area.
>
---
#### [replaced 007] ExpertGen: Scalable Sim-to-Real Expert Policy Learning from Imperfect Behavior Priors
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ExpertGen，解决机器人行为克隆中数据获取成本高的问题。通过模拟学习专家策略，实现高效、可靠的现实部署。**

- **链接: [https://arxiv.org/pdf/2603.15956](https://arxiv.org/pdf/2603.15956)**

> **作者:** Zifan Xu; Ran Gong; Maria Vittoria Minniti; Ahmet Salih Gundogdu; Eric Rosen; Kausik Sivakumar; Riedana Yan; Zixing Wang; Di Deng; Peter Stone; Xiaohan Zhang; Karl Schmeckpeper
>
> **摘要:** Learning generalizable and robust behavior cloning policies requires large volumes of high-quality robotics data. While human demonstrations (e.g., through teleoperation) serve as the standard source for expert behaviors, acquiring such data at scale in the real world is prohibitively expensive. This paper introduces ExpertGen, a framework that automates expert policy learning in simulation to enable scalable sim-to-real transfer. ExpertGen first initializes a behavior prior using a diffusion policy trained on imperfect demonstrations, which may be synthesized by large language models or provided by humans. Reinforcement learning is then used to steer this prior toward high task success by optimizing the diffusion model's initial noise while keep original policy frozen. By keeping the pretrained diffusion policy frozen, ExpertGen regularizes exploration to remain within safe, human-like behavior manifolds, while also enabling effective learning with only sparse rewards. Empirical evaluations on challenging manipulation benchmarks demonstrate that ExpertGen reliably produces high-quality expert policies with no reward engineering. On industrial assembly tasks, ExpertGen achieves a 90.5% overall success rate, while on long-horizon manipulation tasks it attains 85% overall success, outperforming all baseline methods. The resulting policies exhibit dexterous control and remain robust across diverse initial configurations and failure states. To validate sim-to-real transfer, the learned state-based expert policies are further distilled into visuomotor policies via DAgger and successfully deployed on real robotic hardware.
>
---
#### [replaced 008] Joint Magnetometer-IMU Calibration via Maximum A Posteriori Estimation
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于传感器联合标定任务，旨在提高磁强计与IMU的标定精度与效率。通过最大后验估计方法，同时优化标定参数和姿态轨迹，实现更准确且快速的标定。**

- **链接: [https://arxiv.org/pdf/2505.16662](https://arxiv.org/pdf/2505.16662)**

> **作者:** Chuan Huang; Gustaf Hendeby; Isaac Skog
>
> **备注:** Accepted version
>
> **摘要:** This paper presents a new method for jointly calibrating a magnetometer and inertial measurement unit (IMU), focusing on balancing calibration accuracy and computational efficiency. The proposed method is based on a maximum a posteriori estimation framework, treating both the calibration parameters and orientation trajectory of the sensors as unknowns. This method enables efficient optimization of the calibration parameters using analytically derived derivatives. The performance of the proposed method is compared against that of two state-of-the-art methods. Simulation results demonstrate that the proposed method achieves the lowest root mean square error in calibration parameters, increasing the calibration accuracy by 20-30%, while maintaining competitive computational efficiency. Further validation through real-world experiments confirms the practical benefits of the proposed method. The proposed method calibrated 30 magnetometer-IMU pairs in under two minutes on a consumer-grade laptop, which is one order of magnitude faster than the most accurate state-of-the-art algorithm as implemented in this work. Moreover, when calibrated using the proposed method, a magnetic-field-aided inertial navigation system achieved positioning performance comparable to when it is calibrated with the state-of-the-art method. These results demonstrate that the proposed method is a reliable and effective choice for jointly calibrating magnetometer-IMU pairs.
>
---
#### [replaced 009] Latent Linear Quadratic Regulator for Robotic Control Tasks
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决MPC计算成本高的问题。通过将状态空间映射到潜在空间，构建线性动态模型和二次代价函数，实现高效LQR控制。**

- **链接: [https://arxiv.org/pdf/2407.11107](https://arxiv.org/pdf/2407.11107)**

> **作者:** Yuan Zhang; Shaohui Yang; Toshiyuki Ohtsuka; Colin Jones; Joschka Boedecker
>
> **备注:** Accepted at L4DC 2026
>
> **摘要:** Model predictive control (MPC) has played a more crucial role in various robotic control tasks, but its high computational requirements are concerning, especially for nonlinear dynamical models. This paper presents a $\textbf{la}$tent $\textbf{l}$inear $\textbf{q}$uadratic $\textbf{r}$egulator (LaLQR) that maps the state space into a latent space, on which the dynamical model is linear and the cost function is quadratic, allowing the efficient application of LQR. We jointly learn this alternative system by imitating the original MPC. Experiments show LaLQR's superior efficiency and generalization compared to other baselines.
>
---
#### [replaced 010] Early Pruning for Public Transport Routing
- **分类: cs.DS; cs.AI; cs.RO**

- **简介: 该论文属于公共交通路径规划任务，解决转移松弛阶段效率低的问题。通过早期剪枝技术提升算法效率，不牺牲最优性。**

- **链接: [https://arxiv.org/pdf/2603.12592](https://arxiv.org/pdf/2603.12592)**

> **作者:** Andrii Rohovyi; Abdallah Abuaisha; Toby Walsh
>
> **摘要:** Routing algorithms for public transport, particularly the widely used RAPTOR and its variants, often face performance bottlenecks during the transfer relaxation phase, especially on dense transfer graphs, when supporting unlimited transfers. This inefficiency arises from iterating over many potential inter-stop connections (walks, bikes, e-scooters, etc.). To maintain acceptable performance, practitioners often limit transfer distances or exclude certain transfer options, which can reduce path optimality and restrict the multimodal options presented to travellers. This paper introduces Early Pruning, a low-overhead technique that accelerates routing algorithms without compromising optimality. By pre-sorting transfer connections by duration and applying a pruning rule within the transfer loop, the method discards longer transfers at a stop once they cannot yield an earlier arrival than the current best solution. Early Pruning can be integrated with minimal changes to existing codebases and requires only a one-time preprocessing step. The technique preserves Pareto-optimality in extended-criteria settings whenever the additional optimization criteria are monotonically non-decreasing in transfer duration. Across multiple state-of-the-art RAPTOR-based solutions, including RAPTOR, ULTRA-RAPTOR, McRAPTOR, BM-RAPTOR, ULTRA-McRAPTOR, and UBM-RAPTOR and tested on the Switzerland and London transit networks, we achieved query time reductions of up to 57\%. This approach provides a generalizable improvement to the efficiency of transit pathfinding algorithms.
>
---
#### [replaced 011] Phase-Aware Policy Learning for Skateboard Riding of Quadruped Robots via Feature-wise Linear Modulation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决四足机器人滑板骑行中的策略学习问题。通过引入相位感知的强化学习框架，提升滑板控制的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2602.09370](https://arxiv.org/pdf/2602.09370)**

> **作者:** Minsung Yoon; Jeil Jeong; Sung-Eui Yoon
>
> **备注:** ICRA 2026 | Project Page: this https URL | M. Yoon and J. Jeong contributed equally
>
> **摘要:** Skateboards offer a compact and efficient means of transportation as a type of personal mobility device. However, controlling them with legged robots poses several challenges for policy learning due to perception-driven interactions and multi-modal control objectives across distinct skateboarding phases. To address these challenges, we introduce Phase-Aware Policy Learning (PAPL), a reinforcement-learning framework tailored for skateboarding with quadruped robots. PAPL leverages the cyclic nature of skateboarding by integrating phase-conditioned Feature-wise Linear Modulation layers into actor and critic networks, enabling a unified policy that captures phase-dependent behaviors while sharing robot-specific knowledge across phases. Our evaluations in simulation validate command-tracking accuracy and conduct ablation studies quantifying each component's contribution. We also compare locomotion efficiency against leg and wheel-leg baselines and show real-world transferability.
>
---
#### [replaced 012] ASVSim (AirSim for Surface Vehicles): A High-Fidelity Simulation Framework for Autonomous Surface Vehicle Research
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出ASVSim，一个用于自主水面船舶研究的高保真仿真框架，解决缺乏开源仿真工具的问题，支持船舶动力学与传感器模拟，适用于导航算法和深度学习研究。**

- **链接: [https://arxiv.org/pdf/2506.22174](https://arxiv.org/pdf/2506.22174)**

> **作者:** Bavo Lesy; Siemen Herremans; Robin Kerstens; Jan Steckel; Walter Daems; Siegfried Mercelis; Ali Anwar
>
> **备注:** 18 Pages, 13 Figures. Accepted at IEEE ACCESS
>
> **摘要:** The transport industry has recently shown significant interest in unmanned surface vehicles (USVs), specifically for port and inland waterway transport. These systems can improve operational efficiency and safety, which is especially relevant in the European Union, where initiatives such as the Green Deal are driving a shift towards increased use of inland waterways. At the same time, a shortage of qualified personnel is accelerating the adoption of autonomous solutions. However, there is a notable lack of open-source, high-fidelity simulation frameworks and datasets for developing and evaluating such solutions. To address these challenges, we introduce AirSim for Surface Vehicles (ASVSim), an open-source simulation framework specifically designed for autonomous shipping research in inland and port environments. The framework combines simulated vessel dynamics with marine sensor simulation capabilities, including radar and camera systems and supports the generation of synthetic datasets for training computer vision models and reinforcement learning (RL) agents. Built upon Cosys-AirSim, ASVSim provides a comprehensive platform for developing autonomous navigation algorithms and generating synthetic datasets. The simulator supports research of both traditional control methods and deep learning-based approaches. Through experiments in waterway segmentation and autonomous navigation, we demonstrate the capabilities of the simulator in these research areas. ASVSim is provided as an open-source project under the MIT license, making autonomous navigation research accessible to a larger part of the ocean engineering community. See this https URL.
>
---
#### [replaced 013] MRS: Multi-Resolution Skills for HRL Agents
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，解决HRL中子目标选择不精准的问题。提出多分辨率技能（MRS），通过不同时间尺度的模块和元控制器提升性能。**

- **链接: [https://arxiv.org/pdf/2505.21410](https://arxiv.org/pdf/2505.21410)**

> **作者:** Shashank Sharma; Janina Hoffmann; Vinay Namboodiri
>
> **摘要:** Hierarchical reinforcement learning (HRL) decomposes the policy into a manager and a worker, enabling long-horizon planning but introducing a performance gap on tasks requiring agility. We identify a root cause: in subgoal-based HRL, the manager's goal representation is typically learned without constraints on reachability or temporal distance from the current state, preventing precise local subgoal selection. We further show that the optimal subgoal distance is both task- and state-dependent: nearby subgoals enable precise control but amplify prediction noise, while distant subgoals produce smoother motion at the cost of geometric precision. We propose Multi-Resolution Skills (MRS), which learns multiple goal-prediction modules each specialized to a fixed temporal horizon, with a jointly trained meta-controller that selects among them based on the current state. MRS consistently outperforms fixed-resolution baselines and significantly reduces the performance gap between HRL and non-HRL state-of-the-art on DeepMind Control Suite, Gym-Robotics, and long-horizon AntMaze tasks. [Project page: this https URL]
>
---
#### [replaced 014] Adapting Dijkstra for Buffers and Unlimited Transfers
- **分类: cs.DS; cs.AI; cs.RO**

- **简介: 该论文属于公共交通路径规划任务，解决无限换乘下的最优路径问题。通过改进Dijkstra算法，提出TAD方法，在考虑缓冲时间的情况下提升效率。**

- **链接: [https://arxiv.org/pdf/2603.11729](https://arxiv.org/pdf/2603.11729)**

> **作者:** Denys Katkalo; Andrii Rohovyi; Toby Walsh
>
> **摘要:** In recent years, RAPTOR based algorithms have been considered the state-of-the-art for path-finding with unlimited transfers without preprocessing. However, this status largely stems from the evolution of routing research, where Dijkstra-based solutions were superseded by timetable-based algorithms without a systematic comparison. In this work, we revisit classical Dijkstra-based approaches for public transit routing with unlimited transfers and demonstrate that Time-Dependent Dijkstra (TD-Dijkstra) outperforms MR. However, efficient TD-Dijkstra implementations rely on filtering dominated connections during preprocessing, which assumes passengers can always switch to a faster connection. We show that this filtering is unsound when stops have buffer times, as it cannot distinguish between seated passengers who may continue without waiting and transferring passengers who must respect the buffer. To address this limitation, we introduce Transfer Aware Dijkstra (TAD), a modification that scans entire trip sequences rather than individual edges, correctly handling buffer times while maintaining performance advantages over MR. Our experiments on London and Switzerland networks show that we can achieve a greater than two time speed-up over MR while producing optimal results on both networks with and without buffer times.
>
---
#### [replaced 015] Developing a Robotic Surgery Training System for Wide Accessibility and Research
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决机器人手术培训成本高、可及性差的问题。通过开发低成本训练系统，实现远程操作与精准控制，提升培训效果。**

- **链接: [https://arxiv.org/pdf/2505.20562](https://arxiv.org/pdf/2505.20562)**

> **作者:** Walid Shaker; Mustafa Suphi Erden
>
> **备注:** 6 pages, 2025 International Conference on Advanced Robotics and Mechatronics (ICARM), published
>
> **摘要:** Robotic surgery represents a major breakthrough in medical interventions, which has revolutionized surgical procedures. However, the high cost and limited accessibility of robotic surgery systems pose significant challenges for training purposes. This study addresses these issues by developing a cost-effective robotic laparoscopy training system that closely replicates advanced robotic surgery setups to ensure broad access for both on-site and remote users. Key innovations include the design of a low-cost robotic end-effector that effectively mimics high-end laparoscopic instruments. Additionally, a digital twin platform was established, facilitating detailed simulation, testing, and real-time monitoring, which enhances both system development and deployment. Furthermore, teleoperation control was optimized, leading to improved trajectory tracking while maintaining remote center of motion (RCM) constraint, with a RMSE of 5 {\mu}m and reduced system latency to 0.01 seconds. As a result, the system provides smooth, continuous motion and incorporates essential safety features, making it a highly effective tool for laparoscopic training.
>
---
#### [replaced 016] Drift-Based Policy Optimization: Native One-Step Policy Learning for Online Robot Control
- **分类: cs.RO**

- **简介: 该论文提出一种基于漂移的策略优化方法，用于在线机器人控制。解决多步生成策略推理耗时的问题，通过两阶段框架实现单步策略生成，提升推理速度与控制频率。**

- **链接: [https://arxiv.org/pdf/2604.03540](https://arxiv.org/pdf/2604.03540)**

> **作者:** Yuxuan Gao; Yedong Shen; Shiqi Zhang; Wenhao Yu; Yifan Duan; Jia pan; Jiajia Wu; Jiajun Deng; Yanyong Zhang
>
> **摘要:** Although multi-step generative policies achieve strong performance in robotic manipulation by modeling multimodal action distributions, they require multi-step iterative denoising at inference time. Each action therefore needs tens to hundreds of network function evaluations (NFEs), making them costly for high-frequency closed-loop control and online reinforcement learning (RL). To address this limitation, we propose a two-stage framework for native one-step generative policies that shifts refinement from inference to training. First, we introduce the Drift-Based Policy (DBP), which leverages fixed-point drifting objectives to internalize iterative refinement into the model parameters, yielding a one-step generative backbone by design while preserving multimodal action modeling capacity. Second, we develop Drift-Based Policy Optimization (DBPO), an online RL framework that equips the pretrained backbone with a compatible stochastic interface, enabling stable on-policy updates without sacrificing the one-step deployment property. Extensive experiments demonstrate the effectiveness of the proposed framework across offline imitation learning, online fine-tuning, and real-world control scenarios. DBP matches or exceeds the performance of multi-step diffusion policies while achieving up to $100\times$ faster inference. It also consistently outperforms existing one-step baselines on challenging manipulation benchmarks. Moreover, DBPO enables effective and stable policy improvement in online settings. Experiments on a real-world dual-arm robot demonstrate reliable high-frequency control at 105.2 Hz.
>
---
#### [replaced 017] Zero to Autonomy in Real-Time: Online Adaptation of Dynamics in Unstructured Environments
- **分类: cs.RO**

- **简介: 该论文属于自主机器人控制任务，解决动态环境中的实时模型适应问题。通过结合函数编码器与递归最小二乘法，实现快速模型更新，提升导航安全性与准确性。**

- **链接: [https://arxiv.org/pdf/2509.12516](https://arxiv.org/pdf/2509.12516)**

> **作者:** William Ward; Sarah Etter; Jesse Quattrociocchi; Christian Ellis; Adam J. Thorpe; Ufuk Topcu
>
> **备注:** Initial submission to RA-L
>
> **摘要:** Autonomous robots must go from zero prior knowledge to safe control within seconds to operate in unstructured environments. Abrupt terrain changes, such as a sudden transition to ice, create dynamics shifts that can destabilize planners unless the model adapts in real-time. We present a method for online adaptation that combines function encoders with recursive least squares, treating the function encoder coefficients as latent states updated from streaming odometry. This yields constant-time coefficient estimation without gradient-based inner-loop updates, enabling adaptation from only a few seconds of data. We evaluate our approach on a Van der Pol system to highlight algorithmic behavior, in a Unity simulator for high-fidelity off-road navigation, and on a Clearpath Jackal robot, including on a challenging terrain at a local ice rink. Across these settings, our method improves model accuracy and downstream planning, reducing collisions compared to static and meta-learning baselines.
>
---
#### [replaced 018] Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching and Differentiable Optimization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于多机器人路径优化任务，解决集中式轨迹优化计算不可行的问题。提出Flow-Opt方法，通过生成模型和安全过滤器实现高效、平滑的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2510.09204](https://arxiv.org/pdf/2510.09204)**

> **作者:** Simon Idoko; Prajyot Jadhav; Arun Kumar Singh
>
> **摘要:** Centralized trajectory optimization in the joint space of multiple robots allows access to a larger feasible space that can result in smoother trajectories, especially while planning in tight spaces. Unfortunately, it is often computationally intractable beyond a very small swarm size. In this paper, we propose Flow-Opt, a learning-based approach towards improving the computational tractability of centralized multi-robot trajectory optimization. Specifically, we reduce the problem to first learning a generative model to sample different candidate trajectories and then using a learned Safety-Filter(SF) to ensure fast inference-time constraint satisfaction. We propose a flow-matching model with a diffusion transformer (DiT) augmented with permutation invariant robot position and map encoders as the generative model. We develop a custom solver for our SF and equip it with a neural network that predicts context-specific initialization. The initialization network is trained in a self-supervised manner, taking advantage of the differentiability of the SF solver. We advance the state-of-the-art in the following respects. First, we show that we can generate trajectories of tens of robots in cluttered environments in a few tens of milliseconds. This is several times faster than existing centralized optimization approaches. Moreover, our approach also generates smoother trajectories orders of magnitude faster than competing baselines based on diffusion models. Second, each component of our approach can be batched, allowing us to solve a few tens of problem instances in a fraction of a second. We believe this is a first such result; no existing approach provides such capabilities. Finally, our approach can generate a diverse set of trajectories between a given set of start and goal locations, which can capture different collision-avoidance behaviors.
>
---
#### [replaced 019] MacroNav: Multi-Task Context Representation Learning Enables Efficient Navigation in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决未知环境中高效空间理解问题。提出MacroNav框架，结合多任务学习和强化学习，实现高效导航。**

- **链接: [https://arxiv.org/pdf/2511.04320](https://arxiv.org/pdf/2511.04320)**

> **作者:** Kuankuan Sima; Longbin Tang; Zhenyu Yang; Haozhe Ma; Lin Zhao
>
> **备注:** Accepted by IEEE Robotics and Automation Letters
>
> **摘要:** Autonomous navigation in unknown environments requires multi-scale spatial understanding that captures geometric details, topological connectivity, and global structure to support high-level decision making under partial observability. Existing approaches struggle to efficiently capture such multi-scale spatial understanding while maintaining low computational cost for real-time navigation. We present MacroNav, a learning-based navigation framework featuring two key components: (1) a lightweight context encoder trained via multi-task self-supervised learning to capture multi-scale, navigation-centric spatial representations; and (2) a reinforcement learning policy that seamlessly integrates these representations with graph-based reasoning for efficient action selection. Extensive experiments demonstrate the context encoder's effective and robust environmental understanding. Real-world deployments further validate MacroNav's effectiveness, yielding significant gains over state-of-the-art navigation methods in both Success Rate (SR) and Success weighted by Path Length (SPL), with superior computational efficiency.
>
---
#### [replaced 020] An Experimental Characterization of Mechanical Layer Jamming Systems
- **分类: cs.RO**

- **简介: 该论文研究机械层锁紧系统，旨在解决软体机器人中刚度调节问题。通过实验分析设计参数对性能的影响，提升结构的刚度调控能力。**

- **链接: [https://arxiv.org/pdf/2511.07882](https://arxiv.org/pdf/2511.07882)**

> **作者:** Jessica Gumowski; Krishna Manaswi Digumarti; David Howard
>
> **备注:** 6 pages, 9 figures, RoboSoft 2026
>
> **摘要:** Organisms in nature, such as Cephalopods and Pachyderms, exploit stiffness modulation to achieve amazing dexterity in the control of their appendages. In this paper, we explore the phenomenon of layer jamming, which is a popular stiffness modulation mechanism that provides an equivalent capability for soft robots. More specifically, we focus on mechanical layer jamming, which we realise through two-layer multi material structure with tooth-like protrusions. We identify key design parameters for mechanical layer jamming systems, including the ability to modulate stiffness, and perform a variety of comprehensive tests placing the specimens under bending and torsional loads to understand the influence of our selected design parameters (mainly tooth geometry) on the performance of the jammed structures. We note the ability of these structures to produce a peak change in stiffness of 5 times in bending and 3.2 times in torsion. We also measure the force required to separate the two jammed layers, an often ignored parameter in the study of jamming-induced stiffness change. This study aims to shed light on the principled design of mechanical layer jammed systems and guide researchers in the selection of appropriate designs for their specific application domains.
>
---
#### [replaced 021] Implicit Neural Field-Based Process Planning for Multi-Axis Manufacturing: Direct Control over Collision Avoidance and Toolpath Geometry
- **分类: cs.RO**

- **简介: 该论文属于多轴制造中的工艺规划任务，解决碰撞避让和刀具路径控制问题。提出基于隐式神经场的框架，实现层生成与路径设计的联合优化。**

- **链接: [https://arxiv.org/pdf/2511.17578](https://arxiv.org/pdf/2511.17578)**

> **作者:** Neelotpal Dutta; Tianyu Zhang; Tao Liu; Yongxue Chen; Charlie C.L. Wang
>
> **摘要:** Existing curved-layer-based process planning methods for multi-axis manufacturing address collisions only indirectly and generate toolpaths in a post-processing step, leaving toolpath geometry uncontrolled during optimization. We present an implicit neural field-based framework for multi-axis process planning that overcomes these limitations by embedding both layer generation and toolpath design within a single differentiable pipeline. Using sinusoidally activated neural networks to represent layers and toolpaths as implicit fields, our method enables direct evaluation of field values and derivatives at any spatial point, thereby allowing explicit collision avoidance and joint optimization of manufacturing layers and toolpaths. We further investigate how network hyperparameters and objective definitions influence singularity behavior and topology transitions, offering built-in mechanisms for regularization and stability control. The proposed approach is demonstrated on examples in both additive and subtractive manufacturing, validating its generality and effectiveness.
>
---
#### [replaced 022] NemeSys: Toward Online Underwater Exploration with Remote Operator-in-the-loop Adaptive Autonomy
- **分类: cs.RO**

- **简介: 该论文属于水下自主航行器任务，解决GPS和通信受限环境下的实时任务调整问题。提出NemeSys系统，实现低带宽下的在线任务重构与自适应控制。**

- **链接: [https://arxiv.org/pdf/2507.11889](https://arxiv.org/pdf/2507.11889)**

> **作者:** Adnan Abdullah; Alankrit Gupta; Vaishnav Ramesh; Shivali Patel; Md Jahidul Islam
>
> **备注:** 10 pages, V2
>
> **摘要:** Adaptive mission control and dynamic parameter reconfiguration are essential for autonomous underwater vehicles (AUVs) operating in GPS-denied, communication-limited marine environments. However, AUV platforms generally execute static, pre-programmed missions or rely on tethered connections and high-latency acoustic channels for mid-mission updates, significantly limiting their adaptability and responsiveness. In this paper, we introduce NemeSys, a novel AUV system designed to support real-time mission reconfiguration through compact magnetoelectric (ME) signaling. We present the full system design, control architecture, and a mission encoding framework that enables interactive exploration and task adaptation via low-bandwidth communication. The proposed system is validated through analytical modeling, controlled simulation tests, and real-world trials. The mid-mission retasking scenarios, evaluated using the NemeSys digital twin, demonstrate behavior switching latency below 50 ms with only a 13.2 MB peak computational overhead, making the framework suitable for deployment on edge computing hardware. Laboratory tank tests and open-water field trials further confirm stable control and reliable mission execution in dynamic underwater environments. These results establish the feasibility of online mission reconfiguration and highlight NemeSys as a promising step toward responsive, goal-driven adaptive underwater autonomy.
>
---
#### [replaced 023] MAGICIAN: Efficient Long-Term Planning with Imagined Gaussians for Active Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于主动地图构建任务，旨在解决传统方法因贪心策略导致的探索效率低和重建不完整问题。提出MAGICIAN框架，利用想象高斯表示实现长期规划，提升覆盖效率。**

- **链接: [https://arxiv.org/pdf/2603.22650](https://arxiv.org/pdf/2603.22650)**

> **作者:** Shiyao Li; Antoine Guédon; Shizhe Chen; Vincent Lepetit
>
> **备注:** Accepted at CVPR 2026 (Oral). Project webpage: this https URL
>
> **摘要:** Active mapping aims to determine how an agent should move to efficiently reconstruct unknown environments. Most existing approaches rely on greedy next-best-view prediction, resulting in inefficient exploration and incomplete reconstruction. To address this, we introduce MAGICIAN, a novel long-term planning framework that maximizes accumulated surface coverage gain through Imagined Gaussians, a scene representation based on 3D Gaussian Splatting, derived from a pre-trained occupancy network with strong structural priors. This representation enables efficient coverage gain computation for any novel viewpoint via fast volumetric rendering, allowing its integration into a tree-search algorithm for long-horizon planning. We update Imagined Gaussians and refine the trajectory in a closed loop. Our method achieves state-of-the-art performance across indoor and outdoor benchmarks with varying action spaces, highlighting the advantage of long-term planning in active mapping.
>
---
#### [replaced 024] RMGS-SLAM: Real-time Multi-sensor Gaussian Splatting SLAM
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决实时、高精度的定位与建图问题。提出RMGS-SLAM框架，融合多传感器数据，实现高效、连续的3D高斯重建与全局一致性。**

- **链接: [https://arxiv.org/pdf/2604.12942](https://arxiv.org/pdf/2604.12942)**

> **作者:** Dongen Li; Yi Liu; Junqi Liu; Zewen Sun; Zefan Huang; Shuo Sun; Jiahui Liu; Chengran Yuan; Hongliang Guo; Francis E.H. Tay; Marcelo H. Ang Jr
>
> **备注:** The manuscript has been improved, with refined content and updated and corrected experimental results
>
> **摘要:** Achieving real-time Simultaneous Localization and Mapping (SLAM) based on 3D Gaussian splatting (3DGS) in large-scale real-world environments remains challenging, as existing methods still struggle to jointly achieve low-latency pose estimation, continuous 3D Gaussian reconstruction, and long-term global consistency. In this paper, we present a tightly coupled LiDAR-Inertial-Visual 3DGS-based SLAM framework for real-time pose estimation and photorealistic mapping in large-scale real-world scenes. The system executes state estimation and 3D Gaussian primitive initialization in parallel with global Gaussian optimization, enabling continuous dense mapping. To improve Gaussian initialization quality and accelerate optimization convergence, we introduce a cascaded strategy that combines feed-forward predictions with geometric priors derived from voxel-based principal component analysis. To enhance global consistency, we perform loop closure directly on the optimized global Gaussian map by estimating loop constraints through Gaussian-based Generalized Iterative Closest Point registration, followed by pose-graph optimization. We also collect challenging large-scale looped outdoor sequences with hardware-synchronized LiDAR-camera-IMU and ground-truth trajectories for realistic evaluation. Extensive experiments on both public datasets and our dataset demonstrate that the proposed method achieves a state of the art among real-time efficiency, localization accuracy, and rendering quality across diverse real-world scenes.
>
---
#### [replaced 025] PhysMem: Scaling Test-time Physical Memory for Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PhysMem，用于机器人操作中的物理记忆学习，解决VLM在缺乏直接经验时的物理推理问题。通过测试假设提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.20323](https://arxiv.org/pdf/2602.20323)**

> **作者:** Haoyang Li; Yang You; Hao Su; Leonidas Guibas
>
> **摘要:** Reliable object manipulation requires understanding physical properties that vary across objects and environments. Vision-language model (VLM) planners can reason about friction and stability in general terms; however, they often cannot predict how a specific ball will roll on a particular surface or which stone will provide a stable foundation without direct experience. We present PhysMem, a memory framework that enables VLM robot planners to learn physical principles from interaction at test time, without updating model parameters. The system records experiences, generates candidate hypotheses, and verifies them through targeted interaction before promoting validated knowledge to guide future decisions. A central design choice is verification before application: the system tests hypotheses against new observations rather than applying retrieved experience directly, reducing rigid reliance on prior experience when physical conditions change. We evaluate PhysMem on three real-world manipulation tasks and simulation benchmarks across four VLM backbones. On a controlled brick insertion task, principled abstraction achieves 76% success compared to 23% for direct experience retrieval, and real-world experiments show consistent improvement over 30-minute deployment sessions.
>
---
#### [replaced 026] TFusionOcc: T-Primitive Based Object-Centric Multi-Sensor Fusion Framework for 3D Occupancy Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D语义占用预测任务，解决传统方法在建模复杂结构和冗余计算的问题，提出TFusionOcc框架，结合T-primitives和多传感器融合提升精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.06400](https://arxiv.org/pdf/2602.06400)**

> **作者:** Zhenxing Ming; Yaoqi Huang; Julie Stephany Berrio; Mao Shan; Stewart Worrall
>
> **摘要:** The prediction of 3D semantic occupancy enables autonomous vehicles (AVs) to perceive the fine-grained geometric and semantic scene structure for safe navigation and decision-making. Existing methods mainly rely on either voxel-based representations, which incur redundant computation over empty regions, or on object-centric Gaussian primitives, which are limited in modeling complex, non-convex, and asymmetric structures. In this paper, we present TFusionOcc, a T-primitive-based object-centric multi-sensor fusion framework for 3D semantic occupancy prediction. Specifically, we introduce a family of Students t-distribution-based T-primitives, including the plain T-primitive, T-Superquadric, and deformable T-Superquadric with inverse warping, where the deformable T-Superquadric serves as the key geometry-enhancing primitive. We further develop a unified probabilistic formulation based on the Students t-distribution and the T-mixture model (TMM) to jointly model occupancy and semantics, and design a tightly coupled multi-stage fusion architecture to effectively integrate camera and LiDAR cues. Extensive experiments on nuScenes show state-of-the-art performance, while additional evaluations on nuScenes-C demonstrate strong robustness under most corruption scenarios. The code will be available at: this https URL
>
---
#### [replaced 027] Preparation and Motion Study of Magnetically Driven Micro Soft Robot Mimicking the Cownose Ray
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于微机器人研究任务，旨在解决微型机器人在狭窄水下环境中的驱动与控制问题。设计并制造了一种仿鳐鱼的磁驱动软体机器人，通过调整磁场参数实现多种运动模式。**

- **链接: [https://arxiv.org/pdf/2601.15349](https://arxiv.org/pdf/2601.15349)**

> **作者:** Jiaqing Chang; Song Gao; Chaowei Dong; zhaobang Li; Yang Liu
>
> **备注:** There have several mistakes on it
>
> **摘要:** In narrow, unstructured underwater environments such as environmental monitoring and minimally invasive medical procedures, micro soft robots exhibit unique advantages due to their flexible movement capabilities and small size. At the same time, applying bionic technology to the structural design of micro soft robots can significantly improve their swimming performance. However, limited by their miniaturization, these robots are difficult to power internally and usually adopt a wireless power supply method. This study designs and fabricates a magnetically responsive, cownose ray-inspired micro soft robot based on the swimming principle of the cownose ray. The robot is made of a certain proportion of NdFeB and PDMS. Then, a three-dimensional Helmholtz coil is used to generate an oscillating harmonic magnetic field to conduct swimming experiments on the robot, exploring the influence of magnetic field parameters on the robot's swimming performance. The experimental results show that the swimming speed is the fastest at B = 5 mT and f = 11 Hz, reaching 5.25 mm/s, which is about 0.5 body lengths per second. In addition, by adjusting the current direction and frequency of the coil, the robot can perform different swimming modes such as straight swimming, turning swimming, and directional swimming. By employing a stepwise adjustment method, the impact of response errors on the robot's trajectory can be effectively reduced. This study demonstrates a method for magnetically driven micro soft robots, laying a foundation for the application of wireless-driven robots in underwater narrow spaces.
>
---
