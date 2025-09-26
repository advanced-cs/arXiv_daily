# 机器人 cs.RO

- **最新发布 55 篇**

- **更新 31 篇**

## 最新发布

#### [new 001] Learning Terrain-Specialized Policies for Adaptive Locomotion in Challenging Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究四足机器人在复杂地形中的自适应运动控制任务，旨在解决盲态下（无地形信息）运动鲁棒性差的问题。提出了一种基于地形专用策略和课程学习的分层强化学习框架，在仿真中验证了其在多地形场景下的优越适应性和运动性能。**

- **链接: [http://arxiv.org/pdf/2509.20635v1](http://arxiv.org/pdf/2509.20635v1)**

> **作者:** Matheus P. Angarola; Francisco Affonso; Marcelo Becker
>
> **备注:** Accepted to the 22nd International Conference on Advanced Robotics (ICAR 2025). 7 pages
>
> **摘要:** Legged robots must exhibit robust and agile locomotion across diverse, unstructured terrains, a challenge exacerbated under blind locomotion settings where terrain information is unavailable. This work introduces a hierarchical reinforcement learning framework that leverages terrain-specialized policies and curriculum learning to enhance agility and tracking performance in complex environments. We validated our method on simulation, where our approach outperforms a generalist policy by up to 16% in success rate and achieves lower tracking errors as the velocity target increases, particularly on low-friction and discontinuous terrains, demonstrating superior adaptability and robustness across mixed-terrain scenarios.
>
---
#### [new 002] \LARGE GMP$^{3}$: Learning-Driven, Bellman-Guided Trajectory Planning for UAVs in Real-Time on SE(3)
- **分类: cs.RO**

- **简介: 该论文提出GMP³，一种面向无人机的多阶段全局路径规划框架，用于在复杂三维环境中实时生成动态可行轨迹。通过结合强化学习与改进的Bellman算子，并扩展至SE(3)空间，实现了位置与姿态联合优化，解决了无人机协同避障与轨迹平滑问题。**

- **链接: [http://arxiv.org/pdf/2509.21264v1](http://arxiv.org/pdf/2509.21264v1)**

> **作者:** Babak Salamat; Dominik Mattern; Sebastian-Sven Olzem; Gerhard Elsbacher; Christian Seidel; Andrea M. Tonello
>
> **摘要:** We propose $\text{GMP}^{3}$, a multiphase global path planning framework that generates dynamically feasible three-dimensional trajectories for unmanned aerial vehicles (UAVs) operating in cluttered environments. The framework extends traditional path planning from Euclidean position spaces to the Lie group $\mathrm{SE}(3)$, allowing joint learning of translational motion and rotational dynamics. A modified Bellman-based operator is introduced to support reinforcement learning (RL) policy updates while leveraging prior trajectory information for improved convergence. $\text{GMP}^{3}$ is designed as a distributed framework in which agents influence each other and share policy information along the trajectory: each agent refines its assigned segment and shares with its neighbors via a consensus-based scheme, enabling cooperative policy updates and convergence toward a path shaped globally even under kinematic constraints. We also propose DroneManager, a modular ground control software that interfaces the planner with real UAV platforms via the MAVLink protocol, supporting real-time deployment and feedback. Simulation studies and indoor flight experiments validate the effectiveness of the proposed method in constrained 3D environments, demonstrating reliable obstacle avoidance and smooth, feasible trajectories across both position and orientation. The open-source implementation is available at https://github.com/Domattee/DroneManager
>
---
#### [new 003] Latent Activation Editing: Inference-Time Refinement of Learned Policies for Safer Multirobot Navigation
- **分类: cs.RO**

- **简介: 该论文提出一种推理时的潜激活编辑（LAE）方法，用于改进多旋翼飞行器导航的安全性。针对预训练策略在复杂环境中易发生碰撞的问题，LAE通过监测并修改中间激活状态，在不改变模型结构和参数的前提下提升安全性，实验证明能显著减少碰撞。**

- **链接: [http://arxiv.org/pdf/2509.20623v1](http://arxiv.org/pdf/2509.20623v1)**

> **作者:** Satyajeet Das; Darren Chiu; Zhehui Huang; Lars Lindemann; Gaurav S. Sukhatme
>
> **摘要:** Reinforcement learning has enabled significant progress in complex domains such as coordinating and navigating multiple quadrotors. However, even well-trained policies remain vulnerable to collisions in obstacle-rich environments. Addressing these infrequent but critical safety failures through retraining or fine-tuning is costly and risks degrading previously learned skills. Inspired by activation steering in large language models and latent editing in computer vision, we introduce a framework for inference-time Latent Activation Editing (LAE) that refines the behavior of pre-trained policies without modifying their weights or architecture. The framework operates in two stages: (i) an online classifier monitors intermediate activations to detect states associated with undesired behaviors, and (ii) an activation editing module that selectively modifies flagged activations to shift the policy towards safer regimes. In this work, we focus on improving safety in multi-quadrotor navigation. We hypothesize that amplifying a policy's internal perception of risk can induce safer behaviors. We instantiate this idea through a latent collision world model trained to predict future pre-collision activations, thereby prompting earlier and more cautious avoidance responses. Extensive simulations and real-world Crazyflie experiments demonstrate that LAE achieves statistically significant reduction in collisions (nearly 90% fewer cumulative collisions compared to the unedited baseline) and substantially increases the fraction of collision-free trajectories, while preserving task completion. More broadly, our results establish LAE as a lightweight paradigm, feasible on resource-constrained hardware, for post-deployment refinement of learned robot policies.
>
---
#### [new 004] RetoVLA: Reusing Register Tokens for Spatial Reasoning in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出RetoVLA，一种通过复用Vision Transformer中被丢弃的Register Tokens来提升VLA模型空间推理能力的方法，在轻量化结构下实现机器人复杂操作任务性能提升。**

- **链接: [http://arxiv.org/pdf/2509.21243v1](http://arxiv.org/pdf/2509.21243v1)**

> **作者:** Jiyeon Koo; Taewan Cho; Hyunjoon Kang; Eunseom Pyo; Tae Gyun Oh; Taeryang Kim; Andrew Jaeyong Choi
>
> **摘要:** Recent Vision-Language-Action (VLA) models demonstrate remarkable generalization in robotics but are restricted by their substantial size and computational cost, limiting real-world deployment. However, conventional lightweighting methods often sacrifice critical capabilities, particularly spatial reasoning. This creates a trade-off between efficiency and performance. To address this challenge, our work reuses Register Tokens, which were introduced for artifact removal in Vision Transformers but subsequently discarded. We suppose that these tokens contain essential spatial information and propose RetoVLA, a novel architecture that reuses them directly by injecting them into the Action Expert. RetoVLA maintains a lightweight structure while leveraging this repurposed spatial context to enhance reasoning. We demonstrate RetoVLA's effectiveness through a series of comprehensive experiments. On our custom-built 7-DOF robot arm, the model achieves a 17.1%p absolute improvement in success rates for complex manipulation tasks. Our results confirm that reusing Register Tokens directly enhances spatial reasoning, demonstrating that what was previously discarded as an artifact is in fact a valuable, unexplored resource for robotic intelligence. A video demonstration is available at: https://youtu.be/2CseBR-snZg
>
---
#### [new 005] Flight Dynamics to Sensing Modalities: Exploiting Drone Ground Effect for Accurate Edge Detection
- **分类: cs.RO; cs.NI**

- **简介: 该论文提出AirTouch系统，利用无人机地面效应进行高效边缘检测。针对传统方法成本高、计算负担重的问题，通过分析姿态传感器数据实现低功耗（43 mW）、高精度（误差0.051m）的环境边界检测，验证了地面效应作为新型感知模态的优势。**

- **链接: [http://arxiv.org/pdf/2509.21085v1](http://arxiv.org/pdf/2509.21085v1)**

> **作者:** Chenyu Zhao; Jingao Xu; Ciyu Ruan; Haoyang Wang; Shengbo Wang; Jiaqi Li; Jirong Zha; Weijie Hong; Zheng Yang; Yunhao Liu; Xiao-Ping Zhang; Xinlei Chen
>
> **摘要:** Drone-based rapid and accurate environmental edge detection is highly advantageous for tasks such as disaster relief and autonomous navigation. Current methods, using radars or cameras, raise deployment costs and burden lightweight drones with high computational demands. In this paper, we propose AirTouch, a system that transforms the ground effect from a stability "foe" in traditional flight control views, into a "friend" for accurate and efficient edge detection. Our key insight is that analyzing drone basic attitude sensor readings and flight commands allows us to detect ground effect changes. Such changes typically indicate the drone flying over a boundary of two materials, making this information valuable for edge detection. We approach this insight through theoretical analysis, algorithm design, and implementation, fully leveraging the ground effect as a new sensing modality without compromising drone flight stability, thereby achieving accurate and efficient scene edge detection. We also compare this new sensing modality with vision-based methods to clarify its exclusive advantages in resource efficiency and detection capability. Extensive evaluations demonstrate that our system achieves a high detection accuracy with mean detection distance errors of 0.051m, outperforming the baseline method performance by 86%. With such detection performance, our system requires only 43 mW power consumption, contributing to this new sensing modality for low-cost and highly efficient edge detection.
>
---
#### [new 006] Multi-Robot Vision-Based Task and Motion Planning for EV Battery Disassembly and Sorting
- **分类: cs.RO**

- **简介: 该论文研究多机器人协同拆解电动汽车电池的任务与运动规划问题。针对复杂动态场景下的精度、安全和效率需求，提出四层TAMP框架，结合视觉感知与学习运动规划，实现更紧凑、安全的拆解路径。**

- **链接: [http://arxiv.org/pdf/2509.21020v1](http://arxiv.org/pdf/2509.21020v1)**

> **作者:** Abdelaziz Shaarawy; Cansu Erdogan; Rustam Stolkin; Alireza Rastegarpanah
>
> **摘要:** Electric-vehicle (EV) battery disassembly requires precise multi-robot coordination, short and reliable motions, and robust collision safety in cluttered, dynamic scenes. We propose a four-layer task-and-motion planning (TAMP) framework that couples symbolic task planning and cost- and accessibility-aware allocation with a TP-GMM-guided motion planner learned from demonstrations. Stereo vision with YOLOv8 provides real-time component localization, while OctoMap-based 3D mapping and FCL(Flexible Collision Library) checks in MoveIt unify predictive digital-twin collision checking with reactive, vision-based avoidance. Validated on two UR10e robots across cable, busbar, service plug, and three leaf-cell removals, the approach yields substantially more compact and safer motions than a default RRTConnect baseline under identical perception and task assignments: average end-effector path length drops by $-63.3\%$ and makespan by $-8.1\%$; per-arm swept volumes shrink (R1: $0.583\rightarrow0.139\,\mathrm{m}^3$; R2: $0.696\rightarrow0.252\,\mathrm{m}^3$), and mutual overlap decreases by $47\%$ ($0.064\rightarrow0.034\,\mathrm{m}^3$). These results highlight improved autonomy, precision, and safety for multi-robot EV battery disassembly in unstructured, dynamic environments.
>
---
#### [new 007] Incorporating Human-Inspired Ankle Characteristics in a Forced-Oscillation-Based Reduced-Order Model for Walking
- **分类: cs.RO**

- **简介: 该论文提出一种融合人类踝部特性的简化行走模型，旨在提升机器人行走的稳定性和拟人性。通过设计类人踝部动力学，模型在应对小扰动时无需足部控制即可稳定，改善了传统点足模型的步态表现。**

- **链接: [http://arxiv.org/pdf/2509.20689v1](http://arxiv.org/pdf/2509.20689v1)**

> **作者:** Chathura Semasinghe; Siavash Rezazadeh
>
> **摘要:** This paper extends the forced-oscillation-based reduced-order model of walking to a model with ankles and feet. A human-inspired paradigm was designed for the ankle dynamics, which results in improved gait characteristics compared to the point-foot model. In addition, it was shown that while the proposed model can stabilize against large errors in initial conditions through combination of foot placement and ankle strategies, the model is able to stabilize against small perturbations without relying on the foot placement control and solely through the designed proprioceptive ankle scheme. This novel property, which is also observed in humans, can help in better understanding of anthropomorphic walking and its stabilization mechanisms.
>
---
#### [new 008] ImaginationPolicy: Towards Generalizable, Precise and Reliable End-to-End Policy for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出ImaginationPolicy，针对机器人操作任务，旨在解决现有端到端策略泛化性、精度和可靠性不足的问题。工作包括提出新的动作表示方法CoMOK，支持多任务、多模态行为及可变形物体操作，并通过仿真与实验证明其有效性。**

- **链接: [http://arxiv.org/pdf/2509.20841v1](http://arxiv.org/pdf/2509.20841v1)**

> **作者:** Dekun Lu; Wei Gao; Kui Jia
>
> **备注:** First two authors contribute equally. Project page: https://sites.google.com/view/imaginationpolicy
>
> **摘要:** End-to-end robot manipulation policies offer significant potential for enabling embodied agents to understand and interact with the world. Unlike traditional modular pipelines, end-to-end learning mitigates key limitations such as information loss between modules and feature misalignment caused by isolated optimization targets. Despite these advantages, existing end-to-end neural networks for robotic manipulation--including those based on large VLM/VLA models--remain insufficiently performant for large-scale practical deployment. In this paper, we take a step towards an end-to-end manipulation policy that is generalizable, accurate and reliable. To achieve this goal, we propose a novel Chain of Moving Oriented Keypoints (CoMOK) formulation for robotic manipulation. Our formulation is used as the action representation of a neural policy, which can be trained in an end-to-end fashion. Such an action representation is general, as it extends the standard end-effector pose action representation and supports a diverse set of manipulation tasks in a unified manner. The oriented keypoint in our method enables natural generalization to objects with different shapes and sizes, while achieving sub-centimeter accuracy. Moreover, our formulation can easily handle multi-stage tasks, multi-modal robot behaviors, and deformable objects. Extensive simulated and hardware experiments demonstrate the effectiveness of our method.
>
---
#### [new 009] RAM-NAS: Resource-aware Multiobjective Neural Architecture Search Method for Robot Vision Tasks
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RAM-NAS，一种面向机器人视觉任务的资源感知多目标神经架构搜索方法。针对传统NAS在轻量化模型设计中对硬件资源考虑不足的问题，通过子网互蒸馏、DKD损失及延迟预测器优化模型精度与推理速度的平衡，实验证明其在边缘设备上具有更高的效率和可扩展性。**

- **链接: [http://arxiv.org/pdf/2509.20688v1](http://arxiv.org/pdf/2509.20688v1)**

> **作者:** Shouren Mao; Minghao Qin; Wei Dong; Huajian Liu; Yongzhuo Gao
>
> **备注:** Joint first authors: Shouren Mao and Minghao Qin. Published in IEEE/RSJ IROS 2024. This arXiv version adds a joint first-authorship note to correct an omission in the IEEE Xplore version. No technical changes. Please cite the IEEE version
>
> **摘要:** Neural architecture search (NAS) has shown great promise in automatically designing lightweight models. However, conventional approaches are insufficient in training the supernet and pay little attention to actual robot hardware resources. To meet such challenges, we propose RAM-NAS, a resource-aware multi-objective NAS method that focuses on improving the supernet pretrain and resource-awareness on robot hardware devices. We introduce the concept of subnets mutual distillation, which refers to mutually distilling all subnets sampled by the sandwich rule. Additionally, we utilize the Decoupled Knowledge Distillation (DKD) loss to enhance logits distillation performance. To expedite the search process with consideration for hardware resources, we used data from three types of robotic edge hardware to train Latency Surrogate predictors. These predictors facilitated the estimation of hardware inference latency during the search phase, enabling a unified multi-objective evolutionary search to balance model accuracy and latency trade-offs. Our discovered model family, RAM-NAS models, can achieve top-1 accuracy ranging from 76.7% to 81.4% on ImageNet. In addition, the resource-aware multi-objective NAS we employ significantly reduces the model's inference latency on edge hardware for robots. We conducted experiments on downstream tasks to verify the scalability of our methods. The inference time for detection and segmentation is reduced on all three hardware types compared to MobileNetv3-based methods. Our work fills the gap in NAS for robot hardware resource-aware.
>
---
#### [new 010] Revisiting Formal Methods for Autonomous Robots: A Structured Survey
- **分类: cs.RO**

- **简介: 该论文是一篇结构化综述，任务是总结形式化方法在自主机器人系统中的应用。旨在梳理相关研究趋势，分析其发展与变化，特别关注子符号AI的应用。工作包括文献调研、分类及新趋势识别，如形式综合和概率验证技术的兴起。**

- **链接: [http://arxiv.org/pdf/2509.20488v1](http://arxiv.org/pdf/2509.20488v1)**

> **作者:** Atef Azaiez; David A. Anisi; Marie Farrell; Matt Luckcuck
>
> **备注:** Appeal accepted: MOD-66548 This is an appeal request regarding our submission MOD-65174 - 6681725
>
> **摘要:** This paper presents the initial results from our structured literature review on applications of Formal Methods (FM) to Robotic Autonomous Systems (RAS). We describe our structured survey methodology; including database selection and associated search strings, search filters and collaborative review of identified papers. We categorise and enumerate the FM approaches and formalisms that have been used for specification and verification of RAS. We investigate FM in the context of sub-symbolic AI-enabled RAS and examine the evolution of how FM is used over time in this field. This work complements a pre-existing survey in this area and we examine how this research area has matured over time. Specifically, our survey demonstrates that some trends have persisted as observed in a previous survey. Additionally, it recognized new trends that were not considered previously including a noticeable increase in adopting Formal Synthesis approaches as well as Probabilistic Verification Techniques.
>
---
#### [new 011] RuN: Residual Policy for Natural Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文提出RuN，一种用于人形机器人自然运动的残差策略框架。针对现有方法需同时学习动作模仿、速度跟踪和稳定性的问题，RuN将任务分解为预训练运动生成器与轻量残差策略，实现0-2.5 m/s范围内稳定、自然的步态转换。**

- **链接: [http://arxiv.org/pdf/2509.20696v1](http://arxiv.org/pdf/2509.20696v1)**

> **作者:** Qingpeng Li; Chengrui Zhu; Yanming Wu; Xin Yuan; Zhen Zhang; Jian Yang; Yong Liu
>
> **摘要:** Enabling humanoid robots to achieve natural and dynamic locomotion across a wide range of speeds, including smooth transitions from walking to running, presents a significant challenge. Existing deep reinforcement learning methods typically require the policy to directly track a reference motion, forcing a single policy to simultaneously learn motion imitation, velocity tracking, and stability maintenance. To address this, we introduce RuN, a novel decoupled residual learning framework. RuN decomposes the control task by pairing a pre-trained Conditional Motion Generator, which provides a kinematically natural motion prior, with a reinforcement learning policy that learns a lightweight residual correction to handle dynamical interactions. Experiments in simulation and reality on the Unitree G1 humanoid robot demonstrate that RuN achieves stable, natural gaits and smooth walk-run transitions across a broad velocity range (0-2.5 m/s), outperforming state-of-the-art methods in both training efficiency and final performance.
>
---
#### [new 012] KeyWorld: Key Frame Reasoning Enables Effective and Efficient World Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出KeyWorld，一种通过关键帧推理提升机器人世界模型效率与物理合理性的方法。针对逐帧生成的冗余问题，KeyWorld利用DiT生成语义关键帧，并用轻量插值模型补全中间帧，在LIBERO基准上实现了5.68倍加速并提升了轨迹合理性。**

- **链接: [http://arxiv.org/pdf/2509.21027v1](http://arxiv.org/pdf/2509.21027v1)**

> **作者:** Sibo Li; Qianyue Hao; Yu Shang; Yong Li
>
> **摘要:** Robotic world models are a promising paradigm for forecasting future environment states, yet their inference speed and the physical plausibility of generated trajectories remain critical bottlenecks, limiting their real-world applications. This stems from the redundancy of the prevailing frame-to-frame generation approach, where the model conducts costly computation on similar frames, as well as neglecting the semantic importance of key transitions. To address this inefficiency, we propose KeyWorld, a framework that improves text-conditioned robotic world models by concentrating transformers computation on a few semantic key frames while employing a lightweight convolutional model to fill the intermediate frames. Specifically, KeyWorld first identifies significant transitions by iteratively simplifying the robot's motion trajectories, obtaining the ground truth key frames. Then, a DiT model is trained to reason and generate these physically meaningful key frames from textual task descriptions. Finally, a lightweight interpolator efficiently reconstructs the full video by inpainting all intermediate frames. Evaluations on the LIBERO benchmark demonstrate that KeyWorld achieves a 5.68$\times$ acceleration compared to the frame-to-frame generation baseline, and focusing on the motion-aware key frames further contributes to the physical validity of the generated videos, especially on complex tasks. Our approach highlights a practical path toward deploying world models in real-time robotic control and other domains requiring both efficient and effective world models. Code is released at https://anonymous.4open.science/r/Keyworld-E43D.
>
---
#### [new 013] Suction Leap-Hand: Suction Cups on a Multi-fingered Hand Enable Embodied Dexterity and In-Hand Teleoperation
- **分类: cs.RO**

- **简介: 该论文提出Suction Leap-Hand（SLeap Hand），一种集成吸盘的多指机械手，旨在解决传统仿生手在力控抓取和数据采集中的难题。通过单点吸附替代复杂摩擦力控制，简化了操作并提升了任务灵活性，如单手剪纸和写字。**

- **链接: [http://arxiv.org/pdf/2509.20646v1](http://arxiv.org/pdf/2509.20646v1)**

> **作者:** Sun Zhaole; Xiaofeng Mao; Jihong Zhu; Yuanlong Zhang; Robert B. Fisher
>
> **备注:** An IEEE conference paper currently under review
>
> **摘要:** Dexterous in-hand manipulation remains a foundational challenge in robotics, with progress often constrained by the prevailing paradigm of imitating the human hand. This anthropomorphic approach creates two critical barriers: 1) it limits robotic capabilities to tasks humans can already perform, and 2) it makes data collection for learning-based methods exceedingly difficult. Both challenges are caused by traditional force-closure which requires coordinating complex, multi-point contacts based on friction, normal force, and gravity to grasp an object. This makes teleoperated demonstrations unstable and amplifies the sim-to-real gap for reinforcement learning. In this work, we propose a paradigm shift: moving away from replicating human mechanics toward the design of novel robotic embodiments. We introduce the \textbf{S}uction \textbf{Leap}-Hand (SLeap Hand), a multi-fingered hand featuring integrated fingertip suction cups that realize a new form of suction-enabled dexterity. By replacing complex force-closure grasps with stable, single-point adhesion, our design fundamentally simplifies in-hand teleoperation and facilitates the collection of high-quality demonstration data. More importantly, this suction-based embodiment unlocks a new class of dexterous skills that are difficult or even impossible for the human hand, such as one-handed paper cutting and in-hand writing. Our work demonstrates that by moving beyond anthropomorphic constraints, novel embodiments can not only lower the barrier for collecting robust manipulation data but also enable the stable, single-handed completion of tasks that would typically require two human hands. Our webpage is https://sites.google.com/view/sleaphand.
>
---
#### [new 014] Next-Generation Aerial Robots -- Omniorientational Strategies: Dynamic Modeling, Control, and Comparative Analysis
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究新一代全向飞行机器人，旨在解决传统多旋翼欠驱动问题。通过引入可调桨轴角度的配置，提出动态模型与控制策略，设计滑模和改进PID控制器，并进行仿真对比分析，优化能耗与抗干扰能力。**

- **链接: [http://arxiv.org/pdf/2509.21210v1](http://arxiv.org/pdf/2509.21210v1)**

> **作者:** Ali Kafili Gavgani; Amin Talaeizadeh; Aria Alasty; Hossein Nejat Pishkenari; Esmaeil Najafi
>
> **摘要:** Conventional multi-rotors are under-actuated systems, hindering them from independently controlling attitude from position. In this study, we present several distinct configurations that incorporate additional control inputs for manipulating the angles of the propeller axes. This addresses the mentioned limitations, making the systems "omniorientational". We comprehensively derived detailed dynamic models for all introduced configurations and validated by a methodology using Simscape Multibody simulations. Two controllers are designed: a sliding mode controller for robust handling of disturbances and a novel PID-based controller with gravity compensation integrating linear and non-linear allocators, designed for computational efficiency. A custom control allocation strategy is implemented to manage the input-non-affine nature of these systems, seeking to maximize battery life by minimizing the "Power Consumption Factor" defined in this study. Moreover, the controllers effectively managed harsh disturbances and uncertainties. Simulations compare and analyze the proposed configurations and controllers, majorly considering their power consumption. Furthermore, we conduct a qualitative comparison to evaluate the impact of different types of uncertainties on the control system, highlighting areas for potential model or hardware improvements. The analysis in this study provides a roadmap for future researchers to design omniorientational drones based on their design objectives, offering practical insights into configuration selection and controller design. This research aligns with the project SAC-1, one of the objectives of Sharif AgRoLab.
>
---
#### [new 015] SEEC: Stable End-Effector Control with Model-Enhanced Residual Learning for Humanoid Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文针对人形机器人在行走与操作任务中的末端执行器稳定控制问题，提出SEEC框架。通过模型增强的残差学习和扰动生成器，结合强化学习，实现对下体扰动的鲁棒补偿，无需额外训练即可适应不同运动控制器，提升了实际任务中的稳定性和泛化性。**

- **链接: [http://arxiv.org/pdf/2509.21231v1](http://arxiv.org/pdf/2509.21231v1)**

> **作者:** Jaehwi Jang; Zhuoheng Wang; Ziyi Zhou; Feiyang Wu; Ye Zhao
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Arm end-effector stabilization is essential for humanoid loco-manipulation tasks, yet it remains challenging due to the high degrees of freedom and inherent dynamic instability of bipedal robot structures. Previous model-based controllers achieve precise end-effector control but rely on precise dynamics modeling and estimation, which often struggle to capture real-world factors (e.g., friction and backlash) and thus degrade in practice. On the other hand, learning-based methods can better mitigate these factors via exploration and domain randomization, and have shown potential in real-world use. However, they often overfit to training conditions, requiring retraining with the entire body, and still struggle to adapt to unseen scenarios. To address these challenges, we propose a novel stable end-effector control (SEEC) framework with model-enhanced residual learning that learns to achieve precise and robust end-effector compensation for lower-body induced disturbances through model-guided reinforcement learning (RL) with a perturbation generator. This design allows the upper-body policy to achieve accurate end-effector stabilization as well as adapt to unseen locomotion controllers with no additional training. We validate our framework in different simulators and transfer trained policies to the Booster T1 humanoid robot. Experiments demonstrate that our method consistently outperforms baselines and robustly handles diverse and demanding loco-manipulation tasks.
>
---
#### [new 016] Action-Informed Estimation and Planning: Clearing Clutter on Staircases via Quadrupedal Pedipulation
- **分类: cs.RO**

- **简介: 该论文研究机器人在楼梯等复杂环境中自主清除障碍的任务，解决单腿推动物体时因遮挡导致的感知失效问题。提出结合感知与动作的交互感知状态估计方法，通过腿部反馈预测物体位移，提升推物成功率和跟踪精度。**

- **链接: [http://arxiv.org/pdf/2509.20516v1](http://arxiv.org/pdf/2509.20516v1)**

> **作者:** Prasanna Sriganesh; Barath Satheeshkumar; Anushree Sabnis; Matthew Travers
>
> **摘要:** For robots to operate autonomously in densely cluttered environments, they must reason about and potentially physically interact with obstacles to clear a path. Safely clearing a path on challenging terrain, such as a cluttered staircase, requires controlled interaction. For example, a quadrupedal robot that pushes objects out of the way with one leg while maintaining a stable stance with its three other legs. However, tightly coupled physical actions, such as one-legged pushing, create new constraints on the system that can be difficult to predict at design time. In this work, we present a new method that addresses one such constraint, wherein the object being pushed by a quadrupedal robot with one of its legs becomes occluded from the robot's sensors during manipulation. To address this challenge, we present a tightly coupled perception-action framework that enables the robot to perceive clutter, reason about feasible push paths, and execute the clearing maneuver. Our core contribution is an interaction-aware state estimation loop that uses proprioceptive feedback regarding foot contact and leg position to predict an object's displacement during the occlusion. This prediction guides the perception system to robustly re-detect the object after the interaction, closing the loop between action and sensing to enable accurate tracking even after partial pushes. Using this feedback allows the robot to learn from physical outcomes, reclassifying an object as immovable if a push fails due to it being too heavy. We present results of implementing our approach on a Boston Dynamics Spot robot that show our interaction-aware approach achieves higher task success rates and tracking accuracy in pushing objects on stairs compared to open-loop baselines.
>
---
#### [new 017] BactoBot: A Low-Cost, Bacteria-Inspired Soft Underwater Robot for Marine Exploration
- **分类: cs.RO**

- **简介: 该论文提出BactoBot，一种低成本、仿细菌的软体水下机器人，旨在解决传统硬质设备对海洋生态的破坏问题。采用3D打印和硅胶臂设计，实现安全、灵活的水下探索，验证了低成本仿生运动的可行性。**

- **链接: [http://arxiv.org/pdf/2509.20964v1](http://arxiv.org/pdf/2509.20964v1)**

> **作者:** Rubaiyat Tasnim Chowdhury; Nayan Bala; Ronojoy Roy; Tarek Mahmud
>
> **备注:** 8 pages, 4 figures. Project repository available at https://github.com/rubaiyattasnim/BactoBot
>
> **摘要:** Traditional rigid underwater vehicles pose risks to delicate marine ecosystems. This paper presents BactoBot, a low-cost, soft underwater robot designed for safe and gentle marine exploration. Inspired by bacterial flagellar propulsion, BactoBot features 12 flexible, silicone-based arms arranged on a 3D-printed dodecahedral frame. The design provides inherent compliance, redundancy, and the potential for omnidirectional movement. The prototype was fabricated using accessible DIY methods, including food-grade silicone molding, 3D printing, and off-the-shelf microcontrollers. Waterproofing and buoyancy calibration protocols were developed, and the robot was successfully tested in a controlled water tank, demonstrating forward motion and turning. The results validate the feasibility of replicating complex biological locomotion at low cost. The project lays a foundation for environmentally conscious robotic tools, particularly for marine science in resource-constrained settings, and identifies pathways toward autonomous operation and field deployment.
>
---
#### [new 018] Human-like Navigation in a World Built for Humans
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出ReasonNav，一个模仿人类导航行为的模块化系统，旨在解决机器人在大型建筑中导航效率低的问题。通过结合视觉-语言模型实现类人推理能力，提升导航效率与智能化水平。**

- **链接: [http://arxiv.org/pdf/2509.21189v1](http://arxiv.org/pdf/2509.21189v1)**

> **作者:** Bhargav Chandaka; Gloria X. Wang; Haozhe Chen; Henry Che; Albert J. Zhai; Shenlong Wang
>
> **备注:** CoRL 2025. Project website: https://reasonnav.github.io/
>
> **摘要:** When navigating in a man-made environment they haven't visited before--like an office building--humans employ behaviors such as reading signs and asking others for directions. These behaviors help humans reach their destinations efficiently by reducing the need to search through large areas. Existing robot navigation systems lack the ability to execute such behaviors and are thus highly inefficient at navigating within large environments. We present ReasonNav, a modular navigation system which integrates these human-like navigation skills by leveraging the reasoning capabilities of a vision-language model (VLM). We design compact input and output abstractions based on navigation landmarks, allowing the VLM to focus on language understanding and reasoning. We evaluate ReasonNav on real and simulated navigation tasks and show that the agent successfully employs higher-order reasoning to navigate efficiently in large, complex buildings.
>
---
#### [new 019] Equi-RO: A 4D mmWave Radar Odometry via Equivariant Networks
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Equi-RO，一种基于等变网络的4D毫米波雷达里程计方法，用于在无GPS环境下提升自主车辆和机器人的位姿估计精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.20674v1](http://arxiv.org/pdf/2509.20674v1)**

> **作者:** Zeyu Han; Shuocheng Yang; Minghan Zhu; Fang Zhang; Shaobing Xu; Maani Ghaffari; Jianqiang Wang
>
> **摘要:** Autonomous vehicles and robots rely on accurate odometry estimation in GPS-denied environments. While LiDARs and cameras struggle under extreme weather, 4D mmWave radar emerges as a robust alternative with all-weather operability and velocity measurement. In this paper, we introduce Equi-RO, an equivariant network-based framework for 4D radar odometry. Our algorithm pre-processes Doppler velocity into invariant node and edge features in the graph, and employs separate networks for equivariant and invariant feature processing. A graph-based architecture enhances feature aggregation in sparse radar data, improving inter-frame correspondence. Experiments on the open-source dataset and self-collected dataset show Equi-RO outperforms state-of-the-art algorithms in accuracy and robustness. Overall, our method achieves 10.7% and 20.0% relative improvements in translation and rotation accuracy, respectively, compared to the best baseline on the open-source dataset.
>
---
#### [new 020] Rich State Observations Empower Reinforcement Learning to Surpass PID: A Drone Ball Balancing Study
- **分类: cs.RO**

- **简介: 该论文研究无人机球平衡任务，旨在解决无人机通过缆绳控制球在可动横梁上稳定的问题。提出分层控制框架，利用强化学习（RL）策略替代PID控制器，实验表明RL因更丰富的状态观测表现更优，强调了状态表示对控制系统性能的重要性。**

- **链接: [http://arxiv.org/pdf/2509.21122v1](http://arxiv.org/pdf/2509.21122v1)**

> **作者:** Mingjiang Liu; Hailong Huang
>
> **备注:** Accepted for presentation at the Advancements in Aerial Physical Interaction Workshop of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025
>
> **摘要:** This paper addresses a drone ball-balancing task, in which a drone stabilizes a ball atop a movable beam through cable-based interaction. We propose a hierarchical control framework that decouples high-level balancing policy from low-level drone control, and train a reinforcement learning (RL) policy to handle the high-level decision-making. Simulation results show that the RL policy achieves superior performance compared to carefully tuned PID controllers within the same hierarchical structure. Through systematic comparative analysis, we demonstrate that RL's advantage stems not from improved parameter tuning or inherent nonlinear mapping capabilities, but from its ability to effectively utilize richer state observations. These findings underscore the critical role of comprehensive state representation in learning-based systems and suggest that enhanced sensing could be instrumental in improving controller performance.
>
---
#### [new 021] Efficient Differentiable Contact Model with Long-range Influence
- **分类: cs.RO**

- **简介: 该论文属于物理模拟与优化任务，旨在解决不同iable接触模型中梯度行为不稳定的问题。作者提出了接触模型应满足的性质，并设计了一个高效且梯度稳定的新模型，提升了控制与优化任务的效果。**

- **链接: [http://arxiv.org/pdf/2509.20917v1](http://arxiv.org/pdf/2509.20917v1)**

> **作者:** Xiaohan Ye; Kui Wu; Zherong Pan; Taku Komura
>
> **摘要:** With the maturation of differentiable physics, its role in various downstream applications: such as model predictive control, robotic design optimization, and neural PDE solvers, has become increasingly important. However, the derivative information provided by differentiable simulators can exhibit abrupt changes or vanish altogether, impeding the convergence of gradient-based optimizers. In this work, we demonstrate that such erratic gradient behavior is closely tied to the design of contact models. We further introduce a set of properties that a contact model must satisfy to ensure well-behaved gradient information. Lastly, we present a practical contact model for differentiable rigid-body simulators that satisfies all of these properties while maintaining computational efficiency. Our experiments show that, even from simple initializations, our contact model can discover complex, contact-rich control signals, enabling the successful execution of a range of downstream locomotion and manipulation tasks.
>
---
#### [new 022] Normalizing Flows are Capable Visuomotor Policy Learning Models
- **分类: cs.RO**

- **简介: 该论文提出基于Normalizing Flows的视觉运动策略模型，用于解决机器人任务中的高效推理与不确定性量化问题。相比扩散模型，其推理速度提升30倍且性能相当或更优，适用于通用机器人领域。**

- **链接: [http://arxiv.org/pdf/2509.21073v1](http://arxiv.org/pdf/2509.21073v1)**

> **作者:** Simon Kristoffersson Lind; Jialong Li; Maj Stenmark; Volker Krüger
>
> **摘要:** The field of general purpose robotics has recently embraced powerful probabilistic models, such as diffusion models, to model and learn complex behaviors. However, these models often come with significant trade-offs, namely high computational costs for inference and a fundamental inability to quantify output uncertainty. We argue that a model's trustworthiness, a critical factor for reliable, general-purpose robotics, is inherently linked to its ability to provide confidence measures. In this work, we introduce Normalizing Flows Policy, a novel visuomotor policy learning model based on Normalizing Flows. We show that Normalizing Flows are a natural and powerful alternative to diffusion models, providing both a statistically sound measure of confidence and a highly efficient inference process. Through comprehensive experiments across four distinct simulated robotic tasks, we demonstrate that Normalizing Flows Policy achieves performance comparable to, and often surpassing, Diffusion Policy, and it does so not only with improved sample efficiency but also with up to 30 times faster inference. Additionally, our ablation study validates several key architectural and training techniques that enable Normalizing Flows to perform well in this domain.
>
---
#### [new 023] Taxonomy-aware Dynamic Motion Generation on Hyperbolic Manifolds
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出GPHDM模型，用于机器人生成符合运动分类结构且物理一致的动态运动。针对现有方法忽视运动层级结构的问题，将GPDM扩展到双曲流形，并结合分类先验，设计三种新机制，实现结构化、真实的运动生成。**

- **链接: [http://arxiv.org/pdf/2509.21281v1](http://arxiv.org/pdf/2509.21281v1)**

> **作者:** Luis Augenstein; Noémie Jaquier; Tamim Asfour; Leonel Rozo
>
> **备注:** 8 pages, 6 figures, 1 table
>
> **摘要:** Human-like motion generation for robots often draws inspiration from biomechanical studies, which often categorize complex human motions into hierarchical taxonomies. While these taxonomies provide rich structural information about how movements relate to one another, this information is frequently overlooked in motion generation models, leading to a disconnect between the generated motions and their underlying hierarchical structure. This paper introduces the \ac{gphdm}, a novel approach that learns latent representations preserving both the hierarchical structure of motions and their temporal dynamics to ensure physical consistency. Our model achieves this by extending the dynamics prior of the Gaussian Process Dynamical Model (GPDM) to the hyperbolic manifold and integrating it with taxonomy-aware inductive biases. Building on this geometry- and taxonomy-aware frameworks, we propose three novel mechanisms for generating motions that are both taxonomically-structured and physically-consistent: two probabilistic recursive approaches and a method based on pullback-metric geodesics. Experiments on generating realistic motion sequences on the hand grasping taxonomy show that the proposed GPHDM faithfully encodes the underlying taxonomy and temporal dynamics, and generates novel physically-consistent trajectories.
>
---
#### [new 024] BiNoMaP: Learning Category-Level Bimanual Non-Prehensile Manipulation Primitives
- **分类: cs.RO**

- **简介: 该论文提出BiNoMaP，研究双臂非抓取操作技能学习。针对传统单臂和依赖环境结构的局限性，通过视频示范提取轨迹，并结合几何优化与参数化方法，实现跨物体类别的通用非抓取操作原语学习。**

- **链接: [http://arxiv.org/pdf/2509.21256v1](http://arxiv.org/pdf/2509.21256v1)**

> **作者:** Huayi Zhou; Kui Jia
>
> **备注:** under review
>
> **摘要:** Non-prehensile manipulation, encompassing ungraspable actions such as pushing, poking, and pivoting, represents a critical yet underexplored domain in robotics due to its contact-rich and analytically intractable nature. In this work, we revisit this problem from two novel perspectives. First, we move beyond the usual single-arm setup and the strong assumption of favorable external dexterity such as walls, ramps, or edges. Instead, we advocate a generalizable dual-arm configuration and establish a suite of Bimanual Non-prehensile Manipulation Primitives (BiNoMaP). Second, we depart from the prevailing RL-based paradigm and propose a three-stage, RL-free framework to learn non-prehensile skills. Specifically, we begin by extracting bimanual hand motion trajectories from video demonstrations. Due to visual inaccuracies and morphological gaps, these coarse trajectories are difficult to transfer directly to robotic end-effectors. To address this, we propose a geometry-aware post-optimization algorithm that refines raw motions into executable manipulation primitives that conform to specific motion patterns. Beyond instance-level reproduction, we further enable category-level generalization by parameterizing the learned primitives with object-relevant geometric attributes, particularly size, resulting in adaptable and general parameterized manipulation primitives. We validate BiNoMaP across a range of representative bimanual tasks and diverse object categories, demonstrating its effectiveness, efficiency, versatility, and superior generalization capability.
>
---
#### [new 025] Cyber Racing Coach: A Haptic Shared Control Framework for Teaching Advanced Driving Skills
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出一种基于触觉共享控制的“网络赛车教练”框架，旨在解决人类驾驶员在高性能驾驶技能学习中的问题。通过协作式控制与逐步减少辅助的机制，帮助驾驶员提升赛车和紧急避障能力。实验表明其效果优于自学和全辅助方式。**

- **链接: [http://arxiv.org/pdf/2509.20653v1](http://arxiv.org/pdf/2509.20653v1)**

> **作者:** Congkai Shen; Siyuan Yu; Yifan Weng; Haoran Ma; Chen Li; Hiroshi Yasuda; James Dallas; Michael Thompson; John Subosits; Tulga Ersal
>
> **摘要:** This study introduces a haptic shared control framework designed to teach human drivers advanced driving skills. In this context, shared control refers to a driving mode where the human driver collaborates with an autonomous driving system to control the steering of a vehicle simultaneously. Advanced driving skills are those necessary to safely push the vehicle to its handling limits in high-performance driving such as racing and emergency obstacle avoidance. Previous research has demonstrated the performance and safety benefits of shared control schemes using both subjective and objective evaluations. However, these schemes have not been assessed for their impact on skill acquisition on complex and demanding tasks. Prior research on long-term skill acquisition either applies haptic shared control to simple tasks or employs other feedback methods like visual and auditory aids. To bridge this gap, this study creates a cyber racing coach framework based on the haptic shared control paradigm and evaluates its performance in helping human drivers acquire high-performance driving skills. The framework introduces (1) an autonomous driving system that is capable of cooperating with humans in a highly performant driving scenario; and (2) a haptic shared control mechanism along with a fading scheme to gradually reduce the steering assistance from autonomy based on the human driver's performance during training. Two benchmarks are considered: self-learning (no assistance) and full assistance during training. Results from a human subject study indicate that the proposed framework helps human drivers develop superior racing skills compared to the benchmarks, resulting in better performance and consistency.
>
---
#### [new 026] FSGlove: An Inertial-Based Hand Tracking System with Shape-Aware Calibration
- **分类: cs.RO**

- **简介: 该论文提出FSGlove，一种基于惯性测量的高精度手部追踪系统。针对现有手套捕捉自由度不足和忽略手型个性化的问题，设计支持48个自由度追踪与个性化手型重建的系统，并通过DiffHCal方法实现高精度标定。**

- **链接: [http://arxiv.org/pdf/2509.21242v1](http://arxiv.org/pdf/2509.21242v1)**

> **作者:** Yutong Li; Jieyi Zhang; Wenqiang Xu; Tutian Tang; Cewu Lu
>
> **备注:** Presented at IROS 2025, details are available at https://fsglove.robotflow.ai
>
> **摘要:** Accurate hand motion capture (MoCap) is vital for applications in robotics, virtual reality, and biomechanics, yet existing systems face limitations in capturing high-degree-of-freedom (DoF) joint kinematics and personalized hand shape. Commercial gloves offer up to 21 DoFs, which are insufficient for complex manipulations while neglecting shape variations that are critical for contact-rich tasks. We present FSGlove, an inertial-based system that simultaneously tracks up to 48 DoFs and reconstructs personalized hand shapes via DiffHCal, a novel calibration method. Each finger joint and the dorsum are equipped with IMUs, enabling high-resolution motion sensing. DiffHCal integrates with the parametric MANO model through differentiable optimization, resolving joint kinematics, shape parameters, and sensor misalignment during a single streamlined calibration. The system achieves state-of-the-art accuracy, with joint angle errors of less than 2.7 degree, and outperforms commercial alternatives in shape reconstruction and contact fidelity. FSGlove's open-source hardware and software design ensures compatibility with current VR and robotics ecosystems, while its ability to capture subtle motions (e.g., fingertip rubbing) bridges the gap between human dexterity and robotic imitation. Evaluated against Nokov optical MoCap, FSGlove advances hand tracking by unifying the kinematic and contact fidelity. Hardware design, software, and more results are available at: https://sites.google.com/view/fsglove.
>
---
#### [new 027] SLAM-Free Visual Navigation with Hierarchical Vision-Language Perception and Coarse-to-Fine Semantic Topological Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种无需SLAM的视觉导航框架，用于腿部机器人。针对传统SLAM在快速运动、校准和传感器漂移下的脆弱性，采用语义推理与轻量拓扑表示，结合分层视觉-语言感知与粗到细语义路径规划，提升导航鲁棒性与语义理解能力。**

- **链接: [http://arxiv.org/pdf/2509.20739v1](http://arxiv.org/pdf/2509.20739v1)**

> **作者:** Guoyang Zhao; Yudong Li; Weiqing Qi; Kai Zhang; Bonan Liu; Kai Chen; Haoang Li; Jun Ma
>
> **摘要:** Conventional SLAM pipelines for legged robot navigation are fragile under rapid motion, calibration demands, and sensor drift, while offering limited semantic reasoning for task-driven exploration. To deal with these issues, we propose a vision-only, SLAM-free navigation framework that replaces dense geometry with semantic reasoning and lightweight topological representations. A hierarchical vision-language perception module fuses scene-level context with object-level cues for robust semantic inference. And a semantic-probabilistic topological map supports coarse-to-fine planning: LLM-based global reasoning for subgoal selection and vision-based local planning for obstacle avoidance. Integrated with reinforcement-learning locomotion controllers, the framework is deployable across diverse legged robot platforms. Experiments in simulation and real-world settings demonstrate consistent improvements in semantic accuracy, planning quality, and navigation success, while ablation studies further showcase the necessity of both hierarchical perception and fine local planning. This work introduces a new paradigm for SLAM-free, vision-language-driven navigation, shifting robotic exploration from geometry-centric mapping to semantics-driven decision making.
>
---
#### [new 028] MPC-based Deep Reinforcement Learning Method for Space Robotic Control with Fuel Sloshing Mitigation
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究卫星自主对接任务，解决微重力下燃料晃动影响稳定性的难题。提出结合SAC和MPC的方法，提升控制鲁棒性和对接精度，验证了其在仿真和实验中的优越性。**

- **链接: [http://arxiv.org/pdf/2509.21045v1](http://arxiv.org/pdf/2509.21045v1)**

> **作者:** Mahya Ramezani; M. Amin Alandihallaj; Barış Can Yalçın; Miguel Angel Olivares Mendez; Holger Voos
>
> **备注:** Pre-print version submitted to IEEE IROS
>
> **摘要:** This paper presents an integrated Reinforcement Learning (RL) and Model Predictive Control (MPC) framework for autonomous satellite docking with a partially filled fuel tank. Traditional docking control faces challenges due to fuel sloshing in microgravity, which induces unpredictable forces affecting stability. To address this, we integrate Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC) RL algorithms with MPC, leveraging MPC's predictive capabilities to accelerate RL training and improve control robustness. The proposed approach is validated through Zero-G Lab of SnT experiments for planar stabilization and high-fidelity numerical simulations for 6-DOF docking with fuel sloshing dynamics. Simulation results demonstrate that SAC-MPC achieves superior docking accuracy, higher success rates, and lower control effort, outperforming standalone RL and PPO-MPC methods. This study advances fuel-efficient and disturbance-resilient satellite docking, enhancing the feasibility of on-orbit refueling and servicing missions.
>
---
#### [new 029] Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究从单张图像高效构建隐式表面模型的任务，旨在解决传统方法需多视角图像和耗时长的问题。提出FINS框架，结合多分辨率哈希网格编码器与轻量网络，实现快速、高精度的SDF重建，并验证其在机器人路径规划中的应用效果。**

- **链接: [http://arxiv.org/pdf/2509.20681v1](http://arxiv.org/pdf/2509.20681v1)**

> **作者:** Wei-Teng Chu; Tianyi Zhang; Matthew Johnson-Roberson; Weiming Zhi
>
> **摘要:** Implicit representations have been widely applied in robotics for obstacle avoidance and path planning. In this paper, we explore the problem of constructing an implicit distance representation from a single image. Past methods for implicit surface reconstruction, such as \emph{NeuS} and its variants generally require a large set of multi-view images as input, and require long training times. In this work, we propose Fast Image-to-Neural Surface (FINS), a lightweight framework that can reconstruct high-fidelity surfaces and SDF fields based on a single or a small set of images. FINS integrates a multi-resolution hash grid encoder with lightweight geometry and color heads, making the training via an approximate second-order optimizer highly efficient and capable of converging within a few seconds. Additionally, we achieve the construction of a neural surface requiring only a single RGB image, by leveraging pre-trained foundation models to estimate the geometry inherent in the image. Our experiments demonstrate that under the same conditions, our method outperforms state-of-the-art baselines in both convergence speed and accuracy on surface reconstruction and SDF field estimation. Moreover, we demonstrate the applicability of FINS for robot surface following tasks and show its scalability to a variety of benchmark datasets.
>
---
#### [new 030] EEG-Driven AR-Robot System for Zero-Touch Grasping Manipulation
- **分类: cs.RO**

- **简介: 该论文提出一种基于脑电信号（EEG）的增强现实（AR）机器人系统，用于实现零触控抓取操作。针对现有BCI-Robot系统信号不稳定、目标选择不灵活等问题，设计了包含EEG解码、AR神经反馈和自主抓取的闭环框架，提升了控制稳定性和成功率。**

- **链接: [http://arxiv.org/pdf/2509.20656v1](http://arxiv.org/pdf/2509.20656v1)**

> **作者:** Junzhe Wang; Jiarui Xie; Pengfei Hao; Zheng Li; Yi Cai
>
> **备注:** 8 pages, 14 figures, submitted to ICRA 2026
>
> **摘要:** Reliable brain-computer interface (BCI) control of robots provides an intuitive and accessible means of human-robot interaction, particularly valuable for individuals with motor impairments. However, existing BCI-Robot systems face major limitations: electroencephalography (EEG) signals are noisy and unstable, target selection is often predefined and inflexible, and most studies remain restricted to simulation without closed-loop validation. These issues hinder real-world deployment in assistive scenarios. To address them, we propose a closed-loop BCI-AR-Robot system that integrates motor imagery (MI)-based EEG decoding, augmented reality (AR) neurofeedback, and robotic grasping for zero-touch operation. A 14-channel EEG headset enabled individualized MI calibration, a smartphone-based AR interface supported multi-target navigation with direction-congruent feedback to enhance stability, and the robotic arm combined decision outputs with vision-based pose estimation for autonomous grasping. Experiments are conducted to validate the framework: MI training achieved 93.1 percent accuracy with an average information transfer rate (ITR) of 14.8 bit/min; AR neurofeedback significantly improved sustained control (SCI = 0.210) and achieved the highest ITR (21.3 bit/min) compared with static, sham, and no-AR baselines; and closed-loop grasping achieved a 97.2 percent success rate with good efficiency and strong user-reported control. These results show that AR feedback substantially stabilizes EEG-based control and that the proposed framework enables robust zero-touch grasping, advancing assistive robotic applications and future modes of human-robot interaction.
>
---
#### [new 031] SemSight: Probabilistic Bird's-Eye-View Prediction of Multi-Level Scene Semantics for Navigation
- **分类: cs.RO**

- **简介: 该论文提出SemSight，用于目标驱动导航中的多层级语义预测。针对现有方法无法建模房间级语义结构的问题，设计了一种概率鸟瞰图预测模型，并通过掩码监督策略训练，提升了未知区域的语义地图重建与导航效率。**

- **链接: [http://arxiv.org/pdf/2509.20839v1](http://arxiv.org/pdf/2509.20839v1)**

> **作者:** Jiaxuan He; Jiamei Ren; Chongshang Yan; Wenjie Song
>
> **摘要:** In target-driven navigation and autonomous exploration, reasonable prediction of unknown regions is crucial for efficient navigation and environment understanding. Existing methods mostly focus on single objects or geometric occupancy maps, lacking the ability to model room-level semantic structures. We propose SemSight, a probabilistic bird's-eye-view prediction model for multi-level scene semantics. The model jointly infers structural layouts, global scene context, and target area distributions, completing semantic maps of unexplored areas while estimating probability maps for target categories. To train SemSight, we simulate frontier-driven exploration on 2,000 indoor layout graphs, constructing a diverse dataset of 40,000 sequential egocentric observations paired with complete semantic maps. We adopt an encoder-decoder network as the core architecture and introduce a mask-constrained supervision strategy. This strategy applies a binary mask of unexplored areas so that supervision focuses only on unknown regions, forcing the model to infer semantic structures from the observed context. Experimental results show that SemSight improves prediction performance for key functional categories in unexplored regions and outperforms non-mask-supervised approaches on metrics such as Structural Consistency (SC) and Region Recognition Accuracy (PA). It also enhances navigation efficiency in closed-loop simulations, reducing the number of search steps when guiding robots toward target areas.
>
---
#### [new 032] Cross-Modal Instructions for Robot Motion Generation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出CrossInstruct框架，通过跨模态指令（如文本标注）替代物理示教，实现机器人运动生成。属于机器人学习任务，旨在解决示教数据收集困难、难以扩展的问题。方法结合视觉-语言模型与强化学习，实现高效策略生成与泛化。**

- **链接: [http://arxiv.org/pdf/2509.21107v1](http://arxiv.org/pdf/2509.21107v1)**

> **作者:** William Barron; Xiaoxiang Dong; Matthew Johnson-Roberson; Weiming Zhi
>
> **摘要:** Teaching robots novel behaviors typically requires motion demonstrations via teleoperation or kinaesthetic teaching, that is, physically guiding the robot. While recent work has explored using human sketches to specify desired behaviors, data collection remains cumbersome, and demonstration datasets are difficult to scale. In this paper, we introduce an alternative paradigm, Learning from Cross-Modal Instructions, where robots are shaped by demonstrations in the form of rough annotations, which can contain free-form text labels, and are used in lieu of physical motion. We introduce the CrossInstruct framework, which integrates cross-modal instructions as examples into the context input to a foundational vision-language model (VLM). The VLM then iteratively queries a smaller, fine-tuned model, and synthesizes the desired motion over multiple 2D views. These are then subsequently fused into a coherent distribution over 3D motion trajectories in the robot's workspace. By incorporating the reasoning of the large VLM with a fine-grained pointing model, CrossInstruct produces executable robot behaviors that generalize beyond the environment of in the limited set of instruction examples. We then introduce a downstream reinforcement learning pipeline that leverages CrossInstruct outputs to efficiently learn policies to complete fine-grained tasks. We rigorously evaluate CrossInstruct on benchmark simulation tasks and real hardware, demonstrating effectiveness without additional fine-tuning and providing a strong initialization for policies subsequently refined via reinforcement learning.
>
---
#### [new 033] Automotive-ENV: Benchmarking Multimodal Agents in Vehicle Interface Systems
- **分类: cs.RO; cs.CL; F.2.2; I.2.7**

- **简介: 该论文提出Automotive-ENV，首个针对车载GUI的高保真基准环境，并设计ASURADA代理，通过地理信息提升驾驶安全任务性能。**

- **链接: [http://arxiv.org/pdf/2509.21143v1](http://arxiv.org/pdf/2509.21143v1)**

> **作者:** Junfeng Yan; Biao Wu; Meng Fang; Ling Chen
>
> **备注:** 10 pages, 5 figures,
>
> **摘要:** Multimodal agents have demonstrated strong performance in general GUI interactions, but their application in automotive systems has been largely unexplored. In-vehicle GUIs present distinct challenges: drivers' limited attention, strict safety requirements, and complex location-based interaction patterns. To address these challenges, we introduce Automotive-ENV, the first high-fidelity benchmark and interaction environment tailored for vehicle GUIs. This platform defines 185 parameterized tasks spanning explicit control, implicit intent understanding, and safety-aware tasks, and provides structured multimodal observations with precise programmatic checks for reproducible evaluation. Building on this benchmark, we propose ASURADA, a geo-aware multimodal agent that integrates GPS-informed context to dynamically adjust actions based on location, environmental conditions, and regional driving norms. Experiments show that geo-aware information significantly improves success on safety-aware tasks, highlighting the importance of location-based context in automotive environments. We will release Automotive-ENV, complete with all tasks and benchmarking tools, to further the development of safe and adaptive in-vehicle agents.
>
---
#### [new 034] GraspFactory: A Large Object-Centric Grasping Dataset
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出GraspFactory，一个包含超1亿个6-DoF抓取数据的大型数据集，用于训练泛化能力强的机器人抓取模型，解决实际场景中面对新物体时的泛化问题。**

- **链接: [http://arxiv.org/pdf/2509.20550v1](http://arxiv.org/pdf/2509.20550v1)**

> **作者:** Srinidhi Kalgundi Srinivas; Yash Shukla; Adam Arnold; Sachin Chitta
>
> **摘要:** Robotic grasping is a crucial task in industrial automation, where robots are increasingly expected to handle a wide range of objects. However, a significant challenge arises when robot grasping models trained on limited datasets encounter novel objects. In real-world environments such as warehouses or manufacturing plants, the diversity of objects can be vast, and grasping models need to generalize to this diversity. Training large, generalizable robot-grasping models requires geometrically diverse datasets. In this paper, we introduce GraspFactory, a dataset containing over 109 million 6-DoF grasps collectively for the Franka Panda (with 14,690 objects) and Robotiq 2F-85 grippers (with 33,710 objects). GraspFactory is designed for training data-intensive models, and we demonstrate the generalization capabilities of one such model trained on a subset of GraspFactory in both simulated and real-world settings. The dataset and tools are made available for download at https://graspfactory.github.io/.
>
---
#### [new 035] MTRDrive: Memory-Tool Synergistic Reasoning for Robust Autonomous Driving in Corner Cases
- **分类: cs.RO**

- **简介: 该论文提出MTRDrive框架，用于提升端到端自动驾驶在corner case中的鲁棒性。针对视觉-语言模型在分布外场景下的脆弱性问题，结合记忆与动态工具包实现协同推理，并在新Roadwork-VLM基准上验证了其优越的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.20843v1](http://arxiv.org/pdf/2509.20843v1)**

> **作者:** Ziang Luo; Kangan Qian; Jiahua Wang; Yuechen Luo; Jinyu Miao; Zheng Fu; Yunlong Wang; Sicong Jiang; Zilin Huang; Yifei Hu; Yuhao Yang; Hao Ye; Mengmeng Yang; Xiaojian Dong; Kun Jiang; Diange Yang
>
> **备注:** 8 pages
>
> **摘要:** Vision-Language Models(VLMs) have demonstrated significant potential for end-to-end autonomous driving, yet a substantial gap remains between their current capabilities and the reliability necessary for real-world deployment. A critical challenge is their fragility, characterized by hallucinations and poor generalization in out-of-distribution (OOD) scenarios. To bridge this gap, we introduce MTRDrive, a novel framework that integrates procedural driving experiences with a dynamic toolkit to enhance generalization and proactive decision-making. MTRDrive addresses these limitations through a closed-loop system that combines a memory-based experience retrieval mechanism with dynamic toolkits. This synergy enables the model to interact more effectively with its environment, improving both reasoning and decision-making capabilities with the help of our memory-tool synergistic reasoning. Additionally, we introduce a new benchmark based on complex Roadwork construction scenarios to rigorously evaluate zero-shot generalization. Extensive experiments demonstrate the superior effectiveness of our approach. On the public NAVSIM benchmark, our 3B-parameter MTRDrive model achieves an exceptional PDMS of 88.3 without chain-of-thought and sets a state-of-the-art performance bar on high-level planning, with a driving metric score of 79.8\% and a planning accuracy of 82.6\%. Rigorous zero-shot evaluation on the new Roadwork-VLM benchmark shows a strong ability to reason robustly in unseen scenarios, achieving a driving metric score of 80.2\%. These results highlight MTRDrive's potential to advance autonomous driving toward safer and more reliable systems.
>
---
#### [new 036] Autoregressive End-to-End Planning with Time-Invariant Spatial Alignment and Multi-Objective Policy Refinement
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对自动驾驶端到端规划任务，旨在解决自回归模型因时空不一致导致的性能瓶颈。提出TISA模块实现时间不变的空间对齐，并引入运动学动作预测头和多目标DPO优化，提升轨迹可行性与行为精细度，取得SOTA效果。**

- **链接: [http://arxiv.org/pdf/2509.20938v1](http://arxiv.org/pdf/2509.20938v1)**

> **作者:** Jianbo Zhao; Taiyu Ban; Xiangjie Li; Xingtai Gui; Hangning Zhou; Lei Liu; Hongwei Zhao; Bin Li
>
> **摘要:** The inherent sequential modeling capabilities of autoregressive models make them a formidable baseline for end-to-end planning in autonomous driving. Nevertheless, their performance is constrained by a spatio-temporal misalignment, as the planner must condition future actions on past sensory data. This creates an inconsistent worldview, limiting the upper bound of performance for an otherwise powerful approach. To address this, we propose a Time-Invariant Spatial Alignment (TISA) module that learns to project initial environmental features into a consistent ego-centric frame for each future time step, effectively correcting the agent's worldview without explicit future scene prediction. In addition, we employ a kinematic action prediction head (i.e., acceleration and yaw rate) to ensure physically feasible trajectories. Finally, we introduce a multi-objective post-training stage using Direct Preference Optimization (DPO) to move beyond pure imitation. Our approach provides targeted feedback on specific driving behaviors, offering a more fine-grained learning signal than the single, overall objective used in standard DPO. Our model achieves a state-of-the-art 89.8 PDMS on the NAVSIM dataset among autoregressive models. The video document is available at https://tisa-dpo-e2e.github.io/.
>
---
#### [new 037] Building Information Models to Robot-Ready Site Digital Twins (BIM2RDT): An Agentic AI Safety-First Framework
- **分类: cs.RO**

- **简介: 该论文提出BIM2RDT框架，旨在将静态BIM转化为动态、机器人可用的数字孪生，提升施工现场安全与效率。通过融合BIM、IoT和机器人数据，并引入SG-ICP算法优化对齐精度，实现安全事件实时监测与路径优化。**

- **链接: [http://arxiv.org/pdf/2509.20705v1](http://arxiv.org/pdf/2509.20705v1)**

> **作者:** Reza Akhavian; Mani Amani; Johannes Mootz; Robert Ashe; Behrad Beheshti
>
> **摘要:** The adoption of cyber-physical systems and jobsite intelligence that connects design models, real-time site sensing, and autonomous field operations can dramatically enhance digital management in the construction industry. This paper introduces BIM2RDT (Building Information Models to Robot-Ready Site Digital Twins), an agentic artificial intelligence (AI) framework designed to transform static Building Information Modeling (BIM) into dynamic, robot-ready digital twins (DTs) that prioritize safety during execution. The framework bridges the gap between pre-existing BIM data and real-time site conditions by integrating three key data streams: geometric and semantic information from BIM models, activity data from IoT sensor networks, and visual-spatial data collected by robots during site traversal. The methodology introduces Semantic-Gravity ICP (SG-ICP), a point cloud registration algorithm that leverages large language model (LLM) reasoning. Unlike traditional methods, SG-ICP utilizes an LLM to infer object-specific, plausible orientation priors based on BIM semantics, improving alignment accuracy by avoiding convergence on local minima. This creates a feedback loop where robot-collected data updates the DT, which in turn optimizes paths for missions. The framework employs YOLOE object detection and Shi-Tomasi corner detection to identify and track construction elements while using BIM geometry as a priori maps. The framework also integrates real-time Hand-Arm Vibration (HAV) monitoring, mapping sensor-detected safety events to the digital twin using IFC standards for intervention. Experiments demonstrate SG-ICP's superiority over standard ICP, achieving RMSE reductions of 64.3%--88.3% in alignment across scenarios with occluded features, ensuring plausible orientations. HAV integration triggers warnings upon exceeding exposure limits, enhancing compliance with ISO 5349-1.
>
---
#### [new 038] Digital Twin-Guided Robot Path Planning: A Beta-Bernoulli Fusion with Large Language Model as a Sensor
- **分类: cs.RO**

- **简介: 该论文提出一种基于数字孪生的机器人路径规划方法，融合BIM语义地图与自然语言指令。通过将大语言模型视为传感器，采用Beta-Bernoulli贝叶斯更新机制，结合用户提示中的语义和情感信息，提升路径安全性与适应性，适用于建筑领域任务。**

- **链接: [http://arxiv.org/pdf/2509.20709v1](http://arxiv.org/pdf/2509.20709v1)**

> **作者:** Mani Amani; Reza Akhavian
>
> **摘要:** Integrating natural language (NL) prompts into robotic mission planning has attracted significant interest in recent years. In the construction domain, Building Information Models (BIM) encapsulate rich NL descriptions of the environment. We present a novel framework that fuses NL directives with BIM-derived semantic maps via a Beta-Bernoulli Bayesian fusion by interpreting the LLM as a sensor: each obstacle's design-time repulsive coefficient is treated as a Beta(alpha, beta) random variable and LLM-returned danger scores are incorporated as pseudo-counts to update alpha and beta. The resulting posterior mean yields a continuous, context-aware repulsive gain that augments a Euclidean-distance-based potential field for cost heuristics. By adjusting gains based on sentiment and context inferred from user prompts, our method guides robots along safer, more context-aware paths. This provides a numerically stable method that can chain multiple natural commands and prompts from construction workers and foreman to enable planning while giving flexibility to be integrated in any learned or classical AI framework. Simulation results demonstrate that this Beta-Bernoulli fusion yields both qualitative and quantitative improvements in path robustness and validity.
>
---
#### [new 039] RobotDancing: Residual-Action Reinforcement Learning Enables Robust Long-Horizon Humanoid Motion Tracking
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RobotDancing，基于残差动作强化学习，解决人形机器人长期动态运动跟踪中因模型-实际差异导致的误差累积问题。通过统一框架实现端到端训练与零样本仿真到现实的迁移，提升了运动跟踪的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.20717v1](http://arxiv.org/pdf/2509.20717v1)**

> **作者:** Zhenguo Sun; Yibo Peng; Yuan Meng; Xukun Li; Bo-Sheng Huang; Zhenshan Bing; Xinlong Wang; Alois Knoll
>
> **摘要:** Long-horizon, high-dynamic motion tracking on humanoids remains brittle because absolute joint commands cannot compensate model-plant mismatch, leading to error accumulation. We propose RobotDancing, a simple, scalable framework that predicts residual joint targets to explicitly correct dynamics discrepancies. The pipeline is end-to-end--training, sim-to-sim validation, and zero-shot sim-to-real--and uses a single-stage reinforcement learning (RL) setup with a unified observation, reward, and hyperparameter configuration. We evaluate primarily on Unitree G1 with retargeted LAFAN1 dance sequences and validate transfer on H1/H1-2. RobotDancing can track multi-minute, high-energy behaviors (jumps, spins, cartwheels) and deploys zero-shot to hardware with high motion tracking quality.
>
---
#### [new 040] Joint Flow Trajectory Optimization For Feasible Robot Motion Generation from Video Demonstrations
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出JFTO框架，用于从视频演示中生成可行的机器人运动轨迹。针对人体动作直接模仿的不足，其工作重点在于平衡抓取姿态选择、物体轨迹生成和避障问题，通过概率建模实现高质量的机器人操作任务学习。**

- **链接: [http://arxiv.org/pdf/2509.20703v1](http://arxiv.org/pdf/2509.20703v1)**

> **作者:** Xiaoxiang Dong; Matthew Johnson-Roberson; Weiming Zhi
>
> **摘要:** Learning from human video demonstrations offers a scalable alternative to teleoperation or kinesthetic teaching, but poses challenges for robot manipulators due to embodiment differences and joint feasibility constraints. We address this problem by proposing the Joint Flow Trajectory Optimization (JFTO) framework for grasp pose generation and object trajectory imitation under the video-based Learning-from-Demonstration (LfD) paradigm. Rather than directly imitating human hand motions, our method treats demonstrations as object-centric guides, balancing three objectives: (i) selecting a feasible grasp pose, (ii) generating object trajectories consistent with demonstrated motions, and (iii) ensuring collision-free execution within robot kinematics. To capture the multimodal nature of demonstrations, we extend flow matching to $\SE(3)$ for probabilistic modeling of object trajectories, enabling density-aware imitation that avoids mode collapse. The resulting optimization integrates grasp similarity, trajectory likelihood, and collision penalties into a unified differentiable objective. We validate our approach in both simulation and real-world experiments across diverse real-world manipulation tasks.
>
---
#### [new 041] MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MASt3R-Fusion，一种融合视觉前馈模型与IMU、GNSS的SLAM框架，旨在解决低纹理环境和尺度模糊问题。通过引入Sim(3)对齐约束至SE(3)因子图，实现多传感器信息紧耦合，提升定位与建图精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.20757v1](http://arxiv.org/pdf/2509.20757v1)**

> **作者:** Yuxuan Zhou; Xingxing Li; Shengyu Li; Zhuohao Yan; Chunxi Xia; Shaoquan Feng
>
> **摘要:** Visual SLAM is a cornerstone technique in robotics, autonomous driving and extended reality (XR), yet classical systems often struggle with low-texture environments, scale ambiguity, and degraded performance under challenging visual conditions. Recent advancements in feed-forward neural network-based pointmap regression have demonstrated the potential to recover high-fidelity 3D scene geometry directly from images, leveraging learned spatial priors to overcome limitations of traditional multi-view geometry methods. However, the widely validated advantages of probabilistic multi-sensor information fusion are often discarded in these pipelines. In this work, we propose MASt3R-Fusion,a multi-sensor-assisted visual SLAM framework that tightly integrates feed-forward pointmap regression with complementary sensor information, including inertial measurements and GNSS data. The system introduces Sim(3)-based visualalignment constraints (in the Hessian form) into a universal metric-scale SE(3) factor graph for effective information fusion. A hierarchical factor graph design is developed, which allows both real-time sliding-window optimization and global optimization with aggressive loop closures, enabling real-time pose tracking, metric-scale structure perception and globally consistent mapping. We evaluate our approach on both public benchmarks and self-collected datasets, demonstrating substantial improvements in accuracy and robustness over existing visual-centered multi-sensor SLAM systems. The code will be released open-source to support reproducibility and further research (https://github.com/GREAT-WHU/MASt3R-Fusion).
>
---
#### [new 042] MELEGROS: Monolithic Elephant-inspired Gripper with Optical Sensors
- **分类: cs.RO**

- **简介: 该论文提出MELEGROS，一种受象鼻启发的软体夹爪，通过单次3D打印集成光学传感器与气动腔室，实现触觉与本体感知。旨在解决传统多材料结构导致的机械不匹配问题，展示了一种具备多种抓取能力的仿生柔性机器人新范式。**

- **链接: [http://arxiv.org/pdf/2509.20510v1](http://arxiv.org/pdf/2509.20510v1)**

> **作者:** Petr Trunin; Diana Cafiso; Anderson Brazil Nardin; Trevor Exley; Lucia Beccai
>
> **备注:** 15 pages, 6 figures. SI 18 pages, 19 figures. Submitted to Wiley Advanced Science
>
> **摘要:** The elephant trunk exemplifies a natural gripper where structure, actuation, and sensing are seamlessly integrated. Inspired by the distal morphology of the African elephant trunk, we present MELEGROS, a Monolithic ELEphant-inspired GRipper with Optical Sensors, emphasizing sensing as an intrinsic, co-fabricated capability. Unlike multi-material or tendon-based approaches, MELEGROS directly integrates six optical waveguide sensors and five pneumatic chambers into a pneumatically actuated lattice structure (12.5 mm cell size) using a single soft resin and one continuous 3D print. This eliminates mechanical mismatches between sensors, actuators, and body, reducing model uncertainty and enabling simulation-guided sensor design and placement. Only four iterations were required to achieve the final prototype, which features a continuous structure capable of elongation, compression, and bending while decoupling tactile and proprioceptive signals. MELEGROS (132 g) lifts more than twice its weight, performs bioinspired actions such as pinching, scooping, and reaching, and delicately grasps fragile items like grapes. The integrated optical sensors provide distinct responses to touch, bending, and chamber deformation, enabling multifunctional perception. MELEGROS demonstrates a new paradigm for soft robotics where fully embedded sensing and continuous structures inherently support versatile, bioinspired manipulation.
>
---
#### [new 043] Uncertainty-Aware Active Source Tracking of Marine Pollution using Unmanned Surface Vehicles
- **分类: cs.RO**

- **简介: 该论文提出一种面向无人水面船的不确定性感知污染源追踪框架，结合高精度污染扩散模拟与路径规划技术，实现对海洋污染源的高效定位，提升环境监测的自主性。**

- **链接: [http://arxiv.org/pdf/2509.20593v1](http://arxiv.org/pdf/2509.20593v1)**

> **作者:** Song Ma; Richard Bucknall; Yuanchang Liu
>
> **备注:** Accepted for presentation at Oceantech: Marine Robotics & Science Workshop, IROS 2025
>
> **摘要:** This paper proposes an uncertainty-aware marine pollution source tracking framework for unmanned surface vehicles (USVs). By integrating high-fidelity marine pollution dispersion simulation with informative path planning techniques, we demonstrate effective identification of pollution sources in marine environments. The proposed approach is implemented based on Robot Operating System (ROS), processing real-time sensor data to update probabilistic source location estimates. The system progressively refines the estimation of source location while quantifying uncertainty levels in its predictions. Experiments conducted in simulated environments with varying source locations, flow conditions, and starting positions demonstrate the framework's ability to localise pollution sources with high accuracy. Results show that the proposed approach achieves reliable source localisation efficiently. This work contributes to the development of full autonomous environmental monitoring capabilities essential for rapid response to marine pollution incidents.
>
---
#### [new 044] Boosting LiDAR-Based Localization with Semantic Insight: Camera Projection versus Direct LiDAR Segmentation
- **分类: cs.RO**

- **简介: 该论文属于LiDAR定位任务，旨在解决多传感器配置下语义分割精度低的问题。提出将相机语义信息投影到LiDAR点云中，提升定位精度与鲁棒性，并通过多种传感器和实际道路测试验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2509.20486v1](http://arxiv.org/pdf/2509.20486v1)**

> **作者:** Sven Ochs; Philip Schörner; Marc René Zofka; J. Marius Zöllner
>
> **摘要:** Semantic segmentation of LiDAR data presents considerable challenges, particularly when dealing with diverse sensor types and configurations. However, incorporating semantic information can significantly enhance the accuracy and robustness of LiDAR-based localization techniques for autonomous mobile systems. We propose an approach that integrates semantic camera data with LiDAR segmentation to address this challenge. By projecting LiDAR points into the semantic segmentation space of the camera, our method enhances the precision and reliability of the LiDAR-based localization pipeline. For validation, we utilize the CoCar NextGen platform from the FZI Research Center for Information Technology, which offers diverse sensor modalities and configurations. The sensor setup of CoCar NextGen enables a thorough analysis of different sensor types. Our evaluation leverages the state-of-the-art Depth-Anything network for camera image segmentation and an adaptive segmentation network for LiDAR segmentation. To establish a reliable ground truth for LiDAR-based localization, we make us of a Global Navigation Satellite System (GNSS) solution with Real-Time Kinematic corrections (RTK). Additionally, we conduct an extensive 55 km drive through the city of Karlsruhe, Germany, covering a variety of environments, including urban areas, multi-lane roads, and rural highways. This multimodal approach paves the way for more reliable and precise autonomous navigation systems, particularly in complex real-world environments.
>
---
#### [new 045] AnywhereVLA: Language-Conditioned Exploration and Mobile Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AnywhereVLA，一个用于未知室内环境的语言指令驱动移动操作系统。通过结合SLAM、语义地图和语言引导的抓取与放置策略，解决自然语言下的多房间物品搬运问题，实现嵌入式硬件上的实时任务执行。**

- **链接: [http://arxiv.org/pdf/2509.21006v1](http://arxiv.org/pdf/2509.21006v1)**

> **作者:** Konstantin Gubernatorov; Artem Voronov; Roman Voronov; Sergei Pasynkov; Stepan Perminov; Ziang Guo; Dzmitry Tsetserukou
>
> **摘要:** We address natural language pick-and-place in unseen, unpredictable indoor environments with AnywhereVLA, a modular framework for mobile manipulation. A user text prompt serves as an entry point and is parsed into a structured task graph that conditions classical SLAM with LiDAR and cameras, metric semantic mapping, and a task-aware frontier exploration policy. An approach planner then selects visibility and reachability aware pre grasp base poses. For interaction, a compact SmolVLA manipulation head is fine tuned on platform pick and place trajectories for the SO-101 by TheRobotStudio, grounding local visual context and sub-goals into grasp and place proposals. The full system runs fully onboard on consumer-level hardware, with Jetson Orin NX for perception and VLA and an Intel NUC for SLAM, exploration, and control, sustaining real-time operation. We evaluated AnywhereVLA in a multi-room lab under static scenes and normal human motion. In this setting, the system achieves a $46\%$ overall task success rate while maintaining throughput on embedded compute. By combining a classical stack with a fine-tuned VLA manipulation, the system inherits the reliability of geometry-based navigation with the agility and task generalization of language-conditioned manipulation.
>
---
#### [new 046] DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps
- **分类: cs.RO**

- **简介: 该论文提出DAGDiff，用于双臂抓取任务，旨在生成稳定且无碰撞的抓取对。传统方法依赖启发式规则，泛化性差。DAGDiff通过结合几何、稳定性和碰撞感知的引导信号，在SE(3)×SE(3)空间中直接优化抓取对，提升了抓取质量和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.21145v1](http://arxiv.org/pdf/2509.21145v1)**

> **作者:** Md Faizal Karim; Vignesh Vembar; Keshab Patra; Gaurav Singh; K Madhava Krishna
>
> **摘要:** Reliable dual-arm grasping is essential for manipulating large and complex objects but remains a challenging problem due to stability, collision, and generalization requirements. Prior methods typically decompose the task into two independent grasp proposals, relying on region priors or heuristics that limit generalization and provide no principled guarantee of stability. We propose DAGDiff, an end-to-end framework that directly denoises to grasp pairs in the SE(3) x SE(3) space. Our key insight is that stability and collision can be enforced more effectively by guiding the diffusion process with classifier signals, rather than relying on explicit region detection or object priors. To this end, DAGDiff integrates geometry-, stability-, and collision-aware guidance terms that steer the generative process toward grasps that are physically valid and force-closure compliant. We comprehensively evaluate DAGDiff through analytical force-closure checks, collision analysis, and large-scale physics-based simulations, showing consistent improvements over previous work on these metrics. Finally, we demonstrate that our framework generates dual-arm grasps directly on real-world point clouds of previously unseen objects, which are executed on a heterogeneous dual-arm setup where two manipulators reliably grasp and lift them.
>
---
#### [new 047] Leveraging Temporally Extended Behavior Sharing for Multi-task Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究多任务强化学习（MTRL），旨在解决机器人应用中任务数据收集成本高、样本效率低的问题。提出MT-Lévy方法，结合跨任务行为共享与自适应探索策略，提升复杂环境下的探索效率和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.20766v1](http://arxiv.org/pdf/2509.20766v1)**

> **作者:** Gawon Lee; Daesol Cho; H. Jin Kim
>
> **备注:** Accepted for publication in the proceedings of the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Multi-task reinforcement learning (MTRL) offers a promising approach to improve sample efficiency and generalization by training agents across multiple tasks, enabling knowledge sharing between them. However, applying MTRL to robotics remains challenging due to the high cost of collecting diverse task data. To address this, we propose MT-L\'evy, a novel exploration strategy that enhances sample efficiency in MTRL environments by combining behavior sharing across tasks with temporally extended exploration inspired by L\'evy flight. MT-L\'evy leverages policies trained on related tasks to guide exploration towards key states, while dynamically adjusting exploration levels based on task success ratios. This approach enables more efficient state-space coverage, even in complex robotics environments. Empirical results demonstrate that MT-L\'evy significantly improves exploration and sample efficiency, supported by quantitative and qualitative analyses. Ablation studies further highlight the contribution of each component, showing that combining behavior sharing with adaptive exploration strategies can significantly improve the practicality of MTRL in robotics applications.
>
---
#### [new 048] Selective Progress-Aware Querying for Human-in-the-Loop Reinforcement Learning
- **分类: cs.RO; I.2.9; I.2.6; I.2.8**

- **简介: 该论文研究人类参与的强化学习（HiL-RL）任务，旨在解决真实场景中人类反馈成本高且有限的问题。提出SPARQ方法，通过在学习停滞或退步时选择性请求反馈，减少不必要的查询，提升学习效率与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.20541v1](http://arxiv.org/pdf/2509.20541v1)**

> **作者:** Anujith Muraleedharan; Anamika J H
>
> **备注:** Preprint. 8 pages, 3 figures, 1 table, 1 algorithm. CoRL 2025 style (preprint). Code/data to be released
>
> **摘要:** Human feedback can greatly accelerate robot learning, but in real-world settings, such feedback is costly and limited. Existing human-in-the-loop reinforcement learning (HiL-RL) methods often assume abundant feedback, limiting their practicality for physical robot deployment. In this work, we introduce SPARQ, a progress-aware query policy that requests feedback only when learning stagnates or worsens, thereby reducing unnecessary oracle calls. We evaluate SPARQ on a simulated UR5 cube-picking task in PyBullet, comparing against three baselines: no feedback, random querying, and always querying. Our experiments show that SPARQ achieves near-perfect task success, matching the performance of always querying while consuming about half the feedback budget. It also provides more stable and efficient learning than random querying, and significantly improves over training without feedback. These findings suggest that selective, progress-based query strategies can make HiL-RL more efficient and scalable for robots operating under realistic human effort constraints.
>
---
#### [new 049] Boosting Zero-Shot VLN via Abstract Obstacle Map-Based Waypoint Prediction with TopoGraph-and-VisitInfo-Aware Prompting
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉语言导航（VLN）任务，旨在解决连续环境中零样本导航的挑战。提出了一种结合抽象障碍地图的路径预测与拓扑图及访问信息提示的方法，提升了基于大语言模型的导航性能，在R2R-CE和RxR-CE上取得了SOTA结果。**

- **链接: [http://arxiv.org/pdf/2509.20499v1](http://arxiv.org/pdf/2509.20499v1)**

> **作者:** Boqi Li; Siyuan Li; Weiyi Wang; Anran Li; Zhong Cao; Henry X. Liu
>
> **摘要:** With the rapid progress of foundation models and robotics, vision-language navigation (VLN) has emerged as a key task for embodied agents with broad practical applications. We address VLN in continuous environments, a particularly challenging setting where an agent must jointly interpret natural language instructions, perceive its surroundings, and plan low-level actions. We propose a zero-shot framework that integrates a simplified yet effective waypoint predictor with a multimodal large language model (MLLM). The predictor operates on an abstract obstacle map, producing linearly reachable waypoints, which are incorporated into a dynamically updated topological graph with explicit visitation records. The graph and visitation information are encoded into the prompt, enabling reasoning over both spatial structure and exploration history to encourage exploration and equip MLLM with local path planning for error correction. Extensive experiments on R2R-CE and RxR-CE show that our method achieves state-of-the-art zero-shot performance, with success rates of 41% and 36%, respectively, outperforming prior state-of-the-art methods.
>
---
#### [new 050] SGAligner++: Cross-Modal Language-Aided 3D Scene Graph Alignment
- **分类: cs.GR; cs.RO**

- **简介: 该论文提出SGAligner++，用于跨模态3D场景图对齐任务。针对现有方法依赖单一模态数据、处理噪声和低重叠场景效果差的问题，构建了融合语言信息的统一嵌入空间，提升了在真实噪声环境下的对齐精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.20401v1](http://arxiv.org/pdf/2509.20401v1)**

> **作者:** Binod Singh; Sayan Deb Sarkar; Iro Armeni
>
> **摘要:** Aligning 3D scene graphs is a crucial initial step for several applications in robot navigation and embodied perception. Current methods in 3D scene graph alignment often rely on single-modality point cloud data and struggle with incomplete or noisy input. We introduce SGAligner++, a cross-modal, language-aided framework for 3D scene graph alignment. Our method addresses the challenge of aligning partially overlapping scene observations across heterogeneous modalities by learning a unified joint embedding space, enabling accurate alignment even under low-overlap conditions and sensor noise. By employing lightweight unimodal encoders and attention-based fusion, SGAligner++ enhances scene understanding for tasks such as visual localization, 3D reconstruction, and navigation, while ensuring scalability and minimal computational overhead. Extensive evaluations on real-world datasets demonstrate that SGAligner++ outperforms state-of-the-art methods by up to 40% on noisy real-world reconstructions, while enabling cross-modal generalization.
>
---
#### [new 051] Wonder Wins Ways: Curiosity-Driven Exploration through Multi-Agent Contextual Calibration
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究多智能体强化学习中的自主探索任务，旨在解决稀疏奖励环境下内在动机易受环境噪声干扰、探索效率低的问题。提出CERMIC框架，通过多智能体上下文动态校准好奇心，提升去噪能力和信息增益驱动的探索效果。**

- **链接: [http://arxiv.org/pdf/2509.20648v1](http://arxiv.org/pdf/2509.20648v1)**

> **作者:** Yiyuan Pan; Zhe Liu; Hesheng Wang
>
> **摘要:** Autonomous exploration in complex multi-agent reinforcement learning (MARL) with sparse rewards critically depends on providing agents with effective intrinsic motivation. While artificial curiosity offers a powerful self-supervised signal, it often confuses environmental stochasticity with meaningful novelty. Moreover, existing curiosity mechanisms exhibit a uniform novelty bias, treating all unexpected observations equally. However, peer behavior novelty, which encode latent task dynamics, are often overlooked, resulting in suboptimal exploration in decentralized, communication-free MARL settings. To this end, inspired by how human children adaptively calibrate their own exploratory behaviors via observing peers, we propose a novel approach to enhance multi-agent exploration. We introduce CERMIC, a principled framework that empowers agents to robustly filter noisy surprise signals and guide exploration by dynamically calibrating their intrinsic curiosity with inferred multi-agent context. Additionally, CERMIC generates theoretically-grounded intrinsic rewards, encouraging agents to explore state transitions with high information gain. We evaluate CERMIC on benchmark suites including VMAS, Meltingpot, and SMACv2. Empirical results demonstrate that exploration with CERMIC significantly outperforms SoTA algorithms in sparse-reward environments.
>
---
#### [new 052] Large Pre-Trained Models for Bimanual Manipulation in 3D
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究双臂机器人操作任务，旨在提升3D环境下的操作性能。通过将DINOv2的注意力图融入体素表示，生成语义线索并用于行为克隆策略，有效提升了RLBench基准中的表现。**

- **链接: [http://arxiv.org/pdf/2509.20579v1](http://arxiv.org/pdf/2509.20579v1)**

> **作者:** Hanna Yurchyk; Wei-Di Chang; Gregory Dudek; David Meger
>
> **备注:** Accepted to 2025 IEEE-RAS 24th International Conference on Humanoid Robots
>
> **摘要:** We investigate the integration of attention maps from a pre-trained Vision Transformer into voxel representations to enhance bimanual robotic manipulation. Specifically, we extract attention maps from DINOv2, a self-supervised ViT model, and interpret them as pixel-level saliency scores over RGB images. These maps are lifted into a 3D voxel grid, resulting in voxel-level semantic cues that are incorporated into a behavior cloning policy. When integrated into a state-of-the-art voxel-based policy, our attention-guided featurization yields an average absolute improvement of 8.2% and a relative gain of 21.9% across all tasks in the RLBench bimanual benchmark.
>
---
#### [new 053] Finding 3D Positions of Distant Objects from Noisy Camera Movement and Semantic Segmentation Sequences
- **分类: cs.CV; cs.RO; I.4.8; I.4.9**

- **简介: 该论文研究了基于相机运动和语义分割序列的远距离物体3D定位任务，旨在解决计算资源受限或目标过远时传统方法失效的问题。提出了使用粒子滤波器实现单目标和多目标定位，并通过仿真和无人机图像验证了方法的有效性与灵活性。**

- **链接: [http://arxiv.org/pdf/2509.20906v1](http://arxiv.org/pdf/2509.20906v1)**

> **作者:** Julius Pesonen; Arno Solin; Eija Honkavaara
>
> **摘要:** 3D object localisation based on a sequence of camera measurements is essential for safety-critical surveillance tasks, such as drone-based wildfire monitoring. Localisation of objects detected with a camera can typically be solved with dense depth estimation or 3D scene reconstruction. However, in the context of distant objects or tasks limited by the amount of available computational resources, neither solution is feasible. In this paper, we show that the task can be solved using particle filters for both single and multiple target scenarios. The method was studied using a 3D simulation and a drone-based image segmentation sequence with global navigation satellite system (GNSS)-based camera pose estimates. The results showed that a particle filter can be used to solve practical localisation tasks based on camera poses and image segments in these situations where other solutions fail. The particle filter is independent of the detection method, making it flexible for new tasks. The study also demonstrates that drone-based wildfire monitoring can be conducted using the proposed method paired with a pre-existing image segmentation model.
>
---
#### [new 054] Meta-Memory: Retrieving and Integrating Semantic-Spatial Memories for Robot Spatial Reasoning
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出Meta-Memory，一种基于大语言模型的机器人空间推理方法，旨在解决复杂环境中语义-空间记忆的高效检索与整合问题。通过联合语义和空间信息处理自然语言查询，显著提升了机器人在真实场景中的空间问答能力，并在多个基准上取得优异表现。**

- **链接: [http://arxiv.org/pdf/2509.20754v1](http://arxiv.org/pdf/2509.20754v1)**

> **作者:** Yufan Mao; Hanjing Ye; Wenlong Dong; Chengjie Zhang; Hong Zhang
>
> **摘要:** Navigating complex environments requires robots to effectively store observations as memories and leverage them to answer human queries about spatial locations, which is a critical yet underexplored research challenge. While prior work has made progress in constructing robotic memory, few have addressed the principled mechanisms needed for efficient memory retrieval and integration. To bridge this gap, we propose Meta-Memory, a large language model (LLM)-driven agent that constructs a high-density memory representation of the environment. The key innovation of Meta-Memory lies in its capacity to retrieve and integrate relevant memories through joint reasoning over semantic and spatial modalities in response to natural language location queries, thereby empowering robots with robust and accurate spatial reasoning capabilities. To evaluate its performance, we introduce SpaceLocQA, a large-scale dataset encompassing diverse real-world spatial question-answering scenarios. Experimental results show that Meta-Memory significantly outperforms state-of-the-art methods on both the SpaceLocQA and the public NaVQA benchmarks. Furthermore, we successfully deployed Meta-Memory on real-world robotic platforms, demonstrating its practical utility in complex environments. Project page: https://itsbaymax.github.io/meta-memory.github.io/ .
>
---
#### [new 055] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent
- **分类: cs.GR; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出SceneWeaver，一个用于3D室内场景合成的智能框架。针对现有方法在物理合理性、视觉真实性和语义一致性上的不足，SceneWeaver通过语言模型规划和工具迭代优化，实现了高质量、多样化且符合用户指令的3D场景生成。**

- **链接: [http://arxiv.org/pdf/2509.20414v1](http://arxiv.org/pdf/2509.20414v1)**

> **作者:** Yandan Yang; Baoxiong Jia; Shujie Zhang; Siyuan Huang
>
> **备注:** Accepted by NeurIPS 2025, 26 pages
>
> **摘要:** Indoor scene synthesis has become increasingly important with the rise of Embodied AI, which requires 3D environments that are not only visually realistic but also physically plausible and functionally diverse. While recent approaches have advanced visual fidelity, they often remain constrained to fixed scene categories, lack sufficient object-level detail and physical consistency, and struggle to align with complex user instructions. In this work, we present SceneWeaver, a reflective agentic framework that unifies diverse scene synthesis paradigms through tool-based iterative refinement. At its core, SceneWeaver employs a language model-based planner to select from a suite of extensible scene generation tools, ranging from data-driven generative models to visual- and LLM-based methods, guided by self-evaluation of physical plausibility, visual realism, and semantic alignment with user input. This closed-loop reason-act-reflect design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment over successive iterations. Extensive experiments on both common and open-vocabulary room types demonstrate that SceneWeaver not only outperforms prior methods on physical, visual, and semantic metrics, but also generalizes effectively to complex scenes with diverse instructions, marking a step toward general-purpose 3D environment generation. Project website: https://scene-weaver.github.io/.
>
---
## 更新

#### [replaced 001] EC-Diffuser: Multi-Object Manipulation via Entity-Centric Behavior Generation
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.18907v3](http://arxiv.org/pdf/2412.18907v3)**

> **作者:** Carl Qi; Dan Haramati; Tal Daniel; Aviv Tamar; Amy Zhang
>
> **摘要:** Object manipulation is a common component of everyday tasks, but learning to manipulate objects from high-dimensional observations presents significant challenges. These challenges are heightened in multi-object environments due to the combinatorial complexity of the state space as well as of the desired behaviors. While recent approaches have utilized large-scale offline data to train models from pixel observations, achieving performance gains through scaling, these methods struggle with compositional generalization in unseen object configurations with constrained network and dataset sizes. To address these issues, we propose a novel behavioral cloning (BC) approach that leverages object-centric representations and an entity-centric Transformer with diffusion-based optimization, enabling efficient learning from offline image data. Our method first decomposes observations into an object-centric representation, which is then processed by our entity-centric Transformer that computes attention at the object level, simultaneously predicting object dynamics and the agent's actions. Combined with the ability of diffusion models to capture multi-modal behavior distributions, this results in substantial performance improvements in multi-object tasks and, more importantly, enables compositional generalization. We present BC agents capable of zero-shot generalization to tasks with novel compositions of objects and goals, including larger numbers of objects than seen during training. We provide video rollouts on our webpage: https://sites.google.com/view/ec-diffuser.
>
---
#### [replaced 002] GUIDE: A Diffusion-Based Autonomous Robot Exploration Framework Using Global Graph Inference
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.19916v2](http://arxiv.org/pdf/2509.19916v2)**

> **作者:** Zijun Che; Yinghong Zhang; Shengyi Liang; Boyu Zhou; Jun Ma; Jinni Zhou
>
> **摘要:** Autonomous exploration in structured and complex indoor environments remains a challenging task, as existing methods often struggle to appropriately model unobserved space and plan globally efficient paths. To address these limitations, we propose GUIDE, a novel exploration framework that synergistically combines global graph inference with diffusion-based decision-making. We introduce a region-evaluation global graph representation that integrates both observed environmental data and predictions of unexplored areas, enhanced by a region-level evaluation mechanism to prioritize reliable structural inferences while discounting uncertain predictions. Building upon this enriched representation, a diffusion policy network generates stable, foresighted action sequences with significantly reduced denoising steps. Extensive simulations and real-world deployments demonstrate that GUIDE consistently outperforms state-of-the-art methods, achieving up to 18.3% faster coverage completion and a 34.9% reduction in redundant movements.
>
---
#### [replaced 003] Aegis: Automated Error Generation and Identification for Multi-Agent Systems
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.14295v3](http://arxiv.org/pdf/2509.14295v3)**

> **作者:** Fanqi Kong; Ruijie Zhang; Huaxiao Yin; Guibin Zhang; Xiaofei Zhang; Ziang Chen; Zhaowei Zhang; Xiaoyuan Zhang; Song-Chun Zhu; Xue Feng
>
> **摘要:** As Multi-Agent Systems (MAS) become increasingly autonomous and complex, understanding their error modes is critical for ensuring their reliability and safety. However, research in this area has been severely hampered by the lack of large-scale, diverse datasets with precise, ground-truth error labels. To address this bottleneck, we introduce \textbf{AEGIS}, a novel framework for \textbf{A}utomated \textbf{E}rror \textbf{G}eneration and \textbf{I}dentification for Multi-Agent \textbf{S}ystems. By systematically injecting controllable and traceable errors into initially successful trajectories, we create a rich dataset of realistic failures. This is achieved using a context-aware, LLM-based adaptive manipulator that performs sophisticated attacks like prompt injection and response corruption to induce specific, predefined error modes. We demonstrate the value of our dataset by exploring three distinct learning paradigms for the error identification task: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. Our comprehensive experiments show that models trained on AEGIS data achieve substantial improvements across all three learning paradigms. Notably, several of our fine-tuned models demonstrate performance competitive with or superior to proprietary systems an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at https://kfq20.github.io/AEGIS-Website.
>
---
#### [replaced 004] villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.23682v3](http://arxiv.org/pdf/2507.23682v3)**

> **作者:** Xiaoyu Chen; Hangxing Wei; Pushi Zhang; Chuheng Zhang; Kaixin Wang; Yanjiang Guo; Rushuai Yang; Yucen Wang; Xinquan Xiao; Li Zhao; Jianyu Chen; Jiang Bian
>
> **备注:** Project page: https://aka.ms/villa-x
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a popular paradigm for learning robot manipulation policies that can follow language instructions and generalize to novel scenarios. Recent works have begun to explore the incorporation of latent actions, abstract representations of motion between two frames, into VLA pre-training. In this paper, we introduce villa-X, a novel Vision-Language-Latent-Action (ViLLA) framework that advances latent action modeling for learning generalizable robot manipulation policies. Our approach improves both how latent actions are learned and how they are incorporated into VLA pre-training. We demonstrate that villa-X can generate latent action plans in a zero-shot fashion, even for unseen embodiments and open-vocabulary symbolic understanding. This capability enables villa-X to achieve superior performance across diverse simulation tasks in SIMPLER and on two real-world robotic setups involving both gripper and dexterous hand manipulation. These results establish villa-X as a principled and scalable paradigm for learning generalizable robot manipulation policies. We believe it provides a strong foundation for future research.
>
---
#### [replaced 005] HL-IK: A Lightweight Implementation of Human-Like Inverse Kinematics in Humanoid Arms
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20263v2](http://arxiv.org/pdf/2509.20263v2)**

> **作者:** Bingjie Chen; Zihan Wang; Zhe Han; Guoping Pan; Yi Cheng; Houde Liu
>
> **摘要:** Traditional IK methods for redundant humanoid manipulators emphasize end-effector (EE) tracking, frequently producing configurations that are valid mechanically but not human-like. We present Human-Like Inverse Kinematics (HL-IK), a lightweight IK framework that preserves EE tracking while shaping whole-arm configurations to appear human-like, without full-body sensing at runtime. The key idea is a learned elbow prior: using large-scale human motion data retargeted to the robot, we train a FiLM-modulated spatio-temporal attention network (FiSTA) to predict the next-step elbow pose from the EE target and a short history of EE-elbow states.This prediction is incorporated as a small residual alongside EE and smoothness terms in a standard Levenberg-Marquardt optimizer, making HL-IK a drop-in addition to numerical IK stacks. Over 183k simulation steps, HL-IK reduces arm-similarity position and direction error by 30.6% and 35.4% on average, and by 42.2% and 47.4% on the most challenging trajectories. Hardware teleoperation on a robot distinct from simulation further confirms the gains in anthropomorphism. HL-IK is simple to integrate, adaptable across platforms via our pipeline, and adds minimal computation, enabling human-like motions for humanoid robots. Project page: https://hl-ik.github.io/
>
---
#### [replaced 006] V2V-GoT: Vehicle-to-Vehicle Cooperative Autonomous Driving with Multimodal Large Language Models and Graph-of-Thoughts
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.18053v3](http://arxiv.org/pdf/2509.18053v3)**

> **作者:** Hsu-kuang Chiu; Ryo Hachiuma; Chien-Yi Wang; Yu-Chiang Frank Wang; Min-Hung Chen; Stephen F. Smith
>
> **备注:** Our project website: https://eddyhkchiu.github.io/v2vgot.github.io/
>
> **摘要:** Current state-of-the-art autonomous vehicles could face safety-critical situations when their local sensors are occluded by large nearby objects on the road. Vehicle-to-vehicle (V2V) cooperative autonomous driving has been proposed as a means of addressing this problem, and one recently introduced framework for cooperative autonomous driving has further adopted an approach that incorporates a Multimodal Large Language Model (MLLM) to integrate cooperative perception and planning processes. However, despite the potential benefit of applying graph-of-thoughts reasoning to the MLLM, this idea has not been considered by previous cooperative autonomous driving research. In this paper, we propose a novel graph-of-thoughts framework specifically designed for MLLM-based cooperative autonomous driving. Our graph-of-thoughts includes our proposed novel ideas of occlusion-aware perception and planning-aware prediction. We curate the V2V-GoT-QA dataset and develop the V2V-GoT model for training and testing the cooperative driving graph-of-thoughts. Our experimental results show that our method outperforms other baselines in cooperative perception, prediction, and planning tasks. Our project website: https://eddyhkchiu.github.io/v2vgot.github.io/ .
>
---
#### [replaced 007] Security of Deep Reinforcement Learning for Autonomous Driving: A Survey
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2212.06123v3](http://arxiv.org/pdf/2212.06123v3)**

> **作者:** Ambra Demontis; Srishti Gupta; Maura Pintor; Luca Demetrio; Kathrin Grosse; Hsiao-Ying Lin; Chengfang Fang; Battista Biggio; Fabio Roli
>
> **摘要:** Reinforcement learning (RL) enables agents to learn optimal behaviors through interaction with their environment and has been increasingly deployed in safety-critical applications, including autonomous driving. Despite its promise, RL is susceptible to attacks designed either to compromise policy learning or to induce erroneous decisions by trained agents. Although the literature on RL security has grown rapidly and several surveys exist, existing categorizations often fall short in guiding the selection of appropriate defenses for specific systems. In this work, we present a comprehensive survey of 86 recent studies on RL security, addressing these limitations by systematically categorizing attacks and defenses according to defined threat models and single- versus multi-agent settings. Furthermore, we examine the relevance and applicability of state-of-the-art attacks and defense mechanisms within the context of autonomous driving, providing insights to inform the design of robust RL systems.
>
---
#### [replaced 008] Constrained Decoding for Robotics Foundation Models
- **分类: cs.RO; cs.LG; cs.LO**

- **链接: [http://arxiv.org/pdf/2509.01728v2](http://arxiv.org/pdf/2509.01728v2)**

> **作者:** Parv Kapoor; Akila Ganlath; Changliu Liu; Sebastian Scherer; Eunsuk Kang
>
> **摘要:** Recent advances in the development of robotic foundation models have led to promising end-to-end and general-purpose capabilities in robotic systems. These models are pretrained on vast datasets of robot trajectories to process multi-modal inputs and directly output a sequence of action that the system then executes in the real world. Although this approach is attractive from the perspective of improved generalization across diverse tasks, these models are still data-driven and, therefore, lack explicit notions of behavioral correctness and safety constraints. We address these limitations by introducing a constrained decoding framework for robotics foundation models that enforces logical constraints on action trajectories in dynamical systems. Our method ensures that generated actions provably satisfy signal temporal logic (STL) specifications at runtime without retraining, while remaining agnostic of the underlying foundation model. We perform comprehensive evaluation of our approach across state-of-the-art navigation foundation models and we show that our decoding-time interventions are useful not only for filtering unsafe actions but also for conditional action-generation. Videos available on our website: https://constrained-robot-fms.github.io
>
---
#### [replaced 009] SenSnake: A snake robot with contact force sensing for studying locomotion in complex 3-D terrain
- **分类: cs.RO; physics.bio-ph**

- **链接: [http://arxiv.org/pdf/2112.09078v4](http://arxiv.org/pdf/2112.09078v4)**

> **作者:** Divya Ramesh; Qiyuan Fu; Chen Li
>
> **摘要:** Despite advances in a diversity of environments, snake robots are still far behind snakes in traversing complex 3-D terrain with large obstacles. This is due to a lack of understanding of how to control 3-D body bending to push against terrain features to generate and control propulsion. Biological studies suggested that generalist snakes use contact force sensing to adjust body bending in real time to do so. However, studying this sensory-modulated force control in snakes is challenging, due to a lack of basic knowledge of how their force sensing organs work. Here, we take a robophysics approach to make progress, starting by developing a snake robot capable of 3-D body bending with contact force sensing to enable systematic locomotion experiments and force measurements. Through two development and testing iterations, we created a 12-segment robot with 36 piezo-resistive sheet sensors distributed on all segments with compliant shells with a sampling frequency of 30 Hz. The robot measured contact forces while traversing a large obstacle using vertical bending with high repeatability, achieving the goal of providing a platform for systematic experiments. Finally, we explored model-based calibration considering the viscoelastic behavior of the piezo-resistive sensor, which will for useful for future studies.
>
---
#### [replaced 010] GVDepth: Zero-Shot Monocular Depth Estimation for Ground Vehicles based on Probabilistic Cue Fusion
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.06080v2](http://arxiv.org/pdf/2412.06080v2)**

> **作者:** Karlo Koledić; Luka Petrović; Ivan Marković; Ivan Petrović
>
> **备注:** ICCV 2025
>
> **摘要:** Generalizing metric monocular depth estimation presents a significant challenge due to its ill-posed nature, while the entanglement between camera parameters and depth amplifies issues further, hindering multi-dataset training and zero-shot accuracy. This challenge is particularly evident in autonomous vehicles and mobile robotics, where data is collected with fixed camera setups, limiting the geometric diversity. Yet, this context also presents an opportunity: the fixed relationship between the camera and the ground plane imposes additional perspective geometry constraints, enabling depth regression via vertical image positions of objects. However, this cue is highly susceptible to overfitting, thus we propose a novel canonical representation that maintains consistency across varied camera setups, effectively disentangling depth from specific parameters and enhancing generalization across datasets. We also propose a novel architecture that adaptively and probabilistically fuses depths estimated via object size and vertical image position cues. A comprehensive evaluation demonstrates the effectiveness of the proposed approach on five autonomous driving datasets, achieving accurate metric depth estimation for varying resolutions, aspect ratios and camera setups. Notably, we achieve comparable accuracy to existing zero-shot methods, despite training on a single dataset with a single-camera setup. Project website: https://unizgfer-lamor.github.io/gvdepth/
>
---
#### [replaced 011] Pure Vision Language Action (VLA) Models: A Comprehensive Survey
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.19012v2](http://arxiv.org/pdf/2509.19012v2)**

> **作者:** Dapeng Zhang; Jing Sun; Chenghui Hu; Xiaoyan Wu; Zhenlong Yuan; Rui Zhou; Fei Shen; Qingguo Zhou
>
> **摘要:** The emergence of Vision Language Action (VLA) models marks a paradigm shift from traditional policy-based control to generalized robotics, reframing Vision Language Models (VLMs) from passive sequence generators into active agents for manipulation and decision-making in complex, dynamic environments. This survey delves into advanced VLA methods, aiming to provide a clear taxonomy and a systematic, comprehensive review of existing research. It presents a comprehensive analysis of VLA applications across different scenarios and classifies VLA approaches into several paradigms: autoregression-based, diffusion-based, reinforcement-based, hybrid, and specialized methods; while examining their motivations, core strategies, and implementations in detail. In addition, foundational datasets, benchmarks, and simulation platforms are introduced. Building on the current VLA landscape, the review further proposes perspectives on key challenges and future directions to advance research in VLA models and generalizable robotics. By synthesizing insights from over three hundred recent studies, this survey maps the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose VLA methods.
>
---
#### [replaced 012] Growing with Your Embodied Agent: A Human-in-the-Loop Lifelong Code Generation Framework for Long-Horizon Manipulation Skills
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.18597v2](http://arxiv.org/pdf/2509.18597v2)**

> **作者:** Yuan Meng; Zhenguo Sun; Max Fest; Xukun Li; Zhenshan Bing; Alois Knoll
>
> **备注:** update fig 1, typo correction - v2
>
> **摘要:** Large language models (LLMs)-based code generation for robotic manipulation has recently shown promise by directly translating human instructions into executable code, but existing methods remain noisy, constrained by fixed primitives and limited context windows, and struggle with long-horizon tasks. While closed-loop feedback has been explored, corrected knowledge is often stored in improper formats, restricting generalization and causing catastrophic forgetting, which highlights the need for learning reusable skills. Moreover, approaches that rely solely on LLM guidance frequently fail in extremely long-horizon scenarios due to LLMs' limited reasoning capability in the robotic domain, where such issues are often straightforward for humans to identify. To address these challenges, we propose a human-in-the-loop framework that encodes corrections into reusable skills, supported by external memory and Retrieval-Augmented Generation with a hint mechanism for dynamic reuse. Experiments on Ravens, Franka Kitchen, and MetaWorld, as well as real-world settings, show that our framework achieves a 0.93 success rate (up to 27% higher than baselines) and a 42% efficiency improvement in correction rounds. It can robustly solve extremely long-horizon tasks such as "build a house", which requires planning over 20 primitives.
>
---
#### [replaced 013] Streaming Flow Policy: Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21851v2](http://arxiv.org/pdf/2505.21851v2)**

> **作者:** Sunshine Jiang; Xiaolin Fang; Nicholas Roy; Tomás Lozano-Pérez; Leslie Pack Kaelbling; Siddharth Ancha
>
> **备注:** Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Recent advances in diffusion$/$flow-matching policies have enabled imitation learning of complex, multi-modal action trajectories. However, they are computationally expensive because they sample a trajectory of trajectories: a diffusion$/$flow trajectory of action trajectories. They discard intermediate action trajectories, and must wait for the sampling process to complete before any actions can be executed on the robot. We simplify diffusion$/$flow policies by treating action trajectories as flow trajectories. Instead of starting from pure noise, our algorithm samples from a narrow Gaussian around the last action. Then, it incrementally integrates a velocity field learned via flow matching to produce a sequence of actions that constitute a single trajectory. This enables actions to be streamed to the robot on-the-fly during the flow sampling process, and is well-suited for receding horizon policy execution. Despite streaming, our method retains the ability to model multi-modal behavior. We train flows that stabilize around demonstration trajectories to reduce distribution shift and improve imitation learning performance. Streaming flow policy outperforms prior methods while enabling faster policy execution and tighter sensorimotor loops for learning-based robot control. Project website: https://streaming-flow-policy.github.io/
>
---
#### [replaced 014] Occlusion-Aware Consistent Model Predictive Control for Robot Navigation in Occluded Obstacle-Dense Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.04563v4](http://arxiv.org/pdf/2503.04563v4)**

> **作者:** Minzhe Zheng; Lei Zheng; Lei Zhu; Jun Ma
>
> **摘要:** Ensuring safety and motion consistency for robot navigation in occluded, obstacle-dense environments is a critical challenge. In this context, this study presents an occlusion-aware Consistent Model Predictive Control (CMPC) strategy. To account for the occluded obstacles, it incorporates adjustable risk regions that represent their potential future locations. Subsequently, dynamic risk boundary constraints are developed online to ensure safety.The CMPC then constructs multiple locally optimal trajectory branches (each tailored to different risk regions) to strike a balance between safety and performance. A shared consensus segment is generated to ensure smooth transitions between branches without significant velocity fluctuations, further preserving motion consistency. To facilitate high computational efficiency and ensure coordination across local trajectories, we use the alternating direction method of multipliers (ADMM) to decompose the CMPC into manageable sub-problems for parallel solving. The proposed strategy is validated through simulations and real-world experiments on an Ackermann-steering robot platform. The results demonstrate the effectiveness of the proposed CMPC strategy through comparisons with baseline approaches in occluded, obstacle-dense environments.
>
---
#### [replaced 015] GAF: Gaussian Action Field as a 4D Representation for Dynamic World Modeling in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.14135v4](http://arxiv.org/pdf/2506.14135v4)**

> **作者:** Ying Chai; Litao Deng; Ruizhi Shao; Jiajun Zhang; Kangchen Lv; Liangjun Xing; Xiang Li; Hongwen Zhang; Yebin Liu
>
> **备注:** http://chaiying1.github.io/GAF.github.io/project_page/
>
> **摘要:** Accurate scene perception is critical for vision-based robotic manipulation. Existing approaches typically follow either a Vision-to-Action (V-A) paradigm, predicting actions directly from visual inputs, or a Vision-to-3D-to-Action (V-3D-A) paradigm, leveraging intermediate 3D representations. However, these methods often struggle with action inaccuracies due to the complexity and dynamic nature of manipulation scenes. In this paper, we adopt a V-4D-A framework that enables direct action reasoning from motion-aware 4D representations via a Gaussian Action Field (GAF). GAF extends 3D Gaussian Splatting (3DGS) by incorporating learnable motion attributes, allowing 4D modeling of dynamic scenes and manipulation actions. To learn time-varying scene geometry and action-aware robot motion, GAF provides three interrelated outputs: reconstruction of the current scene, prediction of future frames, and estimation of init action via Gaussian motion. Furthermore, we employ an action-vision-aligned denoising framework, conditioned on a unified representation that combines the init action and the Gaussian perception, both generated by the GAF, to further obtain more precise actions. Extensive experiments demonstrate significant improvements, with GAF achieving +11.5385 dB PSNR, +0.3864 SSIM and -0.5574 LPIPS improvements in reconstruction quality, while boosting the average +7.3% success rate in robotic manipulation tasks over state-of-the-art methods.
>
---
#### [replaced 016] Label-Efficient Grasp Joint Prediction with Point-JEPA
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13349v2](http://arxiv.org/pdf/2509.13349v2)**

> **作者:** Jed Guzelkabaagac; Boris Petrović
>
> **备注:** 4 pages, 5 figures. Submitted to IROS 2025 Workshop
>
> **摘要:** We study whether 3D self-supervised pretraining with Point--JEPA enables label-efficient grasp joint-angle prediction. Meshes are sampled to point clouds and tokenized; a ShapeNet-pretrained Point--JEPA encoder feeds a $K{=}5$ multi-hypothesis head trained with winner-takes-all and evaluated by top--logit selection. On a multi-finger hand dataset with strict object-level splits, Point--JEPA improves top--logit RMSE and Coverage@15$^{\circ}$ in low-label regimes (e.g., 26% lower RMSE at 25% data) and reaches parity at full supervision, suggesting JEPA-style pretraining is a practical lever for data-efficient grasp learning.
>
---
#### [replaced 017] OpenGVL - Benchmarking Visual Temporal Progress for Data Curation
- **分类: cs.RO; cs.CL**

- **链接: [http://arxiv.org/pdf/2509.17321v2](http://arxiv.org/pdf/2509.17321v2)**

> **作者:** Paweł Budzianowski; Emilia Wiśnios; Gracjan Góral; Igor Kulakov; Viktor Petrenko; Krzysztof Walas
>
> **摘要:** Data scarcity remains one of the most limiting factors in driving progress in robotics. However, the amount of available robotics data in the wild is growing exponentially, creating new opportunities for large-scale data utilization. Reliable temporal task completion prediction could help automatically annotate and curate this data at scale. The Generative Value Learning (GVL) approach was recently proposed, leveraging the knowledge embedded in vision-language models (VLMs) to predict task progress from visual observations. Building upon GVL, we propose OpenGVL, a comprehensive benchmark for estimating task progress across diverse challenging manipulation tasks involving both robotic and human embodiments. We evaluate the capabilities of publicly available open-source foundation models, showing that open-source model families significantly underperform closed-source counterparts, achieving only approximately $70\%$ of their performance on temporal progress prediction tasks. Furthermore, we demonstrate how OpenGVL can serve as a practical tool for automated data curation and filtering, enabling efficient quality assessment of large-scale robotics datasets. We release the benchmark along with the complete codebase at \href{github.com/budzianowski/opengvl}{OpenGVL}.
>
---
#### [replaced 018] Omni-Roach: A legged robot capable of traversing multiple types of large obstacles and self-righting
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2112.10614v4](http://arxiv.org/pdf/2112.10614v4)**

> **作者:** Jonathan Mi; Yaqing Wang; Chen Li
>
> **摘要:** Robots excel at avoiding obstacles but struggle to traverse complex 3-D terrain with cluttered large obstacles. By contrast, insects like cockroaches excel at doing so. Recent research in our lab elucidated how locomotor transitions emerge from locomotor-environment interaction for diverse locomotor challenges abstracted from complex 3-D terrain and the strategies to overcome them. Here we built on these fundamental insights to develop a cockroach-inspired legged robot, Omni-Roach, that integrated these strategies to achieve multi-modal locomotion and provide a robophysical model to study the trade-off between multi-functionality and performance. The robot was based on the RHex design with six compliant legs and featured a rounded body with two wings that can open and a tail with pitch and yaw degrees of freedom. After two development and testing iterations, our robot was capable of overcoming all locomotor challenges with a high performance and success rate. It traversed cluttered rigid pillars only 1.1x robot body width apart, a 2.5x hip height bump, a 0.75x body length gap, densely cluttered flexible beams only 65% body width apart, and self-righted within 4 seconds. Systematic beam traversal experiments further revealed that a downward-pointing tail oscillating laterally helps roll the body into beam gaps and break frictional and interlocking contact to traverse. Our work highlights the usefulness of multi-functional appendages and exaptation for large obstacle traversal.
>
---
#### [replaced 019] DyDexHandover: Human-like Bimanual Dynamic Dexterous Handover using RGB-only Perception
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.17350v2](http://arxiv.org/pdf/2509.17350v2)**

> **作者:** Haoran Zhou; Yangwei You; Shuaijun Wang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Dynamic in air handover is a fundamental challenge for dual-arm robots, requiring accurate perception, precise coordination, and natural motion. Prior methods often rely on dynamics models, strong priors, or depth sensing, limiting generalization and naturalness. We present DyDexHandover, a novel framework that employs multi-agent reinforcement learning to train an end to end RGB based policy for bimanual object throwing and catching. To achieve more human-like behavior, the throwing policy is guided by a human policy regularization scheme, encouraging fluid and natural motion, and enhancing the generalization capability of the policy. A dual arm simulation environment was built in Isaac Sim for experimental evaluation. DyDexHandover achieves nearly 99 percent success on training objects and 75 percent on unseen objects, while generating human-like throwing and catching behaviors. To our knowledge, it is the first method to realize dual-arm in-air handover using only raw RGB perception.
>
---
#### [replaced 020] Systematic Constraint Formulation and Collision-Free Trajectory Planning Using Space-Time Graphs of Convex Sets
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2508.10203v2](http://arxiv.org/pdf/2508.10203v2)**

> **作者:** Matthew D. Osburn; Cameron K. Peterson; John L. Salmon
>
> **备注:** 16 pages with references, 20 figures
>
> **摘要:** In this paper, we create optimal, collision-free, time-dependent trajectories through cluttered dynamic environments. The many spatial and temporal constraints make finding an initial guess for a numerical solver difficult. Graphs of Convex Sets (GCS) and the recently developed Space-Time Graphs of Convex Sets (ST-GCS) enable us to generate minimum distance collision-free trajectories without providing an initial guess to the solver. We also explore the derivation of general GCS-compatible constraints and document an intuitive strategy for adapting general constraints to the framework. We show that ST-GCS produces equivalent trajectories to the standard GCS formulation when the environment is static, as well as globally optimal trajectories in cluttered dynamic environments.
>
---
#### [replaced 021] Real-Time Out-of-Distribution Failure Prevention via Multi-Modal Reasoning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.10547v2](http://arxiv.org/pdf/2505.10547v2)**

> **作者:** Milan Ganai; Rohan Sinha; Christopher Agia; Daniel Morton; Luigi Di Lillo; Marco Pavone
>
> **备注:** Conference on Robot Learning (CoRL) 2025 (Oral)
>
> **摘要:** While foundation models offer promise toward improving robot safety in out-of-distribution (OOD) scenarios, how to effectively harness their generalist knowledge for real-time, dynamically feasible response remains a crucial problem. We present FORTRESS, a joint reasoning and planning framework that generates semantically safe fallback strategies to prevent safety-critical, OOD failures. At a low frequency under nominal operation, FORTRESS uses multi-modal foundation models to anticipate possible failure modes and identify safe fallback sets. When a runtime monitor triggers a fallback response, FORTRESS rapidly synthesizes plans to fallback goals while inferring and avoiding semantically unsafe regions in real time. By bridging open-world, multi-modal reasoning with dynamics-aware planning, we eliminate the need for hard-coded fallbacks and human safety interventions. FORTRESS outperforms on-the-fly prompting of slow reasoning models in safety classification accuracy on synthetic benchmarks and real-world ANYmal robot data, and further improves system safety and planning success in simulation and on quadrotor hardware for urban navigation. Website can be found at https://milanganai.github.io/fortress.
>
---
#### [replaced 022] Hyperspectral Adapter for Semantic Segmentation with Vision Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20107v2](http://arxiv.org/pdf/2509.20107v2)**

> **作者:** Juana Valeria Hurtado; Rohit Mohan; Abhinav Valada
>
> **摘要:** Hyperspectral imaging (HSI) captures spatial information along with dense spectral measurements across numerous narrow wavelength bands. This rich spectral content has the potential to facilitate robust robotic perception, particularly in environments with complex material compositions, varying illumination, or other visually challenging conditions. However, current HSI semantic segmentation methods underperform due to their reliance on architectures and learning frameworks optimized for RGB inputs. In this work, we propose a novel hyperspectral adapter that leverages pretrained vision foundation models to effectively learn from hyperspectral data. Our architecture incorporates a spectral transformer and a spectrum-aware spatial prior module to extract rich spatial-spectral features. Additionally, we introduce a modality-aware interaction block that facilitates effective integration of hyperspectral representations and frozen vision Transformer features through dedicated extraction and injection mechanisms. Extensive evaluations on three benchmark autonomous driving datasets demonstrate that our architecture achieves state-of-the-art semantic segmentation performance while directly using HSI inputs, outperforming both vision-based and hyperspectral segmentation methods. We make the code available at https://hsi-adapter.cs.uni-freiburg.de.
>
---
#### [replaced 023] Contact-Aware Safety in Soft Robots Using High-Order Control Barrier and Lyapunov Functions
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.03841v2](http://arxiv.org/pdf/2505.03841v2)**

> **作者:** Kiwan Wong; Maximilian Stölzle; Wei Xiao; Cosimo Della Santina; Daniela Rus; Gioele Zardini
>
> **备注:** 8 pages
>
> **摘要:** Robots operating alongside people, particularly in sensitive scenarios such as aiding the elderly with daily tasks or collaborating with workers in manufacturing, must guarantee safety and cultivate user trust. Continuum soft manipulators promise safety through material compliance, but as designs evolve for greater precision, payload capacity, and speed, and increasingly incorporate rigid elements, their injury risk resurfaces. In this letter, we introduce a comprehensive High-Order Control Barrier Function (HOCBF) + High-Order Control Lyapunov Function (HOCLF) framework that enforces strict contact force limits across the entire soft-robot body during environmental interactions. Our approach combines a differentiable Piecewise Cosserat-Segment (PCS) dynamics model with a convex-polygon distance approximation metric, named Differentiable Conservative Separating Axis Theorem (DCSAT), based on the soft robot geometry to enable real-time, whole-body collision detection, resolution, and enforcement of the safety constraints. By embedding HOCBFs into our optimization routine, we guarantee safety, allowing, for instance, safe navigation in operational space under HOCLF-driven motion objectives. Extensive planar simulations demonstrate that our method maintains safety-bounded contacts while achieving precise shape and task-space regulation. This work thus lays a foundation for the deployment of soft robots in human-centric environments with provable safety and performance.
>
---
#### [replaced 024] Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.12846v2](http://arxiv.org/pdf/2507.12846v2)**

> **作者:** Muhammad Fadhil Ginting; Dong-Ki Kim; Xiangyun Meng; Andrzej Reinke; Bandi Jai Krishna; Navid Kayhani; Oriana Peltzer; David D. Fan; Amirreza Shaban; Sung-Kyun Kim; Mykel J. Kochenderfer; Ali-akbar Agha-mohammadi; Shayegan Omidshafiei
>
> **摘要:** As robots become increasingly capable of operating over extended periods -- spanning days, weeks, and even months -- they are expected to accumulate knowledge of their environments and leverage this experience to assist humans more effectively. This paper studies the problem of Long-term Active Embodied Question Answering (LA-EQA), a new task in which a robot must both recall past experiences and actively explore its environment to answer complex, temporally-grounded questions. Unlike traditional EQA settings, which typically focus either on understanding the present environment alone or on recalling a single past observation, LA-EQA challenges an agent to reason over past, present, and possible future states, deciding when to explore, when to consult its memory, and when to stop gathering observations and provide a final answer. Standard EQA approaches based on large models struggle in this setting due to limited context windows, absence of persistent memory, and an inability to combine memory recall with active exploration. To address this, we propose a structured memory system for robots, inspired by the mind palace method from cognitive science. Our method encodes episodic experiences as scene-graph-based world instances, forming a reasoning and planning algorithm that enables targeted memory retrieval and guided navigation. To balance the exploration-recall trade-off, we introduce value-of-information-based stopping criteria that determines when the agent has gathered sufficient information. We evaluate our method on real-world experiments and introduce a new benchmark that spans popular simulation environments and actual industrial sites. Our approach significantly outperforms state-of-the-art baselines, yielding substantial gains in both answer accuracy and exploration efficiency.
>
---
#### [replaced 025] Online Language Splatting
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.09447v3](http://arxiv.org/pdf/2503.09447v3)**

> **作者:** Saimouli Katragadda; Cho-Ying Wu; Yuliang Guo; Xinyu Huang; Guoquan Huang; Liu Ren
>
> **摘要:** To enable AI agents to interact seamlessly with both humans and 3D environments, they must not only perceive the 3D world accurately but also align human language with 3D spatial representations. While prior work has made significant progress by integrating language features into geometrically detailed 3D scene representations using 3D Gaussian Splatting (GS), these approaches rely on computationally intensive offline preprocessing of language features for each input image, limiting adaptability to new environments. In this work, we introduce Online Language Splatting, the first framework to achieve online, near real-time, open-vocabulary language mapping within a 3DGS-SLAM system without requiring pre-generated language features. The key challenge lies in efficiently fusing high-dimensional language features into 3D representations while balancing the computation speed, memory usage, rendering quality and open-vocabulary capability. To this end, we innovatively design: (1) a high-resolution CLIP embedding module capable of generating detailed language feature maps in 18ms per frame, (2) a two-stage online auto-encoder that compresses 768-dimensional CLIP features to 15 dimensions while preserving open-vocabulary capabilities, and (3) a color-language disentangled optimization approach to improve rendering quality. Experimental results show that our online method not only surpasses the state-of-the-art offline methods in accuracy but also achieves more than 40x efficiency boost, demonstrating the potential for dynamic and interactive AI applications.
>
---
#### [replaced 026] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18631v2](http://arxiv.org/pdf/2509.18631v2)**

> **作者:** Shuo Cheng; Liqian Ma; Zhenyang Chen; Ajay Mandlekar; Caelan Garrett; Danfei Xu
>
> **摘要:** Behavior cloning has shown promise for robot manipulation, but real-world demonstrations are costly to acquire at scale. While simulated data offers a scalable alternative, particularly with advances in automated demonstration generation, transferring policies to the real world is hampered by various simulation and real domain gaps. In this work, we propose a unified sim-and-real co-training framework for learning generalizable manipulation policies that primarily leverages simulation and only requires a few real-world demonstrations. Central to our approach is learning a domain-invariant, task-relevant feature space. Our key insight is that aligning the joint distributions of observations and their corresponding actions across domains provides a richer signal than aligning observations (marginals) alone. We achieve this by embedding an Optimal Transport (OT)-inspired loss within the co-training framework, and extend this to an Unbalanced OT framework to handle the imbalance between abundant simulation data and limited real-world examples. We validate our method on challenging manipulation tasks, showing it can leverage abundant simulation data to achieve up to a 30% improvement in the real-world success rate and even generalize to scenarios seen only in simulation.
>
---
#### [replaced 027] Model Agnostic Defense against Adversarial Patch Attacks on Object Detection in Unmanned Aerial Vehicles
- **分类: cs.CV; cs.RO; I.4.4; I.4.9**

- **链接: [http://arxiv.org/pdf/2405.19179v2](http://arxiv.org/pdf/2405.19179v2)**

> **作者:** Saurabh Pathak; Samridha Shrestha; Abdelrahman AlMahmoud
>
> **备注:** published in IROS 2024
>
> **摘要:** Object detection forms a key component in Unmanned Aerial Vehicles (UAVs) for completing high-level tasks that depend on the awareness of objects on the ground from an aerial perspective. In that scenario, adversarial patch attacks on an onboard object detector can severely impair the performance of upstream tasks. This paper proposes a novel model-agnostic defense mechanism against the threat of adversarial patch attacks in the context of UAV-based object detection. We formulate adversarial patch defense as an occlusion removal task. The proposed defense method can neutralize adversarial patches located on objects of interest, without exposure to adversarial patches during training. Our lightweight single-stage defense approach allows us to maintain a model-agnostic nature, that once deployed does not require to be updated in response to changes in the object detection pipeline. The evaluations in digital and physical domains show the feasibility of our method for deployment in UAV object detection pipelines, by significantly decreasing the Attack Success Ratio without incurring significant processing costs. As a result, the proposed defense solution can improve the reliability of object detection for UAVs.
>
---
#### [replaced 028] AnyPlace: Learning Generalized Object Placement for Robot Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.04531v2](http://arxiv.org/pdf/2502.04531v2)**

> **作者:** Yuchi Zhao; Miroslav Bogdanovic; Chengyuan Luo; Steven Tohme; Kourosh Darvish; Alán Aspuru-Guzik; Florian Shkurti; Animesh Garg
>
> **备注:** Accepted at CoRL 2025
>
> **摘要:** Object placement in robotic tasks is inherently challenging due to the diversity of object geometries and placement configurations. To address this, we propose AnyPlace, a two-stage method trained entirely on synthetic data, capable of predicting a wide range of feasible placement poses for real-world tasks. Our key insight is that by leveraging a Vision-Language Model (VLM) to identify rough placement locations, we focus only on the relevant regions for local placement, which enables us to train the low-level placement-pose-prediction model to capture diverse placements efficiently. For training, we generate a fully synthetic dataset of randomly generated objects in different placement configurations (insertion, stacking, hanging) and train local placement-prediction models. We conduct extensive evaluations in simulation, demonstrating that our method outperforms baselines in terms of success rate, coverage of possible placement modes, and precision. In real-world experiments, we show how our approach directly transfers models trained purely on synthetic data to the real world, where it successfully performs placements in scenarios where other models struggle -- such as with varying object geometries, diverse placement modes, and achieving high precision for fine placement. More at: https://any-place.github.io.
>
---
#### [replaced 029] HUNT: High-Speed UAV Navigation and Tracking in Unstructured Environments via Instantaneous Relative Frames
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.19452v2](http://arxiv.org/pdf/2509.19452v2)**

> **作者:** Alessandro Saviolo; Jeffrey Mao; Giuseppe Loianno
>
> **摘要:** Search and rescue operations require unmanned aerial vehicles to both traverse unknown unstructured environments at high speed and track targets once detected. Achieving both capabilities under degraded sensing and without global localization remains an open challenge. Recent works on relative navigation have shown robust tracking by anchoring planning and control to a visible detected object, but cannot address navigation when no target is in the field of view. We present HUNT (High-speed UAV Navigation and Tracking), a real-time framework that unifies traversal, acquisition, and tracking within a single relative formulation. HUNT defines navigation objectives directly from onboard instantaneous observables such as attitude, altitude, and velocity, enabling reactive high-speed flight during search. Once a target is detected, the same perception-control pipeline transitions seamlessly to tracking. Outdoor experiments in dense forests, container compounds, and search-and-rescue operations with vehicles and mannequins demonstrate robust autonomy where global methods fail.
>
---
#### [replaced 030] Stairway to Success: An Online Floor-Aware Zero-Shot Object-Goal Navigation Framework via LLM-Driven Coarse-to-Fine Exploration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.23019v3](http://arxiv.org/pdf/2505.23019v3)**

> **作者:** Zeying Gong; Rong Li; Tianshuai Hu; Ronghe Qiu; Lingdong Kong; Lingfeng Zhang; Guoyang Zhao; Yiyi Ding; Junwei Liang
>
> **备注:** Preprint; 9 pages, 9 figures, 8 tables; Project Page at https://zeying-gong.github.io/projects/ascent
>
> **摘要:** Deployable service and delivery robots struggle to navigate multi-floor buildings to reach object goals, as existing systems fail due to single-floor assumptions and requirements for offline, globally consistent maps. Multi-floor environments pose unique challenges including cross-floor transitions and vertical spatial reasoning, especially navigating unknown buildings. Object-Goal Navigation benchmarks like HM3D and MP3D also capture this multi-floor reality, yet current methods lack support for online, floor-aware navigation. To bridge this gap, we propose \textbf{\textit{ASCENT}}, an online framework for Zero-Shot Object-Goal Navigation that enables robots to operate without pre-built maps or retraining on new object categories. It introduces: (1) a \textbf{Multi-Floor Abstraction} module that dynamically constructs hierarchical representations with stair-aware obstacle mapping and cross-floor topology modeling, and (2) a \textbf{Coarse-to-Fine Reasoning} module that combines frontier ranking with LLM-driven contextual analysis for multi-floor navigation decisions. We evaluate on HM3D and MP3D benchmarks, outperforming state-of-the-art zero-shot approaches, and demonstrate real-world deployment on a quadruped robot.
>
---
#### [replaced 031] LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.13751v3](http://arxiv.org/pdf/2506.13751v3)**

> **作者:** Haoru Xue; Xiaoyu Huang; Dantong Niu; Qiayuan Liao; Thomas Kragerud; Jan Tommy Gravdahl; Xue Bin Peng; Guanya Shi; Trevor Darrell; Koushil Sreenath; Shankar Sastry
>
> **备注:** https://ember-lab-berkeley.github.io/LeVERB-Website/
>
> **摘要:** Vision-language-action (VLA) models have demonstrated strong semantic understanding and zero-shot generalization, yet most existing systems assume an accurate low-level controller with hand-crafted action "vocabulary" such as end-effector pose or root velocity. This assumption confines prior work to quasi-static tasks and precludes the agile, whole-body behaviors required by humanoid whole-body control (WBC) tasks. To capture this gap in the literature, we start by introducing the first sim-to-real-ready, vision-language, closed-loop benchmark for humanoid WBC, comprising over 150 tasks from 10 categories. We then propose LeVERB: Latent Vision-Language-Encoded Robot Behavior, a hierarchical latent instruction-following framework for humanoid vision-language WBC, the first of its kind. At the top level, a vision-language policy learns a latent action vocabulary from synthetically rendered kinematic demonstrations; at the low level, a reinforcement-learned WBC policy consumes these latent verbs to generate dynamics-level commands. In our benchmark, LeVERB can zero-shot attain a 80% success rate on simple visual navigation tasks, and 58.5% success rate overall, outperforming naive hierarchical whole-body VLA implementation by 7.8 times.
>
---
