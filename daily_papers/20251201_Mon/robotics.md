# 机器人 cs.RO

- **最新发布 59 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] Obstruction reasoning for robotic grasping
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对机器人在杂乱环境中抓取目标物体时的遮挡推理问题，提出UNOGrasp模型。通过视觉-语言多步推理，识别遮挡路径并规划清除顺序，结合监督与强化学习优化决策。构建了包含超10万条遮挡路径的UNOBench数据集，显著提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2511.23186v1](https://arxiv.org/pdf/2511.23186v1)**

> **作者:** Runyu Jiao; Matteo Bortolon; Francesco Giuliari; Alice Fasoli; Sergio Povoli; Guofeng Mei; Yiming Wang; Fabio Poiesi
>
> **摘要:** Successful robotic grasping in cluttered environments not only requires a model to visually ground a target object but also to reason about obstructions that must be cleared beforehand. While current vision-language embodied reasoning models show emergent spatial understanding, they remain limited in terms of obstruction reasoning and accessibility planning. To bridge this gap, we present UNOGrasp, a learning-based vision-language model capable of performing visually-grounded obstruction reasoning to infer the sequence of actions needed to unobstruct the path and grasp the target object. We devise a novel multi-step reasoning process based on obstruction paths originated by the target object. We anchor each reasoning step with obstruction-aware visual cues to incentivize reasoning capability. UNOGrasp combines supervised and reinforcement finetuning through verifiable reasoning rewards. Moreover, we construct UNOBench, a large-scale dataset for both training and benchmarking, based on MetaGraspNetV2, with over 100k obstruction paths annotated by humans with obstruction ratios, contact points, and natural-language instructions. Extensive experiments and real-robot evaluations show that UNOGrasp significantly improves obstruction reasoning and grasp success across both synthetic and real-world environments, outperforming generalist and proprietary alternatives. Project website: https://tev-fbk.github.io/UnoGrasp/.
>
---
#### [new 002] From CAD to POMDP: Probabilistic Planning for Robotic Disassembly of End-of-Life Products
- **分类: cs.RO**

- **简介: 该论文研究机器人对报废产品的概率性拆解任务。针对真实产品与设计模型偏差导致的不确定性，提出基于POMDP的规划框架，从CAD数据自动生成可执行策略，结合强化学习与贝叶斯滤波，实现对隐藏状态的在线推断与适应，显著提升拆解效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.23407v1](https://arxiv.org/pdf/2511.23407v1)**

> **作者:** Jan Baumgärtner; Malte Hansjosten; David Hald; Adrian Hauptmannl; Alexander Puchta; Jürgen Fleischer
>
> **摘要:** To support the circular economy, robotic systems must not only assemble new products but also disassemble end-of-life (EOL) ones for reuse, recycling, or safe disposal. Existing approaches to disassembly sequence planning often assume deterministic and fully observable product models, yet real EOL products frequently deviate from their initial designs due to wear, corrosion, or undocumented repairs. We argue that disassembly should therefore be formulated as a Partially Observable Markov Decision Process (POMDP), which naturally captures uncertainty about the product's internal state. We present a mathematical formulation of disassembly as a POMDP, in which hidden variables represent uncertain structural or physical properties. Building on this formulation, we propose a task and motion planning framework that automatically derives specific POMDP models from CAD data, robot capabilities, and inspection results. To obtain tractable policies, we approximate this formulation with a reinforcement-learning approach that operates on stochastic action outcomes informed by inspection priors, while a Bayesian filter continuously maintains beliefs over latent EOL conditions during execution. Using three products on two robotic systems, we demonstrate that this probabilistic planning framework outperforms deterministic baselines in terms of average disassembly time and variance, generalizes across different robot setups, and successfully adapts to deviations from the CAD model, such as missing or stuck parts.
>
---
#### [new 003] Beyond Egocentric Limits: Multi-View Depth-Based Learning for Robust Quadrupedal Locomotion
- **分类: cs.RO**

- **简介: 该论文研究四足机器人敏捷行走任务，针对单一视角感知导致的环境认知局限问题，提出多视角深度学习框架。通过融合本体与外部视角信息，结合教师-学生蒸馏与域随机化，提升对遮挡和传感器失效的鲁棒性，显著增强复杂地形下的运动能力。**

- **链接: [https://arxiv.org/pdf/2511.22744v1](https://arxiv.org/pdf/2511.22744v1)**

> **作者:** Rémy Rahem; Wael Suleiman
>
> **备注:** 12 pages, 6 figures, code available at https://anonymous.4open.science/r/multiview-parkour-6FB8
>
> **摘要:** Recent progress in legged locomotion has allowed highly dynamic and parkour-like behaviors for robots, similar to their biological counterparts. Yet, these methods mostly rely on egocentric (first-person) perception, limiting their performance, especially when the viewpoint of the robot is occluded. A promising solution would be to enhance the robot's environmental awareness by using complementary viewpoints, such as multiple actors exchanging perceptual information. Inspired by this idea, this work proposes a multi-view depth-based locomotion framework that combines egocentric and exocentric observations to provide richer environmental context during agile locomotion. Using a teacher-student distillation approach, the student policy learns to fuse proprioception with dual depth streams while remaining robust to real-world sensing imperfections. To further improve robustness, we introduce extensive domain randomization, including stochastic remote-camera dropouts and 3D positional perturbations that emulate aerial-ground cooperative sensing. Simulation results show that multi-viewpoints policies outperform single-viewpoint baseline in gap crossing, step descent, and other dynamic maneuvers, while maintaining stability when the exocentric camera is partially or completely unavailable. Additional experiments show that moderate viewpoint misalignment is well tolerated when incorporated during training. This study demonstrates that heterogeneous visual feedback improves robustness and agility in quadrupedal locomotion. Furthermore, to support reproducibility, the implementation accompanying this work is publicly available at https://anonymous.4open.science/r/multiview-parkour-6FB8
>
---
#### [new 004] Design, modelling and experimental validation of bipenniform shape memory alloy-based linear actuator integrable with hydraulic stroke amplification mechanism
- **分类: cs.RO**

- **简介: 该论文针对传统电磁驱动器效率低、体积大、成本高等问题，提出一种仿生双羽状形状记忆合金（SMA）线性执行器，结合液压伸缩放大机制。通过建模与实验验证，实现高力输出（257 N）、轻量化（减重67%）、低成本（降32%）及节能（省19%），适用于建筑自动化、航天机器人及医疗假肢等场景。**

- **链接: [https://arxiv.org/pdf/2511.23372v1](https://arxiv.org/pdf/2511.23372v1)**

> **作者:** Kanhaiya Lal Chaurasiya; Ruchira Kumar Pradhan; Yashaswi Sinha; Shivam Gupta; Ujjain Kumar Bidila; Digambar Killedar; Kapil Das Sahu; Bishakh Bhattacharya
>
> **摘要:** The increasing industrial demand for alternative actuators over conventional electromagnetism-based systems having limited efficiency, bulky size, complex design due to in-built gear-train mechanisms, and high production and amortization costs necessitates the innovation in new actuator development. Integrating bio-inspired design principles into linear actuators could bring forth the next generation of adaptive and energy efficient smart material-based actuation systems. The present study amalgamates the advantages of bipenniform architecture, which generates high force in the given physiological region and a high power-to-weight ratio of shape memory alloy (SMA), into a novel bio-inspired SMA-based linear actuator. A mathematical model of a multi-layered bipenniform configuration-based SMA actuator was developed and validated experimentally. The current research also caters to the incorporation of failure mitigation strategies using design failure mode and effects analysis along with the experimental assessment of the performance of the developed actuator. The system has been benchmarked against an industry-developed stepper motor-driven actuator. It has shown promising results generating an actuation force of 257 N with 15 V input voltage, meeting the acceptable range for actuation operation. It further exhibits about 67% reduction in the weight of the drive mechanism, with 80% lesser component, 32% cost reduction, and 19% energy savings and similar envelope dimensions for assembly compatibility with dampers and louvers for easy onsite deployment. The study introduces SMA coil-based actuator as an advanced design that can be deployed for high force-high stroke applications. The bio-inspired SMA-based linear actuator has applications ranging from building automation controls to lightweight actuation systems for space robotics and medical prosthesis.
>
---
#### [new 005] Commanding Humanoid by Free-form Language: A Large Language Action Model with Unified Motion Vocabulary
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对人形机器人语言指令理解与动作执行任务，解决自由语义指令下全身动作生成的多样性与物理合理性难题。提出Humanoid-LLA模型，通过统一运动词汇、可执行控制器及物理感知微调，实现自然语言到真实物理动作的高效映射，显著提升动作自然性、稳定性和成功率。**

- **链接: [https://arxiv.org/pdf/2511.22963v1](https://arxiv.org/pdf/2511.22963v1)**

> **作者:** Zhirui Liu; Kaiyang Ji; Ke Yang; Jingyi Yu; Ye Shi; Jingya Wang
>
> **备注:** Project page: https://humanoidlla.github.io/
>
> **摘要:** Enabling humanoid robots to follow free-form language commands is critical for seamless human-robot interaction, collaborative task execution, and general-purpose embodied intelligence. While recent advances have improved low-level humanoid locomotion and robot manipulation, language-conditioned whole-body control remains a significant challenge. Existing methods are often limited to simple instructions and sacrifice either motion diversity or physical plausibility. To address this, we introduce Humanoid-LLA, a Large Language Action Model that maps expressive language commands to physically executable whole-body actions for humanoid robots. Our approach integrates three core components: a unified motion vocabulary that aligns human and humanoid motion primitives into a shared discrete space; a vocabulary-directed controller distilled from a privileged policy to ensure physical feasibility; and a physics-informed fine-tuning stage using reinforcement learning with dynamics-aware rewards to enhance robustness and stability. Extensive evaluations in simulation and on a real-world Unitree G1 humanoid show that Humanoid-LLA delivers strong language generalization while maintaining high physical fidelity, outperforming existing language-conditioned controllers in motion naturalness, stability, and execution success rate.
>
---
#### [new 006] Threat-Aware UAV Dodging of Human-Thrown Projectiles with an RGB-D Camera
- **分类: cs.RO**

- **简介: 该论文针对无人机在执行任务时面临人类投掷物攻击的威胁，提出基于RGB-D相机的实时避障系统。通过融合人体姿态与深度信息预测攻击轨迹，并设计不确定性感知的避让策略，实现低延迟、高鲁棒性的敏捷躲避，有效提升无人机安全性。**

- **链接: [https://arxiv.org/pdf/2511.22847v1](https://arxiv.org/pdf/2511.22847v1)**

> **作者:** Yuying Zhang; Na Fan; Haowen Zheng; Junning Liang; Zongliang Pan; Qifeng Chen; Ximin Lyu
>
> **摘要:** Uncrewed aerial vehicles (UAVs) performing tasks such as transportation and aerial photography are vulnerable to intentional projectile attacks from humans. Dodging such a sudden and fast projectile poses a significant challenge for UAVs, requiring ultra-low latency responses and agile maneuvers. Drawing inspiration from baseball, in which pitchers' body movements are analyzed to predict the ball's trajectory, we propose a novel real-time dodging system that leverages an RGB-D camera. Our approach integrates human pose estimation with depth information to predict the attacker's motion trajectory and the subsequent projectile trajectory. Additionally, we introduce an uncertainty-aware dodging strategy to enable the UAV to dodge incoming projectiles efficiently. Our perception system achieves high prediction accuracy and outperforms the baseline in effective distance and latency. The dodging strategy addresses temporal and spatial uncertainties to ensure UAV safety. Extensive real-world experiments demonstrate the framework's reliable dodging capabilities against sudden attacks and its outstanding robustness across diverse scenarios.
>
---
#### [new 007] SoftNash: Entropy-Regularized Nash Games for Non-Fighting Virtual Fixtures
- **分类: cs.RO; cs.HC**

- **简介: 该论文针对遥操作中虚拟夹具（VF）与用户对抗导致的负担加重和控制感下降问题，提出Soft-Nash虚拟夹具。通过引入可解释参数τ调节控制器强硬程度，实现从硬性约束到柔和协作的连续过渡，在保持精度的同时显著降低冲突、提升用户感知控制感，验证了其在人机协作中的有效性。**

- **链接: [https://arxiv.org/pdf/2511.22087v1](https://arxiv.org/pdf/2511.22087v1)**

> **作者:** Tai Inui; Jee-Hwan Ryu
>
> **摘要:** Virtual fixtures (VFs) improve precision in teleoperation but often ``fight'' the user, inflating mental workload and eroding the sense of agency. We propose Soft-Nash Virtual Fixtures, a game-theoretic shared-control policy that softens the classic two-player linear-quadratic (LQ) Nash solution by inflating the fixture's effort weight with a single, interpretable scalar parameter $τ$. This yields a continuous dial on controller assertiveness: $τ=0$ recovers a hard, performance-focused Nash / virtual fixture controller, while larger $τ$ reduce gains and pushback, yet preserve the equilibrium structure and continuity of closed-loop stability. We derive Soft-Nash from both a KL-regularized trust-region and a maximum-entropy viewpoint, obtaining a closed-form robot best response that shrinks authority and aligns the fixture with the operator's input as $τ$ grows. We implement Soft-Nash on a 6-DoF haptic device in 3D tracking task ($n=12$). Moderate softness ($τ\approx 1-3$, especially $τ=2$) maintains tracking error statistically indistinguishable from a tuned classic VF while sharply reducing controller-user conflict, lowering NASA-TLX workload, and increasing Sense of Agency (SoAS). A composite BalancedScore that combines normalized accuracy and non-fighting behavior peaks near $τ=2-3$. These results show that a one-parameter Soft-Nash policy can preserve accuracy while improving comfort and perceived agency, providing a practical and interpretable pathway to personalized shared control in haptics and teleoperation.
>
---
#### [new 008] LatBot: Distilling Universal Latent Actions for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出LatBot框架，旨在学习通用的潜在动作表示以提升机器人操作任务的泛化能力。针对现有方法忽视物理先验的问题，通过联合优化未来帧重建与动作序列预测，引入真实世界运动信息，并分解为运动与场景令牌，增强对动态变化的区分能力。在仿真与真实机器人上均实现强少样本迁移性能。**

- **链接: [https://arxiv.org/pdf/2511.23034v1](https://arxiv.org/pdf/2511.23034v1)**

> **作者:** Zuolei Li; Xingyu Gao; Xiaofan Wang; Jianlong Fu
>
> **备注:** Project Page: https://mm-robot.github.io/distill_latent_action/
>
> **摘要:** Learning transferable latent actions from large-scale object manipulation videos can significantly enhance generalization in downstream robotics tasks, as such representations are agnostic to different robot embodiments. Existing approaches primarily rely on visual reconstruction objectives while neglecting physical priors, leading to sub-optimal performance in learning universal representations. To address these challenges, we propose a Universal Latent Action Learning framework that takes task instructions and multiple frames as inputs, and optimizes both future frame reconstruction and action sequence prediction. Unlike prior works, incorporating action predictions (e.g., gripper or hand trajectories and orientations) allows the model to capture richer physical priors such as real-world distances and orientations, thereby enabling seamless transferability to downstream tasks. We further decompose the latent actions into learnable motion and scene tokens to distinguish the robot's active movements from environmental changes, thus filtering out irrelevant dynamics. By distilling the learned latent actions into the latest VLA models, we achieve strong performance across both simulated (SIMPLER and LIBERO) and real-world robot settings. Notably, with only 10 real-world trajectories per task collected on a Franka robot, our approach successfully completes all five challenging tasks, demonstrating strong few-shot transferability in robotic manipulation.
>
---
#### [new 009] Design of an Adaptive Modular Anthropomorphic Dexterous Hand for Human-like Manipulation
- **分类: cs.RO**

- **简介: 该论文针对手部操作中驱动复杂性与灵巧性的矛盾，提出一种基于生物协同的4自由度双驱动指结构，设计了模块化仿人灵巧手。通过生物启发的关节协调映射，构建五指系统并建立运动学模型，实现自适应抓握与操作。实验验证了设计的有效性。**

- **链接: [https://arxiv.org/pdf/2511.22100v1](https://arxiv.org/pdf/2511.22100v1)**

> **作者:** Zelong Zhou; Wenrui Chen; Zeyun Hu; Qiang Diao; Qixin Gao; Yaonan Wang
>
> **备注:** 7 pages, 8 figures
>
> **摘要:** Biological synergies have emerged as a widely adopted paradigm for dexterous hand design, enabling human-like manipulation with a small number of actuators. Nonetheless, excessive coupling tends to diminish the dexterity of hands. This paper tackles the trade-off between actuation complexity and dexterity by proposing an anthropomorphic finger topology with 4 DoFs driven by 2 actuators, and by developing an adaptive, modular dexterous hand based on this finger topology. We explore the biological basis of hand synergies and human gesture analysis, translating joint-level coordination and structural attributes into a modular finger architecture. Leveraging these biomimetic mappings, we design a five-finger modular hand and establish its kinematic model to analyze adaptive grasping and in-hand manipulation. Finally, we construct a physical prototype and conduct preliminary experiments, which validate the effectiveness of the proposed design and analysis.
>
---
#### [new 010] SafeHumanoid: VLM-RAG-driven Control of Upper Body Impedance for Humanoid Robot
- **分类: cs.RO**

- **简介: 该论文提出SafeHumanoid系统，解决人形机器人在交互中需根据场景与人体接近度动态调节阻抗和速度的问题。通过视觉语言模型与检索增强生成结合，实现基于情境的上肢阻抗控制，提升协作安全性与任务适应性。**

- **链接: [https://arxiv.org/pdf/2511.23300v1](https://arxiv.org/pdf/2511.23300v1)**

> **作者:** Yara Mahmoud; Jeffrin Sam; Nguyen Khang; Marcelino Fernando; Issatay Tokmurziyev; Miguel Altamirano Cabrera; Muhammad Haris Khan; Artem Lykov; Dzmitry Tsetserukou
>
> **摘要:** Safe and trustworthy Human Robot Interaction (HRI) requires robots not only to complete tasks but also to regulate impedance and speed according to scene context and human proximity. We present SafeHumanoid, an egocentric vision pipeline that links Vision Language Models (VLMs) with Retrieval-Augmented Generation (RAG) to schedule impedance and velocity parameters for a humanoid robot. Egocentric frames are processed by a structured VLM prompt, embedded and matched against a curated database of validated scenarios, and mapped to joint-level impedance commands via inverse kinematics. We evaluate the system on tabletop manipulation tasks with and without human presence, including wiping, object handovers, and liquid pouring. The results show that the pipeline adapts stiffness, damping, and speed profiles in a context-aware manner, maintaining task success while improving safety. Although current inference latency (up to 1.4 s) limits responsiveness in highly dynamic settings, SafeHumanoid demonstrates that semantic grounding of impedance control is a viable path toward safer, standard-compliant humanoid collaboration.
>
---
#### [new 011] Automated Generation of MDPs Using Logic Programming and LLMs for Robotic Applications
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种融合大语言模型与逻辑编程的框架，用于自动构建机器人应用中的马尔可夫决策过程（MDP）。通过LLM从自然语言中提取结构化知识，自动生成MDP并合成最优策略，实现低人工干预的可执行规划。属于自动化规划与形式化验证任务，旨在解决复杂场景下手动建模耗时难题。**

- **链接: [https://arxiv.org/pdf/2511.23143v1](https://arxiv.org/pdf/2511.23143v1)**

> **作者:** Enrico Saccon; Davide De Martini; Matteo Saveriano; Edoardo Lamon; Luigi Palopoli; Marco Roveri
>
> **备注:** 9 pages, 11 figures, 2 tables, 2 algorithms, accepted for publication in IEEE Robotics and Automation Letters
>
> **摘要:** We present a novel framework that integrates Large Language Models (LLMs) with automated planning and formal verification to streamline the creation and use of Markov Decision Processes (MDP). Our system leverages LLMs to extract structured knowledge in the form of a Prolog knowledge base from natural language (NL) descriptions. It then automatically constructs an MDP through reachability analysis, and synthesises optimal policies using the Storm model checker. The resulting policy is exported as a state-action table for execution. We validate the framework in three human-robot interaction scenarios, demonstrating its ability to produce executable policies with minimal manual effort. This work highlights the potential of combining language models with formal methods to enable more accessible and scalable probabilistic planning in robotics.
>
---
#### [new 012] Adaptive Factor Graph-Based Tightly Coupled GNSS/IMU Fusion for Robust Positionin
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对GNSS/IMU紧耦合定位在信号受限环境下的鲁棒性问题，提出基于因子图的自适应融合框架。通过引入Barron损失函数，动态抑制异常观测，提升对非高斯噪声和离群值的抵抗能力，显著降低定位误差，尤其在城市峡谷环境中表现优异。**

- **链接: [https://arxiv.org/pdf/2511.23017v1](https://arxiv.org/pdf/2511.23017v1)**

> **作者:** Elham Ahmadi; Alireza Olama; Petri Välisuo; Heidi Kuusniemi
>
> **摘要:** Reliable positioning in GNSS-challenged environments remains a critical challenge for navigation systems. Tightly coupled GNSS/IMU fusion improves robustness but remains vulnerable to non-Gaussian noise and outliers. We present a robust and adaptive factor graph-based fusion framework that directly integrates GNSS pseudorange measurements with IMU preintegration factors and incorporates the Barron loss, a general robust loss function that unifies several m-estimators through a single tunable parameter. By adaptively down weighting unreliable GNSS measurements, our approach improves resilience positioning. The method is implemented in an extended GTSAM framework and evaluated on the UrbanNav dataset. The proposed solution reduces positioning errors by up to 41% relative to standard FGO, and achieves even larger improvements over extended Kalman filter (EKF) baselines in urban canyon environments. These results highlight the benefits of Barron loss in enhancing the resilience of GNSS/IMU-based navigation in urban and signal-compromised environments.
>
---
#### [new 013] Mechanistic Finetuning of Vision-Language-Action Models via Few-Shot Demonstrations
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人任务中因物理差异需精细调优的问题，提出“机器人引导”方法。通过少量示范识别任务特定注意力头，实现精准、高效、可解释的微调，显著提升模型在真实机器人上的适应性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22697v1](https://arxiv.org/pdf/2511.22697v1)**

> **作者:** Chancharik Mitra; Yusen Luo; Raj Saravanan; Dantong Niu; Anirudh Pai; Jesse Thomason; Trevor Darrell; Abrar Anwar; Deva Ramanan; Roei Herzig
>
> **摘要:** Vision-Language Action (VLAs) models promise to extend the remarkable success of vision-language models (VLMs) to robotics. Yet, unlike VLMs in the vision-language domain, VLAs for robotics require finetuning to contend with varying physical factors like robot embodiment, environment characteristics, and spatial relationships of each task. Existing fine-tuning methods lack specificity, adapting the same set of parameters regardless of a task's visual, linguistic, and physical characteristics. Inspired by functional specificity in neuroscience, we hypothesize that it is more effective to finetune sparse model representations specific to a given task. In this work, we introduce Robotic Steering, a finetuning approach grounded in mechanistic interpretability that leverages few-shot demonstrations to identify and selectively finetune task-specific attention heads aligned with the physical, visual, and linguistic requirements of robotic tasks. Through comprehensive on-robot evaluations with a Franka Emika robot arm, we demonstrate that Robotic Steering outperforms LoRA while achieving superior robustness under task variation, reduced computational cost, and enhanced interpretability for adapting VLAs to diverse robotic tasks.
>
---
#### [new 014] Fault-Tolerant MARL for CAVs under Observation Perturbations for Highway On-Ramp Merging
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文针对高速匝道汇入场景中自动驾驶车辆（CAVs）在多智能体强化学习（MARL）应用中的观测故障问题。提出一种容错MARL方法，通过对抗性扰动注入与自诊断重构机制，提升系统对观测噪声的鲁棒性，在仿真中显著改善了安全与效率。**

- **链接: [https://arxiv.org/pdf/2511.23193v1](https://arxiv.org/pdf/2511.23193v1)**

> **作者:** Yuchen Shi; Huaxin Pei; Yi Zhang; Danya Yao
>
> **摘要:** Multi-Agent Reinforcement Learning (MARL) holds significant promise for enabling cooperative driving among Connected and Automated Vehicles (CAVs). However, its practical application is hindered by a critical limitation, i.e., insufficient fault tolerance against observational faults. Such faults, which appear as perturbations in the vehicles' perceived data, can substantially compromise the performance of MARL-based driving systems. Addressing this problem presents two primary challenges. One is to generate adversarial perturbations that effectively stress the policy during training, and the other is to equip vehicles with the capability to mitigate the impact of corrupted observations. To overcome the challenges, we propose a fault-tolerant MARL method for cooperative on-ramp vehicles incorporating two key agents. First, an adversarial fault injection agent is co-trained to generate perturbations that actively challenge and harden the vehicle policies. Second, we design a novel fault-tolerant vehicle agent equipped with a self-diagnosis capability, which leverages the inherent spatio-temporal correlations in vehicle state sequences to detect faults and reconstruct credible observations, thereby shielding the policy from misleading inputs. Experiments in a simulated highway merging scenario demonstrate that our method significantly outperforms baseline MARL approaches, achieving near-fault-free levels of safety and efficiency under various observation fault patterns.
>
---
#### [new 015] Incorporating Ephemeral Traffic Waves in A Data-Driven Framework for Microsimulation in CARLA
- **分类: cs.RO; cs.ET**

- **简介: 该论文提出一种基于CARLA的数据驱动微仿真框架，通过引入真实交通波数据重构高速公路交通动态。针对传统仿真难以复现瞬态交通波的问题，利用I-24测试场高精度时空数据，构建以自车为中心的共仿真系统，采用“幽灵单元”控制边界，实现全时空图保真度仿真，支持交通控制与自动驾驶评估。**

- **链接: [https://arxiv.org/pdf/2511.23236v1](https://arxiv.org/pdf/2511.23236v1)**

> **作者:** Alex Richardson; Azhar Hasan; Gabor Karsai; Jonathan Sprinkle
>
> **备注:** Submitted to IEEE IV 2026
>
> **摘要:** This paper introduces a data-driven traffic microsimulation framework in CARLA that reconstructs real-world wave dynamics using high-fidelity time-space data from the I-24 MOTION testbed. Calibration of road networks in microsimulators to reproduce ephemeral phenomena such as traffic waves for large-scale simulation is a process that is fraught with challenges. This work reconsiders the existence of the traffic state data as boundary conditions on an ego vehicle moving through previously recorded traffic data, rather than reproducing those traffic phenomena in a calibrated microsim. Our approach is to autogenerate a 1 mile highway segment corresponding to I-24, and use the I-24 data to power a cosimulation module that injects traffic information into the simulation. The CARLA and cosimulation simulations are centered around an ego vehicle sampled from the empirical data, with autogeneration of "visible" traffic within the longitudinal range of the ego vehicle. Boundary control beyond these visible ranges is achieved using ghost cells behind (upstream) and ahead (downstream) of the ego vehicle. Unlike prior simulation work that focuses on local car-following behavior or abstract geometries, our framework targets full time-space diagram fidelity as the validation objective. Leveraging CARLA's rich sensor suite and configurable vehicle dynamics, we simulate wave formation and dissipation in both low-congestion and high-congestion scenarios for qualitative analysis. The resulting emergent behavior closely mirrors that of real traffic, providing a novel cosimulation framework for evaluating traffic control strategies, perception-driven autonomy, and future deployment of wave mitigation solutions. Our work bridges microscopic modeling with physical experimental data, enabling the first perceptually realistic, boundary-driven simulation of empirical traffic wave phenomena in CARLA.
>
---
#### [new 016] Deadlock-Free Hybrid RL-MAPF Framework for Zero-Shot Multi-Robot Navigation
- **分类: cs.RO**

- **简介: 该论文针对多机器人在复杂环境中导航时易发生死锁的问题，提出一种零样本的混合框架。通过将强化学习的反应式导航与按需的多智能体路径规划（MAPF）结合，利用安全层检测死锁并触发协调控制器，生成全局可行轨迹，显著提升导航成功率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22685v1](https://arxiv.org/pdf/2511.22685v1)**

> **作者:** Haoyi Wang; Licheng Luo; Yiannis Kantaros; Bruno Sinopoli; Mingyu Cai
>
> **摘要:** Multi-robot navigation in cluttered environments presents fundamental challenges in balancing reactive collision avoidance with long-range goal achievement. When navigating through narrow passages or confined spaces, deadlocks frequently emerge that prevent agents from reaching their destinations, particularly when Reinforcement Learning (RL) control policies encounter novel configurations out of learning distribution. Existing RL-based approaches suffer from limited generalization capability in unseen environments. We propose a hybrid framework that seamlessly integrates RL-based reactive navigation with on-demand Multi-Agent Path Finding (MAPF) to explicitly resolve topological deadlocks. Our approach integrates a safety layer that monitors agent progress to detect deadlocks and, when detected, triggers a coordination controller for affected agents. The framework constructs globally feasible trajectories via MAPF and regulates waypoint progression to reduce inter-agent conflicts during navigation. Extensive evaluation on dense multi-agent benchmarks shows that our method boosts task completion from marginal to near-universal success, markedly reducing deadlocks and collisions. When integrated with hierarchical task planning, it enables coordinated navigation for heterogeneous robots, demonstrating that coupling reactive RL navigation with selective MAPF intervention yields a robust, zero-shot performance.
>
---
#### [new 017] Constant-Volume Deformation Manufacturing for Material-Efficient Shaping
- **分类: cs.RO**

- **简介: 该论文提出一种保体积变形制造方法，解决传统增减材制造中体积损失与形状偏差问题。通过实时体积一致性建模、几何引导的变形预测及误差补偿，实现塑料材料高精度、零浪费成形，提升材料利用率至98%以上，推动可持续定制化制造。**

- **链接: [https://arxiv.org/pdf/2511.22042v1](https://arxiv.org/pdf/2511.22042v1)**

> **作者:** Lei Li; Jiale Gong; Ziyang Li; Hong Wang
>
> **备注:** 46 pages, 27 figures
>
> **摘要:** Additive and subtractive manufacturing enable complex geometries but rely on discrete stacking or local removal, limiting continuous and controllable deformation and causing volume loss and shape deviations. We present a volumepreserving digital-mold paradigm that integrates real-time volume-consistency modeling with geometry-informed deformation prediction and an error-compensation strategy to achieve highly predictable shaping of plastic materials. By analyzing deformation patterns and error trends from post-formed point clouds, our method corrects elastic rebound and accumulation errors, maintaining volume consistency and surface continuity. Experiments on five representative geometries demonstrate that the system reproduces target shapes with high fidelity while achieving over 98% material utilization. This approach establishes a digitally driven, reproducible pathway for sustainable, zero-waste shaping of user-defined designs, bridging digital modeling, real-time sensing, and adaptive forming, and advancing next-generation sustainable and customizable manufacturing.
>
---
#### [new 018] Visual-Geometry Diffusion Policy: Robust Generalization via Complementarity-Aware Multimodal Fusion
- **分类: cs.RO**

- **简介: 该论文针对模仿学习中视觉-几何模态融合不均衡导致泛化能力差的问题，提出视觉几何扩散策略（VGDP）。通过模态丢弃机制强制互补使用RGB与点云信息，以轻量级交叉注意力实现模态交互。实验表明，该方法在18个仿真和4个真实任务中显著提升泛化性能，尤其在视觉与空间扰动下表现更优。**

- **链接: [https://arxiv.org/pdf/2511.22445v1](https://arxiv.org/pdf/2511.22445v1)**

> **作者:** Yikai Tang; Haoran Geng; Sheng Zang; Pieter Abbeel; Jitendra Malik
>
> **摘要:** Imitation learning has emerged as a crucial ap proach for acquiring visuomotor skills from demonstrations, where designing effective observation encoders is essential for policy generalization. However, existing methods often struggle to generalize under spatial and visual randomizations, instead tending to overfit. To address this challenge, we propose Visual Geometry Diffusion Policy (VGDP), a multimodal imitation learning framework built around a Complementarity-Aware Fusion Module where modality-wise dropout enforces balanced use of RGB and point-cloud cues, with cross-attention serving only as a lightweight interaction layer. Our experiments show that the expressiveness of the fused latent space is largely induced by the enforced complementarity from modality-wise dropout, with cross-attention serving primarily as a lightweight interaction mechanism rather than the main source of robustness. Across a benchmark of 18 simulated tasks and 4 real-world tasks, VGDP outperforms seven baseline policies with an average performance improvement of 39.1%. More importantly, VGDP demonstrates strong robustness under visual and spatial per turbations, surpassing baselines with an average improvement of 41.5% in different visual conditions and 15.2% in different spatial settings.
>
---
#### [new 019] CAPE: Context-Aware Diffusion Policy Via Proximal Mode Expansion for Collision Avoidance
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对机器人碰撞避障任务，解决数据稀缺导致泛化能力差的问题。提出CAPE框架，通过上下文感知的先验引导迭代去噪，扩展轨迹分布模式，在未见环境中生成无碰撞、达目标的平滑轨迹，显著提升仿真与真实场景下的成功率。**

- **链接: [https://arxiv.org/pdf/2511.22773v1](https://arxiv.org/pdf/2511.22773v1)**

> **作者:** Rui Heng Yang; Xuan Zhao; Leo Maxime Brunswic; Montgomery Alban; Mateo Clemente; Tongtong Cao; Jun Jin; Amir Rasouli
>
> **备注:** 4 tables, 9 figures
>
> **摘要:** In robotics, diffusion models can capture multi-modal trajectories from demonstrations, making them a transformative approach in imitation learning. However, achieving optimal performance following this regiment requires a large-scale dataset, which is costly to obtain, especially for challenging tasks, such as collision avoidance. In those tasks, generalization at test time demands coverage of many obstacles types and their spatial configurations, which are impractical to acquire purely via data. To remedy this problem, we propose Context-Aware diffusion policy via Proximal mode Expansion (CAPE), a framework that expands trajectory distribution modes with context-aware prior and guidance at inference via a novel prior-seeded iterative guided refinement procedure. The framework generates an initial trajectory plan and executes a short prefix trajectory, and then the remaining trajectory segment is perturbed to an intermediate noise level, forming a trajectory prior. Such a prior is context-aware and preserves task intent. Repeating the process with context-aware guided denoising iteratively expands mode support to allow finding smoother, less collision-prone trajectories. For collision avoidance, CAPE expands trajectory distribution modes with collision-aware context, enabling the sampling of collision-free trajectories in previously unseen environments while maintaining goal consistency. We evaluate CAPE on diverse manipulation tasks in cluttered unseen simulated and real-world settings and show up to 26% and 80% higher success rates respectively compared to SOTA methods, demonstrating better generalization to unseen environments.
>
---
#### [new 020] $\mathcal{E}_0$: Enhancing Generalization and Fine-Grained Control in VLA Models via Continuized Discrete Diffusion
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对视觉-语言-动作（VLA）模型泛化能力弱、动作控制粗略的问题，提出E0框架。通过连续化离散扩散机制，实现细粒度动作生成与更强语义控制，提升跨场景、视角的鲁棒性与精度，在多个基准上达到最优性能。**

- **链接: [https://arxiv.org/pdf/2511.21542v1](https://arxiv.org/pdf/2511.21542v1)**

> **作者:** Zhihao Zhan; Jiaying Zhou; Likui Zhang; Qinhan Lv; Hao Liu; Jusheng Zhang; Weizheng Li; Ziliang Chen; Tianshui Chen; Keze Wang; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. Yet existing VLA models still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We introduce E0, a continuized discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. Compared with continuous diffusion policies, E0 offers two key advantages: (1) discrete action tokens align naturally with the symbolic structure of pretrained VLM/VLA backbones, enabling stronger semantic conditioning; and 2. discrete diffusion matches the true quantized nature of real-world robot control-whose hardware constraints (e.g., encoder resolution, control frequency, actuation latency) inherently discretize continuous signals-and therefore benefits from a Bayes-optimal denoiser that models the correct discrete action distribution, leading to stronger generalization. Compared with discrete autoregressive and mask-based discrete diffusion models, E0 supports a significantly larger and finer-grained action vocabulary and avoids the distributional mismatch introduced by masking-based corruptions-yielding more accurate fine-grained action control. We further introduce a spherical viewpoint perturbation augmentation method to improve robustness to camera shifts without additional data. Experiments on LIBERO, VLABench, and ManiSkill show that E0 achieves state-of-the-art performance across 14 diverse environments, outperforming strong baselines by 10.7% on average. Real-world evaluation on a Franka arm confirms that E0 delivers precise, robust, and transferable manipulation, establishing discrete diffusion as a promising direction for generalizable VLA policy learning.
>
---
#### [new 021] Safe Autonomous Lane Changing: Planning with Dynamic Risk Fields and Time-Varying Convex Space Generation
- **分类: cs.RO; cs.HC**

- **简介: 该论文针对自动驾驶变道任务，解决动态环境下的安全与效率平衡问题。提出动态风险场与时变凸可行空间，构建融合风险感知与碰撞规避的优化框架，采用约束iLQR求解，实现安全、平顺、高效的轨迹规划。**

- **链接: [https://arxiv.org/pdf/2511.22829v1](https://arxiv.org/pdf/2511.22829v1)**

> **作者:** Zhen Tian; Zhihao Lin
>
> **摘要:** This paper presents a novel trajectory planning pipeline for complex driving scenarios like autonomous lane changing, by integrating risk-aware planning with guaranteed collision avoidance into a unified optimization framework. We first construct a dynamic risk fields (DRF) that captures both the static and dynamic collision risks from surrounding vehicles. Then, we develop a rigorous strategy for generating time-varying convex feasible spaces that ensure kinematic feasibility and safety requirements. The trajectory planning problem is formulated as a finite-horizon optimal control problem and solved using a constrained iterative Linear Quadratic Regulator (iLQR) algorithm that jointly optimizes trajectory smoothness, control effort, and risk exposure while maintaining strict feasibility. Extensive simulations demonstrate that our method outperforms traditional approaches in terms of safety and efficiency, achieving collision-free trajectories with shorter lane-changing distances (28.59 m) and times (2.84 s) while maintaining smooth and comfortable acceleration patterns. In dense roundabout environments the planner further demonstrates robust adaptability, producing larger safety margins, lower jerk, and superior curvature smoothness compared with APF, MPC, and RRT based baselines. These results confirm that the integrated DRF with convex feasible space and constrained iLQR solver provides a balanced solution for safe, efficient, and comfortable trajectory generation in dynamic and interactive traffic scenarios.
>
---
#### [new 022] 3D Affordance Keypoint Detection for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文针对机器人操作中物体功能理解不足的问题，提出基于RGB-D图像的3D可操作性关键点检测方法。通过引入3D关键点四元组，同时解决“如何操作”与“在何处操作”的问题，突破传统仅关注“是什么”的语义分割局限，显著提升操作指导精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.22195v1](https://arxiv.org/pdf/2511.22195v1)**

> **作者:** Zhiyang Liu; Ruiteng Zhao; Lei Zhou; Chengran Yuan; Yuwei Wu; Sheng Guo; Zhengshen Zhang; Chenchen Liu; Marcelo H Ang
>
> **备注:** Accepted to IROS 2024
>
> **摘要:** This paper presents a novel approach for affordance-informed robotic manipulation by introducing 3D keypoints to enhance the understanding of object parts' functionality. The proposed approach provides direct information about what the potential use of objects is, as well as guidance on where and how a manipulator should engage, whereas conventional methods treat affordance detection as a semantic segmentation task, focusing solely on answering the what question. To address this gap, we propose a Fusion-based Affordance Keypoint Network (FAKP-Net) by introducing 3D keypoint quadruplet that harnesses the synergistic potential of RGB and Depth image to provide information on execution position, direction, and extent. Benchmark testing demonstrates that FAKP-Net outperforms existing models by significant margins in affordance segmentation task and keypoint detection task. Real-world experiments also showcase the reliability of our method in accomplishing manipulation tasks with previously unseen objects.
>
---
#### [new 023] OpenTwinMap: An Open-Source Digital Twin Generator for Urban Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出OpenTwinMap，一个开源、可扩展的Python框架，用于生成高保真城市数字孪生。针对现有工具与特定仿真器绑定、难扩展的问题，该框架融合LiDAR与OSM数据，自动生成道路、地形等静态环境资产，支持并行化处理，可导出至Unreal Engine用于自动驾驶仿真。**

- **链接: [https://arxiv.org/pdf/2511.21925v1](https://arxiv.org/pdf/2511.21925v1)**

> **作者:** Alex Richardson; Jonathan Sprinkle
>
> **摘要:** Digital twins of urban environments play a critical role in advancing autonomous vehicle (AV) research by enabling simulation, validation, and integration with emerging generative world models. While existing tools have demonstrated value, many publicly available solutions are tightly coupled to specific simulators, difficult to extend, or introduce significant technical overhead. For example, CARLA-the most widely used open-source AV simulator-provides a digital twin framework implemented entirely as an Unreal Engine C++ plugin, limiting flexibility and rapid prototyping. In this work, we propose OpenTwinMap, an open-source, Python-based framework for generating high-fidelity 3D urban digital twins. The completed framework will ingest LiDAR scans and OpenStreetMap (OSM) data to produce semantically segmented static environment assets, including road networks, terrain, and urban structures, which can be exported into Unreal Engine for AV simulation. OpenTwinMap emphasizes extensibility and parallelization, lowering the barrier for researchers to adapt and scale the pipeline to diverse urban contexts. We describe the current capabilities of the OpenTwinMap, which includes preprocessing of OSM and LiDAR data, basic road mesh and terrain generation, and preliminary support for CARLA integration.
>
---
#### [new 024] Field-programmable dynamics in a soft magnetic actuator enabling true random number generation and reservoir computing
- **分类: cs.RO; cond-mat.other; cond-mat.str-el**

- **简介: 该论文研究软磁致动器中的复杂动态，旨在解决传统机电系统回避混沌动态的问题。通过设计可调谐、耐疲劳的软磁致动器，实现了真随机数生成与储层计算，并验证其在时间序列预测及人机交互中的应用，拓展了软机器人在智能计算领域的功能边界。**

- **链接: [https://arxiv.org/pdf/2511.23215v1](https://arxiv.org/pdf/2511.23215v1)**

> **作者:** Eduardo Sergio Oliveros-Mata; Oleksandr V. Pylypovskyi; Eleonora Raimondo; Rico Illing; Yevhen Zabila; Lin Guo; Guannan Mu; Mónica Navarro López; Xu Wang; Georgios Tzortzinis; Angelos Filippatos; Gilbert Santiago Cañón Bermúdez; Francesca Garescì; Giovanni Finocchio; Denys Makarov
>
> **摘要:** Complex and even chaotic dynamics, though prevalent in many natural and engineered systems, has been largely avoided in the design of electromechanical systems due to concerns about wear and controlability. Here, we demonstrate that complex dynamics might be particularly advantageous in soft robotics, offering new functionalities beyond motion not easily achievable with traditional actuation methods. We designed and realized resilient magnetic soft actuators capable of operating in a tunable dynamic regime for tens of thousands cycles without fatigue. We experimentally demonstrated the application of these actuators for true random number generation and stochastic computing. {W}e validate soft robots as physical reservoirs capable of performing Mackey--Glass time series prediction. These findings show that exploring the complex dynamics in soft robotics would extend the application scenarios in soft computing, human-robot interaction and collaborative robots as we demonstrate with biomimetic blinking and randomized voice modulation.
>
---
#### [new 025] Analytical Inverse Kinematic Solution for "Moz1" NonSRS 7-DOF Robot arm with novel arm angle
- **分类: cs.RO**

- **简介: 该论文针对Moz1 7-DOF机械臂的逆运动学问题，提出一种基于新臂角表示的解析解法。解决了传统方法在冗余配置下无法定义SEW角及算法奇异性问题，实现了全工作空间内快速、精确的闭式解，可求得每种位姿下的全部16组解。**

- **链接: [https://arxiv.org/pdf/2511.22996v1](https://arxiv.org/pdf/2511.22996v1)**

> **作者:** Ke Chen
>
> **摘要:** This paper presents an analytical solution to the inverse kinematic problem(IKP) for the seven degree-of-freedom (7-DOF) Moz1 Robot Arm with offsets on wrist. We provide closed-form solutions with the novel arm angle . it allow fully self-motion and solve the problem of algorithmic singularities within the workspace. It also provides information on how the redundancy is resolved in a new arm angle representation where traditional SEW angle faied to be defined and how singularities are handled. The solution is simple, fast and exact, providing full solution space (i.e. all 16 solutions) per pose.
>
---
#### [new 026] LLM-Based Generalizable Hierarchical Task Planning and Execution for Heterogeneous Robot Teams with Event-Driven Replanning
- **分类: cs.RO**

- **简介: 该论文提出CoMuRoS系统，解决异构机器人团队在动态环境中实现通用、灵活的任务规划与执行问题。通过中心化任务管理与去中心化执行结合，利用大模型实现自然语言理解、任务分配与事件驱动的实时重规划，支持多机器人协作与人机协同，实现在真实场景中高效应对干扰与失败。**

- **链接: [https://arxiv.org/pdf/2511.22354v1](https://arxiv.org/pdf/2511.22354v1)**

> **作者:** Suraj Borate; Bhavish Rai B; Vipul Pardeshi; Madhu Vadali
>
> **备注:** submitted to ICRA 2026
>
> **摘要:** This paper introduces CoMuRoS (Collaborative Multi-Robot System), a generalizable hierarchical architecture for heterogeneous robot teams that unifies centralized deliberation with decentralized execution, and supports event-driven replanning. A Task Manager LLM interprets natural-language goals, classifies tasks, and allocates subtasks using static rules plus dynamic contexts (task, history, robot and task status, and events).Each robot runs a local LLM that composes executable Python code from primitive skills (ROS2 nodes, policies), while onboard perception (VLMs/image processing) continuously monitors events and classifies them into relevant or irrelevant to the task. Task failures or user intent changes trigger replanning, allowing robots to assist teammates, resume tasks, or request human help. Hardware studies demonstrate autonomous recovery from disruptive events, filtering of irrelevant distractions, and tightly coordinated transport with emergent human-robot cooperation (e.g., multirobot collaborative object recovery success rate: 9/10, coordinated transport: 8/8, human-assisted recovery: 5/5).Simulation studies show intention-aware replanning. A curated textual benchmark spanning 22 scenarios (3 tasks each, around 20 robots) evaluates task allocation, classification, IoU, executability, and correctness, with high average scores (e.g., correctness up to 0.91) across multiple LLMs, a separate replanning set (5 scenarios) achieves 1.0 correctness. Compared with prior LLM-based systems, CoMuRoS uniquely demonstrates runtime, event-driven replanning on physical robots, delivering robust, flexible multi-robot and human-robot collaboration.
>
---
#### [new 027] RealD$^2$iff: Bridging Real-World Gap in Robot Manipulation via Depth Diffusion
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中的仿真到现实（sim2real）视觉差距问题，提出RealD²iff框架。通过逆向的“纯净到噪声”深度扩散方法，模拟真实传感器噪声，构建无标注的成对数据，并实现零样本仿真实现真实世界操作，显著提升泛化性能。**

- **链接: [https://arxiv.org/pdf/2511.22505v1](https://arxiv.org/pdf/2511.22505v1)**

> **作者:** Xiujian Liang; Jiacheng Liu; Mingyang Sun; Qichen He; Cewu Lu; Jianhua Sun
>
> **摘要:** Robot manipulation in the real world is fundamentally constrained by the visual sim2real gap, where depth observations collected in simulation fail to reflect the complex noise patterns inherent to real sensors. In this work, inspired by the denoising capability of diffusion models, we invert the conventional perspective and propose a clean-to-noisy paradigm that learns to synthesize noisy depth, thereby bridging the visual sim2real gap through purely simulation-driven robotic learning. Building on this idea, we introduce RealD$^2$iff, a hierarchical coarse-to-fine diffusion framework that decomposes depth noise into global structural distortions and fine-grained local perturbations. To enable progressive learning of these components, we further develop two complementary strategies: Frequency-Guided Supervision (FGS) for global structure modeling and Discrepancy-Guided Optimization (DGO) for localized refinement. To integrate RealD$^2$iff seamlessly into imitation learning, we construct a pipeline that spans six stages. We provide comprehensive empirical and experimental validation demonstrating the effectiveness of this paradigm. RealD$^2$iff enables two key applications: (1) generating real-world-like depth to construct clean-noisy paired datasets without manual sensor data collection. (2) Achieving zero-shot sim2real robot manipulation, substantially improving real-world performance without additional fine-tuning.
>
---
#### [new 028] Distracted Robot: How Visual Clutter Undermine Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究机器人在杂乱场景中操作性能下降的问题，提出一种基于心理物理学的统一杂乱度度量方法。通过仿真与真实环境实验，评估视觉-语言-动作模型在不同杂乱程度下的表现，发现杂乱显著降低成功率（最高达34%），且各模型对杂乱的敏感性不同。研究表明杂乱度是性能退化的有效预测指标，且微调数据虽有帮助，但无法均衡缓解所有负面影响。**

- **链接: [https://arxiv.org/pdf/2511.22780v1](https://arxiv.org/pdf/2511.22780v1)**

> **作者:** Amir Rasouli; Montgomery Alban; Sajjad Pakdamansavoji; Zhiyuan Li; Zhanguang Zhang; Aaron Wu; Xuan Zhao
>
> **备注:** 12 figures, 2 tables
>
> **摘要:** In this work, we propose an evaluation protocol for examining the performance of robotic manipulation policies in cluttered scenes. Contrary to prior works, we approach evaluation from a psychophysical perspective, therefore we use a unified measure of clutter that accounts for environmental factors as well as the distractors quantity, characteristics, and arrangement. Using this measure, we systematically construct evaluation scenarios in both hyper-realistic simulation and real-world and conduct extensive experimentation on manipulation policies, in particular vision-language-action (VLA) models. Our experiments highlight the significant impact of scene clutter, lowering the performance of the policies, by as much as 34% and show that despite achieving similar average performance across the tasks, different VLA policies have unique vulnerabilities and a relatively low agreement on success scenarios. We further show that our clutter measure is an effective indicator of performance degradation and analyze the impact of distractors in terms of their quantity and occluding influence. At the end, we show that finetuning on enhanced data, although effective, does not equally remedy all negative impacts of clutter on performance.
>
---
#### [new 029] Nonholonomic Narrow Dead-End Escape with Deep Reinforcement Learning
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究阿克曼车辆在狭窄死胡同中的非完整约束下逃逸问题。针对传统规划方法在低测度窄道中采样效率低、连接性差的难题，提出基于深度强化学习的端到端解法：构建可生成可行逃逸轨迹的轨迹生成器与训练环境，训练策略网络以高效完成前后交替的紧凑操作。实验表明，该方法在成功率、动作次数上优于经典规划算法。**

- **链接: [https://arxiv.org/pdf/2511.22338v1](https://arxiv.org/pdf/2511.22338v1)**

> **作者:** Denghan Xiong; Yanzhe Zhao; Yutong Chen; Zichun Wang
>
> **备注:** 14 pages, 5 figures, 1 table, submitted to arXiv
>
> **摘要:** Nonholonomic constraints restrict feasible velocities without reducing configuration-space dimension, which makes collision-free geometric paths generally non-executable for car-like robots. Ackermann steering further imposes curvature bounds and forbids in-place rotation, so escaping from narrow dead ends typically requires tightly sequenced forward and reverse maneuvers. Classical planners that decouple global search and local steering struggle in these settings because narrow passages occupy low-measure regions and nonholonomic reachability shrinks the set of valid connections, which degrades sampling efficiency and increases sensitivity to clearances. We study nonholonomic narrow dead-end escape for Ackermann vehicles and contribute three components. First, we construct a generator that samples multi-phase forward-reverse trajectories compatible with Ackermann kinematics and inflates their envelopes to synthesize families of narrow dead ends that are guaranteed to admit at least one feasible escape. Second, we construct a training environment that enforces kinematic constraints and train a policy using the soft actor-critic algorithm. Third, we evaluate against representative classical planners that combine global search with nonholonomic steering. Across parameterized dead-end families, the learned policy solves a larger fraction of instances, reduces maneuver count, and maintains comparable path length and planning time while under the same sensing and control limits. We provide our project as open source at https://github.com/gitagitty/cisDRL-RobotNav.git
>
---
#### [new 030] A Two Degrees-of-Freedom Floor-Based Robot for Transfer and Rehabilitation Applications
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对康复训练中坐站转换（STS）无法适配不同患者能力的问题，提出一种双自由度楼板式机器人。通过调节阻抗与垂直/前向力，实现个性化训练，同时保持转移辅助功能。实验验证其对自然步态影响小、可精准卸载重心负荷，并提供前向虚拟弹簧辅助起身。**

- **链接: [https://arxiv.org/pdf/2511.22705v1](https://arxiv.org/pdf/2511.22705v1)**

> **作者:** Ian Lalonde; Jeff Denis; Mathieu Lamy; Camille Martin; Karina Lebel; Alexandre Girard
>
> **备注:** 13 pages, 16 figures
>
> **摘要:** The ability to accomplish a sit-to-stand (STS) motion is key to increase functional mobility and reduce rehospitalization risks. While raising aid (transfer) devices and partial bodyweight support (rehabilitation) devices exist, both are unable to adjust the STS training to different mobility levels. Therefore, We have developed an STS training device that allows various configurations of impedance and vertical/forward forces to adapt to many training needs while maintaining commercial raising aid transfer capabilities. Experiments with healthy adults (both men and women) of various heights and weights show that the device 1) has a low impact on the natural STS kinematics, 2) can provide precise weight unloading at the patient's center of mass and 3) can add a forward virtual spring to assist the transfer of the bodyweight to the feet for seat-off, at the start of the STS motion.
>
---
#### [new 031] Bayesian Decentralized Decision-making for Multi-Robot Systems: Sample-efficient Estimation of Event Rates
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多机器人系统在危险环境中的集体决策任务，旨在解决如何在有限观测下高效评估风险区域。提出一种去中心化的贝叶斯框架，利用共轭先验估计事件发生率，实现样本高效的不确定性建模与安全决策，显著提升收敛速度与安全性。**

- **链接: [https://arxiv.org/pdf/2511.22225v1](https://arxiv.org/pdf/2511.22225v1)**

> **作者:** Gabriel Aguirre; Simay Atasoy Bingöl; Heiko Hamann; Jonas Kuckling
>
> **备注:** 7 pages, 3 figures, submitted to IEEE MRS 2025
>
> **摘要:** Effective collective decision-making in swarm robotics often requires balancing exploration, communication and individual uncertainty estimation, especially in hazardous environments where direct measurements are limited or costly. We propose a decentralized Bayesian framework that enables a swarm of simple robots to identify the safer of two areas, each characterized by an unknown rate of hazardous events governed by a Poisson process. Robots employ a conjugate prior to gradually predict the times between events and derive confidence estimates to adapt their behavior. Our simulation results show that the robot swarm consistently chooses the correct area while reducing exposure to hazardous events by being sample-efficient. Compared to baseline heuristics, our proposed approach shows better performance in terms of safety and speed of convergence. The proposed scenario has potential to extend the current set of benchmarks in collective decision-making and our method has applications in adaptive risk-aware sampling and exploration in hazardous, dynamic environments.
>
---
#### [new 032] Bridging Planning and Execution: Multi-Agent Path Finding Under Real-World Deadlines
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对多智能体路径规划（MAPF）中规划与执行脱节的问题，提出REMAP框架，通过引入ExecTimeNet精准估算实际执行时间，使规划考虑真实约束。解决了在真实时限下（MAPF-RD）的高效、可靠路径规划问题，显著提升方案质量。**

- **链接: [https://arxiv.org/pdf/2511.21886v1](https://arxiv.org/pdf/2511.21886v1)**

> **作者:** Jingtian Yan; Shuai Zhou; Stephen F. Smith; Jiaoyang Li
>
> **摘要:** The Multi-Agent Path Finding (MAPF) problem aims to find collision-free paths for multiple agents while optimizing objectives such as the sum of costs or makespan. MAPF has wide applications in domains like automated warehouses, manufacturing systems, and airport logistics. However, most MAPF formulations assume a simplified robot model for planning, which overlooks execution-time factors such as kinodynamic constraints, communication latency, and controller variability. This gap between planning and execution is problematic for time-sensitive applications. To bridge this gap, we propose REMAP, an execution-informed MAPF planning framework that can be combined with leading search-based MAPF planners with minor changes. Our framework integrates the proposed ExecTimeNet to accurately estimate execution time based on planned paths. We demonstrate our method for solving MAPF with Real-world Deadlines (MAPF-RD) problem, where agents must reach their goals before a predefined wall-clock time. We integrate our framework with two popular MAPF methods, MAPF-LNS and CBS. Experiments show that REMAP achieves up to 20% improvement in solution quality over baseline methods (e.g., constant execution speed estimators) on benchmark maps with up to 300 agents.
>
---
#### [new 033] SwordRiding: A Unified Navigation Framework for Quadrotors in Unknown Complex Environments via Online Guiding Vector Fields
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对未知复杂环境中四旋翼无人机实时导航难题，提出SwordRiding框架。通过在线构建引导向量场（GVF），结合欧氏有符号距离场与优化的B样条路径，实现闭环导航，显著提升对风扰等干扰的鲁棒性与实时性能。**

- **链接: [https://arxiv.org/pdf/2511.22043v1](https://arxiv.org/pdf/2511.22043v1)**

> **作者:** Xuchen Liu; Ruocheng Li; Bin Xin; Weijia Yao; Qigeng Duan; Jinqiang Cui; Ben M. Chen; Jie Chen
>
> **备注:** For an experimental demo, see https://www.youtube.com/watch?v=tKYCg266c4o. For the lemma proof, see https://github.com/SmartGroupSystems/GVF_close_loop_planning/blob/main/proofs.md
>
> **摘要:** Although quadrotor navigation has achieved high performance in trajectory planning and control, real-time adaptability in unknown complex environments remains a core challenge. This difficulty mainly arises because most existing planning frameworks operate in an open-loop manner, making it hard to cope with environmental uncertainties such as wind disturbances or external perturbations. This paper presents a unified real-time navigation framework for quadrotors in unknown complex environments, based on the online construction of guiding vector fields (GVFs) from discrete reference path points. In the framework, onboard perception modules build a Euclidean Signed Distance Field (ESDF) representation of the environment, which enables obstacle awareness and path distance evaluation. The system first generates discrete, collision-free path points using a global planner, and then parameterizes them via uniform B-splines to produce a smooth and physically feasible reference trajectory. An adaptive GVF is then synthesized from the ESDF and the optimized B-spline trajectory. Unlike conventional approaches, the method adopts a closed-loop navigation paradigm, which significantly enhances robustness under external disturbances. Compared with conventional GVF methods, the proposed approach directly accommodates discretized paths and maintains compatibility with standard planning algorithms. Extensive simulations and real-world experiments demonstrate improved robustness against external disturbances and superior real-time performance.
>
---
#### [new 034] BUDD-e: an autonomous robotic guide for visually impaired users
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出并实现了一款名为BUDD-e的自主导盲机器人，旨在解决视障人士出行导航难题。通过真实场景测试，验证了其优异性能与用户接受度，展示了在复杂环境中自主导航与人机交互的能力。**

- **链接: [https://arxiv.org/pdf/2511.22541v1](https://arxiv.org/pdf/2511.22541v1)**

> **作者:** Jinyang Li; Marcello Farina; Luca Mozzarelli; Luca Cattaneo; Panita Rattamasanaprapai; Eleonora A. Tagarelli; Matteo Corno; Paolo Perego; Giuseppe Andreoni; Emanuele Lettieri
>
> **备注:** 14 pages
>
> **摘要:** This paper describes the design and the realization of a prototype of the novel guide robot BUDD-e for visually impaired users. The robot has been tested in a real scenario with the help of visually disabled volunteers at ASST Grande Ospedale Metropolitano Niguarda, in Milan. The results of the experimental campaign are throughly described in the paper, displaying its remarkable performance and user-acceptance.
>
---
#### [new 035] Beyond Success: Refining Elegant Robot Manipulation from Mixed-Quality Data via Just-in-Time Intervention
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作模型在机器人操作中执行质量不一的问题，提出通过即时干预机制提升操作优雅性。基于新基准LIBERO-Elegant，构建去耦合的优雅性评估框架，训练一个离线学习的优雅性评判器，在推理时仅于关键节点介入，显著提升任务执行质量，兼顾成功率与执行美感。**

- **链接: [https://arxiv.org/pdf/2511.22555v1](https://arxiv.org/pdf/2511.22555v1)**

> **作者:** Yanbo Mao; Jianlong Fu; Ruoxuan Zhang; Hongxia Xie; Meibao Yao
>
> **摘要:** Vision-Language-Action (VLA) models have enabled notable progress in general-purpose robotic manipulation, yet their learned policies often exhibit variable execution quality. We attribute this variability to the mixed-quality nature of human demonstrations, where the implicit principles that govern how actions should be carried out are only partially satisfied. To address this challenge, we introduce the LIBERO-Elegant benchmark with explicit criteria for evaluating execution quality. Using these criteria, we develop a decoupled refinement framework that improves execution quality without modifying or retraining the base VLA policy. We formalize Elegant Execution as the satisfaction of Implicit Task Constraints (ITCs) and train an Elegance Critic via offline Calibrated Q-Learning to estimate the expected quality of candidate actions. At inference time, a Just-in-Time Intervention (JITI) mechanism monitors critic confidence and intervenes only at decision-critical moments, providing selective, on-demand refinement. Experiments on LIBERO-Elegant and real-world manipulation tasks show that the learned Elegance Critic substantially improves execution quality, even on unseen tasks. The proposed model enables robotic control that values not only whether tasks succeed, but also how they are performed.
>
---
#### [new 036] RSPECT: Robust and Scalable Planner for Energy-Aware Coordination of UAV-UGV Teams in Aerial Monitoring
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究无人机与无人车协同的长期空中监控任务，解决能源受限下路径规划的鲁棒性问题。针对未知环境扰动，提出高效启发式算法RSPECT，实现快速、可扩展的轨迹规划，确保任务在最小时间内完成且对不确定性具有强适应性。**

- **链接: [https://arxiv.org/pdf/2511.21957v1](https://arxiv.org/pdf/2511.21957v1)**

> **作者:** Cahit Ikbal Er; Amin Kashiri; Yasin Yazicioglu
>
> **摘要:** We consider the robust planning of energy-constrained unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs), which act as mobile charging stations, to perform long-horizon aerial monitoring missions. More specifically, given a set of points to be visited by the UAVs and desired final positions of the UAV-UGV teams, the objective is to find a robust plan (the vehicle trajectories) that can be realized without a major revision in the face of uncertainty (e.g., unknown obstacles/terrain, wind) to complete this mission in minimum time. We provide a formal description of this problem as a mixed-integer program (MIP), which is NP-hard. Since exact solution methods are computationally intractable for such problems, we propose RSPECT, a scalable and efficient heuristic. We provide theoretical results on the complexity of our algorithm and the feasibility and robustness of resulting plans. We also demonstrate the performance of our method via simulations and experiments.
>
---
#### [new 037] MLATC: Fast Hierarchical Topological Mapping from 3D LiDAR Point Clouds Based on Adaptive Resonance Theory
- **分类: cs.RO**

- **简介: 该论文针对大场景动态环境中自主机器人构建全局拓扑地图的任务，解决传统方法因全量近邻搜索导致的可扩展性差问题。提出多层自适应共振理论（MLATC），通过分层结构实现从粗到细的近邻搜索，自动调整层次深度，显著提升搜索效率，实现实时建图。**

- **链接: [https://arxiv.org/pdf/2511.22238v1](https://arxiv.org/pdf/2511.22238v1)**

> **作者:** Ryosuke Ofuchi; Yuichiro Toda; Naoki Masuyama; Takayuki Matsuno
>
> **摘要:** This paper addresses the problem of building global topological maps from 3D LiDAR point clouds for autonomous mobile robots operating in large-scale, dynamic, and unknown environments. Adaptive Resonance Theory-based Topological Clustering with Different Topologies (ATC-DT) builds global topological maps represented as graphs while mitigating catastrophic forgetting during sequential processing. However, its winner selection mechanism relies on an exhaustive nearest-neighbor search over all existing nodes, leading to scalability limitations as the map grows. To address this challenge, we propose a hierarchical extension called Multi-Layer ATC (MLATC). MLATC organizes nodes into a hierarchy, enabling the nearest-neighbor search to proceed from coarse to fine resolutions, thereby drastically reducing the number of distance evaluations per query. The number of layers is not fixed in advance. MLATC employs an adaptive layer addition mechanism that automatically deepens the hierarchy when lower layers become saturated, keeping the number of user-defined hyperparameters low. Simulation experiments on synthetic large-scale environments show that MLATC accelerates topological map building compared to the original ATC-DT and exhibits a sublinear, approximately logarithmic scaling of search time with respect to the number of nodes. Experiments on campus-scale real-world LiDAR datasets confirm that MLATC maintains a millisecond-level per-frame runtime and enables real-time global topological map building in large-scale environments, significantly outperforming the original ATC-DT in terms of computational efficiency.
>
---
#### [new 038] BINDER: Instantly Adaptive Mobile Manipulation with Open-Vocabulary Commands
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对动态环境下的开放词汇移动操作任务，解决传统方法在非更新时段丢失环境感知导致的失败问题。提出BINDER框架，通过即时响应模块（IRM）与推理规划模块（DRM）协同，实现持续环境监控与战略规划的融合，显著提升机器人在真实场景中的适应性与成功率。**

- **链接: [https://arxiv.org/pdf/2511.22364v1](https://arxiv.org/pdf/2511.22364v1)**

> **作者:** Seongwon Cho; Daechul Ahn; Donghyun Shin; Hyeonbeom Choi; San Kim; Jonghyun Choi
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Open-vocabulary mobile manipulation (OVMM) requires robots to follow language instructions, navigate, and manipulate while updating their world representation under dynamic environmental changes. However, most prior approaches update their world representation only at discrete update points such as navigation targets, waypoints, or the end of an action step, leaving robots blind between updates and causing cascading failures: overlooked objects, late error detection, and delayed replanning. To address this limitation, we propose BINDER (Bridging INstant and DEliberative Reasoning), a dual process framework that decouples strategic planning from continuous environment monitoring. Specifically, BINDER integrates a Deliberative Response Module (DRM, a multimodal LLM for task planning) with an Instant Response Module (IRM, a VideoLLM for continuous monitoring). The two modules play complementary roles: the DRM performs strategic planning with structured 3D scene updates and guides what the IRM attends to, while the IRM analyzes video streams to update memory, correct ongoing actions, and trigger replanning when necessary. Through this bidirectional coordination, the modules address the trade off between maintaining awareness and avoiding costly updates, enabling robust adaptation under dynamic conditions. Evaluated in three real world environments with dynamic object placement, BINDER achieves substantially higher success and efficiency than SoTA baselines, demonstrating its effectiveness for real world deployment.
>
---
#### [new 039] Improving Robotic Manipulation Robustness via NICE Scene Surgery
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对机器人操作中视觉干扰导致的鲁棒性不足问题，提出NICE框架，通过生成式图像编辑增强场景多样性，无需额外数据采集即可提升模型泛化能力。实验表明，该方法显著改善了复杂场景下的空间可操作性预测与操作成功率，增强了视觉鲁棒性与安全性。**

- **链接: [https://arxiv.org/pdf/2511.22777v1](https://arxiv.org/pdf/2511.22777v1)**

> **作者:** Sajjad Pakdamansavoji; Mozhgan Pourkeshavarz; Adam Sigal; Zhiyuan Li; Rui Heng Yang; Amir Rasouli
>
> **备注:** 11 figures, 3 tables
>
> **摘要:** Learning robust visuomotor policies for robotic manipulation remains a challenge in real-world settings, where visual distractors can significantly degrade performance and safety. In this work, we propose an effective and scalable framework, Naturalistic Inpainting for Context Enhancement (NICE). Our method minimizes out-of-distribution (OOD) gap in imitation learning by increasing visual diversity through construction of new experiences using existing demonstrations. By utilizing image generative frameworks and large language models, NICE performs three editing operations, object replacement, restyling, and removal of distracting (non-target) objects. These changes preserve spatial relationships without obstructing target objects and maintain action-label consistency. Unlike previous approaches, NICE requires no additional robot data collection, simulator access, or custom model training, making it readily applicable to existing robotic datasets. Using real-world scenes, we showcase the capability of our framework in producing photo-realistic scene enhancement. For downstream tasks, we use NICE data to finetune a vision-language model (VLM) for spatial affordance prediction and a vision-language-action (VLA) policy for object manipulation. Our evaluations show that NICE successfully minimizes OOD gaps, resulting in over 20% improvement in accuracy for affordance prediction in highly cluttered scenes. For manipulation tasks, success rate increases on average by 11% when testing in environments populated with distractors in different quantities. Furthermore, we show that our method improves visual robustness, lowering target confusion by 6%, and enhances safety by reducing collision rate by 7%.
>
---
#### [new 040] Seeing before Observable: Potential Risk Reasoning in Autonomous Driving via Vision Language Models
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶中“潜在风险”识别难题，提出PotentialRiskQA数据集与PR-Reasoner模型，通过视觉语言模型实现对未显性显现但可推断的风险的提前识别，解决现有系统缺乏前瞻性安全推理能力的问题。**

- **链接: [https://arxiv.org/pdf/2511.22928v1](https://arxiv.org/pdf/2511.22928v1)**

> **作者:** Jiaxin Liu; Xiangyu Yan; Liang Peng; Lei Yang; Lingjun Zhang; Yuechen Luo; Yueming Tao; Ashton Yu Xuan Tan; Mu Li; Lei Zhang; Ziqi Zhan; Sai Guo; Hong Wang; Jun Li
>
> **摘要:** Ensuring safety remains a key challenge for autonomous vehicles (AVs), especially in rare and complex scenarios. One critical but understudied aspect is the \textbf{potential risk} situations, where the risk is \textbf{not yet observable} but can be inferred from subtle precursors, such as anomalous behaviors or commonsense violations. Recognizing these precursors requires strong semantic understanding and reasoning capabilities, which are often absent in current AV systems due to the scarcity of such cases in existing driving or risk-centric datasets. Moreover, current autonomous driving accident datasets often lack annotations of the causal reasoning chains behind incidents, which are essential for identifying potential risks before they become observable. To address these gaps, we introduce PotentialRiskQA, a novel vision-language dataset designed for reasoning about potential risks prior to observation. Each sample is annotated with structured scene descriptions, semantic precursors, and inferred risk outcomes. Based on this dataset, we further propose PR-Reasoner, a vision-language-model-based framework tailored for onboard potential risk reasoning. Experimental results show that fine-tuning on PotentialRiskQA enables PR-Reasoner to significantly enhance its performance on the potential risk reasoning task compared to baseline VLMs. Together, our dataset and model provide a foundation for developing autonomous systems with improved foresight and proactive safety capabilities, moving toward more intelligent and resilient AVs.
>
---
#### [new 041] SUPER-AD: Semantic Uncertainty-aware Planning for End-to-End Robust Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对端到端自动驾驶中的不确定性盲区问题，提出一种基于摄像头的鲁棒规划框架。通过在BEV空间显式建模摄动不确定性，生成像素级语义-几何可行驶性图，并引入车道跟随正则化以增强规则合规性。方法提升了复杂场景下的安全性和可解释性，在NAVSIM基准上实现领先性能。**

- **链接: [https://arxiv.org/pdf/2511.22865v1](https://arxiv.org/pdf/2511.22865v1)**

> **作者:** Wonjeong Ryu; Seungjun Yu; Seokha Moon; Hojun Choi; Junsung Park; Jinkyu Kim; Hyunjung Shim
>
> **摘要:** End-to-End (E2E) planning has become a powerful paradigm for autonomous driving, yet current systems remain fundamentally uncertainty-blind. They assume perception outputs are fully reliable, even in ambiguous or poorly observed scenes, leaving the planner without an explicit measure of uncertainty. To address this limitation, we propose a camera-only E2E framework that estimates aleatoric uncertainty directly in BEV space and incorporates it into planning. Our method produces a dense, uncertainty-aware drivability map that captures both semantic structure and geometric layout at pixel-level resolution. To further promote safe and rule-compliant behavior, we introduce a lane-following regularization that encodes lane structure and traffic norms. This prior stabilizes trajectory planning under normal conditions while preserving the flexibility needed for maneuvers such as overtaking or lane changes. Together, these components enable robust and interpretable trajectory planning, even under challenging uncertainty conditions. Evaluated on the NAVSIM benchmark, our method achieves state-of-the-art performance, delivering substantial gains on both the challenging NAVHARD and NAVSAFE subsets. These results demonstrate that our principled aleatoric uncertainty modeling combined with driving priors significantly advances the safety and reliability of camera-only E2E autonomous driving.
>
---
#### [new 042] Soft Fluidic Sheet Transistor for Soft Robotic System Enabling Fluid Logic Operations
- **分类: cs.RO**

- **简介: 该论文针对软体机器人系统中高功能与柔性的兼容问题，提出一种基于气压信号的柔性流体晶体管（FST）。通过设计可压缩的聚氨酯片状阀，实现仅用小气压控制主通道的开关，构建了类似电晶体的逻辑门。集成多个FST可实现NOT、NAND、NOR等逻辑运算，并构建自保持锁存电路，最终在单管气压驱动下完成避障行为，验证了其在软体机器人中的可行性。**

- **链接: [https://arxiv.org/pdf/2511.22318v1](https://arxiv.org/pdf/2511.22318v1)**

> **作者:** Yuki Origane; Koya Cho; Hideyuki Tsukagoshi
>
> **备注:** 7 pages, 16 figures
>
> **摘要:** Aiming to achieve both high functionality and flexibility in soft robot system, this paper presents a soft urethane sheet-like valve with an amplifier that can perform logical operations using only pneumatic signals. When the control chamber in the valve is pressurized, the main path is compressed along its central axis, buckling and being pressed,resulting in blockage. This allows control by a pressure signal smaller than that within the main channel. Furthermore, similar to transistors in electrical circuits, when combined, the proposed valve can perform a variety of logical operations. The basic type operates as a NOT logic element, which is named the fluidic sheet transistor (FST). By integrating multiple FSTs, logical operations such as positive logic, NAND, and NOR can be performed on a single sheet. This paper describes the operating principle, fabrication method, and characteristics of the FST,followed by a method for configuring logical operations.Moreover, we demonstrate the construction of a latch circuit(self-holding logic circuit) using FST, introducing a prototype of a fluid robot system that combines a tactile tube as a fluidic detector and fluid actuators. This demonstrates that it is possible to generate behavior that actively changes posture when hitting an obstacle using only air pressure from a single pipe, which verifies the effectiveness of the proposed methods.
>
---
#### [new 043] DiskChunGS: Large-Scale 3D Gaussian SLAM Through Chunk-Based Memory Management
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DiskChunGS，一种基于分块内存管理的3D高斯SLAM系统，解决3D高斯溅射在大规模场景中因GPU显存不足导致的可扩展性问题。通过将场景分块，仅将活动区域保留在GPU内存中，其余部分存于磁盘，实现大场景全局一致重建，支持多种环境与嵌入式平台。**

- **链接: [https://arxiv.org/pdf/2511.23030v1](https://arxiv.org/pdf/2511.23030v1)**

> **作者:** Casimir Feldmann; Maximum Wilder-Smith; Vaishakh Patil; Michael Oechsle; Michael Niemeyer; Keisuke Tateno; Marco Hutter
>
> **摘要:** Recent advances in 3D Gaussian Splatting (3DGS) have demonstrated impressive results for novel view synthesis with real-time rendering capabilities. However, integrating 3DGS with SLAM systems faces a fundamental scalability limitation: methods are constrained by GPU memory capacity, restricting reconstruction to small-scale environments. We present DiskChunGS, a scalable 3DGS SLAM system that overcomes this bottleneck through an out-of-core approach that partitions scenes into spatial chunks and maintains only active regions in GPU memory while storing inactive areas on disk. Our architecture integrates seamlessly with existing SLAM frameworks for pose estimation and loop closure, enabling globally consistent reconstruction at scale. We validate DiskChunGS on indoor scenes (Replica, TUM-RGBD), urban driving scenarios (KITTI), and resource-constrained Nvidia Jetson platforms. Our method uniquely completes all 11 KITTI sequences without memory failures while achieving superior visual quality, demonstrating that algorithmic innovation can overcome the memory constraints that have limited previous 3DGS SLAM methods.
>
---
#### [new 044] MARVO: Marine-Adaptive Radiance-aware Visual Odometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对水下视觉定位难题，提出MARVO框架，融合物理成像模型与学习机制。通过辐射度自适应特征匹配提升复杂水下环境下的特征一致性，结合惯性、压力与视觉信息构建因子图优化系统，利用强化学习优化全局轨迹，实现高精度实时位姿估计。**

- **链接: [https://arxiv.org/pdf/2511.22860v1](https://arxiv.org/pdf/2511.22860v1)**

> **作者:** Sacchin Sundar; Atman Kikani; Aaliya Alam; Sumukh Shrote; A. Nayeemulla Khan; A. Shahina
>
> **备注:** 10 pages, 5 figures, 3 tables, Submitted to CVPR2026
>
> **摘要:** Underwater visual localization remains challenging due to wavelength-dependent attenuation, poor texture, and non-Gaussian sensor noise. We introduce MARVO, a physics-aware, learning-integrated odometry framework that fuses underwater image formation modeling, differentiable matching, and reinforcement-learning optimization. At the front-end, we extend transformer-based feature matcher with a Physics Aware Radiance Adapter that compensates for color channel attenuation and contrast loss, yielding geometrically consistent feature correspondences under turbidity. These semi dense matches are combined with inertial and pressure measurements inside a factor-graph backend, where we formulate a keyframe-based visual-inertial-barometric estimator using GTSAM library. Each keyframe introduces (i) Pre-integrated IMU motion factors, (ii) MARVO-derived visual pose factors, and (iii) barometric depth priors, giving a full-state MAP estimate in real time. Lastly, we introduce a Reinforcement-Learningbased Pose-Graph Optimizer that refines global trajectories beyond local minima of classical least-squares solvers by learning optimal retraction actions on SE(2).
>
---
#### [new 045] SO-Bench: A Structural Output Evaluation of Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文针对多模态大模型在视觉输入下生成结构化输出的能力，提出SO-Bench基准。解决现有缺乏系统评估框架的问题，覆盖四类视觉场景，包含超6500个JSON模式与1800对图像-模式对。通过基准测试揭示模型在结构化推理上的不足，并开展训练优化，推动多模态结构化生成发展。**

- **链接: [https://arxiv.org/pdf/2511.21750v1](https://arxiv.org/pdf/2511.21750v1)**

> **作者:** Di Feng; Kaixin Ma; Feng Nan; Haofeng Chen; Bohan Zhai; David Griffiths; Mingfei Gao; Zhe Gan; Eshan Verma; Yinfei Yang; Zhifeng Chen; Afshin Dehghan
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly deployed in real-world, agentic settings where outputs must not only be correct, but also conform to predefined data schemas. Despite recent progress in structured generation in textual domain, there is still no benchmark that systematically evaluates schema-grounded information extraction and reasoning over visual inputs. In this work, we conduct a comprehensive study of visual structural output capabilities for MLLMs with our carefully designed SO-Bench benchmark. Covering four visual domains, including UI screens, natural images, documents, and charts, SO-Bench is built from over 6.5K diverse JSON schemas and 1.8K curated image-schema pairs with human-verified quality. Benchmarking experiments on open-sourced and frontier proprietary models reveal persistent gaps in predicting accurate, schema compliant outputs, highlighting the need for better multimodal structured reasoning. Beyond benchmarking, we further conduct training experiments to largely improve the model's structured output capability. We plan to make the benchmark available to the community.
>
---
#### [new 046] Massively Parallel Imitation Learning of Mouse Forelimb Musculoskeletal Reaching Dynamics
- **分类: cs.LG; cs.RO; q-bio.NC; q-bio.QM**

- **简介: 该论文属于仿生运动控制建模任务，旨在通过模仿学习还原小鼠前肢抓取的生物力学动态。针对真实运动与模拟间差异大的问题，研究构建了基于JAX和Mujoco-MJX的高速仿真框架，利用神经科学采集的运动数据训练肌肉骨骼模型，引入能量与速度约束，显著提升模拟肌电（EMG）信号预测精度，验证了约束对运动控制建模的关键作用。**

- **链接: [https://arxiv.org/pdf/2511.21848v1](https://arxiv.org/pdf/2511.21848v1)**

> **作者:** Eric Leonardis; Akira Nagamori; Ayesha Thanawalla; Yuanjia Yang; Joshua Park; Hutton Saunders; Eiman Azim; Talmo Pereira
>
> **备注:** Accepted at NeurIPS 2025 Workshop Data on the Brain & Mind: Concrete Applications of AI to Neuroscience and Cognitive Science. 12 pages, 4 figures
>
> **摘要:** The brain has evolved to effectively control the body, and in order to understand the relationship we need to model the sensorimotor transformations underlying embodied control. As part of a coordinated effort, we are developing a general-purpose platform for behavior-driven simulation modeling high fidelity behavioral dynamics, biomechanics, and neural circuit architectures underlying embodied control. We present a pipeline for taking kinematics data from the neuroscience lab and creating a pipeline for recapitulating those natural movements in a biomechanical model. We implement a imitation learning framework to perform a dexterous forelimb reaching task with a musculoskeletal model in a simulated physics environment. The mouse arm model is currently training at faster than 1 million training steps per second due to GPU acceleration with JAX and Mujoco-MJX. We present results that indicate that adding naturalistic constraints on energy and velocity lead to simulated musculoskeletal activity that better predict real EMG signals. This work provides evidence to suggest that energy and control constraints are critical to modeling musculoskeletal motor control.
>
---
#### [new 047] MTR-VP: Towards End-to-End Trajectory Planning through Context-Driven Image Encoding and Multiple Trajectory Prediction
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对自动驾驶端到端轨迹规划任务，提出MTR-VP方法，用视觉Transformer替代地图特征，通过图像与运动状态生成场景上下文嵌入，并结合多轨迹预测提升规划性能。实验表明，多轨迹输出优于单轨迹，而单纯融合视觉与运动特征效果有限。**

- **链接: [https://arxiv.org/pdf/2511.22181v1](https://arxiv.org/pdf/2511.22181v1)**

> **作者:** Maitrayee Keskar; Mohan Trivedi; Ross Greer
>
> **备注:** 8 pages, 3 figures, 4 tables
>
> **摘要:** We present a method for trajectory planning for autonomous driving, learning image-based context embeddings that align with motion prediction frameworks and planning-based intention input. Within our method, a ViT encoder takes raw images and past kinematic state as input and is trained to produce context embeddings, drawing inspiration from those generated by the recent MTR (Motion Transformer) encoder, effectively substituting map-based features with learned visual representations. MTR provides a strong foundation for multimodal trajectory prediction by localizing agent intent and refining motion iteratively via motion query pairs; we name our approach MTR-VP (Motion Transformer for Vision-based Planning), and instead of the learnable intention queries used in the MTR decoder, we use cross attention on the intent and the context embeddings, which reflect a combination of information encoded from the driving scene and past vehicle states. We evaluate our methods on the Waymo End-to-End Driving Dataset, which requires predicting the agent's future 5-second trajectory in bird's-eye-view coordinates using prior camera images, agent pose history, and routing goals. We analyze our architecture using ablation studies, removing input images and multiple trajectory output. Our results suggest that transformer-based methods that are used to combine the visual features along with the kinetic features such as the past trajectory features are not effective at combining both modes to produce useful scene context embeddings, even when intention embeddings are augmented with foundation-model representations of scene context from CLIP and DINOv2, but that predicting a distribution over multiple futures instead of a single future trajectory boosts planning performance.
>
---
#### [new 048] Lips-Jaw and Tongue-Jaw Articulatory Tradeoff in DYNARTmo
- **分类: cs.CL; cs.RO**

- **简介: 该论文研究语音生成中唇-下颌与舌-下颌的协同运动机制，针对动态构音模型DYNARTmo如何实现多发音器间的努力分配问题。通过仿真不同辅音-元音组合，验证了模型能再现真实口腔运动模式，如下颌随发音部位变化、舌颌共动等，证明其在简化假设下仍可有效模拟构音协同效应。**

- **链接: [https://arxiv.org/pdf/2511.22155v1](https://arxiv.org/pdf/2511.22155v1)**

> **作者:** Bernd J. Kröger
>
> **备注:** 12 pages, 3 figures, supplementary material: python code
>
> **摘要:** This paper investigates how the dynamic articulatory model DYNARTmo accounts for articulatory tradeoffs between primary and secondary articulators, with a focus on lips-jaw and tongue-jaw coordination. While DYNARTmo does not implement full task-dynamic second-order biomechanics, it adopts first-order task-space gesture specifications comparable to those used in articulatory phonology and integrates a simplified mechanism for distributing articulatory effort across multiple articulators. We first outline the conceptual relationship between task dynamics and DYNARTmo, emphasizing the distinction between high-level task-space trajectories and their low-level articulatory execution. We then present simulation results for a set of CV syllables that illustrate how jaw displacement varies as a function of both place of articulation (labial, apical, dorsal) and vowel context (/a/, /i/, /u/). The model reproduces empirically attested patterns of articulatory synergy, including jaw-supported apical closures, lower-lip elevation in bilabial stops, tongue-jaw co-movement, and saturation effects in labial constrictions. These results demonstrate that even with computationally simplified assumptions, DYNARTmo can generate realistic spatio-temporal movement patterns that capture key aspects of articulatory tradeoff and synergy across a range of consonant-vowel combinations.
>
---
#### [new 049] Motion-to-Motion Latency Measurement Framework for Connected and Autonomous Vehicle Teleoperation
- **分类: cs.PF; cs.RO**

- **简介: 该论文针对自动驾驶车辆远程操控中的运动延迟问题，提出一种基于霍尔传感器与同步树莓派的运动到运动（M2M）延迟测量框架。旨在解决缺乏标准M2M延迟测量方法的问题，通过中断时间戳实现高精度测量，揭示执行器延迟是主要影响因素。**

- **链接: [https://arxiv.org/pdf/2511.22467v1](https://arxiv.org/pdf/2511.22467v1)**

> **作者:** François Provost; Faisal Hawlader; Mehdi Testouri; Raphaël Frank
>
> **摘要:** Latency is a key performance factor for the teleoperation of Connected and Autonomous Vehicles (CAVs). It affects how quickly an operator can perceive changes in the driving environment and apply corrective actions. Most existing work focuses on Glass-to-Glass (G2G) latency, which captures delays only in the video pipeline. However, there is no standard method for measuring Motion-to-Motion (M2M) latency, defined as the delay between the physical steering movement of the remote operator and the corresponding steering motion in the vehicle. This paper presents an M2M latency measurement framework that uses Hall-effect sensors and two synchronized Raspberry Pi~5 devices. The system records interrupt-based timestamps on both sides to estimate M2M latency, independently of the underlying teleoperation architecture. Precision tests show an accuracy of 10--15~ms, while field results indicate that actuator delays dominate M2M latency, with median values above 750~ms.
>
---
#### [new 050] SimScale: Learning to Drive via Real-World Simulation at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SimScale框架，解决自动驾驶中真实数据稀缺且多样性不足的问题。通过神经渲染与反应式环境生成大规模未见驾驶状态，并设计伪专家提供动作监督。结合真实与仿真数据进行联合训练，显著提升规划模型的鲁棒性与泛化能力，且性能随仿真数据增加而持续提升。**

- **链接: [https://arxiv.org/pdf/2511.23369v1](https://arxiv.org/pdf/2511.23369v1)**

> **作者:** Haochen Tian; Tianyu Li; Haochen Liu; Jiazhi Yang; Yihang Qiu; Guang Li; Junli Wang; Yinfeng Gao; Zhang Zhang; Liang Wang; Hangjun Ye; Tieniu Tan; Long Chen; Hongyang Li
>
> **备注:** Project page: https://opendrivelab.com/SimScale
>
> **摘要:** Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +6.8 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Our simulation data and code would be released.
>
---
#### [new 051] Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Percept-WAM，一种融合2D/3D感知的视觉语言模型，解决自动驾驶中空间理解弱、长尾场景鲁棒性差的问题。通过引入World-PV/BEV tokens和网格条件预测机制，统一感知与决策，提升小物体、远距离等复杂场景下的检测与规划性能。**

- **链接: [https://arxiv.org/pdf/2511.19221v1](https://arxiv.org/pdf/2511.19221v1)**

> **作者:** Jianhua Han; Meng Tian; Jiangtong Zhu; Fan He; Huixin Zhang; Sitong Guo; Dechang Zhu; Hao Tang; Pei Xu; Yuze Guo; Minzhe Niu; Haojie Zhu; Qichao Dong; Xuechao Yan; Siyuan Dong; Lu Hou; Qingqiu Huang; Xiaosong Jia; Hang Xu
>
> **摘要:** Autonomous driving heavily relies on accurate and robust spatial perception. Many failures arise from inaccuracies and instability, especially in long-tail scenarios and complex interactions. However, current vision-language models are weak at spatial grounding and understanding, and VLA systems built on them therefore show limited perception and localization ability. To address these challenges, we introduce Percept-WAM, a perception-enhanced World-Awareness-Action Model that is the first to implicitly integrate 2D/3D scene understanding abilities within a single vision-language model (VLM). Instead of relying on QA-style spatial reasoning, Percept-WAM unifies 2D/3D perception tasks into World-PV and World-BEV tokens, which encode both spatial coordinates and confidence. We propose a grid-conditioned prediction mechanism for dense object perception, incorporating IoU-aware scoring and parallel autoregressive decoding, improving stability in long-tail, far-range, and small-object scenarios. Additionally, Percept-WAM leverages pretrained VLM parameters to retain general intelligence (e.g., logical reasoning) and can output perception results and trajectory control outputs directly. Experiments show that Percept-WAM matches or surpasses classical detectors and segmenters on downstream perception benchmarks, achieving 51.7/58.9 mAP on COCO 2D detection and nuScenes BEV 3D detection. When integrated with trajectory decoders, it further improves planning performance on nuScenes and NAVSIM, e.g., surpassing DiffusionDrive by 2.1 in PMDS on NAVSIM. Qualitative results further highlight its strong open-vocabulary and long-tail generalization.
>
---
#### [new 052] Control Barrier Function for Unknown Systems: An Approximation-free Approach
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文研究未知非线性系统的预定时间安全避障控制问题。针对动态障碍物环境，提出无需在线学习或不确定性估计的逼近自由方法，通过虚拟系统生成安全参考，并用反馈律将真实系统约束在虚拟禁区，实现实时安全与预定时间达目标。**

- **链接: [https://arxiv.org/pdf/2511.23022v1](https://arxiv.org/pdf/2511.23022v1)**

> **作者:** Shubham Sawarkar; Pushpak Jagtap
>
> **摘要:** We study the prescribed-time reach-avoid (PT-RA) control problem for nonlinear systems with unknown dynamics operating in environments with moving obstacles. Unlike robust or learning based Control Barrier Function (CBF) methods, the proposed framework requires neither online model learning nor uncertainty bound estimation. A CBF-based Quadratic Program (CBF-QP) is solved on a simple virtual system to generate a safe reference satisfying PT-RA conditions with respect to time-varying, tightened obstacle and goal sets. The true system is confined to a Virtual Confinement Zone (VCZ) around this reference using an approximation-free feedback law. This construction guarantees real-time safety and prescribed-time target reachability under unknown dynamics and dynamic constraints without explicit model identification or offline precomputation. Simulation results illustrate reliable dynamic obstacle avoidance and timely convergence to the target set.
>
---
#### [new 053] HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶仿真中视图变化大时视觉失真与几何不一致的问题，提出HybridWorldSim框架，融合神经重建与生成模型，实现高保真、可控制的动态场景仿真。构建MIRROR数据集，支持多样化驾驶场景评测，显著提升仿真真实性和可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.22187v1](https://arxiv.org/pdf/2511.22187v1)**

> **作者:** Qiang Li; Yingwenqi Jiang; Tuoxi Li; Duyu Chen; Xiang Feng; Yucheng Ao; Shangyue Liu; Xingchen Yu; Youcheng Cai; Yumeng Liu; Yuexin Ma; Xin Hu; Li Liu; Yu Zhang; Linkun Xu; Bingtao Gao; Xueyuan Wang; Shuchang Zhou; Xianming Liu; Ligang Liu
>
> **摘要:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.
>
---
#### [new 054] DualVLA: Building a Generalizable Embodied Agent via Partial Decoupling of Reasoning and Action
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在综合推理与动作执行时出现的动作性能下降问题，提出DualVLA框架。通过双层数据剪枝与双教师自适应蒸馏，实现推理与动作的解耦优化，在保持强推理能力的同时提升动作准确性。研究还提出VLA Score评估体系，实现对多维度能力的细粒度评测。**

- **链接: [https://arxiv.org/pdf/2511.22134v1](https://arxiv.org/pdf/2511.22134v1)**

> **作者:** Zhen Fang; Zhuoyang Liu; Jiaming Liu; Hao Chen; Yu Zeng; Shiting Huang; Zehui Chen; Lin Chen; Shanghang Zhang; Feng Zhao
>
> **摘要:** To build a generalizable Vision-Language-Action (VLA) model with strong reasoning ability, a common strategy is to first train a specialist VLA on robot demonstrations to acquire reliable manipulation skills, and then incorporate mixed annotated robot data together with multimodal data to restore broader reasoning capabilities. However, we observe that the resulting reasoning VLA often suffers from degraded action performance compared to the specialist model before fine-tuning, a phenomenon we refer to as action degeneration. To address this issue, we propose DualVLA, which enhances action performance through carefully designed post-training while still preserving reasoning capability. We first introduce a dual-layer data pruning method that removes redundant embodied reasoning, preventing it from adversely influencing action learning. To further strengthen action generation, we design a dual-teacher adaptive distillation strategy that assigns different supervision signals to different data domains while maintaining reasoning ability. To fill the evaluation gap for generalist VLAs, we also propose VLA Score, which decouples VLA capability into reasoning, intention, action, and alignment dimensions for a more fine-grained assessment. Experiments show that DualVLA achieves an average success rate of 61.0 in SimplerEnv and an average score of 65.4 across eight competitive multimodal benchmarks, demonstrating a stronger balance between precise action execution and multimodal understanding. Project Website: https://costaliya.github.io/DualVLA/.
>
---
#### [new 055] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对机器人图像与视频分割任务，解决机器人形态多样、结构复杂导致的分割难题。提出RobotSeg模型，通过结构增强记忆关联、自动提示生成和标签高效训练策略，实现无需人工标注的精准分割，并构建了大规模VRS数据集，显著提升分割性能。**

- **链接: [https://arxiv.org/pdf/2511.22950v1](https://arxiv.org/pdf/2511.22950v1)**

> **作者:** Haiyang Mei; Qiming Huang; Hai Ci; Mike Zheng Shou
>
> **备注:** Project page: https://github.com/showlab/RobotSeg
>
> **摘要:** Accurate robot segmentation is a fundamental capability for robotic perception. It enables precise visual servoing for VLA systems, scalable robot-centric data augmentation, accurate real-to-sim transfer, and reliable safety monitoring in dynamic human-robot environments. Despite the strong capabilities of modern segmentation models, surprisingly it remains challenging to segment robots. This is due to robot embodiment diversity, appearance ambiguity, structural complexity, and rapid shape changes. Embracing these challenges, we introduce RobotSeg, a foundation model for robot segmentation in image and video. RobotSeg is built upon the versatile SAM 2 foundation model but addresses its three limitations for robot segmentation, namely the lack of adaptation to articulated robots, reliance on manual prompts, and the need for per-frame training mask annotations, by introducing a structure-enhanced memory associator, a robot prompt generator, and a label-efficient training strategy. These innovations collectively enable a structure-aware, automatic, and label-efficient solution. We further construct the video robot segmentation (VRS) dataset comprising over 2.8k videos (138k frames) with diverse robot embodiments and environments. Extensive experiments demonstrate that RobotSeg achieves state-of-the-art performance on both images and videos, establishing a strong foundation for future advances in robot perception.
>
---
#### [new 056] MG-Nav: Dual-Scale Visual Navigation via Sparse Spatial Memory
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MG-Nav，一种用于零样本视觉导航的双尺度框架。针对复杂场景中全局规划与局部控制的协同难题，构建稀疏空间记忆图（SMG）统一多视图语义与空间结构，结合图像-实例混合检索与几何适配模块，实现跨场景的精准导航与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22609v1](https://arxiv.org/pdf/2511.22609v1)**

> **作者:** Bo Wang; Jiehong Lin; Chenzhi Liu; Xinting Hu; Yifei Yu; Tianjia Liu; Zhongrui Wang; Xiaojuan Qi
>
> **备注:** 10pages, 5 figures
>
> **摘要:** We present MG-Nav (Memory-Guided Navigation), a dual-scale framework for zero-shot visual navigation that unifies global memory-guided planning with local geometry-enhanced control. At its core is the Sparse Spatial Memory Graph (SMG), a compact, region-centric memory where each node aggregates multi-view keyframe and object semantics, capturing both appearance and spatial structure while preserving viewpoint diversity. At the global level, the agent is localized on SMG and a goal-conditioned node path is planned via an image-to-instance hybrid retrieval, producing a sequence of reachable waypoints for long-horizon guidance. At the local level, a navigation foundation policy executes these waypoints in point-goal mode with obstacle-aware control, and switches to image-goal mode when navigating from the final node towards the visual target. To further enhance viewpoint alignment and goal recognition, we introduce VGGT-adapter, a lightweight geometric module built on the pre-trained VGGT model, which aligns observation and goal features in a shared 3D-aware space. MG-Nav operates global planning and local control at different frequencies, using periodic re-localization to correct errors. Experiments on HM3D Instance-Image-Goal and MP3D Image-Goal benchmarks demonstrate that MG-Nav achieves state-of-the-art zero-shot performance and remains robust under dynamic rearrangements and unseen scene conditions.
>
---
#### [new 057] Switching control of underactuated multi-channel systems with input constraints for cooperative manipulation
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究多智能体协作操纵任务，针对欠驱动、输入受限的多通道系统，提出事件触发的切换控制框架。通过混合整数线性规划实现通道分配与稳定控制，结合二次规划保证实时稳定性，理论证明了系统的半全局指数稳定性和渐近稳定性，并在二维、三维自由飞行器及多机器人推拉任务中验证了有效性。**

- **链接: [https://arxiv.org/pdf/2511.22810v1](https://arxiv.org/pdf/2511.22810v1)**

> **作者:** Dongjae Lee; Dimos V. Dimarogonas; H. Jin Kim
>
> **备注:** 14 pages
>
> **摘要:** This work presents an event-triggered switching control framework for a class of nonlinear underactuated multi-channel systems with input constraints. These systems are inspired by cooperative manipulation tasks involving underactuation, where multiple underactuated agents collaboratively push or pull an object to a target pose. Unlike existing approaches for multi-channel systems, our method addresses underactuation and the potential loss of controllability by additionally addressing channel assignment of agents. To simultaneously account for channel assignment, input constraints, and stabilization, we formulate the control problem as a Mixed Integer Linear Programming and derive sufficient conditions for its feasibility. To improve real-time computation efficiency, we introduce an event-triggered control scheme that maintains stability even between switching events through a quadratic programming-based stabilizing controller. We theoretically establish the semi-global exponential stability of the proposed method and the asymptotic stability of its extension to nonprehensile cooperative manipulation under noninstantaneous switching. The proposed framework is further validated through numerical simulations on 2D and 3D free-flyer systems and multi-robot nonprehensile pushing tasks.
>
---
#### [new 058] BiCQL-ML: A Bi-Level Conservative Q-Learning Framework for Maximum Likelihood Inverse Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出BiCQL-ML，一种无策略的离线逆强化学习方法，旨在仅用固定演示数据恢复奖励函数。通过双层框架联合优化奖励与保守Q函数，避免显式策略学习，实现软值匹配下的最大似然估计，理论保证专家策略为软最优，并在基准上提升奖励恢复与下游性能。**

- **链接: [https://arxiv.org/pdf/2511.22210v1](https://arxiv.org/pdf/2511.22210v1)**

> **作者:** Junsung Park
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Offline inverse reinforcement learning (IRL) aims to recover a reward function that explains expert behavior using only fixed demonstration data, without any additional online interaction. We propose BiCQL-ML, a policy-free offline IRL algorithm that jointly optimizes a reward function and a conservative Q-function in a bi-level framework, thereby avoiding explicit policy learning. The method alternates between (i) learning a conservative Q-function via Conservative Q-Learning (CQL) under the current reward, and (ii) updating the reward parameters to maximize the expected Q-values of expert actions while suppressing over-generalization to out-of-distribution actions. This procedure can be viewed as maximum likelihood estimation under a soft value matching principle. We provide theoretical guarantees that BiCQL-ML converges to a reward function under which the expert policy is soft-optimal. Empirically, we show on standard offline RL benchmarks that BiCQL-ML improves both reward recovery and downstream policy performance compared to existing offline IRL baselines.
>
---
#### [new 059] MrGS: Multi-modal Radiance Fields with 3D Gaussian Splatting for RGB-Thermal Novel View Synthesis
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MrGS，一种基于3D高斯点云的多模态辐射场方法，用于RGB-热红外图像的新型视图合成。针对现有方法忽略热传导与朗伯反射特性的问题，通过正交特征提取与物理定律建模，实现高保真多模态场景重建，并减少高斯点数量。**

- **链接: [https://arxiv.org/pdf/2511.22997v1](https://arxiv.org/pdf/2511.22997v1)**

> **作者:** Minseong Kweon; Janghyun Kim; Ukcheol Shin; Jinsun Park
>
> **备注:** Accepted at Thermal Infrared in Robotics (TIRO) Workshop, ICRA 2025 (Best Poster Award)
>
> **摘要:** Recent advances in Neural Radiance Fields (NeRFs) and 3D Gaussian Splatting (3DGS) have achieved considerable performance in RGB scene reconstruction. However, multi-modal rendering that incorporates thermal infrared imagery remains largely underexplored. Existing approaches tend to neglect distinctive thermal characteristics, such as heat conduction and the Lambertian property. In this study, we introduce MrGS, a multi-modal radiance field based on 3DGS that simultaneously reconstructs both RGB and thermal 3D scenes. Specifically, MrGS derives RGB- and thermal-related information from a single appearance feature through orthogonal feature extraction and employs view-dependent or view-independent embedding strategies depending on the degree of Lambertian reflectance exhibited by each modality. Furthermore, we leverage two physics-based principles to effectively model thermal-domain phenomena. First, we integrate Fourier's law of heat conduction prior to alpha blending to model intensity interpolation caused by thermal conduction between neighboring Gaussians. Second, we apply the Stefan-Boltzmann law and the inverse-square law to formulate a depth-aware thermal radiation map that imposes additional geometric constraints on thermal rendering. Experimental results demonstrate that the proposed MrGS achieves high-fidelity RGB-T scene reconstruction while reducing the number of Gaussians.
>
---
## 更新

#### [replaced 001] Quality-guided UAV Surface Exploration for 3D Reconstruction
- **分类: cs.RO**

- **简介: 该论文针对自主无人机三维重建中的探索规划问题，提出一种基于重建质量引导的新型NBV框架。通过利用TSDF表示的不确定性，自适应生成与选择视点，实现高效、高质量的环境覆盖。在仿真中验证了其在覆盖率、重建质量和路径效率上优于传统方法。**

- **链接: [https://arxiv.org/pdf/2511.20353v2](https://arxiv.org/pdf/2511.20353v2)**

> **作者:** Benjamin Sportich; Kenza Boubakri; Olivier Simonin; Alessandro Renzaglia
>
> **摘要:** Reasons for mapping an unknown environment with autonomous robots are wide-ranging, but in practice, they are often overlooked when developing planning strategies. Rapid information gathering and comprehensive structural assessment of buildings have different requirements and therefore necessitate distinct methodologies. In this paper, we propose a novel modular Next-Best-View (NBV) planning framework for aerial robots that explicitly uses a reconstruction quality objective to guide the exploration planning. In particular, our approach introduces new and efficient methods for view generation and selection of viewpoint candidates that are adaptive to the user-defined quality requirements, fully exploiting the uncertainty encoded in a Truncated Signed Distance field (TSDF) representation of the environment. This results in informed and efficient exploration decisions tailored towards the predetermined objective. Finally, we validate our method via extensive simulations in realistic environments. We demonstrate that it successfully adjusts its behavior to the user goal while consistently outperforming conventional NBV strategies in terms of coverage, quality of the final 3D map and path efficiency.
>
---
#### [replaced 002] Steady-State Drifting Equilibrium Analysis of Single-Track Two-Wheeled Robots for Controller Design
- **分类: cs.RO**

- **简介: 该论文研究单轨两轮机器人稳态漂移的平衡特性，旨在解决其在侧滑条件下运动稳定性控制难题。通过构建基于几何与运动学的解析模型，揭示了漂移机理并提出高效算法，显著降低计算量；进而设计模型预测控制器，实现稳定漂移及平衡点切换，验证了方法的有效性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2504.09134v2](https://arxiv.org/pdf/2504.09134v2)**

> **作者:** Feilong Jing; Yang Deng; Boyi Wang; Xudong Zheng; Yifan Sun; Zhang Chen; Bin Liang
>
> **备注:** Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Drifting is an advanced driving technique where the wheeled robot's tire-ground interaction breaks the common non-holonomic pure rolling constraint. This allows high-maneuverability tasks like quick cornering, and steady-state drifting control enhances motion stability under lateral slip conditions. While drifting has been successfully achieved in four-wheeled robot systems, its application to single-track two-wheeled (STTW) robots, such as unmanned motorcycles or bicycles, has not been thoroughly studied. To bridge this gap, this paper extends the drifting equilibrium theory to STTW robots and reveals the mechanism behind the steady-state drifting maneuver. Notably, the counter-steering drifting technique used by skilled motorcyclists is explained through this theory. In addition, an analytical algorithm based on intrinsic geometry and kinematics relationships is proposed, reducing the computation time by four orders of magnitude while maintaining less than 6% error compared to numerical methods. Based on equilibrium analysis, a model predictive controller (MPC) is designed to achieve steady-state drifting and equilibrium points transition, with its effectiveness and robustness validated through simulations.
>
---
#### [replaced 003] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对长时序、高记忆需求的具身智能体决策任务，提出Memo框架。通过在训练中插入周期性摘要标记，实现视觉输入的高效压缩与记忆检索，解决Transformer因上下文过长导致的效率与泛化问题。实验表明，Memo在多任务场景中优于基线模型，具备更强的推理效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.19732v2](https://arxiv.org/pdf/2510.19732v2)**

> **作者:** Gunshi Gupta; Karmesh Yadav; Zsolt Kira; Yarin Gal; Rahaf Aljundi
>
> **备注:** Accepted for Spotlight Presentation at NeurIPS 2025
>
> **摘要:** To enable embodied agents to operate effectively over extended timeframes, it is crucial to develop models that form and access memories to stay contextualized in their environment. In the current paradigm of training transformer-based policies for embodied sequential decision-making tasks, visual inputs often overwhelm the context limits of transformers, while humans can maintain and utilize a lifetime of experience compressed as memories. Significant compression is possible in principle, as much of the input is irrelevant and can be abstracted. However, existing approaches predominantly focus on either recurrent models with fixed-size memory or transformers with full-context reliance. In this work, we propose Memo, a transformer-based architecture and training recipe for reinforcement learning (RL) on memory-intensive, long-horizon tasks. Memo incorporates the creation and retrieval of memory by interleaving periodic summarization tokens with the inputs of a model during training. We demonstrate Memo's effectiveness on a gridworld meta-RL benchmark and a multi-object navigation task in photo-realistic indoor settings. Memo outperforms naive long-context transformer baselines while being more compute and storage efficient. Additionally, Memo generalizes better to longer contexts at inference time and remains robust in streaming settings, where historical context must be truncated to fit inference constraints. Our code is available at: https://github.com/gunshi/memo.
>
---
#### [replaced 004] Robot-mediated physical Human-Human Interaction in Neurorehabilitation: a position paper
- **分类: cs.RO**

- **简介: 该论文提出“机器人中介的人-人物理交互”新范式，旨在融合物理治疗师的临床经验与机器人的精准可控性。针对传统康复中机器人缺乏人性化适应性的问题，研究构建统一分类体系、基于社会心理学的交互框架及无缝协作技术，推动康复机器人与人工治疗的深度融合。**

- **链接: [https://arxiv.org/pdf/2507.17561v2](https://arxiv.org/pdf/2507.17561v2)**

> **作者:** Lorenzo Vianello; Matthew Short; Julia Manczurowsky; Emek Barış Küçüktabak; Francesco Di Tommaso; Alessia Noccaro; Laura Bandini; Shoshana Clark; Alaina Fiorenza; Francesca Lunardini; Alberto Canton; Marta Gandolla; Alessandra L. G. Pedrocchi; Emilia Ambrosini; Manuel Murie-Fernandez; Carmen B. Roman; Jesus Tornero; Natacha Leon; Andrew Sawers; Jim Patton; Domenico Formica; Nevio Luigi Tagliamonte; Georg Rauter; Kilian Baur; Fabian Just; Christopher J. Hasson; Vesna D. Novak; Jose L. Pons
>
> **备注:** Accepted in IEEE Reviews in Biomedical Engineering
>
> **摘要:** Neurorehabilitation conventionally relies on the interaction between a patient and a physical therapist. Robotic systems can improve and enrich the physical feedback provided to patients after neurological injury, but they under-utilize the adaptability and clinical expertise of trained therapists. In this position paper, we advocate for a novel approach that integrates the therapist's clinical expertise and nuanced decision-making with the strength, accuracy, and repeatability of robotics: Robot-mediated physical Human-Human Interaction. This framework, which enables two individuals to physically interact through robotic devices, has been studied across diverse research groups and has recently emerged as a promising link between conventional manual therapy and rehabilitation robotics, harmonizing the strengths of both approaches. This paper presents the rationale of a multidisciplinary team-including engineers, doctors, and physical therapists-for conducting research that utilizes: a unified taxonomy to describe robot-mediated rehabilitation, a framework of interaction based on social psychology, and a technological approach that makes robotic systems seamless facilitators of natural human-human interaction.
>
---
#### [replaced 005] Efficient Learning of Object Placement with Intra-Category Transfer
- **分类: cs.RO**

- **简介: 该论文研究机器人长时序任务的高效学习问题，针对示范学习样本效率低的挑战，提出基于类别内迁移的对象布局学习方法。通过在标准类别框架上建模对象排列，仅需5个示范即可泛化至多种物体，实现如摆桌、整理办公桌等任务的高效学习与鲁棒执行。**

- **链接: [https://arxiv.org/pdf/2411.03408v2](https://arxiv.org/pdf/2411.03408v2)**

> **作者:** Adrian Röfer; Russell Buchanan; Max Argus; Sethu Vijayakumar; Abhinav Valada
>
> **备注:** 12 pages, 8 figures, 3 tables, accepted at RA-L November 2025
>
> **摘要:** Efficient learning from demonstration for long-horizon tasks remains an open challenge in robotics. While significant effort has been directed toward learning trajectories, a recent resurgence of object-centric approaches has demonstrated improved sample efficiency, enabling transferable robotic skills. Such approaches model tasks as a sequence of object poses over time. In this work, we propose a scheme for transferring observed object arrangements to novel object instances by learning these arrangements on canonical class frames. We then employ this scheme to enable a simple yet effective approach for training models from as few as five demonstrations to predict arrangements of a wide range of objects including tableware, cutlery, furniture, and desk spaces. We propose a method for optimizing the learned models to enable efficient learning of tasks such as setting a table or tidying up an office with intra-category transfer, even in the presence of distractors. We present extensive experimental results in simulation and on a real robotic system for table setting which, based on human evaluations, scored 73.3% compared to a human baseline. We make the code and trained models publicly available at https://oplict.cs.uni-freiburg.de.
>
---
#### [replaced 006] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对基于大模型的物体目标导航任务，解决其依赖云端大模型、本地部署效率低的问题。提出EfficientNav，通过语义感知记忆检索与离散缓存机制，提升小模型对导航地图的理解能力并降低推理延迟，实现高效本地化零样本导航。**

- **链接: [https://arxiv.org/pdf/2510.18546v2](https://arxiv.org/pdf/2510.18546v2)**

> **作者:** Zebin Yang; Sunjian Zheng; Tong Xie; Tianshi Xu; Bo Yu; Fan Wang; Jie Tang; Shaoshan Liu; Meng Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Object-goal navigation (ObjNav) tasks an agent with navigating to the location of a specific object in an unseen environment. Embodied agents equipped with large language models (LLMs) and online constructed navigation maps can perform ObjNav in a zero-shot manner. However, existing agents heavily rely on giant LLMs on the cloud, e.g., GPT-4, while directly switching to small LLMs, e.g., LLaMA3.2-11b, suffer from significant success rate drops due to limited model capacity for understanding complex navigation maps, which prevents deploying ObjNav on local devices. At the same time, the long prompt introduced by the navigation map description will cause high planning latency on local devices. In this paper, we propose EfficientNav to enable on-device efficient LLM-based zero-shot ObjNav. To help the smaller LLMs better understand the environment, we propose semantics-aware memory retrieval to prune redundant information in navigation maps. To reduce planning latency, we propose discrete memory caching and attention-based memory clustering to efficiently save and re-use the KV cache. Extensive experimental results demonstrate that EfficientNav achieves 11.1% improvement in success rate on HM3D benchmark over GPT-4-based baselines, and demonstrates 6.7x real-time latency reduction and 4.7x end-to-end latency reduction over GPT-4 planner. Our code is available on https://github.com/PKU-SEC-Lab/EfficientNav.
>
---
#### [replaced 007] The Role of Consequential and Functional Sound in Human-Robot Interaction: Toward Audio Augmented Reality Interfaces
- **分类: cs.RO**

- **简介: 该论文研究人机交互中的因果性与功能性声音作用，旨在提升机器人语音交互体验。通过实验分析声音对感知与行为的影响，探索空间声学在任务传递中的应用，提出音频增强现实接口设计策略，以改善协作效率与用户感受。**

- **链接: [https://arxiv.org/pdf/2511.15956v2](https://arxiv.org/pdf/2511.15956v2)**

> **作者:** Aliyah Smith; Monroe Kennedy
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** As robots become increasingly integrated into everyday environments, understanding how they communicate with humans is critical. Sound offers a powerful channel for interaction, encompassing both operational noises and intentionally designed auditory cues. In this study, we examined the effects of consequential and functional sounds on human perception and behavior, including a novel exploration of spatial sound through localization and handover tasks. Results show that consequential sounds of the Kinova Gen3 manipulator did not negatively affect perceptions, spatial localization is highly accurate for lateral cues but declines for frontal cues, and spatial sounds can simultaneously convey task-relevant information while promoting warmth and reducing discomfort. These findings highlight the potential of functional and transformative auditory design to enhance human-robot collaboration and inform future sound-based interaction strategies.
>
---
#### [replaced 008] Scalable Multisubject Vital Sign Monitoring With mmWave FMCW Radar and FPGA Prototyping
- **分类: eess.SY; cs.RO; eess.SP**

- **简介: 该论文提出一种基于毫米波FMCW雷达与FPGA的多人体征非接触监测系统，旨在解决传统接触式监测带来的不适、感染风险及校准难题。通过硬件加速实现多目标实时监测，显著提升处理速度与资源效率，验证了系统在复杂场景下的可扩展性与实用性。**

- **链接: [https://arxiv.org/pdf/2511.21314v2](https://arxiv.org/pdf/2511.21314v2)**

> **作者:** Jewel Benny; Narahari N. Moudhgalya; Mujeev Khan; Hemant Kumar Meena; Mohd Wajid; Abhishek Srivastava
>
> **备注:** Published in IEEE Sensors Journal
>
> **摘要:** In this work, we introduce an innovative approach to estimate the vital signs of multiple human subjects simultaneously in a non-contact way using a Frequency Modulated Continuous Wave (FMCW) radar-based system. Traditional vital sign monitoring methods often face significant limitations, including subject discomfort with wearable devices, challenges in calibration, and the risk of infection transmission through contact measurement devices. To address these issues, this research is motivated by the need for versatile, non-contact vital monitoring solutions applicable in various critical scenarios. This work also explores the challenges of extending this capability to an arbitrary number of subjects, including hardware and theoretical limitations. Supported by rigorous experimental results and discussions, the paper illustrates the system's potential to redefine vital sign monitoring. An FPGA-based implementation is also presented as proof of concept for a hardware-based and portable solution, improving upon previous works by offering 2.7x faster execution and 18.4% less Look-Up Table (LUT) utilization, as well as providing over 7400x acceleration compared to its software counterpart.
>
---
#### [replaced 009] Vectorized Online POMDP Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对自主机器人在部分可观测环境下的规划问题，提出一种名为VOPP的向量化在线POMDP求解器。通过将规划数据结构转化为张量并实现全向量化计算，消除了传统方法中的依赖与同步瓶颈，显著提升并行效率，实验表明其计算效率比现有先进方法高20倍以上。**

- **链接: [https://arxiv.org/pdf/2510.27191v2](https://arxiv.org/pdf/2510.27191v2)**

> **作者:** Marcus Hoerger; Muhammad Sudrajat; Hanna Kurniawati
>
> **备注:** 9 pages, 3 figures. Submitted to ICRA 2026
>
> **摘要:** Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization of today's hardware, but parallelizing POMDP solvers has been challenging. They rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can quickly offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation that analytically solves part of the optimization component, leaving only the estimation of expectations for numerical computation. VOPP represents all data structures related to planning as a collection of tensors and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel solver with no dependencies and synchronization bottlenecks between parallel computations. Experimental results indicate that VOPP is at least 20X more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver.
>
---
#### [replaced 010] Periodic Skill Discovery
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出周期性技能发现（PSD）框架，针对强化学习中无监督技能发现忽略周期性行为的问题，通过将状态映射到环形潜在空间，自然编码周期性。方法能有效学习多样化周期技能，适用于复杂机器人任务，提升下游任务表现，扩展智能体行为多样性。**

- **链接: [https://arxiv.org/pdf/2511.03187v3](https://arxiv.org/pdf/2511.03187v3)**

> **作者:** Jonghae Park; Daesol Cho; Jusuk Lee; Dongseok Shim; Inkyu Jang; H. Jin Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Unsupervised skill discovery in reinforcement learning (RL) aims to learn diverse behaviors without relying on external rewards. However, current methods often overlook the periodic nature of learned skills, focusing instead on increasing the mutual dependence between states and skills or maximizing the distance traveled in latent space. Considering that many robotic tasks - particularly those involving locomotion - require periodic behaviors across varying timescales, the ability to discover diverse periodic skills is essential. Motivated by this, we propose Periodic Skill Discovery (PSD), a framework that discovers periodic behaviors in an unsupervised manner. The key idea of PSD is to train an encoder that maps states to a circular latent space, thereby naturally encoding periodicity in the latent representation. By capturing temporal distance, PSD can effectively learn skills with diverse periods in complex robotic tasks, even with pixel-based observations. We further show that these learned skills achieve high performance on downstream tasks such as hurdling. Moreover, integrating PSD with an existing skill discovery method offers more diverse behaviors, thus broadening the agent's repertoire. Our code and demos are available at https://jonghaepark.github.io/psd/
>
---
#### [replaced 011] Underactuated Robotic Hand with Grasp State Estimation Using Tendon-Based Proprioception
- **分类: cs.RO**

- **简介: 该论文研究机器人手抓握状态估计任务，针对欠驱动机械手因关节耦合导致状态感知困难的问题，提出仅通过腱式本体感觉实现多变量状态估计。利用集成于手指的高精度串行弹性执行器，结合势能模型，实现了接触时机、关节角、物体刚度及外界扰动的准确估计，验证了该方法在无视觉或触觉依赖下的有效性。**

- **链接: [https://arxiv.org/pdf/2509.12969v2](https://arxiv.org/pdf/2509.12969v2)**

> **作者:** Jae-Hyun Lee; Jonghoo Park; Kyu-Jin Cho
>
> **备注:** 11 pages, 15 figures, 3 tables, Supplementary video
>
> **摘要:** Anthropomorphic underactuated hands are valued for their structural simplicity and inherent adaptability. However, the uncertainty arising from interdependent joint motions makes it challenging to capture various grasp states during hand-object interaction without increasing structural complexity through multiple embedded sensors. This motivates the need for an approach that can extract rich grasp-state information from a single sensing source while preserving the simplicity of underactuation. This study proposes an anthropomorphic underactuated hand that achieves comprehensive grasp state estimation, using only tendon-based proprioception provided by series elastic actuators (SEAs). Our approach is enabled by the design of a compact SEA with high accuracy and reliability that can be seamlessly integrated into sensorless fingers. By coupling accurate proprioceptive measurements with potential energy-based modeling, the system estimates multiple key grasp state variables, including contact timing, joint angles, relative object stiffness, and external disturbances. Finger-level experimental validations and extensive hand-level grasp functionality demonstrations confirmed the effectiveness of the proposed approach. These results highlight tendon-based proprioception as a compact and robust sensing modality for practical manipulation without reliance on vision or tactile feedback.
>
---
#### [replaced 012] Scaling Spatial Intelligence with Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文聚焦多模态基础模型的空间智能提升任务，针对现有模型在空间理解上的不足，构建了包含800万样本的SenseNova-SI数据集，通过系统性数据构建与训练，显著提升模型在多项空间智能基准上的表现，并探索了数据规模、泛化能力及推理机制，推动多模态模型向更强空间认知发展。**

- **链接: [https://arxiv.org/pdf/2511.13719v2](https://arxiv.org/pdf/2511.13719v2)**

> **作者:** Zhongang Cai; Ruisi Wang; Chenyang Gu; Fanyi Pu; Junxiang Xu; Yubo Wang; Wanqi Yin; Zhitao Yang; Chen Wei; Qingping Sun; Tongxi Zhou; Jiaqi Li; Hui En Pang; Oscar Qian; Yukun Wei; Zhiqian Lin; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Xiangyu Fan; Hanming Deng; Lewei Lu; Liang Pan; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/OpenSenseNova/SenseNova-SI; Models: https://huggingface.co/collections/sensenova/sensenova-si
>
> **摘要:** Despite remarkable progress, multimodal foundation models still exhibit surprising deficiencies in spatial intelligence. In this work, we explore scaling up multimodal foundation models to cultivate spatial intelligence within the SenseNova-SI family, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel). We take a principled approach to constructing high-performing and robust spatial intelligence by systematically curating SenseNova-SI-8M: eight million diverse data samples under a rigorous taxonomy of spatial capabilities. SenseNova-SI demonstrates unprecedented performance across a broad range of spatial intelligence benchmarks: 68.7% on VSI-Bench, 43.3% on MMSI, 85.6% on MindCube, 54.6% on ViewSpatial, and 50.1% on SITE, while maintaining strong general multimodal understanding (e.g., 84.9% on MMBench-En). More importantly, we analyze the impact of data scaling, discuss early signs of emergent generalization capabilities enabled by diverse data training, analyze the risk of overfitting and language shortcuts, present a preliminary study on spatial chain-of-thought reasoning, and validate the potential downstream application. SenseNova-SI is an ongoing project, and this report will be updated continuously. All newly trained multimodal foundation models are publicly released to facilitate further research in this direction.
>
---
#### [replaced 013] Trajectory Optimization for In-Hand Manipulation with Tactile Force Control
- **分类: cs.RO**

- **简介: 该论文研究机器人手部操作中小物体的轨迹优化问题，旨在提升抓取与滚动的精度与鲁棒性。针对微型磁力触觉传感器集成带来的状态估计挑战，提出基于非线性规划的轨迹优化框架，并结合力控与状态估计算法，显著提升滚动成功率（提高30%）。**

- **链接: [https://arxiv.org/pdf/2503.08222v2](https://arxiv.org/pdf/2503.08222v2)**

> **作者:** Haegu Lee; Yitaek Kim; Victor Melbye Staven; Christoffer Sloth
>
> **备注:** This paper has been accepted to IROS 2025
>
> **摘要:** The strength of the human hand lies in its ability to manipulate small objects precisely and robustly. In contrast, simple robotic grippers have low dexterity and fail to handle small objects effectively. This is why many automation tasks remain unsolved by robots. This paper presents an optimization-based framework for in-hand manipulation with a robotic hand equipped with compact Magnetic Tactile Sensors (MTSs). The small form factor of the robotic hand from Shadow Robot introduces challenges in estimating the state of the object while satisfying contact constraints. To address this, we formulate a trajectory optimization problem using Nonlinear Programming (NLP) for finger movements while ensuring contact points to change along the geometry of the fingers. Using the optimized trajectory from the solver, we implement and test an open-loop controller for rolling motion. To further enhance robustness and accuracy, we introduce a force controller for the fingers and a state estimator for the object utilizing MTSs. The proposed framework is validated through comparative experiments, showing that incorporating the force control with compliance consideration improves the accuracy and robustness of the rolling motion. Rolling an object with the force controller is 30\% more likely to succeed than running an open-loop controller. The demonstration video is available at https://youtu.be/6J_muL_AyE8.
>
---
#### [replaced 014] BactoBot: A Low-Cost, Bacteria-Inspired Soft Underwater Robot for Marine Exploration
- **分类: cs.RO**

- **简介: 该论文提出BactoBot，一种低成本、仿细菌鞭毛推进的软体水下机器人，旨在解决传统刚性潜水器对海洋生态的破坏问题。通过柔性硅胶臂与防水旋转轴设计，实现安全探索，采用DIY方法降低造价至355美元，验证了软体机器人在资源受限环境下的可行性。**

- **链接: [https://arxiv.org/pdf/2509.20964v2](https://arxiv.org/pdf/2509.20964v2)**

> **作者:** Rubaiyat Tasnim Chowdhury; Nayan Bala; Ronojoy Roy; Tarek Mahmud
>
> **备注:** Revised version. Updated literature review, improved figures, and added clarification on waterproofing methodology
>
> **摘要:** Traditional rigid underwater vehicles pose risks to delicate marine ecosystems due to high-speed propellers and rigid hulls. This paper presents BactoBot, a low-cost, soft underwater robot designed for safe and gentle marine exploration. Inspired by the efficient flagellar propulsion of bacteria, BactoBot features 12 flexible, silicone-based arms arranged on a dodecahedral frame. Unlike high-cost research platforms, this prototype was fabricated using accessible DIY methods, including food-grade silicone molding, FDM 3D printing, and off-the-shelf DC motors. A novel multi-stage waterproofing protocol was developed to seal rotating shafts using a grease-filled chamber system, ensuring reliability at low cost. The robot was successfully tested in a controlled aquatic environment, demonstrating stable forward propulsion and turning maneuvers. With a total fabrication cost of approximately $355 USD, this project validates the feasibility of democratizing soft robotics for marine science in resource-constrained settings.
>
---
#### [replaced 015] Advancing Embodied Intelligence in Robotic-Assisted Endovascular Procedures: A Systematic Review of AI Solutions
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于医疗机器人与人工智能交叉领域，旨在解决内血管手术中人工操作精度低、疲劳和辐射暴露等问题。通过系统综述AI驱动的具身智能技术在机器人辅助内血管手术中的应用，分析其关键技术与挑战，提出以增强临床决策为核心的未来发展方向。**

- **链接: [https://arxiv.org/pdf/2504.15327v3](https://arxiv.org/pdf/2504.15327v3)**

> **作者:** Tianliang Yao; Bo Lu; Markus Kowarschik; Yixuan Yuan; Hubin Zhao; Sebastien Ourselin; Kaspar Althoefer; Junbo Ge; Peng Qi
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Endovascular procedures have revolutionized vascular disease treatment, yet their manual execution is challenged by the demands for high precision, operator fatigue, and radiation exposure. Robotic systems have emerged as transformative solutions to mitigate these inherent limitations. A pivotal moment has arrived, where a confluence of pressing clinical needs and breakthroughs in AI creates an opportunity for a paradigm shift toward Embodied Intelligence (EI), enabling robots to navigate complex vascular networks and adapt to dynamic physiological conditions. Data-driven approaches, leveraging advanced computer vision, medical image analysis, and machine learning, drive this evolution by enabling real-time vessel segmentation, device tracking, and anatomical landmark detection. Reinforcement learning and imitation learning further enhance navigation strategies and replicate expert techniques. This review systematically analyzes the integration of EI into endovascular robotics, identifying profound systemic challenges such as the heterogeneity in validation standards and the gap between human mimicry and machine-native capabilities. Based on this analysis, a conceptual roadmap is proposed that reframes the ultimate objective away from systems that supplant clinical decision-making. This vision of augmented intelligence, where the clinician's role evolves into that of a high-level supervisor, provides a principled foundation for the future of the field.
>
---
#### [replaced 016] Material-informed Gaussian Splatting for 3D World Reconstruction in a Digital Twin
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对数字孪生中3D场景重建任务，解决传统LiDAR-camera融合方法依赖复杂校准、难以表征玻璃等材料的问题。提出仅用摄像头的重建方法：基于多视角图像使用高斯溅射建模，结合视觉模型提取材质掩码，将材质标签投影至网格，并赋予物理材质属性，实现高保真传感器模拟。**

- **链接: [https://arxiv.org/pdf/2511.20348v2](https://arxiv.org/pdf/2511.20348v2)**

> **作者:** Andy Huynh; João Malheiro Silva; Holger Caesar; Tong Duy Son
>
> **备注:** 8 pages, 5 figures. Submitted to IEEE Intelligent Vehicles Symposium (IV) 2026 for possible publication. Revised version (v2) to correct author order
>
> **摘要:** 3D reconstruction for Digital Twins often relies on LiDAR-based methods, which provide accurate geometry but lack the semantics and textures naturally captured by cameras. Traditional LiDAR-camera fusion approaches require complex calibration and still struggle with certain materials like glass, which are visible in images but poorly represented in point clouds. We propose a camera-only pipeline that reconstructs scenes using 3D Gaussian Splatting from multi-view images, extracts semantic material masks via vision models, converts Gaussian representations to mesh surfaces with projected material labels, and assigns physics-based material properties for accurate sensor simulation in modern graphics engines and simulators. This approach combines photorealistic reconstruction with physics-based material assignment, providing sensor simulation fidelity comparable to LiDAR-camera fusion while eliminating hardware complexity and calibration requirements. We validate our camera-only method using an internal dataset from an instrumented test vehicle, leveraging LiDAR as ground truth for reflectivity validation alongside image similarity metrics.
>
---
#### [replaced 017] Enhancing Kinematic Performances of Soft Continuum Robots for Magnetic Actuation
- **分类: cs.RO**

- **简介: 该论文针对软连续体机器人在磁驱动下的运动性能优化问题，提出融合平衡计算与运动学性能的通用框架。通过分析黎曼雅可比谱，建立结构参数、驱动输入与构型几何的全局关联，并设计双层优化算法，实现复杂磁场下的最优磁体布局与结构设计。**

- **链接: [https://arxiv.org/pdf/2507.10950v2](https://arxiv.org/pdf/2507.10950v2)**

> **作者:** Zhiwei Wu; Jiahao Luo; Siyi Wei; Jinhui Zhang
>
> **摘要:** Soft continuum robots achieve complex deformation through elastic equilibrium, making their reachable motions governed jointly by structural design and actuation-induced mechanics. This work develops a general formulation that integrates equilibrium computation with kinematic performances by evaluating Riemannian Jacobian spectra on the equilibrium manifold shaped by internal/external loading. The resulting framework yields a global performance functional that directly links structural parameters, actuation inputs, and the induced configuration space geometry. We apply this general framework to magnetic actuation. Analytical characterization is obtained under weak uniform fields, revealing optimal placement and orientation of the embedded magnet with invariant scale properties. To address nonlinear deformation and spatially varying fields, a two-level optimization algorithm is developed that alternates between energy based equilibrium search and gradient based structural updates. Simulations and physical experiments across uniform field, dipole field, and multi-magnet configurations demonstrate consistent structural tendencies: aligned moments favor distal or mid-distal solutions through constructive torque amplification, whereas opposing moments compress optimal designs toward proximal regions due to intrinsic cancellation zones.
>
---
#### [replaced 018] Holistic Evaluation of Multimodal LLMs on Spatial Intelligence
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **简介: 该论文聚焦多模态大模型的空间智能（SI）评估，提出EASI框架，统一现有与新构建的时空任务基准。通过在八项基准上超十亿令牌的实测，揭示当前顶尖模型（如GPT-5）虽强但仍远逊于人类，且非开源模型无显著优势。研究开放代码与排行榜，推动可复现、持续更新的SI评估。**

- **链接: [https://arxiv.org/pdf/2508.13142v4](https://arxiv.org/pdf/2508.13142v4)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Oscar Qian; Hui En Pang; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/EvolvingLMMs-Lab/EASI/; Leaderboard: https://huggingface.co/spaces/lmms-lab-si/EASI-Leaderboard
>
> **摘要:** Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence (SI). We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a growing collection of newly curated ones, enabling systematic evaluation of state-of-the-art models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in SI, yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail the most advanced multimodal models. EASI is an ongoing community effort: we have open-sourced the EASI codebase that provides a one-stop and reproducible solution with standardized interfaces, integrated protocols and prompts that significantly reduce the friction of configuring and running multiple benchmarks; we have also launched an accompanying EASI leaderboard to provide a continually updated snapshot of model performance across the full SI spectrum, accelerating collective progress toward robust SI.
>
---
#### [replaced 019] Heuristic Step Planning for Learning Dynamic Bipedal Locomotion: A Comparative Study of Model-Based and Model-Free Approaches
- **分类: cs.RO**

- **简介: 该论文研究学习型双足行走的步态规划问题，旨在实现复杂环境下的稳定行走。提出一种基于启发式步态规划与速度跟踪的框架，避免使用复杂的动力学模型。通过对比模型基方法，验证了其在不平地形上的鲁棒性与能效优势。**

- **链接: [https://arxiv.org/pdf/2511.00840v2](https://arxiv.org/pdf/2511.00840v2)**

> **作者:** William Suliman; Ekaterina Chaikovskaia; Egor Davydenko; Roman Gorbachev
>
> **摘要:** This work presents an extended framework for learning-based bipedal locomotion that incorporates a heuristic step-planning strategy guided by desired torso velocity tracking. The framework enables precise interaction between a humanoid robot and its environment, supporting tasks such as crossing gaps and accurately approaching target objects. Unlike approaches based on full or simplified dynamics, the proposed method avoids complex step planners and analytical models. Step planning is primarily driven by heuristic commands, while a Raibert-type controller modulates the foot placement length based on the error between desired and actual torso velocity. We compare our method with a model-based step-planning approach -- the Linear Inverted Pendulum Model (LIPM) controller. Experimental results demonstrate that our approach attains comparable or superior accuracy in maintaining target velocity (up to 80%), significantly greater robustness on uneven terrain (over 50% improvement), and improved energy efficiency. These results suggest that incorporating complex analytical, model-based components into the training architecture may be unnecessary for achieving stable and robust bipedal walking, even in unstructured environments.
>
---
#### [replaced 020] SlotVLA: Towards Modeling of Object-Relation Representations in Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中视觉表示效率与可解释性不足的问题，提出基于对象-关系的紧凑表征。构建LIBERO+数据集，提供细粒度对象标注；设计SlotVLA框架，利用槽注意力实现对象与关系的联合建模，显著减少视觉令牌数，提升泛化能力，推动可解释的多任务机器人操作。**

- **链接: [https://arxiv.org/pdf/2511.06754v2](https://arxiv.org/pdf/2511.06754v2)**

> **作者:** Taisei Hanyu; Nhat Chung; Huy Le; Toan Nguyen; Yuki Ikebe; Anthony Gunderman; Duy Nguyen Ho Minh; Khoa Vo; Tung Kieu; Kashu Yamazaki; Chase Rainwater; Anh Nguyen; Ngan Le
>
> **备注:** under review
>
> **摘要:** Inspired by how humans reason over discrete objects and their relationships, we explore whether compact object-centric and object-relation representations can form a foundation for multitask robotic manipulation. Most existing robotic multitask models rely on dense embeddings that entangle both object and background cues, raising concerns about both efficiency and interpretability. In contrast, we study object-relation-centric representations as a pathway to more structured, efficient, and explainable visuomotor control. Our contributions are two-fold. First, we introduce LIBERO+, a fine-grained benchmark dataset designed to enable and evaluate object-relation reasoning in robotic manipulation. Unlike prior datasets, LIBERO+ provides object-centric annotations that enrich demonstrations with box- and mask-level labels as well as instance-level temporal tracking, supporting compact and interpretable visuomotor representations. Second, we propose SlotVLA, a slot-attention-based framework that captures both objects and their relations for action decoding. It uses a slot-based visual tokenizer to maintain consistent temporal object representations, a relation-centric decoder to produce task-relevant embeddings, and an LLM-driven module that translates these embeddings into executable actions. Experiments on LIBERO+ demonstrate that object-centric slot and object-relation slot representations drastically reduce the number of required visual tokens, while providing competitive generalization. Together, LIBERO+ and SlotVLA provide a compact, interpretable, and effective foundation for advancing object-relation-centric robotic manipulation.
>
---
#### [replaced 021] Safe Multi-Robotic Arm Interaction via 3D Convex Shapes
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对多机械臂在共享空间中易发生碰撞的安全问题，提出基于高阶控制屏障函数（HOCBFs）的在线避障方法。通过构建3D凸形状的安全约束，设计集中式与分布式安全滤波器，结合数值微分降低计算开销，实现多臂安全协作。验证了方法在仿真与真实机器人实验中的有效性。**

- **链接: [https://arxiv.org/pdf/2503.11791v3](https://arxiv.org/pdf/2503.11791v3)**

> **作者:** Ali Umut Kaypak; Shiqing Wei; Prashanth Krishnamurthy; Farshad Khorrami
>
> **备注:** The formal publication is available at DOI: https://doi.org/10.1016/j.robot.2025.105263
>
> **摘要:** Inter-robot collisions pose a significant safety risk when multiple robotic arms operate in close proximity. We present an online collision avoidance methodology leveraging High-Order Control Barrier Functions (HOCBFs) constructed for safe interactions among 3D convex shapes to address this issue. While prior works focused on using Control Barrier Functions (CBFs) for human-robotic arm and single-arm collision avoidance, we explore the problem of collision avoidance between multiple robotic arms operating in a shared space. In our methodology, we utilize the proposed HOCBFs as centralized and decentralized safety filters. These safety filters are compatible with many nominal controllers and ensure safety without significantly restricting the robots' workspace. A key challenge in implementing these filters is the computational overhead caused by the large number of safety constraints and the computation of a Hessian matrix per constraint. We address this challenge by employing numerical differentiation methods to approximate computationally intensive terms. The effectiveness of our method is demonstrated through extensive simulation studies and real-world experiments with Franka Research 3 robotic arms. The project video is available at this link.
>
---
#### [replaced 022] ArtiBench and ArtiBrain: Benchmarking Generalizable Vision-Language Articulated Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉-语言引导的可动物体操作中的泛化难题，提出ArtiBench基准与ArtiBrain框架。任务为跨类别、跨实例的长程多步操作。工作包括构建多环境、多层级评估基准，设计融合高层推理与自适应控制的模块化系统，并通过显式记忆传播可操作性知识，显著提升泛化能力与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.20330v2](https://arxiv.org/pdf/2511.20330v2)**

> **作者:** Yuhan Wu; Tiantian Wei; Shuo Wang; ZhiChao Wang; Yanyong Zhang; Daniel Cremers; Yan Xia
>
> **摘要:** Interactive articulated manipulation requires long-horizon, multi-step interactions with appliances while maintaining physical consistency. Existing vision-language and diffusion-based policies struggle to generalize across parts, instances, and categories. We first introduce ArtiBench, a five-level benchmark covering kitchen, storage, office, and tool environments. ArtiBench enables structured evaluation from cross-part and cross-instance variation to long-horizon multi-object tasks, revealing the core generalization challenges of articulated object manipulation. Building on this benchmark, we propose ArtiBrain, a modular framework that unifies high-level reasoning with adaptive low-level control. ArtiBrain uses a VLM-based Task Reasoner (GPT-4.1) to decompose and validate subgoals, and employs a Hybrid Controller that combines geometry-aware keyframe execution with affordance-guided diffusion for precise and interpretable manipulation. An Affordance Memory Bank continually accumulates successful execution episodes and propagates part-level actionable affordances to unseen articulated parts and configurations. Extensive experiments on ArtiBench show that our ArtiBrain significantly outperforms state-of-the-art multimodal and diffusion-based methods in robustness and generalization. Code and dataset will be released upon acceptance.
>
---
#### [replaced 023] Real-Time Obstacle Avoidance for a Mobile Robot Using CNN-Based Sensor Fusion
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对移动机器人实时避障任务，提出基于CNN的多传感器融合方法，利用RGB-D相机同步图像训练三个CNN模型，实现从视觉输入到低层转向指令的端到端映射。实验表明，NetConEmb模型在未知环境中表现最优，成功率达100%，验证了其有效性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.08095v2](https://arxiv.org/pdf/2509.08095v2)**

> **作者:** Lamiaa H. Zain
>
> **摘要:** Obstacle avoidance is a critical component of the navigation stack required for mobile robots to operate effectively in complex and unknown environments. In this research, three end-to-end Convolutional Neural Networks (CNNs) were trained and evaluated offline and deployed on a differential-drive mobile robot for real-time obstacle avoidance to generate low-level steering commands from synchronized color and depth images acquired by an Intel RealSense D415 RGB-D camera in diverse environments. Offline evaluation showed that the NetConEmb model achieved the best performance with a notably low MedAE of $0.58 \times 10^{-3}$ rad/s. In comparison, the lighter NetEmb architecture, which reduces the number of trainable parameters by approximately 25\% and converges faster, produced comparable results with an RMSE of $21.68 \times 10^{-3}$ rad/s, close to the $21.42 \times 10^{-3}$ rad/s obtained by NetConEmb. Real-time navigation further confirmed NetConEmb's robustness, achieving a 100\% success rate in both known and unknown environments, while NetEmb and NetGated succeeded only in navigating the known environment.
>
---
#### [replaced 024] A K-means Inspired Solution Framework for Large-Scale Multi-Traveling Salesman Problems
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对大规模多旅行商问题（MTSP）的计算复杂度高难题，提出一种受K-means启发的任务分配框架。通过将MTSP重构为空间约束分类问题，利用空间一致性实现快速路径成本估算与高效任务分组，显著降低计算开销。实验表明，该方法在千级代理、万级目标场景下仍能保持高质量解，有效支持大规模无人系统协同。**

- **链接: [https://arxiv.org/pdf/2511.19454v2](https://arxiv.org/pdf/2511.19454v2)**

> **作者:** Xiubin Chen
>
> **摘要:** The Multi-Traveling Salesman Problem (MTSP) is a commonly used mathematical model for multi-agent task allocation. However, as the number of agents and task targets increases, existing optimization-based methods often incur prohibitive computational costs, posing significant challenges to large-scale coordination in unmanned systems. To address this issue, this paper proposes a K-means-inspired task allocation framework that reformulates the MTSP as a spatially constrained classification process. By leveraging spatial coherence, the proposed method enables fast estimation of path costs and efficient task grouping, thereby fundamentally reducing overall computational complexity. Extensive simulation results demonstrate that the framework can maintain high solution quality even in extremely large-scale scenarios-for instance, in tasks involving 1000 agents and 5000 targets. The findings indicate that this "cluster-then-route" decomposition strategy offers an efficient and reliable solution for large-scale multi-agent task allocation.
>
---
#### [replaced 025] MonoDream: Monocular Vision-Language Navigation with Panoramic Dreaming
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对单目视觉语言导航（VLN）中因缺乏全景信息导致性能受限的问题，提出MonoDream框架。通过构建统一导航表征（UNR）并引入潜在全景梦境（LPD）任务，使单目模型能预测未来全景特征，显著提升导航准确性，缩小与全景输入方法的差距。**

- **链接: [https://arxiv.org/pdf/2508.02549v4](https://arxiv.org/pdf/2508.02549v4)**

> **作者:** Shuo Wang; Yongcai Wang; Zhaoxin Fan; Yucheng Wang; Maiyue Chen; Kaihui Wang; Zhizhong Su; Wanting Li; Xudong Cai; Yeying Jin; Deying Li
>
> **摘要:** Vision-Language Navigation (VLN) tasks often leverage panoramic RGB and depth inputs to provide rich spatial cues for action planning, but these sensors can be costly or less accessible in real-world deployments. Recent approaches based on Vision-Language Action (VLA) models achieve strong results with monocular input, yet they still lag behind methods using panoramic RGB-D information. We present MonoDream, a lightweight VLA framework that enables monocular agents to learn a Unified Navigation Representation (UNR). This shared feature representation jointly aligns navigation-relevant visual semantics (e.g., global layout, depth, and future cues) and language-grounded action intent, enabling more reliable action prediction. MonoDream further introduces Latent Panoramic Dreaming (LPD) tasks to supervise the UNR, which train the model to predict latent features of panoramic RGB and depth observations at both current and future steps based on only monocular input. Experiments on multiple VLN benchmarks show that MonoDream consistently improves monocular navigation performance and significantly narrows the gap with panoramic-based agents.
>
---
#### [replaced 026] D-LIO: 6DoF Direct LiDAR-Inertial Odometry based on Simultaneous Truncated Distance Field Mapping
- **分类: cs.RO**

- **简介: 该论文提出D-LIO，一种基于快速截断距离场（Fast-TDF）的6自由度直接激光雷达-惯性里程计方法。它无需特征提取与跟踪，直接利用原始点云进行在线优化，同时构建高精度环境TDF地图，实现高效、鲁棒的定位与建图，适用于多种机器人场景。**

- **链接: [https://arxiv.org/pdf/2505.16726v3](https://arxiv.org/pdf/2505.16726v3)**

> **作者:** Lucia Coto-Elena; J. E. Maese; L. Merino; F. Caballero
>
> **备注:** 9 pages, 3 figures and 43 references
>
> **摘要:** This paper presents a new approach for 6DoF Direct LiDAR-Inertial Odometry (D-LIO) based on the simultaneous mapping of truncated distance fields on CPU. Such continuous representation (in the vicinity of the points) enables working with raw 3D LiDAR data online, avoiding the need of LiDAR feature selection and tracking, simplifying the odometry pipeline and easily generalizing to many scenarios. The method is based on the proposed Fast Truncated Distance Field (Fast-TDF) method as a convenient tool to represent the environment. Such representation enables i) solving the LiDAR point-cloud registration as a nonlinear optimization process without the need of selecting/tracking LiDAR features in the input data, ii) simultaneously producing an accurate truncated distance field map of the environment, and iii) updating such map at constant time independently of its size. The approach is tested using open datasets, aerial and ground. It is also benchmarked against other state-of-the-art odometry approaches, demonstrating the same or better level of accuracy with the added value of an online-generated TDF representation of the environment, that can be used for other robotics tasks as planning or collision avoidance. The source code is publicly available at https://anonymous.4open.science/r/D-LIO
>
---
#### [replaced 027] VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文针对人形机器人自主运动操作能力不足的问题，提出VIRAL框架，通过大规模视觉仿真实现零样本部署。利用教师-学生架构与视觉域随机化，使仅依赖RGB图像的策略在真实机器人上成功完成连续运动操作，无需微调。**

- **链接: [https://arxiv.org/pdf/2511.15200v2](https://arxiv.org/pdf/2511.15200v2)**

> **作者:** Tairan He; Zi Wang; Haoru Xue; Qingwei Ben; Zhengyi Luo; Wenli Xiao; Ye Yuan; Xingye Da; Fernando Castañeda; Shankar Sastry; Changliu Liu; Guanya Shi; Linxi Fan; Yuke Zhu
>
> **备注:** Project website: https://viral-humanoid.github.io/
>
> **摘要:** A key barrier to the real-world deployment of humanoid robots is the lack of autonomous loco-manipulation skills. We introduce VIRAL, a visual sim-to-real framework that learns humanoid loco-manipulation entirely in simulation and deploys it zero-shot to real hardware. VIRAL follows a teacher-student design: a privileged RL teacher, operating on full state, learns long-horizon loco-manipulation using a delta action space and reference state initialization. A vision-based student policy is then distilled from the teacher via large-scale simulation with tiled rendering, trained with a mixture of online DAgger and behavior cloning. We find that compute scale is critical: scaling simulation to tens of GPUs (up to 64) makes both teacher and student training reliable, while low-compute regimes often fail. To bridge the sim-to-real gap, VIRAL combines large-scale visual domain randomization over lighting, materials, camera parameters, image quality, and sensor delays--with real-to-sim alignment of the dexterous hands and cameras. Deployed on a Unitree G1 humanoid, the resulting RGB-based policy performs continuous loco-manipulation for up to 54 cycles, generalizing to diverse spatial and appearance variations without any real-world fine-tuning, and approaching expert-level teleoperation performance. Extensive ablations dissect the key design choices required to make RGB-based humanoid loco-manipulation work in practice.
>
---
#### [replaced 028] Spectral Signature Mapping from RGB Imagery for Terrain-Aware Navigation
- **分类: cs.RO**

- **简介: 该论文针对机器人在户外环境中难以区分视觉相似但物理特性不同的地形问题，提出RS-Net模型，通过RGB图像预测光谱特征，进而推断地形标签与摩擦系数。该方法使轮式与四足机器人能仅依赖RGB摄像头实现精准导航与稳定运动，解决了高成本光谱传感器部署难题。**

- **链接: [https://arxiv.org/pdf/2509.19105v2](https://arxiv.org/pdf/2509.19105v2)**

> **作者:** Sarvesh Prajapati; Ananya Trivedi; Nathaniel Hanson; Bruce Maxwell; Taskin Padir
>
> **备注:** 8 pages, 11 figures, accepted to Robotic Computing & Communication
>
> **摘要:** Successful navigation in outdoor environments requires accurate prediction of the physical interactions between the robot and the terrain. Many prior methods rely on geometric or semantic labels to classify traversable surfaces. However, such labels cannot distinguish visually similar surfaces that differ in material properties. Spectral sensors enable inference of material composition from surface reflectance measured across multiple wavelength bands. Although spectral sensing is gaining traction in robotics, widespread deployment remains constrained by the need for custom hardware integration, high sensor costs, and compute-intensive processing pipelines. In this paper, we present the RGB Image to Spectral Signature Neural Network (RS-Net), a deep neural network designed to bridge the gap between the accessibility of RGB sensing and the rich material information provided by spectral data. RS-Net predicts spectral signatures from RGB patches, which we map to terrain labels and friction coefficients. The resulting terrain classifications are integrated into a sampling-based motion planner for a wheeled robot operating in outdoor environments. Likewise, the friction estimates are incorporated into a contact-force-based MPC for a quadruped robot navigating slippery surfaces. Overall, our framework learns the task-relevant physical properties offline during training and thereafter relies solely on RGB sensing at run time.
>
---
#### [replaced 029] Rethinking Progression of Memory State in Robotic Manipulation: An Object-Centric Perspective
- **分类: cs.RO; cs.CV**

- **简介: 论文针对机器人操作中因视觉相似物体导致的非马尔可夫决策问题，提出LIBERO-Mem基准与Embodied-SlotSSM框架。该框架通过槽位状态建模与关系编码器实现时空一致的物体记忆，提升长期依赖下的动作预测能力，解决了复杂交互中对象历史感知难题。**

- **链接: [https://arxiv.org/pdf/2511.11478v3](https://arxiv.org/pdf/2511.11478v3)**

> **作者:** Nhat Chung; Taisei Hanyu; Toan Nguyen; Huy Le; Frederick Bumgarner; Duy Minh Ho Nguyen; Khoa Vo; Kashu Yamazaki; Chase Rainwater; Tung Kieu; Anh Nguyen; Ngan Le
>
> **备注:** Accepted at AAAI 2026
>
> **摘要:** As embodied agents operate in increasingly complex environments, the ability to perceive, track, and reason about individual object instances over time becomes essential, especially in tasks requiring sequenced interactions with visually similar objects. In these non-Markovian settings, key decision cues are often hidden in object-specific histories rather than the current scene. Without persistent memory of prior interactions (what has been interacted with, where it has been, or how it has changed) visuomotor policies may fail, repeat past actions, or overlook completed ones. To surface this challenge, we introduce LIBERO-Mem, a non-Markovian task suite for stress-testing robotic manipulation under object-level partial observability. It combines short- and long-horizon object tracking with temporally sequenced subgoals, requiring reasoning beyond the current frame. However, vision-language-action (VLA) models often struggle in such settings, with token scaling quickly becoming intractable even for tasks spanning just a few hundred frames. We propose Embodied-SlotSSM, a slot-centric VLA framework built for temporal scalability. It maintains spatio-temporally consistent slot identities and leverages them through two mechanisms: (1) slot-state-space modeling for reconstructing short-term history, and (2) a relational encoder to align the input tokens with action decoding. Together, these components enable temporally grounded, context-aware action prediction. Experiments show Embodied-SlotSSM's baseline performance on LIBERO-Mem and general tasks, offering a scalable solution for non-Markovian reasoning in object-centric robotic policies.
>
---
#### [replaced 030] HAFO: Humanoid Force-Adaptive Control for Intense External Force Interaction Environments
- **分类: cs.RO**

- **简介: 该论文针对人形机器人在强外部干扰下难以实现稳定运动与精准操作的问题，提出HAFO框架。通过双智能体强化学习，联合优化步态与上肢操控策略，利用弹簧阻尼模型建模外力并实现细粒度力控，使机器人在绳索拉力等强干扰下仍能保持稳定，显著提升抗干扰能力。**

- **链接: [https://arxiv.org/pdf/2511.20275v2](https://arxiv.org/pdf/2511.20275v2)**

> **作者:** Chenhui Dong; Haozhe Xu; Wenhao Feng; Zhipeng Wang; Yanmin Zhou; Yifei Zhao; Bin He
>
> **摘要:** Reinforcement learning controllers have made impressive progress in humanoid locomotion and light load manipulation. However, achieving robust and precise motion with strong force interaction remains a significant challenge. Based on the above limitations, this paper proposes HAFO, a dual-agent reinforcement learning control framework that simultaneously optimizes both a robust locomotion strategy and a precise upper-body manipulation strategy through coupled training under external force interaction environments. Simultaneously, we explicitly model the external pulling disturbances through a spring-damper system and achieve fine-grained force control by manipulating the virtual spring. During this process, the reinforcement-learning policy spontaneously generates disturbance-rejection response by exploiting environmental feedback. Moreover, HAFO employs an asymmetric Actor-Critic framework in which the Critic-network access to privileged spring-damping forces guides the actor-network to learn a generalizable, robust policy for resisting external disturbances. The experimental results demonstrate that HAFO achieves stable control of humanoid robot under various strong force interactions, showing remarkable performance in load tasks and ensuring stable robot operation under rope tension disturbances. Project website: hafo-robot.github.io.
>
---
#### [replaced 031] Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis
- **分类: cs.RO; cs.AI**

- **简介: 该论文调研基础模型在自动驾驶场景生成与分析中的应用，旨在解决传统方法生成场景多样性不足、真实性差的问题。工作包括构建统一分类体系，综述方法、数据集、平台与评估指标，并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2506.11526v2](https://arxiv.org/pdf/2506.11526v2)**

> **作者:** Yuan Gao; Mattia Piccinini; Yuchen Zhang; Dingrui Wang; Korbinian Moller; Roberto Brusnicki; Baha Zarrouki; Alessio Gambi; Jan Frederik Totz; Kai Storms; Steven Peters; Andrea Stocco; Bassam Alrifaee; Marco Pavone; Johannes Betz
>
> **备注:** Revised manuscript with separate evaluation metrics table
>
> **摘要:** For autonomous vehicles, safe navigation in complex environments depends on handling a broad range of diverse and rare driving scenarios. Simulation- and scenario-based testing have emerged as key approaches to development and validation of autonomous driving systems. Traditional scenario generation relies on rule-based systems, knowledge-driven models, and data-driven synthesis, often producing limited diversity and unrealistic safety-critical cases. With the emergence of foundation models, which represent a new generation of pre-trained, general-purpose AI models, developers can process heterogeneous inputs (e.g., natural language, sensor data, HD maps, and control actions), enabling the synthesis and interpretation of complex driving scenarios. In this paper, we conduct a survey about the application of foundation models for scenario generation and scenario analysis in autonomous driving (as of May 2025). Our survey presents a unified taxonomy that includes large language models, vision-language models, multimodal large language models, diffusion models, and world models for the generation and analysis of autonomous driving scenarios. In addition, we review the methodologies, open-source datasets, simulation platforms, and benchmark challenges, and we examine the evaluation metrics tailored explicitly to scenario generation and analysis. Finally, the survey concludes by highlighting the open challenges and research questions, and outlining promising future research directions. All reviewed papers are listed in a continuously maintained repository, which contains supplementary materials and is available at https://github.com/TUM-AVS/FM-for-Scenario-Generation-Analysis.
>
---
#### [replaced 032] Differentiable Skill Optimisation for Powder Manipulation in Laboratory Automation
- **分类: cs.RO**

- **简介: 该论文针对实验室自动化中粉末精确操控难题，提出一种可微物理仿真驱动的轨迹优化框架。通过低维技能空间参数化与课程学习策略，实现接触丰富的机器人轨迹端到端优化，显著提升粉末运输任务的成功率与稳定性。**

- **链接: [https://arxiv.org/pdf/2510.01438v2](https://arxiv.org/pdf/2510.01438v2)**

> **作者:** Minglun Wei; Xintong Yang; Yu-Kun Lai; Ze Ji
>
> **备注:** Accepted by IROS 2025 Workshop on Embodied AI and Robotics for Future Scientific Discovery
>
> **摘要:** Robotic automation is accelerating scientific discovery by reducing manual effort in laboratory workflows. However, precise manipulation of powders remains challenging, particularly in tasks such as transport that demand accuracy and stability. We propose a trajectory optimisation framework for powder transport in laboratory settings, which integrates differentiable physics simulation for accurate modelling of granular dynamics, low-dimensional skill-space parameterisation to reduce optimisation complexity, and a curriculum-based strategy that progressively refines task competence over long horizons. This formulation enables end-to-end optimisation of contact-rich robot trajectories while maintaining stability and convergence efficiency. Experimental results demonstrate that the proposed method achieves superior task success rates and stability compared to the reinforcement learning baseline.
>
---
#### [replaced 033] Efficient Path Planning and Task Allocation Algorithm for Boolean Specifications
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究多机器人系统的路径规划与任务分配问题，针对全局布尔规格下的优化难题，利用佩特里网模型的全单模性，将整数规划松弛为高效线性规划，实现大规模系统下安全、高效的协同决策。**

- **链接: [https://arxiv.org/pdf/2506.04881v3](https://arxiv.org/pdf/2506.04881v3)**

> **作者:** Ioana Hustiu; Roozbeh Abolpour; Marius Kloetzer; Cristian Mahulea
>
> **摘要:** This paper addresses path planning and task allocation in multi-robot systems subject to global Boolean specifications defined on the final state. The main contribution is the exploitation of the structural properties of a Petri net model: we prove that the associated constraint matrix is totally unimodular (TU). This property allows relaxing the original Integer Linear Programming (ILP) formulation to a Mixed Integer Linear Programming (MILP) in which all variables are continuous except for those that are corresponding to the atomic propositions in the Boolean specification. This yields a substantial reduction in complexity. In the special case where the specification is a conjunction of atomic propositions of cardinality equal to the team size, i.e., the standard Task-Assignment and Path Finding (TAPF) problem, the formulation reduces to a Linear Programming (LP). Collision-free paths are ensured by introducing intermediate synchronization points only when necessary, while robots move in parallel between them. These structural insights enable a computationally efficient and scalable solution, achieving tractability and safety for large-scale systems with up to 2500 robots.
>
---
#### [replaced 034] Design and Measurements of mmWave FMCW Radar Based Non-Contact Multi-Patient Heart Rate and Breath Rate Monitoring System
- **分类: eess.SY; cs.RO; eess.SP**

- **简介: 该论文提出一种基于毫米波FMCW雷达的非接触式多患者心率与呼吸率监测系统，旨在实现高效、无干扰的多人生命体征同步检测。通过融合多种信号处理方法并利用最小二乘法优化，提升了测量精度与鲁棒性，实验表明心率和呼吸率识别准确率分别超过97%和93%。**

- **链接: [https://arxiv.org/pdf/2511.21255v2](https://arxiv.org/pdf/2511.21255v2)**

> **作者:** Jewel Benny; Pranjal Mahajan; Srayan Sankar Chatterjee; Mohd Wajid; Abhishek Srivastava
>
> **备注:** Presented at BioCAS 2023
>
> **摘要:** Recent developments in mmWave radar technologies have enabled the truly non-contact heart-rate (HR) and breath-rate (BR) measurement approaches, which provides a great ease in patient monitoring. Additionally, these technologies also provide opportunities to simultaneously detect HR and BR of multiple patients, which has become increasingly important for efficient mass monitoring scenarios. In this work, a frequency modulated continuous wave (FMCW) mmWave radar based truly non-contact multiple patient HR and BR monitoring system has been presented. Furthermore, a novel approach is also proposed, which combines multiple processing methods using a least squares solution to improve measurement accuracy, generalization, and handle measurement error. The proposed system has been developed using Texas Instruments' FMCW radar and experimental results with multiple subjects are also presented, which show >97% and >93% accuracy in the measured BR and HR values, respectively.
>
---
#### [replaced 035] ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **简介: 该论文针对视觉语言模型在长视频序列中推理能力弱的问题，提出ROVER框架。通过递归分解视频为短时子任务，实现局部精准推理并保留全局上下文。在机器人操作任务中验证，显著提升任务进度估计、帧级推理和视频问答性能，降低幻觉，线性降低时间复杂度。**

- **链接: [https://arxiv.org/pdf/2508.01943v2](https://arxiv.org/pdf/2508.01943v2)**

> **作者:** Philip Schroeder; Ondrej Biza; Thomas Weng; Hongyin Luo; James Glass
>
> **摘要:** Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: https://rover-vlm.github.io
>
---
#### [replaced 036] PointMapPolicy: Structured Point Cloud Processing for Multi-Modal Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人操作中多模态感知的精度与泛化问题，提出PointMapPolicy方法。通过结构化点云网格实现细粒度几何建模，结合RGB图像提升语义理解，利用xLSTM融合多模态信息，在多个基准上实现领先性能。**

- **链接: [https://arxiv.org/pdf/2510.20406v2](https://arxiv.org/pdf/2510.20406v2)**

> **作者:** Xiaogang Jia; Qian Wang; Anrui Wang; Han A. Wang; Balázs Gyenes; Emiliyan Gospodinov; Xinkai Jiang; Ge Li; Hongyi Zhou; Weiran Liao; Xi Huang; Maximilian Beck; Moritz Reuss; Rudolf Lioutikov; Gerhard Neumann
>
> **摘要:** Robotic manipulation systems benefit from complementary sensing modalities, where each provides unique environmental information. Point clouds capture detailed geometric structure, while RGB images provide rich semantic context. Current point cloud methods struggle to capture fine-grained detail, especially for complex tasks, which RGB methods lack geometric awareness, which hinders their precision and generalization. We introduce PointMapPolicy, a novel approach that conditions diffusion policies on structured grids of points without downsampling. The resulting data type makes it easier to extract shape and spatial relationships from observations, and can be transformed between reference frames. Yet due to their structure in a regular grid, we enable the use of established computer vision techniques directly to 3D data. Using xLSTM as a backbone, our model efficiently fuses the point maps with RGB data for enhanced multi-modal perception. Through extensive experiments on the RoboCasa and CALVIN benchmarks and real robot evaluations, we demonstrate that our method achieves state-of-the-art performance across diverse manipulation tasks. The overview and demos are available on our project page: https://point-map.github.io/Point-Map/
>
---
#### [replaced 037] Trust-Preserved Human-Robot Shared Autonomy enabled by Bayesian Relational Event Modeling
- **分类: cs.RO**

- **简介: 该论文针对人机协作中信任缺失问题，提出一种基于贝叶斯关系事件建模的可信保持共享自主策略。通过动态推断人类信任水平，使机器人主动调节自主性以建立、维护和修复信任。在搜救任务实验中验证了其在任务绩效与用户接受度上的优越性。**

- **链接: [https://arxiv.org/pdf/2311.02009v3](https://arxiv.org/pdf/2311.02009v3)**

> **作者:** Yingke Li; Fumin Zhang
>
> **摘要:** Shared autonomy functions as a flexible framework that empowers robots to operate across a spectrum of autonomy levels, allowing for efficient task execution with minimal human oversight. However, humans might be intimidated by the autonomous decision-making capabilities of robots due to perceived risks and a lack of trust. This paper proposed a trust-preserved shared autonomy strategy that allows robots to seamlessly adjust their autonomy level, striving to optimize team performance and enhance their acceptance among human collaborators. By enhancing the relational event modeling framework with Bayesian learning techniques, this paper enables dynamic inference of human trust based solely on time-stamped relational events communicated within human-robot teams. Adopting a longitudinal perspective on trust development and calibration in human-robot teams, the proposed trust-preserved shared autonomy strategy warrants robots to actively establish, maintain, and repair human trust, rather than merely passively adapting to it. We validate the effectiveness of the proposed approach through a user study on a human-robot collaborative search and rescue scenario. The objective and subjective evaluations demonstrate its merits on both task execution and user acceptability over the baseline approach that does not consider the preservation of trust.
>
---
