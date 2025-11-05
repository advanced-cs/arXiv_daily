# 机器人 cs.RO

- **最新发布 24 篇**

- **更新 32 篇**

## 最新发布

#### [new 001] Dexterous Robotic Piano Playing at Scale
- **分类: cs.RO**

- **简介: 该论文提出OmniPianist，实现机器人无需人类示范即可大规模弹奏近千首乐曲。通过最优传输自动指法规划、2000+智能体强化学习生成百万级轨迹数据集RP1M++，并用Flow Matching Transformer进行模仿学习，解决高维、接触密集的双臂钢琴演奏难题。**

- **链接: [http://arxiv.org/pdf/2511.02504v1](http://arxiv.org/pdf/2511.02504v1)**

> **作者:** Le Chen; Yi Zhao; Jan Schneider; Quankai Gao; Simon Guist; Cheng Qian; Juho Kannala; Bernhard Schölkopf; Joni Pajarinen; Dieter Büchler
>
> **摘要:** Endowing robot hands with human-level dexterity has been a long-standing goal in robotics. Bimanual robotic piano playing represents a particularly challenging task: it is high-dimensional, contact-rich, and requires fast, precise control. We present OmniPianist, the first agent capable of performing nearly one thousand music pieces via scalable, human-demonstration-free learning. Our approach is built on three core components. First, we introduce an automatic fingering strategy based on Optimal Transport (OT), allowing the agent to autonomously discover efficient piano-playing strategies from scratch without demonstrations. Second, we conduct large-scale Reinforcement Learning (RL) by training more than 2,000 agents, each specialized in distinct music pieces, and aggregate their experience into a dataset named RP1M++, consisting of over one million trajectories for robotic piano playing. Finally, we employ a Flow Matching Transformer to leverage RP1M++ through large-scale imitation learning, resulting in the OmniPianist agent capable of performing a wide range of musical pieces. Extensive experiments and ablation studies highlight the effectiveness and scalability of our approach, advancing dexterous robotic piano playing at scale.
>
---
#### [new 002] ZJUNlict Extended Team Description Paper 2025
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属机器人足球系统研发，旨在提升ZJUNlict机器人在高速对抗中的决策与控制精度。工作包括：为v2023机器人集成IMU以优化姿态控制，并改进策略与CUDA模块，增强球路预测与控球判断能力。**

- **链接: [http://arxiv.org/pdf/2511.02315v1](http://arxiv.org/pdf/2511.02315v1)**

> **作者:** Zifei Wu; Lijie Wang; Zhe Yang; Shijie Yang; Liang Wang; Haoran Fu; Yinliang Cai; Rong Xiong
>
> **摘要:** This paper presents the ZJUNlict team's work over the past year, covering both hardware and software advancements. In the hardware domain, the integration of an IMU into the v2023 robot was completed to enhance posture accuracy and angular velocity planning. On the software side, key modules were optimized, including the strategy and CUDA modules, with significant improvements in decision making efficiency, ball pursuit prediction, and ball possession prediction to adapt to high-tempo game dynamics.
>
---
#### [new 003] LACY: A Vision-Language Model-based Language-Action Cycle for Self-Improving Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: LACY提出一种视觉-语言模型驱动的语言-动作循环框架，解决机器人操作中单向指令执行缺乏理解的问题，通过双向映射（L2A/A2L）与语义一致性验证，实现自监督数据增强，提升任务成功率与 grounding 能力。**

- **链接: [http://arxiv.org/pdf/2511.02239v1](http://arxiv.org/pdf/2511.02239v1)**

> **作者:** Youngjin Hong; Houjian Yu; Mingen Li; Changhyun Choi
>
> **备注:** Preprint. Project page: https://vla2026.github.io/LACY/
>
> **摘要:** Learning generalizable policies for robotic manipulation increasingly relies on large-scale models that map language instructions to actions (L2A). However, this one-way paradigm often produces policies that execute tasks without deeper contextual understanding, limiting their ability to generalize or explain their behavior. We argue that the complementary skill of mapping actions back to language (A2L) is essential for developing more holistic grounding. An agent capable of both acting and explaining its actions can form richer internal representations and unlock new paradigms for self-supervised learning. We introduce LACY (Language-Action Cycle), a unified framework that learns such bidirectional mappings within a single vision-language model. LACY is jointly trained on three synergistic tasks: generating parameterized actions from language (L2A), explaining observed actions in language (A2L), and verifying semantic consistency between two language descriptions (L2C). This enables a self-improving cycle that autonomously generates and filters new training data through an active augmentation strategy targeting low-confidence cases, thereby improving the model without additional human labels. Experiments on pick-and-place tasks in both simulation and the real world show that LACY improves task success rates by 56.46% on average and yields more robust language-action grounding for robotic manipulation. Project page: https://vla2026.github.io/LACY/
>
---
#### [new 004] XR-1: Towards Versatile Vision-Language-Action Models via Learning Unified Vision-Motion Representations
- **分类: cs.RO**

- **简介: 论文提出XR-1，面向多机器人、多任务的视觉-语言-动作建模，通过统一视觉-运动编码（UVMC）解决低级动作生成与跨域数据对齐难题，实现跨机器人泛化与高效学习。**

- **链接: [http://arxiv.org/pdf/2511.02776v1](http://arxiv.org/pdf/2511.02776v1)**

> **作者:** Shichao Fan; Kun Wu; Zhengping Che; Xinhua Wang; Di Wu; Fei Liao; Ning Liu; Yixue Zhang; Zhen Zhao; Zhiyuan Xu; Meng Li; Qingjie Liu; Shanghang Zhang; Min Wan; Jian Tang
>
> **摘要:** Recent progress in large-scale robotic datasets and vision-language models (VLMs) has advanced research on vision-language-action (VLA) models. However, existing VLA models still face two fundamental challenges: (i) producing precise low-level actions from high-dimensional observations, (ii) bridging domain gaps across heterogeneous data sources, including diverse robot embodiments and human demonstrations. Existing methods often encode latent variables from either visual dynamics or robotic actions to guide policy learning, but they fail to fully exploit the complementary multi-modal knowledge present in large-scale, heterogeneous datasets. In this work, we present X Robotic Model 1 (XR-1), a novel framework for versatile and scalable VLA learning across diverse robots, tasks, and environments. XR-1 introduces the \emph{Unified Vision-Motion Codes (UVMC)}, a discrete latent representation learned via a dual-branch VQ-VAE that jointly encodes visual dynamics and robotic motion. UVMC addresses these challenges by (i) serving as an intermediate representation between the observations and actions, and (ii) aligning multimodal dynamic information from heterogeneous data sources to capture complementary knowledge. To effectively exploit UVMC, we propose a three-stage training paradigm: (i) self-supervised UVMC learning, (ii) UVMC-guided pretraining on large-scale cross-embodiment robotic datasets, and (iii) task-specific post-training. We validate XR-1 through extensive real-world experiments with more than 14,000 rollouts on six different robot embodiments, spanning over 120 diverse manipulation tasks. XR-1 consistently outperforms state-of-the-art baselines such as $\pi_{0.5}$, $\pi_0$, RDT, UniVLA, and GR00T-N1.5 while demonstrating strong generalization to novel objects, background variations, distractors, and illumination changes. Our project is at https://xr-1-vla.github.io/.
>
---
#### [new 005] TACO: Trajectory-Aware Controller Optimization for Quadrotors
- **分类: cs.RO**

- **简介: TACO提出一种轨迹感知的四旋翼控制器在线优化框架，通过学习模型实时调整控制参数，提升轨迹跟踪性能并增强动态可行性，相比传统方法显著降低误差且运行速度更快，支持真实无人机实时部署。**

- **链接: [http://arxiv.org/pdf/2511.02060v1](http://arxiv.org/pdf/2511.02060v1)**

> **作者:** Hersh Sanghvi; Spencer Folk; Vijay Kumar; Camillo Jose Taylor
>
> **备注:** 8 pages, 6 figures. In submission to ICRA 2026
>
> **摘要:** Controller performance in quadrotor trajectory tracking depends heavily on parameter tuning, yet standard approaches often rely on fixed, manually tuned parameters that sacrifice task-specific performance. We present Trajectory-Aware Controller Optimization (TACO), a framework that adapts controller parameters online based on the upcoming reference trajectory and current quadrotor state. TACO employs a learned predictive model and a lightweight optimization scheme to optimize controller gains in real time with respect to a broad class of trajectories, and can also be used to adapt trajectories to improve dynamic feasibility while respecting smoothness constraints. To enable large-scale training, we also introduce a parallelized quadrotor simulator supporting fast data collection on diverse trajectories. Experiments on a variety of trajectory types show that TACO outperforms conventional, static parameter tuning while operating orders of magnitude faster than black-box optimization baselines, enabling practical real-time deployment on a physical quadrotor. Furthermore, we show that adapting trajectories using TACO significantly reduces the tracking error obtained by the quadrotor.
>
---
#### [new 006] Whole-body motion planning and safety-critical control for aerial manipulation
- **分类: cs.RO**

- **简介: 该论文针对空中操作器在复杂环境中的安全运动规划问题，提出基于超二次曲面的全身体运动规划与安全关键控制框架，实现高精度碰撞规避与动态可行轨迹生成，显著提升规划效率与安全性。**

- **链接: [http://arxiv.org/pdf/2511.02342v1](http://arxiv.org/pdf/2511.02342v1)**

> **作者:** Lin Yang; Jinwoo Lee; Domenico Campolo; H. Jin Kim; Jeonghyun Byun
>
> **备注:** Submitted to 2026 IFAC World Congress with the Journal option (MECHATRONICS)
>
> **摘要:** Aerial manipulation combines the maneuverability of multirotors with the dexterity of robotic arms to perform complex tasks in cluttered spaces. Yet planning safe, dynamically feasible trajectories remains difficult due to whole-body collision avoidance and the conservativeness of common geometric abstractions such as bounding boxes or ellipsoids. We present a whole-body motion planning and safety-critical control framework for aerial manipulators built on superquadrics (SQs). Using an SQ-plus-proxy representation, we model both the vehicle and obstacles with differentiable, geometry-accurate surfaces. Leveraging this representation, we introduce a maximum-clearance planner that fuses Voronoi diagrams with an equilibrium-manifold formulation to generate smooth, collision-aware trajectories. We further design a safety-critical controller that jointly enforces thrust limits and collision avoidance via high-order control barrier functions. In simulation, our approach outperforms sampling-based planners in cluttered environments, producing faster, safer, and smoother trajectories and exceeding ellipsoid-based baselines in geometric fidelity. Actual experiments on a physical aerial-manipulation platform confirm feasibility and robustness, demonstrating consistent performance across simulation and hardware settings. The video can be found at https://youtu.be/hQYKwrWf1Ak.
>
---
#### [new 007] A Quantitative Comparison of Centralised and Distributed Reinforcement Learning-Based Control for Soft Robotic Arms
- **分类: cs.RO**

- **简介: 该论文对比了集中式与分布式强化学习在软体机械臂控制中的性能，解决多段控制下的效率与鲁棒性权衡问题。通过仿真验证，发现集中式训练更快，分布式在段数较多时更高效、鲁棒。**

- **链接: [http://arxiv.org/pdf/2511.02192v1](http://arxiv.org/pdf/2511.02192v1)**

> **作者:** Linxin Hou; Qirui Wu; Zhihang Qin; Neil Banerjee; Yongxin Guo; Cecilia Laschi
>
> **备注:** 7 pages, 4 figures, 2 tables, submitted to RoboSoft 2026
>
> **摘要:** This paper presents a quantitative comparison between centralised and distributed multi-agent reinforcement learning (MARL) architectures for controlling a soft robotic arm modelled as a Cosserat rod in simulation. Using PyElastica and the OpenAI Gym interface, we train both a global Proximal Policy Optimisation (PPO) controller and a Multi-Agent PPO (MAPPO) under identical budgets. Both approaches are based on the arm having $n$ number of controlled sections. The study systematically varies $n$ and evaluates the performance of the arm to reach a fixed target in three scenarios: default baseline condition, recovery from external disturbance, and adaptation to actuator failure. Quantitative metrics used for the evaluation are mean action magnitude, mean final distance, mean episode length, and success rate. The results show that there are no significant benefits of the distributed policy when the number of controlled sections $n\le4$. In very simple systems, when $n\le2$, the centralised policy outperforms the distributed one. When $n$ increases to $4< n\le 12$, the distributed policy shows a high sample efficiency. In these systems, distributed policy promotes a stronger success rate, resilience, and robustness under local observability and yields faster convergence given the same sample size. However, centralised policies achieve much higher time efficiency during training as it takes much less time to train the same size of samples. These findings highlight the trade-offs between centralised and distributed policy in reinforcement learning-based control for soft robotic systems and provide actionable design guidance for future sim-to-real transfer in soft rod-like manipulators.
>
---
#### [new 008] TRACE: Textual Reasoning for Affordance Coordinate Extraction
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出TRACE方法，通过文本推理链（CoR）提升视觉语言模型在机器人操作中的空间定位精度，构建了含推理文本的TRACE数据集，显著提升W2P基准性能，并增强模型可解释性。**

- **链接: [http://arxiv.org/pdf/2511.01999v1](http://arxiv.org/pdf/2511.01999v1)**

> **作者:** Sangyun Park; Jin Kim; Yuchen Cui; Matthew S. Brown
>
> **备注:** ICCV 2025. *Equal contribution. {\dag}Corresponding author
>
> **摘要:** Vision-Language Models (VLMs) struggle to translate high-level instructions into the precise spatial affordances required for robotic manipulation. While visual Chain-of-Thought (CoT) methods exist, they are often computationally intensive. In this work, we introduce TRACE (Textual Reasoning for Affordance Coordinate Extraction), a novel methodology that integrates a textual Chain of Reasoning (CoR) into the affordance prediction process. We use this methodology to create the TRACE dataset, a large-scale collection created via an autonomous pipeline that pairs instructions with explicit textual rationales. By fine-tuning a VLM on this data, our model learns to externalize its spatial reasoning before acting. Our experiments show that our TRACE-tuned model achieves state-of-the-art performance, reaching 48.1% accuracy on the primary Where2Place (W2P) benchmark (a 9.6% relative improvement) and 55.0% on the more challenging W2P(h) subset. Crucially, an ablation study demonstrates that performance scales directly with the amount of reasoning data used, confirming the CoR's effectiveness. Furthermore, analysis of the model's attention maps reveals an interpretable reasoning process where focus shifts dynamically across reasoning steps. This work shows that training VLMs to generate a textual CoR is an effective and robust strategy for enhancing the precision, reliability, and interpretability of VLM-based robot control. Our dataset and code are available at https://github.com/jink-ucla/TRACE
>
---
#### [new 009] Stein-based Optimization of Sampling Distributions in Model Predictive Path Integral Control
- **分类: cs.RO**

- **简介: 该论文提出SOPPI方法，将SVGD引入MPPI控制，动态优化采样分布以提升轨迹表示效率，解决传统MPPI采样不足问题，在多系统中验证了其在低粒子数下的性能优势。**

- **链接: [http://arxiv.org/pdf/2511.02015v1](http://arxiv.org/pdf/2511.02015v1)**

> **作者:** Jace Aldrich; Odest Chadwicke Jenkins
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This paper presents a novel method for Model Predictive Path Integral (MPPI) control that optimizes sample generation towards an optimal trajectory through Stein Variational Gradient Descent (SVGD). MPPI is traditionally reliant on randomly sampled trajectories, often by a Gaussian distribution. The result can lead to sample deprivation, under-representing the space of possible trajectories, and yield suboptimal results. Through introducing SVGD updates in between MPPI environment steps, we present Stein-Optimized Path-Integral Inference (SOPPI), an MPPI/SVGD algorithm that can dynamically update noise distributions at runtime to shape a more optimal representation without an excessive increase in computational requirements. We demonstrate the efficacy of our method systems ranging from a Cart-Pole to a two-dimensional bipedal walking task, indicating improved performance above standard MPPI across a range of hyper-parameters and demonstrate feasibility at lower particle counts. We discuss the applicability of this MPPI/SVGD method to higher degree-of-freedom systems, as well as its potential to new developments in state-of-the-art differentiable simulators.
>
---
#### [new 010] TurboMap: GPU-Accelerated Local Mapping for Visual SLAM
- **分类: cs.RO**

- **简介: 论文提出TurboMap，面向视觉SLAM的局部建图任务，通过GPU加速地图点三角化与融合、CPU优化关键帧裁剪及GPU加速局部光束法平差，显著提升建图速度，同时保持精度。**

- **链接: [http://arxiv.org/pdf/2511.02036v1](http://arxiv.org/pdf/2511.02036v1)**

> **作者:** Parsa Hosseininejad; Kimia Khabiri; Shishir Gopinath; Soudabeh Mohammadhashemi; Karthik Dantu; Steven Y. Ko
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** This paper presents TurboMap, a GPU-accelerated and CPU-optimized local mapping module for visual SLAM systems. We identify key performance bottlenecks in the local mapping process for visual SLAM and address them through targeted GPU and CPU optimizations. Specifically, we offload map point triangulation and fusion to the GPU, accelerate redundant keyframe culling on the CPU, and integrate a GPU-accelerated solver to speed up local bundle adjustment. Our implementation is built on top of ORB-SLAM3 and leverages CUDA for GPU programming. The experimental results show that TurboMap achieves an average speedup of 1.3x in the EuRoC dataset and 1.6x in the TUM-VI dataset in the local mapping module, on both desktop and embedded platforms, while maintaining the accuracy of the original system.
>
---
#### [new 011] TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: TWIST2提出一种无动作捕捉、便携式人形机器人数据采集系统，利用VR实现全身遥操作，支持高效收集高成功率演示数据，并构建基于自视角视觉的层次化策略，实现全身体灵巧操作与动态任务。**

- **链接: [http://arxiv.org/pdf/2511.02832v1](http://arxiv.org/pdf/2511.02832v1)**

> **作者:** Yanjie Ze; Siheng Zhao; Weizhuo Wang; Angjoo Kanazawa; Rocky Duan; Pieter Abbeel; Guanya Shi; Jiajun Wu; C. Karen Liu
>
> **备注:** Website: https://yanjieze.com/TWIST2
>
> **摘要:** Large-scale data has driven breakthroughs in robotics, from language models to vision-language-action models in bimanual manipulation. However, humanoid robotics lacks equally effective data collection frameworks. Existing humanoid teleoperation systems either use decoupled control or depend on expensive motion capture setups. We introduce TWIST2, a portable, mocap-free humanoid teleoperation and data collection system that preserves full whole-body control while advancing scalability. Our system leverages PICO4U VR for obtaining real-time whole-body human motions, with a custom 2-DoF robot neck (cost around $250) for egocentric vision, enabling holistic human-to-humanoid control. We demonstrate long-horizon dexterous and mobile humanoid skills and we can collect 100 demonstrations in 15 minutes with an almost 100% success rate. Building on this pipeline, we propose a hierarchical visuomotor policy framework that autonomously controls the full humanoid body based on egocentric vision. Our visuomotor policy successfully demonstrates whole-body dexterous manipulation and dynamic kicking tasks. The entire system is fully reproducible and open-sourced at https://yanjieze.com/TWIST2 . Our collected dataset is also open-sourced at https://twist-data.github.io .
>
---
#### [new 012] A Step Toward World Models: A Survey on Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于综述任务，旨在厘清机器人世界模型的定义与核心能力。通过分析操控领域方法，提炼其在感知、预测与控制中的作用，构建通用、实用的世界模型发展路线图。**

- **链接: [http://arxiv.org/pdf/2511.02097v1](http://arxiv.org/pdf/2511.02097v1)**

> **作者:** Peng-Fei Zhang; Ying Cheng; Xiaofan Sun; Shijie Wang; Lei Zhu; Heng Tao Shen
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Autonomous agents are increasingly expected to operate in complex, dynamic, and uncertain environments, performing tasks such as manipulation, navigation, and decision-making. Achieving these capabilities requires agents to understand the underlying mechanisms and dynamics of the world, moving beyond purely reactive control or simple replication of observed states. This motivates the development of world models as internal representations that encode environmental states, capture dynamics, and enable prediction, planning, and reasoning. Despite growing interest, the definition, scope, architectures, and essential capabilities of world models remain ambiguous. In this survey, rather than directly imposing a fixed definition and limiting our scope to methods explicitly labeled as world models, we examine approaches that exhibit the core capabilities of world models through a review of methods in robotic manipulation. We analyze their roles across perception, prediction, and control, identify key challenges and solutions, and distill the core components, capabilities, and functions that a real world model should possess. Building on this analysis, we aim to outline a roadmap for developing generalizable and practical world models for robotics.
>
---
#### [new 013] Non-Contact Manipulation of Induced Magnetic Dipoles
- **分类: cs.RO**

- **简介: 该论文研究非接触式诱导磁偶极子的闭环操控，解决导电非磁性物体（如铝球）在低力环境下的精确三维定位问题，通过振荡磁场实现闭环位置控制，推动空间碎片回收等应用。**

- **链接: [http://arxiv.org/pdf/2511.02761v1](http://arxiv.org/pdf/2511.02761v1)**

> **作者:** Seth Stewart; Joseph Pawelski; Steve Ward; Andrew J. Petruska
>
> **摘要:** Extending the field of magnetic manipulation to conductive, non-magnetic objects opens the door for a wide array of applications previously limited to hard or soft magnetic materials. Of particular interest is the recycling of space debris through the use of oscillating magnetic fields, which represent a cache of raw materials in an environment particularly suited to the low forces generated from inductive magnetic manipulation. Building upon previous work that demonstrated 3D open-loop position control by leveraging the opposing dipole moment created from induced eddy currents, this work demonstrates closed-loop position control of a semi-buoyant aluminum sphere in lab tests, and the efficacy of varying methods for force inversion is explored. The closed-loop methods represent a critical first step towards wider applications for 3-DOF position control of induced magnetic dipoles.
>
---
#### [new 014] Kinematic and Ergonomic Design of a Robotic Arm for Precision Laparoscopic Surgery
- **分类: cs.RO**

- **简介: 该论文属于机器人手术系统设计任务，旨在提升腹腔镜手术的精度与医生舒适度。研究设计了一种7自由度带远程中心运动的机械臂，通过仿真验证其显著降低目标误差超50%并减少肌肉疲劳，验证了构型优化与人因工程对手术机器人性能的关键作用。**

- **链接: [http://arxiv.org/pdf/2511.02167v1](http://arxiv.org/pdf/2511.02167v1)**

> **作者:** Tian Hao; Tong Lu; Che Chan
>
> **摘要:** Robotic assistance in minimally invasive surgery can greatly enhance surgical precision and reduce surgeon fatigue. This paper presents a focused investigation on the kinematic and ergonomic design principles for a laparoscopic surgical robotic arm aimed at high-precision tasks. We propose a 7-degree-of-freedom (7-DOF) robotic arm system that incorporates a remote center of motion (RCM) at the instrument insertion point and ergonomic considerations to improve surgeon interaction. The design is implemented on a general-purpose robotic platform, and a series of simulated surgical tasks were performed to evaluate targeting accuracy, task efficiency, and surgeon comfort compared to conventional manual laparoscopy. Experimental results demonstrate that the optimized robotic design achieves significantly improved targeting accuracy (error reduced by over 50%) and shorter task completion times, while substantially lowering operator muscle strain and discomfort. These findings validate the importance of kinematic optimization (such as added articulations and tremor filtering) and human-centered ergonomic design in enhancing the performance of robot-assisted surgery. The insights from this work can guide the development of next-generation surgical robots that improve surgical outcomes and ergonomics for the operating team.
>
---
#### [new 015] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文提出一种融合3D生成AI与视觉语言模型的机器人装配系统，解决多部件物体从文本指令到物理组装的难题，实现零样本部件分解与功能感知，并支持人机对话优化，提升生成精度与可控性。**

- **链接: [http://arxiv.org/pdf/2511.02162v1](http://arxiv.org/pdf/2511.02162v1)**

> **作者:** Alexander Htet Kyaw; Richa Gupta; Dhruv Shah; Anoop Sinha; Kory Mathewson; Stefanie Pender; Sachin Chitta; Yotto Koga; Faez Ahmed; Lawrence Sass; Randall Davis
>
> **备注:** Accepted to NeurIPS 2025, Conference on Neural Information Processing Systems, Creative AI Track
>
> **摘要:** Advances in 3D generative AI have enabled the creation of physical objects from text prompts, but challenges remain in creating objects involving multiple component types. We present a pipeline that integrates 3D generative AI with vision-language models (VLMs) to enable the robotic assembly of multi-component objects from natural language. Our method leverages VLMs for zero-shot, multi-modal reasoning about geometry and functionality to decompose AI-generated meshes into multi-component 3D models using predefined structural and panel components. We demonstrate that a VLM is capable of determining which mesh regions need panel components in addition to structural components, based on object functionality. Evaluation across test objects shows that users preferred the VLM-generated assignments 90.6% of the time, compared to 59.4% for rule-based and 2.5% for random assignment. Lastly, the system allows users to refine component assignments through conversational feedback, enabling greater human control and agency in making physical objects with generative AI and robotics.
>
---
#### [new 016] Census-Based Population Autonomy For Distributed Robotic Teaming
- **分类: cs.RO; cs.MA; cs.SY; eess.SY**

- **简介: 该论文提出基于“人口统计”（census）的分层多机器人自主模型，结合非线性意见动力学与区间规划，解决分布式机器人团队中集体决策与个体行为优化的协同问题，并通过海上无人船实验验证了其在任务分配中的有效性。**

- **链接: [http://arxiv.org/pdf/2511.02147v1](http://arxiv.org/pdf/2511.02147v1)**

> **作者:** Tyler M. Paine; Anastasia Bizyaeva; Michael R. Benjamin
>
> **备注:** 16 pages, 17 figures
>
> **摘要:** Collaborating teams of robots show promise due in their ability to complete missions more efficiently and with improved robustness, attributes that are particularly useful for systems operating in marine environments. A key issue is how to model, analyze, and design these multi-robot systems to realize the full benefits of collaboration, a challenging task since the domain of multi-robot autonomy encompasses both collective and individual behaviors. This paper introduces a layered model of multi-robot autonomy that uses the principle of census, or a weighted count of the inputs from neighbors, for collective decision-making about teaming, coupled with multi-objective behavior optimization for individual decision-making about actions. The census component is expressed as a nonlinear opinion dynamics model and the multi-objective behavior optimization is accomplished using interval programming. This model can be reduced to recover foundational algorithms in distributed optimization and control, while the full model enables new types of collective behaviors that are useful in real-world scenarios. To illustrate these points, a new method for distributed optimization of subgroup allocation is introduced where robots use a gradient descent algorithm to minimize portions of the cost functions that are locally known, while being influenced by the opinion states from neighbors to account for the unobserved costs. With this method the group can collectively use the information contained in the Hessian matrix of the total global cost. The utility of this model is experimentally validated in three categorically different experiments with fleets of autonomous surface vehicles: an adaptive sampling scenario, a high value unit protection scenario, and a competitive game of capture the flag.
>
---
#### [new 017] SuckTac: Camera-based Tactile Sucker for Unstructured Surface Perception and Interaction
- **分类: cs.RO**

- **简介: 论文提出SuckTac，一种基于摄像头的智能吸盘，解决传统吸盘缺乏高精度触觉感知的问题。通过结构优化与嵌入式成像系统，实现对不规则表面形貌、纹理的高密度感知，提升复杂环境下的抓取与操作鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.02294v1](http://arxiv.org/pdf/2511.02294v1)**

> **作者:** Ruiyong Yuan; Jieji Ren; Zhanxuan Peng; Feifei Chen; Guoying Gu
>
> **摘要:** Suckers are significant for robots in picking, transferring, manipulation and locomotion on diverse surfaces. However, most of the existing suckers lack high-fidelity perceptual and tactile sensing, which impedes them from resolving the fine-grained geometric features and interaction status of the target surface. This limits their robust performance with irregular objects and in complex, unstructured environments. Inspired by the adaptive structure and high-performance sensory capabilities of cephalopod suckers, in this paper, we propose a novel, intelligent sucker, named SuckTac, that integrates a camera-based tactile sensor directly within its optimized structure to provide high-density perception and robust suction. Specifically, through joint structure design and optimization and based on a multi-material integrated casting technique, a camera and light source are embedded into the sucker, which enables in-situ, high-density perception of fine details like surface shape, texture and roughness. To further enhance robustness and adaptability, the sucker's mechanical design is also optimized by refining its profile, adding a compliant lip, and incorporating surface microstructure. Extensive experiments, including challenging tasks such as robotic cloth manipulation and soft mobile robot inspection, demonstrate the superior performance and broad applicability of the proposed system.
>
---
#### [new 018] From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究轻量级视觉语言模型在边缘设备上的零样本场景理解与动作识别能力，解决移动机器人实时感知的计算效率与精度平衡问题，通过多场景实测评估小模型部署可行性。**

- **链接: [http://arxiv.org/pdf/2511.02427v1](http://arxiv.org/pdf/2511.02427v1)**

> **作者:** Nicolas Schuler; Lea Dewald; Nick Baldig; Jürgen Graf
>
> **备注:** 15 pages, 6 figures, 1 table; accepted for AI-2025 Forty-fifth SGAI International Conference on Artificial Intelligence CAMBRIDGE, ENGLAND 16-18 DECEMBER 2025
>
> **摘要:** Video Understanding, Scene Interpretation and Commonsense Reasoning are highly challenging tasks enabling the interpretation of visual information, allowing agents to perceive, interact with and make rational decisions in its environment. Large Language Models (LLMs) and Visual Language Models (VLMs) have shown remarkable advancements in these areas in recent years, enabling domain-specific applications as well as zero-shot open vocabulary tasks, combining multiple domains. However, the required computational complexity poses challenges for their application on edge devices and in the context of Mobile Robotics, especially considering the trade-off between accuracy and inference time. In this paper, we investigate the capabilities of state-of-the-art VLMs for the task of Scene Interpretation and Action Recognition, with special regard to small VLMs capable of being deployed to edge devices in the context of Mobile Robotics. The proposed pipeline is evaluated on a diverse dataset consisting of various real-world cityscape, on-campus and indoor scenarios. The experimental evaluation discusses the potential of these small models on edge devices, with particular emphasis on challenges, weaknesses, inherent model biases and the application of the gained information. Supplementary material is provided via the following repository: https://datahub.rz.rptu.de/hstr-csrl-public/publications/scene-interpretation-on-edge-devices/
>
---
#### [new 019] Keeping it Local, Tiny and Real: Automated Report Generation on Edge Computing Devices for Mechatronic-Based Cognitive Systems
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种基于边缘计算的本地化自然语言报告生成管道，用于多模态传感器数据的实时分析，解决机器人系统在隐私敏感场景中依赖云端服务的问题，实现私密、轻量、实时的自动化评估。**

- **链接: [http://arxiv.org/pdf/2511.02507v1](http://arxiv.org/pdf/2511.02507v1)**

> **作者:** Nicolas Schuler; Lea Dewald; Jürgen Graf
>
> **备注:** 6 pages, 4 figures, 1 table; accepted for MECATRONICS-REM 2025 International Conference, PARIS, FRANCE December 3-5 2025
>
> **摘要:** Recent advancements in Deep Learning enable hardware-based cognitive systems, that is, mechatronic systems in general and robotics in particular with integrated Artificial Intelligence, to interact with dynamic and unstructured environments. While the results are impressive, the application of such systems to critical tasks like autonomous driving as well as service and care robotics necessitate the evaluation of large amount of heterogeneous data. Automated report generation for Mobile Robotics can play a crucial role in facilitating the evaluation and acceptance of such systems in various domains. In this paper, we propose a pipeline for generating automated reports in natural language utilizing various multi-modal sensors that solely relies on local models capable of being deployed on edge computing devices, thus preserving the privacy of all actors involved and eliminating the need for external services. In particular, we evaluate our implementation on a diverse dataset spanning multiple domains including indoor, outdoor and urban environments, providing quantitative as well as qualitative evaluation results. Various generated example reports and other supplementary materials are available via a public repository.
>
---
#### [new 020] Many-vs-Many Missile Guidance via Virtual Targets
- **分类: eess.SY; cs.LG; cs.RO; cs.SY**

- **简介: 该论文提出基于归一化流的虚拟目标（VT）方法，解决多对多导弹拦截中如何利用数量优势的问题。通过生成目标轨迹概率分布，引导拦截器追踪虚拟目标而非真实目标，提升拦截概率，显著优于传统确定性预测方法。**

- **链接: [http://arxiv.org/pdf/2511.02526v1](http://arxiv.org/pdf/2511.02526v1)**

> **作者:** Marc Schneider; Walter Fichter
>
> **备注:** will be submitted to Journal of Guidance, Control, and Dynamics as Technical Note
>
> **摘要:** This paper presents a novel approach to many-vs-many missile guidance using virtual targets (VTs) generated by a Normalizing Flows-based trajectory predictor. Rather than assigning n interceptors directly to m physical targets through conventional weapon target assignment algorithms, we propose a centralized strategy that constructs n VT trajectories representing probabilistic predictions of maneuvering target behavior. Each interceptor is guided toward its assigned VT using Zero-Effort-Miss guidance during midcourse flight, transitioning to Proportional Navigation guidance for terminal interception. This approach treats many-vs-many engagements as many-vs-distribution scenarios, exploiting numerical superiority (n > m) by distributing interceptors across diverse trajectory hypotheses rather than pursuing identical deterministic predictions. Monte Carlo simulations across various target-interceptor configurations (1-6 targets, 1-8 interceptors) demonstrate that the VT method matches or exceeds baseline straight-line prediction performance by 0-4.1% when n = m, with improvements increasing to 5.8-14.4% when n > m. The results confirm that probabilistic VTs enable effective exploitation of numerical superiority, significantly increasing interception probability in many-vs-many scenarios.
>
---
#### [new 021] Synthetic Crop-Weed Image Generation and its Impact on Model Generalization
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向农业除草机器人，解决真实数据标注成本高的问题，提出基于Blender的合成作物-杂草图像生成方法，提升模型跨域泛化能力，实验表明合成数据可缩小sim-to-real差距并优于真实数据。**

- **链接: [http://arxiv.org/pdf/2511.02417v1](http://arxiv.org/pdf/2511.02417v1)**

> **作者:** Garen Boyadjian; Cyrille Pierre; Johann Laconte; Riccardo Bertoglio
>
> **摘要:** Precise semantic segmentation of crops and weeds is necessary for agricultural weeding robots. However, training deep learning models requires large annotated datasets, which are costly to obtain in real fields. Synthetic data can reduce this burden, but the gap between simulated and real images remains a challenge. In this paper, we present a pipeline for procedural generation of synthetic crop-weed images using Blender, producing annotated datasets under diverse conditions of plant growth, weed density, lighting, and camera angle. We benchmark several state-of-the-art segmentation models on synthetic and real datasets and analyze their cross-domain generalization. Our results show that training on synthetic images leads to a sim-to-real gap of 10%, surpassing previous state-of-the-art methods. Moreover, synthetic data demonstrates good generalization properties, outperforming real datasets in cross-domain scenarios. These findings highlight the potential of synthetic agricultural datasets and support hybrid strategies for more efficient model training.
>
---
#### [new 022] iFlyBot-VLA Technical Report
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: iFlyBot-VLA提出一种视觉-语言-动作（VLA）模型，通过双层动作表征与混合训练，提升机器人在复杂操作任务中的感知与决策能力，实现语言指令到精确动作的端到端生成。**

- **链接: [http://arxiv.org/pdf/2511.01914v1](http://arxiv.org/pdf/2511.01914v1)**

> **作者:** Yuan Zhang; Chenyu Xue; Wenjie Xu; Chao Ji; Jiajia wu; Jia Pan
>
> **摘要:** We introduce iFlyBot-VLA, a large-scale Vision-Language-Action (VLA) model trained under a novel framework. The main contributions are listed as follows: (1) a latent action model thoroughly trained on large-scale human and robotic manipulation videos; (2) a dual-level action representation framework that jointly supervises both the Vision-Language Model (VLM) and the action expert during training; (3) a mixed training strategy that combines robot trajectory data with general QA and spatial QA datasets, effectively enhancing the 3D perceptual and reasoning capabilities of the VLM backbone. Specifically, the VLM is trained to predict two complementary forms of actions: latent actions, derived from our latent action model pretrained on cross-embodiment manipulation data, which capture implicit high-level intentions; and structured discrete action tokens, obtained through frequency-domain transformations of continuous control signals, which encode explicit low-level dynamics. This dual supervision aligns the representation spaces of language, vision, and action, enabling the VLM to directly contribute to action generation. Experimental results on the LIBERO Franka benchmark demonstrate the superiority of our frame-work, while real-world evaluations further show that iFlyBot-VLA achieves competitive success rates across diverse and challenging manipulation tasks. Furthermore, we plan to open-source a portion of our self-constructed dataset to support future research in the community
>
---
#### [new 023] Path-Coordinated Continual Learning with Neural Tangent Kernel-Justified Plasticity: A Theoretical Framework with Near State-of-the-Art Performance
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出一种基于神经切线核（NTK）的路径协调持续学习框架，解决灾难性遗忘问题。通过理论分析与统计验证，优化参数路径选择，显著降低遗忘率，提升任务保留率，并揭示NTK条件数与学习能力的临界关系。**

- **链接: [http://arxiv.org/pdf/2511.02025v1](http://arxiv.org/pdf/2511.02025v1)**

> **作者:** Rathin Chandra Shit
>
> **备注:** Under review, IEEE Letters
>
> **摘要:** Catastrophic forgetting is one of the fundamental issues of continual learning because neural networks forget the tasks learned previously when trained on new tasks. The proposed framework is a new path-coordinated framework of continual learning that unites the Neural Tangent Kernel (NTK) theory of principled plasticity bounds, statistical validation by Wilson confidence intervals, and evaluation of path quality by the use of multiple metrics. Experimental evaluation shows an average accuracy of 66.7% at the cost of 23.4% catastrophic forgetting on Split-CIFAR10, a huge improvement over the baseline and competitive performance achieved, which is very close to state-of-the-art results. Further, it is found out that NTK condition numbers are predictive indicators of learning capacity limits, showing the existence of a critical threshold at condition number $>10^{11}$. It is interesting to note that the proposed strategy shows a tendency of lowering forgetting as the sequence of tasks progresses (27% to 18%), which is a system stabilization. The framework validates 80% of discovered paths with a rigorous statistical guarantee and maintains 90-97% retention on intermediate tasks. The core capacity limits of the continual learning environment are determined in the analysis, and actionable insights to enhance the adaptive regularization are offered.
>
---
#### [new 024] Cycle-Sync: Robust Global Camera Pose Estimation through Enhanced Cycle-Consistent Synchronization
- **分类: cs.CV; cs.NA; cs.RO; math.NA; stat.ME; 90C26, 90C17, 68Q87, 65C20, 90-08, 60-08; G.1.6; I.4.0**

- **简介: Cycle-Sync提出一种全局相机位姿估计算法，通过增强循环一致性与鲁棒优化，无需距离信息即可精确恢复相机位置与旋转，避免束调整，显著提升鲁棒性与精度。**

- **链接: [http://arxiv.org/pdf/2511.02329v1](http://arxiv.org/pdf/2511.02329v1)**

> **作者:** Shaohan Li; Yunpeng Shi; Gilad Lerman
>
> **备注:** NeurIPS 2025 spotlight paper
>
> **摘要:** We introduce Cycle-Sync, a robust and global framework for estimating camera poses (both rotations and locations). Our core innovation is a location solver that adapts message-passing least squares (MPLS) -- originally developed for group synchronization -- to camera location estimation. We modify MPLS to emphasize cycle-consistent information, redefine cycle consistencies using estimated distances from previous iterations, and incorporate a Welsch-type robust loss. We establish the strongest known deterministic exact-recovery guarantee for camera location estimation, showing that cycle consistency alone -- without access to inter-camera distances -- suffices to achieve the lowest sample complexity currently known. To further enhance robustness, we introduce a plug-and-play outlier rejection module inspired by robust subspace recovery, and we fully integrate cycle consistency into MPLS for rotation synchronization. Our global approach avoids the need for bundle adjustment. Experiments on synthetic and real datasets show that Cycle-Sync consistently outperforms leading pose estimators, including full structure-from-motion pipelines with bundle adjustment.
>
---
## 更新

#### [replaced 001] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.27607v2](http://arxiv.org/pdf/2510.27607v2)**

> **作者:** John Won; Kyungmin Lee; Huiwon Jang; Dongyoung Kim; Jinwoo Shin
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Recently, augmenting vision-language-action models (VLAs) with world-models has shown promise in robotic policy learning. However, it remains challenging to jointly predict next-state observations and action sequences because of the inherent difference between the two modalities. To address this, we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework that handles the modality conflict and enhances the performance of VLAs across diverse tasks. Specifically, we propose a multimodal diffusion transformer architecture that explicitly maintains separate modality streams while enabling cross-modal knowledge sharing. In addition, we propose training techniques such as independent noise perturbations for each modality and a decoupled flow matching loss, which enables the model to learn the joint distribution in a bidirectional manner while avoiding the need for a unified latent space. Furthermore, based on the decoupled training framework, we introduce a sampling method where we sample action and vision tokens asynchronously at different rates, which shows improvement through inference-time scaling. Through experiments on simulated benchmarks such as RoboCasa and GR-1, DUST achieves up to 6% gains over a standard VLA baseline and implicit world-modeling methods, with our inference-time scaling approach providing an additional 2-5% gain on success rate. On real-world tasks with the Franka Research 3, DUST outperforms baselines in success rate by 13%, confirming its effectiveness beyond simulation. Lastly, we demonstrate the effectiveness of DUST in large-scale pretraining with action-free videos from BridgeV2, where DUST leads to significant gain when transferred to the RoboCasa benchmark.
>
---
#### [replaced 002] Rethinking Bimanual Robotic Manipulation: Learning with Decoupled Interaction Framework
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.09186v2](http://arxiv.org/pdf/2503.09186v2)**

> **作者:** Jian-Jian Jiang; Xiao-Ming Wu; Yi-Xiang He; Ling-An Zeng; Yi-Lin Wei; Dandan Zhang; Wei-Shi Zheng
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Bimanual robotic manipulation is an emerging and critical topic in the robotics community. Previous works primarily rely on integrated control models that take the perceptions and states of both arms as inputs to directly predict their actions. However, we think bimanual manipulation involves not only coordinated tasks but also various uncoordinated tasks that do not require explicit cooperation during execution, such as grasping objects with the closest hand, which integrated control frameworks ignore to consider due to their enforced cooperation in the early inputs. In this paper, we propose a novel decoupled interaction framework that considers the characteristics of different tasks in bimanual manipulation. The key insight of our framework is to assign an independent model to each arm to enhance the learning of uncoordinated tasks, while introducing a selective interaction module that adaptively learns weights from its own arm to improve the learning of coordinated tasks. Extensive experiments on seven tasks in the RoboTwin dataset demonstrate that: (1) Our framework achieves outstanding performance, with a 23.5% boost over the SOTA method. (2) Our framework is flexible and can be seamlessly integrated into existing methods. (3) Our framework can be effectively extended to multi-agent manipulation tasks, achieving a 28% boost over the integrated control SOTA. (4) The performance boost stems from the decoupled design itself, surpassing the SOTA by 16.5% in success rate with only 1/6 of the model size.
>
---
#### [replaced 003] FRASA: An End-to-End Reinforcement Learning Agent for Fall Recovery and Stand Up of Humanoid Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.08655v3](http://arxiv.org/pdf/2410.08655v3)**

> **作者:** Clément Gaspard; Marc Duclusaud; Grégoire Passault; Mélodie Daniel; Olivier Ly
>
> **摘要:** Humanoid robotics faces significant challenges in achieving stable locomotion and recovering from falls in dynamic environments. Traditional methods, such as Model Predictive Control (MPC) and Key Frame Based (KFB) routines, either require extensive fine-tuning or lack real-time adaptability. This paper introduces FRASA, a Deep Reinforcement Learning (DRL) agent that integrates fall recovery and stand up strategies into a unified framework. Leveraging the Cross-Q algorithm, FRASA significantly reduces training time and offers a versatile recovery strategy that adapts to unpredictable disturbances. Comparative tests on Sigmaban humanoid robots demonstrate FRASA superior performance against the KFB method deployed in the RoboCup 2023 by the Rhoban Team, world champion of the KidSize League.
>
---
#### [replaced 004] Mobile Robotic Multi-View Photometric Stereo
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.10842v2](http://arxiv.org/pdf/2502.10842v2)**

> **作者:** Suryansh Kumar
>
> **备注:** Acknowledgment Added. Published at International Society Journal of Photogrammetry and Remote Sensing (ISPRS). 32 pages, 14 Figures, 5 Tables
>
> **摘要:** Multi-View Photometric Stereo (MVPS) is a popular method for fine-detailed 3D acquisition of an object from images. Despite its outstanding results on diverse material objects, a typical MVPS experimental setup requires a well-calibrated light source and a monocular camera installed on an immovable base. This restricts the use of MVPS on a movable platform, limiting us from taking MVPS benefits in 3D acquisition for mobile robotics applications. To this end, we introduce a new mobile robotic system for MVPS. While the proposed system brings advantages, it introduces additional algorithmic challenges. Addressing them, in this paper, we further propose an incremental approach for mobile robotic MVPS. Our approach leverages a supervised learning setup to predict per-view surface normal, object depth, and per-pixel uncertainty in model-predicted results. A refined depth map per view is obtained by solving an MVPS-driven optimization problem proposed in this paper. Later, we fuse the refined depth map while tracking the camera pose w.r.t the reference frame to recover globally consistent object 3D geometry. Experimental results show the advantages of our robotic system and algorithm, featuring the local high-frequency surface detail recovery with globally consistent object shape. Our work is beyond any MVPS system yet presented, providing encouraging results on objects with unknown reflectance properties using fewer frames without a tiring calibration and installation process, enabling computationally efficient robotic automation approach to photogrammetry. The proposed approach is nearly 100 times computationally faster than the state-of-the-art MVPS methods such as [1, 2] while maintaining the similar results when tested on subjects taken from the benchmark DiLiGenT MV dataset [3].
>
---
#### [replaced 005] Learning Terrain-Specialized Policies for Adaptive Locomotion in Challenging Environments
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.20635v2](http://arxiv.org/pdf/2509.20635v2)**

> **作者:** Matheus P. Angarola; Francisco Affonso; Marcelo Becker
>
> **备注:** Accepted to the 22nd International Conference on Advanced Robotics (ICAR 2025). 7 pages
>
> **摘要:** Legged robots must exhibit robust and agile locomotion across diverse, unstructured terrains, a challenge exacerbated under blind locomotion settings where terrain information is unavailable. This work introduces a hierarchical reinforcement learning framework that leverages terrain-specialized policies and curriculum learning to enhance agility and tracking performance in complex environments. We validated our method on simulation, where our approach outperforms a generalist policy by up to 16% in success rate and achieves lower tracking errors as the velocity target increases, particularly on low-friction and discontinuous terrains, demonstrating superior adaptability and robustness across mixed-terrain scenarios.
>
---
#### [replaced 006] Closing the Intent-to-Behavior Gap via Fulfillment Priority Logic
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.05818v3](http://arxiv.org/pdf/2503.05818v3)**

> **作者:** Bassel El Mabsout; Abdelrahman Abdelgawad; Renato Mancuso
>
> **摘要:** Practitioners designing reinforcement learning policies face a fundamental challenge: translating intended behavioral objectives into representative reward functions. This challenge stems from behavioral intent requiring simultaneous achievement of multiple competing objectives, typically addressed through labor-intensive linear reward composition that yields brittle results. Consider the ubiquitous robotics scenario where performance maximization directly conflicts with energy conservation. Such competitive dynamics are resistant to simple linear reward combinations. In this paper, we present the concept of objective fulfillment upon which we build Fulfillment Priority Logic (FPL). FPL allows practitioners to define logical formula representing their intentions and priorities within multi-objective reinforcement learning. Our novel Balanced Policy Gradient algorithm leverages FPL specifications to achieve up to 500\% better sample efficiency compared to Soft Actor Critic. Notably, this work constitutes the first implementation of non-linear utility scalarization design, specifically for continuous control problems.
>
---
#### [replaced 007] Grounded Vision-Language Interpreter for Integrated Task and Motion Planning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.03270v2](http://arxiv.org/pdf/2506.03270v2)**

> **作者:** Jeremy Siburian; Keisuke Shirai; Cristian C. Beltran-Hernandez; Masashi Hamaya; Michael Görner; Atsushi Hashimoto
>
> **备注:** Project website: https://omron-sinicx.github.io/ViLaIn-TAMP/
>
> **摘要:** While recent advances in vision-language models have accelerated the development of language-guided robot planners, their black-box nature often lacks safety guarantees and interpretability crucial for real-world deployment. Conversely, classical symbolic planners offer rigorous safety verification but require significant expert knowledge for setup. To bridge the current gap, this paper proposes ViLaIn-TAMP, a hybrid planning framework for enabling verifiable, interpretable, and autonomous robot behaviors. ViLaIn-TAMP comprises three main components: (1) a Vision-Language Interpreter (ViLaIn) adapted from previous work that converts multimodal inputs into structured problem specifications, (2) a modular Task and Motion Planning (TAMP) system that grounds these specifications in actionable trajectory sequences through symbolic and geometric constraint reasoning, and (3) a corrective planning (CP) module which receives concrete feedback on failed solution attempts and feed them with constraints back to ViLaIn to refine the specification. We design challenging manipulation tasks in a cooking domain and evaluate our framework. Experimental results demonstrate that ViLaIn-TAMP outperforms a VLM-as-a-planner baseline by 18% in mean success rate, and that adding the CP module boosts mean success rate by 32%.
>
---
#### [replaced 008] Tactile Displays Driven by Projected Light
- **分类: cs.ET; cs.HC; cs.RO; physics.optics**

- **链接: [http://arxiv.org/pdf/2410.05494v4](http://arxiv.org/pdf/2410.05494v4)**

> **作者:** Max Linnander; Dustin Goetz; Gregory Reardon; Vijay Kumar; Elliot Hawkes; Yon Visell
>
> **摘要:** Tactile displays that lend tangible form to digital content could transform computing interactions. However, achieving the resolution, speed, and dynamic range needed for perceptual fidelity remains challenging. We present a tactile display that directly converts projected light into visible tactile patterns via a photomechanical surface populated with millimeter-scale optotactile pixels. The pixels transduce incident light into mechanical displacements through photostimulated thermal gas expansion, yielding millimeter scale displacements with response times of 2 to 100 milliseconds. Employing projected light for power transmission and addressing renders these displays highly scalable. We demonstrate optically driven displays with up to 1,511 addressable pixels -- several times more pixels than any prior tactile display attaining comparable performance. Perceptual studies confirm that these displays can reproduce diverse spatiotemporal tactile patterns with high fidelity. This research establishes a foundation for practical, versatile high-resolution tactile displays driven by light.
>
---
#### [replaced 009] Light Future: Multimodal Action Frame Prediction via InstructPix2Pix
- **分类: cs.CV; cs.MM; cs.RO; I.2.10; I.4.8**

- **链接: [http://arxiv.org/pdf/2507.14809v2](http://arxiv.org/pdf/2507.14809v2)**

> **作者:** Zesen Zhong; Duomin Zhang; Yijia Li
>
> **备注:** 9 pages including appendix, 4 tables, 8 figures, to be submitted to WACV 2026
>
> **摘要:** Predicting future motion trajectories is a critical capability across domains such as robotics, autonomous systems, and human activity forecasting, enabling safer and more intelligent decision-making. This paper proposes a novel, efficient, and lightweight approach for robot action prediction, offering significantly reduced computational cost and inference latency compared to conventional video prediction models. Importantly, it pioneers the adaptation of the InstructPix2Pix model for forecasting future visual frames in robotic tasks, extending its utility beyond static image editing. We implement a deep learning-based visual prediction framework that forecasts what a robot will observe 100 frames (10 seconds) into the future, given a current image and a textual instruction. We repurpose and fine-tune the InstructPix2Pix model to accept both visual and textual inputs, enabling multimodal future frame prediction. Experiments on the RoboTWin dataset (generated based on real-world scenarios) demonstrate that our method achieves superior SSIM and PSNR compared to state-of-the-art baselines in robot action prediction tasks. Unlike conventional video prediction models that require multiple input frames, heavy computation, and slow inference latency, our approach only needs a single image and a text prompt as input. This lightweight design enables faster inference, reduced GPU demands, and flexible multimodal control, particularly valuable for applications like robotics and sports motion trajectory analytics, where motion trajectory precision is prioritized over visual fidelity.
>
---
#### [replaced 010] Virtual Target Trajectory Prediction for Stochastic Targets
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.01851v2](http://arxiv.org/pdf/2504.01851v2)**

> **作者:** Marc Schneider; Renato Loureiro; Torbjørn Cunis; Walter Fichter
>
> **备注:** Manuscript accepted by Journal of Guidance, Control, and Dynamics
>
> **摘要:** Trajectory prediction of aerial vehicles is a key requirement in applications ranging from missile guidance to UAV collision avoidance. While most prediction methods assume deterministic target motion, real-world targets often exhibit stochastic behaviors such as evasive maneuvers or random gliding patterns. This paper introduces a probabilistic framework based on Conditional Normalizing Flows (CNFs) to model and predict such stochastic dynamics directly from trajectory data. The learned model generates probability distributions of future target positions conditioned on initial states and dynamic parameters, enabling efficient sampling and exact density evaluation. To provide deterministic surrogates compatible with existing guidance and planning algorithms, sampled trajectories are clustered using a time series k-means approach, yielding a set of representative "virtual target" trajectories. The method is target-agnostic, computationally efficient, and requires only trajectory data for training, making it suitable as a drop-in replacement for deterministic predictors. Simulated scenarios with maneuvering and ballistic targets demonstrate that the proposed approach bridges the gap between deterministic assumptions and stochastic reality, advancing guidance and control algorithms for autonomous vehicles.
>
---
#### [replaced 011] MetAdv: A Unified and Interactive Adversarial Testing Platform for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.06534v3](http://arxiv.org/pdf/2508.06534v3)**

> **作者:** Aishan Liu; Jiakai Wang; Tianyuan Zhang; Hainan Li; Jiangfan Liu; Siyuan Liang; Yilong Ren; Xianglong Liu; Dacheng Tao
>
> **备注:** ACM MM 2025 Most Popular Demo Award
>
> **摘要:** Evaluating and ensuring the adversarial robustness of autonomous driving (AD) systems is a critical and unresolved challenge. This paper introduces MetAdv, a novel adversarial testing platform that enables realistic, dynamic, and interactive evaluation by tightly integrating virtual simulation with physical vehicle feedback. At its core, MetAdv establishes a hybrid virtual-physical sandbox, within which we design a three-layer closed-loop testing environment with dynamic adversarial test evolution. This architecture facilitates end-to-end adversarial evaluation, ranging from high-level unified adversarial generation, through mid-level simulation-based interaction, to low-level execution on physical vehicles. Additionally, MetAdv supports a broad spectrum of AD tasks, algorithmic paradigms (e.g., modular deep learning pipelines, end-to-end learning, vision-language models). It supports flexible 3D vehicle modeling and seamless transitions between simulated and physical environments, with built-in compatibility for commercial platforms such as Apollo and Tesla. A key feature of MetAdv is its human-in-the-loop capability: besides flexible environmental configuration for more customized evaluation, it enables real-time capture of physiological signals and behavioral feedback from drivers, offering new insights into human-machine trust under adversarial conditions. We believe MetAdv can offer a scalable and unified framework for adversarial assessment, paving the way for safer AD.
>
---
#### [replaced 012] Kinematify: Open-Vocabulary Synthesis of High-DoF Articulated Objects
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2511.01294v2](http://arxiv.org/pdf/2511.01294v2)**

> **作者:** Jiawei Wang; Dingyou Wang; Jiaming Hu; Qixuan Zhang; Jingyi Yu; Lan Xu
>
> **备注:** project page: https://sites.google.com/deemos.com/kinematify
>
> **摘要:** A deep understanding of kinematic structures and movable components is essential for enabling robots to manipulate objects and model their own articulated forms. Such understanding is captured through articulated objects, which are essential for tasks such as physical simulation, motion planning, and policy learning. However, creating these models, particularly for objects with high degrees of freedom (DoF), remains a significant challenge. Existing methods typically rely on motion sequences or strong assumptions from hand-curated datasets, which hinders scalability. In this paper, we introduce Kinematify, an automated framework that synthesizes articulated objects directly from arbitrary RGB images or textual descriptions. Our method addresses two core challenges: (i) inferring kinematic topologies for high-DoF objects and (ii) estimating joint parameters from static geometry. To achieve this, we combine MCTS search for structural inference with geometry-driven optimization for joint reasoning, producing physically consistent and functionally valid descriptions. We evaluate Kinematify on diverse inputs from both synthetic and real-world environments, demonstrating improvements in registration and kinematic topology accuracy over prior work.
>
---
#### [replaced 013] Neural Network Aided Kalman Filtering with Model Predictive Control Enables Robot-Assisted Drone Recovery on a Wavy Surface
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.09145v3](http://arxiv.org/pdf/2505.09145v3)**

> **作者:** Yimou Wu; Mingyang Liang; Chongfeng Liu; Zhongzhong Cao; Huihuan Qian
>
> **备注:** 17 pages, 51 figures
>
> **摘要:** Recovering a drone on a disturbed water surface remains a significant challenge in maritime robotics. In this paper, we propose a unified framework for robot-assisted drone recovery on a wavy surface that addresses two major tasks: Firstly, accurate prediction of a moving drone's position under wave-induced disturbances using KalmanNet Plus Plus (KalmanNet++), a Neural Network Aided Kalman Filtering we proposed. Secondly, effective motion planning using the desired position we got for a manipulator via Receding Horizon Model Predictive Control (RHMPC). Specifically, we compared multiple prediction methods and proposed KalmanNet Plus Plus to predict the position of the UAV, thereby obtaining the desired position. The KalmanNet++ predicts the drone's future position 0.1\,s ahead, while the manipulator plans a capture trajectory in real time, thus overcoming not only wave-induced base motions but also limited constraints such as torque constraints and joint constraints. For the system design, we provide a collaborative system, comprising a manipulator subsystem and a UAV subsystem, enables drone lifting and drone recovery. Simulation and real-world experiments using wave-disturbed motion data demonstrate that our approach achieves a high success rate - above 95\% and outperforms conventional baseline methods by up to 10\% in efficiency and 20\% in precision. The results underscore the feasibility and robustness of our system, which achieves state-of-the-art performance and offers a practical solution for maritime drone operations.
>
---
#### [replaced 014] Radar-Based Odometry for Low-Speed Driving
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.07683v3](http://arxiv.org/pdf/2509.07683v3)**

> **作者:** Luis Diener; Jens Kalkkuhl; Markus Enzweiler
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** We address automotive odometry for low-speed driving and parking, where centimeter-level accuracy is required due to tight spaces and nearby obstacles. Traditional methods using inertial-measurement units and wheel encoders require vehicle-specific calibration, making them costly for consumer-grade vehicles. To overcome this, we propose a radar-based simultaneous localization and mapping (SLAM) approach that fuses inertial and 4D radar measurements. Our approach tightly couples feature positions and Doppler velocities for accurate localization and robust data association. Key contributions include a tightly coupled radar-Doppler extended Kalman filter, multi-radar support and an information-based feature-pruning strategy. Experiments using both proprietary and public datasets demonstrate high-accuracy localization during low-speed driving.
>
---
#### [replaced 015] Interactive Identification of Granular Materials using Force Measurements
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.17606v2](http://arxiv.org/pdf/2403.17606v2)**

> **作者:** Samuli Hynninen; Tran Nguyen Le; Ville Kyrki
>
> **备注:** Accepted to 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC)
>
> **摘要:** Despite the potential the ability to identify granular materials creates for applications such as robotic cooking or earthmoving, granular material identification remains a challenging area, existing methods mostly relying on shaking the materials in closed containers. This work presents an interactive material identification framework that enables robots to identify a wide range of granular materials using only force-torque measurements. Unlike prior works, the proposed approach uses direct interaction with the materials. The approach is evaluated through experiments with a real-world dataset comprising 11 granular materials, which we also make publicly available. Results show that our method can identify a wide range of granular materials with near-perfect accuracy while relying solely on force measurements obtained from direct interaction. Further, our comprehensive data analysis and experiments show that a high-performancefeature space must combine features related to the force signal's time-domain dynamics and frequency spectrum. We account for this by proposing a combination of the raw signal and its high-frequency magnitude histogram as the suggested feature space representation. We show that the proposed feature space outperforms baselines by a significant margin. The code and data set are available at: https://irobotics.aalto.fi/identify_granular/.
>
---
#### [replaced 016] Generative World Models of Tasks: LLM-Driven Hierarchical Scaffolding for Embodied Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO; 68T05, 90C40, 91A26, 68T42, 93E35; I.2.11; I.2.6; I.2.8; I.2.9; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.04731v3](http://arxiv.org/pdf/2509.04731v3)**

> **作者:** Brennen Hill
>
> **备注:** In the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop: Embodied World Models for Decision Making (EWM)
>
> **摘要:** Recent advances in agent development have focused on scaling model size and raw interaction data, mirroring successes in large language models. However, for complex, long-horizon multi-agent tasks such as robotic soccer, this end-to-end approach often fails due to intractable exploration spaces and sparse rewards. We propose that an effective world model for decision-making must model the world's physics and also its task semantics. A systematic review of 2024 research in low-resource multi-agent soccer reveals a clear trend towards integrating symbolic and hierarchical methods, such as Hierarchical Task Networks (HTNs) and Bayesian Strategy Networks (BSNs), with multi-agent reinforcement learning (MARL). These methods decompose complex goals into manageable subgoals, creating an intrinsic curriculum that shapes agent learning. We formalize this trend into a framework for Hierarchical Task Environments (HTEs), which are essential for bridging the gap between simple, reactive behaviors and sophisticated, strategic team play. Our framework incorporates the use of Large Language Models (LLMs) as generative world models of tasks, capable of dynamically generating this scaffolding. We argue that HTEs provide a mechanism to guide exploration, generate meaningful learning signals, and train agents to internalize hierarchical structure, enabling the development of more capable and general-purpose agents with greater sample efficiency than purely end-to-end approaches.
>
---
#### [replaced 017] Extended Friction Models for the Physics Simulation of Servo Actuators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.08650v4](http://arxiv.org/pdf/2410.08650v4)**

> **作者:** Marc Duclusaud; Grégoire Passault; Vincent Padois; Olivier Ly
>
> **摘要:** Accurate physical simulation is crucial for the development and validation of control algorithms in robotic systems. Recent works in Reinforcement Learning (RL) take notably advantage of extensive simulations to produce efficient robot control. State-of-the-art servo actuator models generally fail at capturing the complex friction dynamics of these systems. This limits the transferability of simulated behaviors to real-world applications. In this work, we present extended friction models that allow to more accurately simulate servo actuator dynamics. We propose a comprehensive analysis of various friction models, present a method for identifying model parameters using recorded trajectories from a pendulum test bench, and demonstrate how these models can be integrated into physics engines. The proposed friction models are validated on four distinct servo actuators and tested on 2R manipulators, showing significant improvements in accuracy over the standard Coulomb-Viscous model. Our results highlight the importance of considering advanced friction effects in the simulation of servo actuators to enhance the realism and reliability of robotic simulations.
>
---
#### [replaced 018] No Plan but Everything Under Control: Robustly Solving Sequential Tasks with Dynamically Composed Gradient Descent
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01732v3](http://arxiv.org/pdf/2503.01732v3)**

> **作者:** Vito Mengers; Oliver Brock
>
> **备注:** Accepted at ICRA25; Supplementary Material under https://www.tu.berlin/robotics/papers/noplan ; 7 pages + 6 figures;
>
> **摘要:** We introduce a novel gradient-based approach for solving sequential tasks by dynamically adjusting the underlying myopic potential field in response to feedback and the world's regularities. This adjustment implicitly considers subgoals encoded in these regularities, enabling the solution of long sequential tasks, as demonstrated by solving the traditional planning domain of Blocks World - without any planning. Unlike conventional planning methods, our feedback-driven approach adapts to uncertain and dynamic environments, as demonstrated by one hundred real-world trials involving drawer manipulation. These experiments highlight the robustness of our method compared to planning and show how interactive perception and error recovery naturally emerge from gradient descent without explicitly implementing them. This offers a computationally efficient alternative to planning for a variety of sequential tasks, while aligning with observations on biological problem-solving strategies.
>
---
#### [replaced 019] Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05635v3](http://arxiv.org/pdf/2508.05635v3)**

> **作者:** Yue Liao; Pengfei Zhou; Siyuan Huang; Donglin Yang; Shengcong Chen; Yuxin Jiang; Yue Hu; Jingbin Cai; Si Liu; Jianlan Luo; Liliang Chen; Shuicheng Yan; Maoqing Yao; Guanghui Ren
>
> **备注:** https://genie-envisioner.github.io/
>
> **摘要:** We introduce Genie Envisioner (GE), a unified world foundation platform for robotic manipulation that integrates policy learning, evaluation, and simulation within a single video-generative framework. At its core, GE-Base is a large-scale, instruction-conditioned video diffusion model that captures the spatial, temporal, and semantic dynamics of real-world robotic interactions in a structured latent space. Built upon this foundation, GE-Act maps latent representations to executable action trajectories through a lightweight, flow-matching decoder, enabling precise and generalizable policy inference across diverse embodiments with minimal supervision. To support scalable evaluation and training, GE-Sim serves as an action-conditioned neural simulator, producing high-fidelity rollouts for closed-loop policy development. The platform is further equipped with EWMBench, a standardized benchmark suite measuring visual fidelity, physical consistency, and instruction-action alignment. Together, these components establish Genie Envisioner as a scalable and practical foundation for instruction-driven, general-purpose embodied intelligence. All code, models, and benchmarks will be released publicly.
>
---
#### [replaced 020] UniCoD: Enhancing Robot Policy via Unified Continuous and Discrete Representation Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.10642v2](http://arxiv.org/pdf/2510.10642v2)**

> **作者:** Jianke Zhang; Yucheng Hu; Yanjiang Guo; Xiaoyu Chen; Yichen Liu; Wenna Chen; Chaochao Lu; Jianyu Chen
>
> **摘要:** Building generalist robot policies that can handle diverse tasks in open-ended environments is a central challenge in robotics. To leverage knowledge from large-scale pretraining, prior work (VLA) has typically built generalist policies either on top of vision-language understanding models (VLMs) or generative models. However, both semantic understanding from vision-language pretraining and visual dynamics modeling from visual-generation pretraining are crucial for embodied robots. Recent unified models of generation and understanding have demonstrated strong capabilities in both comprehension and generation through large-scale pretraining. We posit that robotic policy learning can likewise benefit from the combined strengths of understanding, planning, and continuous future representation learning. Building on this insight, we introduce UniCoD, which acquires the ability to dynamically model high-dimensional visual features through pretraining on over 1M internet-scale instructional manipulation videos. Subsequently, UniCoD is fine-tuned on data collected from the robot embodiment, enabling the learning of mappings from predictive representations to action tokens. Extensive experiments show our approach consistently outperforms baseline methods in terms of 9\% and 12\% across simulation environments and real-world out-of-distribution tasks.
>
---
#### [replaced 021] Integrated Shape-Force Estimation for Continuum Robots: A Virtual-Work and Polynomial-Curvature Framework
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.05418v3](http://arxiv.org/pdf/2501.05418v3)**

> **作者:** Guoqing Zhang; Zihan Chen; Long Wang
>
> **摘要:** Cable-driven continuum robots (CDCRs) are widely used in surgical and inspection tasks that require dexterous manipulation in confined spaces. Existing model-based estimation methods either assume constant curvature or rely on geometry-space interpolants, both of which struggle with accuracy under large deformations and sparse sensing. This letter introduces an integrated shape-force estimation framework that combines cable-tension measurements with tip-pose data to reconstruct backbone shape and estimate external tip force simultaneously. The framework employs polynomial curvature kinematics (PCK) and a virtual-work-based static formulation expressed directly in curvature space, where polynomial modal coefficients serve as generalized coordinates. The proposed method is validated through Cosserat-rod-based simulations and hardware experiments on a torque-cell-enabled CDCR prototype. Results show that the second-order PCK model achieves superior shape and force accuracy, combining a lightweight shape optimization with a closed-form, iteration-free force estimation, offering a compact and robust alternative to prior constant-curvature and geometry-space approaches.
>
---
#### [replaced 022] Replicating Human Anatomy with Vision Controlled Jetting -- A Pneumatic Musculoskeletal Hand and Forearm
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.19077v2](http://arxiv.org/pdf/2404.19077v2)**

> **作者:** Thomas Buchner; Stefan Weirich; Alexander M. Kübler; Wojciech Matusik; Robert K. Katzschmann
>
> **摘要:** The functional replication and actuation of complex structures inspired by nature is a longstanding goal for humanity. Creating such complex structures combining soft and rigid features and actuating them with artificial muscles would further our understanding of natural kinematic structures. We printed a biomimetic hand in a single print process comprised of a rigid skeleton, soft joint capsules, tendons, and printed touch sensors. We showed it's actuation using electric motors. In this work, we expand on this work by adding a forearm that is also closely modeled after the human anatomy and replacing the hand's motors with 22 independently controlled pneumatic artificial muscles (PAMs). Our thin, high-strain (up to 30.1%) PAMs match the performance of state-of-the-art artificial muscles at a lower cost. The system showcases human-like dexterity with independent finger movements, demonstrating successful grasping of various objects, ranging from a small, lightweight coin to a large can of 272g in weight. The performance evaluation, based on fingertip and grasping forces along with finger joint range of motion, highlights the system's potential.
>
---
#### [replaced 023] The Difference between the Left and Right Invariant Extended Kalman Filter
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.04568v2](http://arxiv.org/pdf/2507.04568v2)**

> **作者:** Yixiao Ge; Giulio Delama; Martin Scheiber; Alessandro Fornasier; Pieter van Goor; Stephan Weiss; Robert Mahony
>
> **备注:** 20 pages, 4 figures, submitted to Control Engineering Practice
>
> **摘要:** The extended Kalman filter (EKF) has been the industry standard for state estimation problems over the past sixty years. The Invariant Extended Kalman Filter (IEKF) is a recent development of the EKF for the class of group-affine systems on Lie groups that has shown superior performance for inertial navigation problems. The IEKF comes in two versions, left- and right- handed respectively, and there is a perception in the robotics community that these filters are different and one should choose the handedness of the IEKF to match handedness of the measurement model for a given filtering problem. In this paper, we revisit these algorithms and demonstrate that the left- and right- IEKF algorithms (with reset step) are identical, that is, the choice of the handedness does not affect the IEKF's performance when the reset step is properly implemented. The reset step was not originally proposed as part of the IEKF, however, we provide simulations to show that the reset step improves asymptotic performance of all versions of the the filter, and should be included in all high performance algorithms. The GNSS-aided inertial navigation system (INS) is used as a motivating example to demonstrate the equivalence of the two filters.
>
---
#### [replaced 024] Multi-Objective Planning with Contextual Lexicographic Reward Preferences
- **分类: cs.AI; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2502.10476v2](http://arxiv.org/pdf/2502.10476v2)**

> **作者:** Pulkit Rustagi; Yashwanthi Anand; Sandhya Saisubramanian
>
> **备注:** 9 pages, 5 figures, 2 tables, To appear in Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems (AAMAS) 2025
>
> **摘要:** Autonomous agents are often required to plan under multiple objectives whose preference ordering varies based on context. The agent may encounter multiple contexts during its course of operation, each imposing a distinct lexicographic ordering over the objectives, with potentially different reward functions associated with each context. Existing approaches to multi-objective planning typically consider a single preference ordering over the objectives, across the state space, and do not support planning under multiple objective orderings within an environment. We present Contextual Lexicographic Markov Decision Process (CLMDP), a framework that enables planning under varying lexicographic objective orderings, depending on the context. In a CLMDP, both the objective ordering at a state and the associated reward functions are determined by the context. We employ a Bayesian approach to infer a state-context mapping from expert trajectories. Our algorithm to solve a CLMDP first computes a policy for each objective ordering and then combines them into a single context-aware policy that is valid and cycle-free. The effectiveness of the proposed approach is evaluated in simulation and using a mobile robot.
>
---
#### [replaced 025] Towards Predicting Any Human Trajectory In Context
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.00871v3](http://arxiv.org/pdf/2506.00871v3)**

> **作者:** Ryo Fujii; Hideo Saito; Ryo Hachiuma
>
> **备注:** NeurIPS 2025
>
> **摘要:** Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, the need to fine-tune for each new scenario is often impractical for deployment on edge devices. To address this challenge, we introduce TrajICL, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables adaptation without fine-tuning on the scenario-specific data at inference time without requiring weight updates. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. Project Page: https://fujiry0.github.io/TrajICL-project-page/.
>
---
#### [replaced 026] Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18065v3](http://arxiv.org/pdf/2503.18065v3)**

> **作者:** Ziming Wei; Bingqian Lin; Yunshuang Nie; Jiaqi Chen; Shikui Ma; Hang Xu; Xiaodan Liang
>
> **备注:** Accepted by IEEE Transactions on Neural Networks and Learning Systems
>
> **摘要:** Data scarcity is a long-standing challenge in the Vision-Language Navigation (VLN) field, which extremely hinders the generalization of agents to unseen environments. Previous works primarily rely on additional simulator data or web-collected images/videos to improve the generalization. However, the simulator environments still face limited diversity, and the web-collected data often requires extensive labor to remove the noise. In this paper, we propose a Rewriting-driven AugMentation (RAM) paradigm for VLN, which directly creates the unseen observation-instruction pairs via rewriting human-annotated training data. Benefiting from our rewriting mechanism, new observation-instruction pairs can be obtained in both simulator-free and labor-saving manners to promote generalization. Specifically, we first introduce Object-Enriched Observation Rewriting, where we combine Vision-Language Models (VLMs) and Large Language Models (LLMs) to derive rewritten object-enriched scene descriptions, enabling observation synthesis with diverse objects and spatial layouts via Text-to-Image Generation Models (T2IMs). Then, we propose Observation-Contrast Instruction Rewriting, which generates observation-aligned rewritten instructions by requiring LLMs to reason the difference between original and new observations. We further develop a mixing-then-focusing training strategy with a random observation cropping scheme, effectively enhancing data distribution diversity while suppressing augmentation data noise during training. Experiments on both the discrete environments (R2R, REVERIE, and R4R datasets) and continuous environments (R2R-CE dataset) show the superior performance and impressive generalization ability of our method. Code is available at https://github.com/SaDil13/VLN-RAM.
>
---
#### [replaced 027] DiffVLA++: Bridging Cognitive Reasoning and End-to-End Driving through Metric-Guided Alignment
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17148v4](http://arxiv.org/pdf/2510.17148v4)**

> **作者:** Yu Gao; Anqing Jiang; Yiru Wang; Wang Jijun; Hao Jiang; Zhigang Sun; Heng Yuwen; Wang Shuo; Hao Zhao; Sun Hao
>
> **摘要:** Conventional end-to-end (E2E) driving models are effective at generating physically plausible trajectories, but often fail to generalize to long-tail scenarios due to the lack of essential world knowledge to understand and reason about surrounding environments. In contrast, Vision-Language-Action (VLA) models leverage world knowledge to handle challenging cases, but their limited 3D reasoning capability can lead to physically infeasible actions. In this work we introduce DiffVLA++, an enhanced autonomous driving framework that explicitly bridges cognitive reasoning and E2E planning through metric-guided alignment. First, we build a VLA module directly generating semantically grounded driving trajectories. Second, we design an E2E module with a dense trajectory vocabulary that ensures physical feasibility. Third, and most critically, we introduce a metric-guided trajectory scorer that guides and aligns the outputs of the VLA and E2E modules, thereby integrating their complementary strengths. The experiment on the ICCV 2025 Autonomous Grand Challenge leaderboard shows that DiffVLA++ achieves EPDMS of 49.12.
>
---
#### [replaced 028] End-to-End Crop Row Navigation via LiDAR-Based Deep Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.18608v2](http://arxiv.org/pdf/2509.18608v2)**

> **作者:** Ana Luiza Mineiro; Francisco Affonso; Marcelo Becker
>
> **备注:** Accepted to the 22nd International Conference on Advanced Robotics (ICAR 2025). 7 pages
>
> **摘要:** Reliable navigation in under-canopy agricultural environments remains a challenge due to GNSS unreliability, cluttered rows, and variable lighting. To address these limitations, we present an end-to-end learning-based navigation system that maps raw 3D LiDAR data directly to control commands using a deep reinforcement learning policy trained entirely in simulation. Our method includes a voxel-based downsampling strategy that reduces LiDAR input size by 95.83%, enabling efficient policy learning without relying on labeled datasets or manually designed control interfaces. The policy was validated in simulation, achieving a 100% success rate in straight-row plantations and showing a gradual decline in performance as row curvature increased, tested across varying sinusoidal frequencies and amplitudes.
>
---
#### [replaced 029] RoboTron-Mani: All-in-One Multimodal Large Model for Robotic Manipulation
- **分类: cs.RO; cs.MM**

- **链接: [http://arxiv.org/pdf/2412.07215v2](http://arxiv.org/pdf/2412.07215v2)**

> **作者:** Feng Yan; Fanfan Liu; Liming Zheng; Yufeng Zhong; Yiyang Huang; Zechao Guan; Chengjian Feng; Lin Ma
>
> **摘要:** Recently, robotics has advanced significantly through the integration of larger models and large-scale datasets. However, challenges remain in applying these models to 3D spatial interactions and managing data collection costs. To address these issues, we propose the multimodal robotic manipulation model RoboTron-Mani and the comprehensive dataset RoboData. RoboTron-Mani, on one hand, enhances 3D perception through camera parameters and occupancy supervision. On the other hand, it further incorporates Modality-Isolation-Mask and multimodal decoder blocks based on OpenFlamingo, improving modality fusion and fine-grained perception. RoboData integrats several publicly-available datasets, achieving the first fusion of multi-view images, camera parameters, depth maps, actions, and space alignment, which facilitates comprehensive learning from diverse robotic datasets and offers one complete evaluation system. Trained on RoboData, RoboTron-Mani is the first generalist policy that surpasses expert models, enabling simultaneous evaluation of all tasks across multiple datasets, rather than being limited to specific data or task selections. Specifically, RoboTron-Mani boosts manipulation performance by increasing the average sequence length on CALVIN from 1.7 to 3.5, enabling cross-embodiment generalization, and achieving state-of-the-art results on both simulated and real-world datasets.
>
---
#### [replaced 030] Talk2Event: Grounded Understanding of Dynamic Scenes from Event Cameras
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17664v2](http://arxiv.org/pdf/2507.17664v2)**

> **作者:** Lingdong Kong; Dongyue Lu; Ao Liang; Rong Li; Yuhao Dong; Tianshuai Hu; Lai Xing Ng; Wei Tsang Ooi; Benoit R. Cottereau
>
> **备注:** NeurIPS 2025 Spotlight; 43 pages, 17 figures, 16 tables; Project Page at https://talk2event.github.io
>
> **摘要:** Event cameras offer microsecond-level latency and robustness to motion blur, making them ideal for understanding dynamic environments. Yet, connecting these asynchronous streams to human language remains an open challenge. We introduce Talk2Event, the first large-scale benchmark for language-driven object grounding in event-based perception. Built from real-world driving data, we provide over 30,000 validated referring expressions, each enriched with four grounding attributes -- appearance, status, relation to viewer, and relation to other objects -- bridging spatial, temporal, and relational reasoning. To fully exploit these cues, we propose EventRefer, an attribute-aware grounding framework that dynamically fuses multi-attribute representations through a Mixture of Event-Attribute Experts (MoEE). Our method adapts to different modalities and scene dynamics, achieving consistent gains over state-of-the-art baselines in event-only, frame-only, and event-frame fusion settings. We hope our dataset and approach will establish a foundation for advancing multimodal, temporally-aware, and language-driven perception in real-world robotics and autonomy.
>
---
#### [replaced 031] Using Fiber Optic Bundles to Miniaturize Vision-Based Tactile Sensors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2403.05500v5](http://arxiv.org/pdf/2403.05500v5)**

> **作者:** Julia Di; Zdravko Dugonjic; Will Fu; Tingfan Wu; Romeo Mercado; Kevin Sawyer; Victoria Rose Most; Gregg Kammerer; Stefanie Speidel; Richard E. Fan; Geoffrey Sonn; Mark R. Cutkosky; Mike Lambeta; Roberto Calandra
>
> **备注:** This work has been submitted to the IEEE for possible publication. The CAD design files of DIGIT Pinki are available at https://github.com/facebookresearch/digit-design
>
> **摘要:** Vision-based tactile sensors have recently become popular due to their combination of low cost, very high spatial resolution, and ease of integration using widely available miniature cameras. The associated field of view and focal length, however, are difficult to package in a human-sized finger. In this paper we employ optical fiber bundles to achieve a form factor that, at 15 mm diameter, is smaller than an average human fingertip. The electronics and camera are also located remotely, further reducing package size. The sensor achieves a spatial resolution of 0.22 mm and a minimum force resolution 5 mN for normal and shear contact forces. With these attributes, the DIGIT Pinki sensor is suitable for applications such as robotic and teleoperated digital palpation. We demonstrate its utility for palpation of the prostate gland and show that it can achieve clinically relevant discrimination of prostate stiffness for phantom and ex vivo tissue.
>
---
#### [replaced 032] Adv-BMT: Bidirectional Motion Transformer for Safety-Critical Traffic Scenario Generation
- **分类: cs.RO; cs.AI; cs.GR**

- **链接: [http://arxiv.org/pdf/2506.09485v2](http://arxiv.org/pdf/2506.09485v2)**

> **作者:** Yuxin Liu; Zhenghao Peng; Xuanhao Cui; Bolei Zhou
>
> **摘要:** Scenario-based testing is essential for validating the performance of autonomous driving (AD) systems. However, such testing is limited by the scarcity of long-tailed, safety-critical scenarios in existing datasets collected in the real world. To tackle the data issue, we propose the Adv-BMT framework, which augments real-world scenarios with diverse and realistic adversarial traffic interactions. The core component of Adv-BMT is a bidirectional motion transformer (BMT) model to perform inverse traffic motion predictions, which takes agent information in the last time step of the scenario as input, and reconstructs the traffic in the inverse of chronological order until the initial time step. The Adv-BMT framework is a two-staged pipeline: it first conducts adversarial initializations and then inverse motion predictions. Different from previous work, we do not need any collision data for pretraining, and are able to generate realistic and diverse collision interactions. Our experimental results validate the quality of generated collision scenarios by Adv-BMT: training in our augmented dataset would reduce episode collision rates by 20%. Demo and code are available at: https://metadriverse.github.io/adv-bmt/.
>
---
