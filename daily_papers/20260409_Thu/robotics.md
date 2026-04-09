# 机器人 cs.RO

- **最新发布 28 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] RoSHI: A Versatile Robot-oriented Suit for Human Data In-the-Wild
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RoSHI系统，用于在真实环境中获取人体3D姿态和形状数据。解决人机交互数据采集的便携性、鲁棒性和全局一致性问题。融合IMU与视觉传感器，提升数据质量，适用于机器人学习。**

- **链接: [https://arxiv.org/pdf/2604.07331](https://arxiv.org/pdf/2604.07331)**

> **作者:** Wenjing Margaret Mao; Jefferson Ng; Luyang Hu; Daniel Gehrig; Antonio Loquercio
>
> **备注:** 8 pages, 4 figures. *Equal contribution by first three authors. Project webpage: this https URL
>
> **摘要:** Scaling up robot learning will likely require human data containing rich and long-horizon interactions in the wild. Existing approaches for collecting such data trade off portability, robustness to occlusion, and global consistency. We introduce RoSHI, a hybrid wearable that fuses low-cost sparse IMUs with the Project Aria glasses to estimate the full 3D pose and body shape of the wearer in a metric global coordinate frame from egocentric perception. This system is motivated by the complementarity of the two sensors: IMUs provide robustness to occlusions and high-speed motions, while egocentric SLAM anchors long-horizon motion and stabilizes upper body pose. We collect a dataset of agile activities to evaluate RoSHI. On this dataset, we generally outperform other egocentric baselines and perform comparably to a state-of-the-art exocentric baseline (SAM3D). Finally, we demonstrate that the motion data recorded from our system are suitable for real-world humanoid policy learning. For videos, data and more, visit the project webpage: this https URL
>
---
#### [new 002] AEROS: A Single-Agent Operating Architecture with Embodied Capability Modules
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AEROS系统，解决机器人智能组织与执行的问题。通过模块化设计实现安全、可扩展的单智能体架构。**

- **链接: [https://arxiv.org/pdf/2604.07039](https://arxiv.org/pdf/2604.07039)**

> **作者:** Xue Qin; Simin Luan; Cong Yang; Zhijun Li
>
> **备注:** Submitted to Engineering Applications of Artificial Intelligence (EAAI). 48 pages, 5 figures, 9 tables
>
> **摘要:** Robotic systems lack a principled abstraction for organizing intelligence, capabilities, and execution in a unified manner. Existing approaches either couple skills within monolithic architectures or decompose functionality into loosely coordinated modules or multiple agents, often without a coherent model of identity and control authority. We argue that a robot should be modeled as a single persistent intelligent subject whose capabilities are extended through installable packages. We formalize this view as AEROS (Agent Execution Runtime Operating System), in which each robot corresponds to one persistent agent and capabilities are provided through Embodied Capability Modules (ECMs). Each ECM encapsulates executable skills, models, and tools, while execution constraints and safety guarantees are enforced by a policy-separated runtime. This separation enables modular extensibility, composable capability execution, and consistent system-level safety. We evaluate a reference implementation in PyBullet simulation with a Franka Panda 7-DOF manipulator across eight experiments covering re-planning, failure recovery, policy enforcement, baseline comparison, cross-task generality, ECM hot-swapping, ablation, and failure boundary analysis. Over 100 randomized trials per condition, AEROS achieves 100% task success across three tasks versus baselines (this http URL-style and ProgPrompt-style at 92--93%, flat pipeline at 67--73%), the policy layer blocks all invalid actions with zero false acceptances, runtime benefits generalize across tasks without task-specific tuning, and ECMs load at runtime with 100% post-swap success.
>
---
#### [new 003] Genie Sim PanoRecon: Fast Immersive Scene Generation from Single-View Panorama
- **分类: cs.RO**

- **简介: 该论文提出Genie Sim PanoRecon，用于从单视角全景图快速生成高保真3D场景，解决机器人操作仿真中的场景重建问题。通过并行处理与深度感知融合，实现秒级重构。**

- **链接: [https://arxiv.org/pdf/2604.07105](https://arxiv.org/pdf/2604.07105)**

> **作者:** Zhijun Li; Yongxin Su; Di Yang; Jichao Wang; Zheyuan Xing; Qian Wang; Maoqing Yao
>
> **摘要:** We present Genie Sim PanoRecon, a feed-forward Gaussian-splatting pipeline that delivers high-fidelity, low-cost 3D scenes for robotic manipulation simulation. The panorama input is decomposed into six non-overlapping cube-map faces, processed in parallel, and seamlessly reassembled. To guarantee geometric consistency across views, we devise a depth-aware fusion strategy coupled with a training-free depth-injection module that steers the monocular feed-forward network to generate coherent 3D Gaussians. The whole system reconstructs photo-realistic scenes in seconds and has been integrated into Genie Sim - a LLM-driven simulation platform for embodied synthetic data generation and evaluation - to provide scalable backgrounds for manipulation tasks. For code details, please refer to: this https URL.
>
---
#### [new 004] An RTK-SLAM Dataset for Absolute Accuracy Evaluation in GNSS-Degraded Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决RTK-SLAM系统在GNSS受限环境下绝对精度评估问题。通过构建独立于GNSS的基准数据集，揭示SE(3)对齐带来的误差低估现象。**

- **链接: [https://arxiv.org/pdf/2604.07151](https://arxiv.org/pdf/2604.07151)**

> **作者:** Wei Zhang; Vincent Ress; David Skuddis; Uwe Soergel; Norbert Haala
>
> **备注:** Accepted by ISPRS congress 2026
>
> **摘要:** RTK-SLAM systems integrate simultaneous localization and mapping (SLAM) with real-time kinematic (RTK) GNSS positioning, promising both relative consistency and globally referenced coordinates for efficient georeferenced surveying. A critical and underappreciated issue is that the standard evaluation metric, Absolute Trajectory Error (ATE), first fits an optimal rigid-body transformation between the estimated trajectory and reference before computing errors. This so-called SE(3) alignment absorbs global drift and systematic errors, making trajectories appear more accurate than they are in practice, and is unsuitable for evaluating the global accuracy of RTK-SLAM. We present a geodetically referenced dataset and evaluation methodology that expose this gap. A key design principle is that the RTK receiver is used solely as a system input, while ground truth is established independently via a geodetic total station. This separation is absent from all existing datasets, where GNSS typically serves as (part of) the ground truth. The dataset is collected with a handheld RTK-SLAM device, comprising two scenes. We evaluate LiDAR-inertial, visual-inertial, and LiDAR-visual-inertial RTK-SLAM systems alongside standalone RTK, reporting direct global accuracy and SE(3)-aligned relative accuracy to make the gap explicit. Results show that SE(3) alignment can underestimate absolute positioning error by up to 76\%. RTK-SLAM achieves centimeter-level absolute accuracy in open-sky conditions and maintains decimeter-level global accuracy indoors, where standalone RTK degrades to tens of meters. The dataset, calibration files, and evaluation scripts are publicly available at this https URL.
>
---
#### [new 005] Uncertainty Estimation for Deep Reconstruction in Actuatic Disaster Scenarios with Autonomous Vehicles
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于环境场重建任务，解决自主水下车辆在稀疏观测下的不确定性估计问题。比较了多种方法在不同传感器模型下的表现，发现证据深度学习效果最佳。**

- **链接: [https://arxiv.org/pdf/2604.06387](https://arxiv.org/pdf/2604.06387)**

> **作者:** Samuel Yanes Luis; Alejandro Casado Pérez; Alejandro Mendoza Barrionuevo; Dame Seck Diop; Sergio Toral Marín; Daniel Gutiérrez Reina
>
> **摘要:** Accurate reconstruction of environmental scalar fields from sparse onboard observations is essential for autonomous vehicles engaged in aquatic monitoring. Beyond point estimates, principled uncertainty quantification is critical for active sensing strategies such as Informative Path Planning, where epistemic uncertainty drives data collection decisions. This paper compares Gaussian Processes, Monte Carlo Dropout, Deep Ensembles, and Evidential Deep Learning for simultaneous scalar field reconstruction and uncertainty decomposition under three perceptual models representative of real sensor modalities. Results show that Evidential Deep Learning achieves the best reconstruction accuracy and uncertainty calibration across all sensor configurations at the lowest inference cost, while Gaussian Processes are fundamentally limited by their stationary kernel assumption and become intractable as observation density grows. These findings support Evidential Deep Learning as the preferred method for uncertainty-aware field reconstruction in real-time autonomous vehicle deployments.
>
---
#### [new 006] TAMEn: Tactile-Aware Manipulation Engine for Closed-Loop Data Collection in Contact-Rich Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决接触丰富双臂操作的数据收集问题。提出TAMEn系统，融合触觉与视觉，提升数据质量和策略性能。**

- **链接: [https://arxiv.org/pdf/2604.07335](https://arxiv.org/pdf/2604.07335)**

> **作者:** Longyan Wu; Jieji Ren; Chenghang Jiang; Junxi Zhou; Shijia Peng; Ran Huang; Guoying Gu; Li Chen; Hongyang Li
>
> **摘要:** Handheld paradigms offer an efficient and intuitive way for collecting large-scale demonstration of robot manipulation. However, achieving contact-rich bimanual manipulation through these methods remains a pivotal challenge, which is substantially hindered by hardware adaptability and data efficacy. Prior hardware designs remain gripper-specific and often face a trade-off between tracking precision and portability. Furthermore, the lack of online feasibility checking during demonstration leads to poor replayability. More importantly, existing handheld setups struggle to collect interactive recovery data during robot execution, lacking the authentic tactile information necessary for robust policy refinement. To bridge these gaps, we present TAMEn, a tactile-aware manipulation engine for closed-loop data collection in contact-rich tasks. Our system features a cross-morphology wearable interface that enables rapid adaptation across heterogeneous grippers. To balance data quality and environmental diversity, we implement a dual-modal acquisition pipeline: a precision mode leveraging motion capture for high-fidelity demonstrations, and a portable mode utilizing VR-based tracking for in-the-wild acquisition and tactile-visualized recovery teleoperation. Building on this hardware, we unify large-scale tactile pretraining, task-specific bimanual demonstrations, and human-in-the-loop recovery data into a pyramid-structured data regime, enabling closed-loop policy refinement. Experiments show that our feasibility-aware pipeline significantly improves demonstration replayability, and that the proposed visuo-tactile learning framework increases task success rates from 34% to 75% across diverse bimanual manipulation tasks. We further open-source the hardware and dataset to facilitate reproducibility and support research in visuo-tactile manipulation.
>
---
#### [new 007] Learning-Based Strategy for Composite Robot Assembly Skill Adaptation
- **分类: cs.RO**

- **简介: 该论文属于机器人装配任务，旨在解决工业机器人在复杂接触环境下技能适应问题。通过残差强化学习实现装配技能的模块化与高效适应。**

- **链接: [https://arxiv.org/pdf/2604.06949](https://arxiv.org/pdf/2604.06949)**

> **作者:** Khalil Abuibaid; Aleksandr Sidorenko; Achim Wagner; Martin Ruskowski
>
> **备注:** Accepted at RAAD 2026 (Springer). 6 pages, 4 figures
>
> **摘要:** Contact-rich robotic skills remain challenging for industrial robots due to tight geometric tolerances, frictional variability, and uncertain contact dynamics, particularly when using position-controlled manipulators. This paper presents a reusable and encapsulated skill-based strategy for peg-in-hole assembly, in which adaptation is achieved through Residual Reinforcement Learning (RRL). The assembly process is represented using composite skills with explicit pre-, post-, and invariant conditions, enabling modularity, reusability, and well-defined execution semantics across task variations. Safety and sample efficiency are promoted through RRL by restricting adaptation to residual refinements within each skill during contact-rich interactions, while the overall skill structure and execution flow remain invariant. The proposed approach is evaluated in MuJoCo simulation on a UR5e robot equipped with a Robotiq gripper and trained using SAC and JAX. Results demonstrate that the proposed formulation enables robust execution of assembly skills, highlighting its suitability for industrial automation.
>
---
#### [new 008] Telecom World Models: Unifying Digital Twins, Foundation Models, and Predictive Planning for 6G
- **分类: cs.RO; eess.SP; eess.SY**

- **简介: 该论文提出Telecom World Model（TWM），解决6G网络中动态建模与决策问题，整合数字孪生、基础模型和预测规划，实现不确定性感知的系统建模与优化。**

- **链接: [https://arxiv.org/pdf/2604.06882](https://arxiv.org/pdf/2604.06882)**

> **作者:** Hang Zou; Yuzhi Yang; Lina Bariah; Yu Tian; Yuhuan Lu; Bohao Wang; Anis Bara; Brahim Mefgouda; Hao Liu; Yiwei Tao; Sergy Petrov; Salma Cheour; Nassim Sehad; Sumudu Samarakoon; Chongwen Huang; Samson Lasaulce; Mehdi Bennis; Mérouane Debbah
>
> **摘要:** The integration of machine learning tools into telecom networks, has led to two prevailing paradigms, namely, language-based systems, such as Large Language Models (LLMs), and physics-based systems, such as Digital Twins (DTs). While LLM-based approaches enable flexible interaction and automation, they lack explicit representations of network dynamics. DTs, in contrast, offer a high-fidelity network simulation, but remain scenario-specific and are not designed for learning or decision-making under uncertainty. This gap becomes critical for 6G systems, where decisions must take into account the evolving network states, uncertainty, and the cascading effects of control actions across multiple layers. In this article, we introduce the {Telecom World Model}~(TWM) concept, an architecture for learned, action-conditioned, uncertainty-aware modeling of telecom system dynamics. We decompose the problem into two interacting worlds, a controllable system world consisting of operator-configurable settings and an external world that captures propagation, mobility, traffic, and failures. We propose a three-layer architecture, comprising a field world model for spatial environment prediction, a control/dynamics world model for action-conditioned Key Performance Indicator (KPI) trajectory prediction, and a telecom foundation model layer for intent translation and orchestration. We showcase a comparative analysis between existing paradigms, which demonstrates that TWM jointly provides telecom state grounding, fast action-conditioned roll-outs, calibrated uncertainty, multi-timescale dynamics, model-based planning, and LLM-integrated guardrails. Furthermore, we present a proof-of-concept on network slicing to validate the proposed architecture, showing that the full three-layer pipeline outperforms single-world baselines and accurately predicts KPI trajectories.
>
---
#### [new 009] Train-Small Deploy-Large: Leveraging Diffusion-Based Multi-Robot Planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多机器人路径规划任务，解决机器人数量变化时的泛化问题。通过扩散模型实现小规模训练、大规模部署，提升方法的适应性与效率。**

- **链接: [https://arxiv.org/pdf/2604.06598](https://arxiv.org/pdf/2604.06598)**

> **作者:** Siddharth Singh; Soumee Guha; Qing Chang; Scott Acton
>
> **摘要:** Learning based multi-robot path planning methods struggle to scale or generalize to changes, particularly variations in the number of robots during deployment. Most existing methods are trained on a fixed number of robots and may tolerate a reduced number during testing, but typically fail when the number increases. Additionally, training such methods for a larger number of agents can be both time consuming and computationally expensive. However, analytical methods can struggle to scale computationally or handle dynamic changes in the environment. In this work, we propose to leverage a diffusion model based planner capable of handling dynamically varying number of agents. Our approach is trained on a limited number of agents and generalizes effectively to larger numbers of agents during deployment. Results show that integrating a single shared diffusion model based planner with dedicated inter-agent attention computation and temporal convolution enables a train small deploy-large paradigm with good accuracy. We validate our method across multiple scenarios and compare the performance with existing multi-agent reinforcement learning techniques and heuristic control based methods.
>
---
#### [new 010] CADENCE: Context-Adaptive Depth Estimation for Navigation and Computational Efficiency
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出CADENCE系统，解决自动驾驶中深度估计的计算效率问题。通过动态调整网络复杂度，在保证精度的前提下降低能耗和延迟。属于计算机视觉中的深度估计任务。**

- **链接: [https://arxiv.org/pdf/2604.07286](https://arxiv.org/pdf/2604.07286)**

> **作者:** Timothy K Johnsen; Marco Levorato
>
> **备注:** 7 pages, 7 figures, Accepted for publication at IEEE World AI IoT Congress (AIIoT) 2026
>
> **摘要:** Autonomous vehicles deployed in remote environments typically rely on embedded processors, compact batteries, and lightweight sensors. These hardware limitations conflict with the need to derive robust representations of the environment, which often requires executing computationally intensive deep neural networks for perception. To address this challenge, we present CADENCE, an adaptive system that dynamically scales the computational complexity of a slimmable monocular depth estimation network in response to navigation needs and environmental context. By closing the loop between perception fidelity and actuation requirements, CADENCE ensures high-precision computing is only used when mission-critical. We conduct evaluations on our released open-source testbed that integrates Microsoft AirSim with an NVIDIA Jetson Orin Nano. As compared to a state-of-the-art static approach, CADENCE decreases sensor acquisitions, power consumption, and inference latency by 9.67%, 16.1%, and 74.8%, respectively. The results demonstrate an overall reduction in energy expenditure by 75.0%, along with an increase in navigation accuracy by 7.43%.
>
---
#### [new 011] Robust Quadruped Locomotion via Evolutionary Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文研究四足机器人运动的鲁棒性问题，通过进化强化学习提升策略在不同环境下的适应能力。对比多种方法，发现进化变体在地形变化时表现更优。**

- **链接: [https://arxiv.org/pdf/2604.07224](https://arxiv.org/pdf/2604.07224)**

> **作者:** Brian McAteer; Karl Mason
>
> **备注:** 10 pages, 3 figures. Accepted to the 11th International Conference on Control and Robotics Engineering (ICCRE 2026), Kyoto, Japan, May, 2026, this http URL
>
> **摘要:** Deep reinforcement learning has recently achieved strong results in quadrupedal locomotion, yet policies trained in simulation often fail to transfer when the environment changes. Evolutionary reinforcement learning aims to address this limitation by combining gradient-based policy optimisation with population-driven exploration. This work evaluates four methods on a simulated walking task: DDPG, TD3, and two Cross-Entropy-based variants CEM-DDPG and CEM-TD3. All agents are trained on flat terrain and later tested both on this domain and on a rough terrain not encountered during training. TD3 performs best among the standard deep RL baselines on flat ground with a mean reward of 5927.26, while CEM-TD3 achieves the highest rewards overall during training and evaluation 17611.41. Under the rough-terrain transfer test, performance of the deep RL methods drops sharply. DDPG achieves -1016.32 and TD3 achieves -99.73, whereas the evolutionary variants retain much of their capability. CEM-TD3 records the strongest transfer performance with a mean reward of 19574.33. These findings suggest that incorporating evolutionary search can reduce overfitting and improve policy robustness in locomotion tasks, particularly when deployment conditions differ from those seen during training.
>
---
#### [new 012] Sustainable Transfer Learning for Adaptive Robot Skills
- **分类: cs.RO**

- **简介: 该论文属于机器人技能迁移学习任务，旨在解决机器人技能复用问题。通过策略迁移与微调提升样本效率，减少重新训练需求，实现可持续学习。**

- **链接: [https://arxiv.org/pdf/2604.06943](https://arxiv.org/pdf/2604.06943)**

> **作者:** Khalil Abuibaid; Vinit Hegiste; Nigora Gafur; Achim Wagner; Martin Ruskowski
>
> **备注:** Published in RAAD 2025 (Springer). 7 pages, 5 figures
>
> **摘要:** Learning robot skills from scratch is often time-consuming, while reusing data promotes sustainability and improves sample efficiency. This study investigates policy transfer across different robotic platforms, focusing on peg-in-hole task using reinforcement learning (RL). Policy training is carried out on two different robots. Their policies are transferred and evaluated for zero-shot, fine-tuning, and training from scratch. Results indicate that zero-shot transfer leads to lower success rates and relatively longer task execution times, while fine-tuning significantly improves performance with fewer training time-steps. These findings highlight that policy transfer with adaptation techniques improves sample efficiency and generalization, reducing the need for extensive retraining and supporting sustainable robotic learning.
>
---
#### [new 013] BiDexGrasp: Coordinated Bimanual Dexterous Grasps across Object Geometries and Sizes
- **分类: cs.RO**

- **简介: 该论文提出BiDexGrasp，解决双臂灵巧抓取任务中的数据不足与生成难题。构建了大规模数据集并设计生成框架，实现对不同物体的协调抓取。**

- **链接: [https://arxiv.org/pdf/2604.06589](https://arxiv.org/pdf/2604.06589)**

> **作者:** Mu Lin; Yi-Lin Wei; Jiaxuan Chen; Yuhao Lin; Shuoyu Chen; Jiangran Lyu; Jiayi Chen; Yansong Tang; He Wang; Wei-Shi Zheng
>
> **备注:** Project Page: this https URL
>
> **摘要:** Bimanual dexterous grasping is a fundamental and promising area in robotics, yet its progress is constrained by the lack of comprehensive datasets and powerful generation models. In this work, we propose BiDexGrasp, consists of a large-scale bimanual dexterous grasp dataset and a novel generation model. For dataset, we propose a novel bimanual grasp synthesis pipeline to efficiently annotate physically feasible data for dataset construction. This pipeline addresses the challenges of high-dimensional bimanual grasping through a two-stage synthesis strategy of efficient region-based grasp initialization and decoupled force-closure grasp optimization. Powered by this pipeline, we construct a large-scale bimanual dexterous grasp dataset, comprising 6351 diverse objects with sizes ranging from 30 to 80 cm, along with 9.7 million annotated grasp data. Based on this dataset, we further introduce a bimanual-coordinated and geometry-size-adaptive dexterous grasping generation framework. The framework lies in two key designs: a bimanual coordination module and a geometry-size-adaptive grasp generation strategy to generate coordinated and high-quality grasps on unseen objects. Extensive experiments conducted in both simulation and real world demonstrate the superior performance of our proposed data synthesis pipeline and learned generative framework.
>
---
#### [new 014] Exploring the proprioceptive potential of joint receptors using a biomimetic robotic joint
- **分类: cs.RO; q-bio.NC**

- **简介: 该论文属于神经科学与机器人学交叉研究，旨在探讨关节感受器在本体感觉中的作用。通过仿生机器人关节模拟Type I关节感受器，验证其感知精度，揭示其在本体感觉中的潜在贡献。**

- **链接: [https://arxiv.org/pdf/2604.07038](https://arxiv.org/pdf/2604.07038)**

> **作者:** Akihiro Miki; Shun Hasegawa; Sota Yuzaki; Yuta Sahara; Yoshimoto Ribayashi; Kento Kawaharazuka; Kei Okada
>
> **备注:** 26 pages including supplementary materials (17 pages main text), 6 main figures and 7 supplementary figures. Published in Scientific Reports
>
> **摘要:** In neuroscience, joint receptors have traditionally been viewed as limit detectors, providing positional information only at extreme joint angles, while muscle spindles are considered the primary sensors of joint angle position. However, joint receptors are widely distributed throughout the joint capsule, and their full role in proprioception remains unclear. In this study, we specifically focused on mimicking Type I joint receptors, which respond to slow and sustained movements, and quantified their proprioceptive potential using a biomimetic joint developed with robotics technology. Results showed that Type I-like joint receptors alone enabled proprioceptive sensing with an average error of less than 2 degrees in both bending and twisting motions. These findings suggest that joint receptors may play a greater role in proprioception than previously recognized and that the relative contributions of muscle spindles and joint receptors are differentially weighted within neural networks during development and evolution. Furthermore, this work may prompt new discussions on the differential proprioceptive deficits observed between the elbows and knees in patients with hereditary sensory and autonomic neuropathy type III. Together, these findings highlight the potential of biomimetics-based robotic approaches for advancing interdisciplinary research bridging neuroscience, medicine, and robotics.
>
---
#### [new 015] KITE: Keyframe-Indexed Tokenized Evidence for VLM-Based Robot Failure Analysis
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出KITE，用于机器人故障分析的视觉-语言模型前端。解决长视频处理与故障解释问题，通过关键帧和布局信息生成可解释的提示。**

- **链接: [https://arxiv.org/pdf/2604.07034](https://arxiv.org/pdf/2604.07034)**

> **作者:** Mehdi Hosseinzadeh; King Hang Wong; Feras Dayoub
>
> **备注:** ICRA 2026; Project page: this https URL
>
> **摘要:** We present KITE, a training-free, keyframe-anchored, layout-grounded front-end that converts long robot-execution videos into compact, interpretable tokenized evidence for vision-language models (VLMs). KITE distills each trajectory into a small set of motion-salient keyframes with open-vocabulary detections and pairs each keyframe with a schematic bird's-eye-view (BEV) representation that encodes relative object layout, axes, timestamps, and detection confidence. These visual cues are serialized with robot-profile and scene-context tokens into a unified prompt, allowing the same front-end to support failure detection, identification, localization, explanation, and correction with an off-the-shelf VLM. On the RoboFAC benchmark, KITE with Qwen2.5-VL substantially improves over vanilla Qwen2.5-VL in the training-free setting, with especially large gains on simulation failure detection, identification, and localization, while remaining competitive with a RoboFAC-tuned baseline. A small QLoRA fine-tune further improves explanation and correction quality. We also report qualitative results on real dual-arm robots, demonstrating the practical applicability of KITE as a structured and interpretable front-end for robot failure analysis. Code and models are released on our project page: this https URL
>
---
#### [new 016] Differentiable Environment-Trajectory Co-Optimization for Safe Multi-Agent Navigation
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多智能体导航任务，旨在解决环境与路径协同优化问题。通过联合优化环境配置和代理轨迹，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2604.06972](https://arxiv.org/pdf/2604.06972)**

> **作者:** Zhan Gao; Gabriele Fadini; Stelian Coros; Amanda Prorok
>
> **摘要:** The environment plays a critical role in multi-agent navigation by imposing spatial constraints, rules, and limitations that agents must navigate around. Traditional approaches treat the environment as fixed, without exploring its impact on agents' performance. This work considers environment configurations as decision variables, alongside agent actions, to jointly achieve safe navigation. We formulate a bi-level problem, where the lower-level sub-problem optimizes agent trajectories that minimize navigation cost and the upper-level sub-problem optimizes environment configurations that maximize navigation safety. We develop a differentiable optimization method that iteratively solves the lower-level sub-problem with interior point methods and the upper-level sub-problem with gradient ascent. A key challenge lies in analytically coupling these two levels. We address this by leveraging KKT conditions and the Implicit Function Theorem to compute gradients of agent trajectories w.r.t. environment parameters, enabling differentiation throughout the bi-level structure. Moreover, we propose a novel metric that quantifies navigation safety as a criterion for the upper-level environment optimization, and prove its validity through measure theory. Our experiments validate the effectiveness of the proposed framework in a variety of safety-critical navigation scenarios, inspired from warehouse logistics to urban transportation. The results demonstrate that optimized environments provide navigation guidance, improving both agents' safety and efficiency.
>
---
#### [new 017] Flow Motion Policy: Manipulator Motion Planning with Flow Matching Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动规划任务，旨在解决传统方法生成路径单一、无法优化的问题。提出Flow Motion Policy，通过流匹配模型生成多路径并选择最优解，提升规划成功率与效率。**

- **链接: [https://arxiv.org/pdf/2604.07084](https://arxiv.org/pdf/2604.07084)**

> **作者:** Davood Soleymanzadeh; Xiao Liang; Minghui Zheng
>
> **摘要:** Open-loop end-to-end neural motion planners have recently been proposed to improve motion planning for robotic manipulators. These methods enable planning directly from sensor observations without relying on a privileged collision checker during planning. However, many existing methods generate only a single path for a given workspace across different runs, and do not leverage their open-loop structure for inference-time optimization. To address this limitation, we introduce Flow Motion Policy, an open-loop, end-to-end neural motion planner for robotic manipulators that leverages the stochastic generative formulation of flow matching methods to capture the inherent multi-modality of planning datasets. By modeling a distribution over feasible paths, Flow Motion Policy enables efficient inference-time best-of-$N$ sampling. The method generates multiple end-to-end candidate paths, evaluates their collision status after planning, and executes the first collision-free solution. We benchmark the Flow Motion Policy against representative sampling-based and neural motion planning methods. Evaluation results demonstrate that Flow Motion Policy improves planning success and efficiency, highlighting the effectiveness of stochastic generative policies for end-to-end motion planning and inference-time optimization. Experimental evaluation videos are available via this \href{this https URL}{link}.
>
---
#### [new 018] Occlusion Handling by Pushing for Enhanced Fruit Detection
- **分类: cs.RO**

- **简介: 该论文属于农业机器人任务，旨在解决果实被遮挡的问题。通过深度学习和图像处理技术检测果实及遮挡枝条，并利用机械臂推移枝条以提高可见性。**

- **链接: [https://arxiv.org/pdf/2604.06341](https://arxiv.org/pdf/2604.06341)**

> **作者:** Ege Gursoy; Dana Kulić; Andrea Cherubini
>
> **摘要:** In agricultural robotics, effective observation and localization of fruits present challenges due to occlusions caused by other parts of the tree, such as branches and leaves. These occlusions can result in false fruit localization or impede the robot from picking the fruit. The objective of this work is to push away branches that block the fruit's view to increase their visibility. Our setup consists of an RGB-D camera and a robot arm. First, we detect the occluded fruit in the RGB image and estimate its occluded part via a deep learning generative model in the depth space. The direction to push to clear the occlusions is determined using classic image processing techniques. We then introduce a 3D extension of the 2D Hough transform to detect straight line segments in the point cloud. This extension helps detect tree branches and identify the one mainly responsible for the occlusion. Finally, we clear the occlusion by pushing the branch with the robot arm. Our method uses a combination of deep learning for fruit appearance estimation, classic image processing for push direction determination, and 3D Hough transform for branch detection. We validate our perception methods through real data under different lighting conditions and various types of fruits (i.e. apple, lemon, orange), achieving improved visibility and successful occlusion clearance. We demonstrate the practical application of our approach through a real robot branch pushing demonstration.
>
---
#### [new 019] Self-Discovered Intention-aware Transformer for Multi-modal Vehicle Trajectory Prediction
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于多模态车辆轨迹预测任务，旨在解决传统方法依赖特定图结构或显式意图标注的问题。提出一种基于Transformer的网络，分离空间模块与轨迹生成模块，提升预测性能。**

- **链接: [https://arxiv.org/pdf/2604.07126](https://arxiv.org/pdf/2604.07126)**

> **作者:** Diyi Liu; Zihan Niu; Tu Xu; Lishan Sun
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Predicting vehicle trajectories plays an important role in autonomous driving and ITS applications. Although multiple deep learning algorithms are devised to predict vehicle trajectories, their reliant on specific graph structure (e.g., Graph Neural Network) or explicit intention labeling limit their flexibilities. In this study, we propose a pure Transformer-based network with multiple modals considering their neighboring vehicles. Two separate tracks are employed. One track focuses on predicting the trajectories while the other focuses on predicting the likelihood of each intention considering neighboring vehicles. Study finds that the two track design can increase the performance by separating spatial module from the trajectory generating module. Also, we find the the model can learn an ordered group of trajectories by predicting residual offsets among K trajectories.
>
---
#### [new 020] RichMap: A Reachability Map Balancing Precision, Efficiency, and Flexibility for Rich Robot Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文提出RichMap，用于机器人操作任务的可达性地图表示，解决精度、效率与灵活性的平衡问题，通过优化网格结构提升性能。**

- **链接: [https://arxiv.org/pdf/2604.06778](https://arxiv.org/pdf/2604.06778)**

> **作者:** Yupu Lu; Yuxiang Ma; Jia Pan
>
> **备注:** Accepted by WAFR 2026
>
> **摘要:** This paper presents RichMap, a high-precision reachability map representation designed to balance efficiency and flexibility for versatile robot manipulation tasks. By refining the classic grid-based structure, we propose a streamlined approach that achieves performance close to compact map forms (e.g., RM4D) while maintaining structural flexibility. Our method utilizes theoretical capacity bounds on $\mathbb{S}^2$ (or $SO(3)$) to ensure rigorous coverage and employs an asynchronous pipeline for efficient construction. We validate the map against comprehensive metrics, pursuing high prediction accuracy ($>98\%$), low false positive rates ($1\sim2\%$), and fast large-batch query ($\sim$15 $\mu$s/query). We extend the framework applications to quantify robot workspace similarity via maximum mean discrepancy (MMD) metrics and demonstrate energy-based guidance for diffusion policy transfer, achieving up to $26\%$ improvement for cross-embodiment scenarios in the block pushing experiment.
>
---
#### [new 021] Robots that learn to evaluate models of collective behavior
- **分类: cs.RO**

- **简介: 该论文属于行为模型评估任务，旨在解决传统方法依赖静态数据的问题。通过机器人与真实鱼群的闭环互动，量化模型与实际行为的差异，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2604.07303](https://arxiv.org/pdf/2604.07303)**

> **作者:** Mathis Hocke; Andreas Gerken; David Bierbach; Jens Krause; Tim Landgraf
>
> **摘要:** Understanding and modeling animal behavior is essential for studying collective motion, decision-making, and bio-inspired robotics. Yet, evaluating the accuracy of behavioral models still often relies on offline comparisons to static trajectory statistics. Here we introduce a reinforcement-learning-based framework that uses a biomimetic robotic fish (RoboFish) to evaluate computational models of live fish behavior through closed-loop interaction. We trained policies in simulation using four distinct fish models-a simple constant-follow baseline, two rule-based models, and a biologically grounded convolutional neural network model-and transferred these policies to the real RoboFish setup, where they interacted with live fish. Policies were trained to guide a simulated fish to goal locations, enabling us to quantify how the response of real fish differs from the simulated fish's response. We evaluate the fish models by quantifying the sim-to-real gaps, defined as the Wasserstein distance between simulated and real distributions of behavioral metrics such as goal-reaching performance, inter-individual distances, wall interactions, and alignment. The neural network-based fish model exhibited the smallest gap across goal-reaching performance and most other metrics, indicating higher behavioral fidelity than conventional rule-based models under this benchmark. More importantly, this separation shows that the proposed evaluation can quantitatively distinguish candidate models under matched closed-loop conditions. Our work demonstrates how learning-based robotic experiments can uncover deficiencies in behavioral models and provides a general framework for evaluating animal behavior models through embodied interaction.
>
---
#### [new 022] Towards Multi-Object Nonprehensile Transportation via Shared Teleoperation: A Framework Based on Virtual Object Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于多目标非抓取运输任务，解决轨迹跟踪与托盘方向控制问题。提出共享遥操作框架，结合虚拟对象模型预测控制，提升稳定性和适应性。**

- **链接: [https://arxiv.org/pdf/2604.06932](https://arxiv.org/pdf/2604.06932)**

> **作者:** Xinyang Fan; Zhaoyang Chen; Shu Xin; Yi Ren; Zainan Jiang; Fenglei Ni; Hong Liu
>
> **摘要:** Multi-object nonprehensile transportation in teleoperation demands simultaneous trajectory tracking and tray orientation control. Existing methods often struggle with model dependency, uncertain parameters, and multi-object adaptability. We propose a shared teleoperation framework where humans and robots share positioning control, while the robot autonomously manages orientation to satisfy dynamic constraints. Key contributions include: 1) A theoretical dynamic constraint analysis utilizing a novel virtual object (VO)-based method to simplify constraints for trajectory planning. 2) An MPC-based trajectory smoothing algorithm that enforces real-time constraints and coordinates user tracking with orientation control. 3) Validations demonstrating stable manipulation of nine objects at accelerations up to 2.4 m/s2. Compared to the baseline, our approach reduces sliding distance by 72.45% and eliminates tip-overs (0% vs. 13.9%), proving robust adaptability in complex scenarios.
>
---
#### [new 023] Designing Privacy-Preserving Visual Perception for Robot Navigation Based on User Privacy Preferences
- **分类: cs.RO**

- **简介: 该论文属于隐私保护任务，解决机器人导航中用户隐私泄露问题。通过用户研究，提出基于用户偏好配置的视觉抽象与分辨率策略。**

- **链接: [https://arxiv.org/pdf/2604.06382](https://arxiv.org/pdf/2604.06382)**

> **作者:** Xuying Huang; Sicong Pan; Delphine Reinhardt; Maren Bennewitz
>
> **摘要:** Visual navigation is a fundamental capability of mobile service robots, yet the onboard cameras required for such navigation can capture privacy-sensitive information and raise user privacy concerns. Existing approaches to privacy-preserving navigation-oriented visual perception have largely been driven by technical considerations, with limited grounding in user privacy preferences. In this work, we propose a user-centered approach to designing privacy-preserving visual perception for robot navigation. To investigate how user privacy preferences can inform such design, we conducted two user studies. The results show that users prefer privacy-preserving visual abstractions and capture-time low-resolution preservation mechanisms: their preferred RGB resolution depends both on the desired privacy level and robot proximity during navigation. Based on these findings, we further derive a user-configurable distance-to-resolution privacy policy for privacy-preserving robot visual navigation.
>
---
#### [new 024] Exploiting Aggregate Programming in a Multi-Robot Service Prototype
- **分类: cs.DC; cs.MA; cs.RO**

- **简介: 该论文属于多机器人系统研究，旨在解决复杂环境下的协调问题。通过引入聚合编程方法，设计并验证了一个多机器人服务原型。**

- **链接: [https://arxiv.org/pdf/2604.06876](https://arxiv.org/pdf/2604.06876)**

> **作者:** Giorgio Audrito; Andrea Basso; Daniele Bortoluzzi; Ferruccio Damiani; Giordano Scarso; Gianluca Torta
>
> **备注:** In Proceedings PLACES 2026, arXiv:2604.05737
>
> **摘要:** Multi-robot systems are becoming increasingly relevant within diverse application domains, such as healthcare, exploration, and rescue missions. However, building such systems is still a significant challenge, since it adds the complexities of the physical nature of robots and their environments to those inherent in coordinating any distributed (multi-agent) system. Aggregate Programming (AP) has recently emerged as a promising approach to engineering resilient, distributed systems with proximity-based communication, and is notably supported by practical frameworks. In this paper we present a prototype of a multi-robot service system, which adopts AP for the design and implementation of its coordination software. The prototype has been validated both with simulations, and with tests in a University library.
>
---
#### [new 025] Infrastructure First: Enabling Embodied AI for Science in the Global South
- **分类: cs.CY; cs.RO**

- **简介: 论文探讨了在资源有限的全球南方推广具身AI科学（EAI4S）的基础设施需求，旨在解决实验能力不足问题。任务属于AI与科学实验融合，解决如何通过可靠基础设施实现可持续科研。**

- **链接: [https://arxiv.org/pdf/2604.06722](https://arxiv.org/pdf/2604.06722)**

> **作者:** Shaoshan Liu; Jie Tang; Marwa S. Hassan; Mohamed H. Sharkawy; Moustafa M. G. Fouda; Tiewei Shang; Zixin Wang
>
> **摘要:** Embodied AI for Science (EAI4S) brings intelligence into the laboratory by uniting perception, reasoning, and robotic action to autonomously run experiments in the physical world. For the Global South, this shift is not about adopting advanced automation for its own sake, but about overcoming a fundamental capacity constraint: too few hands to run too many experiments. By enabling continuous, reliable experimentation under limits of manpower, power, and connectivity, EAI4S turns automation from a luxury into essential scientific infrastructure. The main obstacle, however, is not algorithmic capability. It is infrastructure. Open-source AI and foundation models have narrowed the knowledge gap, but EAI4S depends on dependable edge compute, energy-efficient hardware, modular robotic systems, localized data pipelines, and open standards. Without these foundations, even the most capable models remain trapped in well-resourced laboratories. This article argues for an infrastructure-first approach to EAI4S and outlines the practical requirements for deploying embodied intelligence at scale, offering a concrete pathway for Global South institutions to translate AI advances into sustained scientific capacity and competitive research output.
>
---
#### [new 026] MoRight: Motion Control Done Right
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出MoRight，解决视频生成中的运动控制问题，实现对象与相机运动的解耦及因果关系建模，提升视频生成质量与交互性。**

- **链接: [https://arxiv.org/pdf/2604.07348](https://arxiv.org/pdf/2604.07348)**

> **作者:** Shaowei Liu; Xuanchi Ren; Tianchang Shen; Huan Ling; Saurabh Gupta; Shenlong Wang; Sanja Fidler; Jun Gao
>
> **备注:** Project Page: this https URL
>
> **摘要:** Generating motion-controlled videos--where user-specified actions drive physically plausible scene dynamics under freely chosen viewpoints--demands two capabilities: (1) disentangled motion control, allowing users to separately control the object motion and adjust camera viewpoint; and (2) motion causality, ensuring that user-driven actions trigger coherent reactions from other objects rather than merely displacing pixels. Existing methods fall short on both fronts: they entangle camera and object motion into a single tracking signal and treat motion as kinematic displacement without modeling causal relationships between object motion. We introduce MoRight, a unified framework that addresses both limitations through disentangled motion modeling. Object motion is specified in a canonical static-view and transferred to an arbitrary target camera viewpoint via temporal cross-view attention, enabling disentangled camera and object control. We further decompose motion into active (user-driven) and passive (consequence) components, training the model to learn motion causality from data. At inference, users can either supply active motion and MoRight predicts consequences (forward reasoning), or specify desired passive outcomes and MoRight recovers plausible driving actions (inverse reasoning), all while freely adjusting the camera viewpoint. Experiments on three benchmarks demonstrate state-of-the-art performance in generation quality, motion controllability, and interaction awareness.
>
---
#### [new 027] Logical Robots: Declarative Multi-Agent Programming in Logica
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文提出Logical Robots平台，使用Logica语言实现多智能体机器人行为的声明式编程，解决多机器人协同控制问题，整合低层反应与高层规划。**

- **链接: [https://arxiv.org/pdf/2604.06629](https://arxiv.org/pdf/2604.06629)**

> **作者:** Evgeny Skvortsov; Yilin Xia; Ojaswa Garg; Shawn Bowers; Bertram Ludäscher
>
> **备注:** International Conference on Autonomous Agents and Multiagent Systems (AAMAS), May 25-29, 2026. Paphos, Cyprus
>
> **摘要:** We present Logical Robots, an interactive multi-agent simulation platform where autonomous robot behavior is specified declaratively in the logic programming language Logica. Robot behavior is defined by logical predicates that map observations from simulated radar arrays and shared memory to desired motor outputs. This approach allows low-level reactive control and high-level planning to coexist within a single programming environment, providing a coherent framework for exploring multi-agent robot behavior.
>
---
#### [new 028] VGGT-SLAM++
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VGGT-SLAM++，属于视觉SLAM任务，解决大场景下定位与建图的精度和效率问题，通过融合VGGT和DEM实现高精度、低漂移的实时定位与地图构建。**

- **链接: [https://arxiv.org/pdf/2604.06830](https://arxiv.org/pdf/2604.06830)**

> **作者:** Avilasha Mandal; Rajesh Kumar; Sudarshan Sunil Harithas; Chetan Arora
>
> **备注:** 8 pages (main paper) + supplementary material. Accepted at CVPR 2026 Workshop (VOCVALC)
>
> **摘要:** We introduce VGGT-SLAM++, a complete visual SLAM system that leverages the geometry-rich outputs of the Visual Geometry Grounded Transformer (VGGT). The system comprises a visual odometry (front-end) fusing the VGGT feed-forward transformer and a Sim(3) solution, a Digital Elevation Map (DEM)-based graph construction module, and a back-end that jointly enable accurate large-scale mapping with bounded memory. While prior transformer-based SLAM pipelines such as VGGT-SLAM rely primarily on sparse loop closures or global Sim(3) manifold constraints - allowing short-horizon pose drift - VGGT-SLAM++ restores high-cadence local bundle adjustment (LBA) through a spatially corrective back-end. For each VGGT submap, we construct a dense planar-canonical DEM, partition it into patches, and compute their DINOv2 embeddings to integrate the submap into a covisibility graph. Spatial neighbors are retrieved using a Visual Place Recognition (VPR) module within the covisibility window, triggering frequent local optimization that stabilizes trajectories. Across standard SLAM benchmarks, VGGT-SLAM++ achieves state-of-the-art accuracy, substantially reducing short-term drift, accelerating graph convergence, and maintaining global consistency with compact DEM tiles and sublinear retrieval.
>
---
## 更新

#### [replaced 001] Apple: Toward General Active Perception via Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出APPLE框架，利用强化学习解决机器人主动感知问题，旨在提升感知的通用性和灵活性。**

- **链接: [https://arxiv.org/pdf/2505.06182](https://arxiv.org/pdf/2505.06182)**

> **作者:** Tim Schneider; Cristiana de Farias; Roberto Calandra; Liming Chen; Jan Peters
>
> **备注:** 27 pages; 21 figures; accepted at the Fourteenth International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** Active perception is a fundamental skill that enables us humans to deal with uncertainty in our inherently partially observable environment. For senses such as touch, where the information is sparse and local, active perception becomes crucial. In recent years, active perception has emerged as an important research domain in robotics. However, current methods are often bound to specific tasks or make strong assumptions, which limit their generality. To address this gap, this work introduces APPLE (Active Perception Policy Learning) - a novel framework that leverages reinforcement learning (RL) to address a range of different active perception problems. APPLE jointly trains a transformer-based perception module and decision-making policy with a unified optimization objective, learning how to actively gather information. By design, APPLE is not limited to a specific task and can, in principle, be applied to a wide range of active perception problems. We evaluate two variants of APPLE across different tasks, including tactile exploration problems from the Tactile MNIST benchmark. Experiments demonstrate the efficacy of APPLE, achieving high accuracies on both regression and classification tasks. These findings underscore the potential of APPLE as a versatile and general framework for advancing active perception in robotics. Project page: this https URL
>
---
#### [replaced 002] Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming
- **分类: cs.RO**

- **简介: 该论文属于多机器人任务分配与规划任务，解决复杂环境中异构机器人的协调问题。提出OATH框架，结合障碍物感知策略和集群拍卖机制，提升任务分配质量和适应性。**

- **链接: [https://arxiv.org/pdf/2510.14063](https://arxiv.org/pdf/2510.14063)**

> **作者:** Nan Li; Jiming Ren; Haris Miller; Samuel Coogan; Karen M. Feigh; Ye Zhao
>
> **备注:** 24 pages, 19 figures, 5 tables
>
> **摘要:** Multi-Agent Task Assignment and Planning (MATP) has attracted growing attention but remains challenging in terms of scalability, spatial reasoning, and adaptability in obstacle-rich environments. To address these challenges, we propose OATH - Adaptive Obstacle-Aware Task Assignment and Planning for Heterogeneous Robot Teaming - which advances MATP by introducing a novel obstacle-aware strategy for task assignment. First, we develop an adaptive Halton sequence map, the first known application of Halton sampling with obstacle-aware adaptation in MATP, which adjusts sampling density based on obstacle distribution. Second, we propose a cluster-auction-selection framework that integrates obstacle-aware clustering with weighted auctions and intra-cluster task selection. These mechanisms jointly enable effective coordination among heterogeneous robots while maintaining scalability and suboptimal allocation performance. In addition, our framework leverages an LLM to interpret human instructions and directly guide the planner in real time. We validate OATH in both NVIDIA Isaac Sim and real-world hardware experiments using TurtleBot platforms, demonstrating substantial improvements in task assignment quality, scalability, adaptability to dynamic changes, and overall execution performance compared to state-of-the-art MATP baselines. A project website is available at this https URL.
>
---
#### [replaced 003] Before We Trust Them: Decision-Making Failures in Navigation of Foundation Models
- **分类: cs.AI; cs.RO**

- **简介: 论文研究导航决策问题，指出基础模型在导航任务中存在决策失败。通过多项测试评估模型表现，揭示其结构缺陷和安全问题，强调需细致评估才能信任模型。**

- **链接: [https://arxiv.org/pdf/2601.05529](https://arxiv.org/pdf/2601.05529)**

> **作者:** Jua Han; Jaeyoon Seo; Jungbin Min; Sieun Choi; Huichan Seo; Jihie Kim; Jean Oh
>
> **备注:** Corrected author order in metadata; manuscript changed
>
> **摘要:** High success rates on navigation-related tasks do not necessarily translate into reliable decision making by foundation models. To examine this gap, we evaluate current models on six diagnostic tasks spanning three settings: reasoning under complete spatial information, reasoning under incomplete spatial information, and reasoning under safety-relevant information. Our results show that the current metrics may not capture critical limitations of the models and indicate good performance, underscoring the need for failure-focused analysis to understand model limitations and guide future progress. In a path-planning setting with unknown cells, GPT-5 achieved a high success rate of 93%; Yet, the failed cases exhibit fundamental limitations of the models, e.g., the lack of structural spatial understanding essential for navigation. We also find that newer models are not always more reliable than their predecessors on this end. In reasoning under safety-relevant information, Gemini-2.5 Flash achieved only 67% on the challenging emergency-evacuation task, underperforming Gemini-2.0 Flash, which reached 100% under the same condition. Across all evaluations, models exhibited structural collapse, hallucinated reasoning, constraint violations, and unsafe decisions. These findings show that foundation models still exhibit substantial failures in navigation-related decision making and require fine-grained evaluation before they can be trusted.
>
---
#### [replaced 004] Model Predictive Control via Probabilistic Inference: A Tutorial and Survey
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制领域，探讨基于概率推断的模型预测控制（PI-MPC），解决最优控制问题，通过变分推断生成动作，并综述相关研究。**

- **链接: [https://arxiv.org/pdf/2511.08019](https://arxiv.org/pdf/2511.08019)**

> **作者:** Kohei Honda
>
> **备注:** 41 pages, 7 figures
>
> **摘要:** This paper presents a tutorial and survey on Probabilistic Inference-based Model Predictive Control (PI-MPC). PI-MPC reformulates finite-horizon optimal control as inference over an optimal control distribution expressed as a Boltzmann distribution weighted by a control prior, and generates actions through variational inference. In the tutorial part, we derive this formulation and explain action generation via variational inference, highlighting Model Predictive Path Integral (MPPI) control as a representative algorithm with a closed-form sampling update. In the survey part, we organize existing PI-MPC research around key design dimensions, including prior design, multi-modality, constraint handling, scalability, hardware acceleration, and theoretical analysis. This paper provides a unified conceptual perspective on PI-MPC and a practical entry point for researchers and practitioners in robotics and other control applications.
>
---
#### [replaced 005] A Dynamic Toolkit for Transmission Characteristics of Precision Reducers with Explicit Contact Geometry
- **分类: cs.RO**

- **简介: 该论文属于机器人动力学分析任务，旨在解决精密减速器传动特性建模问题。提出一种动态工具包，结合接触理论与数值方法，提升建模精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.02387](https://arxiv.org/pdf/2604.02387)**

> **作者:** Jiacheng Miao; Chao Liu; Qiliang Wang; Yunhui Guan; Weidong He
>
> **备注:** 21 pages, 8 figures
>
> **摘要:** Precision reducers are critical components in robotic systems, directly affecting the motion accuracy and dynamic performance of humanoid robots, quadruped robots, collaborative robots, industrial robots, and SCARA robots. This paper presents a dynamic toolkit for analyzing the transmission characteristics of precision reducers with explicit contact geometry. A unified framework is proposed to address the challenges in modeling accurate contact behaviors, evaluating gear stiffness, and predicting system vibrations. By integrating advanced contact theories and numerical solving methods, the proposed toolkit offers higher precision and computational efficiency compared to traditional dynamics software. The toolkit is designed with a modular, scriptable architecture that supports rapid reconfiguration across diverse reducer topologies. Numerical validation against published benchmarks confirms the accuracy of the proposed approach.
>
---
#### [replaced 006] Splatblox: Traversability-Aware Gaussian Splatting for Outdoor Robot Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Splatblox，用于户外机器人导航任务，解决复杂地形中障碍物与可通行区域的识别问题。通过融合RGB图像和LiDAR数据，构建具有语义信息的ESDF，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2511.18525](https://arxiv.org/pdf/2511.18525)**

> **作者:** Samarth Chopra; Jing Liang; Gershom Seneviratne; Yonghan Lee; Jaehoon Choi; Jianyu An; Stephen Cheng; Dinesh Manocha
>
> **摘要:** We present Splatblox, a real-time system for autonomous navigation in outdoor environments with dense vegetation, irregular obstacles, and complex terrain. Our method fuses segmented RGB images and LiDAR point clouds using Gaussian Splatting to construct a traversability-aware Euclidean Signed Distance Field (ESDF) that jointly encodes geometry and semantics. Updated online, this field enables semantic reasoning to distinguish traversable vegetation (e.g., tall grass) from rigid obstacles (e.g., trees), while LiDAR ensures 360-degree geometric coverage for extended planning horizons. We validate Splatblox on a quadruped robot and demonstrate transfer to a wheeled platform. In field trials across vegetation-rich scenarios, it outperforms state-of-the-art methods with over 50% higher success rate, 40% fewer freezing incidents, 5% shorter paths, and up to 13% faster time to goal, while supporting long-range missions up to 100 meters. Experiment videos and more details can be found on our project page: this https URL
>
---
#### [replaced 007] Can VLMs Unlock Semantic Anomaly Detection? A Framework for Structured Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的异常检测任务，旨在解决罕见语义异常检测难题。提出SAVANT框架，通过语义一致性验证提升VLM的检测性能，并实现高效数据标注与模型优化。**

- **链接: [https://arxiv.org/pdf/2510.18034](https://arxiv.org/pdf/2510.18034)**

> **作者:** Roberto Brusnicki; David Pop; Yuan Gao; Mattia Piccinini; Johannes Betz
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Autonomous driving systems remain critically vulnerable to the long-tail of rare, out-of-distribution semantic anomalies. While VLMs have emerged as promising tools for perception, their application in anomaly detection remains largely restricted to prompting proprietary models - limiting reliability, reproducibility, and deployment feasibility. To address this gap, we introduce SAVANT (Semantic Anomaly Verification/Analysis Toolkit), a novel model-agnostic reasoning framework that reformulates anomaly detection as a layered semantic consistency verification. By applying SAVANT's two-phase pipeline - structured scene description extraction and multi-modal evaluation - existing VLMs achieve significantly higher scores in detecting anomalous driving scenarios from input images. Our approach replaces ad hoc prompting with semantic-aware reasoning, transforming VLM-based detection into a principled decomposition across four semantic domains. We show that across a balanced set of real-world driving scenarios, applying SAVANT improves VLM's absolute recall by approximately 18.5% compared to prompting baselines. Moreover, this gain enables reliable large-scale annotation: leveraging the best proprietary model within our framework, we automatically labeled around 10,000 real-world images with high confidence. We use the resulting high-quality dataset to fine-tune a 7B open-source model (Qwen2.5-VL) to perform single-shot anomaly detection, achieving 90.8% recall and 93.8% accuracy - surpassing all models evaluated while enabling local deployment at near-zero cost. By coupling structured semantic reasoning with scalable data curation, SAVANT provides a practical solution to data scarcity in semantic anomaly detection for autonomous systems. Supplementary material: this https URL
>
---
#### [replaced 008] A1: A Fully Transparent Open-Source, Adaptive and Efficient Truncated Vision-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文提出A1框架，解决VLA模型部署成本高的问题，通过优化推理流程实现高效低耗的机器人操作。**

- **链接: [https://arxiv.org/pdf/2604.05672](https://arxiv.org/pdf/2604.05672)**

> **作者:** Kaidong Zhang; Jian Zhang; Rongtao Xu; Yu Sun; Shuoshuo Xue; Youpeng Wen; Xiaoyu Guo; Minghao Guo; Weijia Liufu; Liu Zihou; Kangyi Ji; Yangsong Zhang; Jiarun Zhu; Jingzhi Liu; Zihang Li; Ruiyi Chen; Meng Cao; Jingming Zhang; Shen Zhao; Xiaojun Chang; Feng Zheng; Ivan Laptev; Xiaodan Liang
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a powerful paradigm for open-world robot manipulation, but their practical deployment is often constrained by cost: billion-scale VLM backbones and iterative diffusion/flow-based action heads incur high latency and compute, making real-time control expensive on commodity hardware. We present A1, a fully open-source and transparent VLA framework designed for low-cost, high-throughput inference without sacrificing manipulation success; Our approach leverages pretrained VLMs that provide implicit affordance priors for action generation. We release the full training stack (training code, data/data-processing pipeline, intermediate checkpoints, and evaluation scripts) to enable end-to-end reproducibility. Beyond optimizing the VLM alone, A1 targets the full inference pipeline by introducing a budget-aware adaptive inference scheme that jointly accelerates the backbone and the action head. Specifically, we monitor action consistency across intermediate VLM layers to trigger early termination, and propose Inter-Layer Truncated Flow Matching that warm-starts denoising across layers, enabling accurate actions with substantially fewer effective denoising iterations. Across simulation benchmarks (LIBERO, VLABench) and real robots (Franka, AgiBot), A1 achieves state-of-the-art success rates while significantly reducing inference cost (e.g., up to 72% lower per-episode latency for flow-matching inference and up to 76.6% backbone computation reduction with minor performance degradation). On RoboChallenge, A1 achieves an average success rate of 29.00%, outperforming baselines including pi0(28.33%), X-VLA (21.33%), and RDT-1B (15.00%).
>
---
#### [replaced 009] AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于图像目标导航任务，解决精确定位问题。通过几何查询方法，实现高精度位姿恢复，提升导航成功率和定位精度。**

- **链接: [https://arxiv.org/pdf/2604.05351](https://arxiv.org/pdf/2604.05351)**

> **作者:** Yijie Deng; Shuaihang Yuan; Yi Fang
>
> **摘要:** Image Goal Navigation (ImageNav) is evaluated by a coarse success criterion, the agent must stop within 1m of the target, which is sufficient for finding objects but falls short for downstream tasks such as grasping that require precise positioning. We introduce AnyImageNav, a training-free system that pushes ImageNav toward this more demanding setting. Our key insight is that the goal image can be treated as a geometric query: any photo of an object, a hallway, or a room corner can be registered to the agent's observations via dense pixel-level correspondences, enabling recovery of the exact 6-DoF camera pose. Our method realizes this through a semantic-to-geometric cascade: a semantic relevance signal guides exploration and acts as a proximity gate, invoking a 3D multi-view foundation model only when the current view is highly relevant to the goal image; the model then self-certifies its registration in a loop for an accurate recovered pose. Our method sets state-of-the-art navigation success rates on Gibson (93.1%) and HM3D (82.6%), and achieves pose recovery that prior methods do not provide: a position error of 0.27m and heading error of 3.41 degrees on Gibson, and 0.21m / 1.23 degrees on HM3D, a 5-10x improvement over adapted baselines.
>
---
#### [replaced 010] STERN: Simultaneous Trajectory Estimation and Relative Navigation for Autonomous Underwater Proximity Operations
- **分类: cs.RO**

- **简介: 该论文属于水下自主导航任务，解决AUV接近操作中的轨迹估计与相对导航问题。通过因子图建模，实现多种场景下的灵活导航方案。**

- **链接: [https://arxiv.org/pdf/2309.08780](https://arxiv.org/pdf/2309.08780)**

> **作者:** Aldo Terán Espinoza; Antonio Terán Espinoza; John Folkesson; Clemens Deutsch; Niklas Rolleberg; Peter Sigray; Jakob Kuttenkeuler
>
> **备注:** v2 updated after revision. Article contains 24 pages and 18 figures. Published in the IEEE Journal of Oceanic Engineering, available at: this https URL
>
> **摘要:** Due to the challenges regarding the limits of their endurance and autonomous capabilities, underwater docking for autonomous underwater vehicles (AUVs) has become a topic of interest for many academic and commercial applications. Herein, we take on the problem of relative navigation for the generalized version of the docking operation, which we address as proximity operations. Proximity operations typically involve only two actors, a chaser and a target. We leverage the similarities to proximity operations (prox-ops) from spacecraft robotic missions to frame the diverse docking scenarios with a set of phases the chaser undergoes on the way to its target. We emphasize the versatility on the use of factor graphs as a generalized representation to model the underlying simultaneous trajectory estimation and relative navigation (STERN) problem that arises with any prox-ops scenario, regardless of the sensor suite or the agents' dynamic constraints. To emphasize the flexibility of factor graphs as the modeling foundation for arbitrary underwater prox-ops, we compile a list of state-of-the-art research in the field and represent the different scenario using the same factor graph representation. We detail the procedure required to model, design, and implement factor graph-based estimators by addressing a long-distance acoustic homing scenario of an AUV to a moving mothership using datasets from simulated and real-world deployments; an analysis of these results is provided to shed light on the flexibility and limitations of the dynamic assumptions of the moving target. A description of our front- and back-end is also presented together with a timing breakdown of all processes to show its potential deployment on a real-time system.
>
---
#### [replaced 011] Differentiable SpaTiaL: Symbolic Learning and Reasoning with Geometric Temporal Logic for Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文提出Differentiable SpaTiaL，解决复杂操作任务中几何与时间约束的联合优化问题。通过构建可微分的几何原语，实现从语义规范到几何配置的端到端映射。**

- **链接: [https://arxiv.org/pdf/2604.02643](https://arxiv.org/pdf/2604.02643)**

> **作者:** Licheng Luo; Kaier Liang; Cristian-Ioan Vasile; Mingyu Cai
>
> **备注:** Code available at: this https URL
>
> **摘要:** Executing complex manipulation in cluttered environments requires satisfying coupled geometric and temporal constraints. Although Spatio-Temporal Logic (SpaTiaL) offers a principled specification framework, its use in gradient-based optimization is limited by non-differentiable geometric operations. Existing differentiable temporal logics focus on the robot's internal state and neglect interactive object-environment relations, while spatial logic approaches that capture such interactions rely on discrete geometry engines that break the computational graph and preclude exact gradient propagation. To overcome this limitation, we propose Differentiable SpaTiaL, a fully tensorized toolbox that constructs smooth, autograd-compatible geometric primitives directly over polygonal sets. To the best of our knowledge, this is the first end-to-end differentiable symbolic spatio-temporal logic toolbox. By analytically deriving differentiable relaxations of key spatial predicates--including signed distance, intersection, containment, and directional relations--we enable an end-to-end differentiable mapping from high-level semantic specifications to low-level geometric configurations, without invoking external discrete solvers. This fully differentiable formulation unlocks two core capabilities: (i) massively parallel trajectory optimization under rigorous spatio-temporal constraints, and (ii) direct learning of spatial logic parameters from demonstrations via backpropagation. Experimental results validate the effectiveness and scalability of the proposed framework.
>
---
#### [replaced 012] LeLaR: The First In-Orbit Demonstration of an AI-Based Satellite Attitude Controller
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 论文介绍了一种基于AI的卫星姿态控制器，用于解决传统控制器设计复杂、对模型不确定性强的问题。通过深度强化学习，在仿真中训练并成功部署到实际卫星，验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2512.19576](https://arxiv.org/pdf/2512.19576)**

> **作者:** Kirill Djebko; Tom Baumann; Erik Dilger; Frank Puppe; Sergio Montenegro
>
> **备注:** Accepted for publication in IEEE Access (DOI: https://doi.org/10.1109/ACCESS.2026.3678816). This is the author's version which has not been fully edited and content may change prior to final publication. 20 pages, 15 figures, 18 tables. The maneuver telemetry datasets are available in the GitHub repository under this https URL
>
> **摘要:** Attitude control is essential for many satellite missions. Classical controllers, however, are time-consuming to design and sensitive to model uncertainties and variations in operational boundary conditions. Deep Reinforcement Learning (DRL) offers a promising alternative by learning adaptive control strategies through autonomous interaction with a simulation environment. Overcoming the Sim2Real gap, which involves deploying an agent trained in simulation onto the real physical satellite, remains a significant challenge. In this work, we present the first successful in-orbit demonstration of an AI-based attitude controller for inertial pointing maneuvers. The controller was trained entirely in simulation and deployed to the InnoCube 3U nanosatellite, which was developed by the Julius-Maximilians-Universität Würzburg in cooperation with the Technische Universität Berlin, and launched in January 2025. We present the AI agent design, the methodology of the training procedure, the discrepancies between the simulation and the observed behavior of the real satellite, and a comparison of the AI-based attitude controller with the classical PD controller of InnoCube. Steady-state metrics confirm the robust performance of the AI-based controller during repeated in-orbit maneuvers.
>
---
#### [replaced 013] Before Humans Join the Team: Diagnosing Coordination Failures in Healthcare Robot Team Simulation
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于人机协作任务，旨在解决医疗机器人团队协调失败问题。通过仿真模拟分析团队结构对协调的影响，为安全的人类接入提供基础。**

- **链接: [https://arxiv.org/pdf/2508.04691](https://arxiv.org/pdf/2508.04691)**

> **作者:** Yuanchen Bai; Zijian Ding; Shaoyue Wen; Xiang Chang; Angelique Taylor
>
> **备注:** Revised version incorporating new analysis and restructuring
>
> **摘要:** As humans move toward collaborating with coordinated robot teams, understanding how these teams coordinate and fail is essential for building trust and ensuring safety. However, exposing human collaborators to coordination failures during early-stage development is costly and risky, particularly in high-stakes domains such as healthcare. We adopt an agent-simulation approach in which all team roles, including the supervisory manager, are instantiated as LLM agents, allowing us to diagnose coordination failures before humans join the team. Using a controllable healthcare scenario, we conduct two studies with different hierarchical configurations to analyze coordination behaviors and failure patterns. Our findings reveal that team structure, rather than contextual knowledge or model capability, constitutes the primary bottleneck for coordination, and expose a tension between reasoning autonomy and system stability. By surfacing these failures in simulation, we prepare the groundwork for safe human integration. These findings inform the design of resilient robot teams with implications for process-level evaluation, transparent coordination protocols, and structured human integration. Supplementary materials, including codes, task agent setup, trace outputs, and annotated examples of coordination failures and reasoning behaviors, are available at: this https URL.
>
---
#### [replaced 014] Precise Aggressive Aerial Maneuvers with Sensorimotor Policies
- **分类: cs.RO**

- **简介: 该论文属于无人机自主导航任务，解决在狭窄空间内精确快速飞行的问题。通过传感器-运动策略实现无人机在受限环境中的高效穿越。**

- **链接: [https://arxiv.org/pdf/2604.05828](https://arxiv.org/pdf/2604.05828)**

> **作者:** Tianyue Wu; Guangtong Xu; Zihan Wang; Junxiao Lin; Tianyang Chen; Yuze Wu; Zhichao Han; Zhiyang Liu; Fei Gao
>
> **备注:** This manuscript was submitted in June 2025. The first revision was submitted in November 2025. The second revision was submitted in February 2026. The first two authors contributed equally to this work
>
> **摘要:** Precise aggressive maneuvers with lightweight onboard sensors remains a key bottleneck in fully exploiting the maneuverability of drones. Such maneuvers are critical for expanding the systems' accessible area by navigating through narrow openings in the environment. Among the most relevant problems, a representative one is aggressive traversal through narrow gaps with quadrotors under SE(3) constraints, which require the quadrotors to leverage a momentary tilted attitude and the asymmetry of the airframe to navigate through gaps. In this paper, we achieve such maneuvers by developing sensorimotor policies directly mapping onboard vision and proprioception into low-level control commands. The policies are trained using reinforcement learning (RL) with end-to-end policy distillation in simulation. We mitigate the fundamental hardness of model-free RL's exploration on the restricted solution space with an initialization strategy leveraging trajectories generated by a model-based planner. Careful sim-to-real design allows the policy to control a quadrotor through narrow gaps with low clearances and high repeatability. For instance, the proposed method enables a quadrotor to navigate a rectangular gap at a 5 cm clearance, tilted at up to 90-degree orientation, without knowledge of the gap's position or orientation. Without training on dynamic gaps, the policy can reactively servo the quadrotor to traverse through a moving gap. The proposed method is also validated by training and deploying policies on challenging tracks of narrow gaps placed closely. The flexibility of the policy learning method is demonstrated by developing policies for geometrically diverse gaps, without relying on manually defined traversal poses and visual features.
>
---
#### [replaced 015] Exploring Conditions for Diffusion models in Robotic Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决预训练视觉表示在控制任务中适应性不足的问题。通过引入可学习的提示，提升视觉表示的动态适应性，从而提高控制性能。**

- **链接: [https://arxiv.org/pdf/2510.15510](https://arxiv.org/pdf/2510.15510)**

> **作者:** Heeseong Shin; Byeongho Heo; Dongyoon Han; Seungryong Kim; Taekyung Kim
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** While pre-trained visual representations have significantly advanced imitation learning, they are often task-agnostic as they remain frozen during policy learning. In this work, we explore leveraging pre-trained text-to-image diffusion models to obtain task-adaptive visual representations for robotic control, without fine-tuning the model itself. However, we find that naively applying textual conditions - a successful strategy in other vision domains - yields minimal or even negative gains in control tasks. We attribute this to the domain gap between the diffusion model's training data and robotic control environments, leading us to argue for conditions that consider the specific, dynamic visual information required for control. To this end, we propose ORCA, which introduces learnable task prompts that adapt to the control environment and visual prompts that capture fine-grained, frame-specific details. Through facilitating task-adaptive representations with our newly devised conditions, our approach achieves state-of-the-art performance on various robotic control benchmarks, significantly surpassing prior methods.
>
---
#### [replaced 016] ODYN: An All-Shifted Non-Interior-Point Method for Quadratic Programming in Robotics and AI
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ODYN，一种用于机器人和AI的新型二次规划求解器，解决密集和稀疏QP问题，无需约束线性独立性，具备强热启动性能。**

- **链接: [https://arxiv.org/pdf/2602.16005](https://arxiv.org/pdf/2602.16005)**

> **作者:** Jose Rojas; Aristotelis Papatheodorou; Sergi Martinez; Andrea Patrizi; Ioannis Havoutis; Carlos Mastalli
>
> **备注:** 20 pages, 12 figures, under-review
>
> **摘要:** We introduce ODYN, a novel all-shifted primal-dual non-interior-point quadratic programming (QP) solver designed to efficiently handle challenging dense and sparse QPs. ODYN combines all-shifted nonlinear complementarity problem (NCP) functions with proximal method of multipliers to robustly address ill-conditioned and degenerate problems, without requiring linear independence of the constraints. It exhibits strong warm-start performance and is well suited to both general-purpose optimization, and robotics and AI applications, including model-based control, estimation, and kernel-based learning methods. We provide an open-source implementation and benchmark ODYN on the Maros-Mészáros test set, demonstrating state-of-the-art convergence performance in small-to-high-scale problems. The results highlight ODYN's superior warm-starting capabilities, which are critical in sequential and real-time settings common in robotics and AI. These advantages are further demonstrated by deploying ODYN as the backend of an SQP-based predictive control framework (OdynSQP), as the implicitly differentiable optimization layer for deep learning (ODYNLayer), and the optimizer of a contact-dynamics simulation (ODYNSim).
>
---
#### [replaced 017] Simultaneous Calibration of Noise Covariance and Kinematics for State Estimation of Legged Robots via Bi-level Optimization
- **分类: cs.RO; math.OC**

- **简介: 该论文属于机器人状态估计任务，解决噪声协方差和运动学参数未知的问题，通过双层优化框架实现联合校准。**

- **链接: [https://arxiv.org/pdf/2510.11539](https://arxiv.org/pdf/2510.11539)**

> **作者:** Denglin Cheng; Jiarong Kang; Xiaobin Xiong
>
> **摘要:** Accurate state estimation is critical for legged and aerial robots operating in dynamic, uncertain environments. A key challenge lies in specifying process and measurement noise covariances, which are typically unknown or manually tuned. In this work, we introduce a bi-level optimization framework that jointly calibrates covariance matrices and kinematic parameters in an estimator-in-the-loop manner. The upper level treats noise covariances and model parameters as optimization variables, while the lower level executes a full-information estimator. Differentiating through the estimator allows direct optimization of trajectory-level objectives, resulting in accurate and consistent state estimates. We validate our approach on quadrupedal and humanoid robots, demonstrating significantly improved estimation accuracy and uncertainty calibration compared to hand-tuned baselines. Our method unifies state estimation, sensor, and kinematics calibration into a principled, data-driven framework applicable across diverse robotic platforms.
>
---
#### [replaced 018] SemanticScanpath: Combining Gaze and Speech for Situated Human-Robot Interaction Using LLMs
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人理解模糊口语和非语言意图的问题。通过结合语音与眼动数据，提升机器人对上下文的感知能力。**

- **链接: [https://arxiv.org/pdf/2503.16548](https://arxiv.org/pdf/2503.16548)**

> **作者:** Elisabeth Menendez; Michael Gienger; Santiago Martínez; Carlos Balaguer; Anna Belardinelli
>
> **摘要:** Large Language Models (LLMs) have substantially improved the conversational capabilities of social robots. Nevertheless, for an intuitive and fluent human-robot interaction, robots should be able to ground the conversation by relating ambiguous or underspecified spoken utterances to the current physical situation and to the intents expressed nonverbally by the user, such as through referential gaze. Here, we propose a representation that integrates speech and gaze to enable LLMs to achieve higher situated awareness and correctly resolve ambiguous requests. Our approach relies on a text-based semantic translation of the scanpath produced by the user, along with the verbal requests. It demonstrates LLMs' capabilities to reason about gaze behavior, robustly ignoring spurious glances or irrelevant objects. We validate the system across multiple tasks and two scenarios, showing its superior generality and accuracy compared to control conditions. We demonstrate an implementation on a robotic platform, closing the loop from request interpretation to execution.
>
---
#### [replaced 019] RK-MPC: Residual Koopman Model Predictive Control for Quadruped Locomotion in Offroad Environments
- **分类: cs.RO**

- **简介: 该论文提出RK-MPC，用于四足机器人在非结构化环境中的运动控制，解决模型预测控制中的精度与实时性问题，通过数据驱动的残差修正提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.04221](https://arxiv.org/pdf/2604.04221)**

> **作者:** Sriram S. K. S. Narayanan; Umesh Vaidya
>
> **摘要:** This paper presents Residual Koopman MPC (RK-MPC), a Koopman-based, data-driven model predictive control framework for quadruped locomotion that improves prediction fidelity while preserving real-time tractability. RK-MPC augments a nominal template model with a compact linear residual predictor learned from data in lifted coordinates, enabling systematic correction of model mismatch induced by contact variability and terrain disturbances with provable bounds on multi-step prediction error. The learned residual model is embedded within a convex quadratic-program MPC formulation, yielding a receding-horizon controller that runs onboard at 500 Hz and retains the structure and constraint-handling advantages of optimization-based control. We evaluate RK-MPC in both Gazebo simulation and Unitree Go1 hardware experiments, demonstrating reliable blind locomotion across contact disturbances, multiple gait schedules, and challenging off-road terrains including grass, gravel, snow, and ice. We further compare against Koopman/EDMD baselines using alternative observable dictionaries, including monomial and $SE(3)$-structured bases, and show that the residual correction improves multi-step prediction and closed-loop performance while reducing sensitivity to the choice of observables. Overall, RK-MPC provides a practical, hardware-validated pathway for data-driven predictive control of quadrupeds in unstructured environments. See this https URL for implementation videos.
>
---
