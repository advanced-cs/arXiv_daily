# 机器人 cs.RO

- **最新发布 36 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] Vision-Language Models on the Edge for Real-Time Robotic Perception
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究在6G边缘计算环境下部署视觉语言模型，解决机器人实时感知中的延迟和资源限制问题。通过对比边缘与云端部署效果，验证了边缘计算的可行性。**

- **链接: [https://arxiv.org/pdf/2601.14921v1](https://arxiv.org/pdf/2601.14921v1)**

> **作者:** Sarat Ahmad; Maryam Hafeez; Syed Ali Raza Zaidi
>
> **摘要:** Vision-Language Models (VLMs) enable multimodal reasoning for robotic perception and interaction, but their deployment in real-world systems remains constrained by latency, limited onboard resources, and privacy risks of cloud offloading. Edge intelligence within 6G, particularly Open RAN and Multi-access Edge Computing (MEC), offers a pathway to address these challenges by bringing computation closer to the data source. This work investigates the deployment of VLMs on ORAN/MEC infrastructure using the Unitree G1 humanoid robot as an embodied testbed. We design a WebRTC-based pipeline that streams multimodal data to an edge node and evaluate LLaMA-3.2-11B-Vision-Instruct deployed at the edge versus in the cloud under real-time conditions. Our results show that edge deployment preserves near-cloud accuracy while reducing end-to-end latency by 5\%. We further evaluate Qwen2-VL-2B-Instruct, a compact model optimized for resource-constrained environments, which achieves sub-second responsiveness, cutting latency by more than half but at the cost of accuracy.
>
---
#### [new 002] Risk Estimation for Automated Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，旨在解决风险估计问题。通过结合碰撞概率与严重性，提出一种通用、准确且高效的计算方法。**

- **链接: [https://arxiv.org/pdf/2601.15018v1](https://arxiv.org/pdf/2601.15018v1)**

> **作者:** Leon Tolksdorf; Arturo Tejada; Jonas Bauernfeind; Christian Birkner; Nathan van de Wouw
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** Safety is a central requirement for automated vehicles. As such, the assessment of risk in automated driving is key in supporting both motion planning technologies and safety evaluation. In automated driving, risk is characterized by two aspects. The first aspect is the uncertainty on the state estimates of other road participants by an automated vehicle. The second aspect is the severity of a collision event with said traffic participants. Here, the uncertainty aspect typically causes the risk to be non-zero for near-collision events. This makes risk particularly useful for automated vehicle motion planning. Namely, constraining or minimizing risk naturally navigates the automated vehicle around traffic participants while keeping a safety distance based on the level of uncertainty and the potential severity of the impending collision. Existing approaches to calculate the risk either resort to empirical modeling or severe approximations, and, hence, lack generalizability and accuracy. In this paper, we combine recent advances in collision probability estimation with the concept of collision severity to develop a general method for accurate risk estimation. The proposed method allows us to assign individual severity functions for different collision constellations, such as, e.g., frontal or side collisions. Furthermore, we show that the proposed approach is computationally efficient, which is beneficial, e.g., in real-time motion planning applications. The programming code for an exemplary implementation of Gaussian uncertainties is also provided.
>
---
#### [new 003] ExPrIS: Knowledge-Level Expectations as Priors for Object Interpretation from Sensor Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人场景理解任务，旨在解决对象识别中语义不一致的问题。通过构建3D语义场景图并融合先验知识，提升对象解释的鲁棒性与一致性。**

- **链接: [https://arxiv.org/pdf/2601.15025v1](https://arxiv.org/pdf/2601.15025v1)**

> **作者:** Marian Renz; Martin Günther; Felix Igelbrink; Oscar Lima; Martin Atzmueller
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this article is published in KI - Künstliche Intelligenz, and is available online at https://doi.org/10.1007/s13218-026-00901-7
>
> **摘要:** While deep learning has significantly advanced robotic object recognition, purely data-driven approaches often lack semantic consistency and fail to leverage valuable, pre-existing knowledge about the environment. This report presents the ExPrIS project, which addresses this challenge by investigating how knowledge-level expectations can serve as to improve object interpretation from sensor data. Our approach is based on the incremental construction of a 3D Semantic Scene Graph (3DSSG). We integrate expectations from two sources: contextual priors from past observations and semantic knowledge from external graphs like ConceptNet. These are embedded into a heterogeneous Graph Neural Network (GNN) to create an expectation-biased inference process. This method moves beyond static, frame-by-frame analysis to enhance the robustness and consistency of scene understanding over time. The report details this architecture, its evaluation, and outlines its planned integration on a mobile robotic platform.
>
---
#### [new 004] On-the-fly hand-eye calibration for the da Vinci surgical robot
- **分类: cs.RO**

- **简介: 该论文属于机器人手术中的工具定位任务，旨在解决电缆驱动机器人因编码器误差导致的定位问题。通过在线计算手眼变换矩阵，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2601.14871v1](https://arxiv.org/pdf/2601.14871v1)**

> **作者:** Zejian Cui; Ferdinando Rodriguez y Baena
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** In Robot-Assisted Minimally Invasive Surgery (RMIS), accurate tool localization is crucial to ensure patient safety and successful task execution. However, this remains challenging for cable-driven robots, such as the da Vinci robot, because erroneous encoder readings lead to pose estimation errors. In this study, we propose a calibration framework to produce accurate tool localization results through computing the hand-eye transformation matrix on-the-fly. The framework consists of two interrelated algorithms: the feature association block and the hand-eye calibration block, which provide robust correspondences for key points detected on monocular images without pre-training, and offer the versatility to accommodate various surgical scenarios by adopting an array of filter approaches, respectively. To validate its efficacy, we test the framework extensively on publicly available video datasets that feature multiple surgical instruments conducting tasks in both in vitro and ex vivo scenarios, under varying illumination conditions and with different levels of key point measurement accuracy. The results show a significant reduction in tool localization errors under the proposed calibration framework, with accuracies comparable to other state-of-the-art methods while being more time-efficient.
>
---
#### [new 005] Agentic AI Meets Edge Computing in Autonomous UAV Swarms
- **分类: cs.RO; cs.AI**

- **简介: 论文探讨将基于大语言模型的智能体AI与边缘计算结合，提升无人机群在高风险场景下的自主性与可靠性。任务属于无人机群自主控制，解决计算资源受限与动态环境挑战。工作包括架构设计与火灾救援案例验证。**

- **链接: [https://arxiv.org/pdf/2601.14437v1](https://arxiv.org/pdf/2601.14437v1)**

> **作者:** Thuan Minh Nguyen; Vu Tuan Truong; Long Bao Le
>
> **摘要:** The integration of agentic AI, powered by large language models (LLMs) with autonomous reasoning, planning, and execution, into unmanned aerial vehicle (UAV) swarms opens new operational possibilities and brings the vision of the Internet of Drones closer to reality. However, infrastructure constraints, dynamic environments, and the computational demands of multi-agent coordination limit real-world deployment in high-risk scenarios such as wildfires and disaster response. This paper investigates the integration of LLM-based agentic AI and edge computing to realize scalable and resilient autonomy in UAV swarms. We first discuss three architectures for supporting UAV swarms - standalone, edge-enabled, and edge-cloud hybrid deployment - each optimized for varying autonomy and connectivity levels. Then, a use case for wildfire search and rescue (SAR) is designed to demonstrate the efficiency of the edge-enabled architecture, enabling high SAR coverage, reduced mission completion times, and a higher level of autonomy compared to traditional approaches. Finally, we highlight open challenges in integrating LLMs and edge computing for mission-critical UAV-swarm applications.
>
---
#### [new 006] Spatially Generalizable Mobile Manipulation via Adaptive Experience Selection and Dynamic Imagination
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，解决样本效率低和空间泛化能力差的问题。通过自适应经验选择和动态想象，提升技能学习与迁移能力。**

- **链接: [https://arxiv.org/pdf/2601.14649v1](https://arxiv.org/pdf/2601.14649v1)**

> **作者:** Ping Zhong; Liangbai Liu; Bolei Chen; Tao Wu; Jiazhi Xia; Chaoxu Mu; Jianxin Wang
>
> **摘要:** Mobile Manipulation (MM) involves long-horizon decision-making over multi-stage compositions of heterogeneous skills, such as navigation and picking up objects. Despite recent progress, existing MM methods still face two key limitations: (i) low sample efficiency, due to ineffective use of redundant data generated during long-term MM interactions; and (ii) poor spatial generalization, as policies trained on specific tasks struggle to transfer to new spatial layouts without additional training. In this paper, we address these challenges through Adaptive Experience Selection (AES) and model-based dynamic imagination. In particular, AES makes MM agents pay more attention to critical experience fragments in long trajectories that affect task success, improving skill chain learning and mitigating skill forgetting. Based on AES, a Recurrent State-Space Model (RSSM) is introduced for Model-Predictive Forward Planning (MPFP) by capturing the coupled dynamics between the mobile base and the manipulator and imagining the dynamics of future manipulations. RSSM-based MPFP can reinforce MM skill learning on the current task while enabling effective generalization to new spatial layouts. Comparative studies across different experimental configurations demonstrate that our method significantly outperforms existing MM policies. Real-world experiments further validate the feasibility and practicality of our method.
>
---
#### [new 007] A Brain-inspired Embodied Intelligence for Fluid and Fast Reflexive Robotics Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决机器人动态稳定性与快速反应问题。提出NeuroVLA框架，模仿生物神经系统结构，实现高效、稳定、具时序记忆的运动控制。**

- **链接: [https://arxiv.org/pdf/2601.14628v1](https://arxiv.org/pdf/2601.14628v1)**

> **作者:** Weiyu Guo; He Zhang; Pengteng Li; Tiefu Cai; Ziyang Chen; Yandong Guo; Xiao He; Yongkui Yang; Ying Sun; Hui Xiong
>
> **摘要:** Recent advances in embodied intelligence have leveraged massive scaling of data and model parameters to master natural-language command following and multi-task control. In contrast, biological systems demonstrate an innate ability to acquire skills rapidly from sparse experience. Crucially, current robotic policies struggle to replicate the dynamic stability, reflexive responsiveness, and temporal memory inherent in biological motion. Here we present Neuromorphic Vision-Language-Action (NeuroVLA), a framework that mimics the structural organization of the bio-nervous system between the cortex, cerebellum, and spinal cord. We adopt a system-level bio-inspired design: a high-level model plans goals, an adaptive cerebellum module stabilizes motion using high-frequency sensors feedback, and a bio-inspired spinal layer executes lightning-fast actions generation. NeuroVLA represents the first deployment of a neuromorphic VLA on physical robotics, achieving state-of-the-art performance. We observe the emergence of biological motor characteristics without additional data or special guidance: it stops the shaking in robotic arms, saves significant energy(only 0.4w on Neuromorphic Processor), shows temporal memory ability and triggers safety reflexes in less than 20 milliseconds.
>
---
#### [new 008] UNCLE-Grasp: Uncertainty-Aware Grasping of Leaf-Occluded Strawberries
- **分类: cs.RO**

- **简介: 该论文属于机器人采摘任务，解决部分遮挡下草莓抓取的不确定性问题。通过建模形状不确定性，提出一种基于多假设的抓取决策方法，提升抓取可靠性。**

- **链接: [https://arxiv.org/pdf/2601.14492v1](https://arxiv.org/pdf/2601.14492v1)**

> **作者:** Malak Mansour; Ali Abouzeid; Zezhou Sun; Qinbo Sun; Dezhen Song; Abdalla Swikir
>
> **摘要:** Robotic strawberry harvesting is challenging under partial occlusion, where leaves induce significant geometric uncertainty and make grasp decisions based on a single deterministic shape estimate unreliable. From a single partial observation, multiple incompatible 3D completions may be plausible, causing grasps that appear feasible on one completion to fail on another. We propose an uncertainty-aware grasping pipeline for partially occluded strawberries that explicitly models completion uncertainty arising from both occlusion and learned shape reconstruction. Our approach uses point cloud completion with Monte Carlo dropout to sample multiple shape hypotheses, generates candidate grasps for each completion, and evaluates grasp feasibility using physically grounded force-closure-based metrics. Rather than selecting a grasp based on a single estimate, we aggregate feasibility across completions and apply a conservative lower confidence bound (LCB) criterion to decide whether a grasp should be attempted or safely abstained. We evaluate the proposed method in simulation and on a physical robot across increasing levels of synthetic and real leaf occlusion. Results show that uncertainty-aware decision making enables reliable abstention from high-risk grasp attempts under severe occlusion while maintaining robust grasp execution when geometric confidence is sufficient, outperforming deterministic baselines in both simulated and physical robot experiments.
>
---
#### [new 009] FARE: Fast-Slow Agentic Robotic Exploration
- **分类: cs.RO**

- **简介: 该论文属于自主机器人探索任务，解决环境感知与路径规划问题。提出FARE框架，结合大语言模型和强化学习，提升探索效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.14681v1](https://arxiv.org/pdf/2601.14681v1)**

> **作者:** Shuhao Liao; Xuxin Lv; Jeric Lew; Shizhe Zhang; Jingsong Liang; Peizhuo Li; Yuhong Cao; Wenjun Wu; Guillaume Sartoretti
>
> **摘要:** This work advances autonomous robot exploration by integrating agent-level semantic reasoning with fast local control. We introduce FARE, a hierarchical autonomous exploration framework that integrates a large language model (LLM) for global reasoning with a reinforcement learning (RL) policy for local decision making. FARE follows a fast-slow thinking paradigm. The slow-thinking LLM module interprets a concise textual description of the unknown environment and synthesizes an agent-level exploration strategy, which is then grounded into a sequence of global waypoints through a topological graph. To further improve reasoning efficiency, this module employs a modularity-based pruning mechanism that reduces redundant graph structures. The fast-thinking RL module executes exploration by reacting to local observations while being guided by the LLM-generated global waypoints. The RL policy is additionally shaped by a reward term that encourages adherence to the global waypoints, enabling coherent and robust closed-loop behavior. This architecture decouples semantic reasoning from geometric decision, allowing each module to operate in its appropriate temporal and spatial scale. In challenging simulated environments, our results show that FARE achieves substantial improvements in exploration efficiency over state-of-the-art baselines. We further deploy FARE on hardware and validate it in complex, large scale $200m\times130m$ building environment.
>
---
#### [new 010] Influence of Operator Expertise on Robot Supervision and Intervention
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，研究操作者技能水平对机器人监督与干预的影响，通过实验分析不同用户在监督机器人时的决策模式和干预行为。**

- **链接: [https://arxiv.org/pdf/2601.15069v1](https://arxiv.org/pdf/2601.15069v1)**

> **作者:** Yanran Jiang; Pavan Sikka; Leimin Tian; Dana Kuliic; Cecile Paris
>
> **摘要:** With increasing levels of robot autonomy, robots are increasingly being supervised by users with varying levels of robotics expertise. As the diversity of the user population increases, it is important to understand how users with different expertise levels approach the supervision task and how this impacts performance of the human-robot team. This exploratory study investigates how operators with varying expertise levels perceive information and make intervention decisions when supervising a remote robot. We conducted a user study (N=27) where participants supervised a robot autonomously exploring four unknown tunnel environments in a simulator, and provided waypoints to intervene when they believed the robot had encountered difficulties. By analyzing the interaction data and questionnaire responses, we identify differing patterns in intervention timing and decision-making strategies across novice, intermediate, and expert users.
>
---
#### [new 011] Probing Prompt Design for Socially Compliant Robot Navigation with Vision Language Models
- **分类: cs.RO**

- **简介: 该论文属于社会合规机器人导航任务，旨在解决小视觉语言模型在导航中决策能力不足的问题，通过设计有效提示提升导航性能。**

- **链接: [https://arxiv.org/pdf/2601.14622v1](https://arxiv.org/pdf/2601.14622v1)**

> **作者:** Ling Xiao; Toshihiko Yamasaki
>
> **摘要:** Language models are increasingly used for social robot navigation, yet existing benchmarks largely overlook principled prompt design for socially compliant behavior. This limitation is particularly relevant in practice, as many systems rely on small vision language models (VLMs) for efficiency. Compared to large language models, small VLMs exhibit weaker decision-making capabilities, making effective prompt design critical for accurate navigation. Inspired by cognitive theories of human learning and motivation, we study prompt design along two dimensions: system guidance (action-focused, reasoning-oriented, and perception-reasoning prompts) and motivational framing, where models compete against humans, other AI systems, or their past selves. Experiments on two socially compliant navigation datasets reveal three key findings. First, for non-finetuned GPT-4o, competition against humans achieves the best performance, while competition against other AI systems performs worst. For finetuned models, competition against the model's past self yields the strongest results, followed by competition against humans, with performance further influenced by coupling effects among prompt design, model choice, and dataset characteristics. Second, inappropriate system prompt design can significantly degrade performance, even compared to direct finetuning. Third, while direct finetuning substantially improves semantic-level metrics such as perception, prediction, and reasoning, it yields limited gains in action accuracy. In contrast, our system prompts produce a disproportionately larger improvement in action accuracy, indicating that the proposed prompt design primarily acts as a decision-level constraint rather than a representational enhancement.
>
---
#### [new 012] Robust Haptic Rendering Using a Nonlinear Impedance Matching Approach (NIMA) for Robotic Laparoscopic Surgery
- **分类: cs.RO**

- **简介: 该论文属于机器人手术中的触觉反馈任务，旨在解决力渲染不准确和系统安全性问题。通过提出非线性阻抗匹配方法（NIMA），提升力反馈精度与安全性。**

- **链接: [https://arxiv.org/pdf/2601.14445v1](https://arxiv.org/pdf/2601.14445v1)**

> **作者:** Aiden Mazidi; Majid Roshanfar; Amir Sayadi; Javad Dargahi; Jake Barralet; Liane S. Feldman; Amir Hooshiar
>
> **摘要:** Background: The integration of haptic feedback into robot-assisted minimally invasive surgery (RAMIS) has long been limited by challenges in accurately rendering forces and ensuring system safety. The need for robust, high-fidelity haptic systems is critical for enhancing the precision and reliability of teleoperated surgical tools. Methods: In this study, we present a Nonlinear Impedance Matching Approach (NIMA) designed to improve force rendering by accurately modelling complex tool-tissue interactions. Based on our previously validated Impedance Matching Approach (IMA), our novel NIMA method includes nonlinear dynamics to capture and render tool-tissue forces effectively. Results: NIMA improves force feedback accuracy with a mean absolute error (MAE) of 0.01 (SD 0.02) N, achieving a 95% reduction in MAE compared to IMA. Furthermore, NIMA effectively eliminates haptic "kickback" by ensuring no force is applied by the haptic device to the user's hand when they release the handle, enhancing both patient safety and user comfort. Conclusion: NIMA's ability to account for nonlinearities in tool-tissue interactions provides an improvement in force fidelity, responsiveness, and precision across various surgical conditions. Our findings promote the advancement of haptic feedback systems for robotic surgery, offering a realistic and reliable interface for robot-assisted surgical procedures.
>
---
#### [new 013] DWPP: Dynamic Window Pure Pursuit Considering Velocity and Acceleration Constraints
- **分类: cs.RO**

- **简介: 该论文属于移动机器人路径跟踪任务，解决传统方法未考虑速度和加速度约束导致的跟踪误差问题，提出DWPP算法，在速度空间中优化指令速度以提高跟踪精度。**

- **链接: [https://arxiv.org/pdf/2601.15006v1](https://arxiv.org/pdf/2601.15006v1)**

> **作者:** Fumiya Ohnishi; Masaki Takahashi
>
> **备注:** 28 pages, 12 figures
>
> **摘要:** Pure pursuit and its variants are widely used for mobile robot path tracking owing to their simplicity and computational efficiency. However, many conventional approaches do not explicitly account for velocity and acceleration constraints, resulting in discrepancies between commanded and actual velocities that result in overshoot and degraded tracking performance. To address this problem, this paper proposes dynamic window pure pursuit (DWPP), which fundamentally reformulates the command velocity computation process to explicitly incorporate velocity and acceleration constraints. Specifically, DWPP formulates command velocity computation in the velocity space (the $v$-$ω$ plane) and selects the command velocity as the point within the dynamic window that is closest to the line $ω= κv$. Experimental results demonstrate that DWPP avoids constraint-violating commands and achieves superior path-tracking accuracy compared with conventional pure pursuit methods. The proposed method has been integrated into the official Nav2 repository and is publicly available (https://github.com/ros-navigation/navigation2).
>
---
#### [new 014] Graph-Based Adaptive Planning for Coordinated Dual-Arm Robotic Disassembly of Electronic Devices (eGRAP)
- **分类: cs.RO**

- **简介: 该论文提出eGRAP系统，用于电子设备的双臂自主拆解。解决的是电子垃圾高效回收问题，通过视觉、动态规划和双臂协作实现自适应任务协调。**

- **链接: [https://arxiv.org/pdf/2601.14998v1](https://arxiv.org/pdf/2601.14998v1)**

> **作者:** Adip Ranjan Das; Maria Koskinopoulou
>
> **备注:** 7 Pages, 8 Figures, 5 Tables
>
> **摘要:** E-waste is growing rapidly while recycling rates remain low. We propose an electronic-device Graph-based Adaptive Planning (eGRAP) that integrates vision, dynamic planning, and dual-arm execution for autonomous disassembly. A camera-equipped arm identifies parts and estimates their poses, and a directed graph encodes which parts must be removed first. A scheduler uses topological ordering of this graph to select valid next steps and assign them to two robot arms, allowing independent tasks to run in parallel. One arm carries a screwdriver (with an eye-in-hand depth camera) and the other holds or handles components. We demonstrate eGRAP on 3.5in hard drives: as parts are unscrewed and removed, the system updates its graph and plan online. Experiments show consistent full disassembly of each HDD, with high success rates and efficient cycle times, illustrating the method's ability to adaptively coordinate dual-arm tasks in real time.
>
---
#### [new 015] MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI
- **分类: cs.RO**

- **简介: 该论文属于自主无人机竞速任务，解决轻量级AI在复杂环境中的实时控制问题。提出MonoRace系统，使用单目相机和IMU实现高精度状态估计与控制，成功夺冠并达到100km/h速度。**

- **链接: [https://arxiv.org/pdf/2601.15222v1](https://arxiv.org/pdf/2601.15222v1)**

> **作者:** Stavrow A. Bahnam; Robin Ferede; Till M. Blaha; Anton E. Lang; Erin Lucassen; Quentin Missinne; Aderik E. C. Verraest; Christophe De Wagter; Guido C. H. E. de Croon
>
> **摘要:** Autonomous drone racing represents a major frontier in robotics research. It requires an Artificial Intelligence (AI) that can run on board light-weight flying robots under tight resource and time constraints, while pushing the physical system to its limits. The state of the art in this area consists of a system with a stereo camera and an inertial measurement unit (IMU) that beat human drone racing champions in a controlled indoor environment. Here, we present MonoRace: an onboard drone racing approach that uses a monocular, rolling-shutter camera and IMU that generalizes to a competition environment without any external motion tracking system. The approach features robust state estimation that combines neural-network-based gate segmentation with a drone model. Moreover, it includes an offline optimization procedure that leverages the known geometry of gates to refine any state estimation parameter. This offline optimization is based purely on onboard flight data and is important for fine-tuning the vital external camera calibration parameters. Furthermore, the guidance and control are performed by a neural network that foregoes inner loop controllers by directly sending motor commands. This small network runs on the flight controller at 500Hz. The proposed approach won the 2025 Abu Dhabi Autonomous Drone Racing Competition (A2RL), outperforming all competing AI teams and three human world champion pilots in a direct knockout tournament. It set a new milestone in autonomous drone racing research, reaching speeds up to 100 km/h on the competition track and successfully coping with problems such as camera interference and IMU saturation.
>
---
#### [new 016] RoboBrain 2.5: Depth in Sight, Time in Mind
- **分类: cs.RO**

- **简介: 该论文提出RoboBrain 2.5，解决机器人感知与操作中的时空建模问题，通过3D空间推理和时间价值估计提升复杂操作能力。**

- **链接: [https://arxiv.org/pdf/2601.14352v1](https://arxiv.org/pdf/2601.14352v1)**

> **作者:** Huajie Tan; Enshen Zhou; Zhiyu Li; Yijie Xu; Yuheng Ji; Xiansheng Chen; Cheng Chi; Pengwei Wang; Huizhu Jia; Yulong Ao; Mingyu Cao; Sixiang Chen; Zhe Li; Mengzhen Liu; Zixiao Wang; Shanyu Rong; Yaoxu Lyu; Zhongxia Zhao; Peterson Co; Yibo Li; Yi Han; Shaoxuan Xie; Guocai Yao; Songjing Wang; Leiduo Zhang; Xi Yang; Yance Jiao; Donghai Shi; Kunchang Xie; Shaokai Nie; Chunlei Men; Yonghua Lin; Zhongyuan Wang; Tiejun Huang; Shanghang Zhang
>
> **备注:** 37 pages, 13 figures, Technical Report
>
> **摘要:** We introduce RoboBrain 2.5, a next-generation embodied AI foundation model that advances general perception, spatial reasoning, and temporal modeling through extensive training on high-quality spatiotemporal supervision. Building upon its predecessor, RoboBrain 2.5 introduces two major capability upgrades. Specifically, it unlocks Precise 3D Spatial Reasoning by shifting from 2D pixel-relative grounding to depth-aware coordinate prediction and absolute metric constraint comprehension, generating complete 3D manipulation traces as ordered keypoint sequences under physical constraints. Complementing this spatial precision, the model establishes Dense Temporal Value Estimation that provides dense, step-aware progress prediction and execution state understanding across varying viewpoints, producing stable feedback signals for downstream learning. Together, these upgrades extend the framework toward more physically grounded and execution-aware embodied intelligence for complex, fine-grained manipulation. The code and checkpoints are available at project website: https://superrobobrain.github.io
>
---
#### [new 017] TacUMI: A Multi-Modal Universal Manipulation Interface for Contact-Rich Tasks
- **分类: cs.RO**

- **简介: 该论文提出TacUMI系统，用于接触丰富的操作任务，解决多模态数据收集与分割问题。通过集成多种传感器，提升任务分解准确性。**

- **链接: [https://arxiv.org/pdf/2601.14550v1](https://arxiv.org/pdf/2601.14550v1)**

> **作者:** Tailai Cheng; Kejia Chen; Lingyun Chen; Liding Zhang; Yue Zhang; Yao Ling; Mahdi Hamad; Zhenshan Bing; Fan Wu; Karan Sharma; Alois Knoll
>
> **摘要:** Task decomposition is critical for understanding and learning complex long-horizon manipulation tasks. Especially for tasks involving rich physical interactions, relying solely on visual observations and robot proprioceptive information often fails to reveal the underlying event transitions. This raises the requirement for efficient collection of high-quality multi-modal data as well as robust segmentation method to decompose demonstrations into meaningful modules. Building on the idea of the handheld demonstration device Universal Manipulation Interface (UMI), we introduce TacUMI, a multi-modal data collection system that integrates additionally ViTac sensors, force-torque sensor, and pose tracker into a compact, robot-compatible gripper design, which enables synchronized acquisition of all these modalities during human demonstrations. We then propose a multi-modal segmentation framework that leverages temporal models to detect semantically meaningful event boundaries in sequential manipulations. Evaluation on a challenging cable mounting task shows more than 90 percent segmentation accuracy and highlights a remarkable improvement with more modalities, which validates that TacUMI establishes a practical foundation for both scalable collection and segmentation of multi-modal demonstrations in contact-rich tasks.
>
---
#### [new 018] CADGrasp: Learning Contact and Collision Aware General Dexterous Grasping in Cluttered Scenes
- **分类: cs.RO**

- **简介: 该论文属于机械臂抓取任务，旨在解决杂乱场景中灵巧抓取的碰撞与接触问题。提出CADGrasp算法，通过两阶段优化生成稳定抓取姿态。**

- **链接: [https://arxiv.org/pdf/2601.15039v1](https://arxiv.org/pdf/2601.15039v1)**

> **作者:** Jiyao Zhang; Zhiyuan Ma; Tianhao Wu; Zeyuan Chen; Hao Dong
>
> **摘要:** Dexterous grasping in cluttered environments presents substantial challenges due to the high degrees of freedom of dexterous hands, occlusion, and potential collisions arising from diverse object geometries and complex layouts. To address these challenges, we propose CADGrasp, a two-stage algorithm for general dexterous grasping using single-view point cloud inputs. In the first stage, we predict sparse IBS, a scene-decoupled, contact- and collision-aware representation, as the optimization target. Sparse IBS compactly encodes the geometric and contact relationships between the dexterous hand and the scene, enabling stable and collision-free dexterous grasp pose optimization. To enhance the prediction of this high-dimensional representation, we introduce an occupancy-diffusion model with voxel-level conditional guidance and force closure score filtering. In the second stage, we develop several energy functions and ranking strategies for optimization based on sparse IBS to generate high-quality dexterous grasp poses. Extensive experiments in both simulated and real-world settings validate the effectiveness of our approach, demonstrating its capability to mitigate collisions while maintaining a high grasp success rate across diverse objects and complex scenes.
>
---
#### [new 019] Moving Beyond Compliance in Soft-Robotic Catheters Through Modularity for Precision Therapies
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，旨在解决内窥镜手术中工具功能不足与反馈缺失的问题。研究提出一种模块化软性导管，实现精准治疗与实时感知。**

- **链接: [https://arxiv.org/pdf/2601.14837v1](https://arxiv.org/pdf/2601.14837v1)**

> **作者:** B. Calmé; N. J. Greenidge; A. Metcalf; A. Bacchetti; G. Loza; D. Kpeglo; P. Lloyd; V. Pensabene; J. H. Chandler; P. Valdastri
>
> **备注:** 31 pages, 6 figures, 7 supplementary figures
>
> **摘要:** Soft robotic instruments could navigate delicate, tortuous anatomy more safely than rigid tools, but clinical adoption is limited by insufficient tip functionalization and real-time feedback at the tissue interface. Few sensing and therapeutic modules are compact, robust, and adaptable enough to measure, and respond to, subtle physiological cues during intraluminal procedures. We present a 1.47 mm diameter modular soft robotic catheter that integrates sensing, actuation, and therapy while retaining the compliance needed for safe endoluminal navigation. Validated across multiple in vivo settings, we emphasize its utility in endoscopic retrograde cholangiopancreatography (ERCP), a highly technical procedure and a key access route to the pancreas, an organ that is fragile, difficult to instrument, and central to diseases such as pancreatic cancer. Our architecture supports up to four independently controlled functional units, allowing customizable combinations of anchoring, manipulation, sensing, and targeted drug delivery. In a live porcine model, we demonstrate semi-autonomous deployment into the pancreatic duct and 7.5 cm of endoscopic navigation within it, a region currently inaccessible with standard catheters. A closed-loop autonomous/shared-control system that combines a learned model, magnetic actuation, onboard shape sensing, and visual marker tracking further improves cannulation accuracy. Together, these results establish a scalable platform for multifunctional soft robotic catheters and a new paradigm for complex endoluminal interventions, with potential to reduce radiation exposure, shorten training, and accelerate clinical translation of soft robotic technologies.
>
---
#### [new 020] Landing-Induced Viscoelastic Changes in an Anthropomimetic Foot Joint Structure are Modulated by Foot Structure and Posture
- **分类: cs.RO; physics.bio-ph**

- **简介: 该论文属于生物力学研究，旨在探究足部结构对落地冲击响应的影响。通过构建仿生足关节模型，分析骨骼结构和姿势如何调节粘弹性特性，以理解人体足弓在冲击吸收中的作用。**

- **链接: [https://arxiv.org/pdf/2601.14634v1](https://arxiv.org/pdf/2601.14634v1)**

> **作者:** Satoru Hashimoto; Yinlai Jiang; Hiroshi Yokoi; Shunta Togo
>
> **备注:** 27 pages, preprint
>
> **摘要:** Cadaveric studies have provided important insights into the mechanics of the human foot arch and plantar fascia. However, repeatedly probing posture-dependent viscoelastic responses immediately after landing impact is difficult in biological specimens, leaving the contribution of skeletal architecture to landing dynamics incompletely understood. In this study, we developed an anthropomimetic foot joint structure aimed at replicating the skeletal geometry of the human foot. Using a vertical drop apparatus that simulates landing and a viscoelastic system-identification model, we investigated how skeletal structure and posture modulate the apparent post-impact viscoelastic response. The results show that the multi-jointed anthropomimetic structure exhibited a higher damping ratio than simplified flat and rigid feet. Moreover, ankle dorsiflexion and toe extension systematically shifted the identified parameters, reducing the damping ratio under the tested conditions. Taken together, these findings indicate that an arch-like, multi-jointed skeletal architecture can enhance impact attenuation in an anthropomimetic mechanical foot, and that morphology and passive posture alone can tune the trade-off between attenuation and rebound. The observed posture-dependent trends are qualitatively consistent with reported differences in human landing strategies, suggesting that skeletal architecture may partly account for the modulation. Furthermore, these results highlight the engineering advantage of anatomically informed skeletal replication for achieving human-like apparent viscoelastic behavior through postural adjustment during landing.
>
---
#### [new 021] Stochastic Decision-Making Framework for Human-Robot Collaboration in Industrial Applications
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决工业场景中机器人如何根据人类情绪和行为进行决策的问题。通过构建随机决策框架，提升协作的安全性与效率。**

- **链接: [https://arxiv.org/pdf/2601.14809v1](https://arxiv.org/pdf/2601.14809v1)**

> **作者:** Muhammad Adel Yusuf; Ali Nasir; Zeeshan Hameed Khan
>
> **备注:** Under Review by IEEE Transactions on Human Machine Systems
>
> **摘要:** Collaborative robots, or cobots, are increasingly integrated into various industrial and service settings to work efficiently and safely alongside humans. However, for effective human-robot collaboration, robots must reason based on human factors such as motivation level and aggression level. This paper proposes an approach for decision-making in human-robot collaborative (HRC) environments utilizing stochastic modeling. By leveraging probabilistic models and control strategies, the proposed method aims to anticipate human actions and emotions, enabling cobots to adapt their behavior accordingly. So far, most of the research has been done to detect the intentions of human co-workers. This paper discusses the theoretical framework, implementation strategies, simulation results, and potential applications of the bilateral collaboration approach for safety and efficiency in collaborative robotics.
>
---
#### [new 022] UniCon: A Unified System for Efficient Robot Learning Transfers
- **分类: cs.RO; cs.SE**

- **简介: 该论文提出UniCon系统，解决异构机器人学习控制部署难题。通过标准化流程和高效数据流，实现跨平台、低延迟的控制器部署与sim-to-real迁移。**

- **链接: [https://arxiv.org/pdf/2601.14617v1](https://arxiv.org/pdf/2601.14617v1)**

> **作者:** Yunfeng Lin; Li Xu; Yong Yu; Jiangmiao Pang; Weinan Zhang
>
> **备注:** in submission, under review
>
> **摘要:** Deploying learning-based controllers across heterogeneous robots is challenging due to platform differences, inconsistent interfaces, and inefficient middleware. To address these issues, we present UniCon, a lightweight framework that standardizes states, control flow, and instrumentation across platforms. It decomposes workflows into execution graphs with reusable components, separating system states from control logic to enable plug-and-play deployment across various robot morphologies. Unlike traditional middleware, it prioritizes efficiency through batched, vectorized data flow, minimizing communication overhead and improving inference latency. This modular, data-oriented approach enables seamless sim-to-real transfer with minimal re-engineering. We demonstrate that UniCon reduces code redundancy when transferring workflows and achieves higher inference efficiency compared to ROS-based systems. Deployed on over 12 robot models from 7 manufacturers, it has been successfully integrated into ongoing research projects, proving its effectiveness in real-world scenarios.
>
---
#### [new 023] HumanoidVLM: Vision-Language-Guided Impedance Control for Contact-Rich Humanoid Manipulation
- **分类: cs.RO**

- **简介: 该论文提出HumanoidVLM，解决人形机器人在复杂接触任务中自适应控制的问题，通过视觉语言引导检索，实现参数和抓取配置的自动选择。**

- **链接: [https://arxiv.org/pdf/2601.14874v1](https://arxiv.org/pdf/2601.14874v1)**

> **作者:** Yara Mahmoud; Yasheerah Yaqoot; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted for publication at LBR of HRI 2026 conference
>
> **摘要:** Humanoid robots must adapt their contact behavior to diverse objects and tasks, yet most controllers rely on fixed, hand-tuned impedance gains and gripper settings. This paper introduces HumanoidVLM, a vision-language driven retrieval framework that enables the Unitree G1 humanoid to select task-appropriate Cartesian impedance parameters and gripper configurations directly from an egocentric RGB image. The system couples a vision-language model for semantic task inference with a FAISS-based Retrieval-Augmented Generation (RAG) module that retrieves experimentally validated stiffness-damping pairs and object-specific grasp angles from two custom databases, and executes them through a task-space impedance controller for compliant manipulation. We evaluate HumanoidVLM on 14 visual scenarios and achieve a retrieval accuracy of 93%. Real-world experiments show stable interaction dynamics, with z-axis tracking errors typically within 1-3.5 cm and virtual forces consistent with task-dependent impedance settings. These results demonstrate the feasibility of linking semantic perception with retrieval-based control as an interpretable path toward adaptive humanoid manipulation.
>
---
#### [new 024] TIDAL: Temporally Interleaved Diffusion and Action Loop for High-Frequency VLA Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出TIDAL框架，解决VLA模型高延迟导致的动态环境执行失败问题。通过双频架构分离语义与控制，提升控制频率并增强实时响应能力。**

- **链接: [https://arxiv.org/pdf/2601.14945v1](https://arxiv.org/pdf/2601.14945v1)**

> **作者:** Yuteng Sun; Haoran Wang; Ruofei Bai; Zhengguo Li; Jun Li; Meng Yee; Chuah; Wei Yun Yau
>
> **摘要:** Large-scale Vision-Language-Action (VLA) models offer semantic generalization but suffer from high inference latency, limiting them to low-frequency batch-and-execute paradigm. This frequency mismatch creates an execution blind spot, causing failures in dynamic environments where targets move during the open-loop execution window. We propose TIDAL (Temporally Interleaved Diffusion and Action Loop), a hierarchical framework that decouples semantic reasoning from high-frequency actuation. TIDAL operates as a backbone-agnostic module for diffusion-based VLAs, using a dual-frequency architecture to redistribute the computational budget. Specifically, a low-frequency macro-intent loop caches semantic embeddings, while a high-frequency micro-control loop interleaves single-step flow integration with execution. This design enables approximately 9 Hz control updates on edge hardware (vs. approximately 2.4 Hz baselines) without increasing marginal overhead. To handle the resulting latency shift, we introduce a temporally misaligned training strategy where the policy learns predictive compensation using stale semantic intent alongside real-time proprioception. Additionally, we address the insensitivity of static vision encoders to velocity by incorporating a differential motion predictor. TIDAL is architectural, making it orthogonal to system-level optimizations. Experiments show a 2x performance gain over open-loop baselines in dynamic interception tasks. Despite a marginal regression in static success rates, our approach yields a 4x increase in feedback frequency and extends the effective horizon of semantic embeddings beyond the native action chunk size. Under non-paused inference protocols, TIDAL remains robust where standard baselines fail due to latency.
>
---
#### [new 025] Systematic Evaluation of Hip Exoskeleton Assistance Parameters for Enhancing Gait Stability During Ground Slip Perturbations
- **分类: cs.RO**

- **简介: 该论文属于辅助设备控制任务，旨在提升行走稳定性以减少老年人跌倒风险。研究通过调整髋外骨骼的助力参数，评估其对步态稳定性的效果。**

- **链接: [https://arxiv.org/pdf/2601.15056v1](https://arxiv.org/pdf/2601.15056v1)**

> **作者:** Maria T. Tagliaferri; Inseung Kang
>
> **摘要:** Falls are the leading cause of injury related hospitalization and mortality among older adults. Consequently, mitigating age-related declines in gait stability and reducing fall risk during walking is a critical goal for assistive devices. Lower-limb exoskeletons have the potential to support users in maintaining stability during walking. However, most exoskeleton controllers are optimized to reduce the energetic cost of walking rather than to improve stability. While some studies report stability benefits with assistance, the effects of specific parameters, such as assistance magnitude and duration, remain unexplored. To address this gap, we systematically modulated the magnitude and duration of torque provided by a bilateral hip exoskeleton during slip perturbations in eight healthy adults, quantifying stability using whole-body angular momentum (WBAM). WBAM responses were governed by a significant interaction between assistance magnitude and duration, with duration determining whether exoskeleton assistance was stabilizing or destabilizing relative to not wearing the exoskeleton device. Compared to an existing energy-optimized controller, experimentally identified stability-optimal parameters reduced WBAM range by 25.7% on average. Notably, substantial inter-subject variability was observed in the parameter combinations that minimized WBAM during perturbations. We found that optimizing exoskeleton assistance for energetic outcomes alone is insufficient for improving reactive stability during gait perturbations. Stability-focused exoskeleton control should prioritize temporal assistance parameters and include user-specific personalization. This study represents an important step toward personalized, stability-focused exoskeleton control, with direct implications for improving stability and reducing fall risk in older adults.
>
---
#### [new 026] HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出HumanDiffusion，用于搜救无人机的人类感知轨迹规划任务。解决在动态环境中自主导航、接近目标人类的问题，通过图像条件扩散模型生成安全轨迹。**

- **链接: [https://arxiv.org/pdf/2601.14973v1](https://arxiv.org/pdf/2601.14973v1)**

> **作者:** Faryal Batool; Iana Zhura; Valerii Serpiva; Roohan Ahmed Khan; Ivan Valuev; Issatay Tokmurziyev; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted at HRI, Late Breaking Report, 2026
>
> **摘要:** Reliable human--robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image-conditioned diffusion planner that generates human-aware navigation trajectories directly from RGB imagery. The system combines YOLO-11--based human detection with diffusion-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real-world indoor mock-disaster scenarios. On a 300-sample test set, the model achieves a mean squared error of 0.02 in pixel-space trajectory reconstruction. Real-world experiments demonstrate an overall mission success rate of 80% across accident-response and search-and-locate tasks with partial occlusions. These results indicate that human-conditioned diffusion planning offers a practical and robust solution for human-aware UAV navigation in time-critical assistance settings.
>
---
#### [new 027] V-CAGE: Context-Aware Generation and Verification for Scalable Long-Horizon Embodied Tasks
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出V-CAGE框架，解决长时序具身任务中场景物理不真实、语义不符等问题，通过上下文感知生成和验证机制，提升数据集的物理与语义一致性。**

- **链接: [https://arxiv.org/pdf/2601.15164v1](https://arxiv.org/pdf/2601.15164v1)**

> **作者:** Yaru Liu; Ao-bo Wang; Nanyang Ye
>
> **摘要:** Learning long-horizon embodied behaviors from synthetic data remains challenging because generated scenes are often physically implausible, language-driven programs frequently "succeed" without satisfying task semantics, and high-level instructions require grounding into executable action sequences. To address these limitations, we introduce V-CAGE, a closed-loop framework for generating robust, semantically aligned manipulation datasets at scale. First, we propose a context-aware instantiation mechanism that enforces geometric consistency during scene synthesis. By dynamically maintaining a map of prohibited spatial areas as objects are placed, our system prevents interpenetration and ensures reachable, conflict-free configurations in cluttered environments. Second, to bridge the gap between abstract intent and low-level control, we employ a hierarchical instruction decomposition module. This decomposes high-level goals (e.g., "get ready for work") into compositional action primitives, facilitating coherent long-horizon planning. Crucially, we enforce semantic correctness through a VLM-based verification loop. Acting as a visual critic, the VLM performs rigorous rejection sampling after each subtask, filtering out "silent failures" where code executes but fails to achieve the visual goal. Experiments demonstrate that V-CAGE yields datasets with superior physical and semantic fidelity, significantly boosting the success rate and generalization of downstream policies compared to non-verified baselines.
>
---
#### [new 028] Rethinking Video Generation Model for the Embodied World
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文聚焦视频生成任务，旨在提升机器人视频的物理真实性。针对数据不足和评估缺失的问题，提出RBench基准和RoVid-X数据集，推动 embodied AI 发展。**

- **链接: [https://arxiv.org/pdf/2601.15282v1](https://arxiv.org/pdf/2601.15282v1)**

> **作者:** Yufan Deng; Zilin Pan; Hongyu Zhang; Xiaojie Li; Ruoqing Hu; Yufei Ding; Yiming Zou; Yan Zeng; Daquan Zhou
>
> **备注:** Github: https://github.com/DAGroup-PKU/ReVidgen/ Project website: https://dagroup-pku.github.io/ReVidgen.github.io/
>
> **摘要:** Video generation models have significantly advanced embodied intelligence, unlocking new possibilities for generating diverse robot data that capture perception, reasoning, and action in the physical world. However, synthesizing high-quality videos that accurately reflect real-world robotic interactions remains challenging, and the lack of a standardized benchmark limits fair comparisons and progress. To address this gap, we introduce a comprehensive robotics benchmark, RBench, designed to evaluate robot-oriented video generation across five task domains and four distinct embodiments. It assesses both task-level correctness and visual fidelity through reproducible sub-metrics, including structural consistency, physical plausibility, and action completeness. Evaluation of 25 representative models highlights significant deficiencies in generating physically realistic robot behaviors. Furthermore, the benchmark achieves a Spearman correlation coefficient of 0.96 with human evaluations, validating its effectiveness. While RBench provides the necessary lens to identify these deficiencies, achieving physical realism requires moving beyond evaluation to address the critical shortage of high-quality training data. Driven by these insights, we introduce a refined four-stage data pipeline, resulting in RoVid-X, the largest open-source robotic dataset for video generation with 4 million annotated video clips, covering thousands of tasks and enriched with comprehensive physical property annotations. Collectively, this synergistic ecosystem of evaluation and data establishes a robust foundation for rigorous assessment and scalable training of video models, accelerating the evolution of embodied AI toward general intelligence.
>
---
#### [new 029] Iterative Refinement Improves Compositional Image Generation
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于文本到图像生成任务，旨在解决复杂提示下的生成问题。通过迭代精炼策略，提升生成图像的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2601.15286v1](https://arxiv.org/pdf/2601.15286v1)**

> **作者:** Shantanu Jaiswal; Mihir Prabhudesai; Nikash Bhardwaj; Zheyang Qin; Amir Zadeh; Chuan Li; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project webpage: https://iterative-img-gen.github.io/
>
> **摘要:** Text-to-image (T2I) models have achieved remarkable progress, yet they continue to struggle with complex prompts that require simultaneously handling multiple objects, relations, and attributes. Existing inference-time strategies, such as parallel sampling with verifiers or simply increasing denoising steps, can improve prompt alignment but remain inadequate for richly compositional settings where many constraints must be satisfied. Inspired by the success of chain-of-thought reasoning in large language models, we propose an iterative test-time strategy in which a T2I model progressively refines its generations across multiple steps, guided by feedback from a vision-language model as the critic in the loop. Our approach is simple, requires no external tools or priors, and can be flexibly applied to a wide range of image generators and vision-language models. Empirically, we demonstrate consistent gains on image generation across benchmarks: a 16.9% improvement in all-correct rate on ConceptMix (k=7), a 13.8% improvement on T2I-CompBench (3D-Spatial category) and a 12.5% improvement on Visual Jenga scene decomposition compared to compute-matched parallel sampling. Beyond quantitative gains, iterative refinement produces more faithful generations by decomposing complex prompts into sequential corrections, with human evaluators preferring our method 58.7% of the time over 41.3% for the parallel baseline. Together, these findings highlight iterative self-correction as a broadly applicable principle for compositional image generation. Results and visualizations are available at https://iterative-img-gen.github.io/
>
---
#### [new 030] From Observation to Prediction: LSTM for Vehicle Lane Change Forecasting on Highway On/Off-Ramps
- **分类: cs.LG; cs.AI; cs.NE; cs.RO**

- **简介: 该论文属于车辆变道预测任务，旨在提升高速公路出入口区域的行车安全。通过LSTM模型分析数据，解决变道行为预测问题。**

- **链接: [https://arxiv.org/pdf/2601.14848v1](https://arxiv.org/pdf/2601.14848v1)**

> **作者:** Mohamed Abouras; Catherine M. Elias
>
> **摘要:** On and off-ramps are understudied road sections even though they introduce a higher level of variation in highway interactions. Predicting vehicles' behavior in these areas can decrease the impact of uncertainty and increase road safety. In this paper, the difference between this Area of Interest (AoI) and a straight highway section is studied. Multi-layered LSTM architecture to train the AoI model with ExiD drone dataset is utilized. In the process, different prediction horizons and different models' workflow are tested. The results show great promise on horizons up to 4 seconds with prediction accuracy starting from about 76% for the AoI and 94% for the general highway scenarios on the maximum horizon.
>
---
#### [new 031] SilentDrift: Exploiting Action Chunking for Stealthy Backdoor Attacks on Vision-Language-Action Models
- **分类: cs.CR; cs.AI; cs.RO**

- **简介: 该论文属于安全领域，针对VLA模型的后门攻击问题，提出SILENTDRIFT方法，利用动作分块漏洞实现隐蔽攻击，提升攻击成功率并保持任务性能。**

- **链接: [https://arxiv.org/pdf/2601.14323v1](https://arxiv.org/pdf/2601.14323v1)**

> **作者:** Bingxin Xu; Yuzhang Shang; Binghui Wang; Emilio Ferrara
>
> **摘要:** Vision-Language-Action (VLA) models are increasingly deployed in safety-critical robotic applications, yet their security vulnerabilities remain underexplored. We identify a fundamental security flaw in modern VLA systems: the combination of action chunking and delta pose representations creates an intra-chunk visual open-loop. This mechanism forces the robot to execute K-step action sequences, allowing per-step perturbations to accumulate through integration. We propose SILENTDRIFT, a stealthy black-box backdoor attack exploiting this vulnerability. Our method employs the Smootherstep function to construct perturbations with guaranteed C2 continuity, ensuring zero velocity and acceleration at trajectory boundaries to satisfy strict kinematic consistency constraints. Furthermore, our keyframe attack strategy selectively poisons only the critical approach phase, maximizing impact while minimizing trigger exposure. The resulting poisoned trajectories are visually indistinguishable from successful demonstrations. Evaluated on the LIBERO, SILENTDRIFT achieves a 93.2% Attack Success Rate with a poisoning rate under 2%, while maintaining a 95.3% Clean Task Success Rate.
>
---
#### [new 032] Explainable OOHRI: Communicating Robot Capabilities and Limitations as Augmented Reality Affordances
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人透明度不足的问题。通过AR界面展示机器人能力与限制，提升用户理解与协作。**

- **链接: [https://arxiv.org/pdf/2601.14587v1](https://arxiv.org/pdf/2601.14587v1)**

> **作者:** Lauren W. Wang; Mohamed Kari; Parastoo Abtahi
>
> **摘要:** Human interaction is essential for issuing personalized instructions and assisting robots when failure is likely. However, robots remain largely black boxes, offering users little insight into their evolving capabilities and limitations. To address this gap, we present explainable object-oriented HRI (X-OOHRI), an augmented reality (AR) interface that conveys robot action possibilities and constraints through visual signifiers, radial menus, color coding, and explanation tags. Our system encodes object properties and robot limits into object-oriented structures using a vision-language model, allowing explanation generation on the fly and direct manipulation of virtual twins spatially aligned within a simulated environment. We integrate the end-to-end pipeline with a physical robot and showcase diverse use cases ranging from low-level pick-and-place to high-level instructions. Finally, we evaluate X-OOHRI through a user study and find that participants effectively issue object-oriented commands, develop accurate mental models of robot limitations, and engage in mixed-initiative resolution.
>
---
#### [new 033] FlowSSC: Universal Generative Monocular Semantic Scene Completion via One-Step Latent Diffusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于语义场景补全任务，解决单目图像中遮挡区域的3D语义生成问题。提出FlowSSC框架，通过单步扩散实现高效高质生成。**

- **链接: [https://arxiv.org/pdf/2601.15250v1](https://arxiv.org/pdf/2601.15250v1)**

> **作者:** Zichen Xi; Hao-Xiang Chen; Nan Xue; Hongyu Yan; Qi-Yuan Feng; Levent Burak Kara; Joaquim Jorge; Qun-Ce Xu
>
> **备注:** Under Review
>
> **摘要:** Semantic Scene Completion (SSC) from monocular RGB images is a fundamental yet challenging task due to the inherent ambiguity of inferring occluded 3D geometry from a single view. While feed-forward methods have made progress, they often struggle to generate plausible details in occluded regions and preserve the fundamental spatial relationships of objects. Such accurate generative reasoning capability for the entire 3D space is critical in real-world applications. In this paper, we present FlowSSC, the first generative framework applied directly to monocular semantic scene completion. FlowSSC treats the SSC task as a conditional generation problem and can seamlessly integrate with existing feed-forward SSC methods to significantly boost their performance. To achieve real-time inference without compromising quality, we introduce Shortcut Flow-matching that operates in a compact triplane latent space. Unlike standard diffusion models that require hundreds of steps, our method utilizes a shortcut mechanism to achieve high-fidelity generation in a single step, enabling practical deployment in autonomous systems. Extensive experiments on SemanticKITTI demonstrate that FlowSSC achieves state-of-the-art performance, significantly outperforming existing baselines.
>
---
#### [new 034] AutoDriDM: An Explainable Benchmark for Decision-Making of Vision-Language Models in Autonomous Driving
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于自主驾驶中的决策任务，旨在解决现有基准对感知与决策评估不均衡的问题。提出AutoDriDM基准，评估视觉语言模型的决策能力，并分析其推理过程。**

- **链接: [https://arxiv.org/pdf/2601.14702v1](https://arxiv.org/pdf/2601.14702v1)**

> **作者:** Zecong Tang; Zixu Wang; Yifei Wang; Weitong Lian; Tianjian Gao; Haoran Li; Tengju Ru; Lingyi Meng; Zhejun Cui; Yichen Zhu; Qi Kang; Kaixuan Wang; Yu Zhang
>
> **备注:** 23 pages. Submitted to ACL ARR 2026 January
>
> **摘要:** Autonomous driving is a highly challenging domain that requires reliable perception and safe decision-making in complex scenarios. Recent vision-language models (VLMs) demonstrate reasoning and generalization abilities, opening new possibilities for autonomous driving; however, existing benchmarks and metrics overemphasize perceptual competence and fail to adequately assess decision-making processes. In this work, we present AutoDriDM, a decision-centric, progressive benchmark with 6,650 questions across three dimensions - Object, Scene, and Decision. We evaluate mainstream VLMs to delineate the perception-to-decision capability boundary in autonomous driving, and our correlation analysis reveals weak alignment between perception and decision-making performance. We further conduct explainability analyses of models' reasoning processes, identifying key failure modes such as logical reasoning errors, and introduce an analyzer model to automate large-scale annotation. AutoDriDM bridges the gap between perception-centered and decision-centered evaluation, providing guidance toward safer and more reliable VLMs for real-world autonomous driving.
>
---
#### [new 035] Implementing Knowledge Representation and Reasoning with Object Oriented Design
- **分类: cs.AI; cs.RO; cs.SE**

- **简介: 该论文属于知识表示与推理任务，旨在解决OOP与KR&R系统集成困难的问题。通过KRROOD框架，将知识作为第一类抽象，实现两者融合，并验证其性能。**

- **链接: [https://arxiv.org/pdf/2601.14840v1](https://arxiv.org/pdf/2601.14840v1)**

> **作者:** Abdelrhman Bassiouny; Tom Schierenbeck; Sorin Arion; Benjamin Alt; Naren Vasantakumaar; Giang Nguyen; Michael Beetz
>
> **备注:** 9 pages, 2 figures, submitted to the 2026 International Joint Conference on Artificial Intelligence (IJCAI)
>
> **摘要:** This paper introduces KRROOD, a framework designed to bridge the integration gap between modern software engineering and Knowledge Representation & Reasoning (KR&R) systems. While Object-Oriented Programming (OOP) is the standard for developing complex applications, existing KR&R frameworks often rely on external ontologies and specialized languages that are difficult to integrate with imperative code. KRROOD addresses this by treating knowledge as a first-class programming abstraction using native class structures, bridging the gap between the logic programming and OOP paradigms. We evaluate the system on the OWL2Bench benchmark and a human-robot task learning scenario. Experimental results show that KRROOD achieves strong performance while supporting the expressive reasoning required for real-world autonomous systems.
>
---
#### [new 036] BayesianVLA: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在新指令和复杂任务中泛化能力差的问题。通过引入贝叶斯分解和潜在动作查询，提升语言引导的行动策略。**

- **链接: [https://arxiv.org/pdf/2601.15197v1](https://arxiv.org/pdf/2601.15197v1)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose BayesianVLA, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, BayesianVLA significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
## 更新

#### [replaced 001] OSMa-Bench: Evaluating Open Semantic Mapping Under Varying Lighting Conditions
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于机器人感知任务，旨在评估不同光照条件下开放语义映射的性能。提出OSMa-Bench框架，通过新数据集和场景图方法分析模型的语义准确性和结构理解能力。**

- **链接: [https://arxiv.org/pdf/2503.10331v3](https://arxiv.org/pdf/2503.10331v3)**

> **作者:** Maxim Popov; Regina Kurkova; Mikhail Iumanov; Jaafar Mahmoud; Sergey Kolyubin
>
> **备注:** Project page: https://be2rlab.github.io/OSMa-Bench/
>
> **摘要:** Open Semantic Mapping (OSM) is a key technology in robotic perception, combining semantic segmentation and SLAM techniques. This paper introduces a dynamically configurable and highly automated LLM/LVLM-powered pipeline for evaluating OSM solutions called OSMa-Bench (Open Semantic Mapping Benchmark). The study focuses on evaluating state-of-the-art semantic mapping algorithms under varying indoor lighting conditions, a critical challenge in indoor environments. We introduce a novel dataset with simulated RGB-D sequences and ground truth 3D reconstructions, facilitating the rigorous analysis of mapping performance across different lighting conditions. Through experiments on leading models such as ConceptGraphs, BBQ, and OpenScene, we evaluate the semantic fidelity of object recognition and segmentation. Additionally, we introduce a Scene Graph evaluation method to analyze the ability of models to interpret semantic structure. The results provide insights into the robustness of these models, forming future research directions for developing resilient and adaptable robotic systems. Project page is available at https://be2rlab.github.io/OSMa-Bench/.
>
---
#### [replaced 002] Semilinear single-track vehicle models with distributed tyre friction dynamics
- **分类: cs.RO**

- **简介: 该论文属于车辆动力学建模任务，旨在解决轮胎瞬态行为对车辆横向运动影响的问题。提出一种基于分布摩擦与橡胶动力学的半线性模型，实现更精确的动态描述。**

- **链接: [https://arxiv.org/pdf/2601.06854v2](https://arxiv.org/pdf/2601.06854v2)**

> **作者:** Luigi Romano; Ole Morten Aamo; Jan Åslund; Erik Frisk
>
> **备注:** 37 pages, 12 figures
>
> **摘要:** This paper introduces a novel family of single-track vehicle models that incorporate a distributed representation of transient tyre dynamics, whilst simultaneously accounting for nonlinear effects induced by friction. The core of the proposed framework is represented by the distributed Friction with Bristle Dynamics (FrBD) model, which unifies and extends classical formulations such as Dahl and LuGre by describing the rolling contact process as a spatially distributed system governed by semilinear partial differential equations (PDEs). This model is systematically integrated into a single-track vehicle framework, where the resulting semilinear ODE-PDE interconnection captures the interaction between lateral vehicle motion and tyre deformation. Two main variants are considered: one with rigid tyre carcass and another with flexible carcass, each admitting a compact state-space representation. Local and global well-posedness properties for the coupled system are established rigorously, highlighting the dissipative and physically consistent properties of the distributed FrBD model. A linearisation procedure is also presented, enabling spectral analysis and transfer function derivation, and potentially facilitating the synthesis of controllers and observers. Numerical simulations demonstrate the model's capability to capture micro-shimmy oscillations and transient lateral responses to advanced steering manoeuvres. The proposed formulation advances the state-of-the-art in vehicle dynamics modelling by providing a physically grounded, mathematically rigorous, and computationally tractable approach to incorporating transient tyre behaviour in lateral vehicle dynamics, when accounting for the effect of limited friction.
>
---
#### [replaced 003] Locomotion Dynamics of an Underactuated Three-Link Robotic Vehicle
- **分类: cs.RO**

- **简介: 该论文研究轮式三连杆蛇形机器人的运动动力学，解决非完整约束下滑动与摩擦的影响问题。通过实验分析并建立包含滑动和摩擦的动态模型，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2407.21540v2](https://arxiv.org/pdf/2407.21540v2)**

> **作者:** Leonid Raz; Yizhar Or
>
> **备注:** Accepted to IEEE Transactions on Robotics, January 2026
>
> **摘要:** The wheeled three-link snake robot is a well-known example of an underactuated system modelled using nonholonomic constraints, preventing lateral slippage (skid) of the wheels. A kinematically controlled configuration assumes that both joint angles are directly prescribed as phase-shifted periodic input. In another configuration of the robot, only one joint is periodically actuated while the second joint is passively governed by a visco-elastic torsion spring. In our work, we constructed the two configurations of the wheeled robot and conducted motion experiments under different actuation inputs. Analysis of the motion tracking measurements reveals a significant amount of wheels' skid, in contrast to the assumptions used in standard nonholonomic models. Therefore, we propose modified dynamic models which include wheels' skid and viscous friction forces, as well as rolling resistance. After parameter fitting, these dynamic models reach good agreement with the motion measurements, including effects of input's frequency on the mean speed and net displacement per period. This illustrates the importance of incorporating wheels' skid and friction into the system's model.
>
---
#### [replaced 004] GuideTouch: An Obstacle Avoidance Device with Tactile Feedback for Visually Impaired
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于无障碍导航任务，旨在解决视觉障碍者检测头部障碍物的问题。提出GuideTouch设备，通过三维感知和触觉反馈实现自主避障。**

- **链接: [https://arxiv.org/pdf/2601.13813v2](https://arxiv.org/pdf/2601.13813v2)**

> **作者:** Timofei Kozlov; Artem Trandofilov; Georgii Gazaryan; Issatay Tokmurziyev; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted for publication at LBR of HRI 2026 conference
>
> **摘要:** Safe navigation for the visually impaired individuals remains a critical challenge, especially concerning head-level obstacles, which traditional mobility aids often fail to detect. We introduce GuideTouch, a compact, affordable, standalone wearable device designed for autonomous obstacle avoidance. The system integrates two vertically aligned Time-of-Flight (ToF) sensors, enabling three-dimensional environmental perception, and four vibrotactile actuators that provide directional haptic feedback. Proximity and direction information is communicated via an intuitive 4-point vibrotactile feedback system located across the user's shoulders and upper chest. For real-world robustness, the device includes a unique centrifugal self-cleaning optical cover mechanism and a sound alarm system for location if the device is dropped. We evaluated the haptic perception accuracy across 22 participants (17 male and 5 female, aged 21-48, mean 25.7, sd 6.1). Statistical analysis confirmed a significant difference between the perception accuracy of different patterns. The system demonstrated high recognition accuracy, achieving an average of 92.9% for single and double motor (primary directional) patterns. Furthermore, preliminary experiments with 14 visually impaired users validated this interface, showing a recognition accuracy of 93.75% for primary directional cues. The results demonstrate that GuideTouch enables intuitive spatial perception and could significantly improve the safety, confidence, and autonomy of users with visual impairments during independent navigation.
>
---
#### [replaced 005] Collision Probability Estimation for Optimization-based Vehicular Motion Planning
- **分类: cs.RO; math.OC**

- **简介: 该论文属于自动驾驶中的运动规划任务，旨在解决碰撞概率估计问题。通过优化方法提高计算效率与确定性，确保路径规划的安全性和可行性。**

- **链接: [https://arxiv.org/pdf/2505.21161v3](https://arxiv.org/pdf/2505.21161v3)**

> **作者:** Leon Tolksdorf; Arturo Tejada; Christian Birkner; Nathan van de Wouw
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Many motion planning algorithms for automated driving require estimating the probability of collision (POC) to account for uncertainties in the measurement and estimation of the motion of road users. Common POC estimation techniques often utilize sampling-based methods that suffer from computational inefficiency and a non-deterministic estimation, i.e., each estimation result for the same inputs is slightly different. In contrast, optimization-based motion planning algorithms require computationally efficient POC estimation, ideally using deterministic estimation, such that typical optimization algorithms for motion planning retain feasibility. Estimating the POC analytically, however, is challenging because it depends on understanding the collision conditions (e.g., vehicle's shape) and characterizing the uncertainty in motion prediction. In this paper, we propose an approach in which we estimate the POC between two vehicles by over-approximating their shapes by a multi-circular shape approximation. The position and heading of the predicted vehicle are modelled as random variables, contrasting with the literature, where the heading angle is often neglected. We guarantee that the provided POC is an over-approximation, which is essential in providing safety guarantees. For the particular case of Gaussian uncertainty in the position and heading, we present a computationally efficient algorithm for computing the POC estimate. This algorithm is then used in a path-following stochastic model predictive controller (SMPC) for motion planning. With the proposed algorithm, the SMPC generates reproducible trajectories while the controller retains its feasibility in the presented test cases and demonstrates the ability to handle varying levels of uncertainty.
>
---
#### [replaced 006] SurfSLAM: Sim-to-Real Underwater Stereo Reconstruction For Real-Time SLAM
- **分类: cs.RO**

- **简介: 该论文属于 underwater SLAM 任务，解决水下立体深度估计与定位问题。通过 sim-to-real 训练和多传感器融合，提升水下实时定位与三维重建精度。**

- **链接: [https://arxiv.org/pdf/2601.10814v2](https://arxiv.org/pdf/2601.10814v2)**

> **作者:** Onur Bagoren; Seth Isaacson; Sacchin Sundar; Yung-Ching Sun; Anja Sheppard; Haoyu Ma; Abrar Shariff; Ram Vasudevan; Katherine A. Skinner
>
> **摘要:** Localization and mapping are core perceptual capabilities for underwater robots. Stereo cameras provide a low-cost means of directly estimating metric depth to support these tasks. However, despite recent advances in stereo depth estimation on land, computing depth from image pairs in underwater scenes remains challenging. In underwater environments, images are degraded by light attenuation, visual artifacts, and dynamic lighting conditions. Furthermore, real-world underwater scenes frequently lack rich texture useful for stereo depth estimation and 3D reconstruction. As a result, stereo estimation networks trained on in-air data cannot transfer directly to the underwater domain. In addition, there is a lack of real-world underwater stereo datasets for supervised training of neural networks. Poor underwater depth estimation is compounded in stereo-based Simultaneous Localization and Mapping (SLAM) algorithms, making it a fundamental challenge for underwater robot perception. To address these challenges, we propose a novel framework that enables sim-to-real training of underwater stereo disparity estimation networks using simulated data and self-supervised finetuning. We leverage our learned depth predictions to develop SurfSLAM, a novel framework for real-time underwater SLAM that fuses stereo cameras with IMU, barometric, and Doppler Velocity Log (DVL) measurements. Lastly, we collect a challenging real-world dataset of shipwreck surveys using an underwater robot. Our dataset features over 24,000 stereo pairs, along with high-quality, dense photogrammetry models and reference trajectories for evaluation. Through extensive experiments, we demonstrate the advantages of the proposed training approach on real-world data for improving stereo estimation in the underwater domain and for enabling accurate trajectory estimation and 3D reconstruction of complex shipwreck sites.
>
---
#### [replaced 007] VR$^2$: A Co-Located Dual-Headset Platform for Touch-Enabled Human-Robot Interaction Research
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出VR2VR平台，用于触觉人机交互研究。解决传统方法难以同时实现物理接触与虚拟体验的问题，通过双头显系统实现同步视觉与触觉反馈。**

- **链接: [https://arxiv.org/pdf/2601.12395v2](https://arxiv.org/pdf/2601.12395v2)**

> **作者:** Chao Wang; Anna Belardinelli; Michael Gienger
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Social-physical human-robot interaction (HRI) is difficult to study: building and programming robots integrating multiple interaction modalities is costly and slow, while VR-based prototypes often lack physical contact capabilities, breaking the visuo-tactile expectations of the user. We present VR2VR, a co-located dual-VR-headset platform for HRI research in which a participant and a hidden operator share the same physical space while experiencing different virtual embodiments. The participant sees an expressive virtual robot that interacts face-to-face in a shared virtual environment. In real time, the robot's upper-body movements, head and gaze behaviors, and facial expressions are mapped from the operator's tracked limbs and face signals. Since the operator is physically co-present and calibrated into the same coordinate frame, the operator can also touch the participant, enabling the participant to perceive robot touch synchronized with the visual perception of the robot's hands on their hands: the operator's finger and hand motion is mapped to the robot avatar using inverse kinematics to support precise contact. Beyond faithful motion retargeting for limb control, our VR2VR system supports social retargeting of multiple nonverbal cues, which can be experimentally varied and investigated while keeping the physical interaction constant. We detail the system design, calibration workflow, and safety considerations, and demonstrate how the platform can be used for experimentation and data collection in a touch-based Wizard-of-Oz HRI study, thus illustrating how VR2VR lowers barriers for rapidly prototyping and rigorously evaluating embodied, contact-based robot behaviors.
>
---
#### [replaced 008] Allocation for Omnidirectional Aerial Robots: Incorporating Power Dynamics
- **分类: cs.RO**

- **简介: 该论文研究倾斜旋翼飞行器的执行器分配问题，旨在解决过驱动与执行器动态带来的协调难题。提出三种新方法，提升系统性能与灵活性。**

- **链接: [https://arxiv.org/pdf/2412.16107v3](https://arxiv.org/pdf/2412.16107v3)**

> **作者:** Eugenio Cuniato; Mike Allenspach; Thomas Stastny; Helen Oleynikova; Roland Siegwart; Michael Pantic
>
> **摘要:** Tilt-rotor aerial robots are more dynamic and versatile than fixed-rotor platforms, since the thrust vector and body orientation are decoupled. However, the coordination of servos and propellers (the allocation problem) is not trivial, especially accounting for overactuation and actuator dynamics. We incrementally build and present three novel allocation methods for tilt-rotor aerial robots, comparing them to state-of-the-art methods on a real system performing dynamic maneuvers. We extend the state-of-the-art geometric allocation into a differential allocation, which uses the platform's redundancy and does not suffer from singularities. We expand it by incorporating actuator dynamics and propeller power dynamics. These allow us to model dynamic propeller acceleration limits, bringing two main advantages: balancing propeller speed without the need for nullspace goals and allowing the platform to selectively turn off propellers during flight, opening the door to new manipulation possibilities. We also use actuator dynamics and limits to normalize the allocation problem, making it easier to tune and allowing it to track 70% faster trajectories than a geometric allocation.
>
---
#### [replaced 009] Warm-Starting Collision-Free Model Predictive Control With Object-Centric Diffusion
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，旨在解决复杂环境中快速生成无碰撞轨迹的问题。通过结合扩散模型与MPC，利用对象中心表示生成可靠轨迹，实现高效实时控制。**

- **链接: [https://arxiv.org/pdf/2601.02873v2](https://arxiv.org/pdf/2601.02873v2)**

> **作者:** Arthur Haffemayer; Alexandre Chapin; Armand Jordana; Krzysztof Wojciechowski; Florent Lamiraux; Nicolas Mansard; Vladimir Petrik
>
> **备注:** An open-source implementation is provided https://ahaffemayer.github.io/diffusion_warmstart_slot/
>
> **摘要:** Acting in cluttered environments requires predicting and avoiding collisions while still achieving precise control. Conventional optimization-based controllers can enforce physical constraints, but they struggle to produce feasible solutions quickly when many obstacles are present. Diffusion models can generate diverse trajectories around obstacles, yet prior approaches lacked a general and efficient way to condition them on scene structure. In this paper, we show that combining diffusion-based warm-starting conditioned with a latent object-centric representation of the scene and with a collision-aware model predictive controller (MPC) yields reliable and efficient motion generation under strict time limits. Our approach conditions a diffusion transformer on the system state, task, and surroundings, using an object-centric slot attention mechanism to provide a compact obstacle representation suitable for control. The sampled trajectories are refined by an optimal control problem that enforces rigid-body dynamics and signed-distance collision constraints, producing feasible motions in real time. On benchmark tasks, this hybrid method achieved markedly higher success rates and lower latency than sampling-based planners or either component alone. Real-robot experiments with a torque-controlled Panda confirm reliable and safe execution with MPC.
>
---
#### [replaced 010] DAPPER: Discriminability-Aware Policy-to-Policy Preference-Based Reinforcement Learning for Query-Efficient Robot Skill Acquisition
- **分类: cs.RO**

- **简介: 该论文属于机器人技能学习任务，解决PbRL查询效率低的问题。通过多策略比较提升轨迹多样性与可判别性，提出DAPPER方法提高学习效率。**

- **链接: [https://arxiv.org/pdf/2505.06357v4](https://arxiv.org/pdf/2505.06357v4)**

> **作者:** Yuki Kadokawa; Jonas Frey; Takahiro Miki; Takamitsu Matsubara; Marco Hutter
>
> **备注:** Accepted for IEEE Robotics & Automation Magazine (RAM)
>
> **摘要:** Preference-based Reinforcement Learning (PbRL) enables policy learning through simple queries comparing trajectories from a single policy. While human responses to these queries make it possible to learn policies aligned with human preferences, PbRL suffers from low query efficiency, as policy bias limits trajectory diversity and reduces the number of discriminable queries available for learning preferences. This paper identifies preference discriminability, which quantifies how easily a human can judge which trajectory is closer to their ideal behavior, as a key metric for improving query efficiency. To address this, we move beyond comparisons within a single policy and instead generate queries by comparing trajectories from multiple policies, as training them from scratch promotes diversity without policy bias. We propose Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning (DAPPER), which integrates preference discriminability with trajectory diversification achieved by multiple policies. DAPPER trains new policies from scratch after each reward update and employs a discriminator that learns to estimate preference discriminability, enabling the prioritized sampling of more discriminable queries. During training, it jointly maximizes the preference reward and preference discriminability score, encouraging the discovery of highly rewarding and easily distinguishable policies. Experiments in simulated and real-world legged robot environments demonstrate that DAPPER outperforms previous methods in query efficiency, particularly under challenging preference discriminability conditions. A supplementary video that facilitates understanding of the proposed framework and its experimental results is available at: https://youtu.be/lRwX8FNN8n4
>
---
#### [replaced 011] DroneVLA: VLA based Aerial Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于航空操作任务，旨在解决非专家用户自然操控无人机抓取物体的问题。通过结合视觉语言动作模型与导航算法，实现从指令到抓取的自动化操作。**

- **链接: [https://arxiv.org/pdf/2601.13809v2](https://arxiv.org/pdf/2601.13809v2)**

> **作者:** Fawad Mehboob; Monijesu James; Amir Habel; Jeffrin Sam; Miguel Altamirano Cabrera; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted for publication at LBR of HRI 2026 conference
>
> **摘要:** As aerial platforms evolve from passive observers to active manipulators, the challenge shifts toward designing intuitive interfaces that allow non-expert users to command these systems naturally. This work introduces a novel concept of autonomous aerial manipulation system capable of interpreting high-level natural language commands to retrieve objects and deliver them to a human user. The system is intended to integrate a MediaPipe based on Grounding DINO and a Vision-Language-Action (VLA) model with a custom-built drone equipped with a 1-DOF gripper and an Intel RealSense RGB-D camera. VLA performs semantic reasoning to interpret the intent of a user prompt and generates a prioritized task queue for grasping of relevant objects in the scene. Grounding DINO and dynamic A* planning algorithm are used to navigate and safely relocate the object. To ensure safe and natural interaction during the handover phase, the system employs a human-centric controller driven by MediaPipe. This module provides real-time human pose estimation, allowing the drone to employ visual servoing to maintain a stable, distinct position directly in front of the user, facilitating a comfortable handover. We demonstrate the system's efficacy through real-world experiments for localization and navigation, which resulted in a 0.164m, 0.070m, and 0.084m of max, mean euclidean, and root-mean squared errors, respectively, highlighting the feasibility of VLA for aerial manipulation operations.
>
---
#### [replaced 012] Teaching Robots Like Dogs: Learning Agile Navigation from Luring, Gesture, and Speech
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决机器人通过人类社交线索学习敏捷导航的问题。通过人机协作框架，提升导航效率与行为一致性。**

- **链接: [https://arxiv.org/pdf/2601.08422v2](https://arxiv.org/pdf/2601.08422v2)**

> **作者:** Taerim Yoon; Dongho Kang; Jin Cheng; Fatemeh Zargarbashi; Yijiang Huang; Minsung Ahn; Stelian Coros; Sungjoon Choi
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** In this work, we aim to enable legged robots to learn how to interpret human social cues and produce appropriate behaviors through physical human guidance. However, learning through physical engagement can place a heavy burden on users when the process requires large amounts of human-provided data. To address this, we propose a human-in-the-loop framework that enables robots to acquire navigational behaviors in a data-efficient manner and to be controlled via multimodal natural human inputs, specifically gestural and verbal commands. We reconstruct interaction scenes using a physics-based simulation and aggregate data to mitigate distributional shifts arising from limited demonstration data. Our progressive goal cueing strategy adaptively feeds appropriate commands and navigation goals during training, leading to more accurate navigation and stronger alignment between human input and robot behavior. We evaluate our framework across six real-world agile navigation scenarios, including jumping over or avoiding obstacles. Our experimental results show that our proposed method succeeds in almost all trials across these scenarios, achieving a 97.15% task success rate with less than 1 hour of demonstration data in total.
>
---
#### [replaced 013] Context-aware Learned Mesh-based Simulation via Trajectory-Level Meta-Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于物理模拟任务，旨在解决传统模拟器速度慢和学习模型依赖单步观测的问题。通过轨迹级元学习，提出M3GN方法，实现高效准确的变形模拟。**

- **链接: [https://arxiv.org/pdf/2511.05234v2](https://arxiv.org/pdf/2511.05234v2)**

> **作者:** Philipp Dahlinger; Niklas Freymuth; Tai Hoang; Tobias Würth; Michael Volpp; Luise Kärger; Gerhard Neumann
>
> **备注:** 35 pages. Submitted to Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Simulating object deformations is a critical challenge across many scientific domains, including robotics, manufacturing, and structural mechanics. Learned Graph Network Simulators (GNSs) offer a promising alternative to traditional mesh-based physics simulators. Their speed and inherent differentiability make them particularly well suited for applications that require fast and accurate simulations, such as robotic manipulation or manufacturing optimization. However, existing learned simulators typically rely on single-step observations, which limits their ability to exploit temporal context. Without this information, these models fail to infer, e.g., material properties. Further, they rely on auto-regressive rollouts, which quickly accumulate error for long trajectories. We instead frame mesh-based simulation as a trajectory-level meta-learning problem. Using Conditional Neural Processes, our method enables rapid adaptation to new simulation scenarios from limited initial data while capturing their latent simulation properties. We utilize movement primitives to directly predict fast, stable and accurate simulations from a single model call. The resulting approach, Movement-primitive Meta-MeshGraphNet (M3GN), provides higher simulation accuracy at a fraction of the runtime cost compared to state-of-the-art GNSs across several tasks.
>
---
