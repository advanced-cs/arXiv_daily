# 机器人 cs.RO

- **最新发布 30 篇**

- **更新 13 篇**

## 最新发布

#### [new 001] ROVER: Robust Loop Closure Verification with Trajectory Prior in Repetitive Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM中的回环验证任务，旨在解决重复环境中因外观相似导致的误检问题。作者提出ROVER方法，利用历史轨迹作为先验约束，通过轨迹一致性评分验证回环真伪，提升鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2508.13488v1](http://arxiv.org/pdf/2508.13488v1)**

> **作者:** Jingwen Yu; Jiayi Yang; Anjun Hu; Jiankun Wang; Ping Tan; Hong Zhang
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Loop closure detection is important for simultaneous localization and mapping (SLAM), which associates current observations with historical keyframes, achieving drift correction and global relocalization. However, a falsely detected loop can be fatal, and this is especially difficult in repetitive environments where appearance-based features fail due to the high similarity. Therefore, verification of a loop closure is a critical step in avoiding false positive detections. Existing works in loop closure verification predominantly focus on learning invariant appearance features, neglecting the prior knowledge of the robot's spatial-temporal motion cue, i.e., trajectory. In this letter, we propose ROVER, a loop closure verification method that leverages the historical trajectory as a prior constraint to reject false loops in challenging repetitive environments. For each loop candidate, it is first used to estimate the robot trajectory with pose-graph optimization. This trajectory is then submitted to a scoring scheme that assesses its compliance with the trajectory without the loop, which we refer to as the trajectory prior, to determine if the loop candidate should be accepted. Benchmark comparisons and real-world experiments demonstrate the effectiveness of the proposed method. Furthermore, we integrate ROVER into state-of-the-art SLAM systems to verify its robustness and efficiency. Our source code and self-collected dataset are available at https://github.com/jarvisyjw/ROVER.
>
---
#### [new 002] Accelerating Signal-Temporal-Logic-Based Task and Motion Planning of Bipedal Navigation using Benders Decomposition
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13407v1](http://arxiv.org/pdf/2508.13407v1)**

> **作者:** Jiming Ren; Xuan Lin; Roman Mineyev; Karen M. Feigh; Samuel Coogan; Ye Zhao
>
> **备注:** 16 pages, 7 figures, 6 tables
>
> **摘要:** Task and motion planning under Signal Temporal Logic constraints is known to be NP-hard. A common class of approaches formulates these hybrid problems, which involve discrete task scheduling and continuous motion planning, as mixed-integer programs (MIP). However, in applications for bipedal locomotion, introduction of non-convex constraints such as kinematic reachability and footstep rotation exacerbates the computational complexity of MIPs. In this work, we present a method based on Benders Decomposition to address scenarios where solving the entire monolithic optimization problem is prohibitively intractable. Benders Decomposition proposes an iterative cutting-plane technique that partitions the problem into a master problem to prototype a plan that meets the task specification, and a series of subproblems for kinematics and dynamics feasibility checks. Our experiments demonstrate that this method achieves faster planning compared to alternative algorithms for solving the resulting optimization program with nonlinear constraints.
>
---
#### [new 003] Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出Embodied-R1，一个用于机器人操作的视觉语言模型，通过“指向”作为统一表示解决感知与行动之间的差距问题。工作包括构建大规模数据集、两阶段强化微调训练，并在多个基准上实现卓越的零样本泛化性能。**

- **链接: [http://arxiv.org/pdf/2508.13998v1](http://arxiv.org/pdf/2508.13998v1)**

> **作者:** Yifu Yuan; Haiqin Cui; Yaoting Huang; Yibin Chen; Fei Ni; Zibin Dong; Pengyi Li; Yan Zheng; Jianye Hao
>
> **备注:** Embodied-R1 technical report
>
> **摘要:** Generalization in embodied AI is hindered by the "seeing-to-doing gap," which stems from data scarcity and embodiment heterogeneity. To address this, we pioneer "pointing" as a unified, embodiment-agnostic intermediate representation, defining four core embodied pointing abilities that bridge high-level vision-language comprehension with low-level action primitives. We introduce Embodied-R1, a 3B Vision-Language Model (VLM) specifically designed for embodied reasoning and pointing. We use a wide range of embodied and general visual reasoning datasets as sources to construct a large-scale dataset, Embodied-Points-200K, which supports key embodied pointing capabilities. We then train Embodied-R1 using a two-stage Reinforced Fine-tuning (RFT) curriculum with a specialized multi-task reward design. Embodied-R1 achieves state-of-the-art performance on 11 embodied spatial and pointing benchmarks. Critically, it demonstrates robust zero-shot generalization by achieving a 56.2% success rate in the SIMPLEREnv and 87.5% across 8 real-world XArm tasks without any task-specific fine-tuning, representing a 62% improvement over strong baselines. Furthermore, the model exhibits high robustness against diverse visual disturbances. Our work shows that a pointing-centric representation, combined with an RFT training paradigm, offers an effective and generalizable pathway to closing the perception-action gap in robotics.
>
---
#### [new 004] Train Once, Deploy Anywhere: Realize Data-Efficient Dynamic Object Manipulation
- **分类: cs.RO**

- **简介: 论文研究动态物体操作的泛化问题，旨在用少量示范实现跨环境、机器人形态和物体形状的高效操作。提出基于熵的理论框架与GEM系统，在仿真和真实场景中验证其高成功率，实现在餐厅无示范条件下97%以上成功操作。**

- **链接: [http://arxiv.org/pdf/2508.14042v1](http://arxiv.org/pdf/2508.14042v1)**

> **作者:** Zhuoling Li; Xiaoyang Wu; Zhenhua Xu; Hengshuang Zhao
>
> **摘要:** Realizing generalizable dynamic object manipulation is important for enhancing manufacturing efficiency, as it eliminates specialized engineering for various scenarios. To this end, imitation learning emerges as a promising paradigm, leveraging expert demonstrations to teach a policy manipulation skills. Although the generalization of an imitation learning policy can be improved by increasing demonstrations, demonstration collection is labor-intensive. To address this problem, this paper investigates whether strong generalization in dynamic object manipulation is achievable with only a few demonstrations. Specifically, we develop an entropy-based theoretical framework to quantify the optimization of imitation learning. Based on this framework, we propose a system named Generalizable Entropy-based Manipulation (GEM). Extensive experiments in simulated and real tasks demonstrate that GEM can generalize across diverse environment backgrounds, robot embodiments, motion dynamics, and object geometries. Notably, GEM has been deployed in a real canteen for tableware collection. Without any in-scene demonstration, it achieves a success rate of over 97% across more than 10,000 operations.
>
---
#### [new 005] Augmenting cobots for sheet-metal SMEs with 3D object recognition and localisation
- **分类: cs.RO; cs.CV**

- **简介: 论文探讨如何通过3D物体识别与定位技术增强协作机器人（cobots）能力，解决中小型企业钣金车间高混合低批量生产中的自动化难题，提升生产效率并优化技工使用。**

- **链接: [http://arxiv.org/pdf/2508.13964v1](http://arxiv.org/pdf/2508.13964v1)**

> **作者:** Martijn Cramer; Yanming Wu; David De Schepper; Eric Demeester
>
> **备注:** 13 pages, 25 figures
>
> **摘要:** Due to high-mix-low-volume production, sheet-metal workshops today are challenged by small series and varying orders. As standard automation solutions tend to fall short, SMEs resort to repetitive manual labour impacting production costs and leading to tech-skilled workforces not being used to their full potential. The COOCK+ ROBUST project aims to transform cobots into mobile and reconfigurable production assistants by integrating existing technologies, including 3D object recognition and localisation. This article explores both the opportunities and challenges of enhancing cobotic systems with these technologies in an industrial setting, outlining the key steps involved in the process. Additionally, insights from a past project, carried out by the ACRO research unit in collaboration with an industrial partner, serves as a concrete implementation example throughout.
>
---
#### [new 006] Multi-Robot Navigation in Social Mini-Games: Definitions, Taxonomy, and Algorithms
- **分类: cs.RO; cs.MA**

- **简介: 论文聚焦多机器人导航中的“社交迷你游戏”（SMG）问题，即机器人在狭窄拥挤环境中与人或其他机器人交互时的路径规划难题。作者提出首个统一的分类体系和评估标准，梳理现有方法并明确其假设与目标，以促进该领域研究规范化发展。**

- **链接: [http://arxiv.org/pdf/2508.13459v1](http://arxiv.org/pdf/2508.13459v1)**

> **作者:** Rohan Chandra; Shubham Singh; Abhishek Jha; Dannon Andrade; Hriday Sainathuni; Katia Sycara
>
> **摘要:** The ``Last Mile Challenge'' has long been considered an important, yet unsolved, challenge for autonomous vehicles, public service robots, and delivery robots. A central issue in this challenge is the ability of robots to navigate constrained and cluttered environments (e.g., doorways, hallways, corridor intersections), often while competing for space with other robots and humans. We refer to these environments as ``Social Mini-Games'' (SMGs). SMGs are tightly coupled, high-agency interactions that arise within general multi-robot navigation (MRN) scenarios. They are identified through certain distinct characteristics and require specialized metrics to evaluate them. Traditional navigation approaches designed for MRN do not perform well in SMGs, which has led to focused research on dedicated SMG solvers (navigation methods specialized to navigate in SMGs), which has flourished in recent years. However, publications on SMG navigation research make different assumptions (on centralized versus decentralized, observability, communication, cooperation, etc.), and have different objective functions (safety versus liveness). These assumptions and objectives are sometimes implicitly assumed or described informally. This makes it difficult to establish appropriate baselines for comparison in research papers, as well as making it difficult for practitioners to find the papers relevant to their concrete application. Such ad-hoc representation of the field also presents a barrier to new researchers wanting to start research in this area. SMG navigation research requires its own taxonomy, definitions, and evaluation protocols to guide effective research moving forward. This survey is the first to catalog SMG solvers using a well-defined and unified taxonomy and to classify existing methods accordingly.
>
---
#### [new 007] Diff-MSM: Differentiable MusculoSkeletal Model for Simultaneous Identification of Human Muscle and Bone Parameters
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Diff-MSM，一种用于同时识别肌肉和骨骼参数的可微分肌骨模型，解决因无法直接测量内部生物力学变量（如关节扭矩）而导致个性化模型参数难以确定的问题。通过端到端自动微分技术，从可测肌激活信号推导出可观测运动，显著提升肌肉参数估计精度。**

- **链接: [http://arxiv.org/pdf/2508.13303v1](http://arxiv.org/pdf/2508.13303v1)**

> **作者:** Yingfan Zhou; Philip Sanderink; Sigurd Jager Lemming; Cheng Fang
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** High-fidelity personalized human musculoskeletal models are crucial for simulating realistic behavior of physically coupled human-robot interactive systems and verifying their safety-critical applications in simulations before actual deployment, such as human-robot co-transportation and rehabilitation through robotic exoskeletons. Identifying subject-specific Hill-type muscle model parameters and bone dynamic parameters is essential for a personalized musculoskeletal model, but very challenging due to the difficulty of measuring the internal biomechanical variables in vivo directly, especially the joint torques. In this paper, we propose using Differentiable MusculoSkeletal Model (Diff-MSM) to simultaneously identify its muscle and bone parameters with an end-to-end automatic differentiation technique differentiating from the measurable muscle activation, through the joint torque, to the resulting observable motion without the need to measure the internal joint torques. Through extensive comparative simulations, the results manifested that our proposed method significantly outperformed the state-of-the-art baseline methods, especially in terms of accurate estimation of the muscle parameters (i.e., initial guess sampled from a normal distribution with the mean being the ground truth and the standard deviation being 10% of the ground truth could end up with an average of the percentage errors of the estimated values as low as 0.05%). In addition to human musculoskeletal modeling and simulation, the new parameter identification technique with the Diff-MSM has great potential to enable new applications in muscle health monitoring, rehabilitation, and sports science.
>
---
#### [new 008] A Three-Level Whole-Body Disturbance Rejection Control Framework for Dynamic Motions in Legged Robots
- **分类: cs.RO**

- **简介: 论文提出三层次全身扰动抑制控制框架（T-WB-DRC），用于提升腿式机器人在模型不确定、外部干扰和故障下的稳定性和鲁棒性，通过新型移动时域扩展状态观测器实现扰动估计与补偿。**

- **链接: [http://arxiv.org/pdf/2508.13531v1](http://arxiv.org/pdf/2508.13531v1)**

> **作者:** Bolin Li; Gewei Zuo; Zhixiang Wang; Xiaotian Ke; Lijun Zhu; Han Ding
>
> **摘要:** This paper presents a control framework designed to enhance the stability and robustness of legged robots in the presence of uncertainties, including model uncertainties, external disturbances, and faults. The framework enables the full-state feedback estimator to estimate and compensate for uncertainties in whole-body dynamics of the legged robots. First, we propose a novel moving horizon extended state observer (MH-ESO) to estimate uncertainties and mitigate noise in legged systems, which can be integrated into the framework for disturbance compensation. Second, we introduce a three-level whole-body disturbance rejection control framework (T-WB-DRC). Unlike the previous two-level approach, this three-level framework considers both the plan based on whole-body dynamics without uncertainties and the plan based on dynamics with uncertainties, significantly improving payload transportation, external disturbance rejection, and fault tolerance. Third, simulations of both humanoid and quadruped robots in the Gazebo simulator demonstrate the effectiveness and versatility of T-WB-DRC. Finally, extensive experimental trials on a quadruped robot validate the robustness and stability of the system when using T-WB-DRC under various disturbance conditions.
>
---
#### [new 009] Blast Hole Seeking and Dipping -- The Navigation and Perception Framework in a Mine Site Inspection Robot
- **分类: cs.RO**

- **简介: 论文提出一种用于矿场钻孔检测机器人的导航与感知框架，解决人工钻孔检测效率低、精度差的问题。通过LiDAR点云处理和2D投影分割，实现钻孔精准定位与自主导航，确保传感器准确放置并避免碰撞。**

- **链接: [http://arxiv.org/pdf/2508.13785v1](http://arxiv.org/pdf/2508.13785v1)**

> **作者:** Liyang Liu; Ehsan Mihankhah; Nathan Wallace; Javier Martinez; Andrew J. Hill
>
> **摘要:** In open-pit mining, holes are drilled into the surface of the excavation site and detonated with explosives to facilitate digging. These blast holes need to be inspected internally for investigation of downhole material types and properties. Knowing these properties can lead to significant savings in material handling costs in downstream processes. Manual hole inspection is slow and expensive, with major limitations in revealing the geometric and geological properties of the holes and their contents. This has been the motivation for the development of our autonomous mine-site inspection robot - "DIPPeR". In this paper, the automation aspect of the project is explained. We present a robust blast hole seeking and detection framework that enables target-based navigation and accurate down-hole sensor positioning. The pipeline first processes point-cloud data collected by the on-board LiDAR sensors, extracting the cone-shaped volume of drill-waste above the ground. By projecting the 3D cone points into a virtual depth image, segmentation is achieved in the 2D domain, yielding a circular hole at the image centre and a collared cone face. We then identify the hole centre using a robust detection module while suppressing non-maximum candidates, ensuring precise sensor placement for down-hole inspection and avoiding collisions with the cavity wall. To enable autonomous hole-seeking, the pipeline automatically adjusts its projection parameters during robot navigation to account for variations in point sparsity and hole opening size, ensuring a consistent hole appearance in 2D images. This allows continuous tracking of the target hole as the robot approaches the goal point. We demonstrate the effectiveness of our navigation and perception system in both high-fidelity simulation environments and on-site field tests. A demonstration video is available at "https://www.youtube.com/watch?v=fRNbcBcaSqE".
>
---
#### [new 010] Toward Deployable Multi-Robot Collaboration via a Symbolically-Guided Decision Transformer
- **分类: cs.RO; cs.AI**

- **简介: 论文提出SGDT框架，结合神经符号规划与决策变压器，解决多机器人协作中长期依赖和部署难题。通过高层符号计划引导低层决策，实现可解释、泛化的多机器人操作。**

- **链接: [http://arxiv.org/pdf/2508.13877v1](http://arxiv.org/pdf/2508.13877v1)**

> **作者:** Rathnam Vidushika Rasanji; Jin Wei-Kocsis; Jiansong Zhang; Dongming Gan; Ragu Athinarayanan; Paul Asunda
>
> **摘要:** Reinforcement learning (RL) has demonstrated great potential in robotic operations. However, its data-intensive nature and reliance on the Markov Decision Process (MDP) assumption limit its practical deployment in real-world scenarios involving complex dynamics and long-term temporal dependencies, such as multi-robot manipulation. Decision Transformers (DTs) have emerged as a promising offline alternative by leveraging causal transformers for sequence modeling in RL tasks. However, their applications to multi-robot manipulations still remain underexplored. To address this gap, we propose a novel framework, Symbolically-Guided Decision Transformer (SGDT), which integrates a neuro-symbolic mechanism with a causal transformer to enable deployable multi-robot collaboration. In the proposed SGDT framework, a neuro-symbolic planner generates a high-level task-oriented plan composed of symbolic subgoals. Guided by these subgoals, a goal-conditioned decision transformer (GCDT) performs low-level sequential decision-making for multi-robot manipulation. This hierarchical architecture enables structured, interpretable, and generalizable decision making in complex multi-robot collaboration tasks. We evaluate the performance of SGDT across a range of task scenarios, including zero-shot and few-shot scenarios. To our knowledge, this is the first work to explore DT-based technology for multi-robot manipulation.
>
---
#### [new 011] Incremental Generalized Hybrid A*
- **分类: cs.RO**

- **简介: 论文提出IGHA*，一种用于复杂动态环境下实时路径规划的增量式树搜索算法，解决传统方法在高维空间中计算效率低的问题。通过动态组织节点扩展，显著减少搜索次数，提升规划速度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.13392v1](http://arxiv.org/pdf/2508.13392v1)**

> **作者:** Sidharth Talia; Oren Salzman; Siddhartha Srinivasa
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** We address the problem of efficiently organizing search over very large trees, which arises in many applications ranging from autonomous driving to aerial vehicles. Here, we are motivated by off-road autonomy, where real-time planning is essential. Classical approaches use graphs of motion primitives and exploit dominance to mitigate the curse of dimensionality and prune expansions efficiently. However, for complex dynamics, repeatedly solving two-point boundary-value problems makes graph construction too slow for fast kinodynamic planning. Hybrid A* (HA*) addressed this challenge by searching over a tree of motion primitives and introducing approximate pruning using a grid-based dominance check. However, choosing the grid resolution is difficult: too coarse risks failure, while too fine leads to excessive expansions and slow planning. We propose Incremental Generalized Hybrid A* (IGHA*), an anytime tree-search framework that dynamically organizes vertex expansions without rigid pruning. IGHA* provably matches or outperforms HA*. For both on-road kinematic and off-road kinodynamic planning queries for a car-like robot, variants of IGHA* use 6x fewer expansions to the best solution compared to an optimized version of HA*. In simulated off-road experiments in a high fidelity simulator, IGHA* outperforms HA*M when both are used in the loop with a model predictive controller. We demonstrate real-time performance both in simulation and on a small-scale off-road vehicle, enabling fast, robust planning under complex dynamics. Code: https://github.com/personalrobotics/IGHAStar
>
---
#### [new 012] CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 论文提出CAST方法，通过生成反事实标签增强机器人数据集的语言 grounding 和细粒度多样性，提升视觉-语言-动作模型对指令的理解与执行能力，显著改善导航任务中的指令跟随表现。**

- **链接: [http://arxiv.org/pdf/2508.13446v1](http://arxiv.org/pdf/2508.13446v1)**

> **作者:** Catherine Glossop; William Chen; Arjun Bhorkar; Dhruv Shah; Sergey Levine
>
> **摘要:** Generalist robots should be able to understand and follow user instructions, but current vision-language-action (VLA) models struggle with following fine-grained commands despite providing a powerful architecture for mapping open-vocabulary natural language instructions to robot actions. One cause for this is a lack of semantic diversity and language grounding in existing robot datasets and, specifically, a lack of fine-grained task diversity for similar observations. To address this, we present a novel method to augment existing robot datasets by leveraging vision language models to create counterfactual labels. Our method improves the language-following capabilities of VLAs by increasing the diversity and granularity of language grounding for robot datasets by generating counterfactual language and actions. We evaluate the resulting model's ability to follow language instructions, ranging from simple object-centric commands to complex referential tasks, by conducting visual language navigation experiments in 3 different indoor and outdoor environments. Our experiments demonstrate that counterfactual relabeling, without any additional data collection, significantly improves instruction-following in VLA policies, making them competitive with state-of-the-art methods and increasing success rate by 27% on navigation tasks.
>
---
#### [new 013] Assessing Pedestrian Behavior Around Autonomous Cleaning Robots in Public Spaces: Findings from a Field Observation
- **分类: cs.RO**

- **简介: 该论文属于人机交互（HRI）研究，旨在探讨行人对自主清洁机器人行为的反应。通过实地观察498名行人，发现机器人类型和移动模式显著影响行人的侧向避让行为，而分心与否无显著差异。研究为优化机器人通信策略提供依据。**

- **链接: [http://arxiv.org/pdf/2508.13699v1](http://arxiv.org/pdf/2508.13699v1)**

> **作者:** Maren Raab; Linda Miller; Zhe Zeng; Pascal Jansen; Martin Baumann; Johannes Kraus
>
> **摘要:** As autonomous robots become more common in public spaces, spontaneous encounters with laypersons are more frequent. For this, robots need to be equipped with communication strategies that enhance momentary transparency and reduce the probability of critical situations. Adapting these robotic strategies requires consideration of robot movements, environmental conditions, and user characteristics and states. While numerous studies have investigated the impact of distraction on pedestrians' movement behavior, limited research has examined this behavior in the presence of autonomous robots. This research addresses the impact of robot type and robot movement pattern on distracted and undistracted pedestrians' movement behavior. In a field setting, unaware pedestrians were videotaped while moving past two working, autonomous cleaning robots. Out of N=498 observed pedestrians, approximately 8% were distracted by smartphones. Distracted and undistracted pedestrians did not exhibit significant differences in their movement behaviors around the robots. Instead, both the larger sweeping robot and the offset rectangular movement pattern significantly increased the number of lateral adaptations compared to the smaller cleaning robot and the circular movement pattern. The offset rectangular movement pattern also led to significantly more close lateral adaptations. Depending on the robot type, the movement patterns led to differences in the distances of lateral adaptations. The study provides initial insights into pedestrian movement behavior around an autonomous cleaning robot in public spaces, contributing to the growing field of HRI research.
>
---
#### [new 014] Trajectory Tracking and Stabilization of Quadrotors Using Deep Koopman Model Predictive Control
- **分类: cs.RO**

- **简介: 论文提出DK-MPC框架，将深度Koopman算子与模型预测控制结合，用于四旋翼轨迹跟踪与稳定控制。通过数据驱动线性化非线性动力学，提升控制精度与实时性，优于传统非线性MPC。**

- **链接: [http://arxiv.org/pdf/2508.13795v1](http://arxiv.org/pdf/2508.13795v1)**

> **作者:** Haitham El-Hussieny
>
> **摘要:** This paper presents a data-driven control framework for quadrotor systems that integrates a deep Koopman operator with model predictive control (DK-MPC). The deep Koopman operator is trained on sampled flight data to construct a high-dimensional latent representation in which the nonlinear quadrotor dynamics are approximated by linear models. This linearization enables the application of MPC to efficiently optimize control actions over a finite prediction horizon, ensuring accurate trajectory tracking and stabilization. The proposed DK-MPC approach is validated through a series of trajectory-following and point-stabilization numerical experiments, where it demonstrates superior tracking accuracy and significantly lower computation time compared to conventional nonlinear MPC. These results highlight the potential of Koopman-based learning methods to handle complex quadrotor dynamics while meeting the real-time requirements of embedded flight control. Future work will focus on extending the framework to more agile flight scenarios and improving robustness against external disturbances.
>
---
#### [new 015] Driving Style Recognition Like an Expert Using Semantic Privileged Information from Large Language Models
- **分类: cs.RO**

- **简介: 论文提出利用大语言模型生成的语义特权信息（SPI）提升驾驶风格识别准确性，解决算法分类与专家判断不一致的问题。通过自然语言描述行为并编码为特征，结合SVM+训练，仅在训练时使用SPI，推理仍用传感器数据，显著提高F1分数。**

- **链接: [http://arxiv.org/pdf/2508.13881v1](http://arxiv.org/pdf/2508.13881v1)**

> **作者:** Zhaokun Chen; Chaopeng Zhang; Xiaohan Li; Wenshuo Wang; Gentiane Venture; Junqiang Xi
>
> **摘要:** Existing driving style recognition systems largely depend on low-level sensor-derived features for training, neglecting the rich semantic reasoning capability inherent to human experts. This discrepancy results in a fundamental misalignment between algorithmic classifications and expert judgments. To bridge this gap, we propose a novel framework that integrates Semantic Privileged Information (SPI) derived from large language models (LLMs) to align recognition outcomes with human-interpretable reasoning. First, we introduce DriBehavGPT, an interactive LLM-based module that generates natural-language descriptions of driving behaviors. These descriptions are then encoded into machine learning-compatible representations via text embedding and dimensionality reduction. Finally, we incorporate them as privileged information into Support Vector Machine Plus (SVM+) for training, enabling the model to approximate human-like interpretation patterns. Experiments across diverse real-world driving scenarios demonstrate that our SPI-enhanced framework outperforms conventional methods, achieving F1-score improvements of 7.6% (car-following) and 7.9% (lane-changing). Importantly, SPI is exclusively used during training, while inference relies solely on sensor data, ensuring computational efficiency without sacrificing performance. These results highlight the pivotal role of semantic behavioral representations in improving recognition accuracy while advancing interpretable, human-centric driving systems.
>
---
#### [new 016] Switch4EAI: Leveraging Console Game Platform for Benchmarking Robotic Athletics
- **分类: cs.RO**

- **简介: 论文提出Switch4EAI，利用任天堂Switch游戏《Just Dance》作为物理基准，评估机器人全身控制策略。解决机器人运动能力缺乏标准化评测的问题，通过游戏动作捕捉与重定向，实现机器人与人类在相同任务下的定量对比。**

- **链接: [http://arxiv.org/pdf/2508.13444v1](http://arxiv.org/pdf/2508.13444v1)**

> **作者:** Tianyu Li; Jeonghwan Kim; Wontaek Kim; Donghoon Baek; Seungeun Rho; Sehoon Ha
>
> **备注:** Workshop Submission
>
> **摘要:** Recent advances in whole-body robot control have enabled humanoid and legged robots to execute increasingly agile and coordinated movements. However, standardized benchmarks for evaluating robotic athletic performance in real-world settings and in direct comparison to humans remain scarce. We present Switch4EAI(Switch-for-Embodied-AI), a low-cost and easily deployable pipeline that leverages motion-sensing console games to evaluate whole-body robot control policies. Using Just Dance on the Nintendo Switch as a representative example, our system captures, reconstructs, and retargets in-game choreography for robotic execution. We validate the system on a Unitree G1 humanoid with an open-source whole-body controller, establishing a quantitative baseline for the robot's performance against a human player. In the paper, we discuss these results, which demonstrate the feasibility of using commercial games platform as physically grounded benchmarks and motivate future work to for benchmarking embodied AI.
>
---
#### [new 017] Unified Hierarchical MPC in Task Executing for Modular Manipulators across Diverse Morphologies
- **分类: cs.RO**

- **简介: 论文提出统一分层模型预测控制（H-MPC）用于模块化机械臂在不同构型下的任务执行，解决控制参数调优繁琐问题。通过高低层MPC协同，实现精准轨迹规划与光滑关节运动，提升控制精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.13513v1](http://arxiv.org/pdf/2508.13513v1)**

> **作者:** Maolin Lei; Edoardo Romiti; Arturo Laurenzi; Cheng Zhou; Wanli Xing; Liang Lu; Nikos G. Tsagarakis
>
> **摘要:** This work proposes a unified Hierarchical Model Predictive Control (H-MPC) for modular manipulators across various morphologies, as the controller can adapt to different configurations to execute the given task without extensive parameter tuning in the controller. The H-MPC divides the control process into two levels: a high-level MPC and a low-level MPC. The high-level MPC predicts future states and provides trajectory information, while the low-level MPC refines control actions by updating the predictive model based on this high-level information. This hierarchical structure allows for the integration of kinematic constraints and ensures smooth joint-space trajectories, even near singular configurations. Moreover, the low-level MPC incorporates secondary linearization by leveraging predictive information from the high-level MPC, effectively capturing the second-order Taylor expansion information of the kinematic model while still maintaining a linearized model formulation. This approach not only preserves the simplicity of a linear control model but also enhances the accuracy of the kinematic representation, thereby improving overall control precision and reliability. To validate the effectiveness of the control policy, we conduct extensive evaluations across different manipulator morphologies and demonstrate the execution of pick-and-place tasks in real-world scenarios.
>
---
#### [new 018] MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出MimicFunc框架，通过功能对应关系从单个RGB-D人类视频中模仿工具操作，解决机器人在几何差异下难以泛化工具操作的问题。该方法利用关键点抽象构建功能坐标系，实现一次学习即可适配新工具，无需繁琐的遥操作数据收集。**

- **链接: [http://arxiv.org/pdf/2508.13534v1](http://arxiv.org/pdf/2508.13534v1)**

> **作者:** Chao Tang; Anxing Xiao; Yuhong Deng; Tianrun Hu; Wenlong Dong; Hanbo Zhang; David Hsu; Hong Zhang
>
> **备注:** Accepted to CoRL 2025
>
> **摘要:** Imitating tool manipulation from human videos offers an intuitive approach to teaching robots, while also providing a promising and scalable alternative to labor-intensive teleoperation data collection for visuomotor policy learning. While humans can mimic tool manipulation behavior by observing others perform a task just once and effortlessly transfer the skill to diverse tools for functionally equivalent tasks, current robots struggle to achieve this level of generalization. A key challenge lies in establishing function-level correspondences, considering the significant geometric variations among functionally similar tools, referred to as intra-function variations. To address this challenge, we propose MimicFunc, a framework that establishes functional correspondences with function frame, a function-centric local coordinate frame constructed with keypoint-based abstraction, for imitating tool manipulation skills. Experiments demonstrate that MimicFunc effectively enables the robot to generalize the skill from a single RGB-D human video to manipulating novel tools for functionally equivalent tasks. Furthermore, leveraging MimicFunc's one-shot generalization capability, the generated rollouts can be used to train visuomotor policies without requiring labor-intensive teleoperation data collection for novel objects. Our code and video are available at https://sites.google.com/view/mimicfunc.
>
---
#### [new 019] A Surveillance Based Interactive Robot
- **分类: cs.RO; cs.AI; cs.CV; I.2.9; I.2.10; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.13319v1](http://arxiv.org/pdf/2508.13319v1)**

> **作者:** Kshitij Kavimandan; Pooja Mangal; Devanshi Mehta
>
> **备注:** 4 pages, 5 figures
>
> **摘要:** We build a mobile surveillance robot that streams video in real time and responds to speech so a user can monitor and steer it from a phone or browser. The system uses two Raspberry Pi 4 units: a front unit on a differential drive base with camera, mic, and speaker, and a central unit that serves the live feed and runs perception. Video is sent with FFmpeg. Objects in the scene are detected using YOLOv3 to support navigation and event awareness. For voice interaction, we use Python libraries for speech recognition, multilingual translation, and text-to-speech, so the robot can take spoken commands and read back responses in the requested language. A Kinect RGB-D sensor provides visual input and obstacle cues. In indoor tests the robot detects common objects at interactive frame rates on CPU, recognises commands reliably, and translates them to actions without manual control. The design relies on off-the-shelf hardware and open software, making it easy to reproduce. We discuss limits and practical extensions, including sensor fusion with ultrasonic range data, GPU acceleration, and adding face and text recognition.
>
---
#### [new 020] Multimodal Data Storage and Retrieval for Embodied AI: A Survey
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于Embodied AI的数据管理任务，旨在解决多模态数据存储与检索难题。通过系统分析五种存储架构和五种检索范式，识别出物理接地缺失、跨模态整合等瓶颈，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2508.13901v1](http://arxiv.org/pdf/2508.13901v1)**

> **作者:** Yihao Lu; Hao Tang
>
> **摘要:** Embodied AI (EAI) agents continuously interact with the physical world, generating vast, heterogeneous multimodal data streams that traditional management systems are ill-equipped to handle. In this survey, we first systematically evaluate five storage architectures (Graph Databases, Multi-Model Databases, Data Lakes, Vector Databases, and Time-Series Databases), focusing on their suitability for addressing EAI's core requirements, including physical grounding, low-latency access, and dynamic scalability. We then analyze five retrieval paradigms (Fusion Strategy-Based Retrieval, Representation Alignment-Based Retrieval, Graph-Structure-Based Retrieval, Generation Model-Based Retrieval, and Efficient Retrieval-Based Optimization), revealing a fundamental tension between achieving long-term semantic coherence and maintaining real-time responsiveness. Based on this comprehensive analysis, we identify key bottlenecks, spanning from the foundational Physical Grounding Gap to systemic challenges in cross-modal integration, dynamic adaptation, and open-world generalization. Finally, we outline a forward-looking research agenda encompassing physics-aware data models, adaptive storage-retrieval co-optimization, and standardized benchmarking, to guide future research toward principled data management solutions for EAI. Our survey is based on a comprehensive review of more than 180 related studies, providing a rigorous roadmap for designing the robust, high-performance data management frameworks essential for the next generation of autonomous embodied systems.
>
---
#### [new 021] Modeling and Control of AWOISV: A Filtered Tube-Based MPC Approach for Simultaneous Tracking of Lateral Position and Heading Angle
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文针对全轮独立转向车辆（AWOISV）的高精度轨迹跟踪问题，提出基于滤波管式模型预测控制（FT-LTVMPC）方法，实现横向位置与航向角的同时精确控制，解决运动模式切换与参数不确定下的鲁棒性难题。**

- **链接: [http://arxiv.org/pdf/2508.13457v1](http://arxiv.org/pdf/2508.13457v1)**

> **作者:** Xu Yang; Jun Ni; Hengyang Feng; Feiyu Wang; Tiezhen Wang
>
> **摘要:** An all-wheel omni-directional independent steering vehicle (AWOISV) is a specialized all-wheel independent steering vehicle with each wheel capable of steering up to 90{\deg}, enabling unique maneuvers like yaw and diagonal movement. This paper introduces a theoretical steering radius angle and sideslip angle (\( \theta_R \)-\(\beta_R \)) representation, based on the position of the instantaneous center of rotation relative to the wheel rotation center, defining the motion modes and switching criteria for AWOISVs. A generalized \( v\)-\(\beta\)-\(r \) dynamic model is developed with forward velocity \(v\), sideslip angle \(\beta\), and yaw rate \(r\) as states, and \(\theta_R\) and \(\beta_R\) as control inputs. This model decouples longitudinal and lateral motions into forward and rotational motions, allowing seamless transitions across all motion modes under specific conditions. A filtered tube-based linear time-varying MPC (FT-LTVMPC) strategy is proposed, achieving simultaneous tracking of lateral position and arbitrary heading angles, with robustness to model inaccuracies and parameter uncertainties. Co-simulation and hardware-in-loop (HIL) experiments confirm that FT-LTVMPC enables high-precision control of both position and heading while ensuring excellent real-time performance.
>
---
#### [new 022] Toward an Interaction-Centered Approach to Robot Trustworthiness
- **分类: cs.RO**

- **简介: 论文提出以交互为中心的机器人可信性框架，旨在解决人机信任错位问题。通过人类意识与透明度两大支柱，提升机器人行为与人类预期的一致性，确保安全有效的协作。**

- **链接: [http://arxiv.org/pdf/2508.13976v1](http://arxiv.org/pdf/2508.13976v1)**

> **作者:** Carlo Mazzola; Hassan Ali; Kristína Malinovská; Igor Farkaš
>
> **备注:** 4 pages, presented at TRUST workshop, organised in conjunction with the IEEE RO-MAN 2025 conference, held in Eindhoven, Netherlands
>
> **摘要:** As robots get more integrated into human environments, fostering trustworthiness in embodied robotic agents becomes paramount for an effective and safe human-robot interaction (HRI). To achieve that, HRI applications must promote human trust that aligns with robot skills and avoid misplaced trust or overtrust, which can pose safety risks and ethical concerns. To achieve that, HRI applications must promote human trust that aligns with robot skills and avoid misplaced trust or overtrust, which can pose safety risks and ethical concerns. In this position paper, we outline an interaction-based framework for building trust through mutual understanding between humans and robots. We emphasize two main pillars: human awareness and transparency, referring to the robot ability to interpret human actions accurately and to clearly communicate its intentions and goals, respectively. By integrating these two pillars, robots can behave in a manner that aligns with human expectations and needs while providing their human partners with both comprehension and control over their actions. We also introduce four components that we think are important for bridging the gap between a human-perceived sense of trust and a robot true capabilities.
>
---
#### [new 023] The Social Context of Human-Robot Interactions
- **分类: cs.RO; cs.AI; cs.HC; cs.MA; I.2.9; I.2**

- **简介: 该论文属于人机交互领域，旨在解决“社会情境”术语使用不统一的问题。作者通过文献综述提出一个概念模型，用于描述和分析人机交互中的社会情境，并应用于现有研究，帮助设计、建模与理解机器人行为。**

- **链接: [http://arxiv.org/pdf/2508.13982v1](http://arxiv.org/pdf/2508.13982v1)**

> **作者:** Sydney Thompson; Kate Candon; Marynel Vázquez
>
> **备注:** To be published in Annual Review of Control, Robotics, and Autonomous Systems
>
> **摘要:** The Human-Robot Interaction (HRI) community often highlights the social context of an interaction as a key consideration when designing, implementing, and evaluating robot behavior. Unfortunately, researchers use the term "social context" in varied ways. This can lead to miscommunication, making it challenging to draw connections between related work on understanding and modeling the social contexts of human-robot interactions. To address this gap, we survey the HRI literature for existing definitions and uses of the term "social context". Then, we propose a conceptual model for describing the social context of a human-robot interaction. We apply this model to existing work, and we discuss a range of attributes of social contexts that can help researchers plan for interactions, develop behavior models for robots, and gain insights after interactions have taken place. We conclude with a discussion of open research questions in relation to understanding and modeling the social contexts of human-robot interactions.
>
---
#### [new 024] AutoMPC: A Code Generator for MPC-based Automated Driving
- **分类: eess.SY; cs.MS; cs.RO; cs.SY; 93-04**

- **简介: 论文提出AutoMPC，一个用于自动驾驶轨迹跟踪的模型预测控制代码生成工具。针对MPC计算复杂、部署困难的问题，该工具自动生成高效C代码，支持灵活配置车辆模型与数值方法，确保实时性与鲁棒性，适用于多种驾驶场景。**

- **链接: [http://arxiv.org/pdf/2508.13656v1](http://arxiv.org/pdf/2508.13656v1)**

> **作者:** Georg Schildbach; Jasper Pflughaupt
>
> **备注:** Technical Documentation
>
> **摘要:** Model Predictive Control (MPC) is a powerful technique to control nonlinear, multi-input multi-output systems subject to input and state constraints. It is now a standard tool for trajectory tracking control of automated vehicles. As such it has been used in many research and development projects. However, MPC faces several challenges to be integrated into industrial production vehicles. The most important ones are its high computational demands and the complexity of implementation. The software packages AutoMPC aims to address both of these challenges. It builds on a robustified version of an active set algorithm for Nonlinear MPC. The algorithm is embedded into a framework for vehicle trajectory tracking, which makes it easy to used, yet highly customizable. Automatic code generation transforms the selections into a standalone, computationally efficient C-code file with static memory allocation. As such it can be readily deployed on a wide range of embedded platforms, e.g., based on Matlab/Simulink or Robot Operating System (ROS). Compared to a previous version of the code, the vehicle model and the numerical integration method can be manually specified, besides basic algorithm parameters. All of this information and all specifications are directly baked into the generated C-code. The algorithm is suitable driving scenarios at low or high speeds, even drifting, and supports direction changes. Multiple simulation scenarios show the versatility and effectiveness of the AutoMPC code, with the guarantee of a feasible solution, a high degree of robustness, and computational efficiency.
>
---
#### [new 025] MR6D: Benchmarking 6D Pose Estimation for Mobile Robots
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MR6D数据集，用于移动机器人在工业环境中的6D姿态估计任务。针对现有数据集多聚焦家用小物体、忽略移动平台特有挑战的问题，MR6D包含92个真实场景和16个大尺寸物体，涵盖远距离视角、复杂遮挡等难点，揭示当前方法性能不足，推动移动端专用姿态估计研究。**

- **链接: [http://arxiv.org/pdf/2508.13775v1](http://arxiv.org/pdf/2508.13775v1)**

> **作者:** Anas Gouda; Shrutarv Awasthi; Christian Blesing; Lokeshwaran Manohar; Frank Hoffmann; Alice Kirchheim
>
> **备注:** accepted CVPR 2025 Workshop on Recovering 6D Object Pose (R6D)
>
> **摘要:** Existing 6D pose estimation datasets primarily focus on small household objects typically handled by robot arm manipulators, limiting their relevance to mobile robotics. Mobile platforms often operate without manipulators, interact with larger objects, and face challenges such as long-range perception, heavy self-occlusion, and diverse camera perspectives. While recent models generalize well to unseen objects, evaluations remain confined to household-like settings that overlook these factors. We introduce MR6D, a dataset designed for 6D pose estimation for mobile robots in industrial environments. It includes 92 real-world scenes featuring 16 unique objects across static and dynamic interactions. MR6D captures the challenges specific to mobile platforms, including distant viewpoints, varied object configurations, larger object sizes, and complex occlusion/self-occlusion patterns. Initial experiments reveal that current 6D pipelines underperform in these settings, with 2D segmentation being another hurdle. MR6D establishes a foundation for developing and evaluating pose estimation methods tailored to the demands of mobile robotics. The dataset is available at https://huggingface.co/datasets/anas-gouda/mr6d.
>
---
#### [new 026] ResPlan: A Large-Scale Vector-Graph Dataset of 17,000 Residential Floor Plans
- **分类: cs.CV; cs.RO; 68T45**

- **简介: 论文提出ResPlan，一个包含1.7万张住宅平面图的大规模数据集，用于推动空间智能研究。它解决现有数据集规模小、多样性不足的问题，提供高保真度和结构多样性的标注数据，支持机器人、AI生成、VR等应用。**

- **链接: [http://arxiv.org/pdf/2508.14006v1](http://arxiv.org/pdf/2508.14006v1)**

> **作者:** Mohamed Abouagour; Eleftherios Garyfallidis
>
> **备注:** 18 pages, 3 figures, 4 tables
>
> **摘要:** We introduce ResPlan, a large-scale dataset of 17,000 detailed, structurally rich, and realistic residential floor plans, created to advance spatial AI research. Each plan includes precise annotations of architectural elements (walls, doors, windows, balconies) and functional spaces (such as kitchens, bedrooms, and bathrooms). ResPlan addresses key limitations of existing datasets such as RPLAN (Wu et al., 2019) and MSD (van Engelenburg et al., 2024) by offering enhanced visual fidelity and greater structural diversity, reflecting realistic and non-idealized residential layouts. Designed as a versatile, general-purpose resource, ResPlan supports a wide range of applications including robotics, reinforcement learning, generative AI, virtual and augmented reality, simulations, and game development. Plans are provided in both geometric and graph-based formats, enabling direct integration into simulation engines and fast 3D conversion. A key contribution is an open-source pipeline for geometry cleaning, alignment, and annotation refinement. Additionally, ResPlan includes structured representations of room connectivity, supporting graph-based spatial reasoning tasks. Finally, we present comparative analyses with existing benchmarks and outline several open benchmark tasks enabled by ResPlan. Ultimately, ResPlan offers a significant advance in scale, realism, and usability, providing a robust foundation for developing and benchmarking next-generation spatial intelligence systems.
>
---
#### [new 027] A Screw Approach to the Approximation of the Local Geometry of the Configuration Space and of the set of Configurations of Certain Rank of Lower Pair Linkages
- **分类: math.DG; cs.NA; cs.RO; math.NA**

- **简介: 论文提出一种基于螺旋理论的高阶局部运动学分析方法，用于研究低副多环机构配置空间的几何特性及特定秩配置集。解决了传统方法因假设运动光滑而无法处理奇异点的问题，通过泰勒展开和螺钉表示实现局部逼近，适用于文献中未覆盖的尖点等复杂奇异情形。**

- **链接: [http://arxiv.org/pdf/2508.13802v1](http://arxiv.org/pdf/2508.13802v1)**

> **作者:** Andreas Mueller
>
> **摘要:** A motion of a mechanism is a curve in its configuration space (c-space). Singularities of the c-space are kinematic singularities of the mechanism. Any mobility analysis of a particular mechanism amounts to investigating the c-space geometry at a given configuration. A higher-order analysis is necessary to determine the finite mobility. To this end, past research lead to approaches using higher-order time derivatives of loop closure constraints assuming (implicitly) that all possible motions are smooth. This continuity assumption limits the generality of these methods. In this paper an approach to the higher-order local mobility analysis of lower pair multi-loop linkages is presented. This is based on a higher-order Taylor series expansion of the geometric constraint mapping, for which a recursive algebraic expression in terms of joint screws is presented. An exhaustive local analysis includes analysis of the set of constraint singularities (configurations where the constraint Jacobian has certain corank). A local approximation of the set of configurations with certain rank is presented, along with an explicit expression for the differentials of Jacobian minors in terms of instantaneous joint screws. The c-space and the set of points of certain corank are therewith locally approximated by an algebraic variety determined algebraically from the mechanism's screw system. Results are shown for a simple planar 4-bar linkage, which exhibits a bifurcation singularity, and for a planar three-loop linkage exhibiting a cusp in c-space. The latter cannot be treated by the higher-order local analysis methods proposed in the literature.
>
---
#### [new 028] Observed Control -- Linearly Scalable Nonlinear Model Predictive Control with Adaptive Horizons
- **分类: math.OC; cs.RO; cs.SY; eess.SY; 49M29 (Primary) 93B45, 93B52, 93B53 (Secondary)**

- **简介: 论文提出观测控制，利用状态估计与模型预测控制的对偶性，实现线性时间复杂度的非线性模型预测控制，解决计算效率低和时间 horizon 固定的问题。通过卡尔曼平滑器实现高效、稳定控制，并支持任意时刻终止与自适应 horizon。**

- **链接: [http://arxiv.org/pdf/2508.13339v1](http://arxiv.org/pdf/2508.13339v1)**

> **作者:** Eugene T. Hamzezadeh; Andrew J. Petruska
>
> **备注:** 16 pages, 8 figures. Submitted to IEEE Transactions on Automatic Control 8/17/2025
>
> **摘要:** This work highlights the duality between state estimation methods and model predictive control. A predictive controller, observed control, is presented that uses this duality to efficiently compute control actions with linear time-horizon length scalability. The proposed algorithms provide exceptional computational efficiency, adaptive time horizon lengths, and early optimization termination criteria. The use of Kalman smoothers as the backend optimization framework provides for a straightforward implementation supported by strong theoretical guarantees. Additionally, a formulation is presented that separates linear model predictive control into purely reactive and anticipatory components, enabling any-time any-horizon observed control while ensuring controller stability for short time horizons. Finally, numerical case studies confirm that nonlinear filter extensions, i.e., the extended Kalman filter and unscented Kalman filter, effectively extend observed control to nonlinear systems and objectives.
>
---
#### [new 029] Towards No-Code Programming of Cobots: Experiments with Code Synthesis by Large Code Models for Conversational Programming
- **分类: cs.CL; cs.RO**

- **简介: 论文研究用大语言模型实现协作机器人无代码编程，解决传统编程难、手动引导表达力弱的问题。提出RATS任务和数据集，评估LLMs生成指令序列和抽象代码的能力，发现其能生成基础代码但难以处理高级抽象。**

- **链接: [http://arxiv.org/pdf/2409.11041v3](http://arxiv.org/pdf/2409.11041v3)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** Accepted to ITL4HRI workshop at RO-MAN 2025 conference
>
> **摘要:** While there has been a lot of research recently on robots in household environments, at the present time, most robots in existence can be found on shop floors, and most interactions between humans and robots happen there. ``Collaborative robots'' (cobots) designed to work alongside humans on assembly lines traditionally require expert programming, limiting ability to make changes, or manual guidance, limiting expressivity of the resulting programs. To address these limitations, we explore using Large Language Models (LLMs), and in particular, their abilities of doing in-context learning, for conversational code generation. As a first step, we define RATS, the ``Repetitive Assembly Task'', a 2D building task designed to lay the foundation for simulating industry assembly scenarios. In this task, a `programmer' instructs a cobot, using natural language, on how a certain assembly is to be built; that is, the programmer induces a program, through natural language. We create a dataset that pairs target structures with various example instructions (human-authored, template-based, and model-generated) and example code. With this, we systematically evaluate the capabilities of state-of-the-art LLMs for synthesising this kind of code, given in-context examples. Evaluating in a simulated environment, we find that LLMs are capable of generating accurate `first order code' (instruction sequences), but have problems producing `higher-order code' (abstractions such as functions, or use of loops).
>
---
#### [new 030] The 9th AI City Challenge
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文介绍第九届AI City Challenge，聚焦计算机视觉与AI在交通、工业自动化和公共安全中的应用。解决多摄像头3D跟踪、视频问答、空间推理及边缘设备高效检测等问题，通过四个赛道推动技术进步与公平评测。**

- **链接: [http://arxiv.org/pdf/2508.13564v1](http://arxiv.org/pdf/2508.13564v1)**

> **作者:** Zheng Tang; Shuo Wang; David C. Anastasiu; Ming-Ching Chang; Anuj Sharma; Quan Kong; Norimasa Kobori; Munkhjargal Gochoo; Ganzorig Batnasan; Munkh-Erdene Otgonbold; Fady Alnajjar; Jun-Wei Hsieh; Tomasz Kornuta; Xiaolong Li; Yilin Zhao; Han Zhang; Subhashree Radhakrishnan; Arihant Jain; Ratnesh Kumar; Vidya N. Murali; Yuxing Wang; Sameer Satish Pusegaonkar; Yizhou Wang; Sujit Biswas; Xunlei Wu; Zhedong Zheng; Pranamesh Chakraborty; Rama Chellappa
>
> **备注:** Summary of the 9th AI City Challenge Workshop in conjunction with ICCV 2025
>
> **摘要:** The ninth AI City Challenge continues to advance real-world applications of computer vision and AI in transportation, industrial automation, and public safety. The 2025 edition featured four tracks and saw a 17% increase in participation, with 245 teams from 15 countries registered on the evaluation server. Public release of challenge datasets led to over 30,000 downloads to date. Track 1 focused on multi-class 3D multi-camera tracking, involving people, humanoids, autonomous mobile robots, and forklifts, using detailed calibration and 3D bounding box annotations. Track 2 tackled video question answering in traffic safety, with multi-camera incident understanding enriched by 3D gaze labels. Track 3 addressed fine-grained spatial reasoning in dynamic warehouse environments, requiring AI systems to interpret RGB-D inputs and answer spatial questions that combine perception, geometry, and language. Both Track 1 and Track 3 datasets were generated in NVIDIA Omniverse. Track 4 emphasized efficient road object detection from fisheye cameras, supporting lightweight, real-time deployment on edge devices. The evaluation framework enforced submission limits and used a partially held-out test set to ensure fair benchmarking. Final rankings were revealed after the competition concluded, fostering reproducibility and mitigating overfitting. Several teams achieved top-tier results, setting new benchmarks in multiple tasks.
>
---
## 更新

#### [replaced 001] Scaling Up without Fading Out: Goal-Aware Sparse GNN for RL-based Generalized Planning
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.10747v2](http://arxiv.org/pdf/2508.10747v2)**

> **作者:** Sangwoo Jeon; Juchul Shin; Gyeong-Tae Kim; YeonJe Cho; Seongwoo Kim
>
> **摘要:** Generalized planning using deep reinforcement learning (RL) combined with graph neural networks (GNNs) has shown promising results in various symbolic planning domains described by PDDL. However, existing approaches typically represent planning states as fully connected graphs, leading to a combinatorial explosion in edge information and substantial sparsity as problem scales grow, especially evident in large grid-based environments. This dense representation results in diluted node-level information, exponentially increases memory requirements, and ultimately makes learning infeasible for larger-scale problems. To address these challenges, we propose a sparse, goal-aware GNN representation that selectively encodes relevant local relationships and explicitly integrates spatial features related to the goal. We validate our approach by designing novel drone mission scenarios based on PDDL within a grid world, effectively simulating realistic mission execution environments. Our experimental results demonstrate that our method scales effectively to larger grid sizes previously infeasible with dense graph representations and substantially improves policy generalization and success rates. Our findings provide a practical foundation for addressing realistic, large-scale generalized planning tasks.
>
---
#### [replaced 002] MindEye-OmniAssist: A Gaze-Driven LLM-Enhanced Assistive Robot System for Implicit Intention Recognition and Task Execution
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.13250v2](http://arxiv.org/pdf/2503.13250v2)**

> **作者:** Zejia Zhang; Bo Yang; Xinxing Chen; Weizhuang Shi; Haoyuan Wang; Wei Luo; Jian Huang
>
> **摘要:** A promising effective human-robot interaction in assistive robotic systems is gaze-based control. However, current gaze-based assistive systems mainly help users with basic grasping actions, offering limited support. Moreover, the restricted intent recognition capability constrains the assistive system's ability to provide diverse assistance functions. In this paper, we propose an open implicit intention recognition framework powered by Large Language Model (LLM) and Vision Foundation Model (VFM), which can process gaze input and recognize user intents that are not confined to predefined or specific scenarios. Furthermore, we implement a gaze-driven LLM-enhanced assistive robot system (MindEye-OmniAssist) that recognizes user's intentions through gaze and assists in completing task. To achieve this, the system utilizes open vocabulary object detector, intention recognition network and LLM to infer their full intentions. By integrating eye movement feedback and LLM, it generates action sequences to assist the user in completing tasks. Real-world experiments have been conducted for assistive tasks, and the system achieved an overall success rate of 41/55 across various undefined tasks. Preliminary results show that the proposed method holds the potential to provide a more user-friendly human-computer interaction interface and significantly enhance the versatility and effectiveness of assistive systems by supporting more complex and diverse task.
>
---
#### [replaced 003] Adaptive Lattice-based Motion Planning
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2508.02350v2](http://arxiv.org/pdf/2508.02350v2)**

> **作者:** Abhishek Dhar; Sarthak Mishra; Spandan Roy; Daniel Axehill
>
> **摘要:** This paper proposes an adaptive lattice-based motion planning solution to address the problem of generating feasible trajectories for systems, represented by a linearly parameterizable non-linear model operating within a cluttered environment. The system model is considered to have uncertain model parameters. The key idea here is to utilize input/output data online to update the model set containing the uncertain system parameter, as well as a dynamic estimated parameter of the model, so that the associated model estimation error reduces over time. This in turn improves the quality of the motion primitives generated by the lattice-based motion planner using a nominal estimated model selected on the basis of suitable criteria. The motion primitives are also equipped with tubes to account for the model mismatch between the nominal estimated model and the true system model, to guarantee collision-free overall motion. The tubes are of uniform size, which is directly proportional to the size of the model set containing the uncertain system parameter. The adaptive learning module guarantees a reduction in the diameter of the model set as well as in the parameter estimation error between the dynamic estimated parameter and the true system parameter. This directly implies a reduction in the size of the implemented tubes and guarantees that the utilized motion primitives go arbitrarily close to the resolution-optimal motion primitives associated with the true model of the system, thus significantly improving the overall motion planning performance over time. The efficiency of the motion planner is demonstrated by a suitable simulation example that considers a drone model represented by Euler-Lagrange dynamics containing uncertain parameters and operating within a cluttered environment.
>
---
#### [replaced 004] LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11528v5](http://arxiv.org/pdf/2505.11528v5)**

> **作者:** Yuhang Huang; Jiazhao Zhang; Shilong Zou; Xinwang Liu; Ruizhen Hu; Kai Xu
>
> **备注:** CoRL 2025
>
> **摘要:** Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
>
---
#### [replaced 005] On the complexity of constrained reconfiguration and motion planning
- **分类: cs.CC; cs.DM; cs.DS; cs.RO; math.CO**

- **链接: [http://arxiv.org/pdf/2508.13032v2](http://arxiv.org/pdf/2508.13032v2)**

> **作者:** Nicolas Bousquet; Remy El Sabeh; Amer E. Mouawad; Naomi Nishimura
>
> **备注:** Looking to incorporate comments from reviewers
>
> **摘要:** Coordinating the motion of multiple agents in constrained environments is a fundamental challenge in robotics, motion planning, and scheduling. A motivating example involves $n$ robotic arms, each represented as a line segment. The objective is to rotate each arm to its vertical orientation, one at a time (clockwise or counterclockwise), without collisions nor rotating any arm more than once. This scenario is an example of the more general $k$-Compatible Ordering problem, where $n$ agents, each capable of $k$ state-changing actions, must transition to specific target states under constraints encoded as a set $\mathcal{G}$ of $k$ pairs of directed graphs. We show that $k$-Compatible Ordering is $\mathsf{NP}$-complete, even when $\mathcal{G}$ is planar, degenerate, or acyclic. On the positive side, we provide polynomial-time algorithms for cases such as when $k = 1$ or $\mathcal{G}$ has bounded treewidth. We also introduce generalized variants supporting multiple state-changing actions per agent, broadening the applicability of our framework. These results extend to a wide range of scheduling, reconfiguration, and motion planning applications in constrained environments.
>
---
#### [replaced 006] Hybrid Machine Learning Model with a Constrained Action Space for Trajectory Prediction
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.03666v2](http://arxiv.org/pdf/2501.03666v2)**

> **作者:** Alexander Fertig; Lakshman Balasubramanian; Michael Botsch
>
> **备注:** Copyright 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** Trajectory prediction is crucial to advance autonomous driving, improving safety, and efficiency. Although end-to-end models based on deep learning have great potential, they often do not consider vehicle dynamic limitations, leading to unrealistic predictions. To address this problem, this work introduces a novel hybrid model that combines deep learning with a kinematic motion model. It is able to predict object attributes such as acceleration and yaw rate and generate trajectories based on them. A key contribution is the incorporation of expert knowledge into the learning objective of the deep learning model. This results in the constraint of the available action space, thus enabling the prediction of physically feasible object attributes and trajectories, thereby increasing safety and robustness. The proposed hybrid model facilitates enhanced interpretability, thereby reinforcing the trustworthiness of deep learning methods and promoting the development of safe planning solutions. Experiments conducted on the publicly available real-world Argoverse dataset demonstrate realistic driving behaviour, with benchmark comparisons and ablation studies showing promising results.
>
---
#### [replaced 007] MCN-SLAM: Multi-Agent Collaborative Neural SLAM with Hybrid Implicit Neural Scene Representation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18678v2](http://arxiv.org/pdf/2506.18678v2)**

> **作者:** Tianchen Deng; Guole Shen; Xun Chen; Shenghai Yuan; Hongming Shen; Guohao Peng; Zhenyu Wu; Jingchuan Wang; Lihua Xie; Danwei Wang; Hesheng Wang; Weidong Chen
>
> **摘要:** Neural implicit scene representations have recently shown promising results in dense visual SLAM. However, existing implicit SLAM algorithms are constrained to single-agent scenarios, and fall difficulties in large-scale scenes and long sequences. Existing NeRF-based multi-agent SLAM frameworks cannot meet the constraints of communication bandwidth. To this end, we propose the first distributed multi-agent collaborative neural SLAM framework with hybrid scene representation, distributed camera tracking, intra-to-inter loop closure, and online distillation for multiple submap fusion. A novel triplane-grid joint scene representation method is proposed to improve scene reconstruction. A novel intra-to-inter loop closure method is designed to achieve local (single-agent) and global (multi-agent) consistency. We also design a novel online distillation method to fuse the information of different submaps to achieve global consistency. Furthermore, to the best of our knowledge, there is no real-world dataset for NeRF-based/GS-based SLAM that provides both continuous-time trajectories groundtruth and high-accuracy 3D meshes groundtruth. To this end, we propose the first real-world Dense slam (DES) dataset covering both single-agent and multi-agent scenarios, ranging from small rooms to large-scale outdoor scenes, with high-accuracy ground truth for both 3D mesh and continuous-time camera trajectory. This dataset can advance the development of the research in both SLAM, 3D reconstruction, and visual foundation model. Experiments on various datasets demonstrate the superiority of the proposed method in both mapping, tracking, and communication. The dataset and code will open-source on https://github.com/dtc111111/mcnslam.
>
---
#### [replaced 008] MolmoAct: Action Reasoning Models that can Reason in Space
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07917v3](http://arxiv.org/pdf/2508.07917v3)**

> **作者:** Jason Lee; Jiafei Duan; Haoquan Fang; Yuquan Deng; Shuo Liu; Boyang Li; Bohan Fang; Jieyu Zhang; Yi Ru Wang; Sangho Lee; Winson Han; Wilbert Pumacay; Angelica Wu; Rose Hendrix; Karen Farley; Eli VanderBilt; Ali Farhadi; Dieter Fox; Ranjay Krishna
>
> **备注:** Appendix include. Code, Data and Weights: https://allenai.org/blog/molmoact
>
> **摘要:** Reasoning is central to purposeful action, yet most robotic foundation models map perception and instructions directly to control, which limits adaptability, generalization, and semantic grounding. We introduce Action Reasoning Models (ARMs), a class of robotic foundation models that integrate perception, planning, and control through a structured three-stage pipeline. Our model, MolmoAct, encodes observations and instructions into depth-aware perception tokens, generates mid-level spatial plans as editable trajectory traces, and predicts precise low-level actions, enabling explainable and steerable behavior. MolmoAct-7B-D achieves strong performance across simulation and real-world settings: 70.5% zero-shot accuracy on SimplerEnv Visual Matching tasks, surpassing closed-source Pi-0 and GR00T N1; 86.6% average success on LIBERO, including an additional 6.3% gain over ThinkAct on long-horizon tasks; and in real-world fine-tuning, an additional 10% (single-arm) and an additional 22.7% (bimanual) task progression over Pi-0-FAST. It also outperforms baselines by an additional 23.3% on out-of-distribution generalization and achieves top human-preference scores for open-ended instruction following and trajectory steering. Furthermore, we release, for the first time, the MolmoAct Dataset -- a mid-training robot dataset comprising over 10,000 high quality robot trajectories across diverse scenarios and tasks. Training with this dataset yields an average 5.5% improvement in general performance over the base model. We release all model weights, training code, our collected dataset, and our action reasoning dataset, establishing MolmoAct as both a state-of-the-art robotics foundation model and an open blueprint for building ARMs that transform perception into purposeful action through structured reasoning. Blogpost: https://allenai.org/blog/molmoact
>
---
#### [replaced 009] DexSinGrasp: Learning a Unified Policy for Dexterous Object Singulation and Grasping in Densely Cluttered Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04516v2](http://arxiv.org/pdf/2504.04516v2)**

> **作者:** Lixin Xu; Zixuan Liu; Zhewei Gui; Jingxiang Guo; Zeyu Jiang; Tongzhou Zhang; Zhixuan Xu; Chongkai Gao; Lin Shao
>
> **摘要:** Grasping objects in cluttered environments remains a fundamental yet challenging problem in robotic manipulation. While prior works have explored learning-based synergies between pushing and grasping for two-fingered grippers, few have leveraged the high degrees of freedom (DoF) in dexterous hands to perform efficient singulation for grasping in cluttered settings. In this work, we introduce DexSinGrasp, a unified policy for dexterous object singulation and grasping. DexSinGrasp enables high-dexterity object singulation to facilitate grasping, significantly improving efficiency and effectiveness in cluttered environments. We incorporate clutter arrangement curriculum learning to enhance success rates and generalization across diverse clutter conditions, while policy distillation enables a deployable vision-based grasping strategy. To evaluate our approach, we introduce a set of cluttered grasping tasks with varying object arrangements and occlusion levels. Experimental results show that our method outperforms baselines in both efficiency and grasping success rate, particularly in dense clutter. Codes, appendix, and videos are available on our website https://nus-lins-lab.github.io/dexsingweb/.
>
---
#### [replaced 010] Hierarchical Reinforcement Learning in Multi-Goal Spatial Navigation with Autonomous Mobile Robots
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.18794v3](http://arxiv.org/pdf/2504.18794v3)**

> **作者:** Brendon Johnson; Alfredo Weitzenfeld
>
> **摘要:** Hierarchical reinforcement learning (HRL) is hypothesized to be able to leverage the inherent hierarchy in learning tasks where traditional reinforcement learning (RL) often fails. In this research, HRL is evaluated and contrasted with traditional RL in complex robotic navigation tasks. We evaluate unique characteristics of HRL, including its ability to create sub-goals and the termination functions. We constructed a number of experiments to test: 1) the differences between RL proximal policy optimization (PPO) and HRL, 2) different ways of creating sub-goals in HRL, 3) manual vs automatic sub-goal creation in HRL, and 4) the effects of the frequency of termination on performance in HRL. These experiments highlight the advantages of HRL over RL and how it achieves these advantages.
>
---
#### [replaced 011] DISCO: Language-Guided Manipulation with Diffusion Policies and Constrained Inpainting
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2406.09767v3](http://arxiv.org/pdf/2406.09767v3)**

> **作者:** Ce Hao; Kelvin Lin; Zhiwei Xue; Siyuan Luo; Harold Soh
>
> **摘要:** Diffusion policies have demonstrated strong performance in generative modeling, making them promising for robotic manipulation guided by natural language instructions. However, generalizing language-conditioned diffusion policies to open-vocabulary instructions in everyday scenarios remains challenging due to the scarcity and cost of robot demonstration datasets. To address this, we propose DISCO, a framework that leverages off-the-shelf vision-language models (VLMs) to bridge natural language understanding with high-performance diffusion policies. DISCO translates linguistic task descriptions into actionable 3D keyframes using VLMs, which then guide the diffusion process through constrained inpainting. However, enforcing strict adherence to these keyframes can degrade performance when the VLM-generated keyframes are inaccurate. To mitigate this, we introduce an inpainting optimization strategy that balances keyframe adherence with learned motion priors from training data. Experimental results in both simulated and real-world settings demonstrate that DISCO outperforms conventional fine-tuned language-conditioned policies, achieving superior generalization in zero-shot, open-vocabulary manipulation tasks.
>
---
#### [replaced 012] Insights from Interviews with Teachers and Students on the Use of a Social Robot in Computer Science Class in Sixth Grade
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.12946v2](http://arxiv.org/pdf/2508.12946v2)**

> **作者:** Ann-Sophie L. Schenk; Stefan Schiffer; Heqiu Song
>
> **备注:** 4 pages, 2 figures, Late Breaking Report accepted for RO-MAN 2025
>
> **摘要:** In this paper we report on first insights from interviews with teachers and students on using social robots in computer science class in sixth grade. Our focus is on learning about requirements and potential applications. We are particularly interested in getting both perspectives, the teachers' and the learners' view on how robots could be used and what features they should or should not have. Results show that teachers as well as students are very open to robots in the classroom. However, requirements are partially quite heterogeneous among the groups. This leads to complex design challenges which we discuss at the end of this paper.
>
---
#### [replaced 013] Integrating emotional intelligence, memory architecture, and gestures to achieve empathetic humanoid robot interaction in an educational setting
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19803v2](http://arxiv.org/pdf/2505.19803v2)**

> **作者:** Fuze Sun; Lingyu Li; Shixiangyue Meng; Xiaoming Teng; Terry R. Payne; Paul Craig
>
> **摘要:** This study investigates the integration of individual human traits into an empathetically adaptive educational robot tutor system designed to improve student engagement and learning outcomes with corresponding Engagement Vector measurement. While prior research in the field of Human-Robot Interaction (HRI) has examined the integration of the traits, such as emotional intelligence, memory-driven personalization, and non-verbal communication, by themselves, they have thus-far neglected to consider their synchronized integration into a cohesive, operational education framework. To address this gap, we customize a Multi-Modal Large Language Model (LLaMa 3.2 from Meta) deployed with modules for human-like traits (emotion, memory and gestures) into an AI-Agent framework. This constitutes to the robot's intelligent core mimicing the human emotional system, memory architecture and gesture control to allow the robot to behave more empathetically while recognizing and responding appropriately to the student's emotional state. It can also recall the student's past learning record and adapt its style of interaction accordingly. This allows the robot tutor to react to the student in a more sympathetic manner by delivering personalized verbal feedback synchronized with relevant gestures. Our study investigates the extent of this effect through the introduction of Engagement Vector Model which can be a surveyor's pole for judging the quality of HRI experience. Quantitative and qualitative results demonstrate that such an empathetic responsive approach significantly improves student engagement and learning outcomes compared with a baseline humanoid robot without these human-like traits. This indicates that robot tutors with empathetic capabilities can create a more supportive, interactive learning experience that ultimately leads to better outcomes for the student.
>
---
