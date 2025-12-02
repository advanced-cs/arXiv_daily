# 机器人 cs.RO

- **最新发布 102 篇**

- **更新 36 篇**

## 最新发布

#### [new 001] Constant-Time Motion Planning with Manipulation Behaviors
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对机器人操作中运动规划效率与可靠性问题，提出行为常时运动规划（B-CTMP）算法。它在预处理基础上实现毫秒级常时查询，融合碰撞规避与抓取/插入等操作行为，确保半结构化环境中任务成功执行，统一了安全、高效、可验证的操纵规划。**

- **链接: [https://arxiv.org/pdf/2512.00939v1](https://arxiv.org/pdf/2512.00939v1)**

> **作者:** Nayesha Gandotra; Itamar Mishani; Maxim Likhachev
>
> **备注:** In submission
>
> **摘要:** Recent progress in contact-rich robotic manipulation has been striking, yet most deployed systems remain confined to simple, scripted routines. One of the key barriers is the lack of motion planning algorithms that can provide verifiable guarantees for safety, efficiency and reliability. To address this, a family of algorithms called Constant-Time Motion Planning (CTMP) was introduced, which leverages a preprocessing phase to enable collision-free motion queries in a fixed, user-specified time budget (e.g., 10 milliseconds). However, existing CTMP methods do not explicitly incorporate the manipulation behaviors essential for object handling. To bridge this gap, we introduce the \textit{Behavioral Constant-Time Motion Planner} (B-CTMP), an algorithm that extends CTMP to solve a broad class of two-step manipulation tasks: (1) a collision-free motion to a behavior initiation state, followed by (2) execution of a manipulation behavior (such as grasping or insertion) to reach the goal. By precomputing compact data structures, B-CTMP guarantees constant-time query in mere milliseconds while ensuring completeness and successful task execution over a specified set of states. We evaluate B-CTMP on two canonical manipulation tasks in simulation, shelf picking and plug insertion,and demonstrate its effectiveness on a real robot. Our results show that B-CTMP unifies collision-free planning and object manipulation within a single constant-time framework, providing provable guarantees of speed and success for manipulation in semi-structured environments.
>
---
#### [new 002] Modality-Augmented Fine-Tuning of Foundation Robot Policies for Cross-Embodiment Manipulation on GR1 and G1
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人跨体态操作任务，提出一种模态增强微调框架。通过引入接触信号、深度信息等多模态数据，提升基础机器人策略在GR1和G1人形机器人上的泛化能力。实验表明，该方法显著提高操作成功率，验证了模态设计对跨体态迁移的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.01358v1](https://arxiv.org/pdf/2512.01358v1)**

> **作者:** Junsung Park; Hogun Kee; Songhwai Oh
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** This paper presents a modality-augmented fine-tuning framework designed to adapt foundation robot policies to diverse humanoid embodiments. We validate our approach across two distinct settings: (i) the GR1 embodiment, utilizing public datasets where we introduce post-processed modalities, including binary contact signals and ZoeDepth-generated metric depth; and (ii) the Unitree G1 embodiment, for which we contribute a novel multi-modal dataset incorporating cuRobo motion planning, inverse kinematics, and ground-truth contact-force measurements. Our experiments demonstrate that modality augmentation consistently enhances policy performance across different embodiments. Specifically, for the GR1, integrating contact-state cues and RGB-D fusion improves online success rates from 51% to 63%. Furthermore, in the G1 "Pick Apple to Bowl" task, our contact-augmented model achieves a success rate of 94%, significantly outperforming the 48% achieved by standard fine-tuning and the 0% baseline of zero-shot transfer. These results highlight that lightweight post-processing effectively strengthens policies for GR1, while high-quality multi-modal data is crucial for reliable transfer to the Unitree G1. Consequently, this work establishes a unified, data-centric pathway for extending foundation robot policies through targeted modality design and multi-modal fine-tuning.
>
---
#### [new 003] Guardian: Detecting Robotic Planning and Execution Errors with Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中故障检测难题，提出基于视觉语言模型的Guardian系统。通过自动合成失败数据，构建新基准，提升故障检测精度与泛化能力，显著改善机器人任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.01946v1](https://arxiv.org/pdf/2512.01946v1)**

> **作者:** Paul Pacaud; Ricardo Garcia; Shizhe Chen; Cordelia Schmid
>
> **备注:** 9 pages, 9 figures, 6 tables
>
> **摘要:** Robust robotic manipulation requires reliable failure detection and recovery. Although current Vision-Language Models (VLMs) show promise, their accuracy and generalization are limited by the scarcity of failure data. To address this data gap, we propose an automatic robot failure synthesis approach that procedurally perturbs successful trajectories to generate diverse planning and execution failures. This method produces not only binary classification labels but also fine-grained failure categories and step-by-step reasoning traces in both simulation and the real world. With it, we construct three new failure detection benchmarks: RLBench-Fail, BridgeDataV2-Fail, and UR5-Fail, substantially expanding the diversity and scale of existing failure datasets. We then train Guardian, a VLM with multi-view images for detailed failure reasoning and detection. Guardian achieves state-of-the-art performance on both existing and newly introduced benchmarks. It also effectively improves task success rates when integrated into a state-of-the-art manipulation system in simulation and real robots, demonstrating the impact of our generated failure data.
>
---
#### [new 004] Learning Dexterous Manipulation Skills from Imperfect Simulations
- **分类: cs.RO**

- **简介: 该论文研究多指灵巧操作的模拟到现实迁移问题，针对仿真中接触与触觉反馈不准确导致性能下降的难题，提出三阶段框架：先在简化仿真中训练基础策略，再通过人机协作采集含触觉的真实数据，最后基于行为克隆学习融合触觉信息的泛化策略，显著提升任务成功率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.02011v1](https://arxiv.org/pdf/2512.02011v1)**

> **作者:** Elvis Hsieh; Wen-Han Hsieh; Yen-Jen Wang; Toru Lin; Jitendra Malik; Koushil Sreenath; Haozhi Qi
>
> **摘要:** Reinforcement learning and sim-to-real transfer have made significant progress in dexterous manipulation. However, progress remains limited by the difficulty of simulating complex contact dynamics and multisensory signals, especially tactile feedback. In this work, we propose \ours, a sim-to-real framework that addresses these limitations and demonstrates its effectiveness on nut-bolt fastening and screwdriving with multi-fingered hands. The framework has three stages. First, we train reinforcement learning policies in simulation using simplified object models that lead to the emergence of correct finger gaits. We then use the learned policy as a skill primitive within a teleoperation system to collect real-world demonstrations that contain tactile and proprioceptive information. Finally, we train a behavior cloning policy that incorporates tactile sensing and show that it generalizes to nuts and screwdrivers with diverse geometries. Experiments across both tasks show high task progress ratios compared to direct sim-to-real transfer and robust performance even on unseen object shapes and under external perturbations. Videos and code are available on https://dexscrew.github.io.
>
---
#### [new 005] AgriLiRa4D: A Multi-Sensor UAV Dataset for Robust SLAM in Challenging Agricultural Fields
- **分类: cs.RO; eess.SP**

- **简介: 该论文针对农业无人机在复杂环境下的鲁棒定位难题，提出AgriLiRa4D多传感器数据集。涵盖三种农田类型与六种飞行模式，融合高精度真值轨迹及多模态传感器数据，支持多传感器SLAM算法研究与评估，推动农业无人机自主导航技术发展。**

- **链接: [https://arxiv.org/pdf/2512.01753v1](https://arxiv.org/pdf/2512.01753v1)**

> **作者:** Zhihao Zhan; Yuhang Ming; Shaobin Li; Jie Yuan
>
> **摘要:** Multi-sensor Simultaneous Localization and Mapping (SLAM) is essential for Unmanned Aerial Vehicles (UAVs) performing agricultural tasks such as spraying, surveying, and inspection. However, real-world, multi-modal agricultural UAV datasets that enable research on robust operation remain scarce. To address this gap, we present AgriLiRa4D, a multi-modal UAV dataset designed for challenging outdoor agricultural environments. AgriLiRa4D spans three representative farmland types-flat, hilly, and terraced-and includes both boundary and coverage operation modes, resulting in six flight sequence groups. The dataset provides high-accuracy ground-truth trajectories from a Fiber Optic Inertial Navigation System with Real-Time Kinematic capability (FINS_RTK), along with synchronized measurements from a 3D LiDAR, a 4D Radar, and an Inertial Measurement Unit (IMU), accompanied by complete intrinsic and extrinsic calibrations. Leveraging its comprehensive sensor suite and diverse real-world scenarios, AgriLiRa4D supports diverse SLAM and localization studies and enables rigorous robustness evaluation against low-texture crops, repetitive patterns, dynamic vegetation, and other challenges of real agricultural environments. To further demonstrate its utility, we benchmark four state-of-the-art multi-sensor SLAM algorithms across different sensor combinations, highlighting the difficulty of the proposed sequences and the necessity of multi-modal approaches for reliable UAV localization. By filling a critical gap in agricultural SLAM datasets, AgriLiRa4D provides a valuable benchmark for the research community and contributes to advancing autonomous navigation technologies for agricultural UAVs. The dataset can be downloaded from: https://zhan994.github.io/AgriLiRa4D.
>
---
#### [new 006] Autonomous Grasping On Quadruped Robot With Task Level Interaction
- **分类: cs.RO**

- **简介: 该论文研究四足机器人自主抓取任务，旨在解决其缺乏物体操作能力及远程操控复杂的问题。通过集成机械臂与夹爪，构建基于ROS的分层控制系统和网页交互界面，实现导航、目标检测与自主抓取。实验表明系统抓取成功率达75%，有效提升了四足机器人在真实场景中的服务应用潜力。**

- **链接: [https://arxiv.org/pdf/2512.01052v1](https://arxiv.org/pdf/2512.01052v1)**

> **作者:** Muhtadin; Mochammad Hilmi Rusydiansyah; Mauridhi Hery Purnomo; I Ketut Eddy Purnama; Chastine Fatichah
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Quadruped robots are increasingly used in various applications due to their high mobility and ability to operate in diverse terrains. However, most available quadruped robots are primarily focused on mobility without object manipulation capabilities. Equipping a quadruped robot with a robotic arm and gripper introduces a challenge in manual control, especially in remote scenarios that require complex commands. This research aims to develop an autonomous grasping system on a quadruped robot using a task-level interaction approach. The system includes hardware integration of a robotic arm and gripper onto the quadruped robot's body, a layered control system designed using ROS, and a web-based interface for human-robot interaction. The robot is capable of autonomously performing tasks such as navigation, object detection, and grasping using GraspNet. Testing was conducted through real-world scenarios to evaluate navigation, object selection and grasping, and user experience. The results show that the robot can perform tasks accurately and consistently, achieving a grasping success rate of 75 % from 12 trials. Therefore, the system demonstrates significant potential in enhancing the capabilities of quadruped robots as service robots in real-world environments.
>
---
#### [new 007] Transforming Monolithic Foundation Models into Embodied Multi-Agent Architectures for Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文针对服务机器人在人机协作中因单体大模型无法适应动态任务需求的问题，提出InteractGen多智能体框架。通过分解感知、规划、决策等职能为专用智能体，实现基于大模型的协同控制，提升任务成功率与协作能力，验证了多智能体架构优于单一模型扩展。**

- **链接: [https://arxiv.org/pdf/2512.00797v1](https://arxiv.org/pdf/2512.00797v1)**

> **作者:** Nan Sun; Bo Mao; Yongchang Li; Chenxu Wang; Di Guo; Huaping Liu
>
> **备注:** 21 pages, 16 figures, 4 tables
>
> **摘要:** Foundation models have become central to unifying perception and planning in robotics, yet real-world deployment exposes a mismatch between their monolithic assumption that a single model can handle all cognitive functions and the distributed, dynamic nature of practical service workflows. Vision-language models offer strong semantic understanding but lack embodiment-aware action capabilities while relying on hand-crafted skills. Vision-Language-Action policies enable reactive manipulation but remain brittle across embodiments, weak in geometric grounding, and devoid of proactive collaboration mechanisms. These limitations indicate that scaling a single model alone cannot deliver reliable autonomy for service robots operating in human-populated settings. To address this gap, we present InteractGen, an LLM-powered multi-agent framework that decomposes robot intelligence into specialized agents for continuous perception, dependency-aware planning, decision and verification, failure reflection, and dynamic human delegation, treating foundation models as regulated components within a closed-loop collective. Deployed on a heterogeneous robot team and evaluated in a three-month open-use study, InteractGen improves task success, adaptability, and human-robot collaboration, providing evidence that multi-agent orchestration offers a more feasible path toward socially grounded service autonomy than further scaling standalone models.
>
---
#### [new 008] GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出GR-RL框架，解决长时程精细操作中人类示范噪声与次优问题。通过离线与在线强化学习，过滤优化示范、增强泛化，并实现83.3%成功率的自主系鞋带任务，使通用视觉-语言-动作模型转化为高精度专家。**

- **链接: [https://arxiv.org/pdf/2512.01801v1](https://arxiv.org/pdf/2512.01801v1)**

> **作者:** Yunfei Li; Xiao Ma; Jiafeng Xu; Yu Cui; Zhongren Cui; Zhigang Han; Liqun Huang; Tao Kong; Yuxiao Liu; Hao Niu; Wanli Peng; Jingchao Qiao; Zeyu Ren; Haixin Shi; Zhi Su; Jiawen Tian; Yuyang Xiao; Shenyu Zhang; Liwei Zheng; Hang Li; Yonghui Wu
>
> **摘要:** We present GR-RL, a robotic learning framework that turns a generalist vision-language-action (VLA) policy into a highly capable specialist for long-horizon dexterous manipulation. Assuming the optimality of human demonstrations is core to existing VLA policies. However, we claim that in highly dexterous and precise manipulation tasks, human demonstrations are noisy and suboptimal. GR-RL proposes a multi-stage training pipeline that filters, augments, and reinforces the demonstrations by reinforcement learning. First, GR-RL learns a vision-language-conditioned task progress, filters the demonstration trajectories, and only keeps the transitions that contribute positively to the progress. Specifically, we show that by directly applying offline RL with sparse reward, the resulting $Q$-values can be treated as a robust progress function. Next, we introduce morphological symmetry augmentation that greatly improves the generalization and performance of GR-RL. Lastly, to better align the VLA policy with its deployment behaviors for high-precision control, we perform online RL by learning a latent space noise predictor. With this pipeline, GR-RL is, to our knowledge, the first learning-based policy that can autonomously lace up a shoe by threading shoelaces through multiple eyelets with an 83.3% success rate, a task requiring long-horizon reasoning, millimeter-level precision, and compliant soft-body interaction. We hope GR-RL provides a step toward enabling generalist robot foundations models to specialize into reliable real-world experts.
>
---
#### [new 009] Is Image-based Object Pose Estimation Ready to Support Grasping?
- **分类: cs.RO**

- **简介: 该论文针对基于单张RGB图像的6-DoF物体位姿估计任务，评估其在机器人抓取中的实用性。通过物理仿真环境，测试五种开源估计算法在指导夹爪和欠驱动手抓取3D模型时的表现，揭示了当前方法在实际应用中的局限性与潜力。**

- **链接: [https://arxiv.org/pdf/2512.01856v1](https://arxiv.org/pdf/2512.01856v1)**

> **作者:** Eric C. Joyce; Qianwen Zhao; Nathaniel Burgdorfer; Long Wang; Philippos Mordohai
>
> **摘要:** We present a framework for evaluating 6-DoF instance-level object pose estimators, focusing on those that require a single RGB (not RGB-D) image as input. Besides gaining intuition about how accurate these estimators are, we are interested in the degree to which they can serve as the sole perception mechanism for robotic grasping. To assess this, we perform grasping trials in a physics-based simulator, using image-based pose estimates to guide a parallel gripper and an underactuated robotic hand in picking up 3D models of objects. Our experiments on a subset of the BOP (Benchmark for 6D Object Pose Estimation) dataset compare five open-source object pose estimators and provide insights that were missing from the literature.
>
---
#### [new 010] A Cross-Embodiment Gripper Benchmark for Rigid-Object Manipulation in Aerial and Industrial Robotics
- **分类: cs.RO**

- **简介: 该论文针对空中与工业机器人中抓取器跨平台应用的评估难题，提出跨体感抓取基准（CEGB）。解决现有基准无法衡量跨平台迁移性与能耗的问题。构建包含转移时间、能耗和理想载荷的新评估体系，验证了轻量化自锁抓取器在快速换装、低功耗及高成功率方面的优势。**

- **链接: [https://arxiv.org/pdf/2512.01598v1](https://arxiv.org/pdf/2512.01598v1)**

> **作者:** Marek Vagas; Martin Varga; Jaroslav Romancik; Ondrej Majercak; Alejandro Suarez; Anibal Ollero; Bram Vanderborght; Ivan Virgala
>
> **摘要:** Robotic grippers are increasingly deployed across industrial, collaborative, and aerial platforms, where each embodiment imposes distinct mechanical, energetic, and operational constraints. Established YCB and NIST benchmarks quantify grasp success, force, or timing on a single platform, but do not evaluate cross-embodiment transferability or energy-aware performance, capabilities essential for modern mobile and aerial manipulation. This letter introduces the Cross-Embodiment Gripper Benchmark (CEGB), a compact and reproducible benchmarking suite extending YCB and selected NIST metrics with three additional components: a transfer-time benchmark measuring the practical effort required to exchange embodiments, an energy-consumption benchmark evaluating grasping and holding efficiency, and an intent-specific ideal payload assessment reflecting design-dependent operational capability. Together, these metrics characterize both grasp performance and the suitability of reusing a single gripper across heterogeneous robotic systems. A lightweight self-locking gripper prototype is implemented as a reference case. Experiments demonstrate rapid embodiment transfer (median ~= 17.6 s across user groups), low holding energy for gripper prototype (~= 1.5 J per 10 s), and consistent grasp performance with cycle times of 3.2 - 3.9 s and success rates exceeding 90%. CEGB thus provides a reproducible foundation for cross-platform, energy-aware evaluation of grippers in aerial and manipulators domains.
>
---
#### [new 011] Integrated YOLOP Perception and Lyapunov-based Control for Autonomous Mobile Robot Navigation on Track
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对非完整差速移动机器人在轨道上的自主导航任务，提出融合多任务视觉感知与李雅普诺夫稳定性控制的实时框架。通过2D-3D投影与多项式拟合实现车道中心线重建，设计基于李雅普诺夫的稳定控制器，确保在无高精地图和卫星定位条件下，系统具备闭环稳定性和轨迹平滑性。**

- **链接: [https://arxiv.org/pdf/2512.01608v1](https://arxiv.org/pdf/2512.01608v1)**

> **作者:** Mo Chen
>
> **备注:** This is a master's graduation thesis that has not been formally published. Uploaded with the author's copyright permission. No confidential content involved
>
> **摘要:** This work presents a real-time autonomous track navigation framework for nonholonomic differential-drive mobile robots by jointly integrating multi-task visual perception and a provably stable tracking controller. The perception pipeline reconstructs lane centerlines using 2D-to-3D camera projection, arc-length based uniform point resampling, and cubic polynomial fitting solved via robust QR least-squares optimization. The controller regulates robot linear and angular velocities through a Lyapunov-stability grounded design, ensuring bounded error dynamics and asymptotic convergence of position and heading deviations even in dynamic and partially perceived lane scenarios, without relying on HD prior maps or global satellite localization. Real-world experiments on embedded platforms verify system fidelity, real-time execution, trajectory smoothness, and closed-loop stability for reliable autonomous navigation.
>
---
#### [new 012] HAVEN: Hierarchical Adversary-aware Visibility-Enabled Navigation with Cover Utilization using Deep Transformer Q-Networks
- **分类: cs.RO**

- **简介: 该论文提出HAVEN框架，解决部分可观测环境下机器人安全导航问题。通过层次化设计，利用深度变换器Q网络（DTQN）结合时空特征与可见性线索，智能选择子目标并利用掩蔽与暴露惩罚机制优化遮蔽物使用，低层控制器实现平滑避障。在2D/3D仿真中显著提升成功率、安全性与效率。**

- **链接: [https://arxiv.org/pdf/2512.00592v1](https://arxiv.org/pdf/2512.00592v1)**

> **作者:** Mihir Chauhan; Damon Conover; Aniket Bera
>
> **摘要:** Autonomous navigation in partially observable environments requires agents to reason beyond immediate sensor input, exploit occlusion, and ensure safety while progressing toward a goal. These challenges arise in many robotics domains, from urban driving and warehouse automation to defense and surveillance. Classical path planning approaches and memoryless reinforcement learning often fail under limited fields of view (FoVs) and occlusions, committing to unsafe or inefficient maneuvers. We propose a hierarchical navigation framework that integrates a Deep Transformer Q-Network (DTQN) as a high-level subgoal selector with a modular low-level controller for waypoint execution. The DTQN consumes short histories of task-aware features, encoding odometry, goal direction, obstacle proximity, and visibility cues, and outputs Q-values to rank candidate subgoals. Visibility-aware candidate generation introduces masking and exposure penalties, rewarding the use of cover and anticipatory safety. A low-level potential field controller then tracks the selected subgoal, ensuring smooth short-horizon obstacle avoidance. We validate our approach in 2D simulation and extend it directly to a 3D Unity-ROS environment by projecting point-cloud perception into the same feature schema, enabling transfer without architectural changes. Results show consistent improvements over classical planners and RL baselines in success rate, safety margins, and time to goal, with ablations confirming the value of temporal memory and visibility-aware candidate design. These findings highlight a generalizable framework for safe navigation under uncertainty, with broad relevance across robotic platforms.
>
---
#### [new 013] SAGAS: Semantic-Aware Graph-Assisted Stitching for Offline Temporal Logic Planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对离线、无模型的线性时序逻辑（LTL）规划任务，解决仅基于碎片化轨迹数据生成满足复杂逻辑约束路径的问题。提出SAGAS框架，通过构建语义感知的可达图与自动机引导的规划，实现高效路径拼接与执行，成功在离线数据上完成多样化LTL任务。**

- **链接: [https://arxiv.org/pdf/2512.00775v1](https://arxiv.org/pdf/2512.00775v1)**

> **作者:** Ruijia Liu; Ancheng Hou; Shaoyuan Li; Xiang Yin
>
> **摘要:** Linear Temporal Logic (LTL) provides a rigorous framework for complex robotic tasks, yet existing methods often rely on accurate dynamics models or expensive online interactions. In this work, we address LTL-constrained control in a challenging offline, model-free setting, utilizing only fixed, task-agnostic datasets of fragmented trajectories. We propose SAGAS, a novel framework combining graph-assisted trajectory stitching with automata-guided planning. First, we construct a latent reachability graph from a learned temporal-distance representation. To bridge the semantic gap, we augment this graph with certified anchor nodes and probabilistic soft labels. We then translate the specification into a Büchi automaton and search the implicit product space to derive a cost-minimal prefix-suffix plan. Finally, a subgoal-conditioned low-level policy is deployed to execute these latent waypoints. Experiments on OGBench locomotion domains demonstrate that SAGAS successfully synthesizes efficient trajectories for diverse LTL tasks, effectively bridging the gap between fragmented offline data and complex logical constraints.
>
---
#### [new 014] A Novel MDP Decomposition Framework for Scalable UAV Mission Planning in Complex and Uncertain Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对复杂不确定环境下无人机任务规划的计算瓶颈问题，提出一种两阶段MDP分解框架。通过因素分解与优先级重组，将大规模MDP拆解为可独立求解的子问题，并保证全局策略等价性，显著提升计算效率与实时性，实现高效、可靠的任务规划。**

- **链接: [https://arxiv.org/pdf/2512.00838v1](https://arxiv.org/pdf/2512.00838v1)**

> **作者:** Md Muzakkir Quamar; Ali Nasir; Sami ELFerik
>
> **摘要:** This paper presents a scalable and fault-tolerant framework for unmanned aerial vehicle (UAV) mission management in complex and uncertain environments. The proposed approach addresses the computational bottleneck inherent in solving large-scale Markov Decision Processes (MDPs) by introducing a two-stage decomposition strategy. In the first stage, a factor-based algorithm partitions the global MDP into smaller, goal-specific sub-MDPs by leveraging domain-specific features such as goal priority, fault states, spatial layout, and energy constraints. In the second stage, a priority-based recombination algorithm solves each sub-MDP independently and integrates the results into a unified global policy using a meta-policy for conflict resolution. Importantly, we present a theoretical analysis showing that, under mild probabilistic independence assumptions, the combined policy is provably equivalent to the optimal global MDP policy. Our work advances artificial intelligence (AI) decision scalability by decomposing large MDPs into tractable subproblems with provable global equivalence. The proposed decomposition framework enhances the scalability of Markov Decision Processes, a cornerstone of sequential decision-making in artificial intelligence, enabling real-time policy updates for complex mission environments. Extensive simulations validate the effectiveness of our method, demonstrating orders-of-magnitude reduction in computation time without sacrificing mission reliability or policy optimality. The proposed framework establishes a practical and robust foundation for scalable decision-making in real-time UAV mission execution.
>
---
#### [new 015] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding
- **分类: cs.RO**

- **简介: 该论文研究循环任务操作，针对模仿学习中历史信息利用不足导致任务超时的问题，提出CycleManip框架，通过成本感知采样与多任务学习增强历史感知与理解，并构建了首个循环操作基准与自动评估工具，实现端到端高效操控，适用于多种机器人平台。**

- **链接: [https://arxiv.org/pdf/2512.01022v1](https://arxiv.org/pdf/2512.01022v1)**

> **作者:** Yi-Lin Wei; Haoran Liao; Yuhao Lin; Pengyue Wang; Zhizhao Liang; Guiliang Liu; Wei-Shi Zheng
>
> **备注:** Project page: https://isee-laboratory.github.io/OmniDexGrasp/
>
> **摘要:** In this paper, we explore an important yet underexplored task in robot manipulation: cycle-based manipulation, where robots need to perform cyclic or repetitive actions with an expected terminal time. These tasks are crucial in daily life, such as shaking a bottle or knocking a nail. However, few prior works have explored this task, leading to two main challenges: 1) the imitation methods often fail to complete these tasks within the expected terminal time due to the ineffective utilization of history; 2) the absence of a benchmark with sufficient data and automatic evaluation tools hinders development of effective solutions in this area. To address these challenges, we first propose the CycleManip framework to achieve cycle-based task manipulation in an end-to-end imitation manner without requiring any extra models, hierarchical structure or significant computational overhead. The core insight is to enhance effective history perception by a cost-aware sampling strategy and to improve historical understanding by multi-task learning. Second, we introduce a cycle-based task manipulation benchmark, which provides diverse cycle-based tasks, and an automatic evaluation method. Extensive experiments conducted in both simulation and real-world settings demonstrate that our method achieves high success rates in cycle-based task manipulation. The results further show strong adaptability performance in general manipulation, and the plug-and-play ability on imitation policies such as Vision-Language-Action (VLA) models. Moreover, the results show that our approach can be applied across diverse robotic platforms, including bi-arm grippers, dexterous hands, and humanoid robots.
>
---
#### [new 016] Hardware-Software Collaborative Computing of Photonic Spiking Reinforcement Learning for Robotic Continuous Control
- **分类: cs.RO; physics.optics**

- **简介: 该论文针对机器人连续控制中的高能耗与高延迟问题，提出光子脉冲强化学习硬件-软件协同架构。结合硅光MZI芯片与电子域脉冲神经网络，实现高效线性计算与非线性激活，首次在真实任务中应用可编程光子芯片，显著提升能效与速度，验证了其在实时决策中的潜力。**

- **链接: [https://arxiv.org/pdf/2512.00427v1](https://arxiv.org/pdf/2512.00427v1)**

> **作者:** Mengting Yu; Shuiying Xiang; Changjian Xie; Yonghang Chen; Haowen Zhao; Xingxing Guo; Yahui Zhang; Yanan Han; Yue Hao
>
> **摘要:** Robotic continuous control tasks impose stringent demands on the energy efficiency and latency of computing architectures due to their high-dimensional state spaces and real-time interaction requirements. Conventional electronic computing platforms face computational bottlenecks, whereas the fusion of photonic computing and spiking reinforcement learning (RL) offers a promising alternative. Here, we propose a novel computing architecture based on photonic spiking RL, which integrates the Twin Delayed Deep Deterministic policy gradient (TD3) algorithm with spiking neural network (SNN). The proposed architecture employs an optical-electronic hybrid computing paradigm wherein a silicon photonic Mach-Zehnder interferometer (MZI) chip executes linear matrix computations, while nonlinear spiking activations are performed in the electronic domain. Experimental validation on the Pendulum-v1 and HalfCheetah-v2 benchmarks demonstrates the system capability for software-hardware co-inference, achieving a control policy reward of 5831 on HalfCheetah-v2, a 23.33% reduction in convergence steps, and an action deviation below 2.2%. Notably, this work represents the first application of a programmable MZI photonic computing chip to robotic continuous control tasks, attaining an energy efficiency of 1.39 TOPS/W and an ultralow computational latency of 120 ps. Such performance underscores the promise of photonic spiking RL for real-time decision-making in autonomous and industrial robotic systems.
>
---
#### [new 017] H-Zero: Cross-Humanoid Locomotion Pretraining Enables Few-shot Novel Embodiment Transfer
- **分类: cs.RO**

- **简介: 该论文针对人形机器人在不同平台间缺乏通用控制策略的问题，提出H-Zero预训练框架。通过在多种人形机器人上预训练通用基线策略，实现对新机器人零样本与少样本迁移，显著减少调参与训练时间，提升跨平台行走能力。**

- **链接: [https://arxiv.org/pdf/2512.00971v1](https://arxiv.org/pdf/2512.00971v1)**

> **作者:** Yunfeng Lin; Minghuan Liu; Yufei Xue; Ming Zhou; Yong Yu; Jiangmiao Pang; Weinan Zhang
>
> **备注:** in submission, under review
>
> **摘要:** The rapid advancement of humanoid robotics has intensified the need for robust and adaptable controllers to enable stable and efficient locomotion across diverse platforms. However, developing such controllers remains a significant challenge because existing solutions are tailored to specific robot designs, requiring extensive tuning of reward functions, physical parameters, and training hyperparameters for each embodiment. To address this challenge, we introduce H-Zero, a cross-humanoid locomotion pretraining pipeline that learns a generalizable humanoid base policy. We show that pretraining on a limited set of embodiments enables zero-shot and few-shot transfer to novel humanoid robots with minimal fine-tuning. Evaluations show that the pretrained policy maintains up to 81% of the full episode duration on unseen robots in simulation while enabling few-shot transfer to unseen humanoids and upright quadrupeds within 30 minutes of fine-tuning.
>
---
#### [new 018] Foundation Models for Trajectory Planning in Autonomous Driving: A Review of Progress and Open Challenges
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦自动驾驶中的轨迹规划任务，针对传统方法依赖手工设计的局限，综述基于多模态基础模型的新范式。系统梳理37种方法，提出统一分类框架，分析其架构、优势与挑战，并评估开源情况，为研究者提供参考。**

- **链接: [https://arxiv.org/pdf/2512.00021v1](https://arxiv.org/pdf/2512.00021v1)**

> **作者:** Kemal Oksuz; Alexandru Buburuzan; Anthony Knittel; Yuhan Yao; Puneet K. Dokania
>
> **备注:** Under review
>
> **摘要:** The emergence of multi-modal foundation models has markedly transformed the technology for autonomous driving, shifting away from conventional and mostly hand-crafted design choices towards unified, foundation-model-based approaches, capable of directly inferring motion trajectories from raw sensory inputs. This new class of methods can also incorporate natural language as an additional modality, with Vision-Language-Action (VLA) models serving as a representative example. In this review, we provide a comprehensive examination of such methods through a unifying taxonomy to critically evaluate their architectural design choices, methodological strengths, and their inherent capabilities and limitations. Our survey covers 37 recently proposed approaches that span the landscape of trajectory planning with foundation models. Furthermore, we assess these approaches with respect to the openness of their source code and datasets, offering valuable information to practitioners and researchers. We provide an accompanying webpage that catalogs the methods based on our taxonomy, available at: https://github.com/fiveai/FMs-for-driving-trajectories
>
---
#### [new 019] Enhancing Cognitive Robotics with Commonsense through LLM-Generated Preconditions and Subgoals
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究认知机器人在日常任务中的可靠性问题。针对传统符号规划因忽略常识细节而失败的问题，提出利用大语言模型生成隐含前提和子目标，将其转化为形式化规划模型，在仿真中执行。实验表明，该方法显著提升任务成功率与环境适应性，增强了机器人在真实场景下的表现。**

- **链接: [https://arxiv.org/pdf/2512.00069v1](https://arxiv.org/pdf/2512.00069v1)**

> **作者:** Ohad Bachner; Bar Gamliel
>
> **摘要:** Robots often fail at everyday tasks because instructions skip commonsense details like hidden preconditions and small subgoals. Traditional symbolic planners need these details to be written explicitly, which is time consuming and often incomplete. In this project we combine a Large Language Model with symbolic planning. Given a natural language task, the LLM suggests plausible preconditions and subgoals. We translate these suggestions into a formal planning model and execute the resulting plan in simulation. Compared to a baseline planner without the LLM step, our system produces more valid plans, achieves a higher task success rate, and adapts better when the environment changes. These results suggest that adding LLM commonsense to classical planning can make robot behavior in realistic scenarios more reliable.
>
---
#### [new 020] XFlowMP: Task-Conditioned Motion Fields for Generative Robot Planning with Schrodinger Bridges
- **分类: cs.RO**

- **简介: 该论文提出XFlowMP，一种任务条件化的生成式机器人运动规划方法，通过施罗丁格桥建模轨迹演化，融合高阶动力学与起止配置，解决任务语义与运动可控性耦合难题。在多个基准上实现更平滑、节能、快速的轨迹生成，并验证了其在真实机械臂上的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00022v1](https://arxiv.org/pdf/2512.00022v1)**

> **作者:** Khang Nguyen; Minh Nhat Vu
>
> **摘要:** Generative robotic motion planning requires not only the synthesis of smooth and collision-free trajectories but also feasibility across diverse tasks and dynamic constraints. Prior planning methods, both traditional and generative, often struggle to incorporate high-level semantics with low-level constraints, especially the nexus between task configurations and motion controllability. In this work, we present XFlowMP, a task-conditioned generative motion planner that models robot trajectory evolution as entropic flows bridging stochastic noises and expert demonstrations via Schrodinger bridges given the inquiry task configuration. Specifically, our method leverages Schrodinger bridges as a conditional flow matching coupled with a score function to learn motion fields with high-order dynamics while encoding start-goal configurations, enabling the generation of collision-free and dynamically-feasible motions. Through evaluations, XFlowMP achieves up to 53.79% lower maximum mean discrepancy, 36.36% smoother motions, and 39.88% lower energy consumption while comparing to the next-best baseline on the RobotPointMass benchmark, and also reducing short-horizon planning time by 11.72%. On long-horizon motions in the LASA Handwriting dataset, our method maintains the trajectories with 1.26% lower maximum mean discrepancy, 3.96% smoother, and 31.97% lower energy. We further demonstrate the practicality of our method on the Kinova Gen3 manipulator, executing planning motions and confirming its robustness in real-world settings.
>
---
#### [new 021] Causal Reinforcement Learning based Agent-Patient Interaction with Clinical Domain Knowledge
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对医疗干预中数据稀缺、决策需可解释性等问题，提出一种融合因果发现与强化学习的框架（CRL），通过构建患者状态与机器人行为间的因果图，提升决策效率与可解释性。在模拟认知护理场景中，CRL优于传统方法，并实现无需微调的轻量级LLM部署。**

- **链接: [https://arxiv.org/pdf/2512.00048v1](https://arxiv.org/pdf/2512.00048v1)**

> **作者:** Wenzheng Zhao; Ran Zhang; Ruth Palan Lopez; Shu-Fen Wung; Fengpei Yuan
>
> **备注:** Accepted by AAAI workshop
>
> **摘要:** Reinforcement Learning (RL) faces significant challenges in adaptive healthcare interventions, such as dementia care, where data is scarce, decisions require interpretability, and underlying patient-state dynamic are complex and causal in nature. In this work, we present a novel framework called Causal structure-aware Reinforcement Learning (CRL) that explicitly integrates causal discovery and reasoning into policy optimization. This method enables an agent to learn and exploit a directed acyclic graph (DAG) that describes the causal dependencies between human behavioral states and robot actions, facilitating more efficient, interpretable, and robust decision-making. We validate our approach in a simulated robot-assisted cognitive care scenario, where the agent interacts with a virtual patient exhibiting dynamic emotional, cognitive, and engagement states. The experimental results show that CRL agents outperform conventional model-free RL baselines by achieving higher cumulative rewards, maintaining desirable patient states more consistently, and exhibiting interpretable, clinically-aligned behavior. We further demonstrate that CRL's performance advantage remains robust across different weighting strategies and hyperparameter settings. In addition, we demonstrate a lightweight LLM-based deployment: a fixed policy is embedded into a system prompt that maps inferred states to actions, producing consistent, supportive dialogue without LLM finetuning. Our work illustrates the promise of causal reinforcement learning for human-robot interaction applications, where interpretability, adaptiveness, and data efficiency are paramount.
>
---
#### [new 022] A Comprehensive Survey on Surgical Digital Twin
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文综述手术数字孪生（SDT）技术，聚焦多模态数据融合、实时计算与临床落地挑战。提出分类体系，梳理核心进展，对比架构设计，揭示验证、安全、数据治理等关键问题，提出可信、标准化的SDT研究路线，推动从实验室走向临床应用。**

- **链接: [https://arxiv.org/pdf/2512.00019v1](https://arxiv.org/pdf/2512.00019v1)**

> **作者:** Afsah Sharaf Khan; Falong Fan; Doohwan DH Kim; Abdurrahman Alshareef; Dong Chen; Justin Kim; Ernest Carter; Bo Liu; Jerzy W. Rozenblit; Bernard Zeigler
>
> **摘要:** With the accelerating availability of multimodal surgical data and real-time computation, Surgical Digital Twins (SDTs) have emerged as virtual counterparts that mirror, predict, and inform decisions across pre-, intra-, and postoperative care. Despite promising demonstrations, SDTs face persistent challenges: fusing heterogeneous imaging, kinematics, and physiology under strict latency budgets; balancing model fidelity with computational efficiency; ensuring robustness, interpretability, and calibrated uncertainty; and achieving interoperability, privacy, and regulatory compliance in clinical environments. This survey offers a critical, structured review of SDTs. We clarify terminology and scope, propose a taxonomy by purpose, model fidelity, and data sources, and synthesize state-of-the-art achievements in deformable registration and tracking, real-time simulation and co-simulation, AR/VR guidance, edge-cloud orchestration, and AI for scene understanding and prediction. We contrast non-robotic twins with robot-in-the-loop architectures for shared control and autonomy, and identify open problems in validation and benchmarking, safety assurance and human factors, lifecycle "digital thread" integration, and scalable data governance. We conclude with a research agenda toward trustworthy, standards-aligned SDTs that deliver measurable clinical benefit. By unifying vocabulary, organizing capabilities, and highlighting gaps, this work aims to guide SDT design and deployment and catalyze translation from laboratory prototypes to routine surgical care.
>
---
#### [new 023] Perturbation-mitigated USV Navigation with Distributionally Robust Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究无人艇在复杂海洋环境下的鲁棒导航任务，针对异方差观测噪声导致的导航失效问题，提出DRIQN方法。通过结合分布鲁棒优化与隐式分位数网络，显式建模噪声子群，提升恶劣条件下的最坏情况性能。实验表明，该方法显著优于现有方法，在成功率、碰撞率、耗时和能耗上均有大幅提升。**

- **链接: [https://arxiv.org/pdf/2512.00030v1](https://arxiv.org/pdf/2512.00030v1)**

> **作者:** Zhaofan Zhang; Minghao Yang; Sihong Xie; Hui Xiong
>
> **摘要:** The robustness of Unmanned Surface Vehicles (USV) is crucial when facing unknown and complex marine environments, especially when heteroscedastic observational noise poses significant challenges to sensor-based navigation tasks. Recently, Distributional Reinforcement Learning (DistRL) has shown promising results in some challenging autonomous navigation tasks without prior environmental information. However, these methods overlook situations where noise patterns vary across different environmental conditions, hindering safe navigation and disrupting the learning of value functions. To address the problem, we propose DRIQN to integrate Distributionally Robust Optimization (DRO) with implicit quantile networks to optimize worst-case performance under natural environmental conditions. Leveraging explicit subgroup modeling in the replay buffer, DRIQN incorporates heterogeneous noise sources and target robustness-critical scenarios. Experimental results based on the risk-sensitive environment demonstrate that DRIQN significantly outperforms state-of-the-art methods, achieving +13.51\% success rate, -12.28\% collision rate and +35.46\% for time saving, +27.99\% for energy saving, compared with the runner-up.
>
---
#### [new 024] Think Fast: Real-Time Kinodynamic Belief-Space Planning for Projectile Interception
- **分类: cs.RO**

- **简介: 该论文针对高速移动目标拦截任务，解决传感器噪声导致状态不确定性及实时决策难题。提出基于运动学动力学动作基元的树状结构，在状态-时间空间中实现多目标可达性编码与实时价值更新，支持动态信念演化下的无缝目标切换。在6自由度机械臂上验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2512.01108v1](https://arxiv.org/pdf/2512.01108v1)**

> **作者:** Gabriel Olin; Lu Chen; Nayesha Gandotra; Maxim Likhachev; Howie Choset
>
> **摘要:** Intercepting fast moving objects, by its very nature, is challenging because of its tight time constraints. This problem becomes further complicated in the presence of sensor noise because noisy sensors provide, at best, incomplete information, which results in a distribution over target states to be intercepted. Since time is of the essence, to hit the target, the planner must begin directing the interceptor, in this case a robot arm, while still receiving information. We introduce an tree-like structure, which is grown using kinodynamic motion primitives in state-time space. This tree-like structure encodes reachability to multiple goals from a single origin, while enabling real-time value updates as the target belief evolves and seamless transitions between goals. We evaluate our framework on an interception task on a 6 DOF industrial arm (ABB IRB-1600) with an onboard stereo camera (ZED 2i). A robust Innovation-based Adaptive Estimation Adaptive Kalman Filter (RIAE-AKF) is used to track the target and perform belief updates.
>
---
#### [new 025] Real-World Reinforcement Learning of Active Perception Behaviors
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究机器人在部分可观测环境下的主动感知行为学习问题。针对传统方法难以生成有效信息获取行为的挑战，提出AAWR方法，利用训练时的额外传感器信息，结合少量示范和粗略初始化，高效训练出具备主动感知能力的策略，在8个任务中显著优于已有方法。**

- **链接: [https://arxiv.org/pdf/2512.01188v1](https://arxiv.org/pdf/2512.01188v1)**

> **作者:** Edward S. Hu; Jie Wang; Xingfang Yuan; Fiona Luo; Muyao Li; Gaspard Lambrechts; Oleh Rybkin; Dinesh Jayaraman
>
> **备注:** NeurIPS 2025 camera ready
>
> **摘要:** A robot's instantaneous sensory observations do not always reveal task-relevant state information. Under such partial observability, optimal behavior typically involves explicitly acting to gain the missing information. Today's standard robot learning techniques struggle to produce such active perception behaviors. We propose a simple real-world robot learning recipe to efficiently train active perception policies. Our approach, asymmetric advantage weighted regression (AAWR), exploits access to "privileged" extra sensors at training time. The privileged sensors enable training high-quality privileged value functions that aid in estimating the advantage of the target policy. Bootstrapping from a small number of potentially suboptimal demonstrations and an easy-to-obtain coarse policy initialization, AAWR quickly acquires active perception behaviors and boosts task performance. In evaluations on 8 manipulation tasks on 3 robots spanning varying degrees of partial observability, AAWR synthesizes reliable active perception behaviors that outperform all prior approaches. When initialized with a "generalist" robot policy that struggles with active perception tasks, AAWR efficiently generates information-gathering behaviors that allow it to operate under severe partial observability for manipulation tasks. Website: https://penn-pal-lab.github.io/aawr/
>
---
#### [new 026] Supporting Productivity Skill Development in College Students through Social Robot Coaching: A Proof-of-Concept
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文提出一种社交助手机器人（SAR）作为大学生时间管理与任务优先级的教育教练，解决传统工具互动性差、辅导资源不足的问题。通过六节课程、聊天交互与数据仪表盘，实现个性化反馈与自我反思。15名学生测试显示系统易用且有效，验证了其在提升学业生产力方面的可行性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.01105v1](https://arxiv.org/pdf/2512.01105v1)**

> **作者:** Himanshi Lalwani; Hanan Salam
>
> **摘要:** College students often face academic challenges that hamper their productivity and well-being. Although self-help books and productivity apps are popular, they often fall short. Books provide generalized, non-interactive guidance, and apps are not inherently educational and can hinder the development of key organizational skills. Traditional productivity coaching offers personalized support, but is resource-intensive and difficult to scale. In this study, we present a proof-of-concept for a socially assistive robot (SAR) as an educational coach and a potential solution to the limitations of existing productivity tools and coaching approaches. The SAR delivers six different lessons on time management and task prioritization. Users interact via a chat interface, while the SAR responds through speech (with a toggle option). An integrated dashboard monitors progress, mood, engagement, confidence per lesson, and time spent per lesson. It also offers personalized productivity insights to foster reflection and self-awareness. We evaluated the system with 15 college students, achieving a System Usability Score of 79.2 and high ratings for overall experience and engagement. Our findings suggest that SAR-based productivity coaching can offer an effective and scalable solution to improve productivity among college students.
>
---
#### [new 027] Estimation of Kinematic Motion from Dashcam Footage
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究基于行车记录仪视频估计车辆运动参数的任务，旨在评估其预测车速、航向角及前车相对位置与速度的准确性。通过同步车载CAN数据与摄像头视频，构建神经网络模型，并提供开源工具与方法供他人复现数据采集与实验。**

- **链接: [https://arxiv.org/pdf/2512.01104v1](https://arxiv.org/pdf/2512.01104v1)**

> **作者:** Evelyn Zhang; Alex Richardson; Jonathan Sprinkle
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** The goal of this paper is to explore the accuracy of dashcam footage to predict the actual kinematic motion of a car-like vehicle. Our approach uses ground truth information from the vehicle's on-board data stream, through the controller area network, and a time-synchronized dashboard camera, mounted to a consumer-grade vehicle, for 18 hours of footage and driving. The contributions of the paper include neural network models that allow us to quantify the accuracy of predicting the vehicle speed and yaw, as well as the presence of a lead vehicle, and its relative distance and speed. In addition, the paper describes how other researchers can gather their own data to perform similar experiments, using open-source tools and off-the-shelf technology.
>
---
#### [new 028] Discovering Self-Protective Falling Policy for Humanoid Robot via Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文研究人形机器人防跌落策略，针对其易坠落且坠落损伤大的问题，采用深度强化学习与课程学习，设计奖励函数与多样化训练环境，使机器人自主发现通过形成“三角”结构降低坠落损伤的自保护行为，并成功实现从仿真到现实平台的迁移。**

- **链接: [https://arxiv.org/pdf/2512.01336v1](https://arxiv.org/pdf/2512.01336v1)**

> **作者:** Diyuan Shi; Shangke Lyu; Donglin Wang
>
> **摘要:** Humanoid robots have received significant research interests and advancements in recent years. Despite many successes, due to their morphology, dynamics and limitation of control policy, humanoid robots are prone to fall as compared to other embodiments like quadruped or wheeled robots. And its large weight, tall Center of Mass, high Degree-of-Freedom would cause serious hardware damages when falling uncontrolled, to both itself and surrounding objects. Existing researches in this field mostly focus on using control based methods that struggle to cater diverse falling scenarios and may introduce unsuitable human prior. On the other hand, large-scale Deep Reinforcement Learning and Curriculum Learning could be employed to incentivize humanoid agent discovering falling protection policy that fits its own nature and property. In this work, with carefully designed reward functions and domain diversification curriculum, we successfully train humanoid agent to explore falling protection behaviors and discover that by forming a `triangle' structure, the falling damages could be significantly reduced with its rigid-material body. With comprehensive metrics and experiments, we quantify its performance with comparison to other methods, visualize its falling behaviors and successfully transfer it to real world platform.
>
---
#### [new 029] Sign Language Recognition using Bidirectional Reservoir Computing
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手语识别任务，旨在解决深度学习模型计算资源消耗大、不适用于边缘设备的问题。提出基于MediaPipe与双向储备池计算（BRC）的高效识别方法，利用手部关节点特征，通过双向处理捕捉时序依赖，实现9秒训练时间与57.71%准确率，显著优于传统深度学习方法。**

- **链接: [https://arxiv.org/pdf/2512.00777v1](https://arxiv.org/pdf/2512.00777v1)**

> **作者:** Nitin Kumar Singh; Arie Rachmad Syulistyo; Yuichiro Tanaka; Hakaru Tamukoh
>
> **摘要:** Sign language recognition (SLR) facilitates communication between deaf and hearing individuals. Deep learning is widely used to develop SLR-based systems; however, it is computationally intensive and requires substantial computational resources, making it unsuitable for resource-constrained devices. To address this, we propose an efficient sign language recognition system using MediaPipe and an echo state network (ESN)-based bidirectional reservoir computing (BRC) architecture. MediaPipe extracts hand joint coordinates, which serve as inputs to the ESN-based BRC architecture. The BRC processes these features in both forward and backward directions, efficiently capturing temporal dependencies. The resulting states of BRC are concatenated to form a robust representation for classification. We evaluated our method on the Word-Level American Sign Language (WLASL) video dataset, achieving a competitive accuracy of 57.71% and a significantly lower training time of only 9 seconds, in contrast to the 55 minutes and $38$ seconds required by the deep learning-based Bi-GRU approach. Consequently, the BRC-based SLR system is well-suited for edge devices.
>
---
#### [new 030] RoboLoc: A Benchmark Dataset for Point Place Recognition and Localization in Indoor-Outdoor Integrated Environments
- **分类: cs.RO**

- **简介: 该论文提出RoboLoc数据集，用于解决机器人在室内外混合环境中进行无GPS定位的挑战。针对现有数据集多聚焦室外、缺乏域间平滑过渡的问题，构建包含真实轨迹与复杂地形的多场景数据，评估点、体素及BEV等模型在跨域场景下的泛化能力，为多环境定位系统提供基准测试平台。**

- **链接: [https://arxiv.org/pdf/2512.01194v1](https://arxiv.org/pdf/2512.01194v1)**

> **作者:** Jaejin Jeon; Seonghoon Ryoo; Sang-Duck Lee; Soomok Lee; Seungwoo Jeong
>
> **摘要:** Robust place recognition is essential for reliable localization in robotics, particularly in complex environments with fre- quent indoor-outdoor transitions. However, existing LiDAR-based datasets often focus on outdoor scenarios and lack seamless domain shifts. In this paper, we propose RoboLoc, a benchmark dataset designed for GPS-free place recognition in indoor-outdoor environments with floor transitions. RoboLoc features real-world robot trajectories, diverse elevation profiles, and transitions between structured indoor and unstructured outdoor domains. We benchmark a variety of state-of-the-art models, point-based, voxel-based, and BEV-based architectures, highlighting their generalizability domain shifts. RoboLoc provides a realistic testbed for developing multi-domain localization systems in robotics and autonomous navigation
>
---
#### [new 031] Modeling and Control of Magnetic Forces between Microrobots
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究多微机器人在共享磁信号下的独立控制问题，旨在实现两微机器人间径向距离的精确调控。针对现有系统仅支持同步运动的局限，提出级联PID+PD控制策略，利用磁场角度ψ调节距离，显著缩短响应时间并平滑轨迹，验证了2D环境下快速精准控制的可行性，为生物医学应用提供新方案。**

- **链接: [https://arxiv.org/pdf/2512.00051v1](https://arxiv.org/pdf/2512.00051v1)**

> **作者:** Amelia Fernández Seguel; Alejandro I. Maass
>
> **备注:** 38 pages, 10 figures
>
> **摘要:** The independent control of multiple magnetic microrobots under a shared global signal presents critical challenges in biomedical applications such as targeted drug delivery and microsurgeries. Most existing systems only allow all agents to move synchronously, limiting their use in applications that require differentiated actuation. This research aims to design a controller capable of regulating the radial distance between micro-agents using only the angle ψof a global magnetic field as the actuation parameter, demonstrating potential for practical applications. The proposed cascade control approach enables faster and more precise adjustment of the inter-agent distance than a proportional controller, while maintaining smooth transitions and avoiding abrupt changes in the orientation of the magnetic field, making it suitable for real-world implementation. A bibliographic review was conducted to develop the physical model, considering magnetic dipole-dipole interactions and velocities in viscous media. A PID controller was implemented to regulate the radial distance, followed by a PD controller in cascade to smooth changes in field orientation. These controllers were simulated in MATLAB, showing that the PID controller reduced convergence time to the desired radius by about 40%. When adding the second controller, the combined PID+PD scheme achieved smooth angular trajectories within similar timeframes, with fluctuations of only \pm 5^\circ. These results validate the feasibility of controlling the radial distance of two microrobots using a shared magnetic field in a fast and precise manner, without abrupt variations in the control angle. However, the model is limited to a 2D environment and two agents, suggesting future research to extend the controller to 3D systems and multiple agents.
>
---
#### [new 032] FOM-Nav: Frontier-Object Maps for Object Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对未知环境中寻物导航任务，解决现有方法在长期记忆与语义信息不足的问题。提出FOM-Nav框架，通过在线构建融合空间前沿与细粒度物体信息的前沿-物体地图，结合视觉语言模型实现高层目标预测与高效路径规划，显著提升导航效率，在多个基准上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2512.01009v1](https://arxiv.org/pdf/2512.01009v1)**

> **作者:** Thomas Chabal; Shizhe Chen; Jean Ponce; Cordelia Schmid
>
> **备注:** Project page: https://www.di.ens.fr/willow/research/fom-nav/
>
> **摘要:** This paper addresses the Object Goal Navigation problem, where a robot must efficiently find a target object in an unknown environment. Existing implicit memory-based methods struggle with long-term memory retention and planning, while explicit map-based approaches lack rich semantic information. To address these challenges, we propose FOM-Nav, a modular framework that enhances exploration efficiency through Frontier-Object Maps and vision-language models. Our Frontier-Object Maps are built online and jointly encode spatial frontiers and fine-grained object information. Using this representation, a vision-language model performs multimodal scene understanding and high-level goal prediction, which is executed by a low-level planner for efficient trajectory generation. To train FOM-Nav, we automatically construct large-scale navigation datasets from real-world scanned environments. Extensive experiments validate the effectiveness of our model design and constructed dataset. FOM-Nav achieves state-of-the-art performance on the MP3D and HM3D benchmarks, particularly in navigation efficiency metric SPL, and yields promising results on a real robot.
>
---
#### [new 033] MILE: A Mechanically Isomorphic Exoskeleton Data Collection System with Fingertip Visuotactile Sensing for Dexterous Manipulation
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文针对灵巧操作中高质量数据缺失问题，提出MILE系统，实现人体手与机器人手的机械同构，通过无畸变运动映射和高分辨率指尖触觉视觉传感，高效采集多模态数据，显著提升操作成功率，推动模仿学习在精细操作中的应用。**

- **链接: [https://arxiv.org/pdf/2512.00324v1](https://arxiv.org/pdf/2512.00324v1)**

> **作者:** Jinda Du; Jieji Ren; Qiaojun Yu; Ningbin Zhang; Yu Deng; Xingyu Wei; Yufei Liu; Guoying Gu; Xiangyang Zhu
>
> **摘要:** Imitation learning provides a promising approach to dexterous hand manipulation, but its effectiveness is limited by the lack of large-scale, high-fidelity data. Existing data-collection pipelines suffer from inaccurate motion retargeting, low data-collection efficiency, and missing high-resolution fingertip tactile sensing. We address this gap with MILE, a mechanically isomorphic teleoperation and data-collection system co-designed from human hand to exoskeleton to robotic hand. The exoskeleton is anthropometrically derived from the human hand, and the robotic hand preserves one-to-one joint-position isomorphism, eliminating nonlinear retargeting and enabling precise, natural control. The exoskeleton achieves a multi-joint mean absolute angular error below one degree, while the robotic hand integrates compact fingertip visuotactile modules that provide high-resolution tactile observations. Built on this retargeting-free interface, we teleoperate complex, contact-rich in-hand manipulation and efficiently collect a multimodal dataset comprising high-resolution fingertip visuotactile signals, RGB-D images, and joint positions. The teleoperation pipeline achieves a mean success rate improvement of 64%. Incorporating fingertip tactile observations further increases the success rate by an average of 25% over the vision-only baseline, validating the fidelity and utility of the dataset. Further details are available at: https://sites.google.com/view/mile-system.
>
---
#### [new 034] Intelligent Systems and Robotics: Revolutionizing Engineering Industries
- **分类: cs.RO**

- **简介: 该论文属于综述性研究，旨在探讨智能系统与机器人技术在工程领域的应用。针对制造业、土木、电气和机械工程中效率、精度与适应性提升的需求，分析AI、机器学习与自主机器人技术的影响，评估其对生产率、安全性和成本的作用，并总结成效与挑战，提出评价方法，推动工程方法创新。**

- **链接: [https://arxiv.org/pdf/2512.00033v1](https://arxiv.org/pdf/2512.00033v1)**

> **作者:** Sathish Krishna Anumula; Sivaramkumar Ponnarangan; Faizal Nujumudeen; Ms. Nilakshi Deka; S. Balamuralitharan; M Venkatesh
>
> **备注:** 9 Pages, 4 figures
>
> **摘要:** A mix of intelligent systems and robotics is making engineering industries much more efficient, precise and able to adapt. How artificial intelligence (AI), machine learning (ML) and autonomous robotic technologies are changing manufacturing, civil, electrical and mechanical engineering is discussed in this paper. Based on recent findings and a suggested way to evaluate intelligent robotic systems in industry, we give an overview of how their use impacts productivity, safety and operational costs. Experience and case studies confirm the benefits this area brings and the problems that have yet to be solved. The findings indicate that intelligent robotics involves more than a technology change; it introduces important new methods in engineering.
>
---
#### [new 035] Balancing Efficiency and Fairness: An Iterative Exchange Framework for Multi-UAV Cooperative Path Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对多无人机协同路径规划（MUCPP）任务，解决效率与公平性难以兼顾的问题。提出一种迭代交换框架，通过联合优化总路程与完工时间，基于A*算法生成避障轨迹，在多地形数据集上实现更优的效率-公平性权衡。**

- **链接: [https://arxiv.org/pdf/2512.00410v1](https://arxiv.org/pdf/2512.00410v1)**

> **作者:** Hongzong Li; Luwei Liao; Xiangguang Dai; Yuming Feng; Rong Feng; Shiqin Tang
>
> **摘要:** Multi-UAV cooperative path planning (MUCPP) is a fundamental problem in multi-agent systems, aiming to generate collision-free trajectories for a team of unmanned aerial vehicles (UAVs) to complete distributed tasks efficiently. A key challenge lies in achieving both efficiency, by minimizing total mission cost, and fairness, by balancing the workload among UAVs to avoid overburdening individual agents. This paper presents a novel Iterative Exchange Framework for MUCPP, balancing efficiency and fairness through iterative task exchanges and path refinements. The proposed framework formulates a composite objective that combines the total mission distance and the makespan, and iteratively improves the solution via local exchanges under feasibility and safety constraints. For each UAV, collision-free trajectories are generated using A* search over a terrain-aware configuration space. Comprehensive experiments on multiple terrain datasets demonstrate that the proposed method consistently achieves superior trade-offs between total distance and makespan compared to existing baselines.
>
---
#### [new 036] Ethically-Aware Participatory Design of a Productivity Social Robot for College Students
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互与伦理设计任务，旨在解决大学生（尤其ADHD群体）因执行功能障碍导致的生产力低下问题。通过参与式设计，联合学生与教练开展访谈与工作坊，识别需求并设计具伦理意识的助产型社交机器人，提出功能特征与伦理准则，推动负责任的技术应用。**

- **链接: [https://arxiv.org/pdf/2512.01111v1](https://arxiv.org/pdf/2512.01111v1)**

> **作者:** Himanshi Lalwani; Hanan Salam
>
> **摘要:** College students often face academic and life stressors affecting productivity, especially students with Attention Deficit Hyperactivity Disorder (ADHD) who experience executive functioning challenges. Conventional productivity tools typically demand sustained self-discipline and consistent use, which many students struggle with, leading to disruptive app-switching behaviors. Socially Assistive Robots (SARs), known for their intuitive and interactive nature, offer promising potential to support productivity in academic environments, having been successfully utilized in domains like education, cognitive development, and mental health. To leverage SARs effectively in addressing student productivity, this study employed a Participatory Design (PD) approach, directly involving college students and a Student Success and Well-Being Coach in the design process. Through interviews and a collaborative workshop, we gathered detailed insights on productivity challenges and identified desirable features for a productivity-focused SAR. Importantly, ethical considerations were integrated from the onset, facilitating responsible and user-aligned design choices. Our contributions include comprehensive insights into student productivity challenges, SAR design preferences, and actionable recommendations for effective robot characteristics. Additionally, we present stakeholder-derived ethical guidelines to inform responsible future implementations of productivity-focused SARs in higher education.
>
---
#### [new 037] VISTAv2: World Imagination for Indoor Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉语言导航（VLN）任务，解决现有方法缺乏在线动作条件预测与显式规划价值的问题。提出VISTAv2，通过条件扩散Transformer生成动作相关的未来视图，结合指令引导融合为在线价值图，提升导航的鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.00041v1](https://arxiv.org/pdf/2512.00041v1)**

> **作者:** Yanjia Huang; Xianshun Jiang; Xiangbo Gao; Mingyang Wu; Zhengzhong Tu
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to follow language instructions while acting in continuous real-world spaces. Prior image imagination based VLN work shows benefits for discrete panoramas but lacks online, action-conditioned predictions and does not produce explicit planning values; moreover, many methods replace the planner with long-horizon objectives that are brittle and slow. To bridge this gap, we propose VISTAv2, a generative world model that rolls out egocentric future views conditioned on past observations, candidate action sequences, and instructions, and projects them into an online value map for planning. Unlike prior approaches, VISTAv2 does not replace the planner. The online value map is fused at score level with the base objective, providing reachability and risk-aware guidance. Concretely, we employ an action-aware Conditional Diffusion Transformer video predictor to synthesize short-horizon futures, align them with the natural language instruction via a vision-language scorer, and fuse multiple rollouts in a differentiable imagination-to-value head to output an imagined egocentric value map. For efficiency, rollouts occur in VAE latent space with a distilled sampler and sparse decoding, enabling inference on a single consumer GPU. Evaluated on MP3D and RoboTHOR, VISTAv2 improves over strong baselines, and ablations show that action-conditioned imagination, instruction-guided value fusion, and the online value-map planner are all critical, suggesting that VISTAv2 offers a practical and interpretable route to robust VLN.
>
---
#### [new 038] Tactile Robotics: Past and Future
- **分类: cs.RO**

- **简介: 该论文回顾了近50年触觉机器人发展史，梳理四代演进脉络，分析技术瓶颈与机遇。通过整合150篇综述，揭示触觉传感在机器人中的关键挑战与未来方向，聚焦e-皮肤、仿生触觉、触觉互联网等新兴领域，旨在为触觉机器人迈向商业化应用提供理论指引。**

- **链接: [https://arxiv.org/pdf/2512.01106v1](https://arxiv.org/pdf/2512.01106v1)**

> **作者:** Nathan F. Lepora
>
> **备注:** Accepted in International Journal of Robotics Research (IJRR)
>
> **摘要:** What is the future of tactile robotics? To help define that future, this article provides a historical perspective on tactile sensing in robotics from the wealth of knowledge and expert opinion in nearly 150 reviews over almost half a century. This history is characterized by a succession of generations: 1965-79 (origins), 1980-94 (foundations and growth), 1995-2009 (tactile winter) and 2010-2024 (expansion and diversification). Recent expansion has led to diverse themes emerging of e-skins, tactile robotic hands, vision-based tactile sensing, soft/biomimetic touch, and the tactile Internet. In the next generation from 2025, tactile robotics could mature to widespread commercial use, with applications in human-like dexterity, understanding human intelligence, and telepresence impacting all robotics and AI. By linking past expert insights to present themes, this article highlights recurring challenges in tactile robotics, showing how the field has evolved, why progress has often stalled, and which opportunities are most likely to define its future.
>
---
#### [new 039] EfficientFlow: Efficient Equivariant Flow Policy Learning for Embodied AI
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对具身智能中的视觉-运动策略学习问题，解决生成式策略数据效率低、采样慢的难题。提出EfficientFlow框架，通过引入等变性提升数据效率，并设计加速正则化策略，实现高效训练与快速推理，在有限数据下取得优异性能。**

- **链接: [https://arxiv.org/pdf/2512.02020v1](https://arxiv.org/pdf/2512.02020v1)**

> **作者:** Jianlei Chang; Ruofeng Mei; Wei Ke; Xiangyu Xu
>
> **备注:** Accepted by AAAI 2026. Project Page: https://efficientflow.github.io/
>
> **摘要:** Generative modeling has recently shown remarkable promise for visuomotor policy learning, enabling flexible and expressive control across diverse embodied AI tasks. However, existing generative policies often struggle with data inefficiency, requiring large-scale demonstrations, and sampling inefficiency, incurring slow action generation during inference. We introduce EfficientFlow, a unified framework for efficient embodied AI with flow-based policy learning. To enhance data efficiency, we bring equivariance into flow matching. We theoretically prove that when using an isotropic Gaussian prior and an equivariant velocity prediction network, the resulting action distribution remains equivariant, leading to improved generalization and substantially reduced data demands. To accelerate sampling, we propose a novel acceleration regularization strategy. As direct computation of acceleration is intractable for marginal flow trajectories, we derive a novel surrogate loss that enables stable and scalable training using only conditional trajectories. Across a wide range of robotic manipulation benchmarks, the proposed algorithm achieves competitive or superior performance under limited data while offering dramatically faster inference. These results highlight EfficientFlow as a powerful and efficient paradigm for high-performance embodied AI.
>
---
#### [new 040] NeuroHJR: Hamilton-Jacobi Reachability-based Obstacle Avoidance in Complex Environments with Physics-Informed Neural Networks
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对复杂环境中自主车辆的实时避障问题，提出NeuroHJR框架，利用物理信息神经网络（PINN）逼近哈密顿-雅可比可达性（HJR）解，避免网格离散化，实现连续状态空间下高效可达集估计，显著降低计算成本，同时保持与传统HJR相当的安全性能。**

- **链接: [https://arxiv.org/pdf/2512.01897v1](https://arxiv.org/pdf/2512.01897v1)**

> **作者:** Granthik Halder; Rudrashis Majumder; Rakshith M R; Rahi Shah; Suresh Sundaram
>
> **备注:** Author-accepted version. Accepted at IEEE 11th Indian Control Conference (ICC), 2025
>
> **摘要:** Autonomous ground vehicles (AGVs) must navigate safely in cluttered environments while accounting for complex dynamics and environmental uncertainty. Hamilton-Jacobi Reachability (HJR) offers formal safety guarantees through the computation of forward and backward reachable sets, but its application is hindered by poor scalability in environments with numerous obstacles. In this paper, we present a novel framework called NeuroHJR that leverages Physics-Informed Neural Networks (PINNs) to approximate the HJR solution for real-time obstacle avoidance. By embedding system dynamics and safety constraints directly into the neural network loss function, our method bypasses the need for grid-based discretization and enables efficient estimation of reachable sets in continuous state spaces. We demonstrate the effectiveness of our approach through simulation results in densely cluttered scenarios, showing that it achieves safety performance comparable to that of classical HJR solvers while significantly reducing the computational cost. This work provides a new step toward real-time, scalable deployment of reachability-based obstacle avoidance in robotics.
>
---
#### [new 041] Sample-Efficient Expert Query Control in Active Imitation Learning via Conformal Prediction
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对主动模仿学习中的专家查询成本过高问题，提出CRSAIL方法。通过基于共形预测的拒绝采样策略，仅在状态稀缺时查询专家，显著降低查询次数（最多减少96%），同时保持专家级性能，且无需实时专家干预，具备良好鲁棒性与可迁移性。**

- **链接: [https://arxiv.org/pdf/2512.00453v1](https://arxiv.org/pdf/2512.00453v1)**

> **作者:** Arad Firouzkouhi; Omid Mirzaeedodangeh; Lars Lindemann
>
> **摘要:** Active imitation learning (AIL) combats covariate shift by querying an expert during training. However, expert action labeling often dominates the cost, especially in GPU-intensive simulators, human-in-the-loop settings, and robot fleets that revisit near-duplicate states. We present Conformalized Rejection Sampling for Active Imitation Learning (CRSAIL), a querying rule that requests an expert action only when the visited state is under-represented in the expert-labeled dataset. CRSAIL scores state novelty by the distance to the $K$-th nearest expert state and sets a single global threshold via conformal prediction. This threshold is the empirical $(1-α)$ quantile of on-policy calibration scores, providing a distribution-free calibration rule that links $α$ to the expected query rate and makes $α$ a task-agnostic tuning knob. This state-space querying strategy is robust to outliers and, unlike safety-gate-based AIL, can be run without real-time expert takeovers: we roll out full trajectories (episodes) with the learner and only afterward query the expert on a subset of visited states. Evaluated on MuJoCo robotics tasks, CRSAIL matches or exceeds expert-level reward while reducing total expert queries by up to 96% vs. DAgger and up to 65% vs. prior AIL methods, with empirical robustness to $α$ and $K$, easing deployment on novel systems with unknown dynamics.
>
---
#### [new 042] NavForesee: A Unified Vision-Language World Model for Hierarchical Planning and Dual-Horizon Navigation Prediction
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对长时序复杂语言指令下的具身导航任务，解决现有模型在未知环境中的长期规划与预测能力不足问题。提出NavForesee，一个统一视觉-语言世界模型，融合显式语言规划与隐式时空预测，实现层级规划与双时间尺度导航预测，通过感知-规划/预测-行动的闭环提升导航性能。**

- **链接: [https://arxiv.org/pdf/2512.01550v1](https://arxiv.org/pdf/2512.01550v1)**

> **作者:** Fei Liu; Shichao Xie; Minghua Luo; Zedong Chu; Junjun Hu; Xiaolong Wu; Mu Xu
>
> **摘要:** Embodied navigation for long-horizon tasks, guided by complex natural language instructions, remains a formidable challenge in artificial intelligence. Existing agents often struggle with robust long-term planning about unseen environments, leading to high failure rates. To address these limitations, we introduce NavForesee, a novel Vision-Language Model (VLM) that unifies high-level language planning and predictive world model imagination within a single, unified framework. Our approach empowers a single VLM to concurrently perform planning and predictive foresight. Conditioned on the full instruction and historical observations, the model is trained to understand the navigation instructions by decomposing the task, tracking its progress, and formulating the subsequent sub-goal. Simultaneously, it functions as a generative world model, providing crucial foresight by predicting short-term environmental dynamics and long-term navigation milestones. The VLM's structured plan guides its targeted prediction, while the imagined future provides rich context to inform the navigation actions, creating a powerful internal feedback loop of perception-planning/prediction-action. We demonstrate through extensive experiments on the R2R-CE and RxR-CE benchmark that NavForesee achieves highly competitive performance in complex scenarios. Our work highlights the immense potential of fusing explicit language planning with implicit spatiotemporal prediction, paving the way for more intelligent and capable embodied agents.
>
---
#### [new 043] DREAMer-VXS: A Latent World Model for Sample-Efficient AGV Exploration in Stochastic, Unobserved Environments
- **分类: cs.RO**

- **简介: 该论文针对自主地面车辆（AGV）在随机、未知环境中的探索问题，提出DREAMer-VXS框架。通过结合卷积变分自编码器与递归状态空间模型，构建潜空间世界模型，实现基于想象轨迹的高效规划，显著降低真实交互需求，提升探索效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.00005v1](https://arxiv.org/pdf/2512.00005v1)**

> **作者:** Agniprabha Chakraborty
>
> **摘要:** The paradigm of learning-based robotics holds immense promise, yet its translation to real-world applications is critically hindered by the sample inefficiency and brittleness of conventional model-free reinforcement learning algorithms. In this work, we address these challenges by introducing DREAMer-VXS, a model-based framework for Autonomous Ground Vehicle (AGV) exploration that learns to plan from imagined latent trajectories. Our approach centers on learning a comprehensive world model from partial and high-dimensional LiDAR observations. This world model is composed of a Convolutional Variational Autoencoder (VAE), which learns a compact representation of the environment's structure, and a Recurrent State-Space Model (RSSM), which models complex temporal dynamics. By leveraging this learned model as a high-speed simulator, the agent can train its navigation policy almost entirely in imagination. This methodology decouples policy learning from real-world interaction, culminating in a 90% reduction in required environmental interactions to achieve expert-level performance when compared to state-of-the-art model-free SAC baselines. The agent's behavior is guided by an actor-critic policy optimized with a composite reward function that balances task objectives with an intrinsic curiosity bonus, promoting systematic exploration of unknown spaces. We demonstrate through extensive simulated experiments that DREAMer-VXS not only learns orders of magnitude faster but also develops more generalizable and robust policies, achieving a 45% increase in exploration efficiency in unseen environments and superior resilience to dynamic obstacles.
>
---
#### [new 044] Much Ado About Noising: Dispelling the Myths of Generative Robotic Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究生成式机器人控制（GCPs）在行为克隆任务中的有效性。针对“GCPs成功源于多模态建模或复杂映射”的普遍认知，作者通过实验发现其优势实为迭代计算与适度随机性结合的结果。提出轻量级两步回归策略MIP，性能媲美甚至超越复杂生成模型，揭示分布拟合并非关键，引导新控制设计方向。**

- **链接: [https://arxiv.org/pdf/2512.01809v1](https://arxiv.org/pdf/2512.01809v1)**

> **作者:** Chaoyi Pan; Giri Anantharaman; Nai-Chieh Huang; Claire Jin; Daniel Pfrommer; Chenyang Yuan; Frank Permenter; Guannan Qu; Nicholas Boffi; Guanya Shi; Max Simchowitz
>
> **摘要:** Generative models, like flows and diffusions, have recently emerged as popular and efficacious policy parameterizations in robotics. There has been much speculation as to the factors underlying their successes, ranging from capturing multi-modal action distribution to expressing more complex behaviors. In this work, we perform a comprehensive evaluation of popular generative control policies (GCPs) on common behavior cloning (BC) benchmarks. We find that GCPs do not owe their success to their ability to capture multi-modality or to express more complex observation-to-action mappings. Instead, we find that their advantage stems from iterative computation, as long as intermediate steps are supervised during training and this supervision is paired with a suitable level of stochasticity. As a validation of our findings, we show that a minimum iterative policy (MIP), a lightweight two-step regression-based policy, essentially matches the performance of flow GCPs, and often outperforms distilled shortcut models. Our results suggest that the distribution-fitting component of GCPs is less salient than commonly believed, and point toward new design spaces focusing solely on control performance. Project page: https://simchowitzlabpublic.github.io/much-ado-about-noising-project/
>
---
#### [new 045] Reinforcement Learning for Gliding Projectile Guidance and Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究基于强化学习的滑翔弹药制导与控制，旨在提升固定翼飞行器在动态环境中的自主导航能力。针对目标跟踪精度与自主性不足的问题，提出一种适用于全轴向的强化学习控制律，验证其在光学制导滑翔机上的可行性与有效性。**

- **链接: [https://arxiv.org/pdf/2512.01066v1](https://arxiv.org/pdf/2512.01066v1)**

> **作者:** Joel Cahn; Antonin Thomas; Philippe Pastor
>
> **备注:** 6 pages
>
> **摘要:** This paper presents the development of a control law, which is intended to be implemented on an optical guided glider. This guiding law follows an innovative approach, the reinforcement learning. This control law is used to make navigation more flexible and autonomous in a dynamic environment. The final objective is to track a target detected with the camera and then guide the glider to this point with high precision. Already applied on quad-copter drones, we wish by this study to demonstrate the applicability of reinforcement learning for fixed-wing aircraft on all of its axis.
>
---
#### [new 046] Real-World Robot Control by Deep Active Inference With a Temporally Hierarchical World Model
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究真实世界机器人控制任务，旨在解决现有深度学习方法在不确定性环境下缺乏探索能力与计算效率低的问题。提出一种分层时序世界模型的深度主动推断框架，通过多尺度状态表示与抽象动作压缩，实现高效、灵活的动作选择，显著提升复杂任务中的成功率与适应性。**

- **链接: [https://arxiv.org/pdf/2512.01924v1](https://arxiv.org/pdf/2512.01924v1)**

> **作者:** Kentaro Fujii; Shingo Murata
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Robots in uncertain real-world environments must perform both goal-directed and exploratory actions. However, most deep learning-based control methods neglect exploration and struggle under uncertainty. To address this, we adopt deep active inference, a framework that accounts for human goal-directed and exploratory actions. Yet, conventional deep active inference approaches face challenges due to limited environmental representation capacity and high computational cost in action selection. We propose a novel deep active inference framework that consists of a world model, an action model, and an abstract world model. The world model encodes environmental dynamics into hidden state representations at slow and fast timescales. The action model compresses action sequences into abstract actions using vector quantization, and the abstract world model predicts future slow states conditioned on the abstract action, enabling low-cost action selection. We evaluate the framework on object-manipulation tasks with a real-world robot. Results show that it achieves high success rates across diverse manipulation tasks and switches between goal-directed and exploratory actions in uncertain settings, while making action selection computationally tractable. These findings highlight the importance of modeling multiple timescale dynamics and abstracting actions and state transitions.
>
---
#### [new 047] Learning Sim-to-Real Humanoid Locomotion in 15 Minutes
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究机器人仿真实现到现实部署的快速学习任务，针对高维、复杂环境下的仿真实现与现实应用间差距问题，提出基于FastSAC/FastTD3的简化训练方案，在单张RTX 4090 GPU上仅用15分钟实现人形机器人步态控制与全身动作跟踪的快速端到端训练，并在强域随机化条件下保持稳定性能。**

- **链接: [https://arxiv.org/pdf/2512.01996v1](https://arxiv.org/pdf/2512.01996v1)**

> **作者:** Younggyo Seo; Carmelo Sferrazza; Juyue Chen; Guanya Shi; Rocky Duan; Pieter Abbeel
>
> **备注:** Project website: https://younggyo.me/fastsac-humanoid
>
> **摘要:** Massively parallel simulation has reduced reinforcement learning (RL) training time for robots from days to minutes. However, achieving fast and reliable sim-to-real RL for humanoid control remains difficult due to the challenges introduced by factors such as high dimensionality and domain randomization. In this work, we introduce a simple and practical recipe based on off-policy RL algorithms, i.e., FastSAC and FastTD3, that enables rapid training of humanoid locomotion policies in just 15 minutes with a single RTX 4090 GPU. Our simple recipe stabilizes off-policy RL algorithms at massive scale with thousands of parallel environments through carefully tuned design choices and minimalist reward functions. We demonstrate rapid end-to-end learning of humanoid locomotion controllers on Unitree G1 and Booster T1 robots under strong domain randomization, e.g., randomized dynamics, rough terrain, and push perturbations, as well as fast training of whole-body human-motion tracking policies. We provide videos and open-source implementation at: https://younggyo.me/fastsac-humanoid.
>
---
#### [new 048] VLASH: Real-Time VLAs via Future-State-Aware Asynchronous Inference
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对视觉-语言-动作模型（VLAs）在机器人任务中实时性差的问题，提出VLASH框架。通过预测未来执行状态实现异步推理，缓解了推理与执行间的时序错位，显著提升反应速度与控制稳定性，无需额外开销即可支持高速高精度任务。**

- **链接: [https://arxiv.org/pdf/2512.01031v1](https://arxiv.org/pdf/2512.01031v1)**

> **作者:** Jiaming Tang; Yufei Sun; Yilong Zhao; Shang Yang; Yujun Lin; Zhuoyang Zhang; James Hou; Yao Lu; Zhijian Liu; Song Han
>
> **摘要:** Vision-Language-Action models (VLAs) are becoming increasingly capable across diverse robotic tasks. However, their real-world deployment remains slow and inefficient: demonstration videos are often sped up by 5-10x to appear smooth, with noticeable action stalls and delayed reactions to environmental changes. Asynchronous inference offers a promising solution to achieve continuous and low-latency control by enabling robots to execute actions and perform inference simultaneously. However, because the robot and environment continue to evolve during inference, a temporal misalignment arises between the prediction and execution intervals. This leads to significant action instability, while existing methods either degrade accuracy or introduce runtime overhead to mitigate it. We propose VLASH, a general asynchronous inference framework for VLAs that delivers smooth, accurate, and fast reaction control without additional overhead or architectural changes. VLASH estimates the future execution-time state by rolling the robot state forward with the previously generated action chunk, thereby bridging the gap between prediction and execution. Experiments show that VLASH achieves up to 2.03x speedup and reduces reaction latency by up to 17.4x compared to synchronous inference while fully preserving the original accuracy. Moreover, it empowers VLAs to handle fast-reaction, high-precision tasks such as playing ping-pong and playing whack-a-mole, where traditional synchronous inference fails. Code is available at https://github.com/mit-han-lab/vlash
>
---
#### [new 049] Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究视觉引导的人形机器人开锁任务，解决模拟到现实的零样本迁移问题。提出教师-学生自举框架，结合分阶段重置探索与GRPO微调，实现仅用RGB图像的端到端控制，在多种门类型上超越人类操作者31.7%的任务效率。**

- **链接: [https://arxiv.org/pdf/2512.01061v1](https://arxiv.org/pdf/2512.01061v1)**

> **作者:** Haoru Xue; Tairan He; Zi Wang; Qingwei Ben; Wenli Xiao; Zhengyi Luo; Xingye Da; Fernando Castañeda; Guanya Shi; Shankar Sastry; Linxi "Jim" Fan; Yuke Zhu
>
> **备注:** https://doorman-humanoid.github.io/
>
> **摘要:** Recent progress in GPU-accelerated, photorealistic simulation has opened a scalable data-generation path for robot learning, where massive physics and visual randomization allow policies to generalize beyond curated environments. Building on these advances, we develop a teacher-student-bootstrap learning framework for vision-based humanoid loco-manipulation, using articulated-object interaction as a representative high-difficulty benchmark. Our approach introduces a staged-reset exploration strategy that stabilizes long-horizon privileged-policy training, and a GRPO-based fine-tuning procedure that mitigates partial observability and improves closed-loop consistency in sim-to-real RL. Trained entirely on simulation data, the resulting policy achieves robust zero-shot performance across diverse door types and outperforms human teleoperators by up to 31.7% in task completion time under the same whole-body control stack. This represents the first humanoid sim-to-real policy capable of diverse articulated loco-manipulation using pure RGB perception.
>
---
#### [new 050] A Hierarchical Framework for Humanoid Locomotion with Supernumerary Limbs
- **分类: cs.RO**

- **简介: 该论文研究人形机器人附加多余肢体（SLs）时的步态稳定性问题。针对SLs引入动态扰动导致的不稳，提出分层控制框架：低层通过模仿与课程学习生成步态，高层主动利用SLs进行动态平衡。实验表明，该方法显著提升稳定性，使质心轨迹更接近无负载基线，降低47%的DTW距离。**

- **链接: [https://arxiv.org/pdf/2512.00077v1](https://arxiv.org/pdf/2512.00077v1)**

> **作者:** Bowen Zhi
>
> **摘要:** The integration of Supernumerary Limbs (SLs) on humanoid robots poses a significant stability challenge due to the dynamic perturbations they introduce. This thesis addresses this issue by designing a novel hierarchical control architecture to improve humanoid locomotion stability with SLs. The core of this framework is a decoupled strategy that combines learning-based locomotion with model-based balancing. The low-level component consists of a walking gait for a Unitree H1 humanoid through imitation learning and curriculum learning. The high-level component actively utilizes the SLs for dynamic balancing. The effectiveness of the system is evaluated in a physics-based simulation under three conditions: baseline gait for an unladen humanoid (baseline walking), walking with a static SL payload (static payload), and walking with the active dynamic balancing controller (dynamic balancing). Our evaluation shows that the dynamic balancing controller improves stability. Compared to the static payload condition, the balancing strategy yields a gait pattern closer to the baseline and decreases the Dynamic Time Warping (DTW) distance of the CoM trajectory by 47\%. The balancing controller also improves the re-stabilization within gait cycles and achieves a more coordinated anti-phase pattern of Ground Reaction Forces (GRF). The results demonstrate that a decoupled, hierarchical design can effectively mitigate the internal dynamic disturbances arising from the mass and movement of the SLs, enabling stable locomotion for humanoids equipped with functional limbs. Code and videos are available here: https://github.com/heyzbw/HuSLs.
>
---
#### [new 051] Learning from Watching: Scalable Extraction of Manipulation Trajectories from Human Videos
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人学习中数据采集成本高的问题，提出利用互联网人类操作视频提取密集的操纵关键点轨迹。通过结合大模型视频理解与点追踪技术，实现对任务相关关键点的全流程精准跟踪，提升数据规模与利用效率，推动更高效、可扩展的机器人学习。**

- **链接: [https://arxiv.org/pdf/2512.00024v1](https://arxiv.org/pdf/2512.00024v1)**

> **作者:** X. Hu; G. Ye
>
> **备注:** Accepted to RSS 2025 Workshop
>
> **摘要:** Collecting high-quality data for training large-scale robotic models typically relies on real robot platforms, which is labor-intensive and costly, whether via teleoperation or scripted demonstrations. To scale data collection, many researchers have turned to leveraging human manipulation videos available online. However, current methods predominantly focus on hand detection or object pose estimation, failing to fully exploit the rich interaction cues embedded in these videos. In this work, we propose a novel approach that combines large foundation models for video understanding with point tracking techniques to extract dense trajectories of all task-relevant keypoints during manipulation. This enables more comprehensive utilization of Internet-scale human demonstration videos. Experimental results demonstrate that our method can accurately track keypoints throughout the entire manipulation process, paving the way for more scalable and data-efficient robot learning.
>
---
#### [new 052] LAP: Fast LAtent Diffusion Planner with Fine-Grained Feature Distillation for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶规划中扩散模型推理慢、忽略高层语义的问题，提出LAP框架。通过在VAE隐空间中进行单步去噪规划，分离高阶意图与低阶运动，结合细粒度特征蒸馏，实现高效高质的多模态路径规划，在nuPlan上达到SOTA性能，推理速度提升10倍。**

- **链接: [https://arxiv.org/pdf/2512.00470v1](https://arxiv.org/pdf/2512.00470v1)**

> **作者:** Jinhao Zhang; Wenlong Xia; Zhexuan Zhou; Youmin Gong; Jie Mei
>
> **摘要:** Diffusion models have demonstrated strong capabilities for modeling human-like driving behaviors in autonomous driving, but their iterative sampling process induces substantial latency, and operating directly on raw trajectory points forces the model to spend capacity on low-level kinematics, rather than high-level multi-modal semantics. To address these limitations, we propose LAtent Planner (LAP), a framework that plans in a VAE-learned latent space that disentangles high-level intents from low-level kinematics, enabling our planner to capture rich, multi-modal driving strategies. We further introduce a fine-grained feature distillation mechanism to guide a better interaction and fusion between the high-level semantic planning space and the vectorized scene context. Notably, LAP can produce high-quality plans in one single denoising step, substantially reducing computational overhead. Through extensive evaluations on the large-scale nuPlan benchmark, LAP achieves state-of-the-art closed-loop performance among learning-based planning methods, while demonstrating an inference speed-up of at most 10 times over previous SOTA approaches.
>
---
#### [new 053] An adaptive experience-based discrete genetic algorithm for multi-trip picking robot task scheduling in smart orchards
- **分类: cs.RO**

- **简介: 该论文研究智能果园中多趟采摘机器人任务调度问题，旨在优化机器人负载、距离与能耗的平衡，缩短作业周期。提出自适应经验驱动的离散遗传算法（AEDGA），通过初始化、聚类局部搜索和经验自适应选择提升求解效率，显著优于现有算法。**

- **链接: [https://arxiv.org/pdf/2512.00057v1](https://arxiv.org/pdf/2512.00057v1)**

> **作者:** Peng Chen; Jing Liangb; Kang-Jia Qiao; Hui Song; Cai-Tong Yue; Kun-Jie Yu; Ponnuthurai Nagaratnam Suganthan; Witold Pedrycz
>
> **摘要:** The continuous innovation of smart robotic technologies is driving the development of smart orchards, significantly enhancing the potential for automated harvesting systems. While multi-robot systems offer promising solutions to address labor shortages and rising costs, the efficient scheduling of these systems presents complex optimization challenges. This research investigates the multi-trip picking robot task scheduling (MTPRTS) problem. The problem is characterized by its provision for robot redeployment while maintaining strict adherence to makespan constraints, and encompasses the interdependencies among robot weight, robot load, and energy consumption, thus introducing substantial computational challenges that demand sophisticated optimization algorithms.To effectively tackle this complexity, metaheuristic approaches, which often utilize local search mechanisms, are widely employed. Despite the critical role of local search in vehicle routing problems, most existing algorithms are hampered by redundant local operations, leading to slower search processes and higher risks of local optima, particularly in large-scale scenarios. To overcome these limitations, we propose an adaptive experience-based discrete genetic algorithm (AEDGA) that introduces three key innovations: (1) integrated load-distance balancing initialization method, (2) a clustering-based local search mechanism, and (3) an experience-based adaptive selection strategy. To ensure solution feasibility under makespan constraints, we develop a solution repair strategy implemented through three distinct frameworks. Comprehensive experiments on 18 proposed test instances and 24 existing test problems demonstrate that AEDGA significantly outperforms eight state-of-the-art algorithms.
>
---
#### [new 054] Reconfigurable Auxetic Devices (RADs) for Robotic Surface Manipulation
- **分类: cs.RO**

- **简介: 该论文研究可重构负泊松比结构在机器人表面操作中的应用。针对传统机器人表面缺乏自适应形变能力的问题，提出基于负泊松比晶格的可重构装置（RADs），通过锁紧或嵌入伺服实现局部区域可调膨胀，利用单元间间隙实现柔性变形，提升表面贴合性与操控灵活性。**

- **链接: [https://arxiv.org/pdf/2512.00072v1](https://arxiv.org/pdf/2512.00072v1)**

> **作者:** Jacob Miske; Ahyan Maya; Ahnaf Inkiad; Jeffrey Ian Lipton
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** Robotic surfaces traditionally use materials with a positive Poisson's ratio to push and pull on a manipulation interface. Auxetic materials with a negative Poisson's ratio may expand in multiple directions when stretched and enable conformable interfaces. Here we demonstrate reconfigurable auxetic lattices for robotic surface manipulation. Our approach enables shape control through reconfigurable locking or embedded servos that underactuate an auxetic lattice structure. Variable expansion of local lattice areas is enabled by backlash between unit cells. Demonstrations of variable surface conformity are presented with characterization metrics. Experimental results are validated against a simplified model of the system, which uses an activation function to model intercell coupling with backlash. Reconfigurable auxetic structures are shown to achieve manipulation via variable surface contraction and expansion. This structure maintains compliance with backlash in contrast with previous work on auxetics, opening new opportunities in adaptive robotic structures for surface manipulation tasks.
>
---
#### [new 055] RoaD: Rollouts as Demonstrations for Closed-Loop Supervised Fine-Tuning of Autonomous Driving Policies
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文针对自动驾驶政策在闭环部署时因协变量偏移导致的误差累积问题，提出RoaD方法。通过利用策略自身生成的闭环轨迹作为示范数据，结合专家引导提升轨迹质量，实现高效闭环微调。实验表明，RoaD在仿真环境中显著提升驾驶性能并减少碰撞，且所需数据远少于强化学习。**

- **链接: [https://arxiv.org/pdf/2512.01993v1](https://arxiv.org/pdf/2512.01993v1)**

> **作者:** Guillermo Garcia-Cobo; Maximilian Igl; Peter Karkus; Zhejun Zhang; Michael Watson; Yuxiao Chen; Boris Ivanovic; Marco Pavone
>
> **备注:** Preprint
>
> **摘要:** Autonomous driving policies are typically trained via open-loop behavior cloning of human demonstrations. However, such policies suffer from covariate shift when deployed in closed loop, leading to compounding errors. We introduce Rollouts as Demonstrations (RoaD), a simple and efficient method to mitigate covariate shift by leveraging the policy's own closed-loop rollouts as additional training data. During rollout generation, RoaD incorporates expert guidance to bias trajectories toward high-quality behavior, producing informative yet realistic demonstrations for fine-tuning. This approach enables robust closed-loop adaptation with orders of magnitude less data than reinforcement learning, and avoids restrictive assumptions of prior closed-loop supervised fine-tuning (CL-SFT) methods, allowing broader applications domains including end-to-end driving. We demonstrate the effectiveness of RoaD on WOSAC, a large-scale traffic simulation benchmark, where it performs similar or better than the prior CL-SFT method; and in AlpaSim, a high-fidelity neural reconstruction-based simulator for end-to-end driving, where it improves driving score by 41\% and reduces collisions by 54\%.
>
---
#### [new 056] L2M-Calib: One-key Calibration Method for LiDAR and Multiple Magnetic Sensors
- **分类: cs.RO**

- **简介: 该论文针对磁传感器与激光雷达融合系统中缺乏高效校准方法的问题，提出L2M-Calib一键校准框架。通过联合优化外参与磁传感器内参，实现高精度、鲁棒的多模态标定，显著提升复杂环境下的感知可靠性。**

- **链接: [https://arxiv.org/pdf/2512.01554v1](https://arxiv.org/pdf/2512.01554v1)**

> **作者:** Qiyang Lyu; Wei Wang; Zhenyu Wu; Hongming Shen; Huiqin Zhou; Danwei Wang
>
> **摘要:** Multimodal sensor fusion enables robust environmental perception by leveraging complementary information from heterogeneous sensing modalities. However, accurate calibration is a critical prerequisite for effective fusion. This paper proposes a novel one-key calibration framework named L2M-Calib for a fused magnetic-LiDAR system, jointly estimating the extrinsic transformation between the two kinds of sensors and the intrinsic distortion parameters of the magnetic sensors. Magnetic sensors capture ambient magnetic field (AMF) patterns, which are invariant to geometry, texture, illumination, and weather, making them suitable for challenging environments. Nonetheless, the integration of magnetic sensing into multimodal systems remains underexplored due to the absence of effective calibration techniques. To address this, we optimize extrinsic parameters using an iterative Gauss-Newton scheme, coupled with the intrinsic calibration as a weighted ridge-regularized total least squares (w-RRTLS) problem, ensuring robustness against measurement noise and ill-conditioned data. Extensive evaluations on both simulated datasets and real-world experiments, including AGV-mounted sensor configurations, demonstrate that our method achieves high calibration accuracy and robustness under various environmental and operational conditions.
>
---
#### [new 057] Arcadia: Toward a Full-Lifecycle Framework for Embodied Lifelong Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Arcadia框架，解决具身智能体终身学习的生命周期问题。针对单一阶段优化难以持续改进与泛化的问题，构建四阶段闭环：自主数据采集、生成式场景重建、共享表示学习、仿真评估演化。通过紧密耦合实现持续进化与真实迁移，推动具身智能向通用化发展。**

- **链接: [https://arxiv.org/pdf/2512.00076v1](https://arxiv.org/pdf/2512.00076v1)**

> **作者:** Minghe Gao; Juncheng Li; Yuze Lin; Xuqi Liu; Jiaming Ji; Xiaoran Pan; Zihan Xu; Xian Li; Mingjie Li; Wei Ji; Rong Wei; Rui Tang; Qizhou Wang; Kai Shen; Jun Xiao; Qi Wu; Siliang Tang; Yueting Zhuang
>
> **摘要:** We contend that embodied learning is fundamentally a lifecycle problem rather than a single-stage optimization. Systems that optimize only one link (data collection, simulation, learning, or deployment) rarely sustain improvement or generalize beyond narrow settings. We introduce Arcadia, a closed-loop framework that operationalizes embodied lifelong learning by tightly coupling four stages: (1) Self-evolving exploration and grounding for autonomous data acquisition in physical environments, (2) Generative scene reconstruction and augmentation for realistic and extensible scene creation, (3) a Shared embodied representation architecture that unifies navigation and manipulation within a single multimodal backbone, and (4) Sim-from-real evaluation and evolution that closes the feedback loop through simulation-based adaptation. This coupling is non-decomposable: removing any stage breaks the improvement loop and reverts to one-shot training. Arcadia delivers consistent gains on navigation and manipulation benchmarks and transfers robustly to physical robots, indicating that a tightly coupled lifecycle: continuous real-world data acquisition, generative simulation update, and shared-representation learning, supports lifelong improvement and end-to-end generalization. We release standardized interfaces enabling reproducible evaluation and cross-model comparison in reusable environments, positioning Arcadia as a scalable foundation for general-purpose embodied agents.
>
---
#### [new 058] A Survey on Improving Human Robot Collaboration through Vision-and-Language Navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **简介: 该论文聚焦视觉-语言导航（VLN）任务，旨在提升人机协作中的多机器人协同能力。针对当前模型在双向沟通、歧义消解与协作决策方面的不足，系统综述了近200篇相关研究，提出需引入主动澄清、实时反馈与动态角色分配机制，以推动高效、可扩展的智能协作系统发展。**

- **链接: [https://arxiv.org/pdf/2512.00027v1](https://arxiv.org/pdf/2512.00027v1)**

> **作者:** Nivedan Yakolli; Avinash Gautam; Abhijit Das; Yuankai Qi; Virendra Singh Shekhawat
>
> **摘要:** Vision-and-Language Navigation (VLN) is a multi-modal, cooperative task requiring agents to interpret human instructions, navigate 3D environments, and communicate effectively under ambiguity. This paper presents a comprehensive review of recent VLN advancements in robotics and outlines promising directions to improve multi-robot coordination. Despite progress, current models struggle with bidirectional communication, ambiguity resolution, and collaborative decision-making in the multi-agent systems. We review approximately 200 relevant articles to provide an in-depth understanding of the current landscape. Through this survey, we aim to provide a thorough resource that inspires further research at the intersection of VLN and robotics. We advocate that the future VLN systems should support proactive clarification, real-time feedback, and contextual reasoning through advanced natural language understanding (NLU) techniques. Additionally, decentralized decision-making frameworks with dynamic role assignment are essential for scalable, efficient multi-robot collaboration. These innovations can significantly enhance human-robot interaction (HRI) and enable real-world deployment in domains such as healthcare, logistics, and disaster response.
>
---
#### [new 059] "Why the face?": Exploring Robot Error Detection Using Instrumented Bystander Reactions
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究机器人如何通过捕捉旁观者面部反应来检测自身错误。针对机器人缺乏感知社会细微反馈能力的问题，提出使用颈挂式摄像头采集下巴区域表情，构建NeckNet-18模型进行3D面部重建，并据此开发更优的误差检测方法，提升机器人在人际互动中的适应性。**

- **链接: [https://arxiv.org/pdf/2512.00262v1](https://arxiv.org/pdf/2512.00262v1)**

> **作者:** Maria Teresa Parreira; Ruidong Zhang; Sukruth Gowdru Lingaraju; Alexandra Bremers; Xuanyu Fang; Adolfo Ramirez-Aristizabal; Manaswi Saha; Michael Kuniavsky; Cheng Zhang; Wendy Ju
>
> **摘要:** How do humans recognize and rectify social missteps? We achieve social competence by looking around at our peers, decoding subtle cues from bystanders - a raised eyebrow, a laugh - to evaluate the environment and our actions. Robots, however, struggle to perceive and make use of these nuanced reactions. By employing a novel neck-mounted device that records facial expressions from the chin region, we explore the potential of previously untapped data to capture and interpret human responses to robot error. First, we develop NeckNet-18, a 3D facial reconstruction model to map the reactions captured through the chin camera onto facial points and head motion. We then use these facial responses to develop a robot error detection model which outperforms standard methodologies such as using OpenFace or video data, generalizing well especially for within-participant data. Through this work, we argue for expanding human-in-the-loop robot sensing, fostering more seamless integration of robots into diverse human environments, pushing the boundaries of social cue detection and opening new avenues for adaptable robotics.
>
---
#### [new 060] Visibility-aware Cooperative Aerial Tracking with Decentralized LiDAR-based Swarms
- **分类: cs.RO**

- **简介: 该论文针对复杂环境中多无人机协同目标跟踪任务，解决单机跟踪局限与协同感知难题。提出基于分布式LiDAR的去中心化框架，引入球面符号距离场实现实时遮挡建模，设计视场对齐与电势分布协同成本，通过分层规划生成安全、可视化的三维协同轨迹，显著提升多目标跟踪的鲁棒性与覆盖率。**

- **链接: [https://arxiv.org/pdf/2512.01280v1](https://arxiv.org/pdf/2512.01280v1)**

> **作者:** Longji Yin; Yunfan Ren; Fangcheng Zhu; Liuyu Shi; Fanze Kong; Benxu Tang; Wenyi Liu; Ximin Lyu; Fu Zhang
>
> **摘要:** Autonomous aerial tracking with drones offers vast potential for surveillance, cinematography, and industrial inspection applications. While single-drone tracking systems have been extensively studied, swarm-based target tracking remains underexplored, despite its unique advantages of distributed perception, fault-tolerant redundancy, and multidirectional target coverage. To bridge this gap, we propose a novel decentralized LiDAR-based swarm tracking framework that enables visibility-aware, cooperative target tracking in complex environments, while fully harnessing the unique capabilities of swarm systems. To address visibility, we introduce a novel Spherical Signed Distance Field (SSDF)-based metric for 3-D environmental occlusion representation, coupled with an efficient algorithm that enables real-time onboard SSDF updating. A general Field-of-View (FOV) alignment cost supporting heterogeneous LiDAR configurations is proposed for consistent target observation. Swarm coordination is enhanced through cooperative costs that enforce inter-robot safe clearance, prevent mutual occlusions, and notably facilitate 3-D multidirectional target encirclement via a novel electrostatic-potential-inspired distribution metric. These innovations are integrated into a hierarchical planner, combining a kinodynamic front-end searcher with a spatiotemporal $SE(3)$ back-end optimizer to generate collision-free, visibility-optimized trajectories.Deployed on heterogeneous LiDAR swarms, our fully decentralized implementation features collaborative perception, distributed planning, and dynamic swarm reconfigurability. Validated through rigorous real-world experiments in cluttered outdoor environments, the proposed system demonstrates robust cooperative tracking of agile targets (drones, humans) while achieving superior visibility maintenance.
>
---
#### [new 061] DPNet: Doppler LiDAR Motion Planning for Highly-Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文针对高动态环境中运动规划精度与频率不足的问题，提出DPNet框架。通过融合多普勒激光雷达的瞬时速度信息，设计D-KalmanNet实现障碍物状态预测，并构建DT-MPC框架实现自适应运动规划，显著提升对快速移动障碍物的响应能力。**

- **链接: [https://arxiv.org/pdf/2512.00375v1](https://arxiv.org/pdf/2512.00375v1)**

> **作者:** Wei Zuo; Zeyi Ren; Chengyang Li; Yikun Wang; Mingle Zhao; Shuai Wang; Wei Sui; Fei Gao; Yik-Chung Wu; Chengzhong Xu
>
> **摘要:** Existing motion planning methods often struggle with rapid-motion obstacles due to an insufficient understanding of environmental changes. To address this limitation, we propose integrating motion planners with Doppler LiDARs which provide not only ranging measurements but also instantaneous point velocities. However, this integration is nontrivial due to the dual requirements of high accuracy and high frequency. To this end, we introduce Doppler Planning Network (DPNet), which tracks and reacts to rapid obstacles using Doppler model-based learning. Particularly, we first propose a Doppler Kalman neural network (D-KalmanNet) to track the future states of obstacles under partially observable Gaussian state space model. We then leverage the estimated motions to construct a Doppler-tuned model predictive control (DT-MPC) framework for ego-motion planning, enabling runtime auto-tuning of the controller parameters. These two model-based learners allow DPNet to maintain lightweight while learning fast environmental changes using minimum data, and achieve both high frequency and high accuracy in tracking and planning. Experiments on both high-fidelity simulator and real-world datasets demonstrate the superiority of DPNet over extensive benchmark schemes.
>
---
#### [new 062] LLM-Driven Corrective Robot Operation Code Generation with Static Text-Based Simulation
- **分类: cs.RO**

- **简介: 该论文针对LLM生成机器人操作代码的可靠性问题，提出一种无需物理实验或动态模拟器的静态文本仿真方法。通过增强LLM的语义理解与状态推理能力，实现对代码执行的静态模拟，构建了可靠的纠错式代码生成框架，在多种机器人任务中验证了其高效性与准确性。**

- **链接: [https://arxiv.org/pdf/2512.02002v1](https://arxiv.org/pdf/2512.02002v1)**

> **作者:** Wenhao Wang; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Recent advances in Large language models (LLMs) have demonstrated their promising capabilities of generating robot operation code to enable LLM-driven robots. To enhance the reliability of operation code generated by LLMs, corrective designs with feedback from the observation of executing code have been increasingly adopted in existing research. However, the code execution in these designs relies on either a physical experiment or a customized simulation environment, which limits their deployment due to the high configuration effort of the environment and the potential long execution time. In this paper, we explore the possibility of directly leveraging LLM to enable static simulation of robot operation code, and then leverage it to design a new reliable LLM-driven corrective robot operation code generation framework. Our framework configures the LLM as a static simulator with enhanced capabilities that reliably simulate robot code execution by interpreting actions, reasoning over state transitions, analyzing execution outcomes, and generating se- mantic observations that accurately capture trajectory dynamics. To validate the performance of our framework, we performed experiments on various operation tasks for different robots, including UAVs and small ground vehicles. The experiment results not only demonstrated the high accuracy of our static text-based simulation but also the reliable code generation of our LLM-driven corrective framework, which achieves a comparable performance with state-of-the-art research while does not rely on dynamic code execution using physical experiments or simulators.
>
---
#### [new 063] Dynamic Log-Gaussian Process Control Barrier Function for Safe Robotic Navigation in Dynamic Environments
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对动态环境中机器人安全导航问题，提出一种基于高斯过程的动态对数高斯过程控制屏障函数（DLGP-CBF）。通过引入对数变换提升屏障函数在稀疏数据下的平滑性与信息量，并显式建模障碍物位置与速度，使控制器能主动响应动态障碍物运动，实现更安全、平滑的路径规划。**

- **链接: [https://arxiv.org/pdf/2512.01668v1](https://arxiv.org/pdf/2512.01668v1)**

> **作者:** Xin Yin; Chenyang Liang; Yanning Guo; Jie Mei
>
> **备注:** To be presented in the 64th IEEE Conference on Decision and Control (CDC 2025)
>
> **摘要:** Control Barrier Functions (CBFs) have emerged as efficient tools to address the safe navigation problem for robot applications. However, synthesizing informative and obstacle motion-aware CBFs online using real-time sensor data remains challenging, particularly in unknown and dynamic scenarios. Motived by this challenge, this paper aims to propose a novel Gaussian Process-based formulation of CBF, termed the Dynamic Log Gaussian Process Control Barrier Function (DLGP-CBF), to enable real-time construction of CBF which are both spatially informative and responsive to obstacle motion. Firstly, the DLGP-CBF leverages a logarithmic transformation of GP regression to generate smooth and informative barrier values and gradients, even in sparse-data regions. Secondly, by explicitly modeling the DLGP-CBF as a function of obstacle positions, the derived safety constraint integrates predicted obstacle velocities, allowing the controller to proactively respond to dynamic obstacles' motion. Simulation results demonstrate significant improvements in obstacle avoidance performance, including increased safety margins, smoother trajectories, and enhanced responsiveness compared to baseline methods.
>
---
#### [new 064] Design And Control of A Robotic Arm For Industrial Applications
- **分类: cs.RO**

- **简介: 该论文针对工业自动化中装配、焊接和物料搬运的需求，设计并控制一种六自由度机器人臂。通过伺服电机与微控制器实现机械结构搭建，结合运动学与动力学分析，采用逆解算法和PID控制，提升定位精度与重复性。实验与仿真验证了系统高精度、低成本、可扩展的可行性。**

- **链接: [https://arxiv.org/pdf/2512.00034v1](https://arxiv.org/pdf/2512.00034v1)**

> **作者:** Sathish Krishna Anumula; SVSV Prasad Sanaboina; Ravi Kumar Nagula; R. Nagaraju
>
> **备注:** 8 Pages and 5 Figures
>
> **摘要:** The growing need to automate processes in industrial settings has led to tremendous growth in the robotic systems and especially the robotic arms. The paper assumes the design, modeling and control of a robotic arm to suit industrial purpose like assembly, welding and material handling. A six-degree-of-freedom (DOF) robotic manipulator was designed based on servo motors and a microcontroller interface with Mechanical links were also fabricated. Kinematic and dynamic analyses have been done in order to provide precise positioning and effective loads. Inverse Kinematics algorithm and Proportional-Integral-Derivative (PID) controller were also applied to improve the precision of control. The ability of the system to carry out tasks with high accuracy and repeatability is confirmed by simulation and experimental testing. The suggested robotic arm is an affordable, expandable, and dependable method of automation of numerous mundane procedures in the manufacturing industry.
>
---
#### [new 065] Hyper-GoalNet: Goal-Conditioned Manipulation Policy Learning with HyperNetworks
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对机器人操作中目标条件策略学习的挑战，提出Hyper-GoalNet框架。通过超网络从目标信息生成特定策略参数，分离目标解析与状态处理，结合动态预测与距离约束提升表征质量。在多种随机化任务中表现优异，实验证明其在复杂环境下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00085v1](https://arxiv.org/pdf/2512.00085v1)**

> **作者:** Pei Zhou; Wanting Yao; Qian Luo; Xunzhe Zhou; Yanchao Yang
>
> **摘要:** Goal-conditioned policy learning for robotic manipulation presents significant challenges in maintaining performance across diverse objectives and environments. We introduce Hyper-GoalNet, a framework that generates task-specific policy network parameters from goal specifications using hypernetworks. Unlike conventional methods that simply condition fixed networks on goal-state pairs, our approach separates goal interpretation from state processing -- the former determines network parameters while the latter applies these parameters to current observations. To enhance representation quality for effective policy generation, we implement two complementary constraints on the latent space: (1) a forward dynamics model that promotes state transition predictability, and (2) a distance-based constraint ensuring monotonic progression toward goal states. We evaluate our method on a comprehensive suite of manipulation tasks with varying environmental randomization. Results demonstrate significant performance improvements over state-of-the-art methods, particularly in high-variability conditions. Real-world robotic experiments further validate our method's robustness to sensor noise and physical uncertainties. Code is available at: https://github.com/wantingyao/hyper-goalnet.
>
---
#### [new 066] Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人操作中3D视觉预训练方法性能不足的问题，提出AFRO框架。它通过生成式扩散过程建模状态预测，联合学习前后向动态以捕捉因果转移结构，避免显式几何重建与动作监督。采用特征差分与逆一致性监督提升特征质量。实验表明，AFRO在16个仿真和4个真实任务中显著提升操作成功率，具备良好可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.00074v1](https://arxiv.org/pdf/2512.00074v1)**

> **作者:** Qiwei Liang; Boyang Cai; Minghao Lai; Sitong Zhuang; Tao Lin; Yan Qin; Yixuan Ye; Jiaming Liang; Renjing Xu
>
> **摘要:** Despite strong results on recognition and segmentation, current 3D visual pre-training methods often underperform on robotic manipulation. We attribute this gap to two factors: the lack of state-action-state dynamics modeling and the unnecessary redundancy of explicit geometric reconstruction. We introduce AFRO, a self-supervised framework that learns dynamics-aware 3D representations without action or reconstruction supervision. AFRO casts state prediction as a generative diffusion process and jointly models forward and inverse dynamics in a shared latent space to capture causal transition structure. To prevent feature leakage in action learning, we employ feature differencing and inverse-consistency supervision, improving the quality and stability of visual features. When combined with Diffusion Policy, AFRO substantially increases manipulation success rates across 16 simulated and 4 real-world tasks, outperforming existing pre-training approaches. The framework also scales favorably with data volume and task complexity. Qualitative visualizations indicate that AFRO learns semantically rich, discriminative features, offering an effective pre-training solution for 3D representation learning in robotics. Project page: https://kolakivy.github.io/AFRO/
>
---
#### [new 067] ICD-Net: Inertial Covariance Displacement Network for Drone Visual-Inertial SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对无人机视觉-惯性SLAM中传感器误差与环境挑战导致的精度下降问题，提出ICD-Net框架。通过神经网络直接从原始惯性数据学习位移估计及不确定性，将其作为残差约束融入VINS-Fusion优化，有效提升轨迹精度与系统鲁棒性，尤其在视觉退化时表现优异。**

- **链接: [https://arxiv.org/pdf/2512.00037v1](https://arxiv.org/pdf/2512.00037v1)**

> **作者:** Tali Orlev Shapira; Itzik Klein
>
> **摘要:** Visual-inertial SLAM systems often exhibit suboptimal performance due to multiple confounding factors including imperfect sensor calibration, noisy measurements, rapid motion dynamics, low illumination, and the inherent limitations of traditional inertial navigation integration methods. These issues are particularly problematic in drone applications where robust and accurate state estimation is critical for safe autonomous operation. In this work, we present ICD-Net, a novel framework that enhances visual-inertial SLAM performance by learning to process raw inertial measurements and generating displacement estimates with associated uncertainty quantification. Rather than relying on analytical inertial sensor models that struggle with real-world sensor imperfections, our method directly extracts displacement maps from sensor data while simultaneously predicting measurement covariances that reflect estimation confidence. We integrate ICD-Net outputs as additional residual constraints into the VINS-Fusion optimization framework, where the predicted uncertainties appropriately weight the neural network contributions relative to traditional visual and inertial terms. The learned displacement constraints provide complementary information that compensates for various error sources in the SLAM pipeline. Our approach can be used under both normal operating conditions and in situations of camera inconsistency or visual degradation. Experimental evaluation on challenging high-speed drone sequences demonstrated that our approach significantly improved trajectory estimation accuracy compared to standard VINS-Fusion, with more than 38% improvement in mean APE and uncertainty estimates proving crucial for maintaining system robustness. Our method shows that neural network enhancement can effectively address multiple sources of SLAM degradation while maintaining real-time performance requirements.
>
---
#### [new 068] Integration of UWB Radar on Mobile Robots for Continuous Obstacle and Environment Mapping
- **分类: cs.RO**

- **简介: 该论文研究移动机器人在无基础设施环境下利用超宽带（UWB）雷达进行障碍物检测与环境建图。针对视觉传感器在低可见度环境中失效的问题，提出基于UWB雷达的免锚点、免视觉特征的实时建图方法，通过三步处理流程有效抑制噪声与多径效应，实现高精度障碍物识别与定位。**

- **链接: [https://arxiv.org/pdf/2512.01018v1](https://arxiv.org/pdf/2512.01018v1)**

> **作者:** Adelina Giurea; Stijn Luchie; Dieter Coppens; Jeroen Hoebeke; Eli De Poorter
>
> **备注:** This paper has been submitted to IEEE Access Journal and is currently undergoing review
>
> **摘要:** This paper presents an infrastructure-free approach for obstacle detection and environmental mapping using ultra-wideband (UWB) radar mounted on a mobile robotic platform. Traditional sensing modalities such as visual cameras and Light Detection and Ranging (LiDAR) fail in environments with poor visibility due to darkness, smoke, or reflective surfaces. In these visioned-impaired conditions, UWB radar offers a promising alternative. To this end, this work explores the suitability of robot-mounted UWB radar for environmental mapping in dynamic, anchor-free scenarios. The study investigates how different materials (metal, concrete and plywood) and UWB radio channels (5 and 9) influence the Channel Impulse Response (CIR). Furthermore, a processing pipeline is proposed to achieve reliable mapping of detected obstacles, consisting of 3 steps: (i) target identification (based on CIR peak detection), (ii) filtering (based on peak properties, signal-to-noise score, and phase-difference of arrival), and (iii) clustering (based on distance estimation and angle-of-arrival estimation). The proposed approach successfully reduces noise and multipath effects, resulting in an obstacle detection precision of at least 82.36% and a recall of 89.46% on channel 9 even when detecting low-reflective materials such as plywood. This work offers a foundation for further development of UWB-based localisation and mapping (SLAM) systems that do not rely on visual features and, unlike conventional UWB localisation systems, do not require on fixed anchor nodes for triangulation.
>
---
#### [new 069] ManualVLA: A Unified VLA Model for Chain-of-Thought Manual Generation and Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文针对长时序机器人任务中规划与操作脱节的问题，提出ManualVLA框架。通过混合变压器架构，先生成包含图文的位置提示的中间操作手册，再基于手册进行分步推理执行，实现从目标状态到可执行流程的映射。利用3D高斯泼溅构建数字孪生数据集，降低真实数据采集成本，显著提升拼装与重排任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.02013v1](https://arxiv.org/pdf/2512.02013v1)**

> **作者:** Chenyang Gu; Jiaming Liu; Hao Chen; Runzhong Huang; Qingpo Wuwu; Zhuoyang Liu; Xiaoqi Li; Ying Li; Renrui Zhang; Peng Jia; Pheng-Ann Heng; Shanghang Zhang
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged, demonstrating strong generalization in robotic scene understanding and manipulation. However, when confronted with long-horizon tasks that require defined goal states, such as LEGO assembly or object rearrangement, existing VLA models still face challenges in coordinating high-level planning with precise manipulation. Therefore, we aim to endow a VLA model with the capability to infer the "how" process from the "what" outcomes, transforming goal states into executable procedures. In this paper, we introduce ManualVLA, a unified VLA framework built upon a Mixture-of-Transformers (MoT) architecture, enabling coherent collaboration between multimodal manual generation and action execution. Unlike prior VLA models that directly map sensory inputs to actions, we first equip ManualVLA with a planning expert that generates intermediate manuals consisting of images, position prompts, and textual instructions. Building upon these multimodal manuals, we design a Manual Chain-of-Thought (ManualCoT) reasoning process that feeds them into the action expert, where each manual step provides explicit control conditions, while its latent representation offers implicit guidance for accurate manipulation. To alleviate the burden of data collection, we develop a high-fidelity digital-twin toolkit based on 3D Gaussian Splatting, which automatically generates manual data for planning expert training. ManualVLA demonstrates strong real-world performance, achieving an average success rate 32% higher than the previous hierarchical SOTA baseline on LEGO assembly and object rearrangement tasks.
>
---
#### [new 070] IGen: Scalable Data Generation for Robot Learning from Open-World Images
- **分类: cs.RO**

- **简介: 该论文针对机器人学习中大规模训练数据稀缺问题，提出IGen框架，从开放世界图像中可扩展生成高质量视觉观测与可执行动作。通过3D场景重建与视觉-语言模型推理，合成真实感的动态视觉序列与末端执行器运动轨迹，实现仅用合成数据训练的策略性能接近真实数据训练水平。**

- **链接: [https://arxiv.org/pdf/2512.01773v1](https://arxiv.org/pdf/2512.01773v1)**

> **作者:** Chenghao Gu; Haolan Kang; Junchao Lin; Jinghe Wang; Duo Wu; Shuzhao Xie; Fanding Huang; Junchen Ge; Ziyang Gong; Letian Li; Hongying Zheng; Changwei Lv; Zhi Wang
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** The rise of generalist robotic policies has created an exponential demand for large-scale training data. However, on-robot data collection is labor-intensive and often limited to specific environments. In contrast, open-world images capture a vast diversity of real-world scenes that naturally align with robotic manipulation tasks, offering a promising avenue for low-cost, large-scale robot data acquisition. Despite this potential, the lack of associated robot actions hinders the practical use of open-world images for robot learning, leaving this rich visual resource largely unexploited. To bridge this gap, we propose IGen, a framework that scalably generates realistic visual observations and executable actions from open-world images. IGen first converts unstructured 2D pixels into structured 3D scene representations suitable for scene understanding and manipulation. It then leverages the reasoning capabilities of vision-language models to transform scene-specific task instructions into high-level plans and generate low-level actions as SE(3) end-effector pose sequences. From these poses, it synthesizes dynamic scene evolution and renders temporally coherent visual observations. Experiments validate the high quality of visuomotor data generated by IGen, and show that policies trained solely on IGen-synthesized data achieve performance comparable to those trained on real-world data. This highlights the potential of IGen to support scalable data generation from open-world images for generalist robotic policy training.
>
---
#### [new 071] MS-PPO: Morphological-Symmetry-Equivariant Policy for Legged Robot Locomotion
- **分类: cs.RO**

- **简介: 该论文针对腿式机器人步态控制任务，解决现有强化学习策略忽视机器人结构对称性与形态导致的训练低效、泛化差问题。提出MS-PPO框架，通过图神经网络显式编码运动学结构与对称性，实现对称等变策略，提升训练稳定性与样本效率，实现在仿真与硬件上的高效泛化。**

- **链接: [https://arxiv.org/pdf/2512.00727v1](https://arxiv.org/pdf/2512.00727v1)**

> **作者:** Sizhe Wei; Xulin Chen; Fengze Xie; Garrett Ethan Katz; Zhenyu Gan; Lu Gan
>
> **摘要:** Reinforcement learning has recently enabled impressive locomotion capabilities on legged robots; however, most policy architectures remain morphology- and symmetry-agnostic, leading to inefficient training and limited generalization. This work introduces MS-PPO, a morphological-symmetry-equivariant policy learning framework that encodes robot kinematic structure and morphological symmetries directly into the policy network. We construct a morphology-informed graph neural architecture that is provably equivariant with respect to the robot's morphological symmetry group actions, ensuring consistent policy responses under symmetric states while maintaining invariance in value estimation. This design eliminates the need for tedious reward shaping or costly data augmentation, which are typically required to enforce symmetry. We evaluate MS-PPO in simulation on Unitree Go2 and Xiaomi CyberDog2 robots across diverse locomotion tasks, including trotting, pronking, slope walking, and bipedal turning, and further deploy the learned policies on hardware. Extensive experiments show that MS-PPO achieves superior training stability, symmetry generalization ability, and sample efficiency in challenging locomotion tasks, compared to state-of-the-art baselines. These findings demonstrate that embedding both kinematic structure and morphological symmetry into policy learning provides a powerful inductive bias for legged robot locomotion control. Our code will be made publicly available at https://lunarlab-gatech.github.io/MS-PPO/.
>
---
#### [new 072] COMET: A Dual Swashplate Autonomous Coaxial Bi-copter AAV with High-Maneuverability and Long-Endurance
- **分类: cs.RO**

- **简介: 该论文针对共轴双旋翼无人机效率、机动性与紧凑性难以兼顾的问题，提出COMET平台，采用双摇臂桨系统。通过台架测试优化效率，飞行实验验证续航与鲁棒性，轨迹跟踪测试评估机动性，证明双摇臂设计优于单摇臂，实现高效高机动长续航自主飞行。**

- **链接: [https://arxiv.org/pdf/2512.01246v1](https://arxiv.org/pdf/2512.01246v1)**

> **作者:** Shuai Wang; Xiaoming Tang; Junning Liang; Haowen Zheng; Biyu Ye; Zhaofeng Liu; Fei Gao; Ximin Lyu
>
> **备注:** 8 pages, 8 figures, accepted at IEEE RA-L
>
> **摘要:** Coaxial bi-copter autonomous aerial vehicles (AAVs) have garnered attention due to their potential for improved rotor system efficiency and compact form factor. However, balancing efficiency, maneuverability, and compactness in coaxial bi-copter systems remains a key design challenge, limiting their practical deployment. This letter introduces COMET, a coaxial bi-copter AAV platform featuring a dual swashplate mechanism. The coaxial bi-copter system's efficiency and compactness are optimized through bench tests, and the whole prototype's efficiency and robustness under varying payload conditions are verified through flight endurance experiments. The maneuverability performance of the system is evaluated in comprehensive trajectory tracking tests. The results indicate that the dual swashplate configuration enhances tracking performance and improves flight efficiency compared to the single swashplate alternative. Successful autonomous flight trials across various scenarios verify COMET's potential for real-world applications.
>
---
#### [new 073] RealAppliance: Let High-fidelity Appliance Assets Controllable and Workable as Aligned Real Manuals
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对家电仿真与真实手册不一致导致的模拟-现实差距问题，构建了100个高保真、机制完整的家电资产数据集RealAppliance，并提出RealAppliance-Bench基准。旨在评估多模态大模型与具身操作规划模型在家电操作任务中的表现，推动家电操控研究发展。**

- **链接: [https://arxiv.org/pdf/2512.00287v1](https://arxiv.org/pdf/2512.00287v1)**

> **作者:** Yuzheng Gao; Yuxing Long; Lei Kang; Yuchong Guo; Ziyan Yu; Shangqing Mao; Jiyao Zhang; Ruihai Wu; Dongjiang Li; Hui Shen; Hao Dong
>
> **摘要:** Existing appliance assets suffer from poor rendering, incomplete mechanisms, and misalignment with manuals, leading to simulation-reality gaps that hinder appliance manipulation development. In this work, we introduce the RealAppliance dataset, comprising 100 high-fidelity appliances with complete physical, electronic mechanisms, and program logic aligned with their manuals. Based on these assets, we propose the RealAppliance-Bench benchmark, which evaluates multimodal large language models and embodied manipulation planning models across key tasks in appliance manipulation planning: manual page retrieval, appliance part grounding, open-loop manipulation planning, and closed-loop planning adjustment. Our analysis of model performances on RealAppliance-Bench provides insights for advancing appliance manipulation research
>
---
#### [new 074] $\mathbf{M^3A}$ Policy: Mutable Material Manipulation Augmentation Policy through Photometric Re-rendering
- **分类: cs.RO**

- **简介: 该论文针对机器人操作中材料泛化难题，提出M³A策略，通过光度重渲染技术，基于单个真实演示生成多样化材质的逼真数据，实现跨材质泛化。解决了真实数据收集成本高、模拟与现实差距大的问题，显著提升机器人在多种材料上的操作成功率。**

- **链接: [https://arxiv.org/pdf/2512.01446v1](https://arxiv.org/pdf/2512.01446v1)**

> **作者:** Jiayi Li; Yuxuan Hu; Haoran Geng; Xiangyu Chen; Chuhao Zhou; Ziteng Cui; Jianfei Yang
>
> **备注:** under submission
>
> **摘要:** Material generalization is essential for real-world robotic manipulation, where robots must interact with objects exhibiting diverse visual and physical properties. This challenge is particularly pronounced for objects made of glass, metal, or other materials whose transparent or reflective surfaces introduce severe out-of-distribution variations. Existing approaches either rely on simulated materials in simulators and perform sim-to-real transfer, which is hindered by substantial visual domain gaps, or depend on collecting extensive real-world demonstrations, which is costly, time-consuming, and still insufficient to cover various materials. To overcome these limitations, we resort to computational photography and introduce Mutable Material Manipulation Augmentation (M$^3$A), a unified framework that leverages the physical characteristics of materials as captured by light transport for photometric re-rendering. The core idea is simple yet powerful: given a single real-world demonstration, we photometrically re-render the scene to generate a diverse set of highly realistic demonstrations with different material properties. This augmentation effectively decouples task-specific manipulation skills from surface appearance, enabling policies to generalize across materials without additional data collection. To systematically evaluate this capability, we construct the first comprehensive multi-material manipulation benchmark spanning both simulation and real-world environments. Extensive experiments show that the M$^3$A policy significantly enhances cross-material generalization, improving the average success rate across three real-world tasks by 58.03\%, and demonstrating robust performance on previously unseen materials.
>
---
#### [new 075] Socially aware navigation for mobile robots: a survey on deep reinforcement learning approaches
- **分类: cs.RO; cs.AI**

- **简介: 该论文聚焦社交意识导航任务，旨在让机器人在人类环境中安全、自然地移动。针对传统方法难以适应社会规范的问题，综述了深度强化学习（DRL）方法，分析其算法与网络架构，并评估评价机制与仿真到现实的迁移挑战。指出当前缺乏统一标准与计算效率问题，呼吁发展融合多方法的混合框架与以人为本的基准。**

- **链接: [https://arxiv.org/pdf/2512.00049v1](https://arxiv.org/pdf/2512.00049v1)**

> **作者:** Ibrahim Khalil Kabir; Muhammad Faizan Mysorewala
>
> **摘要:** Socially aware navigation is a fast-evolving research area in robotics that enables robots to move within human environments while adhering to the implicit human social norms. The advent of Deep Reinforcement Learning (DRL) has accelerated the development of navigation policies that enable robots to incorporate these social conventions while effectively reaching their objectives. This survey offers a comprehensive overview of DRL-based approaches to socially aware navigation, highlighting key aspects such as proxemics, human comfort, naturalness, trajectory and intention prediction, which enhance robot interaction in human environments. This work critically analyzes the integration of value-based, policy-based, and actor-critic reinforcement learning algorithms alongside neural network architectures, such as feedforward, recurrent, convolutional, graph, and transformer networks, for enhancing agent learning and representation in socially aware navigation. Furthermore, we examine crucial evaluation mechanisms, including metrics, benchmark datasets, simulation environments, and the persistent challenges of sim-to-real transfer. Our comparative analysis of the literature reveals that while DRL significantly improves safety, and human acceptance over traditional approaches, the field still faces setback due to non-uniform evaluation mechanisms, absence of standardized social metrics, computational burdens that limit scalability, and difficulty in transferring simulation to real robotic hardware applications. We assert that future progress will depend on hybrid approaches that leverage the strengths of multiple approaches and producing benchmarks that balance technical efficiency with human-centered evaluation.
>
---
#### [new 076] DiG-Flow: Discrepancy-Guided Flow Matching for Robust VLA Models
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在分布偏移和复杂多步任务下性能下降的问题，提出DiG-Flow框架。通过计算观测与动作嵌入间的分布差异，以几何正则化引导流匹配，提升表示鲁棒性。工作包括理论保证与轻量级集成，显著改善复杂任务与小样本场景下的性能。**

- **链接: [https://arxiv.org/pdf/2512.01715v1](https://arxiv.org/pdf/2512.01715v1)**

> **作者:** Wanpeng Zhang; Ye Wang; Hao Luo; Haoqi Yuan; Yicheng Feng; Sipeng Zheng; Qin Jin; Zongqing Lu
>
> **摘要:** Vision-Language-Action (VLA) models trained with flow matching have demonstrated impressive capabilities on robotic manipulation tasks. However, their performance often degrades under distribution shift and on complex multi-step tasks, suggesting that the learned representations may not robustly capture task-relevant semantics. We introduce DiG-Flow, a principled framework that enhances VLA robustness through geometric regularization. Our key insight is that the distributional discrepancy between observation and action embeddings provides a meaningful geometric signal: lower transport cost indicates compatible representations, while higher cost suggests potential misalignment. DiG-Flow computes a discrepancy measure between empirical distributions of observation and action embeddings, maps it to a modulation weight via a monotone function, and applies residual updates to the observation embeddings before flow matching. Crucially, this intervention operates at the representation level without modifying the flow matching path or target vector field. We provide theoretical guarantees showing that discrepancy-guided training provably decreases the training objective, and that guided inference refinement converges with contraction. Empirically, DiG-Flow integrates into existing VLA architectures with negligible overhead and consistently improves performance, with particularly pronounced gains on complex multi-step tasks and under limited training data.
>
---
#### [new 077] SpeedAug: Policy Acceleration via Tempo-Enriched Policy and RL Fine-Tuning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对机器人策略执行速度慢的问题，提出SpeedAug框架。通过构建包含多种执行节奏的预训练行为先验，结合强化学习微调，实现高效策略加速。有效提升样本效率，保持高成功率，解决现有方法因分布偏移和探索低效导致的性能瓶颈。**

- **链接: [https://arxiv.org/pdf/2512.00062v1](https://arxiv.org/pdf/2512.00062v1)**

> **作者:** Taewook Nam; Sung Ju Hwang
>
> **摘要:** Recent advances in robotic policy learning have enabled complex manipulation in real-world environments, yet the execution speed of these policies often lags behind hardware capabilities due to the cost of collecting faster demonstrations. Existing works on policy acceleration reinterpret action sequence for unseen execution speed, thereby encountering distributional shifts from the original demonstrations. Reinforcement learning is a promising approach that adapts policies for faster execution without additional demonstration, but its unguided exploration is sample inefficient. We propose SpeedAug, an RL-based policy acceleration framework that efficiently adapts pre-trained policies for faster task execution. SpeedAug constructs behavior prior that encompasses diverse tempos of task execution by pre-training a policy on speed-augmented demonstrations. Empirical results on robotic manipulation benchmarks show that RL fine-tuning initialized from this tempo-enriched policy significantly improves the sample efficiency of existing RL and policy acceleration methods while maintaining high success rate.
>
---
#### [new 078] Magnetic Tactile-Driven Soft Actuator for Intelligent Grasping and Firmness Evaluation
- **分类: cs.RO**

- **简介: 该论文针对软体机器人缺乏集成触觉传感及形变干扰信号的问题，提出SoftMag磁致触觉驱动器。通过共享架构融合传感与驱动，结合多物理场仿真与神经网络解耦，实现高精度力与位置实时感知，并基于探针法评估物体硬度，验证了其在非破坏性质量检测中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.00907v1](https://arxiv.org/pdf/2512.00907v1)**

> **作者:** Chengjin Du; Federico Bernabei; Zhengyin Du; Sergio Decherchi; Matteo Lo Preti; Lucia Beccai
>
> **备注:** 25 pages, 24 figures
>
> **摘要:** Soft robots are powerful tools for manipulating delicate objects, yet their adoption is hindered by two gaps: the lack of integrated tactile sensing and sensor signal distortion caused by actuator deformations. This paper addresses these challenges by introducing the SoftMag actuator: a magnetic tactile-sensorized soft actuator. Unlike systems relying on attached sensors or treating sensing and actuation separately, SoftMag unifies them through a shared architecture while confronting the mechanical parasitic effect, where deformations corrupt tactile signals. A multiphysics simulation framework models this coupling, and a neural-network-based decoupling strategy removes the parasitic component, restoring sensing fidelity. Experiments including indentation, quasi-static and step actuation, and fatigue tests validate the actuator's performance and decoupling effectiveness. Building upon this foundation, the system is extended into a two-finger SoftMag gripper, where a multi-task neural network enables real-time prediction of tri-axial contact forces and position. Furthermore, a probing-based strategy estimates object firmness during grasping. Validation on apricots shows a strong correlation (Pearson r over 0.8) between gripper-estimated firmness and reference measurements, confirming the system's capability for non-destructive quality assessment. Results demonstrate that combining integrated magnetic sensing, learning-based correction, and real-time inference enables a soft robotic platform that adapts its grasp and quantifies material properties. The framework offers an approach for advancing sensorized soft actuators toward intelligent, material-aware robotics.
>
---
#### [new 079] Reinforcement Learning from Implicit Neural Feedback for Human-Aligned Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种基于隐式脑电反馈的强化学习框架（RLIHF），旨在解决传统强化学习在稀疏奖励下难以有效学习的问题。通过解析用户脑电图中的错误相关电位（ErrPs），实现无需显式操作的连续隐式反馈，提升人机协同机器人控制的自然性与效率。**

- **链接: [https://arxiv.org/pdf/2512.00050v1](https://arxiv.org/pdf/2512.00050v1)**

> **作者:** Suzie Kim
>
> **备注:** Master's thesis, Korea University, 2025. arXiv admin note: substantial text overlap with arXiv:2507.13171
>
> **摘要:** Conventional reinforcement learning (RL) approaches often struggle to learn effective policies under sparse reward conditions, necessitating the manual design of complex, task-specific reward functions. To address this limitation, reinforcement learning from human feedback (RLHF) has emerged as a promising strategy that complements hand-crafted rewards with human-derived evaluation signals. However, most existing RLHF methods depend on explicit feedback mechanisms such as button presses or preference labels, which disrupt the natural interaction process and impose a substantial cognitive load on the user. We propose a novel reinforcement learning from implicit human feedback (RLIHF) framework that utilizes non-invasive electroencephalography (EEG) signals, specifically error-related potentials (ErrPs), to provide continuous, implicit feedback without requiring explicit user intervention. The proposed method adopts a pre-trained decoder to transform raw EEG signals into probabilistic reward components, enabling effective policy learning even in the presence of sparse external rewards. We evaluate our approach in a simulation environment built on the MuJoCo physics engine, using a Kinova Gen2 robotic arm to perform a complex pick-and-place task that requires avoiding obstacles while manipulating target objects. The results show that agents trained with decoded EEG feedback achieve performance comparable to those trained with dense, manually designed rewards. These findings validate the potential of using implicit neural feedback for scalable and human-aligned reinforcement learning in interactive robotics.
>
---
#### [new 080] Fast, Robust, Permutation-and-Sign Invariant SO(3) Pattern Alignment
- **分类: cs.RO; cs.CG; cs.CV**

- **简介: 该论文解决旋转集合在SO(3)上的无对应对齐问题，针对时间不同步、异常值及轴约定未知的挑战。通过将旋转分解为球面上的变换基向量，利用快速鲁棒匹配器对齐，并设计置换与符号不变的包装器，实现线性复杂度下的精确对齐，显著提升速度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.00659v1](https://arxiv.org/pdf/2512.00659v1)**

> **作者:** Anik Sarker; Alan T. Asbeck
>
> **摘要:** We address the correspondence-free alignment of two rotation sets on \(SO(3)\), a core task in calibration and registration that is often impeded by missing time alignment, outliers, and unknown axis conventions. Our key idea is to decompose each rotation into its \emph{Transformed Basis Vectors} (TBVs)-three unit vectors on \(S^2\)-and align the resulting spherical point sets per axis using fast, robust matchers (SPMC, FRS, and a hybrid). To handle axis relabels and sign flips, we introduce a \emph{Permutation-and-Sign Invariant} (PASI) wrapper that enumerates the 24 proper signed permutations, scores them via summed correlations, and fuses the per-axis estimates into a single rotation by projection/Karcher mean. The overall complexity remains linear in the number of rotations (\(\mathcal{O}(n)\)), contrasting with \(\mathcal{O}(N_r^3\log N_r)\) for spherical/\(SO(3)\) correlation. Experiments on EuRoC Machine Hall simulations (axis-consistent) and the ETH Hand-Eye benchmark (\texttt{robot\_arm\_real}) (axis-ambiguous) show that our methods are accurate, 6-60x faster than traditional methods, and robust under extreme outlier ratios (up to 90\%), all without correspondence search.
>
---
#### [new 081] Register Any Point: Scaling 3D Point Cloud Registration by Flow Matching
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D点云配准任务，提出基于流匹配的端到端方法，将配准视为条件生成过程，直接生成对齐点云。通过学习点级速度场与测试时刚性约束，实现高效、高精度的单对及多视图配准，尤其在低重叠场景下表现优异，支持多模态、跨尺度应用。**

- **链接: [https://arxiv.org/pdf/2512.01850v1](https://arxiv.org/pdf/2512.01850v1)**

> **作者:** Yue Pan; Tao Sun; Liyuan Zhu; Lucas Nunes; Iro Armeni; Jens Behley; Cyrill Stachniss
>
> **备注:** 22 pages
>
> **摘要:** Point cloud registration aligns multiple unposed point clouds into a common frame, and is a core step for 3D reconstruction and robot localization. In this work, we cast registration as conditional generation: a learned continuous, point-wise velocity field transports noisy points to a registered scene, from which the pose of each view is recovered. Unlike previous methods that conduct correspondence matching to estimate the transformation between a pair of point clouds and then optimize the pairwise transformations to realize multi-view registration, our model directly generates the registered point cloud. With a lightweight local feature extractor and test-time rigidity enforcement, our approach achieves state-of-the-art results on pairwise and multi-view registration benchmarks, particularly with low overlap, and generalizes across scales and sensor modalities. It further supports downstream tasks including relocalization, multi-robot SLAM, and multi-session map merging. Source code available at: https://github.com/PRBonn/RAP.
>
---
#### [new 082] Accelerating Probabilistic Response-Time Analysis: Revised Critical Instant and Optimized Convolution
- **分类: cs.OS; cs.DS; cs.RO**

- **简介: 该论文针对安全关键实时系统中概率响应时间分析的效率与精度问题，提出基于修正临界时刻的优化卷积方法。通过改进卷积合并顺序，显著加速了最坏情况截止期限失败概率（WCDFP）的计算，相比传统方法提速达一个数量级，同时保证估计结果准确且保守。**

- **链接: [https://arxiv.org/pdf/2512.01381v1](https://arxiv.org/pdf/2512.01381v1)**

> **作者:** Hiroto Takahashi; Atsushi Yano; Takuya Azumi
>
> **备注:** 8 pages, 5 figures. Proceedings of APRIS2025
>
> **摘要:** Accurate estimation of the Worst-Case Deadline Failure Probability (WCDFP) has attracted growing attention as a means to provide safety assurances in complex systems such as robotic platforms and autonomous vehicles. WCDFP quantifies the likelihood of deadline misses under the most pessimistic operating conditions, and safe estimation is essential for dependable real-time applications. However, achieving high accuracy in WCDFP estimation often incurs significant computational cost. Recent studies have revealed that the classical assumption of the critical instant, the activation pattern traditionally considered to trigger the worst-case behavior, can lead to underestimation of WCDFP in probabilistic settings. This observation motivates the use of a revised critical instant formulation that more faithfully captures the true worst-case scenario. This paper investigates convolution-based methods for WCDFP estimation under this revised setting and proposes an optimization technique that accelerates convolution by improving the merge order. Extensive experiments with diverse execution-time distributions demonstrate that the proposed optimized Aggregate Convolution reduces computation time by up to an order of magnitude compared to Sequential Convolution, while retaining accurate and safe-sided WCDFP estimates. These results highlight the potential of the approach to provide both efficiency and reliability in probabilistic timing analysis for safety-critical real-time applications.
>
---
#### [new 083] Data-Driven Modeling and Correction of Vehicle Dynamics
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对非自治车辆动力学建模中的模型不确定性与数据挑战，提出数据驱动的修正框架。通过局部参数化时间依赖输入，结合DRIPS（线性代理模型）与FML（深度神经网络）方法，实现高效数据利用下的精准建模与模型误差修正，适用于复杂非线性系统。**

- **链接: [https://arxiv.org/pdf/2512.00289v1](https://arxiv.org/pdf/2512.00289v1)**

> **作者:** Nguyen Ly; Caroline Tatsuoka; Jai Nagaraj; Jacob Levy; Fernando Palafox; David Fridovich-Keil; Hannah Lu
>
> **摘要:** We develop a data-driven framework for learning and correcting non-autonomous vehicle dynamics. Physics-based vehicle models are often simplified for tractability and therefore exhibit inherent model-form uncertainty, motivating the need for data-driven correction. Moreover, non-autonomous dynamics are governed by time-dependent control inputs, which pose challenges in learning predictive models directly from temporal snapshot data. To address these, we reformulate the vehicle dynamics via a local parameterization of the time-dependent inputs, yielding a modified system composed of a sequence of local parametric dynamical systems. We approximate these parametric systems using two complementary approaches. First, we employ the DRIPS (dimension reduction and interpolation in parameter space) methodology to construct efficient linear surrogate models, equipped with lifted observable spaces and manifold-based operator interpolation. This enables data-efficient learning of vehicle models whose dynamics admit accurate linear representations in the lifted spaces. Second, for more strongly nonlinear systems, we employ FML (Flow Map Learning), a deep neural network approach that approximates the parametric evolution map without requiring special treatment of nonlinearities. We further extend FML with a transfer-learning-based model correction procedure, enabling the correction of misspecified prior models using only a sparse set of high-fidelity or experimental measurements, without assuming a prescribed form for the correction term. Through a suite of numerical experiments on unicycle, simplified bicycle, and slip-based bicycle models, we demonstrate that DRIPS offers robust and highly data-efficient learning of non-autonomous vehicle dynamics, while FML provides expressive nonlinear modeling and effective correction of model-form errors under severe data scarcity.
>
---
#### [new 084] Conceptual Evaluation of Deep Visual Stereo Odometry for the MARWIN Radiation Monitoring Robot in Accelerator Tunnels
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究机器人在加速器隧道中的自主定位问题，针对传统方法在低纹理、复杂环境下的局限性，提出采用基于深度视觉立体里程计（DVSO）的方案。通过纯视觉方式估计位姿与深度，结合3D几何约束提升精度，探索其在辐射监测机器人MARWIN上的应用潜力，以实现更灵活、低成本的自主导航。**

- **链接: [https://arxiv.org/pdf/2512.00080v1](https://arxiv.org/pdf/2512.00080v1)**

> **作者:** André Dehne; Juri Zach; Peer Stelldinger
>
> **摘要:** The MARWIN robot operates at the European XFEL to perform autonomous radiation monitoring in long, monotonous accelerator tunnels where conventional localization approaches struggle. Its current navigation concept combines lidar-based edge detection, wheel/lidar odometry with periodic QR-code referencing, and fuzzy control of wall distance, rotation, and longitudinal position. While robust in predefined sections, this design lacks flexibility for unknown geometries and obstacles. This paper explores deep visual stereo odometry (DVSO) with 3D-geometric constraints as a focused alternative. DVSO is purely vision-based, leveraging stereo disparity, optical flow, and self-supervised learning to jointly estimate depth and ego-motion without labeled data. For global consistency, DVSO can subsequently be fused with absolute references (e.g., landmarks) or other sensors. We provide a conceptual evaluation for accelerator tunnel environments, using the European XFEL as a case study. Expected benefits include reduced scale drift via stereo, low-cost sensing, and scalable data collection, while challenges remain in low-texture surfaces, lighting variability, computational load, and robustness under radiation. The paper defines a research agenda toward enabling MARWIN to navigate more autonomously in constrained, safety-critical infrastructures.
>
---
#### [new 085] Dependent Reachable Sets for the Constant Bearing Pursuit Strategy
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文研究恒定航向追击策略下的跟随者可达集问题，属于多智能体系统中的路径规划与可达性分析任务。针对追击策略导致的依赖性可达集，提出并求解了新的优化问题，推导其几何边界，并通过仿真揭示其形状特征。**

- **链接: [https://arxiv.org/pdf/2512.00273v1](https://arxiv.org/pdf/2512.00273v1)**

> **作者:** Venkata Ramana Makkapati; Tulasi Ram Vechalapu; Vinodhini Comandur; Seth Hutchinson
>
> **备注:** This work has been submitted to a journal for possible publication
>
> **摘要:** This paper introduces a novel reachability problem for the scenario where one agent follows another agent using the constant bearing pursuit strategy, and analyzes the geometry of the reachable set of the follower. Key theoretical results are derived, providing bounds for the associated dependent reachable set. Simulation results are presented to empirically establish the shape of the dependent reachable set. In the process, an original optimization problem for the constant bearing strategy is formulated and analyzed.
>
---
#### [new 086] Describe Anything Anywhere At Any Moment
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DAAAM框架，解决大尺度、实时4D场景理解中语义描述与几何定位的平衡问题。通过优化前端加速局部描述生成，构建分层4D场景图，实现高精度时空记忆。在NaVQA和SG3D基准上显著提升问答与任务接地性能，优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.00565v1](https://arxiv.org/pdf/2512.00565v1)**

> **作者:** Nicolas Gorlo; Lukas Schmid; Luca Carlone
>
> **备注:** 14 pages, 5 figures, 6 tables
>
> **摘要:** Computer vision and robotics applications ranging from augmented reality to robot autonomy in large-scale environments require spatio-temporal memory frameworks that capture both geometric structure for accurate language-grounding as well as semantic detail. Existing methods face a tradeoff, where producing rich open-vocabulary descriptions comes at the expense of real-time performance when these descriptions have to be grounded in 3D. To address these challenges, we propose Describe Anything, Anywhere, at Any Moment (DAAAM), a novel spatio-temporal memory framework for large-scale and real-time 4D scene understanding. DAAAM introduces a novel optimization-based frontend to infer detailed semantic descriptions from localized captioning models, such as the Describe Anything Model (DAM), leveraging batch processing to speed up inference by an order of magnitude for online processing. It leverages such semantic understanding to build a hierarchical 4D scene graph (SG), which acts as an effective globally spatially and temporally consistent memory representation. DAAAM constructs 4D SGs with detailed, geometrically grounded descriptions while maintaining real-time performance. We show that DAAAM's 4D SG interfaces well with a tool-calling agent for inference and reasoning. We thoroughly evaluate DAAAM in the complex task of spatio-temporal question answering on the NaVQA benchmark and show its generalization capabilities for sequential task grounding on the SG3D benchmark. We further curate an extended OC-NaVQA benchmark for large-scale and long-time evaluations. DAAAM achieves state-of-the-art results in both tasks, improving OC-NaVQA question accuracy by 53.6%, position errors by 21.9%, temporal errors by 21.6%, and SG3D task grounding accuracy by 27.8% over the most competitive baselines, respectively. We release our data and code open-source.
>
---
#### [new 087] MM-ACT: Learn from Multimodal Parallel Generation to Act
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出MM-ACT，一种统一的视觉-语言-动作模型，旨在解决机器人通用策略中语义理解与环境交互的难题。通过多模态并行生成与共享上下文学习，提升任务规划与动作预测能力，在仿真与真实机器人上均取得优异性能。**

- **链接: [https://arxiv.org/pdf/2512.00975v1](https://arxiv.org/pdf/2512.00975v1)**

> **作者:** Haotian Liang; Xinyi Chen; Bin Wang; Mingkang Chen; Yitian Liu; Yuhao Zhang; Zanxin Chen; Tianshuo Yang; Yilun Chen; Jiangmiao Pang; Dong Liu; Xiaokang Yang; Yao Mu; Wenqi Shao; Ping Luo
>
> **备注:** 17 pages
>
> **摘要:** A generalist robotic policy needs both semantic understanding for task planning and the ability to interact with the environment through predictive capabilities. To tackle this, we present MM-ACT, a unified Vision-Language-Action (VLA) model that integrates text, image, and action in shared token space and performs generation across all three modalities. MM-ACT adopts a re-mask parallel decoding strategy for text and image generation, and employs a one-step parallel decoding strategy for action generation to improve efficiency. We introduce Context-Shared Multimodal Learning, a unified training paradigm that supervises generation in all three modalities from a shared context, enhancing action generation through cross-modal learning. Experiments were conducted on the LIBERO simulation and Franka real-robot setups as well as RoboTwin2.0 to assess in-domain and out-of-domain performances respectively. Our approach achieves a success rate of 96.3% on LIBERO, 72.0% across three tasks of real Franka, and 52.38% across eight bimanual tasks of RoboTwin2.0 with an additional gain of 9.25% from cross-modal learning. We release our codes, models and data at https://github.com/HHYHRHY/MM-ACT.
>
---
#### [new 088] Real-Time On-the-Go Annotation Framework Using YOLO for Automated Dataset Generation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对农业等领域实时对象检测中数据标注效率低的问题，提出基于YOLO的边缘设备实时标注框架。通过在图像采集时即时标注，显著减少数据准备时间，验证了预训练与单类标注配置在收敛性、性能和鲁棒性上的优势。**

- **链接: [https://arxiv.org/pdf/2512.01165v1](https://arxiv.org/pdf/2512.01165v1)**

> **作者:** Mohamed Abdallah Salem; Ahmed Harb Rabia
>
> **备注:** Copyright 2025 IEEE. This is the author's version of the work that has been accepted for publication in Proceedings of the 5. Interdisciplinary Conference on Electrics and Computer (INTCEC 2025) 15-16 September 2025, Chicago-USA. The final version of record is available at: https://doi.org/10.1109/INTCEC65580.2025.11256048
>
> **摘要:** Efficient and accurate annotation of datasets remains a significant challenge for deploying object detection models such as You Only Look Once (YOLO) in real-world applications, particularly in agriculture where rapid decision-making is critical. Traditional annotation techniques are labor-intensive, requiring extensive manual labeling post data collection. This paper presents a novel real-time annotation approach leveraging YOLO models deployed on edge devices, enabling immediate labeling during image capture. To comprehensively evaluate the efficiency and accuracy of our proposed system, we conducted an extensive comparative analysis using three prominent YOLO architectures (YOLOv5, YOLOv8, YOLOv12) under various configurations: single-class versus multi-class annotation and pretrained versus scratch-based training. Our analysis includes detailed statistical tests and learning dynamics, demonstrating significant advantages of pretrained and single-class configurations in terms of model convergence, performance, and robustness. Results strongly validate the feasibility and effectiveness of our real-time annotation framework, highlighting its capability to drastically reduce dataset preparation time while maintaining high annotation quality.
>
---
#### [new 089] How do trout regulate patterns of muscle contraction to optimize propulsive efficiency during steady swimming
- **分类: physics.flu-dyn; cs.AI; cs.RO**

- **简介: 该论文研究鱼类高效游泳的神经肌肉调控机制，旨在揭示鳟鱼在稳定游动中如何优化肌肉收缩以提升推进效率。通过构建融合多体动力学与流固耦合的数字鳟鱼模型，利用深度强化学习实现肌肉激活的时空控制，发现肌节耦合范围、收缩时长及相位滞后对能量消耗和波形稳定性至关重要，为仿生水下机器人设计提供理论支持。**

- **链接: [https://arxiv.org/pdf/2512.01218v1](https://arxiv.org/pdf/2512.01218v1)**

> **作者:** Tao Li; Chunze Zhang; Weiwei Yao; Junzhao He; Ji Hou; Qin Zhou; Lu Zhang
>
> **摘要:** Understanding efficient fish locomotion offers insights for biomechanics, fluid dynamics, and engineering. Traditional studies often miss the link between neuromuscular control and whole-body movement. To explore energy transfer in carangiform swimming, we created a bio-inspired digital trout. This model combined multibody dynamics, Hill-type muscle modeling, and a high-fidelity fluid-structure interaction algorithm, accurately replicating a real trout's form and properties. Using deep reinforcement learning, the trout's neural system achieved hierarchical spatiotemporal control of muscle activation. We systematically examined how activation strategies affect speed and energy use. Results show that axial myomere coupling-with activation spanning over 0.5 body lengths-is crucial for stable body wave propagation. Moderate muscle contraction duration ([0.1,0.3] of a tail-beat cycle) lets the body and fluid act as a passive damping system, cutting energy use. Additionally, the activation phase lag of myomeres shapes the body wave; if too large, it causes antagonistic contractions that hinder thrust. These findings advance bio-inspired locomotion understanding and aid energy-efficient underwater system design.
>
---
#### [new 090] SPARK: Sim-ready Part-level Articulated Reconstruction with VLM Knowledge
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出SPARK框架，解决从单张RGB图像生成可模拟的关节式3D物体的问题。利用视觉语言模型提取粗略URDF参数并生成部件参考图，结合扩散Transformer生成一致的部件与整体形状，并通过可微正向运动学与渲染优化关节参数，实现物理一致、可直接用于机器人操作等下游任务的高质量资产生成。**

- **链接: [https://arxiv.org/pdf/2512.01629v1](https://arxiv.org/pdf/2512.01629v1)**

> **作者:** Yumeng He; Ying Jiang; Jiayin Lu; Yin Yang; Chenfanfu Jiang
>
> **摘要:** Articulated 3D objects are critical for embodied AI, robotics, and interactive scene understanding, yet creating simulation-ready assets remains labor-intensive and requires expert modeling of part hierarchies and motion structures. We introduce SPARK, a framework for reconstructing physically consistent, kinematic part-level articulated objects from a single RGB image. Given an input image, we first leverage VLMs to extract coarse URDF parameters and generate part-level reference images. We then integrate the part-image guidance and the inferred structure graph into a generative diffusion transformer to synthesize consistent part and complete shapes of articulated objects. To further refine the URDF parameters, we incorporate differentiable forward kinematics and differentiable rendering to optimize joint types, axes, and origins under VLM-generated open-state supervision. Extensive experiments show that SPARK produces high-quality, simulation-ready articulated assets across diverse categories, enabling downstream applications such as robotic manipulation and interaction modeling.
>
---
#### [new 091] GrndCtrl: Grounding World Models via Self-Supervised Reward Alignment
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对视频世界模型缺乏几何接地的问题，提出GrndCtrl框架，通过自监督奖励对齐实现物理可验证结构的建模。利用姿态循环一致性、深度重投影等多奖励机制，基于GRPO优化，提升模型在户外环境中的空间一致性和导航稳定性，解决生成模型与真实物理世界脱节的难题。**

- **链接: [https://arxiv.org/pdf/2512.01952v1](https://arxiv.org/pdf/2512.01952v1)**

> **作者:** Haoyang He; Jay Patrikar; Dong-Ki Kim; Max Smith; Daniel McGann; Ali-akbar Agha-mohammadi; Shayegan Omidshafiei; Sebastian Scherer
>
> **摘要:** Recent advances in video world modeling have enabled large-scale generative models to simulate embodied environments with high visual fidelity, providing strong priors for prediction, planning, and control. Yet, despite their realism, these models often lack geometric grounding, limiting their use in navigation tasks that require spatial coherence and long-horizon stability. We introduce Reinforcement Learning with World Grounding (RLWG), a self-supervised post-training framework that aligns pretrained world models with a physically verifiable structure through geometric and perceptual rewards. Analogous to reinforcement learning from verifiable feedback (RLVR) in language models, RLWG can use multiple rewards that measure pose cycle-consistency, depth reprojection, and temporal coherence. We instantiate this framework with GrndCtrl, a reward-aligned adaptation method based on Group Relative Policy Optimization (GRPO), yielding world models that maintain stable trajectories, consistent geometry, and reliable rollouts for embodied navigation. Like post-training alignment in large language models, GrndCtrl leverages verifiable rewards to bridge generative pretraining and grounded behavior, achieving superior spatial coherence and navigation stability over supervised fine-tuning in outdoor environments.
>
---
#### [new 092] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对工业场景中大量未标注视频数据难以用于视觉-语言-动作（VLA）模型预训练的问题，提出一种端到端无监督框架。通过轻量级运动编码器与基于“潜在动作能量”的动作分割算法，自动提取语义一致的动作原型，生成可直接用于VLA预训练的结构化数据。**

- **链接: [https://arxiv.org/pdf/2511.21428v1](https://arxiv.org/pdf/2511.21428v1)**

> **作者:** Jiajie Zhang; Sören Schwertfeger; Alexander Kleiner
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** We present a novel unsupervised framework to unlock vast unlabeled human demonstration data from continuous industrial video streams for Vision-Language-Action (VLA) model pre-training. Our method first trains a lightweight motion tokenizer to encode motion dynamics, then employs an unsupervised action segmenter leveraging a novel "Latent Action Energy" metric to discover and segment semantically coherent action primitives. The pipeline outputs both segmented video clips and their corresponding latent action sequences, providing structured data directly suitable for VLA pre-training. Evaluations on public benchmarks and a proprietary electric motor assembly dataset demonstrate effective segmentation of key tasks performed by humans at workstations. Further clustering and quantitative assessment via a Vision-Language Model confirm the semantic coherence of the discovered action primitives. To our knowledge, this is the first fully automated end-to-end system for extracting and organizing VLA pre-training data from unstructured industrial videos, offering a scalable solution for embodied AI integration in manufacturing.
>
---
#### [new 093] Active Learning of Fractional-Order Viscoelastic Model Parameters for Realistic Haptic Rendering
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对医疗模拟器中生物组织触觉渲染的现实感问题，提出基于人机协同主动学习的分数阶黏弹性模型参数优化方法。通过个体化反馈优化参数，并聚合群体数据构建通用感知映射，实现跨人群的高真实感触觉呈现，提升医学训练模拟器的仿真效果。**

- **链接: [https://arxiv.org/pdf/2512.00667v1](https://arxiv.org/pdf/2512.00667v1)**

> **作者:** Harun Tolasa; Gorkem Gemalmaz; Volkan Patoglu
>
> **备注:** This work has been submitted to the IEEE for possible publication. 14 pages, 9 figures
>
> **摘要:** Effective medical simulators necessitate realistic haptic rendering of biological tissues that display viscoelastic material properties, such as creep and stress relaxation. Fractional-order models provide an effective means of describing intrinsically time-dependent viscoelastic dynamics with few parameters, as these models can naturally capture memory effects. However, due to the unintuitive frequency-dependent coupling between the order of the fractional element and the other parameters, determining appropriate parameters for fractional-order models that yield high perceived realism remains a significant challenge. In this study, we propose a systematic means of determining the parameters of fractional-order viscoelastic models that optimizes the perceived realism of haptic rendering across general populations. First, we demonstrate that the parameters of fractional-order models can be effectively optimized through active learning, via qualitative feedback-based human-in-the-loop~(HiL) optimizations, to ensure consistently high realism ratings for each individual. Second, we propose a rigorous method to combine HiL optimization results to form an aggregate perceptual map trained on the entire dataset and demonstrate the selection of population-level optimal parameters from this representation that are broadly perceived as realistic across general populations. Finally, we provide evidence of the effectiveness of the generalized fractional-order viscoelastic model parameters by characterizing their perceived realism through human-subject experiments. Overall, generalized fractional-order viscoelastic models established through the proposed HiL optimization and aggregation approach possess the potential to significantly improve the sim-to-real transition performance of medical training simulators.
>
---
#### [new 094] Visual Sync: Multi-Camera Synchronization via Cross-View Object Motion
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对多摄像头视频同步难题，提出VisualSync框架。通过利用跨视角运动点的对极几何约束，结合3D重建与特征匹配，联合优化以毫米级精度估计时间偏移。解决了无标定、非同步视频在复杂场景下的自动同步问题。**

- **链接: [https://arxiv.org/pdf/2512.02017v1](https://arxiv.org/pdf/2512.02017v1)**

> **作者:** Shaowei Liu; David Yifan Yao; Saurabh Gupta; Shenlong Wang
>
> **备注:** Accepted to NeurIPS 2025. Project page: https://stevenlsw.github.io/visualsync/
>
> **摘要:** Today, people can easily record memorable moments, ranging from concerts, sports events, lectures, family gatherings, and birthday parties with multiple consumer cameras. However, synchronizing these cross-camera streams remains challenging. Existing methods assume controlled settings, specific targets, manual correction, or costly hardware. We present VisualSync, an optimization framework based on multi-view dynamics that aligns unposed, unsynchronized videos at millisecond accuracy. Our key insight is that any moving 3D point, when co-visible in two cameras, obeys epipolar constraints once properly synchronized. To exploit this, VisualSync leverages off-the-shelf 3D reconstruction, feature matching, and dense tracking to extract tracklets, relative poses, and cross-view correspondences. It then jointly minimizes the epipolar error to estimate each camera's time offset. Experiments on four diverse, challenging datasets show that VisualSync outperforms baseline methods, achieving an median synchronization error below 50 ms.
>
---
#### [new 095] TrajDiff: End-to-end Autonomous Driving without Perception Annotation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对端到端自动驾驶中感知标注成本高的问题，提出TrajDiff框架，无需感知标注即可从原始传感器数据生成合理轨迹。通过轨迹导向的BEV编码器和扩散变压器，直接生成多样且合理的驾驶轨迹，实现全注释自由的规划，显著提升性能并验证了数据规模收益。**

- **链接: [https://arxiv.org/pdf/2512.00723v1](https://arxiv.org/pdf/2512.00723v1)**

> **作者:** Xingtai Gui; Jianbo Zhao; Wencheng Han; Jikai Wang; Jiahao Gong; Feiyang Tan; Cheng-zhong Xu; Jianbing Shen
>
> **摘要:** End-to-end autonomous driving systems directly generate driving policies from raw sensor inputs. While these systems can extract effective environmental features for planning, relying on auxiliary perception tasks, developing perception annotation-free planning paradigms has become increasingly critical due to the high cost of manual perception annotation. In this work, we propose TrajDiff, a Trajectory-oriented BEV Conditioned Diffusion framework that establishes a fully perception annotation-free generative method for end-to-end autonomous driving. TrajDiff requires only raw sensor inputs and future trajectory, constructing Gaussian BEV heatmap targets that inherently capture driving modalities. We design a simple yet effective trajectory-oriented BEV encoder to extract the TrajBEV feature without perceptual supervision. Furthermore, we introduce Trajectory-oriented BEV Diffusion Transformer (TB-DiT), which leverages ego-state information and the predicted TrajBEV features to directly generate diverse yet plausible trajectories, eliminating the need for handcrafted motion priors. Beyond architectural innovations, TrajDiff enables exploration of data scaling benefits in the annotation-free setting. Evaluated on the NAVSIM benchmark, TrajDiff achieves 87.5 PDMS, establishing state-of-the-art performance among all annotation-free methods. With data scaling, it further improves to 88.5 PDMS, which is comparable to advanced perception-based approaches. Our code and model will be made publicly available.
>
---
#### [new 096] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对对称性破缺环境中的强化学习问题，提出部分群不变马尔可夫决策过程（PI-MDP），通过选择性应用群不变或标准贝尔曼更新，缓解局部对称性破坏导致的误差传播。基于此框架，设计了PE-DQN和PE-SAC算法，在网格世界、运动控制和操作任务中显著提升样本效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.00915v1](https://arxiv.org/pdf/2512.00915v1)**

> **作者:** Junwoo Chang; Minwoo Park; Joohwan Seo; Roberto Horowitz; Jongmin Lee; Jongeun Choi
>
> **备注:** 27 pages, 10 figures
>
> **摘要:** Group symmetries provide a powerful inductive bias for reinforcement learning (RL), enabling efficient generalization across symmetric states and actions via group-invariant Markov Decision Processes (MDPs). However, real-world environments almost never realize fully group-invariant MDPs; dynamics, actuation limits, and reward design usually break symmetries, often only locally. Under group-invariant Bellman backups for such cases, local symmetry-breaking introduces errors that propagate across the entire state-action space, resulting in global value estimation errors. To address this, we introduce Partially group-Invariant MDP (PI-MDP), which selectively applies group-invariant or standard Bellman backups depending on where symmetry holds. This framework mitigates error propagation from locally broken symmetries while maintaining the benefits of equivariance, thereby enhancing sample efficiency and generalizability. Building on this framework, we present practical RL algorithms -- Partially Equivariant (PE)-DQN for discrete control and PE-SAC for continuous control -- that combine the benefits of equivariance with robustness to symmetry-breaking. Experiments across Grid-World, locomotion, and manipulation benchmarks demonstrate that PE-DQN and PE-SAC significantly outperform baselines, highlighting the importance of selective symmetry exploitation for robust and sample-efficient RL.
>
---
#### [new 097] SwiftVLA: Unlocking Spatiotemporal Dynamics for Lightweight VLA Models at Minimal Overhead
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对轻量级视觉-语言-动作（VLA）模型在时空推理能力不足的问题，提出SwiftVLA架构。通过4D视觉几何变换器与时间缓存提取4D特征，引入融合令牌增强多模态表示，并采用掩码重建策略使模型学习有效4D表征，最终实现高效推理。**

- **链接: [https://arxiv.org/pdf/2512.00903v1](https://arxiv.org/pdf/2512.00903v1)**

> **作者:** Chaojun Ni; Cheng Chen; Xiaofeng Wang; Zheng Zhu; Wenzhao Zheng; Boyuan Wang; Tianrun Chen; Guosheng Zhao; Haoyun Li; Zhehao Dong; Qiang Zhang; Yun Ye; Yang Wang; Guan Huang; Wenjun Mei
>
> **摘要:** Vision-Language-Action (VLA) models built on pretrained Vision-Language Models (VLMs) show strong potential but are limited in practicality due to their large parameter counts. To mitigate this issue, using a lightweight VLM has been explored, but it compromises spatiotemporal reasoning. Although some methods suggest that incorporating additional 3D inputs can help, they usually rely on large VLMs to fuse 3D and 2D inputs and still lack temporal understanding. Therefore, we propose SwiftVLA, an architecture that enhances a compact model with 4D understanding while preserving design efficiency. Specifically, our approach features a pretrained 4D visual geometry transformer with a temporal cache that extracts 4D features from 2D images. Then, to enhance the VLM's ability to exploit both 2D images and 4D features, we introduce Fusion Tokens, a set of learnable tokens trained with a future prediction objective to generate unified representations for action generation. Finally, we introduce a mask-and-reconstruct strategy that masks 4D inputs to the VLM and trains the VLA to reconstruct them, enabling the VLM to learn effective 4D representations and allowing the 4D branch to be dropped at inference with minimal performance loss. Experiments in real and simulated environments show that SwiftVLA outperforms lightweight baselines and rivals VLAs up to 7 times larger, achieving comparable performance on edge devices while being 18 times faster and reducing memory footprint by 12 times.
>
---
#### [new 098] Sigma: The Key for Vision-Language-Action Models toward Telepathic Alignment
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对人形机器人认知系统中语义与连续控制间缺乏可时变思维空间的问题，提出名为Sigma的视觉-语言-动作模型。基于pi05_base模型，通过数据预处理、LoRA微调与推理适配器优化，实现无需重训练的意图驱动行为与语义对齐，验证了其在多时标控制误差下降下的稳定性与“心灵感应”式通信能力。**

- **链接: [https://arxiv.org/pdf/2512.00783v1](https://arxiv.org/pdf/2512.00783v1)**

> **作者:** Libo Wang
>
> **备注:** The Sigma model has been open-sourced on Hugging Face. Weights, dataset, some scripts, and logs are all available. The link is: https://huggingface.co/Veltraxor/Sigma
>
> **摘要:** To address the gap in humanoid robot cognitive systems regarding the lack of a time-updable mediating thought space between semantics and continuous control, this study constructs and trains a VLA model named "Sigma" that runs on a single RTX 4090. It uses the open-source pi05_base model as a foundation and preprocesses svla_so101_pickplace into a training dataset. The researcher independently designed an architecture for a vision-language-action model that combines deep semantic understanding and association to achieve telepathic communication. The training process involved repeated optimizations of data preprocessing, LoRA fine-tuning, and the inference-stage adapter. The experiment employed offline closed-loop replay, comparing Sigma with the untuned pure pi05_base_base model under data conditions. Results showed that Sigma exhibited a stable decrease in control MSE across vector, fragment, and entire trajectory timescales, while maintaining the telepathy norm and semantic-text alignment quality unchanged. It demonstrates that mind-responsive alignment control is quantified through an architecture that combines deep understanding of semantics and association without retraining the base model, which provides reproducible experience for semantic alignment and intention-driven behavior in humanoid robots.
>
---
#### [new 099] Data-Centric Visual Development for Self-Driving Labs
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自驱动实验室中因数据稀缺导致的视觉模型训练难题，聚焦移液操作的精确检测。提出融合真实与虚拟数据的混合生成策略，通过人机协作采集真实数据，结合条件生成图像扩充数据集，实现类平衡，显著提升泡泡检测模型精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.02018v1](https://arxiv.org/pdf/2512.02018v1)**

> **作者:** Anbang Liu; Guanzhong Hu; Jiayi Wang; Ping Guo; Han Liu
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** Self-driving laboratories offer a promising path toward reducing the labor-intensive, time-consuming, and often irreproducible workflows in the biological sciences. Yet their stringent precision requirements demand highly robust models whose training relies on large amounts of annotated data. However, this kind of data is difficult to obtain in routine practice, especially negative samples. In this work, we focus on pipetting, the most critical and precision sensitive action in SDLs. To overcome the scarcity of training data, we build a hybrid pipeline that fuses real and virtual data generation. The real track adopts a human-in-the-loop scheme that couples automated acquisition with selective human verification to maximize accuracy with minimal effort. The virtual track augments the real data using reference-conditioned, prompt-guided image generation, which is further screened and validated for reliability. Together, these two tracks yield a class-balanced dataset that enables robust bubble detection training. On a held-out real test set, a model trained entirely on automatically acquired real images reaches 99.6% accuracy, and mixing real and generated data during training sustains 99.4% accuracy while reducing collection and review load. Our approach offers a scalable and cost-effective strategy for supplying visual feedback data to SDL workflows and provides a practical solution to data scarcity in rare event detection and broader vision tasks.
>
---
#### [new 100] Image Generation as a Visual Planner for Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究视觉规划在机器人操作中的应用，旨在解决传统视频生成模型依赖大量特定数据且泛化能力差的问题。作者提出利用预训练图像生成模型（如扩散模型）通过轻量微调（LoRA）实现文本或轨迹条件下的视频生成，使其能作为无需复杂时序建模的视觉规划器，有效生成符合指令的连贯机器人操作视频。**

- **链接: [https://arxiv.org/pdf/2512.00532v1](https://arxiv.org/pdf/2512.00532v1)**

> **作者:** Ye Pang
>
> **备注:** 11 pages 9 figures Under review at CVPR 2026
>
> **摘要:** Generating realistic robotic manipulation videos is an important step toward unifying perception, planning, and action in embodied agents. While existing video diffusion models require large domain-specific datasets and struggle to generalize, recent image generation models trained on language-image corpora exhibit strong compositionality, including the ability to synthesize temporally coherent grid images. This suggests a latent capacity for video-like generation even without explicit temporal modeling. We explore whether such models can serve as visual planners for robots when lightly adapted using LoRA finetuning. We propose a two-part framework that includes: (1) text-conditioned generation, which uses a language instruction and the first frame, and (2) trajectory-conditioned generation, which uses a 2D trajectory overlay and the same initial frame. Experiments on the Jaco Play dataset, Bridge V2, and the RT1 dataset show that both modes produce smooth, coherent robot videos aligned with their respective conditions. Our findings indicate that pretrained image generators encode transferable temporal priors and can function as video-like robotic planners under minimal supervision. Code is released at \href{https://github.com/pangye202264690373/Image-Generation-as-a-Visual-Planner-for-Robotic-Manipulation}{https://github.com/pangye202264690373/Image-Generation-as-a-Visual-Planner-for-Robotic-Manipulation}.
>
---
#### [new 101] Forecasting in Offline Reinforcement Learning for Non-stationary Environments
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对非平稳环境下离线强化学习的性能下降问题，提出FORL框架。通过条件扩散生成候选状态与零样本时间序列模型，实现对未知非平稳性的预测，提升代理在突发、时变偏移下的鲁棒性。在真实时间序列增强的基准上验证了其优越性。**

- **链接: [https://arxiv.org/pdf/2512.01987v1](https://arxiv.org/pdf/2512.01987v1)**

> **作者:** Suzan Ece Ada; Georg Martius; Emre Ugur; Erhan Oztop
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems, NeurIPS 2025
>
> **摘要:** Offline Reinforcement Learning (RL) provides a promising avenue for training policies from pre-collected datasets when gathering additional interaction data is infeasible. However, existing offline RL methods often assume stationarity or only consider synthetic perturbations at test time, assumptions that often fail in real-world scenarios characterized by abrupt, time-varying offsets. These offsets can lead to partial observability, causing agents to misperceive their true state and degrade performance. To overcome this challenge, we introduce Forecasting in Non-stationary Offline RL (FORL), a framework that unifies (i) conditional diffusion-based candidate state generation, trained without presupposing any specific pattern of future non-stationarity, and (ii) zero-shot time-series foundation models. FORL targets environments prone to unexpected, potentially non-Markovian offsets, requiring robust agent performance from the onset of each episode. Empirical evaluations on offline RL benchmarks, augmented with real-world time-series data to simulate realistic non-stationarity, demonstrate that FORL consistently improves performance compared to competitive baselines. By integrating zero-shot forecasting with the agent's experience, we aim to bridge the gap between offline RL and the complexities of real-world, non-stationary environments.
>
---
#### [new 102] X-SYCON: Xylem-Inspired Passive Gradient Control for Communication-Free Swarm Response in Dynamic Disaster Environments
- **分类: physics.soc-ph; cs.MA; cs.RO; eess.SY; nlin.AO**

- **简介: 该论文提出X-SYCON，一种受木质部启发的无通信多智能体系统，用于动态灾难环境下的协同响应。通过被动场动力学实现自组织协调，解决通信中断下的资源分配与避障问题。工作包括设计基于需求与障碍场的局部效用函数，验证其在复杂环境中的高效性与可调性，并建立连续介质模型预测响应范围与容量特性。**

- **链接: [https://arxiv.org/pdf/2512.00018v1](https://arxiv.org/pdf/2512.00018v1)**

> **作者:** Arthur Ji Sung Baek; Geoffrey Martin
>
> **备注:** Preprint v1. 10 pages, 11 figures. Code and data: https://github.com/arthurbaek/biovascular-swarm-robotics
>
> **摘要:** We present X-SYCON, a xylem-inspired multi-agent architecture in which coordination emerges from passive field dynamics rather than explicit planning or communication. Incidents (demands) and obstructions (hazards) continually write diffusing and decaying scalar fields, and agents greedily ascend a local utility $U=φ_{\mathrm{DE}}-κ\,φ_{\mathrm{HZ}}$ with light anti-congestion and separation. A beaconing rule triggered on first contact temporarily deepens the local demand sink, accelerating completion without reducing time-to-first-response. Across dynamic, partially blocked simulated environments, we observe low miss rates and stable throughput with interpretable, tunable trade-offs over carrier count, arrival rate, hazard density, and hazard sensitivity $κ$. We derive that a characteristic hydraulic length scale $\ell\approx\sqrt{D/λ}$ predicts recruitment range in a continuum approximation, and we provide a work-conservation (Ohm-law) bound consistent with sublinear capacity scaling with team size. Empirically: (i) soft hazard penalties yield fewer misses when obstacles already block motion; (ii) throughput saturates sublinearly with carriers while reliability improves sharply; (iii) stronger arrivals can reduce misses by sustaining sinks that recruit help; and (iv) phase-stability regions shrink with hazard density but are recovered by more carriers or higher arrivals. We refer to X-SYCON as an instance of Distributed Passive Computation and Control, and we evaluate it in simulations modeling communication-denied disaster response and other constrained sensing-action regimes.
>
---
## 更新

#### [replaced 001] Ensuring Force Safety in Vision-Guided Robotic Manipulation via Implicit Tactile Calibration
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对视觉引导机器人操作中因缺乏触觉感知导致的力安全问题，提出SafeDiff框架。通过隐式触觉校准实时优化状态规划，生成安全动作轨迹。构建了大规模仿真数据集SafeDoorManip50k，实验证明可有效降低开门时的有害力。**

- **链接: [https://arxiv.org/pdf/2412.10349v2](https://arxiv.org/pdf/2412.10349v2)**

> **作者:** Lai Wei; Jiahua Ma; Yibo Hu; Ruimao Zhang
>
> **备注:** Website URL: see https://i-am-future.github.io/safediff/
>
> **摘要:** In dynamic environments, robots often encounter constrained movement trajectories when manipulating objects with specific properties, such as doors. Therefore, applying the appropriate force is crucial to prevent damage to both the robots and the objects. However, current vision-guided robot state generation methods often falter in this regard, as they lack the integration of tactile perception. To tackle this issue, this paper introduces a novel state diffusion framework termed SafeDiff. It generates a prospective state sequence from the current robot state and visual context observation while incorporating real-time tactile feedback to refine the sequence. As far as we know, this is the first study specifically focused on ensuring force safety in robotic manipulation. It significantly enhances the rationality of state planning, and the safe action trajectory is derived from inverse dynamics based on this refined planning. In practice, unlike previous approaches that concatenate visual and tactile data to generate future robot state sequences, our method employs tactile data as a calibration signal to adjust the robot's state within the state space implicitly. Additionally, we've developed a large-scale simulation dataset called SafeDoorManip50k, offering extensive multimodal data to train and evaluate the proposed method. Extensive experiments show that our visual-tactile model substantially mitigates the risk of harmful forces in the door opening, across both simulated and real-world settings.
>
---
#### [replaced 002] GigaWorld-0: World Models as Data Engine to Empower Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GigaWorld-0，一个用于具身智能的统一世界模型数据引擎。针对真实交互数据稀缺与训练成本高的问题，构建视频与3D生成协同的合成数据系统，实现高保真、物理可信、可控的具身交互数据生成。通过高效训练框架支持大规模训练，使VLA模型在无真实数据参与下显著提升机器人任务泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.19861v2](https://arxiv.org/pdf/2511.19861v2)**

> **作者:** GigaWorld Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jiagang Zhu; Kerui Li; Mengyuan Xu; Qiuping Deng; Siting Wang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yankai Wang; Yu Cao; Yifan Chang; Yuan Xu; Yun Ye; Yang Wang; Yukun Zhou; Zhengyuan Zhang; Zhehao Dong; Zheng Zhu
>
> **备注:** Project Page: https://giga-world-0.github.io/
>
> **摘要:** World models are emerging as a foundational paradigm for scalable, data-efficient embodied AI. In this work, we present GigaWorld-0, a unified world model framework designed explicitly as a data engine for Vision-Language-Action (VLA) learning. GigaWorld-0 integrates two synergistic components: GigaWorld-0-Video, which leverages large-scale video generation to produce diverse, texture-rich, and temporally coherent embodied sequences under fine-grained control of appearance, camera viewpoint, and action semantics; and GigaWorld-0-3D, which combines 3D generative modeling, 3D Gaussian Splatting reconstruction, physically differentiable system identification, and executable motion planning to ensure geometric consistency and physical realism. Their joint optimization enables the scalable synthesis of embodied interaction data that is visually compelling, spatially coherent, physically plausible, and instruction-aligned. Training at scale is made feasible through our efficient GigaTrain framework, which exploits FP8-precision and sparse attention to drastically reduce memory and compute requirements. We conduct comprehensive evaluations showing that GigaWorld-0 generates high-quality, diverse, and controllable data across multiple dimensions. Critically, VLA model (e.g., GigaBrain-0) trained on GigaWorld-0-generated data achieve strong real-world performance, significantly improving generalization and task success on physical robots without any real-world interaction during training.
>
---
#### [replaced 003] SRPO: Self-Referential Policy Optimization for Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文针对视觉-语言-动作模型在机器人操作中依赖专家示范、奖励稀疏的问题，提出自参照策略优化（SRPO）。通过利用模型自身生成的成功轨迹作为参考，结合世界模型的潜在表示，为失败轨迹赋予渐进式奖励，实现高效无监督强化学习。在LIBERO基准上，200步内达99.2%成功率，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2511.15605v2](https://arxiv.org/pdf/2511.15605v2)**

> **作者:** Senyu Fei; Siyin Wang; Li Ji; Ao Li; Shiduo Zhang; Liming Liu; Jinlong Hou; Jingjing Gong; Xianzhong Zhao; Xipeng Qiu
>
> **摘要:** Vision-Language-Action (VLA) models excel in robotic manipulation but are constrained by their heavy reliance on expert demonstrations, leading to demonstration bias and limiting performance. Reinforcement learning (RL) is a vital post-training strategy to overcome these limits, yet current VLA-RL methods, including group-based optimization approaches, are crippled by severe reward sparsity. Relying on binary success indicators wastes valuable information in failed trajectories, resulting in low training efficiency. To solve this, we propose Self-Referential Policy Optimization (SRPO), a novel VLA-RL framework. SRPO eliminates the need for external demonstrations or manual reward engineering by leveraging the model's own successful trajectories, generated within the current training batch, as a self-reference. This allows us to assign a progress-wise reward to failed attempts. A core innovation is the use of latent world representations to measure behavioral progress robustly. Instead of relying on raw pixels or requiring domain-specific fine-tuning, we utilize the compressed, transferable encodings from a world model's latent space. These representations naturally capture progress patterns across environments, enabling accurate, generalized trajectory comparison. Empirical evaluations on the LIBERO benchmark demonstrate SRPO's efficiency and effectiveness. Starting from a supervised baseline with 48.9% success, SRPO achieves a new state-of-the-art success rate of 99.2% in just 200 RL steps, representing a 103% relative improvement without any extra supervision. Furthermore, SRPO shows substantial robustness, achieving a 167% performance improvement on the LIBERO-Plus benchmark.
>
---
#### [replaced 004] Securing the Skies: A Comprehensive Survey on Anti-UAV Methods, Benchmarking, and Future Directions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文聚焦反无人机（anti-UAV）任务，针对无人机带来的安全威胁，系统综述了检测、分类与跟踪技术。研究涵盖多模态传感器与前沿方法，评估主流方案并指出实时性、隐蔽探测及集群应对等挑战，提出未来研究方向以推动智能防御体系发展。**

- **链接: [https://arxiv.org/pdf/2504.11967v3](https://arxiv.org/pdf/2504.11967v3)**

> **作者:** Yifei Dong; Fengyi Wu; Sanjian Zhang; Guangyu Chen; Yuzhi Hu; Masumi Yano; Jingdong Sun; Siyu Huang; Feng Liu; Qi Dai; Zhi-Qi Cheng
>
> **备注:** Best Paper, Accepted at CVPR Workshop Anti-UAV 2025. 16 pages
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are indispensable for infrastructure inspection, surveillance, and related tasks, yet they also introduce critical security challenges. This survey provides a wide-ranging examination of the anti-UAV domain, centering on three core objectives-classification, detection, and tracking-while detailing emerging methodologies such as diffusion-based data synthesis, multi-modal fusion, vision-language modeling, self-supervised learning, and reinforcement learning. We systematically evaluate state-of-the-art solutions across both single-modality and multi-sensor pipelines (spanning RGB, infrared, audio, radar, and RF) and discuss large-scale as well as adversarially oriented benchmarks. Our analysis reveals persistent gaps in real-time performance, stealth detection, and swarm-based scenarios, underscoring pressing needs for robust, adaptive anti-UAV systems. By highlighting open research directions, we aim to foster innovation and guide the development of next-generation defense strategies in an era marked by the extensive use of UAVs.
>
---
#### [replaced 005] PRIMT: Preference-based Reinforcement Learning with Multimodal Feedback and Trajectory Synthesis from Foundation Models
- **分类: cs.RO**

- **简介: 该论文提出PRIMT框架，解决偏好强化学习中依赖大量人工标注、反馈模糊与信用分配困难的问题。通过融合多模态大模型生成合成反馈与轨迹，结合前视采样与后视增强策略，提升训练效率与性能，在移动与操作任务上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2509.15607v2](https://arxiv.org/pdf/2509.15607v2)**

> **作者:** Ruiqi Wang; Dezhong Zhao; Ziqin Yuan; Tianyu Shao; Guohua Chen; Dominic Kao; Sungeun Hong; Byung-Cheol Min
>
> **摘要:** Preference-based reinforcement learning (PbRL) has emerged as a promising paradigm for teaching robots complex behaviors without reward engineering. However, its effectiveness is often limited by two critical challenges: the reliance on extensive human input and the inherent difficulties in resolving query ambiguity and credit assignment during reward learning. In this paper, we introduce PRIMT, a PbRL framework designed to overcome these challenges by leveraging foundation models (FMs) for multimodal synthetic feedback and trajectory synthesis. Unlike prior approaches that rely on single-modality FM evaluations, PRIMT employs a hierarchical neuro-symbolic fusion strategy, integrating the complementary strengths of large language models and vision-language models in evaluating robot behaviors for more reliable and comprehensive feedback. PRIMT also incorporates foresight trajectory generation, which reduces early-stage query ambiguity by warm-starting the trajectory buffer with bootstrapped samples, and hindsight trajectory augmentation, which enables counterfactual reasoning with a causal auxiliary loss to improve credit assignment. We evaluate PRIMT on 2 locomotion and 6 manipulation tasks on various benchmarks, demonstrating superior performance over FM-based and scripted baselines.
>
---
#### [replaced 006] Gentle Object Retraction in Dense Clutter Using Multimodal Force Sensing and Imitation Learning
- **分类: cs.RO**

- **简介: 该论文研究机器人在密集杂乱环境中安全抓取物体的任务。针对传统方法难以避免过度用力的问题，提出结合多模态力觉传感（触觉与力矩估计）与模仿学习的策略。通过在随机场景下训练并评估，证明引入力觉信息可显著提升成功率、减少失败和加快执行速度，最优方案较无力觉基线提升80%。**

- **链接: [https://arxiv.org/pdf/2508.19476v2](https://arxiv.org/pdf/2508.19476v2)**

> **作者:** Dane Brouwer; Joshua Citron; Heather Nolte; Jeannette Bohg; Mark Cutkosky
>
> **备注:** Accepted in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Dense collections of movable objects are common in everyday spaces-from cabinets in a home to shelves in a warehouse. Safely retracting objects from such collections is difficult for robots, yet people do it frequently, leveraging learned experience in tandem with vision and non-prehensile tactile sensing on the sides and backs of their hands and arms. We investigate the role of contact force sensing for training robots to gently reach into constrained clutter and extract objects. The available sensing modalities are (1) "eye-in-hand" vision, (2) proprioception, (3) non-prehensile triaxial tactile sensing, (4) contact wrenches estimated from joint torques, and (5) a measure of object acquisition obtained by monitoring the vacuum line of a suction cup. We use imitation learning to train policies from a set of demonstrations on randomly generated scenes, then conduct an ablation study of wrench and tactile information. We evaluate each policy's performance across 40 unseen environment configurations. Policies employing any force sensing show fewer excessive force failures, an increased overall success rate, and faster completion times. The best performance is achieved using both tactile and wrench information, producing an 80% improvement above the baseline without force information.
>
---
#### [replaced 007] How to Adapt Control Barrier Functions? A Learning-Based Approach with Applications to a VTOL Quadplane
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对带输入约束的系统安全控制问题，提出一种基于学习的在线自适应控制屏障函数（CBF）方法。通过引入局部验证的CBF参数，结合不确定性感知的验证机制，实现对神经网络预测中认知与随机不确定性的建模，在垂直起降四旋翼飞机的复杂飞行任务中保障有限时域安全，提升系统性能与安全性。**

- **链接: [https://arxiv.org/pdf/2504.03038v5](https://arxiv.org/pdf/2504.03038v5)**

> **作者:** Taekyung Kim; Randal W. Beard; Dimitra Panagou
>
> **备注:** 2025 IEEE Conference on Decision and Control (CDC). Project page: https://www.taekyung.me/how-to-adapt-cbf
>
> **摘要:** In this paper, we present a novel theoretical framework for online adaptation of Control Barrier Function (CBF) parameters, i.e., of the class K functions included in the CBF condition, under input constraints. We introduce the concept of locally validated CBF parameters, which are adapted online to guarantee finite-horizon safety, based on conditions derived from Nagumo's theorem and tangent cone analysis. To identify these parameters online, we integrate a learning-based approach with an uncertainty-aware verification process that account for both epistemic and aleatoric uncertainties inherent in neural network predictions. Our method is demonstrated on a VTOL quadplane model during challenging transition and landing maneuvers, showcasing enhanced performance while maintaining safety.
>
---
#### [replaced 008] Have We Scene It All? Scene Graph-Aware Deep Point Cloud Compression
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D点云在多机器人系统中传输效率低的问题，提出基于语义场景图的深度压缩框架。通过语义分割点云并利用FiLM增强编码器，实现高效压缩与高保真重建，支持下游任务，显著提升压缩率与系统性能。**

- **链接: [https://arxiv.org/pdf/2510.08512v2](https://arxiv.org/pdf/2510.08512v2)**

> **作者:** Nikolaos Stathoulopoulos; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Please cite published version. 8 pages, 6 figures
>
> **摘要:** Efficient transmission of 3D point cloud data is critical for advanced perception in centralized and decentralized multi-agent robotic systems, especially nowadays with the growing reliance on edge and cloud-based processing. However, the large and complex nature of point clouds creates challenges under bandwidth constraints and intermittent connectivity, often degrading system performance. We propose a deep compression framework based on semantic scene graphs. The method decomposes point clouds into semantically coherent patches and encodes them into compact latent representations with semantic-aware encoders conditioned by Feature-wise Linear Modulation (FiLM). A folding-based decoder, guided by latent features and graph node attributes, enables structurally accurate reconstruction. Experiments on the SemanticKITTI and nuScenes datasets show that the framework achieves state-of-the-art compression rates, reducing data size by up to 98% while preserving both structural and semantic fidelity. In addition, it supports downstream applications such as multi-robot pose graph optimization and map merging, achieving trajectory accuracy and map alignment comparable to those obtained with raw LiDAR scans.
>
---
#### [replaced 009] A Neuro-inspired Theory of Joint Human-Swarm Interaction
- **分类: cs.HC; cs.MA; cs.NE; cs.RO**

- **简介: 该论文提出一种神经启发的协同系统理论，用于解决人-蜂群交互（HSI）中的动态适应性、鲁棒性与可扩展性问题。基于认知系统工程视角，构建了支持智能协同的人- swarm 交互框架，为设计高效人-蜂群控制回路提供理论指导。**

- **链接: [https://arxiv.org/pdf/2007.04882v2](https://arxiv.org/pdf/2007.04882v2)**

> **作者:** Jonas D. Hasbach; Maren Bennewitz
>
> **备注:** ICRA Workshop on Human-Swarm Interaction 2020
>
> **摘要:** Human-swarm interaction (HSI) is an active research challenge in the realms of swarm robotics and human-factors engineering. Here we apply a cognitive systems engineering perspective and introduce a neuro-inspired joint systems theory of HSI. The mindset defines predictions for adaptive, robust and scalable HSI dynamics and therefore has the potential to inform human-swarm loop design.
>
---
#### [replaced 010] 3EED: Ground Everything Everywhere in 3D
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出3EED，一个大规模多平台3D视觉语言定位基准，涵盖车、无人机、四足机器人采集的RGB与LiDAR数据。针对现有数据集规模小、场景单一、平台受限的问题，构建超12.8万物体、2.2万标注表达的数据集，设计高效标注流程与跨平台对齐方法，推动开放世界中可泛化的3D语言感知研究。**

- **链接: [https://arxiv.org/pdf/2511.01755v2](https://arxiv.org/pdf/2511.01755v2)**

> **作者:** Rong Li; Yuhao Dong; Tianshuai Hu; Ao Liang; Youquan Liu; Dongyue Lu; Liang Pan; Lingdong Kong; Junwei Liang; Ziwei Liu
>
> **备注:** NeurIPS 2025 DB Track; 38 pages, 17 figures, 10 tables; Project Page at https://project-3eed.github.io/
>
> **摘要:** Visual grounding in 3D is the key for embodied agents to localize language-referred objects in open-world environments. However, existing benchmarks are limited to indoor focus, single-platform constraints, and small scale. We introduce 3EED, a multi-platform, multi-modal 3D grounding benchmark featuring RGB and LiDAR data from vehicle, drone, and quadruped platforms. We provide over 128,000 objects and 22,000 validated referring expressions across diverse outdoor scenes -- 10x larger than existing datasets. We develop a scalable annotation pipeline combining vision-language model prompting with human verification to ensure high-quality spatial grounding. To support cross-platform learning, we propose platform-aware normalization and cross-modal alignment techniques, and establish benchmark protocols for in-domain and cross-platform evaluations. Our findings reveal significant performance gaps, highlighting the challenges and opportunities of generalizable 3D grounding. The 3EED dataset and benchmark toolkit are released to advance future research in language-driven 3D embodied perception.
>
---
#### [replaced 011] Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对接触丰富的现实机器人操作中数据聚合（DAgger）的挑战，提出Compliant Residual DAgger（CR-DAgger）。通过柔顺干预接口实现无中断的人类精准修正，并采用融合力反馈的残差策略学习，仅用少量纠正数据即显著提升基础策略成功率，优于重新训练与微调方法。**

- **链接: [https://arxiv.org/pdf/2506.16685v3](https://arxiv.org/pdf/2506.16685v3)**

> **作者:** Xiaomeng Xu; Yifan Hou; Zeyi Liu; Shuran Song
>
> **摘要:** We address key challenges in Dataset Aggregation (DAgger) for real-world contact-rich manipulation: how to collect informative human correction data and how to effectively update policies with this new data. We introduce Compliant Residual DAgger (CR-DAgger), which contains two novel components: 1) a Compliant Intervention Interface that leverages compliance control, allowing humans to provide gentle, accurate delta action corrections without interrupting the ongoing robot policy execution; and 2) a Compliant Residual Policy formulation that learns from human corrections while incorporating force feedback and force control. Our system significantly enhances performance on precise contact-rich manipulation tasks using minimal correction data, improving base policy success rates by over 50\% on two challenging tasks (book flipping and belt assembly) while outperforming both retraining-from-scratch and finetuning approaches. Through extensive real-world experiments, we provide practical guidance for implementing effective DAgger in real-world robot learning tasks. Result videos are available at: https://compliant-residual-dagger.github.io/
>
---
#### [replaced 012] High-Speed Event Vision-Based Tactile Roller Sensor for Large Surface Measurements
- **分类: cs.RO**

- **简介: 该论文针对大表面高精度3D检测中传统触觉视觉传感器速度慢、易受运动模糊影响的问题，提出一种基于事件视觉的滚动式触觉传感器。通过融合类脑相机与多视角立体重建，实现0.5m/s高速连续扫描，精度达100μm以内，显著提升测量效率与准确性。**

- **链接: [https://arxiv.org/pdf/2507.19914v2](https://arxiv.org/pdf/2507.19914v2)**

> **作者:** Akram Khairi; Hussain Sajwani; Abdallah Mohammad Alkilany; Laith AbuAssi; Mohamad Halwani; Islam Mohamed Zaid; Ahmed Awadalla; Dewald Swart; Abdulla Ayyad; Yahya Zweiri
>
> **备注:** Under Review - Project Page: https://akramekhairi.github.io/TheySeeMeRolling/. 14 pages, 11 figures
>
> **摘要:** Inspecting large-scale industrial surfaces like aircraft fuselages for quality control requires capturing their precise 3D surface geometry at high resolution. Vision-based tactile sensors (VBTSs) offer high local resolution but require slow 'press-and-lift' measurements stitched for large areas. Approaches with sliding or roller/belt VBTS designs provide measurements continuity. However, they face significant challenges respectively: sliding struggles with friction/wear and both approaches are speed-limited by conventional camera frame rates and motion blur, making large-area scanning time consuming. Thus, a rapid, continuous, high-resolution method is needed. We introduce a novel tactile sensor integrating a neuromorphic camera in a rolling mechanism to achieve this. Leveraging its high temporal resolution and robustness to motion blur, our system uses a modified event-based multi-view stereo approach for 3D reconstruction. We demonstrate state-of-the-art scanning speeds up to 0.5 m/s, achieving Mean Absolute Error below 100 microns -- 11 times faster than prior continuous tactile sensing methods. A multi-reference Bayesian fusion strategy enhances accuracy (reducing MAE by 25.2\% compared to EMVS) and mitigates curvature errors. We also validate high-speed feature recognition via Braille reading 2.6 times faster than previous approaches.
>
---
#### [replaced 013] Adversarial Exploitation of Data Diversity Improves Visual Localization
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对视觉定位任务中绝对姿态回归（APR）方法泛化能力差的问题，提出利用外观多样性提升鲁棒性。通过将2D图像转为带外观与去模糊特性的3D高斯点云，合成多样化训练数据，并设计双分支对抗训练框架，有效缩小了仿真到真实场景的差距，显著降低定位误差，尤其在复杂动态和光照变化场景下表现优异。**

- **链接: [https://arxiv.org/pdf/2412.00138v2](https://arxiv.org/pdf/2412.00138v2)**

> **作者:** Sihang Li; Siqi Tan; Bowen Chang; Jing Zhang; Chen Feng; Yiming Li
>
> **备注:** 24 pages, 22 figures
>
> **摘要:** Visual localization, which estimates a camera's pose within a known scene, is a fundamental capability for autonomous systems. While absolute pose regression (APR) methods have shown promise for efficient inference, they often struggle with generalization. Recent approaches attempt to address this through data augmentation with varied viewpoints, yet they overlook a critical factor: appearance diversity. In this work, we identify appearance variation as the key to robust localization. Specifically, we first lift real 2D images into 3D Gaussian Splats with varying appearance and deblurring ability, enabling the synthesis of diverse training data that varies not just in poses but also in environmental conditions such as lighting and weather. To fully unleash the potential of the appearance-diverse data, we build a two-branch joint training pipeline with an adversarial discriminator to bridge the syn-to-real gap. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art methods, reducing translation and rotation errors by 50\% and 41\% on indoor datasets, and 38\% and 44\% on outdoor datasets. Most notably, our method shows remarkable robustness in dynamic driving scenarios under varying weather conditions and in day-to-night scenarios, where previous APR methods fail. Project Page: https://ai4ce.github.io/RAP/
>
---
#### [replaced 014] APULSE: A Scalable Hybrid Algorithm for the RCSPP on Large-Scale Dense Graphs
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对大尺度稠密图上的资源约束最短路径问题（RCSPP），提出APULSE混合算法。通过结合A*启发式搜索、脉冲式剪枝和时间分桶策略，显著提升求解效率与可扩展性，有效解决传统方法在复杂场景下计算慢、难以收敛的问题，适用于无人车等实时规划需求。**

- **链接: [https://arxiv.org/pdf/2511.18236v2](https://arxiv.org/pdf/2511.18236v2)**

> **作者:** Nuno Soares; António Grilo
>
> **备注:** This version corrects keywords and reference [9]. 9 pages
>
> **摘要:** The resource-constrained shortest path problem (RCSPP) is a fundamental NP-hard optimization challenge with broad applications, from network routing to autonomous navigation. This problem involves finding a path that minimizes a primary cost subject to a budget on a secondary resource. While various RCSPP solvers exist, they often face critical scalability limitations when applied to the large, dense graphs characteristic of complex, real-world scenarios, making them impractical for time-critical planning. This challenge is particularly acute in domains like mission planning for unmanned ground vehicles (UGVs), which demand solutions on large-scale terrain graphs. This paper introduces APULSE, a hybrid label-setting algorithm designed to efficiently solve the RCSPP on such challenging graphs. APULSE integrates a best-first search guided by an A* heuristic with aggressive, Pulse-style pruning mechanisms and a time-bucketing strategy for effective state-space reduction. A computational study, using a large-scale UGV planning scenario, benchmarks APULSE against state-of-the-art algorithms. The results demonstrate that APULSE consistently finds near-optimal solutions while being orders of magnitude faster and more robust, particularly on large problem instances where competing methods fail. This superior scalability establishes APULSE as an effective solution for RCSPP in complex, large-scale environments, enabling capabilities such as interactive decision support and dynamic replanning.
>
---
#### [replaced 015] RobustVLA: Robustness-Aware Reinforcement Post-Training for Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在分布外场景下泛化能力差的问题，提出RobustVLA方法。通过引入雅可比正则化和光滑性正则化，增强模型对观测噪声和动作扰动的鲁棒性，实现轻量级在线强化学习后训练，显著提升VLA模型的可靠性与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.01331v2](https://arxiv.org/pdf/2511.01331v2)**

> **作者:** Hongyin Zhang; Shuo Zhang; Junxi Jin; Qixin Zeng; Runze Li; Donglin Wang
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as powerful general-purpose policies for robotic manipulation, benefiting from large-scale multi-modal pre-training. However, they often fail to generalize reliably in out-of-distribution deployments, where unavoidable disturbances such as observation noise, sensor errors, or actuation perturbations become prevalent. While recent Reinforcement Learning (RL)-based post-training provides a practical means to adapt pre-trained VLA models, existing methods mainly emphasize reward maximization and overlook robustness to environmental uncertainty. In this work, we introduce RobustVLA, a lightweight online RL post-training method designed to explicitly enhance the resilience of VLA models. Through a systematic robustness analysis, we identify two key regularizations: Jacobian regularization, which mitigates sensitivity to observation noise, and smoothness regularization, which stabilizes policies under action perturbations. Extensive experiments across diverse robotic environments demonstrate that RobustVLA significantly outperforms prior state-of-the-art methods in robustness and reliability. Our results highlight the importance of principled robustness-aware RL post-training as a key step toward improving the reliability and robustness of VLA models.
>
---
#### [replaced 016] Coordinating Spinal and Limb Dynamics for Enhanced Sprawling Robot Mobility
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究如何提升仿蜥蜴四足机器人在复杂地形下的移动能力。针对传统步态设计适应性差、端到端强化学习数据需求高且不稳定的问题，提出结合生物启发步态与深度强化学习的混合控制框架，利用脊柱主动关节增强爬行稳定性与鲁棒性，有效应对地形不确定性。**

- **链接: [https://arxiv.org/pdf/2504.14103v2](https://arxiv.org/pdf/2504.14103v2)**

> **作者:** Merve Atasever; Ali Okhovat; Azhang Nazaripouya; John Nisbet; Omer Kurkutlu; Jyotirmoy V. Deshmukh; Yasemin Ozkan Aydin
>
> **备注:** Initial version of the work has been accepted for presentation at the Mechanical Intelligence in Robotics workshop at ICRA 2025
>
> **摘要:** Sprawling locomotion in vertebrates, particularly salamanders, demonstrates how body undulation and spinal mobility enhance stability, maneuverability, and adaptability across complex terrains. While prior work has separately explored biologically inspired gait design or deep reinforcement learning (DRL), these approaches face inherent limitations: open-loop gait designs often lack adaptability to unforeseen terrain variations, whereas end-to-end DRL methods are data-hungry and prone to unstable behaviors when transferring from simulation to real robots. We propose a hybrid control framework that integrates Hildebrand's biologically grounded gait design with DRL, enabling a salamander-inspired quadruped robot to exploit active spinal joints for robust crawling motion. Our evaluation across multiple robot configurations in target-directed navigation tasks reveals that this hybrid approach systematically improves robustness under environmental uncertainties such as surface irregularities. By bridging structured gait design with learning-based methodology, our work highlights the promise of interdisciplinary control strategies for developing efficient, resilient, and biologically informed spinal actuation in robotic systems.
>
---
#### [replaced 017] AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对自动驾驶中视觉-语言-动作模型的决策可解释性与动作合理性问题，提出AutoDrive-R²框架。通过构建含自省的思维链数据集和基于物理约束的强化学习策略，增强模型推理与自我反思能力，提升轨迹规划的逻辑性与真实性，在nuScenes和Waymo数据集上实现先进性能。**

- **链接: [https://arxiv.org/pdf/2509.01944v2](https://arxiv.org/pdf/2509.01944v2)**

> **作者:** Zhenlong Yuan; Chengxuan Qian; Jing Tang; Rui Chen; Zijian Song; Lei Sun; Xiangxiang Chu; Yujun Cai; Dapeng Zhang; Shuo Li
>
> **摘要:** Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.
>
---
#### [replaced 018] Gemini Robotics 1.5: Pushing the Frontier of Generalist Robots with Advanced Embodied Reasoning, Thinking, and Motion Transfer
- **分类: cs.RO**

- **简介: 该论文提出Gemini Robotics 1.5及ER 1.5模型，面向通用机器人任务，解决物理世界理解与复杂任务执行难题。通过多形态数据融合、动作与思维交替的推理机制及先进具身推理能力，实现感知-思考-行动一体化，显著提升机器人在多步任务中的泛化性与可解释性。**

- **链接: [https://arxiv.org/pdf/2510.03342v3](https://arxiv.org/pdf/2510.03342v3)**

> **作者:** Gemini Robotics Team; Abbas Abdolmaleki; Saminda Abeyruwan; Joshua Ainslie; Jean-Baptiste Alayrac; Montserrat Gonzalez Arenas; Ashwin Balakrishna; Nathan Batchelor; Alex Bewley; Jeff Bingham; Michael Bloesch; Konstantinos Bousmalis; Philemon Brakel; Anthony Brohan; Thomas Buschmann; Arunkumar Byravan; Serkan Cabi; Ken Caluwaerts; Federico Casarini; Christine Chan; Oscar Chang; London Chappellet-Volpini; Jose Enrique Chen; Xi Chen; Hao-Tien Lewis Chiang; Krzysztof Choromanski; Adrian Collister; David B. D'Ambrosio; Sudeep Dasari; Todor Davchev; Meet Kirankumar Dave; Coline Devin; Norman Di Palo; Tianli Ding; Carl Doersch; Adil Dostmohamed; Yilun Du; Debidatta Dwibedi; Sathish Thoppay Egambaram; Michael Elabd; Tom Erez; Xiaolin Fang; Claudio Fantacci; Cody Fong; Erik Frey; Chuyuan Fu; Ruiqi Gao; Marissa Giustina; Keerthana Gopalakrishnan; Laura Graesser; Oliver Groth; Agrim Gupta; Roland Hafner; Steven Hansen; Leonard Hasenclever; Sam Haves; Nicolas Heess; Brandon Hernaez; Alex Hofer; Jasmine Hsu; Lu Huang; Sandy H. Huang; Atil Iscen; Mithun George Jacob; Deepali Jain; Sally Jesmonth; Abhishek Jindal; Ryan Julian; Dmitry Kalashnikov; M. Emre Karagozler; Stefani Karp; Matija Kecman; J. Chase Kew; Donnie Kim; Frank Kim; Junkyung Kim; Thomas Kipf; Sean Kirmani; Ksenia Konyushkova; Li Yang Ku; Yuheng Kuang; Thomas Lampe; Antoine Laurens; Tuan Anh Le; Isabel Leal; Alex X. Lee; Tsang-Wei Edward Lee; Guy Lever; Jacky Liang; Li-Heng Lin; Fangchen Liu; Shangbang Long; Caden Lu; Sharath Maddineni; Anirudha Majumdar; Kevis-Kokitsi Maninis; Andrew Marmon; Sergio Martinez; Assaf Hurwitz Michaely; Niko Milonopoulos; Joss Moore; Robert Moreno; Michael Neunert; Francesco Nori; Joy Ortiz; Kenneth Oslund; Carolina Parada; Emilio Parisotto; Amaris Paryag; Acorn Pooley; Thomas Power; Alessio Quaglino; Haroon Qureshi; Rajkumar Vasudeva Raju; Helen Ran; Dushyant Rao; Kanishka Rao; Isaac Reid; David Rendleman; Krista Reymann; Miguel Rivas; Francesco Romano; Yulia Rubanova; Peter Pastor Sampedro; Pannag R Sanketi; Dhruv Shah; Mohit Sharma; Kathryn Shea; Mohit Shridhar; Charles Shu; Vikas Sindhwani; Sumeet Singh; Radu Soricut; Rachel Sterneck; Ian Storz; Razvan Surdulescu; Jie Tan; Jonathan Tompson; Saran Tunyasuvunakool; Jake Varley; Grace Vesom; Giulia Vezzani; Maria Bauza Villalonga; Oriol Vinyals; René Wagner; Ayzaan Wahid; Stefan Welker; Paul Wohlhart; Chengda Wu; Markus Wulfmeier; Fei Xia; Ted Xiao; Annie Xie; Jinyu Xie; Peng Xu; Sichun Xu; Ying Xu; Zhuo Xu; Jimmy Yan; Sherry Yang; Skye Yang; Yuxiang Yang; Hiu Hong Yu; Wenhao Yu; Wentao Yuan; Yuan Yuan; Jingwei Zhang; Tingnan Zhang; Zhiyuan Zhang; Allan Zhou; Guangyao Zhou; Yuxiang Zhou
>
> **摘要:** General-purpose robots need a deep understanding of the physical world, advanced reasoning, and general and dexterous control. This report introduces the latest generation of the Gemini Robotics model family: Gemini Robotics 1.5, a multi-embodiment Vision-Language-Action (VLA) model, and Gemini Robotics-ER 1.5, a state-of-the-art Embodied Reasoning (ER) model. We are bringing together three major innovations. First, Gemini Robotics 1.5 features a novel architecture and a Motion Transfer (MT) mechanism, which enables it to learn from heterogeneous, multi-embodiment robot data and makes the VLA more general. Second, Gemini Robotics 1.5 interleaves actions with a multi-level internal reasoning process in natural language. This enables the robot to "think before acting" and notably improves its ability to decompose and execute complex, multi-step tasks, and also makes the robot's behavior more interpretable to the user. Third, Gemini Robotics-ER 1.5 establishes a new state-of-the-art for embodied reasoning, i.e., for reasoning capabilities that are critical for robots, such as visual and spatial understanding, task planning, and progress estimation. Together, this family of models takes us a step towards an era of physical agents-enabling robots to perceive, think and then act so they can solve complex multi-step tasks.
>
---
#### [replaced 019] Differentiable Contact Dynamics for Stable Object Placement Under Geometric Uncertainties
- **分类: cs.RO**

- **简介: 该论文针对机器人在几何不确定性下的稳定物体放置任务，提出基于可微接触动力学的算法。通过梯度优化估计几何不确定性，结合信念融合策略提升鲁棒性，在Franka机械臂上实现对多种不确定场景的有效处理。**

- **链接: [https://arxiv.org/pdf/2409.17725v2](https://arxiv.org/pdf/2409.17725v2)**

> **作者:** Linfeng Li; Gang Yang; Lin Shao; David Hsu
>
> **摘要:** From serving a cup of coffee to positioning mechanical parts during assembly, stable object placement is a crucial skill for future robots. It becomes particularly challenging under geometric uncertainties, e.g., when the object pose or shape is not known accurately. This work leverages a differentiable simulation model of contact dynamics to tackle this challenge. We derive a novel gradient that relates force-torque sensor readings to geometric uncertainties, thus enabling uncertainty estimation by minimizing discrepancies between sensor data and model predictions via gradient descent. Gradient-based methods are sensitive to initialization. To mitigate this effect, we maintain a belief over multiple estimates and choose the robot action based on the current belief at each timestep. In experiments on a Franka robot arm, our method achieved promising results on multiple objects under various geometric uncertainties, including the in-hand pose uncertainty of a grasped object, the object shape uncertainty, and the environment uncertainty.
>
---
#### [replaced 020] VITA: Vision-to-Action Flow Matching Policy
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VITA，一种视觉到动作的无噪声、无条件流匹配策略框架。针对传统方法需反复引入视觉信息导致效率低的问题，VITA直接从视觉表征映射到动作潜空间，通过动作自编码器对齐维度并防止潜空间坍塌，实现更快推理与更优性能，在仿真与真实机器人任务中均表现优异。**

- **链接: [https://arxiv.org/pdf/2507.13231v3](https://arxiv.org/pdf/2507.13231v3)**

> **作者:** Dechen Gao; Boqi Zhao; Andrew Lee; Ian Chuang; Hanchu Zhou; Hang Wang; Zhe Zhao; Junshan Zhang; Iman Soltani
>
> **备注:** Project page: https://ucd-dare.github.io/VITA/ Code: https://github.com/ucd-dare/VITA
>
> **摘要:** Conventional flow matching and diffusion-based policies sample through iterative denoising from standard noise distributions (e.g., Gaussian), and require conditioning modules to repeatedly incorporate visual information during the generative process, incurring substantial time and memory overhead. To reduce the complexity, we develop VITA(VIsion-To-Action policy), a noise-free and conditioning-free flow matching policy learning framework that directly flows from visual representations to latent actions. Since the source of the flow is visually grounded, VITA eliminates the need of visual conditioning during generation. As expected, bridging vision and action is challenging, because actions are lower-dimensional, less structured, and sparser than visual representations; moreover, flow matching requires the source and target to have the same dimensionality. To overcome this, we introduce an action autoencoder that maps raw actions into a structured latent space aligned with visual latents, trained jointly with flow matching. To further prevent latent space collapse, we propose flow latent decoding, which anchors the latent generation process by backpropagating the action reconstruction loss through the flow matching ODE (ordinary differential equation) solving steps. We evaluate VITA on 9 simulation and 5 real-world tasks from ALOHA and Robomimic. VITA achieves 1.5x-2x faster inference compared to conventional methods with conditioning modules, while outperforming or matching state-of-the-art policies. Codes, datasets, and demos are available at our project page: https://ucd-dare.github.io/VITA/.
>
---
#### [replaced 021] Automaton Constrained Q-Learning
- **分类: cs.RO**

- **简介: 该论文针对机器人在复杂连续环境中同时满足时序目标与动态安全约束的难题，提出Automaton Constrained Q-Learning（ACQL）算法。通过结合LTL任务规范的自动机表示与目标条件值学习，显式建模阶段目标进展与静态/非静态安全约束，实现高效、可扩展的强化学习。**

- **链接: [https://arxiv.org/pdf/2510.05061v2](https://arxiv.org/pdf/2510.05061v2)**

> **作者:** Anastasios Manganaris; Vittorio Giammarino; Ahmed H. Qureshi
>
> **备注:** 10 main content pages, 4 main content figures, 11 appendix pages, 5 appendix figures, camera ready version submitted to 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Real-world robotic tasks often require agents to achieve sequences of goals while respecting time-varying safety constraints. However, standard Reinforcement Learning (RL) paradigms are fundamentally limited in these settings. A natural approach to these problems is to combine RL with Linear-time Temporal Logic (LTL), a formal language for specifying complex, temporally extended tasks and safety constraints. Yet, existing RL methods for LTL objectives exhibit poor empirical performance in complex and continuous environments. As a result, no scalable methods support both temporally ordered goals and safety simultaneously, making them ill-suited for realistic robotics scenarios. We propose Automaton Constrained Q-Learning (ACQL), an algorithm that addresses this gap by combining goal-conditioned value learning with automaton-guided reinforcement. ACQL supports most LTL task specifications and leverages their automaton representation to explicitly encode stage-wise goal progression and both stationary and non-stationary safety constraints. We show that ACQL outperforms existing methods across a range of continuous control tasks, including cases where prior methods fail to satisfy either goal-reaching or safety constraints. We further validate its real-world applicability by deploying ACQL on a 6-DOF robotic arm performing a goal-reaching task in a cluttered, cabinet-like space with safety constraints. Our results demonstrate that ACQL is a robust and scalable solution for learning robotic behaviors according to rich temporal specifications.
>
---
#### [replaced 022] A Unified Framework for Probabilistic Dynamic-, Trajectory- and Vision-based Virtual Fixtures
- **分类: cs.RO**

- **简介: 该论文提出一种统一的概率化虚拟力场框架，解决人机协作中动态、轨迹与视觉引导的自适应问题。通过融合手动、半自动与全自主模式，实现任务阶段的无缝切换，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2506.10239v2](https://arxiv.org/pdf/2506.10239v2)**

> **作者:** Maximilian Mühlbauer; Bernhard Weber; Sylvain Calinon; Freek Stulp; Alin Albu-Schäffer; João Silvério
>
> **备注:** for the supplementary video, see https://www.youtube.com/watch?v=vUXzcpMbMnY
>
> **摘要:** Probabilistic Virtual Fixtures (VFs) enable the adaptive selection of the most suitable haptic feedback for each phase of a task, based on learned or perceived uncertainty. While keeping the human in the loop remains essential, for instance, to ensure high precision, partial automation of certain task phases is critical for productivity. We present a unified framework for probabilistic VFs that seamlessly switches between manual fixtures, semi-automated fixtures (with the human handling precise tasks), and full autonomy. We introduce a novel probabilistic Dynamical System-based VF for coarse guidance, enabling the robot to autonomously complete certain task phases while keeping the human operator in the loop. For tasks requiring precise guidance, we extend probabilistic position-based trajectory fixtures with automation allowing for seamless human interaction as well as geometry-awareness and optimal impedance gains. For manual tasks requiring very precise guidance, we also extend visual servoing fixtures with the same geometry-awareness and impedance behavior. We validate our approach experimentally on different robots, showcasing multiple operation modes and the ease of programming fixtures.
>
---
#### [replaced 023] A $1000\times$ Faster LLM-enhanced Algorithm For Path Planning in Large-scale Grid Maps
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对大规模网格地图路径规划问题，解决传统算法及LLM方法在效率与性能上的瓶颈。提出iLLM-A*算法，通过优化A*、增量学习生成优质路点及智能选点，实现超1000倍加速、显著降存并优化路径质量。**

- **链接: [https://arxiv.org/pdf/2510.02716v2](https://arxiv.org/pdf/2510.02716v2)**

> **作者:** Junlin Zeng; Xin Zhang; Xiang Zhao; Yan Pan
>
> **摘要:** Path planning in grid maps, arising from various applications, has garnered significant attention. Existing methods, such as A*, Dijkstra, and their variants, work well for small-scale maps but fail to address large-scale ones due to high search time and memory consumption. Recently, Large Language Models (LLMs) have shown remarkable performance in path planning but still suffer from spatial illusion and poor planning performance. Among all the works, LLM-A* \cite{meng2024llm} leverages LLM to generate a series of waypoints and then uses A* to plan the paths between the neighboring waypoints. In this way, the complete path is constructed. However, LLM-A* still suffers from high computational time for large-scale maps. To fill this gap, we conducted a deep investigation into LLM-A* and found its bottleneck, resulting in limited performance. Accordingly, we design an innovative LLM-enhanced algorithm, abbr. as iLLM-A*. iLLM-A* includes 3 carefully designed mechanisms, including the optimization of A*, an incremental learning method for LLM to generate high-quality waypoints, and the selection of the appropriate waypoints for A* for path planning. Finally, a comprehensive evaluation on various grid maps shows that, compared with LLM-A*, iLLM-A* \textbf{1) achieves more than $1000\times$ speedup on average, and up to $2349.5\times$ speedup in the extreme case, 2) saves up to $58.6\%$ of the memory cost, 3) achieves both obviously shorter path length and lower path length standard deviation.}
>
---
#### [replaced 024] SAD-Flower: Flow Matching for Safe, Admissible, and Dynamically Consistent Planning
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文针对数据驱动规划中轨迹安全、可行与动态一致性的缺失问题，提出SAD-Flower框架。通过引入虚拟控制输入，结合非线性控制理论，实现无需重训练的约束保障与动态一致性，显著提升生成轨迹的安全性与可执行性。**

- **链接: [https://arxiv.org/pdf/2511.05355v2](https://arxiv.org/pdf/2511.05355v2)**

> **作者:** Tzu-Yuan Huang; Armin Lederer; Dai-Jie Wu; Xiaobing Dai; Sihua Zhang; Stefan Sosnowski; Shao-Hua Sun; Sandra Hirche
>
> **摘要:** Flow matching (FM) has shown promising results in data-driven planning. However, it inherently lacks formal guarantees for ensuring state and action constraints, whose satisfaction is a fundamental and crucial requirement for the safety and admissibility of planned trajectories on various systems. Moreover, existing FM planners do not ensure the dynamical consistency, which potentially renders trajectories inexecutable. We address these shortcomings by proposing SAD-Flower, a novel framework for generating Safe, Admissible, and Dynamically consistent trajectories. Our approach relies on an augmentation of the flow with a virtual control input. Thereby, principled guidance can be derived using techniques from nonlinear control theory, providing formal guarantees for state constraints, action constraints, and dynamic consistency. Crucially, SAD-Flower operates without retraining, enabling test-time satisfaction of unseen constraints. Through extensive experiments across several tasks, we demonstrate that SAD-Flower outperforms various generative-model-based baselines in ensuring constraint satisfaction.
>
---
#### [replaced 025] Multimodal "Puppeteer": Exploring Robot Teleoperation Via Virtual Counterpart with LLM-Driven Voice and Gesture Interaction in Augmented Reality
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究增强现实（AR）中机器人远程操控的多模态交互。针对传统操控缺乏直观性的问题，提出基于大语言模型的语音与手势结合的虚拟化身操控框架。通过42名用户实验，比较纯手势与多模态方式，发现多模态虽灵活但易引入延迟与负担，提出需根据效率、鲁棒性与用户经验动态适配的交互设计原则。**

- **链接: [https://arxiv.org/pdf/2506.13189v2](https://arxiv.org/pdf/2506.13189v2)**

> **作者:** Yuchong Zhang; Bastian Orthmann; Shichen Ji; Michael Welle; Jonne Van Haastregt; Danica Kragic
>
> **备注:** This work is under peer review
>
> **摘要:** The integration of robotics and augmented reality (AR) offers promising opportunities to enhance human-robot interaction (HRI) by making teleoperation more transparent, spatially grounded, and intuitive. We present a head-mounted AR "puppeteer" framework in which users control a physical robot via interacting with its virtual counterpart robot using large language model (LLM)-driven voice commands and hand-gesture interaction on the Meta Quest 3. In a within-subject user study with 42 participants performing an AR-based robotic pick-and-place pattern-matching task, we compare two interaction conditions: gesture-only (GO) and combined voice+gesture (VG). Our results show that GO currently provides more reliable and efficient control for this time-critical task, while VG introduces additional flexibility but also latency and recognition issues that can increase workload. We further explore how prior robotics experience shapes participants' perceptions of each modality. Based on these findings, we distill a set of evidence-based design guidelines for AR puppeteer metaphoric robot teleoperation, implicating multimodality as an adaptive strategy that must balance efficiency, robustness, and user expertise rather than assuming that additional modalities are universally beneficial. Our work contributes empirical insights into how multimodal (voice+gesture) interaction influences task efficiency, usability, and user experience in AR-based HRI.
>
---
#### [replaced 026] SkillWrapper: Generative Predicate Invention for Skill Abstraction
- **分类: cs.RO**

- **简介: 该论文针对自主智能体在长时序任务中泛化能力弱的问题，提出SkillWrapper方法，通过生成式谓词发明，从RGB图像中学习可规划、可解释的高阶技能抽象表示。利用基础模型主动采集数据，实现对黑箱技能的符号化建模，支持可靠推理与规划，实现在真实世界中解决未见长时序任务。**

- **链接: [https://arxiv.org/pdf/2511.18203v2](https://arxiv.org/pdf/2511.18203v2)**

> **作者:** Ziyi Yang; Benned Hedegaard; Ahmed Jaafar; Yichen Wei; Skye Thompson; Shreyas S. Raman; Haotian Fu; Stefanie Tellex; George Konidaris; David Paulius; Naman Shah
>
> **摘要:** Generalizing from individual skill executions to solving long-horizon tasks remains a core challenge in building autonomous agents. A promising direction is learning high-level, symbolic abstractions of the low-level skills of the agents, enabling reasoning and planning independent of the low-level state space. Among possible high-level representations, object-centric skill abstraction with symbolic predicates has been proven to be efficient because of its compatibility with domain-independent planners. Recent advances in foundation models have made it possible to generate symbolic predicates that operate on raw sensory inputs, a process we call generative predicate invention, to facilitate downstream abstraction learning. However, it remains unclear which formal properties the learned representations must satisfy, and how they can be learned to guarantee these properties. In this paper, we address both questions by presenting a formal theory of generative predicate invention for skill abstraction, resulting in symbolic operators that can be used for provably sound and complete planning. Within this framework, we propose SkillWrapper, a method that leverages foundation models to actively collect robot data and learn human-interpretable, plannable representations of black-box skills, using only RGB image observations. Our extensive empirical evaluation in simulation and on real robots shows that SkillWrapper learns abstract representations that enable solving unseen, long-horizon tasks in the real world with black-box skills.
>
---
#### [replaced 027] RoboArena: Distributed Real-World Evaluation of Generalist Robot Policies
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出RoboArena，一种分布式真实世界评估通用机器人策略的方法。针对现有基准测试标准化过高、难以扩展的问题，通过跨机构的众包方式，让评估者在多样化任务与环境中进行双盲对比，基于偏好反馈聚合排名。实验验证其在可扩展性、鲁棒性和可信度上优于传统方法。**

- **链接: [https://arxiv.org/pdf/2506.18123v2](https://arxiv.org/pdf/2506.18123v2)**

> **作者:** Pranav Atreya; Karl Pertsch; Tony Lee; Moo Jin Kim; Arhan Jain; Artur Kuramshin; Clemens Eppner; Cyrus Neary; Edward Hu; Fabio Ramos; Jonathan Tremblay; Kanav Arora; Kirsty Ellis; Luca Macesanu; Marcel Torne Villasevil; Matthew Leonard; Meedeum Cho; Ozgur Aslan; Shivin Dass; Jie Wang; William Reger; Xingfang Yuan; Xuning Yang; Abhishek Gupta; Dinesh Jayaraman; Glen Berseth; Kostas Daniilidis; Roberto Martin-Martin; Youngwoon Lee; Percy Liang; Chelsea Finn; Sergey Levine
>
> **备注:** Website: https://robo-arena.github.io/
>
> **摘要:** Comprehensive, unbiased, and comparable evaluation of modern generalist policies is uniquely challenging: existing approaches for robot benchmarking typically rely on heavy standardization, either by specifying fixed evaluation tasks and environments, or by hosting centralized ''robot challenges'', and do not readily scale to evaluating generalist policies across a broad range of tasks and environments. In this work, we propose RoboArena, a new approach for scalable evaluation of generalist robot policies in the real world. Instead of standardizing evaluations around fixed tasks, environments, or locations, we propose to crowd-source evaluations across a distributed network of evaluators. Importantly, evaluators can freely choose the tasks and environments they evaluate on, enabling easy scaling of diversity, but they are required to perform double-blind evaluations over pairs of policies. Then, by aggregating preference feedback from pairwise comparisons across diverse tasks and environments, we can derive a ranking of policies. We instantiate our approach across a network of evaluators at seven academic institutions using the DROID robot platform. Through more than 600 pairwise real-robot evaluation episodes across seven generalist policies, we demonstrate that our crowd-sourced approach can more accurately rank the performance of existing generalist policies than conventional, centralized evaluation approaches, while being more scalable, resilient, and trustworthy. We open our evaluation network to the community and hope that it can enable more accessible comparisons of generalist robot policies.
>
---
#### [replaced 028] Adaptive Legged Locomotion via Online Learning for Model Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究四足机器人在未知环境下的自适应步态控制问题。针对模型误差与外部扰动导致的控制性能下降，提出基于在线学习残差动力学与模型预测控制（MPC）的算法，利用随机傅里叶特征逼近残差，并在线更新模型。实验验证其在复杂地形与动态干扰下具有优异的轨迹跟踪能力与次线性动态后悔性能。**

- **链接: [https://arxiv.org/pdf/2510.15626v2](https://arxiv.org/pdf/2510.15626v2)**

> **作者:** Hongyu Zhou; Xiaoyu Zhang; Vasileios Tzoumas
>
> **备注:** IEEE Robotics and Automation Letters
>
> **摘要:** We provide an algorithm for adaptive legged locomotion via online learning and model predictive control. The algorithm is composed of two interacting modules: model predictive control (MPC) and online learning of residual dynamics. The residual dynamics can represent modeling errors and external disturbances. We are motivated by the future of autonomy where quadrupeds will autonomously perform complex tasks despite real-world unknown uncertainty, such as unknown payload and uneven terrains. The algorithm uses random Fourier features to approximate the residual dynamics in reproducing kernel Hilbert spaces. Then, it employs MPC based on the current learned model of the residual dynamics. The model is updated online in a self-supervised manner using least squares based on the data collected while controlling the quadruped. The algorithm enjoys sublinear \textit{dynamic regret}, defined as the suboptimality against an optimal clairvoyant controller that knows how the residual dynamics. We validate our algorithm in Gazebo and MuJoCo simulations, where the quadruped aims to track reference trajectories. The Gazebo simulations include constant unknown external forces up to $12\boldsymbol{g}$, where $\boldsymbol{g}$ is the gravity vector, in flat terrain, slope terrain with $20\degree$ inclination, and rough terrain with $0.25m$ height variation. The MuJoCo simulations include time-varying unknown disturbances with payload up to $8~kg$ and time-varying ground friction coefficients in flat terrain.
>
---
#### [replaced 029] Curvature-Constrained Vector Field for Motion Planning of Nonholonomic Robots
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对非完整机器人运动规划中曲率约束难题，提出一种共设计向量场与控制律的框架。通过构建曲率受限向量场（CVF）和饱和控制律，确保机器人轨迹曲率有界并收敛至目标集，有效解决非完整系统下轨迹曲率与控制协同设计问题。**

- **链接: [https://arxiv.org/pdf/2504.02852v2](https://arxiv.org/pdf/2504.02852v2)**

> **作者:** Yike Qiao; Xiaodong He; An Zhuo; Zhiyong Sun; Weimin Bao; Zhongkui Li
>
> **备注:** IEEE T-RO accepted, 20 pages, 22 figures
>
> **摘要:** Vector fields are advantageous in handling nonholonomic motion planning as they provide reference orientation for robots. However, additionally incorporating curvature constraints becomes challenging, due to the interconnection between the design of the curvature-bounded vector field and the tracking controller under underactuation. In this paper, we present a novel framework to co-develop the vector field and the control laws, guiding the nonholonomic robot to the target configuration with curvature-bounded trajectory. First, we formulate the problem by introducing the target positive limit set, which allows the robot to converge to or pass through the target configuration, depending on different dynamics and tasks. Next, we construct a curvature-constrained vector field (CVF) via blending and distributing basic flow fields in workspace and propose the saturated control laws with a dynamic gain, under which the tracking error's magnitude decreases even when saturation occurs. Under the control laws, kinematically constrained nonholonomic robots are guaranteed to track the reference CVF and converge to the target positive limit set with bounded trajectory curvature. Numerical simulations show that the proposed CVF method outperforms other vector-field-based algorithms. Experiments on Ackermann UGVs and semi-physical fixed-wing UAVs demonstrate that the method can be effectively implemented in real-world scenarios.
>
---
#### [replaced 030] UniFucGrasp: Human-Hand-Inspired Unified Functional Grasp Annotation Strategy and Dataset for Diverse Dexterous Hands
- **分类: cs.RO; cs.CV; eess.IV**

- **简介: 该论文针对灵巧手抓取数据集缺乏功能性标注的问题，提出UniFucGrasp策略与多手功能抓取数据集。基于人体手部生物力学与几何力闭合原理，实现低成本、高效的功能性抓取标注，提升抓取准确率与跨手适应性，有效缓解标注成本高与泛化难问题。**

- **链接: [https://arxiv.org/pdf/2508.03339v2](https://arxiv.org/pdf/2508.03339v2)**

> **作者:** Haoran Lin; Wenrui Chen; Xianchi Chen; Fan Yang; Qiang Diao; Wenxin Xie; Sijie Wu; Kailun Yang; Maojun Li; Yaonan Wang
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L). The project page is at https://haochen611.github.io/UFG
>
> **摘要:** Dexterous grasp datasets are vital for embodied intelligence, but mostly emphasize grasp stability, ignoring functional grasps needed for tasks like opening bottle caps or holding cup handles. Most rely on bulky, costly, and hard-to-control high-DOF Shadow Hands. Inspired by the human hand's underactuated mechanism, we establish UniFucGrasp, a universal functional grasp annotation strategy and dataset for multiple dexterous hand types. Based on biomimicry, it maps natural human motions to diverse hand structures and uses geometry-based force closure to ensure functional, stable, human-like grasps. This method supports low-cost, efficient collection of diverse, high-quality functional grasps. Finally, we establish the first multi-hand functional grasp dataset and provide a synthesis model to validate its effectiveness. Experiments on the UFG dataset, IsaacSim, and complex robotic tasks show that our method improves functional manipulation accuracy and grasp stability, demonstrates improved adaptability across multiple robotic hands, helping to alleviate annotation cost and generalization challenges in dexterous grasping. The project page is at https://haochen611.github.io/UFG.
>
---
#### [replaced 031] A Minimal Subset Approach for Informed Keyframe Sampling in Large-Scale SLAM
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对大规模LiDAR SLAM中的关键帧采样问题，提出一种基于最小子集的在线采样方法（MSA），旨在减少冗余、保留关键信息。通过在特征空间中构建姿态图，有效降低误检率与计算开销，提升定位精度与系统效率，无需人工调参。**

- **链接: [https://arxiv.org/pdf/2501.01791v3](https://arxiv.org/pdf/2501.01791v3)**

> **作者:** Nikolaos Stathoulopoulos; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Please cite the published version. 8 pages, 9 figures
>
> **摘要:** Typical LiDAR SLAM architectures feature a front-end for odometry estimation and a back-end for refining and optimizing the trajectory and map, commonly through loop closures. However, loop closure detection in large-scale missions presents significant computational challenges due to the need to identify, verify, and process numerous candidate pairs for pose graph optimization. Keyframe sampling bridges the front-end and back-end by selecting frames for storing and processing during global optimization. This article proposes an online keyframe sampling approach that constructs the pose graph using the most impactful keyframes for loop closure. We introduce the Minimal Subset Approach (MSA), which optimizes two key objectives: redundancy minimization and information preservation, implemented within a sliding window framework. By operating in the feature space rather than 3-D space, MSA efficiently reduces redundant keyframes while retaining essential information. Evaluations on diverse public datasets show that the proposed approach outperforms naive methods in reducing false positive rates in place recognition, while delivering superior ATE and RPE in metric localization, without the need for manual parameter tuning. Additionally, MSA demonstrates efficiency and scalability by reducing memory usage and computational overhead during loop closure detection and pose graph optimization.
>
---
#### [replaced 032] HybridWorldSim: A Scalable and Controllable High-fidelity Simulator for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶仿真中视图切换不真实、几何不一致的问题，提出HybridWorldSim框架，融合神经重建与生成模型，实现高保真、可控的动态场景仿真。构建MIRROR数据集，支持多样化的城市环境测试。实验表明其显著优于现有方法，为自动驾驶研发提供可靠仿真平台。**

- **链接: [https://arxiv.org/pdf/2511.22187v2](https://arxiv.org/pdf/2511.22187v2)**

> **作者:** Qiang Li; Yingwenqi Jiang; Tuoxi Li; Duyu Chen; Xiang Feng; Yucheng Ao; Shangyue Liu; Xingchen Yu; Youcheng Cai; Yumeng Liu; Yuexin Ma; Xin Hu; Li Liu; Yu Zhang; Linkun Xu; Bingtao Gao; Xueyuan Wang; Shuchang Zhou; Xianming Liu; Ligang Liu
>
> **摘要:** Realistic and controllable simulation is critical for advancing end-to-end autonomous driving, yet existing approaches often struggle to support novel view synthesis under large viewpoint changes or to ensure geometric consistency. We introduce HybridWorldSim, a hybrid simulation framework that integrates multi-traversal neural reconstruction for static backgrounds with generative modeling for dynamic agents. This unified design addresses key limitations of previous methods, enabling the creation of diverse and high-fidelity driving scenarios with reliable visual and spatial consistency. To facilitate robust benchmarking, we further release a new multi-traversal dataset MIRROR that captures a wide range of routes and environmental conditions across different cities. Extensive experiments demonstrate that HybridWorldSim surpasses previous state-of-the-art methods, providing a practical and scalable solution for high-fidelity simulation and a valuable resource for research and development in autonomous driving.
>
---
#### [replaced 033] HiMo: High-Speed Objects Motion Compensation in Point Clouds
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中LiDAR点云因高速动态物体引起的运动畸变问题，提出HiMo框架，通过改进的场景流估计实现非自车运动补偿。提出SeFlow++实现实时高精度场景流估计，并引入新评估指标验证效果，显著提升点云几何一致性和下游任务性能。**

- **链接: [https://arxiv.org/pdf/2503.00803v3](https://arxiv.org/pdf/2503.00803v3)**

> **作者:** Qingwen Zhang; Ajinkya Khoche; Yi Yang; Li Ling; Sina Sharif Mansouri; Olov Andersson; Patric Jensfelt
>
> **备注:** 15 pages, 13 figures, Published in Transactions on Robotics (Volume 41)
>
> **摘要:** LiDAR point cloud is essential for autonomous vehicles, but motion distortions from dynamic objects degrade the data quality. While previous work has considered distortions caused by ego motion, distortions caused by other moving objects remain largely overlooked, leading to errors in object shape and position. This distortion is particularly pronounced in high-speed environments such as highways and in multi-LiDAR configurations, a common setup for heavy vehicles. To address this challenge, we introduce HiMo, a pipeline that repurposes scene flow estimation for non-ego motion compensation, correcting the representation of dynamic objects in point clouds. During the development of HiMo, we observed that existing self-supervised scene flow estimators often produce degenerate or inconsistent estimates under high-speed distortion. We further propose SeFlow++, a real-time scene flow estimator that achieves state-of-the-art performance on both scene flow and motion compensation. Since well-established motion distortion metrics are absent in the literature, we introduce two evaluation metrics: compensation accuracy at a point level and shape similarity of objects. We validate HiMo through extensive experiments on Argoverse 2, ZOD, and a newly collected real-world dataset featuring highway driving and multi-LiDAR-equipped heavy vehicles. Our findings show that HiMo improves the geometric consistency and visual fidelity of dynamic objects in LiDAR point clouds, benefiting downstream tasks such as semantic segmentation and 3D detection. See https://kin-zhang.github.io/HiMo for more details.
>
---
#### [replaced 034] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦无人机场景下的指代多目标跟踪（RMOT）任务，旨在解决地面场景下视角局限、难以实现广域感知与路径规划的问题。提出首个大规模无人机RMOT基准数据集AerialMind，并开发COALA标注框架降低人工成本；同时提出HETrack方法，通过视觉-语言协同学习提升空中场景理解能力。**

- **链接: [https://arxiv.org/pdf/2511.21053v2](https://arxiv.org/pdf/2511.21053v2)**

> **作者:** Chenglizhao Chen; Shaofeng Liang; Runwei Guan; Xiaolou Sun; Haocheng Zhao; Haiyun Jiang; Tao Huang; Henghui Ding; Qing-Long Han
>
> **备注:** AAAI 2026
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to achieve precise object detection and tracking through natural language instructions, representing a fundamental capability for intelligent robotic systems. However, current RMOT research remains mostly confined to ground-level scenarios, which constrains their ability to capture broad-scale scene contexts and perform comprehensive tracking and path planning. In contrast, Unmanned Aerial Vehicles (UAVs) leverage their expansive aerial perspectives and superior maneuverability to enable wide-area surveillance. Moreover, UAVs have emerged as critical platforms for Embodied Intelligence, which has given rise to an unprecedented demand for intelligent aerial systems capable of natural language interaction. To this end, we introduce AerialMind, the first large-scale RMOT benchmark in UAV scenarios, which aims to bridge this research gap. To facilitate its construction, we develop an innovative semi-automated collaborative agent-based labeling assistant (COALA) framework that significantly reduces labor costs while maintaining annotation quality. Furthermore, we propose HawkEyeTrack (HETrack), a novel method that collaboratively enhances vision-language representation learning and improves the perception of UAV scenarios. Comprehensive experiments validated the challenging nature of our dataset and the effectiveness of our method.
>
---
#### [replaced 035] Delta-Triplane Transformers as Occupancy World Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Delta-Triplane Transformers（DTT），用于自动驾驶中的占用世界建模（OWM）任务。针对现有方法计算冗余、效率低的问题，提出基于三平面的紧凑表示与增量预测策略，通过建模占用变化而非全状态，实现更高效、高精度的未来场景预测与路径规划。**

- **链接: [https://arxiv.org/pdf/2503.07338v4](https://arxiv.org/pdf/2503.07338v4)**

> **作者:** Haoran Xu; Peixi Peng; Guang Tan; Yiqian Chang; Yisen Zhao; Yonghong Tian
>
> **摘要:** Occupancy World Models (OWMs) aim to predict future scenes via 3D voxelized representations of the environment to support intelligent motion planning. Existing approaches typically generate full future occupancy states from VAE-style latent encodings, which can be computationally expensive and redundant. We propose Delta-Triplane Transformers (DTT), a novel 4D OWM for autonomous driving, that introduces two key innovations: (1) a triplane based representation that encodes 3D occupancy more compactly than previous approaches, and (2) an incremental prediction strategy for OWM that models {\em changes} in occupancy rather than dealing with full states. The core insight is that changes in the compact 3D latent space are naturally sparser and easier to model, enabling higher accuracy with a lighter-weight architecture. Building on this representation, DTT extracts multi-scale motion features from historical data and iteratively predict future triplane deltas. These deltas are combined with past states to decode future occupancy and ego-motion trajectories. Extensive experiments demonstrate that DTT delivers a 1.44$\times$ speedup (26 FPS) over the state of the art, improves mean IoU to 30.85, and reduces the mean absolute planning error to 1.0 meters. Demo videos are provided in the supplementary material.
>
---
#### [replaced 036] Towards Fully Onboard State Estimation and Trajectory Tracking for UAVs with Suspended Payloads
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对无人机悬吊载荷的位姿估计与轨迹跟踪问题，提出仅依赖标准机载传感器（RTK-GNSS与IMU）的全机载解决方案。通过耦合动力学建模、卡尔曼滤波与模型预测控制，实现高精度、鲁棒的实时控制，在仿真与实测中均验证了其有效性与实用性。**

- **链接: [https://arxiv.org/pdf/2508.11547v2](https://arxiv.org/pdf/2508.11547v2)**

> **作者:** Martin Jiroušek; Tomáš Báča; Martin Saska
>
> **备注:** Updated to match the published version. Added journal reference and DOI
>
> **摘要:** This paper addresses the problem of tracking the position of a cable-suspended payload carried by an unmanned aerial vehicle, with a focus on real-world deployment and minimal hardware requirements. In contrast to many existing approaches that rely on motion-capture systems, additional onboard cameras, or instrumented payloads, we propose a framework that uses only standard onboard sensors--specifically, real-time kinematic global navigation satellite system measurements and data from the onboard inertial measurement unit--to estimate and control the payload's position. The system models the full coupled dynamics of the aerial vehicle and payload, and integrates a linear Kalman filter for state estimation, a model predictive contouring control planner, and an incremental model predictive controller. The control architecture is designed to remain effective despite sensing limitations and estimation uncertainty. Extensive simulations demonstrate that the proposed system achieves performance comparable to control based on ground-truth measurements, with only minor degradation (< 6%). The system also shows strong robustness to variations in payload parameters. Field experiments further validate the framework, confirming its practical applicability and reliable performance in outdoor environments using only off-the-shelf aerial vehicle hardware.
>
---
