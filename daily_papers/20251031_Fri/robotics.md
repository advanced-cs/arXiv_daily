# 机器人 cs.RO

- **最新发布 39 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] FLYINGTRUST: A Benchmark for Quadrotor Navigation Across Scenarios and Vehicles
- **分类: cs.RO**

- **简介: 该论文提出FLYINGTRUST基准框架，用于评估四旋翼无人机在不同平台与场景下的视觉导航性能。针对算法跨平台迁移性差的问题，构建融合动力学与场景结构的标准化评测体系，量化平台能力与环境复杂度对导航鲁棒性的影响，揭示算法偏好与失效模式，推动更通用的导航方法设计。**

- **链接: [http://arxiv.org/pdf/2510.26588v1](http://arxiv.org/pdf/2510.26588v1)**

> **作者:** Gang Li; Chunlei Zhai; Teng Wang; Shaun Li; Shangsong Jiang; Xiangwei Zhu
>
> **摘要:** Visual navigation algorithms for quadrotors often exhibit a large variation in performance when transferred across different vehicle platforms and scene geometries, which increases the cost and risk of field deployment. To support systematic early-stage evaluation, we introduce FLYINGTRUST, a high-fidelity, configurable benchmarking framework that measures how platform kinodynamics and scenario structure jointly affect navigation robustness. FLYINGTRUST models vehicle capability with two compact, physically interpretable indicators: maximum thrust-to-weight ratio and axis-wise maximum angular acceleration. The benchmark pairs a diverse scenario library with a heterogeneous set of real and virtual platforms and prescribes a standardized evaluation protocol together with a composite scoring method that balances scenario importance, platform importance and performance stability. We use FLYINGTRUST to compare representative optimization-based and learning-based navigation approaches under identical conditions, performing repeated trials per platform-scenario combination and reporting uncertainty-aware metrics. The results reveal systematic patterns: navigation success depends predictably on platform capability and scene geometry, and different algorithms exhibit distinct preferences and failure modes across the evaluated conditions. These observations highlight the practical necessity of incorporating both platform capability and scenario structure into algorithm design, evaluation, and selection, and they motivate future work on methods that remain robust across diverse platforms and scenarios.
>
---
#### [new 002] Kinodynamic Task and Motion Planning using VLM-guided and Interleaved Sampling
- **分类: cs.RO**

- **简介: 该论文提出一种基于视觉语言模型（VLM）引导的混合状态树框架，解决长时程任务与运动规划（TAMP）中采样成本高、动态可行性难保障的问题。通过统一符号与数值状态表示，结合VLM视觉反馈与物理仿真验证，实现任务与运动协同决策，显著提升成功率并降低规划时间。**

- **链接: [http://arxiv.org/pdf/2510.26139v1](http://arxiv.org/pdf/2510.26139v1)**

> **作者:** Minseo Kwon; Young J. Kim
>
> **摘要:** Task and Motion Planning (TAMP) integrates high-level task planning with low-level motion feasibility, but existing methods are costly in long-horizon problems due to excessive motion sampling. While LLMs provide commonsense priors, they lack 3D spatial reasoning and cannot ensure geometric or dynamic feasibility. We propose a kinodynamic TAMP framework based on a hybrid state tree that uniformly represents symbolic and numeric states during planning, enabling task and motion decisions to be jointly decided. Kinodynamic constraints embedded in the TAMP problem are verified by an off-the-shelf motion planner and physics simulator, and a VLM guides exploring a TAMP solution and backtracks the search based on visual rendering of the states. Experiments on the simulated domains and in the real world show 32.14% - 1166.67% increased average success rates compared to traditional and LLM-based TAMP planners and reduced planning time on complex problems, with ablations further highlighting the benefits of VLM guidance.
>
---
#### [new 003] Risk-Aware Safety Filters with Poisson Safety Functions and Laplace Guidance Fields
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对机器人在复杂环境中的安全导航问题，提出一种风险感知的安全过滤方法。通过求解泊松方程生成安全函数，结合拉普拉斯方程构建可调风险引导场，融合二者形成风险感知的安全约束，实现对高风险障碍物的优先避让，保障系统安全。**

- **链接: [http://arxiv.org/pdf/2510.25913v1](http://arxiv.org/pdf/2510.25913v1)**

> **作者:** Gilbert Bahati; Ryan M. Bena; Meg Wilkinson; Pol Mestres; Ryan K. Cosner; Aaron D. Ames
>
> **摘要:** Robotic systems navigating in real-world settings require a semantic understanding of their environment to properly determine safe actions. This work aims to develop the mathematical underpinnings of such a representation--specifically, the goal is to develop safety filters that are risk-aware. To this end, we take a two step approach: encoding an understanding of the environment via Poisson's equation, and associated risk via Laplace guidance fields. That is, we first solve a Dirichlet problem for Poisson's equation to generate a safety function that encodes system safety as its 0-superlevel set. We then separately solve a Dirichlet problem for Laplace's equation to synthesize a safe \textit{guidance field} that encodes variable levels of caution around obstacles -- by enforcing a tunable flux boundary condition. The safety function and guidance fields are then combined to define a safety constraint and used to synthesize a risk-aware safety filter which, given a semantic understanding of an environment with associated risk levels of environmental features, guarantees safety while prioritizing avoidance of higher risk obstacles. We demonstrate this method in simulation and discuss how \textit{a priori} understandings of obstacle risk can be directly incorporated into the safety filter to generate safe behaviors that are risk-aware.
>
---
#### [new 004] Cooperative Task Spaces for Multi-Arm Manipulation Control based on Similarity Transformations
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究多机械臂协同操作控制，针对高自由度系统运动协调困难问题，提出基于保角几何代数的协作任务空间方法。通过相似变换抽象复杂系统，构建等效单臂模型，推导雅可比矩阵，实现与经典操作空间控制的无缝集成，并自然嵌入零空间结构以支持次级目标。**

- **链接: [http://arxiv.org/pdf/2510.26362v1](http://arxiv.org/pdf/2510.26362v1)**

> **作者:** Tobias Löw; Cem Bilaloglu; Sylvain Calinon
>
> **摘要:** Many tasks in human environments require collaborative behavior between multiple kinematic chains, either to provide additional support for carrying big and bulky objects or to enable the dexterity that is required for in-hand manipulation. Since these complex systems often have a very high number of degrees of freedom coordinating their movements is notoriously difficult to model. In this article, we present the derivation of the theoretical foundations for cooperative task spaces of multi-arm robotic systems based on geometric primitives defined using conformal geometric algebra. Based on the similarity transformations of these cooperative geometric primitives, we derive an abstraction of complex robotic systems that enables representing these systems in a way that directly corresponds to single-arm systems. By deriving the associated analytic and geometric Jacobian matrices, we then show the straightforward integration of our approach into classical control techniques rooted in operational space control. We demonstrate this using bimanual manipulators, humanoids and multi-fingered hands in optimal control experiments for reaching desired geometric primitives and in teleoperation experiments using differential kinematics control. We then discuss how the geometric primitives naturally embed nullspace structures into the controllers that can be exploited for introducing secondary control objectives. This work, represents the theoretical foundations of this cooperative manipulation control framework, and thus the experiments are presented in an abstract way, while giving pointers towards potential future applications.
>
---
#### [new 005] AgriGS-SLAM: Orchard Mapping Across Seasons via Multi-View Gaussian Splatting SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AgriGS-SLAM，面向果园季节性变化下的实时3D场景重建任务。针对重复结构、外观变化与叶动导致的定位与建图挑战，融合多视角3D高斯点云渲染与激光雷达直接里程计，通过批处理光栅化与梯度驱动地图管理，在保证实时性的同时提升重建精度与轨迹稳定性。**

- **链接: [http://arxiv.org/pdf/2510.26358v1](http://arxiv.org/pdf/2510.26358v1)**

> **作者:** Mirko Usuelli; David Rapado-Rincon; Gert Kootstra; Matteo Matteucci
>
> **摘要:** Autonomous robots in orchards require real-time 3D scene understanding despite repetitive row geometry, seasonal appearance changes, and wind-driven foliage motion. We present AgriGS-SLAM, a Visual--LiDAR SLAM framework that couples direct LiDAR odometry and loop closures with multi-camera 3D Gaussian Splatting (3DGS) rendering. Batch rasterization across complementary viewpoints recovers orchard structure under occlusions, while a unified gradient-driven map lifecycle executed between keyframes preserves fine details and bounds memory. Pose refinement is guided by a probabilistic LiDAR-based depth consistency term, back-propagated through the camera projection to tighten geometry-appearance coupling. We deploy the system on a field platform in apple and pear orchards across dormancy, flowering, and harvesting, using a standardized trajectory protocol that evaluates both training-view and novel-view synthesis to reduce 3DGS overfitting in evaluation. Across seasons and sites, AgriGS-SLAM delivers sharper, more stable reconstructions and steadier trajectories than recent state-of-the-art 3DGS-SLAM baselines while maintaining real-time performance on-tractor. While demonstrated in orchard monitoring, the approach can be applied to other outdoor domains requiring robust multimodal perception.
>
---
#### [new 006] REALMS2 - Resilient Exploration And Lunar Mapping System 2 - A Comprehensive Approach
- **分类: cs.RO**

- **简介: 该论文针对行星探测中多机器人系统的协同作业难题，提出REALMS2框架。基于ROS 2与vSLAM技术实现自主建图，采用自组网保障通信鲁棒性，通过统一GUI控制多机器人，在模拟月面环境中成功完成60%区域测绘，有效应对通信延迟与中断问题。**

- **链接: [http://arxiv.org/pdf/2510.26638v1](http://arxiv.org/pdf/2510.26638v1)**

> **作者:** Dave van der Meer; Loïck P. Chovet; Gabriel M. Garcia; Abhishek Bera; Miguel A. Olivares-Mendez
>
> **备注:** 8 Pages, 8 Figures, Submitted and Accepted to IROS 2025
>
> **摘要:** The European Space Agency (ESA) and the European Space Resources Innovation Centre (ESRIC) created the Space Resources Challenge to invite researchers and companies to propose innovative solutions for Multi-Robot Systems (MRS) space prospection. This paper proposes the Resilient Exploration And Lunar Mapping System 2 (REALMS2), a MRS framework for planetary prospection and mapping. Based on Robot Operating System version 2 (ROS 2) and enhanced with Visual Simultaneous Localisation And Mapping (vSLAM) for map generation, REALMS2 uses a mesh network for a robust ad hoc network. A single graphical user interface (GUI) controls all the rovers, providing a simple overview of the robotic mission. This system is designed for heterogeneous multi-robot exploratory missions, tackling the challenges presented by extraterrestrial environments. REALMS2 was used during the second field test of the ESA-ESRIC Challenge and allowed to map around 60% of the area, using three homogeneous rovers while handling communication delays and blackouts.
>
---
#### [new 007] Hybrid DQN-TD3 Reinforcement Learning for Autonomous Navigation in Dynamic Environments
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对动态环境中自主导航任务，提出混合DQN-TD3强化学习框架。高层DQN选择离散子目标，低层TD3执行连续控制，结合奖励塑形与安全门机制，提升成功率、效率与泛化能力，优于单一算法与规则方法。**

- **链接: [http://arxiv.org/pdf/2510.26646v1](http://arxiv.org/pdf/2510.26646v1)**

> **作者:** Xiaoyi He; Danggui Chen; Zhenshuo Zhang; Zimeng Bai
>
> **备注:** 6 pages, 5 figures; ROS+Gazebo (TurtleBot3) implementation; evaluation with PathBench metrics; code (primary): https://github.com/MayaCHEN-github/HierarchicalRL-robot-navigation; mirror (for reproducibility): https://github.com/ShowyHe/DRL-robot-navigation
>
> **摘要:** This paper presents a hierarchical path-planning and control framework that combines a high-level Deep Q-Network (DQN) for discrete sub-goal selection with a low-level Twin Delayed Deep Deterministic Policy Gradient (TD3) controller for continuous actuation. The high-level module selects behaviors and sub-goals; the low-level module executes smooth velocity commands. We design a practical reward shaping scheme (direction, distance, obstacle avoidance, action smoothness, collision penalty, time penalty, and progress), together with a LiDAR-based safety gate that prevents unsafe motions. The system is implemented in ROS + Gazebo (TurtleBot3) and evaluated with PathBench metrics, including success rate, collision rate, path efficiency, and re-planning efficiency, in dynamic and partially observable environments. Experiments show improved success rate and sample efficiency over single-algorithm baselines (DQN or TD3 alone) and rule-based planners, with better generalization to unseen obstacle configurations and reduced abrupt control changes. Code and evaluation scripts are available at the project repository.
>
---
#### [new 008] Running VLAs at Real-time Speed
- **分类: cs.RO**

- **简介: 该论文聚焦于实时机器人控制任务，旨在解决大型视觉语言模型（VLA）推理速度慢、难以实现实时应用的问题。通过优化推理策略，实现单张消费级GPU上30Hz视频帧率与480Hz轨迹频率的运行，成功完成动态抓取任务，并提出全流式推理框架。**

- **链接: [http://arxiv.org/pdf/2510.26742v1](http://arxiv.org/pdf/2510.26742v1)**

> **作者:** Yunchao Ma; Yizhuang Zhou; Yunhuan Yang; Tiancai Wang; Haoqiang Fan
>
> **备注:** Code is available at https://github.com/Dexmal/realtime-vla
>
> **摘要:** In this paper, we show how to run pi0-level multi-view VLA at 30Hz frame rate and at most 480Hz trajectory frequency using a single consumer GPU. This enables dynamic and real-time tasks that were previously believed to be unattainable by large VLA models. To achieve it, we introduce a bag of strategies to eliminate the overheads in model inference. The real-world experiment shows that the pi0 policy with our strategy achieves a 100% success rate in grasping a falling pen task. Based on the results, we further propose a full streaming inference framework for real-time robot control of VLA. Code is available at https://github.com/Dexmal/realtime-vla.
>
---
#### [new 009] PHUMA: Physically-Grounded Humanoid Locomotion Dataset
- **分类: cs.RO**

- **简介: 该论文提出PHUMA，一个大规模、物理可信的人形机器人行走动作数据集。针对现有数据集稀缺或含物理伪影的问题，通过视频采集与物理约束重定向，确保动作真实可靠。实验表明，基于PHUMA训练的策略在动作模仿与路径追踪任务中均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.26236v1](http://arxiv.org/pdf/2510.26236v1)**

> **作者:** Kyungmin Lee; Sibeen Kim; Minho Park; Hyunseung Kim; Dongyoon Hwang; Hojoon Lee; Jaegul Choo
>
> **摘要:** Motion imitation is a promising approach for humanoid locomotion, enabling agents to acquire humanlike behaviors. Existing methods typically rely on high-quality motion capture datasets such as AMASS, but these are scarce and expensive, limiting scalability and diversity. Recent studies attempt to scale data collection by converting large-scale internet videos, exemplified by Humanoid-X. However, they often introduce physical artifacts such as floating, penetration, and foot skating, which hinder stable imitation. In response, we introduce PHUMA, a Physically-grounded HUMAnoid locomotion dataset that leverages human video at scale, while addressing physical artifacts through careful data curation and physics-constrained retargeting. PHUMA enforces joint limits, ensures ground contact, and eliminates foot skating, producing motions that are both large-scale and physically reliable. We evaluated PHUMA in two sets of conditions: (i) imitation of unseen motion from self-recorded test videos and (ii) path following with pelvis-only guidance. In both cases, PHUMA-trained policies outperform Humanoid-X and AMASS, achieving significant gains in imitating diverse motions. The code is available at https://davian-robotics.github.io/PHUMA.
>
---
#### [new 010] RADRON: Cooperative Localization of Ionizing Radiation Sources by MAVs with Compton Cameras
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RADRON系统，利用微型康普顿相机与多架微型飞行器（MAVs）协同定位电离辐射源。针对传统辐射探测受限于设备重量与实时性问题，通过机载数据融合与动态反馈控制，实现稀疏测量下的实时源定位与移动源追踪，提升探测效率与精度。**

- **链接: [http://arxiv.org/pdf/2510.26018v1](http://arxiv.org/pdf/2510.26018v1)**

> **作者:** Petr Stibinger; Tomas Baca; Daniela Doubravova; Jan Rusnak; Jaroslav Solc; Jan Jakubek; Petr Stepan; Martin Saska
>
> **备注:** 8 pages, 9 figures, submitted for review to IEEE RA-L
>
> **摘要:** We present a novel approach to localizing radioactive material by cooperating Micro Aerial Vehicles (MAVs). Our approach utilizes a state-of-the-art single-detector Compton camera as a highly sensitive, yet miniature detector of ionizing radiation. The detector's exceptionally low weight (40 g) opens up new possibilities of radiation detection by a team of cooperating agile MAVs. We propose a new fundamental concept of fusing the Compton camera measurements to estimate the position of the radiation source in real time even from extremely sparse measurements. The data readout and processing are performed directly onboard and the results are used in a dynamic feedback to drive the motion of the vehicles. The MAVs are stabilized in a tightly cooperating swarm to maximize the information gained by the Compton cameras, rapidly locate the radiation source, and even track a moving radiation source.
>
---
#### [new 011] Towards Reinforcement Learning Based Log Loading Automation
- **分类: cs.RO**

- **简介: 该论文研究基于强化学习的林木装载自动化任务，旨在减轻操作员疲劳。通过构建仿真环境与课程学习策略，训练智能体完成从定位、抓取到运输并放置到装载床的全流程自动装载，成功率达94%。**

- **链接: [http://arxiv.org/pdf/2510.26363v1](http://arxiv.org/pdf/2510.26363v1)**

> **作者:** Ilya Kurinov; Miroslav Ivanov; Grzegorz Orzechowski; Aki Mikkola
>
> **摘要:** Forestry forwarders play a central role in mechanized timber harvesting by picking up and moving logs from the felling site to a processing area or a secondary transport vehicle. Forwarder operation is challenging and physically and mentally exhausting for the operator who must control the machine in remote areas for prolonged periods of time. Therefore, even partial automation of the process may reduce stress on the operator. This study focuses on continuing previous research efforts in application of reinforcement learning agents in automating log handling process, extending the task from grasping which was studied in previous research to full log loading operation. The resulting agent will be capable to automate a full loading procedure from locating and grappling to transporting and delivering the log to a forestry forwarder bed. To train the agent, a trailer type forestry forwarder simulation model in NVIDIA's Isaac Gym and a virtual environment for a typical log loading scenario were developed. With reinforcement learning agents and a curriculum learning approach, the trained agent may be a stepping stone towards application of reinforcement learning agents in automation of the forestry forwarder. The agent learnt grasping a log in a random position from grapple's random position and transport it to the bed with 94% success rate of the best performing agent.
>
---
#### [new 012] Accelerating Real-World Overtaking in F1TENTH Racing Employing Reinforcement Learning Methods
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对自主赛车中真实场景下超车难题，提出基于强化学习的竞速与超车智能体。通过在仿真与实车中训练对抗性策略，显著提升超车成功率（87%），解决现有算法难以安全可靠完成超车的问题。**

- **链接: [http://arxiv.org/pdf/2510.26040v1](http://arxiv.org/pdf/2510.26040v1)**

> **作者:** Emily Steiner; Daniel van der Spuy; Futian Zhou; Afereti Pama; Minas Liarokapis; Henry Williams
>
> **摘要:** While autonomous racing performance in Time-Trial scenarios has seen significant progress and development, autonomous wheel-to-wheel racing and overtaking are still severely limited. These limitations are particularly apparent in real-life driving scenarios where state-of-the-art algorithms struggle to safely or reliably complete overtaking manoeuvres. This is important, as reliable navigation around other vehicles is vital for safe autonomous wheel-to-wheel racing. The F1Tenth Competition provides a useful opportunity for developing wheel-to-wheel racing algorithms on a standardised physical platform. The competition format makes it possible to evaluate overtaking and wheel-to-wheel racing algorithms against the state-of-the-art. This research presents a novel racing and overtaking agent capable of learning to reliably navigate a track and overtake opponents in both simulation and reality. The agent was deployed on an F1Tenth vehicle and competed against opponents running varying competitive algorithms in the real world. The results demonstrate that the agent's training against opponents enables deliberate overtaking behaviours with an overtaking rate of 87% compared 56% for an agent trained just to race.
>
---
#### [new 013] Beyond the Uncanny Valley: A Mixed-Method Investigation of Anthropomorphism in Protective Responses to Robot Abuse
- **分类: cs.RO**

- **简介: 该论文研究机器人拟人化程度对人类保护性反应的影响，旨在突破“恐怖谷”理论在道德领域的局限。通过混合方法实验（问卷、面部表情分析、质性访谈），发现中等拟人机器人引发最强愤怒与道德谴责，推动从财产保护向类动物权利的治理思考，揭示拟人化反而增强道德关怀，为机器人设计与立法提供依据。**

- **链接: [http://arxiv.org/pdf/2510.26082v1](http://arxiv.org/pdf/2510.26082v1)**

> **作者:** Fan Yang; Lingyao Li; Yaxin Hu; Michael Rodgers; Renkai Ma
>
> **摘要:** Robots with anthropomorphic features are increasingly shaping how humans perceive and morally engage with them. Our research investigates how different levels of anthropomorphism influence protective responses to robot abuse, extending the Computers as Social Actors (CASA) and uncanny valley theories into a moral domain. In an experiment, we invite 201 participants to view videos depicting abuse toward a robot with low (Spider), moderate (Two-Foot), or high (Humanoid) anthropomorphism. To provide a comprehensive analysis, we triangulate three modalities: self-report surveys measuring emotions and uncanniness, physiological data from automated facial expression analysis, and qualitative reflections. Findings indicate that protective responses are not linear. The moderately anthropomorphic Two-Foot robot, rated highest in eeriness and "spine-tingling" sensations consistent with the uncanny valley, elicited the strongest physiological anger expressions. Self-reported anger and guilt are significantly higher for both the Two-Foot and Humanoid robots compared to the Spider. Qualitative findings further reveal that as anthropomorphism increases, moral reasoning shifts from technical assessments of property damage to condemnation of the abuser's character, while governance proposals expand from property law to calls for quasi-animal rights and broader societal responsibility. These results suggest that the uncanny valley does not dampen moral concern but paradoxically heightens protective impulses, offering critical implications for robot design, policy, and future legal frameworks.
>
---
#### [new 014] Human-in-the-loop Online Rejection Sampling for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对机器人抓取任务中强化学习不稳定、模仿学习泛化差的问题，提出Hi-ORS方法。通过在线拒绝采样稳定价值估计，并结合人类在线修正提供密集中间监督，实现高效高鲁棒性训练，仅用1.5小时即掌握复杂接触操作，显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.26406v1](http://arxiv.org/pdf/2510.26406v1)**

> **作者:** Guanxing Lu; Rui Zhao; Haitao Lin; He Zhang; Yansong Tang
>
> **备注:** 8 pages
>
> **摘要:** Reinforcement learning (RL) is widely used to produce robust robotic manipulation policies, but fine-tuning vision-language-action (VLA) models with RL can be unstable due to inaccurate value estimates and sparse supervision at intermediate steps. In contrast, imitation learning (IL) is easy to train but often underperforms due to its offline nature. In this paper, we propose Hi-ORS, a simple yet effective post-training method that utilizes rejection sampling to achieve both training stability and high robustness. Hi-ORS stabilizes value estimation by filtering out negatively rewarded samples during online fine-tuning, and adopts a reward-weighted supervised training objective to provide dense intermediate-step supervision. For systematic study, we develop an asynchronous inference-training framework that supports flexible online human-in-the-loop corrections, which serve as explicit guidance for learning error-recovery behaviors. Across three real-world tasks and two embodiments, Hi-ORS fine-tunes a pi-base policy to master contact-rich manipulation in just 1.5 hours of real-world training, outperforming RL and IL baselines by a substantial margin in both effectiveness and efficiency. Notably, the fine-tuned policy exhibits strong test-time scalability by reliably executing complex error-recovery behaviors to achieve better performance.
>
---
#### [new 015] RoboOS-NeXT: A Unified Memory-based Framework for Lifelong, Scalable, and Robust Multi-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文提出RoboOS-NeXT框架，解决多机器人系统在长期协作中面临的可扩展性与鲁棒性难题。通过引入时空体感记忆（STEM）实现统一记忆表征，结合脑-小脑架构，支持动态任务分配与容错协同，在餐厅、超市等场景验证了其在异构机器人协作中的优越性能。**

- **链接: [http://arxiv.org/pdf/2510.26536v1](http://arxiv.org/pdf/2510.26536v1)**

> **作者:** Huajie Tan; Cheng Chi; Xiansheng Chen; Yuheng Ji; Zhongxia Zhao; Xiaoshuai Hao; Yaoxu Lyu; Mingyu Cao; Junkai Zhao; Huaihai Lyu; Enshen Zhou; Ning Chen; Yankai Fu; Cheng Peng; Wei Guo; Dong Liang; Zhuo Chen; Mengsi Lyu; Chenrui He; Yulong Ao; Yonghua Lin; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **摘要:** The proliferation of collaborative robots across diverse tasks and embodiments presents a central challenge: achieving lifelong adaptability, scalable coordination, and robust scheduling in multi-agent systems. Existing approaches, from vision-language-action (VLA) models to hierarchical frameworks, fall short due to their reliance on limited or dividual-agent memory. This fundamentally constrains their ability to learn over long horizons, scale to heterogeneous teams, or recover from failures, highlighting the need for a unified memory representation. To address these limitations, we introduce RoboOS-NeXT, a unified memory-based framework for lifelong, scalable, and robust multi-robot collaboration. At the core of RoboOS-NeXT is the novel Spatio-Temporal-Embodiment Memory (STEM), which integrates spatial scene geometry, temporal event history, and embodiment profiles into a shared representation. This memory-centric design is integrated into a brain-cerebellum framework, where a high-level brain model performs global planning by retrieving and updating STEM, while low-level controllers execute actions locally. This closed loop between cognition, memory, and execution enables dynamic task allocation, fault-tolerant collaboration, and consistent state synchronization. We conduct extensive experiments spanning complex coordination tasks in restaurants, supermarkets, and households. Our results demonstrate that RoboOS-NeXT achieves superior performance across heterogeneous embodiments, validating its effectiveness in enabling lifelong, scalable, and robust multi-robot collaboration. Project website: https://flagopen.github.io/RoboOS/
>
---
#### [new 016] I don't Want You to Die: A Shared Responsibility Framework for Safeguarding Child-Robot Companionship
- **分类: cs.RO**

- **简介: 该论文研究社会机器人服务中断对儿童情感伤害的责任归属问题。基于Moxie停服案例，通过72人调查，提出共享责任框架，揭示公司、家长、开发者与政府共同担责的共识及分歧，探讨保障儿童机器人陪伴持续性的设计与政策路径。**

- **链接: [http://arxiv.org/pdf/2510.26080v1](http://arxiv.org/pdf/2510.26080v1)**

> **作者:** Fan Yang; Renkai Ma; Yaxin Hu; Michael Rodgers; Lingyao Li
>
> **摘要:** Social robots like Moxie are designed to form strong emotional bonds with children, but their abrupt discontinuation can cause significant struggles and distress to children. When these services end, the resulting harm raises complex questions of who bears responsibility when children's emotional bonds are broken. Using the Moxie shutdown as a case study through a qualitative survey of 72 U.S. participants, our findings show that the responsibility is viewed as a shared duty across the robot company, parents, developers, and government. However, these attributions varied by political ideology and parental status of whether they have children. Participants' perceptions of whether the robot service should continue are highly polarized; supporters propose technical, financial, and governmental pathways for continuity, while opponents cite business realities and risks of unhealthy emotional dependency. Ultimately, this research contributes an empirically grounded shared responsibility framework for safeguarding child-robot companionship by detailing how accountability is distributed and contested, informing concrete design and policy implications to mitigate the emotional harm of robot discontinuation.
>
---
#### [new 017] Self-localization on a 3D map by fusing global and local features from a monocular camera
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦于单目相机下的3D地图自定位任务，针对动态障碍物导致传统CNN方法失效的问题，提出融合CNN与视觉变压器（ViT）的新方法，同时利用局部与全局特征，显著提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.26170v1](http://arxiv.org/pdf/2510.26170v1)**

> **作者:** Satoshi Kikuch; Masaya Kato; Tsuyoshi Tasaki
>
> **摘要:** Self-localization on a 3D map by using an inexpensive monocular camera is required to realize autonomous driving. Self-localization based on a camera often uses a convolutional neural network (CNN) that can extract local features that are calculated by nearby pixels. However, when dynamic obstacles, such as people, are present, CNN does not work well. This study proposes a new method combining CNN with Vision Transformer, which excels at extracting global features that show the relationship of patches on whole image. Experimental results showed that, compared to the state-of-the-art method (SOTA), the accuracy improvement rate in a CG dataset with dynamic obstacles is 1.5 times higher than that without dynamic obstacles. Moreover, the self-localization error of our method is 20.1% smaller than that of SOTA on public datasets. Additionally, our robot using our method can localize itself with 7.51cm error on average, which is more accurate than SOTA.
>
---
#### [new 018] Adaptive Inverse Kinematics Framework for Learning Variable-Length Tool Manipulation in Robotics
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出一种自适应逆运动学框架，解决机器人在使用不同长度工具时的精准操控问题。通过仿真学习动作轨迹并迁移至现实，扩展了逆运动学求解器能力，实现多工具、高精度操作，误差低于1cm，显著提升机器人对工具的通用性与适应性。**

- **链接: [http://arxiv.org/pdf/2510.26551v1](http://arxiv.org/pdf/2510.26551v1)**

> **作者:** Prathamesh Kothavale; Sravani Boddepalli
>
> **备注:** 10 pages, 5 figures. Demonstrates a reinforcement learning framework for adaptive tool manipulation with variable-length extensions
>
> **摘要:** Conventional robots possess a limited understanding of their kinematics and are confined to preprogrammed tasks, hindering their ability to leverage tools efficiently. Driven by the essential components of tool usage - grasping the desired outcome, selecting the most suitable tool, determining optimal tool orientation, and executing precise manipulations - we introduce a pioneering framework. Our novel approach expands the capabilities of the robot's inverse kinematics solver, empowering it to acquire a sequential repertoire of actions using tools of varying lengths. By integrating a simulation-learned action trajectory with the tool, we showcase the practicality of transferring acquired skills from simulation to real-world scenarios through comprehensive experimentation. Remarkably, our extended inverse kinematics solver demonstrates an impressive error rate of less than 1 cm. Furthermore, our trained policy achieves a mean error of 8 cm in simulation. Noteworthy, our model achieves virtually indistinguishable performance when employing two distinct tools of different lengths. This research provides an indication of potential advances in the exploration of all four fundamental aspects of tool usage, enabling robots to master the intricate art of tool manipulation across diverse tasks.
>
---
#### [new 019] A New Type of Axis-Angle Attitude Control Law for Rotational Systems: Synthesis, Analysis, and Experiments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对旋转系统姿态控制任务，解决传统四元数方法在大角度误差下比例控制失效及闭环平衡点不唯一的问题。提出基于轴角信息的新控制律，确保唯一渐近稳定平衡点，并通过李雅普诺夫函数证明稳定性。实验与仿真验证了其在姿态恢复中更快的稳定速度。**

- **链接: [http://arxiv.org/pdf/2510.25985v1](http://arxiv.org/pdf/2510.25985v1)**

> **作者:** Francisco M. F. R. Gonçalves; Ryan M. Bena; Néstor O. Pérez-Arancibia
>
> **备注:** 2025 International Conference on Advanced Robotics (ICAR)
>
> **摘要:** Over the past few decades, continuous quaternion-based attitude control has been proven highly effective for driving rotational systems that can be modeled as rigid bodies, such as satellites and drones. However, methods rooted in this approach do not enforce the existence of a unique closed-loop (CL) equilibrium attitude-error quaternion (AEQ); and, for rotational errors about the attitude-error Euler axis larger than {\pi}rad, their proportional-control effect diminishes as the system state moves away from the stable equilibrium of the CL rotational dynamics. In this paper, we introduce a new type of attitude control law that more effectively leverages the attitude-error Euler axis-angle information to guarantee a unique CL equilibrium AEQ and to provide greater flexibility in the use of proportional-control efforts. Furthermore, using two different control laws as examples-through the construction of a strict Lyapunov function for the CL dynamics-we demonstrate that the resulting unique equilibrium of the CL rotational system can be enforced to be uniformly asymptotically stable. To assess and demonstrate the functionality and performance of the proposed approach, we performed numerical simulations and executed dozens of real-time tumble-recovery maneuvers using a small quadrotor. These simulations and flight tests compellingly demonstrate that the proposed axis-angle-based method achieves superior flight performance-compared with that obtained using a high-performance quaternion-based controller-in terms of stabilization time.
>
---
#### [new 020] A Sliding-Window Filter for Online Continuous-Time Continuum Robot State Estimation
- **分类: cs.RO**

- **简介: 该论文针对连续体机器人状态估计任务，解决传统方法在精度与计算效率间的权衡及实时性不足问题。提出一种用于连续时间状态估计的滑动窗口滤波器（SWF），首次实现基于随机模型的在线连续时间估计，兼具高精度与超实时运行能力。**

- **链接: [http://arxiv.org/pdf/2510.26623v1](http://arxiv.org/pdf/2510.26623v1)**

> **作者:** Spencer Teetaert; Sven Lilge; Jessica Burgner-Kahrs; Timothy D. Barfoot
>
> **备注:** 8 pages, 6 figures. Submitted to IEEE-RAS International Conference on Soft Robotics 2026
>
> **摘要:** Stochastic state estimation methods for continuum robots (CRs) often struggle to balance accuracy and computational efficiency. While several recent works have explored sliding-window formulations for CRs, these methods are limited to simplified, discrete-time approximations and do not provide stochastic representations. In contrast, current stochastic filter methods must run at the speed of measurements, limiting their full potential. Recent works in continuous-time estimation techniques for CRs show a principled approach to addressing this runtime constraint, but are currently restricted to offline operation. In this work, we present a sliding-window filter (SWF) for continuous-time state estimation of CRs that improves upon the accuracy of a filter approach while enabling continuous-time methods to operate online, all while running at faster-than-real-time speeds. This represents the first stochastic SWF specifically designed for CRs, providing a promising direction for future research in this area.
>
---
#### [new 021] Heuristic Adaptation of Potentially Misspecified Domain Support for Likelihood-Free Inference in Stochastic Dynamical Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对机器人学中似然未知推断（LFI）因初始支持集错误导致后验不准确的问题，提出三种启发式支持自适应方法（EDGE、MODE、CENTRE），通过动态调整支持集提升推断精度。在柔性物体操控任务中验证，显著改善参数估计与策略学习的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.26656v1](http://arxiv.org/pdf/2510.26656v1)**

> **作者:** Georgios Kamaras; Craig Innes; Subramanian Ramamoorthy
>
> **摘要:** In robotics, likelihood-free inference (LFI) can provide the domain distribution that adapts a learnt agent in a parametric set of deployment conditions. LFI assumes an arbitrary support for sampling, which remains constant as the initial generic prior is iteratively refined to more descriptive posteriors. However, a potentially misspecified support can lead to suboptimal, yet falsely certain, posteriors. To address this issue, we propose three heuristic LFI variants: EDGE, MODE, and CENTRE. Each interprets the posterior mode shift over inference steps in its own way and, when integrated into an LFI step, adapts the support alongside posterior inference. We first expose the support misspecification issue and evaluate our heuristics using stochastic dynamical benchmarks. We then evaluate the impact of heuristic support adaptation on parameter inference and policy learning for a dynamic deformable linear object (DLO) manipulation task. Inference results in a finer length and stiffness classification for a parametric set of DLOs. When the resulting posteriors are used as domain distributions for sim-based policy learning, they lead to more robust object-centric agent performance.
>
---
#### [new 022] DARTS: A Drone-Based AI-Powered Real-Time Traffic Incident Detection System
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出DARTS系统，解决传统交通事件检测依赖固定设施、响应慢、覆盖有限的问题。通过无人机搭载AI算法，实现高机动性实时监控，融合热成像与轻量化模型，提升检测精度与隐私保护，支持事件快速识别、验证及拥堵追踪，显著提升应急响应效率与交通管理灵活性。**

- **链接: [http://arxiv.org/pdf/2510.26004v1](http://arxiv.org/pdf/2510.26004v1)**

> **作者:** Bai Li; Achilleas Kourtellis; Rong Cao; Joseph Post; Brian Porter; Yu Zhang
>
> **备注:** Preprint version. This manuscript is currently under review at Transportation Research Part C: Emerging Technologies. The PDF corresponds to the version submitted in June 2025. The main findings of this work were recognized with the Best Intelligent Transportation Systems Paper Award at the 2025 TRB Annual Meeting
>
> **摘要:** Rapid and reliable incident detection is critical for reducing crash-related fatalities, injuries, and congestion. However, conventional methods, such as closed-circuit television, dashcam footage, and sensor-based detection, separate detection from verification, suffer from limited flexibility, and require dense infrastructure or high penetration rates, restricting adaptability and scalability to shifting incident hotspots. To overcome these challenges, we developed DARTS, a drone-based, AI-powered real-time traffic incident detection system. DARTS integrates drones' high mobility and aerial perspective for adaptive surveillance, thermal imaging for better low-visibility performance and privacy protection, and a lightweight deep learning framework for real-time vehicle trajectory extraction and incident detection. The system achieved 99% detection accuracy on a self-collected dataset and supports simultaneous online visual verification, severity assessment, and incident-induced congestion propagation monitoring via a web-based interface. In a field test on Interstate 75 in Florida, DARTS detected and verified a rear-end collision 12 minutes earlier than the local transportation management center and monitored incident-induced congestion propagation, suggesting potential to support faster emergency response and enable proactive traffic control to reduce congestion and secondary crash risk. Crucially, DARTS's flexible deployment architecture reduces dependence on frequent physical patrols, indicating potential scalability and cost-effectiveness for use in remote areas and resource-constrained settings. This study presents a promising step toward a more flexible and integrated real-time traffic incident detection system, with significant implications for the operational efficiency and responsiveness of modern transportation management.
>
---
#### [new 023] Morphology-Aware Graph Reinforcement Learning for Tensegrity Robot Locomotion
- **分类: cs.RO**

- **简介: 该论文针对张拉整体机器人运动控制难题，提出一种融合图神经网络的强化学习方法。通过建模机器人结构拓扑，增强策略对部件耦合关系的感知，提升学习效率与鲁棒性，实现从仿真到硬件的零样本迁移，显著改善轨迹精度与稳定性。**

- **链接: [http://arxiv.org/pdf/2510.26067v1](http://arxiv.org/pdf/2510.26067v1)**

> **作者:** Chi Zhang; Mingrui Li; Wenzhe Tong; Xiaonan Huang
>
> **摘要:** Tensegrity robots combine rigid rods and elastic cables, offering high resilience and deployability but posing major challenges for locomotion control due to their underactuated and highly coupled dynamics. This paper introduces a morphology-aware reinforcement learning framework that integrates a graph neural network (GNN) into the Soft Actor-Critic (SAC) algorithm. By representing the robot's physical topology as a graph, the proposed GNN-based policy captures coupling among components, enabling faster and more stable learning than conventional multilayer perceptron (MLP) policies. The method is validated on a physical 3-bar tensegrity robot across three locomotion primitives, including straight-line tracking and bidirectional turning. It shows superior sample efficiency, robustness to noise and stiffness variations, and improved trajectory accuracy. Notably, the learned policies transfer directly from simulation to hardware without fine-tuning, achieving stable real-world locomotion. These results demonstrate the advantages of incorporating structural priors into reinforcement learning for tensegrity robot control.
>
---
#### [new 024] Thor: Towards Human-Level Whole-Body Reactions for Intense Contact-Rich Environments
- **分类: cs.RO**

- **简介: 该论文针对人形机器人在高接触交互环境中的全身稳定性与人类级反应问题，提出Thor框架。通过力适应的躯干倾斜奖励函数与分体式强化学习架构，实现上下身协同控制。部署于Unitree G1，显著提升抓握、拉拽与开门能力，性能优于基线。**

- **链接: [http://arxiv.org/pdf/2510.26280v1](http://arxiv.org/pdf/2510.26280v1)**

> **作者:** Gangyang Li; Qing Shi; Youhao Hu; Jincheng Hu; Zhongyuan Wang; Xinlong Wang; Shaqi Luo
>
> **摘要:** Humanoids hold great potential for service, industrial, and rescue applications, in which robots must sustain whole-body stability while performing intense, contact-rich interactions with the environment. However, enabling humanoids to generate human-like, adaptive responses under such conditions remains a major challenge. To address this, we propose Thor, a humanoid framework for human-level whole-body reactions in contact-rich environments. Based on the robot's force analysis, we design a force-adaptive torso-tilt (FAT2) reward function to encourage humanoids to exhibit human-like responses during force-interaction tasks. To mitigate the high-dimensional challenges of humanoid control, Thor introduces a reinforcement learning architecture that decouples the upper body, waist, and lower body. Each component shares global observations of the whole body and jointly updates its parameters. Finally, we deploy Thor on the Unitree G1, and it substantially outperforms baselines in force-interaction tasks. Specifically, the robot achieves a peak pulling force of 167.7 N (approximately 48% of the G1's body weight) when moving backward and 145.5 N when moving forward, representing improvements of 68.9% and 74.7%, respectively, compared with the best-performing baseline. Moreover, Thor is capable of pulling a loaded rack (130 N) and opening a fire door with one hand (60 N). These results highlight Thor's effectiveness in enhancing humanoid force-interaction capabilities.
>
---
#### [new 025] Debate2Create: Robot Co-design via Large Language Model Debates
- **分类: cs.RO; cs.LG; cs.MA**

- **简介: 该论文提出 Debate2Create 框架，通过大语言模型代理在结构化辩论中协同优化机器人形态与控制策略。针对机器人形态与行为耦合导致的自动化设计难题，引入设计、控制代理与评判面板的迭代辩论机制，利用仿真反馈驱动设计进化，实现高效且多样化的机器人自动生成。**

- **链接: [http://arxiv.org/pdf/2510.25850v1](http://arxiv.org/pdf/2510.25850v1)**

> **作者:** Kevin Qiu; Marek Cygan
>
> **摘要:** Automating the co-design of a robot's morphology and control is a long-standing challenge due to the vast design space and the tight coupling between body and behavior. We introduce Debate2Create (D2C), a framework in which large language model (LLM) agents engage in a structured dialectical debate to jointly optimize a robot's design and its reward function. In each round, a design agent proposes targeted morphological modifications, and a control agent devises a reward function tailored to exploit the new design. A panel of pluralistic judges then evaluates the design-control pair in simulation and provides feedback that guides the next round of debate. Through iterative debates, the agents progressively refine their proposals, producing increasingly effective robot designs. Notably, D2C yields diverse and specialized morphologies despite no explicit diversity objective. On a quadruped locomotion benchmark, D2C discovers designs that travel 73% farther than the default, demonstrating that structured LLM-based debate can serve as a powerful mechanism for emergent robot co-design. Our results suggest that multi-agent debate, when coupled with physics-grounded feedback, is a promising new paradigm for automated robot design.
>
---
#### [new 026] Hybrid Consistency Policy: Decoupling Multi-Modal Diversity and Real-Time Efficiency in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文针对机器人操作中视觉-运动策略学习的多模态性与实时效率矛盾问题，提出混合一致性策略（HCP）。通过短时随机前缀结合单步一致性跳跃，实现快速采样与强多样性兼顾，显著降低延迟，提升实际应用可行性。**

- **链接: [http://arxiv.org/pdf/2510.26670v1](http://arxiv.org/pdf/2510.26670v1)**

> **作者:** Qianyou Zhao; Yuliang Shen; Xuanran Zhai; Ce Hao; Duidi Wu; Jin Qi; Jie Hu; Qiaojun Yu
>
> **摘要:** In visuomotor policy learning, diffusion-based imitation learning has become widely adopted for its ability to capture diverse behaviors. However, approaches built on ordinary and stochastic denoising processes struggle to jointly achieve fast sampling and strong multi-modality. To address these challenges, we propose the Hybrid Consistency Policy (HCP). HCP runs a short stochastic prefix up to an adaptive switch time, and then applies a one-step consistency jump to produce the final action. To align this one-jump generation, HCP performs time-varying consistency distillation that combines a trajectory-consistency objective to keep neighboring predictions coherent and a denoising-matching objective to improve local fidelity. In both simulation and on a real robot, HCP with 25 SDE steps plus one jump approaches the 80-step DDPM teacher in accuracy and mode coverage while significantly reducing latency. These results show that multi-modality does not require slow inference, and a switch time decouples mode retention from speed. It yields a practical accuracy efficiency trade-off for robot policies.
>
---
#### [new 027] Adaptive Trajectory Refinement for Optimization-based Local Planning in Narrow Passages
- **分类: cs.RO**

- **简介: 该论文针对移动机器人在狭窄环境中的轨迹规划问题，提出自适应轨迹优化算法。通过分段保守碰撞检测与基于穿透方向的位姿校正，提升路径安全性与规划效率。实验表明，该方法显著提高成功率并加快规划速度。**

- **链接: [http://arxiv.org/pdf/2510.26142v1](http://arxiv.org/pdf/2510.26142v1)**

> **作者:** Hahjin Lee; Young J. Kim
>
> **摘要:** Trajectory planning for mobile robots in cluttered environments remains a major challenge due to narrow passages, where conventional methods often fail or generate suboptimal paths. To address this issue, we propose the adaptive trajectory refinement algorithm, which consists of two main stages. First, to ensure safety at the path-segment level, a segment-wise conservative collision test is applied, where risk-prone trajectory path segments are recursively subdivided until collision risks are eliminated. Second, to guarantee pose-level safety, pose correction based on penetration direction and line search is applied, ensuring that each pose in the trajectory is collision-free and maximally clear from obstacles. Simulation results demonstrate that the proposed method achieves up to 1.69x higher success rates and up to 3.79x faster planning times than state-of-the-art approaches. Furthermore, real-world experiments confirm that the robot can safely pass through narrow passages while maintaining rapid planning performance.
>
---
#### [new 028] Embodied Intelligence for Advanced Bioinspired Microrobotics: Examples and Insights
- **分类: cs.RO**

- **简介: 该论文探讨具身智能（EI）在先进仿生微机器人设计中的应用，聚焦于结构与功能协同设计。旨在解决传统机器人中感知、计算、执行分离导致的效率问题。通过多个微机器人平台实验证明，通过物理结构与环境交互实现智能行为，可提升微尺度机器人的自主性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.26132v1](http://arxiv.org/pdf/2510.26132v1)**

> **作者:** Nestor O. Perez-Arancibia
>
> **备注:** 8 pages, 7 figures, accepted to ICAR 2025
>
> **摘要:** The term embodied intelligence (EI) conveys the notion that body morphology, material properties, interaction with the environment, and control strategies can be purposefully integrated into the process of robotic design to generate intelligent behavior; in particular, locomotion and navigation. In this paper, we discuss EI as a design principle for advanced microrobotics, with a particular focus on co-design -- the simultaneous and interdependent development of physical structure and behavioral function. To illustrate the contrast between EI-inspired systems and traditional architectures that decouple sensing, computation, and actuation, we present and discuss a collection of robots developed by the author and his team at the Autonomous Microrobotic Systems Laboratory (AMSL). These robots exhibit intelligent behavior that emerges from their structural dynamics and the physical interaction between their components and with the environment. Platforms such as the Bee++, RoBeetle, SMALLBug, SMARTI, WaterStrider, VLEIBot+, and FRISSHBot exemplify how feedback loops, decision logics, sensing mechanisms, and smart actuation strategies can be embedded into the physical properties of the robotic system itself. Along these lines, we contend that co-design is not only a method for empirical optimization under constraints, but also an enabler of EI, offering a scalable and robust alternative to classical control for robotics at the mm-to-cm-scale.
>
---
#### [new 029] Curvature-Aware Calibration of Tactile Sensors for Accurate Force Estimation on Non-Planar Surfaces
- **分类: cs.RO**

- **简介: 该论文针对柔性触觉传感器在曲面应用中精度下降的问题，提出曲率感知校准方法。通过神经网络从无负载输出预测局部曲率，实现对一维曲面的精确受力估计。实验验证了该方法在不同曲率物体上均保持高精度与一致性，显著优于传统平面校准。**

- **链接: [http://arxiv.org/pdf/2510.25965v1](http://arxiv.org/pdf/2510.25965v1)**

> **作者:** Luoyan Zhong; Heather Jin Hee Kim; Dylan P. Losey; Cara M. Nunez
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Flexible tactile sensors are increasingly used in real-world applications such as robotic grippers, prosthetic hands, wearable gloves, and assistive devices, where they need to conform to curved and irregular surfaces. However, most existing tactile sensors are calibrated only on flat substrates, and their accuracy and consistency degrade once mounted on curved geometries. This limitation restricts their reliability in practical use. To address this challenge, we develop a calibration model for a widely used resistive tactile sensor design that enables accurate force estimation on one-dimensional curved surfaces. We then train a neural network (a multilayer perceptron) to predict local curvature from baseline sensor outputs recorded under no applied load, achieving an R2 score of 0.91. The proposed approach is validated on five daily objects with varying curvatures under forces from 2 N to 8 N. Results show that the curvature-aware calibration maintains consistent force accuracy across all surfaces, while flat-surface calibration underestimates force as curvature increases. Our results demonstrate that curvature-aware modeling improves the accuracy, consistency, and reliability of flexible tactile sensors, enabling dependable performance across real-world applications.
>
---
#### [new 030] Proxemics and Permeability of the Pedestrian Group
- **分类: physics.soc-ph; cs.GT; cs.MA; cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究行人团体的邻近关系与渗透性，属于人群行为建模任务。针对群体空间结构与个体互动问题，基于自然数据分析发现除公共区外存在三个潜在区域，个体仅短暂低频靠近群体。**

- **链接: [http://arxiv.org/pdf/2510.26571v1](http://arxiv.org/pdf/2510.26571v1)**

> **作者:** Saleh Albeaik; Faisal Alsallum; Mohamad Alrished
>
> **摘要:** People tend to walk in groups, and interactions with those groups have a significant impact on crowd behavior and pedestrian traffic dynamics. Social norms can be seen as unwritten rules regulating people interactions in social settings. This article studies people interactions with groups and the emergence of group proxemics. Group zones, zone occupancy counts and people clearance from the group are studied using naturalistic data. Analysis indicate potential presence of three different zones in addition to the public zone. People tend to remain in the public zone and only progressively get closer to groups, and those closer approaches happen in a low frequency and for brief periods of time.
>
---
#### [new 031] Efficient Collision-Avoidance Constraints for Ellipsoidal Obstacles in Optimal Control: Application to Path-Following MPC and UAVs
- **分类: eess.SY; cs.RO; cs.SY; 93-XX**

- **简介: 该论文针对无人机三维路径跟踪中的椭球障碍物避障问题，提出一种高效、可微的碰撞检测约束与两阶段优化方法，实现模型预测控制（MPC）下的实时避障。首次在真实无人机（Crazyflie）上验证了该MPC避障框架的有效性。**

- **链接: [http://arxiv.org/pdf/2510.26531v1](http://arxiv.org/pdf/2510.26531v1)**

> **作者:** David Leprich; Mario Rosenfelder; Markus Herrmann-Wicklmayr; Kathrin Flaßkamp; Peter Eberhard; Henrik Ebel
>
> **摘要:** This article proposes a modular optimal control framework for local three-dimensional ellipsoidal obstacle avoidance, exemplarily applied to model predictive path-following control. Static as well as moving obstacles are considered. Central to the approach is a computationally efficient and continuously differentiable condition for detecting collisions with ellipsoidal obstacles. A novel two-stage optimization approach mitigates numerical issues arising from the structure of the resulting optimal control problem. The effectiveness of the approach is demonstrated through simulations and real-world experiments with the Crazyflie quadrotor. This represents the first hardware demonstration of an MPC controller of this kind for UAVs in a three-dimensional task.
>
---
#### [new 032] Large Language Model-assisted Autonomous Vehicle Recovery from Immobilization
- **分类: cs.AI; cs.RO**

- **简介: 该论文针对自动驾驶车辆在复杂场景中易陷入停滞的问题，提出StuckSolver框架。基于大语言模型实现自主推理或乘客引导下的恢复决策，无需修改原有系统架构，通过感知数据判断环境并生成恢复指令，显著提升车辆脱困能力。**

- **链接: [http://arxiv.org/pdf/2510.26023v1](http://arxiv.org/pdf/2510.26023v1)**

> **作者:** Zhipeng Bao; Qianwen Li
>
> **备注:** 8 pages
>
> **摘要:** Despite significant advancements in recent decades, autonomous vehicles (AVs) continue to face challenges in navigating certain traffic scenarios where human drivers excel. In such situations, AVs often become immobilized, disrupting overall traffic flow. Current recovery solutions, such as remote intervention (which is costly and inefficient) and manual takeover (which excludes non-drivers and limits AV accessibility), are inadequate. This paper introduces StuckSolver, a novel Large Language Model (LLM) driven recovery framework that enables AVs to resolve immobilization scenarios through self-reasoning and/or passenger-guided decision-making. StuckSolver is designed as a plug-in add-on module that operates on top of the AV's existing perception-planning-control stack, requiring no modification to its internal architecture. Instead, it interfaces with standard sensor data streams to detect immobilization states, interpret environmental context, and generate high-level recovery commands that can be executed by the AV's native planner. We evaluate StuckSolver on the Bench2Drive benchmark and in custom-designed uncertainty scenarios. Results show that StuckSolver achieves near-state-of-the-art performance through autonomous self-reasoning alone and exhibits further improvements when passenger guidance is incorporated.
>
---
#### [new 033] Exploring Object-Aware Attention Guided Frame Association for RGB-D SLAM
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对RGB-D SLAM中帧关联性能不足的问题，提出利用网络梯度生成的层间注意力信息，增强CNN特征对语义物体的空间感知能力。通过融合注意力引导的特征表示，提升了大环境下的帧关联准确率，推动了视觉任务中语义理解与定位的结合。**

- **链接: [http://arxiv.org/pdf/2510.26131v1](http://arxiv.org/pdf/2510.26131v1)**

> **作者:** Ali Caglayan; Nevrez Imamoglu; Oguzhan Guclu; Ali Osman Serhatoglu; Ahmet Burak Can; Ryosuke Nakamura
>
> **备注:** double-column 5 pages, 3 figures
>
> **摘要:** Attention models have recently emerged as a powerful approach, demonstrating significant progress in various fields. Visualization techniques, such as class activation mapping, provide visual insights into the reasoning of convolutional neural networks (CNNs). Using network gradients, it is possible to identify regions where the network pays attention during image recognition tasks. Furthermore, these gradients can be combined with CNN features to localize more generalizable, task-specific attentive (salient) regions within scenes. However, explicit use of this gradient-based attention information integrated directly into CNN representations for semantic object understanding remains limited. Such integration is particularly beneficial for visual tasks like simultaneous localization and mapping (SLAM), where CNN representations enriched with spatially attentive object locations can enhance performance. In this work, we propose utilizing task-specific network attention for RGB-D indoor SLAM. Specifically, we integrate layer-wise attention information derived from network gradients with CNN feature representations to improve frame association performance. Experimental results indicate improved performance compared to baseline methods, particularly for large environments.
>
---
#### [new 034] CorVS: Person Identification via Video Trajectory-Sensor Correspondence in a Real-World Warehouse
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文提出CorVS，一种基于视觉轨迹与传感器数据对应关系的人员识别方法。针对工业仓库中仅靠视觉识别人员不现实的问题，利用深度学习预测轨迹与传感器数据的匹配概率与可靠性，实现高鲁棒性人员识别。通过真实仓库数据验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2510.26369v1](http://arxiv.org/pdf/2510.26369v1)**

> **作者:** Kazuma Kano; Yuki Mori; Shin Katayama; Kenta Urano; Takuro Yonezawa; Nobuo Kawaguchi
>
> **备注:** 7 pages, 3 figures, accepted to IPIN 2025
>
> **摘要:** Worker location data is key to higher productivity in industrial sites. Cameras are a promising tool for localization in logistics warehouses since they also offer valuable environmental contexts such as package status. However, identifying individuals with only visual data is often impractical. Accordingly, several prior studies identified people in videos by comparing their trajectories and wearable sensor measurements. While this approach has advantages such as independence from appearance, the existing methods may break down under real-world conditions. To overcome this challenge, we propose CorVS, a novel data-driven person identification method based on correspondence between visual tracking trajectories and sensor measurements. Firstly, our deep learning model predicts correspondence probabilities and reliabilities for every pair of a trajectory and sensor measurements. Secondly, our algorithm matches the trajectories and sensor measurements over time using the predicted probabilities and reliabilities. We developed a dataset with actual warehouse operations and demonstrated the method's effectiveness for real-world applications.
>
---
#### [new 035] WaveVerif: Acoustic Side-Channel based Verification of Robotic Workflows
- **分类: cs.CR; cs.AI; cs.RO**

- **简介: 该论文提出WaveVerif框架，利用声学侧信道分析实现对机器人工作流程的实时验证。针对机器人执行命令是否正确的问题，通过机器学习分析其运动产生的声音信号，验证动作一致性。在不同条件下，单个动作识别准确率超80%，可有效识别拾取、包装等流程，无需硬件改动，实现低成本被动监控。**

- **链接: [http://arxiv.org/pdf/2510.25960v1](http://arxiv.org/pdf/2510.25960v1)**

> **作者:** Zeynep Yasemin Erdogan; Shishir Nagaraja; Chuadhry Mujeeb Ahmed; Ryan Shah
>
> **备注:** 11 pages, 3 figures, Corresponding Author: Prof. Shishir Nagaraja (shishir.nagaraja@newcastle.ac.uk)
>
> **摘要:** In this paper, we present a framework that uses acoustic side- channel analysis (ASCA) to monitor and verify whether a robot correctly executes its intended commands. We develop and evaluate a machine-learning-based workflow verification system that uses acoustic emissions generated by robotic movements. The system can determine whether real-time behavior is consistent with expected commands. The evaluation takes into account movement speed, direction, and microphone distance. The results show that individual robot movements can be validated with over 80% accuracy under baseline conditions using four different classifiers: Support Vector Machine (SVM), Deep Neural Network (DNN), Recurrent Neural Network (RNN), and Convolutional Neural Network (CNN). Additionally, workflows such as pick-and-place and packing could be identified with similarly high confidence. Our findings demonstrate that acoustic signals can support real-time, low-cost, passive verification in sensitive robotic environments without requiring hardware modifications.
>
---
#### [new 036] Spiking Patches: Asynchronous, Sparse, and Efficient Tokens for Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对事件相机数据的高效处理，提出异步稀疏的Spiking Patches令牌化方法。旨在保留事件相机固有的异步与稀疏特性，同时提升计算效率。实验表明，该方法在手势识别和目标检测任务中显著加速（最高10.4倍），且精度不降反升，为事件视觉提供了新范式。**

- **链接: [http://arxiv.org/pdf/2510.26614v1](http://arxiv.org/pdf/2510.26614v1)**

> **作者:** Christoffer Koo Øhrstrøm; Ronja Güldenring; Lazaros Nalpantidis
>
> **摘要:** We propose tokenization of events and present a tokenizer, Spiking Patches, specifically designed for event cameras. Given a stream of asynchronous and spatially sparse events, our goal is to discover an event representation that preserves these properties. Prior works have represented events as frames or as voxels. However, while these representations yield high accuracy, both frames and voxels are synchronous and decrease the spatial sparsity. Spiking Patches gives the means to preserve the unique properties of event cameras and we show in our experiments that this comes without sacrificing accuracy. We evaluate our tokenizer using a GNN, PCN, and a Transformer on gesture recognition and object detection. Tokens from Spiking Patches yield inference times that are up to 3.4x faster than voxel-based tokens and up to 10.4x faster than frames. We achieve this while matching their accuracy and even surpassing in some cases with absolute improvements up to 3.8 for gesture recognition and up to 1.4 for object detection. Thus, tokenization constitutes a novel direction in event-based vision and marks a step towards methods that preserve the properties of event cameras.
>
---
#### [new 037] On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究视觉-语言-动作（VLA）模型在多模态扰动下的鲁棒性问题，针对真实场景中动作、指令、环境、观测等多模态干扰，提出RobustVLA框架。通过对抗优化与多臂老虎机策略，提升模型对输入输出扰动的鲁棒性，在多种扰动下显著优于基线，尤其在真实机器人上表现优异。**

- **链接: [http://arxiv.org/pdf/2510.00037v3](http://arxiv.org/pdf/2510.00037v3)**

> **作者:** Jianing Guo; Zhenhong Wu; Chang Tu; Yiyao Ma; Xiangqi Kong; Zhiqian Liu; Jiaming Ji; Shuning Zhang; Yuanpei Chen; Kai Chen; Qi Dou; Yaodong Yang; Xianglong Liu; Huijie Zhao; Weifeng Lv; Simin Li
>
> **摘要:** In Vision-Language-Action (VLA) models, robustness to real-world perturbations is critical for deployment. Existing methods target simple visual disturbances, overlooking the broader multi-modal perturbations that arise in actions, instructions, environments, and observations. Here, we first evaluate the robustness of mainstream VLAs under 17 perturbations across four modalities. We find (1) actions as the most fragile modality, (2) Existing visual-robust VLA do not gain robustness in other modality, and (3) pi0 demonstrates superior robustness with a diffusion-based action head. To build multi-modal robust VLAs, we propose RobustVLA against perturbations in VLA inputs and outputs. For output robustness, we perform offline robust optimization against worst-case action noise that maximizes mismatch in flow matching objective. This can be seen as adversarial training, label smoothing, and outlier penalization. For input robustness, we enforce consistent actions across input variations that preserve task semantics. To account for multiple perturbations, we formulate robustness as a multi-armed bandit problem and apply an upper confidence bound algorithm to automatically identify the most harmful noise. Experiments on LIBERO demonstrate our RobustVLA delivers absolute gains over baselines of 12.6% on the pi0 backbone and 10.4% on the OpenVLA backbone across all 17 perturbations, achieving 50.6x faster inference than existing visual-robust VLAs, and a 10.4% gain under mixed perturbations. Our RobustVLA is particularly effective on real-world FR5 robot with limited demonstrations, showing absolute gains by 65.6% under perturbations of four modalities.
>
---
#### [new 038] Enhancing Underwater Object Detection through Spatio-Temporal Analysis and Spatial Attention Networks
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文针对水下目标检测任务，解决动态环境中物体识别精度低的问题。通过引入时空建模与空间注意力机制，改进YOLOv5，提出T-YOLOv5及融合CBAM的版本，显著提升复杂场景下的检测准确率，尤其在运动、遮挡条件下表现优异。**

- **链接: [http://arxiv.org/pdf/2510.25797v1](http://arxiv.org/pdf/2510.25797v1)**

> **作者:** Sai Likhith Karri; Ansh Saxena
>
> **摘要:** This study examines the effectiveness of spatio-temporal modeling and the integration of spatial attention mechanisms in deep learning models for underwater object detection. Specifically, in the first phase, the performance of temporal-enhanced YOLOv5 variant T-YOLOv5 is evaluated, in comparison with the standard YOLOv5. For the second phase, an augmented version of T-YOLOv5 is developed, through the addition of a Convolutional Block Attention Module (CBAM). By examining the effectiveness of the already pre-existing YOLOv5 and T-YOLOv5 models and of the newly developed T-YOLOv5 with CBAM. With CBAM, the research highlights how temporal modeling improves detection accuracy in dynamic marine environments, particularly under conditions of sudden movements, partial occlusions, and gradual motion. The testing results showed that YOLOv5 achieved a mAP@50-95 of 0.563, while T-YOLOv5 and T-YOLOv5 with CBAM outperformed with mAP@50-95 scores of 0.813 and 0.811, respectively, highlighting their superior accuracy and generalization in detecting complex objects. The findings demonstrate that T-YOLOv5 significantly enhances detection reliability compared to the standard model, while T-YOLOv5 with CBAM further improves performance in challenging scenarios, although there is a loss of accuracy when it comes to simpler scenarios.
>
---
#### [new 039] BikeScenes: Online LiDAR Semantic Segmentation for Bicycles
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自行车骑行者安全，提出基于LiDAR的在线语义分割方法BikeScenes。为解决汽车感知技术向自行车场景迁移的域差距问题，构建了包含3021帧标注数据的BikeScenes-lidarseg数据集，并验证了领域特定微调的有效性，显著提升分割精度。**

- **链接: [http://arxiv.org/pdf/2510.25901v1](http://arxiv.org/pdf/2510.25901v1)**

> **作者:** Denniz Goren; Holger Caesar
>
> **摘要:** The vulnerability of cyclists, exacerbated by the rising popularity of faster e-bikes, motivates adapting automotive perception technologies for bicycle safety. We use our multi-sensor 'SenseBike' research platform to develop and evaluate a 3D LiDAR segmentation approach tailored to bicycles. To bridge the automotive-to-bicycle domain gap, we introduce the novel BikeScenes-lidarseg Dataset, comprising 3021 consecutive LiDAR scans around the university campus of the TU Delft, semantically annotated for 29 dynamic and static classes. By evaluating model performance, we demonstrate that fine-tuning on our BikeScenes dataset achieves a mean Intersection-over-Union (mIoU) of 63.6%, significantly outperforming the 13.8% obtained with SemanticKITTI pre-training alone. This result underscores the necessity and effectiveness of domain-specific training. We highlight key challenges specific to bicycle-mounted, hardware-constrained perception systems and contribute the BikeScenes dataset as a resource for advancing research in cyclist-centric LiDAR segmentation.
>
---
## 更新

#### [replaced 001] DiffVLA++: Bridging Cognitive Reasoning and End-to-End Driving through Metric-Guided Alignment
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.17148v3](http://arxiv.org/pdf/2510.17148v3)**

> **作者:** Yu Gao; Anqing Jiang; Yiru Wang; Wang Jijun; Hao Jiang; Zhigang Sun; Heng Yuwen; Wang Shuo; Hao Zhao; Sun Hao
>
> **摘要:** Conventional end-to-end (E2E) driving models are effective at generating physically plausible trajectories, but often fail to generalize to long-tail scenarios due to the lack of essential world knowledge to understand and reason about surrounding environments. In contrast, Vision-Language-Action (VLA) models leverage world knowledge to handle challenging cases, but their limited 3D reasoning capability can lead to physically infeasible actions. In this work we introduce DiffVLA++, an enhanced autonomous driving framework that explicitly bridges cognitive reasoning and E2E planning through metric-guided alignment. First, we build a VLA module directly generating semantically grounded driving trajectories. Second, we design an E2E module with a dense trajectory vocabulary that ensures physical feasibility. Third, and most critically, we introduce a metric-guided trajectory scorer that guides and aligns the outputs of the VLA and E2E modules, thereby integrating their complementary strengths. The experiment on the ICCV 2025 Autonomous Grand Challenge leaderboard shows that DiffVLA++ achieves EPDMS of 49.12.
>
---
#### [replaced 002] SAFE: Multitask Failure Detection for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.09937v2](http://arxiv.org/pdf/2506.09937v2)**

> **作者:** Qiao Gu; Yuanliang Ju; Shengxiang Sun; Igor Gilitschenski; Haruki Nishimura; Masha Itkina; Florian Shkurti
>
> **备注:** NeurIPS 2025 camera ready. Project Page: https://vla-safe.github.io/
>
> **摘要:** While vision-language-action models (VLAs) have shown promising robotic behaviors across a diverse set of manipulation tasks, they achieve limited success rates when deployed on novel tasks out of the box. To allow these policies to safely interact with their environments, we need a failure detector that gives a timely alert such that the robot can stop, backtrack, or ask for help. However, existing failure detectors are trained and tested only on one or a few specific tasks, while generalist VLAs require the detector to generalize and detect failures also in unseen tasks and novel environments. In this paper, we introduce the multitask failure detection problem and propose SAFE, a failure detector for generalist robot policies such as VLAs. We analyze the VLA feature space and find that VLAs have sufficient high-level knowledge about task success and failure, which is generic across different tasks. Based on this insight, we design SAFE to learn from VLA internal features and predict a single scalar indicating the likelihood of task failure. SAFE is trained on both successful and failed rollouts and is evaluated on unseen tasks. SAFE is compatible with different policy architectures. We test it on OpenVLA, $\pi_0$, and $\pi_0$-FAST in both simulated and real-world environments extensively. We compare SAFE with diverse baselines and show that SAFE achieves state-of-the-art failure detection performance and the best trade-off between accuracy and detection time using conformal prediction. More qualitative results and code can be found at the project webpage: https://vla-safe.github.io/
>
---
#### [replaced 003] STATE-NAV: Stability-Aware Traversability Estimation for Bipedal Navigation on Rough Terrain
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01046v3](http://arxiv.org/pdf/2506.01046v3)**

> **作者:** Ziwon Yoon; Lawrence Y. Zhu; Jingxi Lu; Lu Gan; Ye Zhao
>
> **摘要:** Bipedal robots have advantages in maneuvering human-centered environments, but face greater failure risk compared to other stable mobile plarforms such as wheeled or quadrupedal robots. While learning-based traversability has been widely studied for these platforms, bipedal traversability has instead relied on manually designed rules with limited consideration of locomotion stability on rough terrain. In this work, we present the first learning-based traversability estimation and risk-sensitive navigation framework for bipedal robots operating in diverse, uneven environments. TravFormer, a transformer-based neural network, is trained to predict bipedal instability with uncertainty, enabling risk-aware and adaptive planning. Based on the network, we define traversability as stability-aware command velocity-the fastest command velocity that keeps instability below a user-defined limit. This velocity-based traversability is integrated into a hierarchical planner that combines traversability-informed Rapid Random Tree Star (TravRRT*) for time-efficient planning and Model Predictive Control (MPC) for safe execution. We validate our method in MuJoCo simulation and the real world, demonstrating improved navigation performance, with enhanced robustness and time efficiency across varying terrains compared to existing methods.
>
---
#### [replaced 004] Learning to Insert for Constructive Neural Vehicle Routing Solver
- **分类: cs.LG; cs.AI; cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2505.13904v3](http://arxiv.org/pdf/2505.13904v3)**

> **作者:** Fu Luo; Xi Lin; Mengyuan Zhong; Fei Liu; Zhenkun Wang; Jianyong Sun; Qingfu Zhang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Neural Combinatorial Optimisation (NCO) is a promising learning-based approach for solving Vehicle Routing Problems (VRPs) without extensive manual design. While existing constructive NCO methods typically follow an appending-based paradigm that sequentially adds unvisited nodes to partial solutions, this rigid approach often leads to suboptimal results. To overcome this limitation, we explore the idea of insertion-based paradigm and propose Learning to Construct with Insertion-based Paradigm (L2C-Insert), a novel learning-based method for constructive NCO. Unlike traditional approaches, L2C-Insert builds solutions by strategically inserting unvisited nodes at any valid position in the current partial solution, which can significantly enhance the flexibility and solution quality. The proposed framework introduces three key components: a novel model architecture for precise insertion position prediction, an efficient training scheme for model optimization, and an advanced inference technique that fully exploits the insertion paradigm's flexibility. Extensive experiments on both synthetic and real-world instances of the Travelling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that L2C-Insert consistently achieves superior performance across various problem sizes.
>
---
#### [replaced 005] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.20685v2](http://arxiv.org/pdf/2510.20685v2)**

> **作者:** Ming-Ming Yu; Fei Zhu; Wenzhuo Liu; Yirong Yang; Qunbo Wang; Wenjun Wu; Jing Liu
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Embodied agents are expected to perform object navigation in dynamic, open-world environments. However, existing approaches typically rely on static trajectories and a fixed set of object categories during training, overlooking the real-world requirement for continual adaptation to evolving scenarios. To facilitate related studies, we introduce the continual object navigation benchmark, which requires agents to acquire navigation skills for new object categories while avoiding catastrophic forgetting of previously learned knowledge. To tackle this challenge, we propose C-Nav, a continual visual navigation framework that integrates two key innovations: (1) A dual-path anti-forgetting mechanism, which comprises feature distillation that aligns multi-modal inputs into a consistent representation space to ensure representation consistency, and feature replay that retains temporal features within the action decoder to ensure policy consistency. (2) An adaptive sampling strategy that selects diverse and informative experiences, thereby reducing redundancy and minimizing memory overhead. Extensive experiments across multiple model architectures demonstrate that C-Nav consistently outperforms existing approaches, achieving superior performance even compared to baselines with full trajectory retention, while significantly lowering memory requirements. The code will be publicly available at https://bigtree765.github.io/C-Nav-project.
>
---
#### [replaced 006] Towards Predicting Any Human Trajectory In Context
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.00871v2](http://arxiv.org/pdf/2506.00871v2)**

> **作者:** Ryo Fujii; Hideo Saito; Ryo Hachiuma
>
> **备注:** NeurIPS 2025
>
> **摘要:** Predicting accurate future trajectories of pedestrians is essential for autonomous systems but remains a challenging task due to the need for adaptability in different environments and domains. A common approach involves collecting scenario-specific data and performing fine-tuning via backpropagation. However, the need to fine-tune for each new scenario is often impractical for deployment on edge devices. To address this challenge, we introduce \paper, an In-Context Learning (ICL) framework for pedestrian trajectory prediction that enables adaptation without fine-tuning on the scenario-specific data at inference time without requiring weight updates. We propose a spatio-temporal similarity-based example selection (STES) method that selects relevant examples from previously observed trajectories within the same scene by identifying similar motion patterns at corresponding locations. To further refine this selection, we introduce prediction-guided example selection (PG-ES), which selects examples based on both the past trajectory and the predicted future trajectory, rather than relying solely on the past trajectory. This approach allows the model to account for long-term dynamics when selecting examples. Finally, instead of relying on small real-world datasets with limited scenario diversity, we train our model on a large-scale synthetic dataset to enhance its prediction ability by leveraging in-context examples. Extensive experiments demonstrate that TrajICL achieves remarkable adaptation across both in-domain and cross-domain scenarios, outperforming even fine-tuned approaches across multiple public benchmarks. Project Page: https://fujiry0.github.io/TrajICL-project-page/.
>
---
#### [replaced 007] A Constrained Saddle Search Approach for Constructing Singular and Flexible Bar Frameworks
- **分类: cs.RO; math-ph; math.MP; math.OC**

- **链接: [http://arxiv.org/pdf/2503.14807v2](http://arxiv.org/pdf/2503.14807v2)**

> **作者:** Xuenan Li; Mihnea Leonte; Christian D. Santangelo; Miranda Holmes-Cerfon
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Singularity analysis is essential in robot kinematics, as singular configurations cause loss of control and kinematic indeterminacy. This paper models singularities in bar frameworks as saddle points on constrained manifolds. Given an under-constrained, non-singular bar framework, by allowing one edge to vary its length while fixing lengths of others, we define the squared length of the free edge as an energy functional and show that its local saddle points correspond to singular and flexible frameworks. Using our constrained saddle search approach, we identify previously unknown singular and flexible bar frameworks, providing new insights into singular robotics design and analysis.
>
---
#### [replaced 008] SIMS: Surgeon-Intention-driven Motion Scaling for Efficient and Precise Teleoperation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.01216v2](http://arxiv.org/pdf/2503.01216v2)**

> **作者:** Jeonghyeon Yoon; Sanghyeok Park; Hyojae Park; Cholin Kim; Michael C. Yip; Minho Hwang
>
> **摘要:** Telerobotic surgery often relies on a fixed motion scaling factor (MSF) to map the surgeon's hand motions to robotic instruments, but this introduces a trade-off between precision and efficiency: small MSF enables delicate manipulation but slows large movements, while large MSF accelerates transfer at the cost of accuracy. We propose a Surgeon-Intention driven Motion Scaling (SIMS) system, which dynamically adjusts MSF in real time based solely on kinematic cues. SIMS extracts linear speed, tool motion alignment, and dual-arm coordination features to classify motion intent via fuzzy C-means clustering and applies confidence-based updates independently for both arms. In a user study (n=10, three surgical training tasks) conducted on the da Vinci Research Kit, SIMS significantly reduced collisions (mean reduction of 83%), lowered mental and physical workload, and maintained task completion efficiency compared to fixed MSF. These findings demonstrate that SIMS is a practical and lightweight approach for safer, more efficient, and user-adaptive telesurgical control.
>
---
#### [replaced 009] 3D Equivariant Visuomotor Policy Learning via Spherical Projection
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16969v3](http://arxiv.org/pdf/2505.16969v3)**

> **作者:** Boce Hu; Dian Wang; David Klee; Heng Tian; Xupeng Zhu; Haojie Huang; Robert Platt; Robin Walters
>
> **摘要:** Equivariant models have recently been shown to improve the data efficiency of diffusion policy by a significant margin. However, prior work that explored this direction focused primarily on point cloud inputs generated by multiple cameras fixed in the workspace. This type of point cloud input is not compatible with the now-common setting where the primary input modality is an eye-in-hand RGB camera like a GoPro. This paper closes this gap by incorporating into the diffusion policy model a process that projects features from the 2D RGB camera image onto a sphere. This enables us to reason about symmetries in $\mathrm{SO}(3)$ without explicitly reconstructing a point cloud. We perform extensive experiments in both simulation and the real world that demonstrate that our method consistently outperforms strong baselines in terms of both performance and sample efficiency. Our work, Image-to-Sphere Policy ($\textbf{ISP}$), is the first $\mathrm{SO}(3)$-equivariant policy learning framework for robotic manipulation that works using only monocular RGB inputs.
>
---
#### [replaced 010] Falconry-like palm landing by a flapping-wing drone based on the human gesture interaction and distance-aware flight planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17144v2](http://arxiv.org/pdf/2507.17144v2)**

> **作者:** Kazuki Numazato; Keiichiro Kan; Masaki Kitagawa; Yunong Li; Johannes Kubel; Moju Zhao
>
> **备注:** 8 pages, 14 figures
>
> **摘要:** Flapping-wing drones have attracted significant attention due to their biomimetic flight. They are considered more human-friendly due to their characteristics such as low noise and flexible wings, making them suitable for human-drone interactions. However, few studies have explored the practical interaction between humans and flapping-wing drones. On establishing a physical interaction system with flapping-wing drones, we can acquire inspirations from falconers who guide birds of prey to land on their arms. This interaction interprets the human body as a dynamic landing platform, which can be utilized in various scenarios such as crowded or spatially constrained environments. Thus, in this study, we propose a falconry-like interaction system in which a flapping-wing drone performs a palm landing motion on a human hand. To achieve a safe approach toward humans, we design a trajectory planning method that considers both physical and psychological factors of the human safety such as the drone's velocity and distance from the user. We use a commercial flapping platform with our implemented motion planning and conduct experiments to evaluate the palm landing performance and safety. The results demonstrate that our approach enables safe and smooth hand landing interactions. To the best of our knowledge, it is the first time to achieve a contact-based interaction between flapping-wing drones and humans.
>
---
#### [replaced 011] SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.25358v3](http://arxiv.org/pdf/2509.25358v3)**

> **作者:** Qianzhong Chen; Justin Yu; Mac Schwager; Pieter Abbeel; Yide Shentu; Philipp Wu
>
> **摘要:** Large-scale robot learning has recently shown promise for enabling robots to perform complex tasks by integrating perception, control, and language understanding. Yet, it struggles with long-horizon, contact-rich manipulation such as deformable object handling, where demonstration quality is inconsistent. Reward modeling offers a natural solution: by providing grounded progress signals, it transforms noisy demonstrations into stable supervision that generalizes across diverse trajectories. We introduce a stage-aware, video-based reward modeling framework that jointly predicts high-level task stages and fine-grained progress. Reward labels are automatically derived from natural language subtask annotations, ensuring consistent progress estimation across variable-length demonstrations. This design overcomes frame-index labeling, which fails in variable-duration tasks like folding a T-shirt. Our reward model demonstrates robustness to variability, generalization to out-of-distribution settings, and strong utility for policy training. Building on it, we propose Reward-Aligned Behavior Cloning (RA-BC), which filters high-quality data and reweights samples by reward. Experiments show the reward model alone outperforms baselines on validation and real robot rollouts. Integrated into RA-BC, our approach achieves 83% success on folding T-shirts from the flattened state and 67% from the crumpled state -- far surpassing vanilla behavior cloning, which attains only 8% and 0% success. Overall, our results highlight reward modeling as a key enabler for scalable, annotation-efficient, and robust imitation learning in long-horizon manipulation.
>
---
#### [replaced 012] CronusVLA: Towards Efficient and Robust Manipulation via Multi-Frame Vision-Language-Action Modeling
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.19816v2](http://arxiv.org/pdf/2506.19816v2)**

> **作者:** Hao Li; Shuai Yang; Yilun Chen; Xinyi Chen; Xiaoda Yang; Yang Tian; Hanqing Wang; Tai Wang; Dahua Lin; Feng Zhao; Jiangmiao Pang
>
> **备注:** 39 pages, 24 figures
>
> **摘要:** Recent vision-language-action (VLA) models built on pretrained vision-language models (VLMs) have demonstrated strong performance in robotic manipulation. However, these models remain constrained by the single-frame image paradigm and fail to fully leverage the temporal information offered by multi-frame histories, as directly feeding multiple frames into VLM backbones incurs substantial computational overhead and inference latency. We propose CronusVLA, a unified framework that extends single-frame VLA models to the multi-frame paradigm. CronusVLA follows a two-stage process: (1) Single-frame pretraining on large-scale embodied datasets with autoregressive prediction of action tokens, establishing an effective embodied vision-language foundation; (2) Multi-frame post-training, which adapts the prediction of the vision-language backbone from discrete tokens to learnable features, and aggregates historical information via feature chunking. CronusVLA effectively addresses the existing challenges of multi-frame modeling while enhancing performance and observational robustness. To evaluate the robustness under temporal and spatial disturbances, we introduce SimplerEnv-OR, a novel benchmark featuring 24 types of observational disturbances and 120 severity levels. Experiments across three embodiments in simulated and real-world environments demonstrate that CronusVLA achieves leading performance and superior robustness, with a 70.9% success rate on SimplerEnv, a 26.8% improvement over OpenVLA on LIBERO, and the highest robustness score on SimplerEnv-OR. These results highlight the potential of efficient multi-frame adaptation in VLA models for more powerful and robust real-world deployment.
>
---
#### [replaced 013] LiGen: GAN-Augmented Spectral Fingerprinting for Indoor Positioning
- **分类: cs.RO; I.2.9; C.3**

- **链接: [http://arxiv.org/pdf/2508.03024v2](http://arxiv.org/pdf/2508.03024v2)**

> **作者:** Jie Lin; Hsun-Yu Lee; Ho-Ming Li; Fang-Jing Wu
>
> **备注:** 6 pages, 10 figures
>
> **摘要:** Accurate and robust indoor localization is critical for smart building applications, yet existing Wi-Fi-based systems are often vulnerable to environmental conditions. This work presents a novel indoor localization system, called LiGen, that leverages the spectral intensity patterns of ambient light as fingerprints, offering a more stable and infrastructure-free alternative to radio signals. To address the limited spectral data, we design a data augmentation framework based on generative adversarial networks (GANs), featuring two variants: PointGAN, which generates fingerprints conditioned on coordinates, and FreeGAN, which uses a weak localization model to label unconditioned samples. Our positioning model, leveraging a Multi-Layer Perceptron (MLP) architecture to train on synthesized data, achieves submeter-level accuracy, outperforming Wi-Fi-based baselines by over 50\%. LiGen also demonstrates strong robustness in cluttered environments. To the best of our knowledge, this is the first system to combine spectral fingerprints with GAN-based data augmentation for indoor localization.
>
---
#### [replaced 014] Human-assisted Robotic Policy Refinement via Action Preference Optimization
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07127v3](http://arxiv.org/pdf/2506.07127v3)**

> **作者:** Wenke Xia; Yichu Yang; Hongtao Wu; Xiao Ma; Tao Kong; Di Hu
>
> **备注:** Accepted By NeurIPS 2025
>
> **摘要:** Establishing a reliable and iteratively refined robotic system is essential for deploying real-world applications. While Vision-Language-Action (VLA) models are widely recognized as the foundation model for such robotic deployment, their reliance on offline expert demonstrations critically limits their capacity for post-deployment refinement. To mitigate this limitation, we introduce Action Preference Optimization (APO), a method designed to refine VLA models by human-assisted preference alignment gathered through interaction with environments. This method begins with a human-robot collaboration framework for reliable failure correction and interaction trajectory collection through human intervention. However, directly leveraging these interaction trajectories for preference optimization is non-trivial due to the challenges of irreversible robotic actions and token distribution mismatch. To solve this, APO proposes an adaptive reweighting algorithm with binary desirability signals derived from interaction, empowering VLA models effectively suppress failure-prone actions while enhancing corrective action adaptation. Ultimately, APO equips VLA models with the crucial capability to learn from failure, paving the way for their iterative refinement and reliable deployment in dynamic environments. The experiments conducted in simulation and real-world scenarios prove superior generalization and robustness of our human-assisted framework across a variety of manipulation tasks. We believe this work could bring insights for efficient and stable optimization of VLA models through human-robot collaboration. The code and dataset are released at https://github.com/GeWu-Lab/Action-Preference-Optimization
>
---
#### [replaced 015] Loop Closure from Two Views: Revisiting PGO for Scalable Trajectory Estimation through Monocular Priors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.16275v2](http://arxiv.org/pdf/2503.16275v2)**

> **作者:** Tian Yi Lim; Boyang Sun; Marc Pollefeys; Hermann Blum
>
> **摘要:** (Visual) Simultaneous Localization and Mapping (SLAM) remains a fundamental challenge in enabling autonomous systems to navigate and understand large-scale environments. Traditional SLAM approaches struggle to balance efficiency and accuracy, particularly in large-scale settings where extensive computational resources are required for scene reconstruction and Bundle Adjustment (BA). However, this scene reconstruction, in the form of sparse pointclouds of visual landmarks, is often only used within the SLAM system because navigation and planning methods require different map representations. In this work, we therefore investigate a more scalable Visual SLAM (VSLAM) approach without reconstruction, mainly based on approaches for two-view loop closures. By restricting the map to a sparse keyframed pose graph without dense geometry representations, our `2GO' system achieves efficient optimization with competitive absolute trajectory accuracy. In particular, we find that recent advancements in image matching and monocular depth priors enable very accurate trajectory optimization without BA. We conduct extensive experiments on diverse datasets, including large-scale scenarios, and provide a detailed analysis of the trade-offs between runtime, accuracy, and map size. Our results demonstrate that this streamlined approach supports real-time performance, scales well in map size and trajectory duration, and effectively broadens the capabilities of VSLAM for long-duration deployments to large environments.
>
---
#### [replaced 016] FSR-VLN: Fast and Slow Reasoning for Vision-Language Navigation with Hierarchical Multi-modal Scene Graph
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13733v2](http://arxiv.org/pdf/2509.13733v2)**

> **作者:** Xiaolin Zhou; Tingyang Xiao; Liu Liu; Yucheng Wang; Maiyue Chen; Xinrui Meng; Xinjie Wang; Wei Feng; Wei Sui; Zhizhong Su
>
> **备注:** 8 pages
>
> **摘要:** Visual-Language Navigation (VLN) is a fundamental challenge in robotic systems, with broad applications for the deployment of embodied agents in real-world environments. Despite recent advances, existing approaches are limited in long-range spatial reasoning, often exhibiting low success rates and high inference latency, particularly in long-range navigation tasks. To address these limitations, we propose FSR-VLN, a vision-language navigation system that combines a Hierarchical Multi-modal Scene Graph (HMSG) with Fast-to-Slow Navigation Reasoning (FSR). The HMSG provides a multi-modal map representation supporting progressive retrieval, from coarse room-level localization to fine-grained goal view and object identification. Building on HMSG, FSR first performs fast matching to efficiently select candidate rooms, views, and objects, then applies VLM-driven refinement for final goal selection. We evaluated FSR-VLN across four comprehensive indoor datasets collected by humanoid robots, utilizing 87 instructions that encompass a diverse range of object categories. FSR-VLN achieves state-of-the-art (SOTA) performance in all datasets, measured by the retrieval success rate (RSR), while reducing the response time by 82% compared to VLM-based methods on tour videos by activating slow reasoning only when fast intuition fails. Furthermore, we integrate FSR-VLN with speech interaction, planning, and control modules on a Unitree-G1 humanoid robot, enabling natural language interaction and real-time navigation.
>
---
#### [replaced 017] Agile and Cooperative Aerial Manipulation of a Cable-Suspended Load
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2501.18802v2](http://arxiv.org/pdf/2501.18802v2)**

> **作者:** Sihao Sun; Xuerui Wang; Dario Sanalitro; Antonio Franchi; Marco Tognon; Javier Alonso-Mora
>
> **备注:** 38 pages, 11 figures
>
> **摘要:** Quadrotors can carry slung loads to hard-to-reach locations at high speed. Since a single quadrotor has limited payload capacities, using a team of quadrotors to collaboratively manipulate a heavy object is a scalable and promising solution. However, existing control algorithms for multi-lifting systems only enable low-speed and low-acceleration operations due to the complex dynamic coupling between quadrotors and the load, limiting their use in time-critical missions such as search and rescue. In this work, we present a solution to significantly enhance the agility of cable-suspended multi-lifting systems. Unlike traditional cascaded solutions, we introduce a trajectory-based framework that solves the whole-body kinodynamic motion planning problem online, accounting for the dynamic coupling effects and constraints between the quadrotors and the load. The planned trajectory is provided to the quadrotors as a reference in a receding-horizon fashion and is tracked by an onboard controller that observes and compensates for the cable tension. Real-world experiments demonstrate that our framework can achieve at least eight times greater acceleration than state-of-the-art methods to follow agile trajectories. Our method can even perform complex maneuvers such as flying through narrow passages at high speed. Additionally, it exhibits high robustness against load uncertainties and does not require adding any sensors to the load, demonstrating strong practicality.
>
---
