# 机器人 cs.RO

- **最新发布 31 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] From NLVO to NAO: Reactive Robot Navigation using Velocity and Acceleration Obstacles
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决动态环境中多机器人实时避障问题。论文提出基于速度与加速度障碍物（AO和NAO）的新方法，改进传统NLVO模型，更准确考虑机器人动力学约束。通过预测轨迹并计算AO/NAO，实现高效、安全的避碰导航，适用于复杂环境中的自动驾驶车辆应用。**

- **链接: [http://arxiv.org/pdf/2506.06255v1](http://arxiv.org/pdf/2506.06255v1)**

> **作者:** Asher Stern; Zvi Shiller
>
> **备注:** 8 pages, 12 figures. arXiv admin note: text overlap with arXiv:2504.13637
>
> **摘要:** This paper introduces a novel approach for robot navigation in challenging dynamic environments. The proposed method builds upon the concept of Velocity Obstacles (VO) that was later extended to Nonlinear Velocity Obstacles (NLVO) to account for obstacles moving along nonlinear trajectories. The NLVO is extended in this paper to Acceleration Obstacles (AO) and Nonlinear Acceleration Obstacles (NAO) that account for velocity and acceleration constraints. Multi-robot navigation is achieved by using the same avoidance algorithm by all robots. At each time step, the trajectories of all robots are predicted based on their current velocity and acceleration to allow the computation of their respective NLVO, AO and NAO. The introduction of AO and NAO allows the generation of safe avoidance maneuvers that account for the robot dynamic constraints better than could be done with the NLVO alone. This paper demonstrates the use of AO and NAO for robot navigation in challenging environments. It is shown that using AO and NAO enables simultaneous real-time collision avoidance while accounting for robot kinematics and a direct consideration of its dynamic constraints. The presented approach enables reactive and efficient navigation, with potential application for autonomous vehicles operating in complex dynamic environments.
>
---
#### [new 002] Self driving algorithm for an active four wheel drive racecar
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶控制任务，旨在解决四轮驱动赛车在极限操控下的稳定性与性能优化问题。研究采用深度强化学习（DRL）方法，通过PPO算法训练智能体，在模拟赛车环境中实现从车辆状态到转向和四轮扭矩的端到端控制，以提升过弯性能并抑制转向不足，最终在不依赖传统模型的情况下实现高效自主驾驶。**

- **链接: [http://arxiv.org/pdf/2506.06077v1](http://arxiv.org/pdf/2506.06077v1)**

> **作者:** Gergely Bari; Laszlo Palkovics
>
> **摘要:** Controlling autonomous vehicles at their handling limits is a significant challenge, particularly for electric vehicles with active four wheel drive (A4WD) systems offering independent wheel torque control. While traditional Vehicle Dynamics Control (VDC) methods use complex physics-based models, this study explores Deep Reinforcement Learning (DRL) to develop a unified, high-performance controller. We employ the Proximal Policy Optimization (PPO) algorithm to train an agent for optimal lap times in a simulated racecar (TORCS) at the tire grip limit. Critically, the agent learns an end-to-end policy that directly maps vehicle states, like velocities, accelerations, and yaw rate, to a steering angle command and independent torque commands for each of the four wheels. This formulation bypasses conventional pedal inputs and explicit torque vectoring algorithms, allowing the agent to implicitly learn the A4WD control logic needed for maximizing performance and stability. Simulation results demonstrate the RL agent learns sophisticated strategies, dynamically optimizing wheel torque distribution corner-by-corner to enhance handling and mitigate the vehicle's inherent understeer. The learned behaviors mimic and, in aspects of grip utilization, potentially surpass traditional physics-based A4WD controllers while achieving competitive lap times. This research underscores DRL's potential to create adaptive control systems for complex vehicle dynamics, suggesting RL is a potent alternative for advancing autonomous driving in demanding, grip-limited scenarios for racing and road safety.
>
---
#### [new 003] UAV-UGV Cooperative Trajectory Optimization and Task Allocation for Medical Rescue Tasks in Post-Disaster Environments
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于医疗救援任务中的路径规划与任务分配研究。针对灾后基础设施损毁导致的医疗物资配送效率低问题，提出一种结合无人机（UAV）和无人车（UGV）协同工作的优化框架，通过遗传算法进行任务分配，使用改进RRT*算法生成避碰路径，并利用CMA-ES优化任务顺序和路径效率，提高了救援效率，缩短了任务完成时间和行驶距离。**

- **链接: [http://arxiv.org/pdf/2506.06136v1](http://arxiv.org/pdf/2506.06136v1)**

> **作者:** Kaiyuan Chen; Wanpeng Zhao; Yongxi Liu; Yuanqing Xia; Wannian Liang; Shuo Wang
>
> **摘要:** In post-disaster scenarios, rapid and efficient delivery of medical resources is critical and challenging due to severe damage to infrastructure. To provide an optimized solution, we propose a cooperative trajectory optimization and task allocation framework leveraging unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs). This study integrates a Genetic Algorithm (GA) for efficient task allocation among multiple UAVs and UGVs, and employs an informed-RRT* (Rapidly-exploring Random Tree Star) algorithm for collision-free trajectory generation. Further optimization of task sequencing and path efficiency is conducted using Covariance Matrix Adaptation Evolution Strategy (CMA-ES). Simulation experiments conducted in a realistic post-disaster environment demonstrate that our proposed approach significantly improves the overall efficiency of medical rescue operations compared to traditional strategies, showing substantial reductions in total mission completion time and traveled distance. Additionally, the cooperative utilization of UAVs and UGVs effectively balances their complementary advantages, highlighting the system' s scalability and practicality for real-world deployment.
>
---
#### [new 004] PyGemini: Unified Software Development towards Maritime Autonomy Systems
- **分类: cs.RO; cs.SE; cs.SY; eess.SY; D.2.11; I.6.2; I.2.9**

- **简介: 该论文属于软件开发框架任务，旨在解决海上自主系统开发中工具分散、协作困难和验证不足的问题。作者提出了PyGemini框架，采用配置驱动开发方法，整合行为驱动开发、数据导向设计和容器化技术，支持模块化、可维护和可扩展的软件架构，并提供多种 maritime 工具以促进研究与应用。**

- **链接: [http://arxiv.org/pdf/2506.06262v1](http://arxiv.org/pdf/2506.06262v1)**

> **作者:** Kjetil Vasstein; Christian Le; Simon Lervåg Breivik; Trygve Maukon Myhr; Annette Stahl; Edmund Førland Brekke
>
> **备注:** Preprint. Not yet submitted for peer review. Includes 14 figures and 3 tables. 18 pages, 1 appendix
>
> **摘要:** Ensuring the safety and certifiability of autonomous surface vessels (ASVs) requires robust decision-making systems, supported by extensive simulation, testing, and validation across a broad range of scenarios. However, the current landscape of maritime autonomy development is fragmented -- relying on disparate tools for communication, simulation, monitoring, and system integration -- which hampers interdisciplinary collaboration and inhibits the creation of compelling assurance cases, demanded by insurers and regulatory bodies. Furthermore, these disjointed tools often suffer from performance bottlenecks, vendor lock-in, and limited support for continuous integration workflows. To address these challenges, we introduce PyGemini, a permissively licensed, Python-native framework that builds on the legacy of Autoferry Gemini to unify maritime autonomy development. PyGemini introduces a novel Configuration-Driven Development (CDD) process that fuses Behavior-Driven Development (BDD), data-oriented design, and containerization to support modular, maintainable, and scalable software architectures. The framework functions as a stand-alone application, cloud-based service, or embedded library -- ensuring flexibility across research and operational contexts. We demonstrate its versatility through a suite of maritime tools -- including 3D content generation for simulation and monitoring, scenario generation for autonomy validation and training, and generative artificial intelligence pipelines for augmenting imagery -- thereby offering a scalable, maintainable, and performance-oriented foundation for future maritime robotics and autonomy research.
>
---
#### [new 005] TD-TOG Dataset: Benchmarking Zero-Shot and One-Shot Task-Oriented Grasping for Object Generalization
- **分类: cs.RO**

- **简介: 该论文属于任务导向抓取（TOG）研究，旨在解决现有TOG数据集不足、标注不完整的问题。作者构建了包含1,449个真实场景的TD-TOG数据集，并提出Binary-TOG框架，结合零样本与一样本学习，提升模型对未见物体的抓取能力，推动TOG方法在物体泛化上的应用。**

- **链接: [http://arxiv.org/pdf/2506.05576v1](http://arxiv.org/pdf/2506.05576v1)**

> **作者:** Valerija Holomjova; Jamie Grech; Dewei Yi; Bruno Yun; Andrew Starkey; Pascal Meißner
>
> **摘要:** Task-oriented grasping (TOG) is an essential preliminary step for robotic task execution, which involves predicting grasps on regions of target objects that facilitate intended tasks. Existing literature reveals there is a limited availability of TOG datasets for training and benchmarking despite large demand, which are often synthetic or have artifacts in mask annotations that hinder model performance. Moreover, TOG solutions often require affordance masks, grasps, and object masks for training, however, existing datasets typically provide only a subset of these annotations. To address these limitations, we introduce the Top-down Task-oriented Grasping (TD-TOG) dataset, designed to train and evaluate TOG solutions. TD-TOG comprises 1,449 real-world RGB-D scenes including 30 object categories and 120 subcategories, with hand-annotated object masks, affordances, and planar rectangular grasps. It also features a test set for a novel challenge that assesses a TOG solution's ability to distinguish between object subcategories. To contribute to the demand for TOG solutions that can adapt and manipulate previously unseen objects without re-training, we propose a novel TOG framework, Binary-TOG. Binary-TOG uses zero-shot for object recognition, and one-shot learning for affordance recognition. Zero-shot learning enables Binary-TOG to identify objects in multi-object scenes through textual prompts, eliminating the need for visual references. In multi-object settings, Binary-TOG achieves an average task-oriented grasp accuracy of 68.9%. Lastly, this paper contributes a comparative analysis between one-shot and zero-shot learning for object generalization in TOG to be used in the development of future TOG solutions.
>
---
#### [new 006] Astra: Toward General-Purpose Mobile Robots via Hierarchical Multimodal Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于移动机器人导航任务，旨在解决复杂室内环境中的导航适应性问题。作者提出Astra系统，包含全局与局部双模型：Astra-Global通过多模态大模型实现定位与语义地图构建，Astra-Local负责路径规划与里程估计，结合4D时空编码和自监督学习提升鲁棒性，最终在真实机器人上验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2506.06205v1](http://arxiv.org/pdf/2506.06205v1)**

> **作者:** Sheng Chen; Peiyu He; Jiaxin Hu; Ziyang Liu; Yansheng Wang; Tao Xu; Chi Zhang; Chongchong Zhang; Chao An; Shiyu Cai; Duo Cao; Kangping Chen; Shuai Chu; Tianwei Chu; Mingdi Dan; Min Du; Weiwei Fang; Pengyou Fu; Junkai Hu; Xiaowei Jiang; Zhaodi Jiang; Fuxuan Li; Jun Li; Minghui Li; Mingyao Li; Yanchang Li; Zhibin Li; Guangming Liu; Kairui Liu; Lihao Liu; Weizhi Liu; Xiaoshun Liu; Yufei Liu; Yunfei Liu; Qiang Lu; Yuanfei Luo; Xiang Lv; Hongying Ma; Sai Ma; Lingxian Mi; Sha Sa; Hongxiang Shu; Lei Tian; Chengzhi Wang; Jiayu Wang; Kaijie Wang; Qingyi Wang; Renwen Wang; Tao Wang; Wei Wang; Xirui Wang; Chao Wei; Xuguang Wei; Zijun Xia; Zhaohao Xiao; Tingshuai Yan; Liyan Yang; Yifan Yang; Zhikai Yang; Zhong Yin; Li Yuan; Liuchun Yuan; Chi Zhang; Jinyang Zhang; Junhui Zhang; Linge Zhang; Zhenyi Zhang; Zheyu Zhang; Dongjie Zhu; Hang Li; Yangang Zhang
>
> **备注:** Astra Technical Report
>
> **摘要:** Modern robot navigation systems encounter difficulties in diverse and complex indoor environments. Traditional approaches rely on multiple modules with small models or rule-based systems and thus lack adaptability to new environments. To address this, we developed Astra, a comprehensive dual-model architecture, Astra-Global and Astra-Local, for mobile robot navigation. Astra-Global, a multimodal LLM, processes vision and language inputs to perform self and goal localization using a hybrid topological-semantic graph as the global map, and outperforms traditional visual place recognition methods. Astra-Local, a multitask network, handles local path planning and odometry estimation. Its 4D spatial-temporal encoder, trained through self-supervised learning, generates robust 4D features for downstream tasks. The planning head utilizes flow matching and a novel masked ESDF loss to minimize collision risks for generating local trajectories, and the odometry head integrates multi-sensor inputs via a transformer encoder to predict the relative pose of the robot. Deployed on real in-house mobile robots, Astra achieves high end-to-end mission success rate across diverse indoor environments.
>
---
#### [new 007] On-board Mission Replanning for Adaptive Cooperative Multi-Robot Systems
- **分类: cs.RO; cs.LG**

- **简介: 论文研究多机器人系统在复杂环境中的任务动态调整问题，属路径规划与协同控制领域。针对现有方法不支持重新规划、缺乏协作及部署效率低等问题，提出一种基于图注意力网络的在线重规划算法，并验证其高效性与实用性。**

- **链接: [http://arxiv.org/pdf/2506.06094v1](http://arxiv.org/pdf/2506.06094v1)**

> **作者:** Elim Kwan; Rehman Qureshi; Liam Fletcher; Colin Laganier; Victoria Nockles; Richard Walters
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** Cooperative autonomous robotic systems have significant potential for executing complex multi-task missions across space, air, ground, and maritime domains. But they commonly operate in remote, dynamic and hazardous environments, requiring rapid in-mission adaptation without reliance on fragile or slow communication links to centralised compute. Fast, on-board replanning algorithms are therefore needed to enhance resilience. Reinforcement Learning shows strong promise for efficiently solving mission planning tasks when formulated as Travelling Salesperson Problems (TSPs), but existing methods: 1) are unsuitable for replanning, where agents do not start at a single location; 2) do not allow cooperation between agents; 3) are unable to model tasks with variable durations; or 4) lack practical considerations for on-board deployment. Here we define the Cooperative Mission Replanning Problem as a novel variant of multiple TSP with adaptations to overcome these issues, and develop a new encoder/decoder-based model using Graph Attention Networks and Attention Models to solve it effectively and efficiently. Using a simple example of cooperative drones, we show our replanner consistently (90% of the time) maintains performance within 10% of the state-of-the-art LKH3 heuristic solver, whilst running 85-370 times faster on a Raspberry Pi. This work paves the way for increased resilience in autonomous multi-agent systems.
>
---
#### [new 008] Where Do We Look When We Teach? Analyzing Human Gaze Behavior Across Demonstration Devices in Robot Imitation Learning
- **分类: cs.RO**

- **简介: 该论文研究机器人模仿学习中人类示范的凝视行为。任务是提升策略泛化能力。问题在于不同示范设备如何影响凝视行为及任务线索提取。作者设计框架分析多设备下的凝视行为，发现自然行为数据显著提高任务成功率。**

- **链接: [http://arxiv.org/pdf/2506.05808v1](http://arxiv.org/pdf/2506.05808v1)**

> **作者:** Yutaro Ishida; Takamitsu Matsubara; Takayuki Kanai; Kazuhiro Shintani; Hiroshi Bito
>
> **摘要:** Imitation learning for acquiring generalizable policies often requires a large volume of demonstration data, making the process significantly costly. One promising strategy to address this challenge is to leverage the cognitive and decision-making skills of human demonstrators with strong generalization capability, particularly by extracting task-relevant cues from their gaze behavior. However, imitation learning typically involves humans collecting data using demonstration devices that emulate a robot's embodiment and visual condition. This raises the question of how such devices influence gaze behavior. We propose an experimental framework that systematically analyzes demonstrators' gaze behavior across a spectrum of demonstration devices. Our experimental results indicate that devices emulating (1) a robot's embodiment or (2) visual condition impair demonstrators' capability to extract task-relevant cues via gaze behavior, with the extent of impairment depending on the degree of emulation. Additionally, gaze data collected using devices that capture natural human behavior improves the policy's task success rate from 18.8% to 68.8% under environmental shifts.
>
---
#### [new 009] BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人几何装配任务，旨在解决双臂协作组装破碎物体的问题。作者提出了BiAssemble方法，通过点级可操作性学习双臂协同策略，并构建了具有几何多样性和全局可重复性的现实世界基准。实验表明该方法优于现有可操作性和模仿学习方法。**

- **链接: [http://arxiv.org/pdf/2506.06221v1](http://arxiv.org/pdf/2506.06221v1)**

> **作者:** Yan Shen; Ruihai Wu; Yubin Ke; Xinyuan Song; Zeyi Li; Xiaoqi Li; Hongwei Fan; Haoran Lu; Hao dong
>
> **备注:** ICML 2025
>
> **摘要:** Shape assembly, the process of combining parts into a complete whole, is a crucial robotic skill with broad real-world applications. Among various assembly tasks, geometric assembly--where broken parts are reassembled into their original form (e.g., reconstructing a shattered bowl)--is particularly challenging. This requires the robot to recognize geometric cues for grasping, assembly, and subsequent bimanual collaborative manipulation on varied fragments. In this paper, we exploit the geometric generalization of point-level affordance, learning affordance aware of bimanual collaboration in geometric assembly with long-horizon action sequences. To address the evaluation ambiguity caused by geometry diversity of broken parts, we introduce a real-world benchmark featuring geometric variety and global reproducibility. Extensive experiments demonstrate the superiority of our approach over both previous affordance-based and imitation-based methods. Project page: https://sites.google.com/view/biassembly/.
>
---
#### [new 010] Object Navigation with Structure-Semantic Reasoning-Based Multi-level Map and Multimodal Decision-Making LLM
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决未知环境中语义新颖目标导航性能下降的问题。通过构建环境属性地图（EAM）并结合多模态层级推理模块（MHR），提升场景映射准确率与路径效率，实验证明在MP3D和HM3D数据集上取得显著改进。**

- **链接: [http://arxiv.org/pdf/2506.05896v1](http://arxiv.org/pdf/2506.05896v1)**

> **作者:** Chongshang Yan; Jiaxuan He; Delun Li; Yi Yang; Wenjie Song
>
> **备注:** 16 pages, 11 figures
>
> **摘要:** The zero-shot object navigation (ZSON) in unknown open-ended environments coupled with semantically novel target often suffers from the significant decline in performance due to the neglect of high-dimensional implicit scene information and the long-range target searching task. To address this, we proposed an active object navigation framework with Environmental Attributes Map (EAM) and MLLM Hierarchical Reasoning module (MHR) to improve its success rate and efficiency. EAM is constructed by reasoning observed environments with SBERT and predicting unobserved ones with Diffusion, utilizing human space regularities that underlie object-room correlations and area adjacencies. MHR is inspired by EAM to perform frontier exploration decision-making, avoiding the circuitous trajectories in long-range scenarios to improve path efficiency. Experimental results demonstrate that the EAM module achieves 64.5\% scene mapping accuracy on MP3D dataset, while the navigation task attains SPLs of 28.4\% and 26.3\% on HM3D and MP3D benchmarks respectively - representing absolute improvements of 21.4\% and 46.0\% over baseline methods.
>
---
#### [new 011] Learning to Recover: Dynamic Reward Shaping with Wheel-Leg Coordination for Fallen Robots
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决轮腿机器人摔倒后自主恢复问题。传统方法依赖预设动作或简化模型，效果有限。论文提出基于动态奖励塑造和课程学习的框架，结合不对称演员-评论家架构与噪声注入，实现高效训练与强健恢复策略，提升恢复成功率并降低能耗。**

- **链接: [http://arxiv.org/pdf/2506.05516v1](http://arxiv.org/pdf/2506.05516v1)**

> **作者:** Boyuan Deng; Luca Rossini; Jin Wang; Weijie Wang; Nikolaos Tsagarakis
>
> **摘要:** Adaptive recovery from fall incidents are essential skills for the practical deployment of wheeled-legged robots, which uniquely combine the agility of legs with the speed of wheels for rapid recovery. However, traditional methods relying on preplanned recovery motions, simplified dynamics or sparse rewards often fail to produce robust recovery policies. This paper presents a learning-based framework integrating Episode-based Dynamic Reward Shaping and curriculum learning, which dynamically balances exploration of diverse recovery maneuvers with precise posture refinement. An asymmetric actor-critic architecture accelerates training by leveraging privileged information in simulation, while noise-injected observations enhance robustness against uncertainties. We further demonstrate that synergistic wheel-leg coordination reduces joint torque consumption by 15.8% and 26.2% and improves stabilization through energy transfer mechanisms. Extensive evaluations on two distinct quadruped platforms achieve recovery success rates up to 99.1% and 97.8% without platform-specific tuning. The supplementary material is available at https://boyuandeng.github.io/L2R-WheelLegCoordination/
>
---
#### [new 012] Enhanced Trust Region Sequential Convex Optimization for Multi-Drone Thermal Screening Trajectory Planning in Urban Environments
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 该论文属于多无人机轨迹规划任务，旨在解决城市环境中进行体温筛查时的路径优化问题。作者提出了一种改进的信任域序列凸优化算法（TR-SCO），以提升轨迹平滑性、避障能力及覆盖效率。实验表明，该方法在优化效果和计算效率方面优于传统方法。**

- **链接: [http://arxiv.org/pdf/2506.06012v1](http://arxiv.org/pdf/2506.06012v1)**

> **作者:** Kaiyuan Chen; Zhengjie Hu; Shaolin Zhang; Yuanqing Xia; Wannian Liang; Shuo Wang
>
> **摘要:** The rapid detection of abnormal body temperatures in urban populations is essential for managing public health risks, especially during outbreaks of infectious diseases. Multi-drone thermal screening systems offer promising solutions for fast, large-scale, and non-intrusive human temperature monitoring. However, trajectory planning for multiple drones in complex urban environments poses significant challenges, including collision avoidance, coverage efficiency, and constrained flight environments. In this study, we propose an enhanced trust region sequential convex optimization (TR-SCO) algorithm for optimal trajectory planning of multiple drones performing thermal screening tasks. Our improved algorithm integrates a refined convex optimization formulation within a trust region framework, effectively balancing trajectory smoothness, obstacle avoidance, altitude constraints, and maximum screening coverage. Simulation results demonstrate that our approach significantly improves trajectory optimality and computational efficiency compared to conventional convex optimization methods. This research provides critical insights and practical contributions toward deploying efficient multi-drone systems for real-time thermal screening in urban areas. For reader who are interested in our research, we release our source code at https://github.com/Cherry0302/Enhanced-TR-SCO.
>
---
#### [new 013] Advancement and Field Evaluation of a Dual-arm Apple Harvesting Robot
- **分类: cs.RO**

- **简介: 该论文属于农业机器人任务，旨在解决苹果采摘依赖人工的问题。作者设计了一款双臂采摘机器人，集成视觉系统、真空吸力装置和压力传感器，优化了机械结构与协同策略，提升了采摘效率与可靠性，已在实际果园中进行测试验证。**

- **链接: [http://arxiv.org/pdf/2506.05714v1](http://arxiv.org/pdf/2506.05714v1)**

> **作者:** Keyi Zhu; Kyle Lammers; Kaixiang Zhang; Chaaran Arunachalam; Siddhartha Bhattacharya; Jiajia Li; Renfu Lu; Zhaojian Li
>
> **摘要:** Apples are among the most widely consumed fruits worldwide. Currently, apple harvesting fully relies on manual labor, which is costly, drudging, and hazardous to workers. Hence, robotic harvesting has attracted increasing attention in recent years. However, existing systems still fall short in terms of performance, effectiveness, and reliability for complex orchard environments. In this work, we present the development and evaluation of a dual-arm harvesting robot. The system integrates a ToF camera, two 4DOF robotic arms, a centralized vacuum system, and a post-harvest handling module. During harvesting, suction force is dynamically assigned to either arm via the vacuum system, enabling efficient apple detachment while reducing power consumption and noise. Compared to our previous design, we incorporated a platform movement mechanism that enables both in-out and up-down adjustments, enhancing the robot's dexterity and adaptability to varying canopy structures. On the algorithmic side, we developed a robust apple localization pipeline that combines a foundation-model-based detector, segmentation, and clustering-based depth estimation, which improves performance in orchards. Additionally, pressure sensors were integrated into the system, and a novel dual-arm coordination strategy was introduced to respond to harvest failures based on sensor feedback, further improving picking efficiency. Field demos were conducted in two commercial orchards in MI, USA, with different canopy structures. The system achieved success rates of 0.807 and 0.797, with an average picking cycle time of 5.97s. The proposed strategy reduced harvest time by 28% compared to a single-arm baseline. The dual-arm harvesting robot enhances the reliability and efficiency of apple picking. With further advancements, the system holds strong potential for autonomous operation and commercialization for the apple industry.
>
---
#### [new 014] Optimal Robotic Velcro Peeling with Force Feedback
- **分类: cs.RO**

- **简介: 该论文研究机器人剥离魔术贴的任务，旨在解决在环境几何未知且传感器反馈不完整的情况下如何高效完成剥离的问题。作者建立了系统模型，推导了完全可观情况下的最优解，并设计了基于启发式的控制器与状态估计器以应对部分可观情况，最终实现了高效、高成功率的魔术贴剥离。**

- **链接: [http://arxiv.org/pdf/2506.05812v1](http://arxiv.org/pdf/2506.05812v1)**

> **作者:** Jiacheng Yuan; Changhyun Choi; Volkan Isler
>
> **摘要:** We study the problem of peeling a Velcro strap from a surface using a robotic manipulator. The surface geometry is arbitrary and unknown. The robot has access to only the force feedback and its end-effector position. This problem is challenging due to the partial observability of the environment and the incompleteness of the sensor feedback. To solve it, we first model the system with simple analytic state and action models based on quasi-static dynamics assumptions. We then study the fully-observable case where the state of both the Velcro and the robot are given. For this case, we obtain the optimal solution in closed-form which minimizes the total energy cost. Next, for the partially-observable case, we design a state estimator which estimates the underlying state using only force and position feedback. Then, we present a heuristics-based controller that balances exploratory and exploitative behaviors in order to peel the velcro efficiently. Finally, we evaluate our proposed method in environments with complex geometric uncertainties and sensor noises, achieving 100% success rate with less than 80% increase in energy cost compared to the optimal solution when the environment is fully-observable, outperforming the baselines by a large margin.
>
---
#### [new 015] 3DFlowAction: Learning Cross-Embodiment Manipulation from 3D Flow World Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控任务，旨在解决跨形态操作中的动作统一性和泛化性问题。作者提出了3DFlowAction方法，通过构建一个基于3D光流的世界模型，从人类和机器人操作数据中学习，实现对不同机器人在多样场景下的操作动作规划与适应。**

- **链接: [http://arxiv.org/pdf/2506.06199v1](http://arxiv.org/pdf/2506.06199v1)**

> **作者:** Hongyan Zhi; Peihao Chen; Siyuan Zhou; Yubo Dong; Quanxi Wu; Lei Han; Mingkui Tan
>
> **摘要:** Manipulation has long been a challenging task for robots, while humans can effortlessly perform complex interactions with objects, such as hanging a cup on the mug rack. A key reason is the lack of a large and uniform dataset for teaching robots manipulation skills. Current robot datasets often record robot action in different action spaces within a simple scene. This hinders the robot to learn a unified and robust action representation for different robots within diverse scenes. Observing how humans understand a manipulation task, we find that understanding how the objects should move in the 3D space is a critical clue for guiding actions. This clue is embodiment-agnostic and suitable for both humans and different robots. Motivated by this, we aim to learn a 3D flow world model from both human and robot manipulation data. This model predicts the future movement of the interacting objects in 3D space, guiding action planning for manipulation. Specifically, we synthesize a large-scale 3D optical flow dataset, named ManiFlow-110k, through a moving object auto-detect pipeline. A video diffusion-based world model then learns manipulation physics from these data, generating 3D optical flow trajectories conditioned on language instructions. With the generated 3D object optical flow, we propose a flow-guided rendering mechanism, which renders the predicted final state and leverages GPT-4o to assess whether the predicted flow aligns with the task description. This equips the robot with a closed-loop planning ability. Finally, we consider the predicted 3D optical flow as constraints for an optimization policy to determine a chunk of robot actions for manipulation. Extensive experiments demonstrate strong generalization across diverse robotic manipulation tasks and reliable cross-embodiment adaptation without hardware-specific training.
>
---
#### [new 016] Improving Long-Range Navigation with Spatially-Enhanced Recurrent Memory via End-to-End Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决长距离导航中空间记忆不足的问题。通过改进循环神经网络（SRUs），增强其空间记忆能力，并结合强化学习实现端到端训练，提升了导航性能。**

- **链接: [http://arxiv.org/pdf/2506.05997v1](http://arxiv.org/pdf/2506.05997v1)**

> **作者:** Fan Yang; Per Frivik; David Hoeller; Chen Wang; Cesar Cadena; Marco Hutter
>
> **备注:** 21 pages
>
> **摘要:** Recent advancements in robot navigation, especially with end-to-end learning approaches like reinforcement learning (RL), have shown remarkable efficiency and effectiveness. Yet, successful navigation still relies on two key capabilities: mapping and planning, whether explicit or implicit. Classical approaches use explicit mapping pipelines to register ego-centric observations into a coherent map frame for the planner. In contrast, end-to-end learning achieves this implicitly, often through recurrent neural networks (RNNs) that fuse current and past observations into a latent space for planning. While architectures such as LSTM and GRU capture temporal dependencies, our findings reveal a key limitation: their inability to perform effective spatial memorization. This skill is essential for transforming and integrating sequential observations from varying perspectives to build spatial representations that support downstream planning. To address this, we propose Spatially-Enhanced Recurrent Units (SRUs), a simple yet effective modification to existing RNNs, designed to enhance spatial memorization capabilities. We introduce an attention-based architecture with SRUs, enabling long-range navigation using a single forward-facing stereo camera. Regularization techniques are employed to ensure robust end-to-end recurrent training via RL. Experimental results show our approach improves long-range navigation by 23.5% compared to existing RNNs. Furthermore, with SRU memory, our method outperforms the RL baseline with explicit mapping and memory modules, achieving a 29.6% improvement in diverse environments requiring long-horizon mapping and memorization. Finally, we address the sim-to-real gap by leveraging large-scale pretraining on synthetic depth data, enabling zero-shot transfer to diverse and complex real-world environments.
>
---
#### [new 017] End-to-End Framework for Robot Lawnmower Coverage Path Planning using Cellular Decomposition
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，旨在解决自动割草机在复杂草坪中的覆盖路径规划问题。论文提出了一种端到端框架，结合用户输入、自适应分解算法和可视化工具，生成高效覆盖路径，并通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.06028v1](http://arxiv.org/pdf/2506.06028v1)**

> **作者:** Nikunj Shah; Utsav Dey; Kenji Nishimiya
>
> **备注:** 8 pages, ICRA 2025, Workshop on Field Robotics
>
> **摘要:** Efficient Coverage Path Planning (CPP) is necessary for autonomous robotic lawnmowers to effectively navigate and maintain lawns with diverse and irregular shapes. This paper introduces a comprehensive end-to-end pipeline for CPP, designed to convert user-defined boundaries on an aerial map into optimized coverage paths seamlessly. The pipeline includes user input extraction, coordinate transformation, area decomposition and path generation using our novel AdaptiveDecompositionCPP algorithm, preview and customization through an interactive coverage path visualizer, and conversion to actionable GPS waypoints. The AdaptiveDecompositionCPP algorithm combines cellular decomposition with an adaptive merging strategy to reduce non-mowing travel thereby enhancing operational efficiency. Experimental evaluations, encompassing both simulations and real-world lawnmower tests, demonstrate the effectiveness of the framework in coverage completeness and mowing efficiency.
>
---
#### [new 018] Towards Autonomous In-situ Soil Sampling and Mapping in Large-Scale Agricultural Environments
- **分类: cs.RO; cs.ET**

- **简介: 该论文旨在解决传统土壤采样与分析方法在大规模精准农业中效率低、空间分辨率不足的问题。任务是实现自主实地土壤采样与绘图。作者设计了包含采样系统与实验室分析系统的机器人平台，并在澳大利亚农场验证，证明其可快速获取并准确分析土壤样本的关键属性。**

- **链接: [http://arxiv.org/pdf/2506.05653v1](http://arxiv.org/pdf/2506.05653v1)**

> **作者:** Thien Hoang Nguyen; Erik Muller; Michael Rubin; Xiaofei Wang; Fiorella Sibona; Salah Sukkarieh
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics
>
> **摘要:** Traditional soil sampling and analysis methods are labor-intensive, time-consuming, and limited in spatial resolution, making them unsuitable for large-scale precision agriculture. To address these limitations, we present a robotic solution for real-time sampling, analysis and mapping of key soil properties. Our system consists of two main sub-systems: a Sample Acquisition System (SAS) for precise, automated in-field soil sampling; and a Sample Analysis Lab (Lab) for real-time soil property analysis. The system's performance was validated through extensive field trials at a large-scale Australian farm. Experimental results show that the SAS can consistently acquire soil samples with a mass of 50g at a depth of 200mm, while the Lab can process each sample within 10 minutes to accurately measure pH and macronutrients. These results demonstrate the potential of the system to provide farmers with timely, data-driven insights for more efficient and sustainable soil management and fertilizer application.
>
---
#### [new 019] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 论文提出BEAST，一种基于B样条的动作序列编码方法，用于模仿学习中的动作标记化。任务是解决现有标记化方法需额外训练、生成不连续动作的问题。工作包括设计无需训练的B样条编码器，实现快速并行解码和光滑轨迹生成，并在多种模型和任务中验证其效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.06072v1](http://arxiv.org/pdf/2506.06072v1)**

> **作者:** Hongyi Zhou; Weiran Liao; Xi Huang; Yucheng Tang; Fabian Otto; Xiaogang Jia; Xinkai Jiang; Simon Hilber; Ge Li; Qian Wang; Ömer Erdinç Yağmurlu; Nils Blank; Moritz Reuss; Rudolf Lioutikov
>
> **摘要:** We present the B-spline Encoded Action Sequence Tokenizer (BEAST), a novel action tokenizer that encodes action sequences into compact discrete or continuous tokens using B-splines. In contrast to existing action tokenizers based on vector quantization or byte pair encoding, BEAST requires no separate tokenizer training and consistently produces tokens of uniform length, enabling fast action sequence generation via parallel decoding. Leveraging our B-spline formulation, BEAST inherently ensures generating smooth trajectories without discontinuities between adjacent segments. We extensively evaluate BEAST by integrating it with three distinct model architectures: a Variational Autoencoder (VAE) with continuous tokens, a decoder-only Transformer with discrete tokens, and Florence-2, a pretrained Vision-Language Model with an encoder-decoder architecture, demonstrating BEAST's compatibility and scalability with large pretrained models. We evaluate BEAST across three established benchmarks consisting of 166 simulated tasks and on three distinct robot settings with a total of 8 real-world tasks. Experimental results demonstrate that BEAST (i) significantly reduces both training and inference computational costs, and (ii) consistently generates smooth, high-frequency control signals suitable for continuous control tasks while (iii) reliably achieves competitive task success rates compared to state-of-the-art methods.
>
---
#### [new 020] A Soft Robotic Module with Pneumatic Actuation and Enhanced Controllability Using a Shape Memory Alloy Wire
- **分类: cs.RO**

- **简介: 该论文设计了一种结合形状记忆合金（SMA）丝的气动软体机器人模块，旨在提高弯曲角度控制的精度。通过引入SMA丝作为新的应变限制层，并采用闭环控制算法和摄像头反馈，实现对垂直平面内弯曲角度的精确控制，减小了误差范围并缩短了响应时间。**

- **链接: [http://arxiv.org/pdf/2506.05741v1](http://arxiv.org/pdf/2506.05741v1)**

> **作者:** Mohammadnavid Golchin
>
> **摘要:** In this paper, a compressed air-actuated soft robotic module was developed by incorporating a shape memory alloy (SMA) wire into its structure to achieve the desired bending angle with greater precision. First, a fiber-reinforced bending module with a strain-limiting layer made of polypropylene was fabricated. The SMA wire was then placed in a silicon matrix, which was used as a new strain-limiting layer. A simple closed-loop control algorithm was used to regulate the bending angle of the soft robot within its workspace. A camera was utilized to measure the angular changes in the vertical plane. Different angles, ranging from 0 to 65 degrees, were covered to evaluate the performance of the module and the bending angle control algorithm. The experimental tests demonstrate that using the SMA wire results in more precise control of bending in the vertical plane. In addition, it is possible to bend more with less working pressure. The error range was reduced from an average of 5 degrees to 2 degrees, and the rise time was reduced from an average of 19 seconds to 3 seconds.
>
---
#### [new 021] Bridging Perception and Action: Spatially-Grounded Mid-Level Representations for Robot Generalization
- **分类: cs.RO; 68T40**

- **简介: 该论文属于机器人学习任务，旨在提升机器人执行灵巧操作的泛化能力。论文提出了一种基于空间感知的中层表示方法，并结合专家混合策略架构与模仿学习算法，提高策略的学习效果与精度，验证了这类表示在连接感知与行动中的作用。**

- **链接: [http://arxiv.org/pdf/2506.06196v1](http://arxiv.org/pdf/2506.06196v1)**

> **作者:** Jonathan Yang; Chuyuan Kelly Fu; Dhruv Shah; Dorsa Sadigh; Fei Xia; Tingnan Zhang
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** In this work, we investigate how spatially grounded auxiliary representations can provide both broad, high-level grounding as well as direct, actionable information to improve policy learning performance and generalization for dexterous tasks. We study these mid-level representations across three critical dimensions: object-centricity, pose-awareness, and depth-awareness. We use these interpretable mid-level representations to train specialist encoders via supervised learning, then feed them as inputs to a diffusion policy to solve dexterous bimanual manipulation tasks in the real world. We propose a novel mixture-of-experts policy architecture that combines multiple specialized expert models, each trained on a distinct mid-level representation, to improve policy generalization. This method achieves an average success rate that is 11% higher than a language-grounded baseline and 24 percent higher than a standard diffusion policy baseline on our evaluation tasks. Furthermore, we find that leveraging mid-level representations as supervision signals for policy actions within a weighted imitation learning algorithm improves the precision with which the policy follows these representations, yielding an additional performance increase of 10%. Our findings highlight the importance of grounding robot policies not only with broad perceptual tasks but also with more granular, actionable representations. For further information and videos, please visit https://mid-level-moe.github.io.
>
---
#### [new 022] Trajectory Entropy: Modeling Game State Stability from Multimodality Trajectory Prediction
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中多智能体交互建模任务，旨在解决现有层级博弈框架忽略驾驶复杂性和状态变化导致的计算冗余和误差问题。作者提出“轨迹熵”度量，通过多模态轨迹预测提取不确定性信号，并利用信噪比量化智能体博弈状态，进而改进层级博弈框架，提升预测与规划性能。**

- **链接: [http://arxiv.org/pdf/2506.05810v1](http://arxiv.org/pdf/2506.05810v1)**

> **作者:** Yesheng Zhang; Wenjian Sun; Yuheng Chen; Qingwei Liu; Qi Lin; Rui Zhang; Xu Zhao
>
> **备注:** 10 pages
>
> **摘要:** Complex interactions among agents present a significant challenge for autonomous driving in real-world scenarios. Recently, a promising approach has emerged, which formulates the interactions of agents as a level-k game framework. It effectively decouples agent policies by hierarchical game levels. However, this framework ignores both the varying driving complexities among agents and the dynamic changes in agent states across game levels, instead treating them uniformly. Consequently, redundant and error-prone computations are introduced into this framework. To tackle the issue, this paper proposes a metric, termed as Trajectory Entropy, to reveal the game status of agents within the level-k game framework. The key insight stems from recognizing the inherit relationship between agent policy uncertainty and the associated driving complexity. Specifically, Trajectory Entropy extracts statistical signals representing uncertainty from the multimodality trajectory prediction results of agents in the game. Then, the signal-to-noise ratio of this signal is utilized to quantify the game status of agents. Based on the proposed Trajectory Entropy, we refine the current level-k game framework through a simple gating mechanism, significantly improving overall accuracy while reducing computational costs. Our method is evaluated on the Waymo and nuPlan datasets, in terms of trajectory prediction, open-loop and closed-loop planning tasks. The results demonstrate the state-of-the-art performance of our method, with precision improved by up to 19.89% for prediction and up to 16.48% for planning.
>
---
#### [new 023] Gradual Transition from Bellman Optimality Operator to Bellman Operator in Online Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文研究在线强化学习中连续动作空间任务，旨在解决现有方法仅使用贝尔曼算子导致的样本效率低问题。通过引入贝尔曼最优算子加速学习，提出退火策略缓解过高估计偏差，并结合TD3和SAC算法验证有效性。**

- **链接: [http://arxiv.org/pdf/2506.05968v1](http://arxiv.org/pdf/2506.05968v1)**

> **作者:** Motoki Omura; Kazuki Ota; Takayuki Osa; Yusuke Mukuta; Tatsuya Harada
>
> **备注:** Accepted at ICML 2025. Source code: https://github.com/motokiomura/annealed-q-learning
>
> **摘要:** For continuous action spaces, actor-critic methods are widely used in online reinforcement learning (RL). However, unlike RL algorithms for discrete actions, which generally model the optimal value function using the Bellman optimality operator, RL algorithms for continuous actions typically model Q-values for the current policy using the Bellman operator. These algorithms for continuous actions rely exclusively on policy updates for improvement, which often results in low sample efficiency. This study examines the effectiveness of incorporating the Bellman optimality operator into actor-critic frameworks. Experiments in a simple environment show that modeling optimal values accelerates learning but leads to overestimation bias. To address this, we propose an annealing approach that gradually transitions from the Bellman optimality operator to the Bellman operator, thereby accelerating learning while mitigating bias. Our method, combined with TD3 and SAC, significantly outperforms existing approaches across various locomotion and manipulation tasks, demonstrating improved performance and robustness to hyperparameters related to optimality.
>
---
#### [new 024] EqCollide: Equivariant and Collision-Aware Deformable Objects Neural Simulator
- **分类: cs.LG; cs.CE; cs.RO**

- **简介: 该论文属于物理仿真任务，旨在解决可变形物体碰撞模拟中对称性不足、碰撞处理差和扩展性差的问题。作者提出了EqCollide，一种端到端的等变神经场模拟器，通过等变编码器和图神经网络结合神经微分方程，实现对可变形物体及其碰撞的准确、稳定和可扩展的模拟。**

- **链接: [http://arxiv.org/pdf/2506.05797v1](http://arxiv.org/pdf/2506.05797v1)**

> **作者:** Qianyi Chen; Tianrun Gao; Chenbo Jiang; Tailin Wu
>
> **摘要:** Simulating collisions of deformable objects is a fundamental yet challenging task due to the complexity of modeling solid mechanics and multi-body interactions. Existing data-driven methods often suffer from lack of equivariance to physical symmetries, inadequate handling of collisions, and limited scalability. Here we introduce EqCollide, the first end-to-end equivariant neural fields simulator for deformable objects and their collisions. We propose an equivariant encoder to map object geometry and velocity into latent control points. A subsequent equivariant Graph Neural Network-based Neural Ordinary Differential Equation models the interactions among control points via collision-aware message passing. To reconstruct velocity fields, we query a neural field conditioned on control point features, enabling continuous and resolution-independent motion predictions. Experimental results show that EqCollide achieves accurate, stable, and scalable simulations across diverse object configurations, and our model achieves 24.34% to 35.82% lower rollout MSE even compared with the best-performing baseline model. Furthermore, our model could generalize to more colliding objects and extended temporal horizons, and stay robust to input transformed with group action.
>
---
#### [new 025] Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于终身机器人学习任务，旨在解决持续适应中灾难性遗忘与知识共享受限的问题。作者提出DMPEL方法，通过渐进学习低秩专家库并动态组合策略，结合系数回放减轻遗忘。实验表明其在成功率和资源效率上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.05985v1](http://arxiv.org/pdf/2506.05985v1)**

> **作者:** Yuheng Lei; Sitong Mao; Shunbo Zhou; Hongyuan Zhang; Xuelong Li; Ping Luo
>
> **摘要:** A generalist agent must continuously learn and adapt throughout its lifetime, achieving efficient forward transfer while minimizing catastrophic forgetting. Previous work within the dominant pretrain-then-finetune paradigm has explored parameter-efficient fine-tuning for single-task adaptation, effectively steering a frozen pretrained model with a small number of parameters. However, in the context of lifelong learning, these methods rely on the impractical assumption of a test-time task identifier and restrict knowledge sharing among isolated adapters. To address these limitations, we propose Dynamic Mixture of Progressive Parameter-Efficient Expert Library (DMPEL) for lifelong robot learning. DMPEL progressively learn a low-rank expert library and employs a lightweight router to dynamically combine experts into an end-to-end policy, facilitating flexible behavior during lifelong adaptation. Moreover, by leveraging the modular structure of the fine-tuned parameters, we introduce coefficient replay to guide the router in accurately retrieving frozen experts for previously encountered tasks, thereby mitigating catastrophic forgetting. This method is significantly more storage- and computationally-efficient than applying demonstration replay to the entire policy. Extensive experiments on the lifelong manipulation benchmark LIBERO demonstrate that our framework outperforms state-of-the-art lifelong learning methods in success rates across continual adaptation, while utilizing minimal trainable parameters and storage.
>
---
#### [new 026] A Modular Haptic Display with Reconfigurable Signals for Personalized Information Transfer
- **分类: cs.HC; cs.RO**

- **简介: 该论文设计了一种模块化触觉显示系统，结合硬件与信息算法实现个性化反馈。任务是提升人机交互中的触觉传递效果。解决如何根据用户差异动态配置触觉信号的问题。工作包括构建可重构的气动触觉模块，用流体逻辑简化控制，并通过模型优化信息传输效率。**

- **链接: [http://arxiv.org/pdf/2506.05648v1](http://arxiv.org/pdf/2506.05648v1)**

> **作者:** Antonio Alvarez Valdivia; Benjamin A. Christie; Dylan P. Losey; Laura H. Blumenschein
>
> **备注:** This work has been submitted to the IEEE Transactions on Haptics for possible publication
>
> **摘要:** We present a customizable soft haptic system that integrates modular hardware with an information-theoretic algorithm to personalize feedback for different users and tasks. Our platform features modular, multi-degree-of-freedom pneumatic displays, where different signal types, such as pressure, frequency, and contact area, can be activated or combined using fluidic logic circuits. These circuits simplify control by reducing reliance on specialized electronics and enabling coordinated actuation of multiple haptic elements through a compact set of inputs. Our approach allows rapid reconfiguration of haptic signal rendering through hardware-level logic switching without rewriting code. Personalization of the haptic interface is achieved through the combination of modular hardware and software-driven signal selection. To determine which display configurations will be most effective, we model haptic communication as a signal transmission problem, where an agent must convey latent information to the user. We formulate the optimization problem to identify the haptic hardware setup that maximizes the information transfer between the intended message and the user's interpretation, accounting for individual differences in sensitivity, preferences, and perceptual salience. We evaluate this framework through user studies where participants interact with reconfigurable displays under different signal combinations. Our findings support the role of modularity and personalization in creating multimodal haptic interfaces and advance the development of reconfigurable systems that adapt with users in dynamic human-machine interaction contexts.
>
---
#### [new 027] Trajectory Optimization for UAV-Based Medical Delivery with Temporal Logic Constraints and Convex Feasible Set Collision Avoidance
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文研究无人机在城市环境中进行时间敏感型医疗配送的轨迹优化问题。主要解决如何在满足时间窗口、优先级和三维避障约束下，生成安全高效的飞行路径。论文通过信号时序逻辑建模任务目标，并采用凸可行集方法处理障碍物避让，最终将问题转化为可高效求解的凸优化问题。**

- **链接: [http://arxiv.org/pdf/2506.06038v1](http://arxiv.org/pdf/2506.06038v1)**

> **作者:** Kaiyuan Chen; Yuhan Suo; Shaowei Cui; Yuanqing Xia; Wannian Liang; Shuo Wang
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** This paper addresses the problem of trajectory optimization for unmanned aerial vehicles (UAVs) performing time-sensitive medical deliveries in urban environments. Specifically, we consider a single UAV with 3 degree-of-freedom dynamics tasked with delivering blood packages to multiple hospitals, each with a predefined time window and priority. Mission objectives are encoded using Signal Temporal Logic (STL), enabling the formal specification of spatial-temporal constraints. To ensure safety, city buildings are modeled as 3D convex obstacles, and obstacle avoidance is handled through a Convex Feasible Set (CFS) method. The entire planning problem-combining UAV dynamics, STL satisfaction, and collision avoidance-is formulated as a convex optimization problem that ensures tractability and can be solved efficiently using standard convex programming techniques. Simulation results demonstrate that the proposed method generates dynamically feasible, collision-free trajectories that satisfy temporal mission goals, providing a scalable and reliable approach for autonomous UAV-based medical logistics.
>
---
#### [new 028] A Compendium of Autonomous Navigation using Object Detection and Tracking in Unmanned Aerial Vehicles
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文综述了基于目标检测与跟踪的无人机自主导航技术。任务是提升无人机在复杂环境中的自主导航能力，解决信号干扰、实时处理等问题。工作包括分析多种算法在灾害管理、交通监控等场景的应用。**

- **链接: [http://arxiv.org/pdf/2506.05378v1](http://arxiv.org/pdf/2506.05378v1)**

> **作者:** Mohit Arora; Pratyush Shukla; Shivali Chopra
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are one of the most revolutionary inventions of 21st century. At the core of a UAV lies the central processing system that uses wireless signals to control their movement. The most popular UAVs are quadcopters that use a set of four motors, arranged as two on either side with opposite spin. An autonomous UAV is called a drone. Drones have been in service in the US army since the 90's for covert missions critical to national security. It would not be wrong to claim that drones make up an integral part of the national security and provide the most valuable service during surveillance operations. While UAVs are controlled using wireless signals, there reside some challenges that disrupt the operation of such vehicles such as signal quality and range, real time processing, human expertise, robust hardware and data security. These challenges can be solved by programming UAVs to be autonomous, using object detection and tracking, through Computer Vision algorithms. Computer Vision is an interdisciplinary field that seeks the use of deep learning to gain a high-level understanding of digital images and videos for the purpose of automating the task of human visual system. Using computer vision, algorithms for detecting and tracking various objects can be developed suitable to the hardware so as to allow real time processing for immediate judgement. This paper attempts to review the various approaches several authors have proposed for the purpose of autonomous navigation of UAVs by through various algorithms of object detection and tracking in real time, for the purpose of applications in various fields such as disaster management, dense area exploration, traffic vehicle surveillance etc.
>
---
#### [new 029] Equivariant Filter for Relative Attitude and Target Angular Velocity Estimation
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 论文研究相对姿态与角速度估计任务，旨在解决目标物体在航天器交会对接中的相对姿态和角速度精确估计问题。作者设计了一种等变滤波器（EqF），利用对称性建模及非共线向量观测，分析系统可观性和收敛性，并通过仿真与实验验证方法有效性，尤其应对低频测量挑战。**

- **链接: [http://arxiv.org/pdf/2506.06016v1](http://arxiv.org/pdf/2506.06016v1)**

> **作者:** Gil Serrano; Bruno J. Guerreiro; Pedro Lourenço; Rita Cunha
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Accurate estimation of the relative attitude and angular velocity between two rigid bodies is fundamental in aerospace applications such as spacecraft rendezvous and docking. In these scenarios, a chaser vehicle must determine the orientation and angular velocity of a target object using onboard sensors. This work addresses the challenge of designing an Equivariant Filter (EqF) that can reliably estimate both the relative attitude and the target angular velocity using noisy observations of two known, non-collinear vectors fixed in the target frame. To derive the EqF, a symmetry for the system is proposed and an equivariant lift onto the symmetry group is calculated. Observability and convergence properties are analyzed. Simulations demonstrate the filter's performance, with Monte Carlo runs yielding statistically significant results. The impact of low-rate measurements is also examined and a strategy to mitigate this effect is proposed. Experimental results, using fiducial markers and both conventional and event cameras for measurement acquisition, further validate the approach, confirming its effectiveness in a realistic setting.
>
---
#### [new 030] You Only Estimate Once: Unified, One-stage, Real-Time Category-level Articulated Object 6D Pose Estimation for Robotic Grasping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人抓取中的类别级关节物体6D姿态估计任务。它旨在解决现有方法计算成本高、实时性差的问题。论文提出YOEO方法，通过单阶段网络同时输出实例分割和NPCS表示，实现端到端的高效姿态估计。**

- **链接: [http://arxiv.org/pdf/2506.05719v1](http://arxiv.org/pdf/2506.05719v1)**

> **作者:** Jingshun Huang; Haitao Lin; Tianyu Wang; Yanwei Fu; Yu-Gang Jiang; Xiangyang Xue
>
> **备注:** To appear in ICRA 2025
>
> **摘要:** This paper addresses the problem of category-level pose estimation for articulated objects in robotic manipulation tasks. Recent works have shown promising results in estimating part pose and size at the category level. However, these approaches primarily follow a complex multi-stage pipeline that first segments part instances in the point cloud and then estimates the Normalized Part Coordinate Space (NPCS) representation for 6D poses. These approaches suffer from high computational costs and low performance in real-time robotic tasks. To address these limitations, we propose YOEO, a single-stage method that simultaneously outputs instance segmentation and NPCS representations in an end-to-end manner. We use a unified network to generate point-wise semantic labels and centroid offsets, allowing points from the same part instance to vote for the same centroid. We further utilize a clustering algorithm to distinguish points based on their estimated centroid distances. Finally, we first separate the NPCS region of each instance. Then, we align the separated regions with the real point cloud to recover the final pose and size. Experimental results on the GAPart dataset demonstrate the pose estimation capabilities of our proposed single-shot method. We also deploy our synthetically-trained model in a real-world setting, providing real-time visual feedback at 200Hz, enabling a physical Kinova robot to interact with unseen articulated objects. This showcases the utility and effectiveness of our proposed method.
>
---
#### [new 031] Robust sensor fusion against on-vehicle sensor staleness
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶中的感知任务，旨在解决多传感器数据因时间不同步导致的感知退化问题。通过引入时间戳偏移特征和模拟传感器延迟的数据增强策略，提升了模型在传感器延迟下的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.05780v1](http://arxiv.org/pdf/2506.05780v1)**

> **作者:** Meng Fan; Yifan Zuo; Patrick Blaes; Harley Montgomery; Subhasis Das
>
> **备注:** This paper has been accepted by CVPR 2025 Precognition Workshop
>
> **摘要:** Sensor fusion is crucial for a performant and robust Perception system in autonomous vehicles, but sensor staleness, where data from different sensors arrives with varying delays, poses significant challenges. Temporal misalignment between sensor modalities leads to inconsistent object state estimates, severely degrading the quality of trajectory predictions that are critical for safety. We present a novel and model-agnostic approach to address this problem via (1) a per-point timestamp offset feature (for LiDAR and radar both relative to camera) that enables fine-grained temporal awareness in sensor fusion, and (2) a data augmentation strategy that simulates realistic sensor staleness patterns observed in deployed vehicles. Our method is integrated into a perspective-view detection model that consumes sensor data from multiple LiDARs, radars and cameras. We demonstrate that while a conventional model shows significant regressions when one sensor modality is stale, our approach reaches consistently good performance across both synchronized and stale conditions.
>
---
## 更新

#### [replaced 001] Distributed Expectation Propagation for Multi-Object Tracking over Sensor Networks
- **分类: eess.SP; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.18795v2](http://arxiv.org/pdf/2505.18795v2)**

> **作者:** Qing Li; Runze Gan; James R. Hopgood; Michael E. Davies; Simon J. Godsill
>
> **摘要:** In this paper, we present a novel distributed expectation propagation algorithm for multiple sensors, multiple objects tracking in cluttered environments. The proposed framework enables each sensor to operate locally while collaboratively exchanging moment estimates with other sensors, thus eliminating the need to transmit all data to a central processing node. Specifically, we introduce a fast and parallelisable Rao-Blackwellised Gibbs sampling scheme to approximate the tilted distributions, which enhances the accuracy and efficiency of expectation propagation updates. Results demonstrate that the proposed algorithm improves both communication and inference efficiency for multi-object tracking tasks with dynamic sensor connectivity and varying clutter levels.
>
---
#### [replaced 002] HJRNO: Hamilton-Jacobi Reachability with Neural Operators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19989v2](http://arxiv.org/pdf/2504.19989v2)**

> **作者:** Yankai Li; Mo Chen
>
> **摘要:** Ensuring the safety of autonomous systems under uncertainty is a critical challenge. Hamilton-Jacobi reachability (HJR) analysis is a widely used method for guaranteeing safety under worst-case disturbances. In this work, we propose HJRNO, a neural operator-based framework for solving backward reachable tubes (BRTs) efficiently and accurately. By leveraging neural operators, HJRNO learns a mapping between value functions, enabling fast inference with strong generalization across different obstacle shapes and system configurations. We demonstrate that HJRNO achieves low error on random obstacle scenarios and generalizes effectively across varying system dynamics. These results suggest that HJRNO offers a promising foundation model approach for scalable, real-time safety analysis in autonomous systems.
>
---
#### [replaced 003] SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.00211v2](http://arxiv.org/pdf/2503.00211v2)**

> **作者:** Jiawei Zhang; Xuan Yang; Taiqi Wang; Yu Yao; Aleksandr Petiushko; Bo Li
>
> **摘要:** Traditional autonomous driving systems often struggle to connect high-level reasoning with low-level control, leading to suboptimal and sometimes unsafe behaviors. Recent advances in multimodal large language models (MLLMs), which process both visual and textual data, offer an opportunity to unify perception and reasoning. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a framework that enhances MLLM-based autonomous driving by incorporating both unstructured and structured knowledge. First, we introduce a Position-Dependent Cross-Entropy (PDCE) loss to improve low-level control signal predictions when values are represented as text. Second, to explicitly integrate safety knowledge, we develop a reasoning component that translates traffic rules into first-order logic (e.g., "red light $\implies$ stop") and embeds them into a probabilistic graphical model (e.g., Markov Logic Network) to verify predicted actions using recognized environmental attributes. Additionally, our Multimodal Retrieval-Augmented Generation (RAG) model leverages video, control signals, and environmental attributes to learn from past driving experiences. Integrating PDCE, MLN, and Multimodal RAG, SafeAuto outperforms existing baselines across multiple datasets, enabling more accurate, reliable, and safer autonomous driving. The code is available at https://github.com/AI-secure/SafeAuto.
>
---
#### [replaced 004] TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.10345v3](http://arxiv.org/pdf/2412.10345v3)**

> **作者:** Ruijie Zheng; Yongyuan Liang; Shuaiyi Huang; Jianfeng Gao; Hal Daumé III; Andrey Kolobov; Furong Huang; Jianwei Yang
>
> **摘要:** Although large vision-language-action (VLA) models pretrained on extensive robot datasets offer promising generalist policies for robotic learning, they still struggle with spatial-temporal dynamics in interactive robotics, making them less effective in handling complex tasks, such as manipulation. In this work, we introduce visual trace prompting, a simple yet effective approach to facilitate VLA models' spatial-temporal awareness for action prediction by encoding state-action trajectories visually. We develop a new TraceVLA model by finetuning OpenVLA on our own collected dataset of 150K robot manipulation trajectories using visual trace prompting. Evaluations of TraceVLA across 137 configurations in SimplerEnv and 4 tasks on a physical WidowX robot demonstrate state-of-the-art performance, outperforming OpenVLA by 10% on SimplerEnv and 3.5x on real-robot tasks and exhibiting robust generalization across diverse embodiments and scenarios. To further validate the effectiveness and generality of our method, we present a compact VLA model based on 4B Phi-3-Vision, pretrained on the Open-X-Embodiment and finetuned on our dataset, rivals the 7B OpenVLA baseline while significantly improving inference efficiency.
>
---
#### [replaced 005] Mechanically Programming the Cross-Sectional Shape of Soft Growing Robotic Structures for Patient Transfer
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11593v2](http://arxiv.org/pdf/2505.11593v2)**

> **作者:** O. Godson Osele; Kentaro Barhydt; Teagan Sullivan; H. Harry Asada; Allison M. Okamura
>
> **摘要:** Pneumatic soft everting robotic structures have the potential to facilitate human transfer tasks due to their ability to grow underneath humans without sliding friction and their utility as a flexible sling when deflated. Tubular structures naturally yield circular cross-sections when inflated, whereas a robotic sling must be both thin enough to grow between them and their resting surface and wide enough to cradle the human. Recent works have achieved flattened cross-sections by including rigid components into the structure, but this reduces conformability to the human. We present a method of mechanically programming the cross-section of soft everting robotic structures using flexible strips that constrain radial expansion between points along the outer membrane. Our method enables simultaneously wide and thin profiles while maintaining the full multi-axis flexibility of traditional slings. We develop and validate a model relating the geometric design specifications to the fabrication parameters, and experimentally characterize their effects on growth rate. Finally, we prototype a soft growing robotic sling system and demonstrate its use for assisting a single caregiver in bed-to-chair patient transfer.
>
---
#### [replaced 006] An Integrated Visual Servoing Framework for Precise Robotic Pruning Operations in Modern Commercial Orchard
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.07309v2](http://arxiv.org/pdf/2504.07309v2)**

> **作者:** Dawood Ahmed; Basit Muhammad Imran; Martin Churuvija; Manoj Karkee
>
> **摘要:** This study presents a vision-guided robotic control system for automated fruit tree pruning applications. Traditional pruning practices are labor-intensive and limit agricultural efficiency and scalability, highlighting the need for advanced automation. A key challenge is the precise, robust positioning of the cutting tool in complex orchard environments, where dense branches and occlusions make target access difficult. To address this, an Intel RealSense D435 camera is mounted on the flange of a UR5e robotic arm and CoTracker3, a transformer-based point tracker, is utilized for visual servoing control that centers tracked points in the camera view. The system integrates proportional control with iterative inverse kinematics to achieve precise end-effector positioning. The system was validated in Gazebo simulation, achieving a 77.77% success rate within 5mm positional tolerance and 100% success rate within 10mm tolerance, with a mean end-effector error of 4.28 +/- 1.36 mm. The vision controller demonstrated robust performance across diverse target positions within the pixel workspace. The results validate the effectiveness of integrating vision-based tracking with kinematic control for precision agricultural tasks. Future work will focus on real-world implementation and the integration of force sensing for actual cutting operations.
>
---
#### [replaced 007] Beyond Winning Strategies: Admissible and Admissible Winning Strategies for Quantitative Reachability Games
- **分类: cs.GT; cs.FL; cs.LO; cs.RO; D.2.4; I.2.4; I.2.9**

- **链接: [http://arxiv.org/pdf/2408.13369v3](http://arxiv.org/pdf/2408.13369v3)**

> **作者:** Karan Muvvala; Qi Heng Ho; Morteza Lahijanian
>
> **备注:** Accepted to IJCAI 25
>
> **摘要:** Classical reactive synthesis approaches aim to synthesize a reactive system that always satisfies a given specifications. These approaches often reduce to playing a two-player zero-sum game where the goal is to synthesize a winning strategy. However, in many pragmatic domains, such as robotics, a winning strategy does not always exist, yet it is desirable for the system to make an effort to satisfy its requirements instead of "giving up". To this end, this paper investigates the notion of admissible strategies, which formalize "doing-your-best", in quantitative reachability games. We show that, unlike the qualitative case, quantitative admissible strategies are history-dependent even for finite payoff functions, making synthesis a challenging task. In addition, we prove that admissible strategies always exist but may produce undesirable optimistic behaviors. To mitigate this, we propose admissible winning strategies, which enforce the best possible outcome while being admissible. We show that both strategies always exist but are not memoryless. We provide necessary and sufficient conditions for the existence of both strategies and propose synthesis algorithms. Finally, we illustrate the strategies on gridworld and robot manipulator domains.
>
---
#### [replaced 008] Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01563v2](http://arxiv.org/pdf/2506.01563v2)**

> **作者:** Lingfan Bao; Yan Pan; Tianhu Peng; Kanoulas Dimitrios; Chengxu Zhou
>
> **备注:** 7 pages, 2 figures, IEEE conference paper
>
> **摘要:** Effective human-robot interaction requires robots to identify human intentions and generate expressive, socially appropriate motions in real-time. Existing approaches often rely on fixed motion libraries or computationally expensive generative models. We propose a hierarchical framework that combines intention-aware reasoning via in-context learning (ICL) with real-time motion generation using diffusion models. Our system introduces structured prompting with confidence scoring, fallback behaviors, and social context awareness to enable intention refinement and adaptive response. Leveraging large-scale motion datasets and efficient latent-space denoising, the framework generates diverse, physically plausible gestures suitable for dynamic humanoid interactions. Experimental validation on a physical platform demonstrates the robustness and social alignment of our method in realistic scenarios.
>
---
#### [replaced 009] TASTE-Rob: Advancing Video Generation of Task-Oriented Hand-Object Interaction for Generalizable Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.11423v2](http://arxiv.org/pdf/2503.11423v2)**

> **作者:** Hongxiang Zhao; Xingchen Liu; Mutian Xu; Yiming Hao; Weikai Chen; Xiaoguang Han
>
> **备注:** CVPR 2025; Project Page: https://taste-rob.github.io
>
> **摘要:** We address key limitations in existing datasets and models for task-oriented hand-object interaction video generation, a critical approach of generating video demonstrations for robotic imitation learning. Current datasets, such as Ego4D, often suffer from inconsistent view perspectives and misaligned interactions, leading to reduced video quality and limiting their applicability for precise imitation learning tasks. Towards this end, we introduce TASTE-Rob -- a pioneering large-scale dataset of 100,856 ego-centric hand-object interaction videos. Each video is meticulously aligned with language instructions and recorded from a consistent camera viewpoint to ensure interaction clarity. By fine-tuning a Video Diffusion Model (VDM) on TASTE-Rob, we achieve realistic object interactions, though we observed occasional inconsistencies in hand grasping postures. To enhance realism, we introduce a three-stage pose-refinement pipeline that improves hand posture accuracy in generated videos. Our curated dataset, coupled with the specialized pose-refinement framework, provides notable performance gains in generating high-quality, task-oriented hand-object interaction videos, resulting in achieving superior generalizable robotic manipulation. The TASTE-Rob dataset is publicly available to foster further advancements in the field, TASTE-Rob dataset and source code will be made publicly available on our website https://taste-rob.github.io.
>
---
#### [replaced 010] Diffusion Policies for Out-of-Distribution Generalization in Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.NE; cs.RO**

- **链接: [http://arxiv.org/pdf/2307.04726v4](http://arxiv.org/pdf/2307.04726v4)**

> **作者:** Suzan Ece Ada; Erhan Oztop; Emre Ugur
>
> **备注:** Published in IEEE RA-L with IROS presentation option (2024 IEEE/RSJ International Conference on Intelligent Robots and Systems), 8 pages, 7 figures
>
> **摘要:** Offline Reinforcement Learning (RL) methods leverage previous experiences to learn better policies than the behavior policy used for data collection. However, they face challenges handling distribution shifts due to the lack of online interaction during training. To this end, we propose a novel method named State Reconstruction for Diffusion Policies (SRDP) that incorporates state reconstruction feature learning in the recent class of diffusion policies to address the problem of out-of-distribution (OOD) generalization. Our method promotes learning of generalizable state representation to alleviate the distribution shift caused by OOD states. To illustrate the OOD generalization and faster convergence of SRDP, we design a novel 2D Multimodal Contextual Bandit environment and realize it on a 6-DoF real-world UR10 robot, as well as in simulation, and compare its performance with prior algorithms. In particular, we show the importance of the proposed state reconstruction via ablation studies. In addition, we assess the performance of our model on standard continuous control benchmarks (D4RL), namely the navigation of an 8-DoF ant and forward locomotion of half-cheetah, hopper, and walker2d, achieving state-of-the-art results. Finally, we demonstrate that our method can achieve 167% improvement over the competing baseline on a sparse continuous control navigation task where various regions of the state space are removed from the offline RL dataset, including the region encapsulating the goal.
>
---
#### [replaced 011] Is Your Imitation Learning Policy Better than Mine? Policy Comparison with Near-Optimal Stopping
- **分类: cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2503.10966v4](http://arxiv.org/pdf/2503.10966v4)**

> **作者:** David Snyder; Asher James Hancock; Apurva Badithela; Emma Dixon; Patrick Miller; Rares Andrei Ambrus; Anirudha Majumdar; Masha Itkina; Haruki Nishimura
>
> **备注:** 14 + 5 pages, 10 figures, 4 tables. Accepted to RSS 2025
>
> **摘要:** Imitation learning has enabled robots to perform complex, long-horizon tasks in challenging dexterous manipulation settings. As new methods are developed, they must be rigorously evaluated and compared against corresponding baselines through repeated evaluation trials. However, policy comparison is fundamentally constrained by a small feasible sample size (e.g., 10 or 50) due to significant human effort and limited inference throughput of policies. This paper proposes a novel statistical framework for rigorously comparing two policies in the small sample size regime. Prior work in statistical policy comparison relies on batch testing, which requires a fixed, pre-determined number of trials and lacks flexibility in adapting the sample size to the observed evaluation data. Furthermore, extending the test with additional trials risks inducing inadvertent p-hacking, undermining statistical assurances. In contrast, our proposed statistical test is sequential, allowing researchers to decide whether or not to run more trials based on intermediate results. This adaptively tailors the number of trials to the difficulty of the underlying comparison, saving significant time and effort without sacrificing probabilistic correctness. Extensive numerical simulation and real-world robot manipulation experiments show that our test achieves near-optimal stopping, letting researchers stop evaluation and make a decision in a near-minimal number of trials. Specifically, it reduces the number of evaluation trials by up to 32% as compared to state-of-the-art baselines, while preserving the probabilistic correctness and statistical power of the comparison. Moreover, our method is strongest in the most challenging comparison instances (requiring the most evaluation trials); in a multi-task comparison scenario, we save the evaluator more than 160 simulation rollouts.
>
---
#### [replaced 012] Marginalizing and Conditioning Gaussians onto Linear Approximations of Smooth Manifolds with Applications in Robotics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.09871v3](http://arxiv.org/pdf/2409.09871v3)**

> **作者:** Zi Cong Guo; James R. Forbes; Timothy D. Barfoot
>
> **备注:** Final version in IEEE ICRA 2025 (winner of the Best Conference Paper Award)
>
> **摘要:** We present closed-form expressions for marginalizing and conditioning Gaussians onto linear manifolds, and demonstrate how to apply these expressions to smooth nonlinear manifolds through linearization. Although marginalization and conditioning onto axis-aligned manifolds are well-established procedures, doing so onto non-axis-aligned manifolds is not as well understood. We demonstrate the utility of our expressions through three applications: 1) approximation of the projected normal distribution, where the quality of our linearized approximation increases as problem nonlinearity decreases; 2) covariance extraction in Koopman SLAM, where our covariances are shown to be consistent on a real-world dataset; and 3) covariance extraction in constrained GTSAM, where our covariances are shown to be consistent in simulation.
>
---
#### [replaced 013] RoboOS: A Hierarchical Embodied Framework for Cross-Embodiment and Multi-Agent Collaboration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.03673v2](http://arxiv.org/pdf/2505.03673v2)**

> **作者:** Huajie Tan; Xiaoshuai Hao; Cheng Chi; Minglan Lin; Yaoxu Lyu; Mingyu Cao; Dong Liang; Zhuo Chen; Mengsi Lyu; Cheng Peng; Chenrui He; Yulong Ao; Yonghua Lin; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** The dawn of embodied intelligence has ushered in an unprecedented imperative for resilient, cognition-enabled multi-agent collaboration across next-generation ecosystems, revolutionizing paradigms in autonomous manufacturing, adaptive service robotics, and cyber-physical production architectures. However, current robotic systems face significant limitations, such as limited cross-embodiment adaptability, inefficient task scheduling, and insufficient dynamic error correction. While End-to-end VLA models demonstrate inadequate long-horizon planning and task generalization, hierarchical VLA models suffer from a lack of cross-embodiment and multi-agent coordination capabilities. To address these challenges, we introduce RoboOS, the first open-source embodied system built on a Brain-Cerebellum hierarchical architecture, enabling a paradigm shift from single-agent to multi-agent intelligence. Specifically, RoboOS consists of three key components: (1) Embodied Brain Model (RoboBrain), a MLLM designed for global perception and high-level decision-making; (2) Cerebellum Skill Library, a modular, plug-and-play toolkit that facilitates seamless execution of multiple skills; and (3) Real-Time Shared Memory, a spatiotemporal synchronization mechanism for coordinating multi-agent states. By integrating hierarchical information flow, RoboOS bridges Embodied Brain and Cerebellum Skill Library, facilitating robust planning, scheduling, and error correction for long-horizon tasks, while ensuring efficient multi-agent collaboration through Real-Time Shared Memory. Furthermore, we enhance edge-cloud communication and cloud-based distributed inference to facilitate high-frequency interactions and enable scalable deployment. Extensive real-world experiments across various scenarios, demonstrate RoboOS's versatility in supporting heterogeneous embodiments. Project website: https://github.com/FlagOpen/RoboOS
>
---
#### [replaced 014] Haptic bilateral teleoperation system for free-hand dental procedures
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.21288v2](http://arxiv.org/pdf/2503.21288v2)**

> **作者:** Lorenzo Pagliara; Enrico Ferrentino; Andrea Chiacchio; Giovanni Russo
>
> **备注:** 13 pages, 11 figures
>
> **摘要:** Free-hand dental procedures are typically repetitive, time-consuming and require high precision and manual dexterity. Robots can play a key role in improving procedural accuracy and safety, enhancing patient comfort, and reducing operator workload. However, robotic solutions for free-hand procedures remain limited or completely lacking, and their acceptance is still low. To address this gap, we develop a haptic bilateral teleoperation system (HBTS) for free-hand dental procedures (FH-HBTS). The system includes a dedicated mechanical end-effector, compatible with standard clinical tools, and equipped with an endoscopic camera for improved visibility of the intervention site. By ensuring motion and force correspondence between the operator's actions and the robot's movements, monitored through visual feedback, we enhance the operator's sensory awareness and motor accuracy. Furthermore, recognizing the need to ensure procedural safety, we limit interaction forces by scaling the motion references provided to the admittance controller based solely on measured contact forces. This ensures effective force limitation in all contact states without requiring prior knowledge of the environment. The proposed FH-HBTS is validated in a dental scaling procedure using a dental phantom. The results show that the system improves the naturalness, safety, and accuracy of teleoperation, highlighting its potential to enhance free-hand dental procedures.
>
---
#### [replaced 015] DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.21969v3](http://arxiv.org/pdf/2505.21969v3)**

> **作者:** Tianjun Gu; Linfeng Li; Xuhong Wang; Chenghua Gong; Jingyu Gong; Zhizhong Zhang; Yuan Xie; Lizhuang Ma; Xin Tan
>
> **摘要:** Adaptive navigation in unfamiliar environments is crucial for household service robots but remains challenging due to the need for both low-level path planning and high-level scene understanding. While recent vision-language model (VLM) based zero-shot approaches reduce dependence on prior maps and scene-specific training data, they face significant limitations: spatiotemporal discontinuity from discrete observations, unstructured memory representations, and insufficient task understanding leading to navigation failures. We propose DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art performance on both success rate (SR) and success weighted by path length (SPL) metrics, significantly outperforming existing methods. We also introduce a new evaluation metric (AORI) to assess navigation intelligence better. Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot autonomous navigation without requiring prior map building or pre-training.
>
---
#### [replaced 016] ArtVIP: Articulated Digital Assets of Visual Realism, Modular Interaction, and Physical Fidelity for Robot Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.04941v2](http://arxiv.org/pdf/2506.04941v2)**

> **作者:** Zhao Jin; Zhengping Che; Zhen Zhao; Kun Wu; Yuheng Zhang; Yinuo Zhao; Zehui Liu; Qiang Zhang; Xiaozhu Ju; Jing Tian; Yousong Xue; Jian Tang
>
> **摘要:** Robot learning increasingly relies on simulation to advance complex ability such as dexterous manipulations and precise interactions, necessitating high-quality digital assets to bridge the sim-to-real gap. However, existing open-source articulated-object datasets for simulation are limited by insufficient visual realism and low physical fidelity, which hinder their utility for training models mastering robotic tasks in real world. To address these challenges, we introduce ArtVIP, a comprehensive open-source dataset comprising high-quality digital-twin articulated objects, accompanied by indoor-scene assets. Crafted by professional 3D modelers adhering to unified standards, ArtVIP ensures visual realism through precise geometric meshes and high-resolution textures, while physical fidelity is achieved via fine-tuned dynamic parameters. Meanwhile, the dataset pioneers embedded modular interaction behaviors within assets and pixel-level affordance annotations. Feature-map visualization and optical motion capture are employed to quantitatively demonstrate ArtVIP's visual and physical fidelity, with its applicability validated across imitation learning and reinforcement learning experiments. Provided in USD format with detailed production guidelines, ArtVIP is fully open-source, benefiting the research community and advancing robot learning research. Our project is at https://x-humanoid-artvip.github.io/ .
>
---
#### [replaced 017] Field Report on Ground Penetrating Radar for Localization at the Mars Desert Research Station
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.15455v2](http://arxiv.org/pdf/2504.15455v2)**

> **作者:** Anja Sheppard; Katherine A. Skinner
>
> **备注:** Presented at the ICRA Workshop on Field Robotics 2025
>
> **摘要:** In this field report, we detail the lessons learned from our field expedition to collect Ground Penetrating Radar (GPR) data in a Mars analog environment for the purpose of validating GPR localization techniques in rugged environments. Planetary rovers are already equipped with GPR for geologic subsurface characterization. GPR has been successfully used to localize vehicles on Earth, but it has not yet been explored as another modality for localization on a planetary rover. Leveraging GPR for localization can aid in efficient and robust rover pose estimation. In order to demonstrate localizing GPR in a Mars analog environment, we collected over 50 individual survey trajectories during a two-week period at the Mars Desert Research Station (MDRS). In this report, we discuss our methodology, lessons learned, and opportunities for future work.
>
---
#### [replaced 018] A Physics-informed End-to-End Occupancy Framework for Motion Planning of Autonomous Vehicles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.07855v2](http://arxiv.org/pdf/2505.07855v2)**

> **作者:** Shuqi Shen; Junjie Yang; Hongliang Lu; Hui Zhong; Qiming Zhang; Xinhu Zheng
>
> **摘要:** Accurate and interpretable motion planning is essential for autonomous vehicles (AVs) navigating complex and uncertain environments. While recent end-to-end occupancy prediction methods have improved environmental understanding, they typically lack explicit physical constraints, limiting safety and generalization. In this paper, we propose a unified end-to-end framework that integrates verifiable physical rules into the occupancy learning process. Specifically, we embed artificial potential fields (APF) as physics-informed guidance during network training to ensure that predicted occupancy maps are both data-efficient and physically plausible. Our architecture combines convolutional and recurrent neural networks to capture spatial and temporal dependencies while preserving model flexibility. Experimental results demonstrate that our method improves task completion rate, safety margins, and planning efficiency across diverse driving scenarios, confirming its potential for reliable deployment in real-world AV systems.
>
---
#### [replaced 019] Nocturnal eye inspired liquid to gas phase change soft actuator with Laser-Induced-Graphene: enhanced environmental light harvesting and photothermal conversion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.11930v3](http://arxiv.org/pdf/2501.11930v3)**

> **作者:** Maina Sogabe; Youhyun Kim; Hiroki Miyazako; Kenji Kawashima
>
> **备注:** 33pages, 10 figures, journal paper
>
> **摘要:** Robotic systems' mobility is constrained by power sources and wiring. While pneumatic actuators remain tethered to air supplies, we developed a new actuator utilizing light energy. Inspired by nocturnal animals' eyes, we designed a bilayer soft actuator incorporating Laser-Induced Graphene (LIG) on the inner surface of a silicone layer. This design maintains silicone's transparency and flexibility while achieving 54% faster response time compared to conventional actuators through enhanced photothermal conversion.
>
---
#### [replaced 020] Adaptive Locomotion on Mud through Proprioceptive Sensing of Substrate Properties
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.19607v2](http://arxiv.org/pdf/2504.19607v2)**

> **作者:** Shipeng Liu; Jiaze Tang; Siyuan Meng; Feifei Qian
>
> **备注:** 12 pages, 8 figures. Published in Robotics: Science and Systems (RSS'25)
>
> **摘要:** Muddy terrains present significant challenges for terrestrial robots, as subtle changes in composition and water content can lead to large variations in substrate strength and force responses, causing the robot to slip or get stuck. This paper presents a method to estimate mud properties using proprioceptive sensing, enabling a flipper-driven robot to adapt its locomotion through muddy substrates of varying strength. First, we characterize mud reaction forces through actuator current and position signals from a statically mounted robotic flipper. We use the measured force to determine key coefficients that characterize intrinsic mud properties. The proprioceptively estimated coefficients match closely with measurements from a lab-grade load cell, validating the effectiveness of the proposed method. Next, we extend the method to a locomoting robot to estimate mud properties online as it crawls across different mud mixtures. Experimental data reveal that mud reaction forces depend sensitively on robot motion, requiring joint analysis of robot movement with proprioceptive force to determine mud properties correctly. Lastly, we deploy this method in a flipper-driven robot moving across muddy substrates of varying strengths, and demonstrate that the proposed method allows the robot to use the estimated mud properties to adapt its locomotion strategy, and successfully avoid locomotion failures. Our findings highlight the potential of proprioception-based terrain sensing to enhance robot mobility in complex, deformable natural environments, paving the way for more robust field exploration capabilities.
>
---
