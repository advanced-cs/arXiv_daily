# 机器人 cs.RO

- **最新发布 59 篇**

- **更新 28 篇**

## 最新发布

#### [new 001] Parallel Heuristic Search as Inference for Actor-Critic Reinforcement Learning Models
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，解决机器人操作中的高效推理问题。提出并行启发式搜索方法，利用Actor-Critic结构提升运动规划与交互效率。**

- **链接: [http://arxiv.org/pdf/2509.25402v1](http://arxiv.org/pdf/2509.25402v1)**

> **作者:** Hanlan Yang; Itamar Mishani; Luca Pivetti; Zachary Kingston; Maxim Likhachev
>
> **备注:** Submitted for Publication
>
> **摘要:** Actor-Critic models are a class of model-free deep reinforcement learning (RL) algorithms that have demonstrated effectiveness across various robot learning tasks. While considerable research has focused on improving training stability and data sampling efficiency, most deployment strategies have remained relatively simplistic, typically relying on direct actor policy rollouts. In contrast, we propose \pachs{} (\textit{P}arallel \textit{A}ctor-\textit{C}ritic \textit{H}euristic \textit{S}earch), an efficient parallel best-first search algorithm for inference that leverages both components of the actor-critic architecture: the actor network generates actions, while the critic network provides cost-to-go estimates to guide the search. Two levels of parallelism are employed within the search -- actions and cost-to-go estimates are generated in batches by the actor and critic networks respectively, and graph expansion is distributed across multiple threads. We demonstrate the effectiveness of our approach in robotic manipulation tasks, including collision-free motion planning and contact-rich interactions such as non-prehensile pushing. Visit p-achs.github.io for demonstrations and examples.
>
---
#### [new 002] OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于人形机器人运动控制任务，解决人机动作迁移中的物理不真实和交互丢失问题，提出OmniRetarget方法生成保留交互关系的高质量数据。**

- **链接: [http://arxiv.org/pdf/2509.26633v1](http://arxiv.org/pdf/2509.26633v1)**

> **作者:** Lujie Yang; Xiaoyu Huang; Zhen Wu; Angjoo Kanazawa; Pieter Abbeel; Carmelo Sferrazza; C. Karen Liu; Rocky Duan; Guanya Shi
>
> **备注:** Project website: https://omniretarget.github.io
>
> **摘要:** A dominant paradigm for teaching humanoid robots complex skills is to retarget human motions as kinematic references to train reinforcement learning (RL) policies. However, existing retargeting pipelines often struggle with the significant embodiment gap between humans and robots, producing physically implausible artifacts like foot-skating and penetration. More importantly, common retargeting methods neglect the rich human-object and human-environment interactions essential for expressive locomotion and loco-manipulation. To address this, we introduce OmniRetarget, an interaction-preserving data generation engine based on an interaction mesh that explicitly models and preserves the crucial spatial and contact relationships between an agent, the terrain, and manipulated objects. By minimizing the Laplacian deformation between the human and robot meshes while enforcing kinematic constraints, OmniRetarget generates kinematically feasible trajectories. Moreover, preserving task-relevant interactions enables efficient data augmentation, from a single demonstration to different robot embodiments, terrains, and object configurations. We comprehensively evaluate OmniRetarget by retargeting motions from OMOMO, LAFAN1, and our in-house MoCap datasets, generating over 8-hour trajectories that achieve better kinematic constraint satisfaction and contact preservation than widely used baselines. Such high-quality data enables proprioceptive RL policies to successfully execute long-horizon (up to 30 seconds) parkour and loco-manipulation skills on a Unitree G1 humanoid, trained with only 5 reward terms and simple domain randomization shared by all tasks, without any learning curriculum.
>
---
#### [new 003] On the Conic Complementarity of Planar Contacts
- **分类: cs.RO**

- **简介: 该论文属于机器人接触建模任务，解决离散与连续接触模型的统一问题。通过提出平面Signorini条件，建立了一种新的接触力学框架，用于更准确地模拟和控制机器人接触行为。**

- **链接: [http://arxiv.org/pdf/2509.25999v1](http://arxiv.org/pdf/2509.25999v1)**

> **作者:** Yann de Mont-Marin; Louis Montaut; Jean Ponce; Martial Hebert; Justin Carpentier
>
> **摘要:** We present a unifying theoretical result that connects two foundational principles in robotics: the Signorini law for point contacts, which underpins many simulation methods for preventing object interpenetration, and the center of pressure (also known as the zero-moment point), a key concept used in, for instance, optimization-based locomotion control. Our contribution is the planar Signorini condition, a conic complementarity formulation that models general planar contacts between rigid bodies. We prove that this formulation is equivalent to enforcing the punctual Signorini law across an entire contact surface, thereby bridging the gap between discrete and continuous contact models. A geometric interpretation reveals that the framework naturally captures three physical regimes -sticking, separating, and tilting-within a unified complementarity structure. This leads to a principled extension of the classical center of pressure, which we refer to as the extended center of pressure. By establishing this connection, our work provides a mathematically consistent and computationally tractable foundation for handling planar contacts, with implications for both the accurate simulation of contact dynamics and the design of advanced control and optimization algorithms in locomotion and manipulation.
>
---
#### [new 004] VLA Model Post-Training via Action-Chunked PPO and Self Behavior Cloning
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型的后训练任务，旨在解决稀疏奖励和训练不稳定问题。通过结合动作分块PPO与自我行为克隆，提升策略稳定性和效率。**

- **链接: [http://arxiv.org/pdf/2509.25718v1](http://arxiv.org/pdf/2509.25718v1)**

> **作者:** Si-Cheng Wang; Tian-Yu Xiang; Xiao-Hu Zhou; Mei-Jiang Gui; Xiao-Liang Xie; Shi-Qi Liu; Shuang-Yi Wang; Ao-Qun Jin; Zeng-Guang Hou
>
> **摘要:** Reinforcement learning (RL) is a promising avenue for post-training vision-language-action (VLA) models, but practical deployment is hindered by sparse rewards and unstable training. This work mitigates these challenges by introducing an action chunk based on proximal policy optimization (PPO) with behavior cloning using self-collected demonstrations. Aggregating consecutive actions into chunks improves the temporal consistency of the policy and the density of informative feedback. In addition, an auxiliary behavior cloning loss is applied with a dynamically updated demonstration buffer that continually collects high-quality task trials during training. The relative weight between the action-chunked PPO objective and the self behavior clone auxiliary loss is adapted online to stabilize the post-training process. Experiments on the MetaWorld benchmark indicate improved performance over supervised fine-tuning, achieving a high success rate (0.93) and few steps to success (42.17). These results demonstrate the viability of RL for VLA post-training and help lay the groundwork for downstream VLA applications.
>
---
#### [new 005] OmniNav: A Unified Framework for Prospective Exploration and Visual-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决多模态导航与探索问题。提出OmniNav框架，统一处理指令、目标和探索任务，提升导航精度与通用性。**

- **链接: [http://arxiv.org/pdf/2509.25687v1](http://arxiv.org/pdf/2509.25687v1)**

> **作者:** Xinda Xue; Junjun Hu; Minghua Luo; Xie Shichao; Jintao Chen; Zixun Xie; Quan Kuichen; Guo Wei; Mu Xu; Zedong Chu
>
> **摘要:** Embodied navigation presents a core challenge for intelligent robots, requiring the comprehension of visual environments, natural language instructions, and autonomous exploration. Existing models often fall short in offering a unified solution across diverse navigation paradigms, resulting in low success rates and limited generalization. We introduce OmniNav, a unified framework addressing instruct-goal, object-goal, point-goal navigation, and frontier-based exploration within a single architecture. Our approach features a lightweight, low-latency policy that accurately predicts continuous-space waypoints (coordinates and orientations). This policy surpasses action-chunk methods in precision and supports real-world deployment at control frequencies up to 5 Hz. Architecturally, OmniNav employs a fast-slow system design: a fast module generates waypoints using short-horizon visual context and subtasks, while a slow module performs deliberative planning with long-horizon observations and candidate frontiers to select subsequent subgoals and subtasks. This collaboration enhances path efficiency and maintains trajectory coherence, particularly in exploration and memory-intensive scenarios. Crucially, we identify that the primary bottleneck isn't merely navigation policy learning, but a robust understanding of general instructions and objects. To boost generalization, OmniNav integrates large-scale, general-purpose training datasets, including those for image captioning and visual recognition, into a joint multi-task regimen. This significantly improves success rates and robustness. Extensive experiments confirm OmniNav's state-of-the-art performance across various navigation benchmarks, with real-world deployment further validating its efficacy. OmniNav provides practical insights for embodied navigation, charting a scalable path towards versatile, highly generalizable robotic intelligence.
>
---
#### [new 006] SRMP: Search-Based Robot Motion Planning Library
- **分类: cs.RO**

- **简介: 该论文提出SRMP，一个用于机器人运动规划的框架，解决高可靠性与可重复性问题，支持多机器人操作并集成主流仿真工具。**

- **链接: [http://arxiv.org/pdf/2509.25352v1](http://arxiv.org/pdf/2509.25352v1)**

> **作者:** Itamar Mishani; Yorai Shaoul; Ramkumar Natarajan; Jiaoyang Li; Maxim Likhachev
>
> **备注:** Submitted for Publication
>
> **摘要:** Motion planning is a critical component in any robotic system. Over the years, powerful tools like the Open Motion Planning Library (OMPL) have been developed, offering numerous motion planning algorithms. However, existing frameworks often struggle to deliver the level of predictability and repeatability demanded by high-stakes applications -- ranging from ensuring safety in industrial environments to the creation of high-quality motion datasets for robot learning. Complementing existing tools, we introduce SRMP (Search-based Robot Motion Planning), a new software framework tailored for robotic manipulation. SRMP distinguishes itself by generating consistent and reliable trajectories, and is the first software tool to offer motion planning algorithms for multi-robot manipulation tasks. SRMP easily integrates with major simulators, including MuJoCo, Sapien, Genesis, and PyBullet via a Python and C++ API. SRMP includes a dedicated MoveIt! plugin that enables immediate deployment on robot hardware and seamless integration with existing pipelines. Through extensive evaluations, we demonstrate in this paper that SRMP not only meets the rigorous demands of industrial and safety-critical applications but also sets a new standard for consistency in motion planning across diverse robotic systems. Visit srmp.readthedocs.io for SRMP documentation and tutorials.
>
---
#### [new 007] SARM: Stage-Aware Reward Modeling for Long Horizon Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于长期机器人操作任务，解决变形物体操作中奖励信号不稳定的问题。提出阶段感知的奖励建模方法，提升策略训练效果。**

- **链接: [http://arxiv.org/pdf/2509.25358v1](http://arxiv.org/pdf/2509.25358v1)**

> **作者:** Qianzhong Chen; Justin Yu; Mac Schwager; Pieter Abbeel; Fred Shentu; Philipp Wu
>
> **摘要:** Large-scale robot learning has recently shown promise for enabling robots to perform complex tasks by integrating perception, control, and language understanding. Yet, it struggles with long-horizon, contact-rich manipulation such as deformable object handling, where demonstration quality is inconsistent. Reward modeling offers a natural solution: by providing grounded progress signals, it transforms noisy demonstrations into stable supervision that generalizes across diverse trajectories. We introduce a stage-aware, video-based reward modeling framework that jointly predicts high-level task stages and fine-grained progress. Reward labels are automatically derived from natural language subtask annotations, ensuring consistent progress estimation across variable-length demonstrations. This design overcomes frame-index labeling, which fails in variable-duration tasks like folding a T-shirt. Our reward model demonstrates robustness to variability, generalization to out-of-distribution settings, and strong utility for policy training. Building on it, we propose Reward-Aligned Behavior Cloning (RA-BC), which filters high-quality data and reweights samples by reward. Experiments show the reward model alone outperforms baselines on validation and real robot rollouts. Integrated into RA-BC, our approach achieves 83\% success on folding T-shirts from the flattened state and 67\% from the crumpled state -- far surpassing vanilla behavior cloning, which attains only 8\% and 0\% success. Overall, our results highlight reward modeling as a key enabler for scalable, annotation-efficient, and robust imitation learning in long-horizon manipulation.
>
---
#### [new 008] dVLA: Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作（VLA）模型任务，旨在统一视觉、语言和控制，解决机器人执行复杂任务的问题。论文提出dVLA模型，实现高效跨模态推理与部署。**

- **链接: [http://arxiv.org/pdf/2509.25681v1](http://arxiv.org/pdf/2509.25681v1)**

> **作者:** Junjie Wen; Minjie Zhu; Jiaming Liu; Zhiyuan Liu; Yicun Yang; Linfeng Zhang; Shanghang Zhang; Yichen Zhu; Yi Xu
>
> **备注:** technique report
>
> **摘要:** Vision-Language-Action (VLA) models are emerging as a next-generation paradigm for robotics. We introduce dVLA, a diffusion-based VLA that leverages a multimodal chain-of-thought to unify visual perception, language reasoning, and robotic control in a single system. dVLA jointly optimizes perception, language understanding, and action under a single diffusion objective, enabling stronger cross-modal reasoning and better generalization to novel instructions and objects. For practical deployment, we mitigate inference latency by incorporating two acceleration strategies, a prefix attention mask and KV caching, yielding up to around times speedup at test-time inference. We evaluate dVLA in both simulation and the real world: on the LIBERO benchmark, it achieves state-of-the-art performance with a 96.4% average success rate, consistently surpassing both discrete and continuous action policies; on a real Franka robot, it succeeds across a diverse task suite, including a challenging bin-picking task that requires multi-step planning, demonstrating robust real-world performance. Together, these results underscore the promise of unified diffusion frameworks for practical, high-performance VLA robotics.
>
---
#### [new 009] Hierarchical Diffusion Motion Planning with Task-Conditioned Uncertainty-Aware Priors
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，旨在解决轨迹生成中的不确定性与任务对齐问题。通过引入任务条件的结构化高斯噪声模型，提升轨迹成功率与平滑性。**

- **链接: [http://arxiv.org/pdf/2509.25685v1](http://arxiv.org/pdf/2509.25685v1)**

> **作者:** Amelie Minji Kim; Anqi Wu; Ye Zhao
>
> **摘要:** We propose a novel hierarchical diffusion planner that embeds task and motion structure directly in the noise model. Unlike standard diffusion-based planners that use zero-mean, isotropic Gaussian noise, we employ a family of task-conditioned structured Gaussians whose means and covariances are derived from Gaussian Process Motion Planning (GPMP): sparse, task-centric key states or their associated timings (or both) are treated as noisy observations to produce a prior instance. We first generalize the standard diffusion process to biased, non-isotropic corruption with closed-form forward and posterior expressions. Building on this, our hierarchy separates prior instantiation from trajectory denoising: the upper level instantiates a task-conditioned structured Gaussian (mean and covariance), and the lower level denoises the full trajectory under that fixed prior. Experiments on Maze2D goal-reaching and KUKA block stacking show improved success rates, smoother trajectories, and stronger task alignment compared to isotropic baselines. Ablation studies indicate that explicitly structuring the corruption process offers benefits beyond simply conditioning the neural network. Overall, our method concentrates probability mass of prior near feasible, smooth, and semantically meaningful trajectories while maintaining tractability. Our project page is available at https://hta-diffusion.github.io.
>
---
#### [new 010] Unwinding Rotations Reduces VR Sickness in Nonsimulated Immersive Telepresence
- **分类: cs.RO**

- **简介: 该论文属于沉浸式远程交互任务，旨在解决VR眩晕问题。通过实验证明，解耦机器人旋转可提升用户舒适度且不影响任务表现。**

- **链接: [http://arxiv.org/pdf/2509.26439v1](http://arxiv.org/pdf/2509.26439v1)**

> **作者:** Filip Kulisiewicz; Basak Sakcak; Evan G. Center; Juho Kalliokoski; Katherine J. Mimnaugh; Steven M. LaValle; Timo Ojala
>
> **备注:** 24th IEEE International Symposium on Mixed and Augmented Reality (ISMAR)
>
> **摘要:** Immersive telepresence, when a user views the video stream of a $360^\circ$ camera in a remote environment using a Head Mounted Display (HMD), has great potential to improve the sense of being in a remote environment. In most cases of immersive robotic telepresence, the camera is mounted on a mobile robot which increases the portion of the environment that the remote user can explore. However, robot motions can induce unpleasant symptoms associated with Virtual Reality (VR) sickness, degrading the overall user experience. Previous research has shown that unwinding the rotations of the robot, that is, decoupling the rotations that the camera undergoes due to robot motions from what is seen by the user, can increase user comfort and reduce VR sickness. However, that work considered a virtual environment and a simulated robot. In this work, to test whether the same hypotheses hold when the video stream from a real camera is used, we carried out a user study $(n=36)$ in which the unwinding rotations method was compared against coupled rotations in a task completed through a panoramic camera mounted on a robotic arm. Furthermore, within an inspection task which involved translations and rotations in three dimensions, we tested whether unwinding the robot rotations impacted the performance of users. The results show that the users found the unwinding rotations method to be more comfortable and preferable, and that a reduced level of VR sickness can be achieved without a significant impact on task performance.
>
---
#### [new 011] Side Scan Sonar-based SLAM for Autonomous Algae Farm Monitoring
- **分类: cs.RO**

- **简介: 该论文属于水下SLAM任务，旨在解决AUV在海藻农场的精确定位与地图构建问题，通过改进侧扫声呐数据处理提升导航精度。**

- **链接: [http://arxiv.org/pdf/2509.26121v1](http://arxiv.org/pdf/2509.26121v1)**

> **作者:** Julian Valdez; Ignacio Torroba; John Folkesson; Ivan Stenius
>
> **摘要:** The transition of seaweed farming to an alternative food source on an industrial scale relies on automating its processes through smart farming, equivalent to land agriculture. Key to this process are autonomous underwater vehicles (AUVs) via their capacity to automate crop and structural inspections. However, the current bottleneck for their deployment is ensuring safe navigation within farms, which requires an accurate, online estimate of the AUV pose and map of the infrastructure. To enable this, we propose an efficient side scan sonar-based (SSS) simultaneous localization and mapping (SLAM) framework that exploits the geometry of kelp farms via modeling structural ropes in the back-end as sequences of individual landmarks from each SSS ping detection, instead of combining detections into elongated representations. Our method outperforms state of the art solutions in hardware in the loop (HIL) experiments on a real AUV survey in a kelp farm. The framework and dataset can be found at https://github.com/julRusVal/sss_farm_slam.
>
---
#### [new 012] BEV-VLM: Trajectory Planning via Unified BEV Abstraction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶轨迹规划任务，旨在提升规划精度与安全性。通过融合多模态数据生成BEV特征图，并结合VLM进行轨迹规划，实现更准确的环境理解与避障。**

- **链接: [http://arxiv.org/pdf/2509.25249v1](http://arxiv.org/pdf/2509.25249v1)**

> **作者:** Guancheng Chen; Sheng Yang; Tong Zhan; Jian Wang
>
> **摘要:** This paper introduces BEV-VLM, a novel framework for trajectory planning in autonomous driving that leverages Vision-Language Models (VLMs) with Bird's-Eye View (BEV) feature maps as visual inputs. Unlike conventional approaches that rely solely on raw visual data such as camera images, our method utilizes highly compressed and informative BEV representations, which are generated by fusing multi-modal sensor data (e.g., camera and LiDAR) and aligning them with HD Maps. This unified BEV-HD Map format provides a geometrically consistent and rich scene description, enabling VLMs to perform accurate trajectory planning. Experimental results on the nuScenes dataset demonstrate 44.8% improvements in planning accuracy and complete collision avoidance. Our work highlights that VLMs can effectively interpret processed visual representations like BEV features, expanding their applicability beyond raw images in trajectory planning.
>
---
#### [new 013] Conflict-Based Search and Prioritized Planning for Multi-Agent Path Finding Among Movable Obstacles
- **分类: cs.RO**

- **简介: 该论文研究多智能体路径规划问题（M-PAMO），解决在静态和可移动障碍物中寻找无碰撞路径的任务。通过结合CBS、PP和PAMO*方法进行优化。**

- **链接: [http://arxiv.org/pdf/2509.26050v1](http://arxiv.org/pdf/2509.26050v1)**

> **作者:** Shaoli Hu; Shizhe Zhao; Zhongqiang Ren
>
> **摘要:** This paper investigates Multi-Agent Path Finding Among Movable Obstacles (M-PAMO), which seeks collision-free paths for multiple agents from their start to goal locations among static and movable obstacles. M-PAMO arises in logistics and warehouses where mobile robots are among unexpected movable objects. Although Multi-Agent Path Finding (MAPF) and single-agent Path planning Among Movable Obstacles (PAMO) were both studied, M-PAMO remains under-explored. Movable obstacles lead to new fundamental challenges as the state space, which includes both agents and movable obstacles, grows exponentially with respect to the number of agents and movable obstacles. In particular, movable obstacles often closely couple agents together spatially and temporally. This paper makes a first attempt to adapt and fuse the popular Conflict-Based Search (CBS) and Prioritized Planning (PP) for MAPF, and a recent single-agent PAMO planner called PAMO*, together to address M-PAMO. We compare their performance with up to 20 agents and hundreds of movable obstacles, and show the pros and cons of these approaches.
>
---
#### [new 014] Reinforced Embodied Planning with Verifiable Reward for Real-World Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决长时序语言指令执行问题。通过构建框架REVER和模型RoboFarseer，提升机器人规划与验证能力。**

- **链接: [http://arxiv.org/pdf/2509.25852v1](http://arxiv.org/pdf/2509.25852v1)**

> **作者:** Zitong Bo; Yue Hu; Jinming Ma; Mingliang Zhou; Junhui Yin; Yachen Kang; Yuqi Liu; Tong Wu; Diyun Xiang; Hao Chen
>
> **摘要:** Enabling robots to execute long-horizon manipulation tasks from free-form language instructions remains a fundamental challenge in embodied AI. While vision-language models (VLMs) have shown promise as high-level planners, their deployment in the real world is hindered by two gaps: (i) the scarcity of large-scale, sequential manipulation data that couples natural language with multi-step action plans, and (ii) the absence of dense, interpretable rewards for fine-tuning VLMs on planning objectives. To address these issues, we propose REVER, a framework that empowers VLMs to generate and validate long-horizon manipulation plans from natural language instructions in real-world scenarios. Under REVER we train and release RoboFarseer, a VLM incentivized to emit chain-of-thought that perform temporal and spatial reasoning, ensuring physically plausible and logically coherent plans. To obtain training data, we leverage the Universal Manipulation Interface framework to capture hardware-agnostic demonstrations of atomic skills. An automated annotation engine converts each demonstration into vision-instruction-plan triplet. We introduce a verifiable reward that scores the generated plan by its ordered bipartite matching overlap with the ground-truth skill sequence. At run time, the fine-tuned VLM functions both as a planner and as a monitor, verifying step-wise completion. RoboFarseer matches or exceeds the performance of proprietary models that are orders of magnitude larger, while on open-ended planning it surpasses the best baseline by more than 40%. In real-world, long-horizon tasks, the complete system boosts overall success by roughly 60% compared with the same low-level controller without the planner. We will open-source both the dataset and the trained model upon publication.
>
---
#### [new 015] MUVLA: Learning to Explore Object Navigation via Map Understanding
- **分类: cs.RO**

- **简介: 该论文属于对象导航任务，旨在解决如何通过地图理解实现有效探索。MUVLA利用语义地图和历史信息，生成合理的动作序列，提升导航效果。**

- **链接: [http://arxiv.org/pdf/2509.25966v1](http://arxiv.org/pdf/2509.25966v1)**

> **作者:** Peilong Han; Fan Jia; Min Zhang; Yutao Qiu; Hongyao Tang; Yan Zheng; Tiancai Wang; Jianye Hao
>
> **摘要:** In this paper, we present MUVLA, a Map Understanding Vision-Language-Action model tailored for object navigation. It leverages semantic map abstractions to unify and structure historical information, encoding spatial context in a compact and consistent form. MUVLA takes the current and history observations, as well as the semantic map, as inputs and predicts the action sequence based on the description of goal object. Furthermore, it amplifies supervision through reward-guided return modeling based on dense short-horizon progress signals, enabling the model to develop a detailed understanding of action value for reward maximization. MUVLA employs a three-stage training pipeline: learning map-level spatial understanding, imitating behaviors from mixed-quality demonstrations, and reward amplification. This strategy allows MUVLA to unify diverse demonstrations into a robust spatial representation and generate more rational exploration strategies. Experiments on HM3D and Gibson benchmarks demonstrate that MUVLA achieves great generalization and learns effective exploration behaviors even from low-quality or partially successful trajectories.
>
---
#### [new 016] Autonomous Multi-Robot Infrastructure for AI-Enabled Healthcare Delivery and Diagnostics
- **分类: cs.RO; 68T40, 68T05; I.2.9; I.2.6**

- **简介: 该论文属于医疗自动化任务，旨在提升患者监护与护理效率。通过多机器人系统实现健康监测、药物配送和应急响应，解决传统医疗人力不足与响应延迟问题。**

- **链接: [http://arxiv.org/pdf/2509.26106v1](http://arxiv.org/pdf/2509.26106v1)**

> **作者:** Nakhul Kalaivanan; Senthil Arumugam Muthukumaraswamy; Girish Balasubramanian
>
> **备注:** 11 pages, 5 figures, MSc dissertation submission draft, prepared for conference/journal consideration
>
> **摘要:** This research presents a multi-robot system for inpatient care, designed using swarm intelligence principles and incorporating wearable health sensors, RF-based communication, and AI-driven decision support. Within a simulated hospital environment, the system adopts a leader-follower swarm configuration to perform patient monitoring, medicine delivery, and emergency assistance. Due to ethical constraints, live patient trials were not conducted; instead, validation was carried out through controlled self-testing with wearable sensors. The Leader Robot acquires key physiological parameters, including temperature, SpO2, heart rate, and fall detection, and coordinates other robots when required. The Assistant Robot patrols corridors for medicine delivery, while a robotic arm provides direct drug administration. The swarm-inspired leader-follower strategy enhanced communication reliability and ensured continuous monitoring, including automated email alerts to healthcare staff. The system hardware was implemented using Arduino, Raspberry Pi, NRF24L01 RF modules, and a HuskyLens AI camera. Experimental evaluation showed an overall sensor accuracy above 94%, a 92% task-level success rate, and a 96% communication reliability rate, demonstrating system robustness. Furthermore, the AI-enabled decision support was able to provide early warnings of abnormal health conditions, highlighting the potential of the system as a cost-effective solution for hospital automation and patient safety.
>
---
#### [new 017] Memory-Efficient 2D/3D Shape Assembly of Robot Swarms
- **分类: cs.RO**

- **简介: 该论文研究机器人集群形状组装任务，解决图像表示内存占用高的问题，提出一种高效的树状地图表示及分布式控制器，实现更快速、低内存的2D/3D形状构建。**

- **链接: [http://arxiv.org/pdf/2509.26518v1](http://arxiv.org/pdf/2509.26518v1)**

> **作者:** Shuoyu Yue; Pengpeng Li; Yang Xu; Kunrui Ze; Xingjian Long; Huazi Cao; Guibin Sun
>
> **摘要:** Mean-shift-based approaches have recently emerged as the most effective methods for robot swarm shape assembly tasks. These methods rely on image-based representations of target shapes to compute local density gradients and perform mean-shift exploration, which constitute their core mechanism. However, such image representations incur substantial memory overhead, which can become prohibitive for high-resolution or 3D shapes. To overcome this limitation, we propose a memory-efficient tree map representation that hierarchically encodes user-specified shapes and is applicable to both 2D and 3D scenarios. Building on this representation, we design a behavior-based distributed controller that enables assignment-free shape assembly. Comparative 2D and 3D simulations against a state-of-the-art mean-shift algorithm demonstrate one to two orders of magnitude lower memory usage and two to three times faster shape entry while maintaining comparable uniformity. Finally, we validate the framework through physical experiments with 6 to 7 UAVs, confirming its real-world practicality.
>
---
#### [new 018] ISyHand: A Dexterous Multi-finger Robot Hand with an Articulated Palm
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决低成本高灵巧机械手的设计问题。研究提出ISyHand，采用可定制部件和强化学习，实现高效灵活的抓取与操作。**

- **链接: [http://arxiv.org/pdf/2509.26236v1](http://arxiv.org/pdf/2509.26236v1)**

> **作者:** Benjamin A. Richardson; Felix Grüninger; Lukas Mack; Joerg Stueckler; Katherine J. Kuchenbecker
>
> **备注:** Accepted at IEEE Humanoids 2025
>
> **摘要:** The rapid increase in the development of humanoid robots and customized manufacturing solutions has brought dexterous manipulation to the forefront of modern robotics. Over the past decade, several expensive dexterous hands have come to market, but advances in hardware design, particularly in servo motors and 3D printing, have recently facilitated an explosion of cheaper open-source hands. Most hands are anthropomorphic to allow use of standard human tools, and attempts to increase dexterity often sacrifice anthropomorphism. We introduce the open-source ISyHand (pronounced easy-hand), a highly dexterous, low-cost, easy-to-manufacture, on-joint servo-driven robot hand. Our hand uses off-the-shelf Dynamixel motors, fasteners, and 3D-printed parts, can be assembled within four hours, and has a total material cost of about 1,300 USD. The ISyHands's unique articulated-palm design increases overall dexterity with only a modest sacrifice in anthropomorphism. To demonstrate the utility of the articulated palm, we use reinforcement learning in simulation to train the hand to perform a classical in-hand manipulation task: cube reorientation. Our novel, systematic experiments show that the simulated ISyHand outperforms the two most comparable hands in early training phases, that all three perform similarly well after policy convergence, and that the ISyHand significantly outperforms a fixed-palm version of its own design. Additionally, we deploy a policy trained on cube reorientation on the real hand, demonstrating its ability to perform real-world dexterous manipulation.
>
---
#### [new 019] LLM-MCoX: Large Language Model-based Multi-robot Coordinated Exploration and Search
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多机器人协同探索与搜索任务，解决传统方法协调不足的问题。通过引入大语言模型，实现高效环境探索和目标搜索。**

- **链接: [http://arxiv.org/pdf/2509.26324v1](http://arxiv.org/pdf/2509.26324v1)**

> **作者:** Ruiyang Wang; Haolun Tsu; David Hunt; Shaocheng Luo; Jiwoo Kim; Miroslav Pajic
>
> **摘要:** Autonomous exploration and object search in unknown indoor environments remain challenging for multi-robot systems (MRS). Traditional approaches often rely on greedy frontier assignment strategies with limited inter-robot coordination. In this work, we introduce LLM-MCoX (LLM-based Multi-robot Coordinated Exploration and Search), a novel framework that leverages Large Language Models (LLMs) for intelligent coordination of both homogeneous and heterogeneous robot teams tasked with efficient exploration and target object search. Our approach combines real-time LiDAR scan processing for frontier cluster extraction and doorway detection with multimodal LLM reasoning (e.g., GPT-4o) to generate coordinated waypoint assignments based on shared environment maps and robot states. LLM-MCoX demonstrates superior performance compared to existing methods, including greedy and Voronoi-based planners, achieving 22.7% faster exploration times and 50% improved search efficiency in large environments with 6 robots. Notably, LLM-MCoX enables natural language-based object search capabilities, allowing human operators to provide high-level semantic guidance that traditional algorithms cannot interpret.
>
---
#### [new 020] Field Calibration of Hyperspectral Cameras for Terrain Inference
- **分类: cs.RO; eess.IV**

- **简介: 该论文属于遥感或机器人视觉任务，旨在解决地形识别中因光照变化导致的误差问题。通过构建系统采集多波段图像并进行反射率校准，提升地形分析准确性。**

- **链接: [http://arxiv.org/pdf/2509.25663v1](http://arxiv.org/pdf/2509.25663v1)**

> **作者:** Nathaniel Hanson; Benjamin Pyatski; Samuel Hibbard; Gary Lvov; Oscar De La Garza; Charles DiMarzio; Kristen L. Dorsey; Taşkın Padır
>
> **备注:** Accepted to IEEE Robotics & Automation Letters
>
> **摘要:** Intra-class terrain differences such as water content directly influence a vehicle's ability to traverse terrain, yet RGB vision systems may fail to distinguish these properties. Evaluating a terrain's spectral content beyond red-green-blue wavelengths to the near infrared spectrum provides useful information for intra-class identification. However, accurate analysis of this spectral information is highly dependent on ambient illumination. We demonstrate a system architecture to collect and register multi-wavelength, hyperspectral images from a mobile robot and describe an approach to reflectance calibrate cameras under varying illumination conditions. To showcase the practical applications of our system, HYPER DRIVE, we demonstrate the ability to calculate vegetative health indices and soil moisture content from a mobile robot platform.
>
---
#### [new 021] When and How to Express Empathy in Human-Robot Interaction Scenarios
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人如何适时表达共情的问题。工作包括提出whEE框架，用于检测共情线索并生成合适回应。**

- **链接: [http://arxiv.org/pdf/2509.25200v1](http://arxiv.org/pdf/2509.25200v1)**

> **作者:** Christian Arzate Cruz; Edwin C. Montiel-Vazquez; Chikara Maeda; Randy Gomez
>
> **摘要:** Incorporating empathetic behavior into robots can improve their social effectiveness and interaction quality. In this paper, we present whEE (when and how to express empathy), a framework that enables social robots to detect when empathy is needed and generate appropriate responses. Using large language models, whEE identifies key behavioral empathy cues in human interactions. We evaluate it in human-robot interaction scenarios with our social robot, Haru. Results show that whEE effectively identifies and responds to empathy cues, providing valuable insights for designing social robots capable of adaptively modulating their empathy levels across various interaction contexts.
>
---
#### [new 022] Anomaly detection for generic failure monitoring in robotic assembly, screwing and manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作中的异常检测任务，旨在解决不同任务中故障识别问题。通过分析多模态时间序列数据，评估多种自编码器方法的泛化能力和检测效果。**

- **链接: [http://arxiv.org/pdf/2509.26308v1](http://arxiv.org/pdf/2509.26308v1)**

> **作者:** Niklas Grambow; Lisa-Marie Fenner; Felipe Kempkes; Philip Hotz; Dingyuan Wan; Jörg Krüger; Kevin Haninger
>
> **摘要:** Out-of-distribution states in robot manipulation often lead to unpredictable robot behavior or task failure, limiting success rates and increasing risk of damage. Anomaly detection (AD) can identify deviations from expected patterns in data, which can be used to trigger failsafe behaviors and recovery strategies. Prior work has applied data-driven AD to time series data in specific robotic tasks, but its transferability across control strategies and task types has not been shown. Leveraging time series data, such as force/torque signals, allows to directly capture robot-environment interactions, crucial for manipulation and online failure detection. Their broad availability, high sampling rates, and low dimensionality enable high temporal resolution and efficient processing. As robotic tasks can have widely signal characteristics and requirements, AD methods which can be applied in the same way to a wide range of tasks is needed, ideally with good data efficiency. We examine three industrial robotic tasks, each presenting several anomalies. Test scenarios in robotic cabling, screwing, and sanding are built, and multimodal time series data is gathered. Several autoencoder-based methods are compared, evaluating generalization across tasks and control methods (diffusion policy, position, and impedance control). This allows us to validate the integration of AD in complex tasks involving tighter tolerances and variation from both the robot and its environment. Additionally, we evaluate data efficiency, detection latency, and task characteristics which support robust detection. The results indicate reliable detection with AUROC exceeding 0.93 in failures in the cabling and screwing task, such as incorrect or misaligned parts and obstructed targets. In the polishing task, only severe failures were reliably detected, while more subtle failure types remained undetected.
>
---
#### [new 023] Analytic Conditions for Differentiable Collision Detection in Trajectory Optimization
- **分类: cs.RO; cs.CG**

- **简介: 该论文属于运动规划任务，解决碰撞检测与轨迹优化中的非穿透约束问题。通过引入可微的解析条件和光滑近似方法，提高优化效率。**

- **链接: [http://arxiv.org/pdf/2509.26459v1](http://arxiv.org/pdf/2509.26459v1)**

> **作者:** Akshay Jaitly; Devesh K. Jha; Kei Ota; Yuki Shirai
>
> **备注:** 8 pages, 8 figures. Accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Optimization-based methods are widely used for computing fast, diverse solutions for complex tasks such as collision-free movement or planning in the presence of contacts. However, most of these methods require enforcing non-penetration constraints between objects, resulting in a non-trivial and computationally expensive problem. This makes the use of optimization-based methods for planning and control challenging. In this paper, we present a method to efficiently enforce non-penetration of sets while performing optimization over their configuration, which is directly applicable to problems like collision-aware trajectory optimization. We introduce novel differentiable conditions with analytic expressions to achieve this. To enforce non-collision between non-smooth bodies using these conditions, we introduce a method to approximate polytopes as smooth semi-algebraic sets. We present several numerical experiments to demonstrate the performance of the proposed method and compare the performance with other baseline methods recently proposed in the literature.
>
---
#### [new 024] Terrain-Awared LiDAR-Inertial Odometry for Legged-Wheel Robots Based on Radial Basis Function Approximation
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决腿轮机器人在复杂地形中的位姿漂移问题。通过RBF近似地形并引入软约束，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2509.26222v1](http://arxiv.org/pdf/2509.26222v1)**

> **作者:** Yizhe Liu; Han Zhang
>
> **摘要:** An accurate odometry is essential for legged-wheel robots operating in unstructured terrains such as bumpy roads and staircases. Existing methods often suffer from pose drift due to their ignorance of terrain geometry. We propose a terrain-awared LiDAR-Inertial odometry (LIO) framework that approximates the terrain using Radial Basis Functions (RBF) whose centers are adaptively selected and weights are recursively updated. The resulting smooth terrain manifold enables ``soft constraints" that regularize the odometry optimization and mitigates the $z$-axis pose drift under abrupt elevation changes during robot's maneuver. To ensure the LIO's real-time performance, we further evaluate the RBF-related terms and calculate the inverse of the sparse kernel matrix with GPU parallelization. Experiments on unstructured terrains demonstrate that our method achieves higher localization accuracy than the state-of-the-art baselines, especially in the scenarios that have continuous height changes or sparse features when abrupt height changes occur.
>
---
#### [new 025] SDA-PLANNER: State-Dependency Aware Adaptive Planner for Embodied Task Planning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于具身任务规划领域，解决LLM在任务分解中的局限性。提出SDA-PLANNER，具备状态依赖感知和错误自适应能力，提升规划成功率。**

- **链接: [http://arxiv.org/pdf/2509.26375v1](http://arxiv.org/pdf/2509.26375v1)**

> **作者:** Zichao Shen; Chen Gao; Jiaqi Yuan; Tianchen Zhu; Xingcheng Fu; Qingyun Sun
>
> **摘要:** Embodied task planning requires agents to produce executable actions in a close-loop manner within the environment. With progressively improving capabilities of LLMs in task decomposition, planning, and generalization, current embodied task planning methods adopt LLM-based architecture.However, existing LLM-based planners remain limited in three aspects, i.e., fixed planning paradigms, lack of action sequence constraints, and error-agnostic. In this work, we propose SDA-PLANNER, enabling an adaptive planning paradigm, state-dependency aware and error-aware mechanisms for comprehensive embodied task planning. Specifically, SDA-PLANNER introduces a State-Dependency Graph to explicitly model action preconditions and effects, guiding the dynamic revision. To handle execution error, it employs an error-adaptive replanning strategy consisting of Error Backtrack and Diagnosis and Adaptive Action SubTree Generation, which locally reconstructs the affected portion of the plan based on the current environment state. Experiments demonstrate that SDA-PLANNER consistently outperforms baselines in success rate and goal completion, particularly under diverse error conditions.
>
---
#### [new 026] S$^3$E: Self-Supervised State Estimation for Radar-Inertial System
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决雷达点云稀疏和多路径干扰问题，通过融合雷达谱和惯性数据实现自监督定位。**

- **链接: [http://arxiv.org/pdf/2509.25984v1](http://arxiv.org/pdf/2509.25984v1)**

> **作者:** Shengpeng Wang; Yulong Xie; Qing Liao; Wei Wang
>
> **摘要:** Millimeter-wave radar for state estimation is gaining significant attention for its affordability and reliability in harsh conditions. Existing localization solutions typically rely on post-processed radar point clouds as landmark points. Nonetheless, the inherent sparsity of radar point clouds, ghost points from multi-path effects, and limited angle resolution in single-chirp radar severely degrade state estimation performance. To address these issues, we propose S$^3$E, a \textbf{S}elf-\textbf{S}upervised \textbf{S}tate \textbf{E}stimator that employs more richly informative radar signal spectra to bypass sparse points and fuses complementary inertial information to achieve accurate localization. S$^3$E fully explores the association between \textit{exteroceptive} radar and \textit{proprioceptive} inertial sensor to achieve complementary benefits. To deal with limited angle resolution, we introduce a novel cross-fusion technique that enhances spatial structure information by exploiting subtle rotational shift correlations across heterogeneous data. The experimental results demonstrate our method achieves robust and accurate performance without relying on localization ground truth supervision. To the best of our knowledge, this is the first attempt to achieve state estimation by fusing radar spectra and inertial data in a complementary self-supervised manner.
>
---
#### [new 027] Graphite: A GPU-Accelerated Mixed-Precision Graph Optimization Framework
- **分类: cs.RO**

- **简介: 该论文属于图优化任务，旨在提升GPU上的非线性优化效率。通过混合精度和内存优化技术，Graphite实现了比现有方法更快的计算速度和更低的内存占用。**

- **链接: [http://arxiv.org/pdf/2509.26581v1](http://arxiv.org/pdf/2509.26581v1)**

> **作者:** Shishir Gopinath; Karthik Dantu; Steven Y. Ko
>
> **摘要:** We present Graphite, a GPU-accelerated nonlinear graph optimization framework. It provides a CUDA C++ interface to enable the sharing of code between a realtime application, such as a SLAM system, and its optimization tasks. The framework supports techniques to reduce memory usage, including in-place optimization, support for multiple floating point types and mixed-precision modes, and dynamically computed Jacobians. We evaluate Graphite on well-known bundle adjustment problems and find that it achieves similar performance to MegBA, a solver specialized for bundle adjustment, while maintaining generality and using less memory. We also apply Graphite to global visual-inertial bundle adjustment on maps generated from stereo-inertial SLAM datasets, and observe speed ups of up to 59x compared to a CPU baseline. Our results indicate that our solver enables faster large-scale optimization on both desktop and resource-constrained devices.
>
---
#### [new 028] Evolutionary Continuous Adaptive RL-Powered Co-Design for Humanoid Chin-Up Performance
- **分类: cs.RO**

- **简介: 该论文属于机器人控制与设计任务，解决传统设计流程限制问题，通过EA-CoRL框架实现硬件与控制的协同优化，提升人形机器人引体向上性能。**

- **链接: [http://arxiv.org/pdf/2509.26082v1](http://arxiv.org/pdf/2509.26082v1)**

> **作者:** Tianyi Jin; Melya Boukheddimi; Rohit Kumar; Gabriele Fadini; Frank Kirchner
>
> **摘要:** Humanoid robots have seen significant advancements in both design and control, with a growing emphasis on integrating these aspects to enhance overall performance. Traditionally, robot design has followed a sequential process, where control algorithms are developed after the hardware is finalized. However, this can be myopic and prevent robots to fully exploit their hardware capabilities. Recent approaches advocate for co-design, optimizing both design and control in parallel to maximize robotic capabilities. This paper presents the Evolutionary Continuous Adaptive RL-based Co-Design (EA-CoRL) framework, which combines reinforcement learning (RL) with evolutionary strategies to enable continuous adaptation of the control policy to the hardware. EA-CoRL comprises two key components: Design Evolution, which explores the hardware choices using an evolutionary algorithm to identify efficient configurations, and Policy Continuous Adaptation, which fine-tunes a task-specific control policy across evolving designs to maximize performance rewards. We evaluate EA-CoRL by co-designing the actuators (gear ratios) and control policy of the RH5 humanoid for a highly dynamic chin-up task, previously unfeasible due to actuator limitations. Comparative results against state-of-the-art RL-based co-design methods show that EA-CoRL achieves higher fitness score and broader design space exploration, highlighting the critical role of continuous policy adaptation in robot co-design.
>
---
#### [new 029] MLA: A Multisensory Language-Action Model for Multimodal Understanding and Forecasting in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决多模态感知与动作生成问题。提出MLA模型，融合视觉、3D点云和触觉信息，提升物理世界建模与动作控制能力。**

- **链接: [http://arxiv.org/pdf/2509.26642v1](http://arxiv.org/pdf/2509.26642v1)**

> **作者:** Zhuoyang Liu; Jiaming Liu; Jiadong Xu; Nuowei Han; Chenyang Gu; Hao Chen; Kaichen Zhou; Renrui Zhang; Kai Chin Hsieh; Kun Wu; Zhengping Che; Jian Tang; Shanghang Zhang
>
> **摘要:** Vision-language-action models (VLAs) have shown generalization capabilities in robotic manipulation tasks by inheriting from vision-language models (VLMs) and learning action generation. Most VLA models focus on interpreting vision and language to generate actions, whereas robots must perceive and interact within the spatial-physical world. This gap highlights the need for a comprehensive understanding of robotic-specific multisensory information, which is crucial for achieving complex and contact-rich control. To this end, we introduce a multisensory language-action (MLA) model that collaboratively perceives heterogeneous sensory modalities and predicts future multisensory objectives to facilitate physical world modeling. Specifically, to enhance perceptual representations, we propose an encoder-free multimodal alignment scheme that innovatively repurposes the large language model itself as a perception module, directly interpreting multimodal cues by aligning 2D images, 3D point clouds, and tactile tokens through positional correspondence. To further enhance MLA's understanding of physical dynamics, we design a future multisensory generation post-training strategy that enables MLA to reason about semantic, geometric, and interaction information, providing more robust conditions for action generation. For evaluation, the MLA model outperforms the previous state-of-the-art 2D and 3D VLA methods by 12% and 24% in complex, contact-rich real-world tasks, respectively, while also demonstrating improved generalization to unseen configurations. Project website: https://sites.google.com/view/open-mla
>
---
#### [new 030] TacRefineNet: Tactile-Only Grasp Refinement Between Arbitrary In-Hand Object Poses
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决长周期任务中抓取姿态不准确的问题。提出TacRefineNet，利用多指触觉传感实现任意手内姿态的精调。**

- **链接: [http://arxiv.org/pdf/2509.25746v1](http://arxiv.org/pdf/2509.25746v1)**

> **作者:** Shuaijun Wang; Haoran Zhou; Diyun Xiang; Yangwei You
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Despite progress in both traditional dexterous grasping pipelines and recent Vision-Language-Action (VLA) approaches, the grasp execution stage remains prone to pose inaccuracies, especially in long-horizon tasks, which undermines overall performance. To address this "last-mile" challenge, we propose TacRefineNet, a tactile-only framework that achieves fine in-hand pose refinement of known objects in arbitrary target poses using multi-finger fingertip sensing. Our method iteratively adjusts the end-effector pose based on tactile feedback, aligning the object to the desired configuration. We design a multi-branch policy network that fuses tactile inputs from multiple fingers along with proprioception to predict precise control updates. To train this policy, we combine large-scale simulated data from a physics-based tactile model in MuJoCo with real-world data collected from a physical system. Comparative experiments show that pretraining on simulated data and fine-tuning with a small amount of real data significantly improves performance over simulation-only training. Extensive real-world experiments validate the effectiveness of the method, achieving millimeter-level grasp accuracy using only tactile input. To our knowledge, this is the first method to enable arbitrary in-hand pose refinement via multi-finger tactile sensing alone. Project website is available at https://sites.google.com/view/tacrefinenet
>
---
#### [new 031] Learning from Hallucinating Critical Points for Navigation in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态障碍物数据生成难题。提出LfH-CP框架，通过关键点 hallucination 生成多样化障碍轨迹，提升运动规划效果。**

- **链接: [http://arxiv.org/pdf/2509.26513v1](http://arxiv.org/pdf/2509.26513v1)**

> **作者:** Saad Abdul Ghani; Kameron Lee; Xuesu Xiao
>
> **摘要:** Generating large and diverse obstacle datasets to learn motion planning in environments with dynamic obstacles is challenging due to the vast space of possible obstacle trajecto- ries. Inspired by hallucination-based data synthesis approaches, we propose Learning from Hallucinating Critical Points (LfH- CP), a self-supervised framework for creating rich dynamic ob- stacle datasets based on existing optimal motion plans without requiring expensive expert demonstrations or trial-and-error exploration. LfH-CP factorizes hallucination into two stages: first identifying when and where obstacles must appear in order to result in an optimal motion plan, i.e., the critical points, and then procedurally generating diverse trajectories that pass through these points while avoiding collisions. This factorization avoids generative failures such as mode collapse and ensures coverage of diverse dynamic behaviors. We further introduce a diversity metric to quantify dataset richness and show that LfH-CP produces substantially more varied training data than existing baselines. Experiments in simulation demonstrate that planners trained on LfH-CP datasets achieves higher success rates compared to a prior hallucination method.
>
---
#### [new 032] Radio-based Multi-Robot Odometry and Relative Localization
- **分类: cs.RO**

- **简介: 该论文属于多机器人相对定位任务，解决在复杂环境中精确估计无人机与地面机器人相对位置的问题。通过融合UWB、雷达和IMU数据，提出一种鲁棒的相对定位系统。**

- **链接: [http://arxiv.org/pdf/2509.26558v1](http://arxiv.org/pdf/2509.26558v1)**

> **作者:** Andrés Martínez-Silva; David Alejo; Luis Merino; Fernando Caballero
>
> **摘要:** Radio-based methods such as Ultra-Wideband (UWB) and RAdio Detection And Ranging (radar), which have traditionally seen limited adoption in robotics, are experiencing a boost in popularity thanks to their robustness to harsh environmental conditions and cluttered environments. This work proposes a multi-robot UGV-UAV localization system that leverages the two technologies with inexpensive and readily-available sensors, such as Inertial Measurement Units (IMUs) and wheel encoders, to estimate the relative position of an aerial robot with respect to a ground robot. The first stage of the system pipeline includes a nonlinear optimization framework to trilaterate the location of the aerial platform based on UWB range data, and a radar pre-processing module with loosely coupled ego-motion estimation which has been adapted for a multi-robot scenario. Then, the pre-processed radar data as well as the relative transformation are fed to a pose-graph optimization framework with odometry and inter-robot constraints. The system, implemented for the Robotic Operating System (ROS 2) with the Ceres optimizer, has been validated in Software-in-the-Loop (SITL) simulations and in a real-world dataset. The proposed relative localization module outperforms state-of-the-art closed-form methods which are less robust to noise. Our SITL environment includes a custom Gazebo plugin for generating realistic UWB measurements modeled after real data. Conveniently, the proposed factor graph formulation makes the system readily extensible to full Simultaneous Localization And Mapping (SLAM). Finally, all the code and experimental data is publicly available to support reproducibility and to serve as a common open dataset for benchmarking.
>
---
#### [new 033] Exhaustive-Serve-Longest Control for Multi-robot Scheduling Systems
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 该论文研究多机器人调度系统中的在线任务分配问题，提出ESL策略以优化服务与切换，降低等待成本和队列长度。**

- **链接: [http://arxiv.org/pdf/2509.25556v1](http://arxiv.org/pdf/2509.25556v1)**

> **作者:** Mohammad Merati; David Castañón
>
> **摘要:** We study online task allocation for multi-robot, multi-queue systems with stochastic arrivals and switching delays. Time is slotted; each location can host at most one robot per slot; service consumes one slot; switching between locations incurs a one-slot travel delay; and arrivals are independent Bernoulli processes. We formulate a discounted-cost Markov decision process and propose Exhaustive-Serve-Longest (ESL), a simple real-time policy that serves exhaustively when the current location is nonempty and, when idle, switches to a longest unoccupied nonempty location, and we prove the optimality of this policy. As baselines, we tune a fixed-dwell cyclic policy via a discrete-time delay expression and implement a first-come-first-serve policy. Across server-to-location ratios and loads, ESL consistently yields lower discounted holding cost and smaller mean queue lengths, with action-time fractions showing more serving and restrained switching. Its simplicity and robustness make ESL a practical default for real-time multi-robot scheduling systems.
>
---
#### [new 034] State Estimation for Compliant and Morphologically Adaptive Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，解决柔性与形态自适应机器人在非刚体假设下的状态估计问题，提出基于神经网络的方法，提升其在复杂地形中的自主性能。**

- **链接: [http://arxiv.org/pdf/2509.25945v1](http://arxiv.org/pdf/2509.25945v1)**

> **作者:** Valentin Yuryev; Max Polzin; Josie Hughes
>
> **备注:** 8 pages, 10 figures, 1 table, submitted to ICRA 2026
>
> **摘要:** Locomotion robots with active or passive compliance can show robustness to uncertain scenarios, which can be promising for agricultural, research and environmental industries. However, state estimation for these robots is challenging due to the lack of rigid-body assumptions and kinematic changes from morphing. We propose a method to estimate typical rigid-body states alongside compliance-related states, such as soft robot shape in different morphologies and locomotion modes. Our neural network-based state estimator uses a history of states and a mechanism to directly influence unreliable sensors. We test our framework on the GOAT platform, a robot capable of passive compliance and active morphing for extreme outdoor terrain. The network is trained on motion capture data in a novel compliance-centric frame that accounts for morphing-related states. Our method predicts shape-related measurements within 4.2% of the robot's size, velocities within 6.3% and 2.4% of the top linear and angular speeds, respectively, and orientation within 1.5 degrees. We also demonstrate a 300% increase in travel range during a motor malfunction when using our estimator for closed-loop autonomous outdoor operation.
>
---
#### [new 035] Real-time Velocity Profile Optimization for Time-Optimal Maneuvering with Generic Acceleration Constraints
- **分类: cs.RO; cs.SY; eess.SY; math.OC**

- **简介: 该论文属于机器人轨迹规划任务，解决在通用加速度约束下实时计算最优速度曲线的问题。提出FBGA算法，兼顾精度与效率，适用于自主赛车等场景。**

- **链接: [http://arxiv.org/pdf/2509.26428v1](http://arxiv.org/pdf/2509.26428v1)**

> **作者:** Mattia Piazza; Mattia Piccinini; Sebastiano Taddei; Francesco Biral; Enrico Bertolazzi
>
> **摘要:** The computation of time-optimal velocity profiles along prescribed paths, subject to generic acceleration constraints, is a crucial problem in robot trajectory planning, with particular relevance to autonomous racing. However, the existing methods either support arbitrary acceleration constraints at high computational cost or use conservative box constraints for computational efficiency. We propose FBGA, a new \underline{F}orward-\underline{B}ackward algorithm with \underline{G}eneric \underline{A}cceleration constraints, which achieves both high accuracy and low computation time. FBGA operates forward and backward passes to maximize the velocity profile in short, discretized path segments, while satisfying user-defined performance limits. Tested on five racetracks and two vehicle classes, FBGA handles complex, non-convex acceleration constraints with custom formulations. Its maneuvers and lap times closely match optimal control baselines (within $0.11\%$-$0.36\%$), while being up to three orders of magnitude faster. FBGA maintains high accuracy even with coarse discretization, making it well-suited for online multi-query trajectory planning. Our open-source \texttt{C++} implementation is available at: https://anonymous.4open.science/r/FB_public_RAL.
>
---
#### [new 036] Online Mapping for Autonomous Driving: Addressing Sensor Generalization and Dynamic Map Updates in Campus Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主驾驶中的在线地图构建任务，解决静态地图维护难和动态环境更新问题，通过传感器数据实现地图的持续更新与泛化。**

- **链接: [http://arxiv.org/pdf/2509.25542v1](http://arxiv.org/pdf/2509.25542v1)**

> **作者:** Zihan Zhang; Abhijit Ravichandran; Pragnya Korti; Luobin Wang; Henrik I. Christensen
>
> **备注:** 19th International Symposium on Experimental Robotics
>
> **摘要:** High-definition (HD) maps are essential for autonomous driving, providing precise information such as road boundaries, lane dividers, and crosswalks to enable safe and accurate navigation. However, traditional HD map generation is labor-intensive, expensive, and difficult to maintain in dynamic environments. To overcome these challenges, we present a real-world deployment of an online mapping system on a campus golf cart platform equipped with dual front cameras and a LiDAR sensor. Our work tackles three core challenges: (1) labeling a 3D HD map for campus environment; (2) integrating and generalizing the SemVecMap model onboard; and (3) incrementally generating and updating the predicted HD map to capture environmental changes. By fine-tuning with campus-specific data, our pipeline produces accurate map predictions and supports continual updates, demonstrating its practical value in real-world autonomous driving scenarios.
>
---
#### [new 037] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决感知与动作解耦导致的适应性不足问题。提出DP-AG模型，通过扩散驱动的感知-动作交互提升策略适应性。**

- **链接: [http://arxiv.org/pdf/2509.25822v1](http://arxiv.org/pdf/2509.25822v1)**

> **作者:** Jing Wang; Weiting Peng; Jing Tang; Zeyu Gong; Xihua Wang; Bo Tao; Li Cheng
>
> **备注:** 42 pages, 17 figures, 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Existing imitation learning methods decouple perception and action, which overlooks the causal reciprocity between sensory representations and action execution that humans naturally leverage for adaptive behaviors. To bridge this gap, we introduce Action--Guided Diffusion Policy (DP--AG), a unified representation learning that explicitly models a dynamic interplay between perception and action through probabilistic latent dynamics. DP--AG encodes latent observations into a Gaussian posterior via variational inference and evolves them using an action-guided SDE, where the Vector-Jacobian Product (VJP) of the diffusion policy's noise predictions serves as a structured stochastic force driving latent updates. To promote bidirectional learning between perception and action, we introduce a cycle--consistent contrastive loss that organizes the gradient flow of the noise predictor into a coherent perception--action loop, enforcing mutually consistent transitions in both latent updates and action refinements. Theoretically, we derive a variational lower bound for the action-guided SDE, and prove that the contrastive objective enhances continuity in both latent and action trajectories. Empirically, DP--AG significantly outperforms state--of--the--art methods across simulation benchmarks and real-world UR5 manipulation tasks. As a result, our DP--AG offers a promising step toward bridging biological adaptability and artificial policy learning.
>
---
#### [new 038] Emotionally Expressive Robots: Implications for Children's Behavior toward Robot
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，探讨情感表达机器人对儿童行为的影响。研究通过实验分析不同情感表达方式如何影响儿童与机器人的互动行为。**

- **链接: [http://arxiv.org/pdf/2509.25986v1](http://arxiv.org/pdf/2509.25986v1)**

> **作者:** Elisabetta Zibetti; Sureya Waheed Palmer; Rebecca Stower; Salvatore M Anzalone
>
> **摘要:** The growing development of robots with artificial emotional expressiveness raises important questions about their persuasive potential in children's behavior. While research highlights the pragmatic value of emotional expressiveness in human social communication, the extent to which robotic expressiveness can or should influence empathic responses in children is grounds for debate. In a pilot study with 22 children (aged 7-11) we begin to explore the ways in which different levels of embodied expressiveness (body only, face only, body and face) of two basic emotions (happiness and sadness) displayed by an anthropomorphic robot (QTRobot) might modify children's behavior in a child-robot cooperative turn-taking game. We observed that children aligned their behavior to the robot's inferred emotional state. However, higher levels of expressiveness did not result in increased alignment. The preliminary results reported here provide a starting point for reflecting on robotic expressiveness and its role in shaping children's social-emotional behavior toward robots as social peers in the near future.
>
---
#### [new 039] Best of Sim and Real: Decoupled Visuomotor Manipulation via Learning Control in Simulation and Perception in Real
- **分类: cs.RO; I.2.9**

- **简介: 该论文属于机器人操控任务，解决sim-to-real转移问题。通过分离控制与感知，控制在仿真中学习，感知在真实环境中适配，提升数据效率和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.25747v1](http://arxiv.org/pdf/2509.25747v1)**

> **作者:** Jialei Huang; Zhaoheng Yin; Yingdong Hu; Shuo Wang; Xingyu Lin; Yang Gao
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Sim-to-real transfer remains a fundamental challenge in robot manipulation due to the entanglement of perception and control in end-to-end learning. We present a decoupled framework that learns each component where it is most reliable: control policies are trained in simulation with privileged state to master spatial layouts and manipulation dynamics, while perception is adapted only at deployment to bridge real observations to the frozen control policy. Our key insight is that control strategies and action patterns are universal across environments and can be learned in simulation through systematic randomization, while perception is inherently domain-specific and must be learned where visual observations are authentic. Unlike existing end-to-end approaches that require extensive real-world data, our method achieves strong performance with only 10-20 real demonstrations by reducing the complex sim-to-real problem to a structured perception alignment task. We validate our approach on tabletop manipulation tasks, demonstrating superior data efficiency and out-of-distribution generalization compared to end-to-end baselines. The learned policies successfully handle object positions and scales beyond the training distribution, confirming that decoupling perception from control fundamentally improves sim-to-real transfer.
>
---
#### [new 040] Towards Intuitive Human-Robot Interaction through Embodied Gesture-Driven Control with Woven Tactile Skins
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决传统控制方式不够直观的问题。通过编织电容触觉皮肤实现手势驱动控制，提升交互自然性与效率。**

- **链接: [http://arxiv.org/pdf/2509.25951v1](http://arxiv.org/pdf/2509.25951v1)**

> **作者:** ChunPing Lam; Xiangjia Chen; Chenming Wu; Hao Chen; Binzhi Sun; Guoxin Fang; Charlie C. L. Wang; Chengkai Dai; Yeung Yam
>
> **摘要:** This paper presents a novel human-robot interaction (HRI) framework that enables intuitive gesture-driven control through a capacitance-based woven tactile skin. Unlike conventional interfaces that rely on panels or handheld devices, the woven tactile skin integrates seamlessly with curved robot surfaces, enabling embodied interaction and narrowing the gap between human intent and robot response. Its woven design combines fabric-like flexibility with structural stability and dense multi-channel sensing through the interlaced conductive threads. Building on this capability, we define a gesture-action mapping of 14 single- and multi-touch gestures that cover representative robot commands, including task-space motion and auxiliary functions. A lightweight convolution-transformer model designed for gesture recognition in real time achieves an accuracy of near-100%, outperforming prior baseline approaches. Experiments on robot arm tasks, including pick-and-place and pouring, demonstrate that our system reduces task completion time by up to 57% compared with keyboard panels and teach pendants. Overall, our proposed framework demonstrates a practical pathway toward more natural and efficient embodied HRI.
>
---
#### [new 041] CoTaP: Compliant Task Pipeline and Reinforcement Learning of Its Controller with Compliance Modulation
- **分类: cs.RO**

- **简介: 该论文属于人形机器人控制任务，解决环境交互中合规性不足的问题。提出CoTaP框架，结合强化学习与模型基合规控制，提升机器人在真实环境中的适应能力。**

- **链接: [http://arxiv.org/pdf/2509.25443v1](http://arxiv.org/pdf/2509.25443v1)**

> **作者:** Zewen He; Chenyuan Chen; Dilshod Azizov; Yoshihiko Nakamura
>
> **备注:** Submitted to IEEE for possible publication, under review
>
> **摘要:** Humanoid whole-body locomotion control is a critical approach for humanoid robots to leverage their inherent advantages. Learning-based control methods derived from retargeted human motion data provide an effective means of addressing this issue. However, because most current human datasets lack measured force data, and learning-based robot control is largely position-based, achieving appropriate compliance during interaction with real environments remains challenging. This paper presents Compliant Task Pipeline (CoTaP): a pipeline that leverages compliance information in the learning-based structure of humanoid robots. A two-stage dual-agent reinforcement learning framework combined with model-based compliance control for humanoid robots is proposed. In the training process, first a base policy with a position-based controller is trained; then in the distillation, the upper-body policy is combined with model-based compliance control, and the lower-body agent is guided by the base policy. In the upper-body control, adjustable task-space compliance can be specified and integrated with other controllers through compliance modulation on the symmetric positive definite (SPD) manifold, ensuring system stability. We validated the feasibility of the proposed strategy in simulation, primarily comparing the responses to external disturbances under different compliance settings.
>
---
#### [new 042] Kinodynamic Motion Planning for Mobile Robot Navigation across Inconsistent World Models
- **分类: cs.RO**

- **简介: 该论文属于移动机器人路径规划任务，解决因环境模型不一致导致的运动规划安全问题。通过改进搜索策略，提升规划效率与安全性。**

- **链接: [http://arxiv.org/pdf/2509.26339v1](http://arxiv.org/pdf/2509.26339v1)**

> **作者:** Eric R. Damm; Thomas M. Howard
>
> **备注:** Presented at the Robotics: Science and Systems (RSS) 2025 Workshop on Resilient Off-road Autonomous Robotics (ROAR)
>
> **摘要:** Mobile ground robots lacking prior knowledge of an environment must rely on sensor data to develop a model of their surroundings. In these scenarios, consistent identification of obstacles and terrain features can be difficult due to noise and algorithmic shortcomings, which can make it difficult for motion planning systems to generate safe motions. One particular difficulty to overcome is when regions of the cost map switch between being marked as obstacles and free space through successive planning cycles. One potential solution to this, which we refer to as Valid in Every Hypothesis (VEH), is for the planning system to plan motions that are guaranteed to be safe through a history of world models. Another approach is to track a history of world models, and adjust node costs according to the potential penalty of needing to reroute around previously hazardous areas. This work discusses three major iterations on this idea. The first iteration, called PEH, invokes a sub-search for every node expansion that crosses through a divergence point in the world models. The second and third iterations, called GEH and GEGRH respectively, defer the sub-search until after an edge expands into the goal region. GEGRH uses an additional step to revise the graph based on divergent nodes in each world. Initial results showed that, although PEH and GEH find more optimistic solutions than VEH, they are unable to generate solutions in less than one-second, which exceeds our requirements for field deployment. Analysis of results from a field experiment in an unstructured, off-road environment on a Clearpath Robotics Warthog UGV indicate that GEGRH finds lower cost trajectories and has faster average planning times than VEH. Compared to single-hypothesis (SH) search, where only the latest world model is considered, GEGRH generates more conservative plans with a small increase in average planning time.
>
---
#### [new 043] SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习任务，解决流式策略训练不稳定问题，通过重新参数化速度网络提出两种稳定架构，并实现高效端到端训练。**

- **链接: [http://arxiv.org/pdf/2509.25756v1](http://arxiv.org/pdf/2509.25756v1)**

> **作者:** Yixian Zhang; Shu'ang Yu; Tonghe Zhang; Mo Guang; Haojia Hui; Kaiwen Long; Yu Wang; Chao Yu; Wenbo Ding
>
> **摘要:** Training expressive flow-based policies with off-policy reinforcement learning is notoriously unstable due to gradient pathologies in the multi-step action sampling process. We trace this instability to a fundamental connection: the flow rollout is algebraically equivalent to a residual recurrent computation, making it susceptible to the same vanishing and exploding gradients as RNNs. To address this, we reparameterize the velocity network using principles from modern sequential models, introducing two stable architectures: Flow-G, which incorporates a gated velocity, and Flow-T, which utilizes a decoded velocity. We then develop a practical SAC-based algorithm, enabled by a noise-augmented rollout, that facilitates direct end-to-end training of these policies. Our approach supports both from-scratch and offline-to-online learning and achieves state-of-the-art performance on continuous control and robotic manipulation benchmarks, eliminating the need for common workarounds like policy distillation or surrogate objectives.
>
---
#### [new 044] MoReFlow: Motion Retargeting Learning through Unsupervised Flow Matching
- **分类: cs.GR; cs.RO**

- **简介: 该论文属于动作迁移任务，解决不同形态角色间动作迁移问题。提出MoReFlow框架，通过无监督流匹配学习动作嵌入空间对应关系，实现灵活、可逆的迁移。**

- **链接: [http://arxiv.org/pdf/2509.25600v1](http://arxiv.org/pdf/2509.25600v1)**

> **作者:** Wontaek Kim; Tianyu Li; Sehoon Ha
>
> **摘要:** Motion retargeting holds a premise of offering a larger set of motion data for characters and robots with different morphologies. Many prior works have approached this problem via either handcrafted constraints or paired motion datasets, limiting their applicability to humanoid characters or narrow behaviors such as locomotion. Moreover, they often assume a fixed notion of retargeting, overlooking domain-specific objectives like style preservation in animation or task-space alignment in robotics. In this work, we propose MoReFlow, Motion Retargeting via Flow Matching, an unsupervised framework that learns correspondences between characters' motion embedding spaces. Our method consists of two stages. First, we train tokenized motion embeddings for each character using a VQ-VAE, yielding compact latent representations. Then, we employ flow matching with conditional coupling to align the latent spaces across characters, which simultaneously learns conditioned and unconditioned matching to achieve robust but flexible retargeting. Once trained, MoReFlow enables flexible and reversible retargeting without requiring paired data. Experiments demonstrate that MoReFlow produces high-quality motions across diverse characters and tasks, offering improved controllability, generalization, and motion realism compared to the baselines.
>
---
#### [new 045] Boundary-to-Region Supervision for Offline Safe Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于安全强化学习任务，解决静态数据中策略的安全约束问题。通过不对称条件机制B2R，提升安全性和奖励表现。**

- **链接: [http://arxiv.org/pdf/2509.25727v1](http://arxiv.org/pdf/2509.25727v1)**

> **作者:** Huikang Su; Dengyun Peng; Zifeng Zhuang; YuHan Liu; Qiguang Chen; Donglin Wang; Qinghe Liu
>
> **备注:** NeurIPS 2025
>
> **摘要:** Offline safe reinforcement learning aims to learn policies that satisfy predefined safety constraints from static datasets. Existing sequence-model-based methods condition action generation on symmetric input tokens for return-to-go and cost-to-go, neglecting their intrinsic asymmetry: return-to-go (RTG) serves as a flexible performance target, while cost-to-go (CTG) should represent a rigid safety boundary. This symmetric conditioning leads to unreliable constraint satisfaction, especially when encountering out-of-distribution cost trajectories. To address this, we propose Boundary-to-Region (B2R), a framework that enables asymmetric conditioning through cost signal realignment . B2R redefines CTG as a boundary constraint under a fixed safety budget, unifying the cost distribution of all feasible trajectories while preserving reward structures. Combined with rotary positional embeddings , it enhances exploration within the safe region. Experimental results show that B2R satisfies safety constraints in 35 out of 38 safety-critical tasks while achieving superior reward performance over baseline methods. This work highlights the limitations of symmetric token conditioning and establishes a new theoretical and practical approach for applying sequence models to safe RL. Our code is available at https://github.com/HuikangSu/B2R.
>
---
#### [new 046] Robust Visual Localization in Compute-Constrained Environments by Salient Edge Rendering and Weighted Hamming Similarity
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决在计算受限环境下物体位姿估计问题。通过边缘渲染和加权汉明相似性实现鲁棒定位。**

- **链接: [http://arxiv.org/pdf/2509.25520v1](http://arxiv.org/pdf/2509.25520v1)**

> **作者:** Tu-Hoa Pham; Philip Bailey; Daniel Posada; Georgios Georgakis; Jorge Enriquez; Surya Suresh; Marco Dolci; Philip Twu
>
> **备注:** To appear in IEEE Robotics and Automation Letters
>
> **摘要:** We consider the problem of vision-based 6-DoF object pose estimation in the context of the notional Mars Sample Return campaign, in which a robotic arm would need to localize multiple objects of interest for low-clearance pickup and insertion, under severely constrained hardware. We propose a novel localization algorithm leveraging a custom renderer together with a new template matching metric tailored to the edge domain to achieve robust pose estimation using only low-fidelity, textureless 3D models as inputs. Extensive evaluations on synthetic datasets as well as from physical testbeds on Earth and in situ Mars imagery shows that our method consistently beats the state of the art in compute and memory-constrained localization, both in terms of robustness and accuracy, in turn enabling new possibilities for cheap and reliable localization on general-purpose hardware.
>
---
#### [new 047] Sensor optimization for urban wind estimation with cluster-based probabilistic framework
- **分类: cs.LG; cs.RO; physics.flu-dyn**

- **简介: 该论文属于城市风场估计任务，解决复杂地形下传感器优化问题。通过物理信息机器学习框架，实现更准确的风速预测与传感器布局优化。**

- **链接: [http://arxiv.org/pdf/2509.25222v1](http://arxiv.org/pdf/2509.25222v1)**

> **作者:** Yutong Liang; Chang Hou; Guy Y. Cornejo Maceda; Andrea Ianiro; Stefano Discetti; Andrea Meilán-Vila; Didier Sornette; Sandro Claudio Lera; Jialong Chen; Xiaozhou He; Bernd R. Noack
>
> **摘要:** We propose a physics-informed machine-learned framework for sensor-based flow estimation for drone trajectories in complex urban terrain. The input is a rich set of flow simulations at many wind conditions. The outputs are velocity and uncertainty estimates for a target domain and subsequent sensor optimization for minimal uncertainty. The framework has three innovations compared to traditional flow estimators. First, the algorithm scales proportionally to the domain complexity, making it suitable for flows that are too complex for any monolithic reduced-order representation. Second, the framework extrapolates beyond the training data, e.g., smaller and larger wind velocities. Last, and perhaps most importantly, the sensor location is a free input, significantly extending the vast majority of the literature. The key enablers are (1) a Reynolds number-based scaling of the flow variables, (2) a physics-based domain decomposition, (3) a cluster-based flow representation for each subdomain, (4) an information entropy correlating the subdomains, and (5) a multi-variate probability function relating sensor input and targeted velocity estimates. This framework is demonstrated using drone flight paths through a three-building cluster as a simple example. We anticipate adaptations and applications for estimating complete cities and incorporating weather input.
>
---
#### [new 048] Message passing-based inference in an autoregressive active inference agent
- **分类: cs.AI; cs.LG; cs.RO; cs.SY; eess.SY; stat.ML**

- **简介: 该论文属于机器人导航任务，解决如何在不确定环境中优化决策问题。通过消息传递实现自回归主动推理代理，利用预测不确定性调节行动。**

- **链接: [http://arxiv.org/pdf/2509.25482v1](http://arxiv.org/pdf/2509.25482v1)**

> **作者:** Wouter M. Kouw; Tim N. Nisslbeck; Wouter L. N. Nuijten
>
> **备注:** 14 pages, 4 figures, to be published in the proceedings of the International Workshop on Active Inference 2025
>
> **摘要:** We present the design of an autoregressive active inference agent in the form of message passing on a factor graph. Expected free energy is derived and distributed across a planning graph. The proposed agent is validated on a robot navigation task, demonstrating exploration and exploitation in a continuous-valued observation space with bounded continuous-valued actions. Compared to a classical optimal controller, the agent modulates action based on predictive uncertainty, arriving later but with a better model of the robot's dynamics.
>
---
#### [new 049] TimeRewarder: Learning Dense Reward from Passive Videos via Frame-wise Temporal Distance
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，解决稀疏奖励任务中奖励设计困难的问题。通过分析视频中的时间距离，生成密集奖励信号以提升学习效率。**

- **链接: [http://arxiv.org/pdf/2509.26627v1](http://arxiv.org/pdf/2509.26627v1)**

> **作者:** Yuyang Liu; Chuan Wen; Yihang Hu; Dinesh Jayaraman; Yang Gao
>
> **摘要:** Designing dense rewards is crucial for reinforcement learning (RL), yet in robotics it often demands extensive manual effort and lacks scalability. One promising solution is to view task progress as a dense reward signal, as it quantifies the degree to which actions advance the system toward task completion over time. We present TimeRewarder, a simple yet effective reward learning method that derives progress estimation signals from passive videos, including robot demonstrations and human videos, by modeling temporal distances between frame pairs. We then demonstrate how TimeRewarder can supply step-wise proxy rewards to guide reinforcement learning. In our comprehensive experiments on ten challenging Meta-World tasks, we show that TimeRewarder dramatically improves RL for sparse-reward tasks, achieving nearly perfect success in 9/10 tasks with only 200,000 interactions per task with the environment. This approach outperformed previous methods and even the manually designed environment dense reward on both the final success rate and sample efficiency. Moreover, we show that TimeRewarder pretraining can exploit real-world human videos, highlighting its potential as a scalable approach path to rich reward signals from diverse video sources.
>
---
#### [new 050] Infrastructure Sensor-enabled Vehicle Data Generation using Multi-Sensor Fusion for Proactive Safety Applications at Work Zone
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于车辆检测与定位任务，解决工作区安全问题。通过融合相机与LiDAR数据，使用卡尔曼滤波提升轨迹精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.25452v1](http://arxiv.org/pdf/2509.25452v1)**

> **作者:** Suhala Rabab Saba; Sakib Khan; Minhaj Uddin Ahmad; Jiahe Cao; Mizanur Rahman; Li Zhao; Nathan Huynh; Eren Erman Ozguven
>
> **摘要:** Infrastructure-based sensing and real-time trajectory generation show promise for improving safety in high-risk roadway segments such as work zones, yet practical deployments are hindered by perspective distortion, complex geometry, occlusions, and costs. This study tackles these barriers by integrating roadside camera and LiDAR sensors into a cosimulation environment to develop a scalable, cost-effective vehicle detection and localization framework, and employing a Kalman Filter-based late fusion strategy to enhance trajectory consistency and accuracy. In simulation, the fusion algorithm reduced longitudinal error by up to 70 percent compared to individual sensors while preserving lateral accuracy within 1 to 3 meters. Field validation in an active work zone, using LiDAR, a radar-camera rig, and RTK-GPS as ground truth, demonstrated that the fused trajectories closely match real vehicle paths, even when single-sensor data are intermittent or degraded. These results confirm that KF based sensor fusion can reliably compensate for individual sensor limitations, providing precise and robust vehicle tracking capabilities. Our approach thus offers a practical pathway to deploy infrastructure-enabled multi-sensor systems for proactive safety measures in complex traffic environments.
>
---
#### [new 051] Integrator Forwading Design for Unicycles with Constant and Actuated Velocity in Polar Coordinates
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文研究无人车路径规划任务，解决基于极坐标下的自平衡控制问题。通过整合前向设计方法，构建新的平滑转向策略与控制李雅普诺夫函数，扩展了逆最优控制的设计方法。**

- **链接: [http://arxiv.org/pdf/2509.25579v1](http://arxiv.org/pdf/2509.25579v1)**

> **作者:** Miroslav Krstic; Velimir Todorovski; Kwang Hak Kim; Alessandro Astolfi
>
> **摘要:** In a companion paper, we present a modular framework for unicycle stabilization in polar coordinates that provides smooth steering laws through backstepping. Surprisingly, the same problem also allows the application of integrator forwarding. In this work, we leverage this feature and construct new smooth steering laws together with control Lyapunov functions (CLFs), expanding the set of CLFs available for inverse optimal control design. In the case of constant forward velocity (Dubins car), backstepping produces finite-time (deadbeat) parking, and we show that integrator forwarding yields the very same class of solutions. This reveals a fundamental connection between backstepping and forwarding in addressing both the unicycle and, the Dubins car parking problems.
>
---
#### [new 052] OceanGym: A Benchmark Environment for Underwater Embodied Agents
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出OceanGym，一个用于水下具身智能体的基准环境，旨在解决水下感知、决策与适应难题，通过多模态大语言模型提升智能体在复杂海洋环境中的表现。**

- **链接: [http://arxiv.org/pdf/2509.26536v1](http://arxiv.org/pdf/2509.26536v1)**

> **作者:** Yida Xue; Mingjun Mao; Xiangyuan Ru; Yuqi Zhu; Baochang Ren; Shuofei Qiao; Mengru Wang; Shumin Deng; Xinyu An; Ningyu Zhang; Ying Chen; Huajun Chen
>
> **备注:** Work in progress
>
> **摘要:** We introduce OceanGym, the first comprehensive benchmark for ocean underwater embodied agents, designed to advance AI in one of the most demanding real-world environments. Unlike terrestrial or aerial domains, underwater settings present extreme perceptual and decision-making challenges, including low visibility, dynamic ocean currents, making effective agent deployment exceptionally difficult. OceanGym encompasses eight realistic task domains and a unified agent framework driven by Multi-modal Large Language Models (MLLMs), which integrates perception, memory, and sequential decision-making. Agents are required to comprehend optical and sonar data, autonomously explore complex environments, and accomplish long-horizon objectives under these harsh conditions. Extensive experiments reveal substantial gaps between state-of-the-art MLLM-driven agents and human experts, highlighting the persistent difficulty of perception, planning, and adaptability in ocean underwater environments. By providing a high-fidelity, rigorously designed platform, OceanGym establishes a testbed for developing robust embodied AI and transferring these capabilities to real-world autonomous ocean underwater vehicles, marking a decisive step toward intelligent agents capable of operating in one of Earth's last unexplored frontiers. The code and data are available at https://github.com/OceanGPT/OceanGym.
>
---
#### [new 053] Towards Human Engagement with Realistic AI Combat Pilots
- **分类: cs.AI; cs.HC; cs.LG; cs.MA; cs.RO**

- **简介: 该论文属于人机协作任务，旨在实现人类与真实AI战斗机飞行员的实时互动。通过强化学习训练智能体，并集成到仿真系统中，提升训练效果与战术探索。**

- **链接: [http://arxiv.org/pdf/2509.26002v1](http://arxiv.org/pdf/2509.26002v1)**

> **作者:** Ardian Selmonaj; Giacomo Del Rio; Adrian Schneider; Alessandro Antonucci
>
> **备注:** 13th International Conference on Human-Agent Interaction (HAI) 2025
>
> **摘要:** We present a system that enables real-time interaction between human users and agents trained to control fighter jets in simulated 3D air combat scenarios. The agents are trained in a dedicated environment using Multi-Agent Reinforcement Learning. A communication link is developed to allow seamless deployment of trained agents into VR-Forces, a widely used defense simulation tool for realistic tactical scenarios. This integration allows mixed simulations where human-controlled entities engage with intelligent agents exhibiting distinct combat behaviors. Our interaction model creates new opportunities for human-agent teaming, immersive training, and the exploration of innovative tactics in defense contexts.
>
---
#### [new 054] LLM-RG: Referential Grounding in Outdoor Scenarios using Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于户外场景指代消解任务，解决自然语言引用在复杂环境中的定位问题。通过结合视觉语言模型和大语言模型，提升指代识别的准确性。**

- **链接: [http://arxiv.org/pdf/2509.25528v1](http://arxiv.org/pdf/2509.25528v1)**

> **作者:** Pranav Saxena; Avigyan Bhattacharya; Ji Zhang; Wenshan Wang
>
> **摘要:** Referential grounding in outdoor driving scenes is challenging due to large scene variability, many visually similar objects, and dynamic elements that complicate resolving natural-language references (e.g., "the black car on the right"). We propose LLM-RG, a hybrid pipeline that combines off-the-shelf vision-language models for fine-grained attribute extraction with large language models for symbolic reasoning. LLM-RG processes an image and a free-form referring expression by using an LLM to extract relevant object types and attributes, detecting candidate regions, generating rich visual descriptors with a VLM, and then combining these descriptors with spatial metadata into natural-language prompts that are input to an LLM for chain-of-thought reasoning to identify the referent's bounding box. Evaluated on the Talk2Car benchmark, LLM-RG yields substantial gains over both LLM and VLM-based baselines. Additionally, our ablations show that adding 3D spatial cues further improves grounding. Our results demonstrate the complementary strengths of VLMs and LLMs, applied in a zero-shot manner, for robust outdoor referential grounding.
>
---
#### [new 055] The Trajectory Bundle Method: Unifying Sequential-Convex Programming and Sampling-Based Trajectory Optimization
- **分类: math.OC; cs.RO**

- **简介: 该论文属于运动规划与控制任务，解决非凸轨迹优化问题。通过无导数的序列凸规划方法，结合采样技术，实现灵活高效的轨迹优化。**

- **链接: [http://arxiv.org/pdf/2509.26575v1](http://arxiv.org/pdf/2509.26575v1)**

> **作者:** Kevin Tracy; John Z. Zhang; Jon Arrizabalaga; Stefan Schaal; Yuval Tassa; Tom Erez; Zachary Manchester
>
> **摘要:** We present a unified framework for solving trajectory optimization problems in a derivative-free manner through the use of sequential convex programming. Traditionally, nonconvex optimization problems are solved by forming and solving a sequence of convex optimization problems, where the cost and constraint functions are approximated locally through Taylor series expansions. This presents a challenge for functions where differentiation is expensive or unavailable. In this work, we present a derivative-free approach to form these convex approximations by computing samples of the dynamics, cost, and constraint functions and letting the solver interpolate between them. Our framework includes sample-based trajectory optimization techniques like model-predictive path integral (MPPI) control as a special case and generalizes them to enable features like multiple shooting and general equality and inequality constraints that are traditionally associated with derivative-based sequential convex programming methods. The resulting framework is simple, flexible, and capable of solving a wide variety of practical motion planning and control problems.
>
---
#### [new 056] Modular Design of Strict Control Lyapunov Functions for Global Stabilization of the Unicycle in Polar Coordinates
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文属于控制理论任务，解决无人车全局稳定化问题。通过模块化设计严格控制李雅普诺夫函数，实现极坐标下无人车的全局渐近稳定。**

- **链接: [http://arxiv.org/pdf/2509.25575v1](http://arxiv.org/pdf/2509.25575v1)**

> **作者:** Velimir Todorovski; Kwang Hak Kim; Miroslav Krstic
>
> **摘要:** Since the mid-1990s, it has been known that, unlike in Cartesian form where Brockett's condition rules out static feedback stabilization, the unicycle is globally asymptotically stabilizable by smooth feedback in polar coordinates. In this note, we introduce a modular framework for designing smooth feedback laws that achieve global asymptotic stabilization in polar coordinates. These laws are bidirectional, enabling efficient parking maneuvers, and are paired with families of strict control Lyapunov functions (CLFs) constructed in a modular fashion. The resulting CLFs guarantee global asymptotic stability with explicit convergence rates and include barrier variants that yield "almost global" stabilization, excluding only zero-measure subsets of the rotation manifolds. The strictness of the CLFs is further leveraged in our companion paper, where we develop inverse-optimal redesigns with meaningful cost functions and infinite gain margins.
>
---
#### [new 057] Benchmarking Egocentric Visual-Inertial SLAM at City Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉惯性SLAM任务，旨在解决城市尺度下佩戴式设备的精准定位与建图问题。通过构建新数据集和基准，评估现有系统并分析其不足。**

- **链接: [http://arxiv.org/pdf/2509.26639v1](http://arxiv.org/pdf/2509.26639v1)**

> **作者:** Anusha Krishnan; Shaohui Liu; Paul-Edouard Sarlin; Oscar Gentilhomme; David Caruso; Maurizio Monge; Richard Newcombe; Jakob Engel; Marc Pollefeys
>
> **备注:** ICCV 2025
>
> **摘要:** Precise 6-DoF simultaneous localization and mapping (SLAM) from onboard sensors is critical for wearable devices capturing egocentric data, which exhibits specific challenges, such as a wider diversity of motions and viewpoints, prevalent dynamic visual content, or long sessions affected by time-varying sensor calibration. While recent progress on SLAM has been swift, academic research is still driven by benchmarks that do not reflect these challenges or do not offer sufficiently accurate ground truth poses. In this paper, we introduce a new dataset and benchmark for visual-inertial SLAM with egocentric, multi-modal data. We record hours and kilometers of trajectories through a city center with glasses-like devices equipped with various sensors. We leverage surveying tools to obtain control points as indirect pose annotations that are metric, centimeter-accurate, and available at city scale. This makes it possible to evaluate extreme trajectories that involve walking at night or traveling in a vehicle. We show that state-of-the-art systems developed by academia are not robust to these challenges and we identify components that are responsible for this. In addition, we design tracks with different levels of difficulty to ease in-depth analysis and evaluation of less mature approaches. The dataset and benchmark are available at https://www.lamaria.ethz.ch.
>
---
#### [new 058] World Model for AI Autonomous Navigation in Mechanical Thrombectomy
- **分类: cs.LG; cs.RO; eess.IV**

- **简介: 该论文属于机械血栓切除术中的自主导航任务，旨在解决血管结构复杂与实时决策难题。通过TD-MPC2算法提升多任务学习性能。**

- **链接: [http://arxiv.org/pdf/2509.25518v1](http://arxiv.org/pdf/2509.25518v1)**

> **作者:** Harry Robertshaw; Han-Ru Wu; Alejandro Granados; Thomas C Booth
>
> **备注:** Published in Medical Image Computing and Computer Assisted Intervention - MICCAI 2025, Lecture Notes in Computer Science, vol 15968
>
> **摘要:** Autonomous navigation for mechanical thrombectomy (MT) remains a critical challenge due to the complexity of vascular anatomy and the need for precise, real-time decision-making. Reinforcement learning (RL)-based approaches have demonstrated potential in automating endovascular navigation, but current methods often struggle with generalization across multiple patient vasculatures and long-horizon tasks. We propose a world model for autonomous endovascular navigation using TD-MPC2, a model-based RL algorithm. We trained a single RL agent across multiple endovascular navigation tasks in ten real patient vasculatures, comparing performance against the state-of-the-art Soft Actor-Critic (SAC) method. Results indicate that TD-MPC2 significantly outperforms SAC in multi-task learning, achieving a 65% mean success rate compared to SAC's 37%, with notable improvements in path ratio. TD-MPC2 exhibited increased procedure times, suggesting a trade-off between success rate and execution speed. These findings highlight the potential of world models for improving autonomous endovascular navigation and lay the foundation for future research in generalizable AI-driven robotic interventions.
>
---
#### [new 059] Preemptive Spatiotemporal Trajectory Adjustment for Heterogeneous Vehicles in Highway Merging Zones
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于智能交通任务，解决高速公路合流区车辆协同控制问题。通过预判轨迹调整策略，提升交通效率与安全性。**

- **链接: [http://arxiv.org/pdf/2509.25929v1](http://arxiv.org/pdf/2509.25929v1)**

> **作者:** Yuan Li; Xiaoxue Xu; Xiang Dong; Junfeng Hao; Tao Li; Sana Ullaha; Chuangrui Huang; Junjie Niu; Ziyan Zhao; Ting Peng
>
> **摘要:** Aiming at the problem of driver's perception lag and low utilization efficiency of space-time resources in expressway ramp confluence area, based on the preemptive spatiotemporal trajectory Adjustment system, from the perspective of coordinating spatiotemporal resources, the reasonable value of safe space-time distance in trajectory pre-preparation is quantitatively analyzed. The minimum safety gap required for ramp vehicles to merge into the mainline is analyzed by introducing double positioning error and spatiotemporal trajectory tracking error. A merging control strategy for autonomous driving heterogeneous vehicles is proposed, which integrates vehicle type, driving intention, and safety spatiotemporal distance. The specific confluence strategies of ramp target vehicles and mainline cooperative vehicles under different vehicle types are systematically expounded. A variety of traffic flow and speed scenarios are used for full combination simulation. By comparing the time-position-speed diagram, the vehicle operation characteristics and the dynamic difference of confluence are qualitatively analyzed, and the average speed and average delay are used as the evaluation indices to quantitatively evaluate the performance advantages of the preemptive cooperative confluence control strategy. The results show that the maximum average delay improvement rates of mainline and ramp vehicles are 90.24 % and 74.24 %, respectively. The proposed strategy can effectively avoid potential vehicle conflicts and emergency braking behaviors, improve driving safety in the confluence area, and show significant advantages in driving stability and overall traffic efficiency optimization.
>
---
## 更新

#### [replaced 001] Towards autonomous photogrammetric forest inventory using a lightweight under-canopy robotic drone
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12073v3](http://arxiv.org/pdf/2501.12073v3)**

> **作者:** Väinö Karjalainen; Niko Koivumäki; Teemu Hakala; Jesse Muhojoki; Eric Hyyppä; Anand George; Juha Suomalainen; Eija Honkavaara
>
> **备注:** 35 pages, 11 Figures
>
> **摘要:** Drones are increasingly used in forestry to capture high-resolution remote sensing data, supporting enhanced monitoring, assessment, and decision-making processes. While operations above the forest canopy are already highly automated, flying inside forests remains challenging, primarily relying on manual piloting. In dense forests, relying on the Global Navigation Satellite System (GNSS) for localization is not feasible. In addition, the drone must autonomously adjust its flight path to avoid collisions. Recently, advancements in robotics have enabled autonomous drone flights in GNSS-denied obstacle-rich areas. In this article, a step towards autonomous forest data collection is taken by building a prototype of a robotic under-canopy drone utilizing state-of-the-art open source methods and validating its performance for data collection inside forests. Specifically, the study focused on camera-based autonomous flight under the forest canopy and photogrammetric post-processing of the data collected with the low-cost onboard stereo camera. The autonomous flight capability of the prototype was evaluated through multiple test flights in boreal forests. The tree parameter estimation capability was studied by performing diameter at breast height (DBH) estimation. The prototype successfully carried out flights in selected challenging forest environments, and the experiments showed promising performance in forest 3D modeling with a miniaturized stereoscopic photogrammetric system. The DBH estimation achieved a root mean square error (RMSE) of 3.33 - 3.97 cm (10.69 - 12.98 %) across all trees. For trees with a DBH less than 30 cm, the RMSE was 1.16 - 2.56 cm (5.74 - 12.47 %). The results provide valuable insights into autonomous under-canopy forest mapping and highlight the critical next steps for advancing lightweight robotic drone systems for mapping complex forest environments.
>
---
#### [replaced 002] DreamControl: Human-Inspired Whole-Body Humanoid Control for Scene Interaction via Guided Diffusion
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.14353v3](http://arxiv.org/pdf/2509.14353v3)**

> **作者:** Dvij Kalaria; Sudarshan S Harithas; Pushkal Katara; Sangkyung Kwak; Sarthak Bhagat; Shankar Sastry; Srinath Sridhar; Sai Vemprala; Ashish Kapoor; Jonathan Chung-Kuan Huang
>
> **备注:** https://genrobo.github.io/DreamControl/ (under submission)
>
> **摘要:** We introduce DreamControl, a novel methodology for learning autonomous whole-body humanoid skills. DreamControl leverages the strengths of diffusion models and Reinforcement Learning (RL): our core innovation is the use of a diffusion prior trained on human motion data, which subsequently guides an RL policy in simulation to complete specific tasks of interest (e.g., opening a drawer or picking up an object). We demonstrate that this human motion-informed prior allows RL to discover solutions unattainable by direct RL, and that diffusion models inherently promote natural looking motions, aiding in sim-to-real transfer. We validate DreamControl's effectiveness on a Unitree G1 robot across a diverse set of challenging tasks involving simultaneous lower and upper body control and object interaction. Project website at https://genrobo.github.io/DreamControl/
>
---
#### [replaced 003] WorldGym: World Model as An Environment for Policy Evaluation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.00613v3](http://arxiv.org/pdf/2506.00613v3)**

> **作者:** Julian Quevedo; Ansh Kumar Sharma; Yixiang Sun; Varad Suryavanshi; Percy Liang; Sherry Yang
>
> **备注:** https://world-model-eval.github.io
>
> **摘要:** Evaluating robot control policies is difficult: real-world testing is costly, and handcrafted simulators require manual effort to improve in realism and generality. We propose a world-model-based policy evaluation environment (WorldGym), an autoregressive, action-conditioned video generation model which serves as a proxy to real world environments. Policies are evaluated via Monte Carlo rollouts in the world model, with a vision-language model providing rewards. We evaluate a set of VLA-based real-robot policies in the world model using only initial frames from real robots, and show that policy success rates within the world model highly correlate with real-world success rates. Moreoever, we show that WorldGym is able to preserve relative policy rankings across different policy versions, sizes, and training checkpoints. Due to requiring only a single start frame as input, the world model further enables efficient evaluation of robot policies' generalization ability on novel tasks and environments. We find that modern VLA-based robot policies still struggle to distinguish object shapes and can become distracted by adversarial facades of objects. While generating highly realistic object interaction remains challenging, WorldGym faithfully emulates robot motions and offers a practical starting point for safe and reproducible policy evaluation before deployment.
>
---
#### [replaced 004] Ocean Diviner: A Diffusion-Augmented Reinforcement Learning Framework for AUV Robust Control in Underwater Tasks
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.11283v2](http://arxiv.org/pdf/2507.11283v2)**

> **作者:** Jingzehua Xu; Guanwen Xie; Weiyi Liu; Jiwei Tang; Ziteng Yang; Tianxiang Xing; Yiyuan Yang; Shuai Zhang; Xiaofan Li
>
> **备注:** Jingzehua Xu, Guanwen Xie and Weiyi Liu contributed equally to this work
>
> **摘要:** Autonomous Underwater Vehicles (AUVs) are essential for marine exploration, yet their control remains highly challenging due to nonlinear dynamics and uncertain environmental disturbances. This paper presents a diffusion-augmented Reinforcement Learning (RL) framework for robust AUV control, aiming to improve AUV's adaptability in dynamic underwater environments. The proposed framework integrates two core innovations: (1) A diffusion-based action generation framework that produces physically feasible and high-quality actions, enhanced by a high-dimensional state encoding mechanism combining current observations with historical states and actions through a novel diffusion U-Net architecture, significantly improving long-horizon planning capacity for robust control. (2) A sample-efficient hybrid learning architecture that synergizes diffusion-guided exploration with RL policy optimization, where the diffusion model generates diverse candidate actions and the RL critic selects the optimal action, achieving higher exploration efficiency and policy stability in dynamic underwater environments. Extensive simulation experiments validate the framework's superior robustness and flexibility, outperforming conventional control methods in challenging marine conditions, offering enhanced adaptability and reliability for AUV operations in underwater tasks. Finally, we will release the code publicly soon to support future research in this area.
>
---
#### [replaced 005] Multi-Robot Task Planning for Multi-Object Retrieval Tasks with Distributed On-Site Knowledge via Large Language Models
- **分类: cs.RO; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.12838v2](http://arxiv.org/pdf/2509.12838v2)**

> **作者:** Kento Murata; Shoichi Hasegawa; Tomochika Ishikawa; Yoshinobu Hagiwara; Akira Taniguchi; Lotfi El Hafi; Tadahiro Taniguchi
>
> **备注:** Submitted to AROB-ISBC 2026 (Journal Track option)
>
> **摘要:** It is crucial to efficiently execute instructions such as "Find an apple and a banana" or "Get ready for a field trip," which require searching for multiple objects or understanding context-dependent commands. This study addresses the challenging problem of determining which robot should be assigned to which part of a task when each robot possesses different situational on-site knowledge-specifically, spatial concepts learned from the area designated to it by the user. We propose a task planning framework that leverages large language models (LLMs) and spatial concepts to decompose natural language instructions into subtasks and allocate them to multiple robots. We designed a novel few-shot prompting strategy that enables LLMs to infer required objects from ambiguous commands and decompose them into appropriate subtasks. In our experiments, the proposed method achieved 47/50 successful assignments, outperforming random (28/50) and commonsense-based assignment (26/50). Furthermore, we conducted qualitative evaluations using two actual mobile manipulators. The results demonstrated that our framework could handle instructions, including those involving ad hoc categories such as "Get ready for a field trip," by successfully performing task decomposition, assignment, sequential planning, and execution.
>
---
#### [replaced 006] Robot Conga: A Leader-Follower Walking Approach to Sequential Path Following in Multi-Agent Systems
- **分类: cs.RO; cs.SY; eess.SY; math.DS; physics.app-ph; 49; I.2.9**

- **链接: [http://arxiv.org/pdf/2509.16482v2](http://arxiv.org/pdf/2509.16482v2)**

> **作者:** Pranav Tiwari; Soumyodipta Nath
>
> **备注:** 6 Pages, 8 Figures. Both authors have contributed equally
>
> **摘要:** Coordinated path following in multi-agent systems is a key challenge in robotics, with applications in automated logistics, surveillance, and collaborative exploration. Traditional formation control techniques often rely on time-parameterized trajectories and path integrals, which can result in synchronization issues and rigid behavior. In this work, we address the problem of sequential path following, where agents maintain fixed spatial separation along a common trajectory, guided by a leader under centralized control. We introduce Robot Conga, a leader-follower control strategy that updates each agent's desired state based on the leader's spatial displacement rather than time, assuming access to a global position reference, an assumption valid in indoor environments equipped with motion capture, vision-based tracking, or UWB localization systems. The algorithm was validated in simulation using both TurtleBot3 and quadruped (Laikago) robots. Results demonstrate accurate trajectory tracking, stable inter-agent spacing, and fast convergence, with all agents aligning within 250 time steps (approx. 0.25 seconds) in the quadruped case, and almost instantaneously in the TurtleBot3 implementation.
>
---
#### [replaced 007] ADPro: a Test-time Adaptive Diffusion Policy via Manifold-constrained Denoising and Task-aware Initialization for Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.06266v2](http://arxiv.org/pdf/2508.06266v2)**

> **作者:** Zezeng Li; Rui Yang; Ruochen Chen; ZhongXuan Luo; Liming Chen
>
> **摘要:** Diffusion policies have recently emerged as a powerful class of visuomotor controllers for robot manipulation, offering stable training and expressive multi-modal action modeling. However, existing approaches typically treat action generation as an unconstrained denoising process, ignoring valuable a priori knowledge about geometry and control structure. In this work, we propose the Adaptive Diffusion Policy (ADP), a test-time adaptation method that introduces two key inductive biases into the diffusion. First, we embed a geometric manifold constraint that aligns denoising updates with task-relevant subspaces, leveraging the fact that the relative pose between the end-effector and target scene provides a natural gradient direction, and guiding denoising along the geodesic path of the manipulation manifold. Then, to reduce unnecessary exploration and accelerate convergence, we propose an analytically guided initialization: rather than sampling from an uninformative prior, we compute a rough registration between the gripper and target scenes to propose a structured initial noisy action. ADP is compatible with pre-trained diffusion policies and requires no retraining, enabling test-time adaptation that tailors the policy to specific tasks, thereby enhancing generalization across novel tasks and environments. Experiments on RLBench, CALVIN, and real-world datasets show that ADPro, an implementation of ADP, improves success rates, generalization, and sampling efficiency, achieving up to 25% faster execution and 9% points over strong diffusion baselines.
>
---
#### [replaced 008] Trajectory Encryption Cooperative Salvo Guidance
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2509.17341v2](http://arxiv.org/pdf/2509.17341v2)**

> **作者:** Lohitvel Gopikannan; Shashi Ranjan Kumar; Abhinav Sinha
>
> **摘要:** This paper introduces the concept of trajectory encryption in cooperative simultaneous target interception, wherein heterogeneity in guidance principles across a team of unmanned autonomous systems is leveraged as a strategic design feature. By employing a mix of heterogeneous time-to-go formulations leading to a cooperative guidance strategy, the swarm of vehicles is able to generate diverse trajectory families. This diversity expands the feasible solution space for simultaneous target interception, enhances robustness under disturbances, and enables flexible time-to-go adjustments without predictable detouring. From an adversarial perspective, heterogeneity obscures the collective interception intent by preventing straightforward prediction of swarm dynamics, effectively acting as an encryption layer in the trajectory domain. Simulations demonstrate that the swarm of heterogeneous vehicles is able to intercept a moving target simultaneously from a diverse set of initial engagement configurations.
>
---
#### [replaced 009] SoMi-ToM: Evaluating Multi-Perspective Theory of Mind in Embodied Social Interactions
- **分类: cs.CL; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23046v2](http://arxiv.org/pdf/2506.23046v2)**

> **作者:** Xianzhe Fan; Xuhui Zhou; Chuanyang Jin; Kolby Nottingham; Hao Zhu; Maarten Sap
>
> **备注:** 24 pages, 6 figures
>
> **摘要:** Humans continuously infer the states, goals, and behaviors of others by perceiving their surroundings in dynamic, real-world social interactions. However, most Theory of Mind (ToM) benchmarks only evaluate static, text-based scenarios, which have a significant gap compared to real interactions. We propose the SoMi-ToM benchmark, designed to evaluate multi-perspective ToM in embodied multi-agent complex social interactions. This benchmark is based on rich multimodal interaction data generated by the interaction environment SoMi, covering diverse crafting goals and social relationships. Our framework supports multi-level evaluation: (1) first-person evaluation provides multimodal (visual, dialogue, action, etc.) input from a first-person perspective during a task for real-time state inference, (2) third-person evaluation provides complete third-person perspective video and text records after a task for goal and behavior inference. This evaluation method allows for a more comprehensive examination of a model's ToM capabilities from both the subjective immediate experience and the objective global observation. We constructed a challenging dataset containing 35 third-person perspective videos, 363 first-person perspective images, and 1225 expert-annotated multiple-choice questions (three options). On this dataset, we systematically evaluated the performance of human subjects and several state-of-the-art large vision-language models (LVLMs). The results show that LVLMs perform significantly worse than humans on SoMi-ToM: the average accuracy gap between humans and models is 40.1% in first-person evaluation and 26.4% in third-person evaluation. This indicates that future LVLMs need to further improve their ToM capabilities in embodied, complex social interactions.
>
---
#### [replaced 010] Electrostatic Clutch-Based Mechanical Multiplexer with Increased Force Capability
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2501.08469v3](http://arxiv.org/pdf/2501.08469v3)**

> **作者:** Timothy E. Amish; Jeffrey T. Auletta; Chad C. Kessens; Joshua R. Smith; Jeffrey I. Lipton
>
> **摘要:** Robotic systems with many degrees of freedom (DoF) are constrained by the demands of dedicating a motor to each joint, and while mechanical multiplexing reduces actuator count, existing clutch designs are bulky, force-limited, or restricted to one output at a time. The problem addressed in this study is how to achieve high-force, multiplexing that supports both simultaneous and sequential control from a single motor. Here we show an electrostatic capstan clutch-based transmission that enables both single-input-single-output (SISO) and single-input-multiple-output (SIMO) multiplexing. We demonstrated these on a four-DoF tendon-driven robotic hand where a single motor achieved output forces up to 212 N, increased vertical grip strength by 4.09 times, and raised horizontal carrying capacity by 354\% over manufacturer specifications. These results demonstrate that electrostatic multiplexing provides versatile actuation, overcoming the limitations of prior systems.
>
---
#### [replaced 011] AuDeRe: Automated Strategy Decision and Realization in Robot Planning and Control via LLMs
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.03015v2](http://arxiv.org/pdf/2504.03015v2)**

> **作者:** Yue Meng; Fei Chen; Yongchao Chen; Chuchu Fan
>
> **备注:** 8 pages, 14 figures, submitted to the 2026 American Control Conference
>
> **摘要:** Recent advancements in large language models (LLMs) have shown significant promise in various domains, especially robotics. However, most prior LLM-based work in robotic applications either directly predicts waypoints or applies LLMs within fixed tool integration frameworks, offering limited flexibility in exploring and configuring solutions best suited to different tasks. In this work, we propose a framework that leverages LLMs to select appropriate planning and control strategies based on task descriptions, environmental constraints, and system dynamics. These strategies are then executed by calling the available comprehensive planning and control APIs. Our approach employs iterative LLM-based reasoning with performance feedback to refine the algorithm selection. We validate our approach through extensive experiments across tasks of varying complexity, from simple tracking to complex planning scenarios involving spatiotemporal constraints. The results demonstrate that using LLMs to determine planning and control strategies from natural language descriptions significantly enhances robotic autonomy while reducing the need for extensive manual tuning and expert knowledge. Furthermore, our framework maintains generalizability across different tasks and notably outperforms baseline methods that rely on LLMs for direct trajectory, control sequence, or code generation.
>
---
#### [replaced 012] DAGDiff: Guiding Dual-Arm Grasp Diffusion to Stable and Collision-Free Grasps
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.21145v2](http://arxiv.org/pdf/2509.21145v2)**

> **作者:** Md Faizal Karim; Vignesh Vembar; Keshab Patra; Gaurav Singh; K Madhava Krishna
>
> **摘要:** Reliable dual-arm grasping is essential for manipulating large and complex objects but remains a challenging problem due to stability, collision, and generalization requirements. Prior methods typically decompose the task into two independent grasp proposals, relying on region priors or heuristics that limit generalization and provide no principled guarantee of stability. We propose DAGDiff, an end-to-end framework that directly denoises to grasp pairs in the SE(3) x SE(3) space. Our key insight is that stability and collision can be enforced more effectively by guiding the diffusion process with classifier signals, rather than relying on explicit region detection or object priors. To this end, DAGDiff integrates geometry-, stability-, and collision-aware guidance terms that steer the generative process toward grasps that are physically valid and force-closure compliant. We comprehensively evaluate DAGDiff through analytical force-closure checks, collision analysis, and large-scale physics-based simulations, showing consistent improvements over previous work on these metrics. Finally, we demonstrate that our framework generates dual-arm grasps directly on real-world point clouds of previously unseen objects, which are executed on a heterogeneous dual-arm setup where two manipulators reliably grasp and lift them.
>
---
#### [replaced 013] TaBSA -- A framework for training and benchmarking algorithms scheduling tasks for mobile robots working in dynamic environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.16844v3](http://arxiv.org/pdf/2408.16844v3)**

> **作者:** Wojciech Dudek; Daniel Giełdowski; Dominik Belter; Kamil Młodzikowski; Tomasz Winiarski
>
> **备注:** Article submitted to the SoftwareX journal
>
> **摘要:** This article introduces a software framework for benchmarking robot task scheduling algorithms in dynamic and uncertain service environments. The system provides standardized interfaces, configurable scenarios with movable objects, human agents, tools for automated test generation, and performance evaluation. It supports both classical and AI-based methods, enabling repeatable, comparable assessments across diverse tasks and configurations. The framework facilitates diagnosis of algorithm behavior, identification of implementation flaws, and selection or tuning of strategies for specific applications. It includes a SysML-based domain-specific language for structured scenario modeling and integrates with the ROS-based system for runtime execution. Validated on patrol, fall assistance, and pick-and-place tasks, the open-source framework is suited for researchers and integrators developing and testing scheduling algorithms under real-world-inspired conditions.
>
---
#### [replaced 014] Find the Fruit: Zero-Shot Sim2Real RL for Occlusion-Aware Plant Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.16547v2](http://arxiv.org/pdf/2505.16547v2)**

> **作者:** Nitesh Subedi; Hsin-Jung Yang; Devesh K. Jha; Soumik Sarkar
>
> **备注:** 9 Pages, 3 Figures, 1 Table
>
> **摘要:** Autonomous harvesting in the open presents a complex manipulation problem. In most scenarios, an autonomous system has to deal with significant occlusion and require interaction in the presence of large structural uncertainties (every plant is different). Perceptual and modeling uncertainty make design of reliable manipulation controllers for harvesting challenging, resulting in poor performance during deployment. We present a sim2real reinforcement learning (RL) framework for occlusion-aware plant manipulation, where a policy is learned entirely in simulation to reposition stems and leaves to reveal target fruit(s). In our proposed approach, we decouple high-level kinematic planning from low-level compliant control which simplifies the sim2real transfer. This decomposition allows the learned policy to generalize across multiple plants with different stiffness and morphology. In experiments with multiple real-world plant setups, our system achieves up to 86.7% success in exposing target fruits, demonstrating robustness to occlusion variation and structural uncertainty.
>
---
#### [replaced 015] Visual-auditory Extrinsic Contact Estimation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.14608v3](http://arxiv.org/pdf/2409.14608v3)**

> **作者:** Xili Yi; Jayjun Lee; Nima Fazeli
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Robust manipulation often hinges on a robot's ability to perceive extrinsic contacts-contacts between a grasped object and its surrounding environment. However, these contacts are difficult to observe through vision alone due to occlusions, limited resolution, and ambiguous near-contact states. In this paper, we propose a visual-auditory method for extrinsic contact estimation that integrates global scene information from vision with local contact cues obtained through active audio sensing. Our approach equips a robotic gripper with contact microphones and conduction speakers, enabling the system to emit and receive acoustic signals through the grasped object to detect external contacts. We train our perception pipeline entirely in simulation and zero-shot transfer to the real world. To bridge the sim-to-real gap, we introduce a real-to-sim audio hallucination technique, injecting real-world audio samples into simulated scenes with ground-truth contact labels. The resulting multimodal model accurately estimates both the location and size of extrinsic contacts across a range of cluttered and occluded scenarios. We further demonstrate that explicit contact prediction significantly improves policy learning for downstream contact-rich manipulation tasks.
>
---
#### [replaced 016] Video models are zero-shot learners and reasoners
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20328v2](http://arxiv.org/pdf/2509.20328v2)**

> **作者:** Thaddäus Wiedemer; Yuxuan Li; Paul Vicol; Shixiang Shane Gu; Nick Matarese; Kevin Swersky; Been Kim; Priyank Jaini; Robert Geirhos
>
> **备注:** Project page: https://video-zero-shot.github.io/
>
> **摘要:** The remarkable zero-shot capabilities of Large Language Models (LLMs) have propelled natural language processing from task-specific models to unified, generalist foundation models. This transformation emerged from simple primitives: large, generative models trained on web-scale data. Curiously, the same primitives apply to today's generative video models. Could video models be on a trajectory towards general-purpose vision understanding, much like LLMs developed general-purpose language understanding? We demonstrate that Veo 3 can solve a broad variety of tasks it wasn't explicitly trained for: segmenting objects, detecting edges, editing images, understanding physical properties, recognizing object affordances, simulating tool use, and more. These abilities to perceive, model, and manipulate the visual world enable early forms of visual reasoning like maze and symmetry solving. Veo's emergent zero-shot capabilities indicate that video models are on a path to becoming unified, generalist vision foundation models.
>
---
#### [replaced 017] VIMD: Monocular Visual-Inertial Motion and Depth Estimation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.19713v2](http://arxiv.org/pdf/2509.19713v2)**

> **作者:** Saimouli Katragadda; Guoquan Huang
>
> **摘要:** Accurate and efficient dense metric depth estimation is crucial for 3D visual perception in robotics and XR. In this paper, we develop a monocular visual-inertial motion and depth (VIMD) learning framework to estimate dense metric depth by leveraging accurate and efficient MSCKF-based monocular visual-inertial motion tracking. At the core the proposed VIMD is to exploit multi-view information to iteratively refine per-pixel scale, instead of globally fitting an invariant affine model as in the prior work. The VIMD framework is highly modular, making it compatible with a variety of existing depth estimation backbones. We conduct extensive evaluations on the TartanAir and VOID datasets and demonstrate its zero-shot generalization capabilities on the AR Table dataset. Our results show that VIMD achieves exceptional accuracy and robustness, even with extremely sparse points as few as 10-20 metric depth points per image. This makes the proposed VIMD a practical solution for deployment in resource constrained settings, while its robust performance and strong generalization capabilities offer significant potential across a wide range of scenarios.
>
---
#### [replaced 018] Simulated Annealing for Multi-Robot Ergodic Information Acquisition Using Graph-Based Discretization
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.23214v2](http://arxiv.org/pdf/2509.23214v2)**

> **作者:** Benjamin Wong; Aaron Weber; Mohamed M. Safwat; Santosh Devasia; Ashis G. Banerjee
>
> **摘要:** One of the goals of active information acquisition using multi-robot teams is to keep the relative uncertainty in each region at the same level to maintain identical acquisition quality (e.g., consistent target detection) in all the regions. To achieve this goal, ergodic coverage can be used to assign the number of samples according to the quality of observation, i.e., sampling noise levels. However, the noise levels are unknown to the robots. Although this noise can be estimated from samples, the estimates are unreliable at first and can generate fluctuating values. The main contribution of this paper is to use simulated annealing to generate the target sampling distribution, starting from uniform and gradually shifting to an estimated optimal distribution, by varying the coldness parameter of a Boltzmann distribution with the estimated sampling entropy as energy. Simulation results show a substantial improvement of both transient and asymptotic entropy compared to both uniform and direct-ergodic searches. Finally, a demonstration is performed with a TurtleBot swarm system to validate the physical applicability of the algorithm.
>
---
#### [replaced 019] Multi Layered Autonomy and AI Ecologies in Robotic Art Installations
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.02606v4](http://arxiv.org/pdf/2506.02606v4)**

> **作者:** Baoyang Chen; Xian Xu; Huamin Qu
>
> **摘要:** This paper presents Symbiosis of Agents, is a large-scale installation by Baoyang Chen (baoyangchen.com), that embeds AI-driven robots in an immersive, mirror-lined arena, probing the tension between machine agency and artistic authorship. Drawing on early cybernetics, rule-based conceptual art, and seminal robotic works, it orchestrates fluid exchanges among robotic arms, quadruped machines, their environment, and the public. A three tier faith system pilots the ecology: micro-level adaptive tactics, meso-level narrative drives, and a macro-level prime directive. This hierarchy lets behaviors evolve organically in response to environmental cues and even a viewer's breath, turning spectators into co-authors of the unfolding drama. Framed by a speculative terraforming scenario that recalls the historical exploitation of marginalized labor, the piece asks who bears responsibility in AI-mediated futures. Choreographed motion, AI-generated scripts, reactive lighting, and drifting fog cast the robots as collaborators rather than tools, forging a living, emergent artwork. Exhibited internationally, Symbiosis of Agents shows how cybernetic feedback, robotic experimentation, and conceptual rule-making can converge to redefine agency, authorship, and ethics in contemporary art.
>
---
#### [replaced 020] Apple: Toward General Active Perception via Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.06182v3](http://arxiv.org/pdf/2505.06182v3)**

> **作者:** Tim Schneider; Cristiana de Farias; Roberto Calandra; Liming Chen; Jan Peters
>
> **备注:** 16 pages; 13 figures Under Review
>
> **摘要:** Active perception is a fundamental skill that enables us humans to deal with uncertainty in our inherently partially observable environment. For senses such as touch, where the information is sparse and local, active perception becomes crucial. In recent years, active perception has emerged as an important research domain in robotics. However, current methods are often bound to specific tasks or make strong assumptions, which limit their generality. To address this gap, this work introduces APPLE (Active Perception Policy Learning) - a novel framework that leverages reinforcement learning (RL) to address a range of different active perception problems. APPLE jointly trains a transformer-based perception module and decision-making policy with a unified optimization objective, learning how to actively gather information. By design, APPLE is not limited to a specific task and can, in principle, be applied to a wide range of active perception problems. We evaluate two variants of APPLE across different tasks, including tactile exploration problems from the Tactile MNIST benchmark. Experiments demonstrate the efficacy of APPLE, achieving high accuracies on both regression and classification tasks. These findings underscore the potential of APPLE as a versatile and general framework for advancing active perception in robotics.
>
---
#### [replaced 021] OrthoLoC: UAV 6-DoF Localization and Calibration Using Orthographic Geodata
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.18350v2](http://arxiv.org/pdf/2509.18350v2)**

> **作者:** Oussema Dhaouadi; Riccardo Marin; Johannes Meier; Jacques Kaiser; Daniel Cremers
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Accurate visual localization from aerial views is a fundamental problem with applications in mapping, large-area inspection, and search-and-rescue operations. In many scenarios, these systems require high-precision localization while operating with limited resources (e.g., no internet connection or GNSS/GPS support), making large image databases or heavy 3D models impractical. Surprisingly, little attention has been given to leveraging orthographic geodata as an alternative paradigm, which is lightweight and increasingly available through free releases by governmental authorities (e.g., the European Union). To fill this gap, we propose OrthoLoC, the first large-scale dataset comprising 16,425 UAV images from Germany and the United States with multiple modalities. The dataset addresses domain shifts between UAV imagery and geospatial data. Its paired structure enables fair benchmarking of existing solutions by decoupling image retrieval from feature matching, allowing isolated evaluation of localization and calibration performance. Through comprehensive evaluation, we examine the impact of domain shifts, data resolutions, and covisibility on localization accuracy. Finally, we introduce a refinement technique called AdHoP, which can be integrated with any feature matcher, improving matching by up to 95% and reducing translation error by up to 63%. The dataset and code are available at: https://deepscenario.github.io/OrthoLoC.
>
---
#### [replaced 022] LiDAR-BIND-T: Improved and Temporally Consistent Sensor Modality Translation and Fusion for Robotic Applications
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.05728v3](http://arxiv.org/pdf/2509.05728v3)**

> **作者:** Niels Balemans; Ali Anwar; Jan Steckel; Siegfried Mercelis
>
> **摘要:** This paper extends LiDAR-BIND, a modular multi-modal fusion framework that binds heterogeneous sensors (radar, sonar) to a LiDAR-defined latent space, with mechanisms that explicitly enforce temporal consistency. We introduce three contributions: (i) temporal embedding similarity that aligns consecutive latent representations, (ii) a motion-aligned transformation loss that matches displacement between predictions and ground truth LiDAR, and (iii) windowed temporal fusion using a specialised temporal module. We further update the model architecture to better preserve spatial structure. Evaluations on radar/sonar-to-LiDAR translation demonstrate improved temporal and spatial coherence, yielding lower absolute trajectory error and better occupancy map accuracy in Cartographer-based SLAM (Simultaneous Localisation and Mapping). We propose different metrics based on the Fr\'echet Video Motion Distance (FVMD) and a correlation-peak distance metric providing practical temporal quality indicators to evaluate SLAM performance. The proposed temporal LiDAR-BIND, or LiDAR-BIND-T, maintains modular modality fusion while substantially enhancing temporal stability, resulting in improved robustness and performance for downstream SLAM.
>
---
#### [replaced 023] Long-Horizon Visual Imitation Learning via Plan and Code Reflection
- **分类: cs.RO; cs.AI; cs.LG; I.2.9; I.2.10**

- **链接: [http://arxiv.org/pdf/2509.05368v2](http://arxiv.org/pdf/2509.05368v2)**

> **作者:** Quan Chen; Chenrui Shi; Qi Chen; Yuwei Wu; Zhi Gao; Xintong Zhang; Rui Gao; Kun Wu; Yunde Jia
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Learning from long-horizon demonstrations with complex action sequences presents significant challenges for visual imitation learning, particularly in understanding temporal relationships of actions and spatial relationships between objects. In this paper, we propose a new agent framework that incorporates two dedicated reflection modules to enhance both plan and code generation. The plan generation module produces an initial action sequence, which is then verified by the plan reflection module to ensure temporal coherence and spatial alignment with the demonstration video. The code generation module translates the plan into executable code, while the code reflection module verifies and refines the generated code to ensure correctness and consistency with the generated plan. These two reflection modules jointly enable the agent to detect and correct errors in both the plan generation and code generation, improving performance in tasks with intricate temporal and spatial dependencies. To support systematic evaluation, we introduce LongVILBench, a benchmark comprising 300 human demonstrations with action sequences of up to 18 steps. LongVILBench emphasizes temporal and spatial complexity across multiple task types. Experimental results demonstrate that existing methods perform poorly on this benchmark, whereas our new framework establishes a strong baseline for long-horizon visual imitation learning.
>
---
#### [replaced 024] AgriCruiser: An Open Source Agriculture Robot for Over-the-row Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.25056v2](http://arxiv.org/pdf/2509.25056v2)**

> **作者:** Kenny Truong; Yongkyu Lee; Jason Irie; Shivam Kumar Panda; Mohammad Jony; Shahab Ahmad; Md. Mukhlesur Rahman; M. Khalid Jawed
>
> **备注:** GitHub: https://github.com/structuresComp/agri-cruiser
>
> **摘要:** We present the AgriCruiser, an open-source over-the-row agricultural robot developed for low-cost deployment and rapid adaptation across diverse crops and row layouts. The chassis provides an adjustable track width of 1.42 m to 1.57 m, along with a ground clearance of 0.94 m. The AgriCruiser achieves compact pivot turns with radii of 0.71 m to 0.79 m, enabling efficient headland maneuvers. The platform is designed for the integration of the other subsystems, and in this study, a precision spraying system was implemented to assess its effectiveness in weed management. In twelve flax plots, a single robotic spray pass reduced total weed populations (pigweed and Venice mallow) by 24- to 42-fold compared to manual weeding in four flax plots, while also causing less crop damage. Mobility experiments conducted on concrete, asphalt, gravel, grass, and both wet and dry soil confirmed reliable traversal consistent with torque sizing. The complete chassis can be constructed from commodity T-slot extrusion with minimal machining, resulting in a bill of materials costing approximately $5,000 - $6,000, which enables replication and customization. The mentioned results demonstrate that low-cost, reconfigurable over-the-row robots can achieve effective weed management with reduced crop damage and labor requirements, while providing a versatile foundation for phenotyping, sensing, and other agriculture applications. Design files and implementation details are released to accelerate research and adoption of modular agricultural robotics.
>
---
#### [replaced 025] LargeAD: Large-Scale Cross-Sensor Data Pretraining for Autonomous Driving
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.04005v2](http://arxiv.org/pdf/2501.04005v2)**

> **作者:** Lingdong Kong; Xiang Xu; Youquan Liu; Jun Cen; Runnan Chen; Wenwei Zhang; Liang Pan; Kai Chen; Ziwei Liu
>
> **备注:** IEEE TPAMI 2025; 17 pages, 9 figures, 11 tables; Project Page at https://ldkong.com/LargeAD
>
> **摘要:** Recent advancements in vision foundation models (VFMs) have revolutionized visual perception in 2D, yet their potential for 3D scene understanding, particularly in autonomous driving applications, remains underexplored. In this paper, we introduce LargeAD, a versatile and scalable framework designed for large-scale 3D pretraining across diverse real-world driving datasets. Our framework leverages VFMs to extract semantically rich superpixels from 2D images, which are aligned with LiDAR point clouds to generate high-quality contrastive samples. This alignment facilitates cross-modal representation learning, enhancing the semantic consistency between 2D and 3D data. We introduce several key innovations: (i) VFM-driven superpixel generation for detailed semantic representation, (ii) a VFM-assisted contrastive learning strategy to align multimodal features, (iii) superpoint temporal consistency to maintain stable representations across time, and (iv) multi-source data pretraining to generalize across various LiDAR configurations. Our approach achieves substantial gains over state-of-the-art methods in linear probing and fine-tuning for LiDAR-based segmentation and object detection. Extensive experiments on 11 large-scale multi-sensor datasets highlight our superior performance, demonstrating adaptability, efficiency, and robustness in real-world autonomous driving scenarios.
>
---
#### [replaced 026] Controllable Motion Generation via Diffusion Modal Coupling
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.02353v2](http://arxiv.org/pdf/2503.02353v2)**

> **作者:** Luobin Wang; Hongzhan Yu; Chenning Yu; Sicun Gao; Henrik Christensen
>
> **摘要:** Diffusion models have recently gained significant attention in robotics due to their ability to generate multi-modal distributions of system states and behaviors. However, a key challenge remains: ensuring precise control over the generated outcomes without compromising realism. This is crucial for applications such as motion planning or trajectory forecasting, where adherence to physical constraints and task-specific objectives is essential. We propose a novel framework that enhances controllability in diffusion models by leveraging multi-modal prior distributions and enforcing strong modal coupling. This allows us to initiate the denoising process directly from distinct prior modes that correspond to different possible system behaviors, ensuring sampling to align with the training distribution. We evaluate our approach on motion prediction using the Waymo dataset and multi-task control in Maze2D environments. Experimental results show that our framework outperforms both guidance-based techniques and conditioned models with unimodal priors, achieving superior fidelity, diversity, and controllability, even in the absence of explicit conditioning. Overall, our approach provides a more reliable and scalable solution for controllable motion generation in robotics.
>
---
#### [replaced 027] Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.23778v2](http://arxiv.org/pdf/2509.23778v2)**

> **作者:** Zeyuan Zhao; Chaoran Li; Shao Zhang; Ying Wen
>
> **备注:** Preprint Under Review
>
> **摘要:** Multi-Agent Pickup and Delivery (MAPD) is a challenging extension of Multi-Agent Path Finding (MAPF), where agents are required to sequentially complete tasks with fixed-location pickup and delivery demands. Although learning-based methods have made progress in MAPD, they often perform poorly in warehouse-like environments with narrow pathways and long corridors when relying only on local observations for distributed decision-making. Communication learning can alleviate the lack of global information but introduce high computational complexity due to point-to-point communication. To address this challenge, we formulate MAPF as a sequence modeling problem and prove that path-finding policies under sequence modeling possess order-invariant optimality, ensuring its effectiveness in MAPD. Building on this, we propose the Sequential Pathfinder (SePar), which leverages the Transformer paradigm to achieve implicit information exchange, reducing decision-making complexity from exponential to linear while maintaining efficiency and global awareness. Experiments demonstrate that SePar consistently outperforms existing learning-based methods across various MAPF tasks and their variants, and generalizes well to unseen environments. Furthermore, we highlight the necessity of integrating imitation learning in complex maps like warehouses.
>
---
#### [replaced 028] Track Any Motions under Any Disturbances
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.13833v3](http://arxiv.org/pdf/2509.13833v3)**

> **作者:** Zhikai Zhang; Jun Guo; Chao Chen; Jilong Wang; Chenghuai Lin; Yunrui Lian; Han Xue; Zhenrong Wang; Maoqi Liu; Jiangran Lyu; Huaping Liu; He Wang; Li Yi
>
> **摘要:** A foundational humanoid motion tracker is expected to be able to track diverse, highly dynamic, and contact-rich motions. More importantly, it needs to operate stably in real-world scenarios against various dynamics disturbances, including terrains, external forces, and physical property changes for general practical use. To achieve this goal, we propose Any2Track (Track Any motions under Any disturbances), a two-stage RL framework to track various motions under multiple disturbances in the real world. Any2Track reformulates dynamics adaptability as an additional capability on top of basic action execution and consists of two key components: AnyTracker and AnyAdapter. AnyTracker is a general motion tracker with a series of careful designs to track various motions within a single policy. AnyAdapter is a history-informed adaptation module that endows the tracker with online dynamics adaptability to overcome the sim2real gap and multiple real-world disturbances. We deploy Any2Track on Unitree G1 hardware and achieve a successful sim2real transfer in a zero-shot manner. Any2Track performs exceptionally well in tracking various motions under multiple real-world disturbances.
>
---
