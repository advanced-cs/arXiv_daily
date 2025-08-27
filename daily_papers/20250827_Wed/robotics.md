# 机器人 cs.RO

- **最新发布 35 篇**

- **更新 25 篇**

## 最新发布

#### [new 001] AgriChrono: A Multi-modal Dataset Capturing Crop Growth and Lighting Variability with a Field Robot
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 论文提出AgriChrono平台，通过多传感器采集实时动态农业数据，解决现有数据集无法反映真实农田光照、作物生长变化的问题，验证了3D重建模型在复杂环境中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.18694v1](http://arxiv.org/pdf/2508.18694v1)**

> **作者:** Jaehwan Jeong; Tuan-Anh Vu; Mohammad Jony; Shahab Ahmad; Md. Mukhlesur Rahman; Sangpil Kim; M. Khalid Jawed
>
> **摘要:** Existing datasets for precision agriculture have primarily been collected in static or controlled environments such as indoor labs or greenhouses, often with limited sensor diversity and restricted temporal span. These conditions fail to reflect the dynamic nature of real farmland, including illumination changes, crop growth variation, and natural disturbances. As a result, models trained on such data often lack robustness and generalization when applied to real-world field scenarios. In this paper, we present AgriChrono, a novel robotic data collection platform and multi-modal dataset designed to capture the dynamic conditions of real-world agricultural environments. Our platform integrates multiple sensors and enables remote, time-synchronized acquisition of RGB, Depth, LiDAR, and IMU data, supporting efficient and repeatable long-term data collection across varying illumination and crop growth stages. We benchmark a range of state-of-the-art 3D reconstruction models on the AgriChrono dataset, highlighting the difficulty of reconstruction in real-world field environments and demonstrating its value as a research asset for advancing model generalization under dynamic conditions. The code and dataset are publicly available at: https://github.com/StructuresComp/agri-chrono
>
---
#### [new 002] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出MemoryVLA，用于机器人操作中的长时序任务。针对传统VLA忽略时间上下文的问题，设计感知-认知记忆框架，通过预训练模型生成并整合感知与认知标记，实现时间感知的动作生成，在多个任务中取得更高成功率。**

- **链接: [http://arxiv.org/pdf/2508.19236v1](http://arxiv.org/pdf/2508.19236v1)**

> **作者:** Hao Shi; Bin Xie; Yingfei Liu; Lin Sun; Fengrong Liu; Tiancai Wang; Erjin Zhou; Haoqiang Fan; Xiangyu Zhang; Gao Huang
>
> **备注:** The project is available at https://shihao1895.github.io/MemoryVLA
>
> **摘要:** Temporal context is essential for robotic manipulation because such tasks are inherently non-Markovian, yet mainstream VLA models typically overlook it and struggle with long-horizon, temporally dependent tasks. Cognitive science suggests that humans rely on working memory to buffer short-lived representations for immediate control, while the hippocampal system preserves verbatim episodic details and semantic gist of past experience for long-term memory. Inspired by these mechanisms, we propose MemoryVLA, a Cognition-Memory-Action framework for long-horizon robotic manipulation. A pretrained VLM encodes the observation into perceptual and cognitive tokens that form working memory, while a Perceptual-Cognitive Memory Bank stores low-level details and high-level semantics consolidated from it. Working memory retrieves decision-relevant entries from the bank, adaptively fuses them with current tokens, and updates the bank by merging redundancies. Using these tokens, a memory-conditioned diffusion action expert yields temporally aware action sequences. We evaluate MemoryVLA on 150+ simulation and real-world tasks across three robots. On SimplerEnv-Bridge, Fractal, and LIBERO-5 suites, it achieves 71.9%, 72.7%, and 96.5% success rates, respectively, all outperforming state-of-the-art baselines CogACT and pi-0, with a notable +14.6 gain on Bridge. On 12 real-world tasks spanning general skills and long-horizon temporal dependencies, MemoryVLA achieves 84.0% success rate, with long-horizon tasks showing a +26 improvement over state-of-the-art baseline. Project Page: https://shihao1895.github.io/MemoryVLA
>
---
#### [new 003] ZeST: an LLM-based Zero-Shot Traversability Navigation for Unknown Environments
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对未知环境中的机器人导航任务，解决传统数据收集风险问题，通过LLM实现零样本实时可通行性地图生成，提升导航安全性与效率。**

- **链接: [http://arxiv.org/pdf/2508.19131v1](http://arxiv.org/pdf/2508.19131v1)**

> **作者:** Shreya Gummadi; Mateus V. Gasparino; Gianluca Capezzuto; Marcelo Becker; Girish Chowdhary
>
> **摘要:** The advancement of robotics and autonomous navigation systems hinges on the ability to accurately predict terrain traversability. Traditional methods for generating datasets to train these prediction models often involve putting robots into potentially hazardous environments, posing risks to equipment and safety. To solve this problem, we present ZeST, a novel approach leveraging visual reasoning capabilities of Large Language Models (LLMs) to create a traversability map in real-time without exposing robots to danger. Our approach not only performs zero-shot traversability and mitigates the risks associated with real-world data collection but also accelerates the development of advanced navigation systems, offering a cost-effective and scalable solution. To support our findings, we present navigation results, in both controlled indoor and unstructured outdoor environments. As shown in the experiments, our method provides safer navigation when compared to other state-of-the-art methods, constantly reaching the final goal.
>
---
#### [new 004] AutoRing: Imitation Learning--based Autonomous Intraocular Foreign Body Removal Manipulation with Eye Surgical Robot
- **分类: cs.RO**

- **简介: 论文提出AutoRing，通过模仿学习解决眼内异物移除中自主操作的运动学不确定性，结合动态RCM校准和RCM-ACT架构，仅用立体视觉数据训练，实现端到端自主操作。**

- **链接: [http://arxiv.org/pdf/2508.19191v1](http://arxiv.org/pdf/2508.19191v1)**

> **作者:** Yue Wang; Wenjie Deng; Haotian Xue; Di Cui; Yiqi Chen; Mingchuan Zhou; Haochao Ying; Jian Wu
>
> **摘要:** Intraocular foreign body removal demands millimeter-level precision in confined intraocular spaces, yet existing robotic systems predominantly rely on manual teleoperation with steep learning curves. To address the challenges of autonomous manipulation (particularly kinematic uncertainties from variable motion scaling and variation of the Remote Center of Motion (RCM) point), we propose AutoRing, an imitation learning framework for autonomous intraocular foreign body ring manipulation. Our approach integrates dynamic RCM calibration to resolve coordinate-system inconsistencies caused by intraocular instrument variation and introduces the RCM-ACT architecture, which combines action-chunking transformers with real-time kinematic realignment. Trained solely on stereo visual data and instrument kinematics from expert demonstrations in a biomimetic eye model, AutoRing successfully completes ring grasping and positioning tasks without explicit depth sensing. Experimental validation demonstrates end-to-end autonomy under uncalibrated microscopy conditions. The results provide a viable framework for developing intelligent eye-surgical systems capable of complex intraocular procedures.
>
---
#### [new 005] Mimicking associative learning of rats via a neuromorphic robot in open field maze using spatial cell models
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文通过神经形态机器人模仿老鼠关联学习，利用空间细胞模型在开放迷宫中实现实时空间任务学习，解决传统AI高能耗和低适应性问题，提升自主导航能力。**

- **链接: [http://arxiv.org/pdf/2508.18460v1](http://arxiv.org/pdf/2508.18460v1)**

> **作者:** Tianze Liu; Md Abu Bakr Siddique; Hongyu An
>
> **摘要:** Data-driven Artificial Intelligence (AI) approaches have exhibited remarkable prowess across various cognitive tasks using extensive training data. However, the reliance on large datasets and neural networks presents challenges such as highpower consumption and limited adaptability, particularly in SWaP-constrained applications like planetary exploration. To address these issues, we propose enhancing the autonomous capabilities of intelligent robots by emulating the associative learning observed in animals. Associative learning enables animals to adapt to their environment by memorizing concurrent events. By replicating this mechanism, neuromorphic robots can navigate dynamic environments autonomously, learning from interactions to optimize performance. This paper explores the emulation of associative learning in rodents using neuromorphic robots within open-field maze environments, leveraging insights from spatial cells such as place and grid cells. By integrating these models, we aim to enable online associative learning for spatial tasks in real-time scenarios, bridging the gap between biological spatial cognition and robotics for advancements in autonomous systems.
>
---
#### [new 006] An LLM-powered Natural-to-Robotic Language Translation Framework with Correctness Guarantees
- **分类: cs.RO; cs.AI; cs.PL**

- **简介: 该论文提出基于LLM的自然语言到机器人语言翻译框架，通过RSL语言抽象与反馈微调，解决LLM生成代码错误问题，确保程序正确性，提升机器人应用效果。**

- **链接: [http://arxiv.org/pdf/2508.19074v1](http://arxiv.org/pdf/2508.19074v1)**

> **作者:** ZhenDong Chen; ZhanShang Nie; ShiXing Wan; JunYi Li; YongTian Cheng; Shuai Zhao
>
> **摘要:** The Large Language Models (LLM) are increasingly being deployed in robotics to generate robot control programs for specific user tasks, enabling embodied intelligence. Existing methods primarily focus on LLM training and prompt design that utilize LLMs to generate executable programs directly from user tasks in natural language. However, due to the inconsistency of the LLMs and the high complexity of the tasks, such best-effort approaches often lead to tremendous programming errors in the generated code, which significantly undermines the effectiveness especially when the light-weight LLMs are applied. This paper introduces a natural-robotic language translation framework that (i) provides correctness verification for generated control programs and (ii) enhances the performance of LLMs in program generation via feedback-based fine-tuning for the programs. To achieve this, a Robot Skill Language (RSL) is proposed to abstract away from the intricate details of the control programs, bridging the natural language tasks with the underlying robot skills. Then, the RSL compiler and debugger are constructed to verify RSL programs generated by the LLM and provide error feedback to the LLM for refining the outputs until being verified by the compiler. This provides correctness guarantees for the LLM-generated programs before being offloaded to the robots for execution, significantly enhancing the effectiveness of LLM-powered robotic applications. Experiments demonstrate NRTrans outperforms the existing method under a range of LLMs and tasks, and achieves a high success rate for light-weight LLMs.
>
---
#### [new 007] HyperTASR: Hypernetwork-Driven Task-Aware Scene Representations for Robust Manipulation
- **分类: cs.RO**

- **简介: 论文提出HyperTASR，通过超网络动态调整任务感知场景表示，解决机器人操作中静态表示不足的问题，提升任务适应性和表现。**

- **链接: [http://arxiv.org/pdf/2508.18802v1](http://arxiv.org/pdf/2508.18802v1)**

> **作者:** Li Sun; Jiefeng Wu; Feng Chen; Ruizhe Liu; Yanchao Yang
>
> **摘要:** Effective policy learning for robotic manipulation requires scene representations that selectively capture task-relevant environmental features. Current approaches typically employ task-agnostic representation extraction, failing to emulate the dynamic perceptual adaptation observed in human cognition. We present HyperTASR, a hypernetwork-driven framework that modulates scene representations based on both task objectives and the execution phase. Our architecture dynamically generates representation transformation parameters conditioned on task specifications and progression state, enabling representations to evolve contextually throughout task execution. This approach maintains architectural compatibility with existing policy learning frameworks while fundamentally reconfiguring how visual features are processed. Unlike methods that simply concatenate or fuse task embeddings with task-agnostic representations, HyperTASR establishes computational separation between task-contextual and state-dependent processing paths, enhancing learning efficiency and representational quality. Comprehensive evaluations in both simulation and real-world environments demonstrate substantial performance improvements across different representation paradigms. Through ablation studies and attention visualization, we confirm that our approach selectively prioritizes task-relevant scene information, closely mirroring human adaptive perception during manipulation tasks. The project website is at \href{https://lisunphil.github.io/HyperTASR_projectpage/}{lisunphil.github.io/HyperTASR\_projectpage}.
>
---
#### [new 008] Planning-Query-Guided Model Generation for Model-Based Deformable Object Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对变形物体操作中的高效规划问题，提出基于规划查询的动态模型生成方法，通过扩散模型预测区域分辨率并两阶段优化，实现高速规划与性能平衡。**

- **链接: [http://arxiv.org/pdf/2508.19199v1](http://arxiv.org/pdf/2508.19199v1)**

> **作者:** Alex LaGrassa; Zixuan Huang; Dmitry Berenson; Oliver Kroemer
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Efficient planning in high-dimensional spaces, such as those involving deformable objects, requires computationally tractable yet sufficiently expressive dynamics models. This paper introduces a method that automatically generates task-specific, spatially adaptive dynamics models by learning which regions of the object require high-resolution modeling to achieve good task performance for a given planning query. Task performance depends on the complex interplay between the dynamics model, world dynamics, control, and task requirements. Our proposed diffusion-based model generator predicts per-region model resolutions based on start and goal pointclouds that define the planning query. To efficiently collect the data for learning this mapping, a two-stage process optimizes resolution using predictive dynamics as a prior before directly optimizing using closed-loop performance. On a tree-manipulation task, our method doubles planning speed with only a small decrease in task performance over using a full-resolution model. This approach informs a path towards using previous planning and control data to generate computationally efficient yet sufficiently expressive dynamics models for new tasks.
>
---
#### [new 009] VisionSafeEnhanced VPC: Cautious Predictive Control with Visibility Constraints under Uncertainty for Autonomous Robotic Surgery
- **分类: cs.RO**

- **简介: 该论文针对自主机器人手术中腹腔镜控制的不确定性与视野安全问题，提出VisionSafeEnhanced VPC框架，通过GPR量化不确定性并设计概率约束优化轨迹，确保操作安全与高 visibility。**

- **链接: [http://arxiv.org/pdf/2508.18937v1](http://arxiv.org/pdf/2508.18937v1)**

> **作者:** Wang Jiayin; Wei Yanran; Jiang Lei; Guo Xiaoyu; Zheng Ayong; Zhao Weidong; Li Zhongkui
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Autonomous control of the laparoscope in robot-assisted Minimally Invasive Surgery (MIS) has received considerable research interest due to its potential to improve surgical safety. Despite progress in pixel-level Image-Based Visual Servoing (IBVS) control, the requirement of continuous visibility and the existence of complex disturbances, such as parameterization error, measurement noise, and uncertainties of payloads, could degrade the surgeon's visual experience and compromise procedural safety. To address these limitations, this paper proposes VisionSafeEnhanced Visual Predictive Control (VPC), a robust and uncertainty-adaptive framework for autonomous laparoscope control that guarantees Field of View (FoV) safety under uncertainty. Firstly, Gaussian Process Regression (GPR) is utilized to perform hybrid (deterministic + stochastic) quantification of operational uncertainties including residual model uncertainties, stochastic uncertainties, and external disturbances. Based on uncertainty quantification, a novel safety aware trajectory optimization framework with probabilistic guarantees is proposed, where a uncertainty-adaptive safety Control Barrier Function (CBF) condition is given based on uncertainty propagation, and chance constraints are simultaneously formulated based on probabilistic approximation. This uncertainty aware formulation enables adaptive control effort allocation, minimizing unnecessary camera motion while maintaining robustness. The proposed method is validated through comparative simulations and experiments on a commercial surgical robot platform (MicroPort MedBot Toumai) performing a sequential multi-target lymph node dissection. Compared with baseline methods, the framework maintains near-perfect target visibility (>99.9%), reduces tracking e
>
---
#### [new 010] Uncertainty-Resilient Active Intention Recognition for Robotic Assistants
- **分类: cs.RO; cs.AI**

- **简介: 论文针对机器人助手在不确定环境下的意图识别问题，提出基于POMDP的框架，整合传感器数据与规划器，提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.19150v1](http://arxiv.org/pdf/2508.19150v1)**

> **作者:** Juan Carlos Saborío; Marc Vinci; Oscar Lima; Sebastian Stock; Lennart Niecksch; Martin Günther; Alexander Sung; Joachim Hertzberg; Martin Atzmüller
>
> **备注:** (To appear) In Proceedings of ECMR 2025
>
> **摘要:** Purposeful behavior in robotic assistants requires the integration of multiple components and technological advances. Often, the problem is reduced to recognizing explicit prompts, which limits autonomy, or is oversimplified through assumptions such as near-perfect information. We argue that a critical gap remains unaddressed -- specifically, the challenge of reasoning about the uncertain outcomes and perception errors inherent to human intention recognition. In response, we present a framework designed to be resilient to uncertainty and sensor noise, integrating real-time sensor data with a combination of planners. Centered around an intention-recognition POMDP, our approach addresses cooperative planning and acting under uncertainty. Our integrated framework has been successfully tested on a physical robot with promising results.
>
---
#### [new 011] Learning Real-World Acrobatic Flight from Human Preferences
- **分类: cs.RO; cs.LG**

- **简介: 该论文通过偏好强化学习（PbRL）解决无人机空翻飞行控制问题，提出REC方法改进偏好建模与稳定性，在模拟与真实场景中实现高精度动态 maneuvers，验证了 PbRL 在捕捉人类偏好目标的有效性。**

- **链接: [http://arxiv.org/pdf/2508.18817v1](http://arxiv.org/pdf/2508.18817v1)**

> **作者:** Colin Merk; Ismail Geles; Jiaxu Xing; Angel Romero; Giorgia Ramponi; Davide Scaramuzza
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Preference-based reinforcement learning (PbRL) enables agents to learn control policies without requiring manually designed reward functions, making it well-suited for tasks where objectives are difficult to formalize or inherently subjective. Acrobatic flight poses a particularly challenging problem due to its complex dynamics, rapid movements, and the importance of precise execution. In this work, we explore the use of PbRL for agile drone control, focusing on the execution of dynamic maneuvers such as powerloops. Building on Preference-based Proximal Policy Optimization (Preference PPO), we propose Reward Ensemble under Confidence (REC), an extension to the reward learning objective that improves preference modeling and learning stability. Our method achieves 88.4% of the shaped reward performance, compared to 55.2% with standard Preference PPO. We train policies in simulation and successfully transfer them to real-world drones, demonstrating multiple acrobatic maneuvers where human preferences emphasize stylistic qualities of motion. Furthermore, we demonstrate the applicability of our probabilistic reward model in a representative MuJoCo environment for continuous control. Finally, we highlight the limitations of manually designed rewards, observing only 60.7% agreement with human preferences. These results underscore the effectiveness of PbRL in capturing complex, human-centered objectives across both physical and simulated domains.
>
---
#### [new 012] AS2FM: Enabling Statistical Model Checking of ROS 2 Systems for Robust Autonomy
- **分类: cs.RO; cs.FL**

- **简介: 该论文通过AS2FM工具将ROS 2系统转换为JANI格式，实现统计模型检查，解决自主机器人系统设计时的验证难题，提升可靠性。**

- **链接: [http://arxiv.org/pdf/2508.18820v1](http://arxiv.org/pdf/2508.18820v1)**

> **作者:** Christian Henkel; Marco Lampacrescia; Michaela Klauck; Matteo Morelli
>
> **备注:** Accepted at IROS2025
>
> **摘要:** Designing robotic systems to act autonomously in unforeseen environments is a challenging task. This work presents a novel approach to use formal verification, specifically Statistical Model Checking (SMC), to verify system properties of autonomous robots at design-time. We introduce an extension of the SCXML format, designed to model system components including both Robot Operating System 2 (ROS 2) and Behavior Tree (BT) features. Further, we contribute Autonomous Systems to Formal Models (AS2FM), a tool to translate the full system model into JANI. The use of JANI, a standard format for quantitative model checking, enables verification of system properties with off-the-shelf SMC tools. We demonstrate the practical usability of AS2FM both in terms of applicability to real-world autonomous robotic control systems, and in terms of verification runtime scaling. We provide a case study, where we successfully identify problems in a ROS 2-based robotic manipulation use case that is verifiable in less than one second using consumer hardware. Additionally, we compare to the state of the art and demonstrate that our method is more comprehensive in system feature support, and that the verification runtime scales linearly with the size of the model, instead of exponentially.
>
---
#### [new 013] DELIVER: A System for LLM-Guided Coordinated Multi-Robot Pickup and Delivery using Voronoi-Based Relay Planning
- **分类: cs.RO; cs.MA**

- **简介: 论文提出DELIVER系统，用于LLM指导的多机器人协作配送任务，通过Voronoi分区和中继规划实现自然语言指令下的高效协调，验证于仿真与真实硬件，展示高效率及可扩展性。**

- **链接: [http://arxiv.org/pdf/2508.19114v1](http://arxiv.org/pdf/2508.19114v1)**

> **作者:** Alkesh K. Srivastava; Jared Michael Levin; Alexander Derrico; Philip Dames
>
> **备注:** Submission under review at the 2026 IEEE/SICE International Symposium on System Integration (SII 2026)
>
> **摘要:** We present DELIVER (Directed Execution of Language-instructed Item Via Engineered Relay), a fully integrated framework for cooperative multi-robot pickup and delivery driven by natural language commands. DELIVER unifies natural language understanding, spatial decomposition, relay planning, and motion execution to enable scalable, collision-free coordination in real-world settings. Given a spoken or written instruction, a lightweight instance of LLaMA3 interprets the command to extract pickup and delivery locations. The environment is partitioned using a Voronoi tessellation to define robot-specific operating regions. Robots then compute optimal relay points along shared boundaries and coordinate handoffs. A finite-state machine governs each robot's behavior, enabling robust execution. We implement DELIVER on the MultiTRAIL simulation platform and validate it in both ROS2-based Gazebo simulations and real-world hardware using TurtleBot3 robots. Empirical results show that DELIVER maintains consistent mission cost across varying team sizes while reducing per-agent workload by up to 55% compared to a single-agent system. Moreover, the number of active relay agents remains low even as team size increases, demonstrating the system's scalability and efficient agent utilization. These findings underscore DELIVER's modular and extensible architecture for language-guided multi-robot coordination, advancing the frontiers of cyber-physical system integration.
>
---
#### [new 014] Efficient task and path planning for maintenance automation using a robot system
- **分类: cs.RO**

- **简介: 该论文针对机器人维护自动化中的任务与路径规划问题，提出结合CAD数据与实时视觉信息的不确定性补偿方法，采用符号描述和改进采样算法计算拆卸空间，并优化全局路径规划策略，通过实验验证提升效率。**

- **链接: [http://arxiv.org/pdf/2508.18400v1](http://arxiv.org/pdf/2508.18400v1)**

> **作者:** Christian Friedrich; Akos Csiszar; Armin Lechler; Alexander Verl
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** The research and development of intelligent automation solutions is a ground-breaking point for the factory of the future. A promising and challenging mission is the use of autonomous robot systems to automate tasks in the field of maintenance. For this purpose, the robot system must be able to plan autonomously the different manipulation tasks and the corresponding paths. Basic requirements are the development of algorithms with a low computational complexity and the possibility to deal with environmental uncertainties. In this work, an approach is presented, which is especially suited to solve the problem of maintenance automation. For this purpose, offline data from CAD is combined with online data from an RGBD vision system via a probabilistic filter, to compensate uncertainties from offline data. For planning the different tasks, a method is explained, which use a symbolic description, founded on a novel sampling-based method to compute the disassembly space. For path planning we use global state-of-the art algorithms with a method that allows the adaption of the exploration stepsize in order to reduce the planning time. Every method is experimentally validated and discussed.
>
---
#### [new 015] Integration of Robot and Scene Kinematics for Sequential Mobile Manipulation Planning
- **分类: cs.RO**

- **简介: 该论文提出一种序列移动操作规划框架，通过整合机器人与场景运动学构建A-Space，解决长期多步骤移动操作中的协调控制问题，采用三级框架实现高效规划。**

- **链接: [http://arxiv.org/pdf/2508.18627v1](http://arxiv.org/pdf/2508.18627v1)**

> **作者:** Ziyuan Jiao; Yida Niu; Zeyu Zhang; Yangyang Wu; Yao Su; Yixin Zhu; Hangxin Liu; Song-Chun Zhu
>
> **备注:** 20 pages, 13 figures; accepted by Transactions on Robotics
>
> **摘要:** We present a Sequential Mobile Manipulation Planning (SMMP) framework that can solve long-horizon multi-step mobile manipulation tasks with coordinated whole-body motion, even when interacting with articulated objects. By abstracting environmental structures as kinematic models and integrating them with the robot's kinematics, we construct an Augmented Configuration Apace (A-Space) that unifies the previously separate task constraints for navigation and manipulation, while accounting for the joint reachability of the robot base, arm, and manipulated objects. This integration facilitates efficient planning within a tri-level framework: a task planner generates symbolic action sequences to model the evolution of A-Space, an optimization-based motion planner computes continuous trajectories within A-Space to achieve desired configurations for both the robot and scene elements, and an intermediate plan refinement stage selects action goals that ensure long-horizon feasibility. Our simulation studies first confirm that planning in A-Space achieves an 84.6\% higher task success rate compared to baseline methods. Validation on real robotic systems demonstrates fluid mobile manipulation involving (i) seven types of rigid and articulated objects across 17 distinct contexts, and (ii) long-horizon tasks of up to 14 sequential steps. Our results highlight the significance of modeling scene kinematics into planning entities, rather than encoding task-specific constraints, offering a scalable and generalizable approach to complex robotic manipulation.
>
---
#### [new 016] PneuGelSight: Soft Robotic Vision-Based Proprioception and Tactile Sensing
- **分类: cs.RO**

- **简介: 论文提出基于视觉的传感方法，用于软机器人触觉与本体感觉，通过模拟到现实的流程提升性能，解决实际应用中的传感不足问题。**

- **链接: [http://arxiv.org/pdf/2508.18443v1](http://arxiv.org/pdf/2508.18443v1)**

> **作者:** Ruohan Zhang; Uksang Yoo; Yichen Li; Arpit Argawal; Wenzhen Yuan
>
> **备注:** 16 pages, 12 figures, International Journal of Robotics Research (accepted), 2025
>
> **摘要:** Soft pneumatic robot manipulators are popular in industrial and human-interactive applications due to their compliance and flexibility. However, deploying them in real-world scenarios requires advanced sensing for tactile feedback and proprioception. Our work presents a novel vision-based approach for sensorizing soft robots. We demonstrate our approach on PneuGelSight, a pioneering pneumatic manipulator featuring high-resolution proprioception and tactile sensing via an embedded camera. To optimize the sensor's performance, we introduce a comprehensive pipeline that accurately simulates its optical and dynamic properties, facilitating a zero-shot knowledge transition from simulation to real-world applications. PneuGelSight and our sim-to-real pipeline provide a novel, easily implementable, and robust sensing methodology for soft robots, paving the way for the development of more advanced soft robots with enhanced sensory capabilities.
>
---
#### [new 017] Enhanced UAV Path Planning Using the Tangent Intersection Guidance (TIG) Algorithm
- **分类: cs.RO; cs.CV**

- **简介: 论文提出TIG算法用于UAV路径规划，解决静态/动态环境中的高效安全导航问题，通过椭圆切线交点生成子路径并优化，结合贝塞尔曲线平滑，实验证明优于其他算法。**

- **链接: [http://arxiv.org/pdf/2508.18967v1](http://arxiv.org/pdf/2508.18967v1)**

> **作者:** Hichem Cheriet; Khellat Kihel Badra; Chouraqui Samira
>
> **备注:** Accepted for publication in JAMRIS Journal
>
> **摘要:** Efficient and safe navigation of Unmanned Aerial Vehicles (UAVs) is critical for various applications, including combat support, package delivery and Search and Rescue Operations. This paper introduces the Tangent Intersection Guidance (TIG) algorithm, an advanced approach for UAV path planning in both static and dynamic environments. The algorithm uses the elliptic tangent intersection method to generate feasible paths. It generates two sub-paths for each threat, selects the optimal route based on a heuristic rule, and iteratively refines the path until the target is reached. Considering the UAV kinematic and dynamic constraints, a modified smoothing technique based on quadratic B\'ezier curves is adopted to generate a smooth and efficient route. Experimental results show that the TIG algorithm can generate the shortest path in less time, starting from 0.01 seconds, with fewer turning angles compared to A*, PRM, RRT*, Tangent Graph, and Static APPATT algorithms in static environments. Furthermore, in completely unknown and partially known environments, TIG demonstrates efficient real-time path planning capabilities for collision avoidance, outperforming APF and Dynamic APPATT algorithms.
>
---
#### [new 018] Real-time Testing of Satellite Attitude Control With a Reaction Wheel Hardware-In-the-Loop Platform
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出基于硬件在环平台的卫星姿态控制系统实时测试方法，解决控制器在真实设备中的有效性验证问题，构建包含反应轮和模拟器的测试平台，并引入故障注入机制。**

- **链接: [http://arxiv.org/pdf/2508.19164v1](http://arxiv.org/pdf/2508.19164v1)**

> **作者:** Morokot Sakal; George Nehma; Camilo Riano-Rios; Madhur Tiwari
>
> **备注:** 15 pages, 10 figures, 2025 AAS/AIAA Astrodynamics Specialist Conference
>
> **摘要:** We propose the Hardware-in-the-Loop (HIL) test of an adaptive satellite attitude control system with reaction wheel health estimation capabilities. Previous simulations and Software-in-the-Loop testing have prompted further experiments to explore the validity of the controller with real momentum exchange devices in the loop. This work is a step toward a comprehensive testing framework for validation of spacecraft attitude control algorithms. The proposed HIL testbed includes brushless DC motors and drivers that communicate using a CAN bus, an embedded computer that executes control and adaptation laws, and a satellite simulator that produces simulated sensor data, estimated attitude states, and responds to actions of the external actuators. We propose methods to artificially induce failures on the reaction wheels, and present related issues and lessons learned.
>
---
#### [new 019] Engineering Automotive Digital Twins on Standardized Architectures: A Case Study
- **分类: cs.RO**

- **简介: 论文评估ISO 23247架构在汽车数字孪生中的适用性，通过开发自适应巡航控制DT案例，分析其优缺点并提出未来研究方向，解决汽车DT架构标准化不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.18662v1](http://arxiv.org/pdf/2508.18662v1)**

> **作者:** Stefan Ramdhan; Winnie Trandinh; Istvan David; Vera Pantelic; Mark Lawford
>
> **备注:** 7 pages, 6 figures. Submitted to EDTconf 2025
>
> **摘要:** Digital twin (DT) technology has become of interest in the automotive industry. There is a growing need for smarter services that utilize the unique capabilities of DTs, ranging from computer-aided remote control to cloud-based fleet coordination. Developing such services starts with the software architecture. However, the scarcity of DT architectural guidelines poses a challenge for engineering automotive DTs. Currently, the only DT architectural standard is the one defined in ISO 23247. Though not developed for automotive systems, it is one of the few feasible starting points for automotive DTs. In this work, we investigate the suitability of the ISO 23247 reference architecture for developing automotive DTs. Through the case study of developing an Adaptive Cruise Control DT for a 1/10\textsuperscript{th}-scale autonomous vehicle, we identify some strengths and limitations of the reference architecture and begin distilling future directions for researchers, practitioners, and standard developers.
>
---
#### [new 020] Mining the Long Tail: A Comparative Study of Data-Centric Criticality Metrics for Robust Offline Reinforcement Learning in Autonomous Motion Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对Offline RL中数据不平衡导致的自主运动规划策略不安全问题，提出并比较六种数据筛选策略，发现基于模型不确定性的方法显著提升安全性，减少碰撞率。**

- **链接: [http://arxiv.org/pdf/2508.18397v1](http://arxiv.org/pdf/2508.18397v1)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** Offline Reinforcement Learning (RL) presents a promising paradigm for training autonomous vehicle (AV) planning policies from large-scale, real-world driving logs. However, the extreme data imbalance in these logs, where mundane scenarios vastly outnumber rare "long-tail" events, leads to brittle and unsafe policies when using standard uniform data sampling. In this work, we address this challenge through a systematic, large-scale comparative study of data curation strategies designed to focus the learning process on information-rich samples. We investigate six distinct criticality weighting schemes which are categorized into three families: heuristic-based, uncertainty-based, and behavior-based. These are evaluated at two temporal scales, the individual timestep and the complete scenario. We train seven goal-conditioned Conservative Q-Learning (CQL) agents with a state-of-the-art, attention-based architecture and evaluate them in the high-fidelity Waymax simulator. Our results demonstrate that all data curation methods significantly outperform the baseline. Notably, data-driven curation using model uncertainty as a signal achieves the most significant safety improvements, reducing the collision rate by nearly three-fold (from 16.0% to 5.5%). Furthermore, we identify a clear trade-off where timestep-level weighting excels at reactive safety while scenario-level weighting improves long-horizon planning. Our work provides a comprehensive framework for data curation in Offline RL and underscores that intelligent, non-uniform sampling is a critical component for building safe and reliable autonomous agents.
>
---
#### [new 021] QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文研究四足机器人视觉引导的运动控制，解决复杂环境中的鲁棒控制问题，提出QuadKAN框架，结合KAN和样条参数化策略，通过PPO训练，提升效率和可解释性。**

- **链接: [http://arxiv.org/pdf/2508.19153v1](http://arxiv.org/pdf/2508.19153v1)**

> **作者:** Allen Wang; Gavin Tao
>
> **备注:** 14pages, 9 figures, Journal paper
>
> **摘要:** We address vision-guided quadruped motion control with reinforcement learning (RL) and highlight the necessity of combining proprioception with vision for robust control. We propose QuadKAN, a spline-parameterized cross-modal policy instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates a spline encoder for proprioception and a spline fusion head for proprioception-vision inputs. This structured function class aligns the state-to-action mapping with the piecewise-smooth nature of gait, improving sample efficiency, reducing action jitter and energy consumption, and providing interpretable posture-action sensitivities. We adopt Multi-Modal Delay Randomization (MMDR) and perform end-to-end training with Proximal Policy Optimization (PPO). Evaluations across diverse terrains, including both even and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate that QuadKAN achieves consistently higher returns, greater distances, and fewer collisions than state-of-the-art (SOTA) baselines. These results show that spline-parameterized policies offer a simple, effective, and interpretable alternative for robust vision-guided locomotion. A repository will be made available upon acceptance.
>
---
#### [new 022] Maintenance automation: methods for robotics manipulation planning and execution
- **分类: cs.RO**

- **简介: 论文针对机器人维护自动化任务，解决环境不确定性下的规划与执行问题，提出基于CAD/RGBD数据的系统，将符号计划转换为可执行指令并经实验验证。**

- **链接: [http://arxiv.org/pdf/2508.18399v1](http://arxiv.org/pdf/2508.18399v1)**

> **作者:** Christian Friedrich; Ralf Gulde; Armin Lechler; Alexander Verl
>
> **备注:** 11 pages, 12 figures
>
> **摘要:** Automating complex tasks using robotic systems requires skills for planning, control and execution. This paper proposes a complete robotic system for maintenance automation, which can automate disassembly and assembly operations under environmental uncertainties (e.g. deviations between prior plan information). The cognition of the robotic system is based on a planning approach (using CAD and RGBD data) and includes a method to interpret a symbolic plan and transform it to a set of executable robot instructions. The complete system is experimentally evaluated using real-world applications. This work shows the first step to transfer these theoretical results into a practical robotic solution.
>
---
#### [new 023] Deep Sensorimotor Control by Imitating Predictive Models of Human Motion
- **分类: cs.RO**

- **简介: 论文针对机器人传感器运动控制任务，解决如何有效利用人类数据训练策略的问题，提出模仿人类运动预测模型的方法，实现零样本迁移，无需传统方法中的梯度退化和对抗损失，提升跨机器人和任务的性能。**

- **链接: [http://arxiv.org/pdf/2508.18691v1](http://arxiv.org/pdf/2508.18691v1)**

> **作者:** Himanshu Gaurav Singh; Pieter Abbeel; Jitendra Malik; Antonio Loquercio
>
> **备注:** Blog Post: https://hgaurav2k.github.io/trackr/
>
> **摘要:** As the embodiment gap between a robot and a human narrows, new opportunities arise to leverage datasets of humans interacting with their surroundings for robot learning. We propose a novel technique for training sensorimotor policies with reinforcement learning by imitating predictive models of human motions. Our key insight is that the motion of keypoints on human-inspired robot end-effectors closely mirrors the motion of corresponding human body keypoints. This enables us to use a model trained to predict future motion on human data \emph{zero-shot} on robot data. We train sensorimotor policies to track the predictions of such a model, conditioned on a history of past robot states, while optimizing a relatively sparse task reward. This approach entirely bypasses gradient-based kinematic retargeting and adversarial losses, which limit existing methods from fully leveraging the scale and diversity of modern human-scene interaction datasets. Empirically, we find that our approach can work across robots and tasks, outperforming existing baselines by a large margin. In addition, we find that tracking a human motion model can substitute for carefully designed dense rewards and curricula in manipulation tasks. Code, data and qualitative results available at https://jirl-upenn.github.io/track_reward/.
>
---
#### [new 024] SignLoc: Robust Localization using Navigation Signs and Public Maps
- **分类: cs.RO**

- **简介: 该论文提出SignLoc，用于机器人全局定位，通过导航标志和公共地图（如平面图、OSM）实现无需先验地图的鲁棒定位，采用概率模型与蒙特卡洛框架匹配标志信息，在多场景中验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.18606v1](http://arxiv.org/pdf/2508.18606v1)**

> **作者:** Nicky Zimmerman; Joel Loo; Ayush Agrawal; David Hsu
>
> **备注:** Under submission for Robotics and Automation Letters (RA-L)
>
> **摘要:** Navigation signs and maps, such as floor plans and street maps, are widely available and serve as ubiquitous aids for way-finding in human environments. Yet, they are rarely used by robot systems. This paper presents SignLoc, a global localization method that leverages navigation signs to localize the robot on publicly available maps -- specifically floor plans and OpenStreetMap (OSM) graphs--without prior sensor-based mapping. SignLoc first extracts a navigation graph from the input map. It then employs a probabilistic observation model to match directional and locational cues from the detected signs to the graph, enabling robust topo-semantic localization within a Monte Carlo framework. We evaluated SignLoc in diverse large-scale environments: part of a university campus, a shopping mall, and a hospital complex. Experimental results show that SignLoc reliably localizes the robot after observing only one to two signs.
>
---
#### [new 025] HuBE: Cross-Embodiment Human-like Behavior Execution for Humanoid Robots
- **分类: cs.RO**

- **简介: 论文提出HuBE框架，解决人形机器人生成人类样态运动时跨躯体适应与行为适当性问题，通过整合状态、目标姿势及情境，构建HPose数据集并采用骨标度增强，提升运动相似性与效率。**

- **链接: [http://arxiv.org/pdf/2508.19002v1](http://arxiv.org/pdf/2508.19002v1)**

> **作者:** Shipeng Lyu; Fangyuan Wang; Weiwei Lin; Luhao Zhu; David Navarro-Alarcon; Guodong Guo
>
> **备注:** 8 pages, 8 figures,4 tables
>
> **摘要:** Achieving both behavioral similarity and appropriateness in human-like motion generation for humanoid robot remains an open challenge, further compounded by the lack of cross-embodiment adaptability. To address this problem, we propose HuBE, a bi-level closed-loop framework that integrates robot state, goal poses, and contextual situations to generate human-like behaviors, ensuring both behavioral similarity and appropriateness, and eliminating structural mismatches between motion generation and execution. To support this framework, we construct HPose, a context-enriched dataset featuring fine-grained situational annotations. Furthermore, we introduce a bone scaling-based data augmentation strategy that ensures millimeter-level compatibility across heterogeneous humanoid robots. Comprehensive evaluations on multiple commercial platforms demonstrate that HuBE significantly improves motion similarity, behavioral appropriateness, and computational efficiency over state-of-the-art baselines, establishing a solid foundation for transferable and human-like behavior execution across diverse humanoid robots.
>
---
#### [new 026] Direction Informed Trees (DIT*): Optimal Path Planning via Direction Filter and Direction Cost Heuristic
- **分类: cs.RO**

- **简介: 论文针对最优路径规划中启发式冲突问题，提出DIT*算法，通过方向滤波器与成本启发式优化搜索方向，提升收敛效率。**

- **链接: [http://arxiv.org/pdf/2508.19168v1](http://arxiv.org/pdf/2508.19168v1)**

> **作者:** Liding Zhang; Kejia Chen; Kuanqi Cai; Yu Zhang; Yixuan Dang; Yansong Wu; Zhenshan Bing; Fan Wu; Sami Haddadin; Alois Knoll
>
> **备注:** 7 pages, 5 figures, 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Optimal path planning requires finding a series of feasible states from the starting point to the goal to optimize objectives. Popular path planning algorithms, such as Effort Informed Trees (EIT*), employ effort heuristics to guide the search. Effective heuristics are accurate and computationally efficient, but achieving both can be challenging due to their conflicting nature. This paper proposes Direction Informed Trees (DIT*), a sampling-based planner that focuses on optimizing the search direction for each edge, resulting in goal bias during exploration. We define edges as generalized vectors and integrate similarity indexes to establish a directional filter that selects the nearest neighbors and estimates direction costs. The estimated direction cost heuristics are utilized in edge evaluation. This strategy allows the exploration to share directional information efficiently. DIT* convergence faster than existing single-query, sampling-based planners on tested problems in R^4 to R^16 and has been demonstrated in real-world environments with various planning tasks. A video showcasing our experimental results is available at: https://youtu.be/2SX6QT2NOek
>
---
#### [new 027] From Tabula Rasa to Emergent Abilities: Discovering Robot Skills via Real-World Unsupervised Quality-Diversity
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出URSA框架，使机器人在真实世界中无需监督地发现多样化高绩效技能，提升适应能力。**

- **链接: [http://arxiv.org/pdf/2508.19172v1](http://arxiv.org/pdf/2508.19172v1)**

> **作者:** Luca Grillotti; Lisa Coiffard; Oscar Pang; Maxence Faldor; Antoine Cully
>
> **备注:** Accepted at CoRL 2025
>
> **摘要:** Autonomous skill discovery aims to enable robots to acquire diverse behaviors without explicit supervision. Learning such behaviors directly on physical hardware remains challenging due to safety and data efficiency constraints. Existing methods, including Quality-Diversity Actor-Critic (QDAC), require manually defined skill spaces and carefully tuned heuristics, limiting real-world applicability. We propose Unsupervised Real-world Skill Acquisition (URSA), an extension of QDAC that enables robots to autonomously discover and master diverse, high-performing skills directly in the real world. We demonstrate that URSA successfully discovers diverse locomotion skills on a Unitree A1 quadruped in both simulation and the real world. Our approach supports both heuristic-driven skill discovery and fully unsupervised settings. We also show that the learned skill repertoire can be reused for downstream tasks such as real-world damage adaptation, where URSA outperforms all baselines in 5 out of 9 simulated and 3 out of 5 real-world damage scenarios. Our results establish a new framework for real-world robot learning that enables continuous skill discovery with limited human intervention, representing a significant step toward more autonomous and adaptable robotic systems. Demonstration videos are available at http://adaptive-intelligent-robotics.github.io/URSA .
>
---
#### [new 028] Real-Time Model Checking for Closed-Loop Robot Reactive Planning
- **分类: cs.RO; cs.AI; cs.FL; I.2.9; I.2; D.2.4**

- **简介: 该论文提出实时模型检查用于机器人反应式多步规划，解决传统单步规划效率低的问题，通过离散化LiDAR数据和临时控制系统的链式结构，实现实时避障。**

- **链接: [http://arxiv.org/pdf/2508.19186v1](http://arxiv.org/pdf/2508.19186v1)**

> **作者:** Christopher Chandler; Bernd Porr; Giulia Lafratta; Alice Miller
>
> **备注:** 30 pages excluding references, 18 figures, submitted to Formal Aspects of Computing
>
> **摘要:** We present a new application of model checking which achieves real-time multi-step planning and obstacle avoidance on a real autonomous robot. We have developed a small, purpose-built model checking algorithm which generates plans in situ based on "core" knowledge and attention as found in biological agents. This is achieved in real-time using no pre-computed data on a low-powered device. Our approach is based on chaining temporary control systems which are spawned to counteract disturbances in the local environment that disrupt an autonomous agent from its preferred action (or resting state). A novel discretization of 2D LiDAR data sensitive to bounded variations in the local environment is used. Multi-step planning using model checking by forward depth-first search is applied to cul-de-sac and playground scenarios. Both empirical results and informal proofs of two fundamental properties of our approach demonstrate that model checking can be used to create efficient multi-step plans for local obstacle avoidance, improving on the performance of a reactive agent which can only plan one step. Our approach is an instructional case study for the development of safe, reliable and explainable planning in the context of autonomous vehicles.
>
---
#### [new 029] Enhancing Video-Based Robot Failure Detection Using Task Knowledge
- **分类: cs.RO; cs.CV**

- **简介: 论文提出基于视频的机器人故障检测方法，利用时空任务知识与数据增强，提升故障识别性能。**

- **链接: [http://arxiv.org/pdf/2508.18705v1](http://arxiv.org/pdf/2508.18705v1)**

> **作者:** Santosh Thoduka; Sebastian Houben; Juergen Gall; Paul G. Plöger
>
> **备注:** Accepted at ECMR 2025
>
> **摘要:** Robust robotic task execution hinges on the reliable detection of execution failures in order to trigger safe operation modes, recovery strategies, or task replanning. However, many failure detection methods struggle to provide meaningful performance when applied to a variety of real-world scenarios. In this paper, we propose a video-based failure detection approach that uses spatio-temporal knowledge in the form of the actions the robot performs and task-relevant objects within the field of view. Both pieces of information are available in most robotic scenarios and can thus be readily obtained. We demonstrate the effectiveness of our approach on three datasets that we amend, in part, with additional annotations of the aforementioned task-relevant knowledge. In light of the results, we also propose a data augmentation method that improves performance by applying variable frame rates to different parts of the video. We observe an improvement from 77.9 to 80.0 in F1 score on the ARMBench dataset without additional computational expense and an additional increase to 81.4 with test-time augmentation. The results emphasize the importance of spatio-temporal information during failure detection and suggest further investigation of suitable heuristics in future implementations. Code and annotations are available.
>
---
#### [new 030] Are All Marine Species Created Equal? Performance Disparities in Underwater Object Detection
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 论文研究水下目标检测任务，解决不同海洋物种检测性能差异原因及改进方法。通过分析发现定位阶段的前景-背景区分是关键问题，提出根据精度/召回率调整数据分布，并优化定位模块算法以提升低性能物种检测。**

- **链接: [http://arxiv.org/pdf/2508.18729v1](http://arxiv.org/pdf/2508.18729v1)**

> **作者:** Melanie Wille; Tobias Fischer; Scarlett Raine
>
> **备注:** 10 pages
>
> **摘要:** Underwater object detection is critical for monitoring marine ecosystems but poses unique challenges, including degraded image quality, imbalanced class distribution, and distinct visual characteristics. Not every species is detected equally well, yet underlying causes remain unclear. We address two key research questions: 1) What factors beyond data quantity drive class-specific performance disparities? 2) How can we systematically improve detection of under-performing marine species? We manipulate the DUO dataset to separate the object detection task into localization and classification and investigate the under-performance of the scallop class. Localization analysis using YOLO11 and TIDE finds that foreground-background discrimination is the most problematic stage regardless of data quantity. Classification experiments reveal persistent precision gaps even with balanced data, indicating intrinsic feature-based challenges beyond data scarcity and inter-class dependencies. We recommend imbalanced distributions when prioritizing precision, and balanced distributions when prioritizing recall. Improving under-performing classes should focus on algorithmic advances, especially within localization modules. We publicly release our code and datasets.
>
---
#### [new 031] VibES: Induced Vibration for Persistent Event-Based Sensing
- **分类: cs.CV; cs.RO**

- **简介: 该论文旨在解决静态场景下事件相机无事件输出的问题，通过诱导振动与运动补偿技术实现持续事件生成，提升图像重建与边缘检测性能。**

- **链接: [http://arxiv.org/pdf/2508.19094v1](http://arxiv.org/pdf/2508.19094v1)**

> **作者:** Vincenzo Polizzi; Stephen Yang; Quentin Clark; Jonathan Kelly; Igor Gilitschenski; David B. Lindell
>
> **摘要:** Event cameras are a bio-inspired class of sensors that asynchronously measure per-pixel intensity changes. Under fixed illumination conditions in static or low-motion scenes, rigidly mounted event cameras are unable to generate any events, becoming unsuitable for most computer vision tasks. To address this limitation, recent work has investigated motion-induced event stimulation that often requires complex hardware or additional optical components. In contrast, we introduce a lightweight approach to sustain persistent event generation by employing a simple rotating unbalanced mass to induce periodic vibrational motion. This is combined with a motion-compensation pipeline that removes the injected motion and yields clean, motion-corrected events for downstream perception tasks. We demonstrate our approach with a hardware prototype and evaluate it on real-world captured datasets. Our method reliably recovers motion parameters and improves both image reconstruction and edge detection over event-based sensing without motion induction.
>
---
#### [new 032] Towards Training-Free Underwater 3D Object Detection from Sonar Point Clouds: A Comparison of Traditional and Deep Learning Approaches
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究水下3D物体检测任务，解决传统方法因声学环境恶劣和数据稀缺导致的检测难题。通过合成数据训练与模板匹配结合，验证了无需真实数据即可实现83% mAP的鲁棒检测，挑战了深度学习对数据的依赖。**

- **链接: [http://arxiv.org/pdf/2508.18293v1](http://arxiv.org/pdf/2508.18293v1)**

> **作者:** M. Salman Shaukat; Yannik Käckenmeister; Sebastian Bader; Thomas Kirste
>
> **备注:** 12 pages, 7 figures, submitted to IEEE Journal of Oceanic Engineering (IEEE-JOE)
>
> **摘要:** Underwater 3D object detection remains one of the most challenging frontiers in computer vision, where traditional approaches struggle with the harsh acoustic environment and scarcity of training data. While deep learning has revolutionized terrestrial 3D detection, its application underwater faces a critical bottleneck: obtaining sufficient annotated sonar data is prohibitively expensive and logistically complex, often requiring specialized vessels, expert surveyors, and favorable weather conditions. This work addresses a fundamental question: Can we achieve reliable underwater 3D object detection without real-world training data? We tackle this challenge by developing and comparing two paradigms for training-free detection of artificial structures in multibeam echo-sounder point clouds. Our dual approach combines a physics-based sonar simulation pipeline that generates synthetic training data for state-of-the-art neural networks, with a robust model-based template matching system that leverages geometric priors of target objects. Evaluation on real bathymetry surveys from the Baltic Sea reveals surprising insights: while neural networks trained on synthetic data achieve 98% mean Average Precision (mAP) on simulated scenes, they drop to 40% mAP on real sonar data due to domain shift. Conversely, our template matching approach maintains 83% mAP on real data without requiring any training, demonstrating remarkable robustness to acoustic noise and environmental variations. Our findings challenge conventional wisdom about data-hungry deep learning in underwater domains and establish the first large-scale benchmark for training-free underwater 3D detection. This work opens new possibilities for autonomous underwater vehicle navigation, marine archaeology, and offshore infrastructure monitoring in data-scarce environments where traditional machine learning approaches fail.
>
---
#### [new 033] PseudoMapTrainer: Learning Online Mapping without HD Maps
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 论文提出PseudoMapTrainer，用于在线制图任务，解决传统方法依赖昂贵且地理覆盖不足的高精度地图问题。通过生成伪标签（基于多视角图像与预训练分割网络）和改进的掩码处理算法，实现无需真实地图的模型训练与半监督预训练。**

- **链接: [http://arxiv.org/pdf/2508.18788v1](http://arxiv.org/pdf/2508.18788v1)**

> **作者:** Christian Löwens; Thorben Funke; Jingchao Xie; Alexandru Paul Condurache
>
> **备注:** Accepted at ICCV 2025
>
> **摘要:** Online mapping models show remarkable results in predicting vectorized maps from multi-view camera images only. However, all existing approaches still rely on ground-truth high-definition maps during training, which are expensive to obtain and often not geographically diverse enough for reliable generalization. In this work, we propose PseudoMapTrainer, a novel approach to online mapping that uses pseudo-labels generated from unlabeled sensor data. We derive those pseudo-labels by reconstructing the road surface from multi-camera imagery using Gaussian splatting and semantics of a pre-trained 2D segmentation network. In addition, we introduce a mask-aware assignment algorithm and loss function to handle partially masked pseudo-labels, allowing for the first time the training of online mapping models without any ground-truth maps. Furthermore, our pseudo-labels can be effectively used to pre-train an online model in a semi-supervised manner to leverage large-scale unlabeled crowdsourced data. The code is available at github.com/boschresearch/PseudoMapTrainer.
>
---
#### [new 034] Safe Navigation under State Uncertainty: Online Adaptation for Robust Control Barrier Functions
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对状态不确定性下的安全导航问题，提出改进鲁棒控制屏障函数（R-CBF）的方法，通过在线参数适应与约束合并，提升系统鲁棒性与跟踪性能。**

- **链接: [http://arxiv.org/pdf/2508.19159v1](http://arxiv.org/pdf/2508.19159v1)**

> **作者:** Ersin Das; Rahal Nanayakkara; Xiao Tan; Ryan M. Bena; Joel W. Burdick; Paulo Tabuada; Aaron D. Ames
>
> **摘要:** Measurements and state estimates are often imperfect in control practice, posing challenges for safety-critical applications, where safety guarantees rely on accurate state information. In the presence of estimation errors, several prior robust control barrier function (R-CBF) formulations have imposed strict conditions on the input. These methods can be overly conservative and can introduce issues such as infeasibility, high control effort, etc. This work proposes a systematic method to improve R-CBFs, and demonstrates its advantages on a tracked vehicle that navigates among multiple obstacles. A primary contribution is a new optimization-based online parameter adaptation scheme that reduces the conservativeness of existing R-CBFs. In order to reduce the complexity of the parameter optimization, we merge several safety constraints into one unified numerical CBF via Poisson's equation. We further address the dual relative degree issue that typically causes difficulty in vehicle tracking. Experimental trials demonstrate the overall performance improvement of our approach over existing formulations.
>
---
#### [new 035] Interpretable Decision-Making for End-to-End Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对端到端自动驾驶中深度学习决策不可解释的问题，提出通过设计损失函数生成稀疏特征图以增强模型可解释性，并在CARLA基准上验证其提升安全性和性能的效果。**

- **链接: [http://arxiv.org/pdf/2508.18898v1](http://arxiv.org/pdf/2508.18898v1)**

> **作者:** Mona Mirzaie; Bodo Rosenhahn
>
> **备注:** Accepted to the ICCV 2025 2nd Workshop on the Challenge Of Out-of-Label Hazards in Autonomous Driving (2COOOL)
>
> **摘要:** Trustworthy AI is mandatory for the broad deployment of autonomous vehicles. Although end-to-end approaches derive control commands directly from raw data, interpreting these decisions remains challenging, especially in complex urban scenarios. This is mainly attributed to very deep neural networks with non-linear decision boundaries, making it challenging to grasp the logic behind AI-driven decisions. This paper presents a method to enhance interpretability while optimizing control commands in autonomous driving. To address this, we propose loss functions that promote the interpretability of our model by generating sparse and localized feature maps. The feature activations allow us to explain which image regions contribute to the predicted control command. We conduct comprehensive ablation studies on the feature extraction step and validate our method on the CARLA benchmarks. We also demonstrate that our approach improves interpretability, which correlates with reducing infractions, yielding a safer, high-performance driving model. Notably, our monocular, non-ensemble model surpasses the top-performing approaches from the CARLA Leaderboard by achieving lower infraction scores and the highest route completion rate, all while ensuring interpretability.
>
---
## 更新

#### [replaced 001] Ego-Foresight: Self-supervised Learning of Agent-Aware Representations for Improved RL
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2407.01570v3](http://arxiv.org/pdf/2407.01570v3)**

> **作者:** Manuel Serra Nunes; Atabak Dehban; Yiannis Demiris; José Santos-Victor
>
> **备注:** 13 pages, 8 figures, conference
>
> **摘要:** Despite the significant advancements in Deep Reinforcement Learning (RL) observed in the last decade, the amount of training experience necessary to learn effective policies remains one of the primary concerns both in simulated and real environments. Looking to solve this issue, previous work has shown that improved training efficiency can be achieved by separately modeling agent and environment, but usually requiring a supervisory agent mask. In contrast to RL, humans can perfect a new skill from a small number of trials and in most cases do so without a supervisory signal, making neuroscientific studies of human development a valuable source of inspiration for RL. In particular, we explore the idea of motor prediction, which states that humans develop an internal model of themselves and of the consequences that their motor commands have on the immediate sensory inputs. Our insight is that the movement of the agent provides a cue that allows the duality between agent and environment to be learned. To instantiate this idea, we present Ego-Foresight, a self-supervised method for disentangling agent and environment based on motion and prediction. Our main finding is self-supervised agent-awareness by visuomotor prediction of the agent improves sample-efficiency and performance of the underlying RL algorithm. To test our approach, we first study its ability to visually predict agent movement irrespective of the environment, in simulated and real-world robotic data. Then, we integrate Ego-Foresight with a model-free RL algorithm to solve simulated robotic tasks, showing that self-supervised agent-awareness can improve sample-efficiency and performance in RL.
>
---
#### [replaced 002] FlowVLA: Thinking in Motion with a Visual Chain of Thought
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.18269v2](http://arxiv.org/pdf/2508.18269v2)**

> **作者:** Zhide Zhong; Haodong Yan; Junfeng Li; Xiangchen Liu; Xin Gong; Wenxuan Song; Jiayi Chen; Haoang Li
>
> **摘要:** Many Vision-Language-Action (VLA) models are built upon an internal world model trained via direct next-frame prediction ($v_t \rightarrow v_{t+1}$). This paradigm, however, presents a fundamental challenge: it \textbf{conflates} the task of predicting physical motion with that of rendering static appearance, forcing a single mechanism to handle both. This inherent coupling often leads to physically implausible forecasts and inefficient policy learning. To address this limitation, we introduce the \textbf{Visual Chain of Thought (Visual CoT)}, a framework that disentangles these processes by compelling the model to first reason about \textbf{motion dynamics} before generating the future frame's \textbf{visual appearance}. We instantiate this principle by proposing \textbf{FlowVLA}, an autoregressive Transformer that explicitly materializes this reasoning process as ``$v_t \rightarrow f_t \rightarrow v_{t+1}$'', where $f_t$ is an intermediate optical flow prediction. By forcing the model to first commit to a motion plan ($f_t$), FlowVLA learns disentangled dynamics, resulting in more coherent visual predictions and significantly more efficient policy learning. Experiments on challenging robotics manipulation benchmarks demonstrate that FlowVLA achieves state-of-the-art performance with substantially improved sample efficiency, pointing toward a more principled foundation for world modeling in VLAs. Project page: https://irpn-lab.github.io/FlowVLA/
>
---
#### [replaced 003] Trajectory Optimization for UAV-Based Medical Delivery with Temporal Logic Constraints and Convex Feasible Set Collision Avoidance
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2506.06038v2](http://arxiv.org/pdf/2506.06038v2)**

> **作者:** Kaiyuan Chen; Yuhan Suo; Shaowei Cui; Yuanqing Xia; Wannian Liang; Shuo Wang
>
> **备注:** 11 pages, 4 figures
>
> **摘要:** This paper addresses the problem of trajectory optimization for unmanned aerial vehicles (UAVs) performing time-sensitive medical deliveries in urban environments. Specifically, we consider a single UAV with 3 degree-of-freedom dynamics tasked with delivering blood packages to multiple hospitals, each with a predefined time window and priority. Mission objectives are encoded using Signal Temporal Logic (STL), enabling the formal specification of spatial-temporal constraints. To ensure safety, city buildings are modeled as 3D convex obstacles, and obstacle avoidance is handled through a Convex Feasible Set (CFS) method. The entire planning problem-combining UAV dynamics, STL satisfaction, and collision avoidance-is formulated as a convex optimization problem that ensures tractability and can be solved efficiently using standard convex programming techniques. Simulation results demonstrate that the proposed method generates dynamically feasible, collision-free trajectories that satisfy temporal mission goals, providing a scalable and reliable approach for autonomous UAV-based medical logistics.
>
---
#### [replaced 004] Safe Multiagent Coordination via Entropic Exploration
- **分类: cs.MA; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.20361v2](http://arxiv.org/pdf/2412.20361v2)**

> **作者:** Ayhan Alp Aydeniz; Enrico Marchesini; Robert Loftin; Christopher Amato; Kagan Tumer
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Many real-world multiagent learning problems involve safety concerns. In these setups, typical safe reinforcement learning algorithms constrain agents' behavior, limiting exploration -- a crucial component for discovering effective cooperative multiagent behaviors. Moreover, the multiagent literature typically models individual constraints for each agent and has yet to investigate the benefits of using joint team constraints. In this work, we analyze these team constraints from a theoretical and practical perspective and propose entropic exploration for constrained multiagent reinforcement learning (E2C) to address the exploration issue. E2C leverages observation entropy maximization to incentivize exploration and facilitate learning safe and effective cooperative behaviors. Experiments across increasingly complex domains show that E2C agents match or surpass common unconstrained and constrained baselines in task performance while reducing unsafe behaviors by up to $50\%$.
>
---
#### [replaced 005] Enhancing Multi-Robot Semantic Navigation Through Multimodal Chain-of-Thought Score Collaboration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.18292v5](http://arxiv.org/pdf/2412.18292v5)**

> **作者:** Zhixuan Shen; Haonan Luo; Kexun Chen; Fengmao Lv; Tianrui Li
>
> **备注:** 16 pages, 10 figures, Extended Version of accepted AAAI 2025 Paper
>
> **摘要:** Understanding how humans cooperatively utilize semantic knowledge to explore unfamiliar environments and decide on navigation directions is critical for house service multi-robot systems. Previous methods primarily focused on single-robot centralized planning strategies, which severely limited exploration efficiency. Recent research has considered decentralized planning strategies for multiple robots, assigning separate planning models to each robot, but these approaches often overlook communication costs. In this work, we propose Multimodal Chain-of-Thought Co-Navigation (MCoCoNav), a modular approach that utilizes multimodal Chain-of-Thought to plan collaborative semantic navigation for multiple robots. MCoCoNav combines visual perception with Vision Language Models (VLMs) to evaluate exploration value through probabilistic scoring, thus reducing time costs and achieving stable outputs. Additionally, a global semantic map is used as a communication bridge, minimizing communication overhead while integrating observational results. Guided by scores that reflect exploration trends, robots utilize this map to assess whether to explore new frontier points or revisit history nodes. Experiments on HM3D_v0.2 and MP3D demonstrate the effectiveness of our approach. Our code is available at https://github.com/FrankZxShen/MCoCoNav.git.
>
---
#### [replaced 006] Robot Trains Robot: Automatic Real-World Policy Adaptation and Learning for Humanoids
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.12252v2](http://arxiv.org/pdf/2508.12252v2)**

> **作者:** Kaizhe Hu; Haochen Shi; Yao He; Weizhuo Wang; C. Karen Liu; Shuran Song
>
> **备注:** Accepted to The Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Simulation-based reinforcement learning (RL) has significantly advanced humanoid locomotion tasks, yet direct real-world RL from scratch or adapting from pretrained policies remains rare, limiting the full potential of humanoid robots. Real-world learning, despite being crucial for overcoming the sim-to-real gap, faces substantial challenges related to safety, reward design, and learning efficiency. To address these limitations, we propose Robot-Trains-Robot (RTR), a novel framework where a robotic arm teacher actively supports and guides a humanoid robot student. The RTR system provides protection, learning schedule, reward, perturbation, failure detection, and automatic resets. It enables efficient long-term real-world humanoid training with minimal human intervention. Furthermore, we propose a novel RL pipeline that facilitates and stabilizes sim-to-real transfer by optimizing a single dynamics-encoded latent variable in the real world. We validate our method through two challenging real-world humanoid tasks: fine-tuning a walking policy for precise speed tracking and learning a humanoid swing-up task from scratch, illustrating the promising capabilities of real-world humanoid learning realized by RTR-style systems. See https://robot-trains-robot.github.io/ for more info.
>
---
#### [replaced 007] Bayesian Deep Learning for Segmentation for Autonomous Safe Planetary Landing
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2102.10545v3](http://arxiv.org/pdf/2102.10545v3)**

> **作者:** Kento Tomita; Katherine A. Skinner; Koki Ho
>
> **备注:** 18 pages, 9 figures, Accepted by the AIAA Journal of Spacecraft and Rockets, revised from Paper AAS 21-253 presented at the AAS/AIAA Space Flight Mechanics Meeting in 2021
>
> **摘要:** Hazard detection is critical for enabling autonomous landing on planetary surfaces. Current state-of-the-art methods leverage traditional computer vision approaches to automate the identification of safe terrain from input digital elevation models (DEMs). However, performance for these methods can degrade for input DEMs with increased sensor noise. In the last decade, deep learning techniques have been developed for various applications. Nevertheless, their applicability to safety-critical space missions has often been limited due to concerns regarding their outputs' reliability. In response to these limitations, this paper proposes an application of the Bayesian deep-learning segmentation method for hazard detection. The developed approach enables reliable, safe landing site detection by: (i) generating simultaneously a safety prediction map and its uncertainty map via Bayesian deep learning and semantic segmentation; and (ii) using the uncertainty map to filter out the uncertain pixels in the prediction map so that the safe site identification is performed only based on the certain pixels (i.e., pixels for which the model is certain about its safety prediction). Experiments are presented with simulated data based on a Mars HiRISE digital terrain model by varying uncertainty threshold and noise levels to demonstrate the performance of the proposed approach.
>
---
#### [replaced 008] Trajectory-to-Action Pipeline (TAP): Automated Scenario Description Extraction for Autonomous Vehicle Behavior Comparison
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.20353v2](http://arxiv.org/pdf/2502.20353v2)**

> **作者:** Aron Harder; Madhur Behl
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Scenario Description Languages (SDLs) provide structured, interpretable embeddings that represent traffic scenarios encountered by autonomous vehicles (AVs), supporting key tasks such as scenario similarity searches and edge case detection for safety analysis. This paper introduces the Trajectory-to-Action Pipeline (TAP), a scalable and automated method for extracting SDL labels from large trajectory datasets. TAP applies a rules-based cross-entropy optimization approach to learn parameters directly from data, enhancing generalization across diverse driving contexts. Using the Waymo Open Motion Dataset (WOMD), TAP achieves 30% greater precision than Average Displacement Error (ADE) and 24% over Dynamic Time Warping (DTW) in identifying behaviorally similar trajectories. Additionally, TAP enables automated detection of unique driving behaviors, streamlining safety evaluation processes for AV testing. This work provides a foundation for scalable scenario-based AV behavior analysis, with potential extensions for integrating multi-agent contexts.
>
---
#### [replaced 009] Early Failure Detection in Autonomous Surgical Soft-Tissue Manipulation via Uncertainty Quantification
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.10561v2](http://arxiv.org/pdf/2501.10561v2)**

> **作者:** Jordan Thompson; Ronald Koe; Anthony Le; Gabriella Goodman; Daniel S. Brown; Alan Kuntz
>
> **备注:** 6 pages, 6 figures, Accepted to the 2025 RSS OOD Workshop
>
> **摘要:** Autonomous surgical robots are a promising solution to the increasing demand for surgery amid a shortage of surgeons. Recent work has proposed learning-based approaches for the autonomous manipulation of soft tissue. However, due to variability in tissue geometries and stiffnesses, these methods do not always perform optimally, especially in out-of-distribution settings. We propose, develop, and test the first application of uncertainty quantification to learned surgical soft-tissue manipulation policies as an early identification system for task failures. We analyze two different methods of uncertainty quantification, deep ensembles and Monte Carlo dropout, and find that deep ensembles provide a stronger signal of future task success or failure. We validate our approach using the physical daVinci Research Kit (dVRK) surgical robot to perform physical soft-tissue manipulation. We show that we are able to successfully detect out-of-distribution states leading to task failure and request human intervention when necessary while still enabling autonomous manipulation when possible. Our learned tissue manipulation policy with uncertainty-based early failure detection achieves a zero-shot sim2real performance improvement of 47.5% over the prior state of the art in learned soft-tissue manipulation. We also show that our method generalizes well to new types of tissue as well as to a bimanual soft-tissue manipulation task.
>
---
#### [replaced 010] Dojo: A Differentiable Physics Engine for Robotics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2203.00806v5](http://arxiv.org/pdf/2203.00806v5)**

> **作者:** Taylor A. Howell; Simon Le Cleac'h; Jan Brüdigam; Qianzhong Chen; Jiankai Sun; J. Zico Kolter; Mac Schwager; Zachary Manchester
>
> **摘要:** We present Dojo, a differentiable physics engine for robotics that prioritizes stable simulation, accurate contact physics, and differentiability with respect to states, actions, and system parameters. Dojo models hard contact and friction with a nonlinear complementarity problem with second-order cone constraints. We introduce a custom primal-dual interior-point method to solve the second order cone program for stable forward simulation over a broad range of sample rates. We obtain smooth gradient approximations with this solver through the implicit function theorem, giving gradients that are useful for downstream trajectory optimization, policy optimization, and system identification applications. Specifically, we propose to use the central path parameter threshold in the interior point solver as a user-tunable design parameter. A high value gives a smooth approximation to contact dynamics with smooth gradients for optimization and learning, while a low value gives precise simulation rollouts with hard contact. We demonstrate Dojo's differentiability in trajectory optimization, policy learning, and system identification examples. We also benchmark Dojo against MuJoCo, PyBullet, Drake, and Brax on a variety of robot models, and study the stability and simulation quality over a range of sample frequencies and accuracy tolerances. Finally, we evaluate the sim-to-real gap in hardware experiments with a Ufactory xArm 6 robot. Dojo is an open source project implemented in Julia with Python bindings, with code available at https://github.com/dojo-sim/Dojo.jl.
>
---
#### [replaced 011] SE-VLN: A Self-Evolving Vision-Language Navigation Framework Based on Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.13152v3](http://arxiv.org/pdf/2507.13152v3)**

> **作者:** Xiangyu Dong; Haoran Zhao; Jiang Gao; Haozhou Li; Xiaoguang Ma; Yaoming Zhou; Fuhai Chen; Juan Liu
>
> **摘要:** Recent advances in vision-language navigation (VLN) were mainly attributed to emerging large language models (LLMs). These methods exhibited excellent generalization capabilities in instruction understanding and task reasoning. However, they were constrained by the fixed knowledge bases and reasoning abilities of LLMs, preventing fully incorporating experiential knowledge and thus resulting in a lack of efficient evolutionary capacity. To address this, we drew inspiration from the evolution capabilities of natural agents, and proposed a self-evolving VLN framework (SE-VLN) to endow VLN agents with the ability to continuously evolve during testing. To the best of our knowledge, it was the first time that an multimodal LLM-powered self-evolving VLN framework was proposed. Specifically, SE-VLN comprised three core modules, i.e., a hierarchical memory module to transfer successful and failure cases into reusable knowledge, a retrieval-augmented thought-based reasoning module to retrieve experience and enable multi-step decision-making, and a reflection module to realize continual evolution. Comprehensive tests illustrated that the SE-VLN achieved navigation success rates of 57% and 35.2% in unseen environments, representing absolute performance improvements of 23.9% and 15.0% over current state-of-the-art methods on R2R and REVERSE datasets, respectively. Moreover, the SE-VLN showed performance improvement with increasing experience repository, elucidating its great potential as a self-evolving agent framework for VLN.
>
---
#### [replaced 012] FUSELOC: Fusing Global and Local Descriptors to Disambiguate 2D-3D Matching in Visual Localization
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2408.12037v2](http://arxiv.org/pdf/2408.12037v2)**

> **作者:** Son Tung Nguyen; Alejandro Fontan; Michael Milford; Tobias Fischer
>
> **摘要:** Hierarchical visual localization methods achieve state-of-the-art accuracy but require substantial memory as they need to store all database images. Direct 2D-3D matching requires significantly less memory but suffers from lower accuracy due to the larger and more ambiguous search space. We address this ambiguity by fusing local and global descriptors using a weighted average operator. This operator rearranges the local descriptor space so that geographically nearby local descriptors are closer in the feature space according to the global descriptors. This decreases the number of irrelevant competing descriptors, especially if they are geographically distant, thus increasing the correct matching likelihood. We consistently improve the accuracy over local-only systems, and we achieve performance close to hierarchical methods while using 43\% less memory and running 1.6 times faster. Extensive experiments on four challenging datasets -- Cambridge Landmarks, Aachen Day/Night, RobotCar Seasons, and Extended CMU Seasons -- demonstrate that, for the first time, direct matching algorithms can benefit from global descriptors without compromising computational efficiency. Our code is available at \href{https://github.com/sontung/descriptor-disambiguation}{https://github.com/sontung/descriptor-disambiguation}.
>
---
#### [replaced 013] ParticleFormer: A 3D Point Cloud World Model for Multi-Object, Multi-Material Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.23126v4](http://arxiv.org/pdf/2506.23126v4)**

> **作者:** Suning Huang; Qianzhong Chen; Xiaohan Zhang; Jiankai Sun; Mac Schwager
>
> **摘要:** 3D world models (i.e., learning-based 3D dynamics models) offer a promising approach to generalizable robotic manipulation by capturing the underlying physics of environment evolution conditioned on robot actions. However, existing 3D world models are primarily limited to single-material dynamics using a particle-based Graph Neural Network model, and often require time-consuming 3D scene reconstruction to obtain 3D particle tracks for training. In this work, we present ParticleFormer, a Transformer-based point cloud world model trained with a hybrid point cloud reconstruction loss, supervising both global and local dynamics features in multi-material, multi-object robot interactions. ParticleFormer captures fine-grained multi-object interactions between rigid, deformable, and flexible materials, trained directly from real-world robot perception data without an elaborate scene reconstruction. We demonstrate the model's effectiveness both in 3D scene forecasting tasks, and in downstream manipulation tasks using a Model Predictive Control (MPC) policy. In addition, we extend existing dynamics learning benchmarks to include diverse multi-material, multi-object interaction scenarios. We validate our method on six simulation and three real-world experiments, where it consistently outperforms leading baselines by achieving superior dynamics prediction accuracy and less rollout error in downstream visuomotor tasks. Experimental videos are available at https://suninghuang19.github.io/particleformer_page/.
>
---
#### [replaced 014] A Value Function Space Approach for Hierarchical Planning with Signal Temporal Logic Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.01923v2](http://arxiv.org/pdf/2408.01923v2)**

> **作者:** Peiran Liu; Yiting He; Yihao Qin; Hang Zhou; Yiding Ji
>
> **摘要:** Signal Temporal Logic (STL) has emerged as an expressive language for reasoning intricate planning objectives. However, existing STL-based methods often assume full observation and known dynamics, which imposes constraints on real-world applications. To address this challenge, we propose a hierarchical planning framework that starts by constructing the Value Function Space (VFS) for state and action abstraction, which embeds functional information about affordances of the low-level skills. Subsequently, we utilize a neural network to approximate the dynamics in the VFS and employ sampling based optimization to synthesize high-level skill sequences that maximize the robustness measure of the given STL tasks in the VFS. Then those skills are executed in the low-level environment. Empirical evaluations in the Safety Gym and ManiSkill environments demonstrate that our method accomplish the STL tasks without further training in the low-level environments, substantially reducing the training burdens.
>
---
#### [replaced 015] A Third-Order Gaussian Process Trajectory Representation Framework with Closed-Form Kinematics for Continuous-Time Motion Estimation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.22931v5](http://arxiv.org/pdf/2410.22931v5)**

> **作者:** Thien-Minh Nguyen; Ziyu Cao; Kailai Li; William Talbot; Tongxing Jin; Shenghai Yuan; Timothy D. Barfoot; Lihua Xie
>
> **备注:** The paper is currently under review at IEEE Transactions on Robotics (T-RO). The source code has been released, and feedback is welcome
>
> **摘要:** In this paper, we propose a third-order, i.e., white-noise-on-jerk, Gaussian Process (GP) Trajectory Representation (TR) framework for continuous-time (CT) motion estimation (ME) tasks. Our framework features a unified trajectory representation that encapsulates the kinematic models of both $SO(3)\times\mathbb{R}^3$ and $SE(3)$ pose representations. This encapsulation strategy allows users to use the same implementation of measurement-based factors for either choice of pose representation, which facilitates experimentation and comparison to achieve the best model for the ME task. In addition, unique to our framework, we derive the kinematic models with the closed-form temporal derivatives of the local variable of $SO(3)$ and $SE(3)$, which so far has only been approximated based on the Taylor expansion in the literature. Our experiments show that these kinematic models can improve the estimation accuracy in high-speed scenarios. All analytical Jacobians of the interpolated states with respect to the support states of the trajectory representation, as well as the motion prior factors, are also provided for accelerated Gauss-Newton (GN) optimization. Our experiments demonstrate the efficacy and efficiency of the framework in various motion estimation tasks such as localization, calibration, and odometry, facilitating fast prototyping for ME researchers. We release the source code for the benefit of the community. Our project is available at https://github.com/brytsknguyen/gptr.
>
---
#### [replaced 016] Comparative Analysis of UAV Path Planning Algorithms for Efficient Navigation in Urban 3D Environments
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.16515v2](http://arxiv.org/pdf/2508.16515v2)**

> **作者:** Hichem Cheriet; Khellat Kihel Badra; Chouraqui Samira
>
> **备注:** AFROS 2024 Conference
>
> **摘要:** The most crucial challenges for UAVs are planning paths and avoiding obstacles in their way. In recent years, a wide variety of path-planning algorithms have been developed. These algorithms have successfully solved path-planning problems; however, they suffer from multiple challenges and limitations. To test the effectiveness and efficiency of three widely used algorithms, namely A*, RRT*, and Particle Swarm Optimization (PSO), this paper conducts extensive experiments in 3D urban city environments cluttered with obstacles. Three experiments were designed with two scenarios each to test the aforementioned algorithms. These experiments consider different city map sizes, different altitudes, and varying obstacle densities and sizes in the environment. According to the experimental results, the A* algorithm outperforms the others in both computation efficiency and path quality. PSO is especially suitable for tight turns and dense environments, and RRT* offers a balance and works well across all experiments due to its randomized approach to finding solutions.
>
---
#### [replaced 017] DreamVLA: A Vision-Language-Action Model Dreamed with Comprehensive World Knowledge
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04447v3](http://arxiv.org/pdf/2507.04447v3)**

> **作者:** Wenyao Zhang; Hongsi Liu; Zekun Qi; Yunnan Wang; Xinqiang Yu; Jiazhao Zhang; Runpei Dong; Jiawei He; Fan Lu; He Wang; Zhizheng Zhang; Li Yi; Wenjun Zeng; Xin Jin
>
> **摘要:** Recent advances in vision-language-action (VLA) models have shown promise in integrating image generation with action prediction to improve generalization and reasoning in robot manipulation. However, existing methods are limited to challenging image-based forecasting, which suffers from redundant information and lacks comprehensive and critical world knowledge, including dynamic, spatial and semantic information. To address these limitations, we propose DreamVLA, a novel VLA framework that integrates comprehensive world knowledge forecasting to enable inverse dynamics modeling, thereby establishing a perception-prediction-action loop for manipulation tasks. Specifically, DreamVLA introduces a dynamic-region-guided world knowledge prediction, integrated with the spatial and semantic cues, which provide compact yet comprehensive representations for action planning. This design aligns with how humans interact with the world by first forming abstract multimodal reasoning chains before acting. To mitigate interference among the dynamic, spatial and semantic information during training, we adopt a block-wise structured attention mechanism that masks their mutual attention, preventing information leakage and keeping each representation clean and disentangled. Moreover, to model the conditional distribution over future actions, we employ a diffusion-based transformer that disentangles action representations from shared latent features. Extensive experiments on both real-world and simulation environments demonstrate that DreamVLA achieves 76.7% success rate on real robot tasks and 4.44 average length on the CALVIN ABC-D benchmarks.
>
---
#### [replaced 018] Hierarchical Object-Oriented POMDP Planning for Object Rearrangement
- **分类: cs.LG; cs.AI; cs.RO; I.2.9**

- **链接: [http://arxiv.org/pdf/2412.01348v3](http://arxiv.org/pdf/2412.01348v3)**

> **作者:** Rajesh Mangannavar; Alan Fern; Prasad Tadepalli
>
> **备注:** 21 pages, 3 Figures. Preprint. Added more information in Appendix
>
> **摘要:** We present an online planning framework and a new benchmark dataset for solving multi-object rearrangement problems in partially observable, multi-room environments. Current object rearrangement solutions, primarily based on Reinforcement Learning or hand-coded planning methods, often lack adaptability to diverse challenges. To address this limitation, we introduce a novel Hierarchical Object-Oriented Partially Observed Markov Decision Process (HOO-POMDP) planning approach. This approach comprises of (a) an object-oriented POMDP planner generating sub-goals, (b) a set of low-level policies for sub-goal achievement, and (c) an abstraction system converting the continuous low-level world into a representation suitable for abstract planning. To enable rigorous evaluation of rearrangement challenges, we introduce MultiRoomR, a comprehensive benchmark featuring diverse multi-room environments with varying degrees of partial observability (10-30\% initial visibility), blocked paths, obstructed goals, and multiple objects (10-20) distributed across 2-4 rooms. Experiments demonstrate that our system effectively handles these complex scenarios while maintaining robust performance even with imperfect perception, achieving promising results across both existing benchmarks and our new MultiRoomR dataset.
>
---
#### [replaced 019] Enhancing Reusability of Learned Skills for Robot Manipulation via Gaze Information and Motion Bottlenecks
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.18121v3](http://arxiv.org/pdf/2502.18121v3)**

> **作者:** Ryo Takizawa; Izumi Karino; Koki Nakagawa; Yoshiyuki Ohmura; Yasuo Kuniyoshi
>
> **摘要:** Autonomous agents capable of diverse object manipulations should be able to acquire a wide range of manipulation skills with high reusability. Although advances in deep learning have made it increasingly feasible to replicate the dexterity of human teleoperation in robots, generalizing these acquired skills to previously unseen scenarios remains a significant challenge. In this study, we propose a novel algorithm, Gaze-based Bottleneck-aware Robot Manipulation (GazeBot), which enables high reusability of learned motions without sacrificing dexterity or reactivity. By leveraging gaze information and motion bottlenecks, both crucial features for object manipulation, GazeBot achieves high success rates compared with state-of-the-art imitation learning methods, particularly when the object positions and end-effector poses differ from those in the provided demonstrations. Furthermore, the training process of GazeBot is entirely data-driven once a demonstration dataset with gaze data is provided. Videos and code are available at https://crumbyrobotics.github.io/gazebot.
>
---
#### [replaced 020] CAD-Assistant: Tool-Augmented VLLMs as Generic CAD Task Solvers
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.13810v3](http://arxiv.org/pdf/2412.13810v3)**

> **作者:** Dimitrios Mallis; Ahmet Serdar Karadeniz; Sebastian Cavada; Danila Rukhovich; Niki Foteinopoulou; Kseniya Cherenkova; Anis Kacem; Djamila Aouada
>
> **摘要:** We propose CAD-Assistant, a general-purpose CAD agent for AI-assisted design. Our approach is based on a powerful Vision and Large Language Model (VLLM) as a planner and a tool-augmentation paradigm using CAD-specific tools. CAD-Assistant addresses multimodal user queries by generating actions that are iteratively executed on a Python interpreter equipped with the FreeCAD software, accessed via its Python API. Our framework is able to assess the impact of generated CAD commands on geometry and adapts subsequent actions based on the evolving state of the CAD design. We consider a wide range of CAD-specific tools including a sketch image parameterizer, rendering modules, a 2D cross-section generator, and other specialized routines. CAD-Assistant is evaluated on multiple CAD benchmarks, where it outperforms VLLM baselines and supervised task-specific methods. Beyond existing benchmarks, we qualitatively demonstrate the potential of tool-augmented VLLMs as general-purpose CAD solvers across diverse workflows.
>
---
#### [replaced 021] TRAN-D: 2D Gaussian Splatting-based Sparse-view Transparent Object Depth Reconstruction via Physics Simulation for Scene Update
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.11069v3](http://arxiv.org/pdf/2507.11069v3)**

> **作者:** Jeongyun Kim; Seunghoon Jeong; Giseop Kim; Myung-Hwan Jeon; Eunji Jun; Ayoung Kim
>
> **摘要:** Understanding the 3D geometry of transparent objects from RGB images is challenging due to their inherent physical properties, such as reflection and refraction. To address these difficulties, especially in scenarios with sparse views and dynamic environments, we introduce TRAN-D, a novel 2D Gaussian Splatting-based depth reconstruction method for transparent objects. Our key insight lies in separating transparent objects from the background, enabling focused optimization of Gaussians corresponding to the object. We mitigate artifacts with an object-aware loss that places Gaussians in obscured regions, ensuring coverage of invisible surfaces while reducing overfitting. Furthermore, we incorporate a physics-based simulation that refines the reconstruction in just a few seconds, effectively handling object removal and chain-reaction movement of remaining objects without the need for rescanning. TRAN-D is evaluated on both synthetic and real-world sequences, and it consistently demonstrated robust improvements over existing GS-based state-of-the-art methods. In comparison with baselines, TRAN-D reduces the mean absolute error by over 39% for the synthetic TRansPose sequences. Furthermore, despite being updated using only one image, TRAN-D reaches a {\delta} < 2.5 cm accuracy of 48.46%, over 1.5 times that of baselines, which uses six images. Code and more results are available at https://jeongyun0609.github.io/TRAN-D/.
>
---
#### [replaced 022] UAD: Unsupervised Affordance Distillation for Generalization in Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.09284v2](http://arxiv.org/pdf/2506.09284v2)**

> **作者:** Yihe Tang; Wenlong Huang; Yingke Wang; Chengshu Li; Roy Yuan; Ruohan Zhang; Jiajun Wu; Li Fei-Fei
>
> **摘要:** Understanding fine-grained object affordances is imperative for robots to manipulate objects in unstructured environments given open-ended task instructions. However, existing methods of visual affordance predictions often rely on manually annotated data or conditions only on a predefined set of tasks. We introduce UAD (Unsupervised Affordance Distillation), a method for distilling affordance knowledge from foundation models into a task-conditioned affordance model without any manual annotations. By leveraging the complementary strengths of large vision models and vision-language models, UAD automatically annotates a large-scale dataset with detailed $<$instruction, visual affordance$>$ pairs. Training only a lightweight task-conditioned decoder atop frozen features, UAD exhibits notable generalization to in-the-wild robotic scenes and to various human activities, despite only being trained on rendered objects in simulation. Using affordance provided by UAD as the observation space, we show an imitation learning policy that demonstrates promising generalization to unseen object instances, object categories, and even variations in task instructions after training on as few as 10 demonstrations. Project website: https://unsup-affordance.github.io/
>
---
#### [replaced 023] Multi-Touch and Bending Perception Using Electrical Impedance Tomography for Robotics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.13048v2](http://arxiv.org/pdf/2503.13048v2)**

> **作者:** Haofeng Chen; Bedrich Himmel; Bin Li; Xiaojie Wang; Matej Hoffmann
>
> **摘要:** Electrical Impedance Tomography (EIT) offers a promising solution for distributed tactile sensing with minimal wiring and full-surface coverage in robotic applications. However, EIT-based tactile sensors face significant challenges during surface bending. Deformation alters the baseline impedance distribution and couples with touch-induced conductivity variations, complicating signal interpretation. To address this challenge, we present a novel sensing framework that integrates a deep neural network for interaction state classification with a dynamic adaptive reference strategy to decouple touch and deformation signals, while a data-driven regression model translates EIT voltage changes into continuous bending angles. The framework is validated using a magnetic hydrogel composite sensor that conforms to bendable surfaces. Experimental evaluations demonstrate that the proposed framework achieves precise and robust bending angle estimation, high accuracy in distinguishing touch, bending, and idle states, and significantly improves touch localization quality under bending deformation compared to conventional fixed-reference methods. Real-time experiments confirm the system's capability to reliably detect multi-touch interactions and track bending angles across varying deformation conditions. This work paves the way for flexible EIT-based robotic skins capable of rich multimodal sensing in robotics and human-robot interaction.
>
---
#### [replaced 024] Learning Impact-Rich Rotational Maneuvers via Centroidal Velocity Rewards and Sim-to-Real Techniques: A One-Leg Hopper Flip Case Study
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12222v3](http://arxiv.org/pdf/2505.12222v3)**

> **作者:** Dongyun Kang; Gijeong Kim; JongHun Choe; Hajun Kim; Hae-Won Park
>
> **摘要:** Dynamic rotational maneuvers, such as front flips, inherently involve large angular momentum generation and intense impact forces, presenting major challenges for reinforcement learning and sim-to-real transfer. In this work, we propose a general framework for learning and deploying impact-rich, rotation-intensive behaviors through centroidal velocity-based rewards and actuator-aware sim-to-real techniques. We identify that conventional link-level reward formulations fail to induce true whole-body rotation and introduce a centroidal angular velocity reward that accurately captures system-wide rotational dynamics. To bridge the sim-to-real gap under extreme conditions, we model motor operating regions (MOR) and apply transmission load regularization to ensure realistic torque commands and mechanical robustness. Using the one-leg hopper front flip as a representative case study, we demonstrate the first successful hardware realization of a full front flip. Our results highlight that incorporating centroidal dynamics and actuator constraints is critical for reliably executing highly dynamic motions. A supplementary video is available at: https://youtu.be/atMAVI4s1RY
>
---
#### [replaced 025] Steerable Scene Generation with Post Training and Inference-Time Search
- **分类: cs.RO; cs.GR; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.04831v2](http://arxiv.org/pdf/2505.04831v2)**

> **作者:** Nicholas Pfaff; Hongkai Dai; Sergey Zakharov; Shun Iwase; Russ Tedrake
>
> **备注:** Project website: https://steerable-scene-generation.github.io/
>
> **摘要:** Training robots in simulation requires diverse 3D scenes that reflect the specific challenges of downstream tasks. However, scenes that satisfy strict task requirements, such as high-clutter environments with plausible spatial arrangement, are rare and costly to curate manually. Instead, we generate large-scale scene data using procedural models that approximate realistic environments for robotic manipulation, and adapt it to task-specific goals. We do this by training a unified diffusion-based generative model that predicts which objects to place from a fixed asset library, along with their SE(3) poses. This model serves as a flexible scene prior that can be adapted using reinforcement learning-based post training, conditional generation, or inference-time search, steering generation toward downstream objectives even when they differ from the original data distribution. Our method enables goal-directed scene synthesis that respects physical feasibility and scales across scene types. We introduce a novel MCTS-based inference-time search strategy for diffusion models, enforce feasibility via projection and simulation, and release a dataset of over 44 million SE(3) scenes spanning five diverse environments. Website with videos, code, data, and model weights: https://steerable-scene-generation.github.io/
>
---
