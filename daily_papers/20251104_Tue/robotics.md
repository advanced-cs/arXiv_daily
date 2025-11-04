# 机器人 cs.RO

- **最新发布 81 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] AERMANI-VLM: Structured Prompting and Reasoning for Aerial Manipulation with Vision Language Models
- **分类: cs.RO**

- **简介: 论文提出AERMANI-VLM，首次在无需微调下将视觉语言模型用于无人机操作，通过结构化提示分离高层推理与底层控制，利用安全技能库确保动作可行，解决VLM在空中操作中幻觉与不安全问题。**

- **链接: [http://arxiv.org/pdf/2511.01472v1](http://arxiv.org/pdf/2511.01472v1)**

> **作者:** Sarthak Mishra; Rishabh Dev Yadav; Avirup Das; Saksham Gupta; Wei Pan; Spandan Roy
>
> **摘要:** The rapid progress of vision--language models (VLMs) has sparked growing interest in robotic control, where natural language can express the operation goals while visual feedback links perception to action. However, directly deploying VLM-driven policies on aerial manipulators remains unsafe and unreliable since the generated actions are often inconsistent, hallucination-prone, and dynamically infeasible for flight. In this work, we present AERMANI-VLM, the first framework to adapt pretrained VLMs for aerial manipulation by separating high-level reasoning from low-level control, without any task-specific fine-tuning. Our framework encodes natural language instructions, task context, and safety constraints into a structured prompt that guides the model to generate a step-by-step reasoning trace in natural language. This reasoning output is used to select from a predefined library of discrete, flight-safe skills, ensuring interpretable and temporally consistent execution. By decoupling symbolic reasoning from physical action, AERMANI-VLM mitigates hallucinated commands and prevents unsafe behavior, enabling robust task completion. We validate the framework in both simulation and hardware on diverse multi-step pick-and-place tasks, demonstrating strong generalization to previously unseen commands, objects, and environments.
>
---
#### [new 002] FGO MythBusters: Explaining how Kalman Filter variants achieve the same performance as FGO in navigation applications
- **分类: cs.RO**

- **简介: 该论文澄清了SW-FGO与Kalman滤波变体（如EKF）的理论关联，提出Re-FGO框架统一二者，并证明在特定条件下等价，同时凸显SW-FGO在非线性、非高斯场景下的优势。**

- **链接: [http://arxiv.org/pdf/2511.00306v1](http://arxiv.org/pdf/2511.00306v1)**

> **作者:** Baoshan Song; Ruijie Xu; Li-Ta Hsu
>
> **摘要:** Sliding window-factor graph optimization (SW-FGO) has gained more and more attention in navigation research due to its robust approximation to non-Gaussian noises and nonlinearity of measuring models. There are lots of works focusing on its application performance compared to extended Kalman filter (EKF) but there is still a myth at the theoretical relationship between the SW-FGO and EKF. In this paper, we find the necessarily fair condition to connect SW-FGO and Kalman filter variants (KFV) (e.g., EKF, iterative EKF (IEKF), robust EKF (REKF) and robust iterative EKF (RIEKF)). Based on the conditions, we propose a recursive FGO (Re-FGO) framework to represent KFV under SW-FGO formulation. Under explicit conditions (Markov assumption, Gaussian noise with L2 loss, and a one-state window), Re-FGO regenerates exactly to EKF/IEKF/REKF/RIEKF, while SW-FGO shows measurable benefits in nonlinear, non-Gaussian regimes at a predictable compute cost. Finally, after clarifying the connection between them, we highlight the unique advantages of SW-FGO in practical phases, especially on numerical estimation and deep learning integration. The code and data used in this work is open sourced at https://github.com/Baoshan-Song/KFV-FGO-Comparison.
>
---
#### [new 003] Real-DRL: Teach and Learn in Reality
- **分类: cs.RO; cs.AI**

- **简介: 论文提出Real-DRL框架，解决安全关键系统中DRL的实时学习与安全保障问题。通过DRL学生、物理教师与触发器协同，实现安全优先的在线学习，突破仿真到现实的鸿沟与未知风险。**

- **链接: [http://arxiv.org/pdf/2511.00112v1](http://arxiv.org/pdf/2511.00112v1)**

> **作者:** Yanbing Mao; Yihao Cai; Lui Sha
>
> **备注:** 37 pages
>
> **摘要:** This paper introduces the Real-DRL framework for safety-critical autonomous systems, enabling runtime learning of a deep reinforcement learning (DRL) agent to develop safe and high-performance action policies in real plants (i.e., real physical systems to be controlled), while prioritizing safety! The Real-DRL consists of three interactive components: a DRL-Student, a PHY-Teacher, and a Trigger. The DRL-Student is a DRL agent that innovates in the dual self-learning and teaching-to-learn paradigm and the real-time safety-informed batch sampling. On the other hand, PHY-Teacher is a physics-model-based design of action policies that focuses solely on safety-critical functions. PHY-Teacher is novel in its real-time patch for two key missions: i) fostering the teaching-to-learn paradigm for DRL-Student and ii) backing up the safety of real plants. The Trigger manages the interaction between the DRL-Student and the PHY-Teacher. Powered by the three interactive components, the Real-DRL can effectively address safety challenges that arise from the unknown unknowns and the Sim2Real gap. Additionally, Real-DRL notably features i) assured safety, ii) automatic hierarchy learning (i.e., safety-first learning and then high-performance learning), and iii) safety-informed batch sampling to address the learning experience imbalance caused by corner cases. Experiments with a real quadruped robot, a quadruped robot in NVIDIA Isaac Gym, and a cart-pole system, along with comparisons and ablation studies, demonstrate the Real-DRL's effectiveness and unique features.
>
---
#### [new 004] CaRLi-V: Camera-RADAR-LiDAR Point-Wise 3D Velocity Estimation
- **分类: cs.RO**

- **简介: 论文提出CaRLi-V，融合RADAR、LiDAR与相机，实现稠密点云的三维速度估计，解决动态环境中机器人对非刚体目标的精准运动感知问题，通过速度立方体与光流联合求解，提升速度估计精度。**

- **链接: [http://arxiv.org/pdf/2511.01383v1](http://arxiv.org/pdf/2511.01383v1)**

> **作者:** Landson Guo; Andres M. Diaz Aguilar; William Talbot; Turcan Tuna; Marco Hutter; Cesar Cadena
>
> **摘要:** Accurate point-wise velocity estimation in 3D is crucial for robot interaction with non-rigid, dynamic agents, such as humans, enabling robust performance in path planning, collision avoidance, and object manipulation in dynamic environments. To this end, this paper proposes a novel RADAR, LiDAR, and camera fusion pipeline for point-wise 3D velocity estimation named CaRLi-V. This pipeline leverages raw RADAR measurements to create a novel RADAR representation, the velocity cube, which densely represents radial velocities within the RADAR's field-of-view. By combining the velocity cube for radial velocity extraction, optical flow for tangential velocity estimation, and LiDAR for point-wise range measurements through a closed-form solution, our approach can produce 3D velocity estimates for a dense array of points. Developed as an open-source ROS2 package, CaRLi-V has been field-tested against a custom dataset and proven to produce low velocity error metrics relative to ground truth, enabling point-wise velocity estimation for robotic applications.
>
---
#### [new 005] GauDP: Reinventing Multi-Agent Collaboration through Gaussian-Image Synergy in Diffusion Policies
- **分类: cs.RO**

- **简介: GauDP提出一种高斯-图像协同表征，解决多智能体协作中局部控制与全局感知难以兼顾的问题，通过共享3D高斯场实现无额外传感的可扩展模仿学习，在多机械臂操作任务中性能接近点云方法。**

- **链接: [http://arxiv.org/pdf/2511.00998v1](http://arxiv.org/pdf/2511.00998v1)**

> **作者:** Ziye Wang; Li Kang; Yiran Qin; Jiahua Ma; Zhanglin Peng; Lei Bai; Ruimao Zhang
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://ziyeeee.github.io/gaudp.io/
>
> **摘要:** Recently, effective coordination in embodied multi-agent systems has remained a fundamental challenge, particularly in scenarios where agents must balance individual perspectives with global environmental awareness. Existing approaches often struggle to balance fine-grained local control with comprehensive scene understanding, resulting in limited scalability and compromised collaboration quality. In this paper, we present GauDP, a novel Gaussian-image synergistic representation that facilitates scalable, perception-aware imitation learning in multi-agent collaborative systems. Specifically, GauDP constructs a globally consistent 3D Gaussian field from decentralized RGB observations, then dynamically redistributes 3D Gaussian attributes to each agent's local perspective. This enables all agents to adaptively query task-critical features from the shared scene representation while maintaining their individual viewpoints. This design facilitates both fine-grained control and globally coherent behavior without requiring additional sensing modalities (e.g., 3D point cloud). We evaluate GauDP on the RoboFactory benchmark, which includes diverse multi-arm manipulation tasks. Our method achieves superior performance over existing image-based methods and approaches the effectiveness of point-cloud-driven methods, while maintaining strong scalability as the number of agents increases.
>
---
#### [new 006] Embodiment Transfer Learning for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出ET-VLA框架，解决多机器人协作中视觉-语言-动作模型的迁移难题，通过合成预训练（SCP）和具身思维图（EGoT）实现跨机器人形态高效迁移，无需真实演示，显著提升任务性能。**

- **链接: [http://arxiv.org/pdf/2511.01224v1](http://arxiv.org/pdf/2511.01224v1)**

> **作者:** Chengmeng Li; Yaxin Peng
>
> **摘要:** Vision-language-action (VLA) models have significantly advanced robotic learning, enabling training on large-scale, cross-embodiment data and fine-tuning for specific robots. However, state-of-the-art autoregressive VLAs struggle with multi-robot collaboration. We introduce embodiment transfer learning, denoted as ET-VLA, a novel framework for efficient and effective transfer of pre-trained VLAs to multi-robot. ET-VLA's core is Synthetic Continued Pretraining (SCP), which uses synthetically generated data to warm up the model for the new embodiment, bypassing the need for real human demonstrations and reducing data collection costs. SCP enables the model to learn correct actions and precise action token numbers. Following SCP, the model is fine-tuned on target embodiment data. To further enhance the model performance on multi-embodiment, we present the Embodied Graph-of-Thought technique, a novel approach that formulates each sub-task as a node, that allows the VLA model to distinguish the functionalities and roles of each embodiment during task execution. Our work considers bimanual robots, a simple version of multi-robot to verify our approaches. We validate the effectiveness of our method on both simulation benchmarks and real robots covering three different bimanual embodiments. In particular, our proposed ET-VLA \space can outperform OpenVLA on six real-world tasks over 53.2%. We will open-source all codes to support the community in advancing VLA models for robot learning.
>
---
#### [new 007] Model to Model: Understanding the Venus Flytrap Snapping Mechanism and Transferring it to a 3D-printed Bistable Soft Robotic Demonstrator
- **分类: cs.RO**

- **简介: 该论文属于仿生机器人设计任务，旨在揭示捕蝇草双稳态 snapping 机制并将其迁移到人工软体执行器。通过分析叶片几何特征，3D打印了两个仿生双稳态 lobes，成功实现快速凹凸翻转，为开发仿生软体夹持器奠定基础。**

- **链接: [http://arxiv.org/pdf/2511.01350v1](http://arxiv.org/pdf/2511.01350v1)**

> **作者:** Maartje H. M. Wermelink; Renate Sachse; Sebastian Kruppert; Thomas Speck; Falk J. Tauber
>
> **备注:** Conference Proceedings Paper Living machines 2025
>
> **摘要:** The Venus flytrap (Dionaea muscipula) does not only serve as the textbook model for a carnivorous plant, but also has long intrigued both botanists and engineers with its rapidly closing leaf trap. The trap closure is triggered by two consecutive touches of a potential prey, after which the lobes rapidly switch from their concave open-state to their convex close-state and catch the prey within 100-500 ms after being triggered. This transformation from concave to convex is initiated by changes in turgor pressure and the release of stored elastic energy from prestresses in the concave state, which accelerate this movement, leading to inversion of the lobes bi-axial curvature. Possessing two low-energy states, the leaves can be characterized as bistable systems. With our research, we seek to deepen the understanding of Venus flytrap motion mechanics and apply its principles to the design of an artificial bistable lobe actuator. We identified geometrical characteristics, such as dimensional ratios and the thickness gradient in the lobe, and transferred these to two 3D-printed bistable actuator models. One actuator parallels the simulated geometry of a Venus flytrap leaf, the other is a lobe model designed with CAD. Both models display concave-convex bi-stability and snap close. These demonstrators are the first step in the development of an artificial Venus flytrap that mimics the mechanical behavior of the biological model and can be used as a soft fast gripper.
>
---
#### [new 008] SLAP: Shortcut Learning for Abstract Planning
- **分类: cs.RO; cs.LG**

- **简介: SLAP提出一种 Shortcut Learning 方法，利用现有 TAMP 抽象动作，通过强化学习自动发现新动作（如 slap、wiggle），缩短规划路径、提升任务成功率，解决稀疏奖励下长周期决策难题。**

- **链接: [http://arxiv.org/pdf/2511.01107v1](http://arxiv.org/pdf/2511.01107v1)**

> **作者:** Y. Isabel Liu; Bowen Li; Benjamin Eysenbach; Tom Silver
>
> **摘要:** Long-horizon decision-making with sparse rewards and continuous states and actions remains a fundamental challenge in AI and robotics. Task and motion planning (TAMP) is a model-based framework that addresses this challenge by planning hierarchically with abstract actions (options). These options are manually defined, limiting the agent to behaviors that we as human engineers know how to program (pick, place, move). In this work, we propose Shortcut Learning for Abstract Planning (SLAP), a method that leverages existing TAMP options to automatically discover new ones. Our key idea is to use model-free reinforcement learning (RL) to learn shortcuts in the abstract planning graph induced by the existing options in TAMP. Without any additional assumptions or inputs, shortcut learning leads to shorter solutions than pure planning, and higher task success rates than flat and hierarchical RL. Qualitatively, SLAP discovers dynamic physical improvisations (e.g., slap, wiggle, wipe) that differ significantly from the manually-defined ones. In experiments in four simulated robotic environments, we show that SLAP solves and generalizes to a wide range of tasks, reducing overall plan lengths by over 50% and consistently outperforming planning and RL baselines.
>
---
#### [new 009] URDF-Anything: Constructing Articulated Objects with 3D Multimodal Language Model
- **分类: cs.RO; cs.AI; I.2.6**

- **简介: URDF-Anything提出一种基于3D多模态大语言模型的端到端框架，自动从点云与文本输入重建关节对象的几何分割与运动学参数，解决传统人工建模低效问题，显著提升分割精度、参数准确性与物理可执行性。**

- **链接: [http://arxiv.org/pdf/2511.00940v1](http://arxiv.org/pdf/2511.00940v1)**

> **作者:** Zhe Li; Xiang Bai; Jieyu Zhang; Zhuangzhe Wu; Che Xu; Ying Li; Chengkai Hou; Shanghang Zhang
>
> **备注:** Accepted to the 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Constructing accurate digital twins of articulated objects is essential for robotic simulation training and embodied AI world model building, yet historically requires painstaking manual modeling or multi-stage pipelines. In this work, we propose \textbf{URDF-Anything}, an end-to-end automatic reconstruction framework based on a 3D multimodal large language model (MLLM). URDF-Anything utilizes an autoregressive prediction framework based on point-cloud and text multimodal input to jointly optimize geometric segmentation and kinematic parameter prediction. It implements a specialized $[SEG]$ token mechanism that interacts directly with point cloud features, enabling fine-grained part-level segmentation while maintaining consistency with the kinematic parameter predictions. Experiments on both simulated and real-world datasets demonstrate that our method significantly outperforms existing approaches regarding geometric segmentation (mIoU 17\% improvement), kinematic parameter prediction (average error reduction of 29\%), and physical executability (surpassing baselines by 50\%). Notably, our method exhibits excellent generalization ability, performing well even on objects outside the training set. This work provides an efficient solution for constructing digital twins for robotic simulation, significantly enhancing the sim-to-real transfer capability.
>
---
#### [new 010] Maestro: Orchestrating Robotics Modules with Vision-Language Models for Zero-Shot Generalist Robots
- **分类: cs.RO; cs.AI**

- **简介: 论文提出Maestro，利用视觉语言模型动态编排机器人感知、规划与控制模块，构建零样本通用机器人策略，突破传统端到端训练限制，实现高效、可扩展、易适配的机器人任务执行。**

- **链接: [http://arxiv.org/pdf/2511.00917v1](http://arxiv.org/pdf/2511.00917v1)**

> **作者:** Junyao Shi; Rujia Yang; Kaitian Chao; Selina Bingqing Wan; Yifei Shao; Jiahui Lei; Jianing Qian; Long Le; Pratik Chaudhari; Kostas Daniilidis; Chuan Wen; Dinesh Jayaraman
>
> **备注:** Project website: https://maestro-robot.github.io
>
> **摘要:** Today's best-explored routes towards generalist robots center on collecting ever larger "observations-in actions-out" robotics datasets to train large end-to-end models, copying a recipe that has worked for vision-language models (VLMs). We pursue a road less traveled: building generalist policies directly around VLMs by augmenting their general capabilities with specific robot capabilities encapsulated in a carefully curated set of perception, planning, and control modules. In Maestro, a VLM coding agent dynamically composes these modules into a programmatic policy for the current task and scenario. Maestro's architecture benefits from a streamlined closed-loop interface without many manually imposed structural constraints, and a comprehensive and diverse tool repertoire. As a result, it largely surpasses today's VLA models for zero-shot performance on challenging manipulation skills. Further, Maestro is easily extensible to incorporate new modules, easily editable to suit new embodiments such as a quadruped-mounted arm, and even easily adapts from minimal real-world experiences through local code edits.
>
---
#### [new 011] Real-Time Learning of Predictive Dynamic Obstacle Models for Robotic Motion Planning
- **分类: cs.RO; cs.LG; cs.SY; eess.SY; 93C41, 93E11, 37M10; I.2.9; I.2.6; I.2.8**

- **简介: 该论文提出一种实时学习动态障碍物预测模型的方法，基于改进的Hankel-DMD实现噪声数据的去噪与短时预测，支持风险感知运动规划，适用于机器人实时控制。**

- **链接: [http://arxiv.org/pdf/2511.00814v1](http://arxiv.org/pdf/2511.00814v1)**

> **作者:** Stella Kombo; Masih Haseli; Skylar Wei; Joel W. Burdick
>
> **备注:** 10 pages, 6 figures, submitted to IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** Autonomous systems often must predict the motions of nearby agents from partial and noisy data. This paper asks and answers the question: "can we learn, in real-time, a nonlinear predictive model of another agent's motions?" Our online framework denoises and forecasts such dynamics using a modified sliding-window Hankel Dynamic Mode Decomposition (Hankel-DMD). Partial noisy measurements are embedded into a Hankel matrix, while an associated Page matrix enables singular-value hard thresholding (SVHT) to estimate the effective rank. A Cadzow projection enforces structured low-rank consistency, yielding a denoised trajectory and local noise variance estimates. From this representation, a time-varying Hankel-DMD lifted linear predictor is constructed for multi-step forecasts. The residual analysis provides variance-tracking signals that can support downstream estimators and risk-aware planning. We validate the approach in simulation under Gaussian and heavy-tailed noise, and experimentally on a dynamic crane testbed. Results show that the method achieves stable variance-aware denoising and short-horizon prediction suitable for integration into real-time control frameworks.
>
---
#### [new 012] Runge-Kutta Approximations for Direct Coning Compensation Applying Lie Theory
- **分类: cs.RO**

- **简介: 该论文面向导航系统中的陀螺积分问题，提出基于Runge-Kutta方法的新型锥化补偿算法，利用李群理论直接构建高阶校正模型，可简化为经典算法并扩展至更高精度。**

- **链接: [http://arxiv.org/pdf/2511.00412v1](http://arxiv.org/pdf/2511.00412v1)**

> **作者:** John A. Christian; Michael R. Walker II; Wyatt Bridgman; Michael J. Sparapany
>
> **摘要:** The integration of gyroscope measurements is an essential task for most navigation systems. Modern vehicles typically use strapdown systems, such that gyro integration requires coning compensation to account for the sensor's rotation during the integration. Many coning compensation algorithms have been developed and a few are reviewed. This work introduces a new class of coning correction algorithm built directly from the classical Runge-Kutta integration routines. A simple case is shown to collapse to one of the most popular coning algorithms and a clear procedure for generating higher-order algorithms is presented.
>
---
#### [new 013] MARS: Multi-Agent Robotic System with Multimodal Large Language Models for Assistive Intelligence
- **分类: cs.RO; cs.CV; I.2.9; I.2.11; I.2.6; I.4.8**

- **简介: 论文提出MARS系统，利用多模态大语言模型构建四代理机器人系统，解决居家辅助中风险感知、个性化规划与动作落地难题，实现动态环境下的协同智能辅助。**

- **链接: [http://arxiv.org/pdf/2511.01594v1](http://arxiv.org/pdf/2511.01594v1)**

> **作者:** Renjun Gao; Peiyan Zhong
>
> **备注:** 3 figures, 1 table; under review at Multimedia Systems (Springer)
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable capabilities in cross-modal understanding and reasoning, offering new opportunities for intelligent assistive systems, yet existing systems still struggle with risk-aware planning, user personalization, and grounding language plans into executable skills in cluttered homes. We introduce MARS - a Multi-Agent Robotic System powered by MLLMs for assistive intelligence and designed for smart home robots supporting people with disabilities. The system integrates four agents: a visual perception agent for extracting semantic and spatial features from environment images, a risk assessment agent for identifying and prioritizing hazards, a planning agent for generating executable action sequences, and an evaluation agent for iterative optimization. By combining multimodal perception with hierarchical multi-agent decision-making, the framework enables adaptive, risk-aware, and personalized assistance in dynamic indoor environments. Experiments on multiple datasets demonstrate the superior overall performance of the proposed system in risk-aware planning and coordinated multi-agent execution compared with state-of-the-art multimodal models. The proposed approach also highlights the potential of collaborative AI for practical assistive scenarios and provides a generalizable methodology for deploying MLLM-enabled multi-agent systems in real-world environments.
>
---
#### [new 014] STRIDER: Navigation via Instruction-Aligned Structural Decision Space Optimization
- **分类: cs.RO; cs.AI**

- **简介: 论文针对零样本视觉-语言导航（VLN-CE）任务，提出STRIDER框架，通过结构化航点生成与任务对齐调节，优化决策空间，提升长程导航中空间结构与语义指令的对齐性，在R2R-CE和RxR-CE上显著提升成功率。**

- **链接: [http://arxiv.org/pdf/2511.00033v1](http://arxiv.org/pdf/2511.00033v1)**

> **作者:** Diqi He; Xuehao Gao; Hao Li; Junwei Han; Dingwen Zhang
>
> **摘要:** The Zero-shot Vision-and-Language Navigation in Continuous Environments (VLN-CE) task requires agents to navigate previously unseen 3D environments using natural language instructions, without any scene-specific training. A critical challenge in this setting lies in ensuring agents' actions align with both spatial structure and task intent over long-horizon execution. Existing methods often fail to achieve robust navigation due to a lack of structured decision-making and insufficient integration of feedback from previous actions. To address these challenges, we propose STRIDER (Instruction-Aligned Structural Decision Space Optimization), a novel framework that systematically optimizes the agent's decision space by integrating spatial layout priors and dynamic task feedback. Our approach introduces two key innovations: 1) a Structured Waypoint Generator that constrains the action space through spatial structure, and 2) a Task-Alignment Regulator that adjusts behavior based on task progress, ensuring semantic alignment throughout navigation. Extensive experiments on the R2R-CE and RxR-CE benchmarks demonstrate that STRIDER significantly outperforms strong SOTA across key metrics; in particular, it improves Success Rate (SR) from 29% to 35%, a relative gain of 20.7%. Such results highlight the importance of spatially constrained decision-making and feedback-guided execution in improving navigation fidelity for zero-shot VLN-CE.
>
---
#### [new 015] Descriptive Model-based Learning and Control for Bipedal Locomotion
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对双足机器人平衡控制问题，提出一种基于描述模型的控制方法，避免强制全模型拟合低维简化模型，仅约束低维平衡状态，让高维自由度自由演化，实现更高效、类人且鲁棒的行走。**

- **链接: [http://arxiv.org/pdf/2511.00512v1](http://arxiv.org/pdf/2511.00512v1)**

> **作者:** Suraj Kumar; Andy Ruina
>
> **备注:** 8 pages, 15 figures
>
> **摘要:** Bipedal balance is challenging due to its multi-phase, hybrid nature and high-dimensional state space. Traditional balance control approaches for bipedal robots rely on low-dimensional models for locomotion planning and reactive control, constraining the full robot to behave like these simplified models. This involves tracking preset reference paths for the Center of Mass and upper body obtained through low-dimensional models, often resulting in inefficient walking patterns with bent knees. However, we observe that bipedal balance is inherently low-dimensional and can be effectively described with simple state and action descriptors in a low-dimensional state space. This allows the robot's motion to evolve freely in its high-dimensional state space, only constraining its projection in the low-dimensional state space. In this work, we propose a novel control approach that avoids prescribing a low-dimensional model to the full model. Instead, our control framework uses a descriptive model with the minimum degrees of freedom necessary to maintain balance, allowing the remaining degrees of freedom to evolve freely in the high-dimensional space. This results in an efficient human-like walking gait and improved robustness.
>
---
#### [new 016] Don't Just Search, Understand: Semantic Path Planning Agent for Spherical Tensegrity Robots in Unknown Environments
- **分类: cs.RO**

- **简介: 该论文针对球形张力结构机器人在未知环境中的路径规划问题，提出基于大语言模型的语义规划代理SATPlanner，通过自适应观测窗口实现语义推理，显著缩小搜索空间并提升规划效率与成功率。**

- **链接: [http://arxiv.org/pdf/2511.01236v1](http://arxiv.org/pdf/2511.01236v1)**

> **作者:** Junwen Zhang; Changyue Liu; Pengqi Fu; Xiang Guo; Ye Shi; Xudong Liang; Zhijian Wang; Hanzhi Ma
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Endowed with inherent dynamical properties that grant them remarkable ruggedness and adaptability, spherical tensegrity robots stand as prototypical examples of hybrid softrigid designs and excellent mobile platforms. However, path planning for these robots in unknown environments presents a significant challenge, requiring a delicate balance between efficient exploration and robust planning. Traditional path planners, which treat the environment as a geometric grid, often suffer from redundant searches and are prone to failure in complex scenarios due to their lack of semantic understanding. To overcome these limitations, we reframe path planning in unknown environments as a semantic reasoning task. We introduce a Semantic Agent for Tensegrity robots (SATPlanner) driven by a Large Language Model (LLM). SATPlanner leverages high-level environmental comprehension to generate efficient and reliable planning strategies.At the core of SATPlanner is an Adaptive Observation Window mechanism, inspired by the "fast" and "slow" thinking paradigms of LLMs. This mechanism dynamically adjusts the perceptual field of the agent: it narrows for rapid traversal of open spaces and expands to reason about complex obstacle configurations. This allows the agent to construct a semantic belief of the environment, enabling the search space to grow only linearly with the path length (O(L)) while maintaining path quality. We extensively evaluate SATPlanner in 1,000 simulation trials, where it achieves a 100% success rate, outperforming other real-time planning algorithms. Critically, SATPlanner reduces the search space by 37.2% compared to the A* algorithm while achieving comparable, near-optimal path lengths. Finally, the practical feasibility of SATPlanner is validated on a physical spherical tensegrity robot prototype.
>
---
#### [new 017] High-Precision Surgical Robotic System for Intraocular Procedures
- **分类: cs.RO**

- **简介: 该论文设计了一种高精度眼科手术机器人，解决现有系统精度不足与工具交换不畅问题，通过校准与OCT引导，实现0.053mm定位精度，并完成深度学习辅助的白内障自动化手术。**

- **链接: [http://arxiv.org/pdf/2511.01232v1](http://arxiv.org/pdf/2511.01232v1)**

> **作者:** Yu-Ting Lai; Jacob Rosen; Yasamin Foroutani; Ji Ma; Wen-Cheng Wu; Jean-Pierre Hubschman; Tsu-Chin Tsao
>
> **摘要:** Despite the extensive demonstration of robotic systems for both cataract and vitreoretinal procedures, existing technologies or mechanisms still possess insufficient accuracy, precision, and degrees of freedom for instrument manipulation or potentially automated tool exchange during surgical procedures. A new robotic system that focuses on improving tooltip accuracy, tracking performance, and smooth instrument exchange mechanism is therefore designed and manufactured. Its tooltip accuracy, precision, and mechanical capability of maintaining small incision through remote center of motion were externally evaluated using an optical coherence tomography (OCT) system. Through robot calibration and precise coordinate registration, the accuracy of tooltip positioning was measured to be 0.053$\pm$0.031 mm, and the overall performance was demonstrated on an OCT-guided automated cataract lens extraction procedure with deep learning-based pre-operative anatomical modeling and real-time supervision.
>
---
#### [new 018] Embodied Cognition Augmented End2End Autonomous Driving
- **分类: cs.RO; cs.AI; cs.HC; 68T45**

- **简介: 该论文提出E³AD框架，首次将人类脑电认知引入端到端自动驾驶，通过视觉与EEG大模型的对比学习，增强驾驶规划能力，解决传统方法依赖监督标签、泛化性差的问题。**

- **链接: [http://arxiv.org/pdf/2511.01334v1](http://arxiv.org/pdf/2511.01334v1)**

> **作者:** Ling Niu; Xiaoji Zheng; Han Wang; Chen Zheng; Ziyuan Yang; Bokui Chen; Jiangtao Gong
>
> **备注:** 24 pages,4 pages
>
> **摘要:** In recent years, vision-based end-to-end autonomous driving has emerged as a new paradigm. However, popular end-to-end approaches typically rely on visual feature extraction networks trained under label supervision. This limited supervision framework restricts the generality and applicability of driving models. In this paper, we propose a novel paradigm termed $E^{3}AD$, which advocates for comparative learning between visual feature extraction networks and the general EEG large model, in order to learn latent human driving cognition for enhancing end-to-end planning. In this work, we collected a cognitive dataset for the mentioned contrastive learning process. Subsequently, we investigated the methods and potential mechanisms for enhancing end-to-end planning with human driving cognition, using popular driving models as baselines on publicly available autonomous driving datasets. Both open-loop and closed-loop tests are conducted for a comprehensive evaluation of planning performance. Experimental results demonstrate that the $E^{3}AD$ paradigm significantly enhances the end-to-end planning performance of baseline models. Ablation studies further validate the contribution of driving cognition and the effectiveness of comparative learning process. To the best of our knowledge, this is the first work to integrate human driving cognition for improving end-to-end autonomous driving planning. It represents an initial attempt to incorporate embodied cognitive data into end-to-end autonomous driving, providing valuable insights for future brain-inspired autonomous driving systems. Our code will be made available at Github
>
---
#### [new 019] MO-SeGMan: Rearrangement Planning Framework for Multi Objective Sequential and Guided Manipulation in Constrained Environments
- **分类: cs.RO; cs.AI**

- **简介: MO-SeGMan提出一种多目标序列引导操作规划框架，用于约束环境中的复杂重排任务，通过选择性引导搜索与自适应子目标优化，高效减少重规划与机器人移动，提升解的质量与效率。**

- **链接: [http://arxiv.org/pdf/2511.01476v1](http://arxiv.org/pdf/2511.01476v1)**

> **作者:** Cankut Bora Tuncer; Marc Toussaint; Ozgur S. Oguz
>
> **备注:** 8 pages, 8 figures, website:https://sites.google.com/view/mo-segman/
>
> **摘要:** In this work, we introduce MO-SeGMan, a Multi-Objective Sequential and Guided Manipulation planner for highly constrained rearrangement problems. MO-SeGMan generates object placement sequences that minimize both replanning per object and robot travel distance while preserving critical dependency structures with a lazy evaluation method. To address highly cluttered, non-monotone scenarios, we propose a Selective Guided Forward Search (SGFS) that efficiently relocates only critical obstacles and to feasible relocation points. Furthermore, we adopt a refinement method for adaptive subgoal selection to eliminate unnecessary pick-and-place actions, thereby improving overall solution quality. Extensive evaluations on nine benchmark rearrangement tasks demonstrate that MO-SeGMan generates feasible motion plans in all cases, consistently achieving faster solution times and superior solution quality compared to the baselines. These results highlight the robustness and scalability of the proposed framework for complex rearrangement planning problems.
>
---
#### [new 020] AquaROM: shape optimization pipeline for soft swimmers using parametric reduced order models
- **分类: cs.RO**

- **简介: 该论文提出AquaROM，一种基于张量参数化降阶模型的优化框架，用于高效优化受非线性水动力作用的软体游泳机器人形状，解决FEM仿真计算昂贵的问题，实现无数据快速精确优化。**

- **链接: [http://arxiv.org/pdf/2511.01031v1](http://arxiv.org/pdf/2511.01031v1)**

> **作者:** Mathieu Dubied; Paolo Tiso; Robert K. Katzschmann
>
> **摘要:** The efficient optimization of actuated soft structures, particularly under complex nonlinear forces, remains a critical challenge in advancing robotics. Simulations of nonlinear structures, such as soft-bodied robots modeled using the finite element method (FEM), often demand substantial computational resources, especially during optimization. To address this challenge, we propose a novel optimization algorithm based on a tensorial parametric reduced order model (PROM). Our algorithm leverages dimensionality reduction and solution approximation techniques to facilitate efficient solving of nonlinear constrained optimization problems. The well-structured tensorial approach enables the use of analytical gradients within a specifically chosen reduced order basis (ROB), significantly enhancing computational efficiency. To showcase the performance of our method, we apply it to optimizing soft robotic swimmer shapes. These actuated soft robots experience hydrodynamic forces, subjecting them to both internal and external nonlinear forces, which are incorporated into our optimization process using a data-free ROB for fast and accurate computations. This approach not only reduces computational complexity but also unlocks new opportunities to optimize complex nonlinear systems in soft robotics, paving the way for more efficient design and control.
>
---
#### [new 021] When Semantics Connect the Swarm: LLM-Driven Fuzzy Control for Cooperative Multi-Robot Underwater Coverage
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种LLM驱动的模糊控制框架，解决水下多机器人在无GPS、通信受限下的协同覆盖问题。通过语义令牌压缩感知信息，实现无需全局定位的导航与语义通信协同，提升探索效率与适应性。**

- **链接: [http://arxiv.org/pdf/2511.00783v1](http://arxiv.org/pdf/2511.00783v1)**

> **作者:** Jingzehua Xu; Weihang Zhang; Yangyang Li; Hongmiaoyi Zhang; Guanwen Xie; Jiwei Tang; Shuai Zhang; Yi Li
>
> **备注:** This paper has been submitted to IEEE Transactions on Mobile Computing
>
> **摘要:** Underwater multi-robot cooperative coverage remains challenging due to partial observability, limited communication, environmental uncertainty, and the lack of access to global localization. To address these issues, this paper presents a semantics-guided fuzzy control framework that couples Large Language Models (LLMs) with interpretable control and lightweight coordination. Raw multimodal observations are compressed by the LLM into compact, human-interpretable semantic tokens that summarize obstacles, unexplored regions, and Objects Of Interest (OOIs) under uncertain perception. A fuzzy inference system with pre-defined membership functions then maps these tokens into smooth and stable steering and gait commands, enabling reliable navigation without relying on global positioning. Then, we further coordinate multiple robots by introducing semantic communication that shares intent and local context in linguistic form, enabling agreement on who explores where while avoiding redundant revisits. Extensive simulations in unknown reef-like environments show that, under limited sensing and communication, the proposed framework achieves robust OOI-oriented navigation and cooperative coverage with improved efficiency and adaptability, narrowing the gap between semantic cognition and distributed underwater control in GPS-denied, map-free conditions.
>
---
#### [new 022] Tackling the Kidnapped Robot Problem via Sparse Feasible Hypothesis Sampling and Reliable Batched Multi-Stage Inference
- **分类: cs.RO**

- **简介: 该论文针对机器人被绑架后的重定位问题，提出一种基于稀疏可行假设采样与多阶段推理的全局重定位框架，利用SMAD与TAM指标提升非全景激光雷达在静态下的定位效率与精度。**

- **链接: [http://arxiv.org/pdf/2511.01219v1](http://arxiv.org/pdf/2511.01219v1)**

> **作者:** Muhua Zhang; Lei Ma; Ying Wu; Kai Shen; Deqing Huang; Henry Leung
>
> **备注:** 10 pages, 8 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper addresses the Kidnapped Robot Problem (KRP), a core localization challenge of relocalizing a robot in a known map without prior pose estimate when localization loss or at SLAM initialization. For this purpose, a passive 2-D global relocalization framework is proposed. It estimates the global pose efficiently and reliably from a single LiDAR scan and an occupancy grid map while the robot remains stationary, thereby enhancing the long-term autonomy of mobile robots. The proposed framework casts global relocalization as a non-convex problem and solves it via the multi-hypothesis scheme with batched multi-stage inference and early termination, balancing completeness and efficiency. The Rapidly-exploring Random Tree (RRT), under traversability constraints, asymptotically covers the reachable space to generate sparse, uniformly distributed feasible positional hypotheses, fundamentally reducing the sampling space. The hypotheses are preliminarily ordered by the proposed Scan Mean Absolute Difference (SMAD), a coarse beam-error level metric that facilitates the early termination by prioritizing high-likelihood candidates. The SMAD computation is optimized for non-panoramic scans. And the Translation-Affinity Scan-to-Map Alignment Metric (TAM) is proposed for reliable orientation selection at hypothesized positions and accurate final pose evaluation to mitigate degradation in conventional likelihood-field metrics under translational uncertainty induced by sparse hypotheses, as well as non-panoramic LiDAR scan and environmental changes. Real-world experiments on a resource-constrained mobile robot with non-panoramic LiDAR scan demonstrate that the proposed framework outperforms existing methods in both global relocalization success rate and computational efficiency.
>
---
#### [new 023] Thermo-responsive closing and reopening artificial Venus Flytrap utilizing shape memory elastomers
- **分类: cs.RO; physics.bio-ph**

- **简介: 该论文提出一种基于热响应形状记忆弹性体的仿生捕蝇草，首次实现温度触发的自主闭合与 reopening，解决传统人工捕蝇草仅能单向运动的问题，推动双向智能软机器人的发展。**

- **链接: [http://arxiv.org/pdf/2511.01346v1](http://arxiv.org/pdf/2511.01346v1)**

> **作者:** Shun Yoshida; Qingchuan Song; Bastian E. Rapp; Thomas Speck; Falk J. Tauber
>
> **备注:** Conference Proceedings Paper Living Machines 2025
>
> **摘要:** Despite their often perceived static and slow nature, some plants can move faster than the blink of an eye. The rapid snap closure motion of the Venus flytrap (Dionaea muscipula) has long captivated the interest of researchers and engineers alike, serving as a model for plant-inspired soft machines and robots. The translation of the fast snapping closure has inspired the development of various artificial Venus flytrap (AVF) systems. However, translating both the closing and reopening motion of D. muscipula into an autonomous plant inspired soft machine has yet to be achieved. In this study, we present an AVF that autonomously closes and reopens, utilizing novel thermo-responsive UV-curable shape memory materials for soft robotic systems. The life-sized thermo-responsive AVF exhibits closing and reopening motions triggered in a naturally occurring temperature range. The doubly curved trap lobes, built from shape memory polymers, close at 38{\deg}C, while reopening initiates around 45{\deg}C, employing shape memory elastomer strips as antagonistic actuators to facilitate lobe reopening. This work represents the first demonstration of thermo-responsive closing and reopening in an AVF with programmed sequential motion in response to increasing temperature. This approach marks the next step toward autonomously bidirectional moving soft machines/robots.
>
---
#### [new 024] Designing for Distributed Heterogeneous Modularity: On Software Architecture and Deployment of MoonBots
- **分类: cs.RO**

- **简介: 该论文提出一种分布式异构模块化软件架构，解决太空机器人系统在多节点、跨环境下的集成与协同难题，通过ROS2/Zenoh通信与开源Motion Stack实现动态重组与去中心化控制，提升可扩展性与维护性。**

- **链接: [http://arxiv.org/pdf/2511.01437v1](http://arxiv.org/pdf/2511.01437v1)**

> **作者:** Elian Neppel; Shamistan Karimov; Ashutosh Mishra; Gustavo Hernan Diaz Huenupan; Hazal Gozbasi; Kentaro Uno; Shreya Santra; Kazuya Yoshida
>
> **备注:** 6 pages, 8 figures. Accepted at ISPARO 2025
>
> **摘要:** This paper presents the software architecture and deployment strategy behind the MoonBot platform: a modular space robotic system composed of heterogeneous components distributed across multiple computers, networks and ultimately celestial bodies. We introduce a principled approach to distributed, heterogeneous modularity, extending modular robotics beyond physical reconfiguration to software, communication and orchestration. We detail the architecture of our system that integrates component-based design, a data-oriented communication model using ROS2 and Zenoh, and a deployment orchestrator capable of managing complex multi-module assemblies. These abstractions enable dynamic reconfiguration, decentralized control, and seamless collaboration between numerous operators and modules. At the heart of this system lies our open-source Motion Stack software, validated by months of field deployment with self-assembling robots, inter-robot cooperation, and remote operation. Our architecture tackles the significant hurdles of modular robotics by significantly reducing integration and maintenance overhead, while remaining scalable and robust. Although tested with space in mind, we propose generalizable patterns for designing robotic systems that must scale across time, hardware, teams and operational environments.
>
---
#### [new 025] Heuristic Step Planning for Learning Dynamic Bipedal Locomotion: A Comparative Study of Model-Based and Model-Free Approaches
- **分类: cs.RO**

- **简介: 该论文研究类人机器人动态双足行走，提出一种无需复杂动力学模型的启发式步态规划方法，通过Raibert控制器调节步长以跟踪躯干速度。与LIPM模型相比，该方法在速度精度、地形鲁棒性和能效上表现更优，证明模型自由方法足以实现稳定行走。**

- **链接: [http://arxiv.org/pdf/2511.00840v1](http://arxiv.org/pdf/2511.00840v1)**

> **作者:** William Suliman; Ekaterina Chaikovskaia; Egor Davydenko; Roman Gorbachev
>
> **摘要:** This work presents an extended framework for learning-based bipedal locomotion that incorporates a heuristic step-planning strategy guided by desired torso velocity tracking. The framework enables precise interaction between a humanoid robot and its environment, supporting tasks such as crossing gaps and accurately approaching target objects. Unlike approaches based on full or simplified dynamics, the proposed method avoids complex step planners and analytical models. Step planning is primarily driven by heuristic commands, while a Raibert-type controller modulates the foot placement length based on the error between desired and actual torso velocity. We compare our method with a model-based step-planning approach -- the Linear Inverted Pendulum Model (LIPM) controller. Experimental results demonstrate that our approach attains comparable or superior accuracy in maintaining target velocity (up to 80%), significantly greater robustness on uneven terrain (over 50% improvement), and improved energy efficiency. These results suggest that incorporating complex analytical, model-based components into the training architecture may be unnecessary for achieving stable and robust bipedal walking, even in unstructured environments.
>
---
#### [new 026] Phy-Tac: Toward Human-Like Grasping via Physics-Conditioned Tactile Goals
- **分类: cs.RO**

- **简介: 论文提出Phy-Tac，解决机器人抓取中力过大问题，通过物理引导的触觉目标实现力最优稳定抓取。融合几何选位、触觉预测与低力控制，显著提升抓取效率与类人性能。**

- **链接: [http://arxiv.org/pdf/2511.01520v1](http://arxiv.org/pdf/2511.01520v1)**

> **作者:** Shipeng Lyu; Lijie Sheng; Fangyuan Wang; Wenyao Zhang; Weiwei Lin; Zhenzhong Jia; David Navarro-Alarcon; Guodong Guo
>
> **备注:** 9 papges, 10 figures, 3 tables
>
> **摘要:** Humans naturally grasp objects with minimal level required force for stability, whereas robots often rely on rigid, over-squeezing control. To narrow this gap, we propose a human-inspired physics-conditioned tactile method (Phy-Tac) for force-optimal stable grasping (FOSG) that unifies pose selection, tactile prediction, and force regulation. A physics-based pose selector first identifies feasible contact regions with optimal force distribution based on surface geometry. Then, a physics-conditioned latent diffusion model (Phy-LDM) predicts the tactile imprint under FOSG target. Last, a latent-space LQR controller drives the gripper toward this tactile imprint with minimal actuation, preventing unnecessary compression. Trained on a physics-conditioned tactile dataset covering diverse objects and contact conditions, the proposed Phy-LDM achieves superior tactile prediction accuracy, while the Phy-Tac outperforms fixed-force and GraspNet-based baselines in grasp stability and force efficiency. Experiments on classical robotic platforms demonstrate force-efficient and adaptive manipulation that bridges the gap between robotic and human grasping.
>
---
#### [new 027] Deployable Vision-driven UAV River Navigation via Human-in-the-loop Preference Alignment
- **分类: cs.RO**

- **简介: 该论文提出SPAR-H方法，解决无人机视觉导航中仿真到现实的分布偏移与安全风险问题，通过人机交互的逐状态偏好对齐，实现数据高效、安全的在线策略优化。**

- **链接: [http://arxiv.org/pdf/2511.01083v1](http://arxiv.org/pdf/2511.01083v1)**

> **作者:** Zihan Wang; Jianwen Li; Li-Fan Wu; Nina Mahmoudian
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** Rivers are critical corridors for environmental monitoring and disaster response, where Unmanned Aerial Vehicles (UAVs) guided by vision-driven policies can provide fast, low-cost coverage. However, deployment exposes simulation-trained policies with distribution shift and safety risks and requires efficient adaptation from limited human interventions. We study human-in-the-loop (HITL) learning with a conservative overseer who vetoes unsafe or inefficient actions and provides statewise preferences by comparing the agent's proposal with a corrective override. We introduce Statewise Hybrid Preference Alignment for Robotics (SPAR-H), which fuses direct preference optimization on policy logits with a reward-based pathway that trains an immediate-reward estimator from the same preferences and updates the policy using a trust-region surrogate. With five HITL rollouts collected from a fixed novice policy, SPAR-H achieves the highest final episodic reward and the lowest variance across initial conditions among tested methods. The learned reward model aligns with human-preferred actions and elevates nearby non-intervened choices, supporting stable propagation of improvements. We benchmark SPAR-H against imitation learning (IL), direct preference variants, and evaluative reinforcement learning (RL) in the HITL setting, and demonstrate real-world feasibility of continual preference alignment for UAV river following. Overall, dual statewise preferences empirically provide a practical route to data-efficient online adaptation in riverine navigation.
>
---
#### [new 028] Design and Development of a Modular Bucket Drum Excavator for Lunar ISRU
- **分类: cs.RO**

- **简介: 该论文属于月球原位资源利用（ISRU）任务，旨在解决月壤高效挖掘问题。研究设计并试制了一种模块化桶式钻斗，通过沙箱测试验证其连续与批量挖掘效率，证实其轻量化、低能耗特性，并兼容MoonBot机器人平台。**

- **链接: [http://arxiv.org/pdf/2511.00492v1](http://arxiv.org/pdf/2511.00492v1)**

> **作者:** Simon Giel; James Hurrell; Shreya Santra; Ashutosh Mishra; Kentaro Uno; Kazuya Yoshida
>
> **备注:** 6 pages, 4 figures. Accepted at IEEE iSpaRo 2025
>
> **摘要:** In-Situ Resource Utilization (ISRU) is one of the key technologies for enabling sustainable access to the Moon. The ability to excavate lunar regolith is the first step in making lunar resources accessible and usable. This work presents the development of a bucket drum for the modular robotic system MoonBot, as part of the Japanese Moonshot program. A 3D-printed prototype made of PLA was manufactured to evaluate its efficiency through a series of sandbox tests. The resulting tool weighs 4.8 kg and has a volume of 14.06 L. It is capable of continuous excavation at a rate of 777.54 kg/h with a normalized energy consumption of 0.022 Wh/kg. In batch operation, the excavation rate is 172.02 kg/h with a normalized energy consumption of 0.86 Wh per kilogram of excavated material. The obtained results demonstrate the successful implementation of the concept. A key advantage of the developed tool is its compatibility with the modular MoonBot robotic platform, which enables flexible and efficient mission planning. Further improvements may include the integration of sensors and an autonomous control system to enhance the excavation process.
>
---
#### [new 029] Hybrid Neural Network-Based Indoor Localisation System for Mobile Robots Using CSI Data in a Robotics Simulator
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出一种基于CSI数据的混合神经网络（HyNN）模型，用于移动机器人室内定位。通过CNN与MLP融合，将CSI转换为图像进行2D位置估计，并集成ROS与仿真器验证，提升复杂环境下的定位精度与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.01797v1](http://arxiv.org/pdf/2511.01797v1)**

> **作者:** Javier Ballesteros-Jerez; Jesus Martínez-Gómez; Ismael García-Varea; Luis Orozco-Barbosa; Manuel Castillo-Cara
>
> **备注:** 13 pages, 7 figures. Conference paper (ROBOVIS 2025)
>
> **摘要:** We present a hybrid neural network model for inferring the position of mobile robots using Channel State Information (CSI) data from a Massive MIMO system. By leveraging an existing CSI dataset, our approach integrates a Convolutional Neural Network (CNN) with a Multilayer Perceptron (MLP) to form a Hybrid Neural Network (HyNN) that estimates 2D robot positions. CSI readings are converted into synthetic images using the TINTO tool. The localisation solution is integrated with a robotics simulator, and the Robot Operating System (ROS), which facilitates its evaluation through heterogeneous test cases, and the adoption of state estimators like Kalman filters. Our contributions illustrate the potential of our HyNN model in achieving precise indoor localisation and navigation for mobile robots in complex environments. The study follows, and proposes, a generalisable procedure applicable beyond the specific use case studied, making it adaptable to different scenarios and datasets.
>
---
#### [new 030] Improving Needle Penetration via Precise Rotational Insertion Using Iterative Learning Control
- **分类: cs.RO**

- **简介: 该论文针对机器人辅助视网膜注射中因关节错位导致的针具插入不精准问题，提出基于迭代学习控制（ILC）的旋转插入方法，通过OCT反馈迭代优化轨迹，显著提升穿透成功率与安全性。**

- **链接: [http://arxiv.org/pdf/2511.01256v1](http://arxiv.org/pdf/2511.01256v1)**

> **作者:** Yasamin Foroutani; Yasamin Mousavi-Motlagh; Aya Barzelay; Tsu-Chin Tsao
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Achieving precise control of robotic tool paths is often challenged by inherent system misalignments, unmodeled dynamics, and actuation inaccuracies. This work introduces an Iterative Learning Control (ILC) strategy to enable precise rotational insertion of a tool during robotic surgery, improving penetration efficacy and safety compared to straight insertion tested in subretinal injection. A 4 degree of freedom (DOF) robot manipulator is used, where misalignment of the fourth joint complicates the simple application of needle rotation, motivating an ILC approach that iteratively adjusts joint commands based on positional feedback. The process begins with calibrating the forward kinematics for the chosen surgical tool to achieve higher accuracy, followed by successive ILC iterations guided by Optical Coherence Tomography (OCT) volume scans to measure the error and refine control inputs. Experimental results, tested on subretinal injection tasks on ex vivo pig eyes, show that the optimized trajectory resulted in higher success rates in tissue penetration and subretinal injection compared to straight insertion, demonstrating the effectiveness of ILC in overcoming misalignment challenges. This approach offers potential applications for other high precision robot tasks requiring controlled insertions as well.
>
---
#### [new 031] Reducing Robotic Upper-Limb Assessment Time While Maintaining Precision: A Time Series Foundation Model Approach
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出用时间序列基础模型（如Chronos）预测机器人上肢评估中缺失的运动试验，将40-64次试验缩短至8次，同时保持运动参数可靠性（ICC≥0.90），实现高效中风患者运动功能评估。**

- **链接: [http://arxiv.org/pdf/2511.00193v1](http://arxiv.org/pdf/2511.00193v1)**

> **作者:** Faranak Akbarifar; Nooshin Maghsoodi; Sean P Dukelow; Stephen Scott; Parvin Mousavi
>
> **摘要:** Purpose: Visually Guided Reaching (VGR) on the Kinarm robot yields sensitive kinematic biomarkers but requires 40-64 reaches, imposing time and fatigue burdens. We evaluate whether time-series foundation models can replace unrecorded trials from an early subset of reaches while preserving the reliability of standard Kinarm parameters. Methods: We analyzed VGR speed signals from 461 stroke and 599 control participants across 4- and 8-target reaching protocols. We withheld all but the first 8 or 16 reaching trials and used ARIMA, MOMENT, and Chronos models, fine-tuned on 70 percent of subjects, to forecast synthetic trials. We recomputed four kinematic features of reaching (reaction time, movement time, posture speed, maximum speed) on combined recorded plus forecasted trials and compared them to full-length references using ICC(2,1). Results: Chronos forecasts restored ICC >= 0.90 for all parameters with only 8 recorded trials plus forecasts, matching the reliability of 24-28 recorded reaches (Delta ICC <= 0.07). MOMENT yielded intermediate gains, while ARIMA improvements were minimal. Across cohorts and protocols, synthetic trials replaced reaches without materially compromising feature reliability. Conclusion: Foundation-model forecasting can greatly shorten Kinarm VGR assessment time. For the most impaired stroke survivors, sessions drop from 4-5 minutes to about 1 minute while preserving kinematic precision. This forecast-augmented paradigm promises efficient robotic evaluations for assessing motor impairments following stroke.
>
---
#### [new 032] FoldPath: End-to-End Object-Centric Motion Generation via Modulated Implicit Paths
- **分类: cs.RO; cs.AI**

- **简介: FoldPath提出一种端到端的神经场方法，用于物体中心运动生成（OCMG），将机器人轨迹建模为连续函数，替代传统离散点预测与繁琐后处理，仅用70个专家样本即可在工业场景中生成高精度平滑路径。**

- **链接: [http://arxiv.org/pdf/2511.01407v1](http://arxiv.org/pdf/2511.01407v1)**

> **作者:** Paolo Rabino; Gabriele Tiboni; Tatiana Tommasi
>
> **备注:** Accepted at 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Object-Centric Motion Generation (OCMG) is instrumental in advancing automated manufacturing processes, particularly in domains requiring high-precision expert robotic motions, such as spray painting and welding. To realize effective automation, robust algorithms are essential for generating extended, object-aware trajectories across intricate 3D geometries. However, contemporary OCMG techniques are either based on ad-hoc heuristics or employ learning-based pipelines that are still reliant on sensitive post-processing steps to generate executable paths. We introduce FoldPath, a novel, end-to-end, neural field based method for OCMG. Unlike prior deep learning approaches that predict discrete sequences of end-effector waypoints, FoldPath learns the robot motion as a continuous function, thus implicitly encoding smooth output paths. This paradigm shift eliminates the need for brittle post-processing steps that concatenate and order the predicted discrete waypoints. Particularly, our approach demonstrates superior predictive performance compared to recently proposed learning-based methods, and attains generalization capabilities even in real industrial settings, where only a limited amount of 70 expert samples are provided. We validate FoldPath through comprehensive experiments in a realistic simulation environment and introduce new, rigorous metrics designed to comprehensively evaluate long-horizon robotic paths, thus advancing the OCMG task towards practical maturity.
>
---
#### [new 033] A High-Speed Capable Spherical Robot
- **分类: cs.RO; cs.SY; eess.SY; I.2.9**

- **简介: 该论文设计了一种新型球形机器人，通过引入与次摆轴对齐的动量轮，解决原结构无法稳定高速运动的问题，实现最高10 m/s的高速稳定行驶，并提升越障与地形适应能力。**

- **链接: [http://arxiv.org/pdf/2511.01288v1](http://arxiv.org/pdf/2511.01288v1)**

> **作者:** Bixuan Zhang; Fengqi Zhang; Haojie Chen; You Wang; Jie Hao; Zhiyuan Luo; Guang Li
>
> **备注:** 5 pages
>
> **摘要:** This paper designs a new spherical robot structure capable of supporting high-speed motion at up to 10 m/s. Building upon a single-pendulum-driven spherical robot, the design incorporates a momentum wheel with an axis aligned with the secondary pendulum, creating a novel spherical robot structure. Practical experiments with the physical prototype have demonstrated that this new spherical robot can achieve stable high-speed motion through simple decoupled control, which was unattainable with the original structure. The spherical robot designed for high-speed motion not only increases speed but also significantly enhances obstacle-crossing performance and terrain robustness.
>
---
#### [new 034] MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出MOBIUS多模态双足机器人，解决复杂城市环境中稳定移动与操作难题，通过四肢体结构与混合控制架构，实现行走、爬行、攀爬与滚动的无缝切换，集成规划与控制提升能效与安全性。**

- **链接: [http://arxiv.org/pdf/2511.01774v1](http://arxiv.org/pdf/2511.01774v1)**

> **作者:** Alexander Schperberg; Yusuke Tanaka; Stefano Di Cairano; Dennis Hong
>
> **备注:** 23 pages, 20 figures. Collaborative work between the Robotics and Mechanisms Laboratory (RoMeLa) and Mitsubishi Electric Research Laboratories (MERL)
>
> **摘要:** This article presents a Multi-Modal Bipedal Intelligent Urban Scout robot (MOBIUS) capable of walking, crawling, climbing, and rolling. MOBIUS features four limbs--two 6-DoF arms with two-finger grippers for manipulation and climbing, and two 4-DoF legs for locomotion--enabling smooth transitions across diverse terrains without reconfiguration. A hybrid control architecture combines reinforcement learning-based locomotion with model-based predictive and admittance control enhanced for safety by a Reference Governor toward compliant contact interactions. A high-level MIQCP planner autonomously selects locomotion modes to balance stability and energy efficiency. Hardware experiments demonstrate robust gait transitions, dynamic climbing, and full-body load support via pinch grasp. Overall, MOBIUS demonstrates the importance of tight integration between morphology, high-level planning, and control to enable mobile loco-manipulation and grasping, substantially expanding its interaction capabilities, workspace, and traversability.
>
---
#### [new 035] Lateral Velocity Model for Vehicle Parking Applications
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶泊车中横向速度估计不准的问题，基于实车数据发现零侧滑假设的偏差，提出一种仅需两个参数的横向速度模型，显著提升低速泊车场景下的定位精度，适用于消费级车辆。**

- **链接: [http://arxiv.org/pdf/2511.01369v1](http://arxiv.org/pdf/2511.01369v1)**

> **作者:** Luis Diener; Jens Kalkkuhl; Markus Enzweiler
>
> **备注:** This manuscript has been submitted to Vehicle System Dynamics for possible publication
>
> **摘要:** Automated parking requires accurate localization for quick and precise maneuvering in tight spaces. While the longitudinal velocity can be measured using wheel encoders, the estimation of the lateral velocity remains a key challenge due to the absence of dedicated sensors in consumer-grade vehicles. Existing approaches often rely on simplified vehicle models, such as the zero-slip model, which assumes no lateral velocity at the rear axle. It is well established that this assumption does not hold during low-speed driving and researchers thus introduce additional heuristics to account for differences. In this work, we analyze real-world data from parking scenarios and identify a systematic deviation from the zero-slip assumption. We provide explanations for the observed effects and then propose a lateral velocity model that better captures the lateral dynamics of the vehicle during parking. The model improves estimation accuracy, while relying on only two parameters, making it well-suited for integration into consumer-grade applications.
>
---
#### [new 036] Design and Fabrication of Origami-Inspired Knitted Fabrics for Soft Robotics
- **分类: cs.RO**

- **简介: 该论文提出一种折纸启发的针织面料设计方法，解决软机器人中结构完整性与柔顺性难以兼顾的问题。通过编程针法与热熔纱线，实现定向折叠与形变控制，成功构建可穿戴的Miura-ori等折纸机器人。**

- **链接: [http://arxiv.org/pdf/2511.01272v1](http://arxiv.org/pdf/2511.01272v1)**

> **作者:** Sehui Jeong; Magaly C. Aviles; Athena X. Naylor; Cynthia Sung; Allison M. Okamura
>
> **摘要:** Soft robots employing compliant materials and deformable structures offer great potential for wearable devices that are comfortable and safe for human interaction. However, achieving both structural integrity and compliance for comfort remains a significant challenge. In this study, we present a novel fabrication and design method that combines the advantages of origami structures with the material programmability and wearability of knitted fabrics. We introduce a general design method that translates origami patterns into knit designs by programming both stitch and material patterns. The method creates folds in preferred directions while suppressing unintended buckling and bending by selectively incorporating heat fusible yarn to create rigid panels around compliant creases. We experimentally quantify folding moments and show that stitch patterning enhances folding directionality while the heat fusible yarn (1) keeps geometry consistent by reducing edge curl and (2) prevents out-of-plane deformations by stiffening panels. We demonstrate the framework through the successful reproduction of complex origami tessellations, including Miura-ori, Yoshimura, and Kresling patterns, and present a wearable knitted Kaleidocycle robot capable of locomotion. The combination of structural reconfigurability, material programmability, and potential for manufacturing scalability highlights knitted origami as a promising platform for next-generation wearable robotics.
>
---
#### [new 037] Digital Twin based Automatic Reconfiguration of Robotic Systems in Smart Environments
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文提出基于数字孪生的机器人系统自动重配置方法，解决动态环境中传统控制难以快速适应的问题。通过虚拟环境仿真优化轨迹与参数，实现无人干预的实时自适应控制，提升机器人在智能环境中的自主性。**

- **链接: [http://arxiv.org/pdf/2511.00094v1](http://arxiv.org/pdf/2511.00094v1)**

> **作者:** Angelos Alexopoulos; Agorakis Bompotas; Nikitas Rigas Kalogeropoulos; Panagiotis Kechagias; Athanasios P. Kalogeras; Christos Alexakos
>
> **备注:** Accepted for presentation to 11th IEEE International Smart Cities Conference (ISC2 2025)
>
> **摘要:** Robotic systems have become integral to smart environments, enabling applications ranging from urban surveillance and automated agriculture to industrial automation. However, their effective operation in dynamic settings - such as smart cities and precision farming - is challenged by continuously evolving topographies and environmental conditions. Traditional control systems often struggle to adapt quickly, leading to inefficiencies or operational failures. To address this limitation, we propose a novel framework for autonomous and dynamic reconfiguration of robotic controllers using Digital Twin technology. Our approach leverages a virtual replica of the robot's operational environment to simulate and optimize movement trajectories in response to real-world changes. By recalculating paths and control parameters in the Digital Twin and deploying the updated code to the physical robot, our method ensures rapid and reliable adaptation without manual intervention. This work advances the integration of Digital Twins in robotics, offering a scalable solution for enhancing autonomy in smart, dynamic environments.
>
---
#### [new 038] Scaling Cross-Embodiment World Models for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文提出通过共享的粒子状态与动作表示，构建跨形态的统一世界模型，解决不同机器人形态间动作迁移困难的问题。结合模拟与真实数据训练，实现跨自由度灵巧操作的泛化控制。**

- **链接: [http://arxiv.org/pdf/2511.01177v1](http://arxiv.org/pdf/2511.01177v1)**

> **作者:** Zihao He; Bo Ai; Tongzhou Mu; Yulin Liu; Weikang Wan; Jiawei Fu; Yilun Du; Henrik I. Christensen; Hao Su
>
> **摘要:** Cross-embodiment learning seeks to build generalist robots that operate across diverse morphologies, but differences in action spaces and kinematics hinder data sharing and policy transfer. This raises a central question: Is there any invariance that allows actions to transfer across embodiments? We conjecture that environment dynamics are embodiment-invariant, and that world models capturing these dynamics can provide a unified interface across embodiments. To learn such a unified world model, the crucial step is to design state and action representations that abstract away embodiment-specific details while preserving control relevance. To this end, we represent different embodiments (e.g., human hands and robot hands) as sets of 3D particles and define actions as particle displacements, creating a shared representation for heterogeneous data and control problems. A graph-based world model is then trained on exploration data from diverse simulated robot hands and real human hands, and integrated with model-based planning for deployment on novel hardware. Experiments on rigid and deformable manipulation tasks reveal three findings: (i) scaling to more training embodiments improves generalization to unseen ones, (ii) co-training on both simulated and real data outperforms training on either alone, and (iii) the learned models enable effective control on robots with varied degrees of freedom. These results establish world models as a promising interface for cross-embodiment dexterous manipulation.
>
---
#### [new 039] An Enhanced Proprioceptive Method for Soft Robots Integrating Bend Sensors and IMUs
- **分类: cs.RO**

- **简介: 该论文提出一种融合IMU与弯折传感器的增强本体感知方法，用于软体机器人形变估计，通过卡尔曼滤波互补校正IMU漂移，实现高精度、长时稳定的状态感知，显著提升定位准确性。**

- **链接: [http://arxiv.org/pdf/2511.01165v1](http://arxiv.org/pdf/2511.01165v1)**

> **作者:** Dong Heon Han; Mayank Mehta; Runze Zuo; Zachary Wanger; Daniel Bruder
>
> **摘要:** This study presents an enhanced proprioceptive method for accurate shape estimation of soft robots using only off-the-shelf sensors, ensuring cost-effectiveness and easy applicability. By integrating inertial measurement units (IMUs) with complementary bend sensors, IMU drift is mitigated, enabling reliable long-term proprioception. A Kalman filter fuses segment tip orientations from both sensors in a mutually compensatory manner, improving shape estimation over single-sensor methods. A piecewise constant curvature model estimates the tip location from the fused orientation data and reconstructs the robot's deformation. Experiments under no loading, external forces, and passive obstacle interactions during 45 minutes of continuous operation showed a root mean square error of 16.96 mm (2.91% of total length), a 56% reduction compared to IMU-only benchmarks. These results demonstrate that our approach not only enables long-duration proprioception in soft robots but also maintains high accuracy and robustness across these diverse conditions.
>
---
#### [new 040] End-to-End Dexterous Arm-Hand VLA Policies via Shared Autonomy: VR Teleoperation Augmented by Autonomous Hand VLA Policy for Efficient Data Collection
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出一种共享自主框架，结合VR人操臂部与自主手部VLA策略，高效采集高质量臂手协同数据，训练端到端VLA策略，解决机器人灵巧操作数据稀缺与自然性不足问题，实现90%成功率。**

- **链接: [http://arxiv.org/pdf/2511.00139v1](http://arxiv.org/pdf/2511.00139v1)**

> **作者:** Yu Cui; Yujian Zhang; Lina Tao; Yang Li; Xinyu Yi; Zhibin Li
>
> **摘要:** Achieving human-like dexterous manipulation remains a major challenge for general-purpose robots. While Vision-Language-Action (VLA) models show potential in learning skills from demonstrations, their scalability is limited by scarce high-quality training data. Existing data collection methods face inherent constraints: manual teleoperation overloads human operators, while automated planning often produces unnatural motions. We propose a Shared Autonomy framework that divides control between macro and micro motions. A human operator guides the robot's arm pose through intuitive VR teleoperation, while an autonomous DexGrasp-VLA policy handles fine-grained hand control using real-time tactile and visual feedback. This division significantly reduces cognitive load and enables efficient collection of high-quality coordinated arm-hand demonstrations. Using this data, we train an end-to-end VLA policy enhanced with our novel Arm-Hand Feature Enhancement module, which captures both distinct and shared representations of macro and micro movements for more natural coordination. Our Corrective Teleoperation system enables continuous policy improvement through human-in-the-loop failure recovery. Experiments demonstrate that our framework generates high-quality data with minimal manpower and achieves a 90% success rate across diverse objects, including unseen instances. Comprehensive evaluations validate the system's effectiveness in developing dexterous manipulation capabilities.
>
---
#### [new 041] Breaking the Latency Barrier: Synergistic Perception and Control for High-Frequency 3D Ultrasound Servoing
- **分类: cs.RO**

- **简介: 该论文针对机器人超声系统中高频率动态目标跟踪的延迟瓶颈，提出感知与控制协同设计框架，通过双流感知网络和单步流策略，实现60Hz以上闭环控制，精准跟踪高速运动目标并验证了临床有效性。**

- **链接: [http://arxiv.org/pdf/2511.00983v1](http://arxiv.org/pdf/2511.00983v1)**

> **作者:** Yizhao Qian; Yujie Zhu; Jiayuan Luo; Li Liu; Yixuan Yuan; Guochen Ning; Hongen Liao
>
> **摘要:** Real-time tracking of dynamic targets amidst large-scale, high-frequency disturbances remains a critical unsolved challenge in Robotic Ultrasound Systems (RUSS), primarily due to the end-to-end latency of existing systems. This paper argues that breaking this latency barrier requires a fundamental shift towards the synergistic co-design of perception and control. We realize it in a novel framework with two tightly-coupled contributions: (1) a Decoupled Dual-Stream Perception Network that robustly estimates 3D translational state from 2D images at high frequency, and (2) a Single-Step Flow Policy that generates entire action sequences in one inference pass, bypassing the iterative bottleneck of conventional policies. This synergy enables a closed-loop control frequency exceeding 60Hz. On a dynamic phantom, our system not only tracks complex 3D trajectories with a mean error below 6.5mm but also demonstrates robust re-acquisition from over 170mm displacement. Furthermore, it can track targets at speeds of 102mm/s, achieving a terminal error below 1.7mm. Moreover, in-vivo experiments on a human volunteer validate the framework's effectiveness and robustness in a realistic clinical setting. Our work presents a RUSS holistically architected to unify high-bandwidth tracking with large-scale repositioning, a critical step towards robust autonomy in dynamic clinical environments.
>
---
#### [new 042] Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出Alpamayo-R1，一种融合因果推理与轨迹规划的视觉-语言-动作模型，解决自动驾驶长尾场景中决策脆弱问题，通过因果数据集、模块化架构与多阶段训练，显著提升规划准确性与安全性。**

- **链接: [http://arxiv.org/pdf/2511.00088v1](http://arxiv.org/pdf/2511.00088v1)**

> **作者:** NVIDIA; :; Yan Wang; Wenjie Luo; Junjie Bai; Yulong Cao; Tong Che; Ke Chen; Yuxiao Chen; Jenna Diamond; Yifan Ding; Wenhao Ding; Liang Feng; Greg Heinrich; Jack Huang; Peter Karkus; Boyi Li; Pinyi Li; Tsung-Yi Lin; Dongran Liu; Ming-Yu Liu; Langechuan Liu; Zhijian Liu; Jason Lu; Yunxiang Mao; Pavlo Molchanov; Lindsey Pavao; Zhenghao Peng; Mike Ranzinger; Ed Schmerling; Shida Shen; Yunfei Shi; Sarah Tariq; Ran Tian; Tilman Wekel; Xinshuo Weng; Tianjun Xiao; Eric Yang; Xiaodong Yang; Yurong You; Xiaohui Zeng; Wenyuan Zhang; Boris Ivanovic; Marco Pavone
>
> **摘要:** End-to-end architectures trained via imitation learning have advanced autonomous driving by scaling model size and data, yet performance remains brittle in safety-critical long-tail scenarios where supervision is sparse and causal understanding is limited. To address this, we introduce Alpamayo-R1 (AR1), a vision-language-action model (VLA) that integrates Chain of Causation reasoning with trajectory planning to enhance decision-making in complex driving scenarios. Our approach features three key innovations: (1) the Chain of Causation (CoC) dataset, built through a hybrid auto-labeling and human-in-the-loop pipeline producing decision-grounded, causally linked reasoning traces aligned with driving behaviors; (2) a modular VLA architecture combining Cosmos-Reason, a Vision-Language Model pre-trained for Physical AI applications, with a diffusion-based trajectory decoder that generates dynamically feasible plans in real time; (3) a multi-stage training strategy using supervised fine-tuning to elicit reasoning and reinforcement learning (RL) to optimize reasoning quality via large reasoning model feedback and enforce reasoning-action consistency. Evaluation shows AR1 achieves up to a 12% improvement in planning accuracy on challenging cases compared to a trajectory-only baseline, with a 35% reduction in off-road rate and 25% reduction in close encounter rate in closed-loop simulation. RL post-training improves reasoning quality by 45% as measured by a large reasoning model critic and reasoning-action consistency by 37%. Model scaling from 0.5B to 7B parameters shows consistent improvements. On-vehicle road tests confirm real-time performance (99 ms latency) and successful urban deployment. By bridging interpretable reasoning with precise control, AR1 demonstrates a practical path towards Level 4 autonomous driving. We plan to release AR1 models and a subset of the CoC in a future update.
>
---
#### [new 043] Adaptive and Multi-object Grasping via Deformable Origami Modules
- **分类: cs.RO**

- **简介: 该论文提出一种基于可变形折纸模块的多指软体夹爪，实现无需传感与反馈的自适应多目标抓取，解决传统夹爪依赖复杂控制与传感的问题，提升复杂场景下多对象并行操作效率。**

- **链接: [http://arxiv.org/pdf/2511.00516v1](http://arxiv.org/pdf/2511.00516v1)**

> **作者:** Peiyi Wang; Paul A. M. Lefeuvre; Shangwei Zou; Zhenwei Ni; Daniela Rus; Cecilia Laschi
>
> **摘要:** Soft robotics gripper have shown great promise in handling fragile and geometrically complex objects. However, most existing solutions rely on bulky actuators, complex control strategies, or advanced tactile sensing to achieve stable and reliable grasping performance. In this work, we present a multi-finger hybrid gripper featuring passively deformable origami modules that generate constant force and torque output. Each finger composed of parallel origami modules is driven by a 1-DoF actuator mechanism, enabling passive shape adaptability and stable grasping force without active sensing or feedback control. More importantly, we demonstrate an interesting capability in simultaneous multi-object grasping, which allows stacked objects of varied shape and size to be picked, transported and placed independently at different states, significantly improving manipulation efficiency compared to single-object grasping. These results highlight the potential of origami-based compliant structures as scalable modules for adaptive, stable and efficient multi-object manipulation in domestic and industrial pick-and-place scenarios.
>
---
#### [new 044] SonarSweep: Fusing Sonar and Vision for Robust 3D Reconstruction via Plane Sweeping
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: SonarSweep提出一种端到端深度学习框架，融合声呐与视觉数据，解决水下视觉退化环境中的3D重建难题，通过改进平面扫描算法实现高精度稠密深度图重建。**

- **链接: [http://arxiv.org/pdf/2511.00392v1](http://arxiv.org/pdf/2511.00392v1)**

> **作者:** Lingpeng Chen; Jiakun Tang; Apple Pui-Yi Chui; Ziyang Hong; Junfeng Wu
>
> **备注:** 8 pages, 9 figures, conference
>
> **摘要:** Accurate 3D reconstruction in visually-degraded underwater environments remains a formidable challenge. Single-modality approaches are insufficient: vision-based methods fail due to poor visibility and geometric constraints, while sonar is crippled by inherent elevation ambiguity and low resolution. Consequently, prior fusion technique relies on heuristics and flawed geometric assumptions, leading to significant artifacts and an inability to model complex scenes. In this paper, we introduce SonarSweep, a novel, end-to-end deep learning framework that overcomes these limitations by adapting the principled plane sweep algorithm for cross-modal fusion between sonar and visual data. Extensive experiments in both high-fidelity simulation and real-world environments demonstrate that SonarSweep consistently generates dense and accurate depth maps, significantly outperforming state-of-the-art methods across challenging conditions, particularly in high turbidity. To foster further research, we will publicly release our code and a novel dataset featuring synchronized stereo-camera and sonar data, the first of its kind.
>
---
#### [new 045] Design and development of an electronics-free earthworm robot
- **分类: cs.RO**

- **简介: 该论文提出一种无电子元件的仿蚯蚓软体机器人，通过改进气动逻辑门（PLG）实现自主蠕动运动，解决传统气动机器人依赖笨重电子控制系统的问题，验证了纯气动控制的可行性，适用于危险环境中的无缆运动。**

- **链接: [http://arxiv.org/pdf/2511.01347v1](http://arxiv.org/pdf/2511.01347v1)**

> **作者:** Riddhi Das; Joscha Teichmann; Thomas Speck; Falk J. Tauber
>
> **备注:** Conference Proceedings Paper Living Machines 2025
>
> **摘要:** Soft robotic systems have gained widespread attention due to their inherent flexibility, adaptability, and safety, making them well-suited for varied applications. Among bioinspired designs, earthworm locomotion has been extensively studied for its efficient peristaltic motion, enabling movement in confined and unstructured environments. Existing earthworm-inspired robots primarily utilize pneumatic actuation due to its high force-to-weight ratio and ease of implementation. However, these systems often rely on bulky, power-intensive electronic control units, limiting their practicality. In this work, we present an electronics-free, earthworm-inspired pneumatic robot utilizing a modified Pneumatic Logic Gate (PLG) design. By integrating preconfigured PLG units with bellow actuators, we achieved a plug-and-play style modular system capable of peristaltic locomotion without external electronic components. The proposed design reduces system complexity while maintaining efficient actuation. We characterize the bellow actuators under different operating conditions and evaluate the robots locomotion performance. Our findings demonstrate that the modified PLG-based control system effectively generates peristaltic wave propagation, achieving autonomous motion with minimal deviation. This study serves as a proof of concept for the development of electronics-free, peristaltic soft robots. The proposed system has potential for applications in hazardous environments, where untethered, adaptable locomotion is critical. Future work will focus on further optimizing the robot design and exploring untethered operation using onboard compressed air sources.
>
---
#### [new 046] Closed-loop Control of Steerable Balloon Endoscopes for Robot-assisted Transcatheter Intracardiac Procedures
- **分类: cs.RO**

- **简介: 该论文面向机器人辅助经导管心脏手术，解决传统成像与导航精度不足问题，设计了一种可充气平衡式内窥镜，通过气压独立控制视野与弯曲角度，并实现基于图像的闭环控制，提升工具定位稳定性。**

- **链接: [http://arxiv.org/pdf/2511.01199v1](http://arxiv.org/pdf/2511.01199v1)**

> **作者:** Max McCandless; Jonathan Hamid; Sammy Elmariah; Nathaniel Langer; Pierre E. Dupont
>
> **备注:** 8 pages, 11 figures
>
> **摘要:** To move away from open-heart surgery towards safer transcatheter procedures, there is a growing need for improved imaging techniques and robotic solutions to enable simple, accurate tool navigation. Common imaging modalities, such as fluoroscopy and ultrasound, have limitations that can be overcome using cardioscopy, i.e., direct optical visualization inside the beating heart. We present a cardioscope designed as a steerable balloon. As a balloon, it can be collapsed to pass through the vasculature and subsequently inflated inside the heart for visualization and tool delivery through an integrated working channel. Through careful design of balloon wall thickness, a single input, balloon inflation pressure, is used to independently control two outputs, balloon diameter (corresponding to field of view diameter) and balloon bending angle (enabling precise working channel positioning). This balloon technology can be tuned to produce cardioscopes designed for a range of intracardiac tasks. To illustrate this approach, a balloon design is presented for the specific task of aortic leaflet laceration. Image-based closed-loop control of bending angle is also demonstrated as a means of enabling stable orientation control during tool insertion and removal.
>
---
#### [new 047] RobustVLA: Robustness-Aware Reinforcement Post-Training for Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出RobustVLA，面向视觉-语言-动作模型的鲁棒性强化学习后训练任务，解决其在噪声与扰动下泛化差的问题，通过雅可比与平滑正则化提升模型抗干扰能力，显著增强部署可靠性。**

- **链接: [http://arxiv.org/pdf/2511.01331v1](http://arxiv.org/pdf/2511.01331v1)**

> **作者:** Hongyin Zhang; Shuo Zhang; Junxi Jin; Qixin Zeng; Runze Li; Donglin Wang
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as powerful general-purpose policies for robotic manipulation, benefiting from large-scale multi-modal pre-training. However, they often fail to generalize reliably in out-of-distribution deployments, where unavoidable disturbances such as observation noise, sensor errors, or actuation perturbations become prevalent. While recent Reinforcement Learning (RL)-based post-training provides a practical means to adapt pre-trained VLA models, existing methods mainly emphasize reward maximization and overlook robustness to environmental uncertainty. In this work, we introduce RobustVLA, a lightweight online RL post-training method designed to explicitly enhance the resilience of VLA models. Through a systematic robustness analysis, we identify two key regularizations: Jacobian regularization, which mitigates sensitivity to observation noise, and smoothness regularization, which stabilizes policies under action perturbations. Extensive experiments across diverse robotic environments demonstrate that RobustVLA significantly outperforms prior state-of-the-art methods in robustness and reliability. Our results highlight the importance of principled robustness-aware RL post-training as a key step toward improving the reliability and robustness of VLA models.
>
---
#### [new 048] Tailored robotic training improves hand function and proprioceptive processing in stroke survivors with proprioceptive deficits: A randomized controlled trial
- **分类: cs.RO; cs.ET; cs.HC**

- **简介: 该论文属神经康复研究，旨在解决中风后本体感觉缺陷导致的手部功能障碍。通过机器人辅助的两种个性化训练（Propriopixel与Virtual Assistance），显著提升患者手功能与本体感觉神经处理能力，验证了精准康复的有效性。**

- **链接: [http://arxiv.org/pdf/2511.00259v1](http://arxiv.org/pdf/2511.00259v1)**

> **作者:** Andria J. Farrens; Luis Garcia-Fernandez; Raymond Diaz Rojas; Jillian Obeso Estrada; Dylan Reinsdorf; Vicky Chan; Disha Gupta; Joel Perry; Eric Wolbrecht; An Do; Steven C. Cramer; David J. Reinkensmeyer
>
> **备注:** Main manuscript: 38 pages (double spaced, with references), 6 figures, 2 tables and collated supplemental materials (17 pages, double spaced)
>
> **摘要:** Precision rehabilitation aims to tailor movement training to improve outcomes. We tested whether proprioceptively-tailored robotic training improves hand function and neural processing in stroke survivors. Using a robotic finger exoskeleton, we tested two proprioceptively-tailored approaches: Propriopixel Training, which uses robot-facilitated, gamified movements to enhance proprioceptive processing, and Virtual Assistance Training, which reduces robotic aid to increase reliance on self-generated feedback. In a randomized controlled trial, forty-six chronic stroke survivors completed nine 2-hour sessions of Standard, Propriopixel or Virtual training. Among participants with proprioceptive deficits, Propriopixel ((Box and Block Test: 7 +/- 4.2, p=0.002) and Virtual Assistance (4.5 +/- 4.4 , p=0.068) yielded greater gains in hand function (Standard: 0.8 +/- 2.3 blocks). Proprioceptive gains correlated with improvements in hand function. Tailored training enhanced neural sensitivity to proprioceptive cues, evidenced by a novel EEG biomarker, the proprioceptive Contingent Negative Variation. These findings support proprioceptively-tailored training as a pathway to precision neurorehabilitation.
>
---
#### [new 049] Contact Map Transfer with Conditional Diffusion Model for Generalizable Dexterous Grasp Generation
- **分类: cs.RO**

- **简介: 该论文针对机器人灵巧抓取中泛化性差的问题，提出基于条件扩散模型的接触图迁移框架，通过三重物体中心图联合生成与双映射机制，实现跨未知物体的高效稳定抓取生成。**

- **链接: [http://arxiv.org/pdf/2511.01276v1](http://arxiv.org/pdf/2511.01276v1)**

> **作者:** Yiyao Ma; Kai Chen; Kexin Zheng; Qi Dou
>
> **摘要:** Dexterous grasp generation is a fundamental challenge in robotics, requiring both grasp stability and adaptability across diverse objects and tasks. Analytical methods ensure stable grasps but are inefficient and lack task adaptability, while generative approaches improve efficiency and task integration but generalize poorly to unseen objects and tasks due to data limitations. In this paper, we propose a transfer-based framework for dexterous grasp generation, leveraging a conditional diffusion model to transfer high-quality grasps from shape templates to novel objects within the same category. Specifically, we reformulate the grasp transfer problem as the generation of an object contact map, incorporating object shape similarity and task specifications into the diffusion process. To handle complex shape variations, we introduce a dual mapping mechanism, capturing intricate geometric relationship between shape templates and novel objects. Beyond the contact map, we derive two additional object-centric maps, the part map and direction map, to encode finer contact details for more stable grasps. We then develop a cascaded conditional diffusion model framework to jointly transfer these three maps, ensuring their intra-consistency. Finally, we introduce a robust grasp recovery mechanism, identifying reliable contact points and optimizing grasp configurations efficiently. Extensive experiments demonstrate the superiority of our proposed method. Our approach effectively balances grasp quality, generation efficiency, and generalization performance across various tasks. Project homepage: https://cmtdiffusion.github.io/
>
---
#### [new 050] LiDAR-VGGT: Cross-Modal Coarse-to-Fine Fusion for Globally Consistent and Metric-Scale Dense Mapping
- **分类: cs.RO; cs.CV**

- **简介: 论文提出LiDAR-VGGT框架，解决LiDAR与视觉模型在大尺度稠密建图中尺度不一致和标定敏感问题，通过粗到细跨模态融合实现全局一致、度量尺度的彩色点云重建。**

- **链接: [http://arxiv.org/pdf/2511.01186v1](http://arxiv.org/pdf/2511.01186v1)**

> **作者:** Lijie Wang; Lianjie Guo; Ziyi Xu; Qianhao Wang; Fei Gao; Xieyuanli Chen
>
> **摘要:** Reconstructing large-scale colored point clouds is an important task in robotics, supporting perception, navigation, and scene understanding. Despite advances in LiDAR inertial visual odometry (LIVO), its performance remains highly sensitive to extrinsic calibration. Meanwhile, 3D vision foundation models, such as VGGT, suffer from limited scalability in large environments and inherently lack metric scale. To overcome these limitations, we propose LiDAR-VGGT, a novel framework that tightly couples LiDAR inertial odometry with the state-of-the-art VGGT model through a two-stage coarse- to-fine fusion pipeline: First, a pre-fusion module with robust initialization refinement efficiently estimates VGGT poses and point clouds with coarse metric scale within each session. Then, a post-fusion module enhances cross-modal 3D similarity transformation, using bounding-box-based regularization to reduce scale distortions caused by inconsistent FOVs between LiDAR and camera sensors. Extensive experiments across multiple datasets demonstrate that LiDAR-VGGT achieves dense, globally consistent colored point clouds and outperforms both VGGT-based methods and LIVO baselines. The implementation of our proposed novel color point cloud evaluation toolkit will be released as open source.
>
---
#### [new 051] CM-LIUW-Odometry: Robust and High-Precision LiDAR-Inertial-UWB-Wheel Odometry for Extreme Degradation Coal Mine Tunnels
- **分类: cs.RO**

- **简介: 该论文提出CM-LIUW-Odometry，面向极端退化煤矿隧道的SLAM任务，融合LiDAR、IMU、UWB与轮式里程计，通过IESKF紧耦合与自适应模式切换，提升定位精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.01379v1](http://arxiv.org/pdf/2511.01379v1)**

> **作者:** Kun Hu; Menggang Li; Zhiwen Jin; Chaoquan Tang; Eryi Hu; Gongbo Zhou
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) in large-scale, complex, and GPS-denied underground coal mine environments presents significant challenges. Sensors must contend with abnormal operating conditions: GPS unavailability impedes scene reconstruction and absolute geographic referencing, uneven or slippery terrain degrades wheel odometer accuracy, and long, feature-poor tunnels reduce LiDAR effectiveness. To address these issues, we propose CoalMine-LiDAR-IMU-UWB-Wheel-Odometry (CM-LIUW-Odometry), a multimodal SLAM framework based on the Iterated Error-State Kalman Filter (IESKF). First, LiDAR-inertial odometry is tightly fused with UWB absolute positioning constraints to align the SLAM system with a global coordinate. Next, wheel odometer is integrated through tight coupling, enhanced by nonholonomic constraints (NHC) and vehicle lever arm compensation, to address performance degradation in areas beyond UWB measurement range. Finally, an adaptive motion mode switching mechanism dynamically adjusts the robot's motion mode based on UWB measurement range and environmental degradation levels. Experimental results validate that our method achieves superior accuracy and robustness in real-world underground coal mine scenarios, outperforming state-of-the-art approaches. We open source our code of this work on Github to benefit the robotics community.
>
---
#### [new 052] Floor Plan-Guided Visual Navigation Incorporating Depth and Directional Cues
- **分类: cs.RO**

- **简介: 该论文研究基于RGB图像与平面图的室内视觉导航任务，解决视觉与空间信息融合难、定位不准问题。提出GlocDiff扩散模型，融合平面图全局路径与深度特征局部感知，提升导航精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2511.01493v1](http://arxiv.org/pdf/2511.01493v1)**

> **作者:** Wei Huang; Jiaxin Li; Zang Wan; Huijun Di; Wei Liang; Zhu Yang
>
> **摘要:** Guiding an agent to a specific target in indoor environments based solely on RGB inputs and a floor plan is a promising yet challenging problem. Although existing methods have made significant progress, two challenges remain unresolved. First, the modality gap between egocentric RGB observations and the floor plan hinders the integration of visual and spatial information for both local obstacle avoidance and global planning. Second, accurate localization is critical for navigation performance, but remains challenging at deployment in unseen environments due to the lack of explicit geometric alignment between RGB inputs and floor plans. We propose a novel diffusion-based policy, denoted as GlocDiff, which integrates global path planning from the floor plan with local depth-aware features derived from RGB observations. The floor plan offers explicit global guidance, while the depth features provide implicit geometric cues, collectively enabling precise prediction of optimal navigation directions and robust obstacle avoidance. Moreover, GlocDiff introduces noise perturbation during training to enhance robustness against pose estimation errors, and we find that combining this with a relatively stable VO module during inference results in significantly improved navigation performance. Extensive experiments on the FloNa benchmark demonstrate GlocDiff's efficiency and effectiveness in achieving superior navigation performance, and the success of real-world deployments also highlights its potential for widespread practical applications.
>
---
#### [new 053] GenDexHand: Generative Simulation for Dexterous Hands
- **分类: cs.RO; cs.AI**

- **简介: GenDexHand提出一种生成式仿真框架，解决灵巧手任务数据稀缺问题，通过VLM闭环优化环境设计并分解子任务，实现高效、多样化的灵巧操作训练。**

- **链接: [http://arxiv.org/pdf/2511.01791v1](http://arxiv.org/pdf/2511.01791v1)**

> **作者:** Feng Chen; Zhuxiu Xu; Tianzhe Chu; Xunzhe Zhou; Li Sun; Zewen Wu; Shenghua Gao; Zhongyu Li; Yanchao Yang; Yi Ma
>
> **摘要:** Data scarcity remains a fundamental bottleneck for embodied intelligence. Existing approaches use large language models (LLMs) to automate gripper-based simulation generation, but they transfer poorly to dexterous manipulation, which demands more specialized environment design. Meanwhile, dexterous manipulation tasks are inherently more difficult due to their higher degrees of freedom. Massively generating feasible and trainable dexterous hand tasks remains an open challenge. To this end, we present GenDexHand, a generative simulation pipeline that autonomously produces diverse robotic tasks and environments for dexterous manipulation. GenDexHand introduces a closed-loop refinement process that adjusts object placements and scales based on vision-language model (VLM) feedback, substantially improving the average quality of generated environments. Each task is further decomposed into sub-tasks to enable sequential reinforcement learning, reducing training time and increasing success rates. Our work provides a viable path toward scalable training of diverse dexterous hand behaviors in embodied intelligence by offering a simulation-based solution to synthetic data generation. Our website: https://winniechen2002.github.io/GenDexHand/.
>
---
#### [new 054] Fast-SmartWay: Panoramic-Free End-to-End Zero-Shot Vision-and-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: Fast-SmartWay提出一种无需全景图与分步预测的端到端零样本视觉-语言导航方法，仅用三帧前视RGB-D图像与自然语言指令，结合不确定性推理模块，实现低延迟、高鲁棒的实时导航。**

- **链接: [http://arxiv.org/pdf/2511.00933v1](http://arxiv.org/pdf/2511.00933v1)**

> **作者:** Xiangyu Shi; Zerui Li; Yanyuan Qiao; Qi Wu
>
> **摘要:** Recent advances in Vision-and-Language Navigation in Continuous Environments (VLN-CE) have leveraged multimodal large language models (MLLMs) to achieve zero-shot navigation. However, existing methods often rely on panoramic observations and two-stage pipelines involving waypoint predictors, which introduce significant latency and limit real-world applicability. In this work, we propose Fast-SmartWay, an end-to-end zero-shot VLN-CE framework that eliminates the need for panoramic views and waypoint predictors. Our approach uses only three frontal RGB-D images combined with natural language instructions, enabling MLLMs to directly predict actions. To enhance decision robustness, we introduce an Uncertainty-Aware Reasoning module that integrates (i) a Disambiguation Module for avoiding local optima, and (ii) a Future-Past Bidirectional Reasoning mechanism for globally coherent planning. Experiments on both simulated and real-robot environments demonstrate that our method significantly reduces per-step latency while achieving competitive or superior performance compared to panoramic-view baselines. These results demonstrate the practicality and effectiveness of Fast-SmartWay for real-world zero-shot embodied navigation.
>
---
#### [new 055] Kinematify: Open-Vocabulary Synthesis of High-DoF Articulated Objects
- **分类: cs.RO; cs.CV**

- **简介: Kinematify提出一种从RGB图像或文本自动生成高自由度关节物体动力学模型的方法，解决传统方法依赖运动序列与人工数据的可扩展性问题，结合MCTS与几何优化实现拓扑与参数联合推理。**

- **链接: [http://arxiv.org/pdf/2511.01294v1](http://arxiv.org/pdf/2511.01294v1)**

> **作者:** Jiawei Wang; Dingyou Wang; Jiaming Hu; Qixuan Zhang; Jingyi Yu; Lan Xu
>
> **摘要:** A deep understanding of kinematic structures and movable components is essential for enabling robots to manipulate objects and model their own articulated forms. Such understanding is captured through articulated objects, which are essential for tasks such as physical simulation, motion planning, and policy learning. However, creating these models, particularly for complex systems like robots or objects with high degrees of freedom (DoF), remains a significant challenge. Existing methods typically rely on motion sequences or strong assumptions from hand-curated datasets, which hinders scalability. In this paper, we introduce Kinematify, an automated framework that synthesizes articulated objects directly from arbitrary RGB images or text prompts. Our method addresses two core challenges: (i) inferring kinematic topologies for high-DoF objects and (ii) estimating joint parameters from static geometry. To achieve this, we combine MCTS search for structural inference with geometry-driven optimization for joint reasoning, producing physically consistent and functionally valid descriptions. We evaluate Kinematify on diverse inputs from both synthetic and real-world environments, demonstrating improvements in registration and kinematic topology accuracy over prior work.
>
---
#### [new 056] Unified Diffusion VLA: Vision-Language-Action Model via Joint Discrete Denoising Diffusion Process
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Unified Diffusion VLA，通过联合离散去噪扩散过程（JD3P）统一视觉、语言与动作模态，实现理解、生成与执行的同步优化，解决多模态协同不足问题，在多个基准上实现SOTA性能且推理快4倍。**

- **链接: [http://arxiv.org/pdf/2511.01718v1](http://arxiv.org/pdf/2511.01718v1)**

> **作者:** Jiayi Chen; Wenxuan Song; Pengxiang Ding; Ziyang Zhou; Han Zhao; Feilong Tang; Donglin Wang; Haoang Li
>
> **摘要:** Vision-language-action (VLA) models aim to understand natural language instructions and visual observations and to execute corresponding actions as an embodied agent. Recent work integrates future images into the understanding-acting loop, yielding unified VLAs that jointly understand, generate, and act -- reading text and images and producing future images and actions. However, these models either rely on external experts for modality unification or treat image generation and action prediction as separate processes, limiting the benefits of direct synergy between these tasks. Our core philosophy is to optimize generation and action jointly through a synchronous denoising process, where the iterative refinement enables actions to evolve from initialization, under constant and sufficient visual guidance. We ground this philosophy in our proposed Unified Diffusion VLA and Joint Discrete Denoising Diffusion Process (JD3P), which is a joint diffusion process that integrates multiple modalities into a single denoising trajectory to serve as the key mechanism enabling understanding, generation, and acting to be intrinsically synergistic. Our model and theory are built on a unified tokenized space of all modalities and a hybrid attention mechanism. We further propose a two-stage training pipeline and several inference-time techniques that optimize performance and efficiency. Our approach achieves state-of-the-art performance on benchmarks such as CALVIN, LIBERO, and SimplerEnv with 4$\times$ faster inference than autoregressive methods, and we demonstrate its effectiveness through in-depth analysis and real-world evaluations. Our project page is available at https://irpn-eai.github.io/UD-VLA.github.io/.
>
---
#### [new 057] Improving Robustness to Out-of-Distribution States in Imitation Learning via Deep Koopman-Boosted Diffusion Policy
- **分类: cs.RO**

- **简介: 该论文针对模仿学习中对分布外状态鲁棒性不足的问题，提出D3P算法，通过双分支架构解耦视觉与本体感知输入，并引入深Koopman算子建模时序动态，结合生成模型置信度引导动作聚合，提升机器人操作的恢复与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.00555v1](http://arxiv.org/pdf/2511.00555v1)**

> **作者:** Dianye Huang; Nassir Navab; Zhongliang Jiang
>
> **备注:** Accepted by IEEE T-RO
>
> **摘要:** Integrating generative models with action chunking has shown significant promise in imitation learning for robotic manipulation. However, the existing diffusion-based paradigm often struggles to capture strong temporal dependencies across multiple steps, particularly when incorporating proprioceptive input. This limitation can lead to task failures, where the policy overfits to proprioceptive cues at the expense of capturing the visually derived features of the task. To overcome this challenge, we propose the Deep Koopman-boosted Dual-branch Diffusion Policy (D3P) algorithm. D3P introduces a dual-branch architecture to decouple the roles of different sensory modality combinations. The visual branch encodes the visual observations to indicate task progression, while the fused branch integrates both visual and proprioceptive inputs for precise manipulation. Within this architecture, when the robot fails to accomplish intermediate goals, such as grasping a drawer handle, the policy can dynamically switch to execute action chunks generated by the visual branch, allowing recovery to previously observed states and facilitating retrial of the task. To further enhance visual representation learning, we incorporate a Deep Koopman Operator module that captures structured temporal dynamics from visual inputs. During inference, we use the test-time loss of the generative model as a confidence signal to guide the aggregation of the temporally overlapping predicted action chunks, thereby enhancing the reliability of policy execution. In simulation experiments across six RLBench tabletop tasks, D3P outperforms the state-of-the-art diffusion policy by an average of 14.6\%. On three real-world robotic manipulation tasks, it achieves a 15.0\% improvement. Code: https://github.com/dianyeHuang/D3P.
>
---
#### [new 058] Multi-Mapcher: Loop Closure Detection-Free Heterogeneous LiDAR Multi-Session SLAM Leveraging Outlier-Robust Registration for Autonomous Vehicles
- **分类: cs.RO**

- **简介: 论文提出Multi-Mapcher，解决异构LiDAR多会话SLAM中环路检测不可靠问题，摒弃传统环路检测，采用鲁棒点云配准实现大尺度地图对齐，再结合锚节点优化构建全局一致地图，性能更优且更快。**

- **链接: [http://arxiv.org/pdf/2511.00635v1](http://arxiv.org/pdf/2511.00635v1)**

> **作者:** Hyungtae Lim; Daebeom Kim; Hyun Myung
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** As various 3D light detection and ranging (LiDAR) sensors have been introduced to the market, research on multi-session simultaneous localization and mapping (MSS) using heterogeneous LiDAR sensors has been actively conducted. Existing MSS methods mostly rely on loop closure detection for inter-session alignment; however, the performance of loop closure detection can be potentially degraded owing to the differences in the density and field of view (FoV) of the sensors used in different sessions. In this study, we challenge the existing paradigm that relies heavily on loop detection modules and propose a novel MSS framework, called Multi-Mapcher, that employs large-scale map-to-map registration to perform inter-session initial alignment, which is commonly assumed to be infeasible, by leveraging outlier-robust 3D point cloud registration. Next, after finding inter-session loops by radius search based on the assumption that the inter-session initial alignment is sufficiently precise, anchor node-based robust pose graph optimization is employed to build a consistent global map. As demonstrated in our experiments, our approach shows substantially better MSS performance for various LiDAR sensors used to capture the sessions and is faster than state-of-the-art approaches. Our code is available at https://github.com/url-kaist/multi-mapcher.
>
---
#### [new 059] EgoMI: Learning Active Vision and Whole-Body Manipulation from Egocentric Human Demonstrations
- **分类: cs.RO**

- **简介: 论文提出EgoMI框架，从人眼视角演示中学习手眼协同操作，解决机器人因视角动态变化导致的模仿学习分布偏移问题，通过记忆增强策略建模主动头部运动，实现半人形机器人高效迁移。**

- **链接: [http://arxiv.org/pdf/2511.00153v1](http://arxiv.org/pdf/2511.00153v1)**

> **作者:** Justin Yu; Yide Shentu; Di Wu; Pieter Abbeel; Ken Goldberg; Philipp Wu
>
> **摘要:** Imitation learning from human demonstrations offers a promising approach for robot skill acquisition, but egocentric human data introduces fundamental challenges due to the embodiment gap. During manipulation, humans actively coordinate head and hand movements, continuously reposition their viewpoint and use pre-action visual fixation search strategies to locate relevant objects. These behaviors create dynamic, task-driven head motions that static robot sensing systems cannot replicate, leading to a significant distribution shift that degrades policy performance. We present EgoMI (Egocentric Manipulation Interface), a framework that captures synchronized end-effector and active head trajectories during manipulation tasks, resulting in data that can be retargeted to compatible semi-humanoid robot embodiments. To handle rapid and wide-spanning head viewpoint changes, we introduce a memory-augmented policy that selectively incorporates historical observations. We evaluate our approach on a bimanual robot equipped with an actuated camera head and find that policies with explicit head-motion modeling consistently outperform baseline methods. Results suggest that coordinated hand-eye learning with EgoMI effectively bridges the human-robot embodiment gap for robust imitation learning on semi-humanoid embodiments. Project page: https://egocentric-manipulation-interface.github.io
>
---
#### [new 060] Endowing GPT-4 with a Humanoid Body: Building the Bridge Between Off-the-Shelf VLMs and the Physical World
- **分类: cs.RO; cs.AI**

- **简介: 论文提出BiBo框架，将GPT-4等现成视觉语言模型用于控制人形机器人，通过指令编译器将高层指令转为低层控制命令，再用扩散模型生成自适应运动，解决机器人在开放环境中泛化能力弱、数据依赖高的问题。**

- **链接: [http://arxiv.org/pdf/2511.00041v1](http://arxiv.org/pdf/2511.00041v1)**

> **作者:** Yingzhao Jian; Zhongan Wang; Yi Yang; Hehe Fan
>
> **摘要:** Humanoid agents often struggle to handle flexible and diverse interactions in open environments. A common solution is to collect massive datasets to train a highly capable model, but this approach can be prohibitively expensive. In this paper, we explore an alternative solution: empowering off-the-shelf Vision-Language Models (VLMs, such as GPT-4) to control humanoid agents, thereby leveraging their strong open-world generalization to mitigate the need for extensive data collection. To this end, we present \textbf{BiBo} (\textbf{B}uilding humano\textbf{I}d agent \textbf{B}y \textbf{O}ff-the-shelf VLMs). It consists of two key components: (1) an \textbf{embodied instruction compiler}, which enables the VLM to perceive the environment and precisely translate high-level user instructions (e.g., {\small\itshape ``have a rest''}) into low-level primitive commands with control parameters (e.g., {\small\itshape ``sit casually, location: (1, 2), facing: 90$^\circ$''}); and (2) a diffusion-based \textbf{motion executor}, which generates human-like motions from these commands, while dynamically adapting to physical feedback from the environment. In this way, BiBo is capable of handling not only basic interactions but also diverse and complex motions. Experiments demonstrate that BiBo achieves an interaction task success rate of 90.2\% in open environments, and improves the precision of text-guided motion execution by 16.3\% over prior methods. The code will be made publicly available.
>
---
#### [new 061] Gen AI in Automotive: Applications, Challenges, and Opportunities with a Case study on In-Vehicle Experience
- **分类: cs.RO**

- **简介: 该论文综述生成式AI在汽车领域的应用与挑战，聚焦语音人机交互，通过奔驰MBUX案例展示其在个性化车载体验中的优势，旨在推动更安全、用户中心的智能汽车发展。**

- **链接: [http://arxiv.org/pdf/2511.00026v1](http://arxiv.org/pdf/2511.00026v1)**

> **作者:** Chaitanya Shinde; Divya Garikapati
>
> **摘要:** Generative Artificial Intelligence is emerging as a transformative force in the automotive industry, enabling novel applications across vehicle design, manufacturing, autonomous driving, predictive maintenance, and in vehicle user experience. This paper provides a comprehensive review of the current state of GenAI in automotive, highlighting enabling technologies such as Generative Adversarial Networks and Variational Autoencoders. Key opportunities include accelerating autonomous driving validation through synthetic data generation, optimizing component design, and enhancing human machine interaction via personalized and adaptive interfaces. At the same time, the paper identifies significant technical, ethical, and safety challenges, including computational demands, bias, intellectual property concerns, and adversarial robustness, that must be addressed for responsible deployment. A case study on Mercedes Benzs MBUX Virtual Assistant illustrates how GenAI powered voice systems deliver more natural, proactive, and personalized in car interactions compared to legacy rule based assistants. Through this review and case study, the paper outlines both the promise and limitations of GenAI integration in the automotive sector and presents directions for future research and development aimed at achieving safer, more efficient, and user centric mobility. Unlike prior reviews that focus solely on perception or manufacturing, this paper emphasizes generative AI in voice based HMI, bridging safety and user experience perspectives.
>
---
#### [new 062] Lightweight Learning from Actuation-Space Demonstrations via Flow Matching for Whole-Body Soft Robotic Grasping
- **分类: cs.RO**

- **简介: 该论文面向软体机器人全身体抓取任务，解决传统控制依赖复杂反馈的问题，提出基于流匹配的轻量级动作空间学习框架，仅用30次演示即可实现97.5%抓取成功率，无需密集传感或重型控制器。**

- **链接: [http://arxiv.org/pdf/2511.01770v1](http://arxiv.org/pdf/2511.01770v1)**

> **作者:** Liudi Yang; Yang Bai; Yuhao Wang; Ibrahim Alsarraj; Gitta Kutyniok; Zhanchi Wang; Ke Wu
>
> **摘要:** Robotic grasping under uncertainty remains a fundamental challenge due to its uncertain and contact-rich nature. Traditional rigid robotic hands, with limited degrees of freedom and compliance, rely on complex model-based and heavy feedback controllers to manage such interactions. Soft robots, by contrast, exhibit embodied mechanical intelligence: their underactuated structures and passive flexibility of their whole body, naturally accommodate uncertain contacts and enable adaptive behaviors. To harness this capability, we propose a lightweight actuation-space learning framework that infers distributional control representations for whole-body soft robotic grasping, directly from deterministic demonstrations using a flow matching model (Rectified Flow),without requiring dense sensing or heavy control loops. Using only 30 demonstrations (less than 8% of the reachable workspace), the learned policy achieves a 97.5% grasp success rate across the whole workspace, generalizes to grasped-object size variations of +-33%, and maintains stable performance when the robot's dynamic response is directly adjusted by scaling the execution time from 20% to 200%. These results demonstrate that actuation-space learning, by leveraging its passive redundant DOFs and flexibility, converts the body's mechanics into functional control intelligence and substantially reduces the burden on central controllers for this uncertain-rich task.
>
---
#### [new 063] PixelVLA: Advancing Pixel-level Understanding in Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 论文提出PixelVLA，首个支持像素级理解与多模态提示的视觉-语言-动作模型，解决现有模型像素推理弱、依赖文本提示的问题，通过新架构与Pixel-160K数据集，在低训练成本下显著提升机器人操控精度。**

- **链接: [http://arxiv.org/pdf/2511.01571v1](http://arxiv.org/pdf/2511.01571v1)**

> **作者:** Wenqi Liang; Gan Sun; Yao He; Jiahua Dong; Suyan Dai; Ivan Laptev; Salman Khan; Yang Cong
>
> **备注:** 17pages,7 figures, 5 tabels
>
> **摘要:** Vision-Language-Action models (VLAs) are emerging as powerful tools for learning generalizable visuomotor control policies. However, current VLAs are mostly trained on large-scale image-text-action data and remain limited in two key ways: (i) they struggle with pixel-level scene understanding, and (ii) they rely heavily on textual prompts, which reduces their flexibility in real-world settings. To address these challenges, we introduce PixelVLA, the first VLA model designed to support both pixel-level reasoning and multimodal prompting with text and visual inputs. Our approach is built on a new visuomotor instruction tuning framework that integrates a multiscale pixel-aware encoder with a visual prompting encoder. To train PixelVLA effectively, we further propose a two-stage automated annotation pipeline that generates Pixel-160K, a large-scale dataset with pixel-level annotations derived from existing robot data. Experiments on three standard VLA benchmarks and two VLA model variants show that PixelVLA improves manipulation success rates by 10.1%-17.8% over OpenVLA, while requiring only 1.5% of its pretraining cost. These results demonstrate that PixelVLA can be integrated into existing VLAs to enable more accurate, efficient, and versatile robot control in complex environments. The dataset and code will be released as open source.
>
---
#### [new 064] Self-Improving Vision-Language-Action Models with Data Generation via Residual RL
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对视觉-语言-动作模型（VLA）依赖人工数据的瓶颈，提出PLD框架：通过残差强化学习探查失败区域，生成分布对齐的恢复轨迹，并蒸馏回原模型，实现自提升，显著提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2511.00091v1](http://arxiv.org/pdf/2511.00091v1)**

> **作者:** Wenli Xiao; Haotian Lin; Andy Peng; Haoru Xue; Tairan He; Yuqi Xie; Fengyuan Hu; Jimmy Wu; Zhengyi Luo; Linxi "Jim" Fan; Guanya Shi; Yuke Zhu
>
> **备注:** 26 pages
>
> **摘要:** Supervised fine-tuning (SFT) has become the de facto post-training strategy for large vision-language-action (VLA) models, but its reliance on costly human demonstrations limits scalability and generalization. We propose Probe, Learn, Distill (PLD), a three-stage plug-and-play framework that improves VLAs through residual reinforcement learning (RL) and distribution-aware data collection. In Stage 1, we train lightweight residual actors to probe failure regions of the VLA generalist. In Stage 2, we use a hybrid rollout scheme that aligns collected trajectories with the generalist's deployment distribution while capturing recovery behaviors. In Stage 3, we distill the curated trajectories back into the generalist with standard SFT. PLD achieves near-saturated 99% task success on LIBERO, over 50% gains in SimplerEnv, and 100% success on real-world Franka and YAM arm manipulation tasks. Ablations show that residual probing and distribution-aware replay are key to collecting deployment-aligned data that improves both seen and unseen tasks, offering a scalable path toward self-improving VLA models.
>
---
#### [new 065] Lyapunov Stability Learning with Nonlinear Control via Inductive Biases
- **分类: cs.LG; cs.RO**

- **简介: 该论文旨在通过将李雅普诺夫条件作为归纳偏置，提升神经网络学习控制李雅普诺夫函数（CLF）的稳定性与收敛性，实现CLF与控制器的端到端训练，解决传统方法约束复杂、收敛难、ROA小的问题。**

- **链接: [http://arxiv.org/pdf/2511.01283v1](http://arxiv.org/pdf/2511.01283v1)**

> **作者:** Yupu Lu; Shijie Lin; Hao Xu; Zeqing Zhang; Jia Pan
>
> **备注:** Accepted by IEEE Robio 2025
>
> **摘要:** Finding a control Lyapunov function (CLF) in a dynamical system with a controller is an effective way to guarantee stability, which is a crucial issue in safety-concerned applications. Recently, deep learning models representing CLFs have been applied into a learner-verifier framework to identify satisfiable candidates. However, the learner treats Lyapunov conditions as complex constraints for optimisation, which is hard to achieve global convergence. It is also too complicated to implement these Lyapunov conditions for verification. To improve this framework, we treat Lyapunov conditions as inductive biases and design a neural CLF and a CLF-based controller guided by this knowledge. This design enables a stable optimisation process with limited constraints, and allows end-to-end learning of both the CLF and the controller. Our approach achieves a higher convergence rate and larger region of attraction (ROA) in learning the CLF compared to existing methods among abundant experiment cases. We also thoroughly reveal why the success rate decreases with previous methods during learning.
>
---
#### [new 066] Discriminately Treating Motion Components Evolves Joint Depth and Ego-Motion Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DiMoDE框架，针对深度与自运动联合学习中运动成分被混同处理的问题，通过区分刚性运动分量并引入几何约束，提升估计鲁棒性，在多个数据集上实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.01502v1](http://arxiv.org/pdf/2511.01502v1)**

> **作者:** Mengtan Zhang; Zizhan Guo; Hongbo Zhao; Yi Feng; Zuyi Xiong; Yue Wang; Shaoyi Du; Hanli Wang; Rui Fan
>
> **备注:** 18 pages, 14 figures
>
> **摘要:** Unsupervised learning of depth and ego-motion, two fundamental 3D perception tasks, has made significant strides in recent years. However, most methods treat ego-motion as an auxiliary task, either mixing all motion types or excluding depth-independent rotational motions in supervision. Such designs limit the incorporation of strong geometric constraints, reducing reliability and robustness under diverse conditions. This study introduces a discriminative treatment of motion components, leveraging the geometric regularities of their respective rigid flows to benefit both depth and ego-motion estimation. Given consecutive video frames, network outputs first align the optical axes and imaging planes of the source and target cameras. Optical flows between frames are transformed through these alignments, and deviations are quantified to impose geometric constraints individually on each ego-motion component, enabling more targeted refinement. These alignments further reformulate the joint learning process into coaxial and coplanar forms, where depth and each translation component can be mutually derived through closed-form geometric relationships, introducing complementary constraints that improve depth robustness. DiMoDE, a general depth and ego-motion joint learning framework incorporating these designs, achieves state-of-the-art performance on multiple public datasets and a newly collected diverse real-world dataset, particularly under challenging conditions. Our source code will be publicly available at mias.group/DiMoDE upon publication.
>
---
#### [new 067] Supply Chain Exploitation of Secure ROS 2 Systems: A Proof-of-Concept on Autonomous Platform Compromise via Keystore Exfiltration
- **分类: cs.CR; cs.OS; cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种针对ROS 2安全框架的供应链攻击，通过篡改Debian包窃取密钥，实现对自动驾驶平台的未授权注入控制与感知欺骗，揭示了DDS系统中密钥管理与供应链信任的严重漏洞。**

- **链接: [http://arxiv.org/pdf/2511.00140v1](http://arxiv.org/pdf/2511.00140v1)**

> **作者:** Tahmid Hasan Sakib; Yago Romano Martinez; Carter Brady; Syed Rafay Hasan; Terry N. Guo
>
> **备注:** Author-accepted version (preprint). Presented at IEEE MILCOM 2025 Workshops, WS07: 2nd Workshop on Security, Resilience, and Robustness of Systems and Software (SRRSS), Los Angeles, Oct 2025. 6 pages. Primary: cs.CR; cross-lists: cs.RO, cs.OS. Program: https://milcom2025.ieee-milcom.org/workshop/ws07-2nd-workshop-security-resilient-and-robustness-systems-and-software/program
>
> **摘要:** This paper presents a proof-of-concept supply chain attack against the Secure ROS 2 (SROS 2) framework, demonstrated on a Quanser QCar2 autonomous vehicle platform. A Trojan-infected Debian package modifies core ROS 2 security commands to exfiltrate newly generated keystore credentials via DNS in base64-encoded chunks to an attacker-controlled nameserver. Possession of these credentials enables the attacker to rejoin the SROS 2 network as an authenticated participant and publish spoofed control or perception messages without triggering authentication failures. We evaluate this capability on a secure ROS 2 Humble testbed configured for a four-stop-sign navigation routine using an Intel RealSense camera for perception. Experimental results show that control-topic injections can cause forced braking, sustained high-speed acceleration, and continuous turning loops, while perception-topic spoofing can induce phantom stop signs or suppress real detections. The attack generalizes to any data distribution service (DDS)-based robotic system using SROS 2, highlighting the need for both supply chain integrity controls and runtime semantic validation to safeguard autonomous systems against insider and impersonation threats.
>
---
#### [new 068] Bootstrap Off-policy with World Model
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出BOOM框架，解决在线规划与策略行为不一致导致的模型与策略退化问题。通过引导式引导循环，联合学习世界模型与离线策略，利用无似然对齐损失提升样本效率与稳定性，在强化学习任务中实现SOTA性能。**

- **链接: [http://arxiv.org/pdf/2511.00423v1](http://arxiv.org/pdf/2511.00423v1)**

> **作者:** Guojian Zhan; Likun Wang; Xiangteng Zhang; Jiaxin Gao; Masayoshi Tomizuka; Shengbo Eben Li
>
> **备注:** NeurIPS 2025
>
> **摘要:** Online planning has proven effective in reinforcement learning (RL) for improving sample efficiency and final performance. However, using planning for environment interaction inevitably introduces a divergence between the collected data and the policy's actual behaviors, degrading both model learning and policy improvement. To address this, we propose BOOM (Bootstrap Off-policy with WOrld Model), a framework that tightly integrates planning and off-policy learning through a bootstrap loop: the policy initializes the planner, and the planner refines actions to bootstrap the policy through behavior alignment. This loop is supported by a jointly learned world model, which enables the planner to simulate future trajectories and provides value targets to facilitate policy improvement. The core of BOOM is a likelihood-free alignment loss that bootstraps the policy using the planner's non-parametric action distribution, combined with a soft value-weighted mechanism that prioritizes high-return behaviors and mitigates variability in the planner's action quality within the replay buffer. Experiments on the high-dimensional DeepMind Control Suite and Humanoid-Bench show that BOOM achieves state-of-the-art results in both training stability and final performance. The code is accessible at https://github.com/molumitu/BOOM_MBRL.
>
---
#### [new 069] SE(3)-PoseFlow: Estimating 6D Pose Distributions for Uncertainty-Aware Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对6D物体位姿估计中的多模态不确定性问题，提出SE(3)-PoseFlow，基于流匹配在SE(3)流形上建模位姿分布，替代传统确定性回归，提升对称或遮挡场景下的鲁棒性，并支持不确定性感知的机器人操作。**

- **链接: [http://arxiv.org/pdf/2511.01501v1](http://arxiv.org/pdf/2511.01501v1)**

> **作者:** Yufeng Jin; Niklas Funk; Vignesh Prasad; Zechu Li; Mathias Franzius; Jan Peters; Georgia Chalvatzaki
>
> **摘要:** Object pose estimation is a fundamental problem in robotics and computer vision, yet it remains challenging due to partial observability, occlusions, and object symmetries, which inevitably lead to pose ambiguity and multiple hypotheses consistent with the same observation. While deterministic deep networks achieve impressive performance under well-constrained conditions, they are often overconfident and fail to capture the multi-modality of the underlying pose distribution. To address these challenges, we propose a novel probabilistic framework that leverages flow matching on the SE(3) manifold for estimating 6D object pose distributions. Unlike existing methods that regress a single deterministic output, our approach models the full pose distribution with a sample-based estimate and enables reasoning about uncertainty in ambiguous cases such as symmetric objects or severe occlusions. We achieve state-of-the-art results on Real275, YCB-V, and LM-O, and demonstrate how our sample-based pose estimates can be leveraged in downstream robotic manipulation tasks such as active perception for disambiguating uncertain viewpoints or guiding grasp synthesis in an uncertainty-aware manner.
>
---
#### [new 070] 3EED: Ground Everything Everywhere in 3D
- **分类: cs.CV; cs.RO**

- **简介: 论文提出3EED基准，面向多平台（车、无人机、四足机器人）的3D视觉接地任务，解决现有数据集规模小、场景单一问题，构建了超12万物体、2.2万表达的户外多模态数据集，并提出跨平台对齐方法，推动语言驱动的3D感知研究。**

- **链接: [http://arxiv.org/pdf/2511.01755v1](http://arxiv.org/pdf/2511.01755v1)**

> **作者:** Rong Li; Yuhao Dong; Tianshuai Hu; Ao Liang; Youquan Liu; Dongyue Lu; Liang Pan; Lingdong Kong; Junwei Liang; Ziwei Liu
>
> **备注:** NeurIPS 2025 DB Track; 29 pages, 17 figures, 10 tables; Project Page at https://project-3eed.github.io/
>
> **摘要:** Visual grounding in 3D is the key for embodied agents to localize language-referred objects in open-world environments. However, existing benchmarks are limited to indoor focus, single-platform constraints, and small scale. We introduce 3EED, a multi-platform, multi-modal 3D grounding benchmark featuring RGB and LiDAR data from vehicle, drone, and quadruped platforms. We provide over 128,000 objects and 22,000 validated referring expressions across diverse outdoor scenes -- 10x larger than existing datasets. We develop a scalable annotation pipeline combining vision-language model prompting with human verification to ensure high-quality spatial grounding. To support cross-platform learning, we propose platform-aware normalization and cross-modal alignment techniques, and establish benchmark protocols for in-domain and cross-platform evaluations. Our findings reveal significant performance gaps, highlighting the challenges and opportunities of generalizable 3D grounding. The 3EED dataset and benchmark toolkit are released to advance future research in language-driven 3D embodied perception.
>
---
#### [new 071] AeroResQ: Edge-Accelerated UAV Framework for Scalable, Resilient and Collaborative Escape Route Planning in Wildfire Scenarios
- **分类: cs.DC; cs.RO**

- **简介: AeroResQ提出一种边缘加速的无人机协作框架，用于野火场景下的实时逃生路径规划。通过服务无人机检测受困者、协调无人机动态生成最优路径，结合轻量级存储与容错机制，实现低延迟（≤500ms）与高可靠性逃生响应。**

- **链接: [http://arxiv.org/pdf/2511.00038v1](http://arxiv.org/pdf/2511.00038v1)**

> **作者:** Suman Raj; Radhika Mittal; Rajiv Mayani; Pawel Zuk; Anirban Mandal; Michael Zink; Yogesh Simmhan; Ewa Deelman
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** Drone fleets equipped with onboard cameras, computer vision, and Deep Neural Network (DNN) models present a powerful paradigm for real-time spatio-temporal decision-making. In wildfire response, such drones play a pivotal role in monitoring fire dynamics, supporting firefighter coordination, and facilitating safe evacuation. In this paper, we introduce AeroResQ, an edge-accelerated UAV framework designed for scalable, resilient, and collaborative escape route planning during wildfire scenarios. AeroResQ adopts a multi-layer orchestration architecture comprising service drones (SDs) and coordinator drones (CDs), each performing specialized roles. SDs survey fire-affected areas, detect stranded individuals using onboard edge accelerators running fire detection and human pose identification DNN models, and issue requests for assistance. CDs, equipped with lightweight data stores such as Apache IoTDB, dynamically generate optimal ground escape routes and monitor firefighter movements along these routes. The framework proposes a collaborative path-planning approach based on a weighted A* search algorithm, where CDs compute context-aware escape paths. AeroResQ further incorporates intelligent load-balancing and resilience mechanisms: CD failures trigger automated data redistribution across IoTDB replicas, while SD failures initiate geo-fenced re-partitioning and reassignment of spatial workloads to operational SDs. We evaluate AeroResQ using realistic wildfire emulated setup modeled on recent Southern California wildfires. Experimental results demonstrate that AeroResQ achieves a nominal end-to-end latency of <=500ms, much below the 2s request interval, while maintaining over 98% successful task reassignment and completion, underscoring its feasibility for real-time, on-field deployment in emergency response and firefighter safety operations.
>
---
#### [new 072] OmniTrack++: Omnidirectional Multi-Object Tracking by Learning Large-FoV Trajectory Feedback
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 论文提出OmniTrack++，面向全景图像的多目标跟踪任务，解决360°视场下的畸变、分辨率低与身份混淆问题。通过轨迹反馈机制、动态特征稳定与专家记忆模块，提升跟踪鲁棒性，并构建EmboTrack基准数据集验证效果。**

- **链接: [http://arxiv.org/pdf/2511.00510v1](http://arxiv.org/pdf/2511.00510v1)**

> **作者:** Kai Luo; Hao Shi; Kunyu Peng; Fei Teng; Sheng Wu; Kaiwei Wang; Kailun Yang
>
> **备注:** Extended version of CVPR 2025 paper arXiv:2503.04565. Datasets and code will be made publicly available at https://github.com/xifen523/OmniTrack
>
> **摘要:** This paper investigates Multi-Object Tracking (MOT) in panoramic imagery, which introduces unique challenges including a 360{\deg} Field of View (FoV), resolution dilution, and severe view-dependent distortions. Conventional MOT methods designed for narrow-FoV pinhole cameras generalize unsatisfactorily under these conditions. To address panoramic distortion, large search space, and identity ambiguity under a 360{\deg} FoV, OmniTrack++ adopts a feedback-driven framework that progressively refines perception with trajectory cues. A DynamicSSM block first stabilizes panoramic features, implicitly alleviating geometric distortion. On top of normalized representations, FlexiTrack Instances use trajectory-informed feedback for flexible localization and reliable short-term association. To ensure long-term robustness, an ExpertTrack Memory consolidates appearance cues via a Mixture-of-Experts design, enabling recovery from fragmented tracks and reducing identity drift. Finally, a Tracklet Management module adaptively switches between end-to-end and tracking-by-detection modes according to scene dynamics, offering a balanced and scalable solution for panoramic MOT. To support rigorous evaluation, we establish the EmboTrack benchmark, a comprehensive dataset for panoramic MOT that includes QuadTrack, captured with a quadruped robot, and BipTrack, collected with a bipedal wheel-legged robot. Together, these datasets span wide-angle environments and diverse motion patterns, providing a challenging testbed for real-world panoramic perception. Extensive experiments on JRDB and EmboTrack demonstrate that OmniTrack++ achieves state-of-the-art performance, yielding substantial HOTA improvements of +25.5% on JRDB and +43.07% on QuadTrack over the original OmniTrack. Datasets and code will be made publicly available at https://github.com/xifen523/OmniTrack.
>
---
#### [new 073] Model-free source seeking of exponentially convergent unicycle: theoretical and robotic experimental results
- **分类: math.OC; cs.RO**

- **简介: 该论文提出一种无模型的单轮车源寻优方法，解决传统方法仅适用于二次目标函数的局限，首次实现对高次多项式函数的指数收敛，并通过理论、仿真与机器人实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2511.00752v1](http://arxiv.org/pdf/2511.00752v1)**

> **作者:** Rohan Palanikumar; Ahmed A. Elgohary; Victoria Grushkovskaya; Sameh A. Eisa
>
> **摘要:** This paper introduces a novel model-free, real-time unicycle-based source seeking design. This design steers autonomously the unicycle dynamic system towards the extremum point of an objective function or physical/scaler signal that is unknown expression-wise, but accessible via measurements. A key contribution of this paper is that the introduced design converges exponentially to the extremum point of objective functions (or scaler signals) that behave locally like a higher-degree power functions (e.g., fourth degree polynomial function) as opposed to locally quadratic objective functions, the usual case in literature. We provide theoretical and simulation results to support out theoretical results. Also, for the first time in the literature, we provide experimental robotic results that demonstrate the effectiveness of the proposed design and its exponential convergence ability.
>
---
#### [new 074] Saliency-Guided Domain Adaptation for Left-Hand Driving in Autonomous Steering
- **分类: cs.CV; cs.RO**

- **简介: 该论文面向自动驾驶左舵驾驶适配任务，解决右舵数据到左舵场景的域适应问题。通过对比翻转数据预训练+微调策略，发现其能有效提升模型对左侧路标关注，降低转向误差，验证了预处理策略的有效性。**

- **链接: [http://arxiv.org/pdf/2511.01223v1](http://arxiv.org/pdf/2511.01223v1)**

> **作者:** Zahra Mehraban; Sebastien Glaser; Michael Milford; Ronald Schroeter
>
> **摘要:** Domain adaptation is required for automated driving models to generalize well across diverse road conditions. This paper explores a training method for domain adaptation to adapt PilotNet, an end-to-end deep learning-based model, for left-hand driving conditions using real-world Australian highway data. Four training methods were evaluated: (1) a baseline model trained on U.S. right-hand driving data, (2) a model trained on flipped U.S. data, (3) a model pretrained on U.S. data and then fine-tuned on Australian highways, and (4) a model pretrained on flipped U.S. data and then finetuned on Australian highways. This setup examines whether incorporating flipped data enhances the model adaptation by providing an initial left-hand driving alignment. The paper compares model performance regarding steering prediction accuracy and attention, using saliency-based analysis to measure attention shifts across significant road regions. Results show that pretraining on flipped data alone worsens prediction stability due to misaligned feature representations, but significantly improves adaptation when followed by fine-tuning, leading to lower prediction error and stronger focus on left-side cues. To validate this approach across different architectures, the same experiments were done on ResNet, which confirmed similar adaptation trends. These findings emphasize the importance of preprocessing techniques, such as flipped-data pretraining, followed by fine-tuning to improve model adaptation with minimal retraining requirements.
>
---
#### [new 075] Fractional Diffusion Bridge Models
- **分类: cs.LG; cs.AI; cs.CV; cs.RO; stat.ML**

- **简介: 该论文提出分数扩散桥模型（FDBM），用于建模具有长程依赖和记忆效应的非马尔可夫随机过程，解决传统布朗运动模型无法捕捉真实动态的问题。通过马尔可夫近似实现可推断生成，应用于蛋白质构象预测与无配对图像翻译，性能超越基线。**

- **链接: [http://arxiv.org/pdf/2511.01795v1](http://arxiv.org/pdf/2511.01795v1)**

> **作者:** Gabriel Nobis; Maximilian Springenberg; Arina Belova; Rembert Daems; Christoph Knochenhauer; Manfred Opper; Tolga Birdal; Wojciech Samek
>
> **备注:** To appear in NeurIPS 2025 proceedings. This version includes post-camera-ready revisions
>
> **摘要:** We present Fractional Diffusion Bridge Models (FDBM), a novel generative diffusion bridge framework driven by an approximation of the rich and non-Markovian fractional Brownian motion (fBM). Real stochastic processes exhibit a degree of memory effects (correlations in time), long-range dependencies, roughness and anomalous diffusion phenomena that are not captured in standard diffusion or bridge modeling due to the use of Brownian motion (BM). As a remedy, leveraging a recent Markovian approximation of fBM (MA-fBM), we construct FDBM that enable tractable inference while preserving the non-Markovian nature of fBM. We prove the existence of a coupling-preserving generative diffusion bridge and leverage it for future state prediction from paired training data. We then extend our formulation to the Schr\"{o}dinger bridge problem and derive a principled loss function to learn the unpaired data translation. We evaluate FDBM on both tasks: predicting future protein conformations from aligned data, and unpaired image translation. In both settings, FDBM achieves superior performance compared to the Brownian baselines, yielding lower root mean squared deviation (RMSD) of C$_\alpha$ atomic positions in protein structure prediction and lower Fr\'echet Inception Distance (FID) in unpaired image translation.
>
---
#### [new 076] X-TRACK: Physics-Aware xLSTM for Realistic Vehicle Trajectory Prediction
- **分类: cs.LG; cs.RO**

- **简介: 论文提出X-TRACK，一种融合车辆运动学约束的xLSTM模型，用于车辆轨迹预测，解决传统模型生成不现实轨迹的问题，通过物理约束提升预测合理性与准确性，并在高精度数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.00266v1](http://arxiv.org/pdf/2511.00266v1)**

> **作者:** Aanchal Rajesh Chugh; Marion Neumeier; Sebastian Dorn
>
> **摘要:** Recent advancements in Recurrent Neural Network (RNN) architectures, particularly the Extended Long Short Term Memory (xLSTM), have addressed the limitations of traditional Long Short Term Memory (LSTM) networks by introducing exponential gating and enhanced memory structures. These improvements make xLSTM suitable for time-series prediction tasks as they exhibit the ability to model long-term temporal dependencies better than LSTMs. Despite their potential, these xLSTM-based models remain largely unexplored in the context of vehicle trajectory prediction. Therefore, this paper introduces a novel xLSTM-based vehicle trajectory prediction framework, X-TRAJ, and its physics-aware variant, X-TRACK (eXtended LSTM for TRAjectory prediction Constraint by Kinematics), which explicitly integrates vehicle motion kinematics into the model learning process. By introducing physical constraints, the proposed model generates realistic and feasible trajectories. A comprehensive evaluation on the highD and NGSIM datasets demonstrates that X-TRACK outperforms state-of-the-art baselines.
>
---
#### [new 077] Which LiDAR scanning pattern is better for roadside perception: Repetitive or Non-repetitive?
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文研究道路侧LiDAR扫描模式（重复 vs 非重复）对感知性能的影响，提出InfraLiDARs' Benchmark数据集，评估不同扫描模式下3D目标检测算法表现，发现非重复模式虽范围有限但成本低、性能相当，为路侧感知系统选型提供依据。**

- **链接: [http://arxiv.org/pdf/2511.00060v1](http://arxiv.org/pdf/2511.00060v1)**

> **作者:** Zhiqi Qi; Runxin Zhao; Hanyang Zhuang; Chunxiang Wang; Ming Yang
>
> **摘要:** LiDAR-based roadside perception is a cornerstone of advanced Intelligent Transportation Systems (ITS). While considerable research has addressed optimal LiDAR placement for infrastructure, the profound impact of differing LiDAR scanning patterns on perceptual performance remains comparatively under-investigated. The inherent nature of various scanning modes - such as traditional repetitive (mechanical/solid-state) versus emerging non-repetitive (e.g. prism-based) systems - leads to distinct point cloud distributions at varying distances, critically dictating the efficacy of object detection and overall environmental understanding. To systematically investigate these differences in infrastructure-based contexts, we introduce the "InfraLiDARs' Benchmark," a novel dataset meticulously collected in the CARLA simulation environment using concurrently operating infrastructure-based LiDARs exhibiting both scanning paradigms. Leveraging this benchmark, we conduct a comprehensive statistical analysis of the respective LiDAR scanning abilities and evaluate the impact of these distinct patterns on the performance of various leading 3D object detection algorithms. Our findings reveal that non-repetitive scanning LiDAR and the 128-line repetitive LiDAR were found to exhibit comparable detection performance across various scenarios. Despite non-repetitive LiDAR's limited perception range, it's a cost-effective option considering its low price. Ultimately, this study provides insights for setting up roadside perception system with optimal LiDAR scanning patterns and compatible algorithms for diverse roadside applications, and publicly releases the "InfraLiDARs' Benchmark" dataset to foster further research.
>
---
#### [new 078] World Simulation with Video Foundation Models for Physical AI
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出Cosmos-Predict2.5与Cosmos-Transfer2.5，构建基于视频基础模型的物理世界仿真系统，统一文本、图像、视频到世界生成，提升仿真质量与控制精度，支持机器人与自主系统训练，并开源模型与数据。**

- **链接: [http://arxiv.org/pdf/2511.00062v1](http://arxiv.org/pdf/2511.00062v1)**

> **作者:** NVIDIA; :; Arslan Ali; Junjie Bai; Maciej Bala; Yogesh Balaji; Aaron Blakeman; Tiffany Cai; Jiaxin Cao; Tianshi Cao; Elizabeth Cha; Yu-Wei Chao; Prithvijit Chattopadhyay; Mike Chen; Yongxin Chen; Yu Chen; Shuai Cheng; Yin Cui; Jenna Diamond; Yifan Ding; Jiaojiao Fan; Linxi Fan; Liang Feng; Francesco Ferroni; Sanja Fidler; Xiao Fu; Ruiyuan Gao; Yunhao Ge; Jinwei Gu; Aryaman Gupta; Siddharth Gururani; Imad El Hanafi; Ali Hassani; Zekun Hao; Jacob Huffman; Joel Jang; Pooya Jannaty; Jan Kautz; Grace Lam; Xuan Li; Zhaoshuo Li; Maosheng Liao; Chen-Hsuan Lin; Tsung-Yi Lin; Yen-Chen Lin; Huan Ling; Ming-Yu Liu; Xian Liu; Yifan Lu; Alice Luo; Qianli Ma; Hanzi Mao; Kaichun Mo; Seungjun Nah; Yashraj Narang; Abhijeet Panaskar; Lindsey Pavao; Trung Pham; Morteza Ramezanali; Fitsum Reda; Scott Reed; Xuanchi Ren; Haonan Shao; Yue Shen; Stella Shi; Shuran Song; Bartosz Stefaniak; Shangkun Sun; Shitao Tang; Sameena Tasmeen; Lyne Tchapmi; Wei-Cheng Tseng; Jibin Varghese; Andrew Z. Wang; Hao Wang; Haoxiang Wang; Heng Wang; Ting-Chun Wang; Fangyin Wei; Jiashu Xu; Dinghao Yang; Xiaodong Yang; Haotian Ye; Seonghyeon Ye; Xiaohui Zeng; Jing Zhang; Qinsheng Zhang; Kaiwen Zheng; Andrew Zhu; Yuke Zhu
>
> **摘要:** We introduce [Cosmos-Predict2.5], the latest generation of the Cosmos World Foundation Models for Physical AI. Built on a flow-based architecture, [Cosmos-Predict2.5] unifies Text2World, Image2World, and Video2World generation in a single model and leverages [Cosmos-Reason1], a Physical AI vision-language model, to provide richer text grounding and finer control of world simulation. Trained on 200M curated video clips and refined with reinforcement learning-based post-training, [Cosmos-Predict2.5] achieves substantial improvements over [Cosmos-Predict1] in video quality and instruction alignment, with models released at 2B and 14B scales. These capabilities enable more reliable synthetic data generation, policy evaluation, and closed-loop simulation for robotics and autonomous systems. We further extend the family with [Cosmos-Transfer2.5], a control-net style framework for Sim2Real and Real2Real world translation. Despite being 3.5$\times$ smaller than [Cosmos-Transfer1], it delivers higher fidelity and robust long-horizon video generation. Together, these advances establish [Cosmos-Predict2.5] and [Cosmos-Transfer2.5] as versatile tools for scaling embodied intelligence. To accelerate research and deployment in Physical AI, we release source code, pretrained checkpoints, and curated benchmarks under the NVIDIA Open Model License at https://github.com/nvidia-cosmos/cosmos-predict2.5 and https://github.com/nvidia-cosmos/cosmos-transfer2.5. We hope these open resources lower the barrier to adoption and foster innovation in building the next generation of embodied intelligence.
>
---
#### [new 079] EREBUS: End-to-end Robust Event Based Underwater Simulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EREBUS框架，用于生成AUV搭载事件相机的水下仿真数据，解决传统视觉方法在低能见度、高动态场景下性能差的问题，支持如岩石检测等水下视觉任务。**

- **链接: [http://arxiv.org/pdf/2511.01381v1](http://arxiv.org/pdf/2511.01381v1)**

> **作者:** Hitesh Kyatham; Arjun Suresh; Aadi Palnitkar; Yiannis Aloimonos
>
> **备注:** Accepted to ICRA AQUA2SIM Workshop 2025, 6 pages, 3 figures, conference paper
>
> **摘要:** The underwater domain presents a vast array of challenges for roboticists and computer vision researchers alike, such as poor lighting conditions and high dynamic range scenes. In these adverse conditions, traditional vision techniques struggle to adapt and lead to suboptimal performance. Event-based cameras present an attractive solution to this problem, mitigating the issues of traditional cameras by tracking changes in the footage on a frame-by-frame basis. In this paper, we introduce a pipeline which can be used to generate realistic synthetic data of an event-based camera mounted to an AUV (Autonomous Underwater Vehicle) in an underwater environment for training vision models. We demonstrate the effectiveness of our pipeline using the task of rock detection with poor visibility and suspended particulate matter, but the approach can be generalized to other underwater tasks.
>
---
#### [new 080] pacSTL: PAC-Bounded Signal Temporal Logic from Data-Driven Reachability Analysis
- **分类: cs.LO; cs.RO**

- **简介: 论文提出pacSTL，将PAC理论与区间扩展的STL结合，解决机器人系统在不确定性下安全规范的量化验证问题，通过优化原子命题生成PAC有界鲁棒性区间，用于实时监控，并在航海场景验证了有效性。**

- **链接: [http://arxiv.org/pdf/2511.00934v1](http://arxiv.org/pdf/2511.00934v1)**

> **作者:** Elizabeth Dietrich; Hanna Krasowski; Emir Cem Gezer; Roger Skjetne; Asgeir Johan Sørensen; Murat Arcak
>
> **摘要:** Real-world robotic systems must comply with safety requirements in the presence of uncertainty. To define and measure requirement adherence, Signal Temporal Logic (STL) offers a mathematically rigorous and expressive language. However, standard STL cannot account for uncertainty. We address this problem by presenting pacSTL, a framework that combines Probably Approximately Correct (PAC) bounded set predictions with an interval extension of STL through optimization problems on the atomic proposition level. pacSTL provides PAC-bounded robustness intervals on the specification level that can be utilized in monitoring. We demonstrate the effectiveness of this approach through maritime navigation and analyze the efficiency and scalability of pacSTL through simulation and real-world experimentation on model vessels.
>
---
#### [new 081] Pelican-VL 1.0: A Foundation Brain Model for Embodied Intelligence
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: Pelican-VL 1.0 是一种开源具身智能基础模型，旨在提升AI在实体环境中的智能表现。通过金属oop训练框架与DPPO算法，利用4B+令牌数据和千卡GPU训练，实现性能超越百亿级开源模型，逼近闭源系统水平。**

- **链接: [http://arxiv.org/pdf/2511.00108v1](http://arxiv.org/pdf/2511.00108v1)**

> **作者:** Yi Zhang; Che Liu; Xiancong Ren; Hanchu Ni; Shuai Zhang; Zeyuan Ding; Jiayu Hu; Hanzhe Shan; Zhenwei Niu; Zhaoyang Liu; Yue Zhao; Junbo Qi; Qinfan Zhang; Dengjie Li; Yidong Wang; Jiachen Luo; Yong Dai; Jian Tang; Xiaozhu Ju
>
> **摘要:** This report presents Pelican-VL 1.0, a new family of open-source embodied brain models with parameter scales ranging from 7 billion to 72 billion. Our explicit mission is clearly stated as: To embed powerful intelligence into various embodiments. Pelican-VL 1.0 is currently the largest-scale open-source embodied multimodal brain model. Its core advantage lies in the in-depth integration of data power and intelligent adaptive learning mechanisms. Specifically, metaloop distilled a high-quality dataset from a raw dataset containing 4+ billion tokens. Pelican-VL 1.0 is trained on a large-scale cluster of 1000+ A800 GPUs, consuming over 50k+ A800 GPU-hours per checkpoint. This translates to a 20.3% performance uplift from its base model and outperforms 100B-level open-source counterparts by 10.6%, placing it on par with leading proprietary systems on well-known embodied benchmarks. We establish a novel framework, DPPO (Deliberate Practice Policy Optimization), inspired by human metacognition to train Pelican-VL 1.0. We operationalize this as a metaloop that teaches the AI to practice deliberately, which is a RL-Refine-Diagnose-SFT loop.
>
---
## 更新

#### [replaced 001] RoboOmni: Proactive Robot Manipulation in Omni-modal Context
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.23763v3](http://arxiv.org/pdf/2510.23763v3)**

> **作者:** Siyin Wang; Jinlan Fu; Feihong Liu; Xinzhe He; Huangxuan Wu; Junhao Shi; Kexin Huang; Zhaoye Fei; Jingjing Gong; Zuxuan Wu; Yu-Gang Jiang; See-Kiong Ng; Tat-Seng Chua; Xipeng Qiu
>
> **摘要:** Recent advances in Multimodal Large Language Models (MLLMs) have driven rapid progress in Vision-Language-Action (VLA) models for robotic manipulation. Although effective in many scenarios, current approaches largely rely on explicit instructions, whereas in real-world interactions, humans rarely issue instructions directly. Effective collaboration requires robots to infer user intentions proactively. In this work, we introduce cross-modal contextual instructions, a new setting where intent is derived from spoken dialogue, environmental sounds, and visual cues rather than explicit commands. To address this new setting, we present RoboOmni, a Perceiver-Thinker-Talker-Executor framework based on end-to-end omni-modal LLMs that unifies intention recognition, interaction confirmation, and action execution. RoboOmni fuses auditory and visual signals spatiotemporally for robust intention recognition, while supporting direct speech interaction. To address the absence of training data for proactive intention recognition in robotic manipulation, we build OmniAction, comprising 140k episodes, 5k+ speakers, 2.4k event sounds, 640 backgrounds, and six contextual instruction types. Experiments in simulation and real-world settings show that RoboOmni surpasses text- and ASR-based baselines in success rate, inference speed, intention recognition, and proactive assistance.
>
---
#### [replaced 002] Knolling Bot: Teaching Robots the Human Notion of Tidiness
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2310.04566v3](http://arxiv.org/pdf/2310.04566v3)**

> **作者:** Yuhang Hu; Judah Goldfeder; Zhizhuo Zhang; Xinyue Zhu; Ruibo Liu; Philippe Wyder; Jiong Lin; Hod Lipson
>
> **备注:** Accepted at the 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Creative AI Track
>
> **摘要:** For robots to truly collaborate and assist humans, they must understand not only logic and instructions, but also the subtle emotions, aesthetics, and feelings that define our humanity. Human art and aesthetics are among the most elusive concepts-often difficult even for people to articulate-and without grasping these fundamentals, robots will be unable to help in many spheres of daily life. Consider the long-promised robotic butler: automating domestic chores demands more than motion planning. It requires an internal model of cleanliness and tidiness-a challenge largely unexplored by AI. To bridge this gap, we propose an approach that equips domestic robots to perform simple tidying tasks via knolling, the practice of arranging scattered items into neat, space-efficient layouts. Unlike the uniformity of industrial settings, household environments feature diverse objects and highly subjective notions of tidiness. Drawing inspiration from NLP, we treat knolling as a sequential prediction problem and employ a transformer based model to forecast each object's placement. Our method learns a generalizable concept of tidiness, generates diverse solutions adaptable to varying object sets, and incorporates human preferences for personalized arrangements. This work represents a step forward in building robots that internalize human aesthetic sense and can genuinely co-create in our living spaces.
>
---
#### [replaced 003] MarsLGPR: Mars Rover Localization with Ground Penetrating Radar
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04944v2](http://arxiv.org/pdf/2503.04944v2)**

> **作者:** Anja Sheppard; Katherine A. Skinner
>
> **摘要:** In this work, we propose the use of Ground Penetrating Radar (GPR) for rover localization on Mars. Precise pose estimation is an important task for mobile robots exploring planetary surfaces, as they operate in GPS-denied environments. Although visual odometry provides accurate localization, it is computationally expensive and can fail in dim or high-contrast lighting. Wheel encoders can also provide odometry estimation, but are prone to slipping on the sandy terrain encountered on Mars. Although traditionally a scientific surveying sensor, GPR has been used on Earth for terrain classification and localization through subsurface feature matching. The Perseverance rover and the upcoming ExoMars rover have GPR sensors already equipped to aid in the search of water and mineral resources. We propose to leverage GPR to aid in Mars rover localization. Specifically, we develop a novel GPR-based deep learning model that predicts 1D relative pose translation. We fuse our GPR pose prediction method with inertial and wheel encoder data in a filtering framework to output rover localization. We perform experiments in a Mars analog environment and demonstrate that our GPR-based displacement predictions both outperform wheel encoders and improve multi-modal filtering estimates in high-slip environments. Lastly, we present the first dataset aimed at GPR-based localization in Mars analog environments, which will be made publicly available at https://umfieldrobotics.github.io/marslgpr.
>
---
#### [replaced 004] FlexEvent: Towards Flexible Event-Frame Object Detection at Varying Operational Frequencies
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.06708v3](http://arxiv.org/pdf/2412.06708v3)**

> **作者:** Dongyue Lu; Lingdong Kong; Gim Hee Lee; Camille Simon Chane; Wei Tsang Ooi
>
> **备注:** NeurIPS 2025; 28 pages, 14 figures, 10 tables; Code at https://flexevent.github.io/
>
> **摘要:** Event cameras offer unparalleled advantages for real-time perception in dynamic environments, thanks to the microsecond-level temporal resolution and asynchronous operation. Existing event detectors, however, are limited by fixed-frequency paradigms and fail to fully exploit the high-temporal resolution and adaptability of event data. To address these limitations, we propose FlexEvent, a novel framework that enables detection at varying frequencies. Our approach consists of two key components: FlexFuse, an adaptive event-frame fusion module that integrates high-frequency event data with rich semantic information from RGB frames, and FlexTune, a frequency-adaptive fine-tuning mechanism that generates frequency-adjusted labels to enhance model generalization across varying operational frequencies. This combination allows our method to detect objects with high accuracy in both fast-moving and static scenarios, while adapting to dynamic environments. Extensive experiments on large-scale event camera datasets demonstrate that our approach surpasses state-of-the-art methods, achieving significant improvements in both standard and high-frequency settings. Notably, our method maintains robust performance when scaling from 20 Hz to 90 Hz and delivers accurate detection up to 180 Hz, proving its effectiveness in extreme conditions. Our framework sets a new benchmark for event-based object detection and paves the way for more adaptable, real-time vision systems.
>
---
#### [replaced 005] Neuro-Symbolic Imitation Learning: Discovering Symbolic Abstractions for Skill Learning
- **分类: cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.21406v2](http://arxiv.org/pdf/2503.21406v2)**

> **作者:** Leon Keller; Daniel Tanneberg; Jan Peters
>
> **备注:** IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** Imitation learning is a popular method for teaching robots new behaviors. However, most existing methods focus on teaching short, isolated skills rather than long, multi-step tasks. To bridge this gap, imitation learning algorithms must not only learn individual skills but also an abstract understanding of how to sequence these skills to perform extended tasks effectively. This paper addresses this challenge by proposing a neuro-symbolic imitation learning framework. Using task demonstrations, the system first learns a symbolic representation that abstracts the low-level state-action space. The learned representation decomposes a task into easier subtasks and allows the system to leverage symbolic planning to generate abstract plans. Subsequently, the system utilizes this task decomposition to learn a set of neural skills capable of refining abstract plans into actionable robot commands. Experimental results in three simulated robotic environments demonstrate that, compared to baselines, our neuro-symbolic approach increases data efficiency, improves generalization capabilities, and facilitates interpretability.
>
---
#### [replaced 006] DW-A-PRM: A Dynamic Weighted Planner
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.05701v2](http://arxiv.org/pdf/2509.05701v2)**

> **作者:** Siyuan Wang; Shuyi Zhang; Zhen Tian; Yuheng Yao; Gongsen Wang; Yu Zhao
>
> **摘要:** Robot path planning plays a pivotal role in enabling autonomous systems to navigate safely and efficiently in complex and uncertain environments. Despite extensive research on classical graph-based methods and sampling-based planners, achieving an optimal balance between global optimality, computational efficiency, and adaptability to dynamic environments remains an open challenge. To address this issue, this paper proposes a hybrid path planning framework, which integrates heuristic-driven search with probabilistic roadmap construction under a dynamic weighting scheme. By coupling the global guidance of A* with the stochastic exploration of PRM, the method achieves a synergistic balance between search optimality and computational tractability. Comprehensive experiments in diverse simulated environments demonstrate that the proposed method consistently yields smoother and shorter paths while significantly reducing computational overhead compared with conventional approach and other hybrid planners. These results highlight the potential of the proposed framework as an effective and generalizable solution for real-time robotic navigation in complex environments.
>
---
#### [replaced 007] Cosmos-Surg-dVRK: World Foundation Model-based Automated Online Evaluation of Surgical Robot Policy Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.16240v2](http://arxiv.org/pdf/2510.16240v2)**

> **作者:** Lukas Zbinden; Nigel Nelson; Juo-Tung Chen; Xinhao Chen; Ji Woong Kim; Mahdi Azizian; Axel Krieger; Sean Huver
>
> **备注:** minor metadata and notation fixes; +3 citations
>
> **摘要:** The rise of surgical robots and vision-language-action models has accelerated the development of autonomous surgical policies and efficient assessment strategies. However, evaluating these policies directly on physical robotic platforms such as the da Vinci Research Kit (dVRK) remains hindered by high costs, time demands, reproducibility challenges, and variability in execution. World foundation models (WFM) for physical AI offer a transformative approach to simulate complex real-world surgical tasks, such as soft tissue deformation, with high fidelity. This work introduces Cosmos-Surg-dVRK, a surgical finetune of the Cosmos WFM, which, together with a trained video classifier, enables fully automated online evaluation and benchmarking of surgical policies. We evaluate Cosmos-Surg-dVRK using two distinct surgical datasets. On tabletop suture pad tasks, the automated pipeline achieves strong correlation between online rollouts in Cosmos-Surg-dVRK and policy outcomes on the real dVRK Si platform, as well as good agreement between human labelers and the V-JEPA 2-derived video classifier. Additionally, preliminary experiments with ex-vivo porcine cholecystectomy tasks in Cosmos-Surg-dVRK demonstrate promising alignment with real-world evaluations, highlighting the platform's potential for more complex surgical procedures.
>
---
#### [replaced 008] Bellman Diffusion Models
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.12163v2](http://arxiv.org/pdf/2407.12163v2)**

> **作者:** Liam Schramm; Abdeslam Boularias
>
> **摘要:** Diffusion models have seen tremendous success as generative architectures. Recently, they have been shown to be effective at modelling policies for offline reinforcement learning and imitation learning. We explore using diffusion as a model class for the successor state measure (SSM) of a policy. We find that enforcing the Bellman flow constraints leads to a simple Bellman update on the diffusion step distribution.
>
---
#### [replaced 009] Co-MTP: A Cooperative Trajectory Prediction Framework with Multi-Temporal Fusion for Autonomous Driving
- **分类: cs.LG; cs.AI; cs.CV; cs.RO; 68T07; I.2.6**

- **链接: [http://arxiv.org/pdf/2502.16589v3](http://arxiv.org/pdf/2502.16589v3)**

> **作者:** Xinyu Zhang; Zewei Zhou; Zhaoyi Wang; Yangjie Ji; Yanjun Huang; Hong Chen
>
> **备注:** 8 pages, 3 figures, ICRA 2025
>
> **摘要:** Vehicle-to-everything technologies (V2X) have become an ideal paradigm to extend the perception range and see through the occlusion. Exiting efforts focus on single-frame cooperative perception, however, how to capture the temporal cue between frames with V2X to facilitate the prediction task even the planning task is still underexplored. In this paper, we introduce the Co-MTP, a general cooperative trajectory prediction framework with multi-temporal fusion for autonomous driving, which leverages the V2X system to fully capture the interaction among agents in both history and future domains to benefit the planning. In the history domain, V2X can complement the incomplete history trajectory in single-vehicle perception, and we design a heterogeneous graph transformer to learn the fusion of the history feature from multiple agents and capture the history interaction. Moreover, the goal of prediction is to support future planning. Thus, in the future domain, V2X can provide the prediction results of surrounding objects, and we further extend the graph transformer to capture the future interaction among the ego planning and the other vehicles' intentions and obtain the final future scenario state under a certain planning action. We evaluate the Co-MTP framework on the real-world dataset V2X-Seq, and the results show that Co-MTP achieves state-of-the-art performance and that both history and future fusion can greatly benefit prediction.
>
---
#### [replaced 010] Robust Trajectory Generation and Control for Quadrotor Motion Planning with Field-of-View Control Barrier Certification
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.01009v2](http://arxiv.org/pdf/2502.01009v2)**

> **作者:** Lishuo Pan; Mattia Catellani; Lorenzo Sabattini; Nora Ayanian
>
> **备注:** 8 pages, 8 figures, 3 tables, accepted to RA-L 2025
>
> **摘要:** Many approaches to multi-robot coordination are susceptible to failure due to communication loss and uncertainty in estimation. We present a real-time communication-free distributed navigation algorithm certified by control barrier functions, that models and controls the onboard sensing behavior to keep neighbors in the limited field of view for position estimation. The approach is robust to temporary tracking loss and directly synthesizes control to stabilize visual contact through control Lyapunov-barrier functions. The main contributions of this paper are a continuous-time robust trajectory generation and control method certified by control barrier functions for distributed multi-robot systems and a discrete optimization procedure, namely, MPC-CBF, to approximate the certified controller. In addition, we propose a linear surrogate of high-order control barrier function constraints and use sequential quadratic programming to solve MPC-CBF efficiently.
>
---
#### [replaced 011] If They Disagree, Will You Conform? Exploring the Role of Robots' Value Awareness in a Decision-Making Task
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2510.23204v2](http://arxiv.org/pdf/2510.23204v2)**

> **作者:** Giulia Pusceddu; Giulio Antonio Abbo; Francesco Rea; Tony Belpaeme; Alessandra Sciutti
>
> **备注:** Pre-print version
>
> **摘要:** This study investigates whether the opinions of robotic agents can influence human decision-making when robots display value awareness (i.e., the capability of understanding human preferences and prioritizing them in decision-making). We designed an experiment in which participants interacted with two Furhat robots - one programmed to be Value-Aware and the other Non-Value-Aware - during a labeling task for images representing human values. Results indicate that participants distinguished the Value-Aware robot from the Non-Value-Aware one. Although their explicit choices did not indicate a clear preference for one robot over the other, participants directed their gaze more toward the Value-Aware robot. Additionally, the Value-Aware robot was perceived as more loyal, suggesting that value awareness in a social robot may enhance its perceived commitment to the group. Finally, when both robots disagreed with the participant, conformity occurred in about one out of four trials, and participants took longer to confirm their responses, suggesting that two robots expressing dissent may introduce hesitation in decision-making. On one hand, this highlights the potential risk that robots, if misused, could manipulate users for unethical purposes. On the other hand, it reinforces the idea that social robots could encourage reflection in ambiguous situations and help users avoid scams.
>
---
#### [replaced 012] Curvature-Aware Calibration of Tactile Sensors for Accurate Force Estimation on Non-Planar Surfaces
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.25965v2](http://arxiv.org/pdf/2510.25965v2)**

> **作者:** Luoyan Zhong; Heather Jin Hee Kim; Dylan P. Losey; Cara M. Nunez
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Flexible tactile sensors are increasingly used in real-world applications such as robotic grippers, prosthetic hands, wearable gloves, and assistive devices, where they need to conform to curved and irregular surfaces. However, most existing tactile sensors are calibrated only on flat substrates, and their accuracy and consistency degrade once mounted on curved geometries. This limitation restricts their reliability in practical use. To address this challenge, we develop a calibration model for a widely used resistive tactile sensor design that enables accurate force estimation on one-dimensional curved surfaces. We then train a neural network (a multilayer perceptron) to predict local curvature from baseline sensor outputs recorded under no applied load, achieving an R2 score of 0.91. The proposed approach is validated on five daily objects with varying curvatures under forces from 2 N to 8 N. Results show that the curvature-aware calibration maintains consistent force accuracy across all surfaces, while flat-surface calibration underestimates force as curvature increases. Our results demonstrate that curvature-aware modeling improves the accuracy, consistency, and reliability of flexible tactile sensors, enabling dependable performance across real-world applications.
>
---
#### [replaced 013] Infinite-Horizon Value Function Approximation for Model Predictive Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.06760v2](http://arxiv.org/pdf/2502.06760v2)**

> **作者:** Armand Jordana; Sébastien Kleff; Arthur Haffemayer; Joaquim Ortiz-Haro; Justin Carpentier; Nicolas Mansard; Ludovic Righetti
>
> **摘要:** Model Predictive Control has emerged as a popular tool for robots to generate complex motions. However, the real-time requirement has limited the use of hard constraints and large preview horizons, which are necessary to ensure safety and stability. In practice, practitioners have to carefully design cost functions that can imitate an infinite horizon formulation, which is tedious and often results in local minima. In this work, we study how to approximate the infinite horizon value function of constrained optimal control problems with neural networks using value iteration and trajectory optimization. Furthermore, we experimentally demonstrate how using this value function approximation as a terminal cost provides global stability to the model predictive controller. The approach is validated on two toy problems and a real-world scenario with online obstacle avoidance on an industrial manipulator where the value function is conditioned to the goal and obstacle.
>
---
#### [replaced 014] MindJourney: Test-Time Scaling with World Models for Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12508v2](http://arxiv.org/pdf/2507.12508v2)**

> **作者:** Yuncong Yang; Jiageng Liu; Zheyuan Zhang; Siyuan Zhou; Reuben Tan; Jianwei Yang; Yilun Du; Chuang Gan
>
> **备注:** Project Page: https://umass-embodied-agi.github.io/MindJourney
>
> **摘要:** Spatial reasoning in 3D space is central to human cognition and indispensable for embodied tasks such as navigation and manipulation. However, state-of-the-art vision-language models (VLMs) struggle frequently with tasks as simple as anticipating how a scene will look after an egocentric motion: they perceive 2D images but lack an internal model of 3D dynamics. We therefore propose MindJourney, a test-time scaling framework that grants a VLM with this missing capability by coupling it to a controllable world model based on video diffusion. The VLM iteratively sketches a concise camera trajectory, while the world model synthesizes the corresponding view at each step. The VLM then reasons over this multi-view evidence gathered during the interactive exploration. Without any fine-tuning, our MindJourney achieves over an average 7.7% performance boost on the representative spatial reasoning benchmark SAT, showing that pairing VLMs with world models for test-time scaling offers a simple, plug-and-play route to robust 3D reasoning. Meanwhile, our method also improves upon the test-time inference VLMs trained through reinforcement learning, which demonstrates the potential of our method that utilizes world models for test-time scaling.
>
---
#### [replaced 015] Dropping the D: RGB-D SLAM Without the Depth Sensor
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.06216v2](http://arxiv.org/pdf/2510.06216v2)**

> **作者:** Mert Kiray; Alican Karaomer; Benjamin Busam
>
> **摘要:** We present DropD-SLAM, a real-time monocular SLAM system that achieves RGB-D-level accuracy without relying on depth sensors. The system replaces active depth input with three pretrained vision modules: a monocular metric depth estimator, a learned keypoint detector, and an instance segmentation network. Dynamic objects are suppressed using dilated instance masks, while static keypoints are assigned predicted depth values and backprojected into 3D to form metrically scaled features. These are processed by an unmodified RGB-D SLAM back end for tracking and mapping. On the TUM RGB-D benchmark, DropD-SLAM attains 7.4 cm mean ATE on static sequences and 1.8 cm on dynamic sequences, matching or surpassing state-of-the-art RGB-D methods while operating at 22 FPS on a single GPU. These results suggest that modern pretrained vision models can replace active depth sensors as reliable, real-time sources of metric scale, marking a step toward simpler and more cost-effective SLAM systems.
>
---
#### [replaced 016] MOSPA: Human Motion Generation Driven by Spatial Audio
- **分类: cs.GR; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.11949v2](http://arxiv.org/pdf/2507.11949v2)**

> **作者:** Shuyang Xu; Zhiyang Dou; Mingyi Shi; Liang Pan; Leo Ho; Jingbo Wang; Yuan Liu; Cheng Lin; Yuexin Ma; Wenping Wang; Taku Komura
>
> **备注:** NeurIPS 2025 (Spotlight)
>
> **摘要:** Enabling virtual humans to dynamically and realistically respond to diverse auditory stimuli remains a key challenge in character animation, demanding the integration of perceptual modeling and motion synthesis. Despite its significance, this task remains largely unexplored. Most previous works have primarily focused on mapping modalities like speech, audio, and music to generate human motion. As of yet, these models typically overlook the impact of spatial features encoded in spatial audio signals on human motion. To bridge this gap and enable high-quality modeling of human movements in response to spatial audio, we introduce the first comprehensive Spatial Audio-Driven Human Motion (SAM) dataset, which contains diverse and high-quality spatial audio and motion data. For benchmarking, we develop a simple yet effective diffusion-based generative framework for human MOtion generation driven by SPatial Audio, termed MOSPA, which faithfully captures the relationship between body motion and spatial audio through an effective fusion mechanism. Once trained, MOSPA can generate diverse, realistic human motions conditioned on varying spatial audio inputs. We perform a thorough investigation of the proposed dataset and conduct extensive experiments for benchmarking, where our method achieves state-of-the-art performance on this task. Our code and model are publicly available at https://github.com/xsy27/Mospa-Acoustic-driven-Motion-Generation
>
---
#### [replaced 017] A Time-dependent Risk-aware distributed Multi-Agent Path Finder based on A*
- **分类: cs.RO; 68T40**

- **链接: [http://arxiv.org/pdf/2504.19593v2](http://arxiv.org/pdf/2504.19593v2)**

> **作者:** S Nordström; Y Bai; B Lindqvist; G Nikolakopoulos
>
> **备注:** 8 pages, 10 figures, 2 tabels, submited to IROS 2025
>
> **摘要:** Multi-Agent Path-Finding (MAPF) focuses on the collaborative planning of paths for multiple agents within shared spaces, aiming for collision-free navigation. Conventional planning methods often overlook the presence of other agents, which can result in conflicts. In response, this article introduces the A$^*_+$T algorithm, a distributed approach that improves coordination among agents by anticipating their positions based on their movement speeds. The algorithm also considers dynamic obstacles, assessing potential collisions with respect to observed speeds and trajectories, thereby facilitating collision-free path planning in environments populated by other agents and moving objects. It incorporates a risk layer surrounding both dynamic and static entities, enhancing its utility in real-world applications. Each agent functions autonomously while being mindful of the paths chosen by others, effectively addressing the complexities inherent in multi-agent situations. The performance of A$^*_+$T has been rigorously tested in the Gazebo simulation environment and benchmarked against established approaches such as CBS, ECBS, and SIPP. Furthermore, the algorithm has shown competence in single-agent experiments, with results demonstrating its effectiveness in managing dynamic obstacles and affirming its practical relevance across various scenarios.
>
---
#### [replaced 018] Mixed-Density Diffuser: Efficient Planning with Non-uniform Temporal Resolution
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.23026v2](http://arxiv.org/pdf/2510.23026v2)**

> **作者:** Crimson Stambaugh; Rajesh P. N. Rao
>
> **备注:** European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESSAN) (under review)
>
> **摘要:** Recent studies demonstrate that diffusion planners benefit from sparse-step planning over single-step planning. Training models to skip steps in their trajectories helps capture long-term dependencies without additional or memory computational cost. However, predicting excessively sparse plans degrades performance. We hypothesize this temporal density threshold is non-uniform across a temporal horizon and that certain parts of a planned trajectory should be more densely planned. We propose Mixed Density Diffuser (MDD), a diffusion planner where the densities throughout the horizon are tunable hyperparameters. MDD achieves a new SOTA across the Maze2D, Franka Kitchen, and Antmaze D4RL task domains.
>
---
#### [replaced 019] iKap: Kinematics-aware Planning with Imperative Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.09496v3](http://arxiv.org/pdf/2412.09496v3)**

> **作者:** Qihang Li; Zhuoqun Chen; Haoze Zheng; Haonan He; Zitong Zhan; Shaoshu Su; Junyi Geng; Chen Wang
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** Trajectory planning in robotics aims to generate collision-free pose sequences that can be reliably executed. Recently, vision-to-planning systems have gained increasing attention for their efficiency and ability to interpret and adapt to surrounding environments. However, traditional modular systems suffer from increased latency and error propagation, while purely data-driven approaches often overlook the robot's kinematic constraints. This oversight leads to discrepancies between planned trajectories and those that are executable. To address these challenges, we propose iKap, a novel vision-to-planning system that integrates the robot's kinematic model directly into the learning pipeline. iKap employs a self-supervised learning approach and incorporates the state transition model within a differentiable bi-level optimization framework. This integration ensures the network learns collision-free waypoints while satisfying kinematic constraints, enabling gradient back-propagation for end-to-end training. Our experimental results demonstrate that iKap achieves higher success rates and reduced latency compared to the state-of-the-art methods. Besides the complete system, iKap offers a visual-to-planning network that seamlessly works with various controllers, providing a robust solution for robots navigating complex environments.
>
---
#### [replaced 020] RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.14830v2](http://arxiv.org/pdf/2510.14830v2)**

> **作者:** Kun Lei; Huanyu Li; Dongjie Yu; Zhenyu Wei; Lingxiao Guo; Zhennan Jiang; Ziyu Wang; Shiyu Liang; Huazhe Xu
>
> **备注:** https://lei-kun.github.io/RL-100/
>
> **摘要:** Real-world robotic manipulation in homes and factories demands reliability, efficiency, and robustness that approach or surpass skilled human operators. We present RL-100, a real-world reinforcement learning training framework built on diffusion visuomotor policies trained by supervised learning. RL-100 introduces a three-stage pipeline. First, imitation learning leverages human priors. Second, iterative offline reinforcement learning uses an Offline Policy Evaluation procedure, abbreviated OPE, to gate PPO-style updates that are applied in the denoising process for conservative and reliable improvement. Third, online reinforcement learning eliminates residual failure modes. An additional lightweight consistency distillation head compresses the multi-step sampling process in diffusion into a single-step policy, enabling high-frequency control with an order-of-magnitude reduction in latency while preserving task performance. The framework is task-, embodiment-, and representation-agnostic and supports both 3D point clouds and 2D RGB inputs, a variety of robot platforms, and both single-step and action-chunk policies. We evaluate RL-100 on seven real-robot tasks spanning dynamic rigid-body control, such as Push-T and Agile Bowling, fluids and granular pouring, deformable cloth folding, precise dexterous unscrewing, and multi-stage orange juicing. RL-100 attains 100\% success across evaluated trials for a total of 900 out of 900 episodes, including up to 250 out of 250 consecutive trials on one task. The method achieves near-human teleoperation or better time efficiency and demonstrates multi-hour robustness with uninterrupted operation lasting up to two hours.
>
---
#### [replaced 021] Spatiotemporal Calibration for Laser Vision Sensor in Hand-eye System Based on Straight-line Constraint
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.12928v2](http://arxiv.org/pdf/2509.12928v2)**

> **作者:** Peiwen Yang; Mingquan Jiang; Xinyue Shen; Heping Zhang
>
> **备注:** Submitted to IEEE RAL
>
> **摘要:** Laser vision sensors (LVS) are critical perception modules for industrial robots, facilitating real-time acquisition of workpiece geometric data in welding applications. However, the camera communication delay will lead to a temporal desynchronization between captured images and the robot motions. Additionally, hand-eye extrinsic parameters may vary during prolonged measurement. To address these issues, we introduce a measurement model of LVS considering the effect of the camera's time-offset and propose a teaching-free spatiotemporal calibration method utilizing line constraints. This method involves a robot equipped with an LVS repeatedly scanning straight-line fillet welds using S-shaped trajectories. Regardless of the robot's orientation changes, all measured welding positions are constrained to a straight-line, represented by Plucker coordinates. Moreover, a nonlinear optimization model based on straight-line constraints is established. Subsequently, the Levenberg-Marquardt algorithm (LMA) is employed to optimize parameters, including time-offset, hand-eye extrinsic parameters, and straight-line parameters. The feasibility and accuracy of the proposed approach are quantitatively validated through experiments on curved weld scanning. We open-sourced the code, dataset, and simulation report at https://anonymous.4open.science/r/LVS_ST_CALIB-015F/README.md.
>
---
#### [replaced 022] Dual-Regularized Riccati Recursions for Interior-Point Optimal Control
- **分类: math.OC; cs.MS; cs.RO; cs.SY; eess.SY; 49M37, 90C51, 93B45; G.1.6**

- **链接: [http://arxiv.org/pdf/2509.16370v4](http://arxiv.org/pdf/2509.16370v4)**

> **作者:** João Sousa-Pinto; Dominique Orban
>
> **摘要:** We derive closed-form extensions of Riccati's recursions (both sequential and parallel) for solving dual-regularized LQR problems. We show how these methods can be used to solve general constrained, non-convex, discrete-time optimal control problems via a regularized interior point method, while guaranteeing that each primal step is a descent direction of an Augmented Barrier-Lagrangian merit function. We provide MIT-licensed implementations of our methods in C++ and JAX.
>
---
#### [replaced 023] From Grounding to Manipulation: Case Studies of Foundation Model Integration in Embodied Robotic Systems
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.15685v2](http://arxiv.org/pdf/2505.15685v2)**

> **作者:** Xiuchao Sui; Daiying Tian; Qi Sun; Ruirui Chen; Dongkyu Choi; Kenneth Kwok; Soujanya Poria
>
> **备注:** EMNLP 2025 camera ready
>
> **摘要:** Foundation models (FMs) are increasingly used to bridge language and action in embodied agents, yet the operational characteristics of different FM integration strategies remain under-explored -- particularly for complex instruction following and versatile action generation in changing environments. This paper examines three paradigms for building robotic systems: end-to-end vision-language-action (VLA) models that implicitly integrate perception and planning, and modular pipelines incorporating either vision-language models (VLMs) or multimodal large language models (LLMs). We evaluate these paradigms through two focused case studies: a complex instruction grounding task assessing fine-grained instruction understanding and cross-modal disambiguation, and an object manipulation task targeting skill transfer via VLA finetuning. Our experiments in zero-shot and few-shot settings reveal trade-offs in generalization and data efficiency. By exploring performance limits, we distill design implications for developing language-driven physical agents and outline emerging challenges and opportunities for FM-powered robotics in real-world conditions.
>
---
#### [replaced 024] Beyond the Uncanny Valley: A Mixed-Method Investigation of Anthropomorphism in Protective Responses to Robot Abuse
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.26082v2](http://arxiv.org/pdf/2510.26082v2)**

> **作者:** Fan Yang; Lingyao Li; Yaxin Hu; Michael Rodgers; Renkai Ma
>
> **摘要:** Robots with anthropomorphic features are increasingly shaping how humans perceive and morally engage with them. Our research investigates how different levels of anthropomorphism influence protective responses to robot abuse, extending the Computers as Social Actors (CASA) and uncanny valley theories into a moral domain. In an experiment, we invite 201 participants to view videos depicting abuse toward a robot with low (Spider), moderate (Two-Foot), or high (Humanoid) anthropomorphism. To provide a comprehensive analysis, we triangulate three modalities: self-report surveys measuring emotions and uncanniness, physiological data from automated facial expression analysis, and qualitative reflections. Findings indicate that protective responses are not linear. The moderately anthropomorphic Two-Foot robot, rated highest in eeriness and "spine-tingling" sensations consistent with the uncanny valley, elicited the strongest physiological anger expressions. Self-reported anger and guilt are significantly higher for both the Two-Foot and Humanoid robots compared to the Spider. Qualitative findings further reveal that as anthropomorphism increases, moral reasoning shifts from technical assessments of property damage to condemnation of the abuser's character, while governance proposals expand from property law to calls for quasi-animal rights and broader societal responsibility. These results suggest that the uncanny valley does not dampen moral concern but paradoxically heightens protective impulses, offering critical implications for robot design, policy, and future legal frameworks.
>
---
#### [replaced 025] Adaptive Multirobot Virtual Structure Control using Dual Quaternions
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.05560v2](http://arxiv.org/pdf/2504.05560v2)**

> **作者:** Juan I. Giribet; Alejandro S. Ghersin; Ignacio Mas; Harrison Neves Marciano; Daniel Khede Dourado Villa; Mario Sarcinelli-Filho
>
> **摘要:** This paper presents a control strategy based on dual quaternions for the coordinated formation flying of small UAV groups. A virtual structure is employed to define the desired formation, enabling unified control of its position, orientation, and shape. This abstraction makes formation management easier by allowing a low-level controller to compute individual UAV commands efficiently. The proposed controller integrates a pose control module with a geometry-based adaptive strategy, ensuring precise and robust task execution. The effectiveness of the approach is demonstrated through both simulation and experimental results.
>
---
#### [replaced 026] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.24948v3](http://arxiv.org/pdf/2509.24948v3)**

> **作者:** Junjin Xiao; Yandan Yang; Xinyuan Chang; Ronghan Chen; Feng Xiong; Mu Xu; Wei-Shi Zheng; Qing Zhang
>
> **摘要:** Vision-Language-Action (VLA) models trained via imitation learning suffer from significant performance degradation in data-scarce scenarios due to their reliance on large-scale demonstration datasets. Although reinforcement learning (RL)-based post-training has proven effective in addressing data scarcity, its application to VLA models is hindered by the non-resettable nature of real-world environments. This limitation is particularly critical in high-risk domains such as industrial automation, where interactions often induce state changes that are costly or infeasible to revert. Furthermore, existing VLA approaches lack a reliable mechanism for detecting task completion, leading to redundant actions that reduce overall task success rates. To address these challenges, we propose World-Env, an RL-based post-training framework that replaces physical interaction with a low-cost, world model-based virtual simulator. World-Env consists of two key components: (1) a video-based world simulator that generates temporally consistent future visual observations, and (2) a vision-language model (VLM)-guided instant reflector that provides continuous reward signals and predicts action termination. This simulated environment enables VLA models to safely explore and generalize beyond their initial imitation learning distribution. Our method achieves notable performance gains with as few as five expert demonstrations per task. Experiments on complex robotic manipulation tasks demonstrate that World-Env effectively overcomes the data inefficiency, safety constraints, and inefficient execution of conventional VLA models that rely on real-world interaction, offering a practical and scalable solution for post-training in resource-constrained settings. Our code is available at https://github.com/amap-cvlab/world-env.
>
---
#### [replaced 027] UniVLA: Learning to Act Anywhere with Task-centric Latent Actions
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.06111v3](http://arxiv.org/pdf/2505.06111v3)**

> **作者:** Qingwen Bu; Yanting Yang; Jisong Cai; Shenyuan Gao; Guanghui Ren; Maoqing Yao; Ping Luo; Hongyang Li
>
> **备注:** Accepted to RSS 2025. Code is available at https://github.com/OpenDriveLab/UniVLA
>
> **摘要:** A generalist robot should perform effectively across various environments. However, most existing approaches heavily rely on scaling action-annotated data to enhance their capabilities. Consequently, they are often limited to single physical specification and struggle to learn transferable knowledge across different embodiments and environments. To confront these limitations, we propose UniVLA, a new framework for learning cross-embodiment vision-language-action (VLA) policies. Our key innovation is to derive task-centric action representations from videos with a latent action model. This enables us to exploit extensive data across a wide spectrum of embodiments and perspectives. To mitigate the effect of task-irrelevant dynamics, we incorporate language instructions and establish a latent action model within the DINO feature space. Learned from internet-scale videos, the generalist policy can be deployed to various robots through efficient latent action decoding. We obtain state-of-the-art results across multiple manipulation and navigation benchmarks, as well as real-robot deployments. UniVLA achieves superior performance over OpenVLA with less than 1/20 of pretraining compute and 1/10 of downstream data. Continuous performance improvements are observed as heterogeneous data, even including human videos, are incorporated into the training pipeline. The results underscore UniVLA's potential to facilitate scalable and efficient robot policy learning.
>
---
#### [replaced 028] Kinematically Controllable Cable Robots with Reconfigurable End-effectors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.22825v2](http://arxiv.org/pdf/2510.22825v2)**

> **作者:** Nan Zhang
>
> **备注:** 8 pages, 7 figures, Technical Report
>
> **摘要:** To enlarge the translational workspace of cable-driven robots, one common approach is to increase the number of cables. However, this introduces two challenges: (1) cable interference significantly reduces the rotational workspace, and (2) the solution of tensions in cables becomes non-unique, resulting in difficulties for kinematic control of the robot. In this work, we design structurally simple reconfigurable end-effectors for cable robots. By incorporating a spring, a helical-grooved shaft, and a matching nut, relative linear motions between end-effector components are converted into relative rotations, thereby expanding the rotational workspace of the mechanism. Meanwhile, a bearing is introduced to provide an additional rotational degree of freedom, making the mechanism non-redundant. As a result, the robot's motion can be controlled purely through kinematics without additional tension sensing and control.
>
---
#### [replaced 029] Dexterous Contact-Rich Manipulation via the Contact Trust Region
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.02291v4](http://arxiv.org/pdf/2505.02291v4)**

> **作者:** H. J. Terry Suh; Tao Pang; Tong Zhao; Russ Tedrake
>
> **摘要:** What is a good local description of contact dynamics for contact-rich manipulation, and where can we trust this local description? While many approaches often rely on the Taylor approximation of dynamics with an ellipsoidal trust region, we argue that such approaches are fundamentally inconsistent with the unilateral nature of contact. As a remedy, we present the Contact Trust Region (CTR), which captures the unilateral nature of contact while remaining efficient for computation. With CTR, we first develop a Model-Predictive Control (MPC) algorithm capable of synthesizing local contact-rich plans. Then, we extend this capability to plan globally by stitching together local MPC plans, enabling efficient and dexterous contact-rich manipulation. To verify the performance of our method, we perform comprehensive evaluations, both in high-fidelity simulation and on hardware, on two contact-rich systems: a planar IiwaBimanual system and a 3D AllegroHand system. On both systems, our method offers a significantly lower-compute alternative to existing RL-based approaches to contact-rich manipulation. In particular, our Allegro in-hand manipulation policy, in the form of a roadmap, takes fewer than 10 minutes to build offline on a standard laptop using just its CPU, with online inference taking just a few seconds. Experiment data, video and code are available at ctr.theaiinstitute.com.
>
---
#### [replaced 030] Robotic Monitoring of Colorimetric Leaf Sensors for Precision Agriculture
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.13916v3](http://arxiv.org/pdf/2505.13916v3)**

> **作者:** Malakhi Hopkins; Alice Kate Li; Shobhita Kramadhati; Jackson Arnold; Akhila Mallavarapu; Chavez F. K. Lawrence; Anish Bhattacharya; Varun Murali; Sanjeev J. Koppal; Cherie R. Kagan; Vijay Kumar
>
> **备注:** Revised version. Initial version was accepted to the Novel Approaches for Precision Agriculture and Forestry with Autonomous Robots IEEE ICRA Workshop - 2025
>
> **摘要:** Common remote sensing modalities (RGB, multispectral, hyperspectral imaging or LiDAR) are often used to indirectly measure crop health and do not directly capture plant stress indicators. Commercially available direct leaf sensors are bulky, powered electronics that are expensive and interfere with crop growth. In contrast, low-cost, passive and bio-degradable leaf sensors offer an opportunity to advance real-time monitoring as they directly interface with the crop surface while not interfering with crop growth. To this end, we co-design a sensor-detector system, where the sensor is a passive colorimetric leaf sensor that directly measures crop health in a precision agriculture setting, and the detector autonomously obtains optical signals from these leaf sensors. The detector comprises a low size weight and power (SWaP) mobile ground robot with an onboard monocular RGB camera and object detector to localize each leaf sensor, as well as a hyperspectral camera with a motorized mirror and halogen light to acquire hyperspectral images. The sensor's crop health-dependent optical signals can be extracted from the hyperspectral images. The proof-of-concept system is demonstrated in row-crop environments both indoors and outdoors where it is able to autonomously navigate, locate and obtain a hyperspectral image of all leaf sensors present, and acquire interpretable spectral resonance with 80 $\%$ accuracy within a required retrieval distance from the sensor.
>
---
#### [replaced 031] DTAA: A Detect, Track and Avoid Architecture for navigation in spaces with Multiple Velocity Objects
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.08121v2](http://arxiv.org/pdf/2412.08121v2)**

> **作者:** Samuel Nordström; Björn Lindquist; George Nikolakopoulos
>
> **摘要:** Proactive collision avoidance measures are imperative in environments where humans and robots coexist. Moreover, the introduction of high quality legged robots into workplaces highlighted the crucial role of a robust, fully autonomous safety solution for robots to be viable in shared spaces or in co-existence with humans. This article establishes for the first time ever an innovative Detect-Track-and-Avoid Architecture (DTAA) to enhance safety and overall mission performance. The proposed novel architectyre has the merit ot integrating object detection using YOLOv8, utilizing Ultralytics embedded object tracking, and state estimation of tracked objects through Kalman filters. Moreover, a novel heuristic clustering is employed to facilitate active avoidance of multiple closely positioned objects with similar velocities, creating sets of unsafe spaces for the Nonlinear Model Predictive Controller (NMPC) to navigate around. The NMPC identifies the most hazardous unsafe space, considering not only their current positions but also their predicted future locations. In the sequel, the NMPC calculates maneuvers to guide the robot along a path planned by D$^{*}_{+}$ towards its intended destination, while maintaining a safe distance to all identified obstacles. The efficacy of the novelly suggested DTAA framework is being validated by Real-life experiments featuring a Boston Dynamics Spot robot that demonstrates the robot's capability to consistently maintain a safe distance from humans in dynamic subterranean, urban indoor, and outdoor environments.
>
---
#### [replaced 032] Event-RGB Fusion for Spacecraft Pose Estimation Under Harsh Lighting
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05698v3](http://arxiv.org/pdf/2507.05698v3)**

> **作者:** Mohsi Jawaid; Marcus Märtens; Tat-Jun Chin
>
> **备注:** Associated dataset: https://zenodo.org/records/15861758
>
> **摘要:** Spacecraft pose estimation is crucial for autonomous in-space operations, such as rendezvous, docking and on-orbit servicing. Vision-based pose estimation methods, which typically employ RGB imaging sensors, is a compelling solution for spacecraft pose estimation, but are challenged by harsh lighting conditions, which produce imaging artifacts such as glare, over-exposure, blooming and lens flare. Due to their much higher dynamic range, neuromorphic or event sensors are more resilient to extreme lighting conditions. However, event sensors generally have lower spatial resolution and suffer from reduced signal-to-noise ratio during periods of low relative motion. This work addresses these individual sensor limitations by introducing a sensor fusion approach combining RGB and event sensors. A beam-splitter prism was employed to achieve precise optical and temporal alignment. Then, a RANSAC-based technique was developed to fuse the information from the RGB and event channels to achieve pose estimation that leveraged the strengths of the two modalities. The pipeline was complemented by dropout uncertainty estimation to detect extreme conditions that affect either channel. To benchmark the performance of the proposed event-RGB fusion method, we collected a comprehensive real dataset of RGB and event data for satellite pose estimation in a laboratory setting under a variety of challenging illumination conditions. Encouraging results on the dataset demonstrate the efficacy of our event-RGB fusion approach and further supports the usage of event sensors for spacecraft pose estimation. To support community research on this topic, our dataset has been released publicly.
>
---
#### [replaced 033] VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.15530v4](http://arxiv.org/pdf/2510.15530v4)**

> **作者:** Zehao Ni; Yonghao He; Lingfeng Qian; Jilei Mao; Fa Fu; Wei Sui; Hu Su; Junran Peng; Zhipeng Wang; Bin He
>
> **摘要:** In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.
>
---
#### [replaced 034] MARFT: Multi-Agent Reinforcement Fine-Tuning
- **分类: cs.MA; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.16129v4](http://arxiv.org/pdf/2504.16129v4)**

> **作者:** Junwei Liao; Muning Wen; Jun Wang; Weinan Zhang
>
> **备注:** 42 pages
>
> **摘要:** LLM-based Multi-Agent Systems have demonstrated remarkable capabilities in addressing complex, agentic tasks, from generating high-quality presentation slides to even conducting sophisticated scientific research. Meanwhile, RL has been widely recognized for its effectiveness in enhancing agent intelligence, but limited research has investigated the fine-tuning of LaMAS using foundational RL techniques. Moreover, the direct application of MARL methods to LaMAS introduces significant challenges, stemming from the unique characteristics and mechanisms inherent to LaMAS. To address these challenges, this article presents a comprehensive study of LLM-based MARL and proposes a novel paradigm termed Multi-Agent Reinforcement Fine-Tuning (MARFT). We introduce a brand-new MG called Flex-MG, which aligns with the LaMAS optimization in real-world applications and a universal algorithmic framework tailored specifically for LaMAS, outlining the conceptual foundations, key distinctions, and practical implementation strategies. We review the evolution from RL to RFT, setting the stage for a parallel analysis in the multi-agent domain. In the context of LaMAS, we elucidate critical differences between MARL and MARFT. These differences motivate a transition toward a LaMAS-oriented formulation of RFT. Central to this work is a robust and scalable MARFT framework. We detail the core algorithm and provide a complete, open-source implementation to facilitate adoption and further research. The latter sections of the paper explore real-world application perspectives and opening challenges in MARFT. By bridging theoretical underpinnings with practical methodologies, this work serves as a roadmap for researchers seeking to advance MARFT toward resilient and adaptive solutions in agentic systems. Our implementation of the proposed framework is publicly available at: https://github.com/jwliao-ai/MARFT.
>
---
#### [replaced 035] LPAC: Learnable Perception-Action-Communication Loops with Applications to Coverage Control
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2401.04855v5](http://arxiv.org/pdf/2401.04855v5)**

> **作者:** Saurav Agarwal; Ramya Muthukrishnan; Walker Gosrich; Vijay Kumar; Alejandro Ribeiro
>
> **备注:** 20 Pages, 20 figures,
>
> **摘要:** Coverage control is the problem of navigating a robot swarm to collaboratively monitor features or a phenomenon of interest not known a priori. The problem is challenging in decentralized settings with robots that have limited communication and sensing capabilities. We propose a learnable Perception-Action-Communication (LPAC) architecture for the problem, wherein a convolutional neural network (CNN) processes localized perception; a graph neural network (GNN) facilitates robot communications; finally, a shallow multi-layer perceptron (MLP) computes robot actions. The GNN enables collaboration in the robot swarm by computing what information to communicate with nearby robots and how to incorporate received information. Evaluations show that the LPAC models -- trained using imitation learning -- outperform standard decentralized and centralized coverage control algorithms. The learned policy generalizes to environments different from the training dataset, transfers to larger environments with more robots, and is robust to noisy position estimates. The results indicate the suitability of LPAC architectures for decentralized navigation in robot swarms to achieve collaborative behavior.
>
---
#### [replaced 036] A Helping (Human) Hand in Kinematic Structure Estimation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.05301v2](http://arxiv.org/pdf/2503.05301v2)**

> **作者:** Adrian Pfisterer; Xing Li; Vito Mengers; Oliver Brock
>
> **备注:** Accepted at ICRA25; 8 pages + 7 figures; For supplementary material, see https://www.tu.berlin/robotics/papers/helpinghands
>
> **摘要:** Visual uncertainties such as occlusions, lack of texture, and noise present significant challenges in obtaining accurate kinematic models for safe robotic manipulation. We introduce a probabilistic real-time approach that leverages the human hand as a prior to mitigate these uncertainties. By tracking the constrained motion of the human hand during manipulation and explicitly modeling uncertainties in visual observations, our method reliably estimates an object's kinematic model online. We validate our approach on a novel dataset featuring challenging objects that are occluded during manipulation and offer limited articulations for perception. The results demonstrate that by incorporating an appropriate prior and explicitly accounting for uncertainties, our method produces accurate estimates, outperforming two recent baselines by 195% and 140%, respectively. Furthermore, we demonstrate that our approach's estimates are precise enough to allow a robot to manipulate even small objects safely.
>
---
#### [replaced 037] ReactEMG: Zero-Shot, Low-Latency Intent Detection via sEMG
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19815v5](http://arxiv.org/pdf/2506.19815v5)**

> **作者:** Runsheng Wang; Xinyue Zhu; Ava Chen; Jingxi Xu; Lauren Winterbottom; Dawn M. Nilsen; Joel Stein; Matei Ciocarlie
>
> **摘要:** Surface electromyography (sEMG) signals show promise for effective human-computer interfaces, particularly in rehabilitation and prosthetics. However, challenges remain in developing systems that respond quickly and reliably to user intent, across different subjects and without requiring time-consuming calibration. In this work, we propose a framework for EMG-based intent detection that addresses these challenges. Unlike traditional gesture recognition models that wait until a gesture is completed before classifying it, our approach uses a segmentation strategy to assign intent labels at every timestep as the gesture unfolds. We introduce a novel masked modeling strategy that aligns muscle activations with their corresponding user intents, enabling rapid onset detection and stable tracking of ongoing gestures. In evaluations against baseline methods, considering both accuracy and stability for device control, our approach surpasses state-of-the-art performance in zero-shot transfer conditions, demonstrating its potential for wearable robotics and next-generation prosthetic systems. Our project page is available at: https://reactemg.github.io
>
---
