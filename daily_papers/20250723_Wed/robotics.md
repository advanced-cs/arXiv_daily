# 机器人 cs.RO

- **最新发布 28 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Equivariant Goal Conditioned Contrastive Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 论文提出了一种新的对比强化学习方法——Equivariant CRL（ECRL），用于解决目标条件操作任务中的样本效率和空间泛化问题。通过引入等变约束，结构化潜在空间，结合旋转不变的评论器和等变的执行器，提升性能。方法在多种模拟任务中优于基线模型，并扩展至离线RL场景。**

- **链接: [http://arxiv.org/pdf/2507.16139v1](http://arxiv.org/pdf/2507.16139v1)**

> **作者:** Arsh Tangri; Nichols Crawford Taylor; Haojie Huang; Robert Platt
>
> **摘要:** Contrastive Reinforcement Learning (CRL) provides a promising framework for extracting useful structured representations from unlabeled interactions. By pulling together state-action pairs and their corresponding future states, while pushing apart negative pairs, CRL enables learning nontrivial policies without manually designed rewards. In this work, we propose Equivariant CRL (ECRL), which further structures the latent space using equivariant constraints. By leveraging inherent symmetries in goal-conditioned manipulation tasks, our method improves both sample efficiency and spatial generalization. Specifically, we formally define Goal-Conditioned Group-Invariant MDPs to characterize rotation-symmetric robotic manipulation tasks, and build on this by introducing a novel rotation-invariant critic representation paired with a rotation-equivariant actor for Contrastive RL. Our approach consistently outperforms strong baselines across a range of simulated tasks in both state-based and image-based settings. Finally, we extend our method to the offline RL setting, demonstrating its effectiveness across multiple tasks.
>
---
#### [new 002] Compositional Coordination for Multi-Robot Teams with Large Language Models
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于多机器人协调任务，旨在解决传统方法依赖专家手动翻译自然语言任务到代码的问题。论文提出LAN2CB框架，利用大语言模型将自然语言直接转化为可执行代码，包含任务分解与代码生成两个核心组件，并构建了相关数据集。实验表明该方法减少了人工工程量，提升了任务适应性与灵活性。**

- **链接: [http://arxiv.org/pdf/2507.16068v1](http://arxiv.org/pdf/2507.16068v1)**

> **作者:** Zhehui Huang; Guangyao Shi; Yuwei Wu; Vijay Kumar; Gaurav S. Sukhatme
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Multi-robot coordination has traditionally relied on a task-specific and expert-driven pipeline, where natural language mission descriptions are manually translated by domain experts into mathematical formulation, algorithm design, and executable code. This conventional process is labor-intensive, inaccessible to non-experts, and inflexible to changes in mission requirements. Here, we propose LAN2CB (Language to Collective Behavior), a novel framework that leverages large language models (LLMs) to streamline and generalize the multi-robot coordination pipeline. LAN2CB directly converts natural language mission descriptions into executable Python code for multi-robot systems through two key components: (1) Mission Decomposition for Task Representation, which parses the mission into a task graph with dependencies, and (2) Code Generation, which uses the task graph and a structured knowledge base to generate deployable robot control code. We further introduce a dataset of natural language mission specifications to support development and benchmarking. Experimental results in both simulation and real-world settings show that LAN2CB enables effective and flexible multi-robot coordination from natural language, significantly reducing the need for manual engineering while supporting generalization across mission types. Website: https://sites.google.com/view/lan2cb.
>
---
#### [new 003] Therapist-Exoskeleton-Patient Interaction: An Immersive Gait Therapy
- **分类: cs.RO**

- **简介: 该论文属于医疗康复任务，旨在解决中风患者步态康复治疗中治疗师体力消耗大、难以多关节协同的问题。论文提出了一种基于物理人-机器人-人交互（pHRHI）的新型步态康复范式，通过连接治疗师与患者的下肢外骨骼，实现双向互动与力反馈。实验表明，该方法优于传统训练方式，提升了关节活动度、步态指标及患者积极性。**

- **链接: [http://arxiv.org/pdf/2507.16059v1](http://arxiv.org/pdf/2507.16059v1)**

> **作者:** Emek Barış Küçüktabak; Matthew R. Short; Lorenzo Vianello; Daniel Ludvig; Levi Hargrove; Kevin Lynch; Jose Pons
>
> **摘要:** Following a stroke, individuals often experience mobility and balance impairments due to lower-limb weakness and loss of independent joint control. Gait recovery is a key goal of rehabilitation, traditionally achieved through high-intensity therapist-led training. However, manual assistance can be physically demanding and limits the therapist's ability to interact with multiple joints simultaneously. Robotic exoskeletons offer multi-joint support, reduce therapist strain, and provide objective feedback, but current control strategies often limit therapist involvement and adaptability. We present a novel gait rehabilitation paradigm based on physical Human-Robot-Human Interaction (pHRHI), where both the therapist and the post-stroke individual wear lower-limb exoskeletons virtually connected at the hips and knees via spring-damper elements. This enables bidirectional interaction, allowing the therapist to guide movement and receive haptic feedback. In a study with eight chronic stroke patients, pHRHI training outperformed conventional therapist-guided treadmill walking, leading to increased joint range of motion, step metrics, muscle activation, and motivation. These results highlight pHRHI's potential to combine robotic precision with therapist intuition for improved rehabilitation outcomes.
>
---
#### [new 004] AI or Human? Understanding Perceptions of Embodied Robots with LLMs
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在探讨人类如何感知具身机器人（embodied robots）的智能来源（AI或人类）。研究通过图灵测试方式，让参与者在两项任务中辨别机器人由AI还是人类控制，发现人们难以准确区分两者，并分析了影响智能感知的关键因素。**

- **链接: [http://arxiv.org/pdf/2507.16398v1](http://arxiv.org/pdf/2507.16398v1)**

> **作者:** Lavinia Hriscu; Alberto Sanfeliu; Anais Garrell
>
> **摘要:** The pursuit of artificial intelligence has long been associated to the the challenge of effectively measuring intelligence. Even if the Turing Test was introduced as a means of assessing a system intelligence, its relevance and application within the field of human-robot interaction remain largely underexplored. This study investigates the perception of intelligence in embodied robots by performing a Turing Test within a robotic platform. A total of 34 participants were tasked with distinguishing between AI- and human-operated robots while engaging in two interactive tasks: an information retrieval and a package handover. These tasks assessed the robot perception and navigation abilities under both static and dynamic conditions. Results indicate that participants were unable to reliably differentiate between AI- and human-controlled robots beyond chance levels. Furthermore, analysis of participant responses reveals key factors influencing the perception of artificial versus human intelligence in embodied robotic systems. These findings provide insights into the design of future interactive robots and contribute to the ongoing discourse on intelligence assessment in AI-driven systems.
>
---
#### [new 005] Design and Dimensional Optimization of Legged Structures for Construction Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人结构设计任务，旨在解决建筑机器人在复杂地形中自主移动能力不足的问题。通过仿生设计与腿结构多维性能优化，提出改进的工作空间和平均操作性指标，结合图形与数值方法优化腿段比例，并通过仿真验证，最终建立适用于建筑环境的腿部结构性能评估框架，为腿式建筑机器人设计提供依据。**

- **链接: [http://arxiv.org/pdf/2507.16328v1](http://arxiv.org/pdf/2507.16328v1)**

> **作者:** Xiao Liu; Xianlong Yang; Weijun Wang; Wei Feng
>
> **摘要:** Faced with complex and unstructured construction environments, wheeled and tracked robots exhibit significant limitations in terrain adaptability and flexibility, making it difficult to meet the requirements of autonomous operation. Inspired by ants in nature, this paper proposes a leg configuration design and optimization method tailored for construction scenarios, aiming to enhance the autonomous mobility of construction robots. This paper analyzes the full operational motion performance of the leg during both swing and stance phases. First, based on kinematic modeling and multi-dimensional workspace analysis, the concept of an "improved workspace" is introduced, and graphical methods are used to optimize the leg dimensions during the swing phase. Furthermore, a new concept of "average manipulability" is introduced based on the velocity Jacobian matrix, and numerical solutions are applied to obtain the leg segment ratio that maximizes manipulability. To overcome the difficulties associated with traditional analytical methods, virtual prototype simulations are conducted in ADAMS to explore the relationship between the robot body's optimal flexibility and leg segment proportions. In summary, the leg segment proportions with the best comprehensive motion performance are obtained. This study presents the first multi-dimensional quantitative evaluation framework for leg motion performance tailored for construction environments, providing a structural design foundation for legged construction robots to achieve autonomous mobility in complex terrains.
>
---
#### [new 006] A Comprehensive Evaluation of LiDAR Odometry Techniques
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，旨在解决如何利用LiDAR传感器实现最精确的状态估计问题。论文系统总结了LiDAR里程计（LO）管道中的各类技术，并在多种数据集上进行了广泛实验评估，提供了基于实证的未来LO管道设计建议。**

- **链接: [http://arxiv.org/pdf/2507.16000v1](http://arxiv.org/pdf/2507.16000v1)**

> **作者:** Easton Potokar; Michael Kaess
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Light Detection and Ranging (LiDAR) sensors have become the sensor of choice for many robotic state estimation tasks. Because of this, in recent years there has been significant work done to fine the most accurate method to perform state estimation using these sensors. In each of these prior works, an explosion of possible technique combinations has occurred, with each work comparing LiDAR Odometry (LO) "pipelines" to prior "pipelines". Unfortunately, little work up to this point has performed the significant amount of ablation studies comparing the various building-blocks of a LO pipeline. In this work, we summarize the various techniques that go into defining a LO pipeline and empirically evaluate these LO components on an expansive number of datasets across environments, LiDAR types, and vehicle motions. Finally, we make empirically-backed recommendations for the design of future LO pipelines to provide the most accurate and reliable performance.
>
---
#### [new 007] Application of LLM Guided Reinforcement Learning in Formation Control with Collision Avoidance
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体系统任务，旨在解决编队控制与避障中的奖励函数设计问题。利用大语言模型动态生成评估指标，指导强化学习在线调整奖励函数，提升策略收敛效率，实现在复杂环境中的高效协同控制。**

- **链接: [http://arxiv.org/pdf/2507.16382v1](http://arxiv.org/pdf/2507.16382v1)**

> **作者:** Chenhao Yao; Zike Yuan; Xiaoxu Liu; Chi Zhu
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Multi-Agent Systems (MAS) excel at accomplishing complex objectives through the collaborative efforts of individual agents. Among the methodologies employed in MAS, Multi-Agent Reinforcement Learning (MARL) stands out as one of the most efficacious algorithms. However, when confronted with the complex objective of Formation Control with Collision Avoidance (FCCA): designing an effective reward function that facilitates swift convergence of the policy network to an optimal solution. In this paper, we introduce a novel framework that aims to overcome this challenge. By giving large language models (LLMs) on the prioritization of tasks and the observable information available to each agent, our framework generates reward functions that can be dynamically adjusted online based on evaluation outcomes by employing more advanced evaluation metrics rather than the rewards themselves. This mechanism enables the MAS to simultaneously achieve formation control and obstacle avoidance in dynamic environments with enhanced efficiency, requiring fewer iterations to reach superior performance levels. Our empirical studies, conducted in both simulation and real-world settings, validate the practicality and effectiveness of our proposed approach.
>
---
#### [new 008] Guided Reinforcement Learning for Omnidirectional 3D Jumping in Quadruped Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决四足机器人全向3D跳跃的高效与可解释性问题。现有方法在优化控制和强化学习方面存在效率低、样本复杂度高或运动不可预测的问题。论文提出了一种结合贝塞尔曲线与匀加速直线运动模型的引导强化学习方法，实现了更高效且可解释的跳跃控制。**

- **链接: [http://arxiv.org/pdf/2507.16481v1](http://arxiv.org/pdf/2507.16481v1)**

> **作者:** Riccardo Bussola; Michele Focchi; Giulio Turrisi; Claudio Semini; Luigi Palopoli
>
> **摘要:** Jumping poses a significant challenge for quadruped robots, despite being crucial for many operational scenarios. While optimisation methods exist for controlling such motions, they are often time-consuming and demand extensive knowledge of robot and terrain parameters, making them less robust in real-world scenarios. Reinforcement learning (RL) is emerging as a viable alternative, yet conventional end-to-end approaches lack efficiency in terms of sample complexity, requiring extensive training in simulations, and predictability of the final motion, which makes it difficult to certify the safety of the final motion. To overcome these limitations, this paper introduces a novel guided reinforcement learning approach that leverages physical intuition for efficient and explainable jumping, by combining B\'ezier curves with a Uniformly Accelerated Rectilinear Motion (UARM) model. Extensive simulation and experimental results clearly demonstrate the advantages of our approach over existing alternatives.
>
---
#### [new 009] Designing for Difference: How Human Characteristics Shape Perceptions of Collaborative Robots
- **分类: cs.RO; cs.AI; cs.CV; cs.ET; cs.SY; eess.SY**

- **简介: 该论文研究人类特征如何影响对协作机器人行为的接受度，旨在解决辅助机器人在与不同人群协作时的设计责任与包容性问题。通过在线实验，参与者评估不同人机协作场景，结合认知情感映射（CAM）方法获取反思反馈。结果显示，机器人行为类型与协作对象特征显著影响评价，强调亲社会设计的重要性，并验证了反思方法在用户中心设计中的价值。**

- **链接: [http://arxiv.org/pdf/2507.16480v1](http://arxiv.org/pdf/2507.16480v1)**

> **作者:** Sabrina Livanec; Laura Londoño; Michael Gorki; Adrian Röfer; Abhinav Valada; Andrea Kiesel
>
> **摘要:** The development of assistive robots for social collaboration raises critical questions about responsible and inclusive design, especially when interacting with individuals from protected groups such as those with disabilities or advanced age. Currently, research is scarce on how participants assess varying robot behaviors in combination with diverse human needs, likely since participants have limited real-world experience with advanced domestic robots. In the current study, we aim to address this gap while using methods that enable participants to assess robot behavior, as well as methods that support meaningful reflection despite limited experience. In an online study, 112 participants (from both experimental and control groups) evaluated 7 videos from a total of 28 variations of human-robot collaboration types. The experimental group first completed a cognitive-affective mapping (CAM) exercise on human-robot collaboration before providing their ratings. Although CAM reflection did not significantly affect overall ratings, it led to more pronounced assessments for certain combinations of robot behavior and human condition. Most importantly, the type of human-robot collaboration influences the assessment. Antisocial robot behavior was consistently rated as the lowest, while collaboration with aged individuals elicited more sensitive evaluations. Scenarios involving object handovers were viewed more positively than those without them. These findings suggest that both human characteristics and interaction paradigms influence the perceived acceptability of collaborative robots, underscoring the importance of prosocial design. They also highlight the potential of reflective methods, such as CAM, to elicit nuanced feedback, supporting the development of user-centered and socially responsible robotic systems tailored to diverse populations.
>
---
#### [new 010] Topology Optimization of Leg Structures for Construction Robots Based on Variable Density Method
- **分类: cs.RO**

- **简介: 该论文属于结构优化设计任务，旨在解决施工机器人腿部结构在复杂地形中兼顾承载能力与轻量化的问题。采用SIMP变密度拓扑优化方法对腿部关键部件（股骨段）进行优化设计，通过有限元分析验证性能，最终实现减重目标并确保结构合理性。**

- **链接: [http://arxiv.org/pdf/2507.16335v1](http://arxiv.org/pdf/2507.16335v1)**

> **作者:** Xiao Liu; Xianlong Yang; Weijun Wang; Wei Feng
>
> **摘要:** In complex terrain construction environments, there are high demands for robots to achieve both high payload capacity and mobility flexibility. As the key load-bearing component, the optimization of robotic leg structures is of particular importance. Therefore, this study focuses on the optimization of leg structures for construction robots, proposing a topology optimization strategy based on the SIMP (Solid Isotropic Microstructures with Penalization) variable density method along with a structural re-design approach. The design performance is comprehensively validated through finite element analysis using ANSYS. First, static and modal analyses are conducted to evaluate the rationality of the initial design. Then, topology optimization using the SIMP-based variable density method is applied to the femur section, which accounts for the largest proportion of the leg's weight. Based on iterative calculations, the femur undergoes secondary structural reconstruction. After optimization, the mass of the femur is reduced by 19.45\%, and the overall leg mass decreases by 7.92\%, achieving the goal of lightweight design. Finally, static and modal analyses are conducted on the reconstructed leg. The results demonstrate that the optimized leg still meets structural performance requirements, validating the feasibility of lightweight design. This research provides robust theoretical and technical support for lightweight construction robot design and lays a foundation for their efficient operation in complex construction environments.
>
---
#### [new 011] Adaptive Relative Pose Estimation Framework with Dual Noise Tuning for Safe Approaching Maneuvers
- **分类: cs.RO; cs.AI**

- **简介: 论文属于相对位姿估计任务，旨在解决翻滚废弃卫星（如ENVISAT）近距离操作中的导航难题。通过结合卷积神经网络与自适应非线性滤波，构建完整估计框架，引入双噪声自适应策略，提升测量与模型不确定下的估计鲁棒性，支持安全的主动清理任务。**

- **链接: [http://arxiv.org/pdf/2507.16214v1](http://arxiv.org/pdf/2507.16214v1)**

> **作者:** Batu Candan; Simone Servadio
>
> **摘要:** Accurate and robust relative pose estimation is crucial for enabling challenging Active Debris Removal (ADR) missions targeting tumbling derelict satellites such as ESA's ENVISAT. This work presents a complete pipeline integrating advanced computer vision techniques with adaptive nonlinear filtering to address this challenge. A Convolutional Neural Network (CNN), enhanced with image preprocessing, detects structural markers (corners) from chaser imagery, whose 2D coordinates are converted to 3D measurements using camera modeling. These measurements are fused within an Unscented Kalman Filter (UKF) framework, selected for its ability to handle nonlinear relative dynamics, to estimate the full relative pose. Key contributions include the integrated system architecture and a dual adaptive strategy within the UKF: dynamic tuning of the measurement noise covariance compensates for varying CNN measurement uncertainty, while adaptive tuning of the process noise covariance, utilizing measurement residual analysis, accounts for unmodeled dynamics or maneuvers online. This dual adaptation enhances robustness against both measurement imperfections and dynamic model uncertainties. The performance of the proposed adaptive integrated system is evaluated through high-fidelity simulations using a realistic ENVISAT model, comparing estimates against ground truth under various conditions, including measurement outages. This comprehensive approach offers an enhanced solution for robust onboard relative navigation, significantly advancing the capabilities required for safe proximity operations during ADR missions.
>
---
#### [new 012] Trajectory Planning of a Curtain Wall Installation Robot Based on Biomimetic Mechanisms
- **分类: cs.RO**

- **简介: 该论文属于机器人轨迹规划任务，旨在解决施工机器人能耗高的问题。通过模仿人体上肢举重时的能量转换机制，结合肌电信号与运动轨迹数据，构建仿生轨迹规划框架，并应用粒子群优化算法实现动态负载分配。实际应用于幕墙安装任务中，仿真结果显示能量消耗降低48.4%。**

- **链接: [http://arxiv.org/pdf/2507.16305v1](http://arxiv.org/pdf/2507.16305v1)**

> **作者:** Xiao Liu; Weijun Wang; Tianlun Huang; Zhiyong Wang; Wei Feng
>
> **摘要:** As the robotics market rapidly evolves, energy consumption has become a critical issue, particularly restricting the application of construction robots. To tackle this challenge, our study innovatively draws inspiration from the mechanics of human upper limb movements during weight lifting, proposing a bio-inspired trajectory planning framework that incorporates human energy conversion principles. By collecting motion trajectories and electromyography (EMG) signals during dumbbell curls, we construct an anthropomorphic trajectory planning that integrates human force exertion patterns and energy consumption patterns. Utilizing the Particle Swarm Optimization (PSO) algorithm, we achieve dynamic load distribution for robotic arm trajectory planning based on human-like movement features. In practical application, these bio-inspired movement characteristics are applied to curtain wall installation tasks, validating the correctness and superiority of our trajectory planning method. Simulation results demonstrate a 48.4% reduction in energy consumption through intelligent conversion between kinetic and potential energy. This approach provides new insights and theoretical support for optimizing energy use in curtain wall installation robots during actual handling tasks.
>
---
#### [new 013] Benchmarking LLM Privacy Recognition for Social Robot Decision Making
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究社会机器人决策中大语言模型（LLM）的隐私识别能力。任务是评估现成LLM在家庭场景中的隐私敏感度，并探索其作为隐私控制器的潜力。通过构建基于情境完整性理论的隐私场景，调查用户隐私偏好，并与LLMs的判断进行对比，发现二者一致性较低，并进一步测试不同提示策略以提升LLM隐私识别能力。**

- **链接: [http://arxiv.org/pdf/2507.16124v1](http://arxiv.org/pdf/2507.16124v1)**

> **作者:** Dakota Sullivan; Shirley Zhang; Jennica Li; Heather Kirkorian; Bilge Mutlu; Kassem Fawaz
>
> **备注:** 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
>
> **摘要:** Social robots are embodied agents that interact with people while following human communication norms. These robots interact using verbal and non-verbal cues, and share the physical environments of people. While social robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-empowered social robots for enhanced human-robot interaction. To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within home environments. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household social robots. In this study, we present a set of privacy-relevant scenarios crafted through the lens of Contextual Integrity (CI). We first survey users' privacy preferences regarding in-home social robot behaviors and then examine how their privacy orientation affects their choices of these behaviors (N = 450). We then provide the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and find that the agreement between humans and LLMs is low. To further investigate the capabilities of LLMs as a potential privacy controller, we implement four additional prompting strategies and compare their results. Finally, we discuss the implications and potential of AI privacy awareness in human-robot interaction.
>
---
#### [new 014] A Target-based Multi-LiDAR Multi-Camera Extrinsic Calibration System
- **分类: cs.RO; cs.CV**

- **简介: 论文属于自动驾驶中的多传感器外参标定任务，旨在解决多激光雷达与多相机系统间精确对齐的问题。通过使用定制的ChArUco标定板和非线性优化方法，实现了跨模态传感器标定。实验验证了该方法在真实场景中的有效性。**

- **链接: [http://arxiv.org/pdf/2507.16621v1](http://arxiv.org/pdf/2507.16621v1)**

> **作者:** Lorenzo Gentilini; Pierpaolo Serio; Valentina Donzella; Lorenzo Pollini
>
> **摘要:** Extrinsic Calibration represents the cornerstone of autonomous driving. Its accuracy plays a crucial role in the perception pipeline, as any errors can have implications for the safety of the vehicle. Modern sensor systems collect different types of data from the environment, making it harder to align the data. To this end, we propose a target-based extrinsic calibration system tailored for a multi-LiDAR and multi-camera sensor suite. This system enables cross-calibration between LiDARs and cameras with limited prior knowledge using a custom ChArUco board and a tailored nonlinear optimization method. We test the system with real-world data gathered in a warehouse. Results demonstrated the effectiveness of the proposed method, highlighting the feasibility of a unique pipeline tailored for various types of sensors.
>
---
#### [new 015] Scanning Bot: Efficient Scan Planning using Panoramic Cameras
- **分类: cs.RO**

- **简介: 该论文属于三维重建任务，旨在解决使用全景RGB-D相机手动扫描效率低、操作难的问题。作者提出了一种全自动扫描规划方法，生成高效、无碰撞的扫描路径，确保视角间有足够的特征重叠。实验表明，该方法在真实环境中扫描覆盖率高达99%，且扫描速度比现有方法快达3倍。**

- **链接: [http://arxiv.org/pdf/2507.16175v1](http://arxiv.org/pdf/2507.16175v1)**

> **作者:** Euijeong Lee; Kyung Min Han; Young J. Kim
>
> **摘要:** Panoramic RGB-D cameras are known for their ability to produce high quality 3D scene reconstructions. However, operating these cameras involves manually selecting viewpoints and physically transporting the camera, making the generation of a 3D model time consuming and tedious. Additionally, the process can be challenging for novice users due to spatial constraints, such as ensuring sufficient feature overlap between viewpoint frames. To address these challenges, we propose a fully autonomous scan planning that generates an efficient tour plan for environment scanning, ensuring collision-free navigation and adequate overlap between viewpoints within the plan. Extensive experiments conducted in both synthetic and real-world environments validate the performance of our planner against state-of-the-art view planners. In particular, our method achieved an average scan coverage of 99 percent in the real-world experiment, with our approach being up to 3 times faster than state-of-the-art planners in total scan time.
>
---
#### [new 016] Improved Semantic Segmentation from Ultra-Low-Resolution RGB Images Applied to Privacy-Preserving Object-Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人隐私保护与语义分割任务，旨在解决超低分辨率RGB图像下语义分割困难的问题，以实现保护隐私的语义目标导航。作者提出一种联合学习方法，融合特征提取与分割判别，提升超低分辨率下的语义分割效果，从而增强隐私保护场景下的导航成功率。**

- **链接: [http://arxiv.org/pdf/2507.16034v1](http://arxiv.org/pdf/2507.16034v1)**

> **作者:** Xuying Huang; Sicong Pan; Olga Zatsarynna; Juergen Gall; Maren Bennewitz
>
> **备注:** Submitted to RA-L
>
> **摘要:** User privacy in mobile robotics has become a critical concern. Existing methods typically prioritize either the performance of downstream robotic tasks or privacy protection, with the latter often constraining the effectiveness of task execution. To jointly address both objectives, we study semantic-based robot navigation in an ultra-low-resolution setting to preserve visual privacy. A key challenge in such scenarios is recovering semantic segmentation from ultra-low-resolution RGB images. In this work, we introduce a novel fully joint-learning method that integrates an agglomerative feature extractor and a segmentation-aware discriminator to solve ultra-low-resolution semantic segmentation, thereby enabling privacy-preserving, semantic object-goal navigation. Our method outperforms different baselines on ultra-low-resolution semantic segmentation and our improved segmentation results increase the success rate of the semantic object-goal navigation in a real-world privacy-constrained scenario.
>
---
#### [new 017] Distributed Oscillatory Guidance for Formation Flight of Fixed-Wing Drones
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究固定翼无人机编队飞行，解决速度受限下难以协调的问题。提出分布式振荡引导算法，通过叠加振荡行为控制平均速度，无需速度控制。设计基于非对称饱和函数的一致性算法，实现路径跟踪与编队协调。**

- **链接: [http://arxiv.org/pdf/2507.16458v1](http://arxiv.org/pdf/2507.16458v1)**

> **作者:** Yang Xu; Jesús Bautista; José Hinojosa; Héctor García de Marina
>
> **备注:** Yang Xu and Jes\'us Bautista contributed equally to this work. In the proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** The autonomous formation flight of fixed-wing drones is hard when the coordination requires the actuation over their speeds since they are critically bounded and aircraft are mostly designed to fly at a nominal airspeed. This paper proposes an algorithm to achieve formation flights of fixed-wing drones without requiring any actuation over their speed. In particular, we guide all the drones to travel over specific paths, e.g., parallel straight lines, and we superpose an oscillatory behavior onto the guiding vector field that drives the drones to the paths. This oscillation enables control over the average velocity along the path, thereby facilitating inter-drone coordination. Each drone adjusts its oscillation amplitude distributively in a closed-loop manner by communicating with neighboring agents in an undirected and connected graph. A novel consensus algorithm is introduced, leveraging a non-negative, asymmetric saturation function. This unconventional saturation is justified since negative amplitudes do not make drones travel backward or have a negative velocity along the path. Rigorous theoretical analysis of the algorithm is complemented by validation through numerical simulations and a real-world formation flight.
>
---
#### [new 018] FTIN: Frequency-Time Integration Network for Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于惯性里程计任务，旨在提升定位精度。现有方法依赖时域CNN，难以捕捉长期依赖关系。作者提出FTIN网络，融合频域与时域信息，利用频域全局特性和时域Scalar LSTM建模长期依赖。实验表明在多个数据集上性能优异，尤其在RoNIN数据集上显著降低轨迹误差。**

- **链接: [http://arxiv.org/pdf/2507.16120v1](http://arxiv.org/pdf/2507.16120v1)**

> **作者:** Shanshan Zhang; Qi Zhang; Siyue Wang; Tianshui Wen; Ziheng Zhou; Lingxiang Zheng; Yu Yang
>
> **摘要:** In recent years, machine learning has achieved significant advancements in inertial odometry. However, most existing inertial odometry methods primarily rely on CNNs in the time domain. These methods often struggle to capture long-term dependency in inertial measurement unit data, thereby constraining the potential for further improvements in localization accuracy. To address these issues, we propose a novel network architecture that integrates both frequency-domain and time-domain information. Specifically, we leverage the global view and energy compaction properties of frequency-domain learning to effectively model long-term dependency and reduce redundancy in IMU data. Additionally, we introduce a Scalar LSTM to capture sequential dependencies in the time domain, enabling cross-domain information fusion and providing a stable and reliable reference for localization. Experimental evaluations on multiple public datasets (e.g., RIDI, RoNIN, OxIOD, RNIN, TLIO, and IMUNet) demonstrate the effectiveness of the proposed frequency-time domain fusion strategy. Notably, on the RoNIN dataset, our method achieves a 43.0% reduction in absolute trajectory error and a 13.1% reduction in relative trajectory error compared to RoNIN ResNet.
>
---
#### [new 019] GFM-Planner: Perception-Aware Trajectory Planning with Geometric Feature Metric
- **分类: cs.RO**

- **简介: 论文提出GFM-Planner，一种基于几何特征度量的感知感知轨迹规划框架，旨在提升LiDAR定位精度。通过构建Metric Encoding Map并设计快速解码算法，引导机器人选择特征丰富区域的轨迹，从而避免定位退化问题。属于机器人轨迹规划与定位优化任务。**

- **链接: [http://arxiv.org/pdf/2507.16233v1](http://arxiv.org/pdf/2507.16233v1)**

> **作者:** Yue Lin; Xiaoxuan Zhang; Yang Liu; Dong Wang; Huchuan Lu
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Like humans who rely on landmarks for orientation, autonomous robots depend on feature-rich environments for accurate localization. In this paper, we propose the GFM-Planner, a perception-aware trajectory planning framework based on the geometric feature metric, which enhances LiDAR localization accuracy by guiding the robot to avoid degraded areas. First, we derive the Geometric Feature Metric (GFM) from the fundamental LiDAR localization problem. Next, we design a 2D grid-based Metric Encoding Map (MEM) to efficiently store GFM values across the environment. A constant-time decoding algorithm is further proposed to retrieve GFM values for arbitrary poses from the MEM. Finally, we develop a perception-aware trajectory planning algorithm that improves LiDAR localization capabilities by guiding the robot in selecting trajectories through feature-rich areas. Both simulation and real-world experiments demonstrate that our approach enables the robot to actively select trajectories that significantly enhance LiDAR localization accuracy.
>
---
#### [new 020] Morpheus: A Neural-driven Animatronic Face with Hybrid Actuation and Diverse Emotion Control
- **分类: cs.RO**

- **简介: 该论文属于机器人与情感计算任务，旨在解决现有仿生人脸难以有效表达情感的问题。论文提出了一种结合刚性驱动与绳索驱动的混合执行器设计，并通过自建模网络与神经网络实现从语音到表情的自动映射，实现多样化情感控制。**

- **链接: [http://arxiv.org/pdf/2507.16645v1](http://arxiv.org/pdf/2507.16645v1)**

> **作者:** Zongzheng Zhang; Jiawen Yang; Ziqiao Peng; Meng Yang; Jianzhu Ma; Lin Cheng; Huazhe Xu; Hang Zhao; Hao Zhao
>
> **备注:** Accepted to RSS 2025, Project Page: https://jiawenyang-ch.github.io/Morpheus-Hardware-Design/
>
> **摘要:** Previous animatronic faces struggle to express emotions effectively due to hardware and software limitations. On the hardware side, earlier approaches either use rigid-driven mechanisms, which provide precise control but are difficult to design within constrained spaces, or tendon-driven mechanisms, which are more space-efficient but challenging to control. In contrast, we propose a hybrid actuation approach that combines the best of both worlds. The eyes and mouth-key areas for emotional expression-are controlled using rigid mechanisms for precise movement, while the nose and cheek, which convey subtle facial microexpressions, are driven by strings. This design allows us to build a compact yet versatile hardware platform capable of expressing a wide range of emotions. On the algorithmic side, our method introduces a self-modeling network that maps motor actions to facial landmarks, allowing us to automatically establish the relationship between blendshape coefficients for different facial expressions and the corresponding motor control signals through gradient backpropagation. We then train a neural network to map speech input to corresponding blendshape controls. With our method, we can generate distinct emotional expressions such as happiness, fear, disgust, and anger, from any given sentence, each with nuanced, emotion-specific control signals-a feature that has not been demonstrated in earlier systems. We release the hardware design and code at https://github.com/ZZongzheng0918/Morpheus-Hardware and https://github.com/ZZongzheng0918/Morpheus-Software.
>
---
#### [new 021] Fast Task Planning with Neuro-Symbolic Relaxation
- **分类: cs.RO**

- **简介: 该论文属于任务规划领域，旨在解决复杂环境中长视野任务规划的效率与可靠性问题。现有方法因简化任务可能遗漏关键实体，导致失败。论文提出Flax方法，结合神经网络预测与符号规划，通过重要实体筛选、规则放松与任务重构，提升规划成功率与效率。**

- **链接: [http://arxiv.org/pdf/2507.15975v1](http://arxiv.org/pdf/2507.15975v1)**

> **作者:** Qiwei Du; Bowen Li; Yi Du; Shaoshu Su; Taimeng Fu; Zitong Zhan; Zhipeng Zhao; Chen Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Real-world task planning requires long-horizon reasoning over large sets of entities with complex relationships and attributes, leading to a combinatorial explosion for classical symbolic planners. To prune the search space, recent methods prioritize searching on a simplified task only containing a few "important" entities predicted by a neural network. However, such a simple neuro-symbolic (NeSy) integration risks omitting critical entities and wasting resources on unsolvable simplified tasks. To enable Fast and reliable planning, we introduce a NeSy relaxation strategy (Flax), combining neural importance prediction with symbolic expansion. Specifically, we first learn a graph neural network to predict entity importance to create a simplified task and solve it with a symbolic planner. Then, we solve a rule-relaxed task to obtain a quick rough plan, and reintegrate all referenced entities into the simplified task to recover any overlooked but essential elements. Finally, we apply complementary rules to refine the updated task, keeping it both reliable and compact. Extensive experiments are conducted on both synthetic and real-world maze navigation benchmarks where a robot must traverse through a maze and interact with movable objects. The results show that Flax boosts the average success rate by 20.82% and cuts mean wall-clock planning time by 17.65% compared with the state-of-the-art NeSy baseline. We expect that Flax offers a practical path toward fast, scalable, long-horizon task planning in complex environments.
>
---
#### [new 022] Experience is the Best Teacher: Grounding VLMs for Robotics through Self-Generated Memory
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人与视觉语言模型（VLM）结合的任务，旨在解决VLM在真实机器人上泛化能力差的问题。论文提出了ExpTeach框架，通过自生成经验记忆实现VLM的现实世界适应，结合反思机制与长期记忆检索，显著提升了机器人在多任务中的成功率。**

- **链接: [http://arxiv.org/pdf/2507.16713v1](http://arxiv.org/pdf/2507.16713v1)**

> **作者:** Guowei Lan; Kaixian Qu; René Zurbrügg; Changan Chen; Christopher E. Mower; Haitham Bou-Ammar; Marco Hutter
>
> **摘要:** Vision-language models (VLMs) have been widely adopted in robotics to enable autonomous planning. However, grounding VLMs, originally trained on internet data, to diverse real-world robots remains a challenge. This paper presents ExpTeach, a framework that grounds VLMs to physical robots by building a self-generated memory of real-world experiences. In ExpTeach, the VLM autonomously plans actions, verifies outcomes, reflects on failures, and adapts robot behaviors in a closed loop. The self-generated experiences during this process are then summarized into a long-term memory, enabling retrieval of learned knowledge to guide future tasks via retrieval-augmented generation (RAG). Additionally, ExpTeach enhances the spatial understanding of VLMs with an on-demand image annotation module. In experiments, we show that reflection improves success rates from 36% to 84% on four challenging robotic tasks and observe the emergence of intelligent object interactions, including creative tool use. Across extensive tests on 12 real-world scenarios (including eight unseen ones), we find that grounding with long-term memory boosts single-trial success rates from 22% to 80%, demonstrating the effectiveness and generalizability of ExpTeach.
>
---
#### [new 023] Humanoid Robot Whole-body Geometric Calibration with Embedded Sensors and a Single Plane
- **分类: cs.RO**

- **简介: 该论文属于机器人校准任务，旨在解决人形机器人全身几何校准过程繁琐、耗时的问题。作者提出了一种基于单平面、嵌入式力传感器和导纳控制器的新方法，并结合IROC算法选择最优校准姿态。实验验证表明，该方法在TALOS机器人上仅使用31个最优姿态即可显著提高校准精度。**

- **链接: [http://arxiv.org/pdf/2507.16369v1](http://arxiv.org/pdf/2507.16369v1)**

> **作者:** Thanh D V Nguyen; Vincent Bonnet; Pierre Fernbach; David Daney; Florent Lamiraux
>
> **摘要:** Whole-body geometric calibration of humanoid robots using classical robot calibration methods is a timeconsuming and experimentally burdensome task. However, despite its significance for accurate control and simulation, it is often overlooked in the humanoid robotics community. To address this issue, we propose a novel practical method that utilizes a single plane, embedded force sensors, and an admittance controller to calibrate the whole-body kinematics of humanoids without requiring manual intervention. Given the complexity of humanoid robots, it is crucial to generate and determine a minimal set of optimal calibration postures. To do so, we propose a new algorithm called IROC (Information Ranking algorithm for selecting Optimal Calibration postures). IROC requires a pool of feasible candidate postures to build a normalized weighted information matrix for each posture. Then, contrary to other algorithms from the literature, IROC will determine the minimal number of optimal postures that are to be played onto a robot for its calibration. Both IROC and the single-plane calibration method were experimentally validated on a TALOS humanoid robot. The total whole-body kinematics chain was calibrated using solely 31 optimal postures with 3-point contacts on a table by the robot gripper. In a cross-validation experiment, the average root-mean-square (RMS) error was reduced by a factor of 2.3 compared to the manufacturer's model.
>
---
#### [new 024] DWSFormer: A Lightweight Inertial Odometry Network for Complex Motion Modeling
- **分类: cs.RO**

- **简介: 论文提出DWSFormer，一种轻量级惯性里程计网络，用于复杂运动建模。该工作属于惯性定位任务，旨在解决复杂运动下传统方法估计轨迹不准确、存在漂移的问题。通过引入Star Operation、协同注意力机制和多尺度门控卷积单元，提升了运动特征提取和建模能力，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.16121v1](http://arxiv.org/pdf/2507.16121v1)**

> **作者:** Shanshan Zhang; Qi Zhang; Siyue Wang; Tianshui Wen; Ziheng Zhou; Lingxiang Zheng; Yu Yang
>
> **摘要:** Inertial odometry (IO) directly estimates the position of a carrier from inertial sensor measurements and serves as a core technology for the widespread deployment of consumer grade localization systems. While existing IO methods can accurately reconstruct simple and near linear motion trajectories, they often fail to account for drift errors caused by complex motion patterns such as turning. This limitation significantly degrades localization accuracy and restricts the applicability of IO systems in real world scenarios. To address these challenges, we propose a lightweight IO framework. Specifically, inertial data is projected into a high dimensional implicit nonlinear feature space using the Star Operation method, enabling the extraction of complex motion features that are typically overlooked. We further introduce a collaborative attention mechanism that jointly models global motion dynamics across both channel and temporal dimensions. In addition, we design Multi Scale Gated Convolution Units to capture fine grained dynamic variations throughout the motion process, thereby enhancing the model's ability to learn rich and expressive motion representations. Extensive experiments demonstrate that our proposed method consistently outperforms SOTA baselines across six widely used inertial datasets. Compared to baseline models on the RoNIN dataset, it achieves reductions in ATE ranging from 2.26% to 65.78%, thereby establishing a new benchmark in the field.
>
---
#### [new 025] Physics-aware Truck and Drone Delivery Planning Using Optimization & Machine Learning
- **分类: math.OC; cs.RO**

- **简介: 该论文属于物流配送规划任务，旨在解决卡车与无人机协同配送中的路径优化问题。现有方法忽略无人机飞行物理特性，导致规划次优。论文提出结合优化与机器学习的方法，将无人机动力学和能耗模型纳入整体规划，提升配送效率与环保性。**

- **链接: [http://arxiv.org/pdf/2507.16259v1](http://arxiv.org/pdf/2507.16259v1)**

> **作者:** Yineng Sun; Armin Fügenschuh; Vikrant Vaze
>
> **摘要:** Combining an energy-efficient drone with a high-capacity truck for last-mile package delivery can benefit operators and customers by reducing delivery times and environmental impact. However, directly integrating drone flight dynamics into the combinatorially hard truck route planning problem is challenging. Simplified models that ignore drone flight physics can lead to suboptimal delivery plans. We propose an integrated formulation for the joint problem of truck route and drone trajectory planning and a new end-to-end solution approach that combines optimization and machine learning to generate high-quality solutions in practical online runtimes. Our solution method trains neural network predictors based on offline solutions to the drone trajectory optimization problem instances to approximate drone flight times, and uses these approximations to optimize the overall truck-and-drone delivery plan by augmenting an existing order-first-split-second heuristic. Our method explicitly incorporates key kinematics and energy equations in drone trajectory optimization, and thereby outperforms state-of-the-art benchmarks that ignore drone flight physics. Extensive experimentation using synthetic datasets and real-world case studies shows that the integration of drone trajectories into package delivery planning substantially improves system performance in terms of tour duration and drone energy consumption. Our modeling and computational framework can help delivery planners achieve annual savings worth millions of dollars while also benefiting the environment.
>
---
#### [new 026] Improved Wake-Up Time For Euclidean Freeze-Tag Problem
- **分类: cs.CG; cs.DC; cs.RO**

- **简介: 该论文属于算法优化任务，旨在解决欧几里得冻结标签问题（Euclidean Freeze-Tag Problem），即尽快唤醒一组初始处于休眠状态的机器人。研究者改进了唤醒时间的上界，在二维和三维空间中提出了更优的唤醒策略，提升了唤醒效率。**

- **链接: [http://arxiv.org/pdf/2507.16269v1](http://arxiv.org/pdf/2507.16269v1)**

> **作者:** Sharareh Alipour; Arash Ahadi; Kajal Baghestani
>
> **摘要:** The Freeze-Tag Problem (FTP) involves activating a set of initially asleep robots as quickly as possible, starting from a single awake robot. Once activated, a robot can assist in waking up other robots. Each active robot moves at unit speed. The objective is to minimize the makespan, i.e., the time required to activate the last robot. A key performance measure is the wake-up ratio, defined as the maximum time needed to activate any number of robots in any primary positions. This work focuses on the geometric (Euclidean) version of FTP in $\mathbb{R}^d$ under the $\ell_p$ norm, where the initial distance between each asleep robot and the single active robot is at most 1. For $(\mathbb{R}^2, \ell_2)$, we improve the previous upper bound of 4.62 ([7], CCCG 2024) to 4.31. Note that it is known that 3.82 is a lower bound for the wake-up ratio. In $\mathbb{R}^3$, we propose a new strategy that achieves a wake-up ratio of 12 for $(\mathbb{R}^3, \ell_1)$ and 12.76 for $(\mathbb{R}^3, \ell_2)$, improving upon the previous bounds of 13 and $13\sqrt{3}$, respectively, reported in [2].
>
---
#### [new 027] COMPASS: Cooperative Multi-Agent Persistent Monitoring using Spatio-Temporal Attention Network
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多智能体强化学习任务，旨在解决动态目标持续监控问题。通过构建基于时空注意力网络的COMPASS框架，实现多智能体协同监控，提升信息获取效率与不确定性处理能力。**

- **链接: [http://arxiv.org/pdf/2507.16306v1](http://arxiv.org/pdf/2507.16306v1)**

> **作者:** Xingjian Zhang; Yizhuo Wang; Guillaume Sartoretti
>
> **摘要:** Persistent monitoring of dynamic targets is essential in real-world applications such as disaster response, environmental sensing, and wildlife conservation, where mobile agents must continuously gather information under uncertainty. We propose COMPASS, a multi-agent reinforcement learning (MARL) framework that enables decentralized agents to persistently monitor multiple moving targets efficiently. We model the environment as a graph, where nodes represent spatial locations and edges capture topological proximity, allowing agents to reason over structured layouts and revisit informative regions as needed. Each agent independently selects actions based on a shared spatio-temporal attention network that we design to integrate historical observations and spatial context. We model target dynamics using Gaussian Processes (GPs), which support principled belief updates and enable uncertainty-aware planning. We train COMPASS using centralized value estimation and decentralized policy execution under an adaptive reward setting. Our extensive experiments demonstrate that COMPASS consistently outperforms strong baselines in uncertainty reduction, target coverage, and coordination efficiency across dynamic multi-target scenarios.
>
---
#### [new 028] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）推理任务，旨在解决现有方法缺乏显式多步规划与复杂任务适应能力的问题。作者提出ThinkAct框架，通过强化视觉潜在规划，实现高层推理与底层动作执行的结合，支持少样本适应、长视野规划与自我修正行为。**

- **链接: [http://arxiv.org/pdf/2507.16815v1](http://arxiv.org/pdf/2507.16815v1)**

> **作者:** Chi-Pin Huang; Yueh-Hua Wu; Min-Hung Chen; Yu-Chiang Frank Wang; Fu-En Yang
>
> **备注:** Project page: https://jasper0314-huang.github.io/thinkact-vla/
>
> **摘要:** Vision-language-action (VLA) reasoning tasks require agents to interpret multimodal instructions, perform long-horizon planning, and act adaptively in dynamic environments. Existing approaches typically train VLA models in an end-to-end fashion, directly mapping inputs to actions without explicit reasoning, which hinders their ability to plan over multiple steps or adapt to complex task variations. In this paper, we propose ThinkAct, a dual-system framework that bridges high-level reasoning with low-level action execution via reinforced visual latent planning. ThinkAct trains a multimodal LLM to generate embodied reasoning plans guided by reinforcing action-aligned visual rewards based on goal completion and trajectory consistency. These reasoning plans are compressed into a visual plan latent that conditions a downstream action model for robust action execution on target environments. Extensive experiments on embodied reasoning and robot manipulation benchmarks demonstrate that ThinkAct enables few-shot adaptation, long-horizon planning, and self-correction behaviors in complex embodied AI tasks.
>
---
## 更新

#### [replaced 001] Unveiling the Potential of Segment Anything Model 2 for RGB-Thermal Semantic Segmentation with Language Guidance
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.02581v2](http://arxiv.org/pdf/2503.02581v2)**

> **作者:** Jiayi Zhao; Fei Teng; Kai Luo; Guoqiang Zhao; Zhiyong Li; Xu Zheng; Kailun Yang
>
> **备注:** Accepted to IROS 2025. The source code will be made publicly available at https://github.com/iAsakiT3T/SHIFNet
>
> **摘要:** The perception capability of robotic systems relies on the richness of the dataset. Although Segment Anything Model 2 (SAM2), trained on large datasets, demonstrates strong perception potential in perception tasks, its inherent training paradigm prevents it from being suitable for RGB-T tasks. To address these challenges, we propose SHIFNet, a novel SAM2-driven Hybrid Interaction Paradigm that unlocks the potential of SAM2 with linguistic guidance for efficient RGB-Thermal perception. Our framework consists of two key components: (1) Semantic-Aware Cross-modal Fusion (SACF) module that dynamically balances modality contributions through text-guided affinity learning, overcoming SAM2's inherent RGB bias; (2) Heterogeneous Prompting Decoder (HPD) that enhances global semantic information through a semantic enhancement module and then combined with category embeddings to amplify cross-modal semantic consistency. With 32.27M trainable parameters, SHIFNet achieves state-of-the-art segmentation performance on public benchmarks, reaching 89.8% on PST900 and 67.8% on FMB, respectively. The framework facilitates the adaptation of pre-trained large models to RGB-T segmentation tasks, effectively mitigating the high costs associated with data collection while endowing robotic systems with comprehensive perception capabilities. The source code will be made publicly available at https://github.com/iAsakiT3T/SHIFNet.
>
---
#### [replaced 002] Growing Trees with an Agent: Accelerating RRTs with Learned, Multi-Step Episodic Exploration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.06605v2](http://arxiv.org/pdf/2507.06605v2)**

> **作者:** Xinyu Wu
>
> **摘要:** Classical sampling-based motion planners like the RRTs suffer from inefficiencies, particularly in cluttered or high-dimensional spaces, due to their reliance on undirected, random sampling. This paper introduces the Episodic RRT, a novel hybrid planning framework that replaces the primitive of a random point with a learned, multi-step "exploratory episode" generated by a Deep Reinforcement Learning agent. By making the DRL agent the engine of exploration, ERRT transforms the search process from a diffuse, volumetric expansion into a directed, branch-like growth. This paradigm shift yields key advantages: it counters the curse of dimensionality with focused exploration, minimizes expensive collision checks by proactively proposing locally valid paths, and improves connectivity by generating inherently connected path segments. We demonstrate through extensive empirical evaluation across 2D, 3D, and 6D environments that ERRT and its variants consistently and significantly outperform their classical counterparts without any GPU acceleration. In a challenging 6D robotic arm scenario, ERRT achieves a 98% success rate compared to 19% for RRT, is up to 107x faster, reduces collision checks by over 99.6%, and finds initial paths that are nearly 50% shorter. Furthermore, its asymptotically optimal variant, ERRT*, demonstrates vastly superior anytime performance, refining solutions to near-optimality up to 29x faster than standard RRT* in 3D environments. Code: https://xinyuwuu.github.io/Episodic_RRT/.
>
---
#### [replaced 003] One-Shot Affordance Grounding of Deformable Objects in Egocentric Organizing Scenes
- **分类: cs.CV; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2503.01092v2](http://arxiv.org/pdf/2503.01092v2)**

> **作者:** Wanjun Jia; Fan Yang; Mengfei Duan; Xianchi Chen; Yinxi Wang; Yiming Jiang; Wenrui Chen; Kailun Yang; Zhiyong Li
>
> **备注:** Accepted to IROS 2025. Source code and benchmark dataset will be publicly available at https://github.com/Dikay1/OS-AGDO
>
> **摘要:** Deformable object manipulation in robotics presents significant challenges due to uncertainties in component properties, diverse configurations, visual interference, and ambiguous prompts. These factors complicate both perception and control tasks. To address these challenges, we propose a novel method for One-Shot Affordance Grounding of Deformable Objects (OS-AGDO) in egocentric organizing scenes, enabling robots to recognize previously unseen deformable objects with varying colors and shapes using minimal samples. Specifically, we first introduce the Deformable Object Semantic Enhancement Module (DefoSEM), which enhances hierarchical understanding of the internal structure and improves the ability to accurately identify local features, even under conditions of weak component information. Next, we propose the ORB-Enhanced Keypoint Fusion Module (OEKFM), which optimizes feature extraction of key components by leveraging geometric constraints and improves adaptability to diversity and visual interference. Additionally, we propose an instance-conditional prompt based on image data and task context, which effectively mitigates the issue of region ambiguity caused by prompt words. To validate these methods, we construct a diverse real-world dataset, AGDDO15, which includes 15 common types of deformable objects and their associated organizational actions. Experimental results demonstrate that our approach significantly outperforms state-of-the-art methods, achieving improvements of 6.2%, 3.2%, and 2.9% in KLD, SIM, and NSS metrics, respectively, while exhibiting high generalization performance. Source code and benchmark dataset are made publicly available at https://github.com/Dikay1/OS-AGDO.
>
---
#### [replaced 004] A Goal-Oriented Reinforcement Learning-Based Path Planning Algorithm for Modular Self-Reconfigurable Satellites
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.01966v2](http://arxiv.org/pdf/2505.01966v2)**

> **作者:** Bofei Liu; Dong Ye; Zunhao Yao; Zhaowei Sun
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Modular self-reconfigurable satellites refer to satellite clusters composed of individual modular units capable of altering their configurations. The configuration changes enable the execution of diverse tasks and mission objectives. Existing path planning algorithms for reconfiguration often suffer from high computational complexity, poor generalization capability, and limited support for diverse target configurations. To address these challenges, this paper proposes a goal-oriented reinforcement learning-based path planning algorithm. This algorithm is the first to address the challenge that previous reinforcement learning methods failed to overcome, namely handling multiple target configurations. Moreover, techniques such as Hindsight Experience Replay and Invalid Action Masking are incorporated to overcome the significant obstacles posed by sparse rewards and invalid actions. Based on these designs, our model achieves a 95% and 73% success rate in reaching arbitrary target configurations in a modular satellite cluster composed of four and six units, respectively.
>
---
#### [replaced 005] Curating Demonstrations using Online Experience
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.03707v2](http://arxiv.org/pdf/2503.03707v2)**

> **作者:** Annie S. Chen; Alec M. Lessing; Yuejiang Liu; Chelsea Finn
>
> **摘要:** Many robot demonstration datasets contain heterogeneous demonstrations of varying quality. This heterogeneity may benefit policy pre-training, but can hinder robot performance when used with a final imitation learning objective. In particular, some strategies in the data may be less reliable than others or may be underrepresented in the data, leading to poor performance when such strategies are sampled at test time. Moreover, such unreliable or underrepresented strategies can be difficult even for people to discern, and sifting through demonstration datasets is time-consuming and costly. On the other hand, policy performance when trained on such demonstrations can reflect the reliability of different strategies. We thus propose for robots to self-curate based on online robot experience (Demo-SCORE). More specifically, we train and cross-validate a classifier to discern successful policy roll-outs from unsuccessful ones and use the classifier to filter heterogeneous demonstration datasets. Our experiments in simulation and the real world show that Demo-SCORE can effectively identify suboptimal demonstrations without manual curation. Notably, Demo-SCORE achieves over 15-35% higher absolute success rate in the resulting policy compared to the base policy trained with all original demonstrations.
>
---
#### [replaced 006] GEMINUS: Dual-aware Global and Scene-Adaptive Mixture-of-Experts for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14456v2](http://arxiv.org/pdf/2507.14456v2)**

> **作者:** Chi Wan; Yixin Cui; Jiatong Du; Shuo Yang; Yulong Bai; Yanjun Huang
>
> **摘要:** End-to-end autonomous driving requires adaptive and robust handling of complex and diverse traffic environments. However, prevalent single-mode planning methods attempt to learn an overall policy while struggling to acquire diversified driving skills to handle diverse scenarios. Therefore, this paper proposes GEMINUS, a Mixture-of-Experts end-to-end autonomous driving framework featuring a Global Expert, a Scene-Adaptive Experts Group, and equipped with a Dual-aware Router. Specifically, the Global Expert is trained on the overall dataset, possessing robust performance. The Scene-Adaptive Experts are trained on corresponding scene subsets, achieving adaptive performance. The Dual-aware Router simultaneously considers scenario-level features and routing uncertainty to dynamically activate expert modules. Through the effective coupling of the Global Expert and the Scene-Adaptive Experts Group via the Dual-aware Router, GEMINUS achieves adaptive and robust performance in diverse scenarios. GEMINUS outperforms existing methods in the Bench2Drive closed-loop benchmark and achieves state-of-the-art performance in Driving Score and Success Rate, even with only monocular vision input. Furthermore, ablation studies demonstrate significant improvements over the original single-expert baseline: 7.67% in Driving Score, 22.06% in Success Rate, and 19.41% in MultiAbility-Mean. The code will be available at https://github.com/newbrains1/GEMINUS.
>
---
#### [replaced 007] Adaptive Gaussian Mixture Models-based Anomaly Detection for under-constrained Cable-Driven Parallel Robots
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.07714v2](http://arxiv.org/pdf/2507.07714v2)**

> **作者:** Julio Garrido; Javier Vales; Diego Silva-Muñiz; Enrique Riveiro; Pablo López-Matencio; Josué Rivera-Andrade
>
> **备注:** 14 pages, 8 figures, 1 table
>
> **摘要:** Cable-Driven Parallel Robots (CDPRs) are increasingly used for load manipulation tasks involving predefined toolpaths with intermediate stops. At each stop, where the platform maintains a fixed pose and the motors keep the cables under tension, the system must evaluate whether it is safe to proceed by detecting anomalies that could compromise performance (e.g., wind gusts or cable impacts). This paper investigates whether anomalies can be detected using only motor torque data, without additional sensors. It introduces an adaptive, unsupervised outlier detection algorithm based on Gaussian Mixture Models (GMMs) to identify anomalies from torque signals. The method starts with a brief calibration period, just a few seconds, during which a GMM is fit on known anomaly-free data. Real-time torque measurements are then evaluated using Mahalanobis distance from the GMM, with statistically derived thresholds triggering anomaly flags. Model parameters are periodically updated using the latest segments identified as anomaly-free to adapt to changing conditions. Validation includes 14 long-duration test sessions simulating varied wind intensities. The proposed method achieves a 100% true positive rate and 95.4% average true negative rate, with 1-second detection latency. Comparative evaluation against power threshold and non-adaptive GMM methods indicates higher robustness to drift and environmental variation.
>
---
#### [replaced 008] Beacon: A Naturalistic Driving Dataset During Blackouts for Benchmarking Traffic Reconstruction and Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.14208v2](http://arxiv.org/pdf/2412.14208v2)**

> **作者:** Supriya Sarker; Iftekharul Islam; Bibek Poudel; Weizi Li
>
> **备注:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025
>
> **摘要:** Extreme weather and infrastructure vulnerabilities pose significant challenges to urban mobility, particularly at intersections where signals become inoperative. To address this growing concern, we introduce Beacon, a naturalistic driving dataset capturing traffic dynamics during blackouts at two major intersections in Memphis, TN, USA. The dataset provides detailed traffic movements, including timesteps, origin, and destination lanes for each vehicle over four hours of peak periods. We analyze traffic demand, vehicle trajectories, and density across different scenarios, demonstrating high-fidelity reconstruction under unsignalized, signalized, and mixed traffic conditions. We find that integrating robot vehicles (RVs) into traffic flow can substantially reduce intersection delays, with wait time improvements of up to 82.6%. However, this enhanced traffic efficiency comes with varying environmental impacts, as decreased vehicle idling may lead to higher overall CO2 emissions. To the best of our knowledge, Beacon is the first publicly available traffic dataset for naturalistic driving behaviors during blackouts at intersections.
>
---
#### [replaced 009] Human-Machine Shared Control Approach for the Takeover of Cooperative Adaptive Cruise Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.11551v2](http://arxiv.org/pdf/2407.11551v2)**

> **作者:** Haoran Wang; Zhexi Lian; Zhenning Li; Jiawei Wang; Arno Eichberger; Jia Hu; Yongyu Chen; Yongji Gao
>
> **备注:** IEEE Transactions on Intelligent Transportation Systems (2025)
>
> **摘要:** Cooperative Adaptive Cruise Control (CACC) often requires human takeover for tasks such as exiting a freeway. Direct human takeover can pose significant risks, especially given the close-following strategy employed by CACC, which might cause drivers to feel unsafe and execute hard braking, potentially leading to collisions. This research aims to develop a CACC takeover controller that ensures a smooth transition from automated to human control. The proposed CACC takeover maneuver employs an indirect human-machine shared control approach, modeled as a Stackelberg competition where the machine acts as the leader and the human as the follower. The machine guides the human to respond in a manner that aligns with the machine's expectations, aiding in maintaining following stability. Additionally, the human reaction function is integrated into the machine's predictive control system, moving beyond a simple "prediction-planning" pipeline to enhance planning optimality. The controller has been verified to i) enable a smooth takeover maneuver of CACC; ii) ensure string stability in the condition that the platoon has less than 6 CAVs and human control authority is less than 40%; iii) enhance both perceived and actual safety through machine interventions; and iv) reduce the impact on upstream traffic by up to 60%.
>
---
#### [replaced 010] Bio-Skin: A Cost-Effective Thermostatic Tactile Sensor with Multi-Modal Force and Temperature Detection
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07989v2](http://arxiv.org/pdf/2503.07989v2)**

> **作者:** Haoran Guo; Haoyang Wang; Zhengxiong Li; Lingfeng Tao
>
> **备注:** This work has been accepted by IROS2025
>
> **摘要:** Tactile sensors can significantly enhance the perception of humanoid robotics systems by providing contact information that facilitates human-like interactions. However, existing commercial tactile sensors focus on improving the resolution and sensitivity of single-modal detection with high-cost components and densely integrated design, incurring complex manufacturing processes and unaffordable prices. In this work, we present Bio-Skin, a cost-effective multi-modal tactile sensor that utilizes single-axis Hall-effect sensors for planar normal force measurement and bar-shape piezo resistors for 2D shear force measurement. A thermistor coupling with a heating wire is integrated into a silicone body to achieve temperature sensation and thermostatic function analogous to human skin. We also present a cross-reference framework to validate the two modalities of the force sensing signal, improving the sensing fidelity in a complex electromagnetic environment. Bio-Skin has a multi-layer design, and each layer is manufactured sequentially and subsequently integrated, thereby offering a fast production pathway. After calibration, Bio-Skin demonstrates performance metrics-including signal-to-range ratio, sampling rate, and measurement range-comparable to current commercial products, with one-tenth of the cost. The sensor's real-world performance is evaluated using an Allegro hand in object grasping tasks, while its temperature regulation functionality was assessed in a material detection task.
>
---
#### [replaced 011] SciFi-Benchmark: Leveraging Science Fiction To Improve Robot Behavior
- **分类: cs.CL; cs.AI; cs.CY; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10706v2](http://arxiv.org/pdf/2503.10706v2)**

> **作者:** Pierre Sermanet; Anirudha Majumdar; Vikas Sindhwani
>
> **备注:** Minor improvements over previous version
>
> **摘要:** Given the recent rate of progress in artificial intelligence (AI) and robotics, a tantalizing question is emerging: would robots controlled by emerging AI systems be strongly aligned with human values? In this work, we propose a scalable way to probe this question by generating a benchmark spanning the key moments in 824 major pieces of science fiction literature (movies, tv, novels and scientific books) where an agent (AI or robot) made critical decisions (good or bad). We use a state-of-the-art LLM's recollection of each key moment to generate questions in similar situations, the decisions made by the agent, and alternative decisions it could have made (good or bad). We then measure an approximation of how well models align with human values on a set of human-voted answers. We also generate rules that can be automatically improved via an amendment process in order to generate the first Sci-Fi inspired constitutions for promoting ethical behavior in AIs and robots in the real world. Our first finding is that modern LLMs paired with constitutions turn out to be well-aligned with human values (95.8%), contrary to unsettling decisions typically made in Sci-Fi (only 21.2% alignment). Secondly, we find that generated constitutions substantially increase alignment compared to the base model (79.4% to 95.8%), and show resilience to an adversarial prompt setting (23.3% to 92.3%). Additionally, we find that those constitutions are among the top performers on the ASIMOV Benchmark which is derived from real-world images and hospital injury reports. Sci-Fi-inspired constitutions are thus highly aligned and applicable in real-world situations. We release SciFi-Benchmark: a large-scale dataset to advance robot ethics and safety research. It comprises 9,056 questions and 53,384 answers generated through a novel LLM-introspection process, in addition to a smaller human-labeled evaluation set.
>
---
#### [replaced 012] PR2: A Physics- and Photo-realistic Humanoid Testbed with Pilot Study in Competition
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.01559v2](http://arxiv.org/pdf/2409.01559v2)**

> **作者:** Hangxin Liu; Qi Xie; Zeyu Zhang; Tao Yuan; Song Wang; Zaijin Wang; Xiaokun Leng; Lining Sun; Jingwen Zhang; Zhicheng He; Yao Su
>
> **摘要:** This paper presents the development of a Physics-realistic and Photo-realistic humanoid robot testbed, PR2, to facilitate collaborative research between Embodied Artificial Intelligence (Embodied AI) and robotics. PR2 offers high-quality scene rendering and robot dynamic simulation, enabling (i) the creation of diverse scenes using various digital assets, (ii) the integration of advanced perception or foundation models, and (iii) the implementation of planning and control algorithms for dynamic humanoid robot behaviors based on environmental feedback. The beta version of PR2 has been deployed for the simulation track of a nationwide full-size humanoid robot competition for college students, attracting 137 teams and over 400 participants within four months. This competition covered traditional tasks in bipedal walking, as well as novel challenges in loco-manipulation and language-instruction-based object search, marking a first for public college robotics competitions. A retrospective analysis of the competition suggests that future events should emphasize the integration of locomotion with manipulation and perception. By making the PR2 testbed publicly available at https://github.com/pr2-humanoid/PR2-Platform, we aim to further advance education and training in humanoid robotics. Video demonstration: https://pr2-humanoid.github.io/
>
---
#### [replaced 013] eKalibr-Stereo: Continuous-Time Spatiotemporal Calibration for Event-Based Stereo Visual Systems
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04451v2](http://arxiv.org/pdf/2504.04451v2)**

> **作者:** Shuolong Chen; Xingxing Li; Liu Yuan
>
> **摘要:** The bioinspired event camera, distinguished by its exceptional temporal resolution, high dynamic range, and low power consumption, has been extensively studied in recent years for motion estimation, robotic perception, and object detection. In ego-motion estimation, the stereo event camera setup is commonly adopted due to its direct scale perception and depth recovery. For optimal stereo visual fusion, accurate spatiotemporal (extrinsic and temporal) calibration is required. Considering that few stereo visual calibrators orienting to event cameras exist, based on our previous work eKalibr (an event camera intrinsic calibrator), we propose eKalibr-Stereo for accurate spatiotemporal calibration of event-based stereo visual systems. To improve the continuity of grid pattern tracking, building upon the grid pattern recognition method in eKalibr, an additional motion prior-based tracking module is designed in eKalibr-Stereo to track incomplete grid patterns. Based on tracked grid patterns, a two-step initialization procedure is performed to recover initial guesses of piece-wise B-splines and spatiotemporal parameters, followed by a continuous-time batch bundle adjustment to refine the initialized states to optimal ones. The results of extensive real-world experiments show that eKalibr-Stereo can achieve accurate event-based stereo spatiotemporal calibration. The implementation of eKalibr-Stereo is open-sourced at (https://github.com/Unsigned-Long/eKalibr) to benefit the research community.
>
---
#### [replaced 014] Bundle Adjustment in the Eager Mode
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.12190v2](http://arxiv.org/pdf/2409.12190v2)**

> **作者:** Zitong Zhan; Huan Xu; Zihang Fang; Xinpeng Wei; Yaoyu Hu; Chen Wang
>
> **摘要:** Bundle adjustment (BA) is a critical technique in various robotic applications such as simultaneous localization and mapping (SLAM), augmented reality (AR), and photogrammetry. BA optimizes parameters such as camera poses and 3D landmarks to align them with observations. With the growing importance of deep learning in perception systems, there is an increasing need to integrate BA with deep learning frameworks for enhanced reliability and performance. However, widely-used C++-based BA libraries, such as GTSAM, g$^2$o, and Ceres, lack native integration with modern deep learning libraries like PyTorch. This limitation affects their flexibility, adaptability, ease of debugging, and overall implementation efficiency. To address this gap, we introduce an eager-mode BA library seamlessly integrated with PyTorch with high efficiency. Our approach includes GPU-accelerated, differentiable, and sparse operations designed for \nth{2}-order optimization, Lie group and Lie algebra operations, and linear solvers. Our eager-mode BA on GPU demonstrates substantial runtime efficiency, achieving an average speedup of 18.5$\times$, 22$\times$, and 23$\times$ compared to GTSAM, g$^2$o, and Ceres, respectively. The source code will be available at https://github.com/sair-lab/bae.
>
---
#### [replaced 015] Robust Ladder Climbing with a Quadrupedal Robot
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.17731v2](http://arxiv.org/pdf/2409.17731v2)**

> **作者:** Dylan Vogel; Robert Baines; Joseph Church; Julian Lotzer; Karl Werner; Marco Hutter
>
> **备注:** Project website: https://sites.google.com/leggedrobotics.com/climbingladders
>
> **摘要:** Quadruped robots are proliferating in industrial environments where they carry sensor payloads and serve as autonomous inspection platforms. Despite the advantages of legged robots over their wheeled counterparts on rough and uneven terrain, they are still unable to reliably negotiate a ubiquitous feature of industrial infrastructure: ladders. Inability to traverse ladders prevents quadrupeds from inspecting dangerous locations, puts humans in harm's way, and reduces industrial site productivity. In this paper, we learn quadrupedal ladder climbing via a reinforcement learning-based control policy and a complementary hooked end effector. We evaluate the robustness in simulation across different ladder inclinations, rung geometries, and inter-rung spacings. On hardware, we demonstrate zero-shot transfer with an overall 90% success rate at ladder angles ranging from 70{\deg} to 90{\deg}, consistent climbing performance during unmodeled perturbations, and climbing speeds 232x faster than the state of the art. This work expands the scope of industrial quadruped robot applications beyond inspection on nominal terrains to challenging infrastructural features in the environment, highlighting synergies between robot morphology and control policy when performing complex skills. More information can be found at the project website: https://sites.google.com/leggedrobotics.com/climbingladders.
>
---
#### [replaced 016] GR-3 Technical Report
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.15493v2](http://arxiv.org/pdf/2507.15493v2)**

> **作者:** Chilam Cheang; Sijin Chen; Zhongren Cui; Yingdong Hu; Liqun Huang; Tao Kong; Hang Li; Yifeng Li; Yuxiao Liu; Xiao Ma; Hao Niu; Wenxuan Ou; Wanli Peng; Zeyu Ren; Haixin Shi; Jiawen Tian; Hongtao Wu; Xin Xiao; Yuyang Xiao; Jiafeng Xu; Yichu Yang
>
> **备注:** Tech report. Authors are listed in alphabetical order. Project page: https://seed.bytedance.com/GR3/
>
> **摘要:** We report our recent progress towards building generalist robot policies, the development of GR-3. GR-3 is a large-scale vision-language-action (VLA) model. It showcases exceptional capabilities in generalizing to novel objects, environments, and instructions involving abstract concepts. Furthermore, it can be efficiently fine-tuned with minimal human trajectory data, enabling rapid and cost-effective adaptation to new settings. GR-3 also excels in handling long-horizon and dexterous tasks, including those requiring bi-manual manipulation and mobile movement, showcasing robust and reliable performance. These capabilities are achieved through a multi-faceted training recipe that includes co-training with web-scale vision-language data, efficient fine-tuning from human trajectory data collected via VR devices, and effective imitation learning with robot trajectory data. In addition, we introduce ByteMini, a versatile bi-manual mobile robot designed with exceptional flexibility and reliability, capable of accomplishing a wide range of tasks when integrated with GR-3. Through extensive real-world experiments, we show GR-3 surpasses the state-of-the-art baseline method, $\pi_0$, on a wide variety of challenging tasks. We hope GR-3 can serve as a step towards building generalist robots capable of assisting humans in daily life.
>
---
#### [replaced 017] Asymptotically Optimal Lazy Lifelong Sampling-based Algorithm for Efficient Motion Planning in Dynamic Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.06521v3](http://arxiv.org/pdf/2409.06521v3)**

> **作者:** Lu Huang; Jingwen Yu; Jiankun Wang; Xingjian Jing
>
> **摘要:** The paper introduces an asymptotically optimal lifelong sampling-based path planning algorithm that combines the merits of lifelong planning algorithms and lazy search algorithms for rapid replanning in dynamic environments where edge evaluation is expensive. By evaluating only sub-path candidates for the optimal solution, the algorithm saves considerable evaluation time and thereby reduces the overall planning cost. It employs a novel informed rewiring cascade to efficiently repair the search tree when the underlying search graph changes. Theoretical analysis indicates that the proposed algorithm converges to the optimal solution as long as sufficient planning time is given. Planning results on robotic systems with $\mathbb{SE}(3)$ and $\mathbb{R}^7$ state spaces in challenging environments highlight the superior performance of the proposed algorithm over various state-of-the-art sampling-based planners in both static and dynamic motion planning tasks. The experiment of planning for a Turtlebot 4 operating in a dynamic environment with several moving pedestrians further verifies the feasibility and advantages of the proposed algorithm.
>
---
#### [replaced 018] Adapt On-the-Go: Behavior Modulation for Single-Life Robot Deployment
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2311.01059v3](http://arxiv.org/pdf/2311.01059v3)**

> **作者:** Annie S. Chen; Govind Chada; Laura Smith; Archit Sharma; Zipeng Fu; Sergey Levine; Chelsea Finn
>
> **摘要:** To succeed in the real world, robots must cope with situations that differ from those seen during training. We study the problem of adapting on-the-fly to such novel scenarios during deployment, by drawing upon a diverse repertoire of previouslylearned behaviors. Our approach, RObust Autonomous Modulation (ROAM), introduces a mechanism based on the perceived value of pre-trained behaviors to select and adapt pre-trained behaviors to the situation at hand. Crucially, this adaptation process all happens within a single episode at test time, without any human supervision. We demonstrate that ROAM enables a robot to adapt rapidly to changes in dynamics both in simulation and on a real Go1 quadruped, even successfully moving forward with roller skates on its feet. Our approach adapts over 2x as efficiently compared to existing methods when facing a variety of out-of-distribution situations during deployment by effectively choosing and adapting relevant behaviors on-the-fly.
>
---
#### [replaced 019] AI Space Cortex: An Experimental System for Future Era Space Exploration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.06574v3](http://arxiv.org/pdf/2507.06574v3)**

> **作者:** Thomas Touma; Ersin Daş; Erica Tevere; Martin Feather; Ksenia Kolcio; Maurice Prather; Alberto Candela; Ashish Goel; Erik Kramer; Hari Nayar; Lorraine Fesq; Joel W. Burdick
>
> **摘要:** Our Robust, Explainable Autonomy for Scientific Icy Moon Operations (REASIMO) effort contributes to NASA's Concepts for Ocean worlds Life Detection Technology (COLDTech) program, which explores science platform technologies for ocean worlds such as Europa and Enceladus. Ocean world missions pose significant operational challenges. These include long communication lags, limited power, and lifetime limitations caused by radiation damage and hostile conditions. Given these operational limitations, onboard autonomy will be vital for future Ocean world missions. Besides the management of nominal lander operations, onboard autonomy must react appropriately in the event of anomalies. Traditional spacecraft rely on a transition into 'safe-mode' in which non-essential components and subsystems are powered off to preserve safety and maintain communication with Earth. For a severely time-limited Ocean world mission, resolutions to these anomalies that can be executed without Earth-in-the-loop communication and associated delays are paramount for completion of the mission objectives and science goals. To address these challenges, the REASIMO effort aims to demonstrate a robust level of AI-assisted autonomy for such missions, including the ability to detect and recover from anomalies, and to perform missions based on pre-trained behaviors rather than hard-coded, predetermined logic like all prior space missions. We developed an AI-assisted, personality-driven, intelligent framework for control of an Ocean world mission by combining a mix of advanced technologies. To demonstrate the capabilities of the framework, we perform tests of autonomous sampling operations on a lander-manipulator testbed at the NASA Jet Propulsion Laboratory, approximating possible surface conditions such a mission might encounter.
>
---
#### [replaced 020] Progressive-Resolution Policy Distillation: Leveraging Coarse-Resolution Simulations for Time-Efficient Fine-Resolution Policy Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.07477v3](http://arxiv.org/pdf/2412.07477v3)**

> **作者:** Yuki Kadokawa; Hirotaka Tahara; Takamitsu Matsubara
>
> **备注:** accepted for IEEE Transactions on Automation Science and Engineering (T-ASE)
>
> **摘要:** In earthwork and construction, excavators often encounter large rocks mixed with various soil conditions, requiring skilled operators. This paper presents a framework for achieving autonomous excavation using reinforcement learning (RL) through a rock excavation simulator. In the simulation, resolution can be defined by the particle size/number in the whole soil space. Fine-resolution simulations closely mimic real-world behavior but demand significant calculation time and challenging sample collection, while coarse-resolution simulations enable faster sample collection but deviate from real-world behavior. To combine the advantages of both resolutions, we explore using policies developed in coarse-resolution simulations for pre-training in fine-resolution simulations. To this end, we propose a novel policy learning framework called Progressive-Resolution Policy Distillation (PRPD), which progressively transfers policies through some middle-resolution simulations with conservative policy transfer to avoid domain gaps that could lead to policy transfer failure. Validation in a rock excavation simulator and nine real-world rock environments demonstrated that PRPD reduced sampling time to less than 1/7 while maintaining task success rates comparable to those achieved through policy learning in a fine-resolution simulation.
>
---
#### [replaced 021] GeoFlow-SLAM: A Robust Tightly-Coupled RGBD-Inertial and Legged Odometry Fusion SLAM for Dynamic Legged Robotics
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.14247v3](http://arxiv.org/pdf/2503.14247v3)**

> **作者:** Tingyang Xiao; Xiaolin Zhou; Liu Liu; Wei Sui; Wei Feng; Jiaxiong Qiu; Xinjie Wang; Zhizhong Su
>
> **备注:** 8 pages
>
> **摘要:** This paper presents GeoFlow-SLAM, a robust and effective Tightly-Coupled RGBD-inertial SLAM for legged robotics undergoing aggressive and high-frequency motions.By integrating geometric consistency, legged odometry constraints, and dual-stream optical flow (GeoFlow), our method addresses three critical challenges:feature matching and pose initialization failures during fast locomotion and visual feature scarcity in texture-less scenes.Specifically, in rapid motion scenarios, feature matching is notably enhanced by leveraging dual-stream optical flow, which combines prior map points and poses. Additionally, we propose a robust pose initialization method for fast locomotion and IMU error in legged robots, integrating IMU/Legged odometry, inter-frame Perspective-n-Point (PnP), and Generalized Iterative Closest Point (GICP). Furthermore, a novel optimization framework that tightly couples depth-to-map and GICP geometric constraints is first introduced to improve the robustness and accuracy in long-duration, visually texture-less environments. The proposed algorithms achieve state-of-the-art (SOTA) on collected legged robots and open-source datasets. To further promote research and development, the open-source datasets and code will be made publicly available at https://github.com/HorizonRobotics/GeoFlowSlam
>
---
