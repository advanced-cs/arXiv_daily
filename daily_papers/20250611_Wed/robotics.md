# 机器人 cs.RO

- **最新发布 41 篇**

- **更新 32 篇**

## 最新发布

#### [new 001] Deploying SICNav in the Field: Safe and Interactive Crowd Navigation using MPC and Bilevel Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决拥挤环境中安全高效路径规划问题。提出SICNav方法，结合预测与规划，建模人机交互，提升导航可靠性。**

- **链接: [http://arxiv.org/pdf/2506.08851v1](http://arxiv.org/pdf/2506.08851v1)**

> **作者:** Sepehr Samavi; Garvish Bhutani; Florian Shkurti; Angela P. Schoellig
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics (non-archival)
>
> **摘要:** Safe and efficient navigation in crowded environments remains a critical challenge for robots that provide a variety of service tasks such as food delivery or autonomous wheelchair mobility. Classical robot crowd navigation methods decouple human motion prediction from robot motion planning, which neglects the closed-loop interactions between humans and robots. This lack of a model for human reactions to the robot plan (e.g. moving out of the way) can cause the robot to get stuck. Our proposed Safe and Interactive Crowd Navigation (SICNav) method is a bilevel Model Predictive Control (MPC) framework that combines prediction and planning into one optimization problem, explicitly modeling interactions among agents. In this paper, we present a systems overview of the crowd navigation platform we use to deploy SICNav in previously unseen indoor and outdoor environments. We provide a preliminary analysis of the system's operation over the course of nearly 7 km of autonomous navigation over two hours in both indoor and outdoor environments.
>
---
#### [new 002] ROS-related Robotic Systems Development with V-model-based Application of MeROS Metamodel
- **分类: cs.RO**

- **简介: 该论文属于机器人系统开发任务，解决ROS与MBSE集成不足的问题，提出基于MeROS和V模型的协同方法，提升系统一致性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2506.08706v1](http://arxiv.org/pdf/2506.08706v1)**

> **作者:** Tomasz Winiarski; Jan Kaniuka; Daniel Giełdowski; Jakub Ostrysz; Krystian Radlak; Dmytro Kushnir
>
> **备注:** 19 pages
>
> **摘要:** As robotic systems grow increasingly complex, heterogeneous, and safety-critical, the need for structured development methodologies becomes paramount. Although frameworks like the Robot Operating System (ROS) and Model-Based Systems Engineering (MBSE) offer foundational tools, they often lack integration when used together. This paper addresses that gap by aligning the widely recognized V-model development paradigm with the MeROS metamodel SysML-based modeling language tailored for ROS-based systems. We propose a domain-specific methodology that bridges ROS-centric modelling with systems engineering practices. Our approach formalises the structure, behaviour, and validation processes of robotic systems using MeROS, while extending it with a generalized, adaptable V-model compatible with both ROS and ROS 2. Rather than prescribing a fixed procedure, the approach supports project-specific flexibility and reuse, offering guidance across all stages of development. The approach is validated through a comprehensive case study on HeROS, a heterogeneous multi-robot platform comprising manipulators, mobile units, and dynamic test environments. This example illustrates how the MeROS-compatible V-model enhances traceability and system consistency while remaining accessible and extensible for future adaptation. The work contributes a structured, tool-agnostic foundation for developers and researchers seeking to apply MBSE practices in ROS-based projects.
>
---
#### [new 003] Human-Robot Teaming Field Deployments: A Comparison Between Verbal and Non-verbal Communication
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决医疗场景中机器人沟通方式的有效性问题。通过实验比较语音与非语音沟通对医护人员工作负荷的影响。**

- **链接: [http://arxiv.org/pdf/2506.08890v1](http://arxiv.org/pdf/2506.08890v1)**

> **作者:** Tauhid Tanjim; Promise Ekpo; Huajie Cao; Jonathan St. George; Kevin Ching; Hee Rin Lee; Angelique Taylor
>
> **备注:** This is the author's original submitted version of the paper accepted to the 2025 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). \c{opyright} 2025 IEEE. Personal use of this material is permitted. For any other use, please contact IEEE
>
> **摘要:** Healthcare workers (HCWs) encounter challenges in hospitals, such as retrieving medical supplies quickly from crash carts, which could potentially result in medical errors and delays in patient care. Robotic crash carts (RCCs) have shown promise in assisting healthcare teams during medical tasks through guided object searches and task reminders. Limited exploration has been done to determine what communication modalities are most effective and least disruptive to patient care in real-world settings. To address this gap, we conducted a between-subjects experiment comparing the RCC's verbal and non-verbal communication of object search with a standard crash cart in resuscitation scenarios to understand the impact of robot communication on workload and attitudes toward using robots in the workplace. Our findings indicate that verbal communication significantly reduced mental demand and effort compared to visual cues and with a traditional crash cart. Although frustration levels were slightly higher during collaborations with the robot compared to a traditional cart, these research insights provide valuable implications for human-robot teamwork in high-stakes environments.
>
---
#### [new 004] Re4MPC: Reactive Nonlinear MPC for Multi-model Motion Planning via Deep Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于机器人运动规划任务，解决高自由度机器人轨迹计算效率低的问题。提出Re4MPC方法，结合非线性MPC与深度强化学习，提升计算效率和成功率。**

- **链接: [http://arxiv.org/pdf/2506.08344v1](http://arxiv.org/pdf/2506.08344v1)**

> **作者:** Neşet Ünver Akmandor; Sarvesh Prajapati; Mark Zolotas; Taşkın Padır
>
> **备注:** Accepted to the 2025 IEEE International Conference on Automation Science and Engineering (CASE)
>
> **摘要:** Traditional motion planning methods for robots with many degrees-of-freedom, such as mobile manipulators, are often computationally prohibitive for real-world settings. In this paper, we propose a novel multi-model motion planning pipeline, termed Re4MPC, which computes trajectories using Nonlinear Model Predictive Control (NMPC). Re4MPC generates trajectories in a computationally efficient manner by reactively selecting the model, cost, and constraints of the NMPC problem depending on the complexity of the task and robot state. The policy for this reactive decision-making is learned via a Deep Reinforcement Learning (DRL) framework. We introduce a mathematical formulation to integrate NMPC into this DRL framework. To validate our methodology and design choices, we evaluate DRL training and test outcomes in a physics-based simulation involving a mobile manipulator. Experimental results demonstrate that Re4MPC is more computationally efficient and achieves higher success rates in reaching end-effector goals than the NMPC baseline, which computes whole-body trajectories without our learning mechanism.
>
---
#### [new 005] PhyBlock: A Progressive Benchmark for Physical Understanding and Planning via 3D Block Assembly
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出PhyBlock，用于评估视觉语言模型在物理理解和规划方面的能力，解决其在3D环境中的推理不足问题。**

- **链接: [http://arxiv.org/pdf/2506.08708v1](http://arxiv.org/pdf/2506.08708v1)**

> **作者:** Liang Ma; Jiajun Wen; Min Lin; Rongtao Xu; Xiwen Liang; Bingqian Lin; Jun Ma; Yongxin Wang; Ziming Wei; Haokun Lin; Mingfei Han; Meng Cao; Bokui Chen; Ivan Laptev; Xiaodan Liang
>
> **摘要:** While vision-language models (VLMs) have demonstrated promising capabilities in reasoning and planning for embodied agents, their ability to comprehend physical phenomena, particularly within structured 3D environments, remains severely limited. To close this gap, we introduce PhyBlock, a progressive benchmark designed to assess VLMs on physical understanding and planning through robotic 3D block assembly tasks. PhyBlock integrates a novel four-level cognitive hierarchy assembly task alongside targeted Visual Question Answering (VQA) samples, collectively aimed at evaluating progressive spatial reasoning and fundamental physical comprehension, including object properties, spatial relationships, and holistic scene understanding. PhyBlock includes 2600 block tasks (400 assembly tasks, 2200 VQA tasks) and evaluates models across three key dimensions: partial completion, failure diagnosis, and planning robustness. We benchmark 21 state-of-the-art VLMs, highlighting their strengths and limitations in physically grounded, multi-step planning. Our empirical findings indicate that the performance of VLMs exhibits pronounced limitations in high-level planning and reasoning capabilities, leading to a notable decline in performance for the growing complexity of the tasks. Error analysis reveals persistent difficulties in spatial orientation and dependency reasoning. Surprisingly, chain-of-thought prompting offers minimal improvements, suggesting spatial tasks heavily rely on intuitive model comprehension. We position PhyBlock as a unified testbed to advance embodied reasoning, bridging vision-language understanding and real-world physical problem-solving.
>
---
#### [new 006] TensorTouch: Calibration of Tactile Sensors for High Resolution Stress Tensor and Deformation for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在解决高精度应力和形变测量问题。通过结合有限元分析与深度学习，TensorTouch提升光学触觉传感器的解析能力与跨传感器迁移性。**

- **链接: [http://arxiv.org/pdf/2506.08291v1](http://arxiv.org/pdf/2506.08291v1)**

> **作者:** Won Kyung Do; Matthew Strong; Aiden Swann; Boshu Lei; Monroe Kennedy III
>
> **摘要:** Advanced dexterous manipulation involving multiple simultaneous contacts across different surfaces, like pinching coins from ground or manipulating intertwined objects, remains challenging for robotic systems. Such tasks exceed the capabilities of vision and proprioception alone, requiring high-resolution tactile sensing with calibrated physical metrics. Raw optical tactile sensor images, while information-rich, lack interpretability and cross-sensor transferability, limiting their real-world utility. TensorTouch addresses this challenge by integrating finite element analysis with deep learning to extract comprehensive contact information from optical tactile sensors, including stress tensors, deformation fields, and force distributions at pixel-level resolution. The TensorTouch framework achieves sub-millimeter position accuracy and precise force estimation while supporting large sensor deformations crucial for manipulating soft objects. Experimental validation demonstrates 90% success in selectively grasping one of two strings based on detected motion, enabling new contact-rich manipulation capabilities previously inaccessible to robotic systems.
>
---
#### [new 007] Fast Estimation of Globally Optimal Independent Contact Regions for Robust Grasping and Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决快速计算全局最优独立接触区域的问题。提出一种分治算法，提升计算效率与抓取鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.08856v1](http://arxiv.org/pdf/2506.08856v1)**

> **作者:** Jonathan P. King; Harnoor Ahluwalia; Michael Zhang; Nancy S. Pollard
>
> **备注:** Submitted to IEEE Conference on Humanoid Robots
>
> **摘要:** This work presents a fast anytime algorithm for computing globally optimal independent contact regions (ICRs). ICRs are regions such that one contact within each region enables a valid grasp. Locations of ICRs can provide guidance for grasp and manipulation planning, learning, and policy transfer. However, ICRs for modern applications have been little explored, in part due to the expense of computing them, as they have a search space exponential in the number of contacts. We present a divide and conquer algorithm based on incremental n-dimensional Delaunay triangulation that produces results with bounded suboptimality in times sufficient for real-time planning. This paper presents the base algorithm for grasps where contacts lie within a plane. Our experiments show substantial benefits over competing grasp quality metrics and speedups of 100X and more for competing approaches to computing ICRs. We explore robustness of a policy guided by ICRs and outline a path to general 3D implementation. Code will be released on publication to facilitate further development and applications.
>
---
#### [new 008] UAVs Meet Agentic AI: A Multidomain Survey of Autonomous Aerial Intelligence and Agentic UAVs
- **分类: cs.RO; cs.AI**

- **简介: 本文综述了智能无人机（Agentic UAVs）在多个领域的应用与技术挑战，探讨其自主决策与协作能力，旨在推动其未来发展与部署。**

- **链接: [http://arxiv.org/pdf/2506.08045v1](http://arxiv.org/pdf/2506.08045v1)**

> **作者:** Ranjan Sapkota; Konstantinos I. Roumeliotis; Manoj Karkee
>
> **备注:** 40 pages, 6 Figures
>
> **摘要:** Agentic UAVs represent a new frontier in autonomous aerial intelligence, integrating perception, decision-making, memory, and collaborative planning to operate adaptively in complex, real-world environments. Driven by recent advances in Agentic AI, these systems surpass traditional UAVs by exhibiting goal-driven behavior, contextual reasoning, and interactive autonomy. We provide a comprehensive foundation for understanding the architectural components and enabling technologies that distinguish Agentic UAVs from traditional autonomous UAVs. Furthermore, a detailed comparative analysis highlights advancements in autonomy with AI agents, learning, and mission flexibility. This study explores seven high-impact application domains precision agriculture, construction & mining, disaster response, environmental monitoring, infrastructure inspection, logistics, security, and wildlife conservation, illustrating the broad societal value of agentic aerial intelligence. Furthermore, we identify key challenges in technical constraints, regulatory limitations, and data-model reliability, and we present emerging solutions across hardware innovation, learning architectures, and human-AI interaction. Finally, a future roadmap is proposed, outlining pathways toward self-evolving aerial ecosystems, system-level collaboration, and sustainable, equitable deployments. This survey establishes a foundational framework for the future development, deployment, and governance of agentic aerial systems (Agentic UAVs) across diverse societal and industrial domains.
>
---
#### [new 009] Attention-based Learning for 3D Informative Path Planning
- **分类: cs.RO**

- **简介: 该论文属于3D信息路径规划任务，旨在解决空中机器人在有限时间内高效感知环境的问题。通过注意力机制提升路径规划效果，实现探索与利用的平衡。**

- **链接: [http://arxiv.org/pdf/2506.08434v1](http://arxiv.org/pdf/2506.08434v1)**

> **作者:** Rui Zhao; Xingjian Zhang; Yuhong Cao; Yizhuo Wang; Guillaume Sartoretti
>
> **摘要:** In this work, we propose an attention-based deep reinforcement learning approach to address the adaptive informative path planning (IPP) problem in 3D space, where an aerial robot equipped with a downward-facing sensor must dynamically adjust its 3D position to balance sensing footprint and accuracy, and finally obtain a high-quality belief of an underlying field of interest over a given domain (e.g., presence of specific plants, hazardous gas, geological structures, etc.). In adaptive IPP tasks, the agent is tasked with maximizing information collected under time/distance constraints, continuously adapting its path based on newly acquired sensor data. To this end, we leverage attention mechanisms for their strong ability to capture global spatial dependencies across large action spaces, allowing the agent to learn an implicit estimation of environmental transitions. Our model builds a contextual belief representation over the entire domain, guiding sequential movement decisions that optimize both short- and long-term search objectives. Comparative evaluations against state-of-the-art planners demonstrate that our approach significantly reduces environmental uncertainty within constrained budgets, thus allowing the agent to effectively balance exploration and exploitation. We further show our model generalizes well to environments of varying sizes, highlighting its potential for many real-world applications.
>
---
#### [new 010] HiBerNAC: Hierarchical Brain-emulated Robotic Neural Agent Collective for Disentangling Complex Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决复杂操作中的长期规划与多智能体协作问题。提出HiBerNAC框架，结合多模态理解和神经启发机制，提升任务执行效率与成功率。**

- **链接: [http://arxiv.org/pdf/2506.08296v1](http://arxiv.org/pdf/2506.08296v1)**

> **作者:** Hongjun Wu; Heng Zhang; Pengsong Zhang; Jin Wang; Cong Wang
>
> **备注:** 31 pages,5 figures
>
> **摘要:** Recent advances in multimodal vision-language-action (VLA) models have revolutionized traditional robot learning, enabling systems to interpret vision, language, and action in unified frameworks for complex task planning. However, mastering complex manipulation tasks remains an open challenge, constrained by limitations in persistent contextual memory, multi-agent coordination under uncertainty, and dynamic long-horizon planning across variable sequences. To address this challenge, we propose \textbf{HiBerNAC}, a \textbf{Hi}erarchical \textbf{B}rain-\textbf{e}mulated \textbf{r}obotic \textbf{N}eural \textbf{A}gent \textbf{C}ollective, inspired by breakthroughs in neuroscience, particularly in neural circuit mechanisms and hierarchical decision-making. Our framework combines: (1) multimodal VLA planning and reasoning with (2) neuro-inspired reflection and multi-agent mechanisms, specifically designed for complex robotic manipulation tasks. By leveraging neuro-inspired functional modules with decentralized multi-agent collaboration, our approach enables robust and enhanced real-time execution of complex manipulation tasks. In addition, the agentic system exhibits scalable collective intelligence via dynamic agent specialization, adapting its coordination strategy to variable task horizons and complexity. Through extensive experiments on complex manipulation tasks compared with state-of-the-art VLA models, we demonstrate that \textbf{HiBerNAC} reduces average long-horizon task completion time by 23\%, and achieves non-zero success rates (12\textendash 31\%) on multi-path tasks where prior state-of-the-art VLA models consistently fail. These results provide indicative evidence for bridging biological cognition and robotic learning mechanisms.
>
---
#### [new 011] Towards Biosignals-Free Autonomous Prosthetic Hand Control via Imitation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主假肢手控制任务，旨在解决传统肌电控制的高负担问题。通过模仿学习，利用摄像头实现无需生物信号的自动抓取与释放。**

- **链接: [http://arxiv.org/pdf/2506.08795v1](http://arxiv.org/pdf/2506.08795v1)**

> **作者:** Kaijie Shi; Wanglong Lu; Hanli Zhao; Vinicius Prado da Fonseca; Ting Zou; Xianta Jiang
>
> **摘要:** Limb loss affects millions globally, impairing physical function and reducing quality of life. Most traditional surface electromyographic (sEMG) and semi-autonomous methods require users to generate myoelectric signals for each control, imposing physically and mentally taxing demands. This study aims to develop a fully autonomous control system that enables a prosthetic hand to automatically grasp and release objects of various shapes using only a camera attached to the wrist. By placing the hand near an object, the system will automatically execute grasping actions with a proper grip force in response to the hand's movements and the environment. To release the object being grasped, just naturally place the object close to the table and the system will automatically open the hand. Such a system would provide individuals with limb loss with a very easy-to-use prosthetic control interface and greatly reduce mental effort while using. To achieve this goal, we developed a teleoperation system to collect human demonstration data for training the prosthetic hand control model using imitation learning, which mimics the prosthetic hand actions from human. Through training the model using only a few objects' data from one single participant, we have shown that the imitation learning algorithm can achieve high success rates, generalizing to more individuals and unseen objects with a variation of weights. The demonstrations are available at \href{https://sites.google.com/view/autonomous-prosthetic-hand}{https://sites.google.com/view/autonomous-prosthetic-hand}
>
---
#### [new 012] Deep Reinforcement Learning-Based Motion Planning and PDE Control for Flexible Manipulators
- **分类: cs.RO; math-ph; math.MP**

- **简介: 该论文属于机器人运动规划与控制任务，旨在解决柔性机械臂的振动问题。通过结合深度强化学习与非线性PDE控制器，优化轨迹并提升控制精度。**

- **链接: [http://arxiv.org/pdf/2506.08639v1](http://arxiv.org/pdf/2506.08639v1)**

> **作者:** Amir Hossein Barjini; Seyed Adel Alizadeh Kolagar; Sadeq Yaqubi; Jouni Mattila
>
> **摘要:** This article presents a motion planning and control framework for flexible robotic manipulators, integrating deep reinforcement learning (DRL) with a nonlinear partial differential equation (PDE) controller. Unlike conventional approaches that focus solely on control, we demonstrate that the desired trajectory significantly influences endpoint vibrations. To address this, a DRL motion planner, trained using the soft actor-critic (SAC) algorithm, generates optimized trajectories that inherently minimize vibrations. The PDE nonlinear controller then computes the required torques to track the planned trajectory while ensuring closed-loop stability using Lyapunov analysis. The proposed methodology is validated through both simulations and real-world experiments, demonstrating superior vibration suppression and tracking accuracy compared to traditional methods. The results underscore the potential of combining learning-based motion planning with model-based control for enhancing the precision and stability of flexible robotic manipulators.
>
---
#### [new 013] FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉-动作策略任务，旨在解决生成式模型在实时系统中推理成本高的问题。通过引入频率一致性约束，提升动作生成效率与质量。**

- **链接: [http://arxiv.org/pdf/2506.08822v1](http://arxiv.org/pdf/2506.08822v1)**

> **作者:** Yifei Su; Ning Liu; Dong Chen; Zhen Zhao; Kun Wu; Meng Li; Zhiyuan Xu; Zhengping Che; Jian Tang
>
> **摘要:** Generative modeling-based visuomotor policies have been widely adopted in robotic manipulation attributed to their ability to model multimodal action distributions. However, the high inference cost of multi-step sampling limits their applicability in real-time robotic systems. To address this issue, existing approaches accelerate the sampling process in generative modeling-based visuomotor policies by adapting acceleration techniques originally developed for image generation. Despite this progress, a major distinction remains: image generation typically involves producing independent samples without temporal dependencies, whereas robotic manipulation involves generating time-series action trajectories that require continuity and temporal coherence. To effectively exploit temporal information in robotic manipulation, we propose FreqPolicy, a novel approach that first imposes frequency consistency constraints on flow-based visuomotor policies. Our work enables the action model to capture temporal structure effectively while supporting efficient, high-quality one-step action generation. We introduce a frequency consistency constraint that enforces alignment of frequency-domain action features across different timesteps along the flow, thereby promoting convergence of one-step action generation toward the target distribution. In addition, we design an adaptive consistency loss to capture structural temporal variations inherent in robotic manipulation tasks. We assess FreqPolicy on 53 tasks across 3 simulation benchmarks, proving its superiority over existing one-step action generators. We further integrate FreqPolicy into the vision-language-action (VLA) model and achieve acceleration without performance degradation on the 40 tasks of Libero. Besides, we show efficiency and effectiveness in real-world robotic scenarios with an inference frequency 93.5Hz. The code will be publicly available.
>
---
#### [new 014] MoRE: Mixture of Residual Experts for Humanoid Lifelike Gaits Learning on Complex Terrains
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决复杂地形上实现类人步态的问题。通过引入残差专家混合框架，提升机器人在不同地形上的适应能力和步态切换效果。**

- **链接: [http://arxiv.org/pdf/2506.08840v1](http://arxiv.org/pdf/2506.08840v1)**

> **作者:** Dewei Wang; Xinmiao Wang; Xinzhe Liu; Jiyuan Shi; Yingnan Zhao; Chenjia Bai; Xuelong Li
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Humanoid robots have demonstrated robust locomotion capabilities using Reinforcement Learning (RL)-based approaches. Further, to obtain human-like behaviors, existing methods integrate human motion-tracking or motion prior in the RL framework. However, these methods are limited in flat terrains with proprioception only, restricting their abilities to traverse challenging terrains with human-like gaits. In this work, we propose a novel framework using a mixture of latent residual experts with multi-discriminators to train an RL policy, which is capable of traversing complex terrains in controllable lifelike gaits with exteroception. Our two-stage training pipeline first teaches the policy to traverse complex terrains using a depth camera, and then enables gait-commanded switching between human-like gait patterns. We also design gait rewards to adjust human-like behaviors like robot base height. Simulation and real-world experiments demonstrate that our framework exhibits exceptional performance in traversing complex terrains, and achieves seamless transitions between multiple human-like gait patterns.
>
---
#### [new 015] Adaptive Per-Tree Canopy Volume Estimation Using Mobile LiDAR in Structured and Unstructured Orchards
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于果树冠层体积估计任务，解决结构复杂果园中实时、非侵入式监测问题。通过移动LiDAR与自适应分割算法实现精准体积估算。**

- **链接: [http://arxiv.org/pdf/2506.08061v1](http://arxiv.org/pdf/2506.08061v1)**

> **作者:** Ali Abedi; Fernando Cladera; Mohsen Farajijalal; Reza Ehsani
>
> **备注:** 5 pages, 3 figures, Accepted to the Novel Approaches for Precision Agriculture and Forestry with Autonomous Robots IEEE ICRA Workshop - 2025
>
> **摘要:** We present a real-time system for per-tree canopy volume estimation using mobile LiDAR data collected during routine robotic navigation. Unlike prior approaches that rely on static scans or assume uniform orchard structures, our method adapts to varying field geometries via an integrated pipeline of LiDAR-inertial odometry, adaptive segmentation, and geometric reconstruction. We evaluate the system across two commercial orchards, one pistachio orchard with regular spacing and one almond orchard with dense, overlapping crowns. A hybrid clustering strategy combining DBSCAN and spectral clustering enables robust per-tree segmentation, achieving 93% success in pistachio and 80% in almond, with strong agreement to drone derived canopy volume estimates. This work advances scalable, non-intrusive tree monitoring for structurally diverse orchard environments.
>
---
#### [new 016] Bayesian Inverse Physics for Neuro-Symbolic Robot Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决传统方法在动态环境中的适应性与可靠性问题。通过结合物理建模、贝叶斯推理和元学习，提出混合神经符号架构以提升机器人泛化与推理能力。**

- **链接: [http://arxiv.org/pdf/2506.08756v1](http://arxiv.org/pdf/2506.08756v1)**

> **作者:** Octavio Arriaga; Rebecca Adam; Melvin Laux; Lisa Gutzeit; Marco Ragni; Jan Peters; Frank Kirchner
>
> **摘要:** Real-world robotic applications, from autonomous exploration to assistive technologies, require adaptive, interpretable, and data-efficient learning paradigms. While deep learning architectures and foundation models have driven significant advances in diverse robotic applications, they remain limited in their ability to operate efficiently and reliably in unknown and dynamic environments. In this position paper, we critically assess these limitations and introduce a conceptual framework for combining data-driven learning with deliberate, structured reasoning. Specifically, we propose leveraging differentiable physics for efficient world modeling, Bayesian inference for uncertainty-aware decision-making, and meta-learning for rapid adaptation to new tasks. By embedding physical symbolic reasoning within neural models, robots could generalize beyond their training data, reason about novel situations, and continuously expand their knowledge. We argue that such hybrid neuro-symbolic architectures are essential for the next generation of autonomous systems, and to this end, we provide a research roadmap to guide and accelerate their development.
>
---
#### [new 017] MOMAV: A highly symmetrical fully-actuated multirotor drone using optimizing control allocation
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，旨在解决多旋翼无人机的高效控制与对称性问题。通过创新设计和优化控制算法，提升飞行精度与效率。**

- **链接: [http://arxiv.org/pdf/2506.08868v1](http://arxiv.org/pdf/2506.08868v1)**

> **作者:** Marco Ruggia
>
> **备注:** 12 pages, 12 figures, preprint
>
> **摘要:** MOMAV (Marco's Omnidirectional Micro Aerial Vehicle) is a multirotor drone that is fully actuated, meaning it can control its orientation independently of its position. MOMAV is also highly symmetrical, making its flight efficiency largely unaffected by its current orientation. These characteristics are achieved by a novel drone design where six rotor arms align with the vertices of an octahedron, and where each arm can actively rotate along its long axis. Various standout features of MOMAV are presented: The high flight efficiency compared to arm configuration of other fully-actuated drones, the design of an original rotating arm assembly featuring slip-rings used to enable continuous arm rotation, and a novel control allocation algorithm based on sequential quadratic programming (SQP) used to calculate throttle and arm-angle setpoints in flight. Flight tests have shown that MOMAV is able to achieve remarkably low mean position/orientation errors of 6.6mm, 2.1{\deg} ({\sigma}: 3.0mm, 1.0{\deg}) when sweeping position setpoints, and 11.8mm, 3.3{\deg} ({\sigma}: 8.6mm, 2.0{\deg}) when sweeping orientation setpoints.
>
---
#### [new 018] CLONE: Closed-Loop Whole-Body Humanoid Teleoperation for Long-Horizon Tasks
- **分类: cs.RO**

- **简介: 该论文属于人形机器人远距离操作任务，解决长期操作中的位置漂移和肢体协调问题。提出CLONE系统，通过闭环反馈实现高精度全身操作。**

- **链接: [http://arxiv.org/pdf/2506.08931v1](http://arxiv.org/pdf/2506.08931v1)**

> **作者:** Yixuan Li; Yutang Lin; Jieming Cui; Tengyu Liu; Wei Liang; Yixin Zhu; Siyuan Huang
>
> **备注:** 18 pages, 13 figures
>
> **摘要:** Humanoid teleoperation plays a vital role in demonstrating and collecting data for complex humanoid-scene interactions. However, current teleoperation systems face critical limitations: they decouple upper- and lower-body control to maintain stability, restricting natural coordination, and operate open-loop without real-time position feedback, leading to accumulated drift. The fundamental challenge is achieving precise, coordinated whole-body teleoperation over extended durations while maintaining accurate global positioning. Here we show that an MoE-based teleoperation system, CLONE, with closed-loop error correction enables unprecedented whole-body teleoperation fidelity, maintaining minimal positional drift over long-range trajectories using only head and hand tracking from an MR headset. Unlike previous methods that either sacrifice coordination for stability or suffer from unbounded drift, CLONE learns diverse motion skills while preventing tracking error accumulation through real-time feedback, enabling complex coordinated movements such as ``picking up objects from the ground.'' These results establish a new milestone for whole-body humanoid teleoperation for long-horizon humanoid-scene interaction tasks.
>
---
#### [new 019] Noise Analysis and Hierarchical Adaptive Body State Estimator For Biped Robot Walking With ESVC Foot
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，旨在解决ESVC足行走时的噪声与状态估计问题。通过噪声分析和分层自适应估计器设计，提升估计精度与收敛速度。**

- **链接: [http://arxiv.org/pdf/2506.08578v1](http://arxiv.org/pdf/2506.08578v1)**

> **作者:** Boyang Chen; Xizhe Zang; Chao Song; Yue Zhang; Xuehe Zhang; Jie Zhao
>
> **摘要:** The ESVC(Ellipse-based Segmental Varying Curvature) foot, a robot foot design inspired by the rollover shape of the human foot, significantly enhances the energy efficiency of the robot walking gait. However, due to the tilt of the supporting leg, the error of the contact model are amplified, making robot state estimation more challenging. Therefore, this paper focuses on the noise analysis and state estimation for robot walking with the ESVC foot. First, through physical robot experiments, we investigate the effect of the ESVC foot on robot measurement noise and process noise. and a noise-time regression model using sliding window strategy is developed. Then, a hierarchical adaptive state estimator for biped robots with the ESVC foot is proposed. The state estimator consists of two stages: pre-estimation and post-estimation. In the pre-estimation stage, a data fusion-based estimation is employed to process the sensory data. During post-estimation, the acceleration of center of mass is first estimated, and then the noise covariance matrices are adjusted based on the regression model. Following that, an EKF(Extended Kalman Filter) based approach is applied to estimate the centroid state during robot walking. Physical experiments demonstrate that the proposed adaptive state estimator for biped robot walking with the ESVC foot not only provides higher precision than both EKF and Adaptive EKF, but also converges faster under varying noise conditions.
>
---
#### [new 020] Diffusion Models for Safety Validation of Autonomous Driving Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶安全验证任务，旨在解决真实测试风险高、成本大的问题。通过训练扩散模型生成潜在故障场景，实现高效安全验证。**

- **链接: [http://arxiv.org/pdf/2506.08459v1](http://arxiv.org/pdf/2506.08459v1)**

> **作者:** Juanran Wang; Marc R. Schlichting; Harrison Delecki; Mykel J. Kochenderfer
>
> **摘要:** Safety validation of autonomous driving systems is extremely challenging due to the high risks and costs of real-world testing as well as the rarity and diversity of potential failures. To address these challenges, we train a denoising diffusion model to generate potential failure cases of an autonomous vehicle given any initial traffic state. Experiments on a four-way intersection problem show that in a variety of scenarios, the diffusion model can generate realistic failure samples while capturing a wide variety of potential failures. Our model does not require any external training dataset, can perform training and inference with modest computing resources, and does not assume any prior knowledge of the system under test, with applicability to safety validation for traffic intersections.
>
---
#### [new 021] Periodic Bipedal Gait Learning Using Reward Composition Based on a Novel Gait Planner for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人步态学习任务，旨在解决 humanoid 机器人周期性行走问题。通过奖励组合与新型步态规划器，提升学习效率与运动性能。**

- **链接: [http://arxiv.org/pdf/2506.08416v1](http://arxiv.org/pdf/2506.08416v1)**

> **作者:** Bolin Li; Linwei Sun; Xuecong Huang; Yuzhi Jiang; Lijun Zhu
>
> **摘要:** This paper presents a periodic bipedal gait learning method using reward composition, integrated with a real-time gait planner for humanoid robots. First, we introduce a novel gait planner that incorporates dynamics to design the desired joint trajectory. In the gait design process, the 3D robot model is decoupled into two 2D models, which are then approximated as hybrid inverted pendulums (H-LIP) for trajectory planning. The gait planner operates in parallel in real time within the robot's learning environment. Second, based on this gait planner, we design three effective reward functions within a reinforcement learning framework, forming a reward composition to achieve periodic bipedal gait. This reward composition reduces the robot's learning time and enhances locomotion performance. Finally, a gait design example and performance comparison are presented to demonstrate the effectiveness of the proposed method.
>
---
#### [new 022] Ego-centric Learning of Communicative World Models for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶领域的多智能体强化学习任务，旨在解决部分可观测性和非平稳性问题。通过构建通信世界模型实现轻量信息共享与个体学习，提升预测精度和规划效果。**

- **链接: [http://arxiv.org/pdf/2506.08149v1](http://arxiv.org/pdf/2506.08149v1)**

> **作者:** Hang Wang; Dechen Gao; Junshan Zhang
>
> **摘要:** We study multi-agent reinforcement learning (MARL) for tasks in complex high-dimensional environments, such as autonomous driving. MARL is known to suffer from the \textit{partial observability} and \textit{non-stationarity} issues. To tackle these challenges, information sharing is often employed, which however faces major hurdles in practice, including overwhelming communication overhead and scalability concerns. By making use of generative AI embodied in world model together with its latent representation, we develop {\it CALL}, \underline{C}ommunic\underline{a}tive Wor\underline{l}d Mode\underline{l}, for MARL, where 1) each agent first learns its world model that encodes its state and intention into low-dimensional latent representation with smaller memory footprint, which can be shared with other agents of interest via lightweight communication; and 2) each agent carries out ego-centric learning while exploiting lightweight information sharing to enrich her world model, and then exploits its generalization capacity to improve prediction for better planning. We characterize the gain on the prediction accuracy from the information sharing and its impact on performance gap. Extensive experiments are carried out on the challenging local trajectory planning tasks in the CARLA platform to demonstrate the performance gains of using \textit{CALL}.
>
---
#### [new 023] TGRPO :Fine-tuning Vision-Language-Action Model via Trajectory-wise Group Relative Policy Optimization
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的微调任务，旨在解决传统方法依赖静态数据、缺乏环境交互的问题。提出TGRPO方法，融合优势信号提升在线强化学习效果。**

- **链接: [http://arxiv.org/pdf/2506.08440v1](http://arxiv.org/pdf/2506.08440v1)**

> **作者:** Zengjue Chen; Runliang Niu; He Kong; Qi Wang
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) model have demonstrated strong generalization capabilities across diverse scenes, tasks, and robotic platforms when pretrained at large-scale datasets. However, these models still require task-specific fine-tuning in novel environments, a process that relies almost exclusively on supervised fine-tuning (SFT) using static trajectory datasets. Such approaches neither allow robot to interact with environment nor do they leverage feedback from live execution. Also, their success is critically dependent on the size and quality of the collected trajectories. Reinforcement learning (RL) offers a promising alternative by enabling closed-loop interaction and aligning learned policies directly with task objectives. In this work, we draw inspiration from the ideas of GRPO and propose the Trajectory-wise Group Relative Policy Optimization (TGRPO) method. By fusing step-level and trajectory-level advantage signals, this method improves GRPO's group-level advantage estimation, thereby making the algorithm more suitable for online reinforcement learning training of VLA. Experimental results on ten manipulation tasks from the libero-object benchmark demonstrate that TGRPO consistently outperforms various baseline methods, capable of generating more robust and efficient policies across multiple tested scenarios. Our source codes are available at: https://github.com/hahans/TGRPO
>
---
#### [new 024] AI Magnetic Levitation (Maglev) Conveyor for Automated Assembly Production
- **分类: cs.RO**

- **简介: 该论文属于自动化生产任务，旨在解决传统生产线效率低、成本高的问题。提出AI磁悬浮传送系统，结合AI与磁悬浮技术，提升效率与灵活性。**

- **链接: [http://arxiv.org/pdf/2506.08039v1](http://arxiv.org/pdf/2506.08039v1)**

> **作者:** Ray Wai Man Kong
>
> **备注:** 12 pages, 9 Figures
>
> **摘要:** Efficiency, speed, and precision are essential in modern manufacturing. AI Maglev Conveyor system, combining magnetic levitation (maglev) technology with artificial intelligence (AI), revolutionizes automated production processes. This system reduces maintenance costs and downtime by eliminating friction, enhancing operational efficiency. It transports goods swiftly with minimal energy consumption, optimizing resource use and supporting sustainability. AI integration enables real-time monitoring and adaptive control, allowing businesses to respond to production demand fluctuations and streamline supply chain operations. The AI Maglev Conveyor offers smooth, silent operation, accommodating diverse product types and sizes for flexible manufacturing without extensive reconfiguration. AI algorithms optimize routing, reduce cycle times, and improve throughput, creating an agile production line adaptable to market changes. This applied research paper introduces the Maglev Conveyor system, featuring an electromagnetic controller and multiple movers to enhance automation. It offers cost savings as an alternative to setups using six-axis robots or linear motors, with precise adjustments for robotic arm loading. Operating at high speeds minimizes treatment time for delicate components while maintaining precision. Its adaptable design accommodates various materials, facilitating integration of processing stations alongside electronic product assembly. Positioned between linear-axis and robotic systems in cost, the Maglev Conveyor is ideal for flat parts requiring minimal travel, transforming production efficiency across industries. It explores its technical advantages, flexibility, cost reductions, and overall benefits.
>
---
#### [new 025] Modular Recurrence in Contextual MDPs for Universal Morphology Control
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于多机器人控制任务，旨在提升控制器对未见机器人形态的泛化能力。通过模块化递归结构和上下文信息推理，增强了对新动力学、运动学和拓扑结构的适应性。**

- **链接: [http://arxiv.org/pdf/2506.08630v1](http://arxiv.org/pdf/2506.08630v1)**

> **作者:** Laurens Engwegen; Daan Brinks; Wendelin Böhmer
>
> **摘要:** A universal controller for any robot morphology would greatly improve computational and data efficiency. By utilizing contextual information about the properties of individual robots and exploiting their modular structure in the architecture of deep reinforcement learning agents, steps have been made towards multi-robot control. Generalization to new, unseen robots, however, remains a challenge. In this paper we hypothesize that the relevant contextual information is partially observable, but that it can be inferred through interactions for better generalization to contexts that are not seen during training. To this extent, we implement a modular recurrent architecture and evaluate its generalization performance on a large set of MuJoCo robots. The results show a substantial improved performance on robots with unseen dynamics, kinematics, and topologies, in four different environments.
>
---
#### [new 026] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决动态环境中视觉驱动的协作问题。提出VIKI-Bench基准和VIKI-R框架，提升多机器人协作性能。**

- **链接: [http://arxiv.org/pdf/2506.09049v1](http://arxiv.org/pdf/2506.09049v1)**

> **作者:** Li Kang; Xiufeng Song; Heng Zhou; Yiran Qin; Jie Yang; Xiaohong Liu; Philip Torr; Lei Bai; Zhenfei Yin
>
> **备注:** Project page: https://faceong.github.io/VIKI-R/
>
> **摘要:** Coordinating multiple embodied agents in dynamic environments remains a core challenge in artificial intelligence, requiring both perception-driven reasoning and scalable cooperation strategies. While recent works have leveraged large language models (LLMs) for multi-agent planning, a few have begun to explore vision-language models (VLMs) for visual reasoning. However, these VLM-based approaches remain limited in their support for diverse embodiment types. In this work, we introduce VIKI-Bench, the first hierarchical benchmark tailored for embodied multi-agent cooperation, featuring three structured levels: agent activation, task planning, and trajectory perception. VIKI-Bench includes diverse robot embodiments, multi-view visual observations, and structured supervision signals to evaluate reasoning grounded in visual inputs. To demonstrate the utility of VIKI-Bench, we propose VIKI-R, a two-stage framework that fine-tunes a pretrained vision-language model (VLM) using Chain-of-Thought annotated demonstrations, followed by reinforcement learning under multi-level reward signals. Our extensive experiments show that VIKI-R significantly outperforms baselines method across all task levels. Furthermore, we show that reinforcement learning enables the emergence of compositional cooperation patterns among heterogeneous agents. Together, VIKI-Bench and VIKI-R offer a unified testbed and method for advancing multi-agent, visual-driven cooperation in embodied AI systems.
>
---
#### [new 027] Towards Reliable AR-Guided Surgical Navigation: Interactive Deformation Modeling with Data-Driven Biomechanics and Prompts
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于手术导航中的变形建模任务，旨在解决AR导航中因器官形变导致的模型对齐问题。通过数据驱动方法和人机交互机制提升计算效率与精度。**

- **链接: [http://arxiv.org/pdf/2506.08048v1](http://arxiv.org/pdf/2506.08048v1)**

> **作者:** Zheng Han; Jun Zhou; Jialun Pei; Jing Qin; Yingfang Fan; Qi Dou
>
> **摘要:** In augmented reality (AR)-guided surgical navigation, preoperative organ models are superimposed onto the patient's intraoperative anatomy to visualize critical structures such as vessels and tumors. Accurate deformation modeling is essential to maintain the reliability of AR overlays by ensuring alignment between preoperative models and the dynamically changing anatomy. Although the finite element method (FEM) offers physically plausible modeling, its high computational cost limits intraoperative applicability. Moreover, existing algorithms often fail to handle large anatomical changes, such as those induced by pneumoperitoneum or ligament dissection, leading to inaccurate anatomical correspondences and compromised AR guidance. To address these challenges, we propose a data-driven biomechanics algorithm that preserves FEM-level accuracy while improving computational efficiency. In addition, we introduce a novel human-in-the-loop mechanism into the deformation modeling process. This enables surgeons to interactively provide prompts to correct anatomical misalignments, thereby incorporating clinical expertise and allowing the model to adapt dynamically to complex surgical scenarios. Experiments on a publicly available dataset demonstrate that our algorithm achieves a mean target registration error of 3.42 mm. Incorporating surgeon prompts through the interactive framework further reduces the error to 2.78 mm, surpassing state-of-the-art methods in volumetric accuracy. These results highlight the ability of our framework to deliver efficient and accurate deformation modeling while enhancing surgeon-algorithm collaboration, paving the way for safer and more reliable computer-assisted surgeries.
>
---
#### [new 028] Teaching Physical Awareness to LLMs through Sounds
- **分类: cs.SD; cs.AI; cs.MM; cs.RO**

- **简介: 该论文属于物理感知任务，旨在解决LLMs缺乏物理理解的问题。通过声音数据和物理模拟器，提升模型对物理现象的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.08524v1](http://arxiv.org/pdf/2506.08524v1)**

> **作者:** Weiguo Wang; Andy Nie; Wenrui Zhou; Yi Kai; Chengchen Hu
>
> **备注:** ICML 2025
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities in text and multimodal processing, yet they fundamentally lack physical awareness--understanding of real-world physical phenomena. In this work, we present ACORN, a framework that teaches LLMs physical awareness through sound, focusing on fundamental physical phenomena like the Doppler effect, multipath effect, and spatial relationships. To overcome data scarcity, ACORN introduce a physics-based simulator combining real-world sound sources with controlled physical channels to generate diverse training data. Using this simulator, we build AQA-PHY, a comprehensive Audio Question-Answer dataset, and propose an audio encoder that processes both magnitude and phase information. By connecting our audio encoder to state-of-the-art LLMs, we demonstrate reasonable results in both simulated and real-world tasks, such as line-of-sight detection, Doppler effect estimation, and Direction-of-Arrival estimation, paving the way for enabling LLMs to understand physical world.
>
---
#### [new 029] Help or Hindrance: Understanding the Impact of Robot Communication in Action Teams
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机协作任务，研究机器人在时间敏感环境中如何通过多模态通信影响团队工作负荷与感知。通过实验探讨了语音和视觉提示的有效性。**

- **链接: [http://arxiv.org/pdf/2506.08892v1](http://arxiv.org/pdf/2506.08892v1)**

> **作者:** Tauhid Tanjim; Jonathan St. George; Kevin Ching; Hee Rin Lee; Angelique Taylor
>
> **备注:** This is the author's original submitted version of the paper accepted to the 2025 IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). \c{opyright} 2025 IEEE. Personal use of this material is permitted. For any other use, please contact IEEE
>
> **摘要:** The human-robot interaction (HRI) field has recognized the importance of enabling robots to interact with teams. Human teams rely on effective communication for successful collaboration in time-sensitive environments. Robots can play a role in enhancing team coordination through real-time assistance. Despite significant progress in human-robot teaming research, there remains an essential gap in how robots can effectively communicate with action teams using multimodal interaction cues in time-sensitive environments. This study addresses this knowledge gap in an experimental in-lab study to investigate how multimodal robot communication in action teams affects workload and human perception of robots. We explore team collaboration in a medical training scenario where a robotic crash cart (RCC) provides verbal and non-verbal cues to help users remember to perform iterative tasks and search for supplies. Our findings show that verbal cues for object search tasks and visual cues for task reminders reduce team workload and increase perceived ease of use and perceived usefulness more effectively than a robot with no feedback. Our work contributes to multimodal interaction research in the HRI field, highlighting the need for more human-robot teaming research to understand best practices for integrating collaborative robots in time-sensitive environments such as in hospitals, search and rescue, and manufacturing applications.
>
---
#### [new 030] Hybrid Reasoning for Perception, Explanation, and Autonomous Action in Manufacturing
- **分类: cs.AI; cs.HC; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于工业控制任务，旨在解决AI在制造中泛化能力不足和缺乏解释性的问题。提出CIPHER框架，结合视觉、语言和行动，实现自主决策与透明解释。**

- **链接: [http://arxiv.org/pdf/2506.08462v1](http://arxiv.org/pdf/2506.08462v1)**

> **作者:** Christos Margadji; Sebastian W. Pattinson
>
> **摘要:** Industrial processes must be robust and adaptable, as environments and tasks are often unpredictable, while operational errors remain costly and difficult to detect. AI-based control systems offer a path forward, yet typically depend on supervised learning with extensive labelled datasets, which limits their ability to generalize across variable and data-scarce industrial settings. Foundation models could enable broader reasoning and knowledge integration, but rarely deliver the quantitative precision demanded by engineering applications. Here, we introduceControl and Interpretation of Production via Hybrid Expertise and Reasoning (CIPHER): a vision-language-action (VLA) model framework aiming to replicate human-like reasoning for industrial control, instantiated in a commercial-grade 3D printer. It integrates a process expert, a regression model enabling quantitative characterization of system states required for engineering tasks. CIPHER also incorporates retrieval-augmented generation to access external expert knowledge and support physics-informed, chain-of-thought reasoning. This hybrid architecture exhibits strong generalization to out-of-distribution tasks. It interprets visual or textual inputs from process monitoring, explains its decisions, and autonomously generates precise machine instructions, without requiring explicit annotations. CIPHER thus lays the foundations for autonomous systems that act with precision, reason with context, and communicate decisions transparently, supporting safe and trusted deployment in industrial settings.
>
---
#### [new 031] DEKC: Data-Enable Control for Tethered Space Robot Deployment in the Presence of Uncertainty via Koopman Operator Theory
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于空间机器人部署任务，旨在解决未知不确定性的控制问题。提出DEKC框架，利用Koopman理论构建代理模型，提升控制精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.08319v1](http://arxiv.org/pdf/2506.08319v1)**

> **作者:** Ao Jin; Qinyi Wang; Sijie Wen; Ya Liu; Ganghui Shen; Panfeng Huang; Fan Zhang
>
> **备注:** 12 pages
>
> **摘要:** This work focuses the deployment of tethered space robot in the presence of unknown uncertainty. A data-enable framework called DEKC which contains offline training part and online execution part is proposed to deploy tethered space robot in the presence of uncertainty. The main idea of this work is modeling the unknown uncertainty as a dynamical system, which enables high accuracy and convergence of capturing uncertainty. The core part of proposed framework is a proxy model of uncertainty, which is derived from data-driven Koopman theory and is separated with controller design. In the offline stage, the lifting functions associated with Koopman operator are parameterized with deep neural networks. Then by solving an optimization problem, the lifting functions are learned from sampling data. In the online execution stage, the proxy model cooperates the learned lifting functions obtained in the offline phase to capture the unknown uncertainty. Then the output of proxy model is compensated to the baseline controller such that the effect of uncertainty can be attenuated or even eliminated. Furthermore, considering some scenarios in which the performance of proxy model may weaken, a receding-horizon scheme is proposed to update the proxy model online. Finally, the extensive numerical simulations demonstrate the effectiveness of our proposed framework. The implementation of proposed DEKC framework is publicly available at https://github.com/NPU-RCIR/DEKC.git.
>
---
#### [new 032] Efficient Learning of Vehicle Controller Parameters via Multi-Fidelity Bayesian Optimization: From Simulation to Experiment
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于车辆控制参数优化任务，旨在解决传统调参成本高、耗时的问题。通过多保真度贝叶斯优化方法，结合仿真与少量实验数据，提升调参效率。**

- **链接: [http://arxiv.org/pdf/2506.08719v1](http://arxiv.org/pdf/2506.08719v1)**

> **作者:** Yongpeng Zhao; Maik Pfefferkorn; Maximilian Templer; Rolf Findeisen
>
> **备注:** 8 pages, 8 figures, accepted for IEEE IV 2025
>
> **摘要:** Parameter tuning for vehicle controllers remains a costly and time-intensive challenge in automotive development. Traditional approaches rely on extensive real-world testing, making the process inefficient. We propose a multi-fidelity Bayesian optimization approach that efficiently learns optimal controller parameters by leveraging both low-fidelity simulation data and a very limited number of real-world experiments. Our approach significantly reduces the need for manual tuning and expensive field testing while maintaining the standard two-stage development workflow used in industry. The core contribution is the integration of an auto-regressive multi-fidelity Gaussian process model into Bayesian optimization, enabling knowledge transfer between different fidelity levels without requiring additional low-fidelity evaluations during real-world testing. We validate our approach through both simulation studies and realworld experiments. The results demonstrate that our method achieves high-quality controller performance with only very few real-world experiments, highlighting its potential as a practical and scalable solution for intelligent vehicle control tuning in industrial applications.
>
---
#### [new 033] ReCogDrive: A Reinforced Cognitive Framework for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决罕见场景下端到端系统性能下降的问题。通过融合视觉语言模型与扩散规划器，提升驾驶安全性与合理性。**

- **链接: [http://arxiv.org/pdf/2506.08052v1](http://arxiv.org/pdf/2506.08052v1)**

> **作者:** Yongkang Li; Kaixin Xiong; Xiangyu Guo; Fang Li; Sixu Yan; Gangwei Xu; Lijun Zhou; Long Chen; Haiyang Sun; Bing Wang; Guang Chen; Hangjun Ye; Wenyu Liu; Xinggang Wang
>
> **摘要:** Although end-to-end autonomous driving has made remarkable progress, its performance degrades significantly in rare and long-tail scenarios. Recent approaches attempt to address this challenge by leveraging the rich world knowledge of Vision-Language Models (VLMs), but these methods suffer from several limitations: (1) a significant domain gap between the pre-training data of VLMs and real-world driving data, (2) a dimensionality mismatch between the discrete language space and the continuous action space, and (3) imitation learning tends to capture the average behavior present in the dataset, which may be suboptimal even dangerous. In this paper, we propose ReCogDrive, an autonomous driving system that integrates VLMs with diffusion planner, which adopts a three-stage paradigm for training. In the first stage, we use a large-scale driving question-answering datasets to train the VLMs, mitigating the domain discrepancy between generic content and real-world driving scenarios. In the second stage, we employ a diffusion-based planner to perform imitation learning, mapping representations from the latent language space to continuous driving actions. Finally, we fine-tune the diffusion planner using reinforcement learning with NAVSIM non-reactive simulator, enabling the model to generate safer, more human-like driving trajectories. We evaluate our approach on the planning-oriented NAVSIM benchmark, achieving a PDMS of 89.6 and setting a new state-of-the-art that surpasses the previous vision-only SOTA by 5.6 PDMS.
>
---
#### [new 034] Scaling Laws of Motion Forecasting and Planning -- A Technical Report
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究自动驾驶中运动预测与规划的模型缩放规律，探讨模型性能随计算资源变化的规律及最优参数和数据规模比例。**

- **链接: [http://arxiv.org/pdf/2506.08228v1](http://arxiv.org/pdf/2506.08228v1)**

> **作者:** Mustafa Baniodeh; Kratarth Goel; Scott Ettinger; Carlos Fuertes; Ari Seff; Tim Shen; Cole Gulino; Chenjie Yang; Ghassen Jerfel; Dokook Choe; Rui Wang; Vinutha Kallem; Sergio Casas; Rami Al-Rfou; Benjamin Sapp; Dragomir Anguelov
>
> **摘要:** We study the empirical scaling laws of a family of encoder-decoder autoregressive transformer models on the task of joint motion forecasting and planning in the autonomous driving domain. Using a 500 thousand hours driving dataset, we demonstrate that, similar to language modeling, model performance improves as a power-law function of the total compute budget, and we observe a strong correlation between model training loss and model evaluation metrics. Most interestingly, closed-loop metrics also improve with scaling, which has important implications for the suitability of open-loop metrics for model development and hill climbing. We also study the optimal scaling of the number of transformer parameters and the training data size for a training compute-optimal model. We find that as the training compute budget grows, optimal scaling requires increasing the model size 1.5x as fast as the dataset size. We also study inference-time compute scaling, where we observe that sampling and clustering the output of smaller models makes them competitive with larger models, up to a crossover point beyond which a larger models becomes more inference-compute efficient. Overall, our experimental results demonstrate that optimizing the training and inference-time scaling properties of motion forecasting and planning models is a key lever for improving their performance to address a wide variety of driving scenarios. Finally, we briefly study the utility of training on general logged driving data of other agents to improve the performance of the ego-agent, an important research area to address the scarcity of robotics data for large capacity models training.
>
---
#### [new 035] MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于离线强化学习任务，解决源域与目标域动态不匹配的问题。通过模型生成合成数据并结合策略优化，提升目标域的探索能力。**

- **链接: [http://arxiv.org/pdf/2506.08460v1](http://arxiv.org/pdf/2506.08460v1)**

> **作者:** Yihong Guo; Yu Yang; Pan Xu; Anqi Liu
>
> **摘要:** We study the off-dynamics offline reinforcement learning problem, where the goal is to learn a policy from offline datasets collected from source and target domains with mismatched transition. Existing off-dynamics offline RL methods typically either filter source transitions that resemble those of the target domain or apply reward augmentation to source data, both constrained by the limited transitions available from the target domain. As a result, the learned policy is unable to explore target domain beyond the offline datasets. We propose MOBODY, a Model-Based Off-Dynamics offline RL algorithm that addresses this limitation by enabling exploration of the target domain via learned dynamics. MOBODY generates new synthetic transitions in the target domain through model rollouts, which are used as data augmentation during offline policy learning. Unlike existing model-based methods that learn dynamics from a single domain, MOBODY tackles the challenge of mismatched dynamics by leveraging both source and target datasets. Directly merging these datasets can bias the learned model toward source dynamics. Instead, MOBODY learns target dynamics by discovering a shared latent representation of states and transitions across domains through representation learning. To stabilize training, MOBODY incorporates a behavior cloning loss that regularizes the policy. Specifically, we introduce a Q-weighted behavior cloning loss that regularizes the policy toward actions with high target-domain Q-values, rather than uniformly imitating all actions in the dataset. These Q-values are learned from an enhanced target dataset composed of offline target data, augmented source data, and rollout data from the learned target dynamics. We evaluate MOBODY on MuJoCo benchmarks and show that it significantly outperforms state-of-the-art baselines, with especially pronounced improvements in challenging scenarios.
>
---
#### [new 036] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶中的在线高精地图构建任务，旨在解决传感器感知范围有限的问题。通过利用广泛存在的标准地图和文本标注信息，提升远距离检测精度。**

- **链接: [http://arxiv.org/pdf/2506.08997v1](http://arxiv.org/pdf/2506.08997v1)**

> **作者:** Fabian Immel; Jan-Hendrik Pauls; Richard Fehler; Frank Bieder; Jonas Merkert; Christoph Stiller
>
> **摘要:** Autonomous vehicles rely on detailed and accurate environmental information to operate safely. High definition (HD) maps offer a promising solution, but their high maintenance cost poses a significant barrier to scalable deployment. This challenge is addressed by online HD map construction methods, which generate local HD maps from live sensor data. However, these methods are inherently limited by the short perception range of onboard sensors. To overcome this limitation and improve general performance, recent approaches have explored the use of standard definition (SD) maps as prior, which are significantly easier to maintain. We propose SDTagNet, the first online HD map construction method that fully utilizes the information of widely available SD maps, like OpenStreetMap, to enhance far range detection accuracy. Our approach introduces two key innovations. First, in contrast to previous work, we incorporate not only polyline SD map data with manually selected classes, but additional semantic information in the form of textual annotations. In this way, we enrich SD vector map tokens with NLP-derived features, eliminating the dependency on predefined specifications or exhaustive class taxonomies. Second, we introduce a point-level SD map encoder together with orthogonal element identifiers to uniformly integrate all types of map elements. Experiments on Argoverse 2 and nuScenes show that this boosts map perception performance by up to +5.9 mAP (+45%) w.r.t. map construction without priors and up to +3.2 mAP (+20%) w.r.t. previous approaches that already use SD map priors. Code is available at https://github.com/immel-f/SDTagNet
>
---
#### [new 037] Neural-Augmented Kelvinlet: Real-Time Soft Tissue Deformation with Multiple Graspers
- **分类: cs.GR; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于手术机器人领域，解决软组织实时变形模拟问题。通过结合神经网络与Kelvinlet模型，提升模拟的准确性与物理一致性。**

- **链接: [http://arxiv.org/pdf/2506.08043v1](http://arxiv.org/pdf/2506.08043v1)**

> **作者:** Ashkan Shahbazi; Kyvia Pereira; Jon S. Heiselman; Elaheh Akbari; Annie C. Benson; Sepehr Seifi; Xinyuan Liu; Garrison L. Johnston; Erwin Terpstra; Anne Draaisma; Jan-Jaap Severes; Jie Ying Wu; Nabil Simaan; Michael L. Miga; Soheil Kolouri
>
> **摘要:** Fast and accurate simulation of soft tissue deformation is a critical factor for surgical robotics and medical training. In this paper, we introduce a novel physics-informed neural simulator that approximates soft tissue deformations in a realistic and real-time manner. Our framework integrates Kelvinlet-based priors into neural simulators, making it the first approach to leverage Kelvinlets for residual learning and regularization in data-driven soft tissue modeling. By incorporating large-scale Finite Element Method (FEM) simulations of both linear and nonlinear soft tissue responses, our method improves neural network predictions across diverse architectures, enhancing accuracy and physical consistency while maintaining low latency for real-time performance. We demonstrate the effectiveness of our approach by performing accurate surgical maneuvers that simulate the use of standard laparoscopic tissue grasping tools with high fidelity. These results establish Kelvinlet-augmented learning as a powerful and efficient strategy for real-time, physics-aware soft tissue simulation in surgical applications.
>
---
#### [new 038] Confidence Boosts Trust-Based Resilience in Cooperative Multi-Robot Systems
- **分类: eess.SP; cs.MA; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多机器人系统安全任务，解决恶意机器人干扰问题。提出一种基于物理信道信任度的弹性协议，通过调整参数实现协调与效率的平衡。**

- **链接: [http://arxiv.org/pdf/2506.08807v1](http://arxiv.org/pdf/2506.08807v1)**

> **作者:** Luca Ballotta; Áron Vékássy; Stephanie Gil; Michal Yemini
>
> **备注:** This work has been submitted to IEEE for possible publication
>
> **摘要:** Wireless communication-based multi-robot systems open the door to cyberattacks that can disrupt safety and performance of collaborative robots. The physical channel supporting inter-robot communication offers an attractive opportunity to decouple the detection of malicious robots from task-relevant data exchange between legitimate robots. Yet, trustworthiness indications coming from physical channels are uncertain and must be handled with this in mind. In this paper, we propose a resilient protocol for multi-robot operation wherein a parameter {\lambda}t accounts for how confident a robot is about the legitimacy of nearby robots that the physical channel indicates. Analytical results prove that our protocol achieves resilient coordination with arbitrarily many malicious robots under mild assumptions. Tuning {\lambda}t allows a designer to trade between near-optimal inter-robot coordination and quick task execution; see Fig. 1. This is a fundamental performance tradeoff and must be carefully evaluated based on the task at hand. The effectiveness of our approach is numerically verified with experiments involving platoons of autonomous cars where some vehicles are maliciously spoofed.
>
---
#### [new 039] How to Provably Improve Return Conditioned Supervised Learning?
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习领域，解决RCSL方法性能受限的问题。提出Reinforced RCSL框架，通过引入最优回报概念提升性能。**

- **链接: [http://arxiv.org/pdf/2506.08463v1](http://arxiv.org/pdf/2506.08463v1)**

> **作者:** Zhishuai Liu; Yu Yang; Ruhan Wang; Pan Xu; Dongruo Zhou
>
> **备注:** 25 pages, 4 figures, 12 tables
>
> **摘要:** In sequential decision-making problems, Return-Conditioned Supervised Learning (RCSL) has gained increasing recognition for its simplicity and stability in modern decision-making tasks. Unlike traditional offline reinforcement learning (RL) algorithms, RCSL frames policy learning as a supervised learning problem by taking both the state and return as input. This approach eliminates the instability often associated with temporal difference (TD) learning in offline RL. However, RCSL has been criticized for lacking the stitching property, meaning its performance is inherently limited by the quality of the policy used to generate the offline dataset. To address this limitation, we propose a principled and simple framework called Reinforced RCSL. The key innovation of our framework is the introduction of a concept we call the in-distribution optimal return-to-go. This mechanism leverages our policy to identify the best achievable in-dataset future return based on the current state, avoiding the need for complex return augmentation techniques. Our theoretical analysis demonstrates that Reinforced RCSL can consistently outperform the standard RCSL approach. Empirical results further validate our claims, showing significant performance improvements across a range of benchmarks.
>
---
#### [new 040] Communicating Through Avatars in Industry 5.0: A Focus Group Study on Human-Robot Collaboration
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机协作研究，旨在解决工业5.0中工人社交互动减少的问题。通过焦点小组研究，探讨了虚拟形象在提升人机协作体验中的作用与改进方向。**

- **链接: [http://arxiv.org/pdf/2506.08805v1](http://arxiv.org/pdf/2506.08805v1)**

> **作者:** Stina Klein; Pooja Prajod; Katharina Weitz; Matteo Lavit Nicora; Dimitra Tsovaltzi; Elisabeth André
>
> **备注:** Accepted LBW at CHIWORK 2025
>
> **摘要:** The integration of collaborative robots (cobots) in industrial settings raises concerns about worker well-being, particularly due to reduced social interactions. Avatars - designed to facilitate worker interactions and engagement - are promising solutions to enhance the human-robot collaboration (HRC) experience. However, real-world perspectives on avatar-supported HRC remain unexplored. To address this gap, we conducted a focus group study with employees from a German manufacturing company that uses cobots. Before the discussion, participants engaged with a scripted, industry-like HRC demo in a lab setting. This qualitative approach provided valuable insights into the avatar's potential roles, improvements to its behavior, and practical considerations for deploying them in industrial workcells. Our findings also emphasize the importance of personalized communication and task assistance. Although our study's limitations restrict its generalizability, it serves as an initial step in recognizing the potential of adaptive, context-aware avatar interactions in real-world industrial environments.
>
---
#### [new 041] Rethinking Range-View LiDAR Segmentation in Adverse Weather
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于LiDAR分割任务，旨在提升恶劣天气下的分割性能。通过引入模块化结构，增强模型对天气干扰的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.08979v1](http://arxiv.org/pdf/2506.08979v1)**

> **作者:** Longyu Yang; Ping Hu; Lu Zhang; Jun Liu; Yap-Peng Tan; Heng Tao Shen; Xiaofeng Zhu
>
> **摘要:** LiDAR segmentation has emerged as an important task to enrich multimedia experiences and analysis. Range-view-based methods have gained popularity due to their high computational efficiency and compatibility with real-time deployment. However, their generalized performance under adverse weather conditions remains underexplored, limiting their reliability in real-world environments. In this work, we identify and analyze the unique challenges that affect the generalization of range-view LiDAR segmentation in severe weather. To address these challenges, we propose a modular and lightweight framework that enhances robustness without altering the core architecture of existing models. Our method reformulates the initial stem block of standard range-view networks into two branches to process geometric attributes and reflectance intensity separately. Specifically, a Geometric Abnormality Suppression (GAS) module reduces the influence of weather-induced spatial noise, and a Reflectance Distortion Calibration (RDC) module corrects reflectance distortions through memory-guided adaptive instance normalization. The processed features are then fused and passed to the original segmentation pipeline. Extensive experiments on different benchmarks and baseline models demonstrate that our approach significantly improves generalization to adverse weather with minimal inference overhead, offering a practical and effective solution for real-world LiDAR segmentation.
>
---
## 更新

#### [replaced 001] Enhancing Safety of Foundation Models for Visual Navigation through Collision Avoidance via Repulsive Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03834v2](http://arxiv.org/pdf/2506.03834v2)**

> **作者:** Joonkyung Kim; Joonyeol Sim; Woojun Kim; Katia Sycara; Changjoo Nam
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** We propose CARE (Collision Avoidance via Repulsive Estimation), a plug-and-play module that enhances the safety of vision-based navigation without requiring additional range sensors or fine-tuning of pretrained models. While recent foundation models using only RGB inputs have shown strong performance, they often fail to generalize in out-of-distribution (OOD) environments with unseen objects or variations in camera parameters (e.g., field of view, pose, or focal length). Without fine-tuning, these models may generate unsafe trajectories that lead to collisions, requiring costly data collection and retraining. CARE addresses this limitation by seamlessly integrating with any RGB-based navigation system that outputs local trajectories, dynamically adjusting them using repulsive force vectors derived from monocular depth maps. We evaluate CARE by combining it with state-of-the-art vision-based navigation models across multiple robot platforms. CARE consistently reduces collision rates (up to 100%) without sacrificing goal-reaching performance and improves collision-free travel distance by up to 10.7x in exploration tasks.
>
---
#### [replaced 002] Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.10550v2](http://arxiv.org/pdf/2502.10550v2)**

> **作者:** Egor Cherepanov; Nikita Kachaev; Alexey K. Kovalev; Aleksandr I. Panov
>
> **备注:** 42 pages, 2 figures
>
> **摘要:** Memory is crucial for enabling agents to tackle complex tasks with temporal and spatial dependencies. While many reinforcement learning (RL) algorithms incorporate memory, the field lacks a universal benchmark to assess an agent's memory capabilities across diverse scenarios. This gap is particularly evident in tabletop robotic manipulation, where memory is essential for solving tasks with partial observability and ensuring robust performance, yet no standardized benchmarks exist. To address this, we introduce MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark for memory RL, with three key contributions: (1) we propose a comprehensive classification framework for memory-intensive RL tasks, (2) we collect MIKASA-Base -- a unified benchmark that enables systematic evaluation of memory-enhanced agents across diverse scenarios, and (3) we develop MIKASA-Robo (pip install mikasa-robo-suite) -- a novel benchmark of 32 carefully designed memory-intensive tasks that assess memory capabilities in tabletop robotic manipulation. Our work introduces a unified framework to advance memory RL research, enabling more robust systems for real-world use. MIKASA is available at https://tinyurl.com/membenchrobots.
>
---
#### [replaced 003] GigaSLAM: Large-Scale Monocular SLAM with Hierarchical Gaussian Splats
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.08071v2](http://arxiv.org/pdf/2503.08071v2)**

> **作者:** Kai Deng; Yigong Zhang; Jian Yang; Jin Xie
>
> **摘要:** Tracking and mapping in large-scale, unbounded outdoor environments using only monocular RGB input presents substantial challenges for existing SLAM systems. Traditional Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) SLAM methods are typically limited to small, bounded indoor settings. To overcome these challenges, we introduce GigaSLAM, the first RGB NeRF / 3DGS-based SLAM framework for kilometer-scale outdoor environments, as demonstrated on the KITTI, KITTI 360, 4 Seasons and A2D2 datasets. Our approach employs a hierarchical sparse voxel map representation, where Gaussians are decoded by neural networks at multiple levels of detail. This design enables efficient, scalable mapping and high-fidelity viewpoint rendering across expansive, unbounded scenes. For front-end tracking, GigaSLAM utilizes a metric depth model combined with epipolar geometry and PnP algorithms to accurately estimate poses, while incorporating a Bag-of-Words-based loop closure mechanism to maintain robust alignment over long trajectories. Consequently, GigaSLAM delivers high-precision tracking and visually faithful rendering on urban outdoor benchmarks, establishing a robust SLAM solution for large-scale, long-term scenarios, and significantly extending the applicability of Gaussian Splatting SLAM systems to unbounded outdoor environments. GitHub: https://github.com/DengKaiCQ/GigaSLAM.
>
---
#### [replaced 004] Interior Point Differential Dynamic Programming, Redux
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.08278v2](http://arxiv.org/pdf/2504.08278v2)**

> **作者:** Ming Xu; Stephen Gould; Iman Shames
>
> **摘要:** We present IPDDP2, a structure-exploiting algorithm for solving discrete-time, finite-horizon optimal control problems (OCPs) with nonlinear constraints. Inequality constraints are handled using a primal-dual interior point formulation and step acceptance for equality constraints follows a line-search filter approach. The iterates of the algorithm are derived under the Differential Dynamic Programming (DDP) framework. A proof of local quadratic convergence of the IPDDP2 iterates is provided. Our numerical experiments evaluate IPDDP2 on over 500 OCPs derived from five different classes of robotic motion planning problems, three of which are contact-implicit trajectory optimisation problems. IPDDP2 demonstrates improvements in robustness against existing constrained DDP algorithms for contact-implicit planning, while being significantly faster than general-purpose solver IPOPT. We provide a full implementation of IPDDP2 in the Julia programming language.
>
---
#### [replaced 005] Task Reconstruction and Extrapolation for $π_0$ using Text Latent
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.03500v3](http://arxiv.org/pdf/2505.03500v3)**

> **作者:** Quanyi Li
>
> **摘要:** Vision-language-action models (VLAs) often achieve high performance on demonstrated tasks but struggle significantly when required to extrapolate, combining skills learned from different tasks in novel ways. For instance, VLAs might successfully put the cream cheese in the bowl and put the bowl on top of the cabinet, yet still fail to put the cream cheese on top of the cabinet. In this work, we demonstrate that behaviors from distinct tasks can be effectively recombined by manipulating the VLA's internal representations at inference time. Concretely, we identify the text latent by averaging the text tokens' hidden states across all demonstrated trajectories for a specific base task. For executing an extrapolated task, we can temporally interpolate the text latent of the two base tasks and add it back to the text hidden states, so sub-behaviors from the two tasks will be activated sequentially. We evaluate this approach using the newly created libero-ood benchmark, featuring 20 tasks extrapolated from standard LIBERO suites. The results on libero-ood show that all SOTA VLAs achieve < 15% success rate, while $\pi0$ with text latent interpolation reaches an 83% success rate. Further qualitative analysis reveals a tendency for VLAs to exhibit spatial overfitting, mapping object names to demonstrated locations rather than achieving genuine object and goal understanding. Additionally, we find that decoding the text latent yields human-unreadable prompts that can nevertheless instruct the VLA to achieve a 70% success rate on standard LIBERO suites, enabling private instruction or backdoor attacks.
>
---
#### [replaced 006] Digital Twin Synchronization: Bridging the Sim-RL Agent to a Real-Time Robotic Additive Manufacturing Control
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2501.18016v2](http://arxiv.org/pdf/2501.18016v2)**

> **作者:** Matsive Ali; Sandesh Giri; Sen Liu; Qin Yang
>
> **备注:** This paper had been accepted by the 2025 IEEE International Conference on Engineering Reliable Autonomous Systems (ERAS)
>
> **摘要:** With the rapid development of deep reinforcement learning technology, it gradually demonstrates excellent potential and is becoming the most promising solution in the robotics. However, in the smart manufacturing domain, there is still not too much research involved in dynamic adaptive control mechanisms optimizing complex processes. This research advances the integration of Soft Actor-Critic (SAC) with digital twins for industrial robotics applications, providing a framework for enhanced adaptive real-time control for smart additive manufacturing processing. The system architecture combines Unity's simulation environment with ROS2 for seamless digital twin synchronization, while leveraging transfer learning to efficiently adapt trained models across tasks. We demonstrate our methodology using a Viper X300s robot arm with the proposed hierarchical reward structure to address the common reinforcement learning challenges in two distinct control scenarios. The results show rapid policy convergence and robust task execution in both simulated and physical environments demonstrating the effectiveness of our approach.
>
---
#### [replaced 007] Predictability Awareness for Efficient and Robust Multi-Agent Coordination
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.06223v2](http://arxiv.org/pdf/2411.06223v2)**

> **作者:** Roman Chiva Gil; Daniel Jarne Ornia; Khaled A. Mustafa; Javier Alonso Mora
>
> **备注:** Videos and other additional materials can be found at https://romanchiva.github.io/PAProjectPage/
>
> **摘要:** To safely and efficiently solve motion planning problems in multi-agent settings, most approaches attempt to solve a joint optimization that explicitly accounts for the responses triggered in other agents. This often results in solutions with an exponential computational complexity, making these methods intractable for complex scenarios with many agents. While sequential predict-and-plan approaches are more scalable, they tend to perform poorly in highly interactive environments. This paper proposes a method to improve the interactive capabilities of sequential predict-and-plan methods in multi-agent navigation problems by introducing predictability as an optimization objective. We interpret predictability through the use of general prediction models, by allowing agents to predict themselves and estimate how they align with these external predictions. We formally introduce this behavior through the free-energy of the system, which reduces under appropriate bounds to the Kullback-Leibler divergence between plan and prediction, and use this as a penalty for unpredictable trajectories.The proposed interpretation of predictability allows agents to more robustly leverage prediction models, and fosters a soft social convention that accelerates agreement on coordination strategies without the need of explicit high level control or communication. We show how this predictability-aware planning leads to lower-cost trajectories and reduces planning effort in a set of multi-robot problems, including autonomous driving experiments with human driver data, where we show that the benefits of considering predictability apply even when only the ego-agent uses this strategy.
>
---
#### [replaced 008] Robust Perception-Based Navigation using PAC-NMPC with a Learned Value Function
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2309.13171v3](http://arxiv.org/pdf/2309.13171v3)**

> **作者:** Adam Polevoy; Mark Gonzales; Marin Kobilarov; Joseph Moore
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Nonlinear model predictive control (NMPC) is typically restricted to short, finite horizons to limit the computational burden of online optimization. As a result, global planning frameworks are frequently necessary to avoid local minima when using NMPC for navigation in complex environments. By contrast, reinforcement learning (RL) can generate policies that minimize the expected cost over an infinite-horizon and can often avoid local minima, even when operating only on current sensor measurements. However, these learned policies are usually unable to provide performance guarantees (e.g., on collision avoidance), especially when outside of the training distribution. In this paper, we augment Probably Approximately Correct NMPC (PAC-NMPC), a sampling-based stochastic NMPC algorithm capable of providing statistical guarantees of performance and safety, with an approximate perception-based value function trained via RL. We demonstrate in simulation that our algorithm can improve the long-term behavior of PAC-NMPC while outperforming other approaches with regards to safety for both planar car dynamics and more complex, high-dimensional fixed-wing aerial vehicle dynamics. We also demonstrate that, even when our value function is trained in simulation, our algorithm can successfully achieve statistically safe navigation on hardware using a 1/10th scale rally car in cluttered real-world environments using only current sensor information.
>
---
#### [replaced 009] Innate-Values-driven Reinforcement Learning based Cognitive Modeling
- **分类: cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.09160v2](http://arxiv.org/pdf/2411.09160v2)**

> **作者:** Qin Yang
>
> **备注:** The paper had been accepted by the 2025 IEEE Conference on Cognitive and Computational Aspects of Situation Management (CogSIMA). arXiv admin note: text overlap with arXiv:2401.05572
>
> **摘要:** Innate values describe agents' intrinsic motivations, which reflect their inherent interests and preferences for pursuing goals and drive them to develop diverse skills that satisfy their various needs. Traditional reinforcement learning (RL) is learning from interaction based on the feedback rewards of the environment. However, in real scenarios, the rewards are generated by agents' innate value systems, which differ vastly from individuals based on their needs and requirements. In other words, considering the AI agent as a self-organizing system, developing its awareness through balancing internal and external utilities based on its needs in different tasks is a crucial problem for individuals learning to support others and integrate community with safety and harmony in the long term. To address this gap, we propose a new RL model termed innate-values-driven RL (IVRL) based on combined motivations' models and expected utility theory to mimic its complex behaviors in the evolution through decision-making and learning. Then, we introduce two IVRL-based models: IV-DQN and IV-A2C. By comparing them with benchmark algorithms such as DQN, DDQN, A2C, and PPO in the Role-Playing Game (RPG) reinforcement learning test platform VIZDoom, we demonstrated that the IVRL-based models can help the agent rationally organize various needs, achieve better performance effectively.
>
---
#### [replaced 010] Speech to Reality: On-Demand Production using Natural Language, 3D Generative AI, and Discrete Robotic Assembly
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2409.18390v5](http://arxiv.org/pdf/2409.18390v5)**

> **作者:** Alexander Htet Kyaw; Se Hwan Jeon; Miana Smith; Neil Gershenfeld
>
> **备注:** This work has been submitted to the IEEE for possible publication. An updated version will replace this version
>
> **摘要:** We present a system that transforms speech into physical objects using 3D generative AI and discrete robotic assembly. By leveraging natural language input, the system makes design and manufacturing more accessible to individuals without expertise in 3D modeling or robotic programming. While current generative AI models can produce a wide range of 3D digital assets, AI-generated meshes are not directly suitable for robotic fabrication and do not account for fabrication constraints. To address this, we contribute a workflow that integrates natural language processing, 3D generative AI, and discrete robotic assembly. The system automatically analyzes and modifies AI-generated geometry to meet physical constraints, such as component count, overhangs, and connectivity, and produces a feasible robotic assembly sequence and toolpath. The results are demonstrated through the assembly of various objects, ranging from chairs to shelves, which are prompted via speech and realized within 5 minutes using a robotic arm.
>
---
#### [replaced 011] Decentralized Uncertainty-Aware Active Search with a Team of Aerial Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.08507v2](http://arxiv.org/pdf/2410.08507v2)**

> **作者:** Wennie Tabib; John Stecklein; Caleb McDowell; Kshitij Goel; Felix Jonathan; Abhishek Rathod; Meghan Kokoski; Edsel Burkholder; Brian Wallace; Luis Ernesto Navarro-Serment; Nikhil Angad Bakshi; Tejus Gupta; Norman Papernick; David Guttendorf; Erik E. Kahn; Jessica Kasemer; Jesse Holdaway; Jeff Schneider
>
> **备注:** accepted at ISER 2025
>
> **摘要:** Rapid search and rescue is critical to maximizing survival rates following natural disasters. However, these efforts are challenged by the need to search large disaster zones, lack of reliability in the communications infrastructure, and a priori unknown numbers of objects of interest (OOIs), such as injured survivors. Aerial robots are increasingly being deployed for search and rescue due to their high mobility, but there remains a gap in deploying multi-robot autonomous aerial systems for methodical search of large environments. Prior works have relied on preprogrammed paths from human operators or are evaluated only in simulation. We bridge these gaps in the state of the art by developing and demonstrating a decentralized active search system, which biases its trajectories to take additional views of uncertain OOIs. The methodology leverages stochasticity for rapid coverage in communication denied scenarios. When communications are available, robots share poses, goals, and OOI information to accelerate the rate of search. Detections from multiple images and vehicles are fused to provide a mean and covariance for each OOI location. Extensive simulations and hardware experiments in Bloomingdale, OH, are conducted to validate the approach. The results demonstrate the active search approach outperforms greedy coverage-based planning in communication-denied scenarios while maintaining comparable performance in communication-enabled scenarios. The results also demonstrate the ability to detect and localize all a priori unknown OOIs with a mean error of approximately 3m at flight altitudes between 50m-60m.
>
---
#### [replaced 012] Innate-Values-driven Reinforcement Learning based Cooperative Multi-Agent Cognitive Modeling
- **分类: cs.LG; cs.AI; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2401.05572v2](http://arxiv.org/pdf/2401.05572v2)**

> **作者:** Qin Yang
>
> **备注:** This paper had been accepted by the 2025 IEEE Conference on Cognitive and Computational Aspects of Situation Management (CogSIMA)
>
> **摘要:** In multi-agent systems (MAS), the dynamic interaction among multiple decision-makers is driven by their innate values, affecting the environment's state, and can cause specific behavioral patterns to emerge. On the other hand, innate values in cognitive modeling reflect individual interests and preferences for specific tasks and drive them to develop diverse skills and plans, satisfying their various needs and achieving common goals in cooperation. Therefore, building the awareness of AI agents to balance the group utilities and system costs and meet group members' needs in their cooperation is a crucial problem for individuals learning to support their community and even integrate into human society in the long term. However, the current MAS reinforcement learning domain lacks a general intrinsic model to describe agents' dynamic motivation for decision-making and learning from an individual needs perspective in their cooperation. To address the gap, this paper proposes a general MAS innate-values reinforcement learning (IVRL) architecture from the individual preferences angle. We tested the Multi-Agent IVRL Actor-Critic Model in different StarCraft Multi-Agent Challenge (SMAC) settings, which demonstrated its potential to organize the group's behaviours to achieve better performance.
>
---
#### [replaced 013] EVA: An Embodied World Model for Future Video Anticipation
- **分类: cs.CV; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.15461v2](http://arxiv.org/pdf/2410.15461v2)**

> **作者:** Xiaowei Chi; Chun-Kai Fan; Hengyuan Zhang; Xingqun Qi; Rongyu Zhang; Anthony Chen; Chi-min Chan; Wei Xue; Qifeng Liu; Shanghang Zhang; Yike Guo
>
> **摘要:** Video generation models have made significant progress in simulating future states, showcasing their potential as world simulators in embodied scenarios. However, existing models often lack robust understanding, limiting their ability to perform multi-step predictions or handle Out-of-Distribution (OOD) scenarios. To address this challenge, we propose the Reflection of Generation (RoG), a set of intermediate reasoning strategies designed to enhance video prediction. It leverages the complementary strengths of pre-trained vision-language and video generation models, enabling them to function as a world model in embodied scenarios. To support RoG, we introduce Embodied Video Anticipation Benchmark(EVA-Bench), a comprehensive benchmark that evaluates embodied world models across diverse tasks and scenarios, utilizing both in-domain and OOD datasets. Building on this foundation, we devise a world model, Embodied Video Anticipator (EVA), that follows a multistage training paradigm to generate high-fidelity video frames and apply an autoregressive strategy to enable adaptive generalization for longer video sequences. Extensive experiments demonstrate the efficacy of EVA in various downstream tasks like video generation and robotics, thereby paving the way for large-scale pre-trained models in real-world video prediction applications. The video demos are available at \hyperlink{https://sites.google.com/view/icml-eva}{https://sites.google.com/view/icml-eva}.
>
---
#### [replaced 014] LLM-Craft: Robotic Crafting of Elasto-Plastic Objects with Large Language Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2406.08648v3](http://arxiv.org/pdf/2406.08648v3)**

> **作者:** Alison Bartsch; Amir Barati Farimani
>
> **摘要:** When humans create sculptures, we are able to reason about how geometrically we need to alter the clay state to reach our target goal. We are not computing point-wise similarity metrics, or reasoning about low-level positioning of our tools, but instead determining the higher-level changes that need to be made. In this work, we propose LLM-Craft, a novel pipeline that leverages large language models (LLMs) to iteratively reason about and generate deformation-based crafting action sequences. We simplify and couple the state and action representations to further encourage shape-based reasoning. To the best of our knowledge, LLM-Craft is the first system successfully leveraging LLMs for complex deformable object interactions. Through our experiments, we demonstrate that with the LLM-Craft framework, LLMs are able to successfully create a set of simple letter shapes. We explore a variety of rollout strategies, and compare performances of LLM-Craft variants with and without an explicit goal shape images. For videos and prompting details, please visit our project website: https://sites.google.com/andrew.cmu.edu/llmcraft/home
>
---
#### [replaced 015] DemoSpeedup: Accelerating Visuomotor Policies via Entropy-Guided Demonstration Acceleration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.05064v2](http://arxiv.org/pdf/2506.05064v2)**

> **作者:** Lingxiao Guo; Zhengrong Xue; Zijing Xu; Huazhe Xu
>
> **摘要:** Imitation learning has shown great promise in robotic manipulation, but the policy's execution is often unsatisfactorily slow due to commonly tardy demonstrations collected by human operators. In this work, we present DemoSpeedup, a self-supervised method to accelerate visuomotor policy execution via entropy-guided demonstration acceleration. DemoSpeedup starts from training an arbitrary generative policy (e.g., ACT or Diffusion Policy) on normal-speed demonstrations, which serves as a per-frame action entropy estimator. The key insight is that frames with lower action entropy estimates call for more consistent policy behaviors, which often indicate the demands for higher-precision operations. In contrast, frames with higher entropy estimates correspond to more casual sections, and therefore can be more safely accelerated. Thus, we segment the original demonstrations according to the estimated entropy, and accelerate them by down-sampling at rates that increase with the entropy values. Trained with the speedup demonstrations, the resulting policies execute up to 3 times faster while maintaining the task completion performance. Interestingly, these policies could even achieve higher success rates than those trained with normal-speed demonstrations, due to the benefits of reduced decision-making horizons. Project Page: https://demospeedup.github.io/
>
---
#### [replaced 016] BiAssemble: Learning Collaborative Affordance for Bimanual Geometric Assembly
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06221v2](http://arxiv.org/pdf/2506.06221v2)**

> **作者:** Yan Shen; Ruihai Wu; Yubin Ke; Xinyuan Song; Zeyi Li; Xiaoqi Li; Hongwei Fan; Haoran Lu; Hao dong
>
> **备注:** ICML 2025
>
> **摘要:** Shape assembly, the process of combining parts into a complete whole, is a crucial robotic skill with broad real-world applications. Among various assembly tasks, geometric assembly--where broken parts are reassembled into their original form (e.g., reconstructing a shattered bowl)--is particularly challenging. This requires the robot to recognize geometric cues for grasping, assembly, and subsequent bimanual collaborative manipulation on varied fragments. In this paper, we exploit the geometric generalization of point-level affordance, learning affordance aware of bimanual collaboration in geometric assembly with long-horizon action sequences. To address the evaluation ambiguity caused by geometry diversity of broken parts, we introduce a real-world benchmark featuring geometric variety and global reproducibility. Extensive experiments demonstrate the superiority of our approach over both previous affordance-based and imitation-based methods. Project page: https://sites.google.com/view/biassembly/.
>
---
#### [replaced 017] LMRPA: Large Language Model-Driven Efficient Robotic Process Automation for OCR
- **分类: cs.RO; cs.DL; cs.HC; cs.SE**

- **链接: [http://arxiv.org/pdf/2412.18063v2](http://arxiv.org/pdf/2412.18063v2)**

> **作者:** Osama Hosam Abdellaif; Abdelrahman Nader; Ali Hamdi
>
> **备注:** 10 pages , 1 figure , 1 algorithm
>
> **摘要:** This paper introduces LMRPA, a novel Large Model-Driven Robotic Process Automation (RPA) model designed to greatly improve the efficiency and speed of Optical Character Recognition (OCR) tasks. Traditional RPA platforms often suffer from performance bottlenecks when handling high-volume repetitive processes like OCR, leading to a less efficient and more time-consuming process. LMRPA allows the integration of Large Language Models (LLMs) to improve the accuracy and readability of extracted text, overcoming the challenges posed by ambiguous characters and complex text structures.Extensive benchmarks were conducted comparing LMRPA to leading RPA platforms, including UiPath and Automation Anywhere, using OCR engines like Tesseract and DocTR. The results are that LMRPA achieves superior performance, cutting the processing times by up to 52\%. For instance, in Batch 2 of the Tesseract OCR task, LMRPA completed the process in 9.8 seconds, where UiPath finished in 18.1 seconds and Automation Anywhere finished in 18.7 seconds. Similar improvements were observed with DocTR, where LMRPA outperformed other automation tools conducting the same process by completing tasks in 12.7 seconds, while competitors took over 20 seconds to do the same. These findings highlight the potential of LMRPA to revolutionize OCR-driven automation processes, offering a more efficient and effective alternative solution to the existing state-of-the-art RPA models.
>
---
#### [replaced 018] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06072v2](http://arxiv.org/pdf/2506.06072v2)**

> **作者:** Hongyi Zhou; Weiran Liao; Xi Huang; Yucheng Tang; Fabian Otto; Xiaogang Jia; Xinkai Jiang; Simon Hilber; Ge Li; Qian Wang; Ömer Erdinç Yağmurlu; Nils Blank; Moritz Reuss; Rudolf Lioutikov
>
> **摘要:** We present the B-spline Encoded Action Sequence Tokenizer (BEAST), a novel action tokenizer that encodes action sequences into compact discrete or continuous tokens using B-splines. In contrast to existing action tokenizers based on vector quantization or byte pair encoding, BEAST requires no separate tokenizer training and consistently produces tokens of uniform length, enabling fast action sequence generation via parallel decoding. Leveraging our B-spline formulation, BEAST inherently ensures generating smooth trajectories without discontinuities between adjacent segments. We extensively evaluate BEAST by integrating it with three distinct model architectures: a Variational Autoencoder (VAE) with continuous tokens, a decoder-only Transformer with discrete tokens, and Florence-2, a pretrained Vision-Language Model with an encoder-decoder architecture, demonstrating BEAST's compatibility and scalability with large pretrained models. We evaluate BEAST across three established benchmarks consisting of 166 simulated tasks and on three distinct robot settings with a total of 8 real-world tasks. Experimental results demonstrate that BEAST (i) significantly reduces both training and inference computational costs, and (ii) consistently generates smooth, high-frequency control signals suitable for continuous control tasks while (iii) reliably achieves competitive task success rates compared to state-of-the-art methods.
>
---
#### [replaced 019] Atmospheric Density-Compensating Model Predictive Control for Targeted Reentry of Drag-Modulated Spacecraft
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2407.18762v2](http://arxiv.org/pdf/2407.18762v2)**

> **作者:** Alex D. Hayes; Ryan J. Caverly
>
> **备注:** Accepted for publication in the Journal of Guidance, Control, and Dynamics
>
> **摘要:** This paper presents an estimation and control framework that enables the targeted reentry of a drag-modulated spacecraft in the presence of atmospheric density uncertainty. In particular, an extended Kalman filter (EKF) is used to estimate the in-flight density errors relative to the atmospheric density used to generate the nominal guidance trajectory. This information is leveraged within a model predictive control (MPC) strategy to improve tracking performance, reduce control effort, and increase robustness to actuator saturation compared to the state-of-the-art approach. The estimation and control framework is tested in a Monte Carlo simulation campaign with historical space weather data. These simulation efforts demonstrate that the proposed framework is able to stay within 100 km of the guidance trajectory at all points in time for 98.4% of cases. The remaining 1.6% of cases were pushed away from the guidance by large density errors, many due to significant solar storms and flares, that could not physically be compensated for by the drag control device. For the successful cases, the proposed framework was able to guide the spacecraft to the desired location at the entry interface altitude with a mean error of 12.1 km and 99.7% of cases below 100 km.
>
---
#### [replaced 020] TRAVEL: Training-Free Retrieval and Alignment for Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.07306v2](http://arxiv.org/pdf/2502.07306v2)**

> **作者:** Navid Rajabi; Jana Kosecka
>
> **备注:** Accepted to CVPR 2025 Workshop - Foundation Models Meet Embodied Agents
>
> **摘要:** In this work, we propose a modular approach for the Vision-Language Navigation (VLN) task by decomposing the problem into four sub-modules that use state-of-the-art Large Language Models (LLMs) and Vision-Language Models (VLMs) in a zero-shot setting. Given navigation instruction in natural language, we first prompt LLM to extract the landmarks and the order in which they are visited. Assuming the known model of the environment, we retrieve the top-k locations of the last landmark and generate $k$ path hypotheses from the starting location to the last landmark using the shortest path algorithm on the topological map of the environment. Each path hypothesis is represented by a sequence of panoramas. We then use dynamic programming to compute the alignment score between the sequence of panoramas and the sequence of landmark names, which match scores obtained from VLM. Finally, we compute the nDTW metric between the hypothesis that yields the highest alignment score to evaluate the path fidelity. We demonstrate superior performance compared to other approaches that use joint semantic maps like VLMaps on the complex R2R-Habitat instruction dataset and quantify in detail the effect of visual grounding on navigation performance.
>
---
#### [replaced 021] EKF-Based Radar-Inertial Odometry with Online Temporal Calibration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.00661v2](http://arxiv.org/pdf/2502.00661v2)**

> **作者:** Changseung Kim; Geunsik Bae; Woojae Shin; Sen Wang; Hyondong Oh
>
> **备注:** 8 pages, 6 figures, 4 tables
>
> **摘要:** Accurate time synchronization between heterogeneous sensors is crucial for ensuring robust state estimation in multi-sensor fusion systems. Sensor delays often cause discrepancies between the actual time when the event was captured and the time of sensor measurement, leading to temporal misalignment (time offset) between sensor measurement streams. In this paper, we propose an extended Kalman filter (EKF)-based radar-inertial odometry (RIO) framework that estimates the time offset online. The radar ego-velocity measurement model, derived from a single radar scan, is formulated to incorporate the time offset into the update. By leveraging temporal calibration, the proposed RIO enables accurate propagation and measurement updates based on a common time stream. Experiments on both simulated and real-world datasets demonstrate the accurate time offset estimation of the proposed method and its impact on RIO performance, validating the importance of sensor time synchronization. Our implementation of the EKF-RIO with online temporal calibration is available at https://github.com/spearwin/EKF-RIO-TC.
>
---
#### [replaced 022] Robot Pouring: Identifying Causes of Spillage and Selecting Alternative Action Parameters Using Probabilistic Actual Causation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.09395v3](http://arxiv.org/pdf/2502.09395v3)**

> **作者:** Jaime Maldonado; Jonas Krumme; Christoph Zetzsche; Vanessa Didelez; Kerstin Schill
>
> **备注:** 20 pages, 13 figures
>
> **摘要:** In everyday life, we perform tasks (e.g., cooking or cleaning) that involve a large variety of objects and goals. When confronted with an unexpected or unwanted outcome, we take corrective actions and try again until achieving the desired result. The reasoning performed to identify a cause of the observed outcome and to select an appropriate corrective action is a crucial aspect of human reasoning for successful task execution. Central to this reasoning is the assumption that a factor is responsible for producing the observed outcome. In this paper, we investigate the use of probabilistic actual causation to determine whether a factor is the cause of an observed undesired outcome. Furthermore, we show how the actual causation probabilities can be used to find alternative actions to change the outcome. We apply the probabilistic actual causation analysis to a robot pouring task. When spillage occurs, the analysis indicates whether a task parameter is the cause and how it should be changed to avoid spillage. The analysis requires a causal graph of the task and the corresponding conditional probability distributions. To fulfill these requirements, we perform a complete causal modeling procedure (i.e., task analysis, definition of variables, determination of the causal graph structure, and estimation of conditional probability distributions) using data from a realistic simulation of the robot pouring task, covering a large combinatorial space of task parameters. Based on the results, we discuss the implications of the variables' representation and how the alternative actions suggested by the actual causation analysis would compare to the alternative solutions proposed by a human observer. The practical use of the analysis of probabilistic actual causation to select alternative action parameters is demonstrated.
>
---
#### [replaced 023] GRAM: Generalization in Deep RL with a Robust Adaptation Module
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2412.04323v2](http://arxiv.org/pdf/2412.04323v2)**

> **作者:** James Queeney; Xiaoyi Cai; Alexander Schperberg; Radu Corcodel; Mouhacine Benosman; Jonathan P. How
>
> **摘要:** The reliable deployment of deep reinforcement learning in real-world settings requires the ability to generalize across a variety of conditions, including both in-distribution scenarios seen during training as well as novel out-of-distribution scenarios. In this work, we present a framework for dynamics generalization in deep reinforcement learning that unifies these two distinct types of generalization within a single architecture. We introduce a robust adaptation module that provides a mechanism for identifying and reacting to both in-distribution and out-of-distribution environment dynamics, along with a joint training pipeline that combines the goals of in-distribution adaptation and out-of-distribution robustness. Our algorithm GRAM achieves strong generalization performance across in-distribution and out-of-distribution scenarios upon deployment, which we demonstrate through extensive simulation and hardware locomotion experiments on a quadruped robot.
>
---
#### [replaced 024] Edge Computing based Human-Robot Cognitive Fusion: A Medical Case Study in the Autism Spectrum Disorder Therapy
- **分类: cs.RO; cs.AI; cs.DC; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2401.00776v2](http://arxiv.org/pdf/2401.00776v2)**

> **作者:** Qin Yang
>
> **备注:** This paper was accepted by the 2025 IEEE Conference on Cognitive and Computational Aspects of Situation Management (CogSIMA)
>
> **摘要:** In recent years, edge computing has served as a paradigm that enables many future technologies like AI, Robotics, IoT, and high-speed wireless sensor networks (like 5G) by connecting cloud computing facilities and services to the end users. Especially in medical and healthcare applications, it provides remote patient monitoring and increases voluminous multimedia. From the robotics angle, robot-assisted therapy (RAT) is an active-assistive robotic technology in rehabilitation robotics, attracting researchers to study and benefit people with disability like autism spectrum disorder (ASD) children. However, the main challenge of RAT is that the model capable of detecting the affective states of ASD people exists and can recall individual preferences. Moreover, involving expert diagnosis and recommendations to guide robots in updating the therapy approach to adapt to different statuses and scenarios is a crucial part of the ASD therapy process. This paper proposes the architecture of edge cognitive computing by combining human experts and assisted robots collaborating in the same framework to achieve a seamless remote diagnosis, round-the-clock symptom monitoring, emergency warning, therapy alteration, and advanced assistance.
>
---
#### [replaced 025] Evolutionary Policy Optimization
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.19037v2](http://arxiv.org/pdf/2503.19037v2)**

> **作者:** Jianren Wang; Yifan Su; Abhinav Gupta; Deepak Pathak
>
> **备注:** Website at https://yifansu1301.github.io/EPO/
>
> **摘要:** On-policy reinforcement learning (RL) algorithms are widely used for their strong asymptotic performance and training stability, but they struggle to scale with larger batch sizes, as additional parallel environments yield redundant data due to limited policy-induced diversity. In contrast, Evolutionary Algorithms (EAs) scale naturally and encourage exploration via randomized population-based search, but are often sample-inefficient. We propose Evolutionary Policy Optimization (EPO), a hybrid algorithm that combines the scalability and diversity of EAs with the performance and stability of policy gradients. EPO maintains a population of agents conditioned on latent variables, shares actor-critic network parameters for coherence and memory efficiency, and aggregates diverse experiences into a master agent. Across tasks in dexterous manipulation, legged locomotion, and classic control, EPO outperforms state-of-the-art baselines in sample efficiency, asymptotic performance, and scalability.
>
---
#### [replaced 026] From Pixels to Predicates: Learning Symbolic World Models via Pretrained Vision-Language Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00296v3](http://arxiv.org/pdf/2501.00296v3)**

> **作者:** Ashay Athalye; Nishanth Kumar; Tom Silver; Yichao Liang; Jiuguang Wang; Tomás Lozano-Pérez; Leslie Pack Kaelbling
>
> **摘要:** Our aim is to learn to solve long-horizon decision-making problems in complex robotics domains given low-level skills and a handful of short-horizon demonstrations containing sequences of images. To this end, we focus on learning abstract symbolic world models that facilitate zero-shot generalization to novel goals via planning. A critical component of such models is the set of symbolic predicates that define properties of and relationships between objects. In this work, we leverage pretrained vision language models (VLMs) to propose a large set of visual predicates potentially relevant for decision-making, and to evaluate those predicates directly from camera images. At training time, we pass the proposed predicates and demonstrations into an optimization-based model-learning algorithm to obtain an abstract symbolic world model that is defined in terms of a compact subset of the proposed predicates. At test time, given a novel goal in a novel setting, we use the VLM to construct a symbolic description of the current world state, and then use a search-based planning algorithm to find a sequence of low-level skills that achieves the goal. We demonstrate empirically across experiments in both simulation and the real world that our method can generalize aggressively, applying its learned world model to solve problems with a wide variety of object types, arrangements, numbers of objects, and visual backgrounds, as well as novel goals and much longer horizons than those seen at training time.
>
---
#### [replaced 027] LMPOcc: 3D Semantic Occupancy Prediction Utilizing Long-Term Memory Prior from Historical Traversals
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.13596v2](http://arxiv.org/pdf/2504.13596v2)**

> **作者:** Shanshuai Yuan; Julong Wei; Muer Tie; Xiangyun Ren; Zhongxue Gan; Wenchao Ding
>
> **摘要:** Vision-based 3D semantic occupancy prediction is critical for autonomous driving, enabling unified modeling of static infrastructure and dynamic agents. In practice, autonomous vehicles may repeatedly traverse identical geographic locations under varying environmental conditions, such as weather fluctuations and illumination changes. Existing methods in 3D occupancy prediction predominantly integrate adjacent temporal contexts. However, these works neglect to leverage perceptual information, which is acquired from historical traversals of identical geographic locations. In this paper, we propose Longterm Memory Prior Occupancy (LMPOcc), the first 3D occupancy prediction methodology that exploits long-term memory priors derived from historical traversal perceptual outputs. We introduce a plug-and-play architecture that integrates long-term memory priors to enhance local perception while simultaneously constructing global occupancy representations. To adaptively aggregate prior features and current features, we develop an efficient lightweight Current-Prior Fusion module. Moreover, we propose a model-agnostic prior format to ensure compatibility across diverse occupancy prediction baselines. LMPOcc achieves state-of-the-art performance validated on the Occ3D-nuScenes benchmark, especially on static semantic categories. Additionally, experimental results demonstrate LMPOcc's ability to construct global occupancy through multi-vehicle crowdsourcing.
>
---
#### [replaced 028] StereoVAE: A lightweight stereo-matching system using embedded GPUs
- **分类: cs.CV; cs.AI; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2305.11566v3](http://arxiv.org/pdf/2305.11566v3)**

> **作者:** Qiong Chang; Xiang Li; Xin Xu; Xin Liu; Yun Li; Miyazaki Jun
>
> **备注:** Will revise part of the contents
>
> **摘要:** We present a lightweight system for stereo matching through embedded GPUs. It breaks the trade-off between accuracy and processing speed in stereo matching, enabling our embedded system to further improve the matching accuracy while ensuring real-time processing. The main idea of our method is to construct a tiny neural network based on variational auto-encoder (VAE) to upsample and refinement a small size of coarse disparity map, which is first generated by a traditional matching method. The proposed hybrid structure cannot only bring the advantage of traditional methods in terms of computational complexity, but also ensure the matching accuracy under the impact of neural network. Extensive experiments on the KITTI 2015 benchmark demonstrate that our tiny system exhibits high robustness in improving the accuracy of the coarse disparity maps generated by different algorithms, while also running in real-time on embedded GPUs.
>
---
#### [replaced 029] Towards Autonomous Reinforcement Learning for Real-World Robotic Manipulation with Large Language Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.04280v4](http://arxiv.org/pdf/2503.04280v4)**

> **作者:** Niccolò Turcato; Matteo Iovino; Aris Synodinos; Alberto Dalla Libera; Ruggero Carli; Pietro Falco
>
> **摘要:** Recent advancements in Large Language Models (LLMs) and Visual Language Models (VLMs) have significantly impacted robotics, enabling high-level semantic motion planning applications. Reinforcement Learning (RL), a complementary paradigm, enables agents to autonomously optimize complex behaviors through interaction and reward signals. However, designing effective reward functions for RL remains challenging, especially in real-world tasks where sparse rewards are insufficient and dense rewards require elaborate design. In this work, we propose Autonomous Reinforcement learning for Complex Human-Informed Environments (ARCHIE), an unsupervised pipeline leveraging GPT-4, a pre-trained LLM, to generate reward functions directly from natural language task descriptions. The rewards are used to train RL agents in simulated environments, where we formalize the reward generation process to enhance feasibility. Additionally, GPT-4 automates the coding of task success criteria, creating a fully automated, one-shot procedure for translating human-readable text into deployable robot skills. Our approach is validated through extensive simulated experiments on single-arm and bi-manual manipulation tasks using an ABB YuMi collaborative robot, highlighting its practicality and effectiveness. Tasks are demonstrated on the real robot setup.
>
---
#### [replaced 030] Active inference as a unified model of collision avoidance behavior in human drivers
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.02215v3](http://arxiv.org/pdf/2506.02215v3)**

> **作者:** Julian F. Schumann; Johan Engström; Leif Johnson; Matthew O'Kelly; Joao Messias; Jens Kober; Arkady Zgonnikov
>
> **摘要:** Collision avoidance -- involving a rapid threat detection and quick execution of the appropriate evasive maneuver -- is a critical aspect of driving. However, existing models of human collision avoidance behavior are fragmented, focusing on specific scenarios or only describing certain aspects of the avoidance behavior, such as response times. This paper addresses these gaps by proposing a novel computational cognitive model of human collision avoidance behavior based on active inference. Active inference provides a unified approach to modeling human behavior: the minimization of free energy. Building on prior active inference work, our model incorporates established cognitive mechanisms such as evidence accumulation to simulate human responses in two distinct collision avoidance scenarios: front-to-rear lead vehicle braking and lateral incursion by an oncoming vehicle. We demonstrate that our model explains a wide range of previous empirical findings on human collision avoidance behavior. Specifically, the model closely reproduces both aggregate results from meta-analyses previously reported in the literature and detailed, scenario-specific effects observed in a recent driving simulator study, including response timing, maneuver selection, and execution. Our results highlight the potential of active inference as a unified framework for understanding and modeling human behavior in complex real-life driving tasks.
>
---
#### [replaced 031] Adaptive path planning for efficient object search by UAVs in agricultural fields
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02473v2](http://arxiv.org/pdf/2504.02473v2)**

> **作者:** Rick van Essen; Eldert van Henten; Lammert Kooistra; Gert Kootstra
>
> **摘要:** This paper presents an adaptive path planner for object search in agricultural fields using UAVs. The path planner uses a high-altitude coverage flight path and plans additional low-altitude inspections when the detection network is uncertain. The path planner was evaluated in an offline simulation environment containing real-world images. We trained a YOLOv8 detection network to detect artificial plants placed in grass fields to showcase the potential of our path planner. We evaluated the effect of different detection certainty measures, optimized the path planning parameters, investigated the effects of localization errors, and different numbers of objects in the field. The YOLOv8 detection confidence worked best to differentiate between true and false positive detections and was therefore used in the adaptive planner. The optimal parameters of the path planner depended on the distribution of objects in the field. When the objects were uniformly distributed, more low-altitude inspections were needed compared to a non-uniform distribution of objects, resulting in a longer path length. The adaptive planner proved to be robust against localization uncertainty. When increasing the number of objects, the flight path length increased, especially when the objects were uniformly distributed. When the objects were non-uniformly distributed, the adaptive path planner yielded a shorter path than a low-altitude coverage path, even with a high number of objects. Overall, the presented adaptive path planner allowed finding non-uniformly distributed objects in a field faster than a coverage path planner and resulted in a compatible detection accuracy. The path planner is made available at https://github.com/wur-abe/uav_adaptive_planner.
>
---
#### [replaced 032] When Uncertainty Leads to Unsafety: Empirical Insights into the Role of Uncertainty in Unmanned Aerial Vehicle Safety
- **分类: cs.SE; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.08908v2](http://arxiv.org/pdf/2501.08908v2)**

> **作者:** Sajad Khatiri; Fatemeh Mohammadi Amin; Sebastiano Panichella; Paolo Tonella
>
> **备注:** 39 pages
>
> **摘要:** Despite the recent developments in obstacle avoidance and other safety features, autonomous Unmanned Aerial Vehicles (UAVs) continue to face safety challenges. No previous work investigated the relationship between the behavioral uncertainty of a UAV, characterized in this work by inconsistent or erratic control signal patterns, and the unsafety of its flight. By quantifying uncertainty, it is possible to develop a predictor for unsafety, which acts as a flight supervisor. We conducted a large-scale empirical investigation of safety violations using PX4-Autopilot, an open-source UAV software platform. Our dataset of over 5,000 simulated flights, created to challenge obstacle avoidance, allowed us to explore the relation between uncertain UAV decisions and safety violations: up to 89% of unsafe UAV states exhibit significant decision uncertainty, and up to 74% of uncertain decisions lead to unsafe states. Based on these findings, we implemented Superialist (Supervising Autonomous Aerial Vehicles), a runtime uncertainty detector based on autoencoders, the state-of-the-art technology for anomaly detection. Superialist achieved high performance in detecting uncertain behaviors with up to 96% precision and 93% recall. Despite the observed performance degradation when using the same approach for predicting unsafety (up to 74% precision and 87% recall), Superialist enabled early prediction of unsafe states up to 50 seconds in advance.
>
---
