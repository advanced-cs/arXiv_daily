# 机器人 cs.RO

- **最新发布 30 篇**

- **更新 16 篇**

## 最新发布

#### [new 001] Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning
- **分类: cs.RO; cs.AI**

- **简介: 论文提出Isaac Lab，一个GPU加速的多模态机器人学习仿真框架，解决传统仿真平台难以支持大规模、多模态强化与模仿学习的问题，整合高保真物理、传感、域随机化等模块，统一训练流程，并支持未来可微分物理引擎集成。**

- **链接: [http://arxiv.org/pdf/2511.04831v1](http://arxiv.org/pdf/2511.04831v1)**

> **作者:** NVIDIA; :; Mayank Mittal; Pascal Roth; James Tigue; Antoine Richard; Octi Zhang; Peter Du; Antonio Serrano-Muñoz; Xinjie Yao; René Zurbrügg; Nikita Rudin; Lukasz Wawrzyniak; Milad Rakhsha; Alain Denzler; Eric Heiden; Ales Borovicka; Ossama Ahmed; Iretiayo Akinola; Abrar Anwar; Mark T. Carlson; Ji Yuan Feng; Animesh Garg; Renato Gasoto; Lionel Gulich; Yijie Guo; M. Gussert; Alex Hansen; Mihir Kulkarni; Chenran Li; Wei Liu; Viktor Makoviychuk; Grzegorz Malczyk; Hammad Mazhar; Masoud Moghani; Adithyavairavan Murali; Michael Noseworthy; Alexander Poddubny; Nathan Ratliff; Welf Rehberg; Clemens Schwarke; Ritvik Singh; James Latham Smith; Bingjie Tang; Ruchik Thaker; Matthew Trepte; Karl Van Wyk; Fangzhou Yu; Alex Millane; Vikram Ramasamy; Remo Steiner; Sangeeta Subramanian; Clemens Volk; CY Chen; Neel Jawale; Ashwin Varghese Kuruttukulam; Michael A. Lin; Ajay Mandlekar; Karsten Patzwaldt; John Welsh; Huihua Zhao; Fatima Anes; Jean-Francois Lafleche; Nicolas Moënne-Loccoz; Soowan Park; Rob Stepinski; Dirk Van Gelder; Chris Amevor; Jan Carius; Jumyung Chang; Anka He Chen; Pablo de Heras Ciechomski; Gilles Daviet; Mohammad Mohajerani; Julia von Muralt; Viktor Reutskyy; Michael Sauter; Simon Schirm; Eric L. Shi; Pierre Terdiman; Kenny Vilella; Tobias Widmer; Gordon Yeoman; Tiffany Chen; Sergey Grizan; Cathy Li; Lotus Li; Connor Smith; Rafael Wiltz; Kostas Alexis; Yan Chang; David Chu; Linxi "Jim" Fan; Farbod Farshidian; Ankur Handa; Spencer Huang; Marco Hutter; Yashraj Narang; Soha Pouya; Shiwei Sheng; Yuke Zhu; Miles Macklin; Adam Moravanszky; Philipp Reist; Yunrong Guo; David Hoeller; Gavriel State
>
> **备注:** Code and documentation are available here: https://github.com/isaac-sim/IsaacLab
>
> **摘要:** We present Isaac Lab, the natural successor to Isaac Gym, which extends the paradigm of GPU-native robotics simulation into the era of large-scale multi-modal learning. Isaac Lab combines high-fidelity GPU parallel physics, photorealistic rendering, and a modular, composable architecture for designing environments and training robot policies. Beyond physics and rendering, the framework integrates actuator models, multi-frequency sensor simulation, data collection pipelines, and domain randomization tools, unifying best practices for reinforcement and imitation learning at scale within a single extensible platform. We highlight its application to a diverse set of challenges, including whole-body control, cross-embodiment mobility, contact-rich and dexterous manipulation, and the integration of human demonstrations for skill acquisition. Finally, we discuss upcoming integration with the differentiable, GPU-accelerated Newton physics engine, which promises new opportunities for scalable, data-efficient, and gradient-based approaches to robot learning. We believe Isaac Lab's combination of advanced simulation capabilities, rich sensing, and data-center scale execution will help unlock the next generation of breakthroughs in robotics research.
>
---
#### [new 002] EveryDayVLA: A Vision-Language-Action Model for Affordable Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 论文提出EverydayVLA，一种低成本（<300美元）视觉-语言-动作模型，用于机器人操作。它联合输出离散与连续动作，通过自适应规划提升在复杂场景中的可靠性，在LIBERO和真实场景中均超越现有方法，推动家用与科研场景的普及。**

- **链接: [http://arxiv.org/pdf/2511.05397v1](http://arxiv.org/pdf/2511.05397v1)**

> **作者:** Samarth Chopra; Alex McMoil; Ben Carnovale; Evan Sokolson; Rajkumar Kubendran; Samuel Dickerson
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** While Vision-Language-Action (VLA) models map visual inputs and language instructions directly to robot actions, they often rely on costly hardware and struggle in novel or cluttered scenes. We introduce EverydayVLA, a 6-DOF manipulator that can be assembled for under $300, capable of modest payloads and workspace. A single unified model jointly outputs discrete and continuous actions, and our adaptive-horizon ensemble monitors motion uncertainty to trigger on-the-fly re-planning for safe, reliable operation. On LIBERO, EverydayVLA matches state-of-the-art success rates, and in real-world tests it outperforms prior methods by 49% in-distribution and 34.9% out-of-distribution. By combining a state-of-the-art VLA with cost-effective hardware, EverydayVLA democratizes access to a robotic foundation model and paves the way for economical use in homes and research labs alike. Experiment videos and details: https://everydayvla.github.io/
>
---
#### [new 003] Epically Powerful: An open-source software and mechatronics infrastructure for wearable robotic systems
- **分类: cs.RO**

- **简介: 该论文提出开源机器人基础设施Epically Powerful，解决穿戴式机器人开发门槛高的问题，通过统一软硬件框架、提供Python接口与模块化工具，支持快速构建基于QDD执行器的闭环控制系统。**

- **链接: [http://arxiv.org/pdf/2511.05033v1](http://arxiv.org/pdf/2511.05033v1)**

> **作者:** Jennifer K. Leestma; Siddharth R. Nathella; Christoph P. O. Nuesslein; Snehil Mathur; Gregory S. Sawicki; Aaron J. Young
>
> **备注:** 11 pages, 5 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Epically Powerful is an open-source robotics infrastructure that streamlines the underlying framework of wearable robotic systems - managing communication protocols, clocking, actuator commands, visualization, sensor data acquisition, data logging, and more - while also providing comprehensive guides for hardware selection, system assembly, and controller implementation. Epically Powerful contains a code base enabling simplified user implementation via Python that seamlessly interfaces with various commercial state-of-the-art quasi-direct drive (QDD) actuators, single-board computers, and common sensors, provides example controllers, and enables real-time visualization. To further support device development, the package also includes a recommended parts list and compatibility guide and detailed documentation on hardware and software implementation. The goal of Epically Powerful is to lower the barrier to developing and deploying custom wearable robotic systems without a pre-specified form factor, enabling researchers to go from raw hardware to modular, robust devices quickly and effectively. Though originally designed with wearable robotics in mind, Epically Powerful is broadly applicable to other robotic domains that utilize QDD actuators, single-board computers, and sensors for closed-loop control.
>
---
#### [new 004] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 论文提出TwinVLA，通过组合两个预训练单臂视觉-语言-动作模型，实现无需双臂数据的高效双臂操作，解决双臂任务数据稀缺问题，在真实与仿真环境中超越单模型方法，显著提升数据效率。**

- **链接: [http://arxiv.org/pdf/2511.05275v1](http://arxiv.org/pdf/2511.05275v1)**

> **作者:** Hokyun Im; Euijin Jeong; Jianlong Fu; Andrey Kolobov; Youngwoon Lee
>
> **备注:** Project webpage : https://jellyho.github.io/TwinVLA/
>
> **摘要:** Vision-language-action models (VLAs) trained on large-scale robotic datasets have demonstrated strong performance on manipulation tasks, including bimanual tasks. However, because most public datasets focus on single-arm demonstrations, adapting VLAs for bimanual tasks typically requires substantial additional bimanual data and fine-tuning. To address this challenge, we introduce TwinVLA, a modular framework that composes two copies of a pretrained single-arm VLA into a coordinated bimanual VLA. Unlike monolithic cross-embodiment models trained on mixtures of single-arm and bimanual data, TwinVLA improves both data efficiency and performance by composing pretrained single-arm policies. Across diverse bimanual tasks in real-world and simulation settings, TwinVLA outperforms a comparably-sized monolithic RDT-1B model without requiring any bimanual pretraining. Furthermore, it narrows the gap to state-of-the-art model, $\pi_0$ which rely on extensive proprietary bimanual data and compute cost. These results establish our modular composition approach as a data-efficient and scalable path toward high-performance bimanual manipulation, leveraging public single-arm data.
>
---
#### [new 005] ETHOS: A Robotic Encountered-Type Haptic Display for Social Interaction in Virtual Reality
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: ETHOS提出一种动态遇觉式力反馈系统，解决VR中社交触觉交互不自然的问题，通过机械臂与被动道具实时同步，实现握手、击掌等自然接触，兼具高精度与安全性。**

- **链接: [http://arxiv.org/pdf/2511.05379v1](http://arxiv.org/pdf/2511.05379v1)**

> **作者:** Eric Godden; Jacquie Groenewegen; Matthew K. X. J. Pan
>
> **备注:** 8 pages
>
> **摘要:** We present ETHOS (Encountered-Type Haptics for On-demand Social Interaction), a dynamic encountered-type haptic display (ETHD) that enables natural physical contact in virtual reality (VR) during social interactions such as handovers, fist bumps, and high-fives. The system integrates a torque-controlled robotic manipulator with interchangeable passive props (silicone hand replicas and a baton), marker-based physical-virtual registration via a ChArUco board, and a safety monitor that gates motion based on the user's head and hand pose. We introduce two control strategies: (i) a static mode that presents a stationary prop aligned with its virtual counterpart, consistent with prior ETHD baselines, and (ii) a dynamic mode that continuously updates prop position by exponentially blending an initial mid-point trajectory with real-time hand tracking, generating a unique contact point for each interaction. Bench tests show static colocation accuracy of 5.09 +/- 0.94 mm, while user interactions achieved temporal alignment with an average contact latency of 28.53 +/- 31.21 ms across all interaction and control conditions. These results demonstrate the feasibility of recreating socially meaningful haptics in VR. By incorporating essential safety and control mechanisms, ETHOS establishes a practical foundation for high-fidelity, dynamic interpersonal interactions in virtual environments.
>
---
#### [new 006] Procedimiento de auditoría de ciberseguridad para sistemas autónomos: metodología, amenazas y mitigaciones
- **分类: cs.RO; cs.CR**

- **简介: 本文提出一种面向自主系统的网络安全审计方法，基于分层框架、机器人威胁分类与缓解措施，解决其高复杂性带来的安全风险问题，并通过四类机器人平台验证有效性。**

- **链接: [http://arxiv.org/pdf/2511.05185v1](http://arxiv.org/pdf/2511.05185v1)**

> **作者:** Adrián Campazas-Vega; Claudia Álvarez-Aparicio; David Sobrín-Hidalgo; Laura Inyesto-Alonso; Francisco Javier Rodríguez-Lera; Vicente Matellán-Olivera; Ángel Manuel Guerrero-Higueras
>
> **备注:** 32 pages, in Spanish language, 7 tables, 12 Figures. White paper under the TESCAC project
>
> **摘要:** The deployment of autonomous systems has experienced remarkable growth in recent years, driven by their integration into sectors such as industry, medicine, logistics, and domestic environments. This expansion is accompanied by a series of security issues that entail significant risks due to the critical nature of autonomous systems, especially those operating in human-interaction environments. Furthermore, technological advancement and the high operational and architectural complexity of autonomous systems have resulted in an increased attack surface. This article presents a specific security auditing procedure for autonomous systems, based on a layer-structured methodology, a threat taxonomy adapted to the robotic context, and a set of concrete mitigation measures. The validity of the proposed approach is demonstrated through four practical case studies applied to representative robotic platforms: the Vision 60 military quadruped from Ghost Robotics, the A1 robot from Unitree Robotics, the UR3 collaborative arm from Universal Robots, and the Pepper social robot from Aldebaran Robotics.
>
---
#### [new 007] Bioinspired Soft Quadrotors Jointly Unlock Agility, Squeezability, and Collision Resilience
- **分类: cs.RO; J.2**

- **简介: 该论文提出柔性框架四旋翼FlexiQuad，解决传统刚性无人机在复杂环境中 agility、 squeezability 与碰撞耐受性难以兼顾的问题，通过仿生软结构实现高速机动、压缩通过窄缝及抗冲击，突破性能边界。**

- **链接: [http://arxiv.org/pdf/2511.05426v1](http://arxiv.org/pdf/2511.05426v1)**

> **作者:** Luca Girardi; Gabriel Maquignaz; Stefano Mintchev
>
> **备注:** 26 pages, 12 figures, 2 tables, 9 videos (not yet disclosed, awaiting peer review)
>
> **摘要:** Natural flyers use soft wings to seamlessly enable a wide range of flight behaviours, including agile manoeuvres, squeezing through narrow passageways, and withstanding collisions. In contrast, conventional quadrotor designs rely on rigid frames that support agile flight but inherently limit collision resilience and squeezability, thereby constraining flight capabilities in cluttered environments. Inspired by the anisotropic stiffness and distributed mass-energy structures observed in biological organisms, we introduce FlexiQuad, a soft-frame quadrotor design approach that limits this trade-off. We demonstrate a 405-gram FlexiQuad prototype, three orders of magnitude more compliant than conventional quadrotors, yet capable of acrobatic manoeuvres with peak speeds above 80 km/h and linear and angular accelerations exceeding 3 g and 300 rad/s$^2$, respectively. Analysis demonstrates it can replicate accelerations of rigid counterparts up to a thrust-to-weight ratio of 8. Simultaneously, FlexiQuad exhibits fourfold higher collision resilience, surviving frontal impacts at 5 m/s without damage and reducing destabilising forces in glancing collisions by a factor of 39. Its frame can fully compress, enabling flight through gaps as narrow as 70% of its nominal width. Our analysis identifies an optimal structural softness range, from 0.006 to 0.77 N/mm, comparable to that of natural flyers' wings, whereby agility, squeezability, and collision resilience are jointly achieved for FlexiQuad models from 20 to 3000 grams. FlexiQuad expands hovering drone capabilities in complex environments, enabling robust physical interactions without compromising flight performance.
>
---
#### [new 008] Unified Multimodal Diffusion Forcing for Forceful Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出Multimodal Diffusion Forcing（MDF），用于从多模态机器人轨迹中学习时序与跨模态依赖，解决传统模仿学习忽略感官-动作-奖励交互的问题。通过扩散模型重构掩码轨迹，提升力控操作的鲁棒性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.04812v1](http://arxiv.org/pdf/2511.04812v1)**

> **作者:** Zixuan Huang; Huaidian Hou; Dmitry Berenson
>
> **备注:** Project website: https://unified-df.github.io
>
> **摘要:** Given a dataset of expert trajectories, standard imitation learning approaches typically learn a direct mapping from observations (e.g., RGB images) to actions. However, such methods often overlook the rich interplay between different modalities, i.e., sensory inputs, actions, and rewards, which is crucial for modeling robot behavior and understanding task outcomes. In this work, we propose Multimodal Diffusion Forcing, a unified framework for learning from multimodal robot trajectories that extends beyond action generation. Rather than modeling a fixed distribution, MDF applies random partial masking and trains a diffusion model to reconstruct the trajectory. This training objective encourages the model to learn temporal and cross-modal dependencies, such as predicting the effects of actions on force signals or inferring states from partial observations. We evaluate MDF on contact-rich, forceful manipulation tasks in simulated and real-world environments. Our results show that MDF not only delivers versatile functionalities, but also achieves strong performance, and robustness under noisy observations. More visualizations can be found on our website https://unified-df.github.io
>
---
#### [new 009] A semi-analytical approach for computing the largest singularity-free spheres of a class of 6-6 Stewart-Gough platforms for specified orientation workspaces
- **分类: cs.RO**

- **简介: 该论文提出一种半解析方法，计算6-6Stewart-Gough平台在指定姿态工作空间内的最大无奇异性球体（SFS），通过采样姿态并取最小SFS，评估不同构型的运动性能，服务于机器人机构分析与设计。**

- **链接: [http://arxiv.org/pdf/2511.04992v1](http://arxiv.org/pdf/2511.04992v1)**

> **作者:** Bibekananda Patra; Sandipan Bandyopadhyay
>
> **摘要:** This article presents a method for computing the largest singularity-free sphere (SFS) of a 6-6 Stewart-Gough platform manipulator (SGPM) over a specified orientation workspace. For a fixed orientation of the moving platform, the SFS is computed analytically. This process is repeated over a set of samples generated within the orientation workspace, and the smallest among them is designated as the desired SFS for the given orientation workspace. Numerical experiments are performed on four distinct architectures of the SGPM to understand their relative performances w.r.t. SFS volumes over the same orientation workspace. This study demonstrates the potential utility of the proposed computational method both in analysis and design of SGPMs.
>
---
#### [new 010] ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated Multi-Arm Task and Motion Planning & Scheduling
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 论文提出ScheduleStream，首个基于采样的GPU加速框架，用于多臂机器人任务与运动规划调度（TAMPAS），解决传统TAMP无法并行多臂运动的问题，通过异步持久动作建模时序动态，实现高效通用调度规划。**

- **链接: [http://arxiv.org/pdf/2511.04758v1](http://arxiv.org/pdf/2511.04758v1)**

> **作者:** Caelan Garrett; Fabio Ramos
>
> **备注:** Project website: https://schedulestream.github.io
>
> **摘要:** Bimanual and humanoid robots are appealing because of their human-like ability to leverage multiple arms to efficiently complete tasks. However, controlling multiple arms at once is computationally challenging due to the growth in the hybrid discrete-continuous action space. Task and Motion Planning (TAMP) algorithms can efficiently plan in hybrid spaces but generally produce plans, where only one arm is moving at a time, rather than schedules that allow for parallel arm motion. In order to extend TAMP to produce schedules, we present ScheduleStream, the first general-purpose framework for planning & scheduling with sampling operations. ScheduleStream models temporal dynamics using hybrid durative actions, which can be started asynchronously and persist for a duration that's a function of their parameters. We propose domain-independent algorithms that solve ScheduleStream problems without any application-specific mechanisms. We apply ScheduleStream to Task and Motion Planning & Scheduling (TAMPAS), where we use GPU acceleration within samplers to expedite planning. We compare ScheduleStream algorithms to several ablations in simulation and find that they produce more efficient solutions. We demonstrate ScheduleStream on several real-world bimanual robot tasks at https://schedulestream.github.io.
>
---
#### [new 011] Force-Safe Environment Maps and Real-Time Detection for Soft Robot Manipulators
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对软体机器人在精细环境中的安全操作问题，提出将任务空间的力安全约束映射到构型空间，实现实时力安全检测，确保机器人接触时力不超过阈值，支持在复杂环境中安全规划。**

- **链接: [http://arxiv.org/pdf/2511.05307v1](http://arxiv.org/pdf/2511.05307v1)**

> **作者:** Akua K. Dickson; Juan C. Pacheco Garcia; Andrew P. Sabelhaus
>
> **摘要:** Soft robot manipulators have the potential for deployment in delicate environments to perform complex manipulation tasks. However, existing obstacle detection and avoidance methods do not consider limits on the forces that manipulators may exert upon contact with delicate obstacles. This work introduces a framework that maps force safety criteria from task space (i.e. positions along the robot's body) to configuration space (i.e. the robot's joint angles) and enables real-time force safety detection. We incorporate limits on allowable environmental contact forces for given task-space obstacles, and map them into configuration space (C-space) through the manipulator's forward kinematics. This formulation ensures that configurations classified as safe are provably below the maximum force thresholds, thereby allowing us to determine force-safe configurations of the soft robot manipulator in real-time. We validate our approach in simulation and hardware experiments on a two-segment pneumatic soft robot manipulator. Results demonstrate that the proposed method accurately detects force safety during interactions with deformable obstacles, thereby laying the foundation for real-time safe planning of soft manipulators in delicate, cluttered environments.
>
---
#### [new 012] Stable and Robust SLIP Model Control via Energy Conservation-Based Feedback Cancellation for Quadrupedal Applications
- **分类: cs.RO**

- **简介: 该论文提出一种基于能量守恒反馈抵消的SLIP模型控制方法，用于四足机器人稳定奔跑。通过仿真验证，该方法可实现鲁棒的弹跳步态，即使在10%传感器误差下仍保持稳定，解决了动态运动控制的稳定性与鲁棒性问题。**

- **链接: [http://arxiv.org/pdf/2511.05402v1](http://arxiv.org/pdf/2511.05402v1)**

> **作者:** Muhammad Saud Ul Hassan; Derek Vasquez; Hamza Asif; Christian Hubicki
>
> **摘要:** In this paper, we present an energy-conservation based control architecture for stable dynamic motion in quadruped robots. We model the robot as a Spring-loaded Inverted Pendulum (SLIP), a model well-suited to represent the bouncing motion characteristic of running gaits observed in various biological quadrupeds and bio-inspired robotic systems. The model permits leg-orientation control during flight and leg-length control during stance, a design choice inspired by natural quadruped behaviors and prevalent in robotic quadruped systems. Our control algorithm uses the reduced-order SLIP dynamics of the quadruped to track a stable parabolic spline during stance, which is calculated using the principle of energy conservation. Through simulations based on the design specifications of an actual quadruped robot, Ghost Robotics Minitaur, we demonstrate that our control algorithm generates stable bouncing gaits. Additionally, we illustrate the robustness of our controller by showcasing its ability to maintain stable bouncing even when faced with up to a 10% error in sensor measurements.
>
---
#### [new 013] TAPOM: Task-Space Topology-Guided Motion Planning for Manipulating Elongated Object in Cluttered Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出TAPOM方法，用于复杂环境中细长物体的避障操作。通过任务空间拓扑分析生成关键帧，引导低层规划器高效搜索可行路径，解决传统方法在窄缝场景中采样困难与局部极小问题。**

- **链接: [http://arxiv.org/pdf/2511.05052v1](http://arxiv.org/pdf/2511.05052v1)**

> **作者:** Zihao Li; Yiming Zhu; Zhe Zhong; Qinyuan Ren; Yijiang Huang
>
> **摘要:** Robotic manipulation in complex, constrained spaces is vital for widespread applications but challenging, particularly when navigating narrow passages with elongated objects. Existing planning methods often fail in these low-clearance scenarios due to the sampling difficulties or the local minima. This work proposes Topology-Aware Planning for Object Manipulation (TAPOM), which explicitly incorporates task-space topological analysis to enable efficient planning. TAPOM uses a high-level analysis to identify critical pathways and generate guiding keyframes, which are utilized in a low-level planner to find feasible configuration space trajectories. Experimental validation demonstrates significantly high success rates and improved efficiency over state-of-the-art methods on low-clearance manipulation tasks. This approach offers broad implications for enhancing manipulation capabilities of robots in complex real-world environments.
>
---
#### [new 014] Tunable Passivity Control for Centralized Multiport Networked Systems
- **分类: cs.RO**

- **简介: 该论文针对集中式多端口网络化系统的稳定性问题，提出一种无模型的可调集中最优被动控制（TCoPC），通过集中观测能量流并优化耗散分配，突破传统方法对节点被动性与最小相位的限制，提升系统稳定性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2511.05026v1](http://arxiv.org/pdf/2511.05026v1)**

> **作者:** Xingyuan Zhou; Peter Paik; S. Farokh Atashzar
>
> **摘要:** Centralized Multiport Networked Dynamic (CMND) systems have emerged as a key architecture with applications in several complex network systems, such as multilateral telerobotics and multi-agent control. These systems consist of a hub node/subsystem connecting with multiple remote nodes/subsystems via a networked architecture. One challenge for this system is stability, which can be affected by non-ideal network artifacts. Conventional passivity-based approaches can stabilize the system under specialized applications like small-scale networked systems. However, those conventional passive stabilizers have several restrictions, such as distributing compensation across subsystems in a decentralized manner, limiting flexibility, and, at the same time, relying on the restrictive assumptions of node passivity. This paper synthesizes a centralized optimal passivity-based stabilization framework for CMND systems. It consists of a centralized passivity observer monitoring overall energy flow and an optimal passivity controller that distributes the just-needed dissipation among various nodes, guaranteeing strict passivity and, thus, L2 stability. The proposed data-driven model-free approach, i.e., Tunable Centralized Optimal Passivity Control (TCoPC), optimizes total performance based on the prescribed dissipation distribution strategy while ensuring stability. The controller can put high dissipation loads on some sub-networks while relaxing the dissipation on other nodes. Simulation results demonstrate the proposed frameworks performance in a complex task under different time-varying delay scenarios while relaxing the remote nodes minimum phase and passivity assumption, enhancing the scalability and generalizability.
>
---
#### [new 015] ReGen: Generative Robot Simulation via Inverse Design
- **分类: cs.RO**

- **简介: ReGen提出一种基于逆向设计的生成式机器人仿真框架，利用大语言模型从行为描述自动生成复杂仿真场景，解决人工构建仿真效率低的问题，支持可控、多模态与反事实场景生成，提升机器人策略验证与泛化能力。**

- **链接: [http://arxiv.org/pdf/2511.04769v1](http://arxiv.org/pdf/2511.04769v1)**

> **作者:** Phat Nguyen; Tsun-Hsuan Wang; Zhang-Wei Hong; Erfan Aasi; Andrew Silva; Guy Rosman; Sertac Karaman; Daniela Rus
>
> **摘要:** Simulation plays a key role in scaling robot learning and validating policies, but constructing simulations remains a labor-intensive process. This paper introduces ReGen, a generative simulation framework that automates simulation design via inverse design. Given a robot's behavior -- such as a motion trajectory or an objective function -- and its textual description, ReGen infers plausible scenarios and environments that could have caused the behavior. ReGen leverages large language models to synthesize scenarios by expanding a directed graph that encodes cause-and-effect relationships, relevant entities, and their properties. This structured graph is then translated into a symbolic program, which configures and executes a robot simulation environment. Our framework supports (i) augmenting simulations based on ego-agent behaviors, (ii) controllable, counterfactual scenario generation, (iii) reasoning about agent cognition and mental states, and (iv) reasoning with distinct sensing modalities, such as braking due to faulty GPS signals. We demonstrate ReGen in autonomous driving and robot manipulation tasks, generating more diverse, complex simulated environments compared to existing simulations with high success rates, and enabling controllable generation for corner cases. This approach enhances the validation of robot policies and supports data or simulation augmentation, advancing scalable robot learning for improved generalization and robustness. We provide code and example videos at: https://regen-sim.github.io/
>
---
#### [new 016] MoE-DP: An MoE-Enhanced Diffusion Policy for Robust Long-Horizon Robotic Manipulation with Skill Decomposition and Failure Recovery
- **分类: cs.RO**

- **简介: 论文提出MoE-DP，用于长周期机器人操作任务，解决扩散策略抗干扰弱、可解释性差的问题。通过引入MoE层实现技能分解与失败恢复，提升鲁棒性与可解释性，支持无重训的子任务重排。**

- **链接: [http://arxiv.org/pdf/2511.05007v1](http://arxiv.org/pdf/2511.05007v1)**

> **作者:** Baiye Cheng; Tianhai Liang; Suning Huang; Maanping Shao; Feihong Zhang; Botian Xu; Zhengrong Xue; Huazhe Xu
>
> **摘要:** Diffusion policies have emerged as a powerful framework for robotic visuomotor control, yet they often lack the robustness to recover from subtask failures in long-horizon, multi-stage tasks and their learned representations of observations are often difficult to interpret. In this work, we propose the Mixture of Experts-Enhanced Diffusion Policy (MoE-DP), where the core idea is to insert a Mixture of Experts (MoE) layer between the visual encoder and the diffusion model. This layer decomposes the policy's knowledge into a set of specialized experts, which are dynamically activated to handle different phases of a task. We demonstrate through extensive experiments that MoE-DP exhibits a strong capability to recover from disturbances, significantly outperforming standard baselines in robustness. On a suite of 6 long-horizon simulation tasks, this leads to a 36% average relative improvement in success rate under disturbed conditions. This enhanced robustness is further validated in the real world, where MoE-DP also shows significant performance gains. We further show that MoE-DP learns an interpretable skill decomposition, where distinct experts correspond to semantic task primitives (e.g., approaching, grasping). This learned structure can be leveraged for inference-time control, allowing for the rearrangement of subtasks without any re-training.Our video and code are available at the https://moe-dp-website.github.io/MoE-DP-Website/.
>
---
#### [new 017] Design Exploration for Protection and Cleaning of Solar Panels with Case Studies for Space Missions
- **分类: cs.RO**

- **简介: 该论文针对太空任务中太阳能面板的尘埃覆盖与碎片撞击问题，设计并对比了刮板与轨道两种清洁系统，评估了多层材料防护效果，发现刮板系统更高效，软硬层叠结构可有效防护。属于系统设计与实验验证任务。**

- **链接: [http://arxiv.org/pdf/2511.04837v1](http://arxiv.org/pdf/2511.04837v1)**

> **作者:** Cameron Robinson; Ganghee Jang
>
> **备注:** 4 pages, 3 figures (5 assets)
>
> **摘要:** Solar energy is used for many mission-critical applications including space exploration, sensor systems to monitor wildfires, etc. Their operation can be limited or even terminated if solar panels are covered with dust or hit by space debris. To address this issue, we designed panel cleaning mechanisms and tested protective materials. For cleaning mechanisms, we designed and compared a wiper system and a rail system. For protective materials, we found through collision tests that polycarbonate was very promising, though the most important factor was layering a soft material between the panel's surface and a hard material. In the cleaning system comparisons, the wiper-based system was more efficient than the rail-based system in terms of cost, cleaning speed, and total power consumption.
>
---
#### [new 018] iFlyBot-VLM Technical Report
- **分类: cs.RO**

- **简介: iFlyBot-VLM是一项面向具身智能的通用视觉-语言模型，旨在弥合环境感知与机器人控制间的语义鸿沟，通过抽象为可迁移的操作语言，实现跨平台感知-行动闭环，支持空间理解、目标定位、动作生成与任务规划四类核心能力。**

- **链接: [http://arxiv.org/pdf/2511.04976v1](http://arxiv.org/pdf/2511.04976v1)**

> **作者:** Xin Nie; Zhiyuan Cheng; Yuan Zhang; Chao Ji; Jiajia Wu; Yuhan Zhang; Jia Pan
>
> **摘要:** We introduce iFlyBot-VLM, a general-purpose Vision-Language Model (VLM) used to improve the domain of Embodied Intelligence. The central objective of iFlyBot-VLM is to bridge the cross-modal semantic gap between high-dimensional environmental perception and low-level robotic motion control. To this end, the model abstracts complex visual and spatial information into a body-agnostic and transferable Operational Language, thereby enabling seamless perception-action closed-loop coordination across diverse robotic platforms. The architecture of iFlyBot-VLM is systematically designed to realize four key functional capabilities essential for embodied intelligence: 1) Spatial Understanding and Metric Reasoning; 2) Interactive Target Grounding; 3) Action Abstraction and Control Parameter Generation; 4) Task Planning and Skill Sequencing. We envision iFlyBot-VLM as a scalable and generalizable foundation model for embodied AI, facilitating the progression from specialized task-oriented systems toward generalist, cognitively capable agents. We conducted evaluations on 10 current mainstream embodied intelligence-related VLM benchmark datasets, such as Blink and Where2Place, and achieved optimal performance while preserving the model's general capabilities. We will publicly release both the training data and model weights to foster further research and development in the field of Embodied Intelligence.
>
---
#### [new 019] Encoding Biomechanical Energy Margin into Passivity-based Synchronization for Networked Telerobotic Systems
- **分类: cs.RO**

- **简介: 该论文针对网络化遥操作系统的位姿不同步问题，提出一种基于生物力学的双端无源同步器TBPS2，通过融入人体生物力学特性，优化同步性能并降低稳定性约束，提升系统在时延与非无源环境下的跟踪精度与安全性。**

- **链接: [http://arxiv.org/pdf/2511.04994v1](http://arxiv.org/pdf/2511.04994v1)**

> **作者:** Xingyuan Zhou; Peter Paik; S. Farokh Atashzar
>
> **摘要:** Maintaining system stability and accurate position tracking is imperative in networked robotic systems, particularly for haptics-enabled human-robot interaction. Recent literature has integrated human biomechanics into the stabilizers implemented for teleoperation, enhancing force preservation while guaranteeing convergence and safety. However, position desynchronization due to imperfect communication and non-passive behaviors remains a challenge. This paper proposes a two-port biomechanics-aware passivity-based synchronizer and stabilizer, referred to as TBPS2. This stabilizer optimizes position synchronization by leveraging human biomechanics while reducing the stabilizer's conservatism in its activation. We provide the mathematical design synthesis of the stabilizer and the proof of stability. We also conducted a series of grid simulations and systematic experiments, comparing their performance with that of state-of-the-art solutions under varying time delays and environmental conditions.
>
---
#### [new 020] Context-aware Learned Mesh-based Simulation via Trajectory-Level Meta-Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出M3GN，将网格仿真建模为轨迹级元学习任务，利用条件神经过程与运动基元，从少量数据快速适应新场景，提升长期仿真精度与效率，解决传统学习模拟器依赖单步观测与误差累积问题。**

- **链接: [http://arxiv.org/pdf/2511.05234v1](http://arxiv.org/pdf/2511.05234v1)**

> **作者:** Philipp Dahlinger; Niklas Freymuth; Tai Hoang; Tobias Würth; Michael Volpp; Luise Kärger; Gerhard Neumann
>
> **备注:** 35 pages. Submitted to Transactions on Machine Learning Research (TMLR)
>
> **摘要:** Simulating object deformations is a critical challenge across many scientific domains, including robotics, manufacturing, and structural mechanics. Learned Graph Network Simulators (GNSs) offer a promising alternative to traditional mesh-based physics simulators. Their speed and inherent differentiability make them particularly well suited for applications that require fast and accurate simulations, such as robotic manipulation or manufacturing optimization. However, existing learned simulators typically rely on single-step observations, which limits their ability to exploit temporal context. Without this information, these models fail to infer, e.g., material properties. Further, they rely on auto-regressive rollouts, which quickly accumulate error for long trajectories. We instead frame mesh-based simulation as a trajectory-level meta-learning problem. Using Conditional Neural Processes, our method enables rapid adaptation to new simulation scenarios from limited initial data while capturing their latent simulation properties. We utilize movement primitives to directly predict fast, stable and accurate simulations from a single model call. The resulting approach, Movement-primitive Meta-MeshGraphNet (M3GN), provides higher simulation accuracy at a fraction of the runtime cost compared to state-of-the-art GNSs across several tasks.
>
---
#### [new 021] Let Me Show You: Learning by Retrieving from Egocentric Video for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出一种基于视频检索的机器人学习方法（RfV），解决机器人在未知任务中泛化能力弱的问题。通过从人类演示视频库中检索相关示例，提取物体功能与手部轨迹等中层信息，联合生成自适应操控策略，显著提升任务泛化性能。**

- **链接: [http://arxiv.org/pdf/2511.05199v1](http://arxiv.org/pdf/2511.05199v1)**

> **作者:** Yichen Zhu; Feifei Feng
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Robots operating in complex and uncertain environments face considerable challenges. Advanced robotic systems often rely on extensive datasets to learn manipulation tasks. In contrast, when humans are faced with unfamiliar tasks, such as assembling a chair, a common approach is to learn by watching video demonstrations. In this paper, we propose a novel method for learning robot policies by Retrieving-from-Video (RfV), using analogies from human demonstrations to address manipulation tasks. Our system constructs a video bank comprising recordings of humans performing diverse daily tasks. To enrich the knowledge from these videos, we extract mid-level information, such as object affordance masks and hand motion trajectories, which serve as additional inputs to enhance the robot model's learning and generalization capabilities. We further feature a dual-component system: a video retriever that taps into an external video bank to fetch task-relevant video based on task specification, and a policy generator that integrates this retrieved knowledge into the learning cycle. This approach enables robots to craft adaptive responses to various scenarios and generalize to tasks beyond those in the training data. Through rigorous testing in multiple simulated and real-world settings, our system demonstrates a marked improvement in performance over conventional robotic systems, showcasing a significant breakthrough in the field of robotics.
>
---
#### [new 022] Decomposed Object Manipulation via Dual-Actor Policy
- **分类: cs.RO**

- **简介: 该论文提出双演员策略（DAP），将物体操作分解为接近与操作阶段，分别利用功能位姿与运动流视觉先验提升性能，并构建含七任务的模拟数据集，显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2511.05129v1](http://arxiv.org/pdf/2511.05129v1)**

> **作者:** Bin Fan; Jianjian Jiang; Zhuohao Li; Yixiang He; Xiaoming Wu; Yihan Yang; Shengbang Liu; Weishi Zheng
>
> **备注:** 9 pages, 7 figures, 5 tables
>
> **摘要:** Object manipulation, which focuses on learning to perform tasks on similar parts across different types of objects, can be divided into an approaching stage and a manipulation stage. However, previous works often ignore this characteristic of the task and rely on a single policy to directly learn the whole process of object manipulation. To address this problem, we propose a novel Dual-Actor Policy, termed DAP, which explicitly considers different stages and leverages heterogeneous visual priors to enhance each stage. Specifically, we introduce an affordance-based actor to locate the functional part in the manipulation task, thereby improving the approaching process. Following this, we propose a motion flow-based actor to capture the movement of the component, facilitating the manipulation process. Finally, we introduce a decision maker to determine the current stage of DAP and select the corresponding actor. Moreover, existing object manipulation datasets contain few objects and lack the visual priors needed to support training. To address this, we construct a simulated dataset, the Dual-Prior Object Manipulation Dataset, which combines the two visual priors and includes seven tasks, including two challenging long-term, multi-stage tasks. Experimental results on our dataset, the RoboTwin benchmark and real-world scenarios illustrate that our method consistently outperforms the SOTA method by 5.55%, 14.7% and 10.4% on average respectively.
>
---
#### [new 023] Beyond Master and Apprentice: Grounding Foundation Models for Symbiotic Interactive Learning in a Shared Latent Space
- **分类: cs.RO**

- **简介: 该论文提出共生交互学习（SIL），解决传统人机交互中单向指令执行问题，通过共享潜在空间实现人与机器人双向适应，支持主动澄清、建议与计划协同优化，基于基础模型与记忆架构提升长期学习稳定性。**

- **链接: [http://arxiv.org/pdf/2511.05203v1](http://arxiv.org/pdf/2511.05203v1)**

> **作者:** Linus Nwankwo; Björn Ellensohn; Christian Rauch; Elmar Rueckert
>
> **摘要:** Today's autonomous agents can understand free-form natural language instructions and execute long-horizon tasks in a manner akin to human-level reasoning. These capabilities are mostly driven by large-scale pre-trained foundation models (FMs). However, the approaches with which these models are grounded for human-robot interaction (HRI) perpetuate a master-apprentice model, where the apprentice (embodied agent) passively receives and executes the master's (human's) commands without reciprocal learning. This reactive interaction approach does not capture the co-adaptive dynamics inherent in everyday multi-turn human-human interactions. To address this, we propose a Symbiotic Interactive Learning (SIL) approach that enables both the master and the apprentice to co-adapt through mutual, bidirectional interactions. We formalised SIL as a co-adaptation process within a shared latent task space, where the agent and human maintain joint belief states that evolve based on interaction history. This enables the agent to move beyond reactive execution to proactive clarification, adaptive suggestions, and shared plan refinement. To realise these novel behaviours, we leveraged pre-trained FMs for spatial perception and reasoning, alongside a lightweight latent encoder that grounds the models' outputs into task-specific representations. Furthermore, to ensure stability as the tasks evolve, we augment SIL with a memory architecture that prevents the forgetting of learned task-space representations. We validate SIL on both simulated and real-world embodied tasks, including instruction following, information retrieval, query-oriented reasoning, and interactive dialogues. Demos and resources are public at:~\href{https://linusnep.github.io/SIL/}{https://linusnep.github.io/SIL/}.
>
---
#### [new 024] Pixi: Unified Software Development and Distribution for Robotics and AI
- **分类: cs.RO; cs.SE**

- **简介: Pixi提出统一包管理框架，解决机器人与AI领域依赖复杂、复现困难问题，通过锁文件实现跨平台精准复现，集成conda-forge与PyPI，提速依赖解析10倍，降低科研门槛。**

- **链接: [http://arxiv.org/pdf/2511.04827v1](http://arxiv.org/pdf/2511.04827v1)**

> **作者:** Tobias Fischer; Wolf Vollprecht; Bas Zalmstra; Ruben Arts; Tim de Jager; Alejandro Fontan; Adam D Hines; Michael Milford; Silvio Traversaro; Daniel Claes; Scarlett Raine
>
> **备注:** 20 pages, 3 figures, 11 code snippets
>
> **摘要:** The reproducibility crisis in scientific computing constrains robotics research. Existing studies reveal that up to 70% of robotics algorithms cannot be reproduced by independent teams, while many others fail to reach deployment because creating shareable software environments remains prohibitively complex. These challenges stem from fragmented, multi-language, and hardware-software toolchains that lead to dependency hell. We present Pixi, a unified package-management framework that addresses these issues by capturing exact dependency states in project-level lockfiles, ensuring bit-for-bit reproducibility across platforms. Its high-performance SAT solver achieves up to 10x faster dependency resolution than comparable tools, while integration of the conda-forge and PyPI ecosystems removes the need for multiple managers. Adopted in over 5,300 projects since 2023, Pixi reduces setup times from hours to minutes and lowers technical barriers for researchers worldwide. By enabling scalable, reproducible, collaborative research infrastructure, Pixi accelerates progress in robotics and AI.
>
---
#### [new 025] Follow-Me in Micro-Mobility with End-to-End Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究微移动设备（如电动轮椅）的跟随跟随任务，旨在提升用户舒适度。通过端到端模仿学习优化控制器，对比传统人工调参方法，实现在真实场景中实现最优跟随体验。**

- **链接: [http://arxiv.org/pdf/2511.05158v1](http://arxiv.org/pdf/2511.05158v1)**

> **作者:** Sahar Salimpour; Iacopo Catalano; Tomi Westerlund; Mohsen Falahi; Jorge Peña Queralta
>
> **摘要:** Autonomous micro-mobility platforms face challenges from the perspective of the typical deployment environment: large indoor spaces or urban areas that are potentially crowded and highly dynamic. While social navigation algorithms have progressed significantly, optimizing user comfort and overall user experience over other typical metrics in robotics (e.g., time or distance traveled) is understudied. Specifically, these metrics are critical in commercial applications. In this paper, we show how imitation learning delivers smoother and overall better controllers, versus previously used manually-tuned controllers. We demonstrate how DAAV's autonomous wheelchair achieves state-of-the-art comfort in follow-me mode, in which it follows a human operator assisting persons with reduced mobility (PRM). This paper analyzes different neural network architectures for end-to-end control and demonstrates their usability in real-world production-level deployments.
>
---
#### [new 026] Conformalized Non-uniform Sampling Strategies for Accelerated Sampling-based Motion Planning
- **分类: cs.RO**

- **简介: 该论文面向机器人运动规划，解决均匀采样效率低的问题，提出基于置信预测的非均匀采样策略，利用启发式路径预测的不确定性构造可信采样区域，显著加速规划并提升泛化性。**

- **链接: [http://arxiv.org/pdf/2511.04835v1](http://arxiv.org/pdf/2511.04835v1)**

> **作者:** Shubham Natraj; Bruno Sinopoli; Yiannis Kantaros
>
> **摘要:** Sampling-based motion planners (SBMPs) are widely used to compute dynamically feasible robot paths. However, their reliance on uniform sampling often leads to poor efficiency and slow planning in complex environments. We introduce a novel non-uniform sampling strategy that integrates into existing SBMPs by biasing sampling toward `certified' regions. These regions are constructed by (i) generating an initial, possibly infeasible, path using any heuristic path predictor (e.g., A* or vision-language models) and (ii) applying conformal prediction to quantify the predictor's uncertainty. This process yields prediction sets around the initial-guess path that are guaranteed, with user-specified probability, to contain the optimal solution. To our knowledge, this is the first non-uniform sampling approach for SBMPs that provides such probabilistically correct guarantees on the sampling regions. Extensive evaluations demonstrate that our method consistently finds feasible paths faster and generalizes better to unseen environments than existing baselines.
>
---
#### [new 027] SAD-Flower: Flow Matching for Safe, Admissible, and Dynamically Consistent Planning
- **分类: cs.LG; cs.RO; cs.SY; eess.SY**

- **简介: 论文提出SAD-Flower框架，解决流匹配规划中缺乏安全、可行与动力学一致性保障的问题，通过虚拟控制输入实现无重训练的约束满足，确保轨迹可执行且符合动态约束。**

- **链接: [http://arxiv.org/pdf/2511.05355v1](http://arxiv.org/pdf/2511.05355v1)**

> **作者:** Tzu-Yuan Huang; Armin Lederer; Dai-Jie Wu; Xiaobing Dai; Sihua Zhang; Stefan Sosnowski; Shao-Hua Sun; Sandra Hirche
>
> **摘要:** Flow matching (FM) has shown promising results in data-driven planning. However, it inherently lacks formal guarantees for ensuring state and action constraints, whose satisfaction is a fundamental and crucial requirement for the safety and admissibility of planned trajectories on various systems. Moreover, existing FM planners do not ensure the dynamical consistency, which potentially renders trajectories inexecutable. We address these shortcomings by proposing SAD-Flower, a novel framework for generating Safe, Admissible, and Dynamically consistent trajectories. Our approach relies on an augmentation of the flow with a virtual control input. Thereby, principled guidance can be derived using techniques from nonlinear control theory, providing formal guarantees for state constraints, action constraints, and dynamic consistency. Crucially, SAD-Flower operates without retraining, enabling test-time satisfaction of unseen constraints. Through extensive experiments across several tasks, we demonstrate that SAD-Flower outperforms various generative-model-based baselines in ensuring constraint satisfaction.
>
---
#### [new 028] Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文研究在线交互下的离动力学强化学习，提出“极值访问比”度量训练与部署动力学差异，首次实现高效、最优后悔界算法，解决探索困难问题。**

- **链接: [http://arxiv.org/pdf/2511.05396v1](http://arxiv.org/pdf/2511.05396v1)**

> **作者:** Yiting He; Zhishuai Liu; Weixin Wang; Pan Xu
>
> **备注:** 53 pages, 6 figures, 3 tables. Published in Proceedings of the 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** Off-dynamics reinforcement learning (RL), where training and deployment transition dynamics are different, can be formulated as learning in a robust Markov decision process (RMDP) where uncertainties in transition dynamics are imposed. Existing literature mostly assumes access to generative models allowing arbitrary state-action queries or pre-collected datasets with a good state coverage of the deployment environment, bypassing the challenge of exploration. In this work, we study a more realistic and challenging setting where the agent is limited to online interaction with the training environment. To capture the intrinsic difficulty of exploration in online RMDPs, we introduce the supremal visitation ratio, a novel quantity that measures the mismatch between the training dynamics and the deployment dynamics. We show that if this ratio is unbounded, online learning becomes exponentially hard. We propose the first computationally efficient algorithm that achieves sublinear regret in online RMDPs with $f$-divergence based transition uncertainties. We also establish matching regret lower bounds, demonstrating that our algorithm achieves optimal dependence on both the supremal visitation ratio and the number of interaction episodes. Finally, we validate our theoretical results through comprehensive numerical experiments.
>
---
#### [new 029] Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance
- **分类: cs.AI; cs.LG; cs.RO; cs.SE**

- **简介: 该论文探索使用LLM代理清洗汽车维护日志中的噪声（如错字、缺失字段等），以提升预测性维护模型性能，解决数据质量差与专家稀缺问题，验证了LLM在通用数据清洗任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2511.05311v1](http://arxiv.org/pdf/2511.05311v1)**

> **作者:** Valeriu Dimidov; Faisal Hawlader; Sasan Jafarnejad; Raphaël Frank
>
> **摘要:** Economic constraints, limited availability of datasets for reproducibility and shortages of specialized expertise have long been recognized as key challenges to the adoption and advancement of predictive maintenance (PdM) in the automotive sector. Recent progress in large language models (LLMs) presents an opportunity to overcome these barriers and speed up the transition of PdM from research to industrial practice. Under these conditions, we explore the potential of LLM-based agents to support PdM cleaning pipelines. Specifically, we focus on maintenance logs, a critical data source for training well-performing machine learning (ML) models, but one often affected by errors such as typos, missing fields, near-duplicate entries, and incorrect dates. We evaluate LLM agents on cleaning tasks involving six distinct types of noise. Our findings show that LLMs are effective at handling generic cleaning tasks and offer a promising foundation for future industrial applications. While domain-specific errors remain challenging, these results highlight the potential for further improvements through specialized training and enhanced agentic capabilities.
>
---
#### [new 030] Multi-agent Coordination via Flow Matching
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文提出MAC-Flow，用于多智能体协同任务，解决传统方法在行为表达力与推理速度间的权衡问题。通过流模型学习联合行为并蒸馏为快速单步策略，在保持性能的同时实现14.5倍加速。**

- **链接: [http://arxiv.org/pdf/2511.05005v1](http://arxiv.org/pdf/2511.05005v1)**

> **作者:** Dongsu Lee; Daehee Lee; Amy Zhang
>
> **摘要:** This work presents MAC-Flow, a simple yet expressive framework for multi-agent coordination. We argue that requirements of effective coordination are twofold: (i) a rich representation of the diverse joint behaviors present in offline data and (ii) the ability to act efficiently in real time. However, prior approaches often sacrifice one for the other, i.e., denoising diffusion-based solutions capture complex coordination but are computationally slow, while Gaussian policy-based solutions are fast but brittle in handling multi-agent interaction. MAC-Flow addresses this trade-off by first learning a flow-based representation of joint behaviors, and then distilling it into decentralized one-step policies that preserve coordination while enabling fast execution. Across four different benchmarks, including $12$ environments and $34$ datasets, MAC-Flow alleviates the trade-off between performance and computational cost, specifically achieving about $\boldsymbol{\times14.5}$ faster inference compared to diffusion-based MARL methods, while maintaining good performance. At the same time, its inference speed is similar to that of prior Gaussian policy-based offline multi-agent reinforcement learning (MARL) methods.
>
---
## 更新

#### [replaced 001] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models
- **分类: cs.RO; cs.AI; cs.HC**

- **链接: [http://arxiv.org/pdf/2511.02162v2](http://arxiv.org/pdf/2511.02162v2)**

> **作者:** Alexander Htet Kyaw; Richa Gupta; Dhruv Shah; Anoop Sinha; Kory Mathewson; Stefanie Pender; Sachin Chitta; Yotto Koga; Faez Ahmed; Lawrence Sass; Randall Davis
>
> **备注:** Accepted to NeurIPS 2025, Conference on Neural Information Processing Systems, Creative AI Track
>
> **摘要:** Advances in 3D generative AI have enabled the creation of physical objects from text prompts, but challenges remain in creating objects involving multiple component types. We present a pipeline that integrates 3D generative AI with vision-language models (VLMs) to enable the robotic assembly of multi-component objects from natural language. Our method leverages VLMs for zero-shot, multi-modal reasoning about geometry and functionality to decompose AI-generated meshes into multi-component 3D models using predefined structural and panel components. We demonstrate that a VLM is capable of determining which mesh regions need panel components in addition to structural components, based on the object's geometry and functionality. Evaluation across test objects shows that users preferred the VLM-generated assignments 90.6% of the time, compared to 59.4% for rule-based and 2.5% for random assignment. Lastly, the system allows users to refine component assignments through conversational feedback, enabling greater human control and agency in making physical objects with generative AI and robotics.
>
---
#### [replaced 002] Holistic Evaluation of Multimodal LLMs on Spatial Intelligence
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.13142v3](http://arxiv.org/pdf/2508.13142v3)**

> **作者:** Zhongang Cai; Yubo Wang; Qingping Sun; Ruisi Wang; Chenyang Gu; Wanqi Yin; Zhiqian Lin; Zhitao Yang; Chen Wei; Oscar Qian; Hui En Pang; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Jiaqi Li; Xiangyu Fan; Hanming Deng; Lewei Lu; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: https://github.com/EvolvingLMMs-Lab/EASI/
>
> **摘要:** Multimodal models have achieved remarkable progress in recent years. Nevertheless, they continue to exhibit notable limitations in spatial understanding and reasoning, the very capability that anchors artificial general intelligence in the physical world. With the recent release of GPT-5, allegedly the most powerful AI model to date, it is timely to examine where the leading models (GPT, Gemini, Grok, Seed, Qwen, and Intern) stand on the path toward spatial intelligence. We thus propose EASI for holistic Evaluation of multimodAl LLMs on Spatial Intelligence. EASI conceptualizes a comprehensive taxonomy of spatial tasks that unifies existing benchmarks and a standardized protocol for the fair evaluation of state-of-the-art proprietary and open-source models. In this report, we conduct the study across eight key benchmarks, at a cost exceeding ten billion total tokens. Our empirical study then reveals that (1) GPT-5 demonstrates unprecedented strength in spatial intelligence (SI), yet (2) still falls short of human performance significantly across a broad spectrum of SI-tasks. Moreover, we (3) show that SI-tasks expose greater model capability deficiency than non-SI tasks, to the extent that (4) proprietary models do not exhibit a decisive advantage when facing the most difficult ones. In addition, we conduct a qualitative evaluation across a diverse set of scenarios that are intuitive for humans, yet fail even the most advanced multimodal models.
>
---
#### [replaced 003] Mean-Shift Theory and Its Applications in Swarm Robotics: A New Way to Enhance the Efficiency of Multi-Robot Collaboration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.25086v2](http://arxiv.org/pdf/2510.25086v2)**

> **作者:** Guibin Sun; Jinhu Lü; Kexin Liu; Zhenqian Wang; Guanrong Chen
>
> **摘要:** Swarms evolving from collective behaviors among multiple individuals are commonly seen in nature, which enables biological systems to exhibit more efficient and robust collaboration. Creating similar swarm intelligence in engineered robots poses challenges to the design of collaborative algorithms that can be programmed at large scales. The assignment-based method has played an eminent role for a very long time in solving collaboration problems of robot swarms. However, it faces fundamental limitations in terms of efficiency and robustness due to its unscalability to swarm variants. This article presents a tutorial review on recent advances in assignment-free collaboration of robot swarms, focusing on the problem of shape formation. A key theoretical component is the recently developed \emph{mean-shift exploration} strategy, which improves the collaboration efficiency of large-scale swarms by dozens of times. Further, the efficiency improvement is more significant as the swarm scale increases. Finally, this article discusses three important applications of the mean-shift exploration strategy, including precise shape formation, area coverage formation, and maneuvering formation, as well as their corresponding industrial scenarios in smart warehousing, area exploration, and cargo transportation.
>
---
#### [replaced 004] Joint Verification and Refinement of Language Models for Safety-Constrained Planning
- **分类: cs.AI; cs.FL; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.14865v2](http://arxiv.org/pdf/2410.14865v2)**

> **作者:** Yunhao Yang; Neel P. Bhatt; William Ward; Zichao Hu; Joydeep Biswas; Ufuk Topcu
>
> **摘要:** Large language models possess impressive capabilities in generating programs (e.g., Python) from natural language descriptions to execute robotic tasks. However, these generated programs often contain errors that violate externally given task specifications. Without an effective method to verify their correctness, the reliable deployment of language models in real-world systems is practically infeasible. We develop a method that converts generated robot programs into an automaton-based representation and verifies them against task-relevant safety specifications. We establish a theorem that any arbitrary combination of the verified programs will also satisfy the safety specifications. Hence, the method eliminates the need to verify complex programs composed of multiple simpler ones, reducing computation complexity. We then introduce an automated fine-tuning procedure that leverages verification outcomes for supervision. By applying the theorem, this procedure only requires training the model to generate safe sub-components, thereby improving training efficiency. Empirical results on robot applications show a 30 percent increase in the probability of generating specification-compliant programs, with training time reduced by half compared to fine-tuning on generating full programs.
>
---
#### [replaced 005] Toward Engineering AGI: Benchmarking the Engineering Design Capabilities of LLMs
- **分类: cs.CE; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.16204v2](http://arxiv.org/pdf/2509.16204v2)**

> **作者:** Xingang Guo; Yaxin Li; Xiangyi Kong; Yilan Jiang; Xiayu Zhao; Zhihua Gong; Yufan Zhang; Daixuan Li; Tianle Sang; Beixiao Zhu; Gregory Jun; Yingbing Huang; Yiqi Liu; Yuqi Xue; Rahul Dev Kundu; Qi Jian Lim; Yizhou Zhao; Luke Alexander Granger; Mohamed Badr Younis; Darioush Keivan; Nippun Sabharwal; Shreyanka Sinha; Prakhar Agarwal; Kojo Vandyck; Hanlin Mai; Zichen Wang; Aditya Venkatesh; Ayush Barik; Jiankun Yang; Chongying Yue; Jingjie He; Libin Wang; Licheng Xu; Hao Chen; Jinwen Wang; Liujun Xu; Rushabh Shetty; Ziheng Guo; Dahui Song; Manvi Jha; Weijie Liang; Weiman Yan; Bryan Zhang; Sahil Bhandary Karnoor; Jialiang Zhang; Rutva Pandya; Xinyi Gong; Mithesh Ballae Ganesh; Feize Shi; Ruiling Xu; Yifan Zhang; Yanfeng Ouyang; Lianhui Qin; Elyse Rosenbaum; Corey Snyder; Peter Seiler; Geir Dullerud; Xiaojia Shelly Zhang; Zuofu Cheng; Pavan Kumar Hanumolu; Jian Huang; Mayank Kulkarni; Mahdi Namazifar; Huan Zhang; Bin Hu
>
> **备注:** To Appear in NeurIPS 2025 Datasets & Benchmarks Track
>
> **摘要:** Modern engineering, spanning electrical, mechanical, aerospace, civil, and computer disciplines, stands as a cornerstone of human civilization and the foundation of our society. However, engineering design poses a fundamentally different challenge for large language models (LLMs) compared with traditional textbook-style problem solving or factual question answering. Although existing benchmarks have driven progress in areas such as language understanding, code synthesis, and scientific problem solving, real-world engineering design demands the synthesis of domain knowledge, navigation of complex trade-offs, and management of the tedious processes that consume much of practicing engineers' time. Despite these shared challenges across engineering disciplines, no benchmark currently captures the unique demands of engineering design work. In this work, we introduce EngDesign, an Engineering Design benchmark that evaluates LLMs' abilities to perform practical design tasks across nine engineering domains. Unlike existing benchmarks that focus on factual recall or question answering, EngDesign uniquely emphasizes LLMs' ability to synthesize domain knowledge, reason under constraints, and generate functional, objective-oriented engineering designs. Each task in EngDesign represents a real-world engineering design problem, accompanied by a detailed task description specifying design goals, constraints, and performance requirements. EngDesign pioneers a simulation-based evaluation paradigm that moves beyond textbook knowledge to assess genuine engineering design capabilities and shifts evaluation from static answer checking to dynamic, simulation-driven functional verification, marking a crucial step toward realizing the vision of engineering Artificial General Intelligence (AGI).
>
---
#### [replaced 006] Pogobot: an Open-Source, Low-Cost Robot for Swarm Robotics and Programmable Active Matter
- **分类: cs.RO; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.08686v2](http://arxiv.org/pdf/2504.08686v2)**

> **作者:** Alessia Loi; Loona Macabre; Jérémy Fersula; Keivan Amini; Leo Cazenille; Fabien Caura; Alexandre Guerre; Stéphane Gourichon; Laurent Fabre; Olivier Dauchot; Nicolas Bredeche
>
> **摘要:** This paper describes the Pogobot, an open-source platform specifically designed for research at the interface of swarm robotics and active matter. Pogobot features vibration-based or wheel-based locomotion, fast infrared communication, and an array of sensors in a cost-effective package (approx. 250euros/unit). The platform's modular design, comprehensive API, and extensible architecture facilitate the implementation of swarm intelligence algorithms and collective motion. Pogobots offer an accessible alternative to existing platforms while providing advanced capabilities including directional communication between units and fast locomotion, all with a compact form factor. More than 200 Pogobots are already being used on a daily basis in several Universities to study self-organizing systems, programmable active matter, discrete reaction-diffusion-advection systems and computational models of social learning and evolution. This paper details the hardware and software architecture, communication protocols, locomotion mechanisms, and the infrastructure built around the Pogobots.
>
---
#### [replaced 007] Octopus-like Reaching Motion: A Perspective Inspired by Whipping
- **分类: cs.RO; physics.bio-ph**

- **链接: [http://arxiv.org/pdf/2510.25520v2](http://arxiv.org/pdf/2510.25520v2)**

> **作者:** Shengyao Zhang; Yiyuan Zhang; Chenrui Zhang; Yiming Li; Wenci Xin; Yuliang Liufu; Hong Wei Ng; Cecilia Laschi
>
> **备注:** The first two listed authors contributed equally. Yiyuan Zhang is the corresponding author
>
> **摘要:** The stereotypical reaching motion of the octopus arm has drawn growing attention for its efficient control of a highly deformable body. Previous studies suggest that its characteristic bend propagation may share underlying principles with the dynamics of a whip. This work investigates whether whip-like passive dynamics in water can reproduce the kinematic features observed in biological reaching and their similarities and differences. Platform-based whipping tests were performed in water and air while systematically varying material stiffness and driving speed. Image-based quantification revealed that the Ecoflex Gel 2 arm driven at 150 rpm (motor speed) reproduced curvature propagation similar to that observed in octopus reaching. However, its bend-point velocity decreased monotonically rather than exhibiting the biological bell-shaped profile, confirming that the octopus reaching movement is not merely a passive whipping behavior. The absence of propagation in air further highlights the critical role of the surrounding medium in forming octopus-like reaching motion. This study provides a new perspective for understand biological reaching movement, and offers a potential platform for future hydrodynamic research.
>
---
#### [replaced 008] Tactical Decision Making for Autonomous Trucks by Deep Reinforcement Learning with Total Cost of Operation Based Reward
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2403.06524v2](http://arxiv.org/pdf/2403.06524v2)**

> **作者:** Deepthi Pathare; Leo Laine; Morteza Haghir Chehreghani
>
> **备注:** Paper is accepted for publication in Artificial Intelligence Review
>
> **摘要:** We develop a deep reinforcement learning framework for tactical decision making in an autonomous truck, specifically for Adaptive Cruise Control (ACC) and lane change maneuvers in a highway scenario. Our results demonstrate that it is beneficial to separate high-level decision-making processes and low-level control actions between the reinforcement learning agent and the low-level controllers based on physical models. In the following, we study optimizing the performance with a realistic and multi-objective reward function based on Total Cost of Operation (TCOP) of the truck using different approaches; by adding weights to reward components, by normalizing the reward components and by using curriculum learning techniques.
>
---
#### [replaced 009] Affordance-based Robot Manipulation with Flow Matching
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.01083v5](http://arxiv.org/pdf/2409.01083v5)**

> **作者:** Fan Zhang; Michael Gienger
>
> **摘要:** We present a framework for assistive robot manipulation, which focuses on two fundamental challenges: first, efficiently adapting large-scale models to downstream scene affordance understanding tasks, especially in daily living scenarios where gathering multi-task data involving humans requires strenuous effort; second, effectively learning robot action trajectories by grounding the visual affordance model. We tackle the first challenge by employing a parameter-efficient prompt tuning method that prepends learnable text prompts to the frozen vision model to predict manipulation affordances in multi-task scenarios. Then we propose to learn robot action trajectories guided by affordances in a supervised flow matching method. Flow matching represents a robot visuomotor policy as a conditional process of flowing random waypoints to desired robot action trajectories. Finally, we introduce a real-world dataset with 10 tasks across Activities of Daily Living to test our framework. Our extensive evaluation highlights that the proposed prompt tuning method for learning manipulation affordance achieves competitive performance and even outperforms some other finetuning protocols across data scales, while satisfying parameter efficiency. Learning multi-task robot action trajectories with flow matching leads to consistently favorable results in several robot manipulation benchmarks than some alternative behavior cloning methods. This includes more stable training and evaluation, and noticeably faster inference, while maintaining comparable generalization performance to diffusion policy, where flow matching performs marginally better in most cases. Our framework seamlessly unifies affordance learning and action generation with flow matching for robot manipulation.
>
---
#### [replaced 010] Periodic Skill Discovery
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2511.03187v2](http://arxiv.org/pdf/2511.03187v2)**

> **作者:** Jonghae Park; Daesol Cho; Jusuk Lee; Dongseok Shim; Inkyu Jang; H. Jin Kim
>
> **备注:** NeurIPS 2025
>
> **摘要:** Unsupervised skill discovery in reinforcement learning (RL) aims to learn diverse behaviors without relying on external rewards. However, current methods often overlook the periodic nature of learned skills, focusing instead on increasing the mutual dependence between states and skills or maximizing the distance traveled in latent space. Considering that many robotic tasks - particularly those involving locomotion - require periodic behaviors across varying timescales, the ability to discover diverse periodic skills is essential. Motivated by this, we propose Periodic Skill Discovery (PSD), a framework that discovers periodic behaviors in an unsupervised manner. The key idea of PSD is to train an encoder that maps states to a circular latent space, thereby naturally encoding periodicity in the latent representation. By capturing temporal distance, PSD can effectively learn skills with diverse periods in complex robotic tasks, even with pixel-based observations. We further show that these learned skills achieve high performance on downstream tasks such as hurdling. Moreover, integrating PSD with an existing skill discovery method offers more diverse behaviors, thus broadening the agent's repertoire. Our code and demos are available at https://jonghaepark.github.io/psd/
>
---
#### [replaced 011] ReNiL: Event-Driven Pedestrian Bayesian Localization Using IMU for Real-World Applications
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.06053v3](http://arxiv.org/pdf/2508.06053v3)**

> **作者:** Kaixuan Wu; Yuanzhuo Xu; Zejun Zhang; Weiping Zhu; Jian Zhang; Steve Drew; Xiaoguang Niu
>
> **备注:** This work has been submitted to the ACM for possible publication
>
> **摘要:** Pedestrian inertial localization is key for mobile and IoT services because it provides infrastructure-free positioning. Yet most learning-based methods depend on fixed sliding-window integration, struggle to adapt to diverse motion scales and cadences, and yield inconsistent uncertainty, limiting real-world use. We present ReNiL, a Bayesian deep-learning framework for accurate, efficient, and uncertainty-aware pedestrian localization. ReNiL introduces Inertial Positioning Demand Points (IPDPs) to estimate motion at contextually meaningful waypoints instead of dense tracking, and supports inference on IMU sequences at any scale so cadence can match application needs. It couples a motion-aware orientation filter with an Any-Scale Laplace Estimator (ASLE), a dual-task network that blends patch-based self-supervision with Bayesian regression. By modeling displacements with a Laplace distribution, ReNiL provides homogeneous Euclidean uncertainty that integrates cleanly with other sensors. A Bayesian inference chain links successive IPDPs into consistent trajectories. On RoNIN-ds and a new WUDataset covering indoor and outdoor motion from 28 participants, ReNiL achieves state-of-the-art displacement accuracy and uncertainty consistency, outperforming TLIO, CTIN, iMoT, and RoNIN variants while reducing computation. Application studies further show robustness and practicality for mobile and IoT localization, making ReNiL a scalable, uncertainty-aware foundation for next-generation positioning.
>
---
#### [replaced 012] Generalizing Robot Trajectories from Single-Context Human Demonstrations: A Probabilistic Approach
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.05619v2](http://arxiv.org/pdf/2503.05619v2)**

> **作者:** Qian Ying Lee; Suhas Raghavendra Kulkarni; Kenzhi Iskandar Wong; Lin Yang; Bernardo Noronha; Yongjun Wee; Domenico Campolo
>
> **摘要:** Generalizing robot trajectories from human demonstrations to new contexts remains a key challenge in Learning from Demonstration (LfD), particularly when only single-context demonstrations are available. We present a novel Gaussian Mixture Model (GMM)-based approach that enables systematic generalization from single-context demonstrations to a wide range of unseen start and goal configurations. Our method performs component-level reparameterization of the GMM, adapting both mean vectors and covariance matrices, followed by Gaussian Mixture Regression (GMR) to generate smooth trajectories. We evaluate the approach on a dual-arm pick-and-place task with varying box placements, comparing against several baselines. Results show that our method significantly outperforms baselines in trajectory success and fidelity, maintaining accuracy even under combined translational and rotational variations of task configurations. These results demonstrate that our method generalizes effectively while ensuring boundary convergence and preserving the intrinsic structure of demonstrated motions.
>
---
#### [replaced 013] Ethics-Aware Safe Reinforcement Learning for Rare-Event Risk Control in Interactive Urban Driving
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.14926v3](http://arxiv.org/pdf/2508.14926v3)**

> **作者:** Dianzhao Li; Ostap Okhrin
>
> **摘要:** Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding credible and transparent ethical reasoning into routine and emergency maneuvers, particularly to protect vulnerable road users (VRUs) such as pedestrians and cyclists. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that augments standard driving objectives with ethics-aware cost signals. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic, risk-sensitive Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on closed-loop simulation environments derived from large-scale, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing risk to others while maintaining ego performance and comfort. This work provides a reproducible benchmark for Safe RL with explicitly ethics-aware objectives in human-mixed traffic scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy that explicitly protects those most at risk in urban traffic environments. Across two interactive benchmarks and five random seeds, our policy decreases conflict frequency by 25-45% compared to matched task successes while maintaining comfort metrics within 5%.
>
---
#### [replaced 014] GeoAware-VLA: Implicit Geometry Aware Vision-Language-Action Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.14117v3](http://arxiv.org/pdf/2509.14117v3)**

> **作者:** Ali Abouzeid; Malak Mansour; Zezhou Sun; Dezhen Song
>
> **备注:** Under Review, Project Page https://alisharey.github.io/GeoAware-VLA/
>
> **摘要:** Vision-Language-Action (VLA) models often fail to generalize to novel camera viewpoints, a limitation stemming from their difficulty in inferring robust 3D geometry from 2D images. We introduce GeoAware-VLA, a simple yet effective approach that enhances viewpoint invariance by integrating strong geometric priors into the vision backbone. Instead of training a visual encoder or relying on explicit 3D data, we leverage a frozen, pretrained geometric vision model as a feature extractor. A trainable projection layer then adapts these geometrically-rich features for the policy decoder, relieving it of the burden of learning 3D consistency from scratch. Through extensive evaluations on LIBERO benchmark subsets, we show GeoAware-VLA achieves substantial improvements in zero-shot generalization to novel camera poses, boosting success rates by over 2x in simulation. Crucially, these benefits translate to the physical world; our model shows a significant performance gain on a real robot, especially when evaluated from unseen camera angles. Our approach proves effective across both continuous and discrete action spaces, highlighting that robust geometric grounding is a key component for creating more generalizable robotic agents.
>
---
#### [replaced 015] Learning to Navigate Socially Through Proactive Risk Perception
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.07871v3](http://arxiv.org/pdf/2510.07871v3)**

> **作者:** Erjia Xiao; Lingfeng Zhang; Yingbo Tang; Hao Cheng; Renjing Xu; Wenbo Ding; Lei Zhou; Long Chen; Hangjun Ye; Xiaoshuai Hao
>
> **摘要:** In this report, we describe the technical details of our submission to the IROS 2025 RoboSense Challenge Social Navigation Track. This track focuses on developing RGBD-based perception and navigation systems that enable autonomous agents to navigate safely, efficiently, and socially compliantly in dynamic human-populated indoor environments. The challenge requires agents to operate from an egocentric perspective using only onboard sensors including RGB-D observations and odometry, without access to global maps or privileged information, while maintaining social norm compliance such as safe distances and collision avoidance. Building upon the Falcon model, we introduce a Proactive Risk Perception Module to enhance social navigation performance. Our approach augments Falcon with collision risk understanding that learns to predict distance-based collision risk scores for surrounding humans, which enables the agent to develop more robust spatial awareness and proactive collision avoidance behaviors. The evaluation on the Social-HM3D benchmark demonstrates that our method improves the agent's ability to maintain personal space compliance while navigating toward goals in crowded indoor scenes with dynamic human agents, achieving 2nd place among 16 participating teams in the challenge.
>
---
#### [replaced 016] Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11350v5](http://arxiv.org/pdf/2505.11350v5)**

> **作者:** Derek Ming Siang Tan; Shailesh; Boyang Liu; Alok Raj; Qi Xuan Ang; Weiheng Dai; Tanishq Duhan; Jimmy Chiun; Yuhong Cao; Florian Shkurti; Guillaume Sartoretti
>
> **备注:** Accepted for presentation at CORL 2025. Code, models, and data are available at https://search-tta.github.io/
>
> **摘要:** To perform outdoor visual navigation and search, a robot may leverage satellite imagery to generate visual priors. This can help inform high-level search strategies, even when such images lack sufficient resolution for target recognition. However, many existing informative path planning or search-based approaches either assume no prior information, or use priors without accounting for how they were obtained. Recent work instead utilizes large Vision Language Models (VLMs) for generalizable priors, but their outputs can be inaccurate due to hallucination, leading to inefficient search. To address these challenges, we introduce Search-TTA, a multimodal test-time adaptation framework with a flexible plug-and-play interface compatible with various input modalities (e.g., image, text, sound) and planning methods (e.g., RL-based). First, we pretrain a satellite image encoder to align with CLIP's visual encoder to output probability distributions of target presence used for visual search. Second, our TTA framework dynamically refines CLIP's predictions during search using uncertainty-weighted gradient updates inspired by Spatial Poisson Point Processes. To train and evaluate Search-TTA, we curate AVS-Bench, a visual search dataset based on internet-scale ecological data containing 380k images and taxonomy data. We find that Search-TTA improves planner performance by up to 30.0%, particularly in cases with poor initial CLIP predictions due to domain mismatch and limited training data. It also performs comparably with significantly larger VLMs, and achieves zero-shot generalization via emergent alignment to unseen modalities. Finally, we deploy Search-TTA on a real UAV via hardware-in-the-loop testing, by simulating its operation within a large-scale simulation that provides onboard sensing.
>
---
