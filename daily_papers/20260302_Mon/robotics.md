# 机器人 cs.RO

- **最新发布 39 篇**

- **更新 37 篇**

## 最新发布

#### [new 001] Learning Robust Control Policies for Inverted Pose on Miniature Blimp Robots
- **分类: cs.RO**

- **简介: 该论文属于控制任务，旨在解决微型飞艇机器人在倒置状态下的稳定控制问题。通过构建仿真环境、训练强化学习策略并设计映射层，提升其在真实环境中的表现。**

- **链接: [https://arxiv.org/pdf/2602.23972](https://arxiv.org/pdf/2602.23972)**

> **作者:** Yuanlin Yang; Lin Hong; Fumin Zhang
>
> **摘要:** The ability to achieve and maintain inverted poses is essential for unlocking the full agility of miniature blimp robots (MBRs). However, developing reliable control methods for MBRs remains challenging due to their complex and underactuated dynamics. To address this challenge, we propose a novel framework that enables robust control policy learning for inverted pose on MBRs. The proposed framework operates through three core stages: First, a high-fidelity three-dimensional (3D) simulation environment was constructed, which was calibrated against real-world MBR motion data to ensure accurate replication of inverted-state dynamics. Second, a robust policy for MBR inverted control was trained within the simulation environment via a domain randomization strategy and a modified Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm. Third, a mapping layer was designed to bridge the sim-to-real gap for the learned policy deployment. Comprehensive evaluations in the simulation environment demonstrate that the learned policy achieves a higher success rate compared to the energy-shaping controller. Furthermore, experimental results confirm that the learned policy with a mapping layer enables an MBR to achieve and maintain a fully upside-down pose in real-world settings.
>
---
#### [new 002] OmniTrack: General Motion Tracking via Physics-Consistent Reference
- **分类: cs.RO**

- **简介: 该论文属于运动跟踪任务，解决机器人运动跟踪中的物理可行性问题。通过分阶段训练，先生成合理运动轨迹，再训练控制策略，提升跟踪精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.23832](https://arxiv.org/pdf/2602.23832)**

> **作者:** Yuhan Li; Peiyuan Zhi; Yunshen Wang; Tengyu Liu; Sixu Yan; Wenyu Liu; Xinggang Wang; Baoxiong Jia; Siyuan Huang
>
> **备注:** website: this https URL
>
> **摘要:** Learning motion tracking from rich human motion data is a foundational task for achieving general control in humanoid robots, enabling them to perform diverse behaviors. However, discrepancies in morphology and dynamics between humans and robots, combined with data noise, introduce physically infeasible artifacts in reference motions, such as floating and penetration. During both training and execution, these artifacts create a conflict between following inaccurate reference motions and maintaining the robot's stability, hindering the development of a generalizable motion tracking policy. To address these challenges, we introduce OmniTrack, a general tracking framework that explicitly decouples physical feasibility from general motion tracking. In the first stage, a privileged generalist policy generates physically plausible motions that strictly adhere to the robot's dynamics via trajectory rollout in simulation. In the second stage, the general control policy is trained to track these physically feasible motions, ensuring stable and coherent control transfer to the real robot. Experiments show that OmniTrack improves tracking accuracy and demonstrates strong generalization to unseen motions. In real-world tests, OmniTrack achieves hour-long, consistent, and stable tracking, including complex acrobatic motions such as flips and cartwheels. Additionally, we show that OmniTrack supports human-style stable and dynamic online teleoperation, highlighting its robustness and adaptability to varying user inputs.
>
---
#### [new 003] Learning to Build: Autonomous Robotic Assembly of Stable Structures Without Predefined Plans
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人自主装配任务，解决无预设蓝图下的结构构建问题。通过强化学习方法，实现灵活适应环境变化的装配过程。**

- **链接: [https://arxiv.org/pdf/2602.23934](https://arxiv.org/pdf/2602.23934)**

> **作者:** Jingwen Wang; Johannes Kirschner; Paul Rolland; Luis Salamanca; Stefana Parascho
>
> **摘要:** This paper presents a novel autonomous robotic assembly framework for constructing stable structures without relying on predefined architectural blueprints. Instead of following fixed plans, construction tasks are defined through targets and obstacles, allowing the system to adapt more flexibly to environmental uncertainty and variations during the building process. A reinforcement learning (RL) policy, trained using deep Q-learning with successor features, serves as the decision-making component. As a proof of concept, we evaluate the approach on a benchmark of 15 2D robotic assembly tasks of discrete block construction. Experiments using a real-world closed-loop robotic setup demonstrate the feasibility of the method and its ability to handle construction noise. The results suggest that our framework offers a promising direction for more adaptable and robust robotic construction in real-world environments.
>
---
#### [new 004] Printed helicoids with embedded air channels make sensorized segments for soft continuum robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人传感任务，旨在解决软体机器人感知与控制难题。通过嵌入气道和传感器，实现对螺旋结构软体机器人的变形感知与控制。**

- **链接: [https://arxiv.org/pdf/2602.23457](https://arxiv.org/pdf/2602.23457)**

> **作者:** Annan Zhang; Hanna Matusik; Miguel Flores-Acton; Emily R. Sologuren; Joshua Jacob; Daniela Rus
>
> **备注:** Accepted for publication in the proceedings of the 2026 IEEE 9th International Conference on Soft Robotics (RoboSoft)
>
> **摘要:** Soft robots enable safe, adaptive interaction with complex environments but remain difficult to sense and control due to their highly deformable structures. Architected soft materials such as helicoid lattices offer tunable stiffness and strength but are challenging to instrument because of their sparse geometry. We introduce a fabrication method for embedding air channels into helicoid-based soft continuum robots. Multi-material segments fabricated via vision-controlled jetting in a single print interface with PCBs housing miniature pressure sensors and IMUs for distributed deformation sensing. We characterize the mechanical properties of four helicoid designs and validate the sensor response to fundamental deformation modes. To demonstrate the platform's scalability, we construct and mechanically evaluate a meter-scale, 14-DoF cable-driven soft arm capable of open-loop trajectory tracking and object grasping, with tactile-based stiffness detection demonstrated using the gripper sensors. This approach establishes a scalable fabrication strategy for sensorized architected materials in large-scale soft robotic systems.
>
---
#### [new 005] SpikingTac: A Miniaturized Neuromorphic Visuotactile Sensor for High-Precision Dynamic Tactile Imprint Tracking
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在解决动态触觉追踪精度低的问题。提出SpikingTac传感器，实现高精度、高速度的触觉信息处理。**

- **链接: [https://arxiv.org/pdf/2602.23654](https://arxiv.org/pdf/2602.23654)**

> **作者:** Tianyu Jiang; Chaofan Zhang; Shaolin Zhang; Shaowei Cui; Shuo Wang
>
> **摘要:** High-speed event-driven tactile sensors are essential for achieving human-like dynamic manipulation, yet their integration is often limited by the bulkiness of standard event cameras. This paper presents SpikingTac, a miniaturized, highly integrated neuromorphic tactile sensor featuring a custom standalone event camera module, achieved with a total material cost of less than \$150. We construct a global dynamic state map coupled with an unsupervised denoising network to enable precise tracking at a 1000~Hz perception rate and 350~Hz tracking frequency. Addressing the viscoelastic hysteresis of silicone elastomers, we propose a hysteresis-aware incremental update law with a spatial gain damping mechanism. Experimental results demonstrate exceptional zero-point stability, achieving a 100\% return-to-origin success rate with a minimal mean bias of 0.8039 pixels, even under extreme torsional deformations. In dynamic tasks, SpikingTac limits the obstacle-avoidance overshoot to 6.2~mm, representing a 5-fold performance improvement over conventional frame-based sensors. Furthermore, the sensor achieves sub-millimeter geometric accuracy, with Root Mean Square Error (RMSE) of 0.0952~mm in localization and 0.0452~mm in radius measurement.
>
---
#### [new 006] V-MORALS: Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人安全分析任务，解决无全状态信息下区域吸引域估计问题。通过学习潜在空间并生成Morse图，实现基于传感器数据的区域吸引域计算。**

- **链接: [https://arxiv.org/pdf/2602.23524](https://arxiv.org/pdf/2602.23524)**

> **作者:** Faiz Aladin; Ashwin Balasubramanian; Lars Lindemann; Daniel Seita
>
> **摘要:** Reachability analysis has become increasingly important in robotics to distinguish safe from unsafe states. Unfortunately, existing reachability and safety analysis methods often fall short, as they typically require known system dynamics or large datasets to estimate accurate system models, are computationally expensive, and assume full state information. A recent method, called MORALS, aims to address these shortcomings by using topological tools to estimate3DR-eEgnciodnesr of Attraction (ROA) in a low-dimensional latent space. However, MORALS still relies on full state knowledge and has not been studied when only sensor measurements are available. This paper presents Visual Morse Graph-Aided Estimation of Regions of Attraction in a Learned Latent Space (V- MORALS). V-MORALS takes in a dataset of image-based trajectories of a system under a given controller, and learns a latent space for reachability analysis. Using this learned latent space, our method is able to generate well-defined Morse Graphs, from which we can compute ROAs for various systems and controllers. V-MORALS provides capabilities similar to the original MORALS architecture without relying on state knowledge, and using only high-level sensor data. Our project website is at: this https URL.
>
---
#### [new 007] A Reliable Indoor Navigation System for Humans Using AR-based Technique
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于室内导航任务，旨在解决传统导航方法效率低、体验差的问题。通过AR技术结合A*算法，提升导航准确性与实时性。**

- **链接: [https://arxiv.org/pdf/2602.23706](https://arxiv.org/pdf/2602.23706)**

> **作者:** Vijay U.Rathod; Manav S.Sharma; Shambhavi Verma; Aadi Joshi; Sachin Aage; Sujal Shahane
>
> **备注:** 6 pages, 6 figures, 2 tables, Presented at 7th International Conference on Advances in Science and Technology (ICAST 2024-25)
>
> **摘要:** Reliable navigation systems are not available indoors, such as in campuses and small areas. Users must depend on confusing, time-consuming static signage or floor maps. In this paper, an AR-based technique has been applied to campus and small-site navigation, where Vuforia Area Target is used for environment modeling. AI navigation's NavMesh component is used for navigation purposes, and the A* algorithm is used within this component for shortest path calculation. Compared to Dijkstra's algorithm, it can reach a solution about two to three times faster for smaller search spaces. In many cases, Dijkstra's algorithm has difficulty performing well in high-complexity environments where memory usage grows and processing times increase. Compared to older approaches such as GPS, real-time processing and AR overlays can be combined to provide intuitive directions for users while dynamically updating the path in response to environmental changes. Experimental results indicate significantly improved navigation accuracy, better user experience, and greater efficiency compared to traditional methods. These results show that AR technology integrated with existing pathfinding algorithms is feasible and scalable, making it a user-friendly solution for indoor navigation. Although highly effective in limited and defined indoor spaces, further optimization of NavMesh is required for large or highly dynamic environments.
>
---
#### [new 008] MicroPush: A Simulator and Benchmark for Contact-Rich Cell Pushing and Assembly with a Magnetic Rolling Microrobot
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出MicroPush，用于磁性微机器人在复杂环境中的细胞推动与组装任务，解决自主控制与评估难题。**

- **链接: [https://arxiv.org/pdf/2602.23607](https://arxiv.org/pdf/2602.23607)**

> **作者:** Yanda Yang; Sambeeta Das
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Magnetic rolling microrobots enable gentle manipulation in confined microfluidic environments, yet autonomy for contact-rich behaviors such as cell pushing and multi-target assembly remains difficult to develop and evaluate reproducibly. We present MicroPush, an open-source simulator and benchmark suite for magnetic rolling microrobots in cluttered 2D scenes. MicroPush combines an overdamped interaction model with contact-aware stick--slip effects, lightweight near-field damping, optional Poiseuille background flow, and a calibrated mapping from actuation frequency to free-space rolling speed. On top of the simulator core, we provide a modular planning--control stack with a two-phase strategy for contact establishment and goal-directed pushing, together with a deterministic benchmark protocol with fixed tasks, staged execution, and unified CSV logging for single-object transport and hexagonal assembly. We report success, time, and tracking metrics, and an actuation-variation measure $E_{\Delta\omega}$. Results show that controller stability dominates performance under flow disturbances, while planner choice can influence command smoothness over long-horizon sequences via waypoint progression. MicroPush enables reproducible comparison and ablation of planning, control, and learning methods for microscale contact-rich micromanipulation.
>
---
#### [new 009] Geometry-based pneumatic actuators for soft robotics
- **分类: cs.RO**

- **简介: 该论文提出几何气动执行器（GPAs），解决软体机器人复杂运动模式设计难题，通过可配置结构实现可控变形与多状态驱动。**

- **链接: [https://arxiv.org/pdf/2602.24104](https://arxiv.org/pdf/2602.24104)**

> **作者:** Rui Chen; Daniele Leonardis; Domenico Chiaradia; Antonio Frisoli
>
> **摘要:** Soft pneumatic actuators enable safe human-machine interaction with lightweight and powerful applied parts. On the other side, they suffer design limitations as regards complex actuation patterns, including minimum bending radii, multi-states capabilities and structural stability. We present geometry-based pneumatic actuators (GPAs), a design and implementation approach that introduces constraint layers with configurable CNC heat-sealed chambers. The approach achieves predictable deformation, near-zero bending radii, multi-states actuation, and enables customizable and repeatable complex actuated geometries. Mathematical modeling reveals predictable linear angle transformations and validates nonlinear torque-angle relationships across diverse configurations. We demonstrate versatility of the GPAs approach through three applications: a 49 g wrist exoskeleton reducing muscle activity by up to 51%, a 30.8 g haptic interface delivering 8 N force feedback with fast response, and a 208 g bipedal robot achieving multi-gait locomotion. GPAs establish a configurable platform for next-generation wearable robotics, haptic systems, and soft locomotion devices.
>
---
#### [new 010] Humanoid Robots as First Assistants in Endoscopic Surgery
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在验证人形机器人在手术中的可行性。通过实验，证明人形机器人可辅助完成内窥镜手术，解决其在物理和认知需求上的挑战。**

- **链接: [https://arxiv.org/pdf/2602.24156](https://arxiv.org/pdf/2602.24156)**

> **作者:** Sue Min Cho; Jan Emily Mangulabnan; Han Zhang; Zhekai Mao; Yufan He; Pengfei Guo; Daguang Xu; Gregory Hager; Masaru Ishii; Mathias Unberath
>
> **摘要:** Humanoid robots have become a focal point of technological ambition, with claims of surgical capability within years in mainstream discourse. These projections are aspirational yet lack empirical grounding. To date, no humanoid has assisted a surgeon through an actual procedure, let alone performed one. The work described here breaks this new ground. Here we report a proof of concept in which a teleoperated Unitree G1 provided endoscopic visualization while an attending otolaryngologist performed a cadaveric sphenoidectomy. The procedure was completed successfully, with stable visualization maintained throughout. Teleoperation allowed assessment of whether the humanoid form factor could meet the physical demands of surgical assistance in terms of sustenance and precision; the cognitive demands were satisfied -- for now -- by the operator. Post-procedure analysis identified engineering targets for clinical translation, alongside near-term opportunities such as autonomous diagnostic scoping. This work establishes form-factor feasibility for humanoid surgical assistance while identifying challenges for continued development.
>
---
#### [new 011] OmniXtreme: Breaking the Generality Barrier in High-Dynamic Humanoid Control
- **分类: cs.RO**

- **简介: 该论文属于高动态人形机器人控制任务，旨在解决运动跟踪精度与泛化能力之间的矛盾。通过提出OmniXtreme框架，实现高效运动学习与物理执行优化的分离，提升真实场景下的控制性能。**

- **链接: [https://arxiv.org/pdf/2602.23843](https://arxiv.org/pdf/2602.23843)**

> **作者:** Yunshen Wang; Shaohang Zhu; Peiyuan Zhi; Yuhan Li; Jiaxin Li; Yong-Lu Li; Yuchen Xiao; Xingxing Wang; Baoxiong Jia; Siyuan Huang
>
> **摘要:** High-fidelity motion tracking serves as the ultimate litmus test for generalizable, human-level motor skills. However, current policies often hit a "generality barrier": as motion libraries scale in diversity, tracking fidelity inevitably collapses - especially for real-world deployment of high-dynamic motions. We identify this failure as the result of two compounding factors: the learning bottleneck in scaling multi-motion optimization and the physical executability constraints that arise in real-world actuation. To overcome these challenges, we introduce OmniXtreme, a scalable framework that decouples general motor skill learning from sim-to-real physical skill refinement. Our approach uses a flow-matching policy with high-capacity architectures to scale representation capacity without interference-intensive multi-motion RL optimization, followed by an actuation-aware refinement phase that ensures robust performance on physical hardware. Extensive experiments demonstrate that OmniXtreme maintains high-fidelity tracking across diverse, high-difficulty datasets. On real robots, the unified policy successfully executes multiple extreme motions, effectively breaking the long-standing fidelity-scalability trade-off in high-dynamic humanoid control.
>
---
#### [new 012] Cybersecurity of Teleoperated Quadruped Robots: A Systematic Survey of Vulnerabilities, Threats, and Open Defense Gaps
- **分类: cs.RO; cs.CR; eess.SY**

- **简介: 该论文属于网络安全领域，针对遥控四足机器人的安全问题进行系统综述，分析漏洞、威胁及防御不足，提出攻击分类和防护建议。**

- **链接: [https://arxiv.org/pdf/2602.23404](https://arxiv.org/pdf/2602.23404)**

> **作者:** Mohammad Sabouri
>
> **备注:** survey paper; 23 tables; 9 figures; 132 references
>
> **摘要:** Teleoperated quadruped robots are increasingly deployed in safety-critical missions -- industrial inspection, military reconnaissance, and emergency response -- yet the security of their communication and control infrastructure remains insufficiently characterized. Quadrupeds present distinct security challenges arising from dynamic stability constraints, gait-dependent vulnerability windows, substantial kinetic energy, and elevated operator cognitive load. This survey synthesizes peer-reviewed literature and vulnerability disclosures (2019--2025) to provide comprehensive analysis of cybersecurity threats, consequences, and countermeasures for teleoperated quadruped systems. We contribute: (i) a six-layer attack taxonomy spanning perception manipulation, VR/AR operator targeting, communication disruption, control signal attacks, localization spoofing, and network intrusion; (ii) systematic attack-to-consequence mapping with timing characterization; (iii) Technology Readiness Level classification exposing critical maturity gaps between field-deployed communication protections (TRL 7--9) and experimental perception/operator-layer defenses (TRL 3--5); (iv) comparative security analysis of six commercial platforms; (v) pragmatic deployment guidance stratified by implementation timeline; and (vi) eight prioritized research gaps with implementation roadmaps. Limitations: Platform assessments rely on publicly available information. Attack success rates derive from cited studies under controlled conditions and require domain-specific validation.
>
---
#### [new 013] VCA: Vision-Click-Action Framework for Precise Manipulation of Segmented Objects in Target Ambiguous Environments
- **分类: cs.RO**

- **简介: 该论文提出VCA框架，解决视觉-语言-动作模型在目标模糊环境中的对象精确操作问题。通过点击式视觉交互替代文本指令，提升操作精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.23583](https://arxiv.org/pdf/2602.23583)**

> **作者:** Donggeon Kim; Seungwon Jan; Hyeonjun Park; Daegyu Lim
>
> **备注:** Submitted to UR 2026
>
> **摘要:** The reliance on language in Vision-Language-Action (VLA) models introduces ambiguity, cognitive overhead, and difficulties in precise object identification and sequential task execution, particularly in environments with multiple visually similar objects. To address these limitations, we propose Vision-Click-Action (VCA), a framework that replaces verbose textual commands with direct, click-based visual interaction using pretrained segmentation models. By allowing operators to specify target objects clearly through visual selection in the robot's 2D camera view, VCA reduces interpretation errors, lowers cognitive load, and provides a practical and scalable alternative to language-driven interfaces for real-world robotic manipulation. Experimental results validate that the proposed VCA framework achieves effective instance-level manipulation of specified target objects. Experiment videos are available at this https URL.
>
---
#### [new 014] KEEP: A KV-Cache-Centric Memory Management System for Efficient Embodied Planning
- **分类: cs.RO; cs.AI; cs.SE**

- **简介: 该论文属于 embodied planning 任务，解决 LLMs 中记忆管理效率低的问题。提出 KEEP 系统，通过优化 KV 缓存管理提升推理速度与效果。**

- **链接: [https://arxiv.org/pdf/2602.23592](https://arxiv.org/pdf/2602.23592)**

> **作者:** Zebin Yang; Tong Xie; Baotong Lu; Shaoshan Liu; Bo Yu; Meng Li
>
> **备注:** DAC 2026
>
> **摘要:** Memory-augmented Large Language Models (LLMs) have demonstrated remarkable capability for complex and long-horizon embodied planning. By keeping track of past experiences and environmental states, memory enables LLMs to maintain a global view, thereby avoiding repetitive exploration. However, existing approaches often store the memory as raw text, leading to excessively long prompts and high prefill latency. While it is possible to store and reuse the KV caches, the efficiency benefits are greatly undermined due to frequent KV cache updates. In this paper, we propose KEEP, a KV-cache-centric memory management system for efficient embodied planning. KEEP features 3 key innovations: (1) a Static-Dynamic Memory Construction algorithm that reduces KV cache recomputation by mixed-granularity memory group; (2) a Multi-hop Memory Re-computation algorithm that dynamically identifies important cross-attention among different memory groups and reconstructs memory interactions iteratively; (3) a Layer-balanced Memory Loading that eliminates unbalanced KV cache loading and cross-attention computation across different layers. Extensive experimental results have demonstrated that KEEP achieves 2.68x speedup with negligible accuracy loss compared with text-based memory methods on ALFRED dataset. Compared with the KV re-computation method CacheBlend (EuroSys'25), KEEP shows 4.13% success rate improvement and 1.90x time-to-first-token (TTFT) reduction. Our code is available on this https URL.
>
---
#### [new 015] SAGE-LLM: Towards Safe and Generalizable LLM Controller with Fuzzy-CBF Verification and Graph-Structured Knowledge Retrieval for UAV Decision
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机决策任务，解决LLM在安全性和泛化能力上的不足。提出SAGE-LLM框架，结合模糊CBF验证和图结构知识检索，提升无人机控制的安全与泛化性能。**

- **链接: [https://arxiv.org/pdf/2602.23719](https://arxiv.org/pdf/2602.23719)**

> **作者:** Wenzhe Zhao; Yang Zhao; Ganchao Liu; Zhiyu Jiang; Dandan Ma; Zihao Li; Xuelong Li
>
> **摘要:** In UAV dynamic decision, complex and variable hazardous factors pose severe challenges to the generalization capability of algorithms. Despite offering semantic understanding and scene generalization, Large Language Models (LLM) lack domain-specific UAV control knowledge and formal safety assurances, restricting their direct applicability. To bridge this gap, this paper proposes a train-free two-layer decision architecture based on LLMs, integrating high-level safety planning with low-level precise control. The framework introduces three key contributions: 1) A fuzzy Control Barrier Function verification mechanism for semantically-augmented actions, providing provable safety certification for LLM outputs. 2) A star-hierarchical graph-based retrieval-augmented generation system, enabling efficient, elastic, and interpretable scene adaptation. 3) Systematic experimental validation in pursuit-evasion scenarios with unknown obstacles and emergent threats, demonstrating that our SAGE-LLM maintains performance while significantly enhancing safety and generalization without online training. The proposed framework demonstrates strong extensibility, suggesting its potential for generalization to broader embodied intelligence systems and safety-critical control domains.
>
---
#### [new 016] TaCarla: A comprehensive benchmarking dataset for end-to-end autonomous driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出TaCarla数据集，用于端到端自动驾驶研究。解决现有数据集不完整、缺乏多样性及闭环评估的问题，包含多种感知与规划任务的数据，提升模型训练与评估效果。**

- **链接: [https://arxiv.org/pdf/2602.23499](https://arxiv.org/pdf/2602.23499)**

> **作者:** Tugrul Gorgulu; Atakan Dag; M. Esat Kalfaoglu; Halil Ibrahim Kuru; Baris Can Cam; Ozsel Kilinc
>
> **摘要:** Collecting a high-quality dataset is a critical task that demands meticulous attention to detail, as overlooking certain aspects can render the entire dataset unusable. Autonomous driving challenges remain a prominent area of research, requiring further exploration to enhance the perception and planning performance of vehicles. However, existing datasets are often incomplete. For instance, datasets that include perception information generally lack planning data, while planning datasets typically consist of extensive driving sequences where the ego vehicle predominantly drives forward, offering limited behavioral diversity. In addition, many real datasets struggle to evaluate their models, especially for planning tasks, since they lack a proper closed-loop evaluation setup. The CARLA Leaderboard 2.0 challenge, which provides a diverse set of scenarios to address the long-tail problem in autonomous driving, has emerged as a valuable alternative platform for developing perception and planning models in both open-loop and closed-loop evaluation setups. Nevertheless, existing datasets collected on this platform present certain limitations. Some datasets appear to be tailored primarily for limited sensor configuration, with particular sensor configurations. To support end-to-end autonomous driving research, we have collected a new dataset comprising over 2.85 million frames using the CARLA simulation environment for the diverse Leaderboard 2.0 challenge scenarios. Our dataset is designed not only for planning tasks but also supports dynamic object detection, lane divider detection, centerline detection, traffic light recognition, prediction tasks and visual language action models . Furthermore, we demonstrate its versatility by training various models using our dataset. Moreover, we also provide numerical rarity scores to understand how rarely the current state occurs in the dataset.
>
---
#### [new 017] Evaluating Accuracy of Vine Robot Shape Sensing with Distributed Inertial Measurement Units
- **分类: cs.RO**

- **简介: 该论文属于机器人形状感知任务，旨在评估分布式IMU在藤蔓机器人体型传感中的准确性。研究解决了主动转向、长度变化和传感器间距对精度影响的问题，通过实验分析了误差来源及优化方向。**

- **链接: [https://arxiv.org/pdf/2602.24202](https://arxiv.org/pdf/2602.24202)**

> **作者:** Alexis E. Laudenslager; Antonio Alvarez Valdivia; Nathaniel Hanson; Margaret McGuinness
>
> **摘要:** Soft, tip-extending vine robots are well suited for navigating tight, debris-filled environments, making them ideal for urban search and rescue. Sensing the full shape of a vine robot's body is helpful both for localizing information from other sensors placed along the robot body and for determining the robot's configuration within the space being explored. Prior approaches have localized vine robot tips using a single inertial measurement unit (IMU) combined with force sensing or length estimation, while one method demonstrated full-body shape sensing using distributed IMUs on a passively steered robot in controlled maze environments. However, the accuracy of distributed IMU-based shape sensing under active steering, varying robot lengths, and different sensor spacings has not been systematically quantified. In this work, we experimentally evaluate the accuracy of vine robot shape sensing using distributed IMUs along the robot body. We quantify IMU drift, measuring an average orientation drift rate of 1.33 degrees/min across 15 sensors. For passive steering, mean tip position error was 11% of robot length. For active steering, mean tip position error increased to 16%. During growth experiments across lengths from 30-175 cm, mean tip error was 8%, with a positive trend with increasing length. We also analyze the influence of sensor spacing and observe that intermediate spacings can minimize error for single-curvature shapes. These results demonstrate the feasibility of distributed IMU-based shape sensing for vine robots while highlighting key limitations and opportunities for improved modeling and algorithmic integration for field deployment.
>
---
#### [new 018] Physics-Embedded Neural ODEs for Learning Antagonistic Pneumatic Artificial Muscle Dynamics
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人动力学建模任务，旨在解决拮抗气动人工肌肉的非线性、迟滞动态建模与控制问题。通过物理嵌入的神经微分方程框架，实现高精度运动与刚度控制。**

- **链接: [https://arxiv.org/pdf/2602.23670](https://arxiv.org/pdf/2602.23670)**

> **作者:** Xinyao Wang; Jonathan Realmuto
>
> **摘要:** Pneumatic artificial muscles (PAMs) enable compliant actuation for soft wearable, assistive, and interactive robots. When arranged antagonistically, PAMs can provide variable impedance through co-contraction but exhibit coupled, nonlinear, and hysteretic dynamics that challenge modeling and control. This paper presents a hybrid neural ordinary differential equation (Neural ODE) framework that embeds physical structure into a learned model of antagonistic PAM dynamics. The formulation combines parametric joint mechanics and pneumatic state dynamics with a neural network force component that captures antagonistic coupling and rate-dependent hysteresis. The forward model predicts joint motion and chamber pressures with a mean R$^2$ of 0.88 across 225 co-contraction conditions. An inverse formulation, derived from the learned dynamics, computes pressure commands offline for desired motion and stiffness profiles, tracked in closed loop during execution. Experimental validation demonstrates reliable stiffness control across 126-176 N/mm and consistent impedance behavior across operating velocities, in contrast to a static model, which shows degraded stiffness consistency at higher velocities.
>
---
#### [new 019] Autonomous Inspection of Power Line Insulators with UAV on an Unmapped Transmission Tower
- **分类: cs.RO**

- **简介: 该论文属于电力线路绝缘子自主巡检任务，解决无先验地图下无人机精准定位与检测绝缘子的问题。通过融合相机与LiDAR数据，提出检测与定位算法，提升巡检效率与精度。**

- **链接: [https://arxiv.org/pdf/2602.24011](https://arxiv.org/pdf/2602.24011)**

> **作者:** Václav Riss; Vít Krátký; Robert Pěnička; Martin Saska
>
> **备注:** 8 pages, 9 figues
>
> **摘要:** This paper introduces an online inspection algorithm that enables an autonomous UAV to fly around a transmission tower and obtain detailed inspection images without a prior map of the tower. Our algorithm relies on camera-LiDAR sensor fusion for online detection and localization of insulators. In particular, the algorithm is based on insulator detection using a convolutional neural network, projection of LiDAR points onto the image, and filtering them using the bounding boxes. The detection pipeline is coupled with several proposed insulator localization methods based on DBSCAN, RANSAC, and PCA algorithms. The performance of the proposed online inspection algorithm and camera-LiDAR sensor fusion pipeline is demonstrated through simulation and real-world flights. In simulation, we showed that our single-flight inspection strategy can save up to 24 % of total inspection time, compared to the two-flight strategy of scanning the tower and afterwards visiting the inspection waypoints in the optimal way. In a real-world experiment, the best performing proposed method achieves a mean horizontal and vertical localization error for the insulator of 0.16 +- 0.08 m and 0.16 +- 0.11 m, respectively. Compared to the most relevant approach, the proposed method achieves more than an order of magnitude lower variance in horizontal insulator localization error.
>
---
#### [new 020] How IMU Drift Influences Multi-Radar Inertial Odometry for Ground Robots in Subterranean Terrains
- **分类: cs.RO**

- **简介: 该论文属于机器人定位与建图任务，解决IMU漂移影响地下雷达惯性里程计的问题。通过两阶段框架提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.24192](https://arxiv.org/pdf/2602.24192)**

> **作者:** Moumita Mukherjee; Magnus Norén; Anton Koval; Avijit Banerjee; George Nikolakopoulos
>
> **备注:** Accepted in IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** Reliable radar inertial odometry (RIO) requires mitigating IMU bias drift, a challenge that intensifies in subterranean environments due to extreme temperatures and gravity-induced accelerations. Cost-effective IMUs such as the Pixhawk, when paired with FMCW TI IWR6843AOP EVM radars, suffer from drift-induced degradation compounded by sparse, noisy, and flickering radar returns, making fusion less stable than LiDAR-based odometry. Yet, LiDAR fails under smoke, dust, and aerosols, whereas FMCW radars remain compact, lightweight, cost-effective, and robust in these situations. To address these challenges, we propose a two-stage MRIO framework that combines an IMU bias estimator for resilient localization and mapping in GPS-denied subterranean environments affected by smoke. Radar-based ego-velocity estimation is formulated through a least-squares approach and incorporated into an EKF for online IMU bias correction; the corrected IMU accelerations are fused with heterogeneous measurements from multiple radars and an IMU to refine odometry. The proposed framework further supports radar-only mapping by exploiting the robot's estimated translational and rotational displacements. In subterranean field trials, MRIO delivers robust localization and mapping, outperforming EKF-RIO. It maintains accuracy across cost-efficient FMCW radar setups and different IMUs, showing resilience with Pixhawk and higher-grade units such as VectorNav. The implementation will be provided as an open-source resource to the community (code available at this https URL
>
---
#### [new 021] ABPolicy: Asynchronous B-Spline Flow Policy for Real-Time and Smooth Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决同步推理导致的运动不流畅问题。提出ABPolicy，通过异步流匹配和B样条控制点空间实现平滑、实时的控制。**

- **链接: [https://arxiv.org/pdf/2602.23901](https://arxiv.org/pdf/2602.23901)**

> **作者:** Fan Yang; Peiguang Jing; Kaihua Qu; Ningyuan Zhao; Yuting Su
>
> **摘要:** Robotic manipulation requires policies that are smooth and responsive to evolving observations. However, synchronous inference in the raw action space introduces several challenges, including intra-chunk jitter, inter-chunk discontinuities, and stop-and-go execution. These issues undermine a policy's smoothness and its responsiveness to environmental changes. We propose ABPolicy, an asynchronous flow-matching policy that operates in a B-spline control-point action space. First, the B-spline representation ensures intra-chunk smoothness. Second, we introduce bidirectional action prediction coupled with refitting optimization to enforce inter-chunk continuity. Finally, by leveraging asynchronous inference, ABPolicy delivers real-time, continuous updates. We evaluate ABPolicy across seven tasks encompassing both static settings and dynamic settings with moving objects. Empirical results indicate that ABPolicy reduces trajectory jerk, leading to smoother motion and improved performance. Project website: this https URL.
>
---
#### [new 022] Interpretable Multimodal Gesture Recognition for Drone and Mobile Robot Teleoperation via Log-Likelihood Ratio Fusion
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于手势识别任务，旨在解决无人机和移动机器人远程操作中的手势识别问题。通过融合惯性数据和电容传感信号，提升识别性能与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.23694](https://arxiv.org/pdf/2602.23694)**

> **作者:** Seungyeol Baek; Jaspreet Singh; Lala Shakti Swarup Ray; Hymalai Bello; Paul Lukowicz; Sungho Suh
>
> **摘要:** Human operators are still frequently exposed to hazardous environments such as disaster zones and industrial facilities, where intuitive and reliable teleoperation of mobile robots and Unmanned Aerial Vehicles (UAVs) is essential. In this context, hands-free teleoperation enhances operator mobility and situational awareness, thereby improving safety in hazardous environments. While vision-based gesture recognition has been explored as one method for hands-free teleoperation, its performance often deteriorates under occlusions, lighting variations, and cluttered backgrounds, limiting its applicability in real-world operations. To overcome these limitations, we propose a multimodal gesture recognition framework that integrates inertial data (accelerometer, gyroscope, and orientation) from Apple Watches on both wrists with capacitive sensing signals from custom gloves. We design a late fusion strategy based on the log-likelihood ratio (LLR), which not only enhances recognition performance but also provides interpretability by quantifying modality-specific contributions. To support this research, we introduce a new dataset of 20 distinct gestures inspired by aircraft marshalling signals, comprising synchronized RGB video, IMU, and capacitive sensor data. Experimental results demonstrate that our framework achieves performance comparable to a state-of-the-art vision-based baseline while significantly reducing computational cost, model size, and training time, making it well suited for real-time robot control. We therefore underscore the potential of sensor-based multimodal fusion as a robust and interpretable solution for gesture-driven mobile robot and drone teleoperation.
>
---
#### [new 023] Enhancing Vision-Language Navigation with Multimodal Event Knowledge from Real-World Indoor Tour Videos
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决长距离推理和模糊指令问题。通过构建多模态事件知识图谱并融合到导航模型中，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2602.23937](https://arxiv.org/pdf/2602.23937)**

> **作者:** Haoxuan Xu; Tianfu Li; Wenbo Chen; Yi Liu; Xingxing Zuo; Yaoxian Song; Haoang Li
>
> **摘要:** Vision-Language Navigation (VLN) agents often struggle with long-horizon reasoning in unseen environments, particularly when facing ambiguous, coarse-grained instructions. While recent advances use knowledge graph to enhance reasoning, the potential of multimodal event knowledge inspired by human episodic memory remains underexplored. In this work, we propose an event-centric knowledge enhancement strategy for automated process knowledge mining and feature fusion to solve coarse-grained instruction and long-horizon reasoning in VLN task. First, we construct YE-KG, the first large-scale multimodal spatiotemporal knowledge graph, with over 86k nodes and 83k edges, derived from real-world indoor videos. By leveraging multimodal large language models (i.e., LLaVa, GPT4), we extract unstructured video streams into structured semantic-action-effect events to serve as explicit episodic memory. Second, we introduce STE-VLN, which integrates the above graph into VLN models via a Coarse-to-Fine Hierarchical Retrieval mechanism. This allows agents to retrieve causal event sequences and dynamically fuse them with egocentric visual observations. Experiments on REVERIE, R2R, and R2R-CE benchmarks demonstrate the efficiency of our event-centric strategy, outperforming state-of-the-art approaches across diverse action spaces. Our data and code are available on the project website this https URL.
>
---
#### [new 024] Robust Skills, Brittle Grounding: Diagnosing Restricted Generalization in Vision-Language Action Policies via Multi-Object Picking
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言动作策略的泛化能力，旨在解决指令到物体的准确映射问题。通过多物体抓取实验，发现策略更依赖操作技能而非指令理解，建议改进评估指标。**

- **链接: [https://arxiv.org/pdf/2602.24143](https://arxiv.org/pdf/2602.24143)**

> **作者:** David Emukpere; Romain Deffayet; Jean-Michel Renders
>
> **摘要:** Vision-language action (VLA) policies often report strong manipulation benchmark performance with relatively few demonstrations, but it remains unclear whether this reflects robust language-to-object grounding or reliance on object--location correlations that do not transfer beyond the training distribution. We present a controlled multi-object picking study that progressively increases object placement variability up to full workspace randomization and evaluates held-out object--location pairings that break familiar associations without increasing spatial difficulty. Across these stress tests and data scaling, we find that for representative VLA policies, including SmolVLA and $\pi_{0.5}$, execution of the manipulation primitive remains substantially more reliable than instruction-conditioned task success in harder regimes, suggesting that manipulation skill acquisition is decoupled from instruction following. We recommend augmenting manipulation benchmarks with task ladders and decomposed metrics that separately measure primitive execution and instruction-conditioned success to better diagnose instruction-grounded generalization.
>
---
#### [new 025] Tilt-X: Enabling Compliant Aerial Manipulation through a Tiltable-Extensible Continuum Manipulator
- **分类: cs.RO**

- **简介: 该论文属于航空机械臂任务，旨在解决传统 aerial manipulator 操作范围小、受旋翼下洗影响等问题。提出 Tilt-X 系统，集成倾斜、伸缩和缆控结构，扩展操作空间并提升稳定性。**

- **链接: [https://arxiv.org/pdf/2602.23576](https://arxiv.org/pdf/2602.23576)**

> **作者:** Anuraj Uthayasooriyan; Krishna Manaswi Digumarti; Jack Breward; Fernando Vanegas; Julian Galvez-Serna; Felipe Gonzalez
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Aerial manipulators extend the reach and manipulation capabilities of uncrewed multirotor aerial vehicles for inspection, agriculture, sampling, and delivery. Continuum arm aerial manipulation systems offer lightweight, dexterous, and compliant interaction opportunities. Existing designs allow manipulation only below the UAV which restricts their deployability in multiple directions and through clutter. They are also sensitive to propeller downwash. Addressing these limitations, we present Tilt-X, a continuum arm aerial manipulator that integrates a tilting mechanism, a telescopic stage, and a cable-driven continuum section. We present its design and kinematic model and validate it through flight demonstrations. Tilt-X enables a volumetric workspace with up to 75 mm extension and planar orientations between 0$^\circ$ to 90$^\circ$. Experiments comparing end effector pose with and without downwash quantitatively measure its accuracy, providing critical evidence to guide the design and control of reliable aerial manipulators. Results show stabilisation of end effector pose as the manipulator extends out of the propeller influence zone.
>
---
#### [new 026] Demystifying Action Space Design for Robotic Manipulation Policies
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控策略学习任务，旨在解决动作空间设计对策略学习的影响问题。通过大量实验，分析了不同动作表示方式的优劣，提出合理设计动作空间可提升性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.23408](https://arxiv.org/pdf/2602.23408)**

> **作者:** Yuchun Feng; Jinliang Zheng; Zhihao Wang; Dongxiu Liu; Jianxiong Li; Jiangmiao Pang; Tai Wang; Xianyuan Zhan
>
> **摘要:** The specification of the action space plays a pivotal role in imitation-based robotic manipulation policy learning, fundamentally shaping the optimization landscape of policy learning. While recent advances have focused heavily on scaling training data and model capacity, the choice of action space remains guided by ad-hoc heuristics or legacy designs, leading to an ambiguous understanding of robotic policy design philosophies. To address this ambiguity, we conducted a large-scale and systematic empirical study, confirming that the action space does have significant and complex impacts on robotic policy learning. We dissect the action design space along temporal and spatial axes, facilitating a structured analysis of how these choices govern both policy learnability and control stability. Based on 13,000+ real-world rollouts on a bimanual robot and evaluation on 500+ trained models over four scenarios, we examine the trade-offs between absolute vs. delta representations, and joint-space vs. task-space parameterizations. Our large-scale results suggest that properly designing the policy to predict delta actions consistently improves performance, while joint-space and task-space representations offer complementary strengths, favoring control stability and generalization, respectively.
>
---
#### [new 027] TSC: Topology-Conditioned Stackelberg Coordination for Multi-Agent Reinforcement Learning in Interactive Driving
- **分类: cs.RO**

- **简介: 该论文属于多智能体强化学习任务，解决密集交通中安全高效协作问题。提出TSC框架，通过拓扑条件协调实现去中心化决策，提升安全性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.23896](https://arxiv.org/pdf/2602.23896)**

> **作者:** Xiaotong Zhang; Gang Xiong; Yuanjing Wang; Siyu Teng; Alois Knoll; Long Chen
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Safe and efficient autonomous driving in dense traffic is fundamentally a decentralized multi-agent coordination problem, where interactions at conflict points such as merging and weaving must be resolved reliably under partial observability. With only local and incomplete cues, interaction patterns can change rapidly, often causing unstable behaviors such as oscillatory yielding or unsafe commitments. Existing multi-agent reinforcement learning (MARL) approaches either adopt synchronous decision-making, which exacerbate non-stationarity, or depend on centralized sequencing mechanisms that scale poorly as traffic density increases. To address these limitations, we propose Topology-conditioned Stackelberg Coordination (TSC), a learning framework for decentralized interactive driving under communication-free execution, which extracts a time-varying directed priority graph from braid-inspired weaving relations between trajectories, thereby defining local leader-follower dependencies without constructing a global order of play. Conditioned on this graph, TSC endogenously factorizes dense interactions into graph-local Stackelberg subgames and, under centralized training and decentralized execution (CTDE), learns a sequential coordination policy that anticipates leaders via action prediction and trains followers through action-conditioned value learning to approximate local best responses, improving training stability and safety in dense traffic. Experiments across four dense traffic scenarios show that TSC achieves superior performance over representative MARL baselines across key metrics, most notably reducing collisions while maintaining competitive traffic efficiency and control smoothness.
>
---
#### [new 028] Planning from Observation and Interaction
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，解决无先验知识的现实世界任务学习问题。通过基于规划的逆强化学习算法，从观察和交互中学习世界模型，实现高效、成功的在线迁移学习。**

- **链接: [https://arxiv.org/pdf/2602.24121](https://arxiv.org/pdf/2602.24121)**

> **作者:** Tyler Han; Siyang Shen; Rohan Baijal; Harine Ravichandiran; Bat Nemekhbold; Kevin Huang; Sanghun Jung; Byron Boots
>
> **摘要:** Observational learning requires an agent to learn to perform a task by referencing only observations of the performed task. This work investigates the equivalent setting in real-world robot learning where access to hand-designed rewards and demonstrator actions are not assumed. To address this data-constrained setting, this work presents a planning-based Inverse Reinforcement Learning (IRL) algorithm for world modeling from observation and interaction alone. Experiments conducted entirely in the real-world demonstrate that this paradigm is effective for learning image-based manipulation tasks from scratch in under an hour, without assuming prior knowledge, pre-training, or data of any kind beyond task observations. Moreover, this work demonstrates that the learned world model representation is capable of online transfer learning in the real-world from scratch. In comparison to existing approaches, including IRL, RL, and Behavior Cloning (BC), which have more restrictive assumptions, the proposed approach demonstrates significantly greater sample efficiency and success rates, enabling a practical path forward for online world modeling and planning from observation and interaction. Videos and more at: this https URL.
>
---
#### [new 029] Hybrid Offline-Online Reinforcement Learning for Sensorless, High-Precision Force Regulation in Surgical Robotic Grasping
- **分类: cs.RO**

- **简介: 该论文属于手术机器人控制任务，解决无传感高精度夹持力调节问题。通过结合物理建模与混合强化学习，实现无需末端传感的力控制，提升手术器械操作精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.23870](https://arxiv.org/pdf/2602.23870)**

> **作者:** Edoardo Fazzari; Omar Mohamed; Khalfan Hableel; Hamdan Alhadhrami; Cesare Stefanini
>
> **摘要:** Precise grasp force regulation in tendon-driven surgical instruments is fundamentally limited by nonlinear coupling between motor dynamics, transmission compliance, friction, and distal mechanics. Existing solutions typically rely on distal force sensing or analytical compensation, increasing hardware complexity or degrading performance under dynamic motion. We present a sensorless control framework that combines physics-consistent modeling and hybrid reinforcement learning to achieve high-precision distal force regulation in a proximally actuated surgical end-effector. We develop a first-principles digital twin of the da Vinci Xi grasping mechanism that captures coupled electrical, transmission, and jaw dynamics within a unified differential-algebraic formulation. To safely learn control policies in this stiff and highly nonlinear system, we introduce a three-stage pipeline:(i)a receding-horizon CMA-ES oracle that generates dynamically feasible expert trajectories,(ii)fully offline policy learning via Implicit Q-Learning to ensure stable initialization without unsafe exploration, and (iii)online refinement using TD3 for adaptation to on-policy dynamics. The resulting policy directly maps proximal measurements to motor voltages and requires no distal sensing. In simulation, the controller maintains grasp force within 1% of the desired reference during multi-harmonic jaw motion. Hardware experiments demonstrate average force errors below 4% across diverse trajectories, validating sim-to-real transfer. The learned policy contains approximately 71k param and executes at kH rates, enabling real-time deployment. These results demonstrate that high-fidelity modeling combined with structured offline-online RL can recover precise distal force behavior without additional sensing, offering a scalable and mechanically compatible solution for surgical robotic manipulation.
>
---
#### [new 030] Acceleration-Based Control of Fixed-Wing UAVs for Guidance Applications
- **分类: cs.RO**

- **简介: 该论文属于无人机制导控制任务，解决固定翼无人机无法直接执行加速度指令的问题。通过设计外环控制框架，将加速度指令转化为可执行的姿态和推力命令，实现精确跟踪。**

- **链接: [https://arxiv.org/pdf/2602.23821](https://arxiv.org/pdf/2602.23821)**

> **作者:** Jixiang Wang; Siyuan Yang; Ziyi Wu; Siqi Wei; Ashay Wakode; Agata Barcis; Hung Nguyen; Shaoming He
>
> **摘要:** Acceleration-commanded guidance laws (e.g., proportional navigation) are attractive for high-level decision making, but their direct deployment on fixed-wing UAVs is challenging because accelerations are not directly actuated and must be realized through attitude and thrust under flight-envelope constraints. This paper presents an acceleration-level outer-loop control framework that converts commanded tangential and normal accelerations into executable body-rate and normalized thrust commands compatible with mainstream autopilots (e.g., PX4/APM). For the normal channel, we derive an engineering mapping from the desired normal acceleration to roll- and pitch-rate commands that regulate the direction and magnitude of the lift vector under small-angle assumptions. For the tangential channel, we introduce an energy-based formulation inspired by total energy control and identify an empirical thrust-energy acceleration relationship directly from flight data, avoiding explicit propulsion modeling or thrust bench calibration. We further discuss priority handling between normal and tangential accelerations under saturation and non-level maneuvers. Extensive real-flight experiments on a VTOL fixed-wing platform demonstrate accurate acceleration tracking and enable practical implementation of proportional navigation using only body-rate and normalized thrust interfaces.
>
---
#### [new 031] Teleoperated Omni-directional Dual Arm Mobile Manipulation Robotic System with Shared Control for Retail Store
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决零售环境中机器人自主操作困难的问题。通过设计双臂移动机器人和共享控制的遥操作方法，提升机器人在复杂场景中的适应能力。**

- **链接: [https://arxiv.org/pdf/2602.23923](https://arxiv.org/pdf/2602.23923)**

> **作者:** Rolif Lima; Somdeb Saha; Nijil George; Vismay Vakharia; Shubham Parab; Sahil Gaonkar; Vighnesh Vatsal; Kaushik Das
>
> **备注:** This work has been accepted for publication in the Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics (SMC 2024). $©$ IEEE. The final version is available via IEEE Xplore
>
> **摘要:** The swiftly expanding retail sector is increasingly adopting autonomous mobile robots empowered by artificial intelligence and machine learning algorithms to gain an edge in the competitive market. However, these autonomous robots encounter challenges in adapting to the dynamic nature of retail products, often struggling to operate autonomously in novel situations. In this study, we introduce an omni-directional dual-arm mobile robot specifically tailored for use in retail environments. Additionally, we propose a tele-operation method that enables shared control between the robot and a human operator. This approach utilizes a Virtual Reality (VR) motion capture system to capture the operator's commands, which are then transmitted to the robot located remotely in a retail setting. Furthermore, the robot is equipped with heterogeneous grippers on both manipulators, facilitating the handling of a wide range of items. We validate the efficacy of the proposed system through testing in a mockup of retail environment, demonstrating its ability to manipulate various commonly encountered retail items using both single and dual-arm coordinated manipulation techniques.
>
---
#### [new 032] Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles
- **分类: cs.RO**

- **简介: 该论文属于无人机竞速任务，解决复杂障碍环境下的自主飞行问题。通过视觉强化学习框架，提升无人机避障与穿越门的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.24030](https://arxiv.org/pdf/2602.24030)**

> **作者:** Fangyu Sun; Fanxing Li; Yu Hu; Linzuo Zhang; Yueqian Liu; Wenxian Yu; Danping Zou
>
> **摘要:** Autonomous drone racing has attracted increasing interest as a research topic for exploring the limits of agile flight. However, existing studies primarily focus on obstacle-free racetracks, while the perception and dynamic challenges introduced by obstacles remain underexplored, often resulting in low success rates and limited robustness in real-world flight. To this end, we propose a novel vision-based curriculum reinforcement learning framework for training a robust controller capable of addressing unseen obstacles in drone racing. We combine multi-stage cu rriculum learning, domain randomization, and a multi-scene updating strategy to address the conflicting challenges of obstacle avoidance and gate traversal. Our end-to-end control policy is implemented as a single network, allowing high-speed flight of quadrotors in environments with variable obstacles. Both hardware-in-the-loop and real-world experiments demonstrate that our method achieves faster lap times and higher success rates than existing approaches, effectively advancing drone racing in obstacle-rich environments. The video and code are available at: this https URL.
>
---
#### [new 033] FAVLA: A Force-Adaptive Fast-Slow VLA model for Contact-Rich Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出FAVLA模型，解决接触丰富操作中力反馈响应延迟问题。通过分离慢速感知与快速控制，提升机器人操作的反应速度和成功率。**

- **链接: [https://arxiv.org/pdf/2602.23648](https://arxiv.org/pdf/2602.23648)**

> **作者:** Yao Li; Peiyuan Tang; Wuyang Zhang; Chengyang Zhu; Yifan Duan; Weikai Shi; Xiaodong Zhang; Zijiang Yang; Jianmin Ji; Yanyong Zhang
>
> **摘要:** Force/torque feedback can substantially improve Vision-Language-Action (VLA) models on contact-rich manipulation, but most existing approaches fuse all modalities at a single operating frequency. This design ignores the mismatched sampling rates of real robot sensors, forcing downsampling of the high-frequency contact cues needed for reactive correction. Combined with common VLM-action-expert (AE) pipelines that execute action chunks largely open loop between expensive VLM updates, unified-frequency fusion often yields delayed responses to impacts, stick-slip, and force spikes. We propose FAVLA, a force-adaptive fast-slow VLA that decouples slow perception planning from fast contact-aware control. FAVLA runs a slow VLM at a fixed low frequency to encode modalities to produce latent representations and to predict near-future force variation. A fast AE then executes at a variable high frequency, conditioning on the latest force sequence data to generate reactive actions. We further introduce a force adapter that injects high-frequency force features into multiple AE layers, and adaptively schedules the AE's execution frequency based on the VLM's predicted force variation. Extensive experiments on contact-rich tasks demonstrate that FAVLA significantly outperforms baselines, achieving superior reactivity and success rates, especially with a smaller contact force during manipulation.
>
---
#### [new 034] StemVLA:An Open-Source Vision-Language-Action Model with Future 3D Spatial Geometry Knowledge and 4D Historical Representation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出StemVLA，解决机器人操作中的空间推理与长期决策问题，通过融合3D未来空间知识和4D历史时空表示，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2602.23721](https://arxiv.org/pdf/2602.23721)**

> **作者:** Jiasong Xiao; Yutao She; Kai Li; Yuyang Sha; Ziang Cheng; Ziang Tong
>
> **备注:** Preprint
>
> **摘要:** Vision-language-action (VLA) models integrate visual observations and language instructions to predict robot actions, demonstrating promising generalization in manipulation tasks. However, most existing approaches primarily rely on direct mappings from 2D visual inputs to action sequences, without explicitly modeling the underlying 3D spatial structure or temporal world dynamics. Such representations may limit spatial reasoning and long-horizon decision-making in dynamic environments. To address this limitation, we propose StemVLA, a novel framework that explicitly incorporates both future-oriented 3D spatial knowledge and historical 4D spatiotemporal representations into action prediction. First, instead of relying solely on observed images, StemVLA forecasts structured 3D future spatial-geometric world knowledge, enabling the model to anticipate upcoming scene geometry and object configurations. Second, to capture temporal consistency and motion dynamics, we feed historical image frames into a pretrained video-geometry transformer backbone to extract implicit 3D world representations, and further aggregate them across time using a temporal attention module, termed VideoFormer [20], forming a unified 4D historical spatiotemporal representation. By jointly modeling 2D observations, predicted 3D future structure, and aggregated 4D temporal dynamics, StemVLA enables more comprehensive world understanding for robot manipulation. Extensive experiments in simulation demonstrate that StemVLA significantly improves long-horizon task success and achieves state-of-the-art performance on the CALVIN ABC-D benchmark [46], achieving an average sequence length of XXX.
>
---
#### [new 035] SafeGen-LLM: Enhancing Safety Generalization in Task Planning for Robotic Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人任务规划领域，解决安全性和泛化能力不足的问题。提出SafeGen-LLM模型，通过两阶段训练提升任务计划的安全性与跨领域泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.24235](https://arxiv.org/pdf/2602.24235)**

> **作者:** Jialiang Fan; Weizhe Xu; Mengyu Liu; Oleg Sokolsky; Insup Lee; Fangxin Kong
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Safety-critical task planning in robotic systems remains challenging: classical planners suffer from poor scalability, Reinforcement Learning (RL)-based methods generalize poorly, and base Large Language Models (LLMs) cannot guarantee safety. To address this gap, we propose safety-generalizable large language models, named SafeGen-LLM. SafeGen-LLM can not only enhance the safety satisfaction of task plans but also generalize well to novel safety properties in various domains. We first construct a multi-domain Planning Domain Definition Language 3 (PDDL3) benchmark with explicit safety constraints. Then, we introduce a two-stage post-training framework: Supervised Fine-Tuning (SFT) on a constraint-compliant planning dataset to learn planning syntax and semantics, and Group Relative Policy Optimization (GRPO) guided by fine-grained reward machines derived from formal verification to enforce safety alignment and by curriculum learning to better handle complex tasks. Extensive experiments show that SafeGen-LLM achieves strong safety generalization and outperforms frontier proprietary baselines across multi-domain planning tasks and multiple input formats (e.g., PDDLs and natural language).
>
---
#### [new 036] Refining Almost-Safe Value Functions on the Fly
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全控制任务，解决复杂系统中安全值函数设计难题。通过引入refineCBF和HJ-Patch方法，实现在线实时适应，提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2602.23478](https://arxiv.org/pdf/2602.23478)**

> **作者:** Sander Tonkens; Sosuke Kojima; Chenhao Liu; Judy Masri; Sylvia Herbert
>
> **摘要:** Control Barrier Functions (CBFs) are a powerful tool for ensuring robotic safety, but designing or learning valid CBFs for complex systems is a significant challenge. While Hamilton-Jacobi Reachability provides a formal method for synthesizing safe value functions, it scales poorly and is typically performed offline, limiting its applicability in dynamic environments. This paper bridges the gap between offline synthesis and online adaptation. We introduce refineCBF for refining an approximate CBF - whether analytically derived, learned, or even unsafe - via warm-started HJ reachability. We then present its computationally efficient successor, HJ-Patch, which accelerates this process through localized updates. Both methods guarantee the recovery of a safe value function and can ensure monotonic safety improvements during adaptation. Our experiments validate our framework's primary contribution: in-the-loop, real-time adaptation, in simulation (with detailed value function analysis) and on physical hardware. Our experiments on ground vehicles and quadcopters show that our framework can successfully adapt to sudden environmental changes, such as new obstacles and unmodeled wind disturbances, providing a practical path toward deploying formally guaranteed safety in real-world settings.
>
---
#### [new 037] AoE: Always-on Egocentric Human Video Collection for Embodied AI
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出AoE系统，用于低成本收集全身视角的现实交互数据，解决Embodied AI数据稀缺问题，通过手机和云边协同实现高效数据采集与处理。**

- **链接: [https://arxiv.org/pdf/2602.23893](https://arxiv.org/pdf/2602.23893)**

> **作者:** Bowen Yang; Zishuo Li; Yang Sun; Changtao Miao; Yifan Yang; Man Luo; Xiaotong Yan; Feng Jiang; Jinchuan Shi; Yankai Fu; Ning Chen; Junkai Zhao; Pengwei Wang; Guocai Yao; Shanghang Zhang; Hao Chen; Zhe Li; Kai Zhu
>
> **摘要:** Embodied foundation models require large-scale, high-quality real-world interaction data for pre-training and scaling. However, existing data collection methods suffer from high infrastructure costs, complex hardware dependencies, and limited interaction scope, making scalable expansion challenging. In fact, humans themselves are ideal physically embodied agents. Therefore, obtaining egocentric real-world interaction data from globally distributed "human agents" offers advantages of low cost and sustainability. To this end, we propose the Always-on Egocentric (AoE) data collection system, which aims to simplify hardware dependencies by leveraging humans themselves and their smartphones, enabling low-cost, highly efficient, and scene-agnostic real-world interaction data collection to address the challenge of data scarcity. Specifically, we first employ an ergonomic neck-mounted smartphone holder to enable low-barrier, large-scale egocentric data collection through a cloud-edge collaborative architecture. Second, we develop a cross-platform mobile APP that leverages on-device compute for real-time processing, while the cloud hosts automated labeling and filtering pipelines that transform raw videos into high-quality training data. Finally, the AoE system supports distributed Ego video data collection by anyone, anytime, and anywhere. We evaluate AoE on data preprocessing quality and downstream tasks, demonstrating that high-quality egocentric data significantly boosts real-world generalization.
>
---
#### [new 038] Optimization of Edge Directions and Weights for Mixed Guidance Graphs in Lifelong Multi-Agent Path Finding
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决Lifelong MAPF中的引导问题。通过优化边方向和权重，提出MGGO方法，实现更严格的路径引导。**

- **链接: [https://arxiv.org/pdf/2602.23468](https://arxiv.org/pdf/2602.23468)**

> **作者:** Yulun Zhang; Varun Bhatt; Matthew C. Fontaine; Stefanos Nikolaidis; Jiaoyang Li
>
> **摘要:** Multi-Agent Path Finding (MAPF) aims to move agents from their start to goal vertices on a graph. Lifelong MAPF (LMAPF) continuously assigns new goals to agents as they complete current ones. To guide agents' movement in LMAPF, prior works have proposed Guidance Graph Optimization (GGO) methods to optimize a guidance graph, which is a bidirected weighted graph whose directed edges represent moving and waiting actions with edge weights being action costs. Higher edge weights represent higher action costs. However, edge weights only provide soft guidance. An edge with a high weight only discourages agents from using it, instead of prohibiting agents from traversing it. In this paper, we explore the need to incorporate edge directions optimization into GGO, providing strict guidance. We generalize GGO to Mixed Guidance Graph Optimization (MGGO), presenting two MGGO methods capable of optimizing both edge weights and directions. The first optimizes edge directions and edge weights in two phases separately. The second applies Quality Diversity algorithms to optimize a neural network capable of generating edge directions and weights. We also incorporate traffic patterns relevant to edge directions into a GGO method, making it capable of generating edge-direction-aware guidance graphs.
>
---
#### [new 039] Altitude-Aware Visual Place Recognition in Top-Down View
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决高空变化下的视觉位置识别问题。通过分析地面特征密度估计高度，并生成标准查询图像，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2602.23872](https://arxiv.org/pdf/2602.23872)**

> **作者:** Xingyu Shao; Mengfan He; Chunyu Li; Liangzheng Sun; Ziyang Meng
>
> **摘要:** To address the challenge of aerial visual place recognition (VPR) problem under significant altitude variations, this study proposes an altitude-adaptive VPR approach that integrates ground feature density analysis with image classification techniques. The proposed method estimates airborne platforms' relative altitude by analyzing the density of ground features in images, then applies relative altitude-based cropping to generate canonical query images, which are subsequently used in a classification-based VPR strategy for localization. Extensive experiments across diverse terrains and altitude conditions demonstrate that the proposed approach achieves high accuracy and robustness in both altitude estimation and VPR under significant altitude changes. Compared to conventional methods relying on barometric altimeters or Time-of-Flight (ToF) sensors, this solution requires no additional hardware and offers a plug-and-play solution for downstream applications, {making it suitable for small- and medium-sized airborne platforms operating in diverse environments, including rural and urban areas.} Under significant altitude variations, incorporating our relative altitude estimation module into the VPR retrieval pipeline boosts average R@1 and R@5 by 29.85\% and 60.20\%, respectively, compared with applying VPR retrieval alone. Furthermore, compared to traditional {Monocular Metric Depth Estimation (MMDE) methods}, the proposed method reduces the mean error by 202.1 m, yielding average additional improvements of 31.4\% in R@1 and 44\% in R@5. These results demonstrate that our method establishes a robust, vision-only framework for three-dimensional visual place recognition, offering a practical and scalable solution for accurate airborne platforms localization under large altitude variations and limited sensor availability.
>
---
## 更新

#### [replaced 001] SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于社会感知导航任务，旨在解决机器人遵循社会规范的导航问题。通过构建数据集和设计训练方法，提升导航性能与社会合规性。**

- **链接: [https://arxiv.org/pdf/2511.21135](https://arxiv.org/pdf/2511.21135)**

> **作者:** Ziyi Chen; Yingnan Guo; Zedong Chu; Minghua Luo; Yanfen Shen; Mingchao Sun; Junjun Hu; Shichao Xie; Kuan Yang; Pei Shi; Zhining Gu; Lu Liu; Honglin Han; Xiaolong Wu; Mu Xu; Yu Zhang; Ning Guo
>
> **摘要:** Embodied navigation that adheres to social norms remains an open research challenge. Our SocialNav is a foundational model for socially-aware navigation with a hierarchical "brain-action" architecture, capable of understanding high-level social norms and generating low-level, socially compliant trajectories. To enable such dual capabilities, we construct the SocNav Dataset, a large-scale collection of 7 million samples, comprising (1) a Cognitive Activation Dataset providing social reasoning signals such as chain-of-thought explanations and social traversability prediction, and (2) an Expert Trajectories Pyramid aggregating diverse navigation demonstrations from internet videos, simulated environments, and real-world robots. A multi-stage training pipeline is proposed to gradually inject and refine navigation intelligence: we first inject general navigation skills and social norms understanding into the model via imitation learning, and then refine such skills through a deliberately designed Socially-Aware Flow Exploration GRPO (SAFE-GRPO), the first flow-based reinforcement learning framework for embodied navigation that explicitly rewards socially compliant behaviors. SocialNav achieves +38% success rate and +46% social compliance rate compared to the state-of-the-art method, demonstrating strong gains in both navigation performance and social compliance. Our project page: this https URL
>
---
#### [replaced 002] System Design of the Ultra Mobility Vehicle: A Driving, Balancing, and Jumping Bicycle Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在开发一种具备动态移动能力的自行车机器人，解决平衡、跳跃与高效运动问题。工作包括设计优化与强化学习控制，实现多种复杂动作。**

- **链接: [https://arxiv.org/pdf/2602.22118](https://arxiv.org/pdf/2602.22118)**

> **作者:** Benjamin Bokser; Daniel Gonzalez; Surya Singh; Aaron Preston; Alex Bahner; Annika Wollschläger; Arianna Ilvonen; Asa Eckert-Erdheim; Ashwin Khadke; Bilal Hammoud; Dean Molinaro; Fabian Jenelten; Henry Mayne; Howie Choset; Igor Bogoslavskyi; Itic Tinman; James Tigue; Jan Preisig; Kaiyu Zheng; Kenny Sharma; Kim Ang; Laura Lee; Liana Margolese; Nicole Lin; Oscar Frias; Paul Drews; Ravi Boggavarapu; Rick Burnham; Samuel Zapolsky; Sangbae Kim; Scott Biddlestone; Sean Mayorga; Shamel Fahmi; Tyler McCollum; Velin Dimitrov; William Moyne; Yu-Ming Chen; Farbod Farshidian; Marco Hutter; David Perry; Al Rizzi; Gabe Nelson
>
> **备注:** 19 Pages, 11 figures, 3 movies, 2 tables
>
> **摘要:** Trials cyclists and mountain bike riders can hop, jump, balance, and drive on one or both wheels. This versatility allows them to achieve speed and energy-efficiency on smooth terrain and agility over rough terrain. Inspired by these athletes, we present the design and control of a robotic platform, Ultra Mobility Vehicle (UMV), which combines a bicycle and a reaction mass to move dynamically with minimal actuated degrees of freedom. We employ a simulation-driven design optimization process to synthesize a spatial linkage topology with a focus on vertical jump height and momentum-based balancing on a single wheel contact. Using a constrained Reinforcement Learning (RL) framework, we demonstrate zero-shot transfer of diverse athletic behaviors, including track-stands, jumps, wheelies, rear wheel hopping, and front flips. This 23.5 kg robot is capable of high speeds (8 m/s) and jumping on and over large obstacles (1 m tall, or 130% of the robot's nominal height).
>
---
#### [replaced 003] Parallel Continuous-Time Relative Localization with Augmented Clamped Non-Uniform B-Splines
- **分类: cs.RO**

- **简介: 该论文属于多机器人相对定位任务，解决异步测量和时钟偏移问题。提出CT-RIO框架，使用改进的B样条实现高精度、低延迟的相对定位。**

- **链接: [https://arxiv.org/pdf/2602.22006](https://arxiv.org/pdf/2602.22006)**

> **作者:** Jiadong Lu; Zhehan Li; Tao Han; Miao Xu; Chao Xu; Yanjun Cao
>
> **备注:** 26 pages, 23 figures, submitted to IEEE Transactions on Robotics
>
> **摘要:** Accurate relative localization is critical for multi-robot cooperation. In robot swarms, measurements from different robots arrive asynchronously and with clock time-offsets. Although Continuous-Time (CT) formulations have proved effective for handling asynchronous measurements in single-robot SLAM and calibration, extending CT methods to multi-robot settings faces great challenges to achieve high-accuracy, low-latency, and high-frequency performance. Especially, existing CT methods suffer from the inherent query-time delay of unclamped B-splines and high computational cost. This paper proposes CT-RIO, a novel Continuous-Time Relative-Inertial Odometry framework. We employ Clamped Non-Uniform B-splines (C-NUBS) to represent robot states for the first time, eliminating the query-time delay. We further augment C-NUBS with closed-form extension and shrinkage operations that preserve the spline shape, making it suitable for online estimation and enabling flexible knot management. This flexibility leads to the concept of knot-keyknot strategy, which supports spline extension at high-frequency while retaining sparse keyknots for adaptive relative-motion modeling. We then formulate a sliding-window relative localization problem that operates purely on relative kinematics and inter-robot constraints. To meet the demanding computation required at swarm scale, we decompose the tightly-coupled optimization into robot-wise sub-problems and solve them in parallel using incremental asynchronous block coordinate descent. Extensive experiments show that CT-RIO converges from time-offsets as large as 263 ms to sub-millisecond within 3 s, and achieves RMSEs of 0.046 m and 1.8 °. It consistently outperforms state-of-the-art methods, with improvements of up to 60% under high-speed motion.
>
---
#### [replaced 004] Adversarial Fine-tuning in Offline-to-Online Reinforcement Learning for Robust Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习任务，解决静态数据训练策略在动作空间扰动下的脆弱性问题。通过对抗微调和自适应课程策略提升机器人控制的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.13358](https://arxiv.org/pdf/2510.13358)**

> **作者:** Shingo Ayabe; Hiroshi Kera; Kazuhiko Kawamoto
>
> **备注:** 15 main pages, 8 supplementary material pages
>
> **摘要:** Offline reinforcement learning enables sample-efficient policy acquisition without risky online interaction, yet policies trained on static datasets remain brittle under action-space perturbations such as actuator faults. This study introduces an offline-to-online framework that trains policies on clean data and then performs adversarial fine-tuning, where perturbations are injected into executed actions to induce compensatory behavior and improve resilience. A performance-aware curriculum further adjusts the perturbation probability during training via an exponential-moving-average signal, balancing robustness and stability throughout the learning process. Experiments on continuous-control locomotion tasks demonstrate that the proposed method consistently improves robustness over offline-only baselines and converges faster than training from scratch. Matching the fine-tuning and evaluation conditions yields the strongest robustness to action-space perturbations, while the adaptive curriculum strategy mitigates the degradation of nominal performance observed with the linear curriculum strategy. Overall, the results show that adversarial fine-tuning enables adaptive and robust control under uncertain environments, bridging the gap between offline efficiency and online adaptability.
>
---
#### [replaced 005] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对长时域机器人操作任务中的稀疏奖励问题，提出AC3框架，通过连续动作块学习实现高效稳定策略。**

- **链接: [https://arxiv.org/pdf/2508.11143](https://arxiv.org/pdf/2508.11143)**

> **作者:** Jiarui Yang; Bin Zhu; Jingjing Chen; Yu-Gang Jiang
>
> **备注:** 14 pages, 13 figures, Accepted by AAAI 2026 (oral)
>
> **摘要:** Existing reinforcement learning (RL) methods struggle with long-horizon robotic manipulation tasks, particularly those involving sparse rewards. While action chunking is a promising paradigm for robotic manipulation, using RL to directly learn continuous action chunks in a stable and data-efficient manner remains a critical challenge. This paper introduces AC3 (Actor-Critic for Continuous Chunks), a novel RL framework that learns to generate high-dimensional, continuous action sequences. To make this learning process stable and data-efficient, AC3 incorporates targeted stabilization mechanisms for both the actor and the critic. First, to ensure reliable policy improvement, the actor is trained with an asymmetric update rule, learning exclusively from successful trajectories. Second, to enable effective value learning despite sparse rewards, the critic's update is stabilized using intra-chunk $n$-step returns and further enriched by a self-supervised module providing intrinsic rewards at anchor points aligned with each action chunk. We conducted extensive experiments on 25 tasks from the BiGym and RLBench benchmarks. Results show that by using only a few demonstrations and a simple model architecture, AC3 achieves superior success rates on most tasks, validating its effective design.
>
---
#### [replaced 006] Less is More: Lean yet Powerful Vision-Language Model for Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文将自动驾驶视为语言问题，提出Max-V1框架，通过视觉-语言模型实现端到端轨迹预测，解决复杂驾驶策略学习问题。**

- **链接: [https://arxiv.org/pdf/2510.00060](https://arxiv.org/pdf/2510.00060)**

> **作者:** Sheng Yang; Tong Zhan; Guancheng Chen; Yanfeng Lu; Jian Wang
>
> **摘要:** In this work, we reconceptualize autonomous driving as a generalized language problem and formulate the trajectory planning task as next waypoint prediction. We introduce Max-V1, a novel framework for one-stage end-to-end autonomous driving, named in tribute to the renowned Dutch racing driver Max Verstappen. Our framework presents a single-pass generation paradigm that aligns with the inherent sequentiality of driving. This approach leverages the generative capacity of the Vision-Language Model (VLM) to enable end-to-end trajectory prediction directly from front-view camera input. The efficacy of this method is underpinned by a principled supervision strategy derived from statistical modeling. This provides a well-defined learning objective, which makes the framework highly amenable to mastering complex driving policies through imitation learning from large-scale expert demonstrations. Empirically, our method achieves state-of-the-art performance on the nuScenes dataset, delivering an overall improvement of over 30% compared to prior baselines. Furthermore, it exhibits superior generalization performance on cross-domain datasets acquired from diverse vehicles, demonstrating notable potential for cross-vehicle robustness and adaptability. With these empirical strengths, this work introduces a model that enables fundamental driving behaviors, laying the foundation for the development of more capable self-driving agents. Code will be available upon publication.
>
---
#### [replaced 007] Model Predictive Control with Reference Learning for Soft Robotic Intracranial Pressure Waveform Modulation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制任务，旨在通过学习方法实现软体机器人对颅内压波形的精确调控。提出双层框架，结合模型预测控制与贝叶斯优化，解决非线性依赖问题，提升控制精度。**

- **链接: [https://arxiv.org/pdf/2509.13109](https://arxiv.org/pdf/2509.13109)**

> **作者:** Fabian Flürenbrock; Yanick Büchel; Johannes Köhler; Marianne Schmid Daners; Melanie N. Zeilinger
>
> **摘要:** This paper introduces a learning-based control framework for a soft robotic actuator system designed to modulate intracranial pressure (ICP) waveforms, which is essential for studying cerebrospinal fluid dynamics and pathological processes underlying neurological disorders. A two-layer framework is proposed to safely achieve a desired ICP waveform modulation. First, a model predictive controller (MPC) with a disturbance observer is used for offset-free tracking of the system's motor position reference trajectory under safety constraints. Second, to address the unknown nonlinear dependence of ICP on the motor position, we employ a Bayesian optimization (BO) algorithm used for online learning of a motor position reference trajectory that yields the desired ICP modulation. The framework is experimentally validated using a test bench with a brain phantom that replicates realistic ICP dynamics in vitro. Compared to a previously employed proportional-integral-derivative controller, the MPC reduces mean and maximum motor position reference tracking errors by 83 % and 73 %, respectively. In less than 20 iterations, the BO algorithm learns a motor position reference trajectory that yields an ICP waveform with the desired mean and amplitude.
>
---
#### [replaced 008] Generalized Momenta-Based Koopman Formalism for Robust Control of Euler-Lagrangian Systems
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制领域，旨在解决Euler-Lagrangian系统的建模与控制问题。通过引入基于广义动量的Koopman形式化方法，实现更高效、鲁棒的模型构建与控制策略。**

- **链接: [https://arxiv.org/pdf/2509.17010](https://arxiv.org/pdf/2509.17010)**

> **作者:** Rajpal Singh; Aditya Singh; Chidre Shravista Kashyap; Jishnu Keshavan
>
> **摘要:** This paper presents a novel Koopman operator formulation for Euler Lagrangian dynamics that employs an implicit generalized momentum-based state space representation, which decouples a known linear actuation channel from state dependent dynamics and makes the system more amenable to linear Koopman modeling. By leveraging this structural separation, the proposed formulation only requires to learn the unactuated dynamics rather than the complete actuation dependent system, thereby significantly reducing the number of learnable parameters, improving data efficiency, and lowering overall model complexity. In contrast, conventional explicit formulations inherently couple inputs with the state dependent terms in a nonlinear manner, making them more suitable for bilinear Koopman models, which are more computationally expensive to train and deploy. Notably, the proposed scheme enables the formulation of linear models that achieve superior prediction performance compared to conventional bilinear models while remaining substantially more efficient. To realize this framework, we present two neural network architectures that construct Koopman embeddings from actuated or unactuated data, enabling flexible and efficient modeling across different tasks. Robustness is ensured through the integration of a linear Generalized Extended State Observer (GESO), which explicitly estimates disturbances and compensates for them in real time. The combined momentum-based Koopman and GESO framework is validated through comprehensive trajectory tracking simulations and experiments on robotic manipulators, demonstrating superior accuracy, robustness, and learning efficiency relative to state of the art alternatives.
>
---
#### [replaced 009] Towards Intelligible Human-Robot Interaction: An Active Inference Approach to Occluded Pedestrian Scenarios
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决遮挡行人带来的安全挑战。通过主动推理框架，结合RBPF和CEM-MPPI控制器，提升决策的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2602.23109](https://arxiv.org/pdf/2602.23109)**

> **作者:** Kai Chen; Yuyao Huang; Guang Chen
>
> **备注:** 14 pages, 6 figures, Proceedings of the 2026 ACM/IEEE International Conference on Human-Robot Interaction (HRI'26)
>
> **摘要:** The sudden appearance of occluded pedestrians presents a critical safety challenge in autonomous driving. Conventional rule-based or purely data-driven approaches struggle with the inherent high uncertainty of these long-tail scenarios. To tackle this challenge, we propose a novel framework grounded in Active Inference, which endows the agent with a human-like, belief-driven mechanism. Our framework leverages a Rao-Blackwellized Particle Filter (RBPF) to efficiently estimate the pedestrian's hybrid state. To emulate human-like cognitive processes under uncertainty, we introduce a Conditional Belief Reset mechanism and a Hypothesis Injection technique to explicitly model beliefs about the pedestrian's multiple latent intentions. Planning is achieved via a Cross-Entropy Method (CEM) enhanced Model Predictive Path Integral (MPPI) controller, which synergizes the efficient, iterative search of CEM with the inherent robustness of MPPI. Simulation experiments demonstrate that our approach significantly reduces the collision rate compared to reactive, rule-based, and reinforcement learning (RL) baselines, while also exhibiting explainable and human-like driving behavior that reflects the agent's internal belief state.
>
---
#### [replaced 010] Embodiment-Aware Generalist Specialist Distillation for Unified Humanoid Whole-Body Control
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，解决多机器人通用策略训练问题。提出EAGLE框架，通过迭代蒸馏使单一策略控制多种人形机器人，提升泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.02960](https://arxiv.org/pdf/2602.02960)**

> **作者:** Quanquan Peng; Yunfeng Lin; Yufei Xue; Jiangmiao Pang; Weinan Zhang
>
> **摘要:** Humanoid Whole-Body Controllers trained with reinforcement learning (RL) have recently achieved remarkable performance, yet many target a single robot embodiment. Variations in dynamics, degrees of freedom (DoFs), and kinematic topology still hinder a single policy from commanding diverse humanoids. Moreover, obtaining a generalist policy that not only transfers across embodiments but also supports richer behaviors-beyond simple walking to squatting, leaning-remains especially challenging. In this work, we tackle these obstacles by introducing EAGLE, an iterative generalist-specialist distillation framework that produces a single unified policy that controls multiple heterogeneous humanoids without per-robot reward tuning. During each cycle, embodiment-specific specialists are forked from the current generalist, refined on their respective robots, and new skills are distilled back into the generalist by training on the pooled embodiment set. Repeating this loop until performance convergence produces a robust Whole-Body Controller validated on robots such as Unitree H1, G1, and Fourier N1. We conducted experiments on five different robots in simulation and four in real-world settings. Through quantitative evaluations, EAGLE achieves high tracking accuracy and robustness compared to other methods, marking a step toward scalable, fleet-level humanoid control. See more details at this https URL
>
---
#### [replaced 011] LEMON-Mapping: Loop-Enhanced Large-Scale Multi-Session Point Cloud Merging and Optimization for Globally Consistent Mapping
- **分类: cs.RO**

- **简介: 该论文属于多机器人全局一致地图构建任务，解决传统方法在重叠区域出现发散和模糊的问题。提出LEMON-Mapping框架，通过优化回环和空间束调整提升地图精度与一致性。**

- **链接: [https://arxiv.org/pdf/2505.10018](https://arxiv.org/pdf/2505.10018)**

> **作者:** Lijie Wang; Xiaoyi Zhong; Ziyi Xu; Kaixin Chai; Anke Zhao; Tianyu Zhao; Changjian Jiang; Qianhao Wang; Fei Gao
>
> **摘要:** Multi-robot collaboration is becoming increasingly critical and presents significant challenges in modern robotics, especially for building a globally consistent, accurate map. Traditional multi-robot pose graph optimization (PGO) methods ensure basic global consistency but ignore the geometric structure of the map, and only use loop closures as constraints between pose nodes, leading to divergence and blurring in overlapping regions. To address this issue, we propose LEMON-Mapping, a loop-enhanced framework for large-scale, multi-session point cloud fusion and optimization. We re-examine the role of loops for multi-robot mapping and introduce three key innovations. First, we develop a robust loop processing mechanism that rejects outliers and a loop recall strategy to recover mistakenly removed but valid loops. Second, we introduce spatial bundle adjustment for multi-robot maps, reducing divergence and eliminating blurring in overlaps. Third, we design a PGO-based approach that leverages refined bundle adjustment constraints to propagate local accuracy to the entire map. We validate LEMON-Mapping on several public datasets and a self-collected dataset. The experimental results show superior mapping accuracy and global consistency of our framework compared to traditional merging methods. Scalability experiments also demonstrate its strong capability to handle scenarios involving numerous robots.
>
---
#### [replaced 012] CLEAR-IR: Clarity-Enhanced Active Reconstruction of Infrared Imagery
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于红外图像重建任务，旨在解决低光环境下机器人感知困难的问题。通过提出DeepMAO架构，提升红外图像质量并支持机器人视觉任务。**

- **链接: [https://arxiv.org/pdf/2510.04883](https://arxiv.org/pdf/2510.04883)**

> **作者:** Nathan Shankar; Pawel Ladosz; Hujun Yin
>
> **备注:** 8 pages, 6 figures, 2 tables
>
> **摘要:** This paper presents a novel approach for enabling robust robotic perception in dark environments using infrared (IR) stream. IR stream is less susceptible to noise than RGB in low-light conditions. However, it is dominated by active emitter patterns that hinder high-level tasks such as object detection, tracking and localisation. To address this, a Deep Multi-scale Aware Overcomplete (DeepMAO) inspired architecture is proposed that reconstructs clean IR images from emitter populated input, improving both image quality and downstream robotic performance. This approach outperforms existing enhancement techniques and enables reliable operation of vision driven robotic systems across illumination conditions from well-lit to extreme low-light scenes. The results outline the ability of this work to be able to mimic RGB styling from the scene and its applicability on robotics tasks that were trained on RGB images, opening the possibility of doing these tasks in extreme low-light without on-board lighting.
>
---
#### [replaced 013] Point Bridge: 3D Representations for Cross Domain Policy Learning
- **分类: cs.RO**

- **简介: 该论文提出Point Bridge框架，解决仿真到现实的策略迁移问题。通过点云表示和VLM提取特征，实现零样本跨域策略学习，提升真实环境操作性能。**

- **链接: [https://arxiv.org/pdf/2601.16212](https://arxiv.org/pdf/2601.16212)**

> **作者:** Siddhant Haldar; Lars Johannsmeier; Lerrel Pinto; Abhishek Gupta; Dieter Fox; Yashraj Narang; Ajay Mandlekar
>
> **摘要:** Robot foundation models are beginning to deliver on the promise of generalist robotic agents, yet progress remains constrained by the scarcity of large-scale real-world manipulation datasets. Simulation and synthetic data generation offer a scalable alternative, but their usefulness is limited by the visual domain gap between simulation and reality. In this work, we present Point Bridge, a framework that leverages unified, domain-agnostic point-based representations to unlock synthetic datasets for zero-shot sim-to-real policy transfer, without explicit visual or object-level alignment. Point Bridge combines automated point-based representation extraction via Vision-Language Models (VLMs), transformer-based policy learning, and efficient inference-time pipelines to train capable real-world manipulation agents using only synthetic data. With additional co-training on small sets of real demonstrations, Point Bridge further improves performance, substantially outperforming prior vision-based sim-and-real co-training methods. It achieves up to 44% gains in zero-shot sim-to-real transfer and up to 66% with limited real data across both single-task and multitask settings. Videos of the robot are best viewed at: this https URL
>
---
#### [replaced 014] Mixed-Initiative Dialog for Human-Robot Collaborative Manipulation
- **分类: cs.RO; cs.CL; cs.HC; cs.LG; cs.MA**

- **简介: 该论文属于人机协作任务，旨在提升长期合作中机器人与人类的沟通效率。研究提出MICoBot系统，通过多级决策优化任务分配，减少人力负担，提高任务成功率和用户体验。**

- **链接: [https://arxiv.org/pdf/2508.05535](https://arxiv.org/pdf/2508.05535)**

> **作者:** Albert Yu; Chengshu Li; Luca Macesanu; Arnav Balaji; Ruchira Ray; Raymond Mooney; Roberto Martín-Martín
>
> **备注:** Project website at this https URL
>
> **摘要:** Effective robotic systems for long-horizon human-robot collaboration must adapt to a wide range of human partners, whose physical behavior, willingness to assist, and understanding of the robot's capabilities may change over time. This demands a tightly coupled communication loop that grants both agents the flexibility to propose, accept, or decline requests as they coordinate toward completing the task effectively. We apply a Mixed-Initiative dialog paradigm to Collaborative human-roBot teaming and propose MICoBot, a system that handles the common scenario where both agents, using natural language, take initiative in formulating, accepting, or rejecting proposals on who can best complete different steps of a task. To handle diverse, task-directed dialog, and find successful collaborative strategies that minimize human effort, MICoBot makes decisions at three levels: (1) a meta-planner considers human dialog to formulate and code a high-level collaboration strategy, (2) a planner optimally allocates the remaining steps to either agent based on the robot's capabilities (measured by a simulation-pretrained affordance model) and the human's estimated availability to help, and (3) an action executor decides the low-level actions to perform or words to say to the human. In physical robot trials with 18 unique human participants, MICoBot significantly improves task success and user experience over a pure LLM baseline and standard agent allocation models. See additional videos and materials at this https URL.
>
---
#### [replaced 015] Attentive Feature Aggregation or: How Policies Learn to Stop Worrying about Robustness and Attend to Task-Relevant Visual Cues
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-运动策略学习任务，旨在解决预训练视觉表示中冗余信息导致的泛化能力不足问题。通过引入注意力特征聚合机制，提升策略对任务相关视觉线索的聚焦能力。**

- **链接: [https://arxiv.org/pdf/2511.10762](https://arxiv.org/pdf/2511.10762)**

> **作者:** Nikolaos Tsagkas; Andreas Sochopoulos; Duolikun Danier; Sethu Vijayakumar; Alexandros Kouris; Oisin Mac Aodha; Chris Xiaoxuan Lu
>
> **备注:** This paper stems from a split of our earlier work "When Pre-trained Visual Representations Fall Short: Limitations in Visuo-Motor Robot Learning." While "The Temporal Trap" replaces the original and focuses on temporal entanglement, this companion study examines policy robustness and task-relevant visual cue selection. arXiv admin note: text overlap with arXiv:2502.03270
>
> **摘要:** The adoption of pre-trained visual representations (PVRs), leveraging features from large-scale vision models, has become a popular paradigm for training visuomotor policies. However, these powerful representations can encode a broad range of task-irrelevant scene information, making the resulting trained policies vulnerable to out-of-domain visual changes and distractors. In this work we address visuomotor policy feature pooling as a solution to the observed lack of robustness in perturbed scenes. We achieve this via Attentive Feature Aggregation (AFA), a lightweight, trainable pooling mechanism that learns to naturally attend to task-relevant visual cues, ignoring even semantically rich scene distractors. Through extensive experiments in both simulation and the real world, we demonstrate that policies trained with AFA significantly outperform standard pooling approaches in the presence of visual perturbations, without requiring expensive dataset augmentation or fine-tuning of the PVR. Our findings show that ignoring extraneous visual information is a crucial step towards deploying robust and generalisable visuomotor policies. Project Page: this http URL
>
---
#### [replaced 016] SWITCH: Benchmarking Modeling and Handling of Tangible Interfaces in Long-horizon Embodied Scenarios
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出SWITCH基准，用于评估长期沉浸式场景中实体接口的建模与处理能力。解决部分可观测、因果推理和验证等难题，涵盖任务感知VQA、语义UI定位等五项能力。**

- **链接: [https://arxiv.org/pdf/2511.17649](https://arxiv.org/pdf/2511.17649)**

> **作者:** Jieru Lin; Zhiwei Yu; Börje F. Karlsson
>
> **摘要:** Autonomous agents operating in the real world must interact continuously with existing physical and semantic infrastructure, track delayed consequences, and verify outcomes over time. Everyday environments are rich in tangible control interfaces (TCIs)-e.g., light switches, appliance panels, and embedded GUI-posing core challenges for lifelong embodied agents, including partial observability, causal reasoning across time, and failure-aware verification under real-world constraints. Yet, current benchmarks rarely consider such long-horizon interaction and causality requirements. We introduce SWITCH (Semantic World Interface Tasks for Control & Handling), an embodied, task-driven benchmark created through iterative releases to probe these gaps. Its first iteration, SWITCH-Basic, evaluates five complementary abilities-task-aware VQA, semantic UI grounding, action generation, state transition prediction, and result verification-under ego-centric RGB video input and device diversity across 351 tasks spanning 98 real devices/appliances. Results from commercial and open LMMMs reveal systematic failures, highlighting critical gaps for lifelong agent deployment. SWITCH provides data, code, and held-out splits to enable reproducible non-contaminated evaluation and community contributions toward more challenging future iterations of the benchmark and the creation of relevant training data. Benchmark resources are available at: this https URL.
>
---
#### [replaced 017] Off-Road Navigation via Implicit Neural Representation of Terrain Traversability
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决越野环境下的路径规划问题。提出TRAIL框架，利用隐式神经表示建模地形可通行性，优化路径和速度以提升导航性能。**

- **链接: [https://arxiv.org/pdf/2511.18183](https://arxiv.org/pdf/2511.18183)**

> **作者:** Yixuan Jia; Qingyuan Li; Jonathan P. How
>
> **备注:** Full version: 10 pages
>
> **摘要:** Autonomous off-road navigation requires robots to estimate terrain traversability from onboard sensors and plan motion accordingly. Conventional approaches typically rely on sampling-based planners such as MPPI to generate short-term control actions that aim to minimize traversal time and risk measures derived from the traversability estimates. These planners can react quickly but optimize only over a short look-ahead window, limiting their ability to reason about the full path geometry, which is important for navigating in challenging off-road environments. Moreover, they lack the ability to adjust speed based on the terrain-induced vibrations, which is important for smooth navigation on challenging terrains. In this paper, we introduce TRAIL (Traversability with an Implicit Learned Representation), an off-road navigation framework that leverages an implicit neural representation to model terrain properties as a continuous field that can be queried at arbitrary locations. This representation yields spatial gradients that enable integration with a novel gradient-based trajectory optimization method that adapts the path geometry and speed profile based on terrain traversability.
>
---
#### [replaced 018] Flow-Enabled Generalization to Human Demonstrations in Few-Shot Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于模仿学习任务，旨在减少对大量机器人示范数据的依赖。通过引入场景流和点云条件策略，提升从人类视频中泛化的能力。**

- **链接: [https://arxiv.org/pdf/2602.10594](https://arxiv.org/pdf/2602.10594)**

> **作者:** Runze Tang; Penny Sweetser
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Imitation Learning (IL) enables robots to learn complex skills from demonstrations without explicit task modeling, but it typically requires large amounts of demonstrations, creating significant collection costs. Prior work has investigated using flow as an intermediate representation to enable the use of human videos as a substitute, thereby reducing the amount of required robot demonstrations. However, most prior work has focused on the flow, either on the object or on specific points of the robot/hand, which cannot describe the motion of interaction. Meanwhile, relying on flow to achieve generalization to scenarios observed only in human videos remains limited, as flow alone cannot capture precise motion details. Furthermore, conditioning on scene observation to produce precise actions may cause the flow-conditioned policy to overfit to training tasks and weaken the generalization indicated by the flow. To address these gaps, we propose SFCrP, which includes a Scene Flow prediction model for Cross-embodiment learning (SFCr) and a Flow and Cropped point cloud conditioned Policy (FCrP). SFCr learns from both robot and human videos and predicts any point trajectories. FCrP follows the general flow motion and adjusts the action based on observations for precision tasks. Our method outperforms SOTA baselines across various real-world task settings, while also exhibiting strong spatial and instance generalization to scenarios seen only in human videos.
>
---
#### [replaced 019] DropVLA: An Action-Level Backdoor Attack on Vision--Language--Action Models
- **分类: cs.CR; cs.AI; cs.RO**

- **简介: 该论文属于安全领域，研究VLA模型的后门攻击问题。提出DropVLA攻击方法，在有限数据污染下实现对具体动作的控制，验证了其有效性与隐蔽性。**

- **链接: [https://arxiv.org/pdf/2510.10932](https://arxiv.org/pdf/2510.10932)**

> **作者:** Zonghuan Xu; Xiang Zheng; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** 8 pages, 6 tables, 3 figures. Under review
>
> **摘要:** Vision-Language-Action (VLA) models map multimodal perception and language instructions to executable robot actions, making them particularly vulnerable to behavioral backdoor manipulation: a hidden trigger introduced during training can induce unintended physical actions while nominal task performance remains intact. Prior work on VLA backdoors primarily studies untargeted attacks or task-level hijacking, leaving fine-grained control over individual actions largely unexplored. In this work, we present DropVLA, an action-level backdoor attack that forces a reusable action primitive (e.g., open_gripper) to execute at attacker-chosen decision points under a realistic pipeline-black-box setting with limited data-poisoning access, using a window-consistent relabeling scheme for chunked fine-tuning. On OpenVLA-7B evaluated with LIBERO, vision-only poisoning achieves 98.67%-99.83% attack success rate (ASR) with only 0.31% poisoned episodes while preserving 98.50%-99.17% clean-task retention, and successfully triggers the targeted action within 25 control steps at 500 Hz (0.05 s). Text-only triggers are unstable at low poisoning budgets, and combining text with vision provides no consistent ASR improvement over vision-only attacks. The backdoor remains robust to moderate trigger variations and transfers across evaluation suites (96.27%, 99.09%), whereas text-only largely fails (0.72%). We further validate physical-world feasibility on a 7-DoF Franka arm with pi0-fast, demonstrating non-trivial attack efficacy under camera-relative motion that induces image-plane trigger drift. These results reveal that VLA models can be covertly steered at the granularity of safety-critical actions with minimal poisoning and without observable degradation of nominal performance.
>
---
#### [replaced 020] DAGS-SLAM: Dynamic-Aware 3DGS SLAM via Spatiotemporal Motion Probability and Uncertainty-Aware Scheduling
- **分类: cs.RO**

- **简介: 该论文提出DAGS-SLAM，解决动态场景下的实时3D重建与定位问题，通过时空运动概率和不确定性调度提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2602.21644](https://arxiv.org/pdf/2602.21644)**

> **作者:** Li Zhang; Yu-An Liu; Xijia Jiang; Conghao Huang; Danyang Li; Yanyong Zhang
>
> **摘要:** Mobile robots and IoT devices demand real-time localization and dense reconstruction under tight compute and energy budgets. While 3D Gaussian Splatting (3DGS) enables efficient dense SLAM, dynamic objects and occlusions still degrade tracking and mapping. Existing dynamic 3DGS-SLAM often relies on heavy optical flow and per-frame segmentation, which is costly for mobile deployment and brittle under challenging illumination. We present DAGS-SLAM, a dynamic-aware 3DGS-SLAM system that maintains a spatiotemporal motion probability (MP) state per Gaussian and triggers semantics on demand via an uncertainty-aware scheduler. DAGS-SLAM fuses lightweight YOLO instance priors with geometric cues to estimate and temporally update MP, propagates MP to the front-end for dynamic-aware correspondence selection, and suppresses dynamic artifacts in the back-end via MP-guided optimization. Experiments on public dynamic RGB-D benchmarks show improved reconstruction and robust tracking while sustaining real-time throughput on a commodity GPU, demonstrating a practical speed-accuracy tradeoff with reduced semantic invocations toward mobile deployment.
>
---
#### [replaced 021] DECO: Decoupled Multimodal Diffusion Transformer for Bimanual Dexterous Manipulation with a Plugin Tactile Adapter
- **分类: cs.RO; cs.AI**

- **简介: 该论文聚焦于双臂灵巧操作任务，解决多模态信息融合难题。提出DECO模型，通过解耦视觉、本体感觉和触觉信号提升控制效果，并引入轻量触觉适配器增强性能。**

- **链接: [https://arxiv.org/pdf/2602.05513](https://arxiv.org/pdf/2602.05513)**

> **作者:** Xukun Li; Yu Sun; Lei Zhang; Bosheng Huang; Yibo Peng; Yuan Meng; Haojun Jiang; Shaoxuan Xie; Guocai Yao; Alois Knoll; Zhenshan Bing; Xinlong Wang; Zhenguo Sun
>
> **备注:** 17 pages, 8 figures. Project Page: this https URL
>
> **摘要:** Bimanual dexterous manipulation relies on integrating multimodal inputs to perform complex real-world tasks. To address the challenges of effectively combining these modalities, we propose DECO, a decoupled multimodal diffusion transformer that disentangles vision, proprioception, and tactile signals through specialized conditioning pathways, enabling structured and controllable integration of multimodal inputs, with a lightweight adapter for parameter-efficient injection of additional signals. Alongside DECO, we release DECO-50 dataset for bimanual dexterous manipulation with tactile sensing, consisting of 50 hours of data and over 5M frames, collected via teleoperation on real dual-arm robots. We train DECO on DECO-50 and conduct extensive real-world evaluation with over 2,000 robot rollouts. Experimental results show that DECO achieves the best performance across all tasks, with a 72.25% average success rate and a 21% improvement over the baseline. Moreover, the tactile adapter brings an additional 10.25% average success rate across all tasks and a 20% gain on complex contact-rich tasks while tuning less than 10% of the model parameters.
>
---
#### [replaced 022] HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning
- **分类: cs.RO**

- **简介: 该论文提出HALO模型，解决机器人操作中长序列和分布外场景的多模态推理问题。通过统一的文本、视觉和动作推理框架提升任务成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21157](https://arxiv.org/pdf/2602.21157)**

> **作者:** Quanxin Shou; Fangqi Zhu; Shawn Chen; Puxin Yan; Zhengyang Yan; Yikun Miao; Xiaoyi Pang; Zicong Hong; Ruikai Shi; Hao Huang; Jie Zhang; Song Guo
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong performance in robotic manipulation, but often struggle in long-horizon or out-of-distribution scenarios due to the lack of explicit mechanisms for multimodal reasoning and anticipating how the world will evolve under action. Recent works introduce textual chain-of-thought or visual subgoal prediction within VLA models to reason, but still fail to offer a unified human-like reasoning framework for joint textual reasoning, visual foresight, and action prediction. To this end, we propose HALO, a unified VLA model that enables embodied multimodal chain-of-thought (EM-CoT) reasoning through a sequential process of textual task reasoning, visual subgoal prediction for fine-grained guidance, and EM-CoT-augmented action prediction. We instantiate HALO with a Mixture-of-Transformers (MoT) architecture that decouples semantic reasoning, visual foresight, and action prediction into specialized experts while allowing seamless cross-expert collaboration. To enable HALO learning at scale, we introduce an automated pipeline to synthesize EM-CoT training data along with a carefully crafted training recipe. Extensive experiments demonstrate that: (1) HALO achieves superior performance in both simulated and real-world environments, surpassing baseline policy pi_0 by 34.1% on RoboTwin benchmark; (2) all proposed components of the training recipe and EM-CoT design help improve task success rate; and (3) HALO exhibits strong generalization capabilities under aggressive unseen environmental randomization with our proposed EM-CoT reasoning.
>
---
#### [replaced 023] IntentCUA: Learning Intent-level Representations for Skill Abstraction and Multi-Agent Planning in Computer-Use Agents
- **分类: cs.AI; cs.HC; cs.RO**

- **简介: 该论文提出IntentCUA框架，解决计算机使用代理在复杂环境中的任务规划与技能抽象问题，通过意图对齐的计划记忆提升执行稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2602.17049](https://arxiv.org/pdf/2602.17049)**

> **作者:** Seoyoung Lee; Seobin Yoon; Seongbeen Lee; Yoojung Chun; Dayoung Park; Doyeon Kim; Joo Yong Sim
>
> **备注:** 12 pages, 9 figures, AAMAS 2026
>
> **摘要:** Computer-use agents operate over long horizons under noisy perception, multi-window contexts, evolving environment states. Existing approaches, from RL-based planners to trajectory retrieval, often drift from user intent and repeatedly solve routine subproblems, leading to error accumulation and inefficiency. We present IntentCUA, a multi-agent computer-use framework designed to stabilize long-horizon execution through intent-aligned plan memory. A Planner, Plan-Optimizer, and Critic coordinate over shared memory that abstracts raw interaction traces into multi-view intent representations and reusable skills. At runtime, intent prototypes retrieve subgroup-aligned skills and inject them into partial plans, reducing redundant re-planning and mitigating error propagation across desktop applications. In end-to-end evaluations, IntentCUA achieved a 74.83% task success rate with a Step Efficiency Ratio of 0.91, outperforming RL-based and trajectory-centric baselines. Ablations show that multi-view intent abstraction and shared plan memory jointly improve execution stability, with the cooperative multi-agent loop providing the largest gains on long-horizon tasks. These results highlight that system-level intent abstraction and memory-grounded coordination are key to reliable and efficient desktop automation in large, dynamic environments.
>
---
#### [replaced 024] Agile legged locomotion in reconfigurable modular robots
- **分类: cs.RO**

- **简介: 该论文研究模块化机器人敏捷步态问题，旨在解决传统腿式机器人结构单一、难以重构的缺陷。通过自主模块化腿部设计，实现灵活重组与动态运动。**

- **链接: [https://arxiv.org/pdf/2505.00784](https://arxiv.org/pdf/2505.00784)**

> **作者:** Chen Yu; David Matthews; Jingxian Wang; Jing Gu; Douglas Blackiston; Michael Rubenstein; Sam Kriegman
>
> **摘要:** Legged machines are becoming increasingly agile and adaptive but they have so far lacked the morphological diversity of legged animals, which have been rearranged and reshaped to fill millions of niches. Unlike their biological counterparts, legged machines have largely converged over the past decade to canonical quadrupedal and bipedal architectures that cannot be easily reconfigured to meet new tasks or recover from injury. Here we introduce autonomous modular legs: agile yet minimal, single-degree-of-freedom jointed links that can learn complex dynamic behaviors and may be freely attached to form multilegged machines at the meter scale. This enables rapid repair, redesign, and recombination of highly-dynamic modular agents that move quickly and acrobatically (non-quasistatically) through unstructured environments. Because each module is itself a complete agent, the bodies that contain them can sustain deep structural damage that would completely disable other legged robots. We also show how to encode the vast space of possible body configurations into a compact latent design space that can be efficiently explored, revealing a wide diversity of novel legged forms.
>
---
#### [replaced 025] Distributed Lloyd-Based algorithm for uncertainty-aware multi-robot under-canopy flocking
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同任务，解决复杂环境中无人机编队飞行问题。提出一种分布式算法，实现无通信、无先验信息的编队控制，确保避障和保持与邻近无人机的距离。**

- **链接: [https://arxiv.org/pdf/2504.18840](https://arxiv.org/pdf/2504.18840)**

> **作者:** Manuel Boldrer; Vit Kratky; Viktor Walter; Martin Saska
>
> **摘要:** In this letter, we present a distributed algorithm for flocking in complex environments that operates at constant altitude, without explicit communication, no a priori information about the environment, and by using only on-board sensing and computation capabilities. We provide sufficient conditions to guarantee collision avoidance with obstacles and other robots without exceeding a desired maximum distance from a predefined set of neighbors (flocking or proximity maintenance constraint) during the mission. The proposed approach allows to operate in crowded scenarios and to explicitly deal with tracking errors and on-board sensing errors. The algorithm was verified through simulations with varying number of UAVs and also through numerous real-world experiments in a dense forest involving up to four UAVs.
>
---
#### [replaced 026] Apple: Toward General Active Perception via Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出APPLE框架，利用强化学习解决通用主动感知问题，旨在提升机器人在不确定环境中的信息获取能力。**

- **链接: [https://arxiv.org/pdf/2505.06182](https://arxiv.org/pdf/2505.06182)**

> **作者:** Tim Schneider; Cristiana de Farias; Roberto Calandra; Liming Chen; Jan Peters
>
> **备注:** 27 pages; 21 figures; accepted at the Fourteenth International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** Active perception is a fundamental skill that enables us humans to deal with uncertainty in our inherently partially observable environment. For senses such as touch, where the information is sparse and local, active perception becomes crucial. In recent years, active perception has emerged as an important research domain in robotics. However, current methods are often bound to specific tasks or make strong assumptions, which limit their generality. To address this gap, this work introduces APPLE (Active Perception Policy Learning) - a novel framework that leverages reinforcement learning (RL) to address a range of different active perception problems. APPLE jointly trains a transformer-based perception module and decision-making policy with a unified optimization objective, learning how to actively gather information. By design, APPLE is not limited to a specific task and can, in principle, be applied to a wide range of active perception problems. We evaluate two variants of APPLE across different tasks, including tactile exploration problems from the Tactile MNIST benchmark. Experiments demonstrate the efficacy of APPLE, achieving high accuracies on both regression and classification tasks. These findings underscore the potential of APPLE as a versatile and general framework for advancing active perception in robotics. Project page: this https URL
>
---
#### [replaced 027] CO^3: Cooperative Unsupervised 3D Representation Learning for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无监督3D点云表示学习任务，旨在解决户外场景点云学习难题。通过协同对比学习与形状上下文预测，提升模型泛化能力和下游任务性能。**

- **链接: [https://arxiv.org/pdf/2206.04028](https://arxiv.org/pdf/2206.04028)**

> **作者:** Runjian Chen; Yao Mu; Runsen Xu; Wenqi Shao; Chenhan Jiang; Hang Xu; Zhenguo Li; Ping Luo
>
> **摘要:** Unsupervised contrastive learning for indoor-scene point clouds has achieved great successes. However, unsupervised learning point clouds in outdoor scenes remains challenging because previous methods need to reconstruct the whole scene and capture partial views for the contrastive objective. This is infeasible in outdoor scenes with moving objects, obstacles, and sensors. In this paper, we propose CO^3, namely Cooperative Contrastive Learning and Contextual Shape Prediction, to learn 3D representation for outdoor-scene point clouds in an unsupervised manner. CO^3 has several merits compared to existing methods. (1) It utilizes LiDAR point clouds from vehicle-side and infrastructure-side to build views that differ enough but meanwhile maintain common semantic information for contrastive learning, which are more appropriate than views built by previous methods. (2) Alongside the contrastive objective, shape context prediction is proposed as pre-training goal and brings more task-relevant information for unsupervised 3D point cloud representation learning, which are beneficial when transferring the learned representation to downstream detection tasks. (3) As compared to previous methods, representation learned by CO^3 is able to be transferred to different outdoor scene dataset collected by different type of LiDAR sensors. (4) CO^3 improves current state-of-the-art methods on both Once and KITTI datasets by up to 2.58 mAP. We believe CO^3 will facilitate understanding LiDAR point clouds in outdoor scene.
>
---
#### [replaced 028] Mixed formulation and structure-preserving discretization of Cosserat rod dynamics in a port-Hamiltonian framework
- **分类: math.NA; cs.CE; cs.RO; eess.SY; math.DS**

- **简介: 该论文属于计算力学领域，解决有限旋转下杆体动力学的建模问题。提出一种能量框架，采用混合变量和结构保持离散，实现能量动量一致的数值方法。**

- **链接: [https://arxiv.org/pdf/2512.19408](https://arxiv.org/pdf/2512.19408)**

> **作者:** Philipp L. Kinon; Simon R. Eugster; Peter Betsch
>
> **备注:** 39 pages, 16 figures
>
> **摘要:** An energy-based modeling framework for the nonlinear dynamics of spatial Cosserat rods undergoing large displacements and rotations is proposed. The mixed formulation features independent displacement, velocity and stress variables and is further objective and locking-free. Finite rotations are represented using a director formulation that avoids singularities and yields a constant mass matrix. This results in an infinite-dimensional nonlinear port-Hamiltonian (PH) system governed by partial differential-algebraic equations with a quadratic energy functional. Using a time-differentiated compliance form of the stress-strain relations allows for the imposition of kinematic constraints, such as inextensibility or shear-rigidity. A structure-preserving finite element discretization leads to a finite-dimensional system with PH structure, thus facilitating the design of an energy-momentum consistent integration scheme. Dissipative material behavior (via the generalized-Maxwell model) and non-standard actuation approaches (via pneumatic chambers or tendons) integrate naturally into the framework. As illustrated by selected numerical examples, the present framework establishes a new approach to energy-momentum consistent formulations in computational mechanics involving finite rotations.
>
---
#### [replaced 029] Less is more -- the Dispatcher/ Executor principle for multi-task Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出多任务强化学习中的调度器/执行器原则，旨在提升泛化能力和数据效率，解决复杂环境下的决策问题。**

- **链接: [https://arxiv.org/pdf/2312.09120](https://arxiv.org/pdf/2312.09120)**

> **作者:** Martin Riedmiller; Andrea Gesmundo; Tim Hertweck; Roland Hafner
>
> **备注:** Videos showing the results can be found at this https URL
>
> **摘要:** Humans instinctively know how to neglect details when it comes to solve complex decision making problems in environments with unforeseeable variations. This abstraction process seems to be a vital property for most biological systems and helps to 'abstract away' unnecessary details and boost generalisation. In this work we introduce the dispatcher/ executor principle for the design of multi-task Reinforcement Learning controllers. It suggests to partition the controller in two entities, one that understands the task (the dispatcher) and one that computes the controls for the specific device (the executor) - and to connect these two by a strongly regularizing communication channel. The core rationale behind this position paper is that changes in structure and design principles can improve generalisation properties and drastically enforce data-efficiency. It is in some sense a 'yes, and ...' response to the current trend of using large neural networks trained on vast amounts of data and bet on emerging generalisation properties. While we agree on the power of scaling - in the sense of Sutton's 'bitter lesson' - we will give some evidence, that considering structure and adding design principles can be a valuable and critical component in particular when data is not abundant and infinite, but is a precious resource.
>
---
#### [replaced 030] Development of a Deep Learning-Driven Control Framework for Exoskeleton Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决外骨骼机器人实时计算效率问题。通过设计深度学习控制框架，提升轨迹跟踪精度并降低计算负担。**

- **链接: [https://arxiv.org/pdf/2209.12133](https://arxiv.org/pdf/2209.12133)**

> **作者:** Sk Hasan
>
> **摘要:** The purpose of this study is to develop a computationally efficient deep learning based control framework for high degree of freedom exoskeleton robots to address the real time computational limitations associated with conventional model based control. A parallel structured deep neural network was designed for a seven degree of freedom human lower extremity exoskeleton robot. The network consists of four layers with 49 densely connected neurons and was trained using physics based data generated from the analytical dynamic model. During real time implementation, the trained neural network predicts joint torque commands required for trajectory tracking, while a proportional derivative controller compensates for residual prediction errors. Stability of the proposed control scheme was analytically established, and robustness to parameter variations was evaluated using analysis of variance. Comparative simulations were conducted against computed torque, model reference computed torque, sliding mode, adaptive, and linear quadratic controllers under identical robot dynamics. Results demonstrate accurate trajectory tracking with torque profiles comparable to conventional nonlinear controllers while reducing computational burden. These findings suggest that the proposed deep learning based hybrid controller offers an efficient and robust alternative for controlling multi degree of freedom exoskeleton robots.
>
---
#### [replaced 031] Automating the Refinement of Reinforcement Learning Specifications
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决逻辑规范不明确导致策略学习失败的问题。提出AutoSpec框架，通过优化规范提升学习效果。**

- **链接: [https://arxiv.org/pdf/2512.01047](https://arxiv.org/pdf/2512.01047)**

> **作者:** Tanmay Ambadkar; Đorđe Žikelić; Abhinav Verma
>
> **备注:** Fourteenth International Conference on Learning Representations 2026 this https URL
>
> **摘要:** Logical specifications have been shown to help reinforcement learning algorithms in achieving complex tasks. However, when a task is under-specified, agents might fail to learn useful policies. In this work, we explore the possibility of improving coarse-grained logical specifications via an exploration-guided strategy. We propose AutoSpec, a framework that searches for a logical specification refinement whose satisfaction implies satisfaction of the original specification, but which provides additional guidance therefore making it easier for reinforcement learning algorithms to learn useful policies. AutoSpec is applicable to reinforcement learning tasks specified via the SpectRL specification logic. We exploit the compositional nature of specifications written in SpectRL, and design four refinement procedures that modify the abstract graph of the specification by either refining its existing edge specifications or by introducing new edge specifications. We prove that all four procedures maintain specification soundness, i.e. any trajectory satisfying the refined specification also satisfies the original. We then show how AutoSpec can be integrated with existing reinforcement learning algorithms for learning policies from logical specifications. Our experiments demonstrate that AutoSpec yields promising improvements in terms of the complexity of control tasks that can be solved, when refined logical specifications produced by AutoSpec are utilized.
>
---
#### [replaced 032] Motion-aware Event Suppression for Event Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种运动感知事件抑制框架，用于过滤事件相机中由IMOs和自运动引起的噪声。任务是提升事件流处理的准确性与效率，通过实时分割与预测运动实现动态事件的提前抑制。**

- **链接: [https://arxiv.org/pdf/2602.23204](https://arxiv.org/pdf/2602.23204)**

> **作者:** Roberto Pellerito; Nico Messikommer; Giovanni Cioffi; Marco Cannici; Davide Scaramuzza
>
> **摘要:** In this work, we introduce the first framework for Motion-aware Event Suppression, which learns to filter events triggered by IMOs and ego-motion in real time. Our model jointly segments IMOs in the current event stream while predicting their future motion, enabling anticipatory suppression of dynamic events before they occur. Our lightweight architecture achieves 173 Hz inference on consumer-grade GPUs with less than 1 GB of memory usage, outperforming previous state-of-the-art methods on the challenging EVIMO benchmark by 67\% in segmentation accuracy while operating at a 53\% higher inference rate. Moreover, we demonstrate significant benefits for downstream applications: our method accelerates Vision Transformer inference by 83\% via token pruning and improves event-based visual odometry accuracy, reducing Absolute Trajectory Error (ATE) by 13\%.
>
---
#### [replaced 033] Human Autonomy and Sense of Agency in Human-Robot Interaction: A Systematic Literature Review
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于系统综述任务，旨在探讨人机交互中人类自主性和代理感的影响因素，分析现有研究并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2509.22271](https://arxiv.org/pdf/2509.22271)**

> **作者:** Felix Glawe; Tim Schmeckel; Philipp Brauner; Martina Ziefle
>
> **摘要:** Human autonomy and sense of agency are increasingly recognised as critical for user well-being, motivation, and the ethical deployment of robots in human-robot interaction (HRI). Given the rapid development of artificial intelligence, robot capabilities and their potential to function as colleagues and companions are growing. This systematic literature review synthesises 22 empirical studies selected from an initial pool of 728 articles published between 2011 and 2024. Articles were retrieved from major scientific databases and identified based on empirical focus and conceptual relevance, namely, how to preserve and promote human autonomy and sense of agency in HRI. Derived through thematic synthesis, five clusters of potentially influential factors are revealed: robot adaptiveness, communication style, anthropomorphism, presence of a robot and individual differences. Measured through psychometric scales or the intentional binding paradigm, perceptions of autonomy and agency varied across industrial, educational, healthcare, care, and hospitality settings. The review underscores the theoretical differences between both concepts, but their yet entangled use in HRI. Despite increasing interest, the current body of empirical evidence remains limited and fragmented, underscoring the necessity for standardised definitions, more robust operationalisations, and further exploratory and qualitative research. By identifying existing gaps and highlighting emerging trends, this review contributes to the development of human-centered, autonomy-supportive robot design strategies that uphold ethical and psychological principles, ultimately supporting well-being in human-robot interaction.
>
---
#### [replaced 034] Beyond Ground: Map-Free LiDAR Relocalization for UAVs
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于无人机定位任务，解决 UAV 在无地图环境下高精度定位问题。提出 MAILS 框架，提升 LiDAR 特征提取鲁棒性，并构建真实场景数据集。**

- **链接: [https://arxiv.org/pdf/2602.13267](https://arxiv.org/pdf/2602.13267)**

> **作者:** Hengyu Mu; Jianshi Wu; Yuxin Guo; XianLian Lin; Qingyong Hu; Sheng Ao; Chenglu Wen; Cheng Wang
>
> **备注:** 18 pages, 16 figures
>
> **摘要:** Localization is a fundamental capability in unmanned aerial vehicle (UAV) systems. Map-free LiDAR relocalization offers an effective solution for achieving high-precision positioning in environments with weak or unavailable GNSS signals. However, existing LiDAR relocalization methods are primarily tailored to autonomous driving, exhibiting significantly degraded accuracy in UAV scenarios. In this paper, we propose MAILS, a novel map-free LiDAR relocalization framework for UAVs. A Locality-Preserving Sliding Window Attention module is first introduced to extract locally discriminative geometric features from sparse point clouds. To handle substantial yaw rotations and altitude variations encountered during UAV flight, we then design a coordinate-independent feature initialization module and a locally invariant positional encoding mechanism, which together significantly enhance the robustness of feature extraction. Furthermore, existing LiDAR-based relocalization datasets fail to capture real-world UAV flight characteristics, such as irregular trajectories and varying altitudes. To address this gap, we construct a large-scale LiDAR localization dataset for UAVs, which comprises four scenes and various flight trajectories, designed to evaluate UAV relocalization performance under realistic conditions. Extensive experiments demonstrate that our method achieves satisfactory localization precision and consistently outperforms existing techniques by a significant margin. Our code and dataset will be released soon.
>
---
#### [replaced 035] DySL-VLA: Efficient Vision-Language-Action Model Inference via Dynamic-Static Layer-Skipping for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出DySL-VLA，解决机器人操作中VLA模型计算成本高的问题。通过动态跳过不重要层，提升效率并保持精度。**

- **链接: [https://arxiv.org/pdf/2602.22896](https://arxiv.org/pdf/2602.22896)**

> **作者:** Zebin Yang; Yijiahao Qi; Tong Xie; Bo Yu; Shaoshan Liu; Meng Li
>
> **备注:** DAC 2026
>
> **摘要:** Vision-Language-Action (VLA) models have shown remarkable success in robotic tasks like manipulation by fusing a language model's reasoning with a vision model's 3D understanding. However, their high computational cost remains a major obstacle for real-world applications that require real-time performance. We observe that the actions within a task have varying levels of importance: critical steps demand high precision, while less important ones can tolerate more variance. Leveraging this insight, we propose DySL-VLA, a novel framework that addresses computational cost by dynamically skipping VLA layers based on each action's importance. DySL-VLA categorizes its layers into two types: informative layers, which are consistently executed, and incremental layers, which can be selectively skipped. To intelligently skip layers without sacrificing accuracy, we invent a prior-post skipping guidance mechanism to determine when to initiate layer-skipping. We also propose a skip-aware two-stage knowledge distillation algorithm to efficiently train a standard VLA into a DySL-VLA. Our experiments indicate that DySL-VLA achieves 2.1% improvement in success length over Deer-VLA on the Calvin dataset, while simultaneously reducing trainable parameters by a factor of 85.7 and providing a 3.75x speedup relative to the RoboFlamingo baseline at iso-accuracy. Our code is available on this https URL.
>
---
#### [replaced 036] BEV-VLM: Trajectory Planning via Unified BEV Abstraction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶轨迹规划任务，解决传统方法依赖原始视觉数据的问题。通过融合多传感器数据生成BEV特征，并结合VLM进行轨迹规划，提升准确性和安全性。**

- **链接: [https://arxiv.org/pdf/2509.25249](https://arxiv.org/pdf/2509.25249)**

> **作者:** Guancheng Chen; Sheng Yang; Tong Zhan; Jian Wang
>
> **摘要:** This paper introduces BEV-VLM, a novel approach for trajectory planning in autonomous driving that leverages Vision-Language Models (VLMs) with Bird's-Eye View (BEV) feature maps as visual input. Unlike conventional trajectory planning approaches that rely solely on raw visual data (e.g., camera images), our method utilizes a highly compressed and informative BEV representation generated by fusing camera and LiDAR data, with subsequent alignment to High-Definition (HD) maps. This unified BEV-HD map format provides a geometrically consistent and semantically rich scene description, which enables VLMs to perform accurate and robust trajectory planning. Experimental results on the nuScenes dataset demonstrate that, compared with state-of-the-art vision-only methods, our approach achieves a 53.1% improvement in planning accuracy and realizes complete collision avoidance in evaluation scenarios. Our work highlights that VLMs can effectively interpret processed visual representations such as BEV features, expanding their applicability beyond raw image inputs for the task of trajectory planning.
>
---
#### [replaced 037] RoboMIND 2.0: A Multimodal, Bimanual Mobile Manipulation Dataset for Generalizable Embodied Intelligence
- **分类: cs.RO**

- **简介: 该论文提出RoboMIND 2.0数据集，解决机器人多模态、双臂移动操作的泛化问题，包含真实与仿真数据，支持复杂任务学习。**

- **链接: [https://arxiv.org/pdf/2512.24653](https://arxiv.org/pdf/2512.24653)**

> **作者:** Chengkai Hou; Kun Wu; Jiaming Liu; Zhengping Che; Di Wu; Fei Liao; Guangrun Li; Jingyang He; Qiuxuan Feng; Zhao Jin; Chenyang Gu; Zhuoyang Liu; Nuowei Han; Xiangju Mi; Yaoxu Lv; Yankai Fu; Gaole Dai; Langzhe Gu; Tao Li; Yuheng Zhang; Yixue Zhang; Xinhua Wang; Shichao Fan; Meng Li; Zhen Zhao; Ning Liu; Zhiyuan Xu; Pei Ren; Junjie Ji; Haonan Liu; Kuan Cheng; Shanghang Zhang; Jian Tang
>
> **摘要:** While data-driven imitation learning has revolutionized robotic manipulation, current approaches remain constrained by the scarcity of large-scale, diverse real-world demonstrations. Consequently, the ability of existing models to generalize across long-horizon bimanual tasks and mobile manipulation in unstructured environments remains limited. To bridge this gap, we present RoboMIND 2.0, a comprehensive real-world dataset comprising over 310K dual-arm manipulation trajectories collected across six distinct robot embodiments and 739 complex tasks. Crucially, to support research in contact-rich and spatially extended tasks, the dataset incorporates 12K tactile-enhanced episodes and 20K mobile manipulation trajectories. Complementing this physical data, we construct high-fidelity digital twins of our real-world environments, releasing an additional 20K-trajectory simulated dataset to facilitate robust sim-to-real transfer. To fully exploit the potential of RoboMIND 2.0, we propose MIND-2 system, a hierarchical dual-system frame-work optimized via offline reinforcement learning. MIND-2 integrates a high-level semantic planner (MIND-2-VLM) to decompose abstract natural language instructions into grounded subgoals, coupled with a low-level Vision-Language-Action executor (MIND-2-VLA), which generates precise, proprioception-aware motor actions.
>
---
