# 机器人 cs.RO

- **最新发布 56 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Force-Modulated Visual Policy for Robot-Assisted Dressing with Arm Motions
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出一种视觉策略，用于机器人辅助穿衣任务，解决衣物变形、力控制及肢体运动适应问题。通过模拟训练与少量真实数据微调，提升机器人对臂部动作的适应性与安全性，实验显示效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.12741v1](http://arxiv.org/pdf/2509.12741v1)**

> **作者:** Alexis Yihong Hao; Yufei Wang; Navin Sriram Ravie; Bharath Hegde; David Held; Zackory Erickson
>
> **备注:** CoRL 2025
>
> **摘要:** Robot-assisted dressing has the potential to significantly improve the lives of individuals with mobility impairments. To ensure an effective and comfortable dressing experience, the robot must be able to handle challenging deformable garments, apply appropriate forces, and adapt to limb movements throughout the dressing process. Prior work often makes simplifying assumptions -- such as static human limbs during dressing -- which limits real-world applicability. In this work, we develop a robot-assisted dressing system capable of handling partial observations with visual occlusions, as well as robustly adapting to arm motions during the dressing process. Given a policy trained in simulation with partial observations, we propose a method to fine-tune it in the real world using a small amount of data and multi-modal feedback from vision and force sensing, to further improve the policy's adaptability to arm motions and enhance safety. We evaluate our method in simulation with simplified articulated human meshes and in a real world human study with 12 participants across 264 dressing trials. Our policy successfully dresses two long-sleeve everyday garments onto the participants while being adaptive to various kinds of arm motions, and greatly outperforms prior baselines in terms of task completion and user feedback. Video are available at https://dressing-motion.github.io/.
>
---
#### [new 002] NavMoE: Hybrid Model- and Learning-based Traversability Estimation for Local Navigation via Mixture of Experts
- **分类: cs.RO**

- **简介: 该论文提出NavMoE模型，用于机器人局部导航中的可穿越性估计。针对不同地形，结合多种模型并动态选择，提升跨环境适应性与效率。通过懒惰门控机制减少计算成本，实现高效且鲁棒的导航。**

- **链接: [http://arxiv.org/pdf/2509.12747v1](http://arxiv.org/pdf/2509.12747v1)**

> **作者:** Botao He; Amir Hossein Shahidzadeh; Yu Chen; Jiayi Wu; Tianrui Guan; Guofei Chen; Howie Choset; Dinesh Manocha; Glen Chou; Cornelia Fermuller; Yiannis Aloimonos
>
> **摘要:** This paper explores traversability estimation for robot navigation. A key bottleneck in traversability estimation lies in efficiently achieving reliable and robust predictions while accurately encoding both geometric and semantic information across diverse environments. We introduce Navigation via Mixture of Experts (NAVMOE), a hierarchical and modular approach for traversability estimation and local navigation. NAVMOE combines multiple specialized models for specific terrain types, each of which can be either a classical model-based or a learning-based approach that predicts traversability for specific terrain types. NAVMOE dynamically weights the contributions of different models based on the input environment through a gating network. Overall, our approach offers three advantages: First, NAVMOE enables traversability estimation to adaptively leverage specialized approaches for different terrains, which enhances generalization across diverse and unseen environments. Second, our approach significantly improves efficiency with negligible cost of solution quality by introducing a training-free lazy gating mechanism, which is designed to minimize the number of activated experts during inference. Third, our approach uses a two-stage training strategy that enables the training for the gating networks within the hybrid MoE method that contains nondifferentiable modules. Extensive experiments show that NAVMOE delivers a better efficiency and performance balance than any individual expert or full ensemble across different domains, improving cross- domain generalization and reducing average computational cost by 81.2% via lazy gating, with less than a 2% loss in path quality.
>
---
#### [new 003] Bridging Perception and Planning: Towards End-to-End Planning for Signal Temporal Logic Tasks
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人任务与运动规划任务，旨在解决复杂环境下满足信号时序逻辑（STL）规范的路径规划问题。提出S-MSP框架，结合多视角视觉输入与STL约束，实现端到端轨迹生成，并通过结构感知MoE模型提升规划效果。**

- **链接: [http://arxiv.org/pdf/2509.12813v1](http://arxiv.org/pdf/2509.12813v1)**

> **作者:** Bowen Ye; Junyue Huang; Yang Liu; Xiaozhen Qiao; Xiang Yin
>
> **摘要:** We investigate the task and motion planning problem for Signal Temporal Logic (STL) specifications in robotics. Existing STL methods rely on pre-defined maps or mobility representations, which are ineffective in unstructured real-world environments. We propose the \emph{Structured-MoE STL Planner} (\textbf{S-MSP}), a differentiable framework that maps synchronized multi-view camera observations and an STL specification directly to a feasible trajectory. S-MSP integrates STL constraints within a unified pipeline, trained with a composite loss that combines trajectory reconstruction and STL robustness. A \emph{structure-aware} Mixture-of-Experts (MoE) model enables horizon-aware specialization by projecting sub-tasks into temporally anchored embeddings. We evaluate S-MSP using a high-fidelity simulation of factory-logistics scenarios with temporally constrained tasks. Experiments show that S-MSP outperforms single-expert baselines in STL satisfaction and trajectory feasibility. A rule-based \emph{safety filter} at inference improves physical executability without compromising logical correctness, showcasing the practicality of the approach.
>
---
#### [new 004] Spotting the Unfriendly Robot - Towards better Metrics for Interactions
- **分类: cs.RO**

- **简介: 该论文提出两种新指标（冲突强度与责任度），用于评估社交机器人导航中人机交互的质量与合作性，解决现有评价体系无法量化机器人协作行为的问题，推动SRN研究的标准化与安全效率提升。**

- **链接: [http://arxiv.org/pdf/2509.12912v1](http://arxiv.org/pdf/2509.12912v1)**

> **作者:** Raphael Wenzel; Malte Probst
>
> **备注:** Presented at 2025 IEEE Conference on Robotics and Automation (ICRA) Workshop: Advances in Social Navigation: Planning, HRI and Beyond
>
> **摘要:** Establishing standardized metrics for Social Robot Navigation (SRN) algorithms for assessing the quality and social compliance of robot behavior around humans is essential for SRN research. Currently, commonly used evaluation metrics lack the ability to quantify how cooperative an agent behaves in interaction with humans. Concretely, in a simple frontal approach scenario, no metric specifically captures if both agents cooperate or if one agent stays on collision course and the other agent is forced to evade. To address this limitation, we propose two new metrics, a conflict intensity metric and the responsibility metric. Together, these metrics are capable of evaluating the quality of human-robot interactions by showing how much a given algorithm has contributed to reducing a conflict and which agent actually took responsibility of the resolution. This work aims to contribute to the development of a comprehensive and standardized evaluation methodology for SRN, ultimately enhancing the safety, efficiency, and social acceptance of robots in human-centric environments.
>
---
#### [new 005] Unleashing the Power of Discrete-Time State Representation: Ultrafast Target-based IMU-Camera Spatial-Temporal Calibration
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种基于离散时间状态表示的高效IMU-相机时空标定方法，解决传统连续时间方法计算成本高的问题，提升标定效率，适用于大量视觉惯性设备的快速标定。**

- **链接: [http://arxiv.org/pdf/2509.12846v1](http://arxiv.org/pdf/2509.12846v1)**

> **作者:** Junlin Song; Antoine Richard; Miguel Olivares-Mendez
>
> **摘要:** Visual-inertial fusion is crucial for a large amount of intelligent and autonomous applications, such as robot navigation and augmented reality. To bootstrap and achieve optimal state estimation, the spatial-temporal displacements between IMU and cameras must be calibrated in advance. Most existing calibration methods adopt continuous-time state representation, more specifically the B-spline. Despite these methods achieve precise spatial-temporal calibration, they suffer from high computational cost caused by continuous-time state representation. To this end, we propose a novel and extremely efficient calibration method that unleashes the power of discrete-time state representation. Moreover, the weakness of discrete-time state representation in temporal calibration is tackled in this paper. With the increasing production of drones, cellphones and other visual-inertial platforms, if one million devices need calibration around the world, saving one minute for the calibration of each device means saving 2083 work days in total. To benefit both the research and industry communities, our code will be open-source.
>
---
#### [new 006] Contrastive Representation Learning for Robust Sim-to-Real Transfer of Adaptive Humanoid Locomotion
- **分类: cs.RO**

- **简介: 论文提出对比学习框架，解决人形机器人从仿真到现实的适应性运动问题。通过赋予纯本体感觉策略前瞻性能力，实现稳健且主动的步态控制，成功完成零样本仿真到真实环境的迁移。**

- **链接: [http://arxiv.org/pdf/2509.12858v1](http://arxiv.org/pdf/2509.12858v1)**

> **作者:** Yidan Lu; Rurui Yang; Qiran Kou; Mengting Chen; Tao Fan; Peter Cui; Yinzhao Dong; Peng Lu
>
> **摘要:** Reinforcement learning has produced remarkable advances in humanoid locomotion, yet a fundamental dilemma persists for real-world deployment: policies must choose between the robustness of reactive proprioceptive control or the proactivity of complex, fragile perception-driven systems. This paper resolves this dilemma by introducing a paradigm that imbues a purely proprioceptive policy with proactive capabilities, achieving the foresight of perception without its deployment-time costs. Our core contribution is a contrastive learning framework that compels the actor's latent state to encode privileged environmental information from simulation. Crucially, this ``distilled awareness" empowers an adaptive gait clock, allowing the policy to proactively adjust its rhythm based on an inferred understanding of the terrain. This synergy resolves the classic trade-off between rigid, clocked gaits and unstable clock-free policies. We validate our approach with zero-shot sim-to-real transfer to a full-sized humanoid, demonstrating highly robust locomotion over challenging terrains, including 30 cm high steps and 26.5{\deg} slopes, proving the effectiveness of our method. Website: https://lu-yidan.github.io/cra-loco.
>
---
#### [new 007] Toward Ownership Understanding of Objects: Active Question Generation with Large Language Model and Probabilistic Generative Model
- **分类: cs.RO; cs.AI; cs.HC; cs.LG**

- **简介: 该论文提出ActOwL框架，解决机器人理解物体所有权的问题。通过主动生成问题并结合LLM常识知识，提升所有权学习效率。实验表明其在准确率和效率上优于基线方法。属于机器人认知与人机交互任务。**

- **链接: [http://arxiv.org/pdf/2509.12754v1](http://arxiv.org/pdf/2509.12754v1)**

> **作者:** Saki Hashimoto; Shoichi Hasegawa; Tomochika Ishikawa; Akira Taniguchi; Yoshinobu Hagiwara; Lotfi El Hafi; Tadahiro Taniguchi
>
> **备注:** Submitted to AROB-ISBC 2026 (Journal Track option)
>
> **摘要:** Robots operating in domestic and office environments must understand object ownership to correctly execute instructions such as ``Bring me my cup.'' However, ownership cannot be reliably inferred from visual features alone. To address this gap, we propose Active Ownership Learning (ActOwL), a framework that enables robots to actively generate and ask ownership-related questions to users. ActOwL employs a probabilistic generative model to select questions that maximize information gain, thereby acquiring ownership knowledge efficiently to improve learning efficiency. Additionally, by leveraging commonsense knowledge from Large Language Models (LLM), objects are pre-classified as either shared or owned, and only owned objects are targeted for questioning. Through experiments in a simulated home environment and a real-world laboratory setting, ActOwL achieved significantly higher ownership clustering accuracy with fewer questions than baseline methods. These findings demonstrate the effectiveness of combining active inference with LLM-guided commonsense reasoning, advancing the capability of robots to acquire ownership knowledge for practical and socially appropriate task execution.
>
---
#### [new 008] Hydrosoft: Non-Holonomic Hydroelastic Models for Compliant Tactile Manipulation
- **分类: cs.RO**

- **简介: 论文提出Hydrosoft模型，解决触觉传感器在非完整系统中的复杂动力学建模问题。通过扩展状态空间并引入分布式力，实现路径依赖接触力的高效模拟，支持梯度优化与高精度触觉反馈集成。**

- **链接: [http://arxiv.org/pdf/2509.13126v1](http://arxiv.org/pdf/2509.13126v1)**

> **作者:** Miquel Oller; An Dang; Nima Fazeli
>
> **摘要:** Tactile sensors have long been valued for their perceptual capabilities, offering rich insights into the otherwise hidden interface between the robot and grasped objects. Yet their inherent compliance -- a key driver of force-rich interactions -- remains underexplored. The central challenge is to capture the complex, nonlinear dynamics introduced by these passive-compliant elements. Here, we present a computationally efficient non-holonomic hydroelastic model that accurately models path-dependent contact force distributions and dynamic surface area variations. Our insight is to extend the object's state space, explicitly incorporating the distributed forces generated by the compliant sensor. Our differentiable formulation not only accounts for path-dependent behavior but also enables gradient-based trajectory optimization, seamlessly integrating with high-resolution tactile feedback. We demonstrate the effectiveness of our approach across a range of simulated and real-world experiments and highlight the importance of modeling the path dependence of sensor dynamics.
>
---
#### [new 009] Geometric Red-Teaming for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出Geometric Red-Teaming（GRT）框架，通过几何扰动评估机器人操作策略的鲁棒性，生成触发失败的CrashShapes，并通过蓝队优化提升性能。属于机器人操作鲁棒性评估任务，解决传统测试集无法发现潜在失效模式的问题。**

- **链接: [http://arxiv.org/pdf/2509.12379v1](http://arxiv.org/pdf/2509.12379v1)**

> **作者:** Divyam Goel; Yufei Wang; Tiancheng Wu; Guixiu Qiao; Pavel Piliptchak; David Held; Zackory Erickson
>
> **备注:** Accepted at the 9th Annual Conference on Robot Learning (CoRL 2025, Oral)
>
> **摘要:** Standard evaluation protocols in robotic manipulation typically assess policy performance over curated, in-distribution test sets, offering limited insight into how systems fail under plausible variation. We introduce Geometric Red-Teaming (GRT), a red-teaming framework that probes robustness through object-centric geometric perturbations, automatically generating CrashShapes -- structurally valid, user-constrained mesh deformations that trigger catastrophic failures in pre-trained manipulation policies. The method integrates a Jacobian field-based deformation model with a gradient-free, simulator-in-the-loop optimization strategy. Across insertion, articulation, and grasping tasks, GRT consistently discovers deformations that collapse policy performance, revealing brittle failure modes missed by static benchmarks. By combining task-level policy rollouts with constraint-aware shape exploration, we aim to build a general purpose framework for structured, object-centric robustness evaluation in robotic manipulation. We additionally show that fine-tuning on individual CrashShapes, a process we refer to as blue-teaming, improves task success by up to 60 percentage points on those shapes, while preserving performance on the original object, demonstrating the utility of red-teamed geometries for targeted policy refinement. Finally, we validate both red-teaming and blue-teaming results with a real robotic arm, observing that simulated CrashShapes reduce task success from 90% to as low as 22.5%, and that blue-teaming recovers performance to up to 90% on the corresponding real-world geometry -- closely matching simulation outcomes. Videos and code can be found on our project website: https://georedteam.github.io/ .
>
---
#### [new 010] MoiréTac: A Dual-Mode Visuotactile Sensor for Multidimensional Perception Using Moiré Pattern Amplification
- **分类: cs.RO; eess.SP**

- **简介: 论文提出MoiréTac，一种双模式视觉触觉传感器，通过莫尔条纹放大微变形，实现高精度6轴力/扭矩测量与视觉感知。解决传统传感器分辨率低、力-图像关系不明确的问题，应用于机器人灵巧操作。**

- **链接: [http://arxiv.org/pdf/2509.12714v1](http://arxiv.org/pdf/2509.12714v1)**

> **作者:** Kit-Wa Sou; Junhao Gong; Shoujie Li; Chuqiao Lyu; Ziwu Song; Shilong Mu; Wenbo Ding
>
> **摘要:** Visuotactile sensors typically employ sparse marker arrays that limit spatial resolution and lack clear analytical force-to-image relationships. To solve this problem, we present \textbf{Moir\'eTac}, a dual-mode sensor that generates dense interference patterns via overlapping micro-gratings within a transparent architecture. When two gratings overlap with misalignment, they create moir\'e patterns that amplify microscopic deformations. The design preserves optical clarity for vision tasks while producing continuous moir\'e fields for tactile sensing, enabling simultaneous 6-axis force/torque measurement, contact localization, and visual perception. We combine physics-based features (brightness, phase gradient, orientation, and period) from moir\'e patterns with deep spatial features. These are mapped to 6-axis force/torque measurements, enabling interpretable regression through end-to-end learning. Experimental results demonstrate three capabilities: force/torque measurement with R^2 > 0.98 across tested axes; sensitivity tuning through geometric parameters (threefold gain adjustment); and vision functionality for object classification despite moir\'e overlay. Finally, we integrate the sensor into a robotic arm for cap removal with coordinated force and torque control, validating its potential for dexterous manipulation.
>
---
#### [new 011] Empowering Multi-Robot Cooperation via Sequential World Models
- **分类: cs.RO**

- **简介: 论文提出SeqWM框架，解决多机器人协作中联合动态复杂的问题。通过独立的序列化世界模型分解动态，实现意图共享与高效协作。实验显示其在性能和样本效率上优于现有方法，并成功部署于实体机器人。**

- **链接: [http://arxiv.org/pdf/2509.13095v1](http://arxiv.org/pdf/2509.13095v1)**

> **作者:** Zijie Zhao; Honglei Guo; Shengqian Chen; Kaixuan Xu; Bo Jiang; Yuanheng Zhu; Dongbin Zhao
>
> **摘要:** Model-based reinforcement learning (MBRL) has shown significant potential in robotics due to its high sample efficiency and planning capability. However, extending MBRL to multi-robot cooperation remains challenging due to the complexity of joint dynamics. To address this, we propose the Sequential World Model (SeqWM), a novel framework that integrates the sequential paradigm into model-based multi-agent reinforcement learning. SeqWM employs independent, sequentially structured agent-wise world models to decompose complex joint dynamics. Latent rollouts and decision-making are performed through sequential communication, where each agent generates its future trajectory and plans its actions based on the predictions of its predecessors. This design enables explicit intention sharing, enhancing cooperative performance, and reduces communication overhead to linear complexity. Results in challenging simulated environments (Bi-DexHands and Multi-Quad) show that SeqWM outperforms existing state-of-the-art model-free and model-based baselines in both overall performance and sample efficiency, while exhibiting advanced cooperative behaviors such as predictive adaptation and role division. Furthermore, SeqWM has been success fully deployed on physical quadruped robots, demonstrating its effectiveness in real-world multi-robot systems. Demos and code are available at: https://github.com/zhaozijie2022/seqwm-marl
>
---
#### [new 012] A Design Co-Pilot for Task-Tailored Manipulators
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种自动设计任务定制机械臂的方法，解决通用机械臂性能不佳与定制开发周期长的问题。通过可微框架加速设计过程，实现快速优化与人机协作，提升机械臂在复杂环境中的适应能力。**

- **链接: [http://arxiv.org/pdf/2509.13077v1](http://arxiv.org/pdf/2509.13077v1)**

> **作者:** Jonathan Külz; Sehoon Ha; Matthias Althoff
>
> **摘要:** Although robotic manipulators are used in an ever-growing range of applications, robot manufacturers typically follow a ``one-fits-all'' philosophy, employing identical manipulators in various settings. This often leads to suboptimal performance, as general-purpose designs fail to exploit particularities of tasks. The development of custom, task-tailored robots is hindered by long, cost-intensive development cycles and the high cost of customized hardware. Recently, various computational design methods have been devised to overcome the bottleneck of human engineering. In addition, a surge of modular robots allows quick and economical adaptation to changing industrial settings. This work proposes an approach to automatically designing and optimizing robot morphologies tailored to a specific environment. To this end, we learn the inverse kinematics for a wide range of different manipulators. A fully differentiable framework realizes gradient-based fine-tuning of designed robots and inverse kinematics solutions. Our generative approach accelerates the generation of specialized designs from hours with optimization-based methods to seconds, serving as a design co-pilot that enables instant adaptation and effective human-AI collaboration. Numerical experiments show that our approach finds robots that can navigate cluttered environments, manipulators that perform well across a specified workspace, and can be adapted to different hardware constraints. Finally, we demonstrate the real-world applicability of our method by setting up a modular robot designed in simulation that successfully moves through an obstacle course.
>
---
#### [new 013] Collaborative Loco-Manipulation for Pick-and-Place Tasks with Dynamic Reward Curriculum
- **分类: cs.RO**

- **简介: 该论文提出一种分层强化学习方法，用于训练单臂腿式机器人完成抓取与放置任务。通过动态奖励课程提升训练效率，并实现单双机器人协作。解决了复杂任务中长期策略学习与协调控制问题。**

- **链接: [http://arxiv.org/pdf/2509.13239v1](http://arxiv.org/pdf/2509.13239v1)**

> **作者:** Tianxu An; Flavio De Vincenti; Yuntao Ma; Marco Hutter; Stelian Coros
>
> **摘要:** We present a hierarchical RL pipeline for training one-armed legged robots to perform pick-and-place (P&P) tasks end-to-end -- from approaching the payload to releasing it at a target area -- in both single-robot and cooperative dual-robot settings. We introduce a novel dynamic reward curriculum that enables a single policy to efficiently learn long-horizon P&P operations by progressively guiding the agents through payload-centered sub-objectives. Compared to state-of-the-art approaches for long-horizon RL tasks, our method improves training efficiency by 55% and reduces execution time by 18.6% in simulation experiments. In the dual-robot case, we show that our policy enables each robot to attend to different components of its observation space at distinct task stages, promoting effective coordination via autonomous attention shifts. We validate our method through real-world experiments using ANYmal D platforms in both single- and dual-robot scenarios. To our knowledge, this is the first RL pipeline that tackles the full scope of collaborative P&P with two legged manipulators.
>
---
#### [new 014] Responsibility and Engagement - Evaluating Interactions in Social Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于社会机器人导航任务，旨在评估人机交互中的冲突解决行为。提出责任与参与度两个指标，用于量化代理在冲突中的贡献与加剧程度，并通过模拟验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.12890v1](http://arxiv.org/pdf/2509.12890v1)**

> **作者:** Malte Probst; Raphael Wenzel; Monica Dasi
>
> **备注:** under review for 2026 IEEE International Conference on Robotics & Automation (ICRA)
>
> **摘要:** In Social Robot Navigation (SRN), the availability of meaningful metrics is crucial for evaluating trajectories from human-robot interactions. In the SRN context, such interactions often relate to resolving conflicts between two or more agents. Correspondingly, the shares to which agents contribute to the resolution of such conflicts are important. This paper builds on recent work, which proposed a Responsibility metric capturing such shares. We extend this framework in two directions: First, we model the conflict buildup phase by introducing a time normalization. Second, we propose the related Engagement metric, which captures how the agents' actions intensify a conflict. In a comprehensive series of simulated scenarios with dyadic, group and crowd interactions, we show that the metrics carry meaningful information about the cooperative resolution of conflicts in interactions. They can be used to assess behavior quality and foresightedness. We extensively discuss applicability, design choices and limitations of the proposed metrics.
>
---
#### [new 015] Computing forward statics from tendon-length in flexible-joint hyper-redundant manipulators
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文研究柔性关节超冗余绳驱机械臂的正静力学计算问题，提出基于螺钉理论的建模方法和迭代求解策略，利用肌腱长度或张力作为输入，实现开环控制，避免依赖张力测量和状态估计。**

- **链接: [http://arxiv.org/pdf/2509.12444v1](http://arxiv.org/pdf/2509.12444v1)**

> **作者:** Weiting Feng; Kyle L. Walker; Yunjie Yang; Francesco Giorgio-Serchi
>
> **备注:** To be presented at IROS 2025, Hangzhou, China
>
> **摘要:** Hyper-redundant tendon-driven manipulators of- fer greater flexibility and compliance over traditional manipu- lators. A common way of controlling such manipulators relies on adjusting tendon lengths, which is an accessible control parameter. This approach works well when the kinematic configuration is representative of the real operational con- ditions. However, when dealing with manipulators of larger size subject to gravity, it becomes necessary to solve a static force problem, using tendon force as the input and employing a mapping from the configuration space to retrieve tendon length. Alternatively, measurements of the manipulator posture can be used to iteratively adjust tendon lengths to achieve a desired posture. Hence, either tension measurement or state estimation of the manipulator are required, both of which are not always accurately available. Here, we propose a solution by reconciling cables tension and length as the input for the solution of the system forward statics. We develop a screw-based formulation for a tendon-driven, multi-segment, hyper-redundant manipulator with elastic joints and introduce a forward statics iterative solution method that equivalently makes use of either tendon length or tension as the input. This strategy is experimentally validated using a traditional tension input first, subsequently showing the efficacy of the method when exclusively tendon lengths are used. The results confirm the possibility to perform open-loop control in static conditions using a kinematic input only, thus bypassing some of the practical problems with tension measurement and state estimation of hyper-redundant systems.
>
---
#### [new 016] Robust Online Residual Refinement via Koopman-Guided Dynamics Modeling
- **分类: cs.RO**

- **简介: 该论文属于强化学习中的模仿学习任务，旨在解决长时域任务和高精度控制中的误差累积问题。提出KORR框架，结合Koopman理论建模全局动态，指导残差策略更新，提升鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.12562v1](http://arxiv.org/pdf/2509.12562v1)**

> **作者:** Zhefei Gong; Shangke Lyu; Pengxiang Ding; Wei Xiao; Donglin Wang
>
> **摘要:** Imitation learning (IL) enables efficient skill acquisition from demonstrations but often struggles with long-horizon tasks and high-precision control due to compounding errors. Residual policy learning offers a promising, model-agnostic solution by refining a base policy through closed-loop corrections. However, existing approaches primarily focus on local corrections to the base policy, lacking a global understanding of state evolution, which limits robustness and generalization to unseen scenarios. To address this, we propose incorporating global dynamics modeling to guide residual policy updates. Specifically, we leverage Koopman operator theory to impose linear time-invariant structure in a learned latent space, enabling reliable state transitions and improved extrapolation for long-horizon prediction and unseen environments. We introduce KORR (Koopman-guided Online Residual Refinement), a simple yet effective framework that conditions residual corrections on Koopman-predicted latent states, enabling globally informed and stable action refinement. We evaluate KORR on long-horizon, fine-grained robotic furniture assembly tasks under various perturbations. Results demonstrate consistent gains in performance, robustness, and generalization over strong baselines. Our findings further highlight the potential of Koopman-based modeling to bridge modern learning methods with classical control theory.
>
---
#### [new 017] ActiveVLN: Towards Active Exploration via Multi-Turn RL in Vision-and-Language Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉与语言导航（VLN）任务，旨在解决传统方法依赖模仿学习、数据成本高及探索能力不足的问题。提出ActiveVLN框架，通过多轮强化学习实现主动探索，提升导航性能并减少对专家轨迹的依赖。**

- **链接: [http://arxiv.org/pdf/2509.12618v1](http://arxiv.org/pdf/2509.12618v1)**

> **作者:** Zekai Zhang; Weiye Zhu; Hewei Pan; Xiangchen Wang; Rongtao Xu; Xing Sun; Feng Zheng
>
> **摘要:** The Vision-and-Language Navigation (VLN) task requires an agent to follow natural language instructions and navigate through complex environments. Existing MLLM-based VLN methods primarily rely on imitation learning (IL) and often use DAgger for post-training to mitigate covariate shift. While effective, these approaches incur substantial data collection and training costs. Reinforcement learning (RL) offers a promising alternative. However, prior VLN RL methods lack dynamic interaction with the environment and depend on expert trajectories for reward shaping, rather than engaging in open-ended active exploration. This restricts the agent's ability to discover diverse and plausible navigation routes. To address these limitations, we propose ActiveVLN, a VLN framework that explicitly enables active exploration through multi-turn RL. In the first stage, a small fraction of expert trajectories is used for IL to bootstrap the agent. In the second stage, the agent iteratively predicts and executes actions, automatically collects diverse trajectories, and optimizes multiple rollouts via the GRPO objective. To further improve RL efficiency, we introduce a dynamic early-stopping strategy to prune long-tail or likely failed trajectories, along with additional engineering optimizations. Experiments show that ActiveVLN achieves the largest performance gains over IL baselines compared to both DAgger-based and prior RL-based post-training methods, while reaching competitive performance with state-of-the-art approaches despite using a smaller model. Code and data will be released soon.
>
---
#### [new 018] Spatiotemporal Calibration for Laser Vision Sensor in Hand-eye System Based on Straight-line Constraint
- **分类: cs.RO**

- **简介: 论文提出一种基于直线约束的时空标定方法，解决激光视觉传感器在手眼系统中因相机时延和参数漂移导致的时空不同步问题，通过非线性优化提升焊接轨迹精度。**

- **链接: [http://arxiv.org/pdf/2509.12928v1](http://arxiv.org/pdf/2509.12928v1)**

> **作者:** Peiwen Yang; Mingquan Jiang; Xinyue Shen; Heping Zhang
>
> **备注:** Submitted to IEEE RAL
>
> **摘要:** Laser vision sensors (LVS) are critical perception modules for industrial robots, facilitating real-time acquisition of workpiece geometric data in welding applications. However, the camera communication delay will lead to a temporal desynchronization between captured images and the robot motions. Additionally, hand-eye extrinsic parameters may vary during prolonged measurement. To address these issues, we introduce a measurement model of LVS considering the effect of the camera's time-offset and propose a teaching-free spatiotemporal calibration method utilizing line constraints. This method involves a robot equipped with an LVS repeatedly scanning straight-line fillet welds using S-shaped trajectories. Regardless of the robot's orientation changes, all measured welding positions are constrained to a straight-line, represented by Plucker coordinates. Moreover, a nonlinear optimization model based on straight-line constraints is established. Subsequently, the Levenberg-Marquardt algorithm (LMA) is employed to optimize parameters, including time-offset, hand-eye extrinsic parameters, and straight-line parameters. The feasibility and accuracy of the proposed approach are quantitatively validated through experiments on curved weld scanning. We open-sourced the code, dataset, and simulation report at https://anonymous.4open.science/r/LVS_ST_CALIB-015F/README.md.
>
---
#### [new 019] Distributed Event-Triggered Distance-Based Formation Control for Multi-Agent Systems
- **分类: cs.RO**

- **简介: 论文研究多智能体系统的分布式事件触发编队控制，解决资源受限下的协同编队问题。提出基于距离测量的事件触发控制器，仅在误差超限时更新控制，减少控制努力。通过仿真与实验验证其有效性与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.12390v1](http://arxiv.org/pdf/2509.12390v1)**

> **作者:** Evangelos Psomiadis; Panagiotis Tsiotras
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** This paper addresses the problem of collaborative formation control for multi-agent systems with limited resources. We consider a team of robots tasked with achieving a desired formation from arbitrary initial configurations. To reduce unnecessary control updates and conserve resources, we propose a distributed event-triggered formation controller that relies on inter-agent distance measurements. Control updates are triggered only when the measurement error exceeds a predefined threshold, ensuring system stability. The proposed controller is validated through extensive simulations and real-world experiments involving different formations, communication topologies, scalability tests, and variations in design parameters, while also being compared against periodic triggering strategies. Results demonstrate that the event-triggered approach significantly reduces control efforts while preserving formation performance.
>
---
#### [new 020] HARMONIC: A Content-Centric Cognitive Robotic Architecture
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 论文提出HARMONIC认知机器人架构，用于人机团队中的机器人。该架构支持语义感知、类人决策与语言交流，解决数据稀缺、可解释性与安全性问题。通过两个系统验证其在仿真与实体平台的有效性。属于机器人认知架构设计任务。**

- **链接: [http://arxiv.org/pdf/2509.13279v1](http://arxiv.org/pdf/2509.13279v1)**

> **作者:** Sanjay Oruganti; Sergei Nirenburg; Marjorie McShane; Jesse English; Michael K. Roberts; Christian Arndt; Carlos Gonzalez; Mingyo Seo; Luis Sentis
>
> **摘要:** This paper introduces HARMONIC, a cognitive-robotic architecture designed for robots in human-robotic teams. HARMONIC supports semantic perception interpretation, human-like decision-making, and intentional language communication. It addresses the issues of safety and quality of results; aims to solve problems of data scarcity, explainability, and safety; and promotes transparency and trust. Two proof-of-concept HARMONIC-based robotic systems are demonstrated, each implemented in both a high-fidelity simulation environment and on physical robotic platforms.
>
---
#### [new 021] GRATE: a Graph transformer-based deep Reinforcement learning Approach for Time-efficient autonomous robot Exploration
- **分类: cs.RO**

- **简介: 该论文提出GRATE方法，用于提升自主机器人探索效率。针对传统方法在图结构数据推理和时间效率上的不足，采用图Transformer增强环境理解，并结合卡尔曼滤波优化路径可行性，实验表明其在距离和时间上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.12863v1](http://arxiv.org/pdf/2509.12863v1)**

> **作者:** Haozhan Ni; Jingsong Liang; Chenyu He; Yuhong Cao; Guillaume Sartoretti
>
> **摘要:** Autonomous robot exploration (ARE) is the process of a robot autonomously navigating and mapping an unknown environment. Recent Reinforcement Learning (RL)-based approaches typically formulate ARE as a sequential decision-making problem defined on a collision-free informative graph. However, these methods often demonstrate limited reasoning ability over graph-structured data. Moreover, due to the insufficient consideration of robot motion, the resulting RL policies are generally optimized to minimize travel distance, while neglecting time efficiency. To overcome these limitations, we propose GRATE, a Deep Reinforcement Learning (DRL)-based approach that leverages a Graph Transformer to effectively capture both local structure patterns and global contextual dependencies of the informative graph, thereby enhancing the model's reasoning capability across the entire environment. In addition, we deploy a Kalman filter to smooth the waypoint outputs, ensuring that the resulting path is kinodynamically feasible for the robot to follow. Experimental results demonstrate that our method exhibits better exploration efficiency (up to 21.5% in distance and 21.3% in time to complete exploration) than state-of-the-art conventional and learning-based baselines in various simulation benchmarks. We also validate our planner in real-world scenarios.
>
---
#### [new 022] PerchMobi^3: A Multi-Modal Robot with Power-Reuse Quad-Fan Mechanism for Air-Ground-Wall Locomotion
- **分类: cs.RO**

- **简介: 论文提出PerchMobi^3机器人，通过四风扇实现空中、地面和墙面的多模态移动。解决传统设计复杂、效率低的问题，利用风扇同时提供推力和吸附力，无需额外泵装置，验证了其在多种场景下的可行性。**

- **链接: [http://arxiv.org/pdf/2509.12620v1](http://arxiv.org/pdf/2509.12620v1)**

> **作者:** Yikai Chen; Zhi Zheng; Jin Wang; Bingye He; Xiangyu Xu; Jialu Zhang; Huan Yu; Guodong Lu
>
> **备注:** 7 pages, 8 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Achieving seamless integration of aerial flight, ground driving, and wall climbing within a single robotic platform remains a major challenge, as existing designs often rely on additional adhesion actuators that increase complexity, reduce efficiency, and compromise reliability. To address these limitations, we present PerchMobi^3, a quad-fan, negative-pressure, air-ground-wall robot that implements a propulsion-adhesion power-reuse mechanism. By repurposing four ducted fans to simultaneously provide aerial thrust and negative-pressure adhesion, and integrating them with four actively driven wheels, PerchMobi^3 eliminates dedicated pumps while maintaining a lightweight and compact design. To the best of our knowledge, this is the first quad-fan prototype to demonstrate functional power reuse for multi-modal locomotion. A modeling and control framework enables coordinated operation across ground, wall, and aerial domains with fan-assisted transitions. The feasibility of the design is validated through a comprehensive set of experiments covering ground driving, payload-assisted wall climbing, aerial flight, and cross-mode transitions, demonstrating robust adaptability across locomotion scenarios. These results highlight the potential of PerchMobi^3 as a novel design paradigm for multi-modal robotic mobility, paving the way for future extensions toward autonomous and application-oriented deployment.
>
---
#### [new 023] Multi-Robot Task Planning for Multi-Object Retrieval Tasks with Distributed On-Site Knowledge via Large Language Models
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文研究多机器人协同完成多对象检索任务，解决如何根据各机器人局部知识分配子任务的问题。提出基于大语言模型的框架，实现指令分解与任务分配，实验表明其优于随机和常识分配方法。**

- **链接: [http://arxiv.org/pdf/2509.12838v1](http://arxiv.org/pdf/2509.12838v1)**

> **作者:** Kento Murata; Shoichi Hasegawa; Tomochika Ishikawa; Yoshinobu Hagiwara; Akira Taniguchi; Lotfi El Hafi; Tadahiro Taniguchi
>
> **备注:** Submitted to AROB-ISBC 2026 (Journal Track option)
>
> **摘要:** It is crucial to efficiently execute instructions such as "Find an apple and a banana" or "Get ready for a field trip," which require searching for multiple objects or understanding context-dependent commands. This study addresses the challenging problem of determining which robot should be assigned to which part of a task when each robot possesses different situational on-site knowledge-specifically, spatial concepts learned from the area designated to it by the user. We propose a task planning framework that leverages large language models (LLMs) and spatial concepts to decompose natural language instructions into subtasks and allocate them to multiple robots. We designed a novel few-shot prompting strategy that enables LLMs to infer required objects from ambiguous commands and decompose them into appropriate subtasks. In our experiments, the proposed method achieved 47/50 successful assignments, outperforming random (28/50) and commonsense-based assignment (26/50). Furthermore, we conducted qualitative evaluations using two actual mobile manipulators. The results demonstrated that our framework could handle instructions, including those involving ad hoc categories such as "Get ready for a field trip," by successfully performing task decomposition, assignment, sequential planning, and execution.
>
---
#### [new 024] Practical Handling of Dynamic Environments in Decentralised Multi-Robot Patrol
- **分类: cs.RO**

- **简介: 论文研究去中心化多机器人巡逻任务中动态环境的处理方法，旨在提升巡逻效率与适应性。提出新方法，在高度动态环境中显著优于现有方案，并探讨了无需考虑环境动态的场景。**

- **链接: [http://arxiv.org/pdf/2509.13069v1](http://arxiv.org/pdf/2509.13069v1)**

> **作者:** James C. Ward; Arthur Richards; Edmund R. Hunt
>
> **摘要:** Persistent monitoring using robot teams is of interest in fields such as security, environmental monitoring, and disaster recovery. Performing such monitoring in a fully on-line decentralised fashion has significant potential advantages for robustness, adaptability, and scalability of monitoring solutions, including, in principle, the capacity to effectively adapt in real-time to a changing environment. We examine this through the lens of multi-robot patrol, in which teams of patrol robots must persistently minimise time between visits to points of interest, within environments where traversability of routes is highly dynamic. These dynamics must be observed by patrol agents and accounted for in a fully decentralised on-line manner. In this work, we present a new method of monitoring and adjusting for environment dynamics in a decentralised multi-robot patrol team. We demonstrate that our method significantly outperforms realistic baselines in highly dynamic scenarios, and also investigate dynamic scenarios in which explicitly accounting for environment dynamics may be unnecessary or impractical.
>
---
#### [new 025] Deep Generative and Discriminative Digital Twin endowed with Variational Autoencoder for Unsupervised Predictive Thermal Condition Monitoring of Physical Robots in Industry 6.0 and Society 6.0
- **分类: cs.RO; cs.AI; cs.ET; cs.LG; cs.SY; eess.SY**

- **简介: 论文提出一种基于变分自编码器的数字孪生系统，用于工业6.0和社 会6.0中机器人热状态的无监督预测监控。解决传统机器人过热导致停机的问题，通过生成异常热状态并预测运动可行性，提升机器人自主适应与安全性。**

- **链接: [http://arxiv.org/pdf/2509.12740v1](http://arxiv.org/pdf/2509.12740v1)**

> **作者:** Eric Guiffo Kaigom
>
> **备注:** $\copyright$ 2025 the authors. This work has been accepted to the to the 10th IFAC Symposium on Mechatronic Systems & 14th IFAC Symposium on Robotics July 15-18, 2025 || Paris, France for publication under a Creative Commons Licence CC-BY-NC-ND
>
> **摘要:** Robots are unrelentingly used to achieve operational efficiency in Industry 4.0 along with symbiotic and sustainable assistance for the work-force in Industry 5.0. As resilience, robustness, and well-being are required in anti-fragile manufacturing and human-centric societal tasks, an autonomous anticipation and adaption to thermal saturation and burns due to motors overheating become instrumental for human safety and robot availability. Robots are thereby expected to self-sustain their performance and deliver user experience, in addition to communicating their capability to other agents in advance to ensure fully automated thermally feasible tasks, and prolong their lifetime without human intervention. However, the traditional robot shutdown, when facing an imminent thermal saturation, inhibits productivity in factories and comfort in the society, while cooling strategies are hard to implement after the robot acquisition. In this work, smart digital twins endowed with generative AI, i.e., variational autoencoders, are leveraged to manage thermally anomalous and generate uncritical robot states. The notion of thermal difficulty is derived from the reconstruction error of variational autoencoders. A robot can use this score to predict, anticipate, and share the thermal feasibility of desired motion profiles to meet requirements from emerging applications in Industry 6.0 and Society 6.0.
>
---
#### [new 026] Deep Learning for Model-Free Prediction of Thermal States of Robot Joint Motors
- **分类: cs.RO; cs.AI; cs.ET; cs.LG; cs.SY; eess.SY**

- **简介: 论文提出一种基于深度学习的模型-free方法，利用LSTM和前馈网络预测机器人关节电机的热状态。通过采集关节扭矩数据，解决传统模型参数复杂、不确定性高的问题，实现对七自由度冗余机器人的温度动态有效预测。**

- **链接: [http://arxiv.org/pdf/2509.12739v1](http://arxiv.org/pdf/2509.12739v1)**

> **作者:** Trung Kien La; Eric Guiffo Kaigom
>
> **备注:** $\copyright$ 2025 the authors. This work has been accepted to the 10th IFAC Symposium on Mechatronic Systems & 14th IFAC Symposium on Robotics July 15-18, 2025 || Paris, France for publication under a Creative Commons Licence CC-BY-NC-ND
>
> **摘要:** In this work, deep neural networks made up of multiple hidden Long Short-Term Memory (LSTM) and Feedforward layers are trained to predict the thermal behavior of the joint motors of robot manipulators. A model-free and scalable approach is adopted. It accommodates complexity and uncertainty challenges stemming from the derivation, identification, and validation of a large number of parameters of an approximation model that is hardly available. To this end, sensed joint torques are collected and processed to foresee the thermal behavior of joint motors. Promising prediction results of the machine learning based capture of the temperature dynamics of joint motors of a redundant robot with seven joints are presented.
>
---
#### [new 027] Pre-trained Visual Representations Generalize Where it Matters in Model-Based Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY; 68T07, 68T40 (Primary) 93C85, 62L20 (Secondary); I.2.6; I.2.9; I.4.8; F.2.2**

- **简介: 该论文研究预训练视觉模型（PVMs）在基于模型的强化学习（MBRL）中的有效性，旨在提升视觉策略在视觉域变化下的泛化能力。通过实验发现，部分微调的PVMs在极端分布偏移下表现最佳，证明其在模型基于机器人学习中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2509.12531v1](http://arxiv.org/pdf/2509.12531v1)**

> **作者:** Scott Jones; Liyou Zhou; Sebastian W. Pattinson
>
> **摘要:** In visuomotor policy learning, the control policy for the robotic agent is derived directly from visual inputs. The typical approach, where a policy and vision encoder are trained jointly from scratch, generalizes poorly to novel visual scene changes. Using pre-trained vision models (PVMs) to inform a policy network improves robustness in model-free reinforcement learning (MFRL). Recent developments in Model-based reinforcement learning (MBRL) suggest that MBRL is more sample-efficient than MFRL. However, counterintuitively, existing work has found PVMs to be ineffective in MBRL. Here, we investigate PVM's effectiveness in MBRL, specifically on generalization under visual domain shifts. We show that, in scenarios with severe shifts, PVMs perform much better than a baseline model trained from scratch. We further investigate the effects of varying levels of fine-tuning of PVMs. Our results show that partial fine-tuning can maintain the highest average task performance under the most extreme distribution shifts. Our results demonstrate that PVMs are highly successful in promoting robustness in visual policy learning, providing compelling evidence for their wider adoption in model-based robotic learning applications.
>
---
#### [new 028] Safety filtering of robotic manipulation under environment uncertainty: a computational approach
- **分类: cs.RO**

- **简介: 论文提出一种基于物理的机器人安全过滤方法，用于动态不确定环境中的操作任务。通过高保真模拟评估控制策略，结合密集滚动和关键状态稀疏重评估，有效识别并过滤不安全轨迹，提升机器人在参数不确定情况下的操作安全性。**

- **链接: [http://arxiv.org/pdf/2509.12674v1](http://arxiv.org/pdf/2509.12674v1)**

> **作者:** Anna Johansson; Daniel Lindmark; Viktor Wiberg; Martin Servin
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Robotic manipulation in dynamic and unstructured environments requires safety mechanisms that exploit what is known and what is uncertain about the world. Existing safety filters often assume full observability, limiting their applicability in real-world tasks. We propose a physics-based safety filtering scheme that leverages high-fidelity simulation to assess control policies under uncertainty in world parameters. The method combines dense rollout with nominal parameters and parallelizable sparse re-evaluation at critical state-transitions, quantified through generalized factors of safety for stable grasping and actuator limits, and targeted uncertainty reduction through probing actions. We demonstrate the approach in a simulated bimanual manipulation task with uncertain object mass and friction, showing that unsafe trajectories can be identified and filtered efficiently. Our results highlight physics-based sparse safety evaluation as a scalable strategy for safe robotic manipulation under uncertainty.
>
---
#### [new 029] Design and Control of a Perching Drone Inspired by the Prey-Capturing Mechanism of Venus Flytrap
- **分类: cs.RO**

- **简介: 论文设计了一款受捕蝇草启发的快速着陆无人机，解决续航与能耗问题。提出仿生抓取结构和级联EHGO控制方法，实现高速适应目标并提升系统稳定性。**

- **链接: [http://arxiv.org/pdf/2509.13249v1](http://arxiv.org/pdf/2509.13249v1)**

> **作者:** Ye Li; Daming Liu; Yanhe Zhu; Junming Zhang; Yongsheng Luo; Ziqi Wang; Chenyu Liu; Jie Zhao
>
> **摘要:** The endurance and energy efficiency of drones remain critical challenges in their design and operation. To extend mission duration, numerous studies explored perching mechanisms that enable drones to conserve energy by temporarily suspending flight. This paper presents a new perching drone that utilizes an active flexible perching mechanism inspired by the rapid predation mechanism of the Venus flytrap, achieving perching in less than 100 ms. The proposed system is designed for high-speed adaptability to the perching targets. The overall drone design is outlined, followed by the development and validation of the biomimetic perching structure. To enhance the system stability, a cascade extended high-gain observer (EHGO) based control method is developed, which can estimate and compensate for the external disturbance in real time. The experimental results demonstrate the adaptability of the perching structure and the superiority of the cascaded EHGO in resisting wind and perching disturbances.
>
---
#### [new 030] ROOM: A Physics-Based Continuum Robot Simulator for Photorealistic Medical Datasets Generation
- **分类: cs.RO**

- **简介: 该论文提出ROOM模拟器，用于生成逼真的支气管镜训练数据。解决医疗机器人缺乏真实训练环境的问题，利用CT扫描生成多模态传感器数据，并验证其在姿态估计和深度估计中的应用。**

- **链接: [http://arxiv.org/pdf/2509.13177v1](http://arxiv.org/pdf/2509.13177v1)**

> **作者:** Salvatore Esposito; Matías Mattamala; Daniel Rebain; Francis Xiatian Zhang; Kevin Dhaliwal; Mohsen Khadem; Subramanian Ramamoorthy
>
> **摘要:** Continuum robots are advancing bronchoscopy procedures by accessing complex lung airways and enabling targeted interventions. However, their development is limited by the lack of realistic training and test environments: Real data is difficult to collect due to ethical constraints and patient safety concerns, and developing autonomy algorithms requires realistic imaging and physical feedback. We present ROOM (Realistic Optical Observation in Medicine), a comprehensive simulation framework designed for generating photorealistic bronchoscopy training data. By leveraging patient CT scans, our pipeline renders multi-modal sensor data including RGB images with realistic noise and light specularities, metric depth maps, surface normals, optical flow and point clouds at medically relevant scales. We validate the data generated by ROOM in two canonical tasks for medical robotics -- multi-view pose estimation and monocular depth estimation, demonstrating diverse challenges that state-of-the-art methods must overcome to transfer to these medical settings. Furthermore, we show that the data produced by ROOM can be used to fine-tune existing depth estimation models to overcome these challenges, also enabling other downstream applications such as navigation. We expect that ROOM will enable large-scale data generation across diverse patient anatomies and procedural scenarios that are challenging to capture in clinical settings. Code and data: https://github.com/iamsalvatore/room.
>
---
#### [new 031] MinJointTracker: Real-time inertial kinematic chain tracking with joint position estimation and minimal state size
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出MinJointTracker，实现无需校准的实时惯性运动捕捉。解决传统方法依赖离线校准的问题，通过递归贝叶斯估计，在小状态空间中同步估计全局角运动和关节位置，实验验证其在不同场景下的鲁棒性和准确性。**

- **链接: [http://arxiv.org/pdf/2509.12398v1](http://arxiv.org/pdf/2509.12398v1)**

> **作者:** Michael Lorenz; Bertram Taetz; Gabriele Bleser-Taetz; Didier Stricker
>
> **备注:** 10 pages, 2 figures
>
> **摘要:** Inertial motion capture is a promising approach for capturing motion outside the laboratory. However, as one major drawback, most of the current methods require different quantities to be calibrated or computed offline as part of the setup process, such as segment lengths, relative orientations between inertial measurement units (IMUs) and segment coordinate frames (IMU-to-segment calibrations) or the joint positions in the IMU frames. This renders the setup process inconvenient. This work contributes to real-time capable calibration-free inertial tracking of a kinematic chain, i.e. simultaneous recursive Bayesian estimation of global IMU angular kinematics and joint positions in the IMU frames, with a minimal state size. Experimental results on simulated IMU data from a three-link kinematic chain (manipulator study) as well as re-simulated IMU data from healthy humans walking (lower body study) show that the calibration-free and lightweight algorithm provides not only drift-free relative but also drift-free absolute orientation estimates with a global heading reference for only one IMU as well as robust and fast convergence of joint position estimates in the different movement scenarios.
>
---
#### [new 032] Learning to Generate Pointing Gestures in Situated Embodied Conversational Agents
- **分类: cs.RO; cs.HC; cs.LG; 68T07, 68T40; I.2.9; I.2.6**

- **简介: 该论文研究具身对话代理生成指指点点手势的任务，旨在提升人机非语言交流的自然性与准确性。通过模仿学习与强化学习结合，提出一种生成自然、准确手势的框架，并在虚拟现实中验证其优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.12507v1](http://arxiv.org/pdf/2509.12507v1)**

> **作者:** Anna Deichler; Siyang Wang; Simon Alexanderson; Jonas Beskow
>
> **备注:** DOI: 10.3389/frobt.2023.1110534. This is the author's LaTeX version
>
> **摘要:** One of the main goals of robotics and intelligent agent research is to enable natural communication with humans in physically situated settings. While recent work has focused on verbal modes such as language and speech, non-verbal communication is crucial for flexible interaction. We present a framework for generating pointing gestures in embodied agents by combining imitation and reinforcement learning. Using a small motion capture dataset, our method learns a motor control policy that produces physically valid, naturalistic gestures with high referential accuracy. We evaluate the approach against supervised learning and retrieval baselines in both objective metrics and a virtual reality referential game with human users. Results show that our system achieves higher naturalness and accuracy than state-of-the-art supervised models, highlighting the promise of imitation-RL for communicative gesture generation and its potential application to robots.
>
---
#### [new 033] StageACT: Stage-Conditioned Imitation for Robust Humanoid Door Opening
- **分类: cs.RO**

- **简介: 论文提出StageACT框架，解决人形机器人开门任务中的部分可观测性和长时序决策问题。通过阶段条件模仿学习，提升策略鲁棒性，实现更高成功率和更快完成时间。**

- **链接: [http://arxiv.org/pdf/2509.13200v1](http://arxiv.org/pdf/2509.13200v1)**

> **作者:** Moonyoung Lee; Dong Ki Kim; Jai Krishna Bandi; Max Smith; Aileen Liao; Ali-akbar Agha-mohammadi; Shayegan Omidshafiei
>
> **备注:** 7 pages
>
> **摘要:** Humanoid robots promise to operate in everyday human environments without requiring modifications to the surroundings. Among the many skills needed, opening doors is essential, as doors are the most common gateways in built spaces and often limit where a robot can go. Door opening, however, poses unique challenges as it is a long-horizon task under partial observability, such as reasoning about the door's unobservable latch state that dictates whether the robot should rotate the handle or push the door. This ambiguity makes standard behavior cloning prone to mode collapse, yielding blended or out-of-sequence actions. We introduce StageACT, a stage-conditioned imitation learning framework that augments low-level policies with task-stage inputs. This effective addition increases robustness to partial observability, leading to higher success rates and shorter completion times. On a humanoid operating in a real-world office environment, StageACT achieves a 55% success rate on previously unseen doors, more than doubling the best baseline. Moreover, our method supports intentional behavior guidance through stage prompting, enabling recovery behaviors. These results highlight stage conditioning as a lightweight yet powerful mechanism for long-horizon humanoid loco-manipulation.
>
---
#### [new 034] Out of Distribution Detection in Self-adaptive Robots with AI-powered Digital Twins
- **分类: cs.RO; cs.AI; cs.SE**

- **简介: 论文提出ODiSAR方法，利用AI驱动的数字孪生检测自适应机器人中的分布外异常。通过重构误差与预测方差量化不确定性，实现高精度检测并提供可解释性支持，提升机器人在复杂环境中的自适应能力。**

- **链接: [http://arxiv.org/pdf/2509.12982v1](http://arxiv.org/pdf/2509.12982v1)**

> **作者:** Erblin Isaku; Hassan Sartaj; Shaukat Ali; Beatriz Sanguino; Tongtong Wang; Guoyuan Li; Houxiang Zhang; Thomas Peyrucain
>
> **备注:** 15 pages, 4 figures, 3 tables
>
> **摘要:** Self-adaptive robots (SARs) in complex, uncertain environments must proactively detect and address abnormal behaviors, including out-of-distribution (OOD) cases. To this end, digital twins offer a valuable solution for OOD detection. Thus, we present a digital twin-based approach for OOD detection (ODiSAR) in SARs. ODiSAR uses a Transformer-based digital twin to forecast SAR states and employs reconstruction error and Monte Carlo dropout for uncertainty quantification. By combining reconstruction error with predictive variance, the digital twin effectively detects OOD behaviors, even in previously unseen conditions. The digital twin also includes an explainability layer that links potential OOD to specific SAR states, offering insights for self-adaptation. We evaluated ODiSAR by creating digital twins of two industrial robots: one navigating an office environment, and another performing maritime ship navigation. In both cases, ODiSAR forecasts SAR behaviors (i.e., robot trajectories and vessel motion) and proactively detects OOD events. Our results showed that ODiSAR achieved high detection performance -- up to 98\% AUROC, 96\% TNR@TPR95, and 95\% F1-score -- while providing interpretable insights to support self-adaptation.
>
---
#### [new 035] DVDP: An End-to-End Policy for Mobile Robot Visual Docking with RGB-D Perception
- **分类: cs.RO**

- **简介: 该论文提出DVDP方法，解决移动机器人视觉对接任务中初始位置依赖性强的问题。通过RGB-D相机实现端到端对接路径规划，并构建大规模数据集验证方法有效性，实验证明其性能优越且适用于实际场景。**

- **链接: [http://arxiv.org/pdf/2509.13024v1](http://arxiv.org/pdf/2509.13024v1)**

> **作者:** Haohan Min; Zhoujian Li; Yu Yang; Jinyu Chen; Shenghai Yuan
>
> **摘要:** Automatic docking has long been a significant challenge in the field of mobile robotics. Compared to other automatic docking methods, visual docking methods offer higher precision and lower deployment costs, making them an efficient and promising choice for this task. However, visual docking methods impose strict requirements on the robot's initial position at the start of the docking process. To overcome the limitations of current vision-based methods, we propose an innovative end-to-end visual docking method named DVDP(direct visual docking policy). This approach requires only a binocular RGB-D camera installed on the mobile robot to directly output the robot's docking path, achieving end-to-end automatic docking. Furthermore, we have collected a large-scale dataset of mobile robot visual automatic docking dataset through a combination of virtual and real environments using the Unity 3D platform and actual mobile robot setups. We developed a series of evaluation metrics to quantify the performance of the end-to-end visual docking method. Extensive experiments, including benchmarks against leading perception backbones adapted into our framework, demonstrate that our method achieves superior performance. Finally, real-world deployment on the SCOUT Mini confirmed DVDP's efficacy, with our model generating smooth, feasible docking trajectories that meet physical constraints and reach the target pose.
>
---
#### [new 036] A Novel Skill Modeling Approach: Integrating Vergnaud's Scheme with Cognitive Architectures
- **分类: cs.RO**

- **简介: 论文提出一种新技能建模方法，结合Vergnaud方案与认知架构，解决工业操作中技能适应与建模问题，以焊接为例，强调延迟反馈下操作者技能的复杂性与建模重要性。**

- **链接: [http://arxiv.org/pdf/2509.12851v1](http://arxiv.org/pdf/2509.12851v1)**

> **作者:** Antoine Lénat; Olivier Cheminat; Damien Chablat; Camilo Charron
>
> **摘要:** Human-machine interaction is increasingly important in industry, and this trend will only intensify with the rise of Industry 5.0. Human operators have skills that need to be adapted when using machines to achieve the best results. It is crucial to highlight the operator's skills and understand how they use and adapt them [18]. A rigorous description of these skills is necessary to compare performance with and without robot assistance. Predicate logic, used by Vergnaud within Piaget's scheme concept, offers a promising approach. However, this theory doesn't account for cognitive system constraints, such as the timing of actions, the limitation of cognitive resources, the parallelization of tasks, or the activation of automatic gestures contrary to optimal knowledge. Integrating these constraints is essential for representing agent skills understanding skill transfer between biological and mechanical structures. Cognitive architectures models [2] address these needs by describing cognitive structure and can be combined with the scheme for mutual benefit. Welding provides a relevant case study, as it highlights the challenges faced by operators, even highly skilled ones. Welding's complexity stems from the need for constant skill adaptation to variable parameters like part position and process. This adaptation is crucial, as weld quality, a key factor, is only assessed afterward via destructive testing. Thus, the welder is confronted with a complex perception-decision-action cycle, where the evaluation of the impact of his actions is delayed and where errors are definitive. This dynamic underscores the importance of understanding and modeling the skills of operators.
>
---
#### [new 037] The Better You Learn, The Smarter You Prune: Towards Efficient Vision-language-action Models via Differentiable Token Pruning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出LightVLA，一种通过可微分视觉token剪枝提升视觉-语言-动作模型效率的方法。针对VLA模型在资源受限平台部署时计算量大的问题，通过动态评估token重要性实现高效剪枝，提升性能并减少计算开销。**

- **链接: [http://arxiv.org/pdf/2509.12594v1](http://arxiv.org/pdf/2509.12594v1)**

> **作者:** Titong Jiang; Xuefeng Jiang; Yuan Ma; Xin Wen; Bailin Li; Kun Zhan; Peng Jia; Yahui Liu; Sheng Sun; Xianpeng Lang
>
> **备注:** Under review. Project site: https://liauto-research.github.io/LightVLA
>
> **摘要:** We present LightVLA, a simple yet effective differentiable token pruning framework for vision-language-action (VLA) models. While VLA models have shown impressive capability in executing real-world robotic tasks, their deployment on resource-constrained platforms is often bottlenecked by the heavy attention-based computation over large sets of visual tokens. LightVLA addresses this challenge through adaptive, performance-driven pruning of visual tokens: It generates dynamic queries to evaluate visual token importance, and adopts Gumbel softmax to enable differentiable token selection. Through fine-tuning, LightVLA learns to preserve the most informative visual tokens while pruning tokens which do not contribute to task execution, thereby improving efficiency and performance simultaneously. Notably, LightVLA requires no heuristic magic numbers and introduces no additional trainable parameters, making it compatible with modern inference frameworks. Experimental results demonstrate that LightVLA outperforms different VLA models and existing token pruning methods across diverse tasks on the LIBERO benchmark, achieving higher success rates with substantially reduced computational overhead. Specifically, LightVLA reduces FLOPs and latency by 59.1% and 38.2% respectively, with a 2.9% improvement in task success rate. Meanwhile, we also investigate the learnable query-based token pruning method LightVLA* with additional trainable parameters, which also achieves satisfactory performance. Our work reveals that as VLA pursues optimal performance, LightVLA spontaneously learns to prune tokens from a performance-driven perspective. To the best of our knowledge, LightVLA is the first work to apply adaptive visual token pruning to VLA tasks with the collateral goals of efficiency and performance, marking a significant step toward more efficient, powerful and practical real-time robotic systems.
>
---
#### [new 038] Towards Context-Aware Human-like Pointing Gestures with RL Motion Imitation
- **分类: cs.RO; cs.HC; cs.LG; 68T05, 68T40; I.2.9; I.2.6; H.5.2**

- **简介: 该论文属于机器人交互任务，旨在生成自然的人类指指点点手势。通过强化学习与动作模仿，训练出能适应不同情境、精准指向目标的策略，解决了机器人生成自然指向动作的问题。**

- **链接: [http://arxiv.org/pdf/2509.12880v1](http://arxiv.org/pdf/2509.12880v1)**

> **作者:** Anna Deichler; Siyang Wang; Simon Alexanderson; Jonas Beskow
>
> **备注:** Presented at the Context-Awareness in HRI (CONAWA) Workshop, ACM/IEEE International Conference on Human-Robot Interaction (HRI 2022), March 7, 2022
>
> **摘要:** Pointing is a key mode of interaction with robots, yet most prior work has focused on recognition rather than generation. We present a motion capture dataset of human pointing gestures covering diverse styles, handedness, and spatial targets. Using reinforcement learning with motion imitation, we train policies that reproduce human-like pointing while maximizing precision. Results show our approach enables context-aware pointing behaviors in simulation, balancing task performance with natural dynamics.
>
---
#### [new 039] Neural 3D Object Reconstruction with Small-Scale Unmanned Aerial Vehicles
- **分类: cs.RO; cs.AR; cs.CV; cs.ET; cs.SY; eess.SY**

- **简介: 该论文提出一种基于轻量无人机的自主3D重建系统，解决其载重与自主性限制问题。通过实时反馈调整飞行路径，并结合NeRF技术提升重建精度，实现高保真静态物体建模。**

- **链接: [http://arxiv.org/pdf/2509.12458v1](http://arxiv.org/pdf/2509.12458v1)**

> **作者:** Àlmos Veres-Vitàlyos; Genis Castillo Gomez-Raya; Filip Lemic; Daniel Johannes Bugelnig; Bernhard Rinner; Sergi Abadal; Xavier Costa-Pérez
>
> **备注:** 13 pages, 16 figures, 3 tables, 45 references
>
> **摘要:** Small Unmanned Aerial Vehicles (UAVs) exhibit immense potential for navigating indoor and hard-to-reach areas, yet their significant constraints in payload and autonomy have largely prevented their use for complex tasks like high-quality 3-Dimensional (3D) reconstruction. To overcome this challenge, we introduce a novel system architecture that enables fully autonomous, high-fidelity 3D scanning of static objects using UAVs weighing under 100 grams. Our core innovation lies in a dual-reconstruction pipeline that creates a real-time feedback loop between data capture and flight control. A near-real-time (near-RT) process uses Structure from Motion (SfM) to generate an instantaneous pointcloud of the object. The system analyzes the model quality on the fly and dynamically adapts the UAV's trajectory to intelligently capture new images of poorly covered areas. This ensures comprehensive data acquisition. For the final, detailed output, a non-real-time (non-RT) pipeline employs a Neural Radiance Fields (NeRF)-based Neural 3D Reconstruction (N3DR) approach, fusing SfM-derived camera poses with precise Ultra Wide-Band (UWB) location data to achieve superior accuracy. We implemented and validated this architecture using Crazyflie 2.1 UAVs. Our experiments, conducted in both single- and multi-UAV configurations, conclusively show that dynamic trajectory adaptation consistently improves reconstruction quality over static flight paths. This work demonstrates a scalable and autonomous solution that unlocks the potential of miniaturized UAVs for fine-grained 3D reconstruction in constrained environments, a capability previously limited to much larger platforms.
>
---
#### [new 040] NAMOUnc: Navigation Among Movable Obstacles with Decision Making on Uncertainty Interval
- **分类: cs.RO**

- **简介: 论文提出NAMOUnc框架，解决机器人在可移动障碍物中的导航问题，考虑现实中的不确定性。通过估计不确定性并优化避障策略，提升导航的安全性与效率。属于移动机器人路径规划任务。**

- **链接: [http://arxiv.org/pdf/2509.12723v1](http://arxiv.org/pdf/2509.12723v1)**

> **作者:** Kai Zhang; Eric Lucet; Julien Alexandre Dit Sandretto; Shoubin Chen; David Filait
>
> **备注:** 11 pages, ICINCO2025
>
> **摘要:** Navigation among movable obstacles (NAMO) is a critical task in robotics, often challenged by real-world uncertainties such as observation noise, model approximations, action failures, and partial observability. Existing solutions frequently assume ideal conditions, leading to suboptimal or risky decisions. This paper introduces NAMOUnc, a novel framework designed to address these uncertainties by integrating them into the decision-making process. We first estimate them and compare the corresponding time cost intervals for removing and bypassing obstacles, optimizing both the success rate and time efficiency, ensuring safer and more efficient navigation. We validate our method through extensive simulations and real-world experiments, demonstrating significant improvements over existing NAMO frameworks. More details can be found in our website: https://kai-zhang-er.github.io/namo-uncertainty/
>
---
#### [new 041] Bio-inspired tail oscillation enables robot fast crawling on deformable granular terrains
- **分类: cs.RO**

- **简介: 论文研究仿生尾部摆动对机器人在松散颗粒地形中快速爬行的影响。通过模仿弹涂鱼，设计并测试了不同尾部结构与运动策略，发现尾部摆动能提升速度并降低阻力，为变形地面机器人提供新的设计原则。**

- **链接: [http://arxiv.org/pdf/2509.12468v1](http://arxiv.org/pdf/2509.12468v1)**

> **作者:** Shipeng Liu; Meghana Sagare; Shubham Patil; Feifei Qian
>
> **摘要:** Deformable substrates such as sand and mud present significant challenges for terrestrial robots due to complex robot-terrain interactions. Inspired by mudskippers, amphibious animals that naturally adjust their tail morphology and movement jointly to navigate such environments, we investigate how tail design and control can jointly enhance flipper-driven locomotion on granular media. Using a bio-inspired robot modeled after the mudskipper, we experimentally compared locomotion performance between idle and actively oscillating tail configurations. Tail oscillation increased robot speed by 67% and reduced body drag by 46%. Shear force measurements revealed that this improvement was enabled by tail oscillation fluidizing the substrate, thereby reducing resistance. Additionally, tail morphology strongly influenced the oscillation strategy: designs with larger horizontal surface areas leveraged the oscillation-reduced shear resistance more effectively by limiting insertion depth. Based on these findings, we present a design principle to inform tail action selection based on substrate strength and tail morphology. Our results offer new insights into tail design and control for improving robot locomotion on deformable substrates, with implications for agricultural robotics, search and rescue, and environmental exploration.
>
---
#### [new 042] Tendon-Based Proprioception in an Anthropomorphic Underactuated Robotic Hand with Series Elastic Actuators
- **分类: cs.RO**

- **简介: 该论文提出一种基于肌腱的本体感觉系统，用于仿人欠驱动机械手。旨在解决欠驱动系统中传感集成与抓取功能实现的问题。通过串联弹性执行器实现紧凑、可靠的传感，估计抓取关键变量，实现无视觉和触觉反馈的盲抓取与变形物体安全操作。**

- **链接: [http://arxiv.org/pdf/2509.12969v1](http://arxiv.org/pdf/2509.12969v1)**

> **作者:** Jae-Hyun Lee; Jonghoo Park; Kyu-Jin Cho
>
> **备注:** 8 pages, 10 figures, Supplementary video, Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Anthropomorphic underactuated hands are widely employed for their versatility and structural simplicity. In such systems, compact sensing integration and proper interpretation aligned with underactuation are crucial for realizing practical grasp functionalities. This study proposes an anthropomorphic underactuated hand that achieves comprehensive situational awareness of hand-object interaction, utilizing tendon-based proprioception provided by series elastic actuators (SEAs). We developed a compact SEA with high accuracy and reliability that can be seamlessly integrated into sensorless fingers. By coupling proprioceptive sensing with potential energy-based modeling, the system estimates key grasp-related variables, including contact timing, joint angles, relative object stiffness, and finger configuration changes indicating external disturbances. These estimated variables enable grasp posture reconstruction, safe handling of deformable objects, and blind grasping with proprioceptive-only recognition of objects with varying geometry and stiffness. Finger-level experiments and hand-level demonstrations confirmed the effectiveness of the proposed approach. The results demonstrate that tendon-based proprioception serves as a compact and robust sensing modality for practical manipulation without reliance on vision or tactile feedback.
>
---
#### [new 043] An Uncertainty-Weighted Decision Transformer for Navigation in Dense, Complex Driving Scenarios
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种不确定性加权决策变换器（UWDT），用于复杂密集交通场景中的自动驾驶导航。通过结合多通道鸟瞰图与变换器模型，解决低风险状态与高风险决策不平衡问题，提升系统在复杂环岛场景中的安全性和决策效率。**

- **链接: [http://arxiv.org/pdf/2509.13132v1](http://arxiv.org/pdf/2509.13132v1)**

> **作者:** Zhihao Zhang; Chengyang Peng; Minghao Zhu; Ekim Yurtsever; Keith A. Redmill
>
> **摘要:** Autonomous driving in dense, dynamic environments requires decision-making systems that can exploit both spatial structure and long-horizon temporal dependencies while remaining robust to uncertainty. This work presents a novel framework that integrates multi-channel bird's-eye-view occupancy grids with transformer-based sequence modeling for tactical driving in complex roundabout scenarios. To address the imbalance between frequent low-risk states and rare safety-critical decisions, we propose the Uncertainty-Weighted Decision Transformer (UWDT). UWDT employs a frozen teacher transformer to estimate per-token predictive entropy, which is then used as a weight in the student model's loss function. This mechanism amplifies learning from uncertain, high-impact states while maintaining stability across common low-risk transitions. Experiments in a roundabout simulator, across varying traffic densities, show that UWDT consistently outperforms other baselines in terms of reward, collision rate, and behavioral stability. The results demonstrate that uncertainty-aware, spatial-temporal transformers can deliver safer and more efficient decision-making for autonomous driving in complex traffic environments.
>
---
#### [new 044] TeraSim-World: Worldwide Safety-Critical Data Synthesis for End-to-End Autonomous Driving
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出TeraSim-World，用于生成全球范围内的安全关键数据，以支持端到端自动驾驶系统的训练与评估。其解决数据不足与仿真现实差距问题，通过真实地图、交通数据和视频生成模型合成高真实感场景。**

- **链接: [http://arxiv.org/pdf/2509.13164v1](http://arxiv.org/pdf/2509.13164v1)**

> **作者:** Jiawei Wang; Haowei Sun; Xintao Yan; Shuo Feng; Jun Gao; Henry X. Liu
>
> **备注:** 8 pages, 6 figures. Codes and videos are available at https://wjiawei.com/terasim-world-web/
>
> **摘要:** Safe and scalable deployment of end-to-end (E2E) autonomous driving requires extensive and diverse data, particularly safety-critical events. Existing data are mostly generated from simulators with a significant sim-to-real gap or collected from on-road testing that is costly and unsafe. This paper presents TeraSim-World, an automated pipeline that synthesizes realistic and geographically diverse safety-critical data for E2E autonomous driving at anywhere in the world. Starting from an arbitrary location, TeraSim-World retrieves real-world maps and traffic demand from geospatial data sources. Then, it simulates agent behaviors from naturalistic driving datasets, and orchestrates diverse adversities to create corner cases. Informed by street views of the same location, it achieves photorealistic, geographically grounded sensor rendering via the frontier video generation model Cosmos-Drive. By bridging agent and sensor simulations, TeraSim-World provides a scalable and critical~data synthesis framework for training and evaluation of E2E autonomous driving systems.
>
---
#### [new 045] Model Predictive Control with Reference Learning for Soft Robotic Intracranial Pressure Waveform Modulation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种基于模型预测控制与贝叶斯优化的软体机器人控制系统，用于调节颅内压波形。该框架通过MPC实现安全跟踪，结合BO算法在线学习最优电机轨迹，有效提升ICP调控精度。属于控制与医疗机器人领域，解决精准颅内压波形调节问题。**

- **链接: [http://arxiv.org/pdf/2509.13109v1](http://arxiv.org/pdf/2509.13109v1)**

> **作者:** Fabian Flürenbrock; Yanick Büchel; Johannes Köhler; Marianne Schmid Daners; Melanie N. Zeilinger
>
> **摘要:** This paper introduces a learning-based control framework for a soft robotic actuator system designed to modulate intracranial pressure (ICP) waveforms, which is essential for studying cerebrospinal fluid dynamics and pathological processes underlying neurological disorders. A two-layer framework is proposed to safely achieve a desired ICP waveform modulation. First, a model predictive controller (MPC) with a disturbance observer is used for offset-free tracking of the system's motor position reference trajectory under safety constraints. Second, to address the unknown nonlinear dependence of ICP on the motor position, we employ a Bayesian optimization (BO) algorithm used for online learning of a motor position reference trajectory that yields the desired ICP modulation. The framework is experimentally validated using a test bench with a brain phantom that replicates realistic ICP dynamics in vitro. Compared to a previously employed proportional-integral-derivative controller, the MPC reduces mean and maximum motor position reference tracking errors by 83 % and 73 %, respectively. In less than 20 iterations, the BO algorithm learns a motor position reference trajectory that yields an ICP waveform with the desired mean and amplitude.
>
---
#### [new 046] Integrating Trajectory Optimization and Reinforcement Learning for Quadrupedal Jumping with Terrain-Adaptive Landing
- **分类: cs.RO**

- **简介: 该论文属于四足机器人跳跃运动控制任务，旨在解决复杂地形下安全着陆问题。通过结合轨迹优化与强化学习，提出适应性着陆框架，提升机器人在不平地形中的跳跃稳定性与安全性。**

- **链接: [http://arxiv.org/pdf/2509.12776v1](http://arxiv.org/pdf/2509.12776v1)**

> **作者:** Renjie Wang; Shangke Lyu; Xin Lang; Wei Xiao; Donglin Wang
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Jumping constitutes an essential component of quadruped robots' locomotion capabilities, which includes dynamic take-off and adaptive landing. Existing quadrupedal jumping studies mainly focused on the stance and flight phase by assuming a flat landing ground, which is impractical in many real world cases. This work proposes a safe landing framework that achieves adaptive landing on rough terrains by combining Trajectory Optimization (TO) and Reinforcement Learning (RL) together. The RL agent learns to track the reference motion generated by TO in the environments with rough terrains. To enable the learning of compliant landing skills on challenging terrains, a reward relaxation strategy is synthesized to encourage exploration during landing recovery period. Extensive experiments validate the accurate tracking and safe landing skills benefiting from our proposed method in various scenarios.
>
---
#### [new 047] An integrated process for design and control of lunar robotics using AI and simulation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种结合AI与仿真的月球机器人设计与控制集成方法，解决月球设备开发中设计与控制并行优化的问题。通过OpenPLX语言连接CAD模型与高精度仿真，实现自主月球车的导航与运动控制。**

- **链接: [http://arxiv.org/pdf/2509.12367v1](http://arxiv.org/pdf/2509.12367v1)**

> **作者:** Daniel Lindmark; Jonas Andersson; Kenneth Bodin; Tora Bodin; Hugo Börjesson; Fredrik Nordfeldth; Martin Servin
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** We envision an integrated process for developing lunar construction equipment, where physical design and control are explored in parallel. In this paper, we describe a technical framework that supports this process. It relies on OpenPLX, a readable/writable declarative language that links CAD-models and autonomous systems to high-fidelity, real-time 3D simulations of contacting multibody dynamics, machine regolith interaction forces, and non-ideal sensors. To demonstrate its capabilities, we present two case studies, including an autonomous lunar rover that combines a vision-language model for navigation with a reinforcement learning-based control policy for locomotion.
>
---
#### [new 048] UDON: Uncertainty-weighted Distributed Optimization for Multi-Robot Neural Implicit Mapping under Extreme Communication Constraints
- **分类: cs.RO**

- **简介: 论文提出UDON框架，解决多机器人在极端通信条件下神经隐式建图问题。通过不确定性加权分布式优化，提升地图质量与一致性，实验证明其在低通信成功率下仍表现优异。**

- **链接: [http://arxiv.org/pdf/2509.12702v1](http://arxiv.org/pdf/2509.12702v1)**

> **作者:** Hongrui Zhao; Xunlan Zhou; Boris Ivanovic; Negar Mehr
>
> **摘要:** Multi-robot mapping with neural implicit representations enables the compact reconstruction of complex environments. However, it demands robustness against communication challenges like packet loss and limited bandwidth. While prior works have introduced various mechanisms to mitigate communication disruptions, performance degradation still occurs under extremely low communication success rates. This paper presents UDON, a real-time multi-agent neural implicit mapping framework that introduces a novel uncertainty-weighted distributed optimization to achieve high-quality mapping under severe communication deterioration. The uncertainty weighting prioritizes more reliable portions of the map, while the distributed optimization isolates and penalizes mapping disagreement between individual pairs of communicating agents. We conduct extensive experiments on standard benchmark datasets and real-world robot hardware. We demonstrate that UDON significantly outperforms existing baselines, maintaining high-fidelity reconstructions and consistent scene representations even under extreme communication degradation (as low as 1% success rate).
>
---
#### [new 049] Zero to Autonomy in Real-Time: Online Adaptation of Dynamics in Unstructured Environments
- **分类: cs.RO**

- **简介: 论文提出一种在线适应方法，解决自主机器人在非结构化环境中实时应对动力学变化的问题。通过结合函数编码器与递归最小二乘法，实现快速模型更新，提升控制安全性和规划性能。**

- **链接: [http://arxiv.org/pdf/2509.12516v1](http://arxiv.org/pdf/2509.12516v1)**

> **作者:** William Ward; Sarah Etter; Jesse Quattrociocchi; Christian Ellis; Adam J. Thorpe; Ufuk Topcu
>
> **备注:** Submitted to ICRA 2026
>
> **摘要:** Autonomous robots must go from zero prior knowledge to safe control within seconds to operate in unstructured environments. Abrupt terrain changes, such as a sudden transition to ice, create dynamics shifts that can destabilize planners unless the model adapts in real-time. We present a method for online adaptation that combines function encoders with recursive least squares, treating the function encoder coefficients as latent states updated from streaming odometry. This yields constant-time coefficient estimation without gradient-based inner-loop updates, enabling adaptation from only a few seconds of data. We evaluate our approach on a Van der Pol system to highlight algorithmic behavior, in a Unity simulator for high-fidelity off-road navigation, and on a Clearpath Jackal robot, including on a challenging terrain at a local ice rink. Across these settings, our method improves model accuracy and downstream planning, reducing collisions compared to static and meta-learning baselines.
>
---
#### [new 050] Beyond Anthropomorphism: Enhancing Grasping and Eliminating a Degree of Freedom by Fusing the Abduction of Digits Four and Five
- **分类: cs.RO**

- **简介: 论文提出一种16自由度的非拟人机械手SABD，通过融合四、五指的收展关节，扩大抓取范围并减少执行器数量。该设计提升了抓取稳定性与灵活性，适用于复杂抓取任务。**

- **链接: [http://arxiv.org/pdf/2509.13074v1](http://arxiv.org/pdf/2509.13074v1)**

> **作者:** Simon Fritsch; Liam Achenbach; Riccardo Bianco; Nicola Irmiger; Gawain Marti; Samuel Visca; Chenyu Yang; Davide Liconti; Barnabas Gavin Cangan; Robert Jomar Malate; Ronan J. Hinchet; Robert K. Katzschmann
>
> **备注:** First five listed authors have equal contribution
>
> **摘要:** This paper presents the SABD hand, a 16-degree-of-freedom (DoF) robotic hand that departs from purely anthropomorphic designs to achieve an expanded grasp envelope, enable manipulation poses beyond human capability, and reduce the required number of actuators. This is achieved by combining the adduction/abduction (Add/Abd) joint of digits four and five into a single joint with a large range of motion. The combined joint increases the workspace of the digits by 400\% and reduces the required DoFs while retaining dexterity. Experimental results demonstrate that the combined Add/Abd joint enables the hand to grasp objects with a side distance of up to 200 mm. Reinforcement learning-based investigations show that the design enables grasping policies that are effective not only for handling larger objects but also for achieving enhanced grasp stability. In teleoperated trials, the hand successfully performed 86\% of attempted grasps on suitable YCB objects, including challenging non-anthropomorphic configurations. These findings validate the design's ability to enhance grasp stability, flexibility, and dexterous manipulation without added complexity, making it well-suited for a wide range of applications.
>
---
#### [new 051] Safety Critical Model Predictive Control Using Discrete-Time Control Density Functions
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 论文提出MPC-CDF方法，将控制密度函数融入模型预测控制框架，解决非线性系统的安全关键控制问题。通过离散时间设置确保收敛与安全，应用于自主避障导航任务，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.13257v1](http://arxiv.org/pdf/2509.13257v1)**

> **作者:** Sriram S. K. S. Narayanan; Sajad Ahmadi; Javad Mohammadpour Velni; Umesh Vaidya
>
> **摘要:** This paper presents MPC-CDF, a new approach integrating control density functions (CDFs) within a model predictive control (MPC) framework to ensure safety-critical control in nonlinear dynamical systems. By using the dual formulation of the navigation problem, we incorporate CDFs into the MPC framework, ensuring both convergence and safety in a discrete-time setting. These density functions are endowed with a physical interpretation, where the associated measure signifies the occupancy of system trajectories. Leveraging this occupancy-based perspective, we synthesize safety-critical controllers using the proposed MPC-CDF framework. We illustrate the safety properties of this framework using a unicycle model and compare it with a control barrier function-based method. The efficacy of this approach is demonstrated in the autonomous safe navigation of an underwater vehicle, which avoids complex and arbitrary obstacles while achieving the desired level of safety.
>
---
#### [new 052] OnlineHOI: Towards Online Human-Object Interaction Generation and Perception
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出OnlineHOI框架，解决在线人类-物体交互生成与感知任务。传统方法依赖离线数据，而在线场景需实时处理当前及历史信息。本文基于Mamba架构引入记忆机制，实现在线HOI生成与感知的前沿成果。**

- **链接: [http://arxiv.org/pdf/2509.12250v1](http://arxiv.org/pdf/2509.12250v1)**

> **作者:** Yihong Ji; Yunze Liu; Yiyao Zhuo; Weijiang Yu; Fei Ma; Joshua Huang; Fei Yu
>
> **备注:** Accepted at ACM MM 2025
>
> **摘要:** The perception and generation of Human-Object Interaction (HOI) are crucial for fields such as robotics, AR/VR, and human behavior understanding. However, current approaches model this task in an offline setting, where information at each time step can be drawn from the entire interaction sequence. In contrast, in real-world scenarios, the information available at each time step comes only from the current moment and historical data, i.e., an online setting. We find that offline methods perform poorly in an online context. Based on this observation, we propose two new tasks: Online HOI Generation and Perception. To address this task, we introduce the OnlineHOI framework, a network architecture based on the Mamba framework that employs a memory mechanism. By leveraging Mamba's powerful modeling capabilities for streaming data and the Memory mechanism's efficient integration of historical information, we achieve state-of-the-art results on the Core4D and OAKINK2 online generation tasks, as well as the online HOI4D perception task.
>
---
#### [new 053] A Synthetic Data Pipeline for Supporting Manufacturing SMEs in Visual Assembly Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种基于合成数据的视觉装配控制方法，用于帮助制造中小型企业实现高效质量控制。通过CAD数据生成模拟场景并结合目标检测算法，减少数据采集与标注成本，实现实时高精度装配检测。**

- **链接: [http://arxiv.org/pdf/2509.13089v1](http://arxiv.org/pdf/2509.13089v1)**

> **作者:** Jonas Werheid; Shengjie He; Aymen Gannouni; Anas Abdelrazeq; Robert H. Schmitt
>
> **摘要:** Quality control of assembly processes is essential in manufacturing to ensure not only the quality of individual components but also their proper integration into the final product. To assist in this matter, automated assembly control using computer vision methods has been widely implemented. However, the costs associated with image acquisition, annotation, and training of computer vision algorithms pose challenges for integration, especially for small- and medium-sized enterprises (SMEs), which often lack the resources for extensive training, data collection, and manual image annotation. Synthetic data offers the potential to reduce manual data collection and labeling. Nevertheless, its practical application in the context of assembly quality remains limited. In this work, we present a novel approach for easily integrable and data-efficient visual assembly control. Our approach leverages simulated scene generation based on computer-aided design (CAD) data and object detection algorithms. The results demonstrate a time-saving pipeline for generating image data in manufacturing environments, achieving a mean Average Precision (mAP@0.5:0.95) up to 99,5% for correctly identifying instances of synthetic planetary gear system components within our simulated training data, and up to 93% when transferred to real-world camera-captured testing data. This research highlights the effectiveness of synthetic data generation within an adaptable pipeline and underscores its potential to support SMEs in implementing resource-efficient visual assembly control solutions.
>
---
#### [new 054] UrgenGo: Urgency-Aware Transparent GPU Kernel Launching for Autonomous Driving
- **分类: cs.OS; cs.RO**

- **简介: 论文提出UrgenGo，一种无需源码的GPU调度系统，解决自动驾驶中因GPU任务紧迫性不足导致的截止期错过问题。通过透明内核调度优化，实现61%的截止期遗漏率降低。属于实时GPU任务调度优化任务。**

- **链接: [http://arxiv.org/pdf/2509.12207v1](http://arxiv.org/pdf/2509.12207v1)**

> **作者:** Hanqi Zhu; Wuyang Zhang; Xinran Zhang; Ziyang Tao; Xinrui Lin; Yu Zhang; Jianmin Ji; Yanyong Zhang
>
> **摘要:** The rapid advancements in autonomous driving have introduced increasingly complex, real-time GPU-bound tasks critical for reliable vehicle operation. However, the proprietary nature of these autonomous systems and closed-source GPU drivers hinder fine-grained control over GPU executions, often resulting in missed deadlines that compromise vehicle performance. To address this, we present UrgenGo, a non-intrusive, urgency-aware GPU scheduling system that operates without access to application source code. UrgenGo implicitly prioritizes GPU executions through transparent kernel launch manipulation, employing task-level stream binding, delayed kernel launching, and batched kernel launch synchronization. We conducted extensive real-world evaluations in collaboration with a self-driving startup, developing 11 GPU-bound task chains for a realistic autonomous navigation application and implementing our system on a self-driving bus. Our results show a significant 61% reduction in the overall deadline miss ratio, compared to the state-of-the-art GPU scheduler that requires source code modifications.
>
---
#### [new 055] Shapes of Cognition for Computational Cognitive Modeling
- **分类: cs.AI; cs.RO**

- **简介: 论文提出“认知形状”概念，用于构建语言赋能智能代理的计算认知模型。旨在通过典型性、模式识别等机制降低认知负荷，解决复杂环境下的决策与适应问题，并推动可解释、可信的AI系统发展。**

- **链接: [http://arxiv.org/pdf/2509.13288v1](http://arxiv.org/pdf/2509.13288v1)**

> **作者:** Marjorie McShane; Sergei Nirenburg; Sanjay Oruganti; Jesse English
>
> **摘要:** Shapes of cognition is a new conceptual paradigm for the computational cognitive modeling of Language-Endowed Intelligent Agents (LEIAs). Shapes are remembered constellations of sensory, linguistic, conceptual, episodic, and procedural knowledge that allow agents to cut through the complexity of real life the same way as people do: by expecting things to be typical, recognizing patterns, acting by habit, reasoning by analogy, satisficing, and generally minimizing cognitive load to the degree situations permit. Atypical outcomes are treated using shapes-based recovery methods, such as learning on the fly, asking a human partner for help, or seeking an actionable, even if imperfect, situational understanding. Although shapes is an umbrella term, it is not vague: shapes-based modeling involves particular objectives, hypotheses, modeling strategies, knowledge bases, and actual models of wide-ranging phenomena, all implemented within a particular cognitive architecture. Such specificity is needed both to vet our hypotheses and to achieve our practical aims of building useful agent systems that are explainable, extensible, and worthy of our trust, even in critical domains. However, although the LEIA example of shapes-based modeling is specific, the principles can be applied more broadly, giving new life to knowledge-based and hybrid AI.
>
---
#### [new 056] AsyMoE: Leveraging Modal Asymmetry for Enhanced Expert Specialization in Large Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出AsyMoE架构，解决LVLM中视觉与语言模态不对称导致的专家专业化不足问题。通过设计三种专家组，提升跨模态交互与上下文保持能力，实验显示其在准确率和参数效率上优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.12715v1](http://arxiv.org/pdf/2509.12715v1)**

> **作者:** Heng Zhang; Haichuan Hu; Yaomin Shen; Weihao Yu; Yilei Yuan; Haochen You; Guo Cheng; Zijian Zhang; Lubin Gan; Huihui Wei; Hao Zhang; Jin Huang
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive performance on multimodal tasks through scaled architectures and extensive training. However, existing Mixture of Experts (MoE) approaches face challenges due to the asymmetry between visual and linguistic processing. Visual information is spatially complete, while language requires maintaining sequential context. As a result, MoE models struggle to balance modality-specific features and cross-modal interactions. Through systematic analysis, we observe that language experts in deeper layers progressively lose contextual grounding and rely more on parametric knowledge rather than utilizing the provided visual and linguistic information. To address this, we propose AsyMoE, a novel architecture that models this asymmetry using three specialized expert groups. We design intra-modality experts for modality-specific processing, hyperbolic inter-modality experts for hierarchical cross-modal interactions, and evidence-priority language experts to suppress parametric biases and maintain contextual grounding. Extensive experiments demonstrate that AsyMoE achieves 26.58% and 15.45% accuracy improvements over vanilla MoE and modality-specific MoE respectively, with 25.45% fewer activated parameters than dense models.
>
---
## 更新

#### [replaced 001] STRIVE: Structured Representation Integrating VLM Reasoning for Efficient Object Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.06729v2](http://arxiv.org/pdf/2505.06729v2)**

> **作者:** Haokun Zhu; Zongtai Li; Zhixuan Liu; Wenshan Wang; Ji Zhang; Jonathan Francis; Jean Oh
>
> **备注:** We remove OSG and CogNav from Table. 1 for a fair comparison
>
> **摘要:** Vision-Language Models (VLMs) have been increasingly integrated into object navigation tasks for their rich prior knowledge and strong reasoning abilities. However, applying VLMs to navigation poses two key challenges: effectively representing complex environment information and determining \textit{when and how} to query VLMs. Insufficient environment understanding and over-reliance on VLMs (e.g. querying at every step) can lead to unnecessary backtracking and reduced navigation efficiency, especially in continuous environments. To address these challenges, we propose a novel framework that constructs a multi-layer representation of the environment during navigation. This representation consists of viewpoint, object nodes, and room nodes. Viewpoints and object nodes facilitate intra-room exploration and accurate target localization, while room nodes support efficient inter-room planning. Building on this representation, we propose a novel two-stage navigation policy, integrating high-level planning guided by VLM reasoning with low-level VLM-assisted exploration to efficiently locate a goal object. We evaluated our approach on three simulated benchmarks (HM3D, RoboTHOR, and MP3D), and achieved state-of-the-art performance on both the success rate ($\mathord{\uparrow}\, 7.1\%$) and navigation efficiency ($\mathord{\uparrow}\, 12.5\%$). We further validate our method on a real robot platform, demonstrating strong robustness across 15 object navigation tasks in 10 different indoor environments. Project page is available at https://zwandering.github.io/STRIVE.github.io/ .
>
---
#### [replaced 002] Keypoint-based Diffusion for Robotic Motion Planning on the NICOL Robot
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.04076v2](http://arxiv.org/pdf/2509.04076v2)**

> **作者:** Lennart Clasmeier; Jan-Gerrit Habekost; Connor Gäde; Philipp Allgeuer; Stefan Wermter
>
> **备注:** Accepted and published at the 34th International Conference on Artificial Neural Networks (ICANN 2025)
>
> **摘要:** We propose a novel diffusion-based action model for robotic motion planning. Commonly, established numerical planning approaches are used to solve general motion planning problems, but have significant runtime requirements. By leveraging the power of deep learning, we are able to achieve good results in a much smaller runtime by learning from a dataset generated by these planners. While our initial model uses point cloud embeddings in the input to predict keypoint-based joint sequences in its output, we observed in our ablation study that it remained challenging to condition the network on the point cloud embeddings. We identified some biases in our dataset and refined it, which improved the model's performance. Our model, even without the use of the point cloud encodings, outperforms numerical models by an order of magnitude regarding the runtime, while reaching a success rate of up to 90% of collision free solutions on the test set.
>
---
#### [replaced 003] FEWT: Improving Humanoid Robot Perception with Frequency-Enhanced Wavelet-based Transformers
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.11109v2](http://arxiv.org/pdf/2509.11109v2)**

> **作者:** Jiaxin Huang; Hanyu Liu; Yunsheng Ma; Jian Shen; Yilin Zheng; Jiayi Wen; Baishu Wan; Pan Li; Zhigong Song
>
> **摘要:** The embodied intelligence bridges the physical world and information space. As its typical physical embodiment, humanoid robots have shown great promise through robot learning algorithms in recent years. In this study, a hardware platform, including humanoid robot and exoskeleton-style teleoperation cabin, was developed to realize intuitive remote manipulation and efficient collection of anthropomorphic action data. To improve the perception representation of humanoid robot, an imitation learning framework, termed Frequency-Enhanced Wavelet-based Transformer (FEWT), was proposed, which consists of two primary modules: Frequency-Enhanced Efficient Multi-Scale Attention (FE-EMA) and Time-Series Discrete Wavelet Transform (TS-DWT). By combining multi-scale wavelet decomposition with the residual network, FE-EMA can dynamically fuse features from both cross-spatial and frequency-domain. This fusion is able to capture feature information across various scales effectively, thereby enhancing model robustness. Experimental performance demonstrates that FEWT improves the success rate of the state-of-the-art algorithm (Action Chunking with Transformers, ACT baseline) by up to 30% in simulation and by 6-12% in real-world.
>
---
#### [replaced 004] GBPP: Grasp-Aware Base Placement Prediction for Robots via Two-Stage Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.11594v2](http://arxiv.org/pdf/2509.11594v2)**

> **作者:** Jizhuo Chen; Diwen Liu; Jiaming Wang; Harold Soh
>
> **备注:** This paper needs major revision
>
> **摘要:** GBPP is a fast learning based scorer that selects a robot base pose for grasping from a single RGB-D snapshot. The method uses a two stage curriculum: (1) a simple distance-visibility rule auto-labels a large dataset at low cost; and (2) a smaller set of high fidelity simulation trials refines the model to match true grasp outcomes. A PointNet++ style point cloud encoder with an MLP scores dense grids of candidate poses, enabling rapid online selection without full task-and-motion optimization. In simulation and on a real mobile manipulator, GBPP outperforms proximity and geometry only baselines, choosing safer and more reachable stances and degrading gracefully when wrong. The results offer a practical recipe for data efficient, geometry aware base placement: use inexpensive heuristics for coverage, then calibrate with targeted simulation.
>
---
#### [replaced 005] Sign Language: Towards Sign Understanding for Robot Autonomy
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02556v2](http://arxiv.org/pdf/2506.02556v2)**

> **作者:** Ayush Agrawal; Joel Loo; Nicky Zimmerman; David Hsu
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Navigational signs are common aids for human wayfinding and scene understanding, but are underutilized by robots. We argue that they benefit robot navigation and scene understanding, by directly encoding privileged information on actions, spatial regions, and relations. Interpreting signs in open-world settings remains a challenge owing to the complexity of scenes and signs, but recent advances in vision-language models (VLMs) make this feasible. To advance progress in this area, we introduce the task of navigational sign understanding which parses locations and associated directions from signs. We offer a benchmark for this task, proposing appropriate evaluation metrics and curating a test set capturing signs with varying complexity and design across diverse public spaces, from hospitals to shopping malls to transport hubs. We also provide a baseline approach using VLMs, and demonstrate their promise on navigational sign understanding. Code and dataset are available on Github.
>
---
#### [replaced 006] Towards Bio-Inspired Robotic Trajectory Planning via Self-Supervised RNN
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.02171v2](http://arxiv.org/pdf/2507.02171v2)**

> **作者:** Miroslav Cibula; Kristína Malinovská; Matthias Kerzel
>
> **备注:** 12 pages, 4 figures, 2 tables. To be published in 2025 International Conference on Artificial Neural Networks (ICANN) proceedings. This research was funded by the Horizon Europe project TERAIS, GA no. 101079338, and in part by the Slovak Grant Agency for Science (VEGA), project 1/0373/23. The code can be found at https://doi.org/10.5281/zenodo.17127997
>
> **摘要:** Trajectory planning in robotics is understood as generating a sequence of joint configurations that will lead a robotic agent, or its manipulator, from an initial state to the desired final state, thus completing a manipulation task while considering constraints like robot kinematics and the environment. Typically, this is achieved via sampling-based planners, which are computationally intensive. Recent advances demonstrate that trajectory planning can also be performed by supervised sequence learning of trajectories, often requiring only a single or fixed number of passes through a neural architecture, thus ensuring a bounded computation time. Such fully supervised approaches, however, perform imitation learning; they do not learn based on whether the trajectories can successfully reach a goal, but try to reproduce observed trajectories. In our work, we build on this approach and propose a cognitively inspired self-supervised learning scheme based on a recurrent architecture for building a trajectory model. We evaluate the feasibility of the proposed method on a task of kinematic planning for a robotic arm. The results suggest that the model is able to learn to generate trajectories only using given paired forward and inverse kinematics models, and indicate that this novel method could facilitate planning for more complex manipulation tasks requiring adaptive solutions.
>
---
#### [replaced 007] TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.11683v5](http://arxiv.org/pdf/2411.11683v5)**

> **作者:** Xianlong Wang; Hewen Pan; Hangtao Zhang; Minghui Li; Shengshan Hu; Ziqi Zhou; Lulu Xue; Aishan Liu; Yunpeng Jiang; Leo Yu Zhang; Xiaohua Jia
>
> **摘要:** Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.
>
---
#### [replaced 008] Traversing the Narrow Path: A Two-Stage Reinforcement Learning Framework for Humanoid Beam Walking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.20661v3](http://arxiv.org/pdf/2508.20661v3)**

> **作者:** TianChen Huang; Runchen Xu; Yu Wang; Wei Gao; Shiwu Zhang
>
> **备注:** Project website: https://huangtc233.github.io/Traversing-the-Narrow-Path/
>
> **摘要:** Traversing narrow paths is challenging for humanoid robots due to the sparse and safety-critical footholds required. Purely template-based or end-to-end reinforcement learning-based methods suffer from such harsh terrains. This paper proposes a two stage training framework for such narrow path traversing tasks, coupling a template-based foothold planner with a low-level foothold tracker from Stage-I training and a lightweight perception aided foothold modifier from Stage-II training. With the curriculum setup from flat ground to narrow paths across stages, the resulted controller in turn learns to robustly track and safely modify foothold targets to ensure precise foot placement over narrow paths. This framework preserves the interpretability from the physics-based template and takes advantage of the generalization capability from reinforcement learning, resulting in easy sim-to-real transfer. The learned policies outperform purely template-based or reinforcement learning-based baselines in terms of success rate, centerline adherence and safety margins. Validation on a Unitree G1 humanoid robot yields successful traversal of a 0.2m wide and 3m long beam for 20 trials without any failure.
>
---
#### [replaced 009] Learning Environment-Aware Affordance for 3D Articulated Object Manipulation under Occlusions
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2309.07510v5](http://arxiv.org/pdf/2309.07510v5)**

> **作者:** Ruihai Wu; Kai Cheng; Yan Shen; Chuanruo Ning; Guanqi Zhan; Hao Dong
>
> **备注:** In 37th Conference on Neural Information Processing Systems (NeurIPS 2023). Website at https://chengkaiacademycity.github.io/EnvAwareAfford/
>
> **摘要:** Perceiving and manipulating 3D articulated objects in diverse environments is essential for home-assistant robots. Recent studies have shown that point-level affordance provides actionable priors for downstream manipulation tasks. However, existing works primarily focus on single-object scenarios with homogeneous agents, overlooking the realistic constraints imposed by the environment and the agent's morphology, e.g., occlusions and physical limitations. In this paper, we propose an environment-aware affordance framework that incorporates both object-level actionable priors and environment constraints. Unlike object-centric affordance approaches, learning environment-aware affordance faces the challenge of combinatorial explosion due to the complexity of various occlusions, characterized by their quantities, geometries, positions and poses. To address this and enhance data efficiency, we introduce a novel contrastive affordance learning framework capable of training on scenes containing a single occluder and generalizing to scenes with complex occluder combinations. Experiments demonstrate the effectiveness of our proposed approach in learning affordance considering environment constraints. Project page at https://chengkaiacademycity.github.io/EnvAwareAfford/
>
---
#### [replaced 010] Evaluating the Robustness of Open-Source Vision-Language Models to Domain Shift in Object Captioning
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.19579v2](http://arxiv.org/pdf/2506.19579v2)**

> **作者:** Federico Tavella; Amber Drinkwater; Angelo Cangelosi
>
> **摘要:** Vision-Language Models (VLMs) have emerged as powerful tools for generating textual descriptions from visual data. While these models excel on web-scale datasets, their robustness to the domain shifts inherent in many real-world applications remains under-explored. This paper presents a systematic evaluation of VLM performance on a single-view object captioning task when faced with a controlled, physical domain shift. We compare captioning accuracy across two distinct object sets: a collection of multi-material, real-world tools and a set of single-material, 3D-printed items. The 3D-printed set introduces a significant domain shift in texture and material properties, challenging the models' generalization capabilities. Our quantitative results demonstrate that all tested VLMs show a marked performance degradation when describing the 3D-printed objects compared to the real-world tools. This underscores a critical limitation in the ability of current models to generalize beyond surface-level features and highlights the need for more robust architectures for real-world signal processing applications.
>
---
#### [replaced 011] Plane Detection and Ranking via Model Information Optimization
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.09625v2](http://arxiv.org/pdf/2508.09625v2)**

> **作者:** Daoxin Zhong; Jun Li; Meng Yee Michael Chuah
>
> **备注:** Accepted as contributed paper in the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Plane detection from depth images is a crucial subtask with broad robotic applications, often accomplished by iterative methods such as Random Sample Consensus (RANSAC). While RANSAC is a robust strategy with strong probabilistic guarantees, the ambiguity of its inlier threshold criterion makes it susceptible to false positive plane detections. This issue is particularly prevalent in complex real-world scenes, where the true number of planes is unknown and multiple planes coexist. In this paper, we aim to address this limitation by proposing a generalised framework for plane detection based on model information optimization. Building on previous works, we treat the observed depth readings as discrete random variables, with their probability distributions constrained by the ground truth planes. Various models containing different candidate plane constraints are then generated through repeated random sub-sampling to explain our observations. By incorporating the physics and noise model of the depth sensor, we can calculate the information for each model, and the model with the least information is accepted as the most likely ground truth. This information optimization process serves as an objective mechanism for determining the true number of planes and preventing false positive detections. Additionally, the quality of each detected plane can be ranked by summing the information reduction of inlier points for each plane. We validate these properties through experiments with synthetic data and find that our algorithm estimates plane parameters more accurately compared to the default Open3D RANSAC plane segmentation. Furthermore, we accelerate our algorithm by partitioning the depth map using neural network segmentation, which enhances its ability to generate more realistic plane parameters in real-world data.
>
---
#### [replaced 012] ForceVLA: Enhancing VLA Models with a Force-aware MoE for Contact-rich Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.22159v2](http://arxiv.org/pdf/2505.22159v2)**

> **作者:** Jiawen Yu; Hairuo Liu; Qiaojun Yu; Jieji Ren; Ce Hao; Haitong Ding; Guangyu Huang; Guofan Huang; Yan Song; Panpan Cai; Cewu Lu; Wenqiang Zhang
>
> **摘要:** Vision-Language-Action (VLA) models have advanced general-purpose robotic manipulation by leveraging pretrained visual and linguistic representations. However, they struggle with contact-rich tasks that require fine-grained control involving force, especially under visual occlusion or dynamic uncertainty. To address these limitations, we propose ForceVLA, a novel end-to-end manipulation framework that treats external force sensing as a first-class modality within VLA systems. ForceVLA introduces FVLMoE, a force-aware Mixture-of-Experts fusion module that dynamically integrates pretrained visual-language embeddings with real-time 6-axis force feedback during action decoding. This enables context-aware routing across modality-specific experts, enhancing the robot's ability to adapt to subtle contact dynamics. We also introduce \textbf{ForceVLA-Data}, a new dataset comprising synchronized vision, proprioception, and force-torque signals across five contact-rich manipulation tasks. ForceVLA improves average task success by 23.2% over strong pi_0-based baselines, achieving up to 80% success in tasks such as plug insertion. Our approach highlights the importance of multimodal integration for dexterous manipulation and sets a new benchmark for physically intelligent robotic control. Code and data will be released at https://sites.google.com/view/forcevla2025.
>
---
#### [replaced 013] Built Different: Tactile Perception to Overcome Cross-Embodiment Capability Differences in Collaborative Manipulation
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.14896v2](http://arxiv.org/pdf/2409.14896v2)**

> **作者:** William van den Bogert; Madhavan Iyengar; Nima Fazeli
>
> **备注:** 8 pages including references, 8 figures, 2 tables, submitted to ICRA 2026
>
> **摘要:** Tactile sensing is a widely-studied means of implicit communication between robot and human. In this paper, we investigate how tactile sensing can help bridge differences between robotic embodiments in the context of collaborative manipulation. For a robot, learning and executing force-rich collaboration require compliance to human interaction. While compliance is often achieved with admittance control, many commercial robots lack the joint torque monitoring needed for such control. To address this challenge, we present an approach that uses tactile sensors and behavior cloning to transfer policies from robots with these capabilities to those without. We train a single policy that demonstrates positive transfer across embodiments, including robots without torque sensing. We demonstrate this positive transfer on four different tactile-enabled embodiments using the same policy trained on force-controlled robot data. Across multiple proposed metrics, the best performance came from a decomposed tactile shear-field representation combined with a pre-trained encoder, which improved success rates over alternative representations.
>
---
#### [replaced 014] FCRF: Flexible Constructivism Reflection for Long-Horizon Robotic Task Planning with Large Language Models
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.14975v2](http://arxiv.org/pdf/2507.14975v2)**

> **作者:** Yufan Song; Jiatao Zhang; Zeng Gu; Qingmiao Liang; Tuocheng Hu; Wei Song; Shiqiang Zhu
>
> **备注:** 8 pages, 6 figures, IROS 2025
>
> **摘要:** Autonomous error correction is critical for domestic robots to achieve reliable execution of complex long-horizon tasks. Prior work has explored self-reflection in Large Language Models (LLMs) for task planning error correction; however, existing methods are constrained by inflexible self-reflection mechanisms that limit their effectiveness. Motivated by these limitations and inspired by human cognitive adaptation, we propose the Flexible Constructivism Reflection Framework (FCRF), a novel Mentor-Actor architecture that enables LLMs to perform flexible self-reflection based on task difficulty, while constructively integrating historical valuable experience with failure lessons. We evaluated FCRF on diverse domestic tasks through simulation in AlfWorld and physical deployment in the real-world environment. Experimental results demonstrate that FCRF significantly improves overall performance and self-reflection flexibility in complex long-horizon robotic tasks.
>
---
#### [replaced 015] Search-TTA: A Multimodal Test-Time Adaptation Framework for Visual Search in the Wild
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11350v3](http://arxiv.org/pdf/2505.11350v3)**

> **作者:** Derek Ming Siang Tan; Shailesh; Boyang Liu; Alok Raj; Qi Xuan Ang; Weiheng Dai; Tanishq Duhan; Jimmy Chiun; Yuhong Cao; Florian Shkurti; Guillaume Sartoretti
>
> **备注:** Accepted for presentation at CORL 2025. [Link to Paper Website](https://search-tta.github.io/)
>
> **摘要:** To perform outdoor autonomous visual navigation and search, a robot may leverage satellite imagery as a prior map. This can help inform high-level search and exploration strategies, even when such images lack sufficient resolution to allow for visual recognition of targets. However, there are limited training datasets of satellite images with annotated targets that are not directly visible. Furthermore, approaches which leverage large Vision Language Models (VLMs) for generalization may yield inaccurate outputs due to hallucination, leading to inefficient search. To address these challenges, we introduce Search-TTA, a multimodal test-time adaptation framework with a flexible plug-and-play interface compatible with various input modalities (e.g. image, text, sound) and planning methods. First, we pretrain a satellite image encoder to align with CLIP's visual encoder to output probability distributions of target presence used for visual search. Second, our framework dynamically refines CLIP's predictions during search using a test-time adaptation mechanism. Through a novel feedback loop inspired by Spatial Poisson Point Processes, uncertainty-weighted gradient updates are used to correct potentially inaccurate predictions and improve search performance. To train and evaluate Search-TTA, we curate AVS-Bench, a visual search dataset based on internet-scale ecological data that contains up to 380k training and 8k validation images (in- and out-domain). We find that Search-TTA improves planner performance by up to 30.0%, particularly in cases with poor initial CLIP predictions due to limited training data. It also performs comparably with significantly larger VLMs, and achieves zero-shot generalization to unseen modalities. Finally, we deploy Search-TTA on a real UAV via hardware-in-the-loop testing, by simulating its operation within a large-scale simulation that provides onboard sensing.
>
---
#### [replaced 016] Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.05356v2](http://arxiv.org/pdf/2509.05356v2)**

> **作者:** Justus Huebotter; Pablo Lanillos; Marcel van Gerven; Serge Thill
>
> **摘要:** Despite recent progress in training spiking neural networks (SNNs) for classification, their application to continuous motor control remains limited. Here, we demonstrate that fully spiking architectures can be trained end-to-end to control robotic arms with multiple degrees of freedom in continuous environments. Our predictive-control framework combines Leaky Integrate-and-Fire dynamics with surrogate gradients, jointly optimizing a forward model for dynamics prediction and a policy network for goal-directed action. We evaluate this approach on both a planar 2D reaching task and a simulated 6-DOF Franka Emika Panda robot. Results show that SNNs can achieve stable training and accurate torque control, establishing their viability for high-dimensional motor tasks. An extensive ablation study highlights the role of initialization, learnable time constants, and regularization in shaping training dynamics. We conclude that while stable and effective control can be achieved, recurrent spiking networks remain highly sensitive to hyperparameter settings, underscoring the importance of principled design choices.
>
---
#### [replaced 017] TransDiffuser: Diverse Trajectory Generation with Decorrelated Multi-modal Representation for End-to-end Autonomous Driving
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.09315v2](http://arxiv.org/pdf/2505.09315v2)**

> **作者:** Xuefeng Jiang; Yuan Ma; Pengxiang Li; Leimeng Xu; Xin Wen; Kun Zhan; Zhongpu Xia; Peng Jia; Xianpeng Lang; Sheng Sun
>
> **备注:** Under review
>
> **摘要:** In recent years, diffusion models have demonstrated remarkable potential across diverse domains, from vision generation to language modeling. Transferring its generative capabilities to modern end-to-end autonomous driving systems has also emerged as a promising direction. However, existing diffusion-based trajectory generative models often exhibit mode collapse where different random noises converge to similar trajectories after the denoising process.Therefore, state-of-the-art models often rely on anchored trajectories from pre-defined trajectory vocabulary or scene priors in the training set to mitigate collapse and enrich the diversity of generated trajectories, but such inductive bias are not available in real-world deployment, which can be challenged when generalizing to unseen scenarios. In this work, we investigate the possibility of effectively tackling the mode collapse challenge without the assumption of pre-defined trajectory vocabulary or pre-computed scene priors. Specifically, we propose TransDiffuser, an encoder-decoder based generative trajectory planning model, where the encoded scene information and motion states serve as the multi-modal conditional input of the denoising decoder. Different from existing approaches, we exploit a simple yet effective multi-modal representation decorrelation optimization mechanism during the denoising process to enrich the latent representation space which better guides the downstream generation. Without any predefined trajectory anchors or pre-computed scene priors, TransDiffuser achieves the PDMS of 94.85 on the closed-loop planning-oriented benchmark NAVSIM, surpassing previous state-of-the-art methods. Qualitative evaluation further showcases TransDiffuser generates more diverse and plausible trajectories which explore more drivable area.
>
---
#### [replaced 018] RoboMatch: A Unified Mobile-Manipulation Teleoperation Platform with Auto-Matching Network Architecture for Long-Horizon Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.08522v2](http://arxiv.org/pdf/2509.08522v2)**

> **作者:** Hanyu Liu; Yunsheng Ma; Jiaxin Huang; Keqiang Ren; Jiayi Wen; Yilin Zheng; Baishu Wan; Pan Li; Jiejun Hou; Haoru Luan; Zhihua Wang; Zhigong Song
>
> **摘要:** This paper presents RoboMatch, a novel unified teleoperation platform for mobile manipulation with an auto-matching network architecture, designed to tackle long-horizon tasks in dynamic environments. Our system enhances teleoperation performance, data collection efficiency, task accuracy, and operational stability. The core of RoboMatch is a cockpit-style control interface that enables synchronous operation of the mobile base and dual arms, significantly improving control precision and data collection. Moreover, we introduce the Proprioceptive-Visual Enhanced Diffusion Policy (PVE-DP), which leverages Discrete Wavelet Transform (DWT) for multi-scale visual feature extraction and integrates high-precision IMUs at the end-effector to enrich proprioceptive feedback, substantially boosting fine manipulation performance. Furthermore, we propose an Auto-Matching Network (AMN) architecture that decomposes long-horizon tasks into logical sequences and dynamically assigns lightweight pre-trained models for distributed inference. Experimental results demonstrate that our approach improves data collection efficiency by over 20%, increases task success rates by 20-30% with PVE-DP, and enhances long-horizon inference performance by approximately 40% with AMN, offering a robust solution for complex manipulation tasks.
>
---
#### [replaced 019] Data-fused Model Predictive Control with Guarantees: Application to Flying Humanoid Robots
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2509.10353v2](http://arxiv.org/pdf/2509.10353v2)**

> **作者:** Davide Gorbani; Mohamed Elobaid; Giuseppe L'Erario; Hosameldin Awadalla Omer Mohamed; Daniele Pucci
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** This paper introduces a Data-Fused Model Predictive Control (DFMPC) framework that combines physics-based models with data-driven representations of unknown dynamics. Leveraging Willems' Fundamental Lemma and an artificial equilibrium formulation, the method enables tracking of changing, potentially unreachable setpoints while explicitly handling measurement noise through slack variables and regularization. We provide guarantees of recursive feasibility and practical stability under input-output constraints for a specific class of reference signals. The approach is validated on the iRonCub flying humanoid robot, integrating analytical momentum models with data-driven turbine dynamics. Simulations show improved tracking and robustness compared to a purely model-based MPC, while maintaining real-time feasibility.
>
---
#### [replaced 020] Towards Autonomous In-situ Soil Sampling and Mapping in Large-Scale Agricultural Environments
- **分类: cs.RO; cs.ET**

- **链接: [http://arxiv.org/pdf/2506.05653v3](http://arxiv.org/pdf/2506.05653v3)**

> **作者:** Thien Hoang Nguyen; Erik Muller; Michael Rubin; Xiaofei Wang; Fiorella Sibona; Alex McBratney; Salah Sukkarieh
>
> **备注:** Presented at the 2025 IEEE ICRA Workshop on Field Robotics
>
> **摘要:** Traditional soil sampling and analysis methods are labor-intensive, time-consuming, and limited in spatial resolution, making them unsuitable for large-scale precision agriculture. To address these limitations, we present a robotic solution for real-time sampling, analysis and mapping of key soil properties. Our system consists of two main sub-systems: a Sample Acquisition System (SAS) for precise, automated in-field soil sampling; and a Sample Analysis Lab (Lab) for real-time soil property analysis. The system's performance was validated through extensive field trials at a large-scale Australian farm. Experimental results show that the SAS can consistently acquire soil samples with a mass of 50g at a depth of 200mm, while the Lab can process each sample within 10 minutes to accurately measure pH and macronutrients. These results demonstrate the potential of the system to provide farmers with timely, data-driven insights for more efficient and sustainable soil management and fertilizer application.
>
---
#### [replaced 021] Multi-objective task allocation for electric harvesting robots: a hierarchical route reconstruction approach
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.11025v2](http://arxiv.org/pdf/2509.11025v2)**

> **作者:** Peng Chen; Jing Liang; Hui Song; Kang-Jia Qiao; Cai-Tong Yue; Kun-Jie Yu; Ponnuthurai Nagaratnam Suganthan; Witold Pedrycz
>
> **摘要:** The increasing labor costs in agriculture have accelerated the adoption of multi-robot systems for orchard harvesting. However, efficiently coordinating these systems is challenging due to the complex interplay between makespan and energy consumption, particularly under practical constraints like load-dependent speed variations and battery limitations. This paper defines the multi-objective agricultural multi-electrical-robot task allocation (AMERTA) problem, which systematically incorporates these often-overlooked real-world constraints. To address this problem, we propose a hybrid hierarchical route reconstruction algorithm (HRRA) that integrates several innovative mechanisms, including a hierarchical encoding structure, a dual-phase initialization method, task sequence optimizers, and specialized route reconstruction operators. Extensive experiments on 45 test instances demonstrate HRRA's superior performance against seven state-of-the-art algorithms. Statistical analysis, including the Wilcoxon signed-rank and Friedman tests, empirically validates HRRA's competitiveness and its unique ability to explore previously inaccessible regions of the solution space. In general, this research contributes to the theoretical understanding of multi-robot coordination by offering a novel problem formulation and an effective algorithm, thereby also providing practical insights for agricultural automation.
>
---
