# 机器人 cs.RO

- **最新发布 36 篇**

- **更新 20 篇**

## 最新发布

#### [new 001] VER: Vision Expert Transformer for Robot Learning via Foundation Distillation and Dynamic Routing
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人学习任务，旨在解决单一视觉模型泛化能力不足的问题。论文提出VER框架，通过预训练整合多个视觉模型，仅微调轻量路由网络选择专家模型，动态聚焦关键区域，提升多任务性能。**

- **链接: [http://arxiv.org/pdf/2510.05213v1](http://arxiv.org/pdf/2510.05213v1)**

> **作者:** Yixiao Wang; Mingxiao Huo; Zhixuan Liang; Yushi Du; Lingfeng Sun; Haotian Lin; Jinghuan Shang; Chensheng Peng; Mohit Bansal; Mingyu Ding; Masayoshi Tomizuka
>
> **摘要:** Pretrained vision foundation models (VFMs) advance robotic learning via rich visual representations, yet individual VFMs typically excel only in specific domains, limiting generality across tasks. Distilling multiple VFMs into a unified representation for policy can mitigate this limitation but often yields inflexible task-specific feature selection and requires costly full re-training to incorporate robot-domain knowledge. We propose VER, a Vision Expert transformer for Robot learning. During pretraining, VER distills multiple VFMs into a vision expert library. It then fine-tunes only a lightweight routing network (fewer than 0.4% of parameters) to dynamically select task-relevant experts from the pretrained library for downstream robot tasks. We further introduce Patchwise Expert Routing with Curriculum Top-K Annealing to improve both flexibility and precision of dynamic expert selection. Moreover, VER supports parameter-efficient finetuning for scalable expert utilization and adaptive robot-domain knowledge integration. Across 17 diverse robotic tasks and multiple policy heads, VER achieves state-of-the-art performance. We find that VER reduces large-norm outliers in task-irrelevant regions (e.g., background) and concentrates on task-critical regions. Visualizations and codes can be found in https://yixiaowang7.github.io/ver_page/.
>
---
#### [new 002] ARRC: Advanced Reasoning Robot Control - Knowledge-Driven Autonomous Manipulation Using Retrieval-Augmented Generation
- **分类: cs.RO**

- **简介: 论文提出ARRC系统，结合检索增强生成与机器人感知控制，实现从自然语言指令到安全操作的自主执行。属于机器人控制任务，解决语言到动作的可靠转换与安全操控问题，通过知识驱动方法提升计划有效性与适应性。**

- **链接: [http://arxiv.org/pdf/2510.05547v1](http://arxiv.org/pdf/2510.05547v1)**

> **作者:** Eugene Vorobiov; Ammar Jaleel Mahmood; Salim Rezvani; Robin Chhabra
>
> **摘要:** We present ARRC (Advanced Reasoning Robot Control), a practical system that connects natural-language instructions to safe local robotic control by combining Retrieval-Augmented Generation (RAG) with RGB-D perception and guarded execution on an affordable robot arm. The system indexes curated robot knowledge (movement patterns, task templates, and safety heuristics) in a vector database, retrieves task-relevant context for each instruction, and conditions a large language model (LLM) to produce JSON-structured action plans. Plans are executed on a UFactory xArm 850 fitted with a Dynamixel-driven parallel gripper and an Intel RealSense D435 camera. Perception uses AprilTag detections fused with depth to produce object-centric metric poses. Execution is enforced via software safety gates: workspace bounds, speed and force caps, timeouts, and bounded retries. We describe the architecture, knowledge design, integration choices, and a reproducible evaluation protocol for tabletop scan, approach, and pick-place tasks. Experimental results demonstrate the efficacy of the proposed approach. Our design shows that RAG-based planning can substantially improve plan validity and adaptability while keeping perception and low-level control local to the robot.
>
---
#### [new 003] Human-in-the-loop Optimisation in Robot-assisted Gait Training
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人辅助步态训练任务，旨在解决个体间和个体内步态差异对控制策略的影响。研究采用人类闭环优化方法（HILO）结合CMA-ES算法，个性化优化外骨骼下肢关节刚度参数。实验结果显示算法能收敛到个体化参数，但未显著提升受试者表现，揭示了人机共适应和人类行为变异性对个性化控制效果的影响。**

- **链接: [http://arxiv.org/pdf/2510.05780v1](http://arxiv.org/pdf/2510.05780v1)**

> **作者:** Andreas Christou; Andreas Sochopoulos; Elliot Lister; Sethu Vijayakumar
>
> **摘要:** Wearable robots offer a promising solution for quantitatively monitoring gait and providing systematic, adaptive assistance to promote patient independence and improve gait. However, due to significant interpersonal and intrapersonal variability in walking patterns, it is important to design robot controllers that can adapt to the unique characteristics of each individual. This paper investigates the potential of human-in-the-loop optimisation (HILO) to deliver personalised assistance in gait training. The Covariance Matrix Adaptation Evolution Strategy (CMA-ES) was employed to continuously optimise an assist-as-needed controller of a lower-limb exoskeleton. Six healthy individuals participated over a two-day experiment. Our results suggest that while the CMA-ES appears to converge to a unique set of stiffnesses for each individual, no measurable impact on the subjects' performance was observed during the validation trials. These findings highlight the impact of human-robot co-adaptation and human behaviour variability, whose effect may be greater than potential benefits of personalising rule-based assistive controllers. Our work contributes to understanding the limitations of current personalisation approaches in exoskeleton-assisted gait rehabilitation and identifies key challenges for effective implementation of human-in-the-loop optimisation in this domain.
>
---
#### [new 004] The DISTANT Design for Remote Transmission and Steering Systems for Planetary Robotics
- **分类: cs.RO**

- **简介: 论文提出了一种适用于行星探测机器人的远程传动与转向系统设计（DISTANT），旨在解决极端环境下长期运行中的热循环、粉尘污染和机械磨损问题。通过将驱动和转向部件移至探测器内部温控舱，采用双叉臂悬架与卡丹关节等结构，实现独立控制各轮的驱动、转向与悬架。**

- **链接: [http://arxiv.org/pdf/2510.05981v1](http://arxiv.org/pdf/2510.05981v1)**

> **作者:** Cristina Luna; Alba Guerra; Almudena Moreno; Manuel Esquer; Willy Roa; Mateusz Krawczak; Robert Popela; Piotr Osica; Davide Nicolis
>
> **备注:** Paper for 18th Symposium on Advanced Space Technologies in Robotics and Automation (ASTRA), presented on October 7th at Leiden, Netherlands
>
> **摘要:** Planetary exploration missions require robust locomotion systems capable of operating in extreme environments over extended periods. This paper presents the DISTANT (Distant Transmission and Steering Systems) design, a novel approach for relocating rover traction and steering actuators from wheel-mounted positions to a thermally protected warm box within the rover body. The design addresses critical challenges in long-distance traversal missions by protecting sensitive components from thermal cycling, dust contamination, and mechanical wear. A double wishbone suspension configuration with cardan joints and capstan drive steering has been selected as the optimal architecture following comprehensive trade-off analysis. The system enables independent wheel traction, steering control, and suspension management whilst maintaining all motorisation within the protected environment. The design meets a 50 km traverse requirement without performance degradation, with integrated dust protection mechanisms and thermal management solutions. Testing and validation activities are planned for Q1 2026 following breadboard manufacturing at 1:3 scale.
>
---
#### [new 005] Precise and Efficient Collision Prediction under Uncertainty in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的碰撞风险预测任务，旨在解决在不确定驾驶条件下，如何精确高效地估计规划轨迹的碰撞概率问题。论文提出了两种半解析方法，分别基于空间重叠概率和随机边界穿越来计算碰撞概率，兼顾精度与计算效率，适用于实时轨迹规划。**

- **链接: [http://arxiv.org/pdf/2510.05729v1](http://arxiv.org/pdf/2510.05729v1)**

> **作者:** Marc Kaufeld; Johannes Betz
>
> **备注:** 8 pages, submitted to the IEEE ICRA 2026, Vienna, Austria
>
> **摘要:** This research introduces two efficient methods to estimate the collision risk of planned trajectories in autonomous driving under uncertain driving conditions. Deterministic collision checks of planned trajectories are often inaccurate or overly conservative, as noisy perception, localization errors, and uncertain predictions of other traffic participants introduce significant uncertainty into the planning process. This paper presents two semi-analytic methods to compute the collision probability of planned trajectories with arbitrary convex obstacles. The first approach evaluates the probability of spatial overlap between an autonomous vehicle and surrounding obstacles, while the second estimates the collision probability based on stochastic boundary crossings. Both formulations incorporate full state uncertainties, including position, orientation, and velocity, and achieve high accuracy at computational costs suitable for real-time planning. Simulation studies verify that the proposed methods closely match Monte Carlo results while providing significant runtime advantages, enabling their use in risk-aware trajectory planning. The collision estimation methods are available as open-source software: https://github.com/TUM-AVS/Collision-Probability-Estimation
>
---
#### [new 006] Coordinate-Consistent Localization via Continuous-Time Calibration and Fusion of UWB and SLAM Observations
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，旨在解决SLAM坐标不一致和UWB依赖锚点坐标的问题。通过连续时间优化方法，第一阶段校准UWB锚点位置，第二阶段融合UWB与SLAM数据，实现跨会话的一致且精确的定位。**

- **链接: [http://arxiv.org/pdf/2510.05992v1](http://arxiv.org/pdf/2510.05992v1)**

> **作者:** Tien-Dat Nguyen; Thien-Minh Nguyen; Vinh-Hao Nguyen
>
> **摘要:** Onboard simultaneous localization and mapping (SLAM) methods are commonly used to provide accurate localization information for autonomous robots. However, the coordinate origin of SLAM estimate often resets for each run. On the other hand, UWB-based localization with fixed anchors can ensure a consistent coordinate reference across sessions; however, it requires an accurate assignment of the anchor nodes' coordinates. To this end, we propose a two-stage approach that calibrates and fuses UWB data and SLAM data to achieve coordinate-wise consistent and accurate localization in the same environment. In the first stage, we solve a continuous-time batch optimization problem by using the range and odometry data from one full run, incorporating height priors and anchor-to-anchor distance factors to recover the anchors' 3D positions. For the subsequent runs in the second stage, a sliding-window optimization scheme fuses the UWB and SLAM data, which facilitates accurate localization in the same coordinate system. Experiments are carried out on the NTU VIRAL dataset with six scenarios of UAV flight, and we show that calibration using data in one run is sufficient to enable accurate localization in the remaining runs. We release our source code to benefit the community at https://github.com/ntdathp/slam-uwb-calibration.
>
---
#### [new 007] VCoT-Grasp: Grasp Foundation Models with Visual Chain-of-Thought Reasoning for Language-driven Grasp Generation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，旨在解决语言驱动抓取中推理能力弱、泛化性差及依赖复杂模块的问题。作者提出了VCoT-Grasp模型，通过视觉链式推理增强视觉理解，并构建了大规模数据集VCoT-GraspSet。实验表明该方法在杂乱环境中提升了抓取成功率，且能泛化至新物体和背景。**

- **链接: [http://arxiv.org/pdf/2510.05827v1](http://arxiv.org/pdf/2510.05827v1)**

> **作者:** Haoran Zhang; Shuanghao Bai; Wanqi Zhou; Yuedi Zhang; Qi Zhang; Pengxiang Ding; Cheng Chi; Donglin Wang; Badong Chen
>
> **摘要:** Robotic grasping is one of the most fundamental tasks in robotic manipulation, and grasp detection/generation has long been the subject of extensive research. Recently, language-driven grasp generation has emerged as a promising direction due to its practical interaction capabilities. However, most existing approaches either lack sufficient reasoning and generalization capabilities or depend on complex modular pipelines. Moreover, current grasp foundation models tend to overemphasize dialog and object semantics, resulting in inferior performance and restriction to single-object grasping. To maintain strong reasoning ability and generalization in cluttered environments, we propose VCoT-Grasp, an end-to-end grasp foundation model that incorporates visual chain-of-thought reasoning to enhance visual understanding for grasp generation. VCoT-Grasp adopts a multi-turn processing paradigm that dynamically focuses on visual inputs while providing interpretable reasoning traces. For training, we refine and introduce a large-scale dataset, VCoT-GraspSet, comprising 167K synthetic images with over 1.36M grasps, as well as 400+ real-world images with more than 1.2K grasps, annotated with intermediate bounding boxes. Extensive experiments on both VCoT-GraspSet and real robot demonstrate that our method significantly improves grasp success rates and generalizes effectively to unseen objects, backgrounds, and distractors. More details can be found at https://zhanghr2001.github.io/VCoT-Grasp.github.io.
>
---
#### [new 008] GO-Flock: Goal-Oriented Flocking in 3D Unknown Environments with Depth Maps
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于无人机群控制任务，旨在解决在复杂三维未知环境中实现高效避障与群体导航的问题。现有方法易陷入局部最优且效率低。论文提出GO-Flock框架，结合感知模块与新型势场控制策略，实现高效避障与群体行为，并通过仿真与实物实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2510.05553v1](http://arxiv.org/pdf/2510.05553v1)**

> **作者:** Yan Rui Tan; Wenqi Liu; Wai Lun Leong; John Guan Zhong Tan; Wayne Wen Huei Yong; Fan Shi; Rodney Swee Huat Teo
>
> **摘要:** Artificial Potential Field (APF) methods are widely used for reactive flocking control, but they often suffer from challenges such as deadlocks and local minima, especially in the presence of obstacles. Existing solutions to address these issues are typically passive, leading to slow and inefficient collective navigation. As a result, many APF approaches have only been validated in obstacle-free environments or simplified, pseudo 3D simulations. This paper presents GO-Flock, a hybrid flocking framework that integrates planning with reactive APF-based control. GO-Flock consists of an upstream Perception Module, which processes depth maps to extract waypoints and virtual agents for obstacle avoidance, and a downstream Collective Navigation Module, which applies a novel APF strategy to achieve effective flocking behavior in cluttered environments. We evaluate GO-Flock against passive APF-based approaches to demonstrate their respective merits, such as their flocking behavior and the ability to overcome local minima. Finally, we validate GO-Flock through obstacle-filled environment and also hardware-in-the-loop experiments where we successfully flocked a team of nine drones, six physical and three virtual, in a forest environment.
>
---
#### [new 009] Vision-Guided Targeted Grasping and Vibration for Robotic Pollination in Controlled Environments
- **分类: cs.RO**

- **简介: 该论文属于农业机器人任务，旨在解决受控环境中缺乏自然授粉手段的问题。作者提出一种结合视觉引导抓取与振动建模的机器人授粉系统，通过3D植物重建、目标抓取规划和物理仿真优化振动策略。实验验证了抓取成功率高且授粉有效，实现了无损操作。**

- **链接: [http://arxiv.org/pdf/2510.06146v1](http://arxiv.org/pdf/2510.06146v1)**

> **作者:** Jaehwan Jeong; Tuan-Anh Vu; Radha Lahoti; Jiawen Wang; Vivek Alumootil; Sangpil Kim; M. Khalid Jawed
>
> **摘要:** Robotic pollination offers a promising alternative to manual labor and bumblebee-assisted methods in controlled agriculture, where wind-driven pollination is absent and regulatory restrictions limit the use of commercial pollinators. In this work, we present and validate a vision-guided robotic framework that uses data from an end-effector mounted RGB-D sensor and combines 3D plant reconstruction, targeted grasp planning, and physics-based vibration modeling to enable precise pollination. First, the plant is reconstructed in 3D and registered to the robot coordinate frame to identify obstacle-free grasp poses along the main stem. Second, a discrete elastic rod model predicts the relationship between actuation parameters and flower dynamics, guiding the selection of optimal pollination strategies. Finally, a manipulator with soft grippers grasps the stem and applies controlled vibrations to induce pollen release. End-to-end experiments demonstrate a 92.5\% main-stem grasping success rate, and simulation-guided optimization of vibration parameters further validates the feasibility of our approach, ensuring that the robot can safely and effectively perform pollination without damaging the flower. To our knowledge, this is the first robotic system to jointly integrate vision-based grasping and vibration modeling for automated precision pollination.
>
---
#### [new 010] Towards Online Robot Interaction Adaptation to Human Upper-limb Mobility Impairments in Return-to-Work Scenarios
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决上肢运动障碍者在重返工作场景中的协作问题。论文提出一种在线自适应框架，将用户关节活动限制模型集成至分层控制器，使机器人能实时调整行为，引导用户利用残余运动能力完成任务。实验验证了方法在不同肢体障碍下的有效性。**

- **链接: [http://arxiv.org/pdf/2510.05425v1](http://arxiv.org/pdf/2510.05425v1)**

> **作者:** Marta Lagomarsino; Francesco Tassi
>
> **摘要:** Work environments are often inadequate and lack inclusivity for individuals with upper-body disabilities. This paper presents a novel online framework for adaptive human-robot interaction (HRI) that accommodates users' arm mobility impairments, ultimately aiming to promote active work participation. Unlike traditional human-robot collaboration approaches that assume able-bodied users, our method integrates a mobility model for specific joint limitations into a hierarchical optimal controller. This allows the robot to generate reactive, mobility-aware behaviour online and guides the user's impaired limb to exploit residual functional mobility. The framework was tested in handover tasks involving different upper-limb mobility impairments (i.e., emulated elbow and shoulder arthritis, and wrist blockage), under both standing and seated configurations with task constraints using a mobile manipulator, and complemented by quantitative and qualitative comparisons with state-of-the-art ergonomic HRI approaches. Preliminary results indicated that the framework can personalise the interaction to fit within the user's impaired range of motion and encourage joint usage based on the severity of their functional limitations.
>
---
#### [new 011] Oracle-Guided Masked Contrastive Reinforcement Learning for Visuomotor Policies
- **分类: cs.RO; cs.LG**

- **简介: 论文提出OMC-RL框架，用于提升视觉运动策略学习的样本效率和性能。该方法通过解耦为表征学习与策略学习两个阶段，利用掩码Transformer提取时序视觉特征，并引入教师策略提供初期指导，逐步过渡到自主探索，解决了高维视觉输入与动作输出带来的样本效率低和仿真到现实迁移难的问题。**

- **链接: [http://arxiv.org/pdf/2510.05692v1](http://arxiv.org/pdf/2510.05692v1)**

> **作者:** Yuhang Zhang; Jiaping Xiao; Chao Yan; Mir Feroskhan
>
> **摘要:** A prevailing approach for learning visuomotor policies is to employ reinforcement learning to map high-dimensional visual observations directly to action commands. However, the combination of high-dimensional visual inputs and agile maneuver outputs leads to long-standing challenges, including low sample efficiency and significant sim-to-real gaps. To address these issues, we propose Oracle-Guided Masked Contrastive Reinforcement Learning (OMC-RL), a novel framework designed to improve the sample efficiency and asymptotic performance of visuomotor policy learning. OMC-RL explicitly decouples the learning process into two stages: an upstream representation learning stage and a downstream policy learning stage. In the upstream stage, a masked Transformer module is trained with temporal modeling and contrastive learning to extract temporally-aware and task-relevant representations from sequential visual inputs. After training, the learned encoder is frozen and used to extract visual representations from consecutive frames, while the Transformer module is discarded. In the downstream stage, an oracle teacher policy with privileged access to global state information supervises the agent during early training to provide informative guidance and accelerate early policy learning. This guidance is gradually reduced to allow independent exploration as training progresses. Extensive experiments in simulated and real-world environments demonstrate that OMC-RL achieves superior sample efficiency and asymptotic policy performance, while also improving generalization across diverse and perceptually complex scenarios.
>
---
#### [new 012] DeLTa: Demonstration and Language-Guided Novel Transparent Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人透明物体操作任务，旨在解决现有方法在长视野、精确操作透明物体上的局限性。作者提出了DeLTa框架，结合深度估计、6D姿态估计与视觉语言规划，实现基于自然指令的单次示范泛化，提升对新透明物体的操作能力。**

- **链接: [http://arxiv.org/pdf/2510.05662v1](http://arxiv.org/pdf/2510.05662v1)**

> **作者:** Taeyeop Lee; Gyuree Kang; Bowen Wen; Youngho Kim; Seunghyeok Back; In So Kweon; David Hyunchul Shim; Kuk-Jin Yoon
>
> **备注:** Project page: https://sites.google.com/view/DeLTa25/
>
> **摘要:** Despite the prevalence of transparent object interactions in human everyday life, transparent robotic manipulation research remains limited to short-horizon tasks and basic grasping capabilities.Although some methods have partially addressed these issues, most of them have limitations in generalizability to novel objects and are insufficient for precise long-horizon robot manipulation. To address this limitation, we propose DeLTa (Demonstration and Language-Guided Novel Transparent Object Manipulation), a novel framework that integrates depth estimation, 6D pose estimation, and vision-language planning for precise long-horizon manipulation of transparent objects guided by natural task instructions. A key advantage of our method is its single-demonstration approach, which generalizes 6D trajectories to novel transparent objects without requiring category-level priors or additional training. Additionally, we present a task planner that refines the VLM-generated plan to account for the constraints of a single-arm, eye-in-hand robot for long-horizon object manipulation tasks. Through comprehensive evaluation, we demonstrate that our method significantly outperforms existing transparent object manipulation approaches, particularly in long-horizon scenarios requiring precise manipulation capabilities. Project page: https://sites.google.com/view/DeLTa25/
>
---
#### [new 013] A Preview of HoloOcean 2.0
- **分类: cs.RO**

- **简介: 论文介绍了HoloOcean 2.0，属于海洋机器人仿真任务。旨在提升海洋机器人开发与验证的仿真精度与功能。工作包括升级至Unreal Engine 5.3，集成高级动力学模型，支持ROS2，并开发高效声呐模拟、语义传感器与真实环境效果。**

- **链接: [http://arxiv.org/pdf/2510.06160v1](http://arxiv.org/pdf/2510.06160v1)**

> **作者:** Blake Romrell; Abigail Austin; Braden Meyers; Ryan Anderson; Carter Noh; Joshua G. Mangelson
>
> **备注:** 5 pages, 9 figures, submitted to the ICRA 2025 aq2uasim workshop
>
> **摘要:** Marine robotics simulators play a fundamental role in the development of marine robotic systems. With increased focus on the marine robotics field in recent years, there has been significant interest in developing higher fidelitysimulation of marine sensors, physics, and visual rendering capabilities to support autonomous marine robot development and validation. HoloOcean 2.0, the next major release of HoloOcean, brings state-of-the-art features under a general marine simulator capable of supporting a variety of tasks. New features in HoloOcean 2.0 include migration to Unreal Engine (UE) 5.3, advanced vehicle dynamics using models from Fossen, and support for ROS2 using a custom bridge. Additional features are currently in development, including significantly more efficient ray tracing-based sidescan, forward-looking, and bathymetric sonar implementations; semantic sensors; environment generation tools; volumetric environmental effects; and realistic waves.
>
---
#### [new 014] Federated Split Learning for Resource-Constrained Robots in Industrial IoT: Framework Comparison, Optimization Strategies, and Future Directions
- **分类: cs.RO; cs.AI; cs.MA; cs.SY; eess.SY**

- **简介: 该论文研究工业物联网中资源受限机器人的联邦拆分学习（FedSL），属于机器学习与物联网交叉任务。旨在解决数据隐私、通信效率与设备异构性问题。工作包括对比不同FedSL框架、分类融合策略、提出优化方法，并通过仿真验证性能，最后指出未来方向。**

- **链接: [http://arxiv.org/pdf/2510.05713v1](http://arxiv.org/pdf/2510.05713v1)**

> **作者:** Wanli Ni; Hui Tian; Shuai Wang; Chengyang Li; Lei Sun; Zhaohui Yang
>
> **备注:** 9 pages, 5 figures, submitted to the IEEE magazine
>
> **摘要:** Federated split learning (FedSL) has emerged as a promising paradigm for enabling collaborative intelligence in industrial Internet of Things (IoT) systems, particularly in smart factories where data privacy, communication efficiency, and device heterogeneity are critical concerns. In this article, we present a comprehensive study of FedSL frameworks tailored for resource-constrained robots in industrial scenarios. We compare synchronous, asynchronous, hierarchical, and heterogeneous FedSL frameworks in terms of workflow, scalability, adaptability, and limitations under dynamic industrial conditions. Furthermore, we systematically categorize token fusion strategies into three paradigms: input-level (pre-fusion), intermediate-level (intra-fusion), and output-level (post-fusion), and summarize their respective strengths in industrial applications. We also provide adaptive optimization techniques to enhance the efficiency and feasibility of FedSL implementation, including model compression, split layer selection, computing frequency allocation, and wireless resource management. Simulation results validate the performance of these frameworks under industrial detection scenarios. Finally, we outline open issues and research directions of FedSL in future smart manufacturing systems.
>
---
#### [new 015] A Co-Design Framework for Energy-Aware Monoped Jumping with Detailed Actuator Modeling
- **分类: cs.RO**

- **简介: 该论文属于机器人机械设计与控制协同优化任务，旨在解决单腿跳跃机器人在跳高与能耗间的权衡问题。现有方法忽略齿轮箱参数和真实电机质量模型，导致设计难以实现。论文提出三阶段协同优化框架，同时优化机械设计（含齿轮箱）与控制参数，自动生成可直接制造的CAD模型。实验表明，相比基线设计，能耗降低50%，跳高达到0.8米。**

- **链接: [http://arxiv.org/pdf/2510.05923v1](http://arxiv.org/pdf/2510.05923v1)**

> **作者:** Aman Singh; Aastha Mishra; Deepak Kapa; Suryank Joshi; Shishir Kolathaya
>
> **备注:** 7 pages, 8 figures, 1 table, Accepted at IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids) 2025, Aman Singh, Aastha Mishra - Authors contributed equally
>
> **摘要:** A monoped's jump height and energy consumption depend on both, its mechanical design and control strategy. Existing co-design frameworks typically optimize for either maximum height or minimum energy, neglecting their trade-off. They also often omit gearbox parameter optimization and use oversimplified actuator mass models, producing designs difficult to replicate in practice. In this work, we introduce a novel three-stage co-design optimization framework that jointly maximizes jump height while minimizing mechanical energy consumption of a monoped. The proposed method explicitly incorporates realistic actuator mass models and optimizes mechanical design (including gearbox) and control parameters within a unified framework. The resulting design outputs are then used to automatically generate a parameterized CAD model suitable for direct fabrication, significantly reducing manual design iterations. Our experimental evaluations demonstrate a 50 percent reduction in mechanical energy consumption compared to the baseline design, while achieving a jump height of 0.8m. Video presentation is available at http://y2u.be/XW8IFRCcPgM
>
---
#### [new 016] Stable Robot Motions on Manifolds: Learning Lyapunov-Constrained Neural Manifold ODEs
- **分类: cs.RO; cs.LG; math.OC**

- **简介: 该论文属于机器人运动学习任务，旨在解决在黎曼流形上学习稳定动力系统的挑战。为确保机器人轨迹的稳定性，作者提出了基于神经流形常微分方程的方法，通过李雅普诺夫稳定性约束，使系统在满足流形几何约束的同时实现稳定运动。**

- **链接: [http://arxiv.org/pdf/2510.05707v1](http://arxiv.org/pdf/2510.05707v1)**

> **作者:** David Boetius; Abdelrahman Abdelnaby; Ashok Kumar; Stefan Leue; Abdalla Swikir; Fares J. Abu-Dakka
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** Learning stable dynamical systems from data is crucial for safe and reliable robot motion planning and control. However, extending stability guarantees to trajectories defined on Riemannian manifolds poses significant challenges due to the manifold's geometric constraints. To address this, we propose a general framework for learning stable dynamical systems on Riemannian manifolds using neural ordinary differential equations. Our method guarantees stability by projecting the neural vector field evolving on the manifold so that it strictly satisfies the Lyapunov stability criterion, ensuring stability at every system state. By leveraging a flexible neural parameterisation for both the base vector field and the Lyapunov function, our framework can accurately represent complex trajectories while respecting manifold constraints by evolving solutions directly on the manifold. We provide an efficient training strategy for applying our framework and demonstrate its utility by solving Riemannian LASA datasets on the unit quaternion (S^3) and symmetric positive-definite matrix manifolds, as well as robotic motions evolving on \mathbb{R}^3 \times S^3. We demonstrate the performance, scalability, and practical applicability of our approach through extensive simulations and by learning robot motions in a real-world experiment.
>
---
#### [new 017] AD-NODE: Adaptive Dynamics Learning with Neural ODEs for Mobile Robots Control
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于移动机器人控制任务，旨在解决不确定环境中机器人动力学模型适应性问题。作者提出AD-NODE方法，结合神经微分方程与两阶段训练，学习环境隐表示，实现无需直接环境信息的自适应控制。实验验证了其在多种机器人平台上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.05443v1](http://arxiv.org/pdf/2510.05443v1)**

> **作者:** Shao-Yi Yu; Jen-Wei Wang; Maya Horii; Vikas Garg; Tarek Zohdi
>
> **摘要:** Mobile robots, such as ground vehicles and quadrotors, are becoming increasingly important in various fields, from logistics to agriculture, where they automate processes in environments that are difficult to access for humans. However, to perform effectively in uncertain environments using model-based controllers, these systems require dynamics models capable of responding to environmental variations, especially when direct access to environmental information is limited. To enable such adaptivity and facilitate integration with model predictive control, we propose an adaptive dynamics model which bypasses the need for direct environmental knowledge by inferring operational environments from state-action history. The dynamics model is based on neural ordinary equations, and a two-phase training procedure is used to learn latent environment representations. We demonstrate the effectiveness of our approach through goal-reaching and path-tracking tasks on three robotic platforms of increasing complexity: a 2D differential wheeled robot with changing wheel contact conditions, a 3D quadrotor in variational wind fields, and the Sphero BOLT robot under two contact conditions for real-world deployment. Empirical results corroborate that our method can handle temporally and spatially varying environmental changes in both simulation and real-world systems.
>
---
#### [new 018] Correlation-Aware Dual-View Pose and Velocity Estimation for Dynamic Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人动态操作中的状态估计任务，旨在解决机械臂对运动目标的高精度位姿与速度估计问题。论文提出了一种基于双视角视觉传感器的去中心化融合方法，利用自适应扩展卡尔曼滤波和相关性感知融合规则，在李群流形上进行状态估计与融合。实验验证了该方法在目标跟踪中的有效性与优越性。**

- **链接: [http://arxiv.org/pdf/2510.05536v1](http://arxiv.org/pdf/2510.05536v1)**

> **作者:** Mahboubeh Zarei; Robin Chhabra; Farrokh Janabi-Sharifi
>
> **摘要:** Accurate pose and velocity estimation is essential for effective spatial task planning in robotic manipulators. While centralized sensor fusion has traditionally been used to improve pose estimation accuracy, this paper presents a novel decentralized fusion approach to estimate both pose and velocity. We use dual-view measurements from an eye-in-hand and an eye-to-hand vision sensor configuration mounted on a manipulator to track a target object whose motion is modeled as random walk (stochastic acceleration model). The robot runs two independent adaptive extended Kalman filters formulated on a matrix Lie group, developed as part of this work. These filters predict poses and velocities on the manifold $\mathbb{SE}(3) \times \mathbb{R}^3 \times \mathbb{R}^3$ and update the state on the manifold $\mathbb{SE}(3)$. The final fused state comprising the fused pose and velocities of the target is obtained using a correlation-aware fusion rule on Lie groups. The proposed method is evaluated on a UFactory xArm 850 equipped with Intel RealSense cameras, tracking a moving target. Experimental results validate the effectiveness and robustness of the proposed decentralized dual-view estimation framework, showing consistent improvements over state-of-the-art methods.
>
---
#### [new 019] AI-Enabled Capabilities to Facilitate Next-Generation Rover Surface Operations
- **分类: cs.RO**

- **简介: 该论文旨在提升行星探测车的地面操作效率，属于机器人与人工智能交叉任务。主要解决当前探测车行驶速度慢、自主性不足的问题。工作包括开发FASTNAV障碍检测系统、CISRU多机器人协作框架，以及基于深度学习的地形分类系统，通过实地验证展示了其在速度、准确性和安全性方面的提升。**

- **链接: [http://arxiv.org/pdf/2510.05985v1](http://arxiv.org/pdf/2510.05985v1)**

> **作者:** Cristina Luna; Robert Field; Steven Kay
>
> **备注:** Paper for 18th Symposium on Advanced Space Technologies in Robotics and Automation (ASTRA), presented on October 7th at Leiden, Netherlands
>
> **摘要:** Current planetary rovers operate at traverse speeds of approximately 10 cm/s, fundamentally limiting exploration efficiency. This work presents integrated AI systems which significantly improve autonomy through three components: (i) the FASTNAV Far Obstacle Detector (FOD), capable of facilitating sustained 1.0 m/s speeds via computer vision-based obstacle detection; (ii) CISRU, a multi-robot coordination framework enabling human-robot collaboration for in-situ resource utilisation; and (iii) the ViBEKO and AIAXR deep learning-based terrain classification studies. Field validation in Mars analogue environments demonstrated these systems at Technology Readiness Level 4, providing measurable improvements in traverse speed, classification accuracy, and operational safety for next-generation planetary missions.
>
---
#### [new 020] Adaptive Dynamics Planning for Robot Navigation
- **分类: cs.RO**

- **简介: 论文提出了一种自适应动力学规划（ADP）方法，用于机器人导航任务。它通过强化学习动态调整动力学保真度，解决传统方法在复杂环境中计算开销大或动力学考虑不足的问题，提升了导航的成功率、安全性和效率。**

- **链接: [http://arxiv.org/pdf/2510.05330v1](http://arxiv.org/pdf/2510.05330v1)**

> **作者:** Lu Yuanjie; Mao Mingyang; Xu Tong; Wang Linji; Lin Xiaomin; Xiao Xuesu
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Autonomous robot navigation systems often rely on hierarchical planning, where global planners compute collision-free paths without considering dynamics, and local planners enforce dynamics constraints to produce executable commands. This discontinuity in dynamics often leads to trajectory tracking failure in highly constrained environments. Recent approaches integrate dynamics within the entire planning process by gradually decreasing its fidelity, e.g., increasing integration steps and reducing collision checking resolution, for real-time planning efficiency. However, they assume that the fidelity of the dynamics should decrease according to a manually designed scheme. Such static settings fail to adapt to environmental complexity variations, resulting in computational overhead in simple environments or insufficient dynamics consideration in obstacle-rich scenarios. To overcome this limitation, we propose Adaptive Dynamics Planning (ADP), a learning-augmented paradigm that uses reinforcement learning to dynamically adjust robot dynamics properties, enabling planners to adapt across diverse environments. We integrate ADP into three different planners and further design a standalone ADP-based navigation system, benchmarking them against other baselines. Experiments in both simulation and real-world tests show that ADP consistently improves navigation success, safety, and efficiency.
>
---
#### [new 021] A multi-modal tactile fingertip design for robotic hands to enhance dexterous manipulation
- **分类: cs.RO**

- **简介: 该论文设计了一种低成本、紧凑的多模态触觉指尖，用于提升机器人手的灵巧操作能力。主要解决触觉感知在机器人操作中因成本、制造难度和信号提取困难而受限的问题。通过集成应变片和接触式麦克风传感器，实现力和振动检测，并成功应用于不同视觉条件下的操作任务，如纸杯计数与分离。**

- **链接: [http://arxiv.org/pdf/2510.05382v1](http://arxiv.org/pdf/2510.05382v1)**

> **作者:** Zhuowei Xu; Zilin Si; Kevin Zhang; Oliver Kroemer; Zeynep Temel
>
> **摘要:** Tactile sensing holds great promise for enhancing manipulation precision and versatility, but its adoption in robotic hands remains limited due to high sensor costs, manufacturing and integration challenges, and difficulties in extracting expressive and reliable information from signals. In this work, we present a low-cost, easy-to-make, adaptable, and compact fingertip design for robotic hands that integrates multi-modal tactile sensors. We use strain gauge sensors to capture static forces and a contact microphone sensor to measure high-frequency vibrations during contact. These tactile sensors are integrated into a compact design with a minimal sensor footprint, and all sensors are internal to the fingertip and therefore not susceptible to direct wear and tear from interactions. From sensor characterization, we show that strain gauge sensors provide repeatable 2D planar force measurements in the 0-5 N range and the contact microphone sensor has the capability to distinguish contact material properties. We apply our design to three dexterous manipulation tasks that range from zero to full visual occlusion. Given the expressiveness and reliability of tactile sensor readings, we show that different tactile sensing modalities can be used flexibly in different stages of manipulation, solely or together with visual observations to achieve improved task performance. For instance, we can precisely count and unstack a desired number of paper cups from a stack with 100\% success rate which is hard to achieve with vision only.
>
---
#### [new 022] Learning to Crawl: Latent Model-Based Reinforcement Learning for Soft Robotic Adaptive Locomotion
- **分类: cs.RO**

- **简介: 论文研究软体机器人爬行控制，属于强化学习与机器人自适应运动任务。旨在解决模型不准确、传感器噪声干扰和运动步态难以设计的问题。工作提出一种基于潜在动力学模型的强化学习框架，通过传感器数据学习动力学模型，指导策略优化，实现仅依赖噪声反馈的自适应爬行运动。**

- **链接: [http://arxiv.org/pdf/2510.05957v1](http://arxiv.org/pdf/2510.05957v1)**

> **作者:** Vaughn Gzenda; Robin Chhabra
>
> **摘要:** Soft robotic crawlers are mobile robots that utilize soft body deformability and compliance to achieve locomotion through surface contact. Designing control strategies for such systems is challenging due to model inaccuracies, sensor noise, and the need to discover locomotor gaits. In this work, we present a model-based reinforcement learning (MB-RL) framework in which latent dynamics inferred from onboard sensors serve as a predictive model that guides an actor-critic algorithm to optimize locomotor policies. We evaluate the framework on a minimal crawler model in simulation using inertial measurement units and time-of-flight sensors as observations. The learned latent dynamics enable short-horizon motion prediction while the actor-critic discovers effective locomotor policies. This approach highlights the potential of latent-dynamics MB-RL for enabling embodied soft robotic adaptive locomotion based solely on noisy sensor feedback.
>
---
#### [new 023] Multi-Robot Distributed Optimization for Exploration and Mapping of Unknown Environments using Bioinspired Tactile-Sensor
- **分类: cs.RO**

- **简介: 论文提出了一种受生物启发的多机器人系统，用于未知环境的探索与地图构建。该研究属于多机器人协同任务，旨在解决未知环境中高效探索与地图拼接的问题。工作内容包括设计基于触觉传感器的分布式优化策略，实现机器人自主避障与协作建图，并通过实验验证系统在覆盖率、碰撞减少和地图精度上的有效性。**

- **链接: [http://arxiv.org/pdf/2510.06085v1](http://arxiv.org/pdf/2510.06085v1)**

> **作者:** Roman Ibrahimov; Jannik Matthias Heinen
>
> **摘要:** This project proposes a bioinspired multi-robot system using Distributed Optimization for efficient exploration and mapping of unknown environments. Each robot explores its environment and creates a map, which is afterwards put together to form a global 2D map of the environment. Inspired by wall-following behaviors, each robot autonomously explores its neighborhood based on a tactile sensor, similar to the antenna of a cockroach, mounted on the surface of the robot. Instead of avoiding obstacles, robots log collision points when they touch obstacles. This decentralized control strategy ensures effective task allocation and efficient exploration of unknown terrains, with applications in search and rescue, industrial inspection, and environmental monitoring. The approach was validated through experiments using e-puck robots in a simulated 1.5 x 1.5 m environment with three obstacles. The results demonstrated the system's effectiveness in achieving high coverage, minimizing collisions, and constructing accurate 2D maps.
>
---
#### [new 024] Cross-Embodiment Dexterous Hand Articulation Generation via Morphology-Aware Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人手抓取生成任务，旨在解决不同结构机械手的灵巧抓取泛化问题。作者提出一种基于特征抓取的端到端框架，通过形态感知学习，结合手部形态嵌入、物体点云和腕部姿态，预测关节运动。方法使用低维空间回归并引入KAL损失，实现跨形态抓取，在模拟和真实实验中表现优异。**

- **链接: [http://arxiv.org/pdf/2510.06068v1](http://arxiv.org/pdf/2510.06068v1)**

> **作者:** Heng Zhang; Kevin Yuchen Ma; Mike Zheng Shou; Weisi Lin; Yan Wu
>
> **摘要:** Dexterous grasping with multi-fingered hands remains challenging due to high-dimensional articulations and the cost of optimization-based pipelines. Existing end-to-end methods require training on large-scale datasets for specific hands, limiting their ability to generalize across different embodiments. We propose an eigengrasp-based, end-to-end framework for cross-embodiment grasp generation. From a hand's morphology description, we derive a morphology embedding and an eigengrasp set. Conditioned on these, together with the object point cloud and wrist pose, an amplitude predictor regresses articulation coefficients in a low-dimensional space, which are decoded into full joint articulations. Articulation learning is supervised with a Kinematic-Aware Articulation Loss (KAL) that emphasizes fingertip-relevant motions and injects morphology-specific structure. In simulation on unseen objects across three dexterous hands, our model attains a 91.9% average grasp success rate with less than 0.4 seconds inference per grasp. With few-shot adaptation to an unseen hand, it achieves 85.6% success on unseen objects in simulation, and real-world experiments on this few-shot generalized hand achieve an 87% success rate. The code and additional materials will be made available upon publication on our project website https://connor-zh.github.io/cross_embodiment_dexterous_grasping.
>
---
#### [new 025] EmbodiedCoder: Parameterized Embodied Mobile Manipulation via Modern Coding Model
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决现有方法依赖大量标注数据、泛化性差的问题。作者提出EmbodiedCoder，一种无需训练的框架，利用代码模型直接生成可执行的机器人轨迹，实现开放环境中移动机器人的灵活操作。**

- **链接: [http://arxiv.org/pdf/2510.06207v1](http://arxiv.org/pdf/2510.06207v1)**

> **作者:** Zefu Lin; Rongxu Cui; Chen Hanning; Xiangyu Wang; Junjia Xu; Xiaojuan Jin; Chen Wenbo; Hui Zhou; Lue Fan; Wenling Li; Zhaoxiang Zhang
>
> **备注:** Demo Page: https://anonymous.4open.science/w/Embodied-Coder/
>
> **摘要:** Recent advances in control robot methods, from end-to-end vision-language-action frameworks to modular systems with predefined primitives, have advanced robots' ability to follow natural language instructions. Nonetheless, many approaches still struggle to scale to diverse environments, as they often rely on large annotated datasets and offer limited interpretability.In this work, we introduce EmbodiedCoder, a training-free framework for open-world mobile robot manipulation that leverages coding models to directly generate executable robot trajectories. By grounding high-level instructions in code, EmbodiedCoder enables flexible object geometry parameterization and manipulation trajectory synthesis without additional data collection or fine-tuning.This coding-based paradigm provides a transparent and generalizable way to connect perception with manipulation. Experiments on real mobile robots show that EmbodiedCoder achieves robust performance across diverse long-term tasks and generalizes effectively to novel objects and environments.Our results demonstrate an interpretable approach for bridging high-level reasoning and low-level control, moving beyond fixed primitives toward versatile robot intelligence. See the project page at: https://anonymous.4open.science/w/Embodied-Coder/
>
---
#### [new 026] DYMO-Hair: Generalizable Volumetric Dynamics Modeling for Robot Hair Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在解决因头发复杂动力学特性导致的机器人自主护发难题。作者提出了DYMO-Hair系统，通过基于模型的动态学习与规划方法，实现了对多种发型的高效、精准护发操作。**

- **链接: [http://arxiv.org/pdf/2510.06199v1](http://arxiv.org/pdf/2510.06199v1)**

> **作者:** Chengyang Zhao; Uksang Yoo; Arkadeep Narayan Chaudhury; Giljoo Nam; Jonathan Francis; Jeffrey Ichnowski; Jean Oh
>
> **备注:** Project page: https://chengyzhao.github.io/DYMOHair-web/
>
> **摘要:** Hair care is an essential daily activity, yet it remains inaccessible to individuals with limited mobility and challenging for autonomous robot systems due to the fine-grained physical structure and complex dynamics of hair. In this work, we present DYMO-Hair, a model-based robot hair care system. We introduce a novel dynamics learning paradigm that is suited for volumetric quantities such as hair, relying on an action-conditioned latent state editing mechanism, coupled with a compact 3D latent space of diverse hairstyles to improve generalizability. This latent space is pre-trained at scale using a novel hair physics simulator, enabling generalization across previously unseen hairstyles. Using the dynamics model with a Model Predictive Path Integral (MPPI) planner, DYMO-Hair is able to perform visual goal-conditioned hair styling. Experiments in simulation demonstrate that DYMO-Hair's dynamics model outperforms baselines on capturing local deformation for diverse, unseen hairstyles. DYMO-Hair further outperforms baselines in closed-loop hair styling tasks on unseen hairstyles, with an average of 22% lower final geometric error and 42% higher success rate than the state-of-the-art system. Real-world experiments exhibit zero-shot transferability of our system to wigs, achieving consistent success on challenging unseen hairstyles where the state-of-the-art system fails. Together, these results introduce a foundation for model-based robot hair care, advancing toward more generalizable, flexible, and accessible robot hair styling in unconstrained physical environments. More details are available on our project page: https://chengyzhao.github.io/DYMOHair-web/.
>
---
#### [new 027] Active Semantic Perception
- **分类: cs.RO**

- **简介: 论文提出主动语义感知方法，用于室内环境探索任务。通过构建多层场景图，结合大语言模型预测未观测区域，计算信息增益指导路径点选择，提升语义理解效率与准确性。解决了复杂室内环境探索中语义推理与空间决策的问题。**

- **链接: [http://arxiv.org/pdf/2510.05430v1](http://arxiv.org/pdf/2510.05430v1)**

> **作者:** Huayi Tang; Pratik Chaudhari
>
> **摘要:** We develop an approach for active semantic perception which refers to using the semantics of the scene for tasks such as exploration. We build a compact, hierarchical multi-layer scene graph that can represent large, complex indoor environments at various levels of abstraction, e.g., nodes corresponding to rooms, objects, walls, windows etc. as well as fine-grained details of their geometry. We develop a procedure based on large language models (LLMs) to sample plausible scene graphs of unobserved regions that are consistent with partial observations of the scene. These samples are used to compute an information gain of a potential waypoint for sophisticated spatial reasoning, e.g., the two doors in the living room can lead to either a kitchen or a bedroom. We evaluate this approach in complex, realistic 3D indoor environments in simulation. We show using qualitative and quantitative experiments that our approach can pin down the semantics of the environment quicker and more accurately than baseline approaches.
>
---
#### [new 028] Towards Autonomous Tape Handling for Robotic Wound Redressing
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决慢性伤口护理中粘性胶带操作的自动化问题。通过力反馈模仿学习实现胶带初始剥离，通过轨迹优化实现胶带平整粘贴，验证了自主胶带操作在伤口护理自动化中的可行性。**

- **链接: [http://arxiv.org/pdf/2510.06127v1](http://arxiv.org/pdf/2510.06127v1)**

> **作者:** Xiao Liang; Lu Shen; Peihan Zhang; Soofiyan Atar; Florian Richter; Michael Yip
>
> **摘要:** Chronic wounds, such as diabetic, pressure, and venous ulcers, affect over 6.5 million patients in the United States alone and generate an annual cost exceeding \$25 billion. Despite this burden, chronic wound care remains a routine yet manual process performed exclusively by trained clinicians due to its critical safety demands. We envision a future in which robotics and automation support wound care to lower costs and enhance patient outcomes. This paper introduces an autonomous framework for one of the most fundamental yet challenging subtasks in wound redressing: adhesive tape manipulation. Specifically, we address two critical capabilities: tape initial detachment (TID) and secure tape placement. To handle the complex adhesive dynamics of detachment, we propose a force-feedback imitation learning approach trained from human teleoperation demonstrations. For tape placement, we develop a numerical trajectory optimization method based to ensure smooth adhesion and wrinkle-free application across diverse anatomical surfaces. We validate these methods through extensive experiments, demonstrating reliable performance in both quantitative evaluations and integrated wound redressing pipelines. Our results establish tape manipulation as an essential step toward practical robotic wound care automation.
>
---
#### [new 029] Verifier-free Test-Time Sampling for Vision Language Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决视觉-语言-动作模型（VLAs）在高精度任务中的性能限制。现有方法依赖外部验证器，需额外训练且泛化能力差。作者提出MG-Select，利用模型内部特性，在测试时通过KL散度选择最优动作，结合参考分布与联合训练策略，提升任务表现，尤其在分布外任务和少样本场景中效果显著。**

- **链接: [http://arxiv.org/pdf/2510.05681v1](http://arxiv.org/pdf/2510.05681v1)**

> **作者:** Suhyeok Jang; Dongyoung Kim; Changyeon Kim; Youngsuk Kim; Jinwoo Shin
>
> **备注:** 14 pages; 3 figures
>
> **摘要:** Vision-Language-Action models (VLAs) have demonstrated remarkable performance in robot control. However, they remain fundamentally limited in tasks that require high precision due to their single-inference paradigm. While test-time scaling approaches using external verifiers have shown promise, they require additional training and fail to generalize to unseen conditions. We propose Masking Distribution Guided Selection (MG-Select), a novel test-time scaling framework for VLAs that leverages the model's internal properties without requiring additional training or external modules. Our approach utilizes KL divergence from a reference action token distribution as a confidence metric for selecting the optimal action from multiple candidates. We introduce a reference distribution generated by the same VLA but with randomly masked states and language conditions as inputs, ensuring maximum uncertainty while remaining aligned with the target task distribution. Additionally, we propose a joint training strategy that enables the model to learn both conditional and unconditional distributions by applying dropout to state and language conditions, thereby further improving the quality of the reference distribution. Our experiments demonstrate that MG-Select achieves significant performance improvements, including a 28%/35% improvement in real-world in-distribution/out-of-distribution tasks, along with a 168% relative gain on RoboCasa pick-and-place tasks trained with 30 demonstrations.
>
---
#### [new 030] Hybrid Quantum-Classical Policy Gradient for Adaptive Control of Cyber-Physical Systems: A Comparative Study of VQC vs. MLP
- **分类: quant-ph; cs.AI; cs.LG; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于强化学习与控制任务，旨在比较经典神经网络（MLP）与量子神经网络（VQC）在控制环境中的表现。论文通过在CartPole-v1环境中训练两种模型，评估其收敛性、抗噪性和计算效率。结果显示MLP表现更优，但VQC在参数量和扩展性上具潜力，揭示了量子增强架构的未来可能性。**

- **链接: [http://arxiv.org/pdf/2510.06010v1](http://arxiv.org/pdf/2510.06010v1)**

> **作者:** Aueaphum Aueawatthanaphisut; Nyi Wunna Tun
>
> **备注:** 6 pages, 5 figures, 2 tables, 17 equations, 1 algorithm
>
> **摘要:** The comparative evaluation between classical and quantum reinforcement learning (QRL) paradigms was conducted to investigate their convergence behavior, robustness under observational noise, and computational efficiency in a benchmark control environment. The study employed a multilayer perceptron (MLP) agent as a classical baseline and a parameterized variational quantum circuit (VQC) as a quantum counterpart, both trained on the CartPole-v1 environment over 500 episodes. Empirical results demonstrated that the classical MLP achieved near-optimal policy convergence with a mean return of 498.7 +/- 3.2, maintaining stable equilibrium throughout training. In contrast, the VQC exhibited limited learning capability, with an average return of 14.6 +/- 4.8, primarily constrained by circuit depth and qubit connectivity. Noise robustness analysis further revealed that the MLP policy deteriorated gracefully under Gaussian perturbations, while the VQC displayed higher sensitivity at equivalent noise levels. Despite the lower asymptotic performance, the VQC exhibited significantly lower parameter count and marginally increased training time, highlighting its potential scalability for low-resource quantum processors. The results suggest that while classical neural policies remain dominant in current control benchmarks, quantum-enhanced architectures could offer promising efficiency advantages once hardware noise and expressivity limitations are mitigated.
>
---
#### [new 031] MetaVLA: Unified Meta Co-training For Efficient Embodied Adaption
- **分类: cs.AI; cs.RO**

- **简介: 论文提出MetaVLA，旨在解决视觉-语言-动作（VLA）模型在具身推理中泛化能力差、需任务微调的问题。通过统一的元协同训练框架，整合多样辅助任务，实现高效对齐与快速适应，提升多任务表现并减少训练资源消耗。**

- **链接: [http://arxiv.org/pdf/2510.05580v1](http://arxiv.org/pdf/2510.05580v1)**

> **作者:** Chen Li; Zhantao Yang; Han Zhang; Fangyi Chen; Chenchen Zhu; Anudeepsekhar Bolimera; Marios Savvides
>
> **摘要:** Vision-Language-Action (VLA) models show promise in embodied reasoning, yet remain far from true generalists-they often require task-specific fine-tuning, and generalize poorly to unseen tasks. We propose MetaVLA, a unified, backbone-agnostic post-training framework for efficient and scalable alignment. MetaVLA introduces Context-Aware Meta Co-Training, which consolidates diverse target tasks into a single fine-tuning stage while leveraging structurally diverse auxiliary tasks to improve in-domain generalization. Unlike naive multi-task SFT, MetaVLA integrates a lightweight meta-learning mechanism-derived from Attentive Neural Processes-to enable rapid adaptation from diverse contexts with minimal architectural change or inference overhead. On the LIBERO benchmark, MetaVLA with six auxiliary tasks outperforms OpenVLA by up to 8.0% on long-horizon tasks, reduces training steps from 240K to 75K, and cuts GPU time by ~76%. These results show that scalable, low-resource post-training is achievable-paving the way toward general-purpose embodied agents. Code will be available.
>
---
#### [new 032] Information-Theoretic Policy Pre-Training with Empowerment
- **分类: cs.AI; cs.IT; cs.LG; cs.RO; math.IT**

- **简介: 该论文属于强化学习任务，旨在解决策略预训练以提升下游任务适应效率的问题。论文提出基于信息论的“empowerment”作为预训练信号，引入折扣机制平衡短期与长期控制，通过最大化折扣empowerment实现策略初始化，提升数据效率与任务适应能力。**

- **链接: [http://arxiv.org/pdf/2510.05996v1](http://arxiv.org/pdf/2510.05996v1)**

> **作者:** Moritz Schneider; Robert Krug; Narunas Vaskevicius; Luigi Palmieri; Michael Volpp; Joschka Boedecker
>
> **摘要:** Empowerment, an information-theoretic measure of an agent's potential influence on its environment, has emerged as a powerful intrinsic motivation and exploration framework for reinforcement learning (RL). Besides for unsupervised RL and skill learning algorithms, the specific use of empowerment as a pre-training signal has received limited attention in the literature. We show that empowerment can be used as a pre-training signal for data-efficient downstream task adaptation. For this we extend the traditional notion of empowerment by introducing discounted empowerment, which balances the agent's control over the environment across short- and long-term horizons. Leveraging this formulation, we propose a novel pre-training paradigm that initializes policies to maximize discounted empowerment, enabling agents to acquire a robust understanding of environmental dynamics. We analyze empowerment-based pre-training for various existing RL algorithms and empirically demonstrate its potential as a general-purpose initialization strategy: empowerment-maximizing policies with long horizons are data-efficient and effective, leading to improved adaptability in downstream tasks. Our findings pave the way for future research to scale this framework to high-dimensional and complex tasks, further advancing the field of RL.
>
---
#### [new 033] Dropping the D: RGB-D SLAM Without the Depth Sensor
- **分类: cs.CV; cs.RO**

- **简介: 论文提出DropD-SLAM，一种无需深度传感器的实时单目SLAM系统。它属于SLAM任务，旨在解决单目SLAM缺乏度量尺度、依赖深度传感器的问题。通过使用预训练视觉模块估计深度、检测关键点和分割实例，实现RGB-D级精度，取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2510.06216v1](http://arxiv.org/pdf/2510.06216v1)**

> **作者:** Mert Kiray; Alican Karaomer; Benjamin Busam
>
> **摘要:** We present DropD-SLAM, a real-time monocular SLAM system that achieves RGB-D-level accuracy without relying on depth sensors. The system replaces active depth input with three pretrained vision modules: a monocular metric depth estimator, a learned keypoint detector, and an instance segmentation network. Dynamic objects are suppressed using dilated instance masks, while static keypoints are assigned predicted depth values and backprojected into 3D to form metrically scaled features. These are processed by an unmodified RGB-D SLAM back end for tracking and mapping. On the TUM RGB-D benchmark, DropD-SLAM attains 7.4 cm mean ATE on static sequences and 1.8 cm on dynamic sequences, matching or surpassing state-of-the-art RGB-D methods while operating at 22 FPS on a single GPU. These results suggest that modern pretrained vision models can replace active depth sensors as reliable, real-time sources of metric scale, marking a step toward simpler and more cost-effective SLAM systems.
>
---
#### [new 034] Safety-Critical Control with Bounded Inputs: A Closed-Form Solution for Backup Control Barrier Functions
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于安全控制任务，旨在解决输入受限系统在运行时需满足安全性与输入约束的问题。现有方法需实时求解高维优化问题，计算成本高。论文提出一种可在闭合形式下求解的备份控制屏障函数方法，通过在标称控制器与备份控制器间进行最优插值得到安全控制输入，保证系统安全并满足输入限制。方法在双积分器和非线性固定翼飞机上验证有效。**

- **链接: [http://arxiv.org/pdf/2510.05436v1](http://arxiv.org/pdf/2510.05436v1)**

> **作者:** David E. J. van Wijk; Ersin Das; Tamas G. Molnar; Aaron D. Ames; Joel W. Burdick
>
> **备注:** 8 pages, 6 figures. Code available at https://github.com/davidvwijk/OI-CBF
>
> **摘要:** Verifying the safety of controllers is critical for many applications, but is especially challenging for systems with bounded inputs. Backup control barrier functions (bCBFs) offer a structured approach to synthesizing safe controllers that are guaranteed to satisfy input bounds by leveraging the knowledge of a backup controller. While powerful, bCBFs require solving a high-dimensional quadratic program at run-time, which may be too costly for computationally-constrained systems such as aerospace vehicles. We propose an approach that optimally interpolates between a nominal controller and the backup controller, and we derive the solution to this optimization problem in closed form. We prove that this closed-form controller is guaranteed to be safe while obeying input bounds. We demonstrate the effectiveness of the approach on a double integrator and a nonlinear fixed-wing aircraft example.
>
---
#### [new 035] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于视觉-动作预训练任务，旨在解决实体AI缺乏大规模数据的问题。论文提出了D2E框架，利用桌面环境（如游戏）进行预训练，并通过标准化工具、事件预测模型和迁移方法，将桌面数据迁移到实体AI任务中，取得了良好效果。**

- **链接: [http://arxiv.org/pdf/2510.05684v1](http://arxiv.org/pdf/2510.05684v1)**

> **作者:** Suwhan Choi; Jaeyoon Jung; Haebin Seong; Minchan Kim; Minyeong Kim; Yongjun Cho; Yoonshik Kim; Yubeen Park; Youngjae Yu; Yunsung Lee
>
> **摘要:** Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection. Desktop environments -- particularly gaming -- offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning. We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks. Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains. Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152x compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation. Using 1.3K+ hours of data (259 hours of human demonstrations, and 1K+ hours of pseudo-labeled gameplay), we achieve a total of 96.6% success rate on LIBERO manipulation and 83.3% on CANVAS navigation benchmarks. This validates that sensorimotor primitives in digital interactions exhibit sufficient invariance to transfer meaningfully to physical embodied tasks, establishing desktop pretraining as a practical paradigm for robotics. We will make all our work public, including the OWA toolkit, datasets of human-collected and pseudo-labeled, and VAPT-trained models available at https://worv-ai.github.io/d2e/
>
---
#### [new 036] The Safety Challenge of World Models for Embodied AI Agents: A Review
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文综述了具身智能中世界模型的安全挑战，重点分析自动驾驶和机器人领域。任务是评估模型在场景与控制生成中的安全性，识别常见错误并量化分析，以提升预测安全性。**

- **链接: [http://arxiv.org/pdf/2510.05865v1](http://arxiv.org/pdf/2510.05865v1)**

> **作者:** Lorenzo Baraldi; Zifan Zeng; Chongzhe Zhang; Aradhana Nayak; Hongbo Zhu; Feng Liu; Qunli Zhang; Peng Wang; Shiming Liu; Zheng Hu; Angelo Cangelosi; Lorenzo Baraldi
>
> **摘要:** The rapid progress in embodied artificial intelligence has highlighted the necessity for more advanced and integrated models that can perceive, interpret, and predict environmental dynamics. In this context, World Models (WMs) have been introduced to provide embodied agents with the abilities to anticipate future environmental states and fill in knowledge gaps, thereby enhancing agents' ability to plan and execute actions. However, when dealing with embodied agents it is fundamental to ensure that predictions are safe for both the agent and the environment. In this article, we conduct a comprehensive literature review of World Models in the domains of autonomous driving and robotics, with a specific focus on the safety implications of scene and control generation tasks. Our review is complemented by an empirical analysis, wherein we collect and examine predictions from state-of-the-art models, identify and categorize common faults (herein referred to as pathologies), and provide a quantitative evaluation of the results.
>
---
## 更新

#### [replaced 001] CoTaP: Compliant Task Pipeline and Reinforcement Learning of Its Controller with Compliance Modulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.25443v2](http://arxiv.org/pdf/2509.25443v2)**

> **作者:** Zewen He; Chenyuan Chen; Dilshod Azizov; Yoshihiko Nakamura
>
> **备注:** Submitted to IEEE for possible publication, under review
>
> **摘要:** Humanoid whole-body locomotion control is a critical approach for humanoid robots to leverage their inherent advantages. Learning-based control methods derived from retargeted human motion data provide an effective means of addressing this issue. However, because most current human datasets lack measured force data, and learning-based robot control is largely position-based, achieving appropriate compliance during interaction with real environments remains challenging. This paper presents Compliant Task Pipeline (CoTaP): a pipeline that leverages compliance information in the learning-based structure of humanoid robots. A two-stage dual-agent reinforcement learning framework combined with model-based compliance control for humanoid robots is proposed. In the training process, first a base policy with a position-based controller is trained; then in the distillation, the upper-body policy is combined with model-based compliance control, and the lower-body agent is guided by the base policy. In the upper-body control, adjustable task-space compliance can be specified and integrated with other controllers through compliance modulation on the symmetric positive definite (SPD) manifold, ensuring system stability. We validated the feasibility of the proposed strategy in simulation, primarily comparing the responses to external disturbances under different compliance settings.
>
---
#### [replaced 002] RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Interactive Environmental Learning in Physical Embodied Systems
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.01415v4](http://arxiv.org/pdf/2508.01415v4)**

> **作者:** Mingcong Lei; Honghao Cai; Zezhou Cui; Liangchen Tan; Junkun Hong; Gehan Hu; Shuangyu Zhu; Yimou Wu; Shaohan Jiang; Ge Wang; Yuyuan Yang; Junyuan Tan; Zhenglin Wan; Zhen Li; Shuguang Cui; Yiming Zhao; Yatong Han
>
> **摘要:** Embodied agents face persistent challenges in real-world environments, including partial observability, limited spatial reasoning, and high-latency multi-memory integration. We present RoboMemory, a brain-inspired framework that unifies Spatial, Temporal, Episodic, and Semantic memory under a parallelized architecture for efficient long-horizon planning and interactive environmental learning. A dynamic spatial knowledge graph (KG) ensures scalable and consistent memory updates, while a closed-loop planner with a critic module supports adaptive decision-making in dynamic settings. Experiments on EmbodiedBench show that RoboMemory, built on Qwen2.5-VL-72B-Ins, improves average success rates by 25% over its baseline and exceeds the closed-source state-of-the-art (SOTA) Gemini-1.5-Pro by 3%. Real-world trials further confirm its capacity for cumulative learning, with performance improving across repeated tasks. These results highlight RoboMemory as a scalable foundation for memory-augmented embodied intelligence, bridging the gap between cognitive neuroscience and robotic autonomy.
>
---
#### [replaced 003] Decremental Dynamics Planning for Robot Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.20521v2](http://arxiv.org/pdf/2503.20521v2)**

> **作者:** Yuanjie Lu; Tong Xu; Linji Wang; Nick Hawes; Xuesu Xiao
>
> **备注:** 7 pages. Accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Most, if not all, robot navigation systems employ a decomposed planning framework that includes global and local planning. To trade-off onboard computation and plan quality, current systems have to limit all robot dynamics considerations only within the local planner, while leveraging an extremely simplified robot representation (e.g., a point-mass holonomic model without dynamics) in the global level. However, such an artificial decomposition based on either full or zero consideration of robot dynamics can lead to gaps between the two levels, e.g., a global path based on a holonomic point-mass model may not be realizable by a non-holonomic robot, especially in highly constrained obstacle environments. Motivated by such a limitation, we propose a novel paradigm, Decremental Dynamics Planning that integrates dynamic constraints into the entire planning process, with a focus on high-fidelity dynamics modeling at the beginning and a gradual fidelity reduction as the planning progresses. To validate the effectiveness of this paradigm, we augment three different planners with DDP and show overall improved planning performance. We also develop a new DDP-based navigation system, which achieves first place in the simulation phase of the 2025 BARN Challenge. Both simulated and physical experiments validate DDP's hypothesized benefits.
>
---
#### [replaced 004] TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13579v3](http://arxiv.org/pdf/2509.13579v3)**

> **作者:** Momchil S. Tomov; Sang Uk Lee; Hansford Hendrago; Jinwook Huh; Teawon Han; Forbes Howington; Rafael da Silva; Gianmarco Bernasconi; Marc Heim; Samuel Findler; Xiaonan Ji; Alexander Boule; Michael Napoli; Kuo Chen; Jesse Miller; Boaz Floor; Yunqing Hu
>
> **摘要:** We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving.
>
---
#### [replaced 005] pRRTC: GPU-Parallel RRT-Connect for Fast, Consistent, and Low-Cost Motion Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.06757v2](http://arxiv.org/pdf/2503.06757v2)**

> **作者:** Chih H. Huang; Pranav Jadhav; Brian Plancher; Zachary Kingston
>
> **备注:** 7 pages, 7 figures, 1 table. Submitted to IEEE International Conference on Robotics and Automation 2026
>
> **摘要:** Sampling-based motion planning algorithms, like the Rapidly-Exploring Random Tree (RRT) and its widely used variant, RRT-Connect, provide efficient solutions for high-dimensional planning problems faced by real-world robots. However, these methods remain computationally intensive, particularly in complex environments that require many collision checks. To improve performance, recent efforts have explored parallelizing specific components of RRT such as collision checking, or running multiple planners independently. However, little has been done to develop an integrated parallelism approach, co-designed for large-scale parallelism. In this work we present pRRTC, a RRT-Connect based planner co-designed for GPU acceleration across the entire algorithm through parallel expansion and SIMT-optimized collision checking. We evaluate the effectiveness of pRRTC on the MotionBenchMaker dataset using robots with 7, 8, and 14 degrees of freedom (DoF). Compared to the state-of-the-art, pRRTC achieves as much as a 10x speedup on constrained reaching tasks with a 5.4x reduction in standard deviation. pRRTC also achieves a 1.4x reduction in average initial path cost. Finally, we deploy pRRTC on a 14-DoF dual Franka Panda arm setup and demonstrate real-time, collision-free motion planning with dynamic obstacles. We open-source our planner to support the wider community.
>
---
#### [replaced 006] Distilling On-device Language Models for Robot Planning with Minimal Human Intervention
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.17486v2](http://arxiv.org/pdf/2506.17486v2)**

> **作者:** Zachary Ravichandran; Ignacio Hounie; Fernando Cladera; Alejandro Ribeiro; George J. Pappas; Vijay Kumar
>
> **备注:** Accepted to the Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Large language models (LLMs) provide robots with powerful contextual reasoning abilities and a natural human interface. Yet, current LLM-enabled robots typically depend on cloud-hosted models, limiting their usability in environments with unreliable communication infrastructure, such as outdoor or industrial settings. We present PRISM, a framework for distilling small language model (SLM)-enabled robot planners that run on-device with minimal human supervision. Starting from an existing LLM-enabled planner, PRISM automatically synthesizes diverse tasks and environments, elicits plans from the LLM, and uses this synthetic dataset to distill a compact SLM as a drop-in replacement of the source model. We apply PRISM to three LLM-enabled planners for mapping and exploration, manipulation, and household assistance, and we demonstrate that PRISM improves the performance of Llama-3.2-3B from 10-20% of GPT-4o's performance to over 93% - using only synthetic data. We further demonstrate that the distilled planners generalize across heterogeneous robotic platforms (ground and aerial) and diverse environments (indoor and outdoor). We release all software, trained models, and datasets at https://zacravichandran.github.io/PRISM.
>
---
#### [replaced 007] Image-Based Visual Servoing for Enhanced Cooperation of Dual-Arm Manipulation
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2410.19432v4](http://arxiv.org/pdf/2410.19432v4)**

> **作者:** Zizhe Zhang; Yuan Yang; Wenqiang Zuo; Guangming Song; Aiguo Song; Yang Shi
>
> **备注:** 8 pages, 7 figures. Project website: https://zizhe.io/ral-ibvs-enhanced/. This work has been accepted to the IEEE Robotics and Automation Letters in Feb 2025
>
> **摘要:** The cooperation of a pair of robot manipulators is required to manipulate a target object without any fixtures. The conventional control methods coordinate the end-effector pose of each manipulator with that of the other using their kinematics and joint coordinate measurements. Yet, the manipulators' inaccurate kinematics and joint coordinate measurements can cause significant pose synchronization errors in practice. This paper thus proposes an image-based visual servoing approach for enhancing the cooperation of a dual-arm manipulation system. On top of the classical control, the visual servoing controller lets each manipulator use its carried camera to measure the image features of the other's marker and adapt its end-effector pose with the counterpart on the move. Because visual measurements are robust to kinematic errors, the proposed control can reduce the end-effector pose synchronization errors and the fluctuations of the interaction forces of the pair of manipulators on the move. Theoretical analyses have rigorously proven the stability of the closed-loop system. Comparative experiments on real robots have substantiated the effectiveness of the proposed control.
>
---
#### [replaced 008] mindmap: Spatial Memory in Deep Feature Maps for 3D Action Policies
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20297v3](http://arxiv.org/pdf/2509.20297v3)**

> **作者:** Remo Steiner; Alexander Millane; David Tingdahl; Clemens Volk; Vikram Ramasamy; Xinjie Yao; Peter Du; Soha Pouya; Shiwei Sheng
>
> **备注:** Accepted to CoRL 2025 Workshop RemembeRL
>
> **摘要:** End-to-end learning of robot control policies, structured as neural networks, has emerged as a promising approach to robotic manipulation. To complete many common tasks, relevant objects are required to pass in and out of a robot's field of view. In these settings, spatial memory - the ability to remember the spatial composition of the scene - is an important competency. However, building such mechanisms into robot learning systems remains an open research problem. We introduce mindmap (Spatial Memory in Deep Feature Maps for 3D Action Policies), a 3D diffusion policy that generates robot trajectories based on a semantic 3D reconstruction of the environment. We show in simulation experiments that our approach is effective at solving tasks where state-of-the-art approaches without memory mechanisms struggle. We release our reconstruction system, training code, and evaluation tasks to spur research in this direction.
>
---
#### [replaced 009] Interpreting Behaviors and Geometric Constraints as Knowledge Graphs for Robot Manipulation Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2310.03932v2](http://arxiv.org/pdf/2310.03932v2)**

> **作者:** Chen Jiang; Allie Wang; Martin Jagersand
>
> **摘要:** In this paper, we investigate the feasibility of using knowledge graphs to interpret actions and behaviors for robot manipulation control. Equipped with an uncalibrated visual servoing controller, we propose to use robot knowledge graphs to unify behavior trees and geometric constraints, conceptualizing robot manipulation control as semantic events. The robot knowledge graphs not only preserve the advantages of behavior trees in scripting actions and behaviors, but also offer additional benefits of mapping natural interactions between concepts and events, which enable knowledgeable explanations of the manipulation contexts. Through real-world evaluations, we demonstrate the flexibility of the robot knowledge graphs to support explainable robot manipulation control.
>
---
#### [replaced 010] FlowVLA: Visual Chain of Thought-based Motion Reasoning for Vision-Language-Action Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.18269v3](http://arxiv.org/pdf/2508.18269v3)**

> **作者:** Zhide Zhong; Haodong Yan; Junfeng Li; Xiangchen Liu; Xin Gong; Tianran Zhang; Wenxuan Song; Jiayi Chen; Xinhu Zheng; Hesheng Wang; Haoang Li
>
> **摘要:** Many Vision-Language-Action (VLA) models are built upon an internal world model trained via next-frame prediction ``$v_t \rightarrow v_{t+1}$''. However, this paradigm attempts to predict the future frame's appearance directly, without explicitly reasoning about the underlying dynamics. \textbf{This lack of an explicit motion reasoning step} often leads to physically implausible visual forecasts and inefficient policy learning. To address this limitation, we introduce the \textbf{Visual Chain of Thought (Visual CoT)}, a paradigm that compels the model to first reason about \textbf{motion dynamics} before generating the future frame. We instantiate this paradigm by proposing \textbf{FlowVLA}, an autoregressive Transformer that explicitly materializes this reasoning process as ``$v_t \rightarrow f_t \rightarrow v_{t+1}$'', where $f_t$ is an intermediate optical flow prediction that inherently encodes motion. By forcing the model to first follow the motion plan encoded by $f_t$, this process inherently \textbf{aligns the pre-training objective of dynamics prediction with the downstream task of action generation.} We conduct experiments on challenging robotics manipulation benchmarks, as well as real-robot evaluations. Our FlowVLA not only generates \textbf{more coherent and physically plausible visual predictions}, but also achieves state-of-the-art policy performance with \textbf{substantially improved sample efficiency}, pointing toward a more principled foundation for world modeling in VLAs. Project page: https://irpn-lab.github.io/FlowVLA/
>
---
#### [replaced 011] BC-ADMM: An Efficient Non-convex Constrained Optimizer with Robotic Applications
- **分类: math.OC; cs.NA; cs.RO; math.NA**

- **链接: [http://arxiv.org/pdf/2504.05465v2](http://arxiv.org/pdf/2504.05465v2)**

> **作者:** Zherong Pan; Kui Wu
>
> **摘要:** Non-convex constrained optimizations are ubiquitous in robotic applications such as multi-agent navigation, UAV trajectory optimization, and soft robot simulation. For this problem class, conventional optimizers suffer from small step sizes and slow convergence. We propose BC-ADMM, a variant of Alternating Direction Method of Multiplier (ADMM), that can solve a class of non-convex constrained optimizations with biconvex constraint relaxation. Our algorithm allows larger step sizes by breaking the problem into small-scale sub-problems that can be easily solved in parallel. We show that our method has both theoretical convergence speed guarantees and practical convergence guarantees in the asymptotic sense. Through numerical experiments in a row of four robotic applications, we show that BC-ADMM has faster convergence than conventional gradient descent and Newton's method in terms of wall clock time.
>
---
#### [replaced 012] IMPACT: Intelligent Motion Planning with Acceptable Contact Trajectories via Vision-Language Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10110v2](http://arxiv.org/pdf/2503.10110v2)**

> **作者:** Yiyang Ling; Karan Owalekar; Oluwatobiloba Adesanya; Erdem Bıyık; Daniel Seita
>
> **摘要:** Motion planning involves determining a sequence of robot configurations to reach a desired pose, subject to movement and safety constraints. Traditional motion planning finds collision-free paths, but this is overly restrictive in clutter, where it may not be possible for a robot to accomplish a task without contact. In addition, contacts range from relatively benign (e.g. brushing a soft pillow) to more dangerous (e.g. toppling a glass vase), making it difficult to characterize which may be acceptable. In this paper, we propose IMPACT, a novel motion planning framework that uses Vision-Language Models (VLMs) to infer environment semantics, identifying which parts of the environment can best tolerate contact based on object properties and locations. Our approach generates an anisotropic cost map that encodes directional push safety. We pair this map with a contact-aware A* planner to find stable contact-rich paths. We perform experiments using 20 simulation and 10 real-world scenes and assess using task success rate, object displacements, and feedback from human evaluators. Our results over 3200 simulation and 200 real-world trials suggest that IMPACT enables efficient contact-rich motion planning in cluttered settings while outperforming alternative methods and ablations. Our project website is available at https://impact-planning.github.io/.
>
---
#### [replaced 013] Equivariant Filter for Relative Attitude and Target's Angular Velocity Estimation
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2506.06016v2](http://arxiv.org/pdf/2506.06016v2)**

> **作者:** Gil Serrano; Bruno J. Guerreiro; Pedro Lourenço; Rita Cunha
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Accurate estimation of the relative attitude and angular velocity between two rigid bodies is fundamental in aerospace applications such as spacecraft rendezvous and docking. In these scenarios, a chaser vehicle must determine the orientation and angular velocity of a target object using onboard sensors. This work addresses the challenge of designing an Equivariant Filter (EqF) that can reliably estimate both the relative attitude and the target angular velocity using noisy observations of two known, non-collinear vectors fixed in the target frame. To derive the EqF, a symmetry for the system is proposed and an equivariant lift onto the symmetry group is calculated. Observability and convergence properties are analyzed. Simulations demonstrate the filter's performance, with Monte Carlo runs yielding statistically significant results. The impact of low-rate measurements is also examined and a strategy to mitigate this effect is proposed. Experimental results, using fiducial markers and both conventional and event cameras for measurement acquisition, further validate the approach, confirming its effectiveness in a realistic setting.
>
---
#### [replaced 014] Ego-to-Exo: Interfacing Third Person Visuals from Egocentric Views in Real-time for Improved ROV Teleoperation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.00848v4](http://arxiv.org/pdf/2407.00848v4)**

> **作者:** Adnan Abdullah; Ruo Chen; Ioannis Rekleitis; Md Jahidul Islam
>
> **备注:** EgoExo++ (Journal extension), V4, 12 pages
>
> **摘要:** Underwater ROVs (Remotely Operated Vehicles) are unmanned submersible vehicles designed for exploring and operating in the depths of the ocean. Despite using high-end cameras, typical teleoperation engines based on first-person (egocentric) views limit a surface operator's ability to maneuver the ROV in complex deep-water missions. In this paper, we present an interactive teleoperation interface that enhances the operational capabilities via increased situational awareness. This is accomplished by (i) offering on-demand "third"-person (exocentric) visuals from past egocentric views, and (ii) facilitating enhanced peripheral information with augmented ROV pose information in real-time. We achieve this by integrating a 3D geometry-based Ego-to-Exo view synthesis algorithm into a monocular SLAM system for accurate trajectory estimation. The proposed closed-form solution only uses past egocentric views from the ROV and a SLAM backbone for pose estimation, which makes it portable to existing ROV platforms. Unlike data-driven solutions, it is invariant to applications and waterbody-specific scenes. We validate the geometric accuracy of the proposed framework through extensive experiments of 2-DOF indoor navigation and 6-DOF underwater cave exploration in challenging low-light conditions. A subjective evaluation on 15 human teleoperators further confirms the effectiveness of the integrated features for improved teleoperation. We demonstrate the benefits of dynamic Ego-to-Exo view generation and real-time pose rendering for remote ROV teleoperation by following navigation guides such as cavelines inside underwater caves. This new way of interactive ROV teleoperation opens up promising opportunities for future research in subsea telerobotics.
>
---
#### [replaced 015] Capturing a Moving Target by Two Robots in the F2F Model
- **分类: cs.RO; cs.DC**

- **链接: [http://arxiv.org/pdf/2503.15688v2](http://arxiv.org/pdf/2503.15688v2)**

> **作者:** Khaled Jawhar; Evangelos Kranakis
>
> **摘要:** We study a search problem on capturing a moving target on an infinite real line. Two autonomous mobile robots (which can move with a maximum speed of 1) are initially placed at the origin, while an oblivious moving target is initially placed at a distance $d$ away from the origin. The robots can move along the line in any direction, but the target is oblivious, cannot change direction, and moves either away from or toward the origin at a constant speed $v$. Our aim is to design efficient algorithms for the two robots to capture the target. The target is captured only when both robots are co-located with it. The robots communicate with each other only face-to-face (F2F), meaning they can exchange information only when co-located, while the target remains oblivious and has no communication capabilities. We design algorithms under various knowledge scenarios, which take into account the prior knowledge the robots have about the starting distance $d$, the direction of movement (either toward or away from the origin), and the speed $v$ of the target. As a measure of the efficiency of the algorithms, we use the competitive ratio, which is the ratio of the capture time of an algorithm with limited knowledge to the capture time in the full-knowledge model. In our analysis, we are mindful of the cost of changing direction of movement, and show how to accomplish the capture of the target with at most three direction changes (turns).
>
---
#### [replaced 016] CottonSim: A vision-guided autonomous robotic system for cotton harvesting in Gazebo simulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.05317v2](http://arxiv.org/pdf/2505.05317v2)**

> **作者:** Thevathayarajh Thayananthan; Xin Zhang; Yanbo Huang; Jingdao Chen; Nuwan K. Wijewardane; Vitor S. Martins; Gary D. Chesser; Christopher T. Goodin
>
> **备注:** 16 pages, 15 figures, 4 tables
>
> **摘要:** Cotton is a major cash crop in the United States, with the country being a leading global producer and exporter. Nearly all U.S. cotton is grown in the Cotton Belt, spanning 17 states in the southern region. Harvesting remains a critical yet challenging stage, impacted by the use of costly, environmentally harmful defoliants and heavy, expensive cotton pickers. These factors contribute to yield loss, reduced fiber quality, and soil compaction, which collectively threaten long-term sustainability. To address these issues, this study proposes a lightweight, small-scale, vision-guided autonomous robotic cotton picker as an alternative. An autonomous system, built on Clearpath's Husky platform and integrated with the CottonEye perception system, was developed and tested in the Gazebo simulation environment. A virtual cotton field was designed to facilitate autonomous navigation testing. The navigation system used Global Positioning System (GPS) and map-based guidance, assisted by an RGBdepth camera and a YOLOv8nseg instance segmentation model. The model achieved a mean Average Precision (mAP) of 85.2%, a recall of 88.9%, and a precision of 93.0%. The GPS-based approach reached a 100% completion rate (CR) within a $(5e-6)^{\circ}$ threshold, while the map-based method achieved a 96.7% CR within a 0.25 m threshold. The developed Robot Operating System (ROS) packages enable robust simulation of autonomous cotton picking, offering a scalable baseline for future agricultural robotics. CottonSim code and datasets are publicly available on GitHub: https://github.com/imtheva/CottonSim
>
---
#### [replaced 017] Self-Supervised Representation Learning with Joint Embedding Predictive Architecture for Automotive LiDAR Object Detection
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.04969v2](http://arxiv.org/pdf/2501.04969v2)**

> **作者:** Haoran Zhu; Zhenyuan Dong; Kristi Topollai; Beiyao Sha; Anna Choromanska
>
> **摘要:** Recently, self-supervised representation learning relying on vast amounts of unlabeled data has been explored as a pre-training method for autonomous driving. However, directly applying popular contrastive or generative methods to this problem is insufficient and may even lead to negative transfer. In this paper, we present AD-L-JEPA, a novel self-supervised pre-training framework with a joint embedding predictive architecture (JEPA) for automotive LiDAR object detection. Unlike existing methods, AD-L-JEPA is neither generative nor contrastive. Instead of explicitly generating masked regions, our method predicts Bird's-Eye-View embeddings to capture the diverse nature of driving scenes. Furthermore, our approach eliminates the need to manually form contrastive pairs by employing explicit variance regularization to avoid representation collapse. Experimental results demonstrate consistent improvements on the LiDAR 3D object detection downstream task across the KITTI3D, Waymo, and ONCE datasets, while reducing GPU hours by 1.9x-2.7x and GPU memory by 2.8x-4x compared with the state-of-the-art method Occupancy-MAE. Notably, on the largest ONCE dataset, pre-training on 100K frames yields a 1.61 mAP gain, better than all other methods pre-trained on either 100K or 500K frames, and pre-training on 500K frames yields a 2.98 mAP gain, better than all other methods pre-trained on either 500K or 1M frames. AD-L-JEPA constitutes the first JEPA-based pre-training method for autonomous driving. It offers better quality, faster, and more GPU-memory-efficient self-supervised representation learning. The source code of AD-L-JEPA is ready to be released.
>
---
#### [replaced 018] Identifying Uncertainty in Self-Adaptive Robotics with Large Language Models
- **分类: cs.RO; cs.SE**

- **链接: [http://arxiv.org/pdf/2504.20684v2](http://arxiv.org/pdf/2504.20684v2)**

> **作者:** Hassan Sartaj; Jalil Boudjadar; Mirgita Frasheri; Shaukat Ali; Peter Gorm Larsen
>
> **摘要:** Future self-adaptive robots are expected to operate in highly dynamic environments while effectively managing uncertainties. However, identifying the sources and impacts of uncertainties in such robotic systems and defining appropriate mitigation strategies is challenging due to the inherent complexity of self-adaptive robots and the lack of comprehensive knowledge about the various factors influencing uncertainty. Hence, practitioners often rely on intuition and past experiences from similar systems to address uncertainties. In this article, we evaluate the potential of large language models (LLMs) in enabling a systematic and automated approach to identify uncertainties in self-adaptive robotics throughout the software engineering lifecycle. For this evaluation, we analyzed 10 advanced LLMs with varying capabilities across four industrial-sized robotics case studies, gathering the practitioners' perspectives on the LLM-generated responses related to uncertainties. Results showed that practitioners agreed with 63-88% of the LLM responses and expressed strong interest in the practicality of LLMs for this purpose.
>
---
#### [replaced 019] Toward Dynamic Control of Tendon-driven Continuum Robots using Clarke Transform
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.20693v2](http://arxiv.org/pdf/2503.20693v2)**

> **作者:** Christian Muhmann; Reinhard M. Grassmann; Max Bartholdt; Jessica Burgner-Kahrs
>
> **备注:** Accepted for publication at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025), 8 pages, and 8 figures
>
> **摘要:** In this paper, we propose a dynamic model and control framework for tendon-driven continuum robots (TDCRs) with multiple segments and an arbitrary number of tendons per segment. Our approach leverages the Clarke transform, the Euler-Lagrange formalism, and the piecewise constant curvature assumption to formulate a dynamic model on a two-dimensional manifold embedded in the joint space that inherently satisfies tendon constraints. We present linear and constraint-informed controllers that operate directly on this manifold, along with practical methods for preventing negative tendon forces without compromising control fidelity. This opens up new design possibilities for overactuated TDCRs with improved force distribution and stiffness without increasing controller complexity. We validate these approaches in simulation and on a physical prototype with one segment and five tendons, demonstrating accurate dynamic behavior and robust trajectory tracking under real-time conditions.
>
---
#### [replaced 020] Emergent interactions lead to collective frustration in robotic matter
- **分类: cond-mat.soft; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.22148v2](http://arxiv.org/pdf/2507.22148v2)**

> **作者:** Onurcan Bektas; Adolfo Alsina; Steffen Rulands
>
> **摘要:** Current artificial intelligence systems show near-human-level capabilities when deployed in isolation. Systems of a few collaborating intelligent agents are being engineered to perform tasks collectively. This raises the question of whether robotic matter, where many learning and intelligent agents interact, shows emergence of collective behaviour. And if so, which kind of phenomena would such systems exhibit? Here, we study a paradigmatic model for robotic matter: a stochastic many-particle system in which each particle is endowed with a deep neural network that predicts its transitions based on the particles' environments. For a one-dimensional model, we show that robotic matter exhibits complex emergent phenomena, including transitions between long-lived learning regimes, the emergence of particle species, and frustration. We also find a density-dependent phase transition with signatures of criticality. Using active matter theory, we show that this phase transition is a consequence of self-organisation mediated by emergent inter-particle interactions. Our simple model captures key features of more complex forms of robotic systems.
>
---
