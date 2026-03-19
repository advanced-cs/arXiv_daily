# 机器人 cs.RO

- **最新发布 48 篇**

- **更新 40 篇**

## 最新发布

#### [new 001] AgentVLN: Towards Agentic Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决长期导航中的空间感知与语义理解问题。提出AgentVLN框架，结合视觉语言模型与多级表示映射，提升导航准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.17670](https://arxiv.org/pdf/2603.17670)**

> **作者:** Zihao Xin; Wentong Li; Yixuan Jiang; Ziyuan Huang; Bin Wang; Piji Li; Jianke Zhu; Jie Qin; Shengjun Huang
>
> **备注:** 19pages, 4 figures
>
> **摘要:** Vision-and-Language Navigation (VLN) requires an embodied agent to ground complex natural-language instructions into long-horizon navigation in unseen environments. While Vision-Language Models (VLMs) offer strong 2D semantic understanding, current VLN systems remain constrained by limited spatial perception, 2D-3D representation mismatch, and monocular scale ambiguity. In this paper, we propose AgentVLN, a novel and efficient embodied navigation framework that can be deployed on edge computing platforms. We formulate VLN as a Partially Observable Semi-Markov Decision Process (POSMDP) and introduce a VLM-as-Brain paradigm that decouples high-level semantic reasoning from perception and planning via a plug-and-play skill library. To resolve multi-level representation inconsistency, we design a cross-space representation mapping that projects perception-layer 3D topological waypoints into the image plane, yielding pixel-aligned visual prompts for the VLM. Building on this bridge, we integrate a context-aware self-correction and active exploration strategy to recover from occlusions and suppress error accumulation over long trajectories. To further address the spatial ambiguity of instructions in unstructured environments, we propose a Query-Driven Perceptual Chain-of-Thought (QD-PCoT) scheme, enabling the agent with the metacognitive ability to actively seek geometric depth information. Finally, we construct AgentVLN-Instruct, a large-scale instruction-tuning dataset with dynamic stage routing conditioned on target visibility. Extensive experiments show that AgentVLN consistently outperforms prior state-of-the-art methods (SOTA) on long-horizon VLN benchmarks, offering a practical paradigm for lightweight deployment of next-generation embodied navigation models. Code: this https URL.
>
---
#### [new 002] Full Stack Navigation, Mapping, and Planning for the Lunar Autonomy Challenge
- **分类: cs.RO**

- **简介: 该论文属于月球自主导航与建图任务，解决GNSS缺失环境下的定位与地图构建问题。工作包括模块化系统设计、语义分割、视觉里程计、SLAM及分层规划控制，实现高精度定位与地图生成。**

- **链接: [https://arxiv.org/pdf/2603.17232](https://arxiv.org/pdf/2603.17232)**

> **作者:** Adam Dai; Asta Wu; Keidai Iiyama; Guillem Casadesus Vila; Kaila Coimbra; Thomas Deng; Grace Gao
>
> **备注:** Published in the Proceedings of the ION GNSS+ 2025 conference
>
> **摘要:** We present a modular, full-stack autonomy system for lunar surface navigation and mapping developed for the Lunar Autonomy Challenge. Operating in a GNSS-denied, visually challenging environment, our pipeline integrates semantic segmentation, stereo visual odometry, pose graph SLAM with loop closures, and layered planning and control. We leverage lightweight learning-based perception models for real-time segmentation and feature tracking and use a factor-graph backend to maintain globally consistent localization. High-level waypoint planning is designed to promote mapping coverage while encouraging frequent loop closures, and local motion planning uses arc sampling with geometric obstacle checks for efficient, reactive control. We evaluate our approach in the competition's high-fidelity lunar simulator, demonstrating centimeter-level localization accuracy, high-fidelity map generation, and strong repeatability across random seeds and rock distributions. Our solution achieved first place in the final competition evaluation.
>
---
#### [new 003] HeiSD: Hybrid Speculative Decoding for Embodied Vision-Language-Action Models with Kinematic Awareness
- **分类: cs.RO; cs.DB; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型推理速度慢的问题。通过提出HeiSD框架，结合两种SD方法，提升推理速度并保持任务成功率。**

- **链接: [https://arxiv.org/pdf/2603.17573](https://arxiv.org/pdf/2603.17573)**

> **作者:** Zihao Zheng; Zhihao Mao; Sicheng Tian; Maoliang Li; Jiayu Chen; Xinhao Sun; Zhaobo Zhang; Xuanzhe Liu; Donggang Cao; Hong Mei; Xiang Chen
>
> **摘要:** Vision-Language-Action (VLA) Models have become the mainstream solution for robot control, but suffer from slow inference speeds. Speculative Decoding (SD) is a promising acceleration method which can be divided into two categories: drafter-based SD and retrieval-based SD. Existing methods fail to analyze the advantages and disadvantages of these two types of SD in VLA models, leading to their sole application or optimization. In this paper, we analyze the trajectory patterns of robots controlled by the VLA model and derive a key insight: the two types of SD should be used in a hybrid manner. However, achieving hybrid SD in VLA models poses several challenges: (1) draft rejection and persistent errors in retrieval-based SD; (2) difficulty in determining the hybrid boundary. To address these, we propose the HeiSD framework. We propose a retrieval-based SD optimization method in HeiSD,which contains a verify-skip mechanism and a sequence-wise relaxed acceptance strategy. Moreover, we proposed a kinematic-based fused metric in HeiSD to automatically determine the hybrid boundary. Experimental results demonstrate that HeiSD attains a speedup of up to 2.45x in simulation benchmarks and 2.06x~2.41x in real-world scenarios, while sustaining a high task success rate.
>
---
#### [new 004] Huddle: Parallel Shape Assembly using Decentralized, Minimalistic Robots
- **分类: cs.RO**

- **简介: 该论文属于群体机器人任务，解决如何通过去中心化机器人形成任意形状的问题。通过局部交互实现无间隙、无死锁的组装。**

- **链接: [https://arxiv.org/pdf/2603.17768](https://arxiv.org/pdf/2603.17768)**

> **作者:** Khai Yi Chin; Tingwei Meng; Zhe Chen; Daniel Bassett; Yuri Ivanov
>
> **备注:** 16 pages, 6 figures, submitted to DARS 2026
>
> **摘要:** We propose a novel algorithm for forming arbitrarily shaped assemblies using decentralized robots. By relying on local interactions, the algorithm ensures there are no unreachable states or gaps in the assembly, which are global properties. The in-assembly robots attract passing-by robots into expanding the assembly via a simple implementation of signaling and alignment. Our approach is minimalistic, requiring only communication between attached, immediate neighbors. It is motion-agnostic and requires no pose localization, enabling asynchronous and order-independent assembly. We prove the algorithm's correctness and demonstrate its effectiveness in forming a 107-robot assembly.
>
---
#### [new 005] SLAM Adversarial Lab: An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SAL框架，用于评估视觉SLAM在恶劣条件下的鲁棒性。解决SLAM系统在复杂环境下的稳定性问题，通过生成对抗数据集并测试不同算法表现。**

- **链接: [https://arxiv.org/pdf/2603.17165](https://arxiv.org/pdf/2603.17165)**

> **作者:** Mohamed Hefny; Karthik Dantu; Steven Y. Ko
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** We present SAL (SLAM Adversarial Lab), a modular framework for evaluating visual SLAM systems under adversarial conditions such as fog and rain. SAL represents each adversarial condition as a perturbation that transforms an existing dataset into an adversarial dataset. When transforming a dataset, SAL supports severity levels using easily-interpretable real-world units such as meters for fog visibility. SAL's extensible architecture decouples datasets, perturbations, and SLAM algorithms through common interfaces, so users can add new components without rewriting integration code. Moreover, SAL includes a search procedure that finds the severity level of a perturbation at which a SLAM system fails. To showcase the capabilities of SAL, our evaluation integrates seven SLAM algorithms and evaluates them across three datasets under weather, camera, and video transport perturbations.
>
---
#### [new 006] Interpreting Context-Aware Human Preferences for Multi-Objective Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于多目标机器人导航任务，旨在解决如何让机器人理解并适应人类上下文相关的偏好。通过结合大模型与强化学习，实现行为规则的提取与实时导航调整。**

- **链接: [https://arxiv.org/pdf/2603.17510](https://arxiv.org/pdf/2603.17510)**

> **作者:** Tharun Sethuraman; Subham Agrawal; Nils Dengler; Jorge de Heuvel; Teena Hassan; Maren Bennewitz
>
> **摘要:** Robots operating in human-shared environments must not only achieve task-level navigation objectives such as safety and efficiency, but also adapt their behavior to human preferences. However, as human preferences are typically expressed in natural language and depend on environmental context, it is difficult to directly integrate them into low-level robot control policies. In this work, we present a pipeline that enables robots to understand and apply context-dependent navigation preferences by combining foundational models with a Multi-Objective Reinforcement Learning (MORL) navigation policy. Thus, our approach integrates high-level semantic reasoning with low-level motion control. A Vision-Language Model (VLM) extracts structured environmental context from onboard visual observations, while Large Language Models (LLM) convert natural language user feedback into interpretable, context-dependent behavioral rules stored in a persistent but updatable rule memory. A preference translation module then maps contextual information and stored rules into numerical preference vectors that parameterize a pretrained MORL policy for real-time navigation adaptation. We evaluate the proposed framework through quantitative component-level evaluations, a user study, and real-world robot deployments in various indoor environments. Our results demonstrate that the system reliably captures user intent, generates consistent preference vectors, and enables controllable behavior adaptation across diverse contexts. Overall, the proposed pipeline improves the adaptability, transparency, and usability of robots operating in shared human environments, while maintaining safe and responsive real-time control.
>
---
#### [new 007] DexEXO: A Wearability-First Dexterous Exoskeleton for Operator-Agnostic Demonstration and Learning
- **分类: cs.RO**

- **简介: 该论文提出DexEXO，一种注重穿戴性的灵巧外骨骼，解决跨用户数据收集与姿态对齐问题，通过硬件级匹配提升学习效率。**

- **链接: [https://arxiv.org/pdf/2603.17323](https://arxiv.org/pdf/2603.17323)**

> **作者:** Alvin Zhu; Mingzhang Zhu; Beom Jun Kim; Jose Victor S. H. Ramos; Yike Shi; Yufeng Wu; Raayan Dhar; Fuyi Yang; Ruochen Hou; Hanzhang Fang; Quanyou Wang; Yuchen Cui; Dennis W. Hong
>
> **备注:** this https URL
>
> **摘要:** Scaling dexterous robot learning is constrained by the difficulty of collecting high-quality demonstrations across diverse operators. Existing wearable interfaces often trade comfort and cross-user adaptability for kinematic fidelity, while embodiment mismatch between demonstration and deployment requires visual post-processing before policy training. We present DexEXO, a wearability-first hand exoskeleton that aligns visual appearance, contact geometry, and kinematics at the hardware level. DexEXO features a pose-tolerant thumb mechanism and a slider-based finger interface analytically modeled to support hand lengths from 140~mm to 217~mm, reducing operator-specific fitting and enabling scalable cross-operator data collection. A passive hand visually matches the deployed robot, allowing direct policy training from raw wrist-mounted RGB observations. User studies demonstrate improved comfort and usability compared to prior wearable systems. Using visually aligned observations alone, we train diffusion policies that achieve competitive performance while substantially simplifying the end-to-end pipeline. These results show that prioritizing wearability and hardware-level embodiment alignment reduces both human and algorithmic bottlenecks without sacrificing task performance. Project Page: this https URL
>
---
#### [new 008] RoboForge: Physically Optimized Text-guided Whole-Body Locomotion for Humanoids
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，解决文本引导的全身运动生成与物理可行性问题。提出一种无需重定向的物理优化框架，实现生成与控制的双向耦合，提升运动精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.17927](https://arxiv.org/pdf/2603.17927)**

> **作者:** Xichen Yuan; Zhe Li; Bofan Lyu; Kuangji Zuo; Yanshuo Lu; Gen Li; Jianfei Yang
>
> **备注:** 10 pages, 5 figures,submitted to IROS 2026
>
> **摘要:** While generative models have become effective at producing human-like motions from text, transferring these motions to humanoid robots for physical execution remains challenging. Existing pipelines are often limited by retargeting, where kinematic quality is undermined by physical infeasibility, contact-transition errors, and the high cost of real-world dynamical data. We present a unified latent-driven framework that bridges natural language and whole-body humanoid locomotion through a retarget-free, physics-optimized pipeline. Rather than treating generation and control as separate stages, our key insight is to couple them bidirectionally under physical this http URL introduce a Physical Plausibility Optimization (PP-Opt) module as the coupling interface. In the forward direction, PP-Opt refines a teacher-student distillation policy with a plausibility-centric reward to suppress artifacts such as floating, skating, and penetration. In the backward direction, it converts reward-optimized simulation rollouts into high-quality explicit motion data, which is used to fine-tune the motion generator toward a more physically plausible latent distribution. This bidirectional design forms a self-improving cycle: the generator learns a physically grounded latent space, while the controller learns to execute latent-conditioned behaviors with dynamical this http URL experiments on the Unitree G1 humanoid show that our bidirectional optimization improves tracking accuracy and success rates. Across IsaacLab and MuJoCo, the implicit latent-driven pipeline consistently outperforms conventional explicit retargeting baselines in both precision and stability. By coupling diffusion-based motion generation with physical plausibility optimization, our framework provides a practical path toward deployable text-guided humanoid intelligence.
>
---
#### [new 009] Neural Radiance Maps for Extraterrestrial Navigation and Path Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主导航任务，旨在解决外星探测器路径规划问题。通过使用NeRF构建全局地图，结合局部信息进行高效路径规划。**

- **链接: [https://arxiv.org/pdf/2603.17236](https://arxiv.org/pdf/2603.17236)**

> **作者:** Adam Dai; Shubh Gupta; Grace Gao
>
> **备注:** Published in the Proceedings of the ION GNSS+ 2023 Conference
>
> **摘要:** Autonomous vehicles such as the Mars rovers currently lead the vanguard of surface exploration on extraterrestrial planets and moons. In order to accelerate the pace of exploration and science objectives, it is critical to plan safe and efficient paths for these vehicles. However, current rover autonomy is limited by a lack of global maps which can be easily constructed and stored for onboard re-planning. Recently, Neural Radiance Fields (NeRFs) have been introduced as a detailed 3D scene representation which can be trained from sparse 2D images and efficiently stored. We propose to use NeRFs to construct maps for online use in autonomous navigation, and present a planning framework which leverages the NeRF map to integrate local and global information. Our approach interpolates local cost observations across global regions using kernel ridge regression over terrain features extracted from the NeRF map, allowing the rover to re-route itself around untraversable areas discovered during online operation. We validate our approach in high-fidelity simulation and demonstrate lower cost and higher percentage success rate path planning compared to various baselines.
>
---
#### [new 010] Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于边缘计算任务，研究如何在受限条件下部署基础模型。解决系统级约束问题，提出协同设计策略以提升可靠性。**

- **链接: [https://arxiv.org/pdf/2603.16952](https://arxiv.org/pdf/2603.16952)**

> **作者:** Utkarsh Grover; Ravi Ranjan; Mingyang Mao; Trung Tien Dong; Satvik Praveen; Zhenqi Wu; J. Morris Chang; Tinoosh Mohsenin; Yi Sheng; Agoritsa Polyzou; Eiman Kanjo; Xiaomin Lin
>
> **摘要:** Deploying foundation models in embodied edge systems is fundamentally a systems problem, not just a problem of model compression. Real-time control must operate within strict size, weight, and power constraints, where memory traffic, compute latency, timing variability, and safety margins interact directly. The Deployment Gauntlet organizes these constraints into eight coupled barriers that determine whether embodied foundation models can run reliably in practice. Across representative edge workloads, autoregressive Vision-Language-Action policies are constrained primarily by memory bandwidth, whereas diffusion-based controllers are limited more by compute latency and sustained execution cost. Reliable deployment therefore depends on system-level co-design across memory, scheduling, communication, and model architecture, including decompositions that separate fast control from slower semantic reasoning.
>
---
#### [new 011] ProbeFlow: Training-Free Adaptive Flow Matching for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出ProbeFlow，解决VLA模型中流匹配动作头的推理延迟问题，通过动态调度减少计算步骤，提升机器人控制响应速度。**

- **链接: [https://arxiv.org/pdf/2603.17850](https://arxiv.org/pdf/2603.17850)**

> **作者:** Zhou Fang; Jiaqi Wang; Yi Zhou; Qiongfeng Shi
>
> **摘要:** Recent Vision-Language-Action (VLA) models equipped with Flow Matching (FM) action heads achieve state-of-the-art performance in complex robot manipulation. However, the multi-step iterative ODE solving required by FM introduces inference latency that precludes responsive physical control. While current acceleration efforts optimize the Vision-Language Model (VLM) backbone, the action head bottleneck remains overlooked. To address this, we propose ProbeFlow, a training-free adaptive inference framework tai- lored for continuous robotic control. By evaluating geometric trajectory complexity via the cosine similarity between initial and lookahead velocity vectors, ProbeFlow dynamically sched- ules integration steps to prune redundant network evaluations. On the MetaWorld benchmark, it accelerates action decoding by 14.8x (reducing average steps from N = 50 to 2.6) and cuts end-to-end system latency by 2.8x without compromising the manipulation success rate. On the long-horizon LIBERO benchmark, the probe automatically allocates a denser schedule to navigate semantic bottlenecks, effectively resolving the flow solver delay. Real-world physical deployments confirm that ProbeFlow successfully mitigates action decoding latency while ensuring execution stability, offering a highly practical solution for low-latency continuous generative policies.
>
---
#### [new 012] Influence of Gripper Design on Human Demonstration Quality for Robot Learning
- **分类: cs.RO**

- **简介: 论文研究手持夹具设计对机器人学习中人类演示质量的影响，旨在提升医疗机器人操作技能获取效率。通过对比不同夹具与徒手操作的表现，评估其在医疗包装开启任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.17189](https://arxiv.org/pdf/2603.17189)**

> **作者:** Gina L. Georgadarellis; Natalija Beslic; Seonhun Lee; Frank C. Sup IV; Meghan E. Huber
>
> **备注:** To be published in proceedings of 2026 IEEE International Conference on Robotics & Automation
>
> **摘要:** Opening sterile medical packaging is routine for healthcare workers but remains challenging for robots. Learning from demonstration enables robots to acquire manipulation skills directly from humans, and handheld gripper tools such as the Universal Manipulation Interface (UMI) offer a pathway for efficient data collection. However, the effectiveness of these tools depends heavily on their usability. We evaluated UMI in demonstrating a bandage opening task, a common manipulation task in hospital settings, by testing three conditions: distributed load grippers, concentrated load grippers, and bare hands. Eight participants performed timed trials, with task performance assessed by success rate, completion time, and damage, alongside perceived workload using the NASA-TLX questionnaire. Concentrated load grippers improved performance relative to distributed load grippers but remained substantially slower and less effective than hands. These results underscore the importance of ergonomic and mechanical refinements in handheld grippers to reduce user burden and improve demonstration quality, especially for applications in healthcare robotics.
>
---
#### [new 013] Generative Control as Optimization: Time Unconditional Flow Matching for Adaptive and Robust Robotic Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决传统扩散模型和流匹配在推理时计算资源分配不合理的问题。提出GeCO框架，通过优化实现自适应计算和安全检测。**

- **链接: [https://arxiv.org/pdf/2603.17834](https://arxiv.org/pdf/2603.17834)**

> **作者:** Zunzhe Zhang; Runhan Huang; Yicheng Liu; Shaoting Zhu; Linzhan Mou; Hang Zhao
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Diffusion models and flow matching have become a cornerstone of robotic imitation learning, yet they suffer from a structural inefficiency where inference is often bound to a fixed integration schedule that is agnostic to state complexity. This paradigm forces the policy to expend the same computational budget on trivial motions as it does on complex tasks. We introduce Generative Control as Optimization (GeCO), a time-unconditional framework that transforms action synthesis from trajectory integration into iterative optimization. GeCO learns a stationary velocity field in the action-sequence space where expert behaviors form stable attractors. Consequently, test-time inference becomes an adaptive process that allocates computation based on convergence--exiting early for simple states while refining longer for difficult ones. Furthermore, this stationary geometry yields an intrinsic, training-free safety signal, as the field norm at the optimized action serves as a robust out-of-distribution (OOD) detector, remaining low for in-distribution states while significantly increasing for anomalies. We validate GeCO on standard simulation benchmarks and demonstrate seamless scaling to pi0-series Vision-Language-Action (VLA) models. As a plug-and-play replacement for standard flow-matching heads, GeCO improves success rates and efficiency with an optimization-native mechanism for safe deployment. Video and code can be found at this https URL
>
---
#### [new 014] Multi-Source Human-in-the-Loop Digital Twin Testbed for Connected and Autonomous Vehicles in Mixed Traffic Flow
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出MSH-MCCT测试平台，用于研究混合交通中自动驾驶车辆与人工驾驶车辆的交互。解决CAV测试中的真实人类参与问题，通过多源控制和数字孪生技术提升实验灵活性与扩展性。**

- **链接: [https://arxiv.org/pdf/2603.17751](https://arxiv.org/pdf/2603.17751)**

> **作者:** Jianghong Dong; Jiawei Wang; Chunying Yang; Mengchi Cai; Chaoyi Chen; Qing Xu; Jianqiang Wang; Keqiang Li
>
> **摘要:** In the emerging mixed traffic environments, Connected and Autonomous Vehicles (CAVs) have to interact with surrounding human-driven vehicles (HDVs). This paper introduces MSH-MCCT (Multi-Source Human-in-the-Loop Mixed Cloud Control Testbed), a novel CAV testbed that captures complex interactions between various CAVs and HDVs. Utilizing the Mixed Digital Twin concept, which combines Mixed Reality with Digital Twin, MSH-MCCT integrates physical, virtual, and mixed platforms, along with multi-source control inputs. Bridged by the mixed platform, MSH-MCCT allows human drivers and CAV algorithms to operate both physical and virtual vehicles within multiple fields of view. Particularly, this testbed facilitates the coexistence and real-time interaction of physical and virtual CAVs \& HDVs, significantly enhancing the experimental flexibility and scalability. Experiments on vehicle platooning in mixed traffic showcase the potential of MSH-MCCT to conduct CAV testing with multi-source real human drivers in the loop through driving simulators of diverse fidelity. The videos for the experiments are available at our project website: this https URL.
>
---
#### [new 015] TeleDex: Accessible Dexterous Teleoperation
- **分类: cs.RO**

- **简介: 该论文提出TeleDex系统，用于灵活机械手的直观远程操作。解决真实环境中机器人操作政策泛化能力不足的问题，通过手机实现低延迟、低成本的演示收集。**

- **链接: [https://arxiv.org/pdf/2603.17065](https://arxiv.org/pdf/2603.17065)**

> **作者:** Omar Rayyan; Maximilian Gillesm; Yuchen Cui
>
> **备注:** For project website and videos, see this https URL
>
> **摘要:** Despite increasing dataset scale and model capacity, robot manipulation policies still struggle to generalize beyond their training distributions. As a result, deploying state-of-the-art policies in new environments, tasks, or robot embodiments often requires collecting additional demonstrations. Enabling this in real-world deployment settings requires tools that allow users to collect demonstrations quickly, affordably, and with minimal setup. We present TeleDex, an open-source system for intuitive teleoperation of dexterous hands and robotic manipulators using any readily available phone. The system streams low-latency 6-DoF wrist poses and articulated 21-DoF hand state estimates from the phone, which are retargeted to robot arms and multi-fingered hands without requiring external tracking infrastructure. TeleDex supports both a handheld phone-only mode and an optional 3D-printable hand-mounted interface for finger-level teleoperation. By lowering the hardware and setup barriers to dexterous teleoperation, TeleDex enables users to quickly collect demonstrations during deployment to support policy fine-tuning. We evaluate the system across simulation and real-world manipulation tasks, demonstrating its effectiveness as a unified scalable interface for robot teleoperation. All software and hardware designs, along with demonstration videos, are open-source and available at this http URL.
>
---
#### [new 016] Specification-Aware Distribution Shaping for Robotics Foundation Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决机器人执行任务时的安全性和时间约束问题。提出一种无需修改模型参数的分布优化框架，以满足STL约束。**

- **链接: [https://arxiv.org/pdf/2603.17969](https://arxiv.org/pdf/2603.17969)**

> **作者:** Sadık Bera Yüksel; Derya Aksaray
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Robotics foundation models have demonstrated strong capabilities in executing natural language instructions across diverse tasks and environments. However, they remain largely data-driven and lack formal guarantees on safety and satisfaction of time-dependent specifications during deployment. In practice, robots often need to comply with operational constraints involving rich spatio-temporal requirements such as time-bounded goal visits, sequential objectives, and persistent safety conditions. In this work, we propose a specification-aware action distribution optimization framework that enforces a broad class of Signal Temporal Logic (STL) constraints during execution of a pretrained robotics foundation model without modifying its parameters. At each decision step, the method computes a minimally modified action distribution that satisfies a hard STL feasibility constraint by reasoning over the remaining horizon using forward dynamics propagation. We validate the proposed framework in simulation using a state-of-the-art robotics foundation model across multiple environments and complex specifications.
>
---
#### [new 017] VectorWorld: Efficient Streaming World Model via Diffusion Flow on Vector Graphs
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VectorWorld，用于自动驾驶的闭环仿真。解决历史初始化不匹配、采样延迟和长期不一致问题，通过增量生成矢量图块和物理对齐策略，实现高效实时模拟。**

- **链接: [https://arxiv.org/pdf/2603.17652](https://arxiv.org/pdf/2603.17652)**

> **作者:** Chaokang Jiang; Desen Zhou; Jiuming Liu; Kevin Li Sun
>
> **备注:** Under Review
>
> **摘要:** Closed-loop evaluation of autonomous-driving policies requires interactive simulation beyond log replay. However, existing generative world models often degrade in closed loop due to (i) history-free initialization that mismatches policy inputs, (ii) multi-step sampling latency that violates real-time budgets, and (iii) compounding kinematic infeasibility over long horizons. We propose VectorWorld, a streaming world model that incrementally generates ego-centric $64 \mathrm{m}\times 64\mathrm{m}$ lane--agent vector-graph tiles during rollout. VectorWorld aligns initialization with history-conditioned policies by producing a policy-compatible interaction state via a motion-aware gated VAE. It enables real-time outpainting via solver-free one-step masked completion with an edge-gated relational DiT trained with interval-conditioned MeanFlow and JVP-based large-step supervision. To stabilize long-horizon rollouts, we introduce $\Delta$Sim, a physics-aligned non-ego (NPC) policy with hybrid discrete--continuous actions and differentiable kinematic logit shaping. On Waymo open motion and nuPlan, VectorWorld improves map-structure fidelity and initialization validity, and supports stable, real-time $1\mathrm{km}+$ closed-loop rollouts (\href{this https URL}{code}).
>
---
#### [new 018] VolumeDP: Modeling Volumetric Representation for Manipulation Policy Learning
- **分类: cs.RO**

- **简介: 该论文提出VolumeDP，解决机器人操作中2D-3D视觉与动作的不匹配问题，通过3D体素表示提升空间推理与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.17720](https://arxiv.org/pdf/2603.17720)**

> **作者:** Tianxing Zhou; Feiyang Xue; Zhangchen Ye; Tianyuan Yuan; Hang Zhao; Tao Jiang
>
> **摘要:** Imitation learning is a prominent paradigm for robotic manipulation. However, existing visual imitation methods map 2D image observations directly to 3D action outputs, imposing a 2D-3D mismatch that hinders spatial reasoning and degrades robustness. We present VolumeDP, a policy architecture that restores spatial alignment by explicitly reasoning in 3D. VolumeDP first lifts image features into a Volumetric Representation via cross-attention. It then selects task-relevant voxels with a learnable module and converts them into a compact set of spatial tokens, markedly reducing computation while preserving action-critical geometry. Finally, a multi-token decoder conditions on the entire token set to predict actions, thereby avoiding lossy aggregation that collapses multiple spatial tokens into a single descriptor. VolumeDP achieves a state-of-the-art average success rate of 88.8% on the LIBERO simulation benchmark, outperforming the strongest baseline by a substantial 14.8% improvement. It also delivers large performance gains over prior methods on the ManiSkill and LIBERO-Plus benchmarks. Real-world experiments further demonstrate higher success rates and robust generalization to novel spatial layouts, camera viewpoints, and environment backgrounds. Code will be released.
>
---
#### [new 019] FloorPlan-VLN: A New Paradigm for Floor Plan Guided Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决传统方法依赖冗长指令、忽视空间信息的问题。通过引入结构化平面图作为全局先验，提出FP-Nav方法，提升导航效果。**

- **链接: [https://arxiv.org/pdf/2603.17437](https://arxiv.org/pdf/2603.17437)**

> **作者:** Kehan Chen; Yan Huang; Dong An; Jiawei He; Yifei Su; Jing Liu; Nianfeng Liu; Liang Wang
>
> **摘要:** Existing Vision-Language Navigation (VLN) task requires agents to follow verbose instructions, ignoring some potentially useful global spatial priors, limiting their capability to reason about spatial structures. Although human-readable spatial schematics (e.g., floor plans) are ubiquitous in real-world buildings, current agents lack the cognitive ability to comprehend and utilize them. To bridge this gap, we introduce \textbf{FloorPlan-VLN}, a new paradigm that leverages structured semantic floor plans as global spatial priors to enable navigation with only concise instructions. We first construct the FloorPlan-VLN dataset, which comprises over 10k episodes across 72 scenes. It pairs more than 100 semantically annotated floor plans with Matterport3D-based navigation trajectories and concise instructions that omit step-by-step guidance. Then, we propose a simple yet effective method \textbf{FP-Nav} that uses a dual-view, spatio-temporally aligned video sequence, and auxiliary reasoning tasks to align observations, floor plans, and instructions. When evaluated under this new benchmark, our method significantly outperforms adapted state-of-the-art VLN baselines, achieving more than a 60\% relative improvement in navigation success rate. Furthermore, comprehensive noise modeling and real-world deployments demonstrate the feasibility and robustness of FP-Nav to actuation drift and floor plan distortions. These results validate the effectiveness of floor plan guided navigation and highlight FloorPlan-VLN as a promising step toward more spatially intelligent navigation.
>
---
#### [new 020] Rewarding DINO: Predicting Dense Rewards with Vision Foundation Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人任务中的奖励函数预测问题，旨在解决真实世界中稀疏奖励的难题。通过语言条件下的奖励建模，学习通用奖励函数，提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2603.16978](https://arxiv.org/pdf/2603.16978)**

> **作者:** Pierre Krack; Tobias Jülg; Wolfram Burgard; Florian Walter
>
> **备注:** 10 pages, 5 figures, submitted to IEEE
>
> **摘要:** Well-designed dense reward functions in robot manipulation not only indicate whether a task is completed but also encode progress along the way. Generally, designing dense rewards is challenging and usually requires access to privileged state information available only in simulation, not in real-world experiments. This makes reward prediction models that infer task state information from camera images attractive. A common approach is to predict rewards from expert demonstrations based on visual similarity or sequential frame ordering. However, this biases the resulting reward function towards a specific solution and leaves it undefined in states not covered by the demonstrations. In this work, we introduce Rewarding DINO, a method for language-conditioned reward modeling that learns actual reward functions rather than specific trajectories. The model's compact size allows it to serve as a direct replacement for analytical reward functions with comparatively low computational overhead. We train our model on data sampled from 24 Meta-World+ tasks using a rank-based loss and evaluate pairwise accuracy, rank correlation, and calibration. Rewarding DINO achieves competitive performance in tasks from the training set and generalizes to new settings in simulation and the real world, indicating that it learns task semantics. We also test the model with off-the-shelf reinforcement learning algorithms to solve tasks from our Meta-World+ training set.
>
---
#### [new 021] EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉控制任务，旨在解决视频生成与实际机器人动作不匹配的执行性差距问题。通过引入EVA框架，利用逆动力学模型作为奖励机制，提升生成视频的可执行性。**

- **链接: [https://arxiv.org/pdf/2603.17808](https://arxiv.org/pdf/2603.17808)**

> **作者:** Ruixiang Wang; Qingming Liu; Yueci Deng; Guiliang Liu; Zhen Liu; Kui Jia
>
> **备注:** Project page: this https URL
>
> **摘要:** Video generative models are increasingly used as world models for robotics, where a model generates a future visual rollout conditioned on the current observation and task instruction, and an inverse dynamics model (IDM) converts the generated frames into executable robot actions. However, current video world models lack explicit executability constraints. As a result, visually coherent rollouts may still violate rigid-body and kinematic consistency, producing unstable or infeasible control commands when decoded by an IDM. We refer to this mismatch between visual generation and physically executable control as the executability gap. While this gap can be mitigated at inference time using techniques such as rejection sampling, such approaches are inefficient due to the high cost of video generation. In this paper, we leverage the executability gap as a training signal and introduce Executable Video Alignment (EVA), a reinforcement-learning post-training framework for aligning video world models. EVA trains an inverse dynamics model on real robot trajectories and repurposes it as a reward model that evaluates generated videos through the action sequences they induce, encouraging smooth motions measured by velocity, acceleration, and jerk while penalizing actions that violate embodiment constraints. Importantly, the reward remains informative even when generated videos contain severe visual artifacts, since such artifacts typically translate into unstable or out-of-bound actions. Experiments on the RoboTwin benchmark and a real bimanual robot show that EVA reduces embodiment-specific artifacts in generated rollouts and improves downstream task execution success.
>
---
#### [new 022] Contingency-Aware Planning via Certified Neural Hamilton-Jacobi Reachability
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人导航任务，解决高维HJ方程计算限制问题，通过结合学习与采样方法实现安全多目标规划。**

- **链接: [https://arxiv.org/pdf/2603.17022](https://arxiv.org/pdf/2603.17022)**

> **作者:** Kasidit Muenprasitivej; Derya Aksaray
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Hamilton-Jacobi (HJ) reachability provides formal safety guarantees for dynamical systems, but solving high-dimensional HJ partial differential equations limits its use in real-time planning. This paper presents a contingency-aware multi-goal navigation framework that integrates learning-based reachability with sampling-based planning in unknown environments. We use Fourier Neural Operator (FNO) to approximate the solution operator of the Hamilton-Jacobi-Isaacs variational inequality under varying obstacle configurations. We first provide a theoretical under-approximation guarantee on the safe backward reach-avoid set, which enables formal safety certification of the learned reachable sets. Then, we integrate the certified reachable sets with an incremental multi-goal planner, which enforces reachable-set constraints and a recovery policy that guarantees finite-time return to a safe region. Overall, we demonstrate that the proposed framework achieves asymptotically optimal navigation with provable contingency behavior, and validate its performance through real-time deployment on KUKA's youBot in Webots simulation.
>
---
#### [new 023] OmniVLN: Omnidirectional 3D Perception and Token-Efficient LLM Reasoning for Visual-Language Navigation across Air and Ground Platforms
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决多场景下导航精度低和上下文过载问题。提出OmniVLN框架，结合3D感知与高效推理，提升导航成功率与空间理解能力。**

- **链接: [https://arxiv.org/pdf/2603.17351](https://arxiv.org/pdf/2603.17351)**

> **作者:** Zhongyuang Liu; Min He; Shaonan Yu; Xinhang Xu; Muqing Cao; Jianping Li; Jianfei Yang; Lihua Xie
>
> **摘要:** Language-guided embodied navigation requires an agent to interpret object-referential instructions, search across multiple rooms, localize the referenced target, and execute reliable motion toward it. Existing systems remain limited in real indoor environments because narrow field-of-view sensing exposes only a partial local scene at each step, often forcing repeated rotations, delaying target discovery, and producing fragmented spatial understanding; meanwhile, directly prompting LLMs with dense 3D maps or exhaustive object lists quickly exceeds the context budget. We present OmniVLN, a zero-shot visual-language navigation framework that couples omnidirectional 3D perception with token-efficient hierarchical reasoning for both aerial and ground robots. OmniVLN fuses a rotating LiDAR and panoramic vision into a hardware-agnostic mapping stack, incrementally constructs a five-layer Dynamic Scene Graph (DSG) from mesh geometry to room- and building-level structure, and stabilizes high-level topology through persistent-homology-based room partitioning and hybrid geometric/VLM relation verification. For navigation, the global DSG is transformed into an agent-centric 3D octant representation with multi-resolution spatial attention prompting, enabling the LLM to progressively filter candidate rooms, infer egocentric orientation, localize target objects, and emit executable navigation primitives while preserving fine local detail and compact long-range memory. Experiments show that the proposed hierarchical interface improves spatial referring accuracy from 77.27\% to 93.18\%, reduces cumulative prompt tokens by up to 61.7\% in cluttered multi-room settings, and improves navigation success by up to 11.68\% over a flat-list baseline. We will release the code and an omnidirectional multimodal dataset to support reproducible research.
>
---
#### [new 024] DexViTac: Collecting Human Visuo-Tactile-Kinematic Demonstrations for Contact-Rich Dexterous Manipulation
- **分类: cs.RO**

- **简介: 论文提出DexViTac系统，用于收集接触丰富的精细操作的多模态数据。解决人类操作中触觉信息难捕捉的问题，通过高精度视觉、触觉和运动学数据，提升机器人学习效果。**

- **链接: [https://arxiv.org/pdf/2603.17851](https://arxiv.org/pdf/2603.17851)**

> **作者:** Xitong Chen; Yifeng Pan; Min Li; Xiaotian Ding
>
> **备注:** 9 pages, 9 this http URL page: this https URL
>
> **摘要:** Large-scale, high-quality multimodal demonstrations are essential for robot learning of contact-rich dexterous manipulation. While human-centric data collection systems lower the barrier to scaling, they struggle to capture the tactile information during physical interactions. Motivated by this, we present DexViTac, a portable, human-centric data collection system tailored for contact-rich dexterous manipulation. The system enables the high-fidelity acquisition of first-person vision, high-density tactile sensing, end-effector poses, and hand kinematics within unstructured, in-the-wild environments. Building upon this hardware, we propose a kinematics-grounded tactile representation learning algorithm that effectively resolves semantic ambiguities within tactile signals. Leveraging the efficiency of DexViTac, we construct a multimodal dataset comprising over 2,400 visuo-tactile-kinematic demonstrations. Experiments demonstrate that DexViTac achieves a collection efficiency exceeding 248 demonstrations per hour and remains robust against complex visual occlusions. Real-world deployment confirms that policies trained with the proposed dataset and learning strategy achieve an average success rate exceeding 85% across four challenging tasks. This performance significantly outperforms baseline methods, thereby validating the substantial improvement the system provides for learning contact-rich dexterous manipulation. Project page: this https URL.
>
---
#### [new 025] KineVLA: Towards Kinematics-Aware Vision-Language-Action Models with Bi-Level Action Decomposition
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出KineVLA，解决视觉-语言-动作任务中精细运动控制问题，通过双层级分解提升操作的精确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.17524](https://arxiv.org/pdf/2603.17524)**

> **作者:** Gaoge Han; Zhengqing Gao; Ziwen Li; Jiaxin Huang; Shaoli Huang; Fakhri Karray; Mingming Gong; Tongliang Liu
>
> **摘要:** In this paper, we introduce a novel kinematics-rich vision-language-action (VLA) task, in which language commands densely encode diverse kinematic attributes (such as direction, trajectory, orientation, and relative displacement) from initiation through completion, at key moments, unlike existing action instructions that capture kinematics only coarsely or partially, thereby supporting fine-grained and personalized manipulation. In this setting, where task goals remain invariant while execution trajectories must adapt to instruction-level kinematic specifications. To address this challenge, we propose KineVLA, a vision-language-action framework that explicitly decouples goal-level invariance from kinematics-level variability through a bi-level action representation and bi-level reasoning tokens to serve as explicit, supervised intermediate variables that align language and action. To support this task, we construct the kinematics-aware VLA datasets spanning both simulation and real-world robotic platforms, featuring instruction-level kinematic variations and bi-level annotations. Extensive experiments on LIBERO and a Realman-75 robot demonstrate that KineVLA consistently outperforms strong VLA baselines on kinematics-sensitive benchmarks, achieving more precise, controllable, and generalizable manipulation behaviors.
>
---
#### [new 026] P$^{3}$Nav: End-to-End Perception, Prediction and Planning for Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在提升智能体的场景理解与导航能力。针对现有方法缺乏全面感知的问题，提出P³Nav框架，整合感知、预测与规划，增强导航效果。**

- **链接: [https://arxiv.org/pdf/2603.17459](https://arxiv.org/pdf/2603.17459)**

> **作者:** Tianfu Li; Wenbo Chen; Haoxuan Xu; Xinhu Zheng; Haoang Li
>
> **摘要:** In Vision-and-Language Navigation (VLN), an agent is required to plan a path to the target specified by the language instruction, using its visual observations. Consequently, prevailing VLN methods primarily focus on building powerful planners through visual-textual alignment. However, these approaches often bypass the imperative of comprehensive scene understanding prior to planning, leaving the agent with insufficient perception or prediction capabilities. Thus, we propose P$^{3}$Nav, a novel end-to-end framework integrating perception, prediction, and planning in a unified pipeline to strengthen the VLN agent's scene understanding and boost navigation success. Specifically, P$^{3}$Nav augments perception by extracting complementary cues from object-level and map-level perspectives. Subsequently, our P$^{3}$Nav predicts waypoints to model the agent's potential future states, endowing the agent with intrinsic awareness of candidate positions during navigation. Conditioned on these future waypoints, P$^{3}$Nav further forecasts semantic map cues, enabling proactive planning and reducing the strict reliance on purely historical context. Integrating these perceptual and predictive cues, a holistic planning module finally carries out the VLN tasks. Extensive experiments demonstrate that our P$^{3}$Nav achieves new state-of-the-art performance on the REVERIE, R2R-CE, and RxR-CE benchmarks.
>
---
#### [new 027] ReSteer: Quantifying and Refining the Steerability of Multitask Robot Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决多任务策略的可引导性不足问题。通过量化和提升策略的可引导性，提出ReSteer框架，提升机器人在复杂任务中的响应能力。**

- **链接: [https://arxiv.org/pdf/2603.17300](https://arxiv.org/pdf/2603.17300)**

> **作者:** Zhenyang Chen; Alan Tian; Liquan Wang; Benjamin Joffe; Yingyan Celine Lin; Yuxiao Chen; Siddharth Karamcheti; Danfei Xu
>
> **备注:** Project website: this https URL
>
> **摘要:** Despite strong multi-task pretraining, existing policies often exhibit poor task steerability. For example, a robot may fail to respond to a new instruction ``put the bowl in the sink" when moving towards the oven, executing ``close the oven", even though it can complete both tasks when executed separately. We propose ReSteer, a framework to quantify and improve task steerability in multitask robot policies. We conduct an exhaustive evaluation of state-of-the-art policies, revealing a common lack of steerability. We find that steerability is associated with limited overlap among training task trajectory distributions, and introduce a proxy metric to measure this overlap from policy behavior. Building on this insight, ReSteer improves steerability via three components: (i) a steerability estimator that identifies low-steerability states without full-rollout evaluation, (ii) a steerable data generator that synthesizes motion segments from these states, and (iii) a self-refinement pipeline that improves policy steerability using the generated data. In simulation on LIBERO, ReSteer improves steerability by 11\% over 18k rollouts. In real-world experiments, we show that improved steerability is critical for interactive use, enabling users to instruct robots to perform any task at any time. We hope this work motivates further study on quantifying steerability and data collection strategies for large robot policies.
>
---
#### [new 028] Consistency-Driven Dual LSTM Models for Kinematic Control of a Wearable Soft Robotic Arm
- **分类: cs.RO**

- **简介: 该论文属于软体机械臂运动控制任务，解决气动软体执行器的非线性与映射难题。提出双LSTM框架，通过一致性损失提升预测准确性与物理合理性。**

- **链接: [https://arxiv.org/pdf/2603.17672](https://arxiv.org/pdf/2603.17672)**

> **作者:** Xingyu Chen; Yi Xiong; Li Wen
>
> **摘要:** In this paper, we introduce a consistency-driven dual LSTM framework for accurately learning both the forward and inverse kinematics of a pneumatically actuated soft robotic arm integrated into a wearable device. This approach effectively captures the nonlinear and hysteretic behaviors of soft pneumatic actuators while addressing the one-to-many mapping challenge between actuation inputs and end-effector positions. By incorporating a cycle consistency loss, we enhance physical realism and improve the stability of inverse predictions. Extensive experiments-including trajectory tracking, ablation studies, and wearable demonstrations-confirm the effectiveness of our method. Results indicate that the inclusion of the consistency loss significantly boosts prediction accuracy and promotes physical consistency over conventional approaches. Moreover, the wearable soft robotic arm demonstrates strong human-robot collaboration capabilities and adaptability in everyday tasks such as object handover, obstacle-aware pick-and-place, and drawer operation. This work underscores the promising potential of learning-based kinematic models for human-centric, wearable robotic systems.
>
---
#### [new 029] Shielded Reinforcement Learning Under Dynamic Temporal Logic Constraints
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于安全强化学习任务，旨在解决动态时空逻辑约束下的机器人控制问题。通过结合控制屏障函数与无模型RL，确保复杂任务在学习过程中满足STL规范。**

- **链接: [https://arxiv.org/pdf/2603.17152](https://arxiv.org/pdf/2603.17152)**

> **作者:** Sadık Bera Yüksel; Ali Tevfik Buyukkocak; Derya Aksaray
>
> **备注:** 7 pages, 3 figures, 2026 IEEE American Control Conference (ACC)
>
> **摘要:** Reinforcement Learning (RL) has shown promise in various robotics applications, yet its deployment on real systems is still limited due to safety and operational constraints. The safe RL field has gained considerable attention in recent years, which focuses on imposing safety constraints throughout the learning process. However, real systems often require more complex constraints than just safety, such as periodic recharging or time-bounded visits to specific regions. Imposing such spatio-temporal tasks during learning still remains a challenge. Signal Temporal Logic (STL) is a formal language for specifying temporal properties of real-valued signals and provides a way to express such complex tasks. In this paper, we propose a framework that leverages sequential control barrier functions and model-free RL to ensure that the given STL tasks are satisfied throughout the learning process. Our method extends beyond traditional safety constraints by enforcing rich STL specifications, which can involve visits to dynamic targets with unknown trajectories. We also demonstrate the effectiveness of our framework through various simulations.
>
---
#### [new 030] SLowRL: Safe Low-Rank Adaptation Reinforcement Learning for Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决Sim-to-real迁移中的性能下降问题。通过SLowRL框架，安全高效地微调强化学习策略，减少训练时间并避免安全风险。**

- **链接: [https://arxiv.org/pdf/2603.17092](https://arxiv.org/pdf/2603.17092)**

> **作者:** Elham Daneshmand; Shafeef Omar; Glen Berseth; Majid Khadiv; Hsiu-Chin Lin
>
> **摘要:** Sim-to-real transfer of locomotion policies often leads to performance degradation due to the inevitable sim-to-real gap. Naively fine-tuning these policies directly on hardware is problematic, as it poses risks of mechanical failure and suffers from high sample inefficiency. In this paper, we address the challenge of safely and efficiently fine-tuning reinforcement learning (RL) policies for dynamic locomotion tasks. Specifically, we focus on fine-tuning policies learned in simulation directly on hardware, while explicitly enforcing safety constraints. In doing so, we introduce SLowRL, a framework that combines Low-Rank Adaptation (LoRA) with training-time safety enforcement via a recovery policy. We evaluate our method both in simulation and on a real Unitree Go2 quadruped robot for jump and trot tasks. Experimental results show that our method achieves a $46.5\%$ reduction in fine-tuning time and near-zero safety violations compared to standard proximal policy optimization (PPO) baselines. Notably, we find that a rank-1 adaptation alone is sufficient to recover pre-trained performance in the real world, while maintaining stable and safe real-world fine-tuning. These results demonstrate the practicality of safe, efficient fine-tuning for dynamic real-world robotic applications.
>
---
#### [new 031] Physics-informed Deep Mixture-of-Koopmans Vehicle Dynamics Model with Dual-branch Encoder for Distributed Electric-drive Trucks
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于车辆动力学建模任务，旨在解决复杂分布式电动卡车的高精度动态建模问题。提出一种基于Koopman算子和双分支编码器的物理感知方法，提升建模准确性与控制兼容性。**

- **链接: [https://arxiv.org/pdf/2603.17416](https://arxiv.org/pdf/2603.17416)**

> **作者:** Jinyu Miao; Pu Zhang; Rujun Yan; Yifei He; Bowei Zhang; Zheng Fu; Ke Wang; Qi Song; Kun Jiang; Mengmeng Yang; Diange Yang
>
> **备注:** 13 pages, 8 tables, 7 figures
>
> **摘要:** Advanced autonomous driving systems require accurate vehicle dynamics modeling. However, identifying a precise dynamics model remains challenging due to strong nonlinearities and the coupled longitudinal and lateral dynamic characteristics. Previous research has employed physics-based analytical models or neural networks to construct vehicle dynamics representations. Nevertheless, these approaches often struggle to simultaneously achieve satisfactory performance in terms of system identification efficiency, modeling accuracy, and compatibility with linear control strategies. In this paper, we propose a fully data-driven dynamics modeling method tailored for complex distributed electric-drive trucks (DETs), leveraging Koopman operator theory to represent highly nonlinear dynamics in a lifted linear embedding space. To achieve high-precision modeling, we first propose a novel dual-branch encoder which encodes dynamic states and provides a powerful basis for the proposed Koopman-based methods entitled KODE. A physics-informed supervision mechanism, grounded in the geometric consistency of temporal vehicle motion, is incorporated into the training process to facilitate effective learning of both the encoder and the Koopman operator. Furthermore, to accommodate the diverse driving patterns of DETs, we extend the vanilla Koopman operator to a mixture-of-Koopman operator framework, enhancing modeling capability. Simulations conducted in a high-fidelity TruckSim environment and real-world experiments demonstrate that the proposed approach achieves state-of-the-art performance in long-term dynamics state estimation.
>
---
#### [new 032] A Single-Fiber Optical Frequency Domain Reflectometry (OFDR)-Based Shape Sensing of Concentric Tube Steerable Drilling Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人形状感知任务，旨在解决CT-SDR在钻孔过程中实时形状监测的问题。通过集成OFDR光纤与NiTi线，实现高精度连续应变测量。**

- **链接: [https://arxiv.org/pdf/2603.17990](https://arxiv.org/pdf/2603.17990)**

> **作者:** Yash Kulkarni; Mobina Tavangarifard; Daniyal Maroufi; Mohsen Khadem; Justin E. Bird; Jeffrey H. Siewerdsen; Farshid Alambeigi
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** This paper introduces a novel shape-sensing approach for Concentric Tube Steerable Drilling Robots (CT-SDRs) based on Optical Frequency Domain Reflectometry (OFDR). Unlike traditional FBG-based methods, OFDR enables continuous strain measurement along the entire fiber length with enhanced spatial resolution. In the proposed method, a Shape Sensing Assembly (SSA) is first fabricated by integrating a single OFDR fiber with a flat NiTi wire. The calibrated SSA is then routed through and housed within the internal channel of a flexible drilling instrument, which is guided by the pre-shaped NiTi tube of the CT-SDR. In this configuration, the drilling instrument serves as a protective sheath for the SSA during drilling, eliminating the need for integration or adhesion to the instrument surface that is typical of conventional optical sensor approaches. The performance of the proposed SSA, integrated within the cannulated CT-SDR, was thoroughly evaluated under free-bending conditions and during drilling along multiple J-shaped trajectories in synthetic Sawbones phantoms. Results demonstrate accurate and reliable shape-sensing capability, confirming the feasibility and robustness of this integration strategy.
>
---
#### [new 033] SafeLand: Safe Autonomous Landing in Unknown Environments with Bayesian Semantic Mapping
- **分类: cs.RO**

- **简介: 该论文属于自主降落任务，解决无人机在未知环境中安全着陆的问题。通过视觉和轻量传感器构建语义地图，实时识别安全着陆点，确保人机安全。**

- **链接: [https://arxiv.org/pdf/2603.17430](https://arxiv.org/pdf/2603.17430)**

> **作者:** Markus Gross; Andreas Greiner; Sai Bharadhwaj Matha; Felix Soest; Daniel Cremers; Henri Meeß
>
> **摘要:** Autonomous landing of uncrewed aerial vehicles (UAVs) in unknown, dynamic environments poses significant safety challenges, particularly near people and infrastructure, as UAVs transition to routine urban and rural operations. Existing methods often rely on prior maps, heavy sensors like LiDAR, static markers, or fail to handle non-cooperative dynamic obstacles like humans, limiting generalization and real-time performance. To address these challenges, we introduce SafeLand, a lean, vision-based system for safe autonomous landing (SAL) that requires no prior information and operates only with a camera and a lightweight height sensor. Our approach constructs an online semantic ground map via deep learning-based semantic segmentation, optimized for embedded deployment and trained on a consolidation of seven curated public aerial datasets (achieving 70.22% mIoU across 20 classes), which is further refined through Bayesian probabilistic filtering with temporal semantic decay to robustly identify metric-scale landing spots. A behavior tree then governs adaptive landing, iteratively validates the spot, and reacts in real time to dynamic obstacles by pausing, climbing, or rerouting to alternative spots, maximizing human safety. We extensively evaluate our method in 200 simulations and 60 end-to-end field tests across industrial, urban, and rural environments at altitudes up to 100m, demonstrating zero false negatives for human detection. Compared to the state of the art, SafeLand achieves sub-second response latency, substantially lower than previous methods, while maintaining a superior success rate of 95%. To facilitate further research in aerial robotics, we release SafeLand's segmentation model as a plug-and-play ROS package, available at this https URL.
>
---
#### [new 034] From Optimizable to Interactable: Mixed Digital Twin-Empowered Testing of Vehicle-Infrastructure Cooperation Systems
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于车辆-基础设施协同系统测试任务，旨在解决现有测试方法在极端场景下的不足。提出IMPACT框架，结合人机交互与物理虚拟互动，提升测试效果。**

- **链接: [https://arxiv.org/pdf/2603.17497](https://arxiv.org/pdf/2603.17497)**

> **作者:** Jianghong Dong; Chunying Yang; Mengchi Cai; Chaoyi Chen; Qing Xu; Jianqiang Wang; Keqiang Li
>
> **摘要:** Sufficient testing under corner cases is critical for the long-term operation of vehicle-infrastructure cooperation systems (VICS). However, existing corner-case generation methods are primarily AI-driven, and VICS testing under corner cases is typically limited to simulation. In this paper, we introduce an L5 ''Interactable'' level to the VICS digital twin (VICS-DT) taxonomy, extending beyond the conventional L4 ''Optimizable'' level. We further propose an L5-level VICS testing framework, IMPACT (Interactive Mixed-digital-twin Paradigm for Advanced Cooperative vehicle-infrastructure Testing). By enabling direct human interactions with VICS entities, IMPACT incorporates highly uncertain and unpredictable human behaviors into the testing loop, naturally generating high-quality corner cases that complement AI-based methods. Furthermore, the mixedDT-enabled ''Physical-Virtual Action Interaction'' facilitates safe VICS testing under corner cases, incorporating real-world environments and entities rather than purely in simulation. Finally, we implement IMPACT on the I-VIT (Interactive Vehicle-Infrastructure Testbed), and experiments demonstrate its effectiveness. The experimental videos are available at our project website: this https URL.
>
---
#### [new 035] REAL: Robust Extreme Agility via Spatio-Temporal Policy Learning and Physics-Guided Filtering
- **分类: cs.RO**

- **简介: 该论文提出REAL框架，解决极端地形中机器人感知退化导致的运动失败问题。通过结合视觉、本体感觉和时间记忆，提升机器人在恶劣环境下的敏捷性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.17653](https://arxiv.org/pdf/2603.17653)**

> **作者:** Jialong Liu; Dehan Shen; Yanbo Wen; Zeyu Jiang; Changhao Chen
>
> **摘要:** Extreme legged parkour demands rapid terrain assessment and precise foot placement under highly dynamic conditions. While recent learning-based systems achieve impressive agility, they remain fundamentally fragile to perceptual degradation, where even brief visual noise or latency can cause catastrophic failure. To overcome this, we propose Robust Extreme Agility Learning (REAL), an end-to-end framework for reliable parkour under sensory corruption. Instead of relying on perfectly clean perception, REAL tightly couples vision, proprioceptive history, and temporal memory. We distill a cross-modal teacher policy into a deployable student equipped with a FiLM-modulated Mamba backbone to actively filter visual noise and build short-term terrain memory actively. Furthermore, a physics-guided Bayesian state estimator enforces rigid-body consistency during high-impact maneuvers. Validated on a Unitree Go2 quadruped, REAL successfully traverses extreme obstacles even with a 1-meter visual blind zone, while strictly satisfying real-time control constraints with a bounded 13.1 ms inference time.
>
---
#### [new 036] Visual SLAM with DEM Anchoring for Lunar Surface Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决月球表面长期导航中的定位漂移问题。通过融合数字高程模型（DEM）约束，提升导航精度与地图一致性。**

- **链接: [https://arxiv.org/pdf/2603.17229](https://arxiv.org/pdf/2603.17229)**

> **作者:** Adam Dai; Guillem Casadesus Vila; Grace Gao
>
> **备注:** Accepted to IEEE Aerospace Conference 2026
>
> **摘要:** Future lunar missions will require autonomous rovers capable of traversing tens of kilometers across challenging terrain while maintaining accurate localization and producing globally consistent maps. However, the absence of global positioning systems, extreme illumination, and low-texture regolith make long-range navigation on the Moon particularly difficult, as visual-inertial odometry pipelines accumulate drift over extended traverses. To address this challenge, we present a stereo visual simultaneous localization and mapping (SLAM) system that integrates learned feature detection and matching with global constraints from digital elevation models (DEMs). Our front-end employs learning-based feature extraction and matching to achieve robustness to illumination extremes and repetitive terrain, while the back-end incorporates DEM-derived height and surface-normal factors into a pose graph, providing absolute surface constraints that mitigate long-term drift. We validate our approach using both simulated lunar traverse data generated in Unreal Engine and real Moon/Mars analog data collected from Mt. Etna. Results demonstrate that DEM anchoring consistently reduces absolute trajectory error compared to baseline SLAM methods, lowering drift in long-range navigation even in repetitive or visually aliased terrain.
>
---
#### [new 037] Bringing Network Coding into Multi-Robot Systems: Interplay Study for Autonomous Systems over Wireless Communications
- **分类: cs.RO; cs.MA; cs.NI**

- **简介: 该论文属于多机器人系统任务，解决无线通信中的延迟与丢包问题。通过引入网络编码，提升通信可靠性与实时性，优化机器人协同与安全决策。**

- **链接: [https://arxiv.org/pdf/2603.17472](https://arxiv.org/pdf/2603.17472)**

> **作者:** Anil Zaher; Kiril Solovey; Alejandro Cohen
>
> **摘要:** Communication is a core enabler for multi-robot systems (MRS), providing the mechanism through which robots exchange state information, coordinate actions, and satisfy safety constraints. While many MRS autonomy algorithms assume reliable and timely message delivery, realistic wireless channels introduce delay, erasures, and ordering stalls that can degrade performance and compromise safety-critical decisions of the robot task. In this paper, we investigate how transport-layer reliability mechanisms that mitigate communication losses and delays shape the autonomy-communication loop. We show that conventional non-coded retransmission-based protocols introduce long delays that are misaligned with the timeliness requirements of MRS applications, and may render the received data irrelevant. As an alternative, we advocate for adaptive and causal network coding, which proactively injects coded redundancy to achieve the desired delay and throughput that enable relevant data delivery to the robotic task. Specifically, this method adapts to channel conditions between robots and causally tunes the communication rates via efficient algorithms. We present two case studies: cooperative localization under delayed and lossy inter-robot communication, and a safety-critical overtaking maneuver where timely vehicle-to-vehicle message availability determines whether an ego vehicle can abort to avoid a crash. Our results demonstrate that coding-based communication significantly reduces in-order delivery stalls, preserves estimation consistency under delay, and improves deadline reliability relative to retransmission-based transport. Overall, the study highlights the need to jointly design autonomy algorithms and communication mechanisms, and positions network coding as a principled tool for dependable multi-robot operation over wireless networks.
>
---
#### [new 038] Efficient and Reliable Teleoperation through Real-to-Sim-to-Real Shared Autonomy
- **分类: cs.RO**

- **简介: 该论文属于机器人遥操作任务，解决真实环境中精细操作效率低、易出错的问题。通过构建真实到仿真再到真实的共享自主框架，提升操作性能。**

- **链接: [https://arxiv.org/pdf/2603.17016](https://arxiv.org/pdf/2603.17016)**

> **作者:** Shuo Sha; Yixuan Wang; Binghao Huang; Antonio Loquerico; Yunzhu Li
>
> **备注:** Project Page: this https URL
>
> **摘要:** Fine-grained, contact-rich teleoperation remains slow, error-prone, and unreliable in real-world manipulation tasks, even for experienced operators. Shared autonomy offers a promising way to improve performance by combining human intent with automated assistance, but learning effective assistance in simulation requires a faithful model of human behavior, which is difficult to obtain in practice. We propose a real-to-sim-to-real shared autonomy framework that augments human teleoperation with learned corrective behaviors, using a simple yet effective k-nearest-neighbor (kNN) human surrogate to model operator actions in simulation. The surrogate is fit from less than five minutes of real-world teleoperation data and enables stable training of a residual copilot policy with model-free reinforcement learning. The resulting copilot is deployed to assist human operators in real-world fine-grained manipulation tasks. Through simulation experiments and a user study with sixteen participants on industry-relevant tasks, including nut threading, gear meshing, and peg insertion, we show that our system improves task success for novice operators and execution efficiency for experienced operators compared to direct teleoperation and shared-autonomy baselines that rely on expert priors or behavioral-cloning pilots. In addition, copilot-assisted teleoperation produces higher-quality demonstrations for downstream imitation learning.
>
---
#### [new 039] AERR-Nav: Adaptive Exploration-Recovery-Reminiscing Strategy for Zero-Shot Object Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决未知多层环境中机器人导航困难的问题。提出AERR-Nav框架，通过自适应策略提升导航效果。**

- **链接: [https://arxiv.org/pdf/2603.17712](https://arxiv.org/pdf/2603.17712)**

> **作者:** Jingzhi Huang; Junkai Huang; Haoyang Yang; Haoang Li; Yi Wang
>
> **摘要:** Zero-Shot Object Navigation (ZSON) in unknown multi-floor environments presents a significant challenge. Recent methods, mostly based on semantic value greedy waypoint selection, spatial topology-enhanced memory, and Multimodal Large Language Model (MLLM) as a decision-making framework, have led to improvements. However, these architectures struggle to balance exploration and exploitation for ZSON when encountering unseen environments, especially in multi-floor settings, such as robots getting stuck at narrow intersections, endlessly wandering, or failing to find stair entrances. To overcome these challenges, we propose AERR-Nav, a Zero-Shot Object Navigation framework that dynamically adjusts its state based on the robot's environment. Specifically, AERR-Nav has the following two key advantages: (1) An Adaptive Exploration-Recovery-Reminiscing Strategy, enables robots to dynamically transition between three states, facilitating specialized responses to diverse navigation scenarios. (2) An Adaptive Exploration State featuring Fast and Slow-Thinking modes helps robots better balance exploration, exploitation, and higher-level reasoning based on evolving environmental information. Extensive experiments on the HM3D and MP3D benchmarks demonstrate that our AERR-Nav achieves state-of-the-art performance among zero-shot methods. Comprehensive ablation studies further validate the efficacy of our proposed strategy and modules.
>
---
#### [new 040] FastLoop: Parallel Loop Closing with GPU-Acceleration in Visual SLAM
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决循环闭合计算复杂的问题。通过GPU加速优化，提升循环闭合效率，实现更快的处理速度。**

- **链接: [https://arxiv.org/pdf/2603.17201](https://arxiv.org/pdf/2603.17201)**

> **作者:** Soudabeh Mohammadhashemi; Shishir Gopinath; Kimia Khabiri; Parsa Hosseininejad; Karthik Dantu; Steven Y. Ko
>
> **摘要:** Visual SLAM systems combine visual tracking with global loop closure to maintain a consistent map and accurate localization. Loop closure is a computationally expensive process as we need to search across the whole map for matches. This paper presents FastLoop, a GPU-accelerated loop closing module to alleviate this computational complexity. We identify key performance bottlenecks in the loop closing pipeline of visual SLAM and address them through parallel optimizations on the GPU. Specifically, we use task-level and data-level parallelism and integrate a GPU-accelerated pose graph optimization. Our implementation is built on top of ORB-SLAM3 and leverages CUDA for GPU programming. Experimental results show that FastLoop achieves an average speedup of 1.4x and 1.3x on the EuRoC dataset and 3.0x and 2.4x on the TUM-VI dataset for the loop closing module on desktop and embedded platforms, respectively, while maintaining the accuracy of the original system.
>
---
#### [new 041] TrackDeform3D: Markerless and Autonomous 3D Keypoint Tracking and Dataset Collection for Deformable Objects
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D关键点跟踪任务，解决变形物体数据集构建与轨迹跟踪难题。提出TrackDeform3D框架，利用RGB-D相机自主采集高质量数据，提升跟踪精度与数据一致性。**

- **链接: [https://arxiv.org/pdf/2603.17068](https://arxiv.org/pdf/2603.17068)**

> **作者:** Yeheng Zong; Yizhou Chen; Alexander Bowler; Chia-Tung Yang; Ram Vasudevan
>
> **摘要:** Structured 3D representations such as keypoints and meshes offer compact, expressive descriptions of deformable objects, jointly capturing geometric and topological information useful for downstream tasks such as dynamics modeling and motion planning. However, robustly extracting such representations remains challenging, as current perception methods struggle to handle complex deformations. Moreover, large-scale 3D data collection remains a bottleneck: existing approaches either require prohibitive data collection efforts, such as labor-intensive annotation or expensive motion capture setups, or rely on simplifying assumptions that break down in unstructured environments. As a result, large-scale 3D datasets and benchmarks for deformable objects remain scarce. To address these challenges, this paper presents an affordable and autonomous framework for collecting 3D datasets of deformable objects using only RGB-D cameras. The proposed method identifies 3D keypoints and robustly tracks their trajectories, incorporating motion consistency constraints to produce temporally smooth and geometrically coherent data. TrackDeform3D is evaluated against several state-of-the-art tracking methods across diverse object categories and demonstrates consistent improvements in both geometric and tracking accuracy. Using this framework, this paper presents a high-quality, large-scale dataset consisting of 6 deformable objects, totaling 110 minutes of trajectory data.
>
---
#### [new 042] The Port-Hamiltonian Structure of Vehicle Manipulator Systems
- **分类: eess.SY; cs.RO; math.DG**

- **简介: 该论文属于机器人动力学建模任务，旨在解决车辆机械臂系统能量结构表述问题。通过构建保结构的哈密顿模型，揭示系统能量流动与守恒特性，提升控制与仿真效果。**

- **链接: [https://arxiv.org/pdf/2603.16882](https://arxiv.org/pdf/2603.16882)**

> **作者:** Ramy Rashad
>
> **摘要:** This paper presents a port-Hamiltonian formulation of vehicle-manipulator systems (VMS), a broad class of robotic systems including aerial manipulators, underwater manipulators, space robots, and omnidirectional mobile manipulators. Unlike existing Lagrangian formulations that obscure the underlying energetic structure, the proposed port-Hamiltonian formulation explicitly reveals the energy flow and conservation properties of these complex mechanical systems. We derive the port-Hamiltonian dynamics from first principles using Hamiltonian reduction theory. Two complementary formulations are presented: a standard form that directly exposes the energy structure, and an inertially-decoupled form that leverages the principal bundle structure of the VMS configuration space and is particularly suitable for control design and numerical simulation. The coordinate-free geometric approach we follow avoids singularities associated with local parameterizations of the base orientation. We rigorously establish the mathematical equivalence between our port-Hamiltonian formulations and existing reduced Euler-Lagrange and Boltzmann-Hamel equations found in the robotics and geometric mechanics literature.
>
---
#### [new 043] BEV-SLD: Self-Supervised Scene Landmark Detection for Global Localization with LiDAR Bird's-Eye View Images
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BEV-SLD，用于LiDAR全局定位的自监督场景地标检测方法，解决场景特定地标识别问题，通过BEV图像和一致性损失实现精准定位。**

- **链接: [https://arxiv.org/pdf/2603.17159](https://arxiv.org/pdf/2603.17159)**

> **作者:** David Skuddis; Vincent Ress; Wei Zhang; Vincent Ofosu Nyako; Norbert Haala
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** We present BEV-SLD, a LiDAR global localization method building on the Scene Landmark Detection (SLD) concept. Unlike scene-agnostic pipelines, our self-supervised approach leverages bird's-eye-view (BEV) images to discover scene-specific patterns at a prescribed spatial density and treat them as landmarks. A consistency loss aligns learnable global landmark coordinates with per-frame heatmaps, yielding consistent landmark detections across the scene. Across campus, industrial, and forest environments, BEV-SLD delivers robust localization and achieves strong performance compared to state-of-the-art methods.
>
---
#### [new 044] Asymmetric Nash Seeking via Best Response Maps: Global Linear Convergence and Robustness to Inexact Reaction Models
- **分类: cs.GT; cs.MA; cs.RO; eess.SY; math.OC**

- **简介: 该论文属于多智能体决策与控制任务，解决信息不对称下的纳什均衡寻找问题。提出一种无需完全了解对方目标的迭代算法，证明其全局线性收敛性及对近似响应映射的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.17058](https://arxiv.org/pdf/2603.17058)**

> **作者:** Mahdis Rabbani; Navid Mojahed; Shima Nazari
>
> **备注:** 6 Pages, 2 Figures, Preprint submitted to IEEE L-CSS and CDC 2026
>
> **摘要:** Nash equilibria provide a principled framework for modeling interactions in multi-agent decision-making and control. However, many equilibrium-seeking methods implicitly assume that each agent has access to the other agents' objectives and constraints, an assumption that is often unrealistic in practice. This letter studies a class of asymmetric-information two-player constrained games with decoupled feasible sets, in which Player 1 knows its own objective and constraints while Player 2 is available only through a best-response map. For this class of games, we propose an asymmetric projected gradient descent-best response iteration that does not require full mutual knowledge of both players' optimization problems. Under suitable regularity conditions, we establish the existence and uniqueness of the Nash equilibrium and prove global linear convergence of the proposed iteration when the best-response map is exact. Recognizing that best-response maps are often learned or estimated, we further analyze the inexact case and show that, when the approximation error is uniformly bounded by $\varepsilon$, the iterates enter an explicit $O(\varepsilon)$ neighborhood of the true Nash equilibrium. Numerical results on a benchmark game corroborate the predicted convergence behavior and error scaling.
>
---
#### [new 045] L4acados: Learning-based models for acados, applied to Gaussian process-based predictive control
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于控制领域，解决将学习模型融入MPC的问题。提出L4acados框架，实现Python模型与acados的高效集成，提升控制性能与实时性。**

- **链接: [https://arxiv.org/pdf/2411.19258](https://arxiv.org/pdf/2411.19258)**

> **作者:** Amon Lahr; Joshua Näf; Kim P. Wabersich; Jonathan Frey; Pascal Siehl; Andrea Carron; Moritz Diehl; Melanie N. Zeilinger
>
> **摘要:** Incorporating learning-based models, such as artificial neural networks or Gaussian processes, into model predictive control (MPC) strategies can significantly improve control performance and online adaptation capabilities for real-world applications. Still, enabling state-of-the-art implementations of learning-based models for MPC is complicated by the challenge of interfacing machine learning frameworks with real-time optimal control software. This work aims at filling this gap by incorporating external sensitivities in sequential quadratic programming solvers for nonlinear optimal control. To this end, we provide L4acados, a general framework for incorporating Python-based dynamics models in the real-time optimal control software acados. By computing external sensitivities via a user-defined Python module, L4acados enables the implementation of MPC controllers with learning-based residual models in acados, while supporting parallelization of sensitivity computations when preparing the quadratic subproblems. We demonstrate significant speed-ups and superior scaling properties of L4acados compared to available software using a neural-network-based control example. Last, we provide an efficient and modular real-time implementation of Gaussian process-based MPC using L4acados, which is applied to two hardware examples: autonomous miniature racing, as well as motion control of a full-scale autonomous vehicle for an ISO lane change maneuver.
>
---
#### [new 046] Real-Time Online Learning for Model Predictive Control using a Spatio-Temporal Gaussian Process Approximation
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于控制领域，解决实时模型预测控制中高计算成本问题，通过引入时空高斯过程近似实现高效在线学习，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2603.17632](https://arxiv.org/pdf/2603.17632)**

> **作者:** Lars Bartels; Amon Lahr; Andrea Carron; Melanie N. Zeilinger
>
> **备注:** to be published at 2026 IEEE International Conference on Robotics & Automation (ICRA)
>
> **摘要:** Learning-based model predictive control (MPC) can enhance control performance by correcting for model inaccuracies, enabling more precise state trajectory predictions than traditional MPC. A common approach is to model unknown residual dynamics as a Gaussian process (GP), which leverages data and also provides an estimate of the associated uncertainty. However, the high computational cost of online learning poses a major challenge for real-time GP-MPC applications. This work presents an efficient implementation of an approximate spatio-temporal GP model, offering online learning at constant computational complexity. It is optimized for GP-MPC, where it enables improved control performance by learning more accurate system dynamics online in real-time, even for time-varying systems. The performance of the proposed method is demonstrated by simulations and hardware experiments in the exemplary application of autonomous miniature racing.
>
---
#### [new 047] Physics-informed offline reinforcement learning eliminates catastrophic fuel waste in maritime routing
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于航海路径优化任务，旨在减少航运燃料消耗和碳排放。通过物理引导的离线强化学习框架PIER，解决传统方法导致的极端燃料浪费问题，并提升路径安全性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.17319](https://arxiv.org/pdf/2603.17319)**

> **作者:** Aniruddha Bora; Julie Chalfant; Chryssostomos Chryssostomidis
>
> **摘要:** International shipping produces approximately 3% of global greenhouse gas emissions, yet voyage routing remains dominated by heuristic methods. We present PIER (Physics-Informed, Energy-efficient, Risk-aware routing), an offline reinforcement learning framework that learns fuel-efficient, safety-aware routing policies from physics-calibrated environments grounded in historical vessel tracking data and ocean reanalysis products, requiring no online simulator. Validated on one full year (2023) of AIS data across seven Gulf of Mexico routes (840 episodes per method), PIER reduces mean CO2 emissions by 10% relative to great-circle routing. However, PIER's primary contribution is eliminating catastrophic fuel waste: great-circle routing incurs extreme fuel consumption (>1.5x median) in 4.8% of voyages; PIER reduces this to 0.5%, a 9-fold reduction. Per-voyage fuel variance is 3.5x lower (p<0.001), with bootstrap 95% CI for mean savings [2.9%, 15.7%]. Partial validation against observed AIS vessel behavior confirms consistency with the fastest real transits while exhibiting 23.1x lower variance. Crucially, PIER is forecast-independent: unlike A* path optimization whose wave protection degrades 4.5x under realistic forecast uncertainty, PIER maintains constant performance using only local observations. The framework combines physics-informed state construction, demonstration-augmented offline data, and a decoupled post-hoc safety shield, an architecture that transfers to wildfire evacuation, aircraft trajectory optimization, and autonomous navigation in unmapped terrain.
>
---
#### [new 048] GMT: Goal-Conditioned Multimodal Transformer for 6-DOF Object Trajectory Synthesis in 3D Scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GMT模型，解决3D场景中6-DOF物体轨迹生成任务，通过融合几何、语义等信息实现精准轨迹控制。**

- **链接: [https://arxiv.org/pdf/2603.17993](https://arxiv.org/pdf/2603.17993)**

> **作者:** Huajian Zeng; Abhishek Saroha; Daniel Cremers; Xi Wang
>
> **备注:** Accpeted by 3DV 2026. Project Page: this https URL
>
> **摘要:** Synthesizing controllable 6-DOF object manipulation trajectories in 3D environments is essential for enabling robots to interact with complex scenes, yet remains challenging due to the need for accurate spatial reasoning, physical feasibility, and multimodal scene understanding. Existing approaches often rely on 2D or partial 3D representations, limiting their ability to capture full scene geometry and constraining trajectory precision. We present GMT, a multimodal transformer framework that generates realistic and goal-directed object trajectories by jointly leveraging 3D bounding box geometry, point cloud context, semantic object categories, and target end poses. The model represents trajectories as continuous 6-DOF pose sequences and employs a tailored conditioning strategy that fuses geometric, semantic, contextual, and goaloriented information. Extensive experiments on synthetic and real-world benchmarks demonstrate that GMT outperforms state-of-the-art human motion and human-object interaction baselines, such as CHOIS and GIMO, achieving substantial gains in spatial accuracy and orientation control. Our method establishes a new benchmark for learningbased manipulation planning and shows strong generalization to diverse objects and cluttered 3D environments. Project page: https://huajian- this http URL. io/projects/gmt/.
>
---
## 更新

#### [replaced 001] REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于人形机器人运动操作任务，解决离线策略与低层控制器脱节导致的执行不稳定问题。通过强化学习微调扩散策略，提升任务成功率和运动质量。**

- **链接: [https://arxiv.org/pdf/2603.13707](https://arxiv.org/pdf/2603.13707)**

> **作者:** Zhaoyuan Gu; Yipu Chen; Zimeng Chai; Alfred Cueva; Thong Nguyen; Yifan Wu; Huishu Xue; Minji Kim; Isaac Legene; Fukang Liu; Matthew Kim; Ayan Barula; Yongxin Chen; Ye Zhao
>
> **摘要:** Humanoid loco-manipulation requires coordinated high-level motion plans with stable, low-level whole-body execution under complex robot-environment dynamics and long-horizon tasks. While diffusion policies (DPs) show promise for learning from demonstrations, deploying them on humanoids poses critical challenges: the motion planner trained offline is decoupled from the low-level controller, leading to poor command tracking, compounding distribution shift, and task failures. The common approach of scaling demonstration data is prohibitively expensive for high-dimensional humanoid systems. To address this challenge, we present REFINE-DP (REinforcement learning FINE-tuning of Diffusion Policy), a hierarchical framework that jointly optimizes a DP high-level planner and an RL-based low-level loco-manipulation controller. The DP is fine-tuned via a PPO-based diffusion policy gradient to improve task success rate, while the controller is simultaneously updated to accurately track the planner's evolving command distribution, reducing the distributional mismatch that degrades motion quality. We validate REFINE-DP on a humanoid robot performing loco-manipulation tasks, including door traversal and long-horizon object transport. REFINE-DP achieves an over $90\%$ success rate in simulation, even in out-of-distribution cases not seen in the pre-trained data, and enables smooth autonomous task execution in real-world dynamic environments. Our proposed method substantially outperforms pre-trained DP baselines and demonstrates that RL fine-tuning is key to reliable humanoid loco-manipulation. this https URL
>
---
#### [replaced 002] U-ARM : Ultra low-cost general teleoperation interface for robot manipulation
- **分类: cs.RO**

- **简介: 该论文提出U-Arm，一种低成本、通用的机器人操作接口，解决传统遥操作系统成本高、适配性差的问题。通过优化机械设计和控制逻辑，实现高效、兼容的远程操作。**

- **链接: [https://arxiv.org/pdf/2509.02437](https://arxiv.org/pdf/2509.02437)**

> **作者:** Yanwen Zou; Zhaoye Zhou; Chenyang Shi; Zewei Ye; Junda Huang; Yan Ding; Bo Zhao
>
> **摘要:** We propose U-Arm, a low-cost and rapidly adaptable leader-follower teleoperation framework designed to interface with most of commercially available robotic arms. Our system supports teleoperation through three structurally distinct 3D-printed leader arms that share consistent control logic, enabling seamless compatibility with diverse commercial robot configurations. Compared with previous open-source leader-follower interfaces, we further optimized both the mechanical design and servo selection, achieving a bill of materials (BOM) cost of only \$50.5 for the 6-DoF leader arm and \$56.8 for the 7-DoF version. To enhance usability, we mitigate the common challenge in controlling redundant degrees of freedom by %engineering methods mechanical and control optimizations. Experimental results demonstrate that U-Arm achieves 39\% higher data collection efficiency and comparable task success rates across multiple manipulation scenarios compared with Joycon, another low-cost teleoperation interface. We have open-sourced all CAD models of three configs and also provided simulation support for validating teleoperation workflows. We also open-sourced real-world manipulation data collected with U-Arm. The project website is this https URL.
>
---
#### [replaced 003] OGScene3D: Incremental Open-Vocabulary 3D Gaussian Scene Graph Mapping for Scene Understanding
- **分类: cs.RO**

- **简介: 该论文属于场景理解任务，解决机器人在动态环境中构建开放词汇3D语义图的问题。提出OGScene3D系统，实现增量式3D语义映射与场景图构建。**

- **链接: [https://arxiv.org/pdf/2603.16301](https://arxiv.org/pdf/2603.16301)**

> **作者:** Siting Zhu; Ziyun Lu; Guangming Wang; Chenguang Huang; Yongbo Chen; I-Ming Chen; Wolfram Burgard; Hesheng Wang
>
> **摘要:** Open-vocabulary scene understanding is crucial for robotic applications, enabling robots to comprehend complex 3D environmental contexts and supporting various downstream tasks such as navigation and manipulation. However, existing methods require pre-built complete 3D semantic maps to construct scene graphs for scene understanding, which limits their applicability in robotic scenarios where environments are explored incrementally. To address this challenge, we propose OGScene3D, an open-vocabulary scene understanding system that achieves accurate 3D semantic mapping and scene graph construction incrementally. Our system employs a confidence-based Gaussian semantic representation that jointly models semantic predictions and their reliability, enabling robust scene modeling. Building on this representation, we introduce a hierarchical 3D semantic optimization strategy that achieves semantic consistency through local correspondence establishment and global refinement, thereby constructing globally consistent semantic maps. Moreover, we design a long-term global optimization method that leverages temporal memory of historical observations to enhance semantic predictions. By integrating 2D-3D semantic consistency with Gaussian rendering contribution, this method continuously refines the semantic understanding of the entire scene. Furthermore, we develop a progressive graph construction approach that dynamically creates and updates both nodes and semantic relationships, allowing continuous updating of the 3D scene graphs. Extensive experiments on widely used datasets and real-world scenes demonstrate the effectiveness of our OGScene3D on open-vocabulary scene understanding.
>
---
#### [replaced 004] Dual Quaternion Based Contact Modeling for Fast and Smooth Collision Recovery of Quadrotors
- **分类: cs.RO**

- **简介: 该论文属于无人机碰撞恢复任务，解决传统接触模型在快速状态变化中出现的不一致问题，提出基于双四元数的接触建模方法，提升碰撞后稳定性与响应速度。**

- **链接: [https://arxiv.org/pdf/2603.14698](https://arxiv.org/pdf/2603.14698)**

> **作者:** Valentin Gaucher; Wenlong Zhang
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Unmanned aerial vehicles (UAVs) operating in cluttered environments require accurate impact modeling to maintain stability post collisions. However, conventional contact models decouple linear and angular impulses, risking manifold inconsistency during rapid state transitions. This letter presents a dual quaternion reset map that resolves rigid-body impacts directly on the SE(3) manifold. By operating on the unified spatial twist (linear and angular velocities as a single dual entity), the proposed formulation is shown to be algebraically equivalent to the classical Newton impulse model while preserving manifold consistency during discrete state jumps. Building on this framework, a hybrid recovery controller is designed that couples linear and angular momentum to ensure strict energy dissipation across impacts. Hardware-in-the-loop benchmarks demonstrate a 24% reduction in execution latency compared to an optimized matrix-based implementation. High-fidelity MuJoCo simulations validate the controller's response to complex contact dynamics, with Monte Carlo trials showing a 56.3% reduction in post-impact root-mean-square error (RMSE) and a 61.1% decrease in peak kinetic energy compared to decoupled baseline controllers.
>
---
#### [replaced 005] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型的后训练任务，旨在解决数据稀缺、安全性和执行效率问题。提出RehearseVLA框架，利用虚拟环境进行强化学习，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.24948](https://arxiv.org/pdf/2509.24948)**

> **作者:** Junjin Xiao; Yandan Yang; Xinyuan Chang; Ronghan Chen; Feng Xiong; Mu Xu; Wei-Shi Zheng; Qing Zhang
>
> **备注:** Accepted to CVPR2026
>
> **摘要:** Vision-Language-Action (VLA) models trained via imitation learning suffer from significant performance degradation in data-scarce scenarios due to their reliance on large-scale demonstration datasets. Although reinforcement learning (RL)-based post-training has proven effective in addressing data scarcity, its application to VLA models is hindered by the non-resettable nature of real-world environments. This limitation is particularly critical in high-risk domains such as industrial automation, where interactions often induce state changes that are costly or infeasible to revert. Furthermore, existing VLA approaches lack a reliable mechanism for detecting task completion, leading to redundant actions that reduce overall task success rates. To address these challenges, we propose RehearseVLA:, an RL-based post-training framework that replaces physical interaction with a low-cost world model-based virtual simulator. RehearseVLA: consists of two key components: (1) a physically-consistent world simulator that generates temporally consistent future visual observations, and (2) a vision-language model (VLM)-guided instant reflector that provides continuous reward signals and predicts action termination. This simulated environment enables VLA models to safely explore and generalize beyond their initial imitation learning distribution. Our method achieves notable performance gains with as few as five expert demonstrations per task. Experiments on complex robotic manipulation tasks demonstrate that RehearseVLA: effectively overcomes the data inefficiency, safety constraints, and inefficient execution of conventional VLA models that rely on real-world interaction, offering a practical and scalable solution for post-training in resource-constrained settings. Our code is available at this https URL.
>
---
#### [replaced 006] NavThinker: Action-Conditioned World Models for Coupled Prediction and Planning in Social Navigation
- **分类: cs.RO**

- **简介: 该论文属于社会导航任务，解决机器人在动态人类环境中安全行动的问题。提出NavThinker框架，结合预测与规划，提升导航成功率。**

- **链接: [https://arxiv.org/pdf/2603.15359](https://arxiv.org/pdf/2603.15359)**

> **作者:** Tianshuai Hu; Zeying Gong; Lingdong Kong; XiaoDong Mei; Yiyi Ding; Qi Zeng; Ao Liang; Rong Li; Yangyi Zhong; Junwei Liang
>
> **摘要:** Social navigation requires robots to act safely in dynamic human environments. Effective behavior demands thinking ahead: reasoning about how the scene and pedestrians evolve under different robot actions rather than reacting to current observations alone. This creates a coupled prediction-planning challenge, where robot actions and human motion mutually influence each other. To address this challenge, we propose NavThinker, a future-aware framework that couples an action-conditioned world model with on-policy reinforcement learning. The world model operates in the Depth Anything V2 patch feature space and performs autoregressive prediction of future scene geometry and human motion; multi-head decoders then produce future depth maps and human trajectories, yielding a future-aware state aligned with traversability and interaction risk. Crucially, we train the policy with DD-PPO while injecting world-model think-ahead signals via: (i) action-conditioned future features fused into the current observation embedding and (ii) social reward shaping from predicted human trajectories. Experiments on single- and multi-robot Social-HM3D show state-of-the-art navigation success, with zero-shot transfer to Social-MP3D and real-world deployment on a Unitree Go2, validating generalization and practical applicability. Webpage: this https URL.
>
---
#### [replaced 007] Volumetric Ergodic Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决传统ergodic控制未考虑机器人体积的问题。提出一种基于体积状态表示的ergodic控制方法，提升覆盖效率并保持任务完成率。**

- **链接: [https://arxiv.org/pdf/2511.11533](https://arxiv.org/pdf/2511.11533)**

> **作者:** Jueun Kwon; Max M. Sun; Todd Murphey
>
> **备注:** 8 pages, 8 figures; Accepted to 2026 IEEE International Conference on Robotics and Automation (ICRA); Project website: this https URL
>
> **摘要:** Ergodic control synthesizes optimal coverage behaviors over spatial distributions for nonlinear systems. However, existing formulations model the robot as a non-volumetric point, whereas in practice a robot interacts with the environment through its body and sensors with physical volume. In this work, we introduce a new ergodic control formulation that optimizes spatial coverage using a volumetric state representation. Our method preserves the asymptotic coverage guarantees of ergodic control, adds minimal computational overhead for real-time control, and supports arbitrary sample-based volumetric models. We evaluate our method across search and manipulation tasks -- with multiple robot dynamics and end-effector geometries or sensor models -- and show that it improves coverage efficiency by more than a factor of two while maintaining a 100% task completion rate across all experiments, outperforming the standard ergodic control method. Finally, we demonstrate the effectiveness of our method on a robot arm performing mechanical erasing tasks. Project website: this https URL
>
---
#### [replaced 008] Push, Press, Slide: Mode-Aware Planar Contact Manipulation via Reduced-Order Models
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究非抓取式平面操作任务，解决接触力学复杂和计算成本高的问题。通过构建简化的动力学模型，实现快速、优化-free 的操作控制。**

- **链接: [https://arxiv.org/pdf/2603.12399](https://arxiv.org/pdf/2603.12399)**

> **作者:** Melih Özcan; Umut Orguner; Ozgur S. Oguz
>
> **备注:** 8 pages, 13 figures. Submitted to IEEE IROS 2026
>
> **摘要:** Non-prehensile planar manipulation, including pushing and press-and-slide, is critical for diverse robotic tasks, but notoriously challenging due to hybrid contact mechanics, under-actuation, and asymmetric friction limits that traditionally necessitate computationally expensive iterative control. In this paper, we propose a mode-aware framework for planar manipulation with one or two robotic arms based on contact topology selection and reduced-order kinematic modeling. Our core insight is that complex wrench-twist limit surface mechanics can be abstracted into a discrete library of physically intuitive models. We systematically map various single-arm and bimanual contact topologies to simple non-holonomic formulations, e.g. unicycle for simplified press-and-slide motion. By anchoring trajectory generation to these reduced-order models, our framework computes the required object wrench and distributes feasible, friction-bounded contact forces via a direct algebraic allocator. We incorporate manipulator kinematics to ensure long-horizon feasibility and demonstrate our fast, optimization-free approach in simulation across diverse single-arm and bimanual manipulation tasks. Supplementary videos and additional information are available at: this https URL
>
---
#### [replaced 009] TurboMap: GPU-Accelerated Local Mapping for Visual SLAM
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决局部建图的实时性问题。通过GPU加速和优化，提升局部建图效率，减少延迟，同时保持精度。**

- **链接: [https://arxiv.org/pdf/2511.02036](https://arxiv.org/pdf/2511.02036)**

> **作者:** Parsa Hosseininejad; Kimia Khabiri; Shishir Gopinath; Soudabeh Mohammadhashemi; Karthik Dantu; Steven Y. Ko
>
> **摘要:** In real-time Visual SLAM systems, local mapping must operate under strict latency constraints, as delays degrade map quality and increase the risk of tracking failure. GPU parallelization offers a promising way to reduce latency. However, parallelizing local mapping is challenging due to synchronized shared-state updates and the overhead of transferring large map data structures to the GPU. This paper presents TurboMap, a GPU-parallelized and CPU-optimized local mapping backend that holistically addresses these challenges. We restructure Map Point Creation to enable parallel Keypoint Correspondence Search on the GPU, redesign and parallelize Map Point Fusion, optimize Redundant Keyframe Culling on the CPU, and integrate a fast GPU-based Local Bundle Adjustment solver. To minimize data transfer and synchronization costs, we introduce persistent GPU-resident keyframe storage. Experiments on the EuRoC and TUM-VI datasets show average local mapping speedups of 1.3x and 1.6x, respectively, while preserving accuracy.
>
---
#### [replaced 010] SimScale: Learning to Drive via Real-World Simulation at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决真实场景数据不足问题。通过SimScale框架生成大量模拟数据，并结合真实数据训练，提升模型的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.23369](https://arxiv.org/pdf/2511.23369)**

> **作者:** Haochen Tian; Tianyu Li; Haochen Liu; Jiazhi Yang; Yihang Qiu; Guang Li; Junli Wang; Yinfeng Gao; Zhang Zhang; Liang Wang; Hangjun Ye; Tieniu Tan; Long Chen; Hongyang Li
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** Achieving fully autonomous driving systems requires learning rational decisions in a wide span of scenarios, including safety-critical and out-of-distribution ones. However, such cases are underrepresented in real-world corpus collected by human experts. To complement for the lack of data diversity, we introduce a novel and scalable simulation framework capable of synthesizing massive unseen states upon existing driving logs. Our pipeline utilizes advanced neural rendering with a reactive environment to generate high-fidelity multi-view observations controlled by the perturbed ego trajectory. Furthermore, we develop a pseudo-expert trajectory generation mechanism for these newly simulated states to provide action supervision. Upon the synthesized data, we find that a simple co-training strategy on both real-world and simulated samples can lead to significant improvements in both robustness and generalization for various planning methods on challenging real-world benchmarks, up to +8.6 EPDMS on navhard and +2.9 on navtest. More importantly, such policy improvement scales smoothly by increasing simulation data only, even without extra real-world data streaming in. We further reveal several crucial findings of such a sim-real learning system, which we term SimScale, including the design of pseudo-experts and the scaling properties for different policy architectures. Simulation data and code have been released at this https URL.
>
---
#### [replaced 011] Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决视觉-语言-动作模型在遮挡和跨任务迁移中的性能问题。提出TVVE框架，通过任务感知的视角选择和视觉特征路由提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2508.05186](https://arxiv.org/pdf/2508.05186)**

> **作者:** Yongjie Bai; Zhouxia Wang; Yang Liu; Kaijun Luo; Yifan Wen; Mingtong Dai; Weixing Chen; Ziliang Chen; Lingbo Liu; Guanbin Li; Liang Lin
>
> **备注:** 24 pages, 15 figures, Project page: this https URL, Code: this https URL, Accepted at CVPR 2026
>
> **摘要:** Recent vision-language-action (VLA) models for multi-task robot manipulation often rely on fixed camera setups and shared visual encoders, which limit their performance under occlusions and during cross-task transfer. To address these challenges, we propose Task-aware Virtual View Exploration (TVVE), a framework that learns to select task-relevant virtual camera viewpoints and dynamically re-render observations from a reconstructed scene representation using the selected viewpoints. To enable efficient view selection, we train an exploration policy in a pseudo-environment. In addition, we introduce a Task-aware Mixture-of-Experts (TaskMoE) visual encoder that routes visual features to task-specialized experts, mitigating interference in multi-task learning. To evaluate robustness under distribution shifts, we construct RLBench-OG, an out-of-distribution benchmark with visual perturbations and camera pose variations. Experiments on RLBench and RLBench-OG demonstrate that TVVE achieves higher success rates than strong baselines, while real-robot experiments further confirm its robustness to visual disturbances and unseen instructions. Code and visualizations are available at: this https URL.
>
---
#### [replaced 012] TwinTrack: Bridging Vision and Contact Physics for Real-Time Tracking of Unknown Objects in Contact-Rich Scenes
- **分类: cs.RO**

- **简介: 该论文属于实时目标跟踪任务，解决接触密集场景中未知动态物体的6-DoF姿态跟踪问题。通过融合视觉与接触物理信息，提出TwinTrack系统提升跟踪鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2505.22882](https://arxiv.org/pdf/2505.22882)**

> **作者:** Wen Yang; Zhixian Xie; Yiting Wang; Abhijit Tadepalli; Heni Ben Amor; Shan Lin; Wanxin Jin
>
> **备注:** Accepted by IEEE International Conference on Robotics & Automation (ICRA) 2026
>
> **摘要:** Real-time tracking of previously unseen, highly dynamic objects in contact-rich scenes, such as during dexterous in-hand manipulation, remains a major challenge. Pure vision-based approaches often fail under heavy occlusions due to frequent contact interactions and motion blur caused by abrupt impacts. We propose Twintrack, a physics-aware perception system that enables robust, real-time 6-DoF pose tracking of unknown dynamic objects in contact-rich scenes by leveraging contact physics cues. At its core, Twintrack integrates Real2Sim and Sim2Real. Real2Sim combines vision and contact physics to jointly estimate object geometry and physical properties: an initial reconstruction is obtained from vision, then refined by learning a geometry residual and simultaneously estimating physical parameters (e.g., mass, inertia, and friction) based on contact dynamics consistency. Sim2Real achieves robust pose estimation by adaptively fusing a visual tracker with predictions from the updated contact dynamics. Twintrack is implemented on a GPU-accelerated, customized MJX engine to guarantee real-time performance. We evaluate our method on two contact-rich scenarios: object falling with environmental contacts and multi-fingered in-hand manipulation. Results show that, compared to baselines, Twintrack delivers significantly more robust, accurate, and real-time tracking in these challenging settings, with tracking speeds above 20 Hz. Project page: this https URL
>
---
#### [replaced 013] DexGrasp-Zero: A Morphology-Aligned Policy for Zero-Shot Cross-Embodiment Dexterous Grasping
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，解决跨机械臂零样本抓取问题。通过构建形态对齐图网络，实现不同手型间的通用抓取策略迁移。**

- **链接: [https://arxiv.org/pdf/2603.16806](https://arxiv.org/pdf/2603.16806)**

> **作者:** Yuliang Wu; Yanhan Lin; WengKit Lao; Yuhao Lin; Yi-Lin Wei; Wei-Shi Zheng; Ancong Wu
>
> **摘要:** To meet the demands of increasingly diverse dexterous hand hardware, it is crucial to develop a policy that enables zero-shot cross-embodiment grasping without redundant re-learning. Cross-embodiment alignment is challenging due to heterogeneous hand kinematics and physical constraints. Existing approaches typically predict intermediate motion targets and retarget them to each embodiment, which may introduce errors and violate embodiment-specific limits, hindering transfer across diverse hands. To overcome these limitations, we propose DexGrasp-Zero, a policy that learns universal grasping skills from diverse embodiments, enabling zero-shot transfer to unseen hands. We first introduce a morphology-aligned graph representation that maps each hand's kinematic keypoints to anatomically grounded nodes and equips each node with tri-axial orthogonal motion primitives, enabling structural and semantic alignment across different morphologies. Relying on this graph-based representation, we design a Morphology-Aligned Graph Convolutional Network (MAGCN) to encode the graph for policy learning. MAGCN incorporates a Physical Property Injection mechanism that fuses hand-specific physical constraints into the graph features, enabling adaptive compensation for varying link lengths and actuation limits for precise and stable grasping. Our extensive simulation evaluations on the YCB dataset demonstrate that our policy, jointly trained on four heterogeneous hands (Allegro, Shadow, Schunk, Ability), achieves an 85% zero-shot success rate on unseen hardware (LEAP, Inspire), outperforming the state-of-the-art method by 59.5%. Real-world experiments further evaluate our policy on three robot platforms (LEAP, Inspire, Revo2), achieving an 82% average success rate on unseen objects.
>
---
#### [replaced 014] ReTac-ACT: A State-Gated Vision-Tactile Fusion Transformer for Precision Assembly
- **分类: cs.RO**

- **简介: 该论文属于精密装配任务，解决视觉反馈失效时的亚毫米级定位问题。通过融合视觉与触觉信息，提出ReTac-ACT模型提升装配精度。**

- **链接: [https://arxiv.org/pdf/2603.09565](https://arxiv.org/pdf/2603.09565)**

> **作者:** Minchi Ruan; LiangQing Zhou; Hongtong Li; Zongtao Wang; ZhaoMing Lu; Jianwei Zhang; Bin Fang
>
> **摘要:** Precision assembly requires sub-millimeter corrections in contact-rich "last-millimeter" regions where visual feedback fails due to occlusion from the end-effector and workpiece. We present ReTac-ACT (Reconstruction-enhanced Tactile ACT), a vision-tactile imitation learning policy that addresses this challenge through three synergistic mechanisms: (i) bidirectional cross-attention enabling reciprocal visuo-tactile feature enhancement before fusion, (ii) a proprioception-conditioned gating network that dynamically elevates tactile reliance when visual occlusion occurs, and (iii) a tactile reconstruction objective enforcing learning of manipulation-relevant contact information rather than generic visual textures. Evaluated on the standardized NIST Assembly Task Board M1 benchmark, ReTac-ACT achieves 90% peg-in-hole success, substantially outperforming vision-only and generalist baseline methods, and maintains 80% success at industrial-grade 0.1mm clearance. Ablation studies validate that each architectural component is indispensable. The ReTac-ACT codebase and a vision-tactile demonstration dataset covering various clearance levels with both visual and tactile features will be released to support reproducible research.
>
---
#### [replaced 015] Beyond Short-Horizon: VQ-Memory for Robust Long-Horizon Manipulation in Non-Markovian Simulation Benchmarks
- **分类: cs.RO**

- **简介: 该论文属于机器人长期操作任务，解决非马尔可夫环境下复杂操作问题。提出RuleSafe基准和VQ-Memory方法，提升长时序规划与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.09513](https://arxiv.org/pdf/2603.09513)**

> **作者:** Honghui Wang; Zhi Jing; Jicong Ao; Shiji Song; Xuelong Li; Gao Huang; Chenjia Bai
>
> **备注:** 9 pages
>
> **摘要:** The high cost of collecting real-robot data has made robotic simulation a scalable platform for both evaluation and data generation. Yet most existing benchmarks concentrate on simple manipulation tasks such as pick-and-place, failing to capture the non-Markovian characteristics of real-world tasks and the complexity of articulated object interactions. To address this limitation, we present RuleSafe, a new articulated manipulation benchmark built upon a scalable LLM-aided simulation framework. RuleSafe features safes with diverse unlocking mechanisms, such as key locks, password locks, and logic locks, which require different multi-stage reasoning and manipulation strategies. These LLM-generated rules produce non-Markovian and long-horizon tasks that require temporal modeling and memory-based reasoning. We further propose VQ-Memory, a compact and structured temporal representation that uses vector-quantized variational autoencoders (VQ-VAEs) to encode past proprioceptive states into discrete latent tokens. This representation filters low-level noise while preserving high-level task-phase context, providing lightweight yet robust temporal cues that are compatible with existing Vision-Language-Action models (VLA). Extensive experiments on state-of-the-art VLA models and diffusion policies show that VQ-Memory consistently improves long-horizon planning, enhances generalization to unseen configurations, and enables more efficient manipulation with reduced computational cost. Project page: this http URL
>
---
#### [replaced 016] AutoMoT: A Unified Vision-Language-Action Model with Asynchronous Mixture-of-Transformers for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种统一的视觉-语言-动作模型AutoMoT，解决端到端自动驾驶中推理与决策的协同问题，通过异步Transformer混合架构提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.14851](https://arxiv.org/pdf/2603.14851)**

> **作者:** Wenhui Huang; Songyan Zhang; Qihang Huang; Zhidong Wang; Zhiqi Mao; Collister Chua; Zhan Chen; Long Chen; Chen Lv
>
> **摘要:** Integrating vision-language models (VLMs) into end-to-end (E2E) autonomous driving (AD) systems has shown promise in improving scene understanding. However, existing integration strategies suffer from several limitations: they either struggle to resolve distribution misalignment between reasoning and action spaces, underexploit the general reasoning capabilities of pretrained VLMs, or incur substantial inference latency during action policy generation, which degrades driving performance. To address these challenges, we propose \OURS in this work, an end-to-end AD framework that unifies reasoning and action generation within a single vision-language-action (VLA) model. Our approach leverages a mixture-of-transformer (MoT) architecture with joint attention sharing, which preserves the general reasoning capabilities of pre-trained VLMs while enabling efficient fast-slow inference through asynchronous execution at different task frequencies. Extensive experiments on multiple benchmarks, under both open- and closed-loop settings, demonstrate that \OURS achieves competitive performance compared to state-of-the-art methods. We further investigate the functional boundary of pre-trained VLMs in AD, examining when AD-tailored fine-tuning is necessary. Our results show that pre-trained VLMs can achieve competitive multi-task scene understanding performance through semantic prompting alone, while fine-tuning remains essential for action-level tasks such as decision-making and trajectory planning. We refer to \href{this https URL}{Project Page} for the demonstration videos and qualitative results.
>
---
#### [replaced 017] Dynamic-ICP: Doppler-Aware Iterative Closest Point Registration for Dynamic Scenes
- **分类: cs.RO**

- **简介: 该论文属于点云配准任务，解决动态场景中ICP方法失效的问题。通过引入多普勒信息，提升动态环境下的定位精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2511.20292](https://arxiv.org/pdf/2511.20292)**

> **作者:** Dong Wang; Daniel Casado Herraez; Stefan May; Andreas Nüchter
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Reliable odometry in highly dynamic environments remains challenging when it relies on ICP-based registration: ICP assumes near-static scenes and degrades in repetitive or low-texture geometry. We introduce Dynamic-ICP, a Doppler-aware registration framework. The method (i) estimates ego motion from per-point Doppler velocity via robust regression and builds a velocity filter, (ii) clusters dynamic objects and reconstructs object-wise translational velocities from ego-compensated radial measurements, (iii) predicts dynamic points with a constant-velocity model, and (iv) aligns scans using a compact objective that combines point-to-plane geometry residual with a translation-invariant, rotation-only Doppler residual. The approach requires no external sensors or sensor-vehicle calibration and operates directly on FMCW LiDAR range and Doppler velocities. We evaluate Dynamic-ICP on three datasets-HeRCULES, HeLiPR, AevaScenes-focusing on highly dynamic scenes. Dynamic-ICP consistently improves rotational stability and translation accuracy over the state-of-the-art methods. Our approach is also simple to integrate into existing pipelines, runs in real time, and provides a lightweight solution for robust registration in dynamic environments. To encourage further research, the code is available at: this https URL.
>
---
#### [replaced 018] Swarm Self Clustering for Communication denied Environments without Global Positioning
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究无人机群在无GPS和通信限制环境下的自组织聚类问题。通过局部感知与决策，实现无需外部指令的集群形成，提升适应性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2311.17697](https://arxiv.org/pdf/2311.17697)**

> **作者:** Sweksha Jain; Rugved Katole; Leena Vachhani
>
> **备注:** 36 Pages, 15 figures, 8 tables, pre-print version
>
> **摘要:** In this work, we investigate swarm self-clustering, where robots autonomously organize into spatially coherent groups using only local sensing and decision-making, without external commands, global positioning, or inter-robot communication. Each robot forms and maintains clusters by responding to relative distances from nearby neighbors detected through onboard range sensors with limited fields of view. The method is suited for GPS-denied and communication-constrained environments and requires no prior knowledge of cluster size, number, or membership. A mechanism enables robots to alternate between consensus-based and random goal assignment based on local neighborhood size, ensuring robustness, scalability, and untraceable clustering independent of initial conditions. Extensive simulations and real-robot experiments demonstrate empirical convergence, adaptability to dynamic additions, and improved performance over local-only baselines across standard cluster quality metrics.
>
---
#### [replaced 019] LaS-Comp: Zero-shot 3D Completion with Latent-Spatial Consistency
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LaS-Comp，解决3D形状补全任务，通过两阶段设计实现零样本、类别无关的补全，提升完整性与边界一致性。**

- **链接: [https://arxiv.org/pdf/2602.18735](https://arxiv.org/pdf/2602.18735)**

> **作者:** Weilong Yan; Haipeng Li; Hao Xu; Nianjin Ye; Yihao Ai; Shuaicheng Liu; Jingyu Hu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** This paper introduces LaS-Comp, a zero-shot and category-agnostic approach that leverages the rich geometric priors of 3D foundation models to enable 3D shape completion across diverse types of partial observations. Our contributions are threefold: First, \ourname{} harnesses these powerful generative priors for completion through a complementary two-stage design: (i) an explicit replacement stage that preserves the partial observation geometry to ensure faithful completion; and (ii) an implicit refinement stage ensures seamless boundaries between the observed and synthesized regions. Second, our framework is training-free and compatible with different 3D foundation models. Third, we introduce Omni-Comp, a comprehensive benchmark combining real-world and synthetic data with diverse and challenging partial patterns, enabling a more thorough and realistic evaluation. Both quantitative and qualitative experiments demonstrate that our approach outperforms previous state-of-the-art approaches. Our code and data will be available at \href{this https URL}{LaS-Comp}.
>
---
#### [replaced 020] See, Plan, Cut: MPC-Based Autonomous Volumetric Robotic Laser Surgery with OCT Guidance
- **分类: cs.RO**

- **简介: 该论文属于自主机器人激光手术任务，旨在解决传统系统缺乏体积规划和实时反馈的问题。通过集成OCT与MPC控制，实现精准软组织切除。**

- **链接: [https://arxiv.org/pdf/2511.17777](https://arxiv.org/pdf/2511.17777)**

> **作者:** Ravi Prakash; Vincent Y. Wang; Arpit Mishra; Devi Yuliarti; Pei Zhong; Ryan P. McNabb; Patrick J. Codd; Leila J. Bridgeman
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Robotic laser systems offer the potential for sub-millimeter, non-contact, high-precision tissue resection, yet existing platforms lack volumetric planning and intraoperative feedback. We present RATS (Robot-Assisted Tissue Surgery), an intelligent opto-mechanical, optical coherence tomography (OCT)-guided robotic platform designed for autonomous volumetric soft tissue resection in surgical applications. RATS integrates macro-scale RGB-D imaging, micro-scale OCT, and a fiber-coupled surgical laser, calibrated through a novel multistage alignment pipeline that achieves OCT-to-laser calibration accuracy of 0.161+-0.031mm on tissue phantoms and ex vivo porcine tissue. A super-Gaussian laser-tissue interaction (LTI) model characterizes ablation crater morphology with an average RMSE of 0.231+-0.121mm, outperforming Gaussian baselines. A sampling-based model predictive control (MPC) framework operates directly on OCT voxel data to generate constraint-aware resection trajectories with closed-loop feedback, achieving 0.842mm RMSE and improving intersection-over-union agreement by 64.8% compared to feedforward execution. With OCT, RATS detects subsurface structures and modifies the planner's objective to preserve them, demonstrating clinical feasibility.
>
---
#### [replaced 021] CBF-RL: Safety Filtering Reinforcement Learning in Training with Control Barrier Functions
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于强化学习安全控制任务，旨在解决RL训练中安全约束不足的问题。通过引入CBF-RL框架，在训练中嵌入安全约束，提升策略安全性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.14959](https://arxiv.org/pdf/2510.14959)**

> **作者:** Lizhi Yang; Blake Werner; Massimiliano de Sa; Aaron D. Ames
>
> **备注:** To appear at ICRA 2026; sample code for the navigation example with CBF-RL reward core construction can be found at this https URL
>
> **摘要:** Reinforcement learning (RL), while powerful and expressive, can often prioritize performance at the expense of safety. Yet safety violations can lead to catastrophic outcomes in real-world deployments. Control Barrier Functions (CBFs) offer a principled method to enforce dynamic safety -- traditionally deployed online via safety filters. While the result is safe behavior, the fact that the RL policy does not have knowledge of the CBF can lead to conservative behaviors. This paper proposes CBF-RL, a framework for generating safe behaviors with RL by enforcing CBFs in training. CBF-RL has two key attributes: (1) minimally modifying a nominal RL policy to encode safety constraints via a CBF term, (2) and safety filtering of the policy rollouts in training. Theoretically, we prove that continuous-time safety filters can be deployed via closed-form expressions on discrete-time roll-outs. Practically, we demonstrate that CBF-RL internalizes the safety constraints in the learned policy -- both enforcing safer actions and biasing towards safer rewards -- enabling safe deployment without the need for an online safety filter. We validate our framework through ablation studies on navigation tasks and on the Unitree G1 humanoid robot, where CBF-RL enables safer exploration, faster convergence, and robust performance under uncertainty, enabling the humanoid robot to avoid obstacles and climb stairs safely in real-world settings without a runtime safety filter.
>
---
#### [replaced 022] IRIS-SLAM: Unified Geo-Instance Representations for Robust Semantic Localization and Mapping
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出IRIS-SLAM，解决语义定位与建图中的几何-实例统一表示问题，提升地图一致性与回环检测可靠性。**

- **链接: [https://arxiv.org/pdf/2602.18709](https://arxiv.org/pdf/2602.18709)**

> **作者:** Tingyang Xiao; Liu Liu; Wei Feng; Zhengyu Zou; Xiaolin Zhou; Wei Sui; Hao Li; Dingwen Zhang; Zhizhong Su
>
> **备注:** The reason for this withdrawal is that the current version was submitted without the final review and formal authorization of all co-authors. To ensure the academic consensus and integrity of our research group, we have decided to withdraw this submission from the repository
>
> **摘要:** Geometry foundation models have significantly advanced dense geometric SLAM, yet existing systems often lack deep semantic understanding and robust loop closure capabilities. Meanwhile, contemporary semantic mapping approaches are frequently hindered by decoupled architectures and fragile data association. We propose IRIS-SLAM, a novel RGB semantic SLAM system that leverages unified geometric-instance representations derived from an instance-extended foundation model. By extending a geometry foundation model to concurrently predict dense geometry and cross-view consistent instance embeddings, we enable a semantic-synergized association mechanism and instance-guided loop closure detection. Our approach effectively utilizes viewpoint-agnostic semantic anchors to bridge the gap between geometric reconstruction and open-vocabulary mapping. Experimental results demonstrate that IRIS-SLAM significantly outperforms state-of-the-art methods, particularly in map consistency and wide-baseline loop closure reliability.
>
---
#### [replaced 023] MG-Grasp: Metric-Scale Geometric 6-DoF Grasping Framework with Sparse RGB Observations
- **分类: cs.RO**

- **简介: 该论文属于6-DoF抓取任务，解决RGB图像下几何表示不准确的问题。提出MG-Grasp框架，通过多视角重建稠密点云，实现可靠抓取。**

- **链接: [https://arxiv.org/pdf/2603.16270](https://arxiv.org/pdf/2603.16270)**

> **作者:** Kangxu Wang; Siang Chen; Chenxing Jiang; Shaojie Shen; Yixiang Dai; Guijin Wang
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Single-view RGB-D grasp detection remains a common choice in 6-DoF robotic grasping systems, which typically requires a depth sensor. While RGB-only 6-DoF grasp methods has been studied recently, their inaccurate geometric representation is not directly suitable for physically reliable robotic manipulation, thereby hindering reliable grasp generation. To address these limitations, we propose MG-Grasp, a novel depth-free 6-DoF grasping framework that achieves high-quality object grasping. Leveraging two-view 3D foundation model with camera intrinsic/extrinsic, our method reconstructs metric-scale and multi-view consistent dense point clouds from sparse RGB images and generates stable 6-DoF grasp. Experiments on GraspNet-1Billion dataset and real world demonstrate that MG-Grasp achieves state-of-the-art (SOTA) grasp performance among RGB-based 6-DoF grasping methods.
>
---
#### [replaced 024] SAATT Nav: a Socially Aware Autonomous Transparent Transportation Navigation Framework for Wheelchairs
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主导航任务，旨在解决轮椅在社交环境中的安全与信任问题。通过引入社会意识和透明决策机制，提升导航系统的社交适应性和用户信任度。**

- **链接: [https://arxiv.org/pdf/2603.13698](https://arxiv.org/pdf/2603.13698)**

> **作者:** Yutong Zhang; Shaiv Y. Mehra; Bradley S. Duerstock; Juan P. Wachs
>
> **备注:** 8 pages, 4 figures, 2 tables, 1 algorithm. Submitted to IROS 2026
>
> **摘要:** While powered wheelchairs reduce physical fatigue as opposed to manual wheelchairs for individuals with mobility impairment, they demand high cognitive workload due to information processing, decision making and motor coordination. Current autonomous systems lack social awareness in navigation and transparency in decision-making, leading to decreased perceived safety and trust from the user and others in context. This work proposes Socially Aware Autonomous Transparent Transportation (SAATT) Navigation framework for wheelchairs as a potential solution. By implementing a Large Language Model (LLM) informed of user intent and capable of predicting other peoples' intent as a decision-maker for its local controller, it is able to detect and navigate social situations, such as passing pedestrians or a pair conversing. Furthermore, the LLM textually communicates its reasoning at each waypoint for transparency. In this experiment, it is compared against a standard global planner, a representative competing social navigation model, and an Ablation study in three simulated environments varied by social levels in eight metrics categorized under Safety, Social Compliance, Efficiency, and Comfort. Overall, SAATT Nav outperforms in most social situations and equivalently or only slightly worse in the remaining metrics, demonstrating the potential of a socially aware and transparent autonomous navigation system to assist wheelchair users.
>
---
#### [replaced 025] S-VAM: Shortcut Video-Action Model by Self-Distilling Geometric and Semantic Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出S-VAM，解决视频动作模型实时与高精度不足的问题。通过单次前向传播和自蒸馏策略，提升机器人操作效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.16195](https://arxiv.org/pdf/2603.16195)**

> **作者:** Haodong Yan; Zhide Zhong; Jiaguan Zhu; Junjie He; Weilin Yuan; Wenxuan Song; Xin Gong; Yingjie Cai; Guanyi Zhao; Xu Yan; Bingbing Liu; Ying-Cong Chen; Haoang Li
>
> **摘要:** Video action models (VAMs) have emerged as a promising paradigm for robot learning, owing to their powerful visual foresight for complex manipulation tasks. However, current VAMs, typically relying on either slow multi-step video generation or noisy one-step feature extraction, cannot simultaneously guarantee real-time inference and high-fidelity foresight. To address this limitation, we propose S-VAM, a shortcut video-action model that foresees coherent geometric and semantic representations via a single forward pass. Serving as a stable blueprint, these foreseen representations significantly simplify the action prediction. To enable this efficient shortcut, we introduce a novel self-distillation strategy that condenses structured generative priors of multi-step denoising into one-step inference. Specifically, vision foundation model (VFM) representations extracted from the diffusion model's own multi-step generated videos provide teacher targets. Lightweight decouplers, as students, learn to directly map noisy one-step features to these targets. Extensive experiments in simulation and the real world demonstrate that our S-VAM outperforms state-of-the-art methods, enabling efficient and precise manipulation in complex environments. Our project page is this https URL
>
---
#### [replaced 026] Context-Nav: Context-Driven Exploration and Viewpoint-Aware 3D Spatial Reasoning for Instance Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于文本目标实例导航任务，解决在同类物体中精确定位目标的问题。提出Context-Nav方法，通过上下文引导探索和3D空间验证实现精准导航。**

- **链接: [https://arxiv.org/pdf/2603.09506](https://arxiv.org/pdf/2603.09506)**

> **作者:** Won Shik Jang; Ue-Hwan Kim
>
> **备注:** Accepted to CVPR 2026. Code is available at this https URL
>
> **摘要:** Text-goal instance navigation (TGIN) asks an agent to resolve a single, free-form description into actions that reach the correct object instance among same-category distractors. We present \textit{Context-Nav}, which elevates long, contextual captions from a local matching cue to a global exploration prior and verifies candidates through 3D spatial reasoning. First, we compute dense text-image alignments for a value map that ranks frontiers -- guiding exploration toward regions consistent with the entire description rather than early detections. Second, upon observing a candidate, we perform a viewpoint-aware relation check: the agent samples plausible observer poses, aligns local frames, and accepts a target only if the spatial relations can be satisfied from at least one viewpoint. The pipeline requires no task-specific training or fine-tuning; we attain state-of-the-art performance on InstanceNav and CoIN-Bench. Ablations show that (i) encoding full captions into the value map avoids wasted motion and (ii) explicit, viewpoint-aware 3D verification prevents semantically plausible but incorrect stops. This suggests that geometry-grounded spatial reasoning is a scalable alternative to heavy policy training or human-in-the-loop interaction for fine-grained instance disambiguation in cluttered 3D scenes.
>
---
#### [replaced 027] Aion: Towards Hierarchical 4D Scene Graphs with Temporal Flow Dynamics
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主导航任务，解决动态环境中时空表示问题。提出Aion框架，将时间流动态嵌入层次化3D场景图，提升导航预测与规划效果。**

- **链接: [https://arxiv.org/pdf/2512.11903](https://arxiv.org/pdf/2512.11903)**

> **作者:** Iacopo Catalano; Eduardo Montijano; Javier Civera; Julio A. Placed; Jorge Pena-Queralta
>
> **备注:** Accepted at ICRA 2026, 8 pages
>
> **摘要:** Autonomous navigation in dynamic environments requires spatial representations that capture both semantic structure and temporal evolution. 3D Scene Graphs (3DSGs) provide hierarchical multi-resolution abstractions that encode geometry and semantics, but existing extensions toward dynamics largely focus on individual objects or agents. In parallel, Maps of Dynamics (MoDs) model typical motion patterns and temporal regularities, yet are usually tied to grid-based discretizations that lack semantic awareness and do not scale well to large environments. In this paper we introduce Aion, a framework that embeds temporal flow dynamics directly within a hierarchical 3DSG, effectively incorporating the temporal dimension. Aion employs a graph-based sparse MoD representation to capture motion flows over arbitrary time intervals and attaches them to navigational nodes in the scene graph, yielding more interpretable and scalable predictions that improve planning and interaction in complex dynamic environments. We provide the code at this https URL
>
---
#### [replaced 028] AgriChrono: A Multi-modal Dataset Capturing Crop Growth and Lighting Variability with a Field Robot
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文提出AgriChrono数据集，用于农业机器人和3D重建任务，解决真实农田环境下的动态场景建模问题。通过多传感器采集数据，提升AI模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.18694](https://arxiv.org/pdf/2508.18694)**

> **作者:** Jaehwan Jeong; Tuan-Anh Vu; Mohammad Jony; Shahab Ahmad; Md. Mukhlesur Rahman; Sangpil Kim; M. Khalid Jawed
>
> **备注:** Keywords: Agricultural Robotics, In-the-wild Dataset, 3D Reconstruction
>
> **摘要:** Advances in AI and Robotics have accelerated significant initiatives in agriculture, particularly in the areas of robot navigation and 3D digital twin creation. A significant bottleneck impeding this progress is the critical lack of "in-the-wild" datasets that capture the full complexities of real farmland, including non-rigid motion from wind, drastic illumination variance, and morphological changes resulting from growth. This data gap fundamentally limits research on robust AI models for autonomous field navigation and scene-level dynamic 3D reconstruction. In this paper, we present AgriChrono, a modular robotic data collection platform and multi-modal dataset designed to capture these dynamic farmland conditions. Our platform integrates multiple sensors, enabling remote, time-synchronized acquisition of RGB, Depth, LiDAR, IMU, and Pose data for efficient and repeatable long-term data collection in real-world agricultural environments. We successfully collected 18TB of data over one month, documenting the entire growth cycle of Canola under diverse illumination conditions. We benchmark state-of-the-art 3D reconstruction methods on AgriChrono, revealing the profound challenge of reconstructing high-fidelity, dynamic non-rigid scenes in such farmland settings. This benchmark validates AgriChrono as a critical asset for advancing model generalization, and its public release is expected to significantly accelerate research and development in precision agriculture. The code and dataset are publicly available at: this https URL
>
---
#### [replaced 029] PACE: Physics Augmentation for Coordinated End-to-end Reinforcement Learning toward Versatile Humanoid Table Tennis
- **分类: cs.RO**

- **简介: 该论文属于人形机器人乒乓球任务，旨在解决端到端控制策略在快速感知和运动协调上的难题。通过引入物理增强的预测信号和奖励机制，提升了策略性能。**

- **链接: [https://arxiv.org/pdf/2509.21690](https://arxiv.org/pdf/2509.21690)**

> **作者:** Muqun Hu; Wenxi Chen; Wenjing Li; Falak Mandali; Zijian He; Renhong Zhang; Praveen Krisna; Katherine Christian; Leo Benaharon; Dizhi Ma; Karthik Ramani; Yan Gu
>
> **摘要:** Humanoid table tennis (TT) demands rapid perception, proactive whole-body motion, and agile footwork under strict timing--capabilities that remain difficult for end-to-end control policies. We propose a reinforcement learning (RL) framework that maps ball-position observations directly to whole-body joint commands for both arm striking and leg locomotion, strengthened by predictive signals and dense, physics-guided rewards. A lightweight learned predictor, fed with recent ball positions, estimates future ball states and augments the policy's observations for proactive decision-making. During training, a physics-based predictor supplies precise future states to construct dense, informative rewards that lead to effective exploration. The resulting policy attains strong performance across varied serve ranges (hit rate$\geq$96% and success rate$\geq$92%) in simulations. Ablation studies confirm that both the learned predictor and the predictive reward design are critical for end-to-end learning. Deployed zero-shot on a physical Booster T1 humanoid with 23 revolute joints, the policy produces coordinated lateral and forward-backward footwork with accurate, fast returns, suggesting a practical path toward versatile, competitive humanoid TT. We have open-sourced our RL training code at: this https URL
>
---
#### [replaced 030] SO-Bench: A Structural Output Evaluation of Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出SO-Bench基准，用于评估多模态大模型的结构化输出能力，解决其在视觉输入下符合预定义数据模式的问题。**

- **链接: [https://arxiv.org/pdf/2511.21750](https://arxiv.org/pdf/2511.21750)**

> **作者:** Di Feng; Kaixin Ma; Feng Nan; Haofeng Chen; Bohan Zhai; David Griffiths; Mingfei Gao; Zhe Gan; Eshan Verma; Yinfei Yang; Zhifeng Chen; Afshin Dehghan
>
> **备注:** v3 preprint. Added the link to the public benchmark
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly deployed in real-world, agentic settings where outputs must not only be correct, but also conform to predefined data schemas. Despite recent progress in structured generation in textual domain, there is still no benchmark that systematically evaluates schema-grounded information extraction and reasoning over visual inputs. In this work, we conduct a comprehensive study of visual structural output capabilities for MLLMs with our carefully designed SO-Bench benchmark. Covering four visual domains, including UI screens, natural images, documents, and charts, SO-Bench is built from over 6.5K diverse JSON schemas and 1.8K curated image-schema pairs with human-verified quality. Benchmarking experiments on open-sourced and frontier proprietary models reveal persistent gaps in predicting accurate, schema compliant outputs, highlighting the need for better multimodal structured reasoning. Beyond benchmarking, we further conduct training experiments to largely improve the model's structured output capability. We make the benchmark and evaluation publicly available at this https URL
>
---
#### [replaced 031] Beware Untrusted Simulators -- Reward-Free Backdoor Attacks in Reinforcement Learning
- **分类: cs.CR; cs.LG; cs.RO**

- **简介: 该论文属于强化学习安全任务，解决模拟器被攻击问题。提出新攻击方法Daze，在不修改奖励情况下植入动作级后门，实现隐蔽控制。**

- **链接: [https://arxiv.org/pdf/2602.05089](https://arxiv.org/pdf/2602.05089)**

> **作者:** Ethan Rathbun; Wo Wei Lin; Alina Oprea; Christopher Amato
>
> **备注:** 10 pages main body, ICLR 2026
>
> **摘要:** Simulated environments are a key piece in the success of Reinforcement Learning (RL), allowing practitioners and researchers to train decision making agents without running expensive experiments on real hardware. Simulators remain a security blind spot, however, enabling adversarial developers to alter the dynamics of their released simulators for malicious purposes. Therefore, in this work we highlight a novel threat, demonstrating how simulator dynamics can be exploited to stealthily implant action-level backdoors into RL agents. The backdoor then allows an adversary to reliably activate targeted actions in an agent upon observing a predefined ``trigger'', leading to potentially dangerous consequences. Traditional backdoor attacks are limited in their strong threat models, assuming the adversary has near full control over an agent's training pipeline, enabling them to both alter and observe agent's rewards. As these assumptions are infeasible to implement within a simulator, we propose a new attack ``Daze'' which is able to reliably and stealthily implant backdoors into RL agents trained for real world tasks without altering or even observing their rewards. We provide formal proof of Daze's effectiveness in guaranteeing attack success across general RL tasks along with extensive empirical evaluations on both discrete and continuous action space domains. We additionally provide the first example of RL backdoor attacks transferring to real, robotic hardware. These developments motivate further research into securing all components of the RL training pipeline to prevent malicious attacks.
>
---
#### [replaced 032] Lyapunov Constrained Soft Actor-Critic (LC-SAC) using Koopman Operator Theory for Quadrotor Trajectory Tracking
- **分类: eess.SY; cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决安全关键系统中策略稳定性问题。通过结合Lyapunov函数与Koopman理论，提出LC-SAC算法，提升四旋翼轨迹跟踪的稳定性与收敛性。**

- **链接: [https://arxiv.org/pdf/2602.04132](https://arxiv.org/pdf/2602.04132)**

> **作者:** Dhruv S. Kushwaha; Zoleikha A. Biron
>
> **备注:** 11 pages, 7 Figures, submitted to IEEE RA-L
>
> **摘要:** Reinforcement Learning (RL) has achieved remarkable success in solving complex sequential decision-making problems. However, its application to safety-critical physical systems remains constrained by the lack of stability guarantees. Standard RL algorithms prioritize reward maximization, often yielding policies that may induce oscillations or unbounded state divergence. There has been significant work in incorporating Lyapunov-based stability guarantees in RL algorithms with key challenges being selecting a candidate Lyapunov function, computational complexity by using excessive function approximators and conservative policies by incorporating stability criterion in the learning process. In this work we propose a novel Lyapunov-constrained Soft Actor-Critic (LC-SAC) algorithm using Koopman operator theory. We propose use of extended dynamic mode decomposition (EDMD) to produce a linear approximation of the system and use this approximation to derive a closed form solution for candidate Lyapunov function. This derived Lyapunov function is incorporated in the SAC algorithm to further provide guarantees for a policy that stabilizes the nonlinear system. The results are evaluated trajectory tracking of a 2D Quadrotor environment based on safe-control-gym. The proposed algorithm shows training convergence and decaying violations for Lyapunov stability criterion compared to baseline vanilla SAC algorithm. GitHub Repository: this https URL
>
---
#### [replaced 033] Safety Case Patterns for VLA-based driving systems: Insights from SimLingo
- **分类: cs.RO; cs.SE**

- **简介: 该论文属于自动驾驶安全领域，旨在解决VLA系统潜在的安全风险。提出RAISE方法，设计安全案例以保障系统安全性和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.16013](https://arxiv.org/pdf/2603.16013)**

> **作者:** Gerhard Yu; Fuyuki Ishikawa; Oluwafemi Odu; Alvine Boaye Belle
>
> **摘要:** Vision-Language-Action (VLA)-based driving systems represent a significant paradigm shift in autonomous driving since, by combining traffic scene understanding, linguistic interpretation, and action generation, these systems enable more flexible, adaptive, and instruction-responsive driving behaviors. However, despite their growing adoption and potential to support socially responsible autonomous driving as well as understanding high-level human instructions, VLA-based driving systems may exhibit new types of hazardous behaviors. For instance, the integration of open-ended natural language inputs (e.g., user or navigation instructions) into the multimodal control loop, may lead to unpredictable and unsafe behaviors that could endanger vehicle occupants and pedestrians. Hence, assuring the safety of these systems is crucial to help build trust in their operations. To support this, we propose a novel safety case design approach called RAISE. Our approach introduces novel patterns tailored to instruction-based driving systems such as VLA-based driving systems, an extension of Hazard Analysis and Risk Assessment (HARA) detailing safe scenarios and their outcomes, and a design technique to create the safety cases of VLA-based driving systems. A case study on SimLingo illustrates how our approach can be used to construct rigorous, evidence-based safety claims for this emerging class of autonomous driving systems.
>
---
#### [replaced 034] ViSA: Visited-State Augmentation for Generalized Goal-Space Contrastive Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于强化学习领域，解决GCRL中目标空间泛化不足的问题。提出ViSA方法，通过数据增强提升难达目标的值函数估计效果。**

- **链接: [https://arxiv.org/pdf/2603.14887](https://arxiv.org/pdf/2603.14887)**

> **作者:** Issa Nakamura; Tomoya Yamanokuchi; Yuki Kadokawa; Jia Qu; Shun Otsub; Ken Miyamoto; Shotaro Miwa; Takamitsu Matsubara
>
> **备注:** 8 pages, 7 figures, under Review
>
> **摘要:** Goal-Conditioned Reinforcement Learning (GCRL) is a framework for learning a policy that can reach arbitrarily given goals. In particular, Contrastive Reinforcement Learning (CRL) provides a framework for policy updates using an approximation of the value function estimated via contrastive learning, achieving higher sample efficiency compared to conventional methods. However, since CRL treats the visited state as a pseudo-goal during learning, it can accurately estimate the value function only for limited goals. To address this issue, we propose a novel data augmentation approach for CRL called ViSA (Visited-State Augmentation). ViSA consists of two components: 1) generating augmented state samples, with the aim of augmenting hard-to-visit state samples during on-policy exploration, and 2) learning consistent embedding space, which uses an augmented state as auxiliary information to regularize the embedding space by reformulating the objective function of the embedding space based on mutual information. We evaluate ViSA in simulation and real-world robotic tasks and show improved goal-space generalization, which permits accurate value estimation for hard-to-visit goals. Further details can be found on the project page: this https URL
>
---
#### [replaced 035] Grounding Robot Generalization in Training Data via Retrieval-Augmented VLMs
- **分类: cs.RO**

- **简介: 该论文属于机器人泛化评估任务，旨在解决如何准确衡量策略在新场景中的泛化能力。提出RADAR框架，通过检索和VLM分析比较测试任务与训练数据，判断所需泛化类型。**

- **链接: [https://arxiv.org/pdf/2603.11426](https://arxiv.org/pdf/2603.11426)**

> **作者:** Jensen Gao; Dorsa Sadigh; Sandy Huang; Dhruv Shah
>
> **备注:** 12 pages
>
> **摘要:** Recent work on robot manipulation has advanced policy generalization to novel scenarios. However, it is often difficult to characterize how different evaluation settings actually represent generalization from the training distribution of a given policy. To work towards more precise evaluation of generalization in robotics, we propose RADAR, a scalable framework for directly comparing test-time evaluation tasks to policy training data, to determine what form of policy generalization is required. RADAR consists of a two-stage pipeline: first, retrieval using generalist policy embeddings identifies which training examples are relevant for a given evaluation task. Next, vision-language models (VLMs) analyze the evaluation task against the retrieved data, outputting interpretable analysis on how they compare along a variety of axes, and an overall classification of what type of policy generalization is required. Through controlled experiments, we demonstrate that VLMs are effective at analyzing data for generalization, and that our retrieval step effectively identifies examples needed to make accurate classifications with respect to the training data. Furthermore, we scale RADAR to large-scale datasets, where we observe agreement with human-defined benchmark conditions from prior work. We provide demonstrations at this http URL.
>
---
#### [replaced 036] Bundle Adjustment in the Eager Mode
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，解决传统BA库与深度学习框架不兼容的问题，提出一种与PyTorch无缝集成的高效GPU版BA方法。**

- **链接: [https://arxiv.org/pdf/2409.12190](https://arxiv.org/pdf/2409.12190)**

> **作者:** Zitong Zhan; Huan Xu; Zihang Fang; Xinpeng Wei; Yaoyu Hu; Chen Wang
>
> **摘要:** Bundle adjustment (BA) is a critical technique in various robotic applications such as simultaneous localization and mapping (SLAM), augmented reality (AR), and photogrammetry. BA optimizes parameters such as camera poses and 3D landmarks to align them with observations. With the growing importance of deep learning in perception systems, there is an increasing need to integrate BA with deep learning frameworks for enhanced reliability and performance. However, widely-used C++-based BA libraries, such as GTSAM, g$^2$o, and Ceres Solver, lack native integration with modern deep learning libraries like PyTorch. This limitation affects their flexibility, ease of debugging, and overall implementation efficiency. To address this gap, we introduce an eager-mode BA library seamlessly integrated with PyTorch with high efficiency. Our approach includes a sparsity-aware auto-differentiation design and GPU-accelerated sparse operations designed for 2nd-order optimization. Our eager-mode BA on GPU demonstrates substantial runtime efficiency, achieving an average speedup of 18.5$\times$, 22$\times$, and 23$\times$ across all benchmarks compared to GTSAM, g$^2$o, and Ceres, respectively.
>
---
#### [replaced 037] Pretrained Vision-Language-Action Models are Surprisingly Resistant to Forgetting in Continual Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人策略学习任务，解决持续学习中的遗忘问题。研究发现预训练的视觉-语言-动作模型对遗忘具有强抵抗力，通过简单经验回放即可有效保持旧技能。**

- **链接: [https://arxiv.org/pdf/2603.03818](https://arxiv.org/pdf/2603.03818)**

> **作者:** Huihan Liu; Changyeon Kim; Bo Liu; Minghuan Liu; Yuke Zhu
>
> **备注:** Project website: this https URL
>
> **摘要:** Continual learning is a long-standing challenge in robot policy learning, where a policy must acquire new skills over time without catastrophically forgetting previously learned ones. While prior work has extensively studied continual learning in relatively small behavior cloning (BC) policy models trained from scratch, its behavior in modern large-scale pretrained Vision-Language-Action (VLA) models remains underexplored. In this work, we found that pretrained VLAs are remarkably resistant to forgetting compared with smaller policy models trained from scratch. Simple Experience Replay (ER) works surprisingly well on VLAs, sometimes achieving zero forgetting even with a small replay data size. Our analysis reveals that pretraining plays a critical role in downstream continual learning performance: large pretrained models mitigate forgetting with a small replay buffer size while maintaining strong forward learning capabilities. Furthermore, we found that VLAs can retain relevant knowledge from prior tasks despite performance degradation during learning new tasks. This knowledge retention enables rapid recovery of seemingly forgotten skills through finetuning. Together, these insights imply that large-scale pretraining fundamentally changes the dynamics of continual learning, enabling models to continually acquire new skills over time with simple replay. Code and more information can be found at this https URL
>
---
#### [replaced 038] MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于离线强化学习任务，解决动态不匹配下的策略学习问题。提出MOBODY算法，通过学习目标动态来提升策略探索能力。**

- **链接: [https://arxiv.org/pdf/2506.08460](https://arxiv.org/pdf/2506.08460)**

> **作者:** Yihong Guo; Yu Yang; Pan Xu; Anqi Liu
>
> **备注:** Published at ICLR 2026
>
> **摘要:** We study off-dynamics offline reinforcement learning, where the goal is to learn a policy from offline source and limited target datasets with mismatched dynamics. Existing methods either penalize the reward or discard source transitions occurring in parts of the transition space with high dynamics shift. As a result, they optimize the policy using data from low-shift regions, limiting exploration of high-reward states in the target domain that do not fall within these regions. Consequently, such methods often fail when the dynamics shift is significant or the optimal trajectories lie outside the low-shift regions. To overcome this limitation, we propose MOBODY, a Model-Based Off-Dynamics Offline RL algorithm that optimizes a policy using learned target dynamics transitions to explore the target domain, rather than only being trained with the low dynamics-shift transitions. For the dynamics learning, built on the observation that achieving the same next state requires taking different actions in different domains, MOBODY employs separate action encoders for each domain to encode different actions to the shared latent space while sharing a unified representation of states and a common transition function. We further introduce a target Q-weighted behavior cloning loss in policy optimization to avoid out-of-distribution actions, which push the policy toward actions with high target-domain Q-values, rather than high source domain Q-values or uniformly imitating all actions in the offline dataset. We evaluate MOBODY on a wide range of MuJoCo and Adroit benchmarks, demonstrating that it outperforms state-of-the-art off-dynamics RL baselines as well as policy learning methods based on different dynamics learning baselines, with especially pronounced improvements in challenging scenarios where existing methods struggle.
>
---
#### [replaced 039] Mimic Intent, Not Just Trajectories
- **分类: cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决环境变化适应和技能迁移问题。通过分离行为意图与执行细节，提出MINT方法提升泛化能力和效率。**

- **链接: [https://arxiv.org/pdf/2602.08602](https://arxiv.org/pdf/2602.08602)**

> **作者:** Renming Huang; Chendong Zeng; Wenjing Tang; Jintian Cai; Cewu Lu; Panpan Cai
>
> **备注:** Under review
>
> **摘要:** While imitation learning (IL) has achieved impressive success in dexterous manipulation through generative modeling and pretraining, state-of-the-art approaches like Vision-Language-Action (VLA) models still struggle with adaptation to environmental changes and skill transfer. We argue this stems from mimicking raw trajectories without understanding the underlying intent. To address this, we propose explicitly disentangling behavior intent from execution details in end-2-end IL: Mimic Intent, Not just Trajectories(MINT). We achieve this via multi-scale frequency-space tokenization, which enforces a spectral decomposition of action chunk representation. We learn action tokens with a multi-scale coarse-to-fine structure, and force the coarsest token to capture low-frequency global structure and finer tokens to encode high-frequency details. This yields an abstract Intent token that facilitates planning and transfer, and multi-scale Execution tokens that enable precise adaptation to environmental dynamics. Building on this hierarchy, our policy generates trajectories through next-scale autoregression, performing progressive intent-to-execution reasoning, thus boosting learning efficiency and generalization. Crucially, this disentanglement enables one-shot transfer of skills, by simply injecting the Intent token from a demonstration into the autoregressive generation process. Experiments on several manipulation benchmarks and on a real robot demonstrate state-of-the-art success rates, superior inference efficiency, robust generalization against disturbances, and effective one-shot transfer.
>
---
#### [replaced 040] Echo Planning for Autonomous Driving: From Current Observations to Future Trajectories and Back
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EchoP框架，解决自动驾驶中轨迹预测与场景动态不一致的问题。通过CFC循环机制，提升轨迹预测的可靠性与一致性。**

- **链接: [https://arxiv.org/pdf/2505.18945](https://arxiv.org/pdf/2505.18945)**

> **作者:** Jintao Sun; Hu Zhang; Gangyi Ding; Zhedong Zheng
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** Modern end-to-end autonomous driving systems suffer from a critical limitation: their planners lack mechanisms to enforce temporal consistency between predicted trajectories and evolving scene dynamics. This absence of self-supervision allows early prediction errors to compound catastrophically over time. We introduce Echo Planning (EchoP), a new self-correcting framework that establishes an end-to-end Current - Future - Current (CFC) cycle to harmonize trajectory prediction with scene coherence. Our key insight is that plausible future trajectories should be bi-directionally consistent, i.e., not only generated from current observations but also capable of reconstructing them. The CFC mechanism first predicts future trajectories from the Bird's-Eye-View (BEV) scene representation, then inversely maps these trajectories back to estimate the current BEV state. By enforcing consistency between the original and reconstructed BEV representations through a cycle loss, the framework intrinsically penalizes physically implausible or misaligned trajectories. Experiments on nuScenes show that the proposed method yields competitive performance, reducing L2 error (Avg) by -0.04 m and collision rate by -0.12% compared to one-shot planners. Moreover, EchoP seamlessly extends to closed-loop evaluation, i.e., Bench2Drive, attaining a 26.54% success rate. Notably, EchoP requires no additional supervision: the CFC cycle acts as an inductive bias that stabilizes long-horizon planning. Overall, EchoP offers a simple, deployable pathway to improve reliability in safety-critical autonomous driving.
>
---
