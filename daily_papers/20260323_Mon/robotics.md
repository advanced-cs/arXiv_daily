# 机器人 cs.RO

- **最新发布 38 篇**

- **更新 26 篇**

## 最新发布

#### [new 001] Evolving Embodied Intelligence: Graph Neural Network--Driven Co-Design of Morphology and Control in Soft Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于软体机器人设计任务，旨在解决形态与控制协同优化难题。通过图神经网络实现形态感知的控制策略，提升适应性与性能。**

- **链接: [https://arxiv.org/pdf/2603.19582](https://arxiv.org/pdf/2603.19582)**

> **作者:** Jianqiang Wang; Shuaiqun Pan; Alvaro Serra-Gomez; Xiaohan Wei; Yue Xie
>
> **摘要:** The intelligent behavior of robots does not emerge solely from control systems, but from the tight coupling between body and brain, a principle known as embodied intelligence. Designing soft robots that leverage this interaction remains a significant challenge, particularly when morphology and control require simultaneous optimization. A significant obstacle in this co-design process is that morphological evolution can disrupt learned control strategies, making it difficult to reuse or adapt existing knowledge. We address this by develop a Graph Neural Network-based approach for the co-design of morphology and controller. Each robot is represented as a graph, with a graph attention network (GAT) encoding node features and a pooled representation passed through a multilayer perceptron (MLP) head to produce actuator commands or value estimates. During evolution, inheritance follows a topology-consistent mapping: shared GAT layers are reused, MLP hidden layers are transferred intact, matched actuator outputs are copied, and unmatched ones are randomly initialized and fine-tuned. This morphology-aware policy class lets the controller adapt when the body mutates. On the benchmark, our GAT-based approach achieves higher final fitness and stronger adaptability to morphological variations compared to traditional MLP-only co-design methods. These results indicate that graph-structured policies provide a more effective interface between evolving morphologies and control for embodied intelligence.
>
---
#### [new 002] IndoorR2X: Indoor Robot-to-Everything Coordination with LLM-Driven Planning
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出IndoorR2X，解决多机器人与室内物联网设备的协同问题，通过LLM驱动的规划提升场景理解与任务效率。**

- **链接: [https://arxiv.org/pdf/2603.20182](https://arxiv.org/pdf/2603.20182)**

> **作者:** Fan Yang; Soumya Teotia; Shaunak A. Mehta; Prajit KrisshnaKumar; Quanting Xie; Jun Liu; Yueqi Song; Li Wenkai; Atsunori Moteki; Kanji Uchino; Yonatan Bisk
>
> **摘要:** Although robot-to-robot (R2R) communication improves indoor scene understanding beyond what a single robot can achieve, R2R alone cannot overcome partial observability without substantial exploration overhead or scaling team size. In contrast, many indoor environments already include low-cost Internet of Things (IoT) sensors (e.g., cameras) that provide persistent, building-wide context beyond onboard perception. We therefore introduce IndoorR2X, the first benchmark and simulation framework for Large Language Model (LLM)-driven multi-robot task planning with Robot-to-Everything (R2X) perception and communication in indoor environments. IndoorR2X integrates observations from mobile robots and static IoT devices to construct a global semantic state that supports scalable scene understanding, reduces redundant exploration, and enables high-level coordination through LLM-based planning. IndoorR2X provides configurable simulation environments, sensor layouts, robot teams, and task suites to systematically evaluate high-level semantic coordination strategies. Extensive experiments across diverse settings demonstrate that IoT-augmented world modeling improves multi-robot efficiency and reliability, and we highlight key insights and failure modes for advancing LLM-based collaboration between robot teams and indoor IoT sensors.
>
---
#### [new 003] Zero Shot Deformation Reconstruction for Soft Robots Using a Flexible Sensor Array and Cage Based 3D Gaussian Modeling
- **分类: cs.RO**

- **简介: 该论文属于软体机器人变形重建任务，解决无需视觉监督的零样本变形恢复问题。通过柔性传感器阵列和基于笼子的3D高斯模型实现无相机的实时变形重建。**

- **链接: [https://arxiv.org/pdf/2603.19543](https://arxiv.org/pdf/2603.19543)**

> **作者:** Linrui Shou; Zilang Chen; Wenjia Xu; Yiyue Luo; Tingyu Cheng
>
> **摘要:** We present a zero-shot deformation reconstruction framework for soft robots that operates without any visual supervision at inference time. In this work, zero-shot deformation reconstruction is defined as the ability to infer object-wide deformations on previously unseen soft robots without collecting object-specific deformation data or performing any retraining during deployment. Our method assumes access to a static geometric proxy of the undeformed object, which can be obtained from a STL model. During operation, the system relies exclusively on tactile sensing, enabling camera-free deformation inference. The proposed framework integrates a flexible piezoresistive sensor array with a geometry-aware, cage-based 3D Gaussian deformation model. Local tactile measurements are mapped to low-dimensional cage control signals and propagated to dense Gaussian primitives to generate globally consistent shape deformations. A graph attention network regresses cage displacements from tactile input, enforcing spatial smoothness and structural continuity via boundary-aware propagation. Given only a nominal geometric proxy and real-time tactile signals, the system performs zero-shot deformation reconstruction of unseen soft robots in bending and twisting motions, while rendering photorealistic RGB in real time. It achieves 0.67 IoU, 0.65 SSIM, and 3.48 mm Chamfer distance, demonstrating strong zero-shot generalization through explicit coupling of tactile sensing and structured geometric deformation.
>
---
#### [new 004] CeRLP: A Cross-embodiment Robot Local Planning Framework for Visual Navigation
- **分类: cs.RO**

- **简介: 该论文提出CeRLP框架，解决跨机体机器人视觉导航问题，通过统一几何表示和深度校正，提升导航鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.19602](https://arxiv.org/pdf/2603.19602)**

> **作者:** Haoyu Xi; Mingao Tan; Xinming Zhang; Siwei Cheng; Shanze Wang; Yin Gu; Xiaoyu Shen; Wei Zhang
>
> **摘要:** Visual navigation for cross-embodiment robots is challenging due to variations in robot and camera configurations, which can lead to the failure of navigation tasks. Previous approaches typically rely on collecting massive datasets across different robots, which is highly data-intensive, or fine-tuning models, which is time-consuming. Furthermore, both methods often lack explicit consideration of robot geometry. In this paper, we propose a Cross-embodiment Robot Local Planning (CeRLP) framework for general visual navigation, which abstracts visual information into a unified geometric formulation and applies to heterogeneous robots with varying physical dimensions, camera parameters, and camera types. CeRLP introduces a depth estimation scale correction method that utilizes offline pre-calibration to resolve the scale ambiguity of monocular depth estimation, thereby recovering precise metric depth images. Furthermore, CeRLP designs a visual-to-scan abstraction module that projects varying visual inputs into height-adaptive laser scans, making the policy robust to heterogeneous robots. Experiments in simulation environments demonstrate that CeRLP outperforms comparative methods, validating its robust obstacle avoidance capabilities as a local planner. Additionally, extensive real-world experiments verify the effectiveness of CeRLP in tasks such as point-to-point navigation and vision-language navigation, demonstrating its generalization across varying robot and camera configurations.
>
---
#### [new 005] Can LLMs Prove Robotic Path Planning Optimality? A Benchmark for Research-Level Algorithm Verification
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划算法验证任务，旨在解决LLMs在证明算法近似最优性方面的不足。工作包括构建基准测试集，评估LLMs表现，并探索提升推理的方法。**

- **链接: [https://arxiv.org/pdf/2603.19464](https://arxiv.org/pdf/2603.19464)**

> **作者:** Zhengbang Yang; Md. Tasin Tazwar; Minghan Wei; Zhuangdi Zhu
>
> **摘要:** Robotic path planning problems are often NP-hard, and practical solutions typically rely on approximation algorithms with provable performance guarantees for general cases. While designing such algorithms is challenging, formally proving their approximation optimality is even more demanding, which requires domain-specific geometric insights and multi-step mathematical reasoning over complex operational constraints. Recent Large Language Models (LLMs) have demonstrated strong performance on mathematical reasoning benchmarks, yet their ability to assist with research-level optimality proofs in robotic path planning remains under-explored. In this work, we introduce the first benchmark for evaluating LLMs on approximation-ratio proofs of robotic path planning algorithms. The benchmark consists of 34 research-grade proof tasks spanning diverse planning problem types and complexity levels, each requiring structured reasoning over algorithm descriptions, problem constraints, and theoretical guarantees. Our evaluation of state-of-the-art proprietary and open-source LLMs reveals that even the strongest models struggle to produce fully valid proofs without external domain knowledge. However, providing LLMs with task-specific in-context lemmas substantially improves reasoning quality, a factor that is more effective than generic chain-of-thought prompting or supplying the ground-truth approximation ratio as posterior knowledge. We further provide fine-grained error analysis to characterize common logical failures and hallucinations, and demonstrate how each error type can be mitigated through targeted context augmentation.
>
---
#### [new 006] Real-Time Optical Communication Using Event-Based Vision with Moving Transmitters
- **分类: cs.RO**

- **简介: 该论文属于实时光学通信任务，解决多机器人系统中传统RF通信的干扰问题。采用事件相机实现高速、鲁棒的移动发射器跟踪与信息解码。**

- **链接: [https://arxiv.org/pdf/2603.19477](https://arxiv.org/pdf/2603.19477)**

> **作者:** Harmeet Dhillon; Pranay Katyal; Brendan Long; Rohan Walia; Matthew Cleaveland; Kevin Leahy
>
> **备注:** 8 pages, 7 Figures, Submitted to IROS 2026 - Under Review
>
> **摘要:** In multi-robot systems, traditional radio frequency (RF) communication struggles with contention and jamming. Optical communication offers a strong alternative. However, conventional frame-based cameras suffer from limited frame rates, motion blur, and reduced robustness under high dynamic range lighting. Event cameras support microsecond temporal resolution and high dynamic range, making them extremely sensitive to scene changes under fast relative motion with an optical transmitter. Leveraging these strengths, we develop a complete optical communication system capable of tracking moving transmitters and decoding messages in real time. Our system achieves over $95\%$ decoding accuracy for text transmission during motion by implementing a Geometry-Aware Unscented Kalman Filter (GA-UKF), achieving 7x faster processing speed compared to the previous state-of-the-art method, while maintaining equivalent tracking accuracy at transmitting frequencies $\geq$ 1 kHz.
>
---
#### [new 007] Legged Autonomous Surface Science In Analogue Environments (LASSIE): Making Every Robotic Step Count in Planetary Exploration
- **分类: cs.RO**

- **简介: 该论文属于行星探测任务，旨在解决轮式机器人在复杂地形中探索效率低及数据采集灵活性差的问题。通过高机动腿足机器人和类人数据采集算法，提升探测能力与科学适应性。**

- **链接: [https://arxiv.org/pdf/2603.19661](https://arxiv.org/pdf/2603.19661)**

> **作者:** Cristina G. Wilson; Marion Nachon; Shipeng Liu; John G. Ruck; J. Diego Caporale; Benjamin E. McKeeby; Yifeng Zhang; Jordan M. Bretzfelder; John Bush; Alivia M. Eng; Ethan Fulcher; Emmy B. Hughes; Ian C. Rankin; Jelis J. Sostre Cortés; Sophie Silver; Michael R. Zanetti; Ryan C. Ewing; Kenton R. Fisher; Douglas J. Jerolmack; Daniel E. Koditschek; Frances Rivera-Hernández; Thomas F. Shipley; Feifei Qian
>
> **摘要:** The ability to efficiently and effectively explore planetary surfaces is currently limited by the capability of wheeled rovers to traverse challenging terrains, and by pre-programmed data acquisition plans with limited in-situ flexibility. In this paper, we present two novel approaches to address these limitations: (i) high-mobility legged robots that use direct surface interactions to collect rich information about the terrain's mechanics to guide exploration; (ii) human-inspired data acquisition algorithms that enable robots to reason about scientific hypotheses and adapt exploration priorities based on incoming ground-sensing measurements. We successfully verify our approach through lab work and field deployments in two planetary analog environments. The new capability for legged robots to measure soil mechanical properties is shown to enable effective traversal of challenging terrains. When coupled with other geologic properties (e.g., composition, thermal properties, and grain size data etc), soil mechanical measurements reveal key factors governing the formation and development of geologic environments. We then demonstrate how human-inspired algorithms turn terrain-sensing robots into teammates, by supporting more flexible and adaptive data collection decisions with human scientists. Our approach therefore enables exploration of a wider range of planetary environments and new substrate investigation opportunities through integrated human-robot systems that support maximum scientific return.
>
---
#### [new 008] Radar-Inertial Odometry with Online Spatio-Temporal Calibration via Continuous-Time IMU Modeling
- **分类: cs.RO**

- **简介: 该论文属于雷达-惯性里程计任务，解决传感器时空标定问题。通过联合优化空间与时间参数，提升系统在复杂环境下的定位精度。**

- **链接: [https://arxiv.org/pdf/2603.19958](https://arxiv.org/pdf/2603.19958)**

> **作者:** Vlaho-Josip Štironja; Luka Petrović; Juraj Peršić; Ivan Marković; Ivan Petrović
>
> **摘要:** Radar-Inertial Odometry (RIO) has emerged as a robust alternative to vision- and LiDAR-based odometry in challenging conditions such as low light, fog, featureless environments, or in adverse weather. However, many existing RIO approaches assume known radar-IMU extrinsic calibration or rely on sufficient motion excitation for online extrinsic estimation, while temporal misalignment between sensors is often neglected or treated independently. In this work, we present a RIO framework that performs joint online spatial and temporal calibration within a factor-graph optimization formulation, based on continuous-time modeling of inertial measurements using uniform cubic B-splines. The proposed continuous-time representation of acceleration and angular velocity accurately captures the asynchronous nature of radar-IMU measurements, enabling reliable convergence of both the temporal offset and extrinsic calibration parameters, without relying on scan matching, target tracking, or environment-specific assumptions.
>
---
#### [new 009] VAMPO: Policy Optimization for Improving Visual Dynamics in Video Action Models
- **分类: cs.RO**

- **简介: 该论文属于视频动作建模任务，旨在解决现有模型在视觉动态精度上的不足。通过策略优化提升视频动作模型的视觉动态表现，增强下游控制效果。**

- **链接: [https://arxiv.org/pdf/2603.19370](https://arxiv.org/pdf/2603.19370)**

> **作者:** Zirui Ge; Pengxiang Ding; Baohua Yin; Qishen Wang; Zhiyong Xie; Yemin Wang; Jinbo Wang; Hengtao Li; Runze Suo; Wenxuan Song; Han Zhao; Shangke Lyu; Zhaoxin Fan; Haoang Li; Ran Cheng; Cheng Chi; Huibin Ge; Yaozhi Luo; Donglin Wang
>
> **摘要:** Video action models are an appealing foundation for Vision--Language--Action systems because they can learn visual dynamics from large-scale video data and transfer this knowledge to downstream robot control. Yet current diffusion-based video predictors are trained with likelihood-surrogate objectives, which encourage globally plausible predictions without explicitly optimizing the precision-critical visual dynamics needed for manipulation. This objective mismatch often leads to subtle errors in object pose, spatial relations, and contact timing that can be amplified by downstream policies. We propose VAMPO, a post-training framework that directly improves visual dynamics in video action models through policy optimization. Our key idea is to formulate multi-step denoising as a sequential decision process and optimize the denoising policy with rewards defined over expert visual dynamics in latent space. To make this optimization practical, we introduce an Euler Hybrid sampler that injects stochasticity only at the first denoising step, enabling tractable low-variance policy-gradient estimation while preserving the coherence of the remaining denoising trajectory. We further combine this design with GRPO and a verifiable non-adversarial reward. Across diverse simulated and real-world manipulation tasks, VAMPO improves task-relevant visual dynamics, leading to better downstream action generation and stronger generalization. The homepage is this https URL.
>
---
#### [new 010] GustPilot: A Hierarchical DRL-INDI Framework for Wind-Resilient Quadrotor Navigation
- **分类: cs.RO**

- **简介: 该论文属于无人机自主导航任务，解决风扰下稳定飞行问题。提出GustPilot框架，结合DRL与INDI控制器，提升抗风能力。**

- **链接: [https://arxiv.org/pdf/2603.19966](https://arxiv.org/pdf/2603.19966)**

> **作者:** Amir Atef Habel; Roohan Ahmed Khan; Fawad Mehboob; Clement Fortin; Dzmitry Tsetserukou
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Wind disturbances remain a key barrier to reliable autonomous navigation for lightweight quadrotors, where the rapidly varying airflow can destabilize both planning and tracking. This paper introduces GustPilot, a hierarchical wind-resilient navigation stack in which a deep reinforcement learning (DRL) policy generates inertial-frame velocity reference for gate traversal. At the same time, a geometric Incremental Nonlinear Dynamic Inversion (INDI) controller provides low-level tracking with fast residual disturbance rejection. The INDI layer achieves this by providing incremental feedback on both specific linear acceleration and angular acceleration rate, using onboard sensor measurements to reject wind disturbances rapidly. Robustness is obtained through a two-level strategy, wind-aware planning learned via fan-jet domain randomization during training, and rapid execution-time disturbance rejection by the INDI tracking controller. We evaluate GustPilot in real flights on a 50g quad-copter platform against a DRL-PID baseline across four scenarios ranging from no-wind to fully dynamic conditions with a moving gate and a moving disturbance source. Despite being trained only in a minimal single-gate and single-fan setup, the policy generalizes to significantly more complex environments (up to six gates and four fans) without retraining. Across 80 experiments, DRL-INDI achieves a 94.7% versus 55.0% for DRL-PID as average Overall Success Rate (OSR), reduces tracking RMSE up to 50%, and sustains speeds up to 1.34 m/s under wind disturbances up to 3.5 m/s. These results demonstrate that combining DRL-based velocity planning with structured INDI disturbance rejection provides a practical and generalizable approach to wind-resilient autonomous flight navigation.
>
---
#### [new 011] The Robot's Inner Critic: Self-Refinement of Social Behaviors through VLM-based Replanning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人社交行为生成任务，旨在提升机器人行为的自主性和灵活性。通过VLM实现自我评估与优化，减少人工干预，增强跨平台适应性。**

- **链接: [https://arxiv.org/pdf/2603.20164](https://arxiv.org/pdf/2603.20164)**

> **作者:** Jiyu Lim; Youngwoo Yoon; Kwanghyun Park
>
> **备注:** Accepted to ICRA 2026. 8 pages, 9 figures, Project page: this https URL
>
> **摘要:** Conventional robot social behavior generation has been limited in flexibility and autonomy, relying on predefined motions or human feedback. This study proposes CRISP (Critique-and-Replan for Interactive Social Presence), an autonomous framework where a robot critiques and replans its own actions by leveraging a Vision-Language Model (VLM) as a `human-like social critic.' CRISP integrates (1) extraction of movable joints and constraints by analyzing the robot's description file (e.g., MJCF), (2) generation of step-by-step behavior plans based on situational context, (3) generation of low-level joint control code by referencing visual information (joint range-of-motion visualizations), (4) VLM-based evaluation of social appropriateness and naturalness, including pinpointing erroneous steps, and (5) iterative refinement of behaviors through reward-based search. This approach is not tied to a specific robot API; it can generate subtly different, human-like motions on various platforms using only the robot's structure file. In a user study involving five different robot types and 20 scenarios, including mobile manipulators and humanoids, our proposed method achieved significantly higher preference and situational appropriateness ratings compared to previous methods. This research presents a general framework that minimizes human intervention while expanding the robot's autonomous interaction capabilities and cross-platform applicability. Detailed result videos and supplementary information regarding this work are available at: this https URL
>
---
#### [new 012] AGILE: A Comprehensive Workflow for Humanoid Loco-Manipulation Learning
- **分类: cs.RO**

- **简介: 该论文提出AGILE框架，解决人形机器人强化学习的仿真到现实迁移问题。通过标准化流程提升可靠性与可复现性。**

- **链接: [https://arxiv.org/pdf/2603.20147](https://arxiv.org/pdf/2603.20147)**

> **作者:** Huihua Zhao; Rafael Cathomen; Lionel Gulich; Wei Liu; Efe Arda Ongan; Michael Lin; Shalin Jain; Soha Pouya; Yan Chang
>
> **摘要:** Recent advances in reinforcement learning (RL) have enabled impressive humanoid behaviors in simulation, yet transferring these results to new robots remains challenging. In many real deployments, the primary bottleneck is no longer simulation throughput or algorithm design, but the absence of systematic infrastructure that links environment verification, training, evaluation, and deployment in a coherent loop. To address this gap, we present AGILE, an end-to-end workflow for humanoid RL that standardizes the policy-development lifecycle to mitigate common sim-to-real failure modes. AGILE comprises four stages: (1) interactive environment verification, (2) reproducible training, (3) unified evaluation, and (4) descriptor-driven deployment via robot/task configuration descriptors. For evaluation stage, AGILE supports both scenario-based tests and randomized rollouts under a shared suite of motion-quality diagnostics, enabling automated regression testing and principled robustness assessment. AGILE also incorporates a set of training stabilizations and algorithmic enhancements in training stage to improve optimization stability and sim-to-real transfer. With this pipeline in place, we validate AGILE across five representative humanoid skills spanning locomotion, recovery, motion imitation, and loco-manipulation on two hardware platforms (Unitree G1 and Booster T1), achieving consistent sim-to-real transfer. Overall, AGILE shows that a standardized, end-to-end workflow can substantially improve the reliability and reproducibility of humanoid RL development.
>
---
#### [new 013] Beyond detection: cooperative multi-agent reasoning for rapid onboard EO crisis response
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于地球观测任务，旨在解决灾害响应中的实时性问题。通过多智能体协同架构，提升星上处理效率，减少计算负担，实现快速决策。**

- **链接: [https://arxiv.org/pdf/2603.19858](https://arxiv.org/pdf/2603.19858)**

> **作者:** Alejandro D. Mousist; Pedro Delgado de Robles Martín; Raquel Lladró Climent; Julian Cobos Aparicio
>
> **备注:** Accepted for presentation at the ESA's 4S Symposium 2026 Conference (see this https URL)
>
> **摘要:** Rapid identification of hazardous events is essential for next-generation Earth Observation (EO) missions supporting disaster response. However, current monitoring pipelines remain largely ground-centric, introducing latency due to downlink limitations, multi-source data fusion constraints, and the computational cost of exhaustive scene analysis. This work proposes a hierarchical multi-agent architecture for onboard EO processing under strict resource and bandwidth constraints. The system enables the exploitation of complementary multimodal observations by coordinating specialized AI agents within an event-driven decision pipeline. AI agents can be deployed across multiple nodes in a distributed setting, such as satellite platforms. An Early Warning agent generates fast hypotheses from onboard observations and selectively activates domain-specific analysis agents, while a Decision agent consolidates the evidence to issue a final alert. The architecture combines vision-language models, traditional remote sensing analysis tools, and role-specialized agents to enable structured reasoning over multimodal observations while minimizing unnecessary computation. A proof-of-concept implementation was executed on the engineering model of an edge-computing platform currently deployed in orbit, using representative satellite data. Experiments on wildfire and flood monitoring scenarios show that the proposed routing-based pipeline significantly reduces computational overhead while maintaining coherent decision outputs, demonstrating the feasibility of distributed agent-based reasoning for future autonomous EO constellations.
>
---
#### [new 014] Accurate Open-Loop Control of a Soft Continuum Robot Through Visually Learned Latent Representations
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于软体机器人控制任务，旨在解决无反馈的开环控制问题。通过视觉学习的潜在动力学实现精准控制，使用VON和ABCD模型降低跟踪误差。**

- **链接: [https://arxiv.org/pdf/2603.19655](https://arxiv.org/pdf/2603.19655)**

> **作者:** Henrik Krauss; Johann Licher; Naoya Takeishi; Annika Raatz; Takehisa Yairi
>
> **摘要:** This work addresses open-loop control of a soft continuum robot (SCR) from video-learned latent dynamics. Visual Oscillator Networks (VONs) from previous work are used, that provide mechanistically interpretable 2D oscillator latents through an attention broadcast decoder (ABCD). Open-loop, single-shooting optimal control is performed in latent space to track image-specified waypoints without camera feedback. An interactive SCR live simulator enables design of static, dynamic, and extrapolated targets and maps them to model-specific latent waypoints. On a two-segment pneumatic SCR, Koopman, MLP, and oscillator dynamics, each with and without ABCD, are evaluated on setpoint and dynamic trajectories. ABCD-based models consistently reduce image-space tracking error. The VON and ABCD-based Koopman models attains the lowest MSEs. Using an ablation study, we demonstrate that several architecture choices and training settings contribute to the open-loop control performance. Simulation stress tests further confirm static holding, stable extrapolated equilibria, and plausible relaxation to the rest state. To the best of our knowledge, this is the first demonstration that interpretable, video-learned latent dynamics enable reliable long-horizon open-loop control of an SCR.
>
---
#### [new 015] HortiMulti: A Multi-Sensor Dataset for Localisation and Mapping in Horticultural Polytunnels
- **分类: cs.RO**

- **简介: 该论文提出HortiMulti数据集，用于解决农业机器人在多季节温室中的定位与建图问题，包含多种传感器数据及真实场景挑战。**

- **链接: [https://arxiv.org/pdf/2603.20150](https://arxiv.org/pdf/2603.20150)**

> **作者:** Shuoyuan Xu; Zhipeng Zhong; Tiago Barros; Matthew Coombes; Cristiano Premebida; Hao Wu; Cunjia Liu
>
> **摘要:** Agricultural robotics is gaining increasing relevance in both research and real-world deployment. As these systems are expected to operate autonomously in more complex tasks, the availability of representative real-world datasets becomes essential. While domains such as urban and forestry robotics benefit from large and established benchmarks, horticultural environments remain comparatively under-explored despite the economic significance of this sector. To address this gap, we present HortiMulti, a multimodal, cross-season dataset collected in commercial strawberry and raspberry polytunnels across an entire growing season, capturing substantial appearance variation, dynamic foliage, specular reflections from plastic covers, severe perceptual aliasing, and GNSS-unreliable conditions, all of which directly degrade existing localisation and perception algorithms. The sensor suite includes two 3D LiDARs, four RGB cameras, an IMU, GNSS, and wheel odometry. Ground truth trajectories are derived from a combination of Total Station surveying, AprilTag fiducial markers, and LiDAR-inertial odometry, spanning dense, sparse, and marker-free coverage to support evaluation under both controlled and realistic conditions. We release time-synchronised raw measurements, calibration files, reference trajectories, and baseline benchmarks for visual, LiDAR, and multi-sensor SLAM, with results confirming that current state-of-the-art methods remain inadequate for reliable polytunnel deployment, establishing HortiMulti as a one-stop resource for developing and testing robotic perception systems in horticulture environments.
>
---
#### [new 016] A Closed-Form CLF-CBF Controller for Whole-Body Continuum Soft Robot Collision Avoidance
- **分类: cs.RO**

- **简介: 该论文属于机器人安全控制任务，解决软体机械臂在复杂环境中的实时避障问题。提出一种闭式CLF-CBF控制器，确保稳定与安全，提升计算效率。**

- **链接: [https://arxiv.org/pdf/2603.19424](https://arxiv.org/pdf/2603.19424)**

> **作者:** Kiwan Wong; Maximillian Stölzle; Wei Xiao; Daniela Rus
>
> **摘要:** Safe operation is essential for deploying robots in human-centered 3D environments. Soft continuum manipulators provide passive safety through mechanical compliance, but still require active control to achieve reliable collision avoidance. Existing approaches, such as sampling-based planning, are often computationally expensive and lack formal safety guarantees, which limits their use for real-time whole-body avoidance. This paper presents a closed-form Control Lyapunov Function--Control Barrier Function (CLF--CBF) controller for real-time 3D obstacle avoidance in soft continuum manipulators without online optimization. By analytically embedding safety constraints into the control input, the proposed method ensures stability and safety under the stated modeling assumptions, while avoiding feasibility issues commonly encountered in online optimization-based methods. The resulting controller is up to $10\times$ faster than standard CLF--CBF quadratic-programming approaches and up to $100\times$ faster than traditional sampling-based planners. Simulation and hardware experiments on a tendon-driven soft manipulator demonstrate accurate 3D trajectory tracking and robust obstacle avoidance in cluttered environments. These results show that the proposed framework provides a scalable and provably safe control strategy for soft robots operating in dynamic, safety-critical settings.
>
---
#### [new 017] ContractionPPO: Certified Reinforcement Learning via Differentiable Contraction Layers
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出ContractionPPO，用于腿部机器人在非结构化环境中的稳定控制。解决高性能与形式化保证的难题，通过结合PPO与收缩度量层实现稳定控制。**

- **链接: [https://arxiv.org/pdf/2603.19632](https://arxiv.org/pdf/2603.19632)**

> **作者:** Vrushabh Zinage; Narek Harutyunyan; Eric Verheyden; Fred Y. Hadaegh; Soon-Jo Chung
>
> **备注:** Accepted to RA-L journal
>
> **摘要:** Legged locomotion in unstructured environments demands not only high-performance control policies but also formal guarantees to ensure robustness under perturbations. Control methods often require carefully designed reference trajectories, which are challenging to construct in high-dimensional, contact-rich systems such as quadruped robots. In contrast, Reinforcement Learning (RL) directly learns policies that implicitly generate motion, and uniquely benefits from access to privileged information, such as full state and dynamics during training, that is not available at deployment. We present ContractionPPO, a framework for certified robust planning and control of legged robots by augmenting Proximal Policy Optimization (PPO) RL with a state-dependent contraction metric layer. This approach enables the policy to maximize performance while simultaneously producing a contraction metric that certifies incremental exponential stability of the simulated closed-loop system. The metric is parameterized as a Lipschitz neural network and trained jointly with the policy, either in parallel or as an auxiliary head of the PPO backbone. While the contraction metric is not deployed during real-world execution, we derive upper bounds on the worst-case contraction rate and show that these bounds ensure the learned contraction metric generalizes from simulation to real-world deployment. Our hardware experiments on quadruped locomotion demonstrate that ContractionPPO enables robust, certifiably stable control even under strong external perturbations.
>
---
#### [new 018] Generalized Task-Driven Design of Soft Robots via Reduced-Order FEM-based Surrogate Modeling
- **分类: cs.RO**

- **简介: 该论文属于软体机器人设计任务，旨在解决物理准确与计算效率的平衡问题。通过构建降阶FEM代理模型，实现高效、可迁移的设计优化。**

- **链接: [https://arxiv.org/pdf/2603.19794](https://arxiv.org/pdf/2603.19794)**

> **作者:** Yao Yao; David Howard; Perla Maiolino
>
> **摘要:** Task-driven design of soft robots requires models that are physically accurate and computationally efficient, while remaining transferable across actuator designs and task scenarios. However, existing modeling approaches typically face a fundamental trade-off between physical fidelity and computational efficiency, which limits model reuse across design and task variations and constrains scalable task-driven optimization. This paper presents a unified reduced-order finite element method (FEM)-based surrogate modeling pipeline for generalized task-driven soft robot design. High-fidelity FEM simulations characterize actuator behavior at the modular level, from which compact surrogate joint models are constructed for evaluation within a pseudo-rigid body model (PRBM). A meta-model maps actuator design parameters to surrogate representations, enabling rapid instantiation across a parameterized actuator family. The resulting models are embedded into a PRBM-based simulation environment, supporting task-level simulation and optimization under realistic physical constraints. The proposed pipeline is validated through sim-to-real transfer across multiple actuator types, including bellow-type pneumatic actuators and a tendon-driven soft finger, as well as two task-driven design studies: soft gripper co-design via Reinforcement Learning (RL) and 3D actuator shape matching via evolutionary optimization. The results demonstrate high accuracy, efficiency, and reliable reuse, providing a scalable foundation for autonomous task-driven soft robot design.
>
---
#### [new 019] SOFTMAP: Sim2Real Soft Robot Forward Modeling via Topological Mesh Alignment and Physics Prior
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，解决软机械臂从低维指令到三维形状的精确建模问题。提出SOFTMAP框架，结合拓扑对齐与物理先验，实现高精度实时仿真到现实的映射。**

- **链接: [https://arxiv.org/pdf/2603.19384](https://arxiv.org/pdf/2603.19384)**

> **作者:** Ziyong Ma; Uksang Yoo; Jonathan Francis; Weiming Zhi; Jeffrey Ichnowski; Jean Oh
>
> **摘要:** While soft robot manipulators offer compelling advantages over rigid counterparts, including inherent compliance, safe human-robot interaction, and the ability to conform to complex geometries, accurate forward modeling from low-dimensional actuation commands remains an open challenge due to nonlinear material phenomena such as hysteresis and manufacturing variability. We present SOFTMAP, a sim-to-real learning framework for real-time 3D forward modeling of tendon-actuated soft finger manipulators. SOFTMAP combines four components: (1) As-Rigid-As-Possible (ARAP)-based topological alignment that projects simulated and real point clouds into a shared, topologically consistent vertex space; (2) a lightweight MLP forward model pretrained on simulation data to map servo commands to full 3D finger geometry; (3) a residual correction network trained on a small set of real observations to predict per-vertex displacement fields that compensate for sim-to-real discrepancies; and (4) a closed-form linear actuation calibration layer enabling real-time inference at 30 FPS. We evaluate SOFTMAP on both simulated and physical hardware, achieving state-of-the-art shape prediction accuracy with a Chamfer distance of 0.389 mm in simulation and 3.786 mm on hardware, millimeter-level fingertip trajectory tracking across multiple target paths, and a 36.5% improvement in teleoperation task success over the baseline. Our results show that SOFTMAP provides a data-efficient approach for 3D forward modeling and control of soft manipulators.
>
---
#### [new 020] Uncertainty Matters: Structured Probabilistic Online Mapping for Motion Prediction in Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的在线地图生成与轨迹预测任务，旨在解决传统方法忽略结构不确定性的缺陷，通过引入结构化概率模型提升地图质量和运动预测性能。**

- **链接: [https://arxiv.org/pdf/2603.20076](https://arxiv.org/pdf/2603.20076)**

> **作者:** Pritom Gogoi; Faris Janjoš; Bin Yang; Andreas Look
>
> **摘要:** Online map generation and trajectory prediction are critical components of the autonomous driving perception-prediction-planning pipeline. While modern vectorized mapping models achieve high geometric accuracy, they typically treat map estimation as a deterministic task, discarding structural uncertainty. Existing probabilistic approaches often rely on diagonal covariance matrices, which assume independence between points and fail to capture the strong spatial correlations inherent in road geometry. To address this, we propose a structured probabilistic formulation for online map generation. Our method explicitly models intra-element dependencies by predicting a dense covariance matrix, parameterized via a Low-Rank plus Diagonal (LRPD) covariance decomposition. This formulation represents uncertainty as a combination of a low-rank component, which captures global spatial structure, and a diagonal component representing independent local noise, thereby capturing geometric correlations without the prohibitive computational cost of full covariance matrices. Evaluations on the nuScenes dataset demonstrate that our uncertainty-aware framework yields consistent improvements in online map generation quality compared to deterministic baselines. Furthermore, our approach establishes new state-of-the-art performance for map-based motion prediction, highlighting the critical role of uncertainty in planning tasks. Code is published under link-available-soon.
>
---
#### [new 021] Morphology-Consistent Humanoid Interaction through Robot-Centric Video Synthesis
- **分类: cs.RO**

- **简介: 该论文属于人形机器人交互任务，解决运动迁移中的形态差异问题。提出Dream2Act框架，通过生成视频合成实现零样本交互，避免姿态映射错误。**

- **链接: [https://arxiv.org/pdf/2603.19709](https://arxiv.org/pdf/2603.19709)**

> **作者:** Weisheng Xu; Jian Li; Yi Gu; Bin Yang; Haodong Chen; Shuyi Lin; Mingqian Zhou; Jing Tan; Qiwei Wu; Xiangrui Jiang; Taowen Wang; Jiawen Wen; Qiwei Liang; Jiaxi Zhang; Renjing Xu
>
> **摘要:** Equipping humanoid robots with versatile interaction skills typically requires either extensive policy training or explicit human-to-robot motion retargeting. However, learning-based policies face prohibitive data collection costs. Meanwhile, retargeting relies on human-centric pose estimation (e.g., SMPL), introducing a morphology gap. Skeletal scale mismatches result in severe spatial misalignments when mapped to robots, compromising interaction success. In this work, we propose Dream2Act, a robot-centric framework enabling zero-shot interaction through generative video synthesis. Given a third-person image of the robot and target object, our framework leverages video generation models to envision the robot completing the task with morphology-consistent motion. We employ a high-fidelity pose extraction system to recover physically feasible, robot-native joint trajectories from these synthesized dreams, subsequently executed via a general-purpose whole-body controller. Operating strictly within the robot-native coordinate space, Dream2Act avoids retargeting errors and eliminates task-specific policy training. We evaluate Dream2Act on the Unitree G1 across four whole-body mobile interaction tasks: ball kicking, sofa sitting, bag punching, and box hugging. Dream2Act achieves a 37.5% overall success rate, compared to 0% for conventional retargeting. While retargeting fails to establish correct physical contacts due to the morphology gap (with errors compounded during locomotion), Dream2Act maintains robot-consistent spatial alignment, enabling reliable contact formation and substantially higher task completion.
>
---
#### [new 022] PhyGile: Physics-Prefix Guided Motion Generation for Agile General Humanoid Motion Tracking
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出PhyGile，解决人形机器人运动生成中的物理可行性问题。通过物理前缀引导的运动生成方法，提升真实环境下的运动稳定性与敏捷性。**

- **链接: [https://arxiv.org/pdf/2603.19305](https://arxiv.org/pdf/2603.19305)**

> **作者:** Jiacheng Bao; Haoran Yang; Yucheng Xin; Junhong Liu; Yuecheng Xu; Han Liang; Pengfei Han; Xiaoguang Ma; Dong Wang; Bin Zhao
>
> **摘要:** Humanoid robots are expected to execute agile and expressive whole-body motions in real-world settings. Existing text-to-motion generation models are predominantly trained on captured human motion datasets, whose priors assume human biomechanics, actuation, mass distribution, and contact strategies. When such motions are directly retargeted to humanoid robots, the resulting trajectories may satisfy geometric constraints (e.g., joint limits and pose continuity) and appear kinematically reasonable. However, they frequently violate the physical feasibility required for real-world execution. To address these issues, we present PhyGile, a unified framework that closes the loop between robot-native motion generation and General Motion Tracking (GMT). PhyGile performs physics-prefix-guided robot-native motion generation at inference time, directly generating robot-native motions in a 262-dimensional skeletal space with physics-guided prefixes, thereby eliminating inference-time retargeting artifacts and reducing generation-execution discrepancies. Before physics-prefix adaptation, we train the GMT controller with a curriculum-based mixture-of-experts scheme, followed by post-training on unlabeled motion data to improve robustness over large-scale robot motions. During physics-prefix adaptation, the GMT controller is further fine-tuned with generated objectives under physics-derived prefixes, enabling agile and stable execution of complex motions on real robots. Extensive offline and real-robot experiments demonstrate that PhyGile expands the frontier of text-driven humanoid control, enabling stable tracking of agile, highly difficult whole-body motions that go well beyond walking and low-dynamic motions typically achieved by prior methods.
>
---
#### [new 023] KUKAloha: A General, Low-Cost, and Shared-Control based Teleoperation Framework for Construction Robot Arm
- **分类: cs.RO**

- **简介: 该论文提出KUKAloha框架，用于建筑机器人臂的远程操作。解决大尺寸机械臂操作安全性与效率问题，通过人机协同控制提升任务完成效果。**

- **链接: [https://arxiv.org/pdf/2603.20129](https://arxiv.org/pdf/2603.20129)**

> **作者:** Yifan Xu; Qizhang Shen; Vineet Kamat; Carol Menassa
>
> **备注:** 9 pages, 4 figures, 1 table
>
> **摘要:** This paper presents KUKAloha, a general, low-cost, and shared-control teleoperation framework designed for construction robot arms. The proposed system employs a leader-follower paradigm in which a lightweight leading arm enables intuitive human guidance for coarse robot motion, while an autonomous perception module based on AprilTag detection performs precise alignment and grasp execution. By explicitly decoupling human control from fine manipulation, KUKAloha improves safety and repeatability when operating large-scale manipulators. We implement the framework on a KUKA robot arm and conduct a usability study with representative construction manipulation tasks. Experimental results demonstrate that KUKAloha reduces operator workload, improves task completion efficiency, and provides a practical solution for scalable demonstration collection and shared human-robot control in construction environments.
>
---
#### [new 024] Speculative Policy Orchestration: A Latency-Resilient Framework for Cloud-Robotic Manipulation
- **分类: cs.RO; cs.DC**

- **简介: 该论文针对云机器人连续操作任务中的网络延迟问题，提出SPO框架，通过预计算和误差校验实现低延迟控制，提升系统稳定性和安全性。**

- **链接: [https://arxiv.org/pdf/2603.19418](https://arxiv.org/pdf/2603.19418)**

> **作者:** Chanh Nguyen; Shutong Jin; Florian T. Pokorny; Erik Elmroth
>
> **备注:** 9 pages, 7 figures, conference submission
>
> **摘要:** Cloud robotics enables robots to offload high-dimensional motion planning and reasoning to remote servers. However, for continuous manipulation tasks requiring high-frequency control, network latency and jitter can severely destabilize the system, causing command starvation and unsafe physical execution. To address this, we propose Speculative Policy Orchestration (SPO), a latency-resilient cloud-edge framework. SPO utilizes a cloud-hosted world model to pre-compute and stream future kinematic waypoints to a local edge buffer, decoupling execution frequency from network round-trip time. To mitigate unsafe execution caused by predictive drift, the edge node employs an $\epsilon$-tube verifier that strictly bounds kinematic execution errors. The framework is coupled with an Adaptive Horizon Scaling mechanism that dynamically expands or shrinks the speculative pre-fetch depth based on real-time tracking error. We evaluate SPO on continuous RLBench manipulation tasks under emulated network delays. Results show that even when deployed with learned models of modest accuracy, SPO reduces network-induced idle time by over 60% compared to blocking remote inference. Furthermore, SPO discards approximately 60% fewer cloud predictions than static caching baselines. Ultimately, SPO enables fluid, real-time cloud-robotic control while maintaining bounded physical safety.
>
---
#### [new 025] Multi-Agent Motion Planning on Industrial Magnetic Levitation Platforms: A Hybrid ADMM-HOCBF approach
- **分类: cs.RO**

- **简介: 该论文属于多智能体运动规划任务，解决大规模系统中集中式MPC计算复杂度高的问题，提出混合ADMM-HOCBF方法实现安全、高效的分布式控制。**

- **链接: [https://arxiv.org/pdf/2603.19838](https://arxiv.org/pdf/2603.19838)**

> **作者:** Bavo Tistaert; Stan Servaes; Alejandro Gonzalez-Garcia; Ibrahim Ibrahim; Louis Callens; Jan Swevers; Wilm Decré
>
> **备注:** 8 pages, 4 figures, accepted to the European Control Conference 2026
>
> **摘要:** This paper presents a novel hybrid motion planning method for holonomic multi-agent systems. The proposed decentralised model predictive control (MPC) framework tackles the intractability of classical centralised MPC for a growing number of agents while providing safety guarantees. This is achieved by combining a decentralised version of the alternating direction method of multipliers (ADMM) with a centralised high-order control barrier function (HOCBF) architecture. Simulation results show significant improvement in scalability over classical centralised MPC. We validate the efficacy and real-time capability of the proposed method by developing a highly efficient C++ implementation and deploying the resulting trajectories on a real industrial magnetic levitation platform.
>
---
#### [new 026] Not an Obstacle for Dog, but a Hazard for Human: A Co-Ego Navigation System for Guide Dog Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决透明障碍物检测问题。通过融合机器人和用户视角，提出Co-Ego系统提升导航安全性。**

- **链接: [https://arxiv.org/pdf/2603.20121](https://arxiv.org/pdf/2603.20121)**

> **作者:** Ruiping Liu; Jingqi Zhang; Junwei Zheng; Yufan Chen; Peter Seungjune Lee; Di Wen; Kunyu Peng; Jiaming Zhang; Kailun Yang; Katja Mombaur; Rainer Stiefelhagen
>
> **摘要:** Guide dogs offer independence to Blind and Low-Vision (BLV) individuals, yet their limited availability leaves the vast majority of BLV users without access. Quadruped robotic guide dogs present a promising alternative, but existing systems rely solely on the robot's ground-level sensors for navigation, overlooking a critical class of hazards: obstacles that are transparent to the robot yet dangerous at human body height, such as bent branches. We term this the viewpoint asymmetry problem and present the first system to explicitly address it. Our Co-Ego system adopts a dual-branch obstacle avoidance framework that integrates the robot-centric ground sensing with the user's elevated egocentric perspective to ensure comprehensive navigation safety. Deployed on a quadruped robot, the system is evaluated in a controlled user study with sighted participants under blindfold across three conditions: unassisted, single-view, and cross-view fusion. Results demonstrate that cross-view fusion significantly reduces collision times and cognitive load, verifying the necessity of viewpoint complementarity for safe robotic guide dog navigation.
>
---
#### [new 027] Real-Time Structural Detection for Indoor Navigation from 3D LiDAR Using Bird's-Eye-View Images
- **分类: cs.RO**

- **简介: 该论文属于室内导航中的结构检测任务，旨在解决资源受限机器人实时高效感知问题。通过将3D LiDAR数据转为BEV图像，结合YOLO-OBB实现快速准确的结构检测。**

- **链接: [https://arxiv.org/pdf/2603.19830](https://arxiv.org/pdf/2603.19830)**

> **作者:** Guanliang Li; Pedro Espinosa Angulo; David Perez Saura; Santiago Tapia Fernandez
>
> **摘要:** Efficient structural perception is essential for mapping and autonomous navigation on resource-constrained robots. Existing 3D methods are computationally prohibitive, while traditional 2D geometric approaches lack robustness. This paper presents a lightweight, real-time framework that projects 3D LiDAR data into 2D Bird's-Eye-View (BEV) images to enable efficient detection of structural elements relevant to mapping and navigation. Within this representation, we systematically evaluate several feature extraction strategies, including classical geometric techniques (Hough Transform, RANSAC, and LSD) and a deep learning detector based on YOLO-OBB. The resulting detections are integrated through a spatiotemporal fusion module that improves stability and robustness across consecutive frames. Experiments conducted on a standard mobile robotic platform highlight clear performance trade-offs. Classical methods such as Hough and LSD provide fast responses but exhibit strong sensitivity to noise, with LSD producing excessive segment fragmentation that leads to system congestion. RANSAC offers improved robustness but fails to meet real-time constraints. In contrast, the YOLO-OBB-based approach achieves the best balance between robustness and computational efficiency, maintaining an end-to-end latency (satisfying 10 Hz operation) while effectively filtering cluttered observations in a low-power single-board computer (SBC) without using GPU acceleration. The main contribution of this work is a computationally efficient BEV-based perception pipeline enabling reliable real-time structural detection from 3D LiDAR on resource-constrained robotic platforms that cannot rely on GPU-intensive processing.
>
---
#### [new 028] LIORNet: Self-Supervised LiDAR Snow Removal Framework for Autonomous Driving under Adverse Weather Conditions
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云去噪任务，旨在解决雪天环境下LiDAR感知性能下降的问题。提出LIORNet框架，采用自监督学习有效区分噪声与真实点云，无需人工标注。**

- **链接: [https://arxiv.org/pdf/2603.19936](https://arxiv.org/pdf/2603.19936)**

> **作者:** Ji-il Park; Inwook Shim
>
> **备注:** 14 pages, 6 figures, 2 tables
>
> **摘要:** LiDAR sensors provide high-resolution 3D perception and long-range detection, making them indispensable for autonomous driving and robotics. However, their performance significantly degrades under adverse weather conditions such as snow, rain, and fog, where spurious noise points dominate the point cloud and lead to false perception. To address this problem, various approaches have been proposed: distance-based filters exploiting spatial sparsity, intensity-based filters leveraging reflectance distributions, and learning-based methods that adapt to complex environments. Nevertheless, distance-based methods struggle to distinguish valid object points from noise, intensity-based methods often rely on fixed thresholds that lack adaptability to changing conditions, and learning-based methods suffer from the high cost of annotation, limited generalization, and computational overhead. In this study, we propose LIORNet, which eliminates these drawbacks and integrates the strengths of all three paradigms. LIORNet is built upon a U-Net++ backbone and employs a self-supervised learning strategy guided by pseudo-labels generated from multiple physical and statistical cues, including range-dependent intensity thresholds, snow reflectivity, point sparsity, and sensing range constraints. This design enables LIORNet to distinguish noise points from environmental structures without requiring manual annotations, thereby overcoming the difficulty of snow labeling and the limitations of single-principle approaches. Extensive experiments on the WADS and CADC datasets demonstrate that LIORNet outperforms state-of-the-art filtering algorithms in both accuracy and runtime while preserving critical environmental features. These results highlight LIORNet as a practical and robust solution for LiDAR perception in extreme weather, with strong potential for real-time deployment in autonomous driving systems.
>
---
#### [new 029] Unlabeled Multi-Robot Motion Planning with Improved Separation Trade-offs
- **分类: cs.CG; cs.RO**

- **简介: 该论文研究多机器人运动规划任务，解决在受限环境中机器人路径规划问题。通过改进分离条件，提出多项多项式时间算法，提升现有方法的效率与适用性。**

- **链接: [https://arxiv.org/pdf/2603.19502](https://arxiv.org/pdf/2603.19502)**

> **作者:** Tsuri Farhana; Omrit Filtser; Shalev Goldshtein
>
> **摘要:** We study unlabeled multi-robot motion planning for unit-disk robots in a polygonal environment. Although the problem is hard in general, polynomial-time solutions exist under appropriate separation assumptions on start and target positions. Banyassady et al. (SoCG'22) guarantee feasibility in simple polygons under start--start and target--target distances of at least $4$, and start--target distances of at least $3$, but without optimality guarantees. Solovey et al. (RSS'15) provide a near-optimal solution in general polygonal domains, under stricter conditions: start/target positions must have pairwise distance at least $4$, and at least $\sqrt{5}\approx2.236$ from obstacles. This raises the question of whether polynomial-time algorithms can be obtained in even more densely packed environments. In this paper we present a generalized algorithm that achieve different trade-offs on the robots-separation and obstacles-separation bounds, all significantly improving upon the state of the art. Specifically, we obtain polynomial-time constant-approximation algorithms to minimize the total path length when (i) the robots-separation is $2\tfrac{2}{3}$ and the obstacles-separation is $1\tfrac{2}{3}$, or (ii) the robots-separation is $\approx3.291$ and the obstacles-separation $\approx1.354$. Additionally, we introduce a different strategy yielding a polynomial-time solution when the robots-separation is only $2$, and the obstacles-separation is $3$. Finally, we show that without any robots-separation assumption, obstacles-separation of at least $1.5$ may be necessary for a solution to exist.
>
---
#### [new 030] Pedestrian Crossing Intent Prediction via Psychological Features and Transformer Fusion
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于行人过街意图预测任务，旨在提升自动驾驶安全性。通过融合多行为流和不确定性量化方法，提出一种轻量高效模型，有效提升预测准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2603.19533](https://arxiv.org/pdf/2603.19533)**

> **作者:** Sima Ashayer; Hoang H. Nguyen; Yu Liang; Mina Sartipi
>
> **备注:** Accepted to IEEE Intelligent Vehicles Symposium (IV) 2026. 8 pages, 3 figures
>
> **摘要:** Pedestrian intention prediction needs to be accurate for autonomous vehicles to navigate safely in urban environments. We present a lightweight, socially informed architecture for pedestrian intention prediction. It fuses four behavioral streams (attention, position, situation, and interaction) using highway encoders, a compact 4-token Transformer, and global self-attention pooling. To quantify uncertainty, we incorporate two complementary heads: a variational bottleneck whose KL divergence captures epistemic uncertainty, and a Mahalanobis distance detector that identifies distributional shift. Together, these components yield calibrated probabilities and actionable risk scores without compromising efficiency. On the PSI 1.0 benchmark, our model outperforms recent vision language models by achieving 0.9 F1, 0.94 AUC-ROC, and 0.78 MCC by using only structured, interpretable features. On the more diverse PSI 2.0 dataset, where, to the best of our knowledge, no prior results exist, we establish a strong initial baseline of 0.78 F1 and 0.79 AUC-ROC. Selective prediction based on Mahalanobis scores increases test accuracy by up to 0.4 percentage points at 80% coverage. Qualitative attention heatmaps further show how the model shifts its cross-stream focus under ambiguity. The proposed approach is modality-agnostic, easy to integrate with vision language pipelines, and suitable for risk-aware intent prediction on resource-constrained platforms.
>
---
#### [new 031] Spectral Alignment in Forward-Backward Representations via Temporal Abstraction
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决连续环境中前向后向表示的谱不匹配问题。通过时间抽象降低谱复杂度，提升长期规划能力。**

- **链接: [https://arxiv.org/pdf/2603.20103](https://arxiv.org/pdf/2603.20103)**

> **作者:** Seyed Mahdi B. Azad; Jasper Hoffmann; Iman Nematollahi; Hao Zhu; Abhinav Valada; Joschka Boedecker
>
> **摘要:** Forward-backward (FB) representations provide a powerful framework for learning the successor representation (SR) in continuous spaces by enforcing a low-rank factorization. However, a fundamental spectral mismatch often exists between the high-rank transition dynamics of continuous environments and the low-rank bottleneck of the FB architecture, making accurate low-rank representation learning difficult. In this work, we analyze temporal abstraction as a mechanism to mitigate this mismatch. By characterizing the spectral properties of the transition operator, we show that temporal abstraction acts as a low-pass filter that suppresses high-frequency spectral components. This suppression reduces the effective rank of the induced SR while preserving a formal bound on the resulting value function error. Empirically, we show that this alignment is a key factor for stable FB learning, particularly at high discount factors where bootstrapping becomes error-prone. Our results identify temporal abstraction as a principled mechanism for shaping the spectral structure of the underlying MDP and enabling effective long-horizon representations in continuous control.
>
---
#### [new 032] MeanFlow Meets Control: Scaling Sampled-Data Control for Swarms
- **分类: cs.LG; cs.MA; cs.RO; eess.SY**

- **简介: 该论文属于控制任务，解决大规模群体在有限控制更新下的导航问题。通过学习有限时间最优控制系数，实现高效、符合实际采样控制结构的群体控制。**

- **链接: [https://arxiv.org/pdf/2603.20189](https://arxiv.org/pdf/2603.20189)**

> **作者:** Anqi Dong; Yongxin Chen; Karl H. Johansson; Johan Karlsson
>
> **摘要:** Steering large-scale swarms in only a few control updates is challenging because real systems operate in sampled-data form: control inputs are updated intermittently and applied over finite intervals. In this regime, the natural object is not an instantaneous velocity field, but a finite-window control quantity that captures the system response over each sampling interval. Inspired by MeanFlow, we introduce a control-space learning framework for swarm steering under linear time-invariant dynamics. The learned object is the coefficient that parameterizes the finite-horizon minimum-energy control over each interval. We show that this coefficient admits both an integral representation and a local differential identity along bridge trajectories, which leads to a simple stop-gradient training objective. At implementation time, the learned coefficient is used directly in sampled-data updates, so the prescribed dynamics and actuation map are respected by construction. The resulting framework provides a scalable approach to few-step swarm steering that is consistent with the sampled-data structure of real control systems.
>
---
#### [new 033] Exact and Approximate Convex Reformulation of Linear Stochastic Optimal Control with Chance Constraints
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于优化控制任务，解决 stochastic optimal control 中的 chance constraints 问题。通过凸优化方法精确建模线性约束，提供更优的二次约束近似，提升可行性与最优性。**

- **链接: [https://arxiv.org/pdf/2603.19454](https://arxiv.org/pdf/2603.19454)**

> **作者:** Tanmay Dokania; Yashwanth Kumar Nakka
>
> **备注:** Under Review
>
> **摘要:** In this paper, we present an equivalent convex optimization formulation for discrete-time stochastic linear systems subject to linear chance constraints, alongside a tight convex relaxation for quadratic chance constraints. By lifting the state vector to encode moment information explicitly, the formulation captures linear chance constraints on states and controls across multiple time steps exactly, without conservatism, yielding strict improvements in both feasibility and optimality. For quadratic chance constraints, we derive convex approximations that are provably less conservative than existing methods. We validate the framework on minimum-snap trajectory generation for a quadrotor, demonstrating that the proposed approach remains feasible at noise levels an order of magnitude beyond the operating range of prior formulations.
>
---
#### [new 034] DynFlowDrive: Flow-Based Dynamic World Modeling for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决轨迹预测不准确的问题。提出DynFlowDrive模型，通过流式动态建模提升场景演化预测的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.19675](https://arxiv.org/pdf/2603.19675)**

> **作者:** Xiaolu Liu; Yicong Li; Song Wang; Junbo Chen; Angela Yao; Jianke Zhu
>
> **备注:** 18 pages, 6 figs
>
> **摘要:** Recently, world models have been incorporated into the autonomous driving systems to improve the planning reliability. Existing approaches typically predict future states through appearance generation or deterministic regression, which limits their ability to capture trajectory-conditioned scene evolution and leads to unreliable action planning. To address this, we propose DynFlowDrive, a latent world model that leverages flow-based dynamics to model the transition of world states under different driving actions. By adopting the rectifiedflow formulation, the model learns a velocity field that describes how the scene state changes under different driving actions, enabling progressive prediction of future latent states. Building upon this, we further introduce a stability-aware multi-mode trajectory selection strategy that evaluates candidate trajectories according to the stability of the induced scene transitions. Extensive experiments on the nuScenes and NavSim benchmarks demonstrate consistent improvements across diverse driving frameworks without introducing additional inference overhead. Source code will be abaliable at this https URL.
>
---
#### [new 035] Sense4HRI: A ROS 2 HRI Framework for Physiological Sensor Integration and Synchronized Logging
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，解决ROS 2框架中生理信号集成与同步日志的问题。提出Sense4HRI框架，支持生理数据整合与用户状态评估。**

- **链接: [https://arxiv.org/pdf/2603.19914](https://arxiv.org/pdf/2603.19914)**

> **作者:** Manuel Scheibl; Julian Leichert; Sinem Görmez; Britta Wrede
>
> **备注:** 6 pages, 3 figures, submitted at IEEE RO-MAN 2026
>
> **摘要:** Physiological signals are increasingly relevant to estimate the mental states of users in human-robot interaction (HRI), yet ROS 2-based HRI frameworks still lack reusable support to integrate such data streams in a standardized way. Therefore, we propose Sense4HRI, an adapted framework for human-robot interaction in ROS 2 that integrates physiological measurements and derived user-state indicators. The framework is designed to be extensible, allowing the integration of additional physiological sensors, their interpretation, and multimodal fusion to provide a robust assessment of the mental states of users. In addition, it introduces reusable interfaces for timestamped physiological time-series data and supports synchronized logging of physiological signals together with experiment context, enabling interoperable and traceable multimodal analysis within ROS 2-based HRI systems.
>
---
#### [new 036] A Unified Platform and Quality Assurance Framework for 3D Ultrasound Reconstruction with Robotic, Optical, and Electromagnetic Tracking
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于3D超声重建任务，旨在解决重建精度与可重复性问题。提出一个质量保障框架和开放平台，通过定制幻影评估不同跟踪技术的性能。**

- **链接: [https://arxiv.org/pdf/2603.20077](https://arxiv.org/pdf/2603.20077)**

> **作者:** Lewis Howell; Manisha Waterston; Tze Min Wah; James H. Chandler; James R. McLaughlan
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Three-dimensional (3D) Ultrasound (US) can facilitate diagnosis, treatment planning, and image-guided therapy. However, current studies rarely provide a comprehensive evaluation of volumetric accuracy and reproducibility, highlighting the need for robust Quality Assurance (QA) frameworks, particularly for tracked 3D US reconstruction using freehand or robotic acquisition. This study presents a QA framework for 3D US reconstruction and a flexible open source platform for tracked US research. A custom phantom containing geometric inclusions with varying symmetry properties enables straightforward evaluation of optical, electromagnetic, and robotic kinematic tracking for 3D US at different scanning speeds and insonation angles. A standardised pipeline performs real-time segmentation and 3D reconstruction of geometric targets (DSC = 0.97, FPS = 46) without GPU acceleration, followed by automated registration and comparison with ground-truth geometries. Applying this framework showed that our robotic 3D US achieves state-of-the-art reconstruction performance (DSC-3D = 0.94 +- 0.01, HD95 = 1.17 +- 0.12), approaching the spatial resolution limit imposed by the transducer. This work establishes a flexible experimental platform and a reproducible validation methodology for 3D US reconstruction. The proposed framework enables robust cross-platform comparisons and improved reporting practices, supporting the safe and effective clinical translation of 3D ultrasound in diagnostic and image-guided therapy applications.
>
---
#### [new 037] LoD-Loc v3: Generalized Aerial Localization in Dense Cities using Instance Silhouette Alignment
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 论文提出LoD-Loc v3，解决密集城市中航空视觉定位问题。通过实例轮廓对齐和合成数据增强，提升模型泛化能力和定位精度。**

- **链接: [https://arxiv.org/pdf/2603.19609](https://arxiv.org/pdf/2603.19609)**

> **作者:** Shuaibang Peng; Juelin Zhu; Xia Li; Kun Yang; Maojun Zhang; Yu Liu; Shen Yan
>
> **摘要:** We present LoD-Loc v3, a novel method for generalized aerial visual localization in dense urban environments. While prior work LoD-Loc v2 achieves localization through semantic building silhouette alignment with low-detail city models, it suffers from two key limitations: poor cross-scene generalization and frequent failure in dense building scenes. Our method addresses these challenges through two key innovations. First, we develop a new synthetic data generation pipeline that produces InsLoD-Loc - the largest instance segmentation dataset for aerial imagery to date, comprising 100k images with precise instance building annotations. This enables trained models to exhibit remarkable zero-shot generalization capability. Second, we reformulate the localization paradigm by shifting from semantic to instance silhouette alignment, which significantly reduces pose estimation ambiguity in dense scenes. Extensive experiments demonstrate that LoD-Loc v3 outperforms existing state-of-the-art (SOTA) baselines, achieving superior performance in both cross-scene and dense urban scenarios with a large margin. The project is available at this https URL.
>
---
#### [new 038] Mixed Integer vs. Continuous Model Predictive Controllers for Binary Thruster Control: A Comparative Study
- **分类: eess.SY; cs.RO**

- **简介: 论文比较了混合整数MPC与连续MPC结合调制技术在二进制推进器控制中的性能，旨在提升燃料效率和稳定性，适用于航天器姿态控制任务。**

- **链接: [https://arxiv.org/pdf/2603.19796](https://arxiv.org/pdf/2603.19796)**

> **作者:** Franek Stark; Jakob Middelberg; Shubham Vyas
>
> **备注:** Accepted to CEAS EuroGNC 2026
>
> **摘要:** Binary on/off thrusters are commonly used for spacecraft attitude and position control during proximity operations. However, their discrete nature poses challenges for conventional continuous control methods. The control of these discrete actuators is either explicitly formulated as a mixed-integer optimization problem or handled in a two-layer approach, where a continuous controller's output is converted to binary commands using analog-to digital modulation techniques such as Delta-Sigma-modulation. This paper provides the first systematic comparison between these two paradigms for binary thruster control, contrasting continuous Model Predictive Control (MPC) with Delta-Sigma modulation against direct Mixed-Integer MPC (MIMPC) approaches. Furthermore, we propose a new variant of MPC for binary actuated systems, which is informed using the state of the Delta-Sigma Modulator. The two variations for the continuous MPC along with the MIMPC are evaluated through extensive simulations using ESA's REACSA platform. Results demonstrate that while all approaches perform similarly in high-thrust regimes, MIMPC achieves superior fuel efficiency in low-thrust conditions. Continuous MPC with modulation shows instabilities at higher thrust levels, while binary informed MPC, which incorporates modulator dynamics, improves robustness and reduces the efficiency gap to the MIMPC. It can be seen from the simulated and real-system experiments that MIMPC offers complete stability and fuel efficiency benefits, particularly for resource-constrained missions, while continuous control methods remain attractive for computationally limited applications.
>
---
## 更新

#### [replaced 001] FD-VLA: Force-Distilled Vision-Language-Action Model for Contact-Rich Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出FD-VLA，解决接触密集操作中的力感知问题。通过力蒸馏模块，无需物理传感器即可实现力-aware推理，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.02142](https://arxiv.org/pdf/2602.02142)**

> **作者:** Ruiteng Zhao; Wenshuo Wang; Yicheng Ma; Xiaocong Li; Francis E.H. Tay; Marcelo H. Ang Jr.; Haiyue Zhu
>
> **备注:** ICRA 2026 Accepted
>
> **摘要:** Force sensing is a crucial modality for Vision-Language-Action (VLA) frameworks, as it enables fine-grained perception and dexterous manipulation in contact-rich tasks. We present Force-Distilled VLA (FD-VLA), a novel framework that integrates force awareness into contact-rich manipulation without relying on physical force sensors. The core of our approach is a Force Distillation Module (FDM), which distills force by mapping a learnable query token, conditioned on visual observations and robot states, into a predicted force token aligned with the latent representation of actual force signals. During inference, this distilled force token is injected into the pretrained VLM, enabling force-aware reasoning while preserving the integrity of its vision-language semantics. This design provides two key benefits: first, it allows practical deployment across a wide range of robots that lack expensive or fragile force-torque sensors, thereby reducing hardware cost and complexity; second, the FDM introduces an additional force-vision-state fusion prior to the VLM, which improves cross-modal alignment and enhances perception-action robustness in contact-rich scenarios. Surprisingly, our physical experiments show that the distilled force token outperforms direct sensor force measurements as well as other baselines, which highlights the effectiveness of this force-distilled VLA approach.
>
---
#### [replaced 002] Latent Action Diffusion for Cross-Embodiment Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，解决不同机器人末端执行器间技能迁移困难的问题。通过学习统一的潜在动作空间，实现跨体感操控与多机器人控制。**

- **链接: [https://arxiv.org/pdf/2506.14608](https://arxiv.org/pdf/2506.14608)**

> **作者:** Erik Bauer; Elvis Nava; Robert K. Katzschmann
>
> **备注:** 8 pages, 5 figures. Accepted to the 2026 IEEE International Conference on Robotics & Automation (ICRA). Website: this https URL
>
> **摘要:** End-to-end learning is emerging as a powerful paradigm for robotic manipulation, but its effectiveness is limited by data scarcity and the heterogeneity of action spaces across robot embodiments. In particular, diverse action spaces across different end-effectors create barriers for cross-embodiment learning and skill transfer. We address this challenge through diffusion policies learned in a latent action space that unifies diverse end-effector actions. We first show that we can learn a semantically aligned latent action space for anthropomorphic robotic hands, a human hand, and a parallel jaw gripper using encoders trained with a contrastive loss. Second, we show that by using our proposed latent action space for co-training on manipulation data from different end-effectors, we can utilize a single policy for multi-robot control and obtain up to 25.3% improved manipulation success rates, indicating successful skill transfer despite a significant embodiment gap. Our approach using latent cross-embodiment policies presents a new method to unify different action spaces across embodiments, enabling efficient multi-robot control and data sharing across robot setups. This unified representation significantly reduces the need for extensive data collection for each new robot morphology, accelerates generalization across embodiments, and ultimately facilitates more scalable and efficient robotic learning.
>
---
#### [replaced 003] Data Analogies Enable Efficient Cross-Embodiment Transfer
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在提升跨机器人配置的迁移效果。通过分析不同形式的演示数据，发现数据类比能有效促进迁移，优于单纯增加数据量。**

- **链接: [https://arxiv.org/pdf/2603.06450](https://arxiv.org/pdf/2603.06450)**

> **作者:** Jonathan Yang; Chelsea Finn; Dorsa Sadigh
>
> **备注:** 14 pages, 11 Figures, 6 Tables
>
> **摘要:** Generalist robot policies are trained on demonstrations collected across a wide variety of robots, scenes, and viewpoints. Yet it remains unclear how to best organize and scale such heterogeneous data so that it genuinely improves performance in a given target setting. In this work, we ask: what form of demonstration data is most useful for enabling transfer across robot set-ups? We conduct controlled experiments that vary end-effector morphology, robot platform appearance, and camera perspective, and compare the effects of simply scaling the number of demonstrations against systematically broadening the diversity in different ways. Our simulated experiments show that while perceptual shifts such as viewpoint benefit most from broad diversity, morphology shifts benefit far less from unstructured diversity and instead see the largest gains from data analogies, i.e. paired demonstrations that align scenes, tasks, and/or trajectories across different embodiments. Informed by the simulation results, we improve real-world cross-embodiment transfer success by an average of $22.5\%$ over large-scale, unpaired datasets by changing only the composition of the data.
>
---
#### [replaced 004] Pseudo-Simulation for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶评估任务，旨在解决真实测试安全性和模拟真实性不足的问题。提出伪仿真方法，利用真实数据与合成观测结合，提升评估效果。**

- **链接: [https://arxiv.org/pdf/2506.04218](https://arxiv.org/pdf/2506.04218)**

> **作者:** Wei Cao; Marcel Hallgarten; Tianyu Li; Daniel Dauner; Xunjiang Gu; Caojun Wang; Yakov Miron; Marco Aiello; Hongyang Li; Igor Gilitschenski; Boris Ivanovic; Marco Pavone; Andreas Geiger; Kashyap Chitta
>
> **备注:** CoRL 2025, updated with leaderboard snapshot from March 2026
>
> **摘要:** Existing evaluation paradigms for Autonomous Vehicles (AVs) face critical limitations. Real-world evaluation is often challenging due to safety concerns and a lack of reproducibility, whereas closed-loop simulation can face insufficient realism or high computational costs. Open-loop evaluation, while being efficient and data-driven, relies on metrics that generally overlook compounding errors. In this paper, we propose pseudo-simulation, a novel paradigm that addresses these limitations. Pseudo-simulation operates on real datasets, similar to open-loop evaluation, but augments them with synthetic observations generated prior to evaluation using 3D Gaussian Splatting. Our key idea is to approximate potential future states the AV might encounter by generating a diverse set of observations that vary in position, heading, and speed. Our method then assigns a higher importance to synthetic observations that best match the AV's likely behavior using a novel proximity-based weighting scheme. This enables evaluating error recovery and the mitigation of causal confusion, as in closed-loop benchmarks, without requiring sequential interactive simulation. We show that pseudo-simulation is better correlated with closed-loop simulations ($R^2=0.8$) than the best existing open-loop approach ($R^2=0.7$). We also establish a public leaderboard for the community to benchmark new methodologies with pseudo-simulation. Our code is available at this https URL.
>
---
#### [replaced 005] Learning Discrete Abstractions for Visual Rearrangement Tasks Using Vision-Guided Graph Coloring
- **分类: cs.RO**

- **简介: 该论文研究视觉重排任务，旨在自动学习离散抽象表示。通过结合结构约束与视觉距离，提出一种图着色方法，提升规划效率。**

- **链接: [https://arxiv.org/pdf/2509.14460](https://arxiv.org/pdf/2509.14460)**

> **作者:** Abhiroop Ajith; Constantinos Chamzas
>
> **摘要:** Learning abstractions directly from data is a core challenge in robotics. Humans naturally operate at an abstract level, reasoning over high-level subgoals while delegating execution to low-level motor skills -- an ability that enables efficient problem solving in complex environments. In robotics, abstractions and hierarchical reasoning have long been central to planning, yet they are typically hand-engineered, demanding significant human effort and limiting scalability. Automating the discovery of useful abstractions directly from visual data would make planning frameworks more scalable and more applicable to real-world robotic domains. In this work, we focus on rearrangement tasks where the state is represented with raw images, and propose a method to induce discrete, graph-structured abstractions by combining structural constraints with an attention-guided visual distance. Our approach leverages the inherent bipartite structure of rearrangement problems, integrating structural constraints and visual embeddings into a unified framework. This enables the autonomous discovery of abstractions from vision alone, which can subsequently support high-level planning. We evaluate our method on two rearrangement tasks in simulation and show that it consistently identifies meaningful abstractions that facilitate effective planning and outperform existing approaches.
>
---
#### [replaced 006] CoInfra: A Large-Scale Cooperative Infrastructure Perception System and Dataset for Vehicle-Infrastructure Cooperation in Adverse Weather
- **分类: cs.RO**

- **简介: 该论文提出CoInfra系统与数据集，解决V2I协同感知在恶劣天气下的性能评估问题，通过多节点传感器和5G通信实现精准融合。**

- **链接: [https://arxiv.org/pdf/2507.02245](https://arxiv.org/pdf/2507.02245)**

> **作者:** Minghao Ning; Yufeng Yang; Keqi Shu; Shucheng Huang; Jiaming Zhong; Maryam Salehi; Mahdi Rahmani; Jiaming Guo; Yukun Lu; Chen Sun; Aladdin Saleh; Ehsan Hashemi; Amir Khajepour
>
> **备注:** This paper has been submitted to the Transportation Research Part C: Emerging Technologies for review
>
> **摘要:** Vehicle-infrastructure (V2I) cooperative perception can substantially extend the range, coverage, and robustness of autonomous driving systems beyond the limits of onboard-only sensing, particularly in occluded and adverse-weather environments. However, its practical value is still difficult to quantify because existing benchmarks do not adequately capture large-scale multi-node deployments, realistic communication conditions, and adverse-weather operation. This paper presents CoInfra, a deployable cooperative infrastructure perception platform comprising 14 roadside sensor nodes connected through a commercial 5G network, together with a large-scale dataset and an open-source system stack for V2I cooperation research. The system supports synchronized multi-node sensing and delay-aware fusion under real 5G communication constraints. The released dataset covers an eight-node urban roundabout under four weather conditions (sunny, rainy, heavy snow, and freezing rain) and contains 294k LiDAR frames, 589k camera images, and 332k globally consistent 3D bounding boxes. It also includes a synchronized V2I subset collected with an autonomous vehicle. Beyond standard perception benchmarks, we further evaluate whether infrastructure sensing improves awareness of safety-critical traffic participants during roundabout interactions. In structured conflict scenarios, V2I cooperation increases critical-frame completeness from 33%-46% with vehicle-only sensing to 86%-100%. These results show that multi-node infrastructure perception can significantly improve situational awareness in conflict-rich traffic scenarios where vehicle-only sensing is most limited.
>
---
#### [replaced 007] Path Integral Particle Filtering for Hybrid Systems via Saltation Matrices
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决混合系统在接触过程中的不确定性传播问题，提出基于路径积分的粒子滤波方法，利用盐化矩阵提升估计鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.01176](https://arxiv.org/pdf/2603.01176)**

> **作者:** Karthik Shaji; Sreeranj Jayadevan; Bo Yuan; Hongzhe Yu; Yongxin Chen
>
> **摘要:** State estimation for hybrid systems that undergo intermittent contact with their environments, such as extraplanetary robots and satellites undergoing docking operations, is difficult due to the discrete uncertainty propagation during contact. To handle this propagation, this paper presents an optimal-control-based particle filtering method that leverages saltation matrices to map out uncertainty propagation during contact events. By exploiting a path integral filtering framework that exploits the duality between smoothing and optimal control, the resulting state estimation algorithm is robust to outlier effects, flexible to non-Gaussian noise distributions, and handles challenging contact dynamics in hybrid systems. To evaluate the validity and consistency of the proposed approach, this paper tests it against strong baselines on the stochastic dynamics generated by a bouncing ball and spring loaded inverted pendulum.
>
---
#### [replaced 008] Multi-Robot Coordination for Planning under Context Uncertainty
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多机器人在上下文不确定环境中的协同规划问题。针对未知上下文导致的决策偏差，提出两阶段方法，先协同推断上下文，再按优先级进行路径规划，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2603.13748](https://arxiv.org/pdf/2603.13748)**

> **作者:** Pulkit Rustagi; Kyle Hollins Wray; Sandhya Saisubramanian
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Real-world robots often operate in settings where objective priorities depend on the underlying context of operation. When the underlying context is unknown apriori, multiple robots may have to coordinate to gather informative observations to infer the context, since acting based on an incorrect context can lead to misaligned and unsafe behavior. Once the underlying true context is inferred, the robots optimize their task-specific objectives in the preference order induced by the context. We formalize this problem as a Multi-Robot Context-Uncertain Stochastic Shortest Path (MR-CUSSP), which captures context-relevant information at landmark states through joint observations. Our two-stage solution approach is composed of: (1) CIMOP (Coordinated Inference for Multi-Objective Planning) to compute plans that guide robots toward informative landmarks to efficiently infer the true context, and (2) LCBS (Lexicographic Conflict-Based Search) for collision-free multi-robot path planning with lexicographic objective preferences, induced by the context. We evaluate the algorithms using three simulated domains and demonstrate its practical applicability using five mobile robots in the salp domain setup.
>
---
#### [replaced 009] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉与语言导航任务，旨在解决长期记忆构建和误差累积问题。提出DecoVLN框架，通过优化记忆选择和修正策略提升导航性能。**

- **链接: [https://arxiv.org/pdf/2603.13133](https://arxiv.org/pdf/2603.13133)**

> **作者:** Zihao Xin; Wentong Li; Yixuan Jiang; Bin Wang; Runming Cong; Jie Qin; Shengjun Huang
>
> **备注:** 16 pages, 8 figures, CVPR2026
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to follow long-horizon instructions and navigate complex 3D environments. However, existing approaches face two major challenges: constructing an effective long-term memory bank and overcoming the compounding errors problem. To address these issues, we propose DecoVLN, an effective framework designed for robust streaming perception and closed-loop control in long-horizon navigation. First, we formulate long-term memory construction as an optimization problem and introduce adaptive refinement mechanism that selects frames from a historical candidate pool by iteratively optimizing a unified scoring function. This function jointly balances three key criteria: semantic relevance to the instruction, visual diversity from the selected memory, and temporal coverage of the historical trajectory. Second, to alleviate compounding errors, we introduce a state-action pair-level corrective finetuning strategy. By leveraging geodesic distance between states to precisely quantify deviation from the expert trajectory, the agent collects high-quality state-action pairs in the trusted region while filtering out the polluted data with low relevance. This improves both the efficiency and stability of error correction. Extensive experiments demonstrate the effectiveness of DecoVLN, and we have deployed it in real-world environments.
>
---
#### [replaced 010] RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出RobotArena ∞，用于机器人通用能力的基准测试。解决真实世界测试成本高、不安全等问题，通过模拟环境与人类反馈结合，实现高效、可扩展的评估。**

- **链接: [https://arxiv.org/pdf/2510.23571](https://arxiv.org/pdf/2510.23571)**

> **作者:** Yash Jangir; Yidi Zhang; Pang-Chi Lo; Kashu Yamazaki; Chenyu Zhang; Kuan-Hsun Tu; Tsung-Wei Ke; Lei Ke; Yonatan Bisk; Katerina Fragkiadaki
>
> **备注:** Website: this https URL
>
> **摘要:** The pursuit of robot generalists, agents capable of performing diverse tasks across diverse environments, demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. We introduce RobotArena Infinity, a new benchmarking framework that overcomes these challenges by shifting vision-language-action (VLA) evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated vision-language-model-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference comparisons. To measure robustness, we systematically perturb simulated environments along multiple axes, including textures and object placements, stress-testing policy generalization under controlled variation. The result is a continuously evolving, reproducible, and scalable benchmark for real-world-trained robot manipulation policies, addressing a critical missing capability in today's robotics landscape.
>
---
#### [replaced 011] R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于模型基础强化学习任务，旨在解决图像表示中冗余信息的问题。提出R2-Dreamer框架，通过自监督目标减少冗余，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.18202](https://arxiv.org/pdf/2603.18202)**

> **作者:** Naoki Morihira; Amal Nahar; Kartik Bharadwaj; Yasuhiro Kato; Akinobu Hayashi; Tatsuya Harada
>
> **备注:** 20 pages, 12 figures, 2 tables
>
> **摘要:** A central challenge in image-based Model-Based Reinforcement Learning (MBRL) is to learn representations that distill essential information from irrelevant visual details. While promising, reconstruction-based methods often waste capacity on large task-irrelevant regions. Decoder-free methods instead learn robust representations by leveraging Data Augmentation (DA), but reliance on such external regularizers limits versatility. We propose R2-Dreamer, a decoder-free MBRL framework with a self-supervised objective that serves as an internal regularizer, preventing representation collapse without resorting to DA. The core of our method is a redundancy-reduction objective inspired by Barlow Twins, which can be easily integrated into existing frameworks. On DeepMind Control Suite and Meta-World, R2-Dreamer is competitive with strong baselines such as DreamerV3 and TD-MPC2 while training 1.59x faster than DreamerV3, and yields substantial gains on DMC-Subtle with tiny task-relevant objects. These results suggest that an effective internal regularizer can enable versatile, high-performance decoder-free MBRL. Code is available at this https URL.
>
---
#### [replaced 012] From Vocal Instructions to Household Tasks: The Inria TIAGo++ in the euROBIN Service Robots Coopetition
- **分类: cs.RO**

- **简介: 该论文介绍了一种用于服务机器人任务的集成系统，解决语音指令到家庭任务的转化问题。工作包括开发修改的TIAGo++平台和基于LLM的任务规划管道。**

- **链接: [https://arxiv.org/pdf/2412.17861](https://arxiv.org/pdf/2412.17861)**

> **作者:** Fabio Amadio; Clemente Donoso; Dionis Totsila; Raphael Lorenzo; Quentin Rouxel; Olivier Rochel; Enrico Mingo Hoffman; Jean-Baptiste Mouret; Serena Ivaldi
>
> **摘要:** This paper describes the Inria team's integrated robotics system used in the 1st euROBIN coopetition, during which service robots performed voice-activated household tasks in a kitchen setting. The team developed a modified TIAGo++ platform that leverages a whole-body control stack for autonomous and teleoperated modes, and an LLM-based pipeline for instruction understanding and task planning. The key contributions (opens-sourced) are the integration of these components and the design of custom teleoperation devices, addressing practical challenges in the deployment of service robots.
>
---
#### [replaced 013] Uncertainty-Aware Multi-Robot Task Allocation With Strongly Coupled Inter-Robot Rewards
- **分类: cs.RO**

- **简介: 该论文研究多机器人任务分配问题，针对不确定任务需求提出一种基于拍卖的算法，以提高任务完成效率和价值。**

- **链接: [https://arxiv.org/pdf/2509.22469](https://arxiv.org/pdf/2509.22469)**

> **作者:** Ben Rossano; Jaein Lim; Jonathan P. How
>
> **备注:** 9 pages
>
> **摘要:** Allocating tasks to heterogeneous robot teams in environments with uncertain task requirements is a fundamentally challenging problem. Redundantly assigning multiple robots to such tasks is overly conservative, while purely reactive strategies risk costly delays in task completion when the uncertain capabilities become necessary. This paper introduces an auction-based task allocation algorithm that explicitly models uncertain task requirements, leveraging a novel strongly coupled formulation to allocate tasks such that robots with potentially required capabilities are naturally positioned near uncertain tasks. This approach enables robots to remain productive on nearby tasks while simultaneously mitigating large delays in completion time when their capabilities are required. Through a set of simulated disaster relief missions with task deadline constraints, we demonstrate that the proposed approach yields up to a 15% increase in expected mission value compared to redundancy-based methods. Furthermore, we propose a novel framework to approximate uncertainty arising from unmodeled changes in task requirements by leveraging the natural delay between encountering unexpected environmental conditions and confirming whether additional capabilities are required to complete a task. We show that our approach achieves up to an 18% increase in expected mission value using this framework compared to reactive methods that don't leverage this delay.
>
---
#### [replaced 014] Adaptive Relative Pose Estimation Framework with Dual Noise Tuning for Safe Approaching Maneuvers
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于相对姿态估计任务，旨在解决主动碎片清除任务中对翻滚卫星的精确导航问题。通过融合深度学习与自适应滤波技术，提升姿态估计的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2507.16214](https://arxiv.org/pdf/2507.16214)**

> **作者:** Batu Candan; Murat Berke Oktay; Simone Servadio
>
> **摘要:** Accurate and robust relative pose estimation is crucial for enabling challenging Active Debris Removal (ADR) missions targeting tumbling derelict satellites such as ESA's ENVISAT. This work presents a complete pipeline integrating advanced computer vision techniques with adaptive nonlinear filtering to address this challenge. A Convolutional Neural Network (CNN), enhanced with image preprocessing, detects structural markers (corners) from chaser imagery, whose 2D coordinates are converted to 3D measurements using camera modeling. These measurements are fused within an Unscented Kalman Filter (UKF) framework, selected for its ability to handle nonlinear relative dynamics, to estimate the full relative pose. Key contributions include the integrated system architecture and a dual adaptive strategy within the UKF: dynamic tuning of the measurement noise covariance compensates for varying CNN measurement uncertainty, while adaptive tuning of the process noise covariance, utilizing measurement residual analysis, accounts for unmodeled dynamics or maneuvers online. This dual adaptation enhances robustness against both measurement imperfections and dynamic model uncertainties. The performance of the proposed adaptive integrated system is evaluated through high-fidelity simulations using a realistic ENVISAT model, comparing estimates against ground truth under various conditions, including measurement outages. This comprehensive approach offers an enhanced solution for robust onboard relative navigation, significantly advancing the capabilities required for safe proximity operations during ADR missions.
>
---
#### [replaced 015] Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人任务规划领域，解决广义旅行商问题（GTSP）。通过多模态融合学习框架，提升规划效率与准确性。**

- **链接: [https://arxiv.org/pdf/2506.16931](https://arxiv.org/pdf/2506.16931)**

> **作者:** Jiaqi Cheng; Mingfeng Fan; Xuefeng Zhang; Jingsong Liang; Yuhong Cao; Guohua Wu; Guillaume Adrien Sartoretti
>
> **备注:** 14 pages, 6 figures, Proceedings of the Conference on Robot Learning (CoRL 2025)
>
> **摘要:** Effective and efficient task planning is essential for mobile robots, especially in applications like warehouse retrieval and environmental monitoring. These tasks often involve selecting one location from each of several target clusters, forming a Generalized Traveling Salesman Problem (GTSP) that remains challenging to solve both accurately and efficiently. To address this, we propose a Multimodal Fused Learning (MMFL) framework that leverages both graph and image-based representations to capture complementary aspects of the problem, and learns a policy capable of generating high-quality task planning schemes in real time. Specifically, we first introduce a coordinate-based image builder that transforms GTSP instances into spatially informative representations. We then design an adaptive resolution scaling strategy to enhance adaptability across different problem scales, and develop a multimodal fusion module with dedicated bottlenecks that enables effective integration of geometric and spatial features. Extensive experiments show that our MMFL approach significantly outperforms state-of-the-art methods across various GTSP instances while maintaining the computational efficiency required for real-time robotic applications. Physical robot tests further validate its practical effectiveness in real-world scenarios.
>
---
#### [replaced 016] Feasibility Analysis and Constraint Selection in Optimization-Based Controllers
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属于控制优化任务，解决约束可行性分析与选择问题。通过理论分析和算法设计，提升自主系统控制的效率与可行性。**

- **链接: [https://arxiv.org/pdf/2505.05502](https://arxiv.org/pdf/2505.05502)**

> **作者:** Panagiotis Rousseas; Haejoon Lee; Dimos V. Dimarogonas; Dimitra Panagou
>
> **备注:** 13 pages, 4 figures, submitted to IEEE Transactions on Automatic Control
>
> **摘要:** Control synthesis under constraints is at the forefront of research on autonomous systems, in part due to its broad application from low-level control to high-level planning, where computing control inputs is typically cast as a constrained optimization problem. Assessing feasibility of the constraints and selecting among subsets of feasible constraints is a challenging yet crucial problem. In this work, we provide a novel theoretical analysis that yields necessary and sufficient conditions for feasibility assessment of linear constraints and based on this analysis, we develop novel methods for feasible constraint selection in the context of control of autonomous systems. Through a series of simulations, we demonstrate that our algorithms achieve performance comparable to state-of-the-art methods while offering improved computational efficiency. Importantly, our analysis provides a novel theoretical framework for assessing, analyzing and handling constraint infeasibility.
>
---
#### [replaced 017] SG-CoT: An Ambiguity-Aware Robotic Planning Framework using Scene Graph Representations
- **分类: cs.RO**

- **简介: 该论文提出SG-CoT框架，解决机器人规划中的歧义问题。通过场景图与大语言模型结合，提升规划可靠性与准确性。属于机器人自主决策任务。**

- **链接: [https://arxiv.org/pdf/2603.18271](https://arxiv.org/pdf/2603.18271)**

> **作者:** Akshat Rana; Peeyush Agarwal; K.P.S. Rana; Amarjit Malhotra
>
> **备注:** This work has been submitted to the IEEE Robotics and Automation Letters for possible publication
>
> **摘要:** Ambiguity poses a major challenge to large language models (LLMs) used as robotic planners. In this letter, we present Scene Graph-Chain-of-Thought (SG-CoT), a two-stage framework where LLMs iteratively query a scene graph representation of the environment to detect and clarify ambiguities. First, a structured scene graph representation of the environment is constructed from input observations, capturing objects, their attributes, and relationships with other objects. Second, the LLM is equipped with retrieval functions to query portions of the scene graph that are relevant to the provided instruction. This grounds the reasoning process of the LLM in the observation, increasing the reliability of robotic planners under ambiguous situations. SG-CoT also allows the LLM to identify the source of ambiguity and pose a relevant disambiguation question to the user or another robot. Extensive experimentation demonstrates that SG-CoT consistently outperforms prior methods, with a minimum of 10% improvement in question accuracy and a minimum success rate increase of 4% in single-agent and 15% in multi-agent environments, validating its effectiveness for more generalizable robot planning.
>
---
#### [replaced 018] SpikeGrasp: A Benchmark for 6-DoF Grasp Pose Detection from Stereo Spike Streams
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SpikeGrasp，用于6-DoF抓取位姿检测的任务。解决传统系统依赖点云计算的问题，通过神经启发方法直接处理立体事件流，提升在复杂场景中的性能与效率。**

- **链接: [https://arxiv.org/pdf/2510.10602](https://arxiv.org/pdf/2510.10602)**

> **作者:** Zhuoheng Gao; Jiyao Zhang; Zhiyong Xie; Hao Dong; Zhaofei Yu; Rongmei Chen; Guozhang Chen; Tiejun Huang
>
> **备注:** Some real machine experiments need to be supplemented, and the entire paper is incomplete
>
> **摘要:** Most robotic grasping systems rely on converting sensor data into explicit 3D point clouds, which is a computational step not found in biological intelligence. This paper explores a fundamentally different, neuro-inspired paradigm for 6-DoF grasp detection. We introduce SpikeGrasp, a framework that mimics the biological visuomotor pathway, processing raw, asynchronous events from stereo spike cameras, similarly to retinas, to directly infer grasp poses. Our model fuses these stereo spike streams and uses a recurrent spiking neural network, analogous to high-level visual processing, to iteratively refine grasp hypotheses without ever reconstructing a point cloud. To validate this approach, we built a large-scale synthetic benchmark dataset. Experiments show that SpikeGrasp surpasses traditional point-cloud-based baselines, especially in cluttered and textureless scenes, and demonstrates remarkable data efficiency. By establishing the viability of this end-to-end, neuro-inspired approach, SpikeGrasp paves the way for future systems capable of the fluid and efficient manipulation seen in nature, particularly for dynamic objects.
>
---
#### [replaced 019] FORWARD: Dataset of a forwarder operating in rough terrain
- **分类: cs.RO; cs.AI; cs.CE; cs.LG; physics.app-ph**

- **简介: 该论文介绍FORWARD数据集，用于研究森林机械的自主控制与感知。任务是提升森林作业效率与安全性，通过高精度多模态数据支持算法开发与仿真。**

- **链接: [https://arxiv.org/pdf/2511.17318](https://arxiv.org/pdf/2511.17318)**

> **作者:** Mikael Lundbäck; Erik Wallin; Carola Häggström; Mattias Nyström; Andreas Grönlund; Mats Richardson; Petrus Jönsson; William Arnvik; Lucas Hedström; Arvid Fälldin; Martin Servin
>
> **备注:** 33 pages, 24 figures
>
> **摘要:** We present FORWARD, a high-resolution multimodal dataset of a cut-to-length forwarder operating in rough terrain on two harvest sites in the middle part of Sweden. The forwarder is a large Komatsu model equipped with vehicle telematics sensors, including global positioning via satellite navigation, movement sensors, accelerometers, and engine sensors. The forwarder was additionally equipped with cameras, operator vibration sensors, and multiple IMUs. The data includes event time logs recorded at 5 Hz of driving speed, fuel consumption, machine position with centimeter accuracy, and crane use while the forwarder operates in forest areas, aerially laser-scanned with a resolution of around 1500 points per square meter. Production log files (Stanford standard) with time-stamped machine events, extensive video material, and terrain data in various formats are included as well. About 18 hours of regular wood extraction work during three days is annotated from 360-video material into individual work elements and included in the dataset. We also include scenario specifications of conducted experiments on forest roads and in terrain. Scenarios include repeatedly driving the same routes with and without steel tracks, different load weights, and different target driving speeds. The dataset is intended for developing models and algorithms for trafficability, perception, and autonomous control of forest machines using artificial intelligence, simulation, and experiments on physical testbeds. In part, we focus on forwarders traversing terrain, avoiding or handling obstacles, and loading or unloading logs, with consideration for efficiency, fuel consumption, safety, and environmental impact. Other benefits of the open dataset include the ability to explore auto-generation and calibration of forestry machine simulators and automation scenario descriptions using the data recorded in the field.
>
---
#### [replaced 020] CageDroneRF: A Large-Scale RF Benchmark and Toolkit for Drone Perception
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出CDRF，一个大规模RF无人机检测与识别基准，解决数据稀缺和多样性不足的问题，通过真实数据与合成数据结合，提供工具链支持标准化评估。**

- **链接: [https://arxiv.org/pdf/2601.03302](https://arxiv.org/pdf/2601.03302)**

> **作者:** Mohammad Rostami; Atik Faysal; Hongtao Xia; Hadi Kasasbeh; Ziang Gao; Huaxia Wang
>
> **摘要:** We present CageDroneRF (CDRF), a large-scale benchmark for Radio-Frequency (RF) drone detection and identification built from real-world captures and systematically generated synthetic variants. CDRF addresses the scarcity and limited diversity of existing RF datasets by coupling extensive raw recordings with a principled augmentation pipeline that (i)~precisely controls Signal-to-Noise Ratio (SNR), (ii)~injects interfering emitters, and (iii)~applies frequency shifts with label-consistent bounding-box recomputation for detection. The dataset spans a wide range of contemporary drone models, many of which are unavailable in current public datasets, and diverse acquisition conditions, derived from data collected at the Rowan University campus and within a controlled RF-cage facility. CDRF is released with interoperable open-source tools for data generation, preprocessing, augmentation, and evaluation that also operate on existing public benchmarks. It enables standardized benchmarking for classification, open-set recognition, and object detection, supporting rigorous comparisons and reproducible pipelines. By releasing this comprehensive benchmark and tooling, we aim to accelerate progress toward robust, generalizable RF perception models.
>
---
#### [replaced 021] ReMAP-DP: Reprojected Multi-view Aligned PointMaps for Diffusion Policy
- **分类: cs.RO**

- **简介: 该论文提出ReMAP-DP框架，解决机器人操作中2D视觉与3D空间感知的融合问题，通过多视角对齐点云与扩散策略提升精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.14977](https://arxiv.org/pdf/2603.14977)**

> **作者:** Xinzhang Yang; Renjun Wu; Jinyan Liu; Xuesong Li
>
> **备注:** fix some typos
>
> **摘要:** Generalist robot policies built upon 2D visual representations excel at semantic reasoning but inherently lack the explicit 3D spatial awareness required for high-precision tasks. Existing 3D integration methods struggle to bridge this gap due to the structural irregularity of sparse point clouds and the geometric distortion introduced by multi-view orthographic rendering. To overcome these barriers, we present ReMAP-DP, a novel framework synergizing standardized perspective reprojection with a structure-aware dual-stream diffusion policy. By coupling the re-projected views with pixel-aligned PointMaps, our dual-stream architecture leverages learnable modality embeddings to fuse frozen semantic features and explicit geometric descriptors, ensuring precise implicit patch-level alignment. Extensive experiments across simulation and real-world environments demonstrate ReMAP-DP's superior performance in diverse manipulation tasks. On RoboTwin 2.0, it attains a 59.3% average success rate, outperforming the DP3 baseline by +6.6%. On ManiSkill 3, our method yields a 28% improvement over DP3 on the geometrically challenging Stack Cube task. Furthermore, ReMAP-DP exhibits remarkable real-world robustness, executing high-precision and dynamic manipulations with superior data efficiency from only a handful of demonstrations. Project page is available at: this https URL
>
---
#### [replaced 022] Task-Specified Compliance Bounds for Humanoids via Lipschitz-Constrained Policies
- **分类: cs.RO**

- **简介: 该论文研究人形机器人运动控制任务，解决RL方法难以实现定量合规性的问题。提出ALCP方法，通过状态依赖的约束提升稳定性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.16180](https://arxiv.org/pdf/2603.16180)**

> **作者:** Zewen He; Yoshihiko Nakamura
>
> **备注:** Submitted to IEEE for possible publication, under review
>
> **摘要:** Reinforcement learning (RL) has demonstrated substantial potential for humanoid bipedal locomotion and the control of complex motions. To cope with oscillations and impacts induced by environmental interactions, compliant control is widely regarded as an effective remedy. However, the model-free nature of RL makes it difficult to impose task-specified and quantitatively verifiable compliance objectives, and classical model-based stiffness designs are not directly applicable. Lipschitz-Constrained Policies (LCP), which regularize the local sensitivity of a policy via gradient penalties, have recently been used to smooth humanoid motions. Nevertheless, existing LCP-based methods typically employ a single scalar Lipschitz budget and lack an explicit connection to physically meaningful compliance specifications in real-world systems. In this study, we propose an anisotropic Lipschitz-constrained policy (ALCP) that maps a task-space stiffness upper bound to a state-dependent Lipschitz-style constraint on the policy Jacobian. The resulting constraint is enforced during RL training via a hinge-squared spectral-norm penalty, preserving physical interpretability while enabling direction-dependent compliance. Experiments on humanoid robots show that ALCP improves locomotion stability and impact robustness, while reducing oscillations and energy usage.
>
---
#### [replaced 023] TeleDex: Accessible Dexterous Teleoperation
- **分类: cs.RO**

- **简介: 该论文提出TeleDex系统，解决机器人操作政策泛化能力不足的问题。通过手机实现灵活的远程操作，降低硬件门槛，便于快速收集演示数据以优化策略。属于机器人远程操作任务。**

- **链接: [https://arxiv.org/pdf/2603.17065](https://arxiv.org/pdf/2603.17065)**

> **作者:** Omar Rayyan; Maximilian Gilles; Yuchen Cui
>
> **备注:** For project website and videos, see this https URL
>
> **摘要:** Despite increasing dataset scale and model capacity, robot manipulation policies still struggle to generalize beyond their training distributions. As a result, deploying state-of-the-art policies in new environments, tasks, or robot embodiments often requires collecting additional demonstrations. Enabling this in real-world deployment settings requires tools that allow users to collect demonstrations quickly, affordably, and with minimal setup. We present TeleDex, an open-source system for intuitive teleoperation of dexterous hands and robotic manipulators using any readily available phone. The system streams low-latency 6-DoF wrist poses and articulated 21-DoF hand state estimates from the phone, which are retargeted to robot arms and multi-fingered hands without requiring external tracking infrastructure. TeleDex supports both a handheld phone-only mode and an optional 3D-printable hand-mounted interface for finger-level teleoperation. By lowering the hardware and setup barriers to dexterous teleoperation, TeleDex enables users to quickly collect demonstrations during deployment to support policy fine-tuning. We evaluate the system across simulation and real-world manipulation tasks, demonstrating its effectiveness as a unified scalable interface for robot teleoperation. All software and hardware designs, along with demonstration videos, are open-source and available at this http URL.
>
---
#### [replaced 024] EgoSpot:Egocentric Multimodal Control for Hands-Free Mobile Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决肢体障碍者难以使用传统控制器的问题。通过融合眼动、头部动作和语音指令，实现手部自由的机器人操作控制。**

- **链接: [https://arxiv.org/pdf/2306.02393](https://arxiv.org/pdf/2306.02393)**

> **作者:** Ganlin Zhang; Deheng Zhang; Longteng Duan; Guo Han; Yuqian Fu; Danda Pani Paudel; Luc Van Gool; Eric Vollenweider
>
> **摘要:** We propose a novel hands-free control framework for the Boston Dynamics Spot robot using the Microsoft HoloLens 2 mixed-reality headset. Enabling accessible robot control is critical for allowing individuals with physical disabilities to benefit from robotic assistance in daily activities, teleoperation, and remote interaction tasks. However, most existing robot control interfaces rely on manual input devices such as joysticks or handheld controllers, which can be difficult or impossible for users with limited motor capabilities. To address this limitation, we develop an intuitive multimodal control system that leverages egocentric sensing from a wearable device. Our system integrates multiple control signals, including eye gaze, head gestures, and voice commands, to enable hands-free interaction. These signals are fused to support real-time control of both robot locomotion and arm manipulation. Experimental results show that our approach achieves performance comparable to traditional joystick-based control in terms of task completion time and user experience, while significantly improving accessibility and naturalness of interaction. Our results highlight the potential of egocentric multimodal interfaces to make mobile manipulation robots more inclusive and usable for a broader population. A demonstration of the system is available on our project webpage.
>
---
#### [replaced 025] Mash, Spread, Slice! Learning to Manipulate Object States via Visual Spatial Progress
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SPARTA框架，解决物体状态变化的机器人操作任务，如挤压、涂抹和切割。通过视觉空间进展分析，提升控制精度与效率。**

- **链接: [https://arxiv.org/pdf/2509.24129](https://arxiv.org/pdf/2509.24129)**

> **作者:** Priyanka Mandikal; Jiaheng Hu; Shivin Dass; Sagnik Majumder; Roberto Martín-Martín; Kristen Grauman
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Most robot manipulation focuses on changing the kinematic state of objects: picking, placing, opening, or rotating them. However, a wide range of real-world manipulation tasks involve a different class of object state change--such as mashing, spreading, or slicing--where the object's physical and visual state evolve progressively without necessarily changing its position. We present SPARTA, the first unified framework for the family of object state change manipulation tasks. Our key insight is that these tasks share a common structural pattern: they involve spatially-progressing, object-centric changes that can be represented as regions transitioning from an actionable to a transformed state. Building on this insight, SPARTA integrates spatially progressing object change segmentation maps, a visual skill to perceive actionable vs. transformed regions for specific object state change tasks, to generate a) structured policy observations that strip away appearance variability, and b) dense rewards that capture incremental progress over time. These are leveraged in two SPARTA policy variants: reinforcement learning for fine-grained control without demonstrations or simulation; and greedy control for fast, lightweight deployment. We validate SPARTA on a real robot for three challenging tasks across 10 diverse real-world objects, achieving significant improvements in training time and accuracy over sparse rewards and visual goal-conditioned baselines. Our results highlight progress-aware visual representations as a versatile foundation for the broader family of object state manipulation tasks. Project website: this https URL
>
---
#### [replaced 026] World4RL: Diffusion World Models for Policy Refinement with Reinforcement Learning for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出World4RL框架，用于机器人操作策略优化。解决专家数据不足与真实训练成本高的问题，通过扩散模型构建高保真环境进行策略精调。**

- **链接: [https://arxiv.org/pdf/2509.19080](https://arxiv.org/pdf/2509.19080)**

> **作者:** Zhennan Jiang; Kai Liu; Yuxin Qin; Shuai Tian; Yupeng Zheng; Mingcai Zhou; Chao Yu; Haoran Li; Dongbin Zhao
>
> **摘要:** Robotic manipulation policies are commonly initialized through imitation learning, but their performance is limited by the scarcity and narrow coverage of expert data. Reinforcement learning can refine polices to alleviate this limitation, yet real-robot training is costly and unsafe, while training in simulators suffers from the sim-to-real gap. Recent advances in generative models have demonstrated remarkable capabilities in real-world simulation, with diffusion models in particular excelling at generation. This raises the question of how diffusion model-based world models can be combined to enhance pre-trained policies in robotic manipulation. In this work, we propose World4RL, a framework that employs diffusion-based world models as high-fidelity simulators to refine pre-trained policies entirely in imagined environments for robotic manipulation. Unlike prior works that primarily employ world models for planning, our framework enables direct end-to-end policy optimization. World4RL is designed around two principles: pre-training a diffusion world model that captures diverse dynamics on multi-task datasets and refining policies entirely within a frozen world model to avoid online real-world interactions. We further design a two-hot action encoding scheme tailored for robotic manipulation and adopt diffusion backbones to improve modeling fidelity. Extensive simulation and real-world experiments demonstrate that World4RL provides high-fidelity environment modeling and enables consistent policy refinement, yielding significantly higher success rates compared to imitation learning and other baselines.
>
---
