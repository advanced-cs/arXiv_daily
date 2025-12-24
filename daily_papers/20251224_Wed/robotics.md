# 机器人 cs.RO

- **最新发布 20 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] KnowVal: A Knowledge-Augmented and Value-Guided Autonomous Driving System
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出KnowVal系统，面向自动驾驶任务，解决现有数据驱动方法难以建模复杂逻辑与价值对齐的问题。工作包括：构建驾驶知识图谱、设计LLM驱动的知识检索机制、建立人类偏好数据集并训练价值模型，以实现知识增强、语言推理与价值引导的协同规划。**

- **链接: [https://arxiv.org/pdf/2512.20299v1](https://arxiv.org/pdf/2512.20299v1)**

> **作者:** Zhongyu Xia; Wenhao Chen; Yongtao Wang; Ming-Hsuan Yang
>
> **摘要:** Visual-language reasoning, driving knowledge, and value alignment are essential for advanced autonomous driving systems. However, existing approaches largely rely on data-driven learning, making it difficult to capture the complex logic underlying decision-making through imitation or limited reinforcement rewards. To address this, we propose KnowVal, a new autonomous driving system that enables visual-language reasoning through the synergistic integration of open-world perception and knowledge retrieval. Specifically, we construct a comprehensive driving knowledge graph that encodes traffic laws, defensive driving principles, and ethical norms, complemented by an efficient LLM-based retrieval mechanism tailored for driving scenarios. Furthermore, we develop a human-preference dataset and train a Value Model to guide interpretable, value-aligned trajectory assessment. Experimental results show that our method substantially improves planning performance while remaining compatible with existing architectures. Notably, KnowVal achieves the lowest collision rate on nuScenes and state-of-the-art results on Bench2Drive.
>
---
#### [new 002] Gaussian Variational Inference with Non-Gaussian Factors for State Estimation: A UWB Localization Case Study
- **分类: cs.RO; stat.ML**

- **简介: 该论文面向UWB定位中的状态估计任务，解决NLOS和多径导致的非高斯噪声问题。工作包括：将ESGVI算法推广至矩阵李群以处理姿态状态，并引入非高斯因子建模重尾/偏斜噪声，保持稀疏性与无导数特性，开源实现验证了精度提升。**

- **链接: [https://arxiv.org/pdf/2512.19855v1](https://arxiv.org/pdf/2512.19855v1)**

> **作者:** Andrew Stirling; Mykola Lukashchuk; Dmitry Bagaev; Wouter Kouw; James R. Forbes
>
> **摘要:** This letter extends the exactly sparse Gaussian variational inference (ESGVI) algorithm for state estimation in two complementary directions. First, ESGVI is generalized to operate on matrix Lie groups, enabling the estimation of states with orientation components while respecting the underlying group structure. Second, factors are introduced to accommodate heavy-tailed and skewed noise distributions, as commonly encountered in ultra-wideband (UWB) localization due to non-line-of-sight (NLOS) and multipath effects. Both extensions are shown to integrate naturally within the ESGVI framework while preserving its sparse and derivative-free structure. The proposed approach is validated in a UWB localization experiment with NLOS-rich measurements, demonstrating improved accuracy and comparable consistency. Finally, a Python implementation within a factor-graph-based estimation framework is made open-source (https://github.com/decargroup/gvi_ws) to support broader research use.
>
---
#### [new 003] LightTact: A Visual-Tactile Fingertip Sensor for Deformation-Independent Contact Sensing
- **分类: cs.RO**

- **简介: 该论文提出LightTact视觉-触觉指尖传感器，解决轻接触（如液体、超软物）下因无宏观形变导致的触觉感知难题。它基于光学原理实现形变无关接触检测，输出高对比度图像，支持像素级接触分割与多场景灵巧操作，并兼容视觉语言模型。**

- **链接: [https://arxiv.org/pdf/2512.20591v1](https://arxiv.org/pdf/2512.20591v1)**

> **作者:** Changyi Lin; Boda Huo; Mingyang Yu; Emily Ruppel; Bingqing Chen; Jonathan Francis; Ding Zhao
>
> **摘要:** Contact often occurs without macroscopic surface deformation, such as during interaction with liquids, semi-liquids, or ultra-soft materials. Most existing tactile sensors rely on deformation to infer contact, making such light-contact interactions difficult to perceive robustly. To address this, we present LightTact, a visual-tactile fingertip sensor that makes contact directly visible via a deformation-independent, optics-based principle. LightTact uses an ambient-blocking optical configuration that suppresses both external light and internal illumination at non-contact regions, while transmitting only the diffuse light generated at true contacts. As a result, LightTact produces high-contrast raw images in which non-contact pixels remain near-black (mean gray value < 3) and contact pixels preserve the natural appearance of the contacting surface. Built on this, LightTact achieves accurate pixel-level contact segmentation that is robust to material properties, contact force, surface appearance, and environmental lighting. We further integrate LightTact on a robotic arm and demonstrate manipulation behaviors driven by extremely light contact, including water spreading, facial-cream dipping, and thin-film interaction. Finally, we show that LightTact's spatially aligned visual-tactile images can be directly interpreted by existing vision-language models, enabling resistor value reasoning for robotic sorting.
>
---
#### [new 004] LoLA: Long Horizon Latent Action Learning for General Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属机器人操纵任务，旨在解决长时序语言引导操作中历史信息利用不足与动作连贯性差的问题。提出LoLA框架，融合多视角视觉、语言和本体感知，通过状态感知的潜空间重表征，将VL表征物理锚定到机器人运动空间，并在仿真与真实机器人上验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.20166v1](https://arxiv.org/pdf/2512.20166v1)**

> **作者:** Xiaofan Wang; Xingyu Gao; Jianlong Fu; Zuolei Li; Dean Fortier; Galen Mullins; Andrey Kolobov; Baining Guo
>
> **摘要:** The capability of performing long-horizon, language-guided robotic manipulation tasks critically relies on leveraging historical information and generating coherent action sequences. However, such capabilities are often overlooked by existing Vision-Language-Action (VLA) models. To solve this challenge, we propose LoLA (Long Horizon Latent Action Learning), a framework designed for robot manipulation that integrates long-term multi-view observations and robot proprioception to enable multi-step reasoning and action generation. We first employ Vision-Language Models to encode rich contextual features from historical sequences and multi-view observations. We further introduces a key module, State-Aware Latent Re-representation, which transforms visual inputs and language commands into actionable robot motion space. Unlike existing VLA approaches that merely concatenate robot proprioception (e.g., joint angles) with VL embeddings, this module leverages such robot states to explicitly ground VL representations in physical scale through a learnable "embodiment-anchored" latent space. We trained LoLA on diverse robotic pre-training datasets and conducted extensive evaluations on simulation benchmarks (SIMPLER and LIBERO), as well as two real-world tasks on Franka and Bi-Manual Aloha robots. Results show that LoLA significantly outperforms prior state-of-the-art methods (e.g., pi0), particularly in long-horizon manipulation tasks.
>
---
#### [new 005] UrbanV2X: A Multisensory Vehicle-Infrastructure Dataset for Cooperative Navigation in Urban Areas
- **分类: cs.RO**

- **简介: 该论文属于智能交通与自动驾驶数据集构建任务，旨在解决城市复杂环境下车路协同导航缺乏真实多模态数据的问题。作者构建了同步、标定的UrbanV2X多传感器车路协同数据集，并提供基准导航算法评测，已开源。**

- **链接: [https://arxiv.org/pdf/2512.20224v1](https://arxiv.org/pdf/2512.20224v1)**

> **作者:** Qijun Qin; Ziqi Zhang; Yihan Zhong; Feng Huang; Xikun Liu; Runzhi Hu; Hang Chen; Wei Hu; Dongzhe Su; Jun Zhang; Hoi-Fung Ng; Weisong Wen
>
> **备注:** 8 pages, 9 figures, IEEE ITSC 2025
>
> **摘要:** Due to the limitations of a single autonomous vehicle, Cellular Vehicle-to-Everything (C-V2X) technology opens a new window for achieving fully autonomous driving through sensor information sharing. However, real-world datasets supporting vehicle-infrastructure cooperative navigation in complex urban environments remain rare. To address this gap, we present UrbanV2X, a comprehensive multisensory dataset collected from vehicles and roadside infrastructure in the Hong Kong C-V2X testbed, designed to support research on smart mobility applications in dense urban areas. Our onboard platform provides synchronized data from multiple industrial cameras, LiDARs, 4D radar, ultra-wideband (UWB), IMU, and high-precision GNSS-RTK/INS navigation systems. Meanwhile, our roadside infrastructure provides LiDAR, GNSS, and UWB measurements. The entire vehicle-infrastructure platform is synchronized using the Precision Time Protocol (PTP), with sensor calibration data provided. We also benchmark various navigation algorithms to evaluate the collected cooperative data. The dataset is publicly available at https://polyu-taslab.github.io/UrbanV2X/.
>
---
#### [new 006] Pneumatic bladder links with wide range of motion joints for articulated inflatable robots
- **分类: cs.RO**

- **简介: 该论文属于机器人学中的软体机器人任务，旨在解决传统 inflatable 机器人关节运动范围小、负载能力弱的问题。作者提出一种由气动气囊连杆与新型 Hillberry 滚动接触关节组成的可伸缩机械臂结构，实现±150°大角度运动，并验证了多自由度臂的负载能力及腿部运动可行性。**

- **链接: [https://arxiv.org/pdf/2512.20322v1](https://arxiv.org/pdf/2512.20322v1)**

> **作者:** Katsu Uchiyama; Ryuma Niiyama
>
> **备注:** Accepted at IROS2024 (IEEE/RSJ International Conference on Intelligent Robots and Systems)
>
> **摘要:** Exploration of various applications is the frontier of research on inflatable robots. We proposed an articulated robots consisting of multiple pneumatic bladder links connected by rolling contact joints called Hillberry joints. The bladder link is made of a double-layered structure of tarpaulin sheet and polyurethane sheet, which is both airtight and flexible in shape. The integration of the Hilberry joint into an inflatable robot is also a new approach. The rolling contact joint allows wide range of motion of $\pm 150 ^{\circ}$, the largest among the conventional inflatable joints. Using the proposed mechanism for inflatable robots, we demonstrated moving a 500 g payload with a 3-DoF arm and lifting 3.4 kg and 5 kg payloads with 2-DoF and 1-DoF arms, respectively. We also experimented with a single 3-DoF inflatable leg attached to a dolly to show that the proposed structure worked for legged locomotion.
>
---
#### [new 007] Bring My Cup! Personalizing Vision-Language-Action Models with Visual Attentive Prompting
- **分类: cs.RO; cs.AI**

- **简介: 该论文聚焦个性化视觉-语言-动作（VLA）任务，解决VLA模型难以执行“拿我的杯子”等需识别用户专属物体的指令问题。提出无需训练的视觉注意力提示（VAP）方法，利用参考图像实现开放词汇检测与嵌入匹配，动态高亮目标并重写指令，提升实例级操作准确率。**

- **链接: [https://arxiv.org/pdf/2512.20014v1](https://arxiv.org/pdf/2512.20014v1)**

> **作者:** Sangoh Lee; Sangwoo Mo; Wook-Shin Han
>
> **摘要:** While Vision-Language-Action (VLA) models generalize well to generic instructions, they struggle with personalized commands such as "bring my cup", where the robot must act on one specific instance among visually similar objects. We study this setting of manipulating personal objects, in which a VLA must identify and control a user-specific object unseen during training using only a few reference images. To address this challenge, we propose Visual Attentive Prompting (VAP), a simple-yet-effective training-free perceptual adapter that equips frozen VLAs with top-down selective attention. VAP treats the reference images as a non-parametric visual memory, grounds the personal object in the scene through open-vocabulary detection and embedding-based matching, and then injects this grounding as a visual prompt by highlighting the object and rewriting the instruction. We construct two simulation benchmarks, Personalized-SIMPLER and Personalized-VLABench, and a real-world tabletop benchmark to evaluate personalized manipulation across multiple robots and tasks. Experiments show that VAP consistently outperforms generic policies and token-learning baselines in both success rate and correct-object manipulation, helping to bridge the gap between semantic understanding and instance-level control.
>
---
#### [new 008] A Time-efficient Prioritised Scheduling Algorithm to Optimise Initial Flock Formation of Drones
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于多无人机协同控制任务，旨在解决大规模机群初始编队阶段的效率低、易碰撞问题。提出一种基于碰撞风险与通行影响的优先级调度算法，为每架无人机计算最优延迟，实现高效无碰撞编队。实验验证其在5000架规模下优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.19914v1](https://arxiv.org/pdf/2512.19914v1)**

> **作者:** Sujan Warnakulasooriya; Andreas Willig; Xiaobing Wu
>
> **备注:** 35 pages
>
> **摘要:** Drone applications continue to expand across various domains, with flocking offering enhanced cooperative capabilities but introducing significant challenges during initial formation. Existing flocking algorithms often struggle with efficiency and scalability, particularly when potential collisions force drones into suboptimal trajectories. This paper presents a time-efficient prioritised scheduling algorithm that improves the initial formation process of drone flocks. The method assigns each drone a priority based on its number of potential collisions and its likelihood of reaching its target position without permanently obstructing other drones. Using this hierarchy, each drone computes an appropriate delay to ensure a collision-free path. Simulation results show that the proposed algorithm successfully generates collision-free trajectories for flocks of up to 5000 drones and outperforms the coupling-degree-based heuristic prioritised planning method (CDH-PP) in both performance and computational efficiency.
>
---
#### [new 009] Drift-Corrected Monocular VIO and Perception-Aware Planning for Autonomous Drone Racing
- **分类: cs.RO**

- **简介: 该论文面向单目视觉惯性里程计（VIO）在高速无人机竞速中易漂移的问题，提出漂移校正与感知感知协同规划方法：用YOLO检测赛道门+卡尔曼滤波融合全局位置修正VIO；设计感知感知规划器确保门始终可见。属自主无人机竞速任务。**

- **链接: [https://arxiv.org/pdf/2512.20475v1](https://arxiv.org/pdf/2512.20475v1)**

> **作者:** Maulana Bisyir Azhari; Donghun Han; Je In You; Sungjun Park; David Hyunchul Shim
>
> **摘要:** The Abu Dhabi Autonomous Racing League(A2RL) x Drone Champions League competition(DCL) requires teams to perform high-speed autonomous drone racing using only a single camera and a low-quality inertial measurement unit -- a minimal sensor set that mirrors expert human drone racing pilots. This sensor limitation makes the system susceptible to drift from Visual-Inertial Odometry (VIO), particularly during long and fast flights with aggressive maneuvers. This paper presents the system developed for the championship, which achieved a competitive performance. Our approach corrected VIO drift by fusing its output with global position measurements derived from a YOLO-based gate detector using a Kalman filter. A perception-aware planner generated trajectories that balance speed with the need to keep gates visible for the perception system. The system demonstrated high performance, securing podium finishes across multiple categories: third place in the AI Grand Challenge with top speed of 43.2 km/h, second place in the AI Drag Race with over 59 km/h, and second place in the AI Multi-Drone Race. We detail the complete architecture and present a performance analysis based on experimental data from the competition, contributing our insights on building a successful system for monocular vision-based autonomous drone flight.
>
---
#### [new 010] Asynchronous Fast-Slow Vision-Language-Action Policies for Whole-Body Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DuoCore-FS异步快慢VLA框架，解决现有视觉-语言-动作系统因同步执行导致的实时性差、控制不稳定问题。通过潜表示缓冲区与全身动作分词器，实现大VLM（3B）与高速动作生成（30Hz）解耦，提升全身体操作成功率与响应速度。**

- **链接: [https://arxiv.org/pdf/2512.20188v1](https://arxiv.org/pdf/2512.20188v1)**

> **作者:** Teqiang Zou; Hongliang Zeng; Yuxuan Nong; Yifan Li; Kehui Liu; Haotian Yang; Xinyang Ling; Xin Li; Lianyang Ma
>
> **摘要:** Most Vision-Language-Action (VLA) systems integrate a Vision-Language Model (VLM) for semantic reasoning with an action expert generating continuous action signals, yet both typically run at a single unified frequency. As a result, policy performance is constrained by the low inference speed of large VLMs. This mandatory synchronous execution severely limits control stability and real-time performance in whole-body robotic manipulation, which involves more joints, larger motion spaces, and dynamically changing views. We introduce a truly asynchronous Fast-Slow VLA framework (DuoCore-FS), organizing the system into a fast pathway for high-frequency action generation and a slow pathway for rich VLM reasoning. The system is characterized by two key features. First, a latent representation buffer bridges the slow and fast systems. It stores instruction semantics and action-reasoning representation aligned with the scene-instruction context, providing high-level guidance to the fast pathway. Second, a whole-body action tokenizer provides a compact, unified representation of whole-body actions. Importantly, the VLM and action expert are still jointly trained end-to-end, preserving unified policy learning while enabling asynchronous execution. DuoCore-FS supports a 3B-parameter VLM while achieving 30 Hz whole-body action-chunk generation, approximately three times as fast as prior VLA models with comparable model sizes. Real-world whole-body manipulation experiments demonstrate improved task success rates and significantly enhanced responsiveness compared to synchronous Fast-Slow VLA baselines. The implementation of DuoCore-FS, including training, inference, and deployment, is provided to commercial users by Astribot as part of the Astribot robotic platform.
>
---
#### [new 011] FAR-AVIO: Fast and Robust Schur-Complement Based Acoustic-Visual-Inertial Fusion Odometry with Sensor Calibration
- **分类: cs.RO**

- **简介: 该论文提出FAR-AVIO，一种面向水下机器人的快速鲁棒声-视-惯性融合里程计方法。针对水下视觉退化、惯性可观测性弱及实时性差的问题，其基于舒尔补的EKF框架实现紧耦合优化，并引入在线传感器可靠性评估（AWARE）与DVL-IMU外参在线标定，兼顾精度、鲁棒性与效率。**

- **链接: [https://arxiv.org/pdf/2512.20355v1](https://arxiv.org/pdf/2512.20355v1)**

> **作者:** Hao Wei; Peiji Wang; Qianhao Wang; Tong Qin; Fei Gao; Yulin Si
>
> **摘要:** Underwater environments impose severe challenges to visual-inertial odometry systems, as strong light attenuation, marine snow and turbidity, together with weakly exciting motions, degrade inertial observability and cause frequent tracking failures over long-term operation. While tightly coupled acoustic-visual-inertial fusion, typically implemented through an acoustic Doppler Velocity Log (DVL) integrated with visual-inertial measurements, can provide accurate state estimation, the associated graph-based optimization is often computationally prohibitive for real-time deployment on resource-constrained platforms. Here we present FAR-AVIO, a Schur-Complement based, tightly coupled acoustic-visual-inertial odometry framework tailored for underwater robots. FAR-AVIO embeds a Schur complement formulation into an Extended Kalman Filter(EKF), enabling joint pose-landmark optimization for accuracy while maintaining constant-time updates by efficiently marginalizing landmark states. On top of this backbone, we introduce Adaptive Weight Adjustment and Reliability Evaluation(AWARE), an online sensor health module that continuously assesses the reliability of visual, inertial and DVL measurements and adaptively regulates their sigma weights, and we develop an efficient online calibration scheme that jointly estimates DVL-IMU extrinsics, without dedicated calibration manoeuvres. Numerical simulations and real-world underwater experiments consistently show that FAR-AVIO outperforms state-of-the-art underwater SLAM baselines in both localization accuracy and computational efficiency, enabling robust operation on low-power embedded platforms. Our implementation has been released as open source software at https://far-vido.gitbook.io/far-vido-docs.
>
---
#### [new 012] ActionFlow: A Pipelined Action Acceleration for Vision Language Models on Edge
- **分类: cs.AI; cs.RO**

- **简介: 该论文属边缘端视觉语言动作（VLA）模型推理优化任务，旨在解决其在边缘设备上因自回归解码导致的高延迟、低控制频率（仅3–5 Hz）问题。提出ActionFlow框架，含跨请求流水调度、状态打包算子与统一KV环形缓冲，实现2.55×加速，达实时操控（>20 Hz）。**

- **链接: [https://arxiv.org/pdf/2512.20276v1](https://arxiv.org/pdf/2512.20276v1)**

> **作者:** Yuntao Dai; Hang Gu; Teng Wang; Qianyu Cheng; Yifei Zheng; Zhiyong Qiu; Lei Gong; Wenqi Lou; Xuehai Zhou
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a unified paradigm for robotic perception and control, enabling emergent generalization and long-horizon task execution. However, their deployment in dynamic, real-world environments is severely hin dered by high inference latency. While smooth robotic interaction requires control frequencies of 20 to 30 Hz, current VLA models typi cally operate at only 3-5 Hz on edge devices due to the memory bound nature of autoregressive decoding. Existing optimizations often require extensive retraining or compromise model accuracy. To bridge this gap, we introduce ActionFlow, a system-level inference framework tailored for resource-constrained edge plat forms. At the core of ActionFlow is a Cross-Request Pipelin ing strategy, a novel scheduler that redefines VLA inference as a macro-pipeline of micro-requests. The strategy intelligently batches memory-bound Decode phases with compute-bound Prefill phases across continuous time steps to maximize hardware utilization. Furthermore, to support this scheduling, we propose a Cross Request State Packed Forward operator and a Unified KV Ring Buffer, which fuse fragmented memory operations into efficient dense computations. Experimental results demonstrate that ActionFlow achieves a 2.55x improvement in FPS on the OpenVLA-7B model without retraining, enabling real-time dy namic manipulation on edge hardware. Our work is available at https://anonymous.4open.science/r/ActionFlow-1D47.
>
---
#### [new 013] Finite-Time Control Based on Differential Flatness for Wheeled Mobile Robots with Experimental Validation
- **分类: eess.SY; cs.RO**

- **简介: 该论文属机器人运动控制任务，旨在解决轮式移动机器人在风扰、不平路面等干扰下的轨迹跟踪精度与鲁棒性问题。工作包括：利用微分平坦性将模型线性化，设计新型积分非线性超平面滑模控制器，并通过TurtleBot3实验证明其有效性。**

- **链接: [https://arxiv.org/pdf/2512.20229v1](https://arxiv.org/pdf/2512.20229v1)**

> **作者:** Imtiaz Ur Rehman; Moussa Labbadi; Amine Abadi; Lew Lew Yan Voon
>
> **摘要:** A robust tracking control strategy is designed to empower wheeled mobile robots (WMRs) to track predetermined routes while operating in diverse fields and encountering disturbances like strong winds or uneven path conditions, which affect tracking performance. Ensuring the applicability of this tracking method in real-world scenarios is essential. To accomplish this, the WMR model is initially transformed into a linear canonical form by leveraging the differential flatness of its kinematic model, facilitating controller design. Subsequently, a novel integral nonlinear hyperplane-based sliding mode control (INH-SMC) technique is proposed for WMR under disturbances. The stability of the technique is analyzed and verified. Finally, its practical viability is demonstrated through a comparative real-world indoor experiment on a TurtleBot3 WMR subjected to disturbances, confirming the feasibility and efficacy of the proposed approach.
>
---
#### [new 014] A Class of Axis-Angle Attitude Control Laws for Rotational Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属航天器姿态控制任务，旨在解决传统四元数方法灵活性不足、高速翻滚恢复耗时长且控制力矩大的问题。作者提出一类基于广义轴-角表示的新型姿态控制律，利用Lyapunov理论保证全局渐近稳定，并通过仿真与实验验证其在稳定时间与控制能耗上的优越性。**

- **链接: [https://arxiv.org/pdf/2512.19846v1](https://arxiv.org/pdf/2512.19846v1)**

> **作者:** Francisco M. F. R. Gonçalves; Ryan M. Bena; Néstor O. Pérez-Arancibia
>
> **备注:** 6 pages, 4 figures. Accepted for publication in IEEE Control Systems Letters
>
> **摘要:** We introduce a new class of attitude control laws for rotational systems, which generalizes the use of the Euler axis-angle representation beyond quaternion-based formulations. Using basic Lyapunov's stability theory and the notion of extended $K_{\infty}$ functions, we developed a method for determining and enforcing the global asymptotic stability of the single fixed point of the resulting closed-loop (CL) scheme. In contrast with traditional quaternion-based methods, the proposed generalized axis-angle approach enables greater flexibility in the design of the control law, which is of great utility when employed in combination with a switching scheme whose transition state depends on the angular velocity of the controlled rotational system. Through simulation and real-time experimental results, we demonstrate the effectiveness of the proposed approach. According to the recorded data, in the execution of high-speed tumble-recovery maneuvers, the new method consistently achieves shorter stabilization times and requires lower control effort relative to those corresponding to the quaternion-based and geometric-control methods used as benchmarks.
>
---
#### [new 015] Design and Modeling of a Simple-Structured Continuously Variable Transmission Utilizing Shape Memory Alloy Superelasticity for Twisted String Actuator
- **分类: physics.ins-det; cs.RO**

- **简介: 该论文属机器人驱动技术任务，旨在解决Twisted String Actuator（TSA）传动比（TR）调节范围窄、负载适应性差的问题。提出一种基于形状记忆合金（SMA）超弹性特性的轻量、简易连续可变传动机构，并构建融合三大非线性的理论模型。**

- **链接: [https://arxiv.org/pdf/2512.20342v1](https://arxiv.org/pdf/2512.20342v1)**

> **作者:** Chanchan Xu; Shuai Dong; Xiaojie Wang
>
> **备注:** 15pages,Fig14
>
> **摘要:** Twisted String Actuators (TSAs) are widely used in robotics but suffer from a limited range of Transmission Ratio (TR) variation, restricting their efficiency under varying loads.To overcome this, we propose a novel lightweight, simple-structured Continuously Variable Transmission (CVT) mechanism for TSA utilizing Shape Memory Alloy (SMA) superelasticity. The CVT mechanism consists solely of a pair of highly lightweight superelastic SMA rods connecting the ends of twisted strings. These rods deform under external loads, adjusting the inter-string distance to enable continuous TR variation.We develop a comprehensive theoretical model that integrates three critical nonlinearities
>
---
#### [new 016] Detecting Non-Optimal Decisions of Embodied Agents via Diversity-Guided Metamorphic Testing
- **分类: cs.SE; cs.RO**

- **简介: 该论文属AI系统测试任务，旨在检测具身智能体在成功完成任务时仍存在的非最优决策（NoDs）。作者提出NoD-DGMT框架，基于多样性引导的变异测试，设计四类优化性变异关系，并通过多样性选择策略提升检测效率与覆盖率。**

- **链接: [https://arxiv.org/pdf/2512.20083v1](https://arxiv.org/pdf/2512.20083v1)**

> **作者:** Wenzhao Wu; Yahui Tang; Mingfei Cheng; Wenbing Tang; Yuan Zhou; Yang Liu
>
> **摘要:** As embodied agents advance toward real-world deployment, ensuring optimal decisions becomes critical for resource-constrained applications. Current evaluation methods focus primarily on functional correctness, overlooking the non-functional optimality of generated plans. This gap can lead to significant performance degradation and resource waste. We identify and formalize the problem of Non-optimal Decisions (NoDs), where agents complete tasks successfully but inefficiently. We present NoD-DGMT, a systematic framework for detecting NoDs in embodied agent task planning via diversity-guided metamorphic testing. Our key insight is that optimal planners should exhibit invariant behavioral properties under specific transformations. We design four novel metamorphic relations capturing fundamental optimality properties: position detour suboptimality, action optimality completeness, condition refinement monotonicity, and scene perturbation invariance. To maximize detection efficiency, we introduce a diversity-guided selection strategy that actively selects test cases exploring different violation categories, avoiding redundant evaluations while ensuring comprehensive diversity coverage. Extensive experiments on the AI2-THOR simulator with four state-of-the-art planning models demonstrate that NoD-DGMT achieves violation detection rates of 31.9% on average, with our diversity-guided filter improving rates by 4.3% and diversity scores by 3.3 on average. NoD-DGMT significantly outperforms six baseline methods, with 16.8% relative improvement over the best baseline, and demonstrates consistent superiority across different model architectures and task complexities.
>
---
#### [new 017] Learning Skills from Action-Free Videos
- **分类: cs.AI; cs.RO**

- **简介: 该论文属机器人技能学习任务，旨在解决从无动作标注视频中提取可执行、可规划的机器人技能难题。提出SOF框架，利用光流构建动作对齐的隐式技能空间，实现视频到技能再到动作的端到端学习与高阶规划。**

- **链接: [https://arxiv.org/pdf/2512.20052v1](https://arxiv.org/pdf/2512.20052v1)**

> **作者:** Hung-Chieh Fang; Kuo-Han Hung; Chu-Rong Chen; Po-Jung Chou; Chun-Kai Yang; Po-Chen Ko; Yu-Chiang Wang; Yueh-Hua Wu; Min-Hung Chen; Shao-Hua Sun
>
> **摘要:** Learning from videos offers a promising path toward generalist robots by providing rich visual and temporal priors beyond what real robot datasets contain. While existing video generative models produce impressive visual predictions, they are difficult to translate into low-level actions. Conversely, latent-action models better align videos with actions, but they typically operate at the single-step level and lack high-level planning capabilities. We bridge this gap by introducing Skill Abstraction from Optical Flow (SOF), a framework that learns latent skills from large collections of action-free videos. Our key idea is to learn a latent skill space through an intermediate representation based on optical flow that captures motion information aligned with both video dynamics and robot actions. By learning skills in this flow-based latent space, SOF enables high-level planning over video-derived skills and allows for easier translation of these skills into actions. Experiments show that our approach consistently improves performance in both multitask and long-horizon settings, demonstrating the ability to acquire and compose skills directly from raw visual data.
>
---
#### [new 018] LEAD: Minimizing Learner-Expert Asymmetry in End-to-End Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属端到端自动驾驶任务，旨在解决模仿学习中专家（高可见性、低不确定性、明确导航意图）与学生（传感器受限、意图模糊）间的感知与认知不对称问题。作者提出LEAD方法，通过缩小专家-学生差距，显著提升CARLA及真实世界基准性能。**

- **链接: [https://arxiv.org/pdf/2512.20563v1](https://arxiv.org/pdf/2512.20563v1)**

> **作者:** Long Nguyen; Micha Fauth; Bernhard Jaeger; Daniel Dauner; Maximilian Igl; Andreas Geiger; Kashyap Chitta
>
> **摘要:** Simulators can generate virtually unlimited driving data, yet imitation learning policies in simulation still struggle to achieve robust closed-loop performance. Motivated by this gap, we empirically study how misalignment between privileged expert demonstrations and sensor-based student observations can limit the effectiveness of imitation learning. More precisely, experts have significantly higher visibility (e.g., ignoring occlusions) and far lower uncertainty (e.g., knowing other vehicles' actions), making them difficult to imitate reliably. Furthermore, navigational intent (i.e., the route to follow) is under-specified in student models at test time via only a single target point. We demonstrate that these asymmetries can measurably limit driving performance in CARLA and offer practical interventions to address them. After careful modifications to narrow the gaps between expert and student, our TransFuser v6 (TFv6) student policy achieves a new state of the art on all major publicly available CARLA closed-loop benchmarks, reaching 95 DS on Bench2Drive and more than doubling prior performances on Longest6~v2 and Town13. Additionally, by integrating perception supervision from our dataset into a shared sim-to-real pipeline, we show consistent gains on the NAVSIM and Waymo Vision-Based End-to-End driving benchmarks. Our code, data, and models are publicly available at https://github.com/autonomousvision/lead.
>
---
#### [new 019] Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属多机器人协同控制任务，旨在解决通信失效下 swarm 机器人无通信协作避障难题。提出无通信的“应急模型控制”（CMC），基于离线共识规则（如交通规则）设计个体应急轨迹与互避约束，保障闭环递归可行与碰撞避免，并支持即插即用。**

- **链接: [https://arxiv.org/pdf/2512.20391v1](https://arxiv.org/pdf/2512.20391v1)**

> **作者:** Georg Schildbach
>
> **摘要:** Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.
>
---
#### [new 020] Enhancing annotations for 5D apple pose estimation through 3D Gaussian Splatting (3DGS)
- **分类: cs.CV; cs.RO**

- **简介: 该论文属农业机器人中的苹果位姿估计任务，旨在解决因遮挡导致的标注难、不一致问题。提出基于3D高斯泼溅（3DGS）的标注增强管线：重建果园场景→简化人工标注→自动投影生成海量训练标签→训练评估位姿模型。显著减少人工标注量（99.6%），但发现模型仍难学习苹果朝向。**

- **链接: [https://arxiv.org/pdf/2512.20148v1](https://arxiv.org/pdf/2512.20148v1)**

> **作者:** Robert van de Ven; Trim Bresilla; Bram Nelissen; Ard Nieuwenhuizen; Eldert J. van Henten; Gert Kootstra
>
> **备注:** 33 pages, excluding appendices. 17 figures
>
> **摘要:** Automating tasks in orchards is challenging because of the large amount of variation in the environment and occlusions. One of the challenges is apple pose estimation, where key points, such as the calyx, are often occluded. Recently developed pose estimation methods no longer rely on these key points, but still require them for annotations, making annotating challenging and time-consuming. Due to the abovementioned occlusions, there can be conflicting and missing annotations of the same fruit between different images. Novel 3D reconstruction methods can be used to simplify annotating and enlarge datasets. We propose a novel pipeline consisting of 3D Gaussian Splatting to reconstruct an orchard scene, simplified annotations, automated projection of the annotations to images, and the training and evaluation of a pose estimation method. Using our pipeline, 105 manual annotations were required to obtain 28,191 training labels, a reduction of 99.6%. Experimental results indicated that training with labels of fruits that are $\leq95\%$ occluded resulted in the best performance, with a neutral F1 score of 0.927 on the original images and 0.970 on the rendered images. Adjusting the size of the training dataset had small effects on the model performance in terms of F1 score and pose estimation accuracy. It was found that the least occluded fruits had the best position estimation, which worsened as the fruits became more occluded. It was also found that the tested pose estimation method was unable to correctly learn the orientation estimation of apples.
>
---
## 更新

#### [replaced 001] GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出GR-RL框架，解决通用视觉语言动作（VLA）策略在高灵巧、高精度长时程操作中因依赖次优人类示范而性能不足的问题。通过三阶段训练：用离线RL构建鲁棒进度函数过滤示范、形态对称增强泛化、在线RL学习隐空间噪声预测以提升控制精度。**

- **链接: [https://arxiv.org/pdf/2512.01801v3](https://arxiv.org/pdf/2512.01801v3)**

> **作者:** Yunfei Li; Xiao Ma; Jiafeng Xu; Yu Cui; Zhongren Cui; Zhigang Han; Liqun Huang; Tao Kong; Yuxiao Liu; Hao Niu; Wanli Peng; Jingchao Qiao; Zeyu Ren; Haixin Shi; Zhi Su; Jiawen Tian; Yuyang Xiao; Shenyu Zhang; Liwei Zheng; Hang Li; Yonghui Wu
>
> **摘要:** We present GR-RL, a robotic learning framework that turns a generalist vision-language-action (VLA) policy into a highly capable specialist for long-horizon dexterous manipulation. Assuming the optimality of human demonstrations is core to existing VLA policies. However, we claim that in highly dexterous and precise manipulation tasks, human demonstrations are noisy and suboptimal. GR-RL proposes a multi-stage training pipeline that filters, augments, and reinforces the demonstrations by reinforcement learning. First, GR-RL learns a vision-language-conditioned task progress, filters the demonstration trajectories, and only keeps the transitions that contribute positively to the progress. Specifically, we show that by directly applying offline RL with sparse reward, the resulting $Q$-values can be treated as a robust progress function. Next, we introduce morphological symmetry augmentation that greatly improves the generalization and performance of GR-RL. Lastly, to better align the VLA policy with its deployment behaviors for high-precision control, we perform online RL by learning a latent space noise predictor. With this pipeline, GR-RL is, to our knowledge, the first learning-based policy that can autonomously lace up a shoe by threading shoelaces through multiple eyelets with an 83.3% success rate, a task requiring long-horizon reasoning, millimeter-level precision, and compliant soft-body interaction. We hope GR-RL provides a step toward enabling generalist robot foundation models to specialize into reliable real-world experts.
>
---
#### [replaced 002] Conservative Bias in Multi-Teacher Learning: Why Agents Prefer Low-Reward Advisors
- **分类: cs.RO; cs.AI**

- **简介: 该论文属交互式强化学习（IRL）任务，探究多教师场景下智能体的教师选择偏好问题。发现智能体存在“保守偏差”，显著偏好低奖励但高一致性教师（93.16%选择率），而非高奖励教师；通过1250组导航实验验证该现象，并识别关键性能阈值与概念漂移下的优势。**

- **链接: [https://arxiv.org/pdf/2512.17180v2](https://arxiv.org/pdf/2512.17180v2)**

> **作者:** Maher Mesto; Francisco Cruz
>
> **备注:** 10 pages, 5 figures. Accepted at ACRA 2025 (Australasian Conference on Robotics and Automation)
>
> **摘要:** Interactive reinforcement learning (IRL) has shown promise in enabling autonomous agents and robots to learn complex behaviours from human teachers, yet the dynamics of teacher selection remain poorly understood. This paper reveals an unexpected phenomenon in IRL: when given a choice between teachers with different reward structures, learning agents overwhelmingly prefer conservative, low-reward teachers (93.16% selection rate) over those offering 20x higher rewards. Through 1,250 experimental runs in navigation tasks with multiple expert teachers, we discovered: (1) Conservative bias dominates teacher selection: agents systematically choose the lowest-reward teacher, prioritising consistency over optimality; (2) Critical performance thresholds exist at teacher availability rho >= 0.6 and accuracy omega >= 0.6, below which the framework fails catastrophically; (3) The framework achieves 159% improvement over baseline Q-learning under concept drift. These findings challenge fundamental assumptions about optimal teaching in RL and suggest potential implications for human-robot collaboration, where human preferences for safety and consistency may align with the observed agent selection behaviour, potentially informing training paradigms for safety-critical robotic applications.
>
---
#### [replaced 003] LeLaR: The First In-Orbit Demonstration of an AI-Based Satellite Attitude Controller
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于航天智能控制任务，旨在解决卫星姿态控制中传统控制器设计复杂、鲁棒性差及Sim2Real迁移难的问题。作者提出LeLaR系统，首次在轨验证了纯仿真训练的DRL姿态控制器，部署于InnoCube纳米卫星，并对比验证了其优于经典PD控制器的鲁棒性能。**

- **链接: [https://arxiv.org/pdf/2512.19576v2](https://arxiv.org/pdf/2512.19576v2)**

> **作者:** Kirill Djebko; Tom Baumann; Erik Dilger; Frank Puppe; Sergio Montenegro
>
> **备注:** 55 pages, 27 figures, 29 tables. The maneuver telemetry datasets generated and analyzed during this work are available in the GitHub repository under https://github.com/kdjebko/lelar-in-orbit-data
>
> **摘要:** Attitude control is essential for many satellite missions. Classical controllers, however, are time-consuming to design and sensitive to model uncertainties and variations in operational boundary conditions. Deep Reinforcement Learning (DRL) offers a promising alternative by learning adaptive control strategies through autonomous interaction with a simulation environment. Overcoming the Sim2Real gap, which involves deploying an agent trained in simulation onto the real physical satellite, remains a significant challenge. In this work, we present the first successful in-orbit demonstration of an AI-based attitude controller for inertial pointing maneuvers. The controller was trained entirely in simulation and deployed to the InnoCube 3U nanosatellite, which was developed by the Julius-Maximilians-Universität Würzburg in cooperation with the Technische Universität Berlin, and launched in January 2025. We present the AI agent design, the methodology of the training procedure, the discrepancies between the simulation and the observed behavior of the real satellite, and a comparison of the AI-based attitude controller with the classical PD controller of InnoCube. Steady-state metrics confirm the robust performance of the AI-based controller during repeated in-orbit maneuvers.
>
---
#### [replaced 004] Categorical Equivariant Deep Learning: Category-Equivariant Neural Networks and Universal Approximation Theorems
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出范畴等变深度学习（CENNs），统一各类等变网络（群、偏序集、图、层等）。旨在拓展等变学习 beyond 群作用，建模几何、上下文与组合对称性。工作包括构建范畴化等变框架、定义线性/非线性层、证明通用逼近定理，并实例化多类结构。**

- **链接: [https://arxiv.org/pdf/2511.18417v2](https://arxiv.org/pdf/2511.18417v2)**

> **作者:** Yoshihiro Maruyama
>
> **摘要:** We develop a theory of category-equivariant neural networks (CENNs) that unifies group/groupoid-equivariant networks, poset/lattice-equivariant networks, graph and sheaf neural networks. Equivariance is formulated as naturality in a topological category with Radon measures. Formulating linear and nonlinear layers in the categorical setup, we prove the equivariant universal approximation theorem in the general setting: the class of finite-depth CENNs is dense in the space of continuous equivariant transformations. We instantiate the framework for groups/groupoids, posets/lattices, graphs and cellular sheaves, deriving universal approximation theorems for them in a systematic manner. Categorical equivariant deep learning thus allows us to expand the horizons of equivariant deep learning beyond group actions, encompassing not only geometric symmetries but also contextual and compositional symmetries.
>
---
#### [replaced 005] Explainable deep learning improves human mental models of self-driving cars
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属可解释人工智能（XAI）任务，旨在解决自动驾驶中黑箱规划器导致人类驾驶员难以预判失效的问题。作者提出概念包装网络（CW-Net），将深度学习规划器的决策映射到人类可理解的概念上，并在真实无人车上部署验证，证明其能提升驾驶员心智模型、保持因果忠实性与驾驶性能。**

- **链接: [https://arxiv.org/pdf/2411.18714v2](https://arxiv.org/pdf/2411.18714v2)**

> **作者:** Eoin M. Kenny; Akshay Dharmavaram; Sang Uk Lee; Tung Phan-Minh; Shreyas Rajesh; Yunqing Hu; Laura Major; Momchil S. Tomov; Julie A. Shah
>
> **备注:** MST & JAS contributed equally to this work
>
> **摘要:** Self-driving cars increasingly rely on deep neural networks to achieve human-like driving. The opacity of such black-box planners makes it challenging for the human behind the wheel to accurately anticipate when they will fail, with potentially catastrophic consequences. While research into interpreting these systems has surged, most of it is confined to simulations or toy setups due to the difficulty of real-world deployment, leaving the practical utility of such techniques unknown. Here, we introduce the Concept-Wrapper Network (CW-Net), a method for explaining the behavior of machine-learning-based planners by grounding their reasoning in human-interpretable concepts. We deploy CW-Net on a real self-driving car and show that the resulting explanations improve the human driver's mental model of the car, allowing them to better predict its behavior. To our knowledge, this is the first demonstration that explainable deep learning integrated into self-driving cars can be both understandable and useful in a realistic deployment setting. CW-Net accomplishes this level of intelligibility while providing explanations which are causally faithful and do not sacrifice driving performance. Overall, our study establishes a general pathway to interpretability for autonomous agents by way of concept-based explanations, which could help make them more transparent and safe.
>
---
#### [replaced 006] FORWARD: Dataset of a forwarder operating in rough terrain
- **分类: cs.RO; cs.AI; cs.CE; cs.LG; physics.app-ph**

- **简介: 该论文构建了FORWARD数据集，面向森林作业场景，解决前装机在复杂地形下的感知、通行性建模与自主控制问题。工作包括采集高精度多模态数据（GNSS、IMU、视频、振动、引擎等），标注作业行为，设计对比实验，并开源用于AI算法开发与仿真器构建。**

- **链接: [https://arxiv.org/pdf/2511.17318v2](https://arxiv.org/pdf/2511.17318v2)**

> **作者:** Mikael Lundbäck; Erik Wallin; Carola Häggström; Mattias Nyström; Andreas Grönlund; Mats Richardson; Petrus Jönsson; William Arnvik; Lucas Hedström; Arvid Fälldin; Martin Servin
>
> **备注:** 28 pages, 22 figures
>
> **摘要:** We present FORWARD, a high-resolution multimodal dataset of a cut-to-length forwarder operating in rough terrain on two harvest sites in the middle part of Sweden. The forwarder is a large Komatsu model equipped with vehicle telematics sensors, including global positioning via satellite navigation, movement sensors, accelerometers, and engine sensors. The vehicle was additionally equipped with cameras, operator vibration sensors, and multiple IMUs. The data includes event time logs recorded at 5 Hz of driving speed, fuel consumption, vehicle position with centimeter accuracy, and crane use while the vehicle operates in forest areas, aerially laser-scanned with a resolution of around 1500 points per square meter. Production log files (StanForD standard) with time-stamped machine events, extensive video material, and terrain data in various formats are included as well. About 18 hours of regular wood extraction work during three days is annotated from 360-video material into individual work elements and included in the dataset. We also include scenario specifications of conducted experiments on forest roads and in terrain. Scenarios include repeatedly driving the same routes with and without steel tracks, different load weights, and different target driving speeds. The dataset is intended for developing models and algorithms for trafficability, perception, and autonomous control of forest machines using artificial intelligence, simulation, and experiments on physical testbeds. In part, we focus on forwarders traversing terrain, avoiding or handling obstacles, and loading or unloading logs, with consideration for efficiency, fuel consumption, safety, and environmental impact. Other benefits of the open dataset include the ability to explore auto-generation and calibration of forestry machine simulators and automation scenario descriptions using the data recorded in the field.
>
---
#### [replaced 007] Interaction Dataset of Autonomous Vehicles with Traffic Lights and Signs
- **分类: cs.RO; cs.AI**

- **简介: 该论文构建了自动驾驶车辆与交通灯、停车标志交互的公开轨迹数据集。针对现有研究缺乏真实AV交互行为数据的问题，作者从Waymo Motion中提取超8.1万实例，提出规则化识别、轨迹提取与小波去噪方法，显著降低加速度/急动度异常率，支撑行为建模与仿真研究。**

- **链接: [https://arxiv.org/pdf/2501.12536v2](https://arxiv.org/pdf/2501.12536v2)**

> **作者:** Zheng Li; Zhipeng Bao; Haoming Meng; Haotian Shi; Qianwen Li; Handong Yao; Xiaopeng Li
>
> **摘要:** This paper presents the development of a comprehensive dataset capturing interactions between Autonomous Vehicles (AVs) and traffic control devices, specifically traffic lights and stop signs. Derived from the Waymo Motion dataset, our work addresses a critical gap in the existing literature by providing real-world trajectory data on how AVs navigate these traffic control devices. We propose a methodology for identifying and extracting relevant interaction trajectory data from the Waymo Motion dataset, incorporating over 37,000 instances with traffic lights and 44,000 with stop signs. Our methodology includes defining rules to identify various interaction types, extracting trajectory data, and applying a wavelet-based denoising method to smooth the acceleration and speed profiles and eliminate anomalous values, thereby enhancing the trajectory quality. Quality assessment metrics indicate that trajectories obtained in this study have anomaly proportions in acceleration and jerk profiles reduced to near-zero levels across all interaction categories. By making this dataset publicly available, we aim to address the current gap in datasets containing AV interaction behaviors with traffic lights and signs. Based on the organized and published dataset, we can gain a more in-depth understanding of AVs' behavior when interacting with traffic lights and signs. This will facilitate research on AV integration into existing transportation infrastructures and networks, supporting the development of more accurate behavioral models and simulation tools.
>
---
#### [replaced 008] Tactile-based Object Retrieval From Granular Media
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出GEOTACT系统，解决未知形状物体在颗粒介质中触觉检索难题。任务是仅凭触觉反馈实现零视觉的抓取与 retrieval。工作包括：设计端到端触觉学习方法、引入仿真噪声训练、构建渐进式训练课程，并实现从仿真到真实硬件的零样本迁移。**

- **链接: [https://arxiv.org/pdf/2402.04536v3](https://arxiv.org/pdf/2402.04536v3)**

> **作者:** Jingxi Xu; Yinsen Jia; Dongxiao Yang; Patrick Meng; Xinyue Zhu; Zihan Guo; Shuran Song; Matei Ciocarlie
>
> **摘要:** We introduce GEOTACT, the first robotic system capable of grasping and retrieving objects of potentially unknown shapes buried in a granular environment. While important in many applications, ranging from mining and exploration to search and rescue, this type of interaction with granular media is difficult due to the uncertainty stemming from visual occlusion and noisy contact signals. To address these challenges, we use a learning method relying exclusively on touch feedback, trained end-to-end with simulated sensor noise. We show that our problem formulation leads to the natural emergence of learned pushing behaviors that the manipulator uses to reduce uncertainty and funnel the object to a stable grasp despite spurious and noisy tactile readings. We introduce a training curriculum that bootstraps learning in simulated granular environments, enabling zero-shot transfer to real hardware. Despite being trained only on seven objects with primitive shapes, our method is shown to successfully retrieve 35 different objects, including rigid, deformable, and articulated objects with complex shapes. Videos and additional information can be found at https://jxu.ai/geotact.
>
---
#### [replaced 009] DRAE: Dynamic Retrieval-Augmented Expert Networks for Lifelong Learning and Task Adaptation in Robotics
- **分类: cs.RO**

- **简介: 该论文提出DRAE架构，面向机器人终身学习任务，解决灾难性遗忘、知识复用与动态任务适应难题。工作包括：融合稀疏MoE路由、参数化RAG检索、分层RL框架（ReflexNet/SchemaPlanner/HyperOptima）及RSHO协调机制。**

- **链接: [https://arxiv.org/pdf/2507.04661v2](https://arxiv.org/pdf/2507.04661v2)**

> **作者:** Yayu Long; Kewei Chen; Long Jin; Mingsheng Shang
>
> **备注:** Accepted to the main conference of the Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** We introduce Dynamic Retrieval-Augmented Expert Networks (DRAE), a groundbreaking architecture that addresses the challenges of lifelong learning, catastrophic forgetting, and task adaptation by combining the dynamic routing capabilities of Mixture-of-Experts (MoE); leveraging the knowledge-enhancement power of Retrieval-Augmented Generation (RAG); incorporating a novel hierarchical reinforcement learning (RL) framework; and coordinating through ReflexNet-SchemaPlanner-HyperOptima (RSHO).DRAE dynamically routes expert models via a sparse MoE gating mechanism, enabling efficient resource allocation while leveraging external knowledge through parametric retrieval (P-RAG) to augment the learning process. We propose a new RL framework with ReflexNet for low-level task execution, SchemaPlanner for symbolic reasoning, and HyperOptima for long-term context modeling, ensuring continuous adaptation and memory retention. Experimental results show that DRAE significantly outperforms baseline approaches in long-term task retention and knowledge reuse, achieving an average task success rate of 82.5% across a set of dynamic robotic manipulation tasks, compared to 74.2% for traditional MoE models. Furthermore, DRAE maintains an extremely low forgetting rate, outperforming state-of-the-art methods in catastrophic forgetting mitigation. These results demonstrate the effectiveness of our approach in enabling flexible, scalable, and efficient lifelong learning for robotics.
>
---
#### [replaced 010] Let's Think in Two Steps: Mitigating Agreement Bias in MLLMs with Self-Grounded Verification
- **分类: cs.AI; cs.CL; cs.LG; cs.MA; cs.RO**

- **简介: 该论文聚焦MLLM作为验证器的任务，旨在解决其在开放域（如网页导航、机器人操作）中普遍存在的“同意偏差”——即过度认可错误行为的问题。作者提出两步式自扎根验证（SGV）方法，通过先生成先验再条件评估，显著提升失败检测与准确性，并推动下游任务性能突破。**

- **链接: [https://arxiv.org/pdf/2507.11662v2](https://arxiv.org/pdf/2507.11662v2)**

> **作者:** Moises Andrade; Joonhyuk Cha; Brandon Ho; Vriksha Srihari; Karmesh Yadav; Zsolt Kira
>
> **备注:** Our code, models, and data are publicly available at https://mshalimay.github.io/agreement-bias-sgv/
>
> **摘要:** Verifiers--functions assigning rewards to agent behavior--have been key for AI progress in domains like math and code. However, extending gains to domains without clear-cut success criteria (e.g., computer use) remains a challenge: while humans can recognize desired outcomes, translating this intuition into scalable rules is nontrivial. Multimodal Large Language Models (MLLMs) emerge as a promising solution, given their world knowledge, human-preference alignment, and reasoning skills. We evaluate MLLMs as verifiers across web navigation, computer use, and robotic manipulation, and identify a critical limitation: a strong tendency to over-validate agent behavior, a phenomenon we term agreement bias. This bias is pervasive across models, resilient to test-time scaling, and poses risks to existing methods relying on MLLM evaluations. We discuss methods to evaluate and improve MLLM verifiers and introduce Self-Grounded Verification (SGV), a lightweight method that harnesses MLLMs' own sampling mechanisms by modulating (un)conditional generation to better leverage their knowledge, alignment, and reasoning. SGV operates in two steps: first, the MLLM is elicited to generate broad priors about desired behavior, independent of the data under evaluation. Then, conditioned on self-generated priors, it reasons over and evaluates a candidate trajectory. SGV yields more human-aligned evaluations with gains of up to 25pp in failure detection, 14pp in accuracy, and benefits extending to downstream applications. In self-refinement and online supervision, SGV boosts task completion of a GUI specialist in OSWorld, a diffusion policy in robomimic, and a ReAct agent in VisualWebArena--setting a new state of the art, surpassing the previous best by 20pp. We release an updated version of VisualWebArena featuring more human-aligned evaluators, high-fidelity environment parallelism, and speedups of over 10x.
>
---
#### [replaced 011] BUFFER-X: Towards Zero-Shot Point Cloud Registration in Diverse Scenes
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文面向点云配准任务，解决现有方法在新场景中需重训练或调参的泛化瓶颈。提出零样本配准框架BUFFER-X：自适应体素/搜索半径、用最远点采样替代学习型关键点检测、分块尺度归一化，并构建多尺度描述符与层级内点搜索，显著提升跨场景鲁棒性。**

- **链接: [https://arxiv.org/pdf/2503.07940v3](https://arxiv.org/pdf/2503.07940v3)**

> **作者:** Minkyun Seo; Hyungtae Lim; Kanghee Lee; Luca Carlone; Jaesik Park
>
> **备注:** 20 pages, 14 figures. Accepted as a highlight paper at ICCV 2025
>
> **摘要:** Recent advances in deep learning-based point cloud registration have improved generalization, yet most methods still require retraining or manual parameter tuning for each new environment. In this paper, we identify three key factors limiting generalization: (a) reliance on environment-specific voxel size and search radius, (b) poor out-of-domain robustness of learning-based keypoint detectors, and (c) raw coordinate usage, which exacerbates scale discrepancies. To address these issues, we present a zero-shot registration pipeline called BUFFER-X by (a) adaptively determining voxel size/search radii, (b) using farthest point sampling to bypass learned detectors, and (c) leveraging patch-wise scale normalization for consistent coordinate bounds. In particular, we present a multi-scale patch-based descriptor generation and a hierarchical inlier search across scales to improve robustness in diverse scenes. We also propose a novel generalizability benchmark using 11 datasets that cover various indoor/outdoor scenarios and sensor modalities, demonstrating that BUFFER-X achieves substantial generalization without prior information or manual parameter tuning for the test datasets. Our code is available at https://github.com/MIT-SPARK/BUFFER-X.
>
---
#### [replaced 012] Learning Stack-of-Tasks Management for Redundant Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属机器人控制任务，旨在解决冗余机器人Stack-of-Tasks（SoT）控制器需人工设计优先级、激活逻辑与参数的问题。提出基于遗传编程与仿真评估的自动学习框架，端到端优化SoT结构，并在真实移动双臂机器人上验证了泛化性与安全性。**

- **链接: [https://arxiv.org/pdf/2508.10780v2](https://arxiv.org/pdf/2508.10780v2)**

> **作者:** Alessandro Adami; Aris Synodinos; Matteo Iovino; Ruggero Carli; Pietro Falco
>
> **摘要:** This paper presents a novel framework for automatically learning complete Stack-of-Tasks (SoT) controllers for redundant robotic systems, including task priorities, activation logic, and control parameters. Unlike classical SoT pipelines-where task hierarchies are manually defined and tuned-our approach optimizes the full SoT structure directly from a user-specified cost function encoding intuitive preferences such as safety, precision, manipulability, or execution speed. The method combines Genetic Programming with simulation-based evaluation to explore both discrete (priority order, task activation) and continuous (gains, trajectory durations) components of the controller. We validate the framework on a dual-arm mobile manipulator (the ABB mobile-YuMi research platform), demonstrating robust convergence across multiple cost definitions, automatic suppression of irrelevant tasks, and strong resilience to distractors. Learned SoTs exhibit expert-like hierarchical structure and adapt naturally to multi-objective trade-offs. Crucially, all controllers transfer from Gazebo simulation to the real robot, achieving safe and precise motion without additional tuning. Experiments in static and dynamic environments show reliable obstacle avoidance, high tracking accuracy, and predictable behavior in the presence of humans. The proposed method provides an interpretable and scalable alternative to manual SoT design, enabling rapid, user-driven generation of task execution hierarchies for complex robotic systems.
>
---
#### [replaced 013] DRO-EDL-MPC: Evidential Deep Learning-Based Distributionally Robust Model Predictive Control for Safe Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文面向自动驾驶运动规划任务，解决神经感知不确定性导致的控制不安全问题。提出DRO-EDL-MPC算法：基于证据深度学习量化感知的偶然性与认知不确定性，构建动态置信度驱动的分布鲁棒优化框架，并嵌入MPC实现安全、自适应的实时控制。**

- **链接: [https://arxiv.org/pdf/2507.05710v3](https://arxiv.org/pdf/2507.05710v3)**

> **作者:** Hyeongchan Ham; Heejin Ahn
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Safety is a critical concern in motion planning for autonomous vehicles. Modern autonomous vehicles rely on neural network-based perception, but making control decisions based on these inference results poses significant safety risks due to inherent uncertainties. To address this challenge, we present a distributionally robust optimization (DRO) framework that accounts for both aleatoric and epistemic perception uncertainties using evidential deep learning (EDL). Our approach introduces a novel ambiguity set formulation based on evidential distributions that dynamically adjusts the conservativeness according to perception confidence levels. We integrate this uncertainty-aware constraint into model predictive control (MPC), proposing the DRO-EDL-MPC algorithm with computational tractability for autonomous driving applications. Validation in the CARLA simulator demonstrates that our approach maintains efficiency under high perception confidence while enforcing conservative constraints under low confidence.
>
---
#### [replaced 014] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属行为克隆（BC）任务，旨在解决语言条件操控中动作序列的累积误差、物理不连续及语义-物理错位问题。提出CCoL框架，通过视觉-语言-本体感知的连续协同学习与双向交叉注意力对齐语义与物理表征，提升动作执行的连贯性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.14396v5](https://arxiv.org/pdf/2511.14396v5)**

> **作者:** Xiuxiu Qi; Yu Yang; Jiannong Cao; Luyao Bai; Chongshan Fan; Chengtai Cao; Hongpeng Wang
>
> **备注:** Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/
>
> **摘要:** Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.
>
---
#### [replaced 015] Learning Safe Autonomous Driving Policies Using Predictive Safety Representations
- **分类: cs.LG; cs.RO**

- **简介: 该论文属安全强化学习任务，旨在解决自动驾驶中性能优化与安全约束的冲突问题。提出基于预测性安全表征（SRPL）的框架，在Waymo和NuPlan数据集上验证其提升奖励-安全权衡、鲁棒性及跨数据集泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.17586v2](https://arxiv.org/pdf/2512.17586v2)**

> **作者:** Mahesh Keswani; Raunak Bhattacharyya
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Safe reinforcement learning (SafeRL) is a prominent paradigm for autonomous driving, where agents are required to optimize performance under strict safety requirements. This dual objective creates a fundamental tension, as overly conservative policies limit driving efficiency while aggressive exploration risks safety violations. The Safety Representations for Safer Policy Learning (SRPL) framework addresses this challenge by equipping agents with a predictive model of future constraint violations and has shown promise in controlled environments. This paper investigates whether SRPL extends to real-world autonomous driving scenarios. Systematic experiments on the Waymo Open Motion Dataset (WOMD) and NuPlan demonstrate that SRPL can improve the reward-safety tradeoff, achieving statistically significant improvements in success rate (effect sizes r = 0.65-0.86) and cost reduction (effect sizes r = 0.70-0.83), with p < 0.05 for observed improvements. However, its effectiveness depends on the underlying policy optimizer and the dataset distribution. The results further show that predictive safety representations play a critical role in improving robustness to observation noise. Additionally, in zero-shot cross-dataset evaluation, SRPL-augmented agents demonstrate improved generalization compared to non-SRPL methods. These findings collectively demonstrate the potential of predictive safety representations to strengthen SafeRL for autonomous driving.
>
---
#### [replaced 016] Deformable Cluster Manipulation via Whole-Arm Policy Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文面向可变形物体簇操纵任务，解决接触密集、感知不确定、建模困难等挑战。提出基于全臂触觉与点云的无模型强化学习框架，引入分布状态表征与无上下文遮挡启发式方法，实现零样本仿真到真实迁移，在电力线清障中验证有效性。**

- **链接: [https://arxiv.org/pdf/2507.17085v2](https://arxiv.org/pdf/2507.17085v2)**

> **作者:** Jayadeep Jacob; Wenzheng Zhang; Houston Warren; Paulo Borges; Tirthankar Bandyopadhyay; Fabio Ramos
>
> **摘要:** Manipulating clusters of deformable objects presents a substantial challenge with widespread applicability, but requires contact-rich whole-arm interactions. A potential solution must address the limited capacity for realistic model synthesis, high uncertainty in perception, and the lack of efficient spatial abstractions, among others. We propose a novel framework for learning model-free policies integrating two modalities: 3D point clouds and proprioceptive touch indicators, emphasising manipulation with full body contact awareness, going beyond traditional end-effector modes. Our reinforcement learning framework leverages a distributional state representation, aided by kernel mean embeddings, to achieve improved training efficiency and real-time inference. Furthermore, we propose a novel context-agnostic occlusion heuristic to clear deformables from a target region for exposure tasks. We deploy the framework in a power line clearance scenario and observe that the agent generates creative strategies leveraging multiple arm links for de-occlusion. Finally, we perform zero-shot sim-to-real policy transfer, allowing the arm to clear real branches with unknown occlusion patterns, unseen topology, and uncertain dynamics. Website: https://sites.google.com/view/dcmwap/
>
---
#### [replaced 017] LoGoPlanner: Localization Grounded Navigation Policy with Metric-aware Visual Geometry
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LoGoPlanner，面向移动机器人在非结构化环境中的端到端导航任务，旨在解决传统模块化方法误差累积与现有端到端方法依赖外部定位、泛化性差的问题。其通过度量感知视觉几何骨干网络实现隐式定位、历史几何重建与几何条件策略学习，提升规划一致性与跨平台泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.19629v2](https://arxiv.org/pdf/2512.19629v2)**

> **作者:** Jiaqi Peng; Wenzhe Cai; Yuqiang Yang; Tai Wang; Yuan Shen; Jiangmiao Pang
>
> **备注:** Project page:https://steinate.github.io/logoplanner.github.io/
>
> **摘要:** Trajectory planning in unstructured environments is a fundamental and challenging capability for mobile robots. Traditional modular pipelines suffer from latency and cascading errors across perception, localization, mapping, and planning modules. Recent end-to-end learning methods map raw visual observations directly to control signals or trajectories, promising greater performance and efficiency in open-world settings. However, most prior end-to-end approaches still rely on separate localization modules that depend on accurate sensor extrinsic calibration for self-state estimation, thereby limiting generalization across embodiments and environments. We introduce LoGoPlanner, a localization-grounded, end-to-end navigation framework that addresses these limitations by: (1) finetuning a long-horizon visual-geometry backbone to ground predictions with absolute metric scale, thereby providing implicit state estimation for accurate localization; (2) reconstructing surrounding scene geometry from historical observations to supply dense, fine-grained environmental awareness for reliable obstacle avoidance; and (3) conditioning the policy on implicit geometry bootstrapped by the aforementioned auxiliary tasks, thereby reducing error propagation. We evaluate LoGoPlanner in both simulation and real-world settings, where its fully end-to-end design reduces cumulative error while metric-aware geometry memory enhances planning consistency and obstacle avoidance, leading to more than a 27.3\% improvement over oracle-localization baselines and strong generalization across embodiments and environments. The code and models have been made publicly available on the https://steinate.github.io/logoplanner.github.io.
>
---
