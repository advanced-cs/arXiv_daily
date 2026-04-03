# 机器人 cs.RO

- **最新发布 47 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] Distal-Stable Beam for Continuum Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人结构设计任务，旨在解决连续机器人远端刚度不足导致的姿势误差问题。通过设计Distal-Stable Beam结构，实现远端稳定与中间柔性的平衡。**

- **链接: [https://arxiv.org/pdf/2604.01490](https://arxiv.org/pdf/2604.01490)**

> **作者:** Ryouichi Saito; Takahiro Koide; Yuya Tanaka; Yasutaka Nakashima; Motoji Yamamoto; Ayato Kanada
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Continuum robots are well suited for constrained environments but suffer from low distal stiffness, resulting in large posture errors under external loads. In this paper, we propose a novel structural primitive, the Distal-Stable Beam, which achieves a strong stiffness gradient through purely geometric design, maintaining compliance in the intermediate section while ensuring high distal rigidity. The structure consists of two parallel rods and one convergent rod constrained by guide disks, introducing geometric coupling that suppresses deformation modes and preserves distal posture. Experiments show that the distal stiffness is 12 times higher than at the center, corresponding to an approximately 100-fold improvement over a conventional cantilever beam. The proposed mechanism enables simultaneous compliance and distal stability without active stiffness modulation, providing a new design approach for continuum robots requiring both safety and precision.
>
---
#### [new 002] Boosting Vision-Language-Action Finetuning with Feasible Action Neighborhood Prior
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在解决模型在机器人操作中的泛化能力和样本效率问题。通过引入FAN引导的正则化方法，提升模型对可行动作邻域的适应性。**

- **链接: [https://arxiv.org/pdf/2604.01570](https://arxiv.org/pdf/2604.01570)**

> **作者:** Haochen Niu; Kanyu Zhang; Shuyu Yin; Qinghai Guo; Peilin Liu; Fei Wen
>
> **备注:** Accepted by CVPR 2026
>
> **摘要:** In real-world robotic manipulation, states typically admit a neighborhood of near-equivalent actions. That is, for each state, there exist a feasible action neighborhood (FAN) rather than a single correct action, within which motions yield indistinguishable progress. However, prevalent VLA training methodologies are directly inherited from linguistic settings and do not exploit the FAN property, thus leading to poor generalization and low sample efficiency. To address this limitation, we introduce a FAN-guided regularizer that shapes the model's output distribution to align with the geometry of FAN. Concretely, we introduce a Gaussian prior that promotes locally smooth and unimodal predictions around the preferred direction and magnitude. In extensive experiments across both reinforced finetuning (RFT) and supervised finetuning (SFT), our method achieves significant improvement in sample efficiency, and success rate in both in-distribution and out-of-distribution (OOD) scenarios. By aligning with the intrinsic action tolerance of physical manipulation, FAN-guided regularization provides a principled and practical method for sample-efficient, and generalizable VLA adaptation.
>
---
#### [new 003] Integrated Identification of Collaborative Robots for Robot Assisted 3D Printing Processes
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人建模任务，旨在解决协作机器人在3D打印中的精度与控制问题。通过五步集成识别方法建立可靠动态模型，提升误差预测能力。**

- **链接: [https://arxiv.org/pdf/2604.01991](https://arxiv.org/pdf/2604.01991)**

> **作者:** Alessandro Dimauro; Davide Tebaldi; Fabio Pini; Luigi Biagiotti; Francesco Leali
>
> **摘要:** In recent years, the integration of additive manufacturing (AM) and industrial robotics has opened new perspectives for the production of complex components, particularly in the automotive sector. Robot-assisted additive manufacturing processes overcome the dimensional and kinematic limitations of traditional Cartesian systems, enabling non-planar deposition and greater geometric flexibility. However, the increasing dynamic complexity of robotic manipulators introduces challenges related to precision, control, and error prediction. This work proposes a model-based approach equipped with an integrated identification procedure of the system's parameters, including the robot, the actuators and the controllers. We show that the integrated modeling procedure allows to obtain a reliable dynamic model even in the presence of sensory and programming limitations typical of collaborative robots. The manipulator's dynamic model is identified through an integrated five step methodology: starting with geometric and inertial analysis, followed by friction and controller parameters identification, all the way to the remaining parameters identification. The proposed procedure intrinsically ensures the physical consistency of the identified parameters. The identification approach is validated on a real world case study involving a 6-Degrees-Of-Freedom (DoFs) collaborative robot used in a thermoplastic extrusion process. The very good matching between the experimental results given by actual robot and those given by the identified model shows the potential enhancement of precision, control, and error prediction in Robot Assisted 3D Printing Processes.
>
---
#### [new 004] Analysis of Efficient Transmission Methods of Grid Maps for Intelligent Vehicles
- **分类: cs.RO**

- **简介: 该论文属于智能车辆环境建模任务，解决网格地图数据传输效率问题。通过提出基于块的通信管道，结合压缩算法提升传输效率，适用于车内和V2X场景。**

- **链接: [https://arxiv.org/pdf/2604.01753](https://arxiv.org/pdf/2604.01753)**

> **作者:** Robin Dehler; Dominik Authaler; Aryan Thakur; Thomas Wodtko; Michael Buchholz
>
> **备注:** Accepted for 2026 IEEE Intelligent Vehicles Symposium (IV) - DOI will be added after publication
>
> **摘要:** Grid mapping is a fundamental approach to modeling the environment of intelligent vehicles or robots. Compared with object-based environment modeling, grid maps offer the distinct advantage of representing the environment without requiring any assumptions about objects, such as type or shape. For grid-map-based approaches, the environment is divided into cells, each containing information about its respective area, such as occupancy. This representation of the entire environment is crucial for achieving higher levels of autonomy. However, it has the drawback that modeling the scene at the cell level results in inherently large data sizes. Patched grid maps tackle this issue to a certain extent by adapting cell sizes in specific areas. Nevertheless, the data sizes of patched grid maps are still too large for novel distributed processing setups or vehicle-to-everything (V2X) applications. Our work builds on a patch-based grid-map approach and investigates the size problem from a communication perspective. To address this, we propose a patch-based communication pipeline that leverages existing compression algorithms to transmit grid-map data efficiently. We provide a comprehensive analysis of this pipeline for both intra-vehicle and V2X-based communication. The analysis is verified for these use cases with two real-world experiment setups. Finally, we summarize recommended guidelines for the efficient transmission of grid-map data in intelligent transportation systems.
>
---
#### [new 005] Posterior Optimization with Clipped Objective for Bridging Efficiency and Stability in Generative Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决生成策略学习中的不稳定和样本效率低的问题。提出POCO框架，通过后验优化和截断目标提升策略性能。**

- **链接: [https://arxiv.org/pdf/2604.01860](https://arxiv.org/pdf/2604.01860)**

> **作者:** Yuhui Chen; Haoran Li; Zhennan Jiang; Yuxing Qin; Yuxuan Wan; Weiheng Liu; Dongbin Zhao
>
> **摘要:** Expressive generative models have advanced robotic manipulation by capturing complex, multi-modal action distributions over temporally extended trajectories. However, fine-tuning these policies via RL remains challenging due to instability and sample inefficiency. We introduce Posterior Optimization with Clipped Objective (POCO), a principled RL framework that formulates policy improvement as a posterior inference problem tailored for temporal action chunks. Through an Expectation-Maximization procedure, POCO distills a reward-weighted implicit posterior into the policy without likelihood estimation. Furthermore, POCO adopts an offline-to-online paradigm that anchors online exploration to pre-trained priors, and its model-agnostic design scales to fine-tune large VLA models without architectural modifications. Evaluations across 7 simulation benchmarks and 4 contact-rich real-world tasks demonstrate that POCO prevents catastrophic policy collapse, outperforms SOTA baselines, and achieves a 96.7% success rate on real-world tasks. Videos are available at our project website this https URL.
>
---
#### [new 006] Preferential Bayesian Optimization with Crash Feedback
- **分类: cs.RO**

- **简介: 该论文属于参数优化任务，解决硬件系统中因缺乏明确目标函数导致的崩溃问题。通过引入CrashPBO机制，结合用户偏好与崩溃反馈，提升优化效率与安全性。**

- **链接: [https://arxiv.org/pdf/2604.01776](https://arxiv.org/pdf/2604.01776)**

> **作者:** Johanna Menn; David Stenger; Sebastian Trimpe
>
> **摘要:** Bayesian optimization is a popular black-box optimization method for parameter learning in control and robotics. It typically requires an objective function that reflects the user's optimization goal. However, in practical applications, this objective function is often inaccessible due to complex or unmeasurable performance metrics. Preferential Bayesian optimization (PBO) overcomes this limitation by leveraging human feedback through pairwise comparisons, eliminating the need for explicit performance quantification. When applying PBO to hardware systems, such as in quadcopter control, crashes can cause time-consuming experimental resets, wear and tear, or otherwise undesired outcomes. Standard PBO methods cannot incorporate feedback from such crashed experiments, resulting in the exploration of parameters that frequently lead to experimental crashes. We thus introduce CrashPBO, a user-friendly mechanism that enables users to both express preferences and report crashes during the optimization process. Benchmarking on synthetic functions shows that this mechanism reduces crashes by 63% and increases data efficiency. Through experiments on three robotics platforms, we demonstrate the wide applicability and transferability of CrashPBO, highlighting that it provides a flexible, user-friendly framework for parameter learning with human feedback on preferences and crashes.
>
---
#### [new 007] A virtual-variable-length method for robust inverse kinematics of multi-segment continuum robots
- **分类: cs.RO; eess.SY; math.NA**

- **简介: 该论文属于机器人逆运动学任务，旨在解决多段连续机械臂的逆运动学求解问题。提出虚拟变长方法以提高收敛性与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.02256](https://arxiv.org/pdf/2604.02256)**

> **作者:** Weiting Feng; Federico Renda; Yunjie Yang; Francesco Giorgio-Serchi
>
> **备注:** 8 pages, 6 figures, accepted for presentation in IEEE RoboSoft 2026, Kanazawa, Japan
>
> **摘要:** This paper proposes a new, robust method to solve the inverse kinematics (IK) of multi-segment continuum manipulators. Conventional Jacobian-based solvers, especially when initialized from neutral/rest configurations, often exhibit slow convergence and, in certain conditions, may fail to converge (deadlock). The Virtual-Variable-Length (VVL) method proposed here introduces fictitious variations of segments' length during the solution iteration, conferring virtual axial degrees of freedom that alleviate adverse behaviors and constraints, thus enabling or accelerating convergence. Comprehensive numerical experiments were conducted to compare the VVL method against benchmark Jacobian-based and Damped Least Square IK solvers. Across more than $1.8\times 10^6$ randomized trials covering manipulators with two to seven segments, the proposed approach achieved up to a 20$\%$ increase in convergence success rate over the benchmark and a 40-80$\%$ reduction in average iteration count under equivalent accuracy thresholds ($10^{-4}-10^{-8}$). While deadlocks are not restricted to workspace boundaries and may occur at arbitrary poses, our empirical study identifies boundary-proximal configurations as a frequent cause of failed convergence and the VVL method mitigates such occurrences over a statistical sample of test cases.
>
---
#### [new 008] ROS 2-Based LiDAR Perception Framework for Mobile Robots in Dynamic Production Environments, Utilizing Synthetic Data Generation, Transformation-Equivariant 3D Detection and Multi-Object Tracking
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，解决动态环境中LiDAR感知的鲁棒性问题，提出基于ROS 2的框架，结合合成数据与多目标跟踪提升定位与检测精度。**

- **链接: [https://arxiv.org/pdf/2604.02109](https://arxiv.org/pdf/2604.02109)**

> **作者:** Lukas Bergs; Tan Chung; Marmik Thakkar; Alexander Moriz; Amon Göppert; Chinnawut Nantabut; Robert Schmitt
>
> **备注:** Accepted for publication at CIRP ICME 2025; will appear in Procedia CIRP
>
> **摘要:** Adaptive robots in dynamic production environments require robust perception capabilities, including 6D pose estimation and multi-object tracking. To address limitations in real-world data dependency, noise robustness, and spatiotemporal consistency, a LiDAR framework based on the Robot Operating System integrating a synthetic-data-trained Transformation-Equivariant 3D Detection with multi-object-tracking leveraging center poses is proposed. Validated across 72 scenarios with motion capture technology, overall results yield an Intersection over Union of 62.6% for standalone pose estimation, rising to 83.12% with multi-object-tracking integration. Our LiDAR-based framework achieves 91.12% of Higher Order Tracking Accuracy, advancing robustness and versatility of LiDAR-based perception systems for industrial mobile manipulators.
>
---
#### [new 009] Bench2Drive-VL: Benchmarks for Closed-Loop Autonomous Driving with Vision-Language Models
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决VLM在闭环评估中的不足，通过构建Bench2Drive-VL基准，提供闭环测试环境与工具，提升VLM在复杂驾驶场景中的性能评估。**

- **链接: [https://arxiv.org/pdf/2604.01259](https://arxiv.org/pdf/2604.01259)**

> **作者:** Xiaosong Jia; Yuqian Shao; Zhenjie Yang; Qifeng Li; Zhiyuan Zhang; Junchi Yan
>
> **备注:** All codes and annotated datasets are available at \url{this https URL} and \url{this https URL}
>
> **摘要:** With the rise of vision-language models (VLM), their application for autonomous driving (VLM4AD) has gained significant attention. Meanwhile, in autonomous driving, closed-loop evaluation has become widely recognized as a more reliable validation method than open-loop evaluation, as it can evaluate the performance of the model under cumulative errors and out-of-distribution inputs. However, existing VLM4AD benchmarks evaluate the model`s scene understanding ability under open-loop, i.e., via static question-answer (QA) dataset. This kind of evaluation fails to assess the VLMs performance under out-of-distribution states rarely appeared in the human collected this http URL this end, we present Bench2Drive-VL, an extension of Bench2Drive that brings closed-loop evaluation to VLM-based driving, which introduces: (1) DriveCommenter, a closed-loop generator that automatically generates diverse, behavior-grounded question-answer pairs for all driving situations in CARLA,including severe off-route and off-road deviations previously unassessable in simulation. (2) A unified protocol and interface that allows modern VLMs to be directly plugged into the Bench2Drive closed-loop environment to compare with traditional agents. (3) A flexible reasoning and control framework, supporting multi-format visual inputs and configurable graph-based chain-of-thought execution. (4) A complete development ecosystem. Together, these components form a comprehensive closed-loop benchmark for VLM4AD. All codes and annotated datasets are open sourced.
>
---
#### [new 010] PRO-SPECT: Probabilistically Safe Scalable Planning for Energy-Aware Coordinated UAV-UGV Teams in Stochastic Environments
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多智能体路径规划任务，解决UAV-UGV团队在随机环境中的能耗安全问题。通过概率约束建模，提出PRO-SPECT算法实现风险可控的高效规划。**

- **链接: [https://arxiv.org/pdf/2604.02142](https://arxiv.org/pdf/2604.02142)**

> **作者:** Roger Fowler; Cahit Ikbal Er; Benjamin Johnsenberg; Yasin Yazicioglu
>
> **摘要:** We consider energy-aware planning for an unmanned aerial vehicle (UAV) and unmanned ground vehicle (UGV) team operating in a stochastic environment. The UAV must visit a set of air points in minimum time while respecting energy constraints, relying on the UGV as a mobile charging station. Unlike prior work that assumed deterministic travel times or used fixed robustness margins, we model travel times as random variables and bound the probability of failure (energy depletion) across the entire mission to a user-specified risk level. We formulate the problem as a Mixed-Integer Program and propose PRO-SPECT, a polynomial-time algorithm that generates risk-bounded plans. The algorithm supports both offline planning and online re-planning, enabling the team to adapt to disturbances while preserving the risk bound. We provide theoretical results on solution feasibility and time complexity. We also demonstrate the performance of our method via numerical comparisons and simulations.
>
---
#### [new 011] Stop Wandering: Efficient Vision-Language Navigation via Metacognitive Reasoning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，解决传统方法效率低的问题。提出MetaNav，通过元认知机制提升导航效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.02318](https://arxiv.org/pdf/2604.02318)**

> **作者:** Xueying Li; Feng Lyu; Hao Wu; Mingliu Liu; Jia-Nan Liu; Guozi Liu
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Training-free Vision-Language Navigation (VLN) agents powered by foundation models can follow instructions and explore 3D environments. However, existing approaches rely on greedy frontier selection and passive spatial memory, leading to inefficient behaviors such as local oscillation and redundant revisiting. We argue that this stems from a lack of metacognitive capabilities: the agent cannot monitor its exploration progress, diagnose strategy failures, or adapt accordingly. To address this, we propose MetaNav, a metacognitive navigation agent integrating spatial memory, history-aware planning, and reflective correction. Spatial memory builds a persistent 3D semantic map. History-aware planning penalizes revisiting to improve efficiency. Reflective correction detects stagnation and uses an LLM to generate corrective rules that guide future frontier selection. Experiments on GOAT-Bench, HM3D-OVON, and A-EQA show that MetaNav achieves state-of-the-art performance while reducing VLM queries by 20.7%, demonstrating that metacognitive reasoning significantly improves robustness and efficiency.
>
---
#### [new 012] Low-Burden LLM-Based Preference Learning: Personalizing Assistive Robots from Natural Language Feedback for Users with Paralysis
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于个性化机器人控制任务，旨在解决瘫痪用户因传统方法导致的高负担问题。通过自然语言反馈生成安全控制策略，提升用户体验。**

- **链接: [https://arxiv.org/pdf/2604.01463](https://arxiv.org/pdf/2604.01463)**

> **作者:** Keshav Shankar; Dan Ding; Wei Gao
>
> **备注:** This work has been submitted to the 2026 IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
>
> **摘要:** Physically Assistive Robots (PARs) require personalized behaviors to ensure user safety and comfort. However, traditional preference learning methods, like exhaustive pairwise comparisons, cause severe physical and cognitive fatigue for users with profound motor impairments. To solve this, we propose a low-burden, offline framework that translates unstructured natural language feedback directly into deterministic robotic control policies. To safely bridge the gap between ambiguous human speech and robotic code, our pipeline uses Large Language Models (LLMs) grounded in the Occupational Therapy Practice Framework (OTPF). This clinical reasoning decodes subjective user reactions into explicit physical and psychological needs, which are then mapped into transparent decision trees. Before deployment, an automated "LLM-as-a-Judge" verifies the code's structural safety. We validated this system in a simulated meal preparation study with 10 adults with paralysis. Results show our natural language approach significantly reduces user workload compared to traditional baselines. Additionally, independent clinical experts confirmed the generated policies are safe and accurately reflect user preferences.
>
---
#### [new 013] Simulating Realistic LiDAR Data Under Adverse Weather for Autonomous Vehicles: A Physics-Informed Learning Approach
- **分类: cs.RO; eess.IV**

- **简介: 该论文属于自动驾驶中的LiDAR数据仿真任务，旨在解决恶劣天气下LiDAR数据不真实的问题。通过物理信息学习框架，生成更真实的LiDAR数据，提升感知性能。**

- **链接: [https://arxiv.org/pdf/2604.01254](https://arxiv.org/pdf/2604.01254)**

> **作者:** Vivek Anand; Bharat Lohani; Rakesh Mishra; Gaurav Pandey
>
> **摘要:** Accurate LiDAR simulation is crucial for autonomous driving, especially under adverse weather conditions. Existing methods struggle to capture the complex interactions between LiDAR signals and atmospheric phenomena, leading to unrealistic representations. This paper presents a physics-informed learning framework (PICWGAN) for generating realistic LiDAR data under adverse weather conditions. By integrating physicsdriven constraints for modeling signal attenuation and geometryconsistent degradations into a physics-informed learning pipeline, the proposed method reduces the sim-to-real gap. Evaluations on real-world datasets (CADC for snow, Boreas for rain) and the VoxelScape dataset show that our approach closely mimics realworld intensity patterns. Quantitative metrics, including MSE, SSIM, KL divergence, and Wasserstein distance, demonstrate statistically consistent intensity distributions. Additionally, models trained on data enhanced by our framework outperform baselines in downstream 3D object detection, achieving performance comparable to models trained on real-world data. These results highlight the effectiveness of the proposed approach in improving the realism of LiDAR data and enabling robust perception under adverse weather conditions.
>
---
#### [new 014] Learning When to See and When to Feel: Adaptive Vision-Torque Fusion for Contact-Aware Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决接触敏感任务中视觉与力觉信息融合的问题。通过比较不同融合策略，提出一种自适应融合方法，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2604.01414](https://arxiv.org/pdf/2604.01414)**

> **作者:** Jiuzhou Lei; Chang Liu; Yu She; Xiao Liang; Minghui Zheng
>
> **摘要:** Vision-based policies have achieved a good performance in robotic manipulation due to the accessibility and richness of visual observations. However, purely visual sensing becomes insufficient in contact-rich and force-sensitive tasks where force/torque (F/T) signals provide critical information about contact dynamics, alignment, and interaction quality. Although various strategies have been proposed to integrate vision and F/T signals, including auxiliary prediction objectives, mixture-of-experts architectures, and contact-aware gating mechanisms, a comparison of these approaches remains lacking. In this work, we provide a comparison study of different F/T-vision integration strategies within diffusion-based manipulation policies. In addition, we propose an adaptive integration strategy that ignores F/T signals during non-contact phases while adaptively leveraging both vision and torque information during contact. Experimental results demonstrate that our method outperforms the strongest baseline by 14% in success rate, highlighting the importance of contact-aware multimodal fusion for robotic manipulation.
>
---
#### [new 015] Hi-LOAM: Hierarchical Implicit Neural Fields for LiDAR Odometry and Mapping
- **分类: cs.RO**

- **简介: 该论文属于LiDAR位姿估计与建图任务，旨在解决现有方法依赖监督信号或重建精度不足的问题。提出Hi-LOAM框架，利用多尺度隐式神经场实现高精度定位与地图构建。**

- **链接: [https://arxiv.org/pdf/2604.01720](https://arxiv.org/pdf/2604.01720)**

> **作者:** Zhiliu Yang; Jianyuan Zhang; Lianhui Zhao; Jinyu Dai; Zhu Yang
>
> **备注:** This manuscript is the accepted version of IEEE Transactions on Multimedia
>
> **摘要:** LiDAR Odometry and Mapping (LOAM) is a pivotal technique for embodied-AI applications such as autonomous driving and robot navigation. Most existing LOAM frameworks are either contingent on the supervision signal, or lack of the reconstruction fidelity, which are deficient in depicting details of large-scale complex scenes. To overcome these limitations, we propose a multi-scale implicit neural localization and mapping framework using LiDAR sensor, called Hi-LOAM. Hi-LOAM receives LiDAR point cloud as the input data modality, learns and stores hierarchical latent features in multiple levels of hash tables based on an octree structure, then these multi-scale latent features are decoded into signed distance value through shallow Multilayer Perceptrons (MLPs) in the mapping procedure. For pose estimation procedure, we rely on a correspondence-free, scan-to-implicit matching paradigm to estimate optimal pose and register current scan into the submap. The entire training process is conducted in a self-supervised manner, which waives the model pre-training and manifests its generalizability when applied to diverse environments. Extensive experiments on multiple real-world and synthetic datasets demonstrate the superior performance, in terms of the effectiveness and generalization capabilities, of our Hi-LOAM compared to existing state-of-the-art methods.
>
---
#### [new 016] Bridging Discrete Planning and Continuous Execution for Redundant Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决冗余机械臂在离散规划与连续执行间的问题。通过引入优化策略和逆运动学方法，提升路径质量与执行稳定性。**

- **链接: [https://arxiv.org/pdf/2604.02021](https://arxiv.org/pdf/2604.02021)**

> **作者:** Teng Yan; Yue Yu; Yihan Liu; Bingzhuo Zhong
>
> **备注:** 8 pages, 3 figures. Submitted to IFAC World Congress 2026
>
> **摘要:** Voxel-grid reinforcement learning is widely adopted for path planning in redundant manipulators due to its simplicity and reproducibility. However, direct execution through point-wise numerical inverse kinematics on 7-DoF arms often yields step-size jitter, abrupt joint transitions, and instability near singular configurations. This work proposes a bridging framework between discrete planning and continuous execution without modifying the discrete planner itself. On the planning side, step-normalized 26-neighbor Cartesian actions and a geometric tie-breaking mechanism are introduced to suppress unnecessary turns and eliminate step-size oscillations. On the execution side, a task-priority damped least-squares (TP-DLS) inverse kinematics layer is implemented. This layer treats end-effector position as a primary task, while posture and joint centering are handled as subordinate tasks projected into the null space, combined with trust-region clipping and joint velocity constraints. On a 7-DoF manipulator in random sparse, medium, and dense environments, this bridge raises planning success in dense scenes from about 0.58 to 1.00, shortens representative path length from roughly 1.53 m to 1.10 m, and while keeping end-effector error below 1 mm, reduces peak joint accelerations by over an order of magnitude, substantially improving the continuous execution quality of voxel-based RL paths on redundant manipulators.
>
---
#### [new 017] Bridging Large-Model Reasoning and Real-Time Control via Agentic Fast-Slow Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主系统控制任务，旨在解决从语义意图到实时控制的映射问题。提出Agentic Fast-Slow Planning框架，实现感知、决策、规划与控制的分层协同。**

- **链接: [https://arxiv.org/pdf/2604.01681](https://arxiv.org/pdf/2604.01681)**

> **作者:** Jiayi Chen; Shuai Wang; Guangxu Zhu; Chengzhong Xu
>
> **备注:** 8 pages, 12figures
>
> **摘要:** Large foundation models enable powerful reasoning for autonomous systems, but mapping semantic intent to reliable real-time control remains challenging. Existing approaches either (i) let Large Language Models (LLMs) generate trajectories directly - brittle, hard to verify, and latency-prone - or (ii) adjust Model Predictive Control (MPC) objectives online - mixing slow deliberation with fast control and blurring interfaces. We propose Agentic Fast-Slow Planning, a hierarchical framework that decouples perception, reasoning, planning, and control across natural timescales. The framework contains two bridges. Perception2Decision compresses scenes into ego-centric topologies using an on-vehicle Vision-Language Model (VLM) detector, then maps them to symbolic driving directives in the cloud with an LLM decision maker - reducing bandwidth and delay while preserving interpretability. Decision2Trajectory converts directives into executable paths: Semantic-Guided A* embeds language-derived soft costs into classical search to bias solutions toward feasible trajectories, while an Agentic Refinement Module adapts planner hyperparameters using feedback and memory. Finally, MPC tracks the trajectories in real time, with optional cloud-guided references for difficult cases. Experiments in CARLA show that Agentic Fast-Slow Planning improves robustness under perturbations, reducing lateral deviation by up to 45% and completion time by over 12% compared to pure MPC and an A*-guided MPC baseline. Code is available at this https URL.
>
---
#### [new 018] OpenGo: An OpenClaw-Based Robotic Dog with Real-Time Skill Switching
- **分类: cs.RO; cs.AI**

- **简介: 该论文介绍OpenGo，一种具备实时技能切换能力的机器人狗，解决单一机器人适应复杂任务的问题。通过技能库、调度器和自学习框架实现动态环境下的多任务处理。**

- **链接: [https://arxiv.org/pdf/2604.01708](https://arxiv.org/pdf/2604.01708)**

> **作者:** Hanbing Li; Xuewei Cao; Zhiwen Zeng; Yuhan Wu; Yanyong Zhang; Yan Xia
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Adaptation to complex tasks and multiple scenarios remains a significant challenge for a single robot agent. The ability to acquire organize, and switch between a wide range of skills in real time, particularly in dynamic environments, has become a fundamental requirement for embodied intelligence. We introduce OpenGo, an OpenClaw-powered embodied robotic dog capable of switching skills in real time according to the scene and task instructions. Specifically, the agent is equipped with (1) a customizable skill library with easy skill import and autonomous skill validation, (2) a dispatcher that selects and invokes different skills according to task prompts or language instructions, and (3) a self-learning framework that fine-tunes skills based on task completion and human feedback. We deploy the agent in Unitree's Go2 robotic dog and validate its capabilities in self-checking and switching of skills autonomously. In addition, by integrating Feishu-platform communication, we enable natural-language guidance and human feedback, allowing inexperienced users to control the robotic dog through simple instructions.
>
---
#### [new 019] AnchorVLA: Anchored Diffusion for Efficient End-to-End Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出AnchorVLA，解决移动操作中的多动作模型保持与实时反应问题，通过锚定扩散策略提升效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.01567](https://arxiv.org/pdf/2604.01567)**

> **作者:** Jia Syuen Lim; Zhizhen Zhang; Peter Bohm; Brendan Tidd; Zi Huang; Yadan Luo
>
> **摘要:** A central challenge in mobile manipulation is preserving multiple plausible action models while remaining reactive during execution. A bottle in a cluttered scene can often be approached and grasped in multiple valid ways. Robust behavior depends on preserving this action diversity while remaining reactive as the scene evolves. Diffusion policies are appealing because they model multimodal action distributions rather than collapsing to one solution. But in practice, full iterative denoising is costly at control time. Action chunking helps amortize inference, yet it also creates partially open-loop behavior, allowing small mismatches to accumulate into drift. We present AnchorVLA, a diffusion-based VLA policy for mobile manipulation built on the core insight that when sampling begins near a plausible solution manifold, extensive denoising is unnecessary to recover multimodal, valid actions. AnchorVLA combines a lightweight VLA adaptation backbone with an anchored diffusion action head, which denoises locally around anchor trajectories using a truncated diffusion schedule. This retains multimodal action generation while reducing inference cost for closed-loop control. Crucially, to mitigate chunking-induced drift, we introduce a test-time self-correction mechanism via a lightweight residual correction module that makes high-frequency, per-step adjustments during rollout. Across diverse mobile manipulation tasks, AnchorVLA improves success and stability under disturbances and distribution shifts while maintaining low-latency inference. The source code is made available at this https URL.
>
---
#### [new 020] 3-D Relative Localization for Multi-Robot Systems with Angle and Self-Displacement Measurements
- **分类: cs.RO**

- **简介: 该论文属于多机器人系统相对定位任务，解决测量噪声下的3D相对定位问题。提出线性定位理论和MAP估计方法，提升定位精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.01703](https://arxiv.org/pdf/2604.01703)**

> **作者:** Chenyang Liang; Liangming Chen; Baoyi Cui; Jie Mei
>
> **备注:** 29 pages, 28 figures
>
> **摘要:** Realizing relative localization by leveraging inter-robot local measurements is a challenging problem, especially in the presence of measurement noise. Motivated by this challenge, in this paper we propose a novel and systematic 3-D relative localization framework based on inter-robot interior angle and self-displacement measurements. Initially, we propose a linear relative localization theory comprising a distributed linear relative localization algorithm and sufficient conditions for localizability. According to this theory, robots can determine their neighbors' relative positions and orientations in a purely linear manner. Subsequently, in order to deal with measurement noise, we present an advanced Maximum a Posterior (MAP) estimator by addressing three primary challenges existing in the MAP estimator. Firstly, it is common to formulate the MAP problem as an optimization problem, whose inherent non-convexity can result in local optima. To address this issue, we reformulate the linear computation process of the linear relative localization algorithm as a Weighted Total Least Squares (WTLS) optimization problem on manifolds. The optimal solution of the WTLS problem is more accurate, which can then be used as initial values when solving the optimization problem associated with the MAP problem, thereby reducing the risk of falling into local optima. The second challenge is the lack of knowledge of the prior probability density of the robots' relative positions and orientations at the initial time, which is required as an input for the MAP estimator. To deal with it, we combine the WTLS with a Neural Density Estimator (NDE). Thirdly, to prevent the increasing size of the relative positions and orientations to be estimated as the robots continuously move when solving the MAP problem, a marginalization mechanism is designed, which ensures that the computational cost remains constant.
>
---
#### [new 021] AURA: Multimodal Shared Autonomy for Real-World Urban Navigation
- **分类: cs.RO**

- **简介: 该论文提出AURA框架，解决城市导航中人机协作效率低的问题。通过分解高/低层控制，提升导航稳定性与指令跟随能力。**

- **链接: [https://arxiv.org/pdf/2604.01659](https://arxiv.org/pdf/2604.01659)**

> **作者:** Yukai Ma; Honglin He; Selina Song; Wayne Wu; Bolei Zhou
>
> **备注:** 17 pages, 18 figures, 4 tables, conference
>
> **摘要:** Long-horizon navigation in complex urban environments relies heavily on continuous human operation, which leads to fatigue, reduced efficiency, and safety concerns. Shared autonomy, where a Vision-Language AI agent and a human operator collaborate on maneuvering the mobile machine, presents a promising solution to address these issues. However, existing shared autonomy methods often require humans and AI to operate within the same action space, leading to high cognitive overhead. We present Assistive Urban Robot Autonomy (AURA), a new multi-modal framework that decomposes urban navigation into high-level human instruction and low-level AI control. AURA incorporates a Spatial-Aware Instruction Encoder to align various human instructions with visual and spatial context. To facilitate training, we construct MM-CoS, a large-scale dataset comprising teleoperation and vision-language descriptions. Experiments in simulation and the real world demonstrate that AURA effectively follows human instructions, reduces manual operation effort, and improves navigation stability, while enabling online adaptation. Moreover, under similar takeover conditions, our shared autonomy framework reduces the frequency of takeovers by more than 44%. Demo video and more detail are provided in the project page.
>
---
#### [new 022] Causal Scene Narration with Runtime Safety Supervision for Vision-Language-Action Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶任务，解决VLA模型中文本输入碎片化问题。提出CSN方法重构文本输入，提升驾驶表现，并引入安全监督机制。**

- **链接: [https://arxiv.org/pdf/2604.01723](https://arxiv.org/pdf/2604.01723)**

> **作者:** Yun Li; Yidu Zhang; Simon Thompson; Ehsan Javanmardi; Manabu Tsukada
>
> **备注:** 18 pages, 6 figures, 4 tables
>
> **摘要:** Vision-Language-Action (VLA) models for autonomous driving must integrate diverse textual inputs, including navigation commands, hazard warnings, and traffic state descriptions, yet current systems often present these as disconnected fragments, forcing the model to discover on its own which environmental constraints are relevant to the current maneuver. We introduce Causal Scene Narration (CSN), which restructures VLA text inputs through intent-constraint alignment, quantitative grounding, and structured separation, at inference time with zero GPU cost. We complement CSN with Simplex-based runtime safety supervision and training-time alignment via Plackett-Luce DPO with negative log-likelihood (NLL) regularization. A multi-town closed-loop CARLA evaluation shows that CSN improves Driving Score by +31.1% on original LMDrive and +24.5% on the preference-aligned variant. A controlled ablation reveals that causal structure accounts for 39.1% of this gain, with the remainder attributable to information content alone. A perception noise ablation confirms that CSN's benefit is robust to realistic sensing errors. Semantic safety supervision improves Infraction Score, while reactive Time-To-Collision monitoring degrades performance, demonstrating that intent-aware monitoring is needed for VLA systems.
>
---
#### [new 023] Robust Autonomous Control of a Magnetic Millirobot in In Vitro Cardiac Flow
- **分类: cs.RO**

- **简介: 该论文属于自主控制任务，旨在解决磁微机器人在心脏流体中的可靠导航问题。通过视觉引导的控制框架，实现精准定位与路径规划，提升控制鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.01523](https://arxiv.org/pdf/2604.01523)**

> **作者:** Anuruddha Bhattacharjee; Xinhao Chen; Lamar O. Mair; Suraj Raval; Yancy Diaz-Mercado; Axel Krieger
>
> **摘要:** Untethered magnetic millirobots offer significant potential for minimally invasive cardiac therapies; however, achieving reliable autonomous control in pulsatile cardiac flow remains challenging. This work presents a vision-guided control framework enabling precise autonomous navigation of a magnetic millirobot in an in vitro heart phantom under physiologically relevant flow conditions. The system integrates UNet-based localization, A* path planning, and a sliding mode controller with a disturbance observer (SMC-DOB) designed for multi-coil electromagnetic actuation. Although drag forces are estimated using steady-state CFD simulations, the controller compensates for transient pulsatile disturbances during closed-loop operation. In static fluid, the SMC-DOB achieved sub-millimeter accuracy (root-mean-square error, RMSE = 0.49 mm), outperforming PID and MPC baselines. Under moderate pulsatile flow (7 cm/s peak, 20 cP), it reduced RMSE by 37% and peak error by 2.4$\times$ compared to PID. It further maintained RMSE below 2 mm (0.27 body lengths) under elevated pulsatile flow (10 cm/s peak, 20 cP) and under low-viscosity conditions (4.3 cP, 7 cm/s peak), where baseline controllers exhibited unstable or failed tracking. These results demonstrate robust closed-loop magnetic control under time-varying cardiac flow disturbances and support the feasibility of autonomous millirobot navigation for targeted drug delivery.
>
---
#### [new 024] A Graph Neural Network Approach for Solving the Ranked Assignment Problem in Multi-Object Tracking
- **分类: cs.RO**

- **简介: 该论文属于多目标跟踪任务，解决数据关联中的排序分配问题。提出RAPNet方法，利用图神经网络提升准确率。**

- **链接: [https://arxiv.org/pdf/2604.01696](https://arxiv.org/pdf/2604.01696)**

> **作者:** Robin Dehler; Martin Herrmann; Jan Strohbeck; Michael Buchholz
>
> **备注:** 2024 IEEE Intelligent Vehicles Symposium (IV)
>
> **摘要:** Associating measurements with tracks is a crucial step in Multi-Object Tracking (MOT) to guarantee the safety of autonomous vehicles. To manage the exponentially growing number of track hypotheses, truncation becomes necessary. In the $\delta$-Generalized Labeled Multi-Bernoulli ($\delta$-GLMB) filter application, this truncation typically involves the ranked assignment problem, solved by Murty's algorithm or the Gibbs sampling approach, both with limitations in terms of complexity or accuracy, respectively. With the motivation to improve these limitations, this paper addresses the ranked assignment problem arising from data association tasks with an approach that employs Graph Neural Networks (GNNs). The proposed Ranked Assignment Prediction Graph Neural Network (RAPNet) uses bipartite graphs to model the problem, harnessing the computational capabilities of deep learning. The conclusive evaluation compares the RAPNet with Murty's algorithm and the Gibbs sampler, showing accuracy improvements compared to the Gibbs sampler.
>
---
#### [new 025] Cross-Modal Visuo-Tactile Object Perception
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于多模态感知任务，旨在提升机器人在不确定环境下对物体物理属性的估计能力。通过提出CMLF模型，实现视觉与触觉信息的动态融合与推理。**

- **链接: [https://arxiv.org/pdf/2604.02108](https://arxiv.org/pdf/2604.02108)**

> **作者:** Anirvan Dutta; Simone Tasciotti; Claudia Cusseddu; Ang Li; Panayiota Poirazi; Julijana Gjorgjieva; Etienne Burdet; Patrick van der Smagt; Mohsen Kaboli
>
> **备注:** 23 pages, 8 figures, 1 table. Submitted for review to journal
>
> **摘要:** Estimating physical properties is critical for safe and efficient autonomous robotic manipulation, particularly during contact-rich interactions. In such settings, vision and tactile sensing provide complementary information about object geometry, pose, inertia, stiffness, and contact dynamics, such as stick-slip behavior. However, these properties are only indirectly observable and cannot always be modeled precisely (e.g., deformation in non-rigid objects coupled with nonlinear contact friction), making the estimation problem inherently complex and requiring sustained exploitation of visuo-tactile sensory information during action. Existing visuo-tactile perception frameworks have primarily emphasized forceful sensor fusion or static cross-modal alignment, with limited consideration of how uncertainty and beliefs about object properties evolve over time. Inspired by human multi-sensory perception and active inference, we propose the Cross-Modal Latent Filter (CMLF) to learn a structured, causal latent state-space of physical object properties. CMLF supports bidirectional transfer of cross-modal priors between vision and touch and integrates sensory evidence through a Bayesian inference process that evolves over time. Real-world robotic experiments demonstrate that CMLF improves the efficiency and robustness of latent physical properties estimation under uncertainty compared to baseline approaches. Beyond performance gains, the model exhibits perceptual coupling phenomena analogous to those observed in humans, including susceptibility to cross-modal illusions and similar trajectories in learning cross-sensory associations. Together, these results constitutes a significant step toward generalizable, robust and physically consistent cross-modal integration for robotic multi-sensory perception.
>
---
#### [new 026] Efficient Equivariant Transformer for Self-Driving Agent Modeling
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于自动驾驶中的智能体行为建模任务，旨在解决SE(2)-equivariance的高效建模问题。提出DriveGATr架构，通过几何代数方法实现等变性，避免了高成本的显式位置编码。**

- **链接: [https://arxiv.org/pdf/2604.01466](https://arxiv.org/pdf/2604.01466)**

> **作者:** Scott Xu; Dian Chen; Kelvin Wong; Chris Zhang; Kion Fallah; Raquel Urtasun
>
> **备注:** CVPR 2026
>
> **摘要:** Accurately modeling agent behaviors is an important task in self-driving. It is also a task with many symmetries, such as equivariance to the order of agents and objects in the scene or equivariance to arbitrary roto-translations of the entire scene as a whole; i.e., SE(2)-equivariance. The transformer architecture is a ubiquitous tool for modeling these symmetries. While standard self-attention is inherently permutation equivariant, explicit pairwise relative positional encodings have been the standard for introducing SE(2)-equivariance. However, this approach introduces an additional cost that is quadratic in the number of agents, limiting its scalability to larger scenes and batch sizes. In this work, we propose DriveGATr, a novel transformer-based architecture for agent modeling that achieves SE(2)-equivariance without the computational cost of existing methods. Inspired by recent advances in geometric deep learning, DriveGATr encodes scene elements as multivectors in the 2D projective geometric algebra $\mathbb{R}^*_{2,0,1}$ and processes them with a stack of equivariant transformer blocks. Crucially, DriveGATr models geometric relationships using standard attention between multivectors, eliminating the need for costly explicit pairwise relative positional encodings. Experiments on the Waymo Open Motion Dataset demonstrate that DriveGATr is comparable to the state-of-the-art in traffic simulation and establishes a superior Pareto front for performance vs computational cost.
>
---
#### [new 027] Smooth Feedback Motion Planning with Reduced Curvature
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，旨在解决现有方法生成路径弯曲过多的问题。通过引入启发式方法和几何算法，生成更直接的轨迹，减少控制成本。**

- **链接: [https://arxiv.org/pdf/2604.01614](https://arxiv.org/pdf/2604.01614)**

> **作者:** Aref Amiri; Steven M. LaValle
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters
>
> **摘要:** Feedback motion planning over cell decompositions provides a robust method for generating collision-free robot motion with formal guarantees. However, existing algorithms often produce paths with unnecessary bending, leading to slower motion and higher control effort. This paper presents a computationally efficient method to mitigate this issue for a given simplicial decomposition. A heuristic is introduced that systematically aligns and assigns local vector fields to produce more direct trajectories, complemented by a novel geometric algorithm that constructs a maximal star-shaped chain of simplexes around the goal. This creates a large ``funnel'' in which an optimal, direct-to-goal control law can be safely applied. Simulations demonstrate that our method generates measurably more direct paths, reducing total bending by an average of 91.40\% and LQR control effort by an average of 45.47\%. Furthermore, comparative analysis against sampling-based and optimization-based planners confirms the time efficacy and robustness of our approach. While the proposed algorithms work over any finite-dimensional simplicial complex embedded in the collision-free subset of the configuration space, the practical application focuses on low-dimensional ($d\le3$) configuration spaces, where simplicial decomposition is computationally tractable.
>
---
#### [new 028] Open-loop POMDP Simplification and Safe Skipping of Replanning with Formal Performance Guarantees
- **分类: cs.RO**

- **简介: 该论文属于强化学习中的决策规划任务，解决POMDP求解计算复杂的问题。通过引入自适应开环简化框架，实现高效且安全的策略生成与重规划跳过，提供形式化性能保证。**

- **链接: [https://arxiv.org/pdf/2604.01352](https://arxiv.org/pdf/2604.01352)**

> **作者:** Da Kong; Vadim Indelman
>
> **备注:** 18 pages, 5 figures. Accepted to WAFR 2026
>
> **摘要:** Partially Observable Markov Decision Processes (POMDPs) provide a principled mathematical framework for decision-making under uncertainty. However, the exact solution to POMDPs is computationally intractable. In this paper, we address the computational intractability by introducing a novel framework for adaptive open-loop simplification with formal performance guarantees. Our method adaptively interleaves open-loop and closed-loop planning via a topology-based belief tree, enabling a significant reduction in planning complexity. The key contribution lies in the derivation of efficiently computable bounds which provide formal guarantees and can be used to ensure that our simplification can identify the immediate optimal action of the original POMDP problem. Our framework therefore provides computationally tractable performance guarantees for macro-actions within POMDPs. Furthermore, we propose a novel framework for safely skipping replanning during execution, supported by theoretical guarantees on multi-step open-loop action sequences. To the best of our knowledge, this framework is the first to address skipping replanning with formal performance guarantees. Practical online solvers for our proposed simplification are developed, including a sampling-based solver and an anytime solver. Empirical results demonstrate substantial computational speedups while maintaining provable performance guarantees, advancing the tractability and efficiency of POMDP planning.
>
---
#### [new 029] A soft and lightweight fabric-based pneumatic interface for multimodal fingertip tactile feedback
- **分类: cs.RO**

- **简介: 该论文属于触觉反馈任务，旨在解决可穿戴指尖触觉设备在力、重量、成本和便携性上的难题。研究提出一种基于织物的气动装置，实现轻量、柔软且多模式的触觉反馈。**

- **链接: [https://arxiv.org/pdf/2604.01390](https://arxiv.org/pdf/2604.01390)**

> **作者:** Rui Chen; Daniele Leonardis; Antonio Frisoli
>
> **摘要:** Wearable fingertip haptic devices are critical for realistic interaction in virtual reality, augmented reality, and teleoperation, yet existing approaches struggle to simultaneously achieve adequate tactile output, low mass, simple fabrication, and untethered portability. Here we show that fabric-based pneumatic actuation can address this gap. Our device comprises four pneumatic chambers fabricated from thermoplastic polyurethane-coated fabric via computer numerical control heat-sealing, yielding a soft, conformable interface weighing 2.1 g that operates untethered with a wrist-mounted control unit. Mechanical and dynamic characterization confirms that the fabric actuators produce sufficient force, displacement, and bandwidth for fingertip tactile rendering. A psychophysical study with 15 participants demonstrates classification accuracy exceeding 90% across three distinct tactile modes -- contact configuration, directional sliding, and vibrotactile frequency. These findings establish fabric-based pneumatic actuation as a viable technology route for lightweight, low-cost, and multimodal fingertip haptic interfaces.
>
---
#### [new 030] Realistic Lip Motion Generation Based on 3D Dynamic Viseme and Coarticulation Modeling for Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文属于语音驱动唇部运动生成任务，解决人机交互中唇同步不自然的问题。通过构建3D动态视觉库和共音机制，实现高效精准的唇部动作控制。**

- **链接: [https://arxiv.org/pdf/2604.01756](https://arxiv.org/pdf/2604.01756)**

> **作者:** Sheng Li; Jingcheng Huang; Min Li
>
> **备注:** 8 pages,7 figures
>
> **摘要:** Realistic lip synchronization is essential for the natural human-robot non-verbal interaction of humanoid robots. Motivated by this need, this paper presents a lip motion generation framework based on 3D dynamic viseme and coarticulation modeling. By analyzing Chinese pronunciation theory, a 3D dynamic viseme library is constructed based on the ARKit standard, which offers coherent prior trajectories of lips. To resolve motion conflicts within continuous speech streams, a coarticulation mechanism is developed by incorporating initial-final (Shengmu-Yunmu) decoupling and energy modulation. After developing a strategy to retarget high-dimensional spatial lip motion to a 14-DOF lip actuation system of a humanoid head platform, the efficiency and accuracy of the proposed architecture is experimentally validated and demonstrated with quantitative ablation experiments using the metrics of the Pearson Correlation Coefficient (PCC) and the Mean Absolute Jerk (MAJ). This research offers a lightweight, efficient, and highly practical paradigm for the speech-driven lip motion generation of humanoid robots. The 3D dynamic viseme library and real-world deployment videos are available at {this https URL}
>
---
#### [new 031] HyVGGT-VO: Tightly Coupled Hybrid Dense Visual Odometry with Feed-Forward Models
- **分类: cs.RO**

- **简介: 该论文属于视觉里程计任务，旨在解决传统方法与前馈模型在计算效率和稠密重建上的矛盾。提出HyVGGT-VO框架，融合两者优势，提升精度与速度。**

- **链接: [https://arxiv.org/pdf/2604.02107](https://arxiv.org/pdf/2604.02107)**

> **作者:** Junxiang Pan; Lipu Zhou; Baojie Chen
>
> **摘要:** Dense visual odometry (VO), which provides pose estimation and dense 3D reconstruction, serves as the cornerstone for applications ranging from robotics to augmented reality. Recently, feed-forward models have demonstrated remarkable capabilities in dense mapping. However, when these models are used in dense visual SLAM systems, their heavy computational burden restricts them to yielding sparse pose outputs at keyframes while still failing to achieve real-time pose estimation. In contrast, traditional sparse methods provide high computational efficiency and high-frequency pose outputs, but lack the capability for dense reconstruction. To address these limitations, we propose HyVGGT-VO, a novel framework that combines the computational efficiency of sparse VO with the dense reconstruction capabilities of feed-forward models. To the best of our knowledge, this is the first work to tightly couple a traditional VO framework with VGGT, a state-of-the-art feed-forward model. Specifically, we design an adaptive hybrid tracking frontend that dynamically switches between traditional optical flow and the VGGT tracking head to ensure robustness. Furthermore, we introduce a hierarchical optimization framework that jointly refines VO poses and the scale of VGGT predictions to ensure global scale consistency. Our approach achieves an approximately 5x processing speedup compared to existing VGGT-based methods, while reducing the average trajectory error by 85% on the indoor EuRoC dataset and 12% on the outdoor KITTI benchmark. Our code will be publicly available upon acceptance. Project page: this https URL.
>
---
#### [new 032] O-ConNet: Geometry-Aware End-to-End Inference of Over-Constrained Spatial Mechanisms
- **分类: cs.RO**

- **简介: 该论文提出O-ConNet，用于从稀疏点推断空间过约束机构的结构参数和运动轨迹，解决逆向设计问题。**

- **链接: [https://arxiv.org/pdf/2604.02038](https://arxiv.org/pdf/2604.02038)**

> **作者:** Haoyu Sun; Meng Zhao; Tianhao Wang; Jianxu Wu
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Deep learning has shown strong potential for scientific discovery, but its ability to model macroscopic rigid-body kinematic constraints remains underexplored. We study this problem on spatial over-constrained mechanisms and propose O-ConNet, an end-to-end framework that infers mechanism structural parameters from only three sparse reachable points while reconstructing the full motion trajectory, without explicitly solving constraint equations during inference. On a self-constructed Bennett 4R dataset of 42,860 valid samples, O-ConNet achieves Param-MAE 0.276 +/- 0.077 and Traj-MAE 0.145 +/- 0.018 (mean +/- std over 10 runs), outperforming the strongest sequence baseline (LSTM-Seq2Seq) by 65.1 percent and 88.2 percent, respectively. These results suggest that end-to-end learning can capture closed-loop geometric structure and provide a practical route for inverse design of spatial over-constrained mechanisms under extremely sparse observations.
>
---
#### [new 033] Deep Neural Network Based Roadwork Detection for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于道路施工检测任务，旨在解决自动驾驶中动态施工区域识别问题。通过融合YOLO与LiDAR数据，实现施工区域的实时检测与定位。**

- **链接: [https://arxiv.org/pdf/2604.02282](https://arxiv.org/pdf/2604.02282)**

> **作者:** Sebastian Wullrich; Nicolai Steinke; Daniel Goehring
>
> **备注:** 7 pages, 10 figures
>
> **摘要:** Road construction sites create major challenges for both autonomous vehicles and human drivers due to their highly dynamic and heterogeneous nature. This paper presents a real-time system that detects and localizes roadworks by combining a YOLO neural network with LiDAR data. The system identifies individual roadwork objects while driving, merges them into coherent construction sites and records their outlines in world coordinates. The model training was based on an adapted US dataset and a new dataset collected from test drives with a prototype vehicle in Berlin, Germany. Evaluations on real-world road construction sites showed a localization accuracy below 0.5 m. The system can support traffic authorities with up-to-date roadwork data and could enable autonomous vehicles to navigate construction sites more safely in the future.
>
---
#### [new 034] Global Geometry of Orthogonal Foliations in the Control Allocation of Signed-Quadratic Systems
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文研究控制分配中的几何问题，解决冗余解的拓扑结构问题。通过分析正二次系统，建立非线性空域的全局纤维结构，提出基于正交流形的分配方法，避免奇异性和几何退化。**

- **链接: [https://arxiv.org/pdf/2604.01912](https://arxiv.org/pdf/2604.01912)**

> **作者:** Antonio Franchi
>
> **备注:** Multimedia material attached
>
> **摘要:** This work formalizes the differential topology of redundancy resolution for systems governed by signed-quadratic actuation maps. By analyzing the minimally redundant case, the global topology of the continuous fiber bundle defining the nonlinear actuation null-space is established. The distribution orthogonal to these fibers is proven to be globally integrable and governed by an exact logarithmic potential field. This field foliates the actuator space, inducing a structural stratification of all orthants into transverse layers whose combinatorial sizes follow a strictly binomial progression. Within these layers, adjacent orthants are continuously connected via lower-dimensional strata termed reciprocal hinges, while the layers themselves are separated by boundary hyperplanes, or portals, that act as global sections of the fibers. This partition formally distinguishes extremal and transitional layers, which exhibit fundamentally distinct fiber topologies and foliation properties. Through this geometric framework, classical pseudo-linear static allocation strategies are shown to inevitably intersect singular boundary hyperplanes, triggering infinite-derivative kinetic singularities and fragmenting the task space into an exponential number of singularity-separated sectors. In contrast, allocators derived from the orthogonal manifolds yield continuously differentiable global sections with only a linear number of sectors for transversal layers, or can even form a single global diffeomorphism to the task space in the case of the two extremal layers, thus completely avoiding geometric rank-loss and boundary-crossing singularities. These theoretical results directly apply to the control allocation of propeller-driven architectures, including multirotor UAVs, marine, and underwater vehicles.
>
---
#### [new 035] Ego-Grounding for Personalized Question-Answering in Egocentric Videos
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于个性化问答任务，旨在解决egocentric视频中ego-grounding问题。提出MyEgo数据集，评估模型对“我”的理解与记忆能力，发现现有模型表现不佳，强调长时记忆的重要性。**

- **链接: [https://arxiv.org/pdf/2604.01966](https://arxiv.org/pdf/2604.01966)**

> **作者:** Junbin Xiao; Shenglang Zhang; Pengxiang Zhu; Angela Yao
>
> **备注:** To appear at CVPR'26
>
> **摘要:** We present the first systematic analysis of multimodal large language models (MLLMs) in personalized question-answering requiring ego-grounding - the ability to understand the camera-wearer in egocentric videos. To this end, we introduce MyEgo, the first egocentric VideoQA dataset designed to evaluate MLLMs' ability to understand, remember, and reason about the camera wearer. MyEgo comprises 541 long videos and 5K personalized questions asking about "my things", "my activities", and "my past". Benchmarking reveals that competitive MLLMs across variants, including open-source vs. proprietary, thinking vs. non-thinking, small vs. large scales all struggle on MyEgo. Top closed- and open-source models (e.g., GPT-5 and Qwen3-VL) achieve only~46% and 36% accuracy, trailing human performance by near 40% and 50% respectively. Surprisingly, neither explicit reasoning nor model scaling yield consistent improvements. Models improve when relevant evidence is explicitly provided, but gains drop over time, indicating limitations in tracking and remembering "me" and "my past". These findings collectively highlight the crucial role of ego-grounding and long-range memory in enabling personalized QA in egocentric videos. We hope MyEgo and our analyses catalyze further progress in these areas for egocentric personalized assistance. Data and code are available at this https URL
>
---
#### [new 036] DriveDreamer-Policy: A Geometry-Grounded World-Action Model for Unified Generation and Planning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出DriveDreamer-Policy，属于驾驶场景下的世界-动作建模任务，旨在提升物理环境中的几何感知与规划能力。通过整合深度生成、视频预测和路径规划，增强驾驶决策的准确性与连贯性。**

- **链接: [https://arxiv.org/pdf/2604.01765](https://arxiv.org/pdf/2604.01765)**

> **作者:** Yang Zhou; Xiaofeng Wang; Hao Shao; Letian Wang; Guosheng Zhao; Jiangnan Shao; Jiagang Zhu; Tingdong Yu; Zheng Zhu; Guan Huang; Steven L. Waslander
>
> **备注:** 11 pages, 4 figures; Project Website: this https URL
>
> **摘要:** Recently, world-action models (WAM) have emerged to bridge vision-language-action (VLA) models and world models, unifying their reasoning and instruction-following capabilities and spatio-temporal world modeling. However, existing WAM approaches often focus on modeling 2D appearance or latent representations, with limited geometric grounding-an essential element for embodied systems operating in the physical world. We present DriveDreamer-Policy, a unified driving world-action model that integrates depth generation, future video generation, and motion planning within a single modular architecture. The model employs a large language model to process language instructions, multi-view images, and actions, followed by three lightweight generators that produce depth, future video, and actions. By learning a geometry-aware world representation and using it to guide both future prediction and planning within a unified framework, the proposed model produces more coherent imagined futures and more informed driving actions, while maintaining modularity and controllable latency. Experiments on the Navsim v1 and v2 benchmarks demonstrate that DriveDreamer-Policy achieves strong performance on both closed-loop planning and world generation tasks. In particular, our model reaches 89.2 PDMS on Navsim v1 and 88.7 EPDMS on Navsim v2, outperforming existing world-model-based approaches while producing higher-quality future video and depth predictions. Ablation studies further show that explicit depth learning provides complementary benefits to video imagination and improves planning robustness.
>
---
#### [new 037] F3DGS: Federated 3D Gaussian Splatting for Decentralized Multi-Agent World Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多智能体3D重建任务，解决分布式场景下几何不一致和通信开销问题。提出F3DGS框架，通过联邦优化实现共享几何结构的分布式重建。**

- **链接: [https://arxiv.org/pdf/2604.01605](https://arxiv.org/pdf/2604.01605)**

> **作者:** Morui Zhu; Mohammad Dehghani Tezerjani; Mátyás Szántó; Márton Vaitkus; Song Fu; Qing Yang
>
> **备注:** Accepted to the CVPR 2026 SPAR-3D Workshop
>
> **摘要:** We present F3DGS, a federated 3D Gaussian Splatting framework for decentralized multi-agent 3D reconstruction. Existing 3DGS pipelines assume centralized access to all observations, which limits their applicability in distributed robotic settings where agents operate independently, and centralized data aggregation may be restricted. Directly extending centralized training to multi-agent systems introduces communication overhead and geometric inconsistency. F3DGS first constructs a shared geometric scaffold by registering locally merged LiDAR point clouds from multiple clients to initialize a global 3DGS model. During federated optimization, Gaussian positions are fixed to preserve geometric alignment, while each client updates only appearance-related attributes, including covariance, opacity, and spherical harmonic coefficients. The server aggregates these updates using visibility-aware aggregation, weighting each client's contribution by how frequently it observed each Gaussian, resolving the partial-observability challenge inherent to multi-agent exploration. To evaluate decentralized reconstruction, we collect a multi-sequence indoor dataset with synchronized LiDAR, RGB, and IMU measurements. Experiments show that F3DGS achieves reconstruction quality comparable to centralized training while enabling distributed optimization across agents. The dataset, development kit, and source code will be publicly released.
>
---
#### [new 038] MorphoGuard: A Morphology-Based Whole-Body Interactive Motion Controller
- **分类: eess.SY; cs.RO**

- **简介: 该论文提出MorphoGuard，解决机器人全肢体交互控制问题，通过 morphology-constrained 网络实现多接触点精准管理。**

- **链接: [https://arxiv.org/pdf/2604.01517](https://arxiv.org/pdf/2604.01517)**

> **作者:** Chenjin Wang; Zheng Yan; Yanmin Zhou; Runjie Shen; Bin He
>
> **摘要:** Whole-body control (WBC) has demonstrated significant advantages in complex interactive movements of high-dimensional robotic systems. However, when a robot is required to handle dynamic multi-contact combinations along a single kinematic chain-such as pushing open a door with its elbow while grasping an object-it faces major obstacles in terms of complex contact representation and joint configuration coupling. To address this, we propose a new control approach that explicitly manages arbitrary contact combinations, aiming to endow robots with whole-body interactive capabilities. We develop a morphology-constrained WBC network (MorphoGuard)-which is trained on a self-constructed dual-arm physical and simulation platform. A series of model recommendation experiments are designed to systematically investigate the impact of backbone architecture, fusion strategy, and model scale on network performance. To evaluate the control performance, we adopt a multi-object interaction task as the benchmark, requiring the model to simultaneously manipulate multiple target objects to specified positions. Experimental results show that the proposed method achieves a contact point management error of approximately 1 cm, demonstrating its effectiveness in whole-body interactive control.
>
---
#### [new 039] UAV-Track VLA: Embodied Aerial Tracking via Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机视觉跟踪任务，解决动态场景下多模态跟踪问题。提出UAV-Track VLA模型，提升跟踪性能与实时性。**

- **链接: [https://arxiv.org/pdf/2604.02241](https://arxiv.org/pdf/2604.02241)**

> **作者:** Qiyao Zhang; Shuhua Zheng; Jianli Sun; Chengxiang Li; Xianke Wu; Zihan Song; Zhiyong Cui; Yisheng Lv; Yonglin Tian
>
> **摘要:** Embodied visual tracking is crucial for Unmanned Aerial Vehicles (UAVs) executing complex real-world tasks. In dynamic urban scenarios with complex semantic requirements, Vision-Language-Action (VLA) models show great promise due to their cross-modal fusion and continuous action generation capabilities. To benchmark multimodal tracking in such environments, we construct a dedicated evaluation benchmark and a large-scale dataset encompassing over 890K frames, 176 tasks, and 85 diverse objects. Furthermore, to address temporal feature redundancy and the lack of spatial geometric priors in existing VLA models, we propose an improved VLA tracking model, UAV-Track VLA. Built upon the $\pi_{0.5}$ architecture, our model introduces a temporal compression net to efficiently capture inter-frame dynamics. Additionally, a parallel dual-branch decoder comprising a spatial-aware auxiliary grounding head and a flow matching action expert is designed to decouple cross-modal features and generate fine-grained continuous actions. Systematic experiments in the CARLA simulator validate the superior end-to-end performance of our method. Notably, in challenging long-distance pedestrian tracking tasks, UAV-Track VLA achieves a 61.76\% success rate and 269.65 average tracking frames, significantly outperforming existing baselines. Furthermore, it demonstrates robust zero-shot generalization in unseen environments and reduces single-step inference latency by 33.4\% (to 0.0571s) compared to the original $\pi_{0.5}$, enabling highly efficient, real-time UAV control. Data samples and demonstration videos are available at: this https URL\_VLA.
>
---
#### [new 040] Neural Robust Control on Lie Groups Using Contraction Methods (Extended Version)
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制理论任务，旨在解决Lie群上动态系统的鲁棒控制问题。通过联合训练RCCM和神经控制器，确保系统在扰动下保持稳定。**

- **链接: [https://arxiv.org/pdf/2604.01448](https://arxiv.org/pdf/2604.01448)**

> **作者:** Yi Lok Lo; Longhao Qian; Hugh H.T. Liu
>
> **备注:** An extended version of the conference paper submitted for publication in IEEE Conference of Decision and Control
>
> **摘要:** In this paper, we propose a learning framework for synthesizing a robust controller for dynamical systems evolving on a Lie group. A robust control contraction metric (RCCM) and a neural feedback controller are jointly trained to enforce contraction conditions on the Lie group manifold. Sufficient conditions are derived for the existence of such an RCCM and neural controller, ensuring that the geometric constraints imposed by the manifold structure are respected while establishing a disturbance-dependent tube that bounds the output trajectories. As a case study, a feedback controller for a quadrotor is designed using the proposed framework. Its performance is evaluated using numerical simulations and compared with a geometric controller.
>
---
#### [new 041] Model-Based Reinforcement Learning for Control under Time-Varying Dynamics
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决非平稳动态下的控制问题。通过模型基础的强化学习方法，应对系统动态变化，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2604.02260](https://arxiv.org/pdf/2604.02260)**

> **作者:** Klemens Iten; Bruce Lee; Chenhao Li; Lenart Treven; Andreas Krause; Bhavya Sukhija
>
> **备注:** 15 pages, 5 figues, 2 tables. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Learning-based control methods typically assume stationary system dynamics, an assumption often violated in real-world systems due to drift, wear, or changing operating conditions. We study reinforcement learning for control under time-varying dynamics. We consider a continual model-based reinforcement learning setting in which an agent repeatedly learns and controls a dynamical system whose transition dynamics evolve across episodes. We analyze the problem using Gaussian process dynamics models under frequentist variation-budget assumptions. Our analysis shows that persistent non-stationarity requires explicitly limiting the influence of outdated data to maintain calibrated uncertainty and meaningful dynamic regret guarantees. Motivated by these insights, we propose a practical optimistic model-based reinforcement learning algorithm with adaptive data buffer mechanisms and demonstrate improved performance on continuous control benchmarks with non-stationary dynamics.
>
---
#### [new 042] Safety, Security, and Cognitive Risks in World Models
- **分类: cs.CR; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于安全与风险研究任务，探讨世界模型在自主系统中的安全、安全和认知风险，提出威胁模型并验证攻击效果，旨在提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2604.01346](https://arxiv.org/pdf/2604.01346)**

> **作者:** Manoj Parmar
>
> **备注:** 26 pages, 1 figure (6 panels), 2 tables. Empirical proof-of-concept on GRU/RSSM/DreamerV3 architectures
>
> **摘要:** World models -- learned internal simulators of environment dynamics -- are rapidly becoming foundational to autonomous decision-making in robotics, autonomous vehicles, and agentic AI. Yet this predictive power introduces a distinctive set of safety, security, and cognitive risks. Adversaries can corrupt training data, poison latent representations, and exploit compounding rollout errors to cause catastrophic failures in safety-critical deployments. World model-equipped agents are more capable of goal misgeneralisation, deceptive alignment, and reward hacking precisely because they can simulate the consequences of their own actions. Authoritative world model predictions further foster automation bias and miscalibrated human trust that operators lack the tools to audit. This paper surveys the world model landscape; introduces formal definitions of trajectory persistence and representational risk; presents a five-profile attacker capability taxonomy; and develops a unified threat model extending MITRE ATLAS and the OWASP LLM Top 10 to the world model stack. We provide an empirical proof-of-concept on trajectory-persistent adversarial attacks (GRU-RSSM: A_1 = 2.26x amplification, -59.5% reduction under adversarial fine-tuning; stochastic RSSM proxy: A_1 = 0.65x; DreamerV3 checkpoint: non-zero action drift confirmed). We illustrate risks through four deployment scenarios and propose interdisciplinary mitigations spanning adversarial hardening, alignment engineering, NIST AI RMF and EU AI Act governance, and human-factors design. We argue that world models must be treated as safety-critical infrastructure requiring the same rigour as flight-control software or medical devices.
>
---
#### [new 043] UniDriveVLA: Unifying Understanding, Perception, and Action Planning for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决视觉-语言-动作模型中空间感知与语义推理的冲突。提出UniDriveVLA模型，通过专家解耦提升性能。**

- **链接: [https://arxiv.org/pdf/2604.02190](https://arxiv.org/pdf/2604.02190)**

> **作者:** Yongkang Li; Lijun Zhou; Sixu Yan; Bencheng Liao; Tianyi Yan; Kaixin Xiong; Long Chen; Hongwei Xie; Bing Wang; Guang Chen; Hangjun Ye; Wenyu Liu; Haiyang Sun; Xinggang Wang
>
> **备注:** code has been released at this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged in autonomous driving, with the promise of leveraging rich world knowledge to improve the cognitive capabilities of driving systems. However, adapting such models for driving tasks currently faces a critical dilemma between spatial perception and semantic reasoning. Consequently, existing VLA systems are forced into suboptimal compromises: directly adopting 2D Vision-Language Models yields limited spatial perception, whereas enhancing them with 3D spatial representations often impairs the native reasoning capacity of VLMs. We argue that this dilemma largely stems from the coupled optimization of spatial perception and semantic reasoning within shared model parameters. To overcome this, we propose UniDriveVLA, a Unified Driving Vision-Language-Action model based on Mixture-of-Transformers that addresses the perception-reasoning conflict via expert decoupling. Specifically, it comprises three experts for driving understanding, scene perception, and action planning, which are coordinated through masked joint attention. In addition, we combine a sparse perception paradigm with a three-stage progressive training strategy to improve spatial perception while maintaining semantic reasoning capability. Extensive experiments show that UniDriveVLA achieves state-of-the-art performance in open-loop evaluation on nuScenes and closed-loop evaluation on Bench2Drive. Moreover, it demonstrates strong performance across a broad range of perception, prediction, and understanding tasks, including 3D detection, online mapping, motion forecasting, and driving-oriented VQA, highlighting its broad applicability as a unified model for autonomous driving. Code and model have been released at this https URL
>
---
#### [new 044] AffordTissue: Dense Affordance Prediction for Tool-Action Specific Tissue Interaction
- **分类: cs.CV; cs.AI; cs.RO; eess.IV**

- **简介: 该论文提出AffordTissue，解决手术中工具与组织交互区域预测问题，通过多模态框架实现工具-动作特定的密集 affordance 预测，提升手术自动化安全性。**

- **链接: [https://arxiv.org/pdf/2604.01371](https://arxiv.org/pdf/2604.01371)**

> **作者:** Aiza Maksutova; Lalithkumar Seenivasan; Hao Ding; Jiru Xu; Chenhao Yu; Chenyan Jing; Yiqing Shen; Mathias Unberath
>
> **摘要:** Surgical action automation has progressed rapidly toward achieving surgeon-like dexterous control, driven primarily by advances in learning from demonstration and vision-language-action models. While these have demonstrated success in table-top experiments, translating them to clinical deployment remains challenging: current methods offer limited predictability on where instruments will interact on tissue surfaces and lack explicit conditioning inputs to enforce tool-action-specific safe interaction regions. Addressing this gap, we introduce AffordTissue, a multimodal framework for predicting tool-action specific tissue affordance regions as dense heatmaps during cholecystectomy. Our approach combines a temporal vision encoder capturing tool motion and tissue dynamics across multiple viewpoints, language conditioning enabling generalization across diverse instrument-action pairs, and a DiT-style decoder for dense affordance prediction. We establish the first tissue affordance benchmark by curating and annotating 15,638 video clips across 103 cholecystectomy procedures, covering six unique tool-action pairs involving four instruments (hook, grasper, scissors, clipper) and their associated tasks: dissection, grasping, clipping, and cutting. Experiments demonstrate substantial improvement over vision-language model baselines (20.6 px ASSD vs. 60.2 px for Molmo-VLM), showing that our task-specific architecture outperforms large-scale foundation models for dense surgical affordance prediction. By predicting tool-action specific tissue affordance regions, AffordTissue provides explicit spatial reasoning for safe surgical automation, potentially unlocking explicit policy guidance toward appropriate tissue regions and early safe stop when instruments deviate outside predicted safe zones.
>
---
#### [new 045] World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出WAV框架，用于提升世界模型的鲁棒性。针对世界模型在处理次优动作时的不足，通过分解状态预测为可塑性和可达性进行验证，提高样本效率和政策性能。**

- **链接: [https://arxiv.org/pdf/2604.01985](https://arxiv.org/pdf/2604.01985)**

> **作者:** Yuejiang Liu; Fan Feng; Lingjing Kong; Weifeng Lu; Jinzhou Tang; Kun Zhang; Kevin Murphy; Chelsea Finn; Yilun Du
>
> **备注:** Project Website: this https URL
>
> **摘要:** General-purpose world models promise scalable policy evaluation, optimization, and planning, yet achieving the required level of robustness remains challenging. Unlike policy learning, which primarily focuses on optimal actions, a world model must be reliable over a much broader range of suboptimal actions, which are often insufficiently covered by action-labeled interaction data. To address this challenge, we propose World Action Verifier (WAV), a framework that enables world models to identify their own prediction errors and self-improve. The key idea is to decompose action-conditioned state prediction into two factors -- state plausibility and action reachability -- and verify each separately. We show that these verification problems can be substantially easier than predicting future states due to two underlying asymmetries: the broader availability of action-free data and the lower dimensionality of action-relevant features. Leveraging these asymmetries, we augment a world model with (i) a diverse subgoal generator obtained from video corpora and (ii) a sparse inverse model that infers actions from a subset of state features. By enforcing cycle consistency among generated subgoals, inferred actions, and forward rollouts, WAV provides an effective verification mechanism in under-explored regimes, where existing methods typically fail. Across nine tasks spanning MiniGrid, RoboMimic, and ManiSkill, our method achieves 2x higher sample efficiency while improving downstream policy performance by 18%.
>
---
#### [new 046] CompassAD: Intent-Driven 3D Affordance Grounding in Functionally Competing Objects
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于3D affordance grounding任务，解决在功能相似物体中根据意图精准识别目标对象的问题。提出CompassAD基准和CompassNet框架，提升多物体场景下的意图驱动定位效果。**

- **链接: [https://arxiv.org/pdf/2604.02060](https://arxiv.org/pdf/2604.02060)**

> **作者:** Jingliang Li; Jindou Jia; Tuo An; Chuhao Zhou; Xiangyu Chen; Shilin Shan; Boyu Ma; Bofan Lyu; Gen Li; Jianfei Yang
>
> **备注:** Code available at: this http URL
>
> **摘要:** When told to "cut the apple," a robot must choose the knife over nearby scissors, despite both objects affording the same cutting function. In real-world scenes, multiple objects may share identical affordances, yet only one is appropriate under the given task context. We call such cases confusing pairs. However, existing 3D affordance methods largely sidestep this challenge by evaluating isolated single objects, often with explicit category names provided in the query. We formalize Multi-Object Affordance Grounding under Intent-Driven Instructions, a new 3D affordance setting that requires predicting a per-point affordance mask on the correct object within a cluttered multi-object point cloud, conditioned on implicit natural language intent. To study this problem, we construct CompassAD, the first benchmark centered on implicit intent in confusable multi-object scenes. It comprises 30 confusing object pairs spanning 16 affordance types, 6,422 scenes, and 88K+ query-answer pairs. Furthermore, we propose CompassNet, a framework that incorporates two dedicated modules tailored to this task. Instance-bounded Cross Injection (ICI) constrains language-geometry alignment within object boundaries to prevent cross-object semantic leakage. Bi-level Contrastive Refinement (BCR) enforces discrimination at both geometric-group and point levels, sharpening distinctions between target and confusable surfaces. Extensive experiments demonstrate state-of-the-art results on both seen and unseen queries, and deployment on a robotic manipulator confirms effective transfer to real-world grasping in confusing multi-object scenes.
>
---
#### [new 047] Learning Spatial Structure from Pre-Beamforming Per-Antenna Range-Doppler Radar Data via Visibility-Aware Cross-Modal Supervision
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文研究如何从雷达原始数据中直接学习空间结构，解决传统需先进行波束成形的问题。通过端到端方法，无需角度域构建即可恢复几何信息。**

- **链接: [https://arxiv.org/pdf/2604.01921](https://arxiv.org/pdf/2604.01921)**

> **作者:** George Sebastian; Philipp Berthold; Bianca Forkel; Leon Pohl; Mirko Maehlisch
>
> **摘要:** Automotive radar perception pipelines commonly construct angle-domain representations via beamforming before applying learning-based models. This work instead investigates a representational question: can meaningful spatial structure be learned directly from pre-beamforming per-antenna range-Doppler (RD) measurements? Experiments are conducted on a 6-TX x 8-RX (48 virtual antennas) commodity automotive radar employing an A/B chirp-sequence frequency-modulated continuous-wave (CS-FMCW) transmit scheme, in which the effective transmit aperture varies between chirps (single-TX vs. multi-TX), enabling controlled analysis of chirp-dependent transmit configurations. We operate on pre-beamforming per-antenna RD tensors using a dual-chirp shared-weight encoder trained in an end-to-end, fully data-driven manner, and evaluate spatial recoverability using bird's-eye-view (BEV) occupancy as a geometric probe rather than a performance-driven objective. Supervision is visibility-aware and cross-modal, derived from LiDAR with explicit modeling of the radar field-of-view and occlusion-aware LiDAR observability via ray-based visibility. Through chirp ablations (A-only, B-only, A+B), range-band analysis, and physics-aligned baselines, we assess how transmit configurations affect geometric recoverability. The results indicate that spatial structure can be learned directly from pre-beamforming per-antenna RD tensors without explicit angle-domain construction or hand-crafted signal-processing stages.
>
---
## 更新

#### [replaced 001] How Leg Stiffness Affects Energy Economy in Hopping
- **分类: cs.RO**

- **简介: 该论文属于机器人学任务，研究腿 stiffness 对跳跃能量效率的影响，旨在解决不同速度下机器人腿部设计的优化问题，通过参数分析发现可变 stiffness 提升效率。**

- **链接: [https://arxiv.org/pdf/2501.03971](https://arxiv.org/pdf/2501.03971)**

> **作者:** Iskandar Khemakhem; Dominik Tschemernjak; Maximilian Raff; C. David Remy
>
> **摘要:** In the fields of robotics and biomechanics, the integration of elastic elements such as springs and tendons in legged systems has long been recognized for enabling energy-efficient locomotion. Yet, a significant challenge persists: designing a robotic leg that perform consistently across diverse operating conditions, especially varying average forward speeds. It remains unclear whether, for such a range of operating conditions, the stiffness of the elastic elements needs to be varied or if a similar performance can be obtained by changing the motion and actuation while keeping the stiffness fixed. This work explores the influence of the leg stiffness on the energy efficiency of a monopedal robot through an extensive parametric study of its periodic hopping motion. To this end, we formulate an optimal control problem parameterized by average forward speed and leg stiffness, solving it numerically using direct collocation. Our findings indicate that, compared to the use of a fixed stiffness, employing variable stiffness in legged systems improves energy efficiency by 20 % maximally and by 6.8 % on average across a range of speeds.
>
---
#### [replaced 002] V-OCBF: Learning Safety Filters from Offline Data via Value-Guided Offline Control Barrier Functions
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于安全控制任务，旨在解决自主系统中严格状态约束问题。通过离线数据学习神经控制屏障函数，实现安全控制，无需在线交互或先验模型。**

- **链接: [https://arxiv.org/pdf/2512.10822](https://arxiv.org/pdf/2512.10822)**

> **作者:** Mumuksh Tayal; Manan Tayal; Aditya Singh; Shishir Kolathaya; Ravi Prakash
>
> **备注:** 28 pages, 9 figures, 11 tables. Paper accepted at TMLR
>
> **摘要:** Ensuring safety in autonomous systems requires controllers that aim to satisfy state-wise constraints without relying on online this http URL existing Safe Offline RL methods typically enforce soft expected-cost constraints, they struggle to ensure strict state-wise safety. Conversely, Control Barrier Functions (CBFs) offer a principled mechanism to enforce forward invariance, but often rely on expert-designed barrier functions or knowledge of the system dynamics. We introduce Value-Guided Offline Control Barrier Functions (V-OCBF), a framework that learns a neural CBF entirely from offline demonstrations. Unlike prior approaches, V-OCBF does not assume access to the dynamics model; instead, it derives a recursive finite-difference barrier update, enabling model-free learning of a barrier that propagates safety information over time. Moreover, V-OCBF incorporates an expectile-based objective that avoids querying the barrier on out-of-distribution actions and restricts updates to the dataset-supported action set. The learned barrier is then used with a Quadratic Program (QP) formulation to synthesize real-time safe control. Across multiple case studies, V-OCBF yields substantially fewer safety violations than baseline methods while maintaining strong task performance, highlighting its scalability for offline synthesis of safety-critical controllers without online interaction or hand-engineered barriers.
>
---
#### [replaced 003] Pixel Motion Diffusion is What We Need for Robot Control
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DAWN框架，用于机器人控制任务，解决高、低层控制间的映射问题。通过扩散模型和像素运动表示，实现端到端学习与可解释的中间抽象。**

- **链接: [https://arxiv.org/pdf/2509.22652](https://arxiv.org/pdf/2509.22652)**

> **作者:** E-Ro Nguyen; Yichi Zhang; Kanchana Ranasinghe; Xiang Li; Michael S. Ryoo
>
> **备注:** Accepted to CVPR 2026. Project page: this https URL
>
> **摘要:** We present DAWN (Diffusion is All We Need for robot control), a unified diffusion-based framework for language-conditioned robotic manipulation that bridges high-level motion intent and low-level robot action via structured pixel motion representation. In DAWN, both the high-level and low-level controllers are modeled as diffusion processes, yielding a fully trainable, end-to-end system with interpretable intermediate motion abstractions. DAWN achieves state-of-the-art results on the challenging CALVIN benchmark, demonstrating strong multi-task performance, and further validates its effectiveness on MetaWorld. Despite the substantial domain gap between simulation and reality and limited real-world data, we demonstrate reliable real-world transfer with only minimal finetuning, illustrating the practical viability of diffusion-based motion abstractions for robotic control. Our results show the effectiveness of combining diffusion modeling with motion-centric representations as a strong baseline for scalable and robust robot learning. Project page: this https URL
>
---
#### [replaced 004] Emergent Dexterity via Diverse Resets and Large-Scale Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决复杂抓取与操控问题。通过引入OmniReset框架，实现无需人工干预的高效强化学习，提升策略泛化与真实世界适应能力。**

- **链接: [https://arxiv.org/pdf/2603.15789](https://arxiv.org/pdf/2603.15789)**

> **作者:** Patrick Yin; Tyler Westenbroek; Zhengyu Zhang; Joshua Tran; Ignacio Dagnino; Eeshani Shilamkar; Numfor Mbiziwo-Tiapo; Simran Bagaria; Xinlei Liu; Galen Mullins; Andrey Kolobov; Abhishek Gupta
>
> **摘要:** Reinforcement learning in massively parallel physics simulations has driven major progress in sim-to-real robot learning. However, current approaches remain brittle and task-specific, relying on extensive per-task engineering to design rewards, curricula, and demonstrations. Even with this engineering, they often fail on long-horizon, contact-rich manipulation tasks and do not meaningfully scale with compute, as performance quickly saturates when training revisits the same narrow regions of state space. We introduce OmniReset, a simple and scalable framework that enables on-policy reinforcement learning to robustly solve a broad class of dexterous manipulation tasks using a single reward function, fixed algorithm hyperparameters, no curricula, and no human demonstrations. Our key insight is that long-horizon exploration can be dramatically simplified by using simulator resets to systematically expose the RL algorithm to the diverse set of robot-object interactions which underlie dexterous manipulation. OmniReset programmatically generates such resets with minimal human input, converting additional compute directly into broader behavioral coverage and continued performance gains. We show that OmniReset gracefully scales to long-horizon dexterous manipulation tasks beyond the capabilities of existing approaches and is able to learn robust policies over significantly wider ranges of initial conditions than baselines. Finally, we distill OmniReset into visuomotor policies which display robust retrying behavior and substantially higher success rates than baselines when transferred to the real world zero-shot. Project webpage: this https URL
>
---
#### [replaced 005] Constraint-Aware Reinforcement Learning via Adaptive Action Scaling
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于安全强化学习任务，旨在解决训练中因探索导致的不安全行为问题。通过自适应动作缩放机制，在保持任务性能的同时显著减少约束违反。**

- **链接: [https://arxiv.org/pdf/2510.11491](https://arxiv.org/pdf/2510.11491)**

> **作者:** Murad Dawood; Usama Ahmed Siddiquie; Shahram Khorshidi; Maren Bennewitz
>
> **备注:** Accepted in 8th Annual Learning for Dynamics & Control Conference (L4DC)
>
> **摘要:** Safe reinforcement learning (RL) seeks to mitigate unsafe behaviors that arise from exploration during training by reducing constraint violations while maintaining task performance. Existing approaches typically rely on a single policy to jointly optimize reward and safety, which can cause instability due to conflicting objectives, or they use external safety filters that override actions and require prior system knowledge. In this paper, we propose a modular cost-aware regulator that scales the agent's actions based on predicted constraint violations, preserving exploration through smooth action modulation rather than overriding the policy. The regulator is trained to minimize constraint violations while avoiding degenerate suppression of actions. Our approach integrates seamlessly with off-policy RL methods such as SAC and TD3, and achieves state-of-the-art return-to-cost ratios on Safety Gym locomotion tasks with sparse costs, reducing constraint violations by up to 126 times while increasing returns by over an order of magnitude compared to prior methods.
>
---
#### [replaced 006] COMPAct: Computational Optimization and Automated Modular design of Planetary Actuators
- **分类: cs.RO**

- **简介: 该论文属于机器人执行器设计任务，旨在优化齿轮箱参数并自动化CAD生成。通过COMPAct框架，实现高效、轻量的行星齿轮机构设计与3D打印。**

- **链接: [https://arxiv.org/pdf/2510.07197](https://arxiv.org/pdf/2510.07197)**

> **作者:** Aman Singh; Deepak Kapa; Suryank Joshi; Shishir Kolathaya
>
> **备注:** 8 pages, 9 Figures, 2 tables; first two authors contributed equally; published in 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** The optimal design of robotic actuators is a critical area of research, yet limited attention has been given to optimizing gearbox parameters and automating actuator CAD. This paper introduces COMPAct: Computational Optimization and Automated Modular Design of Planetary Actuators, a framework that systematically identifies optimal gearbox parameters for a given motor across four gearbox types, single-stage planetary gearbox (SSPG), compound planetary gearbox (CPG), Wolfrom planetary gearbox (WPG), and double-stage planetary gearbox (DSPG). The framework minimizes mass and actuator width while maximizing efficiency, and further automates actuator CAD generation to enable direct 3D printing without manual redesign. Using this framework, optimal gearbox designs are explored across a wide range of gear ratios, providing insights into the suitability of different gearbox types while automatically generating CAD models for all four gearbox types with varying gear ratios and motors. Two actuator types are fabricated and experimentally evaluated through power efficiency, no-load backlash, and transmission stiffness tests. Experimental results indicate that the SSPG actuator achieves a mechanical efficiency of 60-80%, a no-load backlash of 0.59 deg, and a transmission stiffness of 242.7 Nm/rad, while the CPG actuator demonstrates 60% efficiency, 2.6 deg backlash, and a stiffness of 201.6 Nm/rad. CODE: this https URL VIDEO: this https URL
>
---
#### [replaced 007] Allometric Scaling Laws for Bipedal Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，研究 bipedal 机器人尺寸缩放问题。通过分析数据和仿真，揭示机器人质量、扭矩等参数与腿长的非相似比例关系，提供新的缩放规律。**

- **链接: [https://arxiv.org/pdf/2603.22560](https://arxiv.org/pdf/2603.22560)**

> **作者:** Naomi Oke; Aja Carter; Ben Gu; Steven Man; Cordelia Pride; Sarah Bergbreiter; Aaron M. Johnson
>
> **摘要:** Scaling the design of robots up or down remains a fundamental challenge. While biological systems follow well-established isometric and allometric scaling laws relating mass, stride frequency, velocity, and torque, it is unclear how these relationships translate to robotic systems. In this paper, we generate similar allometric scaling laws for bipedal robots across three orders of magnitude in leg length. First, we conduct a review of legged robots from the literature and extract empirical relationships between leg length (L), body length, mass, and speed. These data show that robot mass scales more closely to L^2, in contrast to the L^3 scaling predicted by isometric scaling. We then perform controlled simulation studies in Drake using three variants of real quasi-passive, hip-actuated walkers with different foot geometries and control strategies. We evaluate the performance of each design scaled with leg length, L. Across all robots, walking velocity follows the expected L^(1/2) trend from dynamic similarity. Minimum required torque scales more closely with m*L than the isometric model of m*L^2. Foot geometry scaled proportionally with L^1. These results provide new insight into how robot designs allometrically scale to different sizes, and how that scaling is different from isometric or biological scaling laws.
>
---
#### [replaced 008] GPA-VGGT:Adapting VGGT to Large Scale Localization by Self-Supervised Learning with Geometry and Physics Aware Loss
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决VGGT模型在无标签场景下的适应问题。通过自监督学习和几何物理损失，提升其大尺度环境中的定位能力。**

- **链接: [https://arxiv.org/pdf/2601.16885](https://arxiv.org/pdf/2601.16885)**

> **作者:** Yangfan Xu; Lilian Zhang; Xiaofeng He; Pengdong Wu; Wenqi Wu; Jun Mao
>
> **摘要:** Transformer-based general visual geometry frameworks have shown promising performance in camera pose estimation and 3D scene understanding. Recent advancements in Visual Geometry Grounded Transformer (VGGT) models have shown great promise in camera pose estimation and 3D reconstruction. However, these models typically rely on ground truth labels for training, posing challenges when adapting to unlabeled and unseen scenes. In this paper, we propose a self-supervised framework to train VGGT with unlabeled data, thereby enhancing its localization capability in large-scale environments. To achieve this, we extend conventional pair-wise relations to sequence-wise geometric constraints for self-supervised learning. Specifically, in each sequence, we sample multiple source frames and geometrically project them onto different target frames, which improves temporal feature consistency. We formulate physical photometric consistency and geometric constraints as a joint optimization loss to circumvent the requirement for hard labels. By training the model with this proposed method, not only the local and global cross-view attention layers but also the camera and depth heads can effectively capture the underlying multi-view geometry. Experiments demonstrate that the model converges within hundreds of iterations and achieves significant improvements in large-scale localization. Our code will be released at this https URL.
>
---
#### [replaced 009] IA-TIGRIS: An Incremental and Adaptive Sampling-Based Planner for Online Informative Path Planning
- **分类: cs.RO**

- **简介: 该论文属于信息路径规划任务，旨在实时优化机器人采集信息的路径。提出IA-TIGRIS算法，通过增量和自适应采样提升信息获取效率，并在两种无人机上验证效果。**

- **链接: [https://arxiv.org/pdf/2502.15961](https://arxiv.org/pdf/2502.15961)**

> **作者:** Brady Moon; Nayana Suvarna; Andrew Jong; Satrajit Chatterjee; Junbin Yuan; Muqing Cao; Sebastian Scherer
>
> **备注:** Published in IEEE Transactions on Robotics, 19 pages, 19 figures
>
> **摘要:** Planning paths that maximize information gain for robotic platforms has wide-ranging applications and significant potential impact. To effectively adapt to real-time data collection, informative path planning must be computed online and be responsive to new observations. In this work, we present IA-TIGRIS (Incremental and Adaptive Tree-based Information Gathering Using Informed Sampling), which is an incremental and adaptive sampling-based informative path planner designed for real-time onboard execution. Our approach leverages past planning efforts through incremental refinement while continuously adapting to updated belief maps. We additionally present detailed implementation and optimization insights to facilitate real-world deployment, along with an array of reward functions tailored to specific missions and behaviors. Extensive simulation results demonstrate IA-TIGRIS generates higher-quality paths compared to baseline methods. We validate our planner on two distinct hardware platforms: a hexarotor unmanned aerial vehicle (UAV) and a fixed-wing UAV, each having different motion models and configuration spaces. Our results show up to a 38% improvement in information gain compared to baseline methods, highlighting the planner's potential for deployment in real-world applications. Project website: this https URL
>
---
#### [replaced 010] MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出MaskAdapt，解决物理角色的灵活运动适应问题。通过两阶段学习，实现部分身体动作的精准调整，提升运动鲁棒性与多样性。**

- **链接: [https://arxiv.org/pdf/2603.29272](https://arxiv.org/pdf/2603.29272)**

> **作者:** Soomin Park; Eunseong Lee; Kwang Bin Lee; Sung-Hee Lee
>
> **备注:** CVPR 2026
>
> **摘要:** We present MaskAdapt, a framework for flexible motion adaptation in physics-based humanoid control. The framework follows a two-stage residual learning paradigm. In the first stage, we train a mask-invariant base policy using stochastic body-part masking and a regularization term that enforces consistent action distributions across masking conditions. This yields a robust motion prior that remains stable under missing observations, anticipating later adaptation in those regions. In the second stage, a residual policy is trained atop the frozen base controller to modify only the targeted body parts while preserving the original behaviors elsewhere. We demonstrate the versatility of this design through two applications: (i) motion composition, where varying masks enable multi-part adaptation within a single sequence, and (ii) text-driven partial goal tracking, where designated body parts follow kinematic targets provided by a pre-trained text-conditioned autoregressive motion generator. Through experiments, MaskAdapt demonstrates strong robustness and adaptability, producing diverse behaviors under masked observations and delivering superior targeted motion adaptation compared to prior work.
>
---
#### [replaced 011] D-SPEAR: Dual-Stream Prioritized Experience Adaptive Replay for Stable Reinforcement Learning in Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，解决强化学习中的训练不稳定问题。提出D-SPEAR框架，分离策略与评论家的经验回放，提升稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2603.27346](https://arxiv.org/pdf/2603.27346)**

> **作者:** Yu Zhang; Karl Mason
>
> **备注:** Accepted at IEEE 11th International Conference on Control and Robotics Engineering (ICCRE 2026)
>
> **摘要:** Robotic manipulation remains challenging for reinforcement learning due to contact-rich dynamics, long horizons, and training instability. Although off-policy actor-critic algorithms such as SAC and TD3 perform well in simulation, they often suffer from policy oscillations and performance collapse in realistic settings, partly due to experience replay strategies that ignore the differing data requirements of the actor and the critic. We propose D-SPEAR: Dual-Stream Prioritized Experience Adaptive Replay, a replay framework that decouples actor and critic sampling while maintaining a shared replay buffer. The critic leverages prioritized replay for efficient value learning, whereas the actor is updated using low-error transitions to stabilize policy optimization. An adaptive anchor mechanism balances uniform and prioritized sampling based on the coefficient of variation of TD errors, and a Huber-based critic objective further improves robustness under heterogeneous reward scales. We evaluate D-SPEAR on challenging robotic manipulation tasks from the robosuite benchmark, including Block-Lifting and Door-Opening. Results demonstrate that D-SPEAR consistently outperforms strong off-policy baselines, including SAC, TD3, and DDPG, in both final performance and training stability, with ablation studies confirming the complementary roles of the actorside and critic-side replay streams.
>
---
#### [replaced 012] TaCarla: A comprehensive benchmarking dataset for end-to-end autonomous driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出TaCarla数据集，用于端到端自动驾驶研究。解决现有数据集不完整、缺乏多样性及闭环评估的问题，涵盖感知与规划任务，并支持多种模型训练。**

- **链接: [https://arxiv.org/pdf/2602.23499](https://arxiv.org/pdf/2602.23499)**

> **作者:** Tugrul Gorgulu; Atakan Dag; M. Esat Kalfaoglu; Halil Ibrahim Kuru; Baris Can Cam; Halil Ibrahim Ozturk; Ozsel Kilinc
>
> **摘要:** Collecting a high-quality dataset is a critical task that demands meticulous attention to detail, as overlooking certain aspects can render the entire dataset unusable. Autonomous driving challenges remain a prominent area of research, requiring further exploration to enhance the perception and planning performance of vehicles. However, existing datasets are often incomplete. For instance, datasets that include perception information generally lack planning data, while planning datasets typically consist of extensive driving sequences where the ego vehicle predominantly drives forward, offering limited behavioral diversity. In addition, many real datasets struggle to evaluate their models, especially for planning tasks, since they lack a proper closed-loop evaluation setup. The CARLA Leaderboard 2.0 challenge, which provides a diverse set of scenarios to address the long-tail problem in autonomous driving, has emerged as a valuable alternative platform for developing perception and planning models in both open-loop and closed-loop evaluation setups. Nevertheless, existing datasets collected on this platform present certain limitations. Some datasets appear to be tailored primarily for limited sensor configuration, with particular sensor configurations. To support end-to-end autonomous driving research, we have collected a new dataset comprising over 2.85 million frames using the CARLA simulation environment for the diverse Leaderboard 2.0 challenge scenarios. Our dataset is designed not only for planning tasks but also supports dynamic object detection, lane divider detection, centerline detection, traffic light recognition, prediction tasks and visual language action models . Furthermore, we demonstrate its versatility by training various models using our dataset. Moreover, we also provide numerical rarity scores to understand how rarely the current state occurs in the dataset.
>
---
#### [replaced 013] ThinkGrasp: A Vision-Language System for Strategic Part Grasping in Clutter
- **分类: cs.RO**

- **简介: 论文提出ThinkGrasp系统，解决杂乱环境中物体抓取问题。通过视觉语言结合GPT-4o进行策略性抓取，有效识别并移除障碍物，提升抓取成功率。**

- **链接: [https://arxiv.org/pdf/2407.11298](https://arxiv.org/pdf/2407.11298)**

> **作者:** Yaoyao Qian; Xupeng Zhu; Ondrej Biza; Shuo Jiang; Linfeng Zhao; Haojie Huang; Yu Qi; Robert Platt
>
> **备注:** Accepted at CoRL 2024. Project Website:(this https URL)
>
> **摘要:** Robotic grasping in cluttered environments remains a significant challenge due to occlusions and complex object arrangements. We have developed ThinkGrasp, a plug-and-play vision-language grasping system that makes use of GPT-4o's advanced contextual reasoning for heavy clutter environment grasping strategies. ThinkGrasp can effectively identify and generate grasp poses for target objects, even when they are heavily obstructed or nearly invisible, by using goal-oriented language to guide the removal of obstructing objects. This approach progressively uncovers the target object and ultimately grasps it with a few steps and a high success rate. In both simulated and real experiments, ThinkGrasp achieved a high success rate and significantly outperformed state-of-the-art methods in heavily cluttered environments or with diverse unseen objects, demonstrating strong generalization capabilities.
>
---
#### [replaced 014] Olaf: Bringing an Animated Character to Life in the Physical World
- **分类: cs.RO; cs.LG**

- **简介: 该论文将动画角色Olaf转化为物理机器人，解决机械设计与运动控制问题。通过强化学习实现逼真动作，优化结构与散热，提升角色可信度。**

- **链接: [https://arxiv.org/pdf/2512.16705](https://arxiv.org/pdf/2512.16705)**

> **作者:** David Müller; Espen Knoop; Dario Mylonopoulos; Agon Serifi; Michael A. Hopkins; Ruben Grandia; Moritz Bächer
>
> **摘要:** Animated characters often move in non-physical ways and have proportions that are far from a typical walking robot. This provides an ideal platform for innovation in both mechanical design and stylized motion control. In this paper, we bring Olaf to life in the physical world, relying on reinforcement learning guided by animation references for control. To create the illusion of Olaf's feet moving along his body, we hide two asymmetric legs under a soft foam skirt. To fit actuators inside the character, we use spherical and planar linkages in the arms, mouth, and eyes. Because the walk cycle results in harsh contact sounds, we introduce additional rewards that noticeably reduce impact noise. The large head, driven by small actuators in the character's slim neck, creates a risk of overheating, amplified by the costume. To keep actuators from overheating, we feed temperature values as additional inputs to policies, introducing new rewards to keep them within bounds. We validate the efficacy of our modeling in simulation and on hardware, demonstrating an unmatched level of believability for a costumed robotic character.
>
---
#### [replaced 015] Vi-TacMan: Articulated Object Manipulation via Vision and Touch
- **分类: cs.RO**

- **简介: 该论文提出Vi-TacMan，解决机器人操作铰接物体的任务。通过结合视觉与触觉反馈，实现精准控制，无需显式运动学模型。**

- **链接: [https://arxiv.org/pdf/2510.06339](https://arxiv.org/pdf/2510.06339)**

> **作者:** Leiyao Cui; Zihang Zhao; Sirui Xie; Wenhuan Zhang; Zhi Han; Yixin Zhu
>
> **备注:** ICRA 2026
>
> **摘要:** Autonomous manipulation of articulated objects remains a fundamental challenge for robots in human environments. Vision-based methods can infer hidden kinematics but can yield imprecise estimates on unfamiliar objects. Tactile approaches achieve robust control through contact feedback but require accurate initialization. This suggests a natural synergy: vision for global guidance, touch for local precision. Yet no framework systematically exploits this complementarity for generalized articulated manipulation. Here we present Vi-TacMan, which uses vision to propose grasps and coarse directions that seed a tactile controller for precise execution. By incorporating surface normals as geometric priors and modeling directions via von Mises-Fisher distributions, our approach achieves significant gains over baselines (all p<0.0001). Critically, manipulation succeeds without explicit kinematic models -- the tactile controller refines coarse visual estimates through real-time contact regulation. Tests on more than 50,000 simulated and diverse real-world objects confirm robust cross-category generalization. This work establishes that coarse visual cues suffice for reliable manipulation when coupled with tactile feedback, offering a scalable paradigm for autonomous systems in unstructured environments.
>
---
#### [replaced 016] What Capable Agents Must Know: Selection Theorems for Robust Decision-Making under Uncertainty
- **分类: cs.LG; cs.AI; cs.RO; q-bio.NC; stat.ML**

- **简介: 该论文研究智能体在不确定性下实现稳健决策的必要结构，解决任务性能与内部表示之间的关系问题。通过选择定理证明强性能需世界模型、信念记忆等结构。**

- **链接: [https://arxiv.org/pdf/2603.02491](https://arxiv.org/pdf/2603.02491)**

> **作者:** Aran Nayebi
>
> **备注:** 23 pages; added PSR recovery (Theorems 3 & 4), and updated related work
>
> **摘要:** As artificial agents become increasingly capable, what internal structure is *necessary* for an agent to act competently under uncertainty? Classical results show that optimal control can be *implemented* using belief states or world models, but not that such representations are required. We prove quantitative "selection theorems" showing that strong task performance (low *average-case regret*) forces world models, belief-like memory and -- under task mixtures -- persistent variables resembling core primitives associated with emotion, along with informational modularity under block-structured tasks. Our results cover stochastic policies, partial observability, and evaluation under task distributions, without assuming optimality, determinism, or access to an explicit model. Technically, we reduce predictive modeling to binary "betting" decisions and show that regret bounds limit probability mass on suboptimal bets, enforcing the predictive distinctions needed to separate high-margin outcomes. In fully observed settings, this yields approximate recovery of the interventional transition kernel; under partial observability, it implies necessity of predictive state and belief-like memory, addressing an open question in prior world-model recovery work.
>
---
#### [replaced 017] Multi-Staged Framework for Safety Analysis of Offloaded Services in Distributed Intelligent Transportation Systems
- **分类: cs.RO**

- **简介: 该论文属于安全分析任务，旨在解决分布式智能交通系统中服务卸载的安全问题。通过构建多阶段框架，确保远程服务的可靠性与数据安全。**

- **链接: [https://arxiv.org/pdf/2602.08821](https://arxiv.org/pdf/2602.08821)**

> **作者:** Robin Dehler; Oliver Schumann; Jona Ruof; Michael Buchholz
>
> **备注:** 2025 IEEE International Conference on Intelligent Transportation Systems (ITSC)
>
> **摘要:** The integration of service-oriented architectures (SOA) with function offloading for distributed, intelligent transportation systems (ITS) offers the opportunity for connected autonomous vehicles (CAVs) to extend their locally available services. One major goal of offloading a subset of functions in the processing chain of a CAV to remote devices is to reduce the overall computational complexity on the CAV. The extension of using remote services, however, requires careful safety analysis, since the remotely created data are corrupted more easily, e.g., through an attacker on the remote device or by intercepting the wireless transmission. To tackle this problem, we first analyze the concept of SOA for distributed environments. From this, we derive a safety framework that validates the reliability of remote services and the data received locally. Since it is possible for the autonomous driving task to offload multiple different services, we propose a specific multi-staged framework for safety analysis dependent on the service composition of local and remote services. For efficiency reasons, we directly include the multi-staged framework for safety analysis in our service-oriented function offloading framework (SOFOF) that we have proposed in earlier work. The evaluation compares the performance of the extended framework considering computational complexity, with energy savings being a major motivation for function offloading, and its capability to detect data from corrupted remote services.
>
---
#### [replaced 018] Physical Human-Robot Interaction: A Critical Review of Safety Constraints
- **分类: eess.SY; cs.RO**

- **简介: 论文探讨物理人机交互中的安全约束，聚焦ISO/TS 15066标准，分析其假设与实际应用，强调能量在安全评估中的作用。任务是提升工业机器人系统安全性。**

- **链接: [https://arxiv.org/pdf/2601.19462](https://arxiv.org/pdf/2601.19462)**

> **作者:** Riccardo Zanella; Federico Califano; Stefano Stramigioli
>
> **摘要:** This paper aims to provide a clear and rigorous understanding of commonly recognized safety constraints in physical human-robot interaction, particularly regarding ISO/TS 15066. We investigate the derivation of these constraints, critically examine the underlying assumptions, and evaluate their practical implications for system-level safety and performance in industrially relevant scenarios. Key design parameters within safety-critical control architectures are identified, and numerical examples are provided to quantify performance degradation arising from typical approximations and design decisions in manufacturing environments. Within this analysis, the fundamental role of energy in safety assessment is emphasized, providing focused insights into energy-based safety methodologies for collaborative industrial robot systems.
>
---
#### [replaced 019] OMCL: Open-vocabulary Monte Carlo Localization
- **分类: cs.RO**

- **简介: 该论文提出OMCL，用于机器人定位任务，解决多传感器环境下地图与观测不匹配的问题。通过视觉-语言特征实现跨模态关联和自然语言初始化，提升定位鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.15557](https://arxiv.org/pdf/2512.15557)**

> **作者:** Evgenii Kruzhkov; Raphael Memmesheimer; Sven Behnke
>
> **备注:** Accepted to IEEE RA-L
>
> **摘要:** Robust robot localization is an important prerequisite for navigation, but it becomes challenging when the map and robot measurements are obtained from different sensors. Prior methods are often tailored to specific environments, relying on closed-set semantics or fine-tuned features. In this work, we extend Monte Carlo Localization with vision-language features, allowing OMCL to robustly compute the likelihood of visual observations given a camera pose and a 3D map created from posed RGB-D images or aligned point clouds. These open-vocabulary features enable us to associate observations and map elements from different modalities, and to natively initialize global localization through natural language descriptions of nearby objects. We evaluate our approach using Matterport3D and Replica for indoor scenes and demonstrate generalization on SemanticKITTI for outdoor scenes.
>
---
#### [replaced 020] When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人决策任务，解决 embodied 代理在推理与行动间的资源分配问题。提出 RARRL 框架，通过强化学习实现自适应推理控制，提升任务成功率与系统效率。**

- **链接: [https://arxiv.org/pdf/2603.16673](https://arxiv.org/pdf/2603.16673)**

> **作者:** Jun Liu; Pu Zhao; Zhenglun Kong; Xuan Shen; Peiyan Dong; Fan Yang; Lin Cui; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Xue Lin; Gaowen Liu; Yanzhi Wang; Dong Huang
>
> **摘要:** Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents.
>
---
#### [replaced 021] DualReg: Dual-Space Filtering and Reinforcement for Rigid Registration
- **分类: cs.RO**

- **简介: 该论文属于刚性配准任务，解决噪声、部分重叠数据和实时处理难题。提出双空间方法，结合特征匹配与局部几何匹配优势，提升配准效率与精度。**

- **链接: [https://arxiv.org/pdf/2508.17034](https://arxiv.org/pdf/2508.17034)**

> **作者:** Jiayi Li; Yuxin Yao; Qiuhang Lu; Juyong Zhang
>
> **备注:** Accepted to CVPR 2026, Project page: this https URL
>
> **摘要:** Noisy, partially overlapping data and the need for real-time processing pose major challenges for rigid registration. Considering that feature-based matching can handle large transformation differences but suffers from limited accuracy, while local geometry-based matching can achieve fine-grained local alignment but relies heavily on a good initial transformation, we propose a novel dual-space paradigm to fully leverage the strengths of both approaches. First, we introduce an efficient filtering mechanism consisting of a computationally lightweight one-point RANSAC algorithm and a subsequent refinement module to eliminate unreliable feature-based correspondences. Subsequently, we treat the filtered correspondences as anchor points, extract geometric proxies, and formulate an effective objective function with a tailored solver to estimate the transformation. Experiments verify our method's effectiveness, as demonstrated by a 32x CPU-time speedup over MAC on KITTI with comparable accuracy. Project page: this https URL.
>
---
