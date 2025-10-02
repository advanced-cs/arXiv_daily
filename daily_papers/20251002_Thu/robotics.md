# 机器人 cs.RO

- **最新发布 42 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] DiSA-IQL: Offline Reinforcement Learning for Robust Soft Robot Control under Distribution Shifts
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于软体机器人控制任务，解决分布偏移问题。提出DiSA-IQL方法，通过惩罚不可靠状态动作对提升离线强化学习的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.00358v1](http://arxiv.org/pdf/2510.00358v1)**

> **作者:** Linjin He; Xinda Qi; Dong Chen; Zhaojian Li; Xiaobo Tan
>
> **摘要:** Soft snake robots offer remarkable flexibility and adaptability in complex environments, yet their control remains challenging due to highly nonlinear dynamics. Existing model-based and bio-inspired controllers rely on simplified assumptions that limit performance. Deep reinforcement learning (DRL) has recently emerged as a promising alternative, but online training is often impractical because of costly and potentially damaging real-world interactions. Offline RL provides a safer option by leveraging pre-collected datasets, but it suffers from distribution shift, which degrades generalization to unseen scenarios. To overcome this challenge, we propose DiSA-IQL (Distribution-Shift-Aware Implicit Q-Learning), an extension of IQL that incorporates robustness modulation by penalizing unreliable state-action pairs to mitigate distribution shift. We evaluate DiSA-IQL on goal-reaching tasks across two settings: in-distribution and out-of-distribution evaluation. Simulation results show that DiSA-IQL consistently outperforms baseline models, including Behavior Cloning (BC), Conservative Q-Learning (CQL), and vanilla IQL, achieving higher success rates, smoother trajectories, and improved robustness. The codes are open-sourced to support reproducibility and to facilitate further research in offline RL for soft robot control.
>
---
#### [new 002] Prometheus: Universal, Open-Source Mocap-Based Teleoperation System with Force Feedback for Dataset Collection in Robot Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决低成本高精度数据收集问题。设计了一个基于动作捕捉的遥操作系统，集成力反馈，提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2510.01023v1](http://arxiv.org/pdf/2510.01023v1)**

> **作者:** S. Satsevich; A. Bazhenov; S. Egorov; A. Erkhov; M. Gromakov; A. Fedoseev; D. Tsetserukou
>
> **摘要:** This paper presents a novel teleoperation system with force feedback, utilizing consumer-grade HTC Vive Track- ers 2.0. The system integrates a custom-built controller, a UR3 robotic arm, and a Robotiq gripper equipped with custom- designed fingers to ensure uniform pressure distribution on an embedded force sensor. Real-time compression force data is transmitted to the controller, enabling operators to perceive the gripping force applied to objects. Experimental results demonstrate that the system enhances task success rates and provides a low-cost solution for large-scale imitation learning data collection without compromising affordability.
>
---
#### [new 003] Shared Object Manipulation with a Team of Collaborative Quadrupeds
- **分类: cs.RO; cs.MA; cs.SY; eess.SY**

- **简介: 该论文属于多机器人协同操作任务，解决团队协作搬运大体积物体的问题。通过改进控制算法，实现腿部机械臂团队稳定高效地共同操控物体。**

- **链接: [http://arxiv.org/pdf/2510.00682v1](http://arxiv.org/pdf/2510.00682v1)**

> **作者:** Shengzhi Wang; Niels Dehio; Xuanqi Zeng; Xian Yang; Lingwei Zhang; Yun-Hui Liu; K. W. Samuel Au
>
> **备注:** 8 pages, 9 figures, submitted to The 2026 American Control Conference
>
> **摘要:** Utilizing teams of multiple robots is advantageous for handling bulky objects. Many related works focus on multi-manipulator systems, which are limited by workspace constraints. In this paper, we extend a classical hybrid motion-force controller to a team of legged manipulator systems, enabling collaborative loco-manipulation of rigid objects with a force-closed grasp. Our novel approach allows the robots to flexibly coordinate their movements, achieving efficient and stable object co-manipulation and transport, validated through extensive simulations and real-world experiments.
>
---
#### [new 004] A Systematic Study of Large Language Models for Task and Motion Planning With PDDLStream
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究将大语言模型应用于任务与运动规划（TAMP），解决其在机器人任务中的规划能力问题。通过16种算法实验，分析LLM在TAMP中的表现与优化方向。**

- **链接: [http://arxiv.org/pdf/2510.00182v1](http://arxiv.org/pdf/2510.00182v1)**

> **作者:** Jorge Mendez-Mendez
>
> **摘要:** Using large language models (LLMs) to solve complex robotics problems requires understanding their planning capabilities. Yet while we know that LLMs can plan on some problems, the extent to which these planning capabilities cover the space of robotics tasks is unclear. One promising direction is to integrate the semantic knowledge of LLMs with the formal reasoning of task and motion planning (TAMP). However, the myriad of choices for how to integrate LLMs within TAMP complicates the design of such systems. We develop 16 algorithms that use Gemini 2.5 Flash to substitute key TAMP components. Our zero-shot experiments across 4,950 problems and three domains reveal that the Gemini-based planners exhibit lower success rates and higher planning times than their engineered counterparts. We show that providing geometric details increases the number of task-planning errors compared to pure PDDL descriptions, and that (faster) non-reasoning LLM variants outperform (slower) reasoning variants in most cases, since the TAMP system can direct the LLM to correct its mistakes.
>
---
#### [new 005] ROSflight 2.0: Lean ROS 2-Based Autopilot for Unmanned Aerial Vehicles
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于无人机自主控制任务，旨在降低UAV研究门槛。通过迁移至ROS 2、增强模块化与仿真环境，提升系统可用性与实验效率。**

- **链接: [http://arxiv.org/pdf/2510.00995v1](http://arxiv.org/pdf/2510.00995v1)**

> **作者:** Jacob Moore; Phil Tokumaru; Ian Reid; Brandon Sutherland; Joseph Ritchie; Gabe Snow; Tim McLain
>
> **备注:** To be submitted to the 2026 IEEE International Conference on Robotics and Automation in Vienna, Austria
>
> **摘要:** ROSflight is a lean, open-source autopilot ecosystem for unmanned aerial vehicles (UAVs). Designed by researchers for researchers, it is built to lower the barrier to entry to UAV research and accelerate the transition from simulation to hardware experiments by maintaining a lean (not full-featured), well-documented, and modular codebase. This publication builds on previous treatments and describes significant additions to the architecture that improve the modularity and usability of ROSflight, including the transition from ROS 1 to ROS 2, supported hardware, low-level actuator mixing, and the simulation environment. We believe that these changes improve the usability of ROSflight and enable ROSflight to accelerate research in areas like advanced-air mobility. Hardware results are provided, showing that ROSflight is able to control a multirotor over a serial connection at 400 Hz while closing all control loops on the companion computer.
>
---
#### [new 006] Trajectory Based Observer Design: A Framework for Lightweight Sensor Fusion
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决非线性系统中传感器融合问题。提出TBOD方法，通过优化预记录轨迹设计轻量级观测器，提升定位精度。**

- **链接: [http://arxiv.org/pdf/2510.00630v1](http://arxiv.org/pdf/2510.00630v1)**

> **作者:** Federico Oliva; Tom Shaked; Daniele Carnevale; Amir Degani
>
> **摘要:** Efficient observer design and accurate sensor fusion are key in state estimation. This work proposes an optimization-based methodology, termed Trajectory Based Optimization Design (TBOD), allowing the user to easily design observers for general nonlinear systems and multi-sensor setups. Starting from parametrized observer dynamics, the proposed method considers a finite set of pre-recorded measurement trajectories from the nominal plant and exploits them to tune the observer parameters through numerical optimization. This research hinges on the classic observer's theory and Moving Horizon Estimators methodology. Optimization is exploited to ease the observer's design, providing the user with a lightweight, general-purpose sensor fusion methodology. TBOD's main characteristics are the capability to handle general sensors efficiently and in a modular way and, most importantly, its straightforward tuning procedure. The TBOD's performance is tested on a terrestrial rover localization problem, combining IMU and ranging sensors provided by Ultra Wide Band antennas, and validated through a motion-capture system. Comparison with an Extended Kalman Filter is also provided, matching its position estimation accuracy and significantly improving in the orientation.
>
---
#### [new 007] Seeing through Uncertainty: Robust Task-Oriented Optimization in Visual Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决数据稀缺下策略泛化问题。提出NeuRO框架，结合感知与优化，提升环境适应能力。**

- **链接: [http://arxiv.org/pdf/2510.00441v1](http://arxiv.org/pdf/2510.00441v1)**

> **作者:** Yiyuan Pan; Yunzhe Xu; Zhe Liu; Hesheng Wang
>
> **摘要:** Visual navigation is a fundamental problem in embodied AI, yet practical deployments demand long-horizon planning capabilities to address multi-objective tasks. A major bottleneck is data scarcity: policies learned from limited data often overfit and fail to generalize OOD. Existing neural network-based agents typically increase architectural complexity that paradoxically become counterproductive in the small-sample regime. This paper introduce NeuRO, a integrated learning-to-optimize framework that tightly couples perception networks with downstream task-level robust optimization. Specifically, NeuRO addresses core difficulties in this integration: (i) it transforms noisy visual predictions under data scarcity into convex uncertainty sets using Partially Input Convex Neural Networks (PICNNs) with conformal calibration, which directly parameterize the optimization constraints; and (ii) it reformulates planning under partial observability as a robust optimization problem, enabling uncertainty-aware policies that transfer across environments. Extensive experiments on both unordered and sequential multi-object navigation tasks demonstrate that NeuRO establishes SoTA performance, particularly in generalization to unseen environments. Our work thus presents a significant advancement for developing robust, generalizable autonomous agents.
>
---
#### [new 008] VLA-RFT: Vision-Language-Action Reinforcement Fine-tuning with Verified Rewards in World Simulators
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型的强化学习任务，旨在解决模型在分布偏移下的鲁棒性问题。通过引入基于数据的世界模型，实现高效、稳定的策略优化。**

- **链接: [http://arxiv.org/pdf/2510.00406v1](http://arxiv.org/pdf/2510.00406v1)**

> **作者:** Hengtao Li; Pengxiang Ding; Runze Suo; Yihao Wang; Zirui Ge; Dongyuan Zang; Kexian Yu; Mingyang Sun; Hongyin Zhang; Donglin Wang; Weihua Su
>
> **摘要:** Vision-Language-Action (VLA) models enable embodied decision-making but rely heavily on imitation learning, leading to compounding errors and poor robustness under distribution shift. Reinforcement learning (RL) can mitigate these issues yet typically demands costly real-world interactions or suffers from sim-to-real gaps. We introduce VLA-RFT, a reinforcement fine-tuning framework that leverages a data-driven world model as a controllable simulator. Trained from real interaction data, the simulator predicts future visual observations conditioned on actions, allowing policy rollouts with dense, trajectory-level rewards derived from goal-achieving references. This design delivers an efficient and action-aligned learning signal, drastically lowering sample requirements. With fewer than 400 fine-tuning steps, VLA-RFT surpasses strong supervised baselines and achieves greater efficiency than simulator-based RL. Moreover, it exhibits strong robustness under perturbed conditions, sustaining stable task execution. Our results establish world-model-based RFT as a practical post-training paradigm to enhance the generalization and robustness of VLA models. For more details, please refer to https://vla-rft.github.io/.
>
---
#### [new 009] HAMLET: Switch your Vision-Language-Action Model into a History-Aware Policy
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决VLAs忽视历史上下文的问题。通过引入时间感知的moment tokens和轻量记忆模块，提升模型对历史信息的利用能力。**

- **链接: [http://arxiv.org/pdf/2510.00695v1](http://arxiv.org/pdf/2510.00695v1)**

> **作者:** Myungkyu Koo; Daewon Choi; Taeyoung Kim; Kyungmin Lee; Changyeon Kim; Youngyo Seo; Jinwoo Shin
>
> **备注:** Project page: https://myungkyukoo.github.io/hamlet/
>
> **摘要:** Inherently, robotic manipulation tasks are history-dependent: leveraging past context could be beneficial. However, most existing Vision-Language-Action models (VLAs) have been designed without considering this aspect, i.e., they rely solely on the current observation, ignoring preceding context. In this paper, we propose HAMLET, a scalable framework to adapt VLAs to attend to the historical context during action prediction. Specifically, we introduce moment tokens that compactly encode perceptual information at each timestep. Their representations are initialized with time-contrastive learning, allowing them to better capture temporally distinctive aspects. Next, we employ a lightweight memory module that integrates the moment tokens across past timesteps into memory features, which are then leveraged for action prediction. Through empirical evaluation, we show that HAMLET successfully transforms a state-of-the-art VLA into a history-aware policy, especially demonstrating significant improvements on long-horizon tasks that require historical context. In particular, on top of GR00T N1.5, HAMLET achieves an average success rate of 76.4% on history-dependent real-world tasks, surpassing the baseline performance by 47.2%. Furthermore, HAMLET pushes prior art performance from 64.1% to 66.4% on RoboCasa Kitchen (100-demo setup) and from 95.6% to 97.7% on LIBERO, highlighting its effectiveness even under generic robot-manipulation benchmarks.
>
---
#### [new 010] RTFF: Random-to-Target Fabric Flattening Policy using Dual-Arm Manipulator
- **分类: cs.RO**

- **简介: 该论文属于机器人服装操作任务，解决布料变形和对齐难题。提出RTFF策略，结合模仿学习与视觉伺服，实现高效、精准的布料平整。**

- **链接: [http://arxiv.org/pdf/2510.00814v1](http://arxiv.org/pdf/2510.00814v1)**

> **作者:** Kai Tang; Dipankar Bhattacharya; Hang Xu; Fuyuki Tokuda; Norman C. Tien; Kazuhiro Kosuge
>
> **备注:** 9 pages, 6 figures, conference
>
> **摘要:** Robotic fabric manipulation in garment production for sewing, cutting, and ironing requires reliable flattening and alignment, yet remains challenging due to fabric deformability, effectively infinite degrees of freedom, and frequent occlusions from wrinkles, folds, and the manipulator's End-Effector (EE) and arm. To address these issues, this paper proposes the first Random-to-Target Fabric Flattening (RTFF) policy, which aligns a random wrinkled fabric state to an arbitrary wrinkle-free target state. The proposed policy adopts a hybrid Imitation Learning-Visual Servoing (IL-VS) framework, where IL learns with explicit fabric models for coarse alignment of the wrinkled fabric toward a wrinkle-free state near the target, and VS ensures fine alignment to the target. Central to this framework is a template-based mesh that offers precise target state representation, wrinkle-aware geometry prediction, and consistent vertex correspondence across RTFF manipulation steps, enabling robust manipulation and seamless IL-VS switching. Leveraging the power of mesh, a novel IL solution for RTFF-Mesh Action Chunking Transformer (MACT)-is then proposed by conditioning the mesh information into a Transformer-based policy. The RTFF policy is validated on a real dual-arm tele-operation system, showing zero-shot alignment to different targets, high accuracy, and strong generalization across fabrics and scales. Project website: https://kaitang98.github.io/RTFF_Policy/
>
---
#### [new 011] What Did I Learn? Operational Competence Assessment for AI-Based Trajectory Planners
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶领域，解决机器学习模型在轨迹规划中的操作风险评估问题。通过知识图谱建模数据集，分析场景覆盖与复杂度以评估模型能力。**

- **链接: [http://arxiv.org/pdf/2510.00619v1](http://arxiv.org/pdf/2510.00619v1)**

> **作者:** Michiel Braat; Maren Buermann; Marijke van Weperen; Jan-Pieter Paardekooper
>
> **备注:** Accepted for publication in proceedings of the 2025 IEEE International Automated Vehicle Validation Conference
>
> **摘要:** Automated driving functions increasingly rely on machine learning for tasks like perception and trajectory planning, requiring large, relevant datasets. The performance of these algorithms depends on how closely the training data matches the task. To ensure reliable functioning, it is crucial to know what is included in the dataset to assess the trained model's operational risk. We aim to enhance the safe use of machine learning in automated driving by developing a method to recognize situations that an automated vehicle has not been sufficiently trained on. This method also improves explainability by describing the dataset at a human-understandable level. We propose modeling driving data as knowledge graphs, representing driving scenes with entities and their relationships. These graphs are queried for specific sub-scene configurations to check their occurrence in the dataset. We estimate a vehicle's competence in a driving scene by considering the coverage and complexity of sub-scene configurations in the training set. Higher complexity scenes require greater coverage for high competence. We apply this method to the NuPlan dataset, modeling it with knowledge graphs and analyzing the coverage of specific driving scenes. This approach helps monitor the competence of machine learning models trained on the dataset, which is essential for trustworthy AI to be deployed in automated driving.
>
---
#### [new 012] Tele-rehabilitation with online skill transfer and adaptation in $\mathbb{R}^3 \times \mathit{S}^3$
- **分类: cs.RO**

- **简介: 该论文属于机器人辅助远程康复任务，旨在解决远程精准康复训练问题。通过双机械臂协同与动态运动基元编码，实现动作传递与自适应调整。**

- **链接: [http://arxiv.org/pdf/2510.00770v1](http://arxiv.org/pdf/2510.00770v1)**

> **作者:** Tianle Ni; Xiao Chen; Hamid Sadeghian; Sami Haddadin
>
> **摘要:** This paper proposes a tele-teaching framework for the domain of robot-assisted tele-rehabilitation. The system connects two robotic manipulators on therapist and patient side via bilateral teleoperation, enabling a therapist to remotely demonstrate rehabilitation exercises that are executed by the patient-side robot. A 6-DoF Dynamical Movement Primitives formulation is employed to jointly encode translational and rotational motions in $\mathbb{R}^3 \times \mathit{S}^3$ space, ensuring accurate trajectory reproduction. The framework supports smooth transitions between therapist-led guidance and patient passive training, while allowing adaptive adjustment of motion. Experiments with 7-DoF manipulators demonstrate the feasibility of the approach, highlighting its potential for personalized and remotely supervised rehabilitation.
>
---
#### [new 013] Real-Time Trajectory Generation and Hybrid Lyapunov-Based Control for Hopping Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决 hopping 机器人无法直接控制空中轨迹的问题。提出一种实时轨迹生成与 Lyapunov 控制方法，实现复杂轨迹跟踪与姿态控制。**

- **链接: [http://arxiv.org/pdf/2510.01138v1](http://arxiv.org/pdf/2510.01138v1)**

> **作者:** Matthew Woodward
>
> **备注:** 7 pages, 4 figures, 4 tables
>
> **摘要:** The advent of rotor-based hopping robots has created very capable hopping platforms with high agility and efficiency, and similar controllability, as compared to their purely flying quadrotor counterparts. Advances in robot performance have increased the hopping height to greater than 4 meters and opened up the possibility for more complex aerial trajectories (i.e., behaviors). However, currently hopping robots do not directly control their aerial trajectory or transition to flight, eliminating the efficiency benefits of a hopping system. Here we show a real-time, computationally efficiency, non-linear drag compensated, trajectory generation methodology and accompanying Lyapunov-based controller. The combined system can create and follow complex aerial trajectories from liftoff to touchdown on horizontal and vertical surfaces, while maintaining strick control over the orientation at touchdown. The computational efficiency provides broad applicability across all size scales of hopping robots while maintaining applicability to quadrotors in general.
>
---
#### [new 014] RoboPilot: Generalizable Dynamic Robotic Manipulation with Dual-thinking Modes
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决复杂任务执行中的鲁棒性问题。提出RoboPilot框架，结合双思维模式与反馈机制，提升任务成功率与适应性。**

- **链接: [http://arxiv.org/pdf/2510.00154v1](http://arxiv.org/pdf/2510.00154v1)**

> **作者:** Xinyi Liu; Mohammadreza Fani Sani; Zewei Zhou; Julius Wirbel; Bahram Zarrin; Roberto Galeazzi
>
> **摘要:** Despite rapid progress in autonomous robotics, executing complex or long-horizon tasks remains a fundamental challenge. Most current approaches follow an open-loop paradigm with limited reasoning and no feedback, resulting in poor robustness to environmental changes and severe error accumulation. We present RoboPilot, a dual-thinking closed-loop framework for robotic manipulation that supports adaptive reasoning for complex tasks in real-world dynamic environments. RoboPilot leverages primitive actions for structured task planning and flexible action generation, while introducing feedback to enable replanning from dynamic changes and execution errors. Chain-of-Thought reasoning further enhances high-level task planning and guides low-level action generation. The system dynamically switches between fast and slow thinking to balance efficiency and accuracy. To systematically evaluate the robustness of RoboPilot in diverse robot manipulation scenarios, we introduce RoboPilot-Bench, a benchmark spanning 21 tasks across 10 categories, including infeasible-task recognition and failure recovery. Experiments show that RoboPilot outperforms state-of-the-art baselines by 25.9\% in task success rate, and the real-world deployment on an industrial robot further demonstrates its robustness in real-world settings.
>
---
#### [new 015] MultiPhysio-HRC: Multimodal Physiological Signals Dataset for industrial Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文属于人机协作领域，旨在解决感知人类心理状态的问题。通过收集多模态生理数据，构建了MultiPhysio-HRC数据集，用于提升机器人的人性化交互能力。**

- **链接: [http://arxiv.org/pdf/2510.00703v1](http://arxiv.org/pdf/2510.00703v1)**

> **作者:** Andrea Bussolan; Stefano Baraldo; Oliver Avram; Pablo Urcola; Luis Montesano; Luca Maria Gambardella; Anna Valente
>
> **摘要:** Human-robot collaboration (HRC) is a key focus of Industry 5.0, aiming to enhance worker productivity while ensuring well-being. The ability to perceive human psycho-physical states, such as stress and cognitive load, is crucial for adaptive and human-aware robotics. This paper introduces MultiPhysio-HRC, a multimodal dataset containing physiological, audio, and facial data collected during real-world HRC scenarios. The dataset includes electroencephalography (EEG), electrocardiography (ECG), electrodermal activity (EDA), respiration (RESP), electromyography (EMG), voice recordings, and facial action units. The dataset integrates controlled cognitive tasks, immersive virtual reality experiences, and industrial disassembly activities performed manually and with robotic assistance, to capture a holistic view of the participants' mental states. Rich ground truth annotations were obtained using validated psychological self-assessment questionnaires. Baseline models were evaluated for stress and cognitive load classification, demonstrating the dataset's potential for affective computing and human-aware robotics research. MultiPhysio-HRC is publicly available to support research in human-centered automation, workplace well-being, and intelligent robotic systems.
>
---
#### [new 016] Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **简介: 该论文属于多智能体运动预测任务，解决长时程预测误差累积和非线性交互问题。提出PINCoDE模型，结合物理约束与神经微分方程，实现高效准确的多机器人轨迹预测。**

- **链接: [http://arxiv.org/pdf/2510.00401v1](http://arxiv.org/pdf/2510.00401v1)**

> **作者:** Shounak Sural; Charles Kekeh; Wenliang Liu; Federico Pecora; Mouhacine Benosman
>
> **摘要:** Long-horizon motion forecasting for multiple autonomous robots is challenging due to non-linear agent interactions, compounding prediction errors, and continuous-time evolution of dynamics. Learned dynamics of such a system can be useful in various applications such as travel time prediction, prediction-guided planning and generative simulation. In this work, we aim to develop an efficient trajectory forecasting model conditioned on multi-agent goals. Motivated by the recent success of physics-guided deep learning for partially known dynamical systems, we develop a model based on neural Controlled Differential Equations (CDEs) for long-horizon motion forecasting. Unlike discrete-time methods such as RNNs and transformers, neural CDEs operate in continuous time, allowing us to combine physics-informed constraints and biases to jointly model multi-robot dynamics. Our approach, named PINCoDE (Physics-Informed Neural Controlled Differential Equations), learns differential equation parameters that can be used to predict the trajectories of a multi-agent system starting from an initial condition. PINCoDE is conditioned on future goals and enforces physics constraints for robot motion over extended periods of time. We adopt a strategy that scales our model from 10 robots to 100 robots without the need for additional model parameters, while producing predictions with an average ADE below 0.5 m for a 1-minute horizon. Furthermore, progressive training with curriculum learning for our PINCoDE model results in a 2.7X reduction of forecasted pose error over 4 minute horizons compared to analytical models.
>
---
#### [new 017] ROSplane 2.0: A Fixed-Wing Autopilot for Research
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于无人机自主控制研究，旨在简化自动驾驶框架的集成与开发。通过ROSplane 2.0提供灵活、易用的工具，加速算法测试与硬件实验。**

- **链接: [http://arxiv.org/pdf/2510.01041v1](http://arxiv.org/pdf/2510.01041v1)**

> **作者:** Ian Reid; Joseph Ritchie; Jacob Moore; Brandon Sutherland; Gabe Snow; Phillip Tokumaru; Tim McLain
>
> **摘要:** Unmanned aerial vehicle (UAV) research requires the integration of cutting-edge technology into existing autopilot frameworks. This process can be arduous, requiring extensive resources, time, and detailed knowledge of the existing system. ROSplane is a lean, open-source fixed-wing autonomy stack built by researchers for researchers. It is designed to accelerate research by providing clearly defined interfaces with an easily modifiable framework. Powered by ROS 2, ROSplane allows for rapid integration of low or high-level control, path planning, or estimation algorithms. A focus on lean, easily understood code and extensive documentation lowers the barrier to entry for researchers. Recent developments to ROSplane improve its capacity to accelerate UAV research, including the transition from ROS 1 to ROS 2, enhanced estimation and control algorithms, increased modularity, and an improved aerodynamic modeling pipeline. This aerodynamic modeling pipeline significantly reduces the effort of transitioning from simulation to real-world testing without requiring expensive system identification or computational fluid dynamics tools. ROSplane's architecture reduces the effort required to integrate new research tools and methods, expediting hardware experimentation.
>
---
#### [new 018] Two stage GNSS outlier detection for factor graph optimization based GNSS-RTK/INS/odometer fusion
- **分类: cs.RO**

- **简介: 该论文属于GNSS定位任务，旨在解决复杂环境中伪距异常值导致的定位精度下降问题。通过两阶段异常值检测方法，提升GNSS-RTK/INS/里程计融合系统的鲁棒性与定位精度。**

- **链接: [http://arxiv.org/pdf/2510.00524v1](http://arxiv.org/pdf/2510.00524v1)**

> **作者:** Baoshan Song; Penggao Yan; Xiao Xia; Yihan Zhong; Weisong Wen; Li-Ta Hsu
>
> **摘要:** Reliable GNSS positioning in complex environments remains a critical challenge due to non-line-of-sight (NLOS) propagation, multipath effects, and frequent signal blockages. These effects can easily introduce large outliers into the raw pseudo-range measurements, which significantly degrade the performance of global navigation satellite system (GNSS) real-time kinematic (RTK) positioning and limit the effectiveness of tightly coupled GNSS-based integrated navigation system. To address this issue, we propose a two-stage outlier detection method and apply the method in a tightly coupled GNSS-RTK, inertial navigation system (INS), and odometer integration based on factor graph optimization (FGO). In the first stage, Doppler measurements are employed to detect pseudo-range outliers in a GNSS-only manner, since Doppler is less sensitive to multipath and NLOS effects compared with pseudo-range, making it a more stable reference for detecting sudden inconsistencies. In the second stage, pre-integrated inertial measurement units (IMU) and odometer constraints are used to generate predicted double-difference pseudo-range measurements, which enable a more refined identification and rejection of remaining outliers. By combining these two complementary stages, the system achieves improved robustness against both gross pseudo-range errors and degraded satellite measuring quality. The experimental results demonstrate that the two-stage detection framework significantly reduces the impact of pseudo-range outliers, and leads to improved positioning accuracy and consistency compared with representative baseline approaches. In the deep urban canyon test, the outlier mitigation method has limits the RMSE of GNSS-RTK/INS/odometer fusion from 0.52 m to 0.30 m, with 42.3% improvement.
>
---
#### [new 019] From Human Hands to Robot Arms: Manipulation Skills Transfer via Trajectory Alignment
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作技能迁移任务，解决人类与机器人形态差异导致的知识转移难题。通过3D轨迹对齐实现技能迁移，提升机器人操作性能。**

- **链接: [http://arxiv.org/pdf/2510.00491v1](http://arxiv.org/pdf/2510.00491v1)**

> **作者:** Han Zhou; Jinjin Cao; Liyuan Ma; Xueji Fang; Guo-jun Qi
>
> **摘要:** Learning diverse manipulation skills for real-world robots is severely bottlenecked by the reliance on costly and hard-to-scale teleoperated demonstrations. While human videos offer a scalable alternative, effectively transferring manipulation knowledge is fundamentally hindered by the significant morphological gap between human and robotic embodiments. To address this challenge and facilitate skill transfer from human to robot, we introduce Traj2Action,a novel framework that bridges this embodiment gap by using the 3D trajectory of the operational endpoint as a unified intermediate representation, and then transfers the manipulation knowledge embedded in this trajectory to the robot's actions. Our policy first learns to generate a coarse trajectory, which forms an high-level motion plan by leveraging both human and robot data. This plan then conditions the synthesis of precise, robot-specific actions (e.g., orientation and gripper state) within a co-denoising framework. Extensive real-world experiments on a Franka robot demonstrate that Traj2Action boosts the performance by up to 27% and 22.25% over $\pi_0$ baseline on short- and long-horizon real-world tasks, and achieves significant gains as human data scales in robot policy learning. Our project website, featuring code and video demonstrations, is available at https://anonymous.4open.science/w/Traj2Action-4A45/.
>
---
#### [new 020] BC-MPPI: A Probabilistic Constraint Layer for Safe Model-Predictive Path-Integral Control
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决MPPI控制缺乏约束保障的问题。提出BC-MPPI，在每步重规划中使用概率代理评估轨迹可行性，提升安全性。**

- **链接: [http://arxiv.org/pdf/2510.00272v1](http://arxiv.org/pdf/2510.00272v1)**

> **作者:** Odichimnma Ezeji; Michael Ziegltrum; Giulio Turrisi; Tommaso Belvedere; Valerio Modugno
>
> **摘要:** Model Predictive Path Integral (MPPI) control has recently emerged as a fast, gradient-free alternative to model-predictive control in highly non-linear robotic tasks, yet it offers no hard guarantees on constraint satisfaction. We introduce Bayesian-Constraints MPPI (BC-MPPI), a lightweight safety layer that attaches a probabilistic surrogate to every state and input constraint. At each re-planning step the surrogate returns the probability that a candidate trajectory is feasible; this joint probability scales the weight given to a candidate, automatically down-weighting rollouts likely to collide or exceed limits and pushing the sampling distribution toward the safe subset; no hand-tuned penalty costs or explicit sample rejection required. We train the surrogate from 1000 offline simulations and deploy the controller on a quadrotor in MuJoCo with both static and moving obstacles. Across K in [100,1500] rollouts BC-MPPI preserves safety margins while satisfying the prescribed probability of violation. Because the surrogate is a stand-alone, version-controlled artefact and the runtime safety score is a single scalar, the approach integrates naturally with verification-and-validation pipelines for certifiable autonomous systems.
>
---
#### [new 021] Learning Human Reaching Optimality Principles from Minimal Observation Inverse Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于逆强化学习任务，旨在通过少量观测数据建模人类手臂运动的动态成本结构，解决传统方法需大量示例和时间的问题。**

- **链接: [http://arxiv.org/pdf/2510.00329v1](http://arxiv.org/pdf/2510.00329v1)**

> **作者:** Sarmad Mehrdad; Maxime Sabbah; Vincent Bonnet; Ludovic Righetti
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** This paper investigates the application of Minimal Observation Inverse Reinforcement Learning (MO-IRL) to model and predict human arm-reaching movements with time-varying cost weights. Using a planar two-link biomechanical model and high-resolution motion-capture data from subjects performing a pointing task, we segment each trajectory into multiple phases and learn phase-specific combinations of seven candidate cost functions. MO-IRL iteratively refines cost weights by scaling observed and generated trajectories in the maximum entropy IRL formulation, greatly reducing the number of required demonstrations and convergence time compared to classical IRL approaches. Training on ten trials per posture yields average joint-angle Root Mean Squared Errors (RMSE) of 6.4 deg and 5.6 deg for six- and eight-segment weight divisions, respectively, versus 10.4 deg using a single static weight. Cross-validation on remaining trials and, for the first time, inter-subject validation on an unseen subject's 20 trials, demonstrates comparable predictive accuracy, around 8 deg RMSE, indicating robust generalization. Learned weights emphasize joint acceleration minimization during movement onset and termination, aligning with smoothness principles observed in biological motion. These results suggest that MO-IRL can efficiently uncover dynamic, subject-independent cost structures underlying human motor control, with potential applications for humanoid robots.
>
---
#### [new 022] GRITS: A Spillage-Aware Guided Diffusion Policy for Robot Food Scooping Tasks
- **分类: cs.RO**

- **简介: 该论文针对机器人食物舀取任务，解决食物洒落问题。提出GRITS框架，利用引导扩散策略减少洒落，提升任务成功率。**

- **链接: [http://arxiv.org/pdf/2510.00573v1](http://arxiv.org/pdf/2510.00573v1)**

> **作者:** Yen-Ling Tai; Yi-Ru Yang; Kuan-Ting Yu; Yu-Wei Chao; Yi-Ting Chen
>
> **摘要:** Robotic food scooping is a critical manipulation skill for food preparation and service robots. However, existing robot learning algorithms, especially learn-from-demonstration methods, still struggle to handle diverse and dynamic food states, which often results in spillage and reduced reliability. In this work, we introduce GRITS: A Spillage-Aware Guided Diffusion Policy for Robot Food Scooping Tasks. This framework leverages guided diffusion policy to minimize food spillage during scooping and to ensure reliable transfer of food items from the initial to the target location. Specifically, we design a spillage predictor that estimates the probability of spillage given current observation and action rollout. The predictor is trained on a simulated dataset with food spillage scenarios, constructed from four primitive shapes (spheres, cubes, cones, and cylinders) with varied physical properties such as mass, friction, and particle size. At inference time, the predictor serves as a differentiable guidance signal, steering the diffusion sampling process toward safer trajectories while preserving task success. We validate GRITS on a real-world robotic food scooping platform. GRITS is trained on six food categories and evaluated on ten unseen categories with different shapes and quantities. GRITS achieves an 82% task success rate and a 4% spillage rate, reducing spillage by over 40% compared to baselines without guidance, thereby demonstrating its effectiveness.
>
---
#### [new 023] Product-oriented Product-Process-Resource Asset Network and its Representation in AutomationML for Asset Administration Shell
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于产品生命周期管理任务，旨在解决传统标准未覆盖产品全生命周期的问题。提出PoPAN模型，结合产品结构与再制造过程，并用AutomationML格式实现，以支持产品数字孪生。**

- **链接: [http://arxiv.org/pdf/2510.00933v1](http://arxiv.org/pdf/2510.00933v1)**

> **作者:** Sara Strakosova; Petr Novak; Petr Kadera
>
> **备注:** This work has been submitted to the IEEE for possible publication. 8 pages, 6 figures
>
> **摘要:** Current products, especially in the automotive sector, pose complex technical systems having a multi-disciplinary mechatronic nature. Industrial standards supporting system engineering and production typically (i) address the production phase only, but do not cover the complete product life cycle, and (ii) focus on production processes and resources rather than the products themselves. The presented approach is motivated by incorporating impacts of end-of-life phase of the product life cycle into the engineering phase. This paper proposes a modelling approach coming up from the Product-Process-Resource (PPR) modeling paradigm. It combines requirements on (i) respecting the product structure as a basis for the model, and (ii) it incorporates repairing, remanufacturing, or upcycling within cyber-physical production systems. The proposed model called PoPAN should accompany the product during the entire life cycle as a digital shadow encapsulated within the Asset Administration Shell of a product. To facilitate the adoption of the proposed paradigm, the paper also proposes serialization of the model in the AutomationML data format. The model is demonstrated on a use-case for disassembling electric vehicle batteries to support their remanufacturing for stationary battery applications.
>
---
#### [new 024] Integrating Offline Pre-Training with Online Fine-Tuning: A Reinforcement Learning Approach for Robot Social Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人社交导航任务，旨在解决离线训练与在线部署间的分布偏移问题。通过整合RTG预测和因果Transformer，提升导航策略的鲁棒性和适应性。**

- **链接: [http://arxiv.org/pdf/2510.00466v1](http://arxiv.org/pdf/2510.00466v1)**

> **作者:** Run Su; Hao Fu; Shuai Zhou; Yingao Fu
>
> **摘要:** Offline reinforcement learning (RL) has emerged as a promising framework for addressing robot social navigation challenges. However, inherent uncertainties in pedestrian behavior and limited environmental interaction during training often lead to suboptimal exploration and distributional shifts between offline training and online deployment. To overcome these limitations, this paper proposes a novel offline-to-online fine-tuning RL algorithm for robot social navigation by integrating Return-to-Go (RTG) prediction into a causal Transformer architecture. Our algorithm features a spatiotem-poral fusion model designed to precisely estimate RTG values in real-time by jointly encoding temporal pedestrian motion patterns and spatial crowd dynamics. This RTG prediction framework mitigates distribution shift by aligning offline policy training with online environmental interactions. Furthermore, a hybrid offline-online experience sampling mechanism is built to stabilize policy updates during fine-tuning, ensuring balanced integration of pre-trained knowledge and real-time adaptation. Extensive experiments in simulated social navigation environments demonstrate that our method achieves a higher success rate and lower collision rate compared to state-of-the-art baselines. These results underscore the efficacy of our algorithm in enhancing navigation policy robustness and adaptability. This work paves the way for more reliable and adaptive robotic navigation systems in real-world applications.
>
---
#### [new 025] TGPO: Temporal Grounded Policy Optimization for Signal Temporal Logic Tasks
- **分类: cs.RO; cs.AI; cs.LG; cs.LO**

- **简介: 该论文属于机器人控制任务，解决复杂长期任务的策略学习问题。针对STL任务的非马尔可夫性和稀疏奖励，提出TGPO方法，通过分解STL为子目标和约束，实现高效策略优化。**

- **链接: [http://arxiv.org/pdf/2510.00225v1](http://arxiv.org/pdf/2510.00225v1)**

> **作者:** Yue Meng; Fei Chen; Chuchu Fan
>
> **摘要:** Learning control policies for complex, long-horizon tasks is a central challenge in robotics and autonomous systems. Signal Temporal Logic (STL) offers a powerful and expressive language for specifying such tasks, but its non-Markovian nature and inherent sparse reward make it difficult to be solved via standard Reinforcement Learning (RL) algorithms. Prior RL approaches focus only on limited STL fragments or use STL robustness scores as sparse terminal rewards. In this paper, we propose TGPO, Temporal Grounded Policy Optimization, to solve general STL tasks. TGPO decomposes STL into timed subgoals and invariant constraints and provides a hierarchical framework to tackle the problem. The high-level component of TGPO proposes concrete time allocations for these subgoals, and the low-level time-conditioned policy learns to achieve the sequenced subgoals using a dense, stage-wise reward signal. During inference, we sample various time allocations and select the most promising assignment for the policy network to rollout the solution trajectory. To foster efficient policy learning for complex STL with multiple subgoals, we leverage the learned critic to guide the high-level temporal search via Metropolis-Hastings sampling, focusing exploration on temporally feasible solutions. We conduct experiments on five environments, ranging from low-dimensional navigation to manipulation, drone, and quadrupedal locomotion. Under a wide range of STL tasks, TGPO significantly outperforms state-of-the-art baselines (especially for high-dimensional and long-horizon cases), with an average of 31.6% improvement in task success rate compared to the best baseline. The code will be available at https://github.com/mengyuest/TGPO
>
---
#### [new 026] Non-submodular Visual Attention for Robot Navigation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人导航任务，解决VIN中的视觉特征选择问题，通过非子模函数和多项式时间算法提升效率与实时性。**

- **链接: [http://arxiv.org/pdf/2510.00942v1](http://arxiv.org/pdf/2510.00942v1)**

> **作者:** Reza Vafaee; Kian Behzad; Milad Siami; Luca Carlone; Ali Jadbabaie
>
> **备注:** 22 pages; Accepted to appear in IEEE Transactions on Robotics (T-RO)
>
> **摘要:** This paper presents a task-oriented computational framework to enhance Visual-Inertial Navigation (VIN) in robots, addressing challenges such as limited time and energy resources. The framework strategically selects visual features using a Mean Squared Error (MSE)-based, non-submodular objective function and a simplified dynamic anticipation model. To address the NP-hardness of this problem, we introduce four polynomial-time approximation algorithms: a classic greedy method with constant-factor guarantees; a low-rank greedy variant that significantly reduces computational complexity; a randomized greedy sampler that balances efficiency and solution quality; and a linearization-based selector based on a first-order Taylor expansion for near-constant-time execution. We establish rigorous performance bounds by leveraging submodularity ratios, curvature, and element-wise curvature analyses. Extensive experiments on both standardized benchmarks and a custom control-aware platform validate our theoretical results, demonstrating that these methods achieve strong approximation guarantees while enabling real-time deployment.
>
---
#### [new 027] Enabling High-Frequency Cross-Modality Visual Positioning Service for Accurate Drone Landing
- **分类: cs.RO**

- **简介: 该论文属于无人机精准着陆任务，解决城市环境中GPS定位不准确的问题。通过引入事件相机和EV-Pose系统，提升6-DoF姿态跟踪的精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.00646v1](http://arxiv.org/pdf/2510.00646v1)**

> **作者:** Haoyang Wang; Xinyu Luo; Wenhua Ding; Jingao Xu; Xuecheng Chen; Ruiyang Duan; Jialong Chen; Haitao Zhang; Yunhao Liu; Xinlei Chen
>
> **备注:** 15 pages, 23 figures
>
> **摘要:** After years of growth, drone-based delivery is transforming logistics. At its core, real-time 6-DoF drone pose tracking enables precise flight control and accurate drone landing. With the widespread availability of urban 3D maps, the Visual Positioning Service (VPS), a mobile pose estimation system, has been adapted to enhance drone pose tracking during the landing phase, as conventional systems like GPS are unreliable in urban environments due to signal attenuation and multi-path propagation. However, deploying the current VPS on drones faces limitations in both estimation accuracy and efficiency. In this work, we redesign drone-oriented VPS with the event camera and introduce EV-Pose to enable accurate, high-frequency 6-DoF pose tracking for accurate drone landing. EV-Pose introduces a spatio-temporal feature-instructed pose estimation module that extracts a temporal distance field to enable 3D point map matching for pose estimation; and a motion-aware hierarchical fusion and optimization scheme to enhance the above estimation in accuracy and efficiency, by utilizing drone motion in the \textit{early stage} of event filtering and the \textit{later stage} of pose optimization. Evaluation shows that EV-Pose achieves a rotation accuracy of 1.34$\degree$ and a translation accuracy of 6.9$mm$ with a tracking latency of 10.08$ms$, outperforming baselines by $>$50\%, \tmcrevise{thus enabling accurate drone landings.} Demo: https://ev-pose.github.io/
>
---
#### [new 028] A Novel Robust Control Method Combining DNN-Based NMPC Approximation and PI Control: Application to Exoskeleton Squat Movements
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决NMPC计算量大且鲁棒性不足的问题。通过结合DNN与PI控制器，提升控制精度与效率。**

- **链接: [http://arxiv.org/pdf/2510.00188v1](http://arxiv.org/pdf/2510.00188v1)**

> **作者:** Alireza Aliyari; Gholamreza Vossoughi
>
> **摘要:** Nonlinear Model Predictive Control (NMPC) is a precise controller, but its heavy computational load often prevents application in robotic systems. Some studies have attempted to approximate NMPC using deep neural networks (NMPC-DNN). However, in the presence of unexpected disturbances or when operating conditions differ from training data, this approach lacks robustness, leading to large tracking errors. To address this issue, for the first time, the NMPC-DNN output is combined with a PI controller (Hybrid NMPC-DNN-PI). The proposed controller is validated by applying it to an exoskeleton robot during squat movement, which has a complex dynamic model and has received limited attention regarding robust nonlinear control design. A human-robot dynamic model with three active joints (ankle, knee, hip) is developed, and more than 5.3 million training samples are used to train the DNN. The results show that, under unseen conditions for the DNN, the tracking error in Hybrid NMPC-DNN-PI is significantly lower compared to NMPC-DNN. Moreover, human joint torques are greatly reduced with the use of the exoskeleton, with RMS values for the studied case reduced by 30.9%, 41.8%, and 29.7% at the ankle, knee, and hip, respectively. In addition, the computational cost of Hybrid NMPC-DNN-PI is 99.93% lower than that of NMPC.
>
---
#### [new 029] Semantic Visual Simultaneous Localization and Mapping: A Survey on State of the Art, Challenges, and Future Directions
- **分类: cs.RO**

- **简介: 该论文属于语义视觉SLAM任务，旨在解决机器人环境建模与定位问题，综述了最新技术、挑战及未来方向。**

- **链接: [http://arxiv.org/pdf/2510.00783v1](http://arxiv.org/pdf/2510.00783v1)**

> **作者:** Thanh Nguyen Canh; Haolan Zhang; Xiem HoangVan; Nak Young Chong
>
> **摘要:** Semantic Simultaneous Localization and Mapping (SLAM) is a critical area of research within robotics and computer vision, focusing on the simultaneous localization of robotic systems and associating semantic information to construct the most accurate and complete comprehensive model of the surrounding environment. Since the first foundational work in Semantic SLAM appeared more than two decades ago, this field has received increasing attention across various scientific communities. Despite its significance, the field lacks comprehensive surveys encompassing recent advances and persistent challenges. In response, this study provides a thorough examination of the state-of-the-art of Semantic SLAM techniques, with the aim of illuminating current trends and key obstacles. Beginning with an in-depth exploration of the evolution of visual SLAM, this study outlines its strengths and unique characteristics, while also critically assessing previous survey literature. Subsequently, a unified problem formulation and evaluation of the modular solution framework is proposed, which divides the problem into discrete stages, including visual localization, semantic feature extraction, mapping, data association, and loop closure optimization. Moreover, this study investigates alternative methodologies such as deep learning and the utilization of large language models, alongside a review of relevant research about contemporary SLAM datasets. Concluding with a discussion on potential future research directions, this study serves as a comprehensive resource for researchers seeking to navigate the complex landscape of Semantic SLAM.
>
---
#### [new 030] Hybrid Training for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决长链式思维生成导致的推理延迟问题。提出Hybrid Training框架，使模型在推理时可选择是否生成思维，提升灵活性与效率。**

- **链接: [http://arxiv.org/pdf/2510.00600v1](http://arxiv.org/pdf/2510.00600v1)**

> **作者:** Pietro Mazzaglia; Cansu Sancaktar; Markus Peschl; Daniel Dijkman
>
> **摘要:** Using Large Language Models to produce intermediate thoughts, a.k.a. Chain-of-thought (CoT), before providing an answer has been a successful recipe for solving complex language tasks. In robotics, similar embodied CoT strategies, generating thoughts before actions, have also been shown to lead to improved performance when using Vision-Language-Action models (VLAs). As these techniques increase the length of the model's generated outputs to include the thoughts, the inference time is negatively affected. Delaying an agent's actions in real-world executions, as in robotic manipulation settings, strongly affects the usability of a method, as tasks require long sequences of actions. However, is the generation of long chains-of-thought a strong prerequisite for achieving performance improvements? In this work, we explore the idea of Hybrid Training (HyT), a framework that enables VLAs to learn from thoughts and benefit from the associated performance gains, while enabling the possibility to leave out CoT generation during inference. Furthermore, by learning to conditionally predict a diverse set of outputs, HyT supports flexibility at inference time, enabling the model to either predict actions directly, generate thoughts or follow instructions. We evaluate the proposed method in a series of simulated benchmarks and real-world experiments.
>
---
#### [new 031] CroSTAta: Cross-State Transition Attention Transformer for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，解决执行变异常见于训练数据的问题。提出Cross-State Transition Attention Transformer，通过状态转移注意力机制提升策略适应性。**

- **链接: [http://arxiv.org/pdf/2510.00726v1](http://arxiv.org/pdf/2510.00726v1)**

> **作者:** Giovanni Minelli; Giulio Turrisi; Victor Barasuol; Claudio Semini
>
> **备注:** Code and data available at https://github.com/iit-DLSLab/croSTAta
>
> **摘要:** Learning robotic manipulation policies through supervised learning from demonstrations remains challenging when policies encounter execution variations not explicitly covered during training. While incorporating historical context through attention mechanisms can improve robustness, standard approaches process all past states in a sequence without explicitly modeling the temporal structure that demonstrations may include, such as failure and recovery patterns. We propose a Cross-State Transition Attention Transformer that employs a novel State Transition Attention (STA) mechanism to modulate standard attention weights based on learned state evolution patterns, enabling policies to better adapt their behavior based on execution history. Our approach combines this structured attention with temporal masking during training, where visual information is randomly removed from recent timesteps to encourage temporal reasoning from historical context. Evaluation in simulation shows that STA consistently outperforms standard cross-attention and temporal modeling approaches like TCN and LSTM networks across all tasks, achieving more than 2x improvement over cross-attention on precision-critical tasks.
>
---
#### [new 032] Compose Your Policies! Improving Diffusion-based or Flow-based Robot Policies via Test-time Distribution-level Composition
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决数据获取成本高的问题。通过组合多个预训练策略的分布得分，提出GPC方法提升性能，无需额外训练。**

- **链接: [http://arxiv.org/pdf/2510.01068v1](http://arxiv.org/pdf/2510.01068v1)**

> **作者:** Jiahang Cao; Yize Huang; Hanzhong Guo; Rui Zhang; Mu Nan; Weijian Mai; Jiaxu Wang; Hao Cheng; Jingkai Sun; Gang Han; Wen Zhao; Qiang Zhang; Yijie Guo; Qihao Zheng; Chunfeng Song; Xiao Li; Ping Luo; Andrew F. Luo
>
> **备注:** Project Page: https://sagecao1125.github.io/GPC-Site/
>
> **摘要:** Diffusion-based models for robotic control, including vision-language-action (VLA) and vision-action (VA) policies, have demonstrated significant capabilities. Yet their advancement is constrained by the high cost of acquiring large-scale interaction datasets. This work introduces an alternative paradigm for enhancing policy performance without additional model training. Perhaps surprisingly, we demonstrate that the composed policies can exceed the performance of either parent policy. Our contribution is threefold. First, we establish a theoretical foundation showing that the convex composition of distributional scores from multiple diffusion models can yield a superior one-step functional objective compared to any individual score. A Gr\"onwall-type bound is then used to show that this single-step improvement propagates through entire generation trajectories, leading to systemic performance gains. Second, motivated by these results, we propose General Policy Composition (GPC), a training-free method that enhances performance by combining the distributional scores of multiple pre-trained policies via a convex combination and test-time search. GPC is versatile, allowing for the plug-and-play composition of heterogeneous policies, including VA and VLA models, as well as those based on diffusion or flow-matching, irrespective of their input visual modalities. Third, we provide extensive empirical validation. Experiments on Robomimic, PushT, and RoboTwin benchmarks, alongside real-world robotic evaluations, confirm that GPC consistently improves performance and adaptability across a diverse set of tasks. Further analysis of alternative composition operators and weighting strategies offers insights into the mechanisms underlying the success of GPC. These results establish GPC as a simple yet effective method for improving control performance by leveraging existing policies.
>
---
#### [new 033] Drones that Think on their Feet: Sudden Landing Decisions with Embodied AI
- **分类: cs.AI; cs.CR; cs.RO**

- **简介: 该论文属于自主无人机任务，解决突发情况下的快速决策问题。通过 embodied AI 实现动态环境下的自适应降落决策，提升系统安全性和韧性。**

- **链接: [http://arxiv.org/pdf/2510.00167v1](http://arxiv.org/pdf/2510.00167v1)**

> **作者:** Diego Ortiz Barbosa; Mohit Agrawal; Yash Malegaonkar; Luis Burbano; Axel Andersson; György Dán; Henrik Sandberg; Alvaro A. Cardenas
>
> **摘要:** Autonomous drones must often respond to sudden events, such as alarms, faults, or unexpected changes in their environment, that require immediate and adaptive decision-making. Traditional approaches rely on safety engineers hand-coding large sets of recovery rules, but this strategy cannot anticipate the vast range of real-world contingencies and quickly becomes incomplete. Recent advances in embodied AI, powered by large visual language models, provide commonsense reasoning to assess context and generate appropriate actions in real time. We demonstrate this capability in a simulated urban benchmark in the Unreal Engine, where drones dynamically interpret their surroundings and decide on sudden maneuvers for safe landings. Our results show that embodied AI makes possible a new class of adaptive recovery and decision-making pipelines that were previously infeasible to design by hand, advancing resilience and safety in autonomous aerial systems.
>
---
#### [new 034] Conflict-Based Search as a Protocol: A Multi-Agent Motion Planning Protocol for Heterogeneous Agents, Solvers, and Independent Tasks
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多智能体运动规划任务，解决异构系统在共享环境中的碰撞问题。通过将冲突搜索作为协议，实现不同规划器的协同路径规划。**

- **链接: [http://arxiv.org/pdf/2510.00425v1](http://arxiv.org/pdf/2510.00425v1)**

> **作者:** Rishi Veerapaneni; Alvin Tang; Haodong He; Sophia Zhao; Viraj Shah; Yidai Cen; Ziteng Ji; Gabriel Olin; Jon Arrizabalaga; Yorai Shaoul; Jiaoyang Li; Maxim Likhachev
>
> **备注:** Project webpage: https://rishi-v.github.io/CBS-Protocol/
>
> **摘要:** Imagine the future construction site, hospital, office, or even sophisticated household with dozens of robots bought from different manufacturers. How can we enable these different systems to effectively move in a shared environment, given that each robot may have its own independent motion planning system? This work shows how we can get efficient collision-free movements between algorithmically heterogeneous agents by using Conflict-Based Search (Sharon et al. 2015) as a protocol. At its core, the CBS Protocol requires one specific single-agent motion planning API; finding a collision-free path that satisfies certain space-time constraints. Given such an API, CBS uses a central planner to find collision-free paths - independent of how the API is implemented. We show how this protocol enables multi-agent motion planning for a heterogeneous team of agents completing independent tasks with a variety of single-agent planners including: Heuristic Search (e.g., A*), Sampling Based Search (e.g., RRT), Optimization (e.g., Direct Collocation), Diffusion, and Reinforcement Learning.
>
---
#### [new 035] Predictive Control Barrier Functions for Discrete-Time Linear Systems with Unmodeled Delays
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文属于控制理论任务，解决离散时间系统中状态约束问题。针对未知相对阶导致的约束难以满足的问题，提出预测控制屏障函数方法，简化安全集计算，无需辅助函数。**

- **链接: [http://arxiv.org/pdf/2510.01059v1](http://arxiv.org/pdf/2510.01059v1)**

> **作者:** Juan Augusto Paredes Salazar; James Usevitch; Ankit Goel
>
> **备注:** 8 pages, 7 figures, submitted to ACC 2026
>
> **摘要:** This paper introduces a predictive control barrier function (PCBF) framework for enforcing state constraints in discrete-time systems with unknown relative degree, which can be caused by input delays or unmodeled input dynamics. Existing discrete-time CBF formulations typically require the construction of auxiliary barrier functions when the relative degree is greater than one, which complicates implementation and may yield conservative safe sets. The proposed PCBF framework addresses this challenge by extending the prediction horizon to construct a CBF for an associated system with relative degree one. As a result, the superlevel set of the PCBF coincides with the safe set, simplifying constraint enforcement and eliminating the need for auxiliary functions. The effectiveness of the proposed method is demonstrated on a discrete-time double integrator with input delay and a bicopter system with position constraints.
>
---
#### [new 036] Strategic Fusion of Vision Language Models: Shapley-Credited Context-Aware Dawid-Skene for Multi-Label Tasks in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对自动驾驶中的多标签任务，解决VLMs的幻觉问题，通过融合模型输出提升可靠性与准确性。**

- **链接: [http://arxiv.org/pdf/2510.01126v1](http://arxiv.org/pdf/2510.01126v1)**

> **作者:** Yuxiang Feng; Keyang Zhang; Hassane Ouchouid; Ashwil Kaniamparambil; Ioannis Souflas; Panagiotis Angeloudis
>
> **备注:** 8 pages
>
> **摘要:** Large vision-language models (VLMs) are increasingly used in autonomous-vehicle (AV) stacks, but hallucination limits their reliability in safety-critical pipelines. We present Shapley-credited Context-Aware Dawid-Skene with Agreement, a game-theoretic fusion method for multi-label understanding of ego-view dashcam video. It learns per-model, per-label, context-conditioned reliabilities from labelled history and, at inference, converts each model's report into an agreement-guardrailed log-likelihood ratio that is combined with a contextual prior and a public reputation state updated via Shapley-based team credit. The result is calibrated, thresholdable posteriors that (i) amplify agreement among reliable models, (ii) preserve uniquely correct single-model signals, and (iii) adapt to drift. To specialise general VLMs, we curate 1,000 real-world dashcam clips with structured annotations (scene description, manoeuvre recommendation, rationale) via an automatic pipeline that fuses HDD ground truth, vehicle kinematics, and YOLOv11 + BoT-SORT tracking, guided by a three-step chain-of-thought prompt; three heterogeneous VLMs are then fine-tuned with LoRA. We evaluate with Hamming distance, Micro-Macro-F1, and average per-video latency. Empirically, the proposed method achieves a 23% reduction in Hamming distance, 55% improvement in Macro-F1, and 47% improvement in Micro-F1 when comparing with the best single model, supporting VLM fusion as a calibrated, interpretable, and robust decision-support component for AV pipelines.
>
---
#### [new 037] KeySG: Hierarchical Keyframe-Based 3D Scene Graphs
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景表示任务，解决传统场景图语义有限和扩展性差的问题。提出KeySG框架，通过分层结构和关键帧优化，提升场景理解与推理效率。**

- **链接: [http://arxiv.org/pdf/2510.01049v1](http://arxiv.org/pdf/2510.01049v1)**

> **作者:** Abdelrhman Werby; Dennis Rotondi; Fabio Scaparro; Kai O. Arras
>
> **摘要:** In recent years, 3D scene graphs have emerged as a powerful world representation, offering both geometric accuracy and semantic richness. Combining 3D scene graphs with large language models enables robots to reason, plan, and navigate in complex human-centered environments. However, current approaches for constructing 3D scene graphs are semantically limited to a predefined set of relationships, and their serialization in large environments can easily exceed an LLM's context window. We introduce KeySG, a framework that represents 3D scenes as a hierarchical graph consisting of floors, rooms, objects, and functional elements, where nodes are augmented with multi-modal information extracted from keyframes selected to optimize geometric and visual coverage. The keyframes allow us to efficiently leverage VLM to extract scene information, alleviating the need to explicitly model relationship edges between objects, enabling more general, task-agnostic reasoning and planning. Our approach can process complex and ambiguous queries while mitigating the scalability issues associated with large scene graphs by utilizing a hierarchical retrieval-augmented generation (RAG) pipeline to extract relevant context from the graph. Evaluated across four distinct benchmarks --including 3D object segmentation and complex query retrieval-- KeySG outperforms prior approaches on most metrics, demonstrating its superior semantic richness and efficiency.
>
---
#### [new 038] A Hierarchical Agentic Framework for Autonomous Drone-Based Visual Inspection
- **分类: cs.MA; cs.AI; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于工业无人机自主检测任务，解决传统方法在物理环境中的自动化不足问题。提出一种分层代理框架和ReActEval方法，实现无人机自主执行复杂视觉检测任务。**

- **链接: [http://arxiv.org/pdf/2510.00259v1](http://arxiv.org/pdf/2510.00259v1)**

> **作者:** Ethan Herron; Xian Yeow Lee; Gregory Sin; Teresa Gonzalez Diaz; Ahmed Farahat; Chetan Gupta
>
> **摘要:** Autonomous inspection systems are essential for ensuring the performance and longevity of industrial assets. Recently, agentic frameworks have demonstrated significant potential for automating inspection workflows but have been limited to digital tasks. Their application to physical assets in real-world environments, however, remains underexplored. In this work, our contributions are two-fold: first, we propose a hierarchical agentic framework for autonomous drone control, and second, a reasoning methodology for individual function executions which we refer to as ReActEval. Our framework focuses on visual inspection tasks in indoor industrial settings, such as interpreting industrial readouts or inspecting equipment. It employs a multi-agent system comprising a head agent and multiple worker agents, each controlling a single drone. The head agent performs high-level planning and evaluates outcomes, while worker agents implement ReActEval to reason over and execute low-level actions. Operating entirely in natural language, ReActEval follows a plan, reason, act, evaluate cycle, enabling drones to handle tasks ranging from simple navigation (e.g., flying forward 10 meters and land) to complex high-level tasks (e.g., locating and reading a pressure gauge). The evaluation phase serves as a feedback and/or replanning stage, ensuring actions align with user objectives while preventing undesirable outcomes. We evaluate the framework in a simulated environment with two worker agents, assessing performance qualitatively and quantitatively based on task completion across varying complexity levels and workflow efficiency. By leveraging natural language processing for agent communication, our approach offers a novel, flexible, and user-accessible alternative to traditional drone-based solutions, enabling autonomous problem-solving for industrial inspection without extensive user intervention.
>
---
#### [new 039] The Formation of Trust in Autonomous Vehicles after Interacting with Robotaxis on Public Roads
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，研究行人与自动驾驶车互动中信任的形成，通过真实实验分析信任变化因素。**

- **链接: [http://arxiv.org/pdf/2510.00120v1](http://arxiv.org/pdf/2510.00120v1)**

> **作者:** Xiang Chang; Zhijie Yi; Yichang Liu; Hongling Sheng; Dengbo He
>
> **备注:** Proceedings of the 69th HFES International Annual Meeting
>
> **摘要:** This study investigates how pedestrian trust, receptivity, and behavior evolve during interactions with Level-4 autonomous vehicles (AVs) at uncontrolled urban intersections in a naturalistic setting. While public acceptance is critical for AV adoption, most prior studies relied on simplified simulations or field tests. We conducted a real-world experiment in a commercial Robotaxi operation zone, where 33 participants repeatedly crossed an uncontrolled intersection with frequent Level-4 Robotaxi traffic. Participants completed the Pedestrian Behavior Questionnaire (PBQ), Pedestrian Receptivity Questionnaire for Fully AVs (PRQF), pre- and post-experiment Trust in AVs Scale, and Personal Innovativeness Scale (PIS). Results showed that trust in AVs significantly increased post-experiment, with the increase positively associated with the Interaction component of PRQF. Additionally, both the Positive and Error subscales of the PBQ significantly influenced trust change. This study reveals how trust forms in real-world pedestrian-AV encounters, offering insights beyond lab-based research by accounting for population heterogeneity.
>
---
#### [new 040] Robust Attitude Control of Nonlinear Multi-Rotor Dynamics with LFT Models and $\mathcal{H}_\infty$ Performance
- **分类: eess.SY; cs.RO; cs.SY; math.OC**

- **简介: 该论文属于无人机姿态控制任务，解决不确定环境下多旋翼非线性动态的稳定问题，通过LFT模型和$\mathcal{H}_\infty$控制实现鲁棒姿态调节。**

- **链接: [http://arxiv.org/pdf/2510.00208v1](http://arxiv.org/pdf/2510.00208v1)**

> **作者:** Tanay Kumar; Raktim Bhattacharya
>
> **备注:** 6 pages, 6 figures, 3 tables, submitted to ACC 2026
>
> **摘要:** Attitude stabilization of unmanned aerial vehicles in uncertain environments presents significant challenges due to nonlinear dynamics, parameter variations, and sensor limitations. This paper presents a comparative study of $\mathcal{H}_\infty$ and classical PID controllers for multi-rotor attitude regulation in the presence of wind disturbances and gyroscope noise. The flight dynamics are modeled using a linear parameter-varying (LPV) framework, where nonlinearities and parameter variations are systematically represented as structured uncertainties within a linear fractional transformation formulation. A robust controller based on $\mathcal{H}_\infty$ formulation is designed using only gyroscope measurements to ensure guaranteed performance bounds. Nonlinear simulation results demonstrate the effectiveness of the robust controllers compared to classical PID control, showing significant improvement in attitude regulation under severe wind disturbances.
>
---
#### [new 041] Less is More: Lean yet Powerful Vision-Language Model for Autonomous Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决轨迹规划问题。提出Max-V1框架，通过视觉语言模型实现端到端路径预测，提升性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.00060v1](http://arxiv.org/pdf/2510.00060v1)**

> **作者:** Sheng Yang; Tong Zhan; Guancheng Chen; Yanfeng Lu; Jian Wang
>
> **摘要:** In this work, we reconceptualize autonomous driving as a generalized language and formulate the trajectory planning task as next waypoint prediction. We introduce Max-V1, a novel framework for one-stage end-to-end autonomous driving. Our framework presents a single-pass generation paradigm that aligns with the inherent sequentiality of driving. This approach leverages the generative capacity of the VLM (Vision-Language Model) to enable end-to-end trajectory prediction directly from front-view camera input. The efficacy of this method is underpinned by a principled supervision strategy derived from statistical modeling. This provides a well-defined learning objective, which makes the framework highly amenable to master complex driving policies through imitation learning from large-scale expert demonstrations. Empirically, our method achieves the state-of-the-art performance on the nuScenes dataset, delivers an overall improvement of over 30% compared to prior baselines. Furthermore, it exhibits superior generalization performance on cross-domain datasets acquired from diverse vehicles, demonstrating notable potential for cross-vehicle robustness and adaptability. Due to these empirical strengths, this work introduces a model enabling fundamental driving behaviors, laying the foundation for the development of more capable self-driving agents. Code will be available upon publication.
>
---
#### [new 042] EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于轨迹预测任务，解决真实世界中因视角噪声导致的预测不准确问题。提出EgoTraj-Bench基准和BiFlow模型，提升预测鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.00405v1](http://arxiv.org/pdf/2510.00405v1)**

> **作者:** Jiayi Liu; Jiaming Zhou; Ke Ye; Kun-Yu Lin; Allan Wang; Junwei Liang
>
> **摘要:** Reliable trajectory prediction from an ego-centric perspective is crucial for robotic navigation in human-centric environments. However, existing methods typically assume idealized observation histories, failing to account for the perceptual artifacts inherent in first-person vision, such as occlusions, ID switches, and tracking drift. This discrepancy between training assumptions and deployment reality severely limits model robustness. To bridge this gap, we introduce EgoTraj-Bench, the first real-world benchmark that grounds noisy, first-person visual histories in clean, bird's-eye-view future trajectories, enabling robust learning under realistic perceptual constraints. Building on this benchmark, we propose BiFlow, a dual-stream flow matching model that concurrently denoises historical observations and forecasts future motion by leveraging a shared latent representation. To better model agent intent, BiFlow incorporates our EgoAnchor mechanism, which conditions the prediction decoder on distilled historical features via feature modulation. Extensive experiments show that BiFlow achieves state-of-the-art performance, reducing minADE and minFDE by 10-15% on average and demonstrating superior robustness. We anticipate that our benchmark and model will provide a critical foundation for developing trajectory forecasting systems truly resilient to the challenges of real-world, ego-centric perception.
>
---
## 更新

#### [replaced 001] Active Shadowing (ASD): Manipulating Perception of Robotic Behaviors via Implicit Virtual Communication
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.01468v4](http://arxiv.org/pdf/2407.01468v4)**

> **作者:** Andrew Boateng; Prakhar Bhartiya; Taha Shaheen; Yu Zhang
>
> **摘要:** Explicit communication is often valued for its directness in presenting information but requires attention during exchange, resulting in cognitive interruptions. On the other hand, implicit communication contributes to tacit and smooth interaction, making it more suitable for teaming, but requires inference for interpretation. This paper studies a novel type of implicit visual communication (IVC) using shadows via visual projection with augmented reality, referred to as active shadowing (ASD). Prior IVC methods, such as legible motion, are often used to influence the perception of robot behavior to make it more understandable. They often require changing the physical robot behavior, resulting in suboptimality. In our work, we investigate how ASD can be used to achieve similar effects without losing optimality. Our evaluations with user studies demonstrates that ASD can effectively creates ''illusions'' that maintain optimal physical behavior without compromising its understandability. We also show that ASD can be more informative than other explicit communication methods, and examine the conditions under which ASD becomes less effective.
>
---
#### [replaced 002] Optimization-based Task and Motion Planning under Signal Temporal Logic Specifications using Logic Network Flow
- **分类: cs.RO; cs.FL**

- **链接: [http://arxiv.org/pdf/2409.19168v3](http://arxiv.org/pdf/2409.19168v3)**

> **作者:** Xuan Lin; Jiming Ren; Samuel Coogan; Ye Zhao
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025
>
> **摘要:** This paper proposes an optimization-based task and motion planning framework, named "Logic Network Flow", to integrate signal temporal logic (STL) specifications into efficient mixed-binary linear programmings. In this framework, temporal predicates are encoded as polyhedron constraints on each edge of the network flow, instead of as constraints between the nodes as in the traditional Logic Tree formulation. Synthesized with Dynamic Network Flows, Logic Network Flows render a tighter convex relaxation compared to Logic Trees derived from these STL specifications. Our formulation is evaluated on several multi-robot motion planning case studies. Empirical results demonstrate that our formulation outperforms Logic Tree formulation in terms of computation time for several planning problems. As the problem size scales up, our method still discovers better lower and upper bounds by exploring fewer number of nodes during the branch-and-bound process, although this comes at the cost of increased computational load for each node when exploring branches.
>
---
#### [replaced 003] How Safe Will I Be Given What I Saw? Calibrated Prediction of Safety Chances for Image-Controlled Autonomy
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.09346v2](http://arxiv.org/pdf/2508.09346v2)**

> **作者:** Zhenjiang Mao; Mrinall Eashaan Umasudhan; Ivan Ruchkin
>
> **备注:** arXiv admin note: text overlap with arXiv:2308.12252
>
> **摘要:** Autonomous robots that rely on deep neural network controllers pose critical challenges for safety prediction, especially under partial observability and distribution shift. Traditional model-based verification techniques are limited in scalability and require access to low-dimensional state models, while model-free methods often lack reliability guarantees. This paper addresses these limitations by introducing a framework for calibrated safety prediction in end-to-end vision-controlled systems, where neither the state-transition model nor the observation model is accessible. Building on the foundation of world models, we leverage variational autoencoders and recurrent predictors to forecast future latent trajectories from raw image sequences and estimate the probability of satisfying safety properties. We distinguish between monolithic and composite prediction pipelines and introduce a calibration mechanism to quantify prediction confidence. In long-horizon predictions from high-dimensional observations, the forecasted inputs to the safety evaluator can deviate significantly from the training distribution due to compounding prediction errors and changing environmental conditions, leading to miscalibrated risk estimates. To address this, we incorporate unsupervised domain adaptation to ensure robustness of safety evaluation under distribution shift in predictions without requiring manual labels. Our formulation provides theoretical calibration guarantees and supports practical evaluation across long prediction horizons. Experimental results on three benchmarks show that our UDA-equipped evaluators maintain high accuracy and substantially lower false positive rates under distribution shift. Similarly, world model-based composite predictors outperform their monolithic counterparts on long-horizon tasks, and our conformal calibration provides reliable statistical bounds.
>
---
#### [replaced 004] RoVerFly: Robust and Versatile Implicit Hybrid Control of Quadrotor-Payload Systems
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.11149v2](http://arxiv.org/pdf/2509.11149v2)**

> **作者:** Mintae Kim; Jiaze Cai; Koushil Sreenath
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Designing robust controllers for precise trajectory tracking with quadrotors is challenging due to nonlinear dynamics and underactuation, and becomes harder with flexible cable-suspended payloads that add degrees of freedom and hybrid dynamics. Classical model-based methods offer stability guarantees but require extensive tuning and often fail to adapt when the configuration changes-when a payload is added or removed, or when its mass or cable length varies. We present RoVerFly, a unified learning-based control framework where a single reinforcement learning (RL) policy functions as an implicit hybrid controller, managing complex dynamics without explicit mode detection or controller switching. Trained with task and domain randomization, the controller is resilient to disturbances and varying dynamics. It achieves strong zero-shot generalization across payload settings-including no payload as well as varying mass and cable length-without re-tuning, while retaining the interpretability and structure of a feedback tracking controller. Code and supplementary materials are available at https://github.com/mintaeshkim/roverfly.
>
---
#### [replaced 005] Beyond Needle(s) in the Embodied Haystack: Environment, Architecture, and Training Considerations for Long Context Reasoning
- **分类: cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16928v2](http://arxiv.org/pdf/2505.16928v2)**

> **作者:** Bosung Kim; Prithviraj Ammanabrolu
>
> **摘要:** We introduce $\infty$-THOR, a new framework for long-horizon embodied tasks that advances long-context understanding in embodied AI. $\infty$-THOR provides: (1) a generation framework for synthesizing scalable, reproducible, and unlimited long-horizon trajectories; (2) a novel embodied QA task, Needle(s) in the Embodied Haystack, where multiple scattered clues across extended trajectories test agents' long-context reasoning ability; and (3) a long-horizon dataset and benchmark suite featuring complex tasks that span hundreds of environment steps, each paired with ground-truth action sequences. To enable this capability, we explore architectural adaptations, including interleaved Goal-State-Action modeling, context extension techniques, and Context Parallelism, to equip LLM-based agents for extreme long-context reasoning and interaction. Experimental results and analyses highlight the challenges posed by our benchmark and provide insights into training strategies and model behaviors under long-horizon conditions. Our work provides a foundation for the next generation of embodied AI systems capable of robust, long-term reasoning and planning.
>
---
#### [replaced 006] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.21046v2](http://arxiv.org/pdf/2508.21046v2)**

> **作者:** Wei Li; Renshan Zhang; Rui Shao; Jie He; Liqiang Nie
>
> **备注:** Accepted to NeurIPS 2025, Project Page: https://jiutian-vl.github.io/CogVLA-page
>
> **摘要:** Recent Vision-Language-Action (VLA) models built on pre-trained Vision-Language Models (VLMs) require extensive post-training, resulting in high computational overhead that limits scalability and deployment.We propose CogVLA, a Cognition-Aligned Vision-Language-Action framework that leverages instruction-driven routing and sparsification to improve both efficiency and performance. CogVLA draws inspiration from human multimodal coordination and introduces a 3-stage progressive architecture. 1) Encoder-FiLM based Aggregation Routing (EFA-Routing) injects instruction information into the vision encoder to selectively aggregate and compress dual-stream visual tokens, forming a instruction-aware latent representation. 2) Building upon this compact visual encoding, LLM-FiLM based Pruning Routing (LFP-Routing) introduces action intent into the language model by pruning instruction-irrelevant visually grounded tokens, thereby achieving token-level sparsity. 3) To ensure that compressed perception inputs can still support accurate and coherent action generation, we introduce V-L-A Coupled Attention (CAtten), which combines causal vision-language attention with bidirectional action parallel decoding. Extensive experiments on the LIBERO benchmark and real-world robotic tasks demonstrate that CogVLA achieves state-of-the-art performance with success rates of 97.4% and 70.0%, respectively, while reducing training costs by 2.5-fold and decreasing inference latency by 2.8-fold compared to OpenVLA. CogVLA is open-sourced and publicly available at https://github.com/JiuTian-VL/CogVLA.
>
---
#### [replaced 007] Vision-driven River Following of UAV via Safe Reinforcement Learning using Semantic Dynamics Model
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.09971v2](http://arxiv.org/pdf/2508.09971v2)**

> **作者:** Zihan Wang; Nina Mahmoudian
>
> **备注:** Submitted to Robotics and Autonomous Systems (RAS) journal
>
> **摘要:** Vision-driven autonomous river following by Unmanned Aerial Vehicles is critical for applications such as rescue, surveillance, and environmental monitoring, particularly in dense riverine environments where GPS signals are unreliable. These safety-critical navigation tasks must satisfy hard safety constraints while optimizing performance. Moreover, the reward in river following is inherently history-dependent (non-Markovian) by which river segment has already been visited, making it challenging for standard safe Reinforcement Learning (SafeRL). To address these gaps, we propose three contributions. First, we introduce Marginal Gain Advantage Estimation, which refines the reward advantage function by using a sliding window baseline computed from historical episodic returns, aligning the advantage estimate with non-Markovian dynamics. Second, we develop a Semantic Dynamics Model based on patchified water semantic masks offering more interpretable and data-efficient short-term prediction of future observations compared to latent vision dynamics models. Third, we present the Constrained Actor Dynamics Estimator architecture, which integrates the actor, cost estimator, and SDM for cost advantage estimation to form a model-based SafeRL framework. Simulation results demonstrate that MGAE achieves faster convergence and superior performance over traditional critic-based methods like Generalized Advantage Estimation. SDM provides more accurate short-term state predictions that enable the cost estimator to better predict potential violations. Overall, CADE effectively integrates safety regulation into model-based RL, with the Lagrangian approach providing a "soft" balance between reward and safety during training, while the safety layer enhances inference by imposing a "hard" action overlay.
>
---
#### [replaced 008] On the Application of Model Predictive Control to a Weighted Coverage Path Planning Problem
- **分类: eess.SY; cs.MA; cs.RO; cs.SY; 93-08**

- **链接: [http://arxiv.org/pdf/2411.08634v2](http://arxiv.org/pdf/2411.08634v2)**

> **作者:** Kilian Schweppe; Ludmila Moshagen; Georg Schildbach
>
> **摘要:** This paper considers the application of Model Predictive Control (MPC) to a weighted coverage path planning (WCPP) problem. The problem appears in a wide range of practical applications, including search and rescue (SAR) missions. The basic setup is that one (or multiple) agents can move around a given search space and collect rewards from a given spatial distribution. Unlike an artificial potential field, each reward can only be collected once. In contrast to a Traveling Salesman Problem (TSP), the agent moves in a continuous space. Moreover, he is not obliged to cover all locations and/or may return to previously visited locations. The WCPP problem is tackled by a new Model Predictive Control (MPC) formulation with so-called Coverage Constraints (CCs). It is shown that the solution becomes more effective if the solver is initialized with a TSP-based heuristic. With and without this initialization, the proposed MPC approach clearly outperforms a naive MPC formulation, as demonstrated in a small simulation study.
>
---
#### [replaced 009] Learning Hierarchical Domain Models Through Environment-Grounded Interaction
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.13497v3](http://arxiv.org/pdf/2505.13497v3)**

> **作者:** Claudius Kienle; Benjamin Alt; Oleg Arenz; Jan Peters
>
> **摘要:** Domain models enable autonomous agents to solve long-horizon tasks by producing interpretable plans. However, in open-world environments, a single general domain model cannot capture the variety of tasks, so agents must generate suitable task-specific models on the fly. Large Language Models (LLMs), with their implicit common knowledge, can generate such domains, but suffer from high error rates that limit their applicability. Hence, related work relies on extensive human feed-back or prior knowledge, which undermines autonomous, open-world deployment. In this work, we propose LODGE, a framework for autonomous domain learning from LLMs and environment grounding. LODGE builds on hierarchical abstractions and automated simulations to identify and correct inconsistencies between abstraction layers and between the model and environment. Our framework is task-agnostic, as it generates predicates, operators, and their preconditions and effects, while only assuming access to a simulator and a set of generic, executable low-level skills. Experiments on two International Planning Competition ( IPC) domains and a robotic assembly domain show that LODGE yields more accurate domain models and higher task success than existing methods, requiring remarkably few environment interactions and no human feedback or demonstrations.
>
---
#### [replaced 010] HetSwarm: Cooperative Navigation of Heterogeneous Swarm in Dynamic and Dense Environments through Impedance-based Guidance
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.06722v2](http://arxiv.org/pdf/2502.06722v2)**

> **作者:** Malaika Zafar; Roohan Ahmed Khan; Aleksey Fedoseev; Kumar Katyayan Jaiswal; Dzmitry Tsetserukou
>
> **备注:** Manuscript has been accepted by ICUAS-2025
>
> **摘要:** With the growing demand for efficient logistics and warehouse management, unmanned aerial vehicles (UAVs) are emerging as a valuable complement to automated guided vehicles (AGVs). UAVs enhance efficiency by navigating dense environments and operating at varying altitudes. However, their limited flight time, battery life, and payload capacity necessitate a supporting ground station. To address these challenges, we propose HetSwarm, a heterogeneous multi-robot system that combines a UAV and a mobile ground robot for collaborative navigation in cluttered and dynamic conditions. Our approach employs an artificial potential field (APF)-based path planner for the UAV, allowing it to dynamically adjust its trajectory in real time. The ground robot follows this path while maintaining connectivity through impedance links, ensuring stable coordination. Additionally, the ground robot establishes temporal impedance links with low-height ground obstacles to avoid local collisions, as these obstacles do not interfere with the UAV's flight. Experimental validation of HetSwarm in diverse environmental conditions demonstrated a 90% success rate across 30 test cases. The ground robot exhibited an average deviation of 45 cm near obstacles, confirming effective collision avoidance. Extensive simulations in the Gym PyBullet environment further validated the robustness of our system for real-world applications, demonstrating its potential for dynamic, real-time task execution in cluttered environments.
>
---
#### [replaced 011] Probabilistic Collision Risk Estimation through Gauss-Legendre Cubature and Non-Homogeneous Poisson Processes
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.18819v2](http://arxiv.org/pdf/2507.18819v2)**

> **作者:** Trent Weiss; Madhur Behl
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Overtaking in high-speed autonomous racing demands precise, real-time estimation of collision risk; particularly in wheel-to-wheel scenarios where safety margins are minimal. Existing methods for collision risk estimation either rely on simplified geometric approximations, like bounding circles, or perform Monte Carlo sampling which leads to overly conservative motion planning behavior at racing speeds. We introduce the Gauss-Legendre Rectangle (GLR) algorithm, a principled two-stage integration method that estimates collision risk by combining Gauss-Legendre with a non-homogeneous Poisson process over time. GLR produces accurate risk estimates that account for vehicle geometry and trajectory uncertainty. In experiments across 446 overtaking scenarios in a high-fidelity Formula One racing simulation, GLR outperforms five state-of-the-art baselines achieving an average error reduction of 77% and surpassing the next-best method by 52%, all while running at 1000 Hz. The framework is general and applicable to broader motion planning contexts beyond autonomous racing.
>
---
#### [replaced 012] Act to See, See to Act: Diffusion-Driven Perception-Action Interplay for Adaptive Policies
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.25822v2](http://arxiv.org/pdf/2509.25822v2)**

> **作者:** Jing Wang; Weiting Peng; Jing Tang; Zeyu Gong; Xihua Wang; Bo Tao; Li Cheng
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025)
>
> **摘要:** Existing imitation learning methods decouple perception and action, which overlooks the causal reciprocity between sensory representations and action execution that humans naturally leverage for adaptive behaviors. To bridge this gap, we introduce Action-Guided Diffusion Policy (DP-AG), a unified representation learning that explicitly models a dynamic interplay between perception and action through probabilistic latent dynamics. DP-AG encodes latent observations into a Gaussian posterior via variational inference and evolves them using an action-guided SDE, where the Vector-Jacobian Product (VJP) of the diffusion policy's noise predictions serves as a structured stochastic force driving latent updates. To promote bidirectional learning between perception and action, we introduce a cycle-consistent contrastive loss that organizes the gradient flow of the noise predictor into a coherent perception-action loop, enforcing mutually consistent transitions in both latent updates and action refinements. Theoretically, we derive a variational lower bound for the action-guided SDE, and prove that the contrastive objective enhances continuity in both latent and action trajectories. Empirically, DP-AG significantly outperforms state-of-the-art methods across simulation benchmarks and real-world UR5 manipulation tasks. As a result, our DP-AG offers a promising step toward bridging biological adaptability and artificial policy learning.
>
---
#### [replaced 013] State Estimation for Compliant and Morphologically Adaptive Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.25945v2](http://arxiv.org/pdf/2509.25945v2)**

> **作者:** Valentin Yuryev; Max Polzin; Josie Hughes
>
> **备注:** 8 pages, 10 figures, 1 table, preprint
>
> **摘要:** Locomotion robots with active or passive compliance can show robustness to uncertain scenarios, which can be promising for agricultural, research and environmental industries. However, state estimation for these robots is challenging due to the lack of rigid-body assumptions and kinematic changes from morphing. We propose a method to estimate typical rigid-body states alongside compliance-related states, such as soft robot shape in different morphologies and locomotion modes. Our neural network-based state estimator uses a history of states and a mechanism to directly influence unreliable sensors. We test our framework on the GOAT platform, a robot capable of passive compliance and active morphing for extreme outdoor terrain. The network is trained on motion capture data in a novel compliance-centric frame that accounts for morphing-related states. Our method predicts shape-related measurements within 4.2% of the robot's size, velocities within 6.3% and 2.4% of the top linear and angular speeds, respectively, and orientation within 1.5 degrees. We also demonstrate a 300% increase in travel range during a motor malfunction when using our estimator for closed-loop autonomous outdoor operation.
>
---
#### [replaced 014] ImpedanceGPT: VLM-driven Impedance Control of Swarm of Mini-drones for Intelligent Navigation in Dynamic Environment
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.02723v2](http://arxiv.org/pdf/2503.02723v2)**

> **作者:** Faryal Batool; Yasheerah Yaqoot; Malaika Zafar; Roohan Ahmed Khan; Muhammad Haris Khan; Aleksey Fedoseev; Dzmitry Tsetserukou
>
> **备注:** Accepted in IROS 2025
>
> **摘要:** Swarm robotics plays a crucial role in enabling autonomous operations in dynamic and unpredictable environments. However, a major challenge remains ensuring safe and efficient navigation in environments filled with both dynamic alive (e.g., humans) and dynamic inanimate (e.g., non-living objects) obstacles. In this paper, we propose ImpedanceGPT, a novel system that combines a Vision-Language Model (VLM) with retrieval-augmented generation (RAG) to enable real-time reasoning for adaptive navigation of mini-drone swarms in complex environments. The key innovation of ImpedanceGPT lies in the integration of VLM and RAG, which provides the drones with enhanced semantic understanding of their surroundings. This enables the system to dynamically adjust impedance control parameters in response to obstacle types and environmental conditions. Our approach not only ensures safe and precise navigation but also improves coordination between drones in the swarm. Experimental evaluations demonstrate the effectiveness of the system. The VLM-RAG framework achieved an obstacle detection and retrieval accuracy of 80 % under optimal lighting. In static environments, drones navigated dynamic inanimate obstacles at 1.4 m/s but slowed to 0.7 m/s with increased separation around humans. In dynamic environments, speed adjusted to 1.0 m/s near hard obstacles, while reducing to 0.6 m/s with higher deflection to safely avoid moving humans.
>
---
#### [replaced 015] Poutine: Vision-Language-Trajectory Pre-Training and Reinforcement Learning Post-Training Enable Robust End-to-End Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.11234v2](http://arxiv.org/pdf/2506.11234v2)**

> **作者:** Luke Rowe; Rodrigue de Schaetzen; Roger Girgis; Christopher Pal; Liam Paull
>
> **摘要:** We present Poutine, a 3B-parameter vision-language model (VLM) tailored for end-to-end autonomous driving in long-tail driving scenarios. Poutine is trained in two stages. To obtain strong base driving capabilities, we train Poutine-Base in a self-supervised vision-language-trajectory (VLT) next-token prediction fashion on 83 hours of CoVLA nominal driving and 11 hours of Waymo long-tail driving. Accompanying language annotations are auto-generated with a 72B-parameter VLM. Poutine is obtained by fine-tuning Poutine-Base with Group Relative Policy Optimization (GRPO) using less than 500 preference-labeled frames from the Waymo validation set. We show that both VLT pretraining and RL fine-tuning are critical to attain strong driving performance in the long-tail. Poutine-Base achieves a rater-feedback score (RFS) of 8.12 on the validation set, nearly matching Waymo's expert ground-truth RFS. The final Poutine model achieves an RFS of 7.99 on the official Waymo test set, placing 1st in the 2025 Waymo Vision-Based End-to-End Driving Challenge by a significant margin. These results highlight the promise of scalable VLT pre-training and lightweight RL fine-tuning to enable robust and generalizable autonomy.
>
---
#### [replaced 016] HR-INR: Continuous Space-Time Video Super-Resolution via Event Camera
- **分类: cs.CV; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.13389v2](http://arxiv.org/pdf/2405.13389v2)**

> **作者:** Yunfan Lu; Yusheng Wang; Zipeng Wang; Pengteng Li; Bin Yang; Hui Xiong
>
> **备注:** Project page: https://github.com/yunfanLu/HR-INR
>
> **摘要:** Continuous space-time video super-resolution (C-STVSR) aims to simultaneously enhance video resolution and frame rate at an arbitrary scale. Recently, implicit neural representation (INR) has been applied to video restoration, representing videos as implicit fields that can be decoded at an arbitrary scale. However, existing INR-based C-STVSR methods typically rely on only two frames as input, leading to insufficient inter-frame motion information. Consequently, they struggle to capture fast, complex motion and long-term dependencies (spanning more than three frames), hindering their performance in dynamic scenes. In this paper, we propose a novel C-STVSR framework, named HR-INR, which captures both holistic dependencies and regional motions based on INR. It is assisted by an event camera -- a novel sensor renowned for its high temporal resolution and low latency. To fully utilize the rich temporal information from events, we design a feature extraction consisting of (1) a regional event feature extractor -- taking events as inputs via the proposed event temporal pyramid representation to capture the regional nonlinear motion and (2) a holistic event-frame feature extractor for long-term dependence and continuity motion. We then propose a novel INR-based decoder with spatiotemporal embeddings to capture long-term dependencies with a larger temporal perception field. We validate the effectiveness and generalization of our method on four datasets (both simulated and real data), showing the superiority of our method. The project page is available at https://github.com/yunfanLu/HR-INR
>
---
#### [replaced 017] LLM-guided Task and Motion Planning using Knowledge-based Reasoning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.07493v3](http://arxiv.org/pdf/2412.07493v3)**

> **作者:** Muhayy Ud Din; Jan Rosell; Waseem Akram; Isiah Zaplana; Maximo A Roa; Irfan Hussain
>
> **备注:** Submitted to knowledge based systems
>
> **摘要:** Performing complex manipulation tasks in dynamic environments requires efficient Task and Motion Planning (TAMP) approaches that combine high-level symbolic plans with low-level motion control. Advances in Large Language Models (LLMs), such as GPT-4, are transforming task planning by offering natural language as an intuitive and flexible way to describe tasks, generate symbolic plans, and reason. However, the effectiveness of LLM-based TAMP approaches is limited due to static and template-based prompting, which limits adaptability to dynamic environments and complex task contexts. To address these limitations, this work proposes a novel Onto-LLM-TAMP framework that employs knowledge-based reasoning to refine and expand user prompts with task-contextual reasoning and knowledge-based environment state descriptions. Integrating domain-specific knowledge into the prompt ensures semantically accurate and context-aware task plans. The proposed framework demonstrates its effectiveness by resolving semantic errors in symbolic plan generation, such as maintaining logical temporal goal ordering in scenarios involving hierarchical object placement. The proposed framework is validated through both simulation and real-world scenarios, demonstrating significant improvements over the baseline approach in terms of adaptability to dynamic environments and the generation of semantically correct task plans.
>
---
#### [replaced 018] DBF-MA: A Differential Bayesian Filtering Planner for Multi-Agent Autonomous Racing Overtakes
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.22937v2](http://arxiv.org/pdf/2509.22937v2)**

> **作者:** Trent Weiss; Amar Kulkarni; Madhur Behl
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** A significant challenge in autonomous racing is to generate overtaking maneuvers. Racing agents must execute these maneuvers on complex racetracks with little room for error. Optimization techniques and graph-based methods have been proposed, but these methods often rely on oversimplified assumptions for collision-avoidance and dynamic constraints. In this work, we present an approach to trajectory synthesis based on an extension of the Differential Bayesian Filtering framework. Our approach for collision-free trajectory synthesis frames the problem as one of Bayesian Inference over the space of Composite Bezier Curves. Our method is derivative-free, does not require a spherical approximation of the vehicle footprint, linearization of constraints, or simplifying upper bounds on collision avoidance. We conduct a closed-loop analysis of DBF-MA and find it successfully overtakes an opponent in 87% of tested scenarios, outperforming existing methods in autonomous overtaking.
>
---
#### [replaced 019] Sampling-Based Global Optimal Control and Estimation via Semidefinite Programming
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.17572v2](http://arxiv.org/pdf/2507.17572v2)**

> **作者:** Antoine Groudiev; Fabian Schramm; Éloïse Berthier; Justin Carpentier; Frederike Dümbgen
>
> **摘要:** Global optimization has gained attraction over the past decades, thanks to the development of both theoretical foundations and efficient numerical routines. Among recent advances, Kernel Sum of Squares (KernelSOS) provides a powerful theoretical framework, combining the expressivity of kernel methods with the guarantees of SOS optimization. In this paper, we take KernelSOS from theory to practice and demonstrate its use on challenging control and robotics problems. We identify and address the practical considerations required to make the method work in applied settings: restarting strategies, systematic calibration of hyperparameters, methods for recovering minimizers, and the combination with fast local solvers. As a proof of concept, the application of KernelSOS to robot localization highlights its competitiveness with existing SOS approaches that rely on heuristics and handcrafted reformulations to render the problem polynomial. Even in the high-dimensional, non-parametric setting of trajectory optimization with simulators treated as black boxes, we demonstrate how KernelSOS can be combined with fast local solvers to uncover higher-quality solutions without compromising overall runtimes.
>
---
#### [replaced 020] Certifiably Optimal Estimation and Calibration in Robotics via Trace-Constrained Semi-Definite Programming
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.23656v2](http://arxiv.org/pdf/2509.23656v2)**

> **作者:** Liangting Wu; Roberto Tron
>
> **备注:** Manuscript submitted to American Control Conference (ACC) 2026
>
> **摘要:** Many nonconvex problems in robotics can be relaxed into convex formulations via Semi-Definite Programming (SDP) that can be solved to global optimality. The practical quality of these solutions, however, critically depends on rounding them to rank-1 matrices, a condition that can be challenging to achieve. In this work, we focus on trace-constrained SDPs (TCSDPs), where the decision variables are Positive Semi-Definite (PSD) matrices with fixed trace values. We show that the latter can be used to design a gradient-based refinement procedure that projects relaxed SDP solutions toward rank-1, low-cost candidates. We also provide fixed-trace SDP relaxations for common robotic quantities, such as rotations and translations, and a modular virtual robot abstraction that simplifies modeling across different problem settings. We demonstrate that our trace-constrained SDP framework can be applied to many robotics tasks, and we showcase its effectiveness through simulations in Perspective-n-Point (PnP) estimation, hand-eye calibration, and dual-robot system calibration.
>
---
#### [replaced 021] Heterogeneous Predictor-based Risk-Aware Planning with Conformal Prediction in Dense, Uncertain Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.11920v2](http://arxiv.org/pdf/2507.11920v2)**

> **作者:** Jeongyong Yang; KwangBin Lee; SooJean Han
>
> **摘要:** Real-time planning among many uncertain, dynamic obstacles is challenging because predicting every agent with high fidelity is both unnecessary and computationally expensive. We present Heterogeneous Predictor-based Risk-Aware Planning (H-PRAP), a framework that allocates prediction effort to where it matters. H-PRAP introduces the Probability-based Collision Risk Index (P-CRI), a closed-form, horizon-level collision index obtained by calibrating a Gaussian surrogate with conformal prediction. P-CRI drives a router that assigns high-risk obstacles to accurate but expensive predictors and low-risk obstacles to lightweight predictors, while preserving distribution-free coverage across heterogeneous predictors through conformal prediction. The selected predictions and their conformal radii are embedded in a chance-constrained model predictive control (MPC) problem, yielding receding-horizon policies with explicit safety margins. We analyze the safety-efficiency trade-off under prediction compute budget: more portion of low-fidelity predictions reduce residual risk from dropped obstacles, but in the same time induces larger conformal radii and degrades trajectory efficiency and shrinks MPC feasibility. Extensive numerical simulations in dense, uncertain environments validate that H-PRAP attains best balance between trajectory success rate (i.e., no collisions) and the time to reach the goal (i.e., trajectory efficiency) compared to single prediction architectures.
>
---
#### [replaced 022] Adaptive Diffusion Constrained Sampling for Bimanual Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.13667v4](http://arxiv.org/pdf/2505.13667v4)**

> **作者:** Haolei Tong; Yuezhe Zhang; Sophie Lueth; Georgia Chalvatzaki
>
> **摘要:** Coordinated multi-arm manipulation requires satisfying multiple simultaneous geometric constraints across high-dimensional configuration spaces, which poses a significant challenge for traditional planning and control methods. In this work, we propose Adaptive Diffusion Constrained Sampling (ADCS), a generative framework that flexibly integrates both equality (e.g., relative and absolute pose constraints) and structured inequality constraints (e.g., proximity to object surfaces) into an energy-based diffusion model. Equality constraints are modeled using dedicated energy networks trained on pose differences in Lie algebra space, while inequality constraints are represented via Signed Distance Functions (SDFs) and encoded into learned constraint embeddings, allowing the model to reason about complex spatial regions. A key innovation of our method is a Transformer-based architecture that learns to weight constraint-specific energy functions at inference time, enabling flexible and context-aware constraint integration. Moreover, we adopt a two-phase sampling strategy that improves precision and sample diversity by combining Langevin dynamics with resampling and density-aware re-weighting. Experimental results on dual-arm manipulation tasks show that ADCS significantly improves sample diversity and generalization across settings demanding precise coordination and adaptive constraint handling.
>
---
#### [replaced 023] Are All Marine Species Created Equal? Performance Disparities in Underwater Object Detection
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.18729v2](http://arxiv.org/pdf/2508.18729v2)**

> **作者:** Melanie Wille; Tobias Fischer; Scarlett Raine
>
> **备注:** 10 pages for main paper, 4 pages for supplementary material
>
> **摘要:** Underwater object detection is critical for monitoring marine ecosystems but poses unique challenges, including degraded image quality, imbalanced class distribution, and distinct visual characteristics. Not every species is detected equally well, yet underlying causes remain unclear. We address two key research questions: 1) What factors beyond data quantity drive class-specific performance disparities? 2) How can we systematically improve detection of under-performing marine species? We manipulate the DUO and RUOD datasets to separate the object detection task into localization and classification and investigate the under-performance of the scallop class. Localization analysis using YOLO11 and TIDE finds that foreground-background discrimination is the most problematic stage regardless of data quantity. Classification experiments reveal persistent precision gaps even with balanced data, indicating intrinsic feature-based challenges beyond data scarcity and inter-class dependencies. We recommend imbalanced distributions when prioritizing precision, and balanced distributions when prioritizing recall. Improving under-performing classes should focus on algorithmic advances, especially within localization modules. We publicly release our code and datasets.
>
---
