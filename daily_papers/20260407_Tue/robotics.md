# 机器人 cs.RO

- **最新发布 63 篇**

- **更新 35 篇**

## 最新发布

#### [new 001] Veo-Act: How Far Can Frontier Video Models Advance Generalizable Robot Manipulation?
- **分类: cs.RO**

- **简介: 该论文研究视频生成模型在机器人操作中的应用，解决通用机器人学习问题。通过结合视频模型与逆动力学模型，提升机器人任务轨迹生成能力。**

- **链接: [https://arxiv.org/pdf/2604.04502](https://arxiv.org/pdf/2604.04502)**

> **作者:** Zhongru Zhang; Chenghan Yang; Qingzhou Lu; Yanjiang Guo; Jianke Zhang; Yucheng Hu; Jianyu Chen
>
> **备注:** 16 pages, 12 figures. Equal contribution by Zhongru Zhang, Chenghan Yang, Qingzhou Lu and Yanjiang Guo. Project lead: Yanjiang Guo
>
> **摘要:** Video generation models have advanced rapidly and are beginning to show a strong understanding of physical dynamics. In this paper, we investigate how far an advanced video generation model such as Veo-3 can support generalizable robotic manipulation. We first study a zero-shot approach in which Veo-3 predicts future image sequences from current robot observations, while an inverse dynamics model IDM recovers the corresponding robot actions. The IDM is trained solely on random-play data, requiring neither human supervision nor expert demonstrations. The key intuition is that, if a video model can generate physically plausible future motions in image space, an IDM can translate those visual trajectories into executable robot actions. We evaluate this "Veo-3+IDM" approach in both simulation and the real world using a high-dimensional dexterous hand. We find that, owing to the strong generalization capability of frontier video models, Veo-3+IDM can consistently generate approximately correct task-level trajectories. However, its low-level control accuracy remains insufficient to solve most tasks reliably. Motivated by this observation, we develop a hierarchical framework, Veo-Act, which uses Veo-3 as a high-level motion planner and a VLA policy as the low-level executor, significantly improving the instruction-following performance of a state-of-the-art vision-language-action policy. Overall, our results suggest that, as video generation models continue to improve, video models can be a valuable component for generalizable robot learning.
>
---
#### [new 002] From Prompt to Physical Action: Structured Backdoor Attacks on LLM-Mediated Robotic Control Systems
- **分类: cs.RO**

- **简介: 该论文研究LLM控制机器人系统中的结构后门攻击问题，通过实验分析不同阶段的后门传播效果，提出防御方案并揭示安全与响应速度的权衡。**

- **链接: [https://arxiv.org/pdf/2604.03890](https://arxiv.org/pdf/2604.03890)**

> **作者:** Mingyang Xie; Jin Wei-Kocsis
>
> **摘要:** The integration of large language models (LLMs) into robotic control pipelines enables natural language interfaces that translate user prompts into executable commands. However, this digital-to-physical interface introduces a critical and underexplored vulnerability: structured backdoor attacks embedded during fine-tuning. In this work, we experimentally investigate LoRA-based supply-chain backdoors in LLM-mediated ROS2 robotic control systems and evaluate their impact on physical robot execution. We construct two poisoned fine-tuning strategies targeting different stages of the command generation pipeline and reveal a key systems-level insight: back-doors embedded at the natural-language reasoning stage do not reliably propagate to executable control outputs, whereas backdoors aligned directly with structured JSON command formats successfully survive translation and trigger physical actions. In both simulation and real-world experiments, backdoored models achieve an average Attack Success Rate of 83% while maintaining over 93% Clean Performance Accuracy (CPA) and sub-second latency, demonstrating both reliability and stealth. We further implement an agentic verification defense using a secondary LLM for semantic consistency checking. Although this reduces the Attack Success Rate (ASR) to 20%, it increases end-to-end latency to 8-9 seconds, exposing a significant security-responsiveness trade-off in real-time robotic systems. These results highlight structural vulnerabilities in LLM-mediated robotic control architectures and underscore the need for robotics-aware defenses for embodied AI systems.
>
---
#### [new 003] Learning-Based Fault Detection for Legged Robots in Remote Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于故障检测任务，旨在解决四足机器人在动态环境中单肢故障的识别问题，通过离线学习方法从本体传感器数据中检测故障并调整步态。**

- **链接: [https://arxiv.org/pdf/2604.03397](https://arxiv.org/pdf/2604.03397)**

> **作者:** Abriana Stewart-Height; Seema Jahagirdar; Nikolai Matni
>
> **摘要:** Operations in hazardous environments put humans, animals, and machines at high risk for physically damaging consequences. In contrast to humans and animals, quadruped robots cannot naturally identify and adjust their locomotion to a severely debilitated limb. The ability to detect limb damage and adjust movement to a new physical morphology is the difference between survival and death for humans and animals. The same can be said for quadruped robots autonomously carrying out remote assignments in dynamic, complex settings. This work presents the development and implementation of an off-line learning-based method to detect single limb faults from proprioceptive sensor data in a quadrupedal robot. The aim of the fault detection technique is to provide the correct output for the controller to select the appropriate tripedal gait to use given the robot's current physical morphology.
>
---
#### [new 004] AnyUser: Translating Sketched User Intent into Domestic Robots
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文提出AnyUser系统，解决家用机器人任务指令的非专家交互问题。通过草图和语言融合理解用户意图，生成可执行动作，提升任务完成效率与用户体验。**

- **链接: [https://arxiv.org/pdf/2604.04811](https://arxiv.org/pdf/2604.04811)**

> **作者:** Songyuan Yang; Huibin Tan; Kailun Yang; Wenjing Yang; Shaowu Yang
>
> **备注:** Accepted to IEEE Transactions on Robotics (T-RO)
>
> **摘要:** We introduce AnyUser, a unified robotic instruction system for intuitive domestic task instruction via free-form sketches on camera images, optionally with language. AnyUser interprets multimodal inputs (sketch, vision, language) as spatial-semantic primitives to generate executable robot actions requiring no prior maps or models. Novel components include multimodal fusion for understanding and a hierarchical policy for robust action generation. Efficacy is shown via extensive evaluations: (1) Quantitative benchmarks on the large-scale dataset showing high accuracy in interpreting diverse sketch-based commands across various simulated domestic scenes. (2) Real-world validation on two distinct robotic platforms, a statically mounted 7-DoF assistive arm (KUKA LBR iiwa) and a dual-arm mobile manipulator (Realman RMC-AIDAL), performing representative tasks like targeted wiping and area cleaning, confirming the system's ability to ground instructions and execute them reliably in physical environments. (3) A comprehensive user study involving diverse demographics (elderly, simulated non-verbal, low technical literacy) demonstrating significant improvements in usability and task specification efficiency, achieving high task completion rates (85.7%-96.4%) and user satisfaction. AnyUser bridges the gap between advanced robotic capabilities and the need for accessible non-expert interaction, laying the foundation for practical assistive robots adaptable to real-world human environments.
>
---
#### [new 005] Drift-Based Policy Optimization: Native One-Step Policy Learning for Online Robot Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决多步生成策略推理慢的问题。提出DBP和DBPO框架，实现单步生成策略，提升推理速度与控制频率。**

- **链接: [https://arxiv.org/pdf/2604.03540](https://arxiv.org/pdf/2604.03540)**

> **作者:** Yuxuan Gao; Yedong Shen; Shiqi Zhang; Wenhao Yu; Yifan Duan; Jia pan; Jiajia Wu; Jiajun Deng; Yanyong Zhang
>
> **摘要:** Although multi-step generative policies achieve strong performance in robotic manipulation by modeling multimodal action distributions, they require multi-step iterative denoising at inference time. Each action therefore needs tens to hundreds of network function evaluations (NFEs), making them costly for high-frequency closed-loop control and online reinforcement learning (RL). To address this limitation, we propose a two-stage framework for native one-step generative policies that shifts refinement from inference to training. First, we introduce the Drift-Based Policy (DBP), which leverages fixed-point drifting objectives to internalize iterative refinement into the model parameters, yielding a one-step generative backbone by design while preserving multimodal action modeling capacity. Second, we develop Drift-Based Policy Optimization (DBPO), an online RL framework that equips the pretrained backbone with a compatible stochastic interface, enabling stable on-policy updates without sacrificing the one-step deployment property. Extensive experiments demonstrate the effectiveness of the proposed framework across offline imitation learning, online fine-tuning, and real-world control scenarios. DBP matches or exceeds the performance of multi-step diffusion policies while achieving up to $100\times$ faster inference. It also consistently outperforms existing one-step baselines on challenging manipulation benchmarks. Moreover, DBPO enables effective and stable policy improvement in online settings. Experiments on a real-world dual-arm robot demonstrate reliable high-frequency control at 105.2 Hz.
>
---
#### [new 006] Surrogate Model-Based Near-Optimal Gain Selection for Approach-Angle-Constrained Two-Phase Pure Proportional Navigation
- **分类: cs.RO**

- **简介: 该论文属于制导优化任务，解决两阶段纯比例导引中导航增益的近优选择问题。通过构建神经网络模型，实现对最优增益的高效预测。**

- **链接: [https://arxiv.org/pdf/2604.03371](https://arxiv.org/pdf/2604.03371)**

> **作者:** Abhigyan Roy; Shreeya Padte; Abel Viji George; Vivek A; Satadal Ghosh
>
> **备注:** 6 pages
>
> **摘要:** In guidance literature, Pure Proportional Navigation (PPN) guidance is widely used for aerodynamically driven vehicles. A two-phase extension of PPN (2pPPN), which uses different navigation gains for an orientation phase and a final phase, has been presented to achieve any desired approach angle within an angular half-space. Recent studies show that the orientation phase can be realized through multiple feasible trajectories, creating an opportunity to select navigation gains that minimize overall guidance effort. This paper addresses the problem of near-optimal gain selection for given initial and desired terminal engagement geometries. Two optimization problems are considered: i) determination of the optimal orientation-phase gain for a specified final-phase gain, and ii) simultaneously determining the optimal gain pair for both phases that minimizes the total guidance effort. Determining the optimal gains analytically for arbitrary engagement geometries is intractable. Numerical simulations further reveal that these optimal gains vary smoothly with respect to the engagement conditions. Exploiting this property, a neural network (NN)-based regression model is developed in this paper to learn the nonlinear mapping between optimal gains and initial and desired terminal engagement geometries. The trained NN serves as a computationally efficient surrogate for generating the optimal gains manifold, enabling near-optimal realization of 2pPPN guidance. Numerical simulation studies demonstrate that the developed NN-based architecture predicts optimal gains with high accuracy, achieving very high (close to 0.9) value of coefficient of determination.
>
---
#### [new 007] Optimizing Neurorobot Policy under Limited Demonstration Data through Preference Regret
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人强化学习任务，解决有限示范数据下的策略优化问题。提出MYOE框架和QMoP-SSM模型，通过偏好遗憾优化控制策略，提升性能与适应性。**

- **链接: [https://arxiv.org/pdf/2604.03523](https://arxiv.org/pdf/2604.03523)**

> **作者:** Viet Dung Nguyen; Yuhang Song; Anh Nguyen; Jamison Heard; Reynold Bailey; Alexander Ororbia
>
> **备注:** 10 pages, 4 figures, 4 tables
>
> **摘要:** Robot reinforcement learning from demonstrations (RLfD) assumes that expert data is abundant; this is usually unrealistic in the real world given data scarcity as well as high collection cost. Furthermore, imitation learning algorithms assume that the data is independently and identically distributed, which ultimately results in poorer performance as gradual errors emerge and compound within test-time trajectories. We address these issues by introducing the "master your own expertise" (MYOE) framework, a self-imitation framework that enables robotic agents to learn complex behaviors from limited demonstration data samples. Inspired by human perception and action, we propose and design what we call the queryable mixture-of-preferences state space model (QMoP-SSM), which estimates the desired goal at every time step. These desired goals are used in computing the "preference regret", which is used to optimize the robot control policy. Our experiments demonstrate the robustness, adaptability, and out-of-sample performance of our agent compared to other state-of-the-art RLfD schemes. The GitHub repository that supports this work can be found at: this https URL.
>
---
#### [new 008] A Multi-View 3D Telepresence System for XR Robot Teleoperation
- **分类: cs.RO**

- **简介: 该论文属于机器人远程操作任务，旨在解决传统屏幕界面在3D可视化和深度感知上的不足。通过多视角VR系统融合点云与RGB流，提升远程操控的准确性和用户体验。**

- **链接: [https://arxiv.org/pdf/2604.03730](https://arxiv.org/pdf/2604.03730)**

> **作者:** Enes Ulas Dincer; Manuel Zaremski; Alexandra Nick; Elias Wucher; Barbara Deml; Gerhard Neumann
>
> **摘要:** Robot teleoperation is critical for applications such as remote maintenance, fleet robotics, search and rescue, and data collection for robot learning. Effective teleoperation requires intuitive 3D visualization with reliable depth cues, which conventional screen-based interfaces often fail to provide. We introduce a multi-view VR telepresence system that (1) fuses geometry from three cameras to produce GPU-accelerated point-cloud rendering on standalone VR hardware, and (2) integrates a wrist-mounted RGB stream to provide high-resolution local detail where point-cloud accuracy is limited. Our pipeline supports real-time rendering of approximately 75k points on the Meta Quest 3. A within-subject study was conducted with 31 participants to compare our system to other visualisation modalities, such as RGB streams, a projection of stereo-vision directly in the VR device and point clouds without providing additional RGB information. Across three different teleoperated manipulation tasks, we measured task success, completion time, perceived workload, and usability. Our system achieved the best overall performance, while the Point Cloud modality without RGB also outperforming the RGB streams and OpenTeleVision. These results show that combining global 3D structure with localized high-resolution detail substantially improves telepresence for manipulation and provides a strong foundation for next-generation robot teleoperation systems.
>
---
#### [new 009] G-EDF-Loc: 3D Continuous Gaussian Distance Field for Robust Gradient-Based 6DoF Localization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于6DoF定位任务，解决恶劣条件下定位精度问题。提出G-EDF-Loc框架，利用连续3D距离场实现高精度实时定位。**

- **链接: [https://arxiv.org/pdf/2604.04525](https://arxiv.org/pdf/2604.04525)**

> **作者:** José E. Maese; Lucía Coto-Elena; Luis Merino; Fernando Caballero
>
> **摘要:** This paper presents a robust 6-DoF localization framework based on a direct, CPU-based scan-to-map registration pipeline. The system leverages G-EDF, a novel continuous and memory-efficient 3D distance field representation. The approach models the Euclidean Distance Field (EDF) using a Block-Sparse Gaussian Mixture Model with adaptive spatial partitioning, ensuring $C^1$ continuity across block transitions and mitigating boundary artifacts. By leveraging the analytical gradients of this continuous map, which maintain Eikonal consistency, the proposed method achieves high-fidelity spatial reconstruction and real-time localization. Experimental results on large-scale datasets demonstrate that G-EDF-Loc performs competitively against state-of-the-art methods, exhibiting exceptional resilience even under severe odometry degradation or in the complete absence of IMU priors.
>
---
#### [new 010] Towards Edge Intelligence via Autonomous Navigation: A Robot-Assisted Data Collection Approach
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于边缘智能任务，旨在解决机器人在复杂环境中数据收集的效率与可靠性问题。提出一种双驱动导航方案，结合通信与学习优化，提升导航、数据收集和模型训练性能。**

- **链接: [https://arxiv.org/pdf/2604.03623](https://arxiv.org/pdf/2604.03623)**

> **作者:** Tingting Huang; Yingyang Chen; Sixian Qin; Zhijian Lin; Jun Li; Li Wang
>
> **备注:** 6 pages, 9 figures, submitted to IEEE International Conference on Communications (ICC) 2026
>
> **摘要:** With the growing demand for large-scale and high-quality data in edge intelligence systems, mobile robots are increasingly deployed to collect data proactively, particularly in complex environments. However, existing robot-assisted data collection methods face significant challenges in achieving reliable and efficient performance, especially in non-line-of-sight (NLoS) environments. This paper proposes a communication-and-learning dual-driven (CLD) autonomous navigation scheme that incorporates region-aware propagation characteristics and a non-point-mass robot representation. This scheme enables simultaneous optimization of navigation, communication, and learning performance. An efficient algorithm based on majorization-minimization (MM) is proposed to solve the non-convex and non-smooth CLD problem. Simulation results demonstrate that the proposed scheme achieves superior performance in collision-avoidance navigation, data collection, and model training compared to benchmark methods. It is also shown that CLD can adapt to different scenarios by flexibly adjusting the weight factor among navigation, communication and learning objectives.
>
---
#### [new 011] OpenRC: An Open-Source Robotic Colonoscopy Framework for Multimodal Data Acquisition and Autonomy Research
- **分类: cs.RO**

- **简介: 该论文提出OpenRC框架，解决机器人结肠镜研究中数据获取与自主性不足的问题。通过多模态数据采集和开放平台，支持结肠镜导航与控制研究。**

- **链接: [https://arxiv.org/pdf/2604.03781](https://arxiv.org/pdf/2604.03781)**

> **作者:** Siddhartha Kapuria; Mohammad Rafiee Javazm; Naruhiko Ikoma; Joga Ivatury; Mohammad Ali Nasseri; Nassir Navab; Farshid Alambeigi
>
> **摘要:** Colorectal cancer screening critically depends on colonoscopy, yet existing platforms offer limited support for systematically studying the coupled dynamics of operator control, instrument motion, and visual feedback. This gap restricts reproducible closed-loop research in robotic colonoscopy, medical imaging, and emerging vision-language-action (VLA) learning paradigms. To address this challenge, we present OpenRC, an open-source modular robotic colonoscopy framework that retrofits conventional scopes while preserving clinical workflow. The framework supports simultaneous recording of video, operator commands, actuation state, and distal tip pose. We experimentally validated motion consistency and quantified cross-modal latency across sensing streams. Using this platform, we collected a multimodal dataset comprising 1,894 teleoperated episodes ~19 hours across 10 structured task variations of routine navigation, failure events, and recovery behaviors. By unifying open hardware and an aligned multimodal dataset, OpenRC provides a reproducible foundation for research in multimodal robotic colonoscopy and surgical autonomy.
>
---
#### [new 012] A Novel Hybrid PID-LQR Controller for Sit-To-Stand Assistance Using a CAD-Integrated Simscape Multibody Lower Limb Exoskeleton
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于康复机器人控制任务，旨在解决下肢外骨骼在坐-站转换中的精确控制问题。通过设计并比较PID、LQR及混合控制器，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2604.03766](https://arxiv.org/pdf/2604.03766)**

> **作者:** Ranjeet Kumbhar; Rajmeet Singh; Appaso M Gadade; Ashish Singla; Irfan Hussain
>
> **摘要:** Precise control of lower limb exoskeletons during sit-to-stand (STS) transitions remains a central challenge in rehabilitation robotics owing to the highly nonlinear, time-varying dynamics of the human-exoskeleton system and the stringent trajectory tracking requirements imposed by clinical safety. This paper presents the systematic design, simulation, and comparative evaluation of three control strategies: a classical Proportional-Integral-Derivative (PID) controller, a Linear Quadratic Regulator (LQR), and a novel Hybrid PID-LQR controller applied to a bilateral lower limb exoskeleton performing the sit-to-stand transition. A high-fidelity, physics-based dynamic model of the exoskeleton is constructed by importing a SolidWorks CAD assembly directly into the MATLAB/Simulink Simscape Multibody environment, preserving accurate geometric and inertial properties of all links. Physiologically representative reference joint trajectories for the hip, knee, and ankle joints are generated using OpenSim musculoskeletal simulation and decomposed into three biomechanical phases: flexion-momentum (0-33%), momentum-transfer (34-66%), and extension (67-100%). The proposed Hybrid PID-LQR controller combines the optimal transient response of LQR with the integral disturbance rejection of PID through a tuned blending coefficient alpha = 0.65. Simulation results demonstrate that the Hybrid PID-LQR achieves RMSE reductions of 72.3% and 70.4% over PID at the hip and knee joints, respectively, reduces settling time by over 90% relative to PID across all joints, and limits overshoot to 2.39%-6.10%, confirming its superiority over both baseline strategies across all evaluated performance metrics and demonstrating strong translational potential for clinical assistive exoskeleton deployment.
>
---
#### [new 013] Do Robots Need Body Language? Comparing Communication Modalities for Legible Motion Intent in Human-Shared Spaces
- **分类: cs.RO; cs.CY; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决机器人在共享空间中动作意图不明确的问题。通过实验比较不同通信方式对人类理解机器人行为的影响。**

- **链接: [https://arxiv.org/pdf/2604.03451](https://arxiv.org/pdf/2604.03451)**

> **作者:** Jonathan Albert Cohen; Kye Shimizu; Allen Song; Vishnu Bharath; Kent Larson; Pattie Maes
>
> **摘要:** Robots in shared spaces often move in ways that are difficult for people to interpret, placing the burden on humans to adapt. High-DoF robots exhibit motion that people read as expressive, intentionally or not, making it important to understand how such cues are perceived. We present an online video study evaluating how different signaling modalities, expressive motion, lights, text, and audio, shape people's ability to understand a quadruped robot's upcoming navigation actions (Boston Dynamics Spot). Across four common scenarios, we measure how each modality influences humans' (1) accuracy in predicting the robot's next navigation action, (2) confidence in that prediction, and (3) trust in the robot to act safely. The study tests how expressive motions compare to explicit channels, whether aligned multimodal cues enhance interpretability, and how conflicting cues affect user confidence and trust. We contribute initial evidence on the relative effectiveness of implicit versus explicit signaling strategies.
>
---
#### [new 014] Outlier-Robust Nonlinear Moving Horizon Estimation using Adaptive Loss Functions
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，旨在解决异常值对MHE的影响问题。通过引入自适应鲁棒损失函数和正则化项，提升估计的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.04862](https://arxiv.org/pdf/2604.04862)**

> **作者:** Nestor Deniz; Guido Sanchez; Fernando Auat Cheein; Leonardo Giovanini
>
> **摘要:** In this work, we propose an adaptive robust loss function framework for MHE, integrating an adaptive robust loss function to reduce the impact of outliers with a regularization term that avoids naive solutions. The proposed approach prioritizes the fitting of uncontaminated data and downweights the contaminated ones. A tuning parameter is incorporated into the framework to control the shape of the loss function for adjusting the estimator's robustness to outliers. The simulation results demonstrate that adaptation occurs in just a few iterations, whereas the traditional behaviour $\mathrm{L_2}$ predominates when the measurements are free of outliers.
>
---
#### [new 015] Efficient Multi-Objective Planning with Weighted Maximization Using Large Neighbourhood Search
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决多目标优化问题。针对加权和方法的局限性，提出基于大邻域搜索的高效加权最大算法，提升计算效率并保证解的质量。**

- **链接: [https://arxiv.org/pdf/2604.04826](https://arxiv.org/pdf/2604.04826)**

> **作者:** Krishna Kalavadia; Shamak Dutta; Yash Vardhan Pant; Stephen L. Smith
>
> **摘要:** Autonomous navigation often requires the simultaneous optimization of multiple objectives. The most common approach scalarizes these into a single cost function using a weighted sum, but this method is unable to find all possible trade-offs and can therefore miss critical solutions. An alternative, the weighted maximum of objectives, can find all Pareto-optimal solutions, including those in non-convex regions of the trade-off space that weighted sum methods cannot find. However, the increased computational complexity of finding weighted maximum solutions in the discrete domain has limited its practical use. To address this challenge, we propose a novel search algorithm based on the Large Neighbourhood Search framework that efficiently solves the weighted maximum planning problem. Through extensive simulations, we demonstrate that our algorithm achieves comparable solution quality to existing weighted maximum planners with a runtime improvement of 1-2 orders of magnitude, making it a viable option for autonomous navigation.
>
---
#### [new 016] Sim2Real-AD: A Modular Sim-to-Real Framework for Deploying VLM-Guided Reinforcement Learning in Real-World Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自主驾驶任务，解决模拟到现实的强化学习部署问题。提出Sim2Real-AD框架，实现无需真实数据的零样本迁移。**

- **链接: [https://arxiv.org/pdf/2604.03497](https://arxiv.org/pdf/2604.03497)**

> **作者:** Zilin Huang; Zhengyang Wan; Zihao Sheng; Boyue Wang; Junwei You; Yue Leng; Sikai Chen
>
> **备注:** 36 pages, 21 figures
>
> **摘要:** Deploying reinforcement learning policies trained in simulation to real autonomous vehicles remains a fundamental challenge, particularly for VLM-guided RL frameworks whose policies are typically learned with simulator-native observations and simulator-coupled action semantics that are unavailable on physical platforms. This paper presents Sim2Real-AD, a modular framework for zero-shot sim-to-real transfer of CARLA-trained VLM-guided RL policies to full-scale vehicles without any real-world RL training data. The framework decomposes the transfer problem into four components: a Geometric Observation Bridge (GOB) that converts monocular front-view images into simulator-compatible bird's-eye-view (BEV) observations, a Physics-Aware Action Mapping (PAM) that translates policy outputs into platform-agnostic physical commands, a Two-Phase Progressive Training (TPT) strategy that stabilizes adaptation by separating action-space and observation-space transfer, and a Real-time Deployment Pipeline (RDP) that integrates perception, policy inference, control conversion, and safety monitoring for closed-loop execution. Simulation experiments show that the framework preserves the relative performance ordering of representative RL algorithms across different reward paradigms and validate the contribution of each module. Zero-shot deployment on a full-scale Ford E-Transit achieves success rates of 90%, 80%, and 75% in car-following, obstacle avoidance, and stop-sign interaction scenarios, respectively. To the best of our knowledge, this study is among the first to demonstrate zero-shot closed-loop deployment of a CARLA-trained VLM-guided RL policy on a full-scale real vehicle without any real-world RL training data. The demo video and code are available at: this https URL.
>
---
#### [new 017] Real-Time Projected Adaptive Control for Closed-Chain Co-Manipulative Continuum Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决CCRs中未知柔性物体参数带来的控制问题。通过自适应控制框架和GVS模型实现精确轨迹跟踪。**

- **链接: [https://arxiv.org/pdf/2604.04286](https://arxiv.org/pdf/2604.04286)**

> **作者:** Rana Danesh; Farrokh Janabi-Sharifi; Farhad Aghili
>
> **摘要:** In co-manipulative continuum robots (CCRs), multiple continuum arms cooperate by grasping a common flexible object, forming a closed-chain deformable mechanical system. The closed-chain coupling induces strong dynamic interactions and internal reaction forces. Moreover, in practical tasks, the flexible object's physical parameters are often unknown and vary between operations, rendering nominal model-based controllers inadequate. This paper presents a projected adaptive control framework for CCRs formulated at the dynamic level. The coupled dynamics are expressed using the Geometric Variable Strain (GVS) representation, yielding a finite-dimensional model that accurately represents the system, preserves the linear-in-parameters structure required for adaptive control, and is suitable for real-time implementation. Closed-chain interactions are enforced through Pfaffian velocity constraints, and an orthogonal projection is used to express the dynamics in the constraint-consistent motion subspace. Based on the projected dynamics, an adaptive control law is developed to compensate online for uncertain dynamic parameters of both the continuum robots and the manipulated flexible object. Lyapunov analysis establishes closed-loop stability and convergence of the task-space tracking errors to zero. Simulation and experiments on a tendon-driven CCR platform validate the proposed framework in task-space regulation and trajectory tracking.
>
---
#### [new 018] HAD: Combining Hierarchical Diffusion with Metric-Decoupled RL for End-to-End Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决端到端路径规划问题。针对轨迹选择困难和强化学习优化不足，提出HAD框架，结合分层扩散与解耦奖励优化，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2604.03581](https://arxiv.org/pdf/2604.03581)**

> **作者:** Wenhao Yao; Xinglong Sun; Zhenxin Li; Shiyi Lan; Zi Wang; Jose M. Alvarez; Zuxuan Wu
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** End-to-end planning has emerged as a dominant paradigm for autonomous driving, where recent models often adopt a scoring-selection framework to choose trajectories from a large set of candidates, with diffusion-based decoding showing strong promise. However, directly selecting from the entire candidate space remains difficult to optimize, and Gaussian perturbations used in diffusion often introduce unrealistic trajectories that complicate the denoising process. In addition, for training these models, reinforcement learning (RL) has shown promise, but existing end-to-end RL approaches typically rely on a single coupled reward without structured signals, limiting optimization effectiveness. To address these challenges, we propose HAD, an end-to-end planning framework with a Hierarchical Diffusion Policy that decomposes planning into a coarse-to-fine process. To improve trajectory generation, we introduce Structure-Preserved Trajectory Expansion, which produces realistic candidates while maintaining kinematic structure. For policy learning, we develop Metric-Decoupled Policy Optimization (MDPO) to enable structured RL optimization across multiple driving objectives. Extensive experiments show that HAD achieves new state-of-the-art performance on both NAVSIM and HUGSIM, outperforming prior arts by a huge margin: +2.3 EPDMS on NAVSIM and +4.9 Route Completion on HUGSIM.
>
---
#### [new 019] DC-Ada: Reward-Only Decentralized Observation-Interface Adaptation for Heterogeneous Multi-Robot Teams
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出DC-Ada方法，解决异构多机器人团队在传感器差异下的性能下降问题，通过适应观测接口提升任务表现。**

- **链接: [https://arxiv.org/pdf/2604.03905](https://arxiv.org/pdf/2604.03905)**

> **作者:** Saad Alqithami
>
> **摘要:** Heterogeneity is a defining feature of deployed multi-robot teams: platforms often differ in sensing modalities, ranges, fields of view, and failure patterns. Controllers trained under nominal sensing can degrade sharply when deployed on robots with missing or mismatched sensors, even when the task and action interface are unchanged. We present DC-Ada, a reward-only decentralized adaptation method that keeps a pretrained shared policy frozen and instead adapts compact per-robot observation transforms to map heterogeneous sensing into a fixed inference interface. DC-Ada is gradient-free and communication-minimal: it uses budgeted accept/reject random search with short common-random-number rollouts under a strict step budget. We evaluate DC-Ada against four baselines in a deterministic 2D multi-robot simulator covering warehouse logistics, search and rescue, and collaborative mapping, across four heterogeneity regimes (H0--H3) and five seeds with a matched budget of $200{,}000$ joint environment steps per run. Results show that heterogeneity can substantially degrade a frozen shared policy and that no single mitigation dominates across all tasks and metrics. Observation normalization is strongest for reward robustness in warehouse logistics and competitive in search and rescue, while the frozen shared policy is strongest for reward in collaborative mapping. DC-Ada offers a useful complementary operating point: it improves completion most clearly in severe coverage-based mapping while requiring only scalar team returns and no policy fine-tuning or persistent communication. These results position DC-Ada as a practical deploy-time adaptation method for heterogeneous teams.
>
---
#### [new 020] Pickalo: Leveraging 6D Pose Estimation for Low-Cost Industrial Bin Picking
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于工业抓取任务，解决复杂环境下低成本6D位姿抓取问题。采用低成本硬件和深度学习方法，实现高效精准的物体定位与抓取。**

- **链接: [https://arxiv.org/pdf/2604.04690](https://arxiv.org/pdf/2604.04690)**

> **作者:** Alessandro Tarsi; Matteo Mastrogiuseppe; Saverio Taliani; Simone Cortinovis; Ugo Pattacini
>
> **摘要:** Bin picking in real industrial environments remains challenging due to severe clutter, occlusions, and the high cost of traditional 3D sensing setups. We present Pickalo, a modular 6D pose-based bin-picking pipeline built entirely on low-cost hardware. A wrist-mounted RGB-D camera actively explores the scene from multiple viewpoints, while raw stereo streams are processed with BridgeDepth to obtain refined depth maps suitable for accurate collision reasoning. Object instances are segmented with a Mask-RCNN model trained purely on photorealistic synthetic data and localized using the zero-shot SAM-6D pose estimator. A pose buffer module fuses multi-view observations over time, handling object symmetries and significantly reducing pose noise. Offline, we generate and curate large sets of antipodal grasp candidates per object; online, a utility-based ranking and fast collision checking are queried for the grasp planning. Deployed on a UR5e with a parallel-jaw gripper and an Intel RealSense D435i, Pickalo achieves up to 600 mean picks per hour with 96-99% grasp success and robust performance over 30-minute runs on densely filled euroboxes. Ablation studies demonstrate the benefits of enhanced depth estimation and of the pose buffer for long-term stability and throughput in realistic industrial conditions. Videos are available at this https URL
>
---
#### [new 021] ROSClaw: A Hierarchical Semantic-Physical Framework for Heterogeneous Multi-Agent Collaboration
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出ROSClaw框架，解决多智能体协作中语义与物理执行脱节的问题。通过统一视觉语言模型控制器，实现任务执行与策略学习的整合，提升多策略鲁棒性与跨平台迁移能力。**

- **链接: [https://arxiv.org/pdf/2604.04664](https://arxiv.org/pdf/2604.04664)**

> **作者:** Rongfeng Zhao; Xuanhao Zhang; Zhaochen Guo; Xiang Shao; Zhongpan Zhu; Bin He; Jie Chen
>
> **摘要:** The integration of large language models (LLMs) with embodied agents has improved high-level reasoning capabilities; however, a critical gap remains between semantic understanding and physical execution. While vision-language-action (VLA) and vision-language-navigation (VLN) systems enable robots to perform manipulation and navigation tasks from natural language instructions, they still struggle with long-horizon sequential and temporally structured tasks. Existing frameworks typically adopt modular pipelines for data collection, skill training, and policy deployment, resulting in high costs in experimental validation and policy optimization. To address these limitations, we propose ROSClaw, an agent framework for heterogeneous robots that integrates policy learning and task execution within a unified vision-language model (VLM) controller. The framework leverages e-URDF representations of heterogeneous robots as physical constraints to construct a sim-to-real topological mapping, enabling real-time access to the physical states of both simulated and real-world agents. We further incorporate a data collection and state accumulation mechanism that stores robot states, multimodal observations, and execution trajectories during real-world execution, enabling subsequent iterative policy optimization. During deployment, a unified agent maintains semantic continuity between reasoning and execution, and dynamically assigns task-specific control to different agents, thereby improving robustness in multi-policy execution. By establishing an autonomous closed-loop framework, ROSClaw minimizes the reliance on robot-specific development workflows. The framework supports hardware-level validation, automated generation of SDK-level control programs, and tool-based execution, enabling rapid cross-platform transfer and continual improvement of robotic skills. Ours project page: this https URL.
>
---
#### [new 022] ReinVBC: A Model-based Reinforcement Learning Approach to Vehicle Braking Controller
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于车辆控制任务，旨在解决传统刹车系统依赖人工校准的问题。通过模型强化学习方法，设计可靠的动力学模型和刹车策略，提升刹车性能。**

- **链接: [https://arxiv.org/pdf/2604.04401](https://arxiv.org/pdf/2604.04401)**

> **作者:** Haoxin Lin; Junjie Zhou; Daheng Xu; Yang Yu
>
> **摘要:** Braking system, the key module to ensure the safety and steer-ability of current vehicles, relies on extensive manual calibration during production. Reducing labor and time consumption while maintaining the Vehicle Braking Controller (VBC) performance greatly benefits the vehicle industry. Model-based methods in offline reinforcement learning, which facilitate policy exploration within a data-driven dynamics model, offer a promising solution for addressing real-world control tasks. This work proposes ReinVBC, which applies an offline model-based reinforcement learning approach to deal with the vehicle braking control problem. We introduce useful engineering designs into the paradigm of model learning and utilization to obtain a reliable vehicle dynamics model and a capable braking policy. Several results demonstrate the capability of our method in real-world vehicle braking and its potential to replace the production-grade anti-lock braking system.
>
---
#### [new 023] RK-MPC: Residual Koopman Model Predictive Control for Quadruped Locomotion in Offroad Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决四足机器人在非结构化环境中运动预测与控制问题。提出RK-MPC框架，结合Koopman模型与数据驱动方法，提升预测精度与实时性。**

- **链接: [https://arxiv.org/pdf/2604.04221](https://arxiv.org/pdf/2604.04221)**

> **作者:** Sriram S. K. S. Narayanan; Umesh Vaidya
>
> **摘要:** This paper presents Residual Koopman MPC (RK-MPC), a Koopman-based, data-driven model predictive control framework for quadruped locomotion that improves prediction fidelity while preserving real-time tractability. RK-MPC augments a nominal template model with a compact linear residual predictor learned from data in lifted coordinates, enabling systematic correction of model mismatch induced by contact variability and terrain disturbances with provable bounds on multi-step prediction error. The learned residual model is embedded within a convex quadratic-program MPC formulation, yielding a receding-horizon controller that runs onboard at 500 Hz and retains the structure and constraint-handling advantages of optimization-based control. We evaluate RK-MPC in both Gazebo simulation and Unitree Go1 hardware experiments, demonstrating reliable blind locomotion across contact disturbances, multiple gait schedules, and challenging off-road terrains including grass, gravel, snow, and ice. We further compare against Koopman/EDMD baselines using alternative observable dictionaries, including monomial and $SE(3)$-structured bases, and show that the residual correction improves multi-step prediction and closed-loop performance while reducing sensitivity to the choice of observables. Overall, RK-MPC provides a practical, hardware-validated pathway for data-driven predictive control of quadrupeds in unstructured environments. See this https URL for implementation videos.
>
---
#### [new 024] WaterSplat-SLAM: Photorealistic Monocular SLAM in Underwater Environment
- **分类: cs.RO**

- **简介: 该论文属于 underwater SLAM 任务，旨在解决水下单目SLAM中地图渲染质量低的问题。提出WaterSplat-SLAM系统，结合语义过滤与高斯映射，实现精准定位与逼真建图。**

- **链接: [https://arxiv.org/pdf/2604.04642](https://arxiv.org/pdf/2604.04642)**

> **作者:** Kangxu Wang; Shaofeng Zou; Chenxing Jiang; Yixiang Dai; Siang Chen; Shaojie Shen; Guijin Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Underwater monocular SLAM is a challenging problem with applications from autonomous underwater vehicles to marine archaeology. However, existing underwater SLAM methods struggle to produce maps with high-fidelity rendering. In this paper, we propose WaterSplat-SLAM, a novel monocular underwater SLAM system that achieves robust pose estimation and photorealistic dense mapping. Specifically, we couple semantic medium filtering into two-view 3D reconstruction prior to enable underwater-adapted camera tracking and depth estimation. Furthermore, we present a semantic-guided rendering and adaptive map management strategy with an online medium-aware Gaussian map, modeling underwater environment in a photorealistic and compact manner. Experiments on multiple underwater datasets demonstrate that WaterSplat-SLAM achieves robust camera tracking and high-fidelity rendering in underwater environments.
>
---
#### [new 025] VA-FastNavi-MARL: Real-Time Robot Control with Multimedia-Driven Meta-Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，解决动态多媒体指令的实时理解与机器人控制问题。通过多模态对齐和元强化学习，实现快速适应与低延迟控制。**

- **链接: [https://arxiv.org/pdf/2604.03998](https://arxiv.org/pdf/2604.03998)**

> **作者:** Yang Zhang; Shengxi Jing; Fengxiang Wang; Yuan Feng; Hong Wang
>
> **备注:** Accepted to the 2026 IEEE International Conference on Multimedia and Expo (ICME 2026)
>
> **摘要:** Interpreting dynamic, heterogeneous multimedia commands with real-time responsiveness is critical for Human-Robot Interaction. We present VA-FastNavi-MARL, a framework that aligns asynchronous audio-visual inputs into a unified latent representation. By treating diverse instructions as a distribution of navigable goals via Meta-Reinforcement Learning, our method enables rapid adaptation to unseen directives with negligible inference overhead. Unlike approaches bottlenecked by heavy sensory processing, our modality-agnostic stream ensures seamless, low-latency control. Validation on a multi-arm workspace confirms that VA-FastNavi-MARL significantly outperforms baselines in sample efficiency and maintains robust, real-time execution even under noisy multimedia streams.
>
---
#### [new 026] Build on Priors: Vision--Language--Guided Neuro-Symbolic Imitation Learning for Data-Efficient Real-World Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，解决数据效率和泛化能力问题。通过结合视觉-语言模型与符号规划，自动构建控制策略，实现少样本、无需人工干预的机器人学习。**

- **链接: [https://arxiv.org/pdf/2604.03759](https://arxiv.org/pdf/2604.03759)**

> **作者:** Pierrick Lorang; Johannes Huemer; Timothy Duggan; Kai Goebel; Patrik Zips; Matthias Scheutz
>
> **摘要:** Enabling robots to learn long-horizon manipulation tasks from a handful of demonstrations remains a central challenge in robotics. Existing neuro-symbolic approaches often rely on hand-crafted symbolic abstractions, semantically labeled trajectories or large demonstration datasets, limiting their scalability and real-world applicability. We present a scalable neuro-symbolic framework that autonomously constructs symbolic planning domains and data-efficient control policies from as few as one to thirty unannotated skill demonstrations, without requiring manual domain engineering. Our method segments demonstrations into skills and employs a Vision-Language Model (VLM) to classify skills and identify equivalent high-level states, enabling automatic construction of a state-transition graph. This graph is processed by an Answer Set Programming solver to synthesize a PDDL planning domain, which an oracle function exploits to isolate the minimal, task-relevant and target relative observation and action spaces for each skill policy. Policies are learned at the control reference level rather than at the raw actuator signal level, yielding a smoother and less noisy learning target. Known controllers can be leveraged for real-world data augmentation by projecting a single demonstration onto other objects in the scene, simultaneously enriching the graph construction process and the dataset for imitation learning. We validate our framework primarily on a real industrial forklift across statistically rigorous manipulation trials, and demonstrate cross-platform generality on a Kinova Gen3 robotic arm across two standard benchmarks. Our results show that grounding control learning, VLM-driven abstraction, and automated planning synthesis into a unified pipeline constitutes a practical path toward scalable, data-efficient, expert-free and interpretable neuro-symbolic robotics.
>
---
#### [new 027] Human-Robot Copilot for Data-Efficient Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决有限示范下策略泛化性差的问题。提出Human-Robot Copilot框架，提升数据效率与控制精度。**

- **链接: [https://arxiv.org/pdf/2604.03613](https://arxiv.org/pdf/2604.03613)**

> **作者:** Rui Yan; Zaitian Gongye; Lars Paulsen; Xuxin Cheng; Xiaolong Wang
>
> **摘要:** Collecting human demonstrations via teleoperation is a common approach for teaching robots task-specific skills. However, when only a limited number of demonstrations are available, policies are prone to entering out-of-distribution (OOD) states due to compounding errors or environmental stochasticity. Existing interactive imitation learning or human-in-the-loop methods try to address this issue by following the Human-Gated DAgger (HG-DAgger) paradigm, an approach that augments demonstrations through selective human intervention during policy execution. Nevertheless, these approaches struggle to balance dexterity and generality: they either provide fine-grained corrections but are limited to specific kinematic structures, or achieve generality at the cost of precise control. To overcome this limitation, we propose the Human-Robot Copilot framework that can leverage a scaling factor for dexterous teleoperation while maintaining compatibility with a wide range of industrial and research manipulators. Experimental results demonstrate that our framework achieves higher performance with the same number of demonstration trajectories. Moreover, since corrective interventions are required only intermittently, the overall data collection process is more efficient and less time-consuming.
>
---
#### [new 028] Towards Considerate Human-Robot Coexistence: A Dual-Space Framework of Robot Design and Human Perception in Healthcare
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 论文探讨医疗机器人与人类共存的动态过程，提出双空间框架以理解人类感知与机器人设计的相互作用。任务是改善人机协作，解决静态态度研究不足的问题，通过访谈分析人类在共存中的主动角色。**

- **链接: [https://arxiv.org/pdf/2604.04374](https://arxiv.org/pdf/2604.04374)**

> **作者:** Yuanchen Bai; Zijian Ding; Ruixiang Han; Niti Parikh; Wendy Ju; Angelique Taylor
>
> **摘要:** The rapid advancement of robotics, spanning expanded capabilities, more intuitive interaction, and more integration into real-world workflows, is reshaping what it means for humans and robots to coexist. Beyond sharing physical space, this coexistence is increasingly characterized by organizational embeddedness, temporal evolution, social situatedness, and open-ended uncertainty. However, prior work has largely focused on static snapshots of attitudes and acceptance, offering limited insight into how perceptions form and evolve, and what active role humans play in shaping coexistence as a dynamic process. We address these gaps through in-depth follow-up interviews with nine participants from a 14-week co-design study on healthcare robots. We identify the human perception space, including four interpretive dimensions (i.e., degree of decomposition, temporal orientation, scope of reasoning, and source of evidence). We enrich the conceptual framework of human-robot coexistence by conceptualizing the mutual relationship between the human perception space and the robot design space as a co-evolving loop, in which human needs, design decisions, situated interpretations, and social mediation continuously reshape one another over time. Building on this, we propose considerate human-robot coexistence, arguing that humans act not only as design contributors but also as interpreters and mediators who actively shape how robots are understood and integrated across deployment stages.
>
---
#### [new 029] Adapting Neural Robot Dynamics on the Fly for Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决动态模型不准确的问题。通过结合离线训练与在线更新，实现神经动力模型的快速适应，提升预测控制性能。**

- **链接: [https://arxiv.org/pdf/2604.04039](https://arxiv.org/pdf/2604.04039)**

> **作者:** Abdullah Altawaitan; Nikolay Atanasov
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Accurate dynamics models are critical for the design of predictive controller for autonomous mobile robots. Physics-based models are often too simple to capture relevant real-world effects, while data-driven models are data-intensive and slow to train. We introduce an approach for fast adaptation of neural robot dynamic models that combines offline training with efficient online updates. Our approach learns an incremental neural dynamics model offline and performs low-rank second-order parameter adaptation online, enabling rapid updates without full retraining. We demonstrate the approach on a real quadrotor robot, achieving robust predictive tracking control in novel operational conditions.
>
---
#### [new 030] Biologically Inspired Event-Based Perception and Sample-Efficient Learning for High-Speed Table Tennis Robots
- **分类: cs.RO**

- **简介: 该论文属于高精度乒乓球机器人任务，解决高速场景下感知与决策难题。提出基于事件的感知方法和高效学习策略，提升检测精度与训练效率。**

- **链接: [https://arxiv.org/pdf/2604.04618](https://arxiv.org/pdf/2604.04618)**

> **作者:** Ziqi Wang; Jingyue Zhao; Xun Xiao; Jichao Yang; Yaohua Wang; Shi Xu; Lei Wang; Huadong Dai
>
> **摘要:** Perception and decision-making in high-speed dynamic scenarios remain challenging for current robots. In contrast, humans and animals can rapidly perceive and make decisions in such environments. Taking table tennis as a typical example, conventional frame-based vision sensors suffer from motion blur, high latency and data redundancy, which can hardly meet real-time, accurate perception requirements. Inspired by the human visual system, event-based perception methods address these limitations through asynchronous sensing, high temporal resolution, and inherently sparse data representations. However, current event-based methods are still restricted to simplified, unrealistic ball-only scenarios. Meanwhile, existing decision-making approaches typically require thousands of interactions with the environment to converge, resulting in significant computational costs. In this work, we present a biologically inspired approach for high-speed table tennis robots, combining event-based perception with sample-efficient learning. On the perception side, we propose an event-based ball detection method that leverages motion cues and geometric consistency, operating directly on asynchronous event streams without frame reconstruction, to achieve robust and efficient detection in real-world rallies. On the decision-making side, we introduce a human-inspired, sample-efficient training strategy that first trains policies in low-speed scenarios, progressively acquiring skills from basic to advanced, and then adapts them to high-speed scenarios, guided by a case-dependent temporally adaptive reward and a reward-threshold mechanism. With the same training episodes, our method improves return-to-target accuracy by 35.8%. These results demonstrate the effectiveness of biologically inspired perception and decision-making for high-speed robotic systems.
>
---
#### [new 031] Activity-Dependent Plasticity in Morphogenetically-Grown Recurrent Networks
- **分类: cs.RO**

- **简介: 该论文研究发育型神经网络的可塑性，解决如何通过进化赋予网络自适应能力的问题。工作包括分析Hebbian与anti-Hebbian塑料性的效果，并通过共进化实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2604.03386](https://arxiv.org/pdf/2604.03386)**

> **作者:** Sergii Medvid; Andrii Valenia; Mykola Glybovets
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Developmental approaches to neural architecture search grow functional networks from compact genomes through self-organisation, but the resulting networks operate with fixed post-growth weights. We characterise Hebbian and anti-Hebbian plasticity across 50,000 morphogenetically grown recurrent controllers (5M+ configurations on CartPole and Acrobot), then test whether co-evolutionary experiments -- where plasticity parameters are encoded in the genome and evolved alongside the developmental architecture -- recover these patterns independently. Our characterisation reveals that (1) anti-Hebbian plasticity significantly outperforms Hebbian for competent networks (Cohen's d = 0.53-0.64), (2) regret (fraction of oracle improvement lost under the best fixed setting) reaches 52-100%, and (3) plasticity's role shifts from fine-tuning to genuine adaptation under non-stationarity. Co-evolution independently discovers these patterns: on CartPole, 70% of runs evolve anti-Hebbian plasticity (p = 0.043); on Acrobot, evolution finds near-zero eta with mixed signs -- exactly matching the characterisation. A random-RNN control shows that anti-Hebbian dominance is generic to small recurrent networks, but the degree of topology-dependence is developmental-specific: regret is 2-6x higher for morphogenetically grown networks than for random graphs with matched topology statistics.
>
---
#### [new 032] frax: Fast Robot Kinematics and Dynamics in JAX
- **分类: cs.RO**

- **简介: 该论文提出frax，一个基于JAX的机器人运动学与动力学库，解决多架构下高性能计算问题，支持CPU、GPU和TPU，提升控制与优化效率。**

- **链接: [https://arxiv.org/pdf/2604.04310](https://arxiv.org/pdf/2604.04310)**

> **作者:** Daniel Morton; Marco Pavone
>
> **备注:** Submitted to the ICRA 2026 Workshop on Frontiers of Optimization for Robotics
>
> **摘要:** In robot control, planning, and learning, there is a need for rigid-body dynamics libraries that are highly performant, easy to use, and compatible with CPUs and accelerators. While existing libraries often excel at either low-latency CPU execution or high-throughput GPU workloads, few provide a unified framework that targets multiple architectures without compromising performance or ease-of-use. To address this, we introduce frax, a JAX-based library for robot kinematics and dynamics, providing a high-performance, pure-Python interface across CPU, GPU, and TPU. Via a fully-vectorized approach to robot dynamics, frax enables efficient real-time control and parallelization, while supporting automatic differentiation for optimization-based methods. On CPU, frax achieves low-microsecond computation times suitable for kilohertz control rates, outperforming common libraries in Python and approaching optimized C++ implementations. On GPU, the same code scales to thousands of instances, reaching upwards of 100 million dynamics evaluations per second. We validate performance on a Franka Panda manipulator and a Unitree G1 humanoid, and release frax as an open-source library.
>
---
#### [new 033] Dynamic Whole-Body Dancing with Humanoid Robots -- A Model-Based Control Approach
- **分类: cs.RO**

- **简介: 该论文研究人形机器人动态全身舞蹈控制，解决真实环境下生成与执行复杂舞蹈动作的问题。通过模型预测控制和轨迹优化实现稳定、生动的舞蹈表演。**

- **链接: [https://arxiv.org/pdf/2604.03999](https://arxiv.org/pdf/2604.03999)**

> **作者:** Shibowen Zhang; Jiayang Wu; Guannan Liu; Helin Zhu; Junjie Liu; Zhehan Li; Junhong Guo; Xiaokun Leng; Hangxin Liu; Jingwen Zhang; Jikai Wang; Zonghai Chen; Zhicheng He; Jiayi Wang; Yao Su
>
> **摘要:** This paper presents an integrated model-based framework for generating and executing dynamic whole-body dance motions on humanoid robots. The framework operates in two stages: offline motion generation and online motion execution, both leveraging future state prediction to enable robust and dynamic dance motions in real-world environments. In the offline motion generation stage, human dance demonstrations are captured via a motion capture (MoCap) system, retargeted to the robot by solving a Quadratic Programming (QP) problem, and further refined using Trajectory Optimization (TO) to ensure dynamic feasibility. In the online motion execution stage, a centroidal dynamics-based Model Predictive Control (MPC) framework tracks the planned motions in real time and proactively adjusts swing foot placement to adapt to real world disturbances. We validate our framework on the full-size humanoid robot Kuavo 4Pro, demonstrating the dynamic dance motions both in simulation and in a four-minute live public performance with a team of four robots. Experimental results show that longer prediction horizons improve both motion expressiveness in planning and stability in execution.
>
---
#### [new 034] Precise Robot Command Understanding Using Grammar-Constrained Large Language Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于人机协作任务，解决工业场景中机器人指令理解的精确性问题。通过结合语法约束与大语言模型，提升指令的结构化与可执行性。**

- **链接: [https://arxiv.org/pdf/2604.04233](https://arxiv.org/pdf/2604.04233)**

> **作者:** Xinyun Huo; Raghav Gnanasambandam; Xinyao Zhang
>
> **备注:** Accepted at ASME MSEC2026
>
> **摘要:** Human-robot collaboration in industrial settings requires precise and reliable communication to enhance operational efficiency. While Large Language Models (LLMs) understand general language, they often lack the domain-specific rigidity needed for safe and executable industrial commands. To address this gap, this paper introduces a novel grammar-constrained LLM that integrates a grammar-driven Natural Language Understanding (NLU) system with a fine-tuned LLM, which enables both conversational flexibility and the deterministic precision required in robotics. Our method employs a two-stage process. First, a fine-tuned LLM performs high-level contextual reasoning and parameter inference on natural language inputs. Second, a Structured Language Model (SLM) and a grammar-based canonicalizer constrain the LLM's output, forcing it into a standardized symbolic format composed of valid action frames and command elements. This process guarantees that generated commands are valid and structured in a robot-readable JSON format. A key feature of the proposed model is a validation and feedback loop. A grammar parser validates the output against a predefined list of executable robotic actions. If a command is invalid, the system automatically generates corrective prompts and re-engages the LLM. This iterative self-correction mechanism allows the model to recover from initial interpretation errors to improve system robustness. We evaluate our grammar-constrained hybrid model against two baselines: a fine-tuned API-based LLM and a standalone grammar-driven NLU model. Using the Human Robot Interaction Corpus (HuRIC) dataset, we demonstrate that the hybrid approach achieves superior command validity, which promotes safer and more effective industrial human-robot collaboration.
>
---
#### [new 035] Learning Dexterous Grasping from Sparse Taxonomy Guidance
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机械臂灵巧抓取任务，解决传统方法依赖密集指令或缺乏可控性的问题。提出GRIT框架，通过稀疏分类指导学习抓取策略，提升泛化能力和实际可控性。**

- **链接: [https://arxiv.org/pdf/2604.04138](https://arxiv.org/pdf/2604.04138)**

> **作者:** Juhan Park; Taerim Yoon; Seungmin Kim; Joonggil Kim; Wontae Ye; Jeongeun Park; Yoonbyung Chai; Geonwoo Cho; Geunwoo Cho; Dohyeong Kim; Kyungjae Lee; Yongjae Kim; Sungjoon Choi
>
> **摘要:** Dexterous manipulation requires planning a grasp configuration suited to the object and task, which is then executed through coordinated multi-finger control. However, specifying grasp plans with dense pose or contact targets for every object and task is impractical. Meanwhile, end-to-end reinforcement learning from task rewards alone lacks controllability, making it difficult for users to intervene when failures occur. To this end, we present GRIT, a two-stage framework that learns dexterous control from sparse taxonomy guidance. GRIT first predicts a taxonomy-based grasp specification from the scene and task context. Conditioned on this sparse command, a policy generates continuous finger motions that accomplish the task while preserving the intended grasp structure. Our result shows that certain grasp taxonomies are more effective for specific object geometries. By leveraging this relationship, GRIT improves generalization to novel objects over baselines and achieves an overall success rate of 87.9%. Moreover, real-world experiments demonstrate controllability, enabling grasp strategies to be adjusted through high-level taxonomy selection based on object geometry and task intent.
>
---
#### [new 036] Visual Prompt Based Reasoning for Offroad Mapping using Multimodal LLMs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于越野导航任务，解决传统方法需多个专用模型的问题。通过结合SAM2和视觉语言模型，实现无需特定地形模型的驾驶区域推理与导航。**

- **链接: [https://arxiv.org/pdf/2604.04564](https://arxiv.org/pdf/2604.04564)**

> **作者:** Abdelmoamen Nasser; Yousef Baba'a; Murad Mebrahtu; Nadya Abdel Madjid; Jorge Dias; Majid Khonji
>
> **摘要:** Traditional approaches to off-road autonomy rely on separate models for terrain classification, height estimation, and quantifying slip or slope conditions. Utilizing several models requires training each component separately, having task specific datasets, and fine-tuning. In this work, we present a zero-shot approach leveraging SAM2 for environment segmentation and a vision-language model (VLM) to reason about drivable areas. Our approach involves passing to the VLM both the original image and the segmented image annotated with numeric labels for each mask. The VLM is then prompted to identify which regions, represented by these numeric labels, are drivable. Combined with planning and control modules, this unified framework eliminates the need for explicit terrain-specific models and relies instead on the inherent reasoning capabilities of the VLM. Our approach surpasses state-of-the-art trainable models on high resolution segmentation datasets and enables full stack navigation in our Isaac Sim offroad environment.
>
---
#### [new 037] Robots Need Some Education: On the complexity of learning in evolutionary robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于进化机器人学领域，探讨将机器人学习引入进化过程中的复杂性，旨在设计合适的算法以提升进化效率。**

- **链接: [https://arxiv.org/pdf/2604.04196](https://arxiv.org/pdf/2604.04196)**

> **作者:** Fuda van Diggelen
>
> **备注:** PhD thesis
>
> **摘要:** Evolutionary Robotics and Robot Learning are two fields in robotics that aim to automatically optimize robot designs. The key difference between them lies in what is being optimized and the time scale involved. Evolutionary Robotics is a field that applies evolutionary computation techniques to evolve the morphologies or controllers, or both. Robot Learning, on the other hand, involves any learning technique aimed at optimizing a robot's controller in a given morphology. In terms of time scales, evolution occurs across multiple generations, whereas learning takes place within the `lifespan' of an individual robot. Integrating Robot Learning with Evolutionary Robotics requires the careful design of suitable learning algorithms in the context of evolutionary robotics. The effects of introducing learning into the evolutionary process are not well-understood and can thus be tricky. This thesis investigates these intricacies and presents several learning algorithms developed for an Evolutionary Robotics context.
>
---
#### [new 038] FORMULA: FORmation MPC with neUral barrier Learning for safety Assurance
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人系统安全控制任务，旨在解决复杂环境中机器人编队的实时安全导航问题。提出FORMULA框架，结合MPC、CLF和神经网络CBF，实现高效、安全的编队控制。**

- **链接: [https://arxiv.org/pdf/2604.04409](https://arxiv.org/pdf/2604.04409)**

> **作者:** Qintong Xie; Weishu Zhan; Peter Chin
>
> **备注:** Accepted to IEEE Intelligent Vehicles Symposium (IV) 2026
>
> **摘要:** Multi-robot systems (MRS) are essential for large-scale applications such as disaster response, material transport, and warehouse logistics, yet ensuring robust, safety-aware formation control in cluttered and dynamic environments remains a major challenge. Existing model predictive control (MPC) approaches suffer from limitations in scalability and provable safety, while control barrier functions (CBFs), though principled for safety enforcement, are difficult to handcraft for large-scale nonlinear systems. This paper presents FORMULA, a safe distributed, learning-enhanced predictive control framework that integrates MPC with Control Lyapunov Functions (CLFs) for stability and neural network-based CBFs for decentralized safety, eliminating manual safety constraint design. This scheme maintains formation integrity during obstacle avoidance, resolves deadlocks in dense configurations, and reduces online computational load. Simulation results demonstrate that FORMULA enables scalable, safety-aware, formation-preserving navigation for multi-robot teams in complex environments.
>
---
#### [new 039] Diffusion Policy with Bayesian Expert Selection for Active Multi-Target Tracking
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于多目标跟踪任务，解决机器人在跟踪中平衡探索与利用的问题。提出基于贝叶斯专家选择的扩散策略，提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2604.03404](https://arxiv.org/pdf/2604.03404)**

> **作者:** Haotian Xiang; Qin Lu; Yaakov Bar-Shalom
>
> **摘要:** Active multi-target tracking requires a mobile robot to balance exploration for undetected targets with exploitation of uncertain tracked ones. Diffusion policies have emerged as a powerful approach for capturing diverse behavioral strategies by learning action sequences from expert demonstrations. However, existing methods implicitly select among strategies through the denoising process, without uncertainty quantification over which strategy to execute. We formulate expert selection for diffusion policies as an offline contextual bandit problem and propose a Bayesian framework for pessimistic, uncertainty-aware strategy selection. A multi-head Variational Bayesian Last Layer (VBLL) model predicts the expected tracking performance of each expert strategy given the current belief state, providing both a point estimate and predictive uncertainty. Following the pessimism principle for offline decision-making, a Lower Confidence Bound (LCB) criterion then selects the expert whose worst-case predicted performance is best, avoiding overcommitment to experts with unreliable predictions. The selected expert conditions a diffusion policy to generate corresponding action sequences. Experiments on simulated indoor tracking scenarios demonstrate that our approach outperforms both the base diffusion policy and standard gating methods, including Mixture-of-Experts selection and deterministic regression baselines.
>
---
#### [new 040] Adaptive Action Chunking at Inference-time for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决动作分块大小固定导致的响应与一致性失衡问题。提出自适应动作分块策略，基于动作熵动态调整分块大小，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2604.04161](https://arxiv.org/pdf/2604.04161)**

> **作者:** Yuanchang Liang; Xiaobo Wang; Kai Wang; Shuo Wang; Xiaojiang Peng; Haoyu Chen; David Kim Huat Chua; Prahlad Vadakkepat
>
> **备注:** accepted by CVPR 2026
>
> **摘要:** In Vision-Language-Action (VLA) models, action chunking (i.e., executing a sequence of actions without intermediate replanning) is a key technique to improve robotic manipulation abilities. However, a large chunk size reduces the model's responsiveness to new information, while a small one increases the likelihood of mode-jumping, jerky behavior resulting from discontinuities between chunks. Therefore, selecting the optimal chunk size is an urgent demand to balance the model's reactivity and consistency. Unfortunately, a dominant trend in current VLA models is an empirical fixed chunk length at inference-time, hindering their superiority and scalability across diverse manipulation tasks. To address this issue, we propose a novel Adaptive Action Chunking (AAC) strategy, which exploits action entropy as the cue to adaptively determine the chunk size based on current predictions. Extensive experiments on a wide range of simulated and real-world robotic manipulation tasks have demonstrated that our approach substantially improves performance over the state-of-the-art alternatives. The videos and source code are publicly available at this https URL.
>
---
#### [new 041] Efficient Onboard Spacecraft Pose Estimation with Event Cameras and Neuromorphic Hardware
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于航天器位姿估计任务，解决空间图像复杂环境下定位难题。结合事件相机与神经形态硬件，提出高效低功耗的6-DoF位姿估计方法。**

- **链接: [https://arxiv.org/pdf/2604.04117](https://arxiv.org/pdf/2604.04117)**

> **作者:** Arunkumar Rathinam; Jules Lecomte; Jost Reelsen; Gregor Lenz; Axel von Arnim; Djamila Aouada
>
> **备注:** AI4SPACE workshop at CVPR 2026
>
> **摘要:** Reliable relative pose estimation is a key enabler for autonomous rendezvous and proximity operations, yet space imagery is notoriously challenging due to extreme illumination, high contrast, and fast target motion. Event cameras provide asynchronous, change-driven measurements that can remain informative when frame-based imagery saturates or blurs, while neuromorphic processors can exploit sparse activations for low-latency, energy-efficient inferences. This paper presents a spacecraft 6-DoF pose-estimation pipeline that couples event-based vision with the BrainChip Akida neuromorphic processor. Using the SPADES dataset, we train compact MobileNet-style keypoint regression networks on lightweight event-frame representations, apply quantization-aware training (8/4-bit), and convert the models to Akida-compatible spiking neural networks. We benchmark three event representations and demonstrate real-time, low-power inference on Akida V1 hardware. We additionally design a heatmap-based model targeting Akida V2 and evaluate it on Akida Cloud, yielding improved pose accuracy. To our knowledge, this is the first end-to-end demonstration of spacecraft pose estimation running on Akida hardware, highlighting a practical route to low-latency, low-power perception for future autonomous space missions.
>
---
#### [new 042] Primitive-based Truncated Diffusion for Efficient Trajectory Generation of Differential Drive Mobile Manipulators
- **分类: cs.RO**

- **简介: 该论文属于移动机械臂轨迹生成任务，解决效率与优化问题。提出基于关键点的截断扩散模型，提升成功率和轨迹多样性。**

- **链接: [https://arxiv.org/pdf/2604.04166](https://arxiv.org/pdf/2604.04166)**

> **作者:** Long Xu; Choilam Wong; Yuhang Zhong; Junxiao Lin; Jialiang Hou; Fei Gao
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** We present a learning-enhanced motion planner for differential drive mobile manipulators to improve efficiency, success rate, and optimality. For task representation encoder, we propose a keypoint sequence extraction module that maps boundary states to 3D space via differentiable forward kinematics. Point clouds and keypoints are encoded separately and fused with attention, enabling effective integration of environment and boundary states information. We also propose a primitive-based truncated diffusion model that samples from a biased distribution. Compared with vanilla diffusion model, this framework improves the efficiency and diversity of the solution. Denoised paths are refined by trajectory optimization to ensure dynamic feasibility and task-specific optimality. In cluttered 3D simulations, our method achieves higher success rate, improved trajectory diversity, and competitive runtime compared to vanilla diffusion and classical baselines. The source code is released at this https URL .
>
---
#### [new 043] CRAFT: Video Diffusion for Bimanual Robot Data Generation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出CRAFT框架，用于生成双臂机器人演示视频，解决真实数据成本高、多样性不足的问题，通过视频扩散技术合成多样且物理合理的操作视频。**

- **链接: [https://arxiv.org/pdf/2604.03552](https://arxiv.org/pdf/2604.03552)**

> **作者:** Jason Chen; I-Chun Arthur Liu; Gaurav Sukhatme; Daniel Seita
>
> **摘要:** Bimanual robot learning from demonstrations is fundamentally limited by the cost and narrow visual diversity of real-world data, which constrains policy robustness across viewpoints, object configurations, and embodiments. We present Canny-guided Robot Data Generation using Video Diffusion Transformers (CRAFT), a video diffusion-based framework for scalable bimanual demonstration generation that synthesizes temporally coherent manipulation videos while producing action labels. By conditioning video diffusion on edge-based structural cues extracted from simulator-generated trajectories, CRAFT produces physically plausible trajectory variations and supports a unified augmentation pipeline spanning object pose changes, camera viewpoints, lighting and background variations, cross-embodiment transfer, and multi-view synthesis. We leverage a pre-trained video diffusion model to convert simulated videos, along with action labels from the simulation trajectories, into action-consistent demonstrations. Starting from only a few real-world demonstrations, CRAFT generates a large, visually diverse set of photorealistic training data, bypassing the need to replay demonstrations on the real robot (Sim2Real). Across simulated and real-world bimanual tasks, CRAFT improves success rates over existing augmentation strategies and straightforward data scaling, demonstrating that diffusion-based video generation can substantially expand demonstration diversity and improve generalization for dual-arm manipulation tasks. Our project website is available at: this https URL
>
---
#### [new 044] Adversarial Robustness Analysis of Cloud-Assisted Autonomous Driving Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究云辅助自动驾驶系统的对抗鲁棒性，分析感知模型和网络攻击对系统安全的影响。通过测试床评估对抗样本和网络延迟的危害，提出增强跨层韧性的必要性。**

- **链接: [https://arxiv.org/pdf/2604.04349](https://arxiv.org/pdf/2604.04349)**

> **作者:** Maher Al Islam; Amr S. El-Wakeel
>
> **摘要:** Autonomous vehicles increasingly rely on deep learning-based perception and control, which impose substantial computational demands. Cloud-assisted architectures offload these functions to remote servers, enabling enhanced perception and coordinated decision-making through the Internet of Vehicles (IoV). However, this paradigm introduces cross-layer vulnerabilities, where adversarial manipulation of perception models and network impairments in the vehicle-cloud link can jointly undermine safety-critical autonomy. This paper presents a hardware-in-the-loop IoV testbed that integrates real-time perception, control, and communication to evaluate such vulnerabilities in cloud-assisted autonomous driving. A YOLOv8-based object detector deployed on the cloud is subjected to whitebox adversarial attacks using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), while network adversaries induce delay and packet loss in the vehicle-cloud loop. Results show that adversarial perturbations significantly degrade perception performance, with PGD reducing detection precision and recall from 0.73 and 0.68 in the clean baseline to 0.22 and 0.15 at epsilon= 0.04. Network delays of 150-250 ms, corresponding to transient losses of approximately 3-4 frames, and packet loss rates of 0.5-5 % further destabilize closed-loop control, leading to delayed actuation and rule violations. These findings highlight the need for cross-layer resilience in cloud-assisted autonomous driving systems.
>
---
#### [new 045] CT-VoxelMap: Efficient Continuous-Time LiDAR-Inertial Odometry with Probabilistic Adaptive Voxel Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决高速或复杂地形下的精准定位问题。通过改进B样条表示和引入 voxel 管理策略，提升系统精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.03747](https://arxiv.org/pdf/2604.03747)**

> **作者:** Lei Zhao; Xingyi Li; Tianchen Deng; Chuan Cao; Han Zhang; Weidong Chen
>
> **摘要:** Maintaining stable and accurate localization during fast motion or on rough terrain remains highly challenging for mobile robots with onboard resources. Currently, multi-sensor fusion methods based on continuous-time representation offer a potential and effective solution to this challenge. Among these, spline-based methods provide an efficient and intuitive approach for continuous-time representation. Previous continuous-time odometry works based on B-splines either treat control points as variables to be estimated or perform estimation in quaternion space, which introduces complexity in deriving analytical Jacobians and often overlooks the fitting error between the spline and the true trajectory over time. To address these issues, we first propose representing the increments of control points on matrix Lie groups as variables to be estimated. Leveraging the feature of the cumulative form of B-splines, we derive a more compact formulation that yields simpler analytical Jacobians without requiring additional boundary condition considerations. Second, we utilize forward propagation information from IMU measurements to estimate fitting errors online and further introduce a hybrid feature-based voxel map management strategy, enhancing system accuracy and robustness. Finally, we propose a re-estimation policy that significantly improves system computational efficiency and robustness. The proposed method is evaluated on multiple challenging public datasets, demonstrating superior performance on most sequences. Detailed ablation studies are conducted to analyze the impact of each module on the overall pose estimation system.
>
---
#### [new 046] DINO-VO: Learning Where to Focus for Enhanced State Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DINO-VO，解决视觉里程计中的场景泛化问题，通过端到端方法提升定位精度。**

- **链接: [https://arxiv.org/pdf/2604.04055](https://arxiv.org/pdf/2604.04055)**

> **作者:** Qi Chen; Guanghao Li; Sijia Hu; Xin Gao; Junpeng Ma; Xiangyang Xue; Jian Pu
>
> **摘要:** We present DINO Patch Visual Odometry (DINO-VO), an end-to-end monocular visual odometry system with strong scene generalization. Current Visual Odometry (VO) systems often rely on heuristic feature extraction strategies, which can degrade accuracy and robustness, particularly in large-scale outdoor environments. DINO-VO addresses these limitations by incorporating a differentiable adaptive patch selector into the end-to-end pipeline, improving the quality of extracted patches and enhancing generalization across diverse datasets. Additionally, our system integrates a multi-task feature extraction module with a differentiable bundle adjustment (BA) module that leverages inverse depth priors, enabling the system to learn and utilize appearance and geometric information effectively. This integration bridges the gap between feature learning and state estimation. Extensive experiments on the TartanAir, KITTI, Euroc, and TUM datasets demonstrate that DINO-VO exhibits strong generalization across synthetic, indoor, and outdoor environments, achieving state-of-the-art tracking accuracy.
>
---
#### [new 047] Element-based Formation Control: a Unified Perspective from Continuum Mechanics
- **分类: eess.SY; cs.MA; cs.RO; math.OC**

- **简介: 该论文属于多智能体形成控制任务，旨在解决传统方法依赖几何约束的问题。通过引入连续介质力学中的变形梯度，建立统一的元素化控制框架，实现多种几何不变性控制。**

- **链接: [https://arxiv.org/pdf/2604.04027](https://arxiv.org/pdf/2604.04027)**

> **作者:** Kun Cao; Lihua Xie
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** This paper establishes a unified element-based framework for formation control by introducing the concept of the deformation gradient from continuum mechanics. Unlike traditional methods that rely on geometric constraints defined on graph edges, we model the formation as a discrete elastic body composed of simplicial elements. By defining a generalized distortion energy based on the local deformation gradient tensor, we derive a family of distributed control laws that can enforce various geometric invariances, including translation, rotation, scaling, and affine transformations. The convergence properties and the features of the proposed controllers are analyzed in detail. Theoretically, we show that the proposed framework serves as a bridge between existing rigidity-based and Laplacian-based approaches. Specifically, we show that rigidity-based controllers are mathematically equivalent to minimizing specific projections of the deformation energy tensor. Furthermore, we establish a rigorous link between the proposed energy minimization and Laplacian-based formation control. Numerical simulations in 2D and 3D validate the effectiveness and the unified nature of the proposed framework.
>
---
#### [new 048] Relational Epipolar Graphs for Robust Relative Camera Pose Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉SLAM中的相对位姿估计任务，旨在解决噪声匹配带来的挑战。通过构建关系图模型，结合图运算优化位姿，提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.04554](https://arxiv.org/pdf/2604.04554)**

> **作者:** Prateeth Rao; Sachit Rao
>
> **备注:** 21 pages, 10 figures, yet to be submitted to IJCV
>
> **摘要:** A key component of Visual Simultaneous Localization and Mapping (VSLAM) is estimating relative camera poses using matched keypoints. Accurate estimation is challenged by noisy correspondences. Classical methods rely on stochastic hypothesis sampling and iterative estimation, while learning-based methods often lack explicit geometric structure. In this work, we reformulate relative pose estimation as a relational inference problem over epipolar correspondence graphs, where matched keypoints are nodes and nearby ones are connected by edges. Graph operations such as pruning, message passing, and pooling estimate a quaternion rotation, translation vector, and the Essential Matrix (EM). Minimizing a loss comprising (i) $\mathcal{L}_2$ differences with ground truth (GT), (ii) Frobenius norm between estimated and GT EMs, (iii) singular value differences, (iv) heading angle differences, and (v) scale differences, yields the relative pose between image pairs. The dense detector-free method LoFTR is used for matching. Experiments on indoor and outdoor benchmarks show improved robustness to dense noise and large baseline variation compared to classical and learning-guided approaches, highlighting the effectiveness of global relational consensus.
>
---
#### [new 049] FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出FlashSAC，一种快速稳定的离策略强化学习算法，用于高维机器人控制任务，解决传统方法在性能和效率上的不足。**

- **链接: [https://arxiv.org/pdf/2604.04539](https://arxiv.org/pdf/2604.04539)**

> **作者:** Donghu Kim; Youngdo Lee; Minho Park; Kinam Kim; I Made Aswin Nahendra; Takuma Seno; Sehee Min; Daniel Palenicek; Florian Vogt; Danica Kragic; Jan Peters; Jaegul Choo; Hojoon Lee
>
> **备注:** preprint, 40pages
>
> **摘要:** Reinforcement learning (RL) is a core approach for robot control when expert demonstrations are unavailable. On-policy methods such as Proximal Policy Optimization (PPO) are widely used for their stability, but their reliance on narrowly distributed on-policy data limits accurate policy evaluation in high-dimensional state and action spaces. Off-policy methods can overcome this limitation by learning from a broader state-action distribution, yet suffer from slow convergence and instability, as fitting a value function over diverse data requires many gradient updates, causing critic errors to accumulate through bootstrapping. We present FlashSAC, a fast and stable off-policy RL algorithm built on Soft Actor-Critic. Motivated by scaling laws observed in supervised learning, FlashSAC sharply reduces gradient updates while compensating with larger models and higher data throughput. To maintain stability at increased scale, FlashSAC explicitly bounds weight, feature, and gradient norms, curbing critic error accumulation. Across over 60 tasks in 10 simulators, FlashSAC consistently outperforms PPO and strong off-policy baselines in both final performance and training efficiency, with the largest gains on high-dimensional tasks such as dexterous manipulation. In sim-to-real humanoid locomotion, FlashSAC reduces training time from hours to minutes, demonstrating the promise of off-policy RL for sim-to-real transfer.
>
---
#### [new 050] Risk-Constrained Belief-Space Optimization for Safe Control under Latent Uncertainty
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于安全控制任务，解决在潜在不确定性下的安全控制问题。提出一种基于CVaR约束的信念空间优化方法，提升控制安全性与成功率。**

- **链接: [https://arxiv.org/pdf/2604.03868](https://arxiv.org/pdf/2604.03868)**

> **作者:** Clinton Enwerem; John S. Baras; Calin Belta
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Many safety-critical control systems must operate under latent uncertainty that sensors cannot directly resolve at decision time. Such uncertainty, arising from unknown physical properties, exogenous disturbances, or unobserved environment geometry, influences dynamics, task feasibility, and safety margins. Standard methods optimize expected performance and offer limited protection against rare but severe outcomes, while robust formulations treat uncertainty conservatively without exploiting its probabilistic structure. We consider partially observed dynamical systems whose dynamics, costs, and safety constraints depend on a latent parameter maintained as a belief distribution, and propose a risk-sensitive belief-space Model Predictive Path Integral (MPPI) control framework that plans under this belief while enforcing a Conditional Value-at-Risk (CVaR) constraint on a trajectory safety margin over the receding horizon. The resulting controller optimizes a risk-regularized performance objective while explicitly constraining the tail risk of safety violations induced by latent parameter variability. We establish three properties of the resulting risk-constrained controller: (1) the CVaR constraint implies a probabilistic safety guarantee, (2) the controller recovers the risk-neutral optimum as the risk weight in the objective tends to zero, and (3) a union-bound argument extends the per-horizon guarantee to cumulative safety over repeated solves. In physics-based simulations of a vision-guided dexterous stowing task in which a grasped object must be inserted into an occupied slot with pose uncertainty exceeding prescribed lateral clearance requirements, our method achieves 82% success with zero contact violations at high risk aversion, compared to 55% and 50% for a risk-neutral configuration and a chance-constrained baseline, both of which incur nonzero exterior contact forces.
>
---
#### [new 051] E-VLA: Event-Augmented Vision-Language-Action Model for Dark and Blurred Scenes
- **分类: cs.CV; cs.MM; cs.RO; eess.IV**

- **简介: 该论文属于机器人视觉-语言-动作任务，解决低光和模糊场景下的感知脆弱问题。通过引入事件流增强VLA模型，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.04834](https://arxiv.org/pdf/2604.04834)**

> **作者:** Jiajun Zhai; Hao Shi; Shangwei Guo; Kailun Yang; Kaiwei Wang
>
> **备注:** Code and dataset will be available at this https URL
>
> **摘要:** Robotic Vision-Language-Action (VLA) models generalize well for open-ended manipulation, but their perception is fragile under sensing-stage degradations such as extreme low light, motion blur, and black clipping. We present E-VLA, an event-augmented VLA framework that improves manipulation robustness when conventional frame-based vision becomes unreliable. Instead of reconstructing images from events, E-VLA directly leverages motion and structural cues in event streams to preserve semantic perception and perception-action consistency under adverse conditions. We build an open-source teleoperation platform with a DAVIS346 event camera and collect a real-world synchronized RGB-event-action manipulation dataset across diverse tasks and illumination settings. We also propose lightweight, pretrained-compatible event integration strategies and study event windowing and fusion for stable deployment. Experiments show that even a simple parameter-free fusion, i.e., overlaying accumulated event maps onto RGB images, could substantially improve robustness in dark and blur-heavy scenes: on Pick-Place at 20 lux, success increases from 0% (image-only) to 60% with overlay fusion and to 90% with our event adapter; under severe motion blur (1000 ms exposure), Pick-Place improves from 0% to 20-25%, and Sorting from 5% to 32.5%. Overall, E-VLA provides systematic evidence that event-driven perception can be effectively integrated into VLA models, pointing toward robust embodied intelligence beyond conventional frame-based imaging. Code and dataset will be available at this https URL.
>
---
#### [new 052] DHFP-PE: Dual-Precision Hybrid Floating Point Processing Element for AI Acceleration
- **分类: cs.AR; cs.RO; eess.AS; eess.IV**

- **简介: 该论文属于AI加速任务，旨在解决低功耗高吞吐量浮点运算问题，提出一种双精度混合浮点处理单元DHFP-PE。**

- **链接: [https://arxiv.org/pdf/2604.04507](https://arxiv.org/pdf/2604.04507)**

> **作者:** Shubham Kumar; Vijay Pratap Sharma; Vaibhav Neema; Santosh Kumar Vishvakarma
>
> **备注:** Accepted in ANRF-sponsored 2nd International Conference on Next Generation Electronics (NEleX-2026)
>
> **摘要:** The rapid adoption of low-precision arithmetic in artificial intelligence and edge computing has created a strong demand for energy-efficient and flexible floating-point multiply-accumulate (MAC) units. This paper presents a fully pipelined dual-precision floating-point MAC processing engine supporting FP8 formats (E4M3, E5M2) and FP4 formats (E2M1, E1M2), specifically optimized for low-power and high-throughput AI workloads. The proposed architecture employs a novel bit-partitioning technique that enables a single 4-bit unit multiplier to operate either as a standard 4x4 multiplier for FP8 or as two parallel 2x2 multipliers for 2-bit operands, achieving 100 percent hardware utilization without duplicating logic. Implemented in 28 nm technology, the proposed processing engine achieves an operating frequency of 1.94 GHz with an area of 0.00396 mm^2 and power consumption of 2.13 mW, resulting in up to 60.4 percent area reduction and 86.6 percent power savings compared to state-of-the-art designs.
>
---
#### [new 053] VitaTouch: Property-Aware Vision-Tactile-Language Model for Robotic Quality Inspection in Manufacturing
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出VitaTouch，解决制造中材料属性和表面特性识别问题。通过融合视觉、触觉与语言信息，提升质量检测精度。**

- **链接: [https://arxiv.org/pdf/2604.03322](https://arxiv.org/pdf/2604.03322)**

> **作者:** Junyi Zong; Qingxuan Jia; Meixian Shi; Tong Li; Jiayuan Li; Zihang Lv; Gang Chen; Fang Deng
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Quality inspection in smart manufacturing requires identifying intrinsic material and surface properties beyond visible geometry, yet vision-only methods remain vulnerable to occlusion and reflection. We propose VitaTouch, a property-aware vision-tactile-language model for material-property inference and natural-language attribute description. VitaTouch uses modality-specific encoders and a dual Q-Former to extract language-relevant visual and tactile features, which are compressed into prefix tokens for a large language model. We align each modality with text and explicitly couple vision and touch through contrastive learning. We also construct VitaSet, a multimodal dataset with 186 objects, 52k images, and 5.1k human-verified instruction-answer pairs. VitaTouch achieves the best performance on HCT and the overall TVL benchmark, while remaining competitive on SSVTP. On VitaSet, it reaches 88.89% hardness accuracy, 75.13% roughness accuracy, and 54.81% descriptor recall; the material-description task further achieves a peak semantic similarity of 0.9009. With LoRA-based fine-tuning, VitaTouch attains 100.0%, 96.0%, and 92.0% accuracy for 2-, 3-, and 5-category defect recognition, respectively, and delivers 94.0% closed-loop recognition accuracy and 94.0% end-to-end sorting success in 100 laboratory robotic trials. More details are available at the project page: this https URL
>
---
#### [new 054] Periodic Event-Triggered Explicit Reference Governor for Constrained Attitude Control on SO(3)
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于姿态控制任务，解决刚体在SO(3)上的约束姿态控制问题，提出PET-ERG方法实现输入饱和和几何约束下的稳定控制。**

- **链接: [https://arxiv.org/pdf/2604.04041](https://arxiv.org/pdf/2604.04041)**

> **作者:** Satoshi Nakano; Masahiro Suzuki; Misa Ohashi; Noboru Chikami; Shusuke Otabe
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This letter addresses the constrained attitude control problem for rigid bodies directly on the special orthogonal group SO(3), avoiding singularities associated with parameterizations such as Euler angles. We propose a novel Periodic Event-Triggered Explicit Reference Governor (PET-ERG) that enforces input saturation and geometric pointing constraints without relying on online optimization. A key feature is a periodic event-triggered supervisory update: the auxiliary reference is updated only at sampled instants when a robust safety condition is met, thereby avoiding continuous-time reference updates and enabling a rigorous stability analysis of the cascade system on the manifold. Through this structured approach, we rigorously establish the asymptotic stability and exponential convergence of the closed-loop system for almost all initial configurations. Numerical simulations validate the effectiveness of the proposed control architecture and demonstrate constraint satisfaction and convergence properties.
>
---
#### [new 055] Super Agents and Confounders: Influence of surrounding agents on vehicle trajectory prediction
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于轨迹预测任务，旨在解决周围车辆干扰导致预测精度下降的问题。通过分析现有模型，提出集成条件信息瓶颈方法，提升预测性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.03463](https://arxiv.org/pdf/2604.03463)**

> **作者:** Daniel Jost; Luca Paparusso; Martin Stoll; Jörg Wagner; Raghu Rajan; Joschka Bödecker
>
> **摘要:** In highly interactive driving scenes, trajectory prediction is conditioned on information from surrounding traffic participants such as cars and pedestrians. Our main contribution is a comprehensive analysis of state-of-the-art trajectory predictors, which reveals a surprising and critical flaw: many surrounding agents degrade prediction accuracy rather than improve it. Using Shapley-based attribution, we rigorously demonstrate that models learn unstable and non-causal decision-making schemes that vary significantly across training runs. Building on these insights, we propose to integrate a Conditional Information Bottleneck (CIB), which does not require additional supervision and is trained to effectively compress agent features as well as ignore those that are not beneficial for the prediction task. Comprehensive experiments using multiple datasets and model architectures demonstrate that this simple yet effective approach not only improves overall trajectory prediction performance in many cases but also increases robustness to different perturbations. Our results highlight the importance of selectively integrating contextual information, which can often contain spurious or misleading signals, in trajectory prediction. Moreover, we provide interpretable metrics for identifying non-robust behavior and present a promising avenue towards a solution.
>
---
#### [new 056] Review and Evaluation of Point-Cloud based Leaf Surface Reconstruction Methods for Agricultural Applications
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云表面重建任务，旨在解决农业中叶片表面重建的难题。通过比较九种方法，评估其在不同数据条件下的性能，为资源受限的农业机器人提供选择依据。**

- **链接: [https://arxiv.org/pdf/2604.03328](https://arxiv.org/pdf/2604.03328)**

> **作者:** Arif Ahmed; Parikshit Maini
>
> **摘要:** Accurate reconstruction of leaf surfaces from 3D point cloud is essential for agricultural applications such as phenotyping. However, real-world plant data (i.e., irregular 3D point cloud) are often complex to reconstruct plant parts accurately. A wide range of surface reconstruction methods has been proposed, including parametric, triangulation-based, implicit, and learning based approaches, yet their relative performance for leaf surface reconstruction remains insufficiently understood. In this work, we present a comparative study of nine representative surface reconstruction methods for leaf surfaces. We evaluate these methods on three publicly available datasets: LAST-STRAW, Pheno4D, and Crops3D - spanning diverse species, sensors, and sensing environments, ranging from clean high-resolution indoor scans to noisy low-resolution field settings. The analysis highlights the trade-offs between surface area estimation accuracy, smoothness, robustness to noise and missing data, and computational cost across different methods. These factors affect the cost and constraints of robotic hardware used in agricultural applications. Our results show that each method exhibits distinct advantages depending on application and resource constraints. The findings provide practical guidance for selecting surface reconstruction techniques for resource constrained robotic platforms.
>
---
#### [new 057] ZeD-MAP: Bundle Adjustment Guided Zero-Shot Depth Maps for Real-Time Aerial Imaging
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于实时三维重建任务，解决无人机影像深度估计的精度与一致性问题。通过集成BA的零样本深度模型，提升实时地图生成的准确性与效率。**

- **链接: [https://arxiv.org/pdf/2604.04667](https://arxiv.org/pdf/2604.04667)**

> **作者:** Selim Ahmet Iz; Francesco Nex; Norman Kerle; Henry Meissner; Ralf Berger
>
> **摘要:** Real-time depth reconstruction from ultra-high-resolution UAV imagery is essential for time-critical geospatial tasks such as disaster response, yet remains challenging due to wide-baseline parallax, large image sizes, low-texture or specular surfaces, occlusions, and strict computational constraints. Recent zero-shot diffusion models offer fast per-image dense predictions without task-specific retraining, and require fewer labelled datasets than transformer-based predictors while avoiding the rigid capture geometry requirement of classical multi-view stereo. However, their probabilistic inference prevents reliable metric accuracy and temporal consistency across sequential frames and overlapping tiles. We present ZeD-MAP, a cluster-level framework that converts a test-time diffusion depth model into a metrically consistent, SLAM-like mapping pipeline by integrating incremental cluster-based bundle adjustment (BA). Streamed UAV frames are grouped into overlapping clusters; periodic BA produces metrically consistent poses and sparse 3D tie-points, which are reprojected into selected frames and used as metric guidance for diffusion-based depth estimation. Validation on ground-marker flights captured at approximately 50 m altitude (GSD is approximately 0.85 cm/px, corresponding to 2,650 square meters ground coverage per frame) with the DLR Modular Aerial Camera System (MACS) shows that our method achieves sub-meter accuracy, with approximately 0.87 m error in the horizontal (XY) plane and 0.12 m in the vertical (Z) direction, while maintaining per-image runtimes between 1.47 and 4.91 seconds. Results are subject to minor noise from manual point-cloud annotation. These findings show that BA-based metric guidance provides consistency comparable to classical photogrammetric methods while significantly accelerating processing, enabling real-time 3D map generation.
>
---
#### [new 058] DriveVA: Video Action Models are Zero-Shot Drivers
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DriveVA，解决自动驾驶中的泛化与轨迹一致性问题。通过联合预测视觉与动作序列，提升模型在未见场景下的表现。**

- **链接: [https://arxiv.org/pdf/2604.04198](https://arxiv.org/pdf/2604.04198)**

> **作者:** Mengmeng Liu; Diankun Zhang; Jiuming Liu; Jianfeng Cui; Hongwei Xie; Guang Chen; Hangjun Ye; Michael Ying Yang; Francesco Nex; Hao Cheng
>
> **摘要:** Generalization is a central challenge in autonomous driving, as real-world deployment requires robust performance under unseen scenarios, sensor domains, and environmental conditions. Recent world-model-based planning methods have shown strong capabilities in scene understanding and multi-modal future prediction, yet their generalization across datasets and sensor configurations remains limited. In addition, their loosely coupled planning paradigm often leads to poor video-trajectory consistency during visual imagination. To overcome these limitations, we propose DriveVA, a novel autonomous driving world model that jointly decodes future visual forecasts and action sequences in a shared latent generative process. DriveVA inherits rich priors on motion dynamics and physical plausibility from well-pretrained large-scale video generation models to capture continuous spatiotemporal evolution and causal interaction patterns. To this end, DriveVA employs a DiT-based decoder to jointly predict future action sequences (trajectories) and videos, enabling tighter alignment between planning and scene evolution. We also introduce a video continuation strategy to strengthen long-duration rollout consistency. DriveVA achieves an impressive closed-loop performance of 90.9 PDM score on the challenge NAVSIM. Extensive experiments also demonstrate the zero-shot capability and cross-domain generalization of DriveVA, which reduces average L2 error and collision rate by 78.9% and 83.3% on nuScenes and 52.5% and 52.4% on the Bench2drive built on CARLA v2 compared with the state-of-the-art world-model-based planner.
>
---
#### [new 059] Optimization-Free Constrained Control with Guaranteed Recursive Feasibility: A CBF-Based Reference Governor Approach
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制领域，解决约束控制中的递归可行性问题。通过结合参考发生器与控制屏障函数，设计一种无需在线优化的控制方法，确保安全约束满足。**

- **链接: [https://arxiv.org/pdf/2604.04001](https://arxiv.org/pdf/2604.04001)**

> **作者:** Satoshi Nakano; Emanuele Garone; Gennaro Notomista
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This letter presents a constrained control framework that integrates Explicit Reference Governors (ERG) with Control Barrier Functions (CBF) to ensure recursive feasibility without online optimization. We formulate the reference update as a virtual control input for an augmented system, governed by a smooth barrier function constructed from the softmin aggregation of Dynamic Safety Margins (DSMs). Unlike standard CBF formulations, the proposed method guarantees the feasibility of safety constraints by design, exploiting the forward invariance properties of the underlying Lyapunov level sets. This allows for the derivation of an explicit, closed-form reference update law that strictly enforces safety while minimizing deviation from a nominal reference trajectory. Theoretical results confirm asymptotic convergence, and numerical simulations demonstrate that the proposed method achieves performance comparable to traditional ERG frameworks.
>
---
#### [new 060] SpectralSplat: Appearance-Disentangled Feed-Forward Gaussian Splatting for Driving Scenes
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于3D场景重建任务，解决几何与外观耦合问题。通过分离外观与几何，实现可控的外观迁移和时间一致的光照重渲染。**

- **链接: [https://arxiv.org/pdf/2604.03462](https://arxiv.org/pdf/2604.03462)**

> **作者:** Quentin Herau; Tianshuo Xu; Depu Meng; Jiezhi Yang; Chensheng Peng; Spencer Sherk; Yihan Hu; Wei Zhan
>
> **备注:** Under review
>
> **摘要:** Feed-forward 3D Gaussian Splatting methods have achieved impressive reconstruction quality for autonomous driving scenes, yet they entangle scene geometry with transient appearance properties such as lighting, weather, and time of day. This coupling prevents relighting, appearance transfer, and consistent rendering across multi-traversal data captured under varying environmental conditions. We present SpectralSplat, a method that disentangles appearance from geometry within a feed-forward Gaussian Splatting framework. Our key insight is to factor color prediction into an appearance-agnostic base stream and and appearance-conditioned adapted stream, both produced by a shared MLP conditioned on a global appearance embedding derived from DINOv2 features. To enforce disentanglement, we train with paired observations generated by a hybrid relighting pipeline that combines physics-based intrinsic decomposition with diffusion based generative refinement, and supervise with complementary consistency, reconstruction, cross-appearance, and base color losses. We further introduce an appearance-adaptable temporal history that stores appearance-agnostic features, enabling accumulated Gaussians to be re-rendered under arbitrary target appearances. Experiments demonstrate that SpectralSplat preserves the reconstruction quality of the underlying backbone while enabling controllable appearance transfer and temporally consistent relighting across driving sequences.
>
---
#### [new 061] MPTF-Net: Multi-view Pyramid Transformer Fusion Network for LiDAR-based Place Recognition
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于LiDAR-based place recognition任务，解决复杂环境中特征表达不足的问题。提出MPTF-Net，结合多视角Transformer融合多尺度BEV特征，提升识别精度与实时性。**

- **链接: [https://arxiv.org/pdf/2604.04513](https://arxiv.org/pdf/2604.04513)**

> **作者:** Shuyuan Li; Zihang Wang; Xieyuanli Chen; Wenkai Zhu; Xiaoteng Fang; Peizhou Ni; Junhao Yang; Dong Kong
>
> **摘要:** LiDAR-based place recognition (LPR) is essential for global localization and loop-closure detection in large-scale SLAM systems. Existing methods typically construct global descriptors from Range Images or BEV representations for matching. BEV is widely adopted due to its explicit 2D spatial layout encoding and efficient retrieval. However, conventional BEV representations rely on simple statistical aggregation, which fails to capture fine-grained geometric structures, leading to performance degradation in complex or repetitive environments. To address this, we propose MPTF-Net, a novel multi-view multi-scale pyramid Transformer fusion network. Our core contribution is a multi-channel NDT-based BEV encoding that explicitly models local geometric complexity and intensity distributions via Normal Distribution Transform, providing a noise-resilient structural prior. To effectively integrate these features, we develop a customized pyramid Transformer module that captures cross-view interactive correlations between Range Image Views (RIV) and NDT-BEV at multiple spatial scales. Extensive experiments on the nuScenes, KITTI and NCLT datasets demonstrate that MPTF-Net achieves state-of-the-art performance, specifically attaining a Recall@1 of 96.31\% on the nuScenes Boston split while maintaining an inference latency of only 10.02 ms, making it highly suitable for real-time autonomous unmanned systems.
>
---
#### [new 062] Learning from Imperfect Demonstrations via Temporal Behavior Tree-Guided Trajectory Repair
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于机器人学习任务，解决从不完美演示中学习控制策略的问题。通过时间行为树引导轨迹修复，提升数据质量并优化强化学习效果。**

- **链接: [https://arxiv.org/pdf/2604.04225](https://arxiv.org/pdf/2604.04225)**

> **作者:** Aniruddh G. Puranic; Sebastian Schirmer; John S. Baras; Calin Belta
>
> **备注:** 12 pages, 4 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Learning robot control policies from demonstrations is a powerful paradigm, yet real-world data is often suboptimal, noisy, or otherwise imperfect, posing significant challenges for imitation and reinforcement learning. In this work, we present a formal framework that leverages Temporal Behavior Trees (TBT), an extension of Signal Temporal Logic (STL) with Behavior Tree semantics, to repair suboptimal trajectories prior to their use in downstream policy learning. Given demonstrations that violate a TBT specification, a model-based repair algorithm corrects trajectory segments to satisfy the formal constraints, yielding a dataset that is both logically consistent and interpretable. The repaired trajectories are then used to extract potential functions that shape the reward signal for reinforcement learning, guiding the agent toward task-consistent regions of the state space without requiring knowledge of the agent's kinematic model. We demonstrate the effectiveness of this framework on discrete grid-world navigation and continuous single and multi-agent reach-avoid tasks, highlighting its potential for data-efficient robot learning in settings where high-quality demonstrations cannot be assumed.
>
---
#### [new 063] Safety-Aligned 3D Object Detection: Single-Vehicle, Cooperative, and End-to-End Perspectives
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D目标检测任务，旨在提升自动驾驶车辆的安全性。针对传统评估标准无法区分安全相关错误的问题，提出安全对齐的评估与优化方法，增强检测系统的安全性。**

- **链接: [https://arxiv.org/pdf/2604.03325](https://arxiv.org/pdf/2604.03325)**

> **作者:** Brian Hsuan-Cheng Liao; Chih-Hong Cheng; Hasan Esen; Alois Knoll
>
> **备注:** 10 pages, 9 figures, 6 tables
>
> **摘要:** Perception plays a central role in connected and autonomous vehicles (CAVs), underpinning not only conventional modular driving stacks, but also cooperative perception systems and recent end-to-end driving models. While deep learning has greatly improved perception performance, its statistical nature makes perfect predictions difficult to attain. Meanwhile, standard training objectives and evaluation benchmarks treat all perception errors equally, even though only a subset is safety-critical. In this paper, we investigate safety-aligned evaluation and optimization for 3D object detection that explicitly characterize high-impact errors. Building on our previously proposed safety-oriented metric, NDS-USC, and safety-aware loss function, EC-IoU, we make three contributions. First, we present an expanded study of single-vehicle 3D object detection models across diverse neural network architectures and sensing modalities, showing that gains under standard metrics such as mAP and NDS may not translate to safety-oriented criteria represented by NDS-USC. With EC-IoU, we reaffirm the benefit of safety-aware fine-tuning for improving safety-critical detection performance. Second, we conduct an ego-centric, safety-oriented evaluation of AV-infrastructure cooperative object detection models, underscoring its superiority over vehicle-only models and demonstrating a safety impact analysis that illustrates the potential contribution of cooperative models to "Vision Zero." Third, we integrate EC-IoU into SparseDrive and show that safety-aware perception hardening can reduce collision rate by nearly 30% and improve system-level safety directly in an end-to-end perception-to-planning framework. Overall, our results indicate that safety-aligned perception evaluation and optimization offer a practical path toward enhancing CAV safety across single-vehicle, cooperative, and end-to-end autonomy settings.
>
---
## 更新

#### [replaced 001] Privacy-Preserving Semantic Segmentation from Ultra-Low-Resolution RGB Inputs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于语义分割任务，解决在隐私敏感环境下低分辨率RGB输入下的语义分割问题。提出一种联合学习框架，提升隐私保护与分割性能的平衡。**

- **链接: [https://arxiv.org/pdf/2507.16034](https://arxiv.org/pdf/2507.16034)**

> **作者:** Xuying Huang; Sicong Pan; Olga Zatsarynna; Juergen Gall; Maren Bennewitz
>
> **备注:** Submit to IJCV Special Issue on Responsible Imaging
>
> **摘要:** RGB-based semantic segmentation has become a mainstream approach for visual perception and is widely applied in a variety of downstream tasks. However, existing methods typically rely on high-resolution RGB inputs, which may expose sensitive visual content in privacy-critical environments. Ultra-low-resolution RGB sensing suppresses sensitive information directly during image acquisition, making it an attractive privacy-preserving alternative. Nevertheless, recovering semantic segmentation from ultra-low-resolution RGB inputs remains highly challenging due to severe visual degradation. In this work, we introduce a novel fully joint-learning framework to mitigate the optimization conflicts exacerbated by visual degradation for ultra-low-resolution semantic segmentation. Experiments demonstrate that our method outperforms representative baselines in semantic segmentation performance and our ultra-low-resolution RGB input achieves a favorable trade-off between privacy preservation and semantic segmentation performance. We deploy our privacy-preserving semantic segmentation method in a real-world robotic object-goal navigation task, demonstrating successful downstream task execution even under severe visual degradation.
>
---
#### [replaced 002] Temporal Reach-Avoid-Stay Control for Differential Drive Systems via Spatiotemporal Tubes
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于路径规划任务，解决差分驱动机器人在动态不确定环境中的安全轨迹生成问题，通过构建时空管状走廊实现时间约束下的避障与到达。**

- **链接: [https://arxiv.org/pdf/2512.05495](https://arxiv.org/pdf/2512.05495)**

> **作者:** Ratnangshu Das; Ahan Basu; Christos Verginis; Pushpak Jagtap
>
> **摘要:** This paper presents a computationally lightweight and robust control framework for differential-drive mobile robots with dynamic uncertainties and external disturbances, guaranteeing the satisfaction of Temporal Reach-Avoid-Stay (T-RAS) specifications. The approach employs circular spatiotemporal tubes (STTs), characterized by smoothly time-varying center and radius, to define dynamic safe corridors that guide the robot from the start region to the goal while avoiding obstacles. In particular, we first develop a sampling-based synthesis algorithm to construct a feasible STT that satisfies the prescribed timing and safety constraints with formal guarantees. To ensure that the robot remains confined within this tube, we then analytically design a closed-form control that is computationally efficient and robust to disturbances. The proposed framework is validated through simulation studies on a differential-drive robot and benchmarked against state-of-the-art methods, demonstrating superior robustness, accuracy, and computational efficiency.
>
---
#### [replaced 003] Decoupling Torque and Stiffness: A Unified Modeling and Control Framework for Antagonistic Artificial Muscles
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决人工肌肉中扭矩与刚度耦合问题。提出统一框架实现两者解耦控制，提升动态交互性能。**

- **链接: [https://arxiv.org/pdf/2511.09104](https://arxiv.org/pdf/2511.09104)**

> **作者:** Amirhossein Kazemipour; Robert K. Katzschmann
>
> **摘要:** Antagonistic artificial muscles can decouple joint torque and stiffness, but contact transients often degrade this independence. We present a unified real-time framework applicable across pneumatic, electrohydraulic, and dielectric elastomer artificial muscle families: a separable Padé force model with a minimal two-state dynamic wrapper, a cascaded inverse-dynamics controller in co-contraction/bias coordinates, and a bio-inspired depth-adaptive interaction policy that schedules stiffness based on penetration depth. The controller runs in under 1 ms per control tick and demonstrates independent torque and stiffness tracking, including a fixed-torque stiffness-step test that preserves torque regulation through stiffness transitions. In a coupled impedance contact protocol simulated across soft-to-rigid environments, comparing depth-adaptive stiffness to fixed-stiffness baselines reveals a shock/load versus stability tradeoff. These results provide a control-oriented foundation for musculoskeletal antagonistic robots to execute adaptive impedance behaviors in dynamic interactions.
>
---
#### [replaced 004] The N-5 Scaling Law: Topological Dimensionality Reduction in the Optimal Design of Fully-actuated Multirotors
- **分类: cs.RO; math.GT; math.OC**

- **简介: 该论文研究多旋翼飞行器的最优设计问题，通过拓扑分析揭示其解空间结构，提出N-5尺度定律，解决几何优化中的对称性与冗余性问题。**

- **链接: [https://arxiv.org/pdf/2512.23619](https://arxiv.org/pdf/2512.23619)**

> **作者:** Antonio Franchi
>
> **摘要:** The geometric design of fully-actuated and omnidirectional N-rotor aerial vehicles is conventionally formulated as a parametric optimization problem, seeking a single optimal set of N orientations within a fixed architectural family. This work departs from that paradigm to investigate the intrinsic topological structure of the optimization landscape itself. We formulate the design problem on the product manifold of Projective Lines \RP^2^N, fixing the rotor positions to the vertices of polyhedral chassis while varying their lines of action. By minimizing a coordinate-invariant Log-Volume isotropy metric, we reveal that the topology of the global optima is governed strictly by the symmetry of the chassis. For generic (irregular) vertex arrangements, the solutions appear as a discrete set of isolated points. However, as the chassis geometry approaches regularity, the solution space undergoes a critical phase transition, collapsing onto an N-dimensional Torus of the lines tangent at the vertexes to the circumscribing sphere of the chassis, and subsequently reducing to continuous 1-dimensional curves driven by Affine Phase Locking. We synthesize these observations into the N-5 Scaling Law: an empirical relationship holding for all examined regular planar polygons and Platonic solids (N <= 10), where the space of optimal configurations consists of K=N-5 disconnected 1D topological branches. We demonstrate that these locking patterns correspond to a sequence of admissible Star Polygons {N/q}, allowing for the exact prediction of optimal phases for arbitrary N. Crucially, this topology reveals a design redundancy that enables optimality-preserving morphing: the vehicle can continuously reconfigure along these branches while preserving optimal isotropic control authority.
>
---
#### [replaced 005] ST-BiBench: Benchmarking Multi-Stream Multimodal Coordination in Bimanual Embodied Tasks for MLLMs
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ST-BiBench，用于评估多流多模态协调能力，解决MLLM在双臂协同任务中的多模态融合与对齐问题。**

- **链接: [https://arxiv.org/pdf/2602.08392](https://arxiv.org/pdf/2602.08392)**

> **作者:** Xin Wu; Zhixuan Liang; Yue Ma; Mengkang Hu; Zhiyuan Qin; Xiu Li
>
> **备注:** 42 pages, 9 figures. Project page:this https URL
>
> **摘要:** Multimodal Large Language Models (MLLMs) have significantly advanced the landscape of embodied AI, yet transitioning to synchronized bimanual coordination introduces formidable challenges in multi-stream multimodal integration. We introduce ST-BiBench, a comprehensive multi-tier framework for evaluating spatio-temporal multimodal coordination. Our approach centers on Strategic Coordination Planning, assessing high-level cross-modal reasoning over multiple action and perception streams. To investigate the "proximity paradox"-where semantically coherent plans fail to align with spatially grounded visual inputs-we incorporate Foundational Spatial Grounding to verify workspace awareness and arm-selection logic. Furthermore, we probe model frontiers through Fine-Grained Action Control, investigating whether MLLMs can directly synthesize high-dimensional continuous action modalities (16-Dim) from complex multimodal metadata. Evaluating 30+ state-of-the-art MLLMs, we uncover a persistent and pervasive "coordination paradox"-a significant gap between high-level strategic reasoning and fine-grained physical execution. Results reveal that while frontier MLLMs excel at logic-driven strategy, they frequently suffer from perception-logic disconnection and multi-stream interference during multimodal fusion. ST-BiBench provides a platform for identifying critical bottlenecks in multi-stream multimodal fusion and cross-modal alignment for complex embodied tasks.
>
---
#### [replaced 006] LongTail Driving Scenarios with Reasoning Traces: The KITScenes LongTail Dataset
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出KITScenes LongTail数据集，用于解决自动驾驶中罕见场景的泛化问题。通过多视角视频、轨迹和推理轨迹，支持少样本学习与指令遵循评估。**

- **链接: [https://arxiv.org/pdf/2603.23607](https://arxiv.org/pdf/2603.23607)**

> **作者:** Royden Wagner; Omer Sahin Tas; Jaime Villa; Felix Hauser; Yinzhe Shen; Marlon Steiner; Dominik Strutz; Carlos Fernandez; Christian Kinzig; Guillermo S. Guitierrez-Cabello; Hendrik Königshof; Fabian Immel; Richard Schwarzkopf; Nils Alexander Rack; Kevin Rösch; Kaiwen Wang; Jan-Hendrik Pauls; Martin Lauer; Igor Gilitschenski; Holger Caesar; Christoph Stiller
>
> **备注:** 21 pages; v2: update MMS values (bugfix)
>
> **摘要:** In real-world domains such as self-driving, generalization to rare scenarios remains a fundamental challenge. To address this, we introduce a new dataset designed for end-to-end driving that focuses on long-tail driving events. We provide multi-view video data, trajectories, high-level instructions, and detailed reasoning traces, facilitating in-context learning and few-shot generalization. The resulting benchmark for multimodal models, such as VLMs and VLAs, goes beyond safety and comfort metrics by evaluating instruction following and semantic coherence between model outputs. The multilingual reasoning traces in English, Spanish, and Chinese are from domain experts with diverse cultural backgrounds. Thus, our dataset is a unique resource for studying how different forms of reasoning affect driving competence. Our dataset is available at: this https URL
>
---
#### [replaced 007] MPCFormer: A physics-informed data-driven approach for explainable socially-aware autonomous driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶任务，旨在解决车辆社交交互理解不足的问题。提出MPCFormer模型，结合物理先验与数据驱动，提升交互可解释性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.03795](https://arxiv.org/pdf/2512.03795)**

> **作者:** Jia Hu; Zhexi Lian; Xuerun Yan; Ruiang Bi; Dou Shen; Yu Ruan; Chunlong Xia; Haoran Wang
>
> **备注:** 17 pages, 17 figures
>
> **摘要:** Autonomous Driving (AD) vehicles still struggle to exhibit human-like behavior in highly dynamic and interactive traffic scenarios. The key challenge lies in AD's limited ability to interact with surrounding vehicles, largely due to a lack of understanding the underlying mechanisms of social interaction. To address this issue, we introduce MPCFormer, an explainable socially-aware autonomous driving approach with physics-informed and data-driven coupled social interaction dynamics. In this model, the dynamics are formulated into a discrete space-state representation, which embeds physics priors to enhance modeling explainability. The dynamics coefficients are learned from naturalistic driving data via a Transformer-based encoder-decoder architecture. To the best of our knowledge, MPCFormer is the first approach to explicitly model the dynamics of multi-vehicle social interactions. The learned social interaction dynamics enable the planner to generate manifold, human-like behaviors when interacting with surrounding traffic. By leveraging the MPC framework, the approach mitigates the potential safety risks typically associated with purely learning-based methods. Open-looped evaluation on NGSIM dataset demonstrates that MPCFormer achieves superior social interaction awareness, yielding the lowest trajectory prediction errors compared with other state-of-the-art approaches. The prediction achieves an ADE as low as 0.86 m over a long prediction horizon of 5 seconds. Close-looped experiments in highly intense interaction scenarios, where consecutive lane changes are required to exit an off-ramp, further validate the effectiveness of MPCFormer. Results show that MPCFormer achieves the highest planning success rate of 94.67%, improves driving efficiency by 15.75%, and reduces the collision rate from 21.25% to 0.5%, outperforming a frontier Reinforcement Learning (RL) based planner.
>
---
#### [replaced 008] Watch Your Step: Learning Semantically-Guided Locomotion in Cluttered Environment
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决腿式机器人在杂乱环境中误踩障碍物的问题。通过引入SemLoco框架，结合语义地图与强化学习，提升足部安全定位能力，减少碰撞，提高安全性。**

- **链接: [https://arxiv.org/pdf/2603.02657](https://arxiv.org/pdf/2603.02657)**

> **作者:** Denan Liang; Yuan Zhu; Ruimeng Liu; Thien-Minh Nguyen; Shenghai Yuan; Lihua Xie
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** Although legged robots demonstrate impressive mobility on rough terrain, using them safely in cluttered environments remains a challenge. A key issue is their inability to avoid stepping on low-lying objects, such as high-cost small devices or cables on flat ground. This limitation arises from a disconnection between high-level semantic understanding and low-level control, combined with errors in elevation maps during real-world operation. To address this, we introduce SemLoco, a Reinforcement Learning (RL) framework designed to avoid obstacles precisely in densely cluttered environments. SemLoco uses a two-stage RL approach that combines both soft and hard constraints. It performs pixel-wise foothold safety inference, which enables more accurate foot placement. Additionally, SemLoco integrates semantic map, allowing it to assign traversability costs instead of relying only on geometric data. SemLoco greatly reduces collisions and improves safety around sensitive objects, enabling reliable navigation in situations where traditional controllers would likely cause damage. Experimental results further show that SemLoco can be effectively applied to more complex, unstructured real-world environments. A demo video can be view at this https URL.
>
---
#### [replaced 009] Teaching Machine Learning Fundamentals with LEGO Robotics
- **分类: cs.RO; cs.AI; cs.CY; cs.HC; cs.LG**

- **简介: 该论文属于教育技术任务，旨在通过LEGO机器人教学机器学习基础。解决如何让青少年理解复杂概念的问题，通过可视化平台和编程-free活动进行教学。**

- **链接: [https://arxiv.org/pdf/2601.19376](https://arxiv.org/pdf/2601.19376)**

> **作者:** Viacheslav Sydora; Guner Dilsad Er; Michael Muehlebach
>
> **备注:** 10 pages, 8 figures
>
> **摘要:** This paper presents the web-based platform Machine Learning with Bricks and an accompanying two-day course designed to teach machine learning concepts to students aged 12 to 17 through programming-free robotics activities. Machine Learning with Bricks is an open source platform and combines interactive visualizations with LEGO robotics to teach three core algorithms: KNN, linear regression, and Q-learning. Students learn by collecting data, training models, and interacting with robots via a web-based interface. Pre- and post-surveys with 14 students indicate statistically significant improvements in self-reported understanding of machine learning algorithms, changes in AI-related terminology toward more technical language, high platform usability, and increased motivation for continued learning. This work suggests that tangible, visualization-based approaches can make machine learning concepts accessible and engaging for young learners while maintaining technical depth. The platform is freely available at this https URL, with video tutorials guiding students through the experiments at this https URL.
>
---
#### [replaced 010] Empowering Multi-Robot Cooperation via Sequential World Models
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作任务，旨在解决物理多机器人系统中模型预测复杂的问题。提出SeqWM框架，通过序列世界模型实现高效协作与行为规划。**

- **链接: [https://arxiv.org/pdf/2509.13095](https://arxiv.org/pdf/2509.13095)**

> **作者:** Zijie Zhao; Honglei Guo; Shengqian Chen; Kaixuan Xu; Bo Jiang; Yuanheng Zhu; Dongbin Zhao
>
> **摘要:** Model-based reinforcement learning (MBRL) has achieved remarkable success in robotics due to its high sample efficiency and planning capability. However, extending MBRL to physical multi-robot cooperation remains challenging due to the complexity of joint dynamics. To address this challenge, we propose the Sequential World Model (SeqWM), a novel framework that integrates the sequential paradigm into multi-robot MBRL. SeqWM employs independent, autoregressive agent-wise world models to represent joint dynamics, where each agent generates its future trajectory and plans its actions based on the predictions of its predecessors. This design lowers modeling complexity and enables the emergence of advanced cooperative behaviors through explicit intention sharing. Experiments on Bi-DexHands and Multi-Quadruped demonstrate that SeqWM outperforms existing state-of-the-art model-based and model-free baselines in both overall performance and sample efficiency, while exhibiting advanced cooperative behaviors such as predictive adaptation, temporal alignment, and role division. Furthermore, SeqWM has been successfully deployed on physical quadruped robots, validating its effectiveness in real-world multi-robot systems. Demos and code are available at: this https URL
>
---
#### [replaced 011] Red-Teaming Vision-Language-Action Models via Quality Diversity Prompt Generation for Robust Robot Policies
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型对指令敏感导致的失败问题。通过Q-DIG方法生成多样且自然的对抗指令，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.12510](https://arxiv.org/pdf/2603.12510)**

> **作者:** Siddharth Srikanth; Freddie Liang; Ya-Chuan Hsu; Varun Bhatt; Shihan Zhao; Henry Chen; Bryon Tjanaka; Minjune Hwang; Akanksha Saran; Daniel Seita; Aaquib Tabrez; Stefanos Nikolaidis
>
> **摘要:** Vision-Language-Action (VLA) models have significant potential to enable general-purpose robotic systems for a range of vision-language tasks. However, the performance of VLA-based robots is highly sensitive to the precise wording of language instructions, and it remains difficult to predict when such robots will fail. We propose Quality Diversity (QD) optimization as a natural framework for red-teaming embodied models, and present Q-DIG (Quality Diversity for Diverse Instruction Generation), which performs red-teaming by scalably identifying diverse, natural language task descriptions that induce failures while remaining task-relevant. Q-DIG integrates QD techniques with Vision-Language Models (VLMs) to generate a broad spectrum of adversarial instructions that expose meaningful vulnerabilities in VLA behavior. Our results across multiple simulation benchmarks show that Q-DIG finds more diverse and meaningful failure modes compared to baseline methods, and that fine-tuning VLAs on the generated instructions improves task success rates. Furthermore, results from a user study highlight that Q-DIG generates prompts judged to be more natural and human-like than those from baselines. Finally, real-world evaluations of Q-DIG prompts show results consistent with simulation, and fine-tuning VLAs on the generated prompts further success rates on unseen instructions. Together, these findings suggest that Q-DIG is a promising approach for identifying vulnerabilities and improving the robustness of VLA-based robots. Our anonymous project website is at this http URL.
>
---
#### [replaced 012] Learning to Grasp Anything by Playing with Random Toys
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在解决机器人泛化能力不足的问题。通过使用简单形状的随机玩具训练，提升机器人对新物体的抓取能力。**

- **链接: [https://arxiv.org/pdf/2510.12866](https://arxiv.org/pdf/2510.12866)**

> **作者:** Dantong Niu; Yuvan Sharma; Baifeng Shi; Rachel Ding; Matteo Gioia; Haoru Xue; Henry Tsai; Konstantinos Kallidromitis; Anirudh Pai; Caitlin Regan; Shankar Sastry; Trevor Darrell; Jitendra Malik; Roei Herzig
>
> **摘要:** Robotic manipulation policies often struggle to generalize to novel objects, limiting their real-world utility. In contrast, cognitive science suggests that children develop generalizable dexterous manipulation skills by mastering a small set of simple toys and then applying that knowledge to more complex items. Inspired by this, we study if similar generalization capabilities can also be achieved by robots. Our results indicate robots can learn generalizable grasping using randomly assembled objects that are composed from just four shape primitives: spheres, cuboids, cylinders, and rings. We show that training on these "toys" enables robust generalization to real-world objects, yielding strong zero-shot performance. Crucially, we find the key to this generalization is an object-centric visual representation induced by our proposed detection pooling mechanism. Evaluated in both simulation and on physical robots, our model achieves a 67% real-world grasping success rate on the YCB dataset, outperforming state-of-the-art approaches that rely on substantially more in-domain data. We further study how zero-shot generalization performance scales by varying the number and diversity of training toys and the demonstrations per toy. We believe this work offers a promising path to scalable and generalizable learning in robotic manipulation. Demonstration videos, code, checkpoints and our dataset are available on our project page: this https URL .
>
---
#### [replaced 013] Allometric Scaling Laws for Bipedal Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，研究 bipedal 机器人尺寸缩放问题。通过分析数据和仿真，揭示机器人质量、扭矩等参数与腿长的非相似比例关系，为机器人尺度设计提供新见解。**

- **链接: [https://arxiv.org/pdf/2603.22560](https://arxiv.org/pdf/2603.22560)**

> **作者:** Naomi Oke; Aja M. Carter; Ben Gu; Steven Man; Cordelia Pride; Sarah Bergbreiter; Aaron M. Johnson
>
> **摘要:** Scaling the design of robots up or down remains a fundamental challenge. While biological systems follow well-established isometric and allometric scaling laws relating mass, stride frequency, velocity, and torque, it is unclear how these relationships translate to robotic systems. In this paper, we generate similar allometric scaling laws for bipedal robots across three orders of magnitude in leg length. First, we conduct a review of legged robots from the literature and extract empirical relationships between leg length (L), body length, mass, and speed. These data show that robot mass scales more closely to L^2, in contrast to the L^3 scaling predicted by isometric scaling. We then perform controlled simulation studies in Drake using three variants of real quasi-passive, hip-actuated walkers with different foot geometries and control strategies. We evaluate the performance of each design scaled with leg length, L. Across all robots, walking velocity follows the expected L^(1/2) trend from dynamic similarity. Minimum required torque scales more closely with m*L than the isometric model of m*L^2. Foot geometry scaled proportionally with L^1. These results provide new insight into how robot designs allometrically scale to different sizes, and how that scaling is different from isometric or biological scaling laws.
>
---
#### [replaced 014] Safe Interactions via Monte Carlo Linear-Quadratic Games
- **分类: cs.RO**

- **简介: 该论文属于人机交互安全领域，旨在解决机器人在不确定人类行为下的安全决策问题。通过构建零和博弈模型，提出MCLQ方法，实现高效、安全的机器人策略优化。**

- **链接: [https://arxiv.org/pdf/2504.06124](https://arxiv.org/pdf/2504.06124)**

> **作者:** Benjamin A. Christie; Dylan P. Losey
>
> **摘要:** Safety is critical during human-robot interaction. But -- because people are inherently unpredictable -- it is often difficult for robots to plan safe behaviors. Instead of relying on our ability to anticipate humans, here we identify robot policies that are robust to unexpected human decisions. We achieve this by formulating human-robot interaction as a zero-sum game, where (in the worst case) the human's actions directly conflict with the robot's objective. Solving for the Nash Equilibrium of this game provides robot policies that maximize safety and performance across a wide range of human actions. Existing approaches attempt to find these optimal policies by leveraging Hamilton-Jacobi analysis (which is intractable) or linear-quadratic approximations (which are inexact). By contrast, in this work we propose a computationally efficient and theoretically justified method that converges towards the Nash Equilibrium policy. Our approach (which we call MCLQ) leverages linear-quadratic games to obtain an initial guess at safe robot behavior, and then iteratively refines that guess with a Monte Carlo search. Not only does MCLQ provide real-time safety adjustments, but it also enables the designer to tune how conservative the robot is -- preventing the system from focusing on unrealistic human behaviors. Our simulations and user study suggest that this approach advances safety in terms of both computation time and expected performance. See videos of our experiments here: this https URL.
>
---
#### [replaced 015] Learning Geometry-Aware Nonprehensile Pushing and Pulling with Dexterous Hands
- **分类: cs.RO**

- **简介: 该论文属于非预握操作任务，旨在解决机器人用手抓取困难物体的问题。通过学习几何感知的推拉动作，生成有效的手部姿态，提升操作灵活性与稳定性。**

- **链接: [https://arxiv.org/pdf/2509.18455](https://arxiv.org/pdf/2509.18455)**

> **作者:** Yunshuang Li; Yiyang Ling; Gaurav S. Sukhatme; Daniel Seita
>
> **备注:** Published at International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Nonprehensile manipulation, such as pushing and pulling, enables robots to move, align, or reposition objects that may be difficult to grasp due to their geometry, size, or relationship to the robot or the environment. Much of the existing work in nonprehensile manipulation relies on parallel-jaw grippers or tools such as rods and spatulas. In contrast, multi-fingered dexterous hands offer richer contact modes and versatility for handling diverse objects to provide stable support over the objects, which compensates for the difficulty of modeling the dynamics of nonprehensile manipulation. Therefore, we propose Geometry-aware Dexterous Pushing and Pulling(GD2P) for nonprehensile manipulation with dexterous robotic hands. We study pushing and pulling by framing the problem as synthesizing and learning pre-contact dexterous hand poses that lead to effective manipulation. We generate diverse hand poses via contact-guided sampling, filter them using physics simulation, and train a diffusion model conditioned on object geometry to predict viable poses. At test time, we sample hand poses and use standard motion planners to select and execute pushing and pulling actions. We perform extensive real-world experiments with an Allegro Hand and a LEAP Hand, demonstrating that GD2P offers a scalable route for generating dexterous nonprehensile manipulation motions with its applicability to different hand morphologies. Our project website is available at: this http URL.
>
---
#### [replaced 016] Towards Safe and Robust Autonomous Vehicle Platooning: A Self-Organizing Cooperative Control Framework
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶车辆编队任务，旨在解决混合交通环境下编队的安全与鲁棒性问题。提出TriCoD框架，融合深度强化学习与模型驱动方法，提升动态适应能力。**

- **链接: [https://arxiv.org/pdf/2408.09468](https://arxiv.org/pdf/2408.09468)**

> **作者:** Chengkai Xu; Zihao Deng; Jiaqi Liu; Aijing Kong; Yu Tang; Chao Huang; Peng Hang
>
> **摘要:** In hybrid traffic environments where human-driven vehicles (HDVs) and autonomous vehicles (AVs) coexist, achieving safe and robust decision-making for AV platooning remains a complex challenge. Existing platooning systems often struggle with dynamic formation management and adaptability, especially under complex and dynamic mixed-traffic conditions. To enhance autonomous vehicle platooning within these hybrid environments, this paper presents TriCoD, a twin-world safety-enhanced Data-Model-Knowledge Triple-Driven Cooperative Decision-making Framework. This framework integrates deep reinforcement learning (DRL) with model-driven approaches, enabling dynamic formation dissolution and reconfiguration through a safety-prioritized twin-world deduction mechanism. The DRL component augments traditional model-driven methods, enhancing both safety and operational efficiency, especially under emergency conditions. Additionally, an adaptive switching mechanism allows the system to seamlessly switch between data-driven and model-driven strategies based on real-time traffic demands, thus optimizing decision-making ability and adaptability. Simulation experiments and hardware-in-the-loop tests demonstrate that the proposed framework significantly improves safety, robustness, and flexibility.
>
---
#### [replaced 017] From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决未知场景下机械臂的推理与决策问题。提出FSD模型，通过空间关系推理生成细粒度操作指导，提升零样本泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.08548](https://arxiv.org/pdf/2505.08548)**

> **作者:** Yifu Yuan; Haiqin Cui; Yibin Chen; Zibin Dong; Fei Ni; Longxin Kou; Jinyi Liu; Pengyi Li; Yan Zheng; Jianye Hao
>
> **备注:** Published as a conference paper at ICLR 2026. Our project homepage: this https URL
>
> **摘要:** Achieving generalization in robotic manipulation remains a critical challenge, particularly for unseen scenarios and novel tasks. Current Vision-Language-Action (VLA) models, while building on top of general Vision-Language Models (VLMs), still fall short of achieving robust zero-shot performance due to the scarcity and heterogeneity prevalent in embodied datasets. To address these limitations, we propose FSD (From Seeing to Doing), a novel vision-language model that generates intermediate representations through spatial relationship reasoning, providing fine-grained guidance for robotic manipulation. Our approach combines a hierarchical data pipeline for training with a self-consistency mechanism that aligns spatial coordinates with visual signals. Through extensive experiments, we comprehensively validated FSD's capabilities in both "seeing" and "doing," achieving outstanding performance across 8 benchmarks for general spatial reasoning and embodied reference abilities, as well as on our proposed more challenging benchmark VABench. We also verified zero-shot capabilities in robot manipulation, demonstrating significant performance improvements over baseline methods in both SimplerEnv and real robot settings. Experimental results show that FSD achieves 40.6% success rate in SimplerEnv and 72% success rate across 8 real-world tasks, outperforming the strongest baseline by 30%.
>
---
#### [replaced 018] Certified Training with Branch-and-Bound for Lyapunov-stable Neural Control
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于神经控制稳定性验证任务，旨在解决神经控制器在特定区域内Lyapunov稳定性的验证问题。通过引入CT-BaB框架，优化认证边界，提升验证效率与稳定性区域。**

- **链接: [https://arxiv.org/pdf/2411.18235](https://arxiv.org/pdf/2411.18235)**

> **作者:** Zhouxing Shi; Haoyu Li; Cho-Jui Hsieh; Huan Zhang
>
> **备注:** L4DC 2026
>
> **摘要:** We study the problem of learning verifiably Lyapunov-stable neural controllers that provably satisfy the Lyapunov asymptotic stability condition within a region-of-attraction (ROA). Unlike previous works that adopted counterexample-guided training without considering the computation of verification in training, we introduce Certified Training with Branch-and-Bound (CT-BaB), a new certified training framework that optimizes certified bounds, thereby reducing the discrepancy between training and test-time verification that also computes certified bounds. To achieve a relatively global guarantee on an entire input region-of-interest, we propose a training-time BaB technique that maintains a dynamic training dataset and adaptively splits hard input subregions into smaller ones, to tighten certified bounds and ease the training. Meanwhile, subregions created by the training-time BaB also inform test-time verification, for a more efficient training-aware verification. We demonstrate that CT-BaB yields verification-friendly models that can be more efficiently verified at test time while achieving stronger verifiable guarantees with larger ROA. On the largest output-feedback 2D Quadrotor system experimented, CT-BaB reduces verification time by over 11X relative to the previous state-of-the-art baseline using Counterexample Guided Inductive Synthesis (CEGIS), while achieving 164X larger ROA. Code is available at this https URL.
>
---
#### [replaced 019] Fine-tuning is Not Enough: A Parallel Framework for Collaborative Imitation and Reinforcement Learning in End-to-end Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决传统模仿学习性能受限问题。提出PaIR-Drive框架，实现模仿与强化学习并行优化，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2603.13842](https://arxiv.org/pdf/2603.13842)**

> **作者:** Zhexi Lian; Haoran Wang; Xuerun Yan; Weimeng Lin; Xianhong Zhang; Yongyu Chen; Jia Hu
>
> **备注:** 11 pages, 7 figures, 6 tables
>
> **摘要:** End-to-end autonomous driving is typically built upon imitation learning (IL), yet its performance is constrained by the quality of human demonstrations. To overcome this limitation, recent methods incorporate reinforcement learning (RL) through sequential fine-tuning. However, such a paradigm remains suboptimal: sequential RL fine-tuning can introduce policy drift and often leads to a performance ceiling due to its dependence on the pretrained IL policy. To address these issues, we propose PaIR-Drive, a general Parallel framework for collaborative Imitation and Reinforcement learning in end-to-end autonomous driving. During training, PaIR-Drive separates IL and RL into two parallel branches with conflict-free training objectives, enabling fully collaborative optimization. This design eliminates the need to retrain RL when applying a new IL policy. During inference, RL leverages the IL policy to further optimize the final plan, allowing performance beyond prior knowledge of IL. Furthermore, we introduce a tree-structured trajectory neural sampler to group relative policy optimization (GRPO) in the RL branch, which enhances exploration capability. Extensive analysis on NAVSIMv1 and v2 benchmark demonstrates that PaIR-Drive achieves Competitive performance of 91.2 PDMS and 87.9 EPDMS, building upon Transfuser and DiffusionDrive IL baselines. PaIR-Drive consistently outperforms existing RL fine-tuning methods, and could even correct human experts' suboptimal behaviors. Qualitative results further confirm that PaIR-Drive can effectively explore and generate high-quality trajectories.
>
---
#### [replaced 020] VERDI: VLM-Embedded Reasoning for Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决部分可观测性和复杂现实环境下的决策问题。通过将VLM的推理过程和常识知识融入AD系统，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2505.15925](https://arxiv.org/pdf/2505.15925)**

> **作者:** Bowen Feng; Zhiting Mei; Julian Ost; Filippo Ghilotti; Baiang Li; Roger Girgis; Anirudha Majumdar; Felix Heide
>
> **摘要:** While autonomous driving (AD) stacks struggle with decision making under partial observability and real-world complexity, human drivers are capable of applying commonsense reasoning to make near-optimal decisions with limited information. Recent work has attempted to leverage finetuned Vision-Language Models (VLMs) for trajectory planning at inference time to emulate human behavior. Despite their success in benchmark evaluations, these methods are often impractical to deploy (a 70B parameter VLM inference at merely 8 tokens per second requires more than 160G of memory), and their monolithic network structure prohibits safety decomposition. To bridge this gap, we propose VLM-Embedded Reasoning for autonomous DrIving (VERDI), a training-time framework that distills the reasoning process and commonsense knowledge of VLMs into the AD stack. VERDI augments modular differentiable end-to-end (e2e) AD models by aligning intermediate module outputs at the perception, prediction, and planning stages with text features explaining the driving reasoning process produced by VLMs. By encouraging alignment in latent space, VERDI enables the modular AD stack to internalize structured reasoning, without incurring the inference-time costs of large VLMs. We evaluate VERDI in both open-loop and closed-loop settings. Our method outperforms existing end-to-end approaches without embedded reasoning by up to 11% in $\ell_{2}$ distance, and achieves the best overall driving performance in the closed-loop HugSim simulator, including a 10% improvement in Non-Collision Rate, while maintaining fast inference speed.
>
---
#### [replaced 021] Informed Hybrid Zonotope-based Motion Planning Algorithm
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决非凸自由空间中的最优路径规划问题。提出HZ-MP算法，通过分解空间和低维采样提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2507.09309](https://arxiv.org/pdf/2507.09309)**

> **作者:** Peng Xie; Johannes Betz; Amr Alanwar
>
> **摘要:** Optimal path planning in nonconvex free spaces poses substantial computational challenges. A common approach formulates such problems as mixed-integer linear programs (MILPs); however, solving general MILPs is computationally intractable and severely limits scalability. To address these limitations, we propose HZ-MP, an informed Hybrid Zonotope-based Motion Planner, which decomposes the obstacle-free space and performs low-dimensional face sampling guided by an ellipsotope heuristic, thereby concentrating exploration on promising transition regions. This structured exploration mitigates the excessive wasted sampling that degrades existing informed planners in narrow-passage or enclosed-goal scenarios. We prove that HZ-MP is probabilistically complete and asymptotically optimal, and demonstrate empirically that it converges to high-quality trajectories within a small number of iterations.
>
---
#### [replaced 022] ActDistill: General Action-Guided Self-Derived Distillation for Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ActDistill，解决VLA模型计算开销大、推理延迟高的问题，通过知识蒸馏将动作预测能力迁移到轻量模型，提升效率。**

- **链接: [https://arxiv.org/pdf/2511.18082](https://arxiv.org/pdf/2511.18082)**

> **作者:** Wencheng Ye; Tianshi Wang; Lei Zhu; Fengling Li; Guoli Yang; Hengtao Shen
>
> **摘要:** Recent Vision-Language-Action (VLA) models have shown impressive flexibility and generalization, yet their deployment in robotic manipulation remains limited by heavy computational overhead and inference latency. In this work, we present ActDistill, a general action-guided self-derived distillation framework that transfers the action prediction capability of any existing VLA model to a lightweight counterpart. Unlike previous efficiency strategies that primarily emphasize vision-language correlations, ActDistill leverages action priors to guide knowledge transfer and model compression, achieving action-oriented efficiency for VLA models. Specifically, we employ a well-trained VLA model as the teacher and introduce a graph-structured encapsulation strategy to explicitly model the hierarchical evolution of action prediction. The student model, derived from the graph-encapsulated teacher, is further equipped with a dynamic router that adaptively selects computation paths based on action prediction demands, guided by hierarchical graph-informed supervision to ensure smooth and efficient evolution. During inference, graph-related auxiliary components are removed, allowing the student to execute only dynamically routed layers and predict high-precision actions with minimal computation and latency. Experiments on embodied benchmarks demonstrate that ActDistill achieves comparable or superior performance to full-scale VLA models while reducing computation by over 50% with up to 1.67 times speedup, thereby establishing a general paradigm toward efficient embodied intelligence.
>
---
#### [replaced 023] RAPTOR: A Foundation Policy for Quadrotor Control
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出RAPTOR方法，解决四旋翼无人机控制的适应性问题。通过元模仿学习训练通用基础策略，实现对多种平台的零样本适应。**

- **链接: [https://arxiv.org/pdf/2509.11481](https://arxiv.org/pdf/2509.11481)**

> **作者:** Jonas Eschmann; Dario Albani; Giuseppe Loianno
>
> **摘要:** Humans are remarkably data-efficient when adapting to new unseen conditions, like driving a new car. In contrast, modern robotic control systems, like neural network policies trained using Reinforcement Learning (RL), are highly specialized for single environments. Because of this overfitting, they are known to break down even under small differences like the Simulation-to-Reality (Sim2Real) gap and require system identification and retraining for even minimal changes to the system. In this work, we present RAPTOR, a method for training a highly adaptive foundation policy for quadrotor control. Our method enables training a single, end-to-end neural-network policy to control a wide variety of quadrotors. We test 10 different real quadrotors from 32 g to 2.4 kg that also differ in motor type (brushed vs. brushless), frame type (soft vs. rigid), propeller type (2/3/4-blade), and flight controller (PX4/Betaflight/Crazyflie/M5StampFly). We find that a tiny, three-layer policy with only 2084 parameters is sufficient for zero-shot adaptation to a wide variety of platforms. The adaptation through in-context learning is made possible by using a recurrence in the hidden layer. The policy is trained through our proposed Meta-Imitation Learning algorithm, where we sample 1000 quadrotors and train a teacher policy for each of them using RL. Subsequently, the 1000 teachers are distilled into a single, adaptive student policy. We find that within milliseconds, the resulting foundation policy adapts zero-shot to unseen quadrotors. We extensively test the capabilities of the foundation policy under numerous conditions (trajectory tracking, indoor/outdoor, wind disturbance, poking, different propellers).
>
---
#### [replaced 024] SERNF: Sample-Efficient Real-World Dexterous Policy Fine-Tuning via Action-Chunked Critics and Normalizing Flows
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操作任务，解决真实世界中精细操控策略微调的样本效率问题。提出SERFN框架，结合归一化流和动作块评论器，提升多模态动作分布下的策略更新稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2602.09580](https://arxiv.org/pdf/2602.09580)**

> **作者:** Chenyu Yang; Denis Tarasov; Davide Liconti; Hehui Zheng; Robert K. Katzschmann
>
> **备注:** this https URL
>
> **摘要:** Real-world fine-tuning of dexterous manipulation policies remains challenging due to limited real-world interaction budgets and highly multimodal action distributions. Diffusion-based policies, while expressive, do not permit conservative likelihood-based updates during fine-tuning because action probabilities are intractable. In contrast, conventional Gaussian policies collapse under multimodality, particularly when actions are executed in chunks, and standard per-step critics fail to align with chunked execution, leading to poor credit assignment. We present SERFN, a sample-efficient off-policy fine-tuning framework with normalizing flow (NF) to address these challenges. The normalizing flow policy yields exact likelihoods for multimodal action chunks, allowing conservative, stable policy updates through likelihood regularization and thereby improving sample efficiency. An action-chunked critic evaluates entire action sequences, aligning value estimation with the policy's temporal structure and improving long-horizon credit assignment. To our knowledge, this is the first demonstration of a likelihood-based, multimodal generative policy combined with chunk-level value learning on real robotic hardware. We evaluate SERFN on two challenging dexterous manipulation tasks in the real world: cutting tape with scissors retrieved from a case, and in-hand cube rotation with a palm-down grasp -- both of which require precise, dexterous control over long horizons. On these tasks, SERFN achieves stable, sample-efficient adaptation where standard methods struggle.
>
---
#### [replaced 025] PlayWorld: Learning Robot World Models from Autonomous Play
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PlayWorld，用于训练高保真机器人世界模型。任务是提升机器人操作中的物理一致性预测。通过自主玩耍数据训练，解决传统方法依赖人类示范的不足。**

- **链接: [https://arxiv.org/pdf/2603.09030](https://arxiv.org/pdf/2603.09030)**

> **作者:** Tenny Yin; Zhiting Mei; Zhonghe Zheng; Miyu Yamane; David Wang; Jade Sceats; Samuel M. Bateman; Lihan Zha; Apurva Badithela; Ola Shorinwa; Anirudha Majumdar
>
> **备注:** Website: this https URL
>
> **摘要:** Action-conditioned video models offer a promising path to building general-purpose robot simulators that can improve directly from data. Yet, despite training on large-scale robot datasets, current state-of-the-art video models still struggle to predict physically consistent robot-object interactions that are crucial in robotic manipulation. To close this gap, we present PlayWorld, a simple, scalable, and fully autonomous pipeline for training high-fidelity video world simulators from interaction experience. In contrast to prior approaches that rely on success-biased human demonstrations, PlayWorld is the first system capable of learning entirely from unsupervised robot self-play, enabling naturally scalable data collection while capturing complex, long-tailed physical interactions essential for modeling realistic object dynamics. Experiments across diverse manipulation tasks show that PlayWorld generates high-quality, physically consistent predictions for contact-rich interactions that are not captured by world models trained on human-collected data. We further demonstrate the versatility of PlayWorld in enabling fine-grained failure prediction and policy evaluation, with up to 40% improvements over human-collected data. Finally, we demonstrate how PlayWorld enables reinforcement learning in the world model, improving policy performance by 65% in success rates when deployed in the real world.
>
---
#### [replaced 026] PalpAid: Multimodal Pneumatic Tactile Sensor for Tissue Palpation
- **分类: eess.SP; cs.RO**

- **简介: 论文提出PalpAid，一种用于机器人手术中恢复触觉的多模态气动触觉传感器，解决手术中触觉信息缺失问题。通过压力与声音检测实现组织识别与分类。**

- **链接: [https://arxiv.org/pdf/2512.19010](https://arxiv.org/pdf/2512.19010)**

> **作者:** Devi Yuliarti; Ravi Prakash; Hiu Ching Cheung; Amy Strong; Patrick J. Codd; Shan Lin
>
> **备注:** IEEE-RAS RoboSoft 2026
>
> **摘要:** The tactile properties of tissue, such as elasticity and stiffness, often play an important role in surgical oncology when identifying tumors and pathological tissue boundaries. Though extremely valuable, robot-assisted surgery comes at the cost of reduced sensory information to the surgeon, with vision being the primary. Sensors proposed to overcome this sensory desert are often bulky, complex, and incompatible with the surgical workflow. We present PalpAid, a multimodal pneumatic tactile sensor to restore touch in robot-assisted surgery. PalpAid is equipped with a microphone and pressure sensor, converting contact force into an internal pressure differential. The pressure sensor acts as an event detector, while the acoustic signature assists in tissue identification. We show the design, fabrication, and assembly of sensory units with characterization tests for robustness to use, repetition cycles, and integration with a robotic system. Finally, we demonstrate the sensor's ability to classify 3D-printed hard objects with varying infills and soft ex vivo tissues. We envision PalpAid to be easily retrofitted with existing surgical/general robotic systems, allowing soft tissue palpation.
>
---
#### [replaced 027] C-NAV: Towards Self-Evolving Continual Object Navigation in Open World
- **分类: cs.RO**

- **简介: 该论文属于持续对象导航任务，解决开放世界中代理需不断适应新物体类别并避免遗忘旧知识的问题。提出C-Nav框架，结合防遗忘机制和自适应采样策略，提升导航性能并降低内存需求。**

- **链接: [https://arxiv.org/pdf/2510.20685](https://arxiv.org/pdf/2510.20685)**

> **作者:** Ming-Ming Yu; Fei Zhu; Wenzhuo Liu; Yirong Yang; Qunbo Wang; Wenjun Wu; Jing Liu
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Embodied agents are expected to perform object navigation in dynamic, open-world environments. However, existing approaches typically rely on static trajectories and a fixed set of object categories during training, overlooking the real-world requirement for continual adaptation to evolving scenarios. To facilitate related studies, we introduce the continual object navigation benchmark, which requires agents to acquire navigation skills for new object categories while avoiding catastrophic forgetting of previously learned knowledge. To tackle this challenge, we propose C-Nav, a continual visual navigation framework that integrates two key innovations: (1) A dual-path anti-forgetting mechanism, which comprises feature distillation that aligns multi-modal inputs into a consistent representation space to ensure representation consistency, and feature replay that retains temporal features within the action decoder to ensure policy consistency. (2) An adaptive sampling strategy that selects diverse and informative experiences, thereby reducing redundancy and minimizing memory overhead. Extensive experiments across multiple model architectures demonstrate that C-Nav consistently outperforms existing approaches, achieving superior performance even compared to baselines with full trajectory retention, while significantly lowering memory requirements. The code will be publicly available at this https URL.
>
---
#### [replaced 028] Distributed Event-Triggered Distance-Based Formation Control for Multi-Agent Systems
- **分类: cs.RO**

- **简介: 该论文属于多智能体系统协同控制任务，解决资源受限下的 formation 控制问题。提出一种基于距离的事件触发控制器，减少控制更新，确保稳定性和避撞。**

- **链接: [https://arxiv.org/pdf/2509.12390](https://arxiv.org/pdf/2509.12390)**

> **作者:** Evangelos Psomiadis; Panagiotis Tsiotras
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** This paper addresses the problem of collaborative formation control for multi-agent systems with limited resources. We consider a team of robots tasked with achieving a desired formation from an arbitrary initial configuration. To reduce unnecessary control updates and conserve resources, we propose a distributed event-triggered formation controller. Unlike the well-studied linear formation control strategies, the proposed controller is nonlinear and relies on inter-agent distance measurements. Control updates are triggered only when the measurement error exceeds a predefined threshold, ensuring system stability while minimizing actuation effort. We also employ a distributed control barrier function to guarantee inter-agent collision avoidance. The proposed controller is validated through extensive simulations and real-world experiments involving different formations, communication topologies, scalability tests, and variations in design parameters, while also being compared against periodic triggering strategies. Results demonstrate that the event-triggered approach significantly reduces control effort while preserving formation performance.
>
---
#### [replaced 029] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于长期机器人操作任务，解决多步骤任务中进展感知不足的问题。提出PALM框架，通过 affordance 理解和子任务进度预测提升执行稳定性与成功率。**

- **链接: [https://arxiv.org/pdf/2601.07060](https://arxiv.org/pdf/2601.07060)**

> **作者:** Yuanzhe Liu; Jingyuan Zhu; Yuchen Mo; Gen Li; Xu Cao; Jin Jin; Yifan Shen; Zhengyuan Li; Tianjiao Yu; Wenzhen Yuan; Fangqiang Ding; Ismini Lourentzou
>
> **备注:** CVPR 2026
>
> **摘要:** Recent advancements in vision-language-action (VLA) models have shown promise in robotic manipulation, yet they continue to struggle with long-horizon, multi-step tasks. Existing methods lack internal reasoning mechanisms that can identify task-relevant interaction cues or track progress within a subtask, leading to critical execution errors such as repeated actions, missed steps, and premature termination. To address these challenges, we introduce PALM, a VLA framework that structures policy learning around interaction-centric affordance reasoning and subtask progress cues. PALM distills complementary affordance representations that capture object relevance, contact geometry, spatial placements, and motion dynamics, and serve as task-relevant anchors for visuomotor control. To further stabilize long-horizon execution, PALM predicts continuous within-subtask progress, enabling seamless subtask transitions. Across extensive simulation and real-world experiments, PALM consistently outperforms baselines, achieving a 91.8% success rate on LIBERO-LONG, a 12.5% improvement in average length on CALVIN ABC->D, and a 2x improvement over real-world baselines across three long-horizon generalization settings.
>
---
#### [replaced 030] Learning Sampled-data Control for Swarms via MeanFlow
- **分类: cs.LG; cs.MA; cs.RO; eess.SY**

- **简介: 该论文属于控制任务，解决有限控制更新下的群体轨迹规划问题。通过改进MeanFlow框架，提出一种新的采样数据学习方法，实现高效群体控制。**

- **链接: [https://arxiv.org/pdf/2603.20189](https://arxiv.org/pdf/2603.20189)**

> **作者:** Anqi Dong; Yongxin Chen; Karl H. Johansson; Johan Karlsson
>
> **摘要:** Steering large-scale swarms with only limited control updates is often needed due to communication or computational constraints, yet most learning-based approaches do not account for this and instead model instantaneous velocity fields. As a result, the natural object for decision making is a finite-window control quantity rather than an infinitesimal one. To address this gap, we consider the recent machine learning framework MeanFlow and generalize it to the setting with general linear dynamic systems. This results in a new sampled-data learning framework that operates directly in control space and that can be applied for swarm steering. To this end, we learn the finite-horizon coefficient that parameterizes the minimum-energy control applied over each interval, and derive a differential identity that connects this quantity to a local bridge-induced supervision signal. This identity leads to a simple stop-gradient regression objective, allowing the interval coefficient field to be learned efficiently from bridge samples. The learned policy is deployed through sampled-data updates, guaranteeing that the resulting controller exactly respects the prescribed linear time-invariant dynamics and actuation channel. The resulting method enables few-step swarm steering at scale, while remaining consistent with the finite-window actuation structure of the underlying control system.
>
---
#### [replaced 031] Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决“感知-行动差距”问题。通过引入“指向”作为统一表示，训练Embodied-R1模型，提升机器人在不同场景下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2508.13998](https://arxiv.org/pdf/2508.13998)**

> **作者:** Yifu Yuan; Haiqin Cui; Yaoting Huang; Yibin Chen; Fei Ni; Zibin Dong; Pengyi Li; Yan Zheng; Hongyao Tang; Jianye Hao
>
> **备注:** Embodied-R1 technical report v2; Published as a conference paper at ICLR 2026
>
> **摘要:** Generalization in embodied AI is hindered by the "seeing-to-doing gap," which stems from data scarcity and embodiment heterogeneity. To address this, we pioneer "pointing" as a unified, embodiment-agnostic intermediate representation, defining four core embodied pointing abilities that bridge high-level vision-language comprehension with low-level action primitives. We introduce Embodied-R1, a 3B Vision-Language Model (VLM) specifically designed for embodied reasoning and pointing. We use a wide range of embodied and general visual reasoning datasets as sources to construct a large-scale dataset, Embodied-Points-200K, which supports key embodied pointing capabilities. We then train Embodied-R1 using a two-stage Reinforced Fine-tuning (RFT) curriculum with a specialized multi-task reward design. Embodied-R1 achieves state-of-the-art performance on 11 embodied spatial and pointing benchmarks. Critically, it demonstrates robust zero-shot generalization by achieving a 56.2% success rate in the SIMPLEREnv and 87.5% across 8 real-world XArm tasks without any task-specific fine-tuning, representing a 62% improvement over strong baselines. Furthermore, the model exhibits high robustness against diverse visual disturbances. Our work shows that a pointing-centric representation, combined with an RFT training paradigm, offers an effective and generalizable pathway to closing the perception-action gap in robotics.
>
---
#### [replaced 032] Acoustic Feedback for Closed-Loop Force Control in Robotic Grinding
- **分类: cs.RO**

- **简介: 该论文属于机器人打磨任务，旨在解决传统力传感成本高、适应性差的问题。通过音频反馈实现闭环力控制，使用低成本麦克风替代力传感器。**

- **链接: [https://arxiv.org/pdf/2602.20596](https://arxiv.org/pdf/2602.20596)**

> **作者:** Zongyuan Zhang; Christopher Lehnert; Will N. Browne; Jonathan M. Roberts
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. 8 pages, 10 figures. Video demonstration: this https URL
>
> **摘要:** Acoustic feedback is a critical indicator for assessing the contact condition between the tool and the workpiece when humans perform grinding tasks with rotary tools. In contrast, robotic grinding systems typically rely on force sensing, with acoustic information largely ignored. This reliance on force sensors is costly and difficult to adapt to different grinding tools, whereas audio sensors (microphones) are low-cost and can be mounted on any medium that conducts grinding sound. This paper introduces a low-cost Acoustic Feedback Robotic Grinding System (AFRG) that captures audio signals with a contact microphone, estimates grinding force from the audio in real time, and enables closed-loop force control of the grinding process. Compared with conventional force-sensing approaches, AFRG achieves a 4-fold improvement in consistency across different grinding disc conditions. AFRG relies solely on a low-cost microphone, which is approximately 200-fold cheaper than conventional force sensors, as the sensing modality, providing an easily deployable, cost-effective robotic grinding solution.
>
---
#### [replaced 033] Anti-bullying Adaptive Cruise Control: A proactive right-of-way protection approach
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决"道路霸凌"式变道问题。提出AACC方法，通过逆最优控制和博弈论，实现对不同驾驶风格的适应性右路保护，提升安全与舒适性。**

- **链接: [https://arxiv.org/pdf/2412.12197](https://arxiv.org/pdf/2412.12197)**

> **作者:** Jia Hu; Zhexi Lian; Haoran Wang; Zihan Zhang; Ruoxi Qian; Duo Li; Jaehyun; Junnian Zheng
>
> **备注:** 16 pages, 19 figures
>
> **摘要:** Adaptive Cruise Control (ACC) systems have been widely commercialized in recent years. However, existing ACC systems remain vulnerable to close-range cut-ins, a behavior that resembles "road bullying". To address this issue, this research proposes an Anti-bullying Adaptive Cruise Control (AACC) approach, which is capable of proactively protecting right-of-way against such "road bullying" cut-ins. To handle diverse "road bullying" cut-in scenarios smoothly, the proposed approach first leverages an online Inverse Optimal Control (IOC) based algorithm for individual driving style identification. Then, based on Stackelberg competition, a game-theoretic-based motion planning framework is presented in which the identified individual driving styles are utilized to formulate cut-in vehicles' reaction functions. By integrating such reaction functions into the ego vehicle's motion planning, the ego vehicle could consider cut-in vehicles' all possible reactions to find its optimal right-of-way protection maneuver. To the best of our knowledge, this research is the first to model vehicles' interaction dynamics and develop an interactive planner that adapts cut-in vehicle's various driving styles. Simulation results show that the proposed approach can prevent "road bullying" cut-ins and be adaptive to different cut-in vehicles' driving styles. It can improve safety and comfort by up to 79.8% and 20.4%. The driving efficiency has benefits by up to 19.33% in traffic flow. The proposed approach can also adopt more flexible driving strategies. Furthermore, the proposed approach can support real-time field implementation by ensuring less than 50 milliseconds computation time.
>
---
#### [replaced 034] Steerable Vision-Language-Action Policies for Embodied Reasoning and Hierarchical Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决如何有效将视觉语言模型的知识应用于机器人行为的问题。通过引入可引导策略，提升低层控制能力，增强任务泛化性能。**

- **链接: [https://arxiv.org/pdf/2602.13193](https://arxiv.org/pdf/2602.13193)**

> **作者:** William Chen; Jagdeep Singh Bhatia; Catherine Glossop; Nikhil Mathihalli; Ria Doshi; Andy Tang; Danny Driess; Karl Pertsch; Sergey Levine
>
> **摘要:** Pretrained vision-language models (VLMs) can make semantic and visual inferences across diverse settings, providing valuable common-sense priors for robotic control. However, effectively grounding this knowledge in robot behaviors remains an open challenge. Prior methods often employ a hierarchical approach where VLMs reason over high-level commands to be executed by separate low-level policies, e.g., vision-language-action models (VLAs). The interface between VLMs and VLAs is usually natural language task instructions, which fundamentally limits how much VLM reasoning can steer low-level behavior. We thus introduce Steerable Policies: VLAs trained on rich synthetic commands at various levels of abstraction, like subtasks, motions, and grounded pixel coordinates. By improving low-level controllability, Steerable Policies can unlock pretrained knowledge in VLMs, enabling improved task generalization. We demonstrate this benefit by controlling our Steerable Policies with both a learned high-level embodied reasoner and an off-the-shelf VLM prompted to reason over command abstractions via in-context learning. Across extensive real-world manipulation experiments, these two novel methods outperform prior embodied reasoning VLAs and VLM-based hierarchical baselines, including on challenging generalization and long-horizon tasks. Website: this http URL
>
---
#### [replaced 035] Low-Cost Teleoperation Extension for Mobile Manipulators
- **分类: cs.RO**

- **简介: 该论文属于机器人 teleoperation 任务，旨在解决高成本问题。使用低成本硬件实现移动机械臂的直观控制，通过手机、手柄和脚踏板组合，提升操作效率与用户体验。**

- **链接: [https://arxiv.org/pdf/2603.07672](https://arxiv.org/pdf/2603.07672)**

> **作者:** Danil Belov; Artem Erkhov; Yaroslav Savotin; Tatiana Podladchikova; Pavel Osinenko; Dzmitry Tsetserukou
>
> **摘要:** Teleoperation of mobile bimanual manipulators requires simultaneous control of high-dimensional systems, often necessitating expensive specialized equipment. We present an open-source teleoperation framework that enables intuitive whole body control using readily available commodity hardware. Our system combines smartphone-based head tracking for camera control, leader arms for bilateral manipulation, and foot pedals for hands-free base navigation. Using a standard smartphone with IMU and display, we eliminate the need for costly VR helmets while maintaining immersive visual feedback. The modular architecture integrates seamlessly with the XLeRobot framework, but can be easily adapted to other types of mobile manipulators. We validate our approach through user studies that demonstrate improved task performance and reduced cognitive load compared to keyboard-based control.
>
---
