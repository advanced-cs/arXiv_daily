# 机器人 cs.RO

- **最新发布 41 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] Advances and Innovations in the Multi-Agent Robotic System (MARS) Challenge
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于多智能体机器人系统任务，旨在解决多智能体协作中的规划与控制问题。通过MARS挑战，探索视觉语言模型在动态环境中的协同操作与任务执行。**

- **链接: [https://arxiv.org/pdf/2601.18733v1](https://arxiv.org/pdf/2601.18733v1)**

> **作者:** Li Kang; Heng Zhou; Xiufeng Song; Rui Li; Bruno N. Y. Chen; Ziye Wang; Ximeng Meng; Stone Tao; Yiran Qin; Xiaohong Liu; Ruimao Zhang; Lei Bai; Yilun Du; Hao Su; Philip Torr; Zhenfei Yin; Ruihao Gong; Yejun Zeng; Fengjun Zhong; Shenghao Jin; Jinyang Guo; Xianglong Liu; Xiaojun Jia; Tianqi Shan; Wenqi Ren; Simeng Qin; Jialing Yang; Xiaoyu Ma; Tianxing Chen; Zixuan Li; Zijian Cai; Yan Qin; Yusen Qin; Qiangyu Chen; Kaixuan Wang; Zhaoming Han; Yao Mu; Ping Luo; Yuanqi Yao; Haoming Song; Jan-Nico Zaech; Fabien Despinoy; Danda Pani Paudel; Luc Van Gool
>
> **备注:** MARS Challenge @ NeurIPS 2025 Workshop on Space in Vision, Language, and Embodied AI. Challenge page: https://mars-eai.github.io/MARS-Challenge-Webpage/
>
> **摘要:** Recent advancements in multimodal large language models and vision-languageaction models have significantly driven progress in Embodied AI. As the field transitions toward more complex task scenarios, multi-agent system frameworks are becoming essential for achieving scalable, efficient, and collaborative solutions. This shift is fueled by three primary factors: increasing agent capabilities, enhancing system efficiency through task delegation, and enabling advanced human-agent interactions. To address the challenges posed by multi-agent collaboration, we propose the Multi-Agent Robotic System (MARS) Challenge, held at the NeurIPS 2025 Workshop on SpaVLE. The competition focuses on two critical areas: planning and control, where participants explore multi-agent embodied planning using vision-language models (VLMs) to coordinate tasks and policy execution to perform robotic manipulation in dynamic environments. By evaluating solutions submitted by participants, the challenge provides valuable insights into the design and coordination of embodied multi-agent systems, contributing to the future development of advanced collaborative AI systems.
>
---
#### [new 002] DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决LLM在复杂环境中导航时因局部错误导致的路径偏差问题。提出DV-VLN框架，通过双重验证机制提升导航可靠性。**

- **链接: [https://arxiv.org/pdf/2601.18492v1](https://arxiv.org/pdf/2601.18492v1)**

> **作者:** Zijun Li; Shijie Li; Zhenxi Zhang; Bin Li; Shoujun Zhou
>
> **摘要:** Vision-and-Language Navigation (VLN) requires an embodied agent to navigate in a complex 3D environment according to natural language instructions. Recent progress in large language models (LLMs) has enabled language-driven navigation with improved interpretability. However, most LLM-based agents still rely on single-shot action decisions, where the model must choose one option from noisy, textualized multi-perspective observations. Due to local mismatches and imperfect intermediate reasoning, such decisions can easily deviate from the correct path, leading to error accumulation and reduced reliability in unseen environments. In this paper, we propose DV-VLN, a new VLN framework that follows a generate-then-verify paradigm. DV-VLN first performs parameter-efficient in-domain adaptation of an open-source LLaMA-2 backbone to produce a structured navigational chain-of-thought, and then verifies candidate actions with two complementary channels: True-False Verification (TFV) and Masked-Entity Verification (MEV). DV-VLN selects actions by aggregating verification successes across multiple samples, yielding interpretable scores for reranking. Experiments on R2R, RxR (English subset), and REVERIE show that DV-VLN consistently improves over direct prediction and sampling-only baselines, achieving competitive performance among language-only VLN agents and promising results compared with several cross-modal systems.Code is available at https://github.com/PlumJun/DV-VLN.
>
---
#### [new 003] EMPM: Embodied MPM for Modeling and Simulation of Deformable Objects
- **分类: cs.RO**

- **简介: 该论文提出EMPM，用于建模和模拟变形物体，解决物理真实、泛化性强且数据高效的问题。通过结合多视角RGB-D视频与MPM物理引擎，实现动态仿真与参数优化。**

- **链接: [https://arxiv.org/pdf/2601.17251v1](https://arxiv.org/pdf/2601.17251v1)**

> **作者:** Yunuo Chen; Yafei Hu; Lingfeng Sun; Tushar Kusnur; Laura Herlant; Chenfanfu Jiang
>
> **摘要:** Modeling deformable objects - especially continuum materials - in a way that is physically plausible, generalizable, and data-efficient remains challenging across 3D vision, graphics, and robotic manipulation. Many existing methods oversimplify the rich dynamics of deformable objects or require large training sets, which often limits generalization. We introduce embodied MPM (EMPM), a deformable object modeling and simulation framework built on a differentiable Material Point Method (MPM) simulator that captures the dynamics of challenging materials. From multi-view RGB-D videos, our approach reconstructs geometry and appearance, then uses an MPM physics engine to simulate object behavior by minimizing the mismatch between predicted and observed visual data. We further optimize MPM parameters online using sensory feedback, enabling adaptive, robust, and physics-aware object representations that open new possibilities for robotic manipulation of complex deformables. Experiments show that EMPM outperforms spring-mass baseline models. Project website: https://embodied-mpm.github.io.
>
---
#### [new 004] Constraint-Aware Discrete-Time PID Gain Optimization for Robotic Joint Control Under Actuator Saturation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决执行器饱和下离散PID参数优化问题。通过分析稳定性、抗饱和策略和贝叶斯优化，提升控制性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.18639v1](https://arxiv.org/pdf/2601.18639v1)**

> **作者:** Ojasva Mishra; Xiaolong Wu; Min Xu
>
> **摘要:** The precise regulation of rotary actuation is fundamental in autonomous robotics, yet practical PID loops deviate from continuous-time theory due to discrete-time execution, actuator saturation, and small delays and measurement imperfections. We present an implementation-aware analysis and tuning workflow for saturated discrete-time joint control. We (i) derive PI stability regions under Euler and exact zero-order-hold (ZOH) discretizations using the Jury criterion, (ii) evaluate a discrete back-calculation anti-windup realization under saturation-dominant regimes, and (iii) propose a hybrid-certified Bayesian optimization workflow that screens analytically unstable candidates and behaviorally unsafe transients while optimizing a robust IAE objective with soft penalties on overshoot and saturation duty. Baseline sweeps ($τ=1.0$~s, $Δt=0.01$~s, $u\in[-10,10]$) quantify rise/settle trends for P/PI/PID. Under a randomized model family emulating uncertainty, delay, noise, quantization, and tighter saturation, robustness-oriented tuning improves median IAE from $0.843$ to $0.430$ while keeping median overshoot below $2\%$. In simulation-only tuning, the certification screen rejects $11.6\%$ of randomly sampled gains within bounds before full robust evaluation, improving sample efficiency without hardware experiments.
>
---
#### [new 005] SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于船舶轨迹预测任务，解决长时 horizon 预测中的方向不一致问题。通过引入语义关键点条件，分解预测为语义决策与局部运动建模，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2601.18537v1](https://arxiv.org/pdf/2601.18537v1)**

> **作者:** Linyong Gan; Zimo Li; Wenxin Xu; Xingjian Li; Jianhua Z. Huang; Enmei Tu; Shuhang Chen
>
> **摘要:** Accurate long-horizon vessel trajectory prediction remains challenging due to compounded uncertainty from complex navigation behaviors and environmental factors. Existing methods often struggle to maintain global directional consistency, leading to drifting or implausible trajectories when extrapolated over long time horizons. To address this issue, we propose a semantic-key-point-conditioned trajectory modeling framework, in which future trajectories are predicted by conditioning on a high-level Next Key Point (NKP) that captures navigational intent. This formulation decomposes long-horizon prediction into global semantic decision-making and local motion modeling, effectively restricting the support of future trajectories to semantically feasible subsets. To efficiently estimate the NKP prior from historical observations, we adopt a pretrain-finetune strategy. Extensive experiments on real-world AIS data demonstrate that the proposed method consistently outperforms state-of-the-art approaches, particularly for long travel durations, directional accuracy, and fine-grained trajectory prediction.
>
---
#### [new 006] Correct-by-Construction Vision-based Pose Estimation using Geometric Generative Models
- **分类: cs.RO**

- **简介: 该论文属于视觉姿态估计任务，旨在解决深度神经网络缺乏可证明正确性的问题。通过结合物理建模与学习方法，设计可验证的神经网络框架，提升自主系统安全性。**

- **链接: [https://arxiv.org/pdf/2601.17556v1](https://arxiv.org/pdf/2601.17556v1)**

> **作者:** Ulices Santa Cruz; Mahmoud Elfar; Yasser Shoukry
>
> **摘要:** We consider the problem of vision-based pose estimation for autonomous systems. While deep neural networks have been successfully used for vision-based tasks, they inherently lack provable guarantees on the correctness of their output, which is crucial for safety-critical applications. We present a framework for designing certifiable neural networks (NNs) for perception-based pose estimation that integrates physics-driven modeling with learning-based estimation. The proposed framework begins by leveraging the known geometry of planar objects commonly found in the environment, such as traffic signs and runway markings, referred to as target objects. At its core, it introduces a geometric generative model (GGM), a neural-network-like model whose parameters are derived from the image formation process of a target object observed by a camera. Once designed, the GGM can be used to train NN-based pose estimators with certified guarantees in terms of their estimation errors. We first demonstrate this framework in uncluttered environments, where the target object is the only object present in the camera's field of view. We extend this using ideas from NN reachability analysis to design certified object NN that can detect the presence of the target object in cluttered environments. Subsequently, the framework consolidates the certified object detector with the certified pose estimator to design a multi-stage perception pipeline that generalizes the proposed approach to cluttered environments, while maintaining its certified guarantees. We evaluate the proposed framework using both synthetic and real images of various planar objects commonly encountered by autonomous vehicles. Using images captured by an event-based camera, we show that the trained encoder can effectively estimate the pose of a traffic sign in accordance with the certified bound provided by the framework.
>
---
#### [new 007] Real-Time Synchronized Interaction Framework for Emotion-Aware Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于情感交互任务，旨在解决人形机器人在社交场景中情感同步问题。通过构建实时框架，实现语音与肢体动作的同步，提升机器人的情感表达能力。**

- **链接: [https://arxiv.org/pdf/2601.17287v1](https://arxiv.org/pdf/2601.17287v1)**

> **作者:** Yanrong Chen; Xihan Bian
>
> **备注:** 6 pages, 6 figures
>
> **摘要:** As humanoid robots increasingly introduced into social scene, achieving emotionally synchronized multimodal interaction remains a significant challenges. To facilitate the further adoption and integration of humanoid robots into service roles, we present a real-time framework for NAO robots that synchronizes speech prosody with full-body gestures through three key innovations: (1) A dual-channel emotion engine where large language model (LLM) simultaneously generates context-aware text responses and biomechanically feasible motion descriptors, constrained by a structured joint movement library; (2) Duration-aware dynamic time warping for precise temporal alignment of speech output and kinematic motion keyframes; (3) Closed-loop feasibility verification ensuring gestures adhere to NAO's physical joint limits through real-time adaptation. Evaluations show 21% higher emotional alignment compared to rule-based systems, achieved by coordinating vocal pitch (arousal-driven) with upper-limb kinematics while maintaining lower-body stability. By enabling seamless sensorimotor coordination, this framework advances the deployment of context-aware social robots in dynamic applications such as personalized healthcare, interactive education, and responsive customer service platforms.
>
---
#### [new 008] NeuroManip: Prosthetic Hand Manipulation System Based on EMG and Eye Tracking Powered by the Neuromorphic Processor AltAi
- **分类: cs.RO**

- **简介: 该论文属于上肢假肢控制任务，旨在提升假手操控的效率与安全性。通过结合EMG和眼动追踪，利用神经形态处理器实现低功耗、实时手势识别。**

- **链接: [https://arxiv.org/pdf/2601.17991v1](https://arxiv.org/pdf/2601.17991v1)**

> **作者:** Roman Akinshin; Elizaveta Lopatina; Kirill Bogatikov; Nikolai Kiz; Anna V. Makarova; Mikhail Lebedev; Miguel Altamirano Cabrera; Dzmitry Tsetserukou; Valerii Kangler
>
> **备注:** This paper has been accepted for publication at LBR of HRI 2026 conference
>
> **摘要:** This paper presents a novel neuromorphic control architecture for upper-limb prostheses that combines surface electromyography (sEMG) with gaze-guided computer vision. The system uses a spiking neural network deployed on the neuromorphic processor AltAi to classify EMG patterns in real time while an eye-tracking headset and scene camera identify the object within the user's focus. In our prototype, the same EMG recognition model that was originally developed for a conventional GPU is deployed as a spiking network on AltAi, achieving comparable accuracy while operating in a sub-watt power regime, which enables a lightweight, wearable implementation. For six distinct functional gestures recorded from upper-limb amputees, the system achieves robust recognition performance comparable to state-of-the-art myoelectric interfaces. When the vision pipeline restricts the decision space to three context-appropriate gestures for the currently viewed object, recognition accuracy increases to roughly 95% while excluding unsafe, object-inappropriate grasps. These results indicate that the proposed neuromorphic, context-aware controller can provide energy-efficient and reliable prosthesis control and has the potential to improve safety and usability in everyday activities for people with upper-limb amputation.
>
---
#### [new 009] Hierarchical Informative Path Planning via Graph Guidance and Trajectory Optimization
- **分类: cs.RO**

- **简介: 该论文属于信息路径规划任务，解决在复杂环境中高效减少目标区域不确定性的难题。提出分层框架，结合全局图规划与局部轨迹优化，提升规划效率与精度。**

- **链接: [https://arxiv.org/pdf/2601.17227v1](https://arxiv.org/pdf/2601.17227v1)**

> **作者:** Avraiem Iskandar; Shamak Dutta; Kevin Murrant; Yash Vardhan Pant; Stephen L. Smith
>
> **摘要:** We study informative path planning (IPP) with travel budgets in cluttered environments, where an agent collects measurements of a latent field modeled as a Gaussian process (GP) to reduce uncertainty at target locations. Graph-based solvers provide global guarantees but assume pre-selected measurement locations, while continuous trajectory optimization supports path-based sensing but is computationally intensive and sensitive to initialization in obstacle-dense settings. We propose a hierarchical framework with three stages: (i) graph-based global planning, (ii) segment-wise budget allocation using geometric and kernel bounds, and (iii) spline-based refinement of each segment with hard constraints and obstacle pruning. By combining global guidance with local refinement, our method achieves lower posterior uncertainty than graph-only and continuous baselines, while running faster than continuous-space solvers (up to 9x faster than gradient-based methods and 20x faster than black-box optimizers) across synthetic cluttered environments and Arctic datasets.
>
---
#### [new 010] PILOT: A Perceptive Integrated Low-level Controller for Loco-manipulation over Unstructured Scenes
- **分类: cs.RO**

- **简介: 该论文提出PILOT框架，解决非结构化场景下人形机器人运动与操作的协同控制问题。通过强化学习融合感知与运动控制，提升稳定性与地形适应能力。**

- **链接: [https://arxiv.org/pdf/2601.17440v1](https://arxiv.org/pdf/2601.17440v1)**

> **作者:** Xinru Cui; Linxi Feng; Yixuan Zhou; Haoqi Han; Zhe Liu; Hesheng Wang
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Humanoid robots hold great potential for diverse interactions and daily service tasks within human-centered environments, necessitating controllers that seamlessly integrate precise locomotion with dexterous manipulation. However, most existing whole-body controllers lack exteroceptive awareness of the surrounding environment, rendering them insufficient for stable task execution in complex, unstructured scenarios.To address this challenge, we propose PILOT, a unified single-stage reinforcement learning (RL) framework tailored for perceptive loco-manipulation, which synergizes perceptive locomotion and expansive whole-body control within a single policy. To enhance terrain awareness and ensure precise foot placement, we design a cross-modal context encoder that fuses prediction-based proprioceptive features with attention-based perceptive representations. Furthermore, we introduce a Mixture-of-Experts (MoE) policy architecture to coordinate diverse motor skills, facilitating better specialization across distinct motion patterns. Extensive experiments in both simulation and on the physical Unitree G1 humanoid robot validate the efficacy of our framework. PILOT demonstrates superior stability, command tracking precision, and terrain traversability compared to existing baselines. These results highlight its potential to serve as a robust, foundational low-level controller for loco-manipulation in unstructured scenes.
>
---
#### [new 011] Breaking Task Impasses Quickly: Adaptive Neuro-Symbolic Learning for Open-World Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于开放世界机器人任务，解决自主系统在未知环境中的适应问题。通过融合符号学习与强化学习，提升机器人快速适应能力。**

- **链接: [https://arxiv.org/pdf/2601.16985v1](https://arxiv.org/pdf/2601.16985v1)**

> **作者:** Pierrick Lorang
>
> **备注:** IEEE ICRA 2025 Doctoral Consortium
>
> **摘要:** Adapting to unforeseen novelties in open-world environments remains a major challenge for autonomous systems. While hybrid planning and reinforcement learning (RL) approaches show promise, they often suffer from sample inefficiency, slow adaptation, and catastrophic forgetting. We present a neuro-symbolic framework integrating hierarchical abstractions, task and motion planning (TAMP), and reinforcement learning to enable rapid adaptation in robotics. Our architecture combines symbolic goal-oriented learning and world model-based exploration to facilitate rapid adaptation to environmental changes. Validated in robotic manipulation and autonomous driving, our approach achieves faster convergence, improved sample efficiency, and superior robustness over state-of-the-art hybrid methods, demonstrating its potential for real-world deployment.
>
---
#### [new 012] TC-IDM: Grounding Video Generation for Executable Zero-shot Robot Motion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动生成任务，旨在解决视觉-语言-动作模型泛化能力不足的问题。提出TC-IDM模型，通过工具轨迹实现视觉规划与物理控制的衔接，提升任务执行成功率。**

- **链接: [https://arxiv.org/pdf/2601.18323v1](https://arxiv.org/pdf/2601.18323v1)**

> **作者:** Weishi Mi; Yong Bao; Xiaowei Chi; Xiaozhu Ju; Zhiyuan Qin; Kuangzhi Ge; Kai Tang; Peidong Jia; Shanghang Zhang; Jian Tang
>
> **摘要:** The vision-language-action (VLA) paradigm has enabled powerful robotic control by leveraging vision-language models, but its reliance on large-scale, high-quality robot data limits its generalization. Generative world models offer a promising alternative for general-purpose embodied AI, yet a critical gap remains between their pixel-level plans and physically executable actions. To this end, we propose the Tool-Centric Inverse Dynamics Model (TC-IDM). By focusing on the tool's imagined trajectory as synthesized by the world model, TC-IDM establishes a robust intermediate representation that bridges the gap between visual planning and physical control. TC-IDM extracts the tool's point cloud trajectories via segmentation and 3D motion estimation from generated videos. Considering diverse tool attributes, our architecture employs decoupled action heads to project these planned trajectories into 6-DoF end-effector motions and corresponding control signals. This plan-and-translate paradigm not only supports a wide range of end-effectors but also significantly improves viewpoint invariance. Furthermore, it exhibits strong generalization capabilities across long-horizon and out-of-distribution tasks, including interacting with deformable objects. In real-world evaluations, the world model with TC-IDM achieves an average success rate of 61.11 percent, with 77.7 percent on simple tasks and 38.46 percent on zero-shot deformable object tasks. It substantially outperforms end-to-end VLA-style baselines and other inverse dynamics models.
>
---
#### [new 013] Grasp-and-Lift: Executable 3D Hand-Object Interaction Reconstruction via Physics-in-the-Loop Optimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手-物体交互重建任务，解决视觉轨迹在物理模拟中不真实的问题。通过优化方法将视觉对齐轨迹转化为物理可行轨迹，提升抓取与提升的稳定性。**

- **链接: [https://arxiv.org/pdf/2601.18121v1](https://arxiv.org/pdf/2601.18121v1)**

> **作者:** Byeonggyeol Choi; Woojin Oh; Jongwoo Lim
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Dexterous hand manipulation increasingly relies on large-scale motion datasets with precise hand-object trajectory data. However, existing resources such as DexYCB and HO3D are primarily optimized for visual alignment but often yield physically implausible interactions when replayed in physics simulators, including penetration, missed contact, and unstable grasps. We propose a simulation-in-the-loop refinement framework that converts these visually aligned trajectories into physically executable ones. Our core contribution is to formulate this as a tractable black-box optimization problem. We parameterize the hand's motion using a low-dimensional, spline-based representation built on sparse temporal keyframes. This allows us to use a powerful gradient-free optimizer, CMA-ES, to treat the high-fidelity physics engine as a black-box objective function. Our method finds motions that simultaneously maximize physical success (e.g., stable grasp and lift) while minimizing deviation from the original human demonstration. Compared to MANIPTRANS-recent transfer pipelines, our approach achieves lower hand and object pose errors during replay and more accurately recovers hand-object physical interactions. Our approach provides a general and scalable method for converting visual demonstrations into physically valid trajectories, enabling the generation of high-fidelity data crucial for robust policy learning.
>
---
#### [new 014] Real-Time, Energy-Efficient, Sampling-Based Optimal Control via FPGA Acceleration
- **分类: cs.RO; cs.AR**

- **简介: 该论文属于机器人控制任务，旨在解决AMR在能耗和延迟上的瓶颈。通过优化MPPI算法在FPGA上的实现，提升实时性和能效。**

- **链接: [https://arxiv.org/pdf/2601.17231v1](https://arxiv.org/pdf/2601.17231v1)**

> **作者:** Tanmay Desai; Brian Plancher; R. Iris Bahar
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Autonomous mobile robots (AMRs), used for search-and-rescue and remote exploration, require fast and robust planning and control schemes. Sampling-based approaches for Model Predictive Control, especially approaches based on the Model Predictive Path Integral Control (MPPI) algorithm, have recently proven both to be highly effective for such applications and to map naturally to GPUs for hardware acceleration. However, both GPU and CPU implementations of such algorithms can struggle to meet tight energy and latency budgets on battery-constrained AMR platforms that leverage embedded compute. To address this issue, we present an FPGA-optimized MPPI design that exposes fine-grained parallelism and eliminates synchronization bottlenecks via deep pipelining and parallelism across algorithmic stages. This results in an average 3.1x to 7.5x speedup over optimized implementations on an embedded GPU and CPU, respectively, while simultaneously achieving a 2.5x to 5.4x reduction in energy usage. These results demonstrate that FPGA architectures are a promising direction for energy-efficient and high-performance edge robotics.
>
---
#### [new 015] Quantifying Ergonomics in the Elevate Soft Robotic Suit
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决软体机器人舒适性与人体工学设计问题。通过实验评估Elevate软甲的肩部辅助性能及舒适度，验证其设计合理性。**

- **链接: [https://arxiv.org/pdf/2601.17249v1](https://arxiv.org/pdf/2601.17249v1)**

> **作者:** Peter Bryan; Rejin John Varghese; Dario Farina
>
> **备注:** 5 pages, 3 figures. Submitted to IEEE-EMBC 2026
>
> **摘要:** Soft robotic suits have the potential to rehabilitate, assist, and augment the human body. The low weight, cost, and minimal form-factor of these devices make them ideal for daily use by both healthy and impaired individuals. However, challenges associated with data-driven, user-specific, and comfort-first design of human-robot interfaces using soft materials limit their widespread translation and adoption. In this work, we present the quantitative evaluation of ergonomics and comfort of the Elevate suit - a cable driven soft robotic suit that assists shoulder elevation. Using a motion-capture system and force sensors, we measured the suit's ergonomics during assisted shoulder elevation up to 70 degrees. Two 4-hour sessions were conducted with one subject, involving transmitting cable tensions of up to 200N with no discomfort reported. We estimated that the pressure applied to the shoulder during assisted movements was within the range seen in a human grasp (approximately 69.1-85.1kPa), and estimated volumetric compression of <3% and <8% across the torso and upper arm, respectively. These results provide early validation of Elevate's ergonomic design in preparation for future studies with patient groups.
>
---
#### [new 016] Fast and Safe Trajectory Optimization for Mobile Manipulators With Neural Configuration Space Distance Field
- **分类: cs.RO**

- **简介: 该论文属于移动机械臂轨迹优化任务，解决复杂环境中碰撞检测与快速规划问题。提出GCDF方法，实现高效安全的轨迹优化。**

- **链接: [https://arxiv.org/pdf/2601.18548v1](https://arxiv.org/pdf/2601.18548v1)**

> **作者:** Yulin Li; Zhiyuan Song; Yiming Li; Zhicheng Song; Kai Chen; Chunxin Zheng; Zhihai Bi; Jiahang Cao; Sylvain Calinon; Fan Shi; Jun Ma
>
> **摘要:** Mobile manipulators promise agile, long-horizon behavior by coordinating base and arm motion, yet whole-body trajectory optimization in cluttered, confined spaces remains difficult due to high-dimensional nonconvexity and the need for fast, accurate collision reasoning. Configuration Space Distance Fields (CDF) enable fixed-base manipulators to model collisions directly in configuration space via smooth, implicit distances. This representation holds strong potential to bypass the nonlinear configuration-to-workspace mapping while preserving accurate whole-body geometry and providing optimization-friendly collision costs. Yet, extending this capability to mobile manipulators is hindered by unbounded workspaces and tighter base-arm coupling. We lift this promise to mobile manipulation with Generalized Configuration Space Distance Fields (GCDF), extending CDF to robots with both translational and rotational joints in unbounded workspaces with tighter base-arm coupling. We prove that GCDF preserves Euclidean-like local distance structure and accurately encodes whole-body geometry in configuration space, and develop a data generation and training pipeline that yields continuous neural GCDFs with accurate values and gradients, supporting efficient GPU-batched queries. Building on this representation, we develop a high-performance sequential convex optimization framework centered on GCDF-based collision reasoning. The solver scales to large numbers of implicit constraints through (i) online specification of neural constraints, (ii) sparsity-aware active-set detection with parallel batched evaluation across thousands of constraints, and (iii) incremental constraint management for rapid replanning under scene changes.
>
---
#### [new 017] Scaling Rough Terrain Locomotion with Automatic Curriculum Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决复杂地形下机器人高速稳定运动的问题。提出LP-ACRL框架，自动调整任务难度，提升机器人在多种地形上的性能。**

- **链接: [https://arxiv.org/pdf/2601.17428v1](https://arxiv.org/pdf/2601.17428v1)**

> **作者:** Ziming Li; Chenhao Li; Marco Hutter
>
> **摘要:** Curriculum learning has demonstrated substantial effectiveness in robot learning. However, it still faces limitations when scaling to complex, wide-ranging task spaces. Such task spaces often lack a well-defined difficulty structure, making the difficulty ordering required by previous methods challenging to define. We propose a Learning Progress-based Automatic Curriculum Reinforcement Learning (LP-ACRL) framework, which estimates the agent's learning progress online and adaptively adjusts the task-sampling distribution, thereby enabling automatic curriculum generation without prior knowledge of the difficulty distribution over the task space. Policies trained with LP-ACRL enable the ANYmal D quadruped to achieve and maintain stable, high-speed locomotion at 2.5 m/s linear velocity and 3.0 rad/s angular velocity across diverse terrains, including stairs, slopes, gravel, and low-friction flat surfaces--whereas previous methods have generally been limited to high speeds on flat terrain or low speeds on complex terrain. Experimental results demonstrate that LP-ACRL exhibits strong scalability and real-world applicability, providing a robust baseline for future research on curriculum generation in complex, wide-ranging robotic learning task spaces.
>
---
#### [new 018] EquiForm: Noise-Robust SE(3)-Equivariant Policy Learning from 3D Point Clouds
- **分类: cs.RO**

- **简介: 该论文提出EquiForm，解决点云感知下的鲁棒机械臂操作问题，通过几何去噪和等变对齐提升噪声环境下的性能。**

- **链接: [https://arxiv.org/pdf/2601.17486v1](https://arxiv.org/pdf/2601.17486v1)**

> **作者:** Zhiyuan Zhang; Yu She
>
> **备注:** Project website: https://ZhangZhiyuanZhang.github.io/equiform-website/ Code will be released
>
> **摘要:** Visual imitation learning with 3D point clouds has advanced robotic manipulation by providing geometry-aware, appearance-invariant observations. However, point cloud-based policies remain highly sensitive to sensor noise, pose perturbations, and occlusion-induced artifacts, which distort geometric structure and break the equivariance assumptions required for robust generalization. Existing equivariant approaches primarily encode symmetry constraints into neural architectures, but do not explicitly correct noise-induced geometric deviations or enforce equivariant consistency in learned representations. We introduce EquiForm, a noise-robust SE(3)-equivariant policy learning framework for point cloud-based manipulation. EquiForm formalizes how noise-induced geometric distortions lead to equivariance deviations in observation-to-action mappings, and introduces a geometric denoising module to restore consistent 3D structure under noisy or incomplete observations. In addition, we propose a contrastive equivariant alignment objective that enforces representation consistency under both rigid transformations and noise perturbations. Built upon these components, EquiForm forms a flexible policy learning pipeline that integrates noise-robust geometric reasoning with modern generative models. We evaluate EquiForm on 16 simulated tasks and 4 real-world manipulation tasks across diverse objects and scene layouts. Compared to state-of-the-art point cloud imitation learning methods, EquiForm achieves an average improvement of 17.2% in simulation and 28.1% in real-world experiments, demonstrating strong noise robustness and spatial generalization.
>
---
#### [new 019] Goal-oriented Communication for Fast and Robust Robotic Fault Detection and Recovery
- **分类: cs.RO**

- **简介: 该论文属于机器人故障检测与恢复任务，旨在解决现有框架延迟高、可靠性差的问题。提出GoC框架，通过语义表示和模型优化实现快速可靠恢复。**

- **链接: [https://arxiv.org/pdf/2601.18765v1](https://arxiv.org/pdf/2601.18765v1)**

> **作者:** Shutong Chen; Adnan Aijaz; Yansha Deng
>
> **备注:** Submit to IEEE for potential publication
>
> **摘要:** Autonomous robotic systems are widely deployed in smart factories and operate in dynamic, uncertain, and human-involved environments that require low-latency and robust fault detection and recovery (FDR). However, existing FDR frameworks exhibit various limitations, such as significant delays in communication and computation, and unreliability in robot motion/trajectory generation, mainly because the communication-computation-control (3C) loop is designed without considering the downstream FDR goal. To address this, we propose a novel Goal-oriented Communication (GoC) framework that jointly designs the 3C loop tailored for fast and robust robotic FDR, with the goal of minimising the FDR time while maximising the robotic task (e.g., workpiece sorting) success rate. For fault detection, our GoC framework innovatively defines and extracts the 3D scene graph (3D-SG) as the semantic representation via our designed representation extractor, and detects faults by monitoring spatial relationship changes in the 3D-SG. For fault recovery, we fine-tune a small language model (SLM) via Low-Rank Adaptation (LoRA) and enhance its reasoning and generalization capabilities via knowledge distillation to generate recovery motions for robots. We also design a lightweight goal-oriented digital twin reconstruction module to refine the recovery motions generated by the SLM when fine-grained robotic control is required, using only task-relevant object contours for digital twin reconstruction. Extensive simulations demonstrate that our GoC framework reduces the FDR time by up to 82.6% and improves the task success rate by up to 76%, compared to the state-of-the-art frameworks that rely on vision language models for fault detection and large language models for fault recovery.
>
---
#### [new 020] AsterNav: Autonomous Aerial Robot Navigation In Darkness Using Passive Computation
- **分类: cs.RO**

- **简介: 论文提出AsterNav系统，解决黑暗环境中微型无人机自主导航问题。通过红外单目相机与结构光结合，实现深度估计与避障，无需外部设施，成功率达95.5%。**

- **链接: [https://arxiv.org/pdf/2601.17550v1](https://arxiv.org/pdf/2601.17550v1)**

> **作者:** Deepak Singh; Shreyas Khobragade; Nitin J. Sanket
>
> **备注:** 8 pages, 10 figures, Published in IEEE Robotics And Automation Letters
>
> **摘要:** Autonomous aerial navigation in absolute darkness is crucial for post-disaster search and rescue operations, which often occur from disaster-zone power outages. Yet, due to resource constraints, tiny aerial robots, perfectly suited for these operations, are unable to navigate in the darkness to find survivors safely. In this paper, we present an autonomous aerial robot for navigation in the dark by combining an Infra-Red (IR) monocular camera with a large-aperture coded lens and structured light without external infrastructure like GPS or motion-capture. Our approach obtains depth-dependent defocus cues (each structured light point appears as a pattern that is depth dependent), which acts as a strong prior for our AsterNet deep depth estimation model. The model is trained in simulation by generating data using a simple optical model and transfers directly to the real world without any fine-tuning or retraining. AsterNet runs onboard the robot at 20 Hz on an NVIDIA Jetson Orin$^\text{TM}$ Nano. Furthermore, our network is robust to changes in the structured light pattern and relative placement of the pattern emitter and IR camera, leading to simplified and cost-effective construction. We successfully evaluate and demonstrate our proposed depth navigation approach AsterNav using depth from AsterNet in many real-world experiments using only onboard sensing and computation, including dark matte obstacles and thin ropes (diameter 6.25mm), achieving an overall success rate of 95.5% with unknown object shapes, locations and materials. To the best of our knowledge, this is the first work on monocular, structured-light-based quadrotor navigation in absolute darkness.
>
---
#### [new 021] Eye-Tracking-Driven Control in Daily Task Assistance for Assistive Robotic Arms
- **分类: cs.RO**

- **简介: 该论文属于辅助机器人控制任务，旨在解决眼动追踪在日常任务中的准确性和任务识别问题。提出一种基于眼动的控制框架，提升残疾人的独立操作能力。**

- **链接: [https://arxiv.org/pdf/2601.17404v1](https://arxiv.org/pdf/2601.17404v1)**

> **作者:** Anke Fischer-Janzen; Thomas M. Wendt; Kristof Van Laerhoven
>
> **备注:** 23 pages, 6 figures, publication in review process
>
> **摘要:** Shared control improves Human-Robot Interaction by reducing the user's workload and increasing the robot's autonomy. It allows robots to perform tasks under the user's supervision. Current eye-tracking-driven approaches face several challenges. These include accuracy issues in 3D gaze estimation and difficulty interpreting gaze when differentiating between multiple tasks. We present an eye-tracking-driven control framework, aimed at enabling individuals with severe physical disabilities to perform daily tasks independently. Our system uses task pictograms as fiducial markers combined with a feature matching approach that transmits data of the selected object to accomplish necessary task related measurements with an eye-in-hand configuration. This eye-tracking control does not require knowledge of the user's position in relation to the object. The framework correctly interpreted object and task selection in up to 97.9% of measurements. Issues were found in the evaluation, that were improved and shared as lessons learned. The open-source framework can be adapted to new tasks and objects due to the integration of state-of-the-art object detection models.
>
---
#### [new 022] MetaWorld: Skill Transfer and Composition in a Hierarchical World Model for Grounding High-Level Instructions
- **分类: cs.RO**

- **简介: 该论文属于人形机器人控制任务，旨在解决语义-物理鸿沟问题。通过构建分层世界模型，结合语义规划与物理控制，提升任务完成效率和运动一致性。**

- **链接: [https://arxiv.org/pdf/2601.17507v1](https://arxiv.org/pdf/2601.17507v1)**

> **作者:** Yutong Shen; Hangxu Liu; Kailin Pei; Ruizhe Xia; Tongtong Feng
>
> **备注:** 8 pages, 4 figures, Submitted to ICLR 2026 World Model Workshop
>
> **摘要:** Humanoid robot loco-manipulation remains constrained by the semantic-physical gap. Current methods face three limitations: Low sample efficiency in reinforcement learning, poor generalization in imitation learning, and physical inconsistency in VLMs. We propose MetaWorld, a hierarchical world model that integrates semantic planning and physical control via expert policy transfer. The framework decouples tasks into a VLM-driven semantic layer and a latent dynamics model operating in a compact state space. Our dynamic expert selection and motion prior fusion mechanism leverages a pre-trained multi-expert policy library as transferable knowledge, enabling efficient online adaptation via a two-stage framework. VLMs serve as semantic interfaces, mapping instructions to executable skills and bypassing symbol grounding. Experiments on Humanoid-Bench show MetaWorld outperforms world model-based RL in task completion and motion coherence. Our code will be found at https://anonymous.4open.science/r/metaworld-2BF4/
>
---
#### [new 023] Quest2ROS2: A ROS 2 Framework for Bi-manual VR Teleoperation
- **分类: cs.RO**

- **简介: 该论文提出Quest2ROS2框架，用于双臂VR远程操作，解决机器人数据收集的作业空间限制问题，通过相对运动控制实现直观操作。**

- **链接: [https://arxiv.org/pdf/2601.18289v1](https://arxiv.org/pdf/2601.18289v1)**

> **作者:** Jialong Li; Zhenguo Wang; Tianci Wang; Maj Stenmark; Volker Krueger
>
> **备注:** HRI 2026
>
> **摘要:** Quest2ROS2 is an open-source ROS2 framework for bi-manual teleoperation designed to scale robot data collection. Extending Quest2ROS, it overcomes workspace limitations via relative motion-based control, calculating robot movement from VR controller pose changes to enable intuitive, pose-independent operation. The framework integrates essential usability and safety features, including real-time RViz visualization, streamlined gripper control, and a pause-and-reset function for smooth transitions. We detail a modular architecture that supports "Side-by-Side" and "Mirror" control modes to optimize operator experience across diverse platforms. Code is available at: https://github.com/Taokt/Quest2ROS2.
>
---
#### [new 024] Delay-Compensated Stiffness Estimation for Robot-Mediated Dyadic Interaction
- **分类: cs.RO; cs.HC; eess.SY**

- **简介: 该论文属于机器人辅助人机交互任务，解决网络延迟导致的刚度估计误差问题。通过引入时序对齐和鲁棒滤波方法，提升远程交互中的刚度感知精度。**

- **链接: [https://arxiv.org/pdf/2601.17812v1](https://arxiv.org/pdf/2601.17812v1)**

> **作者:** Mingtian Du; Suhas Raghavendra Kulkarni; Bernardo Noronha; Domenico Campolo
>
> **摘要:** Robot-mediated human-human (dyadic) interactions enable therapists to provide physical therapy remotely, yet an accurate perception of patient stiffness remains challenging due to network-induced haptic delays. Conventional stiffness estimation methods, which neglect delay, suffer from temporal misalignment between force and position signals, leading to significant estimation errors as delays increase. To address this, we propose a robust, delay-compensated stiffness estimation framework by deriving an algebraic estimator based on quasi-static equilibrium that explicitly accounts for temporally aligning the expert's input with the novice's response. A Normalised Weighted Least Squares (NWLS) implementation is then introduced to robustly filter dynamic bias resulting from the algebraic derivation. Experiments using commercial rehabilitation robots (H-MAN) as the platform demonstrate that the proposed method significantly outperforms the standard estimator, maintaining consistent tracking accuracy under multiple introduced delays. These findings offer a promising solution for achieving high-fidelity haptic perception in remote dyadic interaction, potentially facilitating reliable stiffness assessment in therapeutic settings across networks.
>
---
#### [new 025] Less Is More: Scalable Visual Navigation from Limited Data
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，旨在解决数据稀缺下的导航问题。通过结合传统规划器生成合成轨迹，提升基于模仿学习的导航性能。**

- **链接: [https://arxiv.org/pdf/2601.17815v1](https://arxiv.org/pdf/2601.17815v1)**

> **作者:** Yves Inglin; Jonas Frey; Changan Chen; Marco Hutter
>
> **摘要:** Imitation learning provides a powerful framework for goal-conditioned visual navigation in mobile robots, enabling obstacle avoidance while respecting human preferences and social norms. However, its effectiveness depends critically on the quality and diversity of training data. In this work, we show how classical geometric planners can be leveraged to generate synthetic trajectories that complement costly human demonstrations. We train Less is More (LiMo), a transformer-based visual navigation policy that predicts goal-conditioned SE(2) trajectories from a single RGB observation, and find that augmenting limited expert demonstrations with planner-generated supervision yields substantial performance gains. Through ablations and complementary qualitative and quantitative analyses, we characterize how dataset scale and diversity affect planning performance. We demonstrate real-robot deployment and argue that robust visual navigation is enabled not by simply collecting more demonstrations, but by strategically curating diverse, high-quality datasets. Our results suggest that scalable, embodiment-specific geometric supervision is a practical path toward data-efficient visual navigation.
>
---
#### [new 026] SG-CADVLM: A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全验证任务，旨在解决真实事故数据稀缺问题。通过生成高风险场景，提升测试效率。工作包括提出SG-CADVLM模型，结合上下文解码与多模态输入，提高场景生成准确率。**

- **链接: [https://arxiv.org/pdf/2601.18442v1](https://arxiv.org/pdf/2601.18442v1)**

> **作者:** Hongyi Zhao; Shuo Wang; Qijie He; Ziyuan Pu
>
> **摘要:** Autonomous vehicle safety validation requires testing on safety-critical scenarios, but these events are rare in real-world driving and costly to test due to collision risks. Crash reports provide authentic specifications of safety-critical events, offering a vital alternative to scarce real-world collision trajectory data. This makes them valuable sources for generating realistic high-risk scenarios through simulation. Existing approaches face significant limitations because data-driven methods lack diversity due to their reliance on existing latent distributions, whereas adversarial methods often produce unrealistic scenarios lacking physical fidelity. Large Language Model (LLM) and Vision Language Model (VLM)-based methods show significant promise. However, they suffer from context suppression issues where internal parametric knowledge overrides crash specifications, producing scenarios that deviate from actual accident characteristics. This paper presents SG-CADVLM (A Context-Aware Decoding Powered Vision Language Model for Safety-Critical Scenario Generation), a framework that integrates Context-Aware Decoding with multi-modal input processing to generate safety-critical scenarios from crash reports and road network diagrams. The framework mitigates VLM hallucination issues while enabling the simultaneous generation of road geometry and vehicle trajectories. The experimental results demonstrate that SG-CADVLM generates critical risk scenarios at a rate of 84.4% compared to 12.5% for the baseline methods, representing an improvement of 469%, while producing executable simulations for autonomous vehicle testing.
>
---
#### [new 027] Trustworthy Evaluation of Robotic Manipulation: A New Benchmark and AutoEval Methods
- **分类: cs.RO**

- **简介: 该论文属于机器人操作评估任务，解决现有评价方法不足的问题，提出Eval-Actions基准和AutoEval架构，提升评估的可信度与准确性。**

- **链接: [https://arxiv.org/pdf/2601.18723v1](https://arxiv.org/pdf/2601.18723v1)**

> **作者:** Mengyuan Liu; Juyi Sheng; Peiming Li; Ziyi Wang; Tianming Xu; Tiantian Xu; Hong Liu
>
> **摘要:** Driven by the rapid evolution of Vision-Action and Vision-Language-Action models, imitation learning has significantly advanced robotic manipulation capabilities. However, evaluation methodologies have lagged behind, hindering the establishment of Trustworthy Evaluation for these behaviors. Current paradigms rely on binary success rates, failing to address the critical dimensions of trust: Source Authenticity (i.e., distinguishing genuine policy behaviors from human teleoperation) and Execution Quality (e.g., smoothness and safety). To bridge these gaps, we propose a solution that combines the Eval-Actions benchmark and the AutoEval architecture. First, we construct the Eval-Actions benchmark to support trustworthiness analysis. Distinct from existing datasets restricted to successful human demonstrations, Eval-Actions integrates VA and VLA policy execution trajectories alongside human teleoperation data, explicitly including failure scenarios. This dataset is structured around three core supervision signals: Expert Grading (EG), Rank-Guided preferences (RG), and Chain-of-Thought (CoT). Building on this, we propose the AutoEval architecture: AutoEval leverages Spatio-Temporal Aggregation for semantic assessment, augmented by an auxiliary Kinematic Calibration Signal to refine motion smoothness; AutoEval Plus (AutoEval-P) incorporates the Group Relative Policy Optimization (GRPO) paradigm to enhance logical reasoning capabilities. Experiments show AutoEval achieves Spearman's Rank Correlation Coefficients (SRCC) of 0.81 and 0.84 under the EG and RG protocols, respectively. Crucially, the framework possesses robust source discrimination capabilities, distinguishing between policy-generated and teleoperated videos with 99.6% accuracy, thereby establishing a rigorous standard for trustworthy robotic evaluation. Our project and code are available at https://term-bench.github.io/.
>
---
#### [new 028] Attention-Based Neural-Augmented Kalman Filter for Legged Robot State Estimation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人状态估计任务，解决足滑导致的误差问题。通过引入基于注意力的神经补偿器，提升InEKF在足滑情况下的估计性能。**

- **链接: [https://arxiv.org/pdf/2601.18569v1](https://arxiv.org/pdf/2601.18569v1)**

> **作者:** Seokju Lee; Kyung-Soo Kim
>
> **备注:** 8 pages, 6 figures, Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** In this letter, we propose an Attention-Based Neural-Augmented Kalman Filter (AttenNKF) for state estimation in legged robots. Foot slip is a major source of estimation error: when slip occurs, kinematic measurements violate the no-slip assumption and inject bias during the update step. Our objective is to estimate this slip-induced error and compensate for it. To this end, we augment an Invariant Extended Kalman Filter (InEKF) with a neural compensator that uses an attention mechanism to infer error conditioned on foot-slip severity and then applies this estimate as a post-update compensation to the InEKF state (i.e., after the filter update). The compensator is trained in a latent space, which aims to reduce sensitivity to raw input scales and encourages structured slip-conditioned compensations, while preserving the InEKF recursion. Experiments demonstrate improved performance compared to existing legged-robot state estimators, particularly under slip-prone conditions.
>
---
#### [new 029] ExoGS: A 4D Real-to-Sim-to-Real Framework for Scalable Manipulation Data Collection
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决真实到仿真再到真实的数据收集问题。提出ExoGS框架，通过人体示范捕获环境与交互信息，提升数据效率和策略泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.18629v1](https://arxiv.org/pdf/2601.18629v1)**

> **作者:** Yiming Wang; Ruogu Zhang; Minyang Li; Hao Shi; Junbo Wang; Deyi Li; Jieji Ren; Wenhai Liu; Weiming Wang; Hao-Shu Fang
>
> **摘要:** Real-to-Sim-to-Real technique is gaining increasing interest for robotic manipulation, as it can generate scalable data in simulation while having narrower sim-to-real gap. However, previous methods mainly focused on environment-level visual real-to-sim transfer, ignoring the transfer of interactions, which could be challenging and inefficient to obtain purely in simulation especially for contact-rich tasks. We propose ExoGS, a robot-free 4D Real-to-Sim-to-Real framework that captures both static environments and dynamic interactions in the real world and transfers them seamlessly to a simulated environment. It provides a new solution for scalable manipulation data collection and policy learning. ExoGS employs a self-designed robot-isomorphic passive exoskeleton AirExo-3 to capture kinematically consistent trajectories with millimeter-level accuracy and synchronized RGB observations during direct human demonstrations. The robot, objects, and environment are reconstructed as editable 3D Gaussian Splatting assets, enabling geometry-consistent replay and large-scale data augmentation. Additionally, a lightweight Mask Adapter injects instance-level semantics into the policy to enhance robustness under visual domain shifts. Real-world experiments demonstrate that ExoGS significantly improves data efficiency and policy generalization compared to teleoperation-based baselines. Code and hardware files have been released on https://github.com/zaixiabalala/ExoGS.
>
---
#### [new 030] A Pragmatic VLA Foundation Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LingBot-VLA模型，解决机器人操作中任务泛化与成本效率问题。通过大量真实数据训练，实现跨平台高效部署，并开源相关资源促进研究。**

- **链接: [https://arxiv.org/pdf/2601.18692v1](https://arxiv.org/pdf/2601.18692v1)**

> **作者:** Wei Wu; Fan Lu; Yunnan Wang; Shuai Yang; Shi Liu; Fangjing Wang; Qian Zhu; He Sun; Yong Wang; Shuailei Ma; Yiyu Ren; Kejia Zhang; Hui Yu; Jingmei Zhao; Shuai Zhou; Zhenqi Qiu; Houlong Xiong; Ziyu Wang; Zechen Wang; Ran Cheng; Yong-Lu Li; Yongtao Huang; Xing Zhu; Yujun Shen; Kecheng Zheng
>
> **备注:** Project Webpage: https://technology.robbyant.com/lingbot-vla/, Code: https://github.com/Robbyant/lingbot-vla/
>
> **摘要:** Offering great potential in robotic manipulation, a capable Vision-Language-Action (VLA) foundation model is expected to faithfully generalize across tasks and platforms while ensuring cost efficiency (e.g., data and GPU hours required for adaptation). To this end, we develop LingBot-VLA with around 20,000 hours of real-world data from 9 popular dual-arm robot configurations. Through a systematic assessment on 3 robotic platforms, each completing 100 tasks with 130 post-training episodes per task, our model achieves clear superiority over competitors, showcasing its strong performance and broad generalizability. We have also built an efficient codebase, which delivers a throughput of 261 samples per second per GPU with an 8-GPU training setup, representing a 1.5~2.8$\times$ (depending on the relied VLM base model) speedup over existing VLA-oriented codebases. The above features ensure that our model is well-suited for real-world deployment. To advance the field of robot learning, we provide open access to the code, base model, and benchmark data, with a focus on enabling more challenging tasks and promoting sound evaluation standards.
>
---
#### [new 031] Advancing Improvisation in Human-Robot Construction Collaboration: Taxonomy and Research Roadmap
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决建筑领域中机器人适应性不足的问题。通过构建六级分类体系，分析现有研究并提出未来改进方向。**

- **链接: [https://arxiv.org/pdf/2601.17219v1](https://arxiv.org/pdf/2601.17219v1)**

> **作者:** David Wireko Atibila; Vineet R. Kamat; Carol C. Menassa
>
> **备注:** 73 pages, 8 figures
>
> **摘要:** The construction industry faces productivity stagnation, skilled labor shortages, and safety concerns. While robotic automation offers solutions, construction robots struggle to adapt to unstructured, dynamic sites. Central to this is improvisation, adapting to unexpected situations through creative problem-solving, which remains predominantly human. In construction's unpredictable environments, collaborative human-robot improvisation is essential for workflow continuity. This research develops a six-level taxonomy classifying human-robot collaboration (HRC) based on improvisation capabilities. Through systematic review of 214 articles (2010-2025), we categorize construction robotics across: Manual Work (Level 0), Human-Controlled Execution (Level 1), Adaptive Manipulation (Level 2), Imitation Learning (Level 3), Human-in-Loop BIM Workflow (Level 4), Cloud-Based Knowledge Integration (Level 5), and True Collaborative Improvisation (Level 6). Analysis reveals current research concentrates at lower levels, with critical gaps in experiential learning and limited progression toward collaborative improvisation. A five-dimensional radar framework illustrates progressive evolution of Planning, Cognitive Role, Physical Execution, Learning Capability, and Improvisation, demonstrating how complementary human-robot capabilities create team performance exceeding individual contributions. The research identifies three fundamental barriers: technical limitations in grounding and dialogic reasoning, conceptual gaps between human improvisation and robotics research, and methodological challenges. We recommend future research emphasizing improved human-robot communication via Augmented/Virtual Reality interfaces, large language model integration, and cloud-based knowledge systems to advance toward true collaborative improvisation.
>
---
#### [new 032] DiffusionCinema: Text-to-Aerial Cinematography
- **分类: cs.RO**

- **简介: 该论文提出DiffusionCinema系统，属于文本到影视拍摄任务，解决传统无人机操控繁琐的问题，通过扩散模型自动生成飞行轨迹，实现文本驱动的自动航拍。**

- **链接: [https://arxiv.org/pdf/2601.17412v1](https://arxiv.org/pdf/2601.17412v1)**

> **作者:** Valerii Serpiva; Artem Lykov; Jeffrin Sam; Aleksey Fedoseev; Dzmitry Tsetserukou
>
> **摘要:** We propose a novel Unmanned Aerial Vehicles (UAV) assisted creative capture system that leverages diffusion models to interpret high-level natural language prompts and automatically generate optimal flight trajectories for cinematic video recording. Instead of manually piloting the drone, the user simply describes the desired shot (e.g., "orbit around me slowly from the right and reveal the background waterfall"). Our system encodes the prompt along with an initial visual snapshot from the onboard camera, and a diffusion model samples plausible spatio-temporal motion plans that satisfy both the scene geometry and shot semantics. The generated flight trajectory is then executed autonomously by the UAV to record smooth, repeatable video clips that match the prompt. User evaluation using NASA-TLX showed a significantly lower overall workload with our interface (M = 21.6) compared to a traditional remote controller (M = 58.1), demonstrating a substantial reduction in perceived effort. Mental demand (M = 11.5 vs. 60.5) and frustration (M = 14.0 vs. 54.5) were also markedly lower for our system, confirming clear usability advantages in autonomous text-driven flight control. This project demonstrates a new interaction paradigm: text-to-cinema flight, where diffusion models act as the "creative operator" converting story intentions directly into aerial motion.
>
---
#### [new 033] Online parameter estimation for the Crazyflie quadcopter through an EM algorithm
- **分类: cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于参数估计任务，旨在解决四旋翼无人机参数估计问题。通过EM算法实现在线参数估计，并与离线方法对比，验证其收敛范围更广。**

- **链接: [https://arxiv.org/pdf/2601.17009v1](https://arxiv.org/pdf/2601.17009v1)**

> **作者:** Yanhua Zhao
>
> **备注:** 20 pages, 37 figures
>
> **摘要:** Drones are becoming more and more popular nowadays. They are small in size, low in cost, and reliable in operation. They contain a variety of sensors and can perform a variety of flight tasks, reaching places that are difficult or inaccessible for humans. Earthquakes damage a lot of infrastructure, making it impossible for rescuers to reach some areas. But drones can help. Many amateur and professional photographers like to use drones for aerial photography. Drones play a non-negligible role in agriculture and transportation too. Drones can be used to spray pesticides, and they can also transport supplies. A quadcopter is a four-rotor drone and has been studied in this paper. In this paper, random noise is added to the quadcopter system and its effects on the drone system are studied. An extended Kalman filter has been used to estimate the state based on noisy observations from the sensor. Based on a SDE system, a linear quadratic Gaussian controller has been implemented. The expectation maximization algorithm has been applied for parameter estimation of the quadcopter. The results of offline parameter estimation and online parameter estimation are presented. The results show that the online parameter estimation has a slightly larger range of convergence values than the offline parameter estimation.
>
---
#### [new 034] Quantum-Inspired Episode Selection for Monte Carlo Reinforcement Learning via QUBO Optimization
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决蒙特卡洛方法样本复杂度高的问题。通过将轨迹选择建模为QUBO问题，使用量子启发式算法优化选择，提升收敛速度和策略质量。**

- **链接: [https://arxiv.org/pdf/2601.17570v1](https://arxiv.org/pdf/2601.17570v1)**

> **作者:** Hadi Salloum; Ali Jnadi; Yaroslav Kholodov; Alexander Gasnikov
>
> **备注:** Proceedings of Machine Learning Research tbd: 1_13, 2025 International Conference on Computational Optimization
>
> **摘要:** Monte Carlo (MC) reinforcement learning suffers from high sample complexity, especially in environments with sparse rewards, large state spaces, and correlated trajectories. We address these limitations by reformulating episode selection as a Quadratic Unconstrained Binary Optimization (QUBO) problem and solving it with quantum-inspired samplers. Our method, MC+QUBO, integrates a combinatorial filtering step into standard MC policy evaluation: from each batch of trajectories, we select a subset that maximizes cumulative reward while promoting state-space coverage. This selection is encoded as a QUBO, where linear terms favor high-reward episodes and quadratic terms penalize redundancy. We explore both Simulated Quantum Annealing (SQA) and Simulated Bifurcation (SB) as black-box solvers within this framework. Experiments in a finite-horizon GridWorld demonstrate that MC+QUBO outperforms vanilla MC in convergence speed and final policy quality, highlighting the potential of quantum-inspired optimization as a decision-making subroutine in reinforcement learning.
>
---
#### [new 035] Masked Depth Modeling for Spatial Perception
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于深度补全任务，旨在解决RGB-D相机在复杂场景中的深度精度和覆盖问题。通过引入掩码深度建模和数据清洗流程，提升深度图质量。**

- **链接: [https://arxiv.org/pdf/2601.17895v1](https://arxiv.org/pdf/2601.17895v1)**

> **作者:** Bin Tan; Changjiang Sun; Xiage Qin; Hanat Adai; Zelin Fu; Tianxiang Zhou; Han Zhang; Yinghao Xu; Xing Zhu; Yujun Shen; Nan Xue
>
> **备注:** Tech report, 19 pages, 15 figures and 4 tables
>
> **摘要:** Spatial visual perception is a fundamental requirement in physical-world applications like autonomous driving and robotic manipulation, driven by the need to interact with 3D environments. Capturing pixel-aligned metric depth using RGB-D cameras would be the most viable way, yet it usually faces obstacles posed by hardware limitations and challenging imaging conditions, especially in the presence of specular or texture-less surfaces. In this work, we argue that the inaccuracies from depth sensors can be viewed as "masked" signals that inherently reflect underlying geometric ambiguities. Building on this motivation, we present LingBot-Depth, a depth completion model which leverages visual context to refine depth maps through masked depth modeling and incorporates an automated data curation pipeline for scalable training. It is encouraging to see that our model outperforms top-tier RGB-D cameras in terms of both depth precision and pixel coverage. Experimental results on a range of downstream tasks further suggest that LingBot-Depth offers an aligned latent representation across RGB and depth modalities. We release the code, checkpoint, and 3M RGB-depth pairs (including 2M real data and 1M simulated data) to the community of spatial perception.
>
---
#### [new 036] Beyond Static Datasets: Robust Offline Policy Optimization via Vetted Synthetic Transitions
- **分类: cs.LG; cs.HC; cs.RO**

- **简介: 该论文属于离线强化学习任务，旨在解决数据分布偏差问题。通过合成高置信度转移数据，提升策略优化效果。**

- **链接: [https://arxiv.org/pdf/2601.18107v1](https://arxiv.org/pdf/2601.18107v1)**

> **作者:** Pedram Agand; Mo Chen
>
> **备注:** 11 pages, 2 figures, 2 tables
>
> **摘要:** Offline Reinforcement Learning (ORL) holds immense promise for safety-critical domains like industrial robotics, where real-time environmental interaction is often prohibitive. A primary obstacle in ORL remains the distributional shift between the static dataset and the learned policy, which typically mandates high degrees of conservatism that can restrain potential policy improvements. We present MoReBRAC, a model-based framework that addresses this limitation through Uncertainty-Aware latent synthesis. Instead of relying solely on the fixed data, MoReBRAC utilizes a dual-recurrent world model to synthesize high-fidelity transitions that augment the training manifold. To ensure the reliability of this synthetic data, we implement a hierarchical uncertainty pipeline integrating Variational Autoencoder (VAE) manifold detection, model sensitivity analysis, and Monte Carlo (MC) dropout. This multi-layered filtering process guarantees that only transitions residing within high-confidence regions of the learned dynamics are utilized. Our results on D4RL Gym-MuJoCo benchmarks reveal significant performance gains, particularly in ``random'' and ``suboptimal'' data regimes. We further provide insights into the role of the VAE as a geometric anchor and discuss the distributional trade-offs encountered when learning from near-optimal datasets.
>
---
#### [new 037] Low Cost, High Efficiency: LiDAR Place Recognition in Vineyards with Matryoshka Representation Learning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于农业环境中的定位任务，解决 vineyard 中的 LiDAR 地点识别问题。提出 MinkUNeXt-VINE 方法，采用轻量级设计和多损失学习，提升识别效率与精度。**

- **链接: [https://arxiv.org/pdf/2601.18714v1](https://arxiv.org/pdf/2601.18714v1)**

> **作者:** Judith Vilella-Cantos; Mauro Martini; Marcello Chiaberge; Mónica Ballesta; David Valiente
>
> **摘要:** Localization in agricultural environments is challenging due to their unstructured nature and lack of distinctive landmarks. Although agricultural settings have been studied in the context of object classification and segmentation, the place recognition task for mobile robots is not trivial in the current state of the art. In this study, we propose MinkUNeXt-VINE, a lightweight, deep-learning-based method that surpasses state-of-the-art methods in vineyard environments thanks to its pre-processing and Matryoshka Representation Learning multi-loss approach. Our method prioritizes enhanced performance with low-cost, sparse LiDAR inputs and lower-dimensionality outputs to ensure high efficiency in real-time scenarios. Additionally, we present a comprehensive ablation study of the results on various evaluation cases and two extensive long-term vineyard datasets employing different LiDAR sensors. The results demonstrate the efficiency of the trade-off output produced by this approach, as well as its robust performance on low-cost and low-resolution input data. The code is publicly available for reproduction.
>
---
#### [new 038] Autonomous Mars Rover Module for Soil Sampling and Life Component Analysis
- **分类: eess.SY; astro-ph.EP; astro-ph.IM; cs.RO**

- **简介: 该论文属于火星探测任务，旨在解决外星生命检测问题。设计了自主土壤采样与生物成分分析模块，实现对火星土壤中生命成分的检测。**

- **链接: [https://arxiv.org/pdf/2601.17158v1](https://arxiv.org/pdf/2601.17158v1)**

> **作者:** Bibek Adhikari; Rishab Rijal; Rakesh Yadav; Nikchey Khatri; Sandesh Dhakal
>
> **备注:** 9 pages, 12 figures
>
> **摘要:** The search for extraterrestrial life has long been a primary focus of scientific exploration, driven by rapid advancements in technology and our understanding of the universe. The discovery of water on Mars has sparked significant interest, raising the question of whether life could exist on the planet. This study proposes a novel approach to simulate and illustrate the detection of life using a proof-of-life module integrated into a Mars rover. The module is an autonomous system capable of traveling to designated regions, excavating soil, collecting samples, and performing biochemical testing onboard the rover itself. The project is inherently multidisciplinary, integrating mechanical systems such as a drill mechanism and a vacuum system, alongside biochemical analysis for soil testing. The module is capable of successfully detecting the presence or absence of living components of life from the collected soil particles. This proof-of-life module serves as a proof-of-concept for autonomous life detection in extraterrestrial environments and lays the foundation for future exploration missions.
>
---
#### [new 039] ConceptACT: Episode-Level Concepts for Sample-Efficient Robotic Imitation Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于机器人模仿学习任务，解决传统方法依赖低级数据、效率低的问题。通过引入语义概念提升样本效率，使用改进的Transformer架构实现概念感知的注意力机制。**

- **链接: [https://arxiv.org/pdf/2601.17135v1](https://arxiv.org/pdf/2601.17135v1)**

> **作者:** Jakob Karalus; Friedhelm Schwenker
>
> **摘要:** Imitation learning enables robots to acquire complex manipulation skills from human demonstrations, but current methods rely solely on low-level sensorimotor data while ignoring the rich semantic knowledge humans naturally possess about tasks. We present ConceptACT, an extension of Action Chunking with Transformers that leverages episode-level semantic concept annotations during training to improve learning efficiency. Unlike language-conditioned approaches that require semantic input at deployment, ConceptACT uses human-provided concepts (object properties, spatial relationships, task constraints) exclusively during demonstration collection, adding minimal annotation burden. We integrate concepts using a modified transformer architecture in which the final encoder layer implements concept-aware cross-attention, supervised to align with human annotations. Through experiments on two robotic manipulation tasks with logical constraints, we demonstrate that ConceptACT converges faster and achieves superior sample efficiency compared to standard ACT. Crucially, we show that architectural integration through attention mechanisms significantly outperforms naive auxiliary prediction losses or language-conditioned models. These results demonstrate that properly integrated semantic supervision provides powerful inductive biases for more efficient robot learning.
>
---
#### [new 040] PEAfowl: Perception-Enhanced Multi-View Vision-Language-Action for Bimanual Manipulation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出PEAfowl，解决杂乱场景下双臂操作的视觉-语言-动作策略问题。通过增强空间推理和指令定位，提升任务成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.17885v1](https://arxiv.org/pdf/2601.17885v1)**

> **作者:** Qingyu Fan; Zhaoxiang Li; Yi Lu; Wang Chen; Qiu Shen; Xiao-xiao Long; Yinghao Cai; Tao Lu; Shuo Wang; Xun Cao
>
> **摘要:** Bimanual manipulation in cluttered scenes requires policies that remain stable under occlusions, viewpoint and scene variations. Existing vision-language-action models often fail to generalize because (i) multi-view features are fused via view-agnostic token concatenation, yielding weak 3D-consistent spatial understanding, and (ii) language is injected as global conditioning, resulting in coarse instruction grounding. In this paper, we introduce PEAfowl, a perception-enhanced multi-view VLA policy for bimanual manipulation. For spatial reasoning, PEAfowl predicts per-token depth distributions, performs differentiable 3D lifting, and aggregates local cross-view neighbors to form geometrically grounded, cross-view consistent representations. For instruction grounding, we propose to replace global conditioning with a Perceiver-style text-aware readout over frozen CLIP visual features, enabling iterative evidence accumulation. To overcome noisy and incomplete commodity depth without adding inference overhead, we apply training-only depth distillation from a pretrained depth teacher to supervise the depth-distribution head, providing perception front-end with geometry-aware priors. On RoboTwin 2.0 under domain-randomized setting, PEAfowl improves the strongest baseline by 23.0 pp in success rate, and real-robot experiments further demonstrate reliable sim-to-real transfer and consistent improvements from depth distillation. Project website: https://peafowlvla.github.io/.
>
---
#### [new 041] Acoustic Field Video for Multimodal Scene Understanding
- **分类: cs.HC; cs.CV; cs.RO**

- **简介: 该论文提出声场视频作为多模态输入，用于提升场景理解。解决视觉与音频信息不足的问题，通过实时声场数据增强视觉语言模型性能。**

- **链接: [https://arxiv.org/pdf/2601.17123v1](https://arxiv.org/pdf/2601.17123v1)**

> **作者:** Daehwa Kim; Chris Harrison
>
> **摘要:** We introduce and explore a new multimodal input representation for vision-language models: acoustic field video. Unlike conventional video (RGB with stereo/mono audio), our video stream provides a spatially grounded visualization of sound intensity across a scene, offering a new and powerful dimension of perceptual understanding. Our real-time pipeline uses low-cost beamforming microphone arrays that are already common in smart speakers and increasingly present in robotics and XR headsets, yet this sensing capability remains unutilized for scene understanding. To assess the value of spatial acoustic information, we constructed an evaluation set of 402 question-answer scenes, comparing a state-of-the-art VLM given conventional video with and without paired acoustic field video. Results show a clear and consistent improvement when incorporating spatial acoustic data; the VLM we test improves from 38.3% correct to 67.4%. Our findings highlight that many everyday scene understanding tasks remain underconstrained when relying solely on visual and audio input, and that acoustic field data provides a promising and practical direction for multimodal reasoning. A video demo is available at https://daehwakim.com/seeingsound
>
---
## 更新

#### [replaced 001] Improve the autonomy of the SE2(3) group based Extended Kalman Filter for Integrated Navigation: Application
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于导航系统优化任务，旨在提升SE2(3)框架下导航模型的自主性。通过实验与仿真验证改进后的高精度导航模型性能。**

- **链接: [https://arxiv.org/pdf/2601.16078v2](https://arxiv.org/pdf/2601.16078v2)**

> **作者:** Maosong Wang; Jiarui Cui; Wenqi Wu; Peiqi Li; Xianfei Pan
>
> **备注:** arXiv admin note: substantial text overlap with arXiv:2601.16062. substantial text overlap with arXiv:2601.16062. substantial text overlap with arXiv:2601.16062. substantial text overlap with arXiv:2601.16062
>
> **摘要:** One of the core advantages of SE2(3) Lie group framework for navigation modeling lies in the autonomy of error propagation. In the previous paper, the theoretical analysis of autonomy property of navigation model in inertial, earth and world frames was given. A construction method for SE2(3) group navigation model is proposed to improve the non-inertial navigation model toward full autonomy. This paper serves as a counterpart to previous paper and conducts the real-world strapdown inertial navigation system (SINS)/odometer(ODO) experiments as well as Monte-Carlo simulations to demonstrate the performance of improved SE2(3) group based high-precision navigation models.
>
---
#### [replaced 002] Point Bridge: 3D Representations for Cross Domain Policy Learning
- **分类: cs.RO**

- **简介: 该论文提出Point Bridge框架，解决仿真到现实的策略迁移问题。通过点云表示和视觉语言模型，实现无需视觉对齐的跨域策略学习，提升真实世界操作性能。**

- **链接: [https://arxiv.org/pdf/2601.16212v2](https://arxiv.org/pdf/2601.16212v2)**

> **作者:** Siddhant Haldar; Lars Johannsmeier; Lerrel Pinto; Abhishek Gupta; Dieter Fox; Yashraj Narang; Ajay Mandlekar
>
> **摘要:** Robot foundation models are beginning to deliver on the promise of generalist robotic agents, yet progress remains constrained by the scarcity of large-scale real-world manipulation datasets. Simulation and synthetic data generation offer a scalable alternative, but their usefulness is limited by the visual domain gap between simulation and reality. In this work, we present Point Bridge, a framework that leverages unified, domain-agnostic point-based representations to unlock synthetic datasets for zero-shot sim-to-real policy transfer, without explicit visual or object-level alignment. Point Bridge combines automated point-based representation extraction via Vision-Language Models (VLMs), transformer-based policy learning, and efficient inference-time pipelines to train capable real-world manipulation agents using only synthetic data. With additional co-training on small sets of real demonstrations, Point Bridge further improves performance, substantially outperforming prior vision-based sim-and-real co-training methods. It achieves up to 44% gains in zero-shot sim-to-real transfer and up to 66% with limited real data across both single-task and multitask settings. Videos of the robot are best viewed at: https://pointbridge3d.github.io/
>
---
#### [replaced 003] InteLiPlan: An Interactive Lightweight LLM-Based Planner for Domestic Robot Autonomy
- **分类: cs.RO**

- **简介: 该论文提出InteLiPlan，一个基于大语言模型的轻量级交互式规划框架，用于提升家用机器人自主性。解决机器人任务规划与故障处理问题，通过人机协作实现高效任务执行。**

- **链接: [https://arxiv.org/pdf/2409.14506v4](https://arxiv.org/pdf/2409.14506v4)**

> **作者:** Kim Tien Ly; Kai Lu; Ioannis Havoutis
>
> **摘要:** We introduce an interactive LLM-based framework designed to enhance the autonomy and robustness of domestic robots, targeting embodied intelligence. Our approach reduces reliance on large-scale data and incorporates a robot-agnostic pipeline that embodies an LLM. Our framework, InteLiPlan, ensures that the LLM's decision-making capabilities are effectively aligned with robotic functions, enhancing operational robustness and adaptability, while our human-in-the-loop mechanism allows for real-time human intervention when user instruction is required. We evaluate our method in both simulation and on the real robot platforms, including a Toyota Human Support Robot and an ANYmal D robot with a Unitree Z1 arm. Our method achieves a 95% success rate in the `fetch me' task completion with failure recovery, highlighting its capability in both failure reasoning and task planning. InteLiPlan achieves comparable performance to state-of-the-art LLM-based robotics planners, while using only real-time onboard computing. Project website: https://kimtienly.github.io/InteLiPlan.
>
---
#### [replaced 004] Accurate Calibration and Robust LiDAR-Inertial Odometry for Spinning Actuated LiDAR Systems
- **分类: cs.RO**

- **简介: 该论文属于LiDAR定位任务，解决校准与鲁棒性问题。提出无目标校准方法和自适应LiDAR-惯性里程计，提升扫描完整性和定位稳定性。**

- **链接: [https://arxiv.org/pdf/2601.15946v2](https://arxiv.org/pdf/2601.15946v2)**

> **作者:** Zijie Chen; Xiaowei Liu; Yong Xu; Shenghai Yuan; Jianping Li; Lihua Xie
>
> **备注:** This article has been accepted for publication in IEEE Robotics and Automation Letters (RA-L). Personal use is permitted. All other uses require IEEE permission
>
> **摘要:** Accurate calibration and robust localization are fundamental for downstream tasks in spinning actuated LiDAR applications. Existing methods, however, require parameterizing extrinsic parameters based on different mounting configurations, limiting their generalizability. Additionally, spinning actuated LiDAR inevitably scans featureless regions, which complicates the balance between scanning coverage and localization robustness. To address these challenges, this letter presents a targetless LiDAR-motor calibration (LM-Calibr) on the basis of the Denavit-Hartenberg convention and an environmental adaptive LiDAR-inertial odometry (EVA-LIO). LM-Calibr supports calibration of LiDAR-motor systems with various mounting configurations. Extensive experiments demonstrate its accuracy and convergence across different scenarios, mounting angles, and initial values. Additionally, EVA-LIO adaptively selects downsample rates and map resolutions according to spatial scale. This adaptivity enables the actuator to operate at maximum speed, thereby enhancing scanning completeness while ensuring robust localization, even when LiDAR briefly scans featureless areas. The source code and hardware design are available on GitHub: \textcolor{blue}{\href{https://github.com/zijiechenrobotics/lm_calibr}{github.com/zijiechenrobotics/lm\_calibr}}. The video is available at \textcolor{blue}{\href{https://youtu.be/cZyyrkmeoSk}{youtu.be/cZyyrkmeoSk}}
>
---
#### [replaced 005] Who Is Responsible? Self-Adaptation Under Multiple Concurrent Uncertainties With Unknown Sources in Complex ROS-Based Systems
- **分类: cs.RO**

- **简介: 该论文属于机器人自适应任务，解决复杂ROS系统中多并发不确定性问题。提出基于MAPE-K的自适应方法，处理多重不确定性、级联故障及多种解决策略，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2504.20477v3](https://arxiv.org/pdf/2504.20477v3)**

> **作者:** Andreas Wiedholz; Rafael Paintner; Julian Gleißner; Alwin Hoffmann; Tobias Huber
>
> **摘要:** Robotic systems increasingly operate in dynamic, unpredictable environments, where tightly coupled sensors and software modules increase the probability of a single fault cascading across components and admitting multiple plausible strategies to resolve the underlying uncertainty. Most existing self-adaptive approaches that have been applied to robotics assume predefined one-to-one uncertainty-to-adaptation mappings. We present a ROS2-based self-adaptive approach building upon the MAPE-K feedback loop that addresses (1) multiple simultaneous uncertainties with differing criticality, (2) cascading uncertainties across components, and (3) multiple plausible resolving strategies per detected symptom. Central to our approach is an adaptation rule set which lets designers specify uncertainty patterns, assign criticality levels, and enumerate multiple plausible adaptation strategies. This rule set, combined with an automatically extracted live ROS2 dependency graph, enables lightweight root-cause analysis and strategy ranking to prioritize minimal and effective adaptations. Evaluations on an underwater robot scenario and a perception use case show that our approach can identify root causes among concurrent uncertainties, favours inexpensive adaptations, reduces unnecessary adaptations, and achieves performance comparable to existing baselines designed for sequential uncertainties. The code is publicly available.
>
---
#### [replaced 006] Kinematify: Open-Vocabulary Synthesis of High-DoF Articulated Objects
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Kinematify，解决高自由度关节物体的自动合成问题。通过结合搜索与优化，从图像或文本生成物理合理的关节结构。**

- **链接: [https://arxiv.org/pdf/2511.01294v3](https://arxiv.org/pdf/2511.01294v3)**

> **作者:** Jiawei Wang; Dingyou Wang; Jiaming Hu; Qixuan Zhang; Jingyi Yu; Lan Xu
>
> **备注:** Project Page: https://sites.google.com/deemos.com/kinematify
>
> **摘要:** A deep understanding of kinematic structures and movable components is essential for enabling robots to manipulate objects and model their own articulated forms. Such understanding is captured through articulated objects, which are essential for tasks such as physical simulation, motion planning, and policy learning. However, creating these models, particularly for objects with high degrees of freedom (DoF), remains a significant challenge. Existing methods typically rely on motion sequences or strong assumptions from hand-curated datasets, which hinders scalability. In this paper, we introduce Kinematify, an automated framework that synthesizes articulated objects directly from arbitrary RGB images or textual descriptions. Our method addresses two core challenges: (i) inferring kinematic topologies for high-DoF objects and (ii) estimating joint parameters from static geometry. To achieve this, we combine MCTS search for structural inference with geometry-driven optimization for joint reasoning, producing physically consistent and functionally valid descriptions. We evaluate Kinematify on diverse inputs from both synthetic and real-world environments, demonstrating improvements in registration and kinematic topology accuracy over prior work.
>
---
#### [replaced 007] Foundation Models in Autonomous Driving: A Survey on Scenario Generation and Scenario Analysis
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶领域，旨在解决复杂场景生成与分析问题。通过调研基础模型的应用，提出统一分类并总结方法、数据集及评估指标。**

- **链接: [https://arxiv.org/pdf/2506.11526v3](https://arxiv.org/pdf/2506.11526v3)**

> **作者:** Yuan Gao; Mattia Piccinini; Yuchen Zhang; Dingrui Wang; Korbinian Moller; Roberto Brusnicki; Baha Zarrouki; Alessio Gambi; Jan Frederik Totz; Kai Storms; Steven Peters; Andrea Stocco; Bassam Alrifaee; Marco Pavone; Johannes Betz
>
> **备注:** Final version (Accepted by the IEEE Open Journal of Intelligent Transportation Systems)
>
> **摘要:** For autonomous vehicles, safe navigation in complex environments depends on handling a broad range of diverse and rare driving scenarios. Simulation- and scenario-based testing have emerged as key approaches to development and validation of autonomous driving systems. Traditional scenario generation relies on rule-based systems, knowledge-driven models, and data-driven synthesis, often producing limited diversity and unrealistic safety-critical cases. With the emergence of foundation models, which represent a new generation of pre-trained, general-purpose AI models, developers can process heterogeneous inputs (e.g., natural language, sensor data, HD maps, and control actions), enabling the synthesis and interpretation of complex driving scenarios. In this paper, we conduct a survey about the application of foundation models for scenario generation and scenario analysis in autonomous driving (as of May 2025). Our survey presents a unified taxonomy that includes large language models, vision-language models, multimodal large language models, diffusion models, and world models for the generation and analysis of autonomous driving scenarios. In addition, we review the methodologies, open-source datasets, simulation platforms, and benchmark challenges, and we examine the evaluation metrics tailored explicitly to scenario generation and analysis. Finally, the survey concludes by highlighting the open challenges and research questions, and outlining promising future research directions. All reviewed papers are listed in a continuously maintained repository, which contains supplementary materials and is available at https://github.com/TUM-AVS/FM-for-Scenario-Generation-Analysis.
>
---
#### [replaced 008] Path Planning using a One-shot-sampling Skeleton Map
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于路径规划任务，旨在解决计算安全、快速路径的问题。提出SkelUnet方法，利用深度学习高效生成骨架图，提升路径规划效率与安全性。**

- **链接: [https://arxiv.org/pdf/2507.02328v2](https://arxiv.org/pdf/2507.02328v2)**

> **作者:** Gabriel O. Flores-Aquino; Octavio Gutierrez-Frias; Juan Irving Vasquez
>
> **备注:** Submitted to IEEE Latin America Transactions
>
> **摘要:** Path planning algorithms fundamentally aim to compute collision-free paths, with many works focusing on finding the optimal distance path. However, for several applications, a more suitable approach is to balance response time, path safety, and path length. In this context, a skeleton map is a useful tool in graph-based schemes, as it provides an intrinsic representation of the free workspace. However, standard skeletonization algorithms are computationally expensive, as they are primarly oriented towards image processing tasks. We propose an efficient path-planning methodology that finds safe paths within an acceptable processing time. This methodology leverages a Deep Denoising Autoencoder (DDAE) based on the U-Net architecture to compute a skeletonized version of the navigation map, which we refer to as SkelUnet. The SkelUnet network facilitates exploration of the entire workspace through one-shot sampling (OSS), as opposed to the iterative or probabilistic sampling used by previous algorithms. SkelUnet is trained and tested on a dataset consisting of 12,500 two-dimensional dungeon maps. The motion planning methodology is evaluated in a simulation environment with an Unmanned Aerial Vehicle (UAV) in 250 previously unseen maps and assessed using several navigation metrics to quantify the navigability of the computed paths. The results demonstrate that using SkelUnet to construct the roadmap offers significant advantages, such as connecting all regions of free workspace, providing safer paths, and reducing processing time.
>
---
#### [replaced 009] DGFusion: Depth-Guided Sensor Fusion for Robust Semantic Perception
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于多模态传感器融合任务，旨在提升自动驾驶中的语义感知鲁棒性。通过引入深度信息，提出DGFusion方法，解决传感器在复杂环境下性能下降的问题。**

- **链接: [https://arxiv.org/pdf/2509.09828v3](https://arxiv.org/pdf/2509.09828v3)**

> **作者:** Tim Broedermannn; Christos Sakaridis; Luigi Piccinelli; Wim Abbeloos; Luc Van Gool
>
> **备注:** Code and models are available at https://github.com/timbroed/DGFusion
>
> **摘要:** Robust semantic perception for autonomous vehicles relies on effectively combining multiple sensors with complementary strengths and weaknesses. State-of-the-art sensor fusion approaches to semantic perception often treat sensor data uniformly across the spatial extent of the input, which hinders performance when faced with challenging conditions. By contrast, we propose a novel depth-guided multimodal fusion method that upgrades condition-aware fusion by integrating depth information. Our network, DGFusion, poses multimodal segmentation as a multi-task problem, utilizing the lidar measurements, which are typically available in outdoor sensor suites, both as one of the model's inputs and as ground truth for learning depth. Our corresponding auxiliary depth head helps to learn depth-aware features, which are encoded into spatially varying local depth tokens that condition our attentive cross-modal fusion. Together with a global condition token, these local depth tokens dynamically adapt sensor fusion to the spatially varying reliability of each sensor across the scene, which largely depends on depth. In addition, we propose a robust loss for our depth, which is essential for learning from lidar inputs that are typically sparse and noisy in adverse conditions. Our method achieves state-of-the-art panoptic and semantic segmentation performance on the challenging MUSES and DeLiVER datasets. Code and models are available at https://github.com/timbroed/DGFusion
>
---
#### [replaced 010] Model-free source seeking of exponentially convergent unicycle: theoretical and robotic experimental results
- **分类: math.OC; cs.RO**

- **简介: 该论文属于源定位任务，解决未知信号下的极值点搜索问题。提出一种无需模型的实时控制方法，实现无人车对非二次型信号的指数收敛定位。**

- **链接: [https://arxiv.org/pdf/2511.00752v3](https://arxiv.org/pdf/2511.00752v3)**

> **作者:** Rohan Palanikumar; Ahmed A. Elgohary; Victoria Grushkovskaya; Sameh A. Eisa
>
> **摘要:** This paper introduces a novel model-free, real-time unicycle-based source seeking design. This design autonomously steers the unicycle dynamic system towards the extremum point of an objective function or physical/scalar signal that is unknown expression-wise, but accessible via measurements. A key contribution of this paper is that the introduced design converges exponentially to the extremum point of objective functions (or scalar signals) that behave locally like a higher-degree power function (e.g., fourth-degree polynomial function) as opposed to locally quadratic objective functions, the usual case in literature. We provide theoretical results and design characterization, supported by a variety of simulation results that demonstrate the robustness of the proposed design, including cases with different initial conditions and measurement delays/noise. Also, for the first time in the literature, we provide experimental robotic results that demonstrate the effectiveness of the proposed design and its exponential convergence ability. These experimental results confirm that the proposed exponentially convergent extremum seeking design can be practically realized on a physical robotic platform under real-world sensing and actuation constraints.
>
---
#### [replaced 011] Vision-Proprioception Fusion with Mamba2 in End-to-End Reinforcement Learning for Motion Control
- **分类: cs.RO; cs.AI; cs.CV; eess.IV; eess.SY**

- **简介: 该论文属于运动控制任务，解决端到端强化学习中感知融合效率低的问题。提出SSD-Mamba2框架，实现视觉与本体感知的有效融合，提升控制性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2509.07593v2](https://arxiv.org/pdf/2509.07593v2)**

> **作者:** Xiaowen Tao; Yinuo Wang; Jinzhao Zhou
>
> **备注:** 6 figures and 8 tables. This paper has been accepted by Advanced Engineering Informatics
>
> **摘要:** End-to-end reinforcement learning (RL) for motion control trains policies directly from sensor inputs to motor commands, enabling unified controllers for different robots and tasks. However, most existing methods are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for resource-constrained robotic and autonomous systems in engineering informatics applications.
>
---
#### [replaced 012] Monocular pose estimation of articulated open surgery tools -- in the wild
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于单目6D姿态估计任务，解决开放手术中器械姿态估计问题，通过合成数据生成、姿态估计框架和域适应策略实现精准定位。**

- **链接: [https://arxiv.org/pdf/2407.12138v2](https://arxiv.org/pdf/2407.12138v2)**

> **作者:** Robert Spektor; Tom Friedman; Itay Or; Gil Bolotin; Shlomi Laufer
>
> **备注:** Author Accepted Manuscript (AAM)
>
> **摘要:** This work presents a framework for monocular 6D pose estimation of surgical instruments in open surgery, addressing challenges such as object articulations, specularity, occlusions, and synthetic-to-real domain adaptation. The proposed approach consists of three main components: $(1)$ synthetic data generation pipeline that incorporates 3D scanning of surgical tools with articulation rigging and physically-based rendering; $(2)$ a tailored pose estimation framework combining tool detection with pose and articulation estimation; and $(3)$ a training strategy on synthetic and real unannotated video data, employing domain adaptation with automatically generated pseudo-labels. Evaluations conducted on real data of open surgery demonstrate the good performance and real-world applicability of the proposed framework, highlighting its potential for integration into medical augmented reality and robotic systems. The approach eliminates the need for extensive manual annotation of real surgical data.
>
---
#### [replaced 013] LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，解决模型在新指令和复杂任务中泛化能力差的问题。通过引入贝叶斯分解方法，增强语言指导的准确性。**

- **链接: [https://arxiv.org/pdf/2601.15197v3](https://arxiv.org/pdf/2601.15197v3)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose LangForce, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, LangForce significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [replaced 014] Contact SLAM: An Active Tactile Exploration Policy Based on Physical Reasoning Utilized in Robotic Fine Blind Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人精细盲操作任务，解决视觉受限下的环境感知问题。提出Contact SLAM方法，利用触觉传感和先验知识实现精准操控。**

- **链接: [https://arxiv.org/pdf/2512.10481v2](https://arxiv.org/pdf/2512.10481v2)**

> **作者:** Gaozhao Wang; Xing Liu; Zhenduo Ye; Zhengxiong Liu; Panfeng Huang
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Contact-rich manipulation is difficult for robots to execute and requires accurate perception of the environment. In some scenarios, vision is occluded. The robot can then no longer obtain real-time scene state information through visual feedback. This is called ``blind manipulation". In this manuscript, a novel physically-driven contact cognition method, called ``Contact SLAM", is proposed. It estimates the state of the environment and achieves manipulation using only tactile sensing and prior knowledge of the scene. To maximize exploration efficiency, this manuscript also designs an active exploration policy. The policy gradually reduces uncertainties in the manipulation scene. The experimental results demonstrated the effectiveness and accuracy of the proposed method in several contact-rich tasks, including the difficult and delicate socket assembly task and block-pushing task.
>
---
#### [replaced 015] ORION: Option-Regularized Deep Reinforcement Learning for Cooperative Multi-Agent Online Navigation
- **分类: cs.RO**

- **简介: 该论文提出ORION框架，解决部分已知环境中多智能体协作导航问题，通过深度强化学习实现高效协同与地图不确定性降低。**

- **链接: [https://arxiv.org/pdf/2601.01155v2](https://arxiv.org/pdf/2601.01155v2)**

> **作者:** Shizhe Zhang; Jingsong Liang; Zhitao Zhou; Shuhan Ye; Yizhuo Wang; Ming Siang Derek Tan; Jimmy Chiun; Yuhong Cao; Guillaume Sartoretti
>
> **摘要:** Existing methods for multi-agent navigation typically assume fully known environments, offering limited support for partially known scenarios such as warehouses or factory floors. There, agents may need to plan trajectories that balance their own path optimality with their ability to collect and share information about the environment that can help their teammates reach their own goals. To these ends, we propose ORION, a novel deep reinforcement learning framework for cooperative multi-agent online navigation in partially known environments. Starting from an imperfect prior map, ORION trains agents to make decentralized decisions, coordinate to reach their individual targets, and actively reduce map uncertainty by sharing online observations in a closed perception-action loop. We first design a shared graph encoder that fuses prior map with online perception into a unified representation, providing robust state embeddings under dynamic map discrepancies. At the core of ORION is an option-critic framework that learns to reason about a set of high-level cooperative modes that translate into sequences of low-level actions, allowing agents to switch between individual navigation and team-level exploration adaptively. We further introduce a dual-stage cooperation strategy that enables agents to assist teammates under map uncertainty, thereby reducing the overall makespan. Across extensive maze-like maps and large-scale warehouse environments, our simulation results show that ORION achieves high-quality, real-time decentralized cooperation over varying team sizes, outperforming state-of-the-art classical and learning-based baselines. Finally, we validate ORION on physical robot teams, demonstrating its robustness and practicality for real-world cooperative navigation.
>
---
#### [replaced 016] Cross-Platform Scaling of Vision-Language-Action Models from Edge to Cloud GPUs
- **分类: cs.AI; cs.CV; cs.ET; cs.LG; cs.RO**

- **简介: 该论文研究跨平台VLA模型的性能扩展问题，评估不同架构和硬件上的表现，旨在优化机器人控制中的模型部署。**

- **链接: [https://arxiv.org/pdf/2509.11480v2](https://arxiv.org/pdf/2509.11480v2)**

> **作者:** Amir Taherin; Juyi Lin; Arash Akbari; Arman Akbari; Pu Zhao; Weiwei Chen; David Kaeli; Yanzhi Wang
>
> **备注:** To appear in the Asilomar Conference on Signals, Systems, and Computers 2025
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as powerful generalist policies for robotic control, yet their performance scaling across model architectures and hardware platforms, as well as their associated power budgets, remain poorly understood. This work presents an evaluation of five representative VLA models -- spanning state-of-the-art baselines and two newly proposed architectures -- targeting edge and datacenter GPU platforms. Using the LIBERO benchmark, we measure accuracy alongside system-level metrics, including latency, throughput, and peak memory usage, under varying edge power constraints and high-performance datacenter GPU configurations. Our results identify distinct scaling trends: (1) architectural choices, such as action tokenization and model backbone size, strongly influence throughput and memory footprint; (2) power-constrained edge devices exhibit non-linear performance degradation, with some configurations matching or exceeding older datacenter GPUs; and (3) high-throughput variants can be achieved without significant accuracy loss. These findings provide actionable insights when selecting and optimizing VLAs across a range of deployment constraints. Our work challenges current assumptions about the superiority of datacenter hardware for robotic inference.
>
---
#### [replaced 017] Actor-Critic Cooperative Compensation to Model Predictive Control for Off-Road Autonomous Vehicles Under Unknown Dynamics
- **分类: cs.RO**

- **简介: 该论文属于自主车辆控制任务，解决未知动态下的轨迹跟踪问题。提出AC3MPC框架，结合模型预测控制与深度强化学习，提升越野自动驾驶性能。**

- **链接: [https://arxiv.org/pdf/2503.00577v2](https://arxiv.org/pdf/2503.00577v2)**

> **作者:** Prakhar Gupta; Jonathon M Smereka; Yunyi Jia
>
> **备注:** 7 pages, Accepted at 2025 IEEE ICRA
>
> **摘要:** This study presents an Actor-Critic Cooperative Compensated Model Predictive Controller (AC3MPC) designed to address unknown system dynamics. To avoid the difficulty of modeling highly complex dynamics and ensuring realtime control feasibility and performance, this work uses deep reinforcement learning with a model predictive controller in a cooperative framework to handle unknown dynamics. The model-based controller takes on the primary role as both controllers are provided with predictive information about the other. This improves tracking performance and retention of inherent robustness of the model predictive controller. We evaluate this framework for off-road autonomous driving on unknown deformable terrains that represent sandy deformable soil, sandy and rocky soil, and cohesive clay-like deformable soil. Our findings demonstrate that our controller statistically outperforms standalone model-based and learning-based controllers by upto 29.2% and 10.2%. This framework generalized well over varied and previously unseen terrain characteristics to track longitudinal reference speeds with lower errors. Furthermore, this required significantly less training data compared to purely learning-based controller, while delivering better performance even when under-trained.
>
---
#### [replaced 018] A Computationally Efficient Maximum A Posteriori Sequence Estimation via Stein Variational Inference
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决多模态后验分布下的高效最大后验序列估计问题。提出Stein-MAP-Seq方法，结合变分推断与SVGD，提升计算效率和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2312.08684v4](https://arxiv.org/pdf/2312.08684v4)**

> **作者:** Min-Won Seo; Solmaz S. Kia
>
> **备注:** 20 pages
>
> **摘要:** State estimation in robotic systems presents significant challenges, particularly due to the prevalence of multimodal posterior distributions in real-world scenarios. One effective strategy for handling such complexity is to compute maximum a posteriori (MAP) sequences over a discretized or sampled state space, which enables a concise representation of the most likely state trajectory. However, this approach often incurs substantial computational costs, especially in high-dimensional settings. In this article, we propose a novel MAP sequence estimation method, Stein-MAP-Seq, which effectively addresses multimodality while substantially reducing computational and memory overhead. Our key contribution is a sequential variational inference framework that captures temporal dependencies in dynamical system models and integrates Stein variational gradient descent (SVGD) into a Viterbi-style dynamic programming algorithm, enabling computationally efficient MAP sequence estimation. This integration allows the method to focus computational effort on MAP-consistent modes rather than exhaustively exploring the entire state space. Stein-MAP-Seq inherits the parallelism and mode-seeking behavior of SVGD, allowing particle updates to be efficiently executed on parallel hardware and significantly reducing the number of trajectory candidates required for MAP-sequence recursion compared to conventional methods that rely on hundreds to thousands of particles. We validate the proposed approach on a range of highly multimodal scenarios, including nonlinear dynamics with ambiguous observations, unknown data association with outliers, range-only localization under temporary unobservability, and high-dimensional robotic manipulators. Experimental results demonstrate substantial improvements in estimation accuracy and robustness to multimodality over existing estimation methods.
>
---
#### [replaced 019] Safe Learning for Contact-Rich Robot Tasks: A Survey from Classical Learning-Based Methods to Safe Foundation Models
- **分类: cs.RO**

- **简介: 该论文属于机器人接触任务领域，旨在解决安全学习问题。通过分类回顾安全探索与执行方法，探讨其在基础模型中的应用与挑战。**

- **链接: [https://arxiv.org/pdf/2512.11908v2](https://arxiv.org/pdf/2512.11908v2)**

> **作者:** Heng Zhang; Rui Dai; Gokhan Solak; Pokuang Zhou; Yu She; Arash Ajoudani
>
> **备注:** version 2
>
> **摘要:** Contact-rich tasks pose significant challenges for robotic systems due to inherent uncertainty, complex dynamics, and the high risk of damage during interaction. Recent advances in learning-based control have shown great potential in enabling robots to acquire and generalize complex manipulation skills in such environments, but ensuring safety, both during exploration and execution, remains a critical bottleneck for reliable real-world deployment. This survey provides a comprehensive overview of safe learning-based methods for robot contact-rich tasks. We categorize existing approaches into two main domains: safe exploration and safe execution. We review key techniques, including constrained reinforcement learning, risk-sensitive optimization, uncertainty-aware modeling, control barrier functions, and model predictive safety shields, and highlight how these methods incorporate prior knowledge, task structure, and online adaptation to balance safety and efficiency. A particular emphasis of this survey is on how these safe learning principles extend to and interact with emerging robotic foundation models, especially vision-language models (VLMs) and vision-language-action models (VLAs), which unify perception, language, and control for contact-rich manipulation. We discuss both the new safety opportunities enabled by VLM/VLA-based methods, such as language-level specification of constraints and multimodal grounding of safety signals, and the amplified risks and evaluation challenges they introduce. Finally, we outline current limitations and promising future directions toward deploying reliable, safety-aligned, and foundation-model-enabled robots in complex contact-rich environments. More details and materials are available at our \href{ https://github.com/jack-sherman01/Awesome-Learning4Safe-Contact-rich-tasks}{Project GitHub Repository}.
>
---
#### [replaced 020] Gaussian Variational Inference with Non-Gaussian Factors for State Estimation: A UWB Localization Case Study
- **分类: cs.RO; stat.ML**

- **简介: 该论文属于状态估计任务，解决UWB定位中的非高斯噪声问题。通过扩展ESGVI算法，引入非高斯因子，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.19855v2](https://arxiv.org/pdf/2512.19855v2)**

> **作者:** Andrew Stirling; Mykola Lukashchuk; Dmitry Bagaev; Wouter Kouw; James R. Forbes
>
> **摘要:** This letter extends the exactly sparse Gaussian variational inference (ESGVI) algorithm for state estimation in two complementary directions. First, ESGVI is generalized to operate on matrix Lie groups, enabling the estimation of states with orientation components while respecting the underlying group structure. Second, factors are introduced to accommodate heavy-tailed and skewed noise distributions, as commonly encountered in ultra-wideband (UWB) localization due to non-line-of-sight (NLOS) and multipath effects. Both extensions are shown to integrate naturally within the ESGVI framework while preserving its sparse and derivative-free structure. The proposed approach is validated in a UWB localization experiment with NLOS-rich measurements, demonstrating improved accuracy and comparable consistency. Finally, a Python implementation within a factor-graph-based estimation framework is made open-source (https://github.com/decargroup/gvi_ws) to support broader research use.
>
---
#### [replaced 021] Semantic2D: Enabling Semantic Scene Understanding with 2D Lidar Alone
- **分类: cs.RO**

- **简介: 该论文属于语义场景理解任务，解决2D激光雷达的语义分割问题，提出首个公开数据集和细粒度算法，实现点级语义标注与导航应用。**

- **链接: [https://arxiv.org/pdf/2409.09899v2](https://arxiv.org/pdf/2409.09899v2)**

> **作者:** Zhanteng Xie; Yipeng Pan; Yinqiang Zhang; Jia Pan; Philip Dames
>
> **摘要:** This article presents a complete semantic scene understanding workflow using only a single 2D lidar. This fills the gap in 2D lidar semantic segmentation, thereby enabling the rethinking and enhancement of existing 2D lidar-based algorithms for application in various mobile robot tasks. It introduces the first publicly available 2D lidar semantic segmentation dataset and the first fine-grained semantic segmentation algorithm specifically designed for 2D lidar sensors on autonomous mobile robots. To annotate this dataset, we propose a novel semi-automatic semantic labeling framework that requires minimal human effort and provides point-level semantic annotations. The data was collected by three different types of 2D lidar sensors across twelve indoor environments, featuring a range of common indoor objects. Furthermore, the proposed semantic segmentation algorithm fully exploits raw lidar information -- position, range, intensity, and incident angle -- to deliver stochastic, point-wise semantic segmentation. We present a series of semantic occupancy grid mapping experiments and demonstrate two semantically-aware navigation control policies based on 2D lidar. These results demonstrate that the proposed semantic 2D lidar dataset, semi-automatic labeling framework, and segmentation algorithm are effective and can enhance different components of the robotic navigation pipeline. Multimedia resources are available at: https://youtu.be/P1Hsvj6WUSY.
>
---
#### [replaced 022] PointMapPolicy: Structured Point Cloud Processing for Multi-Modal Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决多模态感知中的精度与泛化问题。提出PointMapPolicy，结合点云与RGB数据，提升环境理解能力。**

- **链接: [https://arxiv.org/pdf/2510.20406v3](https://arxiv.org/pdf/2510.20406v3)**

> **作者:** Xiaogang Jia; Qian Wang; Anrui Wang; Han A. Wang; Balázs Gyenes; Emiliyan Gospodinov; Xinkai Jiang; Ge Li; Hongyi Zhou; Weiran Liao; Xi Huang; Maximilian Beck; Moritz Reuss; Rudolf Lioutikov; Gerhard Neumann
>
> **摘要:** Robotic manipulation systems benefit from complementary sensing modalities, where each provides unique environmental information. Point clouds capture detailed geometric structure, while RGB images provide rich semantic context. Current point cloud methods struggle to capture fine-grained detail, especially for complex tasks, which RGB methods lack geometric awareness, which hinders their precision and generalization. We introduce PointMapPolicy, a novel approach that conditions diffusion policies on structured grids of points without downsampling. The resulting data type makes it easier to extract shape and spatial relationships from observations, and can be transformed between reference frames. Yet due to their structure in a regular grid, we enable the use of established computer vision techniques directly to 3D data. Using xLSTM as a backbone, our model efficiently fuses the point maps with RGB data for enhanced multi-modal perception. Through extensive experiments on the RoboCasa and CALVIN benchmarks and real robot evaluations, we demonstrate that our method achieves state-of-the-art performance across diverse manipulation tasks. The overview and demos are available on our project page: https://point-map.github.io/Point-Map/
>
---
#### [replaced 023] Estimating the Joint Probability of Scenario Parameters with Gaussian Mixture Copula Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于场景参数联合概率建模任务，旨在提升自动驾驶系统安全验证的准确性。通过引入高斯混合Copula模型，解决场景参数依赖关系建模问题，优于传统方法。**

- **链接: [https://arxiv.org/pdf/2506.10098v4](https://arxiv.org/pdf/2506.10098v4)**

> **作者:** Christian Reichenbächer; Philipp Rank; Jochen Hipp; Oliver Bringmann
>
> **备注:** 9 pages, 4 figures; This work has been submitted to the IEEE for possible publication; Code available at: https://codeocean.com/capsule/1003615/tree
>
> **摘要:** This paper presents the first application of Gaussian Mixture Copula Models to the statistical modeling of driving scenarios for the safety validation of automated driving systems. Knowledge of the joint probability distribution of scenario parameters is essential for scenario-based safety assessment, where risk quantification depends on the likelihood of concrete parameter combinations. Gaussian Mixture Copula Models bring together the multimodal expressivity of Gaussian Mixture Models and the flexibility of copulas, enabling separate modeling of marginal distributions and dependence. We benchmark Gaussian Mixture Copula Models against previously proposed approaches - Gaussian Mixture Models and Gaussian Copula Models - using real-world driving data drawn from two scenarios defined in United Nations Regulation No. 157. Our evaluation on approximately 18 million instances of these two scenarios demonstrates that Gaussian Mixture Copula Models consistently surpass Gaussian Copula Models and perform competitively with Gaussian Mixture Models, as measured by both log-likelihood and Sinkhorn distance, with relative performance depending on the scenario. The results are promising for the adoption of Gaussian Mixture Copula Models as a statistical foundation for future scenario-based validation frameworks.
>
---
#### [replaced 024] Close-Fitting Dressing Assistance Based on State Estimation of Feet and Garments with Semantic-based Visual Attention
- **分类: cs.RO**

- **简介: 该论文属于机器人穿衣辅助任务，旨在解决为老年人或行动不便者精准穿紧身衣物的问题。通过多模态感知与语义视觉注意力，提升对衣物和脚部状态的估计能力，实现安全高效的穿衣辅助。**

- **链接: [https://arxiv.org/pdf/2505.03400v2](https://arxiv.org/pdf/2505.03400v2)**

> **作者:** Takuma Tsukakoshi; Tamon Miyake; Tetsuya Ogata; Yushi Wang; Takumi Akaishi; Shigeki Sugano
>
> **备注:** Accepted at RA-L, 2026
>
> **摘要:** As the population continues to age, a shortage of caregivers is expected in the future. Dressing assistance, in particular, is crucial for opportunities for social participation. Especially dressing close-fitting garments, such as socks, remains challenging due to the need for fine force adjustments to handle the friction or snagging against the skin, while considering the shape and position of the garment. This study introduces a method uses multi-modal information including not only robot's camera images, joint angles, joint torques, but also tactile forces for proper force interaction that can adapt to individual differences in humans. Furthermore, by introducing semantic information based on object concepts, rather than relying solely on RGB data, it can be generalized to unseen feet and background. In addition, incorporating depth data helps infer relative spatial relationship between the sock and the foot. To validate its capability for semantic object conceptualization and to ensure safety, training data were collected using a mannequin, and subsequent experiments were conducted with human subjects. In experiments, the robot successfully adapted to previously unseen human feet and was able to put socks on 10 participants, achieving a higher success rate than Action Chunking with Transformer and Diffusion Policy. These results demonstrate that the proposed model can estimate the state of both the garment and the foot, enabling precise dressing assistance for close-fitting garments.
>
---
#### [replaced 025] Towards Real-time Adaptation of Embodied Agent in Human-Robot Collaboration
- **分类: cs.AI; cs.HC; cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决实时适应问题。提出MonTA框架，提升代理的响应与适应能力，优化协作效率。**

- **链接: [https://arxiv.org/pdf/2412.00435v2](https://arxiv.org/pdf/2412.00435v2)**

> **作者:** Shipeng Liu; Boshen Zhang; Zhehui Huang
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Large Language Models (LLMs) have opened transformative possibilities for human-robot collaboration. However, enabling real-time collaboration requires both low latency and robust reasoning, and most LLMs suffer from high latency. To address this gap, we first propose a fine-grained benchmark that explicitly assesses agents' proactive adaptability and temporal responsiveness in the Overcooked-AI environment. Based on evaluation results, we propose MonTA (Monitor-then-Adapt), a hierarchical framework inspired by cognitive science research. MonTA contains three key modules: a lightweight Monitor that operates at high frequency (7 Hz) to detect adaptation needs, and two proficient Adapters for subtask and path adaptation reasoning that provide instructions to humans at a lower frequency. Our results demonstrate that MonTA significantly outperforms baseline agents on our proposed benchmark, achieving superior performance across layouts with varying teaming fluency. User studies confirm the high reasonableness of adaptation plans and consistent language instructions provided by our framework to humans.
>
---
#### [replaced 026] GPA-VGGT:Adapting VGGT to Large Scale Localization by Self-Supervised Learning with Geometry and Physics Aware Loss
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决无监督下大场景定位问题。通过自监督学习和几何物理损失，提升VGGT模型的定位能力。**

- **链接: [https://arxiv.org/pdf/2601.16885v2](https://arxiv.org/pdf/2601.16885v2)**

> **作者:** Yangfan Xu; Lilian Zhang; Xiaofeng He; Pengdong Wu; Wenqi Wu; Jun Mao
>
> **摘要:** Transformer-based general visual geometry frameworks have shown promising performance in camera pose estimation and 3D scene understanding. Recent advancements in Visual Geometry Grounded Transformer (VGGT) models have shown great promise in camera pose estimation and 3D reconstruction. However, these models typically rely on ground truth labels for training, posing challenges when adapting to unlabeled and unseen scenes. In this paper, we propose a self-supervised framework to train VGGT with unlabeled data, thereby enhancing its localization capability in large-scale environments. To achieve this, we extend conventional pair-wise relations to sequence-wise geometric constraints for self-supervised learning. Specifically, in each sequence, we sample multiple source frames and geometrically project them onto different target frames, which improves temporal feature consistency. We formulate physical photometric consistency and geometric constraints as a joint optimization loss to circumvent the requirement for hard labels. By training the model with this proposed method, not only the local and global cross-view attention layers but also the camera and depth heads can effectively capture the underlying multi-view geometry. Experiments demonstrate that the model converges within hundreds of iterations and achieves significant improvements in large-scale localization. Our code will be released at https://github.com/X-yangfan/GPA-VGGT.
>
---
#### [replaced 027] Cross-Level Sensor Fusion with Object Lists via Transformer for 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，解决传感器数据以对象列表形式存在的融合问题。提出基于Transformer的跨层级融合方法，结合对象列表与图像信息提升检测性能。**

- **链接: [https://arxiv.org/pdf/2512.12884v2](https://arxiv.org/pdf/2512.12884v2)**

> **作者:** Xiangzhong Liu; Jiajie Zhang; Hao Shen
>
> **备注:** 6 pages, 3 figures, accepted at IV2025
>
> **摘要:** In automotive sensor fusion systems, smart sensors and Vehicle-to-Everything (V2X) modules are commonly utilized. Sensor data from these systems are typically available only as processed object lists rather than raw sensor data from traditional sensors. Instead of processing other raw data separately and then fusing them at the object level, we propose an end-to-end cross-level fusion concept with Transformer, which integrates highly abstract object list information with raw camera images for 3D object detection. Object lists are fed into a Transformer as denoising queries and propagated together with learnable queries through the latter feature aggregation process. Additionally, a deformable Gaussian mask, derived from the positional and size dimensional priors from the object lists, is explicitly integrated into the Transformer decoder. This directs attention toward the target area of interest and accelerates model training convergence. Furthermore, as there is no public dataset containing object lists as a standalone modality, we propose an approach to generate pseudo object lists from ground-truth bounding boxes by simulating state noise and false positives and negatives. As the first work to conduct cross-level fusion, our approach shows substantial performance improvements over the vision-based baseline on the nuScenes dataset. It demonstrates its generalization capability over diverse noise levels of simulated object lists and real detectors.
>
---
#### [replaced 028] Offline Reinforcement Learning using Human-Aligned Reward Labeling for Autonomous Emergency Braking in Occluded Pedestrian Crossing
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自动驾驶任务，解决真实数据缺乏有效奖励标签的问题。通过生成符合人类安全判断的奖励标签，提升离线强化学习效果。**

- **链接: [https://arxiv.org/pdf/2504.08704v2](https://arxiv.org/pdf/2504.08704v2)**

> **作者:** Vinal Asodia; Barkin Dagda; Yinglong He; Zhenhua Feng; Saber Fallah
>
> **备注:** 39 pages, 14 figures, 1 table
>
> **摘要:** Effective leveraging of real-world driving datasets is crucial for enhancing the training of autonomous driving systems. While Offline Reinforcement Learning enables training autonomous vehicles with such data, most available datasets lack meaningful reward labels. Reward labeling is essential as it provides feedback for the learning algorithm to distinguish between desirable and undesirable behaviors, thereby improving policy performance. This paper presents a novel approach for generating human-aligned reward labels. The proposed approach addresses the challenge of absent reward signals in the real-world datasets by generating labels that reflect human judgment and safety considerations. The reward function incorporates an adaptive safety component that is activated by analyzing semantic segmentation maps, enabling the autonomous vehicle to prioritize safety over efficiency in potential collision scenarios. The proposed method is applied to an occluded pedestrian crossing scenario with varying pedestrian traffic levels, using simulation data. When the generated rewards were used to train various Offline Reinforcement Learning algorithms, each model produced a meaningful policy, demonstrating the method's viability. In addition, the method was applied to a subset of the Audi Autonomous Driving Dataset, and the reward labels were compared to human-annotated reward labels. The findings show a moderate disparity between the two reward sets, and, most interestingly, the method flagged unsafe states that the human annotator missed.
>
---
#### [replaced 029] EquiContact: A Hierarchical SE(3) Vision-to-Force Equivariant Policy for Spatially Generalizable Contact-rich Tasks
- **分类: cs.RO**

- **简介: 该论文针对接触丰富的操作任务（如插销），提出EquiContact框架，解决空间泛化问题。通过层次化视觉-力控制策略，实现SE(3)等变性，提升任务成功率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.10961v3](https://arxiv.org/pdf/2507.10961v3)**

> **作者:** Joohwan Seo; Arvind Kruthiventy; Soomi Lee; Megan Teng; Seoyeon Choi; Xiang Zhang; Jongeun Choi; Roberto Horowitz
>
> **备注:** Submitted to RSS
>
> **摘要:** This paper presents a framework for learning vision-based robotic policies for contact-rich manipulation tasks that generalize spatially across task configurations. We focus on achieving robust spatial generalization of the policy for the peg-in-hole (PiH) task trained from a small number of demonstrations. We propose EquiContact, a hierarchical policy composed of a high-level vision planner (Diffusion Equivariant Descriptor Field, Diff-EDF) and a novel low-level compliant visuomotor policy (Geometric Compliant ACT, G-CompACT). G-CompACT operates using only localized observations (geometrically consistent error vectors (GCEV), force-torque readings, and wrist-mounted RGB images) and produces actions defined in the end-effector frame. Through these design choices, we show that the entire EquiContact pipeline is SE(3)-equivariant, from perception to force control. We also outline three key components for spatially generalizable contact-rich policies: compliance, localized policies, and induced equivariance. Real-world experiments on PiH, screwing, and surface wiping tasks demonstrate a near-perfect success rate and robust generalization to unseen spatial configurations, validating the proposed framework and principles.
>
---
#### [replaced 030] Masked Generative Policy for Robotic Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Masked Generative Policy（MGP），用于视觉-运动模仿学习，解决机器人控制中的复杂任务问题。通过生成并优化动作令牌，提升控制可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2512.09101v2](https://arxiv.org/pdf/2512.09101v2)**

> **作者:** Lipeng Zhuang; Shiyu Fan; Florent P. Audonnet; Yingdong Ru; Edmond S. L. Ho; Gerardo Aragon Camarasa; Paul Henderson
>
> **摘要:** We present Masked Generative Policy (MGP), a novel framework for visuomotor imitation learning. We represent actions as discrete tokens, and train a conditional masked transformer that generates tokens in parallel and then rapidly refines only low-confidence tokens. We further propose two new sampling paradigms: MGP-Short, which performs parallel masked generation with score-based refinement for Markovian tasks, and MGP-Long, which predicts full trajectories in a single pass and dynamically refines low-confidence action tokens based on new observations. With globally coherent prediction and robust adaptive execution capabilities, MGP-Long enables reliable control on complex and non-Markovian tasks that prior methods struggle with. Extensive evaluations on 150 robotic manipulation tasks spanning the Meta-World and LIBERO benchmarks show that MGP achieves both rapid inference and superior success rates compared to state-of-the-art diffusion and autoregressive policies. Specifically, MGP increases the average success rate by 9% across 150 tasks while cutting per-sequence inference time by up to 35x. It further improves the average success rate by 60% in dynamic and missing-observation environments, and solves two non-Markovian scenarios where other state-of-the-art methods fail.
>
---
#### [replaced 031] Goal-oriented Semantic Communication for Robot Arm Reconstruction in Digital Twin: Feature and Temporal Selections
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对数字孪生中机械臂重建任务，提出一种目标导向语义通信框架，通过特征与时间选择降低通信负载，解决高通信开销问题。**

- **链接: [https://arxiv.org/pdf/2411.08835v2](https://arxiv.org/pdf/2411.08835v2)**

> **作者:** Shutong Chen; Emmanouil Spyrakos-Papastavridis; Yichao Jin; Yansha Deng
>
> **备注:** Accepted by IEEE Journal on Selected Areas in Communications
>
> **摘要:** As one of the most promising technologies in industry, the Digital Twin (DT) facilitates real-time monitoring and predictive analysis for real-world systems by precisely reconstructing virtual replicas of physical entities. However, this reconstruction faces unprecedented challenges due to the everincreasing communication overhead, especially for digital robot arm reconstruction. To this end, we propose a novel goal-oriented semantic communication (GSC) framework to extract the GSC information for the robot arm reconstruction task in the DT, with the aim of minimising the communication load under the strict and relaxed reconstruction error constraints. Unlike the traditional reconstruction framework that periodically transmits a reconstruction message for real-time DT reconstruction, our framework implements a feature selection (FS) algorithm to extract the semantic information from the reconstruction message, and a deep reinforcement learning-based temporal selection algorithm to selectively transmit the semantic information over time. We validate our proposed GSC framework through both Pybullet simulations and lab experiments based on the Franka Research 3 robot arm. For a range of distinct robotic tasks, simulation results show that our framework can reduce the communication load by at least 59.5% under strict reconstruction error constraints and 80% under relaxed reconstruction error constraints, compared with traditional communication framework. Also, experimental results confirm the effectiveness of our framework, where the communication load is reduced by 53% in strict constraint case and 74% in relaxed constraint case. The demo is available at: https://youtu.be/2OdeHKxcgnk.
>
---
#### [replaced 032] Haptic Light-Emitting Diodes: Miniature, Luminous Tactile Actuators
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出Haptic Light-Emitting Diodes（HLEDs），一种将光脉冲直接转换为机械力的微型触觉执行器，用于人机交互中的触觉反馈。**

- **链接: [https://arxiv.org/pdf/2601.11043v2](https://arxiv.org/pdf/2601.11043v2)**

> **作者:** Max Linnander; Yon Visell
>
> **摘要:** We present Haptic Light-Emitting Diodes (HLEDs), luminous thermopneumatic actuators that directly convert pulsed light into mechanical forces and displacements. Each device packages a miniature surface-mount LED in a gas-filled cavity that contains a low-inertia graphite photoabsorber. The cavity is sealed by an elastic membrane, which functions as a working diaphragm. Brief optical pulses heat the photoabsorber, which heats the gas. The resulting rapid pressure increases generate forces and displacements at the working diaphragm. Millimeter-scale HLEDs produce forces exceeding 0.4 N and displacements of 0.9 mm at low voltages, with 5 to 100 ms response times, making them attractive as actuators providing tactile feedback in human-machine interfaces. Unusually, these actuators are also light-emitting, as a fraction of optical energy is transmitted through the membrane. These photomechanical actuators have many potential applications in tactile displays, human interface engineering, wearable computing, and other areas.
>
---
#### [replaced 033] DecompGAIL: Learning Realistic Traffic Behaviors with Decomposed Multi-Agent Generative Adversarial Imitation Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于交通行为模仿学习任务，旨在解决多智能体场景下模仿学习不稳定的问题。提出DecompGAIL模型，分解现实性并过滤干扰交互，提升交通行为真实性。**

- **链接: [https://arxiv.org/pdf/2510.06913v2](https://arxiv.org/pdf/2510.06913v2)**

> **作者:** Ke Guo; Haochen Liu; Xiaojun Wu; Chen Lv
>
> **备注:** accepted by ICLR
>
> **摘要:** Realistic traffic simulation is critical for the development of autonomous driving systems and urban mobility planning, yet existing imitation learning approaches often fail to model realistic traffic behaviors. Behavior cloning suffers from covariate shift, while Generative Adversarial Imitation Learning (GAIL) is notoriously unstable in multi-agent settings. We identify a key source of this instability: irrelevant interaction misguidance, where a discriminator penalizes an ego vehicle's realistic behavior due to unrealistic interactions among its neighbors. To address this, we propose Decomposed Multi-agent GAIL (DecompGAIL), which explicitly decomposes realism into ego-map and ego-neighbor components, filtering out misleading neighbor: neighbor and neighbor: map interactions. We further introduce a social PPO objective that augments ego rewards with distance-weighted neighborhood rewards, encouraging overall realism across agents. Integrated into a lightweight SMART-based backbone, DecompGAIL achieves state-of-the-art performance on the WOMD Sim Agents 2025 benchmark.
>
---
