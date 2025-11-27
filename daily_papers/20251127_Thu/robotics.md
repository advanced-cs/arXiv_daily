# 机器人 cs.RO

- **最新发布 28 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] OVAL-Grasp: Open-Vocabulary Affordance Localization for Task Oriented Grasping
- **分类: cs.RO**

- **简介: 该论文提出OVAL-Grasp，一种零样本开放词汇的任务导向抓取方法。针对机器人在未知环境中基于任务精准抓取物体特定部位的难题，利用大语言模型（LLM）识别操作部位，视觉语言模型（VLM）进行分割，生成可抓取区域热图，实现对未见物体和部分遮挡场景的有效抓取，实测正确抓取率达78.3%。**

- **链接: [https://arxiv.org/pdf/2511.20841v1](https://arxiv.org/pdf/2511.20841v1)**

> **作者:** Edmond Tong; Advaith Balaji; Anthony Opipari; Stanley Lewis; Zhen Zeng; Odest Chadwicke Jenkins
>
> **备注:** 10 pages, 7 figures, 3 tables. Presented at the 2025 International Symposium on Experimental Robotics (ISER)
>
> **摘要:** To manipulate objects in novel, unstructured environments, robots need task-oriented grasps that target object parts based on the given task. Geometry-based methods often struggle with visually defined parts, occlusions, and unseen objects. We introduce OVAL-Grasp, a zero-shot open-vocabulary approach to task-oriented, affordance based grasping that uses large-language models and vision-language models to allow a robot to grasp objects at the correct part according to a given task. Given an RGB image and a task, OVAL-Grasp identifies parts to grasp or avoid with an LLM, segments them with a VLM, and generates a 2D heatmap of actionable regions on the object. During our evaluations, we found that our method outperformed two task oriented grasping baselines on experiments with 20 household objects with 3 unique tasks for each. OVAL-Grasp successfully identifies and segments the correct object part 95% of the time and grasps the correct actionable area 78.3% of the time in real-world experiments with the Fetch mobile manipulator. Additionally, OVAL-Grasp finds correct object parts under partial occlusions, demonstrating a part selection success rate of 80% in cluttered scenes. We also demonstrate OVAL-Grasp's efficacy in scenarios that rely on visual features for part selection, and show the benefit of a modular design through our ablation experiments. Our project webpage is available at https://ekjt.github.io/OVAL-Grasp/
>
---
#### [new 002] Improvement of Collision Avoidance in Cut-In Maneuvers Using Time-to-Collision Metrics
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对自动驾驶中复杂的切入场景碰撞避让问题，提出一种融合深度学习与时间到碰撞（TTC）指标的新策略。通过改进TTC计算，提升碰撞预测精度并优化避让决策，显著增强系统在复杂交通场景下的安全性与响应能力。**

- **链接: [https://arxiv.org/pdf/2511.21280v1](https://arxiv.org/pdf/2511.21280v1)**

> **作者:** Jamal Raiyn
>
> **摘要:** This paper proposes a new strategy for collision avoidance system leveraging Time-to-Collision (TTC) metrics for handling cut-in scenarios, which are particularly challenging for autonomous vehicles (AVs). By integrating a deep learning with TTC calculations, the system predicts potential collisions and determines appropriate evasive actions compared to traditional TTC -based approaches.
>
---
#### [new 003] Dynamic Test-Time Compute Scaling in Control Policy: Difficulty-Aware Stochastic Interpolant Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对机器人长时程操作中生成式策略计算效率低的问题，提出DA-SIP框架。通过动态感知任务难度，自适应调整推理预算与求解器配置，实现计算资源的智能分配，在保持高成功率的同时显著降低计算开销。**

- **链接: [https://arxiv.org/pdf/2511.20906v1](https://arxiv.org/pdf/2511.20906v1)**

> **作者:** Inkook Chun; Seungjae Lee; Michael S. Albergo; Saining Xie; Eric Vanden-Eijnden
>
> **摘要:** Diffusion- and flow-based policies deliver state-of-the-art performance on long-horizon robotic manipulation and imitation learning tasks. However, these controllers employ a fixed inference budget at every control step, regardless of task complexity, leading to computational inefficiency for simple subtasks while potentially underperforming on challenging ones. To address these issues, we introduce Difficulty-Aware Stochastic Interpolant Policy (DA-SIP), a framework that enables robotic controllers to adaptively adjust their integration horizon in real time based on task difficulty. Our approach employs a difficulty classifier that analyzes observations to dynamically select the step budget, the optimal solver variant, and ODE/SDE integration at each control cycle. DA-SIP builds upon the stochastic interpolant formulation to provide a unified framework that unlocks diverse training and inference configurations for diffusion- and flow-based policies. Through comprehensive benchmarks across diverse manipulation tasks, DA-SIP achieves 2.6-4.4x reduction in total computation time while maintaining task success rates comparable to fixed maximum-computation baselines. By implementing adaptive computation within this framework, DA-SIP transforms generative robot controllers into efficient, task-aware systems that intelligently allocate inference resources where they provide the greatest benefit.
>
---
#### [new 004] Sampling-Based Optimization with Parallelized Physics Simulator for Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文针对双臂机器人在杂乱环境中的操作难题，提出基于采样的优化框架，利用GPU加速的物理引擎实现高效仿真。通过设计任务相关的代价函数，改进MPPI算法，解决复杂点对点搬运任务，实现实时性能与良好的仿真实现迁移。**

- **链接: [https://arxiv.org/pdf/2511.21264v1](https://arxiv.org/pdf/2511.21264v1)**

> **作者:** Iryna Hurova; Alinjar Dan; Karl Kruusamäe; Arun Kumar Singh
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** In recent years, dual-arm manipulation has become an area of strong interest in robotics, with end-to-end learning emerging as the predominant strategy for solving bimanual tasks. A critical limitation of such learning-based approaches, however, is their difficulty in generalizing to novel scenarios, especially within cluttered environments. This paper presents an alternative paradigm: a sampling-based optimization framework that utilizes a GPU-accelerated physics simulator as its world model. We demonstrate that this approach can solve complex bimanual manipulation tasks in the presence of static obstacles. Our contribution is a customized Model Predictive Path Integral Control (MPPI) algorithm, \textbf{guided by carefully designed task-specific cost functions,} that uses GPU-accelerated MuJoCo for efficiently evaluating robot-object interaction. We apply this method to solve significantly more challenging versions of tasks from the PerAct$^{2}$ benchmark, such as requiring the point-to-point transfer of a ball through an obstacle course. Furthermore, we establish that our method achieves real-time performance on commodity GPUs and facilitates successful sim-to-real transfer by leveraging unique features within MuJoCo. The paper concludes with a statistical analysis of the sample complexity and robustness, quantifying the performance of our approach. The project website is available at: https://sites.google.com/view/bimanualakslabunitartu .
>
---
#### [new 005] Efficient Greedy Algorithms for Feature Selection in Robot Visual Localization
- **分类: cs.RO**

- **简介: 该论文针对机器人视觉定位中的特征冗余问题，提出两种高效贪心特征选择算法。旨在减少计算开销，实时筛选最具信息量的视觉特征，提升定位效率与精度，属于机器人自主导航中的高效特征选择任务。**

- **链接: [https://arxiv.org/pdf/2511.20894v1](https://arxiv.org/pdf/2511.20894v1)**

> **作者:** Vivek Pandey; Amirhossein Mollaei; Nader Motee
>
> **摘要:** Robot localization is a fundamental component of autonomous navigation in unknown environments. Among various sensing modalities, visual input from cameras plays a central role, enabling robots to estimate their position by tracking point features across image frames. However, image frames often contain a large number of features, many of which are redundant or uninformative for localization. Processing all features can introduce significant computational latency and inefficiency. This motivates the need for intelligent feature selection, identifying a subset of features that are most informative for localization over a prediction horizon. In this work, we propose two fast and memory-efficient feature selection algorithms that enable robots to actively evaluate the utility of visual features in real time. Unlike existing approaches with high computational and memory demands, the proposed methods are explicitly designed to reduce both time and memory complexity while achieving a favorable trade-off between computational efficiency and localization accuracy.
>
---
#### [new 006] MarketGen: A Scalable Simulation Platform with Auto-Generated Embodied Supermarket Environments
- **分类: cs.RO**

- **简介: 该论文针对复杂商业场景中机器人代理训练数据匮乏的问题，提出MarketGen平台。通过多模态输入驱动的程序化生成框架，自动生成真实超市环境与任务，构建了包含长时序结算和移动操作任务的基准。实现了可扩展仿真与真实世界迁移，推动了具身智能在商业应用中的研究。**

- **链接: [https://arxiv.org/pdf/2511.21161v1](https://arxiv.org/pdf/2511.21161v1)**

> **作者:** Xu Hu; Yiyang Feng; Junran Peng; Jiawei He; Liyi Chen; Chuanchen Luo; Xucheng Yin; Qing Li; Zhaoxiang Zhang
>
> **备注:** Project Page: https://xuhu0529.github.io/MarketGen
>
> **摘要:** The development of embodied agents for complex commercial environments is hindered by a critical gap in existing robotics datasets and benchmarks, which primarily focus on household or tabletop settings with short-horizon tasks. To address this limitation, we introduce MarketGen, a scalable simulation platform with automatic scene generation for complex supermarket environments. MarketGen features a novel agent-based Procedural Content Generation (PCG) framework. It uniquely supports multi-modal inputs (text and reference images) and integrates real-world design principles to automatically generate complete, structured, and realistic supermarkets. We also provide an extensive and diverse 3D asset library with a total of 1100+ supermarket goods and parameterized facilities assets. Building on this generative foundation, we propose a novel benchmark for assessing supermarket agents, featuring two daily tasks in a supermarket: (1) Checkout Unloading: long-horizon tabletop tasks for cashier agents, and (2) In-Aisle Item Collection: complex mobile manipulation tasks for salesperson agents. We validate our platform and benchmark through extensive experiments, including the deployment of a modular agent system and successful sim-to-real transfer. MarketGen provides a comprehensive framework to accelerate research in embodied AI for complex commercial applications.
>
---
#### [new 007] Model-Based Policy Adaptation for Closed-Loop End-to-End Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对端到端自动驾驶在闭环场景中易出现累积误差、泛化性差的问题，提出基于模型的策略自适应框架MPA。通过几何一致的仿真生成反事实轨迹，训练扩散型策略适配器与多步Q值模型，在推理时生成并优选安全高效路径，显著提升模型在多种场景下的鲁棒性与安全性。**

- **链接: [https://arxiv.org/pdf/2511.21584v1](https://arxiv.org/pdf/2511.21584v1)**

> **作者:** Haohong Lin; Yunzhi Zhang; Wenhao Ding; Jiajun Wu; Ding Zhao
>
> **备注:** Published at NeurIPS 2025: https://openreview.net/forum?id=4OLbpaTKJe
>
> **摘要:** End-to-end (E2E) autonomous driving models have demonstrated strong performance in open-loop evaluations but often suffer from cascading errors and poor generalization in closed-loop settings. To address this gap, we propose Model-based Policy Adaptation (MPA), a general framework that enhances the robustness and safety of pretrained E2E driving agents during deployment. MPA first generates diverse counterfactual trajectories using a geometry-consistent simulation engine, exposing the agent to scenarios beyond the original dataset. Based on this generated data, MPA trains a diffusion-based policy adapter to refine the base policy's predictions and a multi-step Q value model to evaluate long-term outcomes. At inference time, the adapter proposes multiple trajectory candidates, and the Q value model selects the one with the highest expected utility. Experiments on the nuScenes benchmark using a photorealistic closed-loop simulator demonstrate that MPA significantly improves performance across in-domain, out-of-domain, and safety-critical scenarios. We further investigate how the scale of counterfactual data and inference-time guidance strategies affect overall effectiveness.
>
---
#### [new 008] TraceGen: World Modeling in 3D Trace Space Enables Learning from Cross-Embodiment Videos
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文针对机器人在新平台、新场景下仅凭少量示范学习的难题，提出TraceGen世界模型。通过构建统一的3D轨迹空间表示，实现跨主体、跨环境视频的高效利用。基于123K视频数据预训练，仅需5段目标视频即可达80%成功率，显著提升泛化与推理效率。**

- **链接: [https://arxiv.org/pdf/2511.21690v1](https://arxiv.org/pdf/2511.21690v1)**

> **作者:** Seungjae Lee; Yoonkyo Jung; Inkook Chun; Yao-Chih Lee; Zikui Cai; Hongjia Huang; Aayush Talreja; Tan Dat Dao; Yongyuan Liang; Jia-Bin Huang; Furong Huang
>
> **摘要:** Learning new robot tasks on new platforms and in new scenes from only a handful of demonstrations remains challenging. While videos of other embodiments - humans and different robots - are abundant, differences in embodiment, camera, and environment hinder their direct use. We address the small-data problem by introducing a unifying, symbolic representation - a compact 3D "trace-space" of scene-level trajectories - that enables learning from cross-embodiment, cross-environment, and cross-task videos. We present TraceGen, a world model that predicts future motion in trace-space rather than pixel space, abstracting away appearance while retaining the geometric structure needed for manipulation. To train TraceGen at scale, we develop TraceForge, a data pipeline that transforms heterogeneous human and robot videos into consistent 3D traces, yielding a corpus of 123K videos and 1.8M observation-trace-language triplets. Pretraining on this corpus produces a transferable 3D motion prior that adapts efficiently: with just five target robot videos, TraceGen attains 80% success across four tasks while offering 50-600x faster inference than state-of-the-art video-based world models. In the more challenging case where only five uncalibrated human demonstration videos captured on a handheld phone are available, it still reaches 67.5% success on a real robot, highlighting TraceGen's ability to adapt across embodiments without relying on object detectors or heavy pixel-space generation.
>
---
#### [new 009] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AerialMind，首个面向无人机场景的指代多目标跟踪（RMOT）基准。针对现有研究局限于地面场景、难以捕捉大范围上下文的问题，构建了大规模数据集并提出COALA标注框架降低人力成本。同时设计HETrack方法，提升视觉-语言联合表征能力，增强无人机环境下的目标感知与跟踪性能。**

- **链接: [https://arxiv.org/pdf/2511.21053v1](https://arxiv.org/pdf/2511.21053v1)**

> **作者:** Chenglizhao Chen; Shaofeng Liang; Runwei Guan; Xiaolou Sun; Haocheng Zhao; Haiyun Jiang; Tao Huang; Henghui Ding; Qing-Long Han
>
> **备注:** AAAI 2026
>
> **摘要:** Referring Multi-Object Tracking (RMOT) aims to achieve precise object detection and tracking through natural language instructions, representing a fundamental capability for intelligent robotic systems. However, current RMOT research remains mostly confined to ground-level scenarios, which constrains their ability to capture broad-scale scene contexts and perform comprehensive tracking and path planning. In contrast, Unmanned Aerial Vehicles (UAVs) leverage their expansive aerial perspectives and superior maneuverability to enable wide-area surveillance. Moreover, UAVs have emerged as critical platforms for Embodied Intelligence, which has given rise to an unprecedented demand for intelligent aerial systems capable of natural language interaction. To this end, we introduce AerialMind, the first large-scale RMOT benchmark in UAV scenarios, which aims to bridge this research gap. To facilitate its construction, we develop an innovative semi-automated collaborative agent-based labeling assistant (COALA) framework that significantly reduces labor costs while maintaining annotation quality. Furthermore, we propose HawkEyeTrack (HETrack), a novel method that collaboratively enhances vision-language representation learning and improves the perception of UAV scenarios. Comprehensive experiments validated the challenging nature of our dataset and the effectiveness of our method.
>
---
#### [new 010] Maglev-Pentabot: Magnetic Levitation System for Non-Contact Manipulation using Deep Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Maglev-Pentabot系统，利用深度强化学习实现克级物体的非接触操纵。针对现有技术仅限微米级、控制复杂的问题，通过优化电磁铁布局与动作重映射方法，提升可控空间并解决样本稀疏难题，实现实验室尺度灵活操控及任务泛化，具备工业级扩展潜力。**

- **链接: [https://arxiv.org/pdf/2511.21149v1](https://arxiv.org/pdf/2511.21149v1)**

> **作者:** Guoming Huang; Qingyi Zhou; Dianjing Liu; Shuai Zhang; Ming Zhou; Zongfu Yu
>
> **摘要:** Non-contact manipulation has emerged as a transformative approach across various industrial fields. However, current flexible 2D and 3D non-contact manipulation techniques are often limited to microscopic scales, typically controlling objects in the milligram range. In this paper, we present a magnetic levitation system, termed Maglev-Pentabot, designed to address this limitation. The Maglev-Pentabot leverages deep reinforcement learning (DRL) to develop complex control strategies for manipulating objects in the gram range. Specifically, we propose an electromagnet arrangement optimized through numerical analysis to maximize controllable space. Additionally, an action remapping method is introduced to address sample sparsity issues caused by the strong nonlinearity in magnetic field intensity, hence allowing the DRL controller to converge. Experimental results demonstrate flexible manipulation capabilities, and notably, our system can generalize to transport tasks it has not been explicitly trained for. Furthermore, our approach can be scaled to manipulate heavier objects using larger electromagnets, offering a reference framework for industrial-scale robotic applications.
>
---
#### [new 011] Uncertainty Quantification for Visual Object Pose Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对单目视觉目标位姿估计中的不确定性量化问题，提出无需分布假设的分布自由不确定性边界方法。通过像素检测噪声约束，构建非凸不确定性集，并基于S-lemma设计凸优化算法SLUE，得到高概率包含真实位姿的椭球边界。可有效分解为平移与姿态独立边界，实验验证其在翻译不确定性上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.21666v1](https://arxiv.org/pdf/2511.21666v1)**

> **作者:** Lorenzo Shaikewitz; Charis Georgiou; Luca Carlone
>
> **备注:** 18 pages, 9 figures. Code available: https://github.com/MIT-SPARK/PoseUncertaintySets
>
> **摘要:** Quantifying the uncertainty of an object's pose estimate is essential for robust control and planning. Although pose estimation is a well-studied robotics problem, attaching statistically rigorous uncertainty is not well understood without strict distributional assumptions. We develop distribution-free pose uncertainty bounds about a given pose estimate in the monocular setting. Our pose uncertainty only requires high probability noise bounds on pixel detections of 2D semantic keypoints on a known object. This noise model induces an implicit, non-convex set of pose uncertainty constraints. Our key contribution is SLUE (S-Lemma Uncertainty Estimation), a convex program to reduce this set to a single ellipsoidal uncertainty bound that is guaranteed to contain the true object pose with high probability. SLUE solves a relaxation of the minimum volume bounding ellipsoid problem inspired by the celebrated S-lemma. It requires no initial guess of the bound's shape or size and is guaranteed to contain the true object pose with high probability. For tighter uncertainty bounds at the same confidence, we extend SLUE to a sum-of-squares relaxation hierarchy which is guaranteed to converge to the minimum volume ellipsoidal uncertainty bound for a given set of keypoint constraints. We show this pose uncertainty bound can easily be projected to independent translation and axis-angle orientation bounds. We evaluate SLUE on two pose estimation datasets and a real-world drone tracking scenario. Compared to prior work, SLUE generates substantially smaller translation bounds and competitive orientation bounds. We release code at https://github.com/MIT-SPARK/PoseUncertaintySets.
>
---
#### [new 012] SocialNav: Training Human-Inspired Foundation Model for Socially-Aware Embodied Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对遵循社会规范的具身导航任务，提出SocialNav基础模型。通过构建大规模数据集与分阶段训练，融合社会推理与导航技能，实现高成功率与高社会合规性。**

- **链接: [https://arxiv.org/pdf/2511.21135v1](https://arxiv.org/pdf/2511.21135v1)**

> **作者:** Ziyi Chen; Yingnan Guo; Zedong Chu; Minghua Luo; Yanfen Shen; Mingchao Sun; Junjun Hu; Shichao Xie; Kuan Yang; Pei Shi; Zhining Gu; Lu Liu; Honglin Han; Xiaolong Wu; Mu Xu; Yu Zhang
>
> **摘要:** Embodied navigation that adheres to social norms remains an open research challenge. Our \textbf{SocialNav} is a foundational model for socially-aware navigation with a hierarchical "brain-action" architecture, capable of understanding high-level social norms and generating low-level, socially compliant trajectories. To enable such dual capabilities, we construct the SocNav Dataset, a large-scale collection of 7 million samples, comprising (1) a Cognitive Activation Dataset providing social reasoning signals such as chain-of-thought explanations and social traversability prediction, and (2) an Expert Trajectories Pyramid aggregating diverse navigation demonstrations from internet videos, simulated environments, and real-world robots. A multi-stage training pipeline is proposed to gradually inject and refine navigation intelligence: we first inject general navigation skills and social norms understanding into the model via imitation learning, and then refine such skills through a deliberately designed Socially-Aware Flow Exploration GRPO (SAFE-GRPO), the first flow-based reinforcement learning framework for embodied navigation that explicitly rewards socially compliant behaviors. SocialNav achieves +38% success rate and +46% social compliance rate compared to the state-of-the-art method, demonstrating strong gains in both navigation performance and social compliance. Our project page: https://amap-eai.github.io/SocialNav/
>
---
#### [new 013] ACE-F: A Cross Embodiment Foldable System with Force Feedback for Dexterous Teleoperation
- **分类: cs.RO**

- **简介: 该论文提出ACE-F系统，解决机器人遥操作中缺乏力反馈、跨形态泛化及便携性差的问题。通过集成力反馈与可折叠设计，结合逆运动学与软控制器，实现多形态机器人精准、安全的灵巧操作，支持无需额外传感器的跨形态力反馈数据采集，显著提升操作直观性与数据质量。**

- **链接: [https://arxiv.org/pdf/2511.20887v1](https://arxiv.org/pdf/2511.20887v1)**

> **作者:** Rui Yan; Jiajian Fu; Shiqi Yang; Lars Paulsen; Xuxin Cheng; Xiaolong Wang
>
> **摘要:** Teleoperation systems are essential for efficiently collecting diverse and high-quality robot demonstration data, especially for complex, contact-rich tasks. However, current teleoperation platforms typically lack integrated force feedback, cross-embodiment generalization, and portable, user-friendly designs, limiting their practical deployment. To address these limitations, we introduce ACE-F, a cross embodiment foldable teleoperation system with integrated force feedback. Our approach leverages inverse kinematics (IK) combined with a carefully designed human-robot interface (HRI), enabling users to capture precise and high-quality demonstrations effortlessly. We further propose a generalized soft-controller pipeline integrating PD control and inverse dynamics to ensure robot safety and precise motion control across diverse robotic embodiments. Critically, to achieve cross-embodiment generalization of force feedback without additional sensors, we innovatively interpret end-effector positional deviations as virtual force signals, which enhance data collection and enable applications in imitation learning. Extensive teleoperation experiments confirm that ACE-F significantly simplifies the control of various robot embodiments, making dexterous manipulation tasks as intuitive as operating a computer mouse. The system is open-sourced at: https://acefoldable.github.io/
>
---
#### [new 014] Dual-Agent Reinforcement Learning for Adaptive and Cost-Aware Visual-Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文针对视觉惯性里程计（VIO）中精度与计算效率的矛盾，提出双代理强化学习框架。通过轻量RL代理动态控制前端运行时机与状态融合强度，减少高耗时的视觉惯性捆绑调整调用，实现更高效率与更优内存占用下的精准位姿估计。**

- **链接: [https://arxiv.org/pdf/2511.21083v1](https://arxiv.org/pdf/2511.21083v1)**

> **作者:** Feiyang Pan; Shenghe Zheng; Chunyan Yin; Guangbin Dou
>
> **摘要:** Visual-Inertial Odometry (VIO) is a critical component for robust ego-motion estimation, enabling foundational capabilities such as autonomous navigation in robotics and real-time 6-DoF tracking for augmented reality. Existing methods face a well-known trade-off: filter-based approaches are efficient but prone to drift, while optimization-based methods, though accurate, rely on computationally prohibitive Visual-Inertial Bundle Adjustment (VIBA) that is difficult to run on resource-constrained platforms. Rather than removing VIBA altogether, we aim to reduce how often and how heavily it must be invoked. To this end, we cast two key design choices in modern VIO, when to run the visual frontend and how strongly to trust its output, as sequential decision problems, and solve them with lightweight reinforcement learning (RL) agents. Our framework introduces a lightweight, dual-pronged RL policy that serves as our core contribution: (1) a Select Agent intelligently gates the entire VO pipeline based only on high-frequency IMU data; and (2) a composite Fusion Agent that first estimates a robust velocity state via a supervised network, before an RL policy adaptively fuses the full (p, v, q) state. Experiments on the EuRoC MAV and TUM-VI datasets show that, in our unified evaluation, the proposed method achieves a more favorable accuracy-efficiency-memory trade-off than prior GPU-based VO/VIO systems: it attains the best average ATE while running up to 1.77 times faster and using less GPU memory. Compared to classical optimization-based VIO systems, our approach maintains competitive trajectory accuracy while substantially reducing computational load.
>
---
#### [new 015] Hybrid Control for Robotic Nut Tightening Task
- **分类: cs.RO**

- **简介: 该论文针对机器人拧螺母任务，提出一种基于分层运动基元与力/位置控制切换的混合控制方法。解决了初始条件变化下的鲁棒性及接触力过大的问题。通过仿真验证，系统比基准快14%，接触力降低40倍，代码已开源。**

- **链接: [https://arxiv.org/pdf/2511.21366v1](https://arxiv.org/pdf/2511.21366v1)**

> **作者:** Dmitri Kovalenko
>
> **摘要:** An autonomous robotic nut tightening system for a serial manipulator equipped with a parallel gripper is proposed. The system features a hierarchical motion-primitive-based planner and a control-switching scheme that alternates between force and position control. Extensive simulations demonstrate the system's robustness to variance in initial conditions. Additionally, the proposed controller tightens threaded screws 14% faster than the baseline while applying 40 times less contact force on manipulands. For the benefit of the research community, the system's implementation is open-sourced.
>
---
#### [new 016] VacuumVLA: Boosting VLA Capabilities via a Unified Suction and Gripping Tool for Complex Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对视觉语言动作（VLA）机器人在复杂操作中因传统夹爪接触面积小、缺乏吸附力而无法完成如擦玻璃、开无手柄抽屉等问题，提出一种集成真空吸盘与双指夹爪的混合末端执行器。通过软硬件协同设计，实现两种模式自由切换或协同工作，显著拓展任务适应性，并在DexVLA和Pi0框架中验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2511.21557v1](https://arxiv.org/pdf/2511.21557v1)**

> **作者:** Hui Zhou; Siyuan Huang; Minxing Li; Hao Zhang; Lue Fan; Shaoshuai Shi
>
> **备注:** 8 pages
>
> **摘要:** Vision Language Action models have significantly advanced general purpose robotic manipulation by harnessing large scale pretrained vision and language representations. Among existing approaches, a majority of current VLA systems employ parallel two finger grippers as their default end effectors. However, such grippers face inherent limitations in handling certain real world tasks such as wiping glass surfaces or opening drawers without handles due to insufficient contact area or lack of adhesion. To overcome these challenges, we present a low cost, integrated hardware design that combines a mechanical two finger gripper with a vacuum suction unit, enabling dual mode manipulation within a single end effector. Our system supports flexible switching or synergistic use of both modalities, expanding the range of feasible tasks. We validate the efficiency and practicality of our design within two state of the art VLA frameworks: DexVLA and Pi0. Experimental results demonstrate that with the proposed hybrid end effector, robots can successfully perform multiple complex tasks that are infeasible for conventional two finger grippers alone. All hardware designs and controlling systems will be released.
>
---
#### [new 017] Kinematics-Aware Multi-Policy Reinforcement Learning for Force-Capable Humanoid Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文针对高负载工业场景下人形机器人协同操作与主动力交互需求，提出一种分阶段的多策略强化学习框架。通过解耦上肢、下肢与增量指令策略，结合启发式奖励与基于力的课程学习，提升机器人在复杂环境中的灵巧操作与主动受力能力。**

- **链接: [https://arxiv.org/pdf/2511.21169v1](https://arxiv.org/pdf/2511.21169v1)**

> **作者:** Kaiyan Xiao; Zihan Xu; Cheng Zhe; Chengju Liu; Qijun Chen
>
> **摘要:** Humanoid robots, with their human-like morphology, hold great potential for industrial applications. However, existing loco-manipulation methods primarily focus on dexterous manipulation, falling short of the combined requirements for dexterity and proactive force interaction in high-load industrial scenarios. To bridge this gap, we propose a reinforcement learning-based framework with a decoupled three-stage training pipeline, consisting of an upper-body policy, a lower-body policy, and a delta-command policy. To accelerate upper-body training, a heuristic reward function is designed. By implicitly embedding forward kinematics priors, it enables the policy to converge faster and achieve superior performance. For the lower body, a force-based curriculum learning strategy is developed, enabling the robot to actively exert and regulate interaction forces with the environment.
>
---
#### [new 018] $\mathcal{E}_0$: Enhancing Generalization and Fine-Grained Control in VLA Models via Continuized Discrete Diffusion
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在机器人操作中泛化能力差、动作粗略不稳的问题，提出E0框架。通过连续化离散扩散机制，实现细粒度动作生成与更强语义控制，提升泛化性与鲁棒性，在多个基准上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2511.21542v1](https://arxiv.org/pdf/2511.21542v1)**

> **作者:** Zhihao Zhan; Jiaying Zhou; Likui Zhang; Qinhan Lv; Hao Liu; Jusheng Zhang; Weizheng Li; Ziliang Chen; Tianshui Chen; Keze Wang; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-Language-Action (VLA) models offer a unified framework for robotic manipulation by integrating visual perception, language understanding, and control generation. Yet existing VLA models still struggle to generalize across diverse tasks, scenes, and camera viewpoints, and often produce coarse or unstable actions. We introduce E0, a continuized discrete diffusion framework that formulates action generation as iterative denoising over quantized action tokens. Compared with continuous diffusion policies, E0 offers two key advantages: (1) discrete action tokens align naturally with the symbolic structure of pretrained VLM/VLA backbones, enabling stronger semantic conditioning; and 2. discrete diffusion matches the true quantized nature of real-world robot control-whose hardware constraints (e.g., encoder resolution, control frequency, actuation latency) inherently discretize continuous signals-and therefore benefits from a Bayes-optimal denoiser that models the correct discrete action distribution, leading to stronger generalization. Compared with discrete autoregressive and mask-based discrete diffusion models, E0 supports a significantly larger and finer-grained action vocabulary and avoids the distributional mismatch introduced by masking-based corruptions-yielding more accurate fine-grained action control. We further introduce a spherical viewpoint perturbation augmentation method to improve robustness to camera shifts without additional data. Experiments on LIBERO, VLABench, and ManiSkill show that E0 achieves state-of-the-art performance across 14 diverse environments, outperforming strong baselines by 10.7% on average. Real-world evaluation on a Franka arm confirms that E0 delivers precise, robust, and transferable manipulation, establishing discrete diffusion as a promising direction for generalizable VLA policy learning.
>
---
#### [new 019] Neural NMPC through Signed Distance Field Encoding for Collision Avoidance
- **分类: cs.RO**

- **简介: 该论文针对未知环境中飞行机器人自主导航的碰撞避免问题，提出一种基于深度神经网络编码有符号距离场（SDF）的非线性模型预测控制（NMPC）框架。通过双级网络将单张测距图像映射为SDF，嵌入NMPC实现无地图、实时避障。验证了控制器的稳定性与鲁棒性，并在仿真与林区实测中证明其有效性。**

- **链接: [https://arxiv.org/pdf/2511.21312v1](https://arxiv.org/pdf/2511.21312v1)**

> **作者:** Martin Jacquet; Marvin Harms; Kostas Alexis
>
> **备注:** accepted for publication in IJRR
>
> **摘要:** This paper introduces a neural Nonlinear Model Predictive Control (NMPC) framework for mapless, collision-free navigation in unknown environments with Aerial Robots, using onboard range sensing. We leverage deep neural networks to encode a single range image, capturing all the available information about the environment, into a Signed Distance Function (SDF). The proposed neural architecture consists of two cascaded networks: a convolutional encoder that compresses the input image into a low-dimensional latent vector, and a Multi-Layer Perceptron that approximates the corresponding spatial SDF. This latter network parametrizes an explicit position constraint used for collision avoidance, which is embedded in a velocity-tracking NMPC that outputs thrust and attitude commands to the robot. First, a theoretical analysis of the contributed NMPC is conducted, verifying recursive feasibility and stability properties under fixed observations. Subsequently, we evaluate the open-loop performance of the learning-based components as well as the closed-loop performance of the controller in simulations and experiments. The simulation study includes an ablation study, comparisons with two state-of-the-art local navigation methods, and an assessment of the resilience to drifting odometry. The real-world experiments are conducted in forest environments, demonstrating that the neural NMPC effectively performs collision avoidance in cluttered settings against an adversarial reference velocity input and drifting position estimates.
>
---
#### [new 020] Transformer Driven Visual Servoing and Dual Arm Impedance Control for Fabric Texture Matching
- **分类: cs.RO**

- **简介: 该论文针对布料纹理匹配任务，提出基于Transformer的视觉伺服与双臂阻抗控制融合方法，实现布料精准对齐与放置。通过引入差分提取注意力模块，提升姿态差异预测精度，利用合成数据训练实现零样本部署，有效解决真实场景中不同纹理布料的自动对齐难题。**

- **链接: [https://arxiv.org/pdf/2511.21203v1](https://arxiv.org/pdf/2511.21203v1)**

> **作者:** Fuyuki Tokuda; Akira Seino; Akinari Kobayashi; Kai Tang; Kazuhiro Kosuge
>
> **备注:** 8 pages, 11 figures. Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** In this paper, we propose a method to align and place a fabric piece on top of another using a dual-arm manipulator and a grayscale camera, so that their surface textures are accurately matched. We propose a novel control scheme that combines Transformer-driven visual servoing with dualarm impedance control. This approach enables the system to simultaneously control the pose of the fabric piece and place it onto the underlying one while applying tension to keep the fabric piece flat. Our transformer-based network incorporates pretrained backbones and a newly introduced Difference Extraction Attention Module (DEAM), which significantly enhances pose difference prediction accuracy. Trained entirely on synthetic images generated using rendering software, the network enables zero-shot deployment in real-world scenarios without requiring prior training on specific fabric textures. Real-world experiments demonstrate that the proposed system accurately aligns fabric pieces with different textures.
>
---
#### [new 021] Dual Preintegration for Relative State Estimation
- **分类: cs.RO**

- **简介: 该论文针对六自由度相对状态估计中因非线性旋转和大距离导致的精度下降问题，提出双预积分方法。通过融合双平台的IMU预积分，构建高效重线性化约束，提升估计精度与鲁棒性。在虚拟现实控制器追踪任务中验证了其优越性。**

- **链接: [https://arxiv.org/pdf/2511.21189v1](https://arxiv.org/pdf/2511.21189v1)**

> **作者:** Ruican Xia; Hailong Pei
>
> **摘要:** Relative State Estimation perform mutually localization between two mobile agents undergoing six-degree-of-freedom motion. Based on the principle of circular motion, the estimation accuracy is sensitive to nonlinear rotations of the reference platform, particularly under large inter-platform distances. This phenomenon is even obvious for linearized kinematics, because cumulative linearization errors significantly degrade precision. In virtual reality (VR) applications, this manifests as substantial positional errors in 6-DoF controller tracking during rapid rotations of the head-mounted display. The linearization errors introduce drift in the estimate and render the estimator inconsistent. In the field of odometry, IMU preintegration is proposed as a kinematic observation to enable efficient relinearization, thus mitigate linearized error. Building on this theory, we propose dual preintegration, a novel observation integrating IMU preintegration from both platforms. This method serves as kinematic constraints for consecutive relative state and supports efficient relinearization. We also perform observability analysis of the state and analytically formulate the accordingly null space. Algorithm evaluation encompasses both simulations and real-world experiments. Multiple nonlinear rotations on the reference platform are simulated to compare the precision of the proposed method with that of other state-of-the-art (SOTA) algorithms. The field test compares the proposed method and SOTA algorithms in the application of VR controller tracking from the perspectives of bias observability, nonlinear rotation, and background texture. The results demonstrate that the proposed method is more precise and robust than the SOTA algorithms.
>
---
#### [new 022] NOIR 2.0: Neural Signal Operated Intelligent Robots for Everyday Activities
- **分类: cs.RO; cs.AI; cs.HC; cs.LG; eess.SY**

- **简介: 该论文提出NOIR 2.0，一种基于脑电的智能机器人控制系统，旨在通过脑信号实现对机器人日常任务的直接控制。针对传统脑机接口响应慢、适应性差的问题，研究改进了脑解码算法与少样本学习机制，显著提升控制速度与个性化适应能力，大幅降低用户操作时间。**

- **链接: [https://arxiv.org/pdf/2511.20848v1](https://arxiv.org/pdf/2511.20848v1)**

> **作者:** Tasha Kim; Yingke Wang; Hanvit Cho; Alex Hodges
>
> **备注:** Conference on Robot Learning (CoRL 2024), CoRoboLearn
>
> **摘要:** Neural Signal Operated Intelligent Robots (NOIR) system is a versatile brain-robot interface that allows humans to control robots for daily tasks using their brain signals. This interface utilizes electroencephalography (EEG) to translate human intentions regarding specific objects and desired actions directly into commands that robots can execute. We present NOIR 2.0, an enhanced version of NOIR. NOIR 2.0 includes faster and more accurate brain decoding algorithms, which reduce task completion time by 46%. NOIR 2.0 uses few-shot robot learning algorithms to adapt to individual users and predict their intentions. The new learning algorithms leverage foundation models for more sample-efficient learning and adaptation (15 demos vs. a single demo), significantly reducing overall human time by 65%.
>
---
#### [new 023] Dataset Poisoning Attacks on Behavioral Cloning Policies
- **分类: cs.LG; cs.CR; cs.RO**

- **简介: 该论文研究行为克隆（BC）政策在数据投毒攻击下的脆弱性，聚焦于干净标签后门攻击。通过注入视觉触发器，建立虚假关联，发现即使少量污染数据也导致模型看似正常但易受攻击。提出基于熵的测试时触发攻击，揭示其严重安全隐患，强调提升BC政策鲁棒性的紧迫性。**

- **链接: [https://arxiv.org/pdf/2511.20992v1](https://arxiv.org/pdf/2511.20992v1)**

> **作者:** Akansha Kalra; Soumil Datta; Ethan Gilmore; Duc La; Guanhong Tao; Daniel S. Brown
>
> **备注:** Accepted at EAI SmartSP 2025
>
> **摘要:** Behavior Cloning (BC) is a popular framework for training sequential decision policies from expert demonstrations via supervised learning. As these policies are increasingly being deployed in the real world, their robustness and potential vulnerabilities are an important concern. In this work, we perform the first analysis of the efficacy of clean-label backdoor attacks on BC policies. Our backdoor attacks poison a dataset of demonstrations by injecting a visual trigger to create a spurious correlation that can be exploited at test time. We evaluate how policy vulnerability scales with the fraction of poisoned data, the strength of the trigger, and the trigger type. We also introduce a novel entropy-based test-time trigger attack that substantially degrades policy performance by identifying critical states where test-time triggering of the backdoor is expected to be most effective at degrading performance. We empirically demonstrate that BC policies trained on even minimally poisoned datasets exhibit deceptively high, near-baseline task performance despite being highly vulnerable to backdoor trigger attacks during deployment. Our results underscore the urgent need for more research into the robustness of BC policies, particularly as large-scale datasets are increasingly used to train policies for real-world cyber-physical systems. Videos and code are available at https://sites.google.com/view/dataset-poisoning-in-bc.
>
---
#### [new 024] DeeAD: Dynamic Early Exit of Vision-Language Action for Efficient Autonomous Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文针对自动驾驶中视觉-语言-动作（VLA）模型推理延迟高的问题，提出无需训练的DeeAD框架。通过评估中间轨迹的物理可行性，动态提前退出计算，并结合多跳控制器跳过冗余层，实现28%层稀疏度与29%延迟降低，保持规划质量与安全。**

- **链接: [https://arxiv.org/pdf/2511.20720v1](https://arxiv.org/pdf/2511.20720v1)**

> **作者:** Haibo HU; Lianming Huang; Nan Guan; Chun Jason Xue
>
> **摘要:** Vision-Language Action (VLA) models unify perception, reasoning, and trajectory generation for autonomous driving, but suffer from significant inference latency due to deep transformer stacks. We present DeeAD, a training-free, action-guided early-exit framework that accelerates VLA planning by evaluating the physical feasibility of intermediate trajectories. Instead of relying on confidence scores, DeeAD terminates inference when predicted trajectories align with lightweight planning priors (e.g., Navigation or Low-precision Planning) within a tolerable deviation (<2m). To improve efficiency, we introduce a multi-hop controller that adaptively skips redundant layers based on the change rate of scores. DeeAD integrates into existing VLA models, such as ORION, without requiring retraining. Experiments on the Bench2Drive benchmark demonstrate up to 28% transformer-layer sparsity and 29% latency reduction, while preserving planning quality and safety.
>
---
#### [new 025] Predictive Safety Shield for Dyna-Q Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO; eess.SY**

- **简介: 该论文针对强化学习的安全性问题，提出一种预测性安全盾，用于模型基于的Dyna-Q算法。通过安全仿真预测未来状态，动态更新Q值，在保证硬性安全约束的同时提升性能，且对环境分布偏移具有鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.21531v1](https://arxiv.org/pdf/2511.21531v1)**

> **作者:** Jin Pin; Krasowski Hanna; Vanneaux Elena
>
> **摘要:** Obtaining safety guarantees for reinforcement learning is a major challenge to achieve applicability for real-world tasks. Safety shields extend standard reinforcement learning and achieve hard safety guarantees. However, existing safety shields commonly use random sampling of safe actions or a fixed fallback controller, therefore disregarding future performance implications of different safe actions. In this work, we propose a predictive safety shield for model-based reinforcement learning agents in discrete space. Our safety shield updates the Q-function locally based on safe predictions, which originate from a safe simulation of the environment model. This shielding approach improves performance while maintaining hard safety guarantees. Our experiments on gridworld environments demonstrate that even short prediction horizons can be sufficient to identify the optimal path. We observe that our approach is robust to distribution shifts, e.g., between simulation and reality, without requiring additional training.
>
---
#### [new 026] ENACT: Evaluating Embodied Cognition with World Modeling of Egocentric Interaction
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文提出ENACT基准，评估视觉语言模型是否具备具身认知能力。通过双任务（正向与逆向世界建模）在部分可观测的视角交互中检验模型对动作-效应、空间意识及长时记忆的理解。基于机器人仿真生成数据，揭示当前模型在长交互序列中显著落后于人类，且存在人为偏见。**

- **链接: [https://arxiv.org/pdf/2511.20937v1](https://arxiv.org/pdf/2511.20937v1)**

> **作者:** Qineng Wang; Wenlong Huang; Yu Zhou; Hang Yin; Tianwei Bao; Jianwen Lyu; Weiyu Liu; Ruohan Zhang; Jiajun Wu; Li Fei-Fei; Manling Li
>
> **备注:** Preprint version
>
> **摘要:** Embodied cognition argues that intelligence arises from sensorimotor interaction rather than passive observation. It raises an intriguing question: do modern vision-language models (VLMs), trained largely in a disembodied manner, exhibit signs of embodied cognition? We introduce ENACT, a benchmark that casts evaluation of embodied cognition as world modeling from egocentric interaction in a visual question answering (VQA) format. Framed as a partially observable Markov decision process (POMDP) whose actions are scene graph changes, ENACT comprises two complementary sequence reordering tasks: forward world modeling (reorder shuffled observations given actions) and inverse world modeling (reorder shuffled actions given observations). While conceptually simple, solving these tasks implicitly demands capabilities central to embodied cognition-affordance recognition, action-effect reasoning, embodied awareness, and interactive, long-horizon memory from partially observable egocentric input, while avoiding low-level image synthesis that could confound the evaluation. We provide a scalable pipeline that synthesizes QA pairs from robotics simulation (BEHAVIOR) and evaluates models on 8,972 QA pairs spanning long-horizon home-scale activities. Experiments reveal a performance gap between frontier VLMs and humans that widens with interaction horizon. Models consistently perform better on the inverse task than the forward one and exhibit anthropocentric biases, including a preference for right-handed actions and degradation when camera intrinsics or viewpoints deviate from human vision. Website at https://enact-embodied-cognition.github.io/.
>
---
#### [new 027] Design and Measurements of mmWave FMCW Radar Based Non-Contact Multi-Patient Heart Rate and Breath Rate Monitoring System
- **分类: eess.SY; cs.RO; eess.SP**

- **简介: 该论文提出一种基于毫米波FMCW雷达的非接触式多患者心率与呼吸率监测系统，旨在解决大规模场景下精准、无干扰生命体征监测难题。通过融合多种信号处理方法并利用最小二乘法优化，提升测量精度与鲁棒性，实验表明心率和呼吸率检测准确率分别超过97%和93%。**

- **链接: [https://arxiv.org/pdf/2511.21255v1](https://arxiv.org/pdf/2511.21255v1)**

> **作者:** Jewel Benny; Pranjal Mahajan; Srayan Sankar Chatterjee; Mohd Wajid; Abhishek Srivastava
>
> **备注:** Presented at BioCAS 2023
>
> **摘要:** Recent developments in mmWave radar technologies have enabled the truly non-contact heart-rate (HR) and breath-rate (BR) measurement approaches, which provides a great ease in patient monitoring. Additionally, these technologies also provide opportunities to simultaneously detect HR and BR of multiple patients, which has become increasingly important for efficient mass monitoring scenarios. In this work, a frequency modulated continuous wave (FMCW) mmWave radar based truly non-contact multiple patient HR and BR monitoring system has been presented. Furthermore, a novel approach is also proposed, which combines multiple processing methods using a least squares solution to improve measurement accuracy, generalization, and handle measurement error. The proposed system has been developed using Texas Instruments' FMCW radar and experimental results with multiple subjects are also presented, which show >97% and >93% accuracy in the measured BR and HR values, respectively.
>
---
#### [new 028] Scalable Multisubject Vital Sign Monitoring With mmWave FMCW Radar and FPGA Prototyping
- **分类: eess.SY; cs.RO; eess.SP**

- **简介: 该论文提出一种基于毫米波FMCW雷达与FPGA的可扩展多人体征非接触监测方法，解决传统穿戴式设备带来的不适、校准难及感染风险问题。通过硬件加速实现多目标实时监测，显著提升处理速度与资源效率，验证了系统在复杂场景下的可行性与优越性。**

- **链接: [https://arxiv.org/pdf/2511.21314v1](https://arxiv.org/pdf/2511.21314v1)**

> **作者:** Jewel Benny; Narahari N. Moudhgalya; Mujeev Khan; Hemant Kumar Meena; Mohd Wajid; Abhishek Srivastava
>
> **备注:** Published in IEEE Sensors Journal
>
> **摘要:** In this work, we introduce an innovative approach to estimate the vital signs of multiple human subjects simultaneously in a non-contact way using a Frequency Modulated Continuous Wave (FMCW) radar-based system. Traditional vital sign monitoring methods often face significant limitations, including subject discomfort with wearable devices, challenges in calibration, and the risk of infection transmission through contact measurement devices. To address these issues, this research is motivated by the need for versatile, non-contact vital monitoring solutions applicable in various critical scenarios. This work also explores the challenges of extending this capability to an arbitrary number of subjects, including hardware and theoretical limitations. Supported by rigorous experimental results and discussions, the paper illustrates the system's potential to redefine vital sign monitoring. An FPGA-based implementation is also presented as proof of concept for a hardware-based and portable solution, improving upon previous works by offering 2.7x faster execution and 18.4% less Look-Up Table (LUT) utilization, as well as providing over 7400x acceleration compared to its software counterpart.
>
---
## 更新

#### [replaced 001] Rapid and Safe Trajectory Planning over Diverse Scenes through Diffusion Composition
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对复杂环境中机器人轨迹规划的实时性与安全性难题，提出基于扩散模型的快速安全规划方法。通过无场景依赖的数据生成与测试时模型组合，实现对未见场景的泛化能力，并结合轻量级安全过滤器，确保轨迹安全与可行性。实验验证了其在真实车辆上的有效性。**

- **链接: [https://arxiv.org/pdf/2507.04384v3](https://arxiv.org/pdf/2507.04384v3)**

> **作者:** Wule Mao; Zhouheng Li; Yunhao Luo; Yilun Du; Lei Xie
>
> **摘要:** Safe trajectory planning in complex environments must balance stringent collision avoidance with real-time efficiency, which is a long-standing challenge in robotics. In this work, we present a diffusion-based trajectory planning framework that is both rapid and safe. First, we introduce a scene-agnostic, MPC-based data generation pipeline that efficiently produces large volumes of kinematically feasible trajectories. Building on this dataset, our integrated diffusion planner maps raw onboard sensor inputs directly to kinematically feasible trajectories, enabling efficient inference while maintaining strong collision avoidance. To generalize to diverse, previously unseen scenarios, we compose diffusion models at test time, enabling safe behavior without additional training. We further propose a lightweight, rule-based safety filter that, from the candidate set, selects the trajectory meeting safety and kinematic-feasibility requirements. Across seen and unseen settings, the proposed method delivers real-time-capable inference with high safety and stability. Experiments on an F1TENTH vehicle demonstrate practicality on real hardware. Project page: https://rstp-comp-diffuser.github.io/.
>
---
#### [replaced 002] Analytical Solvers for Common Algebraic Equations Arising in Kinematics Problems
- **分类: cs.RO; math.AG**

- **简介: 该论文针对机器人运动学中常见的四类代数方程，提出解析求解方法，涵盖闭式解、数值算法与鲁棒性处理。解决了通用系数下多解枚举与奇异性问题，实现高效准确求解，并提供可复现的Python工具包及生成代码的提示词，为逆运动学求解奠定基础。**

- **链接: [https://arxiv.org/pdf/2509.01010v2](https://arxiv.org/pdf/2509.01010v2)**

> **作者:** Hai-Jun Su
>
> **摘要:** This paper presents analytical solvers for four common types of algebraic equations encountered in robot kinematics: single trigonometric equations, single-angle trigonometric systems, two-angle trigonometric systems, and bilinear two-angle systems. These equations arise frequently in the kinematics problems, particularly in robot kinematics. We provide detailed solution methods, including closed-form expressions, numerical algorithms, and robustness considerations. The solvers are designed to handle general coefficients, manage singularities, and enumerate all real solutions efficiently. These solvers are implemented in Python packages and can be reproduced by prompting Language Lanuage Models. Sampe prompts are also provided in the public code space Github repo. These prompts can generate a working solver code with one single prompt in coding agent such as OpenAI's Codex 5.1. This work serves as a foundation for developing complete inverse kinematics solvers for various robot architectures. Extensive validation and benchmarking demonstrate the effectiveness and reliability of the proposed methods.
>
---
#### [replaced 003] Steering Flexible Linear Objects in Planar Environments by Two Robot Hands Using Euler's Elastica Solutions
- **分类: cs.RO**

- **简介: 该论文研究双机械手在平面环境中操控柔性线状物体（如电缆、绳索）的任务。针对物体形状难以预测的问题，基于欧拉弹性曲线理论，推导出闭式解，并建立非自交、稳定与避障的判据，构建了完整的规划方案并实现验证。**

- **链接: [https://arxiv.org/pdf/2501.02874v3](https://arxiv.org/pdf/2501.02874v3)**

> **作者:** Aharon Levin; Elon Rimon; Amir Shapiro
>
> **摘要:** The manipulation of flexible objects such as cables, wires and fresh food items by robot hands forms a special challenge in robot grasp mechanics. This paper considers the steering of flexible linear objects in planar environments by two robot hands. The flexible linear object, modeled as an elastic non-stretchable rod, is manipulated by varying the gripping endpoint positions while keeping equal endpoint tangents. The flexible linear object shape has a closed form solution in terms of the grasp endpoint positions and tangents, called Euler's elastica. This paper obtains the elastica solutions under the optimal control framework, then uses the elastica solutions to obtain closed-form criteria for non self-intersection, stability and obstacle avoidance of the flexible linear object. The new tools are incorporated into a planning scheme for steering flexible linear objects in planar environments populated by sparsely spaced obstacles. The scheme is fully implemented and demonstrated with detailed examples.
>
---
#### [replaced 004] Connectivity-Preserving Multi-Agent Area Coverage via Optimal-Transport-Based Density-Driven Optimal Control (D2OC)
- **分类: eess.SY; cs.RO**

- **简介: 该论文针对多智能体非均匀区域覆盖任务，解决现有密度驱动方法无法保证通信连通性的问题。提出基于最优传输的连通性保持型密度驱动最优控制（D2OC）方法，通过平滑连通性惩罚项确保通信不中断，实现高效、连通的非均匀覆盖，显著提升收敛速度与覆盖质量。**

- **链接: [https://arxiv.org/pdf/2511.18579v2](https://arxiv.org/pdf/2511.18579v2)**

> **作者:** Kooktae Lee; Ethan Brook
>
> **备注:** Under review in IEEE Control Systems Letters (LCSS). 6 pages
>
> **摘要:** Multi-agent systems play a central role in area coverage tasks across search-and-rescue, environmental monitoring, and precision agriculture. Achieving non-uniform coverage, where spatial priorities vary across the domain, requires coordinating agents while respecting dynamic and communication constraints. Density-driven approaches can distribute agents according to a prescribed reference density, but existing methods do not ensure connectivity. This limitation often leads to communication loss, reduced coordination, and degraded coverage performance. This letter introduces a connectivity-preserving extension of the Density-Driven Optimal Control (D2OC) framework. The coverage objective, defined using the Wasserstein distance between the agent distribution and the reference density, admits a convex quadratic program formulation. Communication constraints are incorporated through a smooth connectivity penalty, which maintains strict convexity, supports distributed implementation, and preserves inter-agent communication without imposing rigid formations. Simulation studies show that the proposed method consistently maintains connectivity, improves convergence speed, and enhances non-uniform coverage quality compared with density-driven schemes that do not incorporate explicit connectivity considerations.
>
---
#### [replaced 005] Heterogeneous Multi-robot Task Allocation for Long-Endurance Missions in Dynamic Scenarios
- **分类: cs.RO**

- **简介: 该论文研究异构多机器人在动态环境中的长时任务分配问题，针对电池限制、任务分割与协同执行等挑战，提出混合整数线性规划模型与启发式算法，并构建在线重规划框架，实现高效任务分配与动态响应。**

- **链接: [https://arxiv.org/pdf/2411.02062v3](https://arxiv.org/pdf/2411.02062v3)**

> **作者:** Alvaro Calvo; Jesus Capitan
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** We present a framework for Multi-Robot Task Allocation (MRTA) in heterogeneous teams performing long-endurance missions in dynamic scenarios. Given the limited battery of robots, especially for aerial vehicles, we allow for robot recharges and the possibility of fragmenting and/or relaying certain tasks. We also address tasks that must be performed by a coalition of robots in a coordinated manner. Given these features, we introduce a new class of heterogeneous MRTA problems which we analyze theoretically and optimally formulate as a Mixed-Integer Linear Program. We then contribute a heuristic algorithm to compute approximate solutions and integrate it into a mission planning and execution architecture capable of reacting to unexpected events by repairing or recomputing plans online. Our experimental results show the relevance of our newly formulated problem in a realistic use case for inspection with aerial robots. We assess the performance of our heuristic solver in comparison with other variants and with exact optimal solutions in small-scale scenarios. In addition, we evaluate the ability of our replanning framework to repair plans online.
>
---
#### [replaced 006] MonoMPC: Monocular Vision Based Navigation with Learned Collision Model and Risk-Aware Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文针对单目视觉导航中因缺乏深度信息导致的碰撞检测难题，提出MonoMPC方法。通过将噪声较大的深度估计作为上下文输入，训练一个预测最小障碍物间隙分布的碰撞模型，并结合风险感知模型预测控制，实现安全高效导航。在真实环境中显著降低碰撞率并提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2508.07387v3](https://arxiv.org/pdf/2508.07387v3)**

> **作者:** Basant Sharma; Prajyot Jadhav; Pranjal Paul; K. Madhava Krishna; Arun Kumar Singh
>
> **摘要:** Navigating unknown environments with a single RGB camera is challenging, as the lack of depth information prevents reliable collision-checking. While some methods use estimated depth to build collision maps, we found that depth estimates from vision foundation models are too noisy for zero-shot navigation in cluttered environments. We propose an alternative approach: instead of using noisy estimated depth for direct collision-checking, we use it as a rich context input to a learned collision model. This model predicts the distribution of minimum obstacle clearance that the robot can expect for a given control sequence. At inference, these predictions inform a risk-aware MPC planner that minimizes estimated collision risk. We proposed a joint learning pipeline that co-trains the collision model and risk metric using both safe and unsafe trajectories. Crucially, our joint-training ensures well calibrated uncertainty in our collision model that improves navigation in highly cluttered environments. Consequently, real-world experiments show reductions in collision-rate and improvements in goal reaching and speed over several strong baselines.
>
---
#### [replaced 007] X-Nav: Learning End-to-End Cross-Embodiment Navigation for Mobile Robots
- **分类: cs.RO**

- **简介: 该论文提出X-Nav框架，解决移动机器人在不同机体形态间导航的泛化问题。通过两阶段学习：先在多种随机生成的机器人形态上训练专家策略，再用Nav-ACT方法提炼出统一的通用策略，实现端到端跨机体导航。实验证明其可零样本迁移至新形态与真实环境，具备良好可扩展性与实用性。**

- **链接: [https://arxiv.org/pdf/2507.14731v2](https://arxiv.org/pdf/2507.14731v2)**

> **作者:** Haitong Wang; Aaron Hao Tan; Angus Fung; Goldie Nejat
>
> **摘要:** Existing navigation methods are primarily designed for specific robot embodiments, limiting their generalizability across diverse robot platforms. In this paper, we introduce X-Nav, a novel framework for end-to-end cross-embodiment navigation where a single unified policy can be deployed across various embodiments for both wheeled and quadrupedal robots. X-Nav consists of two learning stages: 1) multiple expert policies are trained using deep reinforcement learning with privileged observations on a wide range of randomly generated robot embodiments; and 2) a single general policy is distilled from the expert policies via navigation action chunking with transformer (Nav-ACT). The general policy directly maps visual and proprioceptive observations to low-level control commands, enabling generalization to novel robot embodiments. Simulated experiments demonstrated that X-Nav achieved zero-shot transfer to both unseen embodiments and photorealistic environments. A scalability study showed that the performance of X-Nav improves when trained with an increasing number of randomly generated embodiments. An ablation study confirmed the design choices of X-Nav. Furthermore, real-world experiments were conducted to validate the generalizability of X-Nav in real-world environments.
>
---
#### [replaced 008] BRIC: Bridging Kinematic Plans and Physical Control at Test Time
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BRIC框架，解决扩散模型生成的运动规划与物理控制器间执行偏差问题。针对长期人类运动生成任务，通过测试时动态调整控制器并轻量引导扩散模型，实现物理上合理且连贯的动作执行，显著提升复杂场景下的运动一致性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.20431v2](https://arxiv.org/pdf/2511.20431v2)**

> **作者:** Dohun Lim; Minji Kim; Jaewoon Lim; Sungchan Kim
>
> **备注:** Accepted to AAAI'26
>
> **摘要:** We propose BRIC, a novel test-time adaptation (TTA) framework that enables long-term human motion generation by resolving execution discrepancies between diffusion-based kinematic motion planners and reinforcement learning-based physics controllers. While diffusion models can generate diverse and expressive motions conditioned on text and scene context, they often produce physically implausible outputs, leading to execution drift during simulation. To address this, BRIC dynamically adapts the physics controller to noisy motion plans at test time, while preserving pre-trained skills via a loss function that mitigates catastrophic forgetting. In addition, BRIC introduces a lightweight test-time guidance mechanism that steers the diffusion model in the signal space without updating its parameters. By combining both adaptation strategies, BRIC ensures consistent and physically plausible long-term executions across diverse environments in an effective and efficient manner. We validate the effectiveness of BRIC on a variety of long-term tasks, including motion composition, obstacle avoidance, and human-scene interaction, achieving state-of-the-art performance across all tasks.
>
---
#### [replaced 009] Simultaneous Calibration of Noise Covariance and Kinematics for State Estimation of Legged Robots via Bi-level Optimization
- **分类: cs.RO; math.OC**

- **简介: 该论文针对腿式机器人在动态环境中状态估计的噪声协方差与运动学参数未知问题，提出一种双层优化框架。通过嵌套估计算法，联合优化噪声协方差与模型参数，实现数据驱动的状态估计、传感器及运动学协同校准，显著提升估计精度与不确定性校准效果。**

- **链接: [https://arxiv.org/pdf/2510.11539v2](https://arxiv.org/pdf/2510.11539v2)**

> **作者:** Denglin Cheng; Jiarong Kang; Xiaobin Xiong
>
> **摘要:** Accurate state estimation is critical for legged and aerial robots operating in dynamic, uncertain environments. A key challenge lies in specifying process and measurement noise covariances, which are typically unknown or manually tuned. In this work, we introduce a bi-level optimization framework that jointly calibrates covariance matrices and kinematic parameters in an estimator-in-the-loop manner. The upper level treats noise covariances and model parameters as optimization variables, while the lower level executes a full-information estimator. Differentiating through the estimator allows direct optimization of trajectory-level objectives, resulting in accurate and consistent state estimates. We validate our approach on quadrupedal and humanoid robots, demonstrating significantly improved estimation accuracy and uncertainty calibration compared to hand-tuned baselines. Our method unifies state estimation, sensor, and kinematics calibration into a principled, data-driven framework applicable across diverse robotic platforms.
>
---
#### [replaced 010] scipy.spatial.transform: Differentiable Framework-Agnostic 3D Transformations in Python
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对3D刚体变换在不同框架下实现不一致、难以自动微分的问题，提出兼容NumPy、JAX、PyTorch等数组库的可微分3D变换框架。通过重构SciPy.spatial.transform模块，实现跨框架、GPU/TPU加速与自动微分，支持机器人、视觉等领域的可微科学计算。**

- **链接: [https://arxiv.org/pdf/2511.18157v2](https://arxiv.org/pdf/2511.18157v2)**

> **作者:** Martin Schuck; Alexander von Rohr; Angela P. Schoellig
>
> **备注:** Accepted as oral at the 1st Workshop on Differentiable Systems and Scientific Machine Learning @ EurIPS 2025
>
> **摘要:** Three-dimensional rigid-body transforms, i.e. rotations and translations, are central to modern differentiable machine learning pipelines in robotics, vision, and simulation. However, numerically robust and mathematically correct implementations, particularly on SO(3), are error-prone due to issues such as axis conventions, normalizations, composition consistency and subtle errors that only appear in edge cases. SciPy's spatial$.$transform module is a rigorously tested Python implementation. However, it historically only supported NumPy, limiting adoption in GPU-accelerated and autodiff-based workflows. We present a complete overhaul of SciPy's spatial$.$transform functionality that makes it compatible with any array library implementing the Python array API, including JAX, PyTorch, and CuPy. The revised implementation preserves the established SciPy interface while enabling GPU/TPU execution, JIT compilation, vectorized batching, and differentiation via native autodiff of the chosen backend. We demonstrate how this foundation supports differentiable scientific computing through two case studies: (i) scalability of 3D transforms and rotations and (ii) a JAX drone simulation that leverages SciPy's Rotation for accurate integration of rotational dynamics. Our contributions have been merged into SciPy main and will ship in the next release, providing a framework-agnostic, production-grade basis for 3D spatial math in differentiable systems and ML.
>
---
#### [replaced 011] Multi-Agent Monocular Dense SLAM With 3D Reconstruction Priors
- **分类: cs.RO**

- **简介: 该论文提出首个多智能体单目稠密SLAM系统，旨在提升单目SLAM的效率与扩展性。针对现有方法计算成本高、仅支持单机的问题，引入3D重建先验与基于回环检测的映射融合机制，实现多智能体协同高效建图，兼顾精度与实时性。**

- **链接: [https://arxiv.org/pdf/2511.19031v2](https://arxiv.org/pdf/2511.19031v2)**

> **作者:** Yuchen Zhou; Haihang Wu
>
> **摘要:** Monocular Simultaneous Localization and Mapping (SLAM) aims to estimate a robot's pose while simultaneously reconstructing an unknown 3D scene using a single camera. While existing monocular SLAM systems generate detailed 3D geometry through dense scene representations, they are computationally expensive due to the need for iterative optimization. To address this challenge, MASt3R-SLAM utilizes learned 3D reconstruction priors, enabling more efficient and accurate estimation of both 3D structures and camera poses. However, MASt3R-SLAM is limited to single-agent operation. In this paper, we extend MASt3R-SLAM to introduce the first multi-agent monocular dense SLAM system. Each agent performs local SLAM using a 3D reconstruction prior, and their individual maps are fused into a globally consistent map through a loop-closure-based map fusion mechanism. Our approach improves computational efficiency compared to state-of-the-art methods, while maintaining similar mapping accuracy when evaluated on real-world datasets.
>
---
#### [replaced 012] MAPF-HD: Multi-Agent Path Finding in High-Density Environments
- **分类: cs.MA; cs.RO**

- **简介: 该论文研究高密度环境下的多智能体路径规划（MAPF）任务，旨在解决传统整数线性规划方法在高密度场景下计算耗时过长的问题。提出PHANS方法，通过增量式代理与空位交换实现快速路径优化，可在数秒内完成大规模环境下的路径规划，显著提升效率，适用于仓储物流等实际场景。**

- **链接: [https://arxiv.org/pdf/2509.06374v2](https://arxiv.org/pdf/2509.06374v2)**

> **作者:** Hiroya Makino; Seigo Ito
>
> **备注:** 9 pages, 12 figures
>
> **摘要:** Multi-agent path finding (MAPF) involves planning efficient paths for multiple agents to move simultaneously while avoiding collisions. In typical warehouse environments, agents are often sparsely distributed along aisles; however, increasing the agent density can improve space efficiency. When the agent density is high, it becomes necessary to optimize the paths not only for goal-assigned agents but also for those obstructing them. This study proposes a novel MAPF framework for high-density environments (MAPF-HD). Several studies have explored MAPF in similar settings using integer linear programming (ILP). However, ILP-based methods require substantial computation time to optimize all agent paths simultaneously. Even in small grid-based environments with fewer than $100$ cells, these computations can take tens to hundreds of seconds. Such high computational costs render these methods impractical for large-scale applications such as automated warehouses and valet parking. To address these limitations, we introduce the phased null-agent swapping (PHANS) method. PHANS employs a heuristic approach to incrementally swap positions between agents and empty vertices. This method solves the MAPF-HD problem within a few seconds, even in large environments containing more than $700$ cells. The proposed method has the potential to improve efficiency in various real-world applications such as warehouse logistics, traffic management, and crowd control. The implementation is available at https://github.com/ToyotaCRDL/MAPF-in-High-Density-Envs.
>
---
#### [replaced 013] Safety Control of Service Robots with LLMs and Embodied Knowledge Graphs
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对服务机器人在复杂环境中安全行为不足的问题，提出融合大语言模型（LLM）与具身知识图谱（EKG）的安全控制框架。通过预设的具身机器人控制提示（ERCP）引导LLM生成安全指令，并由EKG动态验证动作合规性，提升机器人在真实任务中的安全执行能力。**

- **链接: [https://arxiv.org/pdf/2405.17846v2](https://arxiv.org/pdf/2405.17846v2)**

> **作者:** Yong Qi; Gabriel Kyebambo; Siyuan Xie; Wei Shen; Shenghui Wang; Bitao Xie; Bin He; Zhipeng Wang; Shuo Jiang
>
> **摘要:** Safety limitations in service robotics across various industries have raised significant concerns about the need for robust mechanisms ensuring that robots adhere to safe practices, thereby preventing actions that might harm humans or cause property damage. Despite advances, including the integration of Knowledge Graphs (KGs) with Large Language Models (LLMs), challenges in ensuring consistent safety in autonomous robot actions persist. In this paper, we propose a novel integration of Large Language Models with Embodied Robotic Control Prompts (ERCPs) and Embodied Knowledge Graphs (EKGs) to enhance the safety framework for service robots. ERCPs are designed as predefined instructions that ensure LLMs generate safe and precise responses. These responses are subsequently validated by EKGs, which provide a comprehensive knowledge base ensuring that the actions of the robot are continuously aligned with safety protocols, thereby promoting safer operational practices in varied contexts. Our experimental setup involved diverse real-world tasks, where robots equipped with our framework demonstrated significantly higher compliance with safety standards compared to traditional methods. This integration fosters secure human-robot interactions and positions our methodology at the forefront of AI-driven safety innovations in service robotics.
>
---
#### [replaced 014] Development of a Testbed for Autonomous Vehicles: Integrating MPC Control with Monocular Camera Lane Detection
- **分类: cs.RO**

- **简介: 该论文针对自主车辆路径跟踪精度与稳定性问题，提出融合单目相机车道检测与模型预测控制（MPC）的方法。通过边缘检测与滑动窗口提取车道线，构建基于自行车模型的MPC控制器，在ROS Gazebo仿真中验证，使轨迹跟踪均方根误差降低27.65%，有效提升控制性能。**

- **链接: [https://arxiv.org/pdf/2511.19655v2](https://arxiv.org/pdf/2511.19655v2)**

> **作者:** Shantanu Rahman; Nayeb Hasin; Mainul Islam; Golam Sarowar
>
> **备注:** 49 pages, 23 figures
>
> **摘要:** Autonomous vehicles are becoming popular day by day not only for autonomous road traversal but also for industrial automation, farming and military. Most of the standard vehicles follow the Ackermann style steering mechanism. This has become to de facto standard for large and long faring vehicles. The local planner of an autonomous vehicle controls the low-level vehicle movement upon which the vehicle will perform its motor actuation. In our work, we focus on autonomous vehicles in road and perform experiments to analyze the effect of low-level controllers in the simulation and a real environment. To increase the precision and stability of trajectory tracking in autonomous cars, a novel method that combines lane identification with Model Predictive Control (MPC) is presented. The research focuses on camera-equipped autonomous vehicles and uses methods like edge recognition, sliding window-based straight-line identification for lane line extraction, and dynamic region of interest (ROI) extraction. Next, to follow the identified lane line, an MPC built on a bicycle vehicle dynamics model is created. A single-lane road simulation model is built using ROS Gazebo and tested in order to verify the controller's performance. The root mean square error between the optimal tracking trajectory and the target trajectory was reduced by 27.65% in the simulation results, demonstrating the high robustness and flexibility of the developed controller.
>
---
#### [replaced 015] Floor Plan-Guided Visual Navigation Incorporating Depth and Directional Cues
- **分类: cs.RO**

- **简介: 该论文针对室内视觉导航任务，解决RGB与平面图间模态差异及未知环境定位难题。提出GlocDiff模型，融合平面图全局规划与深度感知的局部视觉特征，通过扩散策略实现精准导航，并引入噪声训练提升鲁棒性，显著提升导航性能与实际应用潜力。**

- **链接: [https://arxiv.org/pdf/2511.01493v2](https://arxiv.org/pdf/2511.01493v2)**

> **作者:** Wei Huang; Jiaxin Li; Zang Wan; Huijun Di; Wei Liang; Zhu Yang
>
> **摘要:** Guiding an agent to a specific target in indoor environments based solely on RGB inputs and a floor plan is a promising yet challenging problem. Although existing methods have made significant progress, two challenges remain unresolved. First, the modality gap between egocentric RGB observations and the floor plan hinders the integration of visual and spatial information for both local obstacle avoidance and global planning. Second, accurate localization is critical for navigation performance, but remains challenging at deployment in unseen environments due to the lack of explicit geometric alignment between RGB inputs and floor plans. We propose a novel diffusion-based policy, denoted as GlocDiff, which integrates global path planning from the floor plan with local depth-aware features derived from RGB observations. The floor plan offers explicit global guidance, while the depth features provide implicit geometric cues, collectively enabling precise prediction of optimal navigation directions and robust obstacle avoidance. Moreover, GlocDiff introduces noise perturbation during training to enhance robustness against pose estimation errors, and we find that combining this with a relatively stable VO module during inference results in significantly improved navigation performance. Extensive experiments on the FloNa benchmark demonstrate GlocDiff's efficiency and effectiveness in achieving superior navigation performance, and the success of real-world deployments also highlights its potential for widespread practical applications.
>
---
