# 机器人 cs.RO

- **最新发布 43 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] VLA-Pro: Cross-Task Procedural Memory Transfer for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出VLA-Pro，解决视觉-语言-动作模型在跨任务泛化中的经验迁移问题。通过存储和检索程序记忆，提升新任务的执行效果。**

- **链接: [https://arxiv.org/pdf/2605.29562](https://arxiv.org/pdf/2605.29562)**

> **作者:** Shengyu Si; Yuanzhuo Lu; Ruimeng Yang; Ziyi Ye; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Vision-Language-Action~(VLA) models have shown strong potential for general-purpose robotic manipulation, yet they still struggle to generalize to unseen tasks that necessitate transferring relevant experience across objects, scenes, and action patterns. This paper proposes VLA-Pro, a plug-and-play framework designed to enhance cross-task generalization by storing task-relevant procedural memories at training time and transferring these memories during inference. Specifically, VLA-Pro stores task-specific LoRA adapters as parameterized procedural memories during training. At inference time, VLA-Pro retrieves relevant procedural memories based on the current multi-modal context and dynamically fuses these memories for generating the current action chunk. Experiments on RoboTwin, RLBench, and real-world manipulation tasks show that VLA-Pro consistently improves cross-task generalization across multiple backbones, achieving up to a 207% relative improvement in simulation and increasing real-world success rate from 5.8% to 65.0%. These results suggest that procedural memory retrieval and adaptation provide an effective mechanism for transferring manipulation experience to novel tasks while preserving modularity and execution stability.
>
---
#### [new 002] VLAConf: Calibrated Task-Success Confidence for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作模型的置信度估计任务，旨在解决现有方法计算效率低和跨架构泛化差的问题。提出VLAConf框架，通过单次前向传播直接估计置信度，提升效率与适用性。**

- **链接: [https://arxiv.org/pdf/2605.29605](https://arxiv.org/pdf/2605.29605)**

> **作者:** Dehao Huang; Aoxiang Gu; Chengjie Zhang; Bolin Zou; Wenlong Dong; Zilang Cen; Yue Wang; Hong Zhang
>
> **备注:** 11 pages, 7 figures
>
> **摘要:** Confidence estimation for Vision-Language-Action (VLA) models is essential for robots to perform manipulation tasks in the open world, providing crucial signals for risk-sensitive decision-making and failure anticipation. Existing confidence estimation methods typically rely on ensemble-based paradigms or action-token probabilities to predict the likelihood of task success. However, they still encounter challenges in computational efficiency and cross-architecture generalizability. These methods usually require repeated sampling, leading to inference inefficiency, and are restricted to VLA models with discrete action outputs, making them difficult to apply to continuous action spaces. To address this issue, we propose VLAConf, a one-class discriminative confidence framework. By leveraging frozen pretrained VLA internal representations, VLAConf directly estimates step-wise anomaly scores in a single forward pass using a lightweight confidence head, thereby eliminating the overhead of exhaustive resampling. We additionally use step-conditioned modeling to encode rollout-phase information along the manipulation trajectory. Experiments on the LIBERO benchmark demonstrate that VLAConf significantly improves the quality of the confidence signal constructed for post-hoc calibration, outperforming existing baselines by a large margin in inference efficiency. The effectiveness of VLAConf is further validated in real-robot experiments. To access the source code and supplementary videos, visit this https URL.
>
---
#### [new 003] Learning to Feel Materials from Multisensory Tactile Data via Interpretable Models
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于材料感知任务，旨在解决人类触觉感知与机器人触觉系统之间的差距。通过构建可解释模型，利用多感官触觉数据提升材料识别准确性。**

- **链接: [https://arxiv.org/pdf/2605.29572](https://arxiv.org/pdf/2605.29572)**

> **作者:** Li Zou; Yasemin Vardar
>
> **备注:** 12 pages, 3 figures, journal
>
> **摘要:** Human tactile perception of materials relies on complex multisensory touch cues, yet the relationship between low-level tactile signals and perceptual representations remains poorly understood. This knowledge gap hinders the integration of touch in digital environments and the development of robots capable of human-like tactile perception. Here, we present an interpretable computational framework for modeling human material perception and recognition using multisensory touch data. Our framework comprises three interconnected models: Model 1 maps finger-surface interaction features to psychophysical sensory attributes, Model 2 classifies materials based on these perceptual representations, and Model 3 directly classifies materials from tactile features. The results showed that combining information from pressing, static contact, and sliding interactions improves prediction accuracy, and that thermal cues are particularly informative for both perceptual modeling and material classification. These findings highlight the importance of thermal and compliance cues, which remain underrepresented in current robotic fingers and haptic displays. Incorporating such cues may enhance artificial systems' ability to approximate human material perception and guide the design of more perceptually grounded haptic interfaces.
>
---
#### [new 004] The Open Motion Planning Library 2.0
- **分类: cs.RO**

- **简介: 本文介绍OMPL 2.0，一个用于实时运动规划的库，解决复杂环境下的路径规划问题，整合AI工作流，提升性能。**

- **链接: [https://arxiv.org/pdf/2605.29301](https://arxiv.org/pdf/2605.29301)**

> **作者:** Weihang Guo; Theodoros Tyrovouzis; Emiliano Flores; Clayton W. Ramsey; Zachary K. Kingston; Ioan A. Şucan; Mark Moll; Lydia E. Kavraki
>
> **摘要:** The Open Motion Planning Library (OMPL), first released in 2008, has become a cornerstone of the motion planning community, providing implementations of a wide range of state-of-the-art sampling-based algorithms. Over almost two decades of continuous development, we have steadily expanded the library with new planners, state spaces, and problem formulations. These additions range from asymptotically optimal and lazy planners to constrained motion planning and planning with temporal-logic goals. Building on this foundation, we introduce OMPL 2.0, a major evolution of the library that targets real-time motion planning through hardware acceleration and integrates seamlessly with modern AI research workflows. We also reflect on how OMPL and the field of motion planning have grown together over the years, and discuss the library's broader impact on the research community.
>
---
#### [new 005] VE2VF: Vision-Enabled to Vision-Free Distillation via Real-world Reinforcement Learning for Robust Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉依赖策略在真实环境中泛化能力差的问题。通过人机协同的强化学习框架，将视觉策略知识迁移至无视觉的学生策略，提升鲁棒性和适应性。**

- **链接: [https://arxiv.org/pdf/2605.29564](https://arxiv.org/pdf/2605.29564)**

> **作者:** Victor Kowalski; Chengxi Li; Dongheui Lee
>
> **摘要:** When using reinforcement learning (RL) for contact-rich robotic manipulation, vision can provide task-relevant information that accelerates learning beyond what proprioception alone can achieve. However, vision-enabled policies tend to overfit to the visual conditions seen during training, limiting their robustness and transferability. We present a human-in-the-loop RL framework that employs teacher-student distillation to achieve robust performance across multiple task variants, trained entirely in the real world without requiring domain randomization or data augmentation. A vision-enabled teacher distills its knowledge into a vision-free student that relies solely on pose, twist, and wrench sensing, combining fast training with strong task generalization. On the real-world NIST assembly benchmark board, our approach achieves 95\% overall success after approximately 50 minutes of training on 3 representative tasks, including robust generalization to 8 unseen task variants. Fine-tuning with distillation achieves full success on the most challenging task. We demonstrate that the resulting policies outperform baselines in both robustness and adaptability.
>
---
#### [new 006] ElegantVLA: Learning When to Think for Efficient Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出ElegantVLA，解决VLA模型计算效率低的问题，通过动态调度提升推理速度，提升机器人控制频率。**

- **链接: [https://arxiv.org/pdf/2605.29438](https://arxiv.org/pdf/2605.29438)**

> **作者:** Ye Li; Huanan Liu; Kangye Ji; Yuan Meng; Jiajun Fan; Yuansong Wang; Shiyu Qin; Chenglei Wu; Shu-Tao Xia; Zhi Wang
>
> **摘要:** Vision-Language-Action (VLA) models are a powerful paradigm for generalist robotic control. However, their high computational cost and limited control frequency hinder real-time robotic manipulation, especially when large vision-language backbones and iterative action heads run at every control step. Existing VLA acceleration methods often optimize individual components or rely on fixed acceleration rules, treating different control steps with largely fixed computation and overlooking the non-uniform reasoning demands of sequential embodied control. Inspired by human motor control, where cognitive and feedback resources concentrate on goal-sensitive stages, we argue that VLA models should learn when to invest full computation and when to reuse prior computation. We propose ElegantVLA, a plug-in phase-adaptive inference framework that accelerates VLA models through intra-model dynamic compute scheduling. ElegantVLA introduces a lightweight scheduler that observes temporal representation similarity, robot-motion cues, and episode progress to jointly allocate computation across the vision encoder, LLM, and action head. For perception-language reasoning, the scheduler selects a five-level Vision-LLM compute mode, from full recomputation to multi-step temporal reuse, based on visual-language representation stability. For action generation, it selects a three-level denoising mode, reusing intermediate denoising states during stable motion while preserving full refinement for goal-sensitive stages. By coordinating these decisions, ElegantVLA offers a general acceleration framework for modern VLA pipelines with explicit action-generation modules, without modifying or retraining the base model. Experiments on GR00T and CogACT achieve up to 2.55x and 3.77x speedup, and on six real-world GR00T tasks ElegantVLA cuts computation by 2.18x while raising control frequency from 13.8 Hz to 26.3 Hz.
>
---
#### [new 007] RoboWits: Unexpected Challenges for Robotic Creative Problem Solving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RoboWits基准，用于评估机器人在意外挑战下的认知推理与创造性问题解决能力。针对现有基准缺乏此类评估的问题，设计了多任务生成框架，构建了多样化测试场景，并对比了不同机器人策略的表现。**

- **链接: [https://arxiv.org/pdf/2605.30326](https://arxiv.org/pdf/2605.30326)**

> **作者:** Chunru Lin; Hongxin Zhang; Fenghao Yu; Zhehuan Chen; Thomas L. Griffiths; Yejin Choi; David Held; Chuang Gan
>
> **备注:** The first two authors contributed equally
>
> **摘要:** The ability to reason, adapt, and creatively solve problems under unexpected challenges is essential for robots operating in real-world environments. However, current robotic benchmarks primarily emphasize skill-level execution and provide limited insight into such cognitive reasoning capabilities. We introduce RoboWits, a bi-manual robotic benchmark designed to systematically evaluate cognitive reasoning, creative tool use, and robustness to unexpected conditions. To enable scalable construction of high-quality reasoning-centric unexpected scenarios, we propose an automated task generation pipeline formulated as a multi-agent cooperative framework, comprising agents for seed task generation and verification, metric generation, scene generation, and task mutation. Using the pipeline, we curated 30 diverse seed tasks and 208 tasks with mutations and graded difficulty across geometry, material, and assembly-based reasoning. We benchmark popular robot policies, pre-trained VLAs, and oracle-state planners. Our results reveal a significant performance gap: while pre-trained VLAs exhibit preliminary success on seed tasks after single-task fine-tuning, they struggle to perform on mutated tasks, implying their brittleness in manipulation tasks requiring reasoning, strategy adaptation, and robustness to deceptive or constrained environments. Project page is available at this https URL.
>
---
#### [new 008] Decentralized LLM-Driven Coordination of Acoustic Robots for Contactless Object Manipulation
- **分类: cs.RO**

- **简介: 该论文属于多机器人系统任务，解决自然语言指令驱动分布式声学机器人进行非接触物体操作的问题。通过LLM和语音识别实现任务规划与协调执行。**

- **链接: [https://arxiv.org/pdf/2605.29378](https://arxiv.org/pdf/2605.29378)**

> **作者:** Yingying Wang; Narsimlu Kemsaram; Sriram Subramanian
>
> **备注:** This paper has been accepted for publication in the Proceedings of the 2026 IEEE 22nd International Conference on Automation Science and Engineering (CASE 2026), August 17-21, 2026, Shenyang, China
>
> **摘要:** Natural language interfaces can simplify interaction with multi-robot systems, especially when non-expert users need to issue high-level commands. Acoustic manipulation using ultrasonic phased arrays also enables contactless object handling for applications such as healthcare, laboratory automation, and precision transport. However, combining large language models (LLMs) with distributed acoustic mobile robots remains underexplored. This paper presents a decentralized framework for natural language-driven coordination of acoustic robots for contactless object manipulation. The system converts spoken instructions into executable multi-robot task plans using Whisper-based speech recognition, LLM-based semantic parsing, structured JSON task representation, and distributed scheduling. The JSON schema encodes robot assignments, temporal dependencies, spatial constraints, and synchronization requirements for sequential, parallel, and synchronized execution. The system is implemented on two TurtleBot3-based acoustic robots, each equipped with an ultrasonic phased array for contactless object transport. Experiments were conducted in three scenarios: sequential execution, parallel multi-robot transport, and synchronized cooperative manipulation. The system achieved task success rates of 96 percent for sequential tasks, 86 percent for parallel execution, and 70 percent for synchronized collaborative transport. These results show that natural language commands can be transformed into distributed robot actions for contactless manipulation, highlighting the potential of LLM-driven automation for human-robot interaction in distributed robotic systems.
>
---
#### [new 009] How to Relieve Distribution Shifts in Semantic Segmentation for Off-Road Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于语义分割任务，针对越野环境中的分布偏移问题，提出ST-Seg框架，通过风格扩展和纹理正则化提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.29599](https://arxiv.org/pdf/2605.29599)**

> **作者:** Ji-Hoon Hwang; Daeyoung Kim; Hyung-Suk Yoon; Dong-Wook Kim; Seung-Woo Seo
>
> **备注:** 8 pages, 6 figures. Accepted to IEEE Robotics and Automation Letters (RA-L). \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses
>
> **摘要:** Semantic segmentation is crucial for autonomous navigation in off-road environments, enabling precise classification of surroundings to identify traversable regions. However, distinctive factors inherent to off-road conditions, such as source-target domain discrepancies and sensor corruption from rough terrain, can result in distribution shifts that alter the data differently from the trained conditions. This often leads to inaccurate semantic label predictions and subsequent failures in navigation tasks. To address this, we propose ST-Seg, a novel framework that expands the source distribution through style expansion (SE) and texture regularization (TR). Unlike prior methods that implicitly apply generalization within a fixed source distribution, ST-Seg offers an intuitive approach for distribution shift. Specifically, SE broadens domain coverage by generating diverse realistic styles, augmenting the limited style information of the source domain. TR stabilizes local texture representation affected by style-augmented learning through a deep texture manifold. Experiments across various distribution-shifted target domains demonstrate the effectiveness of ST-Seg, with substantial improvements over existing methods. These results highlight the robustness of ST-Seg, enhancing the real-world applicability of semantic segmentation for off-road navigation.
>
---
#### [new 010] Gaze2Act: Gaze-Conditioned Vision-Language-Action Policies for Interactive Robot Manipulation
- **分类: cs.RO**

- **简介: 论文提出Gaze2Act框架，解决机器人交互中语言指令不明确的问题。通过融合人类注视，提升目标识别与操作精度，适用于复杂任务控制。**

- **链接: [https://arxiv.org/pdf/2605.30282](https://arxiv.org/pdf/2605.30282)**

> **作者:** Kuangji Zuo; Gen Li; Bofan Lyu; Yanshuo Lu; Boyu Ma; Shijia Han; Xinyu Zhou; Xichen Yuan; Chuhao Zhou; Jiaqi Bai; Geng Li; Jianfei Yang
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have recently shown strong potential for robot learning by following language instructions. However, in practice, language alone is often insufficient to precisely convey human intent. It is difficult to describe which exact object to interact with among similar candidates, where to act on the object, or how the target may change during execution. To address this limitation, we propose Gaze2Act, a novel VLA framework that leverages human gaze as a dynamic and intuitive intent signal for complex interactive manipulation. Gaze2Act first bridges the ego-exo view gap by mapping first-person gaze into the robot's perspective through cross-view semantic matching, producing both an object mask and a gaze point for coarse-to-fine target specification. These cues are then integrated into the policy through perception-level prompting and action-level conditioning, allowing the robot to attend to relevant regions and execute precise interactions under dynamic intent. In a systematic evaluation across seven task categories and 16 real-robot tasks on a Unitree G1 humanoid, Gaze2Act achieves state-of-the-art performance in both intent accuracy and task success rate. It notably outperforms baselines in object disambiguation, fine-grained interaction, and dynamic intent steering. These results demonstrate that human gaze provides a natural, low-burden, and highly expressive modality for human-in-the-loop VLA control.
>
---
#### [new 011] Multi-Resolution End-to-End Deep Neural Network for Optimizing Latency-Accuracy Tradeoff in Autonomous Driving
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于自动驾驶任务，解决实时系统中延迟与准确性的权衡问题。提出多分辨率端到端神经网络，在保证安全的前提下优化延迟。**

- **链接: [https://arxiv.org/pdf/2605.29138](https://arxiv.org/pdf/2605.29138)**

> **作者:** Qitao Weng; Heechul Yun
>
> **备注:** ICCPS 2026
>
> **摘要:** Latency-accuracy tradeoffs are fundamental in real-time applications of deep neural networks (DNNs) for cyber-physical systems. In autonomous driving, in particular, safety depends on both prediction quality and the end-to-end delay from sensing to actuation. We observe that (1) when latency is accounted for, the latency-optimal network configuration varies with scene context and compute availability; and (2) a single fixed-resolution model becomes suboptimal as conditions change. We present a multi-resolution, end-to-end deep neural network for the CARLA urban driving challenge using monocular camera input. Our approach employs a convolutional neural network (CNN) that supports multiple input resolutions through per-resolution batch normalization, enabling runtime selection of an ideal input scale under a latency budget, as well as resolution retargeting, which allows multi-resolution training without access to the original training dataset. We implement and evaluate our multi-resolution end-to-end CNN in CARLA to explore the latency-safety frontier. Results show consistent improvements in per-route safety metrics - lane invasions, red-light infractions, and collisions - relative to fixed-resolution baselines.
>
---
#### [new 012] CA-AC-MPC: CUDA-Accelerated Actor-Critic Model Predictive Control
- **分类: cs.RO; cs.AI; cs.DC**

- **简介: 该论文属于控制领域，解决AC-MPC训练与推理延迟高的问题，通过CUDA加速提升效率，保持控制性能。**

- **链接: [https://arxiv.org/pdf/2605.29155](https://arxiv.org/pdf/2605.29155)**

> **作者:** Antoonio Buo; Vittorio Cammarota; Michele Avagnale; Pierluigi Arpenti; Vincenzo Lippiello; Fabio Ruggiero
>
> **备注:** Accepted for presentation at the 2026 International Conference on Unmanned Aircraft Systems, ICUAS 2026
>
> **摘要:** In the literature, actor-critic model predictive control (AC-MPC) integrates MPC with reinforcement learning to enable high-performance control of complex dynamical systems. However, its differentiable MPC layer requires repeatedly solving an optimization problem in both the forward and backward passes, leading to substantial training and inference latency. This paper tackles this bottleneck introducing a CUDA-accelerated variant that significantly reduces end-to-end execution time while preserving the control performance of the baseline formulation. Simulation results on an agile drone racing task show that our approach achieves state-of-the-art lap times and near-limit dynamic behaviour with markedly reduced training and inference time.
>
---
#### [new 013] EXACT-MPPI: Exact Signed-Distance Navigation for Arbitrary-Footprint Robots from Point Clouds via Path Integral Control
- **分类: cs.RO**

- **简介: 该论文属于机器人局部导航任务，解决复杂足型机器人在密集环境中的安全导航问题。通过EXACT-MPPI框架，直接从点云生成运动指令，无需中间地图，提升导航精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.29663](https://arxiv.org/pdf/2605.29663)**

> **作者:** Chen Peng; Zhikang Ge; Wenwu Lu; Haiming Gao; Stavros Vougioukas; Peng Wei
>
> **摘要:** Ground robots often carry payloads, implements, or other attachments that turn their effective footprint into complex, non-convex shapes. Navigating safely through clutter then requires reasoning about this true geometry, yet most local planners simplify it with convex or inflated proxies and rasterize sensor data into occupancy grids or distance fields. Both choices eliminate feasible motions when clearance is comparable to the footprint geometry. We present EXACT-MPPI, a training-free local navigation framework that maps local point-cloud observations and sparse guidance directly to motion commands, without any intermediate map representation. The framework embeds an analytic, exact signed-distance evaluator into a Model Predictive Path Integral (MPPI) controller. The footprint is represented as a simple polygon for general convex or concave planar shapes, with a rectangle-cover specialization for faster evaluation of rectilinear footprints, enabling footprint-aware collision costs without convex decomposition, inflation, or learned encoders. During each MPPI rollout, observed obstacle points are transformed into the predicted body frame and evaluated against the footprint. All operations are batched in JAX, leveraging GPU parallelism for real-time receding-horizon control. Experiments show that EXACT-MPPI accelerates batched distance evaluation over a learned point-to-robot baseline, preserves feasible motion where convex-footprint planners fail, and remains robust under dense static and moving obstacles. The same framework deploys on differential-drive, Ackermann, omnidirectional, and hybrid-mode platforms by changing only the footprint description and motion model without per-platform training. Pairing exact footprint geometry with sampling-based predictive control thus offers a practical, training-free path to footprint-aware local navigation across diverse robots.
>
---
#### [new 014] FLIP: Real-Time and Resilient Formation Planning for Large-Scale DIstributed Swarms via Point Cloud Registration
- **分类: cs.RO**

- **简介: 该论文属于多智能体协同轨迹规划任务，解决大规模群体编队规划中的计算负载高和性能差问题。通过将编队位置序列计算转化为点云配准问题，实现高效、鲁棒的分布式规划。**

- **链接: [https://arxiv.org/pdf/2605.29704](https://arxiv.org/pdf/2605.29704)**

> **作者:** Yuan Zhou; Guangtong Xu; Zhenyu Hou; Jialiang Hou; Fei Gao
>
> **摘要:** Traditional large-scale formation planning either oversimplify the formation representation which leads to poor performance, or they employ complete collaborative relationships, which results in excessive computational load. To achieve high-performance and large-scale formation planning, we transform the Optimal Formation Position Sequence \cite{c1} (OFPS) calculation problem into a spatiotemporal Point Cloud Registration (PCR) problem. Each agent derives its OFPS by distributively computing the matching result between current positions and the desired formation positions of all other agents. Then each agent optimizes the cooperative formation trajectory by using OFPS. We leverage the PCR method with outlier rejection to rapidly perform large-scale formation position registration. This prevents suboptimal trajectories and failed agents from propagating through the cooperative network and affecting more agents. Consequently, we uniformly achieve resilient, efficient, and distributed trajectory planning for large-scale swarms. The effectiveness and the superiority of the proposed method are demonstrated through large-scale simulations of 120-drone formation, and rigorous benchmarking against state-of-the-art (SOTA) methods.
>
---
#### [new 015] Replicable Simulation-Based Robot Validation through Provenance
- **分类: cs.RO**

- **简介: 该论文属于机器人验证任务，解决仿真测试可复现性问题，通过引入数据溯源和FAIR元数据提升测试过程透明度与可追溯性。**

- **链接: [https://arxiv.org/pdf/2605.29973](https://arxiv.org/pdf/2605.29973)**

> **作者:** Argentina Ortega; Samuel Wiest; Frederik Pasch; Nico Hochgeschwender
>
> **摘要:** Robot behavior is often validated through simulation-based testing, yet the replicability of such campaigns depends critically on transparent documentation of how tests are configured, executed, and post-processed. We argue that data provenance, coupled with the FAIR principles (findability, accessibility, interoperability, and reusability), addresses this gap by explicitly tracking links between artifacts and by attaching machine-readable metadata about file origins and key design decisions. Moreover, provenance and metadata cannot be treated as an afterthought confined to final datasets; they must be integrated into the testing processes that generate those datasets so that evidence can be reconstructed end-to-end. We demonstrate this by augmenting an existing simulation-based testing framework with provenance tracking and metadata collection mechanisms, and by using these extensions to enrich a mobile robot navigation dataset with structured provenance and FAIR-aligned metadata. Finally, we discuss obstacles encountered in this integration -- such as vocabulary alignment, attribute selection, and adoption of domain standards -- and provide actionable recommendations for implementing provenance-centric, FAIR metadata in robotics validation workflows.
>
---
#### [new 016] A Progress-Aware Leader-Follower Midair Docking System for Dual-Drone Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文属于无人机协同任务，解决双机空中对接问题。提出一种进度感知的领导-跟随对接系统，实现可靠、重复的空中操作。**

- **链接: [https://arxiv.org/pdf/2605.29410](https://arxiv.org/pdf/2605.29410)**

> **作者:** Yifan Cai; Jan Ming Kevin Tan; Xiangqi Li; Chenzhe Jin; Narsimlu Kemsaram; Valerio Modugno
>
> **备注:** This paper has been accepted for publication in the Proceedings of the 2026 IEEE 22nd International Conference on Automation Science and Engineering (CASE 2026), August 17-21, 2026, Shenyang, China
>
> **摘要:** Reliable midair docking between small unmanned aerial vehicles (UAVs) is essential for modular aerial cooperation and manipulation, but it requires precise relative-pose control and repeatable platform under tight thrust and payload constraints. We present a dual-drone docking platform where two quadrotors operate in a leader-follower formation and dock using a lightweight modular frame with passive magnetic latching. A progress-aware mission supervisor manages phase transitions: approach, alignment, capture, and settle. This platform integrates a complete hardware-software stack (ROS 2 with Crazyflie/PX4 interfaces) and synchronized logging for benchmark evaluation. We evaluate the platform in simulation and real-world experiments using quantitative metrics such as formation error, baseline and yaw consistency, docking success rate, time-to-dock, and failure-mode statistics. The platform enables statistically grounded comparison of docking supervision and synchronization strategies and provides a practical testbed for modular aerial cooperation and repeatable midair aerial manipulation.
>
---
#### [new 017] Learning and Adaptation in Wire Arc Additive Manufacturing Bead Geometry Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于焊接控制任务，旨在解决WAAM过程中焊缝形状不一致的问题。通过数据驱动模型和自适应控制，提升焊缝高度与宽度的稳定性。**

- **链接: [https://arxiv.org/pdf/2605.29144](https://arxiv.org/pdf/2605.29144)**

> **作者:** Chen-Lung Lu; John Wen
>
> **摘要:** Robotics Wire Arc Additive Manufacturing (WAAM) is governed by complex and nonlinear process dynamics coupling thermal field to the build geometry. The process may be regarded as a multi-input/multi-output dynamical system with welding torch speed and wire feed rate as inputs and weld bead deposition height and width as outputs. In this paper, we use the input/output data to learn a data-driven model and use it for weld planning and control. We show that a simple recurrent neural network architecture and one-step-ahead predictive control can improve the process performance in terms of height and width consistency. To account for the changing thermal conditions during the printing process, we update the learning model using prediction error from the previous layer. This adaptation step further improves the prediction accuracy and controller performance. Experiments on a robotic WAAM testbed with integrated line-scanner feedback significant improvements in height and width consistency compared to constant input and static model baselines. The proposed learning and adaptation framework provides a practical pathway toward robust, data-driven regulation of additive manufacturing processes.
>
---
#### [new 018] A Heterogeneous Architecture for Robot RL Beyond GPU-Dominant Paradigms
- **分类: cs.RO**

- **简介: 该论文属于机器人强化学习任务，旨在提升仿真训练效率。针对GPU依赖过强的问题，提出UniLab架构，实现CPU与GPU协同，提升训练效率并支持多平台运行。**

- **链接: [https://arxiv.org/pdf/2605.30313](https://arxiv.org/pdf/2605.30313)**

> **作者:** Yufei Jia; Zhanxiang Cao; Mingrui Yu; Heng Zhang; Shenyu Chen; Dixuan Jiang; Meng Li; Xiaofan Li; Yiyang Liu; Junzhe Wu; Zheng Li; XiLin Fang; Tingyu Cui; Shengcheng Fu; Haoyang Li; Anqi Wang; Zifan Wang; Dongjie Zhu; Chenyu Cao; Zhenbiao Huang; Ziang Zheng; Jie Lu; Xin Ma; Zhengyang Wei; Xiang Zhao; Tianyue Zhan; Ye He; Yuxiang Chen; Yizhou Jiang; Yue Li; Haizhou Ge; Yuhang Dong; Fan Jia; Ziheng Zhang; Meng Zhang; Xiwa Deng; Zhixing Chen; Hanyang Shao; Chenxin Dong; Yixuan Li; Yizhi Chen; Bokui Chen; Kaifeng Zhang; Hanqing Cui; Yusen Qin; Ruqi Huang; Lei Han; Tiancai Wang; Xiang Li; Yue Gao; Guyue Zhou
>
> **摘要:** Simulation-based RL for contemporary robot control is increasingly organized around GPU-resident simulation: physics, rollout collection, and learning are placed on a single GPU-centric execution path. This paradigm has greatly improved training speed, but it has also encouraged a default assumption that efficient training requires physics to reside on the GPU. We revisit this assumption. Our view is that, in simulation-dominated robot control, the essential question is not which processor runs physics, but whether simulation throughput, policy learning, and runtime synchronization form an efficient end-to-end loop. We present UniLab, a heterogeneous CPU-simulation / GPU-learning architecture that decouples CPU-parallel simulation from GPU policy updates through a unified runtime for data movement, buffering, and synchronization. UniLab is implemented as a complete and extensible training system using MuJoCoUni and MotrixSim CPU-batched physics backends, supporting PPO, SAC, FlashSAC, TD3, and APPO. On representative simulation-based robot control tasks, UniLab improves end-to-end training efficiency by 3--10$\times$ under the same hardware configuration, while reducing dependence on the NVIDIA CUDA-based software stack and supporting cross-platform execution on the Apple macOS platform and the AMD ROCm and Intel XPU accelerator backends. These results show that GPU simulation is an effective path to efficient training, but not a necessary one, broadening the practical system choices available for robot RL training. Project page: this https URL.
>
---
#### [new 019] MARS Policy: Multimodality Only When It Matters
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决多模态策略训练复杂和推理效率低的问题。提出MARS策略，在需要时引入噪声，提升效率与成功率。**

- **链接: [https://arxiv.org/pdf/2605.29766](https://arxiv.org/pdf/2605.29766)**

> **作者:** Jindou Jia; Tuo An; Yuxuan Hu; Gen Li; Jingliang Li; Bohan Hou; Xiangyu Chen; Jiaqi Bai; Bofan Lyu; Jianfei Yang
>
> **备注:** 13 figures, 17 pages
>
> **摘要:** Imitation learning has become a cornerstone for solving complex robotic manipulation tasks. In particular, multimodality, which enables robots to capture diverse yet valid behavioral patterns, has driven the rapid emergence of generative policies as a dominant paradigm in robot learning. However, achieving such multimodality typically relies on stochastic noise initialization and iterative denoising procedures, resulting in substantial training complexity and low inference efficiency. Meanwhile, not all phases of a robotic task inherently require behavioral diversity. Motivated by this insight, we propose the Modality-Adaptive Robot Sampling (MARS) policy, which adaptively invokes tailored stochasticity only when it is truly beneficial, while reverting to an efficient deterministic learning during single-modal phases. In other words, the proper amount of noise is injected only at the proper time. By selectively activating multimodal generation, MARS policy bridges the gap between the multimodal capability of generative policies and the superior training and inference efficiency of deterministic models. Empirical studies across 8 simulated and 4 real-world tasks demonstrate that MARS exhibits robust multimodal expressivity and high efficiency, with a 16.67% success rate improvement and an 83.20% inference latency reduction in real-world tests. Counterintuitively, MARS also outpaces deterministic policies in training efficiency on near-deterministic tasks by more effectively modeling nuanced action diversity.
>
---
#### [new 020] Extreme dynamic symmetry enables omnidirectional and multifunctional robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人设计任务，旨在提升机器人的敏捷性与鲁棒性。通过引入动态对称性，优化机器人运动性能，实现多任务适应能力。**

- **链接: [https://arxiv.org/pdf/2605.29254](https://arxiv.org/pdf/2605.29254)**

> **作者:** Jiaxun Liu; Boxi Xia; Boyuan Chen
>
> **备注:** Published in Science Robotics (2026). Our project website is at:this https URL
>
> **摘要:** Symmetry is a central organizing principle in natural systems, yet its use as a unifying design strategy in robotics has largely remained limited to geometric form. We show that symmetry can instead be leveraged at the level of dynamic actuation capability. We introduce dynamic symmetry, the uniformity of a robot's attainable center-of-mass accelerations, and formalize it through a measure coined as dynamic isotropy. Across more than 1000 simulated morphologies, we found that higher dynamic symmetry consistently improved trajectory tracking, task success, robustness, resiliency, and energy efficiency, with the benefits becoming most pronounced as dynamic isotropy approached its theoretical limit. To study this regime systematically, we developed Argus, a family of spherical robots designed to explore the effects of increasing dynamic symmetry. Members of the Argus family vary in their actuation geometry and dynamic symmetry level while sharing a common architectural principle: radially oriented linear actuators that directly shape the robot's center-of-mass dynamics. Among them, we built a physical 20-leg Argus variant that achieved near-extreme dynamic isotropy and demonstrated orientation-invariant locomotion, agile traversal of cluttered and deformable terrain, rapid self-stabilization, and resilience to partial actuator failures. Its distributed sensing further enabled omnidirectional perception and object interaction during continuous motion. These results show that designing robots for symmetry not only in morphology but also in their attainable dynamics provides a powerful and general pathway toward agility, robustness, and multifunctionality in uncertain terrestrial and extraterrestrial environments.
>
---
#### [new 021] Fisher-Preserving Guidance: Training-Free Manifold Constraints for Safe Diffusion Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对视觉导航中的轨迹预测任务，解决扩散模型在分布外动作时产生的不可靠问题。提出一种无需训练的约束方法，通过低秩雅可比分解实现高效更新，并引入不确定性信号提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.29937](https://arxiv.org/pdf/2605.29937)**

> **作者:** Hao Ren; Zetong Bi; Yiming Zeng; Le Zheng; Zhi Li; Zhaoliang Wan; Lu Qi; Hui Cheng
>
> **备注:** ICML2026
>
> **摘要:** Diffusion models are effective for waypoint prediction in visual navigation, but standard sampling and test time guidance can produce unreliable or inefficient trajectories when updates drift off the training manifold. We propose Fisher Preserving Guidance with Outer Product Span Projection, a training-free inference method that avoids large Fisher drift associated with off-distribution actions while optimizing a task objective. Our method computes the Fisher-preserving update via a low-rank Jacobian factorization, requiring only a single backward pass per step and enabling real-time use. We further introduce Truncated Fisher Denoising Sensitivity as an uncertainty signal and use it for robust multi-sample action blending. Experiments on toy and realistic navigation benchmarks, including Maze2D with TSDF-based guidance, PushT with official Diffusion Policy weights, and visual navigation in simulation and on real robots, demonstrate consistent improvements in performance over strong diffusion-policy baselines without additional training.
>
---
#### [new 022] BORA: Bridging Offline Reinforcement Learning and Online Residual Adaptation for Real-World Dexterous VLA Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出BORA框架，解决真实世界中VLA模型的精细操作问题，通过离线到在线强化学习提升执行可靠性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.30226](https://arxiv.org/pdf/2605.30226)**

> **作者:** Zhongxi Chen; Yifan Han; Yanming Shao; Huanming Liu; Congsheng Xu; Xiaoyu Chen; Yao Mu; Wenzhao Lian
>
> **备注:** 24 pages,11 figures
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for grounding visual-language understanding into real-world robotic manipulation. However, dexterous manipulation remains challenging for VLA policies due to high-dimensional hand control and compounding execution errors, which makes real-world RL post-training essential for bridging the gap between visually grounded action generation and physically reliable dexterous execution. However, high-dimensional dexterous exploration often triggers temporal inconsistency, sample inefficiency and hardware risks in the real world. To address these challenges, we propose BORA, an offline-to-online RL post-training framework designed for real-world dexterous VLA models. In the offline phase, BORA constructs a critic that takes both the VLM's cognition tokens and action chunks as inputs. This design enables action-conditioned value guidance, allowing the critic to evaluate dexterous hand motions beyond visual context alone. During the subsequent online phase, BORA freezes the VLA base and introduces a lightweight, Human-in-the-Loop (HiL) chunk-wise residual adaptation mechanism to mitigate real-world execution errors and further correct the offline-learned intents within the actual physical environment. By inheriting the offline critic and employing intervention-driven rewards, BORA effectively corrects execution discrepancies and adapts to real-world physical variances while preserving the pretrained policy as a stable prior. Extensive evaluations across five complex real-world dexterous tasks demonstrate that BORA significantly outperforms pure imitation learning and traditional decoupled RL baselines, achieving a 33% absolute increase in average success rate under standard settings and up to a 43% improvement in unseen object generalization.
>
---
#### [new 023] Sample-Efficient Diffusion-based Reinforcement Learning with Critic Guidance
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决扩散策略在探索与利用间的平衡问题。提出CGPO方法，通过批评者引导提升策略性能和收敛速度。**

- **链接: [https://arxiv.org/pdf/2605.30056](https://arxiv.org/pdf/2605.30056)**

> **作者:** Shutong Ding; Zejia Zhong; Zhongyi Wang; Ke Hu; Bikang Pan; Jingya Wang; Ye Shi
>
> **备注:** accepted by ICML2026
>
> **摘要:** Recent advances in reinforcement learning (RL) have achieved great successes by leveraging the multimodality and exploration capability of diffusion policies. Among these approaches, one representative branch focuses on the sampling-based policy optimization. This design enables better exploration capability of the diffusion model, particularly at the beginning of training, but suffer from low exploitation in Q-value information, resulting in a slow policy convergence. Another branch pays attention to gradient-based policy optimization, which sufficiently exploits the gradient of the Q function yet tends to collapse into a unimodal policy with low diversity. To address this issue, we propose CGPO, \textbf{C}ritic-\textbf{G}uided diffusion \textbf{P}olicy \textbf{O}ptimization, which effectively balances exploration and exploitation with the training-free guidance technique integrated into the denoising process of diffusion policy. Concretely, CGPO steers action generation toward high-value regions defined by the critic network and uses the guided actions as regression objectives. In this manner, CGPO reduces the time required to obtain high-quality actions and improves final performance with better balance between the exploration-exploitation tradeoff. We validate the effectiveness of CGPO on 5 MuJoCo locomotion tasks, and CGPO achieves state-of-the-art performance compared with existing diffusion-based RL methods. Notably, CGPO is the first success to incorporate diffusion policy into real-world RL, with its superior performance on Franka robot arm grasping tasks. Our official page is released at this https URL.
>
---
#### [new 024] Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文提出Qwen-VLA，一个统一的视觉-语言-动作模型，解决多任务、多环境、多机器人形态的具身智能问题，通过联合预训练和提示条件实现跨任务迁移。**

- **链接: [https://arxiv.org/pdf/2605.30280](https://arxiv.org/pdf/2605.30280)**

> **作者:** Qiuyue Wang; Mingsheng Li; Jian Guan; Jinhui Ye; Sicheng Xie; Yitao Liu; Junhao Chen; Zhixuan Liang; Jie Zhang; Xintong Hu; Xuhong Huang; Pei Lin; Junyang Lin; Dayiheng Liu; Shuai Bai; Jingren Zhou; Jiazhao Zhang; Haoqi Yuan; Gengze Zhou; Hang Yin; Ye Wang; Yiyang Huang; Zixing Lei; Wujian Peng; Delin Chen; Yingming Zheng; Jingyang Fan; Xianwei Zhuang; Xin Zhou; Haoyang Li; Anzhe Chen; Tong Zhang; Xuejing Liu; Yuchong Sun; Ruizhe Chen; Zhaohai Li; Chenxu Lü; Zhibo Yang; Tao Yu; Xionghui Chen
>
> **备注:** 34 pages
>
> **摘要:** Embodied intelligence is often studied through specialized models for individual tasks such as manipulation or navigation, resulting in fragmented capabilities and limited generalization across tasks, environments, and robot embodiments. In this work, we study whether heterogeneous embodied decision-making problems can be unified within a single vision-language-action model. We present Qwen-VLA, a unified embodied foundation model that extends Qwen's vision-language modeling stack from perception, understanding, and reasoning to continuous action and trajectory generation through a DiT-based action decoder. Qwen-VLA is trained with a large-scale joint pretraining recipe over diverse data sources, including robotics manipulation trajectories, human egocentric demonstrations, synthetic simulation data, vision-and-language navigation data, trajectory-centric supervision, and auxiliary vision-language data. To support multiple robot platforms, we introduce embodiment-aware prompt conditioning, where robot-specific textual descriptions specify the current embodiment and control convention. We further cast manipulation, navigation, and trajectory prediction into a unified action-and-trajectory prediction framework, enabling transferable visual grounding, spatial reasoning, and continuous action generation across robot morphologies, task families, and environments. Experiments on manipulation, navigation, and trajectory-centric benchmarks show consistent multi-task performance and out-of-distribution generalization under variations in scene layout, background, lighting, object configuration, and robot embodiment. Qwen-VLA-Instruct achieves 97.9% on LIBERO, 73.7% on Simpler-WidowX, 86.1%/87.2% on RoboTwin-Easy/Hard, 69.0% OSR on R2R, 59.6% SR on RxR, 76.9% average OOD success in real-world ALOHA experiments, and 26.6% zero-shot success on DOMINO dynamic manipulation.
>
---
#### [new 025] PhAIL: A Real-Robot VLA Benchmark and Distributional Methodology
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作策略评估任务，解决真实机器人评估方法不足的问题。提出PhAIL基准和分布评估方法，使用时间到成功CDF进行更精确的性能比较。**

- **链接: [https://arxiv.org/pdf/2605.29710](https://arxiv.org/pdf/2605.29710)**

> **作者:** Sergey Arkhangelskiy
>
> **备注:** 22 pages, 10 figures, 8 tables. Dataset, analysis pipeline, and paper source: this https URL and this https URL
>
> **摘要:** Real-world evaluation of vision-language-action (VLA) policies still rests on binary success rate at a fixed timeout with $N \le 25$ rollouts per condition, almost always without confidence intervals or paired statistical comparison; these cohort sizes struggle to resolve close comparisons reliably. We introduce PhAIL (Physical AI Leaderboard, this https URL), an open real-robot benchmark on a Franka FR3 (dataset, per-rollout artifacts, and end-to-end reference implementation) of a distributional evaluation methodology: the time-to-success cumulative distribution function (CDF) as the evaluation primitive, with two separated jobs. The first is scoring via Human-Relative Throughput (HRT), a dimensionless scalar with bootstrap confidence intervals, anchored to same-fixture human teleoperation. The second is a significance test (Kolmogorov-Smirnov, computed per-object and macro-averaged across objects). On four publicly-available VLAs, the macro-averaged KS test resolves two close comparisons (GR00T vs. ACT, OpenPI vs. ACT) at $N \le 30$ rollouts per (model, object) cell where binary-threshold metrics do not; the closest pair (OpenPI vs. GR00T) remains unresolved within our budget. The best evaluated VLA is $\sim 7\times$ slower per operation (RMST ratio) than the human reference.
>
---
#### [new 026] Joint Angle Estimation with Customized Wristband Based on Online Incremental Learning
- **分类: cs.RO**

- **简介: 该论文属于运动姿态估计任务，旨在解决可穿戴传感器在不同情境下的适应性问题。通过在线增量学习方法，提升手腕角度估计的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.29771](https://arxiv.org/pdf/2605.29771)**

> **作者:** Shuo Wang; Xiaobin Chen; Xiaoming Tao
>
> **摘要:** Intelligent wearable technology plays an increasingly important role in human-computer interaction, motion, and health monitoring. To ensure comfort and practicality of use, one common form for motion monitoring is to utilize soft wearable sensors. However, many research applications regarding wearable sensors are simplistic and difficult to adapt to different situations. This study proposes a system for estimating the angle of the wrist joint using a customized wristband based on an online incremental learning approach. It is a two-stage estimation method: the first stage updates the model based on the wearer's wrist movement characteristics using online learning, integrating real-time data from an IMU as ground truth. The second stage utilizes the updated model for estimation of wrist joint angle solely with the wristband. In other words, model training is completed during data acquisition, allowing the trained model to be used for subsequent angle estimation. This method offers advantages in adapting to data drift caused by variations in different testing configurations, such as the left and right wrists of the same subject, deviations in the wearing position on the same wrist, and even differences among various subjects. The results indicate that the sensors exhibit good performance under strain variations, and the wrist joint trajectory estimation of the proposed system has an approximate error of 15 degree in different scenarios.
>
---
#### [new 027] MonoDuo: Using One Robot Arm to Learn Bimanual Policies
- **分类: cs.RO**

- **简介: 该论文提出MonoDuo框架，解决单臂机器人学习双臂操作策略的问题。通过人类协作收集数据，生成合成演示，提升双臂机器人任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.29298](https://arxiv.org/pdf/2605.29298)**

> **作者:** Sandeep Bajamahal; Lawrence Yunliang Chen; Toru Lin; Zehan Ma; Jitendra Malik; Ken Goldberg
>
> **备注:** Accepted to appear in the 2026 IEEE International Conference on Robotics and Automation (ICRA), Vienna, Austria, 1-5 June 2026
>
> **摘要:** Bimanual coordination is essential for many real-world manipulation tasks, yet learning bimanual robot policies is limited by the scarcity of bimanual robots and datasets. Single-arm robots, however, are widely available in research labs. Can we leverage them to train bimanual robot policies? We present MonoDuo, a framework for learning bimanual manipulation policies using single-arm robot demonstrations paired with human collaboration. MonoDuo collects data by teleoperating a single-arm robot to perform one side of a bimanual task while a human performs the other, then swapping roles to cover both sides. RGB-D observations from a wrist-mounted and fixed camera are augmented into synthetic demonstrations for target bimanual robots using state-of-the-art hand pose estimation, image and point cloud segmentation, and inpainting. These synthetic demonstrations, grounded in real robot kinematics, are used to train bimanual policies. We evaluate MonoDuo on five tasks: box lifting, backpack packing, cloth folding, jacket zipping, and plate handover. Compared to approaches relying solely on human bimanual videos, MonoDuo enables zero-shot deployment on unseen bimanual robot configurations, achieving success rates up to 70%. With only 25 target robot demonstrations, few-shot finetuning further boosts success rates by 65-70% over training from scratch, demonstrating MonoDuo's effectiveness in efficiently transferring knowledge from single-arm robot data to bimanual robot policies.
>
---
#### [new 028] LLM-Guided Future Hypotheses for Horizon-Aware Exploration in Multi-Step Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于多步机器人操作任务，旨在解决在不确定性下探索与策略适应的问题。通过生成短期未来视频作为结构化先验，提升控制与强化学习效果。**

- **链接: [https://arxiv.org/pdf/2605.29864](https://arxiv.org/pdf/2605.29864)**

> **作者:** Mohammad Khoshnazar; Andrew Melnik; Michael Beetz
>
> **摘要:** Multi-step robot manipulation requires acting under uncertainty about how the scene will evolve, making exploration and policy adaptation challenging. We study whether short-horizon, task-consistent future videos can provide useful structured priors for control and reinforcement-learning fine-tuning. We formalize this idea through Future-Experience Conditioning (FEC), a simple interface that conditions closed-loop policies on a latent representation of a short future video. In our simulation setup, future clips are generated in three stages, an LLM reasoner operating over a task ontology initialized from the current scene state, a robot-free digital-twin rollout of the intended object motion, and a mask-free video diffusion model that synthesizes a robot-consistent future clip without requiring segmentation at inference. We instantiate this future-conditioning interface primarily with BC and BC+RL, and compare against a future-conditioned Streaming Flow Policy (SFP) baseline on RoboCasa and CALVIN under NoFuture, GTFuture, GenFuture, and WrongFuture. Generated futures improve performance over no-future conditioning, while mismatched futures degrade it, and our BC+RL instantiation achieves the strongest overall results. An average BC+RL learning-curve analysis across 8 CALVIN tasks further shows that GTFuture improves fastest, GenFuture improves earlier and to a higher level than NoFuture, and WrongFuture remains at zero throughout training. These results suggest that short-horizon future videos can serve as useful structured priors for exploration and policy adaptation under imperfect future predictions. this https URL
>
---
#### [new 029] DynaFLIP: Rethinking Robotics Perception via Tri-Modal-Dynamics Guided Representation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出DynaFLIP，解决机器人感知中运动理解不足的问题。通过多模态预训练，提升视觉表示对操作相关动态的捕捉能力，增强机器人泛化性能。**

- **链接: [https://arxiv.org/pdf/2605.30350](https://arxiv.org/pdf/2605.30350)**

> **作者:** Jusuk Lee; Seungjae Lee; Jonghun Shin; Hoseong Jung; Sungha Kim; Daesol Cho; H. Jin Kim; Jia-Bin Huang; Furong Huang
>
> **备注:** Project website: this https URL
>
> **摘要:** Robot manipulation critically depends on perception that preserves the action-relevant aspects of a scene. Yet most robot learning pipelines are built upon visual encoders pre-trained for static recognition or vision-language alignment, leaving motion understanding to downstream policies. We introduce DynaFLIP, a dynamics-aware multimodal pre-training framework that pushes motion understanding upstream into perception. We construct image-language-3D flow triplets from heterogeneous human and robot videos, and use these triplets as training-time supervision to shape an image-only encoder. Our key idea is to encourage the three modalities to span a small simplex volume in the shared hyperspherical space -- a smaller simplex volume indicating stronger alignment. To avoid the geometric ambiguity and trivial collapse of naive volume minimization, we combine simplex-volume minimization with a cosine regularizer and a contrastive objective. Our analyses show that DynaFLIP focuses on control-relevant regions critical for manipulation. The resulting dynamics-aware representations serve as reusable visual backbones and consistently outperform baselines across diverse downstream policies, including VLAs. We validate this across diverse simulation and real-world setups, with gains reaching +22.5% under out-of-distribution scenarios. Our results suggest that robot generalization improves when visual representations are trained to encode not just what is present, but how the world changes under action.
>
---
#### [new 030] Phase-Conditioned Imitation Learning with Autonomous Failure Recovery for Robust Deformable Object Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决变形物体操作中的状态混淆和失败恢复问题。提出一种分阶段的模仿学习框架，实现鲁棒操作。**

- **链接: [https://arxiv.org/pdf/2605.29407](https://arxiv.org/pdf/2605.29407)**

> **作者:** Dayuan Chen; Kai Tang; Yukuan Zhang; Kazuhiro Kosuge; Yasuhisa Hirata
>
> **备注:** Accepted to IEEE/ASME Transactions on Mechatronics
>
> **摘要:** This paper presents a phase-conditioned, force-aware framework for robust deformable object manipulation. Standard imitation learning policies such as Action Chunking with Transformers (ACT) rely on a Markovian assumption at inference, causing state aliasing when visually similar observations require contradictory actions and preventing autonomous recovery from execution failures. We address this with a closed-loop hierarchical architecture. A FiLM-conditioned ACT encoder modulates feature extraction based on the current task phase, enabling a single unified policy to produce phase-specific behaviors while sharing action dynamics across phases. A multi-modal phase predictor fusing visual, force, and pose feedback estimates the phase in real time, detecting contact failures that are invisible to vision alone and autonomously triggering recovery trajectories. The system is completed by a hybrid impedance controller for compliant execution and a haptic teleoperation interface for force-aware data collection. Ablation studies show that FiLM-based modulation significantly outperforms both unconditioned and token-level conditioned baselines, and t-SNE analysis confirms that FiLM induces well-separated, phase-specific feature representations. Validated on hanging and removing a T-shirt with dual arms, the closed-loop system improves the hanging success rate from 56\% to 87\% through autonomous error recovery. Code and videos: this https URL
>
---
#### [new 031] 3DVLA: Enhancing Vision-Language-Action Models via 3D Spatial and Instance Understanding
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决3D场景理解不足的问题。通过引入3D特征编码、实例估计和遮挡处理机制，提升模型的3D推理能力。**

- **链接: [https://arxiv.org/pdf/2605.29416](https://arxiv.org/pdf/2605.29416)**

> **作者:** Zhongyu Xia; Yousen Tang; Bingqing Wei; Yongtao Wang
>
> **摘要:** Vision-Language-Action models have achieved remarkable progress in robotic manipulation, yet they suffer from a critical limitation: a lack of 3D scene understanding. This deficiency manifests as three intertwined challenges: weak extraction of 3D spatial positions without enforcing multi-view consistency, inadequate 3D instance understanding, and fragile reasoning under occlusion. Although mature 3D perception methods exist, their direct integration into VLA pipelines is hindered by architectural incompatibility and by heavy reliance on costly instance-level annotations. To address the above challenges, we propose 3DVLA, a plug-and-play framework that injects robust 3D reasoning into pretrained VLAs without requiring extra manual labels or discarding VLM priors. Specifically, 3DVLA tackles the three challenges through: (1) pervasive 3D feature encoding with explicit multi-view consistency constraints across all modalities and a Spatially-Conditioned Geometry Aggregation method, (2) an instance estimation module with high-level instance tokens for 3D instance awareness, and (3) a masked self-supervised 3D encoding branch that retains its predictor for visual token completion to handle occlusions. We integrate 3DVLA with multiple VLA baselines and evaluate on LIBERO-Plus and RoboTwin 2.0. Results show consistent and significant gains in manipulation performance, validating both the effectiveness and plug-and-play compatibility of our approach.
>
---
#### [new 032] Human-in-the-Loop Swarms: A Bionic Swarm Approach to Real-World Soil Mapping
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出“仿生群体”系统，解决 swarm 机器人实测成本高的问题。通过人类参与降低开发难度，验证了基于评分的搜索算法在真实环境中的有效性。**

- **链接: [https://arxiv.org/pdf/2605.29091](https://arxiv.org/pdf/2605.29091)**

> **作者:** Petras Swissler; Mohammadali Rashidioun; Nicholas Sahu; Raaid Kabir; Ayodeji Aderibigbe; Oladoyin Kolawole
>
> **备注:** 27 pages, 15 figures. Submitted to Advanced Intelligent Systems
>
> **摘要:** Swarm and field robotics face significant barriers to real-world validation due to the high cost and development time to deploy hardware. This paper introduces the ``Bionic Swarm,'' a novel system that lowers these barriers by abstracting away many of the tasks that are difficult to implement on robots but which do not contribute to the overall algorithm evaluation, giving these tasks to human users. These human users take directions from a smartphone web-app that takes measurements from Bluetooth-connected sensors and relays them to a centralized server. This server runs the swarm algorithm and directs actions to the human users. We evaluate this system through the experimental validation of a geotechnically-focused search algorithm named Score-Biased-Search, which functions by assigning a ``score'' to each location on a reconstructed map, then biases search patterns through areas of higher expected scores, and which exhibits superlinear map reconstruction relative to the number of search agents. After presenting simulation results for the algorithm, we then apply the algorithm on the Bionic Swarm platform to validate its function in a real-world, outdoor setting. This work demonstrates that this human-in-the-loop approach significantly lowers the barrier to entry for field and swarm robotics research.
>
---
#### [new 033] Energy-Aware NECO for Single-Pass Pixel-wise Out-of-Distribution Detection in Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于语义分割任务，解决分布外检测问题。提出Energy-Aware NECO方法，在单次推理中实现像素级分布外检测，提升检测性能与效率。**

- **链接: [https://arxiv.org/pdf/2605.29773](https://arxiv.org/pdf/2605.29773)**

> **作者:** Boyuan Zhang; Huanshan Huang; Yifei Cao
>
> **备注:** 7 pages, 6 figures. Accepted at the ICRA 2026 Workshop on Long-term Deployments in the Wild (LoWi 2026)
>
> **摘要:** Reliable semantic segmentation for mobile robots requires both accurate dense prediction and robust uncertainty estimation under distribution shift. Strong uncertainty baselines such as Monte Carlo Dropout often require repeated stochastic forward passes and are difficult to deploy on edge platforms. We propose Energy-Aware NECO, a single-pass pixel-wise out-of-distribution (OOD) detector for semantic segmentation. The method combines a centered NECO-style geometric ratio computed from decoder features with a logit-based Energy score. Both components are standardized using statistics fitted on a pure in-distribution validation split and fused through a convex combination. We evaluate the method on the miniMUAD subset using true pixel-level OOD labels. The proposed hybrid score achieves an AUROC of 0.8539, outperforming NECO-only (0.8280), Energy-only (0.8171), and an ensemble predictive-entropy baseline (0.8124). Additional qualitative and operating-point analyses show that the hybrid detector improves overall ranking performance while preserving the efficiency advantages of a single-pass design. Code is available at this https URL
>
---
#### [new 034] DGSG-Mind: Dynamic 3D Gaussian Scene Graphs for Long-Term Scene Understanding and Grounding
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DGSG-Mind，用于长期场景理解与定位任务，解决动态3D场景中语义信息整合和实例关联问题，通过混合高斯图结构实现高效场景建模与更新。**

- **链接: [https://arxiv.org/pdf/2605.29879](https://arxiv.org/pdf/2605.29879)**

> **作者:** Luzhou Ge; Xiangyu Zhu; Jinyan Liu; Xuesong Li
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Integrating open-vocabulary semantic information into dynamic 3D scene representations is essential for long-term embodied scene understanding. However, existing methods often suffer from fragile instance association due to incomplete cross-view cues, while their limited ability to handle object-level topological changes restricts long-term robotic task execution. Moreover, current 3D scene understanding methods either rely on simple feature matching without explicit spatial reasoning or assume offline ground-truth 3D geometry. To address these challenges, we present DGSG-Mind, a hybrid instance-aware 3D Gaussian dynamic scene graph system with an embodied reasoning agent. Our system couples a probabilistic voxel grid with explicit 3D Gaussians to enable robust cross-modal instance fusion and incremental semantic mapping. It handles dynamic changes through Gaussian-based visual relocalization and localized masked refinement guided by geometric-semantic consistency. Built on the instance Gaussian map, DGSG-Mind further constructs a hierarchical scene graph and develops the 3D Gaussian Mind, which integrates structural relations, spatial-semantic information, and visually annotated RoI Gaussian renderings for multimodal reasoning. Extensive experiments show that DGSG-Mind achieves the best zero-shot 3DVG performance among methods operating on self-reconstructed maps, while also delivering strong performance in 3D open-vocabulary semantic segmentation and scene reconstruction. We further deploy DGSG-Mind on real-world robots to demonstrate its target-oriented reasoning and dynamic update capabilities. The project page of DGSG-Mind is available at this https URL
>
---
#### [new 035] ReasonBreak: Probing Vulnerabilities in Reasoning-Enabled Vision-Language-Action Models for Autonomous Driving
- **分类: cs.CR; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶任务，研究推理增强的视觉-语言-动作模型的安全性，揭示其在输入扰动下的脆弱性，并提出评估框架与基准。**

- **链接: [https://arxiv.org/pdf/2605.29114](https://arxiv.org/pdf/2605.29114)**

> **作者:** Mohammadreza Teymoorianfard; Jean-Philippe Monteuuis; Jonathan Petit; Amir Houmansadr
>
> **摘要:** Vision-Language-Action (VLA) models with integrated reasoning have been proposed for end-to-end autonomous driving, assuming a tight coupling between reasoning and trajectory generation. However, the robustness of such systems under realistic input perturbations remains largely unexplored. We show that these models are highly vulnerable to realistic input perturbations, achieving up to 89% attack success rate (ASR) on reasoning and up to 72% on trajectory manipulation in closed-loop simulation, leading to increased collision rates and degraded safety metrics. Using NVIDIA's recent Alpamayo models as representative industry-developed VLAs, we conduct the first systematic black-box study of reasoning-enabled VLA models under realistic textual input corruptions, evaluating their impact on reasoning and driving behavior. We introduce a reasoning-aware evaluation framework capturing both semantic and structural aspects of reasoning, along with safety-centric measures. We also introduce a benchmark for evaluating attacks and defenses on reasoning-trajectory interactions in autonomous driving. Our results highlight the need for rigorous evaluation and improved defenses to ensure the safety of reasoning-enabled VLA systems in autonomous driving.
>
---
#### [new 036] Embodied3DBench: Benchmarking Low-Level Embodied Spatial Intelligence of Vision Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Embodied3DBench，评估视觉语言模型在3D环境中的空间感知能力，解决其在交互感知上的不足，通过基准测试和大规模数据训练提升模型表现。**

- **链接: [https://arxiv.org/pdf/2605.29074](https://arxiv.org/pdf/2605.29074)**

> **作者:** Jiyao Zhang; Mingxu Zhang; Yitong Peng; Haoxuan Liu; Chenshuo Wang; Yuxing Long; Haoyang Huang; Dongjiang Li; Nan Duan; Hui Shen; Hao Dong
>
> **摘要:** Are current Vision Language Models (VLMs) ready to comprehend and reason about complex embodied interactions in 3D environments? We introduce Embodied3DBench, a robot-centric benchmark targeting low-level spatial intelligence in embodied 3D environments. To systematically evaluate these foundational perceptual capabilities, the benchmark includes 6 task categories divided into two core groups: Spatial Structural Understanding (Grounding, Spatial Relation Prediction, and Multi-view Correspondence) and Interaction-Oriented Perception (Affordance Prediction, Grasp Point Prediction, and Trajectory Prediction). The benchmark spans 12 subcategories and contains over 21k high-quality question-answer pairs. We evaluate 13 state-of-the-art models, and the results show that while current models exhibit relatively strong high-level spatial reasoning, such as understanding object-to-object positional relations, they remain fragile in interaction-oriented perception, highlighting a significant lack of robust 3D-aware interaction priors. To actively bridge this capability gap revealed by our benchmark, we further synthesize a large-scale training dataset comprising 1.3M QA pairs. Notably, fine-tuning on this dataset yields significant improvements in low-level spatial intelligence. Ultimately, Embodied3DBench fills a critical gap by providing both a systematic evaluation framework and a scalable data solution, setting a clear target for the development of interaction-aware multimodal systems.
>
---
#### [new 037] Momentum Based Reward Design for Low Emission Traffic Signal Control
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于交通信号控制任务，旨在解决传统系统无法适应动态交通的问题。通过设计基于动量的奖励函数，提升交通效率并减少排放。**

- **链接: [https://arxiv.org/pdf/2605.29693](https://arxiv.org/pdf/2605.29693)**

> **作者:** Chinmay Mundane; Amith Manoharan; Arun Singh
>
> **摘要:** Urban traffic congestion is a growing global issue contributing significantly to long commute times and environmental pollution. Traditional traffic signal control systems often fail to adapt to dynamic traffic conditions. Adaptive traffic signal control can improve urban traffic without changing road infrastructure. Deep Reinforcement Learning (DRL) has shown strong performance for this task, but existing delay and queue-based rewards often produce short-sighted or unstable policies. This paper proposes a Momentum-Based Reward Function (MBRF) that encourages vehicles to keep moving rather than penalizing congestion alone. The method is evaluated in SUMO (Simulation of Urban MObility) using standard traffic metrics such as waiting time, queue length, throughput, and CO2 emissions. Results show that the proposed reward produces better throughput-emission trade-offs and more stable learning behavior than delay or queue-based rewards, as well as classical controllers such as Max Pressure and LQF.
>
---
#### [new 038] Distributed Non-Uniform Scaling Control of Multi-Agent Formation with Dynamic Agent Joining
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于多智能体系统控制任务，解决动态加入 agent 时非均匀缩放的问题，提出分布式控制框架保持图拉普拉斯谱特性。**

- **链接: [https://arxiv.org/pdf/2605.29191](https://arxiv.org/pdf/2605.29191)**

> **作者:** Tao He; Gangshan Jing
>
> **备注:** This paper has been accepted by IFAC 2026
>
> **摘要:** Non-uniform scaling control of formation enables multi-agent systems to adjust their shape by scaling with different ratios along different coordinate axes, offering enhanced flexibility in complex environments. However, like most existing formation maneuver strategies, it typically assumes a fixed set of agents, limiting its applicability in scenarios requiring dynamic team expansion. This paper introduces a distributed control framework that enables a formation to incorporate new agents during non-uniform scaling maneuvers in arbitrary dimensions while preserving the spectral properties of the graph Laplacian. Simulation examples validate the effectiveness of the theoretical results.
>
---
#### [new 039] Planning with the Views via Scene Self-Exploration
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文研究视觉语言模型的视角规划能力，解决多步视角变换组合问题。通过构建ViewSuite环境，提出迭代框架提升模型在3D空间中的主动规划能力。**

- **链接: [https://arxiv.org/pdf/2605.29563](https://arxiv.org/pdf/2605.29563)**

> **作者:** Kangrui Wang; Linjie Li; Zhengyuan Yang; Shiqi Chen; Zihan Wang; Li Fei-Fei; Jiajun Wu; Leonidas Guibas; Lijuan Wang; Manling Li
>
> **摘要:** Can VLMs predict how each camera move changes the view, and plan many such moves ahead? We call this capability view planning, requiring (1)understanding how a single action transforms the view, and (2)composing many such transformations across multi-turn plans to identify a target view. We probe both abilities in our proposed ViewSuite, a 3D point-cloud environment on real ScanNet scenes. Across 13 frontier VLMs, a critical planning gap emerges: they possess basic view-action knowledge but fail to compose it across multi-turn plans, with the gap widening as viewpoint distance grows. To close this gap, we propose an iterative framework that alternates self-exploration with view graph distillation. The key insight is that all exploration trajectories, regardless of their outcome, collectively form a view graph that compactly captures how viewpoints connect across a scene. Distilling this graph into diverse supervised tasks reshapes the policy distribution and overcomes the sparse rewards that stall pure RL. This improves Qwen2.5-VL-7B from 2.5% to 47.8% on interactive view planning, surpassing GPT-5.4 Pro (18.5%) and Gemini 3.1 Pro (21.4%). Self-exploration emerges as a promising path toward VLMs that can actively reason and plan in 3D space.
>
---
#### [new 040] Decoupled Thrust-Axis Attitude Control Using Quaternions for Chandrayaan-3 Lunar Landing Mission
- **分类: eess.SY; cs.RO**

- **简介: 论文属于 Chandrayaan-3 月球着陆任务，解决导航与控制耦合问题。提出基于四元数的解耦方法，实现独立推力轴控制，减少交互影响，确保姿态指令正确生成。**

- **链接: [https://arxiv.org/pdf/2605.29409](https://arxiv.org/pdf/2605.29409)**

> **作者:** Aditya Rallapalli; Suraj Kumar; Rijesh M P; Ashok Kumar Kakula; Bharat Kumar GVP
>
> **备注:** 6 pages, 7 figures, Published in Indian Control Conference 2025
>
> **摘要:** Chandrayaan-3 mission achieved a historic milestone with its successful soft landing near the lunar south pole, highlighting the critical role of the navigation, guidance, and control (NGC) system. Navigation provided vehicle state estimates relative to the Moon center, while a polynomial based guidance scheme computed the required acceleration profile to meet terminal landing conditions. This acceleration demand was translated into total thrust magnitude and attitude commands generation. Attitude command generation involved aligning the thrust axis with the required acceleration vector and constraining rotation about the thrust axis, typically governed by mission-specific requirements. Although quaternion-based control laws are preferred for their singularity-free representation, they inherently couple all three rotational axes. This coupling can lead to undesirable interactions between guidance and control, especially during large rotations about the thrust axis, due to the quaternion shortest-path property. This paper proposes a novel quaternion-based decoupling method that enables independent thrust-axis control, mitigating guidance-control interaction and ensuring proper attitude commands generation for lander attitude control.
>
---
#### [new 041] Ultra-Reduced-Impact-Encased-Logging (URIEL): propose a new method for selective sustainable logging and post-harvest silvicultural treatment in tropical forest using airborne robotics systems
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出URIEL方法，用于减少热带森林采伐影响，结合无人机和AI技术，解决可持续采伐与生态保护问题。**

- **链接: [https://arxiv.org/pdf/2605.28883](https://arxiv.org/pdf/2605.28883)**

> **作者:** Daniel Albiero; Gelton Fernando de Morais; Daniela Han; Flávio Roberto de Freitas Gonçalves; Artur Vitório Andrade Santos; Wesllen Lins de Araújo; Alessandra Maia Freire; Cláudio Kiyoshi Umezu; Mateus Peressin; Francesco Toscano; Admilson Írio Ribeiro; Alfeu J. Sguarezi Filho; Américo Ferraz Dias Neto; Angel Pontin Garcia
>
> **备注:** 196 pages, 40 figures, A revolutionary technology to help protect tropical forests. It was developed, scaled, detailed, calculated, and simulated in an advanced computational environment, com viabilidade econômica e social. "E pur si muove"
>
> **摘要:** Tropical forests worldwide are under intense deforestation pressure driven by economic and political interests, and scientific evidence suggests this deforestation contributes to climate change. This paper proposes a novel logging method for tropical forests, Ultra-Reduced-Impact-Encased-Logging (URIEL). This new method is based on heli-logging techniques combined with intensive use of robotics and AI integrated with post-harvest silvicultural treatments performed by drones. The concept of appropriate equipment for this method was developed, dimensions were determined, details were completed in a digital proof of concept, and an effective digital simulation and economic feasibility analysis were carried out for various helicopter-timber-distance combinations. The results demonstrated that a URIEL method has high economic viability and makes it possible to virtually eliminate collateral damage to forests while maintaining ecosystem services. The main conclusion of this paper is that, despite the satisfactory scientific and technological results, the feasibility of a Uriel method depends on the integration of stakeholders intrinsic to the context: high-tech industry; political governments; certified logging companies; and native populations.
>
---
#### [new 042] From General Vision to Reliable Traversability Estimation: Adapting Vision Foundation Models for Unstructured Outdoor Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉导航任务，解决户外环境中可行驶性估计的可靠性问题。通过引入ViTA框架，提升视觉基础模型的适应性，减少误报并增强跨领域泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.29565](https://arxiv.org/pdf/2605.29565)**

> **作者:** Ji-Hoon Hwang; Jisung Bae; Dong-Wook Kim; Yeonkyu Lee; Seung-Woo Seo
>
> **备注:** 8 pages, 5figures
>
> **摘要:** Vision-based approaches have become the dominant paradigm for traversability estimation in unstructured outdoor environments, typically adapting vision foundation models (VFMs) via semantic segmentation supervision. However, this paradigm faces three fundamental challenges that undermine its reliability: the task-agnostic design of VFMs, the ambiguity of traversability annotations, and the discrepancy between semantic labels and physical safety. We propose Vision-to-Traversability Adaptation (ViTA), a framework that adapts VFMs for reliable traversability estimation, instantiated on SAM2. ViTA injects task-specific knowledge through learnable traversability prompts while preserving the VFM's cross-domain generalization. To handle annotation ambiguity, we introduce Perspective-Diversified Training, which estimates semantic uncertainty to suppress confident predictions at ambiguous boundaries. To bridge the semantic-traversability discrepancy, we distill geometric knowledge during training, enabling slope and elevation reasoning from RGB images alone at inference. The semantic and geometric outputs are fused into a continuous traversability score that reflects both semantic uncertainty and geometric risk. Evaluations across diverse domains, including challenging real-world off-road datasets, demonstrate that ViTA achieves state-of-the-art IoU and Precision with substantial false-positive reduction and strong cross-domain generalization.
>
---
#### [new 043] Uncertainty-driven 3D Gaussian Splatting Active Mapping via Anisotropic Visibility Field
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GAVIS，解决3D高斯溅射中的不确定性量化与主动映射问题。通过构建各向异性可见性场，提升预测可靠性与实时性。**

- **链接: [https://arxiv.org/pdf/2605.30342](https://arxiv.org/pdf/2605.30342)**

> **作者:** Shangjie Xue; Jesse Dill; Dhruv Ahuja; Frank Dellaert; Panagiotis Tsiotras; Danfei Xu
>
> **备注:** Accepted to CVPR 2026. Project page this https URL
>
> **摘要:** We present Gaussian Splatting Anisotropic Visibility Field (GAVIS), a novel framework for uncertainty quantification and active mapping in 3DGS. Our key insight is that regions unseen from the training views yield unreliable predictions from the 3DGS. To address this, we introduce a principled and efficient method for quantifying the visibility field in 3DGS, defined as the anisotropic visibility of each particle with respect to the training views, and represented using spherical harmonics. The resulting visibility field is integrated into a Bayesian Network-based uncertainty-aware 3DGS rasterizer, enabling real-time (200 FPS) uncertainty quantification for synthesized views. Active mapping is further performed within a maximum information gain framework building on this formulation. Extensive experiments across diverse environments demonstrate that GAVIS consistently and significantly outperforms prior approaches in both accuracy and efficiency. Moreover, beyond standalone use, our method can be applied post-hoc to improve the performance of existing approaches.
>
---
## 更新

#### [replaced 001] Accelerating trajectory optimization with Sobolev-trained diffusion policies
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于轨迹优化任务，旨在提升优化速度与质量。通过训练扩散策略提供初始猜测，解决传统方法收敛慢、依赖初始轨迹的问题。**

- **链接: [https://arxiv.org/pdf/2604.19011](https://arxiv.org/pdf/2604.19011)**

> **作者:** Théotime Le Hellard; Franki Nguimatsia Tiofack; Quentin Le Lidec; Justin Carpentier
>
> **摘要:** Trajectory Optimization (TO) solvers exploit known system dynamics to compute locally optimal trajectories through iterative improvements. A downside is that each new problem instance is solved independently; therefore, convergence speed and quality of the solution found depend on the initial trajectory proposed. To improve efficiency, a natural approach is to warm-start TO with initial guesses produced by a learned policy trained on trajectories previously generated by the solver. Diffusion-based policies have recently emerged as expressive imitation learning models, making them promising candidates for this role. Yet, a counterintuitive challenge comes from the local optimality of TO demonstrations: when a policy is rolled out, small non-optimal deviations may push it into situations not represented in the training data, triggering compounding errors over long horizons. In this work, we focus on learning-based warm-starting for gradient-based TO solvers that also provide feedback gains. Exploiting this specificity, we derive a first-order loss for Sobolev learning of diffusion-based policies using both trajectories and feedback gains. Through comprehensive experiments, we demonstrate that the resulting policy avoids compounding errors, and so can learn from very few trajectories to provide initial guesses reducing solving time by $2\times$ to $20 \times$. Incorporating first-order information enables predictions with fewer diffusion steps, reducing inference latency.
>
---
#### [replaced 002] VLA-ATTC: Adaptive Test-Time Compute for VLA Models with Relative Action Critic Model
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的优化任务，旨在解决模型在复杂场景下决策不足的问题。提出VLA-ATTC框架，通过动态切换机制和相对动作评价模型提升决策质量。**

- **链接: [https://arxiv.org/pdf/2605.01194](https://arxiv.org/pdf/2605.01194)**

> **作者:** Wenhao Li; Xiu Su; Yichao Cao; Hongyan Xu; Xiaobo Xia; Shan You; Yi Chen; Chang Xu
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable capabilities and generalization in embodied manipulation. However, their decision-making relies on a fast, instinctive process that lacks deliberation. This strategy often leads to suboptimal or catastrophic actions when facing complex or ambiguous scenarios that require greater consideration. In this paper, we introduce \textbf{VLA-ATTC}, a framework that endows VLA models with adaptive test-time compute (TTC). VLA-ATTC employs an uncertainty-based ``cognitive clutch'' to dynamically transition from reflexive execution to a TTC deliberation phase when necessary. During TTC phase, a novel \textbf{Relative Action Critic} (RAC) model identifies the optimal action from generated candidates via pairwise comparisons. This relative mechanism replaces unstable absolute value estimation, significantly simplifying the learning objective. Furthermore, we introduce an efficient sampling strategy to amortize computational costs and an automated data pipeline that curates preference pairs without manual annotation. On the LIBERO-LONG benchmark, VLA-ATTC reduces the failure rate of the SOTA model PI0.5 by over 50\%. We will open-source all the code and weights.
>
---
#### [replaced 003] Simulation-based planning of Motion Sequences for Automated Procedure Optimization in Multi-Robot Assembly Cells
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多机器人装配任务，旨在优化多机器人运动序列以缩短装配时间。通过仿真方法分离核心与移动操作，采用分解策略进行路径规划，提升效率并避免碰撞。**

- **链接: [https://arxiv.org/pdf/2507.23270](https://arxiv.org/pdf/2507.23270)**

> **作者:** Loris Schneider; Marc Ungen; Elias Huber; Jan-Felix Klein
>
> **备注:** Accepted for publication at IEEE CASE 2026
>
> **摘要:** Reconfigurable multi-robot cells offer a promising approach to meet fluctuating assembly demands. However, the recurrent planning of their configurations introduces new challenges, particularly in generating optimized, coordinated multi-robot motion sequences that minimize the assembly duration. This work presents a simulation-based method for generating such optimized sequences. The approach separates assembly steps into task-related core operations and connecting traverse operations. While core operations are constrained and predetermined, traverse operations offer substantial optimization potential. Scheduling the core operations is formulated as an optimization problem, requiring feasible traverse operations to be integrated using a decomposition-based motion planning strategy. Several solution techniques are explored, including a sampling heuristic, tree-based search and gradient-free optimization. For motion planning, a decomposition method is proposed that identifies specific areas in the schedule, which can be solved independently with modified centralized path planning algorithms. The proposed method generates efficient and collision-free multi-robot assembly procedures that outperform a baseline relying on decentralized, robot-individual motion planning. Its effectiveness is demonstrated through simulation experiments.
>
---
#### [replaced 004] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人策略学习任务，旨在解决视觉-语言-动作模型（VLAs）与世界模型结合时的状态和动作联合预测问题。提出DUST框架，通过多模态扩散Transformer和独立噪声扰动提升性能。**

- **链接: [https://arxiv.org/pdf/2510.27607](https://arxiv.org/pdf/2510.27607)**

> **作者:** John Won; Kyungmin Lee; Huiwon Jang; Dongyoung Kim; Jinwoo Shin
>
> **备注:** Accepted at ICML 2026. Project page at this https URL (20 pages, 10 figures)
>
> **摘要:** Augmenting vision-language-action models (VLAs) with world models is promising for robotic policy learning but faces challenges in jointly predicting states and actions due to the modality gap. To address this, we propose DUal-STream diffusion (DUST), a world-model augmented VLA framework featuring a multimodal diffusion transformer that maintains separate modality streams while enabling cross-modal knowledge sharing. In addition, DUST utilizes independent noise perturbations and a decoupled flow matching loss to learn cross-modal causal relationships. We further introduce an asynchronous sampling method for action and vision tokens that enhances performance through inference-time scaling. Experimental results on simulated benchmarks like RoboCasa and GR-1 show that DUST achieves up to 6% gains over state-of-the-art VLA and world-modeling baselines, with inference-time scaling providing an additional 2-5% improvement. In real-world tasks using the Franka Research 3, DUST outperforms baselines by 10% in success rate. Finally, we demonstrate that DUST enables effective transfer learning through both pretraining on action-free videos and joint-training with heterogeneous robot and human datasets.
>
---
#### [replaced 005] Environment-Adaptive Solid-State LiDAR-Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于定位与建图任务，旨在解决极端环境下LiDAR-惯性里程计的精度问题。通过引入局部法向量约束和退化感知地图维护，提升定位准确性和地图一致性。**

- **链接: [https://arxiv.org/pdf/2604.15864](https://arxiv.org/pdf/2604.15864)**

> **作者:** Zhi Zhang; Chalermchon Satirapod; Bingtao Ma; Changjun Gu
>
> **摘要:** Solid-state LiDAR-inertial SLAM has attracted significant attention due to its advantages in speed and robustness. However, achieving accurate mapping in extreme environments remains challenging due to severe geometric degeneracy and unreliable observations, which often lead to ill-conditioned optimization and map inconsistencies. To address these challenges, we propose an environment-adaptive solid-state LiDAR-inertial odometry that integrates local normal-vector constraints with degeneracy-aware map maintenance to enhance localization accuracy. Specifically, we introduce local normal-vector constraints to improve the stability of state estimation, effectively suppressing localization drift in degenerate scenarios. Furthermore, we design a degeneration-guided map update strategy to improve map precision. Benefiting from the refined map representation, localization accuracy is further enhanced in subsequent estimation. Experimental results demonstrate that the proposed method achieves superior mapping accuracy and robustness in extreme and perceptually degraded environments, with an average RMSE reduction of up to 12.8% compared to the baseline method.
>
---
#### [replaced 006] TACO: Temporal Consensus Optimization for Continual Neural Mapping
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决动态环境中持续学习的神经映射问题。提出TACO框架，通过时间共识优化实现无需回放的历史知识利用，平衡记忆效率与适应性。**

- **链接: [https://arxiv.org/pdf/2602.04516](https://arxiv.org/pdf/2602.04516)**

> **作者:** Xunlan Zhou; Hongrui Zhao; Negar Mehr
>
> **备注:** In: Robotics: Science and Systems (RSS 2026)
>
> **摘要:** Neural implicit mapping has emerged as a powerful paradigm for robotic navigation and scene understanding. However, real-world robotic deployment requires continual adaptation to changing environments under strict memory and computation constraints, which existing mapping systems fail to support. Most prior methods rely on replaying historical observations to preserve consistency and assume static scenes. As a result, they cannot adapt to continual learning in dynamic robotic settings. To address these challenges, we propose TACO (TemporAl Consensus Optimization), a replay-free framework for continual neural mapping. We reformulate mapping as a temporal consensus optimization problem, where we treat past model snapshots as temporal neighbors. Intuitively, our approach resembles a model consulting its own past knowledge. We update the current map by enforcing weighted consensus with historical representations. Our method allows reliable past geometry to constrain optimization while permitting unreliable or outdated regions to be revised in response to new observations. TACO achieves a balance between memory efficiency and adaptability without storing or replaying previous data. Through extensive simulated and real-world experiments, we show that TACO robustly adapts to scene changes, and consistently outperforms other continual learning baselines. Code is available at this https URL
>
---
#### [replaced 007] HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出HumanEgo框架，解决从人类第一视角视频中零样本迁移机器人技能的问题。通过构建人机交互表示，提升机器人操作成功率。**

- **链接: [https://arxiv.org/pdf/2605.24934](https://arxiv.org/pdf/2605.24934)**

> **作者:** Zhi Wang; Botao He; Kelin Yu; Seungjae Lee; Ruohan Gao; Furong Huang; Yiannis Aloimonos
>
> **备注:** Project page: this https URL
>
> **摘要:** Human egocentric video captures rich manipulation demonstrations without any robot hardware, yet transferring these skills to robots remains challenging due to the embodiment gap between human and robot in both visual appearance and kinematics. We present HumanEgo, a framework that bridges the embodiment gap by lifting each human demonstration to an entity-level representation of hand-object interaction, and training a flow matching policy with dense auxiliary objectives that amplify supervision from every trajectory. HumanEgo is robot-data-free, hardware-agnostic, data-efficient, and zero-shot human-to-robot transferable. With only 30 minutes of human videos per task, HumanEgo achieves 92.5% average success across four real-world tasks (75% with just 15 minutes), outperforms matched-time robot teleoperation by 41%, and robustly transfers zero-shot across novel robots, cameras, and environments. We release HumanEgo as an easy-to-use, open-source framework for learning robot policies directly from human data: this https URL
>
---
#### [replaced 008] Multifingered force-aware control for humanoid robots
- **分类: cs.RO**

- **简介: 该论文属于人形机器人多指力控任务，解决力分布与稳定接触问题。通过估计力并优化接触点，提升抓取稳定性。**

- **链接: [https://arxiv.org/pdf/2603.08142](https://arxiv.org/pdf/2603.08142)**

> **作者:** Pasquale Marra; Gabriele M. Caddeo; Ugo Pattacini; Lorenzo Natale
>
> **备注:** This work has been accepted for publication in ICRA 2026
>
> **摘要:** In this paper, we address force-aware control and force distribution in robotic platforms with multi-fingered hands. Given a target goal and force estimates from tactile sensors, we design a controller that adapts the motion of the torso, arm, wrist, and fingers, redistributing forces to maintain stable contact with objects of varying mass distribution or unstable contacts. To estimate forces, we collect a dataset of tactile signals and ground-truth force measurements using five Xela magnetic sensors interacting with indenters, and train force estimators. We then introduce a model-based control scheme that minimizes the distance between the Center of Pressure (CoP) and the centroid of the fingertips contact polygon. Since our method relies on estimated forces rather than raw tactile signals, it has the potential to be applied to any sensor capable of force estimation. We validate our framework on a balancing task with five objects, achieving a $82.7\%$ success rate, and further evaluate it in multi-object scenarios, achieving $80\%$ accuracy. Code and data can be found here this https URL.
>
---
#### [replaced 009] Trust, Geometry, and Rules: A Credibility-Aware Reinforcement Learning Framework for Safe USV Navigation under Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决USV在不确定环境下的安全航行问题。提出融合可信度、几何安全和规则感知的强化学习框架，提升导航鲁棒性和合规性。**

- **链接: [https://arxiv.org/pdf/2605.26974](https://arxiv.org/pdf/2605.26974)**

> **作者:** Yuhang Zhang; Shuqi Chai; Yukang Zhang; Liusha Yang; Mingchuan Zhang; Wei Wang; Qingjiang Shi; Quanbo Ge
>
> **摘要:** Autonomous navigation of Unmanned Surface Vehicles (USVs) that is safe and compliant with the International Regulations for Preventing Collisions at Sea (COLREGs) remains a formidable challenge in dynamic maritime environments, particularly when perception systems exhibit miscalibrated uncertainty. Existing Reinforcement Learning (RL)-based methods often falter because state-estimation errors induce unreliable belief states that mislead the value function, while discrete traffic rules introduce discontinuity in the learning objective. To address these challenges, we propose a framework integrating credibility-aware learning, geometric safety shielding, and continuous rule-aware embedding. First, Credibility-Weighted Value Learning (CW-VL) introduces a dynamic trust factor derived from the discrepancy between filter-estimated covariance and empirical error statistics to modulate the critic's heteroscedastic loss, preventing policy overfitting to noisy samples. Second, the Covariance-Inflated Velocity Obstacle (CI-VO) maps position-estimation uncertainty into set-wise angular margins, forming a conservative geometric shield that overrides hazardous exploratory actions. Third, Risk-Aware COLREGs Duty Embedding relaxes binary encounter duties into continuous rule-aware signals, providing smooth sector-transition information and suppressing oscillation from sparse rule rewards. Simulated encounter studies demonstrate improved training robustness against perceptual inconsistency and superior collision avoidance and COLREGs compliance over baselines.
>
---
#### [replaced 010] AttenA+: Rectifying Action Inequality in Robotic Foundation Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决现有模型对动作信息均匀处理导致的性能限制。通过引入AttenA+框架，基于速度调整训练权重，提升复杂任务表现。**

- **链接: [https://arxiv.org/pdf/2605.13548](https://arxiv.org/pdf/2605.13548)**

> **作者:** Daojie Peng; Fulong Ma; Jiahang Cao; Qiang Zhang; Xupeng Xie; Jian Guo; Ping Luo; Andrew F. Luo; Boyu Zhou; Jun Ma
>
> **摘要:** Existing robotic foundation models, while powerful, are predicated on an implicit assumption of temporal homogeneity: treating all actions as equally informative during optimization. This "flat" training paradigm, inherited from language modeling, remains indifferent to the underlying physical hierarchy of manipulation. In reality, robot trajectories are fundamentally heterogeneous, where low-velocity segments often dictate task success through precision-demanding interactions, while high-velocity motions serve as error-tolerant transitions. Such a misalignment between uniform loss weighting and physical criticality fundamentally limits the performance of current Vision-Language-Action (VLA) models and World-Action Models (WAM) in complex, long-horizon tasks. To rectify this, we introduce AttenA+, an architecture-agnostic framework that prioritizes kinematically critical segments via velocity-driven action attention. By reweighting the training objective based on the inverse velocity field, AttenA+ naturally aligns the model's learning capacity with the physical demands of manipulation. As a plug-and-play enhancement, AttenA+ can be integrated into existing backbones without structural modifications or additional parameters. Extensive experiments demonstrate that AttenA+ significantly elevates the ceilings of current state-of-the-art models. Specifically, it improves OpenVLA-OFT to 98.6% (+1.5%) on the Libero benchmark and pushes FastWAM to 92.4% (+0.6%) on RoboTwin 2.0. Real-world validation on a Franka manipulator further showcases its robustness and cross-task generalization. Our work suggests that mining the intrinsic structural priors of action sequences offers a highly efficient, physics-aware complement to standard scaling laws, paving a new path for general-purpose robotic control.
>
---
#### [replaced 011] Sentinel-VLA: A Metacognitive VLA Model with Active Status Monitoring for Dynamic Reasoning and Error Recovery
- **分类: cs.RO**

- **简介: 该论文提出Sentinel-VLA，解决VLA模型推理能力弱、缺乏状态监控和自纠错的问题。通过引入哨兵模块和SECL算法，提升任务成功率。属于视觉-语言-动作模型优化任务。**

- **链接: [https://arxiv.org/pdf/2605.01191](https://arxiv.org/pdf/2605.01191)**

> **作者:** Wenhao Li; Xiu Su; Dan Niu; Yichao Cao; Hongyan Xu; Zhe Qu; Lei Fan; Shan You; Chang Xu
>
> **摘要:** Vision-language-action (VLA) models have advanced the field of embodied manipulation by harnessing broad world knowledge and strong generalization. However, current VLA models still face several key challenges, including limited reasoning capability, lack of status monitoring, and difficulty in self-correction. In this paper, we introduce \textbf{Sentinel-VLA}, a metacognitive VLA model equipped with an active ``sentinel'' module to monitor real-time execution status. Only when necessary, such as during initial planning or upon detecting an error, the model triggers a dynamic reasoning or formulate error recovery solutions. This on-demand reasoning mechanism ensures robust decision-making while minimizing computational overhead. Notably, all training data (spanning 44 tasks and over 2.6 million transitions) is automatically generated and annotated through our designed pipeline. We also propose the Self-Evolving Continual Learning (SECL) algorithm, which allows Sentinel-VLA to identify its capability boundaries and automatically collect data for expansion, paired with Orthogonal Continual Adapter (OC-Adapter) to constrain parameter updates to an orthogonal space, thereby preventing catastrophic forgetting. Real-world experiments demonstrate that Sentinel-VLA boosts the task success rate by over 30\% compared to the SOTA model, PI0. We will open-source all the code, weights, and data generation pipeline.
>
---
#### [replaced 012] Scensory: Real-Time Robotic Olfactory Perception for Joint Identification and Source Localization
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于机器人嗅觉感知任务，旨在解决从扩散化学信号中同时识别真菌种类和定位其来源的问题。通过学习框架Scensory实现高效、实时的环境监测。**

- **链接: [https://arxiv.org/pdf/2509.19318](https://arxiv.org/pdf/2509.19318)**

> **作者:** Yanbaihui Liu; Erica Babusci; Claudia K. Gunsch; Boyuan Chen
>
> **备注:** Our project website is at: this http URL
>
> **摘要:** While robotic perception has advanced rapidly in vision and touch, enabling robots to reason about indoor fungal contamination from weak, diffusion-dominated chemical signals remains an open challenge. We introduce Scensory, a learning-based robotic olfaction framework that simultaneously identifies fungal species and localizes their source from short time series measured by affordable, cross-sensitive VOC sensor arrays. Temporal VOC dynamics encode both chemical and spatial signatures, which we decode through neural networks trained on robot-automated data collection with spatial supervision. Across five fungal species, Scensory achieves up to 89.85% species accuracy and 87.31% source localization accuracy under ambient conditions with 3-7s sensor inputs. These results demonstrate real-time, spatially grounded perception from diffusion-dominated chemical signals, enabling scalable and low-cost source localization for robotic indoor environmental monitoring.
>
---
#### [replaced 013] A Review of Learning-Based Motion Planning: Toward a Data-Driven Optimal Control Approach
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶中的运动规划任务，旨在解决传统方法与学习方法之间的权衡问题。通过提出数据驱动最优控制框架，融合控制理论与机器学习，提升规划的适应性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.11944](https://arxiv.org/pdf/2512.11944)**

> **作者:** Jia Hu; Yang Chang; Haoran Wang
>
> **备注:** 44 pages, 14 figures
>
> **摘要:** Motion planning for autonomous driving (AD) faces a critical trade-off. While traditional rule-based pipelines offer verifiable safety and interpretability, they often fail to generalize in complex scenarios. Conversely, emerging learning-based methods-including imitation learning (IL), reinforcement learning (RL), and generative AI-offer greater adaptability but are often constrained by opacity and safety risks. Existing surveys typically analyze these AI methods in isolation, overlooking the potential of integrating them with rigorous control frameworks. To bridge this gap, this paper presents the first systematic review of the Data-Driven Optimal Control (DDOC) paradigm, explicitly examining how it synergizes the theoretical guarantees of optimal control with the adaptive capabilities of modern machine learning. Building on this framework, we propose the first roadmap for DDOC-based motion planning, structuring its implementation into three critical dimensions: customization, dynamics adaptation, and self-tuning. Finally, to close the remaining reality gap, we identify four future research directions, thereby accelerating the transition to trustworthy and human-like autonomous driving.
>
---
#### [replaced 014] Phantom: Training Robots Without Robots Using Only Human Videos
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决传统方法依赖机器人数据的问题。通过人类视频训练机器人操作策略，无需实际机器人数据。**

- **链接: [https://arxiv.org/pdf/2503.00779](https://arxiv.org/pdf/2503.00779)**

> **作者:** Marion Lepert; Jiaying Fang; Jeannette Bohg
>
> **备注:** Project website at this https URL
>
> **摘要:** Training general-purpose robots requires learning from large and diverse data sources. Current approaches rely heavily on teleoperated demonstrations which are difficult to scale. We present a scalable framework for training manipulation policies directly from human video demonstrations, requiring no robot data. Our method converts human demonstrations into robot-compatible observation-action pairs using hand pose estimation and visual data editing. We inpaint the human arm and overlay a rendered robot to align the visual domains. This enables zero-shot deployment on real hardware without any fine-tuning. We demonstrate strong success rates-up to 92%-on a range of tasks including deformable object manipulation, multi-object sweeping, and insertion. Our approach generalizes to novel environments and supports closed-loop execution. By demonstrating that effective policies can be trained using only human videos, our method broadens the path to scalable robot learning.
>
---
#### [replaced 015] Dynamic Mixture of Progressive Parameter-Efficient Expert Library for Lifelong Robot Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于终身机器人学习任务，解决持续学习中的灾难性遗忘问题。提出DMPEL框架，通过动态组合专家库实现高效知识迁移与存储。**

- **链接: [https://arxiv.org/pdf/2506.05985](https://arxiv.org/pdf/2506.05985)**

> **作者:** Yuheng Lei; Sitong Mao; Shunbo Zhou; Hongyuan Zhang; Xuelong Li; Ping Luo
>
> **备注:** Accepted to Transactions on Machine Learning Research (TMLR) at this https URL . Code is available at this https URL
>
> **摘要:** A generalist agent must continuously learn and adapt throughout its lifetime, achieving efficient forward transfer while minimizing catastrophic forgetting. Previous work within the dominant pretrain-then-finetune paradigm has explored parameter-efficient fine-tuning for single-task adaptation, effectively steering a frozen pretrained model with a small number of parameters. However, in the context of lifelong learning, these methods rely on the impractical assumption of a test-time task identifier and restrict knowledge sharing among isolated adapters. To address these limitations, we propose Dynamic Mixture of Progressive Parameter-Efficient Expert Library (DMPEL) for lifelong robot learning. DMPEL progressively builds a low-rank expert library and employs a lightweight router to dynamically combine experts into an end-to-end policy, enabling flexible and efficient lifelong forward transfer. Furthermore, by leveraging the modular structure of the fine-tuned parameters, we introduce expert coefficient replay, which guides the router to accurately retrieve frozen experts for previously encountered tasks. This technique mitigates forgetting while being significantly more storage- and computation-efficient than experience replay over the entire policy. Extensive experiments on the lifelong robot learning benchmark LIBERO demonstrate that our framework outperforms state-of-the-art lifelong learning methods in success rates during continual adaptation, while utilizing minimal trainable parameters and storage.
>
---
#### [replaced 016] Quasi-Static Control of Discrete Cosserat Rod
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究软体机器人控制问题，通过离散化Cosserat杆模型，设计基于外部力的反馈控制律，实现末端轨迹跟踪与形状控制。**

- **链接: [https://arxiv.org/pdf/2605.01395](https://arxiv.org/pdf/2605.01395)**

> **作者:** Srishti Siddharth
>
> **备注:** Submitted to 17th APCA International Conference on Automatic Control and Soft Computing (CONTROLO 2026)
>
> **摘要:** In this paper, we design feedback control laws for soft robots modelled using the Cosserat rod, which is spatially discretised using the Piecewise Constant Strain (PCS) approach. The PCS approach transforms the nonlinear PDEs describing the Cosserat rod to a system of nonlinear ODEs. This simplification results in a model describing soft robots which is similar to the serial rigid-link manipulators. We design feedback control laws for the quasi-static PCS model by using the external wrenches as control input. The control laws are designed based on state-feedback linearisation in strain and task spaces. An extensive set of numerical results demonstrates the performance of the control laws for end-effector trajectory tracking and shape control of soft robots.
>
---
#### [replaced 017] CoRMA: Contrastive RMA for Contact-Rich Meta-Adaptation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出CoRMA，用于高接触力装配的元适应任务，解决真实环境中装配失败问题。通过语义接触上下文推理实现在线适应，无需演示或梯度更新。**

- **链接: [https://arxiv.org/pdf/2605.22082](https://arxiv.org/pdf/2605.22082)**

> **作者:** Wentian Wang; Chutong Wen; Hongxu Ma; Wuhao Wang; Zhexiong Xue; Abdul Haseeb Nizamani; Dandi Zhou; Xinhai Sun; Jianqiao Zhu
>
> **摘要:** We present CoRMA(Contrastive Robotic Motor Adaptation), a context-based meta-adaptation framework that modifies RMA for force-dominant assembly. CoRMA replaces raw simulator-parameter adaptation with a compact 6D simulator-only semantic contact context describing contact onset, lateral engagement, guided transition, contact direction, and jamming. A deployable causal Transformer adapter infers this context online from force, proprioceptive, and action histories using semantic regression and a force-regime contrastive objective. At deployment, oracle context is removed and replaced by the inferred context, enabling within-episode adaptation without demonstrations, privileged inputs, or gradient updates. We evaluate CoRMA on PegInsert, GearMesh, and NutThread in Isaac Lab / Isaac Sim 5.0 and on a real Marvin arm. Compared with FORGE baselines that achieve high simulation success but degrade substantially on hardware, CoRMA retains higher verified real success under controlled target-pose noise. These results support semantic contact inference as a reusable adaptation interface within a related assembly task family, while broader unseen-task generalization and Real2Sim calibration remain future work.
>
---
#### [replaced 018] Dual Quaternion SE(3) Synchronization with Recovery Guarantees
- **分类: math.OC; cs.CV; cs.RO; eess.SP**

- **简介: 该论文研究SE(3)同步问题，旨在从噪声相对变换中恢复绝对位姿。通过双四元数表示，提出一种两阶段算法，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.00324](https://arxiv.org/pdf/2602.00324)**

> **作者:** Jianing Zhao; Linglingzhi Zhu; Anthony Man-Cho So
>
> **备注:** ICML 2026
>
> **摘要:** Synchronization over the special Euclidean group SE(3) aims to recover absolute poses from noisy pairwise relative transformations and is a core primitive in robotics and 3D vision. Standard approaches often require multi-step heuristic procedures to recover valid poses, which are difficult to analyze and typically lack theoretical guarantees. This paper adopts a dual quaternion representation and formulates SE(3) synchronization directly over the unit dual quaternion. A two-stage algorithm is developed: A spectral initializer computed via the power method on a Hermitian dual quaternion measurement matrix, followed by a dual quaternion generalized power method (DQGPM) that enforces feasibility through per-iteration projection. The estimation error bounds are established for spectral estimators, and DQGPM is shown to admit a finite-iteration error bound and achieves linear error contraction up to an explicit noise-dependent threshold. Experiments on synthetic benchmarks and real-world multi-scan point-set registration demonstrate that the proposed pipeline improves both accuracy and efficiency over representative matrix-based methods.
>
---
#### [replaced 019] Force Sensing for Wearable Human-Robot Interfaces via Fluidic Innervation
- **分类: cs.RO**

- **简介: 该论文属于人机交互传感任务，旨在解决穿戴式机器人接口的传感难题。通过流体神经支配技术，设计3D打印硅胶垫，实现对肢体与设备相互作用的精确力检测。**

- **链接: [https://arxiv.org/pdf/2602.13436](https://arxiv.org/pdf/2602.13436)**

> **作者:** Noah Rubin; Ava Schraeder; Hrishikesh Sahu; Thomas C. Bulea; Lillian Chin
>
> **备注:** 6 pages, 7 figures, accepted to BioRob 2026
>
> **摘要:** Mechanically characterizing the human-machine interface is essential to understanding user behavior and optimizing wearable robot performance. This interface has been challenging to sensorize due to manufacturing complexity and non-linear sensor responses. Here, we measure human limb-device interaction via fluidic innervation, creating a 3D-printed silicone pad with embedded air channels to measure forces. As forces are applied to the pad, the air channels compress, resulting in a pressure change measurable by off-the-shelf pressure transducers. We demonstrate in benchtop testing that pad pressure is highly linearly related to applied force ($R^2 = 0.998$) and confirmed strong linear relationships to isometric knee torque in a clinical dynamometer with strategic pad placement. We built on these idealized settings to test pad performance in more unconstrained settings, including during cyclic dynamic and stepwise isometric bicep curls. Finally, we integrated the sensor into a lower-extremity robotic exoskeleton and recorded pad pressure during repeated squats with the device unpowered. Pad pressure tracked squat phase and overall task dynamics consistently. Collectively, our preliminary results suggest fluidic innervation is a readily customizable sensing modality with high signal-to-noise ratio and temporal resolution for capturing human-machine interaction. In the long-term, this modality may provide an alternative real-time sensing input to control / optimize wearable robotic systems and to capture user function during device use.
>
---
#### [replaced 020] TRUST-Planner: Topology-guided Robust Trajectory Planner for AAVs with Uncertain Obstacle Spatial-temporal Avoidance
- **分类: cs.RO**

- **简介: 该论文属于自主飞行器路径规划任务，解决动态环境中局部极小和死锁问题。提出TRUST-Planner框架，结合拓扑引导与高效避障算法，提升规划成功率与效率。**

- **链接: [https://arxiv.org/pdf/2508.14610](https://arxiv.org/pdf/2508.14610)**

> **作者:** Junzhi Li; Teng Long; Jingliang Sun; Jianxin Zhong
>
> **备注:** Accepted by IEEE Transactions on Industrial Electronics (TIE) for publication. The final version will be available online at this https URL after publication
>
> **摘要:** Despite extensive developments in motion planning of autonomous aerial vehicles (AAVs), existing frameworks faces the challenges of local minima and deadlock in complex dynamic environments, leading to increased collision risks. To address these challenges, we present TRUST-Planner, a topology-guided hierarchical planning framework for robust spatial-temporal obstacle avoidance. In the frontend, a dynamic enhanced visible probabilistic roadmap (DEV-PRM) is proposed to rapidly explore topological paths for global guidance. The backend utilizes a uniform terminal-free minimum control polynomial (UTF-MINCO) and dynamic distance field (DDF) to enable efficient predictive obstacle avoidance and fast parallel computation. Furthermore, an incremental multi-branch trajectory management framework is introduced to enable spatio-temporal topological decision-making, while efficiently leveraging historical information to reduce replanning time. Simulation results show that TRUST-Planner outperforms baseline competitors, achieving a 96\% success rate and millisecond-level computation efficiency in tested complex environments. Real-world experiments further validate the feasibility and practicality of the proposed method.
>
---
#### [replaced 021] Follow Everything: A Leader-Following and Obstacle Avoidance Framework with Goal-Aware Adaptation
- **分类: cs.RO**

- **简介: 该论文属于机器人领导跟随任务，解决领导者形式多样和视觉丢失问题。通过分割模型和目标感知机制，提升跟随鲁棒性与避障能力。**

- **链接: [https://arxiv.org/pdf/2504.19399](https://arxiv.org/pdf/2504.19399)**

> **作者:** Qianyi Zhang; Shijian Ma; Boyi Liu; Jianhao Jiao; Dimitrios Kanoulas
>
> **摘要:** Robust and flexible leader-following is a critical capability for robots to integrate into human society. While existing methods struggle to generalize to leaders of arbitrary form and often fail when the leader temporarily leaves the robot's field of view, this work introduces a unified framework addressing both challenges. First, traditional detection models are replaced with a segmentation model, allowing the leader to be anything. To enhance recognition robustness, a distance frame buffer is implemented that stores leader embeddings at multiple distances, accounting for the unique characteristics of leader-following tasks. Second, a goal-aware adaptation mechanism is designed to govern robot planning states based on the leader's visibility and motion, complemented by a graph-based planner that generates candidate trajectories for each state, ensuring efficient following with obstacle avoidance. Simulations and real-world experiments with a legged robot follower and various leaders (human, ground robot, UAV, legged robot, stop sign) in both indoor and outdoor environments show competitive improvements in follow success rate, reduced visual loss duration, lower collision rate, and decreased leader-follower distance.
>
---
#### [replaced 022] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习领域，旨在解决离线强化学习的效率与性能问题。提出FAN算法，通过简化流策略和分布批评器，提升计算效率并保持高性能。**

- **链接: [https://arxiv.org/pdf/2605.01663](https://arxiv.org/pdf/2605.01663)**

> **作者:** Sungyoung Lee; Dohyeong Kim; Eshan Balachandar; Zelal Su Mustafaoglu; Keshav Pingali
>
> **备注:** ICML 2026
>
> **摘要:** We propose Flow-Anchored Noise-conditioned Q-Learning (FAN), a highly efficient and high-performing offline reinforcement learning (RL) algorithm. Recent work has shown that expressive flow policies and distributional critics improve offline RL performance, but at a high computational cost. Specifically, flow policies require iterative sampling to produce a single action, and distributional critics require computation over multiple samples (e.g., quantiles) to estimate value. To address these inefficiencies while maintaining high performance, we introduce FAN. Our method employs a behavior regularization technique that uses a single flow policy iteration and requires a single Gaussian noise sample for distributional critics. Our theoretical analysis of convergence and performance bounds demonstrates that these simplifications not only improve efficiency but also lead to superior task performance. Experiments on robotic manipulation and locomotion tasks demonstrate that FAN achieves state-of-the-art performance while significantly reducing both training and inference runtimes. We release our code at this https URL.
>
---
#### [replaced 023] ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated Multi-Arm Task and Motion Planning & Scheduling
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出ScheduleStream，解决多机械臂任务与运动规划及调度问题，通过采样方法实现并行动作规划，提升效率。**

- **链接: [https://arxiv.org/pdf/2511.04758](https://arxiv.org/pdf/2511.04758)**

> **作者:** Caelan Garrett; Fabio Ramos
>
> **备注:** Project website: this https URL
>
> **摘要:** Bimanual and humanoid robots are appealing because of their human-like ability to leverage multiple arms to efficiently complete tasks. However, controlling multiple arms at once is computationally challenging due to the growth in the hybrid discrete-continuous action space. Task and Motion Planning (TAMP) algorithms can efficiently plan in hybrid spaces but generally produce plans, where only one arm is moving at a time, rather than schedules that allow for parallel arm motion. In order to extend TAMP to produce schedules, we present ScheduleStream, the first general-purpose framework for planning & scheduling with sampling operations. ScheduleStream models temporal dynamics using hybrid durative actions, which can be started asynchronously and persist for a duration that's a function of their parameters. We propose domain-independent algorithms that solve ScheduleStream problems without any application-specific mechanisms. We apply ScheduleStream to Task and Motion Planning & Scheduling (TAMPAS), where we use GPU acceleration within samplers to expedite planning. We compare ScheduleStream algorithms to several ablations in simulation and find that they produce more efficient solutions. We demonstrate ScheduleStream on several real-world bimanual robot tasks at this https URL.
>
---
#### [replaced 024] SM2ITH: Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，旨在解决动态环境中机器人与人类安全协作的问题。提出SM$^2$ITH框架，结合任务分层优化与人类行为预测，提升协调安全性与效率。**

- **链接: [https://arxiv.org/pdf/2511.17798](https://arxiv.org/pdf/2511.17798)**

> **作者:** Francesco D'Orazio; Sepehr Samavi; Xintong Du; Siqi Zhou; Giuseppe Oriolo; Angela P. Schoellig
>
> **备注:** Accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Mobile manipulators are designed to perform complex sequences of navigation and manipulation tasks in human-centered environments. While recent optimization-based methods such as Hierarchical Task Model Predictive Control (HTMPC) enable efficient multitask execution with strict task priorities, they have so far been applied mainly to static or structured scenarios. Extending these approaches to dynamic human-centered environments requires predictive models that capture how humans react to the actions of the robot. This work introduces Safe Mobile Manipulation with Interactive Human Prediction via Task-Hierarchical Bilevel Model Predictive Control (SM$^2$ITH), a unified framework that combines HTMPC with interactive human motion prediction through bilevel optimization that jointly accounts for robot and human dynamics. The framework is validated on two different mobile manipulators, the Stretch 3 and the Ridgeback-UR10, across three experimental settings: (i) delivery tasks with different navigation and manipulation priorities, (ii) sequential pick-and-place tasks with different human motion prediction models, and (iii) interactions involving adversarial human behavior. Our results highlight how interactive prediction enables safe and efficient coordination, outperforming baselines that rely on weighted objectives or open-loop human models.
>
---
#### [replaced 025] Contrastive Representation Regularization for Vision-Language-Action Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在提升模型对机器人信号的敏感性。通过引入RS-CL损失函数，增强控制相关表示学习，提高机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2510.01711](https://arxiv.org/pdf/2510.01711)**

> **作者:** Taeyoung Kim; Jimin Lee; Myungkyu Koo; Dongyoung Kim; Kyungmin Lee; Changyeon Kim; Younggyo Seo; Jinwoo Shin
>
> **备注:** ICML 2026
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong capabilities in robot manipulation by leveraging rich representations from pre-trained Vision-Language Models (VLMs). However, their representations arguably remain suboptimal, lacking sensitivity to robotic signals such as control actions and proprioceptive information. To address the issue, we introduce Robot State-aware Contrastive Loss (RS-CL), a simple and effective representation regularization for VLA models, designed to bridge the gap between VLM representations and robotic signals. In particular, RS-CL aligns the representations more closely with the robot's proprioceptive states by using relative distances between the states as soft supervision. Complementing the original action prediction objective, RS-CL enhances control-relevant representation learning, while being lightweight and fully compatible with standard VLA training pipelines. Our empirical results demonstrate that RS-CL substantially improves the performance of state-of-the-art VLA models; it pushes the prior art to 69.7% achieving the state-of-the-art performance on the RoboCasa-Kitchen benchmark, and boosts success rates from 45.0% to 58.3% on challenging real-robot manipulation tasks.
>
---
#### [replaced 026] Safety-Critical Adaptive Impedance Control via Nonsmooth Control Barrier Functions under State and Input Constraints
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全控制任务，解决不确定环境下机械臂的安全交互问题。提出一种自适应阻抗控制框架，结合非平滑控制屏障函数和模糊系统，确保状态与输入约束下的安全操作。**

- **链接: [https://arxiv.org/pdf/2605.28367](https://arxiv.org/pdf/2605.28367)**

> **作者:** Faisal Lawan; Xiaoran Han; Joaquin Carrasco; Barry Lennox; Xiaoxiao Cheng
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Safe physical interaction is critical for deploying robotic manipulators in human-robot interaction and contact-rich tasks, where uncertainty, external forces, and actuator limitations can compromise both performance and safety. We propose an online adaptive impedance control framework that enforces joint-state safety while achieving compliant interaction under uncertain dynamics. The approach combines a quadratic-program-based safety filter with a novel composed position-velocity non-smooth control barrier function (NCBF), enabling joint position and velocity constraints to be enforced through a unified relative-degree-one barrier. Unknown dynamics are compensated online using an interval type-2 fuzzy logic system, while actuator torque limits are handled through soft constraints with exact penalty recovery of feasible solutions. A disturbance-observer-enhanced safety mechanism improves robustness against modelling errors and external interaction forces. Using composite Lyapunov analysis, we prove forward invariance of the safe set and the uniform ultimately boundedness of the impedance-tracking error. Simulations on a 7-DOF manipulator with severe parametric uncertainty and external interaction wrenches demonstrate safe constraint satisfaction and robust impedance tracking.
>
---
#### [replaced 027] Learning A Simulation-based Visual Policy for Real-world Peg In Unseen Holes
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉控制任务，解决真实世界中未知形状的插孔问题。通过模拟训练和少量真实数据微调，实现高效泛化。**

- **链接: [https://arxiv.org/pdf/2205.04297](https://arxiv.org/pdf/2205.04297)**

> **作者:** Liang Xie; Hongxiang Yu; Kechun Xu; Tong Yang; Minhang Wang; Haojian Lu; Rong Xiong; Yue Wang
>
> **摘要:** This paper proposes a learning-based visual peg-in-hole that enables training with several shapes in simulation, and adapting to arbitrary unseen shapes in real world with minimal sim-to-real cost. The core idea is to decouple the generalization of the sensory-motor policy to the design of a fast-adaptable perception module and a simulated generic policy module. The framework consists of a segmentation network (SN), a virtual sensor network (VSN), and a controller network (CN). Concretely, the VSN is trained to measure the pose of the unseen shape from a segmented image. After that, given the shape-agnostic pose measurement, the CN is trained to achieve generic peg-in-hole. Finally, when applying to real unseen holes, we only have to fine-tune the SN required by the simulated VSN+CN. To further minimize the transfer cost, we propose to automatically collect and annotate the data for the SN after one-minute human teaching. Simulated and real-world results are presented under the configurations of eye-to/in-hand. An electric vehicle charging system with the proposed policy inside achieves a 10/10 success rate in 2-3s, using only hundreds of auto-labeled samples for the SN transfer.
>
---
#### [replaced 028] GaussianDream: A Feed-Forward 3D Gaussian World Model for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GaussianDream，用于机器人操作任务，解决3D空间建模不足的问题。通过3D高斯世界模型提升动作生成精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.20752](https://arxiv.org/pdf/2605.20752)**

> **作者:** Zijian Zhang; Yuqing Jiang; Qian Cheng; Xiaofan Li; Si Liu; Ding Zhao; Ping Luo; Weitao Zhou; Haibao Yu
>
> **备注:** 19 pages, 9 figures
>
> **摘要:** Vision-language-action (VLA) policies have advanced language-conditioned robotic manipulation by transferring semantic priors from pretrained vision-language models to action generation. However, standard action-imitation learning often lacks sufficient modeling of explicit 3D spatial information, dense geometric supervision, and future environment evolution, all critical for precise robotic interaction. To address this, we propose \textbf{GaussianDream}, a feed-forward 3D Gaussian world-model plug-in. Specifically, we introduce learnable GaussianDream Queries in the encoder, enabling the model to capture current-frame 3D spatial structure and short-horizon future evolution. During training, the latent GaussianDream prefix is processed by a static reconstruction head and a future prediction head to produce current 3D Gaussian scene states and future Gaussian evolution states. The current branch is supervised by RGB rendering and depth, while the future branch uses future RGB, depth, and pseudo 3D scene-flow signals. During inference, GaussianDream discards all auxiliary heads and retains only the learned prefix to condition action generation, without test-time Gaussian reconstruction or future prediction. Experimental results demonstrate that GaussianDream achieves state-of-the-art performance across multiple robotic manipulation benchmarks, reaching \textbf{98.4\%} on LIBERO, \textbf{54.8\%} on RoboCasa Human-50, and \textbf{50.0\%} on real-robot tasks. Compared with existing 3D-enhanced VLA methods, GaussianDream achieves strong accuracy while providing higher inference efficiency than video-based world-model approaches.
>
---
#### [replaced 029] When Should a Robot Think? Resource-Aware Reasoning via Reinforcement Learning for Embodied Robotic Decision-Making
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人决策任务，解决 embodied 代理在资源限制下何时推理、如何推理的问题。提出 RARRL 框架，通过强化学习实现自适应推理控制，提升任务成功率与系统效率。**

- **链接: [https://arxiv.org/pdf/2603.16673](https://arxiv.org/pdf/2603.16673)**

> **作者:** Jun Liu; Pu Zhao; Zhenglun Kong; Xuan Shen; Peiyan Dong; Fan Yang; Lin Cui; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Xue Lin; Gaowen Liu; Yanzhi Wang; Dong Huang
>
> **摘要:** Embodied robotic systems increasingly rely on large language model (LLM)-based agents to support high-level reasoning, planning, and decision-making during interactions with the environment. However, invoking LLM reasoning introduces substantial computational latency and resource overhead, which can interrupt action execution and reduce system reliability. Excessive reasoning may delay actions, while insufficient reasoning often leads to incorrect decisions and task failures. This raises a fundamental question for embodied agents: when should the agent reason, and when should it act? In this work, we propose RARRL (Resource-Aware Reasoning via Reinforcement Learning), a hierarchical framework for resource-aware orchestration of embodied agents. Rather than learning low-level control policies, RARRL learns a high-level orchestration policy that operates at the agent's decision-making layer. This policy enables the agent to adaptively determine whether to invoke reasoning, which reasoning role to employ, and how much computational budget to allocate based on current observations, execution history, and remaining resources. Extensive experiments, including evaluations with empirical latency profiles derived from the ALFRED benchmark, show that RARRL consistently improves task success rates while reducing execution latency and enhancing robustness compared with fixed or heuristic reasoning strategies. These results demonstrate that adaptive reasoning control is essential for building reliable and efficient embodied robotic agents.
>
---
#### [replaced 030] Enhancing Reinforcement Learning in 3D Environments through Semantic Segmentation: A Case Study in ViZDoom
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决3D环境中高维感知带来的内存消耗和部分可观测问题。通过引入语义分割的输入表示，降低内存使用并提升性能。**

- **链接: [https://arxiv.org/pdf/2511.11703](https://arxiv.org/pdf/2511.11703)**

> **作者:** Jin Huang
>
> **备注:** Master's Thesis at the University of Edinburgh (2024)
>
> **摘要:** Reinforcement learning (RL) in 3D environments with high-dimensional sensory input poses two major challenges: (1) the high memory consumption induced by memory buffers required to stabilise learning, and (2) the complexity of learning in partially observable Markov Decision Processes (POMDPs). This project addresses these challenges by proposing two novel input representations: SS-only and RGB+SS, both employing semantic segmentation on RGB colour images. Experiments were conducted in deathmatches of ViZDoom, utilizing perfect segmentation results for controlled evaluation. Our results showed that SS-only was able to reduce the memory consumption of memory buffers by at least 66.6%, and up to 98.6% when a vectorisable lossless compression technique with minimal overhead such as run-length encoding is applied. Meanwhile, RGB+SS significantly enhances RL agents' performance with the additional semantic information provided. Furthermore, we explored density-based heatmapping as a tool to visualise RL agents' movement patterns and evaluate their suitability for data collection. A brief comparison with a previous approach highlights how our method overcame common pitfalls in applying semantic segmentation in 3D environments like ViZDoom.
>
---
#### [replaced 031] Practical Insights on Grasp Strategies for Mobile Manipulation in the Wild
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究移动操作机器人在复杂环境中的抓取策略，旨在解决真实场景下可靠抓取的问题。作者开发了SHOPPER平台，在超市环境中测试并分析抓取失败原因，提出改进策略。**

- **链接: [https://arxiv.org/pdf/2504.12512](https://arxiv.org/pdf/2504.12512)**

> **作者:** Isabella Huang; Richard Cheng; Sangwoon Kim; Dan Kruse; Carolyn Chen; Lukas Kaul; JC Hancock; Shanmuga Harikumar; Mark Tjersland; James Borders; Dan Helmick
>
> **备注:** 8 pages, 8 figures, submitted to IROS 2025
>
> **摘要:** Mobile manipulation robots are continuously advancing, with their grasping capabilities rapidly progressing. However, there are still significant gaps preventing state-of-the-art mobile manipulators from widespread real-world deployments, including their ability to reliably grasp items in unstructured environments. To help bridge this gap, we developed SHOPPER, a mobile manipulation robot platform designed to push the boundaries of reliable and generalizable grasp strategies. We develop these grasp strategies and deploy them in a real-world grocery store -- an exceptionally challenging setting chosen for its vast diversity of manipulable items, fixtures, and layouts. In this work, we present our detailed approach to designing general grasp strategies towards picking any item in a real grocery store. Additionally, we provide an in-depth analysis of our latest real-world field test, discussing key findings related to fundamental failure modes over hundreds of distinct pick attempts. Through our detailed analysis, we aim to offer valuable practical insights and identify key grasping challenges, which can guide the robotics community towards pressing open problems in the field.
>
---
#### [replaced 032] Muscle Synergy Priors Enhance Biomechanical Fidelity in Predictive Musculoskeletal Locomotion Simulation
- **分类: cs.LG; cs.NE; cs.RO**

- **简介: 该论文属于人体运动模拟任务，旨在提升预测性骨科运动仿真的生物力学精度。通过引入肌肉协同控制，优化强化学习框架，实现更真实的运动表现。**

- **链接: [https://arxiv.org/pdf/2603.10474](https://arxiv.org/pdf/2603.10474)**

> **作者:** Ilseung Park; Eunsik Choi; Jangwhan Ahn; Jooeun Ahn
>
> **备注:** Added a manuscript footnote stating "Project page with supplementary videos: this https URL ."
>
> **摘要:** Human locomotion emerges from high-dimensional neuromuscular control, making predictive musculoskeletal simulation challenging. We present a physiology-informed reinforcement-learning framework that constrains control using muscle synergies. We extracted a low-dimensional synergy basis from inverse musculoskeletal analyses of a small set of overground walking trials and used it as the action space for a muscle-driven three-dimensional model trained across variable speeds, slopes and uneven terrain. The resulting controller generated stable gait from 0.7-1.8 m/s and on $\pm$ 6$^{\circ}$ grades and reproduced condition-dependent modulation of joint angles, joint moments and ground reaction forces. Compared with an unconstrained controller, synergy-constrained control reduced non-physiological knee kinematics and kept knee moment profiles within the experimental envelope. Across conditions, simulated vertical ground reaction forces correlated strongly with human measurements, and muscle-activation timing largely fell within inter-subject variability. These results show that embedding neurophysiological structure into reinforcement learning can improve biomechanical fidelity and generalization in predictive human locomotion simulation with limited experimental data.
>
---
#### [replaced 033] SurfFill: Completion of LiDAR Point Clouds via Gaussian Surfel Splatting
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于LiDAR点云补全任务，解决点云缺失问题。通过结合LiDAR与图像数据，利用高斯表面元进行点云补全，提升细节准确性。**

- **链接: [https://arxiv.org/pdf/2512.03010](https://arxiv.org/pdf/2512.03010)**

> **作者:** Svenja Strobel; Matthias Innmann; Bernhard Egger; Marc Stamminger; Linus Franke
>
> **备注:** Project page: this https URL
>
> **摘要:** LiDAR-captured point clouds are often considered the gold standard in active 3D reconstruction. While their accuracy is exceptional in flat regions, the capturing is susceptible to miss small geometric structures and may fail with dark, absorbent materials. Alternatively, capturing multiple photos of the scene and applying 3D photogrammetry can infer these details as they often represent feature-rich regions. However, the accuracy of LiDAR for featureless regions is rarely reached. Therefore, we suggest combining the strengths of LiDAR and camera-based capture by introducing SurfFill: a Gaussian surfel-based LiDAR completion scheme. We analyze LiDAR capturings and attribute LiDAR beam divergence as a main factor for artifacts, manifesting mostly at thin structures and edges. We use this insight to introduce an ambiguity heuristic for completed scans by evaluating the change in density in the point cloud. This allows us to identify points close to missed areas, which we can then use to grow additional points from to complete the scan. For this point growing, we constrain Gaussian surfel reconstruction to focus optimization and densification on these ambiguous areas. Finally, Gaussian primitives of the reconstruction in ambiguous areas are extracted and sampled for points to complete the point cloud. To address the challenges of large-scale reconstruction, we extend this pipeline with a divide-and-conquer scheme for building-sized point cloud completion. We evaluate on the task of LiDAR point cloud completion of synthetic and real-world scenes and find that our method outperforms previous reconstruction methods.
>
---
