# 机器人 cs.RO

- **最新发布 60 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] A Passive Elastic-Folding Mechanism for Stackable Airdrop Sensors
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于环境监测任务，旨在解决空中部署传感器的能耗与成本问题。提出一种被动弹性折叠机构，实现传感器的自动展开与稳定飞行。**

- **链接: [https://arxiv.org/pdf/2603.18861](https://arxiv.org/pdf/2603.18861)**

> **作者:** Damyon Kim; Yuichi Honjo; Tatsuya Iizuka; Naomi Okubo; Naoto Endo; Hiroshi Matsubara; Yoshihiro Kawahara; Naoto Morita; Takuya Sasatani
>
> **备注:** 8 pages, 8 figures, The 2026 IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Air-dispersed sensor networks deployed from aerial robotic systems (e.g., UAVs) provide a low-cost approach to wide-area environmental monitoring. However, existing methods often rely on active actuators for mid-air shape or trajectory control, increasing both power consumption and system cost. Here, we introduce a passive elastic-folding hinge mechanism that transforms sensors from a flat, stackable form into a three-dimensional structure upon release. Hinges are fabricated by laminating commercial sheet materials with rigid printed circuit boards (PCBs) and programming fold angles through a single oven-heating step, enabling scalable production without specialized equipment. Our geometric model links laminate geometry, hinge mechanics, and resulting fold angle, providing a predictive design methodology for target configurations. Laboratory tests confirmed fold angles between 10 degrees and 100 degrees, with a standard deviation of 4 degrees and high repeatability. Field trials further demonstrated reliable data collection and LoRa transmission during dispersion, while the Horizontal Wind Model (HWM)-based trajectory simulations indicated strong potential for wide-area sensing exceeding 10 km.
>
---
#### [new 002] Introducing M: A Modular, Modifiable Social Robot
- **分类: cs.RO; cs.HC**

- **简介: 该论文介绍M平台，解决社会机器人研究中平台复杂、难以修改和部署的问题。通过模块化设计和开源软件，提升研究效率，支持长期可重复的研究。**

- **链接: [https://arxiv.org/pdf/2603.19134](https://arxiv.org/pdf/2603.19134)**

> **作者:** Victor Nikhil Antony; Zhili Gong; Yoonjae Kim; Chien-Ming Huang
>
> **摘要:** We present M, an open-source, low-cost social robot platform designed to reduce platform friction that slows social robotics research by making robots easier to reproduce, modify, and deploy in real-world settings. M combines a modular mechanical design, multimodal sensing, and expressive yet mechanically simple actuation architecture with a ROS2-native software package that cleanly separates perception, expression control, and data management. The platform includes a simulation environment with interface equivalence to hardware to support rapid sim-to-real transfer of interaction behaviors. We demonstrate extensibility through additional sensing/actuation modules and provide example interaction templates for storytelling and two-way conversational coaching. Finally, we report real-world use in participatory design and week-long in-home deployments, showing how M can serve as a practical foundation for longitudinal, reproducible social robotics research.
>
---
#### [new 003] Multi-material Direct Ink Writing and Embroidery for Stretchable Wearable Sensors
- **分类: cs.RO**

- **简介: 该论文属于可穿戴传感器领域，解决智能服装中机械固定与电连接的问题。通过多材料直接墨水书写和刺绣结合，制备出高伸缩性传感器，实现运动检测与软体机器人应用。**

- **链接: [https://arxiv.org/pdf/2603.18354](https://arxiv.org/pdf/2603.18354)**

> **作者:** Lukas Cha; Ryman Hashem; Ria Prakash; Tanguy Declety; Wenze Zhang; Liang He
>
> **备注:** 6 pages, 8 figures, conference
>
> **摘要:** The development of wearable sensing systems for sports performance tracking, rehabilitation, and injury prevention has driven growing demand for smart garments that combine comfort, durability, and accurate motion detection. This paper presents a textile-compatible fabrication workflow that integrates multi-material direct ink writing with automated embroidery to create stretchable strain sensors directly embedded into garments. The process combines sequential multi-material printing of a silicone-carbon grease-silicone stack with automated embroidery that provides both mechanical fixation and electrical interfacing in a single step. The resulting hybrid sensor demonstrates stretchability up to 120% strain while maintaining electrical continuity, with approximately linear behaviour up to 60% strain (R^2 = 0.99), a gauge factor of 31.4, and hysteresis of 22.9%. Repeated loading-unloading tests over 80 cycles show baseline and peak drift of 0.135% and 0.236% per cycle, respectively, indicating moderate cycle-to-cycle stability. Mechanical testing further confirms that the silicone-fabric interface remains intact under large deformation, with failure occurring in the textile rather than at the stitched boundary. As a preliminary proof of concept, the sensor was integrated into wearable elbow and knee sleeves for joint angle monitoring, showing a clear correlation between normalised resistance change and bending angle. By addressing both mechanical fixation and electrical interfacing through embroidery-based integration, this approach provides a reproducible and scalable pathway for incorporating printed stretchable electronics into textile systems for motion capture and soft robotic applications.
>
---
#### [new 004] Robotic Agentic Platform for Intelligent Electric Vehicle Disassembly
- **分类: cs.RO**

- **简介: 该论文提出RAPID平台，解决电动汽车电池拆解自动化问题。通过视觉识别与AI控制，实现高效、灵活的拆解操作，提升回收效率。**

- **链接: [https://arxiv.org/pdf/2603.18520](https://arxiv.org/pdf/2603.18520)**

> **作者:** Zachary Allen; Max Conway; Lyle Antieau; Allen Ponraj; Nikolaus Correll
>
> **摘要:** Electric vehicles (EV) create an urgent need for scalable battery recycling, yet disassembly of EV battery packs remains largely manual due to high design variability. We present our Robotic Agentic Platform for Intelligent Disassembly (RAPID), designed to investigate perception-driven manipulation, flexible automation, and AI-assisted robot programming in realistic recycling scenarios. The system integrates a gantry-mounted industrial manipulator, RGB-D perception, and an automated nut-running tool for fastener removal on a full-scale EV battery pack. An open-vocabulary object detection pipeline achieves 0.9757 mAP50, enabling reliable identification of screws, nuts, busbars, and other components. We experimentally evaluate (n=204) three one-shot fastener removal strategies: taught-in poses (97% success rate, 24 min duration), one-shot vision execution (57%, 29 min), and visual servoing (83%, 36 min), comparing success rate and disassembly time for the battery's top cover fasteners. To support flexible interaction, we introduce agentic AI specifications for robotic disassembly tasks, allowing LLM agents to translate high-level instructions into robot actions through structured tool interfaces and ROS services. We evaluate SmolAgents with GPT-4o-mini and Qwen 3.5 9B/4B on edge hardware. Tool-based interfaces achieve 100% task completion, while automatic ROS service discovery shows 43.3% failure rates, highlighting the need for structured robot APIs for reliable LLM-driven control. This open-source platform enables systematic investigation of human-robot collaboration, agentic robot programming, and increasingly autonomous disassembly workflows, providing a practical foundation for research toward scalable robotic battery recycling.
>
---
#### [new 005] Fire as a Service: Augmenting Robot Simulators with Thermally and Visually Accurate Fire Dynamics
- **分类: cs.RO; cs.GR**

- **简介: 该论文属于机器人仿真任务，旨在解决现有模拟器缺乏真实火灾动态的问题。工作是提出FaaS框架，实现高效、真实的火灾模拟，提升机器人在火场环境中的训练与评估能力。**

- **链接: [https://arxiv.org/pdf/2603.19063](https://arxiv.org/pdf/2603.19063)**

> **作者:** Anton R. Wagner; Madhan Balaji Rao; Helge Wrede; Sören Pirk; Xuesu Xiao
>
> **摘要:** Most existing robot simulators prioritize rigid-body dynamics and photorealistic rendering, but largely neglect the thermally and optically complex phenomena that characterize real-world fire environments. For robots envisioned as future firefighters, this limitation hinders both reliable capability evaluation and the generation of representative training data prior to deployment in hazardous scenarios. To address these challenges, we introduce Fire as a Service (FaaS), a novel, asynchronous co-simulation framework that augments existing robot simulators with high-fidelity and computationally efficient fire simulations. Our pipeline enables robots to experience accurate, multi-species thermodynamic heat transfer and visually consistent volumetric smoke without disrupting high-frequency rigid-body control loops. We demonstrate that our framework can be integrated with diverse robot simulators to generate physically accurate fire behavior, benchmark thermal hazards encountered by robotic platforms, and collect realistic multimodal perceptual data. Crucially, its real-time performance supports human-in-the-loop teleoperation, enabling the successful training of reactive, multimodal policies via Behavioral Cloning. By adding fire dynamics to robot simulations, FaaS provides a scalable pathway toward safer, more reliable deployment of robots in fire scenarios.
>
---
#### [new 006] Shifting Uncertainty to Critical Moments: Towards Reliable Uncertainty Quantification for VLA Model
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型中不确定性量化不可靠的问题。通过改进不确定性信号的提取与分析方法，提升失败预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.18342](https://arxiv.org/pdf/2603.18342)**

> **作者:** Yanchuan Tang; Taowen Wang; Yuefei Chen; Boxuan Zhang; Qiang Guan; Ruixiang Tang
>
> **摘要:** Vision-Language-Action (VLA) models enable general-purpose robotic policies by mapping visual observations and language instructions to low-level actions, but they often lack reliable introspection. A common practice is to compute a token-level uncertainty signal and take its mean over a rollout. However, mean aggregation can dilute short-lived but safety-critical uncertainty spikes in continuous control. In particular, successful rollouts may contain localized high-entropy segments due to benign noise or non-critical micro-adjustments, while failure rollouts can appear low-entropy for most timesteps and only exhibit brief spikes near the onset of failure. We propose a unified uncertainty quantification approach for predicting rollout success versus failure that (1) uses max-based sliding window pooling to preserve transient risk signals, (2) applies motion-aware stability weighting to emphasize high-frequency action oscillations associated with unstable behaviors, and (3) performs DoF-adaptive calibration via Bayesian Optimization to prioritize kinematically critical axes. Experiments on the LIBERO benchmark show that our method substantially improves failure prediction accuracy and yields more reliable signals for failure detection, which can support downstream human-in-the-loop interventions.
>
---
#### [new 007] Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言-动作模型的机制，探讨其如何将多模态输入转化为动作。通过分析不同模型，揭示视觉路径主导动作生成，语言敏感性依赖任务结构，且不同路径编码不同信息。**

- **链接: [https://arxiv.org/pdf/2603.19233](https://arxiv.org/pdf/2603.19233)**

> **作者:** Bryce Grant; Xijia Zhao; Peng Wang
>
> **备注:** Accepted to Multimodal Intelligence Workshop @ ICLR
>
> **摘要:** Vision-Language-Action (VLA) models combine perception, language, and motor control in a single architecture, yet how they translate multimodal inputs into actions remains poorly understood. We apply activation injection, sparse autoencoders (SAEs), and linear probes to six models spanning 80M--7B parameters across 394,000+ rollout episodes on four benchmarks. The visual pathway dominates action generation across all architectures: injecting baseline activations into null-prompt episodes recovers near-identical behavior, while cross-task injection steers robots toward source-task positions (99.8\% of X-VLA episodes align with the source trajectory), exposing spatially bound motor programs tied to scene coordinates rather than abstract task representations. Language sensitivity depends on task structure, not model design: when visual context uniquely specifies the task, language is ignored; when multiple goals share a scene, language becomes essential (X-VLA \texttt{libero\_goal}: 94\%$\to$10\% under wrong prompts vs.\ \texttt{libero\_object}: 60--100\% regardless). In all three multi-pathway architectures (\pizhalf{}, SmolVLA, GR00T), expert pathways encode motor programs while VLM pathways encode goal semantics ($2\times$ greater behavioral displacement from expert injection), and subspace injection confirms these occupy separable activation subspaces. Per-token SAE processing is essential for action fidelity on most architectures, though mean-pooling improves fidelity on X-VLA. Contrastive identification recovers 82+ manipulation concepts, and causal ablation reveals sensitivity spanning 28--92\% zero-effect rates independent of representation width. We release \textbf{Action Atlas} (this https URL) for interactive exploration of VLA representations across all six models.
>
---
#### [new 008] Uncovering Latent Phase Structures and Branching Logic in Locomotion Policies: A Case Study on HalfCheetah
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究深度强化学习在运动控制中的决策机制，旨在揭示策略的相位结构与分支逻辑。通过分析HalfCheetah任务，发现策略能自主形成可解释的周期性相位和分支决策。**

- **链接: [https://arxiv.org/pdf/2603.18084](https://arxiv.org/pdf/2603.18084)**

> **作者:** Daisuke Yasui; Toshitaka Matsuki; Hiroshi Sato
>
> **备注:** Accepted at XAI-2026: The 4th World Conference on eXplainable Artificial Intelligence
>
> **摘要:** In locomotion control tasks, Deep Reinforcement Learning (DRL) has demonstrated high performance; however, the decision-making process of the learned policy remains a black box, making it difficult for humans to understand. On the other hand, in periodic motions such as walking, it is well known that implicit motion phases exist, such as the stance phase and the swing phase. Focusing on this point, this study hypothesizes that a policy trained for locomotion control may also represent a phase structure that is interpretable by humans. To examine this hypothesis in a controlled setting, we consider a locomotion task that is amenable to observing whether a policy autonomously acquires temporally structured phases through interaction with the environment. To verify this hypothesis, in the MuJoCo locomotion benchmark HalfCheetah-v5, the state transition sequences acquired by a policy trained for walking control through interaction with the environment were aggregated into semantic phases based on state similarity and consistency of subsequent transitions. As a result, we demonstrated that the state sequences generated by the trained policy exhibit periodic phase transition structures as well as phase branching. Furthermore, by approximating the states and actions corresponding to each semantic phase using Explainable Boosting Machines (EBMs), we analyzed phase-dependent decision making-namely, which state features the policy function attends to and how it controls action outputs in each phase. These results suggest that neural network-based policies, which are often regarded as black boxes, can autonomously acquire interpretable phase structures and logical branching mechanisms.
>
---
#### [new 009] CSSDF-Net: Safe Motion Planning Based on Neural Implicit Representations of Configuration Space Distance Field
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决高维环境下安全路径生成问题。提出CSSDF-Net，通过学习配置空间的符号距离场，实现高效碰撞检测与梯度查询。**

- **链接: [https://arxiv.org/pdf/2603.18669](https://arxiv.org/pdf/2603.18669)**

> **作者:** Haohua Chen; Yixuan Zhou; Yifan Zhou; Hesheng Wang
>
> **摘要:** High-dimensional manipulator operation in unstructured environments requires a differentiable, scene-agnostic distance query mechanism to guide safe motion generation. Existing geometric collision checkers are typically non-differentiable, while workspace-based implicit distance models are hindered by the highly nonlinear workspace--configuration mapping and often suffer from poor convergence; moreover, self-collision and environment collision are commonly handled as separate constraints. We propose Configuration-Space Signed Distance Field-Net (CSSDF-Net), which learns a continuous signed distance field directly in configuration space to provide joint-space distance and gradient queries under a unified geometric notion of safety. To enable zero-shot generalization without environment-specific retraining, we introduce a spatial-hashing-based data generation pipeline that encodes robot-centric geometric priors and supports efficient retrieval of risk configurations for arbitrary obstacle point sets. The learned distance field is integrated into safety-constrained trajectory optimization and receding-horizon MPC, enabling both offline planning and online reactive avoidance. Experiments on a planar arm and a 7-DoF manipulator demonstrate stable gradients, effective collision avoidance in static and dynamic scenes, and practical inference latency for large-scale point-cloud queries, supporting deployment in previously unseen environments.
>
---
#### [new 010] ATG-MoE: Autoregressive trajectory generation with mixture-of-experts for assembly skill learning
- **分类: cs.RO**

- **简介: 该论文属于机器人装配技能学习任务，旨在解决传统编程和现有方法在灵活性、泛化能力和多技能集成上的不足。提出ATG-MoE模型，实现端到端轨迹生成与多技能融合。**

- **链接: [https://arxiv.org/pdf/2603.19029](https://arxiv.org/pdf/2603.19029)**

> **作者:** Weihang Huang; Chaoran Zhang; Xiaoxin Deng; Hao Zhou; Zhaobo Xu; Shubo Cui; Long Zeng
>
> **备注:** 32 pages, 13 figures
>
> **摘要:** Flexible manufacturing requires robot systems that can adapt to constantly changing tasks, objects, and environments. However, traditional robot programming is labor-intensive and inflexible, while existing learning-based assembly methods often suffer from weak positional generalization, complex multi-stage designs, and limited multi-skill integration capability. To address these issues, this paper proposes ATG-MoE, an end-to-end autoregressive trajectory generation method with mixture of experts for assembly skill learning from demonstration. The proposed method establishes a closed-loop mapping from multi-modal inputs, including RGB-D observations, natural language instructions, and robot proprioception to manipulation trajectories. It integrates multi-modal feature fusion for scene and task understanding, autoregressive sequence modeling for temporally coherent trajectory generation, and a mixture-of-experts architecture for unified multi-skill learning. In contrast to conventional methods that separate visual perception and control or train different skills independently, ATG-MoE directly incorporates visual information into trajectory generation and supports efficient multi-skill integration within a single model. We train and evaluate the proposed method on eight representative assembly skills from a pressure-reducing valve assembly task. Experimental results show that ATG-MoE achieves strong overall performance in simulation, with an average grasp success rate of 96.3% and an average overall success rate of 91.8%, while also demonstrating strong generalization and effective multi-skill integration. Real-world experiments further verify its practicality for multi-skill industrial assembly. The project page can be found at this https URL
>
---
#### [new 011] ADMM-Based Distributed MPC with Control Barrier Functions for Safe Multi-Robot Quadrupedal Locomotion
- **分类: cs.RO; math.OC**

- **简介: 该论文属于多机器人安全轨迹规划任务，解决分布式控制中的安全与效率问题。通过ADMM和CBF结合，实现去中心化轨迹优化，提升计算效率与安全性。**

- **链接: [https://arxiv.org/pdf/2603.19170](https://arxiv.org/pdf/2603.19170)**

> **作者:** Yicheng Zeng; Ruturaj S. Sambhus; Basit Muhammad Imran; Jeeseop Kim; Vittorio Pastore; Kaveh Akbari Hamed
>
> **摘要:** This paper proposes a fully decentralized model predictive control (MPC) framework with control barrier function (CBF) constraints for safety-critical trajectory planning in multi-robot legged systems. The incorporation of CBF constraints introduces explicit inter-agent coupling, which prevents direct decomposition of the resulting optimal control problems. To address this challenge, we reformulate the centralized safety-critical MPC problem using a structured distributed optimization framework based on the alternating direction method of multipliers (ADMM). By introducing a novel node-edge splitting formulation with consensus constraints, the proposed approach decomposes the global problem into independent node-local and edge-local quadratic programs that can be solved in parallel using only neighbor-to-neighbor communication. This enables fully decentralized trajectory optimization with symmetric computational load across agents while preserving safety and dynamic feasibility. The proposed framework is integrated into a hierarchical locomotion control architecture for quadrupedal robots, combining high-level distributed trajectory planning, mid-level nonlinear MPC enforcing single rigid body dynamics, and low-level whole-body control enforcing full-order robot dynamics. The effectiveness of the proposed approach is demonstrated through hardware experiments on two Unitree Go2 quadrupedal robots and numerical simulations involving up to four robots navigating uncertain environments with rough terrain and external disturbances. The results show that the proposed distributed formulation achieves performance comparable to centralized MPC while reducing the average per-cycle planning time by up to 51% in the four-agent case, enabling efficient real-time decentralized implementation.
>
---
#### [new 012] TiBCLaG: A Trigger-induced Bistable Compliant Laparoscopic Grasper
- **分类: cs.RO**

- **简介: 该论文属于医疗机械设计任务，旨在解决传统腹腔镜抓钳成本高、结构复杂的问题。提出一种单体、柔性、双稳态的抓钳结构，通过3D打印实现可靠抓取。**

- **链接: [https://arxiv.org/pdf/2603.18559](https://arxiv.org/pdf/2603.18559)**

> **作者:** Joel J Nellikkunnel; Prabhat Kumar
>
> **备注:** 17 pages, 13 figures
>
> **摘要:** Industrial laparoscopic graspers use multi-link rigid mechanisms manufactured to tight tolerances, resulting in high manufacturing and assembly costs. This work presents the design and proof-of-concept validation of a monolithic, fully compliant, bistable, laparoscopic grasper that eliminates the need for multiple rigid links, thereby reducing part count. The device integrates a compliant trigger and a compliant gripper end-effector, coupled via a control push-rod, to achieve stable grasping without continuous user input. The trigger mechanism is synthesized using a Two-Element Beam Constraint Model as a design framework to control the deformation and stiffness of V-beam-like elements. This technique enables elastic energy storage while preventing snap-through instability. The end-effector is designed as a compliant gripper to achieve adaptive grasping through elastic deformation. Jaws' opening-and-closing performance is demonstrated using nonlinear finite element analysis. The laparoscopic design presented here is fabricated using fused deposition 3D printing. The fabricated prototype demonstrates reliable bistable actuation, confirming the feasibility of such compliant laparoscopic grasper architectures.
>
---
#### [new 013] Meanings and Measurements: Multi-Agent Probabilistic Grounding for Vision-Language Navigation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于视觉语言导航任务，解决复杂语义-度量语言查询的定位问题。提出MAPG框架，通过多智能体概率组合实现精准决策。**

- **链接: [https://arxiv.org/pdf/2603.19166](https://arxiv.org/pdf/2603.19166)**

> **作者:** Swagat Padhan; Lakshya Jain; Bhavya Minesh Shah; Omkar Patil; Thao Nguyen; Nakul Gopalan
>
> **备注:** Equal contribution: Swagat Padhan and Lakshya Jain, 9 pages, 6 figures, paper website: this https URL
>
> **摘要:** Robots collaborating with humans must convert natural language goals into actionable, physically grounded decisions. For example, executing a command such as "go two meters to the right of the fridge" requires grounding semantic references, spatial relations, and metric constraints within a 3D scene. While recent vision language models (VLMs) demonstrate strong semantic grounding capabilities, they are not explicitly designed to reason about metric constraints in physically defined spaces. In this work, we empirically demonstrate that state-of-the-art VLM-based grounding approaches struggle with complex metric-semantic language queries. To address this limitation, we propose MAPG (Multi-Agent Probabilistic Grounding), an agentic framework that decomposes language queries into structured subcomponents and queries a VLM to ground each component. MAPG then probabilistically composes these grounded outputs to produce metrically consistent, actionable decisions in 3D space. We evaluate MAPG on the HM-EQA benchmark and show consistent performance improvements over strong baselines. Furthermore, we introduce a new benchmark, MAPG-Bench, specifically designed to evaluate metric-semantic goal grounding, addressing a gap in existing language grounding evaluations. We also present a real-world robot demonstration showing that MAPG transfers beyond simulation when a structured scene representation is available.
>
---
#### [new 014] Benchmarking Visual Feature Representations for LiDAR-Inertial-Visual Odometry Under Challenging Conditions
- **分类: cs.RO**

- **简介: 该论文属于视觉里程计任务，解决复杂环境下定位精度不足的问题。通过融合LiDAR、IMU和相机数据，提出混合方法提升视觉特征匹配的鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.18589](https://arxiv.org/pdf/2603.18589)**

> **作者:** Eunseon Choi; Junwoo Hong; Daehan Lee; Sanghyun Park; Hyunyoung Jo; Sunyoung Kim; Changho Kang; Seongsam Kim; Yonghan Jung; Jungwook Park; Seul Koo; Soohee Han
>
> **备注:** 14 pages, Publised IEEE Access2026
>
> **摘要:** Accurate localization in autonomous driving is critical for successful missions including environmental mapping and survivor searches. In visually challenging environments, including low-light conditions, overexposure, illumination changes, and high parallax, the performance of conventional visual odometry methods significantly degrade undermining robust robotic navigation. Researchers have recently proposed LiDAR-inertial-visual odometry (LIVO) frameworks, that integrate LiDAR, IMU, and camera sensors, to address these challenges. This paper extends the FAST-LIVO2-based framework by introducing a hybrid approach that integrates direct photometric methods with descriptor-based feature matching. For the descriptor-based feature matching, this work proposes pairs of ORB with the Hamming distance, SuperPoint with SuperGlue, SuperPoint with LightGlue, and XFeat with the mutual nearest neighbor. The proposed configurations are benchmarked by accuracy, computational cost, and feature tracking stability, enabling a quantitative comparison of the adaptability and applicability of visual descriptors. The experimental results reveal that the proposed hybrid approach outperforms the conventional sparse-direct method. Although the sparse-direct method often fails to converge in regions where photometric inconsistency arises due to illumination changes, the proposed approach still maintains robust performance under the same conditions. Furthermore, the hybrid approach with learning-based descriptors enables robust and reliable visual state estimation across challenging environments.
>
---
#### [new 015] Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作模型的可解释性研究，旨在解决模型泛化能力不足的问题。通过训练稀疏自编码器，提取可解释且可操控的特征，提升模型的通用性。**

- **链接: [https://arxiv.org/pdf/2603.19183](https://arxiv.org/pdf/2603.19183)**

> **作者:** Aiden Swann; Lachlain McGranahan; Hugo Buurmeijer; Monroe Kennedy III; Mac Schwager
>
> **备注:** 25 pages, 12 figures
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising approach for general-purpose robot manipulation. However, their generalization is inconsistent: while these models can perform impressively in some settings, fine-tuned variants often fail on novel objects, scenes, and instructions. We apply mechanistic interpretability techniques to better understand the inner workings of VLA models. To probe internal representations, we train Sparse Autoencoders (SAEs) on hidden layer activations of the VLA. SAEs learn a sparse dictionary whose features act as a compact, interpretable basis for the model's computation. We find that the large majority of extracted SAE features correspond to memorized sequences from specific training demonstrations. However, some features correspond to interpretable, general, and steerable motion primitives and semantic properties, offering a promising glimpse toward VLA generalizability. We propose a metric to categorize features according to whether they represent generalizable transferable primitives or episode-specific memorization. We validate these findings through steering experiments on the LIBERO benchmark. We show that individual SAE features causally influence robot behavior. Steering general features induces behaviors consistent with their semantic meaning and can be applied across tasks and scenes. This work provides the first mechanistic evidence that VLAs can learn generalizable features across tasks and scenes. We observe that supervised fine-tuning on small robotics datasets disproportionately amplifies memorization. In contrast, training on larger, more diverse datasets (e.g., DROID) or using knowledge insulation promotes more general features. We provide an open-source codebase and user-friendly interface for activation collection, SAE training, and feature steering. Our project page is located at this http URL
>
---
#### [new 016] OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于接触丰富机器人操作任务，旨在解决触觉信息利用不足的问题。提出OmniVTA框架，结合视觉与触觉信息进行接触建模和控制，提升操作精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.19201](https://arxiv.org/pdf/2603.19201)**

> **作者:** Yuhang Zheng; Songen Gu; Weize Li; Yupeng Zheng; Yujie Zang; Shuai Tian; Xiang Li; Ruihai Wu; Ce Hao; Chen Gao; Si Liu; Haoran Li; Yilun Chen; Shuicheng Yan; Wenchao Ding
>
> **备注:** TARS Robotics Project Page: this https URL
>
> **摘要:** Contact-rich manipulation tasks, such as wiping and assembly, require accurate perception of contact forces, friction changes, and state transitions that cannot be reliably inferred from vision alone. Despite growing interest in visuo-tactile manipulation, progress is constrained by two persistent limitations: existing datasets are small in scale and narrow in task coverage, and current methods treat tactile signals as passive observations rather than using them to model contact dynamics or enable closed-loop control explicitly. In this paper, we present \textbf{OmniViTac}, a large-scale visuo-tactile-action dataset comprising $21{,}000+$ trajectories across $86$ tasks and $100+$ objects, organized into six physics-grounded interaction patterns. Building on this dataset, we propose \textbf{OmniVTA}, a world-model-based visuo-tactile manipulation framework that integrates four tightly coupled modules: a self-supervised tactile encoder, a two-stream visuo-tactile world model for predicting short-horizon contact evolution, a contact-aware fusion policy for action generation, and a 60Hz reflexive controller that corrects deviations between predicted and observed tactile signals in a closed loop. Real-robot experiments across all six interaction categories show that OmniVTA outperforms existing methods and generalizes well to unseen objects and geometric configurations, confirming the value of combining predictive contact modeling with high-frequency tactile feedback for contact-rich manipulation. All data, models, and code will be made publicly available on the project website at this https URL.
>
---
#### [new 017] Proprioceptive-only State Estimation for Legged Robots with Set-Coverage Measurements of Learned Dynamics
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，解决腿式机器人在无感知条件下准确估计状态的问题。通过引入集覆盖噪声模型，提升估计的鲁棒性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.18308](https://arxiv.org/pdf/2603.18308)**

> **作者:** Abhijeet M. Kulkarni; Ioannis Poulakakis; Guoquan Huang
>
> **摘要:** Proprioceptive-only state estimation is attractive for legged robots since it is computationally cheaper and is unaffected by perceptually degraded conditions. The history of joint-level measurements contains rich information that can be used to infer the dynamics of the system and subsequently produce navigational measurements. Recent approaches produce these estimates with learned measurement models and fuse with IMU data, under a Gaussian noise assumption. However, this assumption can easily break down with limited training data and render the estimates inconsistent and potentially divergent. In this work, we propose a proprioceptive-only state estimation framework for legged robots that characterizes the measurement noise using set-coverage statements that do not assume any distribution. We develop a practical and computationally inexpensive method to use these set-coverage measurements with a Gaussian filter in a systematic way. We validate the approach in both simulation and two real-world quadrupedal datasets. Comparison with the Gaussian baselines shows that our proposed method remains consistent and is not prone to drift under real noise scenarios.
>
---
#### [new 018] Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人视觉语言动作（VLA）模型的强化学习任务，旨在解决真实世界与仿真环境间的泛化问题。通过生成式3D世界提升仿真场景多样性，实现高效且泛化的策略训练与成功迁移到现实环境。**

- **链接: [https://arxiv.org/pdf/2603.18532](https://arxiv.org/pdf/2603.18532)**

> **作者:** Andrew Choi; Xinjie Wang; Zhizhong Su; Wei Xu
>
> **摘要:** The strong performance of large vision-language models (VLMs) trained with reinforcement learning (RL) has motivated similar approaches for fine-tuning vision-language-action (VLA) models in robotics. Many recent works fine-tune VLAs directly in the real world to avoid addressing the sim-to-real gap. While real-world RL circumvents sim-to-real issues, it inherently limits the generality of the resulting VLA, as scaling scene and object diversity in the physical world is prohibitively difficult. This leads to the paradoxical outcome of transforming a broadly pretrained model into an overfitted, scene-specific policy. Training in simulation can instead provide access to diverse scenes, but designing those scenes is also costly. In this work, we show that VLAs can be RL fine-tuned without sacrificing generality and with reduced labor by leveraging 3D world generative models. Using these models together with a language-driven scene designer, we generate hundreds of diverse interactive scenes containing unique objects and backgrounds, enabling scalable and highly parallel policy learning. Starting from a pretrained imitation baseline, our approach increases simulation success from 9.7% to 79.8% while achieving a 1.25$\times$ speedup in task completion time. We further demonstrate successful sim-to-real transfer enabled by the quality of the generated digital twins together with domain randomization, improving real-world success from 21.7% to 75% and achieving a 1.13$\times$ speedup. Finally, we further highlight the benefits of leveraging the effectively unlimited data from 3D world generative models through an ablation study showing that increasing scene diversity directly improves zero-shot generalization.
>
---
#### [new 019] ROFT-VINS: Robust Feature Tracking-based Visual-Inertial State Estimation for Harsh Environment
- **分类: cs.RO**

- **简介: 该论文属于视觉-惯性状态估计任务，旨在解决恶劣环境下特征跟踪不稳定的问题。通过深度学习方法提升单目相机的特征跟踪鲁棒性，并集成到VINS-Fusion系统中进行验证。**

- **链接: [https://arxiv.org/pdf/2603.18746](https://arxiv.org/pdf/2603.18746)**

> **作者:** Sanghyun Park; Soohee Han
>
> **备注:** 6 pages, published ICCAS 2024
>
> **摘要:** SLAM (Simultaneous Localization and Mapping) and Odometry are important systems for estimating the position of mobile devices, such as robots and cars, utilizing one or more sensors. Particularly in camera-based SLAM or Odometry, effectively tracking visual features is important as it significantly impacts system performance. In this paper, we propose a method that leverages deep learning to robustly track visual features in monocular camera images. This method operates reliably even in textureless environments and situations with rapid lighting changes. Additionally, we evaluate the performance of our proposed method by integrating it into VINS-Fusion (Monocular-Inertial), a commonly used Visual-Inertial Odometry (VIO) system.
>
---
#### [new 020] V-Dreamer: Automating Robotic Simulation and Trajectory Synthesis via Video Generation Priors
- **分类: cs.RO**

- **简介: 该论文提出V-Dreamer，解决机器人仿真与轨迹生成问题，通过自然语言生成多样化、物理真实的环境和轨迹，提升机器人学习效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.18811](https://arxiv.org/pdf/2603.18811)**

> **作者:** Songjia He; Zixuan Chen; Hongyu Ding; Dian Shao; Jieqi Shi; Chenxu Li; Jing Huo; Yang Gao
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Training generalist robots demands large-scale, diverse manipulation data, yet real-world collection is prohibitively expensive, and existing simulators are often constrained by fixed asset libraries and manual heuristics. To bridge this gap, we present V-Dreamer, a fully automated framework that generates open-vocabulary, simulation-ready manipulation environments and executable expert trajectories directly from natural language instructions. V-Dreamer employs a novel generative pipeline that constructs physically grounded 3D scenes using large language models and 3D generative models, validated by geometric constraints to ensure stable, collision-free layouts. Crucially, for behavior synthesis, we leverage video generation models as rich motion priors. These visual predictions are then mapped into executable robot trajectories via a robust Sim-to-Gen visual-kinematic alignment module utilizing CoTracker3 and VGGT. This pipeline supports high visual diversity and physical fidelity without manual intervention. To evaluate the generated data, we train imitation learning policies on synthesized trajectories encompassing diverse object and environment variations. Extensive evaluations on tabletop manipulation tasks using the Piper robotic arm demonstrate that our policies robustly generalize to unseen objects in simulation and achieve effective sim-to-real transfer, successfully manipulating novel real-world objects.
>
---
#### [new 021] "You've got a friend in me": Co-Designing a Peer Social Robot for Young Newcomers' Language and Cultural Learning
- **分类: cs.RO**

- **简介: 该论文属于教育技术任务，旨在解决社区语言学习中师资不足的问题。通过协作设计，开发了辅助教学的社交机器人Maple，支持儿童英语和文化学习。**

- **链接: [https://arxiv.org/pdf/2603.18804](https://arxiv.org/pdf/2603.18804)**

> **作者:** Neil Fernandes; Cheng Tang; Tehniyat Shahbaz; Alex Hauschildt; Emily Davies-Robinson; Yue Hu; Kerstin Dautenhahn
>
> **摘要:** Community literacy programs supporting young newcomer children in Canada face limited staffing and scarce one-to-one time, which constrains personalized English and cultural learning support. This paper reports on a co-design study with United for Literacy tutors that informed Maple, a table-top, peer-like Socially Assistive Robot (SAR) designed as a practice partner within tutor-mediated sessions. From shadowing and co-design interviews, we derived newcomer-specific requirements and added them in an integrated prototype that uses short story-based activities, multi-modal scaffolding (speech, facial feedback, gesture), and embedded quizzes that support attention while producing tutor-actionable formative signals. We contribute system design implications for tutor-in-the-loop SARs supporting language socialization in community settings and outline directions for child-centered evaluation in authentic programs.
>
---
#### [new 022] Graph-of-Constraints Model Predictive Control for Reactive Multi-agent Task and Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于多智能体任务与运动规划（TAMP）领域，解决动态任务分配和部分有序约束下的协同问题。提出GoC-MPC框架，实现在线适应和扰动恢复。**

- **链接: [https://arxiv.org/pdf/2603.18400](https://arxiv.org/pdf/2603.18400)**

> **作者:** Anastasios Manganaris; Jeremy Lu; Ahmed H. Qureshi; Suresh Jagannathan
>
> **备注:** 8 main content pages, 4 main content figures, camera ready version submitted to IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Sequences of interdependent geometric constraints are central to many multi-agent Task and Motion Planning (TAMP) problems. However, existing methods for handling such constraint sequences struggle with partially ordered tasks and dynamic agent assignments. They typically assume static assignments and cannot adapt when disturbances alter task allocations. To overcome these limitations, we introduce Graph-of-Constraints Model Predictive Control (GoC-MPC), a generalized sequence-of-constraints framework integrated with MPC. GoC-MPC naturally supports partially ordered tasks, dynamic agent coordination, and disturbance recovery. By defining constraints over tracked 3D keypoints, our method robustly solves diverse multi-agent manipulation tasks-coordinating agents and adapting online from visual observations alone, without relying on training data or environment models. Experiments demonstrate that GoC-MPC achieves higher success rates, significantly faster TAMP computation, and shorter overall paths compared to recent baselines, establishing it as an efficient and robust solution for multi-agent manipulation under real-world disturbances. Our supplementary video and code can be found at this https URL .
>
---
#### [new 023] FASTER: Rethinking Real-Time Flow VLAs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型的实时执行任务，旨在解决反应延迟问题。通过重新设计动作采样策略，提出FASTER方法，显著降低反应时间，提升机器人实时响应能力。**

- **链接: [https://arxiv.org/pdf/2603.19199](https://arxiv.org/pdf/2603.19199)**

> **作者:** Yuxiang Lu; Zhe Liu; Xianzhe Fan; Zhenya Yang; Jinghua Hou; Junyi Li; Kaixin Ding; Hengshuang Zhao
>
> **备注:** Project page: this https URL
>
> **摘要:** Real-time execution is crucial for deploying Vision-Language-Action (VLA) models in the physical world. Existing asynchronous inference methods primarily optimize trajectory smoothness, but neglect the critical latency in reacting to environmental changes. By rethinking the notion of reaction in action chunking policies, this paper presents a systematic analysis of the factors governing reaction time. We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon. Moreover, we reveal that the standard practice of applying a constant schedule in flow-based VLAs can be inefficient and forces the system to complete all sampling steps before any movement can start, forming the bottleneck in reaction latency. To overcome this issue, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER). By introducing a Horizon-Aware Schedule, FASTER adaptively prioritizes near-term actions during flow sampling, compressing the denoising of the immediate reaction by tenfold (e.g., in $\pi_{0.5}$ and X-VLA) into a single step, while preserving the quality of long-horizon trajectory. Coupled with a streaming client-server pipeline, FASTER substantially reduces the effective reaction latency on real robots, especially when deployed on consumer-grade GPUs. Real-world experiments, including a highly dynamic table tennis task, prove that FASTER unlocks unprecedented real-time responsiveness for generalist policies, enabling rapid generation of accurate and smooth trajectories.
>
---
#### [new 024] Manufacturing Micro-Patterned Surfaces with Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文属于微结构制造任务，解决传统方法难以规模化生产的问题。通过多机器人系统与熵控算法，实现高效、分布式微图案制造。**

- **链接: [https://arxiv.org/pdf/2603.18260](https://arxiv.org/pdf/2603.18260)**

> **作者:** Annalisa T. Taylor; Malachi Landis; Ping Guo; Todd D. Murphey
>
> **摘要:** Applying micro-patterns to surfaces has been shown to impart useful physical properties such as drag reduction and hydrophobicity. However, current manufacturing techniques cannot produce micro-patterned surfaces at scale due to high-cost machinery and inefficient coverage techniques such as raster-scanning. In this work, we use multiple robots, each equipped with a patterning tool, to manufacture these surfaces. To allow these robots to coordinate during the patterning task, we use the ergodic control algorithm, which specifies coverage objectives using distributions. We demonstrate that robots can divide complicated coverage objectives by communicating compressed representations of their trajectory history both in simulations and experimental trials. Further, we show that robot-produced patterning can lower the coefficient of friction of metallic surfaces. This work demonstrates that distributed multi-robot systems can coordinate to manufacture products that were previously unrealizable at scale.
>
---
#### [new 025] Empathetic Motion Generation for Humanoid Educational Robots via Reasoning-Guided Vision--Language--Motion Diffusion Architecture
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在提升教育机器人生成符合教学意图的肢体动作。通过融合视觉、语言和运动信息，生成结构化且具表现力的教姿。**

- **链接: [https://arxiv.org/pdf/2603.18771](https://arxiv.org/pdf/2603.18771)**

> **作者:** Fuze Sun; Lingyu Li; Lekan Dai; Xinyu Fan
>
> **摘要:** This article suggests a reasoning-guided vision-language-motion diffusion framework (RG-VLMD) for generating instruction-aware co-speech gestures for humanoid robots in educational scenarios. The system integrates multi-modal affective estimation, pedagogical reasoning, and teaching-act-conditioned motion synthesis to enable adaptive and semantically consistent robot behavior. A gated mixture-of-experts model predicts Valence/Arousal from input text, visual, and acoustic features, which then mapped to discrete teaching-act categories through an affect-driven this http URL signals condition a diffusion-based motion generator using clip-level intent and frame-level instructional schedules via additive latent restriction with auxiliary action-group supervision. Compared to a baseline diffusion model, our proposed method produces more structured and distinctive motion patterns, as verified by motion statics and pairwise distance analysis. Generated motion sequences remain physically plausible and can be retargeted to a NAO robot for real-time execution. The results reveal that reasoning-guided instructional conditioning improves gesture controllability and pedagogical expressiveness in educational human-robot interaction.
>
---
#### [new 026] ViTac-Tracing: Visual-Tactile Imitation Learning of Deformable Object Tracing
- **分类: cs.RO**

- **简介: 该论文属于变形物体追踪任务，旨在解决现有方法泛化能力差和现实世界可靠性低的问题。提出一种视觉触觉模仿学习方法，实现1D和2D变形物体的可靠追踪。**

- **链接: [https://arxiv.org/pdf/2603.18784](https://arxiv.org/pdf/2603.18784)**

> **作者:** Yongqiang Zhao; Haining Luo; Yupeng Wang; Emmanouil Spyrakos Papastavridis; Yiannis Demiris; Shan Luo
>
> **备注:** The paper has been accepted by ICRA2026
>
> **摘要:** Deformable objects often appear in unstructured configurations. Tracing deformable objects helps bringing them into extended states and facilitating the downstream manipulation tasks. Due to the requirements for object-specific modeling or sim-to-real transfer, existing tracing methods either lack generalizability across different categories of deformable objects or struggle to complete tasks reliably in the real world. To address this, we propose a novel visual-tactile imitation learning method to achieve one-dimensional (1D) and two-dimensional (2D) deformable object tracing with a unified model. Our method is designed from both local and global perspectives based on visual and tactile sensing. Locally, we introduce a weighted loss that emphasizes actions maintaining contact near the center of the tactile image, improving fine-grained adjustment. Globally, we propose a tracing task loss that helps the policy to regulate task progression. On the hardware side, to compensate for the limited features extracted from visual information, we integrate tactile sensing into a low-cost teleoperation system considering both the teleoperator and the robot. Extensive ablation and comparative experiments on diverse 1D and 2D deformable objects demonstrate the effectiveness of our approach, achieving an average success rate of 80% on seen objects and 65% on unseen objects.
>
---
#### [new 027] Contact Status Recognition and Slip Detection with a Bio-inspired Tactile Hand
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在解决脆弱物体抓取中的滑动检测问题。通过生物启发的触觉手，利用多模态触觉反馈进行接触状态识别与滑动检测。**

- **链接: [https://arxiv.org/pdf/2603.18370](https://arxiv.org/pdf/2603.18370)**

> **作者:** Chengxiao He; Wenhui Yang; Hongliang Zhao; Jiacheng Lv; Yuzhe Shao; Longhui Qin
>
> **备注:** 7 pages, 9 figures
>
> **摘要:** Stable and reliable grasp is critical to robotic manipulations especially for fragile and glazed objects, where the grasp force requires precise control as too large force possibly damages the objects while small force leads to slip and fall-off. Although it is assumed the objects to manipulate is grasped firmly in advance, slip detection and timely prevention are necessary for a robot in unstructured and universal environments. In this work, we addressed this issue by utilizing multimodal tactile feedback from a five-fingered bio-inspired hand. Motivated by human hands, the tactile sensing elements were distributed and embedded into the soft skin of robotic hand, forming 24 tactile channels in total. Different from the threshold method that was widely employed in most existing works, we converted the slip detection problem to contact status recognition in combination with binning technique first and then detected the slip onset time according to the recognition results. After the 24-channel tactile signals passed through discrete wavelet transform, 17 features were extracted from different time and frequency bands. With the optimal 120 features employed for status recognition, the test accuracy reached 96.39% across three different sliding speeds and six kinds of materials. When applied to four new unseen materials, a high accuracy of 91.95% was still achieved, which further validated the generalization of our proposed method. Finally, the performance of slip detection is verified based on the trained model of contact status recognition.
>
---
#### [new 028] DriveVLM-RL: Neuroscience-Inspired Reinforcement Learning with Vision-Language Models for Safe and Deployable Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于自动驾驶任务，旨在解决安全决策难题。通过融合视觉语言模型与强化学习，提出DriveVLM-RL框架，提升驾驶安全性与实时性。**

- **链接: [https://arxiv.org/pdf/2603.18315](https://arxiv.org/pdf/2603.18315)**

> **作者:** Zilin Huang; Zihao Sheng; Zhengyang Wan; Yansong Qu; Junwei You; Sicong Jiang; Sikai Chen
>
> **备注:** 32 pages, 15 figures. Code and demo available online
>
> **摘要:** Ensuring safe decision-making in autonomous vehicles remains a fundamental challenge despite rapid advances in end-to-end learning approaches. Traditional reinforcement learning (RL) methods rely on manually engineered rewards or sparse collision signals, which fail to capture the rich contextual understanding required for safe driving and make unsafe exploration unavoidable in real-world settings. Recent vision-language models (VLMs) offer promising semantic understanding capabilities; however, their high inference latency and susceptibility to hallucination hinder direct application to real-time vehicle control. To address these limitations, this paper proposes DriveVLM-RL, a neuroscience-inspired framework that integrates VLMs into RL through a dual-pathway architecture for safe and deployable autonomous driving. The framework decomposes semantic reward learning into a Static Pathway for continuous spatial safety assessment using CLIP-based contrasting language goals, and a Dynamic Pathway for attention-gated multi-frame semantic risk reasoning using a lightweight detector and a large VLM. A hierarchical reward synthesis mechanism fuses semantic signals with vehicle states, while an asynchronous training pipeline decouples expensive VLM inference from environment interaction. All VLM components are used only during offline training and are removed at deployment, ensuring real-time feasibility. Experiments in the CARLA simulator show significant improvements in collision avoidance, task success, and generalization across diverse traffic scenarios, including strong robustness under settings without explicit collision penalties. These results demonstrate that DriveVLM-RL provides a practical paradigm for integrating foundation models into autonomous driving without compromising real-time feasibility. Demo video and code are available at: this https URL
>
---
#### [new 029] GoalVLM: VLM-driven Object Goal Navigation for Multi-Agent System
- **分类: cs.RO**

- **简介: 该论文提出GoalVLM，解决多智能体开放词汇目标导航问题。通过集成视觉语言模型和空间推理，实现零样本目标定位与探索。**

- **链接: [https://arxiv.org/pdf/2603.18210](https://arxiv.org/pdf/2603.18210)**

> **作者:** MoniJesu James; Amir Atef Habel; Aleksey Fedoseev; Dzmitry Tsetserokou
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Object-goal navigation has traditionally been limited to ground robots with closed-set object vocabularies. Existing multi-agent approaches depend on precomputed probabilistic graphs tied to fixed category sets, precluding generalization to novel goals at test time. We present GoalVLM, a cooperative multi-agent framework for zero-shot, open-vocabulary object navigation. GoalVLM integrates a Vision-Language Model (VLM) directly into the decision loop, SAM3 for text-prompted detection and segmentation, and SpaceOM for spatial reasoning, enabling agents to interpret free-form language goals and score frontiers via zero-shot semantic priors without retraining. Each agent builds a BEV semantic map from depth-projected voxel splatting, while a Goal Projector back-projects detections through calibrated depth into the map for reliable goal localization. A constraint-guided reasoning layer evaluates frontiers through a structured prompt chain (scene captioning, room-type classification, perception gating, multi-frontier ranking), injecting commonsense priors into exploration. We evaluate GoalVLM on GOAT-Bench val_unseen (360 multi-subtask episodes, 1032 sequential object-goal subtasks, HM3D scenes), where each episode requires navigating to a chain of 5-7 open-vocabulary targets. GoalVLM with N=2 agents achieves 55.8% subtask SR and 18.3% SPL, competitive with state-of-the-art methods while requiring no task-specific training. Ablation studies confirm the contributions of VLM-guided frontier reasoning and depth-projected goal localization.
>
---
#### [new 030] Offload or Overload: A Platform Measurement Study of Mobile Robotic Manipulation Workloads
- **分类: cs.RO; cs.AI; cs.NI; eess.SY**

- **简介: 研究移动机器人操作任务中的计算负载，分析在不同平台上的运行效果，探讨卸载与过载的挑战与优化方案。**

- **链接: [https://arxiv.org/pdf/2603.18284](https://arxiv.org/pdf/2603.18284)**

> **作者:** Sara Pohland; Xenofon Foukas; Ganesh Ananthanarayanan; Andrey Kolobov; Sanjeev Mehrotra; Bozidar Radunovic; Ankit Verma
>
> **备注:** 15 pages, 17 figures
>
> **摘要:** Mobile robotic manipulation--the ability of robots to navigate spaces and interact with objects--is a core capability of physical AI. Foundation models have led to breakthroughs in their performance, but at a significant computational cost. We present the first measurement study of mobile robotic manipulation workloads across onboard, edge, and cloud GPU platforms. We find that the full workload stack is infeasible to run on smaller onboard GPUs, while larger onboard GPUs drain robot batteries several hours faster. Offloading alleviates these constraints but introduces its own challenges, as additional network latency degrades task accuracy, and the bandwidth requirement makes naive cloud offloading impractical. Finally, we quantify opportunities and pitfalls of sharing compute across robot fleets. We believe our measurement study will be crucial to designing inference systems for mobile robots.
>
---
#### [new 031] ManiDreams: An Open-Source Library for Robust Object Manipulation via Uncertainty-aware Task-specific Intuitive Physics
- **分类: cs.RO**

- **简介: 该论文提出ManiDreams，解决机器人操作中的不确定性问题。属于机器人操纵任务，通过整合不确定性建模与规划，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.18336](https://arxiv.org/pdf/2603.18336)**

> **作者:** Gaotian Wang; Kejia Ren; Andrew S. Morgan; Kaiyu Hang
>
> **备注:** 9 pages, 10 figures. Project page at this https URL
>
> **摘要:** Dynamics models, whether simulators or learned world models, have long been central to robotic manipulation, but most focus on minimizing prediction error rather than confronting a more fundamental challenge: real-world manipulation is inherently uncertain. We argue that robust manipulation under uncertainty is fundamentally an integration problem: uncertainties must be represented, propagated, and constrained within the planning loop, not merely suppressed during training. We present and open-source ManiDreams, a modular framework for uncertainty-aware manipulation planning over intuitive physics models. It realizes this integration through composable abstractions for distributional state representation, backend-agnostic dynamics prediction, and declarative constraint specification for action optimization. The framework explicitly addresses three sources of uncertainty: perceptual, parametric, and structural. It wraps any base policy with a sample-predict-constrain loop that evaluates candidate actions against distributional outcomes, adding robustness without retraining. Experiments on ManiSkill tasks show that ManiDreams maintains robust performance under various perturbations where the RL baseline degrades significantly. Runnable examples on pushing, picking, catching, and real-world deployment demonstrate flexibility across different policies, optimizers, physics backends, and executors. The framework is publicly available at this https URL
>
---
#### [new 032] Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在提升策略学习效率。通过引入基于动力学的先验知识，构建ABD-Net模型，解决动力学信息未充分利用的问题。**

- **链接: [https://arxiv.org/pdf/2603.19078](https://arxiv.org/pdf/2603.19078)**

> **作者:** Sangwoo Shin; Kunzhao Ren; Xiaobin Xiong; Josiah Hanna
>
> **备注:** Arxiv_r1
>
> **摘要:** Recent work in reinforcement learning has shown that incorporating structural priors for articulated robots, such as link connectivity, into policy networks improves learning efficiency. However, dynamics properties, despite their fundamental role in determining how forces and motion propagate through the body, remain largely underexplored as an inductive bias for policy learning. To address this gap, we present the Articulated-Body Dynamics Network (ABD-Net), a novel graph neural network architecture grounded in the computational structure of forward dynamics. Specifically, we adapt the inertia propagation mechanism from the Articulated Body Algorithm, systematically aggregating inertial quantities from child to parent links in a tree-structured manner, while replacing physical quantities with learnable parameters. Embedding ABD-NET into the policy actor enables dynamics-informed representations that capture how actions propagate through the body, leading to efficient and robust policy learning. Through experiments with simulated humanoid, quadruped, and hopper robots, our approach demonstrates increased sample efficiency and generalization to dynamics shifts compared to transformer-based and GNN baselines. We further validate the learned policy on real Unitree G1 and Go2 robots, state-of-the-art humanoid and quadruped platforms, generating dynamic, versatile and robust locomotion behaviors through sim-to-real transfer with real-time inference.
>
---
#### [new 033] MERGE: Guided Vision-Language Models for Multi-Actor Event Reasoning and Grounding in Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文提出MERGE系统，解决人机协作中的多主体事件推理与定位问题。通过整合视觉语言模型和感知流水线，提升情境感知效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.18988](https://arxiv.org/pdf/2603.18988)**

> **作者:** Joerg Deigmoeller; Nakul Agarwal; Stephan Hasler; Daniel Tanneberg; Anna Belardinelli; Reza Ghoddoosian; Chao Wang; Felix Ocker; Fan Zhang; Behzad Dariush; Michael Gienger
>
> **摘要:** We introduce MERGE, a system for situational grounding of actors, objects, and events in dynamic human-robot group interactions. Effective collaboration in such settings requires consistent situational awareness, built on persistent representations of people and objects and an episodic abstraction of events. MERGE achieves this by uniquely identifying physical instances of actors (humans or robots) and objects and structuring them into actor-action-object relations, ensuring temporal consistency across interactions. Central to MERGE is the integration of Vision-Language Models (VLMs) guided with a perception pipeline: a lightweight streaming module continuously processes visual input to detect changes and selectively invokes the VLM only when necessary. This decoupled design preserves the reasoning power and zero-shot generalization of VLMs while improving efficiency, avoiding both the high monetary cost and the latency of frame-by-frame captioning that leads to fragmented and delayed outputs. To address the absence of suitable benchmarks for multi-actor collaboration, we introduce the GROUND dataset, which offers fine-grained situational annotations of multi-person and human-robot interactions. On this dataset, our approach improves the average grounding score by a factor of 2 compared to the performance of VLM-only baselines - including GPT-4o, GPT-5 and Gemini 2.5 Flash - while also reducing run-time by a factor of 4. The code and data are available at this http URL.
>
---
#### [new 034] Final Report for the Workshop on Robotics & AI in Medicine
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于科技报告任务，旨在总结机器人与AI在医学领域的应用现状及挑战，提出建立国家研究中心以推动相关技术发展。**

- **链接: [https://arxiv.org/pdf/2603.18130](https://arxiv.org/pdf/2603.18130)**

> **作者:** Juan P Wachs
>
> **备注:** 51 pages, 5 figures
>
> **摘要:** The CARE Workshop on Robotics and AI in Medicine, held on December 1, 2025 in Indianapolis, convened leading researchers, clinicians, industry innovators, and federal stakeholders to shape a national vision for advancing robotics and artificial intelligence in healthcare. The event highlighted the accelerating need for coordinated research efforts that bridge engineering innovation with real clinical priorities, emphasizing safety, reliability, and translational readiness with an emphasis on the use of robotics and AI to achieve this readiness goal. Across keynotes, panels, and breakout sessions, participants underscored critical gaps in data availability, standardized evaluation methods, regulatory pathways, and workforce training that hinder the deployment of intelligent robotic systems in surgical, diagnostic, rehabilitative, and assistive contexts. Discussions emphasized the transformative potential of AI enabled robotics to improve precision, reduce provider burden, expand access to specialized care, and enhance patient outcomes particularly in undeserved regions and high risk procedural domains. Special attention was given to austere settings, disaster and relief and military settings. The workshop demonstrated broad consensus on the urgency of establishing a national Center for AI and Robotic Excellence in medicine (CARE). Stakeholders identified priority research thrusts including human robot collaboration, trustworthy autonomy, simulation and digital twins, multi modal sensing, and ethical integration of generative AI into clinical workflows. Participants also articulated the need for high quality datasets, shared test beds, autonomous surgical systems, clinically grounded benchmarks, and sustained interdisciplinary training mechanisms.
>
---
#### [new 035] Inductance-Based Force Self-Sensing in Fiber-Reinforced Pneumatic Twisted-and-Coiled Actuators
- **分类: cs.RO**

- **简介: 该论文属于机器人执行器控制任务，旨在解决FR-PTCA缺乏本征力感知的问题。通过集成镍丝实现电感反馈，建立模型与观测器进行力和位移估计。**

- **链接: [https://arxiv.org/pdf/2603.18555](https://arxiv.org/pdf/2603.18555)**

> **作者:** Yunsong Zhang; Tianlin Li; Mingyang Yang; Feitian Zhang
>
> **摘要:** Fiber-reinforced pneumatic twisted-and-coiled actuators (FR-PTCAs) offer high power density and compliance but their strong hysteresis and lack of intrinsic proprioception limit effective closed-loop control. This paper presents a self-sensing FR-PTCA integrated with a conductive nickel wire that enables intrinsic force estimation and indirect displacement inference via inductance feedback. Experimental characterization reveals that the inductance of the actuator exhibits a deterministic, low-hysteresis inductance-force relationship at constant pressures, in contrast to the strongly hysteretic inductance-length behavior. Leveraging this property, this paper develops a parametric self-sensing model and a nonlinear hybrid observer that integrates an Extended Kalman Filter (EKF) with constrained optimization to resolve the ambiguity in the inductance-force mapping and estimate actuator states. Experimental results demonstrate that the proposed approach achieves force estimation accuracy comparable to that of external load cells and maintains robust performance under varying load conditions.
>
---
#### [new 036] Lightweight Model Predictive Control for Spacecraft Rendezvous Attitude Synchronization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于航天器姿态控制任务，解决 rendezvous 过程中的姿态同步问题，提出两种轻量级MPC方法，提升跟踪精度并降低计算资源需求。**

- **链接: [https://arxiv.org/pdf/2603.18921](https://arxiv.org/pdf/2603.18921)**

> **作者:** Peter Stadler; Alexander Meinert; Niklas Baldauf; Alen Turnwald
>
> **备注:** Accepted at European Control Conference (ECC 2026)
>
> **摘要:** This work introduces two lightweight model predictive control (MPC) approaches for attitude tracking with reaction wheels during spacecraft rendezvous synchronization. Both approaches are based on a novel attitude deviation formulation, which enables the use of inherently linear constraints on angular velocity. We develop a single-loop and a dual-loop MPC; the latter embeds a stabilizing feedback controller within the inner loop, yielding a linear time-invariant system. Both controllers are implemented with CasADi - including automatic code generation - evaluated across various solvers, and validated within the Basilisk astrodynamics simulation framework. The experimental results demonstrate improved tracking accuracy alongside reductions in computational effort and memory consumption. Finally, embedded delivery to an ARM Cortex-M7 - representative of commercial off-the-shelf devices used in New Space platforms - confirms the real-time feasibility of these approaches and highlights their suitability for onboard attitude control in resource-constrained spacecraft rendezvous missions.
>
---
#### [new 037] CAMO: A Conditional Neural Solver for the Multi-objective Multiple Traveling Salesman Problem
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出CAMO，解决多目标多旅行商问题（MOMTSP），通过神经网络实现多智能体协同与多目标优化。**

- **链接: [https://arxiv.org/pdf/2603.19074](https://arxiv.org/pdf/2603.19074)**

> **作者:** Fengxiaoxiao Li; Xiao Mao; Mingfeng Fan; Yifeng Zhang; Yi Li; Tanishq Duhan; Guillaume Sartoretti
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** Robotic systems often require a team of robots to collectively visit multiple targets while optimizing competing objectives, such as total travel cost and makespan. This setting can be formulated as the Multi-Objective Multiple Traveling Salesman Problem (MOMTSP). Although learning-based methods have shown strong performance on the single-agent TSP and multi-objective TSP variants, they rarely address the combined challenges of multi-agent coordination and multi-objective trade-offs, which introduce dual sources of complexity. To bridge this gap, we propose CAMO, a conditional neural solver for MOMTSP that generalizes across varying numbers of targets, agents, and preference vectors, and yields high-quality approximations to the Pareto front (PF). Specifically, CAMO consists of a conditional encoder to fuse preferences into instance representations, enabling explicit control over multi-objective trade-offs, and a collaborative decoder that coordinates all agents by alternating agent selection and node selection to construct multi-agent tours autoregressively. To further improve generalization, we train CAMO with a REINFORCE-based objective over a mixed distribution of problem sizes. Extensive experiments show that CAMO outperforms both neural and conventional heuristics, achieving a closer approximation of PFs. In addition, ablation results validate the contributions of CAMO's key components, and real-world tests on a mobile robot platform demonstrate its practical applicability.
>
---
#### [new 038] REST: Receding Horizon Explorative Steiner Tree for Zero-Shot Object-Goal Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出REST框架，解决零样本物体目标导航问题。通过构建路径树优化选项设计，提升导航效率与成功率。**

- **链接: [https://arxiv.org/pdf/2603.18624](https://arxiv.org/pdf/2603.18624)**

> **作者:** Shuqi Xiao; Maani Ghaffari; Chengzhong Xu; Hui Kong
>
> **摘要:** Zero-shot object-goal navigation (ZSON) requires navigating unknown environments to find a target object without task-specific training. Prior hierarchical training-free solutions invest in scene understanding (\textit{belief}) and high-level decision-making (\textit{policy}), yet overlook the design of \textit{option}, i.e., a subgoal candidate proposed from evolving belief and presented to policy for selection. In practice, options are reduced to isolated waypoints scored independently: single destinations hide the value gathered along the journey; an unstructured collection obscures the relationships among candidates. Our insight is that the option space should be a \textit{tree of paths}. Full paths expose en-route information gain that destination-only scoring systematically neglects; a tree of shared segments enables coarse-to-fine LLM reasoning that dismisses or pursues entire branches before examining individual leaves, compressing the combinatorial path space into an efficient hierarchy. We instantiate this insight in \textbf{REST} (Receding Horizon Explorative Steiner Tree), a training-free framework that (1) builds an explicit open-vocabulary 3D map from online RGB-D streams; (2) grows an agent-centric tree of safe and informative paths as the option space via sampling-based planning; and (3) textualizes each branch into a spatial narrative and selects the next-best path through chain-of-thought LLM reasoning. Across the Gibson, HM3D, and HSSD benchmarks, REST consistently ranks among the top methods in success rate while achieving the best or second-best path efficiency, demonstrating a favorable efficiency-success balance.
>
---
#### [new 039] ReDAG-RT: Global Rate-Priority Scheduling for Real-Time Multi-DAG Execution in ROS 2
- **分类: cs.RO**

- **简介: 该论文提出ReDAG-RT框架，解决ROS 2中多DAG任务的实时调度问题，通过全局速率优先调度提升系统确定性和可靠性。**

- **链接: [https://arxiv.org/pdf/2603.18238](https://arxiv.org/pdf/2603.18238)**

> **作者:** Md. Mehedi Hasan; Rafid Mostafiz; Bikash Kumar Paul; Md. Abir Hossain; Ziaur Rahman
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** ROS 2 has become a dominant middleware for robotic systems, where perception, estimation, planning, and control pipelines are structured as directed acyclic graphs of callbacks executed under a shared executor. However, default ROS 2 executors use best-effort dispatch without cross-DAG priority enforcement, leading to callback contention, structural priority inversion, and deadline instability under concurrent workloads. These limitations restrict deployment in time-critical and safety-sensitive cyber-physical systems. This paper presents ReDAGRT, a user-space global scheduling framework for deterministic multi-DAG execution in unmodified ROS 2. The framework introduces a Rate-Priority driven global ready queue that orders callbacks by activation rate, enforces per-DAG concurrency bounds, and mitigates cross-graph priority inversion without modifying the ROS 2 API, executor interface, or underlying operating system scheduler. We formalize a multi-DAG task model for ROS 2 callback pipelines and analyze cross-DAG interference under Rate-Priority scheduling. Response-time recurrences and schedulability conditions are derived within classical Rate-Monotonic theory. Experiments in a ROS 2 Humble environment compare ReDAGRT against SingleThreadedExecutor and MultiThreadedExecutor using synthetic multi-DAG workloads. Results show up to 29.7 percent reduction in deadline miss rate, 42.9 percent reduction in 99th percentile response time, and 13.7 percent improvement over MultiThreadedExecutor under comparable utilization. Asymmetric per-DAG concurrency bounds further reduce interference by 40.8 percent. These results demonstrate that deterministic and analyzable multi-DAG scheduling can be achieved entirely in the ROS 2 user-space execution layer, providing a practical foundation for real-time robotic middleware in safety-critical systems.
>
---
#### [new 040] PRIOR: Perceptive Learning for Humanoid Locomotion with Reference Gait Priors
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人行走任务，旨在解决复杂地形中自然步态行走的问题。提出PRIOR框架，结合参数化步态生成、状态估计和地形适应奖励，实现高效稳定行走。**

- **链接: [https://arxiv.org/pdf/2603.18979](https://arxiv.org/pdf/2603.18979)**

> **作者:** Chenxi Han; Shilu He; Yi Cheng; Linqi Ye; Houde Liu
>
> **备注:** this https URL
>
> **摘要:** Training perceptive humanoid locomotion policies that traverse complex terrains with natural gaits remains an open challenge, typically demanding multi-stage training pipelines, adversarial objectives, or extensive real-world calibration. We present PRIOR, an efficient and reproducible framework built on Isaac Lab that achieves robust terrain traversal with human-like gaits through a simple yet effective design: (i) a parametric gait generator that supplies stable reference trajectories derived from motion capture without adversarial training, (ii) a GRU-based state estimator that infers terrain geometry directly from egocentric depth images via self-supervised heightmap reconstruction, and (iii) terrain-adaptive footstep rewards that guide foot placement toward traversable regions. Through systematic analysis of depth image resolution trade-offs, we identify configurations that maximize terrain fidelity under real-time constraints, substantially reducing perceptual overhead without degrading traversal performance. Comprehensive experiments across terrains of varying difficulty-including stairs, boxes, and gaps-demonstrate that each component yields complementary and essential performance gains, with the full framework achieving a 100% traversal success rate. We will open-source the complete PRIOR framework, including the training pipeline, parametric gait generator, and evaluation benchmarks, to serve as a reproducible foundation for humanoid locomotion research on Isaac Lab.
>
---
#### [new 041] MemoAct: Atkinson-Shiffrin-Inspired Memory-Augmented Visuomotor Policy for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决记忆依赖任务中的状态跟踪与长时保留问题。提出MemoAct，通过分层记忆结构提升性能。**

- **链接: [https://arxiv.org/pdf/2603.18494](https://arxiv.org/pdf/2603.18494)**

> **作者:** Liufan Tan; Jiale Li; Gangshan Jing
>
> **摘要:** Memory-augmented robotic policies are essential in handling memory-dependent tasks. However, existing approaches typically rely on simple observation window extensions, struggling to simultaneously achieve precise task state tracking and robust long-horizon retention. To overcome these challenges, inspired by the Atkinson-Shiffrin memory model, we propose MemoAct, a hierarchical memory-based policy that leverages distinct memory tiers to tackle specific bottlenecks. Specifically, lossless short-term memory ensures precise task state tracking, while compressed long-term memory enables robust long-horizon retention. To enrich the evaluation landscape, we construct MemoryRTBench based on RoboTwin 2.0, specifically tailored to assess policy capabilities in task state tracking and long-horizon retention. Extensive experiments across simulated and real-world scenarios demonstrate that MemoAct achieves superior performance compared to both existing Markovian baselines and history-aware policies. The project page is \href{this https URL}{available}.
>
---
#### [new 042] Safety-Guaranteed Imitation Learning from Nonlinear Model Predictive Control for Spacecraft Close Proximity Operations
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对航天器近距离操作任务，提出一种安全保障的模仿学习框架，解决实时控制与安全性问题，通过CBF和CLF结合提升控制稳定性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.18910](https://arxiv.org/pdf/2603.18910)**

> **作者:** Alexander Meinert; Niklas Baldauf; Peter Stadler; Alen Turnwald
>
> **备注:** Accepted at European Control Conference (ECC 2026)
>
> **摘要:** This paper presents a safety-guaranteed, runtime-efficient imitation learning framework for spacecraft close proximity control. We leverage Control Barrier Functions (CBFs) for safety certificates and Control Lyapunov Functions (CLFs) for stability as unified design principles across data generation, training, and deployment. First, a nonlinear Model Predictive Control (NMPC) expert enforces CBF constraints to provide safe reference trajectories. Second, we train a neural policy with a novel CBF-CLF-informed loss and DAgger-like rollouts with curriculum weighting, promoting data-efficiency and reducing future safety filter interventions. Third, at deployment a lightweight one-step CBF-CLF quadratic program minimally adjusts the learned control input to satisfy hard safety constraints while encouraging stability. We validate the approach for ESA-compliant close proximity operations, including fly-around with a spherical keep-out zone and final approach inside a conical approach corridor, using the Basilisk high-fidelity simulator with nonlinear dynamics and perturbations. Numerical experiments indicate stable convergence to decision points and strict adherence to safety under the filter, with task performance comparable to the NMPC expert while significantly reducing online computation. A runtime analysis demonstrates real-time feasibility on a commercial off-the-shelf processor, supporting onboard deployment for safety-critical on-orbit servicing.
>
---
#### [new 043] Tendon-Actuated Robots with a Tapered, Flexible Polymer Backbone: Design, Fabrication, and Modeling
- **分类: cs.RO**

- **简介: 该论文属于柔性机器人设计任务，解决传统连续机器人成本高、定制难的问题，通过3D打印和几何锥形设计实现低成本、高灵活性的腱驱动机器人。**

- **链接: [https://arxiv.org/pdf/2603.19124](https://arxiv.org/pdf/2603.19124)**

> **作者:** Harald Minde Hansen; Nandita Gallacher; Nicholas B. Andrews; Kristin Y. Pettersen; Jan Tommy Gravdahl; Mario di Castro
>
> **摘要:** This paper presents the design, modeling, and fabrication of 3D-printed, tendon-actuated continuum robots featuring a flexible, tapered backbone constructed from thermoplastic polyurethane (TPU). Our scalable design incorporates an integrated electronics base housing that enables direct tendon tension control and sensing via actuators and compression load cells. Unlike many continuum robots that are single-purpose and costly, the proposed design prioritizes customizability, rapid assembly, and low cost while enabling high curvature and enhanced distal compliance through geometric tapering, thereby supporting a broad range of compliant robotic inspection and manipulation tasks. We develop a generalized forward kinetostatic model of the tapered backbone based on Cosserat rod theory using a Newtonian approach, extending existing tendon-actuated Cosserat rod formulations to explicitly account for spatially varying backbone cross-sectional geometry. The model captures the graded stiffness profile induced by the tapering and enables systematic exploration of the configuration space as a function of the geometric design parameters. Specifically, we analyze how the backbone taper angle influences the robot's configuration space and manipulability. The model is validated against motion capture data, achieving centimeter-level shape prediction accuracy after calibrating Young's modulus via a line search that minimizes modeling error. We further demonstrate teleoperated grasping using an endoscopic gripper routed along the continuum robot, mounted on a 6-DoF robotic arm. Parameterized iLogic/CAD scripts are provided for rapid geometry generation and scaling. The presented framework establishes a simple, rapid, and reproducible pathway from parametric design to controlled tendon actuation for tapered, tendon-driven continuum robots manufactured using fused deposition modeling 3D printers.
>
---
#### [new 044] SG-CoT: An Ambiguity-Aware Robotic Planning Framework using Scene Graph Representations
- **分类: cs.RO**

- **简介: 该论文提出SG-CoT框架，解决机器人规划中的模糊性问题。通过场景图与大语言模型结合，提升规划可靠性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.18271](https://arxiv.org/pdf/2603.18271)**

> **作者:** Akshat Rana; Peeyush Agarwal; K.P.S. Rana; Amarjit Malhotra
>
> **备注:** This work has been submitted to the IEEE Robotics and Automation Letters for possible publication
>
> **摘要:** Ambiguity poses a major challenge to large language models (LLMs) used as robotic planners. In this letter, we present Scene Graph-Chain-of-Thought (SG-CoT), a two-stage framework where LLMs iteratively query a scene graph representation of the environment to detect and clarify ambiguities. First, a structured scene graph representation of the environment is constructed from input observations, capturing objects, their attributes, and relationships with other objects. Second, the LLM is equipped with retrieval functions to query portions of the scene graph that are relevant to the provided instruction. This grounds the reasoning process of the LLM in the observation, increasing the reliability of robotic planners under ambiguous situations. SG-CoT also allows the LLM to identify the source of ambiguity and pose a relevant disambiguation question to the user or another robot. Extensive experimentation demonstrates that SG-CoT consistently outperforms prior methods, with a minimum of 10% improvement in question accuracy and a minimum success rate increase of 4% in single-agent and 15% in multi-agent environments, validating its effectiveness for more generalizable robot planning.
>
---
#### [new 045] NavTrust: Benchmarking Trustworthiness for Embodied Navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; eess.SY**

- **简介: 该论文属于 embodied navigation 任务，旨在解决真实场景下导航模型鲁棒性不足的问题。通过构建 NavTrust 基准，评估不同输入噪声对导航性能的影响，并提出增强鲁棒性的策略。**

- **链接: [https://arxiv.org/pdf/2603.19229](https://arxiv.org/pdf/2603.19229)**

> **作者:** Huaide Jiang; Yash Chaudhary; Yuping Wang; Zehao Wang; Raghav Sharma; Manan Mehta; Yang Zhou; Lichao Sun; Zhiwen Fan; Zhengzhong Tu; Jiachen Li
>
> **备注:** Project Website: this https URL
>
> **摘要:** There are two major categories of embodied navigation: Vision-Language Navigation (VLN), where agents navigate by following natural language instructions; and Object-Goal Navigation (OGN), where agents navigate to a specified target object. However, existing work primarily evaluates model performance under nominal conditions, overlooking the potential corruptions that arise in real-world settings. To address this gap, we present NavTrust, a unified benchmark that systematically corrupts input modalities, including RGB, depth, and instructions, in realistic scenarios and evaluates their impact on navigation performance. To our best knowledge, NavTrust is the first benchmark that exposes embodied navigation agents to diverse RGB-Depth corruptions and instruction variations in a unified framework. Our extensive evaluation of seven state-of-the-art approaches reveals substantial performance degradation under realistic corruptions, which highlights critical robustness gaps and provides a roadmap toward more trustworthy embodied navigation systems. Furthermore, we systematically evaluate four distinct mitigation strategies to enhance robustness against RGB-Depth and instructions corruptions. Our base models include Uni-NaVid and ETPNav. We deployed them on a real mobile robot and observed improved robustness to corruptions. The project website is: this https URL.
>
---
#### [new 046] Sparse3DTrack: Monocular 3D Object Tracking Using Sparse Supervision
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于单目3D目标跟踪任务，解决标注数据稀缺问题。通过稀疏监督框架，分解任务为2D和3D子问题，提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2603.18298](https://arxiv.org/pdf/2603.18298)**

> **作者:** Nikhil Gosala; B. Ravi Kiran; Senthil Yogamani; Abhinav Valada
>
> **备注:** 22 pages, 8 figures
>
> **摘要:** Monocular 3D object tracking aims to estimate temporally consistent 3D object poses across video frames, enabling autonomous agents to reason about scene dynamics. However, existing state-of-the-art approaches are fully supervised and rely on dense 3D annotations over long video sequences, which are expensive to obtain and difficult to scale. In this work, we address this fundamental limitation by proposing the first sparsely supervised framework for monocular 3D object tracking. Our approach decomposes the task into two sequential sub-problems: 2D query matching and 3D geometry estimation. Both components leverage the spatio-temporal consistency of image sequences to augment a sparse set of labeled samples and learn rich 2D and 3D representations of the scene. Leveraging these learned cues, our model automatically generates high-quality 3D pseudolabels across entire videos, effectively transforming sparse supervision into dense 3D track annotations. This enables existing fully-supervised trackers to effectively operate under extreme label sparsity. Extensive experiments on the KITTI and nuScenes datasets demonstrate that our method significantly improves tracking performance, achieving an improvement of up to 15.50 p.p. while using at most four ground truth annotations per track.
>
---
#### [new 047] HRI-SA: A Multimodal Dataset for Online Assessment of Human Situational Awareness during Remote Human-Robot Teaming
- **分类: cs.RO; cs.HC; cs.LG; cs.MA**

- **简介: 该论文属于人机协作任务，旨在解决远程人机团队中实时评估操作员情境意识的问题。研究构建了HRI-SA数据集，并验证了眼动特征在检测情境意识延迟中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.18344](https://arxiv.org/pdf/2603.18344)**

> **作者:** Hashini Senaratne; Richard Attfield; Samith Widhanapathirana; David Howard; Cecile Paris; Dana Kulic; Leimin Tian
>
> **备注:** This work is currently under peer review
>
> **摘要:** Maintaining situational awareness (SA) is critical in human-robot teams. Yet, under high workload and dynamic conditions, operators often experience SA gaps. Automated detection of SA gaps could provide timely assistance for operators. However, conventional SA measures either disrupt task flow or cannot capture real-time fluctuations, limiting their operational utility. To the best of our knowledge, no publicly available dataset currently supports the systematic evaluation of online human SA assessment in human-robot teaming. To advance the development of online SA assessment tools, we introduce HRI-SA, a multimodal dataset from 30 participants in a realistic search-and-rescue human-robot teaming context, incorporating eye movements, pupil diameter, biosignals, user interactions, and robot data. The experimental protocol included predefined events requiring timely operator assistance, with ground truth SA latency of two types (perceptual and comprehension) systematically obtained by measuring the time between assistance need onset and resolution. We illustrate the utility of this dataset by evaluating standard machine learning models for detecting perceptual SA latencies using generic eye-tracking features and contextual features. Results show that eye-tracking features alone effectively classified perceptual SA latency (recall=88.91%, F1=67.63%) using leave-one-group-out cross-validation, with performance improved through contextual data fusion (recall=91.51%, F1=80.38%). This paper contributes the first public dataset supporting the systematic evaluation of SA throughout a human-robot teaming mission, while also demonstrating the potential of generic eye-tracking features for continuous perceptual SA latency detection in remote human-robot teaming.
>
---
#### [new 048] Rapid Adaptation of Particle Dynamics for Generalized Deformable Object Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人变形物体操作任务，解决未知动态下变形物体操控问题。通过RMA扩展方法RAPiD，结合仿真与真实视觉信息，实现高效变形物体操作。**

- **链接: [https://arxiv.org/pdf/2603.18246](https://arxiv.org/pdf/2603.18246)**

> **作者:** Bohan Wu; Roberto Martín-Martín; Li Fei-Fei
>
> **备注:** 8 pages, ICRA 2026
>
> **摘要:** We address the challenge of learning to manipulate deformable objects with unknown dynamics. In non-rigid objects, the dynamics parameters define how they react to interactions -- how they stretch, bend, compress, and move -- and they are critical to determining the optimal actions to perform a manipulation task successfully. In other robotic domains, such as legged locomotion and in-hand rigid object manipulation, state-of-the-art approaches can handle unknown dynamics using Rapid Motor Adaptation (RMA). Through a supervised procedure in simulation that encodes each rigid object's dynamics, such as mass and position, these approaches learn a policy that conditions actions on a vector of latent dynamic parameters inferred from sequences of state-actions. However, in deformable object manipulation, the object's dynamics not only includes its mass and position, but also how the shape of the object changes. Our key insight is that the recent ground-truth particle positions of a deformable object in simulation capture changes in the object's shape, making it possible to extend RMA to deformable object manipulation. This key insight allows us to develop RAPiD, a two-phase method that learns to perform real-robot deformable object mobile manipulation by: 1) learning a visuomotor policy conditioned on the object's dynamics embedding, which is encoded from the object's privileged information in simulation, such as its mass and ground-truth particle positions, and 2) learning to infer this embedding using non-privileged information instead, such as robot visual observations and actions, so that the learned policy can transfer to the real world. On a mobile manipulator with 22 degrees of freedom, RAPiD enables over 80%+ success rates across two vision-based deformable object mobile manipulation tasks in the real world, under various object dynamics, categories, and instances.
>
---
#### [new 049] Efficient and Versatile Quadrupedal Skating: Optimal Co-design via Reinforcement Learning and Bayesian Optimization
- **分类: cs.RO**

- **简介: 该论文研究 quadrupedal 机器人高效滑行问题，通过硬件与控制协同设计，结合强化学习和贝叶斯优化，提升运动效率与多样性。**

- **链接: [https://arxiv.org/pdf/2603.18408](https://arxiv.org/pdf/2603.18408)**

> **作者:** Hanwen Wang; Zhenlong Fang; Josiah Hanna; Xiaobin Xiong
>
> **摘要:** In this paper, we present a hardware-control co-design approach that enables efficient and versatile roller skating on quadrupedal robots equipped with passive wheels. Passive-wheel skating reduces leg inertia and improves energy efficiency, particularly at high speeds. However, the absence of direct wheel actuation tightly couples mechanical design and control. To unlock the full potential of this modality, we formulate a bilevel optimization framework: an upper-level Bayesian Optimization searches the mechanical design space, while a lower-level Reinforcement Learning trains a motor control policy for each candidate design. The resulting design-policy pairs not only outperform human-engineered baselines, but also exhibit versatile behaviors such as hockey stop (rapid braking by turning sideways to maximize friction) and self-aligning motion (automatic reorientation to improve energy efficiency in the direction of travel), offering the first system-level study of dynamic skating motion on quadrupedal robots.
>
---
#### [new 050] From Inference Efficiency to Embodied Efficiency: Revisiting Efficiency Metrics for Vision-Language-Action Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型研究，旨在解决传统效率指标无法反映实际机器人性能的问题。通过分析系统级 embodied 效率，揭示了模型压缩等方法对执行效果的影响。**

- **链接: [https://arxiv.org/pdf/2603.19131](https://arxiv.org/pdf/2603.19131)**

> **作者:** Zhuofan Li; Hongkun Yang; Zhenyang Chen; Yangxuan Chen; Yingyan; Chaojian Li
>
> **摘要:** Vision-Language-Action (VLA) models have recently enabled embodied agents to perform increasingly complex tasks by jointly reasoning over visual, linguistic, and motor modalities. However, we find that the prevailing notion of ``efficiency'' in current VLA research, characterized by parameters, FLOPs, or token decoding throughput, does not reflect actual performance on robotic platforms. In real-world execution, efficiency is determined by system-level embodied behaviors such as task completion time, trajectory smoothness, cumulative joint rotation, and motion energy. Through controlled studies across model compression, token sparsification, and action sequence compression, we make several observations that challenge common assumptions. (1) Methods that reduce computation under conventional metrics often increase end-to-end execution cost or degrade motion quality, despite maintaining task success rates. (2) System-level embodied efficiency metrics reveal performance differences in the learned action policies that remain hidden under conventional evaluations. (3) Common adaptation methods such as in-context prompting or supervised fine-tuning show only mild and metric-specific improvements in embodied efficiency. While these methods can reduce targeted embodied-efficiency metrics such as jerk or action rate, the resulting gains may come with trade-offs in other metrics, such as longer completion time. Taken together, our results suggest that conventional inference efficiency metrics can overlook important aspects of embodied execution. Incorporating embodied efficiency provides a more complete view of policy behavior and practical performance, enabling fairer and more comprehensive comparisons of VLA models.
>
---
#### [new 051] R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于模型基础强化学习任务，旨在解决图像表示中冗余信息的问题。提出R2-Dreamer框架，通过自监督目标减少冗余，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2603.18202](https://arxiv.org/pdf/2603.18202)**

> **作者:** Naoki Morihira; Amal Nahar; Kartik Bharadwaj; Yasuhiro Kato; Akinobu Hayashi; Tatsuya Harada
>
> **备注:** 20 pages, 12 figures, 2 tables. Published as a conference paper at ICLR 2026. Code available at this https URL
>
> **摘要:** A central challenge in image-based Model-Based Reinforcement Learning (MBRL) is to learn representations that distill essential information from irrelevant visual details. While promising, reconstruction-based methods often waste capacity on large task-irrelevant regions. Decoder-free methods instead learn robust representations by leveraging Data Augmentation (DA), but reliance on such external regularizers limits versatility. We propose R2-Dreamer, a decoder-free MBRL framework with a self-supervised objective that serves as an internal regularizer, preventing representation collapse without resorting to DA. The core of our method is a redundancy-reduction objective inspired by Barlow Twins, which can be easily integrated into existing frameworks. On DeepMind Control Suite and Meta-World, R2-Dreamer is competitive with strong baselines such as DreamerV3 and TD-MPC2 while training 1.59x faster than DreamerV3, and yields substantial gains on DMC-Subtle with tiny task-relevant objects. These results suggest that an effective internal regularizer can enable versatile, high-performance decoder-free MBRL. Code is available at this https URL.
>
---
#### [new 052] RE-SAC: Disentangling aleatoric and epistemic risks in bus fleet control: A stable and robust ensemble DRL approach
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于交通控制任务，解决公交调度中因不确定性导致的策略不稳定问题。通过分离aleatoric和epistemic风险，提出RE-SAC框架提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.18396](https://arxiv.org/pdf/2603.18396)**

> **作者:** Yifan Zhang; Liang Zheng
>
> **摘要:** Bus holding control is challenging due to stochastic traffic and passenger demand. While deep reinforcement learning (DRL) shows promise, standard actor-critic algorithms suffer from Q-value instability in volatile environments. A key source of this instability is the conflation of two distinct uncertainties: aleatoric uncertainty (irreducible noise) and epistemic uncertainty (data insufficiency). Treating these as a single risk leads to value underestimation in noisy states, causing catastrophic policy collapse. We propose a robust ensemble soft actor-critic (RE-SAC) framework to explicitly disentangle these uncertainties. RE-SAC applies Integral Probability Metric (IPM)-based weight regularization to the critic network to hedge against aleatoric risk, providing a smooth analytical lower bound for the robust Bellman operator without expensive inner-loop perturbations. To address epistemic risk, a diversified Q-ensemble penalizes overconfident value estimates in sparsely covered regions. This dual mechanism prevents the ensemble variance from misidentifying noise as a data gap, a failure mode identified in our ablation study. Experiments in a realistic bidirectional bus corridor simulation demonstrate that RE-SAC achieves the highest cumulative reward (approx. -0.4e6) compared to vanilla SAC (-0.55e6). Mahalanobis rareness analysis confirms that RE-SAC reduces Oracle Q-value estimation error by up to 62% in rare out-of-distribution states (MAE of 1647 vs. 4343), demonstrating superior robustness under high traffic variability.
>
---
#### [new 053] Fundamental Limits for Sensor-Based Control via the Gibbs Variational Principle
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文研究反馈控制器性能的理论极限，解决传感器信息与控制效果间的关系问题。通过吉布斯变分原理建立下界，提升控制算法评估与设计的准确性。**

- **链接: [https://arxiv.org/pdf/2603.18454](https://arxiv.org/pdf/2603.18454)**

> **作者:** Vincent Pacelli; Evangelos A. Theodorou
>
> **备注:** 6 pages, 1 figure
>
> **摘要:** Fundamental limits on the performance of feedback controllers are essential for benchmarking algorithms, guiding sensor selection, and certifying task feasibility -- yet few general-purpose tools exist for computing them. Existing information-theoretic approaches overestimate the information a sensor must provide by evaluating it against the uncontrolled system, producing bounds that degrade precisely when feedback is most valuable. We derive a lower bound on the minimum expected cost of any causal feedback controller under partial observations by applying the Gibbs variational principle to the joint path measure over states and observations. The bound applies to nonlinear, nonholonomic, and hybrid dynamics with unbounded costs and admits a self-consistent refinement: any good controller concentrates the state, which limits the information the sensor can extract, which tightens the bound. The resulting fixed-point equation has a unique solution computable by bisection, and we provide conditions under which the free energy minimization is provably convex, yielding a certifiably correct numerical bound. On a nonlinear Dubins car tracking problem, the self-consistent bound captures most of the optimal cost across sensor noise levels, while the open-loop variant is vacuous at low noise.
>
---
#### [new 054] GSMem: 3D Gaussian Splatting as Persistent Spatial Memory for Zero-Shot Embodied Exploration and Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出GSMem框架，解决零样本具身探索与推理中的空间记忆问题。通过3D高斯点云实现持续空间记忆，支持视角重构建和语义定位，提升导航与问答性能。**

- **链接: [https://arxiv.org/pdf/2603.19137](https://arxiv.org/pdf/2603.19137)**

> **作者:** Yiren Lu; Yi Du; Disheng Liu; Yunlai Zhou; Chen Wang; Yu Yin
>
> **备注:** Project page at this https URL
>
> **摘要:** Effective embodied exploration requires agents to accumulate and retain spatial knowledge over time. However, existing scene representations, such as discrete scene graphs or static view-based snapshots, lack \textit{post-hoc re-observability}. If an initial observation misses a target, the resulting memory omission is often irrecoverable. To bridge this gap, we propose \textbf{GSMem}, a zero-shot embodied exploration and reasoning framework built upon 3D Gaussian Splatting (3DGS). By explicitly parameterizing continuous geometry and dense appearance, 3DGS serves as a persistent spatial memory that endows the agent with \textit{Spatial Recollection}: the ability to render photorealistic novel views from optimal, previously unoccupied viewpoints. To operationalize this, GSMem employs a retrieval mechanism that simultaneously leverages parallel object-level scene graphs and semantic-level language fields. This complementary design robustly localizes target regions, enabling the agent to ``hallucinate'' optimal views for high-fidelity Vision-Language Model (VLM) reasoning. Furthermore, we introduce a hybrid exploration strategy that combines VLM-driven semantic scoring with a 3DGS-based coverage objective, balancing task-aware exploration with geometric coverage. Extensive experiments on embodied question answering and lifelong navigation demonstrate the robustness and effectiveness of our framework
>
---
#### [new 055] Computationally Efficient Density-Driven Optimal Control via Analytical KKT Reduction and Contractive MPC
- **分类: math.OC; cs.MA; cs.RO**

- **简介: 该论文属于多智能体系统控制任务，解决密度驱动最优控制的计算效率问题。通过分析KKT结构降维，提出可扩展的QP方法，提升预测控制性能。**

- **链接: [https://arxiv.org/pdf/2603.18503](https://arxiv.org/pdf/2603.18503)**

> **作者:** Julian Martinez; Kooktae Lee
>
> **摘要:** Efficient coordination for collective spatial distribution is a fundamental challenge in multi-agent systems. Prior research on Density-Driven Optimal Control (D2OC) established a framework to match agent trajectories to a desired spatial distribution. However, implementing this as a predictive controller requires solving a large-scale Karush-Kuhn-Tucker (KKT) system, whose computational complexity grows cubically with the prediction horizon. To resolve this, we propose an analytical structural reduction that transforms the T-horizon KKT system into a condensed quadratic program (QP). This formulation achieves O(T) linear scalability, significantly reducing the online computational burden compared to conventional O(T^3) approaches. Furthermore, to ensure rigorous convergence in dynamic environments, we incorporate a contractive Lyapunov constraint and prove the Input-to-State Stability (ISS) of the closed-loop system against reference propagation drift. Numerical simulations verify that the proposed method facilitates rapid density coverage with substantial computational speed-up, enabling long-horizon predictive control for large-scale multi-agent swarms.
>
---
#### [new 056] Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于场景理解任务，解决多模态大语言模型的空间感知不足问题。通过利用视频生成模型的隐式3D先验，提升其几何推理与物理理解能力。**

- **链接: [https://arxiv.org/pdf/2603.19235](https://arxiv.org/pdf/2603.19235)**

> **作者:** Xianjin Wu; Dingkang Liang; Tianrui Feng; Kui Xia; Yumeng Zhang; Xiaofan Li; Xiao Tan; Xiang Bai
>
> **备注:** 31 pages, 12 figures
>
> **摘要:** While Multimodal Large Language Models demonstrate impressive semantic capabilities, they often suffer from spatial blindness, struggling with fine-grained geometric reasoning and physical dynamics. Existing solutions typically rely on explicit 3D modalities or complex geometric scaffolding, which are limited by data scarcity and generalization challenges. In this work, we propose a paradigm shift by leveraging the implicit spatial prior within large-scale video generation models. We posit that to synthesize temporally coherent videos, these models inherently learn robust 3D structural priors and physical laws. We introduce VEGA-3D (Video Extracted Generative Awareness), a plug-and-play framework that repurposes a pre-trained video diffusion model as a Latent World Simulator. By extracting spatiotemporal features from intermediate noise levels and integrating them with semantic representations via a token-level adaptive gated fusion mechanism, we enrich MLLMs with dense geometric cues without explicit 3D supervision. Extensive experiments across 3D scene understanding, spatial reasoning, and embodied manipulation benchmarks demonstrate that our method outperforms state-of-the-art baselines, validating that generative priors provide a scalable foundation for physical-world understanding. Code is publicly available at this https URL.
>
---
#### [new 057] Semantic Segmentation and Depth Estimation for Real-Time Lunar Surface Mapping Using 3D Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于实时月面地图构建任务，解决复杂环境下精准感知与资源限制问题，结合语义分割和深度估计，使用3DGS实现高精度地图重建。**

- **链接: [https://arxiv.org/pdf/2603.18218](https://arxiv.org/pdf/2603.18218)**

> **作者:** Guillem Casadesus Vila; Adam Dai; Grace Gao
>
> **摘要:** Navigation and mapping on the lunar surface require robust perception under challenging conditions, including poorly textured environments, high-contrast lighting, and limited computational resources. This paper presents a real-time mapping framework that integrates dense perception models with a 3D Gaussian Splatting (3DGS) representation. We first benchmark several models on synthetic datasets generated with the LuPNT simulator, selecting a stereo dense depth estimation model based on Gated Recurrent Units for its balance of speed and accuracy in depth estimation, and a convolutional neural network for its superior performance in detecting semantic segments. Using ground truth poses to decouple the local scene understanding from the global state estimation, our pipeline reconstructs a 120-meter traverse with a geometric height accuracy of approximately 3 cm, outperforming a traditional point cloud baseline without LiDAR. The resulting 3DGS map enables novel view synthesis and serves as a foundation for a full SLAM system, where its capacity for joint map and pose optimization would offer significant advantages. Our results demonstrate that combining semantic segmentation and dense depth estimation with learned map representations is an effective approach for creating detailed, large-scale maps to support future lunar surface missions.
>
---
#### [new 058] HEP Statistical Inference for UAV Fault Detection: CLs, LRT, and SBI Applied to Blade Damage
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文将粒子物理统计方法应用于无人机螺旋桨故障检测，解决故障识别与量化问题，采用LRT、CLs和SNPE方法实现准确检测与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2603.18546](https://arxiv.org/pdf/2603.18546)**

> **作者:** Khushiyant
>
> **备注:** 12 Pages, 8 Figures
>
> **摘要:** This paper transfers three statistical methods from particle physics to multirotor propeller fault detection: the likelihood ratio test (LRT) for binary detection, the CLs modified frequentist method for false alarm rate control, and sequential neural posterior estimation (SNPE) for quantitative fault characterization. Operating on spectral features tied to rotor harmonic physics, the system returns three outputs: binary detection, controlled false alarm rates, and calibrated posteriors over fault severity and motor location. On UAV-FD, a hexarotor dataset of 18 real flights with 5% and 10% blade damage, leave-one-flight-out cross-validation gives AUC 0.862 +/- 0.007 (95% CI: 0.849--0.876), outperforming CUSUM (0.708 +/- 0.010), autoencoder (0.753 +/- 0.009), and LSTM autoencoder (0.551). At 5% false alarm rate the system detects 93% of significant and 81% of subtle blade damage. On PADRE, a quadrotor platform, AUC reaches 0.986 after refitting only the generative models. SNPE gives a full posterior over fault severity (90% credible interval coverage 92--100%, MAE 0.012), so the output includes uncertainty rather than just a point estimate or fault flag. Per-flight sequential detection achieves 100% fault detection with 94% overall accuracy.
>
---
#### [new 059] DROID-SLAM in the Wild
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉SLAM任务，解决动态环境中定位与建图问题。通过估计像素级不确定性，提升系统在复杂场景中的鲁棒性与实时性。**

- **链接: [https://arxiv.org/pdf/2603.19076](https://arxiv.org/pdf/2603.19076)**

> **作者:** Moyang Li; Zihan Zhu; Marc Pollefeys; Daniel Barath
>
> **备注:** CVPR 2026, Project Page: this https URL
>
> **摘要:** We present a robust, real-time RGB SLAM system that handles dynamic environments by leveraging differentiable Uncertainty-aware Bundle Adjustment. Traditional SLAM methods typically assume static scenes, leading to tracking failures in the presence of motion. Recent dynamic SLAM approaches attempt to address this challenge using predefined dynamic priors or uncertainty-aware mapping, but they remain limited when confronted with unknown dynamic objects or highly cluttered scenes where geometric mapping becomes unreliable. In contrast, our method estimates per-pixel uncertainty by exploiting multi-view visual feature inconsistency, enabling robust tracking and reconstruction even in real-world environments. The proposed system achieves state-of-the-art camera poses and scene geometry in cluttered dynamic scenarios while running in real time at around 10 FPS. Code and datasets are available at this https URL.
>
---
#### [new 060] Action Draft and Verify: A Self-Verifying Framework for Vision-Language-Action Model
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种自验证框架ADV，用于视觉-语言-动作模型。旨在提升连续动作生成的准确性和鲁棒性，通过结合扩散模型与自回归方法，提高模拟和真实环境中的成功率。**

- **链接: [https://arxiv.org/pdf/2603.18091](https://arxiv.org/pdf/2603.18091)**

> **作者:** Chen Zhao; Zhuoran Wang; Haoyang Li; Shifeng Bao; Guanlin Li; Youhe Feng; Yang Li; Jie Tang; Jing Zhang
>
> **摘要:** Vision-Language-Action (VLA) models have recently demonstrated strong performance across embodied tasks. Modern VLAs commonly employ diffusion action experts to efficiently generate high-precision continuous action chunks, while auto-regressive generation can be slower and less accurate at low-level control. Yet auto-regressive paradigms still provide complementary priors that can improve robustness and generalization in out-of-distribution environments. To leverage both paradigms, we propose Action-Draft-and-Verify (ADV): diffusion action expert drafts multiple candidate action chunks, and the VLM selects one by scoring all candidates in a single forward pass with a perplexity-style metric. Under matched backbones, training data, and action-chunk length, ADV improves success rate by +4.3 points in simulation and +19.7 points in real-world over diffusion-based baseline, with a single-pass VLM reranking overhead.
>
---
## 更新

#### [replaced 001] Aegis: Automated Error Generation and Attribution for Multi-Agent Systems
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多智能体系统错误分析任务，旨在解决错误标注数据稀缺问题。提出Aegis框架，自动生成带标注的错误数据，提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2509.14295](https://arxiv.org/pdf/2509.14295)**

> **作者:** Fanqi Kong; Ruijie Zhang; Huaxiao Yin; Guibin Zhang; Xiaofei Zhang; Ziang Chen; Zhaowei Zhang; Xiaoyuan Zhang; Song-Chun Zhu; Xue Feng
>
> **摘要:** Large language model based multi-agent systems (MAS) have unlocked significant advancements in tackling complex problems, but their increasing capability introduces a structural fragility that makes them difficult to debug. A key obstacle to improving their reliability is the severe scarcity of large-scale, diverse datasets for error attribution, as existing resources rely on costly and unscalable manual annotation. To address this bottleneck, we introduce Aegis, a novel framework for Automated error generation and attribution for multi-agent systems. Aegis constructs a large dataset of 9,533 trajectories with annotated faulty agents and error modes, covering diverse MAS architectures and task domains. This is achieved using a LLM-based manipulator that can adaptively inject context-aware errors into successful execution trajectories. Leveraging fine-grained labels and the structured arrangement of positive-negative sample pairs, Aegis supports three different learning paradigms: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. We develop learning methods for each paradigm. Comprehensive experiments show that trained models consistently achieve substantial improvements in error attribution. Notably, several of our fine-tuned LLMs demonstrate performance competitive with or superior to proprietary models an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at this https URL.
>
---
#### [replaced 002] AdaptPNP: Integrating Prehensile and Non-Prehensile Skills for Adaptive Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出AdaptPNP框架，解决机器人在无法抓取时的混合操作任务。通过整合预握与非预握技能，实现多步骤、自适应的机械臂操作。**

- **链接: [https://arxiv.org/pdf/2511.11052](https://arxiv.org/pdf/2511.11052)**

> **作者:** Jinxuan Zhu; Chenrui Tie; Xinyi Cao; Yuran Wang; Jingxiang Guo; Zixuan Chen; Haonan Chen; Junting Chen; Yangyu Xiao; Ruihai Wu; Lin Shao
>
> **摘要:** Non-prehensile (NP) manipulation, in which robots alter object states without forming stable grasps (for example, pushing, poking, or sliding), significantly broadens robotic manipulation capabilities when grasping is infeasible or insufficient. However, enabling a unified framework that generalizes across different tasks, objects, and environments while seamlessly integrating non-prehensile and prehensile (P) actions remains challenging: robots must determine when to invoke NP skills, select the appropriate primitive for each context, and compose P and NP strategies into robust, multi-step plans. We introduce ApaptPNP, a vision-language model (VLM)-empowered task and motion planning framework that systematically selects and combines P and NP skills to accomplish diverse manipulation objectives. Our approach leverages a VLM to interpret visual scene observations and textual task descriptions, generating a high-level plan skeleton that prescribes the sequence and coordination of P and NP actions. A digital-twin based object-centric intermediate layer predicts desired object poses, enabling proactive mental rehearsal of manipulation sequences. Finally, a control module synthesizes low-level robot commands, with continuous execution feedback enabling online task plan refinement and adaptive replanning through the VLM. We evaluate ApaptPNP across representative P&NP hybrid manipulation tasks in both simulation and real-world environments. These results underscore the potential of hybrid P&NP manipulation as a crucial step toward general-purpose, human-level robotic manipulation capabilities. Project Website: this https URL
>
---
#### [replaced 003] Multimodal Fused Learning for Solving the Generalized Traveling Salesman Problem in Robotic Task Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人任务规划领域，解决广义旅行商问题（GTSP）。提出多模态融合学习框架，结合图与图像表示，提升规划效率与准确性。**

- **链接: [https://arxiv.org/pdf/2506.16931](https://arxiv.org/pdf/2506.16931)**

> **作者:** Jiaqi Chen; Mingfeng Fan; Xuefeng Zhang; Jingsong Liang; Yuhong Cao; Guohua Wu; Guillaume Adrien Sartoretti
>
> **备注:** 14 pages, 6 figures, under review
>
> **摘要:** Effective and efficient task planning is essential for mobile robots, especially in applications like warehouse retrieval and environmental monitoring. These tasks often involve selecting one location from each of several target clusters, forming a Generalized Traveling Salesman Problem (GTSP) that remains challenging to solve both accurately and efficiently. To address this, we propose a Multimodal Fused Learning (MMFL) framework that leverages both graph and image-based representations to capture complementary aspects of the problem, and learns a policy capable of generating high-quality task planning schemes in real time. Specifically, we first introduce a coordinate-based image builder that transforms GTSP instances into spatially informative representations. We then design an adaptive resolution scaling strategy to enhance adaptability across different problem scales, and develop a multimodal fusion module with dedicated bottlenecks that enables effective integration of geometric and spatial features. Extensive experiments show that our MMFL approach significantly outperforms state-of-the-art methods across various GTSP instances while maintaining the computational efficiency required for real-time robotic applications. Physical robot tests further validate its practical effectiveness in real-world scenarios.
>
---
#### [replaced 004] Interleaving Scheduling and Motion Planning with Incremental Learning of Symbolic Space-Time Motion Abstractions
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于任务与运动规划领域，解决多目标导航中的调度与运动规划问题。通过交替使用调度器和运动规划器，并引入增量学习，提高计划的有效性与可行性。**

- **链接: [https://arxiv.org/pdf/2603.10651](https://arxiv.org/pdf/2603.10651)**

> **作者:** Elisa Tosello; Arthur Bit-Monnot; Davide Lusuardi; Alessandro Valentini; Andrea Micheli
>
> **摘要:** Task and Motion Planning combines high-level task sequencing (what to do) with low-level motion planning (how to do it) to generate feasible, collision-free execution plans. However, in many real-world domains, such as automated warehouses, tasks are predefined, shifting the challenge to if, when, and how to execute them safely and efficiently under resource, time and motion constraints. In this paper, we formalize this as the Scheduling and Motion Planning problem for multi-object navigation in shared workspaces. We propose a novel solution framework that interleaves off-the-shelf schedulers and motion planners in an incremental learning loop. The scheduler generates candidate plans, while the motion planner checks feasibility and returns symbolic feedback, i.e., spatial conflicts and timing adjustments, to guide the scheduler towards motion-feasible solutions. We validate our proposal on logistics and job-shop scheduling benchmarks augmented with motion tasks, using state-of-the-art schedulers and sampling-based motion planners. Our results show the effectiveness of our framework in generating valid plans under complex temporal and spatial constraints, where synchronized motion is critical.
>
---
#### [replaced 005] Direct Data-Driven Predictive Control for a Three-dimensional Cable-Driven Soft Robotic Arm
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制任务，旨在解决软机器人动态控制难题。通过设计3D电缆驱动软机械臂，采用DeePC方法实现精确控制，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2510.08953](https://arxiv.org/pdf/2510.08953)**

> **作者:** Cheng Ouyang; Moeen Ul Islam; Dong Chen; Kaixiang Zhang; Zhaojian Li; Xiaobo Tan
>
> **摘要:** Soft robots offer significant advantages in safety and adaptability, yet achieving precise and dynamic control remains a major challenge due to their inherently complex and nonlinear dynamics. Recently, Data-enabled Predictive Control (DeePC) has emerged as a promising model-free approach that bypasses explicit system identification by directly leveraging input-output data. While DeePC has shown success in other domains, its application to soft robots remains underexplored, particularly for three-dimensional (3D) soft robotic systems. This paper addresses this gap by developing and experimentally validating an effective DeePC framework on a 3D, cable-driven soft arm. Specifically, we design and fabricate a soft robotic arm with a thick tubing backbone for stability, a dense silicone body with large cavities for strength and flexibility, and rigid endcaps for secure termination. Using this platform, we implement DeePC with singular value decomposition (SVD)-based dimension reduction for two key control tasks: fixed-point regulation and trajectory tracking in 3D space. Comparative experiments with a baseline model-based controller demonstrate DeePC's superior accuracy, robustness, and adaptability, highlighting its potential as a practical solution for dynamic control of soft robots.
>
---
#### [replaced 006] Manual2Skill++: Connector-Aware General Robotic Assembly from Instruction Manuals via Vision-Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人装配任务，旨在解决装配中连接关系建模问题。通过视觉语言模型从手册中提取连接信息，构建装配图谱，提升装配准确性与通用性。**

- **链接: [https://arxiv.org/pdf/2510.16344](https://arxiv.org/pdf/2510.16344)**

> **作者:** Chenrui Tie; Shengxiang Sun; Yudi Lin; Yanbo Wang; Zhongrui Li; Zhouhan Zhong; Jinxuan Zhu; Yiman Pang; Haonan Chen; Junting Chen; Ruihai Wu; Lin Shao
>
> **摘要:** Assembly hinges on reliably forming connections between parts; yet most robotic approaches plan assembly sequences and part poses while treating connectors as an afterthought. Connections represent the foundational physical constraints of assembly execution; while task planning sequences operations, the precise establishment of these constraints ultimately determines assembly success. In this paper, we treat connections as explicit, primary entities in assembly representation, directly encoding connector types, specifications, and locations for every assembly step. Drawing inspiration from how humans learn assembly tasks through step-by-step instruction manuals, we present Manual2Skill++, a vision-language framework that automatically extracts structured connection information from assembly manuals. We encode assembly tasks as hierarchical graphs where nodes represent parts and sub-assemblies, and edges explicitly model connection relationships between components. A large-scale vision-language model parses symbolic diagrams and annotations in manuals to instantiate these graphs, leveraging the rich connection knowledge embedded in human-designed instructions. We curate a dataset containing over 20 assembly tasks with diverse connector types to validate our representation extraction approach, and evaluate the complete task understanding-to-execution pipeline across four complex assembly scenarios in simulation, spanning furniture, toys, and manufacturing components with real-world correspondence. More detailed information can be found at this https URL
>
---
#### [replaced 007] Distributional Uncertainty and Adaptive Decision-Making in System Co-design
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属于系统协同设计任务，解决多目标不确定性下的设计决策问题。提出分布式协同设计方法，支持自适应决策和概率风险分析。**

- **链接: [https://arxiv.org/pdf/2603.14047](https://arxiv.org/pdf/2603.14047)**

> **作者:** Yujun Huang; Gioele Zardini
>
> **摘要:** Complex engineered systems require coordinated design choices across heterogeneous components under multiple conflicting objectives and uncertain specifications. Monotone co-design provides a compositional framework for such problems by modeling each subsystem as a design problem: a feasible relation between provided functionalities and required resources in partially ordered sets. Existing uncertain co-design models rely on interval bounds, which support worst-case reasoning but cannot represent probabilistic risk or multi-stage adaptive decisions. We develop a distributional extension of co-design that models uncertain design outcomes as distributions over design problems and supports adaptive decision processes through Markov-kernel re-parameterizations. Using quasi-measurable and quasi-universal spaces, we show that the standard co-design interconnection operations remain compositional under this richer notion of uncertainty. We further introduce queries and observations that extract probabilistic design trade-offs, including feasibility probabilities, confidence bounds, and distributions of minimal required resources. A task-driven unmanned aerial vehicle case study illustrates how the framework captures risk-sensitive and information-dependent design choices that interval-based models cannot express.
>
---
#### [replaced 008] Embodied Foundation Models at the Edge: A Survey of Deployment Constraints and Mitigation Strategies
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于边缘计算任务，研究如何在受限条件下部署基础模型。解决系统级协同设计问题，分析内存、计算等约束，提出优化策略。**

- **链接: [https://arxiv.org/pdf/2603.16952](https://arxiv.org/pdf/2603.16952)**

> **作者:** Utkarsh Grover; Ravi Ranjan; Mingyang Mao; Trung Tien Dong; Satvik Praveen; Zhenqi Wu; J. Morris Chang; Tinoosh Mohsenin; Yi Sheng; Agoritsa Polyzou; Eiman Kanjo; Xiaomin Lin
>
> **摘要:** Deploying foundation models in embodied edge systems is fundamentally a systems problem, not just a problem of model compression. Real-time control must operate within strict size, weight, and power constraints, where memory traffic, compute latency, timing variability, and safety margins interact directly. The Deployment Gauntlet organizes these constraints into eight coupled barriers that determine whether embodied foundation models can run reliably in practice. Across representative edge workloads, autoregressive Vision-Language-Action policies are constrained primarily by memory bandwidth, whereas diffusion-based controllers are limited more by compute latency and sustained execution cost. Reliable deployment therefore depends on system-level co-design across memory, scheduling, communication, and model architecture, including decompositions that separate fast control from slower semantic reasoning.
>
---
#### [replaced 009] RoboForge: Physically Optimized Text-guided Whole-Body Locomotion for Humanoids
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，解决文本生成运动在物理执行中的可行性问题。通过物理优化框架，实现文本到全身运动的精准控制。**

- **链接: [https://arxiv.org/pdf/2603.17927](https://arxiv.org/pdf/2603.17927)**

> **作者:** Xichen Yuan; Zhe Li; Bofan Lyu; Kuangji Zuo; Yanshuo Lu; Gen Li; Jianfei Yang
>
> **备注:** 10 pages, 5 figures
>
> **摘要:** While generative models have become effective at producing human-like motions from text, transferring these motions to humanoid robots for physical execution remains challenging. Existing pipelines are often limited by retargeting, where kinematic quality is undermined by physical infeasibility, contact-transition errors, and the high cost of real-world dynamical data. We present a unified latent-driven framework that bridges natural language and whole-body humanoid locomotion through a retarget-free, physics-optimized pipeline. Rather than treating generation and control as separate stages, our key insight is to couple them bidirectionally under physical this http URL introduce a Physical Plausibility Optimization (PP-Opt) module as the coupling interface. In the forward direction, PP-Opt refines a teacher-student distillation policy with a plausibility-centric reward to suppress artifacts such as floating, skating, and penetration. In the backward direction, it converts reward-optimized simulation rollouts into high-quality explicit motion data, which is used to fine-tune the motion generator toward a more physically plausible latent distribution. This bidirectional design forms a self-improving cycle: the generator learns a physically grounded latent space, while the controller learns to execute latent-conditioned behaviors with dynamical this http URL experiments on the Unitree G1 humanoid show that our bidirectional optimization improves tracking accuracy and success rates. Across IsaacLab and MuJoCo, the implicit latent-driven pipeline consistently outperforms conventional explicit retargeting baselines in both precision and stability. By coupling diffusion-based motion generation with physical plausibility optimization, our framework provides a practical path toward deployable text-guided humanoid intelligence.
>
---
#### [replaced 010] Fast Confidence-Aware Human Prediction via Hardware-accelerated Bayesian Inference for Safe Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人安全导航任务，解决人类行为预测问题。通过硬件加速的贝叶斯推断方法，实现高频率、细粒度的人类轨迹预测。**

- **链接: [https://arxiv.org/pdf/2603.01122](https://arxiv.org/pdf/2603.01122)**

> **作者:** Michael Lu; Minh Bui; Xubo Lyu; Mo Chen
>
> **备注:** Update the paper
>
> **摘要:** As robots increasingly integrate into everyday environments, ensuring their safe navigation around humans becomes imperative. Efficient and safe motion planning requires robots to account for human behavior, particularly in constrained spaces such as grocery stores or care homes, where interactions with multiple individuals are common. Prior research has employed Bayesian frameworks to model human rationality based on navigational intent, enabling the prediction of probabilistic trajectories for planning purposes. In this work, we present a simple yet novel approach for confidence-aware prediction that treats future predictions as particles. This framework is highly parallelized and accelerated on an graphics processing unit (GPU). As a result, this enables longer-term predictions at a frequency of 125 Hz and can be easily extended for multi-human predictions. Compared to existing methods, our implementation supports finer prediction time steps, yielding more granular trajectory forecasts. This enhanced resolution allows motion planners to respond effectively to subtle changes in human behavior. We validate our approach through real-world experiments, demonstrating a robot safely navigating among multiple humans with diverse navigational goals. Our results highlight the methods potential for robust and efficient human-robot coexistence in dynamic environments.
>
---
#### [replaced 011] Latent Representations for Visual Proprioception in Inexpensive Robots
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉本体感知任务，旨在解决低成本机器人缺乏精确关节位置信息的问题。通过探索多种潜在表示方法，实现从单目图像中快速回归关节位置。**

- **链接: [https://arxiv.org/pdf/2504.14634](https://arxiv.org/pdf/2504.14634)**

> **作者:** Sahara Sheikholeslami; Ladislau Bölöni
>
> **摘要:** Robotic manipulation requires explicit or implicit knowledge of the robot's joint positions. Precise proprioception is standard in high-quality industrial robots but is often unavailable in inexpensive robots operating in unstructured environments. In this paper, we ask: to what extent can a fast, single-pass regression architecture perform visual proprioception from a single external camera image, available even in the simplest manipulation settings? We explore several latent representations, including CNNs, VAEs, ViTs, and bags of uncalibrated fiducial markers, using fine-tuning techniques adapted to the limited data available. We evaluate the achievable accuracy through experiments on an inexpensive 6-DoF robot.
>
---
#### [replaced 012] Thousand-GPU Large-Scale Training and Optimization Recipe for AI-Native Cloud Embodied Intelligence Infrastructure
- **分类: cs.RO; cs.AI; cs.DC**

- **简介: 该论文属于AI-Native云具身智能任务，解决训练效率与系统瓶颈问题，通过千GPU平台和优化技术提升训练速度与性能。**

- **链接: [https://arxiv.org/pdf/2603.11101](https://arxiv.org/pdf/2603.11101)**

> **作者:** Yongjian Guo; Yunxuan Ma; Haoran Sun; Zhong Guan; Shuai Di; Jing Long; Wanting Xu; Xiaodong Bai; Wen Huang; Yucheng Guo; Chen Zhou; Qiming Yang; Mingxi Luo; Tianyun Zhao; Hedan Yang; Song Wang; Xiaomeng Tian; Xiaolong Xiang; Zhen Sun; Yu Wei; Luqiao Wang; Yuzhen Li; Chenfeng Gu; Junwu Xiong; Yicheng Gong
>
> **摘要:** Embodied intelligence is a key step towards Artificial General Intelligence (AGI), yet its development faces multiple challenges including data, frameworks, infrastructure, and evaluation systems. To address these issues, we have, for the first time in the industry, launched a cloud-based, thousand-GPU distributed training platform for embodied intelligence, built upon the widely adopted LeRobot framework, and have systematically overcome bottlenecks across the entire pipeline. At the data layer, we have restructured the data pipeline to optimize the flow of embodied training data. In terms of training, for the GR00T-N1.5 model, utilizing thousand-GPU clusters and data at the scale of hundreds of millions, the single-round training time has been reduced from 15 hours to just 22 minutes, achieving a 40-fold speedup. At the model layer, by combining variable-length FlashAttention and Data Packing, we have moved from sample redundancy to sequence integration, resulting in a 188% speed increase; {\pi}-0.5 attention optimization has accelerated training by 165%; and FP8 quantization has delivered a 140% speedup. On the infrastructure side, relying on high-performance storage, a 3.2T RDMA network, and a Ray-driven elastic AI data lake, we have achieved deep synergy among data, storage, communication, and computation. We have also built an end-to-end evaluation system, creating a closed loop from training to simulation to assessment. This framework has already been fully validated on thousand-GPU clusters, laying a crucial technical foundation for the development and application of next-generation autonomous intelligent robots, and is expected to accelerate the arrival of the era of human-machine integration.
>
---
#### [replaced 013] From Vocal Instructions to Household Tasks: The Inria TIAGo++ in the euROBIN Service Robots Coopetition
- **分类: cs.RO**

- **简介: 该论文介绍了一个用于服务机器人协作竞赛的系统，解决语音指令到家庭任务的转换问题。工作包括集成控制模块和基于大模型的任务规划，提升机器人自主与远程操作能力。**

- **链接: [https://arxiv.org/pdf/2412.17861](https://arxiv.org/pdf/2412.17861)**

> **作者:** Fabio Amadio; Clemente Donoso; Dionis Totsila; Raphael Lorenzo; Quentin Rouxel; Olivier Rochel; Enrico Mingo Hoffman; Jean-Baptiste Mouret; Serena Ivaldi
>
> **摘要:** This paper describes the Inria team's integrated robotics system used in the 1st euROBIN \textit{coopetition}, during which service robots performed voice-activated household tasks in a kitchen setting. The team developed a modified TIAGo++ platform that leverages a whole-body control stack for autonomous and teleoperated modes, and an LLM-based pipeline for instruction understanding and task planning. The key contributions (opens-sourced) are the integration of these components and the design of custom teleoperation devices, addressing practical challenges in the deployment of service robots.
>
---
#### [replaced 014] HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在提升导航系统在动态环境中的鲁棒性。通过结合轻量拓扑先验与视觉停顿机制，实现更可靠的长期导航。**

- **链接: [https://arxiv.org/pdf/2603.12696](https://arxiv.org/pdf/2603.12696)**

> **作者:** Zihui Yu; Pingcong Li; Bichi Zhang; Sören Schwertfeger
>
> **摘要:** Vision-and-Language Navigation (VLN) is shifting from rigid, step-by-step instruction following toward open-vocabulary, goal-oriented autonomy. Achieving this transition without exhaustive routing prompts requires agents to leverage structural priors. While prior work often assumes computationally heavy 2D/3D metric maps, we instead exploit a lightweight, text-based osmAG (OpenStreetMap Area Graph), a floorplan-level topological representation that is easy to obtain and maintain. However, global planning over a prior map alone is brittle in real-world deployments, where local connectivity can change (e.g., closed doors or crowded passages), leading to execution-time failures. To address this gap, we propose a hierarchical navigation framework HaltNav that couples the robust global planning of osmAG with the local exploration and instruction-grounding capability of VLN. Our approach features an MLLM-based brain module, which is capable of high-level task grounding and obstruction awareness. Conditioned on osmAG, the brain converts the global route into a sequence of localized execution snippets, providing the VLN executor with prior-grounded, goal-centric sub-instructions. Meanwhile, it detects local anomalies via a mechanism we term Reactive Visual Halting (RVH), which interrupts the local control loop, updates osmAG by invalidating the corresponding topology, and triggers replanning to orchestrate a viable detour. To train this halting capability efficiently, we introduce a data synthesis pipeline that leverages generative models to inject realistic obstacles into otherwise navigable scenes, substantially enriching hard negative samples. Extensive experiments demonstrate that our hierarchical framework outperforms several baseline methods without tedious language instructions, and significantly improves robustness for long-horizon vision-language navigation under environmental changes.
>
---
#### [replaced 015] Accelerated Multi-Modal Motion Planning Using Context-Conditioned Diffusion Models
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，旨在解决传统方法在高维空间和复杂环境中的扩展性问题。提出CAMPD模型，利用上下文条件扩散模型生成高质量多模态轨迹，提升泛化能力与效率。**

- **链接: [https://arxiv.org/pdf/2510.14615](https://arxiv.org/pdf/2510.14615)**

> **作者:** Edward Sandra; Lander Vanroye; Dries Dirckx; Ruben Cartuyvels; Jan Swevers; Wilm Decré
>
> **备注:** Accepted for publication at the 2026 IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Classical methods in robot motion planning, such as sampling-based and optimization-based methods, often struggle with scalability towards higher-dimensional state spaces and complex environments. Diffusion models, known for their capability to learn complex, high-dimensional and multi-modal data distributions, provide a promising alternative when applied to motion planning problems and have already shown interesting results. However, most of the current approaches train their model for a single environment, limiting their generalization to environments not seen during training. The techniques that do train a model for multiple environments rely on a specific camera to provide the model with the necessary environmental information and therefore always require that sensor. To effectively adapt to diverse scenarios without the need for retraining, this research proposes Context-Aware Motion Planning Diffusion (CAMPD). CAMPD leverages a classifier-free denoising probabilistic diffusion model, conditioned on sensor-agnostic contextual information. An attention mechanism, integrated in the well-known U-Net architecture, conditions the model on an arbitrary number of contextual parameters. CAMPD is evaluated on a 7-DoF robot manipulator and benchmarked against state-of-the-art approaches on real-world tasks, showing its ability to generalize to unseen environments and generate high-quality, multi-modal trajectories, at a fraction of the time required by existing methods.
>
---
#### [replaced 016] AsgardBench -- Evaluating Visually Grounded Interactive Planning Under Minimal Feedback
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出AsgardBench，用于评估视觉引导的交互式规划能力。任务属于 embodied AI 领域，解决如何在执行中根据视觉反馈调整计划的问题。工作包括构建基准测试集并验证视觉输入对规划的影响。**

- **链接: [https://arxiv.org/pdf/2603.15888](https://arxiv.org/pdf/2603.15888)**

> **作者:** Andrea Tupini; Lars Liden; Reuben Tan; Yu Wang; Jianfeng Gao
>
> **备注:** 19 figures, 6 tables, including appendix
>
> **摘要:** With AsgardBench we aim to evaluate visually grounded, high-level action sequence generation and interactive planning, focusing specifically on plan adaptation during execution based on visual observations rather than navigation or low-level manipulation. In the landscape of embodied AI benchmarks, AsgardBench targets the capability category of interactive planning, which is more sophisticated than offline high-level planning as it requires agents to revise plans in response to environmental feedback, yet remains distinct from low-level execution. Unlike prior embodied AI benchmarks that conflate reasoning with navigation or provide rich corrective feedback that substitutes for perception, AsgardBench restricts agent input to images, action history, and lightweight success/failure signals, isolating interactive planning in a controlled simulator without low-level control noise. The benchmark contains 108 task instances spanning 12 task types, each systematically varied through object state, placement, and scene configuration. These controlled variations create conditional branches in which a single instruction can require different action sequences depending on what the agent observes, emphasizing conditional branching and plan repair during execution. Our evaluations of leading vision language models show that performance drops sharply without visual input, revealing weaknesses in visual grounding and state tracking that ultimately undermine interactive planning. Our benchmark zeroes in on a narrower question: can a model actually use what it sees to adapt a plan when things do not go as expected?
>
---
#### [replaced 017] RhoMorph: Rhombus-shaped Deformable Modular Robots for Stable, Medium-Independent Reconfiguration Motion
- **分类: cs.RO**

- **简介: 该论文属于模块化机器人研究，解决可变形机器人稳定重构问题。提出RhoMorph系统，通过菱形模块实现高效、介质无关的形态变化与运动。**

- **链接: [https://arxiv.org/pdf/2601.19529](https://arxiv.org/pdf/2601.19529)**

> **作者:** Jie Gu; Yirui Sun; Zhihao Xia; Tin Lun Lam; Chunxu Tian; Dan Zhang
>
> **摘要:** In this paper, we present RhoMorph, a novel deformable planar lattice modular self-reconfigurable robot (MSRR) with a rhombus shaped module. Each module consists of a parallelogram skeleton with a single centrally mounted actuator that enables folding and unfolding along its diagonal. The core design philosophy is to achieve essential MSRR functionalities such as morphing, docking, and locomotion with minimal control complexity. This enables a continuous and stable reconfiguration process that is independent of the surrounding medium, allowing the system to reliably form various configurations in diverse environments. To leverage the unique kinematics of RhoMorph, we introduce morphpivoting, a novel motion primitive for reconfiguration that differs from advanced MSRR systems, and propose a strategy for its continuous execution. Finally, a series of physical experiments validate the module's stable reconfiguration ability, as well as its positional and docking accuracy.
>
---
#### [replaced 018] Developing a Discrete-Event Simulator of School Shooter Behavior from VR Data
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于行为模拟任务，旨在解决VR实验中干预策略评估困难的问题。通过构建离散事件模拟器，基于VR数据学习枪手行为模式，实现高效策略评估与训练。**

- **链接: [https://arxiv.org/pdf/2602.06023](https://arxiv.org/pdf/2602.06023)**

> **作者:** Christopher A. McClurg; Alan R. Wagner
>
> **备注:** Accepted for presentation at ANNSIM 2026. Camera-ready version. 13 pages, 4 figures, 4 tables
>
> **摘要:** Virtual reality (VR) has emerged as a powerful tool for evaluating school security measures in high-risk scenarios such as school shootings, offering experimental control and high behavioral fidelity. However, assessing new interventions in VR requires recruiting new participant cohorts for each condition, making large-scale or iterative evaluation difficult. These limitations are especially restrictive when attempting to learn effective intervention strategies, which typically require many training episodes. To address this challenge, we develop a data-driven discrete-event simulator (DES) that models shooter movement and in-region actions as stochastic processes learned from participant behavior in VR studies. We use the simulator to examine the impact of a robot-based shooter intervention strategy. Once shown to reproduce key empirical patterns, the DES enables scalable evaluation and learning of intervention strategies that are infeasible to train directly with human subjects. Overall, this work demonstrates a high-to-mid fidelity simulation workflow that provides a scalable surrogate for developing and evaluating autonomous school-security interventions.
>
---
#### [replaced 019] From Optimizable to Interactable: Mixed Digital Twin-Empowered Testing of Vehicle-Infrastructure Cooperation Systems
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于车辆-基础设施协同系统测试任务，旨在解决传统仿真测试在极端情况下的不足。提出IMPACT框架，结合人类交互与物理虚拟互动，提升测试效果。**

- **链接: [https://arxiv.org/pdf/2603.17497](https://arxiv.org/pdf/2603.17497)**

> **作者:** Jianghong Dong; Chunying Yang; Mengchi Cai; Chaoyi Chen; Qing Xu; Jianqiang Wang; Keqiang Li
>
> **摘要:** Sufficient testing under corner cases is critical for the long-term operation of vehicle-infrastructure cooperation systems (VICS). However, existing corner-case generation methods are primarily AI-driven, and VICS testing under corner cases is typically limited to simulation. In this paper, we introduce an L5 ''Interactable'' level to the VICS digital twin (VICS-DT) taxonomy, extending beyond the conventional L4 ''Optimizable'' level. We further propose an L5-level VICS testing framework, IMPACT (Interactive Mixed-digital-twin Paradigm for Advanced Cooperative vehicle-infrastructure Testing). By enabling direct human interactions with VICS entities, IMPACT incorporates highly uncertain and unpredictable human behaviors into the testing loop, naturally generating high-quality corner cases that complement AI-based methods. Furthermore, the mixedDT-enabled ''Physical-Virtual Action Interaction'' facilitates safe VICS testing under corner cases, incorporating real-world environments and entities rather than purely in simulation. Finally, we implement IMPACT on the I-VIT (Interactive Vehicle-Infrastructure Testbed), and experiments demonstrate its effectiveness. The experimental videos are available at our project website: this https URL.
>
---
#### [replaced 020] TiROD: Tiny Robotics Dataset and Benchmark for Continual Object Detection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于目标检测任务，旨在解决微型机器人在动态环境中持续学习的问题。提出TiROD数据集和基准测试，评估轻量级持续学习策略。**

- **链接: [https://arxiv.org/pdf/2409.16215](https://arxiv.org/pdf/2409.16215)**

> **作者:** Francesco Pasti; Riccardo De Monte; Davide Dalle Pezze; Gian Antonio Susto; Nicola Bellotto
>
> **摘要:** Detecting objects with visual sensors is crucial for numerous mobile robotics applications, from autonomous navigation to inspection. However, robots often need to operate under significant domains shifts from those they were trained in, requiring them to adjust to these changes. Tiny mobile robots, subject to size, power, and computational constraints, face even greater challenges when running and adapting detection models on low-resolution and noisy images. Such adaptability, though, is crucial for real-world deployment, where robots must operate effectively in dynamic and unpredictable settings. In this work, we introduce a new vision benchmark to evaluate lightweight continual learning strategies tailored to the unique characteristics of tiny robotic platforms. Our contributions include: (i) Tiny Robotics Object Detection~(TiROD), a challenging video dataset collected using the onboard camera of a small mobile robot, designed to test object detectors across various domains and classes; (ii) a comprehensive benchmark of several continual learning strategies on different scenarios using NanoDet, a lightweight, real-time object detector for resource-constrained devices.. Our results highlight some key challenges in developing robust and efficient continual learning strategies for object detectors in tiny this http URL; (ii) a benchmark of different continual learning strategies on this dataset using NanoDet, a lightweight object detector. Our results highlight key challenges in developing robust and efficient continual learning strategies for object detectors in tiny robotics.
>
---
#### [replaced 021] Simulation to Rules: A Dual-VLM Framework for Formal Visual Planning
- **分类: cs.RO; cs.AI; cs.CL; cs.SC**

- **简介: 该论文属于视觉规划任务，旨在解决VLM与PDDL在空间和长期推理上的不足。提出VLMFP框架，自动生成PDDL文件，实现跨实例、视觉和规则的泛化能力。**

- **链接: [https://arxiv.org/pdf/2510.03182](https://arxiv.org/pdf/2510.03182)**

> **作者:** Yilun Hao; Yongchao Chen; Chuchu Fan; Yang Zhang
>
> **备注:** 40 pages, 6 figures, 13 tables
>
> **摘要:** Vision Language Models (VLMs) show strong potential for visual planning but struggle with precise spatial and long-horizon reasoning, while Planning Domain Definition Language (PDDL) planners excel at formal long-horizon planning but cannot interpret visual inputs. Recent works combine these complementary advantages by translating visual problems into PDDL. However, while VLMs can generate PDDL problem files satisfactorily, accurately generating PDDL domain files, which encode planning rules, remains challenging and typically requires human expertise or environment interaction. We propose VLMFP, a Dual-VLM-guided framework that autonomously generates both PDDL problem and domain files for formal visual planning. VLMFP combines a SimVLM that simulates action consequences with a GenVLM that generates and iteratively refines PDDL files by aligning symbolic execution with simulated outcomes, enabling multiple levels of generalization across unseen instances, visual appearances, and game rules. We evaluate VLMFP on 6 grid-world domains and demonstrate its generalization capability. On average, SimVLM achieves 87.3% and 86.0% scenario understanding and action simulation for seen and unseen appearances, respectively. With the guidance of SimVLM, VLMFP attains 70.0%, 54.1% planning success on unseen instances in seen and unseen appearances, respectively. We further demonstrate that VLMFP scales to complex long-horizon 3D planning tasks, including multi-robot collaboration and assembly scenarios with partial observability and diverse visual variations. Project page: this https URL.
>
---
#### [replaced 022] PathSpace: Rapid continuous map approximation for efficient SLAM using B-Splines in constrained environments
- **分类: cs.RO**

- **简介: 论文提出PathSpace，属于语义SLAM任务，解决环境表示效率问题。通过B样条实现连续地图近似，提升资源利用率并保持精度。**

- **链接: [https://arxiv.org/pdf/2603.02538](https://arxiv.org/pdf/2603.02538)**

> **作者:** Aduen Benjumea; Andrew Bradley; Alexander Rast; Matthias Rolf
>
> **摘要:** Simultaneous Localization and Mapping (SLAM) plays a crucial role in enabling autonomous vehicles to navigate previously unknown environments. Semantic SLAM mostly extends visual SLAM, leveraging the higher density information available to reason about the environment in a more human-like manner. This allows for better decision making by exploiting prior structural knowledge of the environment, usually in the form of labels. Current semantic SLAM techniques still mostly rely on a dense geometric representation of the environment, limiting their ability to apply constraints based on context. We propose PathSpace, a novel semantic SLAM framework that uses continuous B-splines to represent the environment in a compact manner, while also maintaining and reasoning through the continuous probability density functions required for probabilistic reasoning. This system applies the multiple strengths of B-splines in the context of SLAM to interpolate and fit otherwise discrete sparse environments. We test this framework in the context of autonomous racing, where we exploit pre-specified track characteristics to produce significantly reduced representations at comparable levels of accuracy to traditional landmark based methods and demonstrate its potential in limiting the resources used by a system with minimal accuracy loss.
>
---
#### [replaced 023] Whole-Body Safe Control of Robotic Systems with Koopman Neural Dynamics
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决非线性系统实时安全控制问题。通过学习Koopman模型并结合安全集算法，实现高效、安全的轨迹跟踪与避障。**

- **链接: [https://arxiv.org/pdf/2603.03740](https://arxiv.org/pdf/2603.03740)**

> **作者:** Sebin Jung; Abulikemu Abuduweili; Jiaxing Li; Changliu Liu
>
> **摘要:** Controlling robots with strongly nonlinear, high-dimensional dynamics remains challenging, as direct nonlinear optimization with safety constraints is often intractable in real time. The Koopman operator offers a way to represent nonlinear systems linearly in a lifted space, enabling the use of efficient linear control. We propose a data-driven framework that learns a Koopman embedding and operator from data, and integrates the resulting linear model with the Safe Set Algorithm (SSA). This allows the tracking and safety constraints to be solved in a single quadratic program (QP), ensuring feasibility and optimality without a separate safety filter. We validate the method on a Kinova Gen3 manipulator and a Go2 quadruped, showing accurate tracking and obstacle avoidance.
>
---
#### [replaced 024] 2-D Directed Formation Control Based on Bipolar Coordinates
- **分类: eess.SY; cs.MA; cs.RO**

- **简介: 该论文属于多智能体协同控制任务，解决2-D形状形成问题。提出基于双极坐标的新控制方案，实现全局收敛和鲁棒性，适用于实际应用。**

- **链接: [https://arxiv.org/pdf/2108.00916](https://arxiv.org/pdf/2108.00916)**

> **作者:** Farhad Mehdifar; Charalampos P. Bechlioulis; Julien M. Hendrickx; Dimos V. Dimarogonas
>
> **备注:** 16 pages, 10 figures; minor typos corrected; no change in results
>
> **摘要:** This work proposes a novel 2-D formation control scheme for acyclic triangulated directed graphs (a class of minimally acyclic persistent graphs) based on bipolar coordinates with (almost) global convergence to the desired shape. Prescribed performance control is employed to devise a decentralized control law that avoids singularities and introduces robustness against external disturbances while ensuring predefined transient and steady-state performance for the closed-loop system. Furthermore, it is shown that the proposed formation control scheme can handle formation maneuvering, scaling, and orientation specifications simultaneously. Additionally, the proposed control law is implementable in agents' arbitrarily oriented local coordinate frames using only low-cost onboard vision sensors, which are favorable for practical applications. Finally, a formation maneuvering simulation study verifies the proposed approach.
>
---
#### [replaced 025] UDON: Uncertainty-weighted Distributed Optimization for Multi-Robot Neural Implicit Mapping under Extreme Communication Constraints
- **分类: cs.RO**

- **简介: 该论文属于多机器人神经隐式建图任务，解决极端通信条件下地图重建问题。提出UDON框架，通过不确定性加权和分布式优化提升建图质量。**

- **链接: [https://arxiv.org/pdf/2509.12702](https://arxiv.org/pdf/2509.12702)**

> **作者:** Hongrui Zhao; Xunlan Zhou; Boris Ivanovic; Negar Mehr
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA 2026)
>
> **摘要:** Multi-robot mapping with neural implicit representations enables the compact reconstruction of complex environments. However, it demands robustness against communication challenges like packet loss and limited bandwidth. While prior works have introduced various mechanisms to mitigate communication disruptions, performance degradation still occurs under extremely low communication success rates. This paper presents UDON, a real-time multi-agent neural implicit mapping framework that introduces a novel uncertainty-weighted distributed optimization to achieve high-quality mapping under severe communication deterioration. The uncertainty weighting prioritizes more reliable portions of the map, while the distributed optimization isolates and penalizes mapping disagreement between individual pairs of communicating agents. We conduct extensive experiments on standard benchmark datasets and real-world robot hardware. We demonstrate that UDON significantly outperforms existing baselines, maintaining high-fidelity reconstructions and consistent scene representations even under extreme communication degradation (as low as 1% success rate).
>
---
#### [replaced 026] TrajBooster: Boosting Humanoid Whole-Body Manipulation via Trajectory-Centric Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操控任务，解决 bipedal 人形机器人在缺乏高质量示范数据时的行动空间对齐问题。通过轨迹学习提升动作能力，利用轮式人形数据增强双足 VLA 性能。**

- **链接: [https://arxiv.org/pdf/2509.11839](https://arxiv.org/pdf/2509.11839)**

> **作者:** Jiacheng Liu; Pengxiang Ding; Qihang Zhou; Yuxuan Wu; Da Huang; Zimian Peng; Wei Xiao; Weinan Zhang; Lixin Yang; Cewu Lu; Donglin Wang
>
> **摘要:** Recent Vision-Language-Action models show potential to generalize across embodiments but struggle to quickly align with a new robot's action space when high-quality demonstrations are scarce, especially for bipedal humanoids. We present TrajBooster, a cross-embodiment framework that leverages abundant wheeled-humanoid data to boost bipedal VLA. Our key idea is to use end-effector trajectories as a morphology-agnostic interface. TrajBooster (i) extracts 6D dual-arm end-effector trajectories from real-world wheeled humanoids, (ii) retargets them in simulation to Unitree G1 with a whole-body controller trained via a heuristic-enhanced harmonized online DAgger to lift low-dimensional trajectory references into feasible high-dimensional whole-body actions, and (iii) forms heterogeneous triplets that couple source vision/language with target humanoid-compatible actions to post-pre-train a VLA, followed by only 10 minutes of teleoperation data collection on the target humanoid domain. Deployed on Unitree G1, our policy achieves beyond-tabletop household tasks, enabling squatting, cross-height manipulation, and coordinated whole-body motion with markedly improved robustness and generalization. Results show that TrajBooster allows existing wheeled-humanoid data to efficiently strengthen bipedal humanoid VLA performance, reducing reliance on costly same-embodiment data while enhancing action space understanding and zero-shot skill transfer capabilities. For more details, For more details, please refer to our \href{this https URL}.
>
---
#### [replaced 027] AI-driven Dispensing of Coral Reseeding Devices for Broad-scale Restoration of the Great Barrier Reef
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于珊瑚礁修复任务，旨在解决人工修复效率低的问题。通过AI管道实现珊瑚幼体自动投放，提升修复规模与效率。**

- **链接: [https://arxiv.org/pdf/2509.01019](https://arxiv.org/pdf/2509.01019)**

> **作者:** Scarlett Raine; Emilio Olivastri; Benjamin Moshirian; Tobias Fischer
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Coral reefs are on the brink of collapse, with climate change, ocean acidification, and pollution leading to a projected 70-90% loss of coral species within the next decade. Reef restoration is crucial, but its success hinges on introducing automation to upscale efforts. In this work, we present a highly configurable AI pipeline for the real-time deployment of coral reseeding devices. The pipeline consists of three core components: (i) the image labeling scheme, designed to address data availability and reduce the cost of expert labeling; (ii) the classifier which performs automated analysis of underwater imagery, at the image or patch-level, while also enabling quantitative coral coverage estimation; and (iii) the decision-making module that determines whether deployment should occur based on the classifier's analysis. By reducing reliance on manual experts, our proposed pipeline increases operational range and efficiency of reef restoration. We validate the proposed pipeline at five sites across the Great Barrier Reef, benchmarking its performance against annotations from expert marine scientists. The pipeline achieves 77.8% deployment accuracy, 89.1% accuracy for sub-image patch classification, and real-time model inference at 5.5 frames per second on a Jetson Orin. To address the limited availability of labeled data in this domain and encourage further research, we publicly release a comprehensive, annotated dataset of substrate imagery from the surveyed sites.
>
---
#### [replaced 028] Learning Transferable Friction Models and LuGre Identification Via Physics-Informed Neural Networks
- **分类: cs.RO; cs.LG; eess.SY**

- **简介: 该论文属于机器人摩擦建模任务，旨在解决仿真与实际性能差异问题。通过物理信息神经网络，融合经典模型与学习组件，提升摩擦模型精度与可迁移性。**

- **链接: [https://arxiv.org/pdf/2504.12441](https://arxiv.org/pdf/2504.12441)**

> **作者:** Asutay Ozmen; João P. Hespanha; Katie Byl
>
> **备注:** 7 pages, 8 figures, Accepted to 2026 American Control Conference (ACC)
>
> **摘要:** Accurately modeling friction in robotics remains a core challenge, as robotics simulators like MuJoCo and PyBullet use simplified friction models or heuristics to balance computational efficiency with accuracy, where these simplifications and approximations can lead to substantial differences between simulated and physical performance. In this paper, we present a physics-informed friction estimation framework that enables the integration of well-established friction models with learnable components, requiring only minimal, generic measurement data. Our approach enforces physical consistency yet retains the flexibility to capture complex friction phenomena. We demonstrate, on an underactuated and nonlinear system, that the learned friction models, trained solely on small and noisy datasets, accurately reproduce dynamic friction properties with significantly higher fidelity than the simplified models commonly used in robotics simulators. Crucially, we show that our approach enables the learned models to be transferable to systems they are not trained on. This ability to generalize across multiple systems streamlines friction modeling for complex, underactuated tasks, offering a scalable and interpretable path toward improving friction model accuracy in robotics and control.
>
---
#### [replaced 029] PLM-Net: Perception Latency Mitigation Network for Vision-Based Lateral Control of Autonomous Vehicles
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自动驾驶视觉控制任务，旨在解决感知延迟对横向控制的影响。通过构建PLM-Net框架，预测未来转向动作并实时调整，有效降低控制误差。**

- **链接: [https://arxiv.org/pdf/2407.16740](https://arxiv.org/pdf/2407.16740)**

> **作者:** Aws Khalil; Jaerock Kwon
>
> **摘要:** This study introduces the Perception Latency Mitigation Network (PLM-Net), a modular deep learning framework designed to mitigate perception latency in vision-based imitation-learning lane-keeping systems. Perception latency, defined as the delay between visual sensing and steering actuation, can degrade lateral tracking performance and steering stability. While delay compensation has been extensively studied in classical predictive control systems, its treatment within vision-based imitation-learning architectures under constant and time-varying perception latency remains limited. Rather than reducing latency itself, PLM-Net mitigates its effect on control performance through a plug-in architecture that preserves the original control pipeline. The framework consists of a frozen Base Model (BM), representing an existing lane-keeping controller, and a Timed Action Prediction Model (TAPM), which predicts future steering actions corresponding to discrete latency conditions. Real-time mitigation is achieved by interpolating between model outputs according to the measured latency value, enabling adaptation to both constant and time-varying latency. The framework is evaluated in a closed-loop deterministic simulation environment under fixed-speed conditions to isolate the impact of perception latency. Results demonstrate significant reductions in steering error under multiple latency settings, achieving up to 62% and 78% reductions in Mean Absolute Error (MAE) for constant and time-varying latency cases, respectively. These findings demonstrate the architectural feasibility of modular latency mitigation for vision-based lateral control under controlled simulation settings. The project page including video demonstrations, code, and dataset is publicly released.
>
---
#### [replaced 030] Path Integral Particle Filtering for Hybrid Systems via Saltation Matrices
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决混合系统中接触事件下的状态估计问题。通过路径积分和盐化矩阵，提出一种鲁棒、灵活的粒子滤波方法。**

- **链接: [https://arxiv.org/pdf/2603.01176](https://arxiv.org/pdf/2603.01176)**

> **作者:** Karthik Shaji; Sreeranj Jayadevan; Bo Yuan; Hongzhe Yu; Yongxin Chen
>
> **摘要:** We present an optimal-control-based particle filtering method for state estimation in hybrid systems that undergo intermittent contact with their environments. We follow the path integral filtering framework that exploits the duality between the smoothing problem and optimal control. We leverage saltation matrices to map out the uncertainty propagation during contact events for hybrid systems. The resulting path integral optimal control problem allows for a state estimation algorithm robust to outlier effects, flexible to non-Gaussian noise distributions, that also handles the challenging contact dynamics in hybrid systems. This work offers a computationally efficient and reliable estimation algorithm for hybrid systems with stochastic dynamics. We also present extensive experimental results demonstrating that our approach consistently outperforms strong baselines across multiple settings.
>
---
#### [replaced 031] MLA: A Multisensory Language-Action Model for Multimodal Understanding and Forecasting in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出MLA模型，解决机器人操作中多模态感知与预测问题，通过融合视觉、3D点云和触觉信息提升物理世界建模能力。**

- **链接: [https://arxiv.org/pdf/2509.26642](https://arxiv.org/pdf/2509.26642)**

> **作者:** Zhuoyang Liu; Jiaming Liu; Jiadong Xu; Nuowei Han; Chenyang Gu; Hao Chen; Kaichen Zhou; Renrui Zhang; Kai Chin Hsieh; Kun Wu; Zhengping Che; Jian Tang; Shanghang Zhang
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-language-action models (VLAs) have shown generalization capabilities in robotic manipulation tasks by inheriting from vision-language models (VLMs) and learning action generation. Most VLA models focus on interpreting vision and language to generate actions, whereas robots must perceive and interact within the spatial-physical world. This gap highlights the need for a comprehensive understanding of robotic-specific multisensory information, which is crucial for achieving complex and contact-rich control. To this end, we introduce a multisensory language-action (MLA) model that collaboratively perceives heterogeneous sensory modalities and predicts future multisensory objectives to facilitate physical world modeling. Specifically, to enhance perceptual representations, we propose an encoder-free multimodal alignment scheme that innovatively repurposes the large language model itself as a perception module, directly interpreting multimodal cues by aligning 2D images, 3D point clouds, and tactile tokens through positional correspondence. To further enhance MLA's understanding of physical dynamics, we design a future multisensory generation post-training strategy that enables MLA to reason about semantic, geometric, and interaction information, providing more robust conditions for action generation. For evaluation, the MLA model outperforms the previous state-of-the-art 2D and 3D VLA methods by 12% and 24% in complex, contact-rich real-world tasks, respectively, while also demonstrating improved generalization to unseen configurations.
>
---
#### [replaced 032] U-ARM : Ultra low-cost general teleoperation interface for robot manipulation
- **分类: cs.RO**

- **简介: 该论文提出U-Arm，一种低成本、通用的机器人操作遥操作系统，解决传统接口成本高、兼容性差的问题，通过优化机械设计和控制逻辑，实现高效数据采集与任务执行。**

- **链接: [https://arxiv.org/pdf/2509.02437](https://arxiv.org/pdf/2509.02437)**

> **作者:** Yanwen Zou; Zhaoye Zhou; Chenyang Shi; Zewei Ye; Junda Huang; Yan Ding; Bo Zhao
>
> **摘要:** We propose U-Arm, a low-cost and rapidly adaptable leader-follower teleoperation framework designed to interface with most of commercially available robotic arms. Our system supports teleoperation through three structurally distinct 3D-printed leader arms that share consistent control logic, enabling seamless compatibility with diverse commercial robot configurations. Compared with previous open-source leader-follower interfaces, we further optimized both the mechanical design and servo selection, achieving a bill of materials (BOM) cost of only \$50.5 for the 6-DoF leader arm and \$56.8 for the 7-DoF version. To enhance usability, we mitigate the common challenge in controlling redundant degrees of freedom by %engineering methods mechanical and control optimizations. Experimental results demonstrate that U-Arm achieves 39\% higher data collection efficiency and comparable task success rates across multiple manipulation scenarios compared with Joycon, another low-cost teleoperation interface. We have open-sourced all CAD models of three configs and also provided simulation support for validating teleoperation workflows. We also open-sourced real-world manipulation data collected with U-Arm. The project website is this https URL.
>
---
#### [replaced 033] TwinRL-VLA: Digital Twin-Driven Reinforcement Learning for Real-World Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在真实世界中探索效率低的问题。通过数字孪生与强化学习结合，提升模型的探索能力和泛化性能。**

- **链接: [https://arxiv.org/pdf/2602.09023](https://arxiv.org/pdf/2602.09023)**

> **作者:** Qinwen Xu; Jiaming Liu; Rui Zhou; Shaojun Shi; Nuowei Han; Zhuoyang Liu; Chenyang Gu; Shuo Gu; Yang Yue; Gao Huang; Wenzhao Zheng; Sirui Han; Peng Jia; Shanghang Zhang
>
> **摘要:** Despite strong generalization capabilities, Vision-Language-Action (VLA) models remain constrained by the high cost of expert demonstrations and insufficient real-world interaction. While online reinforcement learning (RL) has shown promise in improving general foundation models, applying RL to VLA manipulation in real-world settings is still hindered by low exploration efficiency and a restricted exploration space. Through systematic real-world experiments, we observe that the effective exploration space of online RL is closely tied to the data distribution of supervised fine-tuning (SFT). Motivated by this observation, we propose TwinRL, a digital twin-real-world collaborative RL framework designed to scale and guide exploration for VLA models. First, a high-fidelity digital twin is efficiently reconstructed from smartphone-captured scenes, enabling realistic bidirectional transfer between real and simulated environments. During the SFT warm-up stage, we introduce an exploration space expansion strategy using digital twins to broaden the support of the data trajectory distribution. Building on this enhanced initialization, we propose a sim-to-real guided exploration strategy to further accelerate online RL. Specifically, TwinRL performs efficient and parallel online RL in the digital twin prior to deployment, effectively bridging the gap between offline and online training stages. Subsequently, we exploit efficient digital twin sampling to identify failure-prone yet informative configurations, which are used to guide targeted human-in-the-loop rollouts on the real robot. In our experiments, TwinRL approaches 100% success in both in-distribution regions covered by real-world demonstrations and out-of-distribution regions, delivering at least a 30% speedup over prior real-world RL methods and requiring only about 20 minutes on average across four tasks.
>
---
