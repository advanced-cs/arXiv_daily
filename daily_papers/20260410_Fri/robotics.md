# 机器人 cs.RO

- **最新发布 51 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] On-Policy Distillation of Language Models for Autonomous Vehicle Motion Planning
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于自动驾驶轨迹规划任务，旨在将大语言模型的知识有效迁移至小模型，解决资源受限环境下的部署问题。通过知识蒸馏和强化学习对比实验，验证了方法的有效性。**

- **链接: [https://arxiv.org/pdf/2604.07944](https://arxiv.org/pdf/2604.07944)**

> **作者:** Amirhossein Afsharrad; Amirhesam Abedsoltan; Ahmadreza Moradipari; Sanjay Lall
>
> **摘要:** Large language models (LLMs) have recently demonstrated strong potential for autonomous vehicle motion planning by reformulating trajectory prediction as a language generation problem. However, deploying capable LLMs in resource-constrained onboard systems remains a fundamental challenge. In this paper, we study how to effectively transfer motion planning knowledge from a large teacher LLM to a smaller, more deployable student model. We build on the GPT-Driver framework, which represents driving scenes as language prompts and generates waypoint trajectories with chain-of-thought reasoning, and investigate two student training paradigms: (i) on-policy generalized knowledge distillation (GKD), which trains the student on its own self-generated outputs using dense token-level feedback from the teacher, and (ii) a dense-feedback reinforcement learning (RL) baseline that uses the teacher's log-probabilities as per-token reward signals in a policy gradient framework. Experiments on the nuScenes benchmark show that GKD substantially outperforms the RL baseline and closely approaches teacher-level performance despite a 5$\times$ reduction in model size. These results highlight the practical value of on-policy distillation as a principled and effective approach to deploying LLM-based planners in autonomous driving systems.
>
---
#### [new 002] Reset-Free Reinforcement Learning for Real-World Agile Driving: An Empirical Study
- **分类: cs.RO**

- **简介: 该论文研究真实世界中无重置强化学习的敏捷驾驶任务，解决模拟与现实性能差距问题。通过对比多种算法，发现残差学习在仿真中有效，但在现实中效果不佳。**

- **链接: [https://arxiv.org/pdf/2604.07672](https://arxiv.org/pdf/2604.07672)**

> **作者:** Kohei Honda; Hirotaka Hosogaya
>
> **备注:** 7 pages, 5 figures,
>
> **摘要:** This paper presents an empirical study of reset-free reinforcement learning (RL) for real-world agile driving, in which a physical 1/10-scale vehicle learns continuously on a slippery indoor track without manual resets. High-speed driving near the limits of tire friction is particularly challenging for learning-based methods because complex vehicle dynamics, actuation delays, and other unmodeled effects hinder both accurate simulation and direct sim-to-real transfer of learned policies. To enable autonomous training on a physical platform, we employ Model Predictive Path Integral control (MPPI) as both the reset policy and the base policy for residual learning, and systematically compare three representative RL algorithms, i.e., PPO, SAC, and TD-MPC2, with and without residual learning in simulation and real-world experiments. Our results reveal a clear gap between simulation and real-world: SAC with residual learning achieves the highest returns in simulation, yet only TD-MPC2 consistently outperforms the MPPI baseline on the physical platform. Moreover, residual learning, while clearly beneficial in simulation, fails to transfer its advantage to the real world and can even degrade performance. These findings reveal that reset-free RL in the real world poses unique challenges absent from simulation, calling for further algorithmic development tailored to training in the wild.
>
---
#### [new 003] A-SLIP: Acoustic Sensing for Continuous In-hand Slip Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决抓取物体滑动的实时检测问题。提出A-SLIP系统，通过多通道声学传感实现连续滑动方向和大小的准确估计。**

- **链接: [https://arxiv.org/pdf/2604.08528](https://arxiv.org/pdf/2604.08528)**

> **作者:** Uksang Yoo; Yuemin Mao; Jean Oh; Jeffrey Ichnowski
>
> **摘要:** Reliable in-hand manipulation requires accurate real-time estimation of slip between a gripper and a grasped object. Existing tactile sensing approaches based on vision, capacitance, or force-torque measurements face fundamental trade-offs in form factor, durability, and their ability to jointly estimate slip direction and magnitude. We present A-SLIP, a multi-channel acoustic sensing system integrated into a parallel-jaw gripper for estimating continuous slip in the grasp plane. The A-SLIP sensor consists of piezoelectric microphones positioned behind a textured silicone contact pad to capture structured contact-induced vibrations. The A-SLIP model processes synchronized multi-channel audio as log-mel spectrograms using a lightweight convolutional network, jointly predicting the presence, direction, and magnitude of slip. Across experiments with robot- and externally induced slip conditions, the fine-tuned four-microphone configuration achieves a mean absolute directional error of 14.1 degrees, outperforms baselines by up to 12 percent in detection accuracy, and reduces directional error by 32 percent. Compared with single-microphone configurations, the multi-channel design reduces directional error by 64 percent and magnitude error by 68 percent, underscoring the importance of spatial acoustic sensing in resolving slip direction ambiguity. We further evaluate A-SLIP in closed-loop reactive control and find that it enables reliable, low-cost, real-time estimation of in-hand slip. Project videos and additional details are available at this https URL.
>
---
#### [new 004] SIM1: Physics-Aligned Simulator as Zero-Shot Data Scaler in Deformable Worlds
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，解决柔性物体模拟与真实世界不匹配的问题。通过物理对齐的仿真引擎SIM1，实现高效数据生成与策略学习。**

- **链接: [https://arxiv.org/pdf/2604.08544](https://arxiv.org/pdf/2604.08544)**

> **作者:** Yunsong Zhou; Hangxu Liu; Xuekun Jiang; Xing Shen; Yuanzhen Zhou; Hui Wang; Baole Fang; Yang Tian; Mulin Yu; Qiaojun Yu; Li Ma; Hengjie Li; Hanqing Wang; Jia Zeng; Jiangmiao Pang
>
> **备注:** Website: this https URL
>
> **摘要:** Robotic manipulation with deformable objects represents a data-intensive regime in embodied learning, where shape, contact, and topology co-evolve in ways that far exceed the variability of rigids. Although simulation promises relief from the cost of real-world data acquisition, prevailing sim-to-real pipelines remain rooted in rigid-body abstractions, producing mismatched geometry, fragile soft dynamics, and motion primitives poorly suited for cloth interaction. We posit that simulation fails not for being synthetic, but for being ungrounded. To address this, we introduce SIM1, a physics-aligned real-to-sim-to-real data engine that grounds simulation in the physical world. Given limited demonstrations, the system digitizes scenes into metric-consistent twins, calibrates deformable dynamics through elastic modeling, and expands behaviors via diffusion-based trajectory generation with quality filtering. This pipeline transforms sparse observations into scaled synthetic supervision with near-demonstration fidelity. Experiments show that policies trained on purely synthetic data achieve parity with real-data baselines at a 1:15 equivalence ratio, while delivering 90% zero-shot success and 50% generalization gains in real-world deployment. These results validate physics-aligned simulation as scalable supervision for deformable manipulation and a practical pathway for data-efficient policy learning.
>
---
#### [new 005] The Sustainability Gap in Robotics: A Large-Scale Survey of Sustainability Awareness in 50,000 Research Articles
- **分类: cs.RO; cs.CY**

- **简介: 该论文属于科技评估任务，旨在分析机器人学研究中可持续性意识的缺失。通过分析5万篇论文，发现可持续性提及率低，提出改进建议。**

- **链接: [https://arxiv.org/pdf/2604.07921](https://arxiv.org/pdf/2604.07921)**

> **作者:** Antun Skuric; Leandro Von Werra; Thomas Wolf
>
> **备注:** 29 pages, 17 figures
>
> **摘要:** We present a large-scale survey of sustainability communication and motivation in robotics research. Our analysis covers nearly 50,000 open-access papers from arXiv's cs.RO category published between 2015 and early 2026. In this study, we quantify how often papers mention social, ecological, and sustainability impacts, and we analyse their alignment with the UN Sustainable Development Goals (SDGs). The results reveal a persistent gap between the field's potential and its stated intent. While a large fraction of robotics papers can be mapped to SDG-relevant domains, explicit sustainability motivation remains remarkably low. Specifically, mentions of sustainability-related impacts are typically below 2%, explicit SDG references stay below 0.1%, and the proportion of sustainability-motivated papers remains below 5%. These trends suggest that while the field of robotics is advancing rapidly, sustainability is not yet a standard part of research framing. We conclude by proposing concrete actions for researchers, conferences, and institutions to close these awareness and motivation gaps, supporting a shift toward more intentional and responsible innovation.
>
---
#### [new 006] EMMa: End-Effector Stability-Oriented Mobile Manipulation for Tracked Rescue Robots
- **分类: cs.RO**

- **简介: 该论文属于救援机器人移动操作任务，旨在解决端效器稳定性问题。提出EMMa框架，优化路径并提升操作稳定性。**

- **链接: [https://arxiv.org/pdf/2604.08292](https://arxiv.org/pdf/2604.08292)**

> **作者:** Yifei Wang; Hao Zhang; Jidong Huang; Shuohang Fang; Haoyao Chen
>
> **备注:** 14 pages, 17 figures
>
> **摘要:** The autonomous operation of tracked mobile manipulators in rescue missions requires not only ensuring the reachability and safety of robot motion but also maintaining stable end-effector manipulation under diverse task demands. However, existing studies have overlooked many end-effector motion properties at both the planning and control levels. This paper presents a motion generation framework for tracked mobile manipulators to achieve stable end-effector operation in complex rescue scenarios. The framework formulates a coordinated path optimization model that couples end-effector and mobile base states and designs compact cost/constraint representations to mitigate nonlinearities and reduce computational complexity. Furthermore, an isolated control scheme with feedforward compensation and feedback regulation is developed to enable coordinated path tracking for the robot. Extensive simulated and real-world experiments on rescue scenarios demonstrate that the proposed framework consistently outperforms SOTA methods across key metrics, including task success rate and end-effector motion stability, validating its effectiveness and robustness in complex mobile manipulation tasks.
>
---
#### [new 007] Fail2Drive: Benchmarking Closed-Loop Driving Generalization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶任务，解决封闭环路驾驶泛化问题。提出Fail2Drive基准，通过配对路线评估模型在分布偏移下的表现，揭示模型缺陷并提供工具促进研究。**

- **链接: [https://arxiv.org/pdf/2604.08535](https://arxiv.org/pdf/2604.08535)**

> **作者:** Simon Gerstenecker; Andreas Geiger; Katrin Renz
>
> **摘要:** Generalization under distribution shift remains a central bottleneck for closed-loop autonomous driving. Although simulators like CARLA enable safe and scalable testing, existing benchmarks rarely measure true generalization: they typically reuse training scenarios at test time. Success can therefore reflect memorization rather than robust driving behavior. We introduce Fail2Drive, the first paired-route benchmark for closed-loop generalization in CARLA, with 200 routes and 17 new scenario classes spanning appearance, layout, behavioral, and robustness shifts. Each shifted route is matched with an in-distribution counterpart, isolating the effect of the shift and turning qualitative failures into quantitative diagnostics. Evaluating multiple state-of-the-art models reveals consistent degradation, with an average success-rate drop of 22.8\%. Our analysis uncovers unexpected failure modes, such as ignoring objects clearly visible in the LiDAR and failing to learn the fundamental concepts of free and occupied space. To accelerate follow-up work, Fail2Drive includes an open-source toolbox for creating new scenarios and validating solvability via a privileged expert policy. Together, these components establish a reproducible foundation for benchmarking and improving closed-loop driving generalization. We open-source all code, data, and tools at this https URL .
>
---
#### [new 008] A Unified Multi-Layer Framework for Skill Acquisition from Imperfect Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，解决现有HRI系统碎片化问题。提出多层框架，提升学习效率、操作直观性和安全性。**

- **链接: [https://arxiv.org/pdf/2604.08341](https://arxiv.org/pdf/2604.08341)**

> **作者:** Zi-Qi Yang; Mehrdad R. Kermani
>
> **备注:** 6 pages, 4 figures. Submitted to a conference proceeding
>
> **摘要:** Current Human-Robot Interaction (HRI) systems for skill teaching are fragmented, and existing approaches in the literature do not offer a cohesive framework that is simultaneously efficient, intuitive, and universally safe. This paper presents a novel, layered control framework that addresses this fundamental gap by enabling robust, compliant Learning from Demonstration (LfD) built upon a foundation of universal robot compliance. The proposed approach is structured in three progressive and interconnected stages. First, we introduce a real-time LfD method that learns both the trajectory and variable impedance from a single demonstration, significantly improving efficiency and reproduction fidelity. To ensure high-quality and intuitive {kinesthetic teaching}, we then present a null-space optimization strategy that proactively manages singularities and provides a consistent interaction feel during human demonstration. Finally, to ensure generalized safety, we introduce a foundational null-space compliance method that enables the entire robot body to compliantly adapt to post-learning external interactions without compromising main task performance. This final contribution transforms the system into a versatile HRI platform, moving beyond end-effector (EE)-specific applications. We validate the complete framework through comprehensive comparative experiments on a 7-DOF KUKA LWR robot. The results demonstrate a safer, more intuitive, and more efficient unified system for a wide range of human-robot collaborative tasks.
>
---
#### [new 009] ActiveGlasses: Learning Manipulation with Active Vision from Ego-centric Human Demonstration
- **分类: cs.RO**

- **简介: 该论文提出ActiveGlasses系统，用于从人类第一视角演示中学习机器人操作。解决真实世界数据收集与零样本迁移问题，通过智能眼镜摄像头实现感知与控制，提升机器人操作能力。**

- **链接: [https://arxiv.org/pdf/2604.08534](https://arxiv.org/pdf/2604.08534)**

> **作者:** Yanwen Zou; Chenyang Shi; Wenye Yu; Han Xue; Jun Lv; Ye Pan; Chuan Wen; Cewu Lu
>
> **摘要:** Large-scale real-world robot data collection is a prerequisite for bringing robots into everyday deployment. However, existing pipelines often rely on specialized handheld devices to bridge the embodiment gap, which not only increases operator burden and limits scalability, but also makes it difficult to capture the naturally coordinated perception-manipulation behaviors of human daily interaction. This challenge calls for a more natural system that can faithfully capture human manipulation and perception behaviors while enabling zero-shot transfer to robotic platforms. We introduce ActiveGlasses, a system for learning robot manipulation from ego-centric human demonstrations with active vision. A stereo camera mounted on smart glasses serves as the sole perception device for both data collection and policy inference: the operator wears it during bare-hand demonstrations, and the same camera is mounted on a 6-DoF perception arm during deployment to reproduce human active vision. To enable zero-transfer, we extract object trajectories from demonstrations and use an object-centric point-cloud policy to jointly predict manipulation and head movement. Across several challenging tasks involving occlusion and precise interaction, ActiveGlasses achieves zero-shot transfer with active vision, consistently outperforms strong baselines under the same hardware setup, and generalizes across two robot platforms.
>
---
#### [new 010] Robust Multi-Agent Target Tracking in Intermittent Communication Environments via Analytical Belief Merging
- **分类: cs.RO**

- **简介: 该论文属于多智能体目标跟踪任务，解决通信受限环境下的信念融合问题。通过解析KL散度优化，提出高效且精确的信念合并方法。**

- **链接: [https://arxiv.org/pdf/2604.07575](https://arxiv.org/pdf/2604.07575)**

> **作者:** Mohamed Abdelnaby; Samuel Honor; Kevin Leahy
>
> **摘要:** Autonomous multi-agent target tracking in GPS-denied and communication-restricted environments (e.g., underwater exploration, subterranean search and rescue, and adversarial domains) forces agents to operate independently and only exchange information during brief reconnection windows. Because transmitting complete observation and trajectory histories is bandwidth-exhaustive, exchanging probabilistic belief maps serves as a highly efficient proxy that preserves the topology of agent knowledge. While minimizing divergence metrics to merge these decentralized beliefs is conceptually sound, traditional approaches often rely on numerical solvers that introduce critical quantization errors and artificial noise floors. In this paper, we formulate the decentralized belief merging problem as Forward and Reverse Kullback-Leibler (KL) divergence optimizations and derive their exact closed-form analytical solutions. By deploying these derivations, we mathematically eliminate optimization artifacts, achieving perfect mathematical fidelity while reducing the computational complexity of the belief merge to $\mathcal{O}(N|S|)$ scalar operations. Furthermore, we propose a novel spatially-aware visit-weighted KL merging strategy that dynamically weighs agent beliefs based on their physical visitation history. Validated across tens of thousands of distributed simulations, extensive sensitivity analysis demonstrates that our proposed method significantly suppresses sensor noise and outperforms standard analytical means in environments characterized by highly degraded sensors and prolonged communication intervals.
>
---
#### [new 011] OpenPRC: A Unified Open-Source Framework for Physics-to-Task Evaluation in Physical Reservoir Computing
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出OpenPRC框架，解决PRC系统评估碎片化问题，整合模拟与实验数据，实现统一的物理到任务评估。**

- **链接: [https://arxiv.org/pdf/2604.07423](https://arxiv.org/pdf/2604.07423)**

> **作者:** Yogesh Phalak; Wen Sin Lor; Apoorva Khairnar; Benjamin Jantzen; Noel Naughton; Suyi Li
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** Physical Reservoir Computing (PRC) leverages the intrinsic nonlinear dynamics of physical substrates, mechanical, optical, spintronic, and beyond, as fixed computational reservoirs, offering a compelling paradigm for energy-efficient and embodied machine learning. However, the practical workflow for developing and evaluating PRC systems remains fragmented: existing tools typically address only isolated parts of the pipeline, such as substrate-specific simulation, digital reservoir benchmarking, or readout training. What is missing is a unified framework that can represent both high-fidelity simulated trajectories and real experimental measurements through the same data interface, enabling reproducible evaluation, analysis, and physics-aware optimization across substrates and data sources. We present OpenPRC, an open-source Python framework that fills this gap through a schema-driven physics-to-task pipeline built around five modules: a GPU-accelerated hybrid RK4-PBD physics engine (demlat), a video-based experimental ingestion layer (this http URL), a modular learning layer (reservoir), information-theoretic analysis and benchmarking tools (analysis), and physics-aware optimization (optimize). A universal HDF5 schema enforces reproducibility and interoperability, allowing GPU-simulated and experimentally acquired trajectories to enter the same downstream workflow without modification. Demonstrated capabilities include simulations of Origami tessellations, video-based trajectory extraction from a physical reservoir, and a common interface for standardized PRC benchmarking, correlation diagnostics, and capacity analysis. The longer-term vision is to serve as a standardizing layer for the PRC community, compatible with external physics engines including PyBullet, PyElastica, and MERLIN.
>
---
#### [new 012] CMP: Robust Whole-Body Tracking for Loco-Manipulation via Competence Manifold Projection
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决腿部移动机械臂在异常输入下的跟踪鲁棒性问题。提出CMP方法，通过安全流形投影提升系统对分布外输入的适应能力，保持任务性能。**

- **链接: [https://arxiv.org/pdf/2604.07457](https://arxiv.org/pdf/2604.07457)**

> **作者:** Ziyang Cheng; Haoyu Wei; Hang Yin; Xiuwei Xu; Bingyao Yu; Jie Zhou; Jiwen Lu
>
> **备注:** 14 pages, 8 figures. Under review. Project page and videos: this https URL
>
> **摘要:** While decoupled control schemes for legged mobile manipulators have shown robustness, learning holistic whole-body control policies for tracking global end-effector poses remains fragile against Out-of-Distribution (OOD) inputs induced by sensor noise or infeasible user commands. To improve robustness against these perturbations without sacrificing task performance and continuity, we propose Competence Manifold Projection (CMP). Specifically, we utilize a Frame-Wise Safety Scheme that transforms the infinite-horizon safety constraint into a computationally efficient single-step manifold inclusion. To instantiate this competence manifold, we employ a Lower-Bounded Safety Estimator that distinguishes unmastered intentions from the training distribution. We then introduce an Isomorphic Latent Space (ILS) that aligns manifold geometry with safety probability, enabling efficient O(1) seamless defense against arbitrary OOD intents. Experiments demonstrate that CMP achieves up to a 10-fold survival rate improvement in typical OOD scenarios where baselines suffer catastrophic failure, incurring under 10% tracking degradation. Notably, the system exhibits emergent ``best-effort'' generalization behaviors to progressively accomplish OOD goals by adhering to the competence boundaries. Result videos are available at: this https URL.
>
---
#### [new 013] Spatio-Temporal Grounding of Large Language Models from Perception Streams
- **分类: cs.RO**

- **简介: 该论文属于视频理解任务，旨在解决LLM在空间和时间关系上的不足。通过引入FESTS框架，将自然语言转化为SpRE，提升模型对视频中时空关系的推理能力。**

- **链接: [https://arxiv.org/pdf/2604.07592](https://arxiv.org/pdf/2604.07592)**

> **作者:** Jacob Anderson; Bardh Hoxha; Georgios Fainekos; Hideki Okamoto; Danil Prokhorov
>
> **摘要:** Embodied-AI agents must reason about how objects move and interact in 3-D space over time, yet existing smaller frontier Large Language Models (LLMs) still mis-handle fine-grained spatial relations, metric distances, and temporal orderings. We introduce the general framework Formally Explainable Spatio-Temporal Scenes (FESTS) that injects verifiable spatio-temporal supervision into an LLM by compiling natural-language queries into Spatial Regular Expression (SpRE) -- a language combining regular expression syntax with S4u spatial logic and extended here with universal and existential quantification. The pipeline matches each SpRE against any structured video log and exports aligned (query, frames, match, explanation) tuples, enabling unlimited training data without manual labels. Training a 3-billion-parameter model on 27k such tuples boosts frame-level F1 from 48.5% to 87.5%, matching GPT-4.1 on complex spatio-temporal reasoning while remaining two orders of magnitude smaller, and, hence, enabling spatio-temporal intelligence for Video LLM.
>
---
#### [new 014] Semantic-Aware UAV Command and Control for Efficient IoT Data Collection
- **分类: cs.RO**

- **简介: 该论文属于无人机协同任务，旨在解决物联网数据高效采集问题。通过语义通信与无人机控制结合，提升图像重建质量。**

- **链接: [https://arxiv.org/pdf/2604.08153](https://arxiv.org/pdf/2604.08153)**

> **作者:** Assane Sankara; Daniel Bonilla Licea; Hajar El Hammouti
>
> **备注:** Accepted for publication at the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) have emerged as a key enabler technology for data collection from Internet of Things (IoT) devices. However, effective data collection is challenged by resource constraints and the need for real-time decision-making. In this work, we propose a novel framework that integrates semantic communication with UAV command-and-control (C&C) to enable efficient image data collection from IoT devices. Each device uses Deep Joint Source-Channel Coding (DeepJSCC) to generate a compact semantic latent representation of its image to enable image reconstruction even under partial transmission. A base station (BS) controls the UAV's trajectory by transmitting acceleration commands. The objective is to maximize the average quality of reconstructed images by maintaining proximity to each device for a sufficient duration within a fixed time horizon. To address the challenging trade-off and account for delayed C&C signals, we model the problem as a Markov Decision Process and propose a Double Deep Q-Learning (DDQN)-based adaptive flight policy. Simulation results show that our approach outperforms baseline methods such as greedy and traveling salesman algorithms, in both device coverage and semantic reconstruction quality.
>
---
#### [new 015] ViVa: A Video-Generative Value Model for Robot Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ViVa，一种基于视频生成的值模型，用于机器人强化学习。解决长时序任务中价值估计不准确的问题，通过结合视觉-语言-动作信息，提升真实场景下的任务执行能力。**

- **链接: [https://arxiv.org/pdf/2604.08168](https://arxiv.org/pdf/2604.08168)**

> **作者:** Jindi Lv; Hao Li; Jie Li; Yifei Nie; Fankun Kong; Yang Wang; Xiaofeng Wang; Zheng Zhu; Chaojun Ni; Qiuping Deng; Hengtao Li; Jiancheng Lv; Guan Huang
>
> **摘要:** Vision-language-action (VLA) models have advanced robot manipulation through large-scale pretraining, but real-world deployment remains challenging due to partial observability and delayed feedback. Reinforcement learning addresses this via value functions, which assess task progress and guide policy improvement. However, existing value models built on vision-language models (VLMs) struggle to capture temporal dynamics, undermining reliable value estimation in long-horizon tasks. In this paper, we propose ViVa, a video-generative value model that repurposes a pretrained video generator for value estimation. Taking the current observation and robot proprioception as input, ViVa jointly predicts future proprioception and a scalar value for the current state. By leveraging the spatiotemporal priors of a pretrained video generator, our approach grounds value estimation in anticipated embodiment dynamics, moving beyond static snapshots to intrinsically couple value with foresight. Integrated into RECAP, ViVa delivers substantial improvements on real-world box assembly. Qualitative analysis across all three tasks confirms that ViVa produces more reliable value signals, accurately reflecting task progress. By leveraging spatiotemporal priors from video corpora, ViVa also generalizes to novel objects, highlighting the promise of video-generative models for value estimation.
>
---
#### [new 016] AgiPIX: Bridging Simulation and Reality in Indoor Aerial Inspection
- **分类: cs.RO**

- **简介: 该论文属于室内自主飞行任务，解决仿真与现实间平台不兼容问题。提出AgiPIX平台，实现硬件与软件协同设计，支持快速迭代与真实环境应用。**

- **链接: [https://arxiv.org/pdf/2604.08009](https://arxiv.org/pdf/2604.08009)**

> **作者:** Sasanka Kuruppu Arachchige; Juan Jose Garcia; Changda Tian; Lauri Suomela; Panos Trahanias; Adriana Tapus; Joni-Kristian Kämäräinen
>
> **备注:** Submitted for ICUAS 2026, 9 pages, 11 figures
>
> **摘要:** Autonomous indoor flight for critical asset inspection presents fundamental challenges in perception, planning, control, and learning. Despite rapid progress, there is still a lack of a compact, active-sensing, open-source platform that is reproducible across simulation and real-world operation. To address this gap, we present Agipix, a co-designed open hardware and software platform for indoor aerial autonomy and critical asset inspection. Agipix features a compact, hardware-synchronized active-sensing platform with onboard GPU-accelerated compute that is capable of agile flight; a containerized ROS~2-based modular autonomy stack; and a photorealistic digital twin of the hardware platform together with a reliable UI. These elements enable rapid iteration via zero-shot transfer of containerized autonomy components between simulation and real flights. We demonstrate trajectory tracking and exploration performance using onboard sensing in industrial indoor environments. All hardware designs, simulation assets, and containerized software are released openly together with documentation.
>
---
#### [new 017] RoboAgent: Chaining Basic Capabilities for Embodied Task Planning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于 embodied task planning 任务，旨在解决VLM在多轮交互、长距离推理中的性能不足。提出RoboAgent框架，通过调用不同子能力分解任务，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2604.07774](https://arxiv.org/pdf/2604.07774)**

> **作者:** Peiran Xu; Jiaqi Zheng; Yadong Mu
>
> **备注:** CVPR 2026
>
> **摘要:** This paper focuses on embodied task planning, where an agent acquires visual observations from the environment and executes atomic actions to accomplish a given task. Although recent Vision-Language Models (VLMs) have achieved impressive results in multimodal understanding and reasoning, their performance remains limited when applied to embodied planning that involves multi-turn interaction, long-horizon reasoning, and extended context analysis. To bridge this gap, we propose RoboAgent, a capability-driven planning pipeline in which the model actively invokes different sub-capabilities. Each capability maintains its own context, and produces intermediate reasoning results or interacts with the environment according to the query given by a scheduler. This framework decomposes complex planning into a sequence of basic vision-language problems that VLMs can better address, enabling a more transparent and controllable reasoning process. The scheduler and all capabilities are implemented with a single VLM, without relying on external tools. To train this VLM, we adopt a multi-stage paradigm that consists of: (1) behavior cloning with expert plans, (2) DAgger training using trajectories collected by the model, and (3) reinforcement learning guided by an expert policy. Across these stages, we exploit the internal information of the environment simulator to construct high-quality supervision for each capability, and we further introduce augmented and synthetic data to enhance the model's performance in more diverse scenarios. Extensive experiments on widely used embodied task planning benchmarks validate the effectiveness of the proposed approach. Our codes will be available at this https URL.
>
---
#### [new 018] State and Trajectory Estimation of Tensegrity Robots via Factor Graphs and Chebyshev Polynomials
- **分类: cs.RO**

- **简介: 该论文属于状态估计任务，解决张力结构机器人动态复杂性带来的状态估计难题。提出基于因子图和切比雪夫多项式的两阶段方法，实现高精度连续时间状态估计。**

- **链接: [https://arxiv.org/pdf/2604.08185](https://arxiv.org/pdf/2604.08185)**

> **作者:** Edgar Granados; Patrick Meng; Charles Tang; Shrimed Sangani; William R. Johnson III; Rebecca Kramer-Bottiglio; Kostas Bekris
>
> **备注:** Accepted at Robotsoft 2026
>
> **摘要:** Tensegrity robots offer compliance and adaptability, but their nonlinear, and underconstrained dynamics make state estimation challenging. Reliable continuous-time estimation of all rigid links is crucial for closed-loop control, system identification, and machine learning; however, conventional methods often fall short. This paper proposes a two-stage approach for robust state or trajectory estimation (i.e., filtering or smoothing) of a cable-driven tensegrity robot. For online state estimation, this work introduces a factor-graph-based method, which fuses measurements from an RGB-D camera with on-board cable length sensors. To the best of the authors' knowledge, this is the first application of factor graphs in this domain. Factor graphs are a natural choice, as they exploit the robot's structural properties and provide effective sensor fusion solutions capable of handling nonlinearities in practice. Both the Mahalanobis distance-based clustering algorithm, used to handle noise, and the Chebyshev polynomial method, used to estimate the most probable velocities and intermediate states, are shown to perform well on simulated and real-world data, compared to an ICP-based algorithm. Results show that the approach provides high fidelity, continuous-time state and trajectory estimates for complex tensegrity robot motions.
>
---
#### [new 019] Learning Without Losing Identity: Capability Evolution for Embodied Agents
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决长期运行中智能体能力进化与身份保持的矛盾。提出ECMs机制，实现能力演进而不改变智能体本质。**

- **链接: [https://arxiv.org/pdf/2604.07799](https://arxiv.org/pdf/2604.07799)**

> **作者:** Xue Qin; Simin Luan; John See; Cong Yang; Zhijun Li
>
> **备注:** 12 pages, 2 figures, 7 tables
>
> **摘要:** Embodied agents are expected to operate persistently in dynamic physical environments, continuously acquiring new capabilities over time. Existing approaches to improving agent performance often rely on modifying the agent itself -- through prompt engineering, policy updates, or structural redesign -- leading to instability and loss of identity in long-lived systems. In this work, we propose a capability-centric evolution paradigm for embodied agents. We argue that a robot should maintain a persistent agent as its cognitive identity, while enabling continuous improvement through the evolution of its capabilities. Specifically, we introduce the concept of Embodied Capability Modules (ECMs), which represent modular, versioned units of embodied functionality that can be learned, refined, and composed over time. We present a unified framework in which capability evolution is decoupled from agent identity. Capabilities evolve through a closed-loop process involving task execution, experience collection, model refinement, and module updating, while all executions are governed by a runtime layer that enforces safety and policy constraints. We demonstrate through simulated embodied tasks that capability evolution improves task success rates from 32.4% to 91.3% over 20 iterations, outperforming both agent-modification baselines and established skill-learning methods (SPiRL, SkiMo), while preserving zero policy drift and zero safety violations. Our results suggest that separating agent identity from capability evolution provides a scalable and safe foundation for long-term embodied intelligence.
>
---
#### [new 020] Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主车辆人机交互任务，解决乘客开放式指令转化为控制信号的问题。通过LLM与多规划器调度框架，提升任务完成率与安全性。**

- **链接: [https://arxiv.org/pdf/2604.08031](https://arxiv.org/pdf/2604.08031)**

> **作者:** Jiawei Liu; Xun Gong; Fen Fang; Muli Yang; Bohao Qu; Yunfeng Hu; Hong Chen; Xulei Yang; Qing Guo
>
> **摘要:** Most Human-Machine Interaction (HMI) research overlooks the maneuvering needs of passengers in autonomous driving (AD). Natural language offers an intuitive interface, yet translating passenger open-ended instructions into control signals, without sacrificing interpretability and traceability, remains a challenge. This study proposes an instruction-realization framework that leverages a large language model (LLM) to interpret instructions, generates executable scripts that schedule multiple model predictive control (MPC)-based motion planners based on real-time feedback, and converts planned trajectories into control signals. This scheduling-centric design decouples semantic reasoning from vehicle control at different timescales, establishing a transparent, traceable decision-making chain from high-level instructions to low-level actions. Due to the absence of high-fidelity evaluation tools, this study introduces a benchmark for open-ended instruction realization in a closed-loop setting. Comprehensive experiments reveal that the framework significantly improves task-completion rates over instruction-realization baselines, reduces LLM query costs, achieves safety and compliance on par with specialized AD approaches, and exhibits considerable tolerance to LLM inference latency. For more qualitative illustrations and a clearer understanding.
>
---
#### [new 021] Active Reward Machine Inference From Raw State Trajectories
- **分类: cs.RO; cs.AI; cs.FL**

- **简介: 该论文属于强化学习任务，旨在从原始状态轨迹中学习奖励机。解决手动定义奖励机的难题，通过轨迹数据直接推断奖励机结构。**

- **链接: [https://arxiv.org/pdf/2604.07480](https://arxiv.org/pdf/2604.07480)**

> **作者:** Mohamad Louai Shehab; Antoine Aspeel; Necmiye Ozay
>
> **摘要:** Reward machines are automaton-like structures that capture the memory required to accomplish a multi-stage task. When combined with reinforcement learning or optimal control methods, they can be used to synthesize robot policies to achieve such tasks. However, specifying a reward machine by hand, including a labeling function capturing high-level features that the decisions are based on, can be a daunting task. This paper deals with the problem of learning reward machines directly from raw state and policy information. As opposed to existing works, we assume no access to observations of rewards, labels, or machine nodes, and show what trajectory data is sufficient for learning the reward machine in this information-scarce regime. We then extend the result to an active learning setting where we incrementally query trajectory extensions to improve data (and indirectly computational) efficiency. Results are demonstrated with several grid world examples.
>
---
#### [new 022] Bird-Inspired Spatial Flapping Wing Mechanism via Coupled Linkages with Single Actuator
- **分类: cs.RO**

- **简介: 该论文属于机械设计任务，旨在解决单驱动下仿生翼运动控制问题。通过耦合连杆机构实现扫动与折叠动作，简化系统结构。**

- **链接: [https://arxiv.org/pdf/2604.07677](https://arxiv.org/pdf/2604.07677)**

> **作者:** Daniel Huczala; Sun-Pill Jung; Frank C. Park
>
> **摘要:** Spatial single-loop mechanisms such as Bennett linkages offer a unique combination of one-degree-of-freedom actuation and nontrivial spatial trajectories, making them attractive for lightweight bio-inspired robotic design. However, although they appear simple and elegant, the geometric task-based synthesis is rather complicated and often avoided in engineering tasks due to the mathematical complexity involved. This paper presents a bird-inspired flapping-wing mechanism built from two coupled spatial four-bars, driven by a single motor. One linkage is actuated to generate the desired spatial sweeping stroke, while the serially coupled linkage remains unactuated and passively switches between extended and folded wing configurations over the stroke cycle. We introduce a simplified kinematic methodology for constructing Bennett linkages from quadrilaterals that contain a desired surface area and further leverage mechanically induced passive state switching. This architecture realizes a coordinated sweep-and-fold wing motion with a single actuation input, reducing weight and control complexity. A 3D-printed prototype is assembled and tested, demonstrating the intended spatial stroke and passive folding behavior.
>
---
#### [new 023] RAGE-XY: RADAR-Aided Longitudinal and Lateral Forces Estimation For Autonomous Race Cars
- **分类: cs.RO**

- **简介: 该论文属于车辆动力学估计任务，旨在提高自动驾驶赛车的横向和纵向力估计精度。通过改进模型和引入雷达校准模块，增强了估计效果。**

- **链接: [https://arxiv.org/pdf/2604.07939](https://arxiv.org/pdf/2604.07939)**

> **作者:** Davide Malvezzi; Nicola Musiu; Eugenio Mascaro; Francesco Iacovacci; Marko Bertogna
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** In this work, we present RAGE-XY, an extended version of RAGE, a real-time estimation framework that simultaneously infers vehicle velocity, tire slip angles, and the forces acting on the vehicle using only standard onboard sensors such as IMUs and RADARs. Compared to the original formulation, the proposed method incorporates an online RADAR calibration module, improving the accuracy of lateral velocity estimation in the presence of sensor misalignment. Furthermore, we extend the underlying vehicle model from a single-track approximation to a tricycle model, enabling the estimation of rear longitudinal tire forces in addition to lateral dynamics. We validate the proposed approach through both high-fidelity simulations and real-world experiments conducted on the EAV-24 autonomous race car, demonstrating improved accuracy and robustness in estimating both lateral and longitudinal vehicle dynamics.
>
---
#### [new 024] Governed Capability Evolution for Embodied Agents: Safe Upgrade, Compatibility Checking, and Runtime Rollback for Embodied Capability Modules
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人系统任务，解决 embodied agents 能力模块升级中的安全与兼容性问题。提出生命周期感知的升级框架，通过多阶段检查确保升级安全可靠。**

- **链接: [https://arxiv.org/pdf/2604.08059](https://arxiv.org/pdf/2604.08059)**

> **作者:** Xue Qin; Simin Luan; John See; Cong Yang; Zhijun Li
>
> **备注:** 46 pages, 3 figures, 10 tables, 7 appendices
>
> **摘要:** Embodied agents are increasingly expected to improve over time by updating their executable capabilities rather than rewriting the agent itself. Prior work has separately studied modular capability packaging, capability evolution, and runtime governance. However, a key systems problem remains underexplored: once an embodied capability module evolves into a new version, how can the hosting system deploy it safely without breaking policy constraints, execution assumptions, or recovery guarantees? We formulate governed capability evolution as a first-class systems problem for embodied agents. We propose a lifecycle-aware upgrade framework in which every new capability version is treated as a governed deployment candidate rather than an immediately executable replacement. The framework introduces four upgrade compatibility checks -- interface, policy, behavioral, and recovery -- and organizes them into a staged runtime pipeline comprising candidate validation, sandbox evaluation, shadow deployment, gated activation, online monitoring, and rollback. We evaluate over 6 rounds of capability upgrade with 15 random seeds. Naive upgrade achieves 72.9% task success but drives unsafe activation to 60% by the final round; governed upgrade retains comparable success (67.4%) while maintaining zero unsafe activations across all rounds (Wilcoxon p=0.003). Shadow deployment reveals 40% of regressions invisible to sandbox evaluation alone, and rollback succeeds in 79.8% of post-activation drift scenarios.
>
---
#### [new 025] Safe Large-Scale Robust Nonlinear MPC in Milliseconds via Reachability-Constrained System Level Synthesis on the GPU
- **分类: cs.RO; cs.AI; eess.SY; math.OC**

- **简介: 该论文属于机器人控制任务，解决高维不确定系统实时安全控制问题。提出GPU-SLS框架，实现快速鲁棒非线性MPC，提升轨迹优化与可达集计算效率。**

- **链接: [https://arxiv.org/pdf/2604.07644](https://arxiv.org/pdf/2604.07644)**

> **作者:** Jeffrey Fang; Glen Chou
>
> **备注:** Under review
>
> **摘要:** We present GPU-SLS, a GPU-parallelized framework for safe, robust nonlinear model predictive control (MPC) that scales to high-dimensional uncertain robotic systems and long planning horizons. Our method jointly optimizes an inequality-constrained, dynamically-feasible nominal trajectory, a tracking controller, and a closed-loop reachable set under disturbance, all in real-time. To efficiently compute nominal trajectories, we develop a sequential quadratic programming procedure with a novel GPU-accelerated quadratic program (QP) solver that uses parallel associative scans and adaptive caching within an alternating direction method of multipliers (ADMM) framework. The same GPU QP backend is used to optimize robust tracking controllers and closed-loop reachable sets via system level synthesis (SLS), enabling reachability-constrained control in both fixed- and receding-horizon settings. We achieve substantial performance gains, reducing nominal trajectory solve times by 97.7% relative to state-of-the-art CPU solvers and 71.8% compared to GPU solvers, while accelerating SLS-based control and reachability by 237x. Despite large problem scales, our method achieves 100% empirical safety, unlike high-dimensional learning-based reachability baselines. We validate our approach on complex nonlinear systems, including whole-body quadrupeds (61D) and humanoids (75D), synthesizing robust control policies online on the GPU in 20 milliseconds on average and scaling to problems with 2 x 10^5 decision variables and 8 x 10^4 constraints. The implementation of our method is available at this https URL.
>
---
#### [new 026] EvoGymCM: Harnessing Continuous Material Stiffness for Soft Robot Co-Design
- **分类: cs.RO**

- **简介: 该论文属于软体机器人设计任务，解决材料刚度离散化限制问题，提出EvoGymCM框架，实现连续材料刚度与形态、控制协同优化。**

- **链接: [https://arxiv.org/pdf/2604.08258](https://arxiv.org/pdf/2604.08258)**

> **作者:** Le Shen; Kangyao Huang; Wentao Zhao; Huaping Liu
>
> **备注:** 8 pages, 11 figures. Preprint. Under review at IROS 2026
>
> **摘要:** In the automated co-design of soft robots, precisely adapting the material stiffness field to task environments is crucial for unlocking their full physical potential. However, mainstream platforms (e.g., EvoGym) strictly discretize the material dimension, artificially restricting the design space and performance of soft robots. To address this, we propose EvoGymCM (EvoGym with Continuous Materials), a benchmark suite formally establishing continuous material stiffness as a first-class design variable alongside morphology and control. Aligning with real-world material mechanisms, EvoGymCM introduces two settings: (i) EvoGymCM-R (Reactive), motivated by programmable materials with dynamically tunable stiffness; and (ii) EvoGymCM-I (Invariant), motivated by traditional materials with invariant stiffness fields. To tackle the resulting high-dimensional coupling, we formulate two Morphology-Material-Control co-design paradigms: (i) Reactive-Material Co-Design, which learns real-time stiffness tuning policies to guide programmable materials; and (ii) Invariant-Material Co-Design, which jointly optimizes morphology and fixed material fields to guide traditional material fabrication. Systematic experiments across diverse tasks demonstrate that continuous material optimization boosts performance and unlocks synergy across morphology, material, and control.
>
---
#### [new 027] Vision-Language Navigation for Aerial Robots: Towards the Era of Large Language Models
- **分类: cs.RO**

- **简介: 该论文属于Aerial VLN任务，旨在让无人机根据自然语言指令自主导航。论文系统梳理了方法分类，分析了技术挑战与评估现状，并提出七个开放问题。**

- **链接: [https://arxiv.org/pdf/2604.07705](https://arxiv.org/pdf/2604.07705)**

> **作者:** Xingyu Xia; Lekai Zhou; Yujie Tang; Xiaozhou Zhu; Hai Zhu; Wen Yao
>
> **备注:** 28 pages, 8 figures
>
> **摘要:** Aerial vision-and-language navigation (Aerial VLN) aims to enable unmanned aerial vehicles (UAVs) to interpret natural language instructions and autonomously navigate complex three-dimensional environments by grounding language in visual perception. This survey provides a critical and analytical review of the Aerial VLN field, with particular attention to the recent integration of large language models (LLMs) and vision-language models (VLMs). We first formally introduce the Aerial VLN problem and define two interaction paradigms: single-instruction and dialog-based, as foundational axes. We then organize the body of Aerial VLN methods into a taxonomy of five architectural categories: sequence-to-sequence and attention-based methods, end-to-end LLM/VLM methods, hierarchical methods, multi-agent methods, and dialog-based navigation methods. For each category, we systematically analyze design rationales, technical trade-offs, and reported performance. We critically assess the evaluation infrastructure for Aerial VLN, including datasets, simulation platforms, and metrics, and identify their gaps in scale, environmental diversity, real-world grounding, and metric coverage. We consolidate cross-method comparisons on shared benchmarks and analyze key architectural trade-offs, including discrete versus continuous actions, end-to-end versus hierarchical designs, and the simulation-to-reality gap. Finally, we synthesize seven concrete open problems: long-horizon instruction grounding, viewpoint robustness, scalable spatial representation, continuous 6-DoF action execution, onboard deployment, benchmark standardization, and multi-UAV swarm navigation, with specific research directions grounded in the evidence presented throughout the survey.
>
---
#### [new 028] A Physical Agentic Loop for Language-Guided Grasping with Execution-State Monitoring
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于语言引导抓取任务，旨在解决机器人执行抓取时失败反馈不明确的问题。通过引入物理代理循环，实现执行状态监控与有限恢复，提升抓取的鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2604.07395](https://arxiv.org/pdf/2604.07395)**

> **作者:** Wenze Wang; Mehdi Hosseinzadeh; Feras Dayoub
>
> **备注:** Project page: this https URL
>
> **摘要:** Robotic manipulation systems that follow language instructions often execute grasp primitives in a largely single-shot manner: a model proposes an action, the robot executes it, and failures such as empty grasps, slips, stalls, timeouts, or semantically wrong grasps are not surfaced to the decision layer in a structured way. Inspired by agentic loops in digital tool-using agents, we reformulate language-guided grasping as a bounded embodied agent operating over grounded execution states, where physical actions expose an explicit tool-state stream. We introduce a physical agentic loop that wraps an unmodified learned manipulation primitive (grasp-and-lift) with (i) an event-based interface and (ii) an execution monitoring layer, Watchdog, which converts noisy gripper telemetry into discrete outcome labels using contact-aware fusion and temporal stabilization. These outcome events, optionally combined with post-grasp semantic verification, are consumed by a deterministic bounded policy that finalizes, retries, or escalates to the user for clarification, guaranteeing finite termination. We validate the resulting loop on a mobile manipulator with an eye-in-hand D405 camera, keeping the underlying grasp model unchanged and evaluating representative scenarios involving visual ambiguity, distractors, and induced execution failures. Results show that explicit execution-state monitoring and bounded recovery enable more robust and interpretable behavior than open-loop execution, while adding minimal architectural overhead. For the source code and demo refer to our project page: this https URL
>
---
#### [new 029] Harnessing Embodied Agents: Runtime Governance for Policy-Constrained Execution
- **分类: cs.RO**

- **简介: 该论文属于智能体系统任务，解决运行时治理问题。提出分离认知与监管的框架，实现安全执行控制，提升系统可靠性。**

- **链接: [https://arxiv.org/pdf/2604.07833](https://arxiv.org/pdf/2604.07833)**

> **作者:** Xue Qin; Simin Luan; John See; Cong Yang; Zhijun Li
>
> **备注:** 36 pages, 3 figures, 10 tables
>
> **摘要:** Embodied agents are evolving from passive reasoning systems into active executors that interact with tools, robots, and physical environments. Once granted execution authority, the central challenge becomes how to keep actions governable at runtime. Existing approaches embed safety and recovery logic inside the agent loop, making execution control difficult to standardize, audit, and adapt. This paper argues that embodied intelligence requires not only stronger agents, but stronger runtime governance. We propose a framework for policy-constrained execution that separates agent cognition from execution oversight. Governance is externalized into a dedicated runtime layer performing policy checking, capability admission, execution monitoring, rollback handling, and human override. We formalize the control boundary among the embodied agent, Embodied Capability Modules (ECMs), and runtime governance layer, and validate through 1000 randomized simulation trials across three governance dimensions. Results show 96.2% interception of unauthorized actions, reduction of unsafe continuation from 100% to 22.2% under runtime drift, and 91.4% recovery success with full policy compliance, substantially outperforming all baselines (p<0.001). By reframing runtime governance as a first-class systems problem, this paper positions policy-constrained execution as a key design principle for embodied agent systems.
>
---
#### [new 030] SANDO: Safe Autonomous Trajectory Planning for Dynamic Unknown Environments
- **分类: cs.RO**

- **简介: 该论文提出SANDO，用于动态未知环境中的安全轨迹规划任务。解决碰撞风险与计算效率的矛盾，通过全局路径规划、优化算法和形式化分析，实现高效且安全的轨迹生成。**

- **链接: [https://arxiv.org/pdf/2604.07599](https://arxiv.org/pdf/2604.07599)**

> **作者:** Kota Kondo; Jesús Tordesillas; Jonathan P. How
>
> **备注:** 20 pages, 17 figures
>
> **摘要:** SANDO is a safe trajectory planner for 3D dynamic unknown environments, where obstacle locations and motions are unknown a priori and a collision-free plan can become unsafe at any moment, requiring fast replanning. Existing soft-constraint planners are fast but cannot guarantee collision-free paths, while hard-constraint methods ensure safety at the cost of longer computation. SANDO addresses this trade-off through three contributions. First, a heat map-based A* global planner steers paths away from high-risk regions using soft costs, and a spatiotemporal safe flight corridor (STSFC) generator produces time-layered polytopes that inflate obstacles only by their worst-case reachable set at each time layer, rather than by the worst case over the entire horizon. Second, trajectory optimization is formulated as a Mixed-Integer Quadratic Program (MIQP) with hard collision-avoidance constraints, and a variable elimination technique reduces the number of decision variables, enabling fast computation. Third, a formal safety analysis establishes collision-free guarantees under explicit velocity-bound and estimation-error assumptions. Ablation studies show that variable elimination yields up to 7.4x speedup in optimization time, and that STSFCs are critical for feasibility in dense dynamic environments. Benchmark simulations against state-of-the-art methods across standardized static benchmarks, obstacle-rich static forests, and dynamic environments show that SANDO consistently achieves the highest success rate with no constraint violations across all difficulty levels; perception-only experiments without ground truth obstacle information confirm robust performance under realistic sensing. Hardware experiments on a UAV with fully onboard planning, perception, and localization demonstrate six safe flights in static environments and ten safe flights among dynamic obstacles.
>
---
#### [new 031] A Soft Robotic Interface for Chick-Robot Affective Interactions
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于动物-机器人交互研究，旨在提升雏鸡对机器人的接受度。通过设计软体情感接口，结合热、视觉和呼吸刺激，评估雏鸡的互动反应，为动物福利和神经科学研究提供安全有效的方案。**

- **链接: [https://arxiv.org/pdf/2604.08443](https://arxiv.org/pdf/2604.08443)**

> **作者:** Jue Chen; Alexander Mielke; Kaspar Althoefer; Elisabetta Versace
>
> **摘要:** The potential of Animal-Robot Interaction (ARI) in welfare applications depends on how much an animal perceives a robotic agent as socially relevant, non-threatening and potentially attractive (acceptance). Here, we present an animal-centered soft robotic affective interface for newly hatched chicks (Gallus gallus). The soft interface provides safe and controllable cues, including warmth, breathing-like rhythmic deformation, and face-like visual stimuli. We evaluated chick acceptance of the interface and chick-robot interactions by measuring spontaneous approach and touch responses during video tracking. Overall, chicks approached and spent increasing time on or near the interface, demonstrating acceptance of the device. Across different layouts, chicks showed strong preference for warm thermal stimulation, which increased over time. Face-like visual cues elicited a swift and stable preference, speeding up the initial approach to the tactile interface. Although the breathing cue did not elicit any preference, neither did it trigger avoidance, paving the way for further exploration. These findings translate affective interface concepts to ARI, demonstrating that appropriate soft, thermal and visual stimuli can sustain early chick-robot interactions. This work establishes a reliable evaluation protocol and a safe baseline for designing multimodal robotic devices for animal welfare and neuroscientific research.
>
---
#### [new 032] Evaluation as Evolution: Transforming Adversarial Diffusion into Closed-Loop Curricula for Autonomous Vehicles
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全评估任务，解决静态数据中稀缺极端事件导致的策略偏差问题。提出$E^2$框架，通过闭环进化课程提升故障发现与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.07378](https://arxiv.org/pdf/2604.07378)**

> **作者:** Yicheng Guo; Jiaqi Liu; Chengkai Xu; Peng Hang; Jian Sun
>
> **摘要:** Autonomous vehicles in interactive traffic environments are often limited by the scarcity of safety-critical tail events in static datasets, which biases learned policies toward average-case behaviors and reduces robustness. Existing evaluation methods attempt to address this through adversarial stress testing, but are predominantly open-loop and post-hoc, making it difficult to incorporate discovered failures back into the training process. We introduce Evaluation as Evolution ($E^2$), a closed-loop framework that transforms adversarial generation from a static validation step into an adaptive evolutionary curriculum. Specifically, $E^2$ formulates adversarial scenario synthesis as transport-regularized sparse control over a learned reverse-time SDE prior. To make this high-dimensional generation tractable, we utilize topology-driven support selection to identify critical interacting agents, and introduce Topological Anchoring to stabilize the process. This approach enables the targeted discovery of failure cases while strictly constraining deviations from realistic data distributions. Empirically, $E^2$ improves collision failure discovery by 9.01% on the nuScenes dataset and up to 21.43% on the nuPlan dataset over the strongest baselines, while maintaining low invalidity and high realism. It further yields substantial robustness gains when the resulting boundary cases are recycled for closed-loop policy fine-tuning.
>
---
#### [new 033] Incremental Residual Reinforcement Learning Toward Real-World Learning for Social Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于社会导航任务，旨在解决机器人在真实环境中学习效率低的问题。提出增量残差强化学习方法，提升适应新环境的能力。**

- **链接: [https://arxiv.org/pdf/2604.07945](https://arxiv.org/pdf/2604.07945)**

> **作者:** Haruto Nagahisa; Kohei Matsumoto; Yuki Tomita; Yuki Hyodo; Ryo Kurazume
>
> **摘要:** As the demand for mobile robots continues to increase, social navigation has emerged as a critical task, driving active research into deep reinforcement learning (RL) approaches. However, because pedestrian dynamics and social conventions vary widely across different regions, simulations cannot easily encompass all possible real-world scenarios. Real-world RL, in which agents learn while operating directly in physical environments, presents a promising solution to this issue. Nevertheless, this approach faces significant challenges, particularly regarding constrained computational resources on edge devices and learning efficiency. In this study, we propose incremental residual RL (IRRL). This method integrates incremental learning, which is a lightweight process that operates without a replay buffer or batch updates, with residual RL, which enhances learning efficiency by training only on the residuals relative to a base policy. Through the simulation experiments, we demonstrated that, despite lacking a replay buffer, IRRL achieved performance comparable to those of conventional replay buffer-based methods and outperformed existing incremental learning approaches. Furthermore, the real-world experiments confirmed that IRRL can enable robots to effectively adapt to previously unseen environments through the real-world learning.
>
---
#### [new 034] EgoVerse: An Egocentric Human Dataset for Robot Learning from Around the World
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出EgoVerse，一个用于机器人学习的人类视角数据集与协作平台，解决数据稀缺与分散问题，通过统一数据收集与共享，促进机器人学习发展。**

- **链接: [https://arxiv.org/pdf/2604.07607](https://arxiv.org/pdf/2604.07607)**

> **作者:** Ryan Punamiya; Simar Kareer; Zeyi Liu; Josh Citron; Ri-Zhao Qiu; Xiongyi Cai; Alexey Gavryushin; Jiaqi Chen; Davide Liconti; Lawrence Y. Zhu; Patcharapong Aphiwetsa; Baoyu Li; Aniketh Cheluva; Pranav Kuppili; Yangcen Liu; Dhruv Patel; Aidan Gao; Hye-Young Chung; Ryan Co; Renee Zbizika; Jeff Liu; Xiaomeng Xu; Haoyu Xiong; Geng Chen; Sebastiano Oliani; Chenyu Yang; Xi Wang; James Fort; Richard Newcombe; Josh Gao; Jason Chong; Garrett Matsuda; Aseem Doriwala; Marc Pollefeys; Robert Katzschmann; Xiaolong Wang; Shuran Song; Judy Hoffman; Danfei Xu
>
> **摘要:** Robot learning increasingly depends on large and diverse data, yet robot data collection remains expensive and difficult to scale. Egocentric human data offer a promising alternative by capturing rich manipulation behavior across everyday environments. However, existing human datasets are often limited in scope, difficult to extend, and fragmented across institutions. We introduce EgoVerse, a collaborative platform for human data-driven robot learning that unifies data collection, processing, and access under a shared framework, enabling contributions from individual researchers, academic labs, and industry partners. The current release includes 1,362 hours (80k episodes) of human demonstrations spanning 1,965 tasks, 240 scenes, and 2,087 unique demonstrators, with standardized formats, manipulation-relevant annotations, and tooling for downstream learning. Beyond the dataset, we conduct a large-scale study of human-to-robot transfer with experiments replicated across multiple labs, tasks, and robot embodiments under shared protocols. We find that policy performance generally improves with increased human data, but that effective scaling depends on alignment between human data and robot learning objectives. Together, the dataset, platform, and study establish a foundation for reproducible progress in human data-driven robot learning. Videos and additional information can be found at this https URL
>
---
#### [new 035] Exploring Temporal Representation in Neural Processes for Multimodal Action Prediction
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人自监督多模态动作预测任务，旨在解决动作序列预测中的时间表示问题。通过改进现有模型，提升其对未见动作的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.08418](https://arxiv.org/pdf/2604.08418)**

> **作者:** Marco Gabriele Fedozzi; Yukie Nagai; Francesco Rea; Alessandra Sciutti
>
> **备注:** Submitted to the AIC 2023 (9th International Workshop on Artificial Intelligence and Cognition)
>
> **摘要:** Inspired by the human ability to understand and predict others, we study the applicability of Conditional Neural Processes (CNP) to the task of self-supervised multimodal action prediction in robotics. Following recent results regarding the ontogeny of the Mirror Neuron System (MNS), we focus on the preliminary objective of self-actions prediction. We find a good MNS-inspired model in the existing Deep Modality Blending Network (DMBN), able to reconstruct the visuo-motor sensory signal during a partially observed action sequence by leveraging the probabilistic generation of CNP. After a qualitative and quantitative evaluation, we highlight its difficulties in generalizing to unseen action sequences, and identify the cause in its inner representation of time. Therefore, we propose a revised version, termed DMBN-Positional Time Encoding (DMBN-PTE), that facilitates learning a more robust representation of temporal information, and provide preliminary results of its effectiveness in expanding the applicability of the architecture. DMBN-PTE figures as a first step in the development of robotic systems that autonomously learn to forecast actions on longer time scales refining their predictions with incoming observations.
>
---
#### [new 036] HEX: Humanoid-Aligned Experts for Cross-Embodiment Whole-Body Manipulation
- **分类: cs.RO**

- **简介: 该论文提出HEX框架，解决人形机器人全身体操控制问题。通过统一状态表示和专家融合机制，提升操控稳定性与任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.07993](https://arxiv.org/pdf/2604.07993)**

> **作者:** Shuanghao Bai; Meng Li; Xinyuan Lv; Jiawei Wang; Xinhua Wang; Fei Liao; Chengkai Hou; Langzhe Gu; Wanqi Zhou; Kun Wu; Ziluo Ding; Zhiyuan Xu; Lei Sun; Shanghang Zhang; Zhengping Che; Jian Tang; Badong Chen
>
> **备注:** Project page: this https URL
>
> **摘要:** Humans achieve complex manipulation through coordinated whole-body control, whereas most Vision-Language-Action (VLA) models treat robot body parts largely independently, making high-DoF humanoid control challenging and often unstable. We present HEX, a state-centric framework for coordinated manipulation on full-sized bipedal humanoid robots. HEX introduces a humanoid-aligned universal state representation for scalable learning across heterogeneous embodiments, and incorporates a Mixture-of-Experts Unified Proprioceptive Predictor to model whole-body coordination and temporal motion dynamics from large-scale multi-embodiment trajectory data. To efficiently capture temporal visual context, HEX uses lightweight history tokens to summarize past observations, avoiding repeated encoding of historical images during inference. It further employs a residual-gated fusion mechanism with a flow-matching action head to adaptively integrate visual-language cues with proprioceptive dynamics for action generation. Experiments on real-world humanoid manipulation tasks show that HEX achieves state-of-the-art performance in task success rate and generalization, particularly in fast-reaction and long-horizon scenarios.
>
---
#### [new 037] Sumo: Dynamic and Generalizable Whole-Body Loco-Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人动态操控任务，解决腿式机器人搬运重物的问题。通过预训练策略与采样规划器结合，实现无需额外训练的通用操控能力。**

- **链接: [https://arxiv.org/pdf/2604.08508](https://arxiv.org/pdf/2604.08508)**

> **作者:** John Z. Zhang; Maks Sorokin; Jan Brüdigam; Brandon Hung; Stephen Phillips; Dmitry Yershov; Farzad Niroui; Tong Zhao; Leonor Fermoselle; Xinghao Zhu; Chao Cao; Duy Ta; Tao Pang; Jiuguang Wang; Preston Culbertson; Zachary Manchester; Simon Le Cléac'h
>
> **摘要:** This paper presents a sim-to-real approach that enables legged robots to dynamically manipulate large and heavy objects with whole-body dexterity. Our key insight is that by performing test-time steering of a pre-trained whole-body control policy with a sample-based planner, we can enable these robots to solve a variety of dynamic loco-manipulation tasks. Interestingly, we find our method generalizes to a diverse set of objects and tasks with no additional tuning or training, and can be further enhanced by flexibly adjusting the cost function at test time. We demonstrate the capabilities of our approach through a variety of challenging loco-manipulation tasks on a Spot quadruped robot in the real world, including uprighting a tire heavier than the robot's nominal lifting capacity and dragging a crowd-control barrier larger and taller than the robot itself. Additionally, we show that the same approach can be generalized to humanoid loco-manipulation tasks, such as opening a door and pushing a table, in simulation. Project code and videos are available at \href{this https URL}{this https URL}.
>
---
#### [new 038] Grasp as You Dream: Imitating Functional Grasping from Generated Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决开放环境中泛化抓取问题。通过生成的人类示范，实现零样本功能抓取，提升数据效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.07517](https://arxiv.org/pdf/2604.07517)**

> **作者:** Chao Tang; Jiacheng Xu; Haofei Lu; Bolin Zou; Wenlong Dong; Hong Zhang; Danica Kragic
>
> **摘要:** Building generalist robots capable of performing functional grasping in everyday, open-world environments remains a significant challenge due to the vast diversity of objects and tasks. Existing methods are either constrained to narrow object/task sets or rely on prohibitively large-scale data collection to capture real-world variability. In this work, we present an alternative approach, GraspDreamer, a method that leverages human demonstrations synthesized by visual generative models (VGMs) (e.g., video generation models) to enable zero-shot functional grasping without labor-intensive data collection. The key idea is that VGMs pre-trained on internet-scale human data implicitly encode generalized priors about how humans interact with the physical world, which can be combined with embodiment-specific action optimization to enable functional grasping with minimal effort. Extensive experiments on the public benchmarks with different robot hands demonstrate the superior data efficiency and generalization performance of GraspDreamer compared to previous methods. Real-world evaluations further validate the effectiveness on real robots. Additionally, we showcase that GraspDreamer can (1) be naturally extended to downstream manipulation tasks, and (2) can generate data to support visuomotor policy learning.
>
---
#### [new 039] Density-Driven Optimal Control: Convergence Guarantees for Stochastic LTI Multi-Agent Systems
- **分类: math.OC; cs.MA; cs.RO; eess.SY**

- **简介: 该论文研究多智能体系统的非均匀区域覆盖问题，提出一种基于随机LTI动态的D²OC方法，通过最小化Wasserstein距离实现分布匹配，并提供收敛性保证。**

- **链接: [https://arxiv.org/pdf/2604.08495](https://arxiv.org/pdf/2604.08495)**

> **作者:** Kooktae Lee
>
> **摘要:** This paper addresses the decentralized non-uniform area coverage problem for multi-agent systems, a critical task in missions with high spatial priority and resource constraints. While existing density-based methods often rely on computationally heavy Eulerian PDE solvers or heuristic planning, we propose Stochastic Density-Driven Optimal Control (D$^2$OC). This is a rigorous Lagrangian framework that bridges the gap between individual agent dynamics and collective distribution matching. By formulating a stochastic MPC-like problem that minimizes the Wasserstein distance as a running cost, our approach ensures that the time-averaged empirical distribution converges to a non-parametric target density under stochastic LTI dynamics. A key contribution is the formal convergence guarantee established via reachability analysis, providing a bounded tracking error even in the presence of process and measurement noise. Numerical results verify that Stochastic D$^2$OC achieves robust, decentralized coverage while outperforming previous heuristic methods in optimality and consistency.
>
---
#### [new 040] CrashSight: A Phase-Aware, Infrastructure-Centric Video Benchmark for Traffic Crash Scene Understanding and Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出CrashSight，一个用于交通碰撞场景理解与推理的视觉语言基准。针对现有基准侧重车辆视角、缺乏基础设施视角的问题，该工作利用路侧摄像头数据，构建包含250个碰撞视频的多层级问答数据集，评估模型在时间与因果推理上的能力。**

- **链接: [https://arxiv.org/pdf/2604.08457](https://arxiv.org/pdf/2604.08457)**

> **作者:** Rui Gan; Junyi Ma; Pei Li; Xingyou Yang; Kai Chen; Sikai Chen; Bin Ran
>
> **摘要:** Cooperative autonomous driving requires traffic scene understanding from both vehicle and infrastructure perspectives. While vision-language models (VLMs) show strong general reasoning capabilities, their performance in safety-critical traffic scenarios remains insufficiently evaluated due to the ego-vehicle focus of existing benchmarks. To bridge this gap, we present \textbf{CrashSight}, a large-scale vision-language benchmark for roadway crash understanding using real-world roadside camera data. The dataset comprises 250 crash videos, annotated with 13K multiple-choice question-answer pairs organized under a two-tier taxonomy. Tier 1 evaluates the visual grounding of scene context and involved parties, while Tier 2 probes higher-level reasoning, including crash mechanics, causal attribution, temporal progression, and post-crash outcomes. We benchmark 8 state-of-the-art VLMs and show that, despite strong scene description capabilities, current models struggle with temporal and causal reasoning in safety-critical scenarios. We provide a detailed analysis of failure scenarios and discuss directions for improving VLM crash understanding. The benchmark provides a standardized evaluation framework for infrastructure-assisted perception in cooperative autonomous driving. The CrashSight benchmark, including the full dataset and code, is accessible at this https URL.
>
---
#### [new 041] Complementary Filtering on SO(3) for Attitude Estimation with Scalar Measurements
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于姿态估计任务，解决基于标量测量的姿态估计问题。通过改进互补滤波器，实现稳定姿态估计，适用于传感受限场景。**

- **链接: [https://arxiv.org/pdf/2604.08099](https://arxiv.org/pdf/2604.08099)**

> **作者:** Alessandro Melis; Soulaimane Berkane; Tarek Hamel
>
> **备注:** Submitted to CDC 2026
>
> **摘要:** Attitude estimation using scalar measurements, corresponding to partial vectorial observations, arises naturally when inertial vectors are not fully observed but only measured along specific body-frame vectors. Such measurements arise in problems involving incomplete vector measurements or attitude constraints derived from heterogeneous sensor information. Building on the classical complementary filter on SO(3), we propose an observer with a modified innovation term tailored to this scalar-output structure. The main result shows that almost-global asymptotic stability is recovered, under suitable persistence of excitation conditions, when at least three inertial vectors are measured along a common body-frame vector, which is consistent with the three-dimensional structure of SO(3). For two-scalar configurations - corresponding either to one inertial vector measured along two body-frame vectors, or to two inertial vectors measured along a common body-frame vector - we further derive sufficient conditions guaranteeing convergence within a reduced basin of attraction. Different examples and numerical results demonstrate the effectiveness of the proposed scalar-based complementary filter for attitude estimation in challenging scenarios involving reduced sensing and/or novel sensing modalities.
>
---
#### [new 042] "Why This Avoidance Maneuver?" Contrastive Explanations in Human-Supervised Maritime Autonomous Navigation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于人机协同导航任务，旨在解决如何解释自主船舶避碰决策的问题。提出对比解释方法，通过比较方案提升理解，但需注意认知负荷。**

- **链接: [https://arxiv.org/pdf/2604.08032](https://arxiv.org/pdf/2604.08032)**

> **作者:** Joel Jose; Andreas Madsen; Andreas Brandsæter; Tor A. Johansen; Erlend M. Coates
>
> **备注:** Submitted to IEEE Intelligent Transportation Systems Conference (ITSC) 2026
>
> **摘要:** Automated maritime collision avoidance will rely on human supervision for the foreseeable future. This necessitates transparency into how the system perceives a scenario and plans a maneuver. However, the causal logic behind avoidance maneuvers is often complex and difficult to convey to a navigator. This paper explores how to explain these factors in a selective, understandable manner for supervisors with a nautical background. We propose a method for generating contrastive explanations, which provide human-centric insights by comparing a system's proposed solution against relevant alternatives. To evaluate this, we developed a framework that uses visual and textual cues to highlight key objectives from a state-of-the-art collision avoidance system. An exploratory user study with four experienced marine officers suggests that contrastive explanations support the understanding of the system's objectives. However, our findings also reveal that while these explanations are highly valuable in complex multi-vessel encounters, they can increase cognitive workload, suggesting that future maritime interfaces may benefit most from demand-driven or scenario-specific explanation strategies.
>
---
#### [new 043] PriPG-RL: Privileged Planner-Guided Reinforcement Learning for Partially Observable Systems with Anytime-Feasible MPC
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，解决部分可观测系统中的策略训练问题。通过引入特权规划器，提升样本效率和政策性能。**

- **链接: [https://arxiv.org/pdf/2604.08036](https://arxiv.org/pdf/2604.08036)**

> **作者:** Mohsen Amiri; Mohsen Amiri; Ali Beikmohammadi; Sindri Magnuśson; Mehdi Hosseinzadeh
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** This paper addresses the problem of training a reinforcement learning (RL) policy under partial observability by exploiting a privileged, anytime-feasible planner agent available exclusively during training. We formalize this as a Partially Observable Markov Decision Process (POMDP) in which a planner agent with access to an approximate dynamical model and privileged state information guides a learning agent that observes only a lossy projection of the true state. To realize this framework, we introduce an anytime-feasible Model Predictive Control (MPC) algorithm that serves as the planner agent. For the learning agent, we propose Planner-to-Policy Soft Actor-Critic (P2P-SAC), a method that distills the planner agent's privileged knowledge to mitigate partial observability and thereby improve both sample efficiency and final policy performance. We support this framework with rigorous theoretical analysis. Finally, we validate our approach in simulation using NVIDIA Isaac Lab and successfully deploy it on a real-world Unitree Go2 quadruped navigating complex, obstacle-rich environments.
>
---
#### [new 044] GEAR: GEometry-motion Alternating Refinement for Articulated Object Modeling with Gaussian Splatting
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出GEAR框架，解决关节物体建模中几何与运动联合优化不稳定及泛化能力弱的问题，通过交替优化提升重建精度与一致性。**

- **链接: [https://arxiv.org/pdf/2604.07728](https://arxiv.org/pdf/2604.07728)**

> **作者:** Jialin Li; Bin Fu; Ruiping Wang; Xilin Chen
>
> **备注:** Accepted to CVPRF2026
>
> **摘要:** High-fidelity interactive digital assets are essential for embodied intelligence and robotic interaction, yet articulated objects remain challenging to reconstruct due to their complex structures and coupled geometry-motion relationships. Existing methods suffer from instability in geometry-motion joint optimization, while their generalization remains limited on complex multi-joint or out-of-distribution objects. To address these challenges, we propose GEAR, an EM-style alternating optimization framework that jointly models geometry and motion as interdependent components within a Gaussian Splatting representation. GEAR treats part segmentation as a latent variable and joint motion parameters as explicit variables, alternately refining them for improved convergence and geometric-motion consistency. To enhance part segmentation quality without sacrificing generalization, we leverage a vanilla 2D segmentation model to provide multi-view part priors, and employ a weakly supervised constraint to regularize the latent variable. Experiments on multiple benchmarks and our newly constructed dataset GEAR-Multi demonstrate that GEAR achieves state-of-the-art results in geometric reconstruction and motion parameters estimation, particularly on complex articulated objects with multiple movable parts.
>
---
#### [new 045] ParkSense: Where Should a Delivery Driver Park? Leveraging Idle AV Compute and Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 论文提出ParkSense系统，解决配送司机精准停车问题。利用闲置自动驾驶计算资源，运行视觉语言模型识别入口和合法停车区，提升配送效率。属于自动驾驶与物流优化任务。**

- **链接: [https://arxiv.org/pdf/2604.07912](https://arxiv.org/pdf/2604.07912)**

> **作者:** Die Hu; Henan Li
>
> **备注:** 7 pages, 3 tables. No university resources were used for this work
>
> **摘要:** Finding parking consumes a disproportionate share of food delivery time, yet no system addresses precise parking-spot selection relative to merchant entrances. We propose ParkSense, a framework that repurposes idle compute during low-risk AV states -- queuing at red lights, traffic congestion, parking-lot crawl -- to run a Vision-Language Model (VLM) on pre-cached satellite and street view imagery, identifying entrances and legal parking zones. We formalize the Delivery-Aware Precision Parking (DAPP) problem, show that a quantized 7B VLM completes inference in 4-8 seconds on HW4-class hardware, and estimate annual per-driver income gains of 3,000-8,000 USD in the U.S. Five open research directions are identified at this unexplored intersection of autonomous driving, computer vision, and last-mile logistics.
>
---
#### [new 046] Karma Mechanisms for Decentralised, Cooperative Multi Agent Path Finding
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决分布式协作路径规划中的冲突问题。提出基于Karma机制的框架，通过协商实现公平且高效的冲突解决。**

- **链接: [https://arxiv.org/pdf/2604.07970](https://arxiv.org/pdf/2604.07970)**

> **作者:** Kevin Riehl; Julius Schlapbach; Anastasios Kouvelas; Michail A. Makridis
>
> **摘要:** Multi-Agent Path Finding (MAPF) is a fundamental coordination problem in large-scale robotic and cyber-physical systems, where multiple agents must compute conflict-free trajectories with limited computational and communication resources. While centralised optimal solvers provide guarantees on solution optimality, their exponential computational complexity limits scalability to large-scale systems and real-time applicability. Existing decentralised heuristics are faster, but result in suboptimal outcomes and high cost disparities. This paper proposes a decentralised coordination framework for cooperative MAPF based on Karma mechanisms - artificial, non-tradeable credits that account for agents' past cooperative behaviour and regulate future conflict resolution decisions. The approach formulates conflict resolution as a bilateral negotiation process that enables agents to resolve conflicts through pairwise replanning while promoting long-term fairness under limited communication and without global priority structures. The mechanism is evaluated in a lifelong robotic warehouse multi-agent pickup-and-delivery scenario with kinematic orientation constraints. The results highlight that the Karma mechanism balances replanning effort across agents, reducing disparity in service times without sacrificing overall efficiency. Code: this https URL
>
---
#### [new 047] Event-Centric World Modeling with Memory-Augmented Retrieval for Embodied Decision-Making
- **分类: cs.LG; cs.IR; cs.RO**

- **简介: 该论文属于机器人决策任务，旨在解决动态环境中自主代理的可解释性与物理一致性问题。提出一种基于事件的环境建模方法，通过记忆检索实现可解释的决策。**

- **链接: [https://arxiv.org/pdf/2604.07392](https://arxiv.org/pdf/2604.07392)**

> **作者:** Fan Zhaowen
>
> **备注:** This is the initial version (v1) released to establish priority for the proposed framework. Subsequent versions will include expanded experimental validation and exhaustive hardware benchmarking
>
> **摘要:** Autonomous agents operating in dynamic and safety-critical environments require decision-making frameworks that are both computationally efficient and physically grounded. However, many existing approaches rely on end-to-end learning, which often lacks interpretability and explicit mechanisms for ensuring consistency with physical constraints. In this work, we propose an event-centric world modeling framework with memory-augmented retrieval for embodied decision-making. The framework represents the environment as a structured set of semantic events, which are encoded into a permutation-invariant latent representation. Decision-making is performed via retrieval over a knowledge bank of prior experiences, where each entry associates an event representation with a corresponding maneuver. The final action is computed as a weighted combination of retrieved solutions, providing a transparent link between decision and stored experiences. The proposed design enables structured abstraction of dynamic environments and supports interpretable decision-making through case-based reasoning. In addition, incorporating physics-informed knowledge into the retrieval process encourages the selection of maneuvers that are consistent with observed system dynamics. Experimental evaluation in UAV flight scenarios demonstrates that the framework operates within real-time control constraints while maintaining interpretable and consistent behavior.
>
---
#### [new 048] Visually-grounded Humanoid Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人形智能体任务，旨在解决数字人类在新场景中基于视觉和目标自主行为的问题。工作包括构建双层架构，实现自主感知与决策。**

- **链接: [https://arxiv.org/pdf/2604.08509](https://arxiv.org/pdf/2604.08509)**

> **作者:** Hang Ye; Xiaoxuan Ma; Fan Lu; Wayne Wu; Kwan-Yee Lin; Yizhou Wang
>
> **备注:** Project page: this https URL
>
> **摘要:** Digital human generation has been studied for decades and supports a wide range of real-world applications. However, most existing systems are passively animated, relying on privileged state or scripted control, which limits scalability to novel environments. We instead ask: how can digital humans actively behave using only visual observations and specified goals in novel scenes? Achieving this would enable populating any 3D environments with digital humans at scale that exhibit spontaneous, natural, goal-directed behaviors. To this end, we introduce Visually-grounded Humanoid Agents, a coupled two-layer (world-agent) paradigm that replicates humans at multiple levels: they look, perceive, reason, and behave like real people in real-world 3D scenes. The World Layer reconstructs semantically rich 3D Gaussian scenes from real-world videos via an occlusion-aware pipeline and accommodates animatable Gaussian-based human avatars. The Agent Layer transforms these avatars into autonomous humanoid agents, equipping them with first-person RGB-D perception and enabling them to perform accurate, embodied planning with spatial awareness and iterative reasoning, which is then executed at the low level as full-body actions to drive their behaviors in the scene. We further introduce a benchmark to evaluate humanoid-scene interaction in diverse reconstructed environments. Experiments show our agents achieve robust autonomous behavior, yielding higher task success rates and fewer collisions than ablations and state-of-the-art planning methods. This work enables active digital human population and advances human-centric embodied AI. Data, code, and models will be open-sourced.
>
---
#### [new 049] Formally Guaranteed Control Adaptation for ODD-Resilient Autonomous Systems
- **分类: cs.LO; cs.RO; cs.SE; eess.SY**

- **简介: 该论文属于自主系统可靠性任务，解决ODD外场景下的系统适应问题。通过动态扩展系统模型，提升系统在未知情况下的可靠性和形式化保障。**

- **链接: [https://arxiv.org/pdf/2604.07414](https://arxiv.org/pdf/2604.07414)**

> **作者:** Gricel Vázquez; Calum Imrie; Sepeedeh Shahbeigi; Nawshin Mannan Proma; Tian Gan; Victoria J Hodge; John Molloy; Simos Gerasimou
>
> **摘要:** Ensuring reliable performance in situations outside the Operational Design Domain (ODD) remains a primary challenge in devising resilient autonomous systems. We explore this challenge by introducing an approach for adapting probabilistic system models to handle out-of-ODD scenarios while, in parallel, providing quantitative guarantees. Our approach dynamically extends the coverage of existing system situation capabilities, supporting the verification and adaptation of the system's behaviour under unanticipated situations. Preliminary results demonstrate that our approach effectively increases system reliability by adapting its behaviour and providing formal guarantees even under unforeseen out-of-ODD situations.
>
---
#### [new 050] WorldMAP: Bootstrapping Vision-Language Navigation Trajectory Prediction with Generative World Models
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文提出WorldMAP，解决视觉-语言导航轨迹预测问题。通过生成世界模型提供结构化监督，提升轨迹预测准确性。**

- **链接: [https://arxiv.org/pdf/2604.07957](https://arxiv.org/pdf/2604.07957)**

> **作者:** Hongjin Chen; Shangyun Jiang; Tonghua Su; Chen Gao; Xinlei Chen; Yong Li; Zhibo Chen
>
> **摘要:** Vision-language models (VLMs) and generative world models are opening new opportunities for embodied navigation. VLMs are increasingly used as direct planners or trajectory predictors, while world models support look-ahead reasoning by imagining future views. Yet predicting a reliable trajectory from a single egocentric observation remains challenging. Current VLMs often generate unstable trajectories, and world models, though able to synthesize plausible futures, do not directly provide the grounded signals needed for navigation learning. This raises a central question: how can generated futures be turned into supervision for grounded trajectory prediction? We present WorldMAP, a teacher--student framework that converts world-model-generated futures into persistent semantic-spatial structure and planning-derived supervision. Its world-model-driven teacher builds semantic-spatial memory from generated videos, grounds task-relevant targets and obstacles, and produces trajectory pseudo-labels through explicit planning. A lightweight student with a multi-hypothesis trajectory head is then trained to predict navigation trajectories directly from vision-language inputs. On Target-Bench, WorldMAP achieves the best ADE and FDE among compared methods, reducing ADE by 18.0% and FDE by 42.1% relative to the best competing baseline, while lifting a small open-source VLM to DTW performance competitive with proprietary models. More broadly, the results suggest that, in embodied navigation, the value of world models may lie less in supplying action-ready imagined evidence than in synthesizing structured supervision for navigation learning.
>
---
#### [new 051] BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BLaDA框架，解决3D场景中语言到精细操作的映射问题，通过语义解析、空间定位和姿态生成实现功能抓取。**

- **链接: [https://arxiv.org/pdf/2604.08410](https://arxiv.org/pdf/2604.08410)**

> **作者:** Fan Yang; Wenrui Chen; Guorun Yan; Ruize Liao; Wanjun Jia; Dongsheng Luo; Kailun Yang; Zhiyong Li; Yaonan Wang
>
> **备注:** Code will be publicly available at this https URL
>
> **摘要:** In unstructured environments, functional dexterous grasping calls for the tight integration of semantic understanding, precise 3D functional localization, and physically interpretable execution. Modular hierarchical methods are more controllable and interpretable than end-to-end VLA approaches, but existing ones still rely on predefined affordance labels and lack the tight semantic--pose coupling needed for functional dexterous manipulation. To address this, we propose BLaDA (Bridging Language to Dexterous Actions in 3DGS fields), an interpretable zero-shot framework that grounds open-vocabulary instructions as perceptual and control constraints for functional dexterous manipulation. BLaDA establishes an interpretable reasoning chain by first parsing natural language into a structured sextuple of manipulation constraints via a Knowledge-guided Language Parsing (KLP) module. To achieve pose-consistent spatial reasoning, we introduce the Triangular Functional Point Localization (TriLocation) module, which utilizes 3D Gaussian Splatting as a continuous scene representation and identifies functional regions under triangular geometric constraints. Finally, the 3D Keypoint Grasp Matrix Transformation Execution (KGT3D+) module decodes these semantic-geometric constraints into physically plausible wrist poses and finger-level commands. Extensive experiments on complex benchmarks demonstrate that BLaDA significantly outperforms existing methods in both affordance grounding precision and the success rate of functional manipulation across diverse categories and tasks. Code will be publicly available at this https URL.
>
---
## 更新

#### [replaced 001] AnyImageNav: Any-View Geometry for Precise Last-Meter Image-Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于图像目标导航任务，解决精确定位问题。通过几何匹配实现高精度位姿恢复，提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2604.05351](https://arxiv.org/pdf/2604.05351)**

> **作者:** Yijie Deng; Shuaihang Yuan; Yi Fang
>
> **摘要:** Image Goal Navigation (ImageNav) is evaluated by a coarse success criterion, the agent must stop within 1m of the target, which is sufficient for finding objects but falls short for downstream tasks such as grasping that require precise positioning. We introduce AnyImageNav, a training-free system that pushes ImageNav toward this more demanding setting. Our key insight is that the goal image can be treated as a geometric query: any photo of an object, a hallway, or a room corner can be registered to the agent's observations via dense pixel-level correspondences, enabling recovery of the exact 6-DoF camera pose. Our method realizes this through a semantic-to-geometric cascade: a semantic relevance signal guides exploration and acts as a proximity gate, invoking a 3D multi-view foundation model only when the current view is highly relevant to the goal image; the model then self-certifies its registration in a loop for an accurate recovered pose. Our method sets state-of-the-art navigation success rates on Gibson (93.1%) and HM3D (82.6%), and achieves pose recovery that prior methods do not provide: a position error of 0.27m and heading error of 3.41 degrees on Gibson, and 0.21m / 1.23 degrees on HM3D, a 5-10x improvement over adapted this http URL project page: this https URL
>
---
#### [replaced 002] DHFP-PE: Dual-Precision Hybrid Floating Point Processing Element for AI Acceleration
- **分类: cs.AR; cs.RO; eess.AS; eess.IV**

- **简介: 该论文属于AI加速任务，旨在解决低功耗高吞吐量浮点乘加单元的设计问题。提出一种双精度浮点处理单元，支持多种低精度格式，提升硬件利用率并降低功耗。**

- **链接: [https://arxiv.org/pdf/2604.04507](https://arxiv.org/pdf/2604.04507)**

> **作者:** Shubham Kumar; Vijay Pratap Sharma; Vaibhav Neema; Santosh Kumar Vishvakarma
>
> **备注:** Accepted in ANRF-sponsored 2nd International Conference on Next Generation Electronics (NEleX-2026)
>
> **摘要:** The rapid adoption of low-precision arithmetic in artificial intelligence and edge computing has created a strong demand for energy-efficient and flexible floating-point multiply-accumulate (MAC) units. This paper presents a dual-precision floating-point MAC processing element supporting FP8 (E4M3, E5M2) and FP4 (2 x E2M1, 2 x E1M2) formats, specifically optimized for low-power and high-throughput AI workloads. The proposed architecture employs a novel bit-partitioning technique that enables a single 4-bit unit multiplier to operate either as a standard 4 x 4 multiplier for FP8 or as two parallel 2 x 2 multipliers for 2-bit operands, achieving maximum hardware utilization without duplicating logic. Implemented in 28 nm technology, the proposed PE achieves an operating frequency of 1.94 GHz with an area of 0.00396 mm^2 and power consumption of 2.13 mW, resulting in up to 60.4% area reduction and 86.6% power savings compared to state-of-the-art designs, making it well suited for energy-constrained AI inference and mixed-precision computing applications when deployed within larger accelerator architectures.
>
---
#### [replaced 003] Iteratively Learning Muscle Memory for Legged Robots to Master Adaptive and High Precision Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决腿式机器人在复杂环境中的精确轨迹跟踪问题。通过结合迭代学习控制与生物启发的扭矩库，提升运动精度与适应性。**

- **链接: [https://arxiv.org/pdf/2507.13662](https://arxiv.org/pdf/2507.13662)**

> **作者:** Jing Cheng; Yasser G. Alqaham; Zhenyu Gan; Amit K. Sanyal
>
> **摘要:** This paper presents a scalable and adaptive control framework for legged robots that integrates Iterative Learning Control (ILC) with a biologically inspired torque library (TL), analogous to muscle memory. The proposed method addresses key challenges in robotic locomotion, including accurate trajectory tracking under unmodeled dynamics and external disturbances. By leveraging the repetitive nature of periodic gaits and extending ILC to nonperiodic tasks, the framework enhances accuracy and generalization across diverse locomotion scenarios. The control architecture is data-enabled, combining a physics-based model derived from hybrid-system trajectory optimization with real-time learning to compensate for model uncertainties and external disturbances. A central contribution is the development of a generalized TL that stores learned control profiles and enables rapid adaptation to changes in speed, terrain, and gravitational conditions-eliminating the need for repeated learning and significantly reducing online computation. The approach is validated on the bipedal robot Cassie and the quadrupedal robot A1 through extensive simulations and hardware experiments. Results demonstrate that the proposed framework reduces joint tracking errors by up to 85% within a few seconds and enables reliable execution of both periodic and nonperiodic gaits, including slope traversal and terrain adaptation. Compared to state-of-the-art whole-body controllers, the learned skills eliminate the need for online computation during execution and achieve control update rates exceeding 30x those of existing methods. These findings highlight the effectiveness of integrating ILC with torque memory as a highly data-efficient and practical solution for legged locomotion in unstructured and dynamic environments.
>
---
#### [replaced 004] HOTFLoc++: End-to-End Hierarchical LiDAR Place Recognition, Re-Ranking, and 6-DoF Metric Localisation in Forests
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出HOTFLoc++，用于森林中的LiDAR场景识别、重排序和6-DoF定位。解决复杂环境下的定位与识别问题，通过多尺度几何验证和联合训练提升性能。**

- **链接: [https://arxiv.org/pdf/2511.09170](https://arxiv.org/pdf/2511.09170)**

> **作者:** Ethan Griffiths; Maryam Haghighat; Simon Denman; Clinton Fookes; Milad Ramezani
>
> **备注:** 8 pages, 2 figures, Accepted for publication in IEEE RA-L (2026)
>
> **摘要:** This article presents HOTFLoc++, an end-to-end hierarchical framework for LiDAR place recognition, re-ranking, and 6-DoF metric localisation in forests. Leveraging an octree-based transformer, our approach extracts features at multiple granularities to increase robustness to clutter, self-similarity, and viewpoint changes in challenging scenarios, including ground-to-ground and ground-to-aerial in forest and urban environments. We propose learnable multi-scale geometric verification to reduce re-ranking failures due to degraded single-scale correspondences. Our joint training protocol enforces multi-scale geometric consistency of the octree hierarchy via joint optimisation of place recognition with re-ranking and localisation, improving place recognition convergence. Our system achieves comparable or lower localisation errors to baselines, with runtime improvements of almost two orders of magnitude over RANSAC-based registration for dense point clouds. Experimental results on public datasets show the superiority of our approach compared to state-of-the-art methods, achieving an average Recall@1 of 90.7% on CS-Wild-Places: an improvement of 29.6 percentage points over baselines, while maintaining high performance on single-source benchmarks with an average Recall@1 of 91.7% and 97.9% on Wild-Places and MulRan, respectively. Our method achieves under 2m and 5$^{\circ}$ error for 97.2% of 6-DoF registration attempts, with our multi-scale re-ranking module reducing localisation errors by ~2x on average. The code is available at this https URL.
>
---
#### [replaced 005] LiloDriver: A Lifelong Learning Framework for Closed-loop Motion Planning in Long-tail Autonomous Driving Scenarios
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决长尾场景下运动规划的适应性问题。提出LiloDriver框架，结合大语言模型与记忆系统，实现持续学习与闭环规划。**

- **链接: [https://arxiv.org/pdf/2505.17209](https://arxiv.org/pdf/2505.17209)**

> **作者:** Huaiyuan Yao; Pengfei Li; Bu Jin; Yupeng Zheng; An Liu; Lisen Mu; Qing Su; Qian Zhang; Yilun Chen; Peng Li
>
> **备注:** 7 pages, 3 figures
>
> **摘要:** Recent advances in autonomous driving research towards motion planners that are robust, safe, and adaptive. However, existing rule-based and data-driven planners lack adaptability to long-tail scenarios, while knowledge-driven methods offer strong reasoning but face challenges in representation, control, and real-world evaluation. To address these challenges, we present LiloDriver, a lifelong learning framework for closed-loop motion planning in long-tail autonomous driving scenarios. By integrating large language models (LLMs) with a memory-augmented planner generation system, LiloDriver continuously adapts to new scenarios without retraining. It features a four-stage architecture including perception, scene encoding, memory-based strategy refinement, and LLM-guided reasoning. Evaluated on the nuPlan benchmark, LiloDriver achieves superior performance in both common and rare driving scenarios, outperforming static rule-based and learning-based planners. Our results highlight the effectiveness of combining structured memory and LLM reasoning to enable scalable, human-like motion planning in real-world autonomous driving. Our code is available at this https URL.
>
---
#### [replaced 006] HiF-VLA: Hindsight, Insight and Foresight through Motion Representation for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出HiF-VLA模型，解决视觉-语言-动作模型在长序列任务中的时间感知问题，通过运动表示实现双向时间推理，提升机器人操作的连贯性与效果。**

- **链接: [https://arxiv.org/pdf/2512.09928](https://arxiv.org/pdf/2512.09928)**

> **作者:** Minghui Lin; Pengxiang Ding; Shu Wang; Zifeng Zhuang; Yang Liu; Xinyang Tong; Wenxuan Song; Shangke Lyu; Siteng Huang; Donglin Wang
>
> **备注:** CVPR 2026, Project page: this https URL, Github: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have recently enabled robotic manipulation by grounding visual and linguistic cues into actions. However, most VLAs assume the Markov property, relying only on the current observation and thus suffering from temporal myopia that degrades long-horizon coherence. In this work, we view motion as a more compact and informative representation of temporal context and world dynamics, capturing inter-state changes while filtering static pixel-level noise. From this perspective, HiF-VLA equips a motion-centric world model for the VLA, enabling agents to reason about temporal dynamics for future evolution during action generation. Building on this idea, we propose HiF-VLA (Hindsight, Insight, and Foresight for VLAs), a unified framework that leverages motion for bidirectional temporal reasoning. HiF-VLA encodes past dynamics through hindsight priors, anticipates future motion via foresight reasoning, and integrates both through a hindsight-modulated joint expert to enable a ''think-while-acting'' paradigm for long-horizon manipulation. As a result, HiF-VLA surpasses strong baselines on LIBERO-Long and CALVIN ABC-D benchmarks, while incurring negligible additional inference latency. Furthermore, HiF-VLA achieves substantial improvements in real-world long-horizon manipulation tasks, demonstrating its broad effectiveness in practical robotic settings.
>
---
#### [replaced 007] Informed Hybrid Zonotope-based Motion Planning Algorithm
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决非凸自由空间中的路径规划问题。提出HZ-MP算法，通过分解空间和低维采样提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2507.09309](https://arxiv.org/pdf/2507.09309)**

> **作者:** Peng Xie; Johannes Betz; Amr Alanwar
>
> **摘要:** Optimal path planning in nonconvex free spaces poses substantial computational challenges. A common approach formulates such problems as mixed-integer linear programs (MILPs); however, solving general MILPs is computationally intractable and severely limits scalability. To address these limitations, we propose HZ-MP, an informed Hybrid Zonotope-based Motion Planner, which decomposes the obstacle-free space and performs low-dimensional face sampling guided by an ellipsotope heuristic, thereby concentrating exploration on promising transition regions. This structured exploration mitigates the excessive wasted sampling that degrades existing informed planners in narrow-passage or enclosed-goal scenarios. We prove that HZ-MP is probabilistically complete and asymptotically optimal, and demonstrate empirically that it converges to high-quality trajectories within a small number of iterations.
>
---
#### [replaced 008] Part$^{2}$GS: Part-aware Modeling of Articulated Objects using 3D Gaussian Splatting
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出Part$^{2}$GS，用于建模可动物体的高保真数字孪生，解决结构与运动一致性问题，通过部分感知的3D高斯表示和物理约束实现稳定运动。**

- **链接: [https://arxiv.org/pdf/2506.17212](https://arxiv.org/pdf/2506.17212)**

> **作者:** Tianjiao Yu; Vedant Shah; Muntasir Wahed; Ying Shen; Kiet A. Nguyen; Ismini Lourentzou
>
> **摘要:** Articulated objects are common in the real world, yet modeling their structure and motion remains a challenging task for 3D reconstruction methods. In this work, we introduce Part$^{2}$GS, a novel framework for modeling articulated digital twins of multi-part objects with high-fidelity geometry and physically consistent articulation. Part$^{2}$GS leverages a part-aware 3D Gaussian representation that encodes articulated components with learnable attributes, enabling structured, disentangled transformations that preserve high-fidelity geometry. To ensure physically consistent motion, we propose a motion-aware canonical representation guided by physics-based constraints, including contact enforcement, velocity consistency, and vector-field alignment. Furthermore, we introduce a field of repel points to prevent part collisions and maintain stable articulation paths, significantly improving motion coherence over baselines. Extensive evaluations on both synthetic and real-world datasets show that Part$^{2}$GS consistently outperforms state-of-the-art methods by up to 10$\times$ in Chamfer Distance for movable parts.
>
---
#### [replaced 009] Multi-agent Reach-avoid MDP via Potential Games and Low-rank Policy Structure
- **分类: eess.SY; cs.GT; cs.MA; cs.RO**

- **简介: 该论文研究多智能体避障MDP问题，通过局部反馈策略和潜在博弈结构，降低通信与计算复杂度，实现高效求解。**

- **链接: [https://arxiv.org/pdf/2410.17690](https://arxiv.org/pdf/2410.17690)**

> **作者:** Adam Casselman; Abraham P. Vinod; Sarah H.Q. Li
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** We optimize finite horizon multi-agent reach-avoid Markov decision process (MDP) via \emph{local feedback policies}. The global feedback policy solution yields global optimality but its communication complexity, memory usage and computation complexity scale exponentially with the number of agents. We mitigate this exponential dependency by restricting the solution space to local feedback policies and show that local feedback policies are rank-one factorizations of global feedback policies, which provides a principled approach to reducing communication complexity and memory usage. Additionally, by demonstrating that multi-agent reach-avoid MDPs over local feedback policies has a potential game structure, we show that iterative best response is a tractable multi-agent learning scheme with guaranteed convergence to deterministic Nash equilibrium, and derive each agent's best response via multiplicative dynamic program (DP) over the joint state space. Numerical simulations across different MDPs and agent sets show that the peak memory usage and offline computation complexity are significantly reduced while the approximation error to the optimal global reach-avoid objective is maintained.
>
---
#### [replaced 010] NaviSplit: Dynamic Multi-Branch Split DNNs for Efficient Distributed Autonomous Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出NaviSplit，解决轻量级无人机自主导航问题。通过动态多分支DNN分割，降低计算和数据传输负担，提升效率与准确性。**

- **链接: [https://arxiv.org/pdf/2406.13086](https://arxiv.org/pdf/2406.13086)**

> **作者:** Timothy K Johnsen; Ian Harshbarger; Zixia Xia; Marco Levorato
>
> **备注:** 6 pages, 3 figures
>
> **摘要:** Lightweight autonomous unmanned aerial vehicles (UAV) are emerging as a central component of a broad range of applications. However, autonomous navigation necessitates the implementation of perception algorithms, often deep neural networks (DNN), that process the input of sensor observations, such as that from cameras and LiDARs, for control logic. The complexity of such algorithms clashes with the severe constraints of these devices in terms of computing power, energy, memory, and execution time. In this paper, we propose NaviSplit, the first instance of a lightweight navigation framework embedding a distributed and dynamic multi-branched neural model. At its core is a DNN split at a compression point, resulting in two model parts: (1) the head model, that is executed at the vehicle, which partially processes and compacts perception from sensors; and (2) the tail model, that is executed at an interconnected compute-capable device, which processes the remainder of the compacted perception and infers navigation commands. Different from prior work, the NaviSplit framework includes a neural gate that dynamically selects a specific head model to minimize channel usage while efficiently supporting the navigation network. In our implementation, the perception model extracts a 2D depth map from a monocular RGB image captured by the drone using the robust simulator Microsoft AirSim. Our results demonstrate that the NaviSplit depth model achieves an extraction accuracy of 72-81% while transmitting an extremely small amount of data (1.2-18 KB) to the edge server. When using the neural gate, as utilized by NaviSplit, we obtain a slightly higher navigation accuracy as compared to a larger static network by 0.3% while significantly reducing the data rate by 95%. To the best of our knowledge, this is the first exemplar of dynamic multi-branched model based on split DNNs for autonomous navigation.
>
---
#### [replaced 011] AI-Driven Marine Robotics: Emerging Trends in Underwater Perception and Ecosystem Monitoring
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 论文探讨AI在海洋机器人中的应用，解决水下环境监测难题。属于计算机视觉与机器人学任务，聚焦提升水下感知与生态监控能力。**

- **链接: [https://arxiv.org/pdf/2509.01878](https://arxiv.org/pdf/2509.01878)**

> **作者:** Scarlett Raine; Tobias Fischer
>
> **备注:** 9 pages, 3 figures, Accepted for Oral Presentation at AAAI Conference on Artificial Intelligence 2026
>
> **摘要:** Marine ecosystems face increasing pressure due to climate change, driving the need for scalable, AI-powered monitoring solutions to inform effective conservation and restoration efforts. This paper examines the rapid emergence of underwater AI as a major research frontier and analyzes the factors that have transformed marine perception from a niche application into a catalyst for AI innovation. We identify three convergent drivers: i) environmental necessity for ecosystem-scale monitoring, ii) democratization of underwater datasets through citizen science platforms, and iii) researcher migration from saturated terrestrial computer vision domains. Our analysis reveals how unique underwater challenges - turbidity, cryptic species detection, expert annotation bottlenecks, and cross-ecosystem generalization - are driving fundamental advances in weakly supervised learning, open-set recognition, and robust perception under degraded conditions. We survey emerging trends in datasets, scene understanding and 3D reconstruction, highlighting the paradigm shift from passive observation toward AI-driven, targeted intervention capabilities. The paper demonstrates how underwater constraints are pushing the boundaries of foundation models, self-supervised learning, and perception, with methodological innovations that extend far beyond marine applications to benefit general computer vision, robotics, and environmental monitoring.
>
---
#### [replaced 012] NaviSlim: Adaptive Context-Aware Navigation and Sensing via Dynamic Slimmable Networks
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出NaviSlim，解决微无人机在资源受限下的导航问题。通过动态调整模型复杂度和传感器功耗，提升计算与能耗效率。**

- **链接: [https://arxiv.org/pdf/2407.01563](https://arxiv.org/pdf/2407.01563)**

> **作者:** Tim Johnsen; Marco Levorato
>
> **备注:** 13 pages, 12 figures
>
> **摘要:** Small-scale autonomous airborne vehicles, such as micro-drones, are expected to be a central component of a broad spectrum of applications ranging from exploration to surveillance and delivery. This class of vehicles is characterized by severe constraints in computing power and energy reservoir, which impairs their ability to support the complex state-of-the-art neural models needed for autonomous operations. The main contribution of this paper is a new class of neural navigation models -- NaviSlim -- capable of adapting the amount of resources spent on computing and sensing in response to the current context (i.e., difficulty of the environment, current trajectory, and navigation goals). Specifically, NaviSlim is designed as a gated slimmable neural network architecture that, different from existing slimmable networks, can dynamically select a slimming factor to autonomously scale model complexity, which consequently optimizes execution time and energy consumption. Moreover, different from existing sensor fusion approaches, NaviSlim can dynamically select power levels of onboard sensors to autonomously reduce power and time spent during sensor acquisition, without the need to switch between different neural networks. By means of extensive training and testing on the robust simulation environment Microsoft AirSim, we evaluate our NaviSlim models on scenarios with varying difficulty and a test set that showed a dynamic reduced model complexity on average between 57-92%, and between 61-80% sensor utilization, as compared to static neural networks designed to match computing and sensing of that required by the most difficult scenario.
>
---
#### [replaced 013] Reflection-Based Task Adaptation for Self-Improving VLA
- **分类: cs.RO**

- **简介: 该论文属于机器人任务适应领域，解决VLA模型在新任务中高效自适应的问题。提出Reflective Self-Adaptation框架，通过双路径机制提升学习效率与任务成功率。**

- **链接: [https://arxiv.org/pdf/2510.12710](https://arxiv.org/pdf/2510.12710)**

> **作者:** Baicheng Li; Dong Wu; Zike Yan; Xinchen Liu; Lusong Li; Zecui Zeng; Hongbin Zha
>
> **摘要:** Pre-trained Vision-Language-Action (VLA) models represent a major leap towards general-purpose robots, yet efficiently adapting them to novel, specific tasks in-situ remains a significant hurdle. While reinforcement learning (RL) is a promising avenue for such adaptation, the process often suffers from low efficiency, hindering rapid task mastery. We introduce Reflective Self-Adaptation, a framework for rapid, autonomous task adaptation without human intervention. Our framework establishes a self-improving loop where the agent learns from its own experience to enhance both strategy and execution. The core of our framework is a dual-pathway architecture that addresses the full adaptation lifecycle. First, a Failure-Driven Reflective RL pathway enables rapid learning by using the VLM's causal reasoning to automatically synthesize a targeted, dense reward function from failure analysis. This provides a focused learning signal that significantly accelerates policy exploration. However, optimizing such proxy rewards introduces a potential risk of "reward hacking," where the agent masters the reward function but fails the actual task. To counteract this, our second pathway, Success-Driven Quality-Guided SFT, grounds the policy in holistic success. It identifies and selectively imitates high-quality successful trajectories, ensuring the agent remains aligned with the ultimate task goal. This pathway is strengthened by a conditional curriculum mechanism to aid initial exploration. We conduct experiments in challenging manipulation tasks. The results demonstrate that our framework achieves faster convergence and higher final success rates compared to representative baselines. Our work presents a robust solution for creating self-improving agents that can efficiently and reliably adapt to new environments.
>
---
#### [replaced 014] UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决无监督下学习缺乏3D结构的潜在动作表示问题。提出UniLACT模型，结合深度信息提升空间先验。**

- **链接: [https://arxiv.org/pdf/2602.20231](https://arxiv.org/pdf/2602.20231)**

> **作者:** Manish Kumar Govind; Dominick Reilly; Pu Wang; Srijan Das
>
> **备注:** this https URL
>
> **摘要:** Latent action representations learned from unlabeled videos have recently emerged as a promising paradigm for pretraining vision-language-action (VLA) models without explicit robot action supervision. However, latent actions derived solely from RGB observations primarily encode appearance-driven dynamics and lack explicit 3D geometric structure, which is essential for precise and contact-rich manipulation. To address this limitation, we introduce UniLACT, a transformer-based VLA model that incorporates geometric structure through depth-aware latent pretraining, enabling downstream policies to inherit stronger spatial priors. To facilitate this process, we propose UniLARN, a unified latent action learning framework based on inverse and forward dynamics objectives that learns a shared embedding space for RGB and depth while explicitly modeling their cross-modal interactions. This formulation produces modality-specific and unified latent action representations that serve as pseudo-labels for the depth-aware pretraining of UniLACT. Extensive experiments in both simulation and real-world settings demonstrate the effectiveness of depth-aware unified latent action representations. UniLACT consistently outperforms RGB-based latent action baselines under in-domain and out-of-domain pretraining regimes, as well as on both seen and unseen manipulation this http URL project page is at this https URL
>
---
#### [replaced 015] "Don't Do That!": Guiding Embodied Systems through Large Language Model-based Constraint Generation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人导航任务，解决将自然语言约束转化为可执行代码的问题。工作是提出STPR框架，利用大语言模型生成Python函数实现约束，提升导航准确性与效率。**

- **链接: [https://arxiv.org/pdf/2506.04500](https://arxiv.org/pdf/2506.04500)**

> **作者:** Amin Seffo; Aladin Djuhera; Masataro Asai; Holger Boche
>
> **备注:** ICLR 2026 Workshop -- Agentic AI in the Wild: From Hallucinations to Reliable Autonomy
>
> **摘要:** Recent advancements in large language models (LLMs) have spurred interest in robotic navigation that incorporates complex spatial, mathematical, and conditional constraints from natural language into the planning problem. Such constraints can be informal yet highly complex, making it challenging to translate into a formal description that can be passed on to a planning algorithm. In this paper, we propose STPR, a constraint generation framework that uses LLMs to translate constraints (expressed as instructions on ``what not to do'') into executable Python functions. STPR leverages the LLM's strong coding capabilities to shift the problem description from language into structured and interpretable code, thus circumventing complex reasoning and avoiding potential hallucinations. We show that these LLM-generated functions accurately describe even complex mathematical constraints, and apply them to point cloud representations with traditional search algorithms. Experiments in a simulated Gazebo environment show that STPR ensures full compliance across several constraints and scenarios, while having short runtimes. We also verify that STPR can be used with smaller code LLMs, making it applicable to a wide range of compact models with low inference cost.
>
---
#### [replaced 016] Force-Aware Residual DAgger via Trajectory Editing for Precision Insertion with Impedance Control
- **分类: cs.RO**

- **简介: 该论文针对高接触精度插入任务，解决模仿学习中的协变量偏移和持续专家监控问题，提出TER-DAgger框架，通过轨迹编辑和力感知机制提升成功率。**

- **链接: [https://arxiv.org/pdf/2603.04038](https://arxiv.org/pdf/2603.04038)**

> **作者:** Yiou Huang; Ning Ma; Weichu Zhao; Zinuo Liu; Jun Sun; Qiufeng Wang; Yaran Chen
>
> **摘要:** Imitation learning (IL) has shown strong potential for contact-rich precision insertion tasks. However, its practical deployment is often hindered by covariate shift and the need for continuous expert monitoring to recover from failures during execution. In this paper, we propose Trajectory Editing Residual Dataset Aggregation (TER-DAgger), a scalable and force-aware human-in-the-loop imitation learning framework that mitigates covariate shift by learning residual policies through optimization-based trajectory editing. This approach smoothly fuses policy rollouts with human corrective trajectories, providing consistent and stable supervision. Second, we introduce a force-aware failure anticipation mechanism that triggers human intervention only when discrepancies arise between predicted and measured end-effector forces, significantly reducing the requirement for continuous expert monitoring. Third, all learned policies are executed within a Cartesian impedance control framework, ensuring compliant and safe behavior during contact-rich interactions. Extensive experiments in both simulation and real-world precision insertion tasks show that TER-DAgger improves the average success rate by over 37\% compared to behavior cloning, human-guided correction, retraining, and fine-tuning baselines, demonstrating its effectiveness in mitigating covariate shift and enabling scalable deployment in contact-rich manipulation.
>
---
#### [replaced 017] Drift-Based Policy Optimization: Native One-Step Policy Learning for Online Robot Control
- **分类: cs.RO**

- **简介: 该论文提出一种基于漂移的策略优化方法，解决机器人控制中多步生成策略计算成本高的问题。通过设计单步生成策略，提升推理速度并保持性能。**

- **链接: [https://arxiv.org/pdf/2604.03540](https://arxiv.org/pdf/2604.03540)**

> **作者:** Yuxuan Gao; Yedong Shen; Shiqi Zhang; Wenhao Yu; Yifan Duan; Jia pan; Jiajia Wu; Jiajun Deng; Yanyong Zhang
>
> **摘要:** Although multi-step generative policies achieve strong performance in robotic manipulation by modeling multimodal action distributions, they require multi-step iterative denoising at inference time. Each action therefore needs tens to hundreds of network function evaluations (NFEs), making them costly for high-frequency closed-loop control and online reinforcement learning (RL). To address this limitation, we propose a two-stage framework for native one-step generative policies that shifts refinement from inference to training. First, we introduce the Drift-Based Policy (DBP), which leverages fixed-point drifting objectives to internalize iterative refinement into the model parameters, yielding a one-step generative backbone by design while preserving multimodal action modeling capacity. Second, we develop Drift-Based Policy Optimization (DBPO), an online RL framework that equips the pretrained backbone with a compatible stochastic interface, enabling stable on-policy updates without sacrificing the one-step deployment property. Extensive experiments demonstrate the effectiveness of the proposed framework across offline imitation learning, online fine-tuning, and real-world control scenarios. DBP matches or exceeds the performance of multi-step diffusion policies while achieving up to $100\times$ faster inference. It also consistently outperforms existing one-step baselines on challenging manipulation benchmarks. Moreover, DBPO enables effective and stable policy improvement in online settings. Experiments on a real-world dual-arm robot demonstrate reliable high-frequency control at 105.2 Hz.
>
---
#### [replaced 018] Deep Learning-Powered Visual SLAM Aimed at Assisting Visually Impaired Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉SLAM任务，旨在解决复杂环境下定位与导航的可靠性问题，通过融合深度学习方法提升性能。**

- **链接: [https://arxiv.org/pdf/2510.20549](https://arxiv.org/pdf/2510.20549)**

> **作者:** Marziyeh Bamdad; Hans-Peter Hutter; Alireza Darvishy
>
> **备注:** 8 pages, 7 figures, 4 tables. Published in the Proceedings of the 20th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP 2025), VISAPP
>
> **摘要:** Despite advancements in SLAM technologies, robust operation under challenging conditions such as low-texture, motion-blur, or challenging lighting remains an open challenge. Such conditions are common in applications such as assistive navigation for the visually impaired. These challenges undermine localization accuracy and tracking stability, reducing navigation reliability and safety. To overcome these limitations, we present SELM-SLAM3, a deep learning-enhanced visual SLAM framework that integrates SuperPoint and LightGlue for robust feature extraction and matching. We evaluated our framework using TUM RGB-D, ICL-NUIM, and TartanAir datasets, which feature diverse and challenging scenarios. SELM-SLAM3 outperforms conventional ORB-SLAM3 by an average of 87.84% and exceeds state-of-the-art RGB-D SLAM systems by 36.77%. Our framework demonstrates enhanced performance under challenging conditions, such as low-texture scenes and fast motion, providing a reliable platform for developing navigation aids for the visually impaired.
>
---
#### [replaced 019] Incorporating Social Awareness into Control of Unknown Multi-Agent Systems: A Real-Time Spatiotemporal Tubes Approach
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体控制任务，解决动态环境中未知系统的行为协调问题。通过引入社会意识，设计实时时空管框架，确保安全、避障和按时到达。**

- **链接: [https://arxiv.org/pdf/2510.25597](https://arxiv.org/pdf/2510.25597)**

> **作者:** Siddhartha Upadhyay; Ratnangshu Das; Pushpak Jagtap
>
> **摘要:** This paper presents a decentralized control framework that incorporates social awareness into multi-agent systems with unknown dynamics to achieve prescribed-time reach-avoid-stay tasks in dynamic environments. Each agent is assigned a social awareness index that quantifies its level of cooperation or self-interest, allowing heterogeneous social behaviors within the system. Building on the spatiotemporal tube (STT) framework, we propose a real-time STT framework that synthesizes tubes online for each agent while capturing its social interactions with others. A closed-form, approximation-free control law is derived to ensure that each agent remains within its evolving STT, thereby avoiding dynamic obstacles while also preventing inter-agent collisions in a socially aware manner, and reaching the target within a prescribed time. The proposed approach provides formal guarantees on safety and timing, and is computationally lightweight, model-free, and robust to unknown disturbances. The effectiveness and scalability of the framework are validated through simulation and hardware experiments on a 2D omnidirectional
>
---
#### [replaced 020] Pseudo-Expert Regularized Offline RL for End-to-End Autonomous Driving in Photorealistic Closed-Loop Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于端到端自动驾驶任务，解决模仿学习的缺陷问题。提出一种仅使用摄像头的离线强化学习框架，通过伪专家轨迹进行行为正则化，提升驾驶安全性和路径完成率。**

- **链接: [https://arxiv.org/pdf/2512.18662](https://arxiv.org/pdf/2512.18662)**

> **作者:** Chihiro Noguchi; Takaki Yamamoto
>
> **备注:** Accepted to CVPR Findings 2026
>
> **摘要:** End-to-end (E2E) autonomous driving models that take only camera images as input and directly predict a future trajectory are appealing for their computational efficiency and potential for improved generalization via unified optimization; however, persistent failure modes remain due to reliance on imitation learning (IL). While online reinforcement learning (RL) could mitigate IL-induced issues, the computational burden of neural rendering-based simulation and large E2E networks renders iterative reward and hyperparameter tuning costly. We introduce a camera-only E2E offline RL framework that performs no additional exploration and trains solely on a fixed simulator dataset. Offline RL offers strong data efficiency and rapid experimental iteration, yet is susceptible to instability from overestimation on out-of-distribution (OOD) actions. To address this, we construct pseudo ground-truth trajectories from expert driving logs and use them as a behavior regularization signal, suppressing imitation of unsafe or suboptimal behavior while stabilizing value learning. Training and closed-loop evaluation are conducted in a neural rendering environment learned from the public nuScenes dataset. Empirically, the proposed method achieves substantial improvements in collision rate and route completion compared with IL baselines. Our code is available at this https URL.
>
---
#### [replaced 021] Horticultural Temporal Fruit Monitoring via 3D Instance Segmentation and Re-Identification using Colored Point Clouds
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于农业机器人领域，解决果树随时间变化的精准监测问题。通过3D点云进行果实实例分割与重识别，提升动态果园中的跟踪精度。**

- **链接: [https://arxiv.org/pdf/2411.07799](https://arxiv.org/pdf/2411.07799)**

> **作者:** Daniel Fusaro; Federico Magistri; Jens Behley; Alberto Pretto; Cyrill Stachniss
>
> **摘要:** Accurate and consistent fruit monitoring over time is a key step toward automated agricultural production systems. However, this task is inherently difficult due to variations in fruit size, shape, occlusion, orientation, and the dynamic nature of orchards where fruits may appear or disappear between observations. In this article, we propose a novel method for fruit instance segmentation and re-identification on 3D terrestrial point clouds collected over time. Our approach directly operates on dense colored point clouds, capturing fine-grained 3D spatial detail. We segment individual fruits using a learning-based instance segmentation method applied directly to the point cloud. For each segmented fruit, we extract a compact and discriminative descriptor using a 3D sparse convolutional neural network. To track fruits across different times, we introduce an attention-based matching network that associates fruits with their counterparts from previous sessions. Matching is performed using a probabilistic assignment scheme, selecting the most likely associations across time. We evaluate our approach on real-world datasets of strawberries and apples, demonstrating that it outperforms existing methods in both instance segmentation and temporal re-identification, enabling robust and precise fruit monitoring across complex and dynamic orchard environments. Keywords = Agricultural Robotics, 3D Fruit Tracking, Instance Segmentation, Deep Learning , Point Clouds, Sparse Convolutional Networks, Temporal Monitoring
>
---
#### [replaced 022] Learning Geometry-Aware Nonprehensile Pushing and Pulling with Dexterous Hands
- **分类: cs.RO**

- **简介: 该论文属于非抓取操作任务，旨在解决机器人手在复杂环境下进行推拉操作的问题。通过生成几何感知的手部姿态，提升操作的稳定性和适应性。**

- **链接: [https://arxiv.org/pdf/2509.18455](https://arxiv.org/pdf/2509.18455)**

> **作者:** Yunshuang Li; Yiyang Ling; Gaurav S. Sukhatme; Daniel Seita
>
> **备注:** Published at International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Nonprehensile manipulation, such as pushing and pulling, enables robots to move, align, or reposition objects that may be difficult to grasp due to their geometry, size, or relationship to the robot or the environment. Much of the existing work in nonprehensile manipulation relies on parallel-jaw grippers or tools such as rods and spatulas. In contrast, multi-fingered dexterous hands offer richer contact modes and versatility for handling diverse objects to provide stable support over the objects, which compensates for the difficulty of modeling the dynamics of nonprehensile manipulation. Therefore, we propose Geometry-aware Dexterous Pushing and Pulling(GD2P) for nonprehensile manipulation with dexterous robotic hands. We study pushing and pulling by framing the problem as synthesizing and learning pre-contact dexterous hand poses that lead to effective manipulation. We generate diverse hand poses via contact-guided sampling, filter them using physics simulation, and train a diffusion model conditioned on object geometry to predict viable poses. At test time, we sample hand poses and use standard motion planners to select and execute pushing and pulling actions. We perform extensive real-world experiments with an Allegro Hand and a LEAP Hand, demonstrating that GD2P offers a scalable route for generating dexterous nonprehensile manipulation motions with its applicability to different hand morphologies. Our project website is available at: this http URL.
>
---
#### [replaced 023] Characterizing the Resilience and Sensitivity of Polyurethane Vision-Based Tactile Sensors
- **分类: cs.RO**

- **简介: 该论文属于传感器性能评估任务，旨在解决硅胶易损、灵敏度高的问题。通过对比聚氨酯与硅胶的耐久性和灵敏度，验证聚氨酯在高负载应用中的优势。**

- **链接: [https://arxiv.org/pdf/2511.07797](https://arxiv.org/pdf/2511.07797)**

> **作者:** Benjamin Davis; Hannah Stuart
>
> **摘要:** Vision-based tactile sensors (VBTSs) are a promising technology for robots, providing them with dense signals that can be translated into a multi-faceted understanding of contact. However, existing VBTS tactile surfaces make use of silicone gels, which provide high sensitivity but easily deteriorate from loading and surface wear. We propose that polyurethane rubber, a typically harder material used for high-load applications like shoe soles, rubber wheels, and industrial gaskets, may provide improved physical gel resilience, potentially at the cost of sensitivity. To compare the resilience and sensitivity of two polyurethane gel formulations against a common silicone baseline, we propose a series of repeatable characterization protocols. Our resilience tests assess sensor durability across normal loading, shear loading, and abrasion. For sensitivity, we introduce learning-free assessments of force and spatial sensitivity to directly measure the physical capabilities of each gel without effects introduced from data and model quality. We also include a bottle cap loosening and tightening demonstration to validate the results of our controlled tests with a real-world example. Our results show that polyurethane yields a more robust sensor. While it sacrifices sensitivity at low forces, the effective force range is largely increased, revealing the utility of polyurethane VBTSs over silicone versions in more rugged, high-load applications.
>
---
